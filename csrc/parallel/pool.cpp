#include "parallel/pool.hpp"

#include <Python.h>
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace sd::parallel {

namespace {

thread_local bool t_in_task = false;

// Spin-wait for up to kSpin before sleeping. Measured by the clock, not by
// iterations: the ARM `yield` hint takes a cycle or two, so a count that
// looks large is over in a few microseconds (docs/design/array.md §10.1).
constexpr auto kSpin = std::chrono::microseconds(100);

template <typename Done>
void spin_until(Done done) {
    const auto until = std::chrono::steady_clock::now() + kSpin;
    for (int i = 0; !done(); ++i) {
        __builtin_arm_yield();
        if ((i & 63) == 63 && std::chrono::steady_clock::now() > until) return;
    }
}

// The performance cores. Not for waiting on slow E-cores -- tasks are handed out
// dynamically, so an E-core just takes fewer -- but because more threads slowed
// the memory-bound operations by 3-12% (14 threads; docs/design/array.md §10.3).
int performance_cores() {
    int n = 0;
    size_t len = sizeof n;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr, 0) != 0 || n < 1)
        n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 1;
}

class Pool {
public:
    explicit Pool(int n) { start(n); }
    ~Pool() { stop(); }

    // The caller counts. An atomic copy of workers_.size() + 1: any thread
    // may ask while set_num_threads rebuilds workers_ (reading the vector
    // itself raced with that; docs/design/array.md §10.3).
    // Whether a job holds the pool now (another Python thread's, with its GIL
    // released). A call that sees this runs serially (available()).
    bool busy() const { return busy_.load(std::memory_order_relaxed); }

    int threads() const { return nthreads_.load(std::memory_order_relaxed); }

    void resize(int n) {
        while (busy_.exchange(true)) std::this_thread::yield();  // wait for a running job to finish
        stop();
        start(n);
        busy_ = false;
    }

    void run(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn) {
        // Another Python thread's job is running (it holds the pool while its
        // GIL is released): do this one serially rather than wait or
        // oversubscribe (docs/design/array.md §10.1). workers_ is read only once the
        // pool is ours: resize() changes it holding busy_ too.
        if (busy_.exchange(true)) {
            serial(ntasks, fn);
            return;
        }
        if (workers_.empty()) {  // one thread
            busy_ = false;
            serial(ntasks, fn);
            return;
        }
        {
            std::lock_guard lock(m_);
            job_ = &fn;
            ntasks_ = ntasks;
            next_ = 0;
            pending_ = static_cast<int>(workers_.size());
            error_ = nullptr;
            generation_.fetch_add(1, std::memory_order_release);
        }
        cv_work_.notify_all();  // for workers that went to sleep
        work();                 // the calling thread takes tasks too
        // Wait for the workers: spin first (they are usually a few us away),
        // then sleep.
        spin_until([&] { return pending_.load(std::memory_order_acquire) == 0; });
        if (pending_.load(std::memory_order_acquire) != 0) {
            std::unique_lock lock(m_);
            cv_done_.wait(lock, [&] { return pending_.load() == 0; });
        }
        job_ = nullptr;
        std::exception_ptr error = error_;
        busy_ = false;
        if (error) std::rethrow_exception(error);
    }

private:
    static void serial(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn) {
        t_in_task = true;
        try {
            for (std::int64_t t = 0; t < ntasks; ++t) fn(t);
        } catch (...) {
            t_in_task = false;
            throw;
        }
        t_in_task = false;
    }

    void work() {
        t_in_task = true;
        for (std::int64_t t; (t = next_.fetch_add(1)) < ntasks_;) {
            try {
                (*job_)(t);
            } catch (...) {
                std::lock_guard lock(m_);
                if (!error_) error_ = std::current_exception();
            }
        }
        t_in_task = false;
    }

    // `seen` is the job generation current when the worker is created: a
    // worker started by set_num_threads must not mistake the last finished
    // job for a new one (it would report it done twice, and a later job
    // would wait forever for a report that never comes -- a hang the
    // resize-under-load test found).
    void worker_main(std::uint64_t seen) {
        // Ask macOS for the highest QoS class, which keeps the thread on the
        // performance cores whenever they are free.
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        for (;;) {
            // Spin a little before sleeping: elementwise calls tend to come in
            // runs, and waking a sleeping thread costs tens of microseconds on
            // macOS (~25 us per job, docs/design/array.md §10.1), more than a whole
            // 1M-element add.
            spin_until([&] { return generation_.load(std::memory_order_acquire) != seen || stop_flag_; });
            if (generation_.load(std::memory_order_acquire) == seen && !stop_flag_) {
                // Nothing came while spinning: sleep. (Only here is the mutex
                // taken: nine workers all grabbing it the moment a job appears
                // was the remaining ~18 us of a small parallel call,
                // docs/design/array.md §10.1.)
                std::unique_lock lock(m_);
                cv_work_.wait(lock, [&] { return stop_ || generation_.load() != seen; });
            }
            if (stop_flag_) return;
            // The job's fields were written before generation_ was bumped
            // (release); this acquire load makes them visible here.
            seen = generation_.load(std::memory_order_acquire);
            work();
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard lock(m_);
                cv_done_.notify_one();
            }
        }
    }

    void start(int n) {
        stop_ = false;
        stop_flag_ = false;
        const std::uint64_t now = generation_.load();
        for (int i = 1; i < n; ++i) workers_.emplace_back([this, now] { worker_main(now); });
        nthreads_.store(static_cast<int>(workers_.size()) + 1, std::memory_order_relaxed);
    }

    void stop() {
        nthreads_.store(1, std::memory_order_relaxed);
        {
            std::lock_guard lock(m_);
            stop_ = true;
            stop_flag_ = true;
        }
        cv_work_.notify_all();
        for (std::thread& t : workers_) t.join();
        workers_.clear();
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_work_, cv_done_;
    const std::function<void(std::int64_t)>* job_ = nullptr;
    std::int64_t ntasks_ = 0;
    std::atomic<std::int64_t> next_{0};
    std::atomic<int> pending_{0};
    std::atomic<std::uint64_t> generation_{0};
    bool stop_ = false;
    std::atomic<bool> stop_flag_{false};  // stop_, readable while spinning
    std::exception_ptr error_;
    std::atomic<bool> busy_{false};
    std::atomic<int> nthreads_{1};
};

Pool& pool() {
    // Never destroyed: joining threads during interpreter shutdown is fragile,
    // and the OS reclaims them at exit anyway.
    static Pool* p = new Pool(performance_cores());
    return *p;
}

}  // namespace

int num_threads() { return pool().threads(); }

void set_num_threads(int n) { pool().resize(n); }

bool available() { return !t_in_task && pool().threads() > 1 && !pool().busy(); }


void parallel_for(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn) {
    if (ntasks <= 0) return;
    // One task or a nested call: a plain loop on this thread, the GIL left as
    // it is (docs/design/array.md §10.2 -- one task is the caller's own size
    // decision, and releasing and retaking the GIL would cost a small call
    // more than it saves; a nested call holds no GIL anyway).
    if (ntasks == 1 || t_in_task) {
        for (std::int64_t t = 0; t < ntasks; ++t) fn(t);
        return;
    }
    GilRelease release;
    if (pool().threads() <= 1) {  // one thread: the tasks in order, without the GIL
        for (std::int64_t t = 0; t < ntasks; ++t) fn(t);
        return;
    }
    pool().run(ntasks, fn);  // spread on the pool (or serial if another thread's job holds it)
}

GilRelease::GilRelease() {
    // A task holds no GIL: the workers never had it, and the calling thread
    // released it before it took its first task.
    if (!t_in_task && PyGILState_Check()) state_ = PyEval_SaveThread();
}

GilRelease::~GilRelease() {
    if (state_) PyEval_RestoreThread(static_cast<PyThreadState*>(state_));
}

}  // namespace sd::parallel
