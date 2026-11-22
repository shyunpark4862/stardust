#include "parallel/pool.hpp"

#include <Python.h>
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>

#include <algorithm>
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

// A job of more than this many tasks, its task length unknown, wakes sleeping
// workers at once; a smaller one only once the caller has timed its own first
// tasks (Pool::run).
constexpr std::int64_t kWakeTasks = 16;
// A job of at most this many tasks, their length unknown, wakes a worker per
// task beyond the caller's at once: after its first task the caller would take
// the last one itself, so timing it cannot help, and a caller splits work this
// coarsely only when the tasks are long (2 tasks of 20 us: 27 us against 40;
// 2 of 0.5 us: 1.7 against 1.1).
constexpr std::int64_t kEagerTasks = 3;

// The sleeping workers the caller signals itself (Pool::wake).
constexpr std::int64_t kDirectWakes = 3;

// The generic timer of the core: a register read, a few ns (steady_clock
// takes ~20 ns, too much to read after every task of a microsecond).
inline std::uint64_t ticks() { return __builtin_arm_rsr64("cntvct_el0"); }
inline std::uint64_t ticks_per_second() { return __builtin_arm_rsr64("cntfrq_el0"); }

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

    void run(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn, std::int64_t task_ns) {
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
            error_ = nullptr;
            // A new generation, open, nobody in it yet (release: the job's
            // fields above are visible to whoever sees it).
            state_.store((generation(state_.load()) + 1) << kGenShift, std::memory_order_release);
        }
        // Spinning workers see the new generation by themselves. Sleeping
        // ones are woken only for a job long enough to repay it: a wake-up
        // takes ~6 us to arrive (~5 more for relayed ones) and ~0.7 us of the
        // caller's time per signal, so the caller alone finishes a short job
        // before a woken worker would start (busy-wait tasks after a 1 ms pause, median: 8 x 1 us 14 us
        // woken, 8 alone; 8 x 3 us 17.5 against 24; 32 x 1 us 20 against 32).
        // How long the job is: the caller's estimate when it gives one; else
        // many tasks count as long, 2-3 tasks too (kEagerTasks), and for the
        // rest the caller times its own first tasks (work(true)): 16 tasks of
        // 20 us, 91 us against 320 alone. Which thread runs a task never
        // changes the result (§10.5).
        bool probe = false;
        if (task_ns > 0) {
            if (ntasks * task_ns >= kWakeWorkNs) wake(ntasks - 1);
        } else if (ntasks > kWakeTasks || ntasks <= kEagerTasks) {
            wake(ntasks - 1);
        } else {
            probe = true;
        }
        work(probe);  // the calling thread takes tasks too
        // The tasks are all taken: close the job, then wait only for the
        // workers that joined it (they may still be running a task). A
        // worker that wakes later sees the job closed and leaves it alone,
        // so a late wake-up never holds up the call
        // (docs/design/array.md §10.1). Spin first (they are usually a few
        // us away), then sleep.
        state_.fetch_or(kClosed, std::memory_order_acq_rel);
        auto done = [&] { return (state_.load(std::memory_order_acquire) & kCount) == 0; };
        spin_until(done);
        if (!done()) {
            std::unique_lock lock(m_);
            cv_done_.wait(lock, done);
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

    // Wakes up to k sleeping workers: kDirectWakes of them here, and the
    // first of those passes the rest on (relay_), so the caller pays for at
    // most kDirectWakes signals. A broadcast to 9 sleepers took the caller
    // ~10 us (8 tasks of 0.5 us after a 1 ms pause: 14 us woken, 4 alone; 3
    // signals and a relay: 10), while a relayed worker arrives only ~5 us
    // after the first (4 tasks of 20 us: 33 us with 1 direct signal, 27 with
    // 3). The job was published under m_, so a worker that goes to sleep later
    // sees it and does not wait; one already asleep is counted in sleepers_.
    void wake(std::int64_t k) {
        const int asleep = sleepers_.load(std::memory_order_relaxed);
        if (asleep == 0 || k <= 0) return;
        k = std::min<std::int64_t>(k, asleep);
        const std::int64_t direct = std::min(k, kDirectWakes);
        if (k > direct) relay_.store(k - direct, std::memory_order_relaxed);
        for (std::int64_t i = 0; i < direct; ++i) cv_work_.notify_one();
    }

    // k more workers, from a woken worker passing a wake-up on.
    void signal(std::int64_t k) {
        if (k >= sleepers_.load(std::memory_order_relaxed)) {
            cv_work_.notify_all();
        } else {
            for (std::int64_t i = 0; i < k; ++i) cv_work_.notify_one();
        }
    }

    // Takes tasks until none are left. `probe` (the caller, in a job of few
    // tasks of unknown length): after each task, estimate the work left from
    // the time per task so far, and wake sleeping workers once it reaches
    // kWakeWorkNs with two tasks or more still untaken.
    void work(bool probe = false) {
        t_in_task = true;
        const std::uint64_t start = probe ? ticks() : 0;
        std::int64_t mine = 0;
        for (std::int64_t t; (t = next_.fetch_add(1)) < ntasks_;) {
            try {
                (*job_)(t);
            } catch (...) {
                std::lock_guard lock(m_);
                if (!error_) error_ = std::current_exception();
            }
            if (probe) {
                ++mine;
                const std::int64_t left = ntasks_ - next_.load(std::memory_order_relaxed);
                if (left < 2) {
                    probe = false;  // the caller takes the last one itself
                } else if ((ticks() - start) * static_cast<std::uint64_t>(left) >=
                           wake_ticks_ * static_cast<std::uint64_t>(mine)) {
                    wake(left);
                    probe = false;
                }
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
            auto fresh = [&] { return generation(state_.load(std::memory_order_acquire)) != seen; };
            spin_until([&] { return fresh() || stop_flag_; });
            if (!fresh() && !stop_flag_) {
                // Nothing came while spinning: sleep. (Only here is the mutex
                // taken: nine workers all grabbing it the moment a job appears
                // was the remaining ~18 us of a small parallel call,
                // docs/design/array.md §10.1.)
                std::unique_lock lock(m_);
                sleepers_.fetch_add(1, std::memory_order_relaxed);
                cv_work_.wait(lock, [&] { return stop_ || fresh(); });
                sleepers_.fetch_sub(1, std::memory_order_relaxed);
                lock.unlock();
                // Pass the caller's wake-up on (Pool::wake), unless the job is over.
                if (const std::int64_t r = relay_.exchange(0, std::memory_order_relaxed);
                    r > 0 && !(state_.load(std::memory_order_acquire) & kClosed))
                    signal(r);
            }
            if (stop_flag_) return;
            // Join the job seen, only while it is open: one compare-and-swap
            // on the generation, the closed bit and the count together, so a
            // worker can never count itself into a job that already closed or
            // into a later one. The job's fields were written before the
            // generation was stored (release); the acquire here makes them
            // visible.
            std::uint64_t s = state_.load(std::memory_order_acquire);
            seen = generation(s);
            bool joined = false;
            while (!joined && !(s & kClosed) && generation(s) == seen)
                joined = state_.compare_exchange_weak(s, s + 1, std::memory_order_acq_rel);
            if (!joined) continue;  // too late: the caller has finished without us
            work();
            const std::uint64_t left = state_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if ((left & kClosed) && (left & kCount) == 0) {  // the caller is waiting for the last one
                std::lock_guard lock(m_);
                cv_done_.notify_one();
            }
        }
    }

    void start(int n) {
        stop_ = false;
        stop_flag_ = false;
        const std::uint64_t now = generation(state_.load());
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
    // next_ and state_ on cache lines of their own (128 bytes on M4): the
    // participants' fetch_add on next_ no longer invalidates the line the idle
    // spinners poll, nor the other way round.
    alignas(128) std::atomic<std::int64_t> next_{0};
    // The job's state in one word: the generation (bits 32-63), whether the
    // job is closed (bit 31: every task taken, the caller waiting) and how
    // many workers joined it and have not left yet (bits 0-30).
    static constexpr int kGenShift = 32;
    static constexpr std::uint64_t kClosed = std::uint64_t{1} << 31;
    static constexpr std::uint64_t kCount = kClosed - 1;
    static std::uint64_t generation(std::uint64_t s) { return s >> kGenShift; }
    alignas(128) std::atomic<std::uint64_t> state_{0};
    alignas(128) bool stop_ = false;
    std::atomic<bool> stop_flag_{false};  // stop_, readable while spinning
    std::exception_ptr error_;
    std::atomic<bool> busy_{false};
    std::atomic<int> nthreads_{1};
    std::atomic<int> sleepers_{0};  // workers waiting on cv_work_ (changed under m_)
    std::atomic<std::int64_t> relay_{0};  // workers the next woken worker is to wake (Pool::wake)
    // kWakeWorkNs in ticks() units
    const std::uint64_t wake_ticks_ = ticks_per_second() * static_cast<std::uint64_t>(kWakeWorkNs) / 1'000'000'000u;
};

// The pool, made on first use. Never destroyed: joining threads during
// interpreter shutdown is fragile, and the OS reclaims them at exit anyway.
std::atomic<Pool*> g_pool{nullptr};
std::atomic<int> g_threads{0};  // the thread count a pool made later starts with (0: the default)

// After fork() only the forking thread exists in the child: the workers are
// gone, and waiting for them would hang the first parallel call. The child
// forgets the old pool without touching it (no join, no destructor: its
// threads and its mutex state are the parent's) and makes a new one, with
// the same thread count, on first use (OpenBLAS and OpenMP runtimes do the
// same).
void after_fork_in_child() {
    if (Pool* old = g_pool.load(std::memory_order_relaxed)) {
        g_threads.store(old->threads(), std::memory_order_relaxed);
        g_pool.store(nullptr, std::memory_order_relaxed);  // leaked on purpose
    }
    t_in_task = false;
}

Pool& pool() {
    Pool* p = g_pool.load(std::memory_order_acquire);
    if (p) [[likely]]
        return *p;
    static const bool registered = [] { return pthread_atfork(nullptr, nullptr, &after_fork_in_child) == 0; }();
    (void)registered;
    const int n = g_threads.load(std::memory_order_relaxed);
    Pool* made = new Pool(n > 0 ? n : performance_cores());
    if (g_pool.compare_exchange_strong(p, made, std::memory_order_acq_rel)) return *made;
    delete made;  // another thread made one first
    return *p;
}

}  // namespace

int num_threads() { return pool().threads(); }

void set_num_threads(int n) { pool().resize(n); }

bool available() { return !t_in_task && pool().threads() > 1 && !pool().busy(); }


void parallel_for(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn) { parallel_for(ntasks, fn, 0); }

void parallel_for(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn, std::int64_t task_ns) {
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
    pool().run(ntasks, fn, task_ns);  // spread on the pool (or serial if another thread's job holds it)
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
