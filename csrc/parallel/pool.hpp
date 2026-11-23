// The thread pool (docs/design/array.md §10).
//
// Resident workers sleep on a condition variable. A parallel job is a number
// of tasks; every participant -- the workers and the calling thread itself --
// takes the next task index from one atomic counter until none are left, so
// fast threads simply take more tasks (no fixed assignment of work to
// threads).
//
// What a task computes never depends on how many threads there are: the
// callers cut work into tasks by size alone, and combine any partial results
// in task order. So a result has the same bits for any thread count today
// (docs/design/array.md §10.5). The promise is narrower: the same machine,
// thread count and input give the same bits on every run (API.md §3).
#pragma once

#include <atomic>
#include <climits>
#include <cstdint>
#include <functional>
#include <optional>

namespace sd::parallel {

// When to split is each area's decision, measured on its own kernels: the
// array area's kParallelThreshold / kParallelTask (array/parallel_limits.hpp),
// linalg's cost model (linalg/cost.hpp). The pool only runs the tasks.

int num_threads();
void set_num_threads(int n);  // n >= 1; 1 means: never use the pool

// True when a parallel_for would actually run in parallel: more than one
// thread, not already inside a task (no nesting), and the pool not busy with
// another Python thread's job. A call that sees a busy pool would run every
// parallel_for serially, so algorithms that pick a path by this take their
// serial one: the same result, faster than the split one run serially.
bool available();

// Runs fn(t) for t in [0, ntasks). In parallel when available() and the
// pool is not busy with another Python thread's job (then this call runs on
// the calling thread alone). With two tasks or more the GIL is released for
// the duration, whatever the thread count -- with one thread the tasks run in
// order on the calling thread, still without the GIL -- so other Python
// threads keep running; the caller decides that the work is worth splitting.
// One task is a plain call that leaves the GIL as it is; so does a call from
// inside a task, which holds none (docs/design/array.md §10.2). An exception
// from a task is rethrown here. Tasks must not touch Python objects.
//
// Workers that have been idle for ~100 us sleep, and waking them is worth it
// only for a long enough job: a wake-up costs the caller ~0.7 us per worker
// (at most 3; those pass it on to the rest) and arrives ~6 us later, so 10
// threads win from about 20 us of work (kWakeWorkNs). Without an estimate of the task length
// (the overload below), a job of more than 16 tasks wakes them at once, one
// of 2 or 3 tasks wakes a worker per further task at once (too few tasks to
// time), and one of 4-16 tasks wakes them as soon as the calling thread,
// timing its own first tasks, sees that the rest is worth it. So a few long
// tasks (folds, solves, frame chunks) no longer run on the caller alone, and
// a short job of 4-16 tasks never pays for a wake-up.
void parallel_for(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn);

// The same, for a caller that knows roughly how long one task takes
// (nanoseconds on one core; 0 or less: unknown, as above): ntasks x task_ns
// of kWakeWorkNs or more wakes min(ntasks - 1, sleeping) workers at once,
// less never wakes anyone (workers still spinning from the last job join
// either way). Few long tasks then get every worker from the start instead of
// after the caller's first task (4 tasks of 100 us: 107 us against 208), and a
// short job of many tasks no longer pays for a wake-up. Only the speed
// depends on it.
void parallel_for(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn, std::int64_t task_ns);

// The work (nanoseconds, all tasks together) from which a job wakes sleeping
// workers (see parallel_for).
inline constexpr std::int64_t kWakeWorkNs = 20'000;

// Releases the GIL for the lifetime of the object, if this thread holds it:
// inside a task, or under another GilRelease, it does nothing, so releases
// may nest.
class GilRelease {
public:
    GilRelease();
    ~GilRelease();
    GilRelease(const GilRelease&) = delete;
    GilRelease& operator=(const GilRelease&) = delete;

private:
    void* state_ = nullptr;
};

// The GIL rule every area follows (API.md §3): work at or above the area's
// parallel threshold runs without the GIL whatever the thread count; work
// below it keeps the GIL, so a small call pays nothing. parallel_for covers
// the calls that split. A call that is large enough but runs on the calling
// thread -- one thread, a busy pool, nothing to split along, or a serial
// algorithm chosen because no pool is available -- holds an Unlocked for
// that stretch: `Unlocked u(n >= kParallelThreshold);`. Like a task, the
// stretch must not touch Python objects (this includes copying or destroying
// an NDArray, whose `base` is a Python reference).
class Unlocked {
public:
    explicit Unlocked(bool release) {
        if (release) release_.emplace();
    }

private:
    std::optional<GilRelease> release_;
};

// The first index (a row, an element) where a check failed, found by tasks
// running in any order: each task offers its first failing index and the
// smallest wins. So an error names the same index for any thread count, and
// the caller builds the message after the parallel part, holding the GIL
// again (tasks must not touch Python objects).
class FirstRow {
public:
    void offer(std::int64_t row) {
        std::int64_t cur = row_.load(std::memory_order_relaxed);
        while (row < cur && !row_.compare_exchange_weak(cur, row, std::memory_order_relaxed)) {}
    }
    // True when an index before `row` already failed: a task past it can stop.
    bool failed_before(std::int64_t row) const { return row_.load(std::memory_order_relaxed) < row; }
    bool any() const { return row_.load() != kNone; }
    std::int64_t row() const { return row_.load(); }

private:
    static constexpr std::int64_t kNone = INT64_MAX;
    std::atomic<std::int64_t> row_{kNone};
};

}  // namespace sd::parallel
