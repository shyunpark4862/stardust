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
// in task order. So every result is bit-for-bit the same for any thread count
// (docs/design/array.md §10.5).
#pragma once

#include <atomic>
#include <climits>
#include <cstdint>
#include <functional>

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
// the calling thread alone). The GIL is released for the duration, whatever
// the thread count, so other Python threads keep running; the caller decides
// that the work is worth it. An exception from a task is rethrown here. Tasks
// must not touch Python objects.
void parallel_for(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn);

// Releases the GIL (if this thread holds it) for the lifetime of the object.
// parallel_for uses it; work that runs on the calling thread alone keeps the
// GIL.
class GilRelease {
public:
    GilRelease();
    ~GilRelease();
    GilRelease(const GilRelease&) = delete;
    GilRelease& operator=(const GilRelease&) = delete;

private:
    void* state_ = nullptr;
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
        while (row < cur && !row_.compare_exchange_weak(cur, row, std::memory_order_relaxed)) {
        }
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
