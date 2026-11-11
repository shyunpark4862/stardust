// The thread pool, copied from stardust-numpy (its M8) as DESIGN.md §1.2 says.
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
// (DESIGN.md §2.6).
#pragma once

#include <climits>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <functional>
#include <optional>

namespace sd::parallel {

// Below this much work (elements, roughly) everything runs on the calling
// thread, which keeps the GIL. One constant for all (DESIGN.md §2.5).
// stardust-numpy used 1 << 15; M11 measured this project's kernels with 1
// and 10 threads: at 64k elements 10 threads still lose on f + g, sum and
// isna, and from 128k they win on f + g (14 -> 8 us), sum (13 -> 5 us) and
// a filter (43 -> 19 us). The chunks of 16k rows mean only a few tasks
// below that, against a fixed wake-up cost of a few microseconds.
inline constexpr std::int64_t kThreshold = 1 << 17;

// How much work one task gets. With the pool's fixed cost per job at ~1.5 us,
// small tasks balance load well.
inline constexpr std::int64_t kTaskWork = 1 << 12;

int num_threads();
void set_num_threads(int n);  // n >= 1; 1 means: never use the pool

// True when a parallel_for would actually run in parallel: more than one
// thread, and not already inside a task (no nesting).
bool available();

// Runs fn(t) for t in [0, ntasks). In parallel when available() and the
// pool is not busy with another Python thread's job (then this call runs on
// the calling thread alone). The GIL is released for the duration, whatever
// the thread count (API.md §20), so other Python threads keep running; the
// caller decides that the work is worth it (see for_chunks and Unlocked). An
// exception from a task is rethrown here. Tasks must not touch Python objects.
void parallel_for(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn);

// Releases the GIL (if this thread holds it) for the lifetime of the object.
class GilRelease {
public:
    GilRelease();
    ~GilRelease();
    GilRelease(const GilRelease&) = delete;
    GilRelease& operator=(const GilRelease&) = delete;

private:
    void* state_ = nullptr;
};

// Rows per task of for_chunks: a multiple of 64, so two tasks never write
// the same byte (or word) of a validity bitmap.
inline constexpr std::int64_t kChunkRows = 1 << 14;

// Number of for_chunks chunks of n rows (for per-chunk partial results).
inline std::int64_t num_chunks(std::int64_t n) { return (n + kChunkRows - 1) / kChunkRows; }

// f(lo, hi) for every chunk [lo, hi) of [0, n): kChunkRows rows each, the
// last one shorter. The chunks depend on n alone, so a kernel that combines
// per-chunk results in chunk order gets the same bits for any thread count
// (DESIGN.md §2.6). Below kThreshold the chunks run in order on the calling
// thread, which keeps the GIL. `cost` scales n for kernels that do several
// times the work of one add per row; it moves the threshold, never the chunks.
template <class F>
void for_chunks(std::int64_t n, F&& f, std::int64_t cost = 1) {
    if (n <= 0) return;
    const std::int64_t ntasks = num_chunks(n);
    auto task = [&](std::int64_t t) { f(t * kChunkRows, std::min(n, (t + 1) * kChunkRows)); };
    if (n * cost < kThreshold) {
        for (std::int64_t t = 0; t < ntasks; ++t) task(t);
        return;
    }
    parallel_for(ntasks, task);
}

// The first row, in row order, where a check failed, found by chunks running
// in any order: each chunk offers its first failing row and the smallest
// wins. So an error names the same row for any thread count (DESIGN.md §2.6),
// and the caller builds the message after the parallel part, holding the
// GIL again (tasks must not touch Python objects).
class FirstRow {
public:
    void offer(std::int64_t row) {
        std::int64_t cur = row_.load(std::memory_order_relaxed);
        while (row < cur && !row_.compare_exchange_weak(cur, row, std::memory_order_relaxed)) {
        }
    }
    // True when a row before `row` already failed: a chunk past it can stop.
    bool failed_before(std::int64_t row) const { return row_.load(std::memory_order_relaxed) < row; }
    bool any() const { return row_.load() != kNone; }
    std::int64_t row() const { return row_.load(); }

private:
    static constexpr std::int64_t kNone = INT64_MAX;
    std::atomic<std::int64_t> row_{kNone};
};

// Releases the GIL for a serial stretch of `work` (elements, roughly) at or
// above kThreshold: other Python threads run meanwhile even with one thread
// (API.md §20). Below the threshold it does nothing.
class Unlocked {
public:
    explicit Unlocked(std::int64_t work) {
        if (work >= kThreshold) release_.emplace();
    }

private:
    std::optional<GilRelease> release_;
};

}  // namespace sd::parallel
