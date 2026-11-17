// The frame area's parallel policy on the shared pool (parallel/pool.hpp):
// when a kernel splits, and into which chunks. The pool only runs tasks; each
// area measures its own threshold (the array area's is
// sd::kParallelThreshold in array/parallel_limits.hpp, 1 << 15).
//
// What a task computes never depends on how many threads there are: the
// chunks depend on the row count alone, and partial results are combined in
// chunk order. So every result is bit-for-bit the same for any thread count
// (docs/design/frame.md §11.5).
#pragma once

#include "parallel/pool.hpp"

#include <algorithm>
#include <cstdint>

namespace sd::frame {

// Below this much work (elements, roughly) everything runs on the calling
// thread, which keeps the GIL. The array area uses 1 << 15 (docs/design/array.md
// §10.4); measured with this area's kernels at 1 and 10 threads
// (docs/design/frame.md §11.2): at 64k elements 10 threads still
// lose on f + g, sum and isna, and from 128k they win on f + g (14 -> 8 us),
// sum (13 -> 5 us) and a filter (43 -> 19 us). The chunks of 16k rows mean
// only a few tasks below that, against a fixed wake-up cost of a few
// microseconds.
inline constexpr std::int64_t kParallelThreshold = 1 << 17;

// Rows per task of for_chunks: a multiple of 64, so two tasks never write
// the same byte (or word) of a validity bitmap.
inline constexpr std::int64_t kChunkRows = 1 << 14;

// Number of for_chunks chunks of n rows (for per-chunk partial results).
inline std::int64_t num_chunks(std::int64_t n) { return (n + kChunkRows - 1) / kChunkRows; }

// parallel_for for work at or above kParallelThreshold, with the GIL released
// whatever the thread count and the task count (docs/api/frame.md §12):
// parallel_for releases it from two tasks on, and a single task is the
// caller's to cover (parallel/pool.hpp). Inside a task, or under an Unlocked,
// the GIL is not held and this is parallel_for alone. Tasks must not touch
// Python objects.
template <class F>
void run_unlocked(std::int64_t ntasks, F&& fn) {
    const parallel::Unlocked unlocked(true);
    parallel::parallel_for(ntasks, fn);
}

// f(lo, hi) for every chunk [lo, hi) of [0, n): kChunkRows rows each, the
// last one shorter. The chunks depend on n alone, so a kernel that combines
// per-chunk results in chunk order gets the same bits for any thread count
// (docs/design/frame.md §11.3). Below kParallelThreshold the chunks run in order on the
// calling thread, which keeps the GIL; at or above it the GIL is released,
// with one thread too (run_unlocked). `cost` scales n for kernels that do
// several times the work of one add per row; it moves the threshold, never
// the chunks.
template <class F>
void for_chunks(std::int64_t n, F&& f, std::int64_t cost = 1) {
    if (n <= 0) return;
    const std::int64_t ntasks = num_chunks(n);
    auto task = [&](std::int64_t t) { f(t * kChunkRows, std::min(n, (t + 1) * kChunkRows)); };
    if (n * cost < kParallelThreshold) {
        for (std::int64_t t = 0; t < ntasks; ++t) task(t);
        return;
    }
    run_unlocked(ntasks, task);
}

// Releases the GIL for a serial stretch of `work` (elements, roughly) at or
// above kParallelThreshold: other Python threads run meanwhile even with one
// thread (docs/api/frame.md §12). Below the threshold it does nothing. The
// shared parallel::Unlocked with this area's threshold.
class Unlocked : public parallel::Unlocked {
public:
    explicit Unlocked(std::int64_t work) : parallel::Unlocked(work >= kParallelThreshold) {}
};

}  // namespace sd::frame
