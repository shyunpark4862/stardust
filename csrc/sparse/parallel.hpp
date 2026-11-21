// When the sparse kernels split work across the thread pool (docs/design/sparse.md §2.6).
//
// Tasks are cut by size alone (entries, rows), never by the thread count, and every kernel here
// either writes disjoint parts of its output or combines per-task results in task order. So
// every result is the same bit for bit for any thread count (API.md §3).
#pragma once

#include "parallel/pool.hpp"

#include <algorithm>
#include <cstdint>

namespace sd::sparse {

// The array area's rule (docs/design/array.md §10.4): from 2^15 elements of work the call splits
// and runs without the GIL, whatever the thread count; below, it runs on the calling thread
// holding the GIL.
inline constexpr std::int64_t kParallelThreshold = 1 << 15;

// Entries (or rows) per task of for_chunks.
inline constexpr std::int64_t kChunk = 1 << 14;

inline std::int64_t num_chunks(std::int64_t n, std::int64_t chunk = kChunk) {
    return n <= 0 ? 0 : (n + chunk - 1) / chunk;
}

// f(t, lo, hi) for every chunk t = [lo, hi) of [0, n), `chunk` items each (the last shorter).
// `work` decides the split and the GIL (default n). Tasks must not touch Python objects.
template <class F>
void for_chunks(std::int64_t n, F&& f, std::int64_t chunk = kChunk, std::int64_t work = -1) {
    const std::int64_t nt = num_chunks(n, chunk);
    if (nt == 0) return;
    if (work < 0) work = n;
    const parallel::Unlocked unlocked(work >= kParallelThreshold);
    auto task = [&](std::int64_t t) { f(t, t * chunk, std::min(n, (t + 1) * chunk)); };
    if (work < kParallelThreshold || nt == 1) {
        for (std::int64_t t = 0; t < nt; ++t) task(t);
    } else {
        parallel::parallel_for(nt, task);
    }
}

// Runs fn(t) for t < ntasks: in parallel from kParallelThreshold units of `work`, without the GIL
// from there whatever the thread count.
template <class F>
void for_tasks(std::int64_t ntasks, std::int64_t work, F&& fn) {
    if (ntasks <= 0) return;
    const parallel::Unlocked unlocked(work >= kParallelThreshold);
    if (work < kParallelThreshold || ntasks == 1) {
        for (std::int64_t t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

}  // namespace sd::sparse
