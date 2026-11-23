// When the array area splits work across the thread pool (docs/design/array.md §10.4).
// Each area measures its own: linalg has a cost model (linalg/cost.hpp); the
// pool itself (parallel/pool.hpp) only runs tasks.
#pragma once

#include <cstdint>

namespace sd {

// Below this much work everything runs on the calling thread. The work of an
// elementwise call is its elements times the operation's weight in float64
// adds (ops/ufunc.hpp, op_weights), so exp starts at ~5.5K elements and f + g at
// ~32K; the full float reductions have their own constant, kReduceThreshold.
inline constexpr std::int64_t kParallelThreshold = 1 << 15;

// How much work one task gets. The pool's fixed cost per job is a few us at
// most (spin and lock-free pickup, docs/design/array.md §10.1), so small tasks
// balance load well.
inline constexpr std::int64_t kParallelTask = 1 << 12;

// The same for the full reductions that stream at ~0.07 ns per element (sum,
// mean, max, min, var's sum of squares): 1 vs 10 threads at 2^15 elements,
// sum 2.35 / 1.95 us, mean 2.34 / 3.07, max 2.35 / 2.66, var 5.35 / 6.17;
// at 2^16 all four win (1.2-1.4x).
inline constexpr std::int64_t kReduceThreshold = 1 << 16;

// The time of `work` elements' worth (a float64 add each: 0.13 ns in cache,
// ~0.16 out of it on one core), in ns: the task-length estimate the array's
// jobs give parallel_for, which wakes sleeping workers only from ~20 us of
// work (parallel::kWakeWorkNs; 125K adds, 16K exps).
inline constexpr std::int64_t work_ns(std::int64_t work) { return work / 6 + 1; }

}  // namespace sd
