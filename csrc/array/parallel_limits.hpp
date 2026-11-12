// When the array area splits work across the thread pool (docs/design/array.md §10.4).
// Each area measures its own: linalg has a cost model (linalg/cost.hpp); the
// pool itself (parallel/pool.hpp) only runs tasks.
#pragma once

#include <cstdint>

namespace sd {

// Below this much work (elements, roughly) everything runs on the calling
// thread. Measured where 10 threads start to win: f + g at ~16K elements, exp
// at ~8K, sum at ~32K; one constant for all, so the most demanding one decides.
inline constexpr std::int64_t kParallelThreshold = 1 << 15;

// How much work one task gets. With the pool's fixed cost per job at ~1.5 us
// (spin and lock-free pickup, docs/design/array.md §10.1), small tasks balance
// load well.
inline constexpr std::int64_t kParallelTask = 1 << 12;

}  // namespace sd
