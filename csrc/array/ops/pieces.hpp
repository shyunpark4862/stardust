// Splitting a range of work into pieces on the thread pool, for the array
// functions that split by hand (docs/design/array.md §10.4).
#pragma once

#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cstdint>

namespace sd {

// fn(t, begin, end) for the pieces [t * per, min(n, (t + 1) * per)) of [0, n):
// on the pool when `work` (elements, roughly) reaches kParallelThreshold,
// there are two pieces or more and the pool is free; otherwise in order on
// the calling thread. The pieces depend on n and per only, never on the
// thread count, so a caller whose pieces combine in a fixed order gets the
// same bits on any number of threads.
// From kParallelThreshold on, the pieces run without the GIL wherever they
// run (docs/design/array.md §10.2): fn must not touch Python objects.
template <typename F>
void for_pieces(std::int64_t n, std::int64_t per, std::int64_t work, F&& fn) {
    per = std::max<std::int64_t>(per, 1);
    const std::int64_t pieces = (n + per - 1) / per;
    auto run = [&](std::int64_t t) { fn(t, t * per, std::min(n, (t + 1) * per)); };
    if (work < kParallelThreshold) {
        for (std::int64_t t = 0; t < pieces; ++t) run(t);
        return;
    }
    const parallel::Unlocked unlocked(true);
    if (pieces > 1 && parallel::available()) {
        parallel::parallel_for(pieces, run);
    } else {
        for (std::int64_t t = 0; t < pieces; ++t) run(t);
    }
}

}  // namespace sd
