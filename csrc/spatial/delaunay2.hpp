// The build of the 2-D Delaunay triangulation (delaunay2.cpp; docs/design/spatial_geometry.md §3.3, milestone
// SPT9), called by the native class _Delaunay (delaunay2.cpp) and by the Voronoi diagram (voronoi.cpp,
// milestone SPT12) for points with 2 columns.
#pragma once

#include "spatial/triangulation.hpp"

#include <cstdint>
#include <vector>

namespace sd::spatial::tri {

enum class Build2 { Ok, TooFew, Collinear, TooMany };

// The triangles of the input indices, counterclockwise, and their neighbours (-1: the hull), before the
// canonical form.
struct Internal2 {
    std::int64_t m = 0, t = 0;
    std::vector<std::int32_t> s, nb;
};

// Triangulates T's points (T.n, T.pts set; d = 2): fills T.rep, T.m, T.range, T.pred and `out`.
Build2 build_delaunay2(Triangulation& T, Internal2& out, bool split);

}  // namespace sd::spatial::tri
