// The build of the 3-D Delaunay triangulation (delaunay3.cpp; docs/design/spatial_geometry.md §3.4, milestone
// SPT10), called by the native class _Delaunay (delaunay2.cpp) for points with 3 columns.
#pragma once

#include "spatial/triangulation.hpp"

#include <cstdint>
#include <vector>

namespace sd::spatial::tri {

enum class Build3 { Ok, TooFew, Coplanar, Memory };

// The tetrahedra of the input indices, positively oriented, and their neighbours (-1: the hull), before
// the canonical form.
struct Internal3 {
    std::int64_t t = 0;
    std::vector<std::int32_t> s, nb;
};

// The bytes one tetrahedron (or ghost) may stand for at the peak of a build and the bytes of one input
// point besides (SPT164): the default limit is what physical memory holds.
inline constexpr std::int64_t kTetBytes = 128;
inline constexpr std::int64_t kPointBytes3 = 200;
// The neighbour encoding (4 * tetrahedron + facet in an int32) holds fewer than 2^29 tetrahedra.
inline constexpr std::int64_t kMaxTets = (std::int64_t{1} << 29) - 16;

// The largest number of tetrahedra (ghosts included) a build of n points may hold: physical memory's share.
std::int64_t default_max_tets(std::int64_t n);

// Triangulates T's points (T.n, T.pts set; d = 3): fills T.rep, T.m, T.range, T.pred and `out`. Memory when
// the tetrahedra would pass max_tets (nothing else is left allocated).
Build3 build_delaunay3(Triangulation& T, Internal3& out, std::int64_t max_tets, bool split);

}  // namespace sd::spatial::tri
