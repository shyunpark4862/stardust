// The 3D renderer of sd.plot (docs/design/plot.md §6): a depth buffer.
//
// Triangles, line strips and discs given in the scene's own coordinates are turned by a rotation
// into view space (x right, y up, z towards the viewer), projected (orthographic, or perspective
// from a distance), and painted into a grid of ss x ss samples per pixel. Every sample keeps the
// nearest surface, so surfaces that cross each other are right; lines and discs are tested
// against that depth and painted over it. The colour of a surface sample is its vertex colours
// under fixed lights, both sides alike, from the normal
// interpolated across the triangle: a key light (View3::light), a weaker fill light from the
// other side, an ambient part, a slight darkening of faces turned away from the viewer, and a
// small clear highlight. The samples of a pixel are averaged at the end; a pixel nothing covers
// stays transparent.
//
// A surface with an alpha under 1 is translucent. Opaque surfaces are painted first; then every
// sample gathers the translucent pieces in front of them (surfaces and lines alike) in the order
// of their depth, the nearest first, up to twelve layers: surfaces that cross or lie behind one
// another show through each other rightly whatever the order they were given in. A translucent
// sheet seen at a slant is more opaque (the way through it is longer), which draws its outline.
//
// The sample pattern is a fixed grid and the rows are painted in bands of a fixed height, each
// band walking the primitives in their given order: the picture is the same at any thread count.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sd::plot {

struct View3 {
    double rot[9];    // row-major: view = rot * scene
    double distance;  // of the eye from the origin along view z; 0: orthographic
    double scale;     // pixels per unit of view x, y (at the origin's depth)
    double cx, cy;    // the pixel the origin lands on
    double zlo, zhi;  // samples of surfaces whose scene z is outside are dropped (a cut at the range)
    double light[3];  // towards the light, in view space (unit)
};

// vertices: 10 doubles each (x, y, z, nx, ny, nz, r, g, b, a), scene coordinates and straight sRGB.
// triangles: 3 vertex indices each. lines: 7 doubles a point (x, y, z, r, g, b, a) in strips whose
// lengths are line_counts; line_width in pixels. discs: 8 doubles each (x, y, z, radius in
// pixels, r, g, b, a). Returns width x height straight RGBA8, rows from the top.
std::vector<std::uint8_t> render3(
    int width,
    int height,
    int ss,
    const View3& view,
    const double* vertices,
    std::size_t nvert,
    const std::int64_t* triangles,
    std::size_t ntri,
    const double* lines,
    const std::int64_t* line_counts,
    std::size_t nstrips,
    double line_width,
    const double* discs,
    std::size_t ndiscs
);

}  // namespace sd::plot
