// The 3D renderer of sd.plot (docs/design/plot.md §6): a depth buffer.
//
// Triangles, line strips and discs given in the scene's own coordinates are turned by a rotation
// into view space (x right, y up, z towards the viewer), projected (orthographic, or perspective
// from a distance), and painted into a grid of ss x ss samples per pixel. Every sample keeps the
// nearest surface, so surfaces that cross each other are right; lines and discs are tested
// against that depth and painted over it. The colour of a surface sample is its vertex colours
// under three fixed lights, both sides alike, from the normal interpolated across the triangle:
// a key light (View3::light), a weaker fill light from the other side and a rim light from
// behind, an ambient part, and white gloss on top (a broad sheen and a sharp highlight of the
// key light, a second sheen of the fill light): View3::shade has the strengths. The samples of a pixel are averaged at the end; a pixel nothing covers
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
    double rot[9];  // row-major: view = rot * scene
    double distance;  // of the eye from the origin along view z; 0: orthographic
    double scale;  // pixels per unit of view x, y (at the origin's depth)
    double cx, cy;  // the pixel the origin lands on
    double zlo, zhi;  // samples of surfaces whose scene z is outside are dropped (a cut at the range)
    double light[3];  // towards the key light, in view space (unit)
    // The shading: the ambient part; the diffuse strengths of the key light, of the fill light
    // (from the other side) and of the rim light (from behind, above); how much a face turned
    // away from the viewer darkens; the broad gloss of the key light (strength, exponent), its
    // sharp highlight (strength, exponent), and the broad gloss of the fill light.
    double shade[10] = {0.52, 0.36, 0.20, 0.10, 0.03, 0.18, 5.0, 0.30, 70.0, 0.12};
};

// vertices: 10 doubles each (x, y, z, nx, ny, nz, r, g, b, a), scene coordinates and straight sRGB.
// triangles: 3 vertex indices each. lines: 7 doubles a point (x, y, z, r, g, b, a) in strips whose
// lengths are line_counts and whose widths in pixels are line_widths. discs: 8 doubles each (x, y, z, radius in
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
    const double* line_widths,
    const double* discs,
    std::size_t ndiscs
);

// How well each of a list of camera directions shows a scene (docs/design/plot.md PL46): two
// numbers a view. The first is the share of the surfaces' area that is seen, each triangle
// weighted by how much of it the others leave uncovered (a small depth buffer of size x size
// samples tells) and by the fourth root of the cosine it is seen under, so that a face seen
// edge on counts for little while a slanted one still counts nearly whole. The second is the spread of the loose points on the screen: the
// square root of the determinant of their covariance there. xyz: 3 doubles a vertex, in the
// box's coordinates; views: 9 doubles each (row-major rotation, view = rot * scene); radius:
// the scene fits in a sphere of it around the origin. The views are scored side by side.
std::vector<double> view_scores(
    const double* xyz,
    std::size_t nvert,
    const std::int64_t* triangles,
    std::size_t ntri,
    const double* points,
    std::size_t npoints,
    const double* views,
    std::size_t nviews,
    double distance,
    double radius,
    int size
);

}  // namespace sd::plot
