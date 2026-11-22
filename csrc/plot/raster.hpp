// The 2D rasterizer of sd.plot (docs/design/plot.md §3.5).
//
// A Canvas is a pixel grid of premultiplied RGBA floats in the sRGB encoding (shapes are
// blended as the SVG and PDF viewers blend them, so the three outputs look alike). Two kinds of
// shape are painted, each with source-over:
//
//   fill    a path's interior. Coverage is the exact area of the outline inside each pixel
//           (signed areas accumulated per edge, then summed along the row), so there is no
//           sample pattern: the result is deterministic and smooth. Overlapping subpaths count
//           once when they turn the same way (glyphs, shapes with holes).
//   stroke  a path's outline of a given width: the stroke's own outline (both sides of the
//           path with joins and caps, a closed polygon per run) is filled the same way. Its
//           pieces may overlap inside sharp turns; they turn the same way, so they count once.
//
// Curves are flattened to segments within kFlatness of a pixel. Coordinates are pixels: x right,
// y down, pixel (i, j) covering [i, i+1) x [j, j+1). Nothing here depends on the thread count.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sd::plot {

inline constexpr double kFlatness = 0.05;    // pixels a flattened curve may leave its curve
inline constexpr double kMiterLimit = 10.0;  // miter length over half the width beyond which a join is a bevel

enum class Cap { Butt = 0, Round = 1, Square = 2 };
enum class Join { Miter = 0, Round = 1 };

// A clip rectangle in pixels; edges may fall inside pixels (coverage is scaled there).
struct Clip {
    double x0, y0, x1, y1;
};

class Canvas {
public:
    Canvas(int width, int height);

    int width() const { return w_; }
    int height() const { return h_; }

    // ops over "MLQCZ" with their coordinates (M, L: 2; Q: 4; C: 6; Z: 0), mapped to pixels by
    // m = (a, b, c, d, e, f): x' = a x + c y + e, y' = b x + d y + f. rgba is straight sRGB.
    void fill(
        const char* ops,
        std::size_t nops,
        const double* xy,
        std::size_t nxy,
        const double m[6],
        const float rgba[4],
        const Clip* clip
    );
    // width in pixels; dash: on/off lengths in pixels (ndash 0: solid).
    void stroke(
        const char* ops,
        std::size_t nops,
        const double* xy,
        std::size_t nxy,
        const double m[6],
        const float rgba[4],
        double width,
        Cap cap,
        Join join,
        const double* dash,
        std::size_t ndash,
        const Clip* clip
    );

    // The path stamped at each of the n points `at` (x, y pairs in the path's own coordinates,
    // added to it before m): its fill (if any) then its stroke (if any), one stamp after another.
    void markers(
        const char* ops,
        std::size_t nops,
        const double* xy,
        std::size_t nxy,
        const double m[6],
        const double* at,
        std::size_t n,
        const float* fill,
        const float* stroke,
        double width,
        const Clip* clip
    );

    // An image of iw x ih straight RGBA8 pixels drawn over the rectangle [x, x + w] x [y, y + h]
    // (pixels of the canvas; edges may fall inside pixels). smooth: bilinear between the image's
    // pixel centres; else each image pixel is a flat square. Rows are painted in parallel.
    void image(
        const std::uint8_t* rgba, int iw, int ih, double x, double y, double w, double h, bool smooth, const Clip* clip
    );

    // Straight (not premultiplied) 8-bit RGBA, row by row from the top.
    std::vector<std::uint8_t> rgba8() const;

private:
    struct Poly {  // flattened subpaths: points of all, the start of each, whether each is closed
        std::vector<double> pts;
        std::vector<std::size_t> start;
        std::vector<char> closed;
    };
    void flatten(
        const char* ops, std::size_t nops, const double* xy, std::size_t nxy, const double m[6], Poly& out
    ) const;
    struct Window {  // the pixels a shape is accumulated in: (bw + 2) floats a row
        std::vector<float> acc;
        std::vector<int> lo, hi;  // per row: the touched columns [lo, hi)
        int bx = 0, by = 0, bw = 0, bh = 0;
    };
    bool begin(Window& win, double x0, double y0, double x1, double y1, const Clip* clip, int rmin, int rmax) const;
    static void edge(Window& win, double x0, double y0, double x1, double y1);
    void composite(Window& win, const float rgba[4], const Clip* clip);

    int w_, h_;
    std::vector<float> px_;   // 4 floats a pixel
    Window win_;  // of fill and stroke (markers use one per band)
};

// A long polyline thinned for drawing: of each run of consecutive points whose x falls in the same
// column of width `column`, the first, the lowest, the highest and the last are kept. Nothing
// wider than a column changes. xy: n points (x, y); returns the kept points.
std::vector<double> decimate(const double* xy, std::size_t n, double column);

// The text of a path for a vector file: SVG path data ("M1 2L3 4Z"), or with `pdf` the PDF
// construction operators one a line ("1 2 m", a quadratic as the same cubic, "h"). Numbers have
// `digits` decimals at most, without trailing zeros. ops over "MLQCZ"; nxy is checked by the caller.
std::string path_text(const char* ops, std::size_t nops, const double* xy, int digits, bool pdf);

// The colours of n values through a table of `entries` RGBA8 colours: (v - vmin) / (vmax - vmin)
// clamped to the table's ends; a NaN is transparent. out has 4 n bytes.
void colormap(
    const double* v, std::size_t n, double vmin, double vmax, const std::uint8_t* table, int entries, std::uint8_t* out
);

// Domain colouring of n complex values (re, im): the argument is the hue; with `bands`, the
// brightness rises through each doubling of the modulus. Zero is black, infinity white, NaN
// transparent. out has 4 n bytes.
void phase_colors(const double* re, const double* im, std::size_t n, bool bands, std::uint8_t* out);

}  // namespace sd::plot
