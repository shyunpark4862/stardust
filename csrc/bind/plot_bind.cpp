// The plot area: the submodule stardust._core.plot (docs/design/plot.md §3.5, §7).
//
// Canvas is the 2D rasterizer (plot/raster.hpp). Python hands a path as its ops (bytes over
// "MLQCZ") and its coordinates (packed doubles), with the matrix to pixels, the colour and the
// clip rectangle as plain numbers: the callers are stardust.plot's own modules, which check
// what users give, so a malformed call here is a programming error and raises ValueError.
#include "bind/plot_bind.hpp"

#include "plot/levels.hpp"
#include "plot/raster.hpp"
#include "plot/zbuffer.hpp"

#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd {

namespace {

using plot::Canvas;

struct Args {
    const char* ops;
    std::size_t nops;
    std::vector<double> xy;
    double m[6];
    float rgba[4];
    plot::Clip clip;
    bool has_clip;
};

std::vector<double> doubles(nb::bytes b) {
    if (b.size() % sizeof(double) != 0) throw std::invalid_argument("packed doubles expected");
    std::vector<double> v(b.size() / sizeof(double));
    if (!v.empty()) std::memcpy(v.data(), b.c_str(), b.size());
    return v;
}

// clip: empty for none, else x0, y0, x1, y1 in pixels
Args unpack(
    nb::bytes ops,
    nb::bytes xy,
    const std::vector<double>& m,
    const std::vector<double>& rgba,
    const std::vector<double>& clip
) {
    Args a;
    a.ops = ops.c_str();
    a.nops = ops.size();
    if (xy.size() % sizeof(double) != 0) throw std::invalid_argument("xy must be packed doubles");
    a.xy.resize(xy.size() / sizeof(double));
    if (!a.xy.empty()) std::memcpy(a.xy.data(), xy.c_str(), xy.size());
    std::size_t need = 0;
    for (std::size_t i = 0; i < a.nops; ++i) {
        switch (a.ops[i]) {
            case 'M':
            case 'L': need += 2; break;
            case 'Q': need += 4; break;
            case 'C': need += 6; break;
            case 'Z': break;
            default: throw std::invalid_argument("ops must be over MLQCZ");
        }
    }
    if (need != a.xy.size()) throw std::invalid_argument("ops and xy do not agree");
    if (m.size() != 6 || rgba.size() != 4) throw std::invalid_argument("matrix needs 6 numbers, rgba 4");
    for (int i = 0; i < 6; ++i) a.m[i] = m[i];
    for (int i = 0; i < 4; ++i) a.rgba[i] = static_cast<float>(rgba[i]);
    a.has_clip = !clip.empty();
    if (a.has_clip) {
        if (clip.size() != 4) throw std::invalid_argument("clip needs 4 numbers");
        a.clip = {clip[0], clip[1], clip[2], clip[3]};
    }
    return a;
}

}  // namespace

void bind_plot(nb::module_& core) {
    nb::module_ m = core.def_submodule("plot", "stardust plot native core");
    nb::class_<Canvas>(m, "Canvas")
        .def(
            "__init__",
            [](Canvas* self, int width, int height) {
                if (width < 1 || height < 1 || width > 65536 || height > 65536 ||
                    static_cast<long long>(width) * height > (1LL << 28)) {
                    throw std::invalid_argument("canvas size out of range");
                }
                new (self) Canvas(width, height);
            }
        )
        .def_prop_ro("width", &Canvas::width)
        .def_prop_ro("height", &Canvas::height)
        .def(
            "fill",
            [](Canvas& c, nb::bytes ops, nb::bytes xy, const std::vector<double>& mat, const std::vector<double>& rgba,
               const std::vector<double>& clip) {
                const Args a = unpack(ops, xy, mat, rgba, clip);
                c.fill(a.ops, a.nops, a.xy.data(), a.xy.size(), a.m, a.rgba, a.has_clip ? &a.clip : nullptr);
            }
        )
        .def(
            "stroke",
            [](Canvas& c, nb::bytes ops, nb::bytes xy, const std::vector<double>& mat, const std::vector<double>& rgba,
               const std::vector<double>& clip, double width, int cap, int join, const std::vector<double>& dash) {
                const Args a = unpack(ops, xy, mat, rgba, clip);
                if (cap < 0 || cap > 2 || join < 0 || join > 1) throw std::invalid_argument("bad cap or join");
                c.stroke(
                    a.ops, a.nops, a.xy.data(), a.xy.size(), a.m, a.rgba, width, static_cast<plot::Cap>(cap),
                    static_cast<plot::Join>(join), dash.data(), dash.size(), a.has_clip ? &a.clip : nullptr
                );
            }
        )
        .def(
            "markers",
            [](Canvas& c, nb::bytes ops, nb::bytes xy, const std::vector<double>& mat, nb::bytes at,
               const std::vector<double>& fill, const std::vector<double>& stroke, double width,
               const std::vector<double>& clip) {
                const std::vector<double> none{0.0, 0.0, 0.0, 0.0};
                const Args a = unpack(ops, xy, mat, none, clip);
                if (at.size() % (2 * sizeof(double)) != 0)
                    throw std::invalid_argument("at must be packed pairs of doubles");
                std::vector<double> pos(at.size() / sizeof(double));
                if (!pos.empty()) std::memcpy(pos.data(), at.c_str(), at.size());
                if ((!fill.empty() && fill.size() != 4) || (!stroke.empty() && stroke.size() != 4)) {
                    throw std::invalid_argument("fill and stroke need 4 numbers or none");
                }
                float f[4], s[4];
                for (std::size_t i = 0; i < fill.size(); ++i) f[i] = static_cast<float>(fill[i]);
                for (std::size_t i = 0; i < stroke.size(); ++i) s[i] = static_cast<float>(stroke[i]);
                c.markers(
                    a.ops, a.nops, a.xy.data(), a.xy.size(), a.m, pos.data(), pos.size() / 2,
                    fill.empty() ? nullptr : f, stroke.empty() ? nullptr : s, width, a.has_clip ? &a.clip : nullptr
                );
            }
        )
        .def(
            "image",
            [](Canvas& c, nb::bytes rgba, int iw, int ih, double x, double y, double w, double h, bool smooth,
               const std::vector<double>& clip) {
                if (iw < 1 || ih < 1 || rgba.size() != static_cast<std::size_t>(iw) * ih * 4) {
                    throw std::invalid_argument("rgba must hold iw x ih pixels of 4 bytes");
                }
                plot::Clip cl{};
                if (!clip.empty()) {
                    if (clip.size() != 4) throw std::invalid_argument("clip needs 4 numbers");
                    cl = {clip[0], clip[1], clip[2], clip[3]};
                }
                c.image(
                    reinterpret_cast<const std::uint8_t*>(rgba.c_str()), iw, ih, x, y, w, h, smooth,
                    clip.empty() ? nullptr : &cl
                );
            }
        )
        .def("rgba8", [](const Canvas& c) {
            const std::vector<std::uint8_t> px = c.rgba8();
            return nb::bytes(reinterpret_cast<const char*>(px.data()), px.size());
        });
    m.def(
        "render3", [](int width, int height, int ss, const std::vector<double>& view, nb::bytes vertices,
                      nb::bytes triangles, nb::bytes lines, const std::vector<std::int64_t>& line_counts,
                      const std::vector<double>& line_widths, nb::bytes discs) {
            if (width < 1 || height < 1 || ss < 1 || ss > 8 ||
                static_cast<long long>(width) * height * ss * ss > (1LL << 28)) {
                throw std::invalid_argument("picture size out of range");
            }
            if (view.size() != 18 && view.size() != 28) throw std::invalid_argument("view needs 18 or 28 numbers");
            if (line_widths.size() != line_counts.size()) throw std::invalid_argument("a width for every strip");
            if (vertices.size() % (10 * sizeof(double)) != 0 || triangles.size() % (3 * sizeof(std::int64_t)) != 0 ||
                lines.size() % (7 * sizeof(double)) != 0 || discs.size() % (8 * sizeof(double)) != 0) {
                throw std::invalid_argument("vertices, triangles, lines or discs have the wrong size");
            }
            plot::View3 v{};
            for (int i = 0; i < 9; ++i) v.rot[i] = view[i];
            v.distance = view[9];
            v.scale = view[10];
            v.cx = view[11];
            v.cy = view[12];
            v.zlo = view[13];
            v.zhi = view[14];
            for (int i = 0; i < 3; ++i) v.light[i] = view[15 + i];
            if (view.size() == 28) {
                for (int i = 0; i < 10; ++i) v.shade[i] = view[18 + i];
            }
            std::vector<double> vert(vertices.size() / sizeof(double)), ln(lines.size() / sizeof(double)),
                ds(discs.size() / sizeof(double));
            std::vector<std::int64_t> tri(triangles.size() / sizeof(std::int64_t));
            if (!vert.empty()) std::memcpy(vert.data(), vertices.c_str(), vertices.size());
            if (!tri.empty()) std::memcpy(tri.data(), triangles.c_str(), triangles.size());
            if (!ln.empty()) std::memcpy(ln.data(), lines.c_str(), lines.size());
            if (!ds.empty()) std::memcpy(ds.data(), discs.c_str(), discs.size());
            std::int64_t total = 0;
            for (const std::int64_t c : line_counts) total += c < 0 ? 0 : c;
            if (static_cast<std::size_t>(total) * 7 != ln.size())
                throw std::invalid_argument("line_counts do not add up to the points");
            const std::vector<std::uint8_t> px = plot::render3(
                width, height, ss, v, vert.data(), vert.size() / 10, tri.data(), tri.size() / 3, ln.data(),
                line_counts.data(), line_counts.size(), line_widths.data(), ds.data(), ds.size() / 8
            );
            return nb::bytes(reinterpret_cast<const char*>(px.data()), px.size());
        }
    );
    // Marching squares (plot/levels.hpp). F: len(ys) rows of len(xs) packed doubles; points: x, y
    // of every crossed edge of the level, in the order of edges().
    nb::class_<plot::Levels>(m, "Levels")
        .def(
            "__init__", [](plot::Levels* self, nb::bytes F, nb::bytes xs,
                           nb::bytes ys) { new (self) plot::Levels(doubles(F), doubles(xs), doubles(ys)); }
        )
        .def(
            "edges",
            [](plot::Levels& self, double level) {
                const std::vector<double> out = self.edges(level);
                return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(double));
            }
        )
        .def(
            "curves",
            [](plot::Levels& self, double level, nb::bytes points) {
                const std::vector<double> out = self.curves(level, doubles(points));
                return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(double));
            }
        )
        .def("inside", [](plot::Levels& self, double level, nb::bytes points) {
            std::string ops;
            std::vector<double> xy;
            self.inside(level, doubles(points), ops, xy);
            return std::pair(
                nb::bytes(ops.data(), ops.size()),
                nb::bytes(reinterpret_cast<const char*>(xy.data()), xy.size() * sizeof(double))
            );
        });
    m.def("path_text", [](nb::bytes ops, nb::bytes xy, int digits, bool pdf) {
        std::size_t need = 0;
        for (std::size_t i = 0; i < ops.size(); ++i) {
            const char op = ops.c_str()[i];
            if (op != 'M' && op != 'L' && op != 'Q' && op != 'C' && op != 'Z')
                throw std::invalid_argument("unknown path op");
            need += op == 'Z' ? 0 : op == 'Q' ? 4 : op == 'C' ? 6 : 2;
        }
        const std::vector<double> v = doubles(xy);
        if (v.size() != need) throw std::invalid_argument("the coordinates do not match the ops");
        const std::string out = plot::path_text(ops.c_str(), ops.size(), v.data(), digits, pdf);
        return nb::str(out.data(), out.size());
    });
    m.def("decimate", [](nb::bytes xy, double column) {
        if (xy.size() % (2 * sizeof(double)) != 0 || !(column > 0.0)) {
            throw std::invalid_argument("xy must be packed pairs, column positive");
        }
        std::vector<double> v(xy.size() / sizeof(double));
        if (!v.empty()) std::memcpy(v.data(), xy.c_str(), xy.size());
        const std::vector<double> out = plot::decimate(v.data(), v.size() / 2, column);
        return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(double));
    });
    m.def("colormap", [](nb::bytes values, double vmin, double vmax, nb::bytes table) {
        if (values.size() % sizeof(double) != 0 || table.size() < 8 || table.size() % 4 != 0) {
            throw std::invalid_argument("values must be packed doubles and table at least two RGBA colours");
        }
        const std::size_t n = values.size() / sizeof(double);
        std::vector<double> v(n);
        if (n) std::memcpy(v.data(), values.c_str(), values.size());
        std::vector<std::uint8_t> out(4 * n);
        plot::colormap(
            v.data(), n, vmin, vmax, reinterpret_cast<const std::uint8_t*>(table.c_str()),
            static_cast<int>(table.size() / 4), out.data()
        );
        return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
    });
    m.def("phase_colors", [](nb::bytes re, nb::bytes im, bool bands) {
        if (re.size() != im.size() || re.size() % sizeof(double) != 0) {
            throw std::invalid_argument("re and im must be packed doubles of one length");
        }
        const std::size_t n = re.size() / sizeof(double);
        std::vector<double> a(n), b(n);
        if (n) {
            std::memcpy(a.data(), re.c_str(), re.size());
            std::memcpy(b.data(), im.c_str(), im.size());
        }
        std::vector<std::uint8_t> out(4 * n);
        plot::phase_colors(a.data(), b.data(), n, bands, out.data());
        return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
    });
}

}  // namespace sd
