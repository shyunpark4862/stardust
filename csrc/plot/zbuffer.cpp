// The 3D renderer of sd.plot (zbuffer.hpp has the method).
#include "plot/zbuffer.hpp"

#include "parallel/pool.hpp"

#include <algorithm>
#include <deque>
#include <array>
#include <cmath>
#include <limits>

namespace sd::plot {

namespace {

struct P {  // a projected point
    double x, y, depth;  // sample coordinates; depth grows towards the viewer
    bool ok;
};

struct Tri {
    P p[3];
    const double* v[3];
    int y0, y1;  // sample rows it may touch
    bool glass;  // some vertex is translucent
};

struct Quad {  // a line segment or a disc, as screen geometry
    double ax, ay, bx, by, da, db, hw;  // a segment a-b of half width hw with depths; a disc has a == b
    float rgba[4];
    int y0, y1;
    bool disc;
    std::int32_t strip;  // of a line piece: the pieces of one strip overlap at their joints and count once there
};

constexpr int kBand = 48;
constexpr int kLayers = 12;  // translucent layers kept per sample, from the nearest
constexpr double kSolid = 0.999;  // an alpha from here on is opaque
constexpr double kFill[3] = {0.7285, -0.2428, 0.6406};  // the fill light: from the right, a little below, in front
constexpr double kRim[3] = {0.0, 0.6, -0.8};  // the rim light: from behind, above

inline P project(const View3& v, const double* s, double k) {
    const double x = v.rot[0] * s[0] + v.rot[1] * s[1] + v.rot[2] * s[2];
    const double y = v.rot[3] * s[0] + v.rot[4] * s[1] + v.rot[5] * s[2];
    const double z = v.rot[6] * s[0] + v.rot[7] * s[1] + v.rot[8] * s[2];
    double f = 1.0;
    if (v.distance > 0.0) {
        const double w = v.distance - z;
        if (!(w > 1e-9)) return {0.0, 0.0, 0.0, false};  // behind the eye
        f = v.distance / w;
    }
    const double px = (v.cx + v.scale * f * x) * k, py = (v.cy - v.scale * f * y) * k;
    return {px, py, z, std::isfinite(px) && std::isfinite(py) && std::isfinite(z)};
}

}  // namespace

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
) {
    const int W = width * ss, H = height * ss;
    const std::size_t n = static_cast<std::size_t>(W) * H;
    std::vector<float> depth(n, -std::numeric_limits<float>::infinity());
    std::vector<float> color(n * 4, 0.0f);  // premultiplied

    std::vector<Tri> tris, glass;
    tris.reserve(ntri);
    double zmin = std::numeric_limits<double>::infinity(), zmax = -zmin;
    // A triangle that reaches out of the height range is cut at the range's two planes before it
    // is projected (the cut of a long triangle seen in perspective is not a straight interpolation
    // on the screen); the new corners live here.
    std::deque<std::array<double, 10>> cut;
    auto emit = [&](const double* a, const double* b, const double* c) {
        Tri tr;
        tr.glass = false;
        const double* v[3] = {a, b, c};
        for (int k = 0; k < 3; ++k) {
            tr.p[k] = project(view, v[k], ss);
            if (!tr.p[k].ok) return;
            tr.v[k] = v[k];
            if (v[k][9] < kSolid) tr.glass = true;
        }
        const double ya = std::min({tr.p[0].y, tr.p[1].y, tr.p[2].y}), yb = std::max({tr.p[0].y, tr.p[1].y, tr.p[2].y});
        if (yb < 0.0 || ya > H) return;
        tr.y0 = std::max(0, static_cast<int>(std::floor(ya)));
        tr.y1 = std::min(H, static_cast<int>(std::ceil(yb)) + 1);
        for (int k = 0; k < 3; ++k) {
            zmin = std::min(zmin, tr.p[k].depth);
            zmax = std::max(zmax, tr.p[k].depth);
        }
        (tr.glass ? glass : tris).push_back(tr);
    };
    for (std::size_t t = 0; t < ntri; ++t) {
        const double* v[3];
        bool ok = true, out = false, under = false, over = false;
        for (int k = 0; k < 3; ++k) {
            const std::int64_t i = triangles[3 * t + k];
            if (i < 0 || static_cast<std::size_t>(i) >= nvert) {
                ok = false;
                break;
            }
            v[k] = vertices + 10 * i;
            for (int c = 0; c < 3; ++c) ok = ok && std::isfinite(v[k][c]);
            under = under || v[k][2] < view.zlo;
            over = over || v[k][2] > view.zhi;
        }
        // a triangle that leaves through the top and through the bottom spans a pole: no wall is drawn there
        if (!ok || (under && over)) continue;
        out = under || over;
        if (!out) {
            emit(v[0], v[1], v[2]);
            continue;
        }
        // Sutherland-Hodgman against z >= zlo, then z <= zhi
        const double* poly[8];
        int n = 3;
        for (int k = 0; k < 3; ++k) poly[k] = v[k];
        for (int side = 0; side < 2 && n >= 3; ++side) {
            const double bound = side == 0 ? view.zlo : view.zhi;
            const auto in = [&](const double* q) { return side == 0 ? q[2] >= bound : q[2] <= bound; };
            const double* next[8];
            int m = 0;
            for (int k = 0; k < n; ++k) {
                const double* a = poly[k];
                const double* b = poly[(k + 1) % n];
                const bool ia = in(a), ib = in(b);
                if (ia) next[m++] = a;
                if (ia != ib) {
                    const double w = (bound - a[2]) / (b[2] - a[2]);
                    std::array<double, 10> q;
                    for (int c = 0; c < 10; ++c) q[c] = a[c] + w * (b[c] - a[c]);
                    q[2] = bound;
                    cut.push_back(q);
                    next[m++] = cut.back().data();
                }
            }
            n = m;
            for (int k = 0; k < n; ++k) poly[k] = next[k];
        }
        for (int k = 1; k + 1 < n; ++k) emit(poly[0], poly[k], poly[k + 1]);
    }

    std::vector<Quad> quads;
    std::size_t at = 0;
    for (std::size_t s = 0; s < nstrips; ++s) {
        const std::size_t count = static_cast<std::size_t>(std::max<std::int64_t>(line_counts[s], 0));
        const double hw = std::max(0.5 * line_widths[s] * ss, 0.35);
        for (std::size_t i = 0; i + 1 < count; ++i) {
            const double* a = lines + 7 * (at + i);
            const double* b = a + 7;
            const P pa = project(view, a, ss), pb = project(view, b, ss);
            if (!pa.ok || !pb.ok) continue;
            if (a[2] < view.zlo || a[2] > view.zhi || b[2] < view.zlo || b[2] > view.zhi) continue;
            Quad q{pa.x, pa.y, pb.x, pb.y, pa.depth, pb.depth, hw, {}, 0, 0, false, static_cast<std::int32_t>(s)};
            for (int k = 0; k < 4; ++k) q.rgba[k] = static_cast<float>(0.5 * (a[3 + k] + b[3 + k]));
            q.y0 = std::max(0, static_cast<int>(std::floor(std::min(pa.y, pb.y) - hw - 1)));
            q.y1 = std::min(H, static_cast<int>(std::ceil(std::max(pa.y, pb.y) + hw + 1)) + 1);
            if (q.y0 < q.y1) quads.push_back(q);
            zmin = std::min({zmin, pa.depth, pb.depth});
            zmax = std::max({zmax, pa.depth, pb.depth});
        }
        at += count;
    }
    for (std::size_t i = 0; i < ndiscs; ++i) {
        const double* d = discs + 8 * i;
        const P p = project(view, d, ss);
        if (!p.ok || !(d[3] > 0.0)) continue;
        const double r = d[3] * ss;
        Quad q{p.x, p.y, p.x, p.y, p.depth, p.depth, r, {}, 0, 0, true, -1};
        for (int k = 0; k < 4; ++k) q.rgba[k] = static_cast<float>(d[4 + k]);
        q.y0 = std::max(0, static_cast<int>(std::floor(p.y - r - 1)));
        q.y1 = std::min(H, static_cast<int>(std::ceil(p.y + r + 1)) + 1);
        if (q.y0 < q.y1) quads.push_back(q);
        zmin = std::min(zmin, p.depth);
        zmax = std::max(zmax, p.depth);
    }
    // lines lie on their surface: they pass the depth test with this allowance
    const float bias = static_cast<float>(std::isfinite(zmax - zmin) ? 0.004 * (zmax - zmin) + 1e-9 : 1e-9);

    const double lx = view.light[0], ly = view.light[1], lz = view.light[2];
    const double hl =
        std::sqrt(lx * lx + ly * ly + (lz + 1.0) * (lz + 1.0));  // the half vector of the light and the view (0, 0, 1)
    const double hx = lx / hl, hy = ly / hl, hz = (lz + 1.0) / hl;
    const double fl = std::sqrt(kFill[0] * kFill[0] + kFill[1] * kFill[1] + (kFill[2] + 1.0) * (kFill[2] + 1.0));
    const double fx = kFill[0] / fl, fy = kFill[1] / fl, fz = (kFill[2] + 1.0) / fl;  // the fill light's half vector

    // The colour of a surface sample (straight, in o) and its alpha. A translucent sheet seen at a
    // slant is looked through along a longer way: its alpha grows towards the outline.
    auto shade_at = [&](const Tri& tr, double w0, double w1, double w2, double* o) -> double {
        const double nx = w0 * tr.v[0][3] + w1 * tr.v[1][3] + w2 * tr.v[2][3];
        const double ny = w0 * tr.v[0][4] + w1 * tr.v[1][4] + w2 * tr.v[2][4];
        const double nz = w0 * tr.v[0][5] + w1 * tr.v[1][5] + w2 * tr.v[2][5];
        // the normal in view space, turned to face the viewer
        double vx = view.rot[0] * nx + view.rot[1] * ny + view.rot[2] * nz;
        double vy = view.rot[3] * nx + view.rot[4] * ny + view.rot[5] * nz;
        double vz = view.rot[6] * nx + view.rot[7] * ny + view.rot[8] * nz;
        const double len = std::sqrt(vx * vx + vy * vy + vz * vz);
        double shade = 1.0, spec = 0.0, facing = 1.0;
        if (len > 0.0) {
            const double sgn = vz < 0.0 ? -1.0 / len : 1.0 / len;
            vx *= sgn;
            vy *= sgn;
            vz *= sgn;
            facing = vz;
            const double* k = view.shade;
            const double key = std::max(0.0, vx * lx + vy * ly + vz * lz);
            const double fill = std::max(0.0, vx * kFill[0] + vy * kFill[1] + vz * kFill[2]);
            const double rim = std::max(0.0, vx * kRim[0] + vy * kRim[1] + vz * kRim[2]);
            shade = (k[0] + k[1] * key + k[2] * fill + k[3] * rim) * (1.0 - k[4] + k[4] * vz);
            const double nh = std::max(0.0, vx * hx + vy * hy + vz * hz);
            const double nf = std::max(0.0, vx * fx + vy * fy + vz * fz);
            spec = k[5] * std::pow(nh, k[6]) + k[7] * std::pow(nh, k[8]) + k[9] * std::pow(nf, k[6]);
        }
        double al = w0 * tr.v[0][9] + w1 * tr.v[1][9] + w2 * tr.v[2][9];
        if (tr.glass) al = 1.0 - std::pow(1.0 - std::min(al, 1.0), 1.0 / std::max(facing, 0.3));
        for (int k = 0; k < 3; ++k) {
            const double base = w0 * tr.v[0][6 + k] + w1 * tr.v[1][6 + k] + w2 * tr.v[2][6 + k];
            o[k] = std::clamp(base * shade + spec, 0.0, 1.0);
        }
        return al;
    };

    // Walks the samples of a triangle inside the rows [r0, r1): f(at, w0, w1, w2, depth).
    auto each_tri = [&](const Tri& tr, int r0, int r1, auto&& f) {
        if (tr.y1 <= r0 || tr.y0 >= r1) return;
        const P &a = tr.p[0], &b = tr.p[1], &c = tr.p[2];
        const double area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        if (area == 0.0) return;
        const double inv = 1.0 / area;
        const int x0 = std::max(0, static_cast<int>(std::floor(std::min({a.x, b.x, c.x}))));
        const int x1 = std::min(W, static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))) + 1);
        const int ya = std::max(tr.y0, r0), yb = std::min(tr.y1, r1);
        for (int y = ya; y < yb; ++y) {
            const double py = y + 0.5;
            for (int x = x0; x < x1; ++x) {
                const double px = x + 0.5;
                const double w0 = ((b.x - px) * (c.y - py) - (c.x - px) * (b.y - py)) * inv;
                const double w1 = ((c.x - px) * (a.y - py) - (a.x - px) * (c.y - py)) * inv;
                const double w2 = 1.0 - w0 - w1;
                if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) continue;
                const double sz = w0 * tr.v[0][2] + w1 * tr.v[1][2] + w2 * tr.v[2][2];
                if (sz < view.zlo || sz > view.zhi) continue;
                f(static_cast<std::size_t>(y) * W + x, w0, w1, w2,
                  static_cast<float>(w0 * a.depth + w1 * b.depth + w2 * c.depth));
            }
        }
    };

    // Walks the samples of a line piece or a disc inside the rows [r0, r1): f(at, depth).
    auto each_quad = [&](const Quad& q, int r0, int r1, auto&& f) {
        if (q.y1 <= r0 || q.y0 >= r1) return;
        const double dx = q.bx - q.ax, dy = q.by - q.ay;
        const double len2 = dx * dx + dy * dy;
        const int x0 = std::max(0, static_cast<int>(std::floor(std::min(q.ax, q.bx) - q.hw - 1)));
        const int x1 = std::min(W, static_cast<int>(std::ceil(std::max(q.ax, q.bx) + q.hw + 1)) + 1);
        const int ya = std::max(q.y0, r0), yb = std::min(q.y1, r1);
        for (int y = ya; y < yb; ++y) {
            const double py = y + 0.5;
            for (int x = x0; x < x1; ++x) {
                const double px = x + 0.5;
                double t = len2 > 0.0 ? ((px - q.ax) * dx + (py - q.ay) * dy) / len2 : 0.0;
                t = std::clamp(t, 0.0, 1.0);
                const double ex = px - (q.ax + t * dx), ey = py - (q.ay + t * dy);
                if (ex * ex + ey * ey > q.hw * q.hw) continue;
                f(static_cast<std::size_t>(y) * W + x, static_cast<float>(q.da + t * (q.db - q.da)));
            }
        }
    };

    auto over = [&](std::size_t at2, const float* rgba) {
        float* o = color.data() + 4 * at2;
        const float a = rgba[3], k = 1.0f - a;
        o[0] = rgba[0] * a + o[0] * k;
        o[1] = rgba[1] * a + o[1] * k;
        o[2] = rgba[2] * a + o[2] * k;
        o[3] = a + o[3] * k;
    };

    const bool peel = !glass.empty();
    std::vector<std::int32_t> mark(peel ? 0 : n, -1);  // the strip last painted on a sample
    auto band = [&](std::int64_t task) {
        const int r0 = static_cast<int>(task) * kBand, r1 = std::min(H, r0 + kBand);
        for (const Tri& tr : tris) {
            each_tri(tr, r0, r1, [&](std::size_t at2, double w0, double w1, double w2, float d) {
                if (!(d > depth[at2])) return;
                depth[at2] = d;
                double c[3];
                const double al = shade_at(tr, w0, w1, w2, c);
                float* o = color.data() + 4 * at2;
                for (int k = 0; k < 3; ++k) o[k] = static_cast<float>(c[k] * al);
                o[3] = static_cast<float>(al);
            });
        }
        if (!peel) {  // nothing translucent but the lines: lines and points test and keep the depth, in their order
            for (const Quad& q : quads) {
                each_quad(q, r0, r1, [&](std::size_t at2, float d) {
                    if (d + bias < depth[at2]) return;
                    if (!q.disc) {
                        if (mark[at2] == q.strip) return;
                        mark[at2] = q.strip;
                    }
                    over(at2, q.rgba);
                    if (d > depth[at2]) depth[at2] = d;  // a line or a point hides what lies behind it
                });
            }
            return;
        }
        // Translucent surfaces: opaque points first, then the translucent samples layer by layer
        // from the nearest, each pass taking of every sample the nearest piece behind the last
        // one taken; the layers are gathered front to back and laid over the opaque picture.
        for (const Quad& q : quads) {
            if (!(q.disc && q.rgba[3] >= kSolid)) continue;
            each_quad(q, r0, r1, [&](std::size_t at2, float d) {
                if (d + bias < depth[at2]) return;
                over(at2, q.rgba);
                if (d > depth[at2]) depth[at2] = d;
            });
        }
        const std::size_t lo = static_cast<std::size_t>(r0) * W, count = static_cast<std::size_t>(r1 - r0) * W;
        const float inf = std::numeric_limits<float>::infinity();
        std::vector<float> last(count, inf), cand(count), acc(count * 4, 0.0f), bw(count * 2);
        std::vector<std::int32_t> who(count), taken(count, -1);  // taken: the strip of the layer last gathered
        const std::int32_t nglass = static_cast<std::int32_t>(glass.size());
        for (int layer = 0; layer < kLayers; ++layer) {
            std::fill(cand.begin(), cand.end(), -inf);
            for (std::int32_t gi = 0; gi < nglass; ++gi) {
                each_tri(glass[gi], r0, r1, [&](std::size_t at2, double w0, double w1, double, float d) {
                    const std::size_t i = at2 - lo;
                    if (!(d > depth[at2]) || !(d < last[i]) || !(d > cand[i])) return;
                    cand[i] = d;
                    who[i] = gi;
                    bw[2 * i] = static_cast<float>(w0);
                    bw[2 * i + 1] = static_cast<float>(w1);
                });
            }
            for (std::size_t qi = 0; qi < quads.size(); ++qi) {
                const Quad& q = quads[qi];
                if (q.disc && q.rgba[3] >= kSolid) continue;
                each_quad(q, r0, r1, [&](std::size_t at2, float d) {
                    const std::size_t i = at2 - lo;
                    d += bias;  // a line lies just in front of its surface
                    if (d < depth[at2] || !(d < last[i]) || !(d > cand[i])) return;
                    if (!q.disc && q.strip == taken[i]) return;
                    cand[i] = d;
                    who[i] = nglass + static_cast<std::int32_t>(qi);
                });
            }
            bool any = false;
            for (std::size_t i = 0; i < count; ++i) {
                if (cand[i] == -inf) {
                    last[i] = -inf;  // nothing more here
                    continue;
                }
                last[i] = cand[i];
                taken[i] = who[i] < nglass ? -1 : quads[who[i] - nglass].strip;
                float* a = acc.data() + 4 * i;
                if (a[3] > 0.996f) continue;  // what lies behind does not show
                any = true;
                float src[4];
                if (who[i] < nglass) {
                    const double w0 = bw[2 * i], w1 = bw[2 * i + 1];
                    double c[3];
                    src[3] = static_cast<float>(shade_at(glass[who[i]], w0, w1, 1.0 - w0 - w1, c));
                    for (int k = 0; k < 3; ++k) src[k] = static_cast<float>(c[k]);
                } else {
                    const Quad& q = quads[who[i] - nglass];
                    for (int k = 0; k < 4; ++k) src[k] = q.rgba[k];
                }
                const float t = (1.0f - a[3]) * src[3];
                a[0] += t * src[0];
                a[1] += t * src[1];
                a[2] += t * src[2];
                a[3] += t;
            }
            if (!any) break;
        }
        for (std::size_t i = 0; i < count; ++i) {
            const float* a = acc.data() + 4 * i;
            if (a[3] <= 0.0f) continue;
            float* o = color.data() + 4 * (lo + i);
            const float k = 1.0f - a[3];
            o[0] = a[0] + o[0] * k;
            o[1] = a[1] + o[1] * k;
            o[2] = a[2] + o[2] * k;
            o[3] = a[3] + o[3] * k;
        }
    };
    const std::int64_t tasks = (H + kBand - 1) / kBand;
    if (tasks > 1 && n >= 65536) {
        sd::parallel::parallel_for(tasks, band);
    } else {
        for (std::int64_t t = 0; t < tasks; ++t) band(t);
    }

    std::vector<std::uint8_t> out(static_cast<std::size_t>(width) * height * 4, 0);
    const float norm = 1.0f / static_cast<float>(ss * ss);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int j = 0; j < ss; ++j) {
                const float* row =
                    color.data() + 4 * (static_cast<std::size_t>(y * ss + j) * W + static_cast<std::size_t>(x) * ss);
                for (int i = 0; i < ss; ++i) {
                    s[0] += row[4 * i];
                    s[1] += row[4 * i + 1];
                    s[2] += row[4 * i + 2];
                    s[3] += row[4 * i + 3];
                }
            }
            if (s[3] <= 0.0f) continue;
            std::uint8_t* o = out.data() + 4 * (static_cast<std::size_t>(y) * width + x);
            const float k = 255.0f / s[3];
            o[0] = static_cast<std::uint8_t>(std::min(255.0f, s[0] * k + 0.5f));
            o[1] = static_cast<std::uint8_t>(std::min(255.0f, s[1] * k + 0.5f));
            o[2] = static_cast<std::uint8_t>(std::min(255.0f, s[2] * k + 0.5f));
            o[3] = static_cast<std::uint8_t>(std::min(255.0f, s[3] * norm * 255.0f + 0.5f));
        }
    }
    return out;
}

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
) {
    // the triangles that can be drawn, with their areas in the scene
    std::vector<std::int64_t> tri;
    std::vector<double> area;
    double total = 0.0;
    for (std::size_t t = 0; t < ntri; ++t) {
        const std::int64_t* q = triangles + 3 * t;
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            ok = ok && q[k] >= 0 && static_cast<std::size_t>(q[k]) < nvert;
            for (int c = 0; ok && c < 3; ++c) ok = std::isfinite(xyz[3 * q[k] + c]);
        }
        if (!ok) continue;
        const double *a = xyz + 3 * q[0], *b = xyz + 3 * q[1], *d = xyz + 3 * q[2];
        const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
        const double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
        const double ar = 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (!(ar > 0.0)) continue;
        tri.insert(tri.end(), {q[0], q[1], q[2]});
        area.push_back(ar);
        total += ar;
    }
    const std::size_t nt = area.size();
    const double scale = 0.5 * size / (1.15 * radius);  // samples per unit at the origin's depth
    std::vector<double> out(2 * nviews, 0.0);
    auto one = [&](std::int64_t vi) {
        const double* r = views + 9 * vi;
        const auto project = [&](const double* p, double& x, double& y, double& z) {
            const double vx = r[0] * p[0] + r[1] * p[1] + r[2] * p[2];
            const double vy = r[3] * p[0] + r[4] * p[1] + r[5] * p[2];
            z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2];
            const double f = distance > 0.0 ? distance / std::max(distance - z, 1e-9) : 1.0;
            x = 0.5 * size + scale * f * vx;
            y = 0.5 * size - scale * f * vy;
        };
        if (nt) {
            std::vector<double> px(nvert), py(nvert), pz(nvert);
            for (std::size_t i = 0; i < nvert; ++i) project(xyz + 3 * i, px[i], py[i], pz[i]);
            const std::size_t n = static_cast<std::size_t>(size) * size;
            std::vector<float> depth(n, -std::numeric_limits<float>::infinity());
            std::vector<std::int32_t> who(n, -1);
            std::vector<double> flat(nt);  // the triangles' areas on the screen
            for (std::size_t t = 0; t < nt; ++t) {
                const std::int64_t i0 = tri[3 * t], i1 = tri[3 * t + 1], i2 = tri[3 * t + 2];
                const double ax = px[i0], ay = py[i0], bx = px[i1], by = py[i1], cx = px[i2], cy = py[i2];
                const double twice = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
                flat[t] = 0.5 * std::fabs(twice);
                if (!(twice != 0.0) || !std::isfinite(twice)) continue;
                const double inv = 1.0 / twice;
                const int x0 = std::max(0, static_cast<int>(std::floor(std::min({ax, bx, cx}))));
                const int x1 = std::min(size, static_cast<int>(std::ceil(std::max({ax, bx, cx}))) + 1);
                const int y0 = std::max(0, static_cast<int>(std::floor(std::min({ay, by, cy}))));
                const int y1 = std::min(size, static_cast<int>(std::ceil(std::max({ay, by, cy}))) + 1);
                for (int y = y0; y < y1; ++y) {
                    const double qy = y + 0.5;
                    for (int x = x0; x < x1; ++x) {
                        const double qx = x + 0.5;
                        const double w0 = ((bx - qx) * (cy - qy) - (cx - qx) * (by - qy)) * inv;
                        const double w1 = ((cx - qx) * (ay - qy) - (ax - qx) * (cy - qy)) * inv;
                        const double w2 = 1.0 - w0 - w1;
                        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) continue;
                        const float d = static_cast<float>(w0 * pz[i0] + w1 * pz[i1] + w2 * pz[i2]);
                        const std::size_t at = static_cast<std::size_t>(y) * size + x;
                        if (d > depth[at]) {
                            depth[at] = d;
                            who[at] = static_cast<std::int32_t>(t);
                        }
                    }
                }
            }
            std::vector<std::int32_t> seen(nt, 0);
            for (std::size_t i = 0; i < n; ++i) {
                if (who[i] >= 0) ++seen[who[i]];
            }
            double sum = 0.0;
            for (std::size_t t = 0; t < nt; ++t) {
                if (!(flat[t] > 0.0)) continue;
                const double share = std::min(1.0, seen[t] / std::max(flat[t], 1.0));
                // the size a triangle facing the eye would have there (nearer ones are drawn larger)
                const double zc = (pz[tri[3 * t]] + pz[tri[3 * t + 1]] + pz[tri[3 * t + 2]]) / 3.0;
                const double f = distance > 0.0 ? distance / std::max(distance - zc, 1e-9) : 1.0;
                const double cosine = std::min(1.0, flat[t] / (area[t] * scale * scale * f * f));
                sum += area[t] * share * std::sqrt(std::sqrt(cosine));
            }
            out[2 * vi] = sum / total;
        }
        if (npoints >= 2) {
            double mx = 0.0, my = 0.0;
            std::vector<double> qx(npoints), qy(npoints);
            std::size_t m = 0;
            for (std::size_t i = 0; i < npoints; ++i) {
                double z;
                if (!(std::isfinite(points[3 * i]) && std::isfinite(points[3 * i + 1]) &&
                      std::isfinite(points[3 * i + 2])))
                    continue;
                project(points + 3 * i, qx[m], qy[m], z);
                mx += qx[m];
                my += qy[m];
                ++m;
            }
            if (m >= 2) {
                mx /= m;
                my /= m;
                double sxx = 0.0, sxy = 0.0, syy = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    sxx += (qx[i] - mx) * (qx[i] - mx);
                    sxy += (qx[i] - mx) * (qy[i] - my);
                    syy += (qy[i] - my) * (qy[i] - my);
                }
                const double det = (sxx * syy - sxy * sxy) / (static_cast<double>(m) * m);
                out[2 * vi + 1] = std::sqrt(std::max(det, 0.0)) / (scale * scale);
            }
        }
    };
    if (nviews > 1) {
        sd::parallel::parallel_for(static_cast<std::int64_t>(nviews), one);
    } else if (nviews == 1) {
        one(0);
    }
    return out;
}

}  // namespace sd::plot
