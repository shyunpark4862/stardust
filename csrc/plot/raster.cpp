// The 2D rasterizer of sd.plot (raster.hpp has the method).
#include "plot/raster.hpp"

#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <cstring>

namespace sd::plot {

namespace {

inline void map(const double m[6], double x, double y, double& ox, double& oy) {
    ox = m[0] * x + m[2] * y + m[4];
    oy = m[1] * x + m[3] * y + m[5];
}

// the length of [lo, hi] inside [a, b]
inline double overlap(double lo, double hi, double a, double b) {
    return std::max(0.0, std::min(hi, b) - std::max(lo, a));
}

}  // namespace

Canvas::Canvas(int width, int height)
    : w_(width), h_(height), px_(static_cast<std::size_t>(width) * height * 4, 0.0f) {}

void Canvas::flatten(
    const char* ops, std::size_t nops, const double* xy, std::size_t nxy, const double m[6], Poly& out
) const {
    out.pts.clear();
    out.start.clear();
    out.closed.clear();
    double cx = 0.0, cy = 0.0;
    std::size_t k = 0;
    auto need = [&](std::size_t n) { return k + n <= nxy; };
    for (std::size_t i = 0; i < nops; ++i) {
        const char op = ops[i];
        if (op == 'M') {
            if (!need(2)) return;
            map(m, xy[k], xy[k + 1], cx, cy);
            k += 2;
            out.start.push_back(out.pts.size() / 2);
            out.closed.push_back(0);
            out.pts.push_back(cx);
            out.pts.push_back(cy);
        } else if (out.start.empty()) {
            return;  // a path must start with M
        } else if (op == 'L') {
            if (!need(2)) return;
            map(m, xy[k], xy[k + 1], cx, cy);
            k += 2;
            out.pts.push_back(cx);
            out.pts.push_back(cy);
        } else if (op == 'Q') {
            if (!need(4)) return;
            double x1, y1, x2, y2;
            map(m, xy[k], xy[k + 1], x1, y1);
            map(m, xy[k + 2], xy[k + 3], x2, y2);
            k += 4;
            const double dd = std::hypot(cx - 2 * x1 + x2, cy - 2 * y1 + y2);
            const int n = std::clamp(static_cast<int>(std::ceil(std::sqrt(0.25 * dd / kFlatness))), 1, 256);
            for (int j = 1; j <= n; ++j) {
                const double t = static_cast<double>(j) / n, u = 1.0 - t;
                out.pts.push_back(u * u * cx + 2 * u * t * x1 + t * t * x2);
                out.pts.push_back(u * u * cy + 2 * u * t * y1 + t * t * y2);
            }
            cx = x2;
            cy = y2;
        } else if (op == 'C') {
            if (!need(6)) return;
            double x1, y1, x2, y2, x3, y3;
            map(m, xy[k], xy[k + 1], x1, y1);
            map(m, xy[k + 2], xy[k + 3], x2, y2);
            map(m, xy[k + 4], xy[k + 5], x3, y3);
            k += 6;
            const double dd = std::max(
                std::hypot(cx - 2 * x1 + x2, cy - 2 * y1 + y2), std::hypot(x1 - 2 * x2 + x3, y1 - 2 * y2 + y3)
            );
            const int n = std::clamp(static_cast<int>(std::ceil(std::sqrt(0.75 * dd / kFlatness))), 1, 256);
            for (int j = 1; j <= n; ++j) {
                const double t = static_cast<double>(j) / n, u = 1.0 - t;
                const double a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
                out.pts.push_back(a * cx + b * x1 + c * x2 + d * x3);
                out.pts.push_back(a * cy + b * y1 + c * y2 + d * y3);
            }
            cx = x3;
            cy = y3;
        } else if (op == 'Z') {
            out.closed.back() = 1;
            const std::size_t s = out.start.back();
            cx = out.pts[2 * s];
            cy = out.pts[2 * s + 1];
        }
    }
}

// Sets the working window to the pixels of [x0, x1] x [y0, y1] inside the canvas and the clip,
// and clears its rows' touched ranges. False when it is empty.
// The rows are also held to [rmin, rmax) (a band of a parallel job).
bool Canvas::begin(
    Window& win, double x0, double y0, double x1, double y1, const Clip* clip, int rmin, int rmax
) const {
    if (clip) {
        x0 = std::max(x0, clip->x0);
        y0 = std::max(y0, clip->y0);
        x1 = std::min(x1, clip->x1);
        y1 = std::min(y1, clip->y1);
    }
    if (!(x0 < x1) || !(y0 < y1)) return false;  // also when a coordinate is NaN
    const double fx0 = std::max(0.0, std::floor(x0)), fy0 = std::max<double>(rmin, std::floor(y0));
    const double fx1 = std::min(static_cast<double>(w_), std::ceil(x1));
    const double fy1 = std::min<double>(rmax, std::ceil(y1));
    if (!(fx0 < fx1) || !(fy0 < fy1)) return false;
    win.bx = static_cast<int>(fx0);
    win.by = static_cast<int>(fy0);
    win.bw = static_cast<int>(fx1) - win.bx;
    win.bh = static_cast<int>(fy1) - win.by;
    const std::size_t need = static_cast<std::size_t>(win.bw + 2) * win.bh;
    if (win.acc.size() < need) win.acc.resize(need);
    win.lo.assign(win.bh, win.bw + 2);
    win.hi.assign(win.bh, 0);
    return true;
}

// Adds the signed area an edge leaves to its right in each pixel of the window (coordinates in
// pixels of the canvas). The row sums of these are the coverage.
void Canvas::edge(Window& win, double x0, double y0, double x1, double y1) {
    if (y0 == y1) return;
    float dir = 1.0f;
    if (y0 > y1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
        dir = -1.0f;
    }
    x0 -= win.bx;
    x1 -= win.bx;
    y0 -= win.by;
    y1 -= win.by;
    if (y1 <= 0.0 || y0 >= win.bh) return;
    const double dxdy = (x1 - x0) / (y1 - y0);
    if (y0 < 0.0) {
        x0 -= y0 * dxdy;
        y0 = 0.0;
    }
    if (y1 > win.bh) y1 = win.bh;
    const double W = win.bw;
    double x = x0;
    const int r0 = static_cast<int>(y0), r1 = static_cast<int>(std::ceil(y1));
    for (int r = r0; r < r1; ++r) {
        const double ya = std::max<double>(r, y0), yb = std::min<double>(r + 1, y1);
        const double dy = yb - ya;
        double xa = x, xb = x + dy * dxdy;
        x = xb;
        if (dy <= 0.0) continue;
        float* row = win.acc.data() + static_cast<std::size_t>(r) * (win.bw + 2);
        // the part left of the window covers every pixel of the row; the part right of it none
        double part = dy;
        if (xa < 0.0 || xb < 0.0 || xa > W || xb > W) {
            double la = xa, lb = xb, ta = 0.0, tb = 1.0;  // the piece inside [0, W] as [ta, tb] of the row's span
            if (la < 0.0 && lb < 0.0) {
                row[0] += dir * static_cast<float>(dy);
                win.lo[r] = 0;
                win.hi[r] = std::max(win.hi[r], 1);
                continue;
            }
            if (la > W && lb > W) continue;
            if (la < 0.0) ta = (0.0 - la) / (lb - la);
            if (lb < 0.0) tb = (0.0 - la) / (lb - la);
            if (la > W) ta = (W - la) / (lb - la);
            if (lb > W) tb = (W - la) / (lb - la);
            const double left = (la < 0.0 ? ta : 0.0) + (lb < 0.0 ? 1.0 - tb : 0.0);
            if (left > 0.0) {
                row[0] += dir * static_cast<float>(dy * left);
                win.lo[r] = 0;
                win.hi[r] = std::max(win.hi[r], 1);
            }
            part = dy * (tb - ta);
            xa = la + ta * (lb - la);
            xb = la + tb * (lb - la);
            xa = std::clamp(xa, 0.0, W);
            xb = std::clamp(xb, 0.0, W);
            if (part <= 0.0) continue;
        }
        const float d = dir * static_cast<float>(part);
        const double xl = std::min(xa, xb), xr = std::max(xa, xb);
        const double fl = std::floor(xl);
        const int i0 = static_cast<int>(fl);
        const int i1 = static_cast<int>(std::ceil(xr));
        win.lo[r] = std::min(win.lo[r], i0);
        win.hi[r] = std::max(win.hi[r], std::max(i1, i0 + 1) + 1);
        if (i1 <= i0 + 1) {
            const float xm = static_cast<float>(0.5 * (xl + xr) - fl);
            row[i0] += d - d * xm;
            row[i0 + 1] += d * xm;
        } else {
            const float s = static_cast<float>(1.0 / (xr - xl));
            const float f0 = static_cast<float>(xl - fl);
            const float a0 = 0.5f * s * (1.0f - f0) * (1.0f - f0);
            const float f1 = static_cast<float>(xr - i1 + 1);
            const float am = 0.5f * s * f1 * f1;
            row[i0] += d * a0;
            if (i1 == i0 + 2) {
                row[i0 + 1] += d * (1.0f - a0 - am);
            } else {
                const float a1 = s * (1.5f - f0);
                row[i0 + 1] += d * (a1 - a0);
                for (int i = i0 + 2; i < i1 - 1; ++i) row[i] += d * s;
                const float a2 = a1 + static_cast<float>(i1 - i0 - 3) * s;
                row[i1 - 1] += d * (1.0f - a2 - am);
            }
            row[i1] += d * am;
        }
    }
}

// Paints the window: the accumulated edge areas become coverage (a running sum along the row,
// its magnitude at most 1) and the colour goes over the pixels. Clears what it read.
void Canvas::composite(Window& win, const float rgba[4], const Clip* clip, bool evenodd) {
    // the coverage of a winding sum: non-zero, or even-odd (a triangle wave of period 2)
    const auto cover = [evenodd](float sum) {
        float c = std::abs(sum);
        if (!evenodd) return std::min(c, 1.0f);
        c = std::fmod(c, 2.0f);
        return c > 1.0f ? 2.0f - c : c;
    };
    const float a = rgba[3];
    const float sr = rgba[0] * a, sg = rgba[1] * a, sb = rgba[2] * a;
    for (int r = 0; r < win.bh; ++r) {
        int lo = win.lo[r], hi = win.hi[r];
        if (lo >= hi) continue;
        float* row = win.acc.data() + static_cast<std::size_t>(r) * (win.bw + 2);
        // left of the first edge nothing is covered; right of the last the sum is back to zero
        float sum = 0.0f;
        const int end = std::min(hi, win.bw + 2);
        for (int i = lo; i < end; ++i) {
            sum += row[i];
            row[i] = cover(sum);
        }
        if (std::abs(sum) > 1e-4f) {  // an edge right of the window was cut off: covered to the end
            const float c = cover(sum);
            for (int i = end; i < win.bw; ++i) row[i] = c;
            hi = win.hi[r] = win.bw + 2;
        }
        hi = std::min(hi, win.bw);
        float cy = 1.0f;
        if (clip) cy = static_cast<float>(overlap(r + win.by, r + win.by + 1.0, clip->y0, clip->y1));
        float* dst = px_.data() + (static_cast<std::size_t>(r + win.by) * w_ + win.bx) * 4;
        if (clip) {
            for (int i = lo; i < hi; ++i) {
                const float cx = static_cast<float>(overlap(i + win.bx, i + win.bx + 1.0, clip->x0, clip->x1));
                const float c = row[i] * cx * cy, k = 1.0f - c * a;
                dst[4 * i] = sr * c + dst[4 * i] * k;
                dst[4 * i + 1] = sg * c + dst[4 * i + 1] * k;
                dst[4 * i + 2] = sb * c + dst[4 * i + 2] * k;
                dst[4 * i + 3] = a * c + dst[4 * i + 3] * k;
            }
        } else {
            for (int i = lo; i < hi; ++i) {
                const float c = row[i], k = 1.0f - c * a;
                dst[4 * i] = sr * c + dst[4 * i] * k;
                dst[4 * i + 1] = sg * c + dst[4 * i + 1] * k;
                dst[4 * i + 2] = sb * c + dst[4 * i + 2] * k;
                dst[4 * i + 3] = a * c + dst[4 * i + 3] * k;
            }
        }
        std::memset(row + lo, 0, static_cast<std::size_t>(std::min(win.hi[r], win.bw + 2) - lo) * sizeof(float));
    }
}

void Canvas::fill(
    const char* ops,
    std::size_t nops,
    const double* xy,
    std::size_t nxy,
    const double m[6],
    const float rgba[4],
    const Clip* clip,
    bool evenodd
) {
    Poly p;
    flatten(ops, nops, xy, nxy, m, p);
    if (p.pts.size() < 6) return;
    for (const double v : p.pts) {
        if (!std::isfinite(v)) return;  // a path with a NaN or an infinity paints nothing
    }
    double x0 = p.pts[0], x1 = x0, y0 = p.pts[1], y1 = y0;
    for (std::size_t i = 0; i < p.pts.size(); i += 2) {
        x0 = std::min(x0, p.pts[i]);
        x1 = std::max(x1, p.pts[i]);
        y0 = std::min(y0, p.pts[i + 1]);
        y1 = std::max(y1, p.pts[i + 1]);
    }
    Window& win = win_;
    if (!begin(win, x0, y0, x1, y1, clip, 0, h_)) return;
    std::memset(win.acc.data(), 0, static_cast<std::size_t>(win.bw + 2) * win.bh * sizeof(float));
    for (std::size_t s = 0; s < p.start.size(); ++s) {
        const std::size_t a = p.start[s], b = s + 1 < p.start.size() ? p.start[s + 1] : p.pts.size() / 2;
        for (std::size_t i = a; i + 1 < b; ++i)
            edge(win, p.pts[2 * i], p.pts[2 * i + 1], p.pts[2 * i + 2], p.pts[2 * i + 3]);
        if (b - a >= 2) edge(win, p.pts[2 * b - 2], p.pts[2 * b - 1], p.pts[2 * a], p.pts[2 * a + 1]);  // fills close
    }
    composite(win, rgba, clip, evenodd);
}

namespace {

// The outline of a stroked run of points (no repeated neighbours): closed polygons appended to
// `out` (points) and `starts`. One side of the run is walked forward and the other back, with
// the join at every inner point and a cap at each end; a closed run gives two loops. On the
// outside of a turn the join is a miter (a bevel beyond the limit) or an arc; on the inside the
// two edges meet at their intersection when it lies within both segments, else they pass
// through the point itself. Pieces that then overlap turn the same way, so the fill counts
// them once.
struct Stroker {
    double hw;
    Cap cap;
    Join join;
    std::vector<double>& out;
    std::vector<std::size_t>& starts;

    void add(double x, double y) {
        out.push_back(x);
        out.push_back(y);
    }

    // an arc around (cx, cy) from direction (ax, ay) to (bx, by) (unit vectors), the short way,
    // without its end points
    void arc(double cx, double cy, double ax, double ay, double bx, double by) {
        const double cross = ax * by - ay * bx, dot = ax * bx + ay * by;
        const double angle = std::atan2(cross, dot);
        const double step = 2.0 * std::acos(std::max(0.0, 1.0 - kFlatness / std::max(hw, kFlatness)));
        const int n = std::clamp(static_cast<int>(std::ceil(std::abs(angle) / std::max(step, 1e-3))), 1, 256);
        for (int i = 1; i < n; ++i) {
            const double t = angle * i / n, c = std::cos(t), sn = std::sin(t);
            add(cx + hw * (ax * c - ay * sn), cy + hw * (ax * sn + ay * c));
        }
    }

    // the side `s` (+1 or -1) of the joint at p between unit directions u (in, length lu) and v (out, length lv)
    void corner(double px, double py, double ux, double uy, double lu, double vx, double vy, double lv, double s) {
        const double nux = s * uy, nuy = -s * ux, nvx = s * vy, nvy = -s * vx;  // the side's normals
        const double cross = ux * vy - uy * vx, dot = ux * vx + uy * vy;
        const double half = 1.0 + dot;  // 2 cos^2 of half the turn
        if (std::abs(cross) < 1e-12 && dot > 0.0) {  // straight on
            add(px + nux * hw, py + nuy * hw);
            return;
        }
        if (s * cross > 0.0) {  // the outside of the turn
            if (join == Join::Round) {
                add(px + nux * hw, py + nuy * hw);
                arc(px, py, nux, nuy, nvx, nvy);
                add(px + nvx * hw, py + nvy * hw);
            } else if (join == Join::Miter && half > 1e-12 && 2.0 / half <= kMiterLimit * kMiterLimit) {
                add(px + (nux + nvx) * hw / half, py + (nuy + nvy) * hw / half);
            } else {
                add(px + nux * hw, py + nuy * hw);
                add(px + nvx * hw, py + nvy * hw);
            }
        } else {  // the inside
            // the edges meet hw tan(turn / 2) back along each segment
            if (half > 1e-12 && hw * std::abs(cross) / half <= std::min(lu, lv)) {
                add(px + (nux + nvx) * hw / half, py + (nuy + nvy) * hw / half);
            } else {
                add(px + nux * hw, py + nuy * hw);
                add(px, py);
                add(px + nvx * hw, py + nvy * hw);
            }
        }
    }

    // the cap at p for a run leaving p in unit direction (dx, dy): from the side -1 to the side +1
    void end(double px, double py, double dx, double dy) {
        const double nx = dy, ny = -dx;  // the +1 side's normal
        if (cap == Cap::Square) {
            add(px - nx * hw - dx * hw, py - ny * hw - dy * hw);
            add(px + nx * hw - dx * hw, py + ny * hw - dy * hw);
        } else if (cap == Cap::Round) {
            arc(px, py, -nx, -ny, -dx, -dy);
            add(px - dx * hw, py - dy * hw);
            arc(px, py, -dx, -dy, nx, ny);
        }
    }

    void run(const std::vector<double>& q, bool closed) {
        const std::size_t n = q.size() / 2;
        if (n == 0) return;
        if (n == 1) {  // a point: its caps alone
            if (cap == Cap::Butt) return;
            starts.push_back(out.size() / 2);
            end(q[0], q[1], 1.0, 0.0);
            add(q[0] + hw * 0.0, q[1] - hw);  // the +1 side of direction (1, 0) is (0, -1)
            end(q[0], q[1], -1.0, 0.0);
            add(q[0], q[1] + hw);
            return;
        }
        const std::size_t nseg = closed ? n : n - 1;
        std::vector<double> dx(nseg), dy(nseg), len(nseg);
        for (std::size_t i = 0; i < nseg; ++i) {
            const std::size_t j = (i + 1) % n;
            const double ex = q[2 * j] - q[2 * i], ey = q[2 * j + 1] - q[2 * i + 1];
            len[i] = std::hypot(ex, ey);
            dx[i] = ex / len[i];
            dy[i] = ey / len[i];
        }
        if (closed) {
            for (const double s : {1.0, -1.0}) {
                starts.push_back(out.size() / 2);
                for (std::size_t k = 0; k < n; ++k) {
                    const std::size_t i = s > 0 ? k : n - 1 - k;  // the other side runs backwards
                    const std::size_t a = (i + n - 1) % n;
                    if (s > 0) corner(q[2 * i], q[2 * i + 1], dx[a], dy[a], len[a], dx[i], dy[i], len[i], 1.0);
                    else corner(q[2 * i], q[2 * i + 1], -dx[i], -dy[i], len[i], -dx[a], -dy[a], len[a], 1.0);
                }
            }
            return;
        }
        starts.push_back(out.size() / 2);
        end(q[0], q[1], dx[0], dy[0]);
        add(q[0] + dy[0] * hw, q[1] - dx[0] * hw);
        for (std::size_t i = 1; i + 1 < n; ++i) {
            corner(q[2 * i], q[2 * i + 1], dx[i - 1], dy[i - 1], len[i - 1], dx[i], dy[i], len[i], 1.0);
        }
        const std::size_t e = n - 1, g = nseg - 1;
        add(q[2 * e] + dy[g] * hw, q[2 * e + 1] - dx[g] * hw);
        end(q[2 * e], q[2 * e + 1], -dx[g], -dy[g]);
        add(q[2 * e] - dy[g] * hw, q[2 * e + 1] + dx[g] * hw);
        for (std::size_t i = n - 2; i >= 1; --i) {  // back along the other side
            corner(q[2 * i], q[2 * i + 1], -dx[i], -dy[i], len[i], -dx[i - 1], -dy[i - 1], len[i - 1], 1.0);
        }
        add(q[0] - dy[0] * hw, q[1] + dx[0] * hw);
    }
};

}  // namespace

void Canvas::stroke(
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
) {
    if (!(width > 0.0)) return;
    Poly p;
    flatten(ops, nops, xy, nxy, m, p);
    if (p.pts.size() < 2) return;
    for (const double v : p.pts) {
        if (!std::isfinite(v)) return;  // a path with a NaN or an infinity paints nothing
    }
    std::vector<double> outline;
    std::vector<std::size_t> starts;
    Stroker st{0.5 * width, cap, join, outline, starts};

    std::vector<double> run, q;
    auto draw = [&](bool closed) {
        q.clear();
        for (std::size_t i = 0; i < run.size(); i += 2) {  // drop repeated points
            if (q.empty() || run[i] != q[q.size() - 2] || run[i + 1] != q[q.size() - 1]) {
                q.push_back(run[i]);
                q.push_back(run[i + 1]);
            }
        }
        if (closed && q.size() >= 4 && q[0] == q[q.size() - 2] && q[1] == q[q.size() - 1]) q.resize(q.size() - 2);
        st.run(q, closed && q.size() >= 6);
    };

    double total = 0.0;
    for (std::size_t i = 0; i < ndash; ++i) total += dash[i];
    const bool dashed = ndash > 0 && total > 0.0;
    for (std::size_t s = 0; s < p.start.size(); ++s) {
        const std::size_t a = p.start[s], b = s + 1 < p.start.size() ? p.start[s + 1] : p.pts.size() / 2;
        if (!dashed) {
            run.assign(p.pts.begin() + 2 * a, p.pts.begin() + 2 * b);
            draw(p.closed[s] != 0);
            continue;
        }
        // walk the subpath (closing it if closed) and cut it at the dash lengths
        std::vector<double> pts(p.pts.begin() + 2 * a, p.pts.begin() + 2 * b);
        if (p.closed[s]) {
            pts.push_back(pts[0]);
            pts.push_back(pts[1]);
        }
        std::size_t di = 0;
        double left = dash[0];
        bool on = true;
        run.clear();
        run.push_back(pts[0]);
        run.push_back(pts[1]);
        for (std::size_t i = 0; i + 3 < pts.size(); i += 2) {
            double px = pts[i], py = pts[i + 1];
            const double qx = pts[i + 2], qy = pts[i + 3];
            double seg = std::hypot(qx - px, qy - py);
            while (seg > left) {
                const double t = left / seg;
                px += t * (qx - px);
                py += t * (qy - py);
                seg -= left;
                if (on) {
                    run.push_back(px);
                    run.push_back(py);
                    draw(false);
                }
                run.clear();
                on = !on;
                di = (di + 1) % ndash;
                left = dash[di];
                if (on) {
                    run.push_back(px);
                    run.push_back(py);
                }
                if (!(left > 0.0)) left = 1e-9;
            }
            left -= seg;
            if (on) {
                run.push_back(qx);
                run.push_back(qy);
            }
        }
        if (on && run.size() >= 2) draw(false);
    }
    if (outline.size() < 6) return;
    double x0 = outline[0], x1 = x0, y0 = outline[1], y1 = y0;
    for (std::size_t i = 0; i < outline.size(); i += 2) {
        x0 = std::min(x0, outline[i]);
        x1 = std::max(x1, outline[i]);
        y0 = std::min(y0, outline[i + 1]);
        y1 = std::max(y1, outline[i + 1]);
    }
    Window& win = win_;
    if (!begin(win, x0, y0, x1, y1, clip, 0, h_)) return;
    std::memset(win.acc.data(), 0, static_cast<std::size_t>(win.bw + 2) * win.bh * sizeof(float));
    for (std::size_t s = 0; s < starts.size(); ++s) {
        const std::size_t a = starts[s], b = s + 1 < starts.size() ? starts[s + 1] : outline.size() / 2;
        for (std::size_t i = a; i + 1 < b; ++i) {
            edge(win, outline[2 * i], outline[2 * i + 1], outline[2 * i + 2], outline[2 * i + 3]);
        }
        if (b - a >= 2) edge(win, outline[2 * b - 2], outline[2 * b - 1], outline[2 * a], outline[2 * a + 1]);
    }
    composite(win, rgba, clip);
}

void Canvas::markers(
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
) {
    if (n == 0 || (!fill && !stroke)) return;
    // The shape's outlines once, around the origin in pixels: every stamp is a translation of them.
    const double m0[6] = {m[0], m[1], m[2], m[3], 0.0, 0.0};
    Poly shape;
    flatten(ops, nops, xy, nxy, m0, shape);
    if (shape.pts.size() < 2) return;
    for (const double v : shape.pts) {
        if (!std::isfinite(v)) return;
    }
    std::vector<double> ring;  // the stroke's outline
    std::vector<std::size_t> ring_start;
    if (stroke && width > 0.0) {
        Stroker st{0.5 * width, Cap::Butt, Join::Round, ring, ring_start};
        std::vector<double> q;
        for (std::size_t s = 0; s < shape.start.size(); ++s) {
            const std::size_t a = shape.start[s],
                              b = s + 1 < shape.start.size() ? shape.start[s + 1] : shape.pts.size() / 2;
            q.clear();
            for (std::size_t i = a; i < b; ++i) {
                if (q.empty() || shape.pts[2 * i] != q[q.size() - 2] || shape.pts[2 * i + 1] != q[q.size() - 1]) {
                    q.push_back(shape.pts[2 * i]);
                    q.push_back(shape.pts[2 * i + 1]);
                }
            }
            const bool closed = shape.closed[s] != 0;
            if (closed && q.size() >= 4 && q[0] == q[q.size() - 2] && q[1] == q[q.size() - 1]) q.resize(q.size() - 2);
            st.run(q, closed && q.size() >= 6);
        }
    }
    double r = 0.0;  // how far a stamp reaches from its point
    for (const double v : shape.pts) r = std::max(r, std::abs(v));
    for (const double v : ring) r = std::max(r, std::abs(v));
    r += 1.0;

    // Stamps are painted in their order within each band of rows, so the picture does not depend
    // on how the bands are shared out.
    constexpr int kBand = 64;
    const std::int64_t tasks = (h_ + kBand - 1) / kBand;
    std::vector<std::vector<std::uint32_t>> in_band(static_cast<std::size_t>(tasks));
    std::vector<double> px(n), py(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double ax = at[2 * i], ay = at[2 * i + 1];
        px[i] = m[0] * ax + m[2] * ay + m[4];
        py[i] = m[1] * ax + m[3] * ay + m[5];
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || px[i] < -r || px[i] > w_ + r || py[i] < -r ||
            py[i] > h_ + r) {
            continue;
        }
        const int b0 = std::max(0, static_cast<int>(std::floor(py[i] - r)) / kBand);
        const int b1 = std::min(static_cast<int>(tasks) - 1, static_cast<int>(std::floor(py[i] + r)) / kBand);
        for (int b = b0; b <= b1; ++b) in_band[b].push_back(static_cast<std::uint32_t>(i));
    }
    auto paint = [&](Window& win, const std::vector<double>& pts, const std::vector<std::size_t>& starts, double ox,
                     double oy, const float* rgba, int rmin, int rmax) {
        double x0 = pts[0], x1 = x0, y0 = pts[1], y1 = y0;
        for (std::size_t i = 0; i < pts.size(); i += 2) {
            x0 = std::min(x0, pts[i]);
            x1 = std::max(x1, pts[i]);
            y0 = std::min(y0, pts[i + 1]);
            y1 = std::max(y1, pts[i + 1]);
        }
        if (!begin(win, x0 + ox, y0 + oy, x1 + ox, y1 + oy, clip, rmin, rmax)) return;
        std::memset(win.acc.data(), 0, static_cast<std::size_t>(win.bw + 2) * win.bh * sizeof(float));
        for (std::size_t s = 0; s < starts.size(); ++s) {
            const std::size_t a = starts[s], b = s + 1 < starts.size() ? starts[s + 1] : pts.size() / 2;
            for (std::size_t i = a; i + 1 < b; ++i) {
                edge(win, pts[2 * i] + ox, pts[2 * i + 1] + oy, pts[2 * i + 2] + ox, pts[2 * i + 3] + oy);
            }
            if (b - a >= 2) edge(win, pts[2 * b - 2] + ox, pts[2 * b - 1] + oy, pts[2 * a] + ox, pts[2 * a + 1] + oy);
        }
        composite(win, rgba, clip);
    };
    auto band = [&](std::int64_t task) {
        Window win;
        const int rmin = static_cast<int>(task) * kBand, rmax = std::min(h_, rmin + kBand);
        for (const std::uint32_t i : in_band[static_cast<std::size_t>(task)]) {
            if (fill && shape.pts.size() >= 6) paint(win, shape.pts, shape.start, px[i], py[i], fill, rmin, rmax);
            if (!ring.empty()) paint(win, ring, ring_start, px[i], py[i], stroke, rmin, rmax);
        }
    };
    if (n >= 2000 && tasks > 1) {
        sd::parallel::parallel_for(tasks, band);
    } else {
        for (std::int64_t t = 0; t < tasks; ++t) band(t);
    }
}

void Canvas::image(
    const std::uint8_t* rgba, int iw, int ih, double x, double y, double w, double h, bool smooth, const Clip* clip
) {
    if (iw < 1 || ih < 1 || !(w > 0.0) || !(h > 0.0) || !std::isfinite(x) || !std::isfinite(y)) return;
    double x0 = x, y0 = y, x1 = x + w, y1 = y + h;
    if (clip) {
        x0 = std::max(x0, clip->x0);
        y0 = std::max(y0, clip->y0);
        x1 = std::min(x1, clip->x1);
        y1 = std::min(y1, clip->y1);
    }
    const int c0 = std::max(0, static_cast<int>(std::floor(x0))), c1 = std::min(w_, static_cast<int>(std::ceil(x1)));
    const int r0 = std::max(0, static_cast<int>(std::floor(y0))), r1 = std::min(h_, static_cast<int>(std::ceil(y1)));
    if (c0 >= c1 || r0 >= r1) return;
    const double kx = iw / w, ky = ih / h;
    constexpr int kBand = 32;  // rows a task: fixed, so the tasks do not depend on the thread count
    const std::int64_t tasks = (r1 - r0 + kBand - 1) / kBand;
    auto rows = [&](std::int64_t task) {
        const int ra = r0 + static_cast<int>(task) * kBand, rb = std::min(r1, ra + kBand);
        for (int r = ra; r < rb; ++r) {
            const float cy = static_cast<float>(overlap(r, r + 1.0, y0, y1));
            const double sy = (r + 0.5 - y) * ky;  // in image pixels
            float* dst = px_.data() + static_cast<std::size_t>(r) * w_ * 4;
            for (int c = c0; c < c1; ++c) {
                const float cov = cy * static_cast<float>(overlap(c, c + 1.0, x0, x1));
                const double sx = (c + 0.5 - x) * kx;
                float s[4];
                if (smooth) {
                    const double fx = std::clamp(sx - 0.5, 0.0, iw - 1.0), fy = std::clamp(sy - 0.5, 0.0, ih - 1.0);
                    const int ix = std::min(static_cast<int>(fx), iw - 2 < 0 ? 0 : iw - 2);
                    const int iy = std::min(static_cast<int>(fy), ih - 2 < 0 ? 0 : ih - 2);
                    const float tx = static_cast<float>(fx - ix), ty = static_cast<float>(fy - iy);
                    const int jx = std::min(ix + 1, iw - 1), jy = std::min(iy + 1, ih - 1);
                    const std::uint8_t* p[4] = {
                        rgba + (static_cast<std::size_t>(iy) * iw + ix) * 4,
                        rgba + (static_cast<std::size_t>(iy) * iw + jx) * 4,
                        rgba + (static_cast<std::size_t>(jy) * iw + ix) * 4,
                        rgba + (static_cast<std::size_t>(jy) * iw + jx) * 4
                    };
                    const float wgt[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};
                    s[0] = s[1] = s[2] = s[3] = 0.0f;
                    for (int q = 0; q < 4; ++q) {  // premultiplied, so a transparent neighbour adds no colour
                        const float a = p[q][3] * (wgt[q] / 255.0f);
                        s[0] += p[q][0] * (a / 255.0f);
                        s[1] += p[q][1] * (a / 255.0f);
                        s[2] += p[q][2] * (a / 255.0f);
                        s[3] += a;
                    }
                } else {
                    const int ix = std::clamp(static_cast<int>(std::floor(sx)), 0, iw - 1);
                    const int iy = std::clamp(static_cast<int>(std::floor(sy)), 0, ih - 1);
                    const std::uint8_t* p = rgba + (static_cast<std::size_t>(iy) * iw + ix) * 4;
                    const float a = p[3] / 255.0f;
                    s[0] = p[0] * (a / 255.0f);
                    s[1] = p[1] * (a / 255.0f);
                    s[2] = p[2] * (a / 255.0f);
                    s[3] = a;
                }
                const float k = 1.0f - s[3] * cov;
                dst[4 * c] = s[0] * cov + dst[4 * c] * k;
                dst[4 * c + 1] = s[1] * cov + dst[4 * c + 1] * k;
                dst[4 * c + 2] = s[2] * cov + dst[4 * c + 2] * k;
                dst[4 * c + 3] = s[3] * cov + dst[4 * c + 3] * k;
            }
        }
    };
    if (static_cast<std::int64_t>(r1 - r0) * (c1 - c0) >= 65536 && tasks > 1) {
        sd::parallel::parallel_for(tasks, rows);
    } else {
        for (std::int64_t t = 0; t < tasks; ++t) rows(t);
    }
}

std::vector<double> decimate(const double* xy, std::size_t n, double column) {
    std::vector<double> out;
    if (n == 0) return out;
    out.reserve(std::min<std::size_t>(2 * n, 1 << 16));
    std::size_t i = 0;
    while (i < n) {
        const double col = std::floor(xy[2 * i] / column);
        std::size_t j = i + 1, lo = i, hi = i;
        while (j < n && std::floor(xy[2 * j] / column) == col) {
            if (xy[2 * j + 1] < xy[2 * lo + 1]) lo = j;
            if (xy[2 * j + 1] > xy[2 * hi + 1]) hi = j;
            ++j;
        }
        // the first, the lowest, the highest and the last point of the run, in their order
        const std::size_t keep[4] = {i, std::min(lo, hi), std::max(lo, hi), j - 1};
        std::size_t last = n;
        for (const std::size_t k : keep) {
            if (k != last) {
                out.push_back(xy[2 * k]);
                out.push_back(xy[2 * k + 1]);
                last = k;
            }
        }
        i = j;
    }
    return out;
}

void colormap(
    const double* v, std::size_t n, double vmin, double vmax, const std::uint8_t* table, int entries, std::uint8_t* out
) {
    const double scale = vmax > vmin ? (entries - 1) / (vmax - vmin) : 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = v[i];
        if (std::isnan(x)) {
            out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = out[4 * i + 3] = 0;
            continue;
        }
        const double t =
            std::clamp((x - vmin) * scale, 0.0, static_cast<double>(entries - 1));  // an infinity takes an end
        std::memcpy(out + 4 * i, table + 4 * static_cast<std::size_t>(t + 0.5), 4);
    }
}

void phase_colors(const double* re, const double* im, std::size_t n, bool bands, std::uint8_t* out) {
    constexpr double kTwoPi = 6.283185307179586;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = re[i], b = im[i];
        std::uint8_t* o = out + 4 * i;
        if (std::isnan(a) || std::isnan(b)) {
            o[0] = o[1] = o[2] = o[3] = 0;
            continue;
        }
        o[3] = 255;
        const double mod = std::hypot(a, b);
        if (mod == 0.0) {
            o[0] = o[1] = o[2] = 0;
            continue;
        }
        if (std::isinf(mod)) {
            o[0] = o[1] = o[2] = 255;
            continue;
        }
        double hue = std::atan2(b, a) / kTwoPi;  // in (-1/2, 1/2]; a positive real is red
        if (hue < 0.0) hue += 1.0;
        double val = 1.0;
        if (bands) {
            const double l = std::log2(mod);
            val = 0.62 + 0.38 * (l - std::floor(l));
        }
        const double sat = 0.85;
        const double h6 = hue * 6.0;
        const int sector = static_cast<int>(h6) % 6;
        const double f = h6 - std::floor(h6);
        const double p = val * (1.0 - sat), q = val * (1.0 - sat * f), t = val * (1.0 - sat * (1.0 - f));
        double r, g, bl;
        switch (sector) {
            case 0:
                r = val;
                g = t;
                bl = p;
                break;
            case 1:
                r = q;
                g = val;
                bl = p;
                break;
            case 2:
                r = p;
                g = val;
                bl = t;
                break;
            case 3:
                r = p;
                g = q;
                bl = val;
                break;
            case 4:
                r = t;
                g = p;
                bl = val;
                break;
            default:
                r = val;
                g = p;
                bl = q;
                break;
        }
        o[0] = static_cast<std::uint8_t>(r * 255.0 + 0.5);
        o[1] = static_cast<std::uint8_t>(g * 255.0 + 0.5);
        o[2] = static_cast<std::uint8_t>(bl * 255.0 + 0.5);
    }
}

std::vector<std::uint8_t> Canvas::rgba8() const {
    const std::size_t n = static_cast<std::size_t>(w_) * h_;
    std::vector<std::uint8_t> out(n * 4);
    const float* p = px_.data();
    for (std::size_t i = 0; i < n; ++i) {
        const float a = p[4 * i + 3];
        if (a <= 0.0f) continue;  // out is zero-initialized
        const float k = 255.0f / a;
        out[4 * i] = static_cast<std::uint8_t>(std::min(255.0f, p[4 * i] * k + 0.5f));
        out[4 * i + 1] = static_cast<std::uint8_t>(std::min(255.0f, p[4 * i + 1] * k + 0.5f));
        out[4 * i + 2] = static_cast<std::uint8_t>(std::min(255.0f, p[4 * i + 2] * k + 0.5f));
        out[4 * i + 3] = static_cast<std::uint8_t>(std::min(255.0f, a * 255.0f + 0.5f));
    }
    return out;
}

namespace {

// v with `digits` decimals at most, trailing zeros dropped, appended to out
void put_number(std::string& out, double v, int digits) {
    static const double kPow[] = {1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9};
    if (!(std::fabs(v) < 1e9) || digits < 0 || digits > 9) {  // far out, or not a number: the plain way
        char buf[64];
        const int n = std::snprintf(buf, sizeof buf, "%.*f", digits < 0 ? 0 : digits, v);
        out.append(buf, static_cast<std::size_t>(n > 0 ? n : 0));
        return;
    }
    long long q = std::llround(std::fabs(v) * kPow[digits]);
    if (q == 0) {
        out += '0';
        return;
    }
    if (v < 0.0) out += '-';
    const long long unit = static_cast<long long>(kPow[digits]);
    const long long whole = q / unit;
    long long frac = q % unit;
    char buf[32];
    int n = 0;
    long long w = whole;
    do {
        buf[n++] = static_cast<char>('0' + w % 10);
        w /= 10;
    } while (w > 0);
    while (n > 0) out += buf[--n];
    if (frac == 0) return;
    int len = digits;
    while (frac % 10 == 0) {
        frac /= 10;
        --len;
    }
    out += '.';
    char fb[16];
    for (int i = len - 1; i >= 0; --i) {
        fb[i] = static_cast<char>('0' + frac % 10);
        frac /= 10;
    }
    out.append(fb, static_cast<std::size_t>(len));
}

}  // namespace

std::string path_text(const char* ops, std::size_t nops, const double* xy, int digits, bool pdf) {
    std::string out;
    out.reserve(nops * 14);
    std::size_t i = 0;
    double x = 0.0, y = 0.0;
    const auto run = [&](const double* v, int count) {
        for (int k = 0; k < count; ++k) {
            if (k) out += ' ';
            put_number(out, v[k], digits);
        }
    };
    for (std::size_t o = 0; o < nops; ++o) {
        const char op = ops[o];
        const int n = op == 'M' || op == 'L' ? 2 : op == 'Q' ? 4 : op == 'C' ? 6 : 0;
        if (!pdf) {
            out += op;
            run(xy + i, n);
        } else {
            if (o) out += '\n';
            if (op == 'M' || op == 'L') {
                run(xy + i, 2);
                out += op == 'M' ? " m" : " l";
                x = xy[i];
                y = xy[i + 1];
            } else if (op == 'C') {
                run(xy + i, 6);
                out += " c";
                x = xy[i + 4];
                y = xy[i + 5];
            } else if (op == 'Q') {
                const double x1 = xy[i], y1 = xy[i + 1], x2 = xy[i + 2], y2 = xy[i + 3];
                const double cb[6] = {
                    x + 2 * (x1 - x) / 3, y + 2 * (y1 - y) / 3, x2 + 2 * (x1 - x2) / 3, y2 + 2 * (y1 - y2) / 3, x2, y2
                };
                run(cb, 6);
                out += " c";
                x = x2;
                y = y2;
            } else {
                out += 'h';
            }
        }
        i += static_cast<std::size_t>(n);
    }
    return out;
}

std::vector<double> simplify(const double* xy, std::size_t n, double tol) {
    if (n < 3) return std::vector<double>(xy, xy + 2 * n);
    std::vector<char> keep(n, 0);
    keep[0] = keep[n - 1] = 1;
    const double t2 = tol * tol;
    std::vector<std::pair<std::size_t, std::size_t>> todo{{0, n - 1}};
    while (!todo.empty()) {
        const auto [a, b] = todo.back();
        todo.pop_back();
        if (b <= a + 1) continue;
        const double ax = xy[2 * a], ay = xy[2 * a + 1], dx = xy[2 * b] - ax, dy = xy[2 * b + 1] - ay;
        const double len2 = dx * dx + dy * dy;
        double far = -1.0;
        std::size_t at = a;
        for (std::size_t i = a + 1; i < b; ++i) {
            const double px = xy[2 * i] - ax, py = xy[2 * i + 1] - ay;
            double d2;
            if (len2 > 0.0) {
                const double cr = px * dy - py * dx;
                d2 = cr * cr / len2;  // to the line through a and b
            } else {
                d2 = px * px + py * py;  // a and b at one place (a closed run): to that place
            }
            if (d2 > far) {
                far = d2;
                at = i;
            }
        }
        if (far > t2) {
            keep[at] = 1;
            todo.push_back({a, at});
            todo.push_back({at, b});
        }
    }
    std::vector<double> out;
    for (std::size_t i = 0; i < n; ++i) {
        if (keep[i]) out.insert(out.end(), {xy[2 * i], xy[2 * i + 1]});
    }
    return out;
}

}  // namespace sd::plot
