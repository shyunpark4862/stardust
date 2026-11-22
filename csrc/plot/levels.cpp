// Level curves of a function sampled on a grid (levels.hpp has the method).
#include "plot/levels.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sd::plot {

namespace {

struct Edge {
    std::int64_t key;
    std::size_t ja, ia, jb, ib;
};

// The four edges of a cell: bottom, right, top, left, each from the corner named first. An
// edge's key: horizontal edges (j, i) -> (j, i + 1) are even, vertical (j, i) -> (j + 1, i) odd.
inline void cell_edges(std::size_t nx, std::size_t j, std::size_t i, Edge e[4]) {
    const auto key = [nx](std::size_t jj, std::size_t ii, int odd) {
        return 2 * static_cast<std::int64_t>(jj * (nx + 1) + ii) + odd;
    };
    e[0] = {key(j, i, 0), j, i, j, i + 1};
    e[1] = {key(j, i + 1, 1), j, i + 1, j + 1, i + 1};
    e[2] = {key(j + 1, i, 0), j + 1, i, j + 1, i + 1};
    e[3] = {key(j, i, 1), j, i, j + 1, i};
}

}  // namespace

Levels::Levels(std::vector<double> F, std::vector<double> xs, std::vector<double> ys)
    : F_(std::move(F)), xs_(std::move(xs)), ys_(std::move(ys)) {
    if (xs_.size() < 2 || ys_.size() < 2 || F_.size() != xs_.size() * ys_.size()) {
        throw std::invalid_argument("the grid must hold len(ys) rows of len(xs) values, two or more each way");
    }
    ny_ = ys_.size() - 1;
    nx_ = xs_.size() - 1;
}

void Levels::prepare(double level) {
    if (ready_ && level == level_) return;
    level_ = level;
    ready_ = true;
    cells_.clear();
    index_.clear();
    edges_.clear();
    const std::size_t w = nx_ + 1;
    for (std::size_t j = 0; j < ny_; ++j) {
        const double* r0 = F_.data() + j * w;
        const double* r1 = r0 + w;
        for (std::size_t i = 0; i < nx_; ++i) {
            const double a = r0[i], b = r0[i + 1], c = r1[i + 1], d = r1[i];
            const int in = (a > level) + (b > level) + (c > level) + (d > level);
            if (in == 0 || in == 4) continue;
            if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(c) && std::isfinite(d))) continue;
            cells_.push_back(static_cast<std::uint32_t>(j * nx_ + i));
            Edge e[4];
            cell_edges(nx_, j, i, e);
            for (const Edge& q : e) {
                const double p = at(q.ja, q.ia), r = at(q.jb, q.ib);
                if ((p > 0.0) == (r > 0.0)) continue;
                if (!index_.emplace(q.key, static_cast<std::int32_t>(index_.size())).second) continue;
                const double t = p / (p - r);
                const double ax = xs_[q.ia], ay = ys_[q.ja], bx = xs_[q.ib], by = ys_[q.jb];
                edges_.insert(edges_.end(), {ax + t * (bx - ax), ay + t * (by - ay), ax, ay, p, bx, by, r});
            }
        }
    }
}

std::vector<double> Levels::edges(double level) {
    prepare(level);
    return edges_;
}

std::vector<double> Levels::curves(double level, const std::vector<double>& points) {
    prepare(level);
    const std::size_t n = index_.size();
    if (points.size() != 2 * n) throw std::invalid_argument("points must hold x, y of every crossed edge");
    std::vector<std::int32_t> link(2 * n, -1);  // up to two neighbours of every edge
    auto join = [&](std::int32_t p, std::int32_t q) {
        for (const auto& [a, b] : {std::pair{p, q}, std::pair{q, p}}) {
            if (link[2 * a] < 0) {
                link[2 * a] = b;
            } else if (link[2 * a + 1] < 0) {
                link[2 * a + 1] = b;
            }
        }
    };
    for (const std::uint32_t cell : cells_) {
        const std::size_t j = cell / nx_, i = cell % nx_;
        Edge e[4];
        cell_edges(nx_, j, i, e);
        std::int32_t id[4] = {-1, -1, -1, -1};
        int found[4], count = 0;
        for (int k = 0; k < 4; ++k) {
            if ((at(e[k].ja, e[k].ia) > 0.0) == (at(e[k].jb, e[k].ib) > 0.0)) continue;
            id[k] = index_.at(e[k].key);
            found[count++] = k;
        }
        if (count == 2) {
            join(id[found[0]], id[found[1]]);
        } else if (count == 4) {  // a saddle: the corners' mean decides which pairs of edges are joined
            const double centre = (at(j, i) + at(j, i + 1) + at(j + 1, i + 1) + at(j + 1, i)) / 4.0;
            if ((centre > 0.0) == (at(j, i) > 0.0)) {
                join(id[0], id[1]);
                join(id[2], id[3]);
            } else {
                join(id[0], id[3]);
                join(id[2], id[1]);
            }
        }
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out;
    out.reserve(2 * n + 64);
    std::vector<char> seen(n, 0);
    auto walk = [&](std::int32_t start) {
        const std::size_t before = out.size();
        if (!out.empty()) out.insert(out.end(), {nan, nan});
        const std::size_t first = out.size();
        std::int32_t prev = -1, cur = start;
        std::size_t count = 0;
        while (true) {
            seen[cur] = 1;
            out.insert(out.end(), {points[2 * cur], points[2 * cur + 1]});
            ++count;
            std::int32_t next = -1;
            for (int k = 0; k < 2; ++k) {
                const std::int32_t c = link[2 * cur + k];
                if (c >= 0 && c != prev && !seen[c]) {
                    next = c;
                    break;
                }
            }
            if (next < 0) break;
            prev = cur;
            cur = next;
        }
        if (count > 2 && (link[2 * cur] == start || link[2 * cur + 1] == start)) {
            const double x = out[first], y = out[first + 1];
            out.insert(out.end(), {x, y});  // closed
        } else if (count < 2) {
            out.resize(before);  // a lone point is no curve
        }
    };
    for (std::size_t k = 0; k < n; ++k) {  // open curves first: they start at an edge with one neighbour
        if (!seen[k] && link[2 * k + 1] < 0) walk(static_cast<std::int32_t>(k));
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (!seen[k]) walk(static_cast<std::int32_t>(k));
    }
    return out;
}

void Levels::inside(double level, const std::vector<double>& points, std::string& ops, std::vector<double>& xy) {
    prepare(level);
    if (points.size() != 2 * index_.size()) throw std::invalid_argument("points must hold x, y of every crossed edge");
    const std::size_t w = nx_ + 1;
    const auto full = [&](std::size_t j, std::size_t i) {
        const double* r0 = F_.data() + j * w;
        const double* r1 = r0 + w;
        // above the level at the four corners (a comparison with a NaN is false; an infinite value is no value)
        return r0[i] > level && r0[i + 1] > level && r1[i + 1] > level && r1[i] > level && std::isfinite(r0[i]) &&
               std::isfinite(r0[i + 1]) && std::isfinite(r1[i + 1]) && std::isfinite(r1[i]);
    };
    for (std::size_t j = 0; j < ny_; ++j) {
        std::size_t i = 0;
        while (i < nx_) {
            if (!full(j, i)) {
                ++i;
                continue;
            }
            std::size_t e = i + 1;
            while (e < nx_ && full(j, e)) ++e;
            ops += "MLLLZ";
            xy.insert(xy.end(), {xs_[i], ys_[j], xs_[e], ys_[j], xs_[e], ys_[j + 1], xs_[i], ys_[j + 1]});
            i = e;
        }
    }
    for (const std::uint32_t cell : cells_) {
        const std::size_t j = cell / nx_, i = cell % nx_;
        Edge e[4];
        cell_edges(nx_, j, i, e);
        // around the cell: a corner, then the crossing of the side that leaves it
        const std::size_t cj[4] = {j, j, j + 1, j + 1}, ci[4] = {i, i + 1, i + 1, i};
        double pts[16];
        int count = 0;
        for (int k = 0; k < 4; ++k) {
            const int nk = (k + 1) % 4;
            const bool s = at(cj[k], ci[k]) > 0.0, t = at(cj[nk], ci[nk]) > 0.0;
            if (s) {
                pts[2 * count] = xs_[ci[k]];
                pts[2 * count + 1] = ys_[cj[k]];
                ++count;
            }
            if (s != t) {
                const std::int32_t id = index_.at(e[k].key);
                pts[2 * count] = points[2 * id];
                pts[2 * count + 1] = points[2 * id + 1];
                ++count;
            }
        }
        // A saddle whose corners' mean is on the other side of its inside corners: as the curves
        // do (curves), the two inside corners are cut off apart, each with its two crossings.
        const bool apart =
            count == 6 && (at(j, i) + at(j, i + 1) + at(j + 1, i + 1) + at(j + 1, i)) / 4.0 > 0.0 != (at(j, i) > 0.0);
        if (apart) {
            // a triangle for each inside corner k: the crossing of the side before it, the corner,
            // the crossing of its own side (side k runs from corner k to corner k + 1)
            for (int k = 0; k < 4; ++k) {
                if (!(at(cj[k], ci[k]) > 0.0)) continue;
                const std::int32_t before = index_.at(e[(k + 3) % 4].key), after = index_.at(e[k].key);
                ops += "MLLZ";
                xy.insert(
                    xy.end(), {points[2 * before], points[2 * before + 1], xs_[ci[k]], ys_[cj[k]], points[2 * after],
                               points[2 * after + 1]}
                );
            }
        } else if (count >= 3) {
            ops += 'M';
            ops.append(static_cast<std::size_t>(count - 1), 'L');
            ops += 'Z';
            xy.insert(xy.end(), pts, pts + 2 * count);
        }
    }
}

}  // namespace sd::plot
