// Euclidean minimum spanning trees (csrc/cluster/mst.hpp; docs/design/cluster.md §3.2, decisions CL17, CL47).
#include "cluster/mst.hpp"

#include "cluster/common.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace sd::cluster {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The vertex v's best edge so far, (d, from): the edge (cur, v) at distance d replaces it when d is smaller or
// equal with a smaller other end (on equal distances the edge of the smaller other end is the smaller edge,
// whichever side of v the ends lie).
inline bool better_to(double d, int64_t cur, double dmin, int64_t from) {
    return d < dmin || (d == dmin && cur < from);
}

// One Prim step's scan of the outside vertices in [v0, v1): offer `cur` to each, and the smallest edge there.
template <class Dist>
Edge prim_scan(
    int64_t v0,
    int64_t v1,
    int64_t cur,
    const Dist& dist,
    std::vector<double>& dmin,
    std::vector<int64_t>& from,
    const std::vector<char>& in,
    int64_t& arg
) {
    Edge best{kInf, -1, -1};
    arg = -1;
    for (int64_t v = v0; v < v1; ++v) {
        const auto vv = static_cast<size_t>(v);
        if (in[vv]) continue;
        dist(cur, v, dmin[vv], from[vv]);
        const Edge e{dmin[vv], std::min(v, from[vv]), std::max(v, from[vv])};
        if (arg < 0 || edge_less(e, best)) best = e, arg = v;
    }
    return best;
}

template <class Dist>
std::vector<Edge> prim(int64_t n, int64_t cost, const Dist& dist) {
    std::vector<Edge> E;
    if (n < 2) return E;
    E.reserve(static_cast<size_t>(n - 1));
    std::vector<double> dmin(static_cast<size_t>(n), kInf);
    std::vector<int64_t> from(static_cast<size_t>(n), -1);
    std::vector<char> in(static_cast<size_t>(n), 0);
    in[0] = 1;
    int64_t cur = 0;
    // tasks of about 2^12 pair-terms, fixed by n and the cost of a pair
    const int64_t task = std::max<int64_t>(64, (int64_t{1} << 12) / std::max<int64_t>(cost, 1));
    const int64_t ntask = (n + task - 1) / task;
    const bool split = n * cost >= kSplitWork && ntask > 1;
    std::vector<Edge> tb(static_cast<size_t>(ntask));
    std::vector<int64_t> ti(static_cast<size_t>(ntask));
    for (int64_t step = 1; step < n; ++step) {
        auto body = [&](int64_t t) {
            int64_t arg;
            tb[static_cast<size_t>(t)] =
                prim_scan(t * task, std::min(n, (t + 1) * task), cur, dist, dmin, from, in, arg);
            ti[static_cast<size_t>(t)] = arg;
        };
        if (split) parallel::parallel_for(ntask, body, task * cost);
        else
            for (int64_t t = 0; t < ntask; ++t) body(t);
        Edge best{kInf, -1, -1};
        int64_t arg = -1;
        for (int64_t t = 0; t < ntask; ++t) {
            const int64_t v = ti[static_cast<size_t>(t)];
            if (v >= 0 && (arg < 0 || edge_less(tb[static_cast<size_t>(t)], best)))
                best = tb[static_cast<size_t>(t)], arg = v;
        }
        in[static_cast<size_t>(arg)] = 1;
        E.push_back(best);
        cur = arg;
    }
    return E;
}

// ---- Boruvka on the KD tree --------------------------------------------------------------------------------------

class Boruvka {
public:
    Boruvka(const spatial::KdTree& t, bool dual)
        : T_(t),
          dual_(dual),
          n_(t.n),
          p_(t.m),
          E_{t.m},
          cut_(t.m, false),
          comp_(static_cast<size_t>(t.n)),
          ncomp_(t.nodes.size()),
          bd_(static_cast<size_t>(t.n)),
          bj_(static_cast<size_t>(t.n)),
          bound_(t.nodes.size()),
          cbest_(static_cast<size_t>(t.n)) {}

    std::vector<Edge> run() {
        std::vector<Edge> out;
        if (n_ < 2) return out;
        out.reserve(static_cast<size_t>(n_ - 1));
        MinUnionFind uf(n_);
        std::vector<int64_t> tasks;
        collect(0, std::max<int64_t>(kLeafSize, n_ / 64), tasks);
        std::vector<Edge> cb(static_cast<size_t>(n_));
        int64_t components = n_;
        while (components > 1) {
            for (int64_t q = 0; q < n_; ++q) comp_[static_cast<size_t>(q)] = uf.find(T_.idx[static_cast<size_t>(q)]);
            for (int64_t id = static_cast<int64_t>(T_.nodes.size()) - 1; id >= 0; --id) {
                const spatial::KdNode& nd = T_.nodes[static_cast<size_t>(id)];
                int64_t c;
                if (nd.right < 0) {
                    c = comp_[static_cast<size_t>(nd.start)];
                    for (int64_t q = nd.start + 1; q < nd.end && c >= 0; ++q)
                        if (comp_[static_cast<size_t>(q)] != c) c = -1;
                } else {
                    const int64_t a = ncomp_[static_cast<size_t>(id + 1)], b = ncomp_[static_cast<size_t>(nd.right)];
                    c = a >= 0 && a == b ? a : -1;
                }
                ncomp_[static_cast<size_t>(id)] = c;
            }
            std::fill(bd_.begin(), bd_.end(), kInf);
            std::fill(bj_.begin(), bj_.end(), int64_t{-1});
            std::fill(bound_.begin(), bound_.end(), kInf);
            for (auto& c : cbest_) c.store(kInf, std::memory_order_relaxed);
            auto body = [&](int64_t t) {
                if (!dual_) {
                    std::vector<int64_t> stack;
                    std::vector<double> clamp(static_cast<size_t>(p_));
                    walk(tasks[static_cast<size_t>(t)], stack, clamp);
                } else {
                    spatial::KdPair pr(p_);
                    visit(tasks[static_cast<size_t>(t)], 0, pr);
                }
            };
            const int64_t nt = static_cast<int64_t>(tasks.size());
            if (n_ >= 2048 && nt > 1) parallel::parallel_for(nt, body, n_ * (p_ + 4) * 30 / nt);
            else
                for (int64_t t = 0; t < nt; ++t) body(t);
            std::fill(cb.begin(), cb.end(), Edge{kInf, -1, -1});
            for (int64_t q = 0; q < n_; ++q) {
                const int64_t j = bj_[static_cast<size_t>(q)];
                if (j < 0) continue;
                const int64_t i = T_.idx[static_cast<size_t>(q)];
                const Edge e{bd_[static_cast<size_t>(q)], std::min(i, j), std::max(i, j)};
                Edge& c = cb[static_cast<size_t>(comp_[static_cast<size_t>(q)])];
                if (c.a < 0 || edge_less(e, c)) c = e;
            }
            for (int64_t r = 0; r < n_; ++r) {
                const Edge& e = cb[static_cast<size_t>(r)];
                if (e.a < 0) continue;
                const int64_t a = uf.find(e.a), b = uf.find(e.b);
                if (a == b) continue;
                uf.unite(a, b);
                out.push_back(e);
                --components;
            }
        }
        return out;
    }

private:
    const spatial::KdTree& T_;
    bool dual_;
    int64_t n_, p_;
    spatial::KdEuclid E_;
    spatial::KdCut cut_;
    std::vector<int64_t> comp_, ncomp_;  // per leaf position, per node (-1: mixed)
    std::vector<double> bd_;             // per leaf position: the smallest distance to another component
    std::vector<int64_t> bj_;            // and its point's key
    std::vector<double> bound_;          // per node: at least every point's radius
    // per component: the smallest distance any of its points has found so far. It only prunes: a point need not
    // look past its component's best (the component keeps its smallest edge), and the cuts are strict, so an
    // edge at exactly that distance is still seen. Tasks lower it concurrently; the edges chosen do not depend
    // on when they do.
    std::vector<std::atomic<double>> cbest_;

    void lower(int64_t c, double d) {
        std::atomic<double>& a = cbest_[static_cast<size_t>(c)];
        double cur = a.load(std::memory_order_relaxed);
        while (d < cur && !a.compare_exchange_weak(cur, d, std::memory_order_relaxed)) {}
    }

    int64_t size(int64_t id) const {
        const spatial::KdNode& nd = T_.nodes[static_cast<size_t>(id)];
        return nd.end - nd.start;
    }

    void collect(int64_t id, int64_t limit, std::vector<int64_t>& tasks) const {
        const spatial::KdNode& nd = T_.nodes[static_cast<size_t>(id)];
        if (nd.right < 0 || nd.end - nd.start <= limit) {
            tasks.push_back(id);
            return;
        }
        collect(id + 1, limit, tasks);
        collect(nd.right, limit, tasks);
    }

    // Whether every pair (x, y) of a node pair whose nearest corners are a, b has a distance above T1.
    bool beyond(const double* a, const double* b, double T1) const {
        const double s = E_.raw(a, b);
        if (spatial::KdEuclid::plain(s)) return s > spatial::KdEuclid::above(T1);
        return E_.finish(s, a, b) > T1;
    }

    double box_sum(int64_t Q, int64_t R, spatial::KdPair& pr) const {
        pr.near(T_.lo(Q), T_.hi(Q), T_.lo(R), T_.hi(R), p_);
        return E_.raw(pr.a.data(), pr.b.data());
    }

    void leaves(int64_t Q, int64_t R) {
        const spatial::KdNode& nq = T_.nodes[static_cast<size_t>(Q)];
        const spatial::KdNode& nr = T_.nodes[static_cast<size_t>(R)];
        double top = 0.0;
        for (int64_t q = nq.start; q < nq.end; ++q) {
            const auto qq = static_cast<size_t>(q);
            const int64_t cq = comp_[qq];
            const int64_t iq = T_.idx[qq];
            const double* x = T_.data.data() + q * p_;
            double b = bd_[qq];
            int64_t bj = bj_[qq];
            const double lim0 = std::min(b, cbest_[static_cast<size_t>(cq)].load(std::memory_order_relaxed));
            double lim = lim0;
            for (int64_t r = nr.start; r < nr.end; ++r) {
                const auto rr = static_cast<size_t>(r);
                if (comp_[rr] == cq) continue;
                const double* y = T_.data.data() + r * p_;
                const double s = E_.raw(x, y);
                if (spatial::KdEuclid::plain(s) && s > spatial::KdEuclid::above(lim)) continue;
                const double d = E_.finish(s, x, y);
                if (d > lim) continue;
                const int64_t j = T_.idx[rr];
                // the edge (d, min(iq, j), max(iq, j)) against the best one: smaller d, then the smaller other end
                if (d < b || (d == b && (bj < 0 || j < bj))) b = d, bj = j, lim = std::min(lim, d);
            }
            bd_[qq] = b;
            bj_[qq] = bj;
            if (b < lim0) lower(cq, b);
            top = std::max(top, std::min(b, cbest_[static_cast<size_t>(cq)].load(std::memory_order_relaxed)));
        }
        // the leaf's radius: the largest best of its points (each pass sees all of them)
        bound_[static_cast<size_t>(Q)] = top;
    }

    // The points of a query leaf one by one: each searches the tree from the root, nearer child first, cutting
    // the nodes of its own component and those beyond its radius (its best, or its component's if smaller).
    void query_leaf(int64_t Q, std::vector<int64_t>& stack, std::vector<double>& clamp) {
        const spatial::KdNode& nq = T_.nodes[static_cast<size_t>(Q)];
        for (int64_t q = nq.start; q < nq.end; ++q) {
            const auto qq = static_cast<size_t>(q);
            const int64_t cq = comp_[qq];
            const int64_t iq = T_.idx[qq];
            const double* x = T_.data.data() + q * p_;
            double b = bd_[qq];
            int64_t bj = bj_[qq];
            auto radius = [&] { return std::min(b, cbest_[static_cast<size_t>(cq)].load(std::memory_order_relaxed)); };
            const double r0 = radius();
            auto bound_sum = [&](int64_t id) {
                const double* lo = T_.lo(id);
                const double* hi = T_.hi(id);
                for (int64_t d = 0; d < p_; ++d) clamp[static_cast<size_t>(d)] = std::min(std::max(x[d], lo[d]), hi[d]);
                return E_.raw(x, clamp.data());
            };
            stack.assign(1, 0);
            while (!stack.empty()) {
                const int64_t id = stack.back();
                stack.pop_back();
                if (ncomp_[static_cast<size_t>(id)] == cq) continue;
                const double T = radius();
                if (T < kInf) {
                    (void)bound_sum(id);
                    if (spatial::kd_beyond(E_, x, clamp.data(), cut_(T))) continue;
                }
                const spatial::KdNode& nd = T_.nodes[static_cast<size_t>(id)];
                if (nd.right >= 0) {
                    const int64_t l = id + 1, r = nd.right;
                    if (bound_sum(r) < bound_sum(l)) stack.push_back(l), stack.push_back(r);
                    else stack.push_back(r), stack.push_back(l);
                    continue;
                }
                for (int64_t r = nd.start; r < nd.end; ++r) {
                    const auto rr = static_cast<size_t>(r);
                    if (comp_[rr] == cq) continue;
                    const double* y = T_.data.data() + r * p_;
                    const double sv = E_.raw(x, y);
                    const double lim = radius();
                    if (spatial::KdEuclid::plain(sv) && sv > spatial::KdEuclid::above(lim)) continue;
                    const double d = E_.finish(sv, x, y);
                    if (d > lim) continue;
                    const int64_t j = T_.idx[rr];
                    if (d < b || (d == b && (bj < 0 || j < bj))) b = d, bj = j;
                }
            }
            bd_[qq] = b;
            bj_[qq] = bj;
            if (b < r0) lower(cq, b);
        }
    }

    void walk(int64_t Q, std::vector<int64_t>& stack, std::vector<double>& clamp) {
        const spatial::KdNode& nd = T_.nodes[static_cast<size_t>(Q)];
        if (nd.right < 0) {
            query_leaf(Q, stack, clamp);
            return;
        }
        walk(Q + 1, stack, clamp);
        walk(nd.right, stack, clamp);
    }

    void visit(int64_t Q, int64_t R, spatial::KdPair& pr) {
        const int64_t cq = ncomp_[static_cast<size_t>(Q)];
        if (cq >= 0 && cq == ncomp_[static_cast<size_t>(R)]) return;
        double radius = bound_[static_cast<size_t>(Q)];
        if (cq >= 0) radius = std::min(radius, cbest_[static_cast<size_t>(cq)].load(std::memory_order_relaxed));
        if (radius < kInf) {
            pr.near(T_.lo(Q), T_.hi(Q), T_.lo(R), T_.hi(R), p_);
            if (beyond(pr.a.data(), pr.b.data(), cut_(radius))) return;
        }
        const spatial::KdNode& nq = T_.nodes[static_cast<size_t>(Q)];
        const spatial::KdNode& nr = T_.nodes[static_cast<size_t>(R)];
        const bool lq = nq.right < 0, lr = nr.right < 0;
        if (lq && lr) {
            leaves(Q, R);
            return;
        }
        // the query node is split first while its radius is unknown (its leaves then get one from their nearest
        // reference leaves), then the larger of the two
        if (!lr && (lq || (radius < kInf && size(R) >= size(Q)))) {
            int64_t r1 = R + 1, r2 = nr.right;
            if (box_sum(Q, r2, pr) < box_sum(Q, r1, pr)) std::swap(r1, r2);
            visit(Q, r1, pr);
            visit(Q, r2, pr);
        } else {
            const int64_t q1 = Q + 1, q2 = nq.right;
            visit(q1, R, pr);
            visit(q2, R, pr);
            bound_[static_cast<size_t>(Q)] = std::max(bound_[static_cast<size_t>(q1)], bound_[static_cast<size_t>(q2)]);
        }
    }
};

}  // namespace

// CL54 (tools/cluster_probe.py proto.mst, tests/cluster/data/prototype.json "mst/time"): Boruvka from where it is
// about as fast as Prim on one thread (and several times faster on ten); never above p = 8.
bool boruvka_shape(int64_t n, int64_t p) {
    return (p <= 3 && n >= 1024) || (p <= 5 && n >= 2048) || (p <= 6 && n >= 4096) || (p <= 8 && n >= 16384);
}

std::vector<Edge> mst_prim(const double* X, int64_t n, int64_t p) {
    const spatial::KdEuclid E{p};
    auto dist = [&](int64_t cur, int64_t v, double& dmin, int64_t& from) {
        const double* x = X + cur * p;
        const double* y = X + v * p;
        const double s = E.raw(x, y);
        if (spatial::KdEuclid::plain(s) && s > spatial::KdEuclid::above(dmin)) return;  // d > dmin: not better
        const double d = E.finish(s, x, y);
        if (better_to(d, cur, dmin, from)) dmin = d, from = cur;
    };
    return prim(n, p + 2, dist);
}

std::vector<Edge> mst_prim_matrix(const double* D, int64_t n) {
    auto dist = [&](int64_t cur, int64_t v, double& dmin, int64_t& from) {
        const double d = D[cpair(n, cur, v)];
        if (better_to(d, cur, dmin, from)) dmin = d, from = cur;
    };
    return prim(n, 2, dist);
}

std::vector<Edge> mst_boruvka(const spatial::KdTree& T, bool dual) { return Boruvka(T, dual).run(); }

}  // namespace sd::cluster
