// Fisher's exact test of an r x c table (milestone H6, docs/design/htest.md §3.7; decision H77):
// the network algorithm of Mehta and Patel (1983, 1986; R's fisher.test, FEXACT), exact in the
// sense of the two-sided rule: the p-value sums the probabilities of every table with the
// observed margins that is at most (1 + 1e-7) times as probable as the observed one (R's
// relErr; the reference tool's rule, tools/htest_ref.py).
//
// A table's probability is K prod_ij 1/x_ij!, K = prod R_i! prod C_j! / N!. The columns are
// filled one at a time; the state after k columns is the multiset of the remaining row sums
// (rows are exchangeable for what is left, so the state is kept sorted: a node). A path from the
// root to a node carries its past value prod 1/x! over the filled columns; the paths into one
// node with the same past value (within 1e-24 of its log, double-double logs of the same
// rational) are merged, their multiplicities added. For each node the longest and shortest
// completions (the largest and smallest prod 1/x! over the rest of the table) are computed
// exactly by a memoized pass over the node graph, and the sum of every completion is closed:
// n'! / (prod r'_i! prod_{j >= k} C_j!). A past value whose longest completion is still within
// the rule contributes its whole subtree at once; one whose shortest completion is beyond it is
// dropped; the others are carried to the next column.
//
// FEXACT merges past values within 3.45e-7 of each other in log (a relative error of that order
// in the sum) to keep their number down; here the merging is exact and the paths meet in the
// middle instead: the pasts are carried forward to the middle column only, the futures of the
// middle nodes come from a backward pass (at each node the futures that can never count for
// the range of pasts reaching it are dropped, those that always count summed into one), and at
// each middle node the pasts and the ascending futures meet by a pointer and running sums, so
// the lists are the square roots of the paths' numbers.
//
// The rows are the smaller dimension (the node is their sums), both orders descending. Logs of
// the factorials are double-doubles (special's loggamma, ~1e-19 absolute), so a table's log
// probability is good to ~1e-18 and the rule's 1e-7 decides every tie (a table exactly as
// probable as the observed one is counted). The work -- compositions enumerated, values carried
// -- and the values held at once are capped (H13): past either the test stops and reports it
// (the caller raises).
#pragma once

#include "htest/moments.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sd::htest::fexact {

using Key = std::vector<std::int32_t>;

struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        std::uint64_t h = 1469598103934665603ULL;
        for (std::int32_t v : k) {
            h ^= static_cast<std::uint32_t>(v);
            h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
};

// log k! for k = 0..n as double-doubles
inline std::vector<DD> log_factorials(std::int64_t n) {
    std::vector<DD> v(static_cast<std::size_t>(n + 1), DD{0.0, 0.0});
    for (std::int64_t k = 2; k <= n; ++k)
        v[static_cast<std::size_t>(k)] = dd::loggamma(DD{static_cast<double>(k + 1), 0.0});
    return v;
}

struct Result {
    double statistic = kNaN, pvalue = kNaN;
    bool capped = false;
};

class Network {
public:
    Network(std::vector<std::int64_t> rows, std::vector<std::int64_t> cols, double cap, double hold)
        : rows_(std::move(rows)), cols_(std::move(cols)), cap_(cap), hold_(hold) {}

    // log_obs_past: sum -log x! over the observed cells; lf the log factorials up to N.
    // The rule's threshold is log P(observed) + log(1 + 1e-7).
    Result run(DD log_obs_past, std::vector<DD> lf) {
        Result res;
        N_ = 0;
        for (auto c : cols_) N_ += c;
        lf_ = std::move(lf);
        DD logK{0.0, 0.0};
        for (auto r : rows_) logK = logK + lf_[static_cast<std::size_t>(r)];
        for (auto c : cols_) logK = logK + lf_[static_cast<std::size_t>(c)];
        logK = logK - lf_[static_cast<std::size_t>(N_)];
        logK_ = logK;
        const DD log_obs = logK + log_obs_past;
        res.statistic = std::min(1.0, stats::num::exp_dd(log_obs));
        thr_ = to_double(log_obs + dd::log(DD{1.0 + 1e-7, 0.0}));
        nc_ = static_cast<int>(cols_.size());
        nr_ = static_cast<int>(rows_.size());
        // the closed sums of the remaining columns: sum_{j >= k} log C_j!
        colfact_.assign(static_cast<std::size_t>(nc_ + 1), DD{0.0, 0.0});
        for (int k = nc_ - 1; k >= 0; --k)
            colfact_[static_cast<std::size_t>(k)] = colfact_[static_cast<std::size_t>(k + 1)] +
                                                    lf_[static_cast<std::size_t>(cols_[static_cast<std::size_t>(k)])];
        memo_.assign(static_cast<std::size_t>(nc_ + 1), {});
        Key root(rows_.begin(), rows_.end());
        Acc p;
        if (!main_pass(root, p)) {
            res.capped = true;
            res.pvalue = kNaN;
            return res;
        }
        res.pvalue = std::min(1.0, to_double(p.value()));
        return res;
    }

private:
    struct Bounds {
        double longest, shortest;  // max and min of sum -log x! over the completions
    };
    struct Past {
        DD lp;
        double m;
    };

    std::vector<std::int64_t> rows_, cols_;
    double cap_, hold_;
    double work_ = 0.0;
    std::int64_t N_ = 0;
    int nr_ = 0, nc_ = 0;
    std::vector<DD> lf_, colfact_;
    DD logK_{0.0, 0.0};
    double thr_ = 0.0;
    std::vector<std::unordered_map<Key, Bounds, KeyHash>> memo_;

    double lfd(std::int64_t v) const { return to_double(lf_[static_cast<std::size_t>(v)]); }

    static void sorted_child(const Key& key, const std::vector<std::int32_t>& x, Key& child) {
        const std::size_t n = key.size();
        child.resize(n);
        for (std::size_t i = 0; i < n; ++i) child[i] = key[i] - x[i];
        // insertion sort, descending (a handful of rows)
        for (std::size_t i = 1; i < n; ++i) {
            const std::int32_t v = child[i];
            std::size_t j = i;
            while (j > 0 && child[j - 1] < v) {
                child[j] = child[j - 1];
                --j;
            }
            child[j] = v;
        }
    }

    // f(x, w) for every x with sum_i x_i = total, 0 <= x_i <= key_i; w = sum -log x_i! (double-double).
    // Returns false when the work cap is passed.
    template <class F>
    bool compositions(const Key& key, std::int64_t total, F&& f) {
        const int n = static_cast<int>(key.size());
        std::vector<std::int32_t> x(static_cast<std::size_t>(n), 0);
        std::vector<std::int64_t> tail(static_cast<std::size_t>(n + 1), 0);  // sum of key[i..]
        for (int i = n - 1; i >= 0; --i)
            tail[static_cast<std::size_t>(i)] =
                tail[static_cast<std::size_t>(i + 1)] + key[static_cast<std::size_t>(i)];
        std::vector<DD> w(static_cast<std::size_t>(n + 1), DD{0.0, 0.0});
        bool ok = true;
        auto rec = [&](auto&& self, int i, std::int64_t left) -> void {
            if (!ok) return;
            if (i == n - 1) {
                x[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(left);
                const DD wi = w[static_cast<std::size_t>(i)] - lf_[static_cast<std::size_t>(left)];
                work_ += 1.0;
                f(x, wi);
                return;
            }
            const std::int64_t rest = tail[static_cast<std::size_t>(i + 1)];
            const std::int64_t lo = std::max<std::int64_t>(0, left - rest);
            const std::int64_t hi = std::min<std::int64_t>(key[static_cast<std::size_t>(i)], left);
            for (std::int64_t v = lo; v <= hi; ++v) {
                x[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(v);
                w[static_cast<std::size_t>(i + 1)] = w[static_cast<std::size_t>(i)] - lf_[static_cast<std::size_t>(v)];
                self(self, i + 1, left - v);
                if (!ok) return;
            }
            if (work_ > cap_) ok = false;
        };
        rec(rec, 0, total);
        return ok && work_ <= cap_;
    }

    double forced(const Key& key) const {
        double s = 0.0;
        for (std::int32_t v : key) s -= lfd(v);
        return s;
    }

    // the longest and shortest completions of node (k, key): k columns filled
    bool bounds(int k, const Key& key, Bounds& out) {
        if (k >= nc_ - 1) {
            out.longest = out.shortest = k == nc_ ? 0.0 : forced(key);
            return true;
        }
        auto& memo = memo_[static_cast<std::size_t>(k)];
        auto it = memo.find(key);
        if (it != memo.end()) {
            out = it->second;
            return true;
        }
        Bounds b{-kInf, kInf};
        Key child;
        bool ok = compositions(key, cols_[static_cast<std::size_t>(k)], [&](const std::vector<std::int32_t>& x, DD w) {
            sorted_child(key, x, child);
            Bounds cb;
            if (!bounds(k + 1, child, cb)) return;
            const double wv = to_double(w);
            b.longest = std::max(b.longest, wv + cb.longest);
            b.shortest = std::min(b.shortest, wv + cb.shortest);
        });
        if (!ok || work_ > cap_) return false;
        memo.emplace(key, b);
        if (static_cast<double>(memo.size()) > hold_) return false;
        out = b;
        return true;
    }

    // log of the sum over every completion of prod 1/x!: n'! / (prod r'! prod_{j >= k} C_j!)
    DD closed_sum(int k, const Key& key) const {
        std::int64_t n = 0;
        DD s{0.0, 0.0};
        for (std::int32_t v : key) {
            n += v;
            s = s - lf_[static_cast<std::size_t>(v)];
        }
        return s + lf_[static_cast<std::size_t>(n)] - colfact_[static_cast<std::size_t>(k)];
    }

    static void merge(std::vector<Past>& v) {
        std::sort(v.begin(), v.end(), [](const Past& a, const Past& b) {
            return a.lp.hi < b.lp.hi || (a.lp.hi == b.lp.hi && a.lp.lo < b.lp.lo);
        });
        std::size_t w = 0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (w > 0) {
                const DD d = v[i].lp - v[w - 1].lp;
                if (std::fabs(d.hi) <= 1e-24 * std::max(1.0, std::fabs(v[i].lp.hi))) {
                    v[w - 1].m += v[i].m;
                    continue;
                }
            }
            v[w++] = v[i];
        }
        v.resize(w);
    }

    struct Range {  // the past values that reach a node from the middle stage
        double amin = kInf, amax = -kInf;
    };
    struct Fut {  // a node's futures: the list kept (ascending after merge) and the log sum of those that always count
        std::vector<Past> list;
        DD always{-kInf, 0.0};
    };

    // log(e^a + e^b) of double-doubles (-inf for an empty sum)
    static DD log_add(DD a, DD b) {
        if (a.hi == -kInf) return b;
        if (b.hi == -kInf) return a;
        if (b.hi > a.hi) std::swap(a, b);
        const DD d = b - a;
        if (d.hi < -800.0) return a;
        return a + dd::log(DD{1.0, 0.0} + stats::num::exp_full(d));
    }

    // the forward pass to the middle stage km (every past decided by the node's longest and
    // shortest completions resolved on the way), the futures of the middle nodes by a backward
    // pass (each node's futures that can never count for the pasts that reach it dropped, those
    // that always count summed), and the two met at the middle nodes: for each past a, the
    // futures f with a + f within the rule, by a pointer into the ascending futures and their
    // running sums.
    bool main_pass(const Key& root, Acc& p) {
        const int km = std::max(1, nc_ / 2);
        const double thrp = thr_ - to_double(logK_);  // the rule on past + future
        std::unordered_map<Key, std::vector<Past>, KeyHash> stage, next;
        stage[root].push_back(Past{DD{0.0, 0.0}, 1.0});
        Key child;
        for (int k = 0; k <= km; ++k) {
            next.clear();
            double next_held = 0.0;
            for (auto& [key, pasts] : stage) {
                merge(pasts);
                Bounds b;
                if (!bounds(k, key, b)) return false;
                const DD fs = closed_sum(k, key);
                std::vector<Past> keep;
                for (const Past& q : pasts) {
                    const DD base = logK_ + q.lp;
                    const double hi = to_double(base) + b.longest, lo = to_double(base) + b.shortest;
                    if (hi <= thr_) p.add(stats::num::exp_dd(base + fs) * q.m);
                    else if (!(lo > thr_)) keep.push_back(q);
                }
                pasts.swap(keep);
                if (k == km || pasts.empty()) continue;
                bool ok = compositions(
                    key, cols_[static_cast<std::size_t>(k)], [&](const std::vector<std::int32_t>& x, DD w) {
                        sorted_child(key, x, child);
                        work_ += static_cast<double>(pasts.size());
                        auto& dst = next[child];
                        for (const Past& q : pasts) dst.push_back(Past{q.lp + w, q.m});
                        next_held += static_cast<double>(pasts.size());
                    }
                );
                if (!ok || work_ > cap_ || next_held > hold_) return false;
            }
            if (k == km) break;
            stage.swap(next);
        }
        // the ranges of the pasts reaching the nodes of the stages km..nc_ - 1
        std::vector<std::unordered_map<Key, Range, KeyHash>> ranges(static_cast<std::size_t>(nc_));
        double held = 0.0;
        for (auto& [key, pasts] : stage) {
            if (pasts.empty()) continue;
            Range r;
            for (const Past& q : pasts)
                r.amin = std::min(r.amin, to_double(q.lp)), r.amax = std::max(r.amax, to_double(q.lp));
            ranges[static_cast<std::size_t>(km)][key] = r;
            held += static_cast<double>(pasts.size());
        }
        if (ranges[static_cast<std::size_t>(km)].empty()) return true;
        for (int k = km; k < nc_ - 1; ++k) {
            auto& here = ranges[static_cast<std::size_t>(k)];
            auto& there = ranges[static_cast<std::size_t>(k + 1)];
            for (auto& [key, r] : here) {
                const Range rr = r;
                bool ok = compositions(
                    key, cols_[static_cast<std::size_t>(k)], [&](const std::vector<std::int32_t>& x, DD w) {
                        sorted_child(key, x, child);
                        Range& rc = there[child];
                        const double wv = to_double(w);
                        rc.amin = std::min(rc.amin, rr.amin + wv), rc.amax = std::max(rc.amax, rr.amax + wv);
                    }
                );
                if (!ok) return false;
            }
        }
        auto classify = [&](Fut& F, DD f, double m, const Range& r) {
            const double fv = to_double(f);
            if (r.amin + fv > thrp) return;
            if (r.amax + fv <= thrp) F.always = log_add(F.always, f + dd::log(DD{m, 0.0}));
            else F.list.push_back(Past{f, m});
        };
        std::unordered_map<Key, Fut, KeyHash> fut, fnext;
        for (auto& [key, r] : ranges[static_cast<std::size_t>(nc_ - 1)]) {
            Fut F;
            DD f{0.0, 0.0};
            for (std::int32_t v : key) f = f - lf_[static_cast<std::size_t>(v)];
            classify(F, f, 1.0, r);
            fnext.emplace(key, std::move(F));
        }
        for (int k = nc_ - 2; k >= km; --k) {
            fut.clear();
            double stage_held = 0.0;
            for (auto& [key, r] : ranges[static_cast<std::size_t>(k)]) {
                Fut F;
                const Range rr = r;
                bool ok = compositions(
                    key, cols_[static_cast<std::size_t>(k)], [&](const std::vector<std::int32_t>& x, DD w) {
                        sorted_child(key, x, child);
                        const Fut& Cf = fnext.find(child)->second;
                        if (Cf.always.hi != -kInf) F.always = log_add(F.always, Cf.always + w);
                        for (const Past& f : Cf.list) classify(F, f.lp + w, f.m, rr);
                        work_ += static_cast<double>(Cf.list.size());
                    }
                );
                if (!ok || work_ > cap_) return false;
                merge(F.list);
                stage_held += static_cast<double>(F.list.size());
                fut.emplace(key, std::move(F));
            }
            if (held + stage_held > hold_) return false;
            fnext.swap(fut);
        }
        // the middle nodes: pasts (ascending) against the futures (ascending)
        for (auto& [key, pasts] : stage) {
            if (pasts.empty()) continue;
            const Fut& F = fnext.find(key)->second;
            const std::size_t nf = F.list.size();
            std::vector<DD> Q(nf);  // Q_j = sum_{i <= j} m_i e^(f_i - f_j)
            for (std::size_t j = 0; j < nf; ++j) {
                Q[j] = DD{F.list[j].m, 0.0};
                if (j > 0) Q[j] = Q[j] + Q[j - 1] * stats::num::exp_full(F.list[j - 1].lp - F.list[j].lp);
            }
            work_ += static_cast<double>(nf + pasts.size());
            std::size_t j = 0;  // the futures with f + a <= thrp: [0, j)
            for (auto it = pasts.rbegin(); it != pasts.rend(); ++it) {
                const double av = to_double(it->lp);
                while (j < nf && to_double(F.list[j].lp) + av <= thrp) ++j;
                const DD base = logK_ + it->lp;
                double tot = 0.0;
                if (F.always.hi != -kInf) tot += stats::num::exp_dd(base + F.always);
                if (j > 0) tot += stats::num::exp_dd(base + F.list[j - 1].lp + dd::log(Q[j - 1]));
                p.add(dd::two_prod(tot, it->m));
            }
        }
        return work_ <= cap_;
    }
};

// The test of one table (cells row-major, r x c, every count >= 0): zero rows and columns are
// dropped, a table left with one row or column (or empty) is the only one with its margins
// (statistic and p-value 1, scipy's rule); the smaller dimension becomes the rows.
inline Result test(std::int64_t r, std::int64_t c, const std::vector<std::int64_t>& cells, double cap, double hold) {
    std::vector<std::int64_t> R(static_cast<std::size_t>(r), 0), C(static_cast<std::size_t>(c), 0);
    for (std::int64_t i = 0; i < r; ++i)
        for (std::int64_t j = 0; j < c; ++j) {
            R[static_cast<std::size_t>(i)] += cells[static_cast<std::size_t>(i * c + j)];
            C[static_cast<std::size_t>(j)] += cells[static_cast<std::size_t>(i * c + j)];
        }
    std::vector<std::int64_t> ri, ci;
    for (std::int64_t i = 0; i < r; ++i)
        if (R[static_cast<std::size_t>(i)] > 0) ri.push_back(i);
    for (std::int64_t j = 0; j < c; ++j)
        if (C[static_cast<std::size_t>(j)] > 0) ci.push_back(j);
    Result res;
    if (ri.size() <= 1 || ci.size() <= 1) {
        res.statistic = res.pvalue = 1.0;
        return res;
    }
    bool transpose = ri.size() > ci.size();
    std::vector<std::int64_t> rows, cols;
    std::vector<std::vector<std::int64_t>> t;  // the kept table, rows x cols
    if (!transpose) {
        for (auto i : ri) rows.push_back(R[static_cast<std::size_t>(i)]);
        for (auto j : ci) cols.push_back(C[static_cast<std::size_t>(j)]);
        for (auto i : ri) {
            t.emplace_back();
            for (auto j : ci) t.back().push_back(cells[static_cast<std::size_t>(i * c + j)]);
        }
    } else {
        for (auto j : ci) rows.push_back(C[static_cast<std::size_t>(j)]);
        for (auto i : ri) cols.push_back(R[static_cast<std::size_t>(i)]);
        for (auto j : ci) {
            t.emplace_back();
            for (auto i : ri) t.back().push_back(cells[static_cast<std::size_t>(i * c + j)]);
        }
    }
    std::int64_t N = 0;
    for (auto v : rows) N += v;
    if (static_cast<double>(N) > cap) {  // (the factorial table alone)
        res.capped = true;
        return res;
    }
    std::sort(rows.begin(), rows.end(), std::greater<>());
    std::sort(cols.begin(), cols.end(), std::greater<>());
    // the observed past value: sum -log x! over its cells (the order does not matter)
    std::vector<DD> lf = log_factorials(N);
    DD obs{0.0, 0.0};
    for (const auto& row : t)
        for (auto v : row) obs = obs - lf[static_cast<std::size_t>(v)];
    Network net(rows, cols, cap, hold);
    return net.run(obs, std::move(lf));
}

}  // namespace sd::htest::fexact
