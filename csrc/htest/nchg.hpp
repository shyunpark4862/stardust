// Fisher's noncentral hypergeometric distribution (milestone H6, docs/design/htest.md §3.7;
// decisions H14, H79): the conditional distribution of the top-left cell X of a 2 x 2 table given
// its margins when the odds ratio is omega,
//
//     P(X = k) proportional to C(n, k) C(M - n, N - k) omega^k,   lo <= k <= hi,
//
// (M the total, n the first row's sum, N the first column's; scipy's nchypergeom_fisher(M, n, N,
// omega)). It is only htest's: the conditional maximum likelihood odds ratio (E X = x) and the
// exact confidence interval (P(X >= x) = alpha, P(X <= x) = alpha) of odds_ratio solve for
// t = log omega.
//
// The log weights relative to the central mode come from the exact ratios of neighbours (one
// double-double log each); at a t the terms are exp(L_k + (k - mode) t - max), every sum one of
// positive terms (the mean's and the tails'), the variance about the mean. The solver is
// Newton's in t inside a bracket that it keeps (bisection when a step leaves it), to |dt| below
// 4 ulp.
//
// A support of more than kFull points (H150) is not held whole: at each t the terms are built
// outwards from the mode at t (the k where the ratio of neighbours w(k + 1) e^t / w(k) crosses 1)
// as running double-double products of those ratios, until a term falls below 2^-110 of the
// mode's (or alpha 2^-60, the tail asked for, when that is smaller): a window of O(sqrt(N))
// points whatever the counts.
#pragma once

#include "htest/moments.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::htest::nchg {

class Dist {
public:
    static constexpr std::int64_t kFull = 1 << 16;
    static constexpr std::int64_t kMaxWindow = 1 << 22;  // a root's window past it: too_big()

    std::int64_t lo = 0, hi = 0, mode = 0;

    Dist(std::int64_t M, std::int64_t n, std::int64_t N) : M_(M), n_(n), N_(N) {
        lo = std::max<std::int64_t>(0, n + N - M);
        hi = std::min(n, N);
        const __int128 m = (static_cast<__int128>(n) + 1) * (static_cast<__int128>(N) + 1) / (M + 2);
        mode = static_cast<std::int64_t>(std::clamp<__int128>(m, lo, hi));
        full_ = hi - lo + 1 <= kFull;
        if (!full_) return;
        lw_.assign(static_cast<std::size_t>(hi - lo + 1), DD{0.0, 0.0});
        DD v{0.0, 0.0};
        for (std::int64_t k = mode; k < hi; ++k) {  // w(k + 1) / w(k) = (n - k)(N - k) / ((k + 1)(M - n - N + k + 1))
            v = v + dd::log(up(k));
            lw_[static_cast<std::size_t>(k + 1 - lo)] = v;
        }
        v = DD{0.0, 0.0};
        for (std::int64_t k = mode; k > lo; --k) {  // w(k - 1) / w(k) = k (M - n - N + k) / ((n - k + 1)(N - k + 1))
            v = v + dd::log(down(k));
            lw_[static_cast<std::size_t>(k - 1 - lo)] = v;
        }
        e_.resize(lw_.size());
    }

    struct At {
        double mean, var, total;
    };

    // the terms at t into e_ (largest 1; the k of e_[0] is base_) and the mean and variance
    At at(double t, double cut = kCut) {
        if (!full_) return window(t, cut);
        base_ = lo;
        double top = -kInf;
        std::vector<DD> L(lw_.size());
        for (std::size_t i = 0; i < lw_.size(); ++i) {
            const double k = static_cast<double>(lo + static_cast<std::int64_t>(i) - mode);
            L[i] = lw_[i] + dd::two_prod(k, t);
            top = std::max(top, L[i].hi);
        }
        Acc s, s1;
        for (std::size_t i = 0; i < lw_.size(); ++i) {
            e_[i] = std::exp(to_double(L[i] - top));
            s.add(e_[i]);
            s1.add(dd::two_prod(e_[i], static_cast<double>(lo + static_cast<std::int64_t>(i) - mode)));
        }
        const double S = to_double(s.value());
        const double mrel = to_double(s1.value()) / S;
        Acc v;
        for (std::size_t i = 0; i < lw_.size(); ++i) {
            const double d = static_cast<double>(lo + static_cast<std::int64_t>(i) - mode) - mrel;
            v.add(e_[i] * d * d);
        }
        return {static_cast<double>(mode) + mrel, to_double(v.value()) / S, S};
    }
    // log P(X >= x) (upper) or log P(X <= x) at t, and d/dt of it; cut the window's (the log of
    // the smallest term kept relative to the largest)
    void tail(double t, std::int64_t x, bool upper, double& lp, double& dlp, double cut = kCut) {
        const At a = at(t, cut);
        Acc s, s1;
        for (std::size_t i = 0; i < e_.size(); ++i) {
            const std::int64_t k = base_ + static_cast<std::int64_t>(i);
            if (upper ? k >= x : k <= x) {
                s.add(e_[i]);
                s1.add(e_[i] * static_cast<double>(k - mode));
            }
        }
        const double S = to_double(s.value());
        lp = S > 0.0 ? std::log(S / a.total) : -kInf;
        dlp = S > 0.0 ? (static_cast<double>(mode) + to_double(s1.value()) / S) - a.mean : 0.0;
    }

    // the t where g(t) = 0 for g increasing; g(t, val, slope)
    template <class G>
    static double solve(G&& g, double t0) {
        double val, slope;
        double t = t0;
        g(t, val, slope);
        if (val == 0.0) return t;
        // a bracket [a, b] with g(a) < 0 < g(b)
        double a, b, step = 1.0;
        if (val < 0.0) {
            a = t, b = t + step;
            for (int i = 0; i < 2000; ++i) {
                g(b, val, slope);
                if (!(val < 0.0)) break;
                a = b, step *= 2.0, b = t + step;
            }
        } else {
            b = t, a = t - step;
            for (int i = 0; i < 2000; ++i) {
                g(a, val, slope);
                if (!(val > 0.0)) break;
                b = a, step *= 2.0, a = t - step;
            }
        }
        t = 0.5 * (a + b);
        for (int i = 0; i < 400; ++i) {
            g(t, val, slope);
            if (val == 0.0) return t;
            if (val < 0.0) a = t;
            else b = t;
            double next = (slope > 0.0 && std::isfinite(val)) ? t - val / slope : 0.5 * (a + b);
            if (!(next > a && next < b)) next = 0.5 * (a + b);
            const double tol = 4.0 * 0x1p-53 * std::max(1.0, std::fabs(t));
            if (std::fabs(next - t) <= tol || b - a <= tol) return next;
            t = next;
        }
        return t;
    }

    // the window at a root found was cut at kMaxWindow points (the counts too large for the
    // exact sums): its value is not to be trusted
    bool too_big() const { return too_big_; }

    // the conditional maximum likelihood odds ratio: E X = x (0 at lo, inf at hi)
    double mle(std::int64_t x) {
        if (x <= lo) return 0.0;
        if (x >= hi) return kInf;
        const double t = solve(
            [&](double tt, double& val, double& slope) {
                const At a = at(tt);
                val = a.mean - static_cast<double>(x), slope = a.var;
            },
            0.0
        );
        too_big_ = too_big_ || cut_short_;
        return std::exp(t);
    }
    // the omega with P(X >= x) = alpha (lower end; 0 at x = lo) or P(X <= x) = alpha (upper; inf at hi)
    double end(std::int64_t x, double alpha, bool lower) {
        if (lower && x <= lo) return 0.0;
        if (!lower && x >= hi) return kInf;
        const double la = std::log(alpha);
        const double cut = std::max(kMinCut, std::min(kCut, la - 60.0 * kLn2));
        const double t = solve(
            [&](double tt, double& val, double& slope) {
                double lp, dlp;
                tail(tt, x, lower, lp, dlp, cut);
                // P(X >= x) rises with t, P(X <= x) falls: the latter's sign turned
                val = lower ? lp - la : la - lp, slope = lower ? dlp : -dlp;
                if (lp == -kInf) val = lower ? -kInf : kInf;
            },
            0.0
        );
        too_big_ = too_big_ || cut_short_;
        return std::exp(t);
    }

private:
    static constexpr double kLn2 = 0.6931471805599453;
    static constexpr double kCut = -110.0 * kLn2;  // 2^-110
    static constexpr double kMinCut = -745.2;  // below the smallest double: every term kept

    std::int64_t M_, n_, N_;
    bool full_ = true, too_big_ = false, cut_short_ = false;
    std::int64_t base_ = 0;
    std::vector<DD> lw_;
    std::vector<double> e_;

    // w(k + 1) / w(k) and w(k - 1) / w(k), the exact factors as double-doubles (every count below 2^53)
    DD up(std::int64_t k) const {
        return dd::two_prod(static_cast<double>(n_ - k), static_cast<double>(N_ - k)) /
               dd::two_prod(static_cast<double>(k + 1), static_cast<double>(M_ - n_ - N_ + k + 1));
    }
    DD down(std::int64_t k) const {
        return dd::two_prod(static_cast<double>(k), static_cast<double>(M_ - n_ - N_ + k)) /
               dd::two_prod(static_cast<double>(n_ - k + 1), static_cast<double>(N_ - k + 1));
    }

    // The terms of a large support at t: from the mode at t (the largest k with w(k) e^{kt} >=
    // w(k - 1) e^{(k-1)t}: the ratio up(k - 1) e^t falls with k, found by bisection) outwards as
    // running products, each side until a term is below e^cut; e_ holds them from base_ on.
    At window(double t, double cut) {
        const DD et = stats::num::exp_full(DD{t, 0.0});
        auto rises = [&](std::int64_t k) {  // w(k) e^t / w(k - 1) >= 1, k > lo
            const DD r = up(k - 1) * et;
            return r.hi > 1.0 || (r.hi == 1.0 && r.lo >= 0.0);
        };
        std::int64_t a = lo, b = hi + 1;  // a: the mode or below, b: past it
        if (std::isinf(et.hi)) a = hi;
        else if (et.hi == 0.0) b = lo + 1;
        while (b - a > 1) {
            const std::int64_t mid = a + (b - a) / 2;
            if (rises(mid)) a = mid;
            else b = mid;
        }
        const std::int64_t m = a;
        std::vector<double> above, below;  // the terms at m + 1, m + 2, ... and m - 1, m - 2, ...
        const double floor_ = std::exp(cut);
        DD v{1.0, 0.0};
        for (std::int64_t k = m; k < hi; ++k) {
            v = v * (up(k) * et);
            if (!(v.hi >= floor_)) break;
            above.push_back(v.hi);
            if (static_cast<std::int64_t>(above.size()) >= kMaxWindow / 2) break;
        }
        const DD eti = DD{1.0, 0.0} / et;
        v = DD{1.0, 0.0};
        for (std::int64_t k = m; k > lo; --k) {
            v = v * (down(k) * eti);
            if (!(v.hi >= floor_)) break;
            below.push_back(v.hi);
            if (static_cast<std::int64_t>(below.size()) >= kMaxWindow / 2) break;
        }
        // cut short: the sums are those of the window only (the solver's bracket still finds its
        // side; a root found on such a window is too_big())
        cut_short_ = static_cast<std::int64_t>(above.size()) >= kMaxWindow / 2 ||
                     static_cast<std::int64_t>(below.size()) >= kMaxWindow / 2;
        base_ = m - static_cast<std::int64_t>(below.size());
        e_.assign(below.rbegin(), below.rend());
        e_.push_back(1.0);
        e_.insert(e_.end(), above.begin(), above.end());
        Acc s, s1;
        for (std::size_t i = 0; i < e_.size(); ++i) {
            s.add(e_[i]);
            s1.add(dd::two_prod(e_[i], static_cast<double>(base_ + static_cast<std::int64_t>(i) - m)));
        }
        const double S = to_double(s.value());
        const double mrel = to_double(s1.value()) / S;
        Acc var;
        for (std::size_t i = 0; i < e_.size(); ++i) {
            const double d = static_cast<double>(base_ + static_cast<std::int64_t>(i) - m) - mrel;
            var.add(e_[i] * d * d);
        }
        return {static_cast<double>(m) + mrel, to_double(var.value()) / S, S};
    }
};

}  // namespace sd::htest::nchg
