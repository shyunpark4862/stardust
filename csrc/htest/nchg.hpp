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
    std::int64_t lo = 0, hi = 0, mode = 0;

    Dist(std::int64_t M, std::int64_t n, std::int64_t N) {
        lo = std::max<std::int64_t>(0, n + N - M);
        hi = std::min(n, N);
        mode = std::clamp<std::int64_t>((n + 1) * (N + 1) / (M + 2), lo, hi);
        lw_.assign(static_cast<std::size_t>(hi - lo + 1), DD{0.0, 0.0});
        DD v{0.0, 0.0};
        for (std::int64_t k = mode; k < hi; ++k) {  // w(k + 1) / w(k) = (n - k)(N - k) / ((k + 1)(M - n - N + k + 1))
            const DD a = dd::two_prod(static_cast<double>(n - k), static_cast<double>(N - k));
            const DD b = dd::two_prod(static_cast<double>(k + 1), static_cast<double>(M - n - N + k + 1));
            v = v + dd::log(a / b);
            lw_[static_cast<std::size_t>(k + 1 - lo)] = v;
        }
        v = DD{0.0, 0.0};
        for (std::int64_t k = mode; k > lo; --k) {  // w(k - 1) / w(k) = k (M - n - N + k) / ((n - k + 1)(N - k + 1))
            const DD a = dd::two_prod(static_cast<double>(k), static_cast<double>(M - n - N + k));
            const DD b = dd::two_prod(static_cast<double>(n - k + 1), static_cast<double>(N - k + 1));
            v = v + dd::log(a / b);
            lw_[static_cast<std::size_t>(k - 1 - lo)] = v;
        }
        e_.resize(lw_.size());
    }

    struct At {
        double mean, var, total;
    };

    // the terms at t into e_ (largest 1) and the mean and variance
    At at(double t) {
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
    // log P(X >= x) (upper) or log P(X <= x) at t, and d/dt of it
    void tail(double t, std::int64_t x, bool upper, double& lp, double& dlp) {
        const At a = at(t);
        Acc s, s1;
        for (std::size_t i = 0; i < e_.size(); ++i) {
            const std::int64_t k = lo + static_cast<std::int64_t>(i);
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
        return std::exp(t);
    }
    // the omega with P(X >= x) = alpha (lower end; 0 at x = lo) or P(X <= x) = alpha (upper; inf at hi)
    double end(std::int64_t x, double alpha, bool lower) {
        if (lower && x <= lo) return 0.0;
        if (!lower && x >= hi) return kInf;
        const double la = std::log(alpha);
        const double t = solve(
            [&](double tt, double& val, double& slope) {
                double lp, dlp;
                tail(tt, x, lower, lp, dlp);
                // P(X >= x) rises with t, P(X <= x) falls: the latter's sign turned
                val = lower ? lp - la : la - lp, slope = lower ? dlp : -dlp;
                if (lp == -kInf) val = lower ? -kInf : kInf;
            },
            0.0
        );
        return std::exp(t);
    }

private:
    std::vector<DD> lw_;
    std::vector<double> e_;
};

}  // namespace sd::htest::nchg
