// The binomial, Bernoulli, Poisson, geometric and negative binomial distributions
// (docs/api/stats.md §2.1, T25).
//
// The pmfs are the incomplete functions' prefixes, whose exponents are formed in
// double-double (special/prefix.hpp): for 1 <= k <= n - 1
//     binomial  C(n, k) p^k q^(n-k)  = B(k, n - k; p) n / (k (n - k)),   B(a, b; x) = x^a y^b / B(a, b)
//     neg. bin. Gamma(k+n)/(Gamma(n) k!) p^n q^k = B(n, k; p) / k
//     Poisson   mu^k e^-mu / k!       = D(k, mu),  D(a, x) = x^a e^-x / Gamma(a + 1)
// (Loader's saddle-point form with the deviance in double-double), the ends from
// k log q and n log p. The cdfs are sd.special's bdtr, nbdtr, pdtr and their
// complements; the log tails below the normal range come from the same pieces as the
// beta and gamma distributions' (stats/dist/beta.hpp, gamma.hpp).
#pragma once

#include "special/ops/distr.hpp"
#include "special/prefix.hpp"
#include "stats/discrete.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace binom_detail {

using num::DD;

// log C(n, k) p^k q^(n-k) for integers 0 <= k <= n, 0 < p < 1, q = 1 - p as a
// double-double (exact).
inline DD log_pmf(double k, double n, double p, DD q) {
    if (k == 0.0) return num::dd::log(q) * n;
    if (k == n) return num::log_dd(p) * n;
    return beta_detail::log_prefix(k, n - k, DD{p, 0.0}, q) + num::log_dd(n) - num::log_dd(k) - num::log_dd(n - k);
}

// True when x y (x, y doubles) is an integer: the product's two parts are exact.
inline bool product_is_int(double x, double y) {
    const DD m = num::dd::two_prod(x, y);
    return m.lo == 0.0 && std::floor(m.hi) == m.hi;
}

}  // namespace binom_detail

struct Binomial {
    static constexpr char name[] = "binomial";
    static constexpr std::array<const char*, 2> params{"n", "p"};

    static bool valid(double n, double p) { return n >= 0.0 && disc::is_int(n) && p >= 0.0 && p <= 1.0; }

    // The degenerate cases (n = 0, p = 0, p = 1): the point mass at `at`.
    static bool degenerate(double n, double p, double& at) {
        if (n == 0.0 || p == 0.0) {
            at = 0.0;
            return true;
        }
        if (p == 1.0) {
            at = n;
            return true;
        }
        return false;
    }

    static double logpmf(double k, double n, double p) {
        if (!disc::is_int(k) || k < 0.0 || k > n) return -kInf;
        double at;
        if (degenerate(n, p, at)) return k == at ? 0.0 : -kInf;
        const num::DD l = binom_detail::log_pmf(k, n, p, num::dd::two_sum(1.0, -p));
        return l.hi + l.lo;
    }
    static double pmf(double k, double n, double p) {
        if (!disc::is_int(k) || k < 0.0 || k > n) return 0.0;
        double at;
        if (degenerate(n, p, at)) return k == at ? 1.0 : 0.0;
        return num::exp_dd(binom_detail::log_pmf(k, n, p, num::dd::two_sum(1.0, -p)));
    }
    // P(X <= floor(x)) (upper: P(X > floor(x))).
    static double tail(double x, double n, double p, bool upper) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? 1.0 : 0.0;
        if (k >= n) return upper ? 0.0 : 1.0;
        return upper ? special::Bdtrc::f(k, n, p) : special::Bdtr::f(k, n, p);
    }
    static double cdf(double x, double n, double p) { return tail(x, n, p, false); }
    static double ccdf(double x, double n, double p) { return tail(x, n, p, true); }
    // log P(X <= k) = log I_q(n - k, k + 1) (upper: log of its complement).
    static double log_tail(double x, double n, double p, bool upper) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? 0.0 : -kInf;
        if (k >= n) return upper ? -kInf : 0.0;
        double at;
        if (degenerate(n, p, at)) return (k >= at) != upper ? 0.0 : -kInf;
        return beta_detail::log_tail(n - k, k + 1.0, num::dd::two_sum(1.0, -p), num::DD{p, 0.0}, upper);
    }
    static double logcdf(double x, double n, double p) { return log_tail(x, n, p, false); }
    static double logccdf(double x, double n, double p) { return log_tail(x, n, p, true); }

    struct Fns {
        double n, p, lo = 0.0, hi, guess;
        double cdf(double k) const { return Binomial::cdf(k, n, p); }
        double ccdf(double k) const { return Binomial::ccdf(k, n, p); }
        double logcdf(double k) const { return Binomial::logcdf(k, n, p); }
        double logccdf(double k) const { return Binomial::logccdf(k, n, p); }
    };
    static Fns fns(double n, double p) { return {n, p, 0.0, n, std::floor(n * p)}; }
    static double icdf(double t, double n, double p) { return disc::icdf(t, fns(n, p)); }
    static double iccdf(double t, double n, double p) { return disc::iccdf(t, fns(n, p)); }
    static double ilogcdf(double t, double n, double p) { return disc::ilogcdf(t, fns(n, p)); }
    static double ilogccdf(double t, double n, double p) { return disc::ilogccdf(t, fns(n, p)); }

    static double mean(double n, double p) { return n * p; }
    static double variance(double n, double p) { return n * p * (1.0 - p); }
    // (q - p) / sqrt(n p q); a point mass has no skewness or kurtosis (NaN).
    static double skewness(double n, double p) {
        double at;
        if (degenerate(n, p, at)) return kNaN;
        return (1.0 - 2.0 * p) / std::sqrt(n * p * (1.0 - p));
    }
    // (1 - 6 p q) / (n p q), the numerator in double-double (it crosses 0 at p = 0.211, 0.789).
    static double kurtosis(double n, double p) {
        double at;
        if (degenerate(n, p, at)) return kNaN;
        const num::DD q = num::dd::two_sum(1.0, -p);
        const num::DD pq = q * p;
        const num::DD num_ = num::DD{1.0, 0.0} - pq * 6.0;
        return (num_.hi + num_.lo) / (n * (pq.hi + pq.lo));
    }
    // floor((n + 1) p); two modes ((n + 1) p and one below) when (n + 1) p is an integer (T25).
    static double mode(double n, double p) {
        double at;
        if (degenerate(n, p, at)) return at;
        if (binom_detail::product_is_int(n + 1.0, p)) return kNaN;
        return std::floor((n + 1.0) * p);
    }
    static double entropy(double n, double p) {
        double at;
        if (degenerate(n, p, at)) return 0.0;
        const num::DD q = num::dd::two_sum(1.0, -p);
        return disc::entropy_sum(std::floor((n + 1.0) * p), 0.0, n,
                                 [&](double k) { return binom_detail::log_pmf(k, n, p, q); });
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double n, double) { return n; }
};

// Bernoulli(p) = binomial(1, p).
struct Bernoulli {
    static constexpr char name[] = "bernoulli";
    static constexpr std::array<const char*, 1> params{"p"};

    static bool valid(double p) { return p >= 0.0 && p <= 1.0; }
    // Closed forms: 1 - p is exact for p >= 1/2 and relative to within an ULP below.
    static double pmf(double k, double p) { return k == 0.0 ? 1.0 - p : (k == 1.0 ? p : 0.0); }
    static double logpmf(double k, double p) {
        return k == 0.0 ? std::log1p(-p) : (k == 1.0 ? std::log(p) : -kInf);
    }
    static double cdf(double x, double p) { return x < 0.0 ? 0.0 : (x < 1.0 ? 1.0 - p : 1.0); }
    static double ccdf(double x, double p) { return x < 0.0 ? 1.0 : (x < 1.0 ? p : 0.0); }
    static double logcdf(double x, double p) { return x < 0.0 ? -kInf : (x < 1.0 ? std::log1p(-p) : 0.0); }
    static double logccdf(double x, double p) { return x < 0.0 ? 0.0 : (x < 1.0 ? std::log(p) : -kInf); }
    static double icdf(double t, double p) { return Binomial::icdf(t, 1.0, p); }
    static double iccdf(double t, double p) { return Binomial::iccdf(t, 1.0, p); }
    static double ilogcdf(double t, double p) { return Binomial::ilogcdf(t, 1.0, p); }
    static double ilogccdf(double t, double p) { return Binomial::ilogccdf(t, 1.0, p); }
    static double mean(double p) { return p; }
    static double variance(double p) { return p * (1.0 - p); }
    static double skewness(double p) { return Binomial::skewness(1.0, p); }
    static double kurtosis(double p) { return Binomial::kurtosis(1.0, p); }
    static double median(double p) { return Binomial::icdf(0.5, 1.0, p); }
    static double mode(double p) { return Binomial::mode(1.0, p); }
    static double entropy(double p) {  // entr(p) + entr(q)
        if (p == 0.0 || p == 1.0) return 0.0;
        const num::DD q = num::dd::two_sum(1.0, -p);
        const num::DD h = -(num::log_dd(p) * p) - num::dd::log(q) * q;
        return h.hi + h.lo;
    }
    static double support_lo(double) { return 0.0; }
    static double support_hi(double) { return 1.0; }
};

namespace poisson_detail {

using num::DD;

// log(mu^k e^-mu / k!) for an integer k >= 1, mu > 0: for k < 13 directly, k! exact;
// from 13 the prefix's form -E - log(sqrt(2 pi k) Gamma*(k)), E = mu - k - k log(mu/k)
// (special/prefix.hpp) in double-double.
inline DD log_pmf(double k, double mu) {
    if (k < 13.0) {
        double f = 1.0;
        for (double j = 2.0; j <= k; j += 1.0) f *= j;
        return num::log_dd(mu) * k - mu - num::log_dd(f);
    }
    const double c = std::log(std::sqrt(2.0 * special::kPi * k) * special::gamma_star(k));
    return -special::gamma_exponent(k, mu) - c;
}

}  // namespace poisson_detail

struct Poisson {
    static constexpr char name[] = "poisson";
    static constexpr std::array<const char*, 1> params{"mu"};

    static bool valid(double mu) { return mu >= 0.0 && std::isfinite(mu); }

    static num::DD log_pmf_dd(double k, double mu) {
        if (k == 0.0) return num::DD{-mu, 0.0};
        return poisson_detail::log_pmf(k, mu);
    }
    static double logpmf(double k, double mu) {
        if (!disc::is_int(k) || k < 0.0) return -kInf;
        if (mu == 0.0) return k == 0.0 ? 0.0 : -kInf;
        const num::DD l = log_pmf_dd(k, mu);
        return l.hi + l.lo;
    }
    static double pmf(double k, double mu) {
        if (!disc::is_int(k) || k < 0.0) return 0.0;
        if (mu == 0.0) return k == 0.0 ? 1.0 : 0.0;
        return num::exp_dd(log_pmf_dd(k, mu));
    }
    static double cdf(double x, double mu) {
        const double k = std::floor(x);
        if (k < 0.0) return 0.0;
        if (mu == 0.0 || std::isinf(k)) return 1.0;
        return special::Pdtr::f(k, mu);
    }
    static double ccdf(double x, double mu) {
        const double k = std::floor(x);
        if (k < 0.0) return 1.0;
        if (mu == 0.0 || std::isinf(k)) return 0.0;
        return special::Pdtrc::f(k, mu);
    }
    // log P(X <= k) = log Q(k + 1, mu) (upper: log P(k + 1, mu)).
    static double log_tail(double x, double mu, bool upper) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? 0.0 : -kInf;
        if (mu == 0.0 || std::isinf(k)) return upper ? -kInf : 0.0;
        return gamma_detail::log_tail(k + 1.0, num::DD{mu, 0.0}, !upper);
    }
    static double logcdf(double x, double mu) { return log_tail(x, mu, false); }
    static double logccdf(double x, double mu) { return log_tail(x, mu, true); }

    struct Fns {
        double mu, lo = 0.0, hi = kInf, guess;
        double cdf(double k) const { return Poisson::cdf(k, mu); }
        double ccdf(double k) const { return Poisson::ccdf(k, mu); }
        double logcdf(double k) const { return Poisson::logcdf(k, mu); }
        double logccdf(double k) const { return Poisson::logccdf(k, mu); }
    };
    static Fns fns(double mu) { return {mu, 0.0, mu == 0.0 ? 0.0 : kInf, std::floor(mu)}; }
    static double icdf(double t, double mu) { return disc::icdf(t, fns(mu)); }
    static double iccdf(double t, double mu) { return disc::iccdf(t, fns(mu)); }
    static double ilogcdf(double t, double mu) { return disc::ilogcdf(t, fns(mu)); }
    static double ilogccdf(double t, double mu) { return disc::ilogccdf(t, fns(mu)); }

    static double mean(double mu) { return mu; }
    static double variance(double mu) { return mu; }
    static double skewness(double mu) { return mu == 0.0 ? kNaN : 1.0 / std::sqrt(mu); }
    static double kurtosis(double mu) { return mu == 0.0 ? kNaN : 1.0 / mu; }
    // floor(mu); an integer mu >= 1 has two modes, mu - 1 and mu (T25).
    static double mode(double mu) {
        if (mu == 0.0) return 0.0;
        return std::floor(mu) == mu ? kNaN : std::floor(mu);
    }
    // The sum over the support, or from mu = 1e4 the asymptotic series
    // (1 + log 2 pi mu)/2 - 1/(12 mu) - 1/(24 mu^2) - 19/(360 mu^3) + ...
    static double entropy(double mu) {
        if (mu == 0.0) return 0.0;
        if (mu >= 1e4) {
            const double r = 1.0 / mu;
            const num::DD h = num::kNormalEntropy + num::log_dd(mu) * 0.5 +
                              r * (-1.0 / 12.0 + r * (-1.0 / 24.0 + r * (-19.0 / 360.0)));
            return h.hi + h.lo;
        }
        return disc::entropy_sum(std::floor(mu), 0.0, kInf, [&](double k) { return log_pmf_dd(k, mu); });
    }
    static double support_lo(double) { return 0.0; }
    static double support_hi(double mu) { return mu == 0.0 ? 0.0 : kInf; }
};

struct Geometric {
    static constexpr char name[] = "geometric";
    static constexpr std::array<const char*, 1> params{"p"};

    static bool valid(double p) { return p > 0.0 && p <= 1.0; }

    // log q = log(1 - p), 1 - p exact as a double-double (per thread, for the last p).
    static num::DD log_q(double p) {
        thread_local num::Memo memo;
        return memo.get(p, 0.0, [&] { return num::dd::log(num::dd::two_sum(1.0, -p)); });
    }

    static num::DD log_pmf_dd(double k, double p) { return log_q(p) * (k - 1.0) + num::log_dd(p); }
    static double logpmf(double k, double p) {
        if (!disc::is_int(k) || k < 1.0) return -kInf;
        if (p == 1.0) return k == 1.0 ? 0.0 : -kInf;
        const num::DD l = log_pmf_dd(k, p);
        return l.hi + l.lo;
    }
    static double pmf(double k, double p) {
        if (!disc::is_int(k) || k < 1.0) return 0.0;
        if (p == 1.0) return k == 1.0 ? 1.0 : 0.0;
        return num::exp_dd(log_pmf_dd(k, p));
    }
    // P(X > k) = q^k, P(X <= k) = 1 - q^k = -expm1(k log q).
    static double cdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 0.0;
        if (p == 1.0 || std::isinf(k)) return 1.0;
        const num::DD e = log_q(p) * k;
        return -std::expm1(e.hi) - std::exp(e.hi) * e.lo;
    }
    static double ccdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 1.0;
        if (p == 1.0 || std::isinf(k)) return 0.0;
        return num::exp_dd(log_q(p) * k);
    }
    static double logccdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 0.0;
        if (p == 1.0 || std::isinf(k)) return -kInf;
        const num::DD e = log_q(p) * k;
        return e.hi + e.lo;
    }
    static double logcdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return -kInf;
        if (p == 1.0 || std::isinf(k)) return 0.0;
        const double c = ccdf(x, p);
        return c <= 0.5 ? std::log1p(-c) : std::log(cdf(x, p));
    }

    struct Fns {
        double p, lo = 1.0, hi, guess;
        double cdf(double k) const { return Geometric::cdf(k, p); }
        double ccdf(double k) const { return Geometric::ccdf(k, p); }
        double logcdf(double k) const { return Geometric::logcdf(k, p); }
        double logccdf(double k) const { return Geometric::logccdf(k, p); }
    };
    static Fns fns(double p) { return {p, 1.0, p == 1.0 ? 1.0 : kInf, std::floor(1.0 / p)}; }
    // The search starts at the closed form ceil(log P(X > k) / log q) (q^k <= the upper tail).
    static Fns at(double p, double log_upper) { return disc::with_guess(fns(p), std::ceil(log_upper / std::log1p(-p))); }
    static double icdf(double t, double p) { return disc::icdf(t, at(p, std::log1p(-t))); }
    static double iccdf(double t, double p) { return disc::iccdf(t, at(p, std::log(t))); }
    static double ilogcdf(double t, double p) { return disc::ilogcdf(t, at(p, std::log(-std::expm1(t)))); }
    static double ilogccdf(double t, double p) { return disc::ilogccdf(t, at(p, t)); }

    static double mean(double p) { return 1.0 / p; }
    static double variance(double p) { return (1.0 - p) / (p * p); }
    static double skewness(double p) { return p == 1.0 ? kNaN : (2.0 - p) / std::sqrt(1.0 - p); }
    static double kurtosis(double p) { return p == 1.0 ? kNaN : 6.0 + p * p / (1.0 - p); }
    static double mode(double) { return 1.0; }
    static double entropy(double p) {  // (entr(p) + entr(q)) / p
        if (p == 1.0) return 0.0;
        const num::DD q = num::dd::two_sum(1.0, -p);
        const num::DD h = (-(num::log_dd(p) * p) - num::dd::log(q) * q) / p;
        return h.hi + h.lo;
    }
    static double support_lo(double) { return 1.0; }
    static double support_hi(double p) { return p == 1.0 ? 1.0 : kInf; }
};

namespace nbinom_detail {

using num::DD;

// log of Gamma(k+n) / (Gamma(n) k!) p^n q^k for an integer k >= 0, n > 0, 0 < p < 1.
inline DD log_pmf(double k, double n, double p, DD q) {
    if (k == 0.0) return num::log_dd(p) * n;
    return beta_detail::log_prefix(n, k, DD{p, 0.0}, q) - num::log_dd(k);
}

}  // namespace nbinom_detail

struct NegativeBinomial {
    static constexpr char name[] = "negativebinomial";
    static constexpr std::array<const char*, 2> params{"n", "p"};

    static bool valid(double n, double p) { return n > 0.0 && std::isfinite(n) && p > 0.0 && p <= 1.0; }

    static double logpmf(double k, double n, double p) {
        if (!disc::is_int(k) || k < 0.0) return -kInf;
        if (p == 1.0) return k == 0.0 ? 0.0 : -kInf;
        const num::DD l = nbinom_detail::log_pmf(k, n, p, num::dd::two_sum(1.0, -p));
        return l.hi + l.lo;
    }
    static double pmf(double k, double n, double p) {
        if (!disc::is_int(k) || k < 0.0) return 0.0;
        if (p == 1.0) return k == 0.0 ? 1.0 : 0.0;
        return num::exp_dd(nbinom_detail::log_pmf(k, n, p, num::dd::two_sum(1.0, -p)));
    }
    static double cdf(double x, double n, double p) {
        const double k = std::floor(x);
        if (k < 0.0) return 0.0;
        if (p == 1.0 || std::isinf(k)) return 1.0;
        return special::Nbdtr::f(k, n, p);
    }
    static double ccdf(double x, double n, double p) {
        const double k = std::floor(x);
        if (k < 0.0) return 1.0;
        if (p == 1.0 || std::isinf(k)) return 0.0;
        return special::Nbdtrc::f(k, n, p);
    }
    // log P(X <= k) = log I_p(n, k + 1).
    static double log_tail(double x, double n, double p, bool upper) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? 0.0 : -kInf;
        if (p == 1.0 || std::isinf(k)) return upper ? -kInf : 0.0;
        return beta_detail::log_tail(n, k + 1.0, num::DD{p, 0.0}, num::dd::two_sum(1.0, -p), upper);
    }
    static double logcdf(double x, double n, double p) { return log_tail(x, n, p, false); }
    static double logccdf(double x, double n, double p) { return log_tail(x, n, p, true); }

    struct Fns {
        double n, p, lo = 0.0, hi, guess;
        double cdf(double k) const { return NegativeBinomial::cdf(k, n, p); }
        double ccdf(double k) const { return NegativeBinomial::ccdf(k, n, p); }
        double logcdf(double k) const { return NegativeBinomial::logcdf(k, n, p); }
        double logccdf(double k) const { return NegativeBinomial::logccdf(k, n, p); }
    };
    static Fns fns(double n, double p) { return {n, p, 0.0, p == 1.0 ? 0.0 : kInf, std::floor(n * (1.0 - p) / p)}; }
    static double icdf(double t, double n, double p) { return disc::icdf(t, fns(n, p)); }
    static double iccdf(double t, double n, double p) { return disc::iccdf(t, fns(n, p)); }
    static double ilogcdf(double t, double n, double p) { return disc::ilogcdf(t, fns(n, p)); }
    static double ilogccdf(double t, double n, double p) { return disc::ilogccdf(t, fns(n, p)); }

    static double mean(double n, double p) { return n * (1.0 - p) / p; }
    static double variance(double n, double p) { return n * (1.0 - p) / (p * p); }
    static double skewness(double n, double p) { return p == 1.0 ? kNaN : (2.0 - p) / std::sqrt(n * (1.0 - p)); }
    static double kurtosis(double n, double p) { return p == 1.0 ? kNaN : 6.0 / n + p * p / (n * (1.0 - p)); }
    // floor((n - 1) q / p) for n > 1 (else 0); two modes when that is a positive integer (T25),
    // tested exactly: (n - 1) q - m p = 0 with every product's parts exact.
    static double mode(double n, double p) {
        if (n <= 1.0 || p == 1.0) return 0.0;
        const num::DD q = num::dd::two_sum(1.0, -p);
        const num::DD a = q * (n - 1.0);  // (n - 1) q (n - 1 exact)
        const double m = std::round((a.hi + a.lo) / p);
        if (m > 0.0) {
            const num::DD r = num::dd::two_prod(n - 1.0, q.hi) + num::dd::two_prod(n - 1.0, q.lo) - num::dd::two_prod(m, p);
            if (r.hi == 0.0 && r.lo == 0.0) return kNaN;
        }
        const num::DD v = a / p;
        return std::floor(v.hi + v.lo);
    }
    static double entropy(double n, double p) {
        if (p == 1.0) return 0.0;
        const num::DD q = num::dd::two_sum(1.0, -p);
        return disc::entropy_sum(std::floor(n * (1.0 - p) / p), 0.0, kInf,
                                 [&](double k) { return nbinom_detail::log_pmf(k, n, p, q); });
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double p) { return p == 1.0 ? 0.0 : kInf; }
};

}  // namespace sd::stats
