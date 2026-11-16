// Double-double linear algebra for the multivariate and matrix distributions (T9, T10;
// docs/design/stats.md §6): a matrix factored in double already moves a quadratic form or
// trace by ~kappa eps (it is the factor of A + E, |E| ~ d eps |A|); in double-double that is
// ~kappa 1e-32.
#pragma once

#include "stats/numeric.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::stats::ddla {

using num::DD;
namespace dd = num::dd;

// sqrt of a positive double-double: s = sqrt(hi), then one Newton correction (a - s^2) / 2s.
inline DD sqrt(DD a) {
    const double s = std::sqrt(a.hi);
    const DD s2 = dd::two_prod(s, s);
    const double corr = ((a.hi - s2.hi) - s2.lo + a.lo) / (2.0 * s);
    return dd::two_sum(s, corr);
}

// log1p(u) for a double-double u > -1: the series for |u| < 2^-10 (1 + u would drop u's
// low bits), else log of 1 + u.
inline DD log1p(DD u) {
    if (std::fabs(u.hi) < 0x1p-10) {  // sum (-1)^(k+1) u^k / k
        DD sum = u, pw = u;
        for (int k = 2; k < 40; ++k) {
            pw = pw * u;
            const DD t = pw / static_cast<double>(k);
            sum = (k % 2 == 0) ? sum - t : sum + t;
            if (std::fabs(t.hi) <= 0x1p-110 * std::fabs(sum.hi)) break;
        }
        return sum;
    }
    return dd::log(u + 1.0);
}

// The lower Cholesky factor L (d x d, row-major, the upper part left 0) of the matrix whose
// lower triangle a(i, j) gives (i >= j), and log det = 2 sum log L_jj. False where a pivot is
// not positive or not finite.
template <class A>
bool cholesky(std::int64_t d, A&& a, std::vector<DD>& L, DD& logdet) {
    L.assign(static_cast<std::size_t>(d * d), DD{0.0, 0.0});
    logdet = DD{0.0, 0.0};
    for (std::int64_t j = 0; j < d; ++j) {
        DD s{a(j, j), 0.0};
        for (std::int64_t k = 0; k < j; ++k) s = s - L[j * d + k] * L[j * d + k];
        if (!(s.hi > 0.0) || !std::isfinite(s.hi)) return false;
        const DD ljj = sqrt(s);
        L[j * d + j] = ljj;
        logdet = logdet + dd::log(ljj) * 2.0;
        for (std::int64_t i = j + 1; i < d; ++i) {
            DD t{a(i, j), 0.0};
            for (std::int64_t k = 0; k < j; ++k) t = t - L[i * d + k] * L[j * d + k];
            if (!std::isfinite(t.hi)) return false;
            L[i * d + j] = t / ljj;
        }
    }
    return true;
}

// b <- L^-1 b (forward substitution), L lower d x d row-major, b a vector of d (stride s).
inline void solve_lower(const std::vector<DD>& L, std::int64_t d, DD* b, std::int64_t s = 1) {
    for (std::int64_t i = 0; i < d; ++i) {
        DD t = b[i * s];
        for (std::int64_t k = 0; k < i; ++k) t = t - L[i * d + k] * b[k * s];
        b[i * s] = t / L[i * d + i];
    }
}

// The inverse of L L^T (d x d, row-major, full): L^-T L^-1 from the columns of L^-1.
inline void inverse_from_cholesky(const std::vector<DD>& L, std::int64_t d, std::vector<DD>& inv) {
    std::vector<DD> Li(static_cast<std::size_t>(d * d), DD{0.0, 0.0});  // L^-1, lower
    for (std::int64_t c = 0; c < d; ++c) {
        std::vector<DD> e(static_cast<std::size_t>(d), DD{0.0, 0.0});
        e[c] = DD{1.0, 0.0};
        solve_lower(L, d, e.data());
        for (std::int64_t i = 0; i < d; ++i) Li[i * d + c] = e[i];
    }
    inv.assign(static_cast<std::size_t>(d * d), DD{0.0, 0.0});
    for (std::int64_t i = 0; i < d; ++i)
        for (std::int64_t j = 0; j <= i; ++j) {
            DD s{0.0, 0.0};
            for (std::int64_t k = i; k < d; ++k) s = s + Li[k * d + i] * Li[k * d + j];
            inv[i * d + j] = s;
            inv[j * d + i] = s;
        }
}

// log Gamma_p(a) = p (p - 1) / 4 log pi + sum_{j=1}^p log Gamma(a + (1 - j) / 2), double-double.
inline DD log_mvgamma(std::int64_t p, double a) {
    DD s = num::kLogPi * (0.25 * static_cast<double>(p * (p - 1)));
    for (std::int64_t j = 1; j <= p; ++j) s = s + num::log_gamma(dd::two_sum(a, 0.5 * static_cast<double>(1 - j)));
    return s;
}

}  // namespace sd::stats::ddla
