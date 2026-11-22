// Continued fractions without a division per term, for the incomplete gamma and beta
// functions (S4: special/ops/incgamma.hpp, incbeta.hpp, noncentral.hpp):
//
//     f = b0 + a1 / (b1 + a2 / (b2 + a3 / (b3 + ...)))
//
// The same two passes as special/cf.hpp's lentz_backward -- a forward pass that finds how
// many terms are needed, a backward pass over them (and four more) for the value, which damps
// each step's rounding where the forward ratios multiply theirs together -- in other
// arithmetic:
//   - forward, the convergents A_n / B_n by their three-term recurrence
//         A_n = b_n A_(n-1) + a_n A_(n-2),  B_n = b_n B_(n-1) + a_n B_(n-2)
//     (two fma each, kept within 2^+-256 by exact powers of two), with Lentz's stopping test
//     |f_n / f_(n-1) - 1| <= tol written as |A_n B_(n-1) - A_(n-1) B_n| <= tol |A_(n-1) B_n|
//     (the difference with one rounding: fma). Lentz takes two divisions a term, on its
//     critical path (c_n = b_n + a_n / c_(n-1), d_n = 1 / (b_n + a_n d_(n-1)));
//   - backward, the tail t_k = a_k / (b_k + t_(k+1)) as a quotient P_k / Q_k,
//         P_k = a_k Q_(k+1),  Q_k = b_k Q_(k+1) + P_(k+1),
//     one division at the end instead of one a term (each step rounds P and Q once: the
//     quotient carries the same two roundings a step as the division form, damped alike).
// The terms' own divisions are independent of these chains, so they overlap.
//
// The convergents are kept within 2^+-256 (so the test's products stay finite), which holds a
// step's growth for coefficients up to 2^200 (a_n up to 2^400); a fraction with larger ones (the
// incomplete gamma's at x = 1e200, in sd.stats' deep inverses) goes to lentz_backward, whose
// ratios need no such bound. So does a test product below 2^-900 (a tail far below the
// convergents' scale).
#pragma once

#include "special/cf.hpp"
#include "special/common.hpp"

#include <cmath>
#include <utility>

namespace sd::special {

namespace cf_fast_detail {

// v and w brought back near 1 by one exact power of two when the larger leaves 2^+-256
// (0 and non-finite values are left: the caller's tests see them)
inline void rescale(double& v0, double& v1, double& w0, double& w1) {
    const double m = std::fmax(std::fabs(v0), std::fabs(w0));
    if (m > 0x1p256 || (m < 0x1p-256 && m > 0.0)) {
        const int e = std::ilogb(m);
        v0 = std::ldexp(v0, -e), v1 = std::ldexp(v1, -e), w0 = std::ldexp(w0, -e), w1 = std::ldexp(w1, -e);
    }
}

}  // namespace cf_fast_detail

// lentz_backward's result (value, terms, converged) for `term(n)` = {a_n, b_n}, n = 1, 2, ...
template <class Term>
CfResult cf_fast(double b0, Term&& term, int max_terms, double tol = kEps) {
    constexpr int kKept = 64;
    constexpr double kBig = 0x1p200;
    if (!(std::fabs(b0) < kBig)) return lentz_backward(b0, term, max_terms);
    double as[kKept + 1], bs[kKept + 1];
    double A1 = 1.0, A0 = b0, B1 = 0.0, B0 = 1.0;  // (A_(n-2), A_(n-1)), (B_(n-2), B_(n-1))
    cf_fast_detail::rescale(B0, B1, A0, A1);
    int n = 1;
    bool converged = false;
    for (; n <= max_terms; ++n) {
        const auto [a, b] = term(n);
        if (!(std::fabs(b) < kBig && std::fabs(a) < kBig * kBig)) return lentz_backward(b0, term, max_terms);
        if (n <= kKept) as[n] = a, bs[n] = b;
        const double An = std::fma(b, A0, a * A1), Bn = std::fma(b, B0, a * B1);
        // |A_n B_(n-1) - A_(n-1) B_n| <= tol |A_(n-1) B_n|
        const double p = A0 * Bn, pe = std::fma(A0, Bn, -p);
        const double diff = std::fma(An, B0, -p) - pe;
        A1 = A0, A0 = An, B1 = B0, B0 = Bn;
        if (std::fabs(diff) <= tol * std::fabs(p)) {
            if (std::fabs(p) < 0x1p-900) return lentz_backward(b0, term, max_terms);
            converged = true;
            break;
        }
        if (!(std::isfinite(An) && std::isfinite(Bn))) break;  // (NaN terms: not converged)
        cf_fast_detail::rescale(B0, B1, A0, A1);
    }
    const int terms = converged ? n : max_terms;
    const int last = terms + 4;
    double P = 0.0, Q = 1.0;
    for (int k = last; k >= 1; --k) {
        double a, b;
        if (k <= kKept && k <= terms) {
            a = as[k], b = bs[k];
        } else {
            const auto ab = term(k);
            a = ab.first, b = ab.second;
        }
        const double Pn = a * Q;
        Q = std::fma(b, Q, P);
        P = Pn;
        const double m = std::fabs(Q);
        if (m > 0x1p512 || (m < 0x1p-512 && m > 0.0)) {
            const int e = std::ilogb(m);
            P = std::ldexp(P, -e), Q = std::ldexp(Q, -e);
        }
    }
    return {b0 + P / Q, terms, converged};
}

// cf_value's form: the value, or NaN if the fraction did not converge in max_terms terms.
template <class Term>
double cf_fast_value(double b0, Term&& term, int max_terms) {
    const CfResult r = cf_fast(b0, term, max_terms);
    return r.converged ? r.value : kNaN;
}

}  // namespace sd::special
