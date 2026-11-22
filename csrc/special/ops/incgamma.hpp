// The regularised incomplete gamma functions and their inverses
// (docs/api/special.md §3, docs/design/special.md §5.3-5.5; S4):
//
//     P(a, x) = gamma(a, x) / Gamma(a)     gammainc
//     Q(a, x) = Gamma(a, x) / Gamma(a)     gammaincc        P + Q = 1
//
// Whichever of P, Q is the smaller is computed; the other is 1 minus it,
// which loses nothing (the smaller one is below ~0.7 in every region). The
// regions (Boost's gamma_incomplete_imp, DiDonato & Morris 1986):
//
//     x < 1.1 and a small     Q = (Gamma(1+a) - 1 - (x^a - 1) - a x^a S) / Gamma(1+a)
//     x < a (+ ~1)            P = D(a, x) sum x^n / ((a+1)...(a+n))       series
//     x > a                   Q = a D(a, x) / (x + 1 - a - 1(1-a) / (x + 3 - a - ...))   Lentz
//     a > 20, |x - a| < 0.4a  Temme's uniform expansion (both converge slowly there)
//
// D(a, x) = x^a e^-x / Gamma(a+1) comes from special/prefix.hpp with its
// exponent in double-double: scipy (its igam, written after Boost's) forms
// a log x - x - lgamma(a) in float64 for |x - a| > 0.4a and loses up to
// ~2000 ULP in the tails.
//
// The inverses solve P(a, x) = p (or Q(a, x) = q, whichever of p, q is the
// smaller, so its relative accuracy carries over) by Halley's method from an
// initial value; x times the derivative is a D.
#pragma once

#include "special/boost.hpp"

#include "special/cf.hpp"
#include "special/ops/cf_fast.hpp"
#include "special/coeffs/temme_igamma.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/dd.hpp"
#include "special/gamma_core.hpp"
#include "special/halley.hpp"
#include "special/ops/halley_fast.hpp"
#include "special/ops/erf.hpp"
#include "special/ops/expint.hpp"
#include "special/prefix.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace sd::special {

namespace igam_detail {

// P and Q, both rounded from the computed smaller one; `xderiv` is x P'(x) =
// x^a e^-x / Gamma(a) = a D (the inverses use it; P' itself overflows for a
// subnormal x and small a, x P' never does).
struct PQ {
    double p, q, xderiv;
};

// erfc(sqrt(E)) for E = hi + lo >= 0, via erf.hpp's pieces:
// exp(-z^2) = exp(-hi) (1 - lo) exactly as z^2 = E.
inline double erfc_sqrt(dd::DD E) {
    const double z = std::sqrt(E.hi);
    if (z < 1.0) return 1.0 - erf_detail::erf_small(z);
    return erf_detail::erfc_tail(z, std::exp(-E.hi), E.lo);
}

// lgamma(1 + a) for 0 <= a < 1 (gamma_core.hpp's Taylor series about 2).
inline double lgamma1p(double a) {
    if (a < 0.5) return lgamma_taylor2(a) - std::log1p(a);
    return lgamma_taylor2(a - 1.0);
}

// The series and the fractions below end within a few thousand terms wherever igam() sends
// an argument to them; one that has not ended by kMaxTerms gives NaN.
inline constexpr int kMaxTerms = 100000;

// Q for a < 1, x < 1.1 (Boost's tgamma_small_upper_part):
//     Gamma(a, x) = (Gamma(1+a) - x^a) / a - x^a S,  S = sum_{n>=1} (-x)^n / (n! (a+n))
// with Gamma(1+a) - 1 and x^a - 1 from expm1, so the difference keeps its
// relative accuracy as a -> 0.
inline PQ small_a_upper(double a, double x) {
    const double tg = std::expm1(lgamma1p(a));  // Gamma(1+a) - 1
    const double pm = std::expm1(a * std::log(x));  // x^a - 1
    double s = 0.0, t = -x;
    bool ended = false;
    for (int n = 1; n < 200; ++n) {  // x < 1.1: the terms fall like x^n / n!
        const double term = t / (a + n);
        s += term;
        if (std::fabs(term) <= std::fabs(s) * 0x1p-56) {
            ended = true;
            break;
        }
        t *= -x / (n + 1);
    }
    if (!ended) return {kNaN, kNaN, kNaN};
    const double xa = pm + 1.0;
    if (a < 0x1p-60) {
        // (Gamma(1+a) - 1) / a = -euler, (x^a - 1) / a = log x to O(a): one
        // rounding at the end, so a subnormal Q keeps its bits
        const double q = a * (-kEuler - std::log(x) - s);
        return {1.0 - q, q, std::exp(-x) * a};
    }
    const double q = (tg - pm - a * xa * s) / (tg + 1.0);
    return {1.0 - q, q, xa * std::exp(-x) * (a / (tg + 1.0))};
}

// sum_n x^n / ((a+1)...(a+n)) to 2^-56 of itself, NaN if not ended in kMaxTerms terms. Two terms a
// step (the second from the first's term times the product of both ratios: the dependent chain
// is one product and one sum per two terms, a third of the series' time; the ratios' divisions
// are independent of it).
inline double lower_series(double a, double x) {
    double sum = 1.0, term = 1.0, ap = a;
    for (int n = 0; n < kMaxTerms; n += 2) {
        const double r1 = x / (ap + 1.0), r2 = x / (ap + 2.0);
        ap += 2.0;
        const double t1 = term * r1;
        term *= r1 * r2;
        sum += t1 + term;
        if (term <= sum * 0x1p-56) return sum;
    }
    return kNaN;
}

// P by the series: D sum_n x^n / ((a+1)...(a+n)). x - 1/(3x) < a, so the
// terms fall at least like (x / (a + n))^n.
inline PQ series_lower(double a, double x) {
    const Scaled d = gamma_prefix(a, x);
    double sum = 1.0;
    if (d.m != 0.0) {
        sum = lower_series(a, x);
        if (!(sum == sum)) return {kNaN, kNaN, kNaN};
    }
    const double dv = value(d);
    // (the product in one rounding; through the split form only where it may leave the normals)
    const double p = dv >= 0x1p-900 ? dv * sum : value(d * scaled(sum));
    return {p, 1.0 - p, dv * a};
}

// Q by Lentz: a D / (x + 1 - a - 1(1-a) / (x + 3 - a - 2(2-a) / ...)), the
// fraction's tail evaluated on its own (cf.hpp: b0 = 0).
inline PQ cf_upper(double a, double x) {
    const Scaled d = gamma_prefix(a, x);
    double tail = 1.0;
    if (d.m != 0.0) {
        tail = cf_fast_value(
            x + 1.0 - a,
            [&](int n) {
                const double k = n;
                return std::pair{-k * (k - a), x + 2.0 * k + 1.0 - a};
            },
            kMaxTerms
        );
        if (std::isnan(tail)) return {kNaN, kNaN, kNaN};
    }
    const double dv = value(d);
    const double q = dv >= 0x1p-900 ? dv * (a / tail) : value(d * scaled(a / tail));
    return {1.0 - q, q, dv * a};
}

// Temme (DLMF 8.12.3, 8.12.4, 8.12.7): with eta = sign(x - a) sqrt(2 E / a),
//     Q = erfc(sqrt E) / 2 + R,   x >= a
//     P = erfc(sqrt E) / 2 - R,   x <  a
//     R = e^-E / sqrt(2 pi a) sum_k c_k(eta) a^-k
// c_k from coeffs/temme_igamma.hpp; terms until a^-k |c_k| is negligible.
template <std::size_t N>
inline double temme_c(double eta, const std::array<double, N>& c) {
    return horner(eta, c);
}
inline PQ temme(double a, double x, dd::DD E) {
    using namespace temme;
    const double eta = std::copysign(std::sqrt(2.0 * E.hi / a), x - a);
    const double w = 1.0 / a;
    // c_0..c_k with a^-(k+1) < 2^-62: k = 13 at a = 20, 6 at 1000, 3 from 1e6
    const int k = std::clamp(static_cast<int>(std::ceil(62.0 * kLn2 / std::log(a))) - 1, 2, 13);
    double c[14] = {};
    c[0] = temme_c(eta, kTemmeC0);
    c[1] = temme_c(eta, kTemmeC1);
    c[2] = temme_c(eta, kTemmeC2);
    switch (k) {  // fallthrough: every c_j with j <= k
        case 13: c[13] = temme_c(eta, kTemmeC13); [[fallthrough]];
        case 12: c[12] = temme_c(eta, kTemmeC12); [[fallthrough]];
        case 11: c[11] = temme_c(eta, kTemmeC11); [[fallthrough]];
        case 10: c[10] = temme_c(eta, kTemmeC10); [[fallthrough]];
        case 9: c[9] = temme_c(eta, kTemmeC9); [[fallthrough]];
        case 8: c[8] = temme_c(eta, kTemmeC8); [[fallthrough]];
        case 7: c[7] = temme_c(eta, kTemmeC7); [[fallthrough]];
        case 6: c[6] = temme_c(eta, kTemmeC6); [[fallthrough]];
        case 5: c[5] = temme_c(eta, kTemmeC5); [[fallthrough]];
        case 4: c[4] = temme_c(eta, kTemmeC4); [[fallthrough]];
        case 3: c[3] = temme_c(eta, kTemmeC3); [[fallthrough]];
        default: break;
    }
    double s = c[k];
    for (int j = k - 1; j >= 0; --j) s = s * w + c[j];
    const double e = std::exp(-E.hi) * (1.0 - E.lo);
    const double r = e * prefix_detail::kInvSqrt2Pi / std::sqrt(a) * s;
    const double half = 0.5 * erfc_sqrt(E);
    const double deriv = e / (std::sqrt(2.0 * kPi * a) * gamma_star(a)) * a;
    if (x >= a) {
        const double q = half + r;
        return {1.0 - q, q, deriv};
    }
    const double p = half - r;
    return {p, 1.0 - p, deriv};
}

// a = 1/2: P = erf(sqrt x), Q = erfc(sqrt x), the smaller computed (P below x = 0.2275,
// where erf(sqrt x) = 1/2); Q by erfc_sqrt, whose exponential is e^-x itself (sqrt's rounding
// does not enter it). x P' = sqrt(x) e^-x / sqrt(pi). (The series and the fraction take up to
// ~60 terms near x = 1 for this shape: the noncentral and central t's BGRAT, chdtr(1, x).)
inline constexpr double kInvSqrtPi = 0.56418958354775628695;
inline PQ half_shape(double x) {
    const double z = std::sqrt(x), xd = z * std::exp(-x) * kInvSqrtPi;
    if (x < 0.2275) {
        const double p = erf_detail::erf_small(z);
        return {p, 1.0 - p, xd};
    }
    const double q = erfc_sqrt(dd::DD{x, 0.0});
    return {1.0 - q, q, xd};
}

// Q for 2a an integer, 1 <= a <= kFiniteShape, in the fraction's region (x > a): the finite
// sums of positive terms
//     Q(n, x) = sum_{j < n} D(j, x),   Q(n + 1/2, x) = erfc(sqrt x) + sum_{j < n} D(j + 1/2, x)
// (D(s, x) = x^s e^-x / Gamma(s + 1)), as D(a - 1, x) (1 + (a-1)/x + (a-1)(a-2)/x^2 + ...),
// D(a - 1, x) = D(a, x) a / x: at most kFiniteShape terms against the fraction's 10 to 60
// (pdtr's integer shapes, chdtr's half-integers).
inline constexpr double kFiniteShape = 30.0;
inline bool finite_shape(double a) { return a >= 1.0 && a <= kFiniteShape && 2.0 * a == std::floor(2.0 * a); }
inline PQ finite_upper(double a, double x) {
    const Scaled d = gamma_prefix(a, x);
    const bool half = a != std::floor(a);
    // the sum in Horner's form from its last term, 1 + c/x (1 + (c-1)/x (1 + ...)), c = a - 1
    // (each step's rounding damped by the next c / x < 1; summed forward, the products' roundings
    // piled up to 3 ULP at a = 30)
    double sum = 1.0;
    for (double c = half ? 1.5 : 1.0; c <= a - 1.0; c += 1.0) sum = std::fma(c / x, sum, 1.0);
    const double dv = value(d), r = (a / x) * sum;
    double q = dv >= 0x1p-900 ? dv * r : value(d * scaled(r));
    if (half) q += erfc_sqrt(dd::DD{x, 0.0});
    return {1.0 - q, q, dv * a};
}

// P and Q for a > 0, x > 0, both finite.
inline PQ igam(double a, double x) {
    // a = 1: P = 1 - e^-x, Q = e^-x, both rounded once (expm1, exp), so log Q is -x to the last
    // bit (Poisson's logcdf(0) = -mu, the chi-square's logccdf at df = 2 = -x/2); the series' P
    // carried an ULP, which log1p(-P) turned into one of -x
    if (a == 1.0) {
        const double e = std::exp(-x);
        return {-std::expm1(-x), e, x * e};
    }
    if (a == 0.5) return half_shape(x);
    if (x < 1.1) {
        // (from x = 1/2 on, every a < 1 too: Q is the smaller side there or near it, and the
        // series' 1 - P carried up to 46 ULP of it, this form 6; measured with Arb, a in
        // (0.2, 1), x in (0.5, 1.1). Boost's rule, a <= 0.75 x, left those to the series.)
        const bool small = x < 0.5 ? a <= -0.4 / std::log(x) : (a <= 0.75 * x || a < 1.0);
        if (small) return small_a_upper(a, x);
        return series_lower(a, x);
    }
    if (a > 20.0 && std::fabs(x - a) < 0.4 * a) return temme(a, x, gamma_exponent(a, x));
    if (x - 1.0 / (3.0 * x) < a) return series_lower(a, x);
    return finite_shape(a) ? finite_upper(a, x) : cf_upper(a, x);
}

// scipy's conventions on the edges; NaN for a < 0, x < 0 or NaN.
// Returns true with `out` set when (a, x) is an edge.
inline bool igam_edge(double a, double x, bool upper, double& out) {
    if (std::isnan(a) || std::isnan(x) || a < 0.0 || x < 0.0) {
        out = kNaN;
    } else if (a == 0.0) {
        out = x == 0.0 ? kNaN : (upper ? 0.0 : 1.0);
    } else if (x == 0.0) {
        out = upper ? 1.0 : 0.0;
    } else if (std::isinf(a)) {
        out = std::isinf(x) ? kNaN : (upper ? 1.0 : 0.0);
    } else if (std::isinf(x)) {
        out = upper ? 0.0 : 1.0;
    } else {
        return false;
    }
    return true;
}

// ---- inverses ---------------------------------------------------------------

// log(F / t) and F' / F for the small tail F (P when lower, else Q), from the
// unrounded pieces: log D = -E - log(sqrt(2 pi a) Gamma*(a)) and the series or
// fraction. P = D s (P'/P = a / (x s)); Q = a D / tail (Q'/Q = -tail / x).
// -E and log t are ~-740 apart from their difference: both double-doubles.
inline std::array<double, 2> log_tail(double a, double x, bool lower, dd::DD lt) {
    const dd::DD E = gamma_exponent(a, x);
    const dd::DD lr = -E - lt;
    const double ld = lr.hi + (lr.lo - std::log(std::sqrt(2.0 * kPi * a) * gamma_star(a)));
    if (lower) {
        double sum = 1.0, term = 1.0, ap = a;
        bool ended = false;
        for (int n = 0; n < kMaxTerms; ++n) {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (term <= sum * 0x1p-56) {
                ended = true;
                break;
            }
        }
        if (!ended) return {kNaN, kNaN};
        return {ld + std::log(sum), a / (x * sum)};
    }
    const double tail = cf_fast_value(
        x + 1.0 - a,
        [&](int n) {
            const double k = n;
            return std::pair{-k * (k - a), x + 2.0 * k + 1.0 - a};
        },
        kMaxTerms
    );
    // (a / tail below the normal range has lost bits, or is 0: the logs apart)
    const double r = a / tail;
    return {ld + (r >= 0x1p-1022 ? std::log(r) : std::log(a) - std::log(tail)), -tail / x};
}

inline double ndtri1(double p) {
    double o;
    Ndtri::chunk(&p, &o, 1);
    return o;
}

// From this shape on the inverse is the normal limit's expansion (igam_inverse).
inline constexpr double kNormalShape = 1e10;

// The x with P(a, x) = p (upper: Q(a, x) = p), 0 < p < 1, a > 0 finite.
// Solves the smaller t of p, 1 - p against its own function (1 - p is exact
// for p >= 1/2). Initial values:
//   xs  P ~ x^a / Gamma(a+1) (x << a): x = (P Gamma(a+1))^(1/a). P <= x^a / Gamma(a+1),
//       so xs is below the root: a lower bound
//   xw  Wilson-Hilferty: x = a (1 - 1/(9a) + z / sqrt(9a))^3, z the normal deviate
//   xq  Q ~ x^(a-1) e^-x / Gamma(a) (x >> a): x = -log(Q Gamma(a)) + (a - 1) log x,
//       by fixed-point steps; below the root for a >= 1, above for a < 1
// x_start (finite, > 0): a start of the caller's for the Halley steps (Boost's, Gammaincinv),
// in place of the ones below; not for the deep tail, which keeps its own.
inline double igam_inverse(double a, double p, bool upper, double x_start = kNaN) {
    const bool lower = upper ? p > 0.5 : p <= 0.5;  // solve P = t (else Q = t)
    const double t = (upper == lower) ? 1.0 - p : p;
    if (a >= kNormalShape) {
        // The Cornish-Fisher expansion about the normal limit, z the deviate of the tail:
        //     x = a + z sqrt(a) + (z^2 - 1)/3 + (z^3 - 7z)/(36 sqrt(a)) - (3z^4 + 7z^2 - 16)/(810 a)
        // The next term is 9 z^5 / (38880 a^(3/2)): below 2e-20 a for every |z| <= 38.5. (The
        // iterations below would sum ~sqrt(a) terms a step.)
        const double z = lower ? ndtri1(t) : -ndtri1(t), z2 = z * z, sa = std::sqrt(a);
        return a + (z * sa + ((z2 - 1.0) / 3.0 +
                              (z * (z2 - 7.0) / (36.0 * sa) - (3.0 * z2 * z2 + 7.0 * z2 - 16.0) / (810.0 * a))));
    }
    double x;
    if (t >= 0x1p-1000 && x_start > 0.0 && x_start < kInf) {
        x = x_start;
    } else {
        const double lga1 = lgamma_pos(a + 1.0);
        const double c = 1.0 / (9.0 * a);
        if (lower) {
            x = std::exp((std::log(t) + lga1) / a);
            if (a >= 1.0) {
                const double v = 1.0 - c + ndtri1(t) * std::sqrt(c);
                if (v > 0.0) x = std::max(x, a * v * v * v);
            }
        } else {
            const double xs = std::exp((std::log1p(-t) + lga1) / a);
            const double L = std::log(t) + lgamma_pos(a);
            auto xq = [&] {
                double y = std::max(1.0, -L);
                for (int i = 0; i < 5; ++i) y = std::max(1.0, -L + (a - 1.0) * std::log(y));
                return y;
            };
            if (a >= 1.0) {
                const double v = 1.0 - c - ndtri1(t) * std::sqrt(c);
                x = std::max(xs, t < 1e-4 && a < 10.0 ? xq() : a * v * v * v);
            } else {
                x = xs < 1.0 ? xs : xq();
            }
        }
    }
    if (x == 0.0) return 0.0;  // the root underflows
    if (!std::isfinite(x)) x = std::max(1.0, a);
    if (t < 0x1p-1000) {
        // Deep tail (t near or below the subnormals): F rounded to float64 is only
        // a few digits there, so the iteration works on log F from the unrounded
        // pieces, in log x for P (x tiny) and in x for Q (x large).
        const dd::DD lt = dd::log(t);
        double step = kInf;  // the last step, relative
        for (int it = 0; it < 100; ++it) {
            const auto [g, dlf] = log_tail(a, x, lower, lt);
            if (std::isnan(g)) return kNaN;  // the series or the fraction did not end
            double xn = lower ? x * std::exp(-g / (x * dlf)) : x - g / dlf;
            if (!(xn > 0.0) || !std::isfinite(xn)) xn = lower ? 0.5 * x : 2.0 * x;
            if (std::fabs(xn - x) <= 0x1p-54 * xn) return xn;
            if (xn == 0.0) return 0.0;
            step = std::fabs(xn - x) / xn;
            x = xn;
        }
        return step <= 0x1p-48 ? x : kNaN;  // (a cycle in the last bits is the root; else no value)
    }
    return halley_fast(x, t, lower, [&](double y) {
        const PQ r = igam(a, y);
        return std::array<double, 4>{
            lower ? r.p : r.q, lower ? r.xderiv : -r.xderiv, (a - 1.0) / y - 1.0, -(a - 1.0) / (y * y)
        };
    });
}

// Boost's initial value for the inverse (find_inverse_gamma, the start of gamma_p_inv and
// gamma_q_inv: DiDonato & Morris's approximations, Temme's for large a), or NaN where it is
// not taken (a target below 1e-300 or a shape from kNormalShape: igam_inverse's own starts; a
// start outside the normal doubles). Boost's own inverses iterate from it to half the digits
// and left up to ~100 ULP at large a, which one Halley step of ours on igam took away (Boost's
// whole inverse and that step: 360 ns); from the start directly, our steps take 130 ns.
inline double boost_start(double a, double p, bool upper) {
    if (!(p >= 1e-300 && a < kNormalShape)) return kNaN;
    const double r = bst::call([&] {
        bool ten = false;
        return boost::math::detail::find_inverse_gamma<double>(
            a, upper ? 1.0 - p : p, upper ? p : 1.0 - p, bst::Policy(), &ten
        );
    });
    return bst::normal(r) && r >= 1e-300 ? r : kNaN;
}

}  // namespace igam_detail

// P(a, x). gammainc(0, x > 0) = 1, gammainc(inf, x < inf) = 0 (scipy).
struct Gammainc : Real {
    static constexpr const char* name = "gammainc";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"a", "x"};
    static constexpr double cost_ns = cost::kGammainc;
    static double f(double a, double x) {
        double out;
        if (igam_detail::igam_edge(a, x, false, out)) return out;
        return igam_detail::igam(a, x).p;
    }
};

// Q(a, x) = 1 - P(a, x).
struct Gammaincc : Real {
    static constexpr const char* name = "gammaincc";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"a", "x"};
    static constexpr double cost_ns = cost::kGammaincc;
    static double f(double a, double x) {
        double out;
        if (igam_detail::igam_edge(a, x, true, out)) return out;
        return igam_detail::igam(a, x).q;
    }
};

// The inverses, with scipy's ends: p = 0 -> 0 and p = 1 -> inf for P (the
// other way for Q) for any a >= 0; NaN for a = 0 or inf inside (0, 1), p
// outside [0, 1] or NaN. A root below the subnormals is 0 (scipy: NaN at
// gammaincinv(5e-324, 0.5)).
// The y with Q(a, y) = q for a < 2^-900 (T52): there Q(a, y) = a E1(y) (1 + O(a (1 +
// |log y|))) to the doubles for every y a double reaches, so E1(y) = c = q / a. Beyond c = 40
// (y < e^-40) E1(y) = -euler - log y + O(y): y = e^(-c - euler), 0 below the subnormals; else
// a bracketed bisection on E1 (decreasing), geometric while the bracket spans binades, then to
// adjacent doubles (at most ~130 steps). Boost's and igam's pieces lose such a shape (NaN, or
// 1e-12 off for a normal a with q < 1e-300, where Boost is not used).
// c is the quotient rounded once, not exp(log q - log a): the two logs near -710 carry 1e-13
// absolutely into c, and the root's relative error is c times c's.
inline double tiny_shape_upper_root(double a, double q) {
    const double c = q / a;  // inf past the doubles: the root e^(-c - euler) is then 0
    if (c > 40.0) return std::exp(-c - kEuler);
    double lo = 0x1p-60, hi = 800.0;  // E1(2^-60) > 40 >= c > E1(800) = 0
    for (int it = 0; it < 400; ++it) {
        const double mid = hi / lo > 2.0 ? std::sqrt(lo) * std::sqrt(hi) : 0.5 * (lo + hi);
        if (!(mid > lo && mid < hi)) break;
        (expint_detail::en(1.0, mid) > c ? lo : hi) = mid;
    }
    // the nearer of the two neighbours
    return std::fabs(expint_detail::en(1.0, lo) - c) <= std::fabs(expint_detail::en(1.0, hi) - c) ? lo : hi;
}

inline bool igam_inverse_edge(double a, double p, bool upper, double& out) {
    if (std::isnan(a) || std::isnan(p) || a < 0.0 || p < 0.0 || p > 1.0) {
        out = kNaN;
    } else if (p == 0.0) {
        out = upper ? kInf : 0.0;
    } else if (p == 1.0) {
        out = upper ? 0.0 : kInf;
    } else if (a == 0.0 || std::isinf(a)) {
        out = kNaN;
    } else {
        return false;
    }
    return true;
}

struct Gammaincinv : Real {
    static constexpr const char* name = "gammaincinv";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"a", "p"};
    static constexpr double cost_ns = cost::kGammaincinv;
    static double f(double a, double p) {
        double out;
        if (igam_inverse_edge(a, p, false, out)) return out;
        // Boost's start (special/boost.hpp: its initial approximation, often to 10 digits), then our
        // Halley steps to the last bit; subnormal roots are ours
        return igam_detail::igam_inverse(a, p, false, igam_detail::boost_start(a, p, false));
    }
};

struct Gammainccinv : Real {
    static constexpr const char* name = "gammainccinv";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"a", "q"};
    static constexpr double cost_ns = cost::kGammainccinv;
    static double f(double a, double q) {
        double out;
        if (igam_inverse_edge(a, q, true, out)) return out;
        if (a < 0x1p-900) [[unlikely]]
            return tiny_shape_upper_root(a, q);
        return igam_detail::igam_inverse(a, q, true, igam_detail::boost_start(a, q, true));  // as gammaincinv
    }
};

}  // namespace sd::special
