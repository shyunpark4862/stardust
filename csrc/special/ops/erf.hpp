// The error function family and the normal distribution (docs/api/special.md
// §2, docs/design/special.md §5.1; S2):
//   erf erfc ndtr log_ndtr        Cephes ndtr.c rationals (coeffs/cephes_ndtr.hpp)
//   ndtri erfinv erfcinv          Wichura's AS241 (coeffs/as241.hpp)
//   erfcx                         the same rationals, libm exp
//
// erf, erfc, ndtr, ndtri (and erfinv, erfcinv, log_ndtr, which share their
// cores) are chunk kernels (kernels/chunked.hpp) in three passes:
//   1. NEON: the argument of exp or log for the whole chunk
//   2. one vForce call (vvexp, vvlog)
//   3. per element: the rational of the element's own region, by branching
// Pass 3 was planned branch-free -- every region's rational in NEON lanes,
// then a select -- and measured (bench/special/s2/variants.md): with three
// regions of degree 5-8 the selects and the extra regions cost more than the
// lanes save, and the M4's branch predictor and wide out-of-order core make
// the scalar branches cheap. Branching was faster or equal on every input
// mix but uniformly random regions; the bits are the same either way.
// _erfc_select and _erfc_estrin keep the measured alternatives.
//
// exp(-x^2) without the error of x^2. Rounding x^2 to hi costs a relative
// error up to 2^-53 x^2 in exp(-hi): 2^-53 * 384 at x = 19.6, which is
// scipy's measured 124 ULP in erfc there. The rounding error of the square is
// itself a float64, lo = fma(x, x, -hi) (exact), and
//     exp(-x^2) = exp(-hi) exp(-lo) = exp(-hi) (1 - lo)
// to within lo^2 < 2^-106 x^4 -- far below an ULP. Cephes' expx2() gets the
// same with a second exp; one fma and one multiply do it here.
//
// Special values follow scipy except where docs/design/special.md §2.4 lists
// them: the subnormal tails of erfc, ndtr and log_ndtr are values, not 0;
// erfcinv(5e-324) is finite; erfinv(-0.0) is -0.0.
#pragma once

#include "special/coeffs/as241.hpp"
#include "special/coeffs/cephes_ndtr.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/kernels/chunked.hpp"
#include "special/kernels/simd.hpp"

#include <Accelerate/Accelerate.h>

#include <array>
#include <cmath>

namespace sd::special {

inline constexpr double kSqrtHalf = 7.07106781186547524401E-1;  // 1/sqrt(2) (Cephes SQRTH)
inline constexpr double kInvSqrtPi = 5.64189583547756286948E-1;  // 1/sqrt(pi)
inline constexpr double kLn2 = 6.93147180559945309417E-1;
inline constexpr double kHalfSqrtHalf = 3.53553390593273762200E-1;  // 1/(2 sqrt 2)

namespace erf_detail {

using simd::cst;
using simd::F2;
using simd::select;

// Calls f.template operator()<V>(i) for the chunk's elements: pairs as NEON
// lanes, the odd last one as a double.
template <class F>
void for_lanes(int n, F&& f) {
    int i = 0;
    for (; i + 2 <= n; i += 2) f.template operator()<F2>(i);
    for (; i < n; ++i) f.template operator()<double>(i);
}

// ---- Cephes erf / erfc, one element ---------------------------------------

// erf(t) for |t| < 1: t T(t^2) / U(t^2).
inline double erf_small(double t) {
    const double z = t * t;
    return t * simd::horner(z, cephes::kNdtrT) / simd::horner1(z, cephes::kNdtrU);
}

// erfc(a) for a >= 1, given e0 = exp(-hi) and the correction c = lo (a^2 =
// hi + lo, or the same for x^2 / 2 in ndtr): (e0 (p - p c)) / q with p / q =
// P/Q below 8, R/S from 8. Cephes' order (z p) / q keeps more bits when e0 is
// subnormal. 0 where e0 is 0 -- also for a = inf, where p, q are inf and c NaN.
inline double erfc_tail(double a, double e0, double c) {
    if (e0 == 0.0) return 0.0;
    double p, q;
    if (a >= 8.0) {
        p = simd::horner(a, cephes::kNdtrR);
        q = simd::horner1(a, cephes::kNdtrS);
    } else {
        p = simd::horner(a, cephes::kNdtrP);
        q = simd::horner1(a, cephes::kNdtrQ);
    }
    return (e0 * (p - p * c)) / q;
}

// Pass 1 of erf and erfc: the argument of exp (-hi) and the correction (lo).
template <class V>
void square_split(const double* x, double* arg, double* lo, int i) {
    const V v = simd::load(x + i, V{});
    const V hi = v * v;
    simd::store(arg + i, -hi);
    simd::store(lo + i, simd::fma(v, v, -hi));
}

// Passes 1 and 2 of erf and erfc: e = exp(-hi), lo.
inline void exp_minus_square(const double* x, double* e, double* lo, int n) {
    alignas(16) double arg[kChunk];
    for_lanes(n, [&]<class V>(int i) { square_split<V>(x, arg, lo, i); });
    vvexp(e, arg, &n);
}

// ---- AS241, one element ---------------------------------------------------

// From q = p - 1/2 and L = log(min(p, 1 - p)): in the central region
// (|q| <= 0.425) the value is q * ratio, where the caller multiplies q in
// itself (erfinv must not form q = y / 2: it underflows for a subnormal y);
// elsewhere `tail`, negated for q < 0, infinite for L = -inf (p = 0 or 1).
struct NdtriParts {
    bool central;
    double ratio, tail;
};
inline NdtriParts ndtri_parts(double q, double L) {
    if (std::fabs(q) <= as241::kSplit1) {
        const double r = as241::kConst1 - q * q;
        return {true, simd::horner(r, as241::kA) / simd::horner(r, as241::kB), 0.0};
    }
    double t;
    if (L == -kInf) {
        t = kInf;
    } else {
        const double s = std::sqrt(-L);  // NaN for L NaN: p outside [0, 1]
        if (s <= as241::kSplit2) {
            const double s1 = s - as241::kConst2;
            t = simd::horner(s1, as241::kC) / simd::horner(s1, as241::kD);
        } else {
            const double s2 = s - as241::kSplit2;
            t = simd::horner(s2, as241::kE) / simd::horner(s2, as241::kF);
        }
    }
    return {false, 0.0, q < 0.0 ? -t : t};
}

// ---- The measured alternatives of pass 3 (erfc only) -----------------------

struct Horner {
    template <class V, std::size_t N> static V p(V x, const std::array<double, N>& c) { return simd::horner(x, c); }
    template <class V, std::size_t N> static V p1(V x, const std::array<double, N>& c) { return simd::horner1(x, c); }
};
struct Estrin {
    template <class V, std::size_t N> static V p(V x, const std::array<double, N>& c) { return simd::estrin(x, c); }
    template <class V, std::size_t N> static V p1(V x, const std::array<double, N>& c) { return simd::estrin1(x, c); }
};

// Every region's rational in NEON lanes, then selected: no branches.
template <class P, class V>
V erfc_select(V x, V e0, V c) {
    const V a = simd::fabs(x);
    const V z = x * x;
    const V small = cst<V>(1.0) - x * P::p(z, cephes::kNdtrT) / P::p1(z, cephes::kNdtrU);
    const auto big = a >= cst<V>(8.0);
    const V p = select(big, P::p(a, cephes::kNdtrR), P::p(a, cephes::kNdtrP));
    const V q = select(big, P::p1(a, cephes::kNdtrS), P::p1(a, cephes::kNdtrQ));
    V y = (e0 * (p - p * c)) / q;
    y = select(e0 == cst<V>(0.0), cst<V>(0.0), y);
    return select(a < cst<V>(1.0), small, select(x < cst<V>(0.0), cst<V>(2.0) - y, y));
}

template <class P>
void erfc_select_chunk(const double* x, double* o, int n) {
    alignas(16) double lo[kChunk], e[kChunk];
    exp_minus_square(x, e, lo, n);
    for_lanes(n, [&]<class V>(int i) {
        simd::store(o + i, erfc_select<P>(simd::load(x + i, V{}), simd::load(e + i, V{}), simd::load(lo + i, V{})));
    });
}

}  // namespace erf_detail

// erf(x): x T(x^2) / U(x^2) for |x| < 1, else sign(x) (1 - erfc(|x|)).
struct Erf : Real {
    static constexpr const char* name = "erf";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kErf;
    static void chunk(const double* x, double* o, int n) {
        using namespace erf_detail;
        alignas(16) double lo[kChunk], e[kChunk];
        exp_minus_square(x, e, lo, n);
        for (int i = 0; i < n; ++i) {
            const double a = std::fabs(x[i]);
            if (a < 1.0) {
                o[i] = erf_small(x[i]);
            } else {
                const double v = 1.0 - erfc_tail(a, e[i], lo[i]);  // NaN stays NaN
                o[i] = x[i] < 0.0 ? -v : v;
            }
        }
    }
};

// erfc(x): 1 - erf(x) for |x| < 1, exp(-x^2) P/Q or R/S beyond, 2 - erfc(-x)
// for x <= -1. Subnormal from x = 26.55 to 27.23 (scipy: 0 from 26.7).
struct Erfc : Real {
    static constexpr const char* name = "erfc";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kErfc;
    static void chunk(const double* x, double* o, int n) {
        using namespace erf_detail;
        alignas(16) double lo[kChunk], e[kChunk];
        exp_minus_square(x, e, lo, n);
        for (int i = 0; i < n; ++i) {
            const double a = std::fabs(x[i]);
            if (a < 1.0) {
                o[i] = 1.0 - erf_small(x[i]);
            } else {
                const double y = erfc_tail(a, e[i], lo[i]);
                o[i] = x[i] < 0.0 ? 2.0 - y : y;
            }
        }
    }
};

// The measured alternatives of erfc's pass 3 (bench/special/s2_variants.py),
// private: every region in NEON lanes then select (same bits as erfc), and
// the same with Estrin's scheme (different bits).
struct ErfcSelect : Real {
    static constexpr const char* name = "_erfc_select";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kErfc;
    static void chunk(const double* x, double* o, int n) {
        erf_detail::erfc_select_chunk<erf_detail::Horner>(x, o, n);
    }
};
struct ErfcEstrin : Real {
    static constexpr const char* name = "_erfc_estrin";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kErfc;
    static void chunk(const double* x, double* o, int n) {
        erf_detail::erfc_select_chunk<erf_detail::Estrin>(x, o, n);
    }
};

// ndtr(x) = erfc(-x / sqrt 2) / 2, with t = x / sqrt 2:
//     |t| < 1:  1/2 + erf(t) / 2
//     else:     y = exp(-x^2 / 2) P(|t|) / Q(|t|) / 2 (R/S from 8);  x < 0: y,  x > 0: 1 - y
// exp(-t^2) is taken as exp(-x^2 / 2) from x itself, split as above: t's own
// rounding would cost 2^-53 * 2 t^2 (scipy: 61 ULP at x = -17.5). The
// rationals are insensitive to it (d log(P/Q)/dt ~ -1/t).
struct Ndtr : Real {
    static constexpr const char* name = "ndtr";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kNdtr;
    static void chunk(const double* x, double* o, int n) {
        using namespace erf_detail;
        alignas(16) double arg[kChunk], lo[kChunk], e[kChunk];
        for_lanes(n, [&]<class V>(int i) {
            const V v = simd::load(x + i, V{});
            const V hi = v * v;
            simd::store(arg + i, cst<V>(-0.5) * hi);
            simd::store(lo + i, cst<V>(0.5) * simd::fma(v, v, -hi));
        });
        vvexp(e, arg, &n);
        for (int i = 0; i < n; ++i) {
            const double t = x[i] * kSqrtHalf;
            const double a = std::fabs(t);
            if (a < 1.0) {
                o[i] = 0.5 + 0.5 * erf_small(t);
            } else {
                const double y = 0.5 * erfc_tail(a, e[i], lo[i]);
                o[i] = x[i] < 0.0 ? y : 1.0 - y;
            }
        }
    }
};

// erfcx(x) = exp(x^2) erfc(x), scalar on libm (not a NEON target, Q5).
// Three regions:
//     |x| < 1:     exp(x^2) (1 - erf(x))
//     1 <= x < 8:  P(x)/Q(x) -- erfc's rational without the exp
//     x >= 8:      the asymptotic series
//                  1/(x sqrt pi) (1 - u + 3u^2 - 15u^3 + ...), u = 1/(2x^2),
//                  17 terms (at x = 8 the 18th is 1e-17 relative)
//     x <= -1:     2 exp(x^2) - erfcx(-x)  (exp(x^2) split as above;
//                  inf from x = -26.63)
// Cephes' R/S, which erfc uses from 8, is fitted only to x = 26.64 (where
// erfc underflows); carried further it measured 21 ULP near x = 1e4.
struct Erfcx : Real {
    static constexpr const char* name = "erfcx";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kErfcx;
    static double asymptotic(double x) {  // x >= 8
        const double u = 0.5 / (x * x);  // 0 once x * x overflows: the series is then 1
        double sum = 1.0;
        for (int k = 17; k >= 1; --k) sum = 1.0 - (2 * k - 1) * u * sum;  // nested: 1 - u (1 - 3u (1 - 5u (...)))
        return (kInvSqrtPi / x) * sum;
    }
    static double positive(double x) {  // x >= 1
        if (x >= 8.0) return asymptotic(x);
        return simd::horner(x, cephes::kNdtrP) / simd::horner1(x, cephes::kNdtrQ);
    }
    static double f(double x) {
        if (std::isnan(x)) return x;
        if (x >= 1.0) return positive(x);
        if (x > -1.0) return std::exp(x * x) * (1.0 - erf_detail::erf_small(x));
        const double hi = x * x;
        const double e = std::exp(hi);
        if (e == kInf) return kInf;
        return 2.0 * (e + e * std::fma(x, x, -hi)) - positive(-x);
    }
};

// log(ndtr(x)) without forming ndtr where it would lose the answer:
//     x >= 0:  log1p(-ndtr(-x))                 ndtr(-x) <= 1/2 is accurate
//     x <  0:  -x^2/2 + log(erfcx(-x / sqrt 2) / 2)
// The second needs neither the subnormal ndtr nor an asymptotic series: the
// two terms have the same sign, and erfcx never underflows. (The design's
// three regions became two; docs/learn/S2-erf.md.)
struct LogNdtr : Real {
    static constexpr const char* name = "log_ndtr";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kLogNdtr;
    static void chunk(const double* x, double* o, int n) {
        alignas(16) double neg[kChunk], nd[kChunk];
        for (int i = 0; i < n; ++i) neg[i] = -std::fabs(x[i]);
        Ndtr::chunk(neg, nd, n);
        for (int i = 0; i < n; ++i) {
            const double v = x[i];
            // -(0.5 v) v: v * v would overflow for |v| > 1.34e154 while -v^2/2 still fits
            o[i] = v >= 0.0 ? std::log1p(-nd[i]) : -(0.5 * v) * v + std::log(0.5 * Erfcx::f(-v * kSqrtHalf));
        }
    }
};

// ndtri(p): AS241 with q = p - 1/2 and L = log(min(p, 1 - p)); 1 - p is
// exact for p >= 1/2 (Sterbenz).
struct Ndtri : Real {
    static constexpr const char* name = "ndtri";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"p"};
    static constexpr double cost_ns = cost::kNdtri;
    static void chunk(const double* p, double* o, int n) {
        using namespace erf_detail;
        alignas(16) double m[kChunk], L[kChunk];
        for_lanes(n, [&]<class V>(int i) {
            const V v = simd::load(p + i, V{});
            simd::store(m + i, select(v < cst<V>(0.5), v, cst<V>(1.0) - v));
        });
        vvlog(L, m, &n);
        for (int i = 0; i < n; ++i) {
            const double q = p[i] - 0.5;
            const NdtriParts r = ndtri_parts(q, L[i]);
            o[i] = r.central ? q * r.ratio : r.tail;
        }
    }
};

// erfinv(y) = ndtri((1 + y) / 2) / sqrt 2, fed to AS241 as q = y / 2 and
// min(p, 1 - p) = (1 - |y|) / 2 (exact for |y| >= 1/2): (1 + y) is never
// formed. The central value is y (ratio / (2 sqrt 2)), not (y / 2) ratio /
// sqrt 2: halving a subnormal y would drop its last bit (erfinv(5e-324)
// came out 0). erfinv(-0.0) = -0.0.
struct Erfinv : Real {
    static constexpr const char* name = "erfinv";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"y"};
    static constexpr double cost_ns = cost::kErfinv;
    static void chunk(const double* y, double* o, int n) {
        using namespace erf_detail;
        alignas(16) double m[kChunk], L[kChunk];
        for_lanes(n, [&]<class V>(int i) {
            const V v = simd::load(y + i, V{});
            simd::store(m + i, cst<V>(0.5) * (cst<V>(1.0) - simd::fabs(v)));
        });
        vvlog(L, m, &n);
        for (int i = 0; i < n; ++i) {
            const NdtriParts r = ndtri_parts(0.5 * y[i], L[i]);
            o[i] = r.central ? y[i] * (r.ratio * kHalfSqrtHalf) : r.tail * kSqrtHalf;
        }
    }
};

// erfcinv(z) = -ndtri(z / 2) / sqrt 2, as q = (z - 1) / 2 and
// log(min(z, 2 - z) / 2) = log(min(z, 2 - z)) - log 2: z / 2 is never
// formed, so z = 5e-324 gives 27.21 (scipy: z / 2 = 0, inf). The sign is
// applied last, so erfcinv(1) = -0.0 as in scipy.
struct Erfcinv : Real {
    static constexpr const char* name = "erfcinv";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"z"};
    static constexpr double cost_ns = cost::kErfcinv;
    static void chunk(const double* z, double* o, int n) {
        using namespace erf_detail;
        alignas(16) double m[kChunk], L[kChunk];
        for_lanes(n, [&]<class V>(int i) {
            const V v = simd::load(z + i, V{});
            const V w = cst<V>(2.0) - v;
            simd::store(m + i, select(v < w, v, w));
        });
        vvlog(L, m, &n);
        for (int i = 0; i < n; ++i) {
            const double d = z[i] - 1.0;  // 2q; exact for z in [1/2, 2]
            const NdtriParts r = ndtri_parts(0.5 * d, L[i] - kLn2);
            o[i] = -(r.central ? d * (r.ratio * kHalfSqrtHalf) : r.tail * kSqrtHalf);
        }
    }
};

}  // namespace sd::special
