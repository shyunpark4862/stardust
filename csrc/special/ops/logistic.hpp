// The logistic function and its relatives, and the information-theory
// helpers (docs/api/special.md §5, docs/design/special.md §5.12; S1).
//
// expit and logit run NEON + vForce in chunks (kernels/chunked.hpp, Q5, Q16);
// the others are scalar definitions on libm, through the n-ary loop.
// Special values follow scipy (docs/api/special.md §5) except expit's
// subnormal tail (docs/design/special.md §2.4).
#pragma once

#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/kernels/chunked.hpp"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <array>
#include <cfloat>
#include <cmath>

namespace sd::special {

// expit(x) = 1 / (1 + e^-x), from e = exp(-|x|) so the exponential never
// overflows:
//     x >= 0:  1 / (1 + e)
//     x <  0:  e / (1 + e)
// Both share the denominator, so the branch is only a choice of numerator:
// one vbslq. For x below about -709.78, e / (1 + e) = e is subnormal; scipy
// returns 0 there (1 / (1 + exp(-x)) with exp(-x) = inf), we return the value.
struct Expit : Real {
    static constexpr const char* name = "expit";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kExpit;

    static void chunk(const double* x, double* o, int n) {
        alignas(16) double t[kChunk], e[kChunk];  // separate buffers: vForce is not documented in place
        int i = 0;
        for (; i + 2 <= n; i += 2) vst1q_f64(t + i, vnegq_f64(vabsq_f64(vld1q_f64(x + i))));
        for (; i < n; ++i) t[i] = -std::fabs(x[i]);
        vvexp(e, t, &n);
        const float64x2_t one = vdupq_n_f64(1.0);
        i = 0;
        for (; i + 2 <= n; i += 2) {
            const float64x2_t v = vld1q_f64(x + i), ev = vld1q_f64(e + i);
            const float64x2_t num = vbslq_f64(vcgezq_f64(v), one, ev);  // NaN: not >= 0, takes e (NaN)
            vst1q_f64(o + i, vdivq_f64(num, vaddq_f64(one, ev)));
        }
        for (; i < n; ++i) o[i] = (x[i] >= 0.0 ? 1.0 : e[i]) / (1.0 + e[i]);
    }
};

// logit(p) = log(p / (1 - p)), in two regions:
//     0.25 <= p <= 0.75:  2 atanh(2p - 1)
//     elsewhere:          log(p / (1 - p))
// Near p = 1/2 the ratio is near 1 and its rounding error would dominate the
// small log; 2p - 1 is exact there (Sterbenz: 2p in [1/2, 2]), so atanh sees
// the exact argument. Outside, |log(p / (1 - p))| >= log 3, and the ratio's
// two roundings cost at most a couple of ULP. (The design's log(p) -
// log1p(-p) would need log1p, which vForce gets wrong at -0 --
// docs/design/array.md §5.4.) Both are computed for the whole chunk and
// selected per element: two vForce calls, no branches. Special values fall
// out: p = +-0 -> log(+-0) = -inf, p = 1 -> log(inf) = inf, p outside [0, 1]
// or NaN -> a negative or NaN ratio -> NaN.
struct Logit : Real {
    static constexpr const char* name = "logit";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"p"};
    static constexpr double cost_ns = cost::kLogit;

    static void chunk(const double* p, double* o, int n) {
        alignas(16) double a[kChunk], r[kChunk], ta[kChunk], lr[kChunk];
        const float64x2_t one = vdupq_n_f64(1.0), two = vdupq_n_f64(2.0);
        int i = 0;
        for (; i + 2 <= n; i += 2) {
            const float64x2_t v = vld1q_f64(p + i);
            vst1q_f64(a + i, vsubq_f64(vmulq_f64(two, v), one));
            vst1q_f64(r + i, vdivq_f64(v, vsubq_f64(one, v)));
        }
        for (; i < n; ++i) {
            a[i] = 2.0 * p[i] - 1.0;
            r[i] = p[i] / (1.0 - p[i]);
        }
        vvatanh(ta, a, &n);
        vvlog(lr, r, &n);
        const float64x2_t lo = vdupq_n_f64(0.25), hi = vdupq_n_f64(0.75);
        i = 0;
        for (; i + 2 <= n; i += 2) {
            const float64x2_t v = vld1q_f64(p + i);
            const uint64x2_t central = vandq_u64(vcgeq_f64(v, lo), vcleq_f64(v, hi));
            vst1q_f64(o + i, vbslq_f64(central, vmulq_f64(two, vld1q_f64(ta + i)), vld1q_f64(lr + i)));
        }
        for (; i < n; ++i) o[i] = (p[i] >= 0.25 && p[i] <= 0.75) ? 2.0 * ta[i] : lr[i];
    }
};

// log(expit(x)) without forming expit, from e = exp(-|x|) (as expit):
//     x >= 0:  -log1p(e)
//     x <  0:  x - log1p(e)     (both terms negative: no cancellation)
// NEON + vForce in chunks (vvexp, then vvlog1p on e >= +0, where vForce's log1p is right; the
// -0 it gets wrong, docs/design/array.md §5.4, does not occur): 2.5x the scalar libm loop.
// x = +inf gives -0, x = -inf gives -inf, NaN gives NaN (as before).
struct LogExpit : Real {
    static constexpr const char* name = "log_expit";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kLogExpit;

    static void chunk(const double* x, double* o, int n) {
        alignas(16) double t[kChunk], e[kChunk], l[kChunk];
        int i = 0;
        for (; i + 2 <= n; i += 2) vst1q_f64(t + i, vnegq_f64(vabsq_f64(vld1q_f64(x + i))));
        for (; i < n; ++i) t[i] = -std::fabs(x[i]);
        vvexp(e, t, &n);
        vvlog1p(l, e, &n);
        i = 0;
        for (; i + 2 <= n; i += 2) {
            const float64x2_t v = vld1q_f64(x + i), lv = vld1q_f64(l + i);
            // x >= 0: -l; else x - l (NaN: not >= 0, x - l is NaN)
            vst1q_f64(o + i, vbslq_f64(vcgezq_f64(v), vnegq_f64(lv), vsubq_f64(v, lv)));
        }
        for (; i < n; ++i) o[i] = x[i] >= 0.0 ? -l[i] : x[i] - l[i];
    }
};

// x log(y), and 0 when x = 0 and y is not NaN (scipy's convention: also for
// y <= 0 and y = inf). Only the product is rounded beyond libm's log.
struct Xlogy : Real {
    static constexpr const char* name = "xlogy";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"x", "y"};
    static constexpr double cost_ns = cost::kXlogy;
    static double f(double x, double y) {
        if (x == 0.0 && !std::isnan(y)) return 0.0;
        return x * std::log(y);
    }
};

// x log1p(y), with the same convention at x = 0.
struct Xlog1py : Real {
    static constexpr const char* name = "xlog1py";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"x", "y"};
    static constexpr double cost_ns = cost::kXlog1py;
    static double f(double x, double y) {
        if (x == 0.0 && !std::isnan(y)) return 0.0;
        return x * std::log1p(y);
    }
};

// -x log(x): 0 at x = 0, -inf for x < 0 (scipy's convention: the concave
// function extended by -inf outside its domain).
struct Entr : Real {
    static constexpr const char* name = "entr";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kEntr;
    static double f(double x) {
        if (std::isnan(x)) return x;
        if (x > 0.0) return -x * std::log(x);
        if (x == 0.0) return 0.0;
        return -kInf;
    }
};

// x log(x / y) for x, y > 0; 0 for x = 0, y >= 0; inf otherwise (scipy's
// convention, the convex function extended by +inf). Three formulas:
//     x / y in [1/2, 2]:  x log1p((x - y) / y)    x - y is exact (Sterbenz);
//                         log(x / y) would carry the ratio's rounding error,
//                         unbounded relative to a log near 0
//     ratio normal:       x log(x / y)
//     ratio 0, subnormal or inf:  x (log x - log y)   the ratio under- or
//                         overflowed, the logs do not
struct RelEntr : Real {
    static constexpr const char* name = "rel_entr";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"x", "y"};
    static constexpr double cost_ns = cost::kRelEntr;
    static double f(double x, double y) {
        if (std::isnan(x) || std::isnan(y)) return kNaN;
        if (x > 0.0 && y > 0.0) {
            const double r = x / y;
            if (r >= 0.5 && r <= 2.0) return x * std::log1p((x - y) / y);
            if (r >= DBL_MIN && r <= DBL_MAX) return x * std::log(r);
            return x * (std::log(x) - std::log(y));
        }
        if (x == 0.0 && y >= 0.0) return 0.0;
        return kInf;
    }
};

}  // namespace sd::special
