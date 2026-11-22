// Building blocks of the chunk kernels (stats/engine.hpp, docs/design/stats.md §3.4): whole-
// chunk elementary functions whose result for an element does not depend on its position or
// the call's length -- exp from stats/vmath.hpp (NEON, 1.003 units against Arb), log and pow
// through vForce (accelerate/vforce.hpp: checked; expm1 and log1p failed and are libm loops
// here) -- and the fallback pass.
//
// Error bounds are in units of kUlp = 2^-53 (one correctly rounded operation: 1 unit; a libm
// call: 2). vForce is looser than libm, measured against double-double references on 1.6e7
// points (bench/stats/opt/a/vforce_error.cpp at commit 4151d2a): vvlog up to 3.78 (libm: 1.00
// on the same points; vvexp, no longer used, 2.06). The bounds use kVfExp (kept at 2.5 for
// ck::exp) and kVfLog; kVf is the larger, for a bound that does not care which.
#pragma once

#include "accelerate/vforce.hpp"
#include "stats/engine.hpp"
#include "stats/vmath.hpp"

#include <cmath>

namespace sd::stats::ck {

inline constexpr double kVfExp = 2.5;
inline constexpr double kVfLog = 4.0;
inline constexpr double kVf = kVfLog;

// out may be the input.
// exp: stats/vmath.hpp (0.52 ULP; measured against Arb on 6e4 points over [-708, 708]: 1.003 units,
// vvexp 1.96 on the same points), kept at kVfExp in the bounds.
inline void exp(double* out, const double* x, int n) { vm::exp(out, x, n); }
inline void log(double* out, const double* x, int n) { vvlog(out, x, &n); }
inline void pow(double* out, const double* base, const double* e, int n) { vvpow(out, e, base, &n); }  // base^e
inline void expm1(double* out, const double* x, int n) {
    for (int i = 0; i < n; ++i) out[i] = std::expm1(x[i]);
}
inline void log1p(double* out, const double* x, int n) {
    for (int i = 0; i < n; ++i) out[i] = std::log1p(x[i]);
}

// out[i] = precise(i) where !fast[i] (fast: the element's bound is inside kFastBudget and it
// is not a special case; a NaN anywhere must leave fast[i] false or be harmless).
template <class F>
inline void fallback(int n, const bool* fast, double* out, F&& precise) {
    bool all = true;  // (a vectorized pass first: most chunks have no element to redo)
    for (int i = 0; i < n; ++i) all &= fast[i];
    if (all) return;
    for (int i = 0; i < n; ++i)
        if (!fast[i]) out[i] = precise(i);
}

}  // namespace sd::stats::ck
