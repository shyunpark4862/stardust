// The limit of a distribution whose shape parameter falls below 2^-1021 (user decision
// 2026-11-19, docs/design/stats.md T50): the two atoms 0 and 1 of Beta(a, b) with
// P(1) = a / (a + b) (BetaBinomial: 0 and n), whichever shape is below. With the other shape
// not small one mass is tiny (Beta(2^-1022, 2): P(1) = 2^-1023) or rounds to 0, the single
// point mass; across 2^-1021 the law is continuous (a = 2^-1022, b = 2^-1021: P(1) = 1/3).
// Every method's value of that law for the engine's limits (at_limit / limit_in /
// limit_par, stats/engine.hpp).
#pragma once

#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace sd::stats::two_point {

// The smallest shape that is not a limit: below it (as the halves of the t and F degrees of
// freedom, StudentT::halvable) the law is taken as the shape's limit at 0.
inline constexpr double kMinShape = 0x1p-1021;

// Atoms lo < hi (lo = hi for a single one: n = 0) with P(hi) = q, P(lo) = qc, both to an
// ULP or two (q + qc = 1; one of them 0 for a point mass). The family's support is [lo, hi]
// (T32: icdf(1) and iccdf(0) are hi whatever the masses).
struct Law {
    double lo, hi, q, qc;
    bool discrete;
    double mode;  // (continuous: the Beta's own rule, T21)
};

// The law of shapes a, b (a or b below kMinShape) on atoms lo, hi: q = a / (a + b),
// qc = b / (a + b), each one rounding (below 1 both shapes are scaled by 2^1000 first, exactly:
// subnormal shapes keep their bits). The mode of the continuous law is the Beta's: 0 for
// b >= 1, 1 for a >= 1, else NaN (two infinite densities; T21); of a single atom that atom.
inline Law of_shapes(double a, double b, double lo, double hi, bool discrete) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    if (lo == hi) return {lo, hi, 0.0, 1.0, discrete, lo};
    const double k = std::max(a, b) < 1.0 ? 0x1p1000 : 1.0;
    const double as = a * k, bs = b * k, s = as + bs;
    const double q = as / s, qc = bs / s;
    double mode = q == 0.0 ? lo : (qc == 0.0 ? hi : (b >= 1.0 ? lo : (a >= 1.0 ? hi : kNaN)));
    if (discrete) mode = q > qc ? hi : (q < qc ? lo : kNaN);  // the larger mass (T25: a tie is NaN)
    return {lo, hi, q, qc, discrete, mode};
}

inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// log P(lo) and log P(hi), from the smaller mass where the other is near 1
inline double log_lo(const Law& t) { return t.qc >= 0.5 ? std::log1p(-t.q) : std::log(t.qc); }
inline double log_hi(const Law& t) { return t.q >= 0.5 ? std::log1p(-t.qc) : std::log(t.q); }

template <class M>
double in(double x, const Law& t) {
    using namespace method;
    const bool at_lo = x == t.lo, at_hi = x == t.hi;
    if constexpr (std::is_same_v<M, Pdf> || std::is_same_v<M, Logpdf>) {
        // a density of atoms: inf at an atom with mass (T46's point mass), 0 elsewhere
        const bool atom = (at_lo && t.qc > 0.0) || (at_hi && t.q > 0.0);
        if constexpr (std::is_same_v<M, Pdf>) return atom ? kInf : 0.0;
        else return atom ? kInf : -kInf;
    } else if constexpr (std::is_same_v<M, Pmf>) {
        if (t.lo == t.hi) return at_lo ? 1.0 : 0.0;
        return at_lo ? t.qc : (at_hi ? t.q : 0.0);
    } else if constexpr (std::is_same_v<M, Logpmf>) {
        if (t.lo == t.hi) return at_lo ? 0.0 : -kInf;
        return at_lo ? log_lo(t) : (at_hi ? log_hi(t) : -kInf);
    } else if constexpr (std::is_same_v<M, Cdf>) {
        return x < t.lo ? 0.0 : (x < t.hi ? t.qc : 1.0);
    } else if constexpr (std::is_same_v<M, Ccdf>) {
        return x < t.lo ? 1.0 : (x < t.hi ? t.q : 0.0);
    } else if constexpr (std::is_same_v<M, Logcdf>) {
        return x < t.lo ? -kInf : (x < t.hi ? log_lo(t) : 0.0);
    } else if constexpr (std::is_same_v<M, Logccdf>) {
        return x < t.lo ? 0.0 : (x < t.hi ? log_hi(t) : -kInf);
    } else if constexpr (std::is_same_v<M, Icdf>) {  // the smallest atom with cdf >= p
        if (!(x >= 0.0 && x <= 1.0)) return kNaN;
        return x == 1.0 ? t.hi : (x <= t.qc ? t.lo : t.hi);
    } else if constexpr (std::is_same_v<M, Iccdf>) {  // the smallest atom with ccdf <= p
        if (!(x >= 0.0 && x <= 1.0)) return kNaN;
        return x == 0.0 ? t.hi : (x >= t.q ? t.lo : t.hi);
    } else if constexpr (std::is_same_v<M, Ilogcdf>) {
        if (!(x <= 0.0)) return kNaN;
        return x == 0.0 ? t.hi : (x <= log_lo(t) ? t.lo : t.hi);
    } else {
        static_assert(std::is_same_v<M, Ilogccdf>);
        if (!(x <= 0.0)) return kNaN;
        return x == -kInf ? t.hi : (x >= log_hi(t) ? t.lo : t.hi);
    }
}

template <class M>
double par(const Law& t) {
    using namespace method;
    const bool point = t.q == 0.0 || t.qc == 0.0 || t.lo == t.hi;
    const double w = t.hi - t.lo;
    if constexpr (std::is_same_v<M, Mean>) return t.lo + w * t.q;
    else if constexpr (std::is_same_v<M, Variance>) return point ? 0.0 : w * w * (t.q * t.qc);
    else if constexpr (std::is_same_v<M, StandardDeviation>) return point ? 0.0 : w * std::sqrt(t.q * t.qc);
    else if constexpr (std::is_same_v<M, Skewness>) return point ? kNaN : (t.qc - t.q) / std::sqrt(t.q * t.qc);
    else if constexpr (std::is_same_v<M, Kurtosis>) return point ? kNaN : (1.0 - 6.0 * (t.q * t.qc)) / (t.q * t.qc);
    else if constexpr (std::is_same_v<M, Median>) return in<Icdf>(0.5, t);
    else if constexpr (std::is_same_v<M, Mode>) return t.mode; else if constexpr (std::is_same_v<M, Entropy>) {
        // no density for an atom: -inf (T47); the Shannon entropy of the masses, exactly 0 for one
        if (!t.discrete) return -kInf;
        if (point) return 0.0;
        return -(t.q * log_hi(t) + t.qc * log_lo(t));
    } else if constexpr (std::is_same_v<M, SupportLo>) {
        return t.lo;
    } else {
        static_assert(std::is_same_v<M, SupportHi>);
        return t.hi;
    }
}

}  // namespace sd::stats::two_point
