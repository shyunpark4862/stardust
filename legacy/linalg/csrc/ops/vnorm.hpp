// The norm of a set of magnitudes, shared by vector_norm, matrix_norm and cond
// (DESIGN.md §2.11: fast unscaled pass, rescaled second pass only on overflow
// or when every term is tiny; empty sets have norm 0).
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace sdla {

inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// Below this, p-th powers may be subnormal and lose bits: 2^-969 = DBL_MIN * 2^53.
inline constexpr double kTiny = 0x1p-969;

enum class Ord { Inf, NegInf, Zero, One, Two, P };

struct VecOrd {
    Ord kind;
    double p;
};

// The norm of a set of magnitudes. `each(f)` calls f(double) once per
// magnitude (it may be called twice, for the scaled pass); `count` is how
// many there are. Empty sets have norm 0 (S11).
template <class Each>
inline double vnorm(VecOrd ord, std::int64_t count, Each&& each) {
    if (count == 0) return 0.0;
    switch (ord.kind) {
        case Ord::Inf: {
            double m = 0;
            bool nan = false;
            each([&](double a) {
                if (a != a) nan = true;
                else if (a > m) m = a;
            });
            return nan ? kNaN : m;
        }
        case Ord::NegInf: {
            double m = kInf;
            bool nan = false;
            each([&](double a) {
                if (a != a) nan = true;
                else if (a < m) m = a;
            });
            return nan ? kNaN : m;
        }
        case Ord::Zero: {  // the number of nonzeros (not a norm; Array API). NaN counts.
            std::int64_t c = 0;
            each([&](double a) { c += (a != 0); });
            return static_cast<double>(c);
        }
        case Ord::One: {
            double s = 0;
            each([&](double a) { s += a; });
            return s;
        }
        case Ord::Two:
        case Ord::P: {
            const bool two = ord.kind == Ord::Two;
            const double p = ord.p;
            double s = 0, amax = 0;
            bool nan = false;
            each([&](double a) {
                if (a != a) {
                    nan = true;
                    return;
                }
                s += two ? a * a : std::pow(a, p);
                if (a > amax) amax = a;
            });
            if (nan) return kNaN;
            if (p > 0) {
                if (amax == kInf) return kInf;
                const double top = two ? amax * amax : std::pow(amax, p);
                if (std::isinf(s) || (amax > 0 && top < kTiny)) {
                    // Rescaled: amax * (sum (a/amax)^p)^(1/p). Every term is <= 1.
                    double t = 0;
                    each([&](double a) {
                        const double r = a / amax;
                        t += two ? r * r : std::pow(r, p);
                    });
                    return amax * (two ? std::sqrt(t) : std::pow(t, 1 / p));
                }
            }
            return two ? std::sqrt(s) : std::pow(s, 1 / p);
        }
    }
    return kNaN;
}

}  // namespace sdla
