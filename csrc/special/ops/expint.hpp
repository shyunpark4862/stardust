// The exponential integrals Ei and E_n (docs/api/special.md §7,
// docs/design/special.md §5.6; S5):
//
//     expi(x)    = Ei(x) = -PV int_{-x}^inf e^-t / t dt
//     expn(n, x) = E_n(x) = int_1^inf e^(-x t) / t^n dt,  n = 0, 1, 2, ...
//
// E_n (Numerical Recipes' expint):
//     x <= 1   the power series, with psi(n) (gamma_core.hpp) for its log term
//     x >  1   e^-x / (x + n - 1 n / (x + n + 2 - 2 (n+1) / ...)), Lentz for the
//              count and the backward pass for the value (cf.hpp), e^-x as m 2^e
//              so the tail stays a subnormal value (scipy: 0 from x = 709.8)
// Large n needs nothing of its own: the fraction's b = x + n converges at once
// and the series falls like x^k / k!.
//
// Ei, x > 0:
//     x in [0.3, 0.45]  Taylor series about the zero x0 = 0.3725 in the exact
//                       d = x - x0 (coeffs/ei_taylor.hpp): the power series
//                       below cancels to nothing there (scipy: 9e4 ULP)
//     x < 40            gamma + log x + sum x^k / (k k!)  (positive terms)
//     x >= 40           e^x / x sum k! / x^k  (asymptotic, smallest term ~1e-17
//                       at 40), e^x as m 2^e: finite up to 716.4 (scipy: inf
//                       from 709.8)
// x < 0: Ei(x) = -E_1(-x).
#pragma once

#include "special/boost.hpp"

#include "special/cf.hpp"
#include "special/coeffs/ei_taylor.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/gamma_core.hpp"

#include <array>
#include <cmath>

namespace sd::special {

namespace expint_detail {

// E_n(x) for an integer n >= 1 (as double), 0 < x < inf.
inline double en(double n, double x) {
    if (x > 750.0) return 0.0;  // E_n(x) < e^-x / x: below half the smallest subnormal
    const double nm1 = n - 1.0;
    if (x <= 1.0) {
        double ans = nm1 != 0.0 ? 1.0 / nm1 : -std::log(x) - kEuler;
        double fact = 1.0;
        bool ended = false;
        for (int i = 1; i < 1000; ++i) {  // x <= 1: the terms fall like x^i / i!
            fact *= -x / i;
            double del;
            if (i != nm1) {
                del = -fact / (i - nm1);
            } else {
                const double psi = digamma_pos(n);  // psi(n) = H_(n-1) - euler
                del = fact * (-std::log(x) + psi);
            }
            ans += del;
            // (not at i = n - 1's neighbour: with n - 1 past the stop the logarithmic term is nothing)
            if (std::fabs(del) <= std::fabs(ans) * 0x1p-56) {
                ended = true;
                break;
            }
        }
        return ended ? ans : kNaN;
    }
    const double tail = cf_value(
        x + n,
        [&](int k) {
            const double kd = k;
            return std::pair{-kd * (nm1 + kd), x + n + 2.0 * kd};
        },
        100000
    );
    return value(exp_scaled(-x) * scaled(1.0 / tail));
}

// Ei(x) for 0 < x < inf.
inline double ei_pos(double x) {
    if (x >= 0.3 && x <= 0.45) {
        const double d = (x - kEiZeroHi) - kEiZeroLo;  // x - hi is exact (Sterbenz)
        return d * horner(d, kEiTaylor);
    }
    if (x < 40.0) {
        double sum = 0.0, term = 1.0;
        bool ended = false;
        for (int k = 1; k < 500; ++k) {  // x < 40: ends by k ~ 150
            term *= x / k;
            const double t = term / k;
            sum += t;
            if (t <= sum * 0x1p-56) {
                ended = true;
                break;
            }
        }
        return ended ? (kEuler + std::log(x)) + sum : kNaN;
    }
    double s = 1.0, t = 1.0;
    for (int k = 1; k < 100; ++k) {
        const double tn = t * k / x;
        if (tn >= t) break;  // the asymptotic terms start to grow
        t = tn;
        s += t;
        if (t <= s * 0x1p-56) break;
    }
    return value(exp_scaled(x) * scaled(s / x));
}

}  // namespace expint_detail

// Ei(x). expi(0) = -inf (both zeros), expi(inf) = inf, expi(-inf) = -0.0.
struct Expi : Real {
    static constexpr const char* name = "expi";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kExpi;
    static double f(double x) {
        // Boost (special/boost.hpp); the subnormal tail below, overflow and a subnormal x (its
        // log there was off by 3e-5 at 2e-323) are ours, out of line
        if (x >= -700.0 && std::fabs(x) >= 0x1p-1022 && x < kInf) [[likely]] {
            const double r = bst::expint(x);
            if (bst::normal(r)) [[likely]]
                return r;
        }
        return ours(x);
    }
    [[gnu::noinline]] static double ours(double x) {
        if (std::isnan(x)) return kNaN;
        if (x == 0.0) return -kInf;
        if (x == kInf) return kInf;
        if (x == -kInf) return -0.0;
        if (x > 0.0) return expint_detail::ei_pos(x);
        return -expint_detail::en(1.0, -x);
    }
};

// E_n(x): n an integer >= 0 (int64 or float64; Q17: anything else is NaN, also
// n = inf -- scipy gives 1.7e-10 there -- and scipy's NaN beyond int32 is a
// value here). E_n(0) = 1/(n-1) for n >= 2 and inf for n = 0, 1; E_n(inf) = 0;
// x < 0 is NaN.
struct Expn : Real {
    static constexpr const char* name = "expn";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"n", "x"};
    static constexpr double cost_ns = cost::kExpn;
    static double f(double n, double x) {
        if (std::isnan(n) || std::isnan(x) || n < 0.0 || std::isinf(n) || n != std::floor(n) || x < 0.0) return kNaN;
        if (x == 0.0) return n <= 1.0 ? kInf : 1.0 / (n - 1.0);
        if (x == kInf) return 0.0;
        if (n == 0.0) return value(exp_scaled(-x) / scaled(x));
        return expint_detail::en(n, x);
    }
};

}  // namespace sd::special
