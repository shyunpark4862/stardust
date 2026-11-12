// Private functions that exercise the engine before any real function exists
// (ROADMAP.md S0): one per shape the dispatcher must handle, and a continued
// fraction with a known value. Bound in stardust._core.special with a leading
// underscore; tests/special/test_engine.py uses them.
#pragma once

#include "special/cf.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"

#include <array>
#include <cmath>
#include <utility>

namespace sd::special {

// Three inputs, one output.
struct Fma3 : Real {
    static constexpr const char* name = "_fma3";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"a", "b", "c"};
    static constexpr double cost_ns = cost::kFma3;
    static double f(double a, double b, double c) { return a * b + c; }  // two roundings: not std::fma
};

// Four inputs, one output: five operands, the most the iterator takes.
struct Sum4 : Real {
    static constexpr const char* name = "_sum4";
    static constexpr int nin = 4, nout = 1;
    static constexpr std::array<const char*, 4> args{"a", "b", "c", "d"};
    static constexpr double cost_ns = cost::kSum4;
    static double f(double a, double b, double c, double d) { return ((a + b) + c) + d; }
};

// One input, four outputs: the shape of airy (S6).
struct Powers4 : Real {
    static constexpr const char* name = "_powers4";
    static constexpr int nin = 1, nout = 4;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kPowers4;
    static std::array<double, 4> f(double x) {
        const double x2 = x * x;
        return {x, x2, x2 * x, x2 * x2};
    }
};

// tan(x) as Lambert's continued fraction, through lentz():
//     tan x = x / (1 - x^2 / (3 - x^2 / (5 - ...)))
// The denominator is the fraction b0 = 1, a_n = -x^2, b_n = 2n + 1. (Handing
// lentz() the whole fraction, b0 = 0 and a1 = x, starts it at `tiny` and
// loses every |x| below about 1e-301 -- see special/cf.hpp.) Accurate to a
// few ULP for |x| < pi/2; it checks lentz() and, through
// tools/special_ref.py, the whole accuracy pipeline.
struct CfTan : Real {
    static constexpr const char* name = "_cf_tan";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kCfTan;
    static double f(double x) {
        if (!std::isfinite(x)) return kNaN;
        if (x == 0.0) return x;  // keeps the sign of zero
        const double x2 = x * x;
        auto term = [&](int n) { return std::pair{-x2, 2.0 * n + 1.0}; };
        return x / lentz(1.0, term, 200).value;
    }
};

}  // namespace sd::special
