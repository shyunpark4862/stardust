// What every special function shares (docs/design/special.md §3.2): the shape
// of an operation, the constants of float64, and polynomial evaluation.
//
// An operation is a struct like the array's (array/kernels/scalar_ops.hpp),
// with a fixed number of inputs and outputs:
//
//     struct Erf : Real {
//         static constexpr const char* name = "erf";
//         static constexpr int nin = 1, nout = 1;
//         static constexpr std::array<const char*, 1> args{"x"};  // names in messages
//         static constexpr double cost_ns = cost::kErf;          // special/cost.hpp
//         static double f(double x);                              // one element
//     };
//
// f is a pure function: no global state, errno never read. Every input is a
// float64 (docs/design/special.md Q6): bool and int64 arrays are converted on
// the way in, complex has no loop -- except the S8 functions (ops/wofz.hpp,
// loggamma.hpp, lambertw.hpp), whose complex Ops take std::complex<double>.
#pragma once

#include "array/kernels/scalar_ops.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace sd::special {

// The loop dtypes: float64 only.
struct Real {
    template <typename T> static constexpr bool ok = ops::is_float<T>;
};

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr double kEps = 0x1p-53;               // half the spacing at 1: unit roundoff (Cephes MACHEP)
inline constexpr double kMaxLog = 7.09782712893383996843e2;   // log(DBL_MAX)
inline constexpr double kMinLog = -7.451332191019412076235e2;  // log(smallest subnormal)

// c[0] x^(N-1) + c[1] x^(N-2) + ... + c[N-1], by Horner's rule: one multiply
// and one add per coefficient, each rounded (-ffp-contract=off: no fused
// multiply-add unless written). The coefficient order is Cephes' polevl.
template <std::size_t N>
constexpr double horner(double x, const std::array<double, N>& c) {
    double r = c[0];
    for (std::size_t i = 1; i < N; ++i) r = r * x + c[i];
    return r;
}

// The same with an implicit leading coefficient 1 (Cephes' p1evl):
// x^N + c[0] x^(N-1) + ... + c[N-1].
template <std::size_t N>
constexpr double horner1(double x, const std::array<double, N>& c) {
    double r = x + c[0];
    for (std::size_t i = 1; i < N; ++i) r = r * x + c[i];
    return r;
}

}  // namespace sd::special
