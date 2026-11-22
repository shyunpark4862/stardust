// Which array operators run on the area's NEON kernels or on vForce
// (docs/design/array.md §5.4): the operator -> function tables over
// array/kernels/vmath.hpp and the wrappers in accelerate/vforce.hpp. A nullptr
// entry in both means the operator stays on libm.
#pragma once

#include "accelerate/vforce.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "array/kernels/vmath.hpp"

namespace sd {

// The operators on the array area's own NEON kernels (array/kernels/vmath.hpp): faster than
// vForce or libm and within 1 ULP of the correctly rounded value (the numbers are in that
// file). Checked before the vForce table.
using VMath = void (*)(double*, const double*, std::int64_t);
template <typename Op> inline constexpr VMath vmath_unary = nullptr;
template <> inline constexpr VMath vmath_unary<ops::Exp> = &vmath::exp;
template <> inline constexpr VMath vmath_unary<ops::Expm1> = &vmath::expm1;
template <> inline constexpr VMath vmath_unary<ops::Log1p> = &vmath::log1p;
template <> inline constexpr VMath vmath_unary<ops::Cos> = &vmath::cos;

// Any strides: contiguous straight through, else through stack buffers of
// vforce_detail::kChunk elements.
inline void vmath_unary_run(
    VMath f, const std::byte* x, std::int64_t sx, std::byte* o, std::int64_t so, std::int64_t n
) {
    using namespace vforce_detail;
    if (sx == sizeof(double) && so == sizeof(double)) {
        f(reinterpret_cast<double*>(o), reinterpret_cast<const double*>(x), n);
        return;
    }
    double in[kChunk], out[kChunk];
    for (std::int64_t off = 0; off < n; off += kChunk) {
        const std::int64_t m = std::min(kChunk, n - off);
        const double* src = gather(x + off * sx, sx, m, in);
        const bool direct = so == sizeof(double);
        double* dst = direct ? reinterpret_cast<double*>(o + off * so) : out;
        f(dst, src, m);
        if (!direct) scatter(out, o + off * so, so, m);
    }
}

template <typename Op> inline constexpr VUnary vforce_unary = nullptr;
template <> inline constexpr VUnary vforce_unary<ops::Log> = &vvlog;
template <> inline constexpr VUnary vforce_unary<ops::Log2> = &vvlog2;
template <> inline constexpr VUnary vforce_unary<ops::Log10> = &vvlog10;
template <> inline constexpr VUnary vforce_unary<ops::Sin> = &vvsin;
template <> inline constexpr VUnary vforce_unary<ops::Asin> = &vvasin;
template <> inline constexpr VUnary vforce_unary<ops::Acos> = &vvacos;
template <> inline constexpr VUnary vforce_unary<ops::Atan> = &vvatan;
template <> inline constexpr VUnary vforce_unary<ops::Sinh> = &vvsinh;
template <> inline constexpr VUnary vforce_unary<ops::Cosh> = &vvcosh;
template <> inline constexpr VUnary vforce_unary<ops::Tanh> = &vvtanh;
template <> inline constexpr VUnary vforce_unary<ops::Asinh> = &vvasinh;
template <> inline constexpr VUnary vforce_unary<ops::Acosh> = &vvacosh;
template <> inline constexpr VUnary vforce_unary<ops::Atanh> = &vvatanh;

template <typename Op> inline constexpr VBinary vforce_binary = nullptr;
template <> inline constexpr VBinary vforce_binary<ops::Atan2> = &vf_atan2;

}  // namespace sd
