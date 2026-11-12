// Which array operators run on vForce (docs/design/array.md §5.4): the
// operator -> function tables over the wrappers in accelerate/vforce.hpp. A
// nullptr entry means the operator stays on libm.
#pragma once

#include "accelerate/vforce.hpp"
#include "array/kernels/scalar_ops.hpp"

namespace sd {

template <typename Op> inline constexpr VUnary vforce_unary = nullptr;
template <> inline constexpr VUnary vforce_unary<ops::Exp> = &vvexp;
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
template <> inline constexpr VBinary vforce_binary<ops::Pow> = &vf_pow;
template <> inline constexpr VBinary vforce_binary<ops::Atan2> = &vf_atan2;

}  // namespace sd
