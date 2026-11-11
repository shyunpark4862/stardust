// Basic types shared by the whole core.
#pragma once

#include <Accelerate/Accelerate.h>  // ILP64 + new LAPACK via compile definitions (DESIGN.md §1.5)

#include <complex>
#include <cstdint>

namespace sdla {

using cplx = std::complex<double>;
using lint = __LAPACK_int;  // 64-bit LAPACK integer
static_assert(sizeof(lint) == 8, "ACCELERATE_LAPACK_ILP64 must be defined");

// The four stardust-numpy dtypes, read from the buffer format of an input.
enum class DType : std::uint8_t { Bool, Int64, Float64, Complex128 };

// What LAPACK computes in for a given input: bool and int64 become float64
// (DESIGN.md §1.4); the result dtype depends on input dtypes only (API §0.3).
inline bool computes_complex(DType d) { return d == DType::Complex128; }

}  // namespace sdla
