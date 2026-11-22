// Elementwise functions (docs/api/array.md §6): the dispatcher between Python operands
// and the loop tables of kernels/elementwise.hpp.
//
// Every call goes through the same four steps:
//   1. operands   each is an sd.ndarray or a Python bool/int/float/complex
//   2. loop       promote the operand dtypes, then take the first dtype at or
//                 above it that the operation has a loop for (numpy's rule).
//                 Bool arithmetic in int64 (docs/design/array.md §2.1) and
//                 "integers divide in float64" fall out of the tables:
//                 add has no bool loop, divide has no int64 loop.
//   3. convert    inputs are cast to the loop dtype chunk by chunk on the way
//                 (buffered casting, kernels/buffered.hpp;
//                 docs/design/array.md §4.3)
//   4. run        NdIter over inputs + output; the output is `out=` (upward
//                 conversion only, overlap-safe: docs/design/array.md §2.5,
//                 §3.5) or a new array laid out like the inputs
// With no array operand at all the result is a Python scalar
// (docs/design/array.md §2.2).
#pragma once

#include "array/kernels/elementwise.hpp"

#include <nanobind/nanobind.h>

#include <array>
#include <cstdint>
#include <span>

namespace sd {

namespace nb = nanobind;

// What elementwise functions accept as an operand (no lists or foreign
// arrays -- convert them with asarray first; docs/design/array.md §2.8).
bool is_operand(nb::handle o);

// `symbol` is the operator when an operator calls (x + y: "+"; -x: "unary -";
// abs(x): "abs()"), which then says so in Python's form ("unsupported operand
// dtypes for +: ..."); nullptr for the function (sd.add), whose binding puts
// the function's name in front of the message (bind/errors.hpp).
// `weights`: the loop's work per element for each loop dtype, in units of a
// float64 add (op_weights below); it sets where the iterator starts to use the
// pool (NdIter::plan). Results never depend on it.
using LoopWeights = std::array<std::int64_t, kNumArrayDTypes>;
inline constexpr LoopWeights kUnitWeights = {1, 1, 1, 1};
nb::object run_unary(
    const LoopTable& table, nb::handle x, nb::handle out, const char* symbol, const LoopWeights& weights = kUnitWeights
);
nb::object run_binary(
    const LoopTable& table,
    bool no_bool_inputs,
    nb::handle a,
    nb::handle b,
    nb::handle out,
    const char* symbol,
    const LoopWeights& weights = kUnitWeights
);

// The cost of an operation's loop per element, per loop dtype {bool, int64,
// float64, complex128}, relative to a float64 add (0.13 ns per element in
// cache on one core), measured at 8192 elements: exp 8 (6 on array/kernels/vmath.hpp's
// kernel, expm1 8, log1p 10, cos 14 there), cos 19, log1p 21 (libm),
// complex log 100. 1 for everything about as cheap as an add. With weight 1 the
// pool starts at 2^15 elements (kParallelThreshold); exp started there too, at
// 32 us of work on one thread (docs/design/array.md §10.4).
template <typename Op> inline constexpr LoopWeights op_weights = kUnitWeights;
#define SD_OP_WEIGHTS(Op, b, i, f, c) template <> inline constexpr LoopWeights op_weights<ops::Op> = {b, i, f, c};
SD_OP_WEIGHTS(Sqrt, 1, 1, 2, 11)
SD_OP_WEIGHTS(Exp, 1, 1, 6, 53)
SD_OP_WEIGHTS(Expm1, 1, 1, 8, 60)
SD_OP_WEIGHTS(Log, 1, 1, 8, 100)
SD_OP_WEIGHTS(Log1p, 1, 1, 10, 100)
SD_OP_WEIGHTS(Log2, 1, 1, 8, 100)
SD_OP_WEIGHTS(Log10, 1, 1, 9, 100)
SD_OP_WEIGHTS(Sin, 1, 1, 6, 65)
SD_OP_WEIGHTS(Cos, 1, 1, 14, 65)
SD_OP_WEIGHTS(Tan, 1, 1, 17, 65)
SD_OP_WEIGHTS(Asin, 1, 1, 7, 65)
SD_OP_WEIGHTS(Acos, 1, 1, 10, 65)
SD_OP_WEIGHTS(Atan, 1, 1, 8, 65)
SD_OP_WEIGHTS(Sinh, 1, 1, 8, 65)
SD_OP_WEIGHTS(Cosh, 1, 1, 7, 65)
SD_OP_WEIGHTS(Tanh, 1, 1, 9, 65)
SD_OP_WEIGHTS(Asinh, 1, 1, 25, 65)
SD_OP_WEIGHTS(Acosh, 1, 1, 21, 65)
SD_OP_WEIGHTS(Atanh, 1, 1, 16, 65)
SD_OP_WEIGHTS(Abs, 1, 1, 1, 14)
SD_OP_WEIGHTS(Sign, 1, 1, 2, 13)
SD_OP_WEIGHTS(Reciprocal, 1, 1, 1, 7)
SD_OP_WEIGHTS(Square, 1, 2, 1, 2)
SD_OP_WEIGHTS(Add, 1, 1, 1, 3)
SD_OP_WEIGHTS(Subtract, 1, 1, 1, 3)
SD_OP_WEIGHTS(Multiply, 1, 2, 1, 3)
SD_OP_WEIGHTS(Divide, 1, 1, 1, 12)
SD_OP_WEIGHTS(FloorDivide, 5, 5, 20, 20)
SD_OP_WEIGHTS(Remainder, 4, 4, 17, 17)
SD_OP_WEIGHTS(Pow, 17, 17, 38, 200)
SD_OP_WEIGHTS(Atan2, 1, 1, 14, 14)
SD_OP_WEIGHTS(Hypot, 1, 1, 9, 9)
SD_OP_WEIGHTS(Logaddexp, 1, 1, 49, 49)
SD_OP_WEIGHTS(Nextafter, 1, 1, 5, 5)
SD_OP_WEIGHTS(Equal, 1, 1, 1, 2)
SD_OP_WEIGHTS(NotEqual, 1, 1, 1, 2)
#undef SD_OP_WEIGHTS

// Any number of inputs (1 to kMaxOperands - 1) and nout outputs, all of one
// dtype (kernels/elementwise.hpp, nary_loop). `names` names the inputs in
// messages ('a', 'b', 'x'). With nout > 1 the result is a tuple and `out`
// must be None. `weight`: work per element in adds (docs/design/special.md
// §3.3). The function name is the binding's to add (bind/errors.hpp).
nb::object run_nary(
    const LoopTable& table,
    int nout,
    std::span<const char* const> names,
    std::span<const nb::handle> objs,
    nb::handle out,
    std::int64_t weight
);

template <typename Op>
nb::object nary(std::span<const nb::handle> objs, nb::handle out, std::int64_t weight) {
    static_assert(Op::args.size() == static_cast<std::size_t>(Op::nin));
    return run_nary(nary_table<Op>, Op::nout, Op::args, objs, out, weight);
}

template <typename Op>
nb::object unary(nb::handle x, nb::handle out, const char* symbol = nullptr) {
    return run_unary(unary_table<Op>, x, out, symbol, op_weights<Op>);
}

// pow's weights for exponent `b`: a Python scalar 2, 0, 1, -1 or 0.5 runs the
// exact-form loops (x * x, 1, x, 1 / x, sqrt: kernels/elementwise.hpp) at the
// cost of an add, and a small non-negative int exponent takes an int loop only
// a few multiplies long. With libm pow's weight, x ** 2 of 10^4 elements went
// to the pool in ~100-element pieces: 1.0 us on one thread, 9.7 on ten.
inline const LoopWeights& pow_weights(nb::handle b) {
    static constexpr LoopWeights general = op_weights<ops::Pow>;
    static constexpr LoopWeights exact = {1, 2, 1, general[3]};      // float: an exact form
    static constexpr LoopWeights small_int = {1, 2, general[2], general[3]};  // float: libm
    PyObject* o = b.ptr();
    double e;
    if (PyFloat_CheckExact(o)) {
        e = PyFloat_AS_DOUBLE(o);
    } else if (PyLong_CheckExact(o)) {
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
        if (overflow != 0 || v < -1 || v > 4) return general;
        if (v >= 3) return small_int;
        e = static_cast<double>(v);
    } else {
        return general;
    }
    return e == 2.0 || e == 0.0 || e == 1.0 || e == -1.0 || e == 0.5 ? exact : general;
}

template <typename Op>
nb::object binary(nb::handle a, nb::handle b, nb::handle out, const char* symbol = nullptr) {
    constexpr bool no_bool = requires { Op::no_bool_inputs; };
    if constexpr (std::is_same_v<Op, ops::Pow>)
        return run_binary(binary_table<Op>, no_bool, a, b, out, symbol, pow_weights(b));
    return run_binary(binary_table<Op>, no_bool, a, b, out, symbol, op_weights<Op>);
}

void bind_ufuncs(nb::module_& m);

}  // namespace sd
