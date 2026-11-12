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

namespace sd {

namespace nb = nanobind;

// What elementwise functions accept as an operand (no lists or foreign
// arrays -- convert them with asarray first; docs/design/array.md §2.8).
bool is_operand(nb::handle o);

// `symbol` is the operator when an operator calls (x + y: "+"; -x: "unary -";
// abs(x): "abs()"), which then says so in Python's form ("unsupported operand
// dtypes for +: ..."); nullptr for the function (sd.add), whose binding puts
// the function's name in front of the message (bind/errors.hpp).
nb::object run_unary(const LoopTable& table, nb::handle x, nb::handle out, const char* symbol);
nb::object run_binary(const LoopTable& table, bool no_bool_inputs, nb::handle a, nb::handle b, nb::handle out,
                      const char* symbol);

template <typename Op>
nb::object unary(nb::handle x, nb::handle out, const char* symbol = nullptr) {
    return run_unary(unary_table<Op>, x, out, symbol);
}

template <typename Op>
nb::object binary(nb::handle a, nb::handle b, nb::handle out, const char* symbol = nullptr) {
    constexpr bool no_bool = requires { Op::no_bool_inputs; };
    return run_binary(binary_table<Op>, no_bool, a, b, out, symbol);
}

void bind_ufuncs(nb::module_& m);

}  // namespace sd
