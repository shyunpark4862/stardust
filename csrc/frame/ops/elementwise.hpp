// Element-wise operators (docs/api/frame.md §3): arithmetic, comparison, logic and bitwise.
#pragma once

#include "frame/column.hpp"

#include <string_view>

namespace sd::frame {

// `a op b` row by row. `scalar_side` is 0 when both columns have the same
// length, 1 or 2 when `a` or `b` is a one-row scalar applied to every row.
// op: add sub mul truediv floordiv mod pow eq ne lt le gt ge and or xor lshift rshift.
Column binary_op(std::string_view op, const Column& a, const Column& b, int scalar_side);

// op: neg abs invert (logical not on bool, bitwise not on int64).
Column unary_op(std::string_view op, const Column& a);

}  // namespace sd::frame
