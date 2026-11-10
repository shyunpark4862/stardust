// Picking rows and handling NA (ARCHITECTURE.md §6.1, API.md §12).
#pragma once

#include "core/column.hpp"

namespace sd {

// Rows of `col` at `positions` (int64, no NA), in that order. Position -1
// gives an NA row (M4's alignment uses it); others must be in [0, length).
Column take(const Column& col, const Column& positions);

// Positions (int64) of the rows of a bool mask that are True; NA counts as
// False (API.md §2).
Column nonzero(const Column& mask);

// bool, no NA: whether each row is NA (or, with `want_na` false, is not).
Column isna(const Column& col, bool want_na);

// The column with its NA rows set to `value`, a one-row column of the same dtype.
Column fill_na(const Column& col, const Column& value);

// Same dtype, length, NA rows and values (bitwise for floats).
bool equals(const Column& a, const Column& b);

// int64 start, start + step, ... (n values).
Column arange(std::int64_t start, std::int64_t step, std::int64_t n);

}  // namespace sd
