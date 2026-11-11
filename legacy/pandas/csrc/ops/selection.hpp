// Picking rows and handling NA (ARCHITECTURE.md §6.1, API.md §12).
#pragma once

#include "core/column.hpp"

#include <vector>

namespace sd {

// Rows of `col` at `positions` (int64, no NA), in that order. Position -1
// gives an NA row (M4's alignment uses it); others must be in [0, length).
Column take(const Column& col, const Column& positions);

// The rows of `col` where the bool `mask` is True (NA counts as False),
// compacted directly: no positions in between (M10). Strings go through
// take(nonzero(mask)).
Column filter(const Column& col, const Column& mask);

// Positions (int64) of the rows of a bool mask that are True; NA counts as
// False (API.md §2).
Column nonzero(const Column& mask);

// bool, no NA: whether each row is NA (or, with `want_na` false, is not).
Column isna(const Column& col, bool want_na);

// The column with its NA rows set to `value`, a one-row column of the same dtype.
Column fill_na(const Column& col, const Column& value);

// Same dtype, length, NA rows and values (bitwise for floats).
bool equals(const Column& a, const Column& b);

// Writes `values` (same dtype; one value per position, or one value for all)
// into the rows of `col` at `positions`, in place. Copy-on-Write
// (ARCHITECTURE.md §3.2): a Buffer that anything else also holds -- another
// Column, a slice, a numpy view -- is copied before the first write.
void set_rows(Column& col, const Column& positions, const Column& values);

// The columns one after another (all of one dtype).
Column concat(const std::vector<Column>& cols);

// int64 start, start + step, ... (n values).
Column arange(std::int64_t start, std::int64_t step, std::int64_t n);

}  // namespace sd
