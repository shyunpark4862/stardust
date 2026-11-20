// Picking rows and handling NA (docs/design/frame.md §7.1, docs/api/frame.md §3.6).
#pragma once

#include "frame/column.hpp"

#include <utility>
#include <vector>

namespace sd::frame {

// Rows of `col` at `positions` (int64, no NA), in that order. Position -1
// gives an NA row (alignment uses it, docs/design/frame.md §6.3); others must be in [0, length).
Column take(const Column& col, const Column& positions);

// The rows of `col` where the bool `mask` is True (NA counts as False),
// compacted directly: no positions in between (docs/design/frame.md §7.3). Strings go through
// take(nonzero(mask)).
Column filter(const Column& col, const Column& mask);

// Positions (int64) of the rows of a bool mask that are True; NA counts as
// False (docs/api/frame.md §1.4).
Column nonzero(const Column& mask);

// bool, no NA: whether each row is NA (or, with `want_na` false, is not).
Column isna(const Column& col, bool want_na);

// bool: whether each value is NaN (docs/api/frame.md §3.6). Columns other than
// float64 have no NaN: all False. NA rows stay NA, as in a comparison.
Column isnan(const Column& col);

// The column with its NA rows set to `value`, a one-row column of the same dtype.
Column fill_na(const Column& col, const Column& value);

// Same dtype, length, NA rows and values (bitwise for floats).
bool equals(const Column& a, const Column& b);

// Writes `values` (same dtype; one value per position, or one value for all)
// into the rows of `col` at `positions`, in place. Copy-on-Write
// (docs/design/frame.md §5.2): a Buffer that anything else also holds -- another
// Column, a slice, a numpy view -- is copied before the first write.
void set_rows(Column& col, const Column& positions, const Column& values);

// The columns one after another (all of one dtype).
Column concat(const std::vector<Column>& cols);

// int64 start, start + step, ... (n values), wrapping as int64 arithmetic does.
Column arange(std::int64_t start, std::int64_t step, std::int64_t n);

// Positions given by position (int64, no NA, negative ones counting from the
// end) as positions in [0, n), with the index of the first one outside
// [-n, n), or -1 when all are inside. The index is the same for any thread
// count (parallel::FirstRow), so the caller's message names the first bad key.
std::pair<Column, std::int64_t> wrap_positions(const Column& positions, std::int64_t n);

// n rows of the one-row `value`: its value in every row, or all NA when it
// is NA (values 0, docs/design/frame.md §4.5). The dtype is the value's.
Column full(const Column& value, std::int64_t n);

}  // namespace sd::frame
