// astype (API.md §12): explicit conversion between Tier 1 dtypes.
#pragma once

#include "core/column.hpp"

namespace sd {

// The column as `target`. NA stays NA. Rounds int -> float, truncates
// float -> int (NaN, inf and out-of-range values raise), maps bool <-> 0/1,
// formats to string like Python's str(), and parses strings strictly.
Column astype(const Column& col, DType target);

}  // namespace sd
