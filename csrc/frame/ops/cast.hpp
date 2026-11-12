// astype (docs/api/frame.md §3.5): explicit conversion between the four
// dtypes (docs/design/frame.md §2.1).
#pragma once

#include "frame/column.hpp"

namespace sd::frame {

// The column as `target`. NA stays NA. Rounds int -> float, truncates
// float -> int (NaN, inf and out-of-range values raise), maps bool <-> 0/1,
// formats to string like Python's str(), and parses strings strictly.
Column astype(const Column& col, DType target);

}  // namespace sd::frame
