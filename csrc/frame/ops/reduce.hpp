// Reductions (docs/api/frame.md §6): sum, mean, min, max, count, std, var.
#pragma once

#include "frame/column.hpp"

#include <string_view>

namespace sd::frame {

// The result as a one-row Column (NA when there is no result), so Series
// can read it out as a scalar and DataFrame can stack it into a row. op is
// one of sum mean min max count std var; ddof is used by std and var.
Column reduce(std::string_view op, const Column& col, bool skipna, std::int64_t ddof);

// The result dtype of `op` (sum mean min max count std var first last) on a
// column of `dt`; TypeError for the combinations docs/api/frame.md §6.3 rules out.
DType reduction_dtype(std::string_view op, DType dt);

// A one-row NA Column of `dt`.
Column na_scalar(DType dt);

}  // namespace sd::frame
