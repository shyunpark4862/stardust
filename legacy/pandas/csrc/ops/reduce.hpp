// Reductions (API.md §15): sum, mean, min, max, count, std, var.
#pragma once

#include "core/column.hpp"

#include <string_view>

namespace sd {

// The result as a one-row Column (NA when there is no result), so Series
// can read it out as a scalar and DataFrame can stack it into a row. op is
// one of sum mean min max count std var; ddof is used by std and var.
Column reduce(std::string_view op, const Column& col, bool skipna, std::int64_t ddof);

}  // namespace sd
