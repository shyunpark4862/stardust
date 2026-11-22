// Reductions (docs/api/frame.md §6): sum, mean, min, max, count, std, var.
#pragma once

#include "frame/column.hpp"

#include <string_view>
#include <vector>

namespace sd::frame {

// The result as a one-row Column (NA when there is no result), so Series
// can read it out as a scalar and DataFrame can stack it into a row. op is
// one of sum mean min max count std var; correction is used by std and var.
Column reduce(std::string_view op, const Column& col, bool skipna, double correction);

// reduce() of several columns (a DataFrame's), the columns in parallel when
// each is below the split threshold but all of them together are not. The
// results are the ones reduce() gives each column alone; an error is the
// first column's, in order.
std::vector<Column> reduce_columns(
    std::string_view op, const std::vector<Column>& cols, bool skipna, double correction
);

// The result dtype of `op` (sum mean min max count std var first last) on a
// column of `dt`; TypeError for the combinations docs/api/frame.md §6.3 rules out.
DType reduction_dtype(std::string_view op, DType dt);

// A one-row NA Column of `dt`.
Column na_scalar(DType dt);

}  // namespace sd::frame
