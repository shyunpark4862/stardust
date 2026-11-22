// Hash join (docs/api/frame.md §9.1, docs/design/frame.md §9.7).
#pragma once

#include "frame/column.hpp"

#include <string_view>
#include <utility>
#include <vector>

namespace sd::frame {

// The rows of a join as (left positions, right positions), -1 where a side
// has no row. how: inner left right outer. Order (docs/api/frame.md §9.1): left rows in
// order, each followed by its matches in right order; outer then appends the
// unmatched right rows in right order; right is the mirror image. Keys
// match as factorize says (NaN = NaN, 0.0 = -0.0); a key holding NA matches
// nothing.
std::pair<Column, Column> join_indices(
    const std::vector<Column>& left, const std::vector<Column>& right, std::string_view how
);

// Whether int64 positions are exactly 0, 1, ..., length - 1.
bool is_arange(const Column& positions);

}  // namespace sd::frame
