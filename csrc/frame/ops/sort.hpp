// Stable argsort by several keys (docs/api/frame.md §8, docs/design/frame.md §9.3).
#pragma once

#include "frame/column.hpp"

#include <vector>

namespace sd::frame {

// The positions that order the rows by keys[0], then keys[1], ... Each key
// has its own direction and NA position; equal rows keep their order (a
// stable sort). Numbers and bools go through an LSD radix sort on 64-bit
// keys, strings through ranks of their distinct values.
Column argsort(const std::vector<Column>& keys, const std::vector<bool>& ascending, const std::vector<bool>& na_last);

}  // namespace sd::frame
