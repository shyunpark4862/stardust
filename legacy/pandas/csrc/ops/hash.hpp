// factorize: values -> dense int codes, the base of unique, duplicated and
// value_counts (M3) and later groupby and merge (DESIGN.md §2.5).
#pragma once

#include "core/column.hpp"

#include <vector>

namespace sd {

struct Factorized {
    Column codes;  // int64, one per row: 0, 1, ... in order of first appearance
    Column first;  // int64, per code: the row where it first appears
    std::int64_t ngroups = 0;
};

// Codes for the rows of one column. Keys follow API.md §1 and §13: NaN
// equals NaN, 0.0 equals -0.0, and all NA rows share one code.
Factorized factorize(const Column& col);

// Codes for rows of several columns taken together (a row's key is the
// tuple of its values). Each column is factorized, the codes are combined
// into one int64 (mixed radix), and that is factorized again so the codes
// are dense and in order of first appearance.
Factorized factorize_rows(const std::vector<Column>& cols);

// Per code: how many rows have it, and the last row that has it.
std::pair<Column, Column> group_counts(const Column& codes, std::int64_t ngroups);

// Positions that order `counts` by value (descending unless `ascending`),
// keeping the original order among equal values (a stable sort).
Column stable_order(const Column& counts, bool ascending);

}  // namespace sd
