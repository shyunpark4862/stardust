// Grouped aggregation (M6, API.md §16). factorize gives every row a group
// code; Groups lays the rows out group by group (a counting sort, so each
// group keeps its rows in their original order) and reduces each group's
// rows as M5 reduces a whole column, sums included: one group's sum has the
// same bits as the Series.sum() of just its rows.
#pragma once

#include "core/column.hpp"

#include <string_view>
#include <vector>

namespace sd {

class Groups {
public:
    // `codes`: int64, 0 .. ngroups - 1 per row (as factorize makes them).
    Groups(const Column& codes, std::int64_t ngroups);

    // Rows per group (int64).
    Column sizes() const;

    // One result per group, in code order: sum mean min max count std var
    // first last, with the dtypes and NA rules of API.md §15 and §16.
    Column reduce(std::string_view op, const Column& col, bool skipna, std::int64_t ddof) const;

private:
    std::int64_t ngroups_;
    std::vector<std::int64_t> start_;  // group g's rows are rows_[start_[g], start_[g + 1])
    std::vector<std::int64_t> rows_;
};

// The order of groups by their keys, ascending (API.md §16): `keys` hold one
// row per group (the group's first row). Numbers by value with NaN after
// +inf and -0.0 equal to 0.0, strings by code point, False < True, NA last;
// several keys compare in turn. Ties keep the group order (a stable sort).
Column sort_groups(const std::vector<Column>& keys);

}  // namespace sd
