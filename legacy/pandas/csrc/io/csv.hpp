// read_csv's core (M9, API.md §19): RFC 4180 tokenizing, then per-column
// type inference and building. Single-threaded; M10 adds NEON scanning and
// M11 parallel chunks.
#pragma once

#include "core/column.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sd {

struct CsvOptions {
    char sep = ',';
    bool header = true;
    std::vector<std::string> names;    // empty: from the header
    std::vector<std::string> usecols;  // empty: every column, in file order
    std::unordered_map<std::string, DType> dtypes;  // columns read as a given dtype (no inference)
    std::optional<DType> all_dtype;                 // every column read as this dtype
    std::vector<std::string> na_values;            // texts read as NA, besides the unquoted empty field
    std::int64_t nrows = -1;                       // -1: all
};

// (column names, columns) of the CSV text in [data, data + size).
std::pair<std::vector<std::string>, std::vector<Column>> read_csv(const char* data, std::size_t size,
                                                                   const CsvOptions& opt);

}  // namespace sd
