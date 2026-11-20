// read_csv's core (docs/api/frame.md §10, docs/design/frame.md §10): RFC 4180
// tokenizing over a NEON structural index, then per-column type inference
// while building. When the pool can run in parallel the body is cut into
// 512 KB byte chunks, tokenized and built on the pool; with nrows, or when
// the pool cannot run in parallel (one thread, or busy), it is a single
// chunk read serially. The result is the same either way
// (docs/design/frame.md §10.6).
#pragma once

#include "frame/column.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sd::frame {

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

// The number of bytes from `data` up to the end of its first `records`
// records (line feeds outside quotes), or -1 when it holds fewer. read_csv's
// nrows= reads the file only that far.
std::int64_t csv_records_end(const char* data, std::size_t size, std::int64_t records);

// (column names, columns) of the CSV text in [data, data + size).
std::pair<std::vector<std::string>, std::vector<Column>> read_csv(
    const char* data, std::size_t size, const CsvOptions& opt
);

}  // namespace sd::frame
