// read_csv's core (docs/api/frame.md §10, docs/design/frame.md §10): RFC 4180
// tokenizing over a NEON structural index, then per-column type inference
// while building. The body is cut into byte chunks (64 KB to 512 KB), each
// indexed and then parsed column by column, on the pool when it can run in
// parallel. The result is the same however the body is cut
// (docs/design/frame.md §10.6).
#pragma once

#include "frame/column.hpp"

#include <memory>
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

// Records (line feeds outside quotes) of one block of a file read in pieces,
// for read_csv's nrows= (it reads the file only as far as needed). From the
// quote state `inside` at the block's start: `end` is the byte just after
// the `records`-th record end in the block, or -1 when the block holds fewer;
// `seen` the records counted (up to `records`); `inside` the state at the
// block's end (when end is -1).
struct CsvScan {
    std::int64_t end;
    std::int64_t seen;
    bool inside;
};
CsvScan csv_scan_records(const char* data, std::size_t size, std::int64_t records, bool inside);

// The bytes of the open file `fd` from offset 0, `size` of them expected (its
// size when opened; a file that grew is read to its end): pieces of 4 MB
// read in parallel with pread, without the GIL. nullopt when a read fails.
struct FileBytes {
    std::unique_ptr<char[]> data;
    std::int64_t size = 0;
};
std::optional<FileBytes> read_fd(int fd, std::int64_t size);

// (column names, columns) of the CSV text in [data, data + size).
std::pair<std::vector<std::string>, std::vector<Column>> read_csv(
    const char* data, std::size_t size, const CsvOptions& opt
);

}  // namespace sd::frame
