// A Column: one typed, nullable array (docs/design/frame.md §4.3). It is a small
// value type -- a few integers and shared_ptrs to Buffers -- so copying one
// copies no data, and it crosses into Python by value (docs/design/frame.md §3.2).
#pragma once

#include "base/buffer.hpp"
#include "base/dtype.hpp"
#include "frame/bitmap.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sd::frame {

// The frame's dtypes are Bool, Int64, Float64 and String of the shared DType
// (base/dtype.hpp, docs/design/frame.md §4.2; Complex128 is the array area's
// only). dtype_name and itemsize come from there too: one byte per bool
// value, 0 or 1 (docs/design/frame.md §4.6); strings keep their bytes
// elsewhere (itemsize 0).
DType parse_dtype(std::string_view name);  // "bool", "int64", "float64", "string"

struct Column {
    DType dtype = DType::Float64;
    std::int64_t length = 0;
    // First row inside the Buffers: a slice moves `offset` and `length` and
    // shares everything else. It counts elements of `values`, bits of
    // `validity` and entries of `offsets` alike.
    std::int64_t offset = 0;
    std::int64_t null_count = 0;
    // bool/int64/float64: the values. string: all the UTF-8 bytes, back to back.
    std::shared_ptr<Buffer> values;
    // Null exactly when null_count == 0, so kernels can skip it (docs/design/frame.md §4.4).
    std::shared_ptr<Buffer> validity;
    // string only: length + 1 int64 positions into `values`; row i is bytes
    // [offsets[i], offsets[i + 1]). They are absolute, so slices share them.
    std::shared_ptr<Buffer> offsets;

    template <class T>
    const T* data() const { return reinterpret_cast<const T*>(values->data()) + offset; }

    const std::uint8_t* validity_bits() const {
        return validity ? reinterpret_cast<const std::uint8_t*>(validity->data()) : nullptr;
    }
    bool is_valid(std::int64_t i) const { return !validity || bitmap::get(validity_bits(), offset + i); }

    const std::int64_t* string_offsets() const { return reinterpret_cast<const std::int64_t*>(offsets->data()) + offset; }
    std::string_view string_at(std::int64_t i) const {
        const std::int64_t* o = string_offsets();
        return {reinterpret_cast<const char*>(values->data()) + o[i], static_cast<std::size_t>(o[i + 1] - o[i])};
    }

    // Rows [start, start + len), sharing the Buffers (zero-copy). The null
    // count is recounted for the range; a slice with no NA drops the bitmap.
    Column slice(std::int64_t start, std::int64_t len) const;

    // The same values as `target`, when that loses nothing (docs/api/frame.md §2.3):
    // int64 -> float64 when every value is exact, float64 -> int64 when every
    // value is integral and in range. The same dtype returns the column itself.
    // Anything else raises (TypeError for a dtype pair, ValueError for a value).
    Column convert(DType target) const;
};

// A column of `length` rows of a fixed-width dtype with fresh, uninitialized
// values and no bitmap; the caller fills it (and adds a bitmap if needed).
Column make_fixed(DType dtype, std::int64_t length);

// Gives `col` a validity bitmap with every bit set, for builders that are
// about to clear the bits of their NA rows. Only valid while offset == 0.
std::uint8_t* add_validity(Column& col);

// The validity of rows [0, n) of `col` as a fresh bitmap starting at bit 0,
// or null when the column has no NA.
std::shared_ptr<Buffer> rebased_validity(const Column& col);

// Validity of an element-wise result: a row has a value when both inputs do.
// `scalar_side` 1 or 2 says that `a` or `b` is a one-row scalar applied to
// every row. Null when no row can be NA.
std::shared_ptr<Buffer> and_validity(const Column& a, const Column& b, int scalar_side, std::int64_t n);

// One byte per row: 1 where the row has a value, 0 where it is NA (all ones
// without a bitmap). Branch-free kernels select with it (docs/design/frame.md §4.4).
std::vector<std::uint8_t> valid_bytes(const Column& col);

// Ends building a fixed-width column whose `validity` (offset 0, possibly
// null) is final: counts the NA rows, drops an all-valid bitmap, and zeroes
// the values of NA rows (docs/design/frame.md §4.5).
void finish_fixed(Column& col);

// True when the int64 `v` converts to a double and back unchanged.
inline bool int_is_exact_double(std::int64_t v) {
    const double d = static_cast<double>(v);
    // 2^63 itself is a double but not an int64; the cast back would overflow.
    return d < 9223372036854775808.0 && static_cast<std::int64_t>(d) == v;
}

// True when the double `d` is integral and fits int64 (NaN and inf do not).
inline bool double_is_exact_int(double d) {
    return d >= -9223372036854775808.0 && d < 9223372036854775808.0 && static_cast<double>(static_cast<std::int64_t>(d)) == d;
}

}  // namespace sd::frame
