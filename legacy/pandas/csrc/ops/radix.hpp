// Stable LSD radix sort on 64-bit keys (M7) and ordering the values of a
// column (M10), shared by sort_values and groupby's group order.
#pragma once

#include "core/column.hpp"
#include "ops/hash.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace sd {

// An unsigned 64-bit key whose order is the value order (API.md §17).
// int64: flip the sign bit, so negatives come first. float64: -0.0 becomes
// 0.0 and every NaN one NaN (both above +inf once mapped); then a positive
// float gets its sign bit set and a negative one all bits flipped -- the
// classic map from IEEE order to unsigned integer order.
inline std::uint64_t radix_key(const Column& c, std::int64_t i) {
    switch (c.dtype) {
    case DType::Int64: return static_cast<std::uint64_t>(c.data<std::int64_t>()[i]) ^ (std::uint64_t{1} << 63);
    case DType::Bool: return c.data<std::uint8_t>()[i];
    case DType::Float64: {
        const std::uint64_t bits = float_key(c.data<double>()[i]);  // NaN and -0.0 normalized (hash.hpp)
        return (bits >> 63) ? ~bits : bits | (std::uint64_t{1} << 63);
    }
    case DType::String: break;
    }
    return 0;
}

// The first 8 bytes of a string as a big-endian number (shorter strings
// padded with zero bytes): comparing prefixes compares the strings' first 8
// bytes in code point order (UTF-8 byte order).
inline std::uint64_t string_prefix(std::string_view s) {
    std::uint8_t b[8] = {};
    std::memcpy(b, s.data(), std::min<std::size_t>(8, s.size()));
    std::uint64_t k = 0;
    for (int j = 0; j < 8; ++j) k = (k << 8) | b[j];
    return k;
}

// Stable LSD radix sort of `rows` by 64-bit keys: eight passes of one byte,
// least significant first, all eight histograms counted in one pre-pass.
// Every pass runs, whatever the data (no skipping of constant digits).
inline void radix_sort(std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows) {
    const std::size_t n = keys.size();
    std::vector<std::array<std::int64_t, 256>> count(8);
    for (auto& h : count) h.fill(0);
    for (std::uint64_t k : keys)
        for (int d = 0; d < 8; ++d) ++count[d][(k >> (8 * d)) & 0xff];
    std::vector<std::uint64_t> keys2(n);
    std::vector<std::int64_t> rows2(n);
    for (int d = 0; d < 8; ++d) {
        std::array<std::int64_t, 256> start{};
        for (int b = 1; b < 256; ++b) start[b] = start[b - 1] + count[d][b - 1];
        for (std::size_t j = 0; j < n; ++j) {
            const auto b = (keys[j] >> (8 * d)) & 0xff;
            const std::int64_t to = start[b]++;
            keys2[to] = keys[j];
            rows2[to] = rows[j];
        }
        keys.swap(keys2);
        rows.swap(rows2);
    }
}

// `rows` (with values, no NA) ordered by the column's values, ascending.
// Numbers and bools: radix sort on radix_key. Strings: radix sort on the
// 8-byte prefix, then a comparison sort only inside runs of equal prefixes
// (M10: comparing every pair of strings cost half of a string groupby).
// Rows with equal values keep their order.
inline void order_rows(const Column& c, std::vector<std::int64_t>& rows) {
    std::vector<std::uint64_t> keys(rows.size());
    if (c.dtype != DType::String) {
        for (std::size_t j = 0; j < rows.size(); ++j) keys[j] = radix_key(c, rows[j]);
        radix_sort(keys, rows);
        return;
    }
    for (std::size_t j = 0; j < rows.size(); ++j) keys[j] = string_prefix(c.string_at(rows[j]));
    radix_sort(keys, rows);
    for (std::size_t a = 0; a < rows.size();) {
        std::size_t b = a + 1;
        while (b < rows.size() && keys[b] == keys[a]) ++b;
        if (b - a > 1)  // equal first 8 bytes: the rest decides
            std::stable_sort(rows.begin() + static_cast<std::ptrdiff_t>(a), rows.begin() + static_cast<std::ptrdiff_t>(b),
                             [&](std::int64_t x, std::int64_t y) { return c.string_at(x) < c.string_at(y); });
        a = b;
    }
}

}  // namespace sd
