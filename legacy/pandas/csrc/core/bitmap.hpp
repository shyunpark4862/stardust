// The validity bitmap (ARCHITECTURE.md §2.3): one bit per row, LSB first
// (row i is bit i % 8 of byte i / 8, as in Arrow); 1 means a value, 0 means NA.
#pragma once

#include <cstdint>

namespace sd::bitmap {

inline std::int64_t num_bytes(std::int64_t nbits) { return (nbits + 7) / 8; }

inline bool get(const std::uint8_t* bits, std::int64_t i) { return (bits[i >> 3] >> (i & 7)) & 1; }

inline void set(std::uint8_t* bits, std::int64_t i) { bits[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7)); }

inline void clear(std::uint8_t* bits, std::int64_t i) { bits[i >> 3] &= static_cast<std::uint8_t>(~(1u << (i & 7))); }

// Number of set bits in [start, start + length). Whole bytes go through
// popcount; the partial bytes at either end are masked first.
inline std::int64_t count_set(const std::uint8_t* bits, std::int64_t start, std::int64_t length) {
    if (length <= 0) return 0;
    const std::int64_t end = start + length;
    std::int64_t first = start >> 3, last = (end - 1) >> 3;
    auto byte_mask = [](std::int64_t lo, std::int64_t hi) {  // bits [lo, hi) within one byte
        return static_cast<std::uint8_t>(((1u << hi) - 1) & ~((1u << lo) - 1));
    };
    if (first == last) return __builtin_popcount(bits[first] & byte_mask(start & 7, ((end - 1) & 7) + 1));
    std::int64_t count = __builtin_popcount(bits[first] & byte_mask(start & 7, 8));
    for (std::int64_t b = first + 1; b < last; ++b) count += __builtin_popcount(bits[b]);
    count += __builtin_popcount(bits[last] & byte_mask(0, ((end - 1) & 7) + 1));
    return count;
}

}  // namespace sd::bitmap
