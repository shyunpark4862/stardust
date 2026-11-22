// The validity bitmap (docs/design/frame.md §4.4): one bit per row, LSB first
// (row i is bit i % 8 of byte i / 8, as in Arrow); 1 means a value, 0 means NA.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace sd::frame::bitmap {

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
    std::int64_t b = first + 1;
    for (; b + 8 <= last; b += 8) {  // eight bytes per popcount
        std::uint64_t w;
        std::memcpy(&w, bits + b, 8);
        count += __builtin_popcountll(w);
    }
    for (; b < last; ++b) count += __builtin_popcount(bits[b]);
    count += __builtin_popcount(bits[last] & byte_mask(0, ((end - 1) & 7) + 1));
    return count;
}

// Copies bits [offset, offset + n) of `src` to bits [0, n) of `dst`, a byte
// at a time: output byte j is input bytes k and k + 1 shifted together. The
// unused high bits of the last output byte are cleared. Never reads past the
// byte that holds bit offset + n - 1.
inline void copy_bits(const std::uint8_t* src, std::int64_t offset, std::int64_t n, std::uint8_t* dst) {
    if (n <= 0) return;
    const std::int64_t nbytes = num_bytes(n);
    const std::int64_t first = offset >> 3, last_src = (offset + n - 1) >> 3;
    const int shift = static_cast<int>(offset & 7);
    for (std::int64_t j = 0; j < nbytes; ++j) {
        const std::int64_t k = first + j;
        unsigned v = src[k] >> shift;
        if (shift != 0 && k + 1 <= last_src) v |= static_cast<unsigned>(src[k + 1]) << (8 - shift);
        dst[j] = static_cast<std::uint8_t>(v);
    }
    if (n & 7) dst[nbytes - 1] &= static_cast<std::uint8_t>((1u << (n & 7)) - 1);
}

// For each byte value b, eight bytes holding bit k of b in byte k (0 or 1).
// Expanding a bitmap through it handles eight rows per lookup, with no
// branch per row (docs/design/frame.md §4.4).
inline constexpr std::array<std::uint64_t, 256> kExpand = [] {
    std::array<std::uint64_t, 256> t{};
    for (unsigned b = 0; b < 256; ++b)
        for (unsigned k = 0; k < 8; ++k) t[b] |= static_cast<std::uint64_t>((b >> k) & 1) << (8 * k);
    return t;
}();

// Bits [0, n) of `bits` (starting at bit 0) as n bytes of 0 or 1.
inline void expand(const std::uint8_t* bits, std::int64_t n, std::uint8_t* out) {
    const std::int64_t full = n / 8;
    for (std::int64_t j = 0; j < full; ++j) std::memcpy(out + 8 * j, &kExpand[bits[j]], 8);
    for (std::int64_t i = full * 8; i < n; ++i) out[i] = get(bits, i);
}

}  // namespace sd::frame::bitmap
