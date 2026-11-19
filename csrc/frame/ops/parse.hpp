// The strict text grammars of docs/api/frame.md §3.5, shared by astype and
// read_csv so that both read "12", "nan" or " 1" the same way
// (docs/design/frame.md §7.5).
#pragma once

#include "frame/ops/pow5_table.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace sd::frame::text {

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

// [+-]? digits, ASCII only.
inline bool int_grammar(std::string_view s) {
    if (!s.empty() && (s[0] == '+' || s[0] == '-')) s.remove_prefix(1);
    if (s.empty()) return false;
    for (char c : s)
        if (!is_digit(c)) return false;
    return true;
}

// An int64 from text in int_grammar; false when it is out of range.
inline bool to_int(std::string_view s, std::int64_t& v) {
    const char* first = s.data() + (s[0] == '+' ? 1 : 0);  // from_chars takes '-' but not '+'
    return std::from_chars(first, s.data() + s.size(), v).ec == std::errc{};
}

// [+-]? ( digits [. digits?] | . digits ) ( [eE] [+-]? digits )?   or   [+-]? (nan | inf | infinity), any case
// Whether `s` is `word` (lowercase) ignoring ASCII case, without copying.
inline bool equals_lower(std::string_view s, std::string_view word) {
    if (s.size() != word.size()) return false;
    for (std::size_t k = 0; k < s.size(); ++k)
        if ((s[k] >= 'A' && s[k] <= 'Z' ? s[k] - 'A' + 'a' : s[k]) != word[k]) return false;
    return true;
}

inline bool float_grammar(std::string_view s) {
    std::size_t i = 0, n = s.size();
    if (i < n && (s[i] == '+' || s[i] == '-')) ++i;
    const std::string_view rest = s.substr(i);
    if (!rest.empty() && !is_digit(rest[0]) && rest[0] != '.')  // only then can it be nan or inf
        return equals_lower(rest, "nan") || equals_lower(rest, "inf") || equals_lower(rest, "infinity");
    std::size_t digits = 0;
    while (i < n && is_digit(s[i])) ++i, ++digits;
    if (i < n && s[i] == '.') {
        ++i;
        while (i < n && is_digit(s[i])) ++i, ++digits;
    }
    if (digits == 0) return false;
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < n && (s[i] == '+' || s[i] == '-')) ++i;
        std::size_t exp_digits = 0;
        while (i < n && is_digit(s[i])) ++i, ++exp_digits;
        if (exp_digits == 0) return false;
    }
    return i == n;
}

// ---- Eisel-Lemire (docs/design/frame.md §10.4): decimal -> double, correctly rounded
//
// The text is read as w * 10^q with w of at most 19 digits. Then:
// 1. Clinger's fast path: |q| <= 22 and w <= 2^53 are both exact doubles, so
//    one IEEE multiply or divide gives the correctly rounded result.
// 2. Eisel-Lemire: w (shifted to fill 64 bits) times a 128-bit truncation of
//    5^q (pow5_table.hpp); the top bits of the product are the mantissa and
//    q * log2(10) the exponent. Lemire showed that this 128-bit product
//    decides the rounding except in cases it can detect; those, more than
//    19 digits, nan and inf go to from_chars. This is the algorithm of
//    fast_float (Lemire, "Number Parsing at a Gigabyte per Second", 2021),
//    written here from the paper.

namespace detail {

inline constexpr double kExactPow10[23] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
                                           1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

// The binary exponent of 10^q plus 63: floor(q * log2(10)) + 63, computed in
// fixed point (152170 + 65536 = log2(10) * 2^16, rounded).
inline int power2_of(std::int64_t q) { return static_cast<int>(((152170 + 65536) * q) >> 16) + 63; }

// The Eisel-Lemire step: the bits of the double nearest w * 10^q (w != 0,
// positive), or false when the 128-bit product cannot decide.
inline bool eisel_lemire(std::uint64_t w, std::int64_t q, std::uint64_t& bits) {
    constexpr int kMantissa = 52, kMinExp = -1023, kInfinite = 0x7FF;
    if (q < kSmallestPowerOfFive) {
        bits = 0;
        return true;
    }
    if (q > kLargestPowerOfFive) {
        bits = std::uint64_t{kInfinite} << kMantissa;
        return true;
    }
    const int lz = __builtin_clzll(w);
    w <<= lz;
    const std::size_t index = 2 * static_cast<std::size_t>(q - kSmallestPowerOfFive);
    unsigned __int128 first = static_cast<unsigned __int128>(w) * kPowerOfFive128[index];
    std::uint64_t hi = static_cast<std::uint64_t>(first >> 64), lo = static_cast<std::uint64_t>(first);
    if ((hi & 0x1FF) == 0x1FF) {  // the low bits of the 64-bit truncation are all ones: add the next word
        const unsigned __int128 second = static_cast<unsigned __int128>(w) * kPowerOfFive128[index + 1];
        const std::uint64_t carry = static_cast<std::uint64_t>(second >> 64);
        lo += carry;
        if (carry > lo) ++hi;
        if ((hi & 0x1FF) == 0x1FF && lo == ~std::uint64_t{0} && (q < -27 || q > 55)) return false;  // undecided
    }
    const int upper = static_cast<int>(hi >> 63);
    const int shift = upper + 64 - kMantissa - 3;
    std::uint64_t mantissa = hi >> shift;
    int power2 = power2_of(q) + upper - lz - kMinExp;
    if (power2 <= 0) {  // a subnormal (or zero)
        if (-power2 + 1 >= 64) {
            bits = 0;
            return true;
        }
        mantissa >>= -power2 + 1;
        mantissa += mantissa & 1;
        mantissa >>= 1;
        power2 = mantissa < (std::uint64_t{1} << kMantissa) ? 0 : 1;
        bits = (mantissa & ~(std::uint64_t{1} << kMantissa)) | (static_cast<std::uint64_t>(power2) << kMantissa);
        return true;
    }
    // Exactly halfway between two doubles: round half to even, not up.
    if (lo <= 1 && q >= -4 && q <= 23 && (mantissa & 3) == 1 && (mantissa << shift) == hi)
        mantissa &= ~std::uint64_t{1};
    mantissa += mantissa & 1;
    mantissa >>= 1;
    if (mantissa >= (std::uint64_t{2} << kMantissa)) {  // rounding carried into a new bit
        mantissa = std::uint64_t{1} << kMantissa;
        ++power2;
    }
    mantissa &= ~(std::uint64_t{1} << kMantissa);
    if (power2 >= kInfinite) {
        bits = std::uint64_t{kInfinite} << kMantissa;
        return true;
    }
    bits = mantissa | (static_cast<std::uint64_t>(power2) << kMantissa);
    return true;
}

// Eight ASCII digits at once (Lemire's SWAR trick): a little-endian load of
// 8 bytes, a test that each byte is '0'..'9', and three multiplies that
// combine them pairwise into the 8-digit number.
inline bool eight_digits(std::uint64_t v) {
    return !(((v + 0x4646464646464646ULL) | (v - 0x3030303030303030ULL)) & 0x8080808080808080ULL);
}
inline std::uint64_t parse_eight(std::uint64_t v) {
    v -= 0x3030303030303030ULL;
    v = (v * 10) + (v >> 8);  // pairs of digits
    return (
        (((v & 0x000000FF000000FFULL) * 0x000F424000000064ULL) +
         (((v >> 16) & 0x000000FF000000FFULL) * 0x0000271000000001ULL)) >>
        32
    );
}

// Digits from p while they last, into w (8 at a time while 8 remain).
inline const char* read_digits(const char* p, const char* end, std::uint64_t& w) {
    while (end - p >= 8) {
        std::uint64_t v;
        std::memcpy(&v, p, 8);
        if (!eight_digits(v)) break;
        w = w * 100000000 + parse_eight(v);
        p += 8;
    }
    while (p < end && is_digit(*p)) w = w * 10 + static_cast<std::uint64_t>(*p++ - '0');
    return p;
}

enum class Parsed { Ok, Bad, Slow };  // Slow: valid text this fast path leaves to the slow one

// One pass over the text: checks the decimal form of float_grammar while
// reading it (docs/design/frame.md §10.4). Bad means the text is not float_grammar; Slow means
// nan, inf, more than 19 significant digits, or a product Eisel-Lemire
// cannot decide.
inline Parsed parse_decimal(std::string_view s, double& out) {
    const char* p = s.data();
    const char* const end = p + s.size();
    bool negative = false;
    if (p < end && (*p == '+' || *p == '-')) negative = *p++ == '-';
    if (p == end) return Parsed::Bad;
    if (!is_digit(*p) && *p != '.') return Parsed::Slow;  // nan, inf, or not a number (float_grammar decides)
    // Every digit goes into w; leading zeros add nothing to it. With more
    // than 19 significant digits w may have wrapped: the slow path.
    const char* const digits_start = p;
    std::uint64_t w = 0;
    p = read_digits(p, end, w);
    std::int64_t ndigits = p - digits_start, q = 0;
    if (p < end && *p == '.') {
        const char* frac = ++p;
        p = read_digits(p, end, w);
        ndigits += p - frac;
        q = -(p - frac);
    }
    if (ndigits == 0) return Parsed::Bad;  // "." or "+.": no digits
    if (p < end && (*p == 'e' || *p == 'E')) {
        ++p;
        bool neg_exp = false;
        if (p < end && (*p == '+' || *p == '-')) neg_exp = *p++ == '-';
        if (p == end) return Parsed::Bad;
        std::int64_t e = 0;
        for (; p < end; ++p) {
            if (!is_digit(*p)) return Parsed::Bad;
            e = std::min<std::int64_t>(e * 10 + (*p - '0'), 1'000'000);  // beyond any double anyway
        }
        q += neg_exp ? -e : e;
    }
    if (p != end) return Parsed::Bad;
    if (ndigits > 19) {
        for (const char* z = digits_start; z < end && (*z == '0' || *z == '.'); ++z) ndigits -= *z == '0';
        if (ndigits > 19) return Parsed::Slow;
    }
    double d;
    if (w == 0) {
        d = 0.0;
    } else if (q >= -22 && q <= 22 && w <= (std::uint64_t{1} << 53)) {  // Clinger's fast path
        d = static_cast<double>(w);
        d = q < 0 ? d / kExactPow10[-q] : d * kExactPow10[q];
    } else {
        std::uint64_t bits;
        if (!eisel_lemire(w, q, bits)) return Parsed::Slow;
        std::memcpy(&d, &bits, sizeof d);
    }
    out = negative ? -d : d;
    return Parsed::Ok;
}

// from_chars, with strtod for overflow and deep underflow.
inline double slow_double(std::string_view s) {
    const char* first = s.data() + (s[0] == '+' ? 1 : 0);
    double v = 0;
    auto res = std::from_chars(first, s.data() + s.size(), v);
    if (res.ec == std::errc::result_out_of_range) {
        std::string copy(s);
        v = std::strtod(copy.c_str(), nullptr);
    }
    return v;
}

}  // namespace detail

// int_grammar and to_int in one pass (with the 8-digit SWAR reads,
// docs/design/frame.md §10.4):
// 1 for an int64, 0 when `s` is not int_grammar, -1 when it is but the
// value is beyond int64.
inline int parse_int64(std::string_view s, std::int64_t& v) {
    const char* p = s.data();
    const char* const end = p + s.size();
    bool negative = false;
    if (p < end && (*p == '+' || *p == '-')) negative = *p++ == '-';
    const char* const digits = p;
    std::uint64_t w = 0;
    p = detail::read_digits(p, end, w);
    if (p == digits || p != end) return 0;
    std::int64_t significant = p - digits;
    for (const char* z = digits; z < end && *z == '0' && significant > 19; ++z) --significant;  // leading zeros
    if (significant > 19) return -1;  // w may have wrapped: certainly beyond int64
    const std::uint64_t limit = negative ? (std::uint64_t{1} << 63) : (std::uint64_t{1} << 63) - 1;
    if (w > limit) return -1;
    v = negative ? static_cast<std::int64_t>(0 - w) : static_cast<std::int64_t>(w);
    return 1;
}

// A double from text in float_grammar, correctly rounded; overflow and deep
// underflow give +-inf and +-0.0 (and subnormals) as Python's float() does.
inline double to_double(std::string_view s) {
    double d;
    return detail::parse_decimal(s, d) == detail::Parsed::Ok ? d : detail::slow_double(s);
}

// float_grammar and to_double in one pass: false when `s` is not a float.
inline bool parse_double(std::string_view s, double& out) {
    switch (detail::parse_decimal(s, out)) {
        case detail::Parsed::Ok: return true;
        case detail::Parsed::Bad: return false;
        case detail::Parsed::Slow: break;
    }
    if (!float_grammar(s)) return false;
    out = detail::slow_double(s);
    return true;
}

}  // namespace sd::frame::text
