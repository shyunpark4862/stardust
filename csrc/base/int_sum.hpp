// The exact sum of int64 values, for the means of integer data in every area
// (array mean(int64), frame Series and groupby means of int64 columns).
//
// Each value v is split into its upper and lower 32 bits,
//     v = hi * 2^32 + lo,   hi = v >> 32 (signed, |hi| <= 2^31), lo = v & 0xFFFFFFFF (< 2^32),
// and the two halves are summed in 64-bit integers: up to 2^31 values neither
// sum can overflow, and the inner loop is two integer adds that vectorize. The
// halves are merged into one __int128, which holds the exact sum of any number
// of int64 values an array can have. Integer addition does not depend on the
// order, so the sum may be split into pieces in any way (on the pool) and the
// pieces merged in any order: the bits never depend on the thread count. The
// mean is then rounded once to double and divided once by the count
// (docs/design/array.md §6.8).
//
// Interface:
//     IntSum s;                 // zero
//     s.add(v);                 // one value (int64; bool as 0 or 1)
//     s.add_range(p, n);        // n contiguous int64 values
//     s.merge(other);           // another piece's sum
//     s.value()                 // the exact sum, __int128
//     s.to_double()             // the sum rounded to nearest double
//     mean_of(s, count)         // s.to_double() / count (NaN for count 0)
#pragma once

#include <cstdint>
#include <limits>

namespace sd {

class IntSum {
public:
    void add(std::int64_t v) {
        hi_ += v >> 32;
        lo_ += static_cast<std::uint64_t>(v) & 0xFFFFFFFFu;
        if (++pending_ == kFlush) flush();
    }

    // n contiguous values; the halves are flushed every kFlush values.
    void add_range(const std::int64_t* p, std::int64_t n) {
        while (n > 0) {
            const std::int64_t m = n < kFlush - pending_ ? n : kFlush - pending_;
            std::int64_t hi = 0;
            std::uint64_t lo = 0;
            for (std::int64_t i = 0; i < m; ++i) {
                hi += p[i] >> 32;
                lo += static_cast<std::uint64_t>(p[i]) & 0xFFFFFFFFu;
            }
            hi_ += hi;
            lo_ += lo;
            pending_ += m;
            if (pending_ == kFlush) flush();
            p += m;
            n -= m;
        }
    }

    void merge(const IntSum& o) { total_ += o.value(); }

    __int128 value() const { return total_ + (static_cast<__int128>(hi_) << 32) + static_cast<__int128>(lo_); }

    // Round to nearest (the compiler's __int128 -> double conversion rounds correctly).
    double to_double() const { return static_cast<double>(value()); }

private:
    // 2^31 values: |sum of hi| <= 2^62 and sum of lo < 2^63, both in range.
    static constexpr std::int64_t kFlush = std::int64_t{1} << 31;

    void flush() {
        total_ += (static_cast<__int128>(hi_) << 32) + static_cast<__int128>(lo_);
        hi_ = 0;
        lo_ = 0;
        pending_ = 0;
    }

    __int128 total_ = 0;
    std::int64_t hi_ = 0;
    std::uint64_t lo_ = 0;
    std::int64_t pending_ = 0;
};

inline double mean_of(const IntSum& s, std::int64_t count) {
    return count == 0 ? std::numeric_limits<double>::quiet_NaN() : s.to_double() / static_cast<double>(count);
}

}  // namespace sd
