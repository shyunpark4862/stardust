// The wide path of the predicates (docs/design/spatial_geometry.md §3.1, decision SPT100): signed
// multi-word integers with addition, subtraction, schoolbook multiplication, the sign and a correctly
// rounded conversion to float64, and the reading of a float64 as integer * 2^e.
//
// Every finite float64 is m * 2^e with an integer m < 2^53 and e >= -1074, so the coordinates of the
// points of one predicate call, divided by 2^(their lowest set bit), are integers below 2^2098. A
// determinant of their differences is then an integer computed without any rounding: its sign is the
// predicate, for every finite input (no overflow, no underflow, no range to check).
//
// Capacity. A difference is below 2^2099; the largest value the predicates build is insphere's
// determinant: a lift (three squares, < 2^4200) times a 3 x 3 minor (six products of three, < 2^6300),
// four of them: below 2^10502, 165 words. kCap = 168 leaves room for the untrimmed top word of a product.
// The length is the number of words in use, so the cost follows the actual size: points in [0, 1) with
// 53-bit fractions are one word, and insphere's result five.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace sd::spatial {

// |x| = m * 2^e with m < 2^53 (x finite; m = 0 for zero).
struct FloatParts {
    std::uint64_t m;
    int e;
};

inline FloatParts float_parts(double x) {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    const std::uint64_t frac = bits & ((std::uint64_t{1} << 52) - 1);
    const int field = static_cast<int>((bits >> 52) & 0x7ff);
    if (field == 0) return {frac, -1074};
    return {frac | (std::uint64_t{1} << 52), field - 1075};
}

// The exponent of the lowest set bit of x, and floor(log2 |x|). x is finite and not zero.
inline int low_exponent(double x) {
    const FloatParts p = float_parts(x);
    return p.e + std::countr_zero(p.m);
}

inline int high_exponent(double x) {
    const FloatParts p = float_parts(x);
    return p.e + 63 - std::countl_zero(p.m);
}

class WideInt {
public:
    static constexpr int kCap = 168;

    WideInt() = default;
    WideInt(const WideInt& o) : n_(o.n_), neg_(o.neg_) { std::memcpy(w_, o.w_, sizeof(std::uint64_t) * n_); }
    WideInt& operator=(const WideInt& o) {
        if (this != &o) {
            n_ = o.n_;
            neg_ = o.neg_;
            std::memcpy(w_, o.w_, sizeof(std::uint64_t) * n_);
        }
        return *this;
    }

    // x / 2^e. x is finite and e <= low_exponent(x) (so the quotient is an integer); zero gives zero.
    static WideInt scaled(double x, int e) {
        WideInt r;
        if (x == 0.0) return r;
        const FloatParts p = float_parts(x);
        const int zeros = std::countr_zero(p.m);
        const std::uint64_t m = p.m >> zeros;  // odd: |x| = m 2^(p.e + zeros)
        const int s = p.e + zeros - e;         // >= 0 by the caller's promise
        const int limb = s / 64, bit = s % 64;
        for (int i = 0; i < limb; ++i) r.w_[i] = 0;
        r.w_[limb] = m << bit;
        r.n_ = limb + 1;
        if (bit != 0) {
            const std::uint64_t hi = m >> (64 - bit);
            if (hi != 0) r.w_[r.n_++] = hi;
        }
        r.trim();
        r.neg_ = std::signbit(x);
        return r;
    }

    int sign() const { return n_ == 0 ? 0 : (neg_ ? -1 : 1); }

    friend WideInt add(const WideInt& a, const WideInt& b) { return combine(a, b, false); }
    friend WideInt sub(const WideInt& a, const WideInt& b) { return combine(a, b, true); }

    friend WideInt mul(const WideInt& a, const WideInt& b) {
        WideInt r;
        if (a.n_ == 0 || b.n_ == 0) return r;
        const int n = a.n_ + b.n_;
        std::memset(r.w_, 0, sizeof(std::uint64_t) * n);
        for (int i = 0; i < a.n_; ++i) {
            std::uint64_t carry = 0;
            const unsigned __int128 ai = a.w_[i];
            for (int j = 0; j < b.n_; ++j) {
                const unsigned __int128 t = ai * b.w_[j] + r.w_[i + j] + carry;
                r.w_[i + j] = static_cast<std::uint64_t>(t);
                carry = static_cast<std::uint64_t>(t >> 64);
            }
            r.w_[i + b.n_] = carry;
        }
        r.n_ = n;
        r.trim();
        r.neg_ = a.neg_ != b.neg_;
        return r;
    }

    // The float64 nearest to value * 2^e (ties to even); +-inf when it overflows, and the correctly
    // rounded subnormal (or zero) when it underflows.
    double to_double(std::int64_t e) const {
        if (n_ == 0) return 0.0;
        const std::int64_t len = bit_length();
        const std::int64_t top = len - 1 + e;  // floor(log2 |value 2^e|)
        const double sgn = neg_ ? -1.0 : 1.0;
        if (top >= 1024) return sgn * std::numeric_limits<double>::infinity();
        const std::int64_t prec = top >= -1022 ? 53 : top + 1075;  // the bits a float64 keeps at this size
        const std::int64_t drop = len - prec;
        if (drop <= 0) return sgn * std::ldexp(static_cast<double>(w_[0]), static_cast<int>(e));
        if (prec < 0) return sgn * 0.0;  // below half of the smallest subnormal
        std::uint64_t kept = prec == 0 ? 0 : (window(drop) & (~std::uint64_t{0} >> (64 - prec)));
        const bool half = bit(drop - 1);
        if (half && (any_below(drop - 1) || (kept & 1))) ++kept;
        return sgn * std::ldexp(static_cast<double>(kept), static_cast<int>(e + drop));
    }

    // For the tests: "-0x1f" style, "0x0" for zero.
    std::string hex() const {
        if (n_ == 0) return "0x0";
        static const char* digits = "0123456789abcdef";
        std::string s = neg_ ? "-0x" : "0x";
        bool started = false;
        for (int i = n_ - 1; i >= 0; --i)
            for (int k = 60; k >= 0; k -= 4) {
                const int d = static_cast<int>((w_[i] >> k) & 0xf);
                if (d != 0) started = true;
                if (started) s.push_back(digits[d]);
            }
        return s;
    }

    int words() const { return n_; }

    // floor(log2 |value|) + 1, 0 for zero (the circumcentre's scaled quotient, voronoi.cpp).
    std::int64_t bits() const { return n_ == 0 ? 0 : bit_length(); }

private:
    void trim() {
        while (n_ > 0 && w_[n_ - 1] == 0) --n_;
        if (n_ == 0) neg_ = false;
    }

    std::int64_t bit_length() const { return 64 * std::int64_t{n_} - std::countl_zero(w_[n_ - 1]); }

    bool bit(std::int64_t pos) const {
        const std::int64_t limb = pos / 64;
        return limb < n_ && ((w_[limb] >> (pos % 64)) & 1) != 0;
    }

    // The 64 bits starting at bit `pos`.
    std::uint64_t window(std::int64_t pos) const {
        const std::int64_t limb = pos / 64;
        const int sh = static_cast<int>(pos % 64);
        if (limb >= n_) return 0;
        std::uint64_t v = w_[limb] >> sh;
        if (sh != 0 && limb + 1 < n_) v |= w_[limb + 1] << (64 - sh);
        return v;
    }

    bool any_below(std::int64_t pos) const {
        const std::int64_t limb = pos / 64;
        for (std::int64_t i = 0; i < limb && i < n_; ++i)
            if (w_[i] != 0) return true;
        const int sh = static_cast<int>(pos % 64);
        return limb < n_ && sh != 0 && (w_[limb] & ((std::uint64_t{1} << sh) - 1)) != 0;
    }

    static int compare_magnitude(const WideInt& a, const WideInt& b) {
        if (a.n_ != b.n_) return a.n_ < b.n_ ? -1 : 1;
        for (int i = a.n_ - 1; i >= 0; --i)
            if (a.w_[i] != b.w_[i]) return a.w_[i] < b.w_[i] ? -1 : 1;
        return 0;
    }

    // |big| + |small| into r (big.n_ >= small.n_).
    static void add_magnitude(const WideInt& big, const WideInt& small, WideInt& r) {
        std::uint64_t carry = 0;
        for (int i = 0; i < big.n_; ++i) {
            const unsigned __int128 t =
                static_cast<unsigned __int128>(big.w_[i]) + (i < small.n_ ? small.w_[i] : 0) + carry;
            r.w_[i] = static_cast<std::uint64_t>(t);
            carry = static_cast<std::uint64_t>(t >> 64);
        }
        r.n_ = big.n_;
        if (carry != 0) r.w_[r.n_++] = carry;
    }

    // |big| - |small| into r (|big| >= |small|).
    static void sub_magnitude(const WideInt& big, const WideInt& small, WideInt& r) {
        std::uint64_t borrow = 0;
        for (int i = 0; i < big.n_; ++i) {
            const std::uint64_t s = i < small.n_ ? small.w_[i] : 0;
            const std::uint64_t d = big.w_[i] - s - borrow;
            borrow = (big.w_[i] < s || (big.w_[i] == s && borrow != 0)) ? 1 : 0;
            r.w_[i] = d;
        }
        r.n_ = big.n_;
        r.trim();
    }

    static WideInt combine(const WideInt& a, const WideInt& b, bool negate_b) {
        WideInt r;
        const bool bneg = b.neg_ != negate_b;
        if (b.n_ == 0) return a;
        if (a.n_ == 0) {
            r = b;
            r.neg_ = bneg;
            return r;
        }
        if (a.neg_ == bneg) {
            if (a.n_ >= b.n_) add_magnitude(a, b, r);
            else add_magnitude(b, a, r);
            r.neg_ = a.neg_;
            return r;
        }
        const int c = compare_magnitude(a, b);
        if (c == 0) return r;
        if (c > 0) {
            sub_magnitude(a, b, r);
            r.neg_ = a.neg_;
        } else {
            sub_magnitude(b, a, r);
            r.neg_ = bneg;
        }
        return r;
    }

    int n_ = 0;
    bool neg_ = false;
    std::uint64_t w_[kCap];
};

inline int sign(const WideInt& a) { return a.sign(); }

}  // namespace sd::spatial
