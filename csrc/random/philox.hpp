// Philox4x64-10 (Salmon, Moraes, Dror, Shaw, "Parallel random numbers: as easy
// as 1, 2, 3", SC 2011; Random123), the generator of sd.random (docs/design/random.md
// §2, R5): a keyed bijection of 256-bit counters, so any counter's output is
// computed directly -- no state, every element in parallel, the same bits on any
// number of threads.
//
// How sd.random uses the counters (docs/design/random.md §3): element e of the
// generator's stream (e = position + index in the call) takes
//   its first word:     word e mod 4 of the block at counter (e / 4, 0, 0, 0)
//   its j-th extra one: words of the blocks at counter (e, j, 0, 0), j >= 1
// (the extra words serve rejection sampling). The two kinds of counter never meet
// (the second word is 0 for the first, >= 1 for the second), so no word is used twice.
#pragma once

#include <array>
#include <cstdint>

namespace sd::random {

using Block = std::array<std::uint64_t, 4>;
struct Key {
    std::uint64_t k0, k1;
};

inline void mulhilo(std::uint64_t a, std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo) {
    const unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
    hi = static_cast<std::uint64_t>(p >> 64);
    lo = static_cast<std::uint64_t>(p);
}

// Ten rounds, the key bumped by the Weyl constants between them (Random123's
// philox4x64_R with R = 10; checked against its known-answer vectors).
inline Block philox(Block c, Key key) {
    constexpr std::uint64_t kM0 = 0xD2E7470EE14C6C93ULL, kM1 = 0xCA5A826395121157ULL;
    constexpr std::uint64_t kW0 = 0x9E3779B97F4A7C15ULL, kW1 = 0xBB67AE8584CAA73BULL;
    std::uint64_t k0 = key.k0, k1 = key.k1;
    for (int r = 0; r < 10; ++r) {
        if (r > 0) k0 += kW0, k1 += kW1;
        std::uint64_t hi0, lo0, hi1, lo1;
        mulhilo(kM0, c[0], hi0, lo0);
        mulhilo(kM1, c[2], hi1, lo1);
        c = {hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0};
    }
    return c;
}

// Four blocks at once, the rounds interleaved: one block is a chain of 20 dependent
// multiplies (latency-bound, ~12 ns); four independent chains fill the multiplier
// (measured: 10^6 uniforms 2.98 -> 1.19 ms). Same bits as philox() on each. The counters
// are (c0[q], c1, 0, 0).
inline void philox4(const std::uint64_t* c0, Key key, std::array<std::uint64_t, 16>& out, std::uint64_t c1 = 0) {
    constexpr std::uint64_t kM0 = 0xD2E7470EE14C6C93ULL, kM1 = 0xCA5A826395121157ULL;
    constexpr std::uint64_t kW0 = 0x9E3779B97F4A7C15ULL, kW1 = 0xBB67AE8584CAA73BULL;
    std::uint64_t a[4], b[4], c[4], d[4];  // the four counter words of each block (b, c, d start at 0)
    for (int q = 0; q < 4; ++q) a[q] = c0[q], b[q] = c1, c[q] = 0, d[q] = 0;
    std::uint64_t k0 = key.k0, k1 = key.k1;
    for (int r = 0; r < 10; ++r) {
        if (r > 0) k0 += kW0, k1 += kW1;
        for (int q = 0; q < 4; ++q) {
            std::uint64_t hi0, lo0, hi1, lo1;
            mulhilo(kM0, a[q], hi0, lo0);
            mulhilo(kM1, c[q], hi1, lo1);
            const std::uint64_t na = hi1 ^ b[q] ^ k0, nc = hi0 ^ d[q] ^ k1;
            a[q] = na, b[q] = lo1, c[q] = nc, d[q] = lo0;
        }
    }
    for (int q = 0; q < 4; ++q) out[4 * q] = a[q], out[4 * q + 1] = b[q], out[4 * q + 2] = c[q], out[4 * q + 3] = d[q];
}

// The words of a stream, by element number: first words in groups of sixteen
// consecutive elements (four blocks computed together), extra words on demand.
class Stream {
public:
    explicit Stream(Key key) : key_(key) {}
    // The first word of element e (consecutive elements share a group).
    std::uint64_t first(std::uint64_t e) {
        const std::uint64_t g = e >> 4;
        if (!have_ || g != group_) {
            const std::uint64_t blocks[4] = {4 * g, 4 * g + 1, 4 * g + 2, 4 * g + 3};
            philox4(blocks, key_, words_);
            group_ = g, have_ = true;
        }
        return words_[e & 15];
    }
    // Extra word k (0, 1, 2, ...) of element e: word k mod 4 of block j = 1 + k / 4.
    std::uint64_t extra(std::uint64_t e, std::uint64_t k) const { return philox({e, 1 + (k >> 2), 0, 0}, key_)[k & 3]; }
    Key key() const { return key_; }

private:
    Key key_;
    std::array<std::uint64_t, 16> words_{};
    std::uint64_t group_ = 0;
    bool have_ = false;
};

// [0, 1) with 53 random bits: the top 53 of the word times 2^-53 (numpy's conversion).
inline double to_unit(std::uint64_t w) { return static_cast<double>(w >> 11) * 0x1p-53; }

// [0, r) for 0 < r (r = 0 meaning 2^64), Lemire's nearly divisionless method
// ("Fast random integer generation in an interval", 2019): the high word of
// w * r, rejecting the rare low words below 2^64 mod r so every value is equally
// likely. Rejections draw element e's extra words, in order.
template <typename Extra>
inline std::uint64_t bounded(std::uint64_t w, std::uint64_t r, Extra&& extra) {
    if (r == 0) return w;  // the full range
    unsigned __int128 m = static_cast<unsigned __int128>(w) * r;
    std::uint64_t lo = static_cast<std::uint64_t>(m);
    if (lo < r) {
        const std::uint64_t t = (0 - r) % r;  // 2^64 mod r
        for (std::uint64_t k = 0; lo < t; ++k) {
            m = static_cast<unsigned __int128>(extra(k)) * r;
            lo = static_cast<std::uint64_t>(m);
        }
    }
    return static_cast<std::uint64_t>(m >> 64);
}

}  // namespace sd::random
