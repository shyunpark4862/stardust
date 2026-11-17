// Samples (T7, docs/design/stats.md §4): element e of a call (e = the generator's
// position + the C-order index) draws its words from sd.random's stream by element
// (docs/design/random.md §3): the first word, then extra words in order for the
// methods that need more (rejection, compositions). So every sample is a function of
// (key, e, parameters) alone: the same bits on any number of threads, and a call split
// in two gives the same values.
//
// A distribution's own method is a specialization (stats/sample_dist.hpp)
//     template <> struct Custom<D> { static double draw(Words& w, params...); };
// without one the sample is by inversion, icdf(u) for u <= 1/2 and iccdf(1 - u) above (the
// smaller side: both tails reach 2^-54), u = (m + 1/2) 2^-53 from the word's top 53 bits
// (never 0 or 1; 1 - u exact).
#pragma once

#include "random/philox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::stats::smp {

// The words of one element.
class Words {
public:
    Words(random::Stream& s, std::uint64_t e) : s_(s), e_(e) {}
    // with the first word computed already (the kernels compute a group of sixteen and hand
    // them out: the stream's own lookup per element kept a sampler's state in memory, the
    // normal's 5.6 ns of which 3.6 were that)
    Words(random::Stream& s, std::uint64_t e, std::uint64_t first) : s_(s), e_(e), first_(first), have_(true) {}
    // and with the first n extra words computed already (whole blocks, n a multiple of four,
    // alive as long as this object: the kernels compute four elements' blocks together, one
    // block alone is latency-bound, 12 ns against 2.7 an element)
    Words(random::Stream& s, std::uint64_t e, std::uint64_t first, const std::uint64_t* extras, std::uint64_t n = 4)
        : s_(s), e_(e), first_(first), pre_(extras), npre_(n), have_(true) {}
    std::uint64_t next() {
        if (!used_) {
            used_ = true;
            return have_ ? first_ : s_.first(e_);
        }
        if (k_ < npre_) return pre_[k_++];
        const std::uint64_t j = 1 + (k_ >> 2);  // extra word k: word k mod 4 of block (e, 1 + k / 4)
        if (j != block_) refill(j);
        return words_[k_++ & 3];
    }
    // (0, 1), symmetric about 1/2: (m + 1/2) 2^-53
    double u() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1p-53; }
    // [0, 1): m 2^-53 (sd.random's random())
    double u01() { return static_cast<double>(next() >> 11) * 0x1p-53; }

private:
    // out of line: next() stays small enough to inline into the samplers' common paths
    [[gnu::noinline]] void refill(std::uint64_t j) {
        words_ = random::philox({e_, j, 0, 0}, s_.key());
        block_ = j;
    }
    random::Stream& s_;
    std::uint64_t e_;
    std::uint64_t k_ = 0, block_ = 0, first_ = 0;
    const std::uint64_t* pre_ = nullptr;
    std::uint64_t npre_ = 0;
    bool used_ = false, have_ = false;
    random::Block words_;  // read only after it is set (block_ != 0)
};

// The first nb extra blocks of four elements, for Words' extras: element q's 4 nb words at
// out + q * 4 * nb (block j's four at 4 (j - 1)); the four elements' blocks interleaved.
inline void extra_blocks4(random::Key key, const std::uint64_t* es, std::uint64_t nb, std::uint64_t* out) {
    std::array<std::uint64_t, 16> blk;
    for (std::uint64_t j = 1; j <= nb; ++j) {
        random::philox4(es, key, blk, j);
        for (std::uint64_t q = 0; q < 4; ++q)
            for (std::uint64_t t = 0; t < 4; ++t) out[q * 4 * nb + 4 * (j - 1) + t] = blk[4 * q + t];
    }
}

template <class D, class... P>
double by_inverse(Words& w, P... p) {
    const double u = w.u();
    return u <= 0.5 ? D::icdf(u, p...) : D::iccdf(1.0 - u, p...);
}

template <class D>
struct Custom {};  // specialized for the distributions with their own method (sample_dist.hpp)

template <class D, class... P>
double sample(Words& w, P... p) {
    if constexpr (requires { Custom<D>::draw(w, p...); }) return Custom<D>::draw(w, p...);
    else return by_inverse<D>(w, p...);
}

// The first words of elements [i0, i1) of a call at stream position pos, a group of sixteen
// at a time (random::Stream's groups: the same words): fn(i, first word of element pos + i).
template <class F>
inline void each_first(random::Key key, std::uint64_t pos, std::int64_t i0, std::int64_t i1, F&& fn) {
    std::array<std::uint64_t, 16> words;
    for (std::int64_t i = i0; i < i1;) {
        const std::uint64_t e = pos + static_cast<std::uint64_t>(i), g = e >> 4;
        const std::uint64_t blocks[4] = {4 * g, 4 * g + 1, 4 * g + 2, 4 * g + 3};
        random::philox4(blocks, key, words);
        const std::int64_t end = std::min<std::int64_t>(i1, i + static_cast<std::int64_t>(16 - (e & 15)));
        for (std::uint64_t q = e & 15; i < end; ++i, ++q) fn(i, words[q]);
    }
}

// fn(i, Words&) for the elements [i0, i1) of a call at stream position pos, each element's
// Words with its first word and its first nb extra blocks computed already, four elements
// at a time (buf: scratch). For the kernels whose elements draw several words each (a row
// of a multivariate sample).
template <class F>
inline void each_words(random::Stream& s, std::uint64_t pos, std::int64_t i0, std::int64_t i1, std::uint64_t nb,
                       std::vector<std::uint64_t>& buf, F&& fn) {
    buf.resize(static_cast<std::size_t>(16 * nb));
    std::uint64_t es[4] = {}, fw[4] = {};
    std::int64_t at4 = 0;
    int have = 0;
    auto flush = [&] {
        for (int q = have; q < 4; ++q) es[q] = es[have - 1];  // a short last group: any element
        extra_blocks4(s.key(), es, nb, buf.data());
        for (int q = 0; q < have; ++q) {
            Words w(s, es[q], fw[q], buf.data() + q * 4 * nb, 4 * nb);
            fn(at4 + q, w);
        }
        have = 0;
    };
    each_first(s.key(), pos, i0, i1, [&](std::int64_t i, std::uint64_t first) {
        if (have == 0) at4 = i;
        es[have] = pos + static_cast<std::uint64_t>(i), fw[have] = first;
        if (++have == 4) flush();
    });
    if (have) flush();
}

}  // namespace sd::stats::smp
