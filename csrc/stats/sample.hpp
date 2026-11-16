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

#include <cmath>
#include <cstdint>

namespace sd::stats::smp {

// The words of one element.
class Words {
public:
    Words(random::Stream& s, std::uint64_t e) : s_(s), e_(e) {}
    // with block 1 (the first four extra words) computed already (the kernel computes four
    // elements' blocks together: one block alone is latency-bound)
    Words(random::Stream& s, std::uint64_t e, const std::uint64_t* block1) : s_(s), e_(e), block_(1) {
        for (int q = 0; q < 4; ++q) words_[q] = block1[q];
    }
    std::uint64_t next() {
        if (!used_) {
            used_ = true;
            return s_.first(e_);
        }
        const std::uint64_t j = 1 + (k_ >> 2);  // extra word k: word k mod 4 of block (e, 1 + k / 4)
        if (j != block_) {
            words_ = random::philox({e_, j, 0, 0}, s_.key());
            block_ = j;
        }
        return words_[k_++ & 3];
    }
    // (0, 1), symmetric about 1/2: (m + 1/2) 2^-53
    double u() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1p-53; }
    // [0, 1): m 2^-53 (sd.random's random())
    double u01() { return static_cast<double>(next() >> 11) * 0x1p-53; }

private:
    random::Stream& s_;
    std::uint64_t e_;
    std::uint64_t k_ = 0, block_ = 0;
    bool used_ = false;
    random::Block words_;  // read only after it is set (block_ != 0)
};

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

}  // namespace sd::stats::smp
