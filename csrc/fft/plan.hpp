// The plans and their cache (docs/design/fft.md §3.6).
//
// A plan belongs to one (kind, length). The complex kind is computed by the mixed radix for 31-smooth
// lengths (from 2^15 on as a four-step, milestone FT6) and by Bluestein for the others (decisions FT36,
// FT64, FT65; no vDSP, decisions FT30, FT66). The real kind (milestone FT2) exists for even lengths n: the
// complex plan of n / 2 (its own, not the cache's) and the twiddles of the real pass (fft/rfft.hpp); an odd
// real length runs the complex plan of n (decision FT43). The DCT kinds (milestone FT4, fft/r2r.hpp) hold
// the inner plan of their real transforms and its tables: Dct2 (DCT-II and DCT-III of n, DST-II and DST-III
// by signs and reversal) the real transform of n (the real kind for even n, the complex kind for odd n) and
// the twiddles e^(-i pi k / 2n); Dct4 (DCT-IV, DST-IV) the complex plan of n / 2 and the twiddles
// e^(-i pi (8j + 1) / 8n) for even n, the complex plan of n and the index tables of the odd algorithm for
// odd n. DCT-I and DST-I run the real kind of 2 (n - 1) and 2 (n + 1). Which path a length takes depends
// on the kind and the length alone, so a line's bits never depend on the batch, the layout, the thread
// count or the cache. A plan is read-only once made.
//
// The cache: one per process, an LRU of shared_ptr under one mutex, at most kCachePlans plans holding
// at most kCacheBytes together (decision FT24); a plan larger than kCacheBytes on its own is made for
// the call and not kept. A plan is made outside the mutex with the GIL released; when two threads
// make the same plan at once, the first to insert it wins and the other's copy is dropped (both give
// the same bits). An evicted plan is freed when the last call using it ends.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "fft/bluestein.hpp"
#include "fft/cfft.hpp"
#include "fft/rfft.hpp"

namespace sd::fft {

enum class Kind : int { Complex = 0, Real = 1, Dct2 = 2, Dct4 = 3 };

inline constexpr int kCachePlans = 32;
inline constexpr int64_t kCacheBytes = int64_t{256} << 20;

class Plan {
public:
    Plan(Kind kind, int64_t n);

    Kind kind() const { return kind_; }
    int64_t n() const { return n_; }
    bool bluestein() const { return half_ ? half_->bluestein() : blue_ != nullptr; }
    const CfftPlan* mixed() const { return half_ ? half_->mixed() : mixed_.get(); }
    const BluesteinPlan* blue() const { return half_ ? half_->blue() : blue_.get(); }
    const Plan* half() const { return half_.get(); }  // the real kind's complex plan of n / 2; a DCT kind's inner
    const double* table() const { return rtw_.data(); }  // the real kind's or a DCT kind's twiddles
    const int64_t* index() const { return idx_.data(); }  // the odd Dct4's index tables (fft/r2r.hpp)

    // y = the transform of x (interleaved complex128, n each); work: work_doubles() doubles.
    void execute(const double* x, double* y, double* work, bool inverse) const {
        if (blue_) {
            blue_->execute(x, y, work, inverse);
        } else {
            mixed_->execute(x, y, work, inverse);
        }
    }
    // The real kind (fft/rfft.hpp): y (n / 2 + 1 interleaved complex) = the half spectrum of the n reals
    // x; work: work_doubles() doubles. None of x, y, work overlap.
    void r2c(const double* x, double* y, double* work) const {
        half_->execute(x, y, work, false);
        r2c_post(y, n_ / 2, rtw_.data());
    }
    // The real kind's inverse: y (n reals) = the unnormalised inverse transform of the Hermitian half
    // spectrum x (n / 2 + 1 interleaved complex; Conj: of conj(x)); z: n doubles, work: work_doubles()
    // doubles. None of x, y, z, work overlap.
    template <bool Conj>
    void c2r(const double* x, double* y, double* z, double* work) const {
        c2r_pre<Conj>(x, z, n_ / 2, rtw_.data());
        half_->execute(z, y, work, true);
    }

    int64_t work_doubles() const { return half_ ? half_->work_doubles() : blue_ ? blue_->work_doubles() : 2 * n_; }
    int64_t bytes() const {
        if (half_) {
            return half_->bytes() + static_cast<int64_t>(rtw_.size() * sizeof(double) + idx_.size() * sizeof(int64_t));
        }
        return blue_ ? blue_->bytes() : mixed_->bytes();
    }
    // The work of one line in the units of the parallel thresholds (about n log2 n).
    int64_t cost() const { return cost_; }
    // True when a line runs its own tasks on the pool (a four-step inside, fft/cfft.hpp): the driver then
    // transforms the lines one after the other.
    bool parallel_inside() const {
        if (half_) return half_->parallel_inside();
        if (blue_) return blue_->inner().four_step();
        return mixed_->four_step();
    }

private:
    Kind kind_;
    int64_t n_;
    int64_t cost_ = 1;
    std::unique_ptr<CfftPlan> mixed_;
    std::unique_ptr<BluesteinPlan> blue_;
    std::unique_ptr<const Plan> half_;  // the real kind
    std::vector<double> rtw_;  // the real kind: real_twiddles(n); a DCT kind: its twiddles
    std::vector<int64_t> idx_;  // the odd Dct4: the index tables
};

// The plan of (kind, n), from the cache or made now (holding the GIL on entry; it is released while a
// plan is made).
std::shared_ptr<const Plan> get_plan(Kind kind, int64_t n);

struct CacheState {
    int64_t plans = 0, bytes = 0;
    std::vector<int64_t> lengths;  // least recently used first
};
CacheState cache_state();
void cache_clear();

}  // namespace sd::fft
