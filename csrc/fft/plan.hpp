// The plans and their cache (docs/design/fft.md §3.6).
//
// A plan belongs to one (kind, length): this milestone has the complex kind, computed by the mixed
// radix for 11-smooth lengths and by Bluestein for the others (decision FT36; first version: no vDSP,
// decision FT30). Which path a length takes depends on the length alone, so a line's bits never
// depend on the batch, the layout, the thread count or the cache. A plan is read-only once made.
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

namespace sd::fft {

enum class Kind : int { Complex = 0 };

inline constexpr int kCachePlans = 32;
inline constexpr int64_t kCacheBytes = int64_t{256} << 20;

class Plan {
public:
    Plan(Kind kind, int64_t n);

    Kind kind() const { return kind_; }
    int64_t n() const { return n_; }
    bool bluestein() const { return blue_ != nullptr; }
    const CfftPlan* mixed() const { return mixed_.get(); }
    const BluesteinPlan* blue() const { return blue_.get(); }

    // y = the transform of x (interleaved complex128, n each); work: work_doubles() doubles.
    void execute(const double* x, double* y, double* work, bool inverse) const {
        if (blue_) {
            blue_->execute(x, y, work, inverse);
        } else {
            mixed_->execute(x, y, work, inverse);
        }
    }
    int64_t work_doubles() const { return blue_ ? blue_->work_doubles() : 2 * n_; }
    int64_t bytes() const { return blue_ ? blue_->bytes() : mixed_->bytes(); }
    // The work of one line in the units of the parallel thresholds (about n log2 n).
    int64_t cost() const { return cost_; }

private:
    Kind kind_;
    int64_t n_;
    int64_t cost_ = 1;
    std::unique_ptr<CfftPlan> mixed_;
    std::unique_ptr<BluesteinPlan> blue_;
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
