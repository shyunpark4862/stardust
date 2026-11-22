// vDSP's double-precision DFT (Accelerate; docs/design/fft.md §3.1, §3.2): which lengths it takes,
// what a setup costs in memory, and a forward/inverse setup pair with the conversions from and to
// sd's interleaved complex128.
//
// Facts from the SDK header vDSP.h and the probes of milestone FT0 (tools/fft_probe.py):
//   zop  (complex, split re/im arrays): n = 2^k, or f 2^k with f in {3, 5, 15} and k >= 3
//   zrop (real <-> complex, n the real length): n = 2^k (k >= 1), or f 2^k with f in {3, 5, 15}
//        and k >= 4; the forward result is 2 X_k, with X_0 and X_(n/2) packed as Or[0] and Oi[0]
//   Interleaved complex: 2^k for 8 <= 2^k <= 4096 and f 2^k (f in {3, 5, 9, 15}, k >= 2) up to
//        4096 (the header promises more)
// A setup is read-only while it executes, so one setup (or a group sharing memory through
// `previous`) may run on any number of threads at once; a group must not be created or destroyed
// while one of its members runs. The inverse setup is made with the forward one as `previous`
// (it then costs almost nothing) and is never shared with another plan.
//
// Setup memory (milestone FT0, tools/fft_probe.py section vdsp; malloc's bytes for n >= 2^14): per
// point zop 14-15 bytes for 2^k and 24-35 for f 2^k, zrop 11-12 and 16-21; the inverse setup made
// with `previous` adds nothing. Decision FT30 leaves vDSP out of the first version's paths (the own
// path was faster for every class); should a class go back to vDSP, the plan cache (FT24: 256 MB in
// all) counts these setups by vdsp_setup_bytes and refuses one above its limit before creating it.
#pragma once

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <cstdint>

namespace sd::fft {

enum class VdspKind { c2c, r2c };

// The odd part f and the power k of n = f 2^k (n >= 1).
inline void odd_part(int64_t n, int64_t& f, int& k) {
    f = n, k = 0;
    while (f % 2 == 0) f /= 2, ++k;
}

inline bool vdsp_accepts(VdspKind kind, int64_t n) {
    if (n < 1 || n > (int64_t{1} << 31)) return false;
    int64_t f;
    int k;
    odd_part(n, f, k);
    if (kind == VdspKind::c2c) return f == 1 || ((f == 3 || f == 5 || f == 15) && k >= 3);
    return (f == 1 && k >= 1) || ((f == 3 || f == 5 || f == 15) && k >= 4);
}

// An upper estimate of the bytes of a forward + inverse setup pair (FT0's measurement with a
// margin: the largest per-point cost seen over every class, 35 bytes, times about 1.4, plus 64 kB).
inline constexpr double kSetupBytesPerPoint = 48.0;
inline int64_t vdsp_setup_bytes(int64_t n) {
    return static_cast<int64_t>(kSetupBytesPerPoint * static_cast<double>(n)) + (int64_t{1} << 16);
}

// A forward and inverse setup of one (kind, length), destroyed together.
class VdspPair {
public:
    VdspPair(VdspKind kind, int64_t n) : kind_(kind), n_(n) {
        const auto len = static_cast<vDSP_Length>(n);
        if (kind == VdspKind::c2c) {
            fwd_ = vDSP_DFT_zop_CreateSetupD(nullptr, len, vDSP_DFT_FORWARD);
            if (fwd_) inv_ = vDSP_DFT_zop_CreateSetupD(fwd_, len, vDSP_DFT_INVERSE);
        } else {
            fwd_ = vDSP_DFT_zrop_CreateSetupD(nullptr, len, vDSP_DFT_FORWARD);
            if (fwd_) inv_ = vDSP_DFT_zrop_CreateSetupD(fwd_, len, vDSP_DFT_INVERSE);
        }
    }
    ~VdspPair() {
        if (inv_) vDSP_DFT_DestroySetupD(inv_);
        if (fwd_) vDSP_DFT_DestroySetupD(fwd_);
    }
    VdspPair(const VdspPair&) = delete;
    VdspPair& operator=(const VdspPair&) = delete;

    bool ok() const { return fwd_ != nullptr && inv_ != nullptr; }
    int64_t n() const { return n_; }
    VdspKind kind() const { return kind_; }
    vDSP_DFT_SetupD forward_setup() const { return fwd_; }
    vDSP_DFT_SetupD inverse_setup() const { return inv_; }

    // c2c: y = DFT(x), x and y interleaved complex128 (n each); work: 2n doubles (the split copy).
    void c2c_forward(const double* x, double* y, double* work) const {
        const auto n = static_cast<vDSP_Length>(n_);
        DSPDoubleSplitComplex s{work, work + n_};
        vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(x), 2, &s, 1, n);
        vDSP_DFT_ExecuteD(fwd_, s.realp, s.imagp, s.realp, s.imagp);
        vDSP_ztocD(&s, 1, reinterpret_cast<DSPDoubleComplex*>(y), 2, n);
    }

    // r2c: y (n/2 + 1 interleaved complex128) = the half spectrum of the n reals x; work: n doubles.
    void r2c_forward(const double* x, double* y, double* work) const {
        const int64_t h = n_ / 2;
        DSPDoubleSplitComplex s{work, work + h};
        vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(x), 2, &s, 1, static_cast<vDSP_Length>(h));
        vDSP_DFT_ExecuteD(fwd_, s.realp, s.imagp, s.realp, s.imagp);
        const float64x2_t half = vdupq_n_f64(0.5);
        const double nyquist = s.imagp[0];
        int64_t k = 0;
        for (; k + 2 <= h; k += 2) {
            const float64x2_t r = vmulq_f64(vld1q_f64(s.realp + k), half);
            const float64x2_t i = vmulq_f64(vld1q_f64(s.imagp + k), half);
            vst2q_f64(y + 2 * k, (float64x2x2_t{{r, i}}));
        }
        for (; k < h; ++k) y[2 * k] = 0.5 * s.realp[k], y[2 * k + 1] = 0.5 * s.imagp[k];
        y[1] = 0.0;
        y[2 * h] = 0.5 * nyquist, y[2 * h + 1] = 0.0;
    }

private:
    VdspKind kind_;
    int64_t n_;
    vDSP_DFT_SetupD fwd_ = nullptr;
    vDSP_DFT_SetupD inv_ = nullptr;
};

// vDSP's interleaved complex DFT (n <= 4096; see above), forward only.
class VdspInterleaved {
public:
    explicit VdspInterleaved(int64_t n) : n_(n) {
        fwd_ = vDSP_DFT_Interleaved_CreateSetupD(
            nullptr, static_cast<vDSP_Length>(n), vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_ComplextoComplex
        );
    }
    ~VdspInterleaved() {
        if (fwd_) vDSP_DFT_Interleaved_DestroySetupD(fwd_);
    }
    VdspInterleaved(const VdspInterleaved&) = delete;
    VdspInterleaved& operator=(const VdspInterleaved&) = delete;

    bool ok() const { return fwd_ != nullptr; }
    void forward(const double* x, double* y) const {
        vDSP_DFT_Interleaved_ExecuteD(
            fwd_, reinterpret_cast<const DSPDoubleComplex*>(x), reinterpret_cast<DSPDoubleComplex*>(y)
        );
    }

private:
    int64_t n_;
    vDSP_DFT_Interleaved_SetupD fwd_ = nullptr;
};

}  // namespace sd::fft
