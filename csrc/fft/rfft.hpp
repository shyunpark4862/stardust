// Prototype of the library's own real-to-complex FFT for even n (docs/design/fft.md §3.4): the n
// reals read as h = n / 2 interleaved complex numbers z_j = x_2j + i x_2j+1 (no copy: that is
// their memory layout), one complex FFT of length h (csrc/fft/cfft.hpp), and one pass of twiddles:
//     E_k = (Z_k + conj Z_(h-k)) / 2,  O_k = -i (Z_k - conj Z_(h-k)) / 2,
//     X_k = E_k + w_n^k O_k,  X_(h-k) = conj(E_k - w_n^k O_k),
// X_0 = Re Z_0 + Im Z_0 and X_h = Re Z_0 - Im Z_0 with imaginary parts exactly +0. The output is the
// half spectrum X_0 .. X_h, h + 1 interleaved complex128. Milestone FT0 timed it against vDSP's
// zrop (decision FT22); FT2 builds the real transforms on it.
#pragma once

#include <arm_neon.h>

#include <cstdint>
#include <vector>

#include "fft/cfft.hpp"
#include "fft/twiddle.hpp"

namespace sd::fft {

class RfftPlan {
public:
    explicit RfftPlan(int64_t n) : n_(n), h_(n / 2), inner_(n / 2) {
        w_.resize(static_cast<size_t>(2 * (h_ / 2 + 1)));
        for (int64_t k = 0; k <= h_ / 2; ++k) {
            const Twiddle w = twiddle(k, n);
            w_[static_cast<size_t>(2 * k)] = w.re, w_[static_cast<size_t>(2 * k + 1)] = w.im;
        }
    }

    int64_t n() const { return n_; }
    const CfftPlan& inner() const { return inner_; }

    // y (h + 1 interleaved complex) = the half spectrum of the n reals x; work: n doubles.
    void forward(const double* x, double* y, double* work) const {
        inner_.forward(x, y, work);
        const int64_t h = h_;
        const double z0r = y[0], z0i = y[1];
        y[0] = z0r + z0i, y[1] = 0.0;
        y[2 * h] = z0r - z0i, y[2 * h + 1] = 0.0;
        const float64x2_t half = vdupq_n_f64(0.5);
        const float64x2_t conj = {1.0, -1.0};
        for (int64_t k = 1; 2 * k <= h; ++k) {
            const float64x2_t zk = vld1q_f64(y + 2 * k);
            const float64x2_t zm = vmulq_f64(vld1q_f64(y + 2 * (h - k)), conj);  // conj Z_(h-k)
            const float64x2_t e = vmulq_f64(vaddq_f64(zk, zm), half);
            const float64x2_t d = vmulq_f64(vsubq_f64(zk, zm), half);
            const float64x2_t o = vmulq_f64(vextq_f64(d, d, 1), conj);  // -i d = (d.im, -d.re)
            const double wr = w_[static_cast<size_t>(2 * k)], wi = w_[static_cast<size_t>(2 * k + 1)];
            const float64x2_t wneg = {-wi, wi};
            // w o = (wr o.re - wi o.im, wr o.im + wi o.re)
            const float64x2_t p = vfmaq_f64(vmulq_n_f64(o, wr), vextq_f64(o, o, 1), wneg);
            vst1q_f64(y + 2 * k, vaddq_f64(e, p));
            if (2 * k != h) vst1q_f64(y + 2 * (h - k), vmulq_f64(vsubq_f64(e, p), conj));
        }
    }

    int64_t bytes() const { return inner_.bytes() + static_cast<int64_t>(w_.size() * sizeof(double)); }

private:
    int64_t n_, h_;
    CfftPlan inner_;
    std::vector<double> w_;  // w_n^k, k = 0 .. h / 2, interleaved
};

}  // namespace sd::fft
