// The real path (docs/design/fft.md §3.4, milestone FT2): the real-to-complex transform of n reals and
// its inverse, the complex-to-real transform of a Hermitian half spectrum, for even n.
//
// Forward. The n reals read as h = n / 2 interleaved complex numbers z_j = x_2j + i x_2j+1 (no copy: that
// is their memory layout), one complex FFT of length h (Z = DFT_h z; the plan of fft/plan.hpp, mixed radix
// or Bluestein), and one pass of twiddles w = w_n^k = e^(-2 pi i k / n):
//     e = (Z_k + conj Z_(h-k)) / 2,  o = -i (Z_k - conj Z_(h-k)) / 2,   (the DFTs of x_even, x_odd at k)
//     X_k = e + w o,  X_(h-k) = conj(e - w o),
// X_0 = Re Z_0 + Im Z_0 and X_h = Re Z_0 - Im Z_0 with imaginary parts exactly +0. The output is the
// half spectrum X_0 .. X_h, h + 1 interleaved complex128.
//
// Inverse (unnormalised: n x_j = sum_k X_k e^(+2 pi i jk / n) over the Hermitian extension of X_0 .. X_h).
// With E_k, O_k as above, X_k = E_k + w O_k and X_(k+h) = conj X_(h-k) = E_k - w O_k, so
//     a = X_k + conj X_(h-k) = 2 E_k,  b = X_k - conj X_(h-k) = 2 w O_k,  o' = conj(w) b = 2 O_k,
//     Z'_k = a + i o' = 2 Z_k,  Z'_(h-k) = conj(a - i o'),
// Z'_0 = (X_0 + X_h) + i (X_0 - X_h), and the unnormalised inverse complex FFT of length h of Z' is
// 2 h z = n z: the n real results in their memory layout. The imaginary parts of X_0 and X_h are never
// read (they have no partner in the Hermitian extension; numpy and scipy drop them too).
//
// Arithmetic. A complex product is the FMA product of fft/cfft.hpp (one product and one fused
// multiply-add per part); the halving and -i, i, conj are exact (the halving unless subnormal). The
// position h / 2 (even h) pairs with itself and is written once. Everything depends on n alone.

//
// Error (docs/design/fft.md §4.2). The map Z -> (X_0 .. X_(n-1)) is sqrt(2) times a unitary map, so the
// inner transform's error eps_h ||Z|| becomes eps_h sqrt(n) ||x|| on the full spectrum (||Z|| = sqrt(h)
// ||x||), with eps_h = kappa L(h) u = kappa (L(n) - 1) u for a mixed-radix h: one kappa u of the
// contract's kappa L(n) u is left for the twiddle pass. Per pair, |X_k_hat - X_k| <= 2u |e| + 5.5u |d| (the
// sum's rounding u (|e| + |d|), e's u |e|, d's u |d| through the product, the product 2u and the twiddle's
// mu = 1.5u), and sum_k (|e_k|^2 + |d_k|^2) = ||Z||^2 / 2 over the pairs, so the pass adds at most
// sqrt(2 (2^2 + 5.5^2)) u ||Z|| = 5.85u sqrt(h) ||x|| to the full spectrum: below kappa u sqrt(n) ||x||. The
// inverse is the same pass in reverse (the same products and sums), so the same count holds.
//
// Milestone FT0 wrote the forward pass as a prototype timed against vDSP's zrop (decision FT22;
// RfftPlan below, used by tools/fft_probe_kernels.cpp); FT2 moved the passes into r2c_post and c2r_pre,
// which the real plans of fft/plan.hpp run, and added the inverse.
#pragma once

#include <arm_neon.h>

#include <cstdint>
#include <vector>

#include "fft/cfft.hpp"
#include "fft/twiddle.hpp"

namespace sd::fft {

// w_n^k, k = 0 .. h / 2 (h = n / 2), interleaved: the twiddles of the real pass of an even length n.
inline std::vector<double> real_twiddles(int64_t n) {
    const int64_t h = n / 2;
    std::vector<double> w(static_cast<size_t>(2 * (h / 2 + 1)));
    for (int64_t k = 0; k <= h / 2; ++k) {
        const Twiddle t = twiddle(k, n);
        w[static_cast<size_t>(2 * k)] = t.re, w[static_cast<size_t>(2 * k + 1)] = t.im;
    }
    return w;
}

// The forward pass: y holds Z_0 .. Z_(h-1) (interleaved) and room for h + 1 entries; on return it holds
// X_0 .. X_h. w: real_twiddles(2h).
inline void r2c_post(double* y, int64_t h, const double* w) {
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
        const double wr = w[2 * k], wi = w[2 * k + 1];
        const float64x2_t wneg = {-wi, wi};
        // w o = (wr o.re - wi o.im, wr o.im + wi o.re)
        const float64x2_t p = vfmaq_f64(vmulq_n_f64(o, wr), vextq_f64(o, o, 1), wneg);
        vst1q_f64(y + 2 * k, vaddq_f64(e, p));
        if (2 * k != h) vst1q_f64(y + 2 * (h - k), vmulq_f64(vsubq_f64(e, p), conj));
    }
}

// The inverse pass: z (h interleaved complex) = Z' of the half spectrum x (h + 1 interleaved complex; the
// imaginary parts of x_0 and x_h are not read). Conj: the pass of conj(x) (hfft). x and z do not overlap.
// w: real_twiddles(2h).
template <bool Conj>
inline void c2r_pre(const double* x, double* z, int64_t h, const double* w) {
    const double x0 = x[0], xh = x[2 * h];
    z[0] = x0 + xh, z[1] = x0 - xh;
    const float64x2_t conj = {1.0, -1.0};
    const float64x2_t itimes = {-1.0, 1.0};
    for (int64_t k = 1; 2 * k <= h; ++k) {
        float64x2_t xk = vld1q_f64(x + 2 * k);
        float64x2_t xm = vld1q_f64(x + 2 * (h - k));
        if constexpr (Conj) {
            xk = vmulq_f64(xk, conj);
        } else {
            xm = vmulq_f64(xm, conj);  // conj x_(h-k) (with Conj: conj(conj x_(h-k)) = x_(h-k))
        }
        const float64x2_t a = vaddq_f64(xk, xm);
        const float64x2_t b = vsubq_f64(xk, xm);
        const double wr = w[2 * k], wi = w[2 * k + 1];
        const float64x2_t wpos = {wi, -wi};
        // conj(w) b = (wr b.re + wi b.im, wr b.im - wi b.re)
        const float64x2_t o = vfmaq_f64(vmulq_n_f64(b, wr), vextq_f64(b, b, 1), wpos);
        const float64x2_t io = vmulq_f64(vextq_f64(o, o, 1), itimes);  // i o' = (-o.im, o.re)
        vst1q_f64(z + 2 * k, vaddq_f64(a, io));
        if (2 * k != h) vst1q_f64(z + 2 * (h - k), vmulq_f64(vsubq_f64(a, io), conj));
    }
}

// The prototype of milestone FT0 (tools/fft_probe_kernels.cpp): the forward real transform of an even n
// on a mixed-radix inner plan of its own.
class RfftPlan {
public:
    explicit RfftPlan(int64_t n) : n_(n), h_(n / 2), inner_(n / 2), w_(real_twiddles(n)) {}

    int64_t n() const { return n_; }
    const CfftPlan& inner() const { return inner_; }

    // y (h + 1 interleaved complex) = the half spectrum of the n reals x; work: n doubles.
    void forward(const double* x, double* y, double* work) const {
        inner_.forward(x, y, work);
        r2c_post(y, h_, w_.data());
    }

    int64_t bytes() const { return inner_.bytes() + static_cast<int64_t>(w_.size() * sizeof(double)); }

private:
    int64_t n_, h_;
    CfftPlan inner_;
    std::vector<double> w_;  // w_n^k, k = 0 .. h / 2, interleaved
};

}  // namespace sd::fft
