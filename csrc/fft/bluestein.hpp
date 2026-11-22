// Bluestein's chirp-z transform for lengths with a prime factor above 31 (docs/design/fft.md §3.3,
// decisions FT36, FT64): with w_j = e^(-pi i j^2 / n) and jk = (j^2 + k^2 - (k - j)^2) / 2,
//     y_k = w_k sum_j (x_j w_j) conj(w_(k-j)),
// a cyclic convolution of length m >= 2n - 1 (decision FT37: m is the smallest 2^a 3^b 5^c at least
// 2n - 1) done with the mixed-radix transform of length m: one forward transform, one product with the
// kernel's transform beta = DFT_m(b) / m (b_l = conj(w_l) for |l| < n, zero elsewhere; computed when
// the plan is made), one inverse transform (unnormalised; the 1/m is in beta).
//
// Accuracy (docs/design/fft.md §4.2). The chirp exponent j^2 mod 2n is exact integer arithmetic, and
// the chirp and beta are computed in double-double (fft/ddmath.hpp) and rounded once to doubles: the
// stored chirp is within u of the true one per entry and beta within u |beta_k| plus far less than u
// max |beta|. (With a double-precision transform of the kernel, beta's error could only be bounded by
// u log m sqrt(m) max |beta|: the kernel has 2n - 1 entries of modulus 1, and an entry of its transform
// is a cancelling sum of them.) Following the five rounded steps through the exact factorisation
// y = P D_w F* D_beta F E D_w x (unnormalised F, F* F = m I), with rho = m max |beta_k| / sqrt(n):
//     ||y_hat - y|| <= (rho (2 kappa log2 m + 6.5) + 3.5) u ||y||  <=  rho (kappa (2 L(m) + 2) + c0) u ||y||,
// the contract's form for these lengths (L_B(n) = 2 L(m) + 2, the bound times rho; rho >= 1 since
// ||y|| <= ||D_beta|| m ||x||).
#pragma once

#include <cstdint>
#include <vector>

#include "fft/cfft.hpp"

namespace sd::fft {

// The smallest 2^a 3^b 5^c >= target (target >= 1, below 2^62).
int64_t smooth5_at_least(int64_t target);

class BluesteinPlan {
public:
    explicit BluesteinPlan(int64_t n);

    int64_t n() const { return n_; }
    int64_t m() const { return m_; }
    double rho() const { return rho_; }
    const CfftPlan& inner() const { return inner_; }

    // y = DFT(x) (inverse: the sign +, no 1/n); x, y interleaved complex128 (n each), work: 6m doubles,
    // none of them overlapping.
    void execute(const double* x, double* y, double* work, bool inverse) const;

    int64_t work_doubles() const { return 6 * m_; }
    int64_t bytes() const {
        return static_cast<int64_t>(sizeof(*this) + (chirp_.size() + beta_.size()) * sizeof(double)) + inner_.bytes();
    }

private:
    template <bool Swap>
    void run(const double* x, double* y, double* work) const;

    int64_t n_, m_;
    double rho_ = 1.0;
    CfftPlan inner_;
    std::vector<double> chirp_;  // w_j, j < n, interleaved
    std::vector<double> beta_;   // beta_k, k < m, interleaved
};

}  // namespace sd::fft
