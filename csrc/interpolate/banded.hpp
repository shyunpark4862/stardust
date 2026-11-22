// The tridiagonal solve of sd.interpolate's splines, without pivoting (docs/design/interpolate.md §3.3;
// decision IP16): one factorization A = LU of the system's matrix, shared by every column of a batch, then a
// forward and a back substitution per column. The columns of a row are contiguous (the right side is (n, K)
// row-major), so a substitution step runs over a row's columns at once.
//
// The systems that use it (CubicSpline's slopes, cubic.cpp) have nonnegative entries and positive pivots in
// exact arithmetic, so L >= 0 and U >= 0 and |L||U| = |LU| = |A|: the componentwise backward error of the
// solve is a few units of roundoff times |A| itself (the proof, row by row, is docs/design/interpolate.md §3.3).
// factor() reports the first pivot that is not positive (or not finite): in floating point that happens
// only for a not-a-knot end whose two last spacings differ by more than about 2^50, where the matrix of the
// rounded entries is itself singular to working precision.
//
//   l[i] = a[i] / u[i - 1]          i >= 1 (a: the subdiagonal, l overwrites it)
//   u[i] = b[i] - l[i] c[i - 1]     (b: the diagonal, u overwrites it; c: the superdiagonal)
//   w[i] = 1 / u[i]
//   forward  r[i] -= l[i] r[i - 1]
//   back     r[n - 1] *= w[n - 1];  r[i] = (r[i] - c[i] r[i + 1]) w[i]
//
// The factorization's chain needs a division per row; the back substitution multiplies by the reciprocals,
// which the factorization takes off its chain: a long column is bound by the latency of these chains, and a
// division in both was 1.2x the time (one more rounding per row, counted in §3.3's proof). One rounding per
// operation: the build compiles with -ffp-contract=off (CMakeLists.txt).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace sd::interpolate {

// Factors in place: a[1 .. n) becomes l, b[0 .. n) becomes u, and w[0 .. n) gets 1 / u. Returns -1, or the
// first row whose pivot is not a positive finite number.
inline std::int64_t tridiagonal_factor(double* a, double* b, const double* c, double* w, std::int64_t n) {
    if (!(b[0] > 0.0 && std::isfinite(b[0]))) return 0;
    w[0] = 1.0 / b[0];
    for (std::int64_t i = 1; i < n; ++i) {
        a[i] = a[i] / b[i - 1];
        b[i] = b[i] - a[i] * c[i - 1];
        if (!(b[i] > 0.0 && std::isfinite(b[i]))) return i;
        w[i] = 1.0 / b[i];
    }
    return -1;
}

// Solves LU X = R in place for the columns [c0, c1) of R (n, K) row-major (w: the pivots' reciprocals).
inline void tridiagonal_solve(
    const double* l,
    const double* w,
    const double* c,
    std::int64_t n,
    double* R,
    std::int64_t K,
    std::int64_t c0,
    std::int64_t c1
) {
    for (std::int64_t i = 1; i < n; ++i) {
        double* r = R + i * K;
        const double* p = r - K;
        const double li = l[i];
        for (std::int64_t j = c0; j < c1; ++j) r[j] = r[j] - li * p[j];
    }
    {
        double* r = R + (n - 1) * K;
        const double wn = w[n - 1];
        for (std::int64_t j = c0; j < c1; ++j) r[j] = r[j] * wn;
    }
    for (std::int64_t i = n - 2; i >= 0; --i) {
        double* r = R + i * K;
        const double* q = r + K;
        const double ci = c[i], wi = w[i];
        for (std::int64_t j = c0; j < c1; ++j) r[j] = (r[j] - ci * q[j]) * wi;
    }
}

// ---------------------------------------------------------------- general bands (B-splines; milestone IP3)
//
// make_interp_spline's collocation systems (bspline.cpp): N rows with at most p entries left of the diagonal and
// q right of it. The rows of the interpolation conditions form a totally nonnegative matrix (Karlin; de Boor), so
// elimination without pivoting gives L >= 0, U >= 0 and |L||U| = |A| (de Boor and Pinkus 1977); the rows of the
// derivative conditions sit right after their end's interpolation row (decision IP35), where the design document
// (docs/design/interpolate.md §3.4) bounds what they add: nothing for the orders 1 .. r at an end, a factor below
// 2 for k = 3's natural condition. The same operations as the tridiagonal solve above: l = a / u, the row
// updates a - l u, the back substitution multiplied by 1 / u (one rounding per operation).
//
// Partial pivoting (LAPACK gbtf2's scheme: the rows swap within the band, the multipliers are kept per step)
// runs for the other derivative conditions (factor(fill, true); decision IP53) and whenever a pivot without it
// is zero or not finite; pivoted() tells which one ran. Both are fixed sequences of operations per column: a
// column of a batch gets the same bits alone or with others, on any thread.
class BandSolver {
public:
    BandSolver(std::int64_t n, std::int64_t p, std::int64_t q)
        : n_(n), p_(p), q_(q), w_(p + q + 1), a_(static_cast<std::size_t>(n * (p + q + 1)), 0.0) {}

    std::int64_t rows() const { return n_; }
    // entry (i, c) while factor() fills the matrix, |c - i| within the band
    double& at(std::int64_t i, std::int64_t c) { return a_[i * w_ + c - i + p_]; }

    // Fills the matrix with fill(*this) (entries through at()) and factors it; returns -1, or the first step whose
    // pivot is zero or not finite even with partial pivoting. The fallback fills the matrix again (no copy of a
    // long band is kept for the rare case); pivot = true goes to partial pivoting at once.
    template <typename Fill>
    std::int64_t factor(const Fill& fill, bool pivot = false) {
        inv_.assign(n_, 0.0);
        if (!pivot) {
            fill(*this);
            if (factor_plain() < 0) return -1;
        }
        pivoted_ = true;
        w_ = 2 * p_ + q_ + 1;
        a_.assign(static_cast<std::size_t>(n_ * w_), 0.0);
        fill(*this);
        piv_.assign(n_, 0);
        lm_.assign(static_cast<std::size_t>(n_ * std::max<std::int64_t>(p_, 1)), 0.0);
        return factor_pivoted();
    }

    bool pivoted() const { return pivoted_; }

    // Solves A X = R in place for the columns [c0, c1) of R (N, K) row-major.
    void solve(double* R, std::int64_t K, std::int64_t c0, std::int64_t c1) const {
        const std::int64_t up = pivoted_ ? p_ + q_ : q_;
        if (!pivoted_) {
            for (std::int64_t i = 1; i < n_; ++i) {
                double* r = R + i * K;
                const double* row = a_.data() + i * w_ - i + p_;  // row[c]: column c of row i
                for (std::int64_t j = std::max<std::int64_t>(0, i - p_); j < i; ++j) {
                    const double l = row[j];
                    if (l == 0.0) continue;
                    const double* pj = R + j * K;
                    for (std::int64_t col = c0; col < c1; ++col) r[col] = r[col] - l * pj[col];
                }
            }
        } else {
            for (std::int64_t j = 0; j < n_; ++j) {
                double* rj = R + j * K;
                if (piv_[j] != j) {
                    double* rb = R + piv_[j] * K;
                    for (std::int64_t col = c0; col < c1; ++col) std::swap(rj[col], rb[col]);
                }
                const std::int64_t iend = std::min(n_ - 1, j + p_);
                for (std::int64_t i = j + 1; i <= iend; ++i) {
                    const double l = lm_[j * p_ + (i - j - 1)];
                    if (l == 0.0) continue;
                    double* ri = R + i * K;
                    for (std::int64_t col = c0; col < c1; ++col) ri[col] = ri[col] - l * rj[col];
                }
            }
        }
        for (std::int64_t i = n_ - 1; i >= 0; --i) {
            double* r = R + i * K;
            const double* row = a_.data() + i * w_ - i + p_;
            const std::int64_t cend = std::min(n_ - 1, i + up);
            for (std::int64_t c = i + 1; c <= cend; ++c) {
                const double u = row[c];
                if (u == 0.0) continue;
                const double* pc = R + c * K;
                for (std::int64_t col = c0; col < c1; ++col) r[col] = r[col] - u * pc[col];
            }
            const double wi = inv_[i];
            for (std::int64_t col = c0; col < c1; ++col) r[col] = r[col] * wi;
        }
    }

private:
    static bool usable(double u) { return u != 0.0 && std::isfinite(u); }

    std::int64_t factor_plain() {
        for (std::int64_t j = 0; j < n_; ++j) {
            const double* rj = a_.data() + j * w_ - j + p_;
            const double u = rj[j];
            if (!usable(u)) return j;
            inv_[j] = 1.0 / u;
            const std::int64_t iend = std::min(n_ - 1, j + p_), cend = std::min(n_ - 1, j + q_);
            for (std::int64_t i = j + 1; i <= iend; ++i) {
                double* ri = a_.data() + i * w_ - i + p_;
                if (ri[j] == 0.0) continue;
                const double l = ri[j] / u;
                ri[j] = l;
                for (std::int64_t c = j + 1; c <= cend; ++c) ri[c] = ri[c] - l * rj[c];
            }
        }
        return -1;
    }

    std::int64_t factor_pivoted() {
        const std::int64_t up = p_ + q_;
        for (std::int64_t j = 0; j < n_; ++j) {
            const std::int64_t iend = std::min(n_ - 1, j + p_), cend = std::min(n_ - 1, j + up);
            std::int64_t best = j;
            double bv = std::fabs(a_[j * w_ + p_]);
            for (std::int64_t i = j + 1; i <= iend; ++i) {
                const double v = std::fabs(a_[i * w_ + j - i + p_]);
                if (v > bv) best = i, bv = v;
            }
            piv_[j] = best;
            if (!usable(bv)) return j;
            if (best != j)
                for (std::int64_t c = j; c <= cend; ++c)
                    std::swap(a_[j * w_ + c - j + p_], a_[best * w_ + c - best + p_]);
            const double* rj = a_.data() + j * w_ - j + p_;
            const double u = rj[j];
            inv_[j] = 1.0 / u;
            for (std::int64_t i = j + 1; i <= iend; ++i) {
                double* ri = a_.data() + i * w_ - i + p_;
                const double l = ri[j] / u;
                lm_[j * p_ + (i - j - 1)] = l;
                ri[j] = 0.0;
                if (l == 0.0) continue;
                for (std::int64_t c = j + 1; c <= cend; ++c) ri[c] = ri[c] - l * rj[c];
            }
        }
        return -1;
    }

    std::int64_t n_, p_, q_, w_;
    std::vector<double> a_, inv_, lm_;
    std::vector<std::int64_t> piv_;
    bool pivoted_ = false;
};

// A small dense system M X = R (m <= a few dozen: the periodic splines' correction and their n <= k case),
// by Gaussian elimination with partial pivoting (first largest on ties). M (m, m) and R (m, K) row-major, both
// overwritten; returns false when a pivot is zero or not finite.
inline bool dense_solve(double* M, std::int64_t m, double* R, std::int64_t K) {
    for (std::int64_t j = 0; j < m; ++j) {
        std::int64_t best = j;
        double bv = std::fabs(M[j * m + j]);
        for (std::int64_t i = j + 1; i < m; ++i)
            if (std::fabs(M[i * m + j]) > bv) best = i, bv = std::fabs(M[i * m + j]);
        if (!(bv != 0.0 && std::isfinite(bv))) return false;
        if (best != j) {
            for (std::int64_t c = 0; c < m; ++c) std::swap(M[j * m + c], M[best * m + c]);
            for (std::int64_t c = 0; c < K; ++c) std::swap(R[j * K + c], R[best * K + c]);
        }
        const double u = M[j * m + j];
        for (std::int64_t i = j + 1; i < m; ++i) {
            const double l = M[i * m + j] / u;
            if (l == 0.0) continue;
            for (std::int64_t c = j + 1; c < m; ++c) M[i * m + c] = M[i * m + c] - l * M[j * m + c];
            for (std::int64_t c = 0; c < K; ++c) R[i * K + c] = R[i * K + c] - l * R[j * K + c];
        }
    }
    for (std::int64_t i = m - 1; i >= 0; --i) {
        for (std::int64_t c = i + 1; c < m; ++c) {
            const double u = M[i * m + c];
            for (std::int64_t col = 0; col < K; ++col) R[i * K + col] = R[i * K + col] - u * R[c * K + col];
        }
        const double w = 1.0 / M[i * m + i];
        for (std::int64_t col = 0; col < K; ++col) R[i * K + col] = R[i * K + col] * w;
    }
    return true;
}

}  // namespace sd::interpolate
