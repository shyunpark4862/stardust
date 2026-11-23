// The banded least squares of sd.smooth's splines (docs/design/smooth.md §3.3; decisions SM30, SM50-SM53): Givens
// QR of a stacked matrix whose rows each cover a few consecutive columns, the back substitution, the band of the
// inverse of R'R (the selected inverse) and the solves a hat row needs. make_lsq_spline, the smoothing spline, the
// P-spline and FITPACK's smoothing (smooth/spline.cpp, smooth/fitpack.cpp) all reduce to it.
//
//   rotation   FITPACK's fpgivs and fprota: dd = |piv| sqrt(1 + (ww / piv)^2) (or the same with the roles swapped),
//              cos = ww / dd, sin = piv / dd, ww <- dd; (a, b) <- (cos a - sin b, cos b + sin a). No square of an
//              entry is formed, so no rotation overflows or underflows before its result does
//   band       R is nb x bw: R[j][i] the entry (j, j + i). A row enters at its first column o and is rotated
//              against R's rows o, o + 1, ... (a zero pivot is skipped). Rows come in nondecreasing o and no row
//              is wider than bw, so the fill of a row stays within its first bw columns and each row costs at
//              most bw^2 / 2 rotations of pairs (the invariant: R's row j has no entry past column o + bw - 1)
//   border     kb dense columns after the band (the periodic splines' wrapped columns): every row of R has them
//              (B[j][0 .. kb)), and the last kb rows form a dense upper triangle T after the band's nb rows; a
//              row's border entries are rotated with the band and then into T
//   solve      the back substitution, T first, then the band rows from the last; a zero or non-finite diagonal
//              is reported (the caller decides: LinAlgError for an unpenalized fit, never for a penalized one)
//   inverse    Sigma = (R'R)^-1 within the band, from R Sigma = R^-T: Sigma_ij = -(1 / R_ii) sum_{l > i} R_il
//              Sigma_lj for j > i and Sigma_ii = (1 / R_ii)(1 / R_ii - sum_{l > i} R_il Sigma_il), i from the last
//              row down; O(nb bw^2) (Hutchinson and de Hoog 1985). Only without a border
//
// One rounding per operation: the build compiles with -ffp-contract=off (CMakeLists.txt).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::smooth {

// FITPACK's fpgivs: the rotation taking (piv, ww) to (0, dd), ww >= 0 the current diagonal; ww becomes dd.
inline void givens(double piv, double& ww, double& c, double& s) {
    const double store = std::abs(piv);
    double dd;
    if (store >= ww) {
        const double r = ww / piv;
        dd = store * std::sqrt(1.0 + r * r);
    } else {
        const double r = piv / ww;
        dd = ww * std::sqrt(1.0 + r * r);
    }
    c = ww / dd;
    s = piv / dd;
    ww = dd;
}

// FITPACK's fprota: a is the incoming row's entry, b the triangle's.
inline void rotate(double c, double s, double& a, double& b) {
    const double s1 = a, s2 = b;
    b = c * s2 + s * s1;
    a = c * s1 - s * s2;
}

class BandQR {
public:
    std::int64_t nb = 0, bw = 0, kb = 0, ncol = 0;
    std::vector<double> R;  // nb x bw
    std::vector<double> B;  // (nb + kb) x kb: the border columns of the band rows, then the triangle T
    std::vector<double> Z;  // (nb + kb) x ncol: the rotated right sides
    std::vector<double> rest;  // ncol: the sum of squares of what the rotations moved off the triangle

    void reset(std::int64_t nb_, std::int64_t bw_, std::int64_t kb_, std::int64_t ncol_) {
        nb = nb_;
        bw = bw_;
        kb = kb_;
        ncol = ncol_;
        R.assign(static_cast<std::size_t>(nb * bw), 0.0);
        B.assign(static_cast<std::size_t>((nb + kb) * kb), 0.0);
        Z.assign(static_cast<std::size_t>((nb + kb) * ncol), 0.0);
        rest.assign(static_cast<std::size_t>(ncol), 0.0);
    }

    // Rotates a row into the triangle. h: bw doubles, the entries of the columns o .. o + bw - 1 (zeros past the
    // row's width; columns >= nb must be zero), hb: kb doubles (the border), y: ncol doubles. All three are used
    // as workspace.
    void add(std::int64_t o, double* h, double* hb, double* y) {
        const std::int64_t last = std::min(nb, o + bw);  // the row's entries end before column o + bw
        for (std::int64_t j = o; j < last; ++j) {
            const std::int64_t w = o + bw - j;  // the window's entries that can be nonzero: columns j .. o + bw - 1
            const double piv = h[0];
            if (piv != 0.0) {
                double* r = R.data() + j * bw;
                double c, s;
                givens(piv, r[0], c, s);
                double* z = Z.data() + j * ncol;
                for (std::int64_t q = 0; q < ncol; ++q) rotate(c, s, y[q], z[q]);
                for (std::int64_t i = 1; i < w; ++i) rotate(c, s, h[i], r[i]);
                double* b = B.data() + j * kb;
                for (std::int64_t i = 0; i < kb; ++i) rotate(c, s, hb[i], b[i]);
            }
            for (std::int64_t i = 1; i < w; ++i) h[i - 1] = h[i];
            h[w - 1] = 0.0;
        }
        for (std::int64_t i = 0; i < kb; ++i) {
            const double piv = hb[i];
            if (piv == 0.0) continue;
            double* t = B.data() + (nb + i) * kb;
            double c, s;
            givens(piv, t[i], c, s);
            double* z = Z.data() + (nb + i) * ncol;
            for (std::int64_t q = 0; q < ncol; ++q) rotate(c, s, y[q], z[q]);
            for (std::int64_t i1 = i + 1; i1 < kb; ++i1) rotate(c, s, hb[i1], t[i1]);
        }
        for (std::int64_t q = 0; q < ncol; ++q) rest[q] += y[q] * y[q];
    }

    // The diagonal entry of row j.
    double diag(std::int64_t j) const { return j < nb ? R[j * bw] : B[j * kb + (j - nb)]; }

    // -1 when every diagonal entry is positive and finite, else the first row that is not.
    std::int64_t first_bad_diagonal() const {
        for (std::int64_t j = 0; j < nb + kb; ++j) {
            const double d = diag(j);
            if (!(d > 0.0 && std::isfinite(d))) return j;
        }
        return -1;
    }

    // The largest |diagonal| over the smallest (inf when one is 0).
    double diagonal_ratio() const {
        double lo = INFINITY, hi = 0.0;
        for (std::int64_t j = 0; j < nb + kb; ++j) {
            const double d = std::abs(diag(j));
            lo = std::min(lo, d);
            hi = std::max(hi, d);
        }
        return lo > 0.0 ? hi / lo : INFINITY;
    }

    // c ((nb + kb) x ncol) from R c = Z.
    void solve(double* c) const {
        for (std::int64_t i = kb - 1; i >= 0; --i) {
            const double* t = B.data() + (nb + i) * kb;
            for (std::int64_t q = 0; q < ncol; ++q) {
                double v = Z[(nb + i) * ncol + q];
                for (std::int64_t i1 = i + 1; i1 < kb; ++i1) v -= t[i1] * c[(nb + i1) * ncol + q];
                c[(nb + i) * ncol + q] = v / t[i];
            }
        }
        for (std::int64_t j = nb - 1; j >= 0; --j) {
            const double* r = R.data() + j * bw;
            const double* b = B.data() + j * kb;
            const std::int64_t w = std::min(bw, nb - j);
            for (std::int64_t q = 0; q < ncol; ++q) {
                double v = Z[j * ncol + q];
                for (std::int64_t i = 1; i < w; ++i) v -= r[i] * c[(j + i) * ncol + q];
                for (std::int64_t i = 0; i < kb; ++i) v -= b[i] * c[(nb + i) * ncol + q];
                c[j * ncol + q] = v / r[0];
            }
        }
    }
};

// The band of Sigma = (R'R)^-1 of a band triangle R (nb x bw, no border): S[i][d] = Sigma_{i, i + d}, d < bw.
inline void selected_inverse(const double* R, std::int64_t nb, std::int64_t bw, double* S) {
    for (std::int64_t i = nb - 1; i >= 0; --i) {
        const double* r = R + i * bw;
        const double inv = 1.0 / r[0];
        const std::int64_t w = std::min(bw, nb - i);
        double* s = S + i * bw;
        for (std::int64_t d = w - 1; d >= 1; --d) {  // Sigma_{i, i+d}: sum_{l > i} R_il Sigma_{l, i+d}
            const std::int64_t j = i + d;
            double acc = 0.0;
            for (std::int64_t e = 1; e < w; ++e) {
                const std::int64_t l = i + e;
                const double sl = l <= j ? S[l * bw + (j - l)] : S[j * bw + (l - j)];
                acc += r[e] * sl;
            }
            s[d] = -acc * inv;
        }
        for (std::int64_t d = w; d < bw; ++d) s[d] = 0.0;
        double acc = 0.0;
        for (std::int64_t e = 1; e < w; ++e) acc += r[e] * s[e];
        s[0] = (inv - acc) * inv;
    }
}

// u = R^-T b in place (b: nb doubles, zero before index o), then v = R^-1 u in place: v = (R'R)^-1 b.
inline void normal_solve(const double* R, std::int64_t nb, std::int64_t bw, std::int64_t o, double* v) {
    for (std::int64_t j = o; j < nb; ++j) {
        double acc = v[j];
        const std::int64_t lo = std::max(o, j - bw + 1);
        for (std::int64_t i = lo; i < j; ++i) acc -= R[i * bw + (j - i)] * v[i];
        v[j] = acc / R[j * bw];
    }
    for (std::int64_t j = nb - 1; j >= 0; --j) {
        double acc = v[j];
        const std::int64_t w = std::min(bw, nb - j);
        for (std::int64_t i = 1; i < w; ++i) acc -= R[j * bw + i] * v[j + i];
        v[j] = acc / R[j * bw];
    }
}

// An estimate of the 1-norm condition number ||R||_1 ||R^-1||_1 of a band triangle (no border; Hager's method as
// LAPACK's dlacn2 runs it, five steps at most, then Higham's alternating vector). Within a factor nb of the 2-norm
// condition of the stacked matrix the triangle came from. inf when a diagonal entry is 0.
inline double condition_estimate(const double* R, std::int64_t nb, std::int64_t bw) {
    if (nb == 0) return 1.0;
    double anorm = 0.0;  // the largest column sum
    for (std::int64_t j = 0; j < nb; ++j) {
        double s = 0.0;
        for (std::int64_t i = std::max<std::int64_t>(0, j - bw + 1); i <= j; ++i) s += std::abs(R[i * bw + (j - i)]);
        anorm = std::max(anorm, s);
    }
    for (std::int64_t j = 0; j < nb; ++j)
        if (!(R[j * bw] != 0.0 && std::isfinite(R[j * bw]))) return INFINITY;
    std::vector<double> x(static_cast<std::size_t>(nb), 1.0 / static_cast<double>(nb)), z(x.size());
    const auto solve_r = [&](std::vector<double>& v) {  // v <- R^-1 v
        for (std::int64_t j = nb - 1; j >= 0; --j) {
            double acc = v[j];
            const std::int64_t w = std::min(bw, nb - j);
            for (std::int64_t i = 1; i < w; ++i) acc -= R[j * bw + i] * v[j + i];
            v[j] = acc / R[j * bw];
        }
    };
    const auto solve_rt = [&](std::vector<double>& v) {  // v <- R^-T v
        for (std::int64_t j = 0; j < nb; ++j) {
            double acc = v[j];
            for (std::int64_t i = std::max<std::int64_t>(0, j - bw + 1); i < j; ++i) acc -= R[i * bw + (j - i)] * v[i];
            v[j] = acc / R[j * bw];
        }
    };
    double est = 0.0;
    std::int64_t last = -1;
    for (int it = 0; it < 5; ++it) {
        solve_r(x);
        double nrm = 0.0;
        for (double v : x) nrm += std::abs(v);
        if (!std::isfinite(nrm)) return INFINITY;
        if (it > 0 && nrm <= est) break;
        est = nrm;
        for (std::int64_t j = 0; j < nb; ++j) z[j] = x[j] >= 0 ? 1.0 : -1.0;
        solve_rt(z);
        std::int64_t jmax = 0;
        for (std::int64_t j = 1; j < nb; ++j)
            if (std::abs(z[j]) > std::abs(z[jmax])) jmax = j;
        if (jmax == last) break;
        last = jmax;
        std::fill(x.begin(), x.end(), 0.0);
        x[jmax] = 1.0;
    }
    for (std::int64_t j = 0; j < nb; ++j)  // Higham's alternating vector
        x[j] = (j % 2 ? -1.0 : 1.0) *
               (1.0 + static_cast<double>(j) / static_cast<double>(std::max<std::int64_t>(nb - 1, 1)));
    solve_r(x);
    double alt = 0.0;
    for (double v : x) alt += std::abs(v);
    est = std::max(est, 2.0 * alt / (3.0 * static_cast<double>(nb)));
    return anorm * est;
}

}  // namespace sd::smooth
