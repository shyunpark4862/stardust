// AAA and the barycentric rationals' poles, residues and zeros (docs/design/interpolate.md §3.6, §6; decisions
// IP20, IP32). scipy 1.18.1's algorithm step by step (scipy.interpolate._bary_rational), on real data in real
// arithmetic (dgesdd, dggev) and on complex data in complex arithmetic (zgesdd, zggev), as scipy:
//
//   samples      the points whose value is not finite are dropped; the points are sorted (complex ones by real
//                then imaginary part) and a repeated point keeps its first sample (numpy's unique)
//   iteration    R starts as the mean of f (numpy's pairwise sum); step m adds the sample with the largest
//                |f - R| (the first of equals, a NaN first: numpy's argmax) as a support point, a column of the
//                Cauchy matrix C = 1 / (z - z_j) and of the Loewner matrix A = (f - f_j) C; the weights are the
//                right singular vector of the smallest singular value of A's rows at the other samples (gesdd,
//                'S'; equal smallest values: their vectors summed and divided by sqrt(count)); once
//                s_max / s_min > 1 / (3 eps) the columns are scaled to unit norm first, for the rest of the run;
//                with fewer rows than columns the weights are the sum of a null-space basis (gesdd 'A', rank
//                tolerance s_max eps max(rows, cols)) over sqrt(its size). R = N / D with N = C (w f), D = C w over
//                the nonzero weights, and f itself where D is not finite (the support points). The run stops when
//                max |f - R| <= rtol max |f| (converged) or after min(max_terms, samples) steps (not converged:
//                no warning, decision IP20). Zero weights are dropped at the end
//   poles        the finite generalized eigenvalues of the arrowhead pair E = [0 w; 1 diag(z_j)], B = diag(0, 1..1)
//                (ggev, decision IP20); zeros: the same with w f in E's first row
//   residues     N(p) / D'(p) with N = sum (w f) / (p - z_j) and D' = -sum w / (p - z_j)^2
//   clean_up     a pole whose |residue| / (its distance to the samples) is below clean_up_tol times the geometric
//                mean of |f| is a Froissart doublet: the support point nearest to each is removed and the weights
//                are refit as the last right singular vector of the Loewner matrix at the other samples (gesdd 'A')
//
// Quotients are numpy's (Smith's algorithm for complex ones) and sums the order the formulas give, so the run is
// scipy's up to the rounding of its BLAS products: on the same data the support points come in the same order
// (tests/interpolate/test_aaa.py). Each LAPACK call is Accelerate's: the same bits for the same thread mode.
//
// Private bindings (the Python layer checks the arguments; z and f of one dtype, float64 or complex128):
//   aaa_fit(z, f, rtol, max_terms) -> (zs, fs, zj, fj, wj, errors, converged)   zs, fs: the samples kept
//   bary_eig(zj, wj, top) -> the finite eigenvalues (complex128): the poles (top = wj) or zeros (top = wj fj)
//   bary_residues(zj, fj, wj, poles) -> complex128
//   aaa_clean_up(zs, fs, zj, fj, wj, tol) -> (doublets, zj, fj, wj)

#include "interpolate/interpolate.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "linalg/errors.hpp"
#include "linalg/lapack.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::interpolate {

namespace nb = nanobind;

namespace {

using linalg::cplx;
using linalg::lint;
namespace lapack = linalg::lapack;

constexpr double kEps = 0x1p-52;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------- numpy's scalar operations

inline double absv(double v) { return std::fabs(v); }
inline double absv(const cplx& v) { return std::hypot(v.real(), v.imag()); }
inline double conjv(double v) { return v; }
inline cplx conjv(const cplx& v) { return {v.real(), -v.imag()}; }
inline bool finitev(double v) { return std::isfinite(v); }
inline bool finitev(const cplx& v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }
inline bool infnanv(double v) { return !std::isfinite(v); }
inline bool infnanv(const cplx& v) { return !finitev(v); }
inline bool zerov(double v) { return v == 0; }
inline bool zerov(const cplx& v) { return v.real() == 0 && v.imag() == 0; }
inline bool samev(double a, double b) { return a == b; }
inline bool samev(const cplx& a, const cplx& b) { return a.real() == b.real() && a.imag() == b.imag(); }
inline bool lessv(double a, double b) { return a < b; }
inline bool lessv(const cplx& a, const cplx& b) {
    return a.real() < b.real() || (a.real() == b.real() && a.imag() < b.imag());
}

// numpy's complex quotient (Smith's algorithm)
inline cplx cdiv(const cplx& a, const cplx& b) {
    const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    const double abr = std::fabs(br), abi = std::fabs(bi);
    if (abr >= abi) {
        if (abr == 0 && abi == 0) return {ar / abr, ai / abi};
        const double rat = bi / br, scl = 1.0 / (br + bi * rat);
        return {(ar + ai * rat) * scl, (ai - ar * rat) * scl};
    }
    const double rat = br / bi, scl = 1.0 / (bi + br * rat);
    return {(ar * rat + ai) * scl, (ai * rat - ar) * scl};
}
inline double divv(double a, double b) { return a / b; }
inline cplx divv(const cplx& a, const cplx& b) { return cdiv(a, b); }
inline double divr(double a, double b) { return a / b; }
inline cplx divr(const cplx& a, double b) { return cdiv(a, cplx(b, 0.0)); }  // complex / float64, as numpy
// numpy's complex product (no fma: the build keeps FMA contraction off)
inline double mulv(double a, double b) { return a * b; }
inline cplx mulv(const cplx& a, const cplx& b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}
inline double addv(double a, double b) { return a + b; }
inline cplx addv(const cplx& a, const cplx& b) { return {a.real() + b.real(), a.imag() + b.imag()}; }
inline double subv(double a, double b) { return a - b; }
inline cplx subv(const cplx& a, const cplx& b) { return {a.real() - b.real(), a.imag() - b.imag()}; }

// numpy's pairwise sum (DOUBLE_pairwise_sum, and CDOUBLE's with four accumulators per part)
double np_sum(const double* a, std::int64_t n) {
    if (n < 8) {
        double r = 0.0;
        for (std::int64_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::int64_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}
cplx np_sum(const cplx* a, std::int64_t c) {
    if (c < 4) {
        double rr = 0.0, ri = 0.0;
        for (std::int64_t i = 0; i < c; ++i) rr += a[i].real(), ri += a[i].imag();
        return {rr, ri};
    }
    if (c <= 64) {
        double r[4], m[4];
        for (int j = 0; j < 4; ++j) r[j] = a[j].real(), m[j] = a[j].imag();
        std::int64_t i = 4;
        for (; i < c - (c % 4); i += 4)
            for (int j = 0; j < 4; ++j) r[j] += a[i + j].real(), m[j] += a[i + j].imag();
        double rr = (r[0] + r[1]) + (r[2] + r[3]), ri = (m[0] + m[1]) + (m[2] + m[3]);
        for (; i < c; ++i) rr += a[i].real(), ri += a[i].imag();
        return {rr, ri};
    }
    std::int64_t n2 = c;  // half of the 2c doubles, cut at a multiple of 8 doubles
    n2 -= n2 % 8;
    const cplx x = np_sum(a, n2 / 2), y = np_sum(a + n2 / 2, c - n2 / 2);
    return {x.real() + y.real(), x.imag() + y.imag()};
}

// ---------------------------------------------------------------- LAPACK

// The SVD of the rows x cols column-major matrix a (destroyed): singular values s and V^H (cols x cols, column
// major). jobz 'S' (rows >= cols) or 'A'. Throws LinAlgError when gesdd does not converge.
template <class T>
void svd(
    char jobz, lint rows, lint cols, std::vector<T>& a, std::vector<double>& s, std::vector<T>& vt, linalg::Scratch& sc
) {
    const lint mn = std::min(rows, cols);
    s.assign(static_cast<std::size_t>(mn), 0.0);
    vt.assign(static_cast<std::size_t>(cols * cols), T{});
    if (rows == 0) {  // scipy's SVD of an empty matrix: no values, V the identity
        for (lint i = 0; i < cols; ++i) vt[i + i * cols] = T{1};
        return;
    }
    const lint ucols = jobz == 'A' ? rows : mn;
    std::vector<T> u(static_cast<std::size_t>(rows * ucols));
    const lint info = lapack::gesdd(jobz, rows, cols, a.data(), rows, s.data(), u.data(), rows, vt.data(), cols, sc);
    if (info != 0) throw linalg::LinAlgError("AAA(): the SVD did not converge");
}

// The finite generalized eigenvalues of the arrowhead pair (support points zj, first row top)
template <class T>
std::vector<cplx> arrowhead_eig(const std::vector<T>& zj, const std::vector<T>& top, linalg::Scratch& sc) {
    const lint m = static_cast<lint>(zj.size()), k = m + 1;
    std::vector<T> E(static_cast<std::size_t>(k * k), T{}), B(static_cast<std::size_t>(k * k), T{});
    for (lint j = 0; j < m; ++j) {
        E[0 + (1 + j) * k] = top[j];
        E[(1 + j) + 0 * k] = T{1};
        E[(1 + j) + (1 + j) * k] = zj[j];
        B[(1 + j) + (1 + j) * k] = T{1};
    }
    std::vector<cplx> w;
    if constexpr (std::is_same_v<T, double>) {
        std::vector<double> ar(k), ai(k), be(k);
        if (lapack::ggev(k, E.data(), k, B.data(), k, ar.data(), ai.data(), be.data(), sc) != 0)
            throw linalg::LinAlgError("the generalized eigenvalue problem did not converge");
        for (lint i = 0; i < k; ++i)
            if (be[i] != 0) w.push_back(cdiv(cplx(ar[i], ai[i]), cplx(be[i], 0.0)));
    } else {
        std::vector<cplx> al(k), be(k);
        if (lapack::ggev(k, E.data(), k, B.data(), k, al.data(), be.data(), sc) != 0)
            throw linalg::LinAlgError("the generalized eigenvalue problem did not converge");
        for (lint i = 0; i < k; ++i)
            if (!zerov(be[i])) w.push_back(cdiv(al[i], be[i]));
    }
    std::vector<cplx> out;
    for (const cplx& v : w)
        if (finitev(v)) out.push_back(v);
    return out;
}

inline cplx as_c(double v) { return {v, 0.0}; }
inline cplx as_c(const cplx& v) { return v; }

template <class T>
std::vector<cplx> residues(
    const std::vector<T>& zj, const std::vector<T>& fj, const std::vector<T>& wj, const std::vector<cplx>& poles
) {
    const std::size_t m = zj.size();
    std::vector<cplx> out(poles.size());
    std::vector<cplx> fw(m);
    for (std::size_t j = 0; j < m; ++j) fw[j] = as_c(mulv(fj[j], wj[j]));
    for (std::size_t p = 0; p < poles.size(); ++p) {
        cplx num{0.0, 0.0}, dd{0.0, 0.0};
        for (std::size_t j = 0; j < m; ++j) {
            const cplx c = cdiv(cplx(1.0, 0.0), subv(poles[p], as_c(zj[j])));
            num = addv(num, mulv(c, fw[j]));
            dd = addv(dd, mulv(mulv(c, c), as_c(wj[j])));
        }
        out[p] = cdiv(num, cplx(-dd.real(), -dd.imag()));
    }
    return out;
}

// ---------------------------------------------------------------- the AAA run

template <class T>
struct Run {
    std::vector<T> zs, fs, zj, fj, wj;
    std::vector<double> errors;
    bool converged = false;
};

template <class T>
void aaa_run(const T* z, const T* f, std::int64_t n, double rtol, std::int64_t max_terms, Run<T>& out) {
    // the samples: finite values, sorted points, the first sample of a repeated point (numpy's unique)
    std::vector<std::int64_t> keep;
    for (std::int64_t i = 0; i < n; ++i)
        if (finitev(f[i])) keep.push_back(i);
    std::stable_sort(keep.begin(), keep.end(), [&](std::int64_t a, std::int64_t b) { return lessv(z[a], z[b]); });
    for (std::size_t i = 0; i < keep.size(); ++i)
        if (i == 0 || !samev(z[keep[i]], z[keep[i - 1]])) out.zs.push_back(z[keep[i]]), out.fs.push_back(f[keep[i]]);
    const std::int64_t M = static_cast<std::int64_t>(out.zs.size());
    if (M == 0) return;
    const std::vector<T>& Z = out.zs;
    const std::vector<T>& F = out.fs;
    const T mean = divr(np_sum(F.data(), M), static_cast<double>(M));
    double fmax = 0.0;
    for (const T& v : F) fmax = std::max(fmax, absv(v));
    const double atol = rtol * fmax;
    std::vector<T> R(M, mean);
    std::vector<char> mask(M, 1);
    std::vector<std::vector<T>> C, A;
    std::vector<T> zj, fj, wj, Am, vt, N(M), Dn(M);
    std::vector<double> s, col_norm;
    linalg::Scratch sc;
    bool ill = false;
    const double ill_tol = 1.0 / (3.0 * kEps);
    const std::int64_t steps = std::min(max_terms, M);
    for (std::int64_t m = 0; m < steps; ++m) {
        // the next support point: numpy's argmax of |f - R| over the other samples (a NaN wins)
        std::int64_t jj = -1;
        double best = -kInf;
        for (std::int64_t i = 0; i < M; ++i) {
            if (!mask[i]) continue;
            const double e = absv(subv(F[i], R[i]));
            if (e != e) {
                jj = i;
                break;
            }
            if (jj < 0 || e > best) best = e, jj = i;
        }
        zj.push_back(Z[jj]);
        fj.push_back(F[jj]);
        std::vector<T> c(M), a(M);
        for (std::int64_t i = 0; i < M; ++i) {
            c[i] = divv(T{1}, subv(Z[i], Z[jj]));
            a[i] = mulv(subv(F[i], F[jj]), c[i]);
        }
        C.push_back(std::move(c));
        A.push_back(std::move(a));
        mask[jj] = 0;
        const lint cols = m + 1;
        lint rows = 0;
        for (std::int64_t i = 0; i < M; ++i) rows += mask[i];
        auto pack = [&](bool scaled) {  // A's rows at the other samples, column major
            Am.resize(static_cast<std::size_t>(rows * cols));
            for (lint j = 0; j < cols; ++j) {
                lint r = 0;
                for (std::int64_t i = 0; i < M; ++i)
                    if (mask[i]) Am[r++ + j * rows] = scaled ? divr(A[j][i], col_norm[j]) : A[j][i];
            }
        };
        wj.assign(static_cast<std::size_t>(cols), T{});
        if (rows >= cols) {
            if (!ill) {
                pack(false);
                svd('S', rows, cols, Am, s, vt, sc);
                if (s[0] / s[cols - 1] > ill_tol) ill = true;
            }
            if (ill) {
                col_norm.assign(static_cast<std::size_t>(cols), 0.0);
                for (lint j = 0; j < cols; ++j) {
                    double q = 0.0;
                    for (std::int64_t i = 0; i < M; ++i)
                        if (mask[i]) {
                            const T v = A[j][i];
                            if constexpr (std::is_same_v<T, double>) q += v * v;
                            else q += v.real() * v.real() + v.imag() * v.imag();
                        }
                    col_norm[j] = std::sqrt(q);
                    if (col_norm[j] == 0) col_norm[j] = 1.0;  // a zero column stays as it is (scipy: NaN)
                }
                pack(true);
                svd('S', rows, cols, Am, s, vt, sc);
            }
            double smin = s[0];
            for (double v : s) smin = std::min(smin, v);
            std::int64_t count = 0;
            for (lint r = 0; r < cols; ++r) {
                if (s[r] != smin) continue;
                for (lint j = 0; j < cols; ++j) {
                    const T v = conjv(vt[r + j * cols]);
                    wj[j] = count == 0 ? v : addv(wj[j], v);
                }
                ++count;
            }
            const double root = std::sqrt(static_cast<double>(count));
            for (lint j = 0; j < cols; ++j) wj[j] = divr(wj[j], root);
            if (ill)
                for (lint j = 0; j < cols; ++j) wj[j] = divr(wj[j], col_norm[j]);
        } else {
            pack(false);
            svd('A', rows, cols, Am, s, vt, sc);
            double smax = 0.0;
            for (double v : s) smax = std::max(smax, v);
            const double tol = smax * kEps * static_cast<double>(std::max(rows, cols));
            lint num = 0;
            for (double v : s) num += v > tol;
            const lint nm = cols - num;
            std::vector<T> q(static_cast<std::size_t>(nm));
            const double root = std::sqrt(static_cast<double>(nm));
            for (lint j = 0; j < cols; ++j) {
                for (lint r = 0; r < nm; ++r) q[r] = conjv(vt[(num + r) + j * cols]);
                wj[j] = divr(np_sum(q.data(), nm), root);
            }
        }
        // the rational at every sample
        std::vector<T> wf(static_cast<std::size_t>(cols));
        for (lint j = 0; j < cols; ++j) wf[j] = mulv(wj[j], fj[j]);
        double max_error = 0.0;
        bool nan = false;
        for (std::int64_t i = 0; i < M; ++i) {
            T nn{}, dd{};
            for (lint j = 0; j < cols; ++j) {
                if (zerov(wj[j])) continue;
                nn = addv(nn, mulv(C[j][i], wf[j]));
                dd = addv(dd, mulv(C[j][i], wj[j]));
            }
            if (infnanv(dd)) dd = T{1}, nn = F[i];
            R[i] = divv(nn, dd);
            const double e = absv(subv(F[i], R[i]));
            if (e != e) nan = true;
            max_error = std::max(max_error, e);
        }
        if (nan) max_error = kNaN;
        out.errors.push_back(max_error);
        if (max_error <= atol) {
            out.converged = true;
            break;
        }
    }
    for (std::size_t j = 0; j < wj.size(); ++j)
        if (!zerov(wj[j])) out.zj.push_back(zj[j]), out.fj.push_back(fj[j]), out.wj.push_back(wj[j]);
}

// ---------------------------------------------------------------- clean_up

template <class T>
std::int64_t clean_up(
    const std::vector<T>& zs,
    const std::vector<T>& fs,
    std::vector<T>& zj,
    std::vector<T>& fj,
    std::vector<T>& wj,
    double tol
) {
    linalg::Scratch sc;
    const std::vector<cplx> poles = arrowhead_eig(zj, wj, sc);
    const std::vector<cplx> res = residues(zj, fj, wj, poles);
    const std::int64_t M = static_cast<std::int64_t>(zs.size());
    std::vector<double> logs(M);
    for (std::int64_t i = 0; i < M; ++i) logs[i] = std::log(absv(fs[i]));
    const double gm = std::exp(np_sum(logs.data(), M) / static_cast<double>(M));
    std::vector<std::size_t> spurious;
    for (std::size_t p = 0; p < poles.size(); ++p) {
        double dist = kInf;
        for (std::int64_t i = 0; i < M; ++i) dist = std::min(dist, absv(subv(poles[p], as_c(zs[i]))));
        if (absv(res[p]) / dist < tol * gm) spurious.push_back(p);
    }
    const std::int64_t ni = static_cast<std::int64_t>(spurious.size());
    if (ni == 0) return 0;
    std::vector<char> drop(zj.size(), 0);
    for (std::size_t p : spurious) {
        std::size_t best = 0;
        double bd = kInf;
        for (std::size_t j = 0; j < zj.size(); ++j) {
            const double d = absv(subv(as_c(zj[j]), poles[p]));
            if (j == 0 || d < bd) bd = d, best = j;
        }
        drop[best] = 1;
    }
    std::vector<T> z2, f2;
    for (std::size_t j = 0; j < zj.size(); ++j)
        if (!drop[j]) z2.push_back(zj[j]), f2.push_back(fj[j]);
    zj = z2;
    fj = f2;
    const lint m = static_cast<lint>(zj.size());
    wj.assign(static_cast<std::size_t>(m), T{});
    if (m == 0) return ni;
    std::vector<std::int64_t> rows;
    for (std::int64_t i = 0; i < M; ++i) {
        bool other = true;
        for (lint j = 0; j < m && other; ++j) other = !samev(zs[i], zj[j]);
        if (other) rows.push_back(i);
    }
    const lint r = static_cast<lint>(rows.size());
    std::vector<T> Am(static_cast<std::size_t>(r * m)), vt;
    std::vector<double> s;
    for (lint j = 0; j < m; ++j)
        for (lint k = 0; k < r; ++k) {
            const std::int64_t i = rows[k];
            const T c = divv(T{1}, subv(zs[i], zj[j]));
            Am[k + j * r] = subv(mulv(fs[i], c), mulv(c, fj[j]));
        }
    svd(r >= m ? 'S' : 'A', r, m, Am, s, vt, sc);  // V^H is the same either way; 'A' only when U is small
    for (lint j = 0; j < m; ++j) wj[j] = conjv(vt[(m - 1) + j * m]);
    return ni;
}

// ---------------------------------------------------------------- bindings

struct In1 {
    const void* p;
    std::int64_t n;
    bool complex;
};

In1 in1(nb::handle o, const char* name) {
    const NDArray& a = require_ndarray(o, name);
    if (a.ndim != 1 || !a.c_contiguous || (a.dtype != DType::Float64 && a.dtype != DType::Complex128))
        throw nb::value_error(
            (std::string("'") + name + "' must be a C-contiguous 1-D float64 or complex128 array").c_str()
        );
    return {a.data, a.shape[0], a.dtype == DType::Complex128};
}

template <class T>
nb::object array_of(const std::vector<T>& v) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(v.size())};
    NDArray a = new_array(std::is_same_v<T, double> ? DType::Float64 : DType::Complex128, shape);
    if (!v.empty()) std::memcpy(a.data, v.data(), v.size() * sizeof(T));
    return wrap(std::move(a));
}

template <class T>
std::vector<T> vector_of(const In1& a) {
    const T* p = static_cast<const T*>(a.p);
    return std::vector<T>(p, p + a.n);
}

template <class T>
nb::object fit_as(const In1& z, const In1& f, double rtol, std::int64_t max_terms) {
    Run<T> run;
    {  // no Python object inside; the GIL kept for small work (API.md §3, as the area's other kernels)
        const parallel::Unlocked unlocked(z.n * std::min(z.n, max_terms) * 8 >= kParallelThreshold);
        aaa_run(static_cast<const T*>(z.p), static_cast<const T*>(f.p), z.n, rtol, max_terms, run);
    }
    return nb::make_tuple(
        array_of(run.zs), array_of(run.fs), array_of(run.zj), array_of(run.fj), array_of(run.wj), array_of(run.errors),
        run.converged
    );
}

nb::object aaa_fit(nb::handle zo, nb::handle fo, double rtol, std::int64_t max_terms) {
    const In1 z = in1(zo, "z"), f = in1(fo, "f");
    if (z.n != f.n || z.complex != f.complex) throw nb::value_error("'z' and 'f' must have one length and one dtype");
    if (max_terms < 1) throw nb::value_error("'max_terms' must be positive");
    return z.complex ? fit_as<cplx>(z, f, rtol, max_terms) : fit_as<double>(z, f, rtol, max_terms);
}

nb::object bary_eig(nb::handle zo, nb::handle wo, nb::handle to) {
    const In1 z = in1(zo, "zj"), w = in1(wo, "wj"), t = in1(to, "top");
    if (z.n != w.n || z.n != t.n || z.complex != w.complex || z.complex != t.complex)
        throw nb::value_error("'zj', 'wj' and 'top' must have one length and one dtype");
    std::vector<cplx> out;
    linalg::Scratch sc;
    const bool big = z.n * z.n * z.n >= kParallelThreshold;
    if (z.complex) {
        const auto zz = vector_of<cplx>(z), tt = vector_of<cplx>(t);
        const parallel::Unlocked unlocked(big);
        out = arrowhead_eig(zz, tt, sc);
    } else {
        const auto zz = vector_of<double>(z), tt = vector_of<double>(t);
        const parallel::Unlocked unlocked(big);
        out = arrowhead_eig(zz, tt, sc);
    }
    return array_of(out);
}

nb::object bary_residues(nb::handle zo, nb::handle fo, nb::handle wo, nb::handle po) {
    const In1 z = in1(zo, "zj"), f = in1(fo, "fj"), w = in1(wo, "wj"), p = in1(po, "poles");
    if (z.n != f.n || z.n != w.n || z.complex != f.complex || z.complex != w.complex || !p.complex)
        throw nb::value_error("'zj', 'fj' and 'wj' must have one length and one dtype, 'poles' complex128");
    const auto pp = vector_of<cplx>(p);
    if (z.complex) return array_of(residues(vector_of<cplx>(z), vector_of<cplx>(f), vector_of<cplx>(w), pp));
    return array_of(residues(vector_of<double>(z), vector_of<double>(f), vector_of<double>(w), pp));
}

template <class T>
nb::object clean_as(const In1& zs, const In1& fs, const In1& zj, const In1& fj, const In1& wj, double tol) {
    std::vector<T> z = vector_of<T>(zj), f = vector_of<T>(fj), w = vector_of<T>(wj);
    const std::vector<T> zz = vector_of<T>(zs), ff = vector_of<T>(fs);
    std::int64_t ni;
    {
        const parallel::Unlocked unlocked(zs.n * zj.n * 8 >= kParallelThreshold);
        ni = clean_up(zz, ff, z, f, w, tol);
    }
    return nb::make_tuple(ni, array_of(z), array_of(f), array_of(w));
}

nb::object aaa_clean_up(nb::handle zso, nb::handle fso, nb::handle zo, nb::handle fo, nb::handle wo, double tol) {
    const In1 zs = in1(zso, "zs"), fs = in1(fso, "fs"), z = in1(zo, "zj"), f = in1(fo, "fj"), w = in1(wo, "wj");
    for (const In1* a : {&fs, &z, &f, &w})
        if (a->complex != zs.complex) throw nb::value_error("the arrays must have one dtype");
    if (zs.n != fs.n || z.n != f.n || z.n != w.n) throw nb::value_error("the arrays' lengths do not match");
    return zs.complex ? clean_as<cplx>(zs, fs, z, f, w, tol) : clean_as<double>(zs, fs, z, f, w, tol);
}

}  // namespace

void bind_aaa(nb::module_& m) {
    m.def("aaa_fit", &aaa_fit, nb::arg("z").none(), nb::arg("f").none(), nb::arg("rtol"), nb::arg("max_terms"));
    m.def("bary_eig", &bary_eig, nb::arg("zj").none(), nb::arg("wj").none(), nb::arg("top").none());
    m.def(
        "bary_residues", &bary_residues, nb::arg("zj").none(), nb::arg("fj").none(), nb::arg("wj").none(),
        nb::arg("poles").none()
    );
    m.def(
        "aaa_clean_up", &aaa_clean_up, nb::arg("zs").none(), nb::arg("fs").none(), nb::arg("zj").none(),
        nb::arg("fj").none(), nb::arg("wj").none(), nb::arg("tol")
    );
}

}  // namespace sd::interpolate
