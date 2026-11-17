// The matrix distributions' kernels (T10, docs/design/stats.md §6.3; decisions T30): "wishart_*"
// and "matnorm_*" in stardust._core.stats. The classes in stardust.stats._matrix flatten the
// batch as the multivariate ones do (parameters (B, ...), points (M, p, p) with a batch index
// each); the parameters' Cholesky factors come from mv_chol (bind/stats_mv.cpp).
//
// Wishart W_p(n, V) and inverse Wishart IW_p(n, Psi), n > p - 1 real:
//     log W(X)  = (n-p-1)/2 log|X| - tr(V^-1 X)/2 - np/2 log 2 - n/2 log|V| - log Gamma_p(n/2)
//     log IW(X) = n/2 log|Psi| - np/2 log 2 - log Gamma_p(n/2) - (n+p+1)/2 log|X| - tr(Psi X^-1)/2
// with X's lower triangle factored in double-double per point (its positive definiteness is
// the support: else pdf 0), V^-1 in double-double per batch, tr(Psi X^-1) = |L_X^-1 L_Psi|_F^2.
// Matrix normal MN(M, U, V) (n x p): -np/2 log 2 pi - p/2 log|U| - n/2 log|V| - |A|_F^2 / 2,
// A = L_U^-1 (X - M) L_V^-T.
//
// The matrix normal's fast path (stats/dd_linalg.hpp, as the multivariate normal's): A by plain
// dot products with the rounded inverses of the two factors, the first product's error bounds
// carried through the second, and |A|_F^2 taken where its bound fits kQuadBudget; every other
// point runs the double-double substitutions.
#include "bind/stats_bind.hpp"

#include "array/convert.hpp"
#include "array/ops/pieces.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "random/philox.hpp"
#include "special/ops/gamma.hpp"
#include "stats/dd_linalg.hpp"
#include "stats/numeric.hpp"
#include "stats/sample.hpp"
#include "stats/sample_dist.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd {

namespace {

using stats::num::DD;
namespace dd = stats::num::dd;
namespace num = stats::num;
namespace la = stats::ddla;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kQuadBudget = la::kQuadBudget;  // the quadratic forms' fast path, 40 u
constexpr double kU = la::kU;

const NDArray& f64(nb::handle o, int ndim, const char* what) {
    const NDArray& a = require_ndarray(o);
    if (a.dtype != DType::Float64 || a.ndim != ndim)
        throw nb::type_error((std::string("matrix: '") + what + "' must be a float64 array of " +
                              std::to_string(ndim) + " dimensions").c_str());
    return a;
}
const NDArray& i64(nb::handle o) {
    const NDArray& a = require_ndarray(o);
    if (a.dtype != DType::Int64 || a.ndim != 1) throw nb::type_error("matrix: 'batch' must be an int64 vector");
    return a;
}
inline double el(const NDArray& a, std::int64_t i) {
    return *reinterpret_cast<const double*>(reinterpret_cast<const char*>(a.data) + i * a.strides[0]);
}
inline double el(const NDArray& a, std::int64_t b, std::int64_t i, std::int64_t j) {
    return *reinterpret_cast<const double*>(reinterpret_cast<const char*>(a.data) + b * a.strides[0] +
                                            i * a.strides[1] + j * a.strides[2]);
}
inline std::int64_t eli(const NDArray& a, std::int64_t i) {
    return *reinterpret_cast<const std::int64_t*>(reinterpret_cast<const char*>(a.data) + i * a.strides[0]);
}
NDArray out_vec(std::int64_t n) {
    const std::int64_t s[] = {n};
    return new_array(DType::Float64, s);
}
// L (B, d, d) as hi/lo arrays -> a double-double lower matrix
void load_L(const NDArray& h, const NDArray& l, std::int64_t b, std::int64_t d, std::vector<DD>& L) {
    L.assign(static_cast<std::size_t>(d * d), DD{0.0, 0.0});
    for (std::int64_t i = 0; i < d; ++i)
        for (std::int64_t j = 0; j <= i; ++j) L[i * d + j] = DD{el(h, b, i, j), el(l, b, i, j)};
}

// ---- Wishart and inverse Wishart --------------------------------------------------------------

// The scaling of arguments far out (Wishart's X, the matrix normal's X - M): 2^-600 brings
// 2^1024 to 2^424, whose products and squares stay in range.
constexpr int kFar = 600;

// The terms of a Wishart log density (the constant's and log|X|'s) below which their
// cancellation needs no check: wishart_logpdf.
constexpr double kPlainTerms = 1e15;

// log pdf at x (M, p, p) (lower triangles read): df (B,), the scale's factor L hi/lo (B, p, p)
// and log det hi/lo (B,), ok (B,). inverse: the inverse Wishart.
nb::object wishart_logpdf(nb::handle x_obj, nb::handle batch_obj, nb::handle df_obj, nb::handle lh_obj,
                          nb::handle ll_obj, nb::handle ldh_obj, nb::handle ldl_obj, nb::handle ok_obj, bool inverse,
                          bool as_pdf) {
    const NDArray &x = f64(x_obj, 3, "x"), &dfa = f64(df_obj, 1, "df"), &Lh = f64(lh_obj, 3, "L"),
                  &Ll = f64(ll_obj, 3, "L"), &ldh = f64(ldh_obj, 1, "logdet"), &ldl = f64(ldl_obj, 1, "logdet"),
                  &ok = f64(ok_obj, 1, "ok");
    const NDArray& bi = i64(batch_obj);
    const std::int64_t M = x.shape[0], p = x.shape[1], B = dfa.shape[0];
    const double pd = static_cast<double>(p);
    // per batch: the constant, and V^-1 (Wishart) or L_Psi (inverse) in double-double
    std::vector<DD> cst(static_cast<std::size_t>(B));
    std::vector<double> cmag(static_cast<std::size_t>(B), 0.0);  // the size of the constant's terms
    std::vector<std::vector<DD>> mats(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) {
        if (el(ok, b) == 0.0) continue;
        const double n = el(dfa, b);
        const DD ld{el(ldh, b), el(ldl, b)};
        const DD lmg = la::log_mvgamma(p, 0.5 * n);
        const DD base = -(dd::kLn2 * (0.5 * n * pd)) - lmg;
        cmag[b] = 0.35 * n * pd + std::fabs(lmg.hi) + std::fabs(ld.hi) * 0.5 * n;
        std::vector<DD> L;
        load_L(Lh, Ll, b, p, L);
        if (!inverse) {
            cst[b] = base - ld * (0.5 * n);
            la::inverse_from_cholesky(L, p, mats[b]);
        } else {
            cst[b] = base + ld * (0.5 * n);
            mats[b] = std::move(L);
        }
    }
    NDArray out = out_vec(M);
    double* po = reinterpret_cast<double*>(out.data);
    {
        nb::gil_scoped_release release;
        for_pieces(M, std::max<std::int64_t>(1, 2048 / std::max<std::int64_t>(1, p * p * p)), M * p * p * p,
                   [&](std::int64_t, std::int64_t m0, std::int64_t m1) {
                       std::vector<DD> LX, col(static_cast<std::size_t>(p));
                       for (std::int64_t m = m0; m < m1; ++m) {
                           const std::int64_t b = eli(bi, m);
                           const double n = el(dfa, b);
                           bool nan = el(ok, b) == 0.0;
                           double big = 0.0;
                           for (std::int64_t i = 0; i < p && !nan; ++i)
                               for (std::int64_t j = 0; j <= i && !nan; ++j) {
                                   nan = std::isnan(el(x, m, i, j));
                                   big = std::max(big, std::fabs(el(x, m, i, j)));
                               }
                           if (nan) {
                               po[m] = kNaN;
                               continue;
                           }
                           if (std::isinf(big)) {  // an infinite entry: the density is 0 (both kinds)
                               po[m] = as_pdf ? 0.0 : -kInf;
                               continue;
                           }
                           // X far out (> 2^500) is factored as 2^-kFar X, so that the Cholesky
                           // products stay in range; log|X| and the trace are put back below.
                           const int shift = big > 0x1p500 ? kFar : 0;
                           DD ldx;
                           const bool pdx = la::cholesky(
                               p, [&](std::int64_t i, std::int64_t j) { return std::ldexp(el(x, m, i, j), -shift); }, LX, ldx);
                           if (!pdx) {  // off the support
                               po[m] = as_pdf ? 0.0 : -kInf;
                               continue;
                           }
                           const std::vector<DD>& A = mats[b];
                           DD tr{0.0, 0.0};
                           if (!inverse) {  // tr(V^-1 X) over X's lower triangle
                               for (std::int64_t i = 0; i < p; ++i) {
                                   tr = tr + A[i * p + i] * std::ldexp(el(x, m, i, i), -shift);
                                   for (std::int64_t j = 0; j < i; ++j)
                                       tr = tr + A[i * p + j] * (2.0 * std::ldexp(el(x, m, i, j), -shift));
                               }
                           } else {  // |L_X^-1 L_Psi|_F^2, column by column
                               for (std::int64_t c = 0; c < p; ++c) {
                                   for (std::int64_t i = 0; i < p; ++i) col[i] = i >= c ? A[i * p + c] : DD{0.0, 0.0};
                                   la::solve_lower(LX, p, col.data());
                                   for (std::int64_t i = 0; i < p; ++i) tr = tr + col[i] * col[i];
                               }
                           }
                           DD htr = tr * 0.5;  // the trace's half: what the density takes (exact)
                           if (shift != 0) {  // log|X| = log|2^-s X| + p s log 2; the trace scales by 2^(+-s)
                               ldx = ldx + dd::kLn2 * (pd * shift);
                               htr = dd::ldexp(htr, inverse ? -shift : shift);
                               if (!std::isfinite(htr.hi)) {  // tr / 2 beyond the doubles: the density is 0
                                   po[m] = as_pdf ? 0.0 : -kInf;
                                   continue;
                               }
                           }
                           // n -+ (p + 1) kept exactly: its rounding (2^-54 of the coefficient)
                           // times log|X| would reach the density
                           const DD coef = inverse ? dd::two_sum(n, pd + 1.0) * -0.5 : dd::two_sum(n, -(pd + 1.0)) * 0.5;
                           const DD lx = ldx * coef;
                           const DD l = cst[b] + lx - htr;
                           // the terms are of size df p log df and cancel to O(p^2 log df): each
                           // is good to ~2^-100 of its size, so from df ~ 1e17 on the
                           // double-double sum no longer holds the value (measured: 1e-12 at
                           // 1e18, nothing at 1e30): NaN
                           // (With the constant's terms and lx below kPlainTerms the test cannot
                           // fail: the trace alone would have to be 1e17 times l, which it is
                           // part of. One comparison for the usual df.)
                           if (!(cmag[b] + std::fabs(lx.hi) <= kPlainTerms)) [[unlikely]] {
                               const double mag = cmag[b] + std::fabs(lx.hi) + std::fabs(htr.hi);
                               if (!(0x1p-100 * mag <= 1e-13 * std::max(1.0, std::fabs(l.hi)))) {
                                   po[m] = kNaN;
                                   continue;
                               }
                           }
                           po[m] = as_pdf ? num::exp_dd(l) : l.hi + l.lo;
                       }
                   });
    }
    return wrap(std::move(out));
}

// The entropies, regrouped (docs/design/stats.md §6.3) so the terms of size n log n cancel
// inside G(a_j) = log Gamma(a) - (a - 1) psi(a) + a - log(a)/2 - log(2 pi)/2 = O(1), a_j =
// n/2 + (1 - j)/2, and -sum a_j + np/2 = p(p-1)/4 exactly:
//   W:  (p+1)/2 log|V| + p(p+1)/2 log 2 + p(p-1)/4 (1 + log pi)
//         + sum_j [G(a_j) + log(a_j)/2 + log(2 pi)/2 + (p - j)/2 psi(a_j)]
//   IW: (p+1)/2 log|Psi| - p(p+1)/2 log 2 + p(p-1)/4 (1 + log pi)
//         + sum_j [G(a_j) + log(a_j)/2 + log(2 pi)/2 - (1 + (p + j)/2) psi(a_j)]
nb::object wishart_entropy(nb::handle df_obj, nb::handle ldh_obj, nb::handle ldl_obj, nb::handle ok_obj,
                           std::int64_t p, bool inverse) {
    const NDArray &dfa = f64(df_obj, 1, "df"), &ldh = f64(ldh_obj, 1, "logdet"), &ldl = f64(ldl_obj, 1, "logdet"),
                  &ok = f64(ok_obj, 1, "ok");
    const std::int64_t B = dfa.shape[0];
    NDArray out = out_vec(B);
    double* po = reinterpret_cast<double*>(out.data);
    const double pd = static_cast<double>(p);
    for (std::int64_t b = 0; b < B; ++b) {
        if (el(ok, b) == 0.0) {
            po[b] = kNaN;
            continue;
        }
        const double n = el(dfa, b);
        DD h = DD{el(ldh, b), el(ldl, b)} * (0.5 * (pd + 1.0)) +
               dd::kLn2 * ((inverse ? -0.5 : 0.5) * pd * (pd + 1.0)) +
               (num::kLogPi + 1.0) * (0.25 * pd * (pd - 1.0));
        double g = 0.0;
        for (std::int64_t j = 1; j <= p; ++j) {
            const DD aj = dd::two_sum(0.5 * n, 0.5 * static_cast<double>(1 - j));
            const double a = aj.hi + aj.lo;
            const double psi = special::Digamma::f(a);
            g += num::entropy_g(a, special::Digamma::f);
            g += inverse ? -(1.0 + 0.5 * (pd + static_cast<double>(j))) * psi : 0.5 * (pd - static_cast<double>(j)) * psi;
            h = h + dd::log(aj) * 0.5 + num::kLogSqrt2Pi;
        }
        h = h + g;
        po[b] = h.hi + h.lo;
    }
    return wrap(std::move(out));
}

// Samples (N, p, p), row i stream element pos + i, batch i mod B. Bartlett: A lower with
// A_jj = sqrt(chi^2(n - j)) (j = 0..p-1, real n: 2 G((n - j)/2)) and A_ij ~ N(0, 1) below;
// W = (L A)(L A)^T. Inverse: X = (L A^-T)(L A^-T)^T with L the factor of Psi (X^-1 is then
// W(n, Psi^-1)).
nb::object wishart_sample(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, std::int64_t N, nb::handle df_obj,
                          nb::handle lh_obj, nb::handle ok_obj, bool inverse) {
    const NDArray &dfa = f64(df_obj, 1, "df"), &Lh = f64(lh_obj, 3, "L"), &ok = f64(ok_obj, 1, "ok");
    const std::int64_t B = dfa.shape[0], p = Lh.shape[1];
    const std::int64_t s3[] = {N, p, p};
    NDArray out = new_array(DType::Float64, s3);
    double* po = reinterpret_cast<double*>(out.data);
    const random::Key key{k0, k1};
    {
        nb::gil_scoped_release release;
        for_pieces(N, std::max<std::int64_t>(1, 1024 / std::max<std::int64_t>(1, p * p)), N * p * p * (p + 20),
                   [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
                       random::Stream s(key);
                       std::vector<double> A(static_cast<std::size_t>(p * p)), T(static_cast<std::size_t>(p * p));
                       for (std::int64_t i = i0; i < i1; ++i) {
                           const std::int64_t b = i % B;
                           double* o = po + i * p * p;
                           if (el(ok, b) == 0.0) {
                               for (std::int64_t k = 0; k < p * p; ++k) o[k] = kNaN;
                               continue;
                           }
                           const double n = el(dfa, b);
                           stats::smp::Words w(s, pos + static_cast<std::uint64_t>(i));
                           for (std::int64_t r = 0; r < p; ++r) {
                               for (std::int64_t c = 0; c < r; ++c) A[r * p + c] = stats::smp::std_normal(w);
                               const auto g = stats::smp::gamma_prep(0.5 * (n - static_cast<double>(r)));
                               A[r * p + r] = std::sqrt(2.0 * stats::smp::gamma_draw(w, g));
                               for (std::int64_t c = r + 1; c < p; ++c) A[r * p + c] = 0.0;
                           }
                           if (inverse) {  // A <- A^-T (upper): solve A^T Y = I column by column
                               std::vector<double> inv(static_cast<std::size_t>(p * p), 0.0);
                               for (std::int64_t c = 0; c < p; ++c) {  // L^-1 e_c for the lower A, then transpose
                                   for (std::int64_t r = 0; r < p; ++r) {
                                       double t = r == c ? 1.0 : 0.0;
                                       for (std::int64_t k = 0; k < r; ++k) t -= A[r * p + k] * inv[k * p + c];
                                       inv[r * p + c] = t / A[r * p + r];
                                   }
                               }
                               for (std::int64_t r = 0; r < p; ++r)
                                   for (std::int64_t c = 0; c < p; ++c) A[r * p + c] = inv[c * p + r];  // (A^-1)^T
                           }
                           // T = L A (L lower; A lower, or upper for the inverse)
                           for (std::int64_t r = 0; r < p; ++r)
                               for (std::int64_t c = 0; c < p; ++c) {
                                   double t = 0.0;
                                   for (std::int64_t k = 0; k <= r; ++k) t += el(Lh, b, r, k) * A[k * p + c];
                                   T[r * p + c] = t;
                               }
                           for (std::int64_t r = 0; r < p; ++r)
                               for (std::int64_t c = 0; c <= r; ++c) {
                                   double t = 0.0;
                                   for (std::int64_t k = 0; k < p; ++k) t += T[r * p + k] * T[c * p + k];
                                   o[r * p + c] = t;
                                   o[c * p + r] = t;
                               }
                       }
                   });
    }
    return wrap(std::move(out));
}

// ---- matrix normal ----------------------------------------------------------------------------

// log pdf at x (M, n, p): mean (B, n, p), the factors of U (B, n, n) and V (B, p, p) hi/lo,
// their log dets hi/lo (B,), ok (B,).
nb::object matnorm_logpdf(nb::handle x_obj, nb::handle batch_obj, nb::handle mean_obj, nb::handle uh_obj,
                          nb::handle ul_obj, nb::handle vh_obj, nb::handle vl_obj, nb::handle ldu_h, nb::handle ldu_l,
                          nb::handle ldv_h, nb::handle ldv_l, nb::handle ok_obj, nb::handle fu_obj, nb::handle fv_obj,
                          bool as_pdf) {
    const NDArray &x = f64(x_obj, 3, "x"), &mean = f64(mean_obj, 3, "mean"), &Uh = f64(uh_obj, 3, "U"),
                  &Ul = f64(ul_obj, 3, "U"), &Vh = f64(vh_obj, 3, "V"), &Vl = f64(vl_obj, 3, "V"),
                  &luh = f64(ldu_h, 1, "logdet"), &lul = f64(ldu_l, 1, "logdet"), &lvh = f64(ldv_h, 1, "logdet"),
                  &lvl = f64(ldv_l, 1, "logdet"), &ok = f64(ok_obj, 1, "ok");
    const NDArray& bi = i64(batch_obj);
    const std::int64_t M = x.shape[0], n = x.shape[1], p = x.shape[2], B = mean.shape[0];
    std::vector<std::vector<DD>> LU(static_cast<std::size_t>(B)), LV(static_cast<std::size_t>(B));
    std::vector<DD> cst(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) {
        if (el(ok, b) == 0.0) continue;
        load_L(Uh, Ul, b, n, LU[b]);
        load_L(Vh, Vl, b, p, LV[b]);
        cst[b] = -(num::kLogSqrt2Pi * static_cast<double>(n * p)) - DD{el(luh, b), el(lul, b)} * (0.5 * static_cast<double>(p)) -
                 DD{el(lvh, b), el(lvl, b)} * (0.5 * static_cast<double>(n));
    }
    NDArray out = out_vec(M);
    double* po = reinterpret_cast<double*>(out.data);
    // the fast path's factors of U (B, fast_size(n)) and V (B, fast_size(p)), contiguous, or none
    const double *pfu = nullptr, *pfv = nullptr;
    const std::int64_t fsu = la::fast_size(n), fsv = la::fast_size(p), np_ = la::padded(n), pp = la::padded(p);
    if (!fu_obj.is_none() && !fv_obj.is_none() && n > 0 && p > 0) {
        const NDArray &fu = f64(fu_obj, 2, "fast"), &fv = f64(fv_obj, 2, "fast");
        if (fu.shape[0] != B || fu.shape[1] != fsu || fu.strides[1] != 8 || fu.strides[0] != 8 * fsu ||
            fv.shape[0] != B || fv.shape[1] != fsv || fv.strides[1] != 8 || fv.strides[0] != 8 * fsv)
            throw nb::value_error("matrix: 'fast' does not fit the factors");
        pfu = reinterpret_cast<const double*>(fu.data), pfv = reinterpret_cast<const double*>(fv.data);
    }
    {
        nb::gil_scoped_release release;
        for_pieces(M, std::clamp<std::int64_t>(16384 / std::max<std::int64_t>(1, n * p * (n + p)), 16, 1024),
                   M * n * p * (n + p), [&](std::int64_t, std::int64_t m0, std::int64_t m1) {
                       std::vector<DD> Y(static_cast<std::size_t>(n * p));
                       // the fast path: R's row (zero past p), Y = R W_V^T and its bounds (n rows
                       // of pp), a column of Y and its bounds (zero past n), the column of A
                       std::vector<double> buf(static_cast<std::size_t>(pfu ? pp + 2 * n * pp + 4 * np_ : 0), 0.0);
                       double *rr = buf.data(), *Yf = rr + pp, *Ef = Yf + n * pp, *yc = Ef + n * pp, *ec = yc + np_,
                              *ac = ec + np_, *ea = ac + np_;
                       for (std::int64_t m = m0; m < m1; ++m) {
                           const std::int64_t b = eli(bi, m);
                           bool nan = el(ok, b) == 0.0, inf = false;
                           for (std::int64_t i = 0; i < n && !nan; ++i)
                               for (std::int64_t j = 0; j < p; ++j) {
                                   const double v = el(x, m, i, j);
                                   nan = nan || std::isnan(v);
                                   inf = inf || std::isinf(v);
                               }
                           if (nan || inf) {
                               po[m] = nan ? kNaN : (as_pdf ? 0.0 : -kInf);
                               continue;
                           }
                           if (pfu) {
                               // Row i of Y = (X - M) W_V^T is W_V applied to row i of X - M (each
                               // entry rounded once), to within Ef; column c of A = W_U Y is W_U
                               // applied to column c of Y with those bounds as its input's. Q is
                               // the sum of the columns' |.|^2 (p more additions: (p + 1) u Q), and
                               // l = c.hi - Q / 2 (its error exactly, by TwoSum) + c.lo rounds once
                               // more; the density exp(hi) (1 + lo) rounds three times (exp allowed
                               // 2 u): 4 u, its budget absolute. Anything non-finite fails the test.
                               const double *fu = pfu + b * fsu, *fv = pfv + b * fsv;
                               for (std::int64_t i = 0; i < n; ++i) {
                                   for (std::int64_t j = 0; j < p; ++j) rr[j] = el(x, m, i, j) - el(mean, b, i, j);
                                   la::fast_solve(fv, p, rr, nullptr, Yf + i * pp, Ef + i * pp);
                               }
                               double q = 0.0, err = 0.0;
                               for (std::int64_t c = 0; c < p; ++c) {
                                   for (std::int64_t i = 0; i < n; ++i) yc[i] = Yf[i * pp + c], ec[i] = Ef[i * pp + c];
                                   la::fast_solve(fu, n, yc, ec, ac, ea);
                                   double e1;
                                   q += la::fast_norm2(ac, ea, n, e1);
                                   err += e1;
                               }
                               err = (err + (static_cast<double>(p) + 1.5) * kU * q) * la::kBoundSlack;
                               const DD c = cst[b];
                               const double t = 0.5 * q, bound = 0.5 * err;
                               const DD s = dd::two_sum(c.hi, -t);
                               const double lo = s.lo + c.lo;
                               if (as_pdf) {
                                   if (bound + kU * (4.0 + kU * std::fabs(c.hi)) <= kQuadBudget && std::fabs(s.hi) <= 700.0) {
                                       po[m] = std::exp(s.hi) * (1.0 + lo);
                                       continue;
                                   }
                               } else {
                                   const double l = s.hi + lo;
                                   if (bound + kU * (std::fabs(l) + kU * std::fabs(c.hi)) <=
                                       kQuadBudget * std::max(1.0, std::fabs(l))) {
                                       po[m] = l;
                                       continue;
                                   }
                               }
                           }
                           // X - M far out (|.| > 2^400) is scaled by 2^-kFar before the solves, so
                           // that neither they nor the squares overflow; Q comes back by ldexp, and
                           // a log density below the doubles is -inf (it was NaN from inf - inf).
                           double big = 0.0;
                           for (std::int64_t i = 0; i < n; ++i)
                               for (std::int64_t j = 0; j < p; ++j) {
                                   Y[i * p + j] = dd::two_sum(el(x, m, i, j), -el(mean, b, i, j));
                                   big = std::max(big, std::fabs(Y[i * p + j].hi));
                               }
                           const int shift = (big > 0x1p400 || !std::isfinite(big)) ? kFar : 0;
                           if (shift != 0)
                               for (std::int64_t i = 0; i < n; ++i)
                                   for (std::int64_t j = 0; j < p; ++j)
                                       Y[i * p + j] = dd::two_sum(std::ldexp(el(x, m, i, j), -shift), -std::ldexp(el(mean, b, i, j), -shift));
                           for (std::int64_t j = 0; j < p; ++j) la::solve_lower(LU[b], n, Y.data() + j, p);  // L_U^-1 (X - M)
                           DD q{0.0, 0.0};
                           for (std::int64_t i = 0; i < n; ++i) {  // rows of Y L_V^-T: L_V z = row
                               la::solve_lower(LV[b], p, Y.data() + i * p);
                               for (std::int64_t j = 0; j < p; ++j) q = q + Y[i * p + j] * Y[i * p + j];
                           }
                           DD hq = q * 0.5;  // Q / 2, what the density takes (exact)
                           if (shift != 0) {
                               hq = dd::ldexp(hq, 2 * shift);
                               if (!std::isfinite(hq.hi)) {  // Q / 2 beyond the doubles: the density is 0
                                   po[m] = as_pdf ? 0.0 : -kInf;
                                   continue;
                               }
                           }
                           const DD l = cst[b] - hq;
                           po[m] = as_pdf ? num::exp_dd(l) : l.hi + l.lo;
                       }
                   });
    }
    return wrap(std::move(out));
}

// Samples (N, n, p): M + L_U Z L_V^T, Z standard normal; row i stream element pos + i.
nb::object matnorm_sample(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, std::int64_t N, nb::handle mean_obj,
                          nb::handle uh_obj, nb::handle vh_obj, nb::handle ok_obj) {
    const NDArray &mean = f64(mean_obj, 3, "mean"), &Uh = f64(uh_obj, 3, "U"), &Vh = f64(vh_obj, 3, "V"),
                  &ok = f64(ok_obj, 1, "ok");
    const std::int64_t B = mean.shape[0], n = mean.shape[1], p = mean.shape[2];
    const std::int64_t s3[] = {N, n, p};
    NDArray out = new_array(DType::Float64, s3);
    double* po = reinterpret_cast<double*>(out.data);
    const random::Key key{k0, k1};
    {
        nb::gil_scoped_release release;
        for_pieces(N, std::max<std::int64_t>(1, 1024 / std::max<std::int64_t>(1, n * p)), N * n * p * (n + p + 8),
                   [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
                       random::Stream s(key);
                       std::vector<double> Z(static_cast<std::size_t>(n * p)), T(static_cast<std::size_t>(n * p));
                       for (std::int64_t i = i0; i < i1; ++i) {
                           const std::int64_t b = i % B;
                           double* o = po + i * n * p;
                           if (el(ok, b) == 0.0) {
                               for (std::int64_t k = 0; k < n * p; ++k) o[k] = kNaN;
                               continue;
                           }
                           stats::smp::Words w(s, pos + static_cast<std::uint64_t>(i));
                           for (std::int64_t k = 0; k < n * p; ++k) Z[k] = stats::smp::std_normal(w);
                           for (std::int64_t r = 0; r < n; ++r)  // T = L_U Z
                               for (std::int64_t c = 0; c < p; ++c) {
                                   double t = 0.0;
                                   for (std::int64_t k = 0; k <= r; ++k) t += el(Uh, b, r, k) * Z[k * p + c];
                                   T[r * p + c] = t;
                               }
                           for (std::int64_t r = 0; r < n; ++r)  // o = M + T L_V^T
                               for (std::int64_t c = 0; c < p; ++c) {
                                   double t = 0.0;
                                   for (std::int64_t k = 0; k <= c; ++k) t += T[r * p + k] * el(Vh, b, c, k);
                                   o[r * p + c] = el(mean, b, r, c) + t;
                               }
                       }
                   });
    }
    return wrap(std::move(out));
}

}  // namespace

void bind_stats_matrix(nb::module_& m) {
    m.def("wishart_logpdf", &wishart_logpdf);
    m.def("wishart_entropy", &wishart_entropy);
    m.def("wishart_sample", &wishart_sample);
    m.def("matnorm_logpdf", &matnorm_logpdf, nb::arg("x"), nb::arg("batch"), nb::arg("mean"), nb::arg("uh"),
          nb::arg("ul"), nb::arg("vh"), nb::arg("vl"), nb::arg("ldu_h"), nb::arg("ldu_l"), nb::arg("ldv_h"),
          nb::arg("ldv_l"), nb::arg("ok"), nb::arg("fast_u").none(), nb::arg("fast_v").none(), nb::arg("as_pdf"));
    m.def("matnorm_sample", &matnorm_sample);
}

}  // namespace sd
