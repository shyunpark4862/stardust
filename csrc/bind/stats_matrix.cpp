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
    std::vector<std::vector<DD>> mats(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) {
        if (el(ok, b) == 0.0) continue;
        const double n = el(dfa, b);
        const DD ld{el(ldh, b), el(ldl, b)};
        const DD base = -(dd::kLn2 * (0.5 * n * pd)) - la::log_mvgamma(p, 0.5 * n);
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
                           for (std::int64_t i = 0; i < p && !nan; ++i)
                               for (std::int64_t j = 0; j <= i && !nan; ++j) nan = std::isnan(el(x, m, i, j));
                           if (nan) {
                               po[m] = kNaN;
                               continue;
                           }
                           DD ldx;
                           const bool pdx =
                               la::cholesky(p, [&](std::int64_t i, std::int64_t j) { return el(x, m, i, j); }, LX, ldx);
                           if (!pdx) {  // off the support
                               po[m] = as_pdf ? 0.0 : -kInf;
                               continue;
                           }
                           const std::vector<DD>& A = mats[b];
                           DD tr{0.0, 0.0};
                           if (!inverse) {  // tr(V^-1 X) over X's lower triangle
                               for (std::int64_t i = 0; i < p; ++i) {
                                   tr = tr + A[i * p + i] * el(x, m, i, i);
                                   for (std::int64_t j = 0; j < i; ++j) tr = tr + A[i * p + j] * (2.0 * el(x, m, i, j));
                               }
                           } else {  // |L_X^-1 L_Psi|_F^2, column by column
                               for (std::int64_t c = 0; c < p; ++c) {
                                   for (std::int64_t i = 0; i < p; ++i) col[i] = i >= c ? A[i * p + c] : DD{0.0, 0.0};
                                   la::solve_lower(LX, p, col.data());
                                   for (std::int64_t i = 0; i < p; ++i) tr = tr + col[i] * col[i];
                               }
                           }
                           const double coef = inverse ? -0.5 * (n + pd + 1.0) : 0.5 * (n - pd - 1.0);
                           const DD l = cst[b] + ldx * coef - tr * 0.5;
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
                          nb::handle ldv_h, nb::handle ldv_l, nb::handle ok_obj, bool as_pdf) {
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
    {
        nb::gil_scoped_release release;
        for_pieces(M, std::max<std::int64_t>(1, 2048 / std::max<std::int64_t>(1, n * p * (n + p))),
                   M * n * p * (n + p), [&](std::int64_t, std::int64_t m0, std::int64_t m1) {
                       std::vector<DD> Y(static_cast<std::size_t>(n * p));
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
                           for (std::int64_t i = 0; i < n; ++i)
                               for (std::int64_t j = 0; j < p; ++j) Y[i * p + j] = dd::two_sum(el(x, m, i, j), -el(mean, b, i, j));
                           for (std::int64_t j = 0; j < p; ++j) la::solve_lower(LU[b], n, Y.data() + j, p);  // L_U^-1 (X - M)
                           DD q{0.0, 0.0};
                           for (std::int64_t i = 0; i < n; ++i) {  // rows of Y L_V^-T: L_V z = row
                               la::solve_lower(LV[b], p, Y.data() + i * p);
                               for (std::int64_t j = 0; j < p; ++j) q = q + Y[i * p + j] * Y[i * p + j];
                           }
                           const DD l = cst[b] - q * 0.5;
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
    m.def("matnorm_logpdf", &matnorm_logpdf);
    m.def("matnorm_sample", &matnorm_sample);
}

}  // namespace sd
