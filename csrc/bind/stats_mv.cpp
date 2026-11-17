// The multivariate distributions' kernels (T9, docs/design/stats.md §6; decisions T29):
// "mv_*", "multinomial_*", "dirichlet_*" in stardust._core.stats. The classes in
// stardust.stats._multivariate flatten the batch: parameters arrive as (B, ...) arrays and
// points as (M, d) rows with a batch index each; the cdf is bind/stats_mvcdf.cpp.
//
// Accuracy (T29: 1e-14 for kappa(cov) <= 1e6, d <= 100): the Cholesky factor, the triangular
// solve and the quadratic form are double-double. A factor rounded to double already moves
// the quadratic form by ~kappa eps (the factor of A + E, |E| ~ d eps |A|); in double-double
// that is ~kappa 1e-32. The log densities are double-double sums of their pieces; the
// densities exp them once (exp(logpdf) of a rounded logpdf of size 700 is 1.5e-13 off).
//
// The fast path (stats/dd_linalg.hpp): z = L^-1 (x - mu) as plain dot products with the
// rounded inverse of the double-double factor, with a rigorous bound on the quadratic form's
// error per point. A point whose bound (and the few roundings after it) fits kQuadBudget
// takes the plain result; every other point (ill-conditioned factors, a density that cancels
// against its constant, d past kFastMaxDim) runs the double-double substitution as before.
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
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd {

namespace {

using stats::num::DD;
namespace dd = stats::num::dd;
namespace num = stats::num;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

const NDArray& f64(nb::handle o, int ndim, const char* what) {
    const NDArray& a = require_ndarray(o);
    if (a.dtype != DType::Float64 || a.ndim != ndim)
        throw nb::type_error((std::string("mv: '") + what + "' must be a float64 array of " + std::to_string(ndim) +
                              " dimensions").c_str());
    return a;
}
const NDArray& i64(nb::handle o, const char* what) {
    const NDArray& a = require_ndarray(o);
    if (a.dtype != DType::Int64 || a.ndim != 1)
        throw nb::type_error((std::string("mv: '") + what + "' must be an int64 vector").c_str());
    return a;
}
inline double el(const NDArray& a, std::int64_t i) {
    return *reinterpret_cast<const double*>(reinterpret_cast<const char*>(a.data) + i * a.strides[0]);
}
inline double el(const NDArray& a, std::int64_t i, std::int64_t j) {
    return *reinterpret_cast<const double*>(reinterpret_cast<const char*>(a.data) + i * a.strides[0] + j * a.strides[1]);
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

constexpr auto dd_log1p = [](DD u) { return stats::ddla::log1p(u); };

constexpr double kQuadBudget = stats::ddla::kQuadBudget;  // the quadratic forms' fast path, 40 u
// every other fast path: the engine's budget (stats/engine.hpp), 16 units of 2^-53
constexpr double kFastBudget = 16.0 * 0x1p-53;
constexpr double kU = stats::ddla::kU;
// gamma_m is taken as (m + 1/2) u: fine while m u < 2^-20
constexpr std::int64_t kFastMaxDim = 4096;

// ---- the Cholesky factor in double-double ---------------------------------------------------

// cov (B, d, d) -> (L_hi, L_lo (B, d, d), logdet_hi, logdet_lo (B,), ok (B,) as 0/1 float64,
// inv_hi, inv_lo (B, d)): the lower factor of each matrix (its lower triangle read; the upper
// is not checked), log det = 2 sum log L_jj and the reciprocals 1 / L_jj (the solves multiply).
// ok = 0 where a pivot is not positive or not finite.
nb::tuple chol(nb::handle cov_obj) {
    const NDArray& c = f64(cov_obj, 3, "cov");
    const std::int64_t B = c.shape[0], d = c.shape[1];
    if (c.shape[2] != d) throw nb::value_error("mv: 'cov' must be square");
    const std::int64_t s3[] = {B, d, d};
    NDArray Lh = new_array(DType::Float64, s3, true), Ll = new_array(DType::Float64, s3, true);
    NDArray ldh = out_vec(B), ldl = out_vec(B), ok = out_vec(B);
    const std::int64_t s2[] = {B, d};
    NDArray ih = new_array(DType::Float64, s2), il = new_array(DType::Float64, s2);
    double *pih = reinterpret_cast<double*>(ih.data), *pil = reinterpret_cast<double*>(il.data);
    double *ph = reinterpret_cast<double*>(Lh.data), *pl = reinterpret_cast<double*>(Ll.data);
    double *pdh = reinterpret_cast<double*>(ldh.data), *pdl = reinterpret_cast<double*>(ldl.data),
           *pok = reinterpret_cast<double*>(ok.data);
    {
        nb::gil_scoped_release release;
        for_pieces(B, std::max<std::int64_t>(1, 4096 / std::max<std::int64_t>(1, d * d * d)), B * d * d * d,
                   [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
                       std::vector<DD> L;
                       for (std::int64_t b = b0; b < b1; ++b) {
                           DD logdet;
                           const bool good =
                               stats::ddla::cholesky(d, [&](std::int64_t i, std::int64_t j) { return el(c, b, i, j); }, L, logdet);
                           pok[b] = good ? 1.0 : 0.0;
                           pdh[b] = good ? logdet.hi : kNaN;
                           pdl[b] = good ? logdet.lo : kNaN;
                           for (std::int64_t i = 0; i < d; ++i) {
                               for (std::int64_t j = 0; j <= i; ++j) {
                                   ph[(b * d + i) * d + j] = good ? L[i * d + j].hi : kNaN;
                                   pl[(b * d + i) * d + j] = good ? L[i * d + j].lo : kNaN;
                               }
                               const DD inv = good ? DD{1.0, 0.0} / L[i * d + i] : DD{kNaN, 0.0};
                               pih[b * d + i] = inv.hi, pil[b * d + i] = inv.lo;
                           }
                       }
                   });
    }
    return nb::make_tuple(wrap(std::move(Lh)), wrap(std::move(Ll)), wrap(std::move(ldh)), wrap(std::move(ldl)),
                          wrap(std::move(ok)), wrap(std::move(ih)), wrap(std::move(il)));
}

// L hi/lo (B, d, d), ok (B,) -> (B, fast_size(d)): the fast path's inverse factor and bound
// constants per batch element (stats/dd_linalg.hpp), NaN where there is none (the fast path
// then fails its own test and every point takes the precise one).
nb::object fast_prep(nb::handle lh_obj, nb::handle ll_obj, nb::handle ok_obj) {
    const NDArray &Lh = f64(lh_obj, 3, "L"), &Ll = f64(ll_obj, 3, "L"), &ok = f64(ok_obj, 1, "ok");
    const std::int64_t B = Lh.shape[0], d = Lh.shape[1];
    const std::int64_t fs = stats::ddla::fast_size(d);
    const std::int64_t s2[] = {B, fs};
    NDArray out = new_array(DType::Float64, s2);
    double* po = reinterpret_cast<double*>(out.data);
    {
        nb::gil_scoped_release release;
        for_pieces(B, std::max<std::int64_t>(1, 4096 / std::max<std::int64_t>(1, d * d * d)), B * d * d * d,
                   [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
                       std::vector<DD> L(static_cast<std::size_t>(d * d));
                       for (std::int64_t b = b0; b < b1; ++b) {
                           bool good = el(ok, b) != 0.0 && d <= kFastMaxDim;
                           if (good) {
                               for (std::int64_t i = 0; i < d; ++i)
                                   for (std::int64_t j = 0; j <= i; ++j) L[i * d + j] = DD{el(Lh, b, i, j), el(Ll, b, i, j)};
                               good = stats::ddla::fast_prep(L, d, po + b * fs);
                           }
                           if (!good) std::fill(po + b * fs, po + (b + 1) * fs, kNaN);
                       }
                   });
    }
    return wrap(std::move(out));
}

// Q = |L^-1 (x - mu)|^2 by forward substitution, z and Q as double-doubles; the dot products
// as Ogita, Rump and Oishi's Dot2 (an exact product by fma, the sum's errors gathered in one
// double: as accurate as double-double arithmetic for these sums, at a third of its cost) and
// the pivots multiplied by their reciprocals. L is the kernel's own contiguous (B, d, d).
DD quad_form(const double* lh, const double* ll, const double* ih, const double* il, const NDArray& x,
             std::int64_t m, const NDArray& mu, std::int64_t b, std::int64_t d, std::vector<double>& zh,
             std::vector<double>& zl) {
    double qh = 0.0, ql = 0.0;
    for (std::int64_t i = 0; i < d; ++i) {
        const DD r = dd::two_sum(el(x, m, i), -el(mu, b, i));
        double sh = r.hi, sl = r.lo;
        const double* rowh = lh + i * d;
        const double* rowl = ll + i * d;
        for (std::int64_t k = 0; k < i; ++k) {
            const double ph = rowh[k] * zh[k];
            const double pe = std::fma(rowh[k], zh[k], -ph);  // exact: rowh zh = ph + pe
            const double t = sh - ph, bb = t - sh;
            sl += ((sh - (t - bb)) + (-ph - bb)) - pe - (rowh[k] * zl[k] + rowl[k] * zh[k]);
            sh = t;
        }
        const DD zi = dd::quick_two_sum(sh, sl) * DD{ih[i], il[i]};
        zh[i] = zi.hi, zl[i] = zi.lo;
        const double ph = zi.hi * zi.hi, pe = std::fma(zi.hi, zi.hi, -ph);
        const double t = qh + ph, bb = t - qh;
        ql += ((qh - (t - bb)) + (ph - bb)) + pe + 2.0 * zi.hi * zi.lo;
        qh = t;
    }
    return dd::quick_two_sum(qh, ql);
}

// The same where Q, or a step toward it, overflows (|x - mu| or 1 / L_ii beyond ~1e154):
// z 2^-E and Q 4^-E, the residual scaled exactly by 2^-E before the substitution. E is the
// exponent of the largest |x_i|, |mu_i| plus that of the largest reciprocal pivot, so the
// scaled z are of order one (times the growth of the substitution).
DD quad_form_scaled(const double* lh, const double* ll, const double* ih, const double* il, const NDArray& x,
                    std::int64_t m, const NDArray& mu, std::int64_t b, std::int64_t d, std::vector<double>& zh,
                    std::vector<double>& zl, int& E) {
    double big = 0.0, pinv = 0.0;
    for (std::int64_t i = 0; i < d; ++i) {
        big = std::max({big, std::fabs(el(x, m, i)), std::fabs(el(mu, b, i))});
        pinv = std::max(pinv, ih[i]);
    }
    const int er = big > 0.0 ? std::ilogb(big) + 2 : 0, ei = pinv > 0.0 ? std::ilogb(pinv) : 0;
    E = er + ei;
    double qh = 0.0, ql = 0.0;
    for (std::int64_t i = 0; i < d; ++i) {
        const DD r = dd::two_sum(std::ldexp(el(x, m, i), -er), -std::ldexp(el(mu, b, i), -er));
        double sh = std::ldexp(r.hi, -ei), sl = std::ldexp(r.lo, -ei);
        const double* rowh = lh + i * d;
        const double* rowl = ll + i * d;
        for (std::int64_t k = 0; k < i; ++k) {
            const double ph = rowh[k] * zh[k];
            const double pe = std::fma(rowh[k], zh[k], -ph);
            const double t = sh - ph, bb = t - sh;
            sl += ((sh - (t - bb)) + (-ph - bb)) - pe - (rowh[k] * zl[k] + rowl[k] * zh[k]);
            sh = t;
        }
        const DD zi = dd::quick_two_sum(sh, sl) * DD{ih[i], il[i]};
        zh[i] = zi.hi, zl[i] = zi.lo;
        const double ph = zi.hi * zi.hi, pe = std::fma(zi.hi, zi.hi, -ph);
        const double t = qh + ph, bb = t - qh;
        ql += ((qh - (t - bb)) + (ph - bb)) + pe + 2.0 * zi.hi * zi.lo;
        qh = t;
    }
    return dd::quick_two_sum(qh, ql);
}

// The log density from Q = q 4^E where Q, or Q / df, is beyond the doubles (NaN where the
// scaled q is not finite and positive either). Normal: c - Q / 2, -inf only where Q / 2
// itself overflows. t: log1p(Q / df) = log q + 2 E log 2 - log df + log1p(df / Q), finite
// for every finite x (the t's log density falls like log |x|, not |x|^2: at x = 1e308 it
// is ~ -(df + d) 709, not -inf).
double log_density_scaled(DD cst, DD q, int E, double df, double hd) {
    if (!(q.hi > 0.0) || !std::isfinite(q.hi) || !std::isfinite(q.lo)) return kNaN;
    if (std::isinf(df)) {
        const double half = std::ldexp(q.hi, 2 * E - 1);
        if (!std::isfinite(half)) return -kInf;
        const DD l = cst - DD{half, std::ldexp(q.lo, 2 * E - 1)};
        return l.hi + l.lo;
    }
    const DD ly = dd::log(q) + dd::kLn2 * (2.0 * E) - dd::log(df);  // log(Q / df)
    const DD lp = ly.hi > 50.0 ? ly : ly + std::log1p(std::exp(-ly.hi));
    const DD l = cst - lp * (0.5 * df + hd);
    return l.hi + l.lo;
}

// log Gamma(a + h) - log Gamma(a) for a >= 12, h > 0, without the cancellation of two
// values of size a log a: (a - 1/2) log1p(h/a) + h log(a + h) - h + S(a + h) - S(a), S the
// Stirling series.
DD lgamma_diff(double a, double h) {
    const DD l1 = dd_log1p(num::div(DD{h, 0.0}, a));
    const DD lah = dd::log(dd::two_sum(a, h));
    const double ds = num::stirling(a + h).s - num::stirling(a).s;
    return l1 * (a - 0.5) + lah * h - h + ds;
}

// The log density of the normal (df = inf) or t (df finite > 0) at the points:
// x (M, d), batch (M,), mu (B, d), L_hi/L_lo (B, d, d), logdet hi/lo (B,), ok (B,), df (B,).
// as_pdf: exp of it (once, from the double-double).
nb::object logpdf(nb::handle x_obj, nb::handle batch_obj, nb::handle mu_obj, nb::handle lh_obj, nb::handle ll_obj,
                  nb::handle ldh_obj, nb::handle ldl_obj, nb::handle ok_obj, nb::handle df_obj, nb::handle ih_obj,
                  nb::handle il_obj, nb::handle fast_obj, bool as_pdf) {
    const NDArray &x = f64(x_obj, 2, "x"), &mu = f64(mu_obj, 2, "mu"), &Lh = f64(lh_obj, 3, "L"),
                  &Ll = f64(ll_obj, 3, "L"), &ldh = f64(ldh_obj, 1, "logdet"), &ldl = f64(ldl_obj, 1, "logdet"),
                  &ok = f64(ok_obj, 1, "ok"), &dfa = f64(df_obj, 1, "df"), &ih = f64(ih_obj, 2, "inv"),
                  &il = f64(il_obj, 2, "inv");
    const NDArray& bi = i64(batch_obj, "batch");
    const std::int64_t M = x.shape[0], d = x.shape[1];
    NDArray out = out_vec(M);
    double* po = reinterpret_cast<double*>(out.data);
    const double hd = 0.5 * static_cast<double>(d);
    const std::int64_t B = mu.shape[0];
    // the fast path's factors (B, fast_size(d)), contiguous, or none
    const double* pf = nullptr;
    const std::int64_t fs = stats::ddla::fast_size(d), dp = stats::ddla::padded(d);
    if (!fast_obj.is_none() && d > 0) {
        const NDArray& fa = f64(fast_obj, 2, "fast");
        if (fa.shape[0] != B || fa.shape[1] != fs || fa.strides[1] != 8 || fa.strides[0] != 8 * fs)
            throw nb::value_error("mv: 'fast' does not fit the factor");
        pf = reinterpret_cast<const double*>(fa.data);
    }
    // the per-batch constants: normal -logdet/2 - d log sqrt(2 pi); t log Gamma((df+d)/2) -
    // log Gamma(df/2) - (d/2) log(df pi) - logdet/2
    std::vector<DD> cst(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) {
        const double df = el(dfa, b);
        if (el(ok, b) == 0.0 || std::isnan(df)) continue;
        const DD half_logdet = DD{el(ldh, b), el(ldl, b)} * 0.5;
        if (std::isinf(df)) {
            cst[b] = -half_logdet - num::kLogSqrt2Pi * static_cast<double>(d);
        } else {
            const double a = 0.5 * df;
            const DD lg = a >= 12.0 ? lgamma_diff(a, hd) : num::log_gamma(dd::two_sum(a, hd)) - num::log_gamma(DD{a, 0.0});
            cst[b] = lg - (num::log_dd(df) + num::kLogPi) * hd - half_logdet;
        }
    }
    const double* plh = reinterpret_cast<const double*>(Lh.data);
    const double* pll = reinterpret_cast<const double*>(Ll.data);
    const double* pih = reinterpret_cast<const double*>(ih.data);
    const double* pil = reinterpret_cast<const double*>(il.data);
    {
        nb::gil_scoped_release release;
        // pieces of a few us at least (a point of the fast path is ~0.1 ns per entry of the
        // factor, and a piece allocates its buffers)
        for_pieces(M, std::clamp<std::int64_t>(16384 / std::max<std::int64_t>(1, d * d), 16, 2048), M * d * d,
                   [&](std::int64_t, std::int64_t m0, std::int64_t m1) {
                       std::vector<double> zh(static_cast<std::size_t>(d)), zl(static_cast<std::size_t>(d));
                       // the fast path's r, z and its bounds (zero past d), and the batch element's mu
                       std::vector<double> buf(static_cast<std::size_t>(3 * dp + d), 0.0);
                       double *r = buf.data(), *z = r + dp, *e = z + dp, *mub = e + dp;
                       const std::int64_t xs1 = x.strides[1];
                       std::int64_t last = -1;
                       double df = kNaN;
                       bool bad = true;  // the batch element's parameters are invalid
                       for (std::int64_t m = m0; m < m1; ++m) {
                           const std::int64_t b = eli(bi, m);
                           if (b != last) {
                               last = b, df = el(dfa, b);
                               bad = el(ok, b) == 0.0 || std::isnan(df);
                               for (std::int64_t i = 0; i < d; ++i) mub[i] = el(mu, b, i);
                           }
                           if (bad) {
                               po[m] = kNaN;
                               continue;
                           }
                           // r = x - mu (rounded: the fast path's) and whether x is all finite, one pass
                           const char* xb = reinterpret_cast<const char*>(x.data) + m * x.strides[0];
                           bool finite = true;
                           for (std::int64_t i = 0; i < d; ++i) {
                               const double xi = *reinterpret_cast<const double*>(xb + i * xs1);
                               finite = finite & (std::fabs(xi) < kInf);
                               r[i] = xi - mub[i];
                           }
                           if (!finite) {  // a NaN: NaN; else an infinite component: density 0
                               bool nan = false;
                               for (std::int64_t i = 0; i < d; ++i) nan = nan || std::isnan(el(x, m, i));
                               po[m] = nan ? kNaN : (as_pdf ? 0.0 : -kInf);
                               continue;
                           }
                           if (pf) {
                               // Q to within err (ddla::fast_norm2), then the same formula in plain
                               // double. c = cst[b] is exact to the double-double; l = c.hi - t (its
                               // error exactly, by TwoSum) + c.lo rounds once: u |l| + u^2 |c.hi|.
                               // Normal: t = Q / 2, off by
                               // err / 2. t: t = a log1p(Q / df), a = (df + d) / 2 the same double as
                               // below; log1p moves by at most err / (df + Q - err), the quotient,
                               // log1p (libm, 2 u allowed) and the product by 4 u t in all. The
                               // density is exp(l), relative error the bound on l, so its budget is
                               // absolute: exp(hi) (1 + lo) rounds three times (libm's exp allowed
                               // 2 u): 4 u.
                               // Anything non-finite fails the test.
                               stats::ddla::fast_solve(pf + b * fs, d, r, nullptr, z, e);
                               double err;
                               const double q = stats::ddla::fast_norm2(z, e, d, err);
                               const DD c = cst[b];
                               double t, bound;
                               if (std::isinf(df)) {
                                   t = 0.5 * q;
                                   bound = 0.5 * err;
                               } else {
                                   const double a = 0.5 * df + hd;
                                   const double den = df + q - err;
                                   t = a * std::log1p(q / df);
                                   bound = den > 0.0 ? a * err / den + 4.5 * kU * t : kInf;
                               }
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
                           const DD q = quad_form(plh + b * d * d, pll + b * d * d, pih + b * d, pil + b * d, x, m, mu,
                                                  b, d, zh, zl);
                           if (!(std::isfinite(q.hi) && std::isfinite(q.lo) && std::isfinite(q.hi / df))) [[unlikely]] {
                               // Q or Q / df overflowed: rescaled, in logs
                               int E = 0;
                               const DD qs = std::isfinite(q.hi) && std::isfinite(q.lo)
                                                 ? q
                                                 : quad_form_scaled(plh + b * d * d, pll + b * d * d, pih + b * d,
                                                                    pil + b * d, x, m, mu, b, d, zh, zl, E);
                               const double l = log_density_scaled(cst[b], qs, E, df, hd);
                               po[m] = as_pdf ? std::exp(l) : l;
                               continue;
                           }
                           const DD l = std::isinf(df) ? cst[b] - q * 0.5
                                                       : cst[b] - dd_log1p(q / df) * (0.5 * df + hd);  // - (df+d)/2 log1p(Q/df)
                           po[m] = as_pdf ? num::exp_dd(l) : l.hi + l.lo;
                       }
                   });
    }
    return wrap(std::move(out));
}

// The entropy per batch: normal (d/2)(1 + log 2 pi) + logdet/2; t (df finite)
//     logdet/2 + log Gamma(a) - log Gamma(a + h) + h log(2 pi a) + (a + h)(psi(a + h) - psi(a))
// a = df/2, h = d/2; for a >= 12 as h (1 + log 2 pi) + [(a+h) dpsi - (a+h-1/2) log1p(h/a) - dS]
// (the pieces of size a log a gone; the bracket is O(h^2 / a)).
nb::object entropy(nb::handle ldh_obj, nb::handle ldl_obj, nb::handle ok_obj, nb::handle df_obj, std::int64_t d) {
    const NDArray &ldh = f64(ldh_obj, 1, "logdet"), &ldl = f64(ldl_obj, 1, "logdet"), &ok = f64(ok_obj, 1, "ok"),
                  &dfa = f64(df_obj, 1, "df");
    const std::int64_t B = ldh.shape[0];
    NDArray out = out_vec(B);
    double* po = reinterpret_cast<double*>(out.data);
    const double h = 0.5 * static_cast<double>(d);
    const DD log2pi = num::kLogSqrt2Pi * 2.0;
    for (std::int64_t b = 0; b < B; ++b) {
        const double df = el(dfa, b);
        if (el(ok, b) == 0.0 || std::isnan(df)) {
            po[b] = kNaN;
            continue;
        }
        const DD half = DD{el(ldh, b), el(ldl, b)} * 0.5;
        const DD normal = (log2pi + 1.0) * h + half;
        if (std::isinf(df)) {
            po[b] = normal.hi + normal.lo;
            continue;
        }
        const double a = 0.5 * df;
        DD v;
        if (a >= 12.0) {
            const double dpsi = num::digamma_diff_asym(a, h);
            const double br = (a + h) * dpsi - (a + h - 0.5) * std::log1p(h / a) -
                              (num::stirling(a + h).s - num::stirling(a).s);
            v = normal + br;
        } else {
            const double dpsi = special::Digamma::f(a + h) - special::Digamma::f(a);
            v = half + num::log_gamma(DD{a, 0.0}) - num::log_gamma(dd::two_sum(a, h)) +
                (num::log_dd(2.0 * a) + num::kLogPi) * h + DD{(a + h) * dpsi, 0.0};
        }
        po[b] = v.hi + v.lo;
    }
    return wrap(std::move(out));
}

// Samples: row i of the (N, d) result is stream element pos + i, its batch i mod B (the
// result's shape is sample shape + batch + (d,)): mu + L z, z standard normal (the
// ziggurat, stats/sample_dist.hpp), times sqrt(df / V), V = 2 G(df/2), for the t.
nb::object sample(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, std::int64_t N, nb::handle mu_obj,
                  nb::handle lh_obj, nb::handle ok_obj, nb::handle df_obj) {
    const NDArray &mu = f64(mu_obj, 2, "mu"), &Lh = f64(lh_obj, 3, "L"), &ok = f64(ok_obj, 1, "ok"),
                  &dfa = f64(df_obj, 1, "df");
    const std::int64_t B = mu.shape[0], d = mu.shape[1];
    const std::int64_t s2[] = {N, d};
    NDArray out = new_array(DType::Float64, s2);
    double* po = reinterpret_cast<double*>(out.data);
    const random::Key key{k0, k1};
    {
        nb::gil_scoped_release release;
        // a row draws d normals (and the t's gamma): its extra blocks four rows at a time
        // (one block alone is 12 ns, 3 ns a normal), and L z as eight interleaved sums per row
        // (one chain is latency-bound, 0.7 ns a term: 2/3 of a row at d = 50)
        const bool dense = Lh.strides[2] == 8 && mu.strides[1] == 8;
        const std::uint64_t nb = static_cast<std::uint64_t>(d / 4 + 2);
        for_pieces(N, std::max<std::int64_t>(4, 4096 / std::max<std::int64_t>(1, d * d)), N * d * (d + 8),
                   [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
                       random::Stream s(key);
                       std::vector<double> z(static_cast<std::size_t>(d));
                       std::vector<std::uint64_t> buf;
                       stats::smp::each_words(s, pos, i0, i1, nb, buf, [&](std::int64_t i, stats::smp::Words& w) {
                           const std::int64_t b = i % B;
                           double* o = po + i * d;
                           const double df = el(dfa, b);
                           if (el(ok, b) == 0.0 || std::isnan(df)) {
                               for (std::int64_t j = 0; j < d; ++j) o[j] = kNaN;
                               return;
                           }
                           for (std::int64_t j = 0; j < d; ++j) z[j] = stats::smp::std_normal(w);
                           double scale = 1.0;
                           if (std::isfinite(df)) {
                               const auto g = stats::smp::gamma_prep(0.5 * df);
                               scale = std::exp(0.5 * (std::log(0.5 * df) - stats::smp::log_gamma_draw(w, g)));
                           }
                           if (dense) {
                               const char* L = reinterpret_cast<const char*>(Lh.data) + b * Lh.strides[0];
                               const double* m = reinterpret_cast<const double*>(reinterpret_cast<const char*>(mu.data) + b * mu.strides[0]);
                               for (std::int64_t r = 0; r < d; ++r)
                                   o[r] = m[r] + stats::ddla::dotn(reinterpret_cast<const double*>(L + r * Lh.strides[1]),
                                                                   z.data(), r + 1) * scale;
                               return;
                           }
                           for (std::int64_t r = 0; r < d; ++r) {
                               double acc = 0.0;
                               for (std::int64_t k = 0; k <= r; ++k) acc += el(Lh, b, r, k) * z[k];
                               o[r] = el(mu, b, r) + acc * scale;
                           }
                       });
                   });
    }
    return wrap(std::move(out));
}

// ---- multinomial -------------------------------------------------------------------------

// log k! as a double-double: a table to 4096 (the sum of the logs, ~1e-19 absolute), then
// log_gamma(k + 1).
struct LogFactorials {
    static constexpr int kN = 4096;
    DD v[kN + 1];
    LogFactorials() {
        v[0] = DD{0.0, 0.0};
        for (int k = 1; k <= kN; ++k) v[k] = v[k - 1] + dd::log(static_cast<double>(k));
    }
};
inline const LogFactorials kLogFact;
inline DD log_factorial(double k) {
    // k + 1 as hi + lo: from 2^53 on k + 1.0 is k (log n! came out log (n-1)!: a pmf of 6e7)
    return k <= LogFactorials::kN ? kLogFact.v[static_cast<int>(k)] : num::log_gamma(dd::two_sum(k, 1.0));
}

// log pmf of counts x (M, K) under (n (B,), p (B, K), ok (B,)), batch (M,):
//     log n! - sum log x_i! + sum x_i log p_i - n log(sum p)
// in double-double (the p as given: dividing by their sum would move every log p_i by an
// ULP, n eps in all). x not non-negative integers summing to n: -inf (pmf 0).
nb::object multinomial_logpmf(nb::handle x_obj, nb::handle batch_obj, nb::handle n_obj, nb::handle p_obj,
                              nb::handle ok_obj, bool as_pmf) {
    const NDArray &x = f64(x_obj, 2, "x"), &na = f64(n_obj, 1, "n"), &p = f64(p_obj, 2, "p"), &ok = f64(ok_obj, 1, "ok");
    const NDArray& bi = i64(batch_obj, "batch");
    const std::int64_t M = x.shape[0], K = x.shape[1], B = p.shape[0];
    NDArray out = out_vec(M);
    double* po = reinterpret_cast<double*>(out.data);
    // per batch: log n! - n log(sum p) and the log p_i
    std::vector<DD> cst(static_cast<std::size_t>(B)), lp(static_cast<std::size_t>(B * K));
    // per batch: whether the terms can be large enough to lose the sum (the test after it).
    // For counts that sum to n their sizes are at most |cst| + log n! + n max |log p_i|
    // (sum log x_i! <= log n!): below kPlainTerms the test cannot fail and is skipped.
    constexpr double kPlainTerms = 4e14;  // 1e-13 2^92 = 4.95e14, with room for the bound's rounding
    std::vector<char> large(static_cast<std::size_t>(B), 0);
    for (std::int64_t b = 0; b < B; ++b) {
        if (el(ok, b) == 0.0) continue;
        DD sp{0.0, 0.0};
        double lmax = 0.0;
        for (std::int64_t i = 0; i < K; ++i) {
            sp = sp + el(p, b, i);
            if (el(p, b, i) > 0.0) {
                lp[b * K + i] = num::log_dd(el(p, b, i));
                lmax = std::max(lmax, std::fabs(lp[b * K + i].hi));
            }
        }
        const DD lfn = log_factorial(el(na, b));
        cst[b] = lfn - dd::log(sp) * el(na, b);
        large[b] = !(std::fabs(cst[b].hi) + std::fabs(lfn.hi) + el(na, b) * lmax <= kPlainTerms);
    }
    {
        nb::gil_scoped_release release;
        for_pieces(M, 256, M * K * 40, [&](std::int64_t, std::int64_t m0, std::int64_t m1) {
            for (std::int64_t m = m0; m < m1; ++m) {
                const std::int64_t b = eli(bi, m);
                bool nan = el(ok, b) == 0.0;
                for (std::int64_t i = 0; i < K && !nan; ++i) nan = std::isnan(el(x, m, i));
                if (nan) {
                    po[m] = kNaN;
                    continue;
                }
                const double n = el(na, b);
                DD sx{0.0, 0.0};
                bool in = true;
                for (std::int64_t i = 0; i < K; ++i) {
                    const double v = el(x, m, i);
                    in = in && v >= 0.0 && std::floor(v) == v && std::isfinite(v) && !(v > 0.0 && el(p, b, i) == 0.0);
                    sx = sx + v;
                }
                in = in && sx.hi == n && sx.lo == 0.0;
                if (!in) {
                    po[m] = as_pmf ? 0.0 : -kInf;
                    continue;
                }
                DD l = cst[b];
                for (std::int64_t i = 0; i < K; ++i) {
                    const double v = el(x, m, i);
                    if (v == 0.0) continue;
                    l = l - log_factorial(v) + lp[b * K + i] * v;
                }
                // the terms are each good to ~2^-92 of their size, n log n (measured: 4e-12 of
                // 7e16 at n = 1e15): where that is no longer below 1e-13 max(1, |l|) (n beyond
                // ~1e13 near the mode) the double-double sum does not hold the value: NaN
                if (large[b]) [[unlikely]] {
                    double mag = std::fabs(cst[b].hi);  // the size of the terms that cancel
                    for (std::int64_t i = 0; i < K; ++i) {
                        const double v = el(x, m, i);
                        if (v == 0.0) continue;
                        mag += std::fabs(log_factorial(v).hi) + std::fabs((lp[b * K + i] * v).hi);
                    }
                    if (!(0x1p-92 * mag <= 1e-13 * std::max(1.0, std::fabs(l.hi)))) {
                        po[m] = kNaN;
                        continue;
                    }
                }
                po[m] = as_pmf ? num::exp_dd(l) : l.hi + l.lo;
            }
        });
    }
    return wrap(std::move(out));
}

// Samples by conditional binomials: X_1 ~ Bin(n, p_1), X_i ~ Bin(n - X_1 - ..., p_i / (p_i + ... + p_K)),
// the remaining mass p_i + ... + p_K summed from the end (no cancellation), the last X_K
// the rest. Row i is stream element pos + i, batch i mod B.
nb::object multinomial_sample(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, std::int64_t N, nb::handle n_obj,
                              nb::handle p_obj, nb::handle ok_obj) {
    const NDArray &na = f64(n_obj, 1, "n"), &p = f64(p_obj, 2, "p"), &ok = f64(ok_obj, 1, "ok");
    const std::int64_t B = p.shape[0], K = p.shape[1];
    const std::int64_t s2[] = {N, K};
    NDArray out = new_array(DType::Float64, s2);
    double* po = reinterpret_cast<double*>(out.data);
    const random::Key key{k0, k1};
    {
        nb::gil_scoped_release release;
        const std::uint64_t nb = static_cast<std::uint64_t>(K / 2 + 1);  // a binomial: one or two words
        for_pieces(N, 256, N * K * 60, [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
            random::Stream s(key);
            std::vector<double> tail(static_cast<std::size_t>(K + 1));
            std::vector<std::uint64_t> buf;
            stats::smp::each_words(s, pos, i0, i1, nb, buf, [&](std::int64_t i, stats::smp::Words& w) {
                const std::int64_t b = i % B;
                double* o = po + i * K;
                if (el(ok, b) == 0.0) {
                    for (std::int64_t j = 0; j < K; ++j) o[j] = kNaN;
                    return;
                }
                tail[K] = 0.0;
                for (std::int64_t j = K - 1; j >= 0; --j) tail[j] = tail[j + 1] + el(p, b, j);
                double rem = el(na, b);
                for (std::int64_t j = 0; j < K; ++j) {
                    if (j == K - 1 || rem == 0.0) {
                        o[j] = j == K - 1 ? rem : 0.0;
                        if (j < K - 1) continue;
                        break;
                    }
                    const double pj = el(p, b, j);
                    const double q = tail[j] > 0.0 ? pj / tail[j] : 0.0;  // P(category j | j or later)
                    double xj;
                    if (q <= 0.0) xj = 0.0;
                    else if (q >= 1.0) xj = rem;
                    else xj = stats::smp::binom_draw(w, stats::smp::binom_prep(rem, q, dd::two_sum(1.0, -q)));
                    o[j] = xj;
                    rem -= xj;
                }
            });
        });
    }
    return wrap(std::move(out));
}

// The entropy, -log n! - n sum p log p + sum_i E[log X_i!], X_i ~ Bin(n, p_i): the expectations
// summed from the mode outwards in double-double (each pmf by exp_full of a double-double log
// pmf) to 2^-110 of the sum; log n! and the expectations are each of size n log n and cancel
// to O(K log n).
constexpr double kEntropySd = 0x1p17;                         // the sum of the categories' standard deviations
constexpr std::int64_t kEntropySteps = std::int64_t{1} << 23;  // terms on one side of one mode (64 kEntropySd)
nb::object multinomial_entropy(nb::handle n_obj, nb::handle p_obj, nb::handle ok_obj) {
    const NDArray &na = f64(n_obj, 1, "n"), &p = f64(p_obj, 2, "p"), &ok = f64(ok_obj, 1, "ok");
    const std::int64_t B = p.shape[0], K = p.shape[1];
    NDArray out = out_vec(B);
    double* po = reinterpret_cast<double*>(out.data);
    {
        nb::gil_scoped_release release;
        for_pieces(B, 1, B * K * 1000, [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
            for (std::int64_t b = b0; b < b1; ++b) {
                if (el(ok, b) == 0.0) {
                    po[b] = kNaN;
                    continue;
                }
                const double n = el(na, b);
                DD sp{0.0, 0.0};
                for (std::int64_t i = 0; i < K; ++i) sp = sp + el(p, b, i);
                // The sums below take ~25 standard deviations of terms per category, each a
                // few double-double log Gammas: the work is bounded by the sum of the
                // standard deviations (kEntropySd: ~3e6 terms, a second), and n is kept to
                // the integers a double counts (n - k exact). Beyond either: NaN.
                double work = 0.0;
                bool proper = false;  // a category with 0 < p < 1
                for (std::int64_t i = 0; i < K; ++i) {
                    const double pd = el(p, b, i) / sp.hi;
                    if (pd > 0.0 && pd < 1.0) proper = true, work += std::sqrt(n * pd * (1.0 - pd));
                }
                if (proper && !(work <= kEntropySd && n <= 0x1p53)) {
                    po[b] = kNaN;
                    continue;
                }
                bool capped = false;
                DD h = -num::log_gamma(dd::two_sum(n, 1.0));
                for (std::int64_t i = 0; i < K; ++i) {
                    const DD pi = DD{el(p, b, i), 0.0} / sp;  // p_i / sum p, double-double
                    if (pi.hi <= 0.0) continue;
                    h = h - dd::log(pi) * pi * n;
                    if (pi.hi >= 1.0) {  // the point mass at n
                        h = h + num::log_gamma(dd::two_sum(n, 1.0));
                        continue;
                    }
                    const double pd = pi.hi + pi.lo;
                    const DD q = dd::two_sum(1.0, -pi.hi) - pi.lo;
                    const double mode = std::floor((n + 1.0) * pd);
                    // the log pmf from double-double log Gammas (~1e-19 absolute): the binomial's
                    // own log pmf rounds a factor's log to a double, 1e-16 of every pmf, which
                    // the expectation of size n log n turns into 4e-11 at n = 1e5
                    const DD lgn = num::log_gamma(DD{n + 1.0, 0.0}), lpi = dd::log(pi), lq = dd::log(q);
                    auto term = [&](double k) {
                        const DD lk = num::log_gamma(DD{k + 1.0, 0.0});
                        const DD lp = lgn - lk - num::log_gamma(DD{n - k + 1.0, 0.0}) + lpi * k + lq * (n - k);
                        return num::exp_full(lp) * lk;
                    };
                    // from the mode outwards until the rest is negligible: past a term t with
                    // ratio r < 1 to the one before, the rest is about t r / (1 - r)
                    auto done = [&](const DD& t, const DD& prev, const DD& e) {
                        const double r = prev.hi != 0.0 ? std::fabs(t.hi / prev.hi) : 0.0;
                        return r < 1.0 && std::fabs(t.hi) * r <= (1.0 - r) * 0x1p-110 * std::fabs(e.hi);
                    };
                    const double k0 = std::min(mode, n);
                    const DD t0 = term(k0);
                    DD e = t0, prev = t0;
                    std::int64_t steps = 0;
                    for (double k = k0 + 1.0; k <= n; k += 1.0) {
                        if (++steps > kEntropySteps) {
                            capped = true;
                            break;
                        }
                        const DD t = term(k);
                        e = e + t;
                        if (k > mode + 1.0 && done(t, prev, e)) break;
                        prev = t;
                    }
                    prev = t0;
                    steps = 0;
                    for (double k = k0 - 1.0; k >= 0.0; k -= 1.0) {
                        if (++steps > kEntropySteps) {
                            capped = true;
                            break;
                        }
                        const DD t = term(k);
                        e = e + t;
                        if (k < mode - 1.0 && done(t, prev, e)) break;
                        prev = t;
                    }
                    h = h + e;
                }
                po[b] = capped ? kNaN : h.hi + h.lo;
            }
        });
    }
    return wrap(std::move(out));
}

// ---- Dirichlet --------------------------------------------------------------------------------

// log pdf of x (M, K) under alpha (B, K): log Gamma(a0) - sum log Gamma(a_i) + sum (a_i - 1) log x_i,
// in double-double. Off the simplex (a negative component, or the sum off 1 by more than
// sqrt(eps) K): -inf. x_i = 0: the term's limit (a_i < 1: +inf, = 1: 0, > 1: -inf).
nb::object dirichlet_logpdf(nb::handle x_obj, nb::handle batch_obj, nb::handle a_obj, nb::handle ok_obj, bool as_pdf) {
    const NDArray &x = f64(x_obj, 2, "x"), &al = f64(a_obj, 2, "alpha"), &ok = f64(ok_obj, 1, "ok");
    const NDArray& bi = i64(batch_obj, "batch");
    const std::int64_t M = x.shape[0], K = x.shape[1];
    NDArray out = out_vec(M);
    double* po = reinterpret_cast<double*>(out.data);
    const double tol = std::sqrt(std::numeric_limits<double>::epsilon()) * static_cast<double>(K);
    const std::int64_t B = al.shape[0];
    std::vector<DD> cst(static_cast<std::size_t>(B));  // log Gamma(a0) - sum log Gamma(a_i), per batch
    for (std::int64_t b = 0; b < B; ++b) {
        if (el(ok, b) == 0.0) continue;
        DD a0{0.0, 0.0}, l{0.0, 0.0};
        for (std::int64_t i = 0; i < K; ++i) a0 = a0 + el(al, b, i), l = l - num::log_gamma(DD{el(al, b, i), 0.0});
        cst[b] = l + num::log_gamma(a0);
    }
    {
        nb::gil_scoped_release release;
        for_pieces(M, 256, M * K * 40, [&](std::int64_t, std::int64_t m0, std::int64_t m1) {
            for (std::int64_t m = m0; m < m1; ++m) {
                const std::int64_t b = eli(bi, m);
                bool nan = el(ok, b) == 0.0;
                for (std::int64_t i = 0; i < K && !nan; ++i) nan = std::isnan(el(x, m, i));
                if (nan) {
                    po[m] = kNaN;
                    continue;
                }
                DD sx{0.0, 0.0};
                bool in = true;
                for (std::int64_t i = 0; i < K; ++i) {
                    in = in && el(x, m, i) >= 0.0 && std::isfinite(el(x, m, i));
                    sx = sx + el(x, m, i);
                }
                in = in && std::fabs((sx - 1.0).hi) <= tol;
                if (!in) {
                    po[m] = as_pdf ? 0.0 : -kInf;
                    continue;
                }
                // The fast path: the same sum with libm's log. A term t_i = (a_i - 1) log x_i is
                // off by the log's 2 u |t_i| only (the product's error is kept exactly by fma, the
                // sum's by TwoSum, in lo), so l = hi + lo is within 2 u sum |t_i| + u |l| (the last
                // rounding; the roundings inside lo are u^2). The density exp(hi) (1 + lo) rounds
                // three more times (exp allowed 2 u): 4 u, its budget absolute. Past kFastBudget
                // (terms that cancel), or anything non-finite: the double-double sum below.
                double edge = 0.0;  // +-inf from a zero component
                {
                    double sh = cst[b].hi, sl = cst[b].lo, mag = 0.0;
                    for (std::int64_t i = 0; i < K; ++i) {
                        const double a = el(al, b, i), v = el(x, m, i);
                        if (a == 1.0) continue;
                        if (v == 0.0) {
                            edge += a < 1.0 ? kInf : -kInf;
                            continue;
                        }
                        // a - 1 exactly (amd): its rounding times log x_i would reach the sum
                        const DD amd = dd::two_sum(a, -1.0);
                        const double am = amd.hi, lg = std::log(v);
                        const double t = am * lg, te = std::fma(am, lg, -t) + amd.lo * lg;
                        const double hi = sh + t, bb = hi - sh;
                        sl += ((sh - (hi - bb)) + (t - bb)) + te;
                        sh = hi;
                        mag += std::fabs(t);
                    }
                    if (edge != 0.0 || std::isnan(edge)) {
                        po[m] = std::isnan(edge) ? kNaN : (as_pdf ? (edge > 0.0 ? kInf : 0.0) : edge);
                        continue;
                    }
                    const double l = sh + sl;
                    const double bound = kU * (2.0 * mag + kU * static_cast<double>(K + 1) * (mag + std::fabs(cst[b].hi)));
                    if (as_pdf) {
                        if (bound + 4.0 * kU <= kFastBudget && std::fabs(l) <= 700.0) {
                            const DD s2 = dd::quick_two_sum(sh, sl);
                            po[m] = std::exp(s2.hi) * (1.0 + s2.lo);
                            continue;
                        }
                    } else if (bound + kU * std::fabs(l) <= kFastBudget * std::max(1.0, std::fabs(l))) {
                        po[m] = l;
                        continue;
                    }
                }
                DD l = cst[b];
                for (std::int64_t i = 0; i < K; ++i) {
                    const double a = el(al, b, i), v = el(x, m, i);
                    if (a == 1.0) continue;
                    l = l + num::log_dd(v) * dd::two_sum(a, -1.0);  // a - 1 exactly
                }
                po[m] = as_pdf ? num::exp_dd(l) : l.hi + l.lo;
            }
        });
    }
    return wrap(std::move(out));
}

// The entropy log B(a) + (a0 - K) psi(a0) - sum (a_i - 1) psi(a_i), regrouped with the O(1)
// pieces G and Gc of stats/numeric.hpp (as the beta's, K = 2): the terms of size a log a
// cancel inside them, and the sum of the a_i against a0 cancels exactly:
//     sum G(a_i) + Gc(a0) + sum log(a_i)/2 + (K - 1) log(2 pi)/2 + (2 - K) psi(a0) - 3 log(a0)/2.
nb::object dirichlet_entropy(nb::handle a_obj, nb::handle ok_obj) {
    const NDArray &al = f64(a_obj, 2, "alpha"), &ok = f64(ok_obj, 1, "ok");
    const std::int64_t B = al.shape[0], K = al.shape[1];
    NDArray out = out_vec(B);
    double* po = reinterpret_cast<double*>(out.data);
    for (std::int64_t b = 0; b < B; ++b) {
        if (el(ok, b) == 0.0) {
            po[b] = kNaN;
            continue;
        }
        DD a0{0.0, 0.0}, logs{0.0, 0.0};
        double g = 0.0;
        for (std::int64_t i = 0; i < K; ++i) {
            const double a = el(al, b, i);
            a0 = a0 + a;
            logs = logs + num::log_dd(a) * 0.5;
            g += num::entropy_g(a, special::Digamma::f);
        }
        g += num::entropy_gc(a0, special::Digamma::f);
        const double km = static_cast<double>(K);
        const DD v = logs + num::kLogSqrt2Pi * (km - 1.0) - dd::log(a0) * 1.5 +
                     DD{g + (2.0 - km) * special::Digamma::f(a0.hi), 0.0};
        po[b] = v.hi + v.lo;
    }
    return wrap(std::move(out));
}

// Samples: G_i = Gamma(a_i) in logs (tiny shapes underflow), X = G / sum G. Row i is stream
// element pos + i, batch i mod B.
nb::object dirichlet_sample(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, std::int64_t N, nb::handle a_obj,
                            nb::handle ok_obj) {
    const NDArray &al = f64(a_obj, 2, "alpha"), &ok = f64(ok_obj, 1, "ok");
    const std::int64_t B = al.shape[0], K = al.shape[1];
    const std::int64_t s2[] = {N, K};
    NDArray out = new_array(DType::Float64, s2);
    double* po = reinterpret_cast<double*>(out.data);
    const random::Key key{k0, k1};
    {
        nb::gil_scoped_release release;
        // a gamma: two words, three below shape 1
        const std::uint64_t nb = static_cast<std::uint64_t>(3 * K / 4 + 1);
        for_pieces(N, 256, N * K * 40, [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
            random::Stream s(key);
            std::vector<double> lg(static_cast<std::size_t>(K));
            std::vector<std::uint64_t> buf;
            stats::smp::each_words(s, pos, i0, i1, nb, buf, [&](std::int64_t i, stats::smp::Words& w) {
                const std::int64_t b = i % B;
                double* o = po + i * K;
                if (el(ok, b) == 0.0) {
                    for (std::int64_t j = 0; j < K; ++j) o[j] = kNaN;
                    return;
                }
                // shapes that cannot underflow (stats/sample_dist.hpp): G / sum G directly
                bool direct = true;
                for (std::int64_t j = 0; j < K; ++j) direct = direct && el(al, b, j) >= stats::smp::kDirectShape;
                if (direct) {
                    double tot = 0.0;
                    for (std::int64_t j = 0; j < K; ++j) {
                        o[j] = stats::smp::gamma_draw(w, stats::smp::gamma_prep(el(al, b, j)));
                        tot += o[j];
                    }
                    if (tot > 0.0 && std::isfinite(tot)) {
                        for (std::int64_t j = 0; j < K; ++j) o[j] /= tot;
                        return;
                    }
                }
                // in logs, from the element's words again (the direct sum left the range: the
                // same element, so still a function of it alone)
                stats::smp::Words wl(s, pos + static_cast<std::uint64_t>(i));
                stats::smp::Words& ww = direct ? wl : w;
                double top = -kInf;
                for (std::int64_t j = 0; j < K; ++j) {
                    lg[j] = stats::smp::log_gamma_draw(ww, stats::smp::gamma_prep(el(al, b, j)));
                    top = std::max(top, lg[j]);
                }
                double tot = 0.0;
                for (std::int64_t j = 0; j < K; ++j) lg[j] = std::exp(lg[j] - top), tot += lg[j];
                for (std::int64_t j = 0; j < K; ++j) o[j] = lg[j] / tot;
            });
        });
    }
    return wrap(std::move(out));
}

}  // namespace

void bind_stats_mv(nb::module_& m) {
    m.def("mv_chol", &chol);
    m.def("mv_fast_prep", &fast_prep);
    m.def("mv_logpdf", &logpdf, nb::arg("x"), nb::arg("batch"), nb::arg("mu"), nb::arg("lh"), nb::arg("ll"),
          nb::arg("ldh"), nb::arg("ldl"), nb::arg("ok"), nb::arg("df"), nb::arg("ih"), nb::arg("il"),
          nb::arg("fast").none(), nb::arg("as_pdf"));
    m.def("mv_entropy", &entropy);
    m.def("mv_sample", &sample);
    m.def("multinomial_logpmf", &multinomial_logpmf);
    m.def("multinomial_sample", &multinomial_sample);
    m.def("multinomial_entropy", &multinomial_entropy);
    m.def("dirichlet_logpdf", &dirichlet_logpdf);
    m.def("dirichlet_entropy", &dirichlet_entropy);
    m.def("dirichlet_sample", &dirichlet_sample);
}

}  // namespace sd
