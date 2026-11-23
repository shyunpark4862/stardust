// The bandwidth selectors of the kernel density estimate (docs/design/smooth.md §3.6, §4.5; decisions SM19,
// SM127-SM132): Sheather and Jones (R's bw.SJ, its binned pair counts and equations), Botev, Grotowski and
// Kroese's improved Sheather-Jones (ISJ), and the cross-validation criteria (least squares: statsmodels' cv_ls,
// or with nb R's bw.ucv; likelihood: statsmodels' cv_ml) with their searches.
//
// _core.smooth.bw_sj(x, method, nb) -> (h, code): method 0 'ste' (solve the equation), 1 'dpi'; code 0, or 1 fewer
//     than 2 points, 2 a scale of 0, 3 TD is not a positive number ("too sparse"), 4 alpha2 is not finite, 5 no root
//     in the search interval
// _core.smooth.bw_isj(x, n_grid) -> (h, code): code 0, or 1 x of no spread, 5 no root of the fixed point in (0, 0.1]
// _core.smooth.bw_cv(X, kind, nb, href) -> (h (p,), status, value, nfev): kind 1 least squares, 2 likelihood; nb
//     > 0 (1-D least squares) R's binned criterion; href (p,) the centre of the search; status 0, -1 / 1 (the end
//     of the interval), 2 (the evaluation cap), -3 (no finite criterion)
// _core.smooth.bw_cv_value(X, kind, nb, h) -> the criterion at h: least squares int f^2 - 2/n sum f_-i(x_i) (or
//     R's ucv), likelihood sum log f_-i(x_i)
//
// The criteria's per-point terms are added pairwise in the sorted order and each point is one function whatever
// task computes it, so the results have the same bits for any thread count (decision SM3).

#include "smooth/gauss_sum.hpp"
#include "smooth/kde.hpp"
#include "smooth/search.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "fft/plan.hpp"
#include "fft/r2r.hpp"
#include "parallel/pool.hpp"
#include "smooth/smooth.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace sd::smooth {

namespace nb = nanobind;

namespace {

using gauss::Axis;
using kde::kInf;
using kde::kNaN;

constexpr double kLog64 = 4.1588830833596715;  // log 64: the searches span h_ref / 64 .. 64 h_ref
constexpr std::int64_t kGrid = 33;
constexpr double kSqrtPi = 1.7724538509055159;

const NDArray& vec_arg(nb::handle o, const char* what, std::int64_t n = -1) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous 1-D float64 array").c_str());
    if (n >= 0 && a.shape[0] != n) throw nb::value_error(("'" + std::string(what) + "' has the wrong length").c_str());
    return a;
}

const NDArray& mat_arg(nb::handle o, const char* what) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 2 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[1] < 1)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous 2-D float64 array").c_str());
    return a;
}

const double* dptr(const NDArray& a) { return reinterpret_cast<const double*>(a.data); }
double* dout(NDArray& a) { return reinterpret_cast<double*>(a.data); }

// ---- Sheather and Jones: R's bw.SJ (the stats package's R function and its C helpers bw_den, bw_phi4, bw_phi6) ----

// R's pair counts: binned from n > nb / 2 on (bw_pair_cnts: the data's bins trunc(|x| / d) sign(x) shifted to
// start at 1, tabulated into nb bins, then bw_den_binned), else bw_den on the points (bins (int)(x / d)).
struct Counts {
    double d = 0;
    std::vector<double> cnt;
};

Counts pair_counts(const double* x, std::int64_t n, std::int64_t nb) {
    Counts c;
    double xmin = kInf, xmax = -kInf;
    for (std::int64_t i = 0; i < n; ++i) xmin = std::min(xmin, x[i]), xmax = std::max(xmax, x[i]);
    c.d = (xmax - xmin) * 1.01 / static_cast<double>(nb);
    c.cnt.assign(nb, 0.0);
    if (static_cast<double>(n) > static_cast<double>(nb) / 2.0) {
        std::vector<double> xx(n);
        double mn = kInf;
        for (std::int64_t i = 0; i < n; ++i) {
            const double s = x[i] > 0 ? 1.0 : (x[i] < 0 ? -1.0 : 0.0);
            xx[i] = std::trunc(std::fabs(x[i]) / c.d) * s;
            mn = std::min(mn, xx[i]);
        }
        std::vector<double> tab(nb, 0.0);
        for (std::int64_t i = 0; i < n; ++i) {
            const double k = xx[i] - mn + 1.0;  // 1-based; tabulate drops values above nb
            if (k >= 1.0 && k <= static_cast<double>(nb)) tab[static_cast<std::int64_t>(k) - 1] += 1.0;
        }
        for (std::int64_t ii = 0; ii < nb; ++ii) {
            const double w = tab[ii];
            c.cnt[0] += w * (w - 1.0);
            for (std::int64_t jj = 0; jj < ii; ++jj) c.cnt[ii - jj] += w * tab[jj];
        }
        c.cnt[0] *= 0.5;
    } else {
        std::vector<std::int64_t> b(n);
        for (std::int64_t i = 0; i < n; ++i) b[i] = static_cast<std::int64_t>(x[i] / c.d);
        for (std::int64_t i = 1; i < n; ++i)
            for (std::int64_t j = 0; j < i; ++j) c.cnt[std::llabs(b[i] - b[j])] += 1.0;
    }
    return c;
}

constexpr double kDelMax = 1000.0;

// R's bw_phi4, bw_phi6 and bw_ucv, summed in R's order.
double phi4(const Counts& c, std::int64_t n, double h) {
    double sum = 0;
    for (std::size_t i = 0; i < c.cnt.size(); ++i) {
        double delta = static_cast<double>(i) * c.d / h;
        delta *= delta;
        if (delta >= kDelMax) break;
        sum += std::exp(-delta / 2) * (delta * delta - 6 * delta + 3) * c.cnt[i];
    }
    sum = 2 * sum + static_cast<double>(n) * 3;
    const double nn = static_cast<double>(n);
    return sum / (nn * (nn - 1) * std::pow(h, 5.0) * std::sqrt(2 * M_PI));
}

double phi6(const Counts& c, std::int64_t n, double h) {
    double sum = 0;
    for (std::size_t i = 0; i < c.cnt.size(); ++i) {
        double delta = static_cast<double>(i) * c.d / h;
        delta *= delta;
        if (delta >= kDelMax) break;
        sum += std::exp(-delta / 2) * (delta * delta * delta - 15 * delta * delta + 45 * delta - 15) * c.cnt[i];
    }
    sum = 2 * sum - 15 * static_cast<double>(n);
    const double nn = static_cast<double>(n);
    return sum / (nn * (nn - 1) * std::pow(h, 7.0) * std::sqrt(2 * M_PI));
}

double ucv(const Counts& c, std::int64_t n, double h) {
    double sum = 0;
    for (std::size_t i = 0; i < c.cnt.size(); ++i) {
        double delta = static_cast<double>(i) * c.d / h;
        delta *= delta;
        if (delta >= kDelMax) break;
        sum += (std::exp(-delta / 4.) - std::sqrt(8.0) * std::exp(-delta / 2.)) * c.cnt[i];
    }
    const double nn = static_cast<double>(n);
    return (0.5 + sum / nn) / (nn * h * kSqrtPi);
}

// R's sd and IQR (type 7: (1 - h) x[lo] + h x[hi]) of sorted x.
double sd_of(const double* x, std::int64_t n) {
    const double m = gauss::sum(x, n) / static_cast<double>(n);
    const double ss = gauss::sum_of(n, [&](std::int64_t i) { return (x[i] - m) * (x[i] - m); });
    return std::sqrt(ss / static_cast<double>(n - 1));
}

double quantile7(const double* xs, std::int64_t n, double prob) {
    const double index = static_cast<double>(n - 1) * prob;
    const std::int64_t lo = static_cast<std::int64_t>(std::floor(index)),
                       hi = static_cast<std::int64_t>(std::ceil(index));
    const double h = index - static_cast<double>(lo);
    if (!(h > 0) || xs[hi] == xs[lo]) return xs[lo];
    return (1 - h) * xs[lo] + h * xs[hi];
}

std::pair<double, int> sj(const double* x0, std::int64_t n, int method, std::int64_t nbins) {
    if (n < 2) return {kNaN, 1};
    std::vector<double> xs(x0, x0 + n);
    std::sort(xs.begin(), xs.end());
    const Counts c = pair_counts(xs.data(), n, nbins);
    const double nn = static_cast<double>(n);
    const double iqr = quantile7(xs.data(), n, 0.75) - quantile7(xs.data(), n, 0.25);
    const double scale = std::min(sd_of(xs.data(), n), iqr / 1.349);
    if (!(scale > 0) || !(c.d > 0)) return {kNaN, 2};
    const double a = 1.24 * scale * std::pow(nn, -1.0 / 7.0);
    const double b = 1.23 * scale * std::pow(nn, -1.0 / 9.0);
    const double c1 = 1.0 / (2.0 * std::sqrt(M_PI) * nn);
    const double TD = -phi6(c, n, b);
    if (!std::isfinite(TD) || TD <= 0) return {kNaN, 3};
    if (method == 1) {
        const double h = std::pow(c1 / phi4(c, n, std::pow(2.394 / (nn * TD), 1.0 / 7.0)), 1.0 / 5.0);
        return std::isfinite(h) && h > 0 ? std::pair<double, int>{h, 0} : std::pair<double, int>{kNaN, 3};
    }
    const double alph2 = 1.357 * std::pow(phi4(c, n, a) / TD, 1.0 / 7.0);
    if (!std::isfinite(alph2)) return {kNaN, 4};
    const double hmax = 1.144 * scale * std::pow(nn, -1.0 / 5.0);
    // fSD(h) / h in s = log(h / hmax): the same sign; the smallest root of the grid's sign changes (decision SM128)
    const auto g = [&](double s) {
        const double h = hmax * vmath::exp1(s);
        return std::pow(c1 / phi4(c, n, alph2 * std::pow(h, 5.0 / 7.0)), 1.0 / 5.0) / h - 1.0;
    };
    const double step = 2.0 * kLog64 / (kGrid - 1);
    double sp = -kLog64, gp = g(sp);
    for (std::int64_t k = 1; k < kGrid; ++k) {
        const double s = -kLog64 + static_cast<double>(k) * step, gv = g(s);
        if (gp == gp && gv == gv && (gp == 0 || std::signbit(gp) != std::signbit(gv))) {
            const ScalarResult r = brentq(g, sp, s, gp, gv, 1e-13, 0.0, 200);
            return {hmax * vmath::exp1(r.x), 0};
        }
        sp = s, gp = gv;
    }
    return {kNaN, 5};
}

// ---- Botev, Grotowski and Kroese's ISJ (decision SM129) ------------------------------------------------------------

std::pair<double, int> isj(const double* x0, std::int64_t n, std::int64_t N, const fft::Plan& plan) {
    std::vector<double> xs(x0, x0 + n);
    std::sort(xs.begin(), xs.end());
    const double range = xs[n - 1] - xs[0];
    if (!(range > 0) || !std::isfinite(range)) return {kNaN, 1};
    const double lo = xs[0] - range / 2, hi = xs[n - 1] + range / 2;
    const double R = hi - lo, delta = R / static_cast<double>(N - 1);
    std::vector<double> c(N), a(N);
    kde::linbin(xs.data(), nullptr, n, lo, delta, 0, N, c.data());
    for (double& v : c) v /= static_cast<double>(n);
    {
        std::vector<double> x(c), tmp(fft::r2r_tmp_doubles(fft::R2r{}, N)), work(plan.work_doubles());
        fft::r2r_line(plan, fft::R2r{}, N, 1.0, x.data(), a.data(), tmp.data(), work.data());
    }
    std::vector<double> a2(N - 1), I(N - 1);
    for (std::int64_t k = 1; k < N; ++k) {
        const double h = a[k] / 2;
        a2[k - 1] = h * h;
        I[k - 1] = static_cast<double>(k) * static_cast<double>(k);
    }
    double distinct = 1;
    for (std::int64_t i = 1; i < n; ++i)
        if (xs[i] != xs[i - 1]) distinct += 1;
    const double pi2 = M_PI * M_PI;
    const auto fsum = [&](int s, double t) {
        return 2.0 * std::pow(M_PI, 2.0 * s) * gauss::sum_of(N - 1, [&](std::int64_t k) {
                   return std::pow(I[k], s) * a2[k] * vmath::exp1(-I[k] * pi2 * t);
               });
    };
    const auto fixed = [&](double t) {
        double f = fsum(7, t);
        for (int s = 6; s >= 2; --s) {
            double K0 = 1;
            for (int j = 1; j <= 2 * s - 1; j += 2) K0 *= j;
            K0 /= std::sqrt(2 * M_PI);
            const double cst = (1 + std::pow(0.5, s + 0.5)) / 3;
            const double time = std::pow(2 * cst * K0 / distinct / f, 2.0 / (3.0 + 2.0 * s));
            f = fsum(s, time);
        }
        return t - std::pow(2 * distinct * std::sqrt(M_PI) * f, -2.0 / 5.0);
    };
    // the smallest root in (0, 0.1]: t_k = 0.1 2^-k from k = 50 up to 0, the first sign change
    const auto g = [&](double lt) { return fixed(vmath::exp1(lt)); };
    const double l01 = std::log(0.1), ln2 = std::log(2.0);
    double lp = l01 - 50 * ln2, gp = g(lp);
    for (int k = 49; k >= 0; --k) {
        const double lt = l01 - k * ln2, gv = g(lt);
        if (gp == gp && gv == gv && (gp == 0 || std::signbit(gp) != std::signbit(gv))) {
            const ScalarResult r = brentq(g, lp, lt, gp, gv, 1e-13, 0.0, 200);
            return {std::sqrt(vmath::exp1(r.x)) * R, 0};
        }
        lp = lt, gp = gv;
    }
    return {kNaN, 5};
}

// ---- cross validation (decisions SM130, SM131) ---------------------------------------------------------------------

// The exact criteria of unweighted points with a product Gaussian kernel of standard deviations h: per point i,
// A_i = sum_j exp(-D_ij / 4) (j = i included), B_i = sum_(j != i) exp(-D_ij / 2) as e^(-D0_i / 2) times the cut sum,
// and the derivative sums of the 1-D searches. The rows are tasks; their terms add pairwise in point order.
struct CvData {
    std::int64_t n = 0, p = 1;
    const double* X = nullptr;  // sorted (1-D) or row-major
    double T = 0;
};

struct Row {
    double A = 0, dA = 0;        // sum e (A), sum e (D / 2 - 1)
    double b0 = 0, B = 0, dB = 0, mB = 0;  // -D0 / 2, the cut sum, sum e (D - 1), sum e D (shifted by e^(-D0/2))
};

struct CvScratch {
    std::vector<double> D, Dx, e;
};

Row cv_row(const CvData& K, const Axis* ax, std::int64_t i, CvScratch& S) {
    Row r;
    const std::int64_t n = K.n, p = K.p;
    if (p == 1) {
        const double* x = K.X;
        const double xi = x[i];
        std::int64_t lo, hi;
        gauss::range_1d(x, n, xi, ax[0], 2.0 * K.T, i, lo, hi);  // D / 2 <= T: the cut of A
        std::int64_t m = hi - lo;
        S.D.resize(m), S.Dx.resize(m), S.e.resize(m);
        for (std::int64_t k = 0; k < m; ++k) {
            S.D[k] = gauss::excess_1d(x[lo + k], xi, 0.0, ax[0]);  // D_ij
            S.Dx[k] = 0.5 * S.D[k];
        }
        gauss::terms(S.Dx.data(), nullptr, 0.0, m, S.e.data());
        double s2[2];
        gauss::sums_of<2>(m, [&](std::int64_t k, double* v) { v[0] = S.e[k], v[1] = S.e[k] * (S.Dx[k] - 1.0); }, s2);
        r.A = s2[0], r.dA = s2[1];
        const std::int64_t c = gauss::nearest_1d(x, n, xi, i);
        if (c < 0) return r;
        const double zc = gauss::z_of(x[c], xi, ax[0]);
        const double d0 = zc * zc;
        gauss::range_1d(x, n, xi, ax[0], K.T, c, lo, hi);
        m = hi - lo;
        S.D.resize(m), S.Dx.resize(m), S.e.resize(m);
        for (std::int64_t k = 0; k < m; ++k) {
            const std::int64_t j = lo + k;
            S.Dx[k] = j == i ? kInf : gauss::excess_1d(x[j], x[c], zc, ax[0]);
            S.D[k] = d0 + S.Dx[k];
        }
        gauss::terms(S.Dx.data(), nullptr, 0.0, m, S.e.data());
        double s3[3];
        gauss::sums_of<3>(
            m,
            [&](std::int64_t k, double* v) {
                const double e = S.e[k], D = e > 0 ? S.D[k] : 0.0;
                v[0] = e, v[1] = e * (D - 1.0), v[2] = e * D;
            },
            s3
        );
        r.b0 = -0.5 * d0, r.B = s3[0], r.dB = s3[1], r.mB = s3[2];
        return r;
    }
    S.D.resize(n);
    gauss::sqdist_nd(K.X, n, p, K.X + i * p, ax, -1, S.D.data());
    // A: every j with D / 2 <= T
    S.Dx.clear();
    for (std::int64_t j = 0; j < n; ++j)
        if (0.5 * S.D[j] <= K.T) S.Dx.push_back(0.5 * S.D[j]);
    std::int64_t m = static_cast<std::int64_t>(S.Dx.size());
    S.e.resize(m);
    gauss::terms(S.Dx.data(), nullptr, 0.0, m, S.e.data());
    r.A = gauss::sum(S.e.data(), m);
    double d0 = kInf;
    for (std::int64_t j = 0; j < n; ++j)
        if (j != i && S.D[j] < d0) d0 = S.D[j];
    if (!(d0 < kInf)) return r;
    S.Dx.clear();
    for (std::int64_t j = 0; j < n; ++j)
        if (j != i && S.D[j] - d0 <= K.T) S.Dx.push_back(S.D[j]);
    m = static_cast<std::int64_t>(S.Dx.size());
    S.e.resize(m);
    gauss::terms(S.Dx.data(), nullptr, d0, m, S.e.data());
    r.b0 = -0.5 * d0, r.B = gauss::sum(S.e.data(), m);
    return r;
}

struct CvValue {
    double value = kNaN, deriv = kNaN;  // the scale-free objective and its derivative in s = log h (1-D)
};

// The objective at h (p,) for kind 1 (least squares: cv(h) prod href) or 2 (likelihood: -mean log f_-i + sum log
// href); want_raw: the criterion itself (cv_ls, or sum log f_-i).
CvValue cv_eval(const CvData& K, int kind, const double* h, const double* href, bool raw) {
    const std::int64_t n = K.n, p = K.p;
    std::vector<Axis> ax(p);
    for (std::int64_t l = 0; l < p; ++l) ax[l] = gauss::axis_of(h[l]);
    std::vector<Row> rows(n);
    const std::int64_t cost = (p == 1 ? 256 : n * p) + 16;
    const auto run = [&](std::int64_t i0, std::int64_t i1) {
        CvScratch S;
        for (std::int64_t i = i0; i < i1; ++i) rows[i] = cv_row(K, ax.data(), i, S);
    };
    if (n * cost >= kParallelThreshold && n > 1 && parallel::available()) {
        const std::int64_t per = std::max<std::int64_t>(1, (std::int64_t{1} << 17) / cost);
        const std::int64_t tasks = (n + per - 1) / per;
        parallel::parallel_for(
            tasks, [&](std::int64_t t) { run(t * per, std::min(n, (t + 1) * per)); }, work_ns(per * cost)
        );
    } else {
        run(0, n);
    }
    const double nn = static_cast<double>(n);
    double ph = 1, pref = 1, lh = 0, lref = 0;
    for (std::int64_t l = 0; l < p; ++l) ph *= h[l], pref *= href ? href[l] : 1.0, lh += std::log(h[l]);
    for (std::int64_t l = 0; l < p; ++l) lref += href ? std::log(href[l]) : 0.0;
    CvValue out;
    if (kind == 1) {
        double s[4];
        gauss::sums_of<4>(
            n,
            [&](std::int64_t i, double* v) {
                const double eb = vmath::exp1(rows[i].b0);
                v[0] = rows[i].A, v[1] = eb * rows[i].B, v[2] = rows[i].dA, v[3] = eb * rows[i].dB;
            },
            s
        );
        const double ca = std::pow(2.0 * kSqrtPi, static_cast<double>(p)),
                     cb = std::pow(kde::kSqrt2Pi, static_cast<double>(p));
        const double cv = s[0] / (nn * nn * ca * ph) - 2.0 * s[1] / (nn * (nn - 1) * cb * ph);
        if (raw) {
            out.value = cv;
            return out;
        }
        out.value = cv * pref;
        // h d(cv)/dh (1-D): (1 / h) [dA / (n^2 2 sqrt pi) - 2 dB / (n (n - 1) sqrt(2 pi))]
        out.deriv = (s[2] / (nn * nn * ca) - 2.0 * s[3] / (nn * (nn - 1) * cb)) / ph * pref;
        return out;
    }
    double s[2];
    gauss::sums_of<2>(
        n,
        [&](std::int64_t i, double* v) {
            const Row& r = rows[i];
            v[0] = r.B > 0 ? r.b0 + std::log(r.B) : -kInf;
            v[1] = r.B > 0 ? r.mB / r.B - 1.0 : 0.0;
        },
        s
    );
    const double ll = s[0] - nn * (std::log(nn - 1.0) + lh + static_cast<double>(p) * std::log(kde::kSqrt2Pi));
    if (raw) {
        out.value = ll;
        return out;
    }
    out.value = -ll / nn + lref;
    out.deriv = -s[1] / nn;
    return out;
}

struct Choice {
    std::vector<double> h;
    std::int64_t status = 0, nfev = 0;
    double value = kNaN;
};

Choice search_1d(const std::function<CvValue(double)>& f, double href, bool derivative) {
    Choice ch;
    double g[kGrid], s[kGrid];
    const double step = 2.0 * kLog64 / (kGrid - 1);
    double best = kInf;
    std::int64_t kb = -1;
    for (std::int64_t k = 0; k < kGrid; ++k) {
        s[k] = -kLog64 + static_cast<double>(k) * step;
        g[k] = f(s[k]).value;
        if (g[k] != g[k]) g[k] = kInf;
        if (g[k] <= best && g[k] < kInf) best = g[k], kb = k;  // ties: the larger bandwidth
    }
    ch.nfev = kGrid;
    if (kb < 0) {
        ch.status = -3;
        return ch;
    }
    double sb = s[kb], fb = best;
    if (kb == 0 || kb == kGrid - 1) {
        ch.status = kb == 0 ? -1 : 1;
    } else {
        bool done = false;
        if (derivative) {
            const double dl = f(s[kb - 1]).deriv, dr = f(s[kb + 1]).deriv;
            ch.nfev += 2;
            if (dl < 0 && dr > 0) {
                const ScalarResult r =
                    brentq([&](double v) { return f(v).deriv; }, s[kb - 1], s[kb + 1], dl, dr, 1e-12, 0.0, 200);
                ch.nfev += r.nfev;
                const double fv = f(r.x).value;
                ++ch.nfev;
                if (fv <= best) sb = r.x, fb = fv;
                if (!r.converged) ch.status = 2;
                done = true;
            }
        }
        if (!done) {
            const ScalarResult r = fminbound(
                [&](double v) {
                    const double fv = f(v).value;
                    return fv == fv ? fv : kInf;
                },
                s[kb - 1], s[kb + 1], 1e-10, 200
            );
            ch.nfev += r.nfev;
            if (r.f < best) sb = r.x, fb = r.f;
            if (!r.converged) ch.status = 2;
        }
    }
    ch.h = {href * vmath::exp1(sb)};
    ch.value = fb;
    return ch;
}

// Nelder-Mead on s = log(h / href) in the box [-log 64, log 64]^p, as kernel_regression's (decision SM88).
Choice search_nd(const std::function<double(const double*)>& f0, const std::vector<double>& href) {
    const std::int64_t N = static_cast<std::int64_t>(href.size());
    const std::int64_t maxfev = 200 * N;
    Choice ch;
    std::vector<std::vector<double>> sim(N + 1, std::vector<double>(N, 0.0));
    std::vector<double> fs(N + 1);
    const auto clip = [](std::vector<double>& v) {
        for (double& x : v) x = std::min(kLog64, std::max(-kLog64, x));
    };
    const auto f = [&](const std::vector<double>& v) {
        ++ch.nfev;
        const double r = f0(v.data());
        return r == r ? r : kInf;
    };
    for (std::int64_t k = 0; k < N; ++k) sim[k + 1][k] = 0.5;
    for (std::int64_t k = 0; k <= N; ++k) fs[k] = f(sim[k]);
    std::vector<std::int64_t> ord(N + 1);
    const auto sort = [&]() {
        std::iota(ord.begin(), ord.end(), std::int64_t{0});
        std::stable_sort(ord.begin(), ord.end(), [&](std::int64_t a, std::int64_t b) { return fs[a] < fs[b]; });
        std::vector<std::vector<double>> s2(N + 1);
        std::vector<double> f2(N + 1);
        for (std::int64_t k = 0; k <= N; ++k) s2[k] = sim[ord[k]], f2[k] = fs[ord[k]];
        sim.swap(s2), fs.swap(f2);
    };
    sort();
    bool converged = false;
    std::vector<double> xbar(N), xr(N), xe(N), xc(N);
    while (ch.nfev < maxfev) {
        double dx = 0, df = 0;
        for (std::int64_t k = 1; k <= N; ++k) {
            for (std::int64_t l = 0; l < N; ++l) dx = std::max(dx, std::fabs(sim[k][l] - sim[0][l]));
            df = std::max(df, fs[k] == fs[0] ? 0.0 : std::fabs(fs[k] - fs[0]));
        }
        if (dx <= 1e-5 && df <= 1e-9) {
            converged = true;
            break;
        }
        for (std::int64_t l = 0; l < N; ++l) {
            double s = 0;
            for (std::int64_t k = 0; k < N; ++k) s += sim[k][l];
            xbar[l] = s / static_cast<double>(N);
        }
        for (std::int64_t l = 0; l < N; ++l) xr[l] = 2.0 * xbar[l] - sim[N][l];
        clip(xr);
        const double fr = f(xr);
        bool shrink = false;
        if (fr < fs[0]) {
            for (std::int64_t l = 0; l < N; ++l) xe[l] = 3.0 * xbar[l] - 2.0 * sim[N][l];
            clip(xe);
            const double fe = f(xe);
            if (fe < fr) sim[N] = xe, fs[N] = fe;
            else sim[N] = xr, fs[N] = fr;
        } else if (fr < fs[N - 1]) {
            sim[N] = xr, fs[N] = fr;
        } else if (fr < fs[N]) {
            for (std::int64_t l = 0; l < N; ++l) xc[l] = 1.5 * xbar[l] - 0.5 * sim[N][l];
            clip(xc);
            const double fc = f(xc);
            if (fc <= fr) sim[N] = xc, fs[N] = fc;
            else shrink = true;
        } else {
            for (std::int64_t l = 0; l < N; ++l) xc[l] = 0.5 * xbar[l] + 0.5 * sim[N][l];
            clip(xc);
            const double fc = f(xc);
            if (fc < fs[N]) sim[N] = xc, fs[N] = fc;
            else shrink = true;
        }
        if (shrink) {
            for (std::int64_t k = 1; k <= N; ++k) {
                for (std::int64_t l = 0; l < N; ++l) sim[k][l] = sim[0][l] + 0.5 * (sim[k][l] - sim[0][l]);
                clip(sim[k]);
                fs[k] = f(sim[k]);
            }
        }
        sort();
    }
    if (!(fs[0] < kInf)) {
        ch.status = -3;
        return ch;
    }
    ch.status = converged ? 0 : 2;
    for (std::int64_t l = 0; l < N; ++l) {
        if (sim[0][l] <= -kLog64) ch.status = -1;
        else if (sim[0][l] >= kLog64 && ch.status != -1) ch.status = 1;
    }
    ch.h.resize(N);
    for (std::int64_t l = 0; l < N; ++l) ch.h[l] = href[l] * vmath::exp1(sim[0][l]);
    ch.value = fs[0];
    return ch;
}

// The data of the criteria: 1-D sorted, n-D sorted by rows (the order of the sums).
std::vector<double> sorted_rows(const double* X, std::int64_t n, std::int64_t p) {
    std::vector<std::int64_t> ord(n);
    std::iota(ord.begin(), ord.end(), std::int64_t{0});
    std::sort(ord.begin(), ord.end(), [&](std::int64_t a, std::int64_t b) {
        for (std::int64_t l = 0; l < p; ++l)
            if (X[a * p + l] != X[b * p + l]) return X[a * p + l] < X[b * p + l];
        return a < b;
    });
    std::vector<double> out(n * p);
    for (std::int64_t k = 0; k < n; ++k)
        for (std::int64_t l = 0; l < p; ++l) out[k * p + l] = X[ord[k] * p + l];
    return out;
}

// ---- bindings ----------------------------------------------------------------------------------------------------

nb::object bw_sj_core(nb::handle x_obj, int method, std::int64_t nbins) {
    const NDArray& xa = vec_arg(x_obj, "x");
    const std::int64_t n = xa.shape[0];
    if (nbins < 1) throw nb::value_error("'nb' must be >= 1");
    std::pair<double, int> r;
    {
        const parallel::Unlocked unlocked(n * n >= kParallelThreshold || nbins * nbins >= kParallelThreshold);
        r = sj(dptr(xa), n, method, nbins);
    }
    return nb::make_tuple(r.first, r.second);
}

nb::object bw_isj_core(nb::handle x_obj, std::int64_t N) {
    const NDArray& xa = vec_arg(x_obj, "x");
    const std::int64_t n = xa.shape[0];
    if (N < 2) throw nb::value_error("'n_grid' must be >= 2");
    if (n < 1) return nb::make_tuple(kNaN, 1);
    const std::shared_ptr<const fft::Plan> plan = fft::get_plan(fft::r2r_kind(fft::R2r{}), N);
    std::pair<double, int> r;
    {
        const parallel::Unlocked unlocked(n + N >= kParallelThreshold);
        r = isj(dptr(xa), n, N, *plan);
    }
    return nb::make_tuple(r.first, r.second);
}

nb::object bw_cv_core(nb::handle X_obj, int kind, std::int64_t nbins, nb::handle href_obj) {
    const NDArray& Xa = mat_arg(X_obj, "X");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray& ra = vec_arg(href_obj, "href", p);
    std::vector<double> href(dptr(ra), dptr(ra) + p);
    if (kind != 1 && kind != 2) throw nb::value_error("'kind' must be 1 or 2");
    NDArray ho;
    {
        const std::int64_t s[1] = {p};
        ho = new_array(DType::Float64, s);
    }
    Choice ch;
    {
        const parallel::Unlocked unlocked(n * n >= kParallelThreshold);
        const std::vector<double> xs = sorted_rows(dptr(Xa), n, p);
        CvData K;
        K.n = n, K.p = p, K.X = xs.data(), K.T = gauss::cut_threshold(n, 1.0);
        if (nbins > 0) {
            const Counts c = pair_counts(xs.data(), n, nbins);
            ch = search_1d(
                [&](double s) {
                    const double h = href[0] * vmath::exp1(s);
                    return CvValue{ucv(c, n, h) * href[0], kNaN};
                },
                href[0], false
            );
        } else if (p == 1) {
            ch = search_1d(
                [&](double s) {
                    const double h = href[0] * vmath::exp1(s);
                    return cv_eval(K, kind, &h, href.data(), false);
                },
                href[0], true
            );
        } else {
            ch = search_nd(
                [&](const double* s) {
                    std::vector<double> h(p);
                    for (std::int64_t l = 0; l < p; ++l) h[l] = href[l] * vmath::exp1(s[l]);
                    return cv_eval(K, kind, h.data(), href.data(), false).value;
                },
                href
            );
        }
        if (!ch.h.empty()) std::copy(ch.h.begin(), ch.h.end(), dout(ho));
    }
    if (ch.h.empty()) return nb::make_tuple(nb::none(), ch.status, kNaN, ch.nfev);
    return nb::make_tuple(wrap(std::move(ho)), ch.status, ch.value, ch.nfev);
}

double bw_cv_value(nb::handle X_obj, int kind, std::int64_t nbins, nb::handle h_obj) {
    const NDArray& Xa = mat_arg(X_obj, "X");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray& ha = vec_arg(h_obj, "h", p);
    const parallel::Unlocked unlocked(n * n >= kParallelThreshold);
    const std::vector<double> xs = sorted_rows(dptr(Xa), n, p);
    if (nbins > 0) return ucv(pair_counts(xs.data(), n, nbins), n, dptr(ha)[0]);
    CvData K;
    K.n = n, K.p = p, K.X = xs.data(), K.T = gauss::cut_threshold(n, 1.0);
    return cv_eval(K, kind, dptr(ha), nullptr, true).value;
}

}  // namespace

void bind_bandwidth(nb::module_& m) {
    m.def("bw_sj", named("bw_sj", &bw_sj_core), nb::arg("x").none(), nb::arg("method"), nb::arg("nb"));
    m.def("bw_isj", named("bw_isj", &bw_isj_core), nb::arg("x").none(), nb::arg("n_grid"));
    m.def(
        "bw_cv", named("bw_cv", &bw_cv_core), nb::arg("X").none(), nb::arg("kind"), nb::arg("nb"),
        nb::arg("href").none()
    );  // fmt: skip
    m.def(
        "bw_cv_value", named("bw_cv_value", &bw_cv_value), nb::arg("X").none(), nb::arg("kind"), nb::arg("nb"),
        nb::arg("h").none()
    );  // fmt: skip
}

}  // namespace sd::smooth
