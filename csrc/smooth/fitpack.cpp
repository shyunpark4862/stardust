// FITPACK's smoothing splines of sd.smooth (docs/design/smooth.md §3.3; decisions SM12, SM61-SM66): the knot
// placement of generate_knots and the smoothing of make_splrep and make_splprep, scipy 1.18.1's
// _fitpack_repro.py (Dierckx's fpcurf, fppara and fpperi) on smooth/bandqr.hpp.
//
//   lsq        the least-squares spline min sum (w_i (y_i - s(x_i)))^2 on the knots t: rows w_i b_i, Givens QR,
//              back substitution; the residuals res_i = sum over the columns of (w_i (y_i - s(x_i)))^2 by
//              evaluating the spline at every point (scipy's fpback), fp their sum
//   periodic   the coefficients repeat, c_{nf + j} = c_j (nf = n - 2k - 1 free ones): the free coefficients
//              k .. nf - 1 form the band and 0 .. k - 1 a dense border of k columns, so every row (a data row
//              or a jump row, wrapped modulo nf) is a band run plus border entries; the last point (a copy of the
//              first) is not a row (fpperi)
//   knots      _generate_knots_impl: from no interior knot, add knots where the residuals are largest (fpknot:
//              the interval of the largest sum of residuals, a point's residual at a knot split in halves, and
//              the middle data point of that interval), nplus knots at a time from the last two decreases,
//              until fp <= s (|fp - s| < 0.001 s) or the knots reach nest or the interpolation's number
//   smoothing  F(p) = fp(p) - s for the spline minimizing sum (w r)^2 + (1 / p^2) sum jumps^2, the jumps of the
//              k-th derivative at the interior knots scaled as fpdisc ((delta / nrint)^k); the rows of the data's
//              triangle and the jump rows divided by p go through one Givens QR in the order of their first
//              column. root_rati's rational iteration (20 steps, |F| < 0.001 s), then bisection on the bracket
//              (100 steps) where scipy warns
//
// Private bindings (the Python layer checks the arguments; x sorted, y (m, ncol), w (m,)):
//   fitpack_knots(x, y, w, xb, xe, k, s, nest, periodic) -> (list of knot arrays, interp)
//       interp: True when the last vector is the interpolation's (the caller appends make_interp_spline's knots)
//   fitpack_fit(x, y, w, xb, xe, k, s, t, periodic) -> (c (nc, ncol), fp, code)
//       code 0 converged, 1 fp(inf) > s (s cannot be reached on t), 2 fp(0) <= s (the least-squares polynomial),
//       3 NaN residuals, 4 singular
//   fitpack_lsq(x, y, w, t, k, periodic) -> (c, residuals, fp)  (tests)
//   fitpack_chord(X) -> u                                     scipy's chord-length parameters of the rows of X
//   fitpack_fpknot(x, t, k, residuals) -> the new knot           (tests, against scipy's _dierckx.fpknot)

#include "smooth/smooth.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "interpolate/deboor.hpp"
#include "parallel/pool.hpp"
#include "smooth/bandqr.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::smooth {

namespace nb = nanobind;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kTol = 0.001;     // curfit's tol
constexpr std::int64_t kMaxit = 20;  // curfit's maxit
constexpr std::int64_t kBisect = 100;

const NDArray& f64_arg(nb::handle o, const char* what, int ndim) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != ndim || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error((std::string("'") + what + "' must be a C-contiguous float64 array").c_str());
    return a;
}

const double* data_of(const NDArray& a) { return reinterpret_cast<const double*>(a.data); }

NDArray vec_array(const std::vector<double>& v) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(v.size())};
    NDArray a = new_array(DType::Float64, shape);
    std::copy(v.begin(), v.end(), reinterpret_cast<double*>(a.data));
    return a;
}

NDArray mat_array(const std::vector<double>& v, std::int64_t rows, std::int64_t cols) {
    const std::int64_t shape[2] = {rows, cols};
    NDArray a = new_array(DType::Float64, shape);
    std::copy(v.begin(), v.begin() + rows * cols, reinterpret_cast<double*>(a.data));
    return a;
}

struct Data {
    const double* x;
    const double* y;
    const double* w;
    std::int64_t m, ncol;
};

// The columns of a spline on knots t: nc basis functions; periodic: nf free ones, band nb = nf - k, border k.
struct Layout {
    std::int64_t N = 0, k = 0, nc = 0, nf = 0, nb = 0, kb = 0;
    bool periodic = false;
};

Layout layout(std::int64_t N, std::int64_t k, bool periodic) {
    Layout L;
    L.N = N;
    L.k = k;
    L.nc = N - k - 1;
    L.periodic = periodic;
    if (periodic) {
        L.nf = L.nc - k;
        L.kb = std::min(k, L.nf);
        L.nb = L.nf - L.kb;
    } else {
        L.nf = L.nc;
        L.nb = L.nc;
        L.kb = 0;
    }
    return L;
}

// The interval l of each sorted x (k <= l <= nc - 1).
std::vector<std::int64_t> intervals(const double* x, std::int64_t m, const std::vector<double>& t, const Layout& L) {
    std::vector<std::int64_t> ls(static_cast<std::size_t>(m));
    std::int64_t l = L.k;
    for (std::int64_t i = 0; i < m; ++i) {
        while (l < L.nc - 1 && t[l + 1] <= x[i]) ++l;
        ls[i] = l;
    }
    return ls;
}

// A row over the coefficient columns o .. o + width - 1 split into the band run (start ob, h) and the border.
// Returns ob (nb for a border-only row).
std::int64_t place_row(
    const Layout& L, std::int64_t o, const double* v, std::int64_t width, std::int64_t bw, double* h, double* hb
) {
    std::fill(h, h + bw, 0.0);
    std::fill(hb, hb + L.kb, 0.0);
    if (!L.periodic) {
        for (std::int64_t e = 0; e < width; ++e) h[e] = v[e];
        return o;
    }
    std::int64_t ob = L.nb;
    for (std::int64_t e = 0; e < width; ++e) {
        const std::int64_t j = (o + e) % L.nf;
        if (j >= L.kb) ob = std::min(ob, j - L.kb);
    }
    for (std::int64_t e = 0; e < width; ++e) {
        const std::int64_t j = (o + e) % L.nf;
        if (j < L.kb) hb[j] += v[e];
        else h[j - L.kb - ob] += v[e];
    }
    return ob;
}

// The full coefficients (nc x ncol) from the solution of the reduced columns.
void expand(const Layout& L, const std::vector<double>& u, std::int64_t ncol, std::vector<double>& c) {
    c.assign(static_cast<std::size_t>(L.nc * ncol), 0.0);
    for (std::int64_t j = 0; j < L.nc; ++j) {
        std::int64_t r = j;
        if (L.periodic) {
            const std::int64_t f = j % L.nf;
            r = f < L.kb ? L.nb + f : f - L.kb;
        }
        for (std::int64_t q = 0; q < ncol; ++q) c[j * ncol + q] = u[r * ncol + q];
    }
}

// The residuals (w_i (y_i - s(x_i)))^2 summed over the columns at every point, and fp over the first `upto`.
double residuals(
    const Data& D,
    const std::vector<double>& t,
    const Layout& L,
    const std::vector<double>& c,
    std::int64_t upto,
    std::vector<double>& res
) {
    res.assign(static_cast<std::size_t>(D.m), 0.0);
    std::vector<double> b(static_cast<std::size_t>(L.k + 1)), hh(b.size());
    const std::vector<std::int64_t> ls = intervals(D.x, D.m, t, L);
    double fp = 0.0;
    for (std::int64_t i = 0; i < D.m; ++i) {
        interpolate::basis_values(t.data(), L.k, ls[i], D.x[i], 0, b.data(), hh.data());
        const std::int64_t o = ls[i] - L.k;
        double r2 = 0.0;
        for (std::int64_t q = 0; q < D.ncol; ++q) {
            double s = 0.0;
            for (std::int64_t e = 0; e <= L.k; ++e) s += b[e] * c[(o + e) * D.ncol + q];
            const double r = D.w[i] * (D.y[i * D.ncol + q] - s);
            r2 += r * r;
        }
        res[i] = r2;
        if (i < upto) fp += r2;
    }
    return fp;
}

// The data's triangle on t (rows w_i b_i of the points used: all but the last when periodic).
void data_triangle(const Data& D, const std::vector<double>& t, const Layout& L, std::int64_t bw, BandQR& Q) {
    Q.reset(L.nb, bw, L.kb, D.ncol);
    const std::int64_t used = L.periodic ? D.m - 1 : D.m;
    const std::vector<std::int64_t> ls = intervals(D.x, used, t, L);
    std::vector<double> b(static_cast<std::size_t>(L.k + 1)), hh(b.size()), h(static_cast<std::size_t>(bw)),
        hb(static_cast<std::size_t>(L.kb)), y(static_cast<std::size_t>(D.ncol));
    for (std::int64_t i = 0; i < used; ++i) {
        const double wi = D.w[i];
        if (wi == 0.0) continue;
        interpolate::basis_values(t.data(), L.k, ls[i], D.x[i], 0, b.data(), hh.data());
        for (double& v : b) v *= wi;
        const std::int64_t ob = place_row(L, ls[i] - L.k, b.data(), L.k + 1, bw, h.data(), hb.data());
        for (std::int64_t q = 0; q < D.ncol; ++q) y[q] = wi * D.y[i * D.ncol + q];
        Q.add(ob, h.data(), hb.data(), y.data());
    }
}

bool solve(const BandQR& Q, const Layout& L, std::int64_t ncol, std::vector<double>& c) {
    if (Q.first_bad_diagonal() >= 0) return false;
    std::vector<double> u(static_cast<std::size_t>((L.nb + L.kb) * ncol));
    Q.solve(u.data());
    expand(L, u, ncol, c);
    return true;
}

struct LsqOut {
    std::vector<double> c, res;
    double fp = kNaN;
    bool ok = false;
};

LsqOut lsq(const Data& D, const std::vector<double>& t, std::int64_t k, bool periodic) {
    const Layout L = layout(static_cast<std::int64_t>(t.size()), k, periodic);
    BandQR Q;
    data_triangle(D, t, L, k + 1, Q);
    LsqOut out;
    out.ok = solve(Q, L, D.ncol, out.c);
    if (!out.ok) {
        out.c.assign(static_cast<std::size_t>(L.nc * D.ncol), kNaN);
        out.res.assign(static_cast<std::size_t>(D.m), kNaN);
        return out;
    }
    out.fp = residuals(D, t, L, out.c, periodic ? D.m - 1 : D.m, out.res);
    return out;
}

// fpknot: the new knot (a data point) in the interval of the largest sum of residuals with a point inside.
double fpknot(const double* x, std::int64_t m, const std::vector<double>& t, std::int64_t k, const double* res) {
    const std::int64_t n = static_cast<std::int64_t>(t.size()), nk1 = n - k - 1;
    std::vector<double> fpint;
    std::vector<std::int64_t> nrdata, starts;
    std::int64_t l = k + 1, start = 0;
    double fpart = 0.0;
    for (std::int64_t it = 0; it < m; ++it) {
        bool fresh = false;
        if (l < nk1 && x[it] >= t[l]) {
            fresh = true;
            ++l;
        }
        const double term = res[it];
        fpart += term;
        if (fresh) {
            const double store = term * 0.5;
            fpint.push_back(fpart - store);
            fpart = store;
            nrdata.push_back(it - start - 1);
            starts.push_back(start);
            start = it;
        }
    }
    fpint.push_back(fpart);
    nrdata.push_back(m - 1 - start - 1);
    starts.push_back(start);
    double fpmax = 0.0;
    std::int64_t number = -1;
    for (std::size_t j = 0; j < fpint.size(); ++j) {
        if (fpmax >= fpint[j] || nrdata[j] <= 0) continue;
        fpmax = fpint[j];
        number = static_cast<std::int64_t>(j);
    }
    if (number < 0) return kNaN;
    return x[starts[number] + nrdata[number] / 2 + 1];
}

void insert_knot(std::vector<double>& t, double v) {
    const auto it = std::lower_bound(t.begin(), t.end(), v);
    t.insert(it, v);
}

// The periodic knots' wrap (fpperi): t[k] = xb, t[n - k - 1] = xe and the k knots beyond each end.
void wrap_knots(std::vector<double>& t, std::int64_t k, double xb, double xe) {
    const std::int64_t n = static_cast<std::int64_t>(t.size());
    const double per = xe - xb;
    t[k] = xb;
    t[n - k - 1] = xe;
    for (std::int64_t j = 1; j <= k; ++j) {
        t[k - j] = t[n - k - j - 1] - per;
        t[n - k + j - 1] = t[k + j] + per;
    }
}

struct KnotsOut {
    std::vector<std::vector<double>> list;
    bool interp = false;
    bool nan = false;
};

KnotsOut generate(const Data& D, double xb, double xe, std::int64_t k, double s, std::int64_t nest, bool periodic) {
    KnotsOut out;
    const double acc = s * kTol;
    const std::int64_t m = D.m;
    const std::int64_t nmin = 2 * (k + 1);
    const std::int64_t nmax = periodic ? m + 2 * k : m + k + 1;
    const double per = xe - xb;
    double fp = 0.0, fpold = 0.0;
    std::vector<double> t;
    std::int64_t nplus = 1;
    if (periodic) {
        std::vector<double> t0(static_cast<std::size_t>(nmin), 0.0);
        for (std::int64_t i = 0; i <= k; ++i) {
            t0[i] = D.x[0] - static_cast<double>(k - i) * per;
            t0[i + k + 1] = D.x[m - 1] + static_cast<double>(i) * per;
        }
        const LsqOut r = lsq(D, t0, k, true);
        if (std::isnan(r.fp)) {
            out.nan = true;
            return out;
        }
        fp = r.fp;
        if (fp - s < acc || nmax == nmin) {
            out.list.push_back(t0);
            return out;
        }
        t.assign(static_cast<std::size_t>(2 * k + 3), 0.0);
        t[k + 1] = D.x[(m + 1) / 2 - 1];
    } else {
        for (std::int64_t i = 0; i <= k; ++i) t.push_back(xb);
        for (std::int64_t i = 0; i <= k; ++i) t.push_back(xe);
    }
    std::int64_t n = static_cast<std::int64_t>(t.size());
    for (std::int64_t iter = 0; iter < m; ++iter) {
        if (periodic) {
            n = static_cast<std::int64_t>(t.size());
            wrap_knots(t, k, xb, xe);
        }
        out.list.push_back(t);
        fpold = fp;
        LsqOut r = lsq(D, t, k, periodic);
        double rs = 0.0;
        for (double v : r.res) rs += v;
        if (std::isnan(rs)) {
            out.nan = true;
            return out;
        }
        fp = r.fp;
        const double fpms = fp - s;
        if (std::abs(fpms) < acc || fpms < 0) return out;
        if (n == nmin) {
            nplus = 1;
        } else {
            const double delta = fpold - fp;
            std::int64_t npl1;
            if (delta > acc) {
                const double v = static_cast<double>(nplus) * fpms / delta;
                npl1 = v >= 4.0 * static_cast<double>(nplus) + 4.0 ? 4 * nplus + 4 : static_cast<std::int64_t>(v);
            } else {
                npl1 = nplus * 2;
            }
            nplus = std::min(nplus * 2, std::max({npl1, nplus / 2, std::int64_t{1}}));
        }
        for (std::int64_t j = 0; j < nplus; ++j) {
            const double v = fpknot(D.x, m, t, k, r.res.data());
            if (std::isnan(v)) {
                out.nan = true;
                return out;
            }
            insert_knot(t, v);
            n = static_cast<std::int64_t>(t.size());
            if (n >= nmax) {
                out.interp = true;
                return out;
            }
            if (n >= nest) {  // scipy yields it without the periodic wrap
                out.list.push_back(t);
                return out;
            }
            if (j < nplus - 1) {
                r = lsq(D, t, k, periodic);
                double s2 = 0.0;
                for (double vv : r.res) s2 += vv;
                if (std::isnan(s2)) {
                    out.nan = true;
                    return out;
                }
            }
        }
    }
    return out;
}

// prodd and disc (fpdisc): the jumps of the k-th derivatives of the B-splines at the interior knots, each row over
// the k + 2 columns jj .. jj + k + 1, times (delta / nrint)^k.
std::vector<double> disc(const std::vector<double>& t, std::int64_t k, std::int64_t& rows) {
    const std::int64_t n = static_cast<std::int64_t>(t.size());
    const double delta = t[n - k - 1] - t[k];
    const std::int64_t nrint = n - 2 * k - 1;
    rows = std::max<std::int64_t>(0, nrint - 1);
    std::vector<double> matr(static_cast<std::size_t>(rows * (k + 2)));
    const double fac = std::pow(delta / static_cast<double>(nrint), static_cast<double>(k));
    for (std::int64_t jj = 0; jj < rows; ++jj) {
        const std::int64_t j = jj + k + 1;
        for (std::int64_t ii = 0; ii < k + 2; ++ii) {
            const std::int64_t i = jj + ii;
            double prod = 1.0;
            for (std::int64_t q = 0; q < k + 2; ++q)
                if (i + q != j) prod *= (t[j] - t[i + q]);
            matr[jj * (k + 2) + ii] = (t[i + k + 1] - t[i]) / prod;
        }
    }
    for (double& v : matr) v *= fac;
    return matr;
}

// F(p): the data's triangle (bandwidth k + 1) and the jump rows / p in one QR (bandwidth k + 2).
struct Smoother {
    const Data* D;
    const std::vector<double>* t;
    Layout L;
    BandQR data;
    std::vector<double> jumps;
    std::int64_t rows = 0;
    double s = 0.0;
    std::vector<double> c;  // the last spline's coefficients
    double fp = kNaN;
    bool singular = false;

    double operator()(double p) {
        const std::int64_t k = L.k, bw = k + 2, ncol = D->ncol;
        BandQR Q;
        Q.reset(L.nb, bw, L.kb, ncol);
        std::vector<double> h(static_cast<std::size_t>(bw)), hb(static_cast<std::size_t>(L.kb)),
            y(static_cast<std::size_t>(ncol)), v(static_cast<std::size_t>(k + 2)), hj(h.size()), hbj(hb.size()),
            yj(y.size());
        const double pinv = 1.0 / p;
        std::int64_t jr = 0;
        const auto jump_rows = [&](std::int64_t upto) {  // the jump rows whose band run starts at or before upto
            for (; jr < rows; ++jr) {
                for (std::int64_t e = 0; e < k + 2; ++e) v[e] = jumps[jr * (k + 2) + e] * pinv;
                const std::int64_t ob = place_row(L, jr, v.data(), k + 2, bw, hj.data(), hbj.data());
                if (ob > upto) return;
                std::fill(yj.begin(), yj.end(), 0.0);
                Q.add(ob, hj.data(), hbj.data(), yj.data());
            }
        };
        for (std::int64_t j = 0; j < L.nb + L.kb; ++j) {
            const std::int64_t ob = j < L.nb ? j : L.nb;
            jump_rows(ob - 1);
            std::fill(h.begin(), h.end(), 0.0);
            std::fill(hb.begin(), hb.end(), 0.0);
            if (j < L.nb)
                for (std::int64_t e = 0; e <= k && j + e < L.nb; ++e) h[e] = data.R[j * (k + 1) + e];
            for (std::int64_t b = 0; b < L.kb; ++b) hb[b] = data.B[j * L.kb + b];
            for (std::int64_t q = 0; q < ncol; ++q) y[q] = data.Z[j * ncol + q];
            Q.add(ob, h.data(), hb.data(), y.data());
        }
        jump_rows(L.nb + L.kb);
        singular = !solve(Q, L, ncol, c);
        if (singular) return kNaN;
        std::vector<double> res;
        fp = residuals(*D, *t, L, c, L.periodic ? D->m - 1 : D->m, res);
        return fp - s;
    }
};

double fprati(double p1, double f1, double p2, double f2, double p3, double f3) {
    const double h1 = f1 * (f2 - f3), h2 = f2 * (f3 - f1), h3 = f3 * (f1 - f2);
    if (p3 == kInf) return -(p2 * h1 + p1 * h2) / h3;
    return -(p1 * p2 * h3 + p2 * p3 * h1 + p1 * p3 * h2) / (p1 * h1 + p2 * h2 + p3 * h3);
}

// root_rati, then bisection on the bracket where it stops without |F| < acc. True when |F| < acc.
bool root(Smoother& f, double p0, double p1, double f1, double p3, double f3, double acc) {
    const double con1 = 0.1, con9 = 0.9, con4 = 0.04;
    int ich1 = 0, ich3 = 0;
    double p = p0;
    for (std::int64_t it = 0; it < kMaxit; ++it) {
        const double p2 = p, f2 = f(p);
        if (std::isnan(f2)) return false;
        if (std::abs(f2) < acc) return true;
        if (ich3 == 0) {
            if (f2 - f3 <= acc) {
                p3 = p2;
                f3 = f2;
                p = p * con4;
                if (p <= p1) p = p1 * con9 + p2 * con1;
                continue;
            }
            if (f2 < 0) ich3 = 1;
        }
        if (ich1 == 0) {
            if (f1 - f2 <= acc) {
                p1 = p2;
                f1 = f2;
                p = p / con4;
                if (p3 != kInf && p <= p3) p = p2 * con1 + p3 * con9;
                continue;
            }
            if (f2 > 0) ich1 = 1;
        }
        if (f1 <= f2 || f2 <= f3) break;
        p = fprati(p1, f1, p2, f2, p3, f3);
        if (f2 < 0) {
            p3 = p2;
            f3 = f2;
        } else {
            p1 = p2;
            f1 = f2;
        }
    }
    // bisection on [p1, p3] with F(p1) > 0 > F(p3); F decreases in p
    double lo = p1, hi = p3;
    if (!(f1 > 0)) lo = 0.0;
    std::int64_t step = 0;
    if (hi == kInf || !(f3 < 0)) {
        double q = std::max(2.0 * lo, 1.0);
        for (; step < kBisect; ++step, q *= 4.0) {
            const double fq = f(q);
            if (std::isnan(fq)) return false;
            if (std::abs(fq) < acc) return true;
            if (fq < 0) {
                hi = q;
                break;
            }
            lo = q;
        }
        if (hi == kInf) return false;
    }
    for (; step < kBisect; ++step) {
        const double mid = lo > 0 ? std::sqrt(lo) * std::sqrt(hi) : 0.5 * (lo + hi);
        const double fm = f(mid);
        if (std::isnan(fm)) return false;
        if (std::abs(fm) < acc) return true;
        if (fm > 0) lo = mid;
        else hi = mid;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------ bindings

Data data_args(nb::handle xo, nb::handle yo, nb::handle wo) {
    const NDArray& xa = f64_arg(xo, "x", 1);
    const NDArray& ya = f64_arg(yo, "y", 2);
    const NDArray& wa = f64_arg(wo, "w", 1);
    return {data_of(xa), data_of(ya), data_of(wa), xa.shape[0], ya.shape[1]};
}

nb::tuple fitpack_knots(
    nb::handle xo,
    nb::handle yo,
    nb::handle wo,
    double xb,
    double xe,
    std::int64_t k,
    double s,
    std::int64_t nest,
    bool periodic
) {
    const Data D = data_args(xo, yo, wo);
    KnotsOut r;
    {
        const parallel::Unlocked unlocked(D.m * (k + 1) * 8 >= kParallelThreshold);
        r = generate(D, xb, xe, k, s, nest, periodic);
    }
    if (r.nan) throw nb::value_error("nan");
    nb::list out;
    for (const auto& t : r.list) out.append(wrap(vec_array(t)));
    return nb::make_tuple(out, r.interp);
}

nb::tuple fitpack_fit(
    nb::handle xo,
    nb::handle yo,
    nb::handle wo,
    double xb,
    double xe,
    std::int64_t k,
    double s,
    nb::handle to,
    bool periodic
) {
    const Data D = data_args(xo, yo, wo);
    const NDArray& ta = f64_arg(to, "t", 1);
    std::vector<double> t(data_of(ta), data_of(ta) + ta.shape[0]);
    const Layout L = layout(static_cast<std::int64_t>(t.size()), k, periodic);
    int code = 0;
    std::vector<double> c;
    double fp = kNaN;
    {
        const parallel::Unlocked unlocked(D.m * (k + 1) * 8 >= kParallelThreshold);
        const double acc = s * kTol;
        if (static_cast<std::int64_t>(t.size()) == 2 * (k + 1)) {
            LsqOut r = lsq(D, t, k, periodic);
            c = r.c;
            fp = r.fp;
            code = r.ok ? (std::isnan(fp) ? 3 : 0) : 4;
        } else {
            Smoother F;
            F.D = &D;
            F.t = &t;
            F.L = L;
            F.s = s;
            data_triangle(D, t, L, k + 1, F.data);
            F.jumps = disc(t, k, F.rows);
            std::vector<double> cinf;
            const bool ok = solve(F.data, L, D.ncol, cinf);
            if (!ok) {
                code = 4;
            } else {
                std::vector<double> res;
                const double fpinf = residuals(D, t, L, cinf, periodic ? D.m - 1 : D.m, res) - s;
                // fp(0): the least-squares polynomial (no interior knot; periodic: the constant)
                std::vector<double> tc;
                if (periodic) {
                    tc.assign(static_cast<std::size_t>(2 * (k + 1)), 0.0);
                    const double per = xe - xb;
                    for (std::int64_t i = 0; i <= k; ++i) {
                        tc[i] = D.x[0] - static_cast<double>(k - i) * per;
                        tc[i + k + 1] = D.x[D.m - 1] + static_cast<double>(i) * per;
                    }
                } else {
                    for (std::int64_t i = 0; i <= k; ++i) tc.push_back(xb);
                    for (std::int64_t i = 0; i <= k; ++i) tc.push_back(xe);
                }
                const double fp0 = lsq(D, tc, k, periodic).fp - s;
                if (std::isnan(fpinf) || std::isnan(fp0)) {
                    code = 3;
                } else if (fpinf > acc) {
                    code = 1;
                    c = cinf;
                    fp = fpinf + s;
                } else if (fp0 <= acc) {
                    code = 2;
                } else {
                    double diag = 0.0;
                    for (std::int64_t j = 0; j < L.nb + L.kb; ++j) diag += F.data.diag(j);
                    const double p0 = static_cast<double>(L.periodic ? L.nf : L.nc) / diag;
                    const bool conv = root(F, p0, 0.0, fp0, kInf, fpinf, acc);
                    if (F.singular) code = 4;
                    else if (std::isnan(F.fp)) code = 3;
                    else code = conv ? 0 : 5;
                    c = F.c;
                    fp = F.fp;
                }
            }
        }
    }
    if (c.empty()) c.assign(static_cast<std::size_t>(L.nc * D.ncol), kNaN);
    return nb::make_tuple(wrap(mat_array(c, L.nc, D.ncol)), fp, code);
}

nb::tuple fitpack_lsq(nb::handle xo, nb::handle yo, nb::handle wo, nb::handle to, std::int64_t k, bool periodic) {
    const Data D = data_args(xo, yo, wo);
    const NDArray& ta = f64_arg(to, "t", 1);
    std::vector<double> t(data_of(ta), data_of(ta) + ta.shape[0]);
    const LsqOut r = lsq(D, t, k, periodic);
    const Layout L = layout(static_cast<std::int64_t>(t.size()), k, periodic);
    return nb::make_tuple(wrap(mat_array(r.c, L.nc, D.ncol)), wrap(vec_array(r.res)), r.fp);
}

// scipy's chord-length parameters of the points X (m, d): the running sum, in order, of the distances
// sqrt(sum_j (X[i+1, j] - X[i, j])^2) (each sum in column order), over its last value, after a 0 (numpy's sum and
// cumulative_sum do these in the same order for d <= 8, so the parameters, and the knots made of them, are the same).
nb::object fitpack_chord(nb::handle xo) {
    const NDArray& X = f64_arg(xo, "x", 2);
    const double* x = data_of(X);
    const std::int64_t m = X.shape[0], d = X.shape[1];
    std::vector<double> u(static_cast<std::size_t>(m), 0.0);
    double acc = 0.0;
    for (std::int64_t i = 1; i < m; ++i) {
        double s = 0.0;
        for (std::int64_t j = 0; j < d; ++j) {
            const double e = x[i * d + j] - x[(i - 1) * d + j];
            s += e * e;
        }
        acc += std::sqrt(s);
        u[i] = acc;
    }
    for (std::int64_t i = 1; i < m; ++i) u[i] = u[i] / acc;
    return wrap(vec_array(u));
}

double fitpack_fpknot(nb::handle xo, nb::handle to, std::int64_t k, nb::handle ro) {
    const NDArray& xa = f64_arg(xo, "x", 1);
    const NDArray& ta = f64_arg(to, "t", 1);
    const NDArray& ra = f64_arg(ro, "residuals", 1);
    std::vector<double> t(data_of(ta), data_of(ta) + ta.shape[0]);
    return fpknot(data_of(xa), xa.shape[0], t, k, data_of(ra));
}

}  // namespace

void bind_fitpack(nb::module_& m) {
    m.def(
        "fitpack_knots", &fitpack_knots, nb::arg("x").none(), nb::arg("y").none(), nb::arg("w").none(), nb::arg("xb"),
        nb::arg("xe"), nb::arg("k"), nb::arg("s"), nb::arg("nest"), nb::arg("periodic")
    );
    m.def(
        "fitpack_fit", &fitpack_fit, nb::arg("x").none(), nb::arg("y").none(), nb::arg("w").none(), nb::arg("xb"),
        nb::arg("xe"), nb::arg("k"), nb::arg("s"), nb::arg("t").none(), nb::arg("periodic")
    );
    m.def(
        "fitpack_lsq", &fitpack_lsq, nb::arg("x").none(), nb::arg("y").none(), nb::arg("w").none(), nb::arg("t").none(),
        nb::arg("k"), nb::arg("periodic")
    );
    m.def("fitpack_chord", &fitpack_chord, nb::arg("x").none());
    m.def(
        "fitpack_fpknot", &fitpack_fpknot, nb::arg("x").none(), nb::arg("t").none(), nb::arg("k"),
        nb::arg("residuals").none()
    );
}

}  // namespace sd::smooth
