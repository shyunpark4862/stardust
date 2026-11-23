// lowess (docs/design/smooth.md §3.1, §3.8; decisions SM27, SM36-SM42): the robust iterations, delta's skipping
// and interpolation, the hat diagonal and the evaluation at new points, on the sorted-window local line of
// smooth/local.hpp.
//
// _core.smooth.lowess_fit(x, y, w, k, it, delta) -> (fitted, leverage, robust_weights, df, rss, xs, ys, ws, rho)
//     x, y (n,) and w (n,) or None: finite float64, C-contiguous, in the caller's order (NaN already dropped),
//     1 <= k <= n, it >= 0, delta >= 0 finite. The data are sorted by (x, y, w, index) (decision SM36), so the
//     result does not depend on the input order. fitted, leverage and robust_weights come back in the input order
//     (robust_weights: those of the last fit, all 1 for it = 0); df = sum of the leverages of the points of positive
//     weight, rss = sum w r^2, both summed in sorted order; xs, ys, ws (None without weights) and rho are the sorted
//     data and robust weights a result keeps for predict.
// _core.smooth.lowess_predict(xs, ys, ws, rho, k, x_new, norms) -> (values, norms or None)
//     the local line at each point of x_new (m,) with the prior weights rho w; NaN where the window has fewer than
//     two positive weights (or x_new is NaN); norms: sum l_j^2 / w_j of each hat row.
//
// Every fitted point is one call of fit_line, whatever task computes it: the results have the same bits for every
// thread count (decision SM3). The robust step (exact median, bisquare) and delta's plan are serial and O(n).

#include "smooth/local.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "smooth/smooth.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::smooth {

namespace nb = nanobind;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const NDArray& vector_arg(nb::handle o, const char* what, std::int64_t n = -1) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous 1-D float64 array").c_str());
    if (n >= 0 && a.shape[0] != n) throw nb::value_error(("'" + std::string(what) + "' has the wrong length").c_str());
    return a;
}

const double* data_of(const NDArray& a) { return reinterpret_cast<const double*>(a.data); }

NDArray vector_out(std::int64_t n) {
    const std::int64_t shape[1] = {n};
    return new_array(DType::Float64, shape);
}

double* out_of(NDArray& a) { return reinterpret_cast<double*>(a.data); }

// Runs fn(i) for i in [0, count), each about `per_point` window points of work, split by size alone.
template <class F>
void for_points(std::int64_t count, std::int64_t per_point, const F& fn) {
    if (count == 0) return;
    const std::int64_t cost = std::max<std::int64_t>(1, per_point) * 8;  // ~8 adds' worth per window point
    const std::int64_t work = count * cost;
    if (work >= kParallelThreshold && count > 1 && parallel::available()) {
        const std::int64_t per = std::max<std::int64_t>(1, (std::int64_t{1} << 17) / cost);
        const std::int64_t tasks = (count + per - 1) / per;
        parallel::parallel_for(
            tasks,
            [&](std::int64_t t) {
                const std::int64_t end = std::min(count, (t + 1) * per);
                for (std::int64_t i = t * per; i < end; ++i) fn(i);
            },
            work_ns(per * cost)
        );
    } else {
        const parallel::Unlocked unlocked(work >= kParallelThreshold);
        for (std::int64_t i = 0; i < count; ++i) fn(i);
    }
}

// delta's plan (statsmodels' update_indices and interpolate_skipped_fits, which R's lowess shares): the points to
// fit, and for every other point either a copy of an earlier tied point or the interpolation between two fits.
struct Plan {
    std::vector<std::int64_t> fit;    // sorted indices fitted
    std::vector<std::int64_t> src;    // per point: the fitted point it copies (itself when fitted), or -1
    std::vector<std::int64_t> left;   // interpolated points: the fits on either side
    std::vector<std::int64_t> right;  //
    std::vector<double> alpha;        // value = alpha fit[right] + (1 - alpha) fit[left]
};

Plan make_plan(const double* x, std::int64_t n, double delta) {
    Plan p;
    p.src.assign(n, -1);
    p.left.assign(n, -1);
    p.right.assign(n, -1);
    p.alpha.assign(n, 0.0);
    if (delta == 0) {  // every point fitted (ties give the same value at the same x0)
        p.fit.resize(n);
        std::iota(p.fit.begin(), p.fit.end(), std::int64_t{0});
        std::iota(p.src.begin(), p.src.end(), std::int64_t{0});
        return p;
    }
    std::int64_t i = 0, last = -1;
    std::vector<std::int64_t> root(n, -1);  // the fitted point whose value a point holds (fits and copies)
    while (true) {
        p.fit.push_back(i);
        p.src[i] = i, root[i] = i;
        if (last < i - 1) {
            const double denom = x[i] - x[last];
            for (std::int64_t j = last + 1; j < i; ++j) {
                p.left[j] = root[last], p.right[j] = i;
                p.alpha[j] = (x[j] - x[last]) / denom;
            }
        }
        last = i;
        std::int64_t kpy = last;  // Python's loop variable after `for k in range(last + 1, n)`
        const double cut = x[last] + delta;
        for (std::int64_t k = last + 1; k < n; ++k) {
            kpy = k;
            if (x[k] > cut) break;
            if (x[k] == x[last]) {
                p.src[k] = root[last], root[k] = root[last];
                last = k;
            }
        }
        i = std::max(kpy - 1, last + 1);
        if (last >= n - 1) break;
    }
    return p;
}

// The robust weights of residuals r (statsmodels' calculate_residual_weights): m = median |r_j| over the points of
// positive weight (exact order statistic, the two middle values' mean for an even count), bisquare (1 - u^2)^2 of
// u = min(1, |r| / (6 m)); m = 0: 1 where r = 0, else 0.
void robust_weights(const double* r, const double* w, std::int64_t n, double* rho, std::vector<double>& buf) {
    buf.clear();
    for (std::int64_t j = 0; j < n; ++j)
        if (!w || w[j] > 0) buf.push_back(std::fabs(r[j]));
    const std::size_t c = buf.size(), mid = c / 2;
    std::nth_element(buf.begin(), buf.begin() + mid, buf.end());
    double m = buf[mid];
    if (c % 2 == 0) {
        const double lower = *std::max_element(buf.begin(), buf.begin() + mid);
        const double s = lower + m;
        m = std::isfinite(s) ? s / 2 : lower / 2 + m / 2;
    }
    if (m == 0) {
        for (std::int64_t j = 0; j < n; ++j) rho[j] = std::fabs(r[j]) > 0 ? 0.0 : 1.0;
        return;
    }
    const double s6 = 6.0 * m;
    for (std::int64_t j = 0; j < n; ++j) {
        double u = std::fabs(r[j]) / s6;
        if (u > 1) u = 1.0;
        const double t = 1.0 - u * u;
        rho[j] = t * t;
    }
}

nb::object lowess_fit(
    nb::handle x_obj, nb::handle y_obj, nb::handle w_obj, std::int64_t k, std::int64_t it, double delta
) {
    const NDArray& xa = vector_arg(x_obj, "x");
    const std::int64_t n = xa.shape[0];
    const NDArray& ya = vector_arg(y_obj, "y", n);
    const NDArray* wa = w_obj.is_none() ? nullptr : &vector_arg(w_obj, "weights", n);
    if (n < 1 || k < 1 || k > n) throw nb::value_error("'k' must be in [1, n]");
    if (it < 0) throw nb::value_error("'it' must be >= 0");
    if (!(delta >= 0) || !std::isfinite(delta)) throw nb::value_error("'delta' must be a finite number >= 0");
    const double *x = data_of(xa), *y = data_of(ya), *w = wa ? data_of(*wa) : nullptr;

    NDArray fitted_o = vector_out(n), lev_o = vector_out(n), rho_o = vector_out(n);
    NDArray xs_o = vector_out(n), ys_o = vector_out(n), rhos_o = vector_out(n);
    NDArray ws_o = wa ? vector_out(n) : NDArray{};
    double df = 0, rss = 0;
    {
        const parallel::Unlocked unlocked(n >= 4096);
        std::vector<std::int64_t> order(n);
        std::iota(order.begin(), order.end(), std::int64_t{0});
        std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
            if (x[a] != x[b]) return x[a] < x[b];
            if (y[a] != y[b]) return y[a] < y[b];
            if (w && w[a] != w[b]) return w[a] < w[b];
            return a < b;
        });
        double *xs = out_of(xs_o), *ys = out_of(ys_o), *rho = out_of(rhos_o);
        double* ws = wa ? out_of(ws_o) : nullptr;
        for (std::int64_t j = 0; j < n; ++j) {
            xs[j] = x[order[j]], ys[j] = y[order[j]], rho[j] = 1.0;
            if (ws) ws[j] = w[order[j]];
        }
        std::vector<double> pw(n), fit(n), resid(n), buf;
        const Plan plan = make_plan(xs, n, delta);
        const std::int64_t nf = static_cast<std::int64_t>(plan.fit.size());
        std::vector<LineFit> fits(n);
        const Sorted1D D{xs, ys, pw.data(), n};
        for (std::int64_t step = 0; step <= it; ++step) {
            for (std::int64_t j = 0; j < n; ++j) pw[j] = ws ? rho[j] * ws[j] : rho[j];
            for_points(nf, k, [&](std::int64_t q) {
                const std::int64_t i = plan.fit[q];
                LineFit F = fit_line(D, k, xs[i], i);
                if (F.kind == LineKind::Few) F.value = ys[i];  // statsmodels' rule: the point's own y
                fits[i] = F;
            });
            for (std::int64_t j = 0; j < n; ++j) {
                if (plan.src[j] >= 0) fit[j] = fits[plan.src[j]].value;
                else {
                    const double a = plan.alpha[j];
                    fit[j] = a * fits[plan.right[j]].value + (1.0 - a) * fits[plan.left[j]].value;
                }
            }
            if (step == it) break;
            for (std::int64_t j = 0; j < n; ++j) resid[j] = ys[j] - fit[j];
            robust_weights(resid.data(), ws, n, rho, buf);
        }
        double *fo = out_of(fitted_o), *lo = out_of(lev_o), *ro = out_of(rho_o);
        for (std::int64_t j = 0; j < n; ++j) {
            double l;
            if (ws && !(ws[j] > 0)) l = 0.0;  // out of the fit, as a row of weight 0 in a regression
            else if (plan.src[j] >= 0) l = hat_at(D, fits[plan.src[j]], j);
            else {
                const double a = plan.alpha[j];
                l = a * hat_at(D, fits[plan.right[j]], j) + (1.0 - a) * hat_at(D, fits[plan.left[j]], j);
            }
            const double r = ys[j] - fit[j];
            df += l;
            rss += ws ? ws[j] * r * r : r * r;
            const std::int64_t o = order[j];
            fo[o] = fit[j], lo[o] = l, ro[o] = rho[j];
        }
    }
    nb::object ws_obj = wa ? wrap(std::move(ws_o)) : nb::none();
    return nb::make_tuple(
        wrap(std::move(fitted_o)), wrap(std::move(lev_o)), wrap(std::move(rho_o)), df, rss, wrap(std::move(xs_o)),
        wrap(std::move(ys_o)), ws_obj, wrap(std::move(rhos_o))
    );
}

nb::object lowess_predict(
    nb::handle xs_obj,
    nb::handle ys_obj,
    nb::handle ws_obj,
    nb::handle rho_obj,
    std::int64_t k,
    nb::handle q_obj,
    bool norms
) {
    const NDArray& xa = vector_arg(xs_obj, "xs");
    const std::int64_t n = xa.shape[0];
    const NDArray& ya = vector_arg(ys_obj, "ys", n);
    const NDArray* wa = ws_obj.is_none() ? nullptr : &vector_arg(ws_obj, "ws", n);
    const NDArray& ra = vector_arg(rho_obj, "rho", n);
    const NDArray& qa = vector_arg(q_obj, "x_new");
    if (n < 1 || k < 1 || k > n) throw nb::value_error("'k' must be in [1, n]");
    const std::int64_t m = qa.shape[0];
    NDArray val_o = vector_out(m);
    NDArray norm_o = norms ? vector_out(m) : NDArray{};
    {
        const parallel::Unlocked unlocked(m * k >= kParallelThreshold);
        const double *xs = data_of(xa), *ys = data_of(ya), *ws = wa ? data_of(*wa) : nullptr, *rho = data_of(ra);
        const double* q = data_of(qa);
        std::vector<double> pw(n);
        for (std::int64_t j = 0; j < n; ++j) pw[j] = ws ? rho[j] * ws[j] : rho[j];
        const Sorted1D D{xs, ys, pw.data(), n};
        double* vo = out_of(val_o);
        double* no = norms ? out_of(norm_o) : nullptr;
        for_points(m, k, [&](std::int64_t i) {
            const LineFit F = fit_line(D, k, q[i]);
            vo[i] = F.kind == LineKind::Few ? kNaN : F.value;
            if (no) no[i] = hat_norm2(D, F, ws);
        });
    }
    nb::object n_obj = norms ? wrap(std::move(norm_o)) : nb::none();
    return nb::make_tuple(wrap(std::move(val_o)), n_obj);
}

}  // namespace

void bind_local(nb::module_& m) {
    m.def(
        "lowess_fit", named("lowess_fit", &lowess_fit), nb::arg("x").none(), nb::arg("y").none(), nb::arg("w").none(),
        nb::arg("k"), nb::arg("it"), nb::arg("delta")
    );  // fmt: skip
    m.def(
        "lowess_predict", named("lowess_predict", &lowess_predict), nb::arg("xs").none(), nb::arg("ys").none(),
        nb::arg("ws").none(), nb::arg("rho").none(), nb::arg("k"), nb::arg("x_new").none(), nb::arg("norms")
    );  // fmt: skip
}

}  // namespace sd::smooth
