// The objective of minimize (docs/design/optimize.md §3.5): the user's fun and its gradient as
// scipy 1.18.1's ScalarFunction gives them to the BFGS and L-BFGS-B iterations.
//
// ScalarFunction keeps the last point and the function value and gradient there, and computes
// either only when asked for at another point (np.array_equal; NaN never equal, -0.0 equal to
// 0.0) or not yet at this one. Its nfev counts the evaluations of fun, the calls of a difference
// gradient included; ngev the gradients. The gradient is one of:
//   - jac(x), the user's callable;
//   - jac=True: fun returns (f, g), memoized by x as scipy's MemoizeJac (a gradient asked for at
//     another point than fun's last call calls fun again; that call counts as a gradient);
//   - scipy's approx_derivative '2-point' (absolute step eps when jac is None, else the relative
//     step finite_diff_rel_step or sqrt(eps) max(1, |x|)) or '3-point' (relative), adjusted to
//     the bounds (diff.hpp), with f0 the value at the point.
// The user's functions get a new float64 sd.ndarray x; a returned value that is not a Python
// float (fun) or a float64 sd.ndarray of shape (n,) (the gradient) goes through Python's check.
#pragma once

#include "optimize/diff.hpp"

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"

#include <nanobind/nanobind.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sd::optimize {

namespace nb = nanobind;

// A new float64 sd.ndarray (n,) or (rows, cols) from row-major data.
inline nb::object out_vector(const double* v, std::size_t n) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(n)};
    NDArray a = new_array(DType::Float64, shape);
    if (n) std::memcpy(a.data, v, n * sizeof(double));
    return wrap(std::move(a));
}
inline nb::object out_matrix(const double* v, std::size_t rows, std::size_t cols) {
    const std::int64_t shape[2] = {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)};
    NDArray a = new_array(DType::Float64, shape);
    if (rows * cols) std::memcpy(a.data, v, rows * cols * sizeof(double));
    return wrap(std::move(a));
}

enum GradMode : int {
    kGradCallable = 0,    // jac(x)
    kGradTwoPoint = 1,    // '2-point', relative steps
    kGradThreePoint = 2,  // '3-point', relative steps
    kGradAbsolute = 3,    // jac=None: '2-point' with the absolute step eps
    kGradWithFun = 4,     // jac=True: fun returns (f, g)
};

struct Objective {
    using vec = std::vector<double>;

    nb::handle fun, jac;  // jac: the callable gradient (kGradCallable), else None
    nb::handle check;     // check(value, what) -> a float ('fun'), an (n,) array ('jac') or a pair ('pair')
    const char* fn = "";
    std::size_t n = 0;
    int mode = kGradCallable;
    vec steps;  // kGradAbsolute: eps per variable; kGradTwoPoint, kGradThreePoint: rel_step (empty: none)
    vec lb, ub;  // the bounds of the differences (+-inf for none)

    // ScalarFunction's memo
    vec x, g;
    double f = 0;
    bool f_updated = false, g_updated = false;
    // MemoizeJac (kGradWithFun)
    vec mx, mg;
    double mf = 0;
    bool m_valid = false;

    std::int64_t nfev = 0, ngev = 0;
    vec xt;  // scratch: the points of the differences

    nb::object array(const double* v) const { return out_vector(v, n); }

    double real(nb::handle o) const {
        if (PyFloat_CheckExact(o.ptr())) return PyFloat_AS_DOUBLE(o.ptr());
        nb::object c = check(o, "fun");
        return PyFloat_AsDouble(c.ptr());
    }

    // A returned gradient into out (n).
    void vector(nb::handle o, double* out) const {
        nb::object keep;
        if (!(is_ndarray(o) && as_ndarray(o).dtype == DType::Float64 && as_ndarray(o).ndim == 1 &&
              as_ndarray(o).shape[0] == static_cast<std::int64_t>(n))) {
            keep = check(o, "jac");
            o = keep;
        }
        const NDArray& a = as_ndarray(o);
        const std::byte* d = a.data;
        const std::int64_t s = a.strides[0];
        for (std::size_t i = 0; i < n; ++i) std::memcpy(&out[i], d + static_cast<std::int64_t>(i) * s, sizeof(double));
    }

    // One call of the user's fun at v (kGradWithFun: through MemoizeJac).
    double call_f(const double* v) {
        if (mode != kGradWithFun) return real(fun(array(v)));
        memo(v);
        return mf;
    }

    // MemoizeJac._compute_if_needed
    void memo(const double* v) {
        bool same = m_valid;
        for (std::size_t i = 0; same && i < n; ++i) same = v[i] == mx[i];
        if (same) return;
        mx.assign(v, v + n);
        nb::object r = fun(array(v));
        if (!(PyTuple_CheckExact(r.ptr()) && PyTuple_GET_SIZE(r.ptr()) == 2)) r = check(r, "pair");
        nb::handle f0 = PyTuple_GET_ITEM(r.ptr(), 0), g0 = PyTuple_GET_ITEM(r.ptr(), 1);
        mg.resize(n);
        vector(g0, mg.data());
        mf = real(f0);
        m_valid = true;
    }

    // ScalarFunction._update_x: a new point unless it equals the last one.
    void set_x(const double* v) {
        bool same = x.size() == n;
        for (std::size_t i = 0; same && i < n; ++i) same = v[i] == x[i];
        if (same) return;
        x.assign(v, v + n);
        f_updated = g_updated = false;
    }

    void update_fun() {
        if (f_updated) return;
        f = call_f(x.data());
        ++nfev;
        f_updated = true;
    }

    void update_grad() {
        if (g_updated) return;
        g.resize(n);
        if (mode == kGradCallable) {
            vector(jac(array(x.data())), g.data());
        } else if (mode == kGradWithFun) {
            memo(x.data());
            g = mg;
        } else {
            update_fun();
            differences();
        }
        ++ngev;
        g_updated = true;
    }

    // ScalarFunction's initial evaluations at x0.
    void start(const double* x0) {
        x.assign(x0, x0 + n);
        f_updated = g_updated = false;
        update_fun();
        update_grad();
    }

    double value(const double* v) {
        set_x(v);
        update_fun();
        return f;
    }
    const vec& gradient(const double* v) {
        set_x(v);
        update_grad();
        return g;
    }
    void value_and_gradient(const double* v) {
        set_x(v);
        update_fun();
        update_grad();
    }

    // approx_derivative of fun at x with f0 = f.
    void differences() {
        xt = x;
        const double f0 = f;
        if (mode == kGradThreePoint) {
            const double rstep = rel_step_3point();
            for (std::size_t j = 0; j < n; ++j) {
                const double t = x[j];
                const double h0 = absolute_step(t, rstep, !steps.empty(), steps.empty() ? 0.0 : steps[j]);
                bool one_sided;
                const double h = adjust_two_sided(t, h0, lb[j], ub[j], one_sided);
                const double x1 = one_sided ? t + h : t - h, x2 = one_sided ? t + 2 * h : t + h;
                xt[j] = x1;
                const double f1 = call_f(xt.data());
                xt[j] = x2;
                const double f2 = call_f(xt.data());
                xt[j] = t;
                g[j] = one_sided ? (-3.0 * f0 + 4 * f1 - f2) / (x2 - t) : (f2 - f1) / (x2 - x1);
            }
            nfev += 2 * static_cast<std::int64_t>(n);
            return;
        }
        const double rstep = rel_step_2point();
        for (std::size_t j = 0; j < n; ++j) {
            const double t = x[j];
            double h;
            if (mode == kGradAbsolute) {
                // approx_derivative's abs_step: eps, unless it vanishes in x + h
                h = steps[j];
                if ((t + h) - t == 0) h = rstep * (t >= 0 ? 1.0 : -1.0) * std::max(1.0, std::abs(t));
            } else {
                h = absolute_step(t, rstep, !steps.empty(), steps.empty() ? 0.0 : steps[j]);
            }
            h = adjust_one_sided(t, h, lb[j], ub[j]);
            xt[j] = t + h;
            const double f1 = call_f(xt.data());
            xt[j] = t;
            g[j] = (f1 - f0) / ((t + h) - t);
        }
        nfev += static_cast<std::int64_t>(n);
    }

    // The checks at x0 (O53): a non-finite value or gradient there is a ValueError.
    void require_finite_start() const {
        if (!std::isfinite(f)) throw nb::value_error("'fun' is not finite at 'x0'");
        for (double v : g)
            if (!std::isfinite(v)) {
                if (mode == kGradCallable || mode == kGradWithFun) throw nb::value_error("'jac' is not finite at 'x0'");
                throw nb::value_error("the differences of 'fun' are not finite at 'x0'");
            }
    }
};

}  // namespace sd::optimize
