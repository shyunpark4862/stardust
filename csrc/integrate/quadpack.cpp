// quad's routes without a weight and the multiple integrals (docs/design/integrate.md §3.1, §3.2,
// decisions C5, C55-C66): scipy 1.18.1's dispatch around QUADPACK's dqagse, dqagie and dqagpe
// (integrate/quadpack.hpp), with the user's function a Python callback (the GIL held).
//
//   - quad_solve: one quad. scipy's _quadpack_py.quad: an empty interval is 0 without a call,
//     reversed limits are swapped and the sign changed, break points (np.unique, those strictly
//     inside) take QAGP, an infinite limit QAGI, the rest QAGS. part 0 takes a real value, 1 and 2
//     the real and the imaginary part of a complex one (complex_func=True: scipy's two quads).
//     The intervals come back as three read-only arrays (their ends (last, 2), integrals and
//     errors) from the largest error down: QUADPACK's iord order for the kmax intervals it ranks, the others (when
//     last > limit/2 + 2) by decreasing error, ties in their order of creation.
//   - nquad_solve: scipy's _NQuad: the innermost variable's quad inside the next one's, each
//     level's range and options computed (by the user's functions if given) at every point of
//     the levels outside it, the innermost function called with all the variables (innermost
//     first). The error is the largest of every quad's (scipy's self.abserr), nfev the innermost
//     quads' evaluations (scipy's neval); the first quad that does not converge, in the order the
//     quads finish, gives the status.
#include "integrate/integrate.hpp"
#include "integrate/quadpack.hpp"

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace sd::integrate {

namespace nb = nanobind;
namespace qp = quadpack;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The user's integrand at one Python float. A value that is not a Python float (part 0) or a
// Python float or complex (parts 1, 2) goes through check(value, 'func'), which gives a float
// (part 0) or a complex, or raises.
struct Fn {
    nb::handle f, check;
    int part = 0;

    double value(nb::handle r) const {
        PyObject* o = r.ptr();
        if (part == 0) {
            if (PyFloat_CheckExact(o)) return PyFloat_AS_DOUBLE(o);
            nb::object c = check(r, "func");
            return PyFloat_AsDouble(c.ptr());
        }
        if (PyComplex_CheckExact(o)) return part == 1 ? PyComplex_RealAsDouble(o) : PyComplex_ImagAsDouble(o);
        if (PyFloat_CheckExact(o)) return part == 1 ? PyFloat_AS_DOUBLE(o) : 0.0;
        nb::object c = check(r, "func");
        return part == 1 ? PyComplex_RealAsDouble(c.ptr()) : PyComplex_ImagAsDouble(c.ptr());
    }
    double operator()(double x) const { return value(f(x)); }
};

// One quad's options (stardust/integrate/_quadpack.py checks them).
struct Opts {
    double atol = 1.49e-8, rtol = 1.49e-8;
    int limit = 50;
    bool has_points = false;
    std::vector<double> points;
};

Opts read_opts(nb::handle t) {
    Opts o;
    nb::tuple tu = nb::borrow<nb::tuple>(t);
    o.atol = nb::cast<double>(tu[0]);
    o.rtol = nb::cast<double>(tu[1]);
    o.limit = nb::cast<int>(tu[2]);
    if (!tu[3].is_none()) {
        o.has_points = true;
        o.points = nb::cast<std::vector<double>>(tu[3]);
    }
    return o;
}

// scipy's quad on [a, b] (a, b not NaN): w holds QUADPACK's results; the value (its sign changed
// for reversed limits) is returned.
template <class F>
double quad(F& f, double a, double b, const Opts& o, qp::Work& w, int& limit_used) {
    limit_used = o.limit;
    if (a == b) {  // scipy's shortcut, before anything else
        w.reset(1);
        return 0.0;
    }
    const bool flip = b < a;
    if (flip) std::swap(a, b);
    if (o.has_points) {
        if (std::isinf(a) || std::isinf(b)) throw nb::value_error("'points' cannot be used with an infinite range");
        std::vector<double> p = o.points;  // np.unique, then a < p < b
        std::sort(p.begin(), p.end());
        p.erase(std::unique(p.begin(), p.end()), p.end());
        std::vector<double> in;
        for (double t : p)
            if (a < t && t < b) in.push_back(t);
        if (static_cast<std::int64_t>(in.size()) >= o.limit)
            throw nb::value_error(("the number of break points (" + std::to_string(in.size()) +
                                   ") must be less than 'limit' (" + std::to_string(o.limit) + ")")
                                      .c_str());
        qp::qagpe(f, a, b, in.data(), static_cast<int>(in.size()), o.atol, o.rtol, o.limit, w);
    } else if (b == kInf && a != -kInf) {
        qp::qagie(f, a, 1, o.atol, o.rtol, o.limit, w);
    } else if (b == kInf) {
        qp::qagie(f, 0.0, 2, o.atol, o.rtol, o.limit, w);
    } else if (a == -kInf) {
        qp::qagie(f, b, -1, o.atol, o.rtol, o.limit, w);
    } else {
        qp::qagse(f, a, b, o.atol, o.rtol, o.limit, w);
    }
    return flip ? -w.result : w.result;
}

// The intervals as three read-only float64 arrays (the result's fields: (last, 2) ends, (last,)
// integrals and errors), from the largest error down (the comment at the top).
nb::object readonly(NDArray&& a) {
    a.writeable = false;
    return wrap(std::move(a));
}

nb::object interval_arrays(const qp::Work& w, int limit) {
    const int last = w.last;
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(last));
    if (last == 1) {
        order.push_back(1);
    } else if (last > 1) {
        const int kmax = last <= limit / 2 + 2 ? last : (limit - last) + 1;  // QUADPACK's kmax
        std::vector<char> ranked(static_cast<std::size_t>(last) + 1, 0);
        for (int k = 1; k <= kmax && k <= last; ++k) {
            const int i = w.iord[k];
            if (i < 1 || i > last || ranked[i]) continue;
            ranked[i] = 1;
            order.push_back(i);
        }
        const auto n_ranked = static_cast<std::ptrdiff_t>(order.size());
        for (int i = 1; i <= last; ++i)
            if (!ranked[i]) order.push_back(i);
        std::stable_sort(order.begin() + n_ranked, order.end(), [&](int x, int y) { return w.elist[x] > w.elist[y]; });
    }
    const std::int64_t s2[2] = {last, 2}, s1[1] = {last};
    NDArray ends = new_array(DType::Float64, s2), vals = new_array(DType::Float64, s1),
            errs = new_array(DType::Float64, s1);
    auto* e = reinterpret_cast<double*>(ends.data);
    auto* v = reinterpret_cast<double*>(vals.data);
    auto* r = reinterpret_cast<double*>(errs.data);
    for (std::size_t k = 0; k < order.size(); ++k) {
        const int i = order[k];
        e[2 * k] = w.alist[i];
        e[2 * k + 1] = w.blist[i];
        v[k] = w.rlist[i];
        r[k] = w.elist[i];
    }
    return nb::make_tuple(readonly(std::move(ends)), readonly(std::move(vals)), readonly(std::move(errs)));
}

nb::object quad_solve(
    const char* fn, nb::callable f, nb::callable check, double a, double b, nb::handle opts, int part
) {
    return call_named(fn, [&]() -> nb::object {
        Fn F{f, check, part};
        const Opts o = read_opts(opts);
        qp::Work w;
        int limit = 0;
        const double r = quad(F, a, b, o, w, limit);
        return nb::make_tuple(r, w.abserr, w.neval, w.ier, w.last, interval_arrays(w, o.limit));
    });
}

// ---- nquad

struct NQuad {
    struct Level {
        nb::object range, opts;  // a callable, or the checked constant
        bool range_fn = false, opts_fn = false;
        double lo = 0.0, hi = 0.0;
        Opts fixed;
    };

    nb::handle func, check, range_check, opts_check;
    std::vector<Level> levels;  // 0 innermost
    std::vector<double> xs;     // the variables, innermost first
    std::vector<qp::Work> works;
    double abserr = 0.0;
    std::int64_t nfev = 0;
    int bad_ier = 0, bad_limit = 0;

    // the variables of the levels outside i (nearest first), as scipy passes them (*args)
    nb::tuple outer(int i) const {
        const auto n = static_cast<Py_ssize_t>(xs.size()) - (i + 1);
        PyObject* t = PyTuple_New(n);
        if (!t) throw nb::python_error();
        for (Py_ssize_t k = 0; k < n; ++k) PyTuple_SET_ITEM(t, k, PyFloat_FromDouble(xs[i + 1 + k]));
        return nb::steal<nb::tuple>(t);
    }

    double innermost() const {
        const auto n = static_cast<Py_ssize_t>(xs.size());
        PyObject* t = PyTuple_New(n);
        if (!t) throw nb::python_error();
        for (Py_ssize_t k = 0; k < n; ++k) PyTuple_SET_ITEM(t, k, PyFloat_FromDouble(xs[k]));
        nb::object args = nb::steal(t);
        PyObject* r = PyObject_Call(func.ptr(), args.ptr(), nullptr);
        if (!r) throw nb::python_error();
        nb::object v = nb::steal(r);
        if (PyFloat_CheckExact(r)) return PyFloat_AS_DOUBLE(r);
        nb::object c = check(v, "func");
        return PyFloat_AsDouble(c.ptr());
    }

    // the integral over level i's variable, the outer ones set
    double integrate(int i) {
        Level& L = levels[static_cast<std::size_t>(i)];
        double lo = L.lo, hi = L.hi;
        if (L.range_fn) {
            nb::tuple args = outer(i);
            nb::object r = nb::steal(PyObject_Call(L.range.ptr(), args.ptr(), nullptr));
            if (!r.is_valid()) throw nb::python_error();
            nb::tuple p = nb::borrow<nb::tuple>(range_check(i, r));
            lo = nb::cast<double>(p[0]);
            hi = nb::cast<double>(p[1]);
        }
        Opts o;
        const Opts* op = &L.fixed;
        if (L.opts_fn) {
            nb::tuple args = outer(i);
            nb::object r = nb::steal(PyObject_Call(L.opts.ptr(), args.ptr(), nullptr));
            if (!r.is_valid()) throw nb::python_error();
            o = read_opts(opts_check(i, r));
            op = &o;
        }
        qp::Work& w = works[static_cast<std::size_t>(i)];
        int limit = 0;
        double v;
        if (i == 0) {
            auto f = [&](double x) {
                xs[0] = x;
                return innermost();
            };
            v = quad(f, lo, hi, *op, w, limit);
            nfev += w.neval;
        } else {
            auto f = [&, i](double x) {
                xs[static_cast<std::size_t>(i)] = x;
                return integrate(i - 1);
            };
            v = quad(f, lo, hi, *op, w, limit);
        }
        if (w.abserr > abserr) abserr = w.abserr;  // Python's max(self.abserr, abserr)
        if (w.ier != 0 && bad_ier == 0) {
            bad_ier = w.ier;
            bad_limit = limit;
        }
        return v;
    }
};

nb::object nquad_solve(
    const char* fn,
    nb::callable func,
    nb::callable check,
    nb::list levels,
    nb::callable range_check,
    nb::callable opts_check
) {
    return call_named(fn, [&]() -> nb::object {
        NQuad q;
        q.func = func;
        q.check = check;
        q.range_check = range_check;
        q.opts_check = opts_check;
        for (nb::handle lv : levels) {
            nb::tuple t = nb::borrow<nb::tuple>(lv);
            NQuad::Level L;
            nb::object rng = t[0], opt = t[1];
            if (PyCallable_Check(rng.ptr())) {
                L.range_fn = true;
                L.range = rng;
            } else {
                nb::tuple p = nb::borrow<nb::tuple>(rng);
                L.lo = nb::cast<double>(p[0]);
                L.hi = nb::cast<double>(p[1]);
            }
            if (PyCallable_Check(opt.ptr())) {
                L.opts_fn = true;
                L.opts = opt;
            } else {
                L.fixed = read_opts(opt);
            }
            q.levels.push_back(std::move(L));
        }
        const int n = static_cast<int>(q.levels.size());
        q.xs.assign(static_cast<std::size_t>(n), 0.0);
        q.works.resize(static_cast<std::size_t>(n));
        const double v = q.integrate(n - 1);
        return nb::make_tuple(v, q.abserr, q.nfev, q.bad_ier, q.bad_limit);
    });
}

}  // namespace

void bind_quadpack(nb::module_& m) {
    m.def(
        "quad_solve", &quad_solve, nb::arg("fn"), nb::arg("f"), nb::arg("check"), nb::arg("a"), nb::arg("b"),
        nb::arg("opts"), nb::arg("part")
    );
    m.def(
        "nquad_solve", &nquad_solve, nb::arg("fn"), nb::arg("func"), nb::arg("check"), nb::arg("levels"),
        nb::arg("range_check"), nb::arg("opts_check")
    );
}

}  // namespace sd::integrate
