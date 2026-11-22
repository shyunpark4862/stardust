// quad's routes without a weight and the multiple integrals (docs/design/integrate.md §3.1, §3.2,
// decisions C5, C55-C66): scipy 1.18.1's dispatch around QUADPACK's dqagse, dqagie and dqagpe
// (integrate/quadpack.hpp), with the user's function a Python callback (the GIL held).
//
//   - quad_solve: one quad. scipy's _quadpack_py.quad: an empty interval is 0 without a call,
//     reversed limits are swapped and the sign changed, break points (np.unique, those strictly
//     inside) take QAGP, an infinite limit QAGI, the rest QAGS; a weight (scipy's _quad_weight,
//     milestone C5) takes QAWO (cos, sin on a finite range), QAWF (cos, sin on [a, inf); (-inf, b]
//     by x -> -x), QAWS (the algebraic-logarithmic end point weights) or QAWC (Cauchy). part 0
//     takes a real value, 1 and 2 the real and the imaginary part of a complex one
//     (complex_func=True: scipy's two quads). The intervals come back as three read-only arrays
//     (their ends (last, 2), integrals and errors) from the largest error down: QUADPACK's iord
//     order for the kmax intervals it ranks, the others (when last > limit/2 + 2) by decreasing
//     error, ties in their order of creation; QAWF's cycles as their integrals, errors and ier
//     (scipy's rslst, erlst, ierlst) in order.
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

// quad's vectorized=True (decision C127): the points of one rule application in one call, a new
// float64 sd.ndarray (n,), the values read from what f returns (a float64 sd.ndarray of that
// shape as it is, anything else through check(value, n), which gives one or raises; parts 1 and
// 2 take a complex128 array, or a float64 one as its real part). prefetch() makes the call and
// the rule's calls of operator() take the values in order; a point that is not the next one
// listed (none in QUADPACK's rules) is evaluated alone, as an array of one.
struct VecFn {
    nb::handle f, check;
    int part = 0;
    std::vector<double> xs, vals;
    std::size_t pos = 0;

    void prefetch(const double* x, int n) {
        xs.assign(x, x + n);
        vals.resize(static_cast<std::size_t>(n));
        pos = 0;
        eval(x, n, vals.data());
    }

    void eval(const double* x, int n, double* out) const {
        const std::int64_t shape[1] = {n};
        NDArray a = new_array(DType::Float64, shape);
        if (n) std::memcpy(a.data, x, static_cast<std::size_t>(n) * sizeof(double));
        nb::object r = f(wrap(std::move(a)));
        const DType want = part == 0 ? DType::Float64 : DType::Complex128;
        auto fits = [&](nb::handle h) {
            if (!is_ndarray(h)) return false;
            const NDArray& v = as_ndarray(h);
            return v.ndim == 1 && v.shape[0] == n && (v.dtype == want || (part != 0 && v.dtype == DType::Float64));
        };
        if (!fits(r)) r = check(r, n);
        const NDArray& v = as_ndarray(r);
        const std::int64_t s = v.strides[0];
        if (v.dtype == DType::Float64) {
            for (int i = 0; i < n; ++i) {
                double t;
                std::memcpy(&t, v.data + i * s, sizeof t);
                out[i] = part == 2 ? 0.0 : t;
            }
            return;
        }
        for (int i = 0; i < n; ++i) {
            double t[2];
            std::memcpy(t, v.data + i * s, sizeof t);
            out[i] = part == 1 ? t[0] : t[1];
        }
    }

    double operator()(double x) {
        if (pos < xs.size() && std::memcmp(&xs[pos], &x, sizeof x) == 0) return vals[pos++];
        double v;
        eval(&x, 1, &v);
        return v;
    }
};

// QAWF on (-inf, b] by x -> -x: sign f(-x), its points handed on negated.
template <class F>
struct Reflected {
    F& f;
    double sign;
    double operator()(double x) { return sign * f(-x); }
    void prefetch(const double* x, int n) {
        std::vector<double> neg(x, x + n);
        for (double& t : neg) t = -t;
        qp::prefetch(f, neg.data(), n);
    }
};

constexpr double kTol = 1.49e-8;  // scipy's epsabs and epsrel
constexpr int kMaxp1 = 50, kLimlst = 50;

// quad's weights (stardust/integrate/_quadpack.py's codes)
enum Weight { kNone = 0, kCos, kSin, kAlg, kAlgLoga, kAlgLogb, kAlgLog, kCauchy };

// One quad's options (stardust/integrate/_quadpack.py checks them; what depends on the range is
// checked here).
struct Opts {
    double atol = kTol, rtol = kTol;
    int limit = 50;
    bool has_points = false;
    std::vector<double> points;
    int weight = kNone;
    double wvar0 = 0.0, wvar1 = 0.0;  // omega or c; (alpha, beta)
    int max_p1 = kMaxp1, limlst = kLimlst;
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
    o.weight = nb::cast<int>(tu[4]);
    if (o.weight == kAlg || o.weight == kAlgLoga || o.weight == kAlgLogb || o.weight == kAlgLog) {
        nb::tuple w = nb::borrow<nb::tuple>(tu[5]);
        o.wvar0 = nb::cast<double>(w[0]);
        o.wvar1 = nb::cast<double>(w[1]);
    } else if (o.weight != kNone) {
        o.wvar0 = nb::cast<double>(tu[5]);
    }
    o.max_p1 = nb::cast<int>(tu[6]);
    o.limlst = nb::cast<int>(tu[7]);
    return o;
}

const char* weight_name(int w) {
    static const char* const names[] = {"", "cos", "sin", "alg", "alg-loga", "alg-logb", "alg-log", "cauchy"};
    return names[w];
}


// The weighted routes (scipy's _quad_weight) on [a, b], a < b, after quad's swap: QAWO, QAWF
// (qawf true: r holds the cycles), QAWS, QAWC.
template <class F>
void quad_weight(F& f, double a, double b, const Opts& o, qp::Work& w, qp::Cycles& r, bool& qawf) {
    const std::string name = std::string("weight='") + weight_name(o.weight) + "'";
    const bool finite = !std::isinf(a) && !std::isinf(b);
    qawf = false;
    if (o.weight == kCos || o.weight == kSin) {
        const int integr = o.weight == kCos ? 1 : 2;
        if (finite) {
            if (o.limlst != kLimlst)
                throw nb::value_error("'limlst' is only used with weight='cos' or 'sin' on an infinite range");
            qp::qawoe(f, a, b, o.wvar0, integr, o.atol, o.rtol, o.limit, 1, o.max_p1, w);
            if (w.last == 0) w.last = 1;  // dqawoe ends on its first rule with last = 0: its one interval
            return;
        }
        if (std::isinf(a) && std::isinf(b)) throw nb::value_error((name + " cannot be used on (-inf, inf)").c_str());
        if (o.atol <= 0.0)
            throw nb::value_error(
                ("'atol' must be > 0 with " + name + " on an infinite range (QAWF uses 'atol' only)").c_str()
            );
        if (o.rtol != kTol)
            throw nb::value_error(
                ("'rtol' is not used with " + name + " on an infinite range (QAWF uses 'atol' only)").c_str()
            );
        qawf = true;
        if (!std::isinf(a)) {
            qp::qawfe(f, a, o.wvar0, integr, o.atol, o.limlst, o.limit, o.max_p1, w, r);
        } else {  // (-inf, b]: scipy's x -> -x, f(-x) for the cosine and -f(-x) for the sine
            const double sign = integr == 1 ? 1.0 : -1.0;
            Reflected<F> g{f, sign};
            qp::qawfe(g, -b, o.wvar0, integr, o.atol, o.limlst, o.limit, o.max_p1, w, r);
        }
        return;
    }
    if (!finite) throw nb::value_error((name + " cannot be used with an infinite range").c_str());
    if (o.weight == kCauchy) {
        if (o.wvar0 == a || o.wvar0 == b)
            throw nb::value_error("'wvar' must not equal 'a' or 'b' with weight='cauchy'");
        qp::qawce(f, a, b, o.wvar0, o.atol, o.rtol, o.limit, w);
        return;
    }
    qp::qawse(f, a, b, o.wvar0, o.wvar1, o.weight - kAlg + 1, o.atol, o.rtol, o.limit, w);
}

// scipy's quad on [a, b] (a, b not NaN): w holds QUADPACK's results (r QAWF's, when qawf); the
// value (its sign changed for reversed limits) is returned.
template <class F>
double quad(F& f, double a, double b, const Opts& o, qp::Work& w, int& limit_used, qp::Cycles& r, bool& qawf) {
    limit_used = o.limit;
    qawf = false;
    if (a == b) {  // scipy's shortcut, before anything else
        w.reset(1);
        return 0.0;
    }
    const bool flip = b < a;
    if (flip) std::swap(a, b);
    if (o.weight != kNone) {
        quad_weight(f, a, b, o, w, r, qawf);
        const double v = qawf ? r.result : w.result;
        return flip ? -v : v;
    }
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

// QAWF's cycles as three read-only arrays: their integrals, errors (float64) and ier (int64), in
// order.
nb::object cycle_arrays(const qp::Cycles& r) {
    const std::int64_t s1[1] = {r.lst};
    NDArray vals = new_array(DType::Float64, s1), errs = new_array(DType::Float64, s1),
            iers = new_array(DType::Int64, s1);
    auto* v = reinterpret_cast<double*>(vals.data);
    auto* e = reinterpret_cast<double*>(errs.data);
    auto* k = reinterpret_cast<std::int64_t*>(iers.data);
    for (int i = 1; i <= r.lst; ++i) {
        v[i - 1] = r.rslst[i];
        e[i - 1] = r.erlst[i];
        k[i - 1] = r.ierlst[i];
    }
    return nb::make_tuple(readonly(std::move(vals)), readonly(std::move(errs)), readonly(std::move(iers)));
}

nb::object quad_solve(
    const char* fn, nb::callable f, nb::callable check, double a, double b, nb::handle opts, int part, bool vectorized
) {
    return call_named(fn, [&]() -> nb::object {
        const Opts o = read_opts(opts);
        qp::Work w;
        qp::Cycles c;
        int limit = 0;
        bool qawf = false;
        double r;
        if (vectorized) {
            VecFn F{f, check, part, {}, {}, 0};
            r = quad(F, a, b, o, w, limit, c, qawf);
        } else {
            Fn F{f, check, part};
            r = quad(F, a, b, o, w, limit, c, qawf);
        }
        if (qawf) return nb::make_tuple(r, c.abserr, c.neval, c.ier, c.lst, cycle_arrays(c), true);
        return nb::make_tuple(r, w.abserr, w.neval, w.ier, w.last, interval_arrays(w, o.limit), false);
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
    std::vector<qp::Cycles> cycles;
    double abserr = 0.0;
    std::int64_t nfev = 0;
    int bad_ier = 0, bad_limit = 0;
    bool bad_qawf = false;  // the first failure was QAWF's (its texts for ier 1, 4, 7)

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
        qp::Cycles& c = cycles[static_cast<std::size_t>(i)];
        int limit = 0;
        bool qawf = false;
        double v;
        if (i == 0) {
            auto f = [&](double x) {
                xs[0] = x;
                return innermost();
            };
            v = quad(f, lo, hi, *op, w, limit, c, qawf);
            nfev += qawf ? c.neval : w.neval;
        } else {
            auto f = [&, i](double x) {
                xs[static_cast<std::size_t>(i)] = x;
                return integrate(i - 1);
            };
            v = quad(f, lo, hi, *op, w, limit, c, qawf);
        }
        const double err = qawf ? c.abserr : w.abserr;
        const int ier = qawf ? c.ier : w.ier;
        if (err > abserr) abserr = err;  // Python's max(self.abserr, abserr)
        if (ier != 0 && bad_ier == 0) {
            bad_ier = ier;
            bad_limit = limit;
            bad_qawf = qawf;
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
        q.cycles.resize(static_cast<std::size_t>(n));
        const double v = q.integrate(n - 1);
        return nb::make_tuple(v, q.abserr, q.nfev, q.bad_ier, q.bad_limit, q.bad_qawf);
    });
}

}  // namespace

void bind_quadpack(nb::module_& m) {
    m.def(
        "quad_solve", &quad_solve, nb::arg("fn"), nb::arg("f"), nb::arg("check"), nb::arg("a"), nb::arg("b"),
        nb::arg("opts"), nb::arg("part"), nb::arg("vectorized") = false
    );
    m.def(
        "nquad_solve", &nquad_solve, nb::arg("fn"), nb::arg("func"), nb::arg("check"), nb::arg("levels"),
        nb::arg("range_check"), nb::arg("opts_check")
    );
}

}  // namespace sd::integrate
