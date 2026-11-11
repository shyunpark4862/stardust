// Norms: vector_norm and matrix_norm (API.md §2; ROADMAP.md M2).
//
// No LAPACK here, so NaN propagates (DESIGN.md §2.4). The p-norms avoid
// overflow and underflow: a fast unscaled pass first, and a second pass
// scaled by the largest magnitude only when the first one overflowed or all
// magnitudes were tiny (DESIGN.md §2.11). matrix_norm's 2, -2 and 'nuc' need
// the SVD and arrive in M5.
#include "ops/norms.hpp"

#include "core/args.hpp"
#include "core/outblock.hpp"
#include "ops/common.hpp"

#include <nanobind/stl/string.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace sdla {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// Below this, p-th powers may be subnormal and lose bits: 2^-969 = DBL_MIN * 2^53.
constexpr double kTiny = 0x1p-969;

enum class Ord { Inf, NegInf, Zero, One, Two, P };

struct VecOrd {
    Ord kind;
    double p;
};

VecOrd parse_vector_ord(nb::handle o, const char* func) {
    if (PyBool_Check(o.ptr()) || !(PyLong_Check(o.ptr()) || PyFloat_Check(o.ptr())))
        throw nb::type_error((std::string(func) + "(): 'ord' must be an int or a float").c_str());
    const double p = PyFloat_AsDouble(o.ptr());
    if (p == -1.0 && PyErr_Occurred()) throw nb::python_error();
    if (std::isnan(p)) throw nb::value_error((std::string(func) + "(): 'ord' must not be NaN").c_str());
    if (p == kInf) return {Ord::Inf, p};
    if (p == -kInf) return {Ord::NegInf, p};
    if (p == 0) return {Ord::Zero, p};
    if (p == 1) return {Ord::One, p};
    if (p == 2) return {Ord::Two, p};
    return {Ord::P, p};
}

// The norm of a set of magnitudes. `each(f)` calls f(double) once per
// magnitude (it may be called twice, for the scaled pass); `count` is how
// many there are. Empty sets have norm 0 (S11).
template <class Each>
double vnorm(VecOrd ord, std::int64_t count, Each&& each) {
    if (count == 0) return 0.0;
    switch (ord.kind) {
        case Ord::Inf: {
            double m = 0;
            bool nan = false;
            each([&](double a) {
                if (a != a) nan = true;
                else if (a > m) m = a;
            });
            return nan ? kNaN : m;
        }
        case Ord::NegInf: {
            double m = kInf;
            bool nan = false;
            each([&](double a) {
                if (a != a) nan = true;
                else if (a < m) m = a;
            });
            return nan ? kNaN : m;
        }
        case Ord::Zero: {  // the number of nonzeros (not a norm; Array API). NaN counts.
            std::int64_t c = 0;
            each([&](double a) { c += (a != 0); });
            return static_cast<double>(c);
        }
        case Ord::One: {
            double s = 0;
            each([&](double a) { s += a; });
            return s;
        }
        case Ord::Two:
        case Ord::P: {
            const bool two = ord.kind == Ord::Two;
            const double p = ord.p;
            double s = 0, amax = 0;
            bool nan = false;
            each([&](double a) {
                if (a != a) {
                    nan = true;
                    return;
                }
                s += two ? a * a : std::pow(a, p);
                if (a > amax) amax = a;
            });
            if (nan) return kNaN;
            if (p > 0) {
                if (amax == kInf) return kInf;
                const double top = two ? amax * amax : std::pow(amax, p);
                if (std::isinf(s) || (amax > 0 && top < kTiny)) {
                    // Rescaled: amax * (sum (a/amax)^p)^(1/p). Every term is <= 1.
                    double t = 0;
                    each([&](double a) {
                        const double r = a / amax;
                        t += two ? r * r : std::pow(r, p);
                    });
                    return amax * (two ? std::sqrt(t) : std::pow(t, 1 / p));
                }
            }
            return two ? std::sqrt(s) : std::pow(s, 1 / p);
        }
    }
    return kNaN;
}

// Calls f(magnitude) for every element of x within `axes` (byte offset `base`
// into x's data). For the 2-norm of complex values, the real and imaginary
// parts are fed separately: |z|^2 = re^2 + im^2, and no hypot is needed.
template <class F>
void each_magnitude(const Input& x, const std::byte* base, const std::vector<Py_ssize_t>& shape,
                    const std::vector<Py_ssize_t>& strides, bool split_complex, F&& f) {
    visit_storage(x.dtype(), [&]<class S>(std::type_identity<S>) {
        auto one = [&](const std::byte* p) {
            const S v = load<S>(p);
            if constexpr (std::is_same_v<S, std::uint8_t>) f(v ? 1.0 : 0.0);
            else if constexpr (std::is_same_v<S, std::int64_t>) f(std::fabs(static_cast<double>(v)));
            else if constexpr (std::is_same_v<S, double>) f(std::fabs(v));
            else if (split_complex) {
                f(std::fabs(v.real()));
                f(std::fabs(v.imag()));
            } else {
                f(std::abs(v));
            }
        };
        // Odometer over the axes; the last one is the inner loop.
        const std::size_t nd = shape.size();
        if (nd == 0) {
            one(base);
            return;
        }
        for (Py_ssize_t s : shape)
            if (s == 0) return;
        std::vector<Py_ssize_t> idx(nd, 0);
        const Py_ssize_t n_last = shape[nd - 1], s_last = strides[nd - 1];
        const std::byte* p = base;
        while (true) {
            for (Py_ssize_t i = 0; i < n_last; ++i) one(p + i * s_last);
            std::size_t d = nd - 1;
            while (d > 0) {
                --d;
                if (++idx[d] < shape[d]) {
                    p += strides[d];
                    goto next;
                }
                p -= strides[d] * (shape[d] - 1);
                idx[d] = 0;
            }
            return;
        next:;
        }
    });
}

std::int64_t product(const std::vector<Py_ssize_t>& s) {
    std::int64_t n = 1;
    for (Py_ssize_t v : s) n *= v;
    return n;
}

// ---------------------------------------------------------- vector_norm --

nb::object vector_norm(nb::handle xo, nb::handle axis_o, bool keepdims, nb::handle ord_o) {
    Input x(xo, "vector_norm", "x");
    const std::vector<int> axes = parse_axes(axis_o, x.ndim(), "vector_norm");
    const VecOrd ord = parse_vector_ord(ord_o, "vector_norm");
    std::vector<Py_ssize_t> kept_shape, kept_strides, red_shape, red_strides;
    std::vector<std::int64_t> out_shape;
    for (int d = 0, j = 0; d < x.ndim(); ++d) {
        const bool reduced = j < static_cast<int>(axes.size()) && axes[j] == d;
        if (reduced) {
            ++j;
            red_shape.push_back(x.shape(d));
            red_strides.push_back(x.stride(d));
            if (keepdims) out_shape.push_back(1);
        } else {
            kept_shape.push_back(x.shape(d));
            kept_strides.push_back(x.stride(d));
            out_shape.push_back(x.shape(d));
        }
    }
    const bool split = ord.kind == Ord::Two && x.dtype() == DType::Complex128;
    const std::int64_t count = product(red_shape) * (split ? 2 : 1);
    auto norm_at = [&](std::int64_t off) {
        return vnorm(ord, count, [&](auto&& f) { each_magnitude(x, x.data() + off, red_shape, red_strides, split, f); });
    };
    if (out_shape.empty()) return py_scalar(norm_at(0));
    Batch b({{kept_shape, kept_strides}}, "vector_norm");
    Out out = alloc_c(out_shape, DType::Float64);
    double* res = out.as<double>();
    run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch&) {
        res[i] = norm_at(off[0]);
        return 0;
    });
    return to_sdnp(out);
}

// ---------------------------------------------------------- matrix_norm --

enum class MatOrd { Fro, One, NegOne, Inf, NegInf };

[[noreturn]] void not_yet(const char* what) {
    PyErr_SetString(PyExc_NotImplementedError, what);
    throw nb::python_error();
}

MatOrd parse_matrix_ord(nb::handle o) {
    if (PyUnicode_Check(o.ptr())) {
        const std::string s = nb::cast<std::string>(o);
        if (s == "fro") return MatOrd::Fro;
        if (s == "nuc") not_yet("matrix_norm(): ord='nuc' needs the SVD, which arrives in milestone M5");
    } else if (!PyBool_Check(o.ptr()) && (PyLong_Check(o.ptr()) || PyFloat_Check(o.ptr()))) {
        const double p = PyFloat_AsDouble(o.ptr());
        if (p == 1) return MatOrd::One;
        if (p == -1) return MatOrd::NegOne;
        if (p == kInf) return MatOrd::Inf;
        if (p == -kInf) return MatOrd::NegInf;
        if (p == 2 || p == -2) not_yet("matrix_norm(): ord=2 and ord=-2 need the SVD, which arrive in milestone M5");
    }
    throw nb::value_error("matrix_norm(): 'ord' must be 'fro', 'nuc', 1, -1, 2, -2, inf or -inf");
}

nb::object matrix_norm(nb::handle xo, bool keepdims, nb::handle ord_o) {
    Input x(xo, "matrix_norm", "x");
    require_matrix(x);
    const MatOrd ord = parse_matrix_ord(ord_o);
    const std::int64_t m = x.shape(-2), n = x.shape(-1), rs = x.stride(-2), cs = x.stride(-1);
    std::vector<std::int64_t> out_shape(x.batch_shape(2).begin(), x.batch_shape(2).end());
    if (keepdims) {
        out_shape.push_back(1);
        out_shape.push_back(1);
    }
    // One norm: sums of magnitudes along the lines (columns for 1, rows for inf), then max/min.
    auto norm_at = [&](std::int64_t off) -> double {
        const std::byte* base = x.data() + off;
        if (ord == MatOrd::Fro) {
            const bool split = x.dtype() == DType::Complex128;
            const std::vector<Py_ssize_t> shape{m, n}, strides{rs, cs};
            return vnorm({Ord::Two, 2.0}, m * n * (split ? 2 : 1),
                         [&](auto&& f) { each_magnitude(x, base, shape, strides, split, f); });
        }
        const bool cols = ord == MatOrd::One || ord == MatOrd::NegOne;
        const std::int64_t lines = cols ? n : m, len = cols ? m : n;
        const std::int64_t line_step = cols ? cs : rs, elem_step = cols ? rs : cs;
        const std::vector<Py_ssize_t> shape{len}, strides{elem_step};
        std::vector<double> sums(static_cast<std::size_t>(lines));
        for (std::int64_t l = 0; l < lines; ++l)
            sums[l] = vnorm({Ord::One, 1.0}, len,
                            [&](auto&& f) { each_magnitude(x, base + l * line_step, shape, strides, false, f); });
        const bool max = ord == MatOrd::One || ord == MatOrd::Inf;
        return vnorm({max ? Ord::Inf : Ord::NegInf, 0.0}, lines, [&](auto&& f) {
            for (double s : sums) f(s);
        });
    };
    if (out_shape.empty()) return py_scalar(norm_at(0));
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "matrix_norm");
    Out out = alloc_c(out_shape, DType::Float64);
    double* res = out.as<double>();
    run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch&) {
        res[i] = norm_at(off[0]);
        return 0;
    });
    return to_sdnp(out);
}

}  // namespace

void bind_norms(nb::module_& m) {
    m.def("vector_norm", &vector_norm, nb::arg().none(), nb::kw_only(), nb::arg("axis").none() = nb::none(),
          nb::arg("keepdims") = false, nb::arg("ord") = 2);
    m.def("matrix_norm", &matrix_norm, nb::arg().none(), nb::kw_only(), nb::arg("keepdims") = false,
          nb::arg("ord") = "fro");
}

}  // namespace sdla
