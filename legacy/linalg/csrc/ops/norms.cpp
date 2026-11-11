// Norms: vector_norm and matrix_norm (API.md §2; ROADMAP.md M2).
//
// No LAPACK here, so NaN propagates (DESIGN.md §2.4). The p-norms avoid
// overflow and underflow: a fast unscaled pass first, and a second pass
// scaled by the largest magnitude only when the first one overflowed or all
// magnitudes were tiny (DESIGN.md §2.11). matrix_norm's 2, -2 and 'nuc' need
// the SVD and arrive in M5.
#include "ops/norms.hpp"

#include "core/args.hpp"
#include "core/cost.hpp"
#include "core/outblock.hpp"
#include "core/pairwise.hpp"
#include "ops/common.hpp"
#include "ops/svals.hpp"
#include "ops/vnorm.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace sdla {

namespace {

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

// The norm of the region of x at `base` with the given shape and strides
// (bytes), walked in logical C order. For the 2-norm of complex values the
// real and imaginary parts are separate terms (|z|^2 = re^2 + im^2, no hypot):
// an extra innermost axis of length 2 over the two doubles.
double region_norm(VecOrd ord, const Input& x, const std::byte* base, int nd, const std::int64_t* shape,
                   const std::int64_t* strides) {
    if (ord.kind == Ord::Two && x.dtype() == DType::Complex128) {
        std::int64_t sh[kMaxWalkDims], st[kMaxWalkDims];
        std::copy(shape, shape + nd, sh);
        std::copy(strides, strides + nd, st);
        sh[nd] = 2;
        st[nd] = sizeof(double);
        auto mag = [](double v) { return std::fabs(v); };  // a lambda type, so it inlines
        Walk<double, decltype(mag)> w(base, nd + 1, sh, st, mag);
        return vnorm(ord, w.count(), w);
    }
    return visit_storage(x.dtype(), [&]<class S>(std::type_identity<S>) {
        auto mag = [](S v) { return magnitude(v); };
        Walk<S, decltype(mag)> w(base, nd, shape, strides, mag);
        return vnorm(ord, w.count(), w);
    });
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
    const std::vector<std::int64_t> rsh(red_shape.begin(), red_shape.end()), rst(red_strides.begin(), red_strides.end());
    auto norm_at = [&](std::int64_t off) {
        return region_norm(ord, x, x.data() + off, static_cast<int>(rsh.size()), rsh.data(), rst.data());
    };
    std::int64_t red_count = 1;
    for (std::int64_t v : rsh) red_count *= v;
    if (out_shape.empty()) return py_scalar(norm_at(0));
    Batch b({{kept_shape, kept_strides}}, "vector_norm");
    Out out = alloc_c(out_shape, DType::Float64);
    double* res = out.as<double>();
    run_batch(b, cost(kStream, double(red_count)), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
        res[i] = norm_at(off[0]);
        return 0;
    });
    return to_sdnp(out);
}

// ---------------------------------------------------------- matrix_norm --

enum class MatOrd { Fro, One, NegOne, Inf, NegInf, Two, NegTwo, Nuc };

MatOrd parse_matrix_ord(nb::handle o) {
    if (PyUnicode_Check(o.ptr())) {
        const std::string s = nb::cast<std::string>(o);
        if (s == "fro") return MatOrd::Fro;
        if (s == "nuc") return MatOrd::Nuc;
    } else if (!PyBool_Check(o.ptr()) && (PyLong_Check(o.ptr()) || PyFloat_Check(o.ptr()))) {
        const double p = PyFloat_AsDouble(o.ptr());
        if (p == -1.0 && PyErr_Occurred()) throw nb::python_error();  // an int too large for a double
        if (p == 1) return MatOrd::One;
        if (p == -1) return MatOrd::NegOne;
        if (p == kInf) return MatOrd::Inf;
        if (p == -kInf) return MatOrd::NegInf;
        if (p == 2) return MatOrd::Two;
        if (p == -2) return MatOrd::NegTwo;
    }
    throw nb::value_error("matrix_norm(): 'ord' must be 'fro', 'nuc', 1, -1, 2, -2, inf or -inf");
}

// ord 2, -2, 'nuc' from the singular values. These call LAPACK, so NaN and inf
// are a ValueError here while the other ords propagate them (DESIGN.md §2.14).
nb::object matrix_norm_svd(const Input& x, MatOrd ord, const std::vector<std::int64_t>& out_shape) {
    const lint m = x.shape(-2), n = x.shape(-1), k = std::min(m, n);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "matrix_norm");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_c(out_shape, DType::Float64);
        double* res = out.as<double>();
        const Failure f = run_batch(b, cost(kSVD, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            double* sv = s.get<double>(static_cast<std::size_t>(std::max<lint>(k, 1)), 1);
            if (const int code = singular_values<T>(x, off[0], m, n, s, sv)) return code;
            double v = 0;  // empty matrices: 0 (S11)
            if (k > 0) {
                if (ord == MatOrd::Two) v = sv[0];
                else if (ord == MatOrd::NegTwo) v = sv[k - 1];
                else
                    v = pairwise_sum<double>(k, [&](std::int64_t b0, std::int64_t len, double* out) {
                        std::copy(sv + b0, sv + b0 + len, out);
                    });
            }
            res[i] = v;
            return 0;
        });
        if (f) raise(b, f, "matrix_norm", "x");
        return out_shape.empty() ? py_scalar(res[0]) : to_sdnp(out);
    });
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
    if (ord == MatOrd::Two || ord == MatOrd::NegTwo || ord == MatOrd::Nuc) return matrix_norm_svd(x, ord, out_shape);
    // One norm: sums of magnitudes along the lines (columns for 1, rows for inf), then max/min.
    auto norm_at = [&](std::int64_t off) -> double {
        const std::byte* base = x.data() + off;
        if (ord == MatOrd::Fro) {
            const std::int64_t sh[2] = {m, n}, st[2] = {rs, cs};
            return region_norm({Ord::Two, 2.0}, x, base, 2, sh, st);
        }
        // One norm: pairwise sums of magnitudes along the lines (columns for 1,
        // rows for inf), then their max/min, NaN propagating.
        const bool cols = ord == MatOrd::One || ord == MatOrd::NegOne;
        const std::int64_t lines = cols ? n : m, len = cols ? m : n;
        const std::int64_t line_step = cols ? cs : rs, elem_step = cols ? rs : cs;
        const bool max = ord == MatOrd::One || ord == MatOrd::Inf;
        return visit_storage(x.dtype(), [&]<class S>(std::type_identity<S>) {  // one dispatch per matrix
            auto line_sum = [&](std::int64_t l) {
                const std::byte* line = base + l * line_step;
                return pairwise_sum<double>(len, [&](std::int64_t b0, std::int64_t k, double* out) {
                    for (std::int64_t t = 0; t < k; ++t) out[t] = magnitude(load<S>(line + (b0 + t) * elem_step));
                });
            };
            double best = max ? 0.0 : kInf;
            bool nan = false;
            auto take = [&](double sum) {
                if (sum != sum) nan = true;
                else best = max ? (sum > best ? sum : best) : (sum < best ? sum : best);
            };
            if (lines * len >= kSumParallelMin && lines > 1 && detail::begin_parallel()) {
                // one large matrix: its lines on the pool (M9); each line's sum is
                // computed as in the serial loop, and max/min do not depend on order
                std::vector<double> sums(static_cast<std::size_t>(lines));
                const std::int64_t per = std::max<std::int64_t>(1, kSumLeaf / std::max<std::int64_t>(len, 1));
                detail::run_tasks((lines + per - 1) / per, [&](std::int64_t t) {
                    for (std::int64_t l = t * per; l < std::min(lines, (t + 1) * per); ++l) sums[l] = line_sum(l);
                });
                for (double v : sums) take(v);
            } else {
                for (std::int64_t l = 0; l < lines; ++l) take(line_sum(l));
            }
            return lines == 0 ? 0.0 : (nan ? kNaN : best);
        });
    };
    if (out_shape.empty()) return py_scalar(norm_at(0));
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "matrix_norm");
    Out out = alloc_c(out_shape, DType::Float64);
    double* res = out.as<double>();
    run_batch(b, cost(kStream, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
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
