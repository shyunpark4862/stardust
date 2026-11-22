// Norms: vector_norm and matrix_norm (docs/api/linalg.md §3).
//
// No LAPACK here, so NaN propagates (docs/design/linalg.md §2.3). The p-norms
// avoid overflow and underflow: a fast unscaled pass first, and a second pass
// scaled by the largest magnitude only when the first one overflowed or all
// magnitudes were tiny (docs/design/linalg.md §10.8). matrix_norm's 2, -2 and
// 'nuc' come from the singular values (svals.hpp), and so check NaN/inf
// (docs/design/linalg.md §2.3).
#include "linalg/ops/norms.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"

#include "linalg/args.hpp"
#include "linalg/cost.hpp"
#include "linalg/fastsum.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pairwise.hpp"
#include "linalg/ops/common.hpp"
#include "linalg/ops/svals.hpp"
#include "linalg/ops/vnorm.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace sd::linalg {

namespace {

VecOrd parse_vector_ord(nb::handle o, const char* func) {
    const std::optional<double> r = real_value(o, "ord");  // numpy integers too (API.md §2)
    if (!r)
        throw nb::type_error(
            (std::string(func) + "(): 'ord' must be an int or a float, not " + py_type_name(o)).c_str()
        );
    const double p = *r;
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
// True if the doubles of a region (shape, byte strides) follow each other in memory in
// logical C order: the 2-norm then reads them in place (vnorm's `direct`).
bool c_contiguous_doubles(int nd, const std::int64_t* shape, const std::int64_t* strides) {
    std::int64_t step = sizeof(double);
    for (int d = nd; d-- > 0;) {
        if (shape[d] == 1) continue;
        if (strides[d] != step) return false;
        step *= shape[d];
    }
    return true;
}

double region_norm(
    VecOrd ord, const Input& x, const std::byte* base, int nd, const std::int64_t* shape, const std::int64_t* strides
) {
    if (ord.kind == Ord::Two && x.dtype() == DType::Complex128) {
        std::int64_t sh[kMaxWalkDims], st[kMaxWalkDims];
        std::copy(shape, shape + nd, sh);
        std::copy(strides, strides + nd, st);
        sh[nd] = 2;
        st[nd] = sizeof(double);
        auto mag = [](double v) { return std::fabs(v); };  // a lambda type, so it inlines
        Walk<double, decltype(mag)> w(base, nd + 1, sh, st, mag);
        const bool direct = c_contiguous_doubles(nd + 1, sh, st);
        return vnorm(ord, w.count(), w, direct ? reinterpret_cast<const double*>(base) : nullptr);
    }
    return visit_storage(x.dtype(), [&]<class S>(std::type_identity<S>) {
        auto mag = [](S v) { return magnitude(v); };
        Walk<S, decltype(mag)> w(base, nd, shape, strides, mag);
        const double* direct = nullptr;
        if constexpr (std::is_same_v<S, double>)
            if (ord.kind == Ord::Two && c_contiguous_doubles(nd, shape, strides))
                direct = reinterpret_cast<const double*>(base);
        return vnorm(ord, w.count(), w, direct);
    });
}

// The fast path of a norm (fastsum.hpp): float64 data (complex128 for the 2-norm, its
// parts as terms) whose region is one dense block of memory in any axis order, or a single
// axis with a positive stride, read in place as `count` doubles `inc` apart from the
// region's base. The region's shape and strides are the same for every norm of a batch, so
// the plan is made once per call.
struct FastPlan {
    bool ok = false;
    std::int64_t count = 0, inc = 1;
};

FastPlan fast_plan(VecOrd ord, DType dt, int nd, const std::int64_t* shape, const std::int64_t* strides) {
    FastPlan f;
    const bool cx = dt == DType::Complex128;
    if (!(dt == DType::Float64 || (cx && ord.kind == Ord::Two))) return f;
    if (ord.kind != Ord::Two && ord.kind != Ord::One && ord.kind != Ord::Inf && ord.kind != Ord::NegInf) return f;
    std::int64_t ext[kMaxWalkDims], st[kMaxWalkDims];
    int k = 0;
    std::int64_t count = 1;
    for (int d = 0; d < nd; ++d) {
        if (shape[d] == 0) return {true, 0, 1};
        if (shape[d] == 1) continue;
        ext[k] = shape[d];
        st[k++] = strides[d];
        count *= shape[d];
    }
    if (cx) {  // the two parts of each element: an innermost axis of 2 doubles
        ext[k] = 2;
        st[k++] = sizeof(double);
        count *= 2;
    }
    if (k == 0) return {true, 1, 1};
    if (k == 1) {
        if (st[0] <= 0 || st[0] % static_cast<std::int64_t>(sizeof(double)) != 0) return f;
        const std::int64_t inc = st[0] / static_cast<std::int64_t>(sizeof(double));
        if (inc != 1 && (ord.kind == Ord::Inf || ord.kind == Ord::NegInf)) return f;  // the max reads runs only
        return {true, count, inc};
    }
    // a dense block: sorted by stride, each axis spans exactly the ones inside it
    int order[kMaxWalkDims];
    for (int i = 0; i < k; ++i) order[i] = i;
    std::sort(order, order + k, [&](int a, int b) { return st[a] < st[b]; });
    std::int64_t step = sizeof(double);
    for (int i = 0; i < k; ++i) {
        if (st[order[i]] != step) return f;
        step *= ext[order[i]];
    }
    return {true, count, 1};
}

// The norm by the fast path, or a negative number when the 2-norm's plain sum of squares
// cannot be trusted (fast::norm_from_squares): the caller then takes the careful path.
double fast_norm(VecOrd ord, const FastPlan& f, const std::byte* base) {
    if (f.count == 0) return 0.0;  // empty sets have norm 0 (docs/design/linalg.md §2.5)
    const double* p = reinterpret_cast<const double*>(base);
    switch (ord.kind) {
        case Ord::Two: return fast::norm_from_squares(fast::sum_squares(f.count, p, f.inc), f.count);
        case Ord::One: return fast::sum_abs(f.count, p, f.inc);
        case Ord::Inf: return fast::extreme<true>(f.count, p);
        case Ord::NegInf: return fast::extreme<false>(f.count, p);
        default: return -1.0;
    }
}

// ---------------------------------------------------------- vector_norm --

nb::object vector_norm(nb::handle xo, nb::handle axis_o, nb::handle keepdims_o, nb::handle ord_o) {
    Input x(xo, "vector_norm", "x");
    const bool keepdims = parse_bool(keepdims_o, "keepdims");
    const std::vector<int> axes = parse_axes(axis_o, x.ndim(), "vector_norm");
    const VecOrd ord = parse_vector_ord(ord_o, "vector_norm");
    std::vector<std::int64_t> kept_shape, kept_strides, red_shape, red_strides;
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
    const std::vector<std::int64_t> rsh(red_shape.begin(), red_shape.end()),
        rst(red_strides.begin(), red_strides.end());
    const FastPlan plan = fast_plan(ord, x.dtype(), static_cast<int>(rsh.size()), rsh.data(), rst.data());
    auto slow = [&](std::int64_t off) __attribute__((noinline)) {
        return region_norm(ord, x, x.data() + off, static_cast<int>(rsh.size()), rsh.data(), rst.data());
    };
    auto norm_at = [&](std::int64_t off) {
        if (plan.ok) {
            const double r = fast_norm(ord, plan, x.data() + off);
            if (r >= 0 || r != r) return r;
        }
        return slow(off);
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
    return to_array(out);
}

// ---------------------------------------------------------- matrix_norm --

enum class MatOrd { Fro, One, NegOne, Inf, NegInf, Two, NegTwo, Nuc };

MatOrd parse_matrix_ord(nb::handle o) {
    if (PyUnicode_Check(o.ptr())) {
        const std::string s = nb::cast<std::string>(o);
        if (s == "fro") return MatOrd::Fro;
        if (s == "nuc") return MatOrd::Nuc;
    } else if (const std::optional<double> r = real_value(o, "ord")) {  // numpy integers too (API.md §2)
        const double p = *r;
        if (p == 1) return MatOrd::One;
        if (p == -1) return MatOrd::NegOne;
        if (p == kInf) return MatOrd::Inf;
        if (p == -kInf) return MatOrd::NegInf;
        if (p == 2) return MatOrd::Two;
        if (p == -2) return MatOrd::NegTwo;
    } else {  // a wrong type: TypeError, as in vector_norm (docs/api/linalg.md §3)
        throw nb::type_error(("matrix_norm(): 'ord' must be a str, an int or a float, not " + py_type_name(o)).c_str());
    }
    throw nb::value_error((std::string("matrix_norm(): 'ord' must be 'fro', 'nuc', 1, -1, 2, -2, inf or -inf, not ") +
                           nb::repr(o).c_str())
                              .c_str());
}

// ord 2, -2, 'nuc' from the singular values. These call LAPACK, so NaN and inf
// are a ValueError here while the other ords propagate them (docs/design/linalg.md §2.3).
nb::object matrix_norm_svd(const Input& x, MatOrd ord, const std::vector<std::int64_t>& out_shape) {
    const lint m = x.shape(-2), n = x.shape(-1), k = std::min(m, n);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "matrix_norm");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_c(out_shape, DType::Float64);
        double* res = out.as<double>();
        const Failure f =
            run_batch(b, cost(kSVD, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                double* sv = s.get<double>(static_cast<std::size_t>(std::max<lint>(k, 1)), 1);
                if (const int code = singular_values<T>(x, off[0], m, n, s, sv)) return code;
                double v = 0;  // empty matrices: 0 (docs/design/linalg.md §2.5)
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
        return out_shape.empty() ? py_scalar(res[0]) : to_array(out);
    });
}

nb::object matrix_norm(nb::handle xo, nb::handle keepdims_o, nb::handle ord_o) {
    Input x(xo, "matrix_norm", "x");
    const bool keepdims = parse_bool(keepdims_o, "keepdims");
    require_matrix(x);
    const MatOrd ord = parse_matrix_ord(ord_o);
    const std::int64_t m = x.shape(-2), n = x.shape(-1), rs = x.stride(-2), cs = x.stride(-1);
    std::vector<std::int64_t> out_shape(x.batch_shape(2).begin(), x.batch_shape(2).end());
    if (keepdims) {
        out_shape.push_back(1);
        out_shape.push_back(1);
    }
    if (ord == MatOrd::Two || ord == MatOrd::NegTwo || ord == MatOrd::Nuc) return matrix_norm_svd(x, ord, out_shape);
    const std::int64_t fsh[2] = {m, n}, fst[2] = {rs, cs};
    const FastPlan fro_plan = ord == MatOrd::Fro ? fast_plan({Ord::Two, 2.0}, x.dtype(), 2, fsh, fst) : FastPlan{};
    // The 1/inf norms: sums of magnitudes along the lines (columns for 1, rows for inf),
    // then their max/min, NaN propagating.
    const bool cols = ord == MatOrd::One || ord == MatOrd::NegOne;
    const std::int64_t lines = cols ? n : m, len = cols ? m : n;
    const std::int64_t line_step = cols ? cs : rs, elem_step = cols ? rs : cs;
    const bool max = ord == MatOrd::One || ord == MatOrd::Inf;
    constexpr std::int64_t d8 = sizeof(double);
    // float64 read in place: lines contiguous (each a fast sum), or the lines side by side
    // (strips of lines summed together, row by row)
    const bool f64 = x.dtype() == DType::Float64;
    const bool line_runs = f64 && elem_step == d8;
    const bool strips = f64 && !line_runs && line_step == d8 && elem_step > 0 && elem_step % d8 == 0;
    auto norm_at = [&](std::int64_t off) -> double {
        const std::byte* base = x.data() + off;
        if (ord == MatOrd::Fro) {
            if (fro_plan.ok) {
                const double r = fast_norm({Ord::Two, 2.0}, fro_plan, base);
                if (r >= 0) return r;
            }
            return region_norm({Ord::Two, 2.0}, x, base, 2, fsh, fst);
        }
        if (lines == 0) return 0.0;
        double best = max ? 0.0 : kInf;
        bool nan = false;
        auto take = [&](double sum) {
            if (sum != sum) nan = true;
            else best = max ? (sum > best ? sum : best) : (sum < best ? sum : best);
        };
        const bool large = lines * len >= kSumParallelMin;
        if (line_runs || strips)
            return fast::line_abs_extreme(
                reinterpret_cast<const double*>(base), lines, len, line_step / d8, elem_step / d8, max
            );
        return visit_storage(x.dtype(), [&]<class S>(std::type_identity<S>) {  // one dispatch per matrix
            auto line_sum = [&](std::int64_t l) {
                const std::byte* line = base + l * line_step;
                return pairwise_sum<double>(len, [&](std::int64_t b0, std::int64_t k, double* out) {
                    for (std::int64_t t = 0; t < k; ++t) out[t] = magnitude(load<S>(line + (b0 + t) * elem_step));
                });
            };
            if (large && lines > 1 && detail::begin_parallel()) {
                std::vector<double> sums(static_cast<std::size_t>(lines));
                const std::int64_t per = std::max<std::int64_t>(1, kSumLeaf / std::max<std::int64_t>(len, 1));
                detail::run_tasks((lines + per - 1) / per, [&](std::int64_t t) {
                    for (std::int64_t l = t * per; l < std::min(lines, (t + 1) * per); ++l) sums[l] = line_sum(l);
                });
                for (double v : sums) take(v);
            } else {
                const parallel::Unlocked unlocked(large);
                for (std::int64_t l = 0; l < lines; ++l) take(line_sum(l));
            }
            return nan ? kNaN : best;
        });
    };
    if (out_shape.empty()) return py_scalar(norm_at(0));
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "matrix_norm");
    Out out = alloc_c(out_shape, DType::Float64);
    double* res = out.as<double>();
    auto slow = [&](std::int64_t off) __attribute__((noinline)) { return norm_at(off); };
    run_batch(b, cost(kStream, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
        if (fro_plan.ok) {  // inline in the batch loop: the 'fro' norm of small matrices
            const double r = fast_norm({Ord::Two, 2.0}, fro_plan, x.data() + off[0]);
            if (r >= 0) {
                res[i] = r;
                return 0;
            }
        }
        res[i] = slow(off[0]);
        return 0;
    });
    return to_array(out);
}

}  // namespace

void bind_norms(nb::module_& m) {
    m.def(
        "vector_norm", named("vector_norm", &vector_norm),
        nb::sig("def vector_norm(x, /, *, axis=None, keepdims=False, ord=2)"), nb::arg().none(), nb::kw_only(),
        nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false, nb::arg("ord").none() = 2
    );
    m.def(
        "matrix_norm", named("matrix_norm", &matrix_norm),
        nb::sig("def matrix_norm(x, /, *, keepdims=False, ord='fro')"), nb::arg().none(), nb::kw_only(),
        nb::arg("keepdims").none() = false, nb::arg("ord").none() = "fro"
    );
}

}  // namespace sd::linalg
