#include "array/ops/reduction.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/buffered.hpp"
#include "array/kernels/cast.hpp"
#include "array/kernels/reduce.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/lanes.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace sd {

namespace {

// ------------------------------------------------------------------ axes

struct Axes {
    std::array<bool, kMaxNdim> reduced{};
    int count = 0;
    std::int64_t per_output = 1;  // how many inputs fold into each output
};

Axes parse_reduce_axes(const NDArray& x, nb::handle axis) {
    Axes ax;
    if (axis.is_none()) {
        for (int d = 0; d < x.ndim; ++d) ax.reduced[d] = true;
    } else {
        for (int d : parse_axes(axis, x.ndim)) ax.reduced[d] = true;  // () reduces nothing
    }
    for (int d = 0; d < x.ndim; ++d)
        if (ax.reduced[d]) {
            ++ax.count;
            ax.per_output *= x.shape[d];
        }
    return ax;
}

std::int64_t output_size(const NDArray& x, const Axes& ax) {
    std::int64_t n = 1;
    for (int d = 0; d < x.ndim; ++d)
        if (!ax.reduced[d]) n *= x.shape[d];
    return n;
}

// The result with keepdims: x's shape with 1 on reduced axes, laid out in x's
// axis order so the iterator still walks x in memory order.
NDArray keepdims_result(const NDArray& x, const Axes& ax, DType dtype, bool zeroed = false) {
    NDArray proto = x;
    for (int d = 0; d < x.ndim; ++d)
        if (ax.reduced[d]) proto.shape[d] = 1;
    return new_array_like(proto, dtype, zeroed);
}

// keepdims=False: drop the reduced axes (a view of the fresh result), or hand
// back a Python scalar when nothing is left (docs/design/array.md §2.2).
nb::object finish(NDArray&& res, const Axes& ax, bool keepdims) {
    if (keepdims) return wrap(std::move(res));
    if (ax.count == res.ndim) return load_python(res.dtype, res.data);
    int n = 0;
    for (int d = 0; d < res.ndim; ++d) {
        if (ax.reduced[d]) continue;
        res.shape[n] = res.shape[d];
        res.strides[n] = res.strides[d];
        ++n;
    }
    res.ndim = n;
    res.update_flags();
    return wrap(std::move(res));
}

template <typename T>
void fill(NDArray& a, T v) {
    T* p = reinterpret_cast<T*>(a.data);
    std::fill(p, p + a.size(), v);
}

// Runs a reduction iterator over (x, out): `loop` works in out's dtype; an x
// of another dtype is converted chunk by chunk on the way (buffered casting,
// docs/design/array.md §4.3).
void run_fold(const NdIter& it, DType from, DType to, Loop loop) {
    if (from == to) {
        it.run(loop, nullptr, Split::Reduction);
        return;
    }
    BufferedLoop buffered;
    buffered.inner = loop;
    buffered.nin = 1;
    buffered.nop = 2;
    buffered.add_cast(0, cast_kernel(from, to), itemsize(to));
    it.run(&buffered_loop, &buffered, Split::Reduction);
}

// ------------------------------------------------------- blocked reductions
//
// When the reduced axes come first in memory -- sum(m, axis=0) of a C-ordered
// matrix -- the kept axes form the innermost run, and run() cuts that into
// pieces of at least kMinInnerPiece: often a single task, so the reduction
// ran on one thread. Instead one reduced axis is cut into a fixed number of
// blocks. Each block folds into its own copy of the output (holding the start
// value) on the pool, then the copies fold into the output in block order.
//
// The blocks depend on the shape, layout and dtypes only, never on the thread
// count, so the result is the same for every thread count -- but for float
// add and multiply it is not numpy's bits: the order of the operations
// changed (docs/design/array.md §6.5, §6.9).

constexpr std::int64_t kMaxBlocks = 16;
// Each block costs extra passes over the output (its start-value copy and
// its fold at the end). Measured on one thread, sum(axis=0) of (64, 2000) in
// 8 blocks of 8 rows was 30% slower; at least 32 indices per block keeps
// that near a tenth.
constexpr std::int64_t kMinBlock = 32;

struct Blocks {
    int axis = -1;
    std::int64_t size = 0, count = 0;  // count == 0: do not block
};

Blocks plan_blocks(const NDArray& x, const NDArray& out, const NdIter& it) {
    // A full reduction in x's own dtype already splits its long runs (the
    // kernel's pieces, or the pairwise sum's blocks). With a cast it does
    // not: buffered casting hands the kernel 4096 elements at a time, too
    // few to split, so sum(bool) and mean(int64) ran on one thread (review
    // fix: 1.47 ms on 1 thread, 1.33 ms on 10 for 1e7 bools). Blocking
    // gives each block its own serial buffered fold.
    if (out.size() <= 1 && x.dtype == out.dtype) return {};
    const NdIter::Plan p = it.plan(Split::Reduction);
    if (p.tasks >= 8 || p.work < 8 * kParallelThreshold) return {};  // run() splits well enough, or too small
    Blocks b;
    for (int d = 0; d < x.ndim; ++d)  // the longest reduced axis
        if (out.shape[d] == 1 && x.shape[d] > 1 && (b.axis < 0 || x.shape[d] > x.shape[b.axis])) b.axis = d;
    if (b.axis < 0) return {};
    const std::int64_t extent = x.shape[b.axis];
    const std::int64_t want = std::min({kMaxBlocks, extent / kMinBlock, p.work / (8 * kParallelTask)});
    if (want < 2 || want <= p.tasks) return {};
    b.size = (extent + want - 1) / want;
    b.count = (extent + b.size - 1) / b.size;
    return b;
}

// fold(x_block, part) folds one block of x into one partial output; combine
// is the reduction loop in out's dtype. Every fold here has a start value
// that the extra fold into `out` leaves alone (0 + p, 1 * p, max(lowest, p),
// false || p, ...), so out keeps it while the partials each hold it too.
template <typename Fold>
void reduce_blocked(const NDArray& x, NDArray& out, const Blocks& b, Loop combine, Fold fold) {
    const std::int64_t bytes = out.nbytes();  // dense, non-negative strides (new_array_like)
    auto store = std::make_unique_for_overwrite<std::byte[]>(static_cast<size_t>(b.count * bytes));
    // The views are made here: copying an NDArray touches Python references,
    // which the tasks (running without the GIL) must not.
    std::vector<NDArray> xs(b.count, x), parts(b.count, out);
    for (std::int64_t t = 0; t < b.count; ++t) {
        const std::int64_t begin = t * b.size;
        xs[t].shape[b.axis] = std::min(x.shape[b.axis] - begin, b.size);
        xs[t].data += begin * x.strides[b.axis];
        parts[t].data = store.get() + t * bytes;
    }
    parallel::parallel_for(b.count, [&](std::int64_t t) {
        std::memcpy(parts[t].data, out.data, static_cast<size_t>(bytes));
        fold(xs[t], parts[t]);
    });
    NDArray all = out;
    all.data = store.get();
    all.shape[b.axis] = b.count;
    all.strides[b.axis] = bytes;
    const IterOperand ops[] = {operand(all), operand(out)};
    NdIter(all.shape_span(), ops).run(combine, nullptr, Split::Reduction);
}

// x folded into `out` (keepdims shape, pre-filled with the starting value) by
// `loop`, which works in out's dtype.
void run_reduce(const NDArray& x, NDArray& out, Loop loop) {
    const IterOperand ops[] = {operand(x), operand(out)};
    const NdIter it(x.shape_span(), ops);
    if (const Blocks b = plan_blocks(x, out, it); b.count) {
        reduce_blocked(x, out, b, loop, [&](const NDArray& xb, NDArray& part) {
            const IterOperand block_ops[] = {operand(xb), operand(part)};
            run_fold(NdIter(xb.shape_span(), block_ops), x.dtype, out.dtype, loop);
        });
        return;
    }
    run_fold(it, x.dtype, out.dtype, loop);
}

NDArray as_dtype(const NDArray& x, DType d) { return x.dtype == d ? x : cast_copy(x, d); }

[[noreturn]] void unsupported(DType d) {
    throw nb::type_error((std::string("not supported for ") + dtype_name(d)).c_str());
}

// sum/prod/cumulative dtype: bool accumulates in int64; dtype= is an
// explicit conversion but never bool: add and multiply have no bool loop
// (docs/design/array.md §6.1, §2.4).
DType accumulate_dtype(const NDArray& x, nb::handle dtype_obj) {
    const std::optional<DType> d = parse_dtype(dtype_obj);
    if (!d) return promote(x.dtype, DType::Int64);
    if (*d == DType::Bool) throw nb::type_error("cannot accumulate in bool; use any() or all()");
    check_castable(x.dtype, *d);
    return *d;
}

// --------------------------------------------------------- sum, prod, any, all

template <typename Op>
nb::object fold(const NDArray& x, const Axes& ax, DType loop_dtype, bool keepdims, auto start) {
    NDArray out = keepdims_result(x, ax, loop_dtype);
    visit_dtype(loop_dtype, [&]<typename T>(std::type_identity<T>) {
        if constexpr (Op::template ok<T>) {
            fill<T>(out, start(std::type_identity<T>{}));
            run_reduce(x, out, &reduce_loop<Op, T>);
        } else {
            unsupported(loop_dtype);
        }
    });
    return finish(std::move(out), ax, keepdims);
}

// min/max have no identity: an output with nothing to fold is an error.
constexpr const char* kEmpty = "undefined for an empty array (or along an empty axis)";

void require_nonempty(const NDArray& x, const Axes& ax) {
    if (ax.per_output == 0 && output_size(x, ax) > 0) throw nb::value_error(kEmpty);
}

// ---------------------------------------------------------- argmin / argmax

using ArgFn = std::int64_t (*)(const std::byte*, std::int64_t, std::int64_t);

void arg_lane(const std::byte* x, std::byte* o, const LaneCtx& c) {
    *reinterpret_cast<std::int64_t*>(o) = (*static_cast<const ArgFn*>(c.user))(x, c.n, c.x_stride);
}

// argmin/argmax of one long contiguous lane on several threads: each piece
// finds its own answer, then the pieces are compared in order -- the first
// piece with a NaN wins, otherwise the best value, earlier piece on a tie --
// so the answer is the serial one exactly.
std::int64_t arg_whole(const std::byte* p, std::int64_t n, std::int64_t item, DType dtype, ArgFn lane, bool max) {
    if (n < kParallelThreshold || !parallel::available()) return lane(p, n, item);
    const std::int64_t pieces = (n + kParallelTask - 1) / kParallelTask;
    std::vector<std::int64_t> at(pieces);
    parallel::parallel_for(pieces, [&](std::int64_t t) {
        const std::int64_t begin = t * kParallelTask;
        at[t] = begin + lane(p + begin * item, std::min(kParallelTask, n - begin), item);
    });
    return visit_dtype(dtype, [&]<typename T>(std::type_identity<T>) -> std::int64_t {
        if constexpr (ops::is_cplx<T>) {
            return at[0];  // unreachable: complex is rejected before
        } else {
            std::int64_t best = at[0];
            for (std::int64_t t = 0; t < pieces; ++t) {
                const T v = load_as<T>(p + at[t] * item);
                if constexpr (ops::is_float<T>) {
                    if (std::isnan(v)) return at[t];  // the first NaN overall
                }
                if (t > 0 && (max ? v > load_as<T>(p + best * item) : v < load_as<T>(p + best * item))) best = at[t];
            }
            return best;
        }
    });
}

template <bool Max>
nb::object arg_extreme_impl(nb::handle x_obj, nb::handle axis_obj, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) unsupported(x.dtype);  // docs/design/array.md §2.6
    // complex was rejected above; its branch only satisfies visit_dtype.
    ArgFn lane_fn = visit_dtype(x.dtype, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_cplx<T>) return &arg_extreme<double, Max>;
        else return &arg_extreme<T, Max>;
    });
    auto empty_error = [] { return nb::value_error(kEmpty); };

    if (axis_obj.is_none()) {
        // Over everything: the index into the flattened (C order) array.
        if (x.size() == 0) throw empty_error();
        NDArray flat = x;
        if (!x.c_contiguous) {
            flat = new_array(x.dtype, x.shape_span());
            assign(flat, x);
        }
        const std::int64_t at = arg_whole(flat.data, x.size(), x.itemsize(), x.dtype, lane_fn, Max);
        if (!keepdims) return to_python(at);
        Axes all;
        for (int d = 0; d < x.ndim; ++d) all.reduced[d] = true;
        NDArray out = keepdims_result(x, all, DType::Int64);
        *reinterpret_cast<std::int64_t*>(out.data) = at;
        return wrap(std::move(out));
    }

    const int axis = normalize_axis(parse_int(axis_obj, "axis"), x.ndim);  // int only: docs/design/array.md §2.8
    Axes ax;
    ax.reduced[axis] = true;
    ax.count = 1;
    ax.per_output = x.shape[axis];
    if (ax.per_output == 0 && output_size(x, ax) > 0) throw empty_error();
    NDArray out = keepdims_result(x, ax, DType::Int64);
    LaneCtx ctx{x.shape[axis], x.strides[axis], 0, &arg_lane, &lane_fn};
    if (ctx.n > 0) run_lanes(x, out, axis, ctx);
    return finish(std::move(out), ax, keepdims);
}

// ------------------------------------------------------------ cumulative

template <typename Op, typename T>
void scan_fn(const std::byte* x, std::byte* o, const LaneCtx& c) {
    scan_lane<Op, T>(x, c.x_stride, o, c.o_stride, c.n);
}

template <typename Op, typename T>
void scan_with_initial_fn(const std::byte* x, std::byte* o, const LaneCtx& c) {
    // Array API: the initial value is the identity, written before the scan.
    *reinterpret_cast<T*>(o) = std::is_same_v<Op, ops::Add> ? T(0) : T(1);
    scan_lane<Op, T>(x, c.x_stride, o + c.o_stride, c.o_stride, c.n);
}

template <typename Op>
nb::object cumulative(nb::handle x_obj, nb::handle axis_obj, nb::handle dtype_obj, bool include_initial) {
    const NDArray& x = require_ndarray(x_obj);
    int axis;
    if (axis_obj.is_none()) {
        if (x.ndim != 1)
            // docs/design/array.md §2.8
            throw nb::value_error("'axis' is required for arrays with more than one dimension");
        axis = 0;
    } else {
        axis = normalize_axis(parse_int(axis_obj, "axis"), x.ndim);
    }
    const DType d = accumulate_dtype(x, dtype_obj);
    const NDArray xin = as_dtype(x, d);
    std::vector<std::int64_t> shape(x.shape_span().begin(), x.shape_span().end());
    if (include_initial) shape[axis] += 1;
    NDArray out = new_array(d, shape);
    LaneCtx ctx{x.shape[axis], xin.strides[axis], out.strides[axis], nullptr};
    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (Op::template ok<T>)  // bool was rejected by accumulate_dtype
            ctx.fn = include_initial ? &scan_with_initial_fn<Op, T> : &scan_fn<Op, T>;
    });
    if (include_initial || ctx.n > 0) run_lanes(xin, out, axis, ctx);
    return wrap(std::move(out));
}

// ------------------------------------------------------------- mean, var

// The dtype mean/var compute in: float64, or complex128 for complex input.
DType inexact(DType d) { return promote(d, DType::Float64); }

// Sum over `ax` in the inexact dtype, divided by the count, keepdims shape.
NDArray mean_keepdims(const NDArray& x, const Axes& ax) {
    const DType d = inexact(x.dtype);
    NDArray out = keepdims_result(x, ax, d, /*zeroed=*/true);  // +0.0 (and 0+0j): the start of a sum
    const double n = static_cast<double>(ax.per_output);
    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T> || ops::is_cplx<T>) {
            run_reduce(x, out, &reduce_loop<ops::Add, T>);
            T* p = reinterpret_cast<T*>(out.data);
            for (std::int64_t i = 0; i < out.size(); ++i) p[i] = p[i] / n;  // empty: 0/0 = nan
        }
    });
    return out;
}

double parse_correction(nb::handle o) {
    if (PyBool_Check(o.ptr()) || !(PyLong_Check(o.ptr()) || PyFloat_Check(o.ptr())))
        throw nb::type_error("'correction' must be an int or a float");
    const double c = PyFloat_AsDouble(o.ptr());
    if (c == -1.0 && PyErr_Occurred()) throw nb::python_error();
    return c;
}

// Two passes: the mean, then the sum of squared deviations from it, divided
// by N - correction. That is NaN whenever N - correction <= 0 (Array API,
// docs/design/array.md §6.8), which also covers empty input.
NDArray var_keepdims(nb::handle x_obj, nb::handle axis_obj, nb::handle correction_obj, Axes& ax) {
    const NDArray& x = require_ndarray(x_obj);
    const double correction = parse_correction(correction_obj);
    ax = parse_reduce_axes(x, axis_obj);
    const NDArray m = mean_keepdims(x, ax);
    const NDArray xin = as_dtype(x, m.dtype);
    NDArray out = keepdims_result(x, ax, DType::Float64, /*zeroed=*/true);
    visit_dtype(m.dtype, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T> || ops::is_cplx<T>) {
            const IterOperand ops[] = {operand(xin), operand(m), operand(out)};
            const NdIter it(xin.shape_span(), ops);
            if (const Blocks b = plan_blocks(xin, out, it); b.count) {
                reduce_blocked(xin, out, b, &reduce_loop<ops::Add, double>, [&](const NDArray& xb, NDArray& part) {
                    const IterOperand block_ops[] = {operand(xb), operand(m), operand(part)};
                    NdIter(xb.shape_span(), block_ops).run(&sq_dev_loop<T>, nullptr, Split::Reduction);
                });
            } else {
                it.run(&sq_dev_loop<T>, nullptr, Split::Reduction);
            }
        }
    });
    const double dof = static_cast<double>(ax.per_output) - correction;
    double* p = reinterpret_cast<double*>(out.data);
    for (std::int64_t i = 0; i < out.size(); ++i)
        p[i] = dof > 0 ? p[i] / dof : std::numeric_limits<double>::quiet_NaN();
    return out;
}

}  // namespace

nb::object sum(nb::handle x_obj, nb::handle axis, nb::handle dtype, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    const Axes ax = parse_reduce_axes(x, axis);
    // Starts from +0, so an empty sum is 0 (and sum([-0.0]) is +0.0, as in numpy).
    return fold<ops::Add>(x, ax, accumulate_dtype(x, dtype), keepdims,
                          []<typename T>(std::type_identity<T>) { return T(0); });
}

nb::object prod(nb::handle x_obj, nb::handle axis, nb::handle dtype, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    const Axes ax = parse_reduce_axes(x, axis);
    return fold<ops::Multiply>(x, ax, accumulate_dtype(x, dtype), keepdims,
                               []<typename T>(std::type_identity<T>) { return T(1); });
}

// max starts from the lowest value of the dtype, min from the highest: then
// the first real element always replaces it (and -inf/-0.0 stay exact).
nb::object max(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) unsupported(x.dtype);  // docs/design/array.md §2.6
    const Axes ax = parse_reduce_axes(x, axis);
    require_nonempty(x, ax);
    return fold<ops::Maximum>(x, ax, x.dtype, keepdims, []<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T>) return -std::numeric_limits<double>::infinity();
        else return std::numeric_limits<T>::lowest();
    });
}

nb::object min(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) unsupported(x.dtype);
    const Axes ax = parse_reduce_axes(x, axis);
    require_nonempty(x, ax);
    return fold<ops::Minimum>(x, ax, x.dtype, keepdims, []<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T>) return std::numeric_limits<double>::infinity();
        else return std::numeric_limits<T>::max();
    });
}

// any/all look at x != 0 (every dtype; nan counts as true).
nb::object any(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    return fold<ops::LogicalOr>(x, parse_reduce_axes(x, axis), DType::Bool, keepdims,
                                []<typename T>(std::type_identity<T>) { return T(false); });
}

nb::object all(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    return fold<ops::LogicalAnd>(x, parse_reduce_axes(x, axis), DType::Bool, keepdims,
                                 []<typename T>(std::type_identity<T>) { return T(true); });
}

nb::object mean(nb::handle x_obj, nb::handle axis, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    const Axes ax = parse_reduce_axes(x, axis);
    return finish(mean_keepdims(x, ax), ax, keepdims);
}

nb::object var(nb::handle x, nb::handle axis, nb::handle correction, bool keepdims) {
    Axes ax;
    NDArray v = var_keepdims(x, axis, correction, ax);
    return finish(std::move(v), ax, keepdims);
}

nb::object stddev(nb::handle x, nb::handle axis, nb::handle correction, bool keepdims) {
    Axes ax;
    NDArray v = var_keepdims(x, axis, correction, ax);
    double* p = reinterpret_cast<double*>(v.data);
    for (std::int64_t i = 0; i < v.size(); ++i) p[i] = std::sqrt(p[i]);
    return finish(std::move(v), ax, keepdims);
}

nb::object argmin(nb::handle x, nb::handle axis, bool keepdims) { return arg_extreme_impl<false>(x, axis, keepdims); }
nb::object argmax(nb::handle x, nb::handle axis, bool keepdims) { return arg_extreme_impl<true>(x, axis, keepdims); }

void bind_reductions(nb::module_& m) {
    using nb::arg;
    const auto axis = arg("axis").none() = nb::none();
    const auto dtype = arg("dtype").none() = nb::none();
    const auto keepdims = arg("keepdims") = false;
    m.def("sum", named("sum", &sum), arg(), nb::kw_only(), axis, dtype, keepdims);
    m.def("prod", named("prod", &prod), arg(), nb::kw_only(), axis, dtype, keepdims);
    m.def("min", named("min", &min), arg(), nb::kw_only(), axis, keepdims);
    m.def("max", named("max", &max), arg(), nb::kw_only(), axis, keepdims);
    m.def("mean", named("mean", &mean), arg(), nb::kw_only(), axis, keepdims);
    m.def("var", named("var", &var), arg(), nb::kw_only(), axis, arg("correction") = 0, keepdims);
    m.def("std", named("std", &stddev), arg(), nb::kw_only(), axis, arg("correction") = 0, keepdims);
    m.def("argmin", named("argmin", &argmin), arg(), nb::kw_only(), axis, keepdims);
    m.def("argmax", named("argmax", &argmax), arg(), nb::kw_only(), axis, keepdims);
    m.def("any", named("any", &any), arg(), nb::kw_only(), axis, keepdims);
    m.def("all", named("all", &all), arg(), nb::kw_only(), axis, keepdims);
    m.def("cumulative_sum", named("cumulative_sum", &cumulative<ops::Add>), arg(), nb::kw_only(), axis, dtype,
          arg("include_initial") = false);
    m.def("cumulative_prod", named("cumulative_prod", &cumulative<ops::Multiply>), arg(), nb::kw_only(), axis, dtype,
          arg("include_initial") = false);
}

}  // namespace sd
