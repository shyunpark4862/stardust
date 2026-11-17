#include "array/ops/ufunc.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/buffered.hpp"
#include "array/kernels/cast.hpp"
#include "array/ops/copy.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace sd {

namespace {

constexpr int kMaxInputs = kMaxOperands - 1;

// `what` names the argument: 'x', 'x1', 'min', ...
DType operand_dtype(nb::handle o, const char* what) {
    if (is_ndarray(o)) return as_ndarray(o).dtype;
    if (std::optional<DType> kind = scalar_dtype(o)) return *kind;
    throw nb::type_error((std::string("'") + what + "' must be an sd.ndarray or a bool, int, float or complex, not " +
                          py_type_name(o))
                             .c_str());
}

// The function: "not supported for complex128 and complex128". An operator
// (see ufunc.hpp): "unsupported operand dtypes for <: complex128 and complex128".
[[noreturn]] void no_loop(std::span<const DType> dtypes, const char* symbol) {
    std::string names;  // "a", "a and b", "a, b and c"
    for (size_t i = 0; i < dtypes.size(); ++i)
        names += (i == 0 ? "" : i + 1 == dtypes.size() ? " and " : ", ") + std::string(dtype_name(dtypes[i]));
    if (symbol == nullptr) throw nb::type_error(("not supported for " + names).c_str());
    throw nb::type_error((std::string("unsupported operand ") + (dtypes.size() > 1 ? "dtypes" : "dtype") + " for " +
                          symbol + ": " + names)
                             .c_str());
}

// numpy's loop selection: the first dtype at or above `d` with a loop.
// Returns that dtype; the table entry says what it writes.
DType select(const LoopTable& table, DType d, std::span<const DType> inputs, const char* symbol) {
    for (int k = static_cast<int>(d); k < kNumArrayDTypes; ++k)
        if (table[k].loop) return static_cast<DType>(k);
    no_loop(inputs, symbol);
}

// A Python int operand becomes an int64 when the promoted operand dtype is
// bool or int64, whatever loop then runs (docs/api/array.md §1.5): so
// divide(int_array, 2**64) and sqrt(2**64) are an OverflowError like
// add(int_array, 2**64), while next to a float or complex operand the int
// becomes a float. prepare() alone would only see the loop dtype (float64
// for divide), so the range is checked here first.
void check_int_scalars(std::span<const nb::handle> objs, DType promoted) {
    if (promoted > DType::Int64) return;
    for (nb::handle o : objs)
        if (!is_ndarray(o) && scalar_dtype(o) == DType::Int64) (void)scalar_from_python<std::int64_t>(o, DType::Int64);
}

// Operand `o` as an array. Arrays are used as they are -- a dtype that
// differs from the loop's is converted chunk by chunk while the loop runs
// (buffered casting, docs/design/array.md §4.3). A Python scalar becomes a
// 1-element array of the loop dtype `to` (broadcast by the iterator);
// conversion follows docs/design/array.md §2.2 (an int that does not fit
// was already rejected by check_int_scalars where it must be an int64).
NDArray prepare(nb::handle o, DType to) {
    if (is_ndarray(o)) return as_ndarray(o);
    const DType kind = *scalar_dtype(o);
    const std::int64_t one[] = {1};
    NDArray s = new_array(to, one);
    visit_dtype(to, [&]<typename T>(std::type_identity<T>) {
        *reinterpret_cast<T*>(s.data) = scalar_from_python<T>(o, kind);
    });
    return s;
}

// A new result array: laid out like the first array operand that already has
// the full shape (an F-ordered input gives an F-ordered result, keeping the
// iterator's coalescing), else C order.
NDArray new_result(std::span<const nb::handle> objs, std::span<const std::int64_t> shape, DType dtype) {
    for (nb::handle o : objs)
        if (is_ndarray(o) && std::ranges::equal(as_ndarray(o).shape_span(), shape))
            return new_array_like(as_ndarray(o), dtype);
    return new_array(dtype, shape);
}

// Runs `loop` over ins + outs. Inputs whose dtype is not the loop's
// (`in_types`) are converted through small buffers on the way in. `weight`
// is the loop's work per element in units of an add (the iterator's
// parallel threshold, docs/design/special.md §3.3); 1 for the array's own.
void run_loop(std::span<const NDArray> ins, std::span<const DType> in_types, std::span<const NDArray> outs,
              std::span<const std::int64_t> shape, Loop loop, std::int64_t weight) {
    std::array<IterOperand, kMaxOperands> ops;
    BufferedLoop buffered;
    buffered.inner = loop;
    buffered.nin = static_cast<int>(ins.size());
    buffered.nop = buffered.nin + static_cast<int>(outs.size());
    for (size_t k = 0; k < ins.size(); ++k) {
        ops[k] = operand(ins[k]);
        if (ins[k].dtype != in_types[k])
            buffered.add_cast(static_cast<int>(k), cast_kernel(ins[k].dtype, in_types[k]), itemsize(in_types[k]));
    }
    for (size_t j = 0; j < outs.size(); ++j) ops[ins.size() + j] = operand(outs[j]);
    NdIter it(shape, std::span(ops.data(), ins.size() + outs.size()));
    it.allow_tiles();  // ufunc kernels treat every element on its own
    if (buffered.needed()) it.run(&buffered_loop, &buffered, Split::Elementwise, weight);
    else it.run(loop, nullptr, Split::Elementwise, weight);
}

// Every operand a Python scalar and no out=: the loop runs once on stack
// buffers (n = 1), with no arrays and no iterator. The same loop the array path
// runs, on the same converted values: the same bits (f(x)[i] == f(x[i])); about
// 90 ns less per call.
nb::object launch_scalars(std::span<const nb::handle> objs, std::span<const DType> in_types, DType result, Loop loop,
                          int nout) {
    const size_t nin = objs.size();
    alignas(16) std::byte buf[kMaxOperands][16];
    std::array<std::byte*, kMaxOperands> args;
    std::array<std::int64_t, kMaxOperands> strides;
    for (size_t k = 0; k < nin; ++k) {
        const DType kind = *scalar_dtype(objs[k]);
        visit_dtype(in_types[k], [&]<typename T>(std::type_identity<T>) {
            *reinterpret_cast<T*>(buf[k]) = scalar_from_python<T>(objs[k], kind);
        });
        args[k] = buf[k];
        strides[k] = itemsize(in_types[k]);
    }
    for (int j = 0; j < nout; ++j) {
        args[nin + static_cast<size_t>(j)] = buf[nin + static_cast<size_t>(j)];
        strides[nin + static_cast<size_t>(j)] = itemsize(result);
    }
    loop(args.data(), 1, strides.data(), nullptr);
    if (nout == 1) return load_python(result, buf[nin]);
    nb::list results;
    for (int j = 0; j < nout; ++j) results.append(load_python(result, buf[nin + static_cast<size_t>(j)]));
    return nb::tuple(results);
}

// x[1:] += x[:-1] and the like on one thread: 1-D operands of one length and
// stride, the overlapping inputs in out's dtype. Rather than copying each
// overlapping input whole, chunks of kOverlapChunk elements go in the order
// that reads every input chunk before any write reaches it (from the end when
// the inputs start at or before out, else from the start), each overlapping
// input's chunk copied to a stack buffer first. The same result as the whole
// copy (numpy's semantics) without its pass over memory (10^6: 300 us against
// numpy's 238; chunked: see docs/design/array.md §3.5). False: not this case.
// On several threads the whole copy runs on the pool instead (same result).
// The chunks run without the GIL from the work at which that iterator run
// would (NdIter::plan: weight x bytes x elements, docs/design/array.md §10.2).
inline constexpr std::int64_t kOverlapChunk = 1024;  // 16 KB per buffered input (complex)

bool overlap_in_chunks(std::span<const NDArray> ins, std::span<const DType> in_types, const std::array<bool, kMaxInputs>& overlaps,
                       NDArray& out, Loop loop, std::int64_t weight) {
    if (out.ndim != 1 || parallel::available()) return false;
    const std::int64_t n = out.shape[0], st = out.strides[0], w = out.itemsize();
    if (st == 0 || n < 2 * kOverlapChunk) return false;
    int dir = 0;  // +1: chunks from the start, -1: from the end
    for (size_t k = 0; k < ins.size(); ++k) {
        const NDArray& a = ins[k];
        if (a.ndim != 1 || a.shape[0] != n || a.dtype != in_types[k]) return false;
        if (!overlaps[k]) continue;
        if (a.strides[0] != st || a.dtype != out.dtype) return false;
        const bool before = st > 0 ? a.data <= out.data : a.data >= out.data;
        const int d = before ? -1 : +1;
        if (dir != 0 && dir != d) return false;
        dir = d;
    }
    alignas(64) std::byte buf[kMaxInputs][kOverlapChunk * 16];
    std::array<std::byte*, kMaxOperands> args;
    std::array<std::int64_t, kMaxOperands> strides;
    const std::int64_t chunks = (n + kOverlapChunk - 1) / kOverlapChunk;
    const parallel::Unlocked unlocked(weight * std::min<std::int64_t>(w, 8) * n >= 8 * kParallelThreshold);
    for (std::int64_t c = 0; c < chunks; ++c) {
        const std::int64_t q = dir > 0 ? c : chunks - 1 - c;
        const std::int64_t i0 = q * kOverlapChunk, len = std::min(kOverlapChunk, n - i0);
        for (size_t k = 0; k < ins.size(); ++k) {
            const std::byte* src = ins[k].data + i0 * ins[k].strides[0];
            if (overlaps[k]) {
                if (st == w) {
                    std::memcpy(buf[k], src, static_cast<size_t>(len * w));
                } else {
                    visit_dtype(out.dtype, [&]<typename T>(std::type_identity<T>) {
                        T* d = reinterpret_cast<T*>(buf[k]);
                        for (std::int64_t i = 0; i < len; ++i) d[i] = *reinterpret_cast<const T*>(src + i * st);
                    });
                }
                args[k] = buf[k];
                strides[k] = w;
            } else {
                args[k] = const_cast<std::byte*>(src);
                strides[k] = ins[k].strides[0];
            }
        }
        args[ins.size()] = out.data + i0 * st;
        strides[ins.size()] = st;
        loop(args.data(), len, strides.data(), nullptr);
    }
    return true;
}

// Steps 3 and 4 (see ufunc.hpp): convert, then run into out= or a new array.
// With nout > 1 the loop writes nout results of dtype `result`, returned as
// a tuple; there is no out= then (the caller passes None).
nb::object launch(std::span<const nb::handle> objs, std::span<const DType> in_types, DType result, Loop loop,
                  nb::handle out_obj, int nout = 1, std::int64_t weight = 1) {
    const size_t nin = objs.size();
    if (out_obj.is_none() && std::ranges::none_of(objs, [](nb::handle o) { return is_ndarray(o); }))
        return launch_scalars(objs, in_types, result, loop, nout);
    std::array<NDArray, kMaxInputs> ins;
    std::array<std::span<const std::int64_t>, kMaxInputs> shapes;  // fixed arrays: no allocation per call
    size_t narrays = 0;
    for (size_t k = 0; k < nin; ++k) {
        ins[k] = prepare(objs[k], in_types[k]);
        if (is_ndarray(objs[k])) shapes[narrays++] = as_ndarray(objs[k]).shape_span();
    }
    const bool scalars_only = narrays == 0;
    Dims dims;
    dims[0] = 1;
    const int ndim = scalars_only ? 1 : broadcast_into(std::span(shapes.data(), narrays), dims);
    const std::span<const std::int64_t> shape(dims.data(), static_cast<size_t>(ndim));
    const std::span<const NDArray> inputs(ins.data(), nin);

    if (!out_obj.is_none()) {
        NDArray& out = require_ndarray(out_obj, "out");
        if (!out.writeable) throw nb::value_error("output array is read-only");
        check_assignable(result, out.dtype);  // docs/design/array.md §2.5
        if (out.dtype == result) {
            // Write straight into out. An input that partly overlaps it is
            // copied first (docs/design/array.md §3.5); one that is the very same
            // view is safe, since each element is read just before the same
            // place is written.
            std::array<bool, kMaxInputs> overlaps{};
            bool any_overlap = false;
            for (size_t k = 0; k < nin; ++k) {
                overlaps[k] = may_share_memory(ins[k], out) && !same_view(ins[k], out);
                any_overlap |= overlaps[k];
            }
            if (any_overlap && overlap_in_chunks(inputs, in_types, overlaps, out, loop, weight)) return nb::borrow(out_obj);
            for (size_t k = 0; k < nin; ++k)
                if (overlaps[k]) ins[k] = cast_copy(ins[k], ins[k].dtype);
            run_loop(inputs, in_types, std::span(&out, 1), out.shape_span(), loop, weight);
        } else {
            // out is a wider dtype: compute, then convert into it.
            NDArray tmp = new_result(objs, shape, result);
            run_loop(inputs, in_types, std::span(&tmp, 1), shape, loop, weight);
            assign(out, tmp);
        }
        return nb::borrow(out_obj);
    }

    if (nout == 1) {
        NDArray res = new_result(objs, shape, result);
        run_loop(inputs, in_types, std::span(&res, 1), shape, loop, weight);
        if (scalars_only) return load_python(result, res.data);
        return wrap(std::move(res));
    }
    std::array<NDArray, kMaxOperands - 1> res;
    for (int j = 0; j < nout; ++j) res[j] = new_result(objs, shape, result);
    run_loop(inputs, in_types, std::span(res.data(), static_cast<size_t>(nout)), shape, loop, weight);
    nb::list results;
    for (int j = 0; j < nout; ++j) {
        if (scalars_only) results.append(load_python(result, res[j].data));
        else results.append(wrap(std::move(res[j])));
    }
    return nb::tuple(results);
}

// ------------------------------------------------------------------ where, clip

template <typename T> constexpr Loop where_entry = &where_loop<T>;
constexpr std::array<Loop, kNumArrayDTypes> where_loops = {
    where_entry<bool>, where_entry<std::int64_t>, where_entry<double>, where_entry<complex128>};

nb::object where(nb::handle cond, nb::handle x1, nb::handle x2, nb::handle out) {
    const DType dc = operand_dtype(cond, "condition");
    if (dc != DType::Bool)
        throw nb::type_error(("'condition' must be bool, not " + std::string(dtype_name(dc)) +
                              "; compare explicitly, e.g. x != 0")
                                 .c_str());
    const DType d = promote(operand_dtype(x1, "x1"), operand_dtype(x2, "x2"));
    const nb::handle objs[] = {cond, x1, x2};
    const DType types[] = {DType::Bool, d, d};
    return launch(objs, types, d, where_loops[static_cast<int>(d)], out);
}

// clip keeps x's dtype (Array API): min and max must convert to it
// implicitly (docs/design/array.md §2.5). No complex: it has no order
// (docs/design/array.md §2.6).
template <typename T> constexpr Loop clip_entry = &clip_loop<T>;
constexpr std::array<Loop, kNumArrayDTypes> clip_loops = {
    clip_entry<bool>, clip_entry<std::int64_t>, clip_entry<double>, nullptr};

// An array min/max broadcasts to x, never x to it: the result has x's shape
// (docs/api/array.md §6.4).
void check_bound_shape(nb::handle x, nb::handle bound, const char* what) {
    if (!is_ndarray(x) || !is_ndarray(bound)) return;
    const std::span<const std::int64_t> xs = as_ndarray(x).shape_span(), bs = as_ndarray(bound).shape_span();
    bool ok = bs.size() <= xs.size();
    for (size_t d = 0; ok && d < bs.size(); ++d) {
        const std::int64_t e = bs[d], t = xs[xs.size() - bs.size() + d];
        ok = e == 1 || e == t;
    }
    if (!ok)
        throw nb::value_error((std::string("'") + what + "' of shape " + shape_str(bs) +
                               " cannot be broadcast to the shape of 'x' " + shape_str(xs))
                                  .c_str());
}

nb::object clip(nb::handle x, nb::handle lo, nb::handle hi, nb::handle out) {
    const DType d = operand_dtype(x, "x");
    if (d == DType::Complex128) throw nb::type_error("not supported for complex128");
    if (!lo.is_none()) check_assignable(operand_dtype(lo, "min"), d);
    if (!hi.is_none()) check_assignable(operand_dtype(hi, "max"), d);
    if (!lo.is_none()) check_bound_shape(x, lo, "min");
    if (!hi.is_none()) check_bound_shape(x, hi, "max");
    const int k = static_cast<int>(d);
    if (lo.is_none() && hi.is_none()) {
        const nb::handle objs[] = {x};
        const DType types[] = {d};
        return launch(objs, types, d, cast_kernel(d, d), out);  // a plain copy
    }
    if (hi.is_none()) {
        const nb::handle objs[] = {x, lo};
        const DType types[] = {d, d};
        return launch(objs, types, d, binary_table<ops::Maximum>[k].loop, out);
    }
    if (lo.is_none()) {
        const nb::handle objs[] = {x, hi};
        const DType types[] = {d, d};
        return launch(objs, types, d, binary_table<ops::Minimum>[k].loop, out);
    }
    const nb::handle objs[] = {x, lo, hi};
    const DType types[] = {d, d, d};
    return launch(objs, types, d, clip_loops[k], out);
}

}  // namespace

bool is_operand(nb::handle o) { return is_ndarray(o) || scalar_dtype(o).has_value(); }

nb::object run_unary(const LoopTable& table, nb::handle x, nb::handle out, const char* symbol) {
    const DType d = operand_dtype(x, "x");
    const DType loop = select(table, d, std::span(&d, 1), symbol);
    const nb::handle objs[] = {x};
    check_int_scalars(objs, d);
    return launch(objs, std::span(&loop, 1), table[static_cast<int>(loop)].out, table[static_cast<int>(loop)].loop,
                  out);
}

nb::object run_binary(const LoopTable& table, bool no_bool_inputs, nb::handle a, nb::handle b, nb::handle out,
                      const char* symbol) {
    const DType dtypes[] = {operand_dtype(a, "x1"), operand_dtype(b, "x2")};
    if (no_bool_inputs && (dtypes[0] == DType::Bool || dtypes[1] == DType::Bool)) no_loop(dtypes, symbol);
    const DType promoted = promote(dtypes[0], dtypes[1]);
    const DType loop = select(table, promoted, dtypes, symbol);
    const nb::handle objs[] = {a, b};
    check_int_scalars(objs, promoted);
    const DType types[] = {loop, loop};
    return launch(objs, types, table[static_cast<int>(loop)].out, table[static_cast<int>(loop)].loop, out);
}

nb::object run_nary(const LoopTable& table, int nout, std::span<const char* const> names,
                    std::span<const nb::handle> objs, nb::handle out, std::int64_t weight) {
    const size_t nin = objs.size();
    std::array<DType, kMaxInputs> dtypes;
    for (size_t k = 0; k < nin; ++k) dtypes[k] = operand_dtype(objs[k], names[k]);
    DType promoted = dtypes[0];
    for (size_t k = 1; k < nin; ++k) promoted = promote(promoted, dtypes[k]);
    const DType loop = select(table, promoted, std::span(dtypes.data(), nin), nullptr);
    check_int_scalars(objs, promoted);
    std::array<DType, kMaxInputs> types;
    types.fill(loop);
    const LoopEntry& e = table[static_cast<int>(loop)];
    return launch(objs, std::span(types.data(), nin), e.out, e.loop, out, nout, weight);
}

namespace {

template <typename Op>
void def_unary(nb::module_& m) {
    m.def(Op::name, named(Op::name, +[](nb::handle x, nb::handle out) { return unary<Op>(x, out); }), nb::arg().none(),
          nb::kw_only(), nb::arg("out").none() = nb::none());
}

template <typename Op>
void def_binary(nb::module_& m) {
    m.def(Op::name, named(Op::name, +[](nb::handle a, nb::handle b, nb::handle out) { return binary<Op>(a, b, out); }),
          nb::arg().none(), nb::arg().none(), nb::kw_only(), nb::arg("out").none() = nb::none());
}

template <typename... Ops>
void def_unaries(nb::module_& m) { (def_unary<Ops>(m), ...); }

template <typename... Ops>
void def_binaries(nb::module_& m) { (def_binary<Ops>(m), ...); }

}  // namespace

void bind_ufuncs(nb::module_& m) {
    using namespace ops;
    def_binaries<Add, Subtract, Multiply, Divide, FloorDivide, Remainder, Pow, Atan2, Maximum, Minimum,
                 Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual, LogicalAnd, LogicalOr, LogicalXor,
                 BitwiseAnd, BitwiseOr, BitwiseXor, LeftShift, RightShift, Copysign, Hypot, Logaddexp, Nextafter>(m);
    def_unaries<Negative, Positive, Abs, Square, Sqrt, Sign, Exp, Expm1, Log, Log1p, Log2, Log10, Sin, Cos, Tan,
                Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Floor, Ceil, Trunc, Round, IsNan, IsInf,
                IsFinite, LogicalNot, BitwiseInvert, Conj, Reciprocal, Signbit>(m);
    m.def("where", named("where", &where), nb::arg().none(), nb::arg().none(), nb::arg().none(), nb::kw_only(), nb::arg("out").none() = nb::none());
    m.def("clip", named("clip", &clip), nb::arg().none(), nb::kw_only(), nb::arg("min").none() = nb::none(),
          nb::arg("max").none() = nb::none(), nb::arg("out").none() = nb::none());
}

}  // namespace sd
