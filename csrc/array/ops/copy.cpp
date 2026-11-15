#include "array/ops/copy.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/cast.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <numeric>
#include <cstdint>
#include <string>

namespace sd {

void check_assignable(DType from, DType to) {
    if (from > to) {
        std::string msg = std::string("cannot implicitly convert ") + dtype_name(from) + " to " + dtype_name(to) +
                          " (that would lose information); convert explicitly with astype";
        throw nb::type_error(msg.c_str());
    }
}

namespace {

// [lo, hi) of the bytes an array can touch.
struct ByteRange {
    std::uintptr_t lo, hi;
};

ByteRange byte_range(const NDArray& a) {
    auto lo = reinterpret_cast<std::uintptr_t>(a.data), hi = lo;
    for (int d = 0; d < a.ndim; ++d) {
        const std::int64_t span = (a.shape[d] - 1) * a.strides[d];
        if (span < 0) lo -= static_cast<std::uintptr_t>(-span);
        else hi += static_cast<std::uintptr_t>(span);
    }
    return {lo, hi + static_cast<std::uintptr_t>(a.itemsize())};
}

void run_cast(const NDArray& src, NDArray& dst, IterMode mode) {
    const IterOperand ops[] = {operand(src), operand(dst)};  // inputs first, output last
    NdIter(dst.shape_span(), ops, mode).run(cast_kernel(src.dtype, dst.dtype));
}

}  // namespace

bool same_view(const NDArray& a, const NDArray& b) {
    return a.data == b.data && a.ndim == b.ndim && a.dtype == b.dtype &&
           std::equal(a.shape.begin(), a.shape.begin() + a.ndim, b.shape.begin()) &&
           std::equal(a.strides.begin(), a.strides.begin() + a.ndim, b.strides.begin());
}

bool may_share_memory(const NDArray& a, const NDArray& b) {
    if (a.size() == 0 || b.size() == 0) return false;
    const ByteRange ra = byte_range(a), rb = byte_range(b);
    if (!(ra.lo < rb.hi && rb.lo < ra.hi)) return false;
    // Interleaved views (x[::2] and x[1::2]) overlap in range but share no byte: every
    // element of either starts at its data pointer plus a multiple of g, the gcd of all
    // the strides, so with the starts d apart modulo g they are disjoint when an element
    // fits in the gap on both sides. (A sufficient test; without it x[::2] += x[1::2]
    // copied its operand first: 120 us of 370 for 10^6 elements.)
    std::int64_t g = 0;
    for (int d = 0; d < a.ndim; ++d)
        if (a.shape[d] > 1) g = std::gcd(g, a.strides[d]);
    for (int d = 0; d < b.ndim; ++d)
        if (b.shape[d] > 1) g = std::gcd(g, b.strides[d]);
    if (g > 0) {
        const auto pa = reinterpret_cast<std::uintptr_t>(a.data), pb = reinterpret_cast<std::uintptr_t>(b.data);
        const auto gap = static_cast<std::int64_t>((pa > pb ? pa - pb : pb - pa) % static_cast<std::uintptr_t>(g));
        const std::int64_t width = std::max(a.itemsize(), b.itemsize());
        if (gap >= width && g - gap >= width) return false;
    }
    return true;
}

void assign(NDArray& dst, const NDArray& src) {
    // x[i] op= y arrives here as x[i] = (the view x[i], already updated in place):
    // nothing to copy (it was a full pass over the data, 119 us for 10^6).
    if (same_view(dst, src)) return;
    if (may_share_memory(dst, src)) {
        // e.g. a[1:] = a[:-1]: walking forward would read values already overwritten.
        NDArray tmp = cast_copy(src, src.dtype);
        run_cast(tmp, dst, IterMode::Full);
        return;
    }
    run_cast(src, dst, IterMode::Full);
}

void assign_checked(NDArray& dst, const NDArray& src) {
    if (!dst.writeable) throw nb::value_error("assignment destination is read-only");
    check_assignable(src.dtype, dst.dtype);
    assign(dst, src);
}

NDArray cast_copy(const NDArray& src, DType to, IterMode mode) {
    check_castable(src.dtype, to);
    NDArray dst = new_array_like(src, to);
    run_cast(src, dst, mode);
    return dst;
}

nb::object astype(nb::handle x, nb::handle dtype_obj, nb::handle copy_obj) {
    const NDArray& a = require_ndarray(x);
    if (dtype_obj.is_none())  // the required dtype argument: None is a wrong value like any other
        throw nb::type_error("'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128 (or their names), not NoneType");
    const DType to = *parse_dtype(dtype_obj);
    const std::optional<bool> copy = parse_copy(copy_obj);
    if (to == a.dtype && copy != true) return nb::borrow(x);
    if (copy == false) throw nb::value_error("copy=False, but a dtype conversion requires a copy");
    return wrap(cast_copy(a, to));
}

namespace {

IterMode parse_mode(const std::string& s) {
    if (s == "naive") return IterMode::Naive;
    if (s == "reorder") return IterMode::Reorder;
    if (s == "full") return IterMode::Full;
    throw nb::value_error("mode must be 'naive', 'reorder' or 'full'");
}

// Private, for tests and measurements: the iteration plan after
// broadcasting/reordering/coalescing (`mode` picks how far to simplify).
nb::dict iter_plan(nb::list arrays, const std::string& mode) {
    std::vector<IterOperand> ops;
    std::vector<std::span<const std::int64_t>> shapes;
    for (nb::handle h : arrays) {
        const NDArray& a = require_ndarray(h, "every operand");
        ops.push_back(operand(a));
        shapes.push_back(a.shape_span());
    }
    if (ops.empty() || ops.size() > static_cast<size_t>(kMaxOperands))
        throw nb::value_error(("between 1 and " + std::to_string(kMaxOperands) + " operands").c_str());
    const std::vector<std::int64_t> shape = broadcast_shapes(shapes);
    const NdIter it(shape, ops, parse_mode(mode));
    nb::list ext, strides;
    for (int d = 0; d < it.ndim(); ++d) ext.append(it.extent(d));
    for (int k = 0; k < it.nop(); ++k) {
        nb::list s;
        for (int d = 0; d < it.ndim(); ++d) s.append(it.stride(k, d));
        strides.append(nb::tuple(s));
    }
    nb::dict out;
    out["shape"] = nb::tuple(ext);
    out["strides"] = strides;
    out["empty"] = it.empty();
    return out;
}

// Private: dst[...] = src with the setitem rules (writeable, upward
// conversion only, overlap: docs/design/array.md §7.5).
void copyto(nb::handle dst_obj, nb::handle src_obj) {
    assign_checked(require_ndarray(dst_obj, "dst"), require_ndarray(src_obj, "src"));
}

}  // namespace

void bind_copy(nb::module_& m) {
    using nb::arg;
    m.def("astype", named("astype", &astype), arg().none(), arg().none(), nb::kw_only(), arg("copy").none() = true);
    m.def("_iter_plan", &iter_plan, arg("arrays"), arg("mode") = "full");
    m.def("_copy_with_mode", [](const NDArray& a, const std::string& mode) {
        return wrap(cast_copy(a, a.dtype, parse_mode(mode)));
    });
    m.def("_copyto", &copyto);
    m.def("_may_share_memory", [](const NDArray& a, const NDArray& b) { return may_share_memory(a, b); });
}

}  // namespace sd
