// flip, roll, repeat, tile, unstack, meshgrid, tril, triu, diff
// (docs/api/array.md §4.4, T0b).
//
// Views where numpy returns views (flip, unstack); everything else is a new
// C-ordered array filled by assign() from views of x, or gathered by the
// advanced-indexing path (repeat).

#include "array/ops/manipulation.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/creation.hpp"
#include "array/ops/indexing.hpp"
#include "array/ops/shape.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace sd {

namespace {

nb::object core(const char* name) { return nb::module_::import_("stardust._core").attr(name); }

std::vector<std::int64_t> shape_of(const NDArray& a) { return {a.shape_span().begin(), a.shape_span().end()}; }

// x as a C-ordered 1-D array (a view when x already is one).
nb::object flattened(nb::handle x) {
    const std::int64_t n = as_ndarray(x).size();
    return reshape(x, std::span<const std::int64_t>(&n, 1), std::nullopt);
}

NDArray new_c_array(DType d, const std::vector<std::int64_t>& shape) { return new_array(d, shape); }

// A view of `a` restricted to [begin, begin + n) along `axis`.
NDArray range_view(const NDArray& a, int axis, std::int64_t begin, std::int64_t n) {
    NDArray v = a;
    v.data += begin * a.strides[axis];
    v.shape[axis] = n;
    v.update_flags();
    return v;
}

// ------------------------------------------------------------------- flip

nb::object flip(nb::handle x, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    std::vector<int> axes;
    if (axis_obj.is_none()) {
        for (int d = 0; d < a.ndim; ++d) axes.push_back(d);
    } else {
        axes = parse_axes(axis_obj, a.ndim);
    }
    NDArray v = view_of(x);
    for (int d : axes) {
        if (a.shape[d] > 0) v.data += (a.shape[d] - 1) * a.strides[d];
        v.strides[d] = -a.strides[d];
    }
    v.update_flags();
    return wrap(std::move(v));
}

// ------------------------------------------------------------------- roll

std::int64_t parse_shift(nb::handle o) {
    std::int64_t v = 0;
    if (index_value(o, v) != IndexResult::Ok)
        throw nb::type_error(("'shift' must be an int or a tuple of ints, not " + py_type_name(o)).c_str());
    return v;
}

nb::object roll(nb::handle x, nb::handle shift_obj, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    if (axis_obj.is_none()) {
        // Array API: the flattened array is rolled, then given x's shape back.
        if (PyTuple_Check(shift_obj.ptr()) || PyList_Check(shift_obj.ptr()))
            throw nb::type_error("'shift' must be an int when 'axis' is None");
        const nb::object flat = flattened(x);
        const nb::object rolled = roll(flat, shift_obj, nb::int_(0));
        return reshape(rolled, a.shape_span(), std::nullopt);
    }
    const std::vector<int> axes = parse_axes(axis_obj, a.ndim);
    std::vector<std::int64_t> shifts;
    if (PyTuple_Check(shift_obj.ptr()) || PyList_Check(shift_obj.ptr())) {
        for (nb::handle h : shift_obj) shifts.push_back(parse_shift(h));
        if (shifts.size() != axes.size())
            throw nb::value_error("'shift' and 'axis' must have the same length when both are tuples");
    } else {
        shifts.assign(axes.size(), parse_shift(shift_obj));
    }
    // Per rolled axis: the amount in [0, n). Each output is made of 2^k
    // blocks, one per choice of "front / back" piece along each rolled axis.
    struct Rolled {
        int axis;
        std::int64_t n, s;
    };
    std::vector<Rolled> rolled;
    for (size_t i = 0; i < axes.size(); ++i) {
        const std::int64_t n = a.shape[axes[i]];
        if (n == 0) continue;
        const std::int64_t s = ((shifts[i] % n) + n) % n;
        if (s) rolled.push_back({axes[i], n, s});
    }
    NDArray out = new_c_array(a.dtype, shape_of(a));
    const int k = static_cast<int>(rolled.size());
    for (int mask = 0; mask < (1 << k); ++mask) {
        NDArray src = a, dst = out;
        bool empty = false;
        for (int i = 0; i < k; ++i) {
            const Rolled& r = rolled[i];
            if (mask >> i & 1) {  // the back piece of x goes to the front
                src = range_view(src, r.axis, r.n - r.s, r.s);
                dst = range_view(dst, r.axis, 0, r.s);
            } else {
                src = range_view(src, r.axis, 0, r.n - r.s);
                dst = range_view(dst, r.axis, r.s, r.n - r.s);
            }
            empty = empty || src.shape[r.axis] == 0;
        }
        if (!empty) assign(dst, src);
    }
    return wrap(std::move(out));
}

// ----------------------------------------------------------------- repeat

nb::object repeat(nb::handle x, nb::handle repeats_obj, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    nb::object src = nb::borrow(x);
    int axis = 0;
    if (axis_obj.is_none()) src = flattened(x);  // numpy and Array API: the flattened array
    else axis = normalize_axis(parse_int(axis_obj, "axis"), a.ndim);
    const NDArray& s = as_ndarray(src);
    const std::int64_t n = s.shape[axis];

    std::vector<std::int64_t> counts;
    std::int64_t one = 0;
    if (index_value(repeats_obj, one) == IndexResult::Ok) {
        counts.assign(n, one);
    } else if (is_ndarray(repeats_obj)) {
        const NDArray& r = as_ndarray(repeats_obj);
        if (r.dtype != DType::Int64)
            throw nb::type_error(("'repeats' must be an int or an int64 array, not " + std::string(dtype_name(r.dtype)))
                                     .c_str());
        if (r.ndim != 1) throw nb::value_error("'repeats' must be 1-D");
        NDArray rc = new_array(DType::Int64, r.shape_span());
        assign(rc, r);
        const std::int64_t* p = reinterpret_cast<const std::int64_t*>(rc.data);
        if (r.shape[0] == 1) counts.assign(n, p[0]);
        else if (r.shape[0] == n) counts.assign(p, p + n);
        else
            throw nb::value_error(("'repeats' has length " + std::to_string(r.shape[0]) + " but the axis has length " +
                                   std::to_string(n))
                                      .c_str());
    } else {
        throw nb::type_error(("'repeats' must be an int or an int64 array, not " + py_type_name(repeats_obj)).c_str());
    }
    std::int64_t total = 0;
    for (std::int64_t c : counts) {
        if (c < 0) throw nb::value_error("'repeats' must be non-negative");
        total += c;
    }
    // Direct block copies from a C-ordered source: for each position before
    // the axis, each element (block of the trailing axes) is written
    // counts[i] times. (Gathering through an index array was 1.3x slower.)
    NDArray src_c = s;
    if (!s.c_contiguous) {
        src_c = new_array(s.dtype, s.shape_span());
        if (src_c.size() > 0) assign(src_c, s);
    }
    std::vector<std::int64_t> shape = shape_of(s);
    shape[axis] = total;
    NDArray out = new_array(s.dtype, shape);
    if (out.size() == 0) return wrap(std::move(out));
    std::int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d) outer *= s.shape[d];
    for (int d = axis + 1; d < s.ndim; ++d) inner *= s.shape[d];
    const std::int64_t item = s.itemsize(), block = inner * item;
    std::vector<std::int64_t> start(n + 1, 0);  // where element i's copies begin along the axis
    for (std::int64_t i = 0; i < n; ++i) start[i + 1] = start[i] + counts[i];
    visit_dtype(s.dtype, [&]<typename T>(std::type_identity<T>) {
        // Rows [o0, o1) of the leading axes, elements [i0, i1) of the axis:
        // each piece writes its own part of the output.
        auto piece = [&](std::int64_t o0, std::int64_t o1, std::int64_t i0, std::int64_t i1) {
            for (std::int64_t o = o0; o < o1; ++o) {
                const std::byte* from = src_c.data + (o * n + i0) * block;
                std::byte* to = out.data + (o * total + start[i0]) * block;
                for (std::int64_t i = i0; i < i1; ++i, from += block) {
                    if (inner == 1) {
                        std::fill_n(reinterpret_cast<T*>(to), counts[i], *reinterpret_cast<const T*>(from));
                        to += counts[i] * item;
                    } else {
                        for (std::int64_t c = 0; c < counts[i]; ++c, to += block)
                            std::memcpy(to, from, static_cast<size_t>(block));
                    }
                }
            }
        };
        if (out.size() < kParallelThreshold || !parallel::available()) {
            piece(0, outer, 0, n);
        } else if (outer > 1) {
            const std::int64_t rows = std::max<std::int64_t>(1, kParallelTask / std::max<std::int64_t>(total * inner, 1));
            parallel::parallel_for((outer + rows - 1) / rows, [&](std::int64_t t) {
                piece(t * rows, std::min(outer, (t + 1) * rows), 0, n);
            });
        } else {
            const std::int64_t per = std::max<std::int64_t>(1, kParallelTask / std::max<std::int64_t>(inner, 1));
            parallel::parallel_for((n + per - 1) / per, [&](std::int64_t t) { piece(0, 1, t * per, std::min(n, (t + 1) * per)); });
        }
    });
    return wrap(std::move(out));
}

// ------------------------------------------------------------------- tile

std::vector<std::int64_t> parse_reps(nb::handle o) {
    std::vector<std::int64_t> reps;
    auto one = [](nb::handle h) {
        std::int64_t v = 0;
        if (index_value(h, v) != IndexResult::Ok)
            throw nb::type_error(("'repetitions' must be a tuple of ints, not " + py_type_name(h)).c_str());
        if (v < 0) throw nb::value_error("'repetitions' must be non-negative");
        return v;
    };
    if (PyTuple_Check(o.ptr()) || PyList_Check(o.ptr())) {
        for (nb::handle h : o) reps.push_back(one(h));
    } else {
        reps.push_back(one(o));  // numpy: an int is (int,)
    }
    return reps;
}

nb::object tile(nb::handle x, nb::handle reps_obj) {
    const NDArray& a = require_ndarray(x);
    std::vector<std::int64_t> reps = parse_reps(reps_obj);
    // Align: missing leading entries of either side count as 1.
    const int nd = std::max<int>(a.ndim, static_cast<int>(reps.size()));
    if (nd > kMaxNdim) throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    reps.insert(reps.begin(), nd - reps.size(), 1);
    NDArray cur = a;
    {
        std::vector<std::int64_t> shape(nd - a.ndim, 1);
        shape.insert(shape.end(), a.shape_span().begin(), a.shape_span().end());
        NDArray v = a;
        v.ndim = nd;
        for (int d = 0; d < nd; ++d) {
            v.shape[d] = shape[d];
            v.strides[d] = d < nd - a.ndim ? 0 : a.strides[d - (nd - a.ndim)];
        }
        v.update_flags();
        cur = v;
    }
    // One copy for as many repeated axes as fit: the view
    // (r0, s0, r1, s1, ...) with stride 0 on every r, copied in C order, is
    // the tiled array of shape (r0*s0, r1*s1, ...).
    std::vector<int> todo;
    for (int d = 0; d < nd; ++d)
        if (reps[d] != 1) todo.push_back(d);
    if (todo.empty()) {
        NDArray out = new_array(a.dtype, cur.shape_span());
        assign(out, cur);
        return wrap(std::move(out));
    }
    size_t next = 0;
    while (next < todo.size()) {
        const size_t room = static_cast<size_t>(kMaxNdim - nd);
        const size_t take = room ? std::min(room, todo.size() - next) : 1;
        std::vector<int> group(todo.begin() + next, todo.begin() + next + take);
        next += take;
        std::vector<std::int64_t> final_shape = shape_of(cur);
        for (int d : group) final_shape[d] *= reps[d];
        NDArray out = new_array(a.dtype, final_shape);
        if (room) {
            NDArray v = cur, dst = out;
            int k = 0;
            std::vector<std::int64_t> wide_shape;
            for (int d = 0; d < nd; ++d) {
                if (std::find(group.begin(), group.end(), d) != group.end()) {
                    v.shape[k] = reps[d];
                    v.strides[k] = 0;
                    wide_shape.push_back(reps[d]);
                    ++k;
                }
                v.shape[k] = cur.shape[d];
                v.strides[k] = cur.strides[d];
                wide_shape.push_back(cur.shape[d]);
                ++k;
            }
            v.ndim = k;
            v.update_flags();
            dst.ndim = k;
            std::int64_t stride = out.itemsize();
            for (int d = k - 1; d >= 0; --d) {
                dst.shape[d] = wide_shape[d];
                dst.strides[d] = stride;
                stride *= std::max<std::int64_t>(wide_shape[d], 1);
            }
            dst.update_flags();
            if (out.size() > 0) assign(dst, v);
        } else {
            // No axis left to spare (32-D): one block copy per repetition.
            const int d = group[0];
            for (std::int64_t r = 0; r < reps[d]; ++r) {
                NDArray dst = range_view(out, d, r * cur.shape[d], cur.shape[d]);
                if (dst.size() > 0) assign(dst, cur);
            }
        }
        cur = out;
    }
    return wrap(std::move(cur));
}

// ---------------------------------------------------------------- unstack

nb::tuple unstack(nb::handle x, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    const int axis = normalize_axis(parse_int(axis_obj, "axis"), a.ndim);
    nb::list out;
    for (std::int64_t i = 0; i < a.shape[axis]; ++i) {
        NDArray v = view_of(x);
        v.data += i * a.strides[axis];
        for (int d = axis; d + 1 < a.ndim; ++d) {
            v.shape[d] = a.shape[d + 1];
            v.strides[d] = a.strides[d + 1];
        }
        v.ndim = a.ndim - 1;
        if (v.ndim == 0)  // no 0-d arrays: a 1-D x unstacks into Python scalars
            out.append(load_python(a.dtype, v.data));
        else {
            v.update_flags();
            out.append(wrap(std::move(v)));
        }
    }
    return nb::tuple(out);
}

// --------------------------------------------------------------- meshgrid

nb::tuple meshgrid(nb::args arrays, nb::handle indexing_obj) {
    if (!nb::isinstance<nb::str>(indexing_obj))
        throw nb::type_error(("'indexing' must be a str, not " + py_type_name(indexing_obj)).c_str());
    const std::string indexing = nb::borrow<nb::str>(indexing_obj).c_str();
    if (indexing != "xy" && indexing != "ij")
        throw nb::value_error(("'indexing' must be 'xy' or 'ij', not '" + indexing + "'").c_str());
    std::vector<std::int64_t> shape;
    for (nb::handle h : arrays) {
        if (!is_ndarray(h)) throw nb::type_error(("every argument must be an sd.ndarray, not " + py_type_name(h)).c_str());
        const NDArray& a = as_ndarray(h);
        if (a.ndim != 1) throw nb::value_error("every array must be 1-D");
        shape.push_back(a.shape[0]);
    }
    const int n = static_cast<int>(shape.size());
    if (n > kMaxNdim) throw nb::value_error(("at most " + std::to_string(kMaxNdim) + " arrays").c_str());
    const bool xy = indexing == "xy" && n >= 2;
    if (xy) std::swap(shape[0], shape[1]);
    nb::list out;
    int i = 0;
    for (nb::handle h : arrays) {
        const NDArray& a = as_ndarray(h);
        const int along = xy && i < 2 ? 1 - i : i;  // 'xy': the first two arrays vary along axes 1 and 0
        // Filled directly in C order: value k of the input over the block of
        // the trailing axes, for every position of the leading ones.
        NDArray o = new_array(a.dtype, shape);
        std::int64_t outer = 1, inner = 1;
        for (int d = 0; d < along; ++d) outer *= shape[d];
        for (int d = along + 1; d < n; ++d) inner *= shape[d];
        const std::int64_t len = shape[along];
        if (o.size() > 0)
            visit_dtype(a.dtype, [&]<typename T>(std::type_identity<T>) {
                T* dst = reinterpret_cast<T*>(o.data);
                auto at = [&](std::int64_t k) { return *reinterpret_cast<const T*>(a.data + k * a.strides[0]); };
                if (inner == 1) {  // the input itself is the last axis: copy it once, then whole rows
                    for (std::int64_t k = 0; k < len; ++k) dst[k] = at(k);
                    for (std::int64_t r = 1; r < outer; ++r) std::memcpy(dst + r * len, dst, static_cast<size_t>(len) * sizeof(T));
                } else {
                    for (std::int64_t r = 0; r < outer; ++r)
                        for (std::int64_t k = 0; k < len; ++k, dst += inner) std::fill_n(dst, inner, at(k));
                }
            });
        out.append(wrap(std::move(o)));
        ++i;
    }
    return nb::tuple(out);
}

// ------------------------------------------------------------- tril, triu

template <bool Lower>
nb::object triangle(nb::handle x, nb::handle k_obj) {
    const NDArray& a = require_ndarray(x);
    const std::int64_t k = parse_int(k_obj, "k");
    if (a.ndim < 2) throw nb::value_error("'x' must have at least 2 dimensions");
    NDArray out = new_array(a.dtype, a.shape_span());
    if (out.size() == 0) return wrap(std::move(out));
    assign(out, a);
    const std::int64_t rows = a.shape[a.ndim - 2], cols = a.shape[a.ndim - 1], item = out.itemsize();
    const std::int64_t mats = out.size() / (rows * cols);
    std::byte* p = out.data;
    for (std::int64_t m = 0; m < mats; ++m)
        for (std::int64_t i = 0; i < rows; ++i) {
            std::byte* row = p + (m * rows + i) * cols * item;
            // tril keeps j <= i + k; triu keeps j >= i + k. All-zero bytes are
            // 0 / False / +0.0 / 0j in every dtype.
            std::int64_t lo, hi;  // the zeroed columns [lo, hi)
            if constexpr (Lower) {
                lo = std::clamp<std::int64_t>(i + k + 1, 0, cols);
                hi = cols;
            } else {
                lo = 0;
                hi = std::clamp<std::int64_t>(i + k, 0, cols);
            }
            if (hi > lo) std::memset(row + lo * item, 0, static_cast<size_t>((hi - lo) * item));
        }
    return wrap(std::move(out));
}

// ------------------------------------------------------------------- diff

// prepend/append: an array, used as is, or a Python scalar, broadcast to x's
// shape with length 1 along the axis (numpy).
nb::object edge(nb::handle v, const NDArray& a, int axis) {
    if (is_ndarray(v)) return nb::borrow(v);
    std::vector<std::int64_t> shape = shape_of(a);
    shape[axis] = 1;
    nb::list s;
    for (std::int64_t e : shape) s.append(nb::int_(e));
    return full(nb::tuple(s), v, nb::none());
}

nb::object diff(nb::handle x, nb::handle axis_obj, nb::handle n_obj, nb::handle prepend, nb::handle append) {
    const NDArray& a = require_ndarray(x);
    const int axis = normalize_axis(parse_int(axis_obj, "axis"), a.ndim);
    const std::int64_t n = parse_int(n_obj, "n");
    if (n < 0) throw nb::value_error("'n' must be non-negative");
    nb::object cur = nb::borrow(x);
    if (!prepend.is_none() || !append.is_none()) {
        nb::list parts;
        if (!prepend.is_none()) parts.append(edge(prepend, a, axis));
        parts.append(cur);
        if (!append.is_none()) parts.append(edge(append, a, axis));
        cur = core("concat")(parts, nb::arg("axis") = axis);
    }
    // n times: x[1:] - x[:-1] along the axis, with sd.subtract (bool becomes
    // int64, as in subtract; numpy uses not_equal there).
    const nb::object subtract = core("subtract");
    for (std::int64_t i = 0; i < n; ++i) {
        const NDArray& c = as_ndarray(cur);
        const std::int64_t len = c.shape[axis];
        if (len == 0) break;
        NDArray hi = range_view(view_of(cur), axis, 1, len - 1);
        NDArray lo = range_view(view_of(cur), axis, 0, len - 1);
        cur = subtract(wrap(std::move(hi)), wrap(std::move(lo)));
    }
    return cur;
}

}  // namespace

void bind_manipulation(nb::module_& m) {
    using nb::arg;
    m.def("flip", named("flip", &flip), arg().none(), nb::kw_only(), arg("axis").none() = nb::none());
    m.def("roll", named("roll", &roll), arg().none(), arg("shift").none(), nb::kw_only(), arg("axis").none() = nb::none());
    m.def("repeat", named("repeat", &repeat), arg().none(), arg().none(), nb::kw_only(), arg("axis").none() = nb::none());
    m.def("tile", named("tile", &tile), arg().none(), arg().none());
    m.def("unstack", named("unstack", &unstack), arg().none(), nb::kw_only(), arg("axis").none() = 0);
    m.def("meshgrid", named("meshgrid", &meshgrid), arg("arrays"), nb::kw_only(), arg("indexing").none() = "xy");
    m.def("tril", named("tril", &triangle<true>), arg().none(), nb::kw_only(), arg("k").none() = 0);
    m.def("triu", named("triu", &triangle<false>), arg().none(), nb::kw_only(), arg("k").none() = 0);
    m.def("diff", named("diff", &diff), arg().none(), nb::kw_only(), arg("axis").none() = -1, arg("n").none() = 1,
          arg("prepend").none() = nb::none(), arg("append").none() = nb::none());
}

}  // namespace sd
