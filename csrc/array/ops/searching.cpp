// take, take_along_axis, nonzero, count_nonzero, searchsorted
// (docs/api/array.md §5.4, §8.4; T0b).

#include "array/ops/searching.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/elementwise.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/creation.hpp"
#include "array/ops/indexing.hpp"
#include "array/ops/lanes.hpp"
#include "array/ops/reduction.hpp"
#include "array/ops/pieces.hpp"
#include "parallel/pool.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace sd {

namespace {

nb::object full_slice() { return nb::slice(nb::none(), nb::none(), nb::none()); }

// x[key] for a function built on indexing: indexing keeps Python's message
// form and raises a Python exception, which the binding's name would not
// reach. Here its IndexError becomes a C++ one, so it gets "take(): " too.
nb::object gather_named(nb::handle x, nb::handle key) {
    try {
        return getitem(x, key);
    } catch (nb::python_error& e) {
        if (!e.matches(PyExc_IndexError)) throw;
        const std::string msg = nb::str(e.value()).c_str();
        throw nb::index_error(msg.c_str());
    }
}

// An index argument: an int64 array, or also a Python int when `int_too`.
// A bool array would be read as a mask by indexing, which is not what take
// means. One message for the argument, whatever is wrong (API.md §2.1).
void check_indices(nb::handle o, const char* what, bool int_too) {
    const std::string expected =
        "'" + std::string(what) + "' must be " + (int_too ? "an int or " : "") + "an int64 array, not ";
    std::int64_t v = 0;
    if (!is_ndarray(o) && (!int_too || index_value(o, v) == IndexResult::NotInt))
        throw nb::type_error((expected + py_type_name(o)).c_str());
    if (is_ndarray(o) && as_ndarray(o).dtype != DType::Int64)
        throw nb::type_error((expected + dtype_name(as_ndarray(o).dtype)).c_str());
}

// ------------------------------------------------------------ take

nb::object take(nb::handle x, nb::handle indices, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    check_indices(indices, "indices", true);
    int axis = 0;
    if (axis_obj.is_none()) {
        if (a.ndim != 1)  // docs/design/array.md §2.8, as cumulative_sum
            throw nb::value_error("'axis' is required for arrays with more than one dimension");
    } else {
        axis = normalize_axis(parse_int(axis_obj, "axis"), a.ndim);
    }
    // x[:, ..., :, indices]: the advanced-indexing gather (bounds checked,
    // negative indices from the end).
    nb::list key;
    for (int d = 0; d < axis; ++d) key.append(full_slice());
    key.append(indices);
    return gather_named(x, nb::tuple(key));
}

// take_along_axis runs elementwise over the result, in memory order: x is
// seen with stride 0 along `axis` (so each position points at x[..., 0, ...]
// of its own lane) and each element reads x_view + index * stride. Every
// axis walks memory in order -- lane by lane, axis=0 of a C matrix was 2.5x
// slower -- and the iterator splits the work over the pool. The indices were
// bounds-checked before.
struct AlongCtx {
    std::int64_t n_x, x_stride;
};

template <typename T>
void along_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void* ctx) {
    const AlongCtx& c = *static_cast<const AlongCtx*>(ctx);
    for (std::int64_t i = 0; i < n; ++i) {
        std::int64_t k = *reinterpret_cast<const std::int64_t*>(args[1] + i * s[1]);
        if (k < 0) k += c.n_x;
        *reinterpret_cast<T*>(args[2] + i * s[2]) = *reinterpret_cast<const T*>(args[0] + i * s[0] + k * c.x_stride);
    }
}

nb::object take_along_axis(nb::handle x, nb::handle indices, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    check_indices(indices, "indices", false);
    const NDArray& ix = as_ndarray(indices);
    if (ix.ndim != a.ndim)
        throw nb::value_error(("'indices' must have as many dimensions as 'x' (" + std::to_string(a.ndim) + "), not " +
                               std::to_string(ix.ndim))
                                  .c_str());
    const int axis = normalize_axis(parse_int(axis_obj, "axis"), a.ndim);
    const int nd = a.ndim;
    // The result: indices' length along the axis, the broadcast of both elsewhere.
    std::vector<std::int64_t> shape(nd);
    for (int d = 0; d < nd; ++d) {
        if (d == axis) {
            shape[d] = ix.shape[d];
        } else if (a.shape[d] == ix.shape[d] || ix.shape[d] == 1) {
            shape[d] = a.shape[d];
        } else if (a.shape[d] == 1) {
            shape[d] = ix.shape[d];
        } else {
            std::vector<std::int64_t> xs(a.shape_span().begin(), a.shape_span().end()),
                is(ix.shape_span().begin(), ix.shape_span().end());
            xs[axis] = is[axis] = 1;
            // IndexError, as for indexing arrays that do not broadcast (§5.2; numpy too).
            throw nb::index_error(
                ("shapes could not be broadcast together: " + shape_str(xs) + " " + shape_str(is)).c_str()
            );
        }
    }
    NDArray out = new_array(a.dtype, shape);
    if (out.size() == 0) return wrap(std::move(out));
    const std::int64_t n_x = a.shape[axis];

    // Bounds first (numpy's message for the first bad index in C order).
    NDArray ic = ix;
    if (!ix.c_contiguous) {
        ic = new_array(DType::Int64, ix.shape_span());
        assign(ic, ix);
    }
    {
        const std::int64_t* q = reinterpret_cast<const std::int64_t*>(ic.data);
        const std::int64_t m = ic.size();
        // A branch-free scan for any bad index (in parallel pieces when long); only then the first one.
        std::vector<char> bad_piece((m + kParallelTask - 1) / kParallelTask, 0);
        for_pieces(m, kParallelTask, m, [&](std::int64_t t, std::int64_t b0, std::int64_t e0) {
            bool b = false;
            for (std::int64_t i = b0; i < e0; ++i) b |= q[i] < -n_x || q[i] >= n_x;
            bad_piece[t] = b;
        });
        const bool bad = std::any_of(bad_piece.begin(), bad_piece.end(), [](char c) { return c != 0; });
        if (bad)
            for (std::int64_t i = 0; i < m; ++i)
                if (q[i] < -n_x || q[i] >= n_x)
                    throw nb::index_error(("index " + std::to_string(q[i]) + " is out of bounds for axis " +
                                           std::to_string(axis) + " with size " + std::to_string(n_x))
                                              .c_str());
    }

    // x and the indices seen with the result's shape (stride 0 where broadcast);
    // x also has stride 0 along the axis: the index supplies that offset.
    auto spread = [&](const NDArray& v, bool zero_axis) {
        NDArray b = v;
        for (int d = 0; d < nd; ++d)
            if ((d == axis && zero_axis) || (d != axis && v.shape[d] != shape[d]) ||
                (d == axis && v.shape[d] != shape[d])) {
                b.shape[d] = shape[d];
                b.strides[d] = 0;
            }
        b.update_flags();
        return b;
    };
    const NDArray xb = spread(a, true), ib = spread(ic, false);
    const IterOperand ops[] = {operand(xb), operand(ib), operand(out)};
    AlongCtx ctx{n_x, a.strides[axis]};
    Loop loop = nullptr;
    visit_dtype(a.dtype, [&]<typename T>(std::type_identity<T>) { loop = &along_loop<T>; });
    NdIter(out.shape_span(), ops).run(loop, &ctx, Split::Elementwise);
    return wrap(std::move(out));
}

// --------------------------------------------------------- nonzero


// Chunks of the C-order index range: each is counted, then filled at its
// prefix offset, both in parallel. The positions are the same for any split.
constexpr std::int64_t kNonzeroChunk = 1 << 16;

template <typename T>
bool truthy(const T& v) {
    if constexpr (ops::is_cplx<T>) return v.real() != 0 || v.imag() != 0;
    else return v != T(0);  // NaN is nonzero
}

nb::tuple nonzero(nb::handle x) {
    const NDArray& a = require_ndarray(x);
    // A C-ordered source: x itself when it already is one, else a copy in its own dtype.
    NDArray c = a;
    if (!a.c_contiguous) {
        c = new_array(a.dtype, a.shape_span());
        if (c.size() > 0) assign(c, a);
    }
    const std::int64_t size = c.size(), nd = a.ndim;
    const std::int64_t chunks = (size + kNonzeroChunk - 1) / kNonzeroChunk;
    std::vector<std::int64_t> start(chunks + 1, 0);
    std::vector<NDArray> out;
    std::vector<std::int64_t*> dst;
    visit_dtype(a.dtype, [&]<typename T>(std::type_identity<T>) {
        const T* p = reinterpret_cast<const T*>(c.data);
        for_pieces(size, kNonzeroChunk, size, [&](std::int64_t t, std::int64_t b, std::int64_t e) {
            std::int64_t k = 0, i = b;
            if constexpr (std::is_same_v<T, bool>) {
                // Bytes (0 or 1), sixteen at a time into byte counters, widened
                // every 255 steps (as count_nonzero; the bool loop was a third of
                // a sparse nonzero).
                const auto* q = reinterpret_cast<const std::uint8_t*>(p);
                while (i + 16 <= e) {
                    uint8x16_t c = vdupq_n_u8(0);
                    const std::int64_t stop = std::min(e - 15, i + 255 * 16);
                    for (; i < stop; i += 16) c = vaddq_u8(c, vld1q_u8(q + i));
                    k += vaddlvq_u8(c);
                }
            }
            for (; i < e; ++i) k += truthy(p[i]);
            start[t + 1] = k;
        });
        for (std::int64_t t = 0; t < chunks; ++t) start[t + 1] += start[t];
        const std::int64_t shape[] = {start[chunks]};
        for (int d = 0; d < nd; ++d) {
            out.push_back(new_array(DType::Int64, shape));
            dst.push_back(reinterpret_cast<std::int64_t*>(out.back().data));
        }
        for_pieces(size, kNonzeroChunk, size, [&](std::int64_t t, std::int64_t b, std::int64_t e) {
            // Branch-free: every position is written at k, and k moves on only
            // past a nonzero one (a random mask defeats branch prediction:
            // 25 ms -> 3.2 ms for 1e7 elements, one thread). `k < end` keeps the writes inside
            // this chunk's part; it is almost always true, so it predicts well.
            std::int64_t k = start[t];
            const std::int64_t end_k = start[t + 1];
            if (nd == 1) {
                std::int64_t* o = dst[0];
                std::int64_t i = b;
                // Blocks of 16 counted first (a vectorized count): a block with no
                // nonzero is skipped whole, a full one written as a run, so a sparse
                // mask costs a count per block instead of a store per element (10^6
                // with 1% nonzero: 276 us against numpy's 210).
                constexpr std::int64_t kRun = 16;
                for (; i + kRun <= e; i += kRun) {
                    std::int64_t c = 0;
                    if constexpr (std::is_same_v<T, bool>) {
                        const uint8x16_t v = vld1q_u8(reinterpret_cast<const std::uint8_t*>(p + i));
                        c = vmaxvq_u8(v) == 0 ? 0 : vminvq_u8(v) != 0 ? kRun : 1;  // 1: some, not all
                    } else {
                        for (std::int64_t j = 0; j < kRun; ++j) c += truthy(p[i + j]);
                    }
                    if (c == 0) continue;
                    if (c == kRun) {
                        for (std::int64_t j = 0; j < kRun; ++j) o[k + j] = i + j;
                        k += kRun;
                        continue;
                    }
                    for (std::int64_t j = 0; j < kRun; ++j) {
                        if (k < end_k) o[k] = i + j;
                        k += truthy(p[i + j]);
                    }
                }
                for (; i < e; ++i) {
                    if (k < end_k) o[k] = i;
                    k += truthy(p[i]);
                }
                return;
            }
            // Row by row of the last axis: the last index is written per
            // element (branch-free, as above); the leading indices are constant
            // along a row, so they are filled once per row over its output range.
            std::int64_t at[kMaxNdim];
            for (std::int64_t d = nd - 1, r = b; d >= 0; --d) {
                at[d] = r % a.shape[d];
                r /= a.shape[d];
            }
            const std::int64_t last = a.shape[nd - 1];
            std::int64_t* o = dst[nd - 1];
            for (std::int64_t i = b; i < e;) {
                const std::int64_t row_end = std::min(e, i + (last - at[nd - 1]));
                const std::int64_t k0 = k;
                for (std::int64_t j = at[nd - 1]; i < row_end; ++i, ++j) {
                    if (k < end_k) o[k] = j;
                    k += truthy(p[i]);
                }
                for (std::int64_t d = 0; d < nd - 1; ++d) std::fill(dst[d] + k0, dst[d] + k, at[d]);
                at[nd - 1] = 0;
                for (std::int64_t d = nd - 2; d >= 0; --d) {
                    if (++at[d] < a.shape[d]) break;
                    at[d] = 0;
                }
            }
        });
    });
    nb::list r;
    for (NDArray& o : out) r.append(wrap(std::move(o)));
    return nb::tuple(r);
}


// ---------------------------------------------------- searchsorted

// The sort order (docs/design/array.md §8.1): NaN is the largest value.
template <typename T>
bool before(T a, T b) {
    if constexpr (ops::is_float<T>) return a < b || (std::isnan(b) && !std::isnan(a));
    else return a < b;
}

nb::object searchsorted(nb::handle x1, nb::handle x2, nb::handle side_obj, nb::handle sorter_obj) {
    const NDArray& a = require_ndarray(x1, "x1");
    if (a.ndim != 1) throw nb::value_error("'x1' must be 1-D");
    const bool right = parse_choice(side_obj, "side", {"left", "right"}) == "right";

    // The values, in the dtype both sides promote to.
    std::optional<DType> v_dtype;
    if (is_ndarray(x2)) v_dtype = as_ndarray(x2).dtype;
    else v_dtype = scalar_dtype(x2);
    if (!v_dtype)
        throw nb::type_error(
            ("'x2' must be an sd.ndarray or a bool, int, float or complex, not " + py_type_name(x2)).c_str()
        );
    const DType d = promote(a.dtype, *v_dtype);
    if (d == DType::Complex128) throw nb::type_error("not supported for complex128");  // no order (§2.6)

    nb::object keys_obj = nb::borrow(x1);
    if (!sorter_obj.is_none()) {
        // An array of positions only: an int would not sort anything.
        check_indices(sorter_obj, "sorter", false);
        if (as_ndarray(sorter_obj).ndim != 1 || as_ndarray(sorter_obj).shape[0] != a.shape[0])
            throw nb::value_error("'sorter' must be a 1-D int64 array as long as 'x1'");
        // Positions in [0, n): a negative one is out of range too (it does not
        // count from the end, as an index would), the same IndexError as a
        // position past the end. The first bad one in order is named.
        {
            const NDArray& so = as_ndarray(sorter_obj);
            const std::int64_t n = a.shape[0];
            for (std::int64_t i = 0; i < n; ++i) {
                const std::int64_t k = load_as<std::int64_t>(so.data + i * so.strides[0]);
                if (k < 0 || k >= n)
                    throw nb::index_error(("index " + std::to_string(k) + " is out of bounds for axis 0 with size " +
                                           std::to_string(n))
                                              .c_str());
            }
        }
        keys_obj = gather_named(x1, sorter_obj);  // x1 in sorted order
    }
    // x1 is searched where it lies, in its own dtype and layout: each probe of
    // the binary search reads one key and converts it to the promoted dtype (the
    // conversion assign() would make). Copying x1 first cost O(n) on every call
    // with an int64 x1 and float values, or a strided x1; a search is O(log n).
    const NDArray& keys = as_ndarray(keys_obj);
    const std::int64_t n = keys.shape[0];
    const std::int64_t ks = keys.strides[0];

    const bool scalar = !is_ndarray(x2);
    std::int64_t one_result = 0;
    NDArray vals, out;
    if (!scalar) {
        const NDArray& v0 = as_ndarray(x2);
        vals = v0.dtype == d && v0.c_contiguous ? v0 : NDArray{};
        if (vals.ndim == 0) {  // the values in the promoted dtype, contiguous
            vals = new_array(d, v0.shape_span());
            if (vals.size() > 0) assign(vals, v0);
        }
        out = new_array(DType::Int64, vals.shape_span());
    }
    std::int64_t* o = scalar ? &one_result : reinterpret_cast<std::int64_t*>(out.data);
    const std::int64_t m = scalar ? 1 : vals.size();
    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (!ops::is_cplx<T>) {
            // The value of a scalar x2, converted straight to the promoted dtype
            // (docs/api/array.md §1.5): an int outside int64 is an OverflowError only
            // when that dtype is an integer one.
            T scalar_value{};
            if (scalar) scalar_value = scalar_from_python<T>(x2, *v_dtype);
            const T* v = scalar ? &scalar_value : reinterpret_cast<const T*>(vals.data);
            // left: the first i with not (k[i] < v); right: the first i with v < k[i].
            // So the answer is the number of keys k with below(k, v). Branch-free
            // halving (a select per step, never a misprediction), and kLanes values
            // searched side by side: every value of a group halves the same lengths,
            // and their probes are independent loads, so kLanes cache misses are in
            // flight at once instead of one (10^5 values in 10^6 keys: one at a time,
            // a chain of dependent misses per value).
            auto below = [right](T k, T vj) { return right ? !before<T>(vj, k) : before<T>(k, vj); };
            constexpr int kLanes = 8;
            auto search = [&](auto&& key) {
                return [&, key](std::int64_t begin, std::int64_t end) {
                    if (n == 0) {
                        std::fill(o + begin, o + end, 0);
                        return;
                    }
                    auto group = [&]<int L>(std::int64_t j) {
                        std::int64_t base[L] = {};
                        for (std::int64_t len = n; len > 1;) {
                            const std::int64_t half = len / 2;
                            for (int l = 0; l < L; ++l)
                                base[l] = below(key(base[l] + half), v[j + l]) ? base[l] + half : base[l];
                            len -= half;
                        }
                        for (int l = 0; l < L; ++l) o[j + l] = base[l] + below(key(base[l]), v[j + l]);
                    };
                    std::int64_t j = begin;
                    for (; j + kLanes <= end; j += kLanes) group.template operator()<kLanes>(j);
                    for (; j < end; ++j) group.template operator()<1>(j);
                };
            };
            auto launch = [&](auto&& run) {
                const std::int64_t work = m * std::max<std::int64_t>(1, static_cast<std::int64_t>(std::log2(n + 1)));
                if (work >= kParallelThreshold && parallel::available()) {
                    const std::int64_t tasks = (m + kParallelTask - 1) / kParallelTask;
                    parallel::parallel_for(tasks, [&](std::int64_t t) {
                        run(t * kParallelTask, std::min(m, (t + 1) * kParallelTask));
                    });
                } else {
                    const parallel::Unlocked unlocked(work >= kParallelThreshold);
                    run(0, m);
                }
            };
            if (keys.dtype == d && ks == static_cast<std::int64_t>(sizeof(T))) {
                const T* k = reinterpret_cast<const T*>(keys.data);
                launch(search([k](std::int64_t i) { return k[i]; }));
            } else {
                visit_dtype(keys.dtype, [&]<typename K>(std::type_identity<K>) {
                    if constexpr (!ops::is_cplx<K>) {
                        const std::byte* kp = keys.data;
                        launch(search([kp, ks](std::int64_t i) { return cast_value<T>(load_as<K>(kp + i * ks)); }));
                    }
                });
            }
        }
    });
    if (scalar) return to_python(one_result);
    return wrap(std::move(out));
}

}  // namespace

void bind_searching(nb::module_& m) {
    using nb::arg;
    m.def(
        "take", named("take", &take), nb::sig("def take(x, indices, /, *, axis=None)"), arg().none(), arg().none(),
        nb::kw_only(), arg("axis").none() = nb::none()
    );
    m.def(
        "take_along_axis", named("take_along_axis", &take_along_axis),
        nb::sig("def take_along_axis(x, indices, /, *, axis=-1)"), arg().none(), arg().none(), nb::kw_only(),
        arg("axis").none() = -1
    );
    m.def("nonzero", named("nonzero", &nonzero), nb::sig("def nonzero(x, /)"), arg().none());
    m.def(
        "count_nonzero", named("count_nonzero", &count_nonzero),
        nb::sig("def count_nonzero(x, /, *, axis=None, keepdims=False)"), arg().none(), nb::kw_only(),
        arg("axis").none() = nb::none(), arg("keepdims").none() = false
    );
    m.def(
        "searchsorted", named("searchsorted", &searchsorted),
        nb::sig("def searchsorted(x1, x2, /, *, side='left', sorter=None)"), arg().none(), arg().none(), nb::kw_only(),
        arg("side").none() = "left", arg("sorter").none() = nb::none()
    );
}

}  // namespace sd
