// take, take_along_axis, nonzero, count_nonzero, searchsorted
// (docs/api/array.md §5.4, §8.4; T0b).

#include "array/ops/searching.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/indexing.hpp"
#include "array/ops/reduction.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace sd {

namespace {

nb::object full_slice() { return nb::slice(nb::none(), nb::none(), nb::none()); }

// An index argument: a Python int or an int64 array. A bool array would be
// read as a mask by indexing, which is not what take means.
void check_indices(nb::handle o, const char* what) {
    std::int64_t v = 0;
    if (!is_ndarray(o) && index_value(o, v) == IndexResult::NotInt)
        throw nb::type_error(("'" + std::string(what) + "' must be an int or an int64 array, not " + py_type_name(o)).c_str());
    if (is_ndarray(o) && as_ndarray(o).dtype != DType::Int64)
        throw nb::type_error(("'" + std::string(what) + "' must be an int64 array, not " +
                              std::string(dtype_name(as_ndarray(o).dtype)))
                                 .c_str());
}

// ------------------------------------------------------------ take

nb::object take(nb::handle x, nb::handle indices, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    check_indices(indices, "indices");
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
    return getitem(x, nb::tuple(key));
}

nb::object take_along_axis(nb::handle x, nb::handle indices, nb::handle axis_obj) {
    const NDArray& a = require_ndarray(x);
    if (!is_ndarray(indices)) throw nb::type_error(("'indices' must be an int64 array, not " + py_type_name(indices)).c_str());
    check_indices(indices, "indices");
    const NDArray& ix = as_ndarray(indices);
    if (ix.ndim != a.ndim)
        throw nb::value_error(("'indices' must have as many dimensions as 'x' (" + std::to_string(a.ndim) + "), not " +
                               std::to_string(ix.ndim))
                                  .c_str());
    const int axis = normalize_axis(parse_int(axis_obj, "axis"), a.ndim);
    // numpy's _make_along_axis_idx: arange along every other axis, shaped to
    // broadcast against indices.
    nb::list key;
    for (int d = 0; d < a.ndim; ++d) {
        if (d == axis) {
            key.append(indices);
            continue;
        }
        std::vector<std::int64_t> shape(a.ndim, 1);
        shape[d] = a.shape[d];
        NDArray r = new_array(DType::Int64, shape);
        std::int64_t* p = reinterpret_cast<std::int64_t*>(r.data);
        for (std::int64_t i = 0; i < a.shape[d]; ++i) p[i] = i;
        key.append(wrap(std::move(r)));
    }
    return getitem(x, nb::tuple(key));
}

// --------------------------------------------------------- nonzero

// x != 0 as a C-ordered bool array (the bool conversion: NaN counts as true).
NDArray truth(const NDArray& a) {
    NDArray m = new_array(DType::Bool, a.shape_span());
    if (m.size() > 0) assign(m, a);
    return m;
}

nb::tuple nonzero(nb::handle x) {
    const NDArray& a = require_ndarray(x);
    const NDArray m = truth(a);
    const bool* p = reinterpret_cast<const bool*>(m.data);
    const std::int64_t size = m.size();
    const std::int64_t count = std::count(p, p + size, true);
    std::vector<NDArray> out;
    std::vector<std::int64_t*> dst;
    const std::int64_t shape[] = {count};
    for (int d = 0; d < a.ndim; ++d) {
        out.push_back(new_array(DType::Int64, shape));
        dst.push_back(reinterpret_cast<std::int64_t*>(out.back().data));
    }
    // Walk the C-order index like an odometer instead of dividing.
    std::vector<std::int64_t> at(a.ndim, 0);
    for (std::int64_t i = 0, k = 0; i < size; ++i) {
        if (p[i]) {
            for (int d = 0; d < a.ndim; ++d) dst[d][k] = at[d];
            ++k;
        }
        for (int d = a.ndim - 1; d >= 0; --d) {
            if (++at[d] < a.shape[d]) break;
            at[d] = 0;
        }
    }
    nb::list r;
    for (NDArray& o : out) r.append(wrap(std::move(o)));
    return nb::tuple(r);
}

nb::object count_nonzero(nb::handle x, nb::handle axis, nb::handle keepdims) {
    const NDArray& a = require_ndarray(x);
    parse_bool(keepdims, "keepdims");
    return sum(wrap(truth(a)), axis, nb::none(), keepdims);  // bool sums in int64
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
    if (!nb::isinstance<nb::str>(side_obj))
        throw nb::type_error(("'side' must be a str, not " + py_type_name(side_obj)).c_str());
    const std::string side = nb::borrow<nb::str>(side_obj).c_str();
    if (side != "left" && side != "right") throw nb::value_error(("'side' must be 'left' or 'right', not '" + side + "'").c_str());
    const bool right = side == "right";

    // The values, in the dtype both sides promote to.
    std::optional<DType> v_dtype;
    if (is_ndarray(x2)) v_dtype = as_ndarray(x2).dtype;
    else v_dtype = scalar_dtype(x2);
    if (!v_dtype) throw nb::type_error(("'x2' must be an sd.ndarray or a Python scalar, not " + py_type_name(x2)).c_str());
    const DType d = promote(a.dtype, *v_dtype);
    if (d == DType::Complex128) throw nb::type_error("not supported for complex128");  // no order (§2.6)

    nb::object keys_obj = nb::borrow(x1);
    if (!sorter_obj.is_none()) {
        check_indices(sorter_obj, "sorter");
        if (!is_ndarray(sorter_obj) || as_ndarray(sorter_obj).ndim != 1 || as_ndarray(sorter_obj).shape[0] != a.shape[0])
            throw nb::value_error("'sorter' must be a 1-D int64 array as long as 'x1'");
        keys_obj = getitem(x1, sorter_obj);  // x1 in sorted order
    }
    NDArray keys = new_array(d, as_ndarray(keys_obj).shape_span());
    if (keys.size() > 0) assign(keys, as_ndarray(keys_obj));

    const bool scalar = !is_ndarray(x2);
    NDArray vals;
    if (scalar) {
        const std::int64_t one[] = {1};
        vals = new_array(d, one);
        nb::list one_value;
        one_value.append(x2);
        assign(vals, as_ndarray(nb::module_::import_("stardust._core").attr("asarray")(one_value)));
    } else {
        vals = new_array(d, as_ndarray(x2).shape_span());
        if (vals.size() > 0) assign(vals, as_ndarray(x2));
    }
    NDArray out = new_array(DType::Int64, vals.shape_span());
    std::int64_t* o = reinterpret_cast<std::int64_t*>(out.data);
    const std::int64_t n = keys.size(), m = vals.size();
    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (!ops::is_cplx<T>) {
            const T* k = reinterpret_cast<const T*>(keys.data);
            const T* v = reinterpret_cast<const T*>(vals.data);
            // left: the first i with not (k[i] < v); right: the first i with v < k[i].
            auto run = [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t j = begin; j < end; ++j) {
                    const T* at = right ? std::upper_bound(k, k + n, v[j], before<T>)
                                        : std::lower_bound(k, k + n, v[j], before<T>);
                    o[j] = at - k;
                }
            };
            const std::int64_t work = m * std::max<std::int64_t>(1, static_cast<std::int64_t>(std::log2(n + 1)));
            if (work >= kParallelThreshold && parallel::available()) {
                const std::int64_t tasks = (m + kParallelTask - 1) / kParallelTask;
                parallel::parallel_for(tasks, [&](std::int64_t t) {
                    run(t * kParallelTask, std::min(m, (t + 1) * kParallelTask));
                });
            } else {
                run(0, m);
            }
        }
    });
    if (scalar) return to_python(o[0]);
    return wrap(std::move(out));
}

}  // namespace

void bind_searching(nb::module_& m) {
    using nb::arg;
    m.def("take", named("take", &take), arg().none(), arg().none(), nb::kw_only(), arg("axis").none() = nb::none());
    m.def("take_along_axis", named("take_along_axis", &take_along_axis), arg().none(), arg().none(), nb::kw_only(),
          arg("axis").none() = -1);
    m.def("nonzero", named("nonzero", &nonzero), arg().none());
    m.def("count_nonzero", named("count_nonzero", &count_nonzero), arg().none(), nb::kw_only(),
          arg("axis").none() = nb::none(), arg("keepdims").none() = false);
    m.def("searchsorted", named("searchsorted", &searchsorted), arg().none(), arg().none(), nb::kw_only(),
          arg("side").none() = "left", arg("sorter").none() = nb::none());
}

}  // namespace sd
