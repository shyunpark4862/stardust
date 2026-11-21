// The order reductions of sd.sparse: max, min, nanmax, nanmin and argmax, argmin along the lines of
// canonical compressed arrays or over the whole array (docs/design/sparse.md §2.4, §3.3).
//
//   _line_minmax(indptr, data, line_len, /, *, op, explicit)
//       Per line, the fold of the stored values with the array's op (maximum, minimum: NaN
//       propagates; nanmax, nanmin: NaN skipped), then -- unless explicit, when the line stores
//       fewer than line_len values -- with the implicit zero. An empty line gives 0. bool, int64 or
//       float64 data; the result has its dtype. One line over all the values (indptr [0, nnz],
//       line_len the size) is the reduction of the whole array.
//   _line_arg(indptr, indices, data, line_len, /, *, op, explicit)
//       Per line, the position of the largest (argmax) or smallest (argmin) value, the implicit
//       zeros included unless explicit: the first NaN if there is one (the array's rule), else the
//       first position holding the extreme value. An empty line gives 0. int64 result.
//   _flat_arg(indptr, indices, data, n_cols, /, *, op, explicit)
//       The same over the whole array in row-major order (CSR lines), as a flat C-order position.
//
// The folds visit each line in index order; lines are cut into fixed ranges by size alone and every
// output element is written by one task, so the results do not depend on the thread count.
#include "array/kernels/scalar_ops.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;

int64_t per_task(int64_t nnz, int64_t n_lines) {
    return std::max<int64_t>(1, kChunk / std::max<int64_t>(1, nnz / std::max<int64_t>(n_lines, 1)));
}

template <class Op, class T>
nb::object minmax_typed(const NDArray& indptr, const NDArray& data, int64_t line_len, bool explicit_) {
    const int64_t n_lines = indptr.shape[0] - 1, nnz = data.shape[0];
    const int64_t shape[] = {n_lines};
    NDArray out = new_array(dtype_of<T>, shape);
    T* o = typed<T>(out);
    const int64_t* ip = i64(indptr);
    const T* v = typed<T>(data);
    for_chunks(
        n_lines,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r) {
                const int64_t b = ip[r], e = ip[r + 1];
                if (b == e) {
                    o[r] = T{};
                    continue;
                }
                T m = v[b];
                for (int64_t p = b + 1; p < e; ++p) m = Op::f(m, v[p]);
                if (!explicit_ && e - b < line_len) m = Op::f(m, T{});
                o[r] = m;
            }
        },
        per_task(nnz, n_lines), nnz + n_lines
    );
    return wrap(std::move(out));
}

template <class Op>
nb::object minmax_dispatch(const NDArray& indptr, const NDArray& data, int64_t line_len, bool explicit_) {
    return visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) -> nb::object {
        if constexpr (Op::template ok<T>) return minmax_typed<Op, T>(indptr, data, line_len, explicit_);
        else throw nb::type_error("_line_minmax(): no loop for this dtype");
    });
}

nb::object line_minmax(
    nb::handle indptr_h, nb::handle data_h, int64_t line_len, const std::string& op, bool explicit_
) {
    const NDArray& indptr = vector_arg(indptr_h, "_line_minmax", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_line_minmax");
    using namespace ops;
    const bool real = data.dtype == DType::Float64;
    if (op == "max") return minmax_dispatch<Maximum>(indptr, data, line_len, explicit_);
    if (op == "min") return minmax_dispatch<Minimum>(indptr, data, line_len, explicit_);
    if (op == "nanmax")
        return real ? minmax_dispatch<NanMaximum>(indptr, data, line_len, explicit_)
                    : minmax_dispatch<Maximum>(indptr, data, line_len, explicit_);
    if (op == "nanmin")
        return real ? minmax_dispatch<NanMinimum>(indptr, data, line_len, explicit_)
                    : minmax_dispatch<Minimum>(indptr, data, line_len, explicit_);
    throw nb::value_error(("_line_minmax(): unknown op " + op).c_str());
}

// The best stored entry of [b, e) for argmax (Max) or argmin: the first NaN, else the first
// position of the extreme value. Returns its entry index.
template <bool Max, class T>
inline int64_t best_entry(const T* v, int64_t b, int64_t e) {
    int64_t best = b;
    if constexpr (std::is_same_v<T, double>) {
        if (v[b] != v[b]) return b;
    }
    for (int64_t p = b + 1; p < e; ++p) {
        if constexpr (std::is_same_v<T, double>) {
            if (v[p] != v[p]) return p;
        }
        if (Max ? v[p] > v[best] : v[p] < v[best]) best = p;
    }
    return best;
}

// The position the arg reduction picks between the best stored value (at position pos) and the
// first implicit zero (at position zero_pos, -1 when there is none).
template <bool Max, class T>
inline int64_t choose(T best, int64_t pos, int64_t zero_pos) {
    if (zero_pos < 0) return pos;
    if constexpr (std::is_same_v<T, double>) {
        if (best != best) return pos;
    }
    const T zero{};
    if (Max ? best > zero : best < zero) return pos;
    if (best == zero) return std::min(pos, zero_pos);
    return zero_pos;
}

// The first index in [0, len) missing from the sorted ix[b, e), or -1.
inline int64_t first_missing(const int64_t* ix, int64_t b, int64_t e, int64_t len) {
    if (e - b >= len) return -1;
    for (int64_t p = b; p < e; ++p)
        if (ix[p] != p - b) return p - b;
    return e - b;
}

template <bool Max, class T>
nb::object arg_typed(
    const NDArray& indptr, const NDArray& indices, const NDArray& data, int64_t line_len, bool explicit_
) {
    const int64_t n_lines = indptr.shape[0] - 1, nnz = data.shape[0];
    NDArray out = new_index(n_lines);
    int64_t* o = i64(out);
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const T* v = typed<T>(data);
    for_chunks(
        n_lines,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r) {
                const int64_t b = ip[r], e = ip[r + 1];
                if (b == e) {
                    o[r] = 0;
                    continue;
                }
                const int64_t best = best_entry<Max>(v, b, e);
                o[r] = explicit_ ? ix[best] : choose<Max>(v[best], ix[best], first_missing(ix, b, e, line_len));
            }
        },
        per_task(nnz, n_lines), nnz + n_lines
    );
    return wrap(std::move(out));
}

template <bool Max, class T>
int64_t flat_arg_typed(
    const NDArray& indptr, const NDArray& indices, const NDArray& data, int64_t n_cols, bool explicit_
) {
    const int64_t n_lines = indptr.shape[0] - 1, nnz = data.shape[0];
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const T* v = typed<T>(data);
    const parallel::Unlocked unlocked(nnz + n_lines >= kParallelThreshold);
    const int64_t best = best_entry<Max>(v, 0, nnz);
    int64_t row = 0;
    while (ip[row + 1] <= best) ++row;
    const int64_t pos = row * n_cols + ix[best];
    if (explicit_) return pos;
    // the first implicit zero in C order: the first line with a gap, at its first gap
    int64_t zero_pos = -1;
    for (int64_t r = 0; r < n_lines; ++r) {
        const int64_t gap = first_missing(ix, ip[r], ip[r + 1], n_cols);
        if (gap >= 0) {
            zero_pos = r * n_cols + gap;
            break;
        }
    }
    return choose<Max>(v[best], pos, zero_pos);
}

template <bool Max>
nb::object arg_dispatch(
    const NDArray& indptr, const NDArray& indices, const NDArray& data, int64_t len, bool explicit_, bool flat
) {
    return visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) -> nb::object {
        if constexpr (std::is_same_v<T, complex128>) throw nb::type_error("_line_arg(): no complex loop");
        else if (flat) return nb::int_(flat_arg_typed<Max, T>(indptr, indices, data, len, explicit_));
        else return arg_typed<Max, T>(indptr, indices, data, len, explicit_);
    });
}

nb::object line_arg(
    nb::handle indptr_h,
    nb::handle indices_h,
    nb::handle data_h,
    int64_t len,
    const std::string& op,
    bool explicit_,
    bool flat
) {
    const NDArray& indptr = vector_arg(indptr_h, "_line_arg", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_line_arg", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_line_arg");
    if (flat && data.shape[0] == 0) throw nb::value_error("_flat_arg(): no stored values");
    if (op == "argmax") return arg_dispatch<true>(indptr, indices, data, len, explicit_, flat);
    if (op == "argmin") return arg_dispatch<false>(indptr, indices, data, len, explicit_, flat);
    throw nb::value_error(("_line_arg(): unknown op " + op).c_str());
}

}  // namespace

void bind_reduce(nb::module_& m) {
    m.def(
        "_line_minmax", &line_minmax, nb::arg("indptr"), nb::arg("data"), nb::arg("line_len"), nb::kw_only(),
        nb::arg("op"), nb::arg("explicit")
    );
    m.def(
        "_line_arg",
        [](nb::handle ip, nb::handle ix, nb::handle d, int64_t len, const std::string& op, bool ex) {
            return line_arg(ip, ix, d, len, op, ex, false);
        },
        nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("line_len"), nb::kw_only(), nb::arg("op"),
        nb::arg("explicit")
    );
    m.def(
        "_flat_arg",
        [](nb::handle ip, nb::handle ix, nb::handle d, int64_t n_cols, const std::string& op, bool ex) {
            return line_arg(ip, ix, d, n_cols, op, ex, true);
        },
        nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("n_cols"), nb::kw_only(), nb::arg("op"),
        nb::arg("explicit")
    );
}

}  // namespace sd::sparse
