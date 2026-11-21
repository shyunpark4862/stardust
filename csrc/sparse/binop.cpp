// Element by element between two sparse arrays, and the dense values at stored positions
// (docs/design/sparse.md §3.2).
//
//   _binop(op, a_indptr, a_indices, a_data, b_indptr, b_indices, b_data, /, *, keep_all)
//       Two canonical compressed arrays of the same lines and dtype (the loop dtype the caller
//       chose by the array's rules; op must have a loop for it). Per line, the two sorted index
//       lists are merged: where both store a value the result is op(a, b), where one stores it
//       op(a, 0) or op(0, b) -- the implicit zero of the other side -- except for multiply, which
//       keeps only the positions both store (an implicit zero is a strong zero: 0 * nan is 0,
//       docs/design/sparse.md §2.4). The op is the array's own (array/kernels/scalar_ops.hpp), so
//       a value is bit for bit what the dense operation gives there. Exact zeros of the result
//       are dropped (scipy's csr_binop_csr) unless keep_all. Comparisons give bool.
//   _fill_true(indptr, indices, flags, n_minor, /)
//       The positions of each line where (stored ? flag : True): what a comparison true between
//       two implicit zeros (==, <=, >=) stores. bool data.
//   _gather_compressed(indptr, indices, x, /, *, by_rows)
//   _gather_coo(major, minor, x, /)
//       The values of the dense x (2-D, or 1-D for a 1-D sparse array; any strides, a broadcast
//       view too) at the stored positions, in stored order: a new contiguous 1-D array.
//
// Lines are independent and cut into fixed ranges by size alone; every output entry is written by
// one task, so the results do not depend on the thread count.
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/arith.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;

nb::object readonly(NDArray&& a) {
    a.writeable = false;
    return wrap(std::move(a));
}

// a cut to its first n elements: a view when that keeps at least half of it, else a copy (the
// memory of the rest goes).
NDArray fitted(NDArray&& a, int64_t n) {
    if (n == a.shape[0]) return std::move(a);
    if (2 * n >= a.shape[0]) {
        a.shape[0] = n;
        return std::move(a);
    }
    const int64_t shape[] = {n};
    NDArray c = new_array(a.dtype, shape);
    std::memcpy(c.data, a.data, static_cast<std::size_t>(n * a.itemsize()));
    return c;
}

// The merged line: emit(position, value) for every result entry of line [ab, ae) x [bb, be).
template <class Op, bool Intersect, class T, class Emit>
inline void merge_line(
    const int64_t* ai,
    const T* av,
    int64_t ab,
    int64_t ae,
    const int64_t* bi,
    const T* bv,
    int64_t bb,
    int64_t be,
    Emit&& emit
) {
    const T zero{};
    int64_t i = ab, j = bb;
    while (i < ae || j < be) {
        if (j >= be || (i < ae && ai[i] < bi[j])) {
            if constexpr (!Intersect) emit(ai[i], Op::f(av[i], zero));
            ++i;
        } else if (i >= ae || bi[j] < ai[i]) {
            if constexpr (!Intersect) emit(bi[j], Op::f(zero, bv[j]));
            ++j;
        } else {
            emit(ai[i], Op::f(av[i], bv[j]));
            ++i;
            ++j;
        }
    }
}

template <class Op, bool Intersect, class T>
nb::tuple binop_typed(
    const NDArray& a_ptr,
    const NDArray& a_ix,
    const NDArray& a_v,
    const NDArray& b_ptr,
    const NDArray& b_ix,
    const NDArray& b_v,
    bool keep_all
) {
    using R = decltype(Op::f(T{}, T{}));
    const int64_t n_lines = a_ptr.shape[0] - 1;
    const int64_t* ap = i64(a_ptr);
    const int64_t* ai = i64(a_ix);
    const int64_t* bp = i64(b_ptr);
    const int64_t* bi = i64(b_ix);
    const T* av = typed<T>(a_v);
    const T* bv = typed<T>(b_v);
    const int64_t nnz = a_ix.shape[0] + b_ix.shape[0];
    const int64_t per = std::max<int64_t>(1, kChunk / std::max<int64_t>(1, nnz / std::max<int64_t>(n_lines, 1)));
    NDArray indptr = new_indptr(n_lines);
    int64_t* op = i64(indptr);
    op[0] = 0;
    if (nnz + n_lines < kParallelThreshold || parallel::num_threads() <= 1) {
        // one pass on the calling thread into arrays of the largest possible size (scipy's way),
        // cut to the entries made
        const int64_t cap = Intersect ? std::min(a_ix.shape[0], b_ix.shape[0]) : nnz;
        NDArray indices = new_index(cap);
        const int64_t cap_shape[] = {cap};
        NDArray data = new_array(dtype_of<R>, cap_shape);
        int64_t* ox = i64(indices);
        R* ov = typed<R>(data);
        int64_t w = 0;
        {
            const parallel::Unlocked unlocked(nnz + n_lines >= kParallelThreshold);
            for (int64_t r = 0; r < n_lines; ++r) {
                merge_line<Op, Intersect>(ai, av, ap[r], ap[r + 1], bi, bv, bp[r], bp[r + 1], [&](int64_t pos, R v) {
                    if (keep_all || is_nonzero(v)) {
                        ox[w] = pos;
                        ov[w] = v;
                        ++w;
                    }
                });
                op[r + 1] = w;
            }
        }
        return nb::make_tuple(
            readonly(std::move(indptr)), readonly(fitted(std::move(indices), w)), wrap(fitted(std::move(data), w))
        );
    }
    for_chunks(
        n_lines,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r) {
                int64_t c = 0;
                merge_line<Op, Intersect>(ai, av, ap[r], ap[r + 1], bi, bv, bp[r], bp[r + 1], [&](int64_t, R v) {
                    c += keep_all || is_nonzero(v);
                });
                op[r + 1] = c;
            }
        },
        per, nnz + n_lines
    );
    for (int64_t r = 0; r < n_lines; ++r) op[r + 1] += op[r];
    const int64_t total = op[n_lines];
    NDArray indices = new_index(total);
    const int64_t shape[] = {total};
    NDArray data = new_array(dtype_of<R>, shape);
    int64_t* ox = i64(indices);
    R* ov = typed<R>(data);
    for_chunks(
        n_lines,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r) {
                int64_t w = op[r];
                merge_line<Op, Intersect>(ai, av, ap[r], ap[r + 1], bi, bv, bp[r], bp[r + 1], [&](int64_t pos, R v) {
                    if (keep_all || is_nonzero(v)) {
                        ox[w] = pos;
                        ov[w] = v;
                        ++w;
                    }
                });
            }
        },
        per, nnz + n_lines
    );
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(data)));
}

template <class Op, bool Intersect = false>
nb::tuple binop_dispatch(
    const NDArray& a_ptr,
    const NDArray& a_ix,
    const NDArray& a_v,
    const NDArray& b_ptr,
    const NDArray& b_ix,
    const NDArray& b_v,
    bool keep_all
) {
    return visit_dtype(a_v.dtype, [&]<class T>(std::type_identity<T>) -> nb::tuple {
        if constexpr (Op::template ok<T>)
            return binop_typed<Op, Intersect, T>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
        else throw nb::type_error("_binop(): no loop for this dtype");
    });
}

nb::tuple binop(
    const std::string& op,
    nb::handle a_ptr_h,
    nb::handle a_ix_h,
    nb::handle a_v_h,
    nb::handle b_ptr_h,
    nb::handle b_ix_h,
    nb::handle b_v_h,
    bool keep_all
) {
    const NDArray& a_ptr = vector_arg(a_ptr_h, "_binop", DType::Int64);
    const NDArray& a_ix = vector_arg(a_ix_h, "_binop", DType::Int64);
    const NDArray& a_v = vector_arg(a_v_h, "_binop");
    const NDArray& b_ptr = vector_arg(b_ptr_h, "_binop", DType::Int64);
    const NDArray& b_ix = vector_arg(b_ix_h, "_binop", DType::Int64);
    const NDArray& b_v = vector_arg(b_v_h, "_binop", a_v.dtype);
    if (a_ptr.shape[0] != b_ptr.shape[0] || a_ix.shape[0] != a_v.shape[0] || b_ix.shape[0] != b_v.shape[0])
        throw nb::value_error("_binop(): lengths differ");
    using namespace ops;
    if (op == "add") return binop_dispatch<Add>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "subtract") return binop_dispatch<Subtract>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "multiply") return binop_dispatch<Multiply, true>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "maximum") return binop_dispatch<Maximum>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "minimum") return binop_dispatch<Minimum>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "equal") return binop_dispatch<Equal>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "not_equal") return binop_dispatch<NotEqual>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "less") return binop_dispatch<Less>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "less_equal") return binop_dispatch<LessEqual>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "greater") return binop_dispatch<Greater>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    if (op == "greater_equal") return binop_dispatch<GreaterEqual>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, keep_all);
    throw nb::value_error(("_binop(): unknown op " + op).c_str());
}

nb::tuple fill_true(nb::handle ptr_h, nb::handle ix_h, nb::handle flags_h, int64_t n_minor) {
    const NDArray& ptr = vector_arg(ptr_h, "_fill_true", DType::Int64);
    const NDArray& ix = vector_arg(ix_h, "_fill_true", DType::Int64);
    const NDArray& flags = vector_arg(flags_h, "_fill_true", DType::Bool);
    const int64_t n_lines = ptr.shape[0] - 1;
    const int64_t* ip = i64(ptr);
    const int64_t* ii = i64(ix);
    const bool* fv = typed<bool>(flags);
    NDArray indptr = new_indptr(n_lines);
    int64_t* op = i64(indptr);
    op[0] = 0;
    const int64_t per = std::max<int64_t>(1, kChunk / std::max<int64_t>(1, n_minor));
    for_chunks(
        n_lines,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r) {
                int64_t c = n_minor - (ip[r + 1] - ip[r]);
                for (int64_t p = ip[r]; p < ip[r + 1]; ++p) c += fv[p];
                op[r + 1] = c;
            }
        },
        per, n_lines * std::max<int64_t>(n_minor, 1)
    );
    for (int64_t r = 0; r < n_lines; ++r) op[r + 1] += op[r];
    const int64_t total = op[n_lines];
    NDArray indices = new_index(total);
    const int64_t shape[] = {total};
    NDArray data = new_array(DType::Bool, shape);
    int64_t* ox = i64(indices);
    bool* ov = typed<bool>(data);
    for_chunks(
        n_lines,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r) {
                int64_t w = op[r], p = ip[r];
                for (int64_t j = 0; j < n_minor; ++j) {
                    const bool stored = p < ip[r + 1] && ii[p] == j;
                    if (!stored || fv[p]) {
                        ox[w] = j;
                        ov[w] = true;
                        ++w;
                    }
                    p += stored;
                }
            }
        },
        per, n_lines * std::max<int64_t>(n_minor, 1)
    );
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(data)));
}

// x[r, c] (or x[c] for 1-D x) as a byte pointer.
struct DenseAt {
    const std::byte* base;
    int64_t rs, cs;
    const std::byte* at(int64_t r, int64_t c) const { return base + r * rs + c * cs; }
};

DenseAt dense_at(const NDArray& x) {
    if (x.ndim == 2) return {x.data, x.strides[0], x.strides[1]};
    if (x.ndim == 1) return {x.data, 0, x.strides[0]};
    throw nb::type_error("gather: 1-D or 2-D only");
}

nb::object gather_compressed(nb::handle indptr_h, nb::handle indices_h, nb::handle x_h, bool by_rows) {
    const NDArray& indptr = vector_arg(indptr_h, "_gather_compressed", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_gather_compressed", DType::Int64);
    const NDArray& x = require_ndarray(x_h, "x");
    const DenseAt d = dense_at(x);
    const int64_t n_lines = indptr.shape[0] - 1, nnz = indices.shape[0];
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const int64_t shape[] = {nnz};
    NDArray out = new_array(x.dtype, shape);
    const int64_t item = x.itemsize();
    std::byte* o = out.data;
    const int64_t per = std::max<int64_t>(1, kChunk / std::max<int64_t>(1, nnz / std::max<int64_t>(n_lines, 1)));
    for_chunks(
        n_lines,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r)
                for (int64_t p = ip[r]; p < ip[r + 1]; ++p) {
                    const std::byte* src = by_rows ? d.at(r, ix[p]) : d.at(ix[p], r);
                    std::memcpy(o + p * item, src, static_cast<std::size_t>(item));
                }
        },
        per, nnz + n_lines
    );
    return wrap(std::move(out));
}

nb::object gather_coo(nb::handle major_h, nb::handle minor_h, nb::handle x_h) {
    const NDArray& minor = vector_arg(minor_h, "_gather_coo", DType::Int64);
    const int64_t nnz = minor.shape[0];
    const int64_t* ma = nullptr;
    if (!major_h.is_none()) {
        const NDArray& major = vector_arg(major_h, "_gather_coo", DType::Int64);
        if (major.shape[0] != nnz) throw nb::value_error("_gather_coo(): lengths differ");
        ma = i64(major);
    }
    const int64_t* mi = i64(minor);
    const NDArray& x = require_ndarray(x_h, "x");
    const DenseAt d = dense_at(x);
    const int64_t shape[] = {nnz};
    NDArray out = new_array(x.dtype, shape);
    const int64_t item = x.itemsize();
    std::byte* o = out.data;
    for_chunks(nnz, [&](int64_t, int64_t lo, int64_t hi) {
        for (int64_t p = lo; p < hi; ++p)
            std::memcpy(o + p * item, d.at(ma ? ma[p] : 0, mi[p]), static_cast<std::size_t>(item));
    });
    return wrap(std::move(out));
}

}  // namespace

void bind_binop(nb::module_& m) {
    m.def(
        "_binop", &binop, nb::arg("op"), nb::arg("a_indptr"), nb::arg("a_indices"), nb::arg("a_data"),
        nb::arg("b_indptr"), nb::arg("b_indices"), nb::arg("b_data"), nb::kw_only(), nb::arg("keep_all") = false
    );
    m.def("_fill_true", &fill_true);
    m.def(
        "_gather_compressed", &gather_compressed, nb::arg("indptr"), nb::arg("indices"), nb::arg("x"), nb::kw_only(),
        nb::arg("by_rows")
    );
    m.def("_gather_coo", &gather_coo, nb::arg("major").none(), nb::arg("minor"), nb::arg("x"));
}

}  // namespace sd::sparse
