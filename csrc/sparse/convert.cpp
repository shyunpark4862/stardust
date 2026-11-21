// Conversions of sd.sparse between the formats and dense arrays, and the pattern operations of
// milestone SP1 (docs/design/sparse.md §3.1, §3.5).
//
//   _expand(indptr, n_major, nnz, /)              the major index of every entry (read-only)
//   _from_dense(x, /, *, by_rows)                 (indptr, indices, data) of the nonzero values
//                                                 of a 1-D or 2-D array (any strides): x != 0, so
//                                                 NaN is stored and -0.0 is not
//   _to_dense(shape, indptr, indices, data, /, *, by_rows)
//   _coo_to_dense(shape, major, minor, data, /, *, distinct)
//                                                 C-ordered dense arrays: zeros plus the stored
//                                                 values, duplicates added in stored order
//   _filter_compressed(n_major, indptr, indices, data, new_major, new_minor, /, *, drop_zeros)
//   _filter_coo(major, minor, data, bound_major, bound_minor, /, *, drop_zeros)
//                                                 the entries inside the bounds (and nonzero with
//                                                 drop_zeros), order kept; None when every entry
//                                                 stays and (compressed) the line count is the same
//   _count_nonzero(data, /)                       how many values are nonzero
//   _bincount_nonzero(idx, data, length, /)       per index value, how many nonzero values
//   _rowcount_nonzero(indptr, data, /)            per line, how many nonzero values
//   _reshape_coords(major, minor, old_shape, new_shape, /)
//                                                 the coordinates of the same C-order positions in
//                                                 the new shape (read-only; major None for 1-D)
//
// The work is cut by sizes alone; every output element is written by one task, or counts are
// combined in task order, so the results do not depend on the thread count.
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;

// Lines per task so that one task covers about kChunk values.
int64_t lines_per_task(int64_t line_len) { return std::max<int64_t>(1, kChunk / std::max<int64_t>(line_len, 1)); }

nb::object readonly(NDArray&& a) {
    a.writeable = false;
    return wrap(std::move(a));
}

nb::tuple from_dense(nb::handle xh, bool by_rows) {
    const NDArray& x = require_ndarray(xh, "x");
    if (x.ndim != 1 && x.ndim != 2) throw nb::value_error("_from_dense(): 1-D or 2-D only");
    const bool two = x.ndim == 2;
    const int64_t rows = two ? x.shape[0] : 1, cols = two ? x.shape[1] : x.shape[0];
    const int64_t rs = two ? x.strides[0] : 0, cs = two ? x.strides[1] : x.strides[0];
    const int64_t n_major = by_rows ? rows : cols, n_minor = by_rows ? cols : rows;
    const int64_t ms = by_rows ? rs : cs, ns = by_rows ? cs : rs;
    const int64_t per = lines_per_task(n_minor), work = n_major * n_minor;
    NDArray indptr = new_index(n_major + 1);
    int64_t* ip = i64(indptr);
    ip[0] = 0;
    visit_dtype(x.dtype, [&]<class T>(std::type_identity<T>) {
        for_chunks(
            n_major,
            [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t r = lo; r < hi; ++r) {
                    const std::byte* p = x.data + r * ms;
                    int64_t c = 0;
                    for (int64_t j = 0; j < n_minor; ++j) {
                        T v;
                        std::memcpy(&v, p + j * ns, sizeof(T));
                        c += is_nonzero(v);
                    }
                    ip[r + 1] = c;
                }
            },
            per, work
        );
    });
    for (int64_t r = 0; r < n_major; ++r) ip[r + 1] += ip[r];
    const int64_t nnz = ip[n_major];
    NDArray indices = new_index(nnz);
    const int64_t shape[] = {nnz};
    NDArray data = new_array(x.dtype, shape);
    int64_t* ix = i64(indices);
    visit_dtype(x.dtype, [&]<class T>(std::type_identity<T>) {
        T* out = typed<T>(data);
        for_chunks(
            n_major,
            [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t r = lo; r < hi; ++r) {
                    const std::byte* p = x.data + r * ms;
                    int64_t w = ip[r];
                    for (int64_t j = 0; j < n_minor; ++j) {
                        T v;
                        std::memcpy(&v, p + j * ns, sizeof(T));
                        if (is_nonzero(v)) {
                            ix[w] = j;
                            out[w] = v;
                            ++w;
                        }
                    }
                }
            },
            per, work
        );
    });
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(data)));
}

std::vector<int64_t> shape_of(nb::handle shape) {
    std::vector<int64_t> s;
    for (nb::handle v : shape) s.push_back(nb::cast<int64_t>(v));
    if (s.empty() || s.size() > 2) throw nb::value_error("shape: 1-D or 2-D only");
    return s;
}

nb::object to_dense(nb::handle shape_h, nb::handle indptr_h, nb::handle indices_h, nb::handle data_h, bool by_rows) {
    const std::vector<int64_t> shape = shape_of(shape_h);
    const NDArray& indptr = vector_arg(indptr_h, "_to_dense", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_to_dense", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_to_dense");
    const int64_t n_major = indptr.shape[0] - 1, cols = shape.size() == 2 ? shape[1] : shape[0];
    NDArray out = new_array(data.dtype, shape, true);
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const int64_t nnz = indices.shape[0];
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        T* o = typed<T>(out);
        for_chunks(
            n_major,
            [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t r = lo; r < hi; ++r)
                    for (int64_t k = ip[r]; k < ip[r + 1]; ++k) {
                        const int64_t pos = by_rows ? r * cols + ix[k] : ix[k] * cols + r;
                        o[pos] = dense_value(v[k]);
                    }
            },
            lines_per_task(n_major > 0 ? nnz / n_major : 0), nnz + n_major
        );
    });
    return wrap(std::move(out));
}

const int64_t* opt_index(nb::handle o, const char* func, int64_t n) {
    if (o.is_none()) return nullptr;
    const NDArray& a = vector_arg(o, func, DType::Int64);
    if (a.shape[0] != n) throw nb::value_error((std::string(func) + "(): lengths differ").c_str());
    return i64(a);
}

nb::object coo_to_dense(nb::handle shape_h, nb::handle major_h, nb::handle minor_h, nb::handle data_h, bool distinct) {
    const std::vector<int64_t> shape = shape_of(shape_h);
    const NDArray& minor = vector_arg(minor_h, "_coo_to_dense", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_coo_to_dense");
    const int64_t nnz = minor.shape[0];
    const int64_t* ma = opt_index(major_h, "_coo_to_dense", nnz);
    const int64_t* mi = i64(minor);
    const int64_t cols = shape.size() == 2 ? shape[1] : shape[0];
    NDArray out = new_array(data.dtype, shape, true);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        T* o = typed<T>(out);
        if (distinct) {
            for_chunks(nnz, [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t k = lo; k < hi; ++k) o[(ma ? ma[k] * cols : 0) + mi[k]] = dense_value(v[k]);
            });
        } else {
            // duplicates add up in stored order: one pass on the calling thread
            const parallel::Unlocked unlocked(nnz >= kParallelThreshold);
            for (int64_t k = 0; k < nnz; ++k) {
                T& d = o[(ma ? ma[k] * cols : 0) + mi[k]];
                d = add_value(d, v[k]);
            }
        }
    });
    return wrap(std::move(out));
}

nb::object filter_compressed(
    int64_t n_major,
    nb::handle indptr_h,
    nb::handle indices_h,
    nb::handle data_h,
    int64_t new_major,
    int64_t new_minor,
    bool drop_zeros
) {
    const NDArray& indptr = vector_arg(indptr_h, "_filter_compressed", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_filter_compressed", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_filter_compressed");
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const int64_t keep_lines = std::min(n_major, new_major), nnz = indices.shape[0];
    NDArray out_ptr = new_index(new_major + 1);
    int64_t* op = i64(out_ptr);
    op[0] = 0;
    const int64_t per = lines_per_task(n_major > 0 ? nnz / n_major : 0);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        for_chunks(
            keep_lines,
            [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t r = lo; r < hi; ++r) {
                    int64_t c = 0;
                    for (int64_t k = ip[r]; k < ip[r + 1]; ++k)
                        c += ix[k] < new_minor && (!drop_zeros || is_nonzero(v[k]));
                    op[r + 1] = c;
                }
            },
            per, nnz + new_major
        );
    });
    for (int64_t r = keep_lines; r < new_major; ++r) op[r + 1] = 0;
    for (int64_t r = 0; r < new_major; ++r) op[r + 1] += op[r];
    const int64_t total = op[new_major];
    if (new_major == n_major && total == nnz) return nb::none();
    NDArray out_ix = new_index(total);
    const int64_t shape[] = {total};
    NDArray out_v = new_array(data.dtype, shape);
    int64_t* ox = i64(out_ix);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        T* ov = typed<T>(out_v);
        for_chunks(
            keep_lines,
            [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t r = lo; r < hi; ++r) {
                    int64_t w = op[r];
                    for (int64_t k = ip[r]; k < ip[r + 1]; ++k)
                        if (ix[k] < new_minor && (!drop_zeros || is_nonzero(v[k]))) {
                            ox[w] = ix[k];
                            ov[w] = v[k];
                            ++w;
                        }
                }
            },
            per, nnz + new_major
        );
    });
    return nb::make_tuple(readonly(std::move(out_ptr)), readonly(std::move(out_ix)), wrap(std::move(out_v)));
}

nb::object filter_coo(
    nb::handle major_h, nb::handle minor_h, nb::handle data_h, int64_t bound_major, int64_t bound_minor, bool drop_zeros
) {
    const NDArray& minor = vector_arg(minor_h, "_filter_coo", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_filter_coo");
    const int64_t nnz = minor.shape[0];
    const int64_t* ma = opt_index(major_h, "_filter_coo", nnz);
    const int64_t* mi = i64(minor);
    const int64_t nt = num_chunks(nnz);
    std::vector<int64_t> count(static_cast<std::size_t>(nt + 1), 0);
    auto keep = [&](const auto* v, int64_t k) {
        return (!ma || ma[k] < bound_major) && mi[k] < bound_minor && (!drop_zeros || is_nonzero(v[k]));
    };
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        for_chunks(nnz, [&](int64_t t, int64_t lo, int64_t hi) {
            int64_t c = 0;
            for (int64_t k = lo; k < hi; ++k) c += keep(v, k);
            count[t + 1] = c;
        });
    });
    for (int64_t t = 0; t < nt; ++t) count[t + 1] += count[t];
    const int64_t total = count[nt];
    if (total == nnz) return nb::none();
    NDArray out_mi = new_index(total);
    NDArray out_ma = ma ? new_index(total) : NDArray();
    const int64_t shape[] = {total};
    NDArray out_v = new_array(data.dtype, shape);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        T* ov = typed<T>(out_v);
        int64_t* omi = i64(out_mi);
        int64_t* oma = ma ? i64(out_ma) : nullptr;
        for_chunks(nnz, [&](int64_t t, int64_t lo, int64_t hi) {
            int64_t w = count[t];
            for (int64_t k = lo; k < hi; ++k)
                if (keep(v, k)) {
                    if (oma) oma[w] = ma[k];
                    omi[w] = mi[k];
                    ov[w] = v[k];
                    ++w;
                }
        });
    });
    nb::object rows = ma ? readonly(std::move(out_ma)) : nb::none();
    return nb::make_tuple(rows, readonly(std::move(out_mi)), wrap(std::move(out_v)));
}

int64_t count_nonzero(nb::handle data_h) {
    const NDArray& data = vector_arg(data_h, "_count_nonzero");
    const int64_t n = data.shape[0], nt = num_chunks(n);
    std::vector<int64_t> count(static_cast<std::size_t>(std::max<int64_t>(nt, 1)), 0);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        for_chunks(n, [&](int64_t t, int64_t lo, int64_t hi) {
            int64_t c = 0;
            for (int64_t k = lo; k < hi; ++k) c += is_nonzero(v[k]);
            count[t] = c;
        });
    });
    int64_t total = 0;
    for (int64_t c : count) total += c;
    return total;
}

nb::object bincount_nonzero(nb::handle idx_h, nb::handle data_h, int64_t length) {
    const NDArray& idx = vector_arg(idx_h, "_bincount_nonzero", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_bincount_nonzero");
    const int64_t n = idx.shape[0];
    NDArray out = new_index(length, true);
    int64_t* o = i64(out);
    const int64_t* ix = i64(idx);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        const parallel::Unlocked unlocked(n >= kParallelThreshold);
        for (int64_t k = 0; k < n; ++k) o[ix[k]] += is_nonzero(v[k]);
    });
    return wrap(std::move(out));
}

nb::object rowcount_nonzero(nb::handle indptr_h, nb::handle data_h) {
    const NDArray& indptr = vector_arg(indptr_h, "_rowcount_nonzero", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_rowcount_nonzero");
    const int64_t n_major = indptr.shape[0] - 1, nnz = data.shape[0];
    NDArray out = new_index(n_major);
    int64_t* o = i64(out);
    const int64_t* ip = i64(indptr);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        for_chunks(
            n_major,
            [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t r = lo; r < hi; ++r) {
                    int64_t c = 0;
                    for (int64_t k = ip[r]; k < ip[r + 1]; ++k) c += is_nonzero(v[k]);
                    o[r] = c;
                }
            },
            lines_per_task(n_major > 0 ? nnz / n_major : 0), nnz + n_major
        );
    });
    return wrap(std::move(out));
}

nb::tuple reshape_coords(nb::handle major_h, nb::handle minor_h, nb::handle old_h, nb::handle new_h) {
    const std::vector<int64_t> old_shape = shape_of(old_h), new_shape = shape_of(new_h);
    const NDArray& minor = vector_arg(minor_h, "_reshape_coords", DType::Int64);
    const int64_t nnz = minor.shape[0];
    const int64_t* ma = opt_index(major_h, "_reshape_coords", nnz);
    const int64_t* mi = i64(minor);
    const unsigned __int128 old_cols = static_cast<unsigned __int128>(old_shape.back());
    const bool two = new_shape.size() == 2;
    const unsigned __int128 new_cols = static_cast<unsigned __int128>(new_shape.back());
    NDArray out_mi = new_index(nnz);
    NDArray out_ma = two ? new_index(nnz) : NDArray();
    int64_t* omi = i64(out_mi);
    int64_t* oma = two ? i64(out_ma) : nullptr;
    for_chunks(nnz, [&](int64_t, int64_t lo, int64_t hi) {
        for (int64_t k = lo; k < hi; ++k) {
            const unsigned __int128 flat =
                (ma ? static_cast<unsigned __int128>(ma[k]) * old_cols : 0) + static_cast<unsigned __int128>(mi[k]);
            if (two) {
                oma[k] = static_cast<int64_t>(flat / new_cols);
                omi[k] = static_cast<int64_t>(flat % new_cols);
            } else {
                omi[k] = static_cast<int64_t>(flat);
            }
        }
    });
    nb::object rows = two ? readonly(std::move(out_ma)) : nb::none();
    return nb::make_tuple(rows, readonly(std::move(out_mi)));
}

}  // namespace

void bind_convert(nb::module_& m) {
    m.def("_expand", [](nb::handle indptr_h, int64_t n_major, int64_t nnz) {
        const NDArray& indptr = vector_arg(indptr_h, "_expand", DType::Int64);
        if (indptr.shape[0] != n_major + 1) throw nb::value_error("_expand(): lengths differ");
        return readonly(expand(indptr, n_major, nnz));
    });
    m.def("_from_dense", &from_dense, nb::arg("x"), nb::kw_only(), nb::arg("by_rows"));
    m.def(
        "_to_dense", &to_dense, nb::arg("shape"), nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::kw_only(),
        nb::arg("by_rows")
    );
    m.def(
        "_coo_to_dense", &coo_to_dense, nb::arg("shape"), nb::arg("major").none(), nb::arg("minor"), nb::arg("data"),
        nb::kw_only(), nb::arg("distinct")
    );
    m.def(
        "_filter_compressed", &filter_compressed, nb::arg("n_major"), nb::arg("indptr"), nb::arg("indices"),
        nb::arg("data"), nb::arg("new_major"), nb::arg("new_minor"), nb::kw_only(), nb::arg("drop_zeros")
    );
    m.def(
        "_filter_coo", &filter_coo, nb::arg("major").none(), nb::arg("minor"), nb::arg("data"), nb::arg("bound_major"),
        nb::arg("bound_minor"), nb::kw_only(), nb::arg("drop_zeros")
    );
    m.def("_count_nonzero", &count_nonzero);
    m.def("_bincount_nonzero", &bincount_nonzero);
    m.def("_rowcount_nonzero", &rowcount_nonzero);
    m.def(
        "_reshape_coords", &reshape_coords, nb::arg("major").none(), nb::arg("minor"), nb::arg("old_shape"),
        nb::arg("new_shape")
    );
}

}  // namespace sd::sparse
