// The storage of sd.sparse (docs/design/sparse.md §2.2): the pieces every kernel shares.
//
// A sparse array lives in Python (src/stardust/sparse) as sd.ndarray fields: COO keeps `data`
// and one int64 index array per axis (`coords`), CSR and CSC keep `data`, `indices` and
// `indptr`. CSR and CSC are always canonical (docs/design/sparse.md §2.3): within every line (a
// row of CSR, a column of CSC) the indices increase strictly. The kernels here read those
// arrays as raw pointers: the Python layer hands them contiguous int64 index arrays it owns (or
// has validated) and contiguous data of one of the four dtypes.
//
// Words used throughout: the *major* axis is the one the lines run along (rows of CSR, columns
// of CSC); the *minor* index is the position inside a line. A COO array converted to CSR has
// major = row, minor = col; to CSC major = col, minor = row. A 1-D array is one line (major
// index 0 for every entry).
#pragma once

#include "array/ndarray.hpp"
#include "base/dtype.hpp"

#include <nanobind/nanobind.h>

#include <complex>
#include <cstdint>
#include <string>

namespace sd::sparse {

void bind_canonical(nanobind::module_& m);
void bind_convert(nanobind::module_& m);
void bind_dok(nanobind::module_& m);
void bind_spmv(nanobind::module_& m);
void bind_binop(nanobind::module_& m);
void bind_spgemm(nanobind::module_& m);
void bind_reduce(nanobind::module_& m);
void bind_index(nanobind::module_& m);
void bind_construct(nanobind::module_& m);
void bind_triangular(nanobind::module_& m);
void bind_iterative(nanobind::module_& m);

// ---- element helpers ----

// The sum of duplicates, left to right in input order (docs/design/sparse.md §2.3). bool adds
// as a logical or (scipy's npy_bool), int64 wraps (scipy, numpy).
template <class T>
inline T add_value(T a, T b) {
    if constexpr (std::is_same_v<T, bool>) return a || b;
    else if constexpr (std::is_same_v<T, std::int64_t>)
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
    else return a + b;
}

// x != 0 as the sparse area counts it: NaN is nonzero, -0.0 is zero, a complex value is nonzero
// when either part is (scipy and numpy's `!= 0`).
template <class T>
inline bool is_nonzero(T v) {
    if constexpr (std::is_same_v<T, complex128>) return v.real() != 0.0 || v.imag() != 0.0;
    else return v != T(0);
}

// The value a dense result holds where `v` is stored: the implicit zero plus v (scipy's
// toarray adds the stored values to zeros, so a stored -0.0 is +0.0 in the dense array).
template <class T>
inline T dense_value(T v) {
    if constexpr (std::is_same_v<T, double>) return 0.0 + v;
    else if constexpr (std::is_same_v<T, complex128>) return complex128(0.0 + v.real(), 0.0 + v.imag());
    else return v;
}

// ---- array access ----

inline const std::int64_t* i64(const NDArray& a) { return reinterpret_cast<const std::int64_t*>(a.data); }
inline std::int64_t* i64(NDArray& a) { return reinterpret_cast<std::int64_t*>(a.data); }
template <class T>
inline const T* typed(const NDArray& a) {
    return reinterpret_cast<const T*>(a.data);
}
template <class T>
inline T* typed(NDArray& a) {
    return reinterpret_cast<T*>(a.data);
}

// A contiguous 1-D sd.ndarray of `dtype` (any dtype when `dtype` is String, used as "any").
// The Python layer guarantees it; a TypeError names the private function otherwise.
const NDArray& vector_arg(nanobind::handle o, const char* func, DType dtype = DType::String);

// A new 1-D int64 array of n elements, uninitialized (or zeroed), writeable.
NDArray new_index(std::int64_t n, bool zeroed = false);

// ---- the canonical compressed form ----

// The canonical compressed arrays of entries (major[k], minor[k], data[k]), k < nnz:
// indptr (n_major + 1), indices and data (one entry per distinct position). Entries are ordered
// by (major, minor); duplicates are summed in input order. `major` may be null (a 1-D array:
// every entry on line 0). `presorted`: within each line the entries already arrive with minor
// strictly increasing (a transpose); then nothing is sorted or summed within a line. The GIL is
// held on entry; the work runs without it when large.
struct Compressed {
    NDArray indptr, indices, data;
};
Compressed compress(
    std::int64_t n_major,
    const std::int64_t* major,
    const std::int64_t* minor,
    const NDArray& data,
    std::int64_t nnz,
    bool presorted
);

// The canonical compressed arrays of the transpose of canonical compressed arrays (CSR -> CSC and
// back): new arrays. The GIL is held on entry.
Compressed transpose(
    std::int64_t n_major, std::int64_t n_minor, const NDArray& indptr, const NDArray& indices, const NDArray& data
);

// The major index of every entry of canonical compressed arrays (indptr of n_major + 1).
NDArray expand(const NDArray& indptr, std::int64_t n_major, std::int64_t nnz);

// Whether the entries are already canonical: major non-decreasing, and within a major index the
// minor index strictly increasing (major null: minor strictly increasing).
bool is_canonical(const std::int64_t* major, const std::int64_t* minor, std::int64_t nnz);

// indptr (n_major + 1) of canonical entries whose major indices are `major` (null: one line).
void indptr_of_sorted(const std::int64_t* major, std::int64_t nnz, std::int64_t n_major, std::int64_t* indptr);

}  // namespace sd::sparse
