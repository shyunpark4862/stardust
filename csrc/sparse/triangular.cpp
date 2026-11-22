// Triangular solves and the pattern tools of sd.sparse (docs/design/sparse.md §3.6).
//
//   _sptrsv(indptr, indices, data, b, /, *, lower, unit_diagonal, func)
//       x with A x = b for the canonical CSR arrays of a square A, reading only the chosen
//       triangle (lower: the entries with column <= row; upper: column >= row; the others are
//       ignored, as sd.linalg.solve_triangular ignores the other triangle) and, unless
//       unit_diagonal, the diagonal. Forward (lower) or backward (upper) substitution:
//           x_i = (b_i - s_i) / a_ii,   s_i = sum_j a_ij x_j over the row's triangle entries,
//       s_i summed as every sparse line sum (sparse/arith.hpp: from zero in ascending column
//       order, fma per term, pieces of kLine past kLine terms). unit_diagonal: x_i = b_i - s_i.
//       b is (n,) or (n, k), C-contiguous, of data's dtype (float64 or complex128); every column
//       is solved on its own, so a column's bits do not depend on k or on the other columns.
//       Columns are split into fixed blocks over the pool (the bits do not depend on the thread
//       count); one right-hand side runs on one thread (no level scheduling: a later candidate,
//       docs/design/sparse.md §3.6).
//       A missing or zero diagonal entry (not unit_diagonal) is LinAlgError("<func>(): singular
//       matrix"), checked before solving.
//   _bandwidth(indptr, indices, /) -> (below, above)
//       For the CSR arrays of a 2-D matrix: the largest row - col and col - row over the stored
//       entries (stored zeros count), each at least 0; (0, 0) without entries (scipy's
//       spbandwidth). A CSC caller swaps the two.
//   _triangular(indptr, indices, /) -> (lower, upper)
//       For the CSR arrays of a 2-D matrix: below == 0 and above == 0 of _bandwidth, stopping
//       early once both are false. A CSC caller swaps the two.
#include "array/kernels/scalar_ops.hpp"
#include "bind/pyutil.hpp"
#include "linalg/errors.hpp"
#include "sparse/arith.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;

constexpr int64_t kColumnBlock = 16;

template <class T>
inline bool is_zero(T v) {
    if constexpr (std::is_same_v<T, complex128>) return v.real() == 0.0 && v.imag() == 0.0;
    else return v == 0.0;
}

template <class T>
inline T minus(T a, T b) {
    if constexpr (std::is_same_v<T, complex128>) return {a.real() - b.real(), a.imag() - b.imag()};
    else return a - b;
}

template <class T>
inline T divide(T a, T b) {
    if constexpr (std::is_same_v<T, complex128>) return ops::cdiv(a, b);
    else return a / b;
}

template <class T>
nb::object sptrsv_typed(
    const NDArray& indptr,
    const NDArray& indices,
    const NDArray& data,
    const NDArray& b,
    bool lower,
    bool unit,
    const std::string& func
) {
    const int64_t n = indptr.shape[0] - 1;
    const int64_t k = b.ndim == 2 ? b.shape[1] : 1;
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const T* v = typed<T>(data);
    const T* bp = typed<T>(b);
    const int64_t nnz = ip[n];
    if (k == 1) {
        // one right-hand side: the search, the singularity check and the substitution in one pass
        // over the rows (the same sums; a singular matrix raises after the pass, nothing returned)
        NDArray out = new_array(dtype_of<T>, b.shape_span());
        T* x = typed<T>(out);
        bool singular = false;
        {
            const parallel::Unlocked unlocked(nnz + n >= kParallelThreshold);
            std::vector<T> parts;
            auto row = [&](int64_t i) {
                const int64_t s = ip[i], e = ip[i + 1];
                const int64_t p = std::lower_bound(ix + s, ix + e, i) - ix;
                const bool has = p < e && ix[p] == i;
                if (!unit && (!has || is_zero(v[p]))) singular = true;
                const int64_t lo = lower ? s : p + (has ? 1 : 0);
                const int64_t hi = lower ? p : e;
                auto step = [&](int64_t q, T acc) { return madd(v[q], x[ix[q]], acc); };
                const T r = minus(bp[i], ordered_sum<T>(lo, hi, step, parts));
                x[i] = unit ? r : divide(r, has ? v[p] : T(1));
            };
            if (lower)
                for (int64_t i = 0; i < n; ++i) row(i);
            else
                for (int64_t i = n - 1; i >= 0; --i) row(i);
        }
        if (singular) throw linalg::LinAlgError(func + "(): singular matrix");
        return wrap(std::move(out));
    }
    // split[i]: the first entry of row i with column >= i; diag[i]: whether that entry is (i, i).
    std::vector<int64_t> split(static_cast<std::size_t>(n));
    std::vector<char> diag(static_cast<std::size_t>(n));
    bool singular = false;
    for_tasks(1, nnz + n, [&](int64_t) {
        for (int64_t i = 0; i < n; ++i) {
            const int64_t p = std::lower_bound(ix + ip[i], ix + ip[i + 1], i) - ix;
            split[static_cast<std::size_t>(i)] = p;
            const bool has = p < ip[i + 1] && ix[p] == i;
            diag[static_cast<std::size_t>(i)] = has;
            if (!unit && (!has || is_zero(v[p]))) singular = true;
        }
    });
    if (singular) throw linalg::LinAlgError(func + "(): singular matrix");
    NDArray out = new_array(dtype_of<T>, b.shape_span());
    T* x = typed<T>(out);
    const int64_t blocks = (k + kColumnBlock - 1) / kColumnBlock;
    for_tasks(blocks, (nnz + n) * std::max<int64_t>(k, 1), [&](int64_t t) {
        const int64_t c0 = t * kColumnBlock, c1 = std::min(k, c0 + kColumnBlock);
        std::vector<T> parts;
        auto row = [&](int64_t i) {
            const int64_t s = split[static_cast<std::size_t>(i)];
            const bool has = diag[static_cast<std::size_t>(i)];
            const int64_t lo = lower ? ip[i] : s + (has ? 1 : 0);
            const int64_t hi = lower ? s : ip[i + 1];
            for (int64_t c = c0; c < c1; ++c) {
                auto step = [&](int64_t p, T acc) { return madd(v[p], x[ix[p] * k + c], acc); };
                const T r = minus(bp[i * k + c], ordered_sum<T>(lo, hi, step, parts));
                x[i * k + c] = unit ? r : divide(r, v[s]);
            }
        };
        if (lower)
            for (int64_t i = 0; i < n; ++i) row(i);
        else
            for (int64_t i = n - 1; i >= 0; --i) row(i);
    });
    return wrap(std::move(out));
}

nb::object sptrsv_entry(
    nb::handle indptr_o,
    nb::handle indices_o,
    nb::handle data_o,
    nb::handle b_o,
    bool lower,
    bool unit,
    const std::string& func
) {
    const NDArray& indptr = vector_arg(indptr_o, "_sptrsv", DType::Int64);
    const NDArray& indices = vector_arg(indices_o, "_sptrsv", DType::Int64);
    const NDArray& data = vector_arg(data_o, "_sptrsv");
    const NDArray& b = require_ndarray(b_o, "b");
    const int64_t n = indptr.shape[0] - 1;
    if (n < 0 || indices.shape[0] != data.shape[0] || b.ndim < 1 || b.ndim > 2 || b.shape[0] != n || !b.c_contiguous ||
        b.dtype != data.dtype)
        throw nb::value_error("_sptrsv(): the arrays do not match");
    if (data.dtype == DType::Float64) return sptrsv_typed<double>(indptr, indices, data, b, lower, unit, func);
    if (data.dtype == DType::Complex128) return sptrsv_typed<complex128>(indptr, indices, data, b, lower, unit, func);
    throw nb::type_error("_sptrsv(): 'data' must be float64 or complex128");
}

nb::tuple bandwidth_entry(nb::handle indptr_o, nb::handle indices_o) {
    const NDArray& indptr = vector_arg(indptr_o, "_bandwidth", DType::Int64);
    const NDArray& indices = vector_arg(indices_o, "_bandwidth", DType::Int64);
    const int64_t n = indptr.shape[0] - 1;
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    // the row's first and last columns bound its gaps; a maximum per chunk of rows, then over chunks
    const int64_t nt = num_chunks(n);
    std::vector<int64_t> below(static_cast<std::size_t>(std::max<int64_t>(nt, 1)), 0), above(below);
    for_chunks(
        n,
        [&](int64_t t, int64_t lo, int64_t hi) {
            int64_t b = 0, a = 0;
            for (int64_t i = lo; i < hi; ++i) {
                if (ip[i] == ip[i + 1]) continue;
                b = std::max(b, i - ix[ip[i]]);
                a = std::max(a, ix[ip[i + 1] - 1] - i);
            }
            below[static_cast<std::size_t>(t)] = b;
            above[static_cast<std::size_t>(t)] = a;
        },
        kChunk, n + ip[n]
    );
    return nb::make_tuple(*std::max_element(below.begin(), below.end()), *std::max_element(above.begin(), above.end()));
}

// (lower, upper) of the CSR arrays: every stored column <= its row, every one >= its row. Stops
// as soon as both are false (a few rows of a general matrix: the middle, first and last row are
// looked at first, as scipy's is_sptriangular does).
nb::tuple triangular_entry(nb::handle indptr_o, nb::handle indices_o) {
    const NDArray& indptr = vector_arg(indptr_o, "_triangular", DType::Int64);
    const NDArray& indices = vector_arg(indices_o, "_triangular", DType::Int64);
    const int64_t n = indptr.shape[0] - 1;
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    bool lower = true, upper = true;
    auto row = [&](int64_t i, bool& lo, bool& up) {
        if (ip[i] == ip[i + 1]) return;
        lo = lo && ix[ip[i + 1] - 1] <= i;
        up = up && ix[ip[i]] >= i;
    };
    if (n > 0)
        for (int64_t i : {n / 2, int64_t(0), n - 1}) row(i, lower, upper);
    if (!lower && !upper) return nb::make_tuple(false, false);
    const int64_t nt = num_chunks(n);
    std::vector<char> lo_ok(static_cast<std::size_t>(std::max<int64_t>(nt, 1)), 1), up_ok(lo_ok);
    std::atomic<bool> both_failed{false};  // only ends the scan early: the answer is (false, false) then
    for_chunks(
        n,
        [&](int64_t t, int64_t lo, int64_t hi) {
            bool l = lower, u = upper;
            for (int64_t i0 = lo; i0 < hi && (l || u); i0 += 1024) {
                if (both_failed.load(std::memory_order_relaxed)) {
                    l = u = false;
                    break;
                }
                bool l2 = true, u2 = true;
                const int64_t i1 = std::min(hi, i0 + 1024);
                for (int64_t i = i0; i < i1; ++i) {
                    if (ip[i] == ip[i + 1]) continue;
                    l2 &= ix[ip[i + 1] - 1] <= i;
                    u2 &= ix[ip[i]] >= i;
                }
                l = l && l2;
                u = u && u2;
            }
            lo_ok[static_cast<std::size_t>(t)] = l;
            up_ok[static_cast<std::size_t>(t)] = u;
            if (!l && !u) both_failed.store(true, std::memory_order_relaxed);
        },
        kChunk, n + ip[n]
    );
    for (int64_t t = 0; t < nt; ++t) {
        lower = lower && lo_ok[static_cast<std::size_t>(t)];
        upper = upper && up_ok[static_cast<std::size_t>(t)];
    }
    if (both_failed.load()) lower = upper = false;
    return nb::make_tuple(lower, upper);
}

}  // namespace

void bind_triangular(nb::module_& m) {
    m.def(
        "_sptrsv", &sptrsv_entry, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("b"), nb::kw_only(),
        nb::arg("lower"), nb::arg("unit_diagonal"), nb::arg("func")
    );
    m.def("_bandwidth", &bandwidth_entry, nb::arg("indptr"), nb::arg("indices"));
    m.def("_triangular", &triangular_entry, nb::arg("indptr"), nb::arg("indices"));
}

}  // namespace sd::sparse
