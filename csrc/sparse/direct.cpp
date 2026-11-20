// Direct solves of sd.sparse through Accelerate's LU (docs/design/sparse.md §3.6).
//
//   _lu_solve(shape, indptr, indices, data, b, /, *, csr)
//       Factors the matrix given by its compressed arrays (CSC, or CSR with csr=True: the
//       CSC of the transpose, handed to Accelerate with the transpose flag) with LU/AMD and
//       solves A x = b for b of shape (n,) or (n, k). indptr and indices are int64, data and b
//       float64 or complex128 (the same dtype), all contiguous; the arrays are trusted to be a
//       canonical compressed matrix (the public functions of SP4 check them). Returns x like b.
//
// Errors: a structurally singular matrix is LinAlgError("singular matrix"); a parameter Accelerate
// rejects (its reportError message) is ValueError; a failed factorization is LinAlgError with
// Accelerate's message; a dimension beyond Accelerate's int is ValueError. The numerical
// singularity check and the argument checks of the public functions come with SP4.
#include "sparse/direct.hpp"

#include "accelerate/sparse.hpp"
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "linalg/errors.hpp"

#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

const NDArray& contiguous(nb::handle o, const char* what, DType dtype) {
    const NDArray& a = require_ndarray(o, what);
    if (a.dtype != dtype || !a.c_contiguous) throw nb::type_error("_lu_solve(): wrong array layout");
    return a;
}

// rows x cols as the user sees it; the stored CSC is cols x rows when csr (the transpose). A
// rectangular matrix reaches Accelerate, which rejects it through reportError.
template <class T>
NDArray lu_solve(
    std::int64_t rows,
    std::int64_t cols,
    const NDArray& indptr,
    const NDArray& indices,
    const NDArray& data,
    const NDArray& b,
    bool csr
) {
    const std::int64_t n = rows;
    const std::int64_t stored_rows = csr ? cols : rows, stored_cols = csr ? rows : cols;
    const std::int64_t k = b.ndim == 2 ? b.shape[1] : 1;
    if (indptr.ndim != 1 || indptr.shape[0] != stored_cols + 1 || b.ndim < 1 || b.ndim > 2 || b.shape[0] != n)
        throw nb::value_error("_lu_solve(): shapes do not match");
    const auto* ip = reinterpret_cast<const std::int64_t*>(indptr.data);
    const auto* ix = reinterpret_cast<const std::int64_t*>(indices.data);
    const auto* v = reinterpret_cast<const T*>(data.data);
    const auto* bb = reinterpret_cast<const T*>(b.data);
    // b is row-major (n, k); Accelerate wants columns.
    std::vector<T> x(static_cast<std::size_t>(n * k));
    for (std::int64_t i = 0; i < n; ++i)
        for (std::int64_t j = 0; j < k; ++j) x[static_cast<std::size_t>(j * n + i)] = bb[i * k + j];
    try {
        nb::gil_scoped_release unlocked;
        const auto a = sparse_solve::Csc::narrow(stored_rows, stored_cols, ip, ix, csr);
        const auto lu = sparse_solve::Lu<T>::factor(a, v);
        lu.solve(x.data(), k, n);
    } catch (const sparse_solve::Error& e) {
        switch (e.kind) {
            case sparse_solve::Failure::Singular: throw linalg::LinAlgError("_lu_solve(): singular matrix");
            case sparse_solve::Failure::Parameter:
                throw nb::value_error((std::string("_lu_solve(): ") + e.what()).c_str());
            default: throw linalg::LinAlgError(std::string("_lu_solve(): ") + e.what());
        }
    } catch (const std::length_error& e) { throw nb::value_error((std::string("_lu_solve(): ") + e.what()).c_str()); }
    NDArray out = new_array(b.dtype, b.shape_span());
    auto* o = reinterpret_cast<T*>(out.data);
    for (std::int64_t i = 0; i < n; ++i)
        for (std::int64_t j = 0; j < k; ++j) o[i * k + j] = x[static_cast<std::size_t>(j * n + i)];
    return out;
}

nb::object lu_solve_entry(
    nb::handle shape, nb::handle indptr_o, nb::handle indices_o, nb::handle data_o, nb::handle b_o, bool csr
) {
    const auto dims = parse_shape(shape);
    if (dims.size() != 2) throw nb::value_error("_lu_solve(): 'shape' must have two entries");
    const NDArray& indptr = contiguous(indptr_o, "indptr", DType::Int64);
    const NDArray& indices = contiguous(indices_o, "indices", DType::Int64);
    const NDArray& data = require_ndarray(data_o, "data");
    const NDArray& b = contiguous(b_o, "b", data.dtype);
    if (!data.c_contiguous || data.ndim != 1 || indices.ndim != 1 || indices.shape[0] != data.shape[0])
        throw nb::value_error("_lu_solve(): 'indices' and 'data' must be 1-D of the same length");
    if (data.dtype == DType::Float64) return wrap(lu_solve<double>(dims[0], dims[1], indptr, indices, data, b, csr));
    if (data.dtype == DType::Complex128)
        return wrap(lu_solve<std::complex<double>>(dims[0], dims[1], indptr, indices, data, b, csr));
    throw nb::type_error("_lu_solve(): 'data' must be float64 or complex128");
}

}  // namespace

void bind_direct(nb::module_& m) {
    m.def(
        "_lu_solve", &lu_solve_entry, nb::arg("shape"), nb::arg("indptr"), nb::arg("indices"), nb::arg("data"),
        nb::arg("b"), nb::kw_only(), nb::arg("csr") = false
    );
}

}  // namespace sd::sparse
