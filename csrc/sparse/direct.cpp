// Direct solves of sd.sparse through Accelerate's LU (docs/design/sparse.md §3.6).
//
//   _lu_factor(shape, indptr, indices, data, /, *, csr, func) -> _LuFactor
//       Factors the square matrix given by its canonical compressed arrays (CSC, or CSR with
//       csr=True: the CSC of the transpose, handed to Accelerate with the transpose flag) with
//       LU/AMD (decision SP59) and checks that it is not singular (SP62):
//         - Accelerate reports a structurally singular matrix (SparseMatrixIsSingular);
//         - a solve inside the condition estimate gives a value that is not finite (an exactly
//           zero pivot, zeroTolerance 0, or overflow);
//         - kappa_1 = ||A||_1 * est(||A^-1||_1) has kappa_1 * eps >= 1. est is LAPACK's 1-norm
//           estimator (lacn2: Hager 1984, Higham 1988; a lower bound of ||A^-1||_1, fixed start,
//           no randomness) driven by solves with the factors of A and A^H. An exactly singular
//           matrix whose rounded pivots are not exactly zero factors "successfully" and solves to
//           about 1/eps along its null space (docs/design/sparse.md §1.4, §7.5); this sees it.
//       All three are LinAlgError("<func>(): singular matrix"). The factor and the estimate run
//       without the GIL. indptr and indices are int64, data float64 or complex128, contiguous;
//       the Python layer guarantees a canonical square matrix of finite values, n >= 1.
//   _LuFactor.solve(b, /, *, trans, func)
//       op(A) x = b, op = 'N', 'T' or 'H', for b of shape (n,) or (n, k) (any strides) of the
//       factor's dtype, or complex128 for a float64 factor (its real and imaginary parts are
//       solved as 2k real columns). The columns are solved together by Accelerate (SP64): the
//       result is fixed by k and the values, and a column may differ in its last bits from the
//       same column solved alone. A solution that is not finite is LinAlgError("<func>():
//       singular matrix"). Returns a new C-contiguous array shaped like b.
//   _LuFactor.cond      the estimate kappa_1 of the check above.
//   _LuFactor.n, .dtype
//
//   _lu_solve(shape, indptr, indices, data, b, /, *, csr)
//       SP0's scaffold of the wrapper (tests/sparse/test_accelerate.py): factor and solve with no
//       singularity check beyond Accelerate's status.
//
// Errors: a parameter Accelerate rejects (its reportError message) is ValueError; a failed
// factorization is LinAlgError with Accelerate's message; a dimension beyond Accelerate's int is
// ValueError.
#include "sparse/direct.hpp"

#include "accelerate/sparse.hpp"
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "linalg/errors.hpp"
#include "linalg/lapack.hpp"
#include "sparse/format.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;
using c128 = std::complex<double>;

constexpr double kEps = std::numeric_limits<double>::epsilon();

const NDArray& contiguous(nb::handle o, const char* what, DType dtype, const char* func) {
    const NDArray& a = require_ndarray(o, what);
    if (a.dtype != dtype || !a.c_contiguous)
        throw nb::type_error((std::string(func) + "(): wrong array layout").c_str());
    return a;
}

[[noreturn]] void raise_failure(const sparse_solve::Error& e, const std::string& func) {
    switch (e.kind) {
        case sparse_solve::Failure::Singular: throw linalg::LinAlgError(func + "(): singular matrix");
        case sparse_solve::Failure::Parameter: throw nb::value_error((func + "(): " + e.what()).c_str());
        default: throw linalg::LinAlgError(func + "(): " + e.what());
    }
}

template <class T>
bool all_finite(const T* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, double>) {
            if (!std::isfinite(x[i])) return false;
        } else if (!std::isfinite(x[i].real()) || !std::isfinite(x[i].imag())) {
            return false;
        }
    }
    return true;
}

template <class T>
double magnitude(T v) {
    if constexpr (std::is_same_v<T, double>) return std::fabs(v);
    else return std::abs(v);
}

// ||A||_1 of the user's matrix: the largest column sum of magnitudes. The stored CSC is A itself
// (lines are columns) or, for csr, A^T (lines are rows: the column sums scatter over indices).
template <class T>
double norm1(int64_t n, const int64_t* ip, const int64_t* ix, const T* v, bool csr) {
    double best = 0.0;
    if (!csr) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t p = ip[j]; p < ip[j + 1]; ++p) s += magnitude(v[p]);
            best = std::max(best, s);
        }
        return best;
    }
    std::vector<double> s(static_cast<std::size_t>(n), 0.0);
    for (int64_t p = 0; p < ip[n]; ++p) s[static_cast<std::size_t>(ix[p])] += magnitude(v[p]);
    for (double x : s) best = std::max(best, x);
    return best;
}

// est(||A^-1||_1) by lacn2's reverse communication; false when a solve is not finite.
template <class T>
bool inverse_norm1(const sparse_solve::Lu<T>& lu, int64_t n, double& est) {
    using linalg::lint;
    std::vector<T> v(static_cast<std::size_t>(n)), x(static_cast<std::size_t>(n));
    std::vector<lint> isgn(static_cast<std::size_t>(n));
    lint isave[3] = {0, 0, 0};
    lint kase = 0;
    est = 0.0;
    for (;;) {
        linalg::lapack::lacn2<T>(static_cast<lint>(n), v.data(), x.data(), isgn.data(), est, kase, isave);
        if (kase == 0) return true;
        lu.solve(x.data(), 1, n, kase == 1 ? 'N' : 'H');
        if (!all_finite(x.data(), n)) return false;
    }
}

struct LuFactor {
    DType dtype = DType::Float64;
    int64_t n = 0;
    double cond = 0.0;
    std::optional<sparse_solve::Lu<double>> real;
    std::optional<sparse_solve::Lu<c128>> complex_;
};

template <class T>
std::unique_ptr<LuFactor> factor_typed(
    int64_t n, const NDArray& indptr, const NDArray& indices, const NDArray& data, bool csr, const std::string& func
) {
    auto out = std::make_unique<LuFactor>();
    out->dtype = data.dtype;
    out->n = n;
    const auto* ip = i64(indptr);
    const auto* ix = i64(indices);
    const auto* v = typed<T>(data);
    bool finite = true;
    double est = 0.0, anorm = 0.0;
    try {
        nb::gil_scoped_release unlocked;
        const auto a = sparse_solve::Csc::narrow(n, n, ip, ix, csr);
        auto lu = sparse_solve::Lu<T>::factor(a, v);
        anorm = norm1(n, ip, ix, v, csr);
        finite = inverse_norm1(lu, n, est);
        if constexpr (std::is_same_v<T, double>) out->real.emplace(std::move(lu));
        else out->complex_.emplace(std::move(lu));
    } catch (const sparse_solve::Error& e) { raise_failure(e, func); } catch (const std::length_error& e) {
        throw nb::value_error((func + "(): " + e.what()).c_str());
    }
    out->cond = anorm * est;
    if (!finite || !(out->cond * kEps < 1.0)) throw linalg::LinAlgError(func + "(): singular matrix");
    return out;
}

std::unique_ptr<LuFactor> lu_factor_entry(
    nb::handle shape, nb::handle indptr_o, nb::handle indices_o, nb::handle data_o, bool csr, const std::string& func
) {
    const auto dims = parse_shape(shape);
    if (dims.size() != 2 || dims[0] != dims[1] || dims[0] < 1)
        throw nb::value_error("_lu_factor(): 'shape' must be square and not empty");
    const NDArray& indptr = contiguous(indptr_o, "indptr", DType::Int64, "_lu_factor");
    const NDArray& indices = contiguous(indices_o, "indices", DType::Int64, "_lu_factor");
    const NDArray& data = require_ndarray(data_o, "data");
    if (!data.c_contiguous || data.ndim != 1 || indices.ndim != 1 || indices.shape[0] != data.shape[0] ||
        indptr.ndim != 1 || indptr.shape[0] != dims[0] + 1)
        throw nb::value_error("_lu_factor(): the compressed arrays do not match");
    if (data.dtype == DType::Float64) return factor_typed<double>(dims[0], indptr, indices, data, csr, func);
    if (data.dtype == DType::Complex128) return factor_typed<c128>(dims[0], indptr, indices, data, csr, func);
    throw nb::type_error("_lu_factor(): 'data' must be float64 or complex128");
}

// b (n,) or (n, k) with any strides -> column-major columns of T (a complex b against a real
// factor: the k real parts, then the k imaginary parts).
template <class T, class B>
void gather(const NDArray& b, int64_t n, int64_t k, std::vector<T>& cols) {
    const int64_t s0 = b.strides[0] / static_cast<int64_t>(sizeof(B));
    const int64_t s1 = b.ndim == 2 ? b.strides[1] / static_cast<int64_t>(sizeof(B)) : 0;
    const auto* bp = reinterpret_cast<const B*>(b.data);
    constexpr bool split = std::is_same_v<T, double> && std::is_same_v<B, c128>;
    cols.resize(static_cast<std::size_t>(n * k * (split ? 2 : 1)));
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i < n; ++i) {
            const B v = bp[i * s0 + j * s1];
            if constexpr (split) {
                cols[static_cast<std::size_t>(j * n + i)] = v.real();
                cols[static_cast<std::size_t>((k + j) * n + i)] = v.imag();
            } else {
                cols[static_cast<std::size_t>(j * n + i)] = static_cast<T>(v);
            }
        }
}

template <class T, class B>
nb::object solve_typed(const sparse_solve::Lu<T>& lu, const NDArray& b, char op, const std::string& func) {
    const int64_t n = lu.n();
    const int64_t k = b.ndim == 2 ? b.shape[1] : 1;
    constexpr bool split = std::is_same_v<T, double> && std::is_same_v<B, c128>;
    std::vector<T> cols;
    gather<T, B>(b, n, k, cols);
    bool finite = true;
    try {
        nb::gil_scoped_release unlocked;
        if (k > 0) lu.solve(cols.data(), split ? 2 * k : k, n, op);
        finite = all_finite(cols.data(), static_cast<int64_t>(cols.size()));
    } catch (const sparse_solve::Error& e) { raise_failure(e, func); } catch (const std::length_error& e) {
        throw nb::value_error((func + "(): " + e.what()).c_str());
    }
    if (!finite) throw linalg::LinAlgError(func + "(): singular matrix");
    NDArray out = new_array(b.dtype, b.shape_span());
    auto* o = typed<B>(out);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < k; ++j) {
            if constexpr (split)
                o[i * k + j] =
                    B(cols[static_cast<std::size_t>(j * n + i)], cols[static_cast<std::size_t>((k + j) * n + i)]);
            else o[i * k + j] = cols[static_cast<std::size_t>(j * n + i)];
        }
    return wrap(std::move(out));
}

nb::object lu_factor_solve(const LuFactor& f, nb::handle b_o, const std::string& trans, const std::string& func) {
    const NDArray& b = require_ndarray(b_o, "b");
    if (b.ndim < 1 || b.ndim > 2 || b.shape[0] != f.n) throw nb::value_error((func + "(): wrong shape of 'b'").c_str());
    if (trans != "N" && trans != "T" && trans != "H") throw nb::value_error((func + "(): wrong 'trans'").c_str());
    const char op = trans[0];
    if (f.real) {
        if (b.dtype == DType::Float64) return solve_typed<double, double>(*f.real, b, op, func);
        if (b.dtype == DType::Complex128) return solve_typed<double, c128>(*f.real, b, op, func);
    } else if (b.dtype == DType::Complex128) {
        return solve_typed<c128, c128>(*f.complex_, b, op, func);
    }
    throw nb::type_error((func + "(): 'b' has the wrong dtype for the factor").c_str());
}

// ---- SP0's scaffold ----

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
    } catch (const sparse_solve::Error& e) { raise_failure(e, "_lu_solve"); } catch (const std::length_error& e) {
        throw nb::value_error((std::string("_lu_solve(): ") + e.what()).c_str());
    }
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
    const NDArray& indptr = contiguous(indptr_o, "indptr", DType::Int64, "_lu_solve");
    const NDArray& indices = contiguous(indices_o, "indices", DType::Int64, "_lu_solve");
    const NDArray& data = require_ndarray(data_o, "data");
    const NDArray& b = contiguous(b_o, "b", data.dtype, "_lu_solve");
    if (!data.c_contiguous || data.ndim != 1 || indices.ndim != 1 || indices.shape[0] != data.shape[0])
        throw nb::value_error("_lu_solve(): 'indices' and 'data' must be 1-D of the same length");
    if (data.dtype == DType::Float64) return wrap(lu_solve<double>(dims[0], dims[1], indptr, indices, data, b, csr));
    if (data.dtype == DType::Complex128)
        return wrap(lu_solve<std::complex<double>>(dims[0], dims[1], indptr, indices, data, b, csr));
    throw nb::type_error("_lu_solve(): 'data' must be float64 or complex128");
}

}  // namespace

void bind_direct(nb::module_& m) {
    nb::class_<LuFactor>(m, "_LuFactor")
        .def_prop_ro("n", [](const LuFactor& f) { return f.n; })
        .def_prop_ro("cond", [](const LuFactor& f) { return f.cond; })
        .def_prop_ro("complex", [](const LuFactor& f) { return !f.real.has_value(); })
        .def("solve", &lu_factor_solve, nb::arg("b"), nb::kw_only(), nb::arg("trans"), nb::arg("func"));
    m.def(
        "_lu_factor", &lu_factor_entry, nb::arg("shape"), nb::arg("indptr"), nb::arg("indices"), nb::arg("data"),
        nb::kw_only(), nb::arg("csr"), nb::arg("func")
    );
    m.def(
        "_lu_solve", &lu_solve_entry, nb::arg("shape"), nb::arg("indptr"), nb::arg("indices"), nb::arg("data"),
        nb::arg("b"), nb::kw_only(), nb::arg("csr") = false
    );
}

}  // namespace sd::sparse
