// Accelerate's Sparse Solvers (Solve.h) for sd.sparse's direct solves (docs/design/sparse.md §1.4,
// §3.6): the one place that calls SparseFactor / SparseSolve.
//
// Fixed choices:
//   - the factorization is SparseFactorizationLU (threshold partial pivoting) with SparseOrderAMD,
//     the only ordering that gives the same bits run after run and under concurrent calls
//     (decision SP18, §7: MTMetis, the default, is neither; Metis differs under concurrency);
//   - reportError is always set: with NULL, a parameter error stops the process (__builtin_trap).
//     The messages are collected per thread and turn the call into an Error even when the status
//     is SparseStatusOK (LU given COLAMD reports a message, factors anyway and returns 0);
//   - zeroTolerance is 0: an exactly zero pivot gives inf in the solution instead of a finite
//     wrong one, which the caller's singularity check sees (§3.6).
//
// Matrices are compressed sparse column (CSC) with 32-bit row indices and dimensions: the int64
// arrays of sd.sparse are narrowed once (Csc::narrow). A CSR matrix is passed as the CSC of its
// transpose with the structure's transpose flag set (no copy of the values).
#pragma once

#include <Accelerate/Accelerate.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sd::sparse_solve {

// Why a call failed. Singular is SparseMatrixIsSingular (structurally singular); the caller adds
// its own numerical check.
enum class Failure { Singular, Parameter, Failed, Internal };

struct Error : std::runtime_error {
    Failure kind;
    Error(Failure k, const std::string& message) : std::runtime_error(message), kind(k) {}
};

namespace detail {

// The messages of the current thread's call in progress (ErrorScope); Accelerate reports a
// parameter error on the thread that made the call.
inline thread_local std::string* sink = nullptr;

inline void report(const char* message) {
    if (sink == nullptr) return;
    std::string m = message ? message : "";
    while (!m.empty() && (m.back() == '\n' || m.back() == ' ')) m.pop_back();
    if (!sink->empty()) *sink += "; ";
    *sink += m;
}

}  // namespace detail

// Collects reportError's messages for the duration of one Accelerate call on this thread.
class ErrorScope {
public:
    ErrorScope() : previous_(detail::sink) { detail::sink = &messages_; }
    ~ErrorScope() { detail::sink = previous_; }
    ErrorScope(const ErrorScope&) = delete;
    ErrorScope& operator=(const ErrorScope&) = delete;
    const std::string& messages() const { return messages_; }

private:
    std::string messages_;
    std::string* previous_;
};

// A matrix structure in Accelerate's form: CSC, int dimensions and row indices, long column starts.
struct Csc {
    int rows = 0;
    int cols = 0;
    bool transpose = false;  // the matrix is the transpose of the stored CSC (a CSR input)
    std::vector<long> col_starts;
    std::vector<int> row_indices;

    // From sd.sparse's int64 arrays of a stored rows x cols CSC (indptr has cols + 1 entries).
    // Throws std::length_error when a dimension does not fit Accelerate's int.
    static Csc narrow(
        std::int64_t rows, std::int64_t cols, const std::int64_t* indptr, const std::int64_t* indices, bool transpose
    ) {
        constexpr std::int64_t kMax = std::numeric_limits<int>::max();
        if (rows > kMax || cols > kMax)
            throw std::length_error("the matrix has more than 2147483647 rows or columns, Accelerate's limit");
        Csc a;
        a.rows = static_cast<int>(rows);
        a.cols = static_cast<int>(cols);
        a.transpose = transpose;
        a.col_starts.assign(indptr, indptr + cols + 1);
        const std::int64_t nnz = indptr[cols];
        a.row_indices.resize(static_cast<std::size_t>(nnz));
        for (std::int64_t k = 0; k < nnz; ++k)
            a.row_indices[static_cast<std::size_t>(k)] = static_cast<int>(indices[k]);
        return a;
    }
};

template <class T>
struct Types;

template <>
struct Types<double> {
    using Matrix = SparseMatrix_Double;
    using Dense = DenseMatrix_Double;
    using Factor = SparseOpaqueFactorization_Double;
    using Value = double;
    static Matrix matrix(const Csc& a, const double* values) {
        Matrix m{};
        m.structure.rowCount = a.rows;
        m.structure.columnCount = a.cols;
        m.structure.columnStarts = const_cast<long*>(a.col_starts.data());
        m.structure.rowIndices = const_cast<int*>(a.row_indices.data());
        m.structure.attributes.kind = SparseOrdinary;
        m.structure.attributes.transpose = a.transpose;
        m.structure.blockSize = 1;
        m.data = const_cast<double*>(values);
        return m;
    }
};

template <>
struct Types<std::complex<double>> {
    using Matrix = SparseMatrix_Complex_Double;
    using Dense = DenseMatrix_Complex_Double;
    using Factor = SparseOpaqueFactorization_Complex_Double;
    using Value = __SPARSE_double_complex;
    static Matrix matrix(const Csc& a, const std::complex<double>* values) {
        Matrix m{};
        m.structure.rowCount = a.rows;
        m.structure.columnCount = a.cols;
        m.structure.columnStarts = const_cast<long*>(a.col_starts.data());
        m.structure.rowIndices = const_cast<int*>(a.row_indices.data());
        m.structure.attributes.kind = SparseOrdinary;
        m.structure.attributes.transpose = a.transpose;
        m.structure.blockSize = 1;
        m.data = reinterpret_cast<Value*>(const_cast<std::complex<double>*>(values));
        return m;
    }
};

// The status of a factorization or a solve and the messages of its call, as an Error (or none).
inline void check(SparseStatus_t status, const std::string& messages) {
    if (status == SparseStatusOK && messages.empty()) return;
    switch (status) {
        case SparseMatrixIsSingular:
            throw Error(Failure::Singular, messages.empty() ? "the matrix is structurally singular" : messages);
        case SparseParameterError: throw Error(Failure::Parameter, messages.empty() ? "invalid parameter" : messages);
        case SparseFactorizationFailed:
            throw Error(Failure::Failed, messages.empty() ? "the factorization failed" : messages);
        case SparseStatusOK:  // a message without a failing status (an option Accelerate ignored)
            throw Error(Failure::Parameter, messages);
        default:  // SparseInternalError, SparseStatusReleased, anything newer
            throw Error(Failure::Internal, messages.empty() ? "Accelerate's sparse solver failed" : messages);
    }
}

// An LU factorization (Accelerate owns it; released with the object).
template <class T>
class Lu {
    using Ty = Types<T>;

public:
    // Factors the square matrix a with values (in a's stored order). Throws Error.
    static Lu factor(const Csc& a, const T* values) {
        SparseSymbolicFactorOptions so{};
        so.control = SparseDefaultControl;
        so.orderMethod = SparseOrderAMD;
        so.order = nullptr;
        so.ignoreRowsAndColumns = nullptr;
        so.malloc = std::malloc;
        so.free = std::free;
        so.reportError = &detail::report;
        SparseNumericFactorOptions no{};
        no.control = SparseDefaultControl;
        no.scalingMethod = SparseScalingDefault;
        no.scaling = nullptr;
        no.pivotTolerance = 0.01;  // Accelerate's default for double
        no.zeroTolerance = 0.0;
        ErrorScope errors;
        Lu lu;
        lu.f_ = SparseFactor(SparseFactorizationLU, Ty::matrix(a, values), so, no);
        lu.owned_ = true;
        check(lu.f_.status, errors.messages());
        lu.n_ = a.rows;
        lu.transposed_ = a.transpose;
        return lu;
    }

    Lu(Lu&& o) noexcept : f_(o.f_), owned_(std::exchange(o.owned_, false)), n_(o.n_), transposed_(o.transposed_) {}
    Lu& operator=(Lu&& o) noexcept {
        if (this != &o) {
            release();
            f_ = o.f_;
            owned_ = std::exchange(o.owned_, false);
            n_ = o.n_;
            transposed_ = o.transposed_;
        }
        return *this;
    }
    Lu(const Lu&) = delete;
    Lu& operator=(const Lu&) = delete;
    ~Lu() { release(); }

    int n() const { return n_; }

    // Solves op(A) X = B in place: xb holds B (n x nrhs, column-major with column stride ld >= n)
    // and receives X. op is 'N' (A), 'T' (A^T) or 'H' (A^H; A^T for real values): the transposed
    // solves go through Accelerate's reference-counted transposed factorization (no new factor),
    // A^H as the conjugate of an A^T solve of the conjugate. The columns of an untransposed solve
    // are solved together (Accelerate's blocked solve); a transposed one goes column by column.
    // Throws Error.
    void solve(T* xb, std::int64_t nrhs, std::int64_t ld, char op = 'N') const {
        constexpr std::int64_t kMax = std::numeric_limits<int>::max();
        if (nrhs > kMax || ld > kMax) throw std::length_error("too many right-hand sides for Accelerate");
        // A solve that runs transposed (op 'T' or 'H' on a CSC factor, 'N' on a CSR one) is wrong
        // for more than one column at a time (SDK 26.5, measured in SP4: residuals of order 1 for
        // k = 2 and 5 on every test matrix, real and complex; one column is right). Such solves go
        // one column at a time.
        if (nrhs > 1 && (transposed_ != (op != 'N'))) {
            for (std::int64_t j = 0; j < nrhs; ++j) solve(xb + j * ld, 1, ld, op);
            return;
        }
        typename Ty::Dense d{};
        d.rowCount = n_;
        d.columnCount = static_cast<int>(nrhs);
        d.columnStride = static_cast<int>(ld);
        d.attributes.kind = SparseOrdinary;
        d.data = reinterpret_cast<typename Ty::Value*>(xb);
        ErrorScope errors;
        if (op == 'N') {
            SparseSolve(f_, d);
        } else {
            // A^H x = b is conj(A^T conj(x)) = b: SparseGetConjugateTranspose's solve stops the
            // process (SIGTRAP) on complex LU factors of a few hundred rows (SDK 26.5, measured in
            // SP4 on random_complex/800), so the conjugate goes around the plain transpose.
            constexpr bool cplx = std::is_same_v<T, std::complex<double>>;
            const bool conj = cplx && op == 'H';
            if (conj) conjugate(xb, nrhs, ld);
            typename Ty::Factor t = SparseGetTranspose(f_);
            SparseSolve(t, d);
            SparseCleanup(t);
            if (conj) conjugate(xb, nrhs, ld);
        }
        check(SparseStatusOK, errors.messages());
    }

private:
    Lu() = default;
    void conjugate(T* xb, std::int64_t nrhs, std::int64_t ld) const {
        if constexpr (std::is_same_v<T, std::complex<double>>)
            for (std::int64_t j = 0; j < nrhs; ++j)
                for (std::int64_t i = 0; i < n_; ++i) xb[j * ld + i] = std::conj(xb[j * ld + i]);
    }
    void release() {
        if (owned_) SparseCleanup(f_);
        owned_ = false;
    }

    typename Ty::Factor f_{};
    bool owned_ = false;
    int n_ = 0;
    bool transposed_ = false;  // factored from a CSR (the structure's transpose flag)
};

}  // namespace sd::sparse_solve
