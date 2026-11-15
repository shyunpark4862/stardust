// Pieces shared by the function implementations.
#pragma once

#include "linalg/batch.hpp"
#include "linalg/errors.hpp"
#include "linalg/input.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pack.hpp"
#include "linalg/types.hpp"
#include "linalg/lapack.hpp"
#include "bind/pyutil.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>

namespace sd::linalg {

namespace nb = nanobind;

// Runs f(std::type_identity<T>) with T = cplx or double.
template <class F>
nb::object with_type(bool complex, F&& f) {
    if (complex) return f(std::type_identity<cplx>{});
    return f(std::type_identity<double>{});
}

// Failure codes of batch bodies (docs/design/linalg.md §2.4). Within one
// batch element the arguments are checked in order; a matrix's singularity
// right after that matrix.
constexpr int kNonFinite = 1;   // NaN or inf in the first matrix argument
constexpr int kSingular = 2;    // exact zero pivot / zero on a triangular diagonal
constexpr int kNonFiniteB = 3;  // NaN or inf in the right-hand side
constexpr int kNotPD = 4;       // Cholesky: not positive definite
constexpr int kNonFiniteU = 5;  // lu_solve: NaN or inf in 'u'
constexpr int kBadPerm = 6;     // lu_solve: 'perm' is not a permutation
constexpr int kNoConvergence = 7;     // the SVD did not converge
constexpr int kNoConvergenceEig = 8;  // an eigenvalue algorithm did not converge
constexpr int kNonFinite2 = 9;        // NaN or inf in the argument named `arg2`
constexpr int kNonFinite3 = 10;       // NaN or inf in the argument named `arg3`
constexpr int kCloseEigenvalues = 11; // Sylvester: a and -b share an eigenvalue (docs/design/linalg.md §10.7)

// Turns a batch failure into the Python exception. `arg` names the first
// matrix argument; the right-hand side is always 'b'. `arg2`, `arg3` name the
// arguments of kNonFinite2 / kNonFinite3, for functions with more inputs.
[[noreturn]] inline void raise(const Batch& b, const Failure& f, const char* func, const char* arg,
                               const char* arg2 = nullptr, const char* arg3 = nullptr) {
    auto msg = [&](const std::string& what) { return failure_message(b, f.index, (std::string(func) + "(): " + what).c_str()); };
    switch (f.code) {
        case kNonFinite: throw nb::value_error(msg(std::string("'") + arg + "' contains NaN or infinity").c_str());
        case kNonFinite2: throw nb::value_error(msg(std::string("'") + arg2 + "' contains NaN or infinity").c_str());
        case kNonFinite3: throw nb::value_error(msg(std::string("'") + arg3 + "' contains NaN or infinity").c_str());
        case kNonFiniteB: throw nb::value_error(msg("'b' contains NaN or infinity").c_str());
        case kNonFiniteU: throw nb::value_error(msg("'u' contains NaN or infinity").c_str());
        case kBadPerm: throw nb::value_error(msg("'perm' is not a permutation of 0..n-1").c_str());
        case kNotPD: throw LinAlgError(msg("matrix is not positive definite"));
        case kNoConvergence: throw LinAlgError(msg("SVD did not converge"));
        case kNoConvergenceEig: throw LinAlgError(msg("eigenvalue algorithm did not converge"));
        case kCloseEigenvalues: throw LinAlgError(msg("a and -b have a common or close eigenvalue"));
        default: throw LinAlgError(msg("singular matrix"));
    }
}

// The right-hand side of a solve: a 1-D b is one vector (M,), anything else a
// stack of matrices (..., M, K) (docs/api/linalg.md §1.3). Checks M against the matrix size n.
struct Rhs {
    bool vec;
    lint k;            // number of columns (1 for a vector)
    int core;          // 1 or 2 trailing core axes
    std::int64_t rs, cs;  // byte strides of rows and columns
};

inline Rhs rhs_of(const Input& y, lint n, const char* func, const char* aname, bool square = true) {
    const bool vec = y.ndim() == 1;
    const lint rows = vec ? y.shape(0) : y.shape(-2);
    if (rows != n) {
        const std::string what = square ? "is " + std::to_string(n) + "x" + std::to_string(n)
                                        : "has " + std::to_string(n) + " rows";
        throw nb::value_error((std::string(func) + "(): '" + aname + "' " + what + " but 'b' has " +
                               std::to_string(rows) + " rows")
                                  .c_str());
    }
    return {vec, vec ? 1 : y.shape(-1), vec ? 1 : 2, vec ? y.stride(0) : y.stride(-2), vec ? 0 : y.stride(-1)};
}

// ---- 16-byte alignment (docs/design/linalg.md §6.4) ----------------------
// Accelerate's LAPACK can round differently when an array is not 16-byte
// aligned (real svd, schur and eigh gave other bits for a batch element whose
// output slot sat at an odd multiple of 8 bytes). So every array LAPACK or
// BLAS gets is 16-byte aligned, and a matrix gives the same bits wherever it
// sits in a batch.

inline bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// n doubles rounded up to a whole number of 16-byte units, for sub-arrays of scratch.
inline std::size_t pad2(std::int64_t n) { return static_cast<std::size_t>((n + 1) & ~std::int64_t(1)); }

// An output slot handed to LAPACK/BLAS: the slot itself when it is 16-byte
// aligned, else scratch memory that is copied to the slot when this object goes
// out of scope. Everything written for the slot goes through get().
template <class T>
class LapackSlot {
public:
    LapackSlot(T* slot, std::int64_t count, Scratch& s, int scratch_slot)
        : slot_(slot), n_(count), p_(aligned16(slot) ? slot : s.get<T>(static_cast<std::size_t>(count), scratch_slot)) {}
    ~LapackSlot() {
        if (p_ != slot_) std::copy(p_, p_ + n_, slot_);
    }
    LapackSlot(const LapackSlot&) = delete;
    LapackSlot& operator=(const LapackSlot&) = delete;
    T* get() const { return p_; }

private:
    T* slot_;
    std::int64_t n_;
    T* p_;
};

// An n x n triangle operand that LAPACK only reads (solve_triangular's a,
// cholesky_solve's l, lu_solve's l and u). When the input already is
// column-major with T elements, LAPACK reads it in place and only the triangle
// is checked (docs/design/linalg.md §5.5); otherwise it is packed into `buf`.
// The arithmetic, and so the result, is the same either way. Returns false on NaN/inf.
template <class T>
bool triangle_operand(const Input& x, std::int64_t off, lint n, bool lower, bool with_diag, T* buf, const T*& a,
                      lint& lda) {
    constexpr std::int64_t sz = sizeof(T);
    const std::byte* p = x.data() + off;
    // in place only when 16-byte aligned, every column too (see aligned16)
    if (x.dtype() == dtype_of<T>() && x.stride(-2) == sz && x.stride(-1) % 16 == 0 &&
        x.stride(-1) / sz >= std::max<lint>(n, 1) && aligned16(p)) {
        a = reinterpret_cast<const T*>(p);
        lda = x.stride(-1) / sz;
        return triangle_finite(a, n, lda, lower, with_diag);
    }
    a = buf;
    lda = std::max<lint>(n, 1);
    return pack_triangle(x.dtype(), p, n, x.stride(-2), x.stride(-1), buf, lda, lower, with_diag, true);
}

// The output of a solve: a vector per batch element or a column-major matrix.
inline Out alloc_rhs(const Batch& b, const Rhs& r, lint n, DType dt) {
    return r.vec ? alloc_vectors(b.shape(), n, dt) : alloc_matrices(b.shape(), n, r.k, dt);
}

// Packs the square matrix of `x` at byte offset `off` into scratch and factors
// it with getrf. Returns 0, kNonFinite or kSingular; `a` and `ipiv` point into scratch.
template <class T>
int lu_factor(const Input& x, std::int64_t off, lint n, Scratch& s, T*& a, lint*& ipiv) {
    a = s.get<T>(static_cast<std::size_t>(n * n));
    ipiv = s.get_ints(static_cast<std::size_t>(n));
    if (!pack(x.dtype(), x.data() + off, n, n, x.stride(-2), x.stride(-1), a, n, true)) return kNonFinite;
    if (n == 0) return 0;
    return lapack::getrf(n, n, a, n, ipiv) > 0 ? kSingular : 0;
}

// The inverse from an LU factorization, written to `r` (n x n, column-major):
// getrs against the identity (docs/design/linalg.md §10.2), so inv(a) == solve(a, I) bit for bit.
template <class T>
void lu_inverse(const T* lu, const lint* ipiv, lint n, T* r) {
    if (n == 0) return;
    for (lint k = 0; k < n * n; ++k) r[k] = T(0);
    for (lint k = 0; k < n; ++k) r[k * n + k] = T(1);
    lapack::getrs('N', n, n, lu, n, ipiv, r, n);
}

template <class T>
T conj_if_complex(T v) {
    if constexpr (std::is_same_v<T, cplx>) return std::conj(v);
    else return v;
}

// mode='reduced' | 'complete' (docs/api/linalg.md §1.1); true for 'complete'.
// Shared by qr, qr_pivoted and svd.
inline bool parse_mode(nb::handle o, const char* func) {
    // A string option: the frame's check_choice format (API.md §2.1).
    if (!PyUnicode_Check(o.ptr()))
        throw nb::type_error(
            (std::string(func) + "(): 'mode' must be 'reduced' or 'complete', not " + py_type_name(o)).c_str());
    const std::string s = nb::cast<std::string>(o);
    if (s == "reduced") return false;
    if (s == "complete") return true;
    throw nb::value_error(
        (std::string(func) + "(): 'mode' must be 'reduced' or 'complete', not " + nb::repr(o).c_str()).c_str());
}

}  // namespace sd::linalg
