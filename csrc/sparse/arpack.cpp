// The eigensolvers of sd.sparse (docs/design/sparse.md §3.8): ARPACK's implicitly restarted
// Lanczos (dsaupd, dseupd: eigsh) and Arnoldi (dnaupd, dneupd and znaupd, zneupd: eigs and the
// complex eigsh) in its order of operations, as scipy 1.18.1 runs it (decision SP83).
//
//   _eigsh(A, M, inv, v0, draw, /, *, k, ncv, which, tol, max_iter, mode, sigma, vectors)
//   _eigs(A, M, inv, v0, draw, /, *, k, ncv, which, tol, max_iter, mode, sigmar, sigmai, vectors)
//
// The routines are ARPACK's (arpack-ng, with scipy's patch of dnaup2), one C++ function per
// Fortran subroutine with its name: getv0, saitr/naitr (the Lanczos/Arnoldi steps with DGKS
// re-orthogonalization), seigt/neigh (the small eigenproblem), sgets/ngets (the shifts), sconv/
// nconv, sapps/napps (the implicit shifts), saup2/naup2 (the restart loop), seupd/neupd (the Ritz
// vectors). Reverse communication is replaced by direct calls of the operators, in the order
// scipy's wrapper (scipy/sparse/linalg/_eigen/arpack/arpack.py, `iterate`) serves the requests:
// `op1` for ido = 1 (OP x given B x), `opfull` for the start vector's OP x, `bx` for ido = 2.
//
// An operator (A, M, inv) is a CSR tuple, a Python callable or None (sparse/krylov.hpp); inv is
// the Python layer's M^-1, (A - sigma M)^-1 or the user's OPinv. v0 is the start vector (the
// Python layer draws it when the user gives none: decision SP84) and draw() returns the random
// vectors ARPACK asks for when an invariant subspace is found.
//
// Vector work is the library's (krylov.hpp): inner products sd.linalg.vecdot's sums, norms
// sd.linalg.vector_norm's, V * w and V^H x row by row with every element's terms in column
// order. All are split by size alone, so the result is the same bit for bit for any thread count
// (API.md §3). The small dense problems are LAPACK's (Accelerate): dsteqr, dlahqr, dtrevc,
// dtrsen, dgeqr2 and their complex forms, as ARPACK calls them.
#include "linalg/errors.hpp"
#include "linalg/lapack.hpp"
#include "sparse/krylov.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using namespace krylov;
using linalg::lint;
using std::int64_t;
namespace lapack = linalg::lapack;

constexpr double kEpsMach = DBL_EPSILON / 2;  // dlamch('E')
constexpr double kUlp = DBL_EPSILON;          // dlamch('P')
constexpr double kSafmin = DBL_MIN;           // dlamch('S')

inline double eps23() {
    static const double v = std::pow(kEpsMach, 2.0 / 3.0);
    return v;
}

// LAPACK's dlapy2: sqrt(x^2 + y^2) without needless overflow.
inline double lapy2(double x, double y) {
    if (std::isnan(x)) return x;
    if (std::isnan(y)) return y;
    const double xa = std::fabs(x), ya = std::fabs(y);
    const double w = std::max(xa, ya), z = std::min(xa, ya);
    if (z == 0 || w > DBL_MAX) return w;
    const double q = z / w;
    return w * std::sqrt(1 + q * q);
}

template <class T>
inline double cabs_(T a) {
    if constexpr (kComplex<T>) return lapy2(a.real(), a.imag());
    else return std::fabs(a);
}
template <class T>
inline double abs1(T a) {  // zabs1: |re| + |im|
    if constexpr (kComplex<T>) return std::fabs(a.real()) + std::fabs(a.imag());
    else return std::fabs(a);
}

// acc + c v (numpy's complex product, fused for real)
template <class T>
inline T madd_(T c, T v, T acc) {
    if constexpr (kComplex<T>) return add(acc, ops::cmul(c, v));
    else return std::fma(c, v, acc);
}

template <class T>
void vcopy(const T* x, T* y, int64_t n) {
    if (n > 0 && x != y) std::memcpy(y, x, static_cast<std::size_t>(n) * sizeof(T));
}

// y[l] = V[:, l]^H x for l < j (dgemv 'T', zgemv 'C')
template <class T>
void gemv_t(const T* V, int64_t n, int64_t j, const T* x, T* y) {
    for (int64_t l = 0; l < j; ++l) y[l] = dot(V + l * n, x, n);
}

// y = (zero ? 0 : y) + sum_l coef[l] V[:, l], every element's terms in column order
template <class T>
void gemv_n(const T* V, int64_t n, int64_t j, const T* coef, T* y, bool zero) {
    for_chunks(
        n,
        [&](int64_t, int64_t lo, int64_t hi) {
            // blocks of rows short enough that y's block stays in registers and L1 across columns
            constexpr int64_t kRows = 256;
            for (int64_t b0 = lo; b0 < hi; b0 += kRows) {
                const int64_t b1 = std::min(hi, b0 + kRows);
                if (zero)
                    for (int64_t i = b0; i < b1; ++i) y[i] = T{};
                int64_t l = 0;
                for (; l + 4 <= j; l += 4) {  // four columns per pass over y (each element's order kept)
                    const T c0 = coef[l], c1 = coef[l + 1], c2 = coef[l + 2], c3 = coef[l + 3];
                    const T *v0 = V + l * n, *v1 = v0 + n, *v2 = v1 + n, *v3 = v2 + n;
                    for (int64_t i = b0; i < b1; ++i)
                        y[i] = madd_(c3, v3[i], madd_(c2, v2[i], madd_(c1, v1[i], madd_(c0, v0[i], y[i]))));
                }
                for (; l < j; ++l) {
                    const T c = coef[l];
                    const T* v = V + l * n;
                    for (int64_t i = b0; i < b1; ++i) y[i] = madd_(c, v[i], y[i]);
                }
            }
        },
        kChunk, n * std::max<int64_t>(j, 1)
    );
}

// y -= V[:, 0:j] h
template <class T>
void orth_sub(const T* V, int64_t n, int64_t j, const T* h, T* y, Vec<T>& tmp) {
    tmp.resize(static_cast<std::size_t>(std::max<int64_t>(j, 1)));
    for (int64_t l = 0; l < j; ++l) tmp[l] = neg(h[l]);
    gemv_n(V, n, j, tmp.data(), y, false);
}

template <class T>
double bnorm(const T* r, const T* br, int64_t n) {
    const T d = dot(r, br, n);
    if constexpr (kComplex<T>) return std::sqrt(lapy2(d.real(), d.imag()));
    else return std::sqrt(std::fabs(d));
}

// C (n x ncols, column-major) := C H_1 ... H_k for the reflectors of geqr2 in a (lda, k columns of
// length m = ncols): dorm2r / zunm2r ('Right', 'Notranspose'), row by row.
template <class T>
void apply_q_right(T* C, int64_t n, int64_t ncols, int64_t k, const T* a, int64_t lda, const T* tau) {
    for (int64_t i = 0; i < k; ++i) {
        // v = (0, ..., 0, 1, a(i+1:m, i))
        const int64_t len = ncols - i;
        std::vector<T> v(static_cast<std::size_t>(len));
        v[0] = T(1);
        for (int64_t r = 1; r < len; ++r) v[r] = a[(i + r) + i * lda];
        const T t = tau[i];
        if (is_zero(t)) continue;
        T* c = C + i * n;  // columns i .. ncols-1
        for_chunks(
            n,
            [&](int64_t, int64_t lo, int64_t hi) {
                std::vector<T> w(static_cast<std::size_t>(hi - lo), T{});
                for (int64_t l = 0; l < len; ++l) {
                    const T vl = v[l];
                    const T* cl = c + l * n;
                    for (int64_t r = lo; r < hi; ++r) w[r - lo] = madd_(vl, cl[r], w[r - lo]);
                }
                for (int64_t l = 0; l < len; ++l) {
                    const T coef = neg(mul(t, conj(v[l])));
                    T* cl = c + l * n;
                    for (int64_t r = lo; r < hi; ++r) cl[r] = madd_(w[r - lo], coef, cl[r]);
                }
            },
            kChunk, n * len
        );
    }
}

// ---------------------------------------------------------------- the problem (scipy's iterate)

template <class T>
struct Problem {
    int mode = 1;
    bool sym = true;  // dsaupd's modes (eigsh) or dnaupd's (eigs)
    bool gen = false;  // bmat = 'G'
    Op<T> A, M, inv;
    double sigma = 0;  // the Cayley shift
    nb::object draw;
    int64_t n = 0, nopx = 0;
    Vec<T> t1, t2;

    void apply(const Op<T>& op, const T* x, T* y) { op.apply(x, y); }

    // ido = 1: y = OP x, with B x at bx
    void op1(T* x, const T* bx, T* y) {
        ++nopx;
        if (sym) {
            switch (mode) {
                case 1: A.apply(x, y); return;
                case 2:
                    A.apply(x, t1.data());
                    vcopy(t1.data(), x, n);
                    inv.apply(x, y);
                    return;
                case 5:
                    A.apply(x, t1.data());
                    for (int64_t i = 0; i < n; ++i) t2[i] = add(t1[i], mul(T(sigma), bx[i]));
                    inv.apply(t2.data(), y);
                    return;
                default: inv.apply(bx, y); return;
            }
        }
        if (mode == 1) A.apply(x, y);
        else if (mode == 2) {
            A.apply(x, t1.data());
            inv.apply(t1.data(), y);
        } else inv.apply(bx, y);
    }

    // the start vector's y = OP x
    void opfull(const T* x, T* y) {
        ++nopx;
        if (mode == 1) {
            A.apply(x, y);
            return;
        }
        if (sym && (mode == 2 || mode == 4)) {
            A.apply(x, t1.data());
            inv.apply(t1.data(), y);
            return;
        }
        if (sym && mode == 5) {
            A.apply(x, t1.data());
            if (gen) {
                M.apply(x, t2.data());
                for (int64_t i = 0; i < n; ++i) t2[i] = add(t1[i], mul(T(sigma), t2[i]));
            } else {
                for (int64_t i = 0; i < n; ++i) t2[i] = add(t1[i], mul(T(sigma), x[i]));
            }
            inv.apply(t2.data(), y);
            return;
        }
        if (!sym && mode == 2) {
            A.apply(x, t1.data());
            inv.apply(t1.data(), y);
            return;
        }
        // shift-invert (sym 3, nonsym 3 and 4)
        if (gen) {
            M.apply(x, t1.data());
            inv.apply(t1.data(), y);
        } else inv.apply(x, y);
    }

    // ido = 2: y = B x
    void bx(const T* x, T* y) {
        if (sym && mode == 4) A.apply(x, y);
        else M.apply(x, y);
    }

    void random(T* r) {
        nb::gil_scoped_acquire gil;
        nb::object out = draw();
        if (!is_ndarray(out)) throw nb::type_error("the start vector's generator did not return an sd.ndarray");
        const NDArray& o = as_ndarray(out);
        if (o.dtype != dtype_of<T> || o.ndim != 1 || o.shape[0] != n || !o.c_contiguous)
            throw nb::value_error("the start vector's generator returned the wrong array");
        std::memcpy(r, o.data, static_cast<std::size_t>(n) * sizeof(T));
    }

    bool native() const { return A.native() && M.native() && inv.native(); }
    int64_t work() const { return A.work() + M.work() + inv.work(); }
};

// ---------------------------------------------------------------- the shared state

template <class T>
struct State {
    Problem<T>& P;
    int64_t n, ncv;
    Vec<T> V, resid, workd, tmp;
    double rnorm = 0;
    explicit State(Problem<T>& p, int64_t ncv_)
        : P(p), n(p.n), ncv(ncv_), V(static_cast<std::size_t>(p.n * ncv_)), resid(p.n), workd(3 * p.n) {}
    T* v(int64_t j) { return V.data() + (j - 1) * n; }  // column j, 1-based
};

// dgetv0 / zgetv0: the start vector r (itry = 1: OP v0 for v0 = resid; else a random one),
// B-orthogonalized against v(:, 1:j-1). Returns ierr (-1: r is numerically in the span of V).
template <class T>
int getv0(State<T>& S, int itry, bool initv, int64_t j) {
    Problem<T>& P = S.P;
    const int64_t n = S.n;
    T* resid = S.resid.data();
    T* wd = S.workd.data();
    if (!initv) P.random(resid);
    if (itry == 1) {
        vcopy(resid, wd, n);
        P.opfull(wd, wd + n);
        vcopy(wd + n, resid, n);
    } else if (itry > 1 && P.gen) {
        vcopy(resid, wd + n, n);
    }
    // the B-norm of the start vector
    if (P.gen) P.bx(wd + n, wd);
    else vcopy(resid, wd, n);
    double rnorm0 = P.gen ? bnorm(resid, wd, n) : nrm2(resid, n);
    S.rnorm = rnorm0;
    if (j == 1) return 0;
    int iter = 0;
    while (true) {
        gemv_t(S.V.data(), n, j - 1, wd, wd + n);
        orth_sub(S.V.data(), n, j - 1, wd + n, resid, S.tmp);
        if (P.gen) {
            vcopy(resid, wd + n, n);
            P.bx(wd + n, wd);
        } else vcopy(resid, wd, n);
        S.rnorm = P.gen ? bnorm(resid, wd, n) : nrm2(resid, n);
        if (S.rnorm > 0.717 * rnorm0) return 0;
        ++iter;
        if (iter <= 5) {
            rnorm0 = S.rnorm;
            continue;
        }
        std::fill(S.resid.begin(), S.resid.end(), T{});
        S.rnorm = 0;
        return -1;
    }
}

// v_j = r / rnorm, p_j = p / rnorm (dscal, or dlascl's careful scaling for a tiny rnorm)
template <class T>
void scale_step(T* vj, T* pj, int64_t n, double rnorm) {
    if (rnorm >= kSafmin) {
        const T t(1.0 / rnorm);
        scal(t, vj, n);
        scal(t, pj, n);
    } else {
        each(n, [&](int64_t i) {
            vj[i] = dvd(vj[i], T(rnorm));
            pj[i] = dvd(pj[i], T(rnorm));
        });
    }
}

// ================================================================ symmetric (dsaupd, dseupd)

enum class Which { LM, SM, LA, SA, BE, LR, SR, LI, SI };

Which parse_which(const std::string& w) {
    if (w == "LM") return Which::LM;
    if (w == "SM") return Which::SM;
    if (w == "LA") return Which::LA;
    if (w == "SA") return Which::SA;
    if (w == "BE") return Which::BE;
    if (w == "LR") return Which::LR;
    if (w == "SR") return Which::SR;
    if (w == "LI") return Which::LI;
    if (w == "SI") return Which::SI;
    throw nb::value_error("unknown 'which'");
}

// dsortr: x1 sorted ('SA' decreasing algebraic, 'SM' decreasing magnitude, 'LA' increasing
// algebraic, 'LM' increasing magnitude), x2 permuted alike; ARPACK's shell sort.
void sortr(Which which, bool apply, int64_t n, double* x1, double* x2) {
    auto before = [which](double a, double b) {  // swap when this holds for (x1[j], x1[j + gap])
        switch (which) {
            case Which::SA: return a < b;
            case Which::SM: return std::fabs(a) < std::fabs(b);
            case Which::LA: return a > b;
            default: return std::fabs(a) > std::fabs(b);  // LM
        }
    };
    for (int64_t gap = n / 2; gap != 0; gap /= 2) {
        for (int64_t i = gap; i <= n - 1; ++i) {
            for (int64_t j = i - gap; j >= 0; j -= gap) {
                if (!before(x1[j], x1[j + gap])) break;
                std::swap(x1[j], x1[j + gap]);
                if (apply) std::swap(x2[j], x2[j + gap]);
            }
        }
    }
}

// dsesrt: x sorted ('LA' increasing / 'SA' decreasing algebraic), the columns of a (na rows)
// permuted alike.
void sesrt(Which which, bool apply, int64_t n, double* x, int64_t na, double* a, int64_t lda) {
    for (int64_t gap = n / 2; gap != 0; gap /= 2) {
        for (int64_t i = gap; i <= n - 1; ++i) {
            for (int64_t j = i - gap; j >= 0; j -= gap) {
                const bool sw = which == Which::SA ? x[j] < x[j + gap] : x[j] > x[j + gap];
                if (!sw) break;
                std::swap(x[j], x[j + gap]);
                if (apply)
                    for (int64_t r = 0; r < na; ++r) std::swap(a[r + j * lda], a[r + (j + gap) * lda]);
            }
        }
    }
}

struct Sym {
    State<double> S;
    int64_t nev, ncv, k0;
    Which which;
    double tol;
    int64_t mxiter;
    std::vector<double> h1, h2;  // H(:, 1) (off-diagonal, h1[0] the dead spot) and H(:, 2) (diagonal)
    std::vector<double> ritz, bounds, q, rz, bd;
    int64_t nconv = 0, iter = 0, np = 0;
    int info = 0;

    Sym(Problem<double>& p, int64_t k, int64_t ncv_, Which w, double tol_, int64_t maxiter)
        : S(p, ncv_),
          nev(k),
          ncv(ncv_),
          k0(k),
          which(w),
          tol(tol_),
          mxiter(maxiter),
          h1(ncv_),
          h2(ncv_),
          ritz(ncv_),
          bounds(ncv_),
          q(static_cast<std::size_t>(ncv_ * ncv_)),
          rz(ncv_),
          bd(ncv_) {
        S.P.t1.resize(p.n);
        S.P.t2.resize(p.n);
    }

    // dsaitr: extend the Lanczos factorization from k to k + np steps. Returns info (> 0: the
    // factorization could not be extended past info steps).
    int64_t saitr(int64_t k, int64_t np_) {
        Problem<double>& P = S.P;
        const int64_t n = S.n;
        double* resid = S.resid.data();
        double* wd = S.workd.data();
        double* pj = wd;          // ipj
        double* rj = wd + n;      // irj
        double* vjw = wd + 2 * n;  // ivj
        int64_t j = k + 1;
        bool rstart = false;
        while (true) {
            if (!(S.rnorm > 0)) {
                int itry = 1;
                while (true) {
                    rstart = true;
                    const int ierr = getv0(S, itry, false, j);
                    if (ierr < 0) {
                        ++itry;
                        if (itry <= 3) continue;
                        return j - 1;
                    }
                    break;
                }
            }
            double* vj = S.v(j);
            vcopy(resid, vj, n);
            scale_step(vj, pj, n, S.rnorm);
            // step 3: r = OP v_j
            vcopy(vj, vjw, n);
            P.op1(vjw, pj, rj);
            vcopy(rj, resid, n);
            if (P.mode != 2) {
                if (P.gen) P.bx(rj, pj);
                else vcopy(resid, pj, n);
            }
            double wnorm;
            if (P.mode == 2) wnorm = bnorm(resid, vjw, n);
            else if (P.gen) wnorm = bnorm(resid, pj, n);
            else wnorm = nrm2(resid, n);
            gemv_t(S.V.data(), n, j, P.mode != 2 ? pj : vjw, rj);
            orth_sub(S.V.data(), n, j, rj, resid, S.tmp);
            h2[j - 1] = rj[j - 1];
            h1[j - 1] = (j == 1 || rstart) ? 0.0 : S.rnorm;
            if (P.gen) {
                vcopy(resid, rj, n);
                P.bx(rj, pj);
            } else vcopy(resid, pj, n);
            S.rnorm = P.gen ? bnorm(resid, pj, n) : nrm2(resid, n);
            if (!(S.rnorm > 0.717 * wnorm)) {
                int it = 0;
                while (true) {
                    gemv_t(S.V.data(), n, j, pj, rj);
                    orth_sub(S.V.data(), n, j, rj, resid, S.tmp);
                    if (j == 1 || rstart) h1[j - 1] = 0;
                    h2[j - 1] = h2[j - 1] + rj[j - 1];
                    if (P.gen) {
                        vcopy(resid, rj, n);
                        P.bx(rj, pj);
                    } else vcopy(resid, pj, n);
                    const double rnorm1 = P.gen ? bnorm(resid, pj, n) : nrm2(resid, n);
                    if (rnorm1 > 0.717 * S.rnorm) {
                        S.rnorm = rnorm1;
                        break;
                    }
                    S.rnorm = rnorm1;
                    ++it;
                    if (it <= 1) continue;
                    std::fill(S.resid.begin(), S.resid.end(), 0.0);
                    S.rnorm = 0;
                    break;
                }
            }
            rstart = false;
            if (h1[j - 1] < 0) {
                h1[j - 1] = -h1[j - 1];
                if (j < k + np_) scal(-1.0, S.v(j + 1), n);
                else scal(-1.0, resid, n);
            }
            ++j;
            if (j > k + np_) return 0;
        }
    }

    // dseigt: the Ritz values of H(1:kp) (ascending) and their bounds rnorm |last component|.
    int seigt(int64_t kp) {
        std::vector<double> e(kp > 1 ? kp - 1 : 1), z(static_cast<std::size_t>(kp * kp)),
            work(std::max<int64_t>(1, 2 * kp - 2));
        for (int64_t i = 0; i < kp; ++i) ritz[i] = h2[i];
        for (int64_t i = 0; i + 1 < kp; ++i) e[i] = h1[i + 1];
        if (lapack::steqr(kp, ritz.data(), e.data(), z.data(), kp, work.data()) != 0) return -1;
        for (int64_t i = 0; i < kp; ++i) bounds[i] = S.rnorm * std::fabs(z[(kp - 1) + i * kp]);
        return 0;
    }

    // dsgets: sort the Ritz values so the wanted kev are last; the np shifts first (largest bounds
    // first when shifting).
    void sgets(bool ishift, int64_t kev, int64_t np_, double* r, double* b) const {
        if (which == Which::BE) {
            sortr(Which::LA, true, kev + np_, r, b);
            const int64_t kevd2 = kev / 2;
            if (kev > 1) {
                const int64_t m = std::min(kevd2, np_), off = std::max(kevd2, np_);
                for (int64_t i = 0; i < m; ++i) {
                    std::swap(r[i], r[off + i]);
                    std::swap(b[i], b[off + i]);
                }
            }
        } else {
            sortr(which, true, kev + np_, r, b);
        }
        if (ishift && np_ > 0) sortr(Which::SM, true, np_, b, r);
    }

    int64_t sconv(int64_t m, const double* r, const double* b) const {
        int64_t c = 0;
        for (int64_t i = 0; i < m; ++i)
            if (b[i] <= tol * std::max(eps23(), std::fabs(r[i]))) ++c;
        return c;
    }

    // dsapps: apply np shifts by implicit QR on the tridiagonal H, update V and the residual.
    void sapps(int64_t kev, int64_t np_, const double* shift) {
        const int64_t n = S.n, kp = kev + np_;
        auto H1 = [this](int64_t i) -> double& { return h1[i - 1]; };
        auto H2 = [this](int64_t i) -> double& { return h2[i - 1]; };
        auto Q = [this](int64_t i, int64_t j) -> double& { return q[(i - 1) + (j - 1) * ncv]; };
        std::fill(q.begin(), q.end(), 0.0);
        for (int64_t i = 1; i <= kp; ++i) Q(i, i) = 1;
        if (np_ == 0) return;
        int64_t itop = 1;
        for (int64_t jj = 1; jj <= np_; ++jj) {
            int64_t istart = itop;
            while (true) {
                int64_t iend = kp;
                for (int64_t i = istart; i <= kp - 1; ++i) {
                    const double big = std::fabs(H2(i)) + std::fabs(H2(i + 1));
                    if (H1(i + 1) <= kEpsMach * big) {
                        H1(i + 1) = 0;
                        iend = i;
                        break;
                    }
                }
                if (istart < iend) {
                    double f = H2(istart) - shift[jj - 1];
                    double g = H1(istart + 1);
                    Rot rot = lartg(f, g);
                    double c = rot.c, s = rot.s.real(), r = rot.r.real();
                    double a1 = c * H2(istart) + s * H1(istart + 1);
                    double a2 = c * H1(istart + 1) + s * H2(istart + 1);
                    double a4 = c * H2(istart + 1) - s * H1(istart + 1);
                    double a3 = c * H1(istart + 1) - s * H2(istart);
                    H2(istart) = c * a1 + s * a2;
                    H2(istart + 1) = c * a4 - s * a3;
                    H1(istart + 1) = c * a3 + s * a4;
                    for (int64_t j = 1; j <= std::min(istart + jj, kp); ++j) {
                        a1 = c * Q(j, istart) + s * Q(j, istart + 1);
                        Q(j, istart + 1) = -s * Q(j, istart) + c * Q(j, istart + 1);
                        Q(j, istart) = a1;
                    }
                    for (int64_t i = istart + 1; i <= iend - 1; ++i) {
                        f = H1(i);
                        g = s * H1(i + 1);
                        H1(i + 1) = c * H1(i + 1);
                        rot = lartg(f, g);
                        c = rot.c;
                        s = rot.s.real();
                        r = rot.r.real();
                        if (r < 0) {
                            r = -r;
                            c = -c;
                            s = -s;
                        }
                        H1(i) = r;
                        a1 = c * H2(i) + s * H1(i + 1);
                        a2 = c * H1(i + 1) + s * H2(i + 1);
                        a3 = c * H1(i + 1) - s * H2(i);
                        a4 = c * H2(i + 1) - s * H1(i + 1);
                        H2(i) = c * a1 + s * a2;
                        H2(i + 1) = c * a4 - s * a3;
                        H1(i + 1) = c * a3 + s * a4;
                        for (int64_t j = 1; j <= std::min(i + jj, kp); ++j) {
                            a1 = c * Q(j, i) + s * Q(j, i + 1);
                            Q(j, i + 1) = -s * Q(j, i) + c * Q(j, i + 1);
                            Q(j, i) = a1;
                        }
                    }
                }
                istart = iend + 1;
                if (H1(iend) < 0) {
                    H1(iend) = -H1(iend);
                    for (int64_t j = 1; j <= kp; ++j) Q(j, iend) = -Q(j, iend);
                }
                if (iend < kp) continue;
                break;
            }
            for (int64_t i = itop; i <= kp - 1; ++i) {
                if (H1(i + 1) > 0) break;
                ++itop;
            }
        }
        for (int64_t i = itop; i <= kp - 1; ++i) {
            const double big = std::fabs(H2(i)) + std::fabs(H2(i + 1));
            if (H1(i + 1) <= kEpsMach * big) H1(i + 1) = 0;
        }
        update_basis(n, kev, kp, H1(kev + 1) > 0, H1(kev + 1), &Q(1, 1));
    }

    // V(:, 1:kev) = V Q(:, 1:kev), V(:, kev+1) = V Q(:, kev+1), r = Q(kp, kev) r + beta V(:, kev+1)
    void update_basis(int64_t n, int64_t kev, int64_t kp, bool use_beta, double beta, const double* Qd) {
        double* wd = S.workd.data();
        auto Qcol = [&](int64_t j) { return Qd + (j - 1) * ncv; };
        if (use_beta) gemv_n(S.V.data(), n, kp, Qcol(kev + 1), wd + n, true);
        for (int64_t i = 1; i <= kev; ++i) {
            gemv_n(S.V.data(), n, kp - i + 1, Qcol(kev - i + 1), wd, true);
            vcopy(wd, S.v(kp - i + 1), n);
        }
        for (int64_t j = 1; j <= kev; ++j) vcopy(S.v(kp - kev + j), S.v(j), n);
        if (use_beta) vcopy(wd + n, S.v(kev + 1), n);
        scal(Qcol(kev)[kp - 1], S.resid.data(), n);
        if (use_beta) axpy(beta, S.v(kev + 1), S.resid.data(), n);
    }

    // dsaup2: the implicitly restarted Lanczos loop.
    void saup2() {
        Problem<double>& P = S.P;
        const int64_t n = S.n;
        const int64_t nev0 = nev, np0 = ncv - nev, kplusp = ncv;
        np = np0;
        nconv = 0;
        iter = 0;
        getv0(S, 1, true, 1);
        if (S.rnorm == 0) {
            info = -9;
            return;
        }
        int64_t r = saitr(0, nev0);
        if (r > 0) {
            np = r;
            mxiter = iter;
            info = -9999;
            return;
        }
        while (true) {
            ++iter;
            r = saitr(nev, np);
            if (r > 0) {
                np = r;
                mxiter = iter;
                info = -9999;
                return;
            }
            if (seigt(kplusp) != 0) {
                info = -8;
                return;
            }
            std::copy(ritz.begin(), ritz.begin() + kplusp, rz.begin());
            std::copy(bounds.begin(), bounds.begin() + kplusp, bd.begin());
            nev = nev0;
            np = np0;
            sgets(true, nev, np, ritz.data(), bounds.data());
            nconv = sconv(nev, ritz.data() + np, bounds.data() + np);
            const int64_t nptemp = np;
            for (int64_t j = 0; j < nptemp; ++j)
                if (bounds[j] == 0) {
                    --np;
                    ++nev;
                }
            if (nconv >= nev0 || iter > mxiter || np == 0) {
                if (which == Which::BE) {
                    sortr(Which::SA, true, kplusp, ritz.data(), bounds.data());
                    const int64_t nevd2 = nev0 / 2, nevm2 = nev0 - nevd2;
                    if (nev > 1) {
                        np = kplusp - nev0;
                        const int64_t m = std::min(nevd2, np);
                        const int64_t off = std::max(kplusp - nevd2 + 1, kplusp - np + 1) - 1;
                        for (int64_t i = 0; i < m; ++i) {
                            std::swap(ritz[nevm2 + i], ritz[off + i]);
                            std::swap(bounds[nevm2 + i], bounds[off + i]);
                        }
                    }
                } else {
                    Which wp = Which::LA;
                    if (which == Which::LM) wp = Which::SM;
                    if (which == Which::SM) wp = Which::LM;
                    if (which == Which::LA) wp = Which::SA;
                    if (which == Which::SA) wp = Which::LA;
                    sortr(wp, true, kplusp, ritz.data(), bounds.data());
                }
                for (int64_t j = 0; j < nev0; ++j) bounds[j] = bounds[j] / std::max(eps23(), std::fabs(ritz[j]));
                sortr(Which::LA, true, nev0, bounds.data(), ritz.data());
                for (int64_t j = 0; j < nev0; ++j) bounds[j] = bounds[j] * std::max(eps23(), std::fabs(ritz[j]));
                if (which == Which::BE) sortr(Which::LA, true, nconv, ritz.data(), bounds.data());
                else sortr(which, true, nconv, ritz.data(), bounds.data());
                h1[0] = S.rnorm;
                if (iter > mxiter && nconv < nev) info = 1;
                if (np == 0 && nconv < nev0) info = 2;
                np = nconv;
                break;
            } else if (nconv < nev) {
                const int64_t nevbef = nev;
                nev = nev + std::min(nconv, np / 2);
                if (nev == 1 && kplusp >= 6) nev = kplusp / 2;
                else if (nev == 1 && kplusp > 2) nev = 2;
                np = kplusp - nev;
                if (nevbef < nev) sgets(true, nev, np, ritz.data(), bounds.data());
            }
            sapps(nev, np, ritz.data());
            double* wd = S.workd.data();
            if (P.gen) {
                vcopy(S.resid.data(), wd + n, n);
                P.bx(wd + n, wd);
                S.rnorm = bnorm(S.resid.data(), wd, n);
            } else {
                vcopy(S.resid.data(), wd, n);
                S.rnorm = nrm2(S.resid.data(), n);
            }
        }
        mxiter = iter;
        nev = nconv;
        if (info == 2) info = 3;  // dsaupd
    }

    // dseupd (howmny 'A'): the converged Ritz values d and, when rvec, vectors z (n x nconv).
    // Returns info (0, or -8 / -17 as ARPACK).
    int seupd(bool rvec, std::vector<double>& d, std::vector<double>& z) {
        Problem<double>& P = S.P;
        const int64_t n = S.n;
        d.clear();
        z.clear();
        if (nconv == 0) return 0;
        const std::string type = P.mode <= 2 ? "REGULR" : P.mode == 3 ? "SHIFTI" : P.mode == 4 ? "BUCKLE" : "CAYLEY";
        const double sigma = P.sigma;
        std::vector<double> ihd(ncv), ihb(ncv), iqm(static_cast<std::size_t>(ncv * ncv)), iw(2 * ncv),
            work(std::max<int64_t>(1, 2 * ncv - 2));
        d.assign(static_cast<std::size_t>(nconv), 0.0);
        if (rvec) {
            bool reord = false;
            std::vector<double> idx(ncv);
            std::vector<char> select(ncv, 0);
            for (int64_t j = 0; j < ncv; ++j) idx[j] = static_cast<double>(j + 1);
            std::vector<double> irz = rz;
            sgets(false, k0, ncv - k0, irz.data(), idx.data());
            int64_t numcnv = 0;
            for (int64_t j = 1; j <= ncv; ++j) {
                const double temp1 = std::max(eps23(), std::fabs(irz[ncv - j]));
                const int64_t jj = static_cast<int64_t>(idx[ncv - j]);
                if (numcnv < nconv && bd[jj - 1] <= tol * temp1) {
                    select[jj - 1] = 1;
                    ++numcnv;
                    if (jj > nconv) reord = true;
                }
            }
            if (numcnv != nconv) return -17;
            for (int64_t i = 0; i + 1 < ncv; ++i) ihb[i] = h1[i + 1];
            for (int64_t i = 0; i < ncv; ++i) ihd[i] = h2[i];
            if (lapack::steqr(ncv, ihd.data(), ihb.data(), iqm.data(), ncv, work.data()) != 0) return -8;
            if (reord) {
                int64_t left = 1, right = ncv;
                if (ncv != 1) {
                    while (true) {
                        if (select[left - 1]) ++left;
                        else if (!select[right - 1]) --right;
                        else {
                            std::swap(ihd[left - 1], ihd[right - 1]);
                            for (int64_t r = 0; r < ncv; ++r)
                                std::swap(iqm[r + (left - 1) * ncv], iqm[r + (right - 1) * ncv]);
                            ++left;
                            --right;
                        }
                        if (!(left < right)) break;
                    }
                }
            }
            std::copy(ihd.begin(), ihd.begin() + nconv, d.begin());
        } else {
            std::copy(ritz.begin(), ritz.begin() + nconv, d.begin());
            std::copy(ritz.begin(), ritz.end(), ihd.begin());
        }
        if (type == "REGULR") {
            if (rvec) sesrt(Which::LA, rvec, nconv, d.data(), ncv, iqm.data(), ncv);
        } else {
            std::copy(ihd.begin(), ihd.end(), iw.begin());
            for (int64_t k = 0; k < ncv; ++k) {
                if (type == "SHIFTI") ihd[k] = 1.0 / ihd[k] + sigma;
                else if (type == "BUCKLE") ihd[k] = sigma * ihd[k] / (ihd[k] - 1.0);
                else ihd[k] = sigma * (ihd[k] + 1.0) / (ihd[k] - 1.0);
            }
            std::copy(ihd.begin(), ihd.begin() + nconv, d.begin());
            sortr(Which::LA, true, nconv, ihd.data(), iw.data());
            if (rvec) {
                sesrt(Which::LA, rvec, nconv, d.data(), ncv, iqm.data(), ncv);
            } else {
                std::vector<double> b(bounds);
                sortr(Which::LA, true, nconv, d.data(), b.data());
            }
        }
        if (!rvec) return 0;
        // the Ritz vectors: V Q(:, 1:nconv) through the QR of Q's first nconv columns
        std::vector<double> tau(ncv), w2(ncv);
        lapack::geqr2(ncv, nconv, iqm.data(), ncv, tau.data(), w2.data());
        apply_q_right(S.V.data(), n, ncv, nconv, iqm.data(), ncv, tau.data());
        if (type != "REGULR") {
            // the last row of the eigenvector matrix (in factored form): Q^T e_ncv
            std::vector<double> e(ncv, 0.0);
            e[ncv - 1] = 1;
            for (int64_t i = 0; i < nconv; ++i) {
                double s = e[i];
                for (int64_t r = i + 1; r < ncv; ++r) s += iqm[r + i * ncv] * e[r];
                s *= tau[i];
                e[i] -= s;
                for (int64_t r = i + 1; r < ncv; ++r) e[r] -= s * iqm[r + i * ncv];
            }
            std::vector<double> pw(nconv);
            for (int64_t k = 0; k < nconv; ++k) {
                if (type == "BUCKLE") pw[k] = e[k] / (iw[k] - 1.0);
                else pw[k] = e[k] / iw[k];
            }
            // z = V(:, 1:nconv) + resid pw^T (the purification, dger)
            const double* resid = S.resid.data();
            for (int64_t j = 0; j < nconv; ++j) {
                double* zj = S.v(j + 1);
                const double t = pw[j];
                each(n, [&](int64_t i) { zj[i] = std::fma(resid[i], t, zj[i]); });
            }
        }
        z.assign(S.V.begin(), S.V.begin() + n * nconv);
        return 0;
    }
};

// ================================================================ nonsymmetric (dnaupd, znaupd)

// dnaitr / znaitr: extend the Arnoldi factorization from k to k + np steps (H: ldh x ncv).
template <class T>
int64_t naitr(State<T>& S, std::vector<T>& h, int64_t ldh, int64_t k, int64_t np_) {
    Problem<T>& P = S.P;
    const int64_t n = S.n;
    auto H = [&](int64_t i, int64_t j) -> T& { return h[(i - 1) + (j - 1) * ldh]; };
    const double smlnum = kSafmin * (static_cast<double>(n) / kUlp);
    T* resid = S.resid.data();
    T* wd = S.workd.data();
    T* pj = wd;
    T* rj = wd + n;
    T* vjw = wd + 2 * n;
    int64_t j = k + 1;
    double betaj = 0;
    while (true) {
        betaj = S.rnorm;
        if (!(S.rnorm > 0)) {
            betaj = 0;
            int itry = 1;
            while (true) {
                const int ierr = getv0(S, itry, false, j);
                if (ierr < 0) {
                    ++itry;
                    if (itry <= 3) continue;
                    return j - 1;
                }
                break;
            }
        }
        T* vj = S.v(j);
        vcopy(resid, vj, n);
        scale_step(vj, pj, n, S.rnorm);
        vcopy(vj, vjw, n);
        P.op1(vjw, pj, rj);
        vcopy(rj, resid, n);
        if (P.gen) P.bx(rj, pj);
        else vcopy(resid, pj, n);
        const double wnorm = P.gen ? bnorm(resid, pj, n) : nrm2(resid, n);
        T* hj = &H(1, j);
        gemv_t(S.V.data(), n, j, pj, hj);
        orth_sub(S.V.data(), n, j, hj, resid, S.tmp);
        if (j > 1) H(j, j - 1) = T(betaj);
        if (P.gen) {
            vcopy(resid, rj, n);
            P.bx(rj, pj);
        } else vcopy(resid, pj, n);
        S.rnorm = P.gen ? bnorm(resid, pj, n) : nrm2(resid, n);
        if (!(S.rnorm > 0.717 * wnorm)) {
            int it = 0;
            while (true) {
                gemv_t(S.V.data(), n, j, pj, rj);
                orth_sub(S.V.data(), n, j, rj, resid, S.tmp);
                for (int64_t l = 0; l < j; ++l) hj[l] = add(hj[l], rj[l]);
                if (P.gen) {
                    vcopy(resid, rj, n);
                    P.bx(rj, pj);
                } else vcopy(resid, pj, n);
                const double rnorm1 = P.gen ? bnorm(resid, pj, n) : nrm2(resid, n);
                if (rnorm1 > 0.717 * S.rnorm) {
                    S.rnorm = rnorm1;
                    break;
                }
                S.rnorm = rnorm1;
                ++it;
                if (it <= 1) continue;
                std::fill(S.resid.begin(), S.resid.end(), T{});
                S.rnorm = 0;
                break;
            }
        }
        ++j;
        if (j > k + np_) {
            for (int64_t i = std::max<int64_t>(1, k); i <= k + np_ - 1; ++i) {
                double tst1 = cabs_(H(i, i)) + cabs_(H(i + 1, i + 1));
                if (tst1 == 0) tst1 = lapack::lanhs1(k + np_, h.data(), ldh);
                if (cabs_(H(i + 1, i)) <= std::max(kUlp * tst1, smlnum)) H(i + 1, i) = T{};
            }
            return 0;
        }
    }
}

// V(:, 1:kev) = V Q(:, 1:kev) and the residual update after napps (H(kev+1, kev) = beta).
template <class T>
void napps_basis(State<T>& S, const std::vector<T>& q, int64_t ldq, int64_t kev, int64_t kp, T beta, bool use_beta) {
    const int64_t n = S.n;
    T* wd = S.workd.data();
    auto Qcol = [&](int64_t j) { return q.data() + (j - 1) * ldq; };
    if (use_beta) gemv_n(S.V.data(), n, kp, Qcol(kev + 1), wd + n, true);
    for (int64_t i = 1; i <= kev; ++i) {
        gemv_n(S.V.data(), n, kp - i + 1, Qcol(kev - i + 1), wd, true);
        vcopy(wd, S.v(kp - i + 1), n);
    }
    for (int64_t j = 1; j <= kev; ++j) vcopy(S.v(kp - kev + j), S.v(j), n);
    if (use_beta) vcopy(wd + n, S.v(kev + 1), n);
    scal(Qcol(kev)[kp - 1], S.resid.data(), n);
    if (use_beta) axpy(beta, S.v(kev + 1), S.resid.data(), n);
}

// dsortc: (xr, xi) sorted ('LM' increasing / 'SM' decreasing magnitude, 'LR' increasing / 'SR'
// decreasing real part, 'LI' increasing / 'SI' decreasing |imaginary part|), y permuted alike.
void sortc(Which which, bool apply, int64_t n, double* xr, double* xi, double* y) {
    auto sw = [&](int64_t a, int64_t b) {
        switch (which) {
            case Which::LM: return lapy2(xr[a], xi[a]) > lapy2(xr[b], xi[b]);
            case Which::SM: return lapy2(xr[a], xi[a]) < lapy2(xr[b], xi[b]);
            case Which::LR: return xr[a] > xr[b];
            case Which::SR: return xr[a] < xr[b];
            case Which::LI: return std::fabs(xi[a]) > std::fabs(xi[b]);
            default: return std::fabs(xi[a]) < std::fabs(xi[b]);  // SI
        }
    };
    for (int64_t gap = n / 2; gap != 0; gap /= 2) {
        for (int64_t i = gap; i <= n - 1; ++i) {
            for (int64_t j = i - gap; j >= 0; j -= gap) {
                if (!sw(j, j + gap)) break;
                std::swap(xr[j], xr[j + gap]);
                std::swap(xi[j], xi[j + gap]);
                if (apply) std::swap(y[j], y[j + gap]);
            }
        }
    }
}

// zsortc: x sorted ('LM' increasing / 'SM' decreasing magnitude, 'LR' / 'SR' real parts, 'LI' /
// 'SI' imaginary parts, increasing / decreasing), y permuted alike.
void zsortc(Which which, bool apply, int64_t n, complex128* x, complex128* y) {
    auto sw = [&](int64_t a, int64_t b) {
        switch (which) {
            case Which::LM: return lapy2(x[a].real(), x[a].imag()) > lapy2(x[b].real(), x[b].imag());
            case Which::SM: return lapy2(x[a].real(), x[a].imag()) < lapy2(x[b].real(), x[b].imag());
            case Which::LR: return x[a].real() > x[b].real();
            case Which::SR: return x[a].real() < x[b].real();
            case Which::LI: return x[a].imag() > x[b].imag();
            default: return x[a].imag() < x[b].imag();  // SI
        }
    };
    for (int64_t gap = n / 2; gap != 0; gap /= 2) {
        for (int64_t i = gap; i <= n - 1; ++i) {
            for (int64_t j = i - gap; j >= 0; j -= gap) {
                if (!sw(j, j + gap)) break;
                std::swap(x[j], x[j + gap]);
                if (apply) std::swap(y[j], y[j + gap]);
            }
        }
    }
}

struct NonsymBase {
    Which which;
    double tol;
    int64_t nev, ncv, mxiter, k0;
    int64_t nconv = 0, iter = 0, np = 0;
    int info = 0;
};

// ---- real nonsymmetric (dnaupd, dneupd)

struct Real : NonsymBase {
    State<double> S;
    std::vector<double> h, q, ritzr, ritzi, bounds, rr, ri, rb, workl;

    Real(Problem<double>& p, int64_t k, int64_t ncv_, Which w, double tol_, int64_t maxiter) : S(p, ncv_) {
        which = w;
        tol = tol_;
        nev = k0 = k;
        ncv = ncv_;
        mxiter = maxiter;
        const std::size_t c = static_cast<std::size_t>(ncv_);
        h.assign(c * c, 0.0);
        q.assign(c * c, 0.0);
        ritzr.assign(c, 0.0);
        ritzi.assign(c, 0.0);
        bounds.assign(c, 0.0);
        rr.assign(c, 0.0);
        ri.assign(c, 0.0);
        rb.assign(c, 0.0);
        workl.assign(c * c + 3 * c, 0.0);
        S.P.t1.resize(p.n);
        S.P.t2.resize(p.n);
    }
    double& H(int64_t i, int64_t j) { return h[(i - 1) + (j - 1) * ncv]; }
    double& Q(int64_t i, int64_t j) { return q[(i - 1) + (j - 1) * ncv]; }

    // dneigh
    int neigh(int64_t kp) {
        std::vector<double> t(static_cast<std::size_t>(kp * kp)), work(3 * kp);
        for (int64_t j = 0; j < kp; ++j)
            for (int64_t i = 0; i < kp; ++i) t[i + j * kp] = h[i + j * ncv];
        // dlahqr on the last row of the Schur vectors (z = e_n^T); the row is updated on its own, so
        // the full Z's last row is the same (Accelerate's dlahqr wants ldz >= n)
        std::vector<double> zz(static_cast<std::size_t>(kp * kp), 0.0);
        for (int64_t i = 0; i < kp; ++i) zz[i + i * kp] = 1;
        if (lapack::lahqr(true, true, kp, 1, kp, t.data(), kp, ritzr.data(), ritzi.data(), 1, kp, zz.data(), kp) != 0)
            return -1;
        for (int64_t j = 0; j < kp; ++j) bounds[j] = zz[(kp - 1) + j * kp];
        lint m = 0;
        if (lapack::trevc('A', nullptr, kp, t.data(), kp, q.data(), ncv, kp, m, work.data()) != 0) return -1;
        bool iconj = false;
        for (int64_t i = 0; i < kp; ++i) {
            double* qi = q.data() + i * ncv;
            if (std::fabs(ritzi[i]) <= 0) {
                const double temp = nrm2(qi, kp);
                scal(1.0 / temp, qi, kp);
            } else if (!iconj) {
                const double temp = lapy2(nrm2(qi, kp), nrm2(qi + ncv, kp));
                scal(1.0 / temp, qi, kp);
                scal(1.0 / temp, qi + ncv, kp);
                iconj = true;
            } else iconj = false;
        }
        std::vector<double> w(kp);
        for (int64_t i = 0; i < kp; ++i) {
            double s = 0;
            for (int64_t r = 0; r < kp; ++r) s = std::fma(q[r + i * ncv], bounds[r], s);
            w[i] = s;
        }
        iconj = false;
        for (int64_t i = 0; i < kp; ++i) {
            if (std::fabs(ritzi[i]) <= 0) bounds[i] = S.rnorm * std::fabs(w[i]);
            else if (!iconj) {
                bounds[i] = S.rnorm * lapy2(w[i], w[i + 1]);
                bounds[i + 1] = bounds[i];
                iconj = true;
            } else iconj = false;
        }
        return 0;
    }

    // dngets
    void ngets(bool ishift, int64_t& kev, int64_t& np_, double* xr, double* xi, double* b) const {
        const int64_t m = kev + np_;
        switch (which) {
            case Which::LM: sortc(Which::LR, true, m, xr, xi, b); break;
            case Which::SM: sortc(Which::SR, true, m, xr, xi, b); break;
            case Which::LR: sortc(Which::LM, true, m, xr, xi, b); break;
            case Which::SR: sortc(Which::SM, true, m, xr, xi, b); break;
            case Which::LI: sortc(Which::LM, true, m, xr, xi, b); break;
            default: sortc(Which::SM, true, m, xr, xi, b); break;
        }
        sortc(which, true, m, xr, xi, b);
        if (np_ >= 1 && (xr[np_] - xr[np_ - 1]) == 0 && (xi[np_] + xi[np_ - 1]) == 0) {
            --np_;
            ++kev;
        }
        if (ishift) sortc(Which::SR, true, np_, b, xr, xi);
    }

    int64_t nconv_(int64_t m, const double* xr, const double* xi, const double* b) const {
        int64_t c = 0;
        for (int64_t i = 0; i < m; ++i)
            if (b[i] <= tol * std::max(eps23(), lapy2(xr[i], xi[i]))) ++c;
        return c;
    }

    // dnapps
    void napps(int64_t& kev, int64_t np_, const double* shiftr, const double* shifti) {
        const int64_t n = S.n, kp = kev + np_;
        const double smlnum = kSafmin * (static_cast<double>(n) / kUlp);
        std::fill(q.begin(), q.end(), 0.0);
        for (int64_t i = 1; i <= kp; ++i) Q(i, i) = 1;
        if (np_ == 0) return;
        std::vector<double> wk(static_cast<std::size_t>(kp) + 3);
        bool cconj = false;
        for (int64_t jj = 1; jj <= np_; ++jj) {
            const double sigmar = shiftr[jj - 1], sigmai = shifti[jj - 1];
            if (cconj) {
                cconj = false;
                continue;
            } else if (jj < np_ && std::fabs(sigmai) > 0) {
                cconj = true;
            } else if (jj == np_ && std::fabs(sigmai) > 0) {
                ++kev;
                continue;
            }
            int64_t istart = 1;
            while (true) {
                int64_t iend = kp;
                for (int64_t i = istart; i <= kp - 1; ++i) {
                    double tst1 = std::fabs(H(i, i)) + std::fabs(H(i + 1, i + 1));
                    if (tst1 == 0) tst1 = lapack::lanhs1(kp - jj + 1, h.data(), ncv);
                    if (std::fabs(H(i + 1, i)) <= std::max(kUlp * tst1, smlnum)) {
                        iend = i;
                        H(i + 1, i) = 0;
                        break;
                    }
                }
                const bool skip = istart == iend || (istart + 1 == iend && std::fabs(sigmai) > 0);
                if (!skip) {
                    const double h11 = H(istart, istart), h21 = H(istart + 1, istart);
                    if (std::fabs(sigmai) <= 0) {
                        double f = h11 - sigmar, g = h21;
                        for (int64_t i = istart; i <= iend - 1; ++i) {
                            const Rot rot = lartg(f, g);
                            double c = rot.c, s = rot.s.real(), r = rot.r.real();
                            if (i > istart) {
                                if (r < 0) {
                                    r = -r;
                                    c = -c;
                                    s = -s;
                                }
                                H(i, i - 1) = r;
                                H(i + 1, i - 1) = 0;
                            }
                            for (int64_t j = i; j <= kp; ++j) {
                                const double t = c * H(i, j) + s * H(i + 1, j);
                                H(i + 1, j) = -s * H(i, j) + c * H(i + 1, j);
                                H(i, j) = t;
                            }
                            for (int64_t j = 1; j <= std::min(i + 2, iend); ++j) {
                                const double t = c * H(j, i) + s * H(j, i + 1);
                                H(j, i + 1) = -s * H(j, i) + c * H(j, i + 1);
                                H(j, i) = t;
                            }
                            for (int64_t j = 1; j <= std::min(i + jj, kp); ++j) {
                                const double t = c * Q(j, i) + s * Q(j, i + 1);
                                Q(j, i + 1) = -s * Q(j, i) + c * Q(j, i + 1);
                                Q(j, i) = t;
                            }
                            if (i < iend - 1) {
                                f = H(i + 1, i);
                                g = H(i + 2, i);
                            }
                        }
                    } else {
                        const double h12 = H(istart, istart + 1), h22 = H(istart + 1, istart + 1),
                                     h32 = H(istart + 2, istart + 1);
                        const double s = 2.0 * sigmar, t = lapy2(sigmar, sigmai);
                        double u[3];
                        u[0] = (h11 * (h11 - s) + t * t) / h21 + h12;
                        u[1] = h11 + h22 - s;
                        u[2] = h32;
                        for (int64_t i = istart; i <= iend - 1; ++i) {
                            const int64_t nr = std::min<int64_t>(3, iend - i + 1);
                            double tau = 0;
                            lapack::larfg(nr, &u[0], &u[1], &tau);
                            if (i > istart) {
                                H(i, i - 1) = u[0];
                                H(i + 1, i - 1) = 0;
                                if (i < iend - 1) H(i + 2, i - 1) = 0;
                            }
                            u[0] = 1;
                            lapack::larf('L', nr, kp - i + 1, u, tau, &H(i, i), ncv, wk.data());
                            const int64_t ir = std::min(i + 3, iend);
                            lapack::larf('R', ir, nr, u, tau, &H(1, i), ncv, wk.data());
                            lapack::larf('R', kp, nr, u, tau, &Q(1, i), ncv, wk.data());
                            if (i < iend - 1) {
                                u[0] = H(i + 1, i);
                                u[1] = H(i + 2, i);
                                if (i < iend - 2) u[2] = H(i + 3, i);
                            }
                        }
                    }
                }
                istart = iend + 1;
                if (iend < kp) continue;
                break;
            }
        }
        for (int64_t i = 1; i <= kev; ++i) {
            double tst1 = std::fabs(H(i, i)) + std::fabs(H(i + 1, i + 1));
            if (tst1 == 0) tst1 = lapack::lanhs1(kev, h.data(), ncv);
            if (std::fabs(H(i + 1, i)) <= std::max(kUlp * tst1, smlnum)) H(i + 1, i) = 0;
        }
        napps_basis(S, q, ncv, kev, kp, H(kev + 1, kev), H(kev + 1, kev) != 0);
    }

    // dnaup2 (with scipy's patch keeping nev <= kplusp - 2)
    void naup2() {
        Problem<double>& P = S.P;
        const int64_t n = S.n;
        const int64_t nev0 = nev, np0 = ncv - nev, kplusp = ncv;
        np = np0;
        nconv = 0;
        iter = 0;
        int64_t numcnv = nev0;
        getv0(S, 1, true, 1);
        if (S.rnorm == 0) {
            info = -9;
            return;
        }
        int64_t r = naitr(S, h, ncv, 0, nev);
        if (r > 0) {
            np = r;
            mxiter = iter;
            info = -9999;
            return;
        }
        while (true) {
            ++iter;
            np = kplusp - nev;
            r = naitr(S, h, ncv, nev, np);
            if (r > 0) {
                np = r;
                mxiter = iter;
                info = -9999;
                return;
            }
            if (neigh(kplusp) != 0) {
                info = -8;
                return;
            }
            std::copy(ritzr.begin(), ritzr.end(), rr.begin());
            std::copy(ritzi.begin(), ritzi.end(), ri.begin());
            std::copy(bounds.begin(), bounds.end(), rb.begin());
            nev = nev0;
            np = np0;
            numcnv = nev;
            ngets(true, nev, np, ritzr.data(), ritzi.data(), bounds.data());
            if (nev == nev0 + 1) numcnv = nev0 + 1;
            nconv = nconv_(nev, ritzr.data() + np, ritzi.data() + np, bounds.data() + np);
            const int64_t nptemp = np;
            for (int64_t j = 0; j < nptemp; ++j)
                if (bounds[j] == 0) {
                    --np;
                    ++nev;
                }
            if (nconv >= numcnv || iter > mxiter || np == 0) {
                H(3, 1) = S.rnorm;
                Which wp = Which::LM;
                switch (which) {
                    case Which::LM: wp = Which::SR; break;
                    case Which::SM: wp = Which::LR; break;
                    case Which::LR: wp = Which::SM; break;
                    case Which::SR: wp = Which::LM; break;
                    case Which::LI: wp = Which::SM; break;
                    default: wp = Which::LM; break;
                }
                sortc(wp, true, kplusp, ritzr.data(), ritzi.data(), bounds.data());
                switch (which) {
                    case Which::LM: wp = Which::SM; break;
                    case Which::SM: wp = Which::LM; break;
                    case Which::LR: wp = Which::SR; break;
                    case Which::SR: wp = Which::LR; break;
                    case Which::LI: wp = Which::SI; break;
                    default: wp = Which::LI; break;
                }
                sortc(wp, true, kplusp, ritzr.data(), ritzi.data(), bounds.data());
                for (int64_t j = 0; j < numcnv; ++j)
                    bounds[j] = bounds[j] / std::max(eps23(), lapy2(ritzr[j], ritzi[j]));
                sortc(Which::LR, true, numcnv, bounds.data(), ritzr.data(), ritzi.data());
                for (int64_t j = 0; j < numcnv; ++j)
                    bounds[j] = bounds[j] * std::max(eps23(), lapy2(ritzr[j], ritzi[j]));
                sortc(which, true, nconv, ritzr.data(), ritzi.data(), bounds.data());
                if (iter > mxiter && nconv < numcnv) info = 1;
                if (np == 0 && nconv < numcnv) info = 2;
                np = nconv;
                break;
            } else if (nconv < numcnv) {
                const int64_t nevbef = nev;
                nev = nev + std::min(nconv, np / 2);
                if (nev == 1 && kplusp >= 6) nev = kplusp / 2;
                else if (nev == 1 && kplusp > 3) nev = 2;
                if (nev > kplusp - 2) nev = kplusp - 2;
                np = kplusp - nev;
                if (nevbef < nev) ngets(true, nev, np, ritzr.data(), ritzi.data(), bounds.data());
            }
            napps(nev, np, ritzr.data(), ritzi.data());
            double* wd = S.workd.data();
            if (P.gen) {
                vcopy(S.resid.data(), wd + n, n);
                P.bx(wd + n, wd);
                S.rnorm = bnorm(S.resid.data(), wd, n);
            } else {
                vcopy(S.resid.data(), wd, n);
                S.rnorm = nrm2(S.resid.data(), n);
            }
        }
        mxiter = iter;
        nev = numcnv;
        if (info == 2) info = 3;  // dnaupd
    }

    // dneupd (howmny 'A'): dr, di (nconv), z (n x nconv; complex pairs as real and imaginary
    // columns). type: 'REGULR' (modes 1, 2), 'SHIFTI' (3, real sigma), 'REALPT' (3), 'IMAGPT' (4).
    int neupd(
        bool rvec,
        double sigmar,
        double sigmai,
        std::vector<double>& dr,
        std::vector<double>& di,
        std::vector<double>& z
    ) {
        Problem<double>& P = S.P;
        const int64_t n = S.n;
        dr.clear();
        di.clear();
        z.clear();
        if (nconv <= 0) return 0;
        const int mode = P.mode;
        const std::string type = mode <= 2                    ? "REGULR"
                                 : mode == 3 && sigmai == 0.0 ? "SHIFTI"
                                 : mode == 3                  ? "REALPT"
                                                              : "IMAGPT";
        const int64_t ldq = ncv;
        std::vector<double> iheigr(ncv), iheigi(ncv), uptri(static_cast<std::size_t>(ncv * ncv)),
            invsub(static_cast<std::size_t>(ncv * ncv), 0.0), workev(3 * ncv);
        H(3, 1) = 0;
        dr.assign(static_cast<std::size_t>(nconv), 0.0);
        di.assign(static_cast<std::size_t>(nconv), 0.0);
        if (rvec) {
            bool reord = false;
            std::vector<double> idx(ncv);
            std::vector<__LAPACK_bool> select(ncv, 0);
            for (int64_t j = 0; j < ncv; ++j) idx[j] = static_cast<double>(j + 1);
            std::vector<double> irr = rr, iri = ri;
            int64_t nevl = k0, npl = ncv - k0;
            ngets(false, nevl, npl, irr.data(), iri.data(), idx.data());
            int64_t numcnv = 0;
            for (int64_t j = 1; j <= ncv; ++j) {
                const double temp1 = std::max(eps23(), lapy2(irr[ncv - j], iri[ncv - j]));
                const int64_t jj = static_cast<int64_t>(idx[ncv - j]);
                if (numcnv < nconv && rb[jj - 1] <= tol * temp1) {
                    select[jj - 1] = 1;
                    ++numcnv;
                    if (jj > nconv) reord = true;
                }
            }
            if (numcnv != nconv) return -15;
            uptri = h;
            for (int64_t i = 0; i < ncv; ++i) invsub[i + i * ldq] = 1;
            if (lapack::lahqr(
                    true, true, ncv, 1, ncv, uptri.data(), ncv, iheigr.data(), iheigi.data(), 1, ncv, invsub.data(), ldq
                ) != 0)
                return -8;
            if (reord) {
                lint m = 0;
                if (lapack::trsen(
                        select.data(), ncv, uptri.data(), ncv, invsub.data(), ldq, iheigr.data(), iheigi.data(), m,
                        workev.data()
                    ) == 1)
                    return 1;
            }
            if (type == "REGULR") {
                std::copy(iheigr.begin(), iheigr.begin() + nconv, dr.begin());
                std::copy(iheigi.begin(), iheigi.begin() + nconv, di.begin());
            }
            std::vector<double> tau(ncv), w2(ncv);
            lapack::geqr2(ncv, nconv, invsub.data(), ldq, tau.data(), w2.data());
            apply_q_right(S.V.data(), n, ncv, nconv, invsub.data(), ldq, tau.data());
            z.assign(S.V.begin(), S.V.begin() + n * nconv);
            for (int64_t j = 1; j <= nconv; ++j) {
                if (invsub[(j - 1) * ldq + j - 1] < 0) {
                    for (int64_t c = 0; c < nconv; ++c) uptri[(j - 1) + c * ncv] = -uptri[(j - 1) + c * ncv];
                    for (int64_t r = 0; r < nconv; ++r) uptri[r + (j - 1) * ncv] = -uptri[r + (j - 1) * ncv];
                }
            }
            for (int64_t j = 0; j < ncv; ++j) select[j] = j < nconv ? 1 : 0;
            lint outncv = 0;
            if (lapack::trevc(
                    'S', select.data(), ncv, uptri.data(), ldq, invsub.data(), ldq, ncv, outncv, workev.data()
                ) != 0)
                return -9;
            bool iconj = false;
            for (int64_t j = 1; j <= nconv; ++j) {
                double* c = invsub.data() + (j - 1) * ldq;
                if (iheigi[j - 1] == 0) {
                    const double temp = nrm2(c, ncv);
                    scal(1.0 / temp, c, ncv);
                } else if (!iconj) {
                    const double temp = lapy2(nrm2(c, ncv), nrm2(c + ldq, ncv));
                    scal(1.0 / temp, c, ncv);
                    scal(1.0 / temp, c + ldq, ncv);
                    iconj = true;
                } else iconj = false;
            }
            // the QR of the eigenvectors of T: Z = Z Q R
            lapack::geqr2(ncv, nconv, invsub.data(), ldq, tau.data(), w2.data());
            apply_q_right(z.data(), n, nconv, nconv, invsub.data(), ldq, tau.data());
            trmm_upper(z.data(), n, nconv, invsub.data(), ldq);
        } else {
            std::copy(ritzr.begin(), ritzr.begin() + nconv, dr.begin());
            std::copy(ritzi.begin(), ritzi.begin() + nconv, di.begin());
            std::copy(ritzr.begin(), ritzr.begin() + nconv, iheigr.begin());
            std::copy(ritzi.begin(), ritzi.begin() + nconv, iheigi.begin());
        }
        if (type == "SHIFTI") {
            for (int64_t k = 0; k < ncv; ++k) {
                const double temp = lapy2(iheigr[k], iheigi[k]);
                iheigr[k] = iheigr[k] / temp / temp + sigmar;
                iheigi[k] = -iheigi[k] / temp / temp + sigmai;
            }
            std::copy(iheigr.begin(), iheigr.begin() + nconv, dr.begin());
            std::copy(iheigi.begin(), iheigi.begin() + nconv, di.begin());
        } else if (type == "REALPT" || type == "IMAGPT") {
            std::copy(iheigr.begin(), iheigr.begin() + nconv, dr.begin());
            std::copy(iheigi.begin(), iheigi.begin() + nconv, di.begin());
        }
        if (rvec && type == "SHIFTI") {
            bool iconj = false;
            for (int64_t j = 1; j <= nconv; ++j) {
                const double* c = invsub.data() + (j - 1) * ldq;
                if (iheigi[j - 1] == 0) {
                    workev[j - 1] = c[ncv - 1] / iheigr[j - 1];
                } else if (!iconj) {
                    const double temp = lapy2(iheigr[j - 1], iheigi[j - 1]);
                    workev[j - 1] = (c[ncv - 1] * iheigr[j - 1] + c[ldq + ncv - 1] * iheigi[j - 1]) / temp / temp;
                    workev[j] = (c[ldq + ncv - 1] * iheigr[j - 1] - c[ncv - 1] * iheigi[j - 1]) / temp / temp;
                    iconj = true;
                } else iconj = false;
            }
            const double* resid = S.resid.data();
            for (int64_t j = 0; j < nconv; ++j) {
                double* zj = z.data() + j * n;
                const double t = workev[j];
                each(n, [&](int64_t i) { zj[i] = std::fma(resid[i], t, zj[i]); });
            }
        }
        return 0;
    }

    // Z := Z R for the upper triangular R (dtrmm 'Right', 'Upper', 'N', 'Non-unit'), row by row.
    template <class T>
    static void trmm_upper(T* Z, int64_t n, int64_t k, const T* R, int64_t ldr) {
        for_chunks(
            n,
            [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t j = k - 1; j >= 0; --j) {
                    T* zj = Z + j * n;
                    const T rjj = R[j + j * ldr];
                    for (int64_t i = lo; i < hi; ++i) zj[i] = mul(rjj, zj[i]);
                    for (int64_t l = 0; l < j; ++l) {
                        const T c = R[l + j * ldr];
                        const T* zl = Z + l * n;
                        for (int64_t i = lo; i < hi; ++i) zj[i] = madd_(c, zl[i], zj[i]);
                    }
                }
            },
            kChunk, n * k
        );
    }
};

// ---- complex (znaupd, zneupd)

struct Cplx : NonsymBase {
    State<complex128> S;
    std::vector<complex128> h, q, ritz, bounds, rz, rb;

    Cplx(Problem<complex128>& p, int64_t k, int64_t ncv_, Which w, double tol_, int64_t maxiter) : S(p, ncv_) {
        which = w;
        tol = tol_;
        nev = k0 = k;
        ncv = ncv_;
        mxiter = maxiter;
        const std::size_t c = static_cast<std::size_t>(ncv_);
        h.assign(c * c, complex128{});
        q.assign(c * c, complex128{});
        ritz.assign(c, complex128{});
        bounds.assign(c, complex128{});
        rz.assign(c, complex128{});
        rb.assign(c, complex128{});
        S.P.t1.resize(p.n);
        S.P.t2.resize(p.n);
    }
    complex128& H(int64_t i, int64_t j) { return h[(i - 1) + (j - 1) * ncv]; }
    complex128& Q(int64_t i, int64_t j) { return q[(i - 1) + (j - 1) * ncv]; }

    // zneigh
    int neigh(int64_t kp) {
        std::vector<complex128> t(static_cast<std::size_t>(kp * kp)), work(2 * kp);
        std::vector<double> rwork(kp);
        for (int64_t j = 0; j < kp; ++j)
            for (int64_t i = 0; i < kp; ++i) t[i + j * kp] = h[i + j * ncv];
        std::fill(q.begin(), q.end(), complex128{});
        for (int64_t i = 0; i < kp; ++i) q[i + i * ncv] = 1;
        if (lapack::lahqr(true, true, kp, 1, kp, t.data(), kp, ritz.data(), 1, kp, q.data(), ncv) != 0) return -1;
        lint m = 0;
        if (lapack::trevc('B', nullptr, kp, t.data(), kp, q.data(), ncv, kp, m, work.data(), rwork.data()) != 0)
            return -1;
        for (int64_t j = 0; j < kp; ++j) {
            complex128* c = q.data() + j * ncv;
            const double temp = nrm2(c, kp);
            const double s = 1.0 / temp;
            for (int64_t i = 0; i < kp; ++i) c[i] = cscale(c[i], s);
        }
        for (int64_t j = 0; j < kp; ++j) bounds[j] = cscale(q[(kp - 1) + j * ncv], S.rnorm);
        return 0;
    }

    void ngets(bool ishift, int64_t kev, int64_t np_, complex128* x, complex128* b) const {
        zsortc(which, true, kev + np_, x, b);
        if (ishift) zsortc(Which::SM, true, np_, b, x);
    }

    int64_t nconv_(int64_t m, const complex128* x, const complex128* b) const {
        int64_t c = 0;
        for (int64_t i = 0; i < m; ++i)
            if (lapy2(b[i].real(), b[i].imag()) <= tol * std::max(eps23(), lapy2(x[i].real(), x[i].imag()))) ++c;
        return c;
    }

    // znapps
    void napps(int64_t kev, int64_t np_, const complex128* shift) {
        const int64_t n = S.n, kp = kev + np_;
        const double smlnum = kSafmin * (static_cast<double>(n) / kUlp);
        std::fill(q.begin(), q.end(), complex128{});
        for (int64_t i = 1; i <= kp; ++i) Q(i, i) = 1;
        if (np_ == 0) return;
        for (int64_t jj = 1; jj <= np_; ++jj) {
            const complex128 sigma = shift[jj - 1];
            int64_t istart = 1;
            while (true) {
                int64_t iend = kp;
                for (int64_t i = istart; i <= kp - 1; ++i) {
                    double tst1 = abs1(H(i, i)) + abs1(H(i + 1, i + 1));
                    if (tst1 == 0) tst1 = lapack::lanhs1(kp - jj + 1, h.data(), ncv);
                    if (std::fabs(H(i + 1, i).real()) <= std::max(kUlp * tst1, smlnum)) {
                        iend = i;
                        H(i + 1, i) = 0;
                        break;
                    }
                }
                if (!(istart == iend || istart > kev)) {
                    const complex128 h11 = H(istart, istart), h21 = H(istart + 1, istart);
                    complex128 f = sub(h11, sigma), g = h21;
                    for (int64_t i = istart; i <= iend - 1; ++i) {
                        const Rot rot = lartg(f, g);
                        const double c = rot.c;
                        const complex128 s = rot.s, sc = std::conj(rot.s);
                        if (i > istart) {
                            H(i, i - 1) = rot.r;
                            H(i + 1, i - 1) = 0;
                        }
                        for (int64_t j = i; j <= kp; ++j) {
                            const complex128 t = add(cscale(H(i, j), c), ops::cmul(s, H(i + 1, j)));
                            H(i + 1, j) = add(neg(ops::cmul(sc, H(i, j))), cscale(H(i + 1, j), c));
                            H(i, j) = t;
                        }
                        for (int64_t j = 1; j <= std::min(i + 2, iend); ++j) {
                            const complex128 t = add(cscale(H(j, i), c), ops::cmul(sc, H(j, i + 1)));
                            H(j, i + 1) = add(neg(ops::cmul(s, H(j, i))), cscale(H(j, i + 1), c));
                            H(j, i) = t;
                        }
                        for (int64_t j = 1; j <= std::min(i + jj, kp); ++j) {
                            const complex128 t = add(cscale(Q(j, i), c), ops::cmul(sc, Q(j, i + 1)));
                            Q(j, i + 1) = add(neg(ops::cmul(s, Q(j, i))), cscale(Q(j, i + 1), c));
                            Q(j, i) = t;
                        }
                        if (i < iend - 1) {
                            f = H(i + 1, i);
                            g = H(i + 2, i);
                        }
                    }
                }
                istart = iend + 1;
                if (iend < kp) continue;
                break;
            }
        }
        // make the sub-diagonal of H real and non-negative
        for (int64_t j = 1; j <= kev; ++j) {
            const complex128 hj = H(j + 1, j);
            if (hj.real() < 0 || hj.imag() != 0) {
                const complex128 t = cdivr(hj, lapy2(hj.real(), hj.imag()));
                const complex128 tc = std::conj(t);
                for (int64_t c = j; c <= kp; ++c) H(j + 1, c) = ops::cmul(tc, H(j + 1, c));
                for (int64_t r = 1; r <= std::min(j + 2, kp); ++r) H(r, j + 1) = ops::cmul(t, H(r, j + 1));
                for (int64_t r = 1; r <= std::min(j + np_ + 1, kp); ++r) Q(r, j + 1) = ops::cmul(t, Q(r, j + 1));
                H(j + 1, j) = complex128(H(j + 1, j).real(), 0.0);
            }
        }
        for (int64_t i = 1; i <= kev; ++i) {
            double tst1 = abs1(H(i, i)) + abs1(H(i + 1, i + 1));
            if (tst1 == 0) tst1 = lapack::lanhs1(kev, h.data(), ncv);
            if (H(i + 1, i).real() <= std::max(kUlp * tst1, smlnum)) H(i + 1, i) = 0;
        }
        napps_basis(S, q, ncv, kev, kp, H(kev + 1, kev), H(kev + 1, kev).real() > 0);
    }

    // znaup2
    void naup2() {
        Problem<complex128>& P = S.P;
        const int64_t n = S.n;
        const int64_t nev0 = nev, np0 = ncv - nev, kplusp = ncv;
        np = np0;
        nconv = 0;
        iter = 0;
        getv0(S, 1, true, 1);
        if (S.rnorm == 0) {
            info = -9;
            return;
        }
        int64_t r = naitr(S, h, ncv, 0, nev);
        if (r > 0) {
            np = r;
            mxiter = iter;
            info = -9999;
            return;
        }
        while (true) {
            ++iter;
            np = kplusp - nev;
            r = naitr(S, h, ncv, nev, np);
            if (r > 0) {
                np = r;
                mxiter = iter;
                info = -9999;
                return;
            }
            if (neigh(kplusp) != 0) {
                info = -8;
                return;
            }
            std::copy(ritz.begin(), ritz.end(), rz.begin());
            std::copy(bounds.begin(), bounds.end(), rb.begin());
            nev = nev0;
            np = np0;
            ngets(true, nev, np, ritz.data(), bounds.data());
            nconv = nconv_(nev, ritz.data() + np, bounds.data() + np);
            const int64_t nptemp = np;
            for (int64_t j = 0; j < nptemp; ++j)
                if (is_zero(bounds[j])) {
                    --np;
                    ++nev;
                }
            if (nconv >= nev0 || iter > mxiter || np == 0) {
                H(3, 1) = complex128(S.rnorm, 0.0);
                Which wp = Which::LM;
                switch (which) {
                    case Which::LM: wp = Which::SM; break;
                    case Which::SM: wp = Which::LM; break;
                    case Which::LR: wp = Which::SR; break;
                    case Which::SR: wp = Which::LR; break;
                    case Which::LI: wp = Which::SI; break;
                    default: wp = Which::LI; break;
                }
                zsortc(wp, true, kplusp, ritz.data(), bounds.data());
                for (int64_t j = 0; j < nev0; ++j)
                    bounds[j] = cdivr(bounds[j], std::max(eps23(), lapy2(ritz[j].real(), ritz[j].imag())));
                zsortc(Which::LM, true, nev0, bounds.data(), ritz.data());
                for (int64_t j = 0; j < nev0; ++j)
                    bounds[j] = cscale(bounds[j], std::max(eps23(), lapy2(ritz[j].real(), ritz[j].imag())));
                zsortc(which, true, nconv, ritz.data(), bounds.data());
                if (iter > mxiter && nconv < nev0) info = 1;
                if (np == 0 && nconv < nev0) info = 2;
                np = nconv;
                break;
            } else if (nconv < nev0) {
                const int64_t nevbef = nev;
                nev = nev + std::min(nconv, np / 2);
                if (nev == 1 && kplusp >= 6) nev = kplusp / 2;
                else if (nev == 1 && kplusp > 3) nev = 2;
                np = kplusp - nev;
                if (nevbef < nev) ngets(true, nev, np, ritz.data(), bounds.data());
            }
            napps(nev, np, ritz.data());
            complex128* wd = S.workd.data();
            if (P.gen) {
                vcopy(S.resid.data(), wd + n, n);
                P.bx(wd + n, wd);
                S.rnorm = bnorm(S.resid.data(), wd, n);
            } else {
                vcopy(S.resid.data(), wd, n);
                S.rnorm = nrm2(S.resid.data(), n);
            }
        }
        mxiter = iter;
        nev = nconv;
        if (info == 2) info = 3;  // znaupd
    }

    // zneupd (howmny 'A')
    int neupd(bool rvec, complex128 sigma, std::vector<complex128>& d, std::vector<complex128>& z) {
        Problem<complex128>& P = S.P;
        const int64_t n = S.n;
        d.clear();
        z.clear();
        if (nconv <= 0) return 0;
        const bool shifti = P.mode == 3;
        const int64_t ldq = ncv;
        std::vector<complex128> iheig(ncv), uptri, invsub(static_cast<std::size_t>(ncv * ncv), complex128{}),
            workev(3 * ncv);
        std::vector<double> rwork(ncv);
        H(3, 1) = 0;
        d.assign(static_cast<std::size_t>(nconv), complex128{});
        if (rvec) {
            bool reord = false;
            std::vector<complex128> idx(ncv);
            std::vector<__LAPACK_bool> select(ncv, 0);
            for (int64_t j = 0; j < ncv; ++j) idx[j] = static_cast<double>(j + 1);
            std::vector<complex128> irz = rz;
            ngets(false, k0, ncv - k0, irz.data(), idx.data());
            int64_t numcnv = 0;
            for (int64_t j = 1; j <= ncv; ++j) {
                const double rtemp = std::max(eps23(), lapy2(irz[ncv - j].real(), irz[ncv - j].imag()));
                const int64_t jj = static_cast<int64_t>(idx[ncv - j].real());
                if (numcnv < nconv && lapy2(rb[jj - 1].real(), rb[jj - 1].imag()) <= tol * rtemp) {
                    select[jj - 1] = 1;
                    ++numcnv;
                    if (jj > nconv) reord = true;
                }
            }
            if (numcnv != nconv) return -15;
            uptri = h;
            for (int64_t i = 0; i < ncv; ++i) invsub[i + i * ldq] = 1;
            if (lapack::lahqr(true, true, ncv, 1, ncv, uptri.data(), ncv, iheig.data(), 1, ncv, invsub.data(), ldq) !=
                0)
                return -8;
            if (reord) {
                lint m = 0;
                if (lapack::trsen(
                        select.data(), ncv, uptri.data(), ncv, invsub.data(), ldq, iheig.data(), m, workev.data()
                    ) == 1)
                    return 1;
            }
            if (!shifti) std::copy(iheig.begin(), iheig.begin() + nconv, d.begin());
            std::vector<complex128> tau(ncv), w2(ncv);
            lapack::geqr2(ncv, nconv, invsub.data(), ldq, tau.data(), w2.data());
            apply_q_right(S.V.data(), n, ncv, nconv, invsub.data(), ldq, tau.data());
            z.assign(S.V.begin(), S.V.begin() + n * nconv);
            for (int64_t j = 1; j <= nconv; ++j) {
                if (invsub[(j - 1) * ldq + j - 1].real() < 0) {
                    for (int64_t c = 0; c < nconv; ++c) uptri[(j - 1) + c * ncv] = neg(uptri[(j - 1) + c * ncv]);
                    for (int64_t r = 0; r < nconv; ++r) uptri[r + (j - 1) * ncv] = neg(uptri[r + (j - 1) * ncv]);
                }
            }
            for (int64_t j = 0; j < ncv; ++j) select[j] = j < nconv ? 1 : 0;
            lint outncv = 0;
            if (lapack::trevc(
                    'S', select.data(), ncv, uptri.data(), ldq, invsub.data(), ldq, ncv, outncv, workev.data(),
                    rwork.data()
                ) != 0)
                return -9;
            for (int64_t j = 1; j <= nconv; ++j) {
                complex128* c = invsub.data() + (j - 1) * ldq;
                const double rtemp = 1.0 / nrm2(c, ncv);
                for (int64_t i = 0; i < ncv; ++i) c[i] = cscale(c[i], rtemp);
            }
            Real::trmm_upper(z.data(), n, nconv, invsub.data(), ldq);
        } else {
            std::copy(ritz.begin(), ritz.begin() + nconv, d.begin());
            std::copy(ritz.begin(), ritz.begin() + nconv, iheig.begin());
        }
        if (shifti) {
            for (int64_t k = 0; k < nconv; ++k) d[k] = add(ops::cdiv(complex128(1.0, 0.0), iheig[k]), sigma);
            if (rvec) {
                for (int64_t j = 0; j < nconv; ++j)
                    if (!is_zero(iheig[j])) workev[j] = ops::cdiv(invsub[j * ldq + ncv - 1], iheig[j]);
                const complex128* resid = S.resid.data();
                for (int64_t j = 0; j < nconv; ++j) {
                    complex128* zj = z.data() + j * n;
                    const complex128 t = workev[j];
                    each(n, [&](int64_t i) { zj[i] = add(zj[i], ops::cmul(resid[i], t)); });
                }
            }
        }
        return 0;
    }
};

// ---------------------------------------------------------------- results

template <class T>
nb::object vector_out(const std::vector<T>& v, int64_t count) {
    const int64_t shape[] = {count};
    NDArray a = new_array(dtype_of<T>, shape);
    if (count) std::memcpy(a.data, v.data(), static_cast<std::size_t>(count) * sizeof(T));
    return wrap(std::move(a));
}

// a column-major n x k block as a C-order (n, k) array
template <class T>
nb::object matrix_out(const T* z, int64_t n, int64_t k) {
    const int64_t shape[] = {n, k};
    NDArray a = new_array(dtype_of<T>, shape);
    T* out = typed<T>(a);
    for_chunks(n, [&](int64_t, int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i)
            for (int64_t j = 0; j < k; ++j) out[i * k + j] = z[i + j * n];
    });
    return wrap(std::move(a));
}

template <class T>
Problem<T> make_problem(
    nb::handle A, nb::handle M, nb::handle inv, nb::handle draw, int64_t n, int mode, bool sym, const char* func
) {
    Problem<T> P;
    P.n = n;
    P.mode = mode;
    P.sym = sym;
    P.A = parse_op<T>(A, n, n, func);
    P.M = parse_op<T>(M, n, n, func);
    P.inv = parse_op<T>(inv, n, n, func);
    P.gen = P.M.kind != Op<T>::Identity || (sym && mode == 4);
    if (sym && mode == 2) P.gen = true;
    P.draw = nb::borrow<nb::object>(draw);
    return P;
}

struct Args {
    int64_t k, ncv, max_iter;
    std::string which;
    double tol;
    int mode;
    bool vectors;
};

nb::dict eigsh_entry(
    nb::handle A, nb::handle M, nb::handle inv, nb::handle v0_o, nb::handle draw, const Args& a, double sigma
) {
    const char* func = "eigsh";
    const NDArray& v0 = vector_arg(v0_o, func, DType::Float64);
    const int64_t n = v0.shape[0];
    Problem<double> P = make_problem<double>(A, M, inv, draw, n, a.mode, true, func);
    P.sigma = sigma;
    Sym s(P, a.k, a.ncv, parse_which(a.which), a.tol, a.max_iter);
    std::memcpy(s.S.resid.data(), v0.data, static_cast<std::size_t>(n) * sizeof(double));
    std::vector<double> d, z;
    int extract = 0;
    {
        const parallel::Unlocked unlocked(P.native() && P.work() + n * a.ncv >= kParallelThreshold);
        s.saup2();
        if (s.info >= 0 || s.info == -9999) extract = s.seupd(a.vectors, d, z);
    }
    nb::dict out;
    const int64_t m = static_cast<int64_t>(d.size());
    out["d"] = vector_out(d, m);
    out["z"] = a.vectors ? matrix_out(z.data(), n, m) : nb::none();
    out["info"] = s.info;
    out["extract"] = extract;
    out["nconv"] = s.nconv;
    out["iter"] = s.mxiter;
    out["nopx"] = P.nopx;
    return out;
}

// scipy's extract for a real matrix: complex eigenvalues and vectors from dneupd's real parts,
// the conjugate pairs put together, at most k of them in the order `which` wants.
void real_extract(
    Real& s,
    Problem<double>& P,
    int64_t k,
    double sigmar,
    double sigmai,
    const std::vector<double>& dr,
    const std::vector<double>& di,
    const std::vector<double>& zr,
    bool vectors,
    std::vector<complex128>& d,
    std::vector<complex128>& z
) {
    const int64_t n = s.S.n;
    int64_t nreturned = static_cast<int64_t>(dr.size());
    const int64_t width = k + 1;
    std::vector<double> zpad(zr);
    zpad.resize(static_cast<std::size_t>(vectors || sigmai != 0 ? n * width : 0), 0.0);
    d.assign(static_cast<std::size_t>(width), complex128{});
    for (int64_t i = 0; i < nreturned; ++i) d[i] = complex128(dr[i], di[i]);
    z.assign(vectors ? static_cast<std::size_t>(n * width) : 0, complex128{});
    auto zrc = [&](int64_t j) { return zpad.data() + j * n; };
    if (vectors)
        for (int64_t j = 0; j < nreturned; ++j)
            for (int64_t i = 0; i < n; ++i) z[i + j * n] = zrc(j)[i];
    std::vector<double> av, bv;
    auto rq = [&](int64_t a, int64_t b) {  // zr_a . (A zr_b)
        av.resize(n);
        P.A.apply(zrc(b), av.data());
        return dot(zrc(a), av.data(), n);
    };
    int64_t i = 0;
    while (i <= k) {
        if (sigmai == 0) {
            if (std::fabs(d[i].imag()) != 0) {
                if (i < k) {
                    if (vectors)
                        for (int64_t r = 0; r < n; ++r) {
                            z[r + i * n] = complex128(zrc(i)[r], zrc(i + 1)[r]);
                            z[r + (i + 1) * n] = std::conj(z[r + i * n]);
                        }
                    ++i;
                } else --nreturned;
            }
        } else {
            if (std::fabs(d[i].imag()) == 0) {
                d[i] = rq(i, i);
            } else if (i < k) {
                if (vectors)
                    for (int64_t r = 0; r < n; ++r) {
                        z[r + i * n] = complex128(zrc(i)[r], zrc(i + 1)[r]);
                        z[r + (i + 1) * n] = std::conj(z[r + i * n]);
                    }
                const double re = rq(i, i) + rq(i + 1, i + 1);
                const double im = rq(i, i + 1) - rq(i + 1, i);
                d[i] = complex128(re, im);
                d[i + 1] = std::conj(d[i]);
                ++i;
            } else --nreturned;
        }
        ++i;
    }
    if (nreturned <= k) {
        d.resize(static_cast<std::size_t>(std::max<int64_t>(nreturned, 0)));
        if (vectors) z.resize(static_cast<std::size_t>(n * std::max<int64_t>(nreturned, 0)));
        return;
    }
    // one more than k: choose by `which` (on 1 / (d - sigma) in shift-invert)
    std::vector<complex128> rd(d.begin(), d.begin() + nreturned);
    if (P.mode >= 3)
        for (auto& v : rd) v = ops::cdiv(complex128(1.0, 0.0), sub(v, complex128(sigmar, sigmai)));
    std::vector<int64_t> ind(static_cast<std::size_t>(nreturned));
    for (int64_t j = 0; j < nreturned; ++j) ind[j] = j;
    const Which w = s.which;
    auto key = [&](int64_t j) {
        if (w == Which::LR || w == Which::SR) return rd[j].real();
        if (w == Which::LI || w == Which::SI) return std::fabs(rd[j].imag());
        return std::hypot(rd[j].real(), rd[j].imag());
    };
    std::stable_sort(ind.begin(), ind.end(), [&](int64_t a, int64_t b) { return key(a) < key(b); });
    std::vector<int64_t> pick;
    if (w == Which::LR || w == Which::LM || w == Which::LI)
        for (int64_t j = 0; j < k; ++j) pick.push_back(ind[nreturned - 1 - j]);
    else
        for (int64_t j = 0; j < k; ++j) pick.push_back(ind[j]);
    std::vector<complex128> d2, z2;
    for (int64_t j : pick) d2.push_back(d[j]);
    if (vectors)
        for (int64_t j : pick)
            for (int64_t r = 0; r < n; ++r) z2.push_back(z[r + j * n]);
    d = std::move(d2);
    z = std::move(z2);
}

nb::dict eigs_entry(
    nb::handle A,
    nb::handle M,
    nb::handle inv,
    nb::handle v0_o,
    nb::handle draw,
    const Args& a,
    double sigmar,
    double sigmai
) {
    const char* func = "eigs";
    const NDArray& v0 = vector_arg(v0_o, func);
    const int64_t n = v0.shape[0];
    nb::dict out;
    std::vector<complex128> d, z;
    int info = 0, extract = 0;
    int64_t nconv = 0, iter = 0, nopx = 0;
    if (v0.dtype == DType::Float64) {
        Problem<double> P = make_problem<double>(A, M, inv, draw, n, a.mode, false, func);
        Real s(P, a.k, a.ncv, parse_which(a.which), a.tol, a.max_iter);
        std::memcpy(s.S.resid.data(), v0.data, static_cast<std::size_t>(n) * sizeof(double));
        {
            const parallel::Unlocked unlocked(P.native() && P.work() + n * a.ncv >= kParallelThreshold);
            s.naup2();
            if (s.info >= 0 || s.info == -9999) {
                std::vector<double> dr, di, zr;
                extract = s.neupd(a.vectors, sigmar, sigmai, dr, di, zr);
                if (extract == 0) real_extract(s, P, a.k, sigmar, sigmai, dr, di, zr, a.vectors, d, z);
            }
        }
        info = s.info;
        nconv = s.nconv;
        iter = s.mxiter;
        nopx = P.nopx;
    } else if (v0.dtype == DType::Complex128) {
        Problem<complex128> P = make_problem<complex128>(A, M, inv, draw, n, a.mode, false, func);
        Cplx s(P, a.k, a.ncv, parse_which(a.which), a.tol, a.max_iter);
        std::memcpy(s.S.resid.data(), v0.data, static_cast<std::size_t>(n) * sizeof(complex128));
        {
            const parallel::Unlocked unlocked(P.native() && P.work() + n * a.ncv >= kParallelThreshold);
            s.naup2();
            if (s.info >= 0 || s.info == -9999) extract = s.neupd(a.vectors, complex128(sigmar, sigmai), d, z);
        }
        info = s.info;
        nconv = s.nconv;
        iter = s.mxiter;
        nopx = P.nopx;
    } else {
        throw nb::type_error("eigs(): 'v0' must be float64 or complex128");
    }
    const int64_t m = static_cast<int64_t>(d.size());
    out["d"] = vector_out(d, m);
    out["z"] = a.vectors ? matrix_out(z.data(), n, m) : nb::none();
    out["info"] = info;
    out["extract"] = extract;
    out["nconv"] = nconv;
    out["iter"] = iter;
    out["nopx"] = nopx;
    return out;
}

}  // namespace

void bind_arpack(nb::module_& m) {
    m.def(
        "_eigsh",
        [](nb::handle A, nb::handle M, nb::handle inv, nb::handle v0, nb::handle draw, int64_t k, int64_t ncv,
           const std::string& which, double tol, int64_t max_iter, int mode, double sigma, bool vectors) {
            const Args a{k, ncv, max_iter, which, tol, mode, vectors};
            return eigsh_entry(A, M, inv, v0, draw, a, sigma);
        },
        nb::arg("A").none(), nb::arg("M").none(), nb::arg("inv").none(), nb::arg("v0"), nb::arg("draw").none(),
        nb::kw_only(), nb::arg("k"), nb::arg("ncv"), nb::arg("which"), nb::arg("tol"), nb::arg("max_iter"),
        nb::arg("mode"), nb::arg("sigma"), nb::arg("vectors")
    );
    m.def(
        "_eigs",
        [](nb::handle A, nb::handle M, nb::handle inv, nb::handle v0, nb::handle draw, int64_t k, int64_t ncv,
           const std::string& which, double tol, int64_t max_iter, int mode, double sigmar, double sigmai,
           bool vectors) {
            const Args a{k, ncv, max_iter, which, tol, mode, vectors};
            return eigs_entry(A, M, inv, v0, draw, a, sigmar, sigmai);
        },
        nb::arg("A").none(), nb::arg("M").none(), nb::arg("inv").none(), nb::arg("v0"), nb::arg("draw").none(),
        nb::kw_only(), nb::arg("k"), nb::arg("ncv"), nb::arg("which"), nb::arg("tol"), nb::arg("max_iter"),
        nb::arg("mode"), nb::arg("sigma_real"), nb::arg("sigma_imag"), nb::arg("vectors")
    );
}

}  // namespace sd::sparse
