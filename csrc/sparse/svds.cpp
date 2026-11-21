// The partial SVD of sd.sparse (docs/design/sparse.md §3.8): svds by Golub-Kahan-Lanczos
// bidiagonalization with thick restarts (Baglama and Reichel 2005's augmented implicitly
// restarted Lanczos bidiagonalization, Ritz vectors; the method of R's irlba), decision SP85.
//
//   _svds(X, XH, v0, draw, /, *, k, ncv, which, tol, max_iter, vectors)
//
// X (m x n, m >= n: the Python layer passes A or A^H) and XH = X^H are operators (sparse/
// krylov.hpp: a CSR tuple or a Python callable); v0 (n) is the start vector and draw(size) gives
// the random vectors needed when the bidiagonalization breaks down (an invariant subspace or a
// rank-deficient X). With P (n x w) and Q (m x w) orthonormal and B (w x w) upper triangular,
//
//   X P = Q B,   X^H Q = P B^T + f e_w^T,
//
// B = U_B S V_B^T gives the Ritz triplets (Q U_B, S, P V_B) with residuals ||X^H u_i - s_i v_i|| =
// ||f|| |U_B(w, i)| (X v_i = s_i u_i exactly). A restart keeps the wanted kk Ritz vectors: P V_B,
// Q U_B, B = [diag(s) | ||f|| U_B(w, :)^T] (the arrow), P(:, kk + 1) = f / ||f||, and extends the
// bidiagonalization again. Every new vector is orthogonalized against all the kept ones by classical
// Gram-Schmidt, twice when the first pass cancels much of it (ARPACK's DGKS test, 0.717). B stays real for complex X (its entries are norms
// and the arrow's products of real numbers), so the small SVD is real (LAPACK dgesdd).
//
// A triplet converges when its residual is at most tol max(s_i, eps^(2/3) s_1) (s_1 the largest
// Ritz value): ARPACK's test (docs/design/sparse.md §4) on the singular triplets.
//
// Vector work is the library's (krylov.hpp), split by size alone: the same bits for any thread
// count (API.md §3).
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

// y = (zero ? 0 : y) + sum_l coef[l] V[:, l]
template <class T>
void combine(const T* V, int64_t n, int64_t j, const T* coef, T* y, bool zero) {
    gaxpy(V, n, j, coef, y, zero);
}

// x -= V[:, 0:j] (V[:, 0:j]^H x) by classical Gram-Schmidt, a second time when the first pass
// cancelled much of x (ARPACK's DGKS test: ||x|| fell below 0.717 of what it was). Returns ||x||.
template <class T>
double orthogonalize(const T* V, int64_t n, int64_t j, T* x, std::vector<T>& c) {
    double before = nrm2(x, n);
    if (j == 0) return before;
    c.resize(static_cast<std::size_t>(j));
    for (int pass = 0; pass < 2; ++pass) {
        dots(V, n, j, x, c.data());
        for (int64_t l = 0; l < j; ++l) c[l] = neg(c[l]);
        combine(V, n, j, c.data(), x, false);
        const double after = nrm2(x, n);
        if (after > 0.717 * before) return after;
        before = after;
    }
    return before;
}

template <class T>
struct Gkl {
    Op<T> X, XH;
    nb::object draw;
    int64_t m, n, k, w, max_iter;
    bool largest;
    double tol;
    std::vector<T> P, Q, f, tmp, c;
    std::vector<double> B;  // w x w, column-major
    int64_t nmv = 0, iter = 0, nconv = 0;
    int status = 0;

    T* p(int64_t j) { return P.data() + j * n; }
    T* q(int64_t j) { return Q.data() + j * m; }
    double& b(int64_t i, int64_t j) { return B[static_cast<std::size_t>(i + j * w)]; }

    void random(T* x, int64_t len) {
        nb::gil_scoped_acquire gil;
        nb::object out = draw(len);
        if (!is_ndarray(out)) throw nb::type_error("the start vector's generator did not return an sd.ndarray");
        const NDArray& o = as_ndarray(out);
        if (o.dtype != dtype_of<T> || o.ndim != 1 || o.shape[0] != len || !o.c_contiguous)
            throw nb::value_error("the start vector's generator returned the wrong array");
        std::memcpy(x, o.data, static_cast<std::size_t>(len) * sizeof(T));
    }

    // x normalized after orthogonalizing against V[:, 0:j]; a random replacement when x is
    // (numerically) in their span. Returns ||x|| before normalizing (0 for a replacement).
    double next(T* x, const T* V, int64_t len, int64_t j, double scale) {
        const double nx = orthogonalize(V, len, j, x, c);
        if (nx > kEps * scale && nx > 0) {
            scal(T(1.0 / nx), x, len);
            return nx;
        }
        for (int attempt = 0; attempt < 3; ++attempt) {
            random(x, len);
            const double nr = orthogonalize(V, len, j, x, c);
            if (nr > 0) {
                scal(T(1.0 / nr), x, len);
                return 0.0;
            }
        }
        throw linalg::LinAlgError("svds(): the bidiagonalization could not be continued");
    }

    void run(const T* v0, std::vector<double>& s_out, std::vector<T>& u_out, std::vector<T>& v_out, bool vectors) {
        P.assign(static_cast<std::size_t>(n * w), T{});
        Q.assign(static_cast<std::size_t>(m * w), T{});
        f.assign(static_cast<std::size_t>(n), T{});
        tmp.assign(static_cast<std::size_t>(std::max(m, n)), T{});
        B.assign(static_cast<std::size_t>(w * w), 0.0);
        std::memcpy(p(0), v0, static_cast<std::size_t>(n) * sizeof(T));
        {
            const double nv = nrm2(p(0), n);
            scal(T(1.0 / nv), p(0), n);
        }
        double scale = 0;  // the largest Ritz value so far (for the breakdown test)
        int64_t kk = 0;
        std::vector<double> U(static_cast<std::size_t>(w * w)), VT(static_cast<std::size_t>(w * w)), s(w), Bc;
        std::vector<int64_t> want;
        double beta = 0;
        linalg::Scratch scratch;
        while (true) {
            // Q(:, kk) = X P(:, kk) - sum_{i < kk} B(i, kk) Q(:, i)
            X.apply(p(kk), q(kk));
            ++nmv;
            if (kk > 0) {
                std::vector<T> coef(static_cast<std::size_t>(kk));
                for (int64_t i = 0; i < kk; ++i) coef[i] = T(-b(i, kk));
                combine(Q.data(), m, kk, coef.data(), q(kk), false);
            }
            b(kk, kk) = next(q(kk), Q.data(), m, kk, scale);
            for (int64_t j = kk; j < w; ++j) {
                // f = X^H Q(:, j) - alpha_j P(:, j)
                XH.apply(q(j), f.data());
                ++nmv;
                axpy(T(-b(j, j)), p(j), f.data(), n);
                if (j + 1 < w) {
                    std::memcpy(p(j + 1), f.data(), static_cast<std::size_t>(n) * sizeof(T));
                    b(j, j + 1) = next(p(j + 1), P.data(), n, j + 1, scale);
                    X.apply(p(j + 1), q(j + 1));
                    ++nmv;
                    axpy(T(-b(j, j + 1)), q(j), q(j + 1), m);
                    b(j + 1, j + 1) = next(q(j + 1), Q.data(), m, j + 1, scale);
                } else {
                    beta = orthogonalize(P.data(), n, w, f.data(), c);
                }
            }
            ++iter;
            // the small SVD (descending s)
            Bc = B;
            if (lapack::gesdd('A', w, w, Bc.data(), w, s.data(), U.data(), w, VT.data(), w, scratch) != 0)
                throw linalg::LinAlgError("svds(): the SVD of the bidiagonal matrix did not converge");
            scale = std::max(scale, s[0]);
            want.clear();
            for (int64_t i = 0; i < w; ++i) want.push_back(largest ? i : w - 1 - i);
            const double floor = std::pow(kEps, 2.0 / 3.0) * s[0];
            auto converged = [&](int64_t i) {
                return beta * std::fabs(U[(w - 1) + i * w]) <= tol * std::max(s[i], floor);
            };
            nconv = 0;
            for (int64_t t = 0; t < k; ++t)
                if (converged(want[t])) ++nconv;
            if (nconv >= k || iter >= max_iter) {
                status = nconv >= k ? 0 : 1;
                break;
            }
            // restart with the wanted kk Ritz vectors
            // ARPACK's adjustment against stagnation (dsaup2): keep more when few have converged
            kk = k + std::min(nconv, (w - k) / 2);
            if (kk == 1 && w >= 6) kk = w / 2;
            else if (kk == 1 && w > 2) kk = 2;
            kk = std::min(kk, w - 1);
            std::vector<T> coef(static_cast<std::size_t>(w));
            std::vector<T> newP(static_cast<std::size_t>(n * kk)), newQ(static_cast<std::size_t>(m * kk));
            for (int64_t t = 0; t < kk; ++t) {
                const int64_t i = want[t];
                for (int64_t l = 0; l < w; ++l) coef[l] = T(VT[i + l * w]);  // V_B(:, i)
                combine(P.data(), n, w, coef.data(), newP.data() + t * n, true);
                for (int64_t l = 0; l < w; ++l) coef[l] = T(U[l + i * w]);  // U_B(:, i)
                combine(Q.data(), m, w, coef.data(), newQ.data() + t * m, true);
            }
            std::copy(newP.begin(), newP.end(), P.begin());
            std::copy(newQ.begin(), newQ.end(), Q.begin());
            std::fill(B.begin(), B.end(), 0.0);
            for (int64_t t = 0; t < kk; ++t) {
                const int64_t i = want[t];
                b(t, t) = s[i];
                b(t, kk) = beta * U[(w - 1) + i * w];
            }
            std::memcpy(p(kk), f.data(), static_cast<std::size_t>(n) * sizeof(T));
            if (beta > kEps * scale && beta > 0) scal(T(1.0 / beta), p(kk), n);
            else {
                for (int64_t t = 0; t < kk; ++t) b(t, kk) = 0;
                next(p(kk), P.data(), n, kk, 1.0e300);  // a random direction
            }
        }
        // the k wanted triplets, ascending in s (scipy's order)
        const int64_t got = status == 0 ? k : nconv;
        std::vector<int64_t> sel;
        for (int64_t t = 0; t < k; ++t)
            if (status == 0 || converged_at(want[t], beta, U, s, w)) sel.push_back(want[t]);
        std::sort(sel.begin(), sel.end(), [&](int64_t a, int64_t b2) {
            return s[a] < s[b2] || (s[a] == s[b2] && a > b2);
        });
        (void)got;
        const int64_t r = static_cast<int64_t>(sel.size());
        s_out.resize(static_cast<std::size_t>(r));
        for (int64_t t = 0; t < r; ++t) s_out[t] = s[sel[t]];
        if (vectors) {
            u_out.assign(static_cast<std::size_t>(m * r), T{});
            v_out.assign(static_cast<std::size_t>(n * r), T{});
            std::vector<T> coef(static_cast<std::size_t>(w));
            for (int64_t t = 0; t < r; ++t) {
                const int64_t i = sel[t];
                for (int64_t l = 0; l < w; ++l) coef[l] = T(U[l + i * w]);
                combine(Q.data(), m, w, coef.data(), u_out.data() + t * m, true);
                for (int64_t l = 0; l < w; ++l) coef[l] = T(VT[i + l * w]);
                combine(P.data(), n, w, coef.data(), v_out.data() + t * n, true);
            }
        }
    }

    bool converged_at(int64_t i, double beta, const std::vector<double>& U, const std::vector<double>& s, int64_t ww) {
        const double floor = std::pow(kEps, 2.0 / 3.0) * s[0];
        return beta * std::fabs(U[(ww - 1) + i * ww]) <= tol * std::max(s[i], floor);
    }
};

template <class T>
nb::object matrix_out(const T* z, int64_t rows, int64_t k) {
    const int64_t shape[] = {rows, k};
    NDArray a = new_array(dtype_of<T>, shape);
    T* out = typed<T>(a);
    for_chunks(rows, [&](int64_t, int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i)
            for (int64_t j = 0; j < k; ++j) out[i * k + j] = z[i + j * rows];
    });
    return wrap(std::move(a));
}

template <class T>
nb::dict svds_typed(
    nb::handle X_o,
    nb::handle XH_o,
    const NDArray& v0,
    nb::handle draw,
    int64_t m,
    int64_t k,
    int64_t ncv,
    const std::string& which,
    double tol,
    int64_t max_iter,
    bool vectors
) {
    const char* func = "svds";
    const int64_t n = v0.shape[0];
    Gkl<T> g;
    g.X = parse_op<T>(X_o, m, n, func);
    g.XH = parse_op<T>(XH_o, n, m, func);
    g.draw = nb::borrow<nb::object>(draw);
    g.m = m;
    g.n = n;
    g.k = k;
    g.w = ncv;
    g.max_iter = max_iter;
    g.largest = which == "LM";
    g.tol = tol;
    std::vector<double> s;
    std::vector<T> u, v;
    {
        const bool native = g.X.native() && g.XH.native();
        const parallel::Unlocked unlocked(native && g.X.work() + g.XH.work() + (m + n) * ncv >= kParallelThreshold);
        g.run(typed<T>(v0), s, u, v, vectors);
    }
    const int64_t r = static_cast<int64_t>(s.size());
    nb::dict out;
    {
        const int64_t shape[] = {r};
        NDArray a = new_array(DType::Float64, shape);
        if (r) std::memcpy(a.data, s.data(), static_cast<std::size_t>(r) * sizeof(double));
        out["s"] = wrap(std::move(a));
    }
    out["u"] = vectors ? matrix_out(u.data(), m, r) : nb::none();
    out["v"] = vectors ? matrix_out(v.data(), n, r) : nb::none();
    out["status"] = g.status;
    out["nconv"] = g.nconv;
    out["iter"] = g.iter;
    out["nmv"] = g.nmv;
    return out;
}

}  // namespace

void bind_svds(nb::module_& m) {
    m.def(
        "_svds",
        [](nb::handle X, nb::handle XH, nb::handle v0_o, nb::handle draw, int64_t rows, int64_t k, int64_t ncv,
           const std::string& which, double tol, int64_t max_iter, bool vectors) {
            const NDArray& v0 = vector_arg(v0_o, "svds");
            if (v0.dtype == DType::Float64)
                return svds_typed<double>(X, XH, v0, draw, rows, k, ncv, which, tol, max_iter, vectors);
            if (v0.dtype == DType::Complex128)
                return svds_typed<complex128>(X, XH, v0, draw, rows, k, ncv, which, tol, max_iter, vectors);
            throw nb::type_error("svds(): 'v0' must be float64 or complex128");
        },
        nb::arg("X"), nb::arg("XH"), nb::arg("v0"), nb::arg("draw").none(), nb::kw_only(), nb::arg("rows"),
        nb::arg("k"), nb::arg("ncv"), nb::arg("which"), nb::arg("tol"), nb::arg("max_iter"), nb::arg("vectors")
    );
}

}  // namespace sd::sparse
