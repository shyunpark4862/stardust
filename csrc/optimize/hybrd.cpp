// root's 'hybr' (docs/design/optimize.md §3.6): MINPACK's hybrj (the user's Jacobian) and hybrd
// (forward differences, fdjac1), Powell's hybrid method, the same steps as scipy 1.18.1 (O9).
//
// The system F(x) = 0 (n equations, n variables) comes from a Python callback, and so does the
// Jacobian; the loop holds the GIL for them. Each iteration takes Powell's dogleg step (the
// Gauss-Newton step of the QR factors, or its mix with the scaled steepest descent step) inside
// the step bound delta, evaluates F there, updates delta from the ratio of the actual and the
// predicted reduction, and updates the QR factors of the Jacobian by Broyden's rank-one formula
// (r1updt, r1mpyq); the Jacobian is computed again after two failed steps in a row. The stopping
// codes are hybrd's info: 1 delta <= xtol ||D x|| (or F = 0), 2 max_nfev, 3 no improvement
// possible (xtol too small), 4 five Jacobians, 5 ten iterations without progress.
//
// scipy 1.15 and later run a C translation of MINPACK (its _minpack module), built with clang's
// default contraction (-ffp-contract=on: a*b + c in one expression is one FMA). Read from that
// machine code (minpack.hpp): enorm's constants are rdwarf = 2^-64.5 and rgiant = 2^63.5
// (rounded); squares written t**2 in MINPACK are rounded on their own (sq); every running sum is
// fused except qrfac's inner product, which clang vectorizes (dot8). This file asks for the same
// contraction (the pragma below), writes the sums with std::fma so that this compiler's
// vectorizer cannot split them, and keeps MINPACK's expressions, 0-based. Its call points are
// scipy's to the bit. qrfac and qform reflect eight columns at a time (minpack.hpp's
// reflect_columns: each column's sum in its own order, so the same bits).
// Differences from scipy, the decisions of docs/design/optimize.md §1.1:
//   - x0's value comes from the caller (Python's shape check; nfev counts it once). scipy calls
//     fun twice more at x0 (its shape check and its wrapper), and its nfev counts both.
//   - A trial point where F is not finite is a failed step (MINPACK's arithmetic does that by
//     itself: the actual reduction is -1), and the rank-one update, which would spread the
//     non-finite values into the factors, is skipped for it.
//   - A Jacobian that is not finite stops with an error (O26); fdjac1 tries the other side once
//     when one side is not finite (least_squares' '2-point', O27).
//   - jac=True: fun returns (F, J), memoized by x as scipy's MemoizeJac.
#include "optimize/root.hpp"

#include "optimize/minpack.hpp"

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "linalg/input.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#pragma clang fp contract(on)

namespace sd::optimize {

namespace nb = nanobind;
using linalg::Input;

namespace {

using vec = std::vector<double>;
using lint = std::int64_t;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kGiant = std::numeric_limits<double>::max();

using minpack::enorm;
using minpack::sq;

// ---- the system and its Jacobian (Python callbacks)

enum JacMode : int {
    kJacCallable = 0,  // jac(x)
    kJacWithFun = 1,   // jac=True: fun returns (F, J)
    kJacDiff = 2,      // fdjac1's forward differences
};

struct System {
    nb::handle fun, jac;  // jac: the callable (kJacCallable), else None
    // check(value, what) -> a float64 sd.ndarray (n,) ('fun'), (n, n) ('jac') or a pair of them
    // ('pair'), called when a returned value is not already one
    nb::handle check;
    const char* fn = "";
    lint n = 0;
    int mode = kJacCallable;
    double epsfcn = 0;   // fdjac1's epsfcn
    lint ml = 0, mu = 0;  // fdjac1's band (ml + mu + 1 >= n: dense)
    lint nfev = 0, njev = 0;
    // MemoizeJac (kJacWithFun): the last point fun was called at and the values it returned
    vec mx, mf, mJ;
    bool m_valid = false;

    nb::object args(const double* x) const {
        const std::int64_t shape[1] = {static_cast<std::int64_t>(n)};
        NDArray a = new_array(DType::Float64, shape);
        std::memcpy(a.data, x, static_cast<std::size_t>(n) * sizeof(double));
        return wrap(std::move(a));
    }

    nb::object conform(nb::object o, bool matrix, const char* what) const {
        if (is_ndarray(o)) {
            const NDArray& a = as_ndarray(o);
            if (a.dtype == DType::Float64 && a.ndim == (matrix ? 2 : 1) && a.shape[0] == n &&
                (!matrix || a.shape[1] == n))
                return o;
        }
        return check(o, what);
    }

    // a float64 (n,) array into v; false when some value is not finite
    bool read_vector(nb::handle o, double* v) const {
        Input in(o, fn, "fun");
        const std::byte* d = in.data();
        const std::int64_t s = in.stride(0);
        bool ok = true;
        for (lint i = 0; i < n; ++i) {
            std::memcpy(&v[i], d + i * s, sizeof(double));
            ok = ok && std::isfinite(v[i]);
        }
        return ok;
    }
    // a float64 (n, n) array into J (column-major); false when some value is not finite
    bool read_matrix(nb::handle o, double* J) const {
        Input in(o, fn, "jac");
        const std::byte* d = in.data();
        const std::int64_t s0 = in.stride(0), s1 = in.stride(1);
        bool ok = true;
        for (lint j = 0; j < n; ++j)
            for (lint i = 0; i < n; ++i) {
                double v;
                std::memcpy(&v, d + i * s0 + j * s1, sizeof v);
                J[i + j * n] = v;
                ok = ok && std::isfinite(v);
            }
        return ok;
    }

    // calls fun at x (kJacWithFun) and keeps its pair
    void call_pair(const double* x) {
        nb::object pair = check(fun(args(x)), "pair");
        nb::object v = pair[0], J = pair[1];
        remember(x, v, J);
    }
    void remember(const double* x, nb::handle v, nb::handle J) {
        mx.assign(x, x + n);
        mf.resize(static_cast<std::size_t>(n));
        read_vector(v, mf.data());
        mJ.resize(static_cast<std::size_t>(n * n));
        read_matrix(J, mJ.data());
        m_valid = true;
    }
    bool remembered(const double* x) const { return m_valid && std::equal(mx.begin(), mx.end(), x); }

    // F(x) into f; false when some value is not finite.
    bool eval(const double* x, double* f) {
        ++nfev;
        if (mode == kJacWithFun) {
            if (!remembered(x)) call_pair(x);
            bool ok = true;
            for (lint i = 0; i < n; ++i) {
                f[i] = mf[i];
                ok = ok && std::isfinite(f[i]);
            }
            return ok;
        }
        return read_vector(conform(fun(args(x)), false, "fun"), f);
    }

    // The Jacobian at x (fvec = F(x)) into fjac (n x n column-major); false when it is not
    // finite. wa1, wa2: n.
    bool jacobian(double* x, const double* fvec, double* fjac, double* wa1, double* wa2) {
        ++njev;
        if (mode == kJacCallable) return read_matrix(conform(jac(args(x)), true, "jac"), fjac);
        if (mode == kJacWithFun) {
            if (!remembered(x)) call_pair(x);
            std::copy(mJ.begin(), mJ.end(), fjac);
            for (double v : mJ)
                if (!std::isfinite(v)) return false;
            return true;
        }
        return fdjac1(x, fvec, fjac, wa1, wa2);
    }

    // fdjac1: (F(x + h e_j) - F(x)) / h, h = sqrt(max(epsfcn, eps)) |x_j| (that at 0); with a
    // band (ml + mu + 1 < n) the columns k, k + msum, ... share one evaluation. A side where F
    // is not finite is tried once from the other side.
    bool fdjac1(double* x, const double* fvec, double* fjac, double* wa1, double* wa2) {
        const double eps = std::sqrt(std::max(epsfcn, kEps));
        const lint msum = ml + mu + 1;
        if (msum >= n) {
            for (lint j = 0; j < n; ++j) {
                const double temp = x[j];
                double h = eps * std::abs(temp);
                if (h == 0) h = eps;
                bool ok = false;
                for (int side = 0; side < 2 && !ok; ++side) {
                    if (side == 1) h = -h;
                    x[j] = temp + h;
                    ok = eval(x, wa1);
                    x[j] = temp;
                }
                if (!ok) return false;
                for (lint i = 0; i < n; ++i) fjac[i + j * n] = (wa1[i] - fvec[i]) / h;
            }
            return true;
        }
        for (lint k = 0; k < msum; ++k) {
            double sign = 1;
            bool ok = false;
            for (int side = 0; side < 2 && !ok; ++side) {
                if (side == 1) sign = -1;
                for (lint j = k; j < n; j += msum) {
                    wa2[j] = x[j];
                    double h = eps * std::abs(wa2[j]);
                    if (h == 0) h = eps;
                    x[j] = wa2[j] + sign * h;
                }
                ok = eval(x, wa1);
                for (lint j = k; j < n; j += msum) x[j] = wa2[j];
            }
            if (!ok) return false;
            for (lint j = k; j < n; j += msum) {
                double h = eps * std::abs(wa2[j]);
                if (h == 0) h = eps;
                h = sign * h;
                for (lint i = 0; i < n; ++i) {
                    fjac[i + j * n] = 0;
                    if (i >= j - mu && i <= j + ml) fjac[i + j * n] = (wa1[i] - fvec[i]) / h;
                }
            }
        }
        return true;
    }
};

// ---- MINPACK (Moré, Garbow, Hillstrom 1980), 0-based; n x n matrices column-major; the upper
// triangle R packed by rows (lr = n (n + 1) / 2)

// qrfac without pivoting: the Householder QR of a in place (R's strict upper triangle in a, its
// diagonal in rdiag, the vectors below); acnorm: a's column norms.
void qrfac(lint n, double* a, double* rdiag, double* acnorm) {
    auto A = [&](lint i, lint j) -> double& { return a[i + j * n]; };
    for (lint j = 0; j < n; ++j) {
        acnorm[j] = enorm(n, a + j * n);
        rdiag[j] = acnorm[j];
    }
    for (lint j = 0; j < n; ++j) {
        double ajnorm = enorm(n - j, &A(j, j));
        if (ajnorm != 0) {
            if (A(j, j) < 0) ajnorm = -ajnorm;
            for (lint i = j; i < n; ++i) A(i, j) = A(i, j) / ajnorm;
            A(j, j) = A(j, j) + 1.0;
            // each column k > j: temp = dot8(A(j:, j), A(j:, k)) / A(j, j), A(j:, k) -= temp A(j:, j)
            minpack::reflect_columns(n - j, (n - j) / 8 * 8, &A(j, j), &A(j, 0), n, j + 1, n);
        }
        rdiag[j] = -ajnorm;
    }
}

// qform: the orthogonal Q of qrfac's factored form, in place.
void qform(lint n, double* q, double* wa) {
    auto Q = [&](lint i, lint j) -> double& { return q[i + j * n]; };
    for (lint j = 1; j < n; ++j)
        for (lint i = 0; i < j; ++i) Q(i, j) = 0;
    for (lint l = 0; l < n; ++l) {
        const lint k = n - 1 - l;
        for (lint i = k; i < n; ++i) {
            wa[i] = Q(i, k);
            Q(i, k) = 0;
        }
        Q(k, k) = 1;
        if (wa[k] == 0) continue;
        // each column j >= k: temp = (the fused sum of Q(k:, j) wa(k:)) / wa(k), Q(k:, j) -= temp wa(k:)
        minpack::reflect_columns(n - k, 0, &wa[k], &Q(k, 0), n, k, n);
    }
}

// dogleg: the convex combination of the Gauss-Newton and the scaled gradient direction that
// minimizes ||R x - qtb|| along the dogleg path within ||D x|| <= delta.
void dogleg(
    lint n, const double* r, const double* diag, const double* qtb, double delta, double* x, double* wa1, double* wa2
) {
    lint jj = n * (n + 1) / 2;
    for (lint k = 1; k <= n; ++k) {
        const lint j = n - k;
        jj -= k;
        lint l = jj + 1;
        double sum = 0;
        for (lint i = j + 1; i < n; ++i) {
            sum = std::fma(r[l], x[i], sum);
            ++l;
        }
        double temp = r[jj];
        if (temp == 0) {
            l = j;
            for (lint i = 0; i <= j; ++i) {
                temp = std::max(temp, std::abs(r[l]));
                l = l + n - i - 1;
            }
            temp = kEps * temp;
            if (temp == 0) temp = kEps;
        }
        x[j] = (qtb[j] - sum) / temp;
    }
    for (lint j = 0; j < n; ++j) {
        wa1[j] = 0;
        wa2[j] = diag[j] * x[j];
    }
    const double qnorm = enorm(n, wa2);
    if (qnorm <= delta) return;
    lint l = 0;
    for (lint j = 0; j < n; ++j) {
        const double temp = qtb[j];
        for (lint i = j; i < n; ++i) {
            wa1[i] = wa1[i] + r[l] * temp;
            ++l;
        }
        wa1[j] = wa1[j] / diag[j];
    }
    const double gnorm = enorm(n, wa1);
    double sgnorm = 0;
    double alpha = delta / qnorm;
    if (gnorm != 0) {
        for (lint j = 0; j < n; ++j) wa1[j] = (wa1[j] / gnorm) / diag[j];
        l = 0;
        for (lint j = 0; j < n; ++j) {
            double sum = 0;
            for (lint i = j; i < n; ++i) {
                sum = std::fma(r[l], wa1[i], sum);
                ++l;
            }
            wa2[j] = sum;
        }
        double temp = enorm(n, wa2);
        sgnorm = (gnorm / temp) / temp;
        alpha = 0;
        if (sgnorm < delta) {
            const double bnorm = enorm(n, qtb);
            const double dq = delta / qnorm, sd2 = sq(sgnorm / delta);
            temp = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta);
            temp = temp - dq * sd2 + std::sqrt(sq(temp - dq) + (1.0 - sq(dq)) * (1.0 - sd2));
            alpha = (dq * (1.0 - sd2)) / temp;
        }
    }
    const double temp = (1.0 - alpha) * std::min(sgnorm, delta);
    for (lint j = 0; j < n; ++j) x[j] = temp * wa1[j] + alpha * x[j];
}

// r1updt (m = n): the packed upper triangle S of S + u v^T made upper triangular again by
// Givens rotations; v and w receive the rotations (r1mpyq's input). Returns sing.
bool r1updt(lint n, double* s, const double* u, double* v, double* w) {
    const lint m = n;
    lint jj = (n * (2 * m - n + 1)) / 2 - (m - n) - 1;
    lint l = jj;
    for (lint i = n - 1; i < m; ++i) {
        w[i] = s[l];
        ++l;
    }
    for (lint nmj = 1; nmj <= n - 1; ++nmj) {
        const lint j = n - 1 - nmj;
        jj = jj - (m - j);
        w[j] = 0;
        if (v[j] == 0) continue;
        double cs, sn, tau;
        if (std::abs(v[n - 1]) < std::abs(v[j])) {
            const double cotan = v[n - 1] / v[j];
            sn = 0.5 / std::sqrt(0.25 + 0.25 * sq(cotan));
            cs = sn * cotan;
            tau = 1;
            if (std::abs(cs) * kGiant > 1) tau = 1 / cs;
        } else {
            const double tan = v[j] / v[n - 1];
            cs = 0.5 / std::sqrt(0.25 + 0.25 * sq(tan));
            sn = cs * tan;
            tau = sn;
        }
        v[n - 1] = sn * v[j] + cs * v[n - 1];
        v[j] = tau;
        l = jj;
        for (lint i = j; i < m; ++i) {
            const double temp = cs * s[l] - sn * w[i];
            w[i] = sn * s[l] + cs * w[i];
            s[l] = temp;
            ++l;
        }
    }
    for (lint i = 0; i < m; ++i) w[i] = w[i] + v[n - 1] * u[i];
    bool sing = false;
    for (lint j = 0; j < n - 1; ++j) {
        if (w[j] != 0) {
            double cs, sn, tau;
            if (std::abs(s[jj]) < std::abs(w[j])) {
                const double cotan = s[jj] / w[j];
                sn = 0.5 / std::sqrt(0.25 + 0.25 * sq(cotan));
                cs = sn * cotan;
                tau = 1;
                if (std::abs(cs) * kGiant > 1) tau = 1 / cs;
            } else {
                const double tan = w[j] / s[jj];
                cs = 0.5 / std::sqrt(0.25 + 0.25 * sq(tan));
                sn = cs * tan;
                tau = sn;
            }
            l = jj;
            for (lint i = j; i < m; ++i) {
                const double temp = cs * s[l] + sn * w[i];
                w[i] = -sn * s[l] + cs * w[i];
                s[l] = temp;
                ++l;
            }
            w[j] = tau;
        }
        if (s[jj] == 0) sing = true;
        jj = jj + (m - j);
    }
    l = jj;
    for (lint i = n - 1; i < m; ++i) {
        s[l] = w[i];
        ++l;
    }
    if (s[jj] == 0) sing = true;
    return sing;
}

// r1mpyq: a (m x n, leading dimension lda) times the rotations r1updt stored in v and w.
void r1mpyq(lint m, lint n, double* a, lint lda, const double* v, const double* w) {
    auto A = [&](lint i, lint j) -> double& { return a[i + j * lda]; };
    for (lint nmj = 1; nmj <= n - 1; ++nmj) {
        const lint j = n - 1 - nmj;
        double cs, sn;
        if (std::abs(v[j]) > 1) {
            cs = 1 / v[j];
            sn = std::sqrt(1 - cs * cs);
        } else {
            sn = v[j];
            cs = std::sqrt(1 - sn * sn);
        }
        for (lint i = 0; i < m; ++i) {
            const double temp = cs * A(i, j) - sn * A(i, n - 1);
            A(i, n - 1) = sn * A(i, j) + cs * A(i, n - 1);
            A(i, j) = temp;
        }
    }
    for (lint j = 0; j < n - 1; ++j) {
        double cs, sn;
        if (std::abs(w[j]) > 1) {
            cs = 1 / w[j];
            sn = std::sqrt(1 - cs * cs);
        } else {
            sn = w[j];
            cs = std::sqrt(1 - sn * sn);
        }
        for (lint i = 0; i < m; ++i) {
            const double temp = cs * A(i, j) + sn * A(i, n - 1);
            A(i, n - 1) = -sn * A(i, j) + cs * A(i, n - 1);
            A(i, j) = temp;
        }
    }
}

constexpr int kJacNonfinite = -1;  // the Jacobian is not finite (an error, O26)

// hybrd / hybrj's loop. fvec holds F(x0) on entry. diag: empty for mode 1 (the scaling from the
// Jacobian's column norms), else mode 2's. Returns info (1-5) or kJacNonfinite.
int hybrid(System& S, vec& x, vec& fvec, double xtol, lint maxfev, double factor, vec diag, lint& n_iter) {
    const lint n = S.n;
    const auto un = static_cast<std::size_t>(n);
    const bool mode2 = !diag.empty();
    if (!mode2) diag.assign(un, 0.0);
    vec fjac(un * un), r(un * (un + 1) / 2), qtf(un), wa1(un), wa2(un), wa3(un), wa4(un);
    auto F = [&](lint i, lint j) -> double& { return fjac[static_cast<std::size_t>(i + j * n)]; };
    double fnorm = enorm(n, fvec.data());
    lint iter = 1, ncsuc = 0, ncfail = 0, nslow1 = 0, nslow2 = 0;
    double delta = 0, xnorm = 0;
    int info = 0;
    // the O(n^3) factorization and Q, and the O(n^2) rotations, run without the GIL when large
    // (the 10 us rule, about 2e5 flops; they touch no Python object)
    const bool big = 2.0 * double(n) * double(n) * double(n) > 2e5, big2 = 4.0 * double(n) * double(n) > 2e5;
    for (;;) {
        bool jeval = true;
        if (!S.jacobian(x.data(), fvec.data(), fjac.data(), wa1.data(), wa2.data())) return kJacNonfinite;
        {
            std::optional<nb::gil_scoped_release> rel;
            if (big) rel.emplace();
            qrfac(n, fjac.data(), wa1.data(), wa2.data());
        }
        if (iter == 1) {
            if (!mode2)
                for (lint j = 0; j < n; ++j) diag[j] = wa2[j] == 0 ? 1.0 : wa2[j];
            for (lint j = 0; j < n; ++j) wa3[j] = diag[j] * x[j];
            xnorm = enorm(n, wa3.data());
            delta = factor * xnorm;
            if (delta == 0) delta = factor;
        }
        for (lint i = 0; i < n; ++i) qtf[i] = fvec[i];
        for (lint j = 0; j < n; ++j) {
            if (F(j, j) == 0) continue;
            double sum = 0;
            for (lint i = j; i < n; ++i) sum = std::fma(F(i, j), qtf[i], sum);
            const double temp = -sum / F(j, j);
            for (lint i = j; i < n; ++i) qtf[i] = qtf[i] + F(i, j) * temp;
        }
        for (lint j = 0; j < n; ++j) {
            lint l = j;
            for (lint i = 0; i < j; ++i) {
                r[l] = F(i, j);
                l = l + n - i - 1;
            }
            r[l] = wa1[j];
        }
        {
            std::optional<nb::gil_scoped_release> rel;
            if (big) rel.emplace();
            qform(n, fjac.data(), wa1.data());
        }
        if (!mode2)
            for (lint j = 0; j < n; ++j) diag[j] = std::max(diag[j], wa2[j]);
        for (;;) {
            dogleg(n, r.data(), diag.data(), qtf.data(), delta, wa1.data(), wa2.data(), wa3.data());
            for (lint j = 0; j < n; ++j) {
                wa1[j] = -wa1[j];
                wa2[j] = x[j] + wa1[j];
                wa3[j] = diag[j] * wa1[j];
            }
            const double pnorm = enorm(n, wa3.data());
            if (iter == 1) delta = std::min(delta, pnorm);
            const bool finite = S.eval(wa2.data(), wa4.data());
            ++n_iter;
            const double fnorm1 = enorm(n, wa4.data());
            double actred = -1;
            if (fnorm1 < fnorm) actred = 1.0 - sq(fnorm1 / fnorm);
            lint l = 0;
            for (lint i = 0; i < n; ++i) {
                double sum = 0;
                for (lint j = i; j < n; ++j) {
                    sum = std::fma(r[l], wa1[j], sum);
                    ++l;
                }
                wa3[i] = qtf[i] + sum;
            }
            const double temp = enorm(n, wa3.data());
            double prered = 0;
            if (temp < fnorm) prered = 1.0 - sq(temp / fnorm);
            double ratio = 0;
            if (prered > 0) ratio = actred / prered;
            if (ratio < 0.1) {
                ncsuc = 0;
                ++ncfail;
                delta = 0.5 * delta;
            } else {
                ncfail = 0;
                ++ncsuc;
                if (ratio >= 0.5 || ncsuc > 1) delta = std::max(delta, pnorm / 0.5);
                if (std::abs(ratio - 1.0) <= 0.1) delta = pnorm / 0.5;
            }
            if (ratio >= 1e-4) {
                for (lint j = 0; j < n; ++j) {
                    x[j] = wa2[j];
                    wa2[j] = diag[j] * x[j];
                    fvec[j] = wa4[j];
                }
                xnorm = enorm(n, wa2.data());
                fnorm = fnorm1;
                ++iter;
            }
            ++nslow1;
            if (actred >= 0.001) nslow1 = 0;
            if (jeval) ++nslow2;
            if (actred >= 0.1) nslow2 = 0;
            if (delta <= xtol * xnorm || fnorm == 0) info = 1;
            if (info != 0) return info;
            if (S.nfev >= maxfev) info = 2;
            if (0.1 * std::max(0.1 * delta, pnorm) <= kEps * xnorm) info = 3;
            if (nslow2 == 5) info = 4;
            if (nslow1 == 10) info = 5;
            if (info != 0) return info;
            if (ncfail == 2) break;  // the Jacobian again
            jeval = false;
            if (!finite) continue;  // no rank-one update from non-finite values
            for (lint j = 0; j < n; ++j) {
                double sum = 0;
                for (lint i = 0; i < n; ++i) sum = std::fma(F(i, j), wa4[i], sum);
                wa2[j] = (sum - wa3[j]) / pnorm;
                wa1[j] = diag[j] * ((diag[j] * wa1[j]) / pnorm);
                if (ratio >= 1e-4) qtf[j] = sum;
            }
            {
                std::optional<nb::gil_scoped_release> rel;
                if (big2) rel.emplace();
                r1updt(n, r.data(), wa1.data(), wa2.data(), wa3.data());
                r1mpyq(n, n, fjac.data(), n, wa2.data(), wa3.data());
                r1mpyq(1, n, qtf.data(), 1, wa2.data(), wa3.data());
            }
        }
    }
}

nb::object vector_out(const vec& v) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(v.size())};
    NDArray a = new_array(DType::Float64, shape);
    if (!v.empty()) std::memcpy(a.data, v.data(), v.size() * sizeof(double));
    return wrap(std::move(a));
}

// hybr_solve(func, fun, jac, check, x0, f0, mode, xtol, max_nfev, ml, mu, epsfcn, factor, diag)
//   -> (x [n] list, F(x) (n,), info, nfev, njev, n_iter)
// fun(x) -> F (n,); mode 0: jac(x) -> (n, n); 1: fun(x) -> (F, J) (jac None); 2: fdjac1's
// differences with ml, mu, epsfcn. f0: fun's value at x0 (the caller's call, counted in nfev;
// with mode 1 the pair). diag: mode 2's scaling, empty for mode 1. info -1: the Jacobian at the
// returned x is not finite.
nb::object hybr_solve(
    const std::string& func,
    nb::callable fun,
    nb::object jac,
    nb::object check,
    const vec& x0,
    nb::object f0,
    int mode,
    double xtol,
    std::int64_t max_nfev,
    std::int64_t ml,
    std::int64_t mu,
    double epsfcn,
    double factor,
    const vec& diag
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        System S;
        S.fun = fun;
        S.jac = jac;
        S.check = check;
        S.fn = fn;
        S.n = static_cast<lint>(x0.size());
        S.mode = mode;
        S.epsfcn = epsfcn;
        S.ml = ml;
        S.mu = mu;
        vec x = x0, f(x0.size());
        if (mode == kJacWithFun) {
            nb::object v = f0[0], J = f0[1];
            S.read_vector(v, f.data());
            S.remember(x.data(), v, J);
        } else {
            S.read_vector(f0, f.data());
        }
        S.nfev = 1;
        lint n_iter = 0;
        const int info = hybrid(S, x, f, xtol, max_nfev, factor, diag, n_iter);
        nb::list xs;
        for (double v : x) xs.append(v);
        return nb::make_tuple(
            xs, vector_out(f), info, static_cast<std::int64_t>(S.nfev), static_cast<std::int64_t>(S.njev),
            static_cast<std::int64_t>(n_iter)
        );
    });
}

}  // namespace

void bind_hybr(nb::module_& m) {
    m.def(
        "hybr_solve", &hybr_solve, nb::arg("func"), nb::arg("fun"), nb::arg("jac").none(), nb::arg("check"),
        nb::arg("x0"), nb::arg("f0"), nb::arg("mode"), nb::arg("xtol"), nb::arg("max_nfev"), nb::arg("ml"),
        nb::arg("mu"), nb::arg("epsfcn"), nb::arg("factor"), nb::arg("diag")
    );
}

}  // namespace sd::optimize
