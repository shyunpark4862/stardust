// Nonlinear least squares' iterations (docs/design/optimize.md §3.4, docs/design/regression.md
// §3.7): sd.optimize.least_squares and sd.regression.nls; sd.optimize.root's 'lm' (§3.6) runs
// the lmder loop with m = n (lm_solve).
//
// The residual vector r(x) (length m) comes from a Python callback (least_squares: the
// user's fun; nls: sqrt(w) (f(x, *theta) - y) on the rows used), and so does a user Jacobian;
// the loop holds the GIL for them. Two methods:
//   - 'lm': MINPACK's lmder (Moré 1978; scipy's leastsq and least_squares(method='lm')): the
//     Levenberg-Marquardt step of lmpar/qrsolv on the column-pivoted QR of J, the step bound
//     update and the ftol, xtol, gtol tests of lmder (factor 100, mode 1).
//   - 'trf': scipy's least_squares(method='trf') (trf_bounds, the 'exact' subproblem, x_scale
//     1, no loss; Branch, Coleman and Li 1999): Coleman-Li scaling, the trust-region subproblem
//     through the SVD, the reflected and the gradient step, strictly feasible iterates. With
//     infinite bounds this is scipy's trf_no_bounds (v = 1, dv = 0: the same steps).
// Difference Jacobians (whose calls nfev counts): MINPACK's fdjac2 (nls), or scipy's
// approx_derivative '2-point' and '3-point' with its steps and their adjustment to the bounds
// (least_squares; diff.hpp).
// The m x p work per Jacobian is one Householder QR (LAPACK geqrf, then ormqr for Q0^T r):
// J = Q0 R0. Everything after it is p-sized: lmder's pivoted QR of J is the pivoted QR of R0
// (Q0 keeps the columns' norms and inner products), and TRF's SVD of [J d; D] is the SVD of
// [R0 d; D] (with the first p entries of U^T [r; 0] from Q0^T r). With fewer residuals than
// variables (trf only) J gets zero rows up to p: the same problem.
// A trial point whose residuals are not all finite is rejected and the trust region shrunk
// (regression G74): lm takes its norm as +inf (lmder's own rule then shrinks the bound
// tenfold), trf sets Delta to a quarter of the step (scipy's rule).
#include "optimize/lsq.hpp"

#include "optimize/diff.hpp"

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "linalg/input.hpp"
#include "linalg/lapack.hpp"

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

namespace sd::optimize {

namespace nb = nanobind;
using linalg::Input;
using linalg::lint;
using linalg::Scratch;
namespace lapack = linalg::lapack;

namespace {

using vec = std::vector<double>;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kDwarf = std::numeric_limits<double>::min();

// The stopping codes (scipy's least_squares status; stardust/optimize/_lsq.py and
// stardust/regression/_nls.py read them): > 0 converged.
enum Status : int {
    sMaxNfev = 0,      // max_nfev reached
    sGtol = 1,         // the gradient test
    sFtol = 2,         // the relative reduction of the sum of squares
    sXtol = 3,         // the relative step
    sBoth = 4,         // ftol and xtol together
    sJacNonfinite = -1,
    sStartNonfinite = -2,
};

// Euclidean norm, scaled when the plain sum of squares under- or overflows.
double enorm(const double* v, lint n) {
    double s = 0;
    for (lint i = 0; i < n; ++i) s += v[i] * v[i];
    if (std::isfinite(s) && s > 1e-280) return std::sqrt(s);
    double mx = 0;
    for (lint i = 0; i < n; ++i) mx = std::max(mx, std::abs(v[i]));
    if (mx == 0 || !std::isfinite(mx)) return mx;
    double t = 0;
    for (lint i = 0; i < n; ++i) {
        const double q = v[i] / mx;
        t += q * q;
    }
    return mx * std::sqrt(t);
}
double enorm(const vec& v) { return enorm(v.data(), static_cast<lint>(v.size())); }

double dot(const vec& a, const vec& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// ---- the residual vector and its Jacobian (Python callbacks)

// The difference Jacobians (diff.hpp)
enum Diff : int {
    kFdjac2 = 0,      // MINPACK's fdjac2 (epsfcn = eps), turned back at a bound or a non-finite side
    kTwoPoint = 1,    // scipy's '2-point'
    kThreePoint = 2,  // scipy's '3-point'
};

struct Model {
    nb::handle fun, jac;  // jac: None for differences
    // None (nls: its callbacks return float64 sd.ndarrays of the right shape), or a Python
    // callable check(value, what) -> such an array, called when a returned value is not one
    nb::handle check;
    const char* fn;
    lint m, p;
    int diff = kFdjac2;
    const vec* rel_step = nullptr;  // scipy's diff_step, one per variable (nullptr: none)
    bool array_x = false;           // x as an sd.ndarray (least_squares), else a list of floats (nls)
    const vec* lb = nullptr;        // the bounds the differences keep to (fdjac2: TRF's only)
    const vec* ub = nullptr;
    double epsfcn = 0;              // fdjac2's epsfcn (root's 'eps'): h = sqrt(max(epsfcn, eps)) |x_j|
    bool minpack_h = false;         // fdjac2 divides by h (MINPACK, root), not by (x_j + h) - x_j (nls)
    lint nfev = 0, njev = 0;
    Scratch sc;
    vec wb;  // '3-point': the second value

    nb::object args(const vec& x) const {
        if (array_x) {
            const std::int64_t shape[1] = {static_cast<std::int64_t>(x.size())};
            NDArray a = new_array(DType::Float64, shape);
            if (!x.empty()) std::memcpy(a.data, x.data(), x.size() * sizeof(double));
            return wrap(std::move(a));
        }
        nb::list a;
        for (double v : x) a.append(v);
        return a;
    }

    // A returned value as a float64 sd.ndarray of shape (rows,) or (rows, cols): taken as it
    // is when it is one, else through check (which raises for a wrong one).
    nb::object conform(nb::object o, lint rows, lint cols, const char* what) const {
        if (check.is_none()) return o;
        if (is_ndarray(o)) {
            const NDArray& a = as_ndarray(o);
            if (a.dtype == DType::Float64 && a.ndim == (cols < 0 ? 1 : 2) && a.shape[0] == rows &&
                (cols < 0 || a.shape[1] == cols))
                return o;
        }
        return check(o, what);
    }

    // r(x) into r (m); false when some value is not finite.
    bool eval(const vec& x, double* r) {
        nb::object o = conform(fun(args(x)), m, -1, "fun");
        ++nfev;
        Input in(o, fn, "f");  // float64 (m,)
        const std::byte* d = in.data();
        const std::int64_t s = in.stride(0);
        bool ok = true;
        for (lint i = 0; i < m; ++i) {
            double v;
            std::memcpy(&v, d + i * s, sizeof v);
            r[i] = v;
            ok = ok && std::isfinite(v);
        }
        return ok;
    }

    // J (m x p, column-major) at x, r = r(x). false when it is not finite. wa: m.
    bool jacobian(vec& x, const double* r, double* J, double* wa) {
        ++njev;
        if (!jac.is_none()) {
            nb::object o = conform(jac(args(x)), m, p, "jac");
            Input in(o, fn, "jac");  // float64 (m, p)
            const std::byte* d = in.data();
            const std::int64_t s0 = in.stride(0), s1 = in.stride(1);
            bool ok = true;
            for (lint j = 0; j < p; ++j)
                for (lint i = 0; i < m; ++i) {
                    double v;
                    std::memcpy(&v, d + i * s0 + j * s1, sizeof v);
                    J[i + j * m] = v;
                    ok = ok && std::isfinite(v);
                }
            return ok;
        }
        if (diff == kTwoPoint) return two_point(x, r, J, wa);
        if (diff == kThreePoint) return three_point(x, r, J, wa);
        // MINPACK's fdjac2: h = sqrt(max(epsfcn, eps)) |x_j| (that at 0), the step actually
        // taken ((x_j + h) - x_j) in the quotient (nls) or h itself (MINPACK; root); backwards
        // when forwards leaves the bounds or the model is not finite there.
        const double eps = std::sqrt(std::max(epsfcn, kEps));
        for (lint j = 0; j < p; ++j) {
            const double t = x[j];
            double h = eps * std::abs(t);
            if (h == 0) h = eps;
            if (ub && t + h > (*ub)[j]) h = -h;
            bool ok = false;
            for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
                if (attempt == 1) {
                    h = -h;
                    if (lb && (t + h < (*lb)[j] || t + h > (*ub)[j])) break;
                }
                x[j] = t + h;
                const double hh = minpack_h ? h : x[j] - t;
                ok = eval(x, wa);
                x[j] = t;
                if (ok)
                    for (lint i = 0; i < m; ++i) J[i + j * m] = (wa[i] - r[i]) / hh;
            }
            if (!ok) return false;
        }
        return true;
    }

    double step(lint j, double t, double rstep) const {
        const bool has = rel_step != nullptr;
        return absolute_step(t, rstep, has, has ? (*rel_step)[static_cast<std::size_t>(j)] : 0.0);
    }

    // scipy's '2-point': (r(x + h e_j) - r(x)) / ((x_j + h) - x_j), h from absolute_step and
    // adjust_one_sided; a side where r is not finite is tried once from the other side when
    // that stays within the bounds.
    bool two_point(vec& x, const double* r, double* J, double* wa) {
        const double rstep = rel_step_2point();
        for (lint j = 0; j < p; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            const double t = x[uj];
            double h = adjust_one_sided(t, step(j, t, rstep), (*lb)[uj], (*ub)[uj]);
            bool ok = false;
            for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
                if (attempt == 1) {
                    h = -h;
                    if (t + h < (*lb)[uj] || t + h > (*ub)[uj]) break;
                }
                x[uj] = t + h;
                ok = eval(x, wa);
                x[uj] = t;
                const double dx = (t + h) - t;
                if (ok)
                    for (lint i = 0; i < m; ++i) J[i + j * m] = (wa[i] - r[i]) / dx;
            }
            if (!ok) return false;
        }
        return true;
    }

    // scipy's '3-point': (r(x + h e_j) - r(x - h e_j)) / ((x_j + h) - (x_j - h)), or at a
    // bound (-3 r(x) + 4 r(x + h e_j) - r(x + 2h e_j)) / ((x_j + 2h) - x_j) (adjust_two_sided).
    bool three_point(vec& x, const double* r, double* J, double* wa) {
        const double rstep = rel_step_3point();
        wb.resize(static_cast<std::size_t>(m));
        for (lint j = 0; j < p; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            const double t = x[uj];
            bool one_sided;
            const double h = adjust_two_sided(t, step(j, t, rstep), (*lb)[uj], (*ub)[uj], one_sided);
            const double x1 = one_sided ? t + h : t - h, x2 = one_sided ? t + 2 * h : t + h;
            x[uj] = x1;
            bool ok = eval(x, wa);
            x[uj] = x2;
            ok = eval(x, wb.data()) && ok;
            x[uj] = t;
            if (!ok) return false;
            const double* w2 = wb.data();
            if (one_sided) {
                const double dx = x2 - t;
                for (lint i = 0; i < m; ++i) J[i + j * m] = (-3.0 * r[i] + 4 * wa[i] - w2[i]) / dx;
            } else {
                const double dx = x2 - x1;
                for (lint i = 0; i < m; ++i) J[i + j * m] = (w2[i] - wa[i]) / dx;
            }
        }
        return true;
    }

    // J (destroyed) = Q0 R0: R0 (p x p column-major, upper) and qtf = (Q0^T r)[0:p]. With
    // m < p (trf), [J; 0] and [r; 0] (p rows).
    void qr(double* J, const double* r, double* R0, double* qtf) {
        if (m < p) {
            const auto up = static_cast<std::size_t>(p);
            vec Jp(up * up, 0.0), rp(up, 0.0);
            for (lint j = 0; j < p; ++j)
                for (lint i = 0; i < m; ++i) Jp[static_cast<std::size_t>(i + j * p)] = J[i + j * m];
            for (lint i = 0; i < m; ++i) rp[static_cast<std::size_t>(i)] = r[i];
            qr_rows(p, Jp.data(), rp.data(), R0, qtf);
            return;
        }
        qr_rows(m, J, r, R0, qtf);
    }

    void qr_rows(lint rows, double* J, const double* r, double* R0, double* qtf) {
        vec tau(static_cast<std::size_t>(p)), c(r, r + rows);
        {
            // the GIL is let go for a large factorization (linalg's 10 us rule, about 2e5 flops)
            std::optional<nb::gil_scoped_release> rel;
            if (2.0 * double(rows) * double(p) * double(p) > 2e5) rel.emplace();
            lapack::geqrf(rows, p, J, rows, tau.data(), sc);
            lapack::ormqr('T', rows, 1, p, J, rows, tau.data(), c.data(), rows, sc);
        }
        for (lint j = 0; j < p; ++j)
            for (lint i = 0; i < p; ++i) R0[i + j * p] = i <= j ? J[i + j * rows] : 0.0;
        for (lint i = 0; i < p; ++i) qtf[i] = c[i];
    }
};

// What the caller keeps of the last iterate: its residuals and Jacobian (least_squares'
// result; nls recomputes them with its accurate Jacobian).
struct Final {
    bool want = false;
    vec J;             // m x p column-major: the Jacobian at the last point it was computed
    bool fresh = false;  // ... which is the current x
    void keep(const double* Jm, lint m, lint p) {
        if (!want) return;
        J.assign(Jm, Jm + m * p);
        fresh = true;
    }
};

// ---- MINPACK (Moré, Garbow, Hillstrom 1980), 0-based, n x n column-major matrices

// qrfac with column pivoting of the n x n matrix a: on return the strict upper triangle of
// R is in a, its diagonal in rdiag, the Householder vectors below; acnorm: a's column norms.
void qrfac(lint n, double* a, lint* ipvt, double* rdiag, double* acnorm, double* wa) {
    auto A = [&](lint i, lint j) -> double& { return a[i + j * n]; };
    for (lint j = 0; j < n; ++j) {
        acnorm[j] = enorm(a + j * n, n);
        rdiag[j] = acnorm[j];
        wa[j] = rdiag[j];
        ipvt[j] = j;
    }
    for (lint j = 0; j < n; ++j) {
        lint kmax = j;
        for (lint k = j; k < n; ++k)
            if (rdiag[k] > rdiag[kmax]) kmax = k;
        if (kmax != j) {
            for (lint i = 0; i < n; ++i) std::swap(A(i, j), A(i, kmax));
            rdiag[kmax] = rdiag[j];
            wa[kmax] = wa[j];
            std::swap(ipvt[j], ipvt[kmax]);
        }
        double ajnorm = enorm(&A(j, j), n - j);
        if (ajnorm != 0) {
            if (A(j, j) < 0) ajnorm = -ajnorm;
            for (lint i = j; i < n; ++i) A(i, j) /= ajnorm;
            A(j, j) += 1;
            for (lint k = j + 1; k < n; ++k) {
                double sum = 0;
                for (lint i = j; i < n; ++i) sum += A(i, j) * A(i, k);
                const double temp = sum / A(j, j);
                for (lint i = j; i < n; ++i) A(i, k) -= temp * A(i, j);
                if (rdiag[k] != 0) {
                    const double t = A(j, k) / rdiag[k];
                    rdiag[k] *= std::sqrt(std::max(0.0, 1 - t * t));
                    const double q = rdiag[k] / wa[k];
                    if (0.05 * q * q <= kEps) {
                        rdiag[k] = j + 1 < n ? enorm(&A(j + 1, k), n - j - 1) : 0.0;
                        wa[k] = rdiag[k];
                    }
                }
            }
        }
        rdiag[j] = -ajnorm;
    }
}

// qrsolv: solves min || [A; D] x - [b; 0] || given the pivoted QR of A (r's upper triangle,
// ipvt, qtb = Q^T b); the strict lower triangle of r receives S's strict upper triangle
// (transposed), sdiag S's diagonal.
void qrsolv(
    lint n, double* r, const lint* ipvt, const double* diag, const double* qtb, double* x, double* sdiag, double* wa
) {
    auto R = [&](lint i, lint j) -> double& { return r[i + j * n]; };
    for (lint j = 0; j < n; ++j) {
        for (lint i = j; i < n; ++i) R(i, j) = R(j, i);
        x[j] = R(j, j);
        wa[j] = qtb[j];
    }
    for (lint j = 0; j < n; ++j) {
        const lint l = ipvt[j];
        if (diag[l] != 0) {
            for (lint k = j; k < n; ++k) sdiag[k] = 0;
            sdiag[j] = diag[l];
            double qtbpj = 0;
            for (lint k = j; k < n; ++k) {
                if (sdiag[k] == 0) continue;
                double c, s;
                if (std::abs(R(k, k)) < std::abs(sdiag[k])) {
                    const double cotan = R(k, k) / sdiag[k];
                    s = 0.5 / std::sqrt(0.25 + 0.25 * cotan * cotan);
                    c = s * cotan;
                } else {
                    const double tan = sdiag[k] / R(k, k);
                    c = 0.5 / std::sqrt(0.25 + 0.25 * tan * tan);
                    s = c * tan;
                }
                R(k, k) = c * R(k, k) + s * sdiag[k];
                const double temp = c * wa[k] + s * qtbpj;
                qtbpj = -s * wa[k] + c * qtbpj;
                wa[k] = temp;
                for (lint i = k + 1; i < n; ++i) {
                    const double t = c * R(i, k) + s * sdiag[i];
                    sdiag[i] = -s * R(i, k) + c * sdiag[i];
                    R(i, k) = t;
                }
            }
        }
        sdiag[j] = R(j, j);
        R(j, j) = x[j];
    }
    lint nsing = n;
    for (lint j = 0; j < n; ++j) {
        if (sdiag[j] == 0 && nsing == n) nsing = j;
        if (nsing < n) wa[j] = 0;
    }
    for (lint k = 0; k < nsing; ++k) {
        const lint j = nsing - 1 - k;
        double sum = 0;
        for (lint i = j + 1; i < nsing; ++i) sum += R(i, j) * wa[i];
        wa[j] = (wa[j] - sum) / sdiag[j];
    }
    for (lint j = 0; j < n; ++j) x[ipvt[j]] = wa[j];
}

// lmpar: the Levenberg-Marquardt parameter par whose step x has ||D x|| within 10% of delta
// (or the Gauss-Newton step when it is inside), r and ipvt from qrfac.
void lmpar(
    lint n,
    double* r,
    const lint* ipvt,
    const double* diag,
    const double* qtb,
    double delta,
    double& par,
    double* x,
    double* sdiag,
    double* wa1,
    double* wa2
) {
    auto R = [&](lint i, lint j) -> double& { return r[i + j * n]; };
    lint nsing = n;
    for (lint j = 0; j < n; ++j) {
        wa1[j] = qtb[j];
        if (R(j, j) == 0 && nsing == n) nsing = j;
        if (nsing < n) wa1[j] = 0;
    }
    for (lint k = 0; k < nsing; ++k) {
        const lint j = nsing - 1 - k;
        wa1[j] /= R(j, j);
        const double temp = wa1[j];
        for (lint i = 0; i < j; ++i) wa1[i] -= R(i, j) * temp;
    }
    for (lint j = 0; j < n; ++j) x[ipvt[j]] = wa1[j];
    lint iter = 0;
    for (lint j = 0; j < n; ++j) wa2[j] = diag[j] * x[j];
    double dxnorm = enorm(wa2, n);
    double fp = dxnorm - delta;
    if (fp <= 0.1 * delta) {
        par = 0;
        return;
    }
    double parl = 0;
    if (nsing >= n) {
        for (lint j = 0; j < n; ++j) {
            const lint l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for (lint j = 0; j < n; ++j) {
            double sum = 0;
            for (lint i = 0; i < j; ++i) sum += R(i, j) * wa1[i];
            wa1[j] = (wa1[j] - sum) / R(j, j);
        }
        const double temp = enorm(wa1, n);
        parl = ((fp / delta) / temp) / temp;
    }
    for (lint j = 0; j < n; ++j) {
        double sum = 0;
        for (lint i = 0; i <= j; ++i) sum += R(i, j) * qtb[i];
        wa1[j] = sum / diag[ipvt[j]];
    }
    const double gnorm = enorm(wa1, n);
    double paru = gnorm / delta;
    if (paru == 0) paru = kDwarf / std::min(delta, 0.1);
    par = std::max(par, parl);
    par = std::min(par, paru);
    if (par == 0) par = gnorm / dxnorm;
    for (;;) {
        ++iter;
        if (par == 0) par = std::max(kDwarf, 0.001 * paru);
        const double sp = std::sqrt(par);
        for (lint j = 0; j < n; ++j) wa1[j] = sp * diag[j];
        qrsolv(n, r, ipvt, wa1, qtb, x, sdiag, wa2);
        for (lint j = 0; j < n; ++j) wa2[j] = diag[j] * x[j];
        dxnorm = enorm(wa2, n);
        const double temp = fp;
        fp = dxnorm - delta;
        if (std::abs(fp) <= 0.1 * delta || (parl == 0 && fp <= temp && temp < 0) || iter == 10) break;
        for (lint j = 0; j < n; ++j) {
            const lint l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for (lint j = 0; j < n; ++j) {
            wa1[j] /= sdiag[j];
            const double t = wa1[j];
            for (lint i = j + 1; i < n; ++i) wa1[i] -= R(i, j) * t;
        }
        const double t = enorm(wa1, n);
        const double parc = ((fp / delta) / t) / t;
        if (fp > 0) parl = std::max(parl, par);
        if (fp < 0) paru = std::min(paru, par);
        par = std::max(parl, par + parc);
    }
}

// lmder's options beyond the tolerances: least_squares and nls use the defaults (factor 100,
// mode 1: the scaling from the Jacobian's column norms) and scipy's status; root takes factor,
// mode 2's diag and lmder's own info (1-8).
struct LmOptions {
    double factor = 100;
    const vec* diag = nullptr;  // mode 2's scaling (nullptr: mode 1)
    bool info = false;          // return lmder's info, not least_squares' status
    bool direct = false;        // m = n: no QR of J before lmder's (root)
};

// lmder's loop.
int levenberg_marquardt(
    Model& M,
    vec& x,
    vec& fvec,
    bool have_f0,
    double ftol,
    double xtol,
    double gtol,
    lint maxfev,
    lint& n_iter,
    Final& fin,
    const LmOptions& opt = {}
) {
    const lint m = M.m, n = M.p;
    const auto un = static_cast<std::size_t>(n), um = static_cast<std::size_t>(m);
    vec wa4(um), J(um * un), R0(un * un), qtf0(un), fjac(un * un), diag(un), qtf(un), wa1(un), wa2(un), wa3(un),
        wa5(un), xt(un);
    std::vector<lint> ipvt(un);
    if (!have_f0 && !M.eval(x, fvec.data())) return sStartNonfinite;
    double fnorm = enorm(fvec);
    double par = 0, delta = 0, xnorm = 0;
    lint iter = 1;
    int info = 0;
    const double factor = opt.factor;
    const bool mode2 = opt.diag != nullptr;
    if (mode2) diag = *opt.diag;
    auto F = [&](lint i, lint j) -> double& { return fjac[static_cast<std::size_t>(i + j * n)]; };
    for (;;) {
        if (!M.jacobian(x, fvec.data(), J.data(), wa4.data())) return sJacNonfinite;
        ++n_iter;
        fin.keep(J.data(), m, n);
        if (opt.direct) {  // m = n: lmder's own QR of J (qrfac below), Q^T f from its reflections
            R0.assign(J.begin(), J.end());
            qtf0.assign(fvec.begin(), fvec.end());
        } else {
            M.qr(J.data(), fvec.data(), R0.data(), qtf0.data());
        }
        fjac = R0;
        qrfac(n, fjac.data(), ipvt.data(), wa1.data(), wa2.data(), wa3.data());
        if (iter == 1) {
            if (!mode2)
                for (lint j = 0; j < n; ++j) diag[j] = wa2[j] == 0 ? 1.0 : wa2[j];
            for (lint j = 0; j < n; ++j) wa3[j] = diag[j] * x[j];
            xnorm = enorm(wa3);
            delta = factor * xnorm;
            if (delta == 0) delta = factor;
        }
        for (lint i = 0; i < n; ++i) wa5[i] = qtf0[i];
        for (lint j = 0; j < n; ++j) {
            if (F(j, j) != 0) {
                double sum = 0;
                for (lint i = j; i < n; ++i) sum += F(i, j) * wa5[i];
                const double temp = -sum / F(j, j);
                for (lint i = j; i < n; ++i) wa5[i] += F(i, j) * temp;
            }
            F(j, j) = wa1[j];
            qtf[j] = wa5[j];
        }
        double gnorm = 0;
        if (fnorm != 0) {
            for (lint j = 0; j < n; ++j) {
                const lint l = ipvt[j];
                if (wa2[l] == 0) continue;
                double sum = 0;
                for (lint i = 0; i <= j; ++i) sum += F(i, j) * (qtf[i] / fnorm);
                gnorm = std::max(gnorm, std::abs(sum / wa2[l]));
            }
        }
        if (gnorm <= gtol) info = 4;
        if (info != 0) break;
        if (!mode2)
            for (lint j = 0; j < n; ++j) diag[j] = std::max(diag[j], wa2[j]);
        double ratio = 0;
        do {
            lmpar(
                n, fjac.data(), ipvt.data(), diag.data(), qtf.data(), delta, par, wa1.data(), wa2.data(), wa3.data(),
                wa5.data()
            );
            for (lint j = 0; j < n; ++j) {
                wa1[j] = -wa1[j];
                xt[j] = x[j] + wa1[j];
                wa3[j] = diag[j] * wa1[j];
            }
            const double pnorm = enorm(wa3);
            if (iter == 1) delta = std::min(delta, pnorm);
            const bool ok = M.eval(xt, wa4.data());
            const double fnorm1 = ok ? enorm(wa4) : kInf;  // G74: not finite = rejected, bound / 10
            double actred = -1;
            if (0.1 * fnorm1 < fnorm) actred = 1 - (fnorm1 / fnorm) * (fnorm1 / fnorm);
            for (lint j = 0; j < n; ++j) {
                wa3[j] = 0;
                const double temp = wa1[ipvt[j]];
                for (lint i = 0; i <= j; ++i) wa3[i] += F(i, j) * temp;
            }
            const double temp1 = enorm(wa3) / fnorm, temp2 = std::sqrt(par) * pnorm / fnorm;
            const double prered = temp1 * temp1 + temp2 * temp2 / 0.5;
            const double dirder = -(temp1 * temp1 + temp2 * temp2);
            ratio = prered != 0 ? actred / prered : 0.0;
            if (ratio <= 0.25) {
                double temp = actred >= 0 ? 0.5 : 0.5 * dirder / (dirder + 0.5 * actred);
                if (0.1 * fnorm1 >= fnorm || temp < 0.1) temp = 0.1;
                delta = temp * std::min(delta, pnorm / 0.1);
                par /= temp;
            } else if (par == 0 || ratio >= 0.75) {
                delta = pnorm / 0.5;
                par *= 0.5;
            }
            if (ratio >= 1e-4) {
                x = xt;
                for (lint j = 0; j < n; ++j) wa2[j] = diag[j] * x[j];
                fvec.swap(wa4);
                xnorm = enorm(wa2);
                fnorm = fnorm1;
                ++iter;
                fin.fresh = false;
            }
            const bool fsmall = std::abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1;
            if (fsmall) info = 1;
            if (delta <= xtol * xnorm) info = 2;
            if (fsmall && info == 2) info = 3;
            if (info != 0) break;
            if (M.nfev >= maxfev) info = 5;
            if (std::abs(actred) <= kEps && prered <= kEps && 0.5 * ratio <= 1) info = 6;
            if (delta <= kEps * xnorm) info = 7;
            if (gnorm <= kEps) info = 8;
            if (info != 0) break;
        } while (ratio < 1e-4);
        if (info != 0) break;
    }
    // the result's Jacobian at the last iterate (scipy's least_squares evaluates jac(x) there)
    if (fin.want && !fin.fresh) {
        fin.J.resize(um * un);
        if (!M.jacobian(x, fvec.data(), fin.J.data(), wa4.data())) return sJacNonfinite;
        fin.fresh = true;
    }
    if (opt.info) return info;
    switch (info) {
        case 1:
        case 6: return sFtol;
        case 2:
        case 7: return sXtol;
        case 3: return sBoth;
        case 4:
        case 8: return sGtol;
        default: return sMaxNfev;
    }
}

// ---- scipy's trust region reflective (scipy/optimize/_lsq/trf.py and common.py), p-space

struct Quad {  // the model 1/2 ||J_h s||^2 + 1/2 s^T diag(dh) s + g_h^T s with J_h = Q0 Rh
    lint p;
    const vec& Rh;  // p x p column-major, upper
    const vec& gh;
    const vec& dh;
    vec mul(const vec& s) const {
        vec out(static_cast<std::size_t>(p), 0.0);
        for (lint j = 0; j < p; ++j)
            for (lint i = 0; i <= j; ++i) out[i] += Rh[i + j * p] * s[j];
        return out;
    }
    double dsd(const vec& a, const vec& b) const {
        double s = 0;
        for (lint i = 0; i < p; ++i) s += a[i] * dh[i] * b[i];
        return s;
    }
    double value(const vec& s) const {  // evaluate_quadratic
        const vec Js = mul(s);
        return 0.5 * (dot(Js, Js) + dsd(s, s)) + dot(s, gh);
    }
    // build_quadratic_1d: f(t) = a t^2 + b t + c along s from s0
    void line(const vec& s, const vec* s0, double& a, double& b, double& c) const {
        const vec v = mul(s);
        a = 0.5 * (dot(v, v) + dsd(s, s));
        b = dot(gh, s);
        c = 0;
        if (s0) {
            const vec u = mul(*s0);
            b += dot(u, v) + dsd(*s0, s);
            c = 0.5 * dot(u, u) + dot(gh, *s0) + 0.5 * dsd(*s0, *s0);
        }
    }
};

std::pair<double, double> minimize_quadratic_1d(double a, double b, double lb, double ub, double c) {
    double ts[3] = {lb, ub, 0};
    int nt = 2;
    if (a != 0) {
        const double ext = -0.5 * b / a;
        if (lb < ext && ext < ub) ts[nt++] = ext;
    }
    int best = 0;
    double ybest = ts[0] * (a * ts[0] + b) + c;
    for (int i = 1; i < nt; ++i) {
        const double y = ts[i] * (a * ts[i] + b) + c;
        if (y < ybest) {
            ybest = y;
            best = i;
        }
    }
    return {ts[best], ybest};
}

bool in_bounds(const vec& x, const vec& lb, const vec& ub) {
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!(x[i] >= lb[i] && x[i] <= ub[i])) return false;
    return true;
}

// the step size along s from x to the first bound, and which bounds it hits (sign of s)
double step_size_to_bound(const vec& x, const vec& s, const vec& lb, const vec& ub, std::vector<int>* hits) {
    const std::size_t p = x.size();
    vec steps(p, kInf);
    for (std::size_t i = 0; i < p; ++i)
        if (s[i] != 0) steps[i] = std::max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i]);
    double mn = kInf;
    for (double v : steps) mn = std::min(mn, v);
    if (hits) {
        hits->assign(p, 0);
        for (std::size_t i = 0; i < p; ++i)
            if (steps[i] == mn) (*hits)[i] = s[i] > 0 ? 1 : (s[i] < 0 ? -1 : 0);
    }
    return mn;
}

// the two t with ||x + t s|| = Delta (x inside)
std::pair<double, double> intersect_trust_region(const vec& x, const vec& s, double Delta) {
    const double a = dot(s, s);
    if (a == 0) return {0.0, 0.0};
    const double b = dot(x, s);
    const double c = std::min(0.0, dot(x, x) - Delta * Delta);
    const double d = std::sqrt(b * b - a * c);
    const double q = -(b + std::copysign(d, b));
    const double t1 = q / a, t2 = q != 0 ? c / q : 0.0;
    return t1 < t2 ? std::make_pair(t1, t2) : std::make_pair(t2, t1);
}

void make_strictly_feasible0(vec& x, const vec& lb, const vec& ub) {  // rstep = 0
    for (std::size_t i = 0; i < x.size(); ++i) {
        int active = 0;
        if (x[i] <= lb[i]) active = -1;
        if (x[i] >= ub[i]) active = 1;
        if (active == -1) x[i] = std::nextafter(lb[i], ub[i]);
        if (active == 1) x[i] = std::nextafter(ub[i], lb[i]);
        if (x[i] < lb[i] || x[i] > ub[i]) x[i] = 0.5 * (lb[i] + ub[i]);
    }
}

void cl_scaling(const vec& x, const vec& g, const vec& lb, const vec& ub, vec& v, vec& dv) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        v[i] = 1;
        dv[i] = 0;
        if (g[i] < 0 && std::isfinite(ub[i])) {
            v[i] = ub[i] - x[i];
            dv[i] = -1;
        }
        if (g[i] > 0 && std::isfinite(lb[i])) {
            v[i] = x[i] - lb[i];
            dv[i] = 1;
        }
    }
}

// solve_lsq_trust_region: min ||J_h p + f|| + reg with ||p|| <= Delta from the SVD
// (s descending, V = Vt^T column-major p x p in Vt, uf = U^T f)
vec solve_lsq_trust_region(lint p, lint m, const vec& uf, const vec& s, const vec& Vt, double Delta, double& alpha) {
    const auto up = static_cast<std::size_t>(p);
    auto Vdot = [&](const vec& w) {
        vec out(up, 0.0);
        for (lint i = 0; i < p; ++i) {
            double t = 0;
            for (lint k = 0; k < p; ++k) t += Vt[k + i * p] * w[k];
            out[i] = t;
        }
        return out;
    };
    vec suf(up);
    for (std::size_t i = 0; i < up; ++i) suf[i] = s[i] * uf[i];
    auto phi = [&](double al, double& dphi) {
        vec q(up);
        double t = 0;
        for (std::size_t i = 0; i < up; ++i) {
            const double den = s[i] * s[i] + al;
            q[i] = suf[i] / den;
            t += suf[i] * suf[i] / (den * den * den);
        }
        const double pn = enorm(q);
        dphi = -t / pn;
        return pn - Delta;
    };
    bool full_rank = false;
    if (m >= p) full_rank = s[up - 1] > kEps * double(m) * s[0];
    if (full_rank) {
        vec w(up);
        for (std::size_t i = 0; i < up; ++i) w[i] = uf[i] / s[i];
        vec pp = Vdot(w);
        for (double& v : pp) v = -v;
        if (enorm(pp) <= Delta) {
            alpha = 0;
            return pp;
        }
    }
    double alpha_upper = enorm(suf) / Delta, alpha_lower = 0;
    if (full_rank) {
        double dphi;
        const double ph = phi(0.0, dphi);
        alpha_lower = -ph / dphi;
    }
    if (!full_rank && alpha == 0) alpha = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
    for (int it = 0; it < 10; ++it) {
        if (alpha < alpha_lower || alpha > alpha_upper)
            alpha = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
        double dphi;
        const double ph = phi(alpha, dphi);
        if (ph < 0) alpha_upper = alpha;
        const double ratio = ph / dphi;
        alpha_lower = std::max(alpha_lower, alpha - ratio);
        alpha -= (ph + Delta) * ratio / Delta;
        if (std::abs(ph) < 0.01 * Delta) break;
    }
    vec w(up);
    for (std::size_t i = 0; i < up; ++i) w[i] = suf[i] / (s[i] * s[i] + alpha);
    vec pp = Vdot(w);
    const double pn = enorm(pp);
    for (double& v : pp) v = -v * (Delta / pn);
    return pp;
}

// select_step: the best of the trust-region step (scaled back from the bound it crosses),
// its reflection and the gradient step. Returns the predicted reduction.
double select_step(
    const vec& x,
    const Quad& Q,
    const vec& gh,
    vec pp,
    vec ph,
    const vec& d,
    double Delta,
    const vec& lb,
    const vec& ub,
    double theta,
    vec& step,
    vec& step_h
) {
    const std::size_t p = x.size();
    vec xp(p);
    for (std::size_t i = 0; i < p; ++i) xp[i] = x[i] + pp[i];
    if (in_bounds(xp, lb, ub)) {
        step = pp;
        step_h = ph;
        return -Q.value(ph);
    }
    std::vector<int> hits;
    const double p_stride = step_size_to_bound(x, pp, lb, ub, &hits);
    vec rh = ph, r(p);
    for (std::size_t i = 0; i < p; ++i)
        if (hits[i] != 0) rh[i] = -rh[i];
    for (std::size_t i = 0; i < p; ++i) r[i] = d[i] * rh[i];
    for (std::size_t i = 0; i < p; ++i) {
        pp[i] *= p_stride;
        ph[i] *= p_stride;
    }
    vec xb(p);
    for (std::size_t i = 0; i < p; ++i) xb[i] = x[i] + pp[i];
    const double to_tr = intersect_trust_region(ph, rh, Delta).second;
    const double to_bound = step_size_to_bound(xb, r, lb, ub, nullptr);
    double r_stride = std::min(to_bound, to_tr), r_lo, r_hi;
    if (r_stride > 0) {
        r_lo = (1 - theta) * p_stride / r_stride;
        r_hi = r_stride == to_bound ? theta * to_bound : to_tr;
    } else {
        r_lo = 0;
        r_hi = -1;
    }
    double r_value = kInf;
    if (r_lo <= r_hi) {
        double a, b, c;
        Q.line(rh, &ph, a, b, c);
        const auto [t, val] = minimize_quadratic_1d(a, b, r_lo, r_hi, c);
        r_value = val;
        for (std::size_t i = 0; i < p; ++i) {
            rh[i] = rh[i] * t + ph[i];
            r[i] = rh[i] * d[i];
        }
    }
    for (std::size_t i = 0; i < p; ++i) {
        pp[i] *= theta;
        ph[i] *= theta;
    }
    const double p_value = Q.value(ph);
    vec agh(p), ag(p);
    for (std::size_t i = 0; i < p; ++i) {
        agh[i] = -gh[i];
        ag[i] = d[i] * agh[i];
    }
    const double tr2 = Delta / enorm(agh);
    const double tb2 = step_size_to_bound(x, ag, lb, ub, nullptr);
    const double ag_stride0 = tb2 < tr2 ? theta * tb2 : tr2;
    double a, b, c;
    Q.line(agh, nullptr, a, b, c);
    const auto [ag_stride, ag_value] = minimize_quadratic_1d(a, b, 0, ag_stride0, 0);
    for (std::size_t i = 0; i < p; ++i) {
        agh[i] *= ag_stride;
        ag[i] *= ag_stride;
    }
    if (p_value < r_value && p_value < ag_value) {
        step = pp;
        step_h = ph;
        return -p_value;
    }
    if (r_value < p_value && r_value < ag_value) {
        step = r;
        step_h = rh;
        return -r_value;
    }
    step = ag;
    step_h = agh;
    return -ag_value;
}

int trust_region_reflective(
    Model& M,
    vec& x,
    vec& f,
    bool have_f0,
    const vec& lb,
    const vec& ub,
    double ftol,
    double xtol,
    double gtol,
    lint max_nfev,
    lint& n_iter,
    Final& fin
) {
    const lint m = M.m, p = M.p;
    const auto up = static_cast<std::size_t>(p), um = static_cast<std::size_t>(m);
    vec fnew(um), wa(um), J(um * up), R0(up * up), qtf(up), g(up), v(up), dv(up), d(up), dh(up), gh(up), Rh(up * up),
        A(2 * up * up), U(2 * up * up), s(up), Vt(up * up), uf(up), step, step_h;
    if (!have_f0 && !M.eval(x, f.data())) return sStartNonfinite;
    auto refresh = [&]() {  // J, its QR and the gradient g = J^T f = R0^T Q0^T f at x
        if (!M.jacobian(x, f.data(), J.data(), wa.data())) return false;
        ++n_iter;
        fin.keep(J.data(), m, p);
        M.qr(J.data(), f.data(), R0.data(), qtf.data());
        for (lint j = 0; j < p; ++j) {
            double t = 0;
            for (lint i = 0; i <= j; ++i) t += R0[i + j * p] * qtf[i];
            g[j] = t;
        }
        return true;
    };
    if (!refresh()) return sJacNonfinite;
    double cost = 0.5 * dot(f, f);
    cl_scaling(x, g, lb, ub, v, dv);
    double Delta = 0;
    {
        vec t(up);
        for (std::size_t i = 0; i < up; ++i) t[i] = x[i] / std::sqrt(v[i]);
        Delta = enorm(t);
        if (Delta == 0) Delta = 1;
    }
    double alpha = 0;
    int status = -100;  // none
    for (;;) {
        cl_scaling(x, g, lb, ub, v, dv);
        double g_norm = 0;
        for (std::size_t i = 0; i < up; ++i) g_norm = std::max(g_norm, std::abs(g[i] * v[i]));
        if (g_norm < gtol) status = sGtol;
        if (status != -100 || M.nfev >= max_nfev) break;
        for (std::size_t i = 0; i < up; ++i) {
            d[i] = std::sqrt(v[i]);
            dh[i] = g[i] * dv[i];
            gh[i] = d[i] * g[i];
        }
        for (lint j = 0; j < p; ++j)
            for (lint i = 0; i < p; ++i) Rh[i + j * p] = R0[i + j * p] * d[j];
        // the SVD of [Rh; diag(sqrt(dh))] (2p x p)
        for (lint j = 0; j < p; ++j)
            for (lint i = 0; i < 2 * p; ++i)
                A[i + j * 2 * p] = i < p ? Rh[i + j * p] : (i - p == j ? std::sqrt(dh[j]) : 0.0);
        if (lapack::gesdd('S', 2 * p, p, A.data(), 2 * p, s.data(), U.data(), 2 * p, Vt.data(), p, M.sc) != 0)
            return sJacNonfinite;
        for (lint k = 0; k < p; ++k) {
            double t = 0;
            for (lint i = 0; i < p; ++i) t += U[i + k * 2 * p] * qtf[i];
            uf[k] = t;
        }
        const Quad Q{p, Rh, gh, dh};
        const double theta = std::max(0.995, 1 - g_norm);
        double actual_reduction = -1;
        while (actual_reduction <= 0 && M.nfev < max_nfev) {
            vec ph = solve_lsq_trust_region(p, m, uf, s, Vt, Delta, alpha);
            vec pp(up);
            for (std::size_t i = 0; i < up; ++i) pp[i] = d[i] * ph[i];
            const double predicted = select_step(x, Q, gh, pp, ph, d, Delta, lb, ub, theta, step, step_h);
            vec xn(up);
            for (std::size_t i = 0; i < up; ++i) xn[i] = x[i] + step[i];
            make_strictly_feasible0(xn, lb, ub);
            const bool ok = M.eval(xn, fnew.data());
            const double step_h_norm = enorm(step_h);
            if (!ok) {  // G74
                Delta = 0.25 * step_h_norm;
                if (!(Delta > 0)) {
                    status = sXtol;
                    break;
                }
                continue;
            }
            const double cost_new = 0.5 * dot(fnew, fnew);
            actual_reduction = cost - cost_new;
            // update_tr_radius
            double ratio;
            if (predicted > 0) ratio = actual_reduction / predicted;
            else if (predicted == 0 && actual_reduction == 0) ratio = 1;
            else ratio = 0;
            double Delta_new = Delta;
            if (ratio < 0.25) Delta_new = 0.25 * step_h_norm;
            else if (ratio > 0.75 && step_h_norm > 0.95 * Delta) Delta_new = 2 * Delta;
            // check_termination
            const double step_norm = enorm(step);
            const bool fsat = actual_reduction < ftol * cost && ratio > 0.25;
            const bool xsat = step_norm < xtol * (xtol + enorm(x));
            if (fsat && xsat) status = sBoth;
            else if (fsat) status = sFtol;
            else if (xsat) status = sXtol;
            if (actual_reduction > 0) {
                x = xn;
                f.swap(fnew);
                cost = cost_new;
            }
            if (status != -100) break;
            alpha *= Delta / Delta_new;
            Delta = Delta_new;
        }
        if (actual_reduction > 0 && !refresh()) return sJacNonfinite;
    }
    return status == -100 ? sMaxNfev : status;
}

nb::object vector_out(const vec& v) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(v.size())};
    NDArray a = new_array(DType::Float64, shape);
    if (!v.empty()) std::memcpy(a.data, v.data(), v.size() * sizeof(double));
    return wrap(std::move(a));
}

nb::object matrix_out(const vec& Jc, lint m, lint p) {  // column-major m x p -> (m, p)
    const std::int64_t shape[2] = {static_cast<std::int64_t>(m), static_cast<std::int64_t>(p)};
    NDArray a = new_array(DType::Float64, shape);
    auto* d = reinterpret_cast<double*>(a.data);
    for (lint i = 0; i < m; ++i)
        for (lint j = 0; j < p; ++j) d[i * p + j] = Jc[static_cast<std::size_t>(i + j * m)];
    return wrap(std::move(a));
}

// lsq_solve(func, fun, jac, check, x0, lb, ub, method, ftol, xtol, gtol, max_nfev, m, diff,
//           diff_step, array_x, f0, final)
//   -> (x [p] list, status, nfev, njev, n_iter, f (m,) or None, J (m, p) or None)
// fun(x) -> residuals (m,); jac(x) -> (m, p), or None for differences (diff: 0 fdjac2, 1
// '2-point', 2 '3-point'; diff_step: scipy's relative steps, empty for none). x is a list of
// floats, or a new float64 sd.ndarray when array_x. check: see Model. method 0 'lm', 1 'trf'
// (lb, ub: the bounds, +-inf for none). f0: the residuals at x0 when the caller has them (its
// call counts in nfev). final: return the last iterate's residuals and Jacobian (lm
// evaluates the Jacobian there when its last one is elsewhere).
nb::object lsq_solve(
    const std::string& func,
    nb::callable fun,
    nb::object jac,
    nb::object check,
    const vec& x0,
    const vec& lb,
    const vec& ub,
    int method,
    double ftol,
    double xtol,
    double gtol,
    std::int64_t max_nfev,
    std::int64_t m,
    int diff,
    const vec& diff_step,
    bool array_x,
    nb::object f0,
    bool final
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Model M;
        M.fun = fun;
        M.jac = jac;
        M.check = check;
        M.fn = fn;
        M.m = static_cast<lint>(m);
        M.p = static_cast<lint>(x0.size());
        M.diff = diff;
        M.array_x = array_x;
        if (!diff_step.empty()) M.rel_step = &diff_step;
        if (diff != kFdjac2 || method == 1) {
            M.lb = &lb;
            M.ub = &ub;
        }
        vec x = x0, f(static_cast<std::size_t>(m));
        const bool have_f0 = !f0.is_none();
        if (have_f0) {
            Input in(f0, fn, "f0");  // float64 (m,), checked by the Python layer
            for (lint i = 0; i < M.m; ++i)
                std::memcpy(&f[static_cast<std::size_t>(i)], in.data() + i * in.stride(0), sizeof(double));
            M.nfev = 1;
        }
        Final fin;
        fin.want = final;
        lint n_iter = 0;
        int status;
        if (method == 0) status = levenberg_marquardt(M, x, f, have_f0, ftol, xtol, gtol, max_nfev, n_iter, fin);
        else status = trust_region_reflective(M, x, f, have_f0, lb, ub, ftol, xtol, gtol, max_nfev, n_iter, fin);
        nb::list xs;
        for (double v : x) xs.append(v);
        nb::object fo = nb::none(), Jo = nb::none();
        if (final && status >= 0) {
            fo = vector_out(f);
            Jo = matrix_out(fin.J, M.m, M.p);
        }
        return nb::make_tuple(
            xs, status, static_cast<std::int64_t>(M.nfev), static_cast<std::int64_t>(M.njev),
            static_cast<std::int64_t>(n_iter), fo, Jo
        );
    });
}

// lm_solve(func, fun, jac, check, x0, f0, ftol, xtol, gtol, max_nfev, epsfcn, factor, diag)
//   -> (x [n] list, f (m,), info, nfev, njev, n_iter)
// root's 'lm': lmder (jac(x) -> (m, n)) or lmdif (jac None: fdjac2 with epsfcn) from f0 =
// fun(x0) (the caller's call, counted in nfev), lmder's info (1-8; -1: a Jacobian is not
// finite). diag: mode 2's scaling, empty for mode 1.
nb::object lm_solve(
    const std::string& func,
    nb::callable fun,
    nb::object jac,
    nb::object check,
    const vec& x0,
    nb::object f0,
    double ftol,
    double xtol,
    double gtol,
    std::int64_t max_nfev,
    double epsfcn,
    double factor,
    const vec& diag
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Model M;
        M.fun = fun;
        M.jac = jac;
        M.check = check;
        M.fn = fn;
        M.p = static_cast<lint>(x0.size());
        M.diff = kFdjac2;
        M.array_x = true;
        M.epsfcn = epsfcn;
        M.minpack_h = true;
        Input in(f0, fn, "f0");  // float64 (m,), checked by the Python layer
        M.m = static_cast<lint>(in.shape(0));
        vec x = x0, f(static_cast<std::size_t>(M.m));
        for (lint i = 0; i < M.m; ++i)
            std::memcpy(&f[static_cast<std::size_t>(i)], in.data() + i * in.stride(0), sizeof(double));
        M.nfev = 1;
        Final fin;
        LmOptions opt;
        opt.factor = factor;
        if (!diag.empty()) opt.diag = &diag;
        opt.info = true;
        opt.direct = M.m == M.p;
        lint n_iter = 0;
        const int info = levenberg_marquardt(M, x, f, true, ftol, xtol, gtol, max_nfev, n_iter, fin, opt);
        nb::list xs;
        for (double v : x) xs.append(v);
        return nb::make_tuple(
            xs, vector_out(f), info, static_cast<std::int64_t>(M.nfev), static_cast<std::int64_t>(M.njev),
            static_cast<std::int64_t>(n_iter)
        );
    });
}

}  // namespace

void bind_lsq(nb::module_& m) {
    m.def(
        "lsq_solve", &lsq_solve, nb::arg("func"), nb::arg("fun"), nb::arg("jac").none(), nb::arg("check").none(),
        nb::arg("x0"), nb::arg("lb"), nb::arg("ub"), nb::arg("method"), nb::arg("ftol"), nb::arg("xtol"),
        nb::arg("gtol"), nb::arg("max_nfev"), nb::arg("m"), nb::arg("diff"), nb::arg("diff_step"), nb::arg("array_x"),
        nb::arg("f0").none(), nb::arg("final")
    );
    m.def(
        "lm_solve", &lm_solve, nb::arg("func"), nb::arg("fun"), nb::arg("jac").none(), nb::arg("check"), nb::arg("x0"),
        nb::arg("f0"), nb::arg("ftol"), nb::arg("xtol"), nb::arg("gtol"), nb::arg("max_nfev"), nb::arg("epsfcn"),
        nb::arg("factor"), nb::arg("diag")
    );
}

}  // namespace sd::optimize
