// Linear least squares with bounds (docs/design/optimize.md §3.4): lsq_linear's 'trf' and
// 'bvls' and nnls, the same steps as scipy 1.18.1 (O9).
//
//   - lsq_linear: the unbounded solution by numpy's lstsq (LAPACK gelsd, rcond -1); when it is
//     outside the bounds, 'trf' is scipy's trf_linear (Branch, Coleman and Li's trust region
//     reflective method for a quadratic: one pivoted QR of A, then per iteration the Coleman-Li
//     scaling, the regularized Gauss-Newton step from that QR by Givens rotations, the reflected
//     and the gradient step) and 'bvls' scipy's bvls (Stark and Parker 1995: an active set
//     method with numpy's lstsq on the free columns).
//   - nnls: scipy 1.18.1's C version of Lawson and Hanson's NNLS (its _slsqplib module, read
//     from the machine code: LAPACK dlarfgp, dlarf and dlartgp in place of Lawson and Hanson's
//     H12 and G1, a column taken in when |beta| > 100 ulp(||its part in P||), alpha starting at
//     2, a rejected column keeping its scaled reflection vector). Its work arrays are laid out as
//     scipy's wrapper lays them out (x, A, b, w, zz in one buffer), and numpy's lstsq's as numpy
//     lays them out: Accelerate's results depend on the arrays' alignment. One difference: when
//     dropping columns leaves P empty, scipy's loop goes on with a negative count (it would write
//     before its index array); here the inner loop ends there.
// Every array computation is the numpy or LAPACK call scipy makes, through Accelerate as numpy
// and scipy call it: A.dot(x) is cblas_dgemv on the C-ordered A, A.T.dot(r) the same matrix
// column-major, np.dot of two vectors 0.0 + cblas_ddot (the product when they have one element),
// norm sqrt(x.dot(x)); scalar arithmetic is Python's (no fused multiply-add), except the parts
// scipy compiled (nnls, givens_elimination), whose fused forms are written with std::fma.
#include "optimize/linear.hpp"

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

// ---- numpy's operations

// np.dot of two vectors: the product of one element, else 0.0 + ddot (BLAS's -0.0 becomes 0.0)
double npdot(const double* a, const double* b, lint n) {
    if (n == 1) return a[0] * b[0];
    return 0.0 + cblas_ddot(n, a, 1, b, 1);
}

// np.max (NaN propagates) and np.argmax / np.argmin (the first extreme, the first NaN)
double npmax(const double* a, lint n) {
    double m = a[0];
    for (lint i = 1; i < n; ++i)
        if (std::isnan(m) || !(a[i] <= m)) m = std::isnan(m) ? m : a[i];
    return m;
}
lint npargmax(const double* a, lint n) {
    lint k = 0;
    for (lint i = 0; i < n; ++i) {
        if (std::isnan(a[i])) return i;
        if (a[i] > a[k]) k = i;
    }
    return k;
}
lint npargmin(const double* a, lint n) {
    lint k = 0;
    for (lint i = 0; i < n; ++i) {
        if (std::isnan(a[i])) return i;
        if (a[i] < a[k]) k = i;
    }
    return k;
}

// y = M.dot(x) for a C-ordered (rows, cols) M, as numpy's cblas_matrixproduct computes it
void np_matvec(const double* M, lint rows, lint cols, const double* x, double* y) {
    if (cols == 1) {  // a column times a number: daxpy into zeros (rows > 1) or the product
        for (lint i = 0; i < rows; ++i) y[i] = rows == 1 ? x[0] * M[i] : 0.0 + x[0] * M[i];
        return;
    }
    if (rows == 1) {
        y[0] = 0.0 + cblas_ddot(cols, M, 1, x, 1);
        return;
    }
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0, M, cols, x, 1, 0.0, y, 1);
}
// y = M.T.dot(x) for a C-ordered (rows, cols) M (M.T is Fortran-ordered)
void np_rmatvec(const double* M, lint rows, lint cols, const double* x, double* y) {
    if (rows == 1) {
        for (lint j = 0; j < cols; ++j) y[j] = cols == 1 ? x[0] * M[j] : 0.0 + x[0] * M[j];
        return;
    }
    if (cols == 1) {
        y[0] = 0.0 + cblas_ddot(rows, M, 1, x, 1);
        return;
    }
    cblas_dgemv(CblasColMajor, CblasNoTrans, cols, rows, 1.0, M, cols, x, 1, 0.0, y, 1);
}

struct Problem {
    lint m = 0, n = 0;
    vec A;  // C-ordered (m, n)
    vec b;
    Scratch sc;

    // r = A x - b, g = A^T r (when g is given), 0.5 r.r
    double residual(const vec& x, vec& r, vec* g) const {
        np_matvec(A.data(), m, n, x.data(), r.data());
        for (lint i = 0; i < m; ++i) r[i] = r[i] - b[i];
        if (g) np_rmatvec(A.data(), m, n, r.data(), g->data());
        return 0.5 * npdot(r.data(), r.data(), m);
    }

    // numpy's lstsq(M, rhs, rcond): M C-ordered (rows, cols); x (cols). LAPACK gelsd on
    // numpy's buffers: a, b and s in one block, work then iwork in another (Accelerate's
    // results depend on the alignment of the arrays it is given).
    void lstsq(const double* M, lint rows, lint cols, const double* rhs, double rcond, double* x) {
        if (cols == 0) return;
        const lint ldb = std::max<lint>(1, std::max(rows, cols)), lda = std::max<lint>(1, rows), nrhs = 1;
        const lint mn = std::min(rows, cols);
        buf1.assign(static_cast<std::size_t>(rows * cols + ldb + mn), 0.0);
        double *a = buf1.data(), *bb = a + rows * cols, *sv = bb + ldb;
        for (lint i = 0; i < rows; ++i)
            for (lint j = 0; j < cols; ++j) a[i + j * rows] = M[i * cols + j];
        for (lint i = 0; i < rows; ++i) bb[i] = rhs[i];
        lint rank = 0, info = 0, lwork = -1, iq = 0;
        double q = 0;
        dgelsd_(&rows, &cols, &nrhs, a, &lda, bb, &ldb, sv, &rcond, &rank, &q, &lwork, &iq, &info);
        lwork = static_cast<lint>(q);
        const auto nw = static_cast<std::size_t>(lwork), ni = static_cast<std::size_t>(std::max<lint>(1, iq));
        buf2.resize(nw + ni);
        dgelsd_(
            &rows, &cols, &nrhs, a, &lda, bb, &ldb, sv, &rcond, &rank, buf2.data(), &lwork,
            reinterpret_cast<lint*>(buf2.data() + nw), &info
        );
        for (lint j = 0; j < cols; ++j) x[j] = bb[j];
    }
    vec buf1, buf2;
};

bool in_bounds(const vec& x, const vec& lb, const vec& ub) {
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!(x[i] >= lb[i] && x[i] <= ub[i])) return false;
    return true;
}

// ---- scipy/optimize/_lsq/common.py

// step_size_to_bound: the step along s from x to the nearest bound, and which bounds it hits
// (the sign of s there, 0 elsewhere)
double step_size_to_bound(const vec& x, const vec& s, const vec& lb, const vec& ub, std::vector<int>* hits) {
    const std::size_t n = x.size();
    vec steps(n, kInf);
    for (std::size_t i = 0; i < n; ++i)
        if (s[i] != 0) steps[i] = std::max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i]);
    double mn = steps[0];
    for (std::size_t i = 1; i < n; ++i)
        if (std::isnan(mn) || !(steps[i] >= mn)) mn = std::isnan(mn) ? mn : steps[i];
    if (hits) {
        hits->assign(n, 0);
        for (std::size_t i = 0; i < n; ++i)
            if (steps[i] == mn) (*hits)[i] = s[i] > 0 ? 1 : (s[i] < 0 ? -1 : 0);
    }
    return mn;
}

// find_active_constraints
std::vector<int> find_active(const vec& x, const vec& lb, const vec& ub, double rtol) {
    const std::size_t n = x.size();
    std::vector<int> active(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (rtol == 0) {
            if (x[i] <= lb[i]) active[i] = -1;
            if (x[i] >= ub[i]) active[i] = 1;
            continue;
        }
        const double lo = x[i] - lb[i], up = ub[i] - x[i];
        const double lt = rtol * std::max(1.0, std::abs(lb[i])), ut = rtol * std::max(1.0, std::abs(ub[i]));
        if (std::isfinite(lb[i]) && lo <= std::min(up, lt)) active[i] = -1;
        if (std::isfinite(ub[i]) && up <= std::min(lo, ut)) active[i] = 1;
    }
    return active;
}

// make_strictly_feasible
vec make_strictly_feasible(const vec& x, const vec& lb, const vec& ub, double rstep) {
    vec out = x;
    const std::vector<int> active = find_active(x, lb, ub, rstep);
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (active[i] == -1)
            out[i] = rstep == 0 ? std::nextafter(lb[i], ub[i]) : lb[i] + rstep * std::max(1.0, std::abs(lb[i]));
        else if (active[i] == 1)
            out[i] = rstep == 0 ? std::nextafter(ub[i], lb[i]) : ub[i] - rstep * std::max(1.0, std::abs(ub[i]));
    }
    for (std::size_t i = 0; i < x.size(); ++i)
        if (out[i] < lb[i] || out[i] > ub[i]) out[i] = 0.5 * (lb[i] + ub[i]);
    return out;
}

// reflective_transformation (the point only)
vec reflect(const vec& y, const vec& lb, const vec& ub) {
    if (in_bounds(y, lb, ub)) return y;
    vec x = y;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const bool lf = std::isfinite(lb[i]), uf = std::isfinite(ub[i]);
        if (lf && !uf) x[i] = std::max(y[i], 2 * lb[i] - y[i]);
        else if (!lf && uf) x[i] = std::min(y[i], 2 * ub[i] - y[i]);
        else if (lf && uf) {
            const double d = ub[i] - lb[i];
            const double t = std::fmod(y[i] - lb[i], 2 * d);
            // numpy's remainder: the sign of the divisor
            const double tt = (t != 0 && (t < 0) != (2 * d < 0)) ? t + 2 * d : (t == 0 ? std::copysign(0.0, 2 * d) : t);
            x[i] = lb[i] + std::min(tt, 2 * d - tt);
        }
    }
    return x;
}

// minimize_quadratic_1d: (t, y) minimizing a t^2 + b t + c on [lb, ub]
std::pair<double, double> minimize_quadratic_1d(double a, double b, double lb, double ub, double c) {
    double t[3] = {lb, ub, 0};
    int k = 2;
    if (a != 0) {
        const double extremum = -0.5 * b / a;
        if (lb < extremum && extremum < ub) t[k++] = extremum;
    }
    double y[3];
    for (int i = 0; i < k; ++i) y[i] = t[i] * (a * t[i] + b) + c;
    int best = 0;
    for (int i = 0; i < k; ++i) {
        if (std::isnan(y[i])) {
            best = i;
            break;
        }
        if (y[i] < y[best]) best = i;
    }
    return {t[best], y[best]};
}

// ---- scipy/optimize/_lsq/trf_linear.py

struct Trf {
    Problem& P;
    const vec& lb;
    const vec& ub;
    lint m, n;
    vec Ad_buf, Js, Js0;  // scratch for products

    Trf(Problem& p, const vec& l, const vec& u) : P(p), lb(l), ub(u), m(p.m), n(p.n) {
        Ad_buf.resize(static_cast<std::size_t>(n));
        Js.resize(static_cast<std::size_t>(m));
        Js0.resize(static_cast<std::size_t>(m));
    }

    // J.dot(s) for J = A (d empty) or A_h = A diag(d) (right_multiplied_operator)
    void Jdot(const vec& d, const vec& s, vec& out) {
        if (d.empty()) {
            np_matvec(P.A.data(), m, n, s.data(), out.data());
            return;
        }
        for (lint j = 0; j < n; ++j) Ad_buf[j] = s[j] * d[j];
        np_matvec(P.A.data(), m, n, Ad_buf.data(), out.data());
    }

    // evaluate_quadratic(J, g, s, diag)
    double evaluate_quadratic(const vec& d, const vec& g, const vec& s, const vec* diag) {
        Jdot(d, s, Js);
        double q = npdot(Js.data(), Js.data(), m);
        if (diag) {
            vec sd(static_cast<std::size_t>(n));
            for (lint j = 0; j < n; ++j) sd[j] = s[j] * (*diag)[j];
            q = q + npdot(sd.data(), s.data(), n);
        }
        const double l = npdot(s.data(), g.data(), n);
        return 0.5 * q + l;
    }

    // build_quadratic_1d(J, g, s, diag, s0) -> (a, b, c)
    void build_quadratic_1d(
        const vec& d, const vec& g, const vec& s, const vec* diag, const vec* s0, double& a, double& b, double& c
    ) {
        Jdot(d, s, Js);
        a = npdot(Js.data(), Js.data(), m);
        vec t(static_cast<std::size_t>(n));
        if (diag) {
            for (lint j = 0; j < n; ++j) t[j] = s[j] * (*diag)[j];
            a = a + npdot(t.data(), s.data(), n);
        }
        a = a * 0.5;
        b = npdot(g.data(), s.data(), n);
        c = 0;
        if (s0) {
            Jdot(d, *s0, Js0);
            b = b + npdot(Js0.data(), Js.data(), m);
            c = 0.5 * npdot(Js0.data(), Js0.data(), m) + npdot(g.data(), s0->data(), n);
            if (diag) {
                for (lint j = 0; j < n; ++j) t[j] = (*s0)[j] * (*diag)[j];
                b = b + npdot(t.data(), s.data(), n);
                c = c + 0.5 * npdot(t.data(), s0->data(), n);
            }
        }
    }

    // select_step
    vec select_step(const vec& x, const vec& g_h, const vec& c_h, vec p, vec p_h, const vec& d, double theta) {
        vec xp(static_cast<std::size_t>(n));
        for (lint j = 0; j < n; ++j) xp[j] = x[j] + p[j];
        if (in_bounds(xp, lb, ub)) return p;
        std::vector<int> hits;
        const double p_stride = step_size_to_bound(x, p, lb, ub, &hits);
        vec r_h = p_h, r(static_cast<std::size_t>(n));
        for (lint j = 0; j < n; ++j)
            if (hits[j] != 0) r_h[j] = r_h[j] * -1;
        for (lint j = 0; j < n; ++j) r[j] = d[j] * r_h[j];
        for (lint j = 0; j < n; ++j) {
            p[j] = p[j] * p_stride;
            p_h[j] = p_h[j] * p_stride;
        }
        vec x_on_bound(static_cast<std::size_t>(n));
        for (lint j = 0; j < n; ++j) x_on_bound[j] = x[j] + p[j];
        double r_stride_u = step_size_to_bound(x_on_bound, r, lb, ub, nullptr);
        const double r_stride_l = (1 - theta) * r_stride_u;
        r_stride_u = r_stride_u * theta;
        double r_value;
        if (r_stride_u > 0) {
            double a, b, c;
            build_quadratic_1d(d, g_h, r_h, &c_h, &p_h, a, b, c);
            const auto [r_stride, rv] = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c);
            r_value = rv;
            for (lint j = 0; j < n; ++j) r_h[j] = p_h[j] + r_h[j] * r_stride;
            for (lint j = 0; j < n; ++j) r[j] = d[j] * r_h[j];
        } else {
            r_value = kInf;
        }
        for (lint j = 0; j < n; ++j) {
            p_h[j] = p_h[j] * theta;
            p[j] = p[j] * theta;
        }
        const double p_value = evaluate_quadratic(d, g_h, p_h, &c_h);
        vec ag_h(static_cast<std::size_t>(n)), ag(static_cast<std::size_t>(n));
        for (lint j = 0; j < n; ++j) ag_h[j] = -g_h[j];
        for (lint j = 0; j < n; ++j) ag[j] = d[j] * ag_h[j];
        double ag_stride_u = step_size_to_bound(x, ag, lb, ub, nullptr);
        ag_stride_u = ag_stride_u * theta;
        double a, b, c;
        build_quadratic_1d(d, g_h, ag_h, &c_h, nullptr, a, b, c);
        const auto [ag_stride, ag_value] = minimize_quadratic_1d(a, b, 0, ag_stride_u, 0);
        for (lint j = 0; j < n; ++j) ag[j] = ag[j] * ag_stride;
        if (p_value < r_value && p_value < ag_value) return p;
        if (r_value < p_value && r_value < ag_value) return r;
        return ag;
    }

    // backtracking: (step, cost_change), x unchanged
    std::pair<vec, double> backtracking(const vec& g, const vec& x, const vec& p, double theta, double p_dot_g) {
        double alpha = 1;
        vec x_new(static_cast<std::size_t>(n)), step(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
        double cost_change;
        static const vec none;
        for (;;) {
            for (lint j = 0; j < n; ++j) y[j] = x[j] + alpha * p[j];
            x_new = reflect(y, lb, ub);
            for (lint j = 0; j < n; ++j) step[j] = x_new[j] - x[j];
            cost_change = -evaluate_quadratic(none, g, step, nullptr);
            if (cost_change > -0.1 * alpha * p_dot_g) break;
            alpha = alpha * 0.5;
        }
        const std::vector<int> active = find_active(x_new, lb, ub, 1e-10);
        if (std::any_of(active.begin(), active.end(), [](int a) { return a != 0; })) {
            for (lint j = 0; j < n; ++j) y[j] = x[j] + theta * alpha * p[j];
            x_new = make_strictly_feasible(reflect(y, lb, ub), lb, ub, 0);
            for (lint j = 0; j < n; ++j) step[j] = x_new[j] - x[j];
            cost_change = -evaluate_quadratic(none, g, step, nullptr);
        }
        return {step, cost_change};
    }
};

// givens_elimination(S, v, diag): zero out diag(diag) below the upper triangular S (n x n,
// C-ordered) by Givens rotations (LAPACK dlartg, BLAS drot), applied to v as well
void givens_elimination(lint n, double* S, double* v, const double* diag) {
    vec diag_row(static_cast<std::size_t>(n));
    const lint one = 1;
    for (lint i = 0; i < n; ++i) {
        if (diag[i] == 0) continue;
        for (lint k = i + 1; k < n; ++k) diag_row[k] = 0;
        diag_row[i] = diag[i];
        double u = 0;
        for (lint j = i; j < n; ++j) {
            if (diag_row[j] == 0) continue;
            double f = S[j * n + j], g = diag_row[j], cs, sn, r;
            dlartg_(&f, &g, &cs, &sn, &r);
            S[j * n + j] = r;
            lint k = n - j - 1;
            if (k > 0) drot_(&k, &S[j * n + j + 1], &one, &diag_row[j + 1], &one, &cs, &sn);
            f = v[j];
            v[j] = std::fma(cs, f, sn * u);
            u = std::fma(-sn, f, cs * u);
        }
    }
}

// trf_linear; x_lsq outside the bounds. Returns the status; x, r, g, cost, optimality, n_iter.
int trf_linear(
    Problem& P,
    const vec& x_lsq,
    const vec& lb,
    const vec& ub,
    double tol,
    lint max_iter,
    vec& x,
    vec& r,
    vec& g,
    double& cost,
    double& g_norm,
    lint& n_iter
) {
    const lint m = P.m, n = P.n, k = std::min(m, n);
    const auto un = static_cast<std::size_t>(n);
    x = make_strictly_feasible(reflect(x_lsq, lb, ub), lb, ub, 0.1);
    // QT, R, perm = scipy.linalg.qr(A, mode='economic', pivoting=True)
    vec qr(static_cast<std::size_t>(m * n)), tau(static_cast<std::size_t>(k));
    std::vector<lint> jpvt(un, 0);
    for (lint i = 0; i < m; ++i)
        for (lint j = 0; j < n; ++j) qr[i + j * m] = P.A[i * n + j];
    vec R(un * un, 0.0),
        Q(static_cast<std::size_t>(m * k));  // R: n x n C-ordered (zero rows when m < n); Q: C-ordered (m, k)
    {
        lapack::geqp3(m, n, qr.data(), m, jpvt.data(), tau.data(), P.sc);
        for (lint i = 0; i < k; ++i)
            for (lint j = i; j < n; ++j) R[i * n + j] = qr[i + j * m];
        lapack::orgqr(m, k, k, qr.data(), m, tau.data(), P.sc);
        for (lint i = 0; i < m; ++i)
            for (lint j = 0; j < k; ++j) Q[i * k + j] = qr[i + j * m];
    }
    std::vector<lint> perm(un);
    for (lint j = 0; j < n; ++j) perm[j] = jpvt[j] - 1;

    r.assign(static_cast<std::size_t>(m), 0.0);
    g.assign(un, 0.0);
    cost = P.residual(x, r, &g);
    if (max_iter < 0) max_iter = 100;
    int status = -2;  // None
    vec v(un), dv(un), d(un), g_h(un), diag_h(un), diag_root_h(un), QTr(un, 0.0), Rd(un * un), vv(un), xs(un), p_h(un),
        p(un), drp(un), y(un);
    Trf T(P, lb, ub);
    static const vec none;
    lint iteration = 0;
    for (iteration = 0; iteration < max_iter; ++iteration) {
        for (lint j = 0; j < n; ++j) {  // CL_scaling_vector
            v[j] = 1;
            dv[j] = 0;
            if (g[j] < 0 && std::isfinite(ub[j])) {
                v[j] = ub[j] - x[j];
                dv[j] = -1;
            }
            if (g[j] > 0 && std::isfinite(lb[j])) {
                v[j] = x[j] - lb[j];
                dv[j] = 1;
            }
        }
        for (lint j = 0; j < n; ++j) y[j] = std::abs(g[j] * v[j]);
        g_norm = npmax(y.data(), n);
        if (g_norm < tol) status = 1;
        if (status != -2) break;
        for (lint j = 0; j < n; ++j) {
            diag_h[j] = g[j] * dv[j];
            diag_root_h[j] = std::sqrt(diag_h[j]);
            d[j] = std::sqrt(v[j]);
            g_h[j] = d[j] * g[j];
        }
        // QTr[:k] = QT.dot(r) (QT: (k, m) Fortran-ordered)
        if (k == 1) {
            QTr[0] = m == 1 ? Q[0] * r[0] : 0.0 + cblas_ddot(m, Q.data(), 1, r.data(), 1);
        } else {
            np_rmatvec(Q.data(), m, k, r.data(), QTr.data());
        }
        // p_h = -regularized_lsq_with_qr(m, n, R * d[perm], QTr, perm, diag_root_h)
        for (lint i = 0; i < n; ++i)
            for (lint j = 0; j < n; ++j) Rd[i * n + j] = R[i * n + j] * d[perm[j]];
        vv = QTr;
        for (lint j = 0; j < n; ++j) drp[j] = diag_root_h[perm[j]];
        givens_elimination(n, Rd.data(), vv.data(), drp.data());
        double mx = std::abs(Rd[0]);
        for (lint j = 1; j < n; ++j) {
            const double a = std::abs(Rd[j * n + j]);
            if (std::isnan(mx) || !(a <= mx)) mx = std::isnan(mx) ? mx : a;
        }
        const double threshold = kEps * static_cast<double>(std::max(m, n)) * mx;
        std::vector<lint> nns;
        for (lint j = 0; j < n; ++j)
            if (std::abs(Rd[j * n + j]) > threshold) nns.push_back(j);
        std::fill(xs.begin(), xs.end(), 0.0);
        const lint kk = static_cast<lint>(nns.size());
        if (kk > 0) {
            vec Rs(static_cast<std::size_t>(kk * kk)), bs(static_cast<std::size_t>(kk));
            for (lint a = 0; a < kk; ++a) {
                bs[a] = vv[nns[a]];
                for (lint c = 0; c < kk; ++c) Rs[a * kk + c] = Rd[nns[a] * n + nns[c]];
            }
            // scipy's solve_triangular: the C-ordered R read as Fortran R^T, lower, transposed
            // (a 1 x 1 matrix is Fortran-ordered too: upper, not transposed; the same)
            lapack::trtrs('L', 'T', 'N', kk, 1, Rs.data(), kk, bs.data(), kk);
            for (lint a = 0; a < kk; ++a) xs[perm[nns[a]]] = bs[a];
        }
        for (lint j = 0; j < n; ++j) p_h[j] = -xs[j];
        for (lint j = 0; j < n; ++j) p[j] = d[j] * p_h[j];
        const double p_dot_g = npdot(p.data(), g.data(), n);
        if (p_dot_g > 0) status = -1;
        const double theta = 1 - std::min(0.005, g_norm);
        vec step = T.select_step(x, g_h, diag_h, p, p_h, d, theta);
        double cost_change = -T.evaluate_quadratic(none, g, step, nullptr);
        if (cost_change < 0) {
            // scipy's backtracking returns the x it was given: the iterate stays
            auto [st, cc] = T.backtracking(g, x, p, theta, p_dot_g);
            step = std::move(st);
            cost_change = cc;
        } else {
            for (lint j = 0; j < n; ++j) y[j] = x[j] + step[j];
            x = make_strictly_feasible(y, lb, ub, 0);
        }
        np_matvec(P.A.data(), m, n, x.data(), r.data());
        for (lint i = 0; i < m; ++i) r[i] = r[i] - P.b[i];
        np_rmatvec(P.A.data(), m, n, r.data(), g.data());
        if (cost_change < tol * cost) status = 2;
        cost = 0.5 * npdot(r.data(), r.data(), m);
    }
    n_iter = iteration < max_iter ? iteration + 1 : max_iter;
    return status == -2 ? 0 : status;
}

// bvls (scipy/optimize/_lsq/bvls.py); x_lsq outside the bounds. Returns the status; x, r, g,
// cost, the KKT optimality, on_bound (the active mask), n_iter.
int bvls(
    Problem& P,
    const vec& x_lsq,
    const vec& lb,
    const vec& ub,
    double tol,
    lint max_iter,
    vec& x,
    vec& r,
    vec& g,
    double& cost,
    double& optimality,
    std::vector<int>& on_bound,
    lint& n_iter
) {
    const lint m = P.m, n = P.n;
    const auto un = static_cast<std::size_t>(n);
    x = x_lsq;
    on_bound.assign(un, 0);
    for (lint j = 0; j < n; ++j)
        if (x[j] <= lb[j]) {
            x[j] = lb[j];
            on_bound[j] = -1;
        }
    for (lint j = 0; j < n; ++j)
        if (x[j] >= ub[j]) {
            x[j] = ub[j];
            on_bound[j] = 1;
        }
    std::vector<char> active(un);
    std::vector<lint> free_set;
    for (lint j = 0; j < n; ++j) {
        active[j] = on_bound[j] != 0;
        if (!active[j]) free_set.push_back(j);
    }
    r.assign(static_cast<std::size_t>(m), 0.0);
    g.assign(un, 0.0);
    cost = P.residual(x, r, &g);
    vec xa(un), bf(static_cast<std::size_t>(m)), z(un), Af;
    // b - A.dot(x * active_set)
    auto b_free = [&]() {
        for (lint j = 0; j < n; ++j) xa[j] = x[j] * (active[j] ? 1.0 : 0.0);
        np_matvec(P.A.data(), m, n, xa.data(), bf.data());
        for (lint i = 0; i < m; ++i) bf[i] = P.b[i] - bf[i];
    };
    // A[:, free] and numpy's lstsq of it (rcond None: eps max(m, k)), z (the free variables')
    auto solve_free = [&](const std::vector<lint>& fs) {
        const lint kf = static_cast<lint>(fs.size());
        Af.resize(static_cast<std::size_t>(m * kf));
        for (lint i = 0; i < m; ++i)
            for (lint a = 0; a < kf; ++a) Af[i * kf + a] = P.A[i * n + fs[a]];
        b_free();
        P.lstsq(Af.data(), m, kf, bf.data(), kEps * static_cast<double>(std::max(m, kf)), z.data());
    };
    lint iteration = 0;
    while (!free_set.empty()) {
        ++iteration;
        solve_free(free_set);
        const lint kf = static_cast<lint>(free_set.size());
        std::vector<char> viol(static_cast<std::size_t>(kf), 0);
        bool any = false;
        for (lint a = 0; a < kf; ++a) {
            const lint j = free_set[a];
            const bool lbv = z[a]<lb[j], ubv = z[a]> ub[j];
            if (lbv) {
                x[j] = lb[j];
                active[j] = 1;
                on_bound[j] = -1;
            }
            viol[a] = lbv || ubv;
            any = any || viol[a];
        }
        for (lint a = 0; a < kf; ++a) {
            const lint j = free_set[a];
            if (z[a] > ub[j]) {
                x[j] = ub[j];
                active[j] = 1;
                on_bound[j] = 1;
            }
        }
        for (lint a = 0; a < kf; ++a)
            if (!viol[a]) x[free_set[a]] = z[a];
        np_matvec(P.A.data(), m, n, x.data(), r.data());
        for (lint i = 0; i < m; ++i) r[i] = r[i] - P.b[i];
        cost = 0.5 * npdot(r.data(), r.data(), m);
        np_rmatvec(P.A.data(), m, n, r.data(), g.data());
        if (!any) break;
        std::vector<lint> kept;
        for (lint a = 0; a < kf; ++a)
            if (!viol[a]) kept.push_back(free_set[a]);
        free_set = std::move(kept);
    }
    if (max_iter < 0) max_iter = n;
    max_iter += iteration;
    int status = -2;
    vec t(un);
    auto kkt = [&]() {  // compute_kkt_optimality
        for (lint j = 0; j < n; ++j) t[j] = on_bound[j] == 0 ? std::abs(g[j]) : g[j] * on_bound[j];
        return npmax(t.data(), n);
    };
    optimality = kkt();
    lint it = iteration;
    for (it = iteration; it < max_iter; ++it) {
        if (optimality < tol) status = 1;
        if (status != -2) break;
        for (lint j = 0; j < n; ++j) t[j] = g[j] * on_bound[j];
        on_bound[npargmax(t.data(), n)] = 0;
        vec x_free;
        std::vector<lint> fs;
        for (;;) {
            fs.clear();
            for (lint j = 0; j < n; ++j) {
                active[j] = on_bound[j] != 0;
                if (!active[j]) fs.push_back(j);
            }
            const lint kf = static_cast<lint>(fs.size());
            x_free.resize(static_cast<std::size_t>(kf));
            for (lint a = 0; a < kf; ++a) x_free[a] = x[fs[a]];
            solve_free(fs);
            std::vector<lint> vidx;  // lbv then ubv (positions in the free set)
            lint n_lbv = 0;
            for (lint a = 0; a < kf; ++a)
                if (z[a] < lb[fs[a]]) vidx.push_back(a);
            n_lbv = static_cast<lint>(vidx.size());
            for (lint a = 0; a < kf; ++a)
                if (z[a] > ub[fs[a]]) vidx.push_back(a);
            if (!vidx.empty()) {
                vec alphas(vidx.size());
                for (std::size_t q = 0; q < vidx.size(); ++q) {
                    const lint a = vidx[q];
                    const double num = static_cast<lint>(q) < n_lbv ? lb[fs[a]] - x_free[a] : ub[fs[a]] - x_free[a];
                    alphas[q] = num / (z[a] - x_free[a]);
                }
                const lint i = npargmin(alphas.data(), static_cast<lint>(alphas.size()));
                const lint i_free = vidx[static_cast<std::size_t>(i)];
                const double alpha = alphas[static_cast<std::size_t>(i)];
                for (lint a = 0; a < kf; ++a) x_free[a] = x_free[a] * (1 - alpha);
                for (lint a = 0; a < kf; ++a) x_free[a] = x_free[a] + alpha * z[a];
                for (lint a = 0; a < kf; ++a) x[fs[a]] = x_free[a];
                on_bound[fs[i_free]] = i < n_lbv ? -1 : 1;
            } else {
                for (lint a = 0; a < kf; ++a) x[fs[a]] = z[a];
                break;
            }
        }
        np_matvec(P.A.data(), m, n, x.data(), r.data());
        for (lint i = 0; i < m; ++i) r[i] = r[i] - P.b[i];
        const double cost_new = 0.5 * npdot(r.data(), r.data(), m);
        const double cost_change = cost - cost_new;
        if (cost_change < tol * cost) status = 2;
        cost = cost_new;
        np_rmatvec(P.A.data(), m, n, r.data(), g.data());
        optimality = kkt();
    }
    // scipy's nit = iteration + 1 with Python's loop variable: the last value it took
    const lint last = it < max_iter ? it : (max_iter > iteration ? max_iter - 1 : iteration);
    n_iter = last + 1;
    return status == -2 ? 0 : status;
}

// ---- nnls: scipy 1.18.1's __nnls (Lawson and Hanson, LAPACK reflections and rotations)

// The work buffer of scipy's _nnls wrapper: x (n), A (m x n, column-major), b (m), w (n), zz (m)
struct NnlsWork {
    vec buf;
    lint m, n;
    NnlsWork(lint m_, lint n_) : buf(static_cast<std::size_t>((n_ + 2) * m_ + 3 * n_)), m(m_), n(n_) {}
    double* x() { return buf.data(); }
    double* a() { return buf.data() + n; }
    double* b() { return buf.data() + n + m * n; }
    double* w() { return buf.data() + n + m * n + m; }
    double* zz() { return buf.data() + 2 * n + m * n + m; }
};

// Returns info: 1 solved, 3 the iteration cap. iters: scipy's iteration count.
int nnls_kernel(NnlsWork& W, lint maxiter, lint& iters) {
    const lint m = W.m, n = W.n;
    double *x = W.x(), *a = W.a(), *b = W.b(), *w = W.w(), *zz = W.zz();
    std::vector<lint> idx(static_cast<std::size_t>(n));
    const lint one = 1;
    const char L = 'L';
    for (lint i = 0; i < n; ++i) idx[i] = i;
    for (lint i = 0; i < n; ++i) x[i] = 0;
    lint nsetp = 0, iter = 0, izmax = 0;
    double tau = 0, beta = 0, work = 0;
    int info = 1;
    const lint mn = std::min(m, n);
    // solve the triangular system for zz[0:nsetp] (A's columns idx[0:nsetp], rows 0:nsetp)
    auto solve = [&]() {
        lint jj = 0;
        for (lint k = 0; k < nsetp; ++k) {
            const lint i = nsetp - 1 - k;
            if (k != 0)
                for (lint t = 0; t <= i; ++t) zz[t] = std::fma(-a[t + jj * m], zz[i + 1], zz[t]);
            jj = idx[i];
            zz[i] = zz[i] / a[i + jj * m];
        }
    };
    while (nsetp < mn) {
        // the dual w = A^T b on the rows nsetp.. for the columns not in P
        for (lint iz = nsetp; iz < n; ++iz) {
            const lint j = idx[iz];
            w[j] = cblas_ddot(m - nsetp, &a[nsetp + j * m], 1, &b[nsetp], 1);
        }
        // take in the column with the largest dual that passes the tests
        bool taken = false;
        for (;;) {
            if (nsetp >= n) break;
            double wmax = 0;
            for (lint iz = nsetp; iz < n; ++iz) {
                const double wj = w[idx[iz]];
                if (wj > wmax) {
                    wmax = wj;
                    izmax = iz;
                }
            }
            if (!(wmax > 0)) break;
            const lint j = idx[izmax];
            const lint nn = m - nsetp;
            beta = a[nsetp + j * m];
            dlarfgp_(&nn, &beta, &a[nsetp + 1 + j * m], &one, &tau);
            double spacing = 0;
            if (nsetp >= 1) {
                const double unorm = cblas_dnrm2(nsetp, &a[j * m], 1);
                if (unorm > 0) spacing = std::nextafter(unorm, 2 * unorm) - unorm;
            }
            if (!(std::abs(beta) > 100 * spacing)) {
                w[j] = 0;
                continue;
            }
            std::memcpy(zz, b, static_cast<std::size_t>(m) * sizeof(double));
            const double save = a[nsetp + j * m];
            a[nsetp + j * m] = 1;
            dlarf_(&L, &nn, &one, &a[nsetp + j * m], &one, &tau, &zz[nsetp], &nn, &work);
            if (!(zz[nsetp] / beta > 0)) {
                a[nsetp + j * m] = save;
                w[j] = 0;
                continue;
            }
            std::memcpy(b, zz, static_cast<std::size_t>(m) * sizeof(double));
            idx[izmax] = idx[nsetp];
            idx[nsetp] = j;
            const lint old = nsetp;
            ++nsetp;
            for (lint k = nsetp; k < n; ++k)
                dlarf_(&L, &nn, &one, &a[old + j * m], &one, &tau, &a[old + idx[k] * m], &nn, &work);
            a[old + j * m] = beta;
            for (lint i = nsetp; i < m; ++i) a[i + j * m] = 0;
            w[j] = 0;
            taken = true;
            break;
        }
        if (!taken) break;
        solve();
        // the inner loop: move to the solution on P, dropping the columns that reach 0
        for (;;) {
            ++iter;
            if (iter >= maxiter) {
                info = 3;
                goto done;
            }
            if (nsetp < 1) break;
            double alpha = 2;
            lint jj = 0;
            for (lint ip = 0; ip < nsetp; ++ip) {
                if (!(zz[ip] <= 0)) continue;
                const lint l = idx[ip];
                const double t = -x[l] / (zz[ip] - x[l]);
                if (alpha > t) {
                    alpha = t;
                    jj = ip;
                }
            }
            if (alpha == 2) {
                for (lint ip = 0; ip < nsetp; ++ip) x[idx[ip]] = zz[ip];
                break;
            }
            for (lint ip = 0; ip < nsetp; ++ip) {
                const lint l = idx[ip];
                x[l] = std::fma(alpha, zz[ip] - x[l], x[l]);
            }
            lint i = idx[jj];
            for (;;) {
                x[i] = 0;
                for (lint k = jj + 1; k < nsetp; ++k) {
                    const lint ii = idx[k];
                    idx[k - 1] = ii;
                    double cs, sn;
                    dlartgp_(&a[k - 1 + ii * m], &a[k + ii * m], &cs, &sn, &a[k - 1 + ii * m]);
                    a[k + ii * m] = 0;
                    for (lint l = 0; l < n; ++l) {
                        if (l == ii) continue;
                        const double t1 = a[k - 1 + l * m], t2 = a[k + l * m];
                        a[k - 1 + l * m] = std::fma(cs, t1, sn * t2);
                        a[k + l * m] = std::fma(-sn, t1, cs * t2);
                    }
                    const double t1 = b[k - 1], t2 = b[k];
                    b[k - 1] = std::fma(cs, t1, sn * t2);
                    b[k] = std::fma(-sn, t1, cs * t2);
                }
                --nsetp;
                idx[nsetp] = i;
                // a column of P whose value is not positive (roundoff) leaves too
                bool again = false;
                for (lint ip = 0; ip < nsetp; ++ip)
                    if (x[idx[ip]] <= 0) {
                        jj = ip;
                        i = idx[ip];
                        again = true;
                        break;
                    }
                if (!again) break;
            }
            std::memcpy(zz, b, static_cast<std::size_t>(m) * sizeof(double));
            solve();
        }
    }
done:
    iters = iter;
    return info;
}

nb::object vector_out(const double* v, lint n) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(n)};
    NDArray a = new_array(DType::Float64, shape);
    if (n) std::memcpy(a.data, v, static_cast<std::size_t>(n) * sizeof(double));
    return wrap(std::move(a));
}

// A (float64 (m, n)) and b (float64 (m,)) into P
void load(Problem& P, nb::handle A, nb::handle b, const char* fn) {
    Input ia(A, fn, "A"), ib(b, fn, "b");
    P.m = ia.shape(0);
    P.n = ia.shape(1);
    P.A.resize(static_cast<std::size_t>(P.m * P.n));
    P.b.resize(static_cast<std::size_t>(P.m));
    const std::int64_t s0 = ia.stride(0), s1 = ia.stride(1), sb = ib.stride(0);
    for (lint i = 0; i < P.m; ++i) {
        for (lint j = 0; j < P.n; ++j) std::memcpy(&P.A[i * P.n + j], ia.data() + i * s0 + j * s1, sizeof(double));
        std::memcpy(&P.b[i], ib.data() + i * sb, sizeof(double));
    }
    for (double v : P.A)
        if (!std::isfinite(v)) throw nb::value_error("'A' must be finite");
    for (double v : P.b)
        if (!std::isfinite(v)) throw nb::value_error("'b' must be finite");
}

// lsq_linear_solve(func, A, b, lb, ub, method, tol, max_iter)
//   -> (x (n,), r (m,), cost, optimality, active_mask [n], status, n_iter)
// method 0 'trf', 1 'bvls'; max_iter -1: scipy's default (trf 100, bvls n).
nb::object lsq_linear_solve(
    const std::string& func,
    nb::handle A,
    nb::handle b,
    const vec& lb,
    const vec& ub,
    int method,
    double tol,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Problem P;
        load(P, A, b, fn);
        const lint m = P.m, n = P.n;
        vec x, r, g;
        double cost = 0, optimality = 0;
        std::vector<int> active;
        lint n_iter = 0;
        int status;
        {
            std::optional<nb::gil_scoped_release> rel;
            if (2.0 * double(m) * double(n) * double(n) > 2e5) rel.emplace();
            // the unbounded solution: numpy's lstsq(A, b, rcond=-1)
            vec x_lsq(static_cast<std::size_t>(n));
            P.lstsq(P.A.data(), m, n, P.b.data(), -1.0, x_lsq.data());
            if (in_bounds(x_lsq, lb, ub)) {
                x = x_lsq;
                r.assign(static_cast<std::size_t>(m), 0.0);
                g.assign(static_cast<std::size_t>(n), 0.0);
                cost = P.residual(x, r, &g);
                for (lint j = 0; j < n; ++j) g[j] = std::abs(g[j]);
                optimality = npmax(g.data(), n);
                active.assign(static_cast<std::size_t>(n), 0);
                status = 3;
            } else if (method == 0) {
                status = trf_linear(P, x_lsq, lb, ub, tol, max_iter, x, r, g, cost, optimality, n_iter);
                active = find_active(x, lb, ub, tol);
            } else {
                status = bvls(P, x_lsq, lb, ub, tol, max_iter, x, r, g, cost, optimality, active, n_iter);
            }
        }
        nb::list am;
        for (int v : active) am.append(v);
        return nb::make_tuple(
            vector_out(x.data(), n), vector_out(r.data(), m), cost, optimality, am, status,
            static_cast<std::int64_t>(n_iter)
        );
    });
}

// nnls_solve(func, A, b, max_iter) -> (x (n,), r (m,), cost, info, n_iter); info 1 solved, 3 the
// iteration cap.
nb::object nnls_solve(const std::string& func, nb::handle A, nb::handle b, std::int64_t max_iter) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Problem P;
        load(P, A, b, fn);
        const lint m = P.m, n = P.n;
        NnlsWork W(m, n);
        for (lint i = 0; i < m; ++i)
            for (lint j = 0; j < n; ++j) W.a()[i + j * m] = P.A[i * n + j];
        std::memcpy(W.b(), P.b.data(), static_cast<std::size_t>(m) * sizeof(double));
        lint iters = 0;
        int info;
        vec x(static_cast<std::size_t>(n)), r(static_cast<std::size_t>(m));
        double cost;
        {
            std::optional<nb::gil_scoped_release> rel;
            if (2.0 * double(m) * double(n) * double(n) > 2e5) rel.emplace();
            info = nnls_kernel(W, max_iter, iters);
            std::memcpy(x.data(), W.x(), static_cast<std::size_t>(n) * sizeof(double));
            cost = P.residual(x, r, nullptr);
        }
        return nb::make_tuple(
            vector_out(x.data(), n), vector_out(r.data(), m), cost, info, static_cast<std::int64_t>(iters)
        );
    });
}

}  // namespace

void bind_linear(nb::module_& m) {
    m.def(
        "lsq_linear_solve", &lsq_linear_solve, nb::arg("func"), nb::arg("A"), nb::arg("b"), nb::arg("lb"),
        nb::arg("ub"), nb::arg("method"), nb::arg("tol"), nb::arg("max_iter")
    );
    m.def("nnls_solve", &nnls_solve, nb::arg("func"), nb::arg("A"), nb::arg("b"), nb::arg("max_iter"));
}

}  // namespace sd::optimize
