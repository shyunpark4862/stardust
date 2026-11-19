// minimize(method='bfgs') (docs/design/optimize.md §3.5): scipy 1.18.1's _minimize_bfgs, the same
// steps (O9).
//
// The inverse Hessian approximation H starts at the identity (or hess_inv0); each iteration
// searches along p = -H g with scipy's _line_search_wolfe12: line_search_wolfe1 (MINPACK-2's
// dcsrch, scipy's Python DCSRCH and dcstep; c1, c2, xtol 1e-14, steps in [1e-100, 1e100], 100
// trials), and when that finds no step, line_search_wolfe2 (Nocedal and Wright's algorithms 3.5
// and 3.6: doubling, then zoom with cubic and quadratic interpolation, 10 trials each); no step
// from either stops with status 2. Then H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T as
// scipy forms it: the two matrix products by BLAS (dgemm; the GIL is let go for them by the
// 10 us rule). It stops when the gradient's norm ('norm', the largest |g_i| by default) is at
// most gtol, at max_iter iterations, when the step is below xrtol (|x| + xrtol), or when f is not
// finite.
//
// The arithmetic is scipy's (numpy's, without fused multiply-adds): the dot products and the
// matrix-vector product are numpy's BLAS calls (Accelerate's cblas_ddot, cblas_dgemv,
// cblas_dgemm, which give numpy's bits), the 2-norms numpy's pairwise sums. The interpolations
// of the wolfe2 zoom run under np.errstate(raise) in scipy: a division by zero, an overflow or an
// invalid operation there gives up the interpolation; here each operation is checked for the
// same (numpy's rules for the flags), and the 2 x 2 product is numpy's dgemv.
#include "optimize/minimize.hpp"

#include "optimize/objective.hpp"

#include "bind/errors.hpp"
#include "linalg/types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sd::optimize {

namespace {

using vec = std::vector<double>;
using linalg::lint;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

enum Warn : int { wSuccess = 0, wMaxIter = 1, wPrLoss = 2, wNaN = 3 };

double dot(const vec& a, const vec& b) { return cblas_ddot(static_cast<lint>(a.size()), a.data(), 1, b.data(), 1); }

// numpy's pairwise sum of a contiguous double array (the add reduction, from +0.0).
double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + static_cast<std::size_t>(j)];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// scipy's vecnorm(x, ord): max |x_i| (inf), min |x_i| (-inf), else sum |x_i|^ord ^ (1 / ord)
// with numpy's power (its fast paths for 2, 0.5, 1, -1, 0) and pairwise sum.
double vecnorm(const vec& x, double ord, vec& work) {
    const std::size_t n = x.size();
    if (ord == kInf || ord == -kInf) {
        double m = std::abs(x[0]);
        for (std::size_t i = 1; i < n; ++i) {
            const double a = std::abs(x[i]);
            if (m != m || a != a) m = kNaN;
            else if (ord > 0 ? a > m : a < m) m = a;
        }
        return m;
    }
    work.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = std::abs(x[i]);
        if (ord == 2) work[i] = a * a;
        else if (ord == 0.5) work[i] = std::sqrt(a);
        else if (ord == 1) work[i] = a;
        else if (ord == -1) work[i] = 1.0 / a;
        else if (ord == 0) work[i] = 1.0;
        else work[i] = std::pow(a, ord);
    }
    return std::pow(pairwise_sum(work.data(), n), 1.0 / ord);
}

// Python's max and min of numbers: the first argument unless a later one compares greater
// (smaller); NaN wins only from the front.
inline double pymax(double a, double b) { return b > a ? b : a; }
inline double pymin(double a, double b) { return b < a ? b : a; }
inline double pymax3(double a, double b, double c) { return pymax(pymax(a, b), c); }
inline double np_sign(double x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : (x == 0 ? 0.0 : kNaN)); }
inline double np_clip(double x, double lo, double hi) {
    if (x != x) return x;
    const double t = x > lo ? x : lo;
    return t < hi ? t : hi;
}

// ---- scipy's _dcsrch.py (DCSRCH and dcstep, in Python: numpy's arithmetic)

void dcstep(
    double& stx,
    double& fx,
    double& dx,
    double& sty,
    double& fy,
    double& dy,
    double& stp,
    double fp,
    double dp,
    bool& brackt,
    double stpmin,
    double stpmax
) {
    const double sgnd = np_sign(dp) * np_sign(dx);
    double stpf, stpc, stpq, theta, s, gamma, p, q, r;
    if (fp > fx) {
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        s = pymax3(std::abs(theta), std::abs(dx), std::abs(dp));
        gamma = s * std::sqrt(std::pow(theta / s, 2) - (dx / s) * (dp / s));
        if (stp < stx) gamma *= -1;
        p = (gamma - dx) + theta;
        q = ((gamma - dx) + gamma) + dp;
        r = p / q;
        stpc = stx + r * (stp - stx);
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
        if (std::abs(stpc - stx) <= std::abs(stpq - stx)) stpf = stpc;
        else stpf = stpc + (stpq - stpc) / 2.0;
        brackt = true;
    } else if (sgnd < 0.0) {
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
        s = pymax3(std::abs(theta), std::abs(dx), std::abs(dp));
        gamma = s * std::sqrt(std::pow(theta / s, 2) - (dx / s) * (dp / s));
        if (stp > stx) gamma *= -1;
        p = (gamma - dp) + theta;
        q = ((gamma - dp) + gamma) + dx;
        r = p / q;
        stpc = stp + r * (stx - stp);
        stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (std::abs(stpc - stp) > std::abs(stpq - stp)) stpf = stpc;
        else stpf = stpq;
        brackt = true;
    } else if (std::abs(dp) < std::abs(dx)) {
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
        s = pymax3(std::abs(theta), std::abs(dx), std::abs(dp));
        const double rad = std::pow(theta / s, 2) - (dx / s) * (dp / s);
        gamma = s * std::sqrt(rad > 0 ? rad : 0.0);  // Python's max(0, rad)
        if (stp > stx) gamma = -gamma;
        p = (gamma - dp) + theta;
        q = (gamma + (dx - dp)) + gamma;
        r = p / q;
        if (r < 0 && gamma != 0) stpc = stp + r * (stx - stp);
        else if (stp > stx) stpc = stpmax;
        else stpc = stpmin;
        stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (brackt) {
            if (std::abs(stpc - stp) < std::abs(stpq - stp)) stpf = stpc;
            else stpf = stpq;
            if (stp > stx) stpf = pymin(stp + 0.66 * (sty - stp), stpf);
            else stpf = pymax(stp + 0.66 * (sty - stp), stpf);
        } else {
            if (std::abs(stpc - stp) > std::abs(stpq - stp)) stpf = stpc;
            else stpf = stpq;
            stpf = np_clip(stpf, stpmin, stpmax);
        }
    } else {
        if (brackt) {
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
            s = pymax3(std::abs(theta), std::abs(dy), std::abs(dp));
            gamma = s * std::sqrt(std::pow(theta / s, 2) - (dy / s) * (dp / s));
            if (stp > sty) gamma = -gamma;
            p = (gamma - dp) + theta;
            q = ((gamma - dp) + gamma) + dy;
            r = p / q;
            stpc = stp + r * (sty - stp);
            stpf = stpc;
        } else if (stp > stx) {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }
    if (fp > fx) {
        sty = stp;
        fy = fp;
        dy = dp;
    } else {
        if (sgnd < 0) {
            sty = stx;
            fy = fx;
            dy = dx;
        }
        stx = stp;
        fx = fp;
        dx = dp;
    }
    stp = stpf;
}

enum Task { tStart, tFG, tConv, tWarn, tError };

struct Dcsrch {
    double ftol, gtol, xtol, stpmin, stpmax;
    bool brackt = false;
    int stage = 1;
    double ginit = 0, gtest = 0, gx = 0, gy = 0, finit = 0, fx = 0, fy = 0, stx = 0, sty = 0, stmin = 0, stmax = 0;
    double width = 0, width1 = 0;

    // DCSRCH._iterate
    Task iterate(double& stp, double f, double g, Task task) {
        const double p5 = 0.5, p66 = 0.66, xtrapl = 1.1, xtrapu = 4.0;
        if (task == tStart) {
            bool err = stp < stpmin || stp > stpmax || g >= 0 || ftol < 0 || gtol < 0 || xtol < 0 || stpmin < 0 ||
                       stpmax < stpmin;
            if (err) return tError;
            brackt = false;
            stage = 1;
            finit = f;
            ginit = g;
            gtest = ftol * ginit;
            width = stpmax - stpmin;
            width1 = width / p5;
            stx = 0.0;
            fx = finit;
            gx = ginit;
            sty = 0.0;
            fy = finit;
            gy = ginit;
            stmin = 0;
            stmax = stp + xtrapu * stp;
            return tFG;
        }
        const double ftest = finit + stp * gtest;
        if (stage == 1 && f <= ftest && g >= 0) stage = 2;
        Task out = tFG;
        bool warn = false;
        if (brackt && (stp <= stmin || stp >= stmax)) warn = true;
        if (brackt && stmax - stmin <= xtol * stmax) warn = true;
        if (stp == stpmax && f <= ftest && g <= gtest) warn = true;
        if (stp == stpmin && (f > ftest || g >= gtest)) warn = true;
        if (warn) out = tWarn;
        if (f <= ftest && std::abs(g) <= gtol * -ginit) out = tConv;
        if (out == tWarn || out == tConv) return out;
        if (stage == 1 && f <= fx && f > ftest) {
            const double fm = f - stp * gtest;
            double fxm = fx - stx * gtest;
            double fym = fy - sty * gtest;
            const double gm = g - gtest;
            double gxm = gx - gtest;
            double gym = gy - gtest;
            dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);
            fx = fxm + stx * gtest;
            fy = fym + sty * gtest;
            gx = gxm + gtest;
            gy = gym + gtest;
        } else {
            dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax);
        }
        if (brackt) {
            if (std::abs(sty - stx) >= p66 * width1) stp = stx + p5 * (sty - stx);
            width1 = width;
            width = std::abs(sty - stx);
        }
        if (brackt) {
            stmin = pymin(stx, sty);
            stmax = pymax(stx, sty);
        } else {
            stmin = stp + xtrapl * (stp - stx);
            stmax = stp + xtrapu * (stp - stx);
        }
        stp = np_clip(stp, stpmin, stpmax);
        if ((brackt && (stp <= stmin || stp >= stmax)) || (brackt && stmax - stmin <= xtol * stmax)) stp = stx;
        return tFG;
    }
};

// ---- the BFGS iterations

struct Bfgs {
    Objective& obj;
    std::size_t n;
    double c1, c2;
    vec xk, pk, gfk, xt, gval, work;

    Bfgs(Objective& o, double c1_, double c2_) : obj(o), n(o.n), c1(c1_), c2(c2_) {}

    // phi(s) = f(xk + s pk); derphi(s) = grad(xk + s pk) . pk (the gradient kept in gval)
    double phi(double s) {
        for (std::size_t i = 0; i < n; ++i) xt[i] = xk[i] + s * pk[i];
        return obj.value(xt.data());
    }
    double derphi(double s) {
        for (std::size_t i = 0; i < n; ++i) xt[i] = xk[i] + s * pk[i];
        gval = obj.gradient(xt.data());
        return dot(gval, pk);
    }

    // line_search_wolfe1 (scalar_search_wolfe1 with DCSRCH): true with the step, its value and
    // gval at it; false when DCSRCH gives no step.
    bool wolfe1(double old_fval, double old_old_fval, double& stp_out, double& fval) {
        const double derphi0 = dot(gfk, pk);
        double alpha1;
        if (derphi0 != 0) {
            alpha1 = pymin(1.0, 1.01 * 2 * (old_fval - old_old_fval) / derphi0);
            if (alpha1 < 0) alpha1 = 1.0;
        } else {
            alpha1 = 1.0;
        }
        gval = gfk;
        Dcsrch ls{c1, c2, 1e-14, 1e-100, 1e100};  // DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)
        double phi1 = old_fval, derphi1 = derphi0;
        Task task = tStart;
        double stp = alpha1;
        int i;
        for (i = 0; i < 100; ++i) {
            stp = alpha1;
            task = ls.iterate(stp, phi1, derphi1, task);
            if (!std::isfinite(stp)) {
                task = tWarn;
                break;
            }
            if (task == tFG) {
                alpha1 = stp;
                phi1 = phi(stp);
                derphi1 = derphi(stp);
            } else {
                break;
            }
        }
        if (i == 100 || task != tConv) return false;  // scipy: no convergence in 100 trials, a warning or an error
        stp_out = stp;
        fval = phi1;
        return true;
    }

    // ---- line_search_wolfe2

    // numpy's errstate(divide, over, invalid = 'raise') for the interpolations: every operation
    // checked; ok turns false at the first flag.
    struct Checked {
        bool ok = true;
        double add(double a, double b) { return flag(a + b, a, b); }
        double sub(double a, double b) { return flag(a - b, a, b); }
        double mul(double a, double b) { return flag(a * b, a, b); }
        double div(double a, double b) {
            const double r = a / b;
            if (b == 0 && std::isfinite(a) && a != 0) ok = false;  // divide
            return flag(r, a, b);
        }
        double pow(double a, double e) {
            const double r = std::pow(a, e);
            return flag(r, a, e);
        }
        double sqrt(double a) {
            if (a < 0) ok = false;
            return std::sqrt(a);
        }
        double flag(double r, double a, double b) {
            if (std::isnan(r) && !std::isnan(a) && !std::isnan(b)) ok = false;  // invalid
            if (std::isinf(r) && std::isfinite(a) && std::isfinite(b)) ok = false;  // overflow (or divide)
            return r;
        }
    };

    // scipy's _cubicmin: the minimizer of the cubic through (a, fa) with slope fpa, (b, fb),
    // (c, fc); false when none (an error flag or a non-finite result).
    static bool cubicmin(double a, double fa, double fpa, double b, double fb, double c, double fc, double& xmin) {
        Checked k;
        const double C = fpa;
        const double db = k.sub(b, a);
        const double dc = k.sub(c, a);
        const double denom = k.mul(k.pow(k.mul(db, dc), 2), k.sub(db, dc));
        double d1[4];
        d1[0] = k.pow(dc, 2);
        d1[1] = -k.pow(db, 2);
        d1[2] = -k.pow(dc, 3);
        d1[3] = k.pow(db, 3);
        double v[2] = {k.sub(k.sub(fb, fa), k.mul(C, db)), k.sub(k.sub(fc, fa), k.mul(C, dc))};
        double AB[2];
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0, d1, 2, v, 1, 0.0, AB, 1);
        double A = k.div(AB[0], denom);
        double B = k.div(AB[1], denom);
        const double radical = k.sub(k.mul(B, B), k.mul(k.mul(3, A), C));
        xmin = k.add(a, k.div(k.add(-B, k.sqrt(radical)), k.mul(3, A)));
        if (!k.ok) return false;
        return std::isfinite(xmin);
    }

    // scipy's _quadmin: the minimizer of the parabola through (a, fa) with slope fpa and (b, fb).
    static bool quadmin(double a, double fa, double fpa, double b, double fb, double& xmin) {
        Checked k;
        const double D = fa;
        const double C = fpa;
        const double db = k.sub(b, k.mul(a, 1.0));
        const double B = k.div(k.sub(k.sub(fb, D), k.mul(C, db)), k.mul(db, db));
        xmin = k.sub(a, k.div(C, k.mul(2.0, B)));
        if (!k.ok) return false;
        return std::isfinite(xmin);
    }

    // scipy's _zoom; false when it finds no step in 10 trials.
    bool zoom(
        double a_lo,
        double a_hi,
        double phi_lo,
        double phi_hi,
        double derphi_lo,
        double phi0,
        double derphi0,
        double& a_star,
        double& val_star
    ) {
        const int maxiter = 10;
        int i = 0;
        const double delta1 = 0.2, delta2 = 0.1;
        double phi_rec = phi0, a_rec = 0;
        double a_j = 0, cchk = 0;
        for (;;) {
            const double dalpha = a_hi - a_lo;
            double a, b;
            if (dalpha < 0) {
                a = a_hi;
                b = a_lo;
            } else {
                a = a_lo;
                b = a_hi;
            }
            bool have = false;
            if (i > 0) {
                cchk = delta1 * dalpha;
                have = cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec, a_j);
            }
            if (i == 0 || !have || a_j > b - cchk || a_j < a + cchk) {
                const double qchk = delta2 * dalpha;
                have = quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_j);
                if (!have || a_j > b - qchk || a_j < a + qchk) a_j = a_lo + 0.5 * dalpha;
            }
            const double phi_aj = phi(a_j);
            if (phi_aj > phi0 + c1 * a_j * derphi0 || phi_aj >= phi_lo) {
                phi_rec = phi_hi;
                a_rec = a_hi;
                a_hi = a_j;
                phi_hi = phi_aj;
            } else {
                const double derphi_aj = derphi(a_j);
                if (std::abs(derphi_aj) <= -c2 * derphi0) {
                    a_star = a_j;
                    val_star = phi_aj;
                    return true;
                }
                if (derphi_aj * (a_hi - a_lo) >= 0) {
                    phi_rec = phi_hi;
                    a_rec = a_hi;
                    a_hi = a_lo;
                    phi_hi = phi_lo;
                } else {
                    phi_rec = phi_lo;
                    a_rec = a_lo;
                }
                a_lo = a_j;
                phi_lo = phi_aj;
                derphi_lo = derphi_aj;
            }
            i += 1;
            if (i > maxiter) return false;
        }
    }

    // line_search_wolfe2 (scalar_search_wolfe2, amax 1e100, 10 trials): 1 with the step, its
    // value and gval at it; 2 with the step but no gradient there (the trials ran out); 0 when it
    // finds none.
    int wolfe2(double old_fval, double old_old_fval, double& alpha_star, double& phi_star) {
        const double amax = 1e100;
        const double derphi0 = dot(gfk, pk);
        const double phi0 = old_fval;
        double alpha0 = 0, alpha1;
        if (derphi0 != 0) alpha1 = pymin(1.0, 1.01 * 2 * (phi0 - old_old_fval) / derphi0);
        else alpha1 = 1.0;
        if (alpha1 < 0) alpha1 = 1.0;
        alpha1 = pymin(alpha1, amax);
        double phi_a1 = phi(alpha1);
        double phi_a0 = phi0, derphi_a0 = derphi0;
        for (int i = 0; i < 10; ++i) {
            if (alpha1 == 0 || alpha0 > amax) return 0;
            const bool not_first = i > 0;
            if (phi_a1 > phi0 + c1 * alpha1 * derphi0 || (phi_a1 >= phi_a0 && not_first))
                return zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi0, derphi0, alpha_star, phi_star) ? 1 : 0;
            const double derphi_a1 = derphi(alpha1);
            if (std::abs(derphi_a1) <= -c2 * derphi0) {
                alpha_star = alpha1;
                phi_star = phi_a1;
                return 1;
            }
            if (derphi_a1 >= 0)
                return zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi0, derphi0, alpha_star, phi_star) ? 1 : 0;
            double alpha2 = 2 * alpha1;
            alpha2 = pymin(alpha2, amax);
            alpha0 = alpha1;
            alpha1 = alpha2;
            phi_a0 = phi_a1;
            phi_a1 = phi(alpha1);
            derphi_a0 = derphi_a1;
        }
        alpha_star = alpha1;
        phi_star = phi_a1;
        return 2;
    }
};

// bfgs_solve(func, fun, jac, check, mode, x0, H0, gtol, norm, steps, lb, ub, max_iter, xrtol, c1, c2)
//   -> (x, fun, jac, hess_inv (n, n), status, n_iter, nfev, njev)
// mode: objective.hpp's GradMode; steps: eps (kGradAbsolute) or rel_step (empty: none); H0:
// hess_inv0 row-major n x n, or empty for the identity. status: scipy's warnflag.
nb::object bfgs_solve(
    const std::string& func,
    nb::callable fun,
    nb::object jac,
    nb::callable check,
    int mode,
    const vec& x0,
    const vec& H0,
    double gtol,
    double norm,
    const vec& steps,
    std::int64_t max_iter,
    double xrtol,
    double c1,
    double c2
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const std::size_t n = x0.size();
        Objective obj;
        obj.fun = fun;
        obj.jac = jac;
        obj.check = check;
        obj.fn = fn;
        obj.n = n;
        obj.mode = mode;
        obj.steps = steps;
        obj.lb.assign(n, -kInf);
        obj.ub.assign(n, kInf);
        obj.start(x0.data());
        obj.require_finite_start();

        Bfgs B(obj, c1, c2);
        B.xt.resize(n);
        double old_fval = obj.value(x0.data());
        B.gfk = obj.gradient(x0.data());
        std::int64_t k = 0;
        const auto ln = static_cast<lint>(n);
        vec Hk(n * n, 0.0), A1(n * n), A2(n * n), T(n * n);
        if (H0.empty()) {
            for (std::size_t i = 0; i < n; ++i) Hk[i * n + i] = 1.0;
        } else {
            Hk = H0;
        }
        const bool big = 2.0 * double(n) * double(n) * double(n) > 2e5;  // the 10 us rule
        double old_old_fval = old_fval + std::sqrt(dot(B.gfk, B.gfk)) / 2;
        B.xk = x0;
        B.pk.resize(n);
        int warnflag = 0;
        vec sk(n), yk(n), xkp1(n), gfkp1;
        double gnorm = vecnorm(B.gfk, norm, B.work);
        while (gnorm > gtol && k < max_iter) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, ln, ln, 1.0, Hk.data(), ln, B.gfk.data(), 1, 0.0, B.pk.data(), 1);
            for (std::size_t i = 0; i < n; ++i) B.pk[i] = -B.pk[i];
            double alpha_k, fval;
            bool have_g = true;
            if (B.wolfe1(old_fval, old_old_fval, alpha_k, fval)) {
                gfkp1 = B.gval;
            } else {
                const int r = B.wolfe2(old_fval, old_old_fval, alpha_k, fval);
                if (r == 0) {
                    warnflag = wPrLoss;
                    break;
                }
                have_g = r == 1;
                if (have_g) gfkp1 = B.gval;
            }
            old_old_fval = old_fval;
            old_fval = fval;
            for (std::size_t i = 0; i < n; ++i) {
                sk[i] = alpha_k * B.pk[i];
                xkp1[i] = B.xk[i] + sk[i];
            }
            B.xk = xkp1;
            if (!have_g) gfkp1 = obj.gradient(xkp1.data());
            for (std::size_t i = 0; i < n; ++i) yk[i] = gfkp1[i] - B.gfk[i];
            B.gfk = gfkp1;
            k += 1;
            gnorm = vecnorm(B.gfk, norm, B.work);
            if (gnorm <= gtol) break;
            if (alpha_k * vecnorm(B.pk, 2, B.work) <= xrtol * (xrtol + vecnorm(B.xk, 2, B.work))) break;
            if (!std::isfinite(old_fval)) {
                warnflag = wPrLoss;
                break;
            }
            const double rhok_inv = dot(yk, sk);
            const double rhok = rhok_inv == 0. ? 1000.0 : 1. / rhok_inv;
            {
                std::optional<nb::gil_scoped_release> rel;
                if (big) rel.emplace();
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        const double I = i == j ? 1.0 : 0.0;
                        A1[i * n + j] = I - sk[i] * yk[j] * rhok;
                        A2[i * n + j] = I - yk[i] * sk[j] * rhok;
                    }
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, ln, ln, ln, 1.0, Hk.data(), ln, A2.data(), ln, 0.0,
                    T.data(), ln
                );
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, ln, ln, ln, 1.0, A1.data(), ln, T.data(), ln, 0.0,
                    Hk.data(), ln
                );
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) Hk[i * n + j] = Hk[i * n + j] + rhok * sk[i] * sk[j];
            }
        }
        const double fval = old_fval;
        if (warnflag == wPrLoss) {
        } else if (k >= max_iter) {
            warnflag = wMaxIter;
        } else {
            bool nan = std::isnan(gnorm) || std::isnan(fval);
            for (double v : B.xk) nan = nan || std::isnan(v);
            if (nan) warnflag = wNaN;
        }
        return nb::make_tuple(
            out_vector(B.xk.data(), n), fval, out_vector(B.gfk.data(), n), out_matrix(Hk.data(), n, n), warnflag, k,
            obj.nfev, obj.ngev
        );
    });
}

}  // namespace

void bind_bfgs(nb::module_& m) {
    m.def(
        "bfgs_solve", &bfgs_solve, nb::arg("func"), nb::arg("fun"), nb::arg("jac").none(), nb::arg("check"),
        nb::arg("mode"), nb::arg("x0"), nb::arg("H0"), nb::arg("gtol"), nb::arg("norm"), nb::arg("steps"),
        nb::arg("max_iter"), nb::arg("xrtol"), nb::arg("c1"), nb::arg("c2")
    );
}

}  // namespace sd::optimize
