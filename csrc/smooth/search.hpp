// The one-variable searches of sd.smooth's tuning parameters (docs/design/smooth.md §3.3; decisions SM14, SM55):
// Brent's bounded minimizer and Brent's root finder on a C++ function, each with a fixed cap on the evaluations.
// The algorithms are optimize's (csrc/optimize/scalar.cpp: scipy's fminbound and brentq.c), written again here
// for a native callback: optimize's kernels take Python callbacks and stay as they are.
//
//   fminbound  scipy's _minimize_scalar_bounded on [a, b]: golden section and parabolic steps, stopping when
//              |x - xm| <= 2 tol1 - (b - a) / 2 with tol1 = sqrt(eps) |x| + xatol / 3, or after maxfun values
//   brentq     scipy's brentq.c on a bracket with f(a) f(b) <= 0: inverse quadratic or linear steps, else
//              bisection; it stops when |f| <= ftol, the bracket is below xtol, or after maxiter values
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sd::smooth {

struct ScalarResult {
    double x = NAN, f = NAN;
    std::int64_t nfev = 0;
    bool converged = false;
};

template <typename F>
ScalarResult fminbound(F&& f, double a, double b, double xatol, std::int64_t maxfun) {
    const double sqrt_eps = std::sqrt(2.2e-16);
    const double golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
    double fulc = a + golden_mean * (b - a);
    double nfc = fulc, xf = fulc;
    double rat = 0.0, e = 0.0;
    double x = xf;
    double fx = f(x);
    std::int64_t num = 1;
    double ffulc = fx, fnfc = fx;
    double xm = 0.5 * (a + b);
    double tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
    double tol2 = 2.0 * tol1;
    bool converged = true;
    auto sign = [](double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); };
    while (std::abs(xf - xm) > (tol2 - 0.5 * (b - a))) {
        bool golden = true;
        if (std::abs(e) > tol1) {
            golden = false;
            double r = (xf - nfc) * (fx - ffulc);
            double q = (xf - fulc) * (fx - fnfc);
            double p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::abs(q);
            r = e;
            e = rat;
            if ((std::abs(p) < std::abs(0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b - xf))) {
                rat = p / q;
                x = xf + rat;
                if (((x - a) < tol2) || ((b - x) < tol2)) {
                    const double si = sign(xm - xf) + ((xm - xf) == 0 ? 1.0 : 0.0);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if (golden) {
            e = xf >= xm ? a - xf : b - xf;
            rat = golden_mean * e;
        }
        const double si = sign(rat) + (rat == 0 ? 1.0 : 0.0);
        x = xf + si * std::max(std::abs(rat), tol1);
        const double fu = f(x);
        num += 1;
        if (fu <= fx) {
            if (x >= xf) a = xf;
            else b = xf;
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if (x < xf) a = x;
            else b = x;
            if ((fu <= fnfc) || (nfc == xf)) {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if (num >= maxfun) {
            converged = false;
            break;
        }
    }
    return {xf, fx, num, converged};
}

// fa = f(a), fb = f(b) already known, of opposite signs (or one of them 0).
template <typename F>
ScalarResult brentq(
    F&& f, double xpre, double xcur, double fpre, double fcur, double xtol, double ftol, std::int64_t maxiter
) {
    double xblk = 0.0, fblk = 0.0, spre = 0.0, scur = 0.0;
    std::int64_t nfev = 0;
    if (fpre == 0) return {xpre, fpre, nfev, true};
    if (fcur == 0) return {xcur, fcur, nfev, true};
    for (std::int64_t i = 0; i < maxiter; ++i) {
        if (fpre != 0 && fcur != 0 && (std::signbit(fpre) != std::signbit(fcur))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (std::abs(fblk) < std::abs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        const double delta = xtol / 2;
        const double sbis = (xblk - xcur) / 2;
        if (std::abs(fcur) <= ftol || std::abs(sbis) < delta) return {xcur, fcur, nfev, true};
        if (std::abs(spre) > delta && std::abs(fcur) < std::abs(fpre)) {
            double stry;
            if (xpre == xblk) {
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                const double dpre = (fpre - fcur) / (xpre - xcur);
                const double dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2 * std::abs(stry) < std::min(std::abs(spre), 3 * std::abs(sbis) - delta)) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if (std::abs(scur) > delta) xcur += scur;
        else xcur += (sbis > 0 ? delta : -delta);
        fcur = f(xcur);
        ++nfev;
    }
    return {xcur, fcur, nfev, false};
}

}  // namespace sd::smooth
