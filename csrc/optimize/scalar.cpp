// root_scalar's and minimize_scalar's iterations (docs/design/optimize.md §3.2, §3.3), each the
// same steps as scipy 1.18.1 (O9) on a function of one real variable computed by a Python
// callback (the loop holds the GIL for it):
//   - brentq: scipy's C brentq (Zeros/brentq.c; Brent 1973: inverse quadratic interpolation,
//     secant and bisection steps);
//   - toms748: scipy's TOMS748Solver (Alefeld, Potra and Shi 1995, algorithm 4.1 for k = 1,
//     4.2 for k = 2), in its order of operations (the divided differences and Neville's
//     interpolation of _zeros_py.py);
//   - bisect: scipy's C bisect (Zeros/bisect.c);
//   - newton, halley, secant: scipy's newton() for a scalar x0; newton without fprime takes
//     the derivative by scipy's '2-point' difference (approx_derivative, as root_scalar does),
//     reusing f at the iterate (scipy computes it twice);
//   - brent: scipy's Brent class (Brent's parabolic interpolation and golden section), with
//     scipy's bracket() for a two-point bracket;
//   - golden: scipy's _minimize_scalar_golden;
//   - bounded: scipy's _minimize_scalar_bounded (fminbound, Brent's fmin).
// A NaN from f stops brentq, toms748 and bisect at that point (scipy's _wrap_nan_raise); the
// other methods go on with it like scipy's. toms748 stops at an infinite value the same way
// (scipy raises ValueError there; O45); brentq and bisect only use its sign.
#include "optimize/scalar.hpp"

#include "bind/errors.hpp"
#include "optimize/diff.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sd::optimize {

namespace nb = nanobind;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// root_scalar's stopping codes (scipy's flags; stardust/optimize/_scalar.py reads them).
enum RootStatus : int {
    rConverged = 0,
    rSignErr = -1,     // f(a) and f(b) have the same sign: a ValueError (O32)
    rConvErr = -2,     // max_iter, a zero derivative, a secant step of zero
    rValueErr = -3,    // f returned NaN (brentq, toms748, bisect) or +-inf (toms748)
};
// Why newton and secant stopped with rConvErr (scipy warns with these).
enum RootReason : int { kMaxIter = 0, kDerivZero = 1, kSecantTol = 2 };

// minimize_scalar's stopping codes (bounded: scipy's flag; brent, golden: the same numbers).
enum MinStatus : int {
    mConverged = 0,
    mMaxIter = 1,
    mNaN = 2,
    mNoBracket = 3,       // scipy's BracketError
    mBracketLimit = 4,    // scipy's bracket() iteration limit (a RuntimeError there)
    mBadBracketF = -1,    // a three-point bracket without f(xb) below both ends: a ValueError
};

// A Python callback of one real variable. A value that is not a Python float goes through
// check(value, what), which returns a float or raises.
struct Fn {
    nb::handle f, check;
    const char* what;
    std::int64_t n = 0;

    double operator()(double x) {
        nb::object r = f(x);
        ++n;
        if (PyFloat_CheckExact(r.ptr())) return PyFloat_AS_DOUBLE(r.ptr());
        nb::object c = check(r, what);
        return PyFloat_AsDouble(c.ptr());
    }
};

// numpy's isclose(a, b, rtol, atol): |a - b| <= atol + rtol |b| when both are finite, else a == b.
bool isclose(double a, double b, double rtol, double atol) {
    if (std::isfinite(a) && std::isfinite(b)) return std::abs(a - b) <= atol + rtol * std::abs(b);
    return a == b;
}

// numpy's sign: 0 at +-0, NaN at NaN.
double npsign(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return x == 0 ? 0.0 : kNaN;
}

// numpy's maximum: NaN when either is NaN.
double npmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return kNaN;
    return a >= b ? a : b;
}

struct RootOut {
    double x = kNaN;
    int status = rConverged;
    std::int64_t iterations = 0;
    double fa = kNaN, fb = kNaN;  // the bracket's ends (a sign error's message)
    int reason = kMaxIter;
    double aux = kNaN;      // secant: p1 - p0 when q1 == q0
    std::int64_t nder = 0;  // newton, halley: the derivatives (fprime's calls or differences)
};

// ---- brentq (scipy's Zeros/brentq.c)

// scipy's C solvers are built with clang's default -ffp-contract=on, which fuses a * b + c into
// one rounding; this build has contraction off, so the fused operations are written out. The
// extrapolation's numerator must be fma(fblk, dblk, -(fpre * dpre)) for the steps to be scipy's
// bit for bit (the 204 root problems of tests/optimize/functions.py); the tolerance tests take
// the same form.

RootOut brentq(Fn& f, double xa, double xb, double xtol, double rtol, std::int64_t iter) {
    RootOut o;
    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    double delta, stry, dpre, dblk;
    fpre = f(xpre);
    if (std::isnan(fpre)) return {xpre, rValueErr};
    fcur = f(xcur);
    if (std::isnan(fcur)) return {xcur, rValueErr};
    o.fa = fpre;
    o.fb = fcur;
    if (fpre == 0) {
        o.x = xpre;
        return o;
    }
    if (fcur == 0) {
        o.x = xcur;
        return o;
    }
    if (std::signbit(fpre) == std::signbit(fcur)) {
        o.status = rSignErr;
        return o;
    }
    for (std::int64_t i = 0; i < iter; i++) {
        o.iterations++;
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
        delta = std::fma(rtol, std::abs(xcur), xtol) / 2;
        sbis = (xblk - xcur) / 2;
        if (fcur == 0 || std::abs(sbis) < delta) {
            o.x = xcur;
            return o;
        }
        if (std::abs(spre) > delta && std::abs(fcur) < std::abs(fpre)) {
            if (xpre == xblk) {
                stry = -fcur * (xcur - xpre) / (fcur - fpre);  // interpolate
            } else {
                dpre = (fpre - fcur) / (xpre - xcur);  // extrapolate
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * std::fma(fblk, dblk, -(fpre * dpre)) / (dblk * dpre * (fblk - fpre));
            }
            if (2 * std::abs(stry) < std::min(std::abs(spre), std::fma(3, std::abs(sbis), -delta))) {
                spre = scur;  // good short step
                scur = stry;
            } else {
                spre = sbis;  // bisect
                scur = sbis;
            }
        } else {
            spre = sbis;  // bisect
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if (std::abs(scur) > delta) xcur += scur;
        else xcur += (sbis > 0 ? delta : -delta);
        fcur = f(xcur);
        if (std::isnan(fcur)) {
            o.x = xcur;
            o.status = rValueErr;
            return o;
        }
    }
    o.x = xcur;
    o.status = rConvErr;
    return o;
}

// ---- bisect (scipy's Zeros/bisect.c)

RootOut bisect(Fn& f, double xa, double xb, double xtol, double rtol, std::int64_t iter) {
    RootOut o;
    double dm, xm, fm, fa, fb;
    fa = f(xa);
    if (std::isnan(fa)) return {xa, rValueErr};
    fb = f(xb);
    if (std::isnan(fb)) return {xb, rValueErr};
    o.fa = fa;
    o.fb = fb;
    if (fa == 0) {
        o.x = xa;
        return o;
    }
    if (fb == 0) {
        o.x = xb;
        return o;
    }
    // the signs by their sign bits, not the product fa * fb, which underflows to 0 when both
    // values are tiny
    if (std::signbit(fa) == std::signbit(fb)) {
        o.status = rSignErr;
        return o;
    }
    dm = xb - xa;
    for (std::int64_t i = 0; i < iter; i++) {
        o.iterations++;
        dm *= .5;
        xm = xa + dm;
        fm = f(xm);
        if (std::isnan(fm)) {
            o.x = xm;
            o.status = rValueErr;
            return o;
        }
        if (std::signbit(fm) == std::signbit(fa)) xa = xm;
        if (fm == 0 || std::abs(dm) < std::fma(rtol, std::abs(xm), xtol)) {
            o.x = xm;
            return o;
        }
    }
    o.x = xa;
    o.status = rConvErr;
    return o;
}

// ---- toms748 (scipy's TOMS748Solver and its helpers in _zeros_py.py)

// Stops the solver: f was not finite at x.
struct Stop {
    double x, fx;
    int status;
};

class Toms748 {
public:
    Toms748(Fn& f, double xtol, double rtol, std::int64_t maxiter, int k)
        : f_(f), xtol_(xtol), rtol_(rtol), maxiter_(maxiter), k_(k) {}

    RootOut solve(double a, double b) {
        RootOut o;
        try {
            o = run(a, b);
        } catch (const Stop& s) {
            o.x = s.x;
            o.status = s.status;
            o.fa = s.fx;
        }
        o.iterations = iterations_;
        return o;
    }

private:
    Fn& f_;
    double xtol_, rtol_;
    std::int64_t maxiter_;
    int k_;
    std::int64_t iterations_ = 0;
    double ab_[2] = {kNaN, kNaN}, fab_[2] = {kNaN, kNaN};
    double d_ = kNaN, fd_ = kNaN, e_ = kNaN, fe_ = kNaN;
    bool have_e_ = false;  // scipy's e = None before the second step

    double callf(double x) {
        const double fx = f_(x);
        if (!std::isfinite(fx)) throw Stop{x, fx, rValueErr};  // _wrap_nan_raise; O45 for +-inf
        return fx;
    }

    double mid() const { return (ab_[0] + ab_[1]) / 2.0; }

    // _update_bracket: replaces the end whose f has c's sign; returns the discarded end.
    void update_bracket(double c, double fc, double& rx, double& rfx) {
        const int idx = npsign(fab_[0]) * npsign(fc) > 0 ? 0 : 1;
        rx = ab_[idx];
        rfx = fab_[idx];
        fab_[idx] = fc;
        ab_[idx] = c;
    }

    // _secant
    static double secant(double x0, double x1, double f0, double f1) {
        if (f0 == f1) return kNaN;
        if (std::abs(f1) > std::abs(f0)) return (-f0 / f1 * x1 + x0) / (1 - f0 / f1);
        return (-f1 / f0 * x0 + x1) / (1 - f1 / f0);
    }

    // _notclose(fs, rtol=0, atol=32 eps): all nonzero and finite, no two within atol.
    static bool notclose(const double* fs, int n, double atol) {
        for (int i = 0; i < n; ++i)
            if (fs[i] == 0 || !std::isfinite(fs[i])) return false;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (std::abs(fs[i] - fs[j]) <= atol) return false;
        return true;
    }

    // _interpolated_poly(xvals, fvals, 0): Neville's algorithm, numpy's order of operations.
    static double interpolated_poly(const double* xv, const double* fv) {
        constexpr int N = 4;
        double Q[N][N] = {}, D[N][N] = {};
        for (int i = 0; i < N; ++i) Q[i][0] = D[i][0] = fv[i];
        for (int k = 1; k < N; ++k)
            for (int j = 0; j < N - k; ++j) {
                const double alpha = D[k + j][k - 1] - Q[k - 1 + j][k - 1];
                const double diffik = xv[j] - xv[k + j];
                Q[k + j][k] = (xv[k + j] - 0.0) / diffik * alpha;
                D[k + j][k] = (xv[j] - 0.0) / diffik * alpha;
            }
        return ((Q[N - 1][1] + Q[N - 1][2]) + Q[N - 1][3]) + Q[N - 1][0];
    }

    // _newton_quadratic(ab, fab, d, fd, k)
    double newton_quadratic(double d, double fd, int k) const {
        const double a = ab_[0], b = ab_[1], fa = fab_[0], fb = fab_[1];
        const double B = (fb - fa) / (b - a);
        const double A = ((fd - fb) / (d - b) - B) / (d - a);
        auto P = [&](double x) { return (A * (x - b) + B) * (x - a) + fa; };
        double r;
        if (A == 0) {
            r = a - fa / B;
        } else {
            r = npsign(A) * npsign(fa) > 0 ? a : b;
            for (int i = 0; i < k; ++i) {
                const double r1 = r - P(r) / (B + A * (2 * r - a - b));
                if (!(a < r1 && r1 < b)) {
                    if (a < r && r < b) return r;
                    r = (a + b) / 2.0;
                    break;
                }
                r = r1;
            }
        }
        return r;
    }

    RootOut done(double x, int status = rConverged) const {
        RootOut o;
        o.x = x;
        o.status = status;
        return o;
    }

    RootOut run(double a, double b) {
        ab_[0] = a;
        ab_[1] = b;
        const double fa = callf(a);
        if (fa == 0) return done(a);
        const double fb = callf(b);
        if (fb == 0) return done(b);
        if (npsign(fb) * npsign(fa) > 0) {
            RootOut o;
            o.status = rSignErr;
            o.fa = fa;
            o.fb = fb;
            return o;
        }
        fab_[0] = fa;
        fab_[1] = fb;
        // The first step only has two x-values.
        double c = secant(ab_[0], ab_[1], fab_[0], fab_[1]);
        if (!(ab_[0] < c && c < ab_[1])) c = mid();
        const double fc = callf(c);
        if (fc == 0) return done(c);
        update_bracket(c, fc, d_, fd_);
        have_e_ = false;
        iterations_ += 1;
        for (;;) {
            int status;
            const double xn = iterate(status);
            if (status == rConverged || status == rConvErr) return done(xn, status);
        }
    }

    // iterate(): algorithm 4.1 (k = 1) or 4.2 (k = 2); status 1 while in progress.
    double iterate(int& status) {
        iterations_ += 1;
        double d = d_, fd = fd_, e = e_, fe = fe_;
        bool have_e = have_e_;
        const double ab_width = ab_[1] - ab_[0];
        double c = kNaN;
        bool have_c = false;
        for (int nsteps = 2; nsteps < k_ + 2; ++nsteps) {
            const double fs[4] = {fab_[0], fab_[1], fd, fe};
            if (have_e && notclose(fs, 4, 32 * kEps)) {
                const double xv[4] = {fab_[0], fab_[1], fd, fe}, fv[4] = {ab_[0], ab_[1], d, e};
                const double c0 = interpolated_poly(xv, fv);
                if (ab_[0] < c0 && c0 < ab_[1]) {
                    c = c0;
                    have_c = true;
                }
            }
            if (!have_c) {
                c = newton_quadratic(d, fd, nsteps);
                have_c = true;
            }
            const double fc = callf(c);
            if (fc == 0) {
                status = rConverged;
                return c;
            }
            e = d;
            fe = fd;
            have_e = true;
            update_bracket(c, fc, d, fd);
        }
        // u is the end with the smaller |f|
        const int uix = std::abs(fab_[0]) < std::abs(fab_[1]) ? 0 : 1;
        const double u = ab_[uix], fu = fab_[uix];
        const double A = (fab_[1] - fab_[0]) / (ab_[1] - ab_[0]);
        c = u - 2 * fu / A;
        if (std::abs(c - u) > 0.5 * (ab_[1] - ab_[0])) {
            c = mid();
        } else if (isclose(c, u, kEps, 0)) {
            // c didn't change (much): the ends' f differ vastly, or the root is next to u
            int e0, e1;
            std::frexp(fab_[uix], &e0);
            std::frexp(fab_[1 - uix], &e1);
            if (e0 < e1 - 50) {
                c = (31 * ab_[uix] + ab_[1 - uix]) / 32;
            } else {
                const double mm = uix == 0 ? 1 : -1;
                const double adj = mm * std::abs(c) * rtol_ + mm * xtol_;
                c = u + adj;
            }
            if (!(ab_[0] < c && c < ab_[1])) c = mid();
        }
        const double fc = callf(c);
        if (fc == 0) {
            status = rConverged;
            return c;
        }
        e = d;
        fe = fd;
        have_e = true;
        update_bracket(c, fc, d, fd);
        // the bracket did not shrink enough: bisect
        if (ab_[1] - ab_[0] > 0.5 * ab_width) {
            e = d;
            fe = fd;
            const double z = mid();
            const double fz = callf(z);
            if (fz == 0) {
                status = rConverged;
                return z;
            }
            update_bracket(z, fz, d, fd);
        }
        d_ = d;
        fd_ = fd;
        e_ = e;
        fe_ = fe;
        have_e_ = have_e;
        // get_status
        if (isclose(ab_[0], ab_[1], rtol_, xtol_)) {
            status = rConverged;
            return mid();
        }
        status = iterations_ >= maxiter_ ? rConvErr : 1;
        return mid();
    }
};

// ---- newton, halley, secant (scipy's newton() for a scalar x0)

RootOut newton(Fn& f, Fn* fprime, Fn* fprime2, double x0, double tol, double rtol, std::int64_t maxiter) {
    RootOut o;
    double p0 = x0, p = kNaN;
    std::int64_t itr = 0;
    const double rstep = rel_step_2point();
    for (itr = 0; itr < maxiter; ++itr) {
        const double fval = f(p0);
        if (fval == 0) {
            o.x = p0;
            o.iterations = itr;
            return o;
        }
        double fder;
        o.nder += 1;
        if (fprime) {
            fder = (*fprime)(p0);
        } else {  // scipy's approx_derivative '2-point' (its f(p0) is fval)
            const double h = absolute_step(p0, rstep, false, 0.0);
            const double x1 = p0 + h;
            fder = (f(x1) - fval) / (x1 - p0);
        }
        if (fder == 0) {
            o.x = p0;
            o.iterations = itr + 1;
            o.status = rConvErr;
            o.reason = kDerivZero;
            return o;
        }
        double newton_step = fval / fder;
        if (fprime2) {  // Halley's method, only when the denominator stays near 1
            const double fder2 = (*fprime2)(p0);
            const double adj = newton_step * fder2 / fder / 2;
            if (std::abs(adj) < 1) newton_step /= 1.0 - adj;
        }
        p = p0 - newton_step;
        if (isclose(p, p0, rtol, tol)) {
            o.x = p;
            o.iterations = itr + 1;
            return o;
        }
        p0 = p;
    }
    o.x = p;
    o.iterations = itr;
    o.status = rConvErr;
    return o;
}

RootOut secant(Fn& f, double x0, bool have_x1, double x1, double tol, double rtol, std::int64_t maxiter) {
    RootOut o;
    double p0 = x0, p1, p = kNaN;
    if (have_x1) {
        p1 = x1;
    } else {
        const double eps = 1e-4;
        p1 = x0 * (1 + eps);
        p1 += (p1 >= 0 ? eps : -eps);
    }
    double q0 = f(p0);
    double q1 = f(p1);
    if (std::abs(q1) < std::abs(q0)) {
        std::swap(p0, p1);
        std::swap(q0, q1);
    }
    std::int64_t itr = 0;
    for (itr = 0; itr < maxiter; ++itr) {
        if (q1 == q0) {
            if (p1 != p0) {
                o.reason = kSecantTol;
                o.aux = p1 - p0;
            }
            o.x = (p1 + p0) / 2.0;
            o.iterations = itr + 1;
            o.status = rConvErr;
            return o;
        }
        if (std::abs(q1) > std::abs(q0)) p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1);
        else p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0);
        if (isclose(p, p1, rtol, tol)) {
            o.x = p;
            o.iterations = itr + 1;
            return o;
        }
        p0 = p1;
        q0 = q1;
        p1 = p;
        q1 = f(p1);
    }
    o.x = p;
    o.iterations = itr;
    o.status = rConvErr;
    return o;
}

// root_solve(func, f, fprime, fprime2, check, method, a, b, x0, x1, xtol, rtol, max_iter, k)
//   -> (x, status, n_iter, nfev, njev, nhev, fa, fb, reason, aux)
// (nfev: f's calls, the differences' included; njev: the derivatives; nhev: fprime2's calls)
// method: 0 brentq, 1 toms748, 2 bisect, 3 newton (fprime None: the 2-point difference),
// 4 halley, 5 secant (x1 None: scipy's default second point). check(value, what) -> float for
// a returned value that is not a float. fa, fb: f at the bracket's ends (a sign error), or
// toms748's non-finite value in fa (rValueErr).
nb::object root_solve(
    const std::string& func,
    nb::callable f,
    nb::object fprime,
    nb::object fprime2,
    nb::callable check,
    int method,
    double a,
    double b,
    double x0,
    nb::object x1,
    double xtol,
    double rtol,
    std::int64_t max_iter,
    int k
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Fn F{f, check, "f"}, G{fprime, check, "fprime"}, H{fprime2, check, "fprime2"};
        RootOut o;
        switch (method) {
            case 0: o = brentq(F, a, b, xtol, rtol, max_iter); break;
            case 1: o = Toms748(F, xtol, rtol, max_iter, k).solve(a, b); break;
            case 2: o = bisect(F, a, b, xtol, rtol, max_iter); break;
            case 3:
            case 4:
                o = newton(F, fprime.is_none() ? nullptr : &G, method == 4 ? &H : nullptr, x0, xtol, rtol, max_iter);
                break;
            default:
                o = secant(F, x0, !x1.is_none(), x1.is_none() ? 0.0 : nb::cast<double>(x1), xtol, rtol, max_iter);
                break;
        }
        return nb::make_tuple(o.x, o.status, o.iterations, F.n, o.nder, H.n, o.fa, o.fb, o.reason, o.aux);
    });
}

// ---- minimize_scalar

struct MinOut {
    double x = kNaN, fun = kNaN;
    int status = mConverged;
    std::int64_t nit = 0, nfev = 0;
    double fa = kNaN, fb = kNaN, fc = kNaN;  // a bad three-point bracket's values
};

// scipy's bracket(func, xa, xb) (grow_limit 110, maxiter 1000): the downhill search for three
// points with f(xb) below both ends. status: 0 found, mNoBracket, mBracketLimit.
struct Bracket {
    double xa, xb, xc, fa, fb, fc;
    std::int64_t funcalls;
    int status;
};

Bracket bracket(Fn& f, double xa, double xb) {
    constexpr double gold = 1.618034, verysmall = 1e-21, grow_limit = 110.0;
    constexpr std::int64_t maxiter = 1000;
    double fa = f(xa), fb = f(xb);
    if (fa < fb) {
        std::swap(xa, xb);
        std::swap(fa, fb);
    }
    double xc = xb + gold * (xb - xa);
    double fc = f(xc);
    std::int64_t funcalls = 3, iter = 0;
    while (fc < fb) {
        const double tmp1 = (xb - xa) * (fb - fc);
        const double tmp2 = (xb - xc) * (fb - fa);
        const double val = tmp2 - tmp1;
        const double denom = std::abs(val) < verysmall ? 2.0 * verysmall : 2.0 * val;
        double w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        const double wlim = xb + grow_limit * (xc - xb);
        if (iter > maxiter) return {xa, xb, xc, fa, fb, fc, funcalls, mBracketLimit};
        iter += 1;
        double fw;
        if ((w - xc) * (xb - w) > 0.0) {
            fw = f(w);
            funcalls += 1;
            if (fw < fc) {
                xa = xb;
                xb = w;
                fa = fb;
                fb = fw;
                break;
            } else if (fw > fb) {
                xc = w;
                fc = fw;
                break;
            }
            w = xc + gold * (xc - xb);
            fw = f(w);
            funcalls += 1;
        } else if ((w - wlim) * (wlim - xc) >= 0.0) {
            w = wlim;
            fw = f(w);
            funcalls += 1;
        } else if ((w - wlim) * (xc - w) > 0.0) {
            fw = f(w);
            funcalls += 1;
            if (fw < fc) {
                xb = xc;
                xc = w;
                w = xc + gold * (xc - xb);
                fb = fc;
                fc = fw;
                fw = f(w);
                funcalls += 1;
            }
        } else {
            w = xc + gold * (xc - xb);
            fw = f(w);
            funcalls += 1;
        }
        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }
    const bool cond1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
    const bool cond2 = (xa < xb && xb < xc) || (xc < xb && xb < xa);
    const bool cond3 = std::isfinite(xa) && std::isfinite(xb) && std::isfinite(xc);
    return {xa, xb, xc, fa, fb, fc, funcalls, cond1 && cond2 && cond3 ? 0 : mNoBracket};
}

// The bracket of brent and golden (scipy's get_bracket_info): none (bracket(f) from 0, 1), two
// points (bracket(f, xa, xb)) or three (already ordered xa < xb < xc by the caller; f(xb) must
// be below both). A failed search gives scipy's _recover_from_bracket_error result.
bool get_bracket(Fn& f, const std::vector<double>& br, Bracket& B, MinOut& o) {
    if (br.size() == 3) {
        B.xa = br[0];
        B.xb = br[1];
        B.xc = br[2];
        if (B.xa > B.xc) std::swap(B.xa, B.xc);
        B.fa = f(B.xa);
        B.fb = f(B.xb);
        B.fc = f(B.xc);
        B.funcalls = 3;
        if (!((B.fb < B.fa) && (B.fb < B.fc))) {
            o.status = mBadBracketF;
            o.fa = B.fa;
            o.fb = B.fb;
            o.fc = B.fc;
            o.nfev = 3;
            return false;
        }
        return true;
    }
    B = br.size() == 2 ? bracket(f, br[0], br[1]) : bracket(f, 0.0, 1.0);
    if (B.status == 0) return true;
    // _recover_from_bracket_error: the best of the three points (NaN if any is NaN)
    const double xs[3] = {B.xa, B.xb, B.xc}, fs[3] = {B.fa, B.fb, B.fc};
    bool nan = false;
    for (int i = 0; i < 3; ++i) nan = nan || std::isnan(xs[i]) || std::isnan(fs[i]);
    if (nan) {
        o.x = o.fun = kNaN;
    } else {
        int imin = 0;
        for (int i = 1; i < 3; ++i)
            if (fs[i] < fs[imin]) imin = i;
        o.x = xs[imin];
        o.fun = fs[imin];
    }
    o.status = B.status;
    o.nit = 0;
    o.nfev = B.funcalls;
    return false;
}

// scipy's Brent.optimize
MinOut brent(Fn& f, const std::vector<double>& br, double tol, std::int64_t maxiter) {
    MinOut o;
    Bracket B{};
    if (!get_bracket(f, br, B, o)) return o;
    const double xa = B.xa, xc = B.xc;
    std::int64_t funcalls = B.funcalls;
    constexpr double mintol = 1.0e-11, cg = 0.3819660;
    double x, w, v, fx, fw, fv;
    x = w = v = B.xb;
    fw = fv = fx = B.fb;
    double a, b;
    if (xa < xc) {
        a = xa;
        b = xc;
    } else {
        a = xc;
        b = xa;
    }
    double deltax = 0.0, rat = 0.0;
    std::int64_t iter = 0;
    while (iter < maxiter) {
        const double tol1 = tol * std::abs(x) + mintol;
        const double tol2 = 2.0 * tol1;
        const double xmid = 0.5 * (a + b);
        if (std::abs(x - xmid) < (tol2 - 0.5 * (b - a))) break;  // converged
        if (std::abs(deltax) <= tol1) {
            deltax = x >= xmid ? a - x : b - x;  // golden section step
            rat = cg * deltax;
        } else {  // parabolic step
            const double tmp1 = (x - w) * (fx - fv);
            double tmp2 = (x - v) * (fx - fw);
            double p = (x - v) * tmp2 - (x - w) * tmp1;
            tmp2 = 2.0 * (tmp2 - tmp1);
            if (tmp2 > 0.0) p = -p;
            tmp2 = std::abs(tmp2);
            const double dx_temp = deltax;
            deltax = rat;
            if ((p > tmp2 * (a - x)) && (p < tmp2 * (b - x)) && (std::abs(p) < std::abs(0.5 * tmp2 * dx_temp))) {
                rat = p * 1.0 / tmp2;  // the parabolic step is useful
                const double u = x + rat;
                if ((u - a) < tol2 || (b - u) < tol2) rat = xmid - x >= 0 ? tol1 : -tol1;
            } else {
                deltax = x >= xmid ? a - x : b - x;  // golden section step
                rat = cg * deltax;
            }
        }
        double u;
        if (std::abs(rat) < tol1) u = rat >= 0 ? x + tol1 : x - tol1;  // at least tol1
        else u = x + rat;
        const double fu = f(u);
        funcalls += 1;
        if (fu > fx) {
            if (u < x) a = u;
            else b = u;
            if ((fu <= fw) || (w == x)) {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if ((fu <= fv) || (v == x) || (v == w)) {
                v = u;
                fv = fu;
            }
        } else {
            if (u >= x) a = x;
            else b = x;
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
        iter += 1;
    }
    o.x = x;
    o.fun = fx;
    o.nit = iter;
    o.nfev = funcalls;
    o.status = mConverged;
    if (iter >= maxiter) o.status = mMaxIter;
    if (std::isnan(x) || std::isnan(fx)) o.status = mNaN;
    return o;
}

// scipy's _minimize_scalar_golden
MinOut golden(Fn& f, const std::vector<double>& br, double tol, std::int64_t maxiter) {
    MinOut o;
    Bracket B{};
    if (!get_bracket(f, br, B, o)) return o;
    const double xa = B.xa, xb = B.xb, xc = B.xc;
    std::int64_t funcalls = B.funcalls;
    constexpr double gR = 0.61803399;  // golden ratio conjugate
    constexpr double gC = 1.0 - gR;
    double x3 = xc, x0 = xa, x1, x2;
    if (std::abs(xc - xb) > std::abs(xb - xa)) {
        x1 = xb;
        x2 = xb + gC * (xc - xb);
    } else {
        x2 = xb;
        x1 = xb - gC * (xb - xa);
    }
    double f1 = f(x1), f2 = f(x2);
    funcalls += 2;
    std::int64_t nit = 0;
    for (std::int64_t i = 0; i < maxiter; ++i) {
        if (std::abs(x3 - x0) <= tol * (std::abs(x1) + std::abs(x2))) break;
        if (f2 < f1) {
            x0 = x1;
            x1 = x2;
            x2 = gR * x1 + gC * x3;
            f1 = f2;
            f2 = f(x2);
        } else {
            x3 = x2;
            x2 = x1;
            x1 = gR * x2 + gC * x0;
            f2 = f1;
            f1 = f(x1);
        }
        funcalls += 1;
        nit += 1;
    }
    if (f1 < f2) {
        o.x = x1;
        o.fun = f1;
    } else {
        o.x = x2;
        o.fun = f2;
    }
    o.nit = nit;
    o.nfev = funcalls;
    o.status = mConverged;
    if (nit >= maxiter) o.status = mMaxIter;
    if (std::isnan(o.fun) || std::isnan(o.x)) o.status = mNaN;
    return o;
}

// scipy's _minimize_scalar_bounded (fminbound): nfev = nit = the function calls
MinOut bounded(Fn& f, double x1, double x2, double xatol, std::int64_t maxfun) {
    MinOut o;
    int flag = 0;
    const double sqrt_eps = std::sqrt(2.2e-16);
    const double golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
    double a = x1, b = x2;
    double fulc = a + golden_mean * (b - a);
    double nfc = fulc, xf = fulc;
    double rat = 0.0, e = 0.0;
    double x = xf;
    double fx = f(x);
    std::int64_t num = 1;
    double fu = kInf;
    double ffulc = fx, fnfc = fx;
    double xm = 0.5 * (a + b);
    double tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
    double tol2 = 2.0 * tol1;
    while (std::abs(xf - xm) > (tol2 - 0.5 * (b - a))) {
        bool golden = true;
        if (std::abs(e) > tol1) {  // a parabolic fit
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
                rat = (p + 0.0) / q;  // the parabola is acceptable
                x = xf + rat;
                if (((x - a) < tol2) || ((b - x) < tol2)) {
                    const double si = npsign(xm - xf) + ((xm - xf) == 0 ? 1.0 : 0.0);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if (golden) {  // a golden-section step
            e = xf >= xm ? a - xf : b - xf;
            rat = golden_mean * e;
        }
        const double si = npsign(rat) + (rat == 0 ? 1.0 : 0.0);
        x = xf + si * npmax(std::abs(rat), tol1);
        fu = f(x);
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
            flag = 1;
            break;
        }
    }
    if (std::isnan(xf) || std::isnan(fx) || std::isnan(fu)) flag = 2;
    o.x = xf;
    o.fun = fx;
    o.status = flag;
    o.nit = o.nfev = num;
    return o;
}

// min_solve(func, fun, check, method, bracket, lo, hi, xtol, max_iter)
//   -> (x, fun, status, n_iter, nfev, fa, fb, fc)
// method: 0 brent, 1 golden (bracket: [], two or three points, three already ordered by the
// caller), 2 bounded (lo, hi; xtol is xatol, max_iter the function calls).
nb::object min_solve(
    const std::string& func,
    nb::callable fun,
    nb::callable check,
    int method,
    const std::vector<double>& br,
    double lo,
    double hi,
    double xtol,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Fn F{fun, check, "fun"};
        MinOut o;
        if (method == 0) o = brent(F, br, xtol, max_iter);
        else if (method == 1) o = golden(F, br, xtol, max_iter);
        else o = bounded(F, lo, hi, xtol, max_iter);
        return nb::make_tuple(o.x, o.fun, o.status, o.nit, o.nfev, o.fa, o.fb, o.fc);
    });
}

}  // namespace

void bind_scalar(nb::module_& m) {
    m.def(
        "root_solve", &root_solve, nb::arg("func"), nb::arg("f"), nb::arg("fprime").none(), nb::arg("fprime2").none(),
        nb::arg("check"), nb::arg("method"), nb::arg("a"), nb::arg("b"), nb::arg("x0"), nb::arg("x1").none(),
        nb::arg("xtol"), nb::arg("rtol"), nb::arg("max_iter"), nb::arg("k")
    );
    m.def(
        "min_solve", &min_solve, nb::arg("func"), nb::arg("fun"), nb::arg("check"), nb::arg("method"),
        nb::arg("bracket"), nb::arg("lo"), nb::arg("hi"), nb::arg("xtol"), nb::arg("max_iter")
    );
}

}  // namespace sd::optimize
