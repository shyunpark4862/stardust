// The batched scalar functions (docs/design/optimize.md §3.2, §3.3; scipy 1.18.1's
// optimize.elementwise): find_root (Chandrupatla 1997, scipy's _chandrupatla), find_minimum
// (Chandrupatla 1998, _chandrupatla_minimize), bracket_root and bracket_minimum (_bracket.py),
// the same steps per element as scipy (O9).
//
// Every element has its own state here; one iteration evaluates the user's function once, on a
// new float64 sd.ndarray of the elements still in progress (in their order; O38), and updates
// their state. The counts nit and nfev are the iterations and evaluations until the element
// stopped, as in scipy's loop (eim._loop). The first evaluations (f at each initial point) take
// all elements. Status codes are scipy's: 0 converged, -1 sign or bracket error (bracket_*:
// a limit reached), -2 max_iter, -3 a non-finite value, -5 an invalid initial bracket, and 2
// for a side of bracket_root stopped because the other side converged (internal).
//
// scipy runs each step (the new points, the update, each stopping test, the next step's
// parameters) as array operations over the elements in progress, then compresses them. Each of
// them is elementwise, so here one pass over the elements does the update, the tests, the
// compression and the next step's parameters, and one more writes the points of the next call.
// The loop's parts (the passes over the elements, the call, the bookkeeping) are the calculus
// area's too: csrc/elementwise/loop.hpp (docs/design/integrate.md §6.2).
#include "optimize/scalar.hpp"

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "elementwise/loop.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace sd::optimize {

namespace nb = nanobind;

namespace {

// the loop's parts (for_elements, Call, Progress, advance, ...) and scipy's status codes
using namespace sd::elementwise;

// the codes' meanings here besides scipy's (sSignErr: find_* where the bracket does not enclose a
// root or minimum)
constexpr int sLimits = -1;   // bracket_*: a limit (xmin or xmax) reached
constexpr int sOneSide = 2;  // bracket_root: the other side found the bracket (internal)

// ---- find_root: scipy's _chandrupatla

nb::object find_root_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle args,
    nb::handle xl,
    nb::handle xr,
    double xatol,
    double xrtol,
    double fatol,
    double frtol,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const vec xl0 = read(xl), xr0 = read(xr);
        const std::size_t n = xl0.size();
        Call F(f, check, args, n);
        vec fl0, fr0;
        F.all(xl0, fl0);
        F.all(xr0, fr0);
        // one element's state, together in memory: a pass over the elements reads it in order
        struct E {
            double x1, f1, x2, f2, x3, f3, t, xmin, fmin, frt;
        };
        std::vector<E> S(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double frt = frtol * npmin(std::abs(fl0[i]), std::abs(fr0[i]));
            S[i] = {xl0[i], fl0[i], xr0[i], fr0[i], kNaN, kNaN, 0.5, kNaN, kNaN, frt};
        }
        Progress P(n);
        std::int64_t nit = 0, nfev = 2;

        // check_termination for element i (true: it stops), then, for one that goes on,
        // post_termination_check (the next t: inverse quadratic interpolation or bisection)
        auto step = [&](std::size_t i) -> bool {
            E& e = S[i];
            const bool lo = std::abs(e.f1) < std::abs(e.f2);
            e.xmin = lo ? e.x1 : e.x2;
            e.fmin = lo ? e.f1 : e.f2;
            bool s = false;
            if (std::abs(e.fmin) <= fatol + e.frt) {
                P.status[i] = sConverged;
                s = true;
            }
            if (!s && npsign(e.f1) == npsign(e.f2)) {
                e.xmin = e.fmin = kNaN;
                P.status[i] = sSignErr;
                s = true;
            }
            const bool x_nonfinite = !(std::isfinite(e.x1) && std::isfinite(e.x2));
            if (!s && (x_nonfinite || (std::isnan(e.f1) && std::isnan(e.f2)))) {
                e.xmin = e.fmin = kNaN;
                P.status[i] = sValueErr;
                s = true;
            }
            const double dx = std::abs(e.x2 - e.x1);
            const double tol = std::abs(e.xmin) * xrtol + xatol;
            if (dx < tol) {
                P.status[i] = sConverged;
                s = true;
            }
            if (s) return true;
            const double xi1 = (e.x1 - e.x2) / (e.x3 - e.x2);
            const double phi1 = (e.f1 - e.f2) / (e.f3 - e.f2);
            const double alpha = (e.x3 - e.x1) / (e.x2 - e.x1);
            double ti = 0.5;
            if (((1 - std::sqrt(1 - xi1)) < phi1) && (phi1 < std::sqrt(xi1)))
                ti = e.f1 / (e.f1 - e.f2) * e.f3 / (e.f3 - e.f2) - alpha * e.f1 / (e.f3 - e.f1) * e.f2 / (e.f2 - e.f3);
            const double tl = 0.5 * tol / dx;
            e.t = npmin(npmax(ti, tl), 1 - tl);
            return false;
        };

        std::vector<std::uint32_t> on;
        // the first test (before any iteration) has no step after it: t stays 0.5
        advance(P, on, [&](std::size_t r) {
            const auto i = static_cast<std::size_t>(P.active[r]);
            const bool s = step(i);
            if (s) P.stopped(i, nit, nfev);
            S[i].t = 0.5;
            return !s;
        });
        while (nit < max_iter && !P.active.empty()) {
            const std::size_t k = P.active.size();
            F.start(k);
            for_elements(k, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r) {
                    const E& e = S[static_cast<std::size_t>(P.active[r])];
                    F.put(r, e.x1 + e.t * (e.x2 - e.x1));
                }
            });
            F.run(P.active.data());
            nfev += 1;
            nit += 1;
            advance(P, on, [&](std::size_t r) {
                const auto i = static_cast<std::size_t>(P.active[r]);
                E& e = S[i];
                const double fx = F.value(r);
                e.x3 = e.x2;
                e.f3 = e.f2;
                if (npsign(fx) == npsign(e.f1)) {
                    e.x3 = e.x1;
                    e.f3 = e.f1;
                } else {
                    e.x2 = e.x1;
                    e.f2 = e.f1;
                }
                e.x1 = F.x(r);
                e.f1 = fx;
                const bool s = step(i);
                if (s) P.stopped(i, nit, nfev);
                return !s;
            });
        }
        P.finish(nit, nfev);
        vec xmin(n), fmin(n), bl(n), br(n), fl(n), fr(n);
        for (std::size_t i = 0; i < n; ++i) {
            const E& e = S[i];
            const bool keep = e.x1 < e.x2;
            xmin[i] = e.xmin;
            fmin[i] = e.fmin;
            bl[i] = keep ? e.x1 : e.x2;
            br[i] = keep ? e.x2 : e.x1;
            fl[i] = keep ? e.f1 : e.f2;
            fr[i] = keep ? e.f2 : e.f1;
        }
        return nb::make_tuple(
            out(xmin), out(fmin), out(bl), out(br), out(fl), out(fr), out(P.status), out(P.nit), out(P.nfev)
        );
    });
}

// ---- find_minimum: scipy's _chandrupatla_minimize

// The order of three points as numpy's argsort (stable; NaN last).
void sort3(double* x, double* f) {
    int idx[3] = {0, 1, 2};
    auto less = [&](int a, int b) { return x[a] < x[b] || (std::isnan(x[b]) && !std::isnan(x[a])); };
    std::stable_sort(idx, idx + 3, less);
    const double xs[3] = {x[0], x[1], x[2]}, fs[3] = {f[0], f[1], f[2]};
    for (int j = 0; j < 3; ++j) {
        x[j] = xs[idx[j]];
        f[j] = fs[idx[j]];
    }
}

nb::object find_minimum_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle args,
    nb::handle xa,
    nb::handle xb,
    nb::handle xc,
    double xatol,
    double xrtol,
    double fatol,
    double frtol,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const vec xa0 = read(xa), xb0 = read(xb), xc0 = read(xc);
        const std::size_t n = xa0.size();
        Call F(f, check, args, n);
        vec fa0, fb0, fc0;
        F.all(xa0, fa0);
        F.all(xb0, fb0);
        F.all(xc0, fc0);
        struct E {  // one element's state, together in memory
            double x1, f1, x2, f2, x3, f3, q0, xtol;
        };
        std::vector<E> S(n);
        for (std::size_t i = 0; i < n; ++i) {
            double xs[3] = {xa0[i], xb0[i], xc0[i]}, fs[3] = {fa0[i], fb0[i], fc0[i]};
            sort3(xs, fs);
            S[i] = {xs[0], fs[0], xs[1], fs[1], xs[2], fs[2], xs[2], 0.0};
        }
        const double phi = 0.5 + 0.5 * std::sqrt(5.0);
        Progress P(n);
        std::int64_t nit = 0, nfev = 3;

        // check_termination for element i: true when it stops
        auto stops = [&](std::size_t i) -> bool {
            E& e = S[i];
            bool s = false;
            if ((e.f2 > e.f1) || (e.f2 > e.f3)) {
                e.x2 = e.f2 = kNaN;
                P.status[i] = sSignErr;
                s = true;
            }
            const bool finite = std::isfinite(e.x1 + e.x2 + e.x3 + e.f1 + e.f2 + e.f3);
            if (!(finite || s)) {
                e.x2 = e.f2 = kNaN;
                P.status[i] = sValueErr;
                s = true;
            }
            if (std::abs(e.x3 - e.x2) < std::abs(e.x2 - e.x1)) {
                std::swap(e.x1, e.x3);
                std::swap(e.f1, e.f3);
            }
            e.xtol = std::abs(e.x2) * xrtol + xatol;
            bool c = std::abs(e.x3 - e.x2) <= 2 * e.xtol;
            const double ftol = std::abs(e.f2) * frtol + fatol;
            c = c || (e.f1 - 2 * e.f2 + e.f3) <= 2 * ftol;
            if (c && !s) {
                P.status[i] = sConverged;
                s = true;
            }
            return s;
        };

        std::vector<std::uint32_t> on;
        auto test = [&](std::size_t r) {
            const auto i = static_cast<std::size_t>(P.active[r]);
            const bool s = stops(i);
            if (s) P.stopped(i, nit, nfev);
            return !s;
        };
        advance(P, on, test);
        while (nit < max_iter && !P.active.empty()) {
            const std::size_t k = P.active.size();
            F.start(k);
            for_elements(k, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r) {
                    E& e = S[static_cast<std::size_t>(P.active[r])];
                    const double x21 = e.x2 - e.x1;
                    const double x32 = e.x3 - e.x2;
                    const double A = x21 * (e.f3 - e.f2);
                    const double B = x32 * (e.f1 - e.f2);
                    const double C = A / (A + B);
                    const double q1 = 0.5 * (C * (e.x1 - e.x3) + e.x2 + e.x3);
                    double x;
                    if (std::abs(q1 - e.q0) < 0.5 * std::abs(x21)) {  // the parabola's minimum
                        x = q1;
                        if (std::abs(q1 - e.x2) <= e.xtol) x = e.x2 + npsign(x32) * e.xtol;
                    } else {  // golden section
                        x = e.x2 + (2 - phi) * x32;
                    }
                    e.q0 = q1;
                    F.put(r, x);
                }
            });
            F.run(P.active.data());
            nfev += 1;
            nit += 1;
            advance(P, on, [&](std::size_t r) {
                const auto i = static_cast<std::size_t>(P.active[r]);
                E& e = S[i];
                const double xi = F.x(r), fi = F.value(r);
                if (npsign(xi - e.x2) == npsign(e.x3 - e.x2)) {
                    if (fi > e.f2) {
                        e.x3 = xi;
                        e.f3 = fi;
                    } else {
                        e.x1 = e.x2;
                        e.f1 = e.f2;
                        e.x2 = xi;
                        e.f2 = fi;
                    }
                } else {
                    if (fi > e.f2) {
                        e.x1 = xi;
                        e.f1 = fi;
                    } else {
                        e.x3 = e.x2;
                        e.f3 = e.f2;
                        e.x2 = xi;
                        e.f2 = fi;
                    }
                }
                return test(r);
            });
        }
        P.finish(nit, nfev);
        vec x(n), fx(n), bl(n), br(n), fl(n), fr(n);
        for (std::size_t i = 0; i < n; ++i) {
            const E& e = S[i];
            const bool swap = e.x1 >= e.x3;
            x[i] = e.x2;
            fx[i] = e.f2;
            bl[i] = swap ? e.x3 : e.x1;
            br[i] = swap ? e.x1 : e.x3;
            fl[i] = swap ? e.f3 : e.f1;
            fr[i] = swap ? e.f1 : e.f3;
        }
        return nb::make_tuple(
            out(x), out(fx), out(bl), out(x), out(br), out(fl), out(fx), out(fr), out(P.status), out(P.nit), out(P.nfev)
        );
    });
}

// ---- bracket_root: scipy's _bracket_root (each element is two sides, searched together)

nb::object bracket_root_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle args,
    nb::handle xl0_,
    nb::handle xr0_,
    bool xr0_given,
    nb::handle xmin_,
    nb::handle xmax_,
    nb::handle factor_,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        vec xl0 = read(xl0_), xr0 = read(xr0_), xmin = read(xmin_), xmax = read(xmax_), fac = read(factor_);
        const std::size_t n = xl0.size(), m = 2 * n;
        Call F(f, check, args, n);  // a side i is element i mod n
        if (!xr0_given)
            for (std::size_t i = 0; i < n; ++i) xr0[i] = xl0[i] + npmin((xmax[i] - xl0[i]) / 8, 1.0);
        vec fl0, fr0;
        F.all(xl0, fl0);
        F.all(xr0, fr0);
        // sides: 0..n-1 the left ends (towards xmin), n..2n-1 the right ends (towards xmax); one
        // side's state together in memory
        struct E {
            double x, f, x_last, f_last, x0, limit, factor, d;
            bool invalid;
        };
        std::vector<E> S(m);
        Progress P(m);
        for (std::size_t i = 0; i < n; ++i) {
            const bool invalid = !((xmin[i] <= xl0[i]) && (xl0[i] < xr0[i]) && (xr0[i] <= xmax[i]));
            S[i] = {xl0[i], fl0[i], xr0[i], fr0[i], xr0[i], xmin[i], fac[i], 0.0, invalid};
            S[n + i] = {xr0[i], fr0[i], xl0[i], fl0[i], xl0[i], xmax[i], fac[i], 0.0, invalid};
            if (invalid) P.status[i] = P.status[n + i] = sInputErr;
        }
        for (E& e : S) e.d = std::isinf(e.limit) ? e.x - e.x0 : e.limit - e.x;
        std::int64_t nit = 0, nfev = 1;
        // conv: the other side of a side that converged; hit: the side in progress r found a
        // sign change in this pass (its other side is i + n mod 2n, one per side, so the writes
        // of a pass never meet)
        std::vector<char> stop, conv(m, 0), hit;

        // check_termination: the sign test on every side in progress first (stop, and conv for
        // the sides that found a sign change), then the other side of those, the limits and
        // the non-finite values, then the compression; each a pass over the sides in progress
        // (for_elements: split over the threads, without the GIL, when there are many)
        auto first = [&](std::size_t r, std::size_t i) {
            const E& e = S[i];
            stop[r] = e.invalid;
            hit[r] = 0;
            const double sf = npsign(e.f), sfl = npsign(e.f_last);
            if (((sfl == -sf) || (sfl == 0) || (sf == 0)) && !stop[r]) {
                P.status[i] = sConverged;
                stop[r] = 1;
                conv[(i + n) % m] = 1;
                hit[r] = 1;
            }
        };
        auto rest = [&]() {
            const std::size_t k = P.active.size();
            for_elements(k, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r) {
                    const auto i = static_cast<std::size_t>(P.active[r]);
                    const E& e = S[i];
                    if (conv[i] && !stop[r]) {
                        P.status[i] = sOneSide;
                        stop[r] = 1;
                    }
                    if (e.x == e.limit && !stop[r]) {
                        P.status[i] = sLimits;
                        stop[r] = 1;
                    }
                    if (!(std::isfinite(e.x) && std::isfinite(e.f)) && !stop[r]) {
                        P.status[i] = sValueErr;
                        stop[r] = 1;
                    }
                    if (stop[r]) P.stopped(i, nit, nfev);
                }
            });
            for_elements(k, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r)
                    if (hit[r]) conv[(static_cast<std::size_t>(P.active[r]) + n) % m] = 0;
            });
            std::size_t w = 0;
            for (std::size_t r = 0; r < k; ++r)
                if (!stop[r]) P.active[w++] = P.active[r];
            P.active.resize(w);
        };

        stop.assign(m, 0);
        hit.assign(m, 0);
        for_elements(m, [&](std::size_t r0, std::size_t r1) {
            for (std::size_t r = r0; r < r1; ++r) first(r, r);
        });
        rest();
        while (nit < max_iter && !P.active.empty()) {
            const std::size_t k = P.active.size();
            F.start(k);
            for_elements(k, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r) {
                    E& e = S[static_cast<std::size_t>(P.active[r])];
                    if (std::isinf(e.limit)) {
                        e.d *= e.factor;
                        F.put(r, e.x0 + e.d);
                    } else {
                        e.d /= e.factor;
                        F.put(r, e.limit - e.d);
                    }
                }
            });
            F.run(P.active.data());
            nfev += 1;
            nit += 1;
            stop.assign(k, 0);
            hit.assign(k, 0);
            for_elements(k, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r) {
                    const auto i = static_cast<std::size_t>(P.active[r]);
                    E& e = S[i];
                    e.x_last = e.x;
                    e.f_last = e.f;
                    e.x = F.x(r);
                    e.f = F.value(r);
                    first(r, i);
                }
            });
            rest();
        }
        P.finish(nit, nfev);
        // the bracket of the side that converged (the narrower when both did; the left side's
        // when both are as narrow: scipy then joins the two sides into a reversed interval
        // (the right side's left end, the left side's right end) that need not bracket a root).
        // Without a bracket the interval is the two sides' reach, sorted also for an invalid
        // start (xl0 > xr0), so that xl <= xr always.
        vec bl(n), br(n), fl(n), fr(n);
        ivec status(n), nits(n), nfevs(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t a = i, b = n + i;
            const double xal = S[a].x, xar = S[a].x_last, xbl = S[b].x_last, xbr = S[b].x;
            const double fal = S[a].f, far = S[a].f_last, fbl = S[b].f_last, fbr = S[b].f;
            double xl = xal, fl_ = fal, xr = xbr, fr_ = fbr;
            const std::int64_t sa = P.status[a], sb = P.status[b];
            const double da = xar - xal, db = xbr - xbl;
            const bool i1 = ((da <= db) && (sa == 0)) || ((sa == 0) && (sb != 0));
            const bool i2 = !i1 && (sb == 0);
            if (i1) {
                xr = xar;
                fr_ = far;
            }
            if (i2) {
                xl = xbl;
                fl_ = fbl;
            }
            if (xl > xr) {
                std::swap(xl, xr);
                std::swap(fl_, fr_);
            }
            bl[i] = xl;
            br[i] = xr;
            fl[i] = fl_;
            fr[i] = fr_;
            nits[i] = std::max(P.nit[a], P.nit[b]);
            nfevs[i] = P.nfev[a] + P.nfev[b];
            status[i] = sa == 0 ? sa : sb;
        }
        return nb::make_tuple(out(bl), out(br), out(fl), out(fr), out(status), out(nits), out(nfevs));
    });
}

// ---- bracket_minimum: scipy's _bracket_minimum

nb::object bracket_minimum_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle args,
    nb::handle xm0_,
    nb::handle xl0_,
    bool xl0_given,
    nb::handle xr0_,
    bool xr0_given,
    nb::handle xmin_,
    nb::handle xmax_,
    nb::handle factor_,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const vec xm = read(xm0_), xmin = read(xmin_), xmax = read(xmax_), fac = read(factor_);
        vec xl = read(xl0_), xr = read(xr0_);
        const std::size_t n = xm.size();
        Call F(f, check, args, n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!xl0_given) xl[i] = xm[i] - npmin((xm[i] - xmin[i]) / 16, 0.5);
            if (!xr0_given) xr[i] = xm[i] + npmin((xmax[i] - xm[i]) / 16, 0.5);
        }
        vec fl, fm, fr;
        F.all(xl, fl);
        F.all(xm, fm);
        F.all(xr, fr);
        struct E {  // one element's state, together in memory
            double xl, xm, xr, fl, fm, fr, limit, step, xr0, factor;
            bool limited, invalid;
        };
        std::vector<E> S(n);
        Progress P(n);
        for (std::size_t i = 0; i < n; ++i) {
            E& e = S[i];
            e = {xl[i], xm[i], xr[i], fl[i], fm[i], fr[i], 0.0, 0.0, 0.0, fac[i], false, false};
            e.invalid = !((xmin[i] <= e.xl) && (e.xl < e.xm) && (e.xm < e.xr) && (e.xr <= xmax[i]));
            const bool comp = e.fl < e.fr;  // search towards the lower end
            if (comp) {
                std::swap(e.xl, e.xr);
                std::swap(e.fl, e.fr);
            }
            e.limit = comp ? xmin[i] : xmax[i];
            e.limited = !std::isinf(e.limit);
            e.step = e.limited ? e.limit - e.xr : e.xr - e.xm;
            if (e.limited) e.factor = 1 / e.factor;
            e.xr0 = e.xr;
            if (e.invalid) P.status[i] = sInputErr;
        }
        std::int64_t nit = 0, nfev = 3;

        // check_termination for element i: true when it stops
        auto stops = [&](std::size_t i) -> bool {
            const E& e = S[i];
            bool s = e.invalid;
            const bool found = ((e.fl >= e.fm) && (e.fr > e.fm)) || ((e.fl > e.fm) && (e.fr >= e.fm));
            if (found && !s) {
                P.status[i] = sConverged;
                s = true;
            }
            if (e.xr == e.limit && !s) {
                P.status[i] = sLimits;
                s = true;
            }
            if (!(std::isfinite(e.xr) && std::isfinite(e.fr)) && !s) {
                P.status[i] = sValueErr;
                s = true;
            }
            return s;
        };

        std::vector<std::uint32_t> on;
        auto test = [&](std::size_t r) {
            const auto i = static_cast<std::size_t>(P.active[r]);
            const bool s = stops(i);
            if (s) P.stopped(i, nit, nfev);
            return !s;
        };
        advance(P, on, test);
        while (nit < max_iter && !P.active.empty()) {
            const std::size_t k = P.active.size();
            F.start(k);
            for_elements(k, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r) {
                    E& e = S[static_cast<std::size_t>(P.active[r])];
                    e.step *= e.factor;
                    double x;
                    if (e.limited) {
                        x = e.limit - e.step;
                        if (x == e.xr) x = e.limit;
                    } else {
                        x = e.xr0 + e.step;
                    }
                    F.put(r, x);
                }
            });
            F.run(P.active.data());
            nfev += 1;
            nit += 1;
            advance(P, on, [&](std::size_t r) {
                const auto i = static_cast<std::size_t>(P.active[r]);
                E& e = S[i];
                e.xl = e.xm;
                e.xm = e.xr;
                e.xr = F.x(r);
                e.fl = e.fm;
                e.fm = e.fr;
                e.fr = F.value(r);
                return test(r);
            });
        }
        P.finish(nit, nfev);
        vec bl(n), bm(n), br(n), gl(n), gm(n), gr(n);
        for (std::size_t i = 0; i < n; ++i) {
            const E& e = S[i];
            const bool swap = e.xl > e.xr;
            bl[i] = swap ? e.xr : e.xl;
            br[i] = swap ? e.xl : e.xr;
            gl[i] = swap ? e.fr : e.fl;
            gr[i] = swap ? e.fl : e.fr;
            bm[i] = e.xm;
            gm[i] = e.fm;
        }
        return nb::make_tuple(
            out(bl), out(bm), out(br), out(gl), out(gm), out(gr), out(P.status), out(P.nit), out(P.nfev)
        );
    });
}

}  // namespace

void bind_batch(nb::module_& m) {
    m.def(
        "find_root_solve", &find_root_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"), nb::arg("args"),
        nb::arg("xl"), nb::arg("xr"), nb::arg("xatol"), nb::arg("xrtol"), nb::arg("fatol"), nb::arg("frtol"),
        nb::arg("max_iter")
    );
    m.def(
        "find_minimum_solve", &find_minimum_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"), nb::arg("args"),
        nb::arg("x1"), nb::arg("x2"), nb::arg("x3"), nb::arg("xatol"), nb::arg("xrtol"), nb::arg("fatol"),
        nb::arg("frtol"), nb::arg("max_iter")
    );
    m.def(
        "bracket_root_solve", &bracket_root_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"), nb::arg("args"),
        nb::arg("xl0"), nb::arg("xr0"), nb::arg("xr0_given"), nb::arg("xmin"), nb::arg("xmax"), nb::arg("factor"),
        nb::arg("max_iter")
    );
    m.def(
        "bracket_minimum_solve", &bracket_minimum_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"),
        nb::arg("args"), nb::arg("xm0"), nb::arg("xl0"), nb::arg("xl0_given"), nb::arg("xr0"), nb::arg("xr0_given"),
        nb::arg("xmin"), nb::arg("xmax"), nb::arg("factor"), nb::arg("max_iter")
    );
}

}  // namespace sd::optimize
