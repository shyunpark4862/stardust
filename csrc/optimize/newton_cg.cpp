// minimize(method='newton-cg') (docs/design/optimize.md §3.5): scipy 1.18.1's _minimize_newtoncg,
// the same steps (O9).
//
// Each iteration solves H p = -g from p = 0 by conjugate gradients (at most 20n steps; stops when
// sum |r_i| <= min(1/2, sqrt(|g|_1)) |g|_1, at a curvature p.Hp in [0, 3 eps], or at a negative
// curvature, which in the first step gives the steepest descent (r.r / -p.Hp) (-g)); a curvature
// in [0, 3 eps] that is also at most 3 eps |g|^2 (zero, not small because g is; and not because
// the difference product's probe x + eps p rounded to x) in the first step gives -g (scipy's p = 0 would stop the iteration at a point that is not stationary: a
// saddle, a linear function), a NaN curvature stops with status 3
// 'NaN result encountered.' at the iterate (scipy goes on for 20n steps), then searches
// along p with scipy's _line_search_wolfe12 (wolfe.hpp; Newton-CG passes scipy's defaults:
// line_search_wolfe1 with steps in [1e-8, 50], line_search_wolfe2 without amax, and no previous
// value at the first iteration). It stops when the step's |.|_1 is at most n xtol, at max_iter
// iterations, when the line search finds no step (2) or when CG does not converge in 20n steps
// (3). The products H p are the matrix of hess (computed once per iteration), hessp(x, p), the
// difference operator of hess='2-point' / '3-point' (hessian.hpp), or without either scipy's
// approx_fhess_p: (grad(x + eps p) - grad(x)) / eps with grad(x) the iteration's (scipy calls it
// again through ScalarFunction's memo after the first product, half the gradients of a run).
// When the line search ends at a NaN value (wolfe2 grows the step into the region where f is
// NaN), the result is the last iterate with a finite value (L-BFGS-B's rule, O59).
//
// The counts: nfev and njev ScalarFunction's (every gradient, the differences' included), nhev
// the calls of hess (the first at x0, before the iterations; O68) or the products of hessp and
// of the difference operator. The result's jac is the gradient at the returned x (O70; scipy's
// is the one at the start of the last iteration): from the memo when it is there, else one more
// call, counted in njev.
//
// The arithmetic is numpy's: dot products and the matrix-vector product by Accelerate's BLAS as
// numpy calls it (hessian.hpp's np_dot and np_matvec), the 1-norms numpy's pairwise sums, no fused
// multiply-adds.
#include "optimize/hessian.hpp"
#include "optimize/minimize.hpp"
#include "optimize/objective.hpp"
#include "optimize/wolfe.hpp"

#include "bind/errors.hpp"
#include "linalg/types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace sd::optimize {

namespace {

using namespace linesearch;

// the kernel's status: scipy's warnflag, and 4 for its 'NaN result encountered.' (status 3)
enum NcgStatus : int { nSuccess = 0, nMaxIter = 1, nPrLoss = 2, nCg = 3, nNaN = 4 };

double abs_sum(const vec& a, vec& work) {
    work.resize(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) work[i] = std::abs(a[i]);
    return pairwise_sum(work.data(), a.size());
}

// newton_cg_solve(func, fun, jac, hess, hessp, check, mode, hmode, x0, xtol, eps, max_iter, c1, c2)
//   -> (x, fun, jac, status, n_iter, nfev, njev, nhev)
// mode: objective.hpp's GradMode (callable or with fun); hmode: hessian.hpp's HessMode; hessp:
// the user's product or None; eps: the steps of approx_fhess_p (n values).
nb::object newton_cg_solve(
    const std::string& func,
    nb::callable fun,
    nb::object jac,
    nb::object hess,
    nb::object hessp,
    nb::callable check,
    int mode,
    int hmode,
    const vec& x0,
    double xtol,
    const vec& eps,
    std::int64_t max_iter,
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
        obj.lb.assign(n, -kInf);
        obj.ub.assign(n, kInf);
        obj.start(x0.data());  // ScalarFunction: fun(x0), then the gradient
        Hessian hs(obj, hess, hessp, hmode);
        if (hmode != kHessNone) hs.update(x0.data());  // hess(x0), or the difference operator
        obj.require_finite_start();
        if (hmode == kHessCallable && !hs.matrix_finite()) throw nb::value_error("'hess' is not finite at 'x0'");
        const bool user_hessp = !hessp.is_none();
        const bool big = 2.0 * double(n) * double(n) > 2e5;  // the 10 us rule for the products

        LineSearch ls(obj, c1, c2, 1e-8, 50.0, kInf);
        ls.xt.resize(n);
        std::int64_t hcalls = 0;
        const std::int64_t cg_maxiter = 20 * static_cast<std::int64_t>(n);
        const double xtol_n = double(n) * xtol;
        const double eps3 = 3 * std::numeric_limits<double>::epsilon();
        double update_l1norm = std::numeric_limits<double>::max();
        vec xk = x0, b(n), xsupi(n), ri(n), psupi(n), Ap(n), update(n), work, gfk = obj.g, xe(n);
        std::int64_t k = 0;
        double old_fval = obj.value(x0.data()), old_old_fval = 0;
        bool has_old_old = false;
        int status = nSuccess;
        while (update_l1norm > xtol_n) {
            if (k >= max_iter) {
                status = nMaxIter;
                break;
            }
            const vec& g = obj.gradient(xk.data());
            for (std::size_t i = 0; i < n; ++i) b[i] = -g[i];
            const double maggrad = abs_sum(b, work);
            const double eta = pymin(0.5, std::sqrt(maggrad));
            const double termcond = eta * maggrad;
            for (std::size_t i = 0; i < n; ++i) {
                xsupi[i] = 0.0;
                ri[i] = -b[i];
                psupi[i] = -ri[i];
            }
            int i = 0;
            double dri0 = np_dot(ri.data(), ri.data(), n);
            if (hmode == kHessCallable) {
                hs.update(xk.data());
                hcalls += 1;
            }
            bool converged = false, nan_curv = false;
            bool vanished = false;  // approx_fhess_p's probe x + eps p rounded to x: Ap says nothing
            for (std::int64_t k2 = 0; k2 < cg_maxiter; ++k2) {
                if (abs_sum(ri, work) <= termcond) {
                    converged = true;
                    break;
                }
                if (hmode == kHessCallable) {
                    std::optional<nb::gil_scoped_release> rel;
                    if (big) rel.emplace();
                    np_matvec(hs.H->data(), psupi.data(), Ap.data(), n);
                } else if (hmode == kHessTwoPoint || hmode == kHessThreePoint) {
                    hs.update(xk.data());  // sf.hess(xk).dot(p)
                    hs.difference_product(psupi.data(), Ap.data());
                    hcalls += 1;
                } else if (user_hessp) {
                    hs.user_product(xk.data(), psupi.data(), Ap.data());
                    hcalls += 1;
                } else {
                    // approx_fhess_p(xk, p, fprime, epsilon) with fprime(xk) the iteration's -b
                    vanished = true;
                    for (std::size_t j = 0; j < n; ++j) {
                        xe[j] = xk[j] + eps[j] * psupi[j];
                        vanished = vanished && xe[j] == xk[j];
                    }
                    const vec& f2 = obj.gradient(xe.data());
                    for (std::size_t j = 0; j < n; ++j) Ap[j] = (f2[j] - (-b[j])) / eps[j];
                }
                const double curv = np_dot(psupi.data(), Ap.data(), n);
                if (std::isnan(curv)) {
                    nan_curv = true;
                    break;
                }
                if (0 <= curv && curv <= eps3) {
                    // no curvature along -g at the first step (also relative to |g|^2, so not a
                    // tiny g near the solution): -g, the line search sizes it
                    if (i == 0 && curv <= eps3 * dri0 && !vanished) {
                        for (std::size_t j = 0; j < n; ++j) xsupi[j] = b[j];
                    }
                    converged = true;
                    break;
                } else if (curv < 0) {
                    if (i == 0) {  // fall back to the steepest descent direction
                        const double t = dri0 / (-curv);
                        for (std::size_t j = 0; j < n; ++j) xsupi[j] = t * b[j];
                    }
                    converged = true;
                    break;
                }
                const double alphai = dri0 / curv;
                for (std::size_t j = 0; j < n; ++j) {
                    xsupi[j] = xsupi[j] + alphai * psupi[j];
                    ri[j] = ri[j] + alphai * Ap[j];
                }
                const double dri1 = np_dot(ri.data(), ri.data(), n);
                const double betai = dri1 / dri0;
                for (std::size_t j = 0; j < n; ++j) psupi[j] = -ri[j] + betai * psupi[j];
                i += 1;
                dri0 = dri1;
            }
            if (nan_curv) {
                status = nNaN;
                break;
            }
            if (!converged) {
                status = nCg;
                break;
            }
            ls.xk = xk;
            ls.pk = xsupi;
            for (std::size_t j = 0; j < n; ++j) gfk[j] = -b[j];
            ls.gfk = gfk;
            double alphak, fval;
            if (!ls.wolfe1(old_fval, old_old_fval, has_old_old, alphak, fval)) {
                const int r = ls.wolfe2(old_fval, old_old_fval, has_old_old, alphak, fval);
                if (r == 0) {
                    status = nPrLoss;
                    break;
                }
            }
            if (std::isnan(fval) && !std::isnan(old_fval)) {  // the last finite iterate (O59's rule)
                k += 1;
                status = nNaN;
                break;
            }
            old_old_fval = old_fval;
            has_old_old = true;
            old_fval = fval;
            for (std::size_t j = 0; j < n; ++j) {
                update[j] = alphak * xsupi[j];
                xk[j] = xk[j] + update[j];
            }
            k += 1;
            update_l1norm = abs_sum(update, work);
        }
        if (status == nSuccess && (std::isnan(old_fval) || std::isnan(update_l1norm))) status = nNaN;
        const vec& gx = obj.gradient(xk.data());  // O70: at x
        const std::int64_t nhev = hmode == kHessCallable ? hs.calls : hcalls;
        return nb::make_tuple(
            out_vector(xk.data(), n), old_fval, out_vector(gx.data(), n), status, k, obj.nfev, obj.ngev, nhev
        );
    });
}

}  // namespace

void bind_newton_cg(nb::module_& m) {
    m.def(
        "newton_cg_solve", &newton_cg_solve, nb::arg("func"), nb::arg("fun"), nb::arg("jac").none(),
        nb::arg("hess").none(), nb::arg("hessp").none(), nb::arg("check"), nb::arg("mode"), nb::arg("hmode"),
        nb::arg("x0"), nb::arg("xtol"), nb::arg("eps"), nb::arg("max_iter"), nb::arg("c1"), nb::arg("c2")
    );
}

}  // namespace sd::optimize
