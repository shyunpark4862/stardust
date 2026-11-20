// minimize(method='bfgs') (docs/design/optimize.md §3.5): scipy 1.18.1's _minimize_bfgs, the same
// steps (O9).
//
// The inverse Hessian approximation H starts at the identity (or hess_inv0); each iteration
// searches along p = -H g with scipy's _line_search_wolfe12 (wolfe.hpp: line_search_wolfe1 with
// steps in [1e-100, 1e100], then line_search_wolfe2 with amax 1e100); no step from either stops
// with status 2. Then H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T, BFGS's update: for n >= 2
// in the O(n^2) order (O99): with v = H y (H is symmetric), H - rho (s v^T + v s^T) + (rho^2 y^T v
// + rho) s s^T, each entry written so that H stays symmetric to the bit. scipy forms the same
// matrix by two O(n^3) products, so the steps differ from scipy's by rounding (the one exception
// to O9 in minimize). For n = 1 it is scipy's product form (O102): the expanded form computes
// H - 2H + H + s/y there and loses the curvature to cancellation when s/y << H. It stops when the
// gradient's norm ('norm', the largest |g_i| by default) is at most gtol, at max_iter
// iterations, when the step is below xrtol (|x| + xrtol), or when f is not finite. The direction's
// product H g and the update run without the GIL when large (the 10 us rule).
//
// The rest of the arithmetic is scipy's (numpy's, without fused multiply-adds): the dot products
// and the matrix-vector product are numpy's BLAS calls (Accelerate's cblas_ddot, cblas_dgemv), the
// 2-norms numpy's pairwise sums.
#include "optimize/minimize.hpp"

#include "optimize/objective.hpp"
#include "optimize/wolfe.hpp"

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

using namespace linesearch;

enum Warn : int { wSuccess = 0, wMaxIter = 1, wPrLoss = 2, wNaN = 3 };

// bfgs_solve(func, fun, jac, check, mode, x0, H0, gtol, norm, steps, max_iter, xrtol, c1, c2)
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

        LineSearch B(obj, c1, c2, 1e-100, 1e100, 1e100);
        B.xt.resize(n);
        double old_fval = obj.value(x0.data());
        B.gfk = obj.gradient(x0.data());
        std::int64_t k = 0;
        const auto ln = static_cast<lint>(n);
        vec Hk(n * n, 0.0), Hy(n);
        if (H0.empty()) {
            for (std::size_t i = 0; i < n; ++i) Hk[i * n + i] = 1.0;
        } else {
            Hk = H0;
        }
        const bool big = 8.0 * double(n) * double(n) > 2e5;  // the 10 us rule (the update's flops)
        double old_old_fval = old_fval + std::sqrt(dot(B.gfk, B.gfk)) / 2;
        B.xk = x0;
        B.pk.resize(n);
        int warnflag = 0;
        vec sk(n), yk(n), xkp1(n), gfkp1;
        double gnorm = vecnorm(B.gfk, norm, B.work);
        while (gnorm > gtol && k < max_iter) {
            {
                std::optional<nb::gil_scoped_release> rel;
                if (big) rel.emplace();
                cblas_dgemv(
                    CblasRowMajor, CblasNoTrans, ln, ln, 1.0, Hk.data(), ln, B.gfk.data(), 1, 0.0, B.pk.data(), 1
                );
            }
            for (std::size_t i = 0; i < n; ++i) B.pk[i] = -B.pk[i];
            double alpha_k, fval;
            bool have_g = true;
            if (B.wolfe1(old_fval, old_old_fval, true, alpha_k, fval)) {
                gfkp1 = B.gval;
            } else {
                const int r = B.wolfe2(old_fval, old_old_fval, true, alpha_k, fval);
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
                if (n == 1) {  // scipy's (1 - rho s y) H (1 - rho y s) + rho s s, its products by dgemm
                    const double a1 = 1.0 - sk[0] * yk[0] * rhok, a2 = 1.0 - yk[0] * sk[0] * rhok;
                    double t = 0;
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1.0, Hk.data(), 1, &a2, 1, 0.0, &t, 1
                    );
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1.0, &a1, 1, &t, 1, 0.0, Hk.data(), 1
                    );
                    Hk[0] = Hk[0] + rhok * sk[0] * sk[0];
                } else {
                    cblas_dgemv(
                        CblasRowMajor, CblasNoTrans, ln, ln, 1.0, Hk.data(), ln, yk.data(), 1, 0.0, Hy.data(), 1
                    );
                    double yHy = 0;
                    for (std::size_t i = 0; i < n; ++i) yHy += yk[i] * Hy[i];
                    const double c = rhok * rhok * yHy + rhok;
                    // entry (i, j) and (j, i) are the same expression with its symmetric terms
                    // swapped (s_i v_j + v_i s_j, s_i s_j), so a symmetric H stays symmetric
                    for (std::size_t i = 0; i < n; ++i) {
                        const double si = sk[i], vi = Hy[i];
                        double* row = Hk.data() + i * n;
                        for (std::size_t j = 0; j < n; ++j)
                            row[j] = row[j] - rhok * (si * Hy[j] + vi * sk[j]) + c * (si * sk[j]);
                    }
                }
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
