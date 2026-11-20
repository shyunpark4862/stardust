// minimize(method='trust-ncg' / 'trust-exact') (docs/design/optimize.md §3.5): scipy 1.18.1's
// _minimize_trust_region with its subproblems CGSteihaugSubproblem (_trustregion_ncg.py) and
// IterativeSubproblem (_trustregion_exact.py), the same steps (O9).
//
// The driver: while |g| >= gtol, solve the subproblem in the radius, predict m(p) = f + g.p +
// p.Bp / 2, compute f at x + p (trust-exact first builds that point's subproblem, which computes
// hess there), stop with status 2 when the predicted reduction is not positive, shrink the radius
// by 1/4 when rho = actual / predicted < 1/4, double it (up to max_trust_radius) when rho > 3/4
// and the step is on the boundary, accept the step when rho > eta; then stop when |g| < gtol
// (status 0) or at max_iter iterations (status 1). max_iter = 0 stops at x0 before an iteration
// (O75; scipy checks the cap only after the first).
//
// trust-ncg: Steihaug-Toint conjugate gradients from 0 (Nocedal and Wright's algorithm 7.2) with
// the tolerance min(1/2, sqrt|g|) |g|; a direction of nonpositive curvature goes to the boundary
// point with the lower model value, a step out of the region to the boundary. scipy's loop has no
// cap; here 20n steps (Newton-CG's cap; with a symmetric Hessian scipy's runs take about n, at
// most 1.2n) stop with status 3, as a linalg error (a nonsymmetric hess can make it cycle). The products are
// the user's hessp, the difference operator of hess='2-point' / '3-point' (hessian.hpp), or the
// matrix of hess (computed when first needed at a point).
// trust-exact: the nearly exact solution of Moré and Sorensen as scipy's IterativeSubproblem:
// Cholesky factorizations of H + lambda I (LAPACK potrf, upper, the lower triangle cleared as
// scipy's wrapper does), the Newton step on lambda, the hard case by Cline-Moler-Stewart-
// Wilkinson's estimate of the smallest singular value, the bounds of Conn, Gould and Toint
// (Gershgorin, the Frobenius and infinity norms), k_easy 0.1, k_hard 0.2, at most
// subproblem_max_iter factorization steps.
//
// The counts: nfev and njev ScalarFunction's, nhev the calls of hess (the first at x0) and the
// products of hessp and of the difference operator, without the 1 scipy adds for its placeholder
// hess when only hessp is given (O68); trust-ncg computes the result's hess at x when the last
// subproblem had not (as scipy's m.hess) and counts that call (O69; scipy does not).
//
// Where scipy raises (scipy.linalg's check_finite on |g|, on the Hessian of a new subproblem and
// inside the subproblem, math.sqrt of a negative number) the kernel stops with a negative status
// and the point, and Python raises ValueError.
//
// The arithmetic is numpy's and scipy.linalg's: dot products, matrix-vector products and 2-norms
// by Accelerate's BLAS as numpy calls it (hessian.hpp's np_dot and np_matvec; cblas_dnrm2), potrf,
// potrs, trtrs and lange from Accelerate's LAPACK on scipy's layouts (the column-major copy of
// H + lambda I; a leading block of the factor solved as its transpose, as scipy.linalg's
// solve_triangular does for a non-contiguous slice), sums numpy's pairwise sums, Python's max and
// min, no fused multiply-adds.
#include "optimize/hessian.hpp"
#include "optimize/minimize.hpp"
#include "optimize/objective.hpp"
#include "optimize/wolfe.hpp"

#include "bind/errors.hpp"
#include "linalg/lapack.hpp"
#include "linalg/types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sd::optimize {

namespace {

using namespace linesearch;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kUpdateCoeff = 0.01, kEasy = 0.1, kHard = 0.2;

// the kernel's status: scipy's warnflag (0-3), or where scipy raises (negative)
enum TrStatus : int {
    trSuccess = 0,
    trMaxIter = 1,
    trBadPrediction = 2,
    trLinAlg = 3,
    trJacNotFinite = -1,
    trHessNotFinite = -2,
    trSubproblemNotFinite = -3,
};

// Where scipy raises: the status and the point (unwinds to the driver).
struct Stop {
    int status;
};

bool all_finite(const double* a, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(a[i])) return false;
    return true;
}

double ddot(const vec& a, const vec& b) { return np_dot(a.data(), b.data(), a.size()); }

// scipy.linalg.norm of a vector: BLAS nrm2 after check_finite.
double nrm2(const vec& a) {
    if (!all_finite(a.data(), a.size())) throw Stop{trSubproblemNotFinite};
    return cblas_dnrm2(static_cast<lint>(a.size()), a.data(), 1);
}

// The subproblem at a point (BaseQuadraticSubproblem): fun, jac, |jac| and hess when first asked
// for; trust-exact's constants of the Hessian and its state between solves.
struct Model {
    vec x;
    double f = 0;
    vec g;
    std::shared_ptr<const HessMatrix> H;
    double gmag = 0;
    bool has_f = false, has_g = false, has_gmag = false, has_H = false;
    // IterativeSubproblem
    double gersh_lb = 0, gersh_ub = 0, hinf = 0, hfro = 0, close_to_zero = 0;
    double previous_tr = -1, lambda_lb = 0;
};

struct Trust {
    Objective& obj;
    Hessian& hs;
    std::size_t n;
    lint ln;
    int hmode;
    bool user_hessp, exact;
    std::int64_t sub_max_iter;
    std::int64_t nhessp = 0;
    bool big;  // the 10 us rule for the factorizations
    vec Bp, Hr, U, C, work, pp, pm, zmin, vv, ww;

    Trust(Objective& o, Hessian& h, int hm, bool uh, bool ex, std::int64_t smi)
        : obj(o), hs(h), n(o.n), ln(static_cast<lint>(o.n)), hmode(hm), user_hessp(uh), exact(ex), sub_max_iter(smi) {
        big = double(n) * double(n) * double(n) / 3 > 2e5;
        Bp.resize(n);
    }

    double fun(Model& m) {
        if (!m.has_f) {
            m.f = obj.value(m.x.data());
            m.has_f = true;
        }
        return m.f;
    }
    const vec& jac(Model& m) {
        if (!m.has_g) {
            m.g = obj.gradient(m.x.data());
            m.has_g = true;
        }
        return m.g;
    }
    double jac_mag(Model& m) {
        if (!m.has_gmag) {
            const vec& g = jac(m);
            if (!all_finite(g.data(), n)) throw Stop{trJacNotFinite};
            m.gmag = cblas_dnrm2(ln, g.data(), 1);
            m.has_gmag = true;
        }
        return m.gmag;
    }
    const HessMatrix& hess(Model& m) {
        if (!m.has_H) {
            hs.update(m.x.data());
            m.H = hs.H;
            m.has_H = true;
        }
        return *m.H;
    }
    // m.hessp(p) into out
    void hessp(Model& m, const double* p, double* out) {
        if (user_hessp) {
            hs.user_product(m.x.data(), p, out);
            ++nhessp;
        } else if (hmode == kHessTwoPoint || hmode == kHessThreePoint) {
            hs.update(m.x.data());  // sf.hess(x).dot(p)
            hs.difference_product(p, out);
            ++nhessp;
        } else {
            np_matvec(hess(m).data(), p, out, n);
        }
    }
    // m(p) = fun + jac.p + 0.5 p.hessp(p)
    double model_value(Model& m, const vec& p) {
        const double f = fun(m);
        const double t1 = ddot(jac(m), p);
        hessp(m, p.data(), Bp.data());
        const double t2 = ddot(p, Bp);
        return f + t1 + 0.5 * t2;
    }

    // The two t with |z + t d| = r, low to high (get_boundaries_intersections).
    static std::pair<double, double> intersections(const vec& z, const vec& d, double r) {
        const double a = ddot(d, d);
        const double b = 2 * ddot(z, d);
        const double c = ddot(z, z) - r * r;
        const double disc = b * b - 4 * a * c;
        if (disc < 0) throw Stop{trSubproblemNotFinite};  // math.sqrt raises
        const double sd = std::sqrt(disc);
        const double aux = b + std::copysign(sd, b);
        const double ta = -aux / (2 * a);
        const double tb = -2 * c / aux;
        if (tb < ta) return {tb, ta};
        return {ta, tb};
    }

    // ---- trust-ncg

    void solve_ncg(Model& m, double tr, vec& p, bool& hits) {
        const double jm = jac_mag(m);
        const double tolerance = pymin(0.5, std::sqrt(jm)) * jm;
        p.assign(n, 0.0);
        if (jm < tolerance) {
            hits = false;
            return;
        }
        vec z(n, 0.0), r = jac(m), d(n), Bd(n), zn(n), rn(n), pa(n), pb(n);
        for (std::size_t i = 0; i < n; ++i) d[i] = -r[i];
        const std::size_t cap = 20 * n;  // scipy has none: a nonsymmetric hess can cycle forever
        for (std::size_t step = 0;; ++step) {
            if (step == cap) throw Stop{trLinAlg};
            hessp(m, d.data(), Bd.data());
            const double dBd = ddot(d, Bd);
            if (dBd <= 0) {
                const auto [ta, tb] = intersections(z, d, tr);
                for (std::size_t i = 0; i < n; ++i) {
                    pa[i] = z[i] + ta * d[i];
                    pb[i] = z[i] + tb * d[i];
                }
                const double va = model_value(m, pa);
                const double vb = model_value(m, pb);
                p = va < vb ? pa : pb;
                hits = true;
                return;
            }
            const double r_squared = ddot(r, r);
            const double alpha = r_squared / dBd;
            for (std::size_t i = 0; i < n; ++i) zn[i] = z[i] + alpha * d[i];
            if (nrm2(zn) >= tr) {
                const auto tab = intersections(z, d, tr);
                const double tb = tab.second;
                for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + tb * d[i];
                hits = true;
                return;
            }
            for (std::size_t i = 0; i < n; ++i) rn[i] = r[i] + alpha * Bd[i];
            const double rn_squared = ddot(rn, rn);
            if (std::sqrt(rn_squared) < tolerance) {
                p = zn;
                hits = false;
                return;
            }
            const double beta = rn_squared / r_squared;
            for (std::size_t i = 0; i < n; ++i) d[i] = -rn[i] + beta * d[i];
            std::swap(z, zn);
            std::swap(r, rn);
        }
    }

    // ---- trust-exact

    // IterativeSubproblem.__init__: hess at the point (scipy's norm(hess, inf) checks it is
    // finite), its Gershgorin bounds and norms.
    void init_exact(Model& m) {
        const HessMatrix& H = hess(m);
        if (!all_finite(H.data(), n * n)) throw Stop{trHessNotFinite};
        work.resize(n);
        double lb = 0, ub = 0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) work[j] = std::abs(H[i * n + j]);
            const double rs = pairwise_sum(work.data(), n);
            const double d = H[i * n + i], da = std::abs(d);
            const double l = d + da - rs, u = d - da + rs;
            if (i == 0 || l < lb) lb = l;
            if (i == 0 || u > ub) ub = u;
        }
        m.gersh_lb = lb;
        m.gersh_ub = ub;
        // norm(hess, inf): lange('1', hess.T), the row-major matrix read as column-major
        work.resize(n);
        m.hinf = dlange_("1", &ln, &ln, H.data(), &ln, work.data());
        m.hfro = std::sqrt(np_dot(H.data(), H.data(), n * n));  // numpy: sqrt(ravel . ravel)
        m.close_to_zero = double(n) * kEps * m.hinf;
    }

    // numpy's H + lambda np.eye(n), row-major, into Hr.
    void shifted(const HessMatrix& H, double lambda) {
        Hr.resize(n * n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) Hr[i * n + j] = H[i * n + j] + lambda * (i == j ? 1.0 : 0.0);
    }

    // The upper Cholesky factor of H + lambda I in out (column-major as scipy's copy for potrf,
    // the lower triangle cleared as its wrapper does): potrf's info. potrf reads only the upper
    // triangle, so only that is copied (by tiles: the copy is a transpose).
    lint factor(const HessMatrix& H, double lambda, vec& out) {
        out.resize(n * n);
        constexpr std::size_t kTile = 64;
        for (std::size_t ib = 0; ib < n; ib += kTile)
            for (std::size_t jb = ib; jb < n; jb += kTile) {
                const std::size_t ie = std::min(ib + kTile, n), je = std::min(jb + kTile, n);
                for (std::size_t i = ib; i < ie; ++i)
                    for (std::size_t j = std::max(jb, i); j < je; ++j)
                        out[i + j * n] = H[i * n + j] + lambda * (i == j ? 1.0 : 0.0);
            }
        lint info;
        {
            std::optional<nb::gil_scoped_release> rel;
            if (big) rel.emplace();
            info = linalg::lapack::potrf('U', ln, out.data(), ln);
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) out[j + i * n] = 0.0;
        return info;
    }

    // estimate_smallest_singular_value(U): s_min and z_min (into zmin).
    double smallest_singular(const vec& Uf) {
        vec p(n, 0.0);
        ww.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            const double ukk = Uf[k + k * n];
            const double wp = (1 - p[k]) / ukk;
            const double wm = (-1 - p[k]) / ukk;
            const std::size_t m = n - k - 1;
            pp.resize(m);
            pm.resize(m);
            for (std::size_t j = 0; j < m; ++j) {
                const double u = Uf[k + (k + 1 + j) * n];
                pp[j] = p[k + 1 + j] + u * wp;
                pm[j] = p[k + 1 + j] + u * wm;
            }
            if (!all_finite(pp.data(), m)) throw Stop{trSubproblemNotFinite};
            work.resize(m);
            for (std::size_t j = 0; j < m; ++j) work[j] = std::abs(pp[j]);
            const double sp = std::abs(wp) + pairwise_sum(work.data(), m);
            if (!all_finite(pm.data(), m)) throw Stop{trSubproblemNotFinite};
            for (std::size_t j = 0; j < m; ++j) work[j] = std::abs(pm[j]);
            const double sm = std::abs(wm) + pairwise_sum(work.data(), m);
            if (sp >= sm) {
                ww[k] = wp;
                for (std::size_t j = 0; j < m; ++j) p[k + 1 + j] = pp[j];
            } else {
                ww[k] = wm;
                for (std::size_t j = 0; j < m; ++j) p[k + 1 + j] = pm[j];
            }
        }
        if (!all_finite(Uf.data(), n * n) || !all_finite(ww.data(), n)) throw Stop{trSubproblemNotFinite};
        vv = ww;
        if (linalg::lapack::trtrs('U', 'N', 'N', ln, 1, Uf.data(), ln, vv.data(), ln) > 0) throw Stop{trLinAlg};
        const double v_norm = nrm2(vv);
        const double w_norm = nrm2(ww);
        zmin.resize(n);
        for (std::size_t i = 0; i < n; ++i) zmin[i] = vv[i] / v_norm;
        return w_norm / v_norm;
    }

    // singular_leading_submatrix(H, U, k): delta, and v into vv; hkk is H[k-1, k-1].
    double leading_submatrix(double hkk, const vec& Uf, lint k) {
        const auto km = static_cast<std::size_t>(k - 1);
        work.resize(km);
        for (std::size_t i = 0; i < km; ++i) {
            const double u = Uf[i + km * n];
            work[i] = u * u;
        }
        const double delta = pairwise_sum(work.data(), km) - hkk;
        vv.assign(n, 0.0);
        vv[km] = 1;
        if (k != 1) {
            vec rhs(km), L(km * km);
            for (std::size_t i = 0; i < km; ++i) rhs[i] = -Uf[i + km * n];
            for (std::size_t j = 0; j < km; ++j)
                for (std::size_t i = 0; i < km; ++i) {
                    if (!std::isfinite(Uf[i + j * n])) throw Stop{trSubproblemNotFinite};
                }
            if (!all_finite(rhs.data(), km)) throw Stop{trSubproblemNotFinite};
            const auto lk = static_cast<lint>(km);
            lint info;
            if (km == 1) {  // a 1 x 1 slice is Fortran-contiguous: solved as it is
                L[0] = Uf[0];
                info = linalg::lapack::trtrs('U', 'N', 'N', lk, 1, L.data(), lk, rhs.data(), lk);
            } else {  // a non-contiguous slice: its transpose, copied, lower and transposed
                for (std::size_t j = 0; j < km; ++j)
                    for (std::size_t i = 0; i < km; ++i) L[i + j * km] = Uf[j + i * n];
                info = linalg::lapack::trtrs('L', 'T', 'N', lk, 1, L.data(), lk, rhs.data(), lk);
            }
            if (info > 0) throw Stop{trLinAlg};
            for (std::size_t i = 0; i < km; ++i) vv[i] = rhs[i];
        }
        return delta;
    }

    // IterativeSubproblem.solve
    void solve_exact(Model& m, double tr, vec& p, bool& hits) {
        const double jm = jac_mag(m);
        const HessMatrix& H = *m.H;
        // _initial_values
        double lambda_ub = pymax(0.0, jm / tr + pymin(pymin(-m.gersh_lb, m.hfro), m.hinf));
        double dmin = H[0];
        for (std::size_t i = 1; i < n; ++i) dmin = pymin(dmin, H[i * n + i]);
        double lambda_lb = pymax(pymax(0.0, -dmin), jm / tr - pymin(pymin(m.gersh_ub, m.hfro), m.hinf));
        if (tr < m.previous_tr) lambda_lb = pymax(m.lambda_lb, lambda_lb);
        double lambda_current;
        if (lambda_lb == 0) lambda_current = 0;
        else
            lambda_current =
                pymax(std::sqrt(lambda_lb * lambda_ub), lambda_lb + kUpdateCoeff * (lambda_ub - lambda_lb));

        hits = true;
        bool already = false, have_p = false;
        lint info = 0;
        std::int64_t niter = 0;
        const vec& g = jac(m);
        vec w(n), Hp(n);
        while (niter < sub_max_iter) {
            if (already) already = false;
            else info = factor(H, lambda_current, U);
            niter += 1;
            if (info == 0 && jm > m.close_to_zero) {
                // cho_solve((U, False), -jac)
                if (!all_finite(U.data(), n * n)) throw Stop{trSubproblemNotFinite};
                p.resize(n);
                for (std::size_t i = 0; i < n; ++i) p[i] = -g[i];
                {
                    std::optional<nb::gil_scoped_release> rel;
                    if (big) rel.emplace();
                    linalg::lapack::potrs('U', ln, 1, U.data(), ln, p.data(), ln);
                }
                have_p = true;
                const double p_norm = nrm2(p);
                if (p_norm <= tr && lambda_current == 0) {
                    hits = false;
                    break;
                }
                w = p;  // solve_triangular(U, p, trans='T')
                if (linalg::lapack::trtrs('U', 'T', 'N', ln, 1, U.data(), ln, w.data(), ln) > 0) throw Stop{trLinAlg};
                const double w_norm = nrm2(w);
                const double q = p_norm / w_norm;
                const double delta_lambda = q * q * (p_norm - tr) / tr;
                const double lambda_new = lambda_current + delta_lambda;
                if (p_norm < tr) {  // inside the boundary
                    const double s_min = smallest_singular(U);
                    const auto [ta, tb] = intersections(p, zmin, tr);
                    const double step_len = std::abs(tb) < std::abs(ta) ? tb : ta;
                    shifted(H, lambda_current);  // np.dot(H, p), H = hess + lambda I
                    np_matvec(Hr.data(), p.data(), Hp.data(), n);
                    const double quadratic_term = ddot(p, Hp);
                    const double relative_error =
                        (step_len * step_len * (s_min * s_min)) / (quadratic_term + lambda_current * (tr * tr));
                    if (relative_error <= kHard) {
                        for (std::size_t i = 0; i < n; ++i) p[i] = p[i] + step_len * zmin[i];
                        break;
                    }
                    lambda_ub = lambda_current;
                    lambda_lb = pymax(lambda_lb, lambda_current - s_min * s_min);
                    if (factor(H, lambda_new, C) == 0) {
                        lambda_current = lambda_new;
                        already = true;
                        std::swap(U, C);
                        info = 0;
                    } else {
                        lambda_lb = pymax(lambda_lb, lambda_new);
                        lambda_current = pymax(
                            std::sqrt(std::abs(lambda_lb * lambda_ub)),
                            lambda_lb + kUpdateCoeff * (lambda_ub - lambda_lb)
                        );
                    }
                } else {  // outside the boundary
                    const double relative_error = std::abs(p_norm - tr) / tr;
                    if (relative_error <= kEasy) break;
                    lambda_lb = lambda_current;
                    lambda_current = lambda_new;
                }
            } else if (info == 0 && jm <= m.close_to_zero) {
                if (lambda_current == 0) {
                    p.assign(n, 0.0);
                    have_p = true;
                    hits = false;
                    break;
                }
                const double s_min = smallest_singular(U);
                const double step_len = tr;
                p.resize(n);
                for (std::size_t i = 0; i < n; ++i) p[i] = step_len * zmin[i];
                have_p = true;
                if (step_len * step_len * (s_min * s_min) <= kHard * lambda_current * (tr * tr)) break;
                lambda_ub = lambda_current;
                lambda_lb = pymax(lambda_lb, lambda_current - s_min * s_min);
                lambda_current =
                    pymax(std::sqrt(lambda_lb * lambda_ub), lambda_lb + kUpdateCoeff * (lambda_ub - lambda_lb));
            } else {  // the factorization failed
                const auto kk = static_cast<std::size_t>(info - 1);
                const double delta = leading_submatrix(H[kk * n + kk] + lambda_current * 1.0, U, info);
                const double v_norm = nrm2(vv);
                lambda_lb = pymax(lambda_lb, lambda_current + delta / (v_norm * v_norm));
                lambda_current = pymax(
                    std::sqrt(std::abs(lambda_lb * lambda_ub)), lambda_lb + kUpdateCoeff * (lambda_ub - lambda_lb)
                );
            }
        }
        m.lambda_lb = lambda_lb;
        m.previous_tr = tr;
        if (!have_p) throw Stop{trLinAlg};  // scipy: no step was ever computed (UnboundLocalError)
    }
};

// trust_solve(func, fun, jac, hess, hessp, check, mode, hmode, exact, x0, initial_trust_radius,
//             max_trust_radius, eta, gtol, max_iter, subproblem_max_iter)
//   -> (x, fun, jac, hess or None, status, n_iter, nfev, njev, nhev)
// status: scipy's warnflag, or (negative) where scipy raises, with x the point.
nb::object trust_solve(
    const std::string& func,
    nb::callable fun,
    nb::object jac,
    nb::object hess,
    nb::object hessp,
    nb::callable check,
    int mode,
    int hmode,
    bool exact,
    const vec& x0,
    double initial_trust_radius,
    double max_trust_radius,
    double eta,
    double gtol,
    std::int64_t max_iter,
    std::int64_t subproblem_max_iter
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

        Trust T(obj, hs, hmode, !hessp.is_none(), exact, subproblem_max_iter);
        Model m;
        m.x = x0;
        double tr = initial_trust_radius;
        std::int64_t k = 0;
        int status = trSuccess;
        vec p, xp(n);
        bool hits = false;
        vec where;
        try {
            if (exact) T.init_exact(m);
            while (T.jac_mag(m) >= gtol) {
                if (max_iter == 0) {  // O75
                    status = trMaxIter;
                    break;
                }
                try {
                    if (exact) T.solve_exact(m, tr, p, hits);
                    else T.solve_ncg(m, tr, p, hits);
                } catch (const Stop& s) {
                    if (s.status != trLinAlg) throw;
                    status = trLinAlg;
                    break;
                }
                const double predicted_value = T.model_value(m, p);
                for (std::size_t i = 0; i < n; ++i) xp[i] = m.x[i] + p[i];
                Model mp;
                mp.x = xp;
                if (exact) {
                    where = xp;
                    T.init_exact(mp);
                }
                const double fm = T.fun(m);
                const double actual_reduction = fm - T.fun(mp);
                const double predicted_reduction = fm - predicted_value;
                if (predicted_reduction <= 0) {
                    status = trBadPrediction;
                    break;
                }
                const double rho = actual_reduction / predicted_reduction;
                if (rho < 0.25) tr *= 0.25;
                else if (rho > 0.75 && hits) tr = pymin(2 * tr, max_trust_radius);
                if (rho > eta) m = std::move(mp);
                k += 1;
                if (T.jac_mag(m) < gtol) {
                    status = trSuccess;
                    break;
                }
                if (k >= max_iter) {
                    status = trMaxIter;
                    break;
                }
            }
        } catch (const Stop& s) {
            const vec& at = s.status == trHessNotFinite && !where.empty() ? where : m.x;
            return nb::make_tuple(
                out_vector(at.data(), n), nb::none(), nb::none(), nb::none(), s.status, k, obj.nfev, obj.ngev, 0
            );
        }
        const double f = T.fun(m);
        const vec& g = T.jac(m);
        nb::object H = nb::none();
        if (hmode == kHessCallable) {
            const HessMatrix& Hm = T.hess(m);
            H = out_matrix(Hm.data(), n, n);
        }
        const std::int64_t nhev = hs.calls + T.nhessp;  // O68, O69: the calls made
        return nb::make_tuple(
            out_vector(m.x.data(), n), f, out_vector(g.data(), n), H, status, k, obj.nfev, obj.ngev, nhev
        );
    });
}

}  // namespace

void bind_trust(nb::module_& m) {
    m.def(
        "trust_solve", &trust_solve, nb::arg("func"), nb::arg("fun"), nb::arg("jac").none(), nb::arg("hess").none(),
        nb::arg("hessp").none(), nb::arg("check"), nb::arg("mode"), nb::arg("hmode"), nb::arg("exact"), nb::arg("x0"),
        nb::arg("initial_trust_radius"), nb::arg("max_trust_radius"), nb::arg("eta"), nb::arg("gtol"),
        nb::arg("max_iter"), nb::arg("subproblem_max_iter")
    );
}

}  // namespace sd::optimize
