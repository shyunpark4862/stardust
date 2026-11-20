// minimize(method='nelder-mead') (docs/design/optimize.md §3.5): scipy 1.18.1's
// _minimize_neldermead, the same steps (O9).
//
// The simplex of N + 1 vertices starts at x0 and x0 with x0[k] scaled by 1.05 (0.00025 when it is
// 0), or is the user's initial_simplex; with bounds, a vertex above an upper bound is reflected
// into the box and every vertex clipped. Each iteration tries the reflection, then the
// expansion, the outside or the inside contraction, or shrinks the simplex towards its best
// vertex (coefficients 1, 2, 1/2, 1/2, or the adaptive ones of Gao and Han 2012 with adaptive);
// with bounds every new point is clipped. It stops when every vertex is within xatol of the best
// in each coordinate and every value within fatol of the best, or at max_nfev evaluations (a
// call that would pass it is not made: the iteration stops there) or max_iter iterations.
//
// The vertices are kept sorted by value with numpy's argsort (its introsort: insertion sort
// below 17 elements, so equal values keep their order only there), as scipy sorts them; the
// sums and the coefficients are numpy's, in its order (the centroid adds the rows from +0.0 on,
// one after the other). Sorting moves row indices here, not the rows.
#include "optimize/minimize.hpp"

#include "optimize/objective.hpp"

#include "bind/errors.hpp"

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
using ivec = std::vector<std::int64_t>;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// numpy's float order: NaN after everything.
inline bool np_less(double a, double b) { return a < b || (b != b && a == a); }

int msb(std::int64_t n) {
    int d = 0;
    while (n > 1) {
        n >>= 1;
        ++d;
    }
    return d;
}

// numpy's aheapsort_ on t[0, n) (the depth-limit fallback of its introsort).
void aheapsort(const double* v, std::int64_t* t, std::int64_t n) {
    std::int64_t* a = t - 1;  // 1-based
    for (std::int64_t l = n >> 1; l > 0; --l) {
        const std::int64_t tmp = a[l];
        std::int64_t i = l, j = l << 1;
        while (j <= n) {
            if (j < n && np_less(v[a[j]], v[a[j + 1]])) ++j;
            if (np_less(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }
    for (; n > 1;) {
        const std::int64_t tmp = a[n];
        a[n] = a[1];
        n -= 1;
        std::int64_t i = 1, j = 2;
        while (j <= n) {
            if (j < n && np_less(v[a[j]], v[a[j + 1]])) ++j;
            if (np_less(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }
}

// numpy's np.argsort(v) (kind 'quicksort': aquicksort_ for doubles).
void argsort(const double* v, std::int64_t num, std::int64_t* t) {
    for (std::int64_t i = 0; i < num; ++i) t[i] = i;
    if (num < 2) return;
    std::int64_t* pl = t;
    std::int64_t* pr = t + num - 1;
    std::vector<std::int64_t*> stack;
    std::vector<int> depth;
    int cdepth = msb(num) * 2;
    for (;;) {
        if (cdepth < 0) {
            aheapsort(v, pl, pr - pl + 1);
        } else {
            while ((pr - pl) > 15) {
                std::int64_t* pm = pl + ((pr - pl) >> 1);
                if (np_less(v[*pm], v[*pl])) std::swap(*pm, *pl);
                if (np_less(v[*pr], v[*pm])) std::swap(*pr, *pm);
                if (np_less(v[*pm], v[*pl])) std::swap(*pm, *pl);
                const double vp = v[*pm];
                std::int64_t* pi = pl;
                std::int64_t* pj = pr - 1;
                std::swap(*pm, *pj);
                for (;;) {
                    do ++pi;
                    while (np_less(v[*pi], vp));
                    do --pj;
                    while (np_less(vp, v[*pj]));
                    if (pi >= pj) break;
                    std::swap(*pi, *pj);
                }
                std::int64_t* pk = pr - 1;
                std::swap(*pi, *pk);
                if (pi - pl < pr - pi) {
                    stack.push_back(pi + 1);
                    stack.push_back(pr);
                    pr = pi - 1;
                } else {
                    stack.push_back(pl);
                    stack.push_back(pi - 1);
                    pl = pi + 1;
                }
                depth.push_back(--cdepth);
            }
            for (std::int64_t* pi = pl + 1; pi <= pr; ++pi) {
                const std::int64_t vi = *pi;
                const double vp = v[vi];
                std::int64_t* pj = pi;
                std::int64_t* pk = pi - 1;
                while (pj > pl && np_less(vp, v[*pk])) *pj-- = *pk--;
                *pj = vi;
            }
        }
        if (stack.empty()) break;
        pr = stack.back();
        stack.pop_back();
        pl = stack.back();
        stack.pop_back();
        cdepth = depth.back();
        depth.pop_back();
    }
}

// numpy's clip for doubles: NaN stays, max(x, lo) then min(., hi) with numpy's comparisons.
inline double np_clip(double x, double lo, double hi) {
    if (x != x) return x;
    const double t = x > lo ? x : lo;
    return t < hi ? t : hi;
}

struct Simplex {
    std::size_t N;
    vec pool;          // (N + 1) rows of N
    ivec order;        // order[k]: the row of the k-th vertex
    vec fsim;          // the vertices' values, in vertex order
    ivec ind;          // scratch: argsort's result
    vec ftmp;
    double* row(std::size_t k) { return pool.data() + static_cast<std::size_t>(order[k]) * N; }

    void sort() {
        argsort(fsim.data(), static_cast<std::int64_t>(N + 1), ind.data());
        ivec o(N + 1);
        for (std::size_t k = 0; k <= N; ++k) {
            o[k] = order[static_cast<std::size_t>(ind[k])];
            ftmp[k] = fsim[static_cast<std::size_t>(ind[k])];
        }
        order.swap(o);
        fsim.swap(ftmp);
    }
};

// nelder_mead_solve(func, fun, check, x0, sim0, lb, ub, bounded, adaptive, xatol, fatol,
//                   max_iter, max_nfev)
//   -> (x, fun, n_iter, nfev, status, sim (N + 1, N), fsim (N + 1,))
// sim0: the initial simplex, row-major (N + 1) x N, or empty for scipy's. max_iter and
// max_nfev: < 0 for no cap (scipy's inf). status: scipy's warnflag (0 converged, 1 max_nfev,
// 2 max_iter).
nb::object nelder_mead_solve(
    const std::string& func,
    nb::callable fun,
    nb::callable check,
    const vec& x0,
    const vec& sim0,
    const vec& lb,
    const vec& ub,
    bool bounded,
    bool adaptive,
    double xatol,
    double fatol,
    std::int64_t max_iter,
    std::int64_t max_nfev
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Objective obj;
        obj.fun = fun;
        obj.check = check;
        obj.fn = fn;
        obj.n = x0.size();
        const std::size_t N = x0.size();
        const double maxiter = max_iter < 0 ? kInf : static_cast<double>(max_iter);
        const double maxfun = max_nfev < 0 ? kInf : static_cast<double>(max_nfev);

        double rho, chi, psi, sigma;
        if (adaptive) {
            const double dim = static_cast<double>(N);
            rho = 1;
            chi = 1 + 2 / dim;
            psi = 0.75 - 1 / (2 * dim);
            sigma = 1 - 1 / dim;
        } else {
            rho = 1;
            chi = 2;
            psi = 0.5;
            sigma = 0.5;
        }
        const double nonzdelt = 0.05, zdelt = 0.00025;

        Simplex S;
        S.N = N;
        S.pool.assign((N + 1) * N, 0.0);
        S.order.resize(N + 1);
        S.ind.resize(N + 1);
        S.ftmp.resize(N + 1);
        for (std::size_t k = 0; k <= N; ++k) S.order[k] = static_cast<std::int64_t>(k);
        if (sim0.empty()) {
            for (std::size_t j = 0; j < N; ++j) S.pool[j] = x0[j];
            for (std::size_t k = 0; k < N; ++k) {
                double* y = S.pool.data() + (k + 1) * N;
                for (std::size_t j = 0; j < N; ++j) y[j] = x0[j];
                y[k] = y[k] != 0 ? (1 + nonzdelt) * y[k] : zdelt;
            }
        } else {
            S.pool = sim0;
        }
        if (bounded) {
            for (std::size_t k = 0; k <= N; ++k)
                for (std::size_t j = 0; j < N; ++j) {
                    double& v = S.pool[k * N + j];
                    if (v > ub[j]) v = 2 * ub[j] - v;
                    v = np_clip(v, lb[j], ub[j]);
                }
        }
        S.fsim.assign(N + 1, kInf);

        double fcalls = 0;
        // scipy's _wrap_scalar_function_maxfun_validation: no call past max_nfev
        auto call = [&](const double* v, double& out) -> bool {
            if (fcalls >= maxfun) return false;
            fcalls += 1;
            out = obj.call_f(v);
            return true;
        };
        auto clip = [&](double* v) {
            if (bounded)
                for (std::size_t j = 0; j < N; ++j) v[j] = np_clip(v[j], lb[j], ub[j]);
        };

        for (std::size_t k = 0; k <= N; ++k) {
            double v;
            if (!call(S.row(k), v)) break;
            S.fsim[k] = v;
            if (k == 0 && !std::isfinite(v))
                throw nb::value_error(
                    sim0.empty() ? "'fun' is not finite at 'x0'"
                                 : "'fun' is not finite at the first vertex of 'initial_simplex'"
                );
        }
        S.sort();
        S.sort();

        vec xbar(N), xr(N), xe(N), xc(N), xcc(N);
        double iterations = 1;
        const bool big = double(N) * double(N) > 2e5;  // the 10 us rule: the centroid and the test
        while (fcalls < maxfun && iterations < maxiter) {
            bool stop = false;
            {
                std::optional<nb::gil_scoped_release> rel;
                if (big) rel.emplace();
                // scipy's test max|sim[1:] - sim[0]| <= xatol and max|fsim[0] - fsim[1:]| <= fatol:
                // every entry within (NaN never is), so the first one that is not ends it
                const double* s0 = S.row(0);
                bool small = true;
                for (std::size_t k = 1; k <= N && small; ++k) {
                    const double* sk = S.row(k);
                    for (std::size_t j = 0; j < N; ++j)
                        if (!(std::abs(sk[j] - s0[j]) <= xatol)) {
                            small = false;
                            break;
                        }
                }
                for (std::size_t k = 1; k <= N && small; ++k) small = std::abs(S.fsim[0] - S.fsim[k]) <= fatol;
                stop = small;
                if (!stop) {
                    double* __restrict xb = xbar.data();
                    for (std::size_t j = 0; j < N; ++j) xb[j] = 0.0;
                    for (std::size_t k = 0; k < N; ++k) {
                        const double* __restrict sk = S.row(k);
                        for (std::size_t j = 0; j < N; ++j) xb[j] = xb[j] + sk[j];
                    }
                    const double dN = static_cast<double>(N);
                    for (std::size_t j = 0; j < N; ++j) xb[j] = xb[j] / dN;
                }
            }
            if (stop) break;
            double* last = S.row(N);
            bool limit = false;
            do {
                for (std::size_t j = 0; j < N; ++j) xr[j] = (1 + rho) * xbar[j] - rho * last[j];
                clip(xr.data());
                double fxr;
                if (!call(xr.data(), fxr)) {
                    limit = true;
                    break;
                }
                bool doshrink = false;
                if (fxr < S.fsim[0]) {
                    const double a = 1 + rho * chi, b = rho * chi;
                    for (std::size_t j = 0; j < N; ++j) xe[j] = a * xbar[j] - b * last[j];
                    clip(xe.data());
                    double fxe;
                    if (!call(xe.data(), fxe)) {
                        limit = true;
                        break;
                    }
                    if (fxe < fxr) {
                        std::copy(xe.begin(), xe.end(), last);
                        S.fsim[N] = fxe;
                    } else {
                        std::copy(xr.begin(), xr.end(), last);
                        S.fsim[N] = fxr;
                    }
                } else if (fxr < S.fsim[N - 1]) {
                    std::copy(xr.begin(), xr.end(), last);
                    S.fsim[N] = fxr;
                } else if (fxr < S.fsim[N]) {
                    const double a = 1 + psi * rho, b = psi * rho;
                    for (std::size_t j = 0; j < N; ++j) xc[j] = a * xbar[j] - b * last[j];
                    clip(xc.data());
                    double fxc;
                    if (!call(xc.data(), fxc)) {
                        limit = true;
                        break;
                    }
                    if (fxc <= fxr) {
                        std::copy(xc.begin(), xc.end(), last);
                        S.fsim[N] = fxc;
                    } else {
                        doshrink = true;
                    }
                } else {
                    const double a = 1 - psi;
                    for (std::size_t j = 0; j < N; ++j) xcc[j] = a * xbar[j] + psi * last[j];
                    clip(xcc.data());
                    double fxcc;
                    if (!call(xcc.data(), fxcc)) {
                        limit = true;
                        break;
                    }
                    if (fxcc < S.fsim[N]) {
                        std::copy(xcc.begin(), xcc.end(), last);
                        S.fsim[N] = fxcc;
                    } else {
                        doshrink = true;
                    }
                }
                if (doshrink) {
                    const double* s0 = S.row(0);
                    for (std::size_t k = 1; k <= N; ++k) {
                        double* sk = S.row(k);
                        for (std::size_t j = 0; j < N; ++j) sk[j] = s0[j] + sigma * (sk[j] - s0[j]);
                        clip(sk);
                        double v;
                        if (!call(sk, v)) {
                            limit = true;
                            break;
                        }
                        S.fsim[k] = v;
                    }
                }
            } while (false);
            if (!limit) iterations += 1;
            S.sort();
        }

        double fval = S.fsim[0];
        for (std::size_t k = 1; k <= N; ++k) {
            const double v = S.fsim[k];
            fval = (fval != fval || v != v) ? kNaN : (v < fval ? v : fval);
        }
        int warnflag = 0;
        if (fcalls >= maxfun) warnflag = 1;
        else if (iterations >= maxiter) warnflag = 2;
        vec sim((N + 1) * N);
        for (std::size_t k = 0; k <= N; ++k)
            std::copy(S.row(k), S.row(k) + N, sim.begin() + static_cast<std::ptrdiff_t>(k * N));
        // n_iter: the iterations done (scipy's nit counts one more: its loop starts at 1)
        return nb::make_tuple(
            out_vector(sim.data(), N), fval, static_cast<std::int64_t>(iterations) - 1,
            static_cast<std::int64_t>(fcalls), warnflag, out_matrix(sim.data(), N + 1, N),
            out_vector(S.fsim.data(), N + 1)
        );
    });
}

}  // namespace

void bind_nelder_mead(nb::module_& m) {
    m.def(
        "nelder_mead_solve", &nelder_mead_solve, nb::arg("func"), nb::arg("fun"), nb::arg("check"), nb::arg("x0"),
        nb::arg("sim0"), nb::arg("lb"), nb::arg("ub"), nb::arg("bounded"), nb::arg("adaptive"), nb::arg("xatol"),
        nb::arg("fatol"), nb::arg("max_iter"), nb::arg("max_nfev")
    );
}

}  // namespace sd::optimize
