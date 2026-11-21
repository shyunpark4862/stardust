// Variable selection's linear core (docs/design/regression.md §3.8, G7b).
//
// Every candidate model of a selection is a least squares fit of y on some columns of the
// design [intercept X], on the same rows. All of them come from one upper triangle R (m x m,
// m = the design's columns + 1) of the whitened rows [intercept X y]: R^T R = A^T W A, so a
// column set's residual sum of squares is that of the same columns of R, a problem of m rows
// whatever the number of rows n. Columns are the design's in their order, y the last.
//   - select_qr: the triangle of the rows of each group (one group: every row; the folds of a
//     cross-validation: one per fold), by Householder QR (geqrf) of fixed blocks of rows
//     stacked under the triangle so far; select_merge: the triangle of several groups'
//     triangles stacked in order (every group, or every group but one: a fold's training
//     rows). The blocks depend on n and m only: the same bits for any thread count.
//   - select_eval: a model (a set of columns) and its candidates (columns added or dropped).
//     The model is factored once (Householder over its columns in order; a column with
//     |r_jj| <= rtol ||x_j|| is aliased and left out, the rule of the fit, G21). A candidate
//     that adds columns continues that factorization on their trailing blocks (O(m t^2)); one
//     that drops columns D takes beta_D's quadratic form in (R_UU^T R_UU)^-1 = R_UU^-1 R_UU^-T
//     (O(s t^2 + s^2) given R_UU^-1); a model with an aliased selected column refactors. The
//     value is the candidate's residual sum of squares on R's rows or, given a second
//     triangle T (a fold's test rows), the sum of squares of T [beta; -1] with the candidate's
//     coefficients fitted on R (each row's sum in double-double).
//   - select_subsets: the best column set of each size (the columns added to the forced ones;
//     a set with an aliased column is not one)
//     by Furnival and Wilson's leaps and bounds on one triangle: the triangle of an ordering of
//     the columns gives the residual sum of squares of every prefix at once (the squares of y's
//     column below the prefix), a block of columns moves to the end of a region by adjacent
//     Givens swaps, and a region whose largest set (its lower bound) is no better than the
//     best of every size it can still reach is not searched.
#include <cmath>

#include "regression/dot2.hpp"

#include "bind/errors.hpp"
#include "linalg/ops/common.hpp"
#include "parallel/pool.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace sd::regression {

namespace nb = nanobind;
using linalg::Input;
using linalg::lint;
using linalg::Out;
using linalg::Scratch;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 0x1p-52;

inline double rd(const std::byte* p) {
    double v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

// sum x_i^2 in double-double (Dot2), rounded once
double sumsq(const double* x, lint n) {
    double P = 0.0, S = 0.0;
    for (lint i = 0; i < n; ++i) dot2::row_step(P, S, x[i], x[i]);
    return P + S;
}

// c := (I - tau v v^T) c over n entries (v[0] = 1)
inline void reflect(const double* v, double tau, double* c, lint n) {
    if (tau == 0.0) return;
    double d = c[0];
    for (lint i = 1; i < n; ++i) d += v[i] * c[i];
    d *= tau;
    c[0] -= d;
    for (lint i = 1; i < n; ++i) c[i] -= d * v[i];
}

// A Householder reflector on col[0..n): col[0] becomes beta, v (n entries, v[0] = 1) the
// reflector's vector, col[1..n) zero. Returns tau.
inline double house(double* col, lint n, double* v) {
    double alpha = col[0], tau = 0.0;
    if (n > 1) linalg::lapack::larfg(n, &alpha, col + 1, &tau);
    v[0] = 1.0;
    for (lint i = 1; i < n; ++i) {
        v[i] = col[i];
        col[i] = 0.0;
    }
    col[0] = alpha;
    return tau;
}

std::vector<lint> int_list(nb::handle o, const char* fn, const char* arg) {
    Input a(o, fn, arg);
    if (a.ndim() != 1 || a.dtype() != DType::Int64)
        throw nb::value_error((std::string(fn) + "(): '" + arg + "' must be 1-D int64").c_str());
    std::vector<lint> out(static_cast<std::size_t>(a.shape(0)));
    for (lint i = 0; i < a.shape(0); ++i) std::memcpy(&out[i], a.data() + i * a.stride(0), sizeof(lint));
    return out;
}

// The (..., m, m) triangles of R as column-major copies, the leading axes flattened.
std::vector<double> triangles(const Input& R, lint& count, lint& m, const char* fn) {
    if ((R.ndim() != 2 && R.ndim() != 3) || R.dtype() != DType::Float64 || R.shape(-1) != R.shape(-2))
        throw nb::value_error((std::string(fn) + "(): a triangle must be float64 (m, m) or (c, m, m)").c_str());
    m = R.shape(-1);
    count = R.ndim() == 3 ? R.shape(0) : 1;
    std::vector<double> out(static_cast<std::size_t>(count * m * m));
    for (lint f = 0; f < count; ++f) {
        const std::byte* base = R.data() + (R.ndim() == 3 ? f * R.stride(0) : 0);
        for (lint j = 0; j < m; ++j)
            for (lint i = 0; i < m; ++i)
                out[f * m * m + j * m + i] = i <= j ? rd(base + i * R.stride(-2) + j * R.stride(-1)) : 0.0;
    }
    return out;
}

// ||R[:, j]|| for the design columns (the whitened column's norm over the rows of the fit)
void col_norms(const double* R, lint m, double* nrm) {
    for (lint j = 0; j + 1 < m; ++j) nrm[j] = cblas_dnrm2(j + 1, R + j * m, 1);
}

// ---- one model's factorization

struct Factor {
    lint m = 0, s = 0;
    std::vector<double> W;   // m x m column-major: the reflectors applied to every column
    std::vector<lint> used;  // the design column of each row 0..s-1
    std::vector<char> in;    // per design column: 1 used, 2 selected and aliased, 0 not selected
    bool aliased = false;

    double at(lint i, lint j) const { return W[static_cast<std::size_t>(j * m + i)]; }
};

// Householder over the selected columns in their order (aliased ones skipped), the reflectors
// applied to every other column (the candidates' columns and y).
void factor(const double* R, const double* nrm, lint m, const std::vector<lint>& sel, double tol, Factor& F) {
    F.m = m;
    F.s = 0;
    F.W.assign(R, R + m * m);
    F.used.clear();
    F.in.assign(static_cast<std::size_t>(std::max<lint>(m - 1, 0)), 0);
    F.aliased = false;
    std::vector<double> v(static_cast<std::size_t>(m));
    for (lint j : sel) {
        double* cj = F.W.data() + j * m;
        const lint s = F.s, len = m - s;
        const double nr = cblas_dnrm2(len, cj + s, 1);
        if (!(nr > tol * nrm[j])) {
            F.in[j] = 2;
            F.aliased = true;
            continue;
        }
        const double tau = house(cj + s, len, v.data());
        for (lint c = 0; c < m; ++c) {
            if (c == j || (c < m - 1 && F.in[c] == 1)) continue;
            reflect(v.data(), tau, F.W.data() + c * m + s, len);
        }
        F.in[j] = 1;
        F.used.push_back(j);
        ++F.s;
    }
}

// beta (s) of R_UU beta = rhs (s)
void back_solve(const Factor& F, const double* rhs, double* beta) {
    const lint s = F.s;
    for (lint a = s - 1; a >= 0; --a) {
        double t = rhs[a];
        for (lint b = a + 1; b < s; ++b) t -= F.at(a, F.used[b]) * beta[b];
        beta[a] = t / F.at(a, F.used[a]);
    }
}

// R_UU^-1 (s x s column-major, upper)
void tri_inverse(const Factor& F, std::vector<double>& Ri) {
    const lint s = F.s;
    Ri.assign(static_cast<std::size_t>(s * s), 0.0);
    for (lint b = 0; b < s; ++b) {
        double* col = Ri.data() + b * s;
        col[b] = 1.0 / F.at(b, F.used[b]);
        for (lint a = b - 1; a >= 0; --a) {
            double t = 0.0;
            for (lint c = a + 1; c <= b; ++c) t += F.at(a, F.used[c]) * col[c];
            col[a] = -t / F.at(a, F.used[a]);
        }
    }
}

// Dropping the model's columns at positions pos (ascending, of `used`): returns the residual
// sum of squares' increase u^T u, u from T^T u = beta_D with V^T = Q T, V = R^-1[pos, :]; when
// nb is given, the coefficients after the drop (beta - R^-1 Q u, exact zeros at pos).
double drop_update(
    const Factor& F, const std::vector<double>& Ri, const double* beta, const std::vector<lint>& pos, double* nb
) {
    const lint s = F.s, t = static_cast<lint>(pos.size());
    std::vector<double> Vt(static_cast<std::size_t>(s * t)), taus(static_cast<std::size_t>(t)),
        refl(static_cast<std::size_t>(s * t)), u(static_cast<std::size_t>(t));
    for (lint q = 0; q < t; ++q)
        for (lint i = 0; i < s; ++i) Vt[q * s + i] = i >= pos[q] ? Ri[i * s + pos[q]] : 0.0;
    for (lint q = 0; q < t; ++q) {
        double* col = Vt.data() + q * s;
        double* v = refl.data() + q * s;
        taus[q] = house(col + q, s - q, v + q);
        for (lint c = q + 1; c < t; ++c) reflect(v + q, taus[q], Vt.data() + c * s + q, s - q);
    }
    for (lint q = 0; q < t; ++q) {
        double r = beta[pos[q]];
        for (lint a = 0; a < q; ++a) r -= Vt[q * s + a] * u[a];
        u[q] = r / Vt[q * s + q];
    }
    const double inc = sumsq(u.data(), t);
    if (nb != nullptr) {
        std::vector<double> w(static_cast<std::size_t>(s), 0.0);
        for (lint q = 0; q < t; ++q) w[q] = u[q];
        for (lint q = t - 1; q >= 0; --q) reflect(refl.data() + q * s + q, taus[q], w.data() + q, s - q);
        for (lint a = 0; a < s; ++a) {
            double d = 0.0;
            for (lint b = a; b < s; ++b) d += Ri[b * s + a] * w[b];
            nb[a] = beta[a] - d;
        }
        for (lint q = 0; q < t; ++q) nb[pos[q]] = 0.0;
    }
    return inc;
}

// Adding the columns C (ascending, outside the model): the trailing blocks of C and y factored
// on. Returns the residual sum of squares; rank_add the columns kept; with beta_out (given the
// model's s coefficients' storage of s + |C|), the coefficients of the model's columns then
// the kept added columns (cols_out: their design columns).
double add_update(
    const Factor& F,
    const double* nrm,
    const std::vector<lint>& C,
    double tol,
    lint& rank_add,
    const double* zmodel,
    std::vector<double>* beta_out,
    std::vector<lint>* cols_out
) {
    const lint m = F.m, s = F.s, len = m - s, t = static_cast<lint>(C.size()), y = m - 1;
    std::vector<double> B(static_cast<std::size_t>(len * (t + 1))), v(static_cast<std::size_t>(std::max<lint>(len, 1)));
    for (lint q = 0; q < t; ++q) std::copy(F.W.data() + C[q] * m + s, F.W.data() + C[q] * m + m, B.data() + q * len);
    std::copy(F.W.data() + y * m + s, F.W.data() + y * m + m, B.data() + t * len);
    std::vector<lint> acc;
    lint r = 0;
    for (lint q = 0; q < t; ++q) {
        double* col = B.data() + q * len;
        const double nr = cblas_dnrm2(len - r, col + r, 1);
        if (!(nr > tol * nrm[C[q]])) continue;
        const double tau = house(col + r, len - r, v.data());
        for (lint c = q + 1; c <= t; ++c) reflect(v.data(), tau, B.data() + c * len + r, len - r);
        acc.push_back(q);
        ++r;
    }
    rank_add = r;
    const double ssr = sumsq(B.data() + t * len + r, len - r);
    if (beta_out != nullptr) {
        std::vector<double> bA(static_cast<std::size_t>(r));
        for (lint a = r - 1; a >= 0; --a) {
            double z = B[t * len + a];
            for (lint b = a + 1; b < r; ++b) z -= B[acc[b] * len + a] * bA[b];
            bA[a] = z / B[acc[a] * len + a];
        }
        std::vector<double> rhs(zmodel, zmodel + s);
        for (lint a = 0; a < s; ++a)
            for (lint b = 0; b < r; ++b) rhs[a] -= F.at(a, C[acc[b]]) * bA[b];
        beta_out->assign(static_cast<std::size_t>(s + r), 0.0);
        back_solve(F, rhs.data(), beta_out->data());
        cols_out->assign(F.used.begin(), F.used.end());
        for (lint b = 0; b < r; ++b) {
            (*beta_out)[s + b] = bA[b];
            cols_out->push_back(C[acc[b]]);
        }
    }
    return ssr;
}

// sum over T's rows of (T [beta; -1])^2, each row's sum in double-double
double test_sse(const double* T, lint m, const std::vector<lint>& cols, const double* beta) {
    std::vector<double> r(static_cast<std::size_t>(m));
    const lint y = m - 1, nc = static_cast<lint>(cols.size());
    for (lint i = 0; i < m; ++i) {
        double P = 0.0, S = 0.0;
        dot2::row_step(P, S, T[y * m + i], -1.0);
        for (lint q = 0; q < nc; ++q)
            if (cols[q] >= i) dot2::row_step(P, S, T[cols[q] * m + i], beta[q]);
        r[i] = P + S;
    }
    return sumsq(r.data(), m);
}

inline double tol_for(double rtol, double nobs, lint k) {
    return rtol >= 0 ? rtol : std::max(nobs, static_cast<double>(k)) * kEps;
}

// ---- select_qr, select_merge

// [R; block] -> R: geqrf of the (r + b) x m stack, R its upper triangle (rows beyond the
// stack's height zero)
void stack_qr(std::vector<double>& R, lint& rrows, const double* block, lint brows, lint m, Scratch& sc) {
    const lint h = rrows + brows;
    std::vector<double> S(static_cast<std::size_t>(h * m));
    for (lint j = 0; j < m; ++j) {
        for (lint i = 0; i < rrows; ++i) S[j * h + i] = R[j * m + i];
        for (lint i = 0; i < brows; ++i) S[j * h + rrows + i] = block[j * brows + i];
    }
    std::vector<double> tau(static_cast<std::size_t>(std::max<lint>(std::min(h, m), 1)));
    linalg::lapack::geqrf(h, m, S.data(), h, tau.data(), sc);
    rrows = std::min(h, m);
    std::fill(R.begin(), R.end(), 0.0);
    for (lint j = 0; j < m; ++j)
        for (lint i = 0; i <= std::min(j, rrows - 1); ++i) R[j * m + i] = S[j * h + i];
}

// select_qr(y (n,), X (n, p), w (n,) | None, u (n,) | None, intercept, labels (n,) int64 | None,
// groups) -> (groups, m, m): the triangle of each group's whitened rows [u or 1, X, y] sqrt(w)
// (rows of weight 0 and of label < 0 left out)
nb::object select_qr(
    nb::handle yo, nb::handle xo, nb::handle wo, nb::handle uo, bool intercept, nb::handle lo, std::int64_t groups
) {
    const char* fn = "select_qr";
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), X(xo, fn, "X");
        std::optional<Input> w, u, lab;
        if (!wo.is_none()) w.emplace(wo, fn, "w");
        if (!uo.is_none()) u.emplace(uo, fn, "u");
        if (!lo.is_none()) lab.emplace(lo, fn, "labels");
        const lint n = y.shape(0), p = X.shape(1), j0 = intercept ? 1 : 0, m = p + j0 + 1;
        const lint G = std::max<std::int64_t>(groups, 1);
        Out oo = linalg::alloc_c({G, m, m}, DType::Float64);
        const lint kRows = std::max<lint>(1024, 4 * m);
        auto label = [&](lint t) -> std::int64_t {
            if (!lab) return 0;
            std::int64_t v;
            std::memcpy(&v, lab->data() + t * lab->stride(0), sizeof v);
            return v;
        };
        auto one = [&](lint g) {
            Scratch sc;
            std::vector<double> R(static_cast<std::size_t>(m * m), 0.0), blk(static_cast<std::size_t>(kRows * m));
            lint rrows = 0, b = 0;
            auto flush = [&]() {
                if (b == 0) return;
                if (b < kRows) {  // compact the partial block's columns
                    std::vector<double> c(static_cast<std::size_t>(b * m));
                    for (lint j = 0; j < m; ++j)
                        std::copy(blk.data() + j * kRows, blk.data() + j * kRows + b, c.data() + j * b);
                    stack_qr(R, rrows, c.data(), b, m, sc);
                } else {
                    stack_qr(R, rrows, blk.data(), b, m, sc);
                }
                b = 0;
            };
            for (lint t = 0; t < n; ++t) {
                if (label(t) != g) continue;
                double sw = 1.0;
                if (w) {
                    const double wt = rd(w->data() + t * w->stride(0));
                    if (!(wt > 0.0)) continue;
                    sw = std::sqrt(wt);
                }
                if (intercept) blk[b] = sw * (u ? rd(u->data() + t * u->stride(0)) : 1.0);
                for (lint j = 0; j < p; ++j)
                    blk[(j0 + j) * kRows + b] = sw * rd(X.data() + t * X.stride(0) + j * X.stride(1));
                blk[(m - 1) * kRows + b] = sw * rd(y.data() + t * y.stride(0));
                if (++b == kRows) flush();
            }
            flush();
            double* out = oo.as<double>() + g * m * m;
            for (lint i = 0; i < m; ++i)
                for (lint j = 0; j < m; ++j) out[i * m + j] = R[j * m + i];
        };
        if (G >= 2 && parallel::available()) {
            parallel::parallel_for(G, [&](std::int64_t g) { one(static_cast<lint>(g)); });
        } else {
            parallel::GilRelease nogil;
            for (lint g = 0; g < G; ++g) one(g);
        }
        return linalg::to_array(oo);
    });
}

// select_merge(T (g, m, m), leave_out) -> leave_out: (g, m, m), group f's the triangle of every
// other group's stacked in order (a fold's training rows); else (m, m) of all of them
nb::object select_merge(nb::handle to, bool leave_out) {
    const char* fn = "select_merge";
    return call_named(fn, [&]() -> nb::object {
        Input T(to, fn, "T");
        lint G, m;
        const std::vector<double> Ts = triangles(T, G, m, fn);
        const lint outs = leave_out ? G : 1;
        Out oo = leave_out ? linalg::alloc_c({G, m, m}, DType::Float64) : linalg::alloc_c({m, m}, DType::Float64);
        auto one = [&](lint f) {
            Scratch sc;
            std::vector<double> R(static_cast<std::size_t>(m * m), 0.0);
            lint rrows = 0;
            for (lint g = 0; g < G; ++g) {
                if (leave_out && g == f) continue;
                // the group's triangle as an m-row block (column-major already)
                stack_qr(R, rrows, Ts.data() + g * m * m, m, m, sc);
            }
            double* out = oo.as<double>() + f * m * m;
            for (lint i = 0; i < m; ++i)
                for (lint j = 0; j < m; ++j) out[i * m + j] = R[j * m + i];
        };
        if (outs >= 2 && parallel::available()) {
            parallel::parallel_for(outs, [&](std::int64_t f) { one(static_cast<lint>(f)); });
        } else {
            parallel::GilRelease nogil;
            for (lint f = 0; f < outs; ++f) one(f);
        }
        return linalg::to_array(oo);
    });
}

// ---- select_eval

// select_eval(R (c, m, m) | (m, m), T like R | None, nobs (c,), rtol (< 0: max(nobs, k) eps per
// model), sel (int64, ascending), cand_cols (int64), cand_ptr (int64, ncand + 1), cand_drop
// (int64, ncand)) -> (cur (c,), cur_rank (c,), cur_in (c, m - 1) int64 (1 used, 2 aliased),
// vals (c, ncand), ranks (c, ncand))
nb::object select_eval(
    nb::handle ro,
    nb::handle to,
    nb::handle nobs_o,
    double rtol,
    nb::handle selo,
    nb::handle cco,
    nb::handle cpo,
    nb::handle cdo
) {
    const char* fn = "select_eval";
    return call_named(fn, [&]() -> nb::object {
        Input Ri(ro, fn, "R");
        lint c, m;
        const std::vector<double> Rs = triangles(Ri, c, m, fn);
        std::vector<double> Ts;
        const bool test = !to.is_none();
        if (test) {
            Input Ti(to, fn, "T");
            lint c2, m2;
            Ts = triangles(Ti, c2, m2, fn);
            if (c2 != c || m2 != m) throw nb::value_error("select_eval(): T must be shaped like R");
        }
        Input no(nobs_o, fn, "nobs");
        std::vector<double> nobs(static_cast<std::size_t>(c));
        for (lint f = 0; f < c; ++f) nobs[f] = rd(no.data() + f * no.stride(0));
        const std::vector<lint> sel = int_list(selo, fn, "sel"), cc = int_list(cco, fn, "cand_cols"),
                                cp = int_list(cpo, fn, "cand_ptr"), cd = int_list(cdo, fn, "cand_drop");
        const lint ncand = static_cast<lint>(cp.size()) - 1, ns = static_cast<lint>(sel.size()), d = m - 1;
        Out curo = linalg::alloc_c({c}, DType::Float64), crko = linalg::alloc_c({c}, DType::Float64),
            cino = linalg::alloc_c({c, d}, DType::Int64), valo = linalg::alloc_c({c, ncand}, DType::Float64),
            rko = linalg::alloc_c({c, ncand}, DType::Float64);
        std::vector<Factor> F(static_cast<std::size_t>(c));
        std::vector<std::vector<double>> nrm(static_cast<std::size_t>(c)), beta(static_cast<std::size_t>(c)),
            Rinv(static_cast<std::size_t>(c));
        bool any_drop = false;
        for (lint q = 0; q < ncand; ++q) any_drop |= cd[q] != 0;
        auto prep = [&](lint f) {
            const double* R = Rs.data() + f * m * m;
            nrm[f].resize(static_cast<std::size_t>(std::max<lint>(d, 1)));
            col_norms(R, m, nrm[f].data());
            factor(R, nrm[f].data(), m, sel, tol_for(rtol, nobs[f], ns), F[f]);
            const Factor& Ff = F[f];
            std::vector<double> z(static_cast<std::size_t>(Ff.s));
            for (lint a = 0; a < Ff.s; ++a) z[a] = Ff.at(a, d);
            beta[f].resize(static_cast<std::size_t>(Ff.s));
            back_solve(Ff, z.data(), beta[f].data());
            if (any_drop && !Ff.aliased) tri_inverse(Ff, Rinv[f]);
            double cur;
            if (test) {
                cur = test_sse(Ts.data() + f * m * m, m, Ff.used, beta[f].data());
            } else {
                cur = sumsq(Ff.W.data() + d * m + Ff.s, m - Ff.s);
            }
            curo.as<double>()[f] = cur;
            crko.as<double>()[f] = static_cast<double>(Ff.s);
            for (lint j = 0; j < d; ++j) cino.as<std::int64_t>()[f * d + j] = Ff.in[j];
        };
        auto cand = [&](lint task) {
            const lint f = task / ncand, q = task % ncand;
            const Factor& Ff = F[f];
            std::vector<lint> C(cc.begin() + cp[q], cc.begin() + cp[q + 1]);
            double val = kNaN, rank = kNaN;
            const double* T = test ? Ts.data() + f * m * m : nullptr;
            if (!cd[q]) {
                lint ra = 0;
                const double tol = tol_for(rtol, nobs[f], ns + static_cast<lint>(C.size()));
                if (test) {
                    std::vector<double> z(static_cast<std::size_t>(Ff.s)), b;
                    for (lint a = 0; a < Ff.s; ++a) z[a] = Ff.at(a, d);
                    std::vector<lint> cols;
                    add_update(Ff, nrm[f].data(), C, tol, ra, z.data(), &b, &cols);
                    val = test_sse(T, m, cols, b.data());
                } else {
                    val = add_update(Ff, nrm[f].data(), C, tol, ra, nullptr, nullptr, nullptr);
                }
                rank = static_cast<double>(Ff.s + ra);
            } else {
                bool formula = !Ff.aliased;
                std::vector<lint> pos;
                for (lint j : C) {
                    const auto it = std::find(Ff.used.begin(), Ff.used.end(), j);
                    if (it == Ff.used.end()) {
                        formula = false;
                        break;
                    }
                    pos.push_back(static_cast<lint>(it - Ff.used.begin()));
                }
                if (formula) {
                    if (test) {
                        std::vector<double> nb(static_cast<std::size_t>(Ff.s));
                        drop_update(Ff, Rinv[f], beta[f].data(), pos, nb.data());
                        val = test_sse(T, m, Ff.used, nb.data());
                    } else {
                        const double cur = curo.as<double>()[f];
                        val = cur + drop_update(Ff, Rinv[f], beta[f].data(), pos, nullptr);
                    }
                    rank = static_cast<double>(Ff.s - static_cast<lint>(pos.size()));
                } else {  // refactor the smaller model
                    std::vector<lint> rest;
                    for (lint j : sel)
                        if (std::find(C.begin(), C.end(), j) == C.end()) rest.push_back(j);
                    Factor G;
                    const double* R = Rs.data() + f * m * m;
                    factor(R, nrm[f].data(), m, rest, tol_for(rtol, nobs[f], static_cast<lint>(rest.size())), G);
                    if (test) {
                        std::vector<double> z(static_cast<std::size_t>(G.s)), b(static_cast<std::size_t>(G.s));
                        for (lint a = 0; a < G.s; ++a) z[a] = G.at(a, d);
                        back_solve(G, z.data(), b.data());
                        val = test_sse(T, m, G.used, b.data());
                    } else {
                        val = sumsq(G.W.data() + d * m + G.s, m - G.s);
                    }
                    rank = static_cast<double>(G.s);
                }
            }
            valo.as<double>()[f * ncand + q] = val;
            rko.as<double>()[f * ncand + q] = rank;
        };
        {
            parallel::GilRelease nogil;
            if (c >= 2 && parallel::available()) {
                parallel::parallel_for(c, [&](std::int64_t f) { prep(static_cast<lint>(f)); });
            } else {
                for (lint f = 0; f < c; ++f) prep(f);
            }
            const lint tasks = c * ncand;
            const double work = double(tasks) * double(m) * double(m);
            if (tasks >= 2 && work >= 1e6 && parallel::available()) {
                parallel::parallel_for(tasks, [&](std::int64_t t) { cand(static_cast<lint>(t)); });
            } else {
                for (lint t = 0; t < tasks; ++t) cand(t);
            }
        }
        return nb::make_tuple(
            linalg::to_array(curo), linalg::to_array(crko), linalg::to_array(cino), linalg::to_array(valo),
            linalg::to_array(rko)
        );
    });
}

// ---- select_subsets: leaps and bounds

struct Leaps {
    lint m, d, nbase, nblocks, max_size;
    double tol;
    std::vector<double> W;        // m x m column-major, the triangle of the current order
    std::vector<lint> colat;      // design column at each position 0..d-1
    std::vector<lint> blk;        // block at each block position
    std::vector<lint> bsize;      // per block: its column count
    std::vector<double> nrm;      // per design column
    std::vector<double> suffix;   // suffix[i] = sum_{r >= i} W(r, y)^2, i in 0..m
    std::vector<char> alive;      // per position: not aliased
    std::vector<lint> cumrank;    // cumrank[j]: the positions [0, j) not aliased
    std::vector<lint> bstart;     // per block position (and nblocks): its first column position
    lint base_rank = 0;
    std::vector<double> best;     // per size
    std::vector<std::vector<char>> best_set;
    std::int64_t moves = 0;

    double& at(lint i, lint j) { return W[static_cast<std::size_t>(j * m + i)]; }

    lint colstart(lint q) const { return bstart[q]; }  // q = nblocks: d

    void refresh() {
        suffix.assign(static_cast<std::size_t>(m + 1), 0.0);
        const lint y = m - 1;
        double P = 0.0, S = 0.0;
        for (lint i = m - 1; i >= 0; --i) {
            dot2::row_step(P, S, at(i, y), at(i, y));
            suffix[i] = P + S;
        }
        alive.assign(static_cast<std::size_t>(d), 0);
        cumrank.assign(static_cast<std::size_t>(d + 1), 0);
        for (lint j = 0; j < d; ++j) {
            alive[j] = std::fabs(at(j, j)) > tol * nrm[colat[j]];
            cumrank[j + 1] = cumrank[j] + alive[j];
        }
        bstart.assign(static_cast<std::size_t>(nblocks + 1), nbase);
        for (lint b = 0; b < nblocks; ++b) bstart[b + 1] = bstart[b] + bsize[blk[b]];
    }

    // swap the columns at positions j and j + 1, then a Givens rotation on rows j, j + 1
    void swap(lint j) {
        for (lint i = 0; i <= j + 1; ++i) std::swap(at(i, j), at(i, j + 1));
        std::swap(colat[j], colat[j + 1]);
        const double a = at(j, j), b = at(j + 1, j);
        if (b == 0.0) return;
        const double r = std::hypot(a, b), cs = a / r, sn = b / r;
        for (lint c = j; c < m; ++c) {
            const double u = at(j, c), v = at(j + 1, c);
            at(j, c) = cs * u + sn * v;
            at(j + 1, c) = -sn * u + cs * v;
        }
        at(j + 1, j) = 0.0;
    }

    // the block at block position g to block position lim (the blocks between move up one)
    void move_block(lint g, lint lim) {
        const lint b = blk[g], t = bsize[b], a = colstart(g), e = colstart(lim + 1);
        for (lint k = 0; k < t; ++k)
            for (lint j = a; j < e - 1; ++j) swap(j);
        for (lint q = g; q < lim; ++q) blk[q] = blk[q + 1];
        blk[lim] = b;
        ++moves;
        refresh();
    }

    lint rank_upto(lint col) const { return cumrank[col]; }  // non-aliased among positions [0, col)

    // the prefixes of blocks 0..j for j in [from, to]
    // (from = -1: the forced columns alone first)
    void report(lint from, lint to) {
        // a block with an aliased column disqualifies the prefix: the triangle's row of an
        // aliased column is a direction outside the set's span, so its residual sum of squares
        // would be too small (with single columns the same set without it has the same fit)
        bool dead = false;
        lint col = nbase;
        for (lint q = 0; q < from; ++q) {
            const lint e = col + bsize[blk[q]];
            lint add = 0;
            for (lint j = col; j < e; ++j) add += alive[j];
            dead |= add < e - col;
            col = e;
        }
        for (lint q = from; q <= to; ++q) {
            if (q >= 0) {
                const lint e = col + bsize[blk[q]];
                lint add = 0;
                for (lint j = col; j < e; ++j) add += alive[j];
                dead |= add < e - col;
                col = e;
            }
            if (dead) continue;
            const lint size = rank_upto(col) - base_rank;
            if (size < 0 || size > max_size) continue;
            const double rss = suffix[col];
            if (rss < best[size]) {
                best[size] = rss;
                auto& set = best_set[size];
                std::fill(set.begin(), set.end(), 0);
                for (lint b = 0; b <= q; ++b) set[blk[b]] = 1;
            }
        }
    }

    // The largest size in [lo, hi] (and at most cap) that a set with the lower bound lb could
    // still improve, or -1: the sizes above it have a best at most lb already.
    lint reach(double lb, lint lo, lint hi, lint cap) const {
        for (lint k = std::min({hi, max_size, cap}); k >= std::max<lint>(lo, 0); --k)
            if (lb < best[k]) return k;
        return -1;
    }

    // Blocks at positions < i fixed in, the region [i, lim], its prefixes reported already; only
    // the sets of at most `cap` columns are wanted (larger ones cannot improve a best).
    void explore(lint i, lint lim, lint cap) {
        for (lint g = lim - 1; g >= i; --g) {
            // the sets searched have the fixed blocks and at least one more, each adding rank (a
            // set with a block that adds none is not reported): sizes from one past the fixed part's
            const lint lo = rank_upto(colstart(g)) - base_rank + 1;
            if (lo > cap || lo > max_size) continue;  // nothing wanted there: no move either
            move_block(g, lim);
            const lint e = colstart(lim);
            const lint top = reach(suffix[e], lo, rank_upto(e) - base_rank, cap);
            if (top < 0) continue;
            report(g, lim - 1);
            explore(g, lim - 1, top);
        }
    }
};

// select_subsets(R (m, m), nobs, rtol, base (int64), block_cols (int64), block_ptr (int64,
// blocks + 1), order (int64: the blocks' starting order), max_size) -> (rss (max_size + 1,),
// sets (max_size + 1, blocks) bool, moves)
nb::object select_subsets(
    nb::handle ro,
    double nobs,
    double rtol,
    nb::handle bo,
    nb::handle bco,
    nb::handle bpo,
    nb::handle oo_,
    std::int64_t max_size
) {
    const char* fn = "select_subsets";
    return call_named(fn, [&]() -> nb::object {
        Input Ri(ro, fn, "R");
        lint one, m;
        const std::vector<double> R0 = triangles(Ri, one, m, fn);
        const std::vector<lint> base = int_list(bo, fn, "base"), bc = int_list(bco, fn, "block_cols"),
                                bp = int_list(bpo, fn, "block_ptr"), order = int_list(oo_, fn, "order");
        const lint nblocks = static_cast<lint>(bp.size()) - 1, S = max_size;
        Out rsso = linalg::alloc_c({S + 1}, DType::Float64), seto = linalg::alloc_c({S + 1, nblocks}, DType::Bool);
        std::int64_t moves = 0;
        {
            parallel::GilRelease nogil;
            Leaps L;
            L.m = m;
            L.d = m - 1;
            L.nbase = static_cast<lint>(base.size());
            L.nblocks = nblocks;
            L.max_size = S;
            L.tol = tol_for(rtol, nobs, m - 1);
            L.nrm.resize(static_cast<std::size_t>(std::max<lint>(L.d, 1)));
            col_norms(R0.data(), m, L.nrm.data());
            L.bsize.resize(static_cast<std::size_t>(nblocks));
            for (lint b = 0; b < nblocks; ++b) L.bsize[b] = bp[b + 1] - bp[b];
            L.blk = order;
            // the starting order: base, then the blocks in `order`, then y
            for (lint j : base) L.colat.push_back(j);
            for (lint b : order)
                for (lint k = bp[b]; k < bp[b + 1]; ++k) L.colat.push_back(bc[k]);
            std::vector<double> A(static_cast<std::size_t>(m * m));
            for (lint p = 0; p < L.d; ++p)
                std::copy(R0.data() + L.colat[p] * m, R0.data() + L.colat[p] * m + m, A.data() + p * m);
            std::copy(R0.data() + (m - 1) * m, R0.data() + m * m, A.data() + (m - 1) * m);
            Scratch sc;
            std::vector<double> tau(static_cast<std::size_t>(m));
            linalg::lapack::geqrf(m, m, A.data(), m, tau.data(), sc);
            for (lint j = 0; j < m; ++j)
                for (lint i = j + 1; i < m; ++i) A[j * m + i] = 0.0;
            L.W = std::move(A);
            L.refresh();
            L.base_rank = L.rank_upto(L.nbase);
            L.best.assign(static_cast<std::size_t>(S + 1), kInf);
            L.best_set.assign(static_cast<std::size_t>(S + 1), std::vector<char>(static_cast<std::size_t>(nblocks), 0));
            L.report(-1, nblocks - 1);
            if (nblocks >= 2) L.explore(0, nblocks - 1, S);
            for (lint k = 0; k <= S; ++k) {
                rsso.as<double>()[k] = std::isinf(L.best[k]) ? kNaN : L.best[k];
                for (lint b = 0; b < nblocks; ++b) seto.as<bool>()[k * nblocks + b] = L.best_set[k][b] != 0;
            }
            moves = L.moves;
        }
        return nb::make_tuple(linalg::to_array(rsso), linalg::to_array(seto), moves);
    });
}

}  // namespace

void bind_select(nb::module_& m) {
    m.def(
        "select_qr", &select_qr, nb::arg("y").none(), nb::arg("X").none(), nb::arg("w").none(), nb::arg("u").none(),
        nb::arg("intercept"), nb::arg("labels").none(), nb::arg("groups")
    );
    m.def("select_merge", &select_merge, nb::arg("T").none(), nb::arg("leave_out"));
    m.def(
        "select_eval", &select_eval, nb::arg("R").none(), nb::arg("T").none(), nb::arg("nobs").none(), nb::arg("rtol"),
        nb::arg("sel").none(), nb::arg("cand_cols").none(), nb::arg("cand_ptr").none(), nb::arg("cand_drop").none()
    );
    m.def(
        "select_subsets", &select_subsets, nb::arg("R").none(), nb::arg("nobs"), nb::arg("rtol"),
        nb::arg("base").none(), nb::arg("block_cols").none(), nb::arg("block_ptr").none(), nb::arg("order").none(),
        nb::arg("max_size")
    );
}

}  // namespace sd::regression
