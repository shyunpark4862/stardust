// The multivariate normal and t cdfs (T9, docs/design/stats.md §6.2; decisions T16,
// T29): mv_cdf in stardust._core.stats, the kernel of stardust.stats'
// MultivariateNormal.cdf and MultivariateT.cdf. The algorithms are in
// stats/mvcdf.hpp.
//
//     mv_cdf(cov, x, batch, df, atol, max_points) -> (values, errors)
//
// cov (B, d, d) float64: covariance (or the t's shape) matrices; only the lower
// triangle is read, as LAPACK's potrf 'L' does. x (M, d) float64: the upper
// limits minus the mean. batch (M,) int64: the matrix of each point. df (B,)
// float64: +inf for the normal. atol > 0: the QMC target (3 standard errors).
// max_points >= 0: the QMC's cap on integrand evaluations per point (0: 10^6 n,
// scipy's default, n the variables integrated).
//
// Per point, in this order: NaN where the matrix is not positive definite (a
// Cholesky pivot <= 0, or a non-finite entry) or df is not > 0; NaN where x has a
// NaN; 0 where x has a -inf; the +inf coordinates dropped (the marginal of the
// rest, the same df for the t); 1 when none is left. The error is 0 for the exact
// formulas (up to 3 finite coordinates, normal or t) and NaN with a NaN value;
// for the QMC (4 or more) it is 3 standard errors over the shifts.
//
// The points run on the pool. The QMC points run one per task when there are
// enough of them, else one after another with each one's lattice on the pool:
// a point's value never depends on which, nor on the thread count. The t's QMC
// shares one chi table per distinct df of the call (built from df alone).
#include "bind/stats_bind.hpp"

#include "array/ops/pieces.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "special/ops/distr.hpp"
#include "stats/mvcdf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd {

namespace {

using namespace stats::mvcdf;

const NDArray& arg(nb::handle o, const char* what, DType t, int ndim) {
    const NDArray& a = require_ndarray(o, what);
    if (a.dtype != t || a.ndim != ndim)
        throw nb::type_error((std::string("mv_cdf: '") + what + "' has the wrong dtype or ndim").c_str());
    return a;
}

// The elements of a in C order, any strides.
template <class T>
std::vector<T> flat(const NDArray& a) {
    std::vector<T> v(static_cast<std::size_t>(a.size()));
    std::int64_t idx[3] = {0, 0, 0};
    for (std::size_t e = 0; e < v.size(); ++e) {
        std::int64_t off = 0;
        for (int k = 0; k < a.ndim; ++k) off += idx[k] * a.strides[k];
        v[e] = *reinterpret_cast<const T*>(a.data + off);
        for (int k = a.ndim - 1; k >= 0; --k) {
            if (++idx[k] < a.shape[k]) break;
            idx[k] = 0;
        }
    }
    return v;
}

// Cholesky of the lower triangle, only to test positive definiteness (potrf's rule).
bool positive_definite(const double* S, int d) {
    std::vector<double> L(static_cast<std::size_t>(d) * d, 0.0);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j <= i; ++j)
            if (!std::isfinite(S[i * d + j])) return false;
    for (int j = 0; j < d; ++j) {
        double v = S[j * d + j];
        for (int k = 0; k < j; ++k) v -= L[j * d + k] * L[j * d + k];
        if (!(v > 0.0)) return false;
        const double l = std::sqrt(v);
        L[j * d + j] = l;
        for (int i = j + 1; i < d; ++i) {
            double s = S[i * d + j];
            for (int k = 0; k < j; ++k) s -= L[i * d + k] * L[j * d + k];
            L[i * d + j] = s / l;
        }
    }
    return true;
}

nb::tuple mv_cdf(nb::handle cov_o, nb::handle x_o, nb::handle batch_o, nb::handle df_o, double atol,
                 std::int64_t max_points) {
    const NDArray& cov_a = arg(cov_o, "cov", DType::Float64, 3);
    const NDArray& x_a = arg(x_o, "x", DType::Float64, 2);
    const NDArray& batch_a = arg(batch_o, "batch", DType::Int64, 1);
    const NDArray& df_a = arg(df_o, "df", DType::Float64, 1);
    const std::int64_t B = cov_a.shape[0], d64 = cov_a.shape[1], M = x_a.shape[0];
    if (cov_a.shape[2] != d64 || x_a.shape[1] != d64 || batch_a.shape[0] != M || df_a.shape[0] != B)
        throw nb::value_error("mv_cdf: expects cov (B, d, d), x (M, d), batch (M,), df (B,)");
    if (!(atol > 0.0)) throw nb::value_error("mv_cdf: atol must be > 0");
    if (max_points < 0) throw nb::value_error("mv_cdf: max_points must be >= 0");
    const int d = static_cast<int>(d64);
    const std::vector<double> cov = flat<double>(cov_a), x = flat<double>(x_a), df = flat<double>(df_a);
    const std::vector<std::int64_t> batch = flat<std::int64_t>(batch_a);
    for (std::int64_t p = 0; p < M; ++p)
        if (batch[p] < 0 || batch[p] >= B) throw nb::index_error("mv_cdf: batch index out of range");

    const std::int64_t shape[] = {M};
    NDArray val_a = new_array(DType::Float64, shape), err_a = new_array(DType::Float64, shape);
    double* val = reinterpret_cast<double*>(val_a.data);
    double* err = reinterpret_cast<double*>(err_a.data);
    {
        nb::gil_scoped_release release;
        const std::size_t dd = static_cast<std::size_t>(d) * d;
        // per matrix: valid (positive definite, df > 0) and the standard deviations
        std::vector<char> ok(static_cast<std::size_t>(B));
        std::vector<double> sdev(static_cast<std::size_t>(B) * d);
        for_pieces(B, 64, B * d * d * d, [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
            for (std::int64_t b = b0; b < b1; ++b) {
                const double* S = cov.data() + b * dd;
                ok[b] = df[b] > 0.0 && positive_definite(S, d);
                for (int i = 0; i < d; ++i) sdev[b * d + i] = std::sqrt(S[i * d + i]);
            }
        });
        // the exact value of point p with n <= 3 finite coordinates (normal or t)
        auto exact = [&](std::int64_t p) {
            const std::int64_t b = batch[p];
            const double* xp = x.data() + p * d;
            int idx[3], n = 0;
            for (int i = 0; i < d; ++i)
                if (xp[i] < kInf) idx[n++] = i;
            const bool normal = df[b] == kInf;
            const double* S = cov.data() + b * dd;
            const double* sd = sdev.data() + b * d;
            auto h = [&](int q) { return xp[idx[q]] / sd[idx[q]]; };
            auto r = [&](int q, int s) {
                const int a = std::max(idx[q], idx[s]), c = std::min(idx[q], idx[s]);
                return std::clamp(S[a * d + c] / (sd[a] * sd[c]), -1.0, 1.0);
            };
            if (n == 1) return normal ? Phi(h(0)) : special::Stdtr::f(df[b], h(0));
            if (n == 2) return normal ? bvn(h(0), h(1), r(0, 1)) : bvt(h(0), h(1), r(0, 1), df[b]);
            Tri T;
            for (int q = 0; q < 3; ++q) {
                T.h[q] = h(q);
                T.r[q][q] = 1.0;
                for (int s = 0; s < q; ++s) T.r[q][s] = T.r[s][q] = r(q, s);
            }
            return normal ? tvn(T) : tvt(T, df[b]);
        };
        // pass 1: the special values and the exact normals (~1 us each); the exact t's
        // (kind 2, ~0.2-30 ms each) and the QMC points (kind 1) are listed for later
        std::vector<char> kind(static_cast<std::size_t>(M), 0);
        for_pieces(M, 16, M * 256 * std::max(d, 1), [&](std::int64_t, std::int64_t p0, std::int64_t p1) {
            for (std::int64_t p = p0; p < p1; ++p) {
                const std::int64_t b = batch[p];
                const double* xp = x.data() + p * d;
                val[p] = kNaN, err[p] = kNaN;
                if (!ok[b]) continue;
                bool nan = false, neg = false;
                int n = 0;
                for (int i = 0; i < d; ++i) {
                    nan = nan || std::isnan(xp[i]);
                    neg = neg || xp[i] == -kInf;
                    n += xp[i] < kInf;
                }
                if (nan) continue;
                err[p] = 0.0;
                if (neg || n == 0) {
                    val[p] = neg ? 0.0 : 1.0;
                } else if (n > 3) {
                    kind[p] = 1;
                } else if (df[b] != kInf && n > 1) {
                    kind[p] = 2;
                } else {
                    val[p] = exact(p);
                }
            }
        });
        // pass 1b: the exact t's, one per task
        std::vector<std::int64_t> ts;
        for (std::int64_t p = 0; p < M; ++p)
            if (kind[p] == 2) ts.push_back(p);
        for_pieces(static_cast<std::int64_t>(ts.size()), 1, static_cast<std::int64_t>(ts.size()) << 20,
                   [&](std::int64_t, std::int64_t q0, std::int64_t q1) {
                       for (std::int64_t q = q0; q < q1; ++q) val[ts[q]] = exact(ts[q]);
                   });
        // pass 2: the QMC points
        std::vector<std::int64_t> qs;
        for (std::int64_t p = 0; p < M; ++p)
            if (kind[p] == 1) qs.push_back(p);
        if (!qs.empty()) {
            // one chi table per distinct df of the t's QMC points (a function of df alone)
            std::vector<double> dfs;
            for (std::int64_t p : qs)
                if (std::isfinite(df[batch[p]])) dfs.push_back(df[batch[p]]);
            std::sort(dfs.begin(), dfs.end());
            dfs.erase(std::unique(dfs.begin(), dfs.end()), dfs.end());
            std::vector<ChiTable> chis(dfs.size());
            for_pieces(static_cast<std::int64_t>(dfs.size()), 1, static_cast<std::int64_t>(dfs.size()) << 20,
                       [&](std::int64_t, std::int64_t c0, std::int64_t c1) {
                           for (std::int64_t c = c0; c < c1; ++c) chis[c] = chi_table(dfs[c]);
                       });
            auto one = [&](std::int64_t p) {
                const std::int64_t b = batch[p];
                const double* xp = x.data() + p * d;
                const double* S = cov.data() + b * dd;
                std::vector<int> idx;
                for (int i = 0; i < d; ++i)
                    if (xp[i] < kInf) idx.push_back(i);
                const int n = static_cast<int>(idx.size());
                std::vector<double> Sub(static_cast<std::size_t>(n) * n), lim(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i) {
                    lim[i] = xp[idx[i]];
                    for (int j = 0; j < n; ++j) {
                        const int a = std::max(idx[i], idx[j]), c = std::min(idx[i], idx[j]);
                        Sub[static_cast<std::size_t>(i) * n + j] = S[a * d + c];
                    }
                }
                Problem P;
                if (std::isfinite(df[b]))
                    P.chi = &chis[std::lower_bound(dfs.begin(), dfs.end(), df[b]) - dfs.begin()];
                if (!reorder(std::move(Sub), std::move(lim), n, df[b], P)) {
                    val[p] = err[p] = kNaN;
                    return;
                }
                const std::int64_t cap = max_points > 0 ? max_points : 1000000 * static_cast<std::int64_t>(n);
                const Estimate e = qmc(P, atol, cap);
                val[p] = e.value, err[p] = e.error;
            };
            const std::int64_t nq = static_cast<std::int64_t>(qs.size());
            if (nq >= parallel::num_threads()) {
                for_pieces(nq, 1, std::numeric_limits<std::int64_t>::max() / 2,
                           [&](std::int64_t, std::int64_t q0, std::int64_t q1) {
                               for (std::int64_t q = q0; q < q1; ++q) one(qs[static_cast<std::size_t>(q)]);
                           });
            } else {
                for (std::int64_t p : qs) one(p);
            }
        }
    }
    return nb::make_tuple(wrap(std::move(val_a)), wrap(std::move(err_a)));
}

}  // namespace

void bind_stats_mvcdf(nb::module_& m) {
    m.def("mv_cdf", &mv_cdf, nb::arg("cov"), nb::arg("x"), nb::arg("batch"), nb::arg("df"), nb::arg("atol"),
          nb::arg("max_points"));
}

}  // namespace sd
