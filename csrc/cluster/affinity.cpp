// Affinity propagation (docs/design/cluster.md §2.9, §3.12; decisions CL40, CL91-CL94): scikit-learn 1.9.1's
// _affinity_propagation without its noise. Points are numbered by their key (CL7: observations in key order;
// a precomputed similarity matrix keeps the input index), so every argmax's tie goes to the smaller key, and the
// floating-point operations are scikit-learn's, one by one:
//
//   S          -sqeuclidean (spatial's difference form) of the observations, or the given similarities; a -0.0
//              becomes +0.0 (scikit-learn's S + noise does that with zero noise), the preferences on the diagonal.
//   R          per row i: Y = max_k (A + S)_ik at I (the first), Y2 the largest of the others; the new
//              R_ik = S_ik - Y (S_iI - Y2 at k = I) and R = fl(fl(R d) + fl(R_new (1 - d))).
//   A          Rp_ik = max(R_ik, 0) (R_kk on the diagonal), colsum_k = Rp_0k + Rp_1k + ... (numpy's reduction along
//              axis 0, row by row); T_ik = max(Rp_ik - colsum_k, 0) (Rp_kk - colsum_k on the diagonal) and
//              A = fl(fl(A d) - fl(T (1 - d))).
//   stop       E_k = A_kk + R_kk > 0; after convergence_iter iterations, stop when every E_k stayed the same over the
//              last convergence_iter and at least one is true (n_iter = it + 1), else after max_iter.
//   exemplars  I = the points with E; every point to the exemplar of largest S (the first in I on a tie), each
//              exemplar to itself; each cluster's exemplar becomes its member of largest column sum of S over the
//              cluster's rows (rows in key order, the first on a tie); then the assignment again.
//   equal      n = 1, or every off-diagonal similarity equal and every preference equal: scikit-learn's answer
//              without iterating (each point its own exemplar when the preference is above the similarities, else
//              one cluster with the first point).
//
// The rows (the A and R updates) and the column sums (blocks of columns, each scanning the rows in order) run side
// by side; every value is one point's or one column's fixed sequence of operations, so the result has the same bits
// for any thread count.
//
// _core.cluster:
//   affinity_propagation(X, S, pref, damping, max_iter, convergence_iter)
//       -> (exemplars (K,) keys in the order of I, assignment (n,) the index into I of each key, n_iter, converged,
//           the preference used when pref is None (the median of S) or nan)
//       X: observations in key order or None; S: (n, n) similarities or None; pref: (n,) or None
#include "cluster/cluster.hpp"
#include "cluster/partition.hpp"
#include "parallel/pool.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::cluster {

namespace {

constexpr int64_t kRowTask = 16;   // rows of a task
constexpr int64_t kColBlock = 256;  // columns of a column-sum task

inline double plus_zero(double v) { return v + 0.0; }  // -0.0 to +0.0, anything else unchanged

// numpy's median of the n^2 values (the mean of the two middle ones for an even count)
double median_of(const std::vector<double>& S) {
    std::vector<double> v(S);
    const size_t m = v.size(), h = m / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(h), v.end());
    const double hi = v[h];
    if (m % 2) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(h));
    return (lo + hi) / 2.0;
}

struct Out {
    std::vector<int64_t> I, c;
    int64_t n_iter = 0;
    bool converged = false;
};

// c_i = the index into I of the exemplar of largest S_i (the first on a tie), then c[I[q]] = q
void assign(const std::vector<double>& S, int64_t n, const std::vector<int64_t>& I, std::vector<int64_t>& c) {
    const int64_t K = static_cast<int64_t>(I.size());
    c.assign(static_cast<size_t>(n), 0);
    auto body = [&](int64_t t) {
        for (int64_t i = t * kRowTask; i < std::min(n, (t + 1) * kRowTask); ++i) {
            const double* s = S.data() + i * n;
            int64_t best = 0;
            for (int64_t q = 1; q < K; ++q)
                if (s[I[static_cast<size_t>(q)]] > s[I[static_cast<size_t>(best)]]) best = q;
            c[static_cast<size_t>(i)] = best;
        }
    };
    const int64_t nt = (n + kRowTask - 1) / kRowTask;
    if (nt > 1 && n * K >= kSplitWork) parallel::parallel_for(nt, body, kRowTask * K);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    for (int64_t q = 0; q < K; ++q) c[static_cast<size_t>(I[static_cast<size_t>(q)])] = q;
}

Out propagate(std::vector<double>& S, int64_t n, double d, int64_t max_iter, int64_t conv) {
    Out out;
    const double omd = 1.0 - d;
    std::vector<double> R(static_cast<size_t>(n * n), 0.0), A(static_cast<size_t>(n * n), 0.0);
    std::vector<double> col(static_cast<size_t>(n)), Akk(static_cast<size_t>(n));
    std::vector<unsigned char> ring(static_cast<size_t>(n * conv), 0), E(static_cast<size_t>(n), 0);
    std::vector<int64_t> count(static_cast<size_t>(n), 0);
    const int64_t nt = (n + kRowTask - 1) / kRowTask;
    const int64_t nc = (n + kColBlock - 1) / kColBlock;
    const bool split = n * n >= kSplitWork;
    // One row's A update (with the previous iteration's R and column sums) and R update. Signed zeros aside (NEON's
    // max gives +0.0 where numpy's maximum may keep -0.0), these are scikit-learn's operations; a zero's sign never
    // changes a comparison or a nonzero value here, so the exemplars, labels and iterations are the same.
    // cnext (one thread; CL100): the new R's column sums made row by row as R's rows are written, the same additions in the
    // same row order as cols() below (block by block, its pairs and its tail), so R is not read a second time
    auto rows = [&](int64_t t, bool update_a, double* cnext) {
        const float64x2_t vd = vdupq_n_f64(d), vo = vdupq_n_f64(omd), z = vdupq_n_f64(0.0);
        const double ninf = -std::numeric_limits<double>::infinity();
        std::vector<double> tmpv(static_cast<size_t>(n) + 2);
        double* v = tmpv.data();
        for (int64_t i = t * kRowTask; i < std::min(n, (t + 1) * kRowTask); ++i) {
            double* a = A.data() + i * n;
            double* r = R.data() + i * n;
            const double* s = S.data() + i * n;
            int64_t k = 0;
            if (update_a) {
                const double aii = a[i], rii = r[i];
                for (; k + 2 <= n; k += 2) {
                    const float64x2_t rp = vmaxq_f64(vld1q_f64(r + k), z);
                    const float64x2_t tk = vmaxq_f64(vsubq_f64(rp, vld1q_f64(col.data() + k)), z);
                    const float64x2_t an = vsubq_f64(vmulq_f64(vld1q_f64(a + k), vd), vmulq_f64(tk, vo));
                    vst1q_f64(a + k, an);
                    vst1q_f64(v + k, vaddq_f64(an, vld1q_f64(s + k)));
                }
                for (; k < n; ++k) {
                    const double rp = r[k] >= 0.0 ? r[k] : 0.0;
                    double tk = rp - col[static_cast<size_t>(k)];
                    tk = tk > 0.0 ? tk : 0.0;
                    a[k] = a[k] * d - tk * omd;
                    v[k] = a[k] + s[k];
                }
                a[i] = aii * d - (rii - col[static_cast<size_t>(i)]) * omd;  // the diagonal is not clipped
                v[i] = a[i] + s[i];
            } else {
                for (; k + 2 <= n; k += 2) vst1q_f64(v + k, vaddq_f64(vld1q_f64(a + k), vld1q_f64(s + k)));
                for (; k < n; ++k) v[k] = a[k] + s[k];
            }
            // Y = max (A + S), its first place I, Y2 = the largest of the others (scikit-learn: tmp[I] = -inf): one pass,
            // lane by lane the largest, its first place (strict >) and the second largest of the values seen, the
            // two lanes met at the end (the larger, the first place on a tie)
            double best, second;
            int64_t at;
            {
                float64x2_t b1 = vdupq_n_f64(ninf), b2 = b1;
                uint64x2_t a1 = vdupq_n_u64(0), kv = {0, 1};
                const uint64x2_t two = vdupq_n_u64(2);
                for (k = 0; k + 2 <= n; k += 2, kv = vaddq_u64(kv, two)) {
                    const float64x2_t x = vld1q_f64(v + k);
                    const uint64x2_t gt = vcgtq_f64(x, b1);
                    b2 = vmaxq_f64(b2, vminq_f64(b1, x));
                    a1 = vbslq_u64(gt, kv, a1);
                    b1 = vmaxq_f64(b1, x);
                }
                const double l0 = vgetq_lane_f64(b1, 0), l1 = vgetq_lane_f64(b1, 1);
                const auto j0 = static_cast<int64_t>(vgetq_lane_u64(a1, 0)),
                           j1 = static_cast<int64_t>(vgetq_lane_u64(a1, 1));
                const bool other = (l1 > l0) | ((l1 == l0) & (j1 < j0));
                best = other ? l1 : l0, at = other ? j1 : j0;
                second = std::max(std::max(vgetq_lane_f64(b2, 0), vgetq_lane_f64(b2, 1)), std::min(l0, l1));
                for (; k < n; ++k) {  // the odd last value
                    const double x = v[k];
                    second = std::max(second, std::min(best, x));
                    at = x > best ? k : at;
                    best = std::max(best, x);
                }
            }
            const double rat = r[at];
            const float64x2_t vb = vdupq_n_f64(best);
            for (k = 0; k + 2 <= n; k += 2)
                vst1q_f64(
                    r + k, vaddq_f64(vmulq_f64(vld1q_f64(r + k), vd), vmulq_f64(vsubq_f64(vld1q_f64(s + k), vb), vo))
                );
            for (; k < n; ++k) r[k] = r[k] * d + (s[k] - best) * omd;
            r[at] = rat * d + (s[at] - second) * omd;
            if (cnext) {
                const double ci = cnext[i];
                for (int64_t k0 = 0; k0 < n; k0 += kColBlock) {
                    const int64_t k1 = std::min(n, k0 + kColBlock);
                    int64_t q = k0;
                    if (i == 0) {
                        for (; q < k1; ++q) cnext[q] = q == 0 ? r[q] : (r[q] >= 0.0 ? r[q] : 0.0);
                        continue;
                    }
                    for (; q + 2 <= k1; q += 2)
                        vst1q_f64(cnext + q, vaddq_f64(vld1q_f64(cnext + q), vmaxq_f64(vld1q_f64(r + q), z)));
                    for (; q < k1; ++q) cnext[q] = cnext[q] + (r[q] >= 0.0 ? r[q] : 0.0);
                }
                if (i > 0) cnext[i] = ci + r[i];
            }
        }
    };
    auto cols = [&](int64_t t) {
        const int64_t k0 = t * kColBlock, k1 = std::min(n, (t + 1) * kColBlock);
        double* c = col.data();
        const float64x2_t z = vdupq_n_f64(0.0);
        for (int64_t k = k0; k < k1; ++k) {
            const double v = R[static_cast<size_t>(k)];
            c[k] = k == 0 ? v : (v >= 0.0 ? v : 0.0);
        }
        for (int64_t i = 1; i < n; ++i) {
            const double* r = R.data() + i * n;
            const bool diag = i >= k0 && i < k1;
            const double ci = diag ? c[i] : 0.0;
            int64_t k = k0;
            for (; k + 2 <= k1; k += 2) vst1q_f64(c + k, vaddq_f64(vld1q_f64(c + k), vmaxq_f64(vld1q_f64(r + k), z)));
            for (; k < k1; ++k) c[k] = c[k] + (r[k] >= 0.0 ? r[k] : 0.0);
            if (diag) c[i] = ci + r[i];
        }
    };
    int64_t it = 0;
    std::vector<double> cn(static_cast<size_t>(n));
    for (; it < max_iter; ++it) {
        const bool upd = it > 0;
        if (nt > 1 && split && parallel::available()) {
            parallel::parallel_for(nt, [&](int64_t t) { rows(t, upd, nullptr); }, kRowTask * n * 8);
            if (nc > 1) parallel::parallel_for(nc, cols, kColBlock * n * 2);
            else cols(0);
        } else {
            for (int64_t t = 0; t < nt; ++t) rows(t, upd, cn.data());
            col.swap(cn);
        }
        int64_t K = 0;
        const int64_t slot = it % conv;
        for (int64_t k = 0; k < n; ++k) {
            const double rkk = R[static_cast<size_t>(k * n + k)];
            const double akk = A[static_cast<size_t>(k * n + k)] * d - (rkk - col[static_cast<size_t>(k)]) * omd;
            Akk[static_cast<size_t>(k)] = akk;
            const unsigned char e = (akk + rkk) > 0.0;
            E[static_cast<size_t>(k)] = e;
            K += e;
            unsigned char& old = ring[static_cast<size_t>(k * conv + slot)];
            count[static_cast<size_t>(k)] += static_cast<int64_t>(e) - static_cast<int64_t>(old);
            old = e;
        }
        if (it >= conv) {
            bool steady = true;
            for (int64_t k = 0; k < n && steady; ++k) {
                const int64_t v = count[static_cast<size_t>(k)];
                steady = v == conv || v == 0;
            }
            if (steady && K > 0) {
                out.converged = true;
                break;
            }
        }
    }
    out.n_iter = std::min(it + 1, max_iter);
    for (int64_t k = 0; k < n; ++k)
        if (E[static_cast<size_t>(k)]) out.I.push_back(k);
    const int64_t K = static_cast<int64_t>(out.I.size());
    if (K == 0) return out;
    assign(S, n, out.I, out.c);
    // each cluster's member of largest column sum over the cluster's rows
    std::vector<std::vector<int64_t>> members(static_cast<size_t>(K));
    for (int64_t i = 0; i < n; ++i) members[static_cast<size_t>(out.c[static_cast<size_t>(i)])].push_back(i);
    auto refine = [&](int64_t q) {
        const std::vector<int64_t>& ii = members[static_cast<size_t>(q)];
        const int64_t m = static_cast<int64_t>(ii.size());
        std::vector<double> sum(static_cast<size_t>(m));
        for (int64_t b = 0; b < m; ++b)
            sum[static_cast<size_t>(b)] = S[static_cast<size_t>(ii[0] * n + ii[static_cast<size_t>(b)])];
        for (int64_t a = 1; a < m; ++a) {
            const double* s = S.data() + ii[static_cast<size_t>(a)] * n;
            for (int64_t b = 0; b < m; ++b)
                sum[static_cast<size_t>(b)] = sum[static_cast<size_t>(b)] + s[ii[static_cast<size_t>(b)]];
        }
        int64_t best = 0;
        for (int64_t b = 1; b < m; ++b)
            if (sum[static_cast<size_t>(b)] > sum[static_cast<size_t>(best)]) best = b;
        out.I[static_cast<size_t>(q)] = ii[static_cast<size_t>(best)];
    };
    if (K > 1 && split) parallel::parallel_for(K, refine, n * n / K + 1);
    else
        for (int64_t q = 0; q < K; ++q) refine(q);
    assign(S, n, out.I, out.c);
    return out;
}

nb::object affinity_py(nb::handle Xo, nb::handle So, nb::handle po, double damping, int64_t max_iter, int64_t conv) {
    int64_t n, p = 0;
    const double* X = nullptr;
    const double* S0 = nullptr;
    if (!Xo.is_none()) {
        X = matrix_arg(Xo, "X", n, p);
    } else {
        int64_t m;
        S0 = matrix_arg(So, "S", n, m);
        if (m != n) throw nb::value_error("S must be square");
    }
    const double* pref = nullptr;
    if (!po.is_none()) {
        int64_t m;
        pref = vector_arg(po, "pref", m);
        if (m != n) throw nb::value_error("pref: one a point");
    }
    if (n < 1 || max_iter < 1 || conv < 1) throw nb::value_error("n, max_iter, convergence_iter");
    Out out;
    double used = std::numeric_limits<double>::quiet_NaN();
    {
        const parallel::Unlocked unlocked(n * n >= kSplitWork);
        std::vector<double> S(static_cast<size_t>(n * n));
        if (X) {
            auto body = [&](int64_t t) {
                for (int64_t i = t * kRowTask; i < std::min(n, (t + 1) * kRowTask); ++i)
                    for (int64_t k = 0; k < n; ++k)
                        S[static_cast<size_t>(i * n + k)] = i == k ? 0.0 : plus_zero(-sqd(X + i * p, X + k * p, p));
            };
            const int64_t nt = (n + kRowTask - 1) / kRowTask;
            if (nt > 1 && n * n * p >= kSplitWork) parallel::parallel_for(nt, body, kRowTask * n * p);
            else
                for (int64_t t = 0; t < nt; ++t) body(t);
        } else {
            for (int64_t q = 0; q < n * n; ++q) S[static_cast<size_t>(q)] = S0[q];
        }
        std::vector<double> pr(static_cast<size_t>(n));
        if (pref) {
            for (int64_t k = 0; k < n; ++k) pr[static_cast<size_t>(k)] = pref[k];
        } else {
            used = median_of(S);
            std::fill(pr.begin(), pr.end(), used);
        }
        // scikit-learn's equal case, decided on S before the preferences go on the diagonal
        bool equal = n == 1;
        if (!equal) {
            bool eq_pref = true, eq_sim = true;
            for (int64_t k = 1; k < n && eq_pref; ++k) eq_pref = pr[static_cast<size_t>(k)] == pr[0];
            const double first = S[1];
            for (int64_t i = 0; i < n && eq_sim && eq_pref; ++i)
                for (int64_t k = 0; k < n; ++k)
                    if (k != i && !(S[static_cast<size_t>(i * n + k)] == first)) {
                        eq_sim = false;
                        break;
                    }
            equal = eq_pref && eq_sim;
        }
        if (equal) {
            const bool all = pr[0] > S[static_cast<size_t>(n - 1)];
            out.converged = true, out.n_iter = 0;
            if (all) {
                out.I.resize(static_cast<size_t>(n)), out.c.resize(static_cast<size_t>(n));
                for (int64_t k = 0; k < n; ++k) out.I[static_cast<size_t>(k)] = k, out.c[static_cast<size_t>(k)] = k;
            } else {
                out.I.assign(1, 0), out.c.assign(static_cast<size_t>(n), 0);
            }
        } else {
            for (auto& v : S) v = plus_zero(v);
            for (int64_t k = 0; k < n; ++k) S[static_cast<size_t>(k * n + k)] = plus_zero(pr[static_cast<size_t>(k)]);
            out = propagate(S, n, damping, max_iter, conv);
        }
    }
    const int64_t K = static_cast<int64_t>(out.I.size());
    if (out.c.empty()) out.c.assign(static_cast<size_t>(n), -1);
    return nb::make_tuple(wrap(int_array(out.I, K)), wrap(int_array(out.c, n)), out.n_iter, out.converged, used);
}

}  // namespace

void bind_affinity(nb::module_& m) {
    m.def(
        "affinity_propagation", &affinity_py, nb::arg("X").none(), nb::arg("S").none(), nb::arg("pref").none(),
        nb::arg("damping"), nb::arg("max_iter"), nb::arg("convergence_iter")
    );
}

}  // namespace sd::cluster
