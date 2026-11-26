// The clustering indices, the gap statistic and Gower's distance (docs/design/cluster.md §2.6, §3.8, §3.9, §4.6;
// decisions CL13, CL14, CL36-CL38 and milestone CL5's, CL82-CL90). Points arrive in key order (CL7: the rank of the
// lexicographic order; precomputed distances keep the input index), and labels as codes 0 .. K - 1.
//
//   table      the contingency table of two codings: its nonzero cells row by row, row and column sums (int64).
//   external   from the table: the pair counts sum C(n_ij, 2), sum C(a_i, 2), sum C(b_j, 2) (exact integers; the
//              Python layer makes ARI, RI, FM and the pair confusion matrix from them, rounding once), the mutual
//              information MI = sum (n_ij / n) L(n n_ij, a_i b_j) and the entropies H = sum (a_i / n) L(n, a_i),
//              L(x, y) = log(x / y) for exact integers x, y (log1p((x - y) / y) when x / y is within [1/2, 3/2], so
//              that L has a relative error of a few u everywhere), each sum by the pairwise tree in cell order; and
//              the expected mutual information of AMI (Vinh et al. 2010): for each pair of a row size alpha and a
//              column size beta, the hypergeometric weights w(c) of the cell count c by the ratio recurrence
//              w(c + 1) / w(c) = (alpha - c)(beta - c) / ((c + 1)(n - alpha - beta + c + 1)) outward from the mode
//              (w = 1 there), cut where the remaining tail is below 2^-80 of the sum, normalised by their sum (no
//              lgamma differences: scikit-learn's cancel to 10^-8 at n = 10^6), times the count of such row and
//              column pairs; the pair values in (alpha, beta) order by the pairwise tree.
//   silhouette per point: the mean distance to each cluster (its sum in the cluster's key order, four
//              interleaved accumulators by the position in the cluster), a its own cluster's (self excluded), b
//              the smallest other's, s = (b - a) / max(a, b) (0 for a singleton and for a = b = 0); observations
//              by spatial's tile kernel (euclidean, sqeuclidean, cityblock, chebyshev: the bits of cdist) on the
//              data scaled by CL43's power of two (exact; the ratio is the same), or a condensed matrix. The mean
//              by CL48's chunk sums in key order.
//   ch_db      Calinski-Harabasz and Davies-Bouldin on the points less the first one (both are invariant under
//              translation; far from the origin the means keep their digits) and CL43's scale: the cluster means
//              shifted by the cluster's first point (x_f + sum (x - x_f) / n_c: exact for a cluster of equal
//              points), CL48's chunk sums.
//   gap        R cluster's clusGap: log W_k for k = 1 .. k_max of the data and of n_reference reference sets drawn
//              uniformly in the box of the data's principal axes (scaledPCA: the eigenvectors of the centred
//              cross-product by cyclic Jacobi) or of its own axes (original); reference set b draws element
//              b 2^40 + i p + d (CL41); each set's clusterings by sd's kmeans (sub-key at start 2^23 + s k_max + k)
//              or PAM; W_k = 1/2 sum_r D_r / n_r, D_r = sum over the pairs of cluster r of d^power (spatial's
//              euclidean, on CL43's grid: log W less e power log 2). The sets run side by side, each on its own
//              thread.
//   gower      R's daisy(metric = "gower"): per pair, sum_j w_j delta_j / sum_j w_j over the variables both have
//              (interval |x - y| / range, nominal and symmetric binary [x != y], asymmetric binary without the
//              pairs where both are 0).
//
// _core.cluster:
//   contingency(a, b, ka, kb) -> (rows, cols, counts, row sums, col sums)
//   external(a, b, ka, kb, emi) -> (pairs cells, pairs rows, pairs cols, mi, h_a, h_b, emi or nan)
//   silhouette(X, D, codes, K, metric) -> (values (n,) by key, mean)      metric 0 euclidean, 1 sqeuclidean,
//                                                                          2 cityblock, 3 chebyshev; D condensed
//   ch_db(X, codes, K) -> (calinski_harabasz, davies_bouldin)
//   gap(X, order, k_max, B, d_power, reference, method, key0, key1) -> (log_w, e_log_w, se, log_w_ref (B, k_max))
//   gap_logw(X, codes, K, d_power) -> log W of a partition (rows in key order)
//   gower(V, types, scale, weights) -> condensed distances
//   subkey(key0, key1, r) -> (key0', key1')   a sub-call's two words at start r of the stream (CL41)
#include "cluster/cluster.hpp"
#include "cluster/partition.hpp"
#include "parallel/pool.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sd::cluster {

namespace {

using u128 = unsigned __int128;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// The pairwise sum of v[0 .. m) in index order: halves down to eight terms, each added left to right.
double psum(const double* v, int64_t m) {
    if (m <= 0) return 0.0;
    if (m <= 8) {
        double s = v[0];
        for (int64_t i = 1; i < m; ++i) s = s + v[i];
        return s;
    }
    const int64_t h = m / 2;
    return psum(v, h) + psum(v + h, m - h);
}

// CL48's chunk sum of v[0 .. n): kChunk values a chunk, added one by one, the chunks met by the pairwise tree.
double chunk_sum(const double* v, int64_t n) {
    double s = 0.0;
    pairwise_chunks(
        chunks_of(n), 1,
        [&](int64_t c, double* out) {
            double t = 0.0;
            for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) t = t + v[i];
            out[0] = t;
        },
        &s, kChunk
    );
    return s;
}

// log(x / y) of two positive exact integers (the relative error a few u: §4.6)
double log_ratio(u128 x, u128 y) {
    const double fy = static_cast<double>(y);
    if (x >= y) {
        const u128 d = x - y;
        if (2 * d <= y) return std::log1p(static_cast<double>(d) / fy);
    } else {
        const u128 d = y - x;
        if (2 * d <= y) return std::log1p(-(static_cast<double>(d) / fy));
    }
    return std::log(static_cast<double>(x) / fy);
}

// ---- the contingency table ---------------------------------------------------------------------------------------

struct Table {
    std::vector<int64_t> row, col, cnt;  // the nonzero cells, row by row
    std::vector<int64_t> a, b;           // row and column sums
    int64_t n = 0;
};

Table make_table(const int64_t* A, const int64_t* B, int64_t n, int64_t ka, int64_t kb) {
    Table T;
    T.n = n;
    T.a.assign(static_cast<size_t>(ka), 0);
    T.b.assign(static_cast<size_t>(kb), 0);
    for (int64_t i = 0; i < n; ++i) ++T.a[static_cast<size_t>(A[i])], ++T.b[static_cast<size_t>(B[i])];
    const double cells = static_cast<double>(ka) * static_cast<double>(kb);
    if (cells <= std::max(4.0 * static_cast<double>(n), 1048576.0)) {
        std::vector<int64_t> M(static_cast<size_t>(ka * kb), 0);
        for (int64_t i = 0; i < n; ++i) ++M[static_cast<size_t>(A[i] * kb + B[i])];
        for (int64_t r = 0; r < ka; ++r)
            for (int64_t c = 0; c < kb; ++c)
                if (const int64_t v = M[static_cast<size_t>(r * kb + c)]) {
                    T.row.push_back(r), T.col.push_back(c), T.cnt.push_back(v);
                }
        return T;
    }
    std::vector<std::uint64_t> key(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        key[static_cast<size_t>(i)] =
            static_cast<std::uint64_t>(A[i]) * static_cast<std::uint64_t>(kb) + static_cast<std::uint64_t>(B[i]);
    std::sort(key.begin(), key.end());
    for (int64_t i = 0; i < n;) {
        int64_t j = i + 1;
        while (j < n && key[static_cast<size_t>(j)] == key[static_cast<size_t>(i)]) ++j;
        const std::uint64_t k = key[static_cast<size_t>(i)];
        T.row.push_back(static_cast<int64_t>(k / static_cast<std::uint64_t>(kb)));
        T.col.push_back(static_cast<int64_t>(k % static_cast<std::uint64_t>(kb)));
        T.cnt.push_back(j - i);
        i = j;
    }
    return T;
}

void codes_args(
    nb::handle ao, nb::handle bo, int64_t ka, int64_t kb, const int64_t*& A, const int64_t*& B, int64_t& n
) {
    int64_t m;
    A = index_arg(ao, "a", n);
    B = index_arg(bo, "b", m);
    if (m != n) throw nb::value_error("a, b: one code a point");
    if (ka < 1 || kb < 1) throw nb::value_error("ka, kb");
    for (int64_t i = 0; i < n; ++i)
        if (A[i] < 0 || A[i] >= ka || B[i] < 0 || B[i] >= kb) throw nb::value_error("codes out of range");
}

nb::object contingency_py(nb::handle ao, nb::handle bo, int64_t ka, int64_t kb) {
    const int64_t *A, *B;
    int64_t n;
    codes_args(ao, bo, ka, kb, A, B, n);
    Table T;
    {
        const parallel::Unlocked unlocked(n >= kSplitWork);
        T = make_table(A, B, n, ka, kb);
    }
    const int64_t m = static_cast<int64_t>(T.cnt.size());
    return nb::make_tuple(
        wrap(int_array(T.row, m)), wrap(int_array(T.col, m)), wrap(int_array(T.cnt, m)), wrap(int_array(T.a, ka)),
        wrap(int_array(T.b, kb))
    );
}

// ---- MI, entropies and the expected MI ---------------------------------------------------------------------------

int64_t pairs_of(int64_t v) { return v * (v - 1) / 2; }

double mutual_info(const Table& T) {
    if (T.a.size() == 1 || T.b.size() == 1) return 0.0;  // one side one cluster: every term is log 1
    const int64_t m = static_cast<int64_t>(T.cnt.size());
    std::vector<double> t(static_cast<size_t>(m));
    const double n = static_cast<double>(T.n);
    for (int64_t q = 0; q < m; ++q) {
        const int64_t c = T.cnt[static_cast<size_t>(q)];
        const u128 x = static_cast<u128>(T.n) * static_cast<u128>(c);
        const u128 y = static_cast<u128>(T.a[static_cast<size_t>(T.row[static_cast<size_t>(q)])]) *
                       static_cast<u128>(T.b[static_cast<size_t>(T.col[static_cast<size_t>(q)])]);
        t[static_cast<size_t>(q)] = (static_cast<double>(c) / n) * log_ratio(x, y);
    }
    return std::max(psum(t.data(), m), 0.0);
}

double entropy(const std::vector<int64_t>& v, int64_t n) {
    std::vector<double> t;
    for (int64_t c : v)
        if (c > 0) t.push_back((static_cast<double>(c) / static_cast<double>(n)) * log_ratio(n, c));
    if (t.size() <= 1) return 0.0;
    return psum(t.data(), static_cast<int64_t>(t.size()));
}

// E[MI] of one row size alpha and one column size beta under the hypergeometric model: sum over the cell count c
// of P(c) (c / n) L(n c, alpha beta), P from the ratio recurrence normalised by its sum.
double emi_pair(int64_t alpha, int64_t beta, int64_t n, std::vector<double>& w) {
    const int64_t lo = std::max<int64_t>(0, alpha + beta - n), hi = std::min(alpha, beta);
    if (hi <= 0) return 0.0;
    int64_t mode =
        static_cast<int64_t>((static_cast<u128>(alpha + 1) * static_cast<u128>(beta + 1)) / static_cast<u128>(n + 2));
    mode = std::clamp(mode, lo, hi);
    const double rest = static_cast<double>(n - alpha - beta);  // n - alpha - beta + c + 1 = rest + c + 1
    constexpr double kCut = 0x1p-80;
    // downward from the mode (stored reversed), then upward
    std::vector<double> down;
    double S = 1.0;
    {
        double v = 1.0;
        for (int64_t c = mode; c > lo;) {
            // w(c - 1) = w(c) c (rest + c) / ((alpha - c + 1)(beta - c + 1))
            const double num = static_cast<double>(c) * (rest + static_cast<double>(c));
            const double den = static_cast<double>(alpha - c + 1) * static_cast<double>(beta - c + 1);
            const double r = num / den;
            v = v * r;
            --c;
            down.push_back(v);
            S = S + v;
            if (r < 1.0 && v <= kCut * S * (1.0 - r)) break;
        }
    }
    w.clear();
    w.assign(down.rbegin(), down.rend());
    const int64_t c_first = mode - static_cast<int64_t>(down.size());
    w.push_back(1.0);
    {
        double v = 1.0;
        for (int64_t c = mode; c < hi;) {
            const double num = static_cast<double>(alpha - c) * static_cast<double>(beta - c);
            const double den = static_cast<double>(c + 1) * (rest + static_cast<double>(c + 1));
            const double r = num / den;
            v = v * r;
            ++c;
            w.push_back(v);
            S = S + v;
            if (r < 1.0 && v <= kCut * S * (1.0 - r)) break;
        }
    }
    double tot = 0.0, num = 0.0;
    const double nd = static_cast<double>(n);
    const u128 ab = static_cast<u128>(alpha) * static_cast<u128>(beta);
    for (size_t q = 0; q < w.size(); ++q) {
        const int64_t c = c_first + static_cast<int64_t>(q);
        tot = tot + w[q];
        if (c > 0) {
            const double t = (static_cast<double>(c) / nd) * log_ratio(static_cast<u128>(n) * static_cast<u128>(c), ab);
            num = num + w[q] * t;
        }
    }
    return num / tot;
}

double expected_mi(const Table& T) {
    if (T.a.size() == 1 || T.b.size() == 1) return 0.0;
    auto sizes = [](const std::vector<int64_t>& v) {  // (distinct size, how many) ascending
        std::vector<int64_t> s(v);
        std::sort(s.begin(), s.end());
        std::vector<std::pair<int64_t, int64_t>> out;
        for (int64_t x : s)
            if (!out.empty() && out.back().first == x) ++out.back().second;
            else out.push_back({x, 1});
        return out;
    };
    const auto A = sizes(T.a), B = sizes(T.b);
    const int64_t na = static_cast<int64_t>(A.size()), nb_ = static_cast<int64_t>(B.size());
    std::vector<double> val(static_cast<size_t>(na * nb_));
    auto body = [&](int64_t q) {
        std::vector<double> w;
        const auto& [alpha, ca] = A[static_cast<size_t>(q / nb_)];
        const auto& [beta, cb] = B[static_cast<size_t>(q % nb_)];
        val[static_cast<size_t>(q)] =
            (static_cast<double>(ca) * static_cast<double>(cb)) * emi_pair(alpha, beta, T.n, w);
    };
    const int64_t m = na * nb_;
    if (m > 1) parallel::parallel_for(m, body, 20000);
    else body(0);
    return psum(val.data(), m);
}

nb::object external_py(nb::handle ao, nb::handle bo, int64_t ka, int64_t kb, bool emi) {
    const int64_t *A, *B;
    int64_t n;
    codes_args(ao, bo, ka, kb, A, B, n);
    Table T;
    u128 sc = 0, sa = 0, sb = 0;
    double mi, ha, hb, e = kNaN;
    {
        const parallel::Unlocked unlocked(n >= kSplitWork || emi);
        T = make_table(A, B, n, ka, kb);
        for (int64_t v : T.cnt) sc += static_cast<u128>(pairs_of(v));
        for (int64_t v : T.a) sa += static_cast<u128>(pairs_of(v));
        for (int64_t v : T.b) sb += static_cast<u128>(pairs_of(v));
        mi = mutual_info(T);
        ha = entropy(T.a, n);
        hb = entropy(T.b, n);
        if (emi) e = expected_mi(T);
    }
    return nb::make_tuple(static_cast<int64_t>(sc), static_cast<int64_t>(sa), static_cast<int64_t>(sb), mi, ha, hb, e);
}

// ---- silhouette ------------------------------------------------------------------------------------------------

struct Groups {  // the points grouped by cluster, each cluster in key order
    std::vector<int64_t> start, member, size;
};

Groups group(const int64_t* codes, int64_t n, int64_t K) {
    Groups G;
    G.size.assign(static_cast<size_t>(K), 0);
    for (int64_t i = 0; i < n; ++i) ++G.size[static_cast<size_t>(codes[i])];
    G.start.assign(static_cast<size_t>(K + 1), 0);
    for (int64_t c = 0; c < K; ++c)
        G.start[static_cast<size_t>(c + 1)] = G.start[static_cast<size_t>(c)] + G.size[static_cast<size_t>(c)];
    G.member.resize(static_cast<size_t>(n));
    std::vector<int64_t> at(G.start.begin(), G.start.end() - 1);
    for (int64_t i = 0; i < n; ++i) G.member[static_cast<size_t>(at[static_cast<size_t>(codes[i])]++)] = i;
    return G;
}

// The four interleaved sums of a cluster's values (by the position mod 4 in the cluster) met as (s0 + s1) + (s2 + s3)
inline double meet4(const double* acc, int64_t m) {
    return m == 1   ? acc[0]
           : m == 2 ? acc[0] + acc[1]
           : m == 3 ? (acc[0] + acc[1]) + acc[2]
                    : (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

double sil_value(const double* S, const Groups& G, int64_t K, int64_t own) {
    const int64_t m = G.size[static_cast<size_t>(own)];
    if (m <= 1) return 0.0;
    const double a = S[own] / static_cast<double>(m - 1);
    double b = std::numeric_limits<double>::infinity();
    for (int64_t c = 0; c < K; ++c)
        if (c != own && G.size[static_cast<size_t>(c)] > 0)
            b = std::min(b, S[c] / static_cast<double>(G.size[static_cast<size_t>(c)]));
    const double mx = std::max(a, b);
    if (mx == 0.0) return 0.0;
    return (b - a) / mx;
}

enum class Fin { Euclid, Sq, None };

template <class Op, Fin F>
void sil_observations(
    const double* X, int64_t n, int64_t p, const int64_t* codes, const Groups& G, int64_t K, double* out
) {
    // the grouped rows packed in panels of 8 (spatial's tile layout)
    const int64_t np = (n + spatial::kPanel - 1) / spatial::kPanel;
    std::vector<double> panels(static_cast<size_t>(np * p * spatial::kPanel));
    std::vector<double> Gx(static_cast<size_t>(n * p));
    for (int64_t r = 0; r < n; ++r) {
        const double* x = X + G.member[static_cast<size_t>(r)] * p;
        std::copy(x, x + p, Gx.begin() + r * p);
    }
    for (int64_t q = 0; q < np; ++q) {
        const int64_t rows = std::min<int64_t>(spatial::kPanel, n - q * spatial::kPanel);
        spatial::pack_panel(Gx.data() + q * spatial::kPanel * p, p, rows, p, panels.data() + q * p * spatial::kPanel);
    }
    std::vector<int64_t> code_r(static_cast<size_t>(n)), pos_r(static_cast<size_t>(n));
    std::vector<char> last_r(static_cast<size_t>(n));
    for (int64_t c = 0; c < K; ++c)
        for (int64_t r = G.start[static_cast<size_t>(c)]; r < G.start[static_cast<size_t>(c + 1)]; ++r) {
            code_r[static_cast<size_t>(r)] = c;
            pos_r[static_cast<size_t>(r)] = r - G.start[static_cast<size_t>(c)];  // the place in the cluster
            last_r[static_cast<size_t>(r)] = r + 1 == G.start[static_cast<size_t>(c + 1)];
        }
    constexpr int64_t kPts = 4;
    const int64_t ntask = (n + kPts * 8 - 1) / (kPts * 8);
    auto body = [&](int64_t t) {
        std::vector<double> S(static_cast<size_t>(kPts * K));
        double sums[kPts * spatial::kPanel];
        for (int64_t i0 = t * kPts * 8; i0 < std::min(n, (t + 1) * kPts * 8); i0 += kPts) {
            const int64_t R = std::min(kPts, n - i0);
            const double* xr[kPts];
            for (int r = 0; r < kPts; ++r) xr[r] = X + (i0 + std::min<int64_t>(r, R - 1)) * p;
            double acc[kPts][4] = {};
            for (int64_t q = 0; q < np; ++q) {
                spatial::tile<Op, kPts, false>(xr, panels.data() + q * p * spatial::kPanel, nullptr, p, sums);
                const int64_t r0 = q * spatial::kPanel, rows = std::min<int64_t>(spatial::kPanel, n - r0);
                for (int64_t c = 0; c < rows; ++c) {
                    const int64_t r = r0 + c;
                    const int64_t pos = pos_r[static_cast<size_t>(r)], ps = pos & 3;
                    for (int i = 0; i < kPts; ++i) {
                        double d = sums[i * spatial::kPanel + c];
                        if constexpr (F == Fin::Euclid) {
                            d = spatial::euclid_finish<false>(d, xr[i], Gx.data() + r * p, nullptr, p);
                        } else if constexpr (F == Fin::Sq) {
                            d = spatial::sq_finish<false>(d, xr[i], Gx.data() + r * p, nullptr, p);
                        }
                        acc[i][ps] = pos < 4 ? d : acc[i][ps] + d;
                    }
                    if (last_r[static_cast<size_t>(r)]) {
                        const int64_t cc = code_r[static_cast<size_t>(r)];
                        const int64_t m = G.size[static_cast<size_t>(cc)];
                        for (int i = 0; i < kPts; ++i) S[static_cast<size_t>(i * K + cc)] = meet4(acc[i], m);
                    }
                }
            }
            for (int64_t i = 0; i < R; ++i) out[i0 + i] = sil_value(S.data() + i * K, G, K, codes[i0 + i]);
        }
    };
    if (ntask > 1 && n * n * p >= kSplitWork) parallel::parallel_for(ntask, body, 8 * kPts * n * (p + 4));
    else
        for (int64_t t = 0; t < ntask; ++t) body(t);
}

void sil_condensed(
    const double* D, int64_t n, const int64_t* codes, const Groups& G, int64_t K, double scale, double* out
) {
    constexpr int64_t kPts = 32;
    const int64_t ntask = (n + kPts - 1) / kPts;
    auto body = [&](int64_t t) {
        std::vector<double> S(static_cast<size_t>(K));
        for (int64_t i = t * kPts; i < std::min(n, (t + 1) * kPts); ++i) {
            for (int64_t c = 0; c < K; ++c) {
                double acc[4] = {0.0, 0.0, 0.0, 0.0};
                const int64_t b = G.start[static_cast<size_t>(c)], e = G.start[static_cast<size_t>(c + 1)];
                for (int64_t r = b; r < e; ++r) {
                    const int64_t j = G.member[static_cast<size_t>(r)];
                    const double d = i == j ? 0.0 : D[cpair(n, i, j)] * scale;
                    const int64_t ps = (r - b) & 3;
                    acc[ps] = r - b < 4 ? d : acc[ps] + d;
                }
                S[static_cast<size_t>(c)] = meet4(acc, e - b);
            }
            out[i] = sil_value(S.data(), G, K, codes[i]);
        }
    };
    if (ntask > 1 && n * n >= kSplitWork) parallel::parallel_for(ntask, body, kPts * n * 2);
    else
        for (int64_t t = 0; t < ntask; ++t) body(t);
}

nb::object silhouette_py(nb::handle Xo, nb::handle Do, nb::handle co, int64_t K, int metric) {
    int64_t n, p = 0, m;
    const int64_t* codes = index_arg(co, "codes", n);
    const double* X = nullptr;
    const double* D = nullptr;
    if (!Xo.is_none()) {
        int64_t nn;
        X = matrix_arg(Xo, "X", nn, p);
        if (nn != n) throw nb::value_error("X: one row a code");
    } else {
        D = vector_arg(Do, "D", m);
        if (m != n * (n - 1) / 2) throw nb::value_error("D: condensed");
    }
    for (int64_t i = 0; i < n; ++i)
        if (codes[i] < 0 || codes[i] >= K) throw nb::value_error("codes out of range");
    std::vector<double> out(static_cast<size_t>(n));
    double mean;
    {
        const parallel::Unlocked unlocked(n * n >= kSplitWork);
        const Groups G = group(codes, n, K);
        if (X) {
            const int e = scale_exponent(max_abs_of(X, n * p));
            std::vector<double> Xs;
            if (e != 0) Xs = scaled_copy(X, n * p, e), X = Xs.data();
            if (metric == 0) sil_observations<spatial::SqOp, Fin::Euclid>(X, n, p, codes, G, K, out.data());
            else if (metric == 1) sil_observations<spatial::SqOp, Fin::Sq>(X, n, p, codes, G, K, out.data());
            else if (metric == 2) sil_observations<spatial::AbsOp, Fin::None>(X, n, p, codes, G, K, out.data());
            else sil_observations<spatial::MaxOp, Fin::None>(X, n, p, codes, G, K, out.data());
        } else {
            const int e = scale_exponent(max_abs_of(D, m));
            sil_condensed(D, n, codes, G, K, std::ldexp(1.0, e), out.data());
        }
        mean = chunk_sum(out.data(), n) / static_cast<double>(n);
    }
    return nb::make_tuple(wrap(float_array(out, n)), mean);
}

// ---- Calinski-Harabasz and Davies-Bouldin ----------------------------------------------------------------------

// sum over the cluster's members (in key order) of f(row) - x_f, chunks of kChunk members met by the pairwise tree
void member_sums(const double* X, int64_t p, const Groups& G, int64_t c, double* out) {
    const int64_t b = G.start[static_cast<size_t>(c)], e = G.start[static_cast<size_t>(c + 1)];
    const double* xf = X + G.member[static_cast<size_t>(b)] * p;
    const int64_t m = e - b;
    PairwiseStack st(p);
    std::vector<double> buf(static_cast<size_t>(p));
    for (int64_t q = 0; q < m; q += kChunk) {
        std::fill(buf.begin(), buf.end(), 0.0);
        for (int64_t r = b + q; r < std::min(e, b + q + kChunk); ++r) {
            const double* x = X + G.member[static_cast<size_t>(r)] * p;
            for (int64_t d = 0; d < p; ++d) buf[static_cast<size_t>(d)] = buf[static_cast<size_t>(d)] + (x[d] - xf[d]);
        }
        st.push(buf.data());
    }
    st.finish(out);
}

nb::object ch_db_py(nb::handle Xo, nb::handle co, int64_t K) {
    int64_t n, p, m;
    const double* X = matrix_arg(Xo, "X", n, p);
    const int64_t* codes = index_arg(co, "codes", m);
    if (m != n) throw nb::value_error("codes: one a row");
    for (int64_t i = 0; i < n; ++i)
        if (codes[i] < 0 || codes[i] >= K) throw nb::value_error("codes out of range");
    double ch, db;
    {
        const parallel::Unlocked unlocked(n * p >= kSplitWork);
        // both indices are invariant under translation and scaling: the points less the first (exact for the
        // coordinates within a factor 2 of it: far from the origin the means keep their digits), then CL43's scale
        std::vector<double> Y(static_cast<size_t>(n * p));
        for (int64_t i = 0; i < n; ++i)
            for (int64_t d = 0; d < p; ++d) Y[static_cast<size_t>(i * p + d)] = X[i * p + d] - X[d];
        const int e = scale_exponent(max_abs_of(Y.data(), n * p));
        if (e != 0) Y = scaled_copy(Y.data(), n * p, e);
        X = Y.data();
        const Groups G = group(codes, n, K);
        std::vector<double> mu(static_cast<size_t>(K * p));
        auto mean_of = [&](int64_t c) {
            double* o = mu.data() + c * p;
            member_sums(X, p, G, c, o);
            const double* xf = X + G.member[static_cast<size_t>(G.start[static_cast<size_t>(c)])] * p;
            const double sz = static_cast<double>(G.size[static_cast<size_t>(c)]);
            for (int64_t d = 0; d < p; ++d) o[d] = xf[d] + o[d] / sz;
        };
        if (K > 1 && n * p >= kSplitWork) parallel::parallel_for(K, mean_of, n / K * p + 1);
        else
            for (int64_t c = 0; c < K; ++c) mean_of(c);
        // the overall mean, shifted by the first point
        std::vector<double> all(static_cast<size_t>(p));
        pairwise_chunks(
            chunks_of(n), p,
            [&](int64_t c, double* o) {
                std::fill(o, o + p, 0.0);
                for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i)
                    for (int64_t d = 0; d < p; ++d) o[d] = o[d] + (X[i * p + d] - X[d]);
            },
            all.data(), kChunk * p
        );
        for (int64_t d = 0; d < p; ++d)
            all[static_cast<size_t>(d)] = X[d] + all[static_cast<size_t>(d)] / static_cast<double>(n);
        // W: the squared distances to the own mean, in key order (CL48)
        std::vector<double> wq(static_cast<size_t>(n)), dq(static_cast<size_t>(n));
        auto rows = [&](int64_t t) {
            for (int64_t i = t * kChunk; i < std::min(n, (t + 1) * kChunk); ++i) {
                const double* c = mu.data() + codes[i] * p;
                double s;  // one sum of squares: sqd's and the euclidean distance's (spatial's finishes)
                spatial::pair_scalar<spatial::SqOp, false>(X + i * p, c, nullptr, p, &s);
                wq[static_cast<size_t>(i)] = spatial::sq_finish<false>(s, X + i * p, c, nullptr, p);
                dq[static_cast<size_t>(i)] = spatial::euclid_finish<false>(s, X + i * p, c, nullptr, p);
            }
        };
        const int64_t nt = chunks_of(n);
        if (nt > 1) parallel::parallel_for(nt, rows, kChunk * p);
        else rows(0);
        const double W = chunk_sum(wq.data(), n);
        std::vector<double> bt(static_cast<size_t>(K));
        for (int64_t c = 0; c < K; ++c)
            bt[static_cast<size_t>(c)] =
                static_cast<double>(G.size[static_cast<size_t>(c)]) * sqd(mu.data() + c * p, all.data(), p);
        const double B = psum(bt.data(), K);
        ch = W == 0.0 ? 1.0 : (B * static_cast<double>(n - K)) / (W * static_cast<double>(K - 1));
        // Davies-Bouldin: each cluster's mean distance to its mean (the members' sum in key order, CL48 chunks)
        std::vector<double> s(static_cast<size_t>(K));
        for (int64_t c = 0; c < K; ++c) {
            const int64_t b = G.start[static_cast<size_t>(c)], e2 = G.start[static_cast<size_t>(c + 1)];
            std::vector<double> v(static_cast<size_t>(e2 - b));
            for (int64_t r = b; r < e2; ++r)
                v[static_cast<size_t>(r - b)] = dq[static_cast<size_t>(G.member[static_cast<size_t>(r)])];
            s[static_cast<size_t>(c)] = chunk_sum(v.data(), e2 - b) / static_cast<double>(e2 - b);
        }
        std::vector<double> dc(static_cast<size_t>(K * K), 0.0);
        bool all_s0 = true, all_d0 = true;
        for (int64_t c = 0; c < K; ++c) all_s0 = all_s0 && s[static_cast<size_t>(c)] == 0.0;
        for (int64_t c = 0; c < K; ++c)
            for (int64_t f = c + 1; f < K; ++f) {
                double t;
                const double* x = mu.data() + c * p;
                const double* y = mu.data() + f * p;
                spatial::pair_scalar<spatial::SqOp, false>(x, y, nullptr, p, &t);
                const double d = spatial::euclid_finish<false>(t, x, y, nullptr, p);
                dc[static_cast<size_t>(c * K + f)] = dc[static_cast<size_t>(f * K + c)] = d;
                all_d0 = all_d0 && d == 0.0;
            }
        if (all_s0 || all_d0) db = 0.0;
        else {
            std::vector<double> score(static_cast<size_t>(K));
            for (int64_t c = 0; c < K; ++c) {
                double best = 0.0;
                for (int64_t f = 0; f < K; ++f) {
                    const double d = dc[static_cast<size_t>(c * K + f)];
                    if (f == c || d == 0.0) continue;  // scikit-learn's inf distance: a ratio of 0
                    best = std::max(best, (s[static_cast<size_t>(c)] + s[static_cast<size_t>(f)]) / d);
                }
                score[static_cast<size_t>(c)] = best;
            }
            db = psum(score.data(), K) / static_cast<double>(K);
        }
    }
    return nb::make_tuple(ch, db);
}

// ---- gap statistic ---------------------------------------------------------------------------------------------

// The sums of squares of four pairs (x[r], y[r]) side by side, each in pair_scalar's order (sqd4 without its finish)
inline void sqsum4(const double* const* x, const double* const* y, int64_t p, double* out) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int64_t b = 0; b < p; b += spatial::kBlock) {
        double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
        const int64_t e = std::min(p, b + spatial::kBlock);
        for (int64_t d = b; d < e; ++d) {
            const double z0 = x[0][d] - y[0][d], z1 = x[1][d] - y[1][d], z2 = x[2][d] - y[2][d], z3 = x[3][d] - y[3][d];
            t0 = std::fma(z0, z0, t0), t1 = std::fma(z1, z1, t1), t2 = std::fma(z2, z2, t2), t3 = std::fma(z3, z3, t3);
        }
        if (b == 0) s0 = t0, s1 = t1, s2 = t2, s3 = t3;
        else s0 = s0 + t0, s1 = s1 + t1, s2 = s2 + t2, s3 = s3 + t3;
    }
    out[0] = s0, out[1] = s1, out[2] = s2, out[3] = s3;
}

// log W of a partition of rows in key order: W = 1/2 sum_r D_r / n_r over the clusters in first-seen order,
// D_r = sum over i < j in r of d(i, j)^power (spatial's euclidean; rows' sums met by the pairwise tree)
double log_w(const double* X, int64_t n, int64_t p, const int64_t* lab, int64_t K, double power) {
    // on CL43's grid (d scales exactly by 2^e, W by 2^(e power)), the log taken back
    const int e = scale_exponent(max_abs_of(X, n * p));
    std::vector<double> Xs;
    if (e != 0) Xs = scaled_copy(X, n * p, e), X = Xs.data();
    std::vector<int64_t> map(static_cast<size_t>(K), -1);
    int64_t next = 0;
    std::vector<int64_t> code(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        int64_t& v = map[static_cast<size_t>(lab[i])];
        if (v < 0) v = next++;
        code[static_cast<size_t>(i)] = v;
    }
    const Groups G = group(code.data(), n, next);
    std::vector<double> rowsum, part(static_cast<size_t>(next));
    std::vector<double> rows;
    for (int64_t c = 0; c < next; ++c) {
        const int64_t b = G.start[static_cast<size_t>(c)], e = G.start[static_cast<size_t>(c + 1)], m = e - b;
        rows.clear();
        rows.reserve(static_cast<size_t>(rows.size() + m * p));
        for (int64_t r = b; r < e; ++r) {
            const double* x = X + G.member[static_cast<size_t>(r)] * p;
            rows.insert(rows.end(), x, x + p);
        }
        rowsum.assign(static_cast<size_t>(std::max<int64_t>(m - 1, 0)), 0.0);
        for (int64_t i = 0; i + 1 < m; ++i) {
            const double* x = rows.data() + i * p;
            double acc[4] = {0.0, 0.0, 0.0, 0.0};
            int64_t q = 0;
            for (int64_t j = i + 1; j < m; j += 4, ++q) {
                const int R = static_cast<int>(std::min<int64_t>(4, m - j));
                const double* xs[4] = {x, x, x, x};
                const double* ys[4];
                for (int r = 0; r < 4; ++r) ys[r] = rows.data() + (j + std::min(r, R - 1)) * p;
                double s4[4];
                sqsum4(xs, ys, p, s4);  // pair_scalar's sums side by side; the finish gives spatial's euclidean
                for (int r = 0; r < R; ++r) {
                    double d = spatial::euclid_finish<false>(s4[r], x, ys[r], nullptr, p);
                    d = power == 1.0 ? d : power == 2.0 ? d * d : std::pow(d, power);
                    acc[r] = q == 0 ? d : acc[r] + d;
                }
            }
            rowsum[static_cast<size_t>(i)] = meet4(acc, m - 1 - i);
        }
        part[static_cast<size_t>(c)] = psum(rowsum.data(), m - 1) / static_cast<double>(m);
    }
    double W = 0.0;
    for (int64_t c = 0; c < next; ++c) W = W + part[static_cast<size_t>(c)];
    const double lw = std::log(0.5 * W);
    if (!e) return lw;
    // lw - t log 2 with log 2 as hi + lo and the product inside an fma (one rounding of the result)
    const double t = static_cast<double>(e) * power;
    return std::fma(-t, 0x1.62e42fefa39efp-1, lw) - t * 0x1.abc9e3b39803fp-56;
}

// The eigenvectors of a symmetric p x p matrix by cyclic Jacobi (columns of V by decreasing eigenvalue, the index
// on a tie; each column's largest |component| made positive, the first on a tie)
std::vector<double> jacobi_vectors(std::vector<double> C, int64_t p) {
    std::vector<double> V(static_cast<size_t>(p * p), 0.0);
    for (int64_t i = 0; i < p; ++i) V[static_cast<size_t>(i * p + i)] = 1.0;
    auto at = [&](int64_t i, int64_t j) -> double& { return C[static_cast<size_t>(i * p + j)]; };
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0, diag = 0.0;
        for (int64_t i = 0; i < p; ++i) {
            diag += at(i, i) * at(i, i);
            for (int64_t j = i + 1; j < p; ++j) off += at(i, j) * at(i, j);
        }
        if (!(off > 0x1p-104 * diag)) break;
        for (int64_t a = 0; a < p; ++a)
            for (int64_t b = a + 1; b < p; ++b) {
                const double apq = at(a, b);
                if (apq == 0.0) continue;
                const double theta = (at(b, b) - at(a, a)) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int64_t k = 0; k < p; ++k) {  // columns a, b
                    const double x = at(k, a), y = at(k, b);
                    at(k, a) = c * x - s * y, at(k, b) = s * x + c * y;
                }
                for (int64_t k = 0; k < p; ++k) {  // rows a, b
                    const double x = at(a, k), y = at(b, k);
                    at(a, k) = c * x - s * y, at(b, k) = s * x + c * y;
                }
                for (int64_t k = 0; k < p; ++k) {
                    double& x = V[static_cast<size_t>(k * p + a)];
                    double& y = V[static_cast<size_t>(k * p + b)];
                    const double vx = x, vy = y;
                    x = c * vx - s * vy, y = s * vx + c * vy;
                }
            }
    }
    std::vector<int64_t> idx(static_cast<size_t>(p));
    std::iota(idx.begin(), idx.end(), int64_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t i, int64_t j) { return at(i, i) > at(j, j); });
    std::vector<double> out(static_cast<size_t>(p * p));
    for (int64_t q = 0; q < p; ++q) {
        const int64_t col = idx[static_cast<size_t>(q)];
        int64_t big = 0;
        for (int64_t k = 1; k < p; ++k)
            if (std::fabs(V[static_cast<size_t>(k * p + col)]) > std::fabs(V[static_cast<size_t>(big * p + col)]))
                big = k;
        const double sg = V[static_cast<size_t>(big * p + col)] < 0 ? -1.0 : 1.0;
        for (int64_t k = 0; k < p; ++k) out[static_cast<size_t>(k * p + q)] = sg * V[static_cast<size_t>(k * p + col)];
    }
    return out;
}

int64_t distinct_rows_key(const double* X, int64_t n, int64_t p) {
    int64_t d = n > 0 ? 1 : 0;
    for (int64_t i = 1; i < n; ++i)
        if (!std::equal(X + i * p, X + (i + 1) * p, X + (i - 1) * p)) ++d;
    return d;
}

constexpr std::uint64_t kSubStart = std::uint64_t{1} << 23;

std::pair<int64_t, int64_t> subkey_of(int64_t key0, int64_t key1, std::uint64_t r) {
    random::Stream st(key_of(key0, key1));
    return {static_cast<int64_t>(st.first(element(r, 0))), static_cast<int64_t>(st.first(element(r, 1)))};
}

nb::object gap_py(
    nb::handle Xo,
    nb::handle ordo,
    int64_t k_max,
    int64_t B,
    double power,
    int reference,
    int method,
    int64_t key0,
    int64_t key1
) {
    int64_t n, p, m;
    const double* X0 = matrix_arg(Xo, "X", n, p);
    const int64_t* order = index_arg(ordo, "order", m);
    if (m != n) throw nb::value_error("order: one index a row");
    if (k_max < 1 || B < 1) throw nb::value_error("k_max, B");
    std::vector<double> lw(static_cast<size_t>((B + 1) * k_max));
    {
        const parallel::Unlocked unlocked(true);
        std::vector<double> X(static_cast<size_t>(n * p));
        for (int64_t r = 0; r < n; ++r) std::copy(X0 + order[r] * p, X0 + (order[r] + 1) * p, X.begin() + r * p);
        // the centre (shifted by the first point, CL48 chunks), the axes and the box
        std::vector<double> mean(static_cast<size_t>(p));
        pairwise_chunks(
            chunks_of(n), p,
            [&](int64_t c, double* o) {
                std::fill(o, o + p, 0.0);
                for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i)
                    for (int64_t d = 0; d < p; ++d)
                        o[d] = o[d] + (X[static_cast<size_t>(i * p + d)] - X[static_cast<size_t>(d)]);
            },
            mean.data(), kChunk * p
        );
        for (int64_t d = 0; d < p; ++d)
            mean[static_cast<size_t>(d)] =
                X[static_cast<size_t>(d)] + mean[static_cast<size_t>(d)] / static_cast<double>(n);
        std::vector<double> Xc(static_cast<size_t>(n * p));
        for (int64_t i = 0; i < n; ++i)
            for (int64_t d = 0; d < p; ++d)
                Xc[static_cast<size_t>(i * p + d)] = X[static_cast<size_t>(i * p + d)] - mean[static_cast<size_t>(d)];
        std::vector<double> V;
        if (reference == 0) {
            std::vector<double> C(static_cast<size_t>(p * p));
            pairwise_chunks(
                chunks_of(n), p * p,
                [&](int64_t c, double* o) {
                    std::fill(o, o + p * p, 0.0);
                    for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) {
                        const double* z = Xc.data() + i * p;
                        for (int64_t a = 0; a < p; ++a)
                            for (int64_t b = 0; b < p; ++b) o[a * p + b] = o[a * p + b] + z[a] * z[b];
                    }
                },
                C.data(), kChunk * p * p
            );
            V = jacobi_vectors(C, p);
            std::vector<double> rot(static_cast<size_t>(n * p));
            for (int64_t i = 0; i < n; ++i)
                for (int64_t q = 0; q < p; ++q) {
                    double s = 0.0;
                    for (int64_t e = 0; e < p; ++e)
                        s = s + Xc[static_cast<size_t>(i * p + e)] * V[static_cast<size_t>(e * p + q)];
                    rot[static_cast<size_t>(i * p + q)] = s;
                }
            Xc.swap(rot);
        }
        std::vector<double> lo(static_cast<size_t>(p), kInfD), hi(static_cast<size_t>(p), -kInfD);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t d = 0; d < p; ++d) {
                const double v = Xc[static_cast<size_t>(i * p + d)];
                lo[static_cast<size_t>(d)] = std::min(lo[static_cast<size_t>(d)], v);
                hi[static_cast<size_t>(d)] = std::max(hi[static_cast<size_t>(d)], v);
            }
        auto body = [&](int64_t s) {
            std::vector<double> Zk;
            const double* Z = X.data();
            if (s > 0) {
                random::Stream st(key_of(key0, key1));
                const std::uint64_t r = static_cast<std::uint64_t>(s - 1);
                std::vector<double> z1(static_cast<size_t>(p)), Zg(static_cast<size_t>(n * p));
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t d = 0; d < p; ++d) {
                        const double u = uniform(st, element(r, static_cast<std::uint64_t>(i * p + d)));
                        z1[static_cast<size_t>(d)] =
                            lo[static_cast<size_t>(d)] + (hi[static_cast<size_t>(d)] - lo[static_cast<size_t>(d)]) * u;
                    }
                    for (int64_t d = 0; d < p; ++d) {
                        double v;
                        if (reference == 0) {
                            v = 0.0;
                            for (int64_t e = 0; e < p; ++e)
                                v = v + z1[static_cast<size_t>(e)] * V[static_cast<size_t>(d * p + e)];
                        } else {
                            v = z1[static_cast<size_t>(d)];
                        }
                        Zg[static_cast<size_t>(i * p + d)] = v + mean[static_cast<size_t>(d)];
                    }
                }
                const std::vector<int64_t> oz = lex_order(Zg.data(), n, p);
                Zk.resize(static_cast<size_t>(n * p));
                for (int64_t i = 0; i < n; ++i)
                    std::copy(
                        Zg.begin() + oz[static_cast<size_t>(i)] * p, Zg.begin() + (oz[static_cast<size_t>(i)] + 1) * p,
                        Zk.begin() + i * p
                    );
                Z = Zk.data();
                if (distinct_rows_key(Z, n, p) < k_max)
                    throw nb::value_error("a reference set has fewer distinct points than k_max");
            }
            std::vector<int64_t> lab(static_cast<size_t>(n), 0);
            for (int64_t k = 1; k <= k_max; ++k) {
                if (k > 1) {
                    if (method == 0) {
                        const auto [a, b] =
                            subkey_of(key0, key1, kSubStart + static_cast<std::uint64_t>(s * k_max + k));
                        lab = kmeans_labels(Z, n, p, k, key_of(a, b));
                    } else {
                        lab = pam_labels(Z, n, p, k);
                    }
                }
                lw[static_cast<size_t>(s * k_max + k - 1)] = log_w(Z, n, p, lab.data(), k, power);
            }
        };
        parallel::parallel_for(B + 1, body, n * n * p);
    }
    std::vector<double> e(static_cast<size_t>(k_max)), se(static_cast<size_t>(k_max)),
        first(lw.begin(), lw.begin() + k_max);
    std::vector<double> ref(lw.begin() + k_max, lw.end());
    for (int64_t k = 0; k < k_max; ++k) {
        double s = 0.0;
        for (int64_t b = 0; b < B; ++b) s = s + ref[static_cast<size_t>(b * k_max + k)];
        const double mu = s / static_cast<double>(B);
        double v = 0.0;
        for (int64_t b = 0; b < B; ++b) {
            const double z = ref[static_cast<size_t>(b * k_max + k)] - mu;
            v = v + z * z;
        }
        e[static_cast<size_t>(k)] = mu;
        se[static_cast<size_t>(k)] =
            B > 1 ? std::sqrt((1.0 + 1.0 / static_cast<double>(B)) * (v / static_cast<double>(B - 1))) : kNaN;
    }
    return nb::make_tuple(
        wrap(float_array(first, k_max)), wrap(float_array(e, k_max)), wrap(float_array(se, k_max)),
        wrap(float_array(ref, B, k_max))
    );
}

double gap_logw_py(nb::handle Xo, nb::handle co, int64_t K, double power) {
    int64_t n, p, m;
    const double* X = matrix_arg(Xo, "X", n, p);
    const int64_t* codes = index_arg(co, "codes", m);
    if (m != n) throw nb::value_error("codes: one a row");
    for (int64_t i = 0; i < n; ++i)
        if (codes[i] < 0 || codes[i] >= K) throw nb::value_error("codes out of range");
    const parallel::Unlocked unlocked(n * n * p >= kSplitWork);
    return log_w(X, n, p, codes, K, power);
}

// ---- Gower's distance ------------------------------------------------------------------------------------------

nb::object gower_py(nb::handle Vo, nb::handle to, nb::handle so, nb::handle wo) {
    int64_t n, p, m;
    const double* V = matrix_arg(Vo, "V", n, p);
    const int64_t* type = index_arg(to, "types", m);
    if (m != p) throw nb::value_error("types: one a column");
    const double* scale = vector_arg(so, "scale", m);
    if (m != p) throw nb::value_error("scale: one a column");
    const double* w = vector_arg(wo, "weights", m);
    if (m != p) throw nb::value_error("weights: one a column");
    const int64_t total = n * (n - 1) / 2;
    std::vector<double> out(static_cast<size_t>(total));
    {
        const parallel::Unlocked unlocked(total * p >= kSplitWork);
        constexpr int64_t kRows = 16;
        const int64_t nt = (n + kRows - 1) / kRows;
        auto body = [&](int64_t t) {
            for (int64_t i = t * kRows; i < std::min(n, (t + 1) * kRows); ++i) {
                const double* x = V + i * p;
                double* o = out.data() + (i < n - 1 ? cidx(n, i, i + 1) : 0);
                for (int64_t j = i + 1; j < n; ++j) {
                    const double* y = V + j * p;
                    double num = 0.0, den = 0.0;
                    for (int64_t q = 0; q < p; ++q) {
                        const double a = x[q], b = y[q];
                        if (std::isnan(a) || std::isnan(b)) continue;
                        const double wq = w[q];
                        switch (type[q]) {
                            case 0:
                                den = den + wq;
                                num = num + wq * (std::fabs(a - b) / scale[q]);
                                break;
                            case 3:
                                if (a == 0.0 && b == 0.0) break;
                                den = den + wq;
                                if (a != b) num = num + wq;
                                break;
                            default:  // nominal, symmetric binary
                                den = den + wq;
                                if (a != b) num = num + wq;
                        }
                    }
                    o[j - i - 1] = den > 0.0 ? num / den : kNaN;
                }
            }
        };
        if (nt > 1 && total * p >= kSplitWork) parallel::parallel_for(nt, body, kRows * n * p);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
    }
    return wrap(float_array(out, total));
}

nb::object subkey_py(int64_t key0, int64_t key1, int64_t r) {
    const auto [a, b] = subkey_of(key0, key1, static_cast<std::uint64_t>(r));
    return nb::make_tuple(a, b);
}

}  // namespace

void bind_metrics(nb::module_& m) {
    m.def("contingency", &contingency_py, nb::arg("a"), nb::arg("b"), nb::arg("ka"), nb::arg("kb"));
    m.def("external", &external_py, nb::arg("a"), nb::arg("b"), nb::arg("ka"), nb::arg("kb"), nb::arg("emi"));
    m.def(
        "silhouette", &silhouette_py, nb::arg("X").none(), nb::arg("D").none(), nb::arg("codes"), nb::arg("K"),
        nb::arg("metric")
    );
    m.def("ch_db", &ch_db_py, nb::arg("X"), nb::arg("codes"), nb::arg("K"));
    m.def(
        "gap", &gap_py, nb::arg("X"), nb::arg("order"), nb::arg("k_max"), nb::arg("B"), nb::arg("d_power"),
        nb::arg("reference"), nb::arg("method"), nb::arg("key0"), nb::arg("key1")
    );
    m.def("gap_logw", &gap_logw_py, nb::arg("X"), nb::arg("codes"), nb::arg("K"), nb::arg("d_power"));
    m.def("gower", &gower_py, nb::arg("V"), nb::arg("types"), nb::arg("scale"), nb::arg("weights"));
    m.def("subkey", &subkey_py, nb::arg("key0"), nb::arg("key1"), nb::arg("r"));
}

}  // namespace sd::cluster
