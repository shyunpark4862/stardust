// The two candidates of sd.graph's dense all-pairs path (docs/design/graph.md §3.2 FW, §9 item 2), timed
// by tools/graph_probe.py (section fw). Built with sd's flags (-O3 -mcpu=native -ffp-contract=off).
//
//   dijkstra  per-source Dijkstra on the dense matrix with no heap: each step relaxes the popped node's
//             row and finds the next node in the same pass (NEON, ties to the smallest index); one task
//             per source, sources handed out by an atomic counter. Its distances are the left-to-right
//             sums' least fixed point (the definition of §3.2)
//   fw        Floyd-Warshall in 64 x 64 blocks: the diagonal block, then its row and column, then the
//             rest as a register-blocked min-plus product (4 x 8 tiles); the threads split the blocks
//             of each phase and meet at a barrier. Its sums join paths in the middle (d(i,k) + d(k,j))
//
//   graph_probe_fw N THREADS DENSITY SEED REPS
//
// prints one line per candidate: the best time of REPS runs in ms, and for dijkstra the share of
// entries whose bits differ from fw and the largest relative difference; checks dijkstra's rows
// against a binary-heap Dijkstra (the same bits) on a few sources.
#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr int B = 64;  // the FW block

// ---------------------------------------------------------------- per-source dense Dijkstra

// One source: d (n) gets the distances. key holds the tentative distance of an unvisited node and +inf
// for a visited one; bias is 0 for unvisited and +inf for visited, so a relaxed sum never lowers a
// visited key. The relax of row u and the search for the next (key, index) minimum are one pass.
void dijkstra_one(const double* W, std::int64_t n, std::int64_t s, double* d, double* key, double* bias) {
    for (std::int64_t j = 0; j < n; ++j) {
        key[j] = INF;
        bias[j] = 0.0;
        d[j] = INF;
    }
    key[s] = 0.0;
    std::int64_t u = s;
    double du = 0.0;
    const std::int64_t n8 = n & ~std::int64_t(7);
    for (std::int64_t step = 0; step < n; ++step) {
        d[u] = du;
        key[u] = INF;
        bias[u] = INF;
        const double* row = W + u * n;
        const float64x2_t vdu = vdupq_n_f64(du);
        float64x2_t m0 = vdupq_n_f64(INF), m1 = m0, m2 = m0, m3 = m0;
        // per-lane index of the minimum: lanes hold j and j + 1 of each pair
        uint64x2_t i0 = vdupq_n_u64(0), i1 = i0, i2 = i0, i3 = i0;
        uint64x2_t cur = {0, 1};
        const uint64x2_t two = vdupq_n_u64(2), eight = vdupq_n_u64(8);
        std::int64_t j = 0;
        for (; j < n8; j += 8) {
            float64x2_t w0 = vld1q_f64(row + j), w1 = vld1q_f64(row + j + 2);
            float64x2_t w2 = vld1q_f64(row + j + 4), w3 = vld1q_f64(row + j + 6);
            float64x2_t b0 = vld1q_f64(bias + j), b1 = vld1q_f64(bias + j + 2);
            float64x2_t b2 = vld1q_f64(bias + j + 4), b3 = vld1q_f64(bias + j + 6);
            float64x2_t k0 = vld1q_f64(key + j), k1 = vld1q_f64(key + j + 2);
            float64x2_t k2 = vld1q_f64(key + j + 4), k3 = vld1q_f64(key + j + 6);
            k0 = vminq_f64(k0, vaddq_f64(vaddq_f64(vdu, w0), b0));
            k1 = vminq_f64(k1, vaddq_f64(vaddq_f64(vdu, w1), b1));
            k2 = vminq_f64(k2, vaddq_f64(vaddq_f64(vdu, w2), b2));
            k3 = vminq_f64(k3, vaddq_f64(vaddq_f64(vdu, w3), b3));
            vst1q_f64(key + j, k0);
            vst1q_f64(key + j + 2, k1);
            vst1q_f64(key + j + 4, k2);
            vst1q_f64(key + j + 6, k3);
            uint64x2_t c1 = vaddq_u64(cur, two), c2 = vaddq_u64(c1, two), c3 = vaddq_u64(c2, two);
            uint64x2_t l0 = vcltq_f64(k0, m0), l1 = vcltq_f64(k1, m1);
            uint64x2_t l2 = vcltq_f64(k2, m2), l3 = vcltq_f64(k3, m3);
            m0 = vbslq_f64(l0, k0, m0);
            m1 = vbslq_f64(l1, k1, m1);
            m2 = vbslq_f64(l2, k2, m2);
            m3 = vbslq_f64(l3, k3, m3);
            i0 = vbslq_u64(l0, cur, i0);
            i1 = vbslq_u64(l1, c1, i1);
            i2 = vbslq_u64(l2, c2, i2);
            i3 = vbslq_u64(l3, c3, i3);
            cur = vaddq_u64(cur, eight);
        }
        // reduce the eight lanes: the smallest key, then the smallest index
        double best = INF;
        std::int64_t arg = -1;
        auto take = [&](float64x2_t m, uint64x2_t ix) {
            for (int l = 0; l < 2; ++l) {
                double v = l ? vgetq_lane_f64(m, 1) : vgetq_lane_f64(m, 0);
                auto x = static_cast<std::int64_t>(l ? vgetq_lane_u64(ix, 1) : vgetq_lane_u64(ix, 0));
                if (v < best || (v == best && v < INF && x < arg)) {
                    best = v;
                    arg = x;
                }
            }
        };
        take(m0, i0);
        take(m1, i1);
        take(m2, i2);
        take(m3, i3);
        for (; j < n; ++j) {
            double k = std::min(key[j], (du + row[j]) + bias[j]);
            key[j] = k;
            if (k < best || (k == best && k < INF && j < arg)) {
                best = k;
                arg = j;
            }
        }
        if (best == INF) break;  // the rest is unreachable
        u = arg;
        du = best;
    }
}

// The same with NaN for a visited key: FMIN keeps a NaN, so the relax min(key, du + w) leaves a visited
// node alone with no mask, and FMINNM, which passes over a NaN, gives the block minima. The pass keeps
// the minimum of each block of 64 keys; the next node is the first index of the first block holding the
// smallest key. The keys and their order are those of dijkstra_one.
void dijkstra_one_nan(const double* W, std::int64_t n, std::int64_t s, double* d, double* key) {
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    for (std::int64_t j = 0; j < n; ++j) {
        key[j] = INF;
        d[j] = INF;
    }
    key[s] = 0.0;
    std::int64_t u = s;
    double du = 0.0;
    for (std::int64_t step = 0; step < n; ++step) {
        d[u] = du;
        key[u] = NaN;
        const double* row = W + u * n;
        const float64x2_t vdu = vdupq_n_f64(du);
        double best = INF;
        std::int64_t best_block = -1;
        std::int64_t j = 0;
        for (; j + 64 <= n; j += 64) {
            float64x2_t m0 = vdupq_n_f64(INF), m1 = m0, m2 = m0, m3 = m0;
            for (int q = 0; q < 64; q += 8) {
                const std::int64_t o = j + q;
                float64x2_t k0 = vminq_f64(vld1q_f64(key + o), vaddq_f64(vdu, vld1q_f64(row + o)));
                float64x2_t k1 = vminq_f64(vld1q_f64(key + o + 2), vaddq_f64(vdu, vld1q_f64(row + o + 2)));
                float64x2_t k2 = vminq_f64(vld1q_f64(key + o + 4), vaddq_f64(vdu, vld1q_f64(row + o + 4)));
                float64x2_t k3 = vminq_f64(vld1q_f64(key + o + 6), vaddq_f64(vdu, vld1q_f64(row + o + 6)));
                vst1q_f64(key + o, k0);
                vst1q_f64(key + o + 2, k1);
                vst1q_f64(key + o + 4, k2);
                vst1q_f64(key + o + 6, k3);
                m0 = vminnmq_f64(m0, k0);
                m1 = vminnmq_f64(m1, k1);
                m2 = vminnmq_f64(m2, k2);
                m3 = vminnmq_f64(m3, k3);
            }
            double bm = vminnmvq_f64(vminnmq_f64(vminnmq_f64(m0, m1), vminnmq_f64(m2, m3)));
            if (bm < best) {
                best = bm;
                best_block = j;
            }
        }
        std::int64_t arg = -1;
        for (std::int64_t t = j; t < n; ++t) {  // the tail
            double k = std::min(key[t], du + row[t]);
            if (std::isnan(key[t])) k = key[t];
            key[t] = k;
            if (k < best) {
                best = k;
                best_block = -1;
                arg = t;
            }
        }
        if (best == INF) break;
        if (best_block >= 0)
            for (arg = best_block; !(key[arg] == best); ++arg) {}
        u = arg;
        du = best;
    }
}

void dijkstra_all(const double* W, std::int64_t n, double* D, int threads, bool nan_form) {
    std::atomic<std::int64_t> next{0};
    auto work = [&] {
        std::vector<double> key(n), bias(n);
        for (std::int64_t s; (s = next.fetch_add(1)) < n;) {
            if (nan_form) dijkstra_one_nan(W, n, s, D + s * n, key.data());
            else dijkstra_one(W, n, s, D + s * n, key.data(), bias.data());
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
}

// the check: a binary-heap Dijkstra of one source, ties by (distance, index)
std::vector<double> heap_dijkstra(const double* W, std::int64_t n, std::int64_t s) {
    std::vector<double> d(n, INF);
    std::vector<char> done(n, 0);
    using E = std::pair<double, std::int64_t>;
    std::priority_queue<E, std::vector<E>, std::greater<E>> q;
    d[s] = 0.0;
    q.push({0.0, s});
    while (!q.empty()) {
        auto [du, u] = q.top();
        q.pop();
        if (done[u]) continue;
        done[u] = 1;
        for (std::int64_t v = 0; v < n; ++v) {
            double w = W[u * n + v];
            if (v == u || w == INF) continue;
            double t = du + w;
            if (t < d[v]) {
                d[v] = t;
                q.push({t, v});
            }
        }
    }
    return d;
}

// ---------------------------------------------------------------- blocked Floyd-Warshall

// a block of D is D + bi * B * n + bj * B (row stride n); n is a multiple of B here
inline double* blk(double* D, std::int64_t n, std::int64_t bi, std::int64_t bj) { return D + bi * B * n + bj * B; }

// C = min(C, A (x) Bm) where A, Bm may alias C (the diagonal and the row/column phases): k in order
void fw_dependent(double* C, const double* A, const double* Bm, std::int64_t n) {
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < B; ++i) {
            const float64x2_t a = vdupq_n_f64(A[i * n + k]);
            double* c = C + i * n;
            const double* b = Bm + k * n;
            for (int j = 0; j < B; j += 2)
                vst1q_f64(c + j, vminq_f64(vld1q_f64(c + j), vaddq_f64(a, vld1q_f64(b + j))));
        }
    }
}

// C = min(C, A (x) Bm) with C apart from A and Bm: 4 x 8 register tiles
void fw_independent(double* C, const double* A, const double* Bm, std::int64_t n) {
    for (int i = 0; i < B; i += 4) {
        for (int j = 0; j < B; j += 8) {
            float64x2_t c[4][4];
            for (int r = 0; r < 4; ++r)
                for (int q = 0; q < 4; ++q) c[r][q] = vld1q_f64(C + (i + r) * n + j + 2 * q);
            for (int k = 0; k < B; ++k) {
                const double* b = Bm + k * n + j;
                float64x2_t b0 = vld1q_f64(b), b1 = vld1q_f64(b + 2), b2 = vld1q_f64(b + 4), b3 = vld1q_f64(b + 6);
                for (int r = 0; r < 4; ++r) {
                    float64x2_t a = vdupq_n_f64(A[(i + r) * n + k]);
                    c[r][0] = vminq_f64(c[r][0], vaddq_f64(a, b0));
                    c[r][1] = vminq_f64(c[r][1], vaddq_f64(a, b1));
                    c[r][2] = vminq_f64(c[r][2], vaddq_f64(a, b2));
                    c[r][3] = vminq_f64(c[r][3], vaddq_f64(a, b3));
                }
            }
            for (int r = 0; r < 4; ++r)
                for (int q = 0; q < 4; ++q) vst1q_f64(C + (i + r) * n + j + 2 * q, c[r][q]);
        }
    }
}

// D (n x n, n a multiple of B) holds W with 0 on the diagonal; becomes the distances
void fw_all(double* D, std::int64_t n, int threads) {
    const std::int64_t nb = n / B;
    std::barrier sync(threads);
    auto work = [&](int t) {
        for (std::int64_t k = 0; k < nb; ++k) {
            double* dkk = blk(D, n, k, k);
            if (t == 0) fw_dependent(dkk, dkk, dkk, n);
            sync.arrive_and_wait();
            // the row and the column of block k: 2 (nb - 1) blocks
            for (std::int64_t x = t; x < 2 * (nb - 1); x += threads) {
                std::int64_t m = x % (nb - 1);
                m += (m >= k);
                if (x < nb - 1) {
                    double* c = blk(D, n, k, m);
                    fw_dependent(c, dkk, c, n);
                } else {
                    double* c = blk(D, n, m, k);
                    fw_dependent(c, c, dkk, n);
                }
            }
            sync.arrive_and_wait();
            for (std::int64_t x = t; x < (nb - 1) * (nb - 1); x += threads) {
                std::int64_t i = x / (nb - 1), j = x % (nb - 1);
                i += (i >= k);
                j += (j >= k);
                fw_independent(blk(D, n, i, j), blk(D, n, i, k), blk(D, n, k, j), n);
            }
            sync.arrive_and_wait();
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& t : pool) t.join();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: graph_probe_fw N THREADS DENSITY SEED REPS\n");
        return 2;
    }
    const std::int64_t n = std::atoll(argv[1]);
    const int threads = std::atoi(argv[2]);
    const double density = std::atof(argv[3]);
    const std::uint64_t seed = std::strtoull(argv[4], nullptr, 10);
    const int reps = std::atoi(argv[5]);
    if (n % B) {
        std::fprintf(stderr, "N must be a multiple of %d\n", B);
        return 2;
    }
    // the graph: each off-diagonal entry is an edge with probability `density`, weight in [0.01, 1.01)
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> wd(0.01, 1.01), pd(0.0, 1.0);
    std::vector<double> W(static_cast<std::size_t>(n * n), INF);
    for (std::int64_t i = 0; i < n; ++i)
        for (std::int64_t j = 0; j < n; ++j)
            if (i != j && pd(g) < density) W[i * n + j] = wd(g);
    std::vector<double> Wd = W;  // the Dijkstra form never reads the diagonal's own row entry as an edge
    for (std::int64_t i = 0; i < n; ++i) Wd[i * n + i] = INF;
    std::vector<double> Dd(static_cast<std::size_t>(n * n)), Df(static_cast<std::size_t>(n * n));

    std::vector<double> Dn(static_cast<std::size_t>(n * n));
    double best_d = INF, best_n = INF, best_f = INF;
    auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        dijkstra_all(Wd.data(), n, Dd.data(), threads, false);
        auto t1 = Clock::now();
        dijkstra_all(Wd.data(), n, Dn.data(), threads, true);
        auto t2 = Clock::now();
        Df = W;
        for (std::int64_t i = 0; i < n; ++i) Df[i * n + i] = 0.0;
        auto t3 = Clock::now();
        fw_all(Df.data(), n, threads);
        auto t4 = Clock::now();
        best_d = std::min(best_d, ms(t0, t1));
        best_n = std::min(best_n, ms(t1, t2));
        best_f = std::min(best_f, ms(t3, t4));
    }
    std::int64_t differ = 0;
    double worst = 0.0;
    for (std::int64_t k = 0; k < n * n; ++k) {
        if (std::memcmp(&Dd[k], &Df[k], 8) != 0) {
            ++differ;
            if (std::isfinite(Dd[k]) && Df[k] > 0) worst = std::max(worst, std::fabs(Dd[k] - Df[k]) / Df[k]);
        }
    }
    const bool forms_equal = std::memcmp(Dd.data(), Dn.data(), static_cast<std::size_t>(n * n) * 8) == 0;
    int heap_bad = forms_equal ? 0 : 1;
    for (std::int64_t s : {std::int64_t(0), n / 3, n - 1}) {
        auto h = heap_dijkstra(Wd.data(), n, s);
        if (std::memcmp(h.data(), Dd.data() + s * n, static_cast<std::size_t>(n) * 8) != 0) ++heap_bad;
    }
    std::printf(
        "n=%lld threads=%d density=%.3f dijkstra_mask_ms=%.3f dijkstra_nan_ms=%.3f fw_ms=%.3f "
        "ratio_mask=%.3f ratio_nan=%.3f differ=%.6f worst_rel=%.3e mismatch=%d\n",
        static_cast<long long>(n), threads, density, best_d, best_n, best_f, best_d / best_f, best_n / best_f,
        static_cast<double>(differ) / static_cast<double>(n * n), worst, heap_bad
    );
    return heap_bad ? 1 : 0;
}
