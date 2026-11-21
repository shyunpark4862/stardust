// Resampling and Monte Carlo (milestone H9, docs/design/htest.md §3.10).
//
//   perm_resamples   permutation_test's resamples start .. start + count of k samples (lanes,
//                    n_s): 'independent' (the pooled observations dealt out again), 'pairings'
//                    (each sample's order permuted) or 'samples' (each observation's values
//                    permuted between the samples); every permutation in scipy's order (exact)
//                    or random ones
//   boot_resamples   bootstrap's resamples (with replacement; paired: one index set for all)
//   jackknife        the leave-one-out samples of BCa's acceleration
//   resample_pvalue  the share of a null distribution at least as extreme as the observed
//                    statistic, scipy's rule (a relative tolerance of 100 eps; +1 in the numerator
//                    and the denominator unless the null distribution is exact)
//   lane_quantiles   numpy's 'linear' quantiles of each lane (bootstrap's intervals)
//   score_percentile scipy's percentile of score (BCa's bias correction)
//   bws              the Baumgartner-Weiss-Schindler statistic of (ranked) samples
//   table_stat       chi2_contingency's Pearson statistic or the probability of a table
//   table_null       the same of tables drawn under independence given the margins (or every
//                    permutation of the untabulated rows' labels)
//   euclidean        the distance matrix of points (rows)
//   mgc              multiscale_graphcorr (htest/mgc.hpp)
//   gof_fit          goodness_of_fit's fits and statistics (htest/mle.hpp)
//
// The random resamples come from sd.random's Philox stream at (key, position): resample j's
// draw t is element position + j D + t of the stream (D the draws of one resample), so a
// resample is a function of (key, position, j) alone: the same bits on any number of threads
// and whatever the batches (docs/design/random.md §3). A uniform index below r is Lemire's
// bounded draw (random::bounded, its rejections the element's extra words).
#include "htest/resampling.hpp"

#include "array/ndarray.hpp"
#include "array/ops/pieces.hpp"
#include "bind/pyutil.hpp"
#include "htest/lanes.hpp"
#include "htest/mgc.hpp"
#include "htest/mle.hpp"
#include "htest/moments.hpp"
#include "random/philox.hpp"
#include "stats/dist/discrete_more.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

namespace m = stats::method;
using u64 = std::uint64_t;
using u128 = unsigned __int128;
using i64 = std::int64_t;

std::vector<Rows> rows_of(nb::list samples) {
    std::vector<Rows> s;
    for (nb::handle h : samples) s.emplace_back(rows_arg(h));
    return s;
}

// C(n, k) for C(n, k) < 2^63 (the multiplicative formula; each partial product is a binomial)
u64 binom(u64 n, u64 k) {
    if (k > n) return 0;
    k = std::min(k, n - k);
    u128 r = 1;
    for (u64 i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return static_cast<u64>(r);
}

u64 factorial(u64 n) {
    u64 f = 1;
    for (u64 i = 2; i <= n; ++i) f *= i;
    return f;
}

// A uniform index below r (r >= 1) from element e of the stream.
inline u64 below(random::Stream& st, u64 e, u64 r) {
    return random::bounded(st.first(e), r, [&](u64 q) { return st.extra(e, q); });
}

// a[0..m) shuffled in place (Fisher-Yates from the top: draw t of element e0 + t)
template <class T>
void shuffle(random::Stream& st, u64 e0, T* a, i64 m) {
    for (i64 t = 0; t + 1 < m; ++t) {
        const i64 i = m - 1 - t;
        const auto r = static_cast<i64>(below(st, e0 + static_cast<u64>(t), static_cast<u64>(i + 1)));
        std::swap(a[i], a[r]);
    }
}

// the permutation of rank r of 0..m-1 in lexicographic order (itertools.permutations' order)
void lehmer(u64 r, i64 m, i64* out, std::vector<i64>& avail) {
    avail.resize(static_cast<std::size_t>(m));
    std::iota(avail.begin(), avail.end(), i64{0});
    for (i64 i = 0; i < m; ++i) {
        const u64 f = factorial(static_cast<u64>(m - 1 - i));
        const auto d = static_cast<std::size_t>(r / f);
        r %= f;
        out[i] = avail[d];
        avail.erase(avail.begin() + static_cast<std::ptrdiff_t>(d));
    }
}

// ---- permutation_test's resamples ----------------------------------------------------------

enum PermType { kIndependent = 0, kPairings = 1, kSamples = 2 };

nb::list perm_resamples(nb::list samples, int ptype, bool exact, u64 k0, u64 k1, u64 pos, i64 start, i64 count) {
    const std::vector<Rows> s = rows_of(samples);
    const auto k = static_cast<i64>(s.size());
    const i64 L = s[0].lanes;
    std::vector<i64> off(static_cast<std::size_t>(k + 1), 0);
    for (i64 i = 0; i < k; ++i)
        off[static_cast<std::size_t>(i + 1)] = off[static_cast<std::size_t>(i)] + s[static_cast<std::size_t>(i)].n;
    const i64 N = off[static_cast<std::size_t>(k)];
    const i64 n = s[0].n;  // pairings and samples: the common length
    std::vector<NDArray> outs;
    std::vector<double*> op;
    for (i64 i = 0; i < k; ++i) {
        const i64 shape[] = {count, L, s[static_cast<std::size_t>(i)].n};
        outs.push_back(new_array(DType::Float64, shape));
        op.push_back(reinterpret_cast<double*>(outs.back().data));
    }
    // the observations of the pooled sample: (sample, index) of pooled position p
    std::vector<i64> who(static_cast<std::size_t>(N)), at(static_cast<std::size_t>(N));
    for (i64 i = 0; i < k; ++i)
        for (i64 t = 0; t < s[static_cast<std::size_t>(i)].n; ++t)
            who[static_cast<std::size_t>(off[static_cast<std::size_t>(i)] + t)] = i,
                                                                            at[static_cast<std::size_t>(
                                                                                off[static_cast<std::size_t>(i)] + t
                                                                            )] = t;
    // 'independent' exact: the arrangements of the samples after sample i (a multinomial)
    std::vector<u64> rest(static_cast<std::size_t>(k), 1);
    if (ptype == kIndependent && exact) {
        u64 r = 1;
        i64 left = s[static_cast<std::size_t>(k - 1)].n;
        for (i64 i = k - 2; i >= 0; --i) {
            rest[static_cast<std::size_t>(i)] = r;
            left += s[static_cast<std::size_t>(i)].n;
            r *= binom(static_cast<u64>(left), static_cast<u64>(s[static_cast<std::size_t>(i)].n));
        }
    }
    const u64 D = ptype == kIndependent ? static_cast<u64>(N - 1)
                  : ptype == kPairings  ? static_cast<u64>(k * (n - 1))
                                        : static_cast<u64>(n * (k - 1));
    const i64 work = std::max<i64>(1, L * N);
    const i64 per = std::max<i64>(1, kParallelTask / work);
    for_pieces(count, per, count * work, [&](i64, i64 b, i64 e) {
        random::Stream st(random::Key{k0, k1});
        std::vector<i64> idx(static_cast<std::size_t>(std::max(N, k * n)));
        std::vector<i64> pool, avail;
        for (i64 j = b; j < e; ++j) {
            const u64 J = static_cast<u64>(start + j);
            const u64 e0 = pos + J * D;
            if (ptype == kIndependent) {
                if (!exact) {
                    std::iota(idx.begin(), idx.begin() + N, i64{0});
                    shuffle(st, e0, idx.data(), N);
                } else {
                    pool.resize(static_cast<std::size_t>(N));
                    std::iota(pool.begin(), pool.end(), i64{0});
                    u64 r = J;
                    for (i64 i = 0; i + 1 < k; ++i) {
                        const u64 ri = r / rest[static_cast<std::size_t>(i)];
                        r %= rest[static_cast<std::size_t>(i)];
                        // the combination of rank ri of size n_i from pool (lexicographic)
                        const i64 ni = s[static_cast<std::size_t>(i)].n;
                        const auto M = static_cast<i64>(pool.size());
                        std::vector<i64> keep;
                        keep.reserve(static_cast<std::size_t>(M - ni));
                        i64 chosen = 0, x = 0;
                        u64 rr = ri;
                        while (x < M) {
                            if (chosen < ni) {
                                const u64 c = binom(static_cast<u64>(M - x - 1), static_cast<u64>(ni - chosen - 1));
                                if (rr < c) {
                                    idx[static_cast<std::size_t>(off[static_cast<std::size_t>(i)] + chosen)] =
                                        pool[static_cast<std::size_t>(x)];
                                    ++chosen, ++x;
                                    continue;
                                }
                                rr -= c;
                            }
                            keep.push_back(pool[static_cast<std::size_t>(x)]);
                            ++x;
                        }
                        pool.swap(keep);
                    }
                    for (std::size_t t = 0; t < pool.size(); ++t)
                        idx[static_cast<std::size_t>(off[static_cast<std::size_t>(k - 1)]) + t] = pool[t];
                }
                for (i64 i = 0; i < k; ++i) {
                    const i64 ni = s[static_cast<std::size_t>(i)].n;
                    double* o = op[static_cast<std::size_t>(i)] + j * L * ni;
                    for (i64 t = 0; t < ni; ++t) {
                        const i64 p = idx[static_cast<std::size_t>(off[static_cast<std::size_t>(i)] + t)];
                        const Rows& src = s[static_cast<std::size_t>(who[static_cast<std::size_t>(p)])];
                        const i64 a = at[static_cast<std::size_t>(p)];
                        for (i64 l = 0; l < L; ++l) o[l * ni + t] = src.at(l, a);
                    }
                }
            } else if (ptype == kPairings) {
                const u64 f = exact ? factorial(static_cast<u64>(n)) : 1;
                u64 r = J;
                for (i64 i = k - 1; i >= 0; --i) {
                    i64* p = idx.data() + i * n;
                    if (exact) {
                        lehmer(r % f, n, p, avail);
                        r /= f;
                    } else {
                        std::iota(p, p + n, i64{0});
                        shuffle(st, e0 + static_cast<u64>(i * (n - 1)), p, n);
                    }
                }
                for (i64 i = 0; i < k; ++i) {
                    double* o = op[static_cast<std::size_t>(i)] + j * L * n;
                    const i64* p = idx.data() + i * n;
                    for (i64 t = 0; t < n; ++t)
                        for (i64 l = 0; l < L; ++l) o[l * n + t] = s[static_cast<std::size_t>(i)].at(l, p[t]);
                }
            } else {
                const u64 f = exact ? factorial(static_cast<u64>(k)) : 1;
                u64 r = J;
                for (i64 t = n - 1; t >= 0; --t) {
                    i64* p = idx.data() + t * k;
                    if (exact) {
                        lehmer(r % f, k, p, avail);
                        r /= f;
                    } else {
                        std::iota(p, p + k, i64{0});
                        shuffle(st, e0 + static_cast<u64>(t * (k - 1)), p, k);
                    }
                }
                for (i64 i = 0; i < k; ++i) {
                    double* o = op[static_cast<std::size_t>(i)] + j * L * n;
                    for (i64 t = 0; t < n; ++t) {
                        const Rows& src = s[static_cast<std::size_t>(idx[static_cast<std::size_t>(t * k + i)])];
                        for (i64 l = 0; l < L; ++l) o[l * n + t] = src.at(l, t);
                    }
                }
            }
        }
    });
    nb::list lst;
    for (NDArray& a : outs) lst.append(wrap(std::move(a)));
    return lst;
}

// ---- bootstrap ------------------------------------------------------------------------------

nb::list boot_resamples(nb::list samples, bool paired, u64 k0, u64 k1, u64 pos, i64 start, i64 count) {
    const std::vector<Rows> s = rows_of(samples);
    const auto k = static_cast<i64>(s.size());
    const i64 L = s[0].lanes;
    std::vector<i64> off(static_cast<std::size_t>(k + 1), 0);
    for (i64 i = 0; i < k; ++i)
        off[static_cast<std::size_t>(i + 1)] = off[static_cast<std::size_t>(i)] + s[static_cast<std::size_t>(i)].n;
    const u64 D = paired ? static_cast<u64>(s[0].n) : static_cast<u64>(off[static_cast<std::size_t>(k)]);
    std::vector<NDArray> outs;
    std::vector<double*> op;
    for (i64 i = 0; i < k; ++i) {
        const i64 shape[] = {L, count, s[static_cast<std::size_t>(i)].n};
        outs.push_back(new_array(DType::Float64, shape));
        op.push_back(reinterpret_cast<double*>(outs.back().data));
    }
    const i64 work = std::max<i64>(1, L * off[static_cast<std::size_t>(k)]);
    const i64 per = std::max<i64>(1, kParallelTask / work);
    for_pieces(count, per, count * work, [&](i64, i64 b, i64 e) {
        random::Stream st(random::Key{k0, k1});
        std::vector<i64> idx;
        for (i64 j = b; j < e; ++j) {
            const u64 e0 = pos + static_cast<u64>(start + j) * D;
            for (i64 i = 0; i < k; ++i) {
                const i64 ni = s[static_cast<std::size_t>(i)].n;
                if (!paired || i == 0) {
                    idx.resize(static_cast<std::size_t>(ni));
                    const u64 base = e0 + (paired ? 0 : static_cast<u64>(off[static_cast<std::size_t>(i)]));
                    for (i64 t = 0; t < ni; ++t)
                        idx[static_cast<std::size_t>(t)] =
                            static_cast<i64>(below(st, base + static_cast<u64>(t), static_cast<u64>(ni)));
                }
                double* o = op[static_cast<std::size_t>(i)];
                for (i64 l = 0; l < L; ++l) {
                    double* row = o + (l * count + j) * ni;
                    for (i64 t = 0; t < ni; ++t)
                        row[t] = s[static_cast<std::size_t>(i)].at(l, idx[static_cast<std::size_t>(t)]);
                }
            }
        }
    });
    nb::list lst;
    for (NDArray& a : outs) lst.append(wrap(std::move(a)));
    return lst;
}

nb::object jackknife(nb::handle x, i64 start, i64 count) {
    const Rows a(rows_arg(x));
    const i64 n = a.n, L = a.lanes;
    const i64 shape[] = {L, count, n - 1};
    NDArray out = new_array(DType::Float64, shape);
    auto* o = reinterpret_cast<double*>(out.data);
    for (i64 l = 0; l < L; ++l)
        for (i64 j = 0; j < count; ++j) {
            const i64 drop = start + j;
            double* row = o + (l * count + j) * (n - 1);
            i64 c = 0;
            for (i64 t = 0; t < n; ++t)
                if (t != drop) row[c++] = a.at(l, t);
        }
    return wrap(std::move(out));
}

// ---- p-values and quantiles --------------------------------------------------------------

nb::object resample_pvalue(nb::handle null, nb::handle obs, int alt, int adjust) {
    const Rows nd(rows_arg(null));  // (R, lanes), or (R, 1): one null distribution for every lane
    const i64 R = nd.lanes, L = require_ndarray(obs).shape[0];
    if (nd.n != L && nd.n != 1) throw nb::value_error("resample_pvalue: the null distribution's lanes");
    const NDArray& ob = vec_arg(obs, L);
    const i64 col = nd.n == 1 ? 0 : 1;
    Outs out(1, L);
    constexpr double kEps100 = 100.0 * 0x1p-52;
    for (i64 l = 0; l < L; ++l) {
        const double o = vec_at(ob, l);
        if (o != o) {
            out.p[0][l] = kNaN;
            continue;
        }
        const double g = std::fabs(kEps100 * o);
        i64 le = 0, ge = 0;
        for (i64 r = 0; r < R; ++r) {
            const double v = nd.at(r, l * col);
            le += v <= o + g;
            ge += v >= o - g;
        }
        const double den = static_cast<double>(R + adjust);
        const double pl = static_cast<double>(le + adjust) / den, pg = static_cast<double>(ge + adjust) / den;
        double p = alt == 1 ? pl : alt == 2 ? pg : 2.0 * std::min(pl, pg);
        out.p[0][l] = std::min(1.0, std::max(0.0, p));
    }
    return wrap(std::move(out.arrays[0]));
}

// numpy's lerp: a + (b - a) t, or b - (b - a)(1 - t) from t = 1/2 on
inline double lerp(double a, double b, double t) {
    const double d = b - a;
    return t >= 0.5 ? b - d * (1.0 - t) : a + d * t;
}

nb::object lane_quantiles(nb::handle dist, nb::handle p) {
    const Rows d(rows_arg(dist)), q(rows_arg(p));
    const i64 L = d.lanes, R = d.n, K = q.n;
    const i64 shape[] = {L, K};
    NDArray out = new_array(DType::Float64, shape);
    auto* o = reinterpret_cast<double*>(out.data);
    each_lane(L, 16 * R, [&](i64 l) {
        std::vector<double> v(static_cast<std::size_t>(R));
        bool nan = false;
        for (i64 r = 0; r < R; ++r) {
            v[static_cast<std::size_t>(r)] = d.at(l, r);
            nan |= v[static_cast<std::size_t>(r)] != v[static_cast<std::size_t>(r)];
        }
        if (!nan) std::sort(v.begin(), v.end());
        for (i64 c = 0; c < K; ++c) {
            const double pc = q.at(l, c);
            if (nan || R == 0 || !(pc >= 0.0 && pc <= 1.0)) {
                o[l * K + c] = kNaN;
                continue;
            }
            const double h = pc * static_cast<double>(R - 1);
            const double fl = std::floor(h);
            const auto j = static_cast<i64>(fl);
            const i64 j1 = std::min(j + 1, R - 1);
            o[l * K + c] = lerp(v[static_cast<std::size_t>(j)], v[static_cast<std::size_t>(j1)], h - fl);
        }
    });
    return wrap(std::move(out));
}

nb::object score_percentile(nb::handle dist, nb::handle score) {
    const Rows d(rows_arg(dist));
    const i64 L = d.lanes, R = d.n;
    const NDArray& sc = vec_arg(score, L);
    Outs out(1, L);
    for (i64 l = 0; l < L; ++l) {
        const double s = vec_at(sc, l);
        if (s != s) {
            out.p[0][l] = kNaN;
            continue;
        }
        i64 c = 0;
        for (i64 r = 0; r < R; ++r) {
            const double v = d.at(l, r);
            c += (v < s) + (v <= s);
        }
        out.p[0][l] = static_cast<double>(c) / static_cast<double>(2 * R);
    }
    return wrap(std::move(out.arrays[0]));
}

// ---- Baumgartner-Weiss-Schindler -------------------------------------------------------------
//
// scipy's _bws_statistic of the (ranked) samples: with R and H sorted,
//   Bx = 1/n sum (R_i - (m + n)/n i)^2 / (i/(n + 1) (1 - i/(n + 1)) m (m + n)/n)
// and By alike; B = (Bx + By)/2 two-sided, (Bx - By)/2 with signed squares one-sided. The sums are
// double-doubles.

double bws_side(std::vector<double>& v, double other, int alt) {
    std::sort(v.begin(), v.end());
    const auto n = static_cast<double>(v.size());
    const double tot = n + other;
    Acc s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double ii = static_cast<double>(i + 1);
        const double num = v[i] - tot / n * ii;
        const double sq = alt == 0 ? num * num : num * std::fabs(num);
        const double q = ii / (n + 1.0);
        s.add(sq / (q * (1.0 - q) * other * tot / n));
    }
    return to_double(s.value()) / n;
}

nb::object bws(nb::handle x, nb::handle y, int alt) {
    const Rows a(rows_arg(x)), b(rows_arg(y));
    Outs out(1, a.lanes);
    each_lane(a.lanes, 16 * (a.n + b.n), [&](i64 l) {
        std::vector<double> u(static_cast<std::size_t>(a.n)), w(static_cast<std::size_t>(b.n));
        bool nan = false;
        for (i64 t = 0; t < a.n; ++t)
            u[static_cast<std::size_t>(t)] = a.at(l, t),
            nan |= u[static_cast<std::size_t>(t)] != u[static_cast<std::size_t>(t)];
        for (i64 t = 0; t < b.n; ++t)
            w[static_cast<std::size_t>(t)] = b.at(l, t),
            nan |= w[static_cast<std::size_t>(t)] != w[static_cast<std::size_t>(t)];
        if (nan) {
            out.p[0][l] = kNaN;
            return;
        }
        const double bx = bws_side(u, static_cast<double>(b.n), alt), by = bws_side(w, static_cast<double>(a.n), alt);
        out.p[0][l] = alt == 0 ? (bx + by) / 2.0 : (bx - by) / 2.0;
    });
    return wrap(std::move(out.arrays[0]));
}

// ---- tables under independence ---------------------------------------------------------------
//
// kind 0: Pearson's sum (t - e)^2 / e with e = R_i C_j / N (chi2_contingency without correction);
// kind 1: the table's probability given its margins, exp(sum log R_i! + sum log C_j! - log N! -
// sum log t_ij!) (fisher_exact's statistic of an r x c table).

struct Table {
    i64 r = 0, c = 0;
    std::vector<double> rows, cols, expected;
    double N = 0.0, logmarg = 0.0;
};

double lfact(double v) {
    int sign;
    return lgamma_r(v + 1.0, &sign);
}

Table table_of(const NDArray& t) {
    Table T;
    T.r = t.shape[0], T.c = t.shape[1];
    T.rows.assign(static_cast<std::size_t>(T.r), 0.0);
    T.cols.assign(static_cast<std::size_t>(T.c), 0.0);
    for (i64 i = 0; i < T.r; ++i)
        for (i64 j = 0; j < T.c; ++j) {
            const double v = *reinterpret_cast<const double*>(t.data + i * t.strides[0] + j * t.strides[1]);
            T.rows[static_cast<std::size_t>(i)] += v, T.cols[static_cast<std::size_t>(j)] += v, T.N += v;
        }
    T.expected.resize(static_cast<std::size_t>(T.r * T.c));
    for (i64 i = 0; i < T.r; ++i)
        for (i64 j = 0; j < T.c; ++j)
            T.expected[static_cast<std::size_t>(i * T.c + j)] =
                T.rows[static_cast<std::size_t>(i)] * T.cols[static_cast<std::size_t>(j)] / T.N;
    double lm = -lfact(T.N);
    for (double v : T.rows) lm += lfact(v);
    for (double v : T.cols) lm += lfact(v);
    T.logmarg = lm;
    return T;
}

double table_value(const Table& T, const double* t, int kind) {
    if (kind == 0) {
        Acc s;
        for (i64 q = 0; q < T.r * T.c; ++q) {
            const double e = T.expected[static_cast<std::size_t>(q)];
            const double d = t[q] - e;
            s.add(d * d / e);
        }
        return to_double(s.value());
    }
    double l = T.logmarg;
    for (i64 q = 0; q < T.r * T.c; ++q) l -= lfact(t[q]);
    return std::exp(l);
}

double table_stat(nb::handle table, int kind) {
    const NDArray& t = require_ndarray(table);
    const Table T = table_of(t);
    std::vector<double> v(static_cast<std::size_t>(T.r * T.c));
    for (i64 i = 0; i < T.r; ++i)
        for (i64 j = 0; j < T.c; ++j)
            v[static_cast<std::size_t>(i * T.c + j)] =
                *reinterpret_cast<const double*>(t.data + i * t.strides[0] + j * t.strides[1]);
    return table_value(T, v.data(), kind);
}

// count tables: exact, every permutation of the N column labels against the row labels in
// lexicographic order (count = N!); else tables drawn row by row, each cell a hypergeometric draw
// given what is left of its column and row (one uniform per cell but the last row and column:
// element position + j (r - 1)(c - 1) + q), by inversion.
nb::object table_null(nb::handle table, int kind, bool exact, u64 k0, u64 k1, u64 pos, i64 count) {
    const NDArray& t = require_ndarray(table);
    const Table T = table_of(t);
    const i64 r = T.r, c = T.c;
    const auto Nn = static_cast<i64>(T.N);
    std::vector<i64> rowlab, collab;
    if (exact) {
        for (i64 i = 0; i < r; ++i)
            for (i64 j = 0; j < c; ++j) {
                const auto v =
                    static_cast<i64>(*reinterpret_cast<const double*>(t.data + i * t.strides[0] + j * t.strides[1]));
                for (i64 q = 0; q < v; ++q) rowlab.push_back(i), collab.push_back(j);
            }
    }
    Outs out(1, count);
    const u64 D = static_cast<u64>((r - 1) * (c - 1));
    const i64 work = exact ? Nn : r * c * 8;
    const i64 per = std::max<i64>(1, kParallelTask / std::max<i64>(1, work));
    for_pieces(count, per, count * work, [&](i64, i64 b, i64 e) {
        random::Stream st(random::Key{k0, k1});
        std::vector<double> tab(static_cast<std::size_t>(r * c)), colrem(static_cast<std::size_t>(c));
        std::vector<i64> perm(static_cast<std::size_t>(Nn)), avail;
        for (i64 j = b; j < e; ++j) {
            std::fill(tab.begin(), tab.end(), 0.0);
            if (exact) {
                lehmer(static_cast<u64>(j), Nn, perm.data(), avail);
                for (i64 q = 0; q < Nn; ++q)
                    tab[static_cast<std::size_t>(
                        rowlab[static_cast<std::size_t>(q)] * c +
                        collab[static_cast<std::size_t>(perm[static_cast<std::size_t>(q)])]
                    )] += 1.0;
            } else {
                std::copy(T.cols.begin(), T.cols.end(), colrem.begin());
                double left = T.N;
                u64 q = 0;
                for (i64 i = 0; i + 1 < r; ++i) {
                    double need = T.rows[static_cast<std::size_t>(i)];
                    double pool = left;
                    for (i64 jj = 0; jj + 1 < c; ++jj) {
                        double v = 0.0;
                        if (need > 0.0 && colrem[static_cast<std::size_t>(jj)] > 0.0) {
                            const u64 w = st.first(pos + static_cast<u64>(j) * D + q);
                            const double u = (static_cast<double>(w >> 11) + 0.5) * 0x1p-53;
                            v = u <= 0.5 ? stats::method_of<stats::Hypergeometric, m::Icdf>(
                                               u, pool, colrem[static_cast<std::size_t>(jj)], need
                                           )
                                         : stats::method_of<stats::Hypergeometric, m::Iccdf>(
                                               1.0 - u, pool, colrem[static_cast<std::size_t>(jj)], need
                                           );
                        }
                        ++q;
                        tab[static_cast<std::size_t>(i * c + jj)] = v;
                        pool -= colrem[static_cast<std::size_t>(jj)];
                        colrem[static_cast<std::size_t>(jj)] -= v;
                        need -= v;
                    }
                    tab[static_cast<std::size_t>(i * c + c - 1)] = need;
                    colrem[static_cast<std::size_t>(c - 1)] -= need;
                    left -= T.rows[static_cast<std::size_t>(i)];
                }
                for (i64 jj = 0; jj < c; ++jj)
                    tab[static_cast<std::size_t>((r - 1) * c + jj)] = colrem[static_cast<std::size_t>(jj)];
            }
            out.p[0][j] = table_value(T, tab.data(), kind);
        }
    });
    return wrap(std::move(out.arrays[0]));
}

// ---- multiscale_graphcorr ----------------------------------------------------------------------

nb::object euclidean(nb::handle x) {
    const Rows a(rows_arg(x));  // (n, p)
    const i64 n = a.lanes, p = a.n;
    const i64 shape[] = {n, n};
    NDArray out = new_array(DType::Float64, shape);
    auto* o = reinterpret_cast<double*>(out.data);
    each_lane(n, 4 * n * p, [&](i64 i) {
        for (i64 j = 0; j < n; ++j) {
            double s = 0.0;
            for (i64 q = 0; q < p; ++q) {
                const double d = a.at(i, q) - a.at(j, q);
                s += d * d;
            }
            o[i * n + j] = std::sqrt(s);
        }
    });
    return wrap(std::move(out));
}

nb::tuple mgc_test(nb::handle distx, nb::handle disty, i64 reps, u64 k0, u64 k1, u64 pos) {
    const Rows X(rows_arg(distx)), Y(rows_arg(disty));
    const i64 n = X.lanes;
    const mgc::Prepared px = mgc::prepare(X), py = mgc::prepare(Y);
    mgc::Scratch sc(n);
    std::vector<i64> ident(static_cast<std::size_t>(n));
    std::iota(ident.begin(), ident.end(), i64{0});
    const double thr = mgc::threshold(n);
    const mgc::Stat obs = mgc::statistic(px, py, ident.data(), sc, thr);
    const i64 shape[] = {px.m, py.m};
    NDArray map = new_array(DType::Float64, shape);
    std::copy(sc.corr.begin(), sc.corr.begin() + px.m * py.m, reinterpret_cast<double*>(map.data));
    Outs null(1, reps);
    const i64 work = std::max<i64>(1, 4 * n * n);
    const i64 per = std::max<i64>(1, kParallelTask / work);
    for_pieces(reps, per, reps * work, [&](i64, i64 b, i64 e) {
        random::Stream st(random::Key{k0, k1});
        mgc::Scratch s(n);
        std::vector<i64> perm(static_cast<std::size_t>(n));
        for (i64 j = b; j < e; ++j) {
            std::iota(perm.begin(), perm.end(), i64{0});
            shuffle(st, pos + static_cast<u64>(j) * static_cast<u64>(n - 1), perm.data(), n);
            null.p[0][j] = mgc::statistic(px, py, perm.data(), s, thr).stat;
        }
    });
    return nb::make_tuple(obs.stat, wrap(std::move(map)), obs.k, obs.l, wrap(std::move(null.arrays[0])));
}

// ---- goodness_of_fit -----------------------------------------------------------------------------
//
// rows (lanes, n): the fit of the family to each lane with the parameters `fixed` (NaN: free)
// held, then the statistic (0 'ad', 1 'ks', 2 'cvm', 3 'filliben', -1 none) of the fitted
// distribution. Returns (statistic (lanes,), params (lanes, k)).
nb::tuple gof_fit(nb::handle x, int family, nb::handle fixed, int stat) {
    const Rows a(rows_arg(x));
    const NDArray& fx = require_ndarray(fixed);
    const auto* fv = reinterpret_cast<const double*>(fx.data);
    const int k = mle::n_params(family);
    std::vector<double> fixd(fv, fv + k);
    const i64 shape[] = {a.lanes, k};
    NDArray params = new_array(DType::Float64, shape);
    auto* pp = reinterpret_cast<double*>(params.data);
    Outs out(1, a.lanes);
    std::vector<double> med;
    if (stat == 3) med = mle::beta_medians(a.n);
    each_lane(a.lanes, 64 * a.n, [&](i64 l) {
        std::vector<double> v(static_cast<std::size_t>(a.n));
        for (i64 t = 0; t < a.n; ++t) v[static_cast<std::size_t>(t)] = a.at(l, t);
        std::sort(v.begin(), v.end());
        double th[3];
        const bool ok = mle::fit(family, v, fixd.data(), th);
        for (int q = 0; q < k; ++q) pp[l * k + q] = ok ? th[q] : kNaN;
        out.p[0][l] = ok && stat >= 0 ? mle::statistic(family, th, v, stat, med) : kNaN;
    });
    return nb::make_tuple(wrap(std::move(out.arrays[0])), wrap(std::move(params)));
}

}  // namespace

void bind_resampling(nb::module_& m) {
    m.def("perm_resamples", &perm_resamples);
    m.def("boot_resamples", &boot_resamples);
    m.def("jackknife", &jackknife);
    m.def("resample_pvalue", &resample_pvalue);
    m.def("lane_quantiles", &lane_quantiles);
    m.def("score_percentile", &score_percentile);
    m.def("bws", &bws);
    m.def("table_stat", &table_stat);
    m.def("table_null", &table_null);
    m.def("euclidean", &euclidean);
    m.def("mgc", &mgc_test);
    m.def("gof_fit", &gof_fit);
}

}  // namespace sd::htest
