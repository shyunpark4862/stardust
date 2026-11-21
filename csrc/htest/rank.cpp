// The rank tests (milestone H3, docs/design/htest.md §3.4).
//
// The Python side (stardust.htest._rank) checks the arguments and arranges every sample as a 2-D
// float64 array, one lane (one test of a batch) per row; each kernel returns the per-lane results
// as 1-D arrays. Lanes are independent: the same lane gives the same bits alone or in a batch,
// on any thread count.
//
//   mwu           mannwhitneyu: U1, p-value, z (normal approximation), method, exact work
//   ranksums      the rank sum z and its p-value
//   wilcoxon      the signed rank statistic, p-value, z, method, exact work
//   kruskal       H, p-value, df;  friedman: the Friedman chi^2, p-value, df
//   brunnermunzel W, p-value, df (Satterthwaite's, for the t distribution)
//   ansari        AB, p-value, method, exact work;  mood: z, p-value
//   median_test   the chi^2 (Cressie-Read) of the 2 x k table, p-value, grand median, df, table
//   page          L, p-value, method, exact work, a flag for ranks that are not integers or halves
//
// Ranks are exact: the values sorted with a tag of their sample, a tie run of t values from
// position b gets the midrank (2b + t + 1)/2, kept doubled as an integer, so every rank sum, U,
// T+ and AB is an exact integer (and the exact tests compare integers: no tie is split or merged
// by rounding). The statistics that subtract their means (H, the Friedman chi^2, Mood's T, the
// normal approximations' numerators) are formed from those integers in double-double; the
// p-values are sd.stats' distributions (Normal, ChiSquared, StudentT) called through their own
// methods, or the exact null distributions of htest/rank_exact.hpp (with ties, conditional on
// the observed midranks: decision H45).
//
// Degenerate lanes give NaN without warnings (H11): an empty sample, every value tied (a zero
// variance), a NaN under 'propagate'. The method a lane used (H12) is decided on its numbers
// (with 'propagate', on the values other than the NaNs).
#include "htest/rank.hpp"

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "htest/divergence.hpp"
#include "htest/lanes.hpp"
#include "htest/moments.hpp"
#include "htest/rank_exact.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/normal.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

namespace m = stats::method;
using exact::i128;
using exact::u128;

double chi2_sf(double x, double df) { return stats::method_of<stats::ChiSquared, m::Ccdf>(x, df); }
double norm_sf(double z) { return stats::method_of<stats::Normal, m::Ccdf>(z, 0.0, 1.0); }
double norm_cdf(double z) { return stats::method_of<stats::Normal, m::Cdf>(z, 0.0, 1.0); }
double t_sf(double x, double df) { return stats::method_of<stats::StudentT, m::Ccdf>(x, df); }
double t_cdf(double x, double df) { return stats::method_of<stats::StudentT, m::Cdf>(x, df); }

enum Alt { kTwoSided = 0, kLess = 1, kGreater = 2 };
enum Requested { kAuto = 0, kExactReq = 1, kAsymptoticReq = 2 };
constexpr double kUsedExact = 0.0, kUsedAsymptotic = 1.0;
// the exact distributions' memory: at most this many counts at once (H13, with the work cap)
constexpr double kStateCap = 1e7;

// scipy's _get_pvalue for a symmetric distribution (the standard normal, Student's t)
template <class Sf, class Cdf>
double symmetric_p(double x, int alt, Sf sf, Cdf cdf) {
    if (x != x) return x;
    if (alt == kTwoSided) return std::min(1.0, 2.0 * sf(std::fabs(x)));
    return alt == kLess ? cdf(x) : sf(x);
}
double normal_p(double z, int alt) { return symmetric_p(z, alt, norm_sf, norm_cdf); }

// sqrt of a nonnegative double-double: one Newton correction of sqrt(hi)
DD sqrt_dd(DD a) {
    if (!(a.hi > 0.0)) return DD{a.hi == 0.0 ? 0.0 : kNaN, 0.0};
    const double s = std::sqrt(a.hi);
    const DD s2 = dd::two_prod(s, s);
    return dd::two_sum(s, ((a.hi - s2.hi) - s2.lo + a.lo) / (2.0 * s));
}
DD dd_of(i128 v) { return exact::to_dd(v); }

// The exact counts a batch's lanes ask for again and again (lanes of one size without ties share
// the null distribution: the counts depend on the sizes and the bound alone): kept per thread.
// They are exact integers, so a remembered count is the count; a lane alone or in a batch, on any
// thread, gets the same bits. Each memo is dropped when it holds kMemo entries.
constexpr std::size_t kMemo = 1 << 16;
struct Key3 {
    std::int64_t a, b, c;
    bool operator<(const Key3& o) const { return a != o.a ? a < o.a : (b != o.b ? b < o.b : c < o.c); }
};
template <class F>
u128 memo_count(std::map<Key3, u128>& m, Key3 k, F&& count) {
    if (const auto it = m.find(k); it != m.end()) return it->second;
    if (m.size() >= kMemo) m.clear();
    const u128 v = count();
    m.emplace(k, v);
    return v;
}
// #{U <= b} of Mann-Whitney (exact::mwu_lower)
u128 mwu_lower_memo(std::int64_t N, std::int64_t c, std::int64_t b) {
    thread_local std::map<Key3, u128> m;
    return memo_count(m, {N, c, b}, [&] { return exact::mwu_lower(N, c, b); });
}
double ratio(DD a, DD b) { return to_double(a / b); }
double sgn(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }

// ---- ranks --------------------------------------------------------------------------------

// Lane r of the samples as keys tagged by their sample, without NaNs; n[i] each sample's count.
// True when a NaN was met.
bool lane_keys(const std::vector<Rows>& s, std::int64_t r, std::vector<Key>& keys, std::vector<std::int64_t>& n) {
    keys.clear();
    n.assign(s.size(), 0);
    bool nan = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        for (std::int64_t j = 0; j < s[i].n; ++j) {
            const double x = s[i].at(r, j);
            if (x != x) {
                nan = true;
                continue;
            }
            keys.push_back(Key{x, 0.0, static_cast<std::int32_t>(i)});
            ++n[i];
        }
    }
    return nan;
}

// f(b, t, first) for every run of equal keys of sorted v: t values from position b
template <class F>
void tie_runs(const std::vector<Key>& v, F&& f) {
    const auto n = static_cast<std::int64_t>(v.size());
    std::int64_t i = 0;
    while (i < n) {
        std::int64_t e = i + 1;
        while (e < n && key_equal(v[static_cast<std::size_t>(e)], v[static_cast<std::size_t>(i)])) ++e;
        f(i, e - i, i);
        i = e;
    }
}
std::int64_t tag_count(const std::vector<Key>& v, std::int64_t first, std::int64_t t, std::int32_t tag) {
    std::int64_t c = 0;
    for (std::int64_t k = first; k < first + t; ++k) c += v[static_cast<std::size_t>(k)].tag == tag;
    return c;
}

// ---- exact tails over double-double counts ------------------------------------------------

std::int64_t floor_div(std::int64_t a, std::int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return -floor_div(-a, b); }

// Scores with multiplicities (ascending, distinct)
struct Scores {
    std::vector<std::int64_t> v, t;
    void add(std::int64_t value, std::int64_t mult) {
        v.push_back(value);
        t.push_back(mult);
    }
    // merged by value (runs of equal scores from different ranks: Ansari's symmetric halves)
    void merge() {
        std::vector<std::pair<std::int64_t, std::int64_t>> q(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) q[i] = {v[i], t[i]};
        std::sort(q.begin(), q.end());
        v.clear(), t.clear();
        for (const auto& [a, b] : q) {
            if (!v.empty() && v.back() == a) t.back() += b;
            else add(a, b);
        }
    }
    std::int64_t total() const {
        std::int64_t s = 0;
        for (std::size_t i = 0; i < v.size(); ++i) s += v[i] * t[i];
        return s;
    }
};

// The distribution of the sum of c scores drawn without replacement: raw = base + g s, the
// counts by s as double-doubles (htest/rank_exact.hpp's grouped_counts).
struct Grouped {
    std::int64_t c = 0, base = 0, g = 1, smax = 0;
    double work = 0.0;
    std::vector<std::int64_t> vs, t;
    std::vector<DD> d;
    DD total{0.0, 0.0};

    Grouped(const Scores& sc, std::int64_t c_) : c(c_) {
        if (sc.v.empty()) return;
        const std::int64_t vmin = sc.v.front();
        base = c * vmin;
        std::int64_t gg = 0, N = 0;
        for (std::size_t i = 0; i < sc.v.size(); ++i) gg = std::gcd(gg, sc.v[i] - vmin), N += sc.t[i];
        g = gg == 0 ? 1 : gg;
        for (std::size_t i = 0; i < sc.v.size(); ++i) vs.push_back((sc.v[i] - vmin) / g), t.push_back(sc.t[i]);
        std::int64_t left = c;
        for (std::size_t i = vs.size(); i-- > 0 && left > 0;) {
            const std::int64_t take = std::min(left, t[i]);
            smax += take * vs[i];
            left -= take;
        }
        const double states = static_cast<double>(c + 1) * static_cast<double>(smax + 1);
        work =
            std::max(static_cast<double>(N) * static_cast<double>(c) * static_cast<double>(smax + 1), states * 100.0);
        if (states > kStateCap) work = std::max(work, states * 1000.0);
    }
    void run() {
        d = exact::grouped_counts(vs, t, c, smax);
        Acc a;
        for (const DD& x : d) a.add(x);
        total = a.value();
    }
    // P(raw <= v), P(raw >= v)
    double le(std::int64_t v) const {
        const std::int64_t s = floor_div(v - base, g);
        if (s < 0) return 0.0;
        Acc a;
        for (std::int64_t i = 0; i <= std::min(s, smax); ++i) a.add(d[static_cast<std::size_t>(i)]);
        return ratio(a.value(), total);
    }
    double ge(std::int64_t v) const {
        const std::int64_t s = std::max<std::int64_t>(0, ceil_div(v - base, g));
        if (s > smax) return 0.0;
        Acc a;
        for (std::int64_t i = smax; i >= s; --i) a.add(d[static_cast<std::size_t>(i)]);
        return ratio(a.value(), total);
    }
};

double count_ratio(u128 a, u128 b) { return ratio(exact::to_dd(a), exact::to_dd(b)); }

struct Out5 {
    double stat = kNaN, p = kNaN, z = kNaN, method = kNaN, work = 0.0;
};

// ---- Mann-Whitney U -----------------------------------------------------------------------

Out5 mwu_lane(
    std::vector<Key>& keys, std::int64_t n1, std::int64_t n2, bool cont, int alt, int req, double cap, bool decide
) {
    Out5 o;
    const std::int64_t N = n1 + n2, c = std::min(n1, n2);
    sort_keys(keys);
    std::int64_t twoR1 = 0;
    i128 T3 = 0;
    bool ties = false;
    Scores sc;
    tie_runs(keys, [&](std::int64_t b, std::int64_t t, std::int64_t first) {
        const std::int64_t a = 2 * b + t + 1;
        twoR1 += tag_count(keys, first, t, 0) * a;
        T3 += static_cast<i128>(t) * t * t - t;
        ties = ties || t > 1;
        sc.add(a, t);
    });
    const std::int64_t twoU1 = twoR1 - n1 * (n1 + 1), twoU2 = 2 * n1 * n2 - twoU1;
    const bool ints =
        !ties && c > 0 && exact::log2_binom(static_cast<double>(N), static_cast<double>(c)) < exact::kIntBits;
    // the tail the p-value needs, #{U <= b} (b = U2 for 'greater', U1 for 'less', the smaller for
    // two-sided), counted from the nearer end: #{U <= b} = C(N, c) - #{U <= top - b - 1}
    const std::int64_t top = c * (N - c);
    const std::int64_t need = (alt == kGreater ? twoU2 : (alt == kLess ? twoU1 : std::min(twoU1, twoU2))) / 2;
    const bool may = req == kExactReq || (req == kAuto && (n1 <= 8 || n2 <= 8) && !ties);
    double work = 0.0;
    if (may)
        work = ints ? static_cast<double>(c) * (static_cast<double>(std::min(need, top - need)) + 1.0)
                    : Grouped(sc, c).work;
    const bool use_exact = req == kExactReq || (may && work <= cap);
    o.method = use_exact ? kUsedExact : kUsedAsymptotic;
    if (decide || n1 == 0 || n2 == 0) return o;
    if (use_exact && work > cap) {
        o.work = work;
        return o;
    }
    o.stat = 0.5 * static_cast<double>(twoU1);
    if (use_exact) {
        // le = P(U1' <= U1), ge = P(U1' >= U1). Without ties the null is symmetric and x's U and
        // y's are alike: ge = P(U' <= U2), counted for the smaller sample from the nearer end
        // (scipy's two-sided 2 P(U' >= max(U1, U2)) is then 2 min(le, ge)). With ties it need not
        // be symmetric (tie runs need not mirror each other): both tails are counted, of the
        // smaller sample's rank sum (x's U1 = n1 n2 - U2).
        double le, ge;
        if (ints) {
            const u128 total = exact::binom_u128(N, c);
            auto lower = [&](std::int64_t w) {
                const std::int64_t b = floor_div(w, 2);
                const u128 v = 2 * b >= top ? total - mwu_lower_memo(N, c, top - b - 1) : mwu_lower_memo(N, c, b);
                return count_ratio(v, total);
            };
            le = lower(twoU1), ge = lower(twoU2);
        } else {
            Grouped gd(sc, c);
            gd.run();
            if (c == n1) {
                le = gd.le(twoU1 + n1 * (n1 + 1)), ge = gd.ge(twoU1 + n1 * (n1 + 1));
            } else {
                le = gd.ge(twoU2 + n2 * (n2 + 1)), ge = gd.le(twoU2 + n2 * (n2 + 1));
            }
        }
        o.p = alt == kGreater ? ge : (alt == kLess ? le : std::min(1.0, 2.0 * std::min(le, ge)));
        return o;
    }
    const double s = alt == kGreater ? 1.0 : (alt == kLess ? -1.0 : sgn(static_cast<double>(twoU1 - n1 * n2)));
    const std::int64_t num2 = twoU1 - n1 * n2 - (cont ? static_cast<std::int64_t>(s) : 0);
    const i128 Nn = N, NN = Nn * (Nn - 1);
    const DD var = dd_of(static_cast<i128>(n1) * n2 * ((Nn + 1) * NN - T3)) / (12.0 * static_cast<double>(NN));
    o.z = ratio(DD{0.5 * static_cast<double>(num2), 0.0}, sqrt_dd(var));
    if (!(var.hi > 0.0)) o.z = kNaN;
    o.p = normal_p(o.z, alt);
    return o;
}

// ---- Wilcoxon's signed ranks --------------------------------------------------------------

enum ZeroMethod { kWilcox = 0, kPratt = 1, kZsplit = 2 };

struct Signed {
    double hi, lo;
    int sign;
};

Out5 wilcoxon_lane(std::vector<Signed>& d, int zm, bool corr, int alt, int req, double cap, bool decide) {
    Out5 o;
    const auto n_all = static_cast<std::int64_t>(d.size());
    std::int64_t nz = 0;
    for (const Signed& v : d) nz += v.sign == 0;
    std::vector<Key> keys;
    for (const Signed& v : d)
        if (!(zm == kWilcox && v.sign == 0)) keys.push_back(Key{v.hi, v.lo, static_cast<std::int32_t>(v.sign + 1)});
    sort_keys(keys);
    const auto count = static_cast<std::int64_t>(keys.size());
    std::int64_t twoRp = 0, twoRm = 0, twoRz = 0;
    i128 T3 = 0;
    bool ties = false;
    std::vector<std::int64_t> nonzero;  // doubled ranks of the nonzero differences
    tie_runs(keys, [&](std::int64_t b, std::int64_t t, std::int64_t first) {
        const std::int64_t a = 2 * b + t + 1;
        const std::int64_t cp = tag_count(keys, first, t, 2), cm = tag_count(keys, first, t, 0);
        twoRp += cp * a, twoRm += cm * a, twoRz += (t - cp - cm) * a;
        for (std::int64_t k = 0; k < cp + cm; ++k) nonzero.push_back(a);
        ties = ties || t > 1;
        const bool zero_run = keys[static_cast<std::size_t>(first)].tag == 1;
        if (!(zm == kPratt && zero_run)) T3 += static_cast<i128>(t) * t * t - t;
    });
    const std::int64_t konst = zm == kZsplit ? twoRz / 2 : 0;  // (the zeros' doubled ranks sum to t(t + 1): even)
    twoRp += konst, twoRm += konst;
    std::int64_t g = 0, S = 0;
    for (std::int64_t a : nonzero) g = std::gcd(g, a);
    g = g == 0 ? 1 : g;
    for (std::int64_t a : nonzero) S += a / g;
    const auto nn = static_cast<double>(nonzero.size());
    const double work = std::max(nn * static_cast<double>(S + 1), static_cast<double>(S + 1) * 100.0);
    bool use_exact;
    if (req == kAuto) {
        use_exact = n_all <= 50 && (!(ties || nz > 0) || n_all <= 13) && work <= cap;
    } else {
        use_exact = req == kExactReq;
    }
    o.method = use_exact ? kUsedExact : kUsedAsymptotic;
    if (decide || count == 0) return o;
    if (use_exact && work > cap) {
        o.work = work;
        return o;
    }
    o.stat = 0.5 * static_cast<double>(alt == kTwoSided ? std::min(twoRp, twoRm) : twoRp);
    if (use_exact) {
        // T+ = konst + sum a_i B_i over the nonzero differences, B_i fair and independent
        std::vector<std::int64_t> a;
        for (std::int64_t v : nonzero) a.push_back(v / g);
        double le, ge;
        const std::int64_t lo_s = floor_div(twoRp - konst, g), hi_s = ceil_div(twoRp - konst, g);
        if (a.size() <= 124) {
            // (the last lane's counts kept per thread: a batch's lanes without ties have the same a)
            thread_local std::vector<std::int64_t> last_a;
            thread_local std::vector<u128> last_c;
            if (a != last_a) last_c = exact::signed_rank_counts(a), last_a = a;
            const std::vector<u128>& c = last_c;
            u128 tot = 0, below = 0, above = 0;
            for (std::int64_t s = 0; s <= S; ++s) {
                const u128 v = c[static_cast<std::size_t>(s)];
                tot += v;
                if (s <= lo_s) below += v;
                if (s >= hi_s) above += v;
            }
            le = count_ratio(below, tot), ge = count_ratio(above, tot);
        } else {
            const std::vector<DD> c = exact::signed_rank_probs(a);
            Acc below, above;
            for (std::int64_t s = 0; s <= std::min(lo_s, S); ++s) below.add(c[static_cast<std::size_t>(s)]);
            for (std::int64_t s = S; s >= std::max<std::int64_t>(hi_s, 0); --s)
                above.add(c[static_cast<std::size_t>(s)]);
            le = to_double(below.value()), ge = to_double(above.value());
        }
        o.p = alt == kLess ? le : (alt == kGreater ? ge : std::min(1.0, 2.0 * std::min(le, ge)));
        return o;
    }
    i128 mn4 = static_cast<i128>(count) * (count + 1);
    i128 x = static_cast<i128>(count) * (count + 1) * (2 * count + 1);
    if (zm == kPratt) {
        mn4 -= static_cast<i128>(nz) * (nz + 1);
        x -= static_cast<i128>(nz) * (nz + 1) * (2 * nz + 1);
    }
    const i128 se48 = 2 * x - T3;
    i128 num4 = 2 * static_cast<i128>(twoRp) - mn4;
    if (corr) {
        const double s = alt == kGreater ? 1.0 : (alt == kLess ? -1.0 : sgn(static_cast<double>(num4)));
        num4 -= 2 * static_cast<i128>(s);
    }
    if (se48 > 0) o.z = ratio(dd_of(num4) / 4.0, sqrt_dd(dd_of(se48) / 48.0));
    o.p = normal_p(o.z, alt);
    if (alt == kTwoSided) o.z = -std::fabs(o.z);
    return o;
}

// ---- Ansari-Bradley -----------------------------------------------------------------------

// The Gaussian binomials of Ansari-Bradley's score halves by k (N = 2k or 2k + 1), made once:
// a batch's lanes and repeated calls of one size share them (the integer path's k <= 65).
const exact::AnsariCounts& ansari_counts(std::int64_t k) {
    static std::mutex lock;
    static std::map<std::int64_t, std::unique_ptr<const exact::AnsariCounts>> made;
    const std::lock_guard<std::mutex> g(lock);
    auto it = made.find(k);
    if (it == made.end()) it = made.emplace(k, std::make_unique<const exact::AnsariCounts>(k)).first;
    return *it->second;
}

Out5 ansari_lane(std::vector<Key>& keys, std::int64_t n, std::int64_t mm, int alt, int req, double cap, bool decide) {
    // n = len(x), mm = len(y) (scipy's names)
    Out5 o;
    const std::int64_t N = n + mm;
    sort_keys(keys);
    std::int64_t twoAB = 0, A = 0;
    i128 fac4 = 0;
    bool ties = false;
    std::int64_t runs = 0;
    Scores sc;
    tie_runs(keys, [&](std::int64_t b, std::int64_t t, std::int64_t first) {
        const std::int64_t r2 = 2 * b + t + 1, s2 = std::min(r2, 2 * (N + 1) - r2);
        twoAB += tag_count(keys, first, t, 0) * s2;
        A += t * s2;
        fac4 += static_cast<i128>(t) * s2 * s2;
        ties = ties || t > 1;
        ++runs;
        sc.add(s2, t);
    });
    const std::int64_t c = std::min(n, mm), k = N / 2;
    const bool ints =
        !ties && n > 0 && mm > 0 && exact::log2_binom(static_cast<double>(N), static_cast<double>(n)) < exact::kIntBits;
    const bool may = req == kExactReq || (req == kAuto && mm < 55 && n < 55 && !ties);
    double work = 0.0;
    if (may) {
        if (!ints) sc.merge();
        work = ints ? std::pow(static_cast<double>(k + 1), 3.0) : Grouped(sc, c).work;
    }
    const bool use_exact = req == kExactReq || (may && work <= cap);
    o.method = use_exact ? kUsedExact : kUsedAsymptotic;
    if (decide || n == 0 || mm == 0) return o;
    if (use_exact && work > cap) {
        o.work = work;
        return o;
    }
    o.stat = 0.5 * static_cast<double>(twoAB);
    if (runs == 1) return o;  // every value equal: AB is fixed, no distribution (H157: p NaN)
    if (use_exact) {
        double cdf, sf;
        if (ints) {
            const exact::AnsariCounts& ac = ansari_counts(k);
            const u128 total = exact::binom_u128(N, n);
            const std::int64_t ab = twoAB / 2;
            thread_local std::map<Key3, u128> memo;  // (memo_count: the same counts)
            auto lower = [&](std::int64_t b) { return memo_count(memo, {N, n, b}, [&] { return ac.lower(N, n, b); }); };
            cdf = count_ratio(lower(ab), total);
            sf = count_ratio(total - lower(ab - 1), total);
        } else {
            Grouped gd(sc, c);
            gd.run();
            if (c == n) {
                cdf = gd.le(twoAB), sf = gd.ge(twoAB);
            } else {  // AB of x = A - (y's)
                cdf = gd.ge(A - twoAB), sf = gd.le(A - twoAB);
            }
        }
        o.p = alt == kGreater ? cdf : (alt == kLess ? sf : std::min(1.0, 2.0 * std::min(cdf, sf)));
        return o;
    }
    const i128 Nn = N, nm = static_cast<i128>(n) * mm;
    const bool odd = N % 2 == 1;
    const DD mean = odd ? dd_of(static_cast<i128>(n) * (Nn + 1) * (Nn + 1)) / (4.0 * static_cast<double>(N))
                        : dd_of(static_cast<i128>(n) * (Nn + 2)) / 4.0;
    DD var;
    if (ties) {
        var = odd ? dd_of(nm * (4 * Nn * fac4 - (Nn + 1) * (Nn + 1) * (Nn + 1) * (Nn + 1))) /
                        dd_of(16 * Nn * Nn * (Nn - 1))
                  : dd_of(nm * (4 * fac4 - Nn * (Nn + 2) * (Nn + 2))) / dd_of(16 * Nn * (Nn - 1));
    } else {
        var = odd ? dd_of(nm * (Nn + 1) * (3 + Nn * Nn)) / dd_of(48 * Nn * Nn)
                  : dd_of(nm * (Nn + 2) * (Nn - 2)) / dd_of(48 * (Nn - 1));
    }
    if (var.hi > 0.0) o.z = ratio(mean - DD{0.5 * static_cast<double>(twoAB), 0.0}, sqrt_dd(var));
    o.p = normal_p(o.z, alt);
    return o;
}

// ---- the kernels over lanes ---------------------------------------------------------------

std::vector<Rows> rows_list(nb::list samples) {
    std::vector<Rows> s;
    for (nb::handle h : samples) s.emplace_back(rows_arg(h));
    return s;
}
std::int64_t total_n(const std::vector<Rows>& s) {
    std::int64_t t = 0;
    for (const Rows& r : s) t += r.n;
    return t;
}

template <class Lane>
nb::tuple two_sample(nb::handle x, nb::handle y, bool omit, int outs, Lane&& lane) {
    const std::vector<Rows> s{Rows(rows_arg(x)), Rows(rows_arg(y))};
    Outs out(outs, s[0].lanes);
    each_lane(s[0].lanes, 32 * total_n(s), [&](std::int64_t r) {
        std::vector<Key> keys;
        std::vector<std::int64_t> n;
        const bool nan = lane_keys(s, r, keys, n);
        lane(keys, n[0], n[1], nan && !omit, out, r);
    });
    return out.tuple();
}

nb::tuple mwu(nb::handle x, nb::handle y, bool cont, int alt, int req, double cap, bool omit) {
    return two_sample(
        x, y, omit, 5, [&](std::vector<Key>& k, std::int64_t n1, std::int64_t n2, bool nan, Outs& out, std::int64_t r) {
            const Out5 o = mwu_lane(k, n1, n2, cont, alt, req, cap, nan);
            out.p[0][r] = o.stat, out.p[1][r] = o.p, out.p[2][r] = o.z, out.p[3][r] = o.method, out.p[4][r] = o.work;
        }
    );
}

nb::tuple ranksums(nb::handle x, nb::handle y, int alt, bool omit) {
    return two_sample(
        x, y, omit, 2, [&](std::vector<Key>& k, std::int64_t n1, std::int64_t n2, bool nan, Outs& out, std::int64_t r) {
            double z = kNaN, p = kNaN;
            if (!nan && n1 > 0 && n2 > 0) {
                sort_keys(k);
                std::int64_t twoR1 = 0, runs = 0;
                tie_runs(k, [&](std::int64_t b, std::int64_t t, std::int64_t first) {
                    twoR1 += tag_count(k, first, t, 0) * (2 * b + t + 1);
                    ++runs;
                });
                const std::int64_t N = n1 + n2;
                if (runs > 1) {  // every value equal: NaN (H157)
                    const DD var = dd_of(static_cast<i128>(n1) * n2 * (N + 1)) / 12.0;
                    z = ratio(DD{0.5 * static_cast<double>(twoR1 - n1 * (N + 1)), 0.0}, sqrt_dd(var));
                    p = normal_p(z, alt);
                }
            }
            out.p[0][r] = z, out.p[1][r] = p;
        }
    );
}

nb::tuple wilcoxon(nb::handle x, nb::handle y, int zm, bool corr, int alt, int req, double cap, bool omit) {
    const Rows a(rows_arg(x));
    Rows b;
    const bool paired = !y.is_none();
    if (paired) b = Rows(rows_arg(y));
    Outs out(5, a.lanes);
    each_lane(a.lanes, 32 * a.n, [&](std::int64_t r) {
        std::vector<Signed> d;
        bool nan = false, bad = false;
        for (std::int64_t j = 0; j < a.n; ++j) {
            const double u = a.at(r, j), v = paired ? b.at(r, j) : 0.0;
            if (u != u || v != v) {
                nan = true;
                continue;
            }
            DD e = paired ? dd::two_sum(u, -v) : DD{u, 0.0};
            if (e.hi != e.hi) {  // inf - inf
                bad = true;
                continue;
            }
            if (!std::isfinite(e.hi)) e.lo = 0.0;
            const int s = e.hi > 0.0 ? 1 : (e.hi < 0.0 ? -1 : 0);
            const DD m = abs_dd(e);
            d.push_back(Signed{m.hi, m.lo, s});
        }
        Out5 o = wilcoxon_lane(d, zm, corr, alt, req, cap, (nan && !omit) || bad);
        out.p[0][r] = o.stat, out.p[1][r] = o.p, out.p[2][r] = o.z, out.p[3][r] = o.method, out.p[4][r] = o.work;
    });
    return out.tuple();
}

nb::tuple ansari(nb::handle x, nb::handle y, int alt, int req, double cap, bool omit) {
    return two_sample(
        x, y, omit, 4, [&](std::vector<Key>& k, std::int64_t n1, std::int64_t n2, bool nan, Outs& out, std::int64_t r) {
            const Out5 o = ansari_lane(k, n1, n2, alt, req, cap, nan);
            out.p[0][r] = o.stat, out.p[1][r] = o.p, out.p[2][r] = o.method, out.p[3][r] = o.work;
        }
    );
}

// Mood's squared rank test with Mielke's tie correction (scipy's): T = sum over the runs of
// cx phi, phi the run's mean of (I - (N + 1)/2)^2 (without ties the sum of (r - (N + 1)/2)^2).
// Kept as 4T = sum cx S/t with S = sum (2I - N - 1)^2 an exact integer.
nb::tuple mood(nb::handle x, nb::handle y, int alt, bool omit) {
    return two_sample(
        x, y, omit, 2, [&](std::vector<Key>& k, std::int64_t mx, std::int64_t ny, bool nan, Outs& out, std::int64_t r) {
            double z = kNaN, p = kNaN;
            const std::int64_t N = mx + ny;
            if (!nan && mx > 0 && ny > 0 && N >= 3) {
                sort_keys(k);
                Acc T4;
                i128 ties = 0;
                const i128 Nn = N;
                tie_runs(k, [&](std::int64_t b, std::int64_t t, std::int64_t first) {
                    const i128 tt = t, u0 = 2 * static_cast<i128>(b) + 1 - Nn;
                    const i128 S = tt * u0 * u0 + 2 * u0 * tt * (tt - 1) + 4 * ((tt - 1) * tt * (2 * tt - 1) / 6);
                    const std::int64_t cx = tag_count(k, first, t, 0);
                    if (cx) T4.add((dd_of(S) * static_cast<double>(cx)) / static_cast<double>(t));
                    const i128 q = Nn - 2 * static_cast<i128>(b) - tt;
                    ties += tt * (tt * tt - 1) * (tt * tt - 4 + 15 * q * q);
                });
                const DD E4 = dd_of(static_cast<i128>(mx) * (Nn * Nn - 1)) / 3.0;
                const DD var = (dd_of((Nn + 1) * (Nn * Nn - 4)) - dd_of(ties) / dd_of(Nn * (Nn - 1))) *
                               (static_cast<double>(mx) * static_cast<double>(ny)) / 180.0;
                if (var.hi > 0.0) z = ratio((T4.value() - E4) / 4.0, sqrt_dd(var));
                p = normal_p(z, alt);
            }
            out.p[0][r] = z, out.p[1][r] = p;
        }
    );
}

// Brunner and Munzel: with P the placement of an observation (the other sample's values below it,
// ties counting half; 2P an integer), x's variance is sum (P - U1/nx)^2 / (nx - 1), formed from
// the integers nx 2P - 2U1; the statistic is (nx ny / 2 - U1) / sqrt(nx Sx + ny Sy).
nb::tuple brunnermunzel(nb::handle x, nb::handle y, int alt, int dist, bool omit) {
    return two_sample(
        x, y, omit, 3, [&](std::vector<Key>& k, std::int64_t nx, std::int64_t ny, bool nan, Outs& out, std::int64_t r) {
            double w = kNaN, p = kNaN, df = kNaN;
            if (!nan && nx >= 2 && ny >= 2) {
                sort_keys(k);
                struct Run {
                    std::int64_t cx, cy, px, py;
                };
                std::vector<Run> runs;
                std::int64_t px = 0, py = 0, twoU1 = 0;
                tie_runs(k, [&](std::int64_t, std::int64_t t, std::int64_t first) {
                    const std::int64_t cx = tag_count(k, first, t, 0), cy = t - cx;
                    runs.push_back(Run{cx, cy, px, py});
                    twoU1 += cx * (2 * py + cy);
                    px += cx, py += cy;
                });
                const std::int64_t twoU2 = 2 * nx * ny - twoU1;
                Acc sx, sy;
                for (const Run& u : runs) {
                    if (u.cx) {
                        const double a = static_cast<double>(nx * (2 * u.py + u.cy) - twoU1);
                        sx.add(dd::two_prod(a, a) * static_cast<double>(u.cx));
                    }
                    if (u.cy) {
                        const double a = static_cast<double>(ny * (2 * u.px + u.cx) - twoU2);
                        sy.add(dd::two_prod(a, a) * static_cast<double>(u.cy));
                    }
                }
                const DD vx = sx.value() / (4.0 * static_cast<double>(nx) * static_cast<double>(nx - 1));  // nx Sx
                const DD vy = sy.value() / (4.0 * static_cast<double>(ny) * static_cast<double>(ny - 1));  // ny Sy
                const DD V = vx + vy;
                if (V.hi > 0.0) {
                    w = ratio(DD{0.5 * static_cast<double>(nx * ny - twoU1), 0.0}, sqrt_dd(V));
                    df =
                        ratio(V * V, (vx * vx) / static_cast<double>(nx - 1) + (vy * vy) / static_cast<double>(ny - 1));
                    if (dist == 0) {
                        p = symmetric_p(
                            -w, alt, [&](double v) { return t_sf(v, df); }, [&](double v) { return t_cdf(v, df); }
                        );
                    } else {
                        p = normal_p(-w, alt);
                    }
                }
            }
            out.p[0][r] = w, out.p[1][r] = p, out.p[2][r] = df;
        }
    );
}

// Kruskal-Wallis: H = 3 (N - 1) sum_i D_i^2 / n_i / (N^3 - N - sum (t^3 - t)), D_i = 2 R_i - n_i (N + 1)
// an exact integer (12/(N(N+1)) sum n_i (Rbar_i - (N+1)/2)^2 over scipy's tie correction).
nb::tuple kruskal(nb::list samples, bool omit) {
    const std::vector<Rows> s = rows_list(samples);
    const auto k = static_cast<std::int64_t>(s.size());
    Outs out(3, s[0].lanes);
    each_lane(s[0].lanes, 32 * total_n(s), [&](std::int64_t r) {
        std::vector<Key> keys;
        std::vector<std::int64_t> n;
        const bool nan = lane_keys(s, r, keys, n);
        double h = kNaN, p = kNaN;
        const bool empty = std::any_of(n.begin(), n.end(), [](std::int64_t v) { return v == 0; });
        if (!(nan && !omit) && !empty) {
            sort_keys(keys);
            std::vector<std::int64_t> twoR(static_cast<std::size_t>(k), 0);
            i128 T3 = 0;
            tie_runs(keys, [&](std::int64_t b, std::int64_t t, std::int64_t first) {
                for (std::int64_t i = first; i < first + t; ++i)
                    twoR[static_cast<std::size_t>(keys[static_cast<std::size_t>(i)].tag)] += 2 * b + t + 1;
                T3 += static_cast<i128>(t) * t * t - t;
            });
            const auto N = static_cast<std::int64_t>(keys.size());
            Acc sum;
            for (std::int64_t i = 0; i < k; ++i) {
                const double D =
                    static_cast<double>(twoR[static_cast<std::size_t>(i)] - n[static_cast<std::size_t>(i)] * (N + 1));
                sum.add(dd::two_prod(D, D) / static_cast<double>(n[static_cast<std::size_t>(i)]));
            }
            const i128 Nn = N, den = Nn * Nn * Nn - Nn - T3;
            if (den > 0) {
                h = ratio(sum.value() * (3.0 * static_cast<double>(N - 1)), dd_of(den));
                p = chi2_sf(h, static_cast<double>(k - 1));
            }
        }
        out.p[0][r] = h, out.p[1][r] = p, out.p[2][r] = static_cast<double>(k - 1);
    });
    return out.tuple();
}

// Friedman: k paired samples of n; each observation's k values ranked; with D_j = 2 R_j - n (k + 1)
// the statistic is 3 (k - 1) sum D_j^2 / (k (k^2 - 1) n - sum t (t^2 - 1)).
nb::tuple friedman(nb::list samples, bool omit) {
    const std::vector<Rows> s = rows_list(samples);
    const auto k = static_cast<std::int64_t>(s.size());
    const std::int64_t n0 = s[0].n;
    Outs out(3, s[0].lanes);
    each_lane(s[0].lanes, 16 * k * n0, [&](std::int64_t r) {
        std::vector<std::int64_t> twoR(static_cast<std::size_t>(k), 0);
        std::vector<std::pair<double, std::int64_t>> row(static_cast<std::size_t>(k));
        i128 T = 0;
        std::int64_t n = 0;
        bool nan = false;
        for (std::int64_t j = 0; j < n0 && !nan; ++j) {
            bool has = false;
            for (std::int64_t i = 0; i < k; ++i) {
                const double v = s[static_cast<std::size_t>(i)].at(r, j);
                has = has || v != v;
                row[static_cast<std::size_t>(i)] = {v, i};
            }
            if (has) {
                nan = !omit;
                continue;
            }
            std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (std::int64_t i = 0; i < k;) {
                std::int64_t e = i + 1;
                while (e < k && row[static_cast<std::size_t>(e)].first == row[static_cast<std::size_t>(i)].first) ++e;
                const std::int64_t t = e - i;
                for (std::int64_t q = i; q < e; ++q)
                    twoR[static_cast<std::size_t>(row[static_cast<std::size_t>(q)].second)] += i + e + 1;
                T += static_cast<i128>(t) * (t * t - 1);
                i = e;
            }
            ++n;
        }
        double stat = kNaN, p = kNaN;
        if (!nan && n > 0) {
            Acc sum;
            for (std::int64_t i = 0; i < k; ++i) {
                const double D = static_cast<double>(twoR[static_cast<std::size_t>(i)] - n * (k + 1));
                sum.add(dd::two_prod(D, D));
            }
            const i128 den = static_cast<i128>(k) * (k * k - 1) * n - T;
            if (den > 0) {
                stat = ratio(sum.value() * (3.0 * static_cast<double>(k - 1)), dd_of(den));
                p = chi2_sf(stat, static_cast<double>(k - 1));
            }
        }
        out.p[0][r] = stat, out.p[1][r] = p, out.p[2][r] = static_cast<double>(k - 1);
    });
    return out.tuple();
}

// Mood's median test: the grand median (the middle value, or the exact mean of the two middle
// ones as a double-double), the 2 x k table of the counts above and below it (ties as asked), and
// scipy's chi2_contingency of the table: expected counts R_i C_j / N, Yates' correction when the
// table has one degree of freedom, the Cressie-Read statistic (htest/divergence.hpp).
enum Ties { kBelow = 0, kAbove = 1, kIgnore = 2 };

nb::tuple median_test(nb::list samples, int ties, bool correction, double lam, bool omit) {
    const std::vector<Rows> s = rows_list(samples);
    const auto k = static_cast<std::int64_t>(s.size());
    const std::int64_t lanes = s[0].lanes;
    Outs out(4, lanes);
    const std::int64_t shape[] = {lanes, 2 * k};
    NDArray table = new_array(DType::Float64, shape);
    double* tab = reinterpret_cast<double*>(table.data);
    each_lane(lanes, 8 * total_n(s), [&](std::int64_t r) {
        // the lane's values, sample by sample (start[i] where sample i begins)
        std::vector<double> v, w;
        std::vector<std::int64_t> start(static_cast<std::size_t>(k + 1), 0);
        bool nan = false, empty = false;
        for (std::int64_t i = 0; i < k; ++i) {
            const Rows& g = s[static_cast<std::size_t>(i)];
            for (std::int64_t j = 0; j < g.n; ++j) {
                const double x = g.at(r, j);
                if (x != x) nan = true;
                else v.push_back(x);
            }
            start[static_cast<std::size_t>(i + 1)] = static_cast<std::int64_t>(v.size());
            empty = empty || start[static_cast<std::size_t>(i + 1)] == start[static_cast<std::size_t>(i)];
        }
        double stat = kNaN, p = kNaN, med = kNaN;
        double* row = tab + r * 2 * k;
        std::fill(row, row + 2 * k, 0.0);
        if (!(nan && !omit) && !empty) {
            w = v;
            const auto N = static_cast<std::int64_t>(w.size());
            const auto mid = w.begin() + N / 2;
            std::nth_element(w.begin(), mid, w.end());
            DD M{*mid, 0.0};
            if (N % 2 == 0) {
                const DD s2 = dd::two_sum(*std::max_element(w.begin(), mid), *mid);
                M = DD{0.5 * s2.hi, 0.5 * s2.lo};
            }
            med = M.hi;
            // x > M exactly: x > M.hi, or x = M.hi above a negative low part (and below likewise)
            const bool eq_up = M.lo < 0.0, eq_dn = M.lo > 0.0;
            std::vector<std::int64_t> above(static_cast<std::size_t>(k), 0), below(static_cast<std::size_t>(k), 0);
            for (std::int64_t i = 0; i < k; ++i) {
                std::int64_t up = 0, dn = 0, eq = 0;
                for (std::int64_t q = start[static_cast<std::size_t>(i)]; q < start[static_cast<std::size_t>(i + 1)];
                     ++q) {
                    const double x = v[static_cast<std::size_t>(q)];
                    up += x > M.hi, dn += x < M.hi, eq += x == M.hi;
                }
                if (eq_up) up += eq, eq = 0;
                if (eq_dn) dn += eq, eq = 0;
                if (ties == kAbove) up += eq;
                if (ties == kBelow) dn += eq;
                above[static_cast<std::size_t>(i)] = up, below[static_cast<std::size_t>(i)] = dn;
            }
            std::int64_t R0 = 0, R1 = 0;
            bool zero_col = false;
            for (std::int64_t j = 0; j < k; ++j) {
                row[j] = static_cast<double>(above[static_cast<std::size_t>(j)]);
                row[k + j] = static_cast<double>(below[static_cast<std::size_t>(j)]);
                R0 += above[static_cast<std::size_t>(j)], R1 += below[static_cast<std::size_t>(j)];
                zero_col = zero_col || above[static_cast<std::size_t>(j)] + below[static_cast<std::size_t>(j)] == 0;
            }
            if (R0 > 0 && R1 > 0 && !zero_col) {
                const std::int64_t T = R0 + R1;
                div::Divergence dv(lam);
                for (int i = 0; i < 2; ++i) {
                    const std::int64_t Ri = i == 0 ? R0 : R1;
                    for (std::int64_t j = 0; j < k; ++j) {
                        const std::int64_t O =
                            i == 0 ? above[static_cast<std::size_t>(j)] : below[static_cast<std::size_t>(j)];
                        const std::int64_t C = above[static_cast<std::size_t>(j)] + below[static_cast<std::size_t>(j)];
                        const DD E = dd_of(static_cast<i128>(Ri) * C) / static_cast<double>(T);
                        DD d = DD{static_cast<double>(O), 0.0} - E;
                        if (correction && k == 2) {  // Yates: |O - E| less min(1/2, |O - E|)
                            const DD ad = abs_dd(d);
                            d = (ad.hi < 0.5 || (ad.hi == 0.5 && ad.lo <= 0.0)) ? DD{0.0, 0.0}
                                                                                : d - DD{0.5 * sgn(d.hi), 0.0};
                        }
                        dv.add(E, d);
                    }
                }
                stat = dv.value();
                p = chi2_sf(stat, static_cast<double>(k - 1));
            }
        }
        out.p[0][r] = stat, out.p[1][r] = p, out.p[2][r] = med, out.p[3][r] = static_cast<double>(k - 1);
    });
    nb::tuple t = out.tuple();
    return nb::make_tuple(t[0], t[1], t[2], t[3], wrap(std::move(table)));
}

// Page's L: each row's values ranked (doubled midranks) unless already ranks; L = sum_j p_j C_j
// of the column sums C_j. The exact distribution is each row's over the permutations of its
// values, convolved (with ties conditional on each row's midranks, H45); without ties it depends
// on (m, n) alone and is made once (scipy keeps its tables between calls too). The normal
// approximation is scipy's (no tie correction).

// The exact upper tails of L' (doubled and divided by g: L2 / g = base + s): up[s] = #{>= s}.
struct PageTail {
    std::vector<DD> up;
    std::int64_t base = 0, g = 1;
    DD total{1.0, 0.0};
    double p(DD L2) const {
        const double s0 = std::ceil(to_double(L2) / static_cast<double>(g) - static_cast<double>(base) - 1e-9);
        if (s0 >= static_cast<double>(up.size())) return 0.0;
        const auto s = static_cast<std::size_t>(std::max(0.0, s0));
        return std::min(1.0, ratio(up[s], total));
    }
};

// rows: each row's doubled values (integers), divided by their common g
PageTail page_tail(const std::vector<std::vector<std::int64_t>>& rows, std::int64_t g) {
    PageTail pt;
    pt.g = g;
    std::map<std::vector<std::int64_t>, std::pair<std::vector<std::uint64_t>, std::int64_t>> cache;
    std::vector<DD> dist{DD{1.0, 0.0}};
    for (const auto& key : rows) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            std::int64_t low;
            std::vector<std::uint64_t> c = exact::page_row_counts(key, low);
            it = cache.emplace(key, std::make_pair(std::move(c), low)).first;
        }
        const std::vector<std::uint64_t>& c = it->second.first;
        pt.base += it->second.second;
        std::vector<DD> nd(dist.size() + c.size() - 1, DD{0.0, 0.0});
        for (std::size_t u = 0; u < dist.size(); ++u) {
            if (dist[u].hi == 0.0) continue;
            for (std::size_t t = 0; t < c.size(); ++t)
                if (c[t]) nd[u + t] = nd[u + t] + dist[u] * static_cast<double>(c[t]);
        }
        dist.swap(nd);
        Acc tc;
        for (std::uint64_t q : c) tc.add(static_cast<double>(q));
        pt.total = pt.total * tc.value();
    }
    pt.up.assign(dist.size(), DD{0.0, 0.0});
    Acc a;
    for (std::size_t s = dist.size(); s-- > 0;) {
        a.add(dist[s]);
        pt.up[s] = a.value();
    }
    return pt;
}

// the distribution without ties, by (m, n): made once
const PageTail& page_plain(std::int64_t m, std::int64_t n) {
    static std::mutex lock;
    static std::map<std::pair<std::int64_t, std::int64_t>, std::unique_ptr<const PageTail>> made;
    const std::lock_guard<std::mutex> gd(lock);
    auto it = made.find({m, n});
    if (it == made.end()) {
        std::vector<std::int64_t> row(static_cast<std::size_t>(n));
        for (std::int64_t j = 0; j < n; ++j)
            row[static_cast<std::size_t>(j)] = j + 1;  // the doubled ranks 2..2n over g = 2
        const std::vector<std::vector<std::int64_t>> rows(static_cast<std::size_t>(m), row);
        it = made.emplace(std::make_pair(m, n), std::make_unique<const PageTail>(page_tail(rows, 2))).first;
    }
    return *it->second;
}

nb::tuple page(
    nb::handle data, std::int64_t mrows, std::int64_t ncols, bool ranked, nb::handle pred_o, int req, double cap
) {
    const Rows a(rows_arg(data));
    const NDArray& pa = require_ndarray(pred_o);
    std::vector<double> pred(static_cast<std::size_t>(ncols));
    for (std::int64_t j = 0; j < ncols; ++j)
        pred[static_cast<std::size_t>(j)] = *reinterpret_cast<const double*>(pa.data + j * pa.strides[0]);
    const bool shape_exact = !(ncols > 8 || (mrows > 12 && ncols > 3) || mrows > 20);
    const bool may = req == kExactReq || (req == kAuto && shape_exact);
    const double nbits = static_cast<double>(mrows) * std::lgamma(static_cast<double>(ncols) + 1.0) / std::log(2.0);
    Outs out(5, a.lanes);
    each_lane(a.lanes, 16 * a.n, [&](std::int64_t r) {
        std::vector<double> colsum(static_cast<std::size_t>(ncols), 0.0), r2(static_cast<std::size_t>(ncols));
        std::vector<std::pair<double, std::int64_t>> v(static_cast<std::size_t>(ncols));
        std::vector<std::vector<std::int64_t>> rows;
        bool integral = true, plain = true;  // plain: every row's doubled values are 2, 4, ..., 2n
        for (std::int64_t i = 0; i < mrows; ++i) {
            if (ranked) {
                for (std::int64_t j = 0; j < ncols; ++j) r2[static_cast<std::size_t>(j)] = 2.0 * a.at(r, i * ncols + j);
            } else {
                for (std::int64_t j = 0; j < ncols; ++j) v[static_cast<std::size_t>(j)] = {a.at(r, i * ncols + j), j};
                std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
                for (std::int64_t q = 0; q < ncols;) {
                    std::int64_t e = q + 1;
                    while (e < ncols && v[static_cast<std::size_t>(e)].first == v[static_cast<std::size_t>(q)].first)
                        ++e;
                    for (std::int64_t u = q; u < e; ++u)
                        r2[static_cast<std::size_t>(v[static_cast<std::size_t>(u)].second)] =
                            static_cast<double>(q + e + 1);
                    plain = plain && e == q + 1;
                    q = e;
                }
            }
            for (std::int64_t j = 0; j < ncols; ++j) {
                const double x = r2[static_cast<std::size_t>(j)];
                colsum[static_cast<std::size_t>(j)] += x;
                integral = integral && x == std::floor(x) && std::fabs(x) < 0x1p40;
            }
            if (may && integral) {
                std::vector<std::int64_t> key(static_cast<std::size_t>(ncols));
                for (std::int64_t j = 0; j < ncols; ++j)
                    key[static_cast<std::size_t>(j)] = static_cast<std::int64_t>(r2[static_cast<std::size_t>(j)]);
                std::sort(key.begin(), key.end());
                if (ranked)
                    for (std::int64_t j = 0; j < ncols; ++j)
                        plain = plain && key[static_cast<std::size_t>(j)] == 2 * (j + 1);
                rows.push_back(std::move(key));
            }
        }
        plain = plain && integral;
        Acc twoL;
        for (std::int64_t j = 0; j < ncols; ++j)
            twoL.add(dd::two_prod(pred[static_cast<std::size_t>(j)], colsum[static_cast<std::size_t>(j)]));
        const DD L2 = twoL.value();
        // the exact work: each distinct row's subset programme, then the convolutions
        double work = 0.0;
        std::int64_t g = 0;
        if (may && integral && !plain) {
            for (const auto& row : rows)
                for (std::int64_t x : row) g = std::gcd(g, x);
            g = g == 0 ? 1 : g;
            std::map<std::vector<std::int64_t>, int> distinct;
            double range = 0.0;
            for (auto& row : rows) {
                for (std::int64_t& x : row) x /= g;
                std::int64_t lo = 0, hi = 0;
                for (std::int64_t j = 0; j < ncols; ++j)
                    lo += (j + 1) * row[static_cast<std::size_t>(ncols - 1 - j)],
                        hi += (j + 1) * row[static_cast<std::size_t>(j)];
                if (distinct.emplace(row, 0).second)
                    work += std::ldexp(
                        static_cast<double>(ncols) * static_cast<double>(hi + 1),
                        static_cast<int>(std::min<std::int64_t>(ncols, 60))
                    );
                range += static_cast<double>(hi - lo + 1);
                work += static_cast<double>(hi - lo + 1) * range;
            }
        } else if (may && plain) {
            const double R = static_cast<double>(ncols) * static_cast<double>(ncols * ncols - 1) / 6.0 +
                             1.0;  // the row's range of L
            work = std::ldexp(static_cast<double>(ncols) * R, static_cast<int>(std::min<std::int64_t>(ncols, 60))) +
                   R * R * static_cast<double>(mrows) * static_cast<double>(mrows) / 2.0;
        }
        if (may && (!integral || nbits > 1000.0 || ncols > 20)) work = std::numeric_limits<double>::infinity();
        const bool use_exact = req == kExactReq || (may && work <= cap);
        double p = kNaN, w = 0.0;
        if (use_exact && work > cap) {
            w = work;
        } else if (use_exact) {
            p = plain ? page_plain(mrows, ncols).p(L2) : page_tail(rows, g).p(L2);
        } else {
            const double nn = static_cast<double>(ncols), mm = static_cast<double>(mrows);
            const DD num = L2 * 2.0 - DD{mm * nn * (nn + 1.0) * (nn + 1.0), 0.0};  // 4 (L - E0)
            const DD V0 = dd_of(static_cast<i128>(mrows) * ncols * ncols * (ncols + 1) * (ncols * ncols - 1)) / 144.0;
            p = norm_sf(ratio(num / 4.0, sqrt_dd(V0)));
        }
        out.p[0][r] = 0.5 * to_double(L2), out.p[1][r] = p, out.p[2][r] = use_exact ? kUsedExact : kUsedAsymptotic;
        out.p[3][r] = w, out.p[4][r] = integral ? 0.0 : 1.0;
    });
    return out.tuple();
}

// The exact counts against closed forms (tests): kind 0, #{U <= b} of [N, c]_q; kind 1, Ansari's
// #{AB <= b} for c of N; kind 2, C(N, c); as doubles.
double exact_count(int kind, std::int64_t N, std::int64_t c, std::int64_t b) {
    switch (kind) {
        case 0: return to_double(exact::to_dd(exact::mwu_lower(N, c, b)));
        case 1: return to_double(exact::to_dd(exact::AnsariCounts(N / 2).lower(N, c, b)));
        default: return to_double(exact::to_dd(exact::binom_u128(N, c)));
    }
}

}  // namespace

void bind_rank(nb::module_& m) {
    m.def("mwu", &mwu);
    m.def("ranksums", &ranksums);
    m.def(
        "wilcoxon", &wilcoxon, nb::arg("x"), nb::arg("y").none(), nb::arg("zero_method"), nb::arg("correction"),
        nb::arg("alternative"), nb::arg("method"), nb::arg("cap"), nb::arg("omit")
    );
    m.def("ansari", &ansari);
    m.def("mood", &mood);
    m.def("brunnermunzel", &brunnermunzel);
    m.def("kruskal", &kruskal);
    m.def("friedman", &friedman);
    m.def("median_test", &median_test);
    m.def("page", &page);
    m.def("exact_count", &exact_count);
}

}  // namespace sd::htest
