// The tests of goodness of fit between samples (milestone H4, docs/design/htest.md §3.5).
//
// The Python side (stardust.htest._gof) checks the arguments and arranges the samples as 2-D
// float64 arrays, one lane (one test of a batch) per row; each kernel returns the per-lane
// results as 1-D arrays. Lanes are independent: the same lane gives the same bits alone or in a
// batch, on any thread count.
//
//   ks2       ks_2samp: D, p-value, the location and sign of D, the method used, the exact work
//   cvm2      cramervonmises_2samp: T, p-value, the method used, the exact work
//   ad_ksamp  anderson_ksamp (one test, k samples): the standardized statistic, p-value
//   epps      epps_singleton_2samp: the statistic W, p-value, df (the pinv's rank: H59)
//
// The statistics are exact where they are rational in the ranks: D = h/lcm(n1, n2) from integer
// gaps, the Cramér-von Mises U from doubled midranks (4U an integer), the Anderson-Darling sums
// from integer counts (each term one double-double quotient). The exact null distributions are
// htest/ks2.hpp's and htest/cvm2.hpp's, conditional on the ties (H55, H58). The p-values are sd.stats' ChiSquared, the
// one-sample Kolmogorov distribution (htest/ks.hpp: scipy's asymptotic two-sample p-value is
// kstwo at round(n1 n2 / (n1 + n2))), Csörgő and Faraway's limit V (htest/cvm.hpp), and scipy's
// interpolation of Scholz and Stephens' table. Degenerate lanes give NaN without warnings (H11).
#include "htest/gof2.hpp"

#include "htest/cvm.hpp"
#include "htest/cvm2.hpp"
#include "htest/ks.hpp"
#include "htest/ks2.hpp"
#include "htest/lanes.hpp"
#include "htest/moments.hpp"
#include "htest/rank_exact.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

namespace m = stats::method;
using exact::i128;

double chi2_sf(double x, double df) { return stats::method_of<stats::ChiSquared, m::Ccdf>(x, df); }

enum Alt { kTwoSided = 0, kLess = 1, kGreater = 2 };
enum Requested { kAuto = 0, kExactReq = 1, kAsymptoticReq = 2 };
constexpr double kUsedExact = 0.0, kUsedAsymptotic = 1.0;
constexpr double kStateCap = 1e7;  // the exact distributions' entries held at once (H13, with the work cap)
constexpr double kPlanStates = 4e6;  // the Cramér-von Mises plan's states (blocks x counts) at most
constexpr double kTryBytes = 256.0 * 1024 * 1024;  // a try past the plan's bound: its storage at most (H152)
std::mutex& try_mutex() {
    static std::mutex m;
    return m;
}

DD dd_of(i128 v) { return exact::to_dd(v); }
double ratio(DD a, DD b) { return to_double(a / b); }

// sqrt of a nonnegative double-double: one Newton correction of sqrt(hi)
DD sqrt_dd(DD a) {
    if (!(a.hi > 0.0)) return DD{a.hi == 0.0 ? 0.0 : kNaN, 0.0};
    const double s = std::sqrt(a.hi);
    const DD s2 = dd::two_prod(s, s);
    return dd::two_sum(s, ((a.hi - s2.hi) - s2.lo + a.lo) / (2.0 * s));
}

// A lane's values without NaNs; true when one was met.
bool values(const Rows& a, std::int64_t r, std::vector<double>& v) {
    v.clear();
    bool nan = false;
    for (std::int64_t j = 0; j < a.n; ++j) {
        const double x = a.at(r, j);
        if (x != x) nan = true;
        else v.push_back(x);
    }
    return nan;
}

// ---- ks_2samp ----------------------------------------------------------------------------
//
// The empirical distribution functions at each distinct pooled value v (right-continuous: the
// counts i of x <= v and j of y <= v), g = i a - j b in units of 1/L, L = lcm(n1, n2). D+ = max g,
// D- = -min g (both at least 0: g = 0 at the last value), the location the first value where the
// extreme is reached (scipy's argmax over the pooled sorted sample); two-sided D is D- when
// D- > D+ (sign -1), else D+ (sign +1), as scipy's. method 'auto' is exact up to
// max(n1, n2) = 10000 (scipy's MAX_AUTO_N; the work there stays within the cap); with ties the
// exact distribution is conditional on them (H55). The asymptotic p-values are scipy's: two-sided
// kstwo.sf(D, round(n1 n2 / (n1 + n2))), one-sided Hodges' exp(-2 z^2 - 2 z (m + 2n) /
// (3 sqrt(m n (m + n)))), z = D sqrt(m n / (m + n)), m >= n.

constexpr std::int64_t kMaxAutoN = 10000;

struct KsLane {
    double D = kNaN, p = kNaN, loc = kNaN, sign = 0.0, method = kNaN, work = 0.0;
};

KsLane ks2_lane(std::vector<double>& x, std::vector<double>& y, int alt, int req, double cap, bool nan) {
    KsLane o;
    const auto n1 = static_cast<std::int64_t>(x.size()), n2 = static_cast<std::int64_t>(y.size());
    const bool may = req == kExactReq || (req == kAuto && std::max(n1, n2) <= kMaxAutoN);
    o.method = may ? kUsedExact : kUsedAsymptotic;
    if (nan || n1 == 0 || n2 == 0) return o;
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    const std::int64_t L = std::lcm(n1, n2), a = L / n1, b = L / n2;
    std::int64_t i = 0, j = 0, gmax = 0, gmin = 0;
    double vmax = kNaN, vmin = kNaN;
    bool first = true, ties = false;
    std::vector<std::int64_t> blocks;
    while (i < n1 || j < n2) {
        const double v = (j >= n2 || (i < n1 && x[static_cast<std::size_t>(i)] <= y[static_cast<std::size_t>(j)]))
                             ? x[static_cast<std::size_t>(i)]
                             : y[static_cast<std::size_t>(j)];
        const std::int64_t i0 = i, j0 = j;
        while (i < n1 && x[static_cast<std::size_t>(i)] == v) ++i;
        while (j < n2 && y[static_cast<std::size_t>(j)] == v) ++j;
        const std::int64_t t = (i - i0) + (j - j0);
        ties = ties || t > 1;
        blocks.push_back(t);
        const std::int64_t g = i * a - j * b;
        if (first || g > gmax) gmax = g, vmax = v;
        if (first || g < gmin) gmin = g, vmin = v;
        first = false;
    }
    std::int64_t h;
    if (alt == kGreater) {
        h = gmax, o.loc = vmax, o.sign = 1.0;
    } else if (alt == kLess) {
        h = -gmin, o.loc = vmin, o.sign = -1.0;
    } else if (-gmin > gmax) {
        h = -gmin, o.loc = vmin, o.sign = -1.0;
    } else {
        h = gmax, o.loc = vmax, o.sign = 1.0;
    }
    o.D = static_cast<double>(h) / static_cast<double>(L);
    if (may) {
        ks2::Problem P;
        P.na = n1, P.nb = n2, P.h = h, P.alt = alt, P.blocks = ties ? &blocks : nullptr;
        const bool closed = !ties && n1 == n2;
        const double work = closed ? static_cast<double>(n1) : ks2::work(P);
        if (work > cap) {
            if (req == kExactReq) {
                o.work = work;
                return o;
            }
            o.method = kUsedAsymptotic;
        } else {
            o.p = closed ? ks2::equal_tail(n1, h, alt == kTwoSided) : ks2::tail(P);
            return o;
        }
    }
    const double mm = static_cast<double>(std::max(n1, n2)), nn = static_cast<double>(std::min(n1, n2));
    const double en = mm * nn / (mm + nn);
    if (alt == kTwoSided) {
        o.p = ks::kstwo_sf(static_cast<std::int64_t>(std::nearbyint(en)), o.D);
    } else {
        const double z = std::sqrt(en) * o.D;
        const double expt = -2.0 * z * z - 2.0 * z * (mm + 2.0 * nn) / std::sqrt(mm * nn * (mm + nn)) / 3.0;
        o.p = std::exp(expt);
    }
    o.p = std::clamp(o.p, 0.0, 1.0);
    return o;
}

nb::tuple ks2_test(nb::handle xo, nb::handle yo, int alt, int req, double cap, bool omit) {
    const Rows x(rows_arg(xo)), y(rows_arg(yo));
    Outs out(6, x.lanes);
    each_lane(x.lanes, 32 * (x.n + y.n), [&](std::int64_t r) {
        std::vector<double> xv, yv;
        const bool nx_nan = values(x, r, xv), ny_nan = values(y, r, yv);
        const bool nan = nx_nan || ny_nan;
        const KsLane l = ks2_lane(xv, yv, alt, req, cap, nan && !omit);
        out.p[0][r] = l.D, out.p[1][r] = l.p, out.p[2][r] = l.loc, out.p[3][r] = l.sign;
        out.p[4][r] = l.method, out.p[5][r] = l.work;
    });
    return out.tuple();
}

// ---- cramervonmises_2samp -------------------------------------------------------------------
//
// 4U = nx sum_x (2r - 2i)^2 + ny sum_y (2s - 2j)^2 from the doubled midranks (an integer,
// 128-bit), T = (6 (4U) - 4k(4k - 1)) / (24 k N) rounded once. method 'auto' is exact up to
// max(nx, ny) = 20 (scipy's). The asymptotic p-value is scipy's: tn = 1/6 + (T - E T) /
// sqrt(45 Var T) (Anderson's eq. 11, 14, 15; T - E T an exact quotient of integers here),
// max(0, 1 - V(tn)) for tn >= 0.003 and 1 below.

constexpr std::int64_t kCvmAutoN = 20;

// sum_{m=1..k} (A - 2m)^2 in 128 bits
i128 run_sum128(i128 A, i128 k) { return k * A * A - 2 * A * k * (k + 1) + 2 * k * (k + 1) * (2 * k + 1) / 3; }

struct CvmLane {
    double T = kNaN, p = kNaN, method = kNaN, work = 0.0;
};

CvmLane cvm2_lane(std::vector<double>& x, std::vector<double>& y, int req, double cap, bool nan) {
    CvmLane o;
    const auto nx = static_cast<std::int64_t>(x.size()), ny = static_cast<std::int64_t>(y.size());
    const bool may = req == kExactReq || (req == kAuto && std::max(nx, ny) <= kCvmAutoN);
    o.method = may ? kUsedExact : kUsedAsymptotic;
    if (nan || nx < 2 || ny < 2) return o;
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    const std::int64_t N = nx + ny;
    i128 U4 = 0;
    std::int64_t i = 0, j = 0;
    bool ties = false;
    std::vector<std::int64_t> blocks;
    while (i < nx || j < ny) {
        const double v = (j >= ny || (i < nx && x[static_cast<std::size_t>(i)] <= y[static_cast<std::size_t>(j)]))
                             ? x[static_cast<std::size_t>(i)]
                             : y[static_cast<std::size_t>(j)];
        const std::int64_t i0 = i, j0 = j;
        while (i < nx && x[static_cast<std::size_t>(i)] == v) ++i;
        while (j < ny && y[static_cast<std::size_t>(j)] == v) ++j;
        const std::int64_t t = (i - i0) + (j - j0), p0 = i0 + j0;
        const i128 twoR = 2 * static_cast<i128>(p0) + t + 1;
        U4 += static_cast<i128>(nx) * run_sum128(twoR - 2 * static_cast<i128>(i0), i - i0) +
              static_cast<i128>(ny) * run_sum128(twoR - 2 * static_cast<i128>(j0), j - j0);
        ties = ties || t > 1;
        blocks.push_back(t);
    }
    const i128 k = static_cast<i128>(nx) * ny, NN = N;
    o.T = ratio(dd_of(6 * U4 - 4 * k * (4 * k - 1)), dd_of(24 * k * NN));
    if (may) {
        const std::int64_t G = std::gcd(nx, ny);
        const i128 v = U4 / G;
        double work = std::numeric_limits<double>::infinity();
        cvm2::Problem P;
        P.nx = nx, P.ny = ny, P.blocks = ties ? &blocks : nullptr;
        cvm2::Plan plan;
        if (v < (static_cast<i128>(1) << 60) && N < (1 << 20)) {
            P.v = static_cast<std::int64_t>(v);
            plan = cvm2::plan(P, kPlanStates);
            work = plan.held > kStateCap ? std::max(plan.entries, plan.held * 100.0) : plan.entries;
        }
        if (work <= cap) {
            o.p = cvm2::tail(plan);
            return o;
        }
        if (req == kExactReq) {
            // the plan's bound is far above the work usually done: try within the cap and
            // kTryBytes of storage, one lane at a time (H152: the lanes of a batch run on the pool,
            // and each try may hold its whole storage bound)
            double p = -1.0;
            if (plan.ok) {
                const std::lock_guard<std::mutex> one(try_mutex());
                p = cvm2::tail(plan, cap, kTryBytes);
            }
            if (p >= 0.0) {
                o.p = p;
                return o;
            }
            o.work = work;
            return o;
        }
        o.method = kUsedAsymptotic;
    }
    // T - E T = (6 (4U) - 4k(4k - 1) - 4k(N + 1)) / (24 k N); 45 Var T = num / (4 N^2 k)
    const DD diff = dd_of(6 * U4 - 4 * k * (4 * k - 1) - 4 * k * (NN + 1)) / dd_of(24 * k * NN);
    const i128 vnum = (NN + 1) * (4 * k * NN - 3 * (static_cast<i128>(nx) * nx + static_cast<i128>(ny) * ny) - 2 * k);
    const DD var45 = dd_of(vnum) / dd_of(4 * NN * NN * k);
    const DD tn = DD{1.0, 0.0} / 6.0 + diff / sqrt_dd(var45);
    const double t = to_double(tn);
    o.p = t >= 0.003 ? std::max(0.0, 1.0 - cvm::cdf_inf(t)) : (t == t ? 1.0 : kNaN);
    return o;
}

nb::tuple cvm2_test(nb::handle xo, nb::handle yo, int req, double cap, bool omit) {
    const Rows x(rows_arg(xo)), y(rows_arg(yo));
    Outs out(4, x.lanes);
    each_lane(x.lanes, 32 * (x.n + y.n), [&](std::int64_t r) {
        std::vector<double> xv, yv;
        const bool nx_nan = values(x, r, xv), ny_nan = values(y, r, yv);
        const bool nan = nx_nan || ny_nan;
        const CvmLane l = cvm2_lane(xv, yv, req, cap, nan && !omit);
        out.p[0][r] = l.T, out.p[1][r] = l.p, out.p[2][r] = l.method, out.p[3][r] = l.work;
    });
    return out.tuple();
}

// ---- anderson_ksamp ------------------------------------------------------------------------
//
// Scholz and Stephens' (1987) A2kN (variant 'midrank': their A2akN, eq. 7; 'right': eq. 6;
// 'continuous': eq. 3), each term one double-double quotient of exact integers (the doubled
// midrank counts of eq. 7 are integers), summed in double-double; its standardization
// (A2kN - (k - 1)) / sigma_N with sigma_N^2 from the harmonic sums H, h and g in double-double;
// the p-value scipy's: below the table's smallest critical value 0.25, above its largest 0.001,
// between them exp of the least-squares quadratic of log(level) on the critical values
// b0 + b1/sqrt(k - 1) + b2/(k - 1) (their Table 2; np.polyfit's fit, solved here exactly in
// double-double from the doubles numpy forms).

enum Variant { kMidrank = 0, kRight = 1, kContinuous = 2 };

// Solve the 3 x 3 system A z = r in double-doubles (Gaussian elimination, partial pivoting).
bool solve3(std::array<std::array<DD, 3>, 3> A, std::array<DD, 3> r, std::array<DD, 3>& z) {
    for (int c = 0; c < 3; ++c) {
        int piv = c;
        for (int i = c + 1; i < 3; ++i)
            if (std::fabs(A[i][c].hi) > std::fabs(A[piv][c].hi)) piv = i;
        if (A[piv][c].hi == 0.0) return false;
        std::swap(A[c], A[piv]);
        std::swap(r[c], r[piv]);
        for (int i = c + 1; i < 3; ++i) {
            const DD f = A[i][c] / A[c][c];
            for (int k = c; k < 3; ++k) A[i][k] = A[i][k] - f * A[c][k];
            r[i] = r[i] - f * r[c];
        }
    }
    for (int c = 2; c >= 0; --c) {
        DD s = r[c];
        for (int k = c + 1; k < 3; ++k) s = s - A[c][k] * z[k];
        z[c] = s / A[c][c];
    }
    return true;
}

double ad_ksamp_p(double A2, std::int64_t k) {
    static constexpr double b0[] = {0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085};
    static constexpr double b1[] = {-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615};
    static constexpr double b2[] = {-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154};
    static constexpr double sig[] = {0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001};
    if (A2 != A2) return kNaN;
    const double mm = static_cast<double>(k - 1), sm = std::sqrt(mm);
    double crit[7], cmin = std::numeric_limits<double>::infinity(), cmax = -cmin;
    for (int l = 0; l < 7; ++l) {
        crit[l] = b0[l] + b1[l] / sm + b2[l] / mm;
        cmin = std::min(cmin, crit[l]), cmax = std::max(cmax, crit[l]);
    }
    if (A2 < cmin) return 0.25;
    if (A2 > cmax) return 0.001;
    // the least-squares quadratic log(sig) ~ c0 + c1 x + c2 x^2 by its normal equations
    std::array<DD, 5> sx{};
    std::array<DD, 3> sy{};
    for (int l = 0; l < 7; ++l) {
        const DD x{crit[l], 0.0}, y = dd::log(sig[l]);
        DD xp{1.0, 0.0};
        for (int e = 0; e < 5; ++e) {
            sx[static_cast<std::size_t>(e)] = sx[static_cast<std::size_t>(e)] + xp;
            if (e < 3) sy[static_cast<std::size_t>(e)] = sy[static_cast<std::size_t>(e)] + xp * y;
            xp = xp * x;
        }
    }
    std::array<std::array<DD, 3>, 3> A;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) A[i][j] = sx[static_cast<std::size_t>(i + j)];
    std::array<DD, 3> c;
    if (!solve3(A, sy, c)) return kNaN;
    const DD X{A2, 0.0};
    const DD e = c[0] + X * (c[1] + X * c[2]);
    return std::exp(e.hi) * (1.0 + e.lo);
}

// One lane r of the samples' rows: (statistic, p-value) into o[0], o[1].
void ad_lane(const std::vector<Rows>& rows, std::int64_t r, int variant, double* o0, double* o1) {
    std::vector<std::vector<double>> s;
    bool nan = false;
    for (const Rows& a : rows) {
        s.emplace_back();
        nan = values(a, r, s.back()) || nan;
        std::sort(s.back().begin(), s.back().end());
    }
    const auto k = static_cast<std::int64_t>(s.size());
    struct {
        double* p[2];
    } out{{o0, o1}};
    *o0 = *o1 = kNaN;
    std::int64_t N = 0;
    bool empty = false;
    for (const auto& v : s) N += static_cast<std::int64_t>(v.size()), empty = empty || v.empty();
    if (nan || empty || N < 4) return;
    std::vector<double> Z;
    Z.reserve(static_cast<std::size_t>(N));
    for (const auto& v : s) Z.insert(Z.end(), v.begin(), v.end());
    std::sort(Z.begin(), Z.end());
    // the distinct values and their multiplicities
    std::vector<double> zs;
    std::vector<std::int64_t> l;
    for (double z : Z) {
        if (!zs.empty() && zs.back() == z) ++l.back();
        else zs.push_back(z), l.push_back(1);
    }
    const auto J = static_cast<std::int64_t>(zs.size());
    if (J < 2) return;
    const double Nd = static_cast<double>(N);
    Acc tot;
    for (std::int64_t i = 0; i < k; ++i) {
        const auto& v = s[static_cast<std::size_t>(i)];
        const auto ni = static_cast<std::int64_t>(v.size());
        Acc si;
        std::size_t q = 0;  // v[0..q) <= the current value
        if (variant == kContinuous) {
            // j = 1..N-1 over the pooled sorted values (ties repeated), M = #v <= Z_j
            for (std::int64_t jj = 1; jj < N; ++jj) {
                const double z = Z[static_cast<std::size_t>(jj - 1)];
                while (q < v.size() && v[q] <= z) ++q;
                const i128 X = static_cast<i128>(N) * static_cast<i128>(q) - static_cast<i128>(jj) * ni;
                si.add(dd_of(X * X) / dd::two_prod(static_cast<double>(jj), static_cast<double>(N - jj)));
            }
        } else {
            std::int64_t left = 0;
            for (std::int64_t jj = 0; jj < J; ++jj) {
                const double z = zs[static_cast<std::size_t>(jj)];
                const std::size_t q0 = q;
                while (q < v.size() && v[q] < z) ++q;
                const std::size_t qlt = q;
                while (q < v.size() && v[q] <= z) ++q;
                (void)q0;
                const std::int64_t lj = l[static_cast<std::size_t>(jj)];
                if (variant == kMidrank) {
                    // eq. 7 doubled: X = N (2M - f) - (2 left + l) n_i, D = (2B)(2N - 2B) - N l
                    const i128 twoM = static_cast<i128>(q) + static_cast<i128>(qlt), twoB = 2 * left + lj;
                    const i128 X = static_cast<i128>(N) * twoM - twoB * ni;
                    const i128 D = twoB * (2 * static_cast<i128>(N) - twoB) - static_cast<i128>(N) * lj;
                    si.add(dd_of(X * X) * static_cast<double>(lj) / (dd_of(D) * Nd));
                } else if (jj + 1 < J) {
                    // eq. 6 over the distinct values but the last: B the pooled count <= z
                    const i128 B = left + lj;
                    const i128 X = static_cast<i128>(N) * static_cast<i128>(q) - B * ni;
                    si.add(dd_of(X * X) * static_cast<double>(lj) / (dd_of(B * (static_cast<i128>(N) - B)) * Nd));
                }
                left += lj;
            }
        }
        tot.add(si.value() / static_cast<double>(ni));
    }
    DD A2kN = tot.value();
    if (variant == kMidrank) A2kN = A2kN * (Nd - 1.0) / Nd;
    if (variant == kContinuous) A2kN = A2kN / Nd;
    // sigma_N^2: H = sum 1/n_i, h = sum_{i<N} 1/i, g = sum_{i <= N-2} sum_{j=i+1}^{N-1} 1/((N - i) j)
    Acc Hs, hs, gs;
    for (const auto& v : s) Hs.add(DD{1.0, 0.0} / static_cast<double>(v.size()));
    std::vector<DD> suffix(static_cast<std::size_t>(N + 1), DD{0.0, 0.0});  // sum_{j' >= j}^{N-1} 1/j'
    {
        Acc run;
        for (std::int64_t jj = N - 1; jj >= 1; --jj) {
            run.add(DD{1.0, 0.0} / static_cast<double>(jj));
            suffix[static_cast<std::size_t>(jj)] = run.value();
        }
        hs.add(suffix[1]);
    }
    for (std::int64_t ii = 1; ii <= N - 2; ++ii)
        gs.add(suffix[static_cast<std::size_t>(ii + 1)] / static_cast<double>(N - ii));
    const DD H = Hs.value(), hh = hs.value(), g = gs.value();
    const double kd = static_cast<double>(k);
    const DD one{1.0, 0.0};
    const DD a = (g * 4.0 - 6.0) * (kd - 1.0) + (one * 10.0 - g * 6.0) * H;
    const DD b =
        (g * 2.0 - 4.0) * (kd * kd) + hh * (8.0 * kd) + (g * 2.0 - hh * 14.0 - 4.0) * H - hh * 8.0 + g * 4.0 - 6.0;
    const DD c =
        (hh * 6.0 + g * 2.0 - 2.0) * (kd * kd) + (hh * 4.0 - g * 4.0 + 6.0) * kd + (hh * 2.0 - 6.0) * H + hh * 4.0;
    const DD d = (hh * 2.0 + 6.0) * (kd * kd) - hh * (4.0 * kd);
    const DD Nn{Nd, 0.0};
    const DD num = ((a * Nn + b) * Nn + c) * Nn + d;
    const DD den = dd::two_prod(Nd - 1.0, Nd - 2.0) * (Nd - 3.0);
    const DD sig2 = num / den;
    const DD A2 = (A2kN - (kd - 1.0)) / sqrt_dd(sig2);
    *out.p[0] = to_double(A2);
    *out.p[1] = ad_ksamp_p(*out.p[0], k);
}

// samples: k (lanes, n_i) arrays (lanes equal: one test per lane; anderson_ksamp's resamples).
nb::tuple ad_ksamp(nb::list samples, int variant) {
    std::vector<Rows> rows;
    std::int64_t n = 0;
    for (nb::handle h : samples) rows.emplace_back(rows_arg(h)), n += rows.back().n;
    const std::int64_t lanes = rows.empty() ? 0 : rows.front().lanes;
    Outs out(2, lanes);
    each_lane(lanes, 32 * n + 256, [&](std::int64_t r) { ad_lane(rows, r, variant, out.p[0] + r, out.p[1] + r); });
    return out.tuple();
}

// ---- epps_singleton_2samp -------------------------------------------------------------------
//
// scipy's: sigma = iqr(pooled)/2 (numpy's linear percentiles, here in double-double), the
// points t/sigma, g(x) = (cos(t_k x / sigma), sin(t_k x / sigma))_k (the arguments as
// double-doubles), the biased covariances of g over each sample and est = (N/nx) cov_x +
// (N/ny) cov_y, the difference d of the two means of g, and W = N d' est^+ d. est^+ is numpy's
// pinv (singular values at most 1e-15 of the largest left out); the degrees of freedom are
// numpy's matrix_rank of it. With every value kept, est^+ d is solved in double-double. For
// max(nx, ny) < 25 W is multiplied by Epps and Singleton's small-sample correction
// 1 / (1 + N^-0.45 + 10.1 (nx^-1.7 + ny^-1.7)); the p-value is chi^2 with that many degrees of
// freedom. Fewer than 5 values in a sample, a value that is not finite, or an IQR of 0 give NaN.

// numpy's linear percentile (q in [0, 1]) of sorted values, in double-double
DD percentile(const std::vector<double>& z, double q) {
    const double idx = q * static_cast<double>(z.size() - 1);
    const double lo = std::floor(idx), frac = idx - lo;
    const auto i = static_cast<std::size_t>(lo);
    if (frac == 0.0 || i + 1 >= z.size()) return {z[i], 0.0};
    const DD diff = dd::two_sum(z[i + 1], -z[i]);
    return DD{z[i], 0.0} + diff * frac;
}

// cyclic Jacobi on a symmetric matrix: eigenvalues ev, eigenvectors in V's columns
void jacobi(std::vector<double> A, int n, std::vector<double>& ev, std::vector<double>& V) {
    V.assign(static_cast<std::size_t>(n * n), 0.0);
    for (int i = 0; i < n; ++i) V[static_cast<std::size_t>(i * n + i)] = 1.0;
    auto at = [&](std::vector<double>& M, int i, int j) -> double& { return M[static_cast<std::size_t>(i * n + j)]; };
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0, diag = 0.0;
        for (int i = 0; i < n; ++i) {
            diag += at(A, i, i) * at(A, i, i);
            for (int j = i + 1; j < n; ++j) off += at(A, i, j) * at(A, i, j);
        }
        if (off <= 1e-40 * diag || off == 0.0) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = at(A, p, q);
                if (apq == 0.0) continue;
                const double theta = (at(A, q, q) - at(A, p, p)) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = at(A, k, p), akq = at(A, k, q);
                    at(A, k, p) = c * akp - s * akq;
                    at(A, k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = at(A, p, k), aqk = at(A, q, k);
                    at(A, p, k) = c * apk - s * aqk;
                    at(A, q, k) = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = at(V, k, p), vkq = at(V, k, q);
                    at(V, k, p) = c * vkp - s * vkq;
                    at(V, k, q) = s * vkp + c * vkq;
                }
            }
        }
    }
    ev.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) ev[static_cast<std::size_t>(i)] = at(A, i, i);
}

// solve the symmetric system S z = r in double-doubles (Gaussian elimination, partial pivoting)
bool solve_dd(std::vector<DD> S, std::vector<DD> r, int n, std::vector<DD>& z) {
    auto at = [&](int i, int j) -> DD& { return S[static_cast<std::size_t>(i * n + j)]; };
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int i = c + 1; i < n; ++i)
            if (std::fabs(at(i, c).hi) > std::fabs(at(piv, c).hi)) piv = i;
        if (at(piv, c).hi == 0.0) return false;
        if (piv != c) {
            for (int k = 0; k < n; ++k) std::swap(at(c, k), at(piv, k));
            std::swap(r[static_cast<std::size_t>(c)], r[static_cast<std::size_t>(piv)]);
        }
        for (int i = c + 1; i < n; ++i) {
            const DD f = at(i, c) / at(c, c);
            for (int k = c; k < n; ++k) at(i, k) = at(i, k) - f * at(c, k);
            r[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(i)] - f * r[static_cast<std::size_t>(c)];
        }
    }
    z.assign(static_cast<std::size_t>(n), DD{0.0, 0.0});
    for (int c = n - 1; c >= 0; --c) {
        DD s = r[static_cast<std::size_t>(c)];
        for (int k = c + 1; k < n; ++k) s = s - at(c, k) * z[static_cast<std::size_t>(k)];
        z[static_cast<std::size_t>(c)] = s / at(c, c);
    }
    return true;
}

struct EsLane {
    double w = kNaN, p = kNaN, df = kNaN;
};

// the g vectors of one sample (rows: cos for each t, then sin), their means and biased covariance
void es_moments(const std::vector<double>& x, const std::vector<DD>& ts, std::vector<DD>& mean, std::vector<DD>& cov) {
    const int T = static_cast<int>(ts.size()), m2 = 2 * T;
    const auto n = static_cast<std::int64_t>(x.size());
    std::vector<DD> g(static_cast<std::size_t>(m2 * n));
    for (std::int64_t i = 0; i < n; ++i) {
        for (int k = 0; k < T; ++k) {
            const DD arg = ts[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(i)];
            const double c = std::cos(arg.hi), s = std::sin(arg.hi);
            g[static_cast<std::size_t>(k * n + i)] = dd::two_sum(c, -s * arg.lo);
            g[static_cast<std::size_t>((T + k) * n + i)] = dd::two_sum(s, c * arg.lo);
        }
    }
    mean.assign(static_cast<std::size_t>(m2), DD{0.0, 0.0});
    for (int k = 0; k < m2; ++k) {
        Acc a;
        for (std::int64_t i = 0; i < n; ++i) a.add(g[static_cast<std::size_t>(k * n + i)]);
        mean[static_cast<std::size_t>(k)] = a.value() / static_cast<double>(n);
        for (std::int64_t i = 0; i < n; ++i)
            g[static_cast<std::size_t>(k * n + i)] =
                g[static_cast<std::size_t>(k * n + i)] - mean[static_cast<std::size_t>(k)];
    }
    cov.assign(static_cast<std::size_t>(m2 * m2), DD{0.0, 0.0});
    for (int k = 0; k < m2; ++k) {
        for (int l = k; l < m2; ++l) {
            Acc a;
            for (std::int64_t i = 0; i < n; ++i)
                a.add(g[static_cast<std::size_t>(k * n + i)] * g[static_cast<std::size_t>(l * n + i)]);
            const DD v = a.value() / static_cast<double>(n);
            cov[static_cast<std::size_t>(k * m2 + l)] = cov[static_cast<std::size_t>(l * m2 + k)] = v;
        }
    }
}

EsLane epps_lane(std::vector<double>& x, std::vector<double>& y, const std::vector<double>& t, bool nan) {
    EsLane o;
    const auto nx = static_cast<std::int64_t>(x.size()), ny = static_cast<std::int64_t>(y.size());
    if (nan || nx < 5 || ny < 5) return o;
    for (double v : x)
        if (!std::isfinite(v)) return o;
    for (double v : y)
        if (!std::isfinite(v)) return o;
    {  // a lane past H1's range in its unit of 2^e (W depends on x / IQR only: H160)
        double big = 0.0;
        for (double v : x) big = std::max(big, std::fabs(v));
        for (double v : y) big = std::max(big, std::fabs(v));
        const int e = unit_exponent(big);
        if (e) {
            const double f = std::ldexp(1.0, -std::clamp(e, -1000, 1000)),
                         f2 = std::ldexp(1.0, -e + std::clamp(e, -1000, 1000));
            for (double& v : x) v = v * f * f2;
            for (double& v : y) v = v * f * f2;
        }
    }
    // each sample constant (H20's rule): the g vectors are too, their covariance 0 but for the
    // rounding of the means (H153; on the values in their unit)
    if (moments_of(x.data(), nullptr, nx).constant() && moments_of(y.data(), nullptr, ny).constant()) return o;
    std::vector<double> z(x);
    z.insert(z.end(), y.begin(), y.end());
    std::sort(z.begin(), z.end());
    const DD iqr = percentile(z, 0.75) - percentile(z, 0.25);
    if (!(iqr.hi > 0.0)) return o;
    const DD sigma = iqr / 2.0;
    std::vector<DD> ts;
    for (double tk : t) ts.push_back(DD{tk, 0.0} / sigma);
    const int m2 = 2 * static_cast<int>(t.size());
    std::vector<DD> mx, cx, my, cy;
    es_moments(x, ts, mx, cx);
    es_moments(y, ts, my, cy);
    const double N = static_cast<double>(nx + ny);
    const DD fx = DD{N, 0.0} / static_cast<double>(nx), fy = DD{N, 0.0} / static_cast<double>(ny);
    std::vector<DD> S(static_cast<std::size_t>(m2 * m2)), d(static_cast<std::size_t>(m2));
    std::vector<double> Sd(static_cast<std::size_t>(m2 * m2));
    for (std::size_t i = 0; i < S.size(); ++i) {
        S[i] = fx * cx[i] + fy * cy[i];
        Sd[i] = to_double(S[i]);
    }
    for (int k = 0; k < m2; ++k)
        d[static_cast<std::size_t>(k)] = mx[static_cast<std::size_t>(k)] - my[static_cast<std::size_t>(k)];
    std::vector<double> ev, V;
    jacobi(Sd, m2, ev, V);
    double smax = 0.0;
    for (double e : ev) smax = std::max(smax, std::fabs(e));
    // the g vectors are bounded by 1: a covariance of the rounding's size (at most N eps^2) is 0
    if (!(smax > N * 0x1p-104) || !std::isfinite(smax)) return o;
    // numpy's pinv keeps the singular values above 1e-15 of the largest; matrix_rank of the pinv
    // counts its singular values (the kept 1/s) above max(1/s) m eps
    const double cut = 1e-15 * smax;
    double smin_kept = std::numeric_limits<double>::infinity();
    int kept = 0;
    for (double e : ev)
        if (std::fabs(e) > cut) ++kept, smin_kept = std::min(smin_kept, std::fabs(e));
    int rank = 0;
    const double tol = (1.0 / smin_kept) * static_cast<double>(m2) * 0x1p-52;
    for (double e : ev)
        if (std::fabs(e) > cut && 1.0 / std::fabs(e) > tol) ++rank;
    DD W{0.0, 0.0};
    std::vector<DD> zsol;
    if (kept == m2 && solve_dd(S, d, m2, zsol)) {
        Acc a;
        for (int k = 0; k < m2; ++k) a.add(d[static_cast<std::size_t>(k)] * zsol[static_cast<std::size_t>(k)]);
        W = a.value() * N;
    } else {
        Acc a;
        for (int c = 0; c < m2; ++c) {
            const double e = ev[static_cast<std::size_t>(c)];
            if (!(std::fabs(e) > cut)) continue;
            Acc u;
            for (int k = 0; k < m2; ++k)
                u.add(d[static_cast<std::size_t>(k)] * V[static_cast<std::size_t>(k * m2 + c)]);
            const DD uv = u.value();
            a.add(uv * uv / e);
        }
        W = a.value() * N;
    }
    if (std::max(nx, ny) < 25) {
        const double corr =
            1.0 / (1.0 + std::pow(N, -0.45) +
                   10.1 * (std::pow(static_cast<double>(nx), -1.7) + std::pow(static_cast<double>(ny), -1.7)));
        W = W * corr;
    }
    o.w = to_double(W);
    o.df = static_cast<double>(rank);
    o.p = rank > 0 ? chi2_sf(o.w, o.df) : kNaN;
    return o;
}

nb::tuple epps(nb::handle xo, nb::handle yo, nb::handle to, bool omit) {
    const Rows x(rows_arg(xo)), y(rows_arg(yo));
    const NDArray& ta = require_ndarray(to);
    if (ta.dtype != DType::Float64 || ta.ndim != 1) throw nb::type_error("htest: expects a 1-D float64 array of t");
    std::vector<double> t(static_cast<std::size_t>(ta.shape[0]));
    for (std::int64_t i = 0; i < ta.shape[0]; ++i) t[static_cast<std::size_t>(i)] = vec_at(ta, i);
    Outs out(3, x.lanes);
    each_lane(x.lanes, 64 * static_cast<std::int64_t>(t.size()) * (x.n + y.n), [&](std::int64_t r) {
        std::vector<double> xv, yv;
        const bool nx_nan = values(x, r, xv), ny_nan = values(y, r, yv);
        const bool nan = nx_nan || ny_nan;
        const EsLane l = epps_lane(xv, yv, t, nan && !omit);
        out.p[0][r] = l.w, out.p[1][r] = l.p, out.p[2][r] = l.df;
    });
    return out.tuple();
}

// the exact tails at given statistics (tests): kind 0 two-sample KS (sizes, h, alternative, no
// ties), kind 1 the equal-size closed form, kind 2 the Cramér-von Mises tail P(4U/G >= v)
double exact_tail(int kind, std::int64_t n1, std::int64_t n2, std::int64_t v, int alt) {
    if (kind == 2) {
        cvm2::Problem P;
        P.nx = n1, P.ny = n2, P.v = v;
        const cvm2::Plan L = cvm2::plan(P, kPlanStates);
        return L.ok ? cvm2::tail(L) : kNaN;
    }
    if (kind == 1) return ks2::equal_tail(n1, v, alt == kTwoSided);
    ks2::Problem P;
    P.na = n1, P.nb = n2, P.h = v, P.alt = alt;
    return ks2::tail(P);
}

}  // namespace

void bind_gof2(nb::module_& m) {
    m.def("ks2", &ks2_test);
    m.def("cvm2", &cvm2_test);
    m.def("ad_ksamp", &ad_ksamp);
    m.def("epps", &epps);
    m.def("exact_tail", &exact_tail);
}

}  // namespace sd::htest
