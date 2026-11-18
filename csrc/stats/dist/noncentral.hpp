// The noncentral chi-square, Student t and F distributions (docs/api/stats.md §2.1,
// docs/design/stats.md T26). nc = 0 is the central distribution exactly.
//
//   * cdfs and inverses: sd.special's chndtr, nctdtr, ncfdtr and their complements (both
//     tails summed directly, measured to probability 1e-300: the guaranteed region); the
//     upper-tail inverses solve on the given upper probability (1 - q is not formed). The
//     log forms and the inverses take the tails as m 2^e (special's *_tail, the sums before
//     their rounding to a double), so a tail far below the double range keeps its value and
//     a log target e^lp is never formed (it underflows below lp = -745).
//   * densities: chi-square and F as Poisson mixtures of gamma and beta densities, every
//     term positive, summed from the largest term outwards in logs (double-double);
//     the t as the integral f(t) = E[S phi(t S - nc)], S = sqrt(V / df), whose integrand is
//     positive everywhere (the series' two families cancel in the tail away from nc),
//     by the trapezoid rule on a sinh map about its peak, as sd.special's nct_opposite.
//   * moments: chi-square in closed form; F and t as products of independent factors,
//     F = (X / dfn)(dfd / V) and T = (Z + nc) R, R = sqrt(df / V), whose central moments
//     expand into positive terms; R's moment ratios through D_k = log E[R^k] - k log E[R]
//     by their 1/df series (coeffs/nct_moments.hpp), the linear terms cancelled exactly.
//   * mode: the root of the density's derivative (1e-14, T26); entropy: -int f log f by the
//     trapezoid rule, halving the step to convergence (its accuracy documented as loose).
//   * the bulk (speed inside the contract, docs/design/stats.md §3.4): where their series
//     from j = 0 are short, the densities are the central density times a hypergeometric
//     sum (log_density_bulk), cdf and ccdf sd.special's sums of the bulk (nc_detail::lite,
//     inside chndtr, ncfdtr, nctdtr), the log forms their logs (log_tail_either) and icdf,
//     iccdf Newton steps on them (ncd::newton). Each hands over to the definitions above
//     where it does not apply: the tails are theirs, unchanged.
//   * bounded time: every loop has a fixed most count of steps (stated at each), and a sum,
//     an integral or a search that has not ended within it gives NaN, never the part done
//     so far. Inside the guaranteed region nothing comes near those counts (but the entropy
//     of a tail too heavy for the range of x: NaN); outside it a method ends by the same
//     sums, by a form that is right there, or as NaN.
#pragma once

#include "special/ops/distr.hpp"
#include "special/ops/noncentral.hpp"
#include "stats/coeffs/nct_moments.hpp"
#include "stats/discrete.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/lifetime.hpp"
#include "stats/dist/normal.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace sd::stats {

namespace ncd {

using num::DD;
namespace dd = num::dd;

// log sum_j t_j for a positive, log-concave series given log t_j (double-double, costly)
// and t_(j+1) / t_j (cheap): from the largest term j* (found from the estimate j0 by
// the ratio) outwards, the ratios resynchronized every 8 terms. Also E[j] under the
// weights t_j / sum (for the modes).
//
// Bounded: 64 steps each way to the peak, then at most kSeriesTerms terms each way. The
// terms span 9 standard deviations of their bell on each side, 9 sqrt(j* / 2) for the
// chi-square's and under 9 sqrt(j*) for the F's: in the guaranteed region j* <= 1.6e6
// (the chi-square at sqrt(nc x) = 6.3e6, where its closed form takes over; the F's j* is
// below nc / 2 + dfd / 2 + 1), 1.2e4 terms a side. A series that has not faded within
// the count, or whose peak is past 2^52 (j + 1 is no longer exact), is not summed: NaN.
struct Series {
    DD log;
    double mean_j;
};
inline constexpr int kSeriesTerms = 1 << 16;
inline constexpr Series kNoSeries{{kNaN, 0.0}, kNaN};
// The work of the densities on this thread, for the entropy integrals (ncd::entropy, which
// bounds its time by it): a series adds two units a term, the t's integral one a node.
inline thread_local long work_done = 0;
template <class LogTerm, class Ratio>
Series log_series(double j0, double jmin, LogTerm&& lt, Ratio&& ratio) {
    if (!(j0 < 0x1p52)) return kNoSeries;
    double j = std::max(jmin, std::floor(j0));
    for (int i = 0; i < 64 && ratio(j) > 1.0; ++i) j += 1.0;
    for (int i = 0; i < 64 && j > jmin && ratio(j - 1.0) < 1.0; ++i) j -= 1.0;
    // A bell too wide for the count, seen before it is summed: the ratios fall in j, so the
    // term kSeriesTerms above the peak is at least ratio(j + kSeriesTerms)^kSeriesTerms; above
    // 2^-43 it has not faded (the sum is under 2^17 peaks). Only a peak beyond 2^24 can be
    // that wide.
    if (j >= 0x1p24 && kSeriesTerms * std::log(ratio(j + kSeriesTerms)) > -29.0) return kNoSeries;
    const DD l0 = lt(j);
    disc::KahanSum sum, sumj;
    sum.add(1.0);
    sumj.add(j);
    double t = 1.0;
    bool faded = false;
    int up = 0, down = 0;  // terms taken
    for (int i = 1; i <= kSeriesTerms; ++i) {  // up
        up = i;
        const double k = j + i;
        t *= ratio(k - 1.0);
        if (i % 8 == 0) {
            const DD d = lt(k) - l0;
            t = std::exp(d.hi + d.lo);
        }
        sum.add(t);
        sumj.add(t * k);
        if (!(t > sum.s * 0x1p-60)) {  // (a NaN term ends it too: the sum is NaN with it)
            faded = true;
            break;
        }
    }
    if (!faded) return kNoSeries;
    t = 1.0;
    faded = false;
    for (int i = 1; i <= kSeriesTerms; ++i) {  // down
        if (j - i < jmin) {
            faded = true;  // the first term is in
            break;
        }
        down = i;
        const double k = j - i;
        t /= ratio(k);
        if (i % 8 == 0) {
            const DD d = lt(k) - l0;
            t = std::exp(d.hi + d.lo);
        }
        sum.add(t);
        sumj.add(t * k);
        if (!(t > sum.s * 0x1p-60)) {
            faded = true;
            break;
        }
    }
    if (!faded) return kNoSeries;
    work_done += 2 * (up + down);
    return {l0 + std::log(sum.value()), sumj.value() / sum.value()};
}

// e^l for a log density l: 0 below the doubles (exp_dd's 1 + l.lo is for a small l.lo; at
// l ~ -1e99 it is negative, and the product -0).
inline double density(DD l) { return l.hi < -746.0 ? 0.0 : num::exp_dd(l); }

// sqrt(v) for a double v >= 0 as a double-double: the rounded root and its remainder
// (v - s^2 exactly, by fma) over 2 s.
inline DD sqrt_dd(double v) {
    const double r = std::sqrt(v);
    if (!(r > 0.0 && r < kInf)) return {r, 0.0};
    return dd::quick_two_sum(r, std::fma(-r, r, v) / (2.0 * r));
}

using special::Scaled;

// ---- the bulk (docs/design/stats.md §3.4): the densities as the central density times a
// hypergeometric series from j = 0, and the inverses by Newton steps on sd.special's sums
// of the bulk (special/ops/noncentral.hpp, nc_detail::lite). Where one does not apply (too
// many terms, a value outside a double's range, the tails) the definitions above and below
// take over unchanged.
namespace lite = special::nc_detail::lite;
using lite::Lite;

inline DD to_dd(Lite v) { return dd::quick_two_sum(v.h, v.l); }

// sum_j r_0 r_1 ... r_(j-1) (j >= 0) for positive ratios falling in j (ratio.next(): r_0,
// r_1, ... in turn), in "double + error" arithmetic (lite: a term after j steps is exact to
// j 2^-100 or so), ended where the geometric bound on the rest, t_(j+1) / (1 - r_j), is
// below tol of the sum. False: not ended within kLiteTerms, or a term past 2^960.
template <class R>
bool hyper_sum(R ratio, double tol, Lite& out) {
    Lite t{1.0, 0.0}, S{0.0, 0.0};
    for (int j = 0; j < lite::kLiteTerms; ++j) {
        lite::add(S, t);
        const Lite r = ratio.next();
        t = lite::mul(t, r);
        if (r.h < 1.0 && t.h <= tol * S.h * (1.0 - r.h)) {
            out = S;
            return true;
        }
        if (!(t.h < 0x1p960)) return false;
    }
    return false;
}
// r_j = z (c + j) / ((j + 1)(a + j)) (the F's 1F1(c; a; z)); without c: z / ((j + 1)(a + j))
// (the chi-square's 0F1(; a; z)). Falling in j for c >= a.
struct HyperRatio {
    Lite z, a, c;
    bool with_c;
    double j1 = 1.0;
    Lite next() {
        const Lite den = lite::mul(a, Lite{j1, 0.0});
        const Lite r = lite::quot(with_c ? lite::mul(z, c) : z, den);
        lite::bump(a);
        if (with_c) lite::bump(c);
        j1 += 1.0;
        return r;
    }
};
// r_m = q (c + m) / ((m + h)(m + 1)), h = 1/2 or 3/2 (the t's two families; the
// denominators are exact doubles). Falling in m for c >= h.
struct HalfRatio {
    Lite q, c;
    double h, m = 0.0;
    Lite next() {
        const Lite r = lite::quot(lite::mul(q, c), Lite{(m + h) * (m + 1.0), 0.0});
        lite::bump(c);
        m += 1.0;
        return r;
    }
};

// The tail target of an inverse: the side whose probability is small, and that probability
// as m 2^e (the log forms' targets e^lp reach e^-1e10).
struct Target {
    bool upper;  // the side solved
    Scaled p;
};
// icdf (upper: iccdf) at p, 0 < p < 1: the smaller side (1 - p exact for p >= 1/2)
inline Target target_p(double p, bool upper) {
    const bool small = p <= 0.5;
    return {upper == small, special::scaled(small ? p : 1.0 - p)};
}
// ilogcdf (upper: ilogccdf) at lp < 0: e^lp itself below 1/2, else the other side 1 - e^lp.
// (e^lp as m 2^e holds down to lp = -2^62; a finite target below that is not solved: NaN.)
inline bool log_target_ok(double lp) { return lp > -0x1p62; }
inline Target target_log(double lp, bool upper) {
    if (lp > -num::kLn2.hi) return {!upper, special::scaled(-std::expm1(lp))};
    return {upper, special::nc_detail::exp_wide(lp)};
}

// The x (positive: on a log scale; else real, on asinh) whose tail (tail(x, upper, wide) as
// m 2^e: the sd.special sums before their rounding to a double) is the target, by
// sd.special's bracketing and zeroin (special/ops/noncentral.hpp) on log(tail / target):
// a ratio of m 2^e values, so a target far below the double range (or a subnormal one, of
// few bits as a double result) is solved as 0.3 is.
//
// A tail the wide sum does not give (NaN: not summed; 0: below the m 2^e form, the tails
// being positive at every x the solver tries) is below the target when the target is above
// 2^-2800 and the tail below e^-2000 (a 0, or 0 by the probability's own sum, wide = false):
// only its side of the root is needed there. Else it is unknown, and NaN is the answer
// (the solver stops at it).
template <class Tail>
double invert(const Tail& tail, Target tg, double x0, bool positive) {
    const bool shallow = tg.p.e > -2800;
    auto g = [&](double x) {
        Scaled t = tail(x, tg.upper, true);
        if (t.m != t.m || t.m == 0.0) {
            const bool below = shallow && (t.m == 0.0 || tail(x, tg.upper, false).m == 0.0);
            t = below ? Scaled{0.0, 0} : Scaled{kNaN, 0};
        }
        const double l = special::log_abs(t / tg.p);
        return tg.upper ? -l : l;  // increasing in x
    };
    namespace nd = special::nc_detail;
    if (positive) {
        if (!(x0 > 0.0 && x0 < kInf)) x0 = 1.0;
        return nd::solve(g, x0, [](double x) { return std::log(x); }, [](double u) { return std::exp(u); }, 0.125, 0.0,
                         kInf);
    }
    if (!std::isfinite(x0)) x0 = 0.0;
    return nd::solve(g, x0, [](double t) { return std::asinh(t); }, [](double u) { return std::sinh(u); }, 0.125, -kInf,
                     kInf);
}

// The root of tail(x) = p (the side and p of tg) by Newton steps on log(tail / p), in log x
// for a positive variable: eval(x, upper, tail, f) gives the tail by the sums of the bulk
// (relative error inside kFastBudget: special's *_lite) and the density f (a derivative:
// 1e-8 is enough). The root's error is the tail's, times the tail's length tail / f: the
// contract's max(1, kappa). A step d is the last when the next error, K d^2 + 2^-30 d (the
// quadratic constant K = d / d_prev^2 measured from the two steps, both already in the
// quadratic regime: d_prev below 2^-5 of the scale max(|x|, tail / f) and d below d_prev / 8;
// 2^-30 for the density's own error, a plain-double sum's), is below 2^-52 of that scale, or
// when d itself is below 2^-48 of it (the tail's own rounding). False (the
// bracketing solver then): the sums do not apply at an iterate, or no end in 12 steps.
template <class Eval>
bool newton(const Eval& eval, Target tg, double x0, bool positive, double& out) {
    const double p = special::value(tg.p);
    if (!(p >= 0x1p-900 && std::isfinite(x0)) || (positive && !(x0 > 0.0))) return false;
    double x = x0, prev = 0.0;
    for (int it = 0; it < 12; ++it) {
        double tail, f;
        if (!eval(x, tg.upper, tail, f) || !(f > 0.0 && tail > 0.0 && std::isfinite(f))) return false;
        const double len = tail / f;
        double d = -std::log(tail / p) * len;  // lower tail: increasing in x
        if (tg.upper) d = -d;
        const double scale = std::max(std::fabs(x), len);
        const double ad = std::fabs(d);
        double xn;
        if (positive) {
            const double du = std::clamp(d / x, -2.0, 2.0);
            xn = x + x * std::expm1(du);  // the step in log x (not x + d: the constant K is the log scale's)
        } else {
            xn = x + std::clamp(d, -4.0 * scale, 4.0 * scale);
        }
        if (!std::isfinite(xn) || (positive && !(xn > 0.0))) return false;
        if (ad <= 0x1p-48 * scale ||
            (it > 0 && prev <= 0x1p-5 * scale && ad <= 0x1p-3 * prev &&
             ad * (ad * ad / (prev * prev) + 0x1p-30) <= 0x1p-52 * scale)) {
            out = xn;
            return true;
        }
        prev = std::fabs(xn - x);  // the move made (a clamped step is not d)
        x = xn;
    }
    return false;
}

// log P from both kinds of sum, each side from the bulk's where it has one (eval(upper, v):
// a tail with a relative error inside kFastBudget, special's *_lite; at least 2^-1000) and
// else from the wide one (tail(upper, wide), as m 2^e: its exponent kept, a tail of e^-1e4
// is no 0): the asked side's log where it is at most 1/2 (the tail's relative error, 14
// units for a lower sum, plus the log's 2 |l|, against the mixed bound max(1, |l|)), else
// log1p of minus the other side. What one kind has found is kept when the other must
// finish: an asked tail the bulk puts above 1/2 leaves one wide sum to run, the other
// side's, not both.
//
// An asked side that was not summed (NaN) has no log: NaN. So has one that is 0 at a point
// inside the support (`interior`): the tails are positive there, and a 0 is one below the
// m 2^e form (e^-(2^62)). The other side, wanted only as 1 - it, is taken as the
// probability (wide = false: 0 where it is below e^-2000, whose log1p is -0).
template <class Eval, class Tail>
double log_tail_either(const Eval& eval, const Tail& tail, bool upper, bool interior) {
    double v;
    if (eval(upper, v)) {
        if (v <= 0.5) return std::log(v);
    } else {
        const Scaled s = tail(upper, true);
        const double sv = special::value(s);
        if (sv != sv) return kNaN;
        if (sv <= 0.5) return s.m == 0.0 ? (interior ? kNaN : -kInf) : special::log_abs(s);
    }
    if (eval(!upper, v) && v <= 0.5) return std::log1p(-v);
    return std::log1p(-special::value(tail(!upper, false)));
}

// -int f log f dx by the trapezoid rule in u, x = to_x(u) (dx = jac(u) du), the step
// halved until two estimates agree to 2^-50 (logpdf: a double-double log density, -inf
// outside the support's interior).
//
// Bounded: a sweep runs out from u = 0 each way until the integrand has faded to 1e-22 of
// its largest value, or to the end of the doubles (x = 0 or inf). What is cut there is
// about v / (p h) nodes' worth, v the last node and e^(-p h) the fall per node there (the
// tails are powers of x, geometric in u; p h from the last node and the one 31 before, the
// subnormal x being too coarse for neighbours): above 2^-40 of the sum the integral does
// not fit the range of x (an F with dfd below 0.08 or so), and neither does a pole whose
// density passes the doubles: NaN. At most 12 halvings, and kEntropyWork
// units of work in all: 16 a node and what its density adds to work_done (so about a
// second at most, whatever the densities cost). A NaN density, or no agreement by then,
// is NaN. The rule's error squares with each halving, so two estimates agreeing to 2^-26
// already settle the second to the last bit: where rounding alone keeps the last pair
// apart by more than 2^-50, that is accepted.
inline constexpr long kEntropyWork = 1L << 25;
template <class LogPdf, class ToX, class Jac>
double entropy(LogPdf&& logpdf, ToX&& to_x, Jac&& jac) {
    work_done = 0;
    // the sum over u = offset + k h, k in Z; false: not an integral (see above)
    auto sweep = [&](double h, double offset, double& out) {
        disc::KahanSum s;
        double mag = 0.0, sum_abs = 0.0;
        for (int dir = -1; dir <= 1; dir += 2) {
            double recent[32];  // |v| at this side's last 32 nodes (node i at i % 32)
            int n = 0;
            for (int k = (dir < 0 ? 0 : 1);; ++k) {
                work_done += 16;
                if (work_done > kEntropyWork) return false;
                const double u = offset + dir * k * h;
                const DD l = logpdf(to_x(u));
                const double lv = l.hi + l.lo;
                if (lv != lv) return false;  // a density not summed
                if (lv == -kInf) {  // the end of the range
                    if (n < 32) return false;
                    const double last = recent[(n - 1) % 32], ref = recent[n % 32];
                    if (!(last == 0.0 || (ref > last && 31.0 * last <= 0x1p-40 * sum_abs * std::log(ref / last)))) return false;
                    break;
                }
                const double v = -density(l) * lv * jac(u);
                if (!std::isfinite(v)) return false;  // (a density past the doubles, next to a pole)
                s.add(v);
                const double av = std::fabs(v);
                recent[n++ % 32] = av;
                sum_abs += av;
                mag = std::max(mag, av);
                if (av <= 1e-22 * mag && k > 16) break;
            }
        }
        out = s.value();
        return true;
    };
    double h = 0.5, total, part;
    if (!sweep(h, 0.0, total)) return kNaN;  // the sum at the grid points
    double est = total * h, gap = kInf;
    for (int it = 0; it < 12; ++it) {
        if (!sweep(h, 0.5 * h, part)) return kNaN;  // the midpoints
        total += part;
        h *= 0.5;
        const double next = total * h;
        gap = std::fabs(next - est);
        est = next;
        if (gap <= 0x1p-50 * std::fabs(next) && it > 1) return next;
    }
    return gap <= 0x1p-26 * std::fabs(est) ? est : kNaN;
}

}  // namespace ncd

// ---------------------------------------------------------------- chi-square

struct NoncentralChiSquared {
    static constexpr char name[] = "noncentralchisquared";
    static constexpr std::array<const char*, 2> params{"df", "nc"};

    static bool valid(double df, double nc) { return df > 0.0 && std::isfinite(df) && nc >= 0.0 && std::isfinite(nc); }

    // f(x) = sum_j Pois(j; nc/2) chi2(x; df + 2j), chi2(x; k) = g(x/2; k/2) / 2 (g the standard
    // gamma density, gamma_detail::log_density_std): every term positive. `shift` lowers
    // the gamma shapes by `shift` (the mode's f_{df-2}). The shape a + j is a double-double
    // (rounded, it moved the density by s eps |log y - psi(s)|: 2.4e-13 at df = 7e3): the
    // density at its double part, moved by the rest to first order. Below x = 2^-1020, x/2
    // would round in the subnormals (1e-5 of the density at x = 4e-319): the terms from
    // log y = log x - log 2 directly, (s - 1) log y - y - log Gamma(s).
    static ncd::Series series(double x, double df, double nc, double shift = 0.0) {
        const double mu = 0.5 * nc, y = 0.5 * x, a = 0.5 * df - shift;
        const num::DD yd{y, 0.0};
        const bool tiny = x < 0x1p-1020;
        const num::DD ly = num::log_dd(x) - num::kLn2;
        auto lt = [&](double j) {
            const num::DD sh = num::dd::two_sum(a, j);
            num::DD l = gamma_detail::log_density_std(j + 1.0, num::DD{mu, 0.0}) - num::kLn2;
            if (tiny) return l + (sh - 1.0) * ly - y - special::nc_detail::lgamma_dd(sh);
            l = l + gamma_detail::log_density_std(sh.hi, yd);
            if (sh.lo != 0.0) l = l + sh.lo * (ly.hi - special::Digamma::f(sh.hi));
            return l;
        };
        auto ratio = [&](double j) { return mu * y / ((j + 1.0) * (a + j)); };  // t_(j+1) / t_j
        const double jmin = a > 0.0 ? 0.0 : 1.0;  // shape a + j > 0
        const double b = a + 1.0;  // the peak: (j + 1)(j + a) = mu y
        const double j0 = 0.5 * (-b + std::sqrt((a - 1.0) * (a - 1.0) + 4.0 * mu * y));
        return ncd::log_series(j0, jmin, lt, ratio);
    }
    // sum_k (-1)^k a_k / z^k, a_k = prod_(i <= k) (m^2 - (2 i - 1)^2) / (8 i): Hankel's expansion of
    // sqrt(2 pi z) e^-z I_nu(z), m = 2 nu, for m^2 + 1 <= 16 z: the terms are below 2^k / k!,
    // under 2^-90 of the sum by k = 36 (kHankelTerms is twice that), and what is left after
    // term k is within 2 chi(k) e^(pi (m^2 - 1) / (8 z)), under 1e4, of the first term left out
    // (Olver's bound, DLMF 10.40(iii)); the part in e^-2z is below e^-5e5 here.
    static constexpr int kHankelTerms = 72;
    static num::DD hankel(num::DD m, num::DD z) {
        num::DD term{1.0, 0.0}, sum{1.0, 0.0};
        for (int k = 1; k <= kHankelTerms; ++k) {
            const double odd = 2.0 * k - 1.0;
            term = term * ((m - odd) * (m + odd)) / (z * (-8.0 * k));
            sum = sum + term;
            if (!(std::fabs(term.hi) > 0x1p-90 * sum.hi)) return sum;
        }
        return {kNaN, 0.0};
    }
    // The series' sum far out, where its bell (9 sqrt(z) terms about j* = z / 2, z = sqrt(nc x) =
    // 2 sqrt(mu y)) is long: in closed form,
    //     f(x) = e^(-(x + nc) / 2) (x / nc)^(nu / 2) I_nu(z) / 2,   nu = df/2 - 1 (less `shift`),
    // with I_nu by Hankel's expansion, so with m = 2 nu
    //     log f = -(3/2) log 2 - log(pi) / 2 + ((m - 1) / 4) log x - ((m + 1) / 4) log nc
    //             - (sqrt x - sqrt nc)^2 / 2 + log H_m(z),
    // every part a double-double (the roots' difference to 2^-100 of the roots: the square
    // keeps its bits where x is next to nc). E[j] = (z / 2) I_(nu+1)(z) / I_nu(z). Taken from
    // z = 2^18 (below it the series is 5e3 terms at most) where (|m| + 2)^2 + 1 <= 16 z (both
    // expansions); false: the series.
    static bool series_far(double x, double df, double nc, double shift, ncd::Series& out) {
        const num::DD sx = ncd::sqrt_dd(x), sn = ncd::sqrt_dd(nc);
        const num::DD z = sx * sn;
        const num::DD m = num::dd::two_sum(df, -2.0) - 2.0 * shift;
        const double am = std::fabs(m.hi) + 2.0;
        if (!(z.hi >= 0x1p18 && am * am + 1.0 <= 16.0 * z.hi)) return false;
        const num::DD h0 = hankel(m, z), h1 = hankel(m + 2.0, z);
        const num::DD d = sx - sn;
        // (d / 2 times d: d^2 itself can pass the largest double at x next to it)
        const num::DD l = (m - 1.0) * 0.25 * num::log_dd(x) - (m + 1.0) * 0.25 * num::log_dd(nc) - d * 0.5 * d -
                          num::kLn2 * 1.5 - num::kLogPi * 0.5 + num::dd::log(h0);
        const num::DD mean = z * 0.5 * h1 / h0;
        out = {l, mean.hi + mean.lo};
        return true;
    }
    // the density at x = 0: df < 2 inf, df = 2 e^(-nc/2) / 2, df > 2 zero
    static double density0(double df, double nc) { return df < 2.0 ? kInf : (df == 2.0 ? 0.5 * std::exp(-0.5 * nc) : 0.0); }
    // The bulk: f(x) = chi2(x; df) e^(-mu) 0F1(; a; mu y) (the series above with its j = 0
    // term taken out: t_j / t_0 = (mu y)^j / (j! (a)_j)), the series from j = 0 (hyper_sum)
    // while it has at most kLiteTerms terms; in logs, each a double-double (the central
    // density's, Gamma::log_density; log of the sum; mu = nc / 2 exact), so the density has
    // the error of its one exponential (exp_dd: 4 units of 2^-53) at any size.
    static bool log_density_bulk(double x, double df, double nc, num::DD& l) {
        const double mu = 0.5 * nc, y = 0.5 * x, a = 0.5 * df;
        const num::DD z = num::dd::two_prod(mu, y);
        // the largest term is near j = sqrt(z); the terms fall off over ~ z^(1/4) around it
        if (!(x >= 0x1p-1020 && z.hi <= 6e4 && z.hi >= 0x1p-1000)) return false;
        ncd::Lite s;
        if (!ncd::hyper_sum(ncd::HyperRatio{{z.hi, z.lo}, {a, 0.0}, {0.0, 0.0}, false}, ncd::lite::kLiteTol, s)) return false;
        l = Gamma::log_density(x, a, 2.0) - mu + num::dd::log(ncd::to_dd(s));
        return true;
    }
    // the terms' sum and E[j] (the mode's): the closed form far out, else the series
    static ncd::Series summed(double x, double df, double nc) {
        ncd::Series far;
        return series_far(x, df, nc, 0.0, far) ? far : series(x, df, nc);
    }
    // log f(x), 0 < x < inf
    static num::DD log_density(double x, double df, double nc) {
        num::DD l;
        return log_density_bulk(x, df, nc, l) ? l : summed(x, df, nc).log;
    }
    static double pdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::pdf(x, df);
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return density0(df, nc);
        return ncd::density(log_density(x, df, nc));
    }
    static double logpdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::logpdf(x, df);
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return std::log(density0(df, nc));
        const num::DD l = log_density(x, df, nc);
        return l.hi + l.lo;
    }
    static double cdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::cdf(x, df);
        return x <= 0.0 ? 0.0 : special::Chndtr::f(df, nc, x);
    }
    static double ccdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::ccdf(x, df);
        return x <= 0.0 ? 1.0 : special::Chndtrc::f(df, nc, x);
    }
    // the tail as m 2^e (sd.special's chndtr sum before its rounding: the log forms and
    // inverses reach tails far below the double range); wide = false: the probability's sum
    static special::Scaled tail(double x, double df, double nc, bool upper, bool wide = true) {
        return special::nc_detail::chndtr_tail(df, nc, x, upper, wide);
    }
    // log P(X > x) far out, where the wide sum does not give it (its peak j* = sqrt(mu y),
    // mu = nc/2, y = x/2, past what the walk reaches, 1e8; a value below the m 2^e form): from
    // the density. With Q(s, y) between D and D / (1 - (s - 1) / y), D = y^(s-1) e^-y / Gamma(s)
    // (s >= 1; between D (1 - 1 / y) and D for s < 1), and sum_j w_j D(df/2 + j) = 2 f(x),
    //     2 f (1 - 1 / y) <= P(X > x) <= 2 f / (1 - eps),   eps = (df/2 + 2 j* + 64) / y
    // (the terms beyond j = 2 j* + 64 are under 2^-128 of the sum: their ratios, mu y /
    // ((j + 1)(df/2 + j)), are below 1/4 from j* + 32 on; and those beyond y / 2, where the
    // bound on Q is 1, are under e^(-y / 5) of it for eps <= 1/4). Taken where that is inside
    // the log's bound, 2 (eps + 1 / y) <= 1e-15 |log 2 f| (in the guaranteed region: wherever
    // j* > 1e8); false elsewhere.
    static bool far_upper_log(double x, double df, double nc, num::DD& l) {
        const double mu = 0.5 * nc, y = 0.5 * x;
        const double jstar = std::sqrt(mu) * std::sqrt(y);
        if (!std::isfinite(x)) return false;
        const double eps = (0.5 * df + 2.0 * jstar + 65.0) / y;  // (with the lower side's 1 / y)
        if (!(eps <= 0.25)) return false;
        const num::DD lf = log_density(x, df, nc) + num::kLn2;
        if (!(2.0 * eps <= 1e-15 * std::fabs(lf.hi))) return false;
        l = lf;
        return true;
    }
    // the tails for the inverses: the far upper one from the density where the wide sum does
    // not give it (NaN, or 0 at a finite x). One below e^-(2^62), the least target
    // (ncd::log_target_ok), as 2^-(2^63 - 2^60): below every target.
    static special::Scaled tail_far(double x, double df, double nc, bool upper, bool wide) {
        const special::Scaled s = tail(x, df, nc, upper, wide);
        num::DD l;
        if ((s.m == s.m && s.m != 0.0) || !upper || !wide || !far_upper_log(x, df, nc, l)) return s;
        if (!(l.hi > -0x1p62)) return {0.5, -(std::int64_t{7} << 60)};
        return special::nc_detail::exp_wide(l);
    }
    static double log_of_tail(double x, double df, double nc, bool upper) {  // 0 < x
        const bool inside = std::isfinite(x);
        const double v = ncd::log_tail_either([&](bool u, double& t) { return inside && special::nc_detail::ncx2_lite(df, nc, x, u, t); },
                                              [&](bool u, bool wide) { return tail(x, df, nc, u, wide); }, upper, inside);
        num::DD l;
        if (v != v && upper && far_upper_log(x, df, nc, l)) return l.hi + l.lo;
        return v;
    }
    static double logcdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::logcdf(x, df);
        if (x <= 0.0) return -kInf;
        return log_of_tail(x, df, nc, false);
    }
    static double logccdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::logccdf(x, df);
        if (x <= 0.0) return 0.0;
        return log_of_tail(x, df, nc, true);
    }
    static double inv(double p, double df, double nc, bool upper) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (nc == 0.0) return upper ? ChiSquared::iccdf(p, df) : ChiSquared::icdf(p, df);
        if (p == 0.0) return upper ? kInf : 0.0;
        if (p == 1.0) return upper ? 0.0 : kInf;
        const double x0 = special::nc_detail::ncx2_start(df, nc, upper ? 1.0 - p : p);
        const ncd::Target tg = ncd::target_p(p, upper);
        double out;
        if (ncd::newton([&](double x, bool u, double& t, double& f) { return special::nc_detail::ncx2_lite(df, nc, x, u, t, &f); },
                        tg, x0, true, out))
            return out;
        return ncd::invert([&](double x, bool u, bool wide) { return tail_far(x, df, nc, u, wide); }, tg, x0, true);
    }
    static double icdf(double p, double df, double nc) { return inv(p, df, nc, false); }
    static double iccdf(double p, double df, double nc) { return inv(p, df, nc, true); }
    // log p itself as the target (e^lp is not formed: it underflows below lp = -745)
    static double from_log(double lp, double df, double nc, bool upper) {
        if (nc == 0.0) return upper ? ChiSquared::ilogccdf(lp, df) : ChiSquared::ilogcdf(lp, df);
        if (!(lp <= 0.0)) return kNaN;
        if (lp == -kInf) return inv(0.0, df, nc, upper);
        if (!ncd::log_target_ok(lp)) return kNaN;
        const ncd::Target tg = ncd::target_log(lp, upper);
        if (tg.p.m == 0.0) return inv(1.0, df, nc, upper);  // 1 - e^lp rounds to 0: p = 1
        // start: Patnaik's c chi-square(k) (ncx2_start) at the log target
        const double c = (df + 2.0 * nc) / (df + nc), k = (df + nc) / c;
        const double x0 = c * (upper ? ChiSquared::ilogccdf(lp, k) : ChiSquared::ilogcdf(lp, k));
        return ncd::invert([&](double x, bool u, bool wide) { return tail_far(x, df, nc, u, wide); }, tg, x0, true);
    }
    static double ilogcdf(double lp, double df, double nc) { return from_log(lp, df, nc, false); }
    static double ilogccdf(double lp, double df, double nc) { return from_log(lp, df, nc, true); }

    static double mean(double df, double nc) { return df + nc; }
    static double variance(double df, double nc) { return 2.0 * (df + 2.0 * nc); }
    static double standard_deviation(double df, double nc) { return std::sqrt(2.0 * (df + 2.0 * nc)); }
    // (the powers of v = df + 2 nc pass the doubles from v = 1e205 and 1e154: the ratio first there)
    static double skewness(double df, double nc) {  // 2^(3/2) (df + 3 nc) / (df + 2 nc)^(3/2)
        if (nc == 0.0) return ChiSquared::skewness(df);  // the central one's bits
        const double v = df + 2.0 * nc;
        if (!(v < 0x1p600)) return 2.8284271247461903 * ((df + 3.0 * nc) / v) / std::sqrt(v);
        return 2.8284271247461903 * (df + 3.0 * nc) / (v * std::sqrt(v));
    }
    static double kurtosis(double df, double nc) {  // 12 (df + 4 nc) / (df + 2 nc)^2
        if (nc == 0.0) return ChiSquared::kurtosis(df);
        const double v = df + 2.0 * nc;
        if (!(v < 0x1p500)) return 12.0 * ((df + 4.0 * nc) / v) / v;
        return 12.0 * (df + 4.0 * nc) / (v * v);
    }
    // f' = (f_(df-2) - f_df) / 2, so the mode solves y = A(y) - 1, A = E[shape] under the
    // weights of f_df's terms at y = x/2 (each term's log-derivative is (a_j - 1)/y - 1):
    // phi(y) = A(y) - 1 - y falls from positive to negative. df < 2: the density is
    // infinite at 0 (T21: that point); df = 2 with nc <= 2 too has its peak at 0.
    static double mode(double df, double nc) {
        if (nc == 0.0) return ChiSquared::mode(df);
        if (df < 2.0) return 0.0;
        auto phi = [&](double x) {
            const ncd::Series s = summed(x, df, nc);
            return 2.0 * ((0.5 * df - 1.0) + s.mean_j) - x;  // in x = 2y (df/2 - 1 exact)
        };
        const double at0 = phi(1e-300);
        if (!(at0 > 0.0)) return at0 != at0 ? kNaN : 0.0;
        double x0 = std::max(df + nc - 2.0, 1.0);
        auto g = [&](double x) { return -phi(x); };  // increasing
        return special::nc_detail::solve(g, x0, [](double x) { return std::log(x); },
                                         [](double u) { return std::exp(u); }, 0.125, 0.0, kInf);
    }
    static double entropy(double df, double nc) {
        if (nc == 0.0) return ChiSquared::entropy(df);
        const double m = std::max(df + nc, 1e-3);  // x = m e^u
        if (!std::isfinite(m)) return kNaN;
        return ncd::entropy([&](double x) { return x > 0.0 && std::isfinite(x) ? log_density(x, df, nc) : num::DD{-kInf, 0.0}; },
                            [&](double u) { return m * std::exp(u); }, [&](double u) { return m * std::exp(u); });
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

// ---------------------------------------------------------------- F

namespace ncf_detail {

using num::DD;

// The central moments 2..4 of P = bX + aY + XY for independent centred X, Y with central
// moments mx[i], my[i] (mx[0] = 1, mx[1] = 0), a, b > 0: the expansion of P^k into
// X^i Y^j, each term c_ij mx[i] my[j]; all positive here (a, b > 0, odd moments >= 0).
struct Central {
    double m2, m3, m4;
};
inline Central product_central(double a, double b, const double mx[5], const double my[5]) {
    // P^k as coefficients c[i][j] of X^i Y^j, P = b X + a Y + X Y
    double c[5][5] = {{1.0}}, n[5][5];
    double out[5] = {0.0};
    for (int k = 1; k <= 4; ++k) {
        for (auto& r : n)
            for (double& v : r) v = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                if (c[i][j] == 0.0) continue;
                n[i + 1][j] += c[i][j] * b;
                n[i][j + 1] += c[i][j] * a;
                n[i + 1][j + 1] += c[i][j];
            }
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) c[i][j] = n[i][j];
        double s = 0.0;
        for (int i = 0; i <= 4; ++i)
            for (int j = 0; j <= 4; ++j)
                if (c[i][j] != 0.0) s += c[i][j] * mx[i] * my[j];
        out[k] = s;
    }
    return {out[2], out[3], out[4]};
}

}  // namespace ncf_detail

struct NoncentralF {
    static constexpr char name[] = "noncentralf";
    static constexpr std::array<const char*, 3> params{"dfn", "dfd", "nc"};

    static bool valid(double dfn, double dfd, double nc) {
        return dfn > 0.0 && std::isfinite(dfn) && dfd > 0.0 && std::isfinite(dfd) && nc >= 0.0 && std::isfinite(nc);
    }
    // One infinite degree of freedom, as special.ncfdtr: dfd = inf is the noncentral
    // chi-square(dfn, nc) / dfn; dfn = inf is dfd / chi-square(dfd) (the noncentral numerator
    // over dfn goes to 1), the inverse gamma (dfd/2, dfd/2). Both infinite stays NaN.
    static bool at_limit(double dfn, double dfd, double nc) {
        if (!(nc >= 0.0 && std::isfinite(nc))) return false;
        return (dfd == kInf && dfn > 0.0 && std::isfinite(dfn)) || (dfn == kInf && dfd > 0.0 && std::isfinite(dfd));
    }
    template <class M>
    static double limit_in(double x, double dfn, double dfd, double nc) {
        if (dfd == kInf) return scaled_of<NoncentralChiSquared, M>(dfn, x, dfn, nc);
        return method_of<InverseGamma, M>(x, 0.5 * dfd, 0.5 * dfd);
    }
    template <class M>
    static double limit_par(double dfn, double dfd, double nc) {
        if (dfd == kInf) return scaled_of<NoncentralChiSquared, M>(dfn, dfn, nc);
        return method_of<InverseGamma, M>(0.5 * dfd, 0.5 * dfd);
    }

    // f(x) = sum_j Pois(j; nc/2) beta(w; dfn/2 + j, dfd/2) dw/dx, w = dfn x / (dfn x + dfd):
    // log term = log Pois + log(w^a v^b / B(a, b)) - log x (beta_detail::log_prefix; dw/dx =
    // w v / x). The shape a0 + j a double-double, as the chi-square's (d log prefix / da =
    // log w - psi(a) + psi(a + b)); below x = 2^-1020, where dfn x rounds in the subnormals
    // (2e-6 of the density at x = 5e-319), the terms from log w = log dfn + log x -
    // log(dfn x + dfd) and log v = -log1p(dfn x / dfd) directly, log B by lgamma_dd.
    static ncd::Series series(double x, double dfn, double dfd, double nc) {
        const special::distr_detail::Arg g = special::distr_detail::f_arg(dfn, dfd, x);
        const double mu = 0.5 * nc, a0 = 0.5 * dfn, b = 0.5 * dfd, w = g.x.hi;
        const num::DD lx = num::log_dd(x);
        // (w itself below the normals too: dfd far above dfn x. The log B there is three log
        // Gammas, which cancel past its bits beyond a shape of 2^40: not summed.)
        const bool tiny = x < 0x1p-1020 || g.x.hi < 0x1p-1000;
        if (tiny && (a0 + mu > 0x1p40 || b > 0x1p40)) return ncd::kNoSeries;
        // (v = dfd / (dfn x + dfd) in the subnormals, x next to the largest double: its log is
        // off by half its last bit, 2^-1075 / v, times b in the density's. Inside the region
        // that is below 3e-12 of a log of 700 and more; where a dfn far beyond it brings v to
        // its last bits, or to 0, the series is not summed.)
        if (g.y.hi < 0x1p-1022 && !(b * 0x1p-1074 <= 2e-11 * g.y.hi)) return ncd::kNoSeries;
        const num::DD lw = num::log_dd(dfn) + lx - num::log_dd(dfn * x + dfd);
        const double lv = -std::log1p(dfn * x / dfd), lw1 = lw.hi;
        namespace nd = special::nc_detail;
        auto lt = [&](double j) {
            const num::DD sh = num::dd::two_sum(a0, j);
            const num::DD lp = gamma_detail::log_density_std(j + 1.0, num::DD{mu, 0.0}) - lx;
            if (tiny) {
                const num::DD lbeta = nd::lgamma_dd(sh) + nd::lgamma_dd(num::DD{b, 0.0}) - nd::lgamma_dd(sh + b);
                return lp + lw * sh + lv * b - lbeta;
            }
            num::DD l = lp + beta_detail::log_prefix(sh.hi, b, g.x, g.y);
            if (sh.lo != 0.0)
                l = l + sh.lo * (lw1 - special::Digamma::f(sh.hi) + special::Digamma::f(sh.hi + b));
            return l;
        };
        auto ratio = [&](double j) { return mu * w * (a0 + j + b) / ((j + 1.0) * (a0 + j)); };
        // the peak: (j + 1)(j + a0) = mu w (j + a0 + b)
        const double B = a0 + 1.0 - mu * w, C = a0 - mu * w * (a0 + b);
        const double j0 = 0.5 * (-B + std::sqrt(std::max(0.0, B * B - 4.0 * C)));
        return ncd::log_series(j0, 0.0, lt, ratio);
    }
    // the density at 0: dfn < 2 inf, dfn = 2 e^(-nc/2) (the j = 0 term alone), dfn > 2 zero
    static double density0(double dfn, double nc) { return dfn < 2.0 ? kInf : (dfn == 2.0 ? std::exp(-0.5 * nc) : 0.0); }
    // The bulk: f(x) = F(x; dfn, dfd) e^(-mu) 1F1(a0 + b; a0; mu w) (the series above over its
    // j = 0 term: t_j / t_0 = (mu w)^j (a0 + b)_j / (j! (a0)_j)), as the chi-square's.
    // The series' sum (false: not in the bulk).
    static bool bulk_sum(double x, double dfn, double dfd, double nc, special::distr_detail::Arg& g, ncd::Lite& s) {
        const double mu = 0.5 * nc, a0 = 0.5 * dfn, b = 0.5 * dfd;
        if (!(x >= 0x1p-1020 && x <= 0x1p1000 && mu <= 6e4)) return false;
        g = special::distr_detail::f_arg(dfn, dfd, x);
        if (!(g.x.hi >= 0x1p-900 && g.y.hi >= 0x1p-900)) return false;
        const ncd::Lite z = ncd::lite::mul({mu, 0.0}, {g.x.hi, g.x.lo});
        // the largest term: (j + 1)(j + a0) = z (j + a0 + b)
        const double B = a0 + 1.0 - z.h, C = a0 - z.h * (a0 + b);
        if (!(0.5 * (-B + std::sqrt(std::max(0.0, B * B - 4.0 * C))) <= ncd::lite::kLiteTerms)) return false;
        return ncd::hyper_sum(ncd::HyperRatio{z, {a0, 0.0}, ncd::lite::sum2(a0, b), true}, ncd::lite::kLiteTol, s);
    }
    static bool log_density_bulk(double x, double dfn, double dfd, double nc, num::DD& l) {
        special::distr_detail::Arg g;
        ncd::Lite s;
        if (!bulk_sum(x, dfn, dfd, nc, g, s)) return false;
        l = F::log_density(x, dfn, dfd) - 0.5 * nc + num::dd::log(ncd::to_dd(s));
        return true;
    }
    // The density itself without the logs (the central density's double-double logs are
    // half of the time): F(x; dfn, dfd) e^(-mu) = T(a0) e^(-mu) a0 / x with T(a0) = w^a0 v^b /
    // (a0 B(a0, b)) by the powers (special's beta_first_pow: 12 units of 2^-53), so with a0 / x,
    // the sum's rounding and two products 16 units: the budget, at most. Outside the powers'
    // range, or a result outside the normal numbers: the logs.
    static bool pdf_bulk(double x, double dfn, double dfd, double nc, double& out) {
        special::distr_detail::Arg g;
        ncd::Lite s;
        double tw;
        if (!bulk_sum(x, dfn, dfd, nc, g, s) ||
            !special::nc_detail::beta_first_pow(0.5 * dfn, 0.5 * dfd, g.x, g.y, num::DD{0.5 * nc, 0.0}, tw))
            return false;
        out = tw * ncd::lite::value(s) * (0.5 * dfn / x);
        return out >= 0x1p-1000 && out <= 0x1p1000;
    }
    // log f(x), 0 < x < inf
    static num::DD log_density(double x, double dfn, double dfd, double nc) {
        num::DD l;
        return log_density_bulk(x, dfn, dfd, nc, l) ? l : series(x, dfn, dfd, nc).log;
    }
    static double pdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::pdf(x, dfn, dfd);
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return density0(dfn, nc);
        double out;
        if (pdf_bulk(x, dfn, dfd, nc, out)) return out;
        return ncd::density(log_density(x, dfn, dfd, nc));
    }
    static double logpdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::logpdf(x, dfn, dfd);
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return dfn == 2.0 ? -0.5 * nc : std::log(density0(dfn, nc));
        const num::DD l = log_density(x, dfn, dfd, nc);
        return l.hi + l.lo;
    }
    static double cdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::cdf(x, dfn, dfd);
        return x <= 0.0 ? 0.0 : special::Ncfdtr::f(dfn, dfd, nc, x);
    }
    static double ccdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::ccdf(x, dfn, dfd);
        return x <= 0.0 ? 1.0 : special::Ncfdtrc::f(dfn, dfd, nc, x);
    }
    static special::Scaled tail(double x, double dfn, double dfd, double nc, bool upper, bool wide = true) {
        return special::nc_detail::ncfdtr_tail(dfn, dfd, nc, x, upper, wide);
    }
    static double log_of_tail(double x, double dfn, double dfd, double nc, bool upper) {  // 0 < x
        const bool inside = std::isfinite(x);
        return ncd::log_tail_either([&](bool u, double& v) { return inside && special::nc_detail::ncf_lite(dfn, dfd, nc, x, u, v); },
                                    [&](bool u, bool wide) { return tail(x, dfn, dfd, nc, u, wide); }, upper, inside);
    }
    static double logcdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::logcdf(x, dfn, dfd);
        if (x <= 0.0) return -kInf;
        return log_of_tail(x, dfn, dfd, nc, false);
    }
    static double logccdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::logccdf(x, dfn, dfd);
        if (x <= 0.0) return 0.0;
        return log_of_tail(x, dfn, dfd, nc, true);
    }
    static double inv(double p, double dfn, double dfd, double nc, bool upper) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (nc == 0.0) return upper ? F::iccdf(p, dfn, dfd) : F::icdf(p, dfn, dfd);
        if (p == 0.0) return upper ? kInf : 0.0;
        if (p == 1.0) return upper ? 0.0 : kInf;
        const double c = (dfn + 2.0 * nc) / (dfn + nc), k = (dfn + nc) / c;  // Patnaik's start
        const double x0 = c * k / dfn * special::Fdtri::f(k, dfd, upper ? 1.0 - p : p);
        const ncd::Target tg = ncd::target_p(p, upper);
        double out;
        if (ncd::newton([&](double x, bool u, double& t, double& f) { return special::nc_detail::ncf_lite(dfn, dfd, nc, x, u, t, &f); },
                        tg, x0, true, out))
            return out;
        return ncd::invert([&](double x, bool u, bool wide) { return tail(x, dfn, dfd, nc, u, wide); }, tg, x0, true);
    }
    static double icdf(double p, double dfn, double dfd, double nc) { return inv(p, dfn, dfd, nc, false); }
    static double iccdf(double p, double dfn, double dfd, double nc) { return inv(p, dfn, dfd, nc, true); }
    static double from_log(double lp, double dfn, double dfd, double nc, bool upper) {
        if (nc == 0.0) return upper ? F::ilogccdf(lp, dfn, dfd) : F::ilogcdf(lp, dfn, dfd);
        if (!(lp <= 0.0)) return kNaN;
        if (lp == -kInf) return inv(0.0, dfn, dfd, nc, upper);
        if (!ncd::log_target_ok(lp)) return kNaN;
        const ncd::Target tg = ncd::target_log(lp, upper);
        if (tg.p.m == 0.0) return inv(1.0, dfn, dfd, nc, upper);
        const double c = (dfn + 2.0 * nc) / (dfn + nc), k = (dfn + nc) / c;  // Patnaik's start, at the log target
        const double x0 = c * k / dfn * (upper ? F::ilogccdf(lp, k, dfd) : F::ilogcdf(lp, k, dfd));
        return ncd::invert([&](double x, bool u, bool wide) { return tail(x, dfn, dfd, nc, u, wide); }, tg, x0, true);
    }
    static double ilogcdf(double lp, double dfn, double dfd, double nc) { return from_log(lp, dfn, dfd, nc, false); }
    static double ilogccdf(double lp, double dfn, double dfd, double nc) { return from_log(lp, dfn, dfd, nc, true); }

    // F = A B, A = X / dfn (X ~ chi-square(dfn, nc)), B = dfd / V (V ~ chi-square(dfd)).
    // A's central moments from X's cumulants 2^(r-1) (r-1)! (dfn + r nc); B = dfd W, W an
    // inverse gamma (dfd/2, 1/2).
    static ncf_detail::Central central(double dfn, double dfd, double nc, double& ma, double& mb) {
        const double s2 = 2.0 * (dfn + 2.0 * nc), s3 = 8.0 * (dfn + 3.0 * nc);
        const double s4 = 48.0 * (dfn + 4.0 * nc) + 3.0 * s2 * s2;
        const double d = dfn;
        const double mx[5] = {1.0, 0.0, s2 / (d * d), s3 / (d * d * d), s4 / (d * d * d * d)};
        const double al = 0.5 * dfd;  // W: alpha = dfd/2, beta = 1/2; B = dfd W
        const double r1 = al - 1.0, r2 = al - 2.0, r3 = al - 3.0, r4 = al - 4.0;
        const double bw = 0.5 * dfd;  // beta times dfd
        const double v2 = bw * bw / (r1 * r1 * r2);
        const double v3 = 4.0 * bw * bw * bw / (r1 * r1 * r1 * r2 * r3);
        const double v4 = 3.0 * (al + 5.0) * bw * bw * bw * bw / (r1 * r1 * r1 * r1 * r2 * r3 * r4);
        const double my[5] = {1.0, 0.0, v2, dfd > 6.0 ? v3 : 0.0, dfd > 8.0 ? v4 : 0.0};
        ma = (dfn + nc) / dfn;
        mb = bw / r1;  // dfd / (dfd - 2)
        return ncf_detail::product_central(ma, mb, mx, my);
    }
    static double mean(double dfn, double dfd, double nc) {
        return dfd > 2.0 ? (dfn + nc) / dfn * (dfd / (dfd - 2.0)) : kInf;
    }
    // The central moments' products (dfn^4, the fourth power of 1 + nc / dfn, the square of
    // dfn + 2 nc, dfd^7) stay inside the doubles for these; beyond, the moments are not
    // formed: NaN.
    static bool moments_in_range(double dfn, double dfd, double nc) {
        return dfn <= 1e50 && nc <= 1e50 && nc <= 1e50 * dfn && dfd <= 1e40;
    }
    static double variance(double dfn, double dfd, double nc) {
        if (!(dfd > 4.0)) return dfd > 2.0 ? kInf : kNaN;
        if (!moments_in_range(dfn, dfd, nc)) return kNaN;
        double ma, mb;
        return central(dfn, dfd, nc, ma, mb).m2;
    }
    static double standard_deviation(double dfn, double dfd, double nc) {
        if (!(dfd > 4.0)) return dfd > 2.0 ? kInf : kNaN;
        if (!moments_in_range(dfn, dfd, nc)) return kNaN;
        double ma, mb;
        return std::sqrt(central(dfn, dfd, nc, ma, mb).m2);
    }
    static double skewness(double dfn, double dfd, double nc) {
        if (!(dfd > 6.0)) return dfd > 4.0 ? kInf : kNaN;
        if (!moments_in_range(dfn, dfd, nc)) return kNaN;
        double ma, mb;
        const ncf_detail::Central c = central(dfn, dfd, nc, ma, mb);
        return c.m3 / (c.m2 * std::sqrt(c.m2));
    }
    // m4 / m2^2 - 3 cancels (to 2e-13 at a kurtosis of 0.016): the excess m4 - 3 m2^2 of
    // P = b X + a Y + X Y in the cumulants of X and Y, every term positive (X a scaled
    // noncentral chi-square, Y a scaled inverse gamma: all their cumulants positive):
    //   k4X k4Y + 3 k4X m2Y^2 + 3 m2X^2 k4Y + 6 m2X^2 m2Y^2 + 4 b k4X m3Y + 12 b m2X^2 m3Y
    //   + 6 b^2 k4X m2Y + 12 b^2 m2X^2 m2Y + b^4 k4X + 4 a m3X k4Y + 12 a m3X m2Y^2
    //   + 12 a b m3X m3Y + 12 a b^2 m3X m2Y + 6 a^2 m2X k4Y + 12 a^2 m2X m2Y^2 + 12 a^2 b m2X m3Y + a^4 k4Y,
    // k4X = 48 (dfn + 4 nc) / dfn^4, k4Y = m2Y^2 (30 al - 66) / ((al - 3)(al - 4)), al = dfd/2.
    static double kurtosis(double dfn, double dfd, double nc) {
        if (!(dfd > 8.0)) return dfd > 4.0 ? kInf : kNaN;
        if (!moments_in_range(dfn, dfd, nc)) return kNaN;
        double a, b;
        const ncf_detail::Central c = central(dfn, dfd, nc, a, b);
        const double d = dfn, d2 = d * d;
        const double m2 = 2.0 * (dfn + 2.0 * nc) / d2, m3 = 8.0 * (dfn + 3.0 * nc) / (d2 * d);
        const double k4x = 48.0 * (dfn + 4.0 * nc) / (d2 * d2);
        const double al = 0.5 * dfd, r1 = al - 1.0, r2 = al - 2.0, r3 = al - 3.0, r4 = al - 4.0, bw = 0.5 * dfd;
        const double n2 = bw * bw / (r1 * r1 * r2), n3 = 4.0 * bw * bw * bw / (r1 * r1 * r1 * r2 * r3);
        const double k4y = n2 * n2 * (30.0 * al - 66.0) / (r3 * r4);
        const double m22 = m2 * m2, n22 = n2 * n2, b2 = b * b, a2 = a * a;
        const double ex = k4x * k4y + 3.0 * k4x * n22 + 3.0 * m22 * k4y + 6.0 * m22 * n22 + 4.0 * b * k4x * n3 +
                          12.0 * b * m22 * n3 + 6.0 * b2 * k4x * n2 + 12.0 * b2 * m22 * n2 + b2 * b2 * k4x +
                          4.0 * a * m3 * k4y + 12.0 * a * m3 * n22 + 12.0 * a * b * m3 * n3 + 12.0 * a * b2 * m3 * n2 +
                          6.0 * a2 * m2 * k4y + 12.0 * a2 * m2 * n22 + 12.0 * a2 * b * m2 * n3 + a2 * a2 * k4y;
        return ex / (c.m2 * c.m2);
    }
    // f'/f = E[(a_j - 1)/x - (a_j + b) dfn / (dfn x + dfd)] under the terms' weights, so the
    // mode solves x = (A - 1) dfd / (dfn (1 + b)), A = E[a_j]; dfn < 2: the density is
    // infinite at 0 (T21).
    static double mode(double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::mode(dfn, dfd);
        if (dfn < 2.0) return 0.0;
        const double b = 0.5 * dfd;
        auto phi = [&](double x) {
            const ncd::Series s = series(x, dfn, dfd, nc);
            return ((0.5 * dfn - 1.0) + s.mean_j) * dfd - dfn * (1.0 + b) * x;
        };
        const double at0 = phi(1e-300);
        if (!(at0 > 0.0)) return at0 != at0 ? kNaN : 0.0;
        auto g = [&](double x) { return -phi(x) / (dfn * (1.0 + b) * x); };  // increasing, scale-free
        const double x0 = std::max(1e-3, (0.5 * dfn + 0.5 * nc - 1.0) * dfd / (dfn * (1.0 + b)));
        return special::nc_detail::solve(g, x0, [](double x) { return std::log(x); },
                                         [](double u) { return std::exp(u); }, 0.125, 0.0, kInf);
    }
    static double entropy(double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::entropy(dfn, dfd);
        const double m = std::max((dfn + nc) / dfn, 1e-3);
        if (!std::isfinite(m)) return kNaN;
        return ncd::entropy(
            [&](double x) { return x > 0.0 && std::isfinite(x) ? log_density(x, dfn, dfd, nc) : num::DD{-kInf, 0.0}; },
            [&](double u) { return m * std::exp(u); }, [&](double u) { return m * std::exp(u); });
    }
    static double support_lo(double, double, double) { return 0.0; }
    static double support_hi(double, double, double) { return kInf; }
};

// ---------------------------------------------------------------- t

namespace nct_detail {

using num::DD;
namespace dd = num::dd;

// log of M_k(t) = int_0^inf s^k phi(t s - nc) f_S(s) ds, S = sqrt(V / df) (k = 1: the
// density). With s = e^v and z = t s - nc,
//     s^k phi(z) f_S(s) ds = e^F(v) dv / sqrt(2 pi),
//     F(v) = -z^2/2 + log(df/pi)/2 - log Gamma*(df/2) - (df/2)(e^2v - 1 - 2v) + (k - 1) v + v
// (as special's nct_opposite, with s^k): split at the stationary point s*, the root of
// (t^2 + df) s^2 - nc t s - (df + k) = 0, as F* (double-double) plus a remainder without
// cancellation; trapezoid in u over v - v* = c A sinh(u / A).
struct Moment {
    DD log;         // log M_k(t)
    double mean_z;  // E[t S - nc] under M_k's weights (computed when asked)
};
inline Moment moment(int k, double df, double nc, double t, bool want_z = false) {
    const double a = -nc;
    const double C0 = df + k;
    // s*: in w = |t| s for |t| > 1 ((1 + df/t^2) w^2 + a sign(t) w - C0 = 0): t^2 overflows past 1e154
    double sstar;
    if (std::fabs(t) > 1.0) {
        const double at = std::fabs(t), A = 1.0 + df / at / at, B = t > 0.0 ? a : -a;
        const double disc = std::sqrt(B * B + 4.0 * A * C0);
        sstar = (B >= 0.0 ? 2.0 * C0 / (B + disc) : (-B + disc) / (2.0 * A)) / at;
    } else {
        const double A2 = t * t + df, B1 = a * t;
        const double disc = std::sqrt(B1 * B1 + 4.0 * A2 * C0);
        sstar = B1 >= 0.0 ? 2.0 * C0 / (B1 + disc) : (-B1 + disc) / (2.0 * A2);
    }
    const double p = t * sstar;
    const DD zdd = dd::two_prod(t, sstar) + a;
    const double z = zdd.hi;
    const DD s2 = dd::two_prod(sstar, sstar);
    const DD lnS = dd::log(sstar);
    const DD hstar = s2 - 1.0 - lnS * 2.0;
    const DD Fstar = -(zdd * zdd) * 0.5 + (dd::log(df) - dd::log(special::kPi)) * 0.5 - hstar * (0.5 * df) -
                     std::log(special::gamma_star(0.5 * df)) + lnS * static_cast<double>(k);
    const double c1 = (-(zdd * DD{p, 0.0}) - (s2 - 1.0) * df).hi + k;
    const double kappa = p * p + z * p + 2.0 * df * sstar * sstar;
    // The grid is laid about s* as the peak (special's nct_opposite_s): one that the rounding
    // of s* leaves further than the peak's width from it (|nc| or |t| beyond 1e15 or so) is not
    // integrated.
    if (!(std::fabs(c1) <= std::sqrt(kappa))) return {DD{kNaN, 0.0}, kNaN};
    const double sigma = 1.0 / std::sqrt(std::max(kappa, 1e-300));
    using special::nc_detail::em1mx;
    auto integrand = [&](double delta) {
        const double em = em1mx(delta), q = p * std::expm1(delta);
        const double dF = c1 * delta - z * p * em - 0.5 * q * q - 0.5 * df * (sstar * sstar) * em1mx(2.0 * delta);
        return std::exp(dF);
    };
    constexpr double kStep = 0.25, kA = 8.0;
    const double c = std::min(sigma, 0.4);
    // z at a node, z* + p (e^delta - 1) (z* to double-double: no cancellation near z = 0)
    auto zat = [&](double delta) { return zdd.hi + (zdd.lo + p * std::expm1(delta)); };
    disc::KahanSum I, Iz;
    I.add(integrand(0.0));
    if (want_z) Iz.add(integrand(0.0) * zat(0.0));
    for (int dir = -1; dir <= 1; dir += 2) {
        bool faded = false;
        for (int i = 1; i < special::nc_detail::kNctNodes; ++i) {  // (while sinh(u) is finite)
            const double u = dir * i * kStep / kA;
            const double delta = c * kA * std::sinh(u);
            const double v = integrand(delta) * std::cosh(u);
            I.add(v);
            if (want_z) Iz.add(v * zat(delta));
            if (!(v > 0x1p-60 * I.s) && i > 8) {  // (a NaN node ends it: the sum is NaN)
                faded = true;
                ncd::work_done += i;
                break;
            }
        }
        if (!faded) return {DD{kNaN, 0.0}, kNaN};
    }
    return {Fstar - num::kLogSqrt2Pi + std::log(I.value() * kStep * c), want_z ? Iz.value() / I.value() : 0.0};
}
inline DD log_moment(int k, double df, double nc, double t) { return moment(k, df, nc, t).log; }

// R = sqrt(df / V), x = df / 2: L_1 = log E[R] and D_k = log E[R^k] - k L_1 (k = 2..4) as
// double-doubles: the 1/x series (coeffs/nct_moments.hpp) from x = 32, else from
// log_gamma (no cancellation that matters there). (From x = 8 the series' truncation left
// 2e-13 of the moments at x = 8.6: its coefficients grow faster than its terms' x^-n fall
// there; at 32 it is below 1e-19.)
inline DD series_term(int r, double x) {
    DD s{0.0, 0.0};
    const DD inv = num::div(DD{1.0, 0.0}, x);
    DD p = inv;
    double prev = kInf;
    for (int n = 0; n < kNctTerms; ++n) {
        const DD term = DD{kNct[r][n][0], kNct[r][n][1]} * p;
        const double m = std::fabs(term.hi);
        if (m > prev) break;  // the asymptotic series turns
        s = s + term;
        if (m <= std::fabs(s.hi) * 0x1p-110) break;
        prev = m;
        p = p * inv;
    }
    return s;
}
inline DD log_raw(double k, double x) {  // log E[R^k] = log Gamma(x - k/2) - log Gamma(x) + (k/2) log x
    return num::log_gamma(DD{x, 0.0} - 0.5 * k) - num::log_gamma(DD{x, 0.0}) + num::log_dd(x) * (0.5 * k);
}
struct RMoments {
    DD l1, d[5];  // d[2..4]
};
inline RMoments r_moments(double df, int kmax) {
    const double x = 0.5 * df;
    RMoments m{};
    if (x >= 32.0) {
        m.l1 = series_term(0, x);
        for (int k = 2; k <= kmax; ++k) m.d[k] = series_term(k - 1, x);
    } else {
        m.l1 = log_raw(1.0, x);
        for (int k = 2; k <= kmax; ++k) m.d[k] = log_raw(k, x) - m.l1 * static_cast<double>(k);
    }
    return m;
}

}  // namespace nct_detail

struct NoncentralT {
    static constexpr char name[] = "noncentralt";
    static constexpr std::array<const char*, 2> params{"df", "nc"};

    static bool valid(double df, double nc) { return df > 0.0 && std::isfinite(df) && std::isfinite(nc); }
    // df = inf: the normal (nc, 1), every method (as special.nctdtr)
    static bool at_limit(double df, double nc) { return df == kInf && std::isfinite(nc); }
    template <class M>
    static double limit_in(double x, double, double nc) { return method_of<Normal, M>(x, nc, 1.0); }
    template <class M>
    static double limit_par(double, double nc) { return method_of<Normal, M>(nc, 1.0); }

    // The bulk: the density's series in u = nc t sqrt(2 / (df + t^2)),
    //     f(t) = t(t; df) e^(-mu) sum_j Gamma(c + j/2) / Gamma(c) u^j / j!,   c = (df + 1) / 2, mu = nc^2 / 2,
    // its even and odd terms two hypergeometric series in q = u^2 / 4 = mu t^2 / (df + t^2):
    //     E = sum_m (c)_m q^m / ((1/2)_m m!),   O = sum_m (c + 1/2)_m q^m / ((3/2)_m m!),
    //     f = t(t; df) e^(-mu) (E + sign(nc t) G |u| O),   G = Gamma(c + 1/2) / Gamma(c).
    // On the side of nc every term is positive. On the other the two cancel (which is why the
    // definition is the integral): taken only while (E + P) / (E - P) <= 2^9, P = G |u| O, with
    // E, O and |u| in "double + error" arithmetic (their truncation set 2^-12 lower) and G to
    // 3e-19 (log_gamma's 1e-19 twice, exp_full), so the difference keeps 2^9 * 3e-19 = 1.5e-16.
    // In logs as the other two; t = 0 and the far tails: the integral.
    static bool log_density_bulk(double t, double df, double nc, num::DD& l) {
        if (!(t != 0.0 && std::fabs(t) < 0x1p500 && df <= 0x1p40)) return false;
        const num::DD mu = num::dd::two_prod(nc, nc) * 0.5;
        const special::distr_detail::Arg g = special::distr_detail::t_arg(df, t);  // g.y = t^2 / (df + t^2)
        if (!(g.y.hi >= 0x1p-900 && mu.hi >= 0x1p-900)) return false;
        const ncd::Lite q = ncd::lite::mul({mu.hi, mu.lo}, {g.y.hi, g.y.lo});
        const double c = 0.5 * df + 0.5;
        // the largest terms: m^2 ~ q (c + m)
        if (!(0.5 * (q.h + std::sqrt(q.h * q.h + 4.0 * c * q.h)) <= 250.0)) return false;
        const bool same = (nc > 0.0) == (t > 0.0);
        const double tol = same ? ncd::lite::kLiteTol : ncd::lite::kLiteTol * 0x1p-12;
        ncd::Lite E, O;
        if (!ncd::hyper_sum(ncd::HalfRatio{q, ncd::lite::sum2(0.5 * df, 0.5), 0.5}, tol, E) ||
            !ncd::hyper_sum(ncd::HalfRatio{q, ncd::lite::sum2(0.5 * df, 1.0), 1.5}, tol, O))
            return false;
        // |u| = |nc| sqrt(2 y) (the root's remainder by fma); G per df
        const double s = std::sqrt(2.0 * g.y.hi);
        const ncd::Lite root{s, (std::fma(-s, s, 2.0 * g.y.hi) + 2.0 * g.y.lo) / (2.0 * s)};
        thread_local num::Memo memo;
        const num::DD G = memo.get(df, 0.0, [&] {
            return num::exp_full(num::log_gamma(num::dd::two_sum(0.5 * df, 1.0)) - num::log_gamma(num::dd::two_sum(0.5 * df, 0.5)));
        });
        const ncd::Lite P = ncd::lite::mul(ncd::lite::mul(ncd::lite::mul({G.hi, G.lo}, {std::fabs(nc), 0.0}), root), O);
        ncd::Lite total = E;
        if (same) {
            ncd::lite::add(total, P);
        } else {
            ncd::lite::add(total, {-P.h, -P.l});
            if (!(total.h * 0x1p9 >= E.h + P.h)) return false;
        }
        l = StudentT::log_density(t, df) - mu + num::dd::log(ncd::to_dd(total));
        return std::isfinite(l.hi);
    }
    // log f(t), t finite
    static num::DD log_density(double t, double df, double nc) {
        num::DD l;
        return log_density_bulk(t, df, nc, l) ? l : nct_detail::log_moment(1, df, nc, t);
    }
    static double pdf(double t, double df, double nc) {
        if (nc == 0.0) return StudentT::pdf(t, df);
        if (std::isinf(t)) return 0.0;
        return ncd::density(log_density(t, df, nc));
    }
    static double logpdf(double t, double df, double nc) {
        if (nc == 0.0) return StudentT::logpdf(t, df);
        if (std::isinf(t)) return -kInf;
        const num::DD l = log_density(t, df, nc);
        return l.hi + l.lo;
    }
    static double cdf(double t, double df, double nc) {
        return nc == 0.0 ? StudentT::cdf(t, df) : special::Nctdtr::f(df, nc, t);
    }
    static double ccdf(double t, double df, double nc) {
        return nc == 0.0 ? StudentT::ccdf(t, df) : special::Nctdtrc::f(df, nc, t);
    }
    static special::Scaled tail(double t, double df, double nc, bool upper, bool wide = true) {
        return special::nc_detail::nctdtr_tail(df, nc, t, upper, wide);
    }
    static double log_of_tail(double t, double df, double nc, bool upper) {
        const bool inside = std::isfinite(t);
        return ncd::log_tail_either([&](bool u, double& v) { return inside && special::nc_detail::nct_lite(df, nc, t, u, v); },
                                    [&](bool u, bool wide) { return tail(t, df, nc, u, wide); }, upper, inside);
    }
    static double logcdf(double t, double df, double nc) {
        if (nc == 0.0) return StudentT::logcdf(t, df);
        return log_of_tail(t, df, nc, false);
    }
    static double logccdf(double t, double df, double nc) {
        if (nc == 0.0) return StudentT::logccdf(t, df);
        return log_of_tail(t, df, nc, true);
    }
    // start: the special's (P(T <= t) ~ Phi of a normalized t)
    static double start(double p, double df, double nc, bool upper) {
        const double pl = upper ? 1.0 - p : p;
        const double z = special::distr_detail::one<special::Ndtri>(std::clamp(pl, 1e-300, 1.0 - 1e-16));
        const double A = 1.0 - 0.25 / df, qa = A * A - z * z / (2.0 * df);
        double t0 = nc + z;
        if (qa > 0.0) {
            const double d = A * A * nc * nc - qa * (nc * nc - z * z);
            if (d >= 0.0) t0 = (A * nc + std::copysign(std::sqrt(d), z)) / qa;
        }
        return t0;
    }
    // the bracketing solver on the wide tails (0 < p < 1, nc != 0)
    static double inv_precise(double p, double df, double nc, bool upper) {
        return ncd::invert([&](double t, bool u, bool wide) { return tail(t, df, nc, u, wide); }, ncd::target_p(p, upper),
                           start(p, df, nc, upper), false);
    }
    static double inv(double p, double df, double nc, bool upper) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (nc == 0.0) return upper ? StudentT::iccdf(p, df) : StudentT::icdf(p, df);
        if (p == 0.0) return upper ? kInf : -kInf;
        if (p == 1.0) return upper ? -kInf : kInf;
        double out;
        if (ncd::newton([&](double t, bool u, double& v, double& f) { return special::nc_detail::nct_lite(df, nc, t, u, v, &f); },
                        ncd::target_p(p, upper), start(p, df, nc, upper), false, out))
            return out;
        return inv_precise(p, df, nc, upper);
    }
    static double icdf(double p, double df, double nc) { return inv(p, df, nc, false); }
    static double iccdf(double p, double df, double nc) { return inv(p, df, nc, true); }
    static double from_log(double lp, double df, double nc, bool upper) {
        if (nc == 0.0) return upper ? StudentT::ilogccdf(lp, df) : StudentT::ilogcdf(lp, df);
        if (!(lp <= 0.0)) return kNaN;
        if (lp == -kInf) return inv(0.0, df, nc, upper);
        if (!ncd::log_target_ok(lp)) return kNaN;
        const ncd::Target tg = ncd::target_log(lp, upper);
        if (tg.p.m == 0.0) return inv(1.0, df, nc, upper);
        // start: the central t's quantile at the log target, shifted by nc
        const double t0 = nc + (upper ? StudentT::ilogccdf(lp, df) : StudentT::ilogcdf(lp, df));
        return ncd::invert([&](double t, bool u, bool wide) { return tail(t, df, nc, u, wide); }, tg, t0, false);
    }
    static double ilogcdf(double lp, double df, double nc) { return from_log(lp, df, nc, false); }
    static double ilogccdf(double lp, double df, double nc) { return from_log(lp, df, nc, true); }

    // T = U R, U ~ N(nc, 1), R = sqrt(df / V) with r = E[R] and central moments y2..y4
    // (= r^k times E_2, E_3 - 3 E_2, E_4 - 4 E_3 + 6 E_2, E_k = e^(D_k) - 1):
    //   Var T = nc^2 y2 + E[R^2],                       E[R^2] = df / (df - 2)
    //   mu_3  = nc (nc^2 y3 + 3 y3 + 6 r y2)
    //   mu_4 - 3 Var^2 = nc^4 k4(R) + 6 nc^2 (2 r y3 + y4 - y2^2) + 3 Var(R^2),
    //   k4(R) = r^4 (E_4 - 4 E_3 + 6 E_2 - 3 E_2^2),    Var(R^2) = 2 df^2 / ((df - 2)^2 (df - 4)),
    // every term positive (T - nc r = nc (R - r) + R Z, Z ~ N(0, 1) independent of R).
    struct Moments {
        double r, y2, y3, y4, k4;
    };
    static Moments moments(double df, int kmax) {
        const nct_detail::RMoments m = nct_detail::r_moments(df, kmax);
        Moments o{};
        o.r = num::exp_dd(m.l1);
        const num::DD e2 = num::expm1_full(m.d[2]);
        o.y2 = o.r * o.r * (e2.hi + e2.lo);
        if (kmax >= 3) {
            const num::DD e3 = num::expm1_full(m.d[3]);
            const num::DD n3 = e3 - e2 * 3.0;
            o.y3 = o.r * o.r * o.r * (n3.hi + n3.lo);
            if (kmax >= 4) {
                const num::DD e4 = num::expm1_full(m.d[4]);
                const num::DD n4 = e4 - e3 * 4.0 + e2 * 6.0;
                const num::DD k4 = n4 - e2 * e2 * 3.0;
                const double r4 = o.r * o.r * o.r * o.r;
                o.y4 = r4 * (n4.hi + n4.lo);
                o.k4 = r4 * (k4.hi + k4.lo);
            }
        }
        return o;
    }
    static double mean(double df, double nc) {
        if (!(df > 1.0)) return kNaN;
        if (nc == 0.0) return 0.0;
        return nc * num::exp_dd(nct_detail::r_moments(df, 1).l1);
    }
    static double variance(double df, double nc) {
        if (!(df > 2.0)) return df > 1.0 ? kInf : kNaN;
        const Moments m = moments(df, 2);
        if (!(std::fabs(nc) < 0x1p500)) return nc * m.y2 * nc + df / (df - 2.0);  // (nc^2 passes the doubles)
        return nc * nc * m.y2 + df / (df - 2.0);
    }
    static double standard_deviation(double df, double nc) {
        if (!(df > 2.0)) return df > 1.0 ? kInf : kNaN;
        return std::sqrt(variance(df, nc));
    }
    // The third and fourth central moments of R are differences of the E_k: y3 keeps
    // 2^-104 df of its value and k4 2^-104 df^2, and they weigh nc^2 / df and nc^4 / df^2 of the
    // skewness and the kurtosis. Beyond min(df, nc^2) = 1e17 (skewness) and min(df^2, nc^4) =
    // 1e17 (kurtosis) that is no longer 1e-15, and beyond |nc| = 1e75 the powers of nc pass
    // the doubles: NaN.
    static double skewness(double df, double nc) {
        if (!(df > 3.0)) return kNaN;
        if (nc == 0.0) return 0.0;
        if (!(std::fabs(nc) <= 1e75 && (df <= 1e17 || std::fabs(nc) <= 3e8))) return kNaN;
        const Moments m = moments(df, 3);
        const double v = nc * nc * m.y2 + df / (df - 2.0);
        const double mu3 = nc * (nc * nc * m.y3 + 3.0 * m.y3 + 6.0 * m.r * m.y2);
        return mu3 / (v * std::sqrt(v));
    }
    static double kurtosis(double df, double nc) {
        if (!(df > 4.0)) return df > 2.0 ? kInf : kNaN;
        if (nc == 0.0) return StudentT::kurtosis(df);
        if (!(std::fabs(nc) <= 1e75 && (df <= 3e8 || std::fabs(nc) <= 1.7e4))) return kNaN;
        const Moments m = moments(df, 4);
        const double v = nc * nc * m.y2 + df / (df - 2.0);
        const double vr2 = 2.0 * df * df / ((df - 2.0) * (df - 2.0) * (df - 4.0));
        const double n2 = nc * nc;
        const double ex = n2 * n2 * m.k4 + 6.0 * n2 * (2.0 * m.r * m.y3 + m.y4 - m.y2 * m.y2) + 3.0 * vr2;
        return ex / (v * v);
    }
    // f'(t) = -(t M_3 - nc M_2) = -M_2 E[t S - nc] (the expectation under M_2's weights
    // s^2 phi(t s - nc) f_S(s)): the mode solves E[z] = 0, z = t s - nc, summed on one grid.
    // (As t M_3 / M_2 = nc, two moments' independent rounding, nc eps, met a flat g: 4e-13 of
    // the mode at df = 0.6, nc = 75; E[z] carries E|z| eps.)
    //
    // E[z] is of size 1 / |nc| where the integrand's z are of size 1, each to an eps: the root
    // keeps about 0.1 eps |nc| / sqrt(1 + df) of itself (measured against Arb: 1.1e-14 at
    // df = 0.1, nc = 500; 1.6e-13 at df = 9, nc = 1e5). The mode is given while that is below
    // 3e-15, |nc| <= 128 sqrt(1 + df) (every |nc| <= 100 of the guaranteed region); else NaN.
    static double mode_root(double df, double nc) {
        auto g = [&](double t) { return nct_detail::moment(2, df, nc, t, true).mean_z; };
        const double t0 = nc * std::sqrt(df / (df + 2.5));
        return special::nc_detail::solve(g, t0, [](double t) { return std::asinh(t); },
                                         [](double u) { return std::sinh(u); }, 0.125, -kInf, kInf);
    }
    static double mode(double df, double nc) {
        if (nc == 0.0) return 0.0;
        if (!(std::fabs(nc) <= 128.0 * std::sqrt(1.0 + df))) return kNaN;
        return mode_root(df, nc);
    }
    static double entropy(double df, double nc) {
        if (nc == 0.0) return StudentT::entropy(df);
        const double m = mode_root(df, nc);  // (the grid's centre: any point of the bell)
        const double s = df > 2.0 ? std::sqrt(variance(df, nc)) : 1.0 + std::fabs(nc);
        return ncd::entropy([&](double t) { return std::isfinite(t) ? log_density(t, df, nc) : num::DD{-kInf, 0.0}; },
                            [&](double u) { return m + s * std::sinh(u); }, [&](double u) { return s * std::cosh(u); });
    }
    static double support_lo(double, double) { return -kInf; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
