// The nuisance parameter of Barnard's and Boschloo's exact tests (milestone H6,
// docs/design/htest.md §3.7; decision H78): the p-value is sup over pi in [0, 1] of
//
//     f(pi) = sum over the tables (x1, x2) of the rejection region of
//             C(n1, x1) C(n2, x2) pi^s (1 - pi)^(N - s),   s = x1 + x2, N = n1 + n2,
//
// a polynomial of degree N in Bernstein form: f = sum_s beta_s B_s(pi), B_s the binomial pmf
// of (N, pi), beta_s = P(region | s) in [0, 1] (the hypergeometric mass of the region's tables
// with that total). scipy maximizes it by shgo from a grid of n points, which may miss the
// supremum without telling. Here the supremum is certified to a relative 2^-48:
//
//   - beta_s comes from the hypergeometric pmf of each total: its log at the mode (sd.stats'
//     double-double log pmf) and the exact ratios of neighbours in double-doubles, kept with a
//     power of two (no underflow), summed over the region's tables; beta is then normalized by
//     its largest value (so a deep tail is computed at the scale of 1: sup f >= max beta_s /
//     (2.5 sqrt N));
//   - f, f' and f'' at a point are sums over s of the beta's (differences) times B_s, B_s by its
//     ratio recurrence in double-doubles from the binomial's log pmf at the mode;
//   - on an interval [a, b] (midpoint m, half width h), f <= f(m) + |f'(m)| h + max(0, f''(m))
//     h^2 / 2 + M3 h^3 / 6 (Taylor's form), M3 >= |f'''| on [a, b] from the third differences
//     of beta and the largest B_s on [a, b] (at the end nearer s / N, or Stirling's bound of
//     B_s(s / N) when s / N is inside), each rounding covered;
//   - branch and bound: n intervals first (the n of scipy's grid: decision H78), the largest
//     bound bisected until every bound is within (1 + 2^-48) of the largest value seen; the
//     value is that largest f (a value of f, to rounding). The intervals evaluated are capped:
//     past the cap (or below a width of 1e-15) the result is NaN (docs/design/htest.md §4).
#pragma once

#include "htest/moments.hpp"
#include "stats/dist/discrete_basic.hpp"
#include "stats/dist/discrete_more.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

namespace sd::htest::nuisance {

inline constexpr double kU = 0x1p-53;  // the unit roundoff

// A double-double times 2^e: the hypergeometric terms of one total relative to the mode's.
struct Scaled {
    DD m{1.0, 0.0};
    int e = 0;
    void renorm() {
        if (m.hi != 0.0 && std::fabs(m.hi) < 0x1p-600) {
            m = dd::ldexp(m, 600);
            e -= 600;
        }
    }
};

// One total s of a 2 x 2 table with column sums n1, n2: x1 from lo to hi, P(X1 = x1) =
// C(n1, x1) C(n2, s - x1) / C(N, s) = exp(lmode) r[x1 - lo] 2^e[x1 - lo].
struct Total {
    std::int64_t lo = 0, hi = 0, mode = 0;
    DD lmode{0.0, 0.0};
    std::vector<DD> r;
    std::vector<int> e;

    Total(std::int64_t n1, std::int64_t n2, std::int64_t s) {
        const std::int64_t N = n1 + n2;
        lo = std::max<std::int64_t>(0, s - n2);
        hi = std::min<std::int64_t>(n1, s);
        mode = std::clamp<std::int64_t>((s + 1) * (n1 + 1) / (N + 2), lo, hi);
        const double Nd = static_cast<double>(N), n1d = static_cast<double>(n1), sd = static_cast<double>(s);
        lmode = stats::Hypergeometric::log_pmf(static_cast<double>(mode), Nd, n1d, sd);
        const auto len = static_cast<std::size_t>(hi - lo + 1);
        r.assign(len, DD{0.0, 0.0});
        e.assign(len, 0);
        r[static_cast<std::size_t>(mode - lo)] = DD{1.0, 0.0};
        Scaled v;
        for (std::int64_t x = mode; x < hi; ++x) {  // P(x + 1) / P(x) = (n1 - x)(s - x) / ((x + 1)(n2 - s + x + 1))
            const double a = static_cast<double>((n1 - x) * (s - x)),
                         b = static_cast<double>((x + 1) * (n2 - s + x + 1));
            v.m = v.m * (DD{a, 0.0} / DD{b, 0.0});
            v.renorm();
            r[static_cast<std::size_t>(x + 1 - lo)] = v.m, e[static_cast<std::size_t>(x + 1 - lo)] = v.e;
        }
        v = Scaled{};
        for (std::int64_t x = mode; x > lo; --x) {  // P(x - 1) / P(x) = x (n2 - s + x) / ((n1 - x + 1)(s - x + 1))
            const double a = static_cast<double>(x * (n2 - s + x)), b = static_cast<double>((n1 - x + 1) * (s - x + 1));
            v.m = v.m * (DD{a, 0.0} / DD{b, 0.0});
            v.renorm();
            r[static_cast<std::size_t>(x - 1 - lo)] = v.m, e[static_cast<std::size_t>(x - 1 - lo)] = v.e;
        }
    }
    // log P(X1 = x1)
    DD log_at(std::int64_t x) const {
        const auto i = static_cast<std::size_t>(x - lo);
        return lmode + dd::log(r[i]) + stats::num::kLn2 * static_cast<double>(e[i]);
    }
    // log of the sum of P(X1 = x1) over the x1 with in(x1); -inf when there is none
    template <class In>
    DD log_sum(In&& in) const {
        int emax = -(1 << 30);
        for (std::int64_t x = lo; x <= hi; ++x)
            if (in(x)) emax = std::max(emax, e[static_cast<std::size_t>(x - lo)]);
        if (emax == -(1 << 30)) return DD{-kInf, 0.0};
        Acc acc;
        for (std::int64_t x = lo; x <= hi; ++x) {
            const auto i = static_cast<std::size_t>(x - lo);
            if (!in(x) || e[i] - emax < -1100) continue;
            acc.add(dd::ldexp(r[i], e[i] - emax));
        }
        return lmode + dd::log(acc.value()) + stats::num::kLn2 * static_cast<double>(emax);
    }
};

// The maximization of f = sum_s beta_s B_s(pi) over [0, 1].
class Sup {
public:
    // lbeta[s] = log beta_s (double-doubles; -inf for an empty total), s = 0..N
    explicit Sup(const std::vector<DD>& lbeta) : N_(static_cast<std::int64_t>(lbeta.size()) - 1) {
        DD top{-kInf, 0.0};
        for (const DD& v : lbeta)
            if (v.hi > top.hi || (v.hi == top.hi && v.lo > top.lo)) top = v;
        lbmax_ = top;
        bt_.resize(lbeta.size());
        for (std::size_t s = 0; s < lbeta.size(); ++s)
            bt_[s] = lbeta[s].hi == -kInf ? 0.0 : stats::num::exp_dd(lbeta[s] - top);
        rho_.resize(static_cast<std::size_t>(std::max<std::int64_t>(N_, 0)));
        for (std::int64_t s = 0; s < N_; ++s)
            rho_[static_cast<std::size_t>(s)] =
                DD{static_cast<double>(N_ - s), 0.0} / DD{static_cast<double>(s + 1), 0.0};
        B_.resize(lbeta.size());
    }

    // log sup f (NaN past the cap: `cap` evaluations of O(N) each); n the first grid's size
    DD run(std::int64_t n, double cap) {
        if (lbmax_.hi == -kInf) return DD{-kInf, 0.0};
        if (N_ == 0) return lbmax_;
        evals_ = 0;
        cap_ = cap;
        L_ = std::max(bt_[0], bt_[static_cast<std::size_t>(N_)]);
        Lbest_ = DD{L_, 0.0};
        std::priority_queue<Piece> heap;
        const double tol = 1.0 + 0x1p-48;
        auto push = [&](double a, double b) -> bool {
            Piece p{a, b, 0.0};
            if (!bound(p)) return false;
            if (p.u > L_ * tol) heap.push(p);
            return true;
        };
        for (std::int64_t i = 0; i < n; ++i) {
            const double a = static_cast<double>(i) / static_cast<double>(n),
                         b = static_cast<double>(i + 1) / static_cast<double>(n);
            if (i > 0) value(a);
            if (!push(a, b)) return DD{kNaN, 0.0};
        }
        while (!heap.empty()) {
            const Piece p = heap.top();
            if (p.u <= L_ * tol) break;
            heap.pop();
            const double m = 0.5 * (p.a + p.b);
            if (!(p.a < m && m < p.b) || p.b - p.a < 1e-15) return DD{kNaN, 0.0};
            if (!push(p.a, m) || !push(m, p.b)) return DD{kNaN, 0.0};
        }
        return Lbest_.hi > 0.0 ? lbmax_ + dd::log(Lbest_) : DD{-kInf, 0.0};
    }

private:
    struct Piece {
        double a, b, u;
        bool operator<(const Piece& o) const { return u < o.u; }
    };
    std::int64_t N_;
    DD lbmax_{0.0, 0.0};
    std::vector<double> bt_;
    std::vector<DD> rho_, B_;
    double L_ = 0.0, cap_ = 0.0;
    DD Lbest_{0.0, 0.0};
    double evals_ = 0.0;

    double b(std::int64_t s) const { return s < 0 || s > N_ ? 0.0 : bt_[static_cast<std::size_t>(s)]; }

    // B_s(pi) for every s (B_), 0 < pi < 1, by the ratio recurrence from the mode
    void binomials(double pi) {
        const DD q = DD{pi, 0.0} / dd::two_sum(1.0, -pi);
        const double Nd = static_cast<double>(N_);
        const std::int64_t s0 = std::clamp<std::int64_t>(static_cast<std::int64_t>(std::floor((Nd + 1.0) * pi)), 0, N_);
        const DD l0 = stats::binom_detail::log_pmf(static_cast<double>(s0), Nd, pi, dd::two_sum(1.0, -pi));
        std::fill(B_.begin(), B_.end(), DD{0.0, 0.0});
        DD v{stats::num::exp_dd(l0), 0.0};
        B_[static_cast<std::size_t>(s0)] = v;
        for (std::int64_t s = s0; s < N_ && v.hi > 1e-300; ++s) {
            v = v * (rho_[static_cast<std::size_t>(s)] * q);
            B_[static_cast<std::size_t>(s + 1)] = v;
        }
        v = B_[static_cast<std::size_t>(s0)];
        for (std::int64_t s = s0; s > 0 && v.hi > 1e-300; --s) {
            v = v / (rho_[static_cast<std::size_t>(s - 1)] * q);
            B_[static_cast<std::size_t>(s - 1)] = v;
        }
    }

    // f(pi) (normalized), a value of the polynomial to a few rounding errors
    DD value(double pi) {
        evals_ += 1.0;
        if (pi <= 0.0) return DD{bt_[0], 0.0};
        if (pi >= 1.0) return DD{bt_[static_cast<std::size_t>(N_)], 0.0};
        binomials(pi);
        Acc f;
        for (std::int64_t s = 0; s <= N_; ++s) {
            const double bs = bt_[static_cast<std::size_t>(s)];
            if (bs != 0.0) f.add(B_[static_cast<std::size_t>(s)] * bs);
        }
        const DD v = f.value();
        if (to_double(v) > L_) L_ = to_double(v), Lbest_ = v;
        return v;
    }

    // the upper bound of f on [p.a, p.b] into p.u; false past the cap
    bool bound(Piece& p) {
        if (evals_ * static_cast<double>(N_ + 1) > cap_) return false;
        const double a = p.a, bb = p.b, m = 0.5 * (a + bb), h = 0.5 * (bb - a);
        const double fm = to_double(value(m));
        // f' and f'' at m from the binomials just made (B_ at m), the side whose factor is smaller
        const bool low = m <= 0.5;
        Acc d1, d2;
        double a1 = 0.0, a2 = 0.0;
        for (std::int64_t s = 0; s <= N_; ++s) {
            const double B = to_double(B_[static_cast<std::size_t>(s)]);
            if (B == 0.0) continue;
            double c1, c2, g1, g2;
            if (low) {
                const double w1 = static_cast<double>(N_ - s), w2 = w1 * static_cast<double>(N_ - s - 1);
                c1 = w1 * (b(s + 1) - b(s)), g1 = w1 * (b(s + 1) + b(s));
                c2 = w2 * (b(s + 2) - 2.0 * b(s + 1) + b(s)), g2 = w2 * (b(s + 2) + 2.0 * b(s + 1) + b(s));
            } else {
                const double w1 = static_cast<double>(s), w2 = w1 * static_cast<double>(s - 1);
                c1 = w1 * (b(s) - b(s - 1)), g1 = w1 * (b(s) + b(s - 1));
                c2 = w2 * (b(s) - 2.0 * b(s - 1) + b(s - 2)), g2 = w2 * (b(s) + 2.0 * b(s - 1) + b(s - 2));
            }
            d1.add(dd::two_prod(c1, B)), d2.add(dd::two_prod(c2, B));
            a1 += g1 * B, a2 += g2 * B;
        }
        const double fac = low ? 1.0 / (1.0 - m) : 1.0 / m;
        const double e1 = 8.0 * kU * a1 * fac * (1.0 + 1e-10), e2 = 8.0 * kU * a2 * fac * fac * (1.0 + 1e-10);
        const double f1 = std::fabs(to_double(d1.value())) * fac + e1;
        const double f2 = to_double(d2.value()) * fac * fac + e2;
        const double M3 = third(a, bb);
        if (!std::isfinite(M3)) {
            p.u = kInf;
            return true;
        }
        p.u = fm * (1.0 + 8.0 * kU) + f1 * h + std::max(0.0, f2) * h * h / 2.0 + M3 * h * h * h / 6.0 + 1e-290;
        return true;
    }

    // B_s(pi) of one s at one point by its log pmf (bounds only)
    double pmf(std::int64_t s, double pi) const {
        if (pi <= 0.0) return s == 0 ? 1.0 : 0.0;
        if (pi >= 1.0) return s == N_ ? 1.0 : 0.0;
        return stats::num::exp_dd(
            stats::binom_detail::log_pmf(static_cast<double>(s), static_cast<double>(N_), pi, dd::two_sum(1.0, -pi))
        );
    }

    // an upper bound of |f'''| on [a, b] (inf when neither form's factor is finite)
    double third(double a, double bb) {
        if (N_ < 3) return 0.0;
        const double flow = bb < 1.0 ? 1.0 / ((1.0 - bb) * (1.0 - bb) * (1.0 - bb)) : kInf;
        const double fhigh = a > 0.0 ? 1.0 / (a * a * a) : kInf;
        if (!std::isfinite(flow) && !std::isfinite(fhigh)) return kInf;
        const bool low = flow <= fhigh;
        const double Nd = static_cast<double>(N_);
        // the largest B_s on [a, b]: at a below N a, at b above N b, else Stirling's bound
        const auto sa = static_cast<std::int64_t>(std::floor(Nd * a)),
                   sb = static_cast<std::int64_t>(std::ceil(Nd * bb));
        std::vector<double> top(static_cast<std::size_t>(N_ + 1), 0.0);
        if (sa >= 0) {
            double v = pmf(std::min(sa, N_), a);
            for (std::int64_t s = std::min(sa, N_); s >= 0; --s) {
                top[static_cast<std::size_t>(s)] = v;
                if (s > 0 && a > 0.0) v *= static_cast<double>(s) / (Nd - s + 1.0) * ((1.0 - a) / a);
                else v = 0.0;
                if (v < 1e-300) v = 0.0;
            }
        }
        if (sb <= N_) {
            double v = pmf(std::max<std::int64_t>(sb, 0), bb);
            for (std::int64_t s = std::max<std::int64_t>(sb, 0); s <= N_; ++s) {
                top[static_cast<std::size_t>(s)] = std::max(top[static_cast<std::size_t>(s)], v);
                if (s < N_ && bb < 1.0) v *= (Nd - s) / (s + 1.0) * (bb / (1.0 - bb));
                else v = 0.0;
                if (v < 1e-300) v = 0.0;
            }
        }
        for (std::int64_t s = std::max<std::int64_t>(sa + 1, 0); s < std::min(sb, N_ + 1); ++s) {
            const double sd = static_cast<double>(s);
            top[static_cast<std::size_t>(s)] =
                s == 0 || s == N_
                    ? 1.0
                    : std::min(1.0, std::sqrt(Nd / (2.0 * M_PI * sd * (Nd - sd))) * std::exp(1.0 / (12.0 * Nd)));
        }
        double acc = 0.0;
        for (std::int64_t s = 0; s <= N_; ++s) {
            const double t = top[static_cast<std::size_t>(s)];
            if (t == 0.0) continue;
            double w, d, g;
            if (low) {
                w = (Nd - s) * (Nd - s - 1.0) * (Nd - s - 2.0);
                d = b(s + 3) - 3.0 * b(s + 2) + 3.0 * b(s + 1) - b(s);
                g = b(s + 3) + 3.0 * b(s + 2) + 3.0 * b(s + 1) + b(s);
            } else {
                const double sd = static_cast<double>(s);
                w = sd * (sd - 1.0) * (sd - 2.0);
                d = b(s) - 3.0 * b(s - 1) + 3.0 * b(s - 2) - b(s - 3);
                g = b(s) + 3.0 * b(s - 1) + 3.0 * b(s - 2) + b(s - 3);
            }
            if (w <= 0.0) continue;
            acc += w * (std::fabs(d) + 16.0 * kU * g) * t;
        }
        evals_ += 1.0;
        return acc * (low ? flow : fhigh) * 1.01;
    }
};

}  // namespace sd::htest::nuisance
