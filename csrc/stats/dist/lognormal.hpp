// The log-normal distribution: log X ~ N(mu, sigma^2) (mu and sigma of the
// underlying normal; T18). z = (log x - mu) / sigma with log x a double-double
// (stats/numeric.hpp): the tails multiply log x's rounding by |z| |log x| / sigma.
// The inverses exponentiate mu + sigma z in double-double: exp(E) multiplies E's
// rounding by |E| (up to 709).
#pragma once

#include "special/ops/distr.hpp"
#include "stats/chunk.hpp"
#include "stats/chunk_dd.hpp"
#include "stats/dist/normal.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

struct LogNormal {
    static constexpr char name[] = "lognormal";
    static constexpr std::array<const char*, 2> params{"mu", "sigma"};

    static bool valid(double mu, double sigma) { return std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma); }

    // z for finite x > 0.
    static num::DD z_of(double x, double mu, double sigma) { return num::div(num::log_dd(x) + (-mu), sigma); }

    // -(z^2 / 2 + log(sigma sqrt(2 pi)) + log x)
    static num::DD log_density(double x, double mu, double sigma) {
        const num::DD lx = num::log_dd(x);  // once, for z and the 1/x
        const num::DD z = num::div(lx + (-mu), sigma);
        if (std::fabs(z.hi) > normal_detail::kHugeZ) {  // as the normal's: z^2 in double (inf: -inf)
            return {-0.5 * z.hi * z.hi - std::log(sigma) - num::kLogSqrt2Pi.hi - lx.hi, 0.0};
        }
        return -(normal_detail::neg_log_density(z, sigma) + lx);
    }
    static double pdf(double x, double mu, double sigma) {
        if (x <= 0.0 || std::isinf(x)) return 0.0;
        return num::exp_dd(log_density(x, mu, sigma));
    }
    static double logpdf(double x, double mu, double sigma) {
        if (x <= 0.0 || std::isinf(x)) return -kInf;
        const num::DD l = log_density(x, mu, sigma);
        return l.hi + l.lo;
    }
    static double tail(double x, double mu, double sigma, bool upper) {
        if (x <= 0.0) return upper ? 1.0 : 0.0;
        if (std::isinf(x)) return upper ? 0.0 : 1.0;
        return normal_detail::tail_z(z_of(x, mu, sigma), upper);
    }
    static double cdf(double x, double mu, double sigma) { return tail(x, mu, sigma, false); }
    static double ccdf(double x, double mu, double sigma) { return tail(x, mu, sigma, true); }
    static double log_tail(double x, double mu, double sigma, bool upper) {
        if (x <= 0.0) return upper ? 0.0 : -kInf;
        if (std::isinf(x)) return upper ? -kInf : 0.0;
        return normal_detail::log_tail_z(z_of(x, mu, sigma), upper);
    }
    static double logcdf(double x, double mu, double sigma) { return log_tail(x, mu, sigma, false); }
    static double logccdf(double x, double mu, double sigma) { return log_tail(x, mu, sigma, true); }

    // ---- chunk kernels --------------------------------------------------------------------
    //
    // z = (log x - mu) / sigma and log x for the chunk in the scalar definition's double-
    // doubles, ckd::log_dd for dd::log: log x is off by kLq |log x| units and z by
    // dz = kLq |log x| / sigma (absolute). NaN where x is not in [2^-999, 2^999].
    static void z_chunk(const Chunk<2>& c, double* zh, double* zl, double* dz, double* lh, double* ll) {
        for (int i = 0; i < c.n; ++i) zl[i] = 0.0;
        ckd::log_dd(c.x, zl, lh, ll, c.n);
        for (int i = 0; i < c.n; ++i) {
            const num::DD z = num::div(num::DD{lh[i], ll[i]} + (-c.p[0][i]), c.p[1][i]);
            const bool ok = std::isfinite(z.hi);
            zh[i] = ok ? z.hi : kNaN, zl[i] = z.lo;
            dz[i] = ckd::kLq * std::fabs(lh[i]) / c.p[1][i];
        }
    }
    // The normal's kernels at that z (normal_detail::tail_z_chunk, log_tail_z_chunk). dz reaches
    // the tail times phi / Phi <= |z| + 0.8 (Mills' ratio; 0.8 at most on the large side),
    // taken while that is below 2 units: with the kernels' own 1 and 11, inside the budget
    // (for the log: phi / Phi dz is absolute, and relative to |log Phi| >= log 2 on the small
    // side, to Phi(-|z|) on the large one, at most 1.5 times as much).
    static void tail_chunk(const Chunk<2>& c, double* out, bool upper, bool log) {
        double zh[kChunkSize], zl[kChunkSize], dz[kChunkSize], lh[kChunkSize], ll[kChunkSize];
        bool fast[kChunkSize];
        z_chunk(c, zh, zl, dz, lh, ll);
        if (upper) {
            for (int i = 0; i < c.n; ++i) zh[i] = -zh[i], zl[i] = -zl[i];
        }
        if (log) {
            normal_detail::log_tail_z_chunk(zh, zl, out, fast, c.n);
        } else {
            normal_detail::tail_z_chunk(zh, zl, out, c.n);
            for (int i = 0; i < c.n; ++i) fast[i] = out[i] >= 0x1p-1000;
        }
        for (int i = 0; i < c.n; ++i) fast[i] = fast[i] && (std::fabs(zh[i]) + 0.8) * dz[i] <= 2.0;  // false for NaN
        ck::fallback(c.n, fast, out, [&](int i) {
            return log ? log_tail(c.x[i], c.p[0][i], c.p[1][i], upper) : tail(c.x[i], c.p[0][i], c.p[1][i], upper);
        });
    }
    static void cdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, false, false); }
    static void ccdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, true, false); }
    static void logcdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, false, true); }
    static void logccdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, true, true); }
    // -(z^2 / 2 + log(sigma sqrt(2 pi)) + log x) in the scalar definition's double-doubles: its
    // absolute error is |z| dz + kLq |log x| and the last bit of the sum (1); logpdf against
    // the mixed bound (T20), pdf with exp's kVfExp + 1 more.
    static void density_chunk(const Chunk<2>& c, double* out, bool log) {
        double zh[kChunkSize], zl[kChunkSize], dz[kChunkSize], lh[kChunkSize], ll[kChunkSize], kh[kChunkSize],
            kl[kChunkSize];
        bool fast[kChunkSize];
        z_chunk(c, zh, zl, dz, lh, ll);
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[1][i]) + num::kLogSqrt2Pi; });
        for (int i = 0; i < c.n; ++i) {
            const num::DD z{zh[i], zl[i]};
            const num::DD e = -(((z * z) * 0.5 + num::DD{kh[i], kl[i]}) + num::DD{lh[i], ll[i]});
            const double abs = std::fabs(z.hi) * dz[i] + ckd::kLq * std::fabs(lh[i]) + 1.0;
            if (log) {
                out[i] = e.hi + e.lo;
                fast[i] = out[i] == out[i] && abs <= kFastBudget * std::max(1.0, std::fabs(out[i]));  // false for NaN
            } else {
                zh[i] = e.hi, zl[i] = e.lo;
                dz[i] = abs + ck::kVfExp + 1.0;
            }
        }
        if (!log) {
            ckd::exp_dd(out, zh, zl, c.n);
            for (int i = 0; i < c.n; ++i) fast[i] = dz[i] <= kFastBudget && out[i] >= 0x1p-1000;
        }
        ck::fallback(c.n, fast, out, [&](int i) {
            return log ? logpdf(c.x[i], c.p[0][i], c.p[1][i]) : pdf(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void pdf_chunk(const Chunk<2>& c, double* out) { density_chunk(c, out, false); }
    static void logpdf_chunk(const Chunk<2>& c, double* out) { density_chunk(c, out, true); }
    // The scalar definition's first branch for the chunk (special's ndtri kernel, sigma |z| <=
    // 4), exp by vForce: within kVfExp + 3 of it.
    static void quantile_chunk(const Chunk<2>& c, double* out, bool upper) {
        double z[kChunkSize], el[kChunkSize];
        bool fast[kChunkSize];
        special::Ndtri::chunk(c.x, z, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i], mu = c.p[0][i], sigma = c.p[1][i];
            const double zi = upper ? -z[i] : z[i];
            fast[i] = p > 0.0 && p < 1.0 && sigma * std::fabs(zi) <= 4.0;
            const num::DD e = num::dd::two_prod(sigma, zi) + sigma * 0.0 + mu;
            z[i] = e.hi, el[i] = e.lo;
        }
        ckd::exp_dd(out, z, el, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = fast[i] && out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;
        ck::fallback(c.n, fast, out, [&](int i) {
            return upper ? iccdf(c.x[i], c.p[0][i], c.p[1][i]) : icdf(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void icdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, false); }
    static void iccdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, true); }

    // exp(mu + sigma z), z and the exponent double-doubles (normal_detail::quantile_z).
    // e^(2 g) for a finite double-double g (2 g itself may overflow).
    static double exp_twice(num::DD g) {
        if (g.hi > 355.0) return kInf;
        if (g.hi < -373.0) return 0.0;
        return num::exp_dd(g * 2.0);
    }
    // sigma^2 as a double-double; +inf where it overflows (the product's low part is NaN there).
    static num::DD sq(double sigma) {
        const num::DD s2 = num::dd::two_prod(sigma, sigma);
        return std::isinf(s2.hi) ? num::DD{kInf, 0.0} : s2;
    }
    // log(e^s2 - 1) for a finite s2 = sigma^2 > 0.
    static num::DD log_expm1(num::DD s2, double sigma) {
        if (s2.hi > 40.0) return s2;
        if (s2.hi < 0x1p-60) return num::log_dd(sigma) * 2.0;
        return num::log_dd(expm1_dd(s2));
    }
    static double from_z(num::DD z, double mu, double sigma) {
        if (std::isinf(z.hi)) return z.hi > 0.0 ? kInf : 0.0;
        if (!(std::fabs(sigma * z.hi) <= 1e300)) {
            // sigma z or its sum with mu may overflow: half the exponent (a product beyond
            // the range is beyond 2 |mu| too, so its sign decides)
            const double h = 0.5 * sigma;
            if (std::isinf(h * z.hi)) return z.hi > 0.0 ? kInf : 0.0;
            return exp_twice(num::dd::two_prod(h, z.hi) + h * z.lo + 0.5 * mu);
        }
        return num::exp_dd(num::dd::two_prod(sigma, z.hi) + sigma * z.lo + mu);
    }
    // Without the Newton step while sigma |z| <= 4: ndtri's few ULP in z reach x multiplied
    // by sigma |z| (exp of mu + sigma z), at most ~22 ULP then.
    static double icdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p > 0.0 && p < 1.0) {
            const double z = normal_detail::ndtri(p);
            if (sigma * std::fabs(z) <= 4.0) return from_z(num::DD{z, 0.0}, mu, sigma);
        }
        return from_z(normal_detail::quantile_z(p, false), mu, sigma);
    }
    static double iccdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p > 0.0 && p < 1.0) {
            const double z = -normal_detail::ndtri(p);
            if (sigma * std::fabs(z) <= 4.0) return from_z(num::DD{z, 0.0}, mu, sigma);
        }
        return from_z(normal_detail::quantile_z(p, true), mu, sigma);
    }
    static double ilogcdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        return from_z(normal_detail::quantile_z_log(lp, false), mu, sigma);
    }
    static double ilogccdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        return from_z(normal_detail::quantile_z_log(lp, true), mu, sigma);
    }

    // t = e^(sigma^2) - 1 with sigma^2 = s2.hi + s2.lo: expm1(hi) + e^hi lo.
    static double expm1_dd(num::DD s2) {
        const double e = std::exp(s2.hi);
        return std::isinf(e) ? e : std::expm1(s2.hi) + e * s2.lo;
    }

    // A sigma^2 beyond the range is beyond 2 |mu| as well: e^(mu + k sigma^2) is then inf
    // (0 for the mode's k = -1) whatever mu. Exponents that may overflow as a sum are
    // formed by halves (exp_twice).
    static double mean(double mu, double sigma) {  // exp(mu + sigma^2 / 2)
        const num::DD s2 = sq(sigma);
        if (std::isinf(s2.hi)) return kInf;
        return exp_twice(s2 * 0.25 + 0.5 * mu);
    }
    // (e^s2 - 1) e^(s2 + 2 mu), s2 = sigma^2; past s2 = 700 the first factor is e^s2.
    static double variance(double mu, double sigma) {
        const num::DD s2 = sq(sigma);
        if (std::isinf(s2.hi)) return kInf;
        if (s2.hi > 700.0) {  // e^(2 s2 + 2 mu) = e^(4 g)
            const num::DD g = s2 * 0.5 + 0.5 * mu;
            return g.hi > 178.0 ? kInf : (g.hi < -187.0 ? 0.0 : num::exp_dd(g * 4.0));
        }
        const double a = expm1_dd(s2);
        const num::DD h = s2 * 0.5 + mu;
        if (a >= 0x1p-900 && std::fabs(h.hi) < 350.0) return a * num::exp_dd(h * 2.0);
        return exp_twice(h + log_expm1(s2, sigma) * 0.5);  // a factor leaves the range: through the logs
    }
    static double standard_deviation(double mu, double sigma) {  // sqrt(e^s2 - 1) e^(s2 / 2 + mu)
        const num::DD s2 = sq(sigma);
        if (std::isinf(s2.hi)) return kInf;
        if (s2.hi > 700.0) return exp_twice(s2 * 0.5 + 0.5 * mu);
        const double a = expm1_dd(s2);
        const num::DD h = s2 * 0.5 + mu;
        if (a >= 0x1p-900 && std::fabs(h.hi) < 700.0) return std::sqrt(a) * num::exp_dd(h);
        return num::exp_dd(h + log_expm1(s2, sigma) * 0.5);
    }
    static double skewness(double, double sigma) {  // (t + 3) sqrt(t), t = e^s2 - 1
        const num::DD s2 = sq(sigma);
        if (std::isinf(s2.hi)) return kInf;
        if (s2.hi < 0x1p-60) return 3.0 * sigma;  // sqrt(t) = sigma (s2 may underflow)
        const double t = expm1_dd(s2);
        return (t + 3.0) * std::sqrt(t);
    }
    static double kurtosis(double, double sigma) {
        const num::DD s2 = sq(sigma);
        if (std::isinf(s2.hi)) return kInf;
        const double t = expm1_dd(s2);
        return t * (16.0 + t * (15.0 + t * (6.0 + t)));
    }
    static double median(double mu, double) { return std::exp(mu); }
    static double mode(double mu, double sigma) {  // exp(mu - sigma^2)
        const num::DD s2 = sq(sigma);
        if (std::isinf(s2.hi)) return 0.0;
        return exp_twice(s2 * -0.5 + 0.5 * mu);
    }
    static double entropy(double mu, double sigma) {  // mu + 1/2 + log(sigma sqrt(2 pi))
        const num::DD h = num::kNormalEntropy + num::log_dd(sigma) + mu;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
