// The generalized linear models' numerical core (docs/design/regression.md §3.4).
//
// One regression per batch element, through linalg's run_batch (one task per regression,
// the same bits for any thread count):
//   1. Checks (inf is an error, NaN gives an all-NaN result, y must lie in the family's
//      range), the rows of the fit (prior weight var_weights x trials x freq_weights > 0),
//      and R's starting values (family$initialize: binomial (m y + 1/2)/(m + 1), poisson
//      y + 1/10, gaussian and gamma y), eta_0 = g(mu_0).
//   2. IRLS: each iteration is one weighted least squares fit of the working response
//      z = eta - offset + (y - mu)/mu' with weights f m mu'^2 / V(mu). Before the deviance's
//      convergence it is solved by the normal equations (Cholesky, G40) over the columns the
//      last QR kept (all before any); an iteration whose Cholesky fails, is too
//      ill-conditioned or meets G21's aliasing rule goes through the least squares core (regression/lsq.hpp: the same QR and
//      aliased columns as ols) without refinement. A step that leaves the valid range of mu or eta, gives a non-finite
//      deviance or raises the deviance ((D - D_old)/(|D| + 0.1) >= tol, glm2's rule) is
//      halved towards the previous coefficients (towards eta_0 before there are any).
//      Converged when |D - D_old|/(|D| + 0.1) < tol (R's glm.fit); then the iterations go
//      on while the coefficients still move by more than rounding (the step after the next,
//      predicted from the last two steps' ratio, above 2^-50 |beta|, and the steps still
//      contracting). These polishing iterations refine (as the final solve) and, for a
//      non-canonical link, take Newton's steps (the observed information: its weights in the
//      least squares fit when they are all positive, else a k x k correction of Fisher's
//      step), so that they converge quadratically.
//   3. The final solve: one more weighted fit at the last iterate's weights through the
//      full core (refinement on the augmented system, G15; the covariance (R^T R)^-1 =
//      (X^T W X)^-1, or a sandwich whose bread is the observed information).
//   4. Separation (binomial, poisson with the log link): the last Fisher step d at beta_t
//      certifies that the MLE exists when every boundary row (y = 0 or 1; poisson y = 0)
//      has |x_i' d| < |y_i - mu_i| / (2 mu'_i) (the score weights minus the step's stay of
//      the boundary's sign: a strictly positive combination of the rows that sums to zero,
//      Stiemke's alternative). Otherwise the failing rows S are tested for a direction of
//      recession: d projected onto the null space of the other rows' X (least squares)
//      must have s_i x_i' b > sqrt(eps) sum_j |x_ij b_j| on S. A certified S is fitted at
//      the boundary (mu = y) and the rest again by IRLS (the limit of the likelihood's
//      maximization), and the test repeats on the rest.
//   5. Deviance, Pearson chi-square and log-likelihood per family without cancellation
//      (y log(y/mu) - (y - mu) as mu ((1 + r) log1p(r) - r), r = (y - mu)/mu, with a series
//      near r = 0), summed in double-double over fixed chunks of rows.
#include "regression/glm.hpp"

#include "regression/lsq.hpp"

#include "bind/errors.hpp"
#include "linalg/ops/common.hpp"
#include "parallel/pool.hpp"
#include "special/dd.hpp"
#include "special/ops/erf.hpp"
#include "special/ops/gamma.hpp"
#include "special/ops/logistic.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sd::regression {

namespace nb = nanobind;
using namespace lsq;
using linalg::Batch;
using linalg::Failure;
using linalg::Input;
using linalg::Out;
using DD = special::dd::DD;
using special::dd::operator+;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kPolishSteps = 3;                  // refinement steps of the polishing iterations
// G44's test: Fisher scoring converges linearly near the maximum, its steps shrinking by the
// rate rho = ||I - I_E^-1 I_O|| each; a step not below half the one before means rho >= 1/2,
// i.e. more than 50 iterations to rounding. Three such steps in a row: a single large ratio
// is common far from the maximum (the direction turns), three in a row are not in fits that
// converge (measured rates 0.04-0.15; the slowest well-conditioned fits of the accuracy tables
// never trigger it).
constexpr double kSlowRatio = 0.5;
constexpr int kSlowRuns = 3;
constexpr double kSqrtEps = 0x1p-26;               // sqrt(2^-52): the direction test's margin
constexpr double kInvSqrt2Pi = 0.39894228040143267794;

// The Python layer maps the names (stardust/regression/_glm.py).
enum Family : int { fGaussian = 0, fBinomial, fPoisson, fGamma };
enum Link : int { lIdentity = 0, lLog, lLogit, lProbit, lCloglog, lSqrt, lInverse };
// scale: the family's default, Pearson chi2 / df, deviance / df, a fixed value
enum ScaleMode : int { sDefault = 0, sX2, sDev, sFixed };

// The per-regression scalars, in this order (stardust/regression/_glm.py reads them).
enum GStat : int {
    gNobs = 0, gRank, gDev, gNullDev, gPearson, gScale, gLlf, gLlnull, gNIter, gConverged, gSeparation, gIsNaN,
    gNGroups, gLags, gNStat
};

// Failure codes of one regression.
constexpr int eInfY = 1, eInfX = 2, eInfVar = 3, eNegVar = 4, eInfFreq = 5, eNegFreq = 6, eTrials = 7,
              eInfOffset = 8, eBadY = 9, eNoStart = 10;

inline double rd(const std::byte* p) {
    double v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
inline std::int64_t rd_i64(const std::byte* p) {
    std::int64_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
    e = std::fma(a, b, -p);
}

// Row passes: chunks of rows fixed by n alone, partial sums added in order (the same bits
// for any thread count), on the pool from kRowParallelMin rows.
constexpr lint kRowChunk = 4096;
constexpr lint kRowParallelMin = 1 << 16;

template <class F>
void for_row_chunks(lint n, F&& body) {
    const lint nc = (n + kRowChunk - 1) / kRowChunk;
    auto one = [&](lint c) { body(c, c * kRowChunk, std::min(n, (c + 1) * kRowChunk)); };
    if (nc >= 2 && n >= kRowParallelMin) {
        parallel::parallel_for(nc, [&](std::int64_t c) { one(static_cast<lint>(c)); });
    } else {
        for (lint c = 0; c < nc; ++c) one(c);
    }
}
inline lint row_chunks(lint n) { return std::max<lint>(1, (n + kRowChunk - 1) / kRowChunk); }

// ---- deviance pieces without cancellation

// r - log1p(r) (>= 0) for r > -1, x = 1 + r given exactly. On [-1/2, 1] by the atanh series
// in s = r/(2 + r): log1p(r) = 2 atanh(s) and r - 2s = r s, so r - log1p(r) = r s - 2 s^3
// (1/3 + s^2/5 + ...), no cancellation (|s| <= 1/3: 17 terms leave < 2^-56).
inline double mu_fn(double r, double x) {
    if (r >= -0.5 && r <= 1.0) {
        const double s = r / (2.0 + r), s2 = s * s;
        double t = 0.0;
        for (int j = 17; j >= 0; --j) t = 1.0 / (2 * j + 3) + s2 * t;
        return r * s - 2.0 * s * s2 * t;
    }
    return r > 1.0 ? r - std::log1p(r) : r - std::log(x);
}

// (1 + r) log1p(r) - r = x log x - (x - 1) (>= 0), x = 1 + r = y/mu: r^2 - (1 + r) mu_fn(r)
// on [-1/2, 1] (cancellation at most 4 times), x log x - r outside.
inline double phi_fn(double r, double x) {
    if (r >= -0.5 && r <= 1.0) return r * r - (1.0 + r) * mu_fn(r, x);
    return x * std::log(x) - r;
}

// log mu and log(1 - mu) of the binomial from the two accurate complements.
inline double log_mu(double mu, double muc) { return mu < 0.5 ? std::log(mu) : std::log1p(-muc); }
inline double log_muc(double mu, double muc) { return muc < 0.5 ? std::log(muc) : std::log1p(-mu); }

// y - mu, exact where y is at the binomial's boundary
inline double y_minus_mu(int fam, double y, double mu, double muc) {
    if (fam == fBinomial) {
        if (y == 1.0) return muc;
        if (y == 0.0) return -mu;
    }
    return y - mu;
}

inline double variance(int fam, double mu, double muc) {
    switch (fam) {
        case fBinomial: return mu * muc;
        case fPoisson: return mu;
        case fGamma: return mu * mu;
        default: return 1.0;
    }
}

// The unit deviance (no weight).
inline double unit_dev(int fam, double y, double mu, double muc) {
    switch (fam) {
        case fGaussian: {
            const double d = y - mu;
            return d * d;
        }
        case fPoisson:
            if (y == 0.0) return 2.0 * mu;
            return 2.0 * mu * phi_fn((y - mu) / mu, y / mu);
        case fGamma:
            return 2.0 * mu_fn((y - mu) / mu, y / mu);
        default:  // binomial
            if (y == 0.0) return -2.0 * log_muc(mu, muc);
            if (y == 1.0) return -2.0 * log_mu(mu, muc);
            return 2.0 * (mu * phi_fn((y - mu) / mu, y / mu) + muc * phi_fn((mu - y) / muc, (1.0 - y) / muc));
    }
}

// One regression's model: per-row arrays are contiguous (n).
struct Model {
    int fam, link;
    lint n, p, k;
    bool intercept;
    double rtol;
    const double* y;
    const double* v;    // var_weights
    const double* f;    // freq_weights
    const double* tr;   // trials
    const double* m;    // v * trials: the unit deviance's weight
    const double* pw;   // f * m: the fit's prior weight (0: out of the fit)
    const double* o;    // offset (+ log exposure)
    const std::byte* X;
    std::int64_t xrs, xcs;
};

inline bool canonical(const Model& M) {
    return (M.fam == fGaussian && M.link == lIdentity) || (M.fam == fBinomial && M.link == lLogit) ||
           (M.fam == fPoisson && M.link == lLog) || (M.fam == fGamma && M.link == lInverse);
}

// mu, 1 - mu (binomial) and mu' = dmu/deta for rows [i0, i1), through sd.special's kernels.
void link_eval(const Model& M, const double* eta, double* mu, double* muc, double* d1, lint i0, lint i1) {
    alignas(16) double neg[special::kChunk];
    for (lint b = i0; b < i1; b += special::kChunk) {
        const int len = static_cast<int>(std::min<lint>(special::kChunk, i1 - b));
        const double* e = eta + b;
        double *u = mu + b, *uc = muc + b, *g = d1 + b;
        switch (M.link) {
            case lIdentity:
                for (int i = 0; i < len; ++i) u[i] = e[i], uc[i] = 1.0 - e[i], g[i] = 1.0;
                break;
            case lLog:
                for (int i = 0; i < len; ++i) u[i] = std::exp(e[i]), uc[i] = -std::expm1(e[i]), g[i] = u[i];
                break;
            case lLogit:
                for (int i = 0; i < len; ++i) neg[i] = -e[i];
                special::Expit::chunk(e, u, len);
                special::Expit::chunk(neg, uc, len);
                for (int i = 0; i < len; ++i) g[i] = u[i] * uc[i];
                break;
            case lProbit:
                for (int i = 0; i < len; ++i) neg[i] = -e[i];
                special::Ndtr::chunk(e, u, len);
                special::Ndtr::chunk(neg, uc, len);
                for (int i = 0; i < len; ++i) {
                    // phi(eta) with eta^2 split exactly (h + l): e^(-h/2) (1 - l/2)
                    const double h = e[i] * e[i], l = std::fma(e[i], e[i], -h);
                    g[i] = std::exp(-0.5 * h) * (1.0 - 0.5 * l) * kInvSqrt2Pi;
                }
                break;
            case lCloglog:
                for (int i = 0; i < len; ++i) {
                    const double t = std::exp(e[i]);
                    uc[i] = std::exp(-t);
                    u[i] = -std::expm1(-t);
                    g[i] = std::isinf(t) ? 0.0 : t * uc[i];
                }
                break;
            case lSqrt:
                for (int i = 0; i < len; ++i) u[i] = e[i] * e[i], uc[i] = 1.0 - u[i], g[i] = 2.0 * e[i];
                break;
            default:  // inverse
                for (int i = 0; i < len; ++i) u[i] = 1.0 / e[i], uc[i] = 1.0 - u[i], g[i] = -u[i] * u[i];
                break;
        }
    }
}

// g(mu) for the starting values, rows [i0, i1)
void link_fun(const Model& M, const double* mu, double* eta, lint i0, lint i1) {
    for (lint b = i0; b < i1; b += special::kChunk) {
        const int len = static_cast<int>(std::min<lint>(special::kChunk, i1 - b));
        const double* u = mu + b;
        double* e = eta + b;
        switch (M.link) {
            case lIdentity: std::copy(u, u + len, e); break;
            case lLog: for (int i = 0; i < len; ++i) e[i] = std::log(u[i]); break;
            case lLogit: special::Logit::chunk(u, e, len); break;
            case lProbit: special::Ndtri::chunk(u, e, len); break;
            case lCloglog: for (int i = 0; i < len; ++i) e[i] = std::log(-std::log1p(-u[i])); break;
            case lSqrt: for (int i = 0; i < len; ++i) e[i] = std::sqrt(u[i]); break;
            default: for (int i = 0; i < len; ++i) e[i] = 1.0 / u[i]; break;
        }
    }
}

// R's validmu and valideta for a row of the fit
inline bool row_valid(const Model& M, double eta, double mu, double muc) {
    if (!std::isfinite(mu) || !std::isfinite(eta)) return false;
    switch (M.fam) {
        case fBinomial: return mu > 0.0 && muc > 0.0;
        case fPoisson: return mu > 0.0 && (M.link == lIdentity || M.link == lSqrt ? eta > 0.0 : true);
        case fGamma: return mu > 0.0;
        default: return true;
    }
}

// The IRLS state at one iterate.
struct State {
    double *eta, *mu, *muc, *d1;
    DD dev{0.0, 0.0};
    bool valid = true;
};

// mu, 1 - mu, mu' of every row from eta; the fit rows' validity and deviance (Dot2-free:
// one unit deviance per row, summed in double-double over fixed chunks)
void evaluate(const Model& M, State& s, double* part) {
    const lint n = M.n;
    std::vector<std::uint8_t> bad(static_cast<std::size_t>(row_chunks(n)), 0);
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        link_eval(M, s.eta, s.mu, s.muc, s.d1, i0, i1);
        DD acc{0.0, 0.0};
        std::uint8_t b = 0;
        for (lint i = i0; i < i1; ++i) {
            if (!(M.pw[i] > 0.0)) continue;
            if (!row_valid(M, s.eta[i], s.mu[i], s.muc[i])) {
                b = 1;
                continue;
            }
            acc = acc + DD{M.pw[i] * unit_dev(M.fam, M.y[i], s.mu[i], s.muc[i]), 0.0};
        }
        part[2 * c] = acc.hi;
        part[2 * c + 1] = acc.lo;
        bad[static_cast<std::size_t>(c)] = b;
    });
    DD dev{0.0, 0.0};
    bool valid = true;
    for (lint c = 0; c < row_chunks(n); ++c) {
        dev = dev + DD{part[2 * c], part[2 * c + 1]};
        if (bad[static_cast<std::size_t>(c)]) valid = false;
    }
    s.dev = dev;
    s.valid = valid && std::isfinite(dev.hi);
}

// The working weights, residuals rho = (y - mu)/mu' and response z = eta - offset + rho of
// a state (identity link: z = y - offset exactly). Rows out of the fit, or whose mu' or
// weight is not positive and finite (R's `good`), get weight 0.
void working(const Model& M, const State& s, double* z, double* w, double* rho) {
    for_row_chunks(M.n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            const double g = s.d1[i];
            const double V = variance(M.fam, s.mu[i], s.muc[i]);
            double wi = M.pw[i] > 0.0 ? M.pw[i] * (g * g / V) : 0.0;
            if (!(wi > 0.0 && std::isfinite(wi) && g != 0.0)) wi = 0.0;
            const double r = y_minus_mu(M.fam, M.y[i], s.mu[i], s.muc[i]) / g;
            w[i] = wi;
            rho[i] = r;
            if (wi == 0.0) z[i] = 0.0;
            else if (M.link == lIdentity) z[i] = M.y[i] - M.o[i];
            else z[i] = (s.eta[i] - M.o[i]) + r;
        }
    });
}

// eta = X beta + offset for every row (Dot2 over the columns with a coefficient; NaN
// marks an aliased column, which takes no part)
void xbeta_all(const Model& M, const double* beta, double* eta) {
    for_row_chunks(M.n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            double P = 0.0, S = 0.0;
            for (lint c = 0; c < M.k; ++c) {
                const double b = beta[c];
                if (std::isnan(b)) continue;
                const double x = (M.intercept && c == 0) ? 1.0 : rd(M.X + i * M.xrs + (c - (M.intercept ? 1 : 0)) * M.xcs);
                double hp, ep, q;
                two_prod(x, b, hp, ep);
                two_sum(P, hp, P, q);
                S += q + ep;
            }
            eta[i] = (P + S) + M.o[i];
        }
    });
}

// One inner weighted least squares fit (no refinement unless steps > 0): beta (NaN for
// aliased columns), kept, eta = X beta + offset for every row. Returns the rank.
lint solve_lean(const Model& M, const double* z, const double* w, double* beta, std::uint8_t* kept, double* eta,
                int steps, Scratch& sc, double* R = nullptr, const double* hdiff = nullptr,
                const double* newton_b0 = nullptr, const double* newton_eta0 = nullptr) {
    Problem pb{M.n, M.p, M.k, M.intercept, true, false, false, M.rtol, kNonrobust, steps, -1, 0, false};
    pb.lean = true;
    Data d{reinterpret_cast<const std::byte*>(z), 8, M.X, M.xrs, M.xcs, reinterpret_cast<const std::byte*>(w), 8,
           nullptr, 0, nullptr, 0};
    double st[kNStat];
    d.hdiff = hdiff;
    d.newton_b0 = newton_b0;
    d.newton_eta0 = newton_eta0;
    Dest o{beta, nullptr, R, eta, nullptr, kept, st, nullptr};
    fit_one(pb, d, o, sc);
    for (lint i = 0; i < M.n; ++i) eta[i] += M.o[i];
    return static_cast<lint>(st[kRank]);
}

// The buffers of one IRLS run.
struct Work {
    lint n, k;
    std::unique_ptr<double[]> mem;
    std::unique_ptr<std::uint8_t[]> kmem;
    State cur, nxt;
    double *z, *w, *rho, *part;
    double *beta, *beta_new, *R;
    std::uint8_t *kept, *kept_new;
    bool have_beta = false;
    bool have_kept = false;  // kept holds the last QR's columns
    Work(lint n_, lint k_) : n(n_), k(k_) {
        const lint nn = std::max<lint>(n, 1), kk = std::max<lint>(k, 1);
        mem.reset(new double[static_cast<std::size_t>(11 * nn + 2 * row_chunks(n) + 2 * kk + kk * kk)]);
        double* q = mem.get();
        auto take = [&](lint c) {
            double* r = q;
            q += c;
            return r;
        };
        cur.eta = take(nn), cur.mu = take(nn), cur.muc = take(nn), cur.d1 = take(nn);
        nxt.eta = take(nn), nxt.mu = take(nn), nxt.muc = take(nn), nxt.d1 = take(nn);
        z = take(nn), w = take(nn), rho = take(nn), part = take(2 * row_chunks(n));
        beta = take(kk), beta_new = take(kk), R = take(kk * kk);
        kmem.reset(new std::uint8_t[static_cast<std::size_t>(2 * kk)]);
        kept = kmem.get(), kept_new = kmem.get() + kk;
    }
};

// The observed information's weight is w_i - c_i with c_i = pw (y - mu) (mu''/V - mu'^2 V'/V^2)
// (0 for a canonical link, where the two informations agree). c for every row whose working
// weight is positive (0 elsewhere).
void obs_correction(const Model& M, const State& s, const double* w, double* c) {
    for_row_chunks(M.n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            if (!(w[i] > 0.0)) {
                c[i] = 0.0;
                continue;
            }
            const double mu = s.mu[i], muc = s.muc[i], g = s.d1[i], e = s.eta[i];
            double g2 = 0.0;  // mu''
            switch (M.link) {
                case lLog: g2 = mu; break;
                case lLogit: g2 = g * (muc - mu); break;
                case lProbit: g2 = -e * g; break;
                case lCloglog: g2 = g * (1.0 - std::exp(e)); break;
                case lSqrt: g2 = 2.0; break;
                case lInverse: g2 = 2.0 * mu * mu * mu; break;
                default: g2 = 0.0; break;
            }
            const double V = variance(M.fam, mu, muc);
            double Vp = 0.0;  // V'(mu)
            switch (M.fam) {
                case fBinomial: Vp = muc - mu; break;
                case fPoisson: Vp = 1.0; break;
                case fGamma: Vp = 2.0 * mu; break;
                default: Vp = 0.0; break;
            }
            c[i] = M.pw[i] * y_minus_mu(M.fam, M.y[i], mu, muc) * (g2 / V - g * g * Vp / (V * V));
        }
    });
}



// An inner iteration by the normal equations (G40): X_K^T W X_K b = X_K^T W z over the
// columns K kept by the last QR (all columns before any QR), Cholesky (potrf) and two
// triangular solves. Aliasing stays G21's: a column the rule would drop makes this return
// false and the QR decides. The Gram matrix and right-hand side are summed over fixed chunks of
// rows (each chunk's dsyrk/dgemv into its own slot, the slots added in order: the same bits
// for any thread count). Returns false, and leaves the QR path to the caller, when the
// Cholesky factorization fails or its condition estimate says the normal equations would
// lose too much: kappa(X^T W X) u > kCholMaxErr (kappa from LAPACK's trcon on the factor,
// squared). An inner step only has to be accurate enough not to slow the IRLS: a relative
// error e in the step makes the iteration contract by about e per step before the quadratic
// regime, and the deviance's criterion (tol 1e-8) needs the coefficients to about sqrt(tol)
// = 1e-4; with e <= 1e-4 the iteration counts stay those of the QR path. The polishing
// iterations and the final solve (QR and refinement) set the final accuracy.
constexpr double kCholMaxErr = 1e-4;
// Rows x columns below which the inner iterations stay QR: there the QR costs no more than
// the Gram matrix's setup (measured, docs/design/regression.md §3.4: a logistic (50, 3) fit
// 18.7 us by QR, 19.9 us by Cholesky; (200, 3) the same; (500, 5) 13% faster by Cholesky)
constexpr double kCholMinWork = 512;

bool solve_chol(const Model& M, const double* z, const double* w, const std::uint8_t* keep, double* beta,
                std::uint8_t* kept, double* eta) {
    const lint n = M.n, k = M.k;
    std::vector<lint> K;
    for (lint j = 0; j < k; ++j)
        if (keep[j]) K.push_back(j);
    const lint kk = static_cast<lint>(K.size());
    if (kk == 0) return false;
    const lint nc = row_chunks(n);
    const std::size_t slot = static_cast<std::size_t>(kk * kk + kk);
    std::vector<double> part(static_cast<std::size_t>(nc) * slot, 0.0);
    std::vector<std::uint8_t> any(static_cast<std::size_t>(nc), 0);
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        std::vector<lint> rows;
        rows.reserve(static_cast<std::size_t>(i1 - i0));
        for (lint i = i0; i < i1; ++i)
            if (w[i] > 0.0) rows.push_back(i);
        const lint len = static_cast<lint>(rows.size());
        if (len == 0) return;
        any[static_cast<std::size_t>(c)] = 1;
        // the chunk's sqrt(w) x_K rows, one after another (kk x len column-major: a row of X
        // is read and written contiguously)
        std::vector<double> A(static_cast<std::size_t>(len * kk)), v(static_cast<std::size_t>(len));
        for (lint t = 0; t < len; ++t) {
            const lint i = rows[t];
            const double s = std::sqrt(w[i]);
            v[t] = s * z[i];
            double* a = A.data() + t * kk;
            const std::byte* xi = M.X + i * M.xrs;
            for (lint jj = 0; jj < kk; ++jj) {
                const lint col = K[jj];
                a[jj] = s * ((M.intercept && col == 0) ? 1.0 : rd(xi + (col - (M.intercept ? 1 : 0)) * M.xcs));
            }
        }
        double* G = part.data() + static_cast<std::size_t>(c) * slot;
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, kk, len, 1.0, A.data(), kk, 0.0, G, kk);
        cblas_dgemv(CblasColMajor, CblasNoTrans, kk, len, 1.0, A.data(), kk, v.data(), 1, 0.0, G + kk * kk, 1);
    });
    std::vector<double> G(slot, 0.0);
    for (lint c = 0; c < nc; ++c) {
        if (!any[static_cast<std::size_t>(c)]) continue;
        const double* q = part.data() + static_cast<std::size_t>(c) * slot;
        for (std::size_t e = 0; e < slot; ++e) G[e] += q[e];
    }
    double* U = G.data();
    double* h = G.data() + kk * kk;
    std::vector<double> nrm(static_cast<std::size_t>(kk));
    for (lint jj = 0; jj < kk; ++jj) nrm[jj] = std::sqrt(U[jj * kk + jj]);  // ||sqrt(w) x_j||
    if (linalg::lapack::potrf('U', kk, U, kk) > 0) return false;
    const double rc = linalg::lapack::trcon_1('U', kk, U, kk);
    if (!(rc > 0.0) || 0x1p-53 / (rc * rc) > kCholMaxErr) return false;
    // G21's aliasing rule on the Cholesky factor, which is the QR's R up to signs: a column
    // with |U_jj| <= rtol ||x_j|| (with a factor 2 of margin for the factor's rounding) would be
    // aliased; the QR decides it
    lint m = 0;
    for (lint i = 0; i < n; ++i) m += w[i] > 0.0;
    const double tol = M.rtol >= 0 ? M.rtol : static_cast<double>(std::max(m, kk)) * 0x1p-52;
    for (lint jj = 0; jj < kk; ++jj)
        if (!(std::fabs(U[jj * kk + jj]) > 2.0 * tol * nrm[jj])) return false;
    linalg::lapack::trtrs('U', 'T', 'N', kk, 1, U, kk, h, kk);
    linalg::lapack::trtrs('U', 'N', 'N', kk, 1, U, kk, h, kk);
    for (lint j = 0; j < k; ++j) beta[j] = kNaN, kept[j] = 0;
    for (lint jj = 0; jj < kk; ++jj) {
        if (!std::isfinite(h[jj])) return false;
        beta[K[jj]] = h[jj];
        kept[K[jj]] = 1;
    }
    // eta = X beta + offset in plain double: an inner iterate's error in eta enters the next
    // least squares fit only at second order
    for_row_chunks(n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            double acc = 0.0;
            for (lint jj = 0; jj < kk; ++jj) {
                const lint col = K[jj];
                const double x = (M.intercept && col == 0) ? 1.0
                                                           : rd(M.X + i * M.xrs + (col - (M.intercept ? 1 : 0)) * M.xcs);
                acc += x * h[jj];
            }
            eta[i] = acc + M.o[i];
        }
    });
    return true;
}

struct IrlsOut {
    int iters = 0;
    bool converged = false;
    bool ok = true;  // a valid iterate with coefficients
};

inline bool rises(const DD& a, const DD& b, double tol) {  // (b - a)/(|b| + 0.1) >= tol
    const double d = (b.hi - a.hi) + (b.lo - a.lo);
    return d / (std::fabs(b.hi) + 0.1) >= tol;
}
inline bool settled(const DD& a, const DD& b, double tol) {  // |b - a|/(|b| + 0.1) < tol
    const double d = (b.hi - a.hi) + (b.lo - a.lo);
    return std::fabs(d) / (std::fabs(b.hi) + 0.1) < tol;
}

// IRLS from the evaluated state W.cur (with coefficients W.beta when W.have_beta), at most
// max_lean iterations. On return W.cur is the last iterate and W.beta its coefficients.
IrlsOut irls(const Model& M, Work& W, int max_lean, double tol, int inner_steps, bool inner_chol, Scratch& sc,
             int newton = 2) {
    IrlsOut out;
    const lint k = M.k;
    bool conv = false;
    int t_conv = 0;
    double prevstep = kInf;
    // G44: Fisher scoring seen to be slow before the deviance's convergence (kSlowRuns
    // iterations in a row whose step is not below kSlowRatio of the one before) switches to
    // Newton's steps for the rest (a non-canonical link; for a canonical one they agree)
    bool newton_mode = false;
    int slow_runs = 0;
    for (int t = 1; t <= max_lean; ++t) {
        working(M, W.cur, W.z, W.w, W.rho);
        // past the deviance's convergence, Newton's steps (quadratic) in place of Fisher's
        // (linear for a non-canonical link): the least squares core's refined Newton step
        // through the thin Q (G42)
        const bool polish_newton = newton > 0 && (conv || newton_mode) && !canonical(M) && W.have_beta;
        if (polish_newton) {  // c and the state's eta - offset, scratch until the next evaluate
            obs_correction(M, W.cur, W.w, W.nxt.mu);
            for (lint i = 0; i < M.n; ++i) W.nxt.muc[i] = W.cur.eta[i] - M.o[i];
        }
        if (!W.have_kept) std::fill(W.kept, W.kept + k, std::uint8_t(1));  // every column until a QR drops one
        // before the deviance's convergence: the normal equations over the columns the last QR
        // kept (G40), the QR when they cannot (failure, ill-conditioning, G21's rule)
        const bool by_chol = !conv && !polish_newton && inner_chol && double(M.n) * double(M.k) >= kCholMinWork &&
                             solve_chol(M, W.z, W.w, W.kept, W.beta_new, W.kept_new, W.nxt.eta);
        if (!by_chol) {
            // the polishing iterations refine as the final solve does: their iterate's error is
            // what the final step starts from
            solve_lean(M, W.z, W.w, W.beta_new, W.kept_new, W.nxt.eta, conv ? kPolishSteps : inner_steps, sc, W.R,
                       polish_newton ? W.nxt.mu : nullptr, polish_newton ? W.beta : nullptr,
                       polish_newton ? W.nxt.muc : nullptr);
            W.have_kept = true;
        }
        evaluate(M, W.nxt, W.part);
        auto ok_state = [&](bool first) {
            return W.nxt.valid && !(!first && rises(W.cur.dev, W.nxt.dev, tol));
        };
        const bool first = !W.have_beta;
        bool eta_only = false;
        bool halved = false;
        if (!ok_state(first)) {
            halved = true;
            bool ok = false;
            if (!W.have_beta) {
                // no previous coefficients (the start is eta_0 = g(mu_0), not X beta): halve
                // towards the least squares fit of the current eta with the same weights when
                // that is a valid iterate, else (below) towards eta itself
                double* z0 = W.nxt.mu;  // scratch until the next evaluate
                for (lint i = 0; i < M.n; ++i) z0[i] = W.w[i] > 0.0 ? W.cur.eta[i] - M.o[i] : 0.0;
                double* keep_eta = W.rho;  // the step's eta, kept aside (rho is not needed again)
                std::copy(W.nxt.eta, W.nxt.eta + M.n, keep_eta);
                solve_lean(M, z0, W.w, W.beta, W.kept, W.nxt.eta, 0, sc);
                evaluate(M, W.nxt, W.part);
                if (W.nxt.valid) {
                    W.have_beta = true;
                } else {
                    std::copy(keep_eta, keep_eta + M.n, W.nxt.eta);
                    evaluate(M, W.nxt, W.part);
                }
            }
            if (W.have_beta) {
                // halve the step towards the previous coefficients
                for (int h = 0; h < 64 && !ok; ++h) {
                    bool same = true;
                    for (lint c = 0; c < k; ++c) {
                        const double a = std::isnan(W.beta_new[c]) ? 0.0 : W.beta_new[c];
                        const double b = std::isnan(W.beta[c]) ? 0.0 : W.beta[c];
                        const double mid = 0.5 * (a + b);
                        if (mid != a) same = false;
                        W.beta_new[c] = (std::isnan(W.beta_new[c]) && std::isnan(W.beta[c])) ? kNaN : mid;
                    }
                    xbeta_all(M, W.beta_new, W.nxt.eta);
                    evaluate(M, W.nxt, W.part);
                    ok = ok_state(first);
                    if (same) break;
                }
                for (lint c = 0; c < k; ++c) W.kept_new[c] = !std::isnan(W.beta_new[c]);
            } else {
                // no previous coefficients (the start is eta_0 = g(mu_0), not X beta): halve
                // eta towards the start; the next iteration's least squares fit starts from there
                for (int h = 0; h < 64 && !ok; ++h) {
                    for (lint i = 0; i < M.n; ++i) W.nxt.eta[i] = 0.5 * (W.nxt.eta[i] + W.cur.eta[i]);
                    evaluate(M, W.nxt, W.part);
                    ok = ok_state(first);
                }
                eta_only = true;
            }
            if (!ok) {
                // stuck: the last valid iterate stays
                out.iters = t;
                out.converged = conv;
                out.ok = W.have_beta;
                return out;
            }
        }
        // the coefficients' step (columns kept in both)
        double step = kInf, bn = 0.0;
        if (W.have_beta) {
            step = 0.0;
            for (lint c = 0; c < k; ++c) {
                if (std::isnan(W.beta_new[c])) continue;
                bn = std::max(bn, std::fabs(W.beta_new[c]));
                if (!std::isnan(W.beta[c])) step = std::max(step, std::fabs(W.beta_new[c] - W.beta[c]));
            }
        }
        const bool dev_conv = settled(W.cur.dev, W.nxt.dev, tol);  // halved or not (glm2)
        std::swap(W.cur, W.nxt);
        std::swap(W.beta, W.beta_new);
        std::swap(W.kept, W.kept_new);
        W.have_beta = !eta_only;
        out.iters = t;
        if (!conv && !newton_mode && newton > 1 && !canonical(M)) {
            // the ratio of successive Fisher steps; a halved step, the first one and an
            // iteration from a state without coefficients say nothing about the rate
            if (!halved && step < kInf && prevstep < kInf && prevstep > 0.0 && step >= kSlowRatio * prevstep) {
                if (++slow_runs >= kSlowRuns) newton_mode = true;
            } else {
                slow_runs = 0;
            }
        }
        if (dev_conv && !conv) {
            conv = true;
            t_conv = t;
        }
        if (conv) {
            // go on while the coefficients still move by more than rounding: stop when this
            // step is below it, or when the step that follows the next one (|d| r^2, r =
            // |d|/|d_prev|, an upper bound for quadratic convergence and the rate of a linear
            // one) would be, or when the steps stop contracting. The next step is the final
            // solve's, Newton's for every link (G42); with a non-canonical link the next step
            // itself (|d| r) must be below rounding, a margin measured to matter there (the
            // Newton steps' accuracy depends on the observed information, binomial log 9e-15
            // against 9e-16 at kappa 1e6)
            const double lim = 0x1p-50 * bn;
            if (step <= lim) break;
            if (prevstep < kInf) {
                const double r = step / prevstep;
                const double after = canonical(M) ? step * r * r : step * r;
                if (after <= lim) break;
                // not contracting: the rounding's floor. Not at the deviance's convergence
                // itself with a non-canonical link, whose next step is the first of Newton's
                // (it may be larger than a slow Fisher step)
                if (r > 0.5 && (canonical(M) || t > t_conv)) break;
            }
            if (!canonical(M) && t == t_conv && !newton_mode) {
                prevstep = kInf;  // the ratios are Newton's from here
                continue;
            }
        }
        prevstep = step;
    }
    out.converged = conv;
    return out;
}

// The starting state: mu_0 from R's family$initialize, eta_0 = g(mu_0). Returns false when
// no valid start exists (the gaussian family's log link with no positive mean).
bool start(const Model& M, State& s, double* part) {
    const lint n = M.n;
    double* mu0 = s.mu;  // mu_0 first, then evaluate() recomputes mu from eta_0 (as glm.fit)
    double ybar = 0.0;
    bool any_nonpos = false;
    if (M.fam == fGaussian && M.link == lLog) {
        DD sy{0, 0}, sw{0, 0};
        for (lint i = 0; i < n; ++i)
            if (M.pw[i] > 0.0) {
                sy = sy + DD{M.pw[i] * M.y[i], 0.0};
                sw = sw + DD{M.pw[i], 0.0};
                if (!(M.y[i] > 0.0)) any_nonpos = true;
            }
        ybar = sw.hi > 0 ? sy.hi / sw.hi : kNaN;
        if (any_nonpos && !(ybar > 0.0)) return false;
    }
    for (lint i = 0; i < n; ++i) {
        const double y = M.y[i];
        switch (M.fam) {
            case fBinomial: {
                const double w = M.m[i];
                mu0[i] = (w * y + 0.5) / (w + 1.0);
                break;
            }
            case fPoisson: mu0[i] = y + 0.1; break;
            case fGaussian: mu0[i] = (M.link == lLog && !(y > 0.0)) ? ybar : y; break;
            default: mu0[i] = y; break;
        }
    }
    link_fun(M, mu0, s.eta, 0, n);
    evaluate(M, s, part);
    return true;
}

// ---- the regression

struct Opts {
    int fam, link;
    bool intercept;
    double rtol;
    int cov;
    std::int64_t maxlags, gtot;
    int max_iter;
    double tol;
    int scale_mode;
    double scale_value;
    bool has_var, has_freq, has_trials, has_offset;
    int inner_steps;
    int final_steps;
    bool inner_chol;
    int newton;
};

struct Outs {
    double *params, *cov, *R, *mu, *eta, *rresp, *rpear, *rdev, *rwork, *stats;
    std::uint8_t* kept;
};

void fill_nan(lint n, lint k, const Outs& o) {
    std::fill(o.params, o.params + k, kNaN);
    std::fill(o.cov, o.cov + k * k, kNaN);
    std::fill(o.R, o.R + k * k, kNaN);
    for (double* a : {o.mu, o.eta, o.rresp, o.rpear, o.rdev, o.rwork}) std::fill(a, a + n, kNaN);
    std::fill(o.kept, o.kept + k, std::uint8_t(0));
    std::fill(o.stats, o.stats + gNStat, kNaN);
    o.stats[gRank] = 0;
    o.stats[gIsNaN] = 1;
    o.stats[gConverged] = 0;
    o.stats[gSeparation] = 0;
}

// The boundary rows' sign: +1 where mu can go to 1 (binomial y = 1, but not with the log
// link, whose eta stays below 0), -1 where it can go to 0 (binomial y = 0, poisson y = 0
// with the log link); 0 for every other row.
inline int boundary(const Model& M, double y) {
    if (M.fam == fBinomial) {
        if (y == 0.0) return -1;
        if (y == 1.0 && M.link != lLog) return 1;
    } else if (M.fam == fPoisson && M.link == lLog && y == 0.0) {
        return -1;
    }
    return 0;
}

// Rows of the fit that fail the existence certificate of the Fisher step d taken from the
// state (rho, w) (deta = X d, X^T W X d = score): a boundary row passes when it was in the
// fit's least squares and |deta_i| < |rho_i|/2.
std::vector<lint> failing(const Model& M, const double* w, const double* rho, const double* eta_from,
                          const double* eta_to) {
    std::vector<lint> S;
    for (lint i = 0; i < M.n; ++i) {
        if (!(M.pw[i] > 0.0) || boundary(M, M.y[i]) == 0) continue;
        const double de = eta_to[i] - eta_from[i];
        if (!(w[i] > 0.0 && std::fabs(de) < 0.5 * std::fabs(rho[i]))) S.push_back(i);
    }
    return S;
}

// The direction test: a direction b (k, NaN taken as 0) projected onto the null space of
// the other rows of the fit must point every row of S (shrunk to the rows it does) to its
// boundary by more than sqrt(eps) of the terms. Returns the certified rows (empty: none).
std::vector<lint> direction(const Model& M, const double* bdir, std::vector<lint> S, Scratch& sc) {
    const lint n = M.n, k = M.k;
    std::vector<double> b(static_cast<std::size_t>(k)), c(static_cast<std::size_t>(std::max<lint>(k, 1)));
    std::vector<double> v(static_cast<std::size_t>(std::max<lint>(n, 1))), w1(static_cast<std::size_t>(std::max<lint>(n, 1)));
    std::vector<double> e(static_cast<std::size_t>(std::max<lint>(n, 1)));
    std::vector<std::uint8_t> kept(static_cast<std::size_t>(std::max<lint>(k, 1)));
    std::vector<double> zero(static_cast<std::size_t>(std::max<lint>(n, 1)), 0.0);
    Model Mz = M;
    Mz.o = zero.data();
    auto x_at = [&](lint i, lint col) {
        return (M.intercept && col == 0) ? 1.0 : rd(M.X + i * M.xrs + (col - (M.intercept ? 1 : 0)) * M.xcs);
    };
    for (int round = 0; round < 64 && !S.empty(); ++round) {
        std::vector<std::uint8_t> inS(static_cast<std::size_t>(n), 0);
        for (lint i : S) inS[static_cast<std::size_t>(i)] = 1;
        for (lint col = 0; col < k; ++col) b[col] = std::isnan(bdir[col]) ? kNaN : bdir[col];
        bool anyR = false;
        for (lint i = 0; i < n; ++i) {
            w1[i] = (M.pw[i] > 0.0 && !inS[static_cast<std::size_t>(i)]) ? 1.0 : 0.0;
            anyR = anyR || w1[i] > 0.0;
        }
        if (anyR) {
            xbeta_all(Mz, b.data(), v.data());  // X b
            solve_lean(Mz, v.data(), w1.data(), c.data(), kept.data(), e.data(), 0, sc);
            for (lint col = 0; col < k; ++col) {
                const double bb = std::isnan(b[col]) ? 0.0 : b[col];
                const double cc = std::isnan(c[col]) ? 0.0 : c[col];
                b[col] = bb - cc;
            }
        } else {
            for (lint col = 0; col < k; ++col) b[col] = std::isnan(b[col]) ? 0.0 : b[col];
        }
        std::vector<lint> good;
        for (lint i : S) {
            double P = 0.0, Sd = 0.0, mag = 0.0;
            for (lint col = 0; col < k; ++col) {
                const double t = x_at(i, col) * b[col];
                double hp, ep, q;
                two_prod(x_at(i, col), b[col], hp, ep);
                two_sum(P, hp, P, q);
                Sd += q + ep;
                mag += std::fabs(t);
            }
            const double u = boundary(M, M.y[i]) * (P + Sd);
            if (u > kSqrtEps * mag) good.push_back(i);
        }
        if (good.size() == S.size()) return S;
        S.swap(good);
    }
    return {};
}

int glm_one(const Opts& op, lint n, lint p, const std::byte* yb, std::int64_t ys, const std::byte* Xb,
            std::int64_t xrs, std::int64_t xcs, const std::byte* ab, std::int64_t ars, std::int64_t acs,
            const std::byte* gb, std::int64_t gs, const Outs& o, Scratch& sc) {
    const lint k = p + (op.intercept ? 1 : 0);
    // ---- 1. checks
    bool nan = false;
    for (lint i = 0; i < n; ++i) {
        const double v = rd(yb + i * ys);
        if (std::isnan(v)) nan = true;
        else if (std::isinf(v)) return eInfY;
    }
    {
        Data dx{nullptr, 0, Xb, xrs, xcs, nullptr, 0, nullptr, 0, nullptr, 0};
        if (scan_x(dx, n, p, nan)) return eInfX;
    }
    for (int r = 0; r < 4; ++r) {
        const bool has = r == 0 ? op.has_var : r == 1 ? op.has_freq : r == 2 ? op.has_trials : op.has_offset;
        if (!has) continue;
        for (lint i = 0; i < n; ++i) {
            const double v = rd(ab + r * ars + i * acs);
            if (std::isnan(v)) {
                nan = true;
                continue;
            }
            if (std::isinf(v)) return r == 0 ? eInfVar : r == 1 ? eInfFreq : r == 2 ? eTrials : eInfOffset;
            if (r == 0 && v < 0) return eNegVar;
            if (r == 1 && v < 0) return eNegFreq;
            if (r == 2 && !(v > 0)) return eTrials;
        }
    }
    if (gb)
        for (lint i = 0; i < n; ++i)
            if (rd_i64(gb + i * gs) < 0) nan = true;
    if (nan) {
        fill_nan(n, k, o);
        return 0;
    }
    // y in the family's range (every row, as R)
    for (lint i = 0; i < n; ++i) {
        const double y = rd(yb + i * ys);
        bool okv = true;
        switch (op.fam) {
            case fBinomial: okv = op.has_trials ? (y >= 0.0 && y <= 1.0) : (y == 0.0 || y == 1.0); break;
            case fPoisson: okv = y >= 0.0; break;
            case fGamma: okv = y > 0.0; break;
            default: break;
        }
        if (!okv) return eBadY;
    }

    // ---- per-row arrays
    const lint nn = std::max<lint>(n, 1);
    std::unique_ptr<double[]> rows(new double[static_cast<std::size_t>(8 * nn)]);
    double *y = rows.get(), *v = y + nn, *f = v + nn, *tr = f + nn, *m = tr + nn, *pw = m + nn, *off = pw + nn,
           *aux = off + nn;
    DD nobs{0.0, 0.0};
    bool has_offset = false;
    for (lint i = 0; i < n; ++i) {
        y[i] = rd(yb + i * ys);
        v[i] = op.has_var ? rd(ab + 0 * ars + i * acs) : 1.0;
        f[i] = op.has_freq ? rd(ab + 1 * ars + i * acs) : 1.0;
        tr[i] = op.has_trials ? rd(ab + 2 * ars + i * acs) : 1.0;
        off[i] = op.has_offset ? rd(ab + 3 * ars + i * acs) : 0.0;
        has_offset = has_offset || off[i] != 0.0;
        m[i] = v[i] * tr[i];
        pw[i] = f[i] * m[i];
        if (pw[i] > 0.0) nobs = nobs + DD{f[i], 0.0};
    }
    Model M{op.fam, op.link, n, p, k, op.intercept, op.rtol, y, v, f, tr, m, pw, off, Xb, xrs, xcs};

    // ---- 2. IRLS
    Work W(n, k);
    if (!start(M, W.cur, W.part)) return eNoStart;
    int total_iters = 0;
    IrlsOut ir;
    if (W.cur.valid) {
        ir = irls(M, W, std::max(op.max_iter - 1, 0), op.tol, op.inner_steps, op.inner_chol, sc, op.newton);
    } else {
        ir.ok = false;  // the start itself is not valid (cannot happen for R's starts)
    }
    total_iters = ir.iters;

    // ---- 3. the final solve at the last iterate's weights
    std::vector<double> smult, hdiff;
    working(M, W.cur, W.z, W.w, W.rho);
    if (op.cov != kNonrobust) {
        if (op.has_freq) {
            smult.resize(static_cast<std::size_t>(n));
            for (lint i = 0; i < n; ++i) smult[i] = f[i] > 0 ? 1.0 / std::sqrt(f[i]) : 0.0;
        }
    }
    // a non-canonical link: the observed information's correction, for the robust bread and for
    // the final Newton step through the thin Q (G42)
    const bool newton_final = op.newton > 0 && !canonical(M) && ir.ok && W.have_beta;
    if (!canonical(M) && (op.cov != kNonrobust || newton_final)) {
        hdiff.resize(static_cast<std::size_t>(n));
        obs_correction(M, W.cur, W.w, hdiff.data());
    }
    std::uint8_t newton_used = 0;
    std::vector<double> eta0;
    std::vector<double> eta_fin(static_cast<std::size_t>(nn)), resid_fin(static_cast<std::size_t>(nn));
    double fstats[kNStat];
    {
        Problem pb{n, p, k, op.intercept, true, false, gb != nullptr, op.rtol, op.cov, 3, op.maxlags, op.gtot, false};
        pb.steps = op.final_steps;
        pb.glm = true;
        Data d{reinterpret_cast<const std::byte*>(W.z), 8, Xb, xrs, xcs, reinterpret_cast<const std::byte*>(W.w), 8,
               nullptr, 0, gb, gs};
        d.smult = smult.empty() ? nullptr : smult.data();
        d.hdiff = hdiff.empty() ? nullptr : hdiff.data();
        d.nobs_eff = nobs.hi;
        d.newton_b0 = newton_final ? W.beta : nullptr;
        if (newton_final) {
            eta0.resize(static_cast<std::size_t>(n));
            for (lint i = 0; i < n; ++i) eta0[i] = W.cur.eta[i] - off[i];
            d.newton_eta0 = eta0.data();
        }
        Dest dst{o.params, o.cov, o.R, eta_fin.data(), resid_fin.data(), o.kept, fstats, nullptr};
        dst.newton = &newton_used;
        fit_one(pb, d, dst, sc);
    }
    total_iters += 1;
    // the final state (Newton's step for a non-canonical link, G42)
    for (lint i = 0; i < n; ++i) eta_fin[i] += off[i];
    std::copy(eta_fin.begin(), eta_fin.begin() + n, W.nxt.eta);
    evaluate(M, W.nxt, W.part);
    // the last iterate has coefficients (at least one iteration, not halved in eta alone)
    const bool have_beta = ir.ok && W.have_beta;
    bool use_final = have_beta && W.nxt.valid && !rises(W.cur.dev, W.nxt.dev, op.tol);
    if (!have_beta && W.nxt.valid) use_final = true;  // no iterate with coefficients: the final solve is all there is
    bool converged = ir.converged || (ir.ok && use_final && settled(W.cur.dev, W.nxt.dev, op.tol));
    const lint rank = static_cast<lint>(fstats[kRank]);
    o.stats[gNGroups] = fstats[kNGroups];
    o.stats[gLags] = fstats[kLags];

    // ---- 4. separation
    std::vector<std::uint8_t> sep(static_cast<std::size_t>(nn), 0);
    bool separated = false;
    State* fin = use_final ? &W.nxt : &W.cur;
    const bool capable = op.fam == fBinomial || (op.fam == fPoisson && op.link == lLog);
    std::unique_ptr<Work> RW;  // the restricted fit's buffers
    if (capable && have_beta) {
        // the certificate is about Fisher's step (X^T W X d = score, which the least squares
        // solve satisfies to rounding): when the final step was Newton's (a non-canonical link,
        // G42), Fisher's is solved again from the same weights (refined)
        const double* eta_cert = eta_fin.data();
        const double* beta_cert = o.params;
        std::vector<double> etaF, betaF;
        if (newton_used) {
            etaF.resize(static_cast<std::size_t>(nn));
            betaF.resize(static_cast<std::size_t>(std::max<lint>(k, 1)));
            std::vector<std::uint8_t> keptF(static_cast<std::size_t>(std::max<lint>(k, 1)));
            solve_lean(M, W.z, W.w, betaF.data(), keptF.data(), etaF.data(), kPolishSteps, sc);
            eta_cert = etaF.data();
            beta_cert = betaF.data();
        }
        std::vector<lint> S = failing(M, W.w, W.rho, W.cur.eta, eta_cert);
        std::vector<double> bdir(static_cast<std::size_t>(std::max<lint>(k, 1)));
        for (lint c = 0; c < k; ++c)
            bdir[c] = (std::isnan(beta_cert[c]) || std::isnan(W.beta[c])) ? kNaN : beta_cert[c] - W.beta[c];
        std::vector<double> pw_r(pw, pw + n);
        Model MR = M;
        MR.pw = pw_r.data();
        const State* from = &W.cur;  // the state the certificate's step starts from
        for (int round = 0; round < 8 && !S.empty(); ++round) {
            std::vector<lint> cert = direction(MR, bdir.data(), S, sc);
            if (cert.empty()) break;
            separated = true;
            for (lint i : cert) {
                sep[static_cast<std::size_t>(i)] = 1;
                pw_r[i] = 0.0;
            }
            // the rest again, from the current state
            if (!RW) RW = std::make_unique<Work>(n, k);
            Work& R = *RW;
            if (from->eta != R.cur.eta) std::copy(from->eta, from->eta + n, R.cur.eta);
            evaluate(MR, R.cur, R.part);
            bool any = false;
            for (lint i = 0; i < n; ++i) any = any || pw_r[i] > 0.0;
            if (!any) {
                fin = nullptr;
                break;
            }
            R.have_beta = false;
            R.have_kept = false;  // other rows: aliasing again from a QR
            IrlsOut rr = irls(MR, R, op.max_iter, op.tol, op.inner_steps, op.inner_chol, sc);
            total_iters += rr.iters;
            converged = converged && rr.converged;
            fin = &R.cur;
            if (!(rr.ok && R.have_beta)) break;  // no iterate with coefficients to test from
            // the certificate on the rest: one more Fisher step from its last iterate
            working(MR, R.cur, R.z, R.w, R.rho);
            solve_lean(MR, R.z, R.w, R.beta_new, R.kept_new, R.nxt.eta, 0, sc);
            S = failing(MR, R.w, R.rho, R.cur.eta, R.nxt.eta);
            for (lint c = 0; c < k; ++c)
                bdir[c] = (std::isnan(R.beta_new[c]) || std::isnan(R.beta[c])) ? kNaN : R.beta_new[c] - R.beta[c];
            from = &R.cur;
        }
    }

    // ---- 5. outputs
    if (separated) {
        std::fill(o.params, o.params + k, kNaN);
        std::fill(o.cov, o.cov + k * k, kNaN);
    } else if (!use_final) {
        // the final solve's step was not taken: the last iterate's coefficients
        for (lint c = 0; c < k; ++c) o.params[c] = o.kept[c] ? W.beta[c] : kNaN;
    }
    if (!have_beta && !use_final) {
        std::fill(o.params, o.params + k, kNaN);
        std::fill(o.cov, o.cov + k * k, kNaN);
    }
    // per row: mu, eta, residuals; sums of deviance and Pearson chi2 over the fit's rows
    const double* feta = fin ? fin->eta : nullptr;
    const double* fmu = fin ? fin->mu : nullptr;
    const double* fmuc = fin ? fin->muc : nullptr;
    const double* fd1 = fin ? fin->d1 : nullptr;
    const lint nc = row_chunks(n);
    std::vector<double> part(static_cast<std::size_t>(6 * nc));
    const bool gid = op.fam == fGaussian && op.link == lIdentity;
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        DD dv{0, 0}, pe{0, 0}, slw{0, 0};
        for (lint i = i0; i < i1; ++i) {
            double mu, muc, g, e;
            if (sep[static_cast<std::size_t>(i)]) {
                const int s = boundary(M, y[i]);
                mu = s > 0 ? 1.0 : 0.0;
                muc = 1.0 - mu;
                e = s > 0 ? kInf : -kInf;
                o.mu[i] = mu;
                o.eta[i] = e;
                o.rresp[i] = o.rpear[i] = o.rdev[i] = o.rwork[i] = 0.0;
                continue;
            }
            if (!fin) {
                o.mu[i] = o.eta[i] = o.rresp[i] = o.rpear[i] = o.rdev[i] = o.rwork[i] = kNaN;
                continue;
            }
            mu = fmu[i], muc = fmuc[i], g = fd1[i], e = feta[i];
            o.mu[i] = mu;
            o.eta[i] = e;
            const double r = y_minus_mu(op.fam, y[i], mu, muc);
            const double V = variance(op.fam, mu, muc);
            const double ud = unit_dev(op.fam, y[i], mu, muc);
            o.rresp[i] = r;
            o.rpear[i] = std::sqrt(m[i]) * r / std::sqrt(V);
            const double dd = m[i] * ud;
            o.rdev[i] = std::copysign(std::sqrt(std::max(dd, 0.0)), r);
            o.rwork[i] = r / g;
            if (pw[i] > 0.0) {
                dv = dv + DD{f[i] * dd, 0.0};
                pe = pe + DD{pw[i] * r * r / V, 0.0};
                if (op.has_var) slw = slw + DD{f[i] * std::log(v[i]), 0.0};
            }
        }
        double* q = part.data() + 6 * c;
        q[0] = dv.hi, q[1] = dv.lo, q[2] = pe.hi, q[3] = pe.lo, q[4] = slw.hi, q[5] = slw.lo;
    });
    DD dev{0, 0}, pear{0, 0}, slogw{0, 0};
    for (lint c = 0; c < nc; ++c) {
        const double* q = part.data() + 6 * c;
        dev = dev + DD{q[0], q[1]};
        pear = pear + DD{q[2], q[3]};
        slogw = slogw + DD{q[4], q[5]};
    }
    double D = dev.hi, P = pear.hi;
    if (gid && use_final && !separated) D = P = fstats[kSsr];  // the least squares' own sum (as ols)
    const double nob = nobs.hi;
    const double df_resid = nob - static_cast<double>(rank);
    double phi = 1.0;
    switch (op.scale_mode) {
        case sX2: phi = P / df_resid; break;
        case sDev: phi = D / df_resid; break;
        case sFixed: phi = op.scale_value; break;
        default: phi = (op.fam == fGaussian || op.fam == fGamma) ? P / df_resid : 1.0; break;
    }
    // log-likelihood at the scale (statsmodels' loglike; gaussian identity: concentrated)
    // the log-likelihood's terms free of mu, once for llf and llnull: v lgamma(y + 1)
    // (poisson), lgamma(v/scale) (gamma), v log C(n, s) (binomial); 0 without a lgamma where
    // the value is exactly 0 (y = 0, 1; s = 0, n)
    if (op.fam != fGaussian) {
        for_row_chunks(n, [&](lint, lint i0, lint i1) {
            double vprev = kNaN, gprev = 0.0;
            for (lint i = i0; i < i1; ++i) {
                if (!(pw[i] > 0.0)) continue;
                const double yy = y[i];
                if (op.fam == fPoisson) {
                    aux[i] = (yy == 0.0 || yy == 1.0) ? 0.0 : v[i] * special::Gammaln::f(yy + 1.0);
                } else if (op.fam == fGamma) {
                    if (v[i] != vprev) vprev = v[i], gprev = special::Gammaln::f(v[i] / phi);
                    aux[i] = gprev;
                } else {
                    const double nt = tr[i], sv = nt * yy, fl = nt - sv;
                    aux[i] = (sv == 0.0 || fl == 0.0)
                                 ? 0.0
                                 : v[i] * (special::Gammaln::f(nt + 1.0) - special::Gammaln::f(sv + 1.0) -
                                           special::Gammaln::f(fl + 1.0));
                }
            }
        });
    }
    auto llf_at = [&](const double* mus, const double* mucs, double mconst, double mcconst, const std::uint8_t* sp,
                      bool concentrated, double ssr_c) -> double {
        if (concentrated) {
            // -n/2 (log(2 pi) + log(ssr/n) + 1) + sum f log v / 2, as ols's llf
            return -nob / 2.0 * (std::log(2.0 * M_PI) + std::log(ssr_c / nob) + 1.0) + slogw.hi / 2.0;
        }
        std::vector<double> lp(static_cast<std::size_t>(2 * nc));
        for_row_chunks(n, [&](lint c, lint i0, lint i1) {
            DD acc{0, 0};
            for (lint i = i0; i < i1; ++i) {
                if (!(pw[i] > 0.0) || (sp && sp[i])) continue;  // separated rows: 0 at the boundary
                const double mu = mus ? mus[i] : mconst, muc = mucs ? mucs[i] : mcconst;
                const double yy = y[i];
                double l;
                switch (op.fam) {
                    case fGaussian: {
                        const double d = yy - mu;
                        l = -0.5 * (v[i] * d * d / phi + std::log(phi / v[i]) + std::log(2.0 * M_PI));
                        break;
                    }
                    case fPoisson:
                        l = (v[i] * ((yy == 0.0 ? 0.0 : yy * std::log(mu)) - mu) - aux[i]) / phi;
                        break;
                    case fGamma: {
                        const double ws = v[i] / phi;
                        l = ws * std::log(ws * yy / mu) - ws * yy / mu - aux[i] - std::log(yy);
                        break;
                    }
                    default: {
                        const double nt = tr[i], sv = nt * yy, fl = nt - sv;
                        l = 0.0;
                        if (sv != 0.0) l += sv * log_mu(mu, muc);
                        if (fl != 0.0) l += fl * log_muc(mu, muc);
                        l = aux[i] + v[i] * l;
                        break;
                    }
                }
                acc = acc + DD{f[i] * l, 0.0};
            }
            lp[2 * c] = acc.hi;
            lp[2 * c + 1] = acc.lo;
        });
        DD t{0, 0};
        for (lint c = 0; c < nc; ++c) t = t + DD{lp[2 * c], lp[2 * c + 1]};
        return t.hi;
    };
    double llf = kNaN;
    if (gid) llf = llf_at(nullptr, nullptr, 0, 0, nullptr, true, D);
    else if (fin) llf = llf_at(fmu, fmuc, 0, 0, sep.data(), false, 0);
    else if (separated) llf = 0.0;  // every row at its boundary

    // the null model: the intercept alone (closed form without an offset, else IRLS), or
    // eta = offset without an intercept (R's glm)
    double Dnull = kNaN, llnull = kNaN;
    {
        std::vector<double> mun(static_cast<std::size_t>(nn)), mucn(static_cast<std::size_t>(nn));
        bool const_mu = false;
        double mc = kNaN, mcc = kNaN;
        bool okn = true;
        if (op.intercept && !has_offset) {
            DD sy{0, 0}, sy1{0, 0}, sw{0, 0};
            for (lint i = 0; i < n; ++i)
                if (pw[i] > 0.0) {
                    double h, l;
                    two_prod(pw[i], y[i], h, l);
                    sy = sy + DD{h, l};
                    two_prod(pw[i], 1.0 - y[i], h, l);
                    sy1 = sy1 + DD{h, l};
                    sw = sw + DD{pw[i], 0.0};
                }
            mc = special::dd::operator/(sy, sw).hi;
            mcc = special::dd::operator/(sy1, sw).hi;
            const_mu = true;
        } else if (op.intercept) {
            Model M0 = M;
            M0.p = 0;
            M0.k = 1;
            Work W0(n, 1);
            okn = start(M0, W0.cur, W0.part) && W0.cur.valid;
            if (okn) {
                IrlsOut r0 = irls(M0, W0, op.max_iter, op.tol, 0, op.inner_chol, sc);
                okn = r0.ok;
                std::copy(W0.cur.mu, W0.cur.mu + n, mun.begin());
                std::copy(W0.cur.muc, W0.cur.muc + n, mucn.begin());
            }
        } else {
            std::vector<double> e0(off, off + n), d0(static_cast<std::size_t>(nn));
            link_eval(M, e0.data(), mun.data(), mucn.data(), d0.data(), 0, n);
        }
        if (okn) {
            DD dn{0, 0}, sn{0, 0};
            for (lint i = 0; i < n; ++i) {
                if (!(pw[i] > 0.0)) continue;
                const double mu = const_mu ? mc : mun[i], muc = const_mu ? mcc : mucn[i];
                dn = dn + DD{pw[i] * unit_dev(op.fam, y[i], mu, muc), 0.0};
                if (gid) {
                    const double d = y[i] - mu;
                    sn = sn + DD{pw[i] * d * d, 0.0};
                }
            }
            Dnull = dn.hi;
            if (gid) llnull = llf_at(nullptr, nullptr, 0, 0, nullptr, true, sn.hi);
            else llnull = const_mu ? llf_at(nullptr, nullptr, mc, mcc, nullptr, false, 0)
                                   : llf_at(mun.data(), mucn.data(), 0, 0, nullptr, false, 0);
        }
    }

    double* st = o.stats;
    st[gNobs] = nob;
    st[gRank] = static_cast<double>(rank);
    st[gDev] = D;
    st[gNullDev] = Dnull;
    st[gPearson] = P;
    st[gScale] = phi;
    st[gLlf] = llf;
    st[gLlnull] = llnull;
    st[gNIter] = static_cast<double>(total_iters);
    st[gConverged] = converged ? 1.0 : 0.0;
    st[gSeparation] = separated ? 1.0 : 0.0;
    st[gIsNaN] = 0.0;
    return 0;
}

[[noreturn]] void raise_failure(const Batch& b, const Failure& f, const char* func, int fam, bool has_trials) {
    std::string what;
    switch (f.code) {
        case eInfY: what = "'y' contains infinity"; break;
        case eInfX: what = "'X' contains infinity"; break;
        case eInfVar: what = "'var_weights' contains infinity"; break;
        case eNegVar: what = "'var_weights' must be >= 0"; break;
        case eInfFreq: what = "'freq_weights' contains infinity"; break;
        case eNegFreq: what = "'freq_weights' must be >= 0"; break;
        case eTrials: what = "'trials' must be finite and > 0"; break;
        case eInfOffset: what = "'offset' contains infinity"; break;
        case eNoStart:
            what = "cannot find valid starting values: the log link of the gaussian family needs a positive mean "
                   "of 'y'";
            break;
        default:
            switch (fam) {
                case fBinomial:
                    what = has_trials ? "'y' must be in [0, 1] for the binomial family (proportions of 'trials')"
                                      : "'y' must be 0 or 1 for the binomial family without 'trials'";
                    break;
                case fPoisson: what = "'y' must be >= 0 for the poisson family"; break;
                default: what = "'y' must be > 0 for the gamma family"; break;
            }
    }
    throw nb::value_error(linalg::failure_message(b, f.index, (std::string(func) + "(): " + what).c_str()).c_str());
}

std::vector<std::int64_t> with(std::vector<std::int64_t> shape, std::initializer_list<std::int64_t> tail) {
    shape.insert(shape.end(), tail);
    return shape;
}

// fit(func, y (..., n), X (..., n, p), aux (..., 4, n) [var_weights, freq_weights, trials,
//     offset], groups (..., n) | None, gtot, family, link, has_var, has_freq, has_trials,
//     has_offset, intercept, rtol, cov, maxlags, max_iter, tol, scale_mode, scale_value,
//     inner_steps, final_steps)
// -> (params, cov, R, mu, eta, kept, stats, resid_response, resid_pearson, resid_deviance,
//     resid_working). cov is (X^T W X)^-1 (unscaled) for cov = nonrobust, else the sandwich.
// inner_steps, final_steps: the refinement steps of the inner iterations (0) and of the final
// solve (3); inner_chol: the inner iterations by the normal equations (true, G40); newton,
// for a non-canonical link: 2 Newton's polishing and final steps and the switch of slow
// Fisher scoring (G42, G44), 1 the former only, 0 Fisher's steps throughout.
// Only the benchmarks (bench/regression/g2/) change them.
nb::object fit(const std::string& func, nb::handle yo, nb::handle xo, nb::handle ao, nb::handle go, std::int64_t gtot,
               int family, int link, bool has_var, bool has_freq, bool has_trials, bool has_offset, bool intercept,
               double rtol, int cov, std::int64_t maxlags, int max_iter, double tol, int scale_mode, double scale_value,
               int inner_steps, int final_steps, bool inner_chol, int newton) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), X(xo, fn, "X"), A(ao, fn, "aux");
        const bool has_g = !go.is_none();
        std::optional<Input> g;
        if (has_g) g.emplace(go, fn, "groups");
        if (cov == kCluster && !has_g) throw nb::value_error("fit(): cluster needs groups");
        Opts op{family, link, intercept, rtol, cov, maxlags, gtot, max_iter, tol, scale_mode, scale_value,
                has_var, has_freq, has_trials, has_offset, inner_steps, final_steps, inner_chol, newton};
        const linalg::BatchOperand none{};
        Batch b({{y.batch_shape(1), y.batch_strides(1)},
                 {X.batch_shape(2), X.batch_strides(2)},
                 {A.batch_shape(2), A.batch_strides(2)},
                 has_g ? linalg::BatchOperand{g->batch_shape(1), g->batch_strides(1)} : none},
                fn);
        const lint n = X.shape(-2), p = X.shape(-1), k = p + (intercept ? 1 : 0);
        Out po = linalg::alloc_c(with(b.shape(), {k}), DType::Float64);
        Out co = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
        Out ro = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
        Out mo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out eo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out ko = linalg::alloc_c(with(b.shape(), {k}), DType::Bool);
        Out so = linalg::alloc_c(with(b.shape(), {gNStat}), DType::Float64);
        Out r1 = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out r2 = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out r3 = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out r4 = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        // estimated serial ns of one regression: about 6 iterations of a least squares fit
        const double nk = double(n) * double(std::max<lint>(k, 1));
        const double element_ns = 3000.0 + 6.0 * (nk * (0.6 * double(k) + 12.0)) + (cov ? 2.0 * nk * k : 0.0);
        const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            Outs o{po.as<double>() + i * k, co.as<double>() + i * k * k, ro.as<double>() + i * k * k,
                   mo.as<double>() + i * n, eo.as<double>() + i * n, r1.as<double>() + i * n, r2.as<double>() + i * n,
                   r3.as<double>() + i * n, r4.as<double>() + i * n, so.as<double>() + i * gNStat,
                   ko.as<std::uint8_t>() + i * k};
            return glm_one(op, n, p, y.data() + off[0], y.stride(-1), X.data() + off[1], X.stride(-2), X.stride(-1),
                           A.data() + off[2], A.stride(-2), A.stride(-1), has_g ? g->data() + off[3] : nullptr,
                           has_g ? g->stride(-1) : 0, o, sc);
        });
        if (f) raise_failure(b, f, fn, family, has_trials);
        return nb::make_tuple(linalg::to_array(po), linalg::to_array(co), linalg::to_array(ro), linalg::to_array(mo),
                              linalg::to_array(eo), linalg::to_array(ko), linalg::to_array(so), linalg::to_array(r1),
                              linalg::to_array(r2), linalg::to_array(r3), linalg::to_array(r4));
    });
}

}  // namespace

void bind_glm(nb::module_& m) {
    m.def("glm_fit", &fit, nb::arg("func"), nb::arg("y").none(), nb::arg("X").none(), nb::arg("aux").none(),
          nb::arg("groups").none(), nb::arg("gtot"), nb::arg("family"), nb::arg("link"), nb::arg("has_var"),
          nb::arg("has_freq"), nb::arg("has_trials"), nb::arg("has_offset"), nb::arg("intercept"), nb::arg("rtol"),
          nb::arg("cov"), nb::arg("maxlags"), nb::arg("max_iter"), nb::arg("tol"), nb::arg("scale_mode"),
          nb::arg("scale_value"), nb::arg("inner_steps") = 0, nb::arg("final_steps") = 3,
          nb::arg("inner_chol") = true, nb::arg("newton") = 2);
}

}  // namespace sd::regression
