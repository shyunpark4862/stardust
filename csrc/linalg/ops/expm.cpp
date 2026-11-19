// The matrix exponential (docs/api/linalg.md §7; docs/design/linalg.md §11).
// LAPACK has none, so it is written here: scaling and squaring with Padé
// approximants, the algorithm of Al-Mohy & Higham 2009 ("A new scaling and
// squaring algorithm for the matrix exponential", SIAM J. Matrix Anal. Appl.
// 31(3), Algorithm 5.1), which is also scipy.linalg.expm's.
//
//   exp(A) = (exp(2^-s A))^(2^s), and exp(B) ~ r_m(B) = (V - U)^-1 (V + U),
//
// the [m/m] Padé approximant, U and V polynomials in B (U odd, V even). The
// degree m in {3, 5, 7, 9, 13} and the number of squarings s are the cheapest
// pair whose backward error is below the unit roundoff, judged from the norms
// ||A^k||^(1/k) (which can be far below ||A|| for nonnormal A: less scaling,
// fewer squarings, less error) plus a correction ell() that adds squarings
// when |A|'s powers say the bound would be too optimistic.
//
// Decisions (docs/design/linalg.md §11):
//   - The norms of A^2 .. A^6 are exact: those powers are formed with gemm,
//     and only when a test needs them (bounds from known norms first). So are
//     ||A^8|| and ||A^10|| below n = kExpmEstimateMin; from there on they are
//     estimated with LAPACK's lacn2 instead (exact first, then the estimate
//     after measuring, docs/design/linalg.md §11.2; scipy switches to a
//     randomized 1-norm estimator for n >= 400). ||  |A|^p ||_1 is exact too: |A| is
//     nonnegative, so its 1-norm is the largest entry of e^T |A|^p (p gemv).
//   - Diagonal input: exp of the diagonal, exactly. Triangular input: after
//     every squaring the diagonal and the first off-diagonal are overwritten
//     with their exact values (the paper's Code Fragment 2.1, as scipy;
//     docs/design/linalg.md §11.3). The other triangle of the result is exactly 0.
//   - Finite input only (docs/design/linalg.md §11.4: the final solve is LAPACK's
//     getrf/getrs).
#include "linalg/ops/expm.hpp"
#include "bind/errors.hpp"

#include "linalg/cost.hpp"
#include "linalg/outblock.hpp"
#include "linalg/ops/common.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <utility>

namespace sd::linalg {

namespace {

// From this size on, ||A^8|| and ||A^10|| are estimated, not formed (measured,
// docs/design/linalg.md §11.2: the estimate is faster from n = 100 on,
// 1.03-1.15x; it changed none of the measured results). A variable only so that
// measurements can move it (_core._set_expm_estimate_min).
lint kExpmEstimateMin = 100;

// theta_m: the largest ||2^-s A|| (in the eta sense) for which r_m's backward
// error is below 2^-53 (Al-Mohy & Higham 2009, Table 3.1; theta_13 = 4.25 as
// in scipy).
constexpr double kTheta3 = 1.495585217958292e-2;
constexpr double kTheta5 = 2.539398330063230e-1;
constexpr double kTheta7 = 9.504178996162932e-1;
constexpr double kTheta9 = 2.097847961257068e0;
constexpr double kTheta13 = 4.25;

// The Padé coefficients b_0 .. b_m (b_j proportional to (2m-j)! / (j! (m-j)!), b_m = 1).
// clang-format off
constexpr double kB3[] = {120., 60., 12., 1.};
constexpr double kB5[] = {30240., 15120., 3360., 420., 30., 1.};
constexpr double kB7[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
constexpr double kB9[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                          2162160.,     110880.,     3960.,       90.,        1.};
constexpr double kB13[] = {64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
                           129060195264000.,   10559470521600.,    670442572800.,    33522128640.,
                           1323241920.,        40840800.,          960960.,          16380.,
                           182.,               1.};
// clang-format on

// log2 |c_(2m+1)|, the leading coefficient of r_m's backward error series:
// (m!)^2 / ((2m)! (2m+1)!).
double log2_err_coeff(int m) {
    double l = 0;
    for (int k = 2; k <= m; ++k) l += 2 * std::log2(double(k));
    for (int k = 2; k <= 2 * m; ++k) l -= std::log2(double(k));
    for (int k = 2; k <= 2 * m + 1; ++k) l -= std::log2(double(k));
    return l;
}

// x[0..count) *= 2^k: one rounding per element, as ldexp, but a multiplication
// by the exact power of two vectorizes (ldexp is a call per element).
template <class T>
void scale_by_pow2(T* x, lint count, int k) {
    double* d = reinterpret_cast<double*>(x);
    const lint nd = count * static_cast<lint>(sizeof(T) / sizeof(double));
    if (k >= -1022 && k <= 1023) {  // 2^k is a normal double
        const double f = std::ldexp(1.0, k);
        for (lint i = 0; i < nd; ++i) d[i] *= f;
    } else {
        for (lint i = 0; i < nd; ++i) d[i] = std::ldexp(d[i], k);
    }
}
inline double scale2(double v, int k) { return std::ldexp(v, k); }
inline cplx scale2(cplx v, int k) { return {std::ldexp(v.real(), k), std::ldexp(v.imag(), k)}; }

// |v|. For complex, max * sqrt(1 + (min/max)^2): no overflow, within a couple
// of ulps of hypot, and branch-free so that it vectorizes (hypot is a call).
// Used for norms that only steer the choice of degree and scaling.
inline double modulus(double v) { return std::fabs(v); }
inline double modulus(cplx v) {
    const double a = std::fabs(v.real()), b = std::fabs(v.imag());
    const double hi = std::max(a, b), lo = std::min(a, b);
    const double r = lo / (hi > 0 ? hi : 1.0);
    return hi * std::sqrt(1.0 + r * r);
}

inline bool nonzero(double v) { return v != 0.0; }
inline bool nonzero(cplx v) { return v.real() != 0.0 || v.imag() != 0.0; }

// ||a||_1 of an n x n column-major matrix: the largest column sum of |a_ij|.
// Eight partial sums per column (a fixed order, so deterministic): one chain
// of dependent additions would cost a latency per element.
template <class T>
double norm1(const T* a, lint n) {
    double best = 0;
    for (lint j = 0; j < n; ++j) {
        const T* col = a + j * n;
        double p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        lint i = 0;
        for (; i + 8 <= n; i += 8)
            for (int q = 0; q < 8; ++q) p[q] += modulus(col[i + q]);
        double s = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
        for (; i < n; ++i) s += modulus(col[i]);
        best = std::max(best, s);
    }
    return best;
}

// dst = sum of coef * src over the terms (+ c0 on the diagonal), or dst += that.
template <class T>
void combine(T* dst, lint n, bool accumulate, double c0, std::initializer_list<std::pair<double, const T*>> terms) {
    const lint nn = n * n;
    for (lint k = 0; k < nn; ++k) {
        T acc = accumulate ? dst[k] : T(0);
        for (const auto& [c, src] : terms) acc += src[k] * c;
        dst[k] = acc;
    }
    if (c0 != 0)
        for (lint d = 0; d < n; ++d) dst[d * n + d] += c0;
}

// ||  |A|^p ||_1, in log2, for p = 1, 2, ... in turn. Works with
// B = 2^-e |A| (||B||_1 <= 1), so that no power overflows: log2 ||B^p|| + p e.
class AbsPowers {
public:
    AbsPowers(const double* b, lint n, int e, double* r, double* t) : b_(b), n_(n), e_(e), r_(r), t_(t) {
        std::fill(r_, r_ + n_, 1.0);  // e^T
    }
    double log2norm(int p) {
        while (p_ < p) {  // r^T <- r^T B, i.e. r <- B^T r
            cblas_dgemv(CblasColMajor, CblasTrans, n_, n_, 1.0, b_, n_, r_, 1, 0.0, t_, 1);
            std::swap(r_, t_);
            ++p_;
        }
        const double m = *std::max_element(r_, r_ + n_);
        return std::log2(m) + double(p_) * e_;  // -inf when B^p underflowed to 0
    }

private:
    const double* b_;
    lint n_;
    int e_;
    double* r_;
    double* t_;
    int p_ = 0;
};

// ||P Q||_1 estimated without forming P Q: LAPACK's 1-norm estimator (lacn2),
// which asks for products with P Q and (P Q)^H, here two gemv each. A lower
// bound, usually within a few percent; deterministic. v, x, y: n each.
template <class T>
double norm1_est_product(lint n, const T* P, const T* Q, T* v, T* x, T* y, lint* isgn) {
    double est = 0;
    lint kase = 0, isave[3] = {0, 0, 0};
    for (;;) {
        lapack::lacn2(n, v, x, isgn, est, kase, isave);
        if (kase == 0) return est;
        if (kase == 1) {  // x <- P Q x
            lapack::gemv('N', n, Q, x, y);
            lapack::gemv('N', n, P, y, x);
        } else {  // x <- Q^H P^H x
            lapack::gemv('C', n, P, x, y);
            lapack::gemv('C', n, Q, y, x);
        }
    }
}

// exp(a_jj) and the off-diagonal factor of a 2x2 triangular block's
// exponential: [[l1, t], [0, l2]] -> t * (e^l2 - e^l1) / (l2 - l1), as
// exp(mean) sinch(d/2) when l1 and l2 are close (the difference quotient would
// cancel), else the difference quotient itself (Al-Mohy & Higham 2009, (2.2)).
template <class T>
T exp_divided_difference(T l1, T l2) {
    const T d = l2 - l1;
    if (std::abs(d) < 1.0) {
        const T h = d * 0.5;
        const T sinch = nonzero(h) ? std::sinh(h) / h : T(1);
        return std::exp(l1 + h) * sinch;
    }
    return (std::exp(l2) - std::exp(l1)) / d;
}

enum Shape { kFull, kUpper, kLower, kDiagonal };

template <class T>
int expm_one(const Input& x, std::int64_t off, lint n, T* o, Scratch& s) {
    if (n == 0) return 0;
    const lint nn = n * n;
    // Eight n x n arrays in one scratch slot, each 16-byte aligned
    // (docs/design/linalg.md §6.4). The last one holds |A| as doubles.
    const std::size_t stride = std::is_same_v<T, cplx> ? static_cast<std::size_t>(nn) : pad2(nn);
    T* base = s.get<T>(8 * stride, 0);
    T* buf[8];
    for (int k = 0; k < 8; ++k) buf[k] = base + k * stride;
    T *A = buf[0], *A2 = buf[1], *A4 = buf[2], *A6 = buf[3];
    if (!pack(x.dtype(), x.data() + off, n, n, x.stride(-2), x.stride(-1), A, n, true)) return kNonFinite;

    // ---- structure
    bool lower = false, upper = false;  // any nonzero strictly below / above the diagonal
    for (lint j = 0; j < n; ++j)
        for (lint i = 0; i < n; ++i)
            if (i != j && nonzero(A[j * n + i])) (i > j ? lower : upper) = true;
    const Shape shape = !lower && !upper ? kDiagonal : (!lower ? kUpper : (!upper ? kLower : kFull));
    if (shape == kDiagonal) {
        std::fill(o, o + nn, T(0));
        for (lint d = 0; d < n; ++d) o[d * n + d] = std::exp(A[d * n + d]);
        return 0;
    }
    // For the triangular fix-ups: the diagonal and the first off-diagonal of A.
    T* lam = s.get<T>(static_cast<std::size_t>(2 * n), 1);
    T* tt = lam + n;
    if (shape != kFull)
        for (lint j = 0; j < n; ++j) {
            lam[j] = A[j * n + j];
            if (j + 1 < n) tt[j] = shape == kUpper ? A[(j + 1) * n + j] : A[j * n + j + 1];
        }

    // ---- guard: ||A|| <= 2^100, so that A^2 .. A^10 cannot overflow
    // exp(A) = exp(2^-s0 A)^(2^s0): s0 squarings up front.
    double nA = norm1(A, n);
    int s0 = 0;
    if (std::isinf(nA)) {
        // Finite entries whose column sum overflows (docs/design/linalg.md §11.4:
        // frexp(inf) gives no exponent, so the guard would be skipped and the
        // squaring count come out near INT_MAX). With 2^k >= 4n every column sum
        // of 2^-k A is below DBL_MAX (a modulus is at most sqrt(2) DBL_MAX).
        int k = 2;
        while ((lint(1) << (k - 2)) < n) ++k;
        s0 = k;
        scale_by_pow2(A, nn, -k);
        nA = norm1(A, n);
    }
    int e;
    std::frexp(nA, &e);  // nA = f 2^e, f in [0.5, 1)
    if (e > 100) {
        s0 += e - 100;
        scale_by_pow2(A, nn, -(e - 100));
        nA = norm1(A, n);
        std::frexp(nA, &e);
    }

    // ---- |A| / 2^e for ell()
    double* absb = reinterpret_cast<double*>(buf[7]);
    if (e >= -1022) {  // 2^-e <= 2^1022, a normal double: one exact product
        const double inv2e = std::ldexp(1.0, -e);
        for (lint k = 0; k < nn; ++k) absb[k] = modulus(A[k]) * inv2e;
    } else {
        // A subnormal norm: 2^-e itself would overflow to inf (|A|/2^e would be
        // inf/NaN and ell() would ask for ~2^31 squarings, docs/design/linalg.md
        // §11.4). Two exact products instead; every result is <= 1, so neither
        // overflows.
        const double f1 = std::ldexp(1.0, 1000), f2 = std::ldexp(1.0, -e - 1000);
        for (lint k = 0; k < nn; ++k) absb[k] = modulus(A[k]) * f1 * f2;
    }
    double* vecs = s.get<double>(static_cast<std::size_t>(2 * pad2(n)), 2);
    AbsPowers absp(absb, n, e, vecs, vecs + pad2(n));
    const double log2nA = std::log2(nA);
    // ell(2^-shift A, m): squarings to add so that the bound is not too optimistic.
    auto ell = [&](int m, int shift) -> int {
        const double la = log2_err_coeff(m) + absp.log2norm(2 * m + 1) - log2nA - 2.0 * m * shift;
        const double v = std::ceil((la + 53.0) / (2.0 * m));  // log2(alpha / u) / 2m, u = 2^-53
        return v > 0 ? static_cast<int>(v) : 0;               // also for la = -inf
    };

    // ---- degree and scaling
    // d_k = ||A^k||^(1/k). A power is formed only when a test needs its exact
    // norm: first each test is tried with upper bounds from the norms already
    // known (||A^(i+j)|| <= ||A^i|| ||A^j||). A bound that passes means the
    // exact value passes too, so the degree is the one the exact test picks
    // (up to rounding right at a threshold); e.g. degree 3 then needs A^2 only.
    // From n = kExpmEstimateMin on, ||A^8|| and ||A^10|| are estimated
    // (lacn2) instead of formed, as the paper and scipy do (chosen after
    // measuring: two gemm were 12-14% of a degree-13 call at n >= 500;
    // docs/design/linalg.md §11.2).
    const bool estimate = n >= kExpmEstimateMin;
    T* A8 = buf[4];
    T *ev = nullptr, *ex = nullptr, *ey = nullptr;
    lint* isgn = nullptr;
    auto est = [&](const T* P, const T* Q) {
        if (!ev) {
            const std::size_t vs = std::is_same_v<T, cplx> ? static_cast<std::size_t>(n) : pad2(n);
            ev = s.get<T>(3 * vs, 3);
            ex = ev + vs;
            ey = ex + vs;
            isgn = s.get_ints(static_cast<std::size_t>(n), 0);
        }
        return norm1_est_product(n, P, Q, ev, ex, ey, isgn);
    };
    lapack::gemm_square(n, A, A, A2);
    const double n2 = norm1(A2, n);
    double n4 = -1, n6 = -1, n8 = -1;  // -1: not known yet
    bool have8 = false;                 // A^8 formed
    auto need4 = [&] {
        if (n4 < 0) {
            lapack::gemm_square(n, A2, A2, A4);
            n4 = norm1(A4, n);
        }
    };
    auto need6 = [&] {
        need4();
        if (n6 < 0) {
            lapack::gemm_square(n, A4, A2, A6);
            n6 = norm1(A6, n);
        }
    };
    auto form8 = [&] {  // r_9 uses A^8
        need4();
        if (!have8) {
            lapack::gemm_square(n, A4, A4, A8);
            have8 = true;
            if (n8 < 0) n8 = norm1(A8, n);
        }
    };
    auto norm8 = [&] {
        need4();
        if (n8 < 0) {
            if (estimate) n8 = est(A4, A4);
            else form8();
        }
    };
    auto root = [](double v, int k) { return std::pow(v, 1.0 / k); };
    // the best bound on d4, d6, d8 known so far, or the value once known
    auto d4 = [&] { return n4 >= 0 ? root(n4, 4) : root(n2, 2); };
    auto d6 = [&] { return n6 >= 0 ? root(n6, 6) : n4 >= 0 ? root(n4 * n2, 6) : root(n2, 2); };
    // (||A^8|| <= ||A^4||^2 gives d8 <= ||A^4||^(1/4); ^(1/2) would be no bound
    // when ||A^4|| < 1, docs/design/linalg.md §11.2)
    auto d8 = [&] { return n8 >= 0 ? root(n8, 8) : n4 >= 0 ? root(n4, 4) : root(n2, 2); };
    // eta <= theta with eta = max(dx(), dy()): on the bounds first, then exact
    auto passes = [&](double theta, auto dx, auto dy, auto form) {
        if (std::max(dx(), dy()) <= theta) return true;
        form();
        return std::max(dx(), dy()) <= theta;
    };
    int m = 0, s1 = 0;
    if (passes(kTheta3, d4, d6, need6) && ell(3, 0) == 0) m = 3;
    else if ((need4(), passes(kTheta5, d4, d6, need6)) && ell(5, 0) == 0) m = 5;  // r_5 uses A^4
    else if ((need6(), passes(kTheta7, d6, d8, norm8)) && ell(7, 0) == 0) m = 7;  // r_7 uses A^6
    else if ((norm8(), std::max(d6(), d8()) <= kTheta9) && ell(9, 0) == 0) {
        m = 9;
        form8();
    } else {
        m = 13;
        double n10;
        if (estimate) {
            n10 = est(A4, A6);
        } else {
            T* A10 = buf[4];  // A^8 is not needed by r_13
            lapack::gemm_square(n, A4, A6, A10);
            n10 = norm1(A10, n);
        }
        const double d10 = root(n10, 10);
        const double eta3 = std::max(d6(), d8());
        const double eta5 = std::min(eta3, std::max(d8(), d10));
        s1 = eta5 <= kTheta13 ? 0 : static_cast<int>(std::ceil(std::log2(eta5 / kTheta13)));
        s1 += ell(13, s1);
        if (s1 > 0) {
            scale_by_pow2(A, nn, -s1);
            scale_by_pow2(A2, nn, -2 * s1);
            scale_by_pow2(A4, nn, -4 * s1);
            scale_by_pow2(A6, nn, -6 * s1);
        }
    }
    const int sq = s0 + s1;  // squarings

    // ---- U and V (Higham 2005's evaluation schemes)
    T *W = buf[4], *P = buf[5], *U = buf[6];  // P: the polynomial being built, then V, then X
    if (m == 13) {
        const double* b = kB13;
        combine<T>(W, n, false, 0, {{b[13], A6}, {b[11], A4}, {b[9], A2}});
        lapack::gemm_square(n, A6, W, P);
        combine<T>(P, n, true, b[1], {{b[7], A6}, {b[5], A4}, {b[3], A2}});
        lapack::gemm_square(n, A, P, U);  // U = A [A6 (b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I]
        combine<T>(W, n, false, 0, {{b[12], A6}, {b[10], A4}, {b[8], A2}});
        lapack::gemm_square(n, A6, W, P);
        combine<T>(P, n, true, b[0], {{b[6], A6}, {b[4], A4}, {b[2], A2}});  // V
    } else {
        // U = A (sum b_(2k+1) A^(2k)), V = sum b_(2k) A^(2k), k = 0 .. (m-1)/2
        const double* b = m == 3 ? kB3 : m == 5 ? kB5 : m == 7 ? kB7 : kB9;
        const T* pw[4] = {A2, A4, A6, A8};
        const int top = (m - 1) / 2;  // highest power A^(2 top)
        std::fill(P, P + nn, T(0));
        for (int k = top; k >= 1; --k) combine<T>(P, n, true, 0, {{b[2 * k + 1], pw[k - 1]}});
        for (lint d = 0; d < n; ++d) P[d * n + d] += b[1];
        lapack::gemm_square(n, A, P, U);
        std::fill(P, P + nn, T(0));
        for (int k = top; k >= 1; --k) combine<T>(P, n, true, 0, {{b[2 * k], pw[k - 1]}});
        for (lint d = 0; d < n; ++d) P[d * n + d] += b[0];
    }
    // r_m = (V - U)^-1 (V + U) = I + 2 (V - U)^-1 U. The second form adds I
    // exactly to a small correction, so small ||A|| lose no last bit on the
    // diagonal (measured, docs/design/linalg.md §11.1; scipy does the same).
    T* Q = buf[4];
    for (lint k = 0; k < nn; ++k) Q[k] = P[k] - U[k];
    lint* ipiv = s.get_ints(static_cast<std::size_t>(n));
    if (lapack::getrf(n, n, Q, n, ipiv) > 0) return kSingular;  // not seen: V - U is well conditioned for these degrees
    lapack::getrs('N', n, n, Q, n, ipiv, U, n);
    for (lint k = 0; k < nn; ++k) P[k] = U[k] * 2.0;
    for (lint d = 0; d < n; ++d) P[d * n + d] += 1.0;

    // ---- squaring, with the triangular fix-ups (Code Fragment 2.1)
    auto fix = [&](T* X, int k) {  // X ~ exp(2^-k A): its exact diagonal and first off-diagonal
        for (lint j = 0; j < n; ++j) X[j * n + j] = std::exp(scale2(lam[j], -k));
        for (lint j = 0; j + 1 < n; ++j) {
            const T v = scale2(tt[j], -k) * exp_divided_difference(scale2(lam[j], -k), scale2(lam[j + 1], -k));
            if (shape == kUpper) X[(j + 1) * n + j] = v;
            else X[j * n + j + 1] = v;
        }
    };
    LapackSlot<T> os(o, nn, s, Scratch::kAlign0);
    T* X = P;
    if (shape != kFull) fix(X, sq);
    T* spare = U;
    for (int k = 1; k <= sq; ++k) {
        T* dst = k == sq ? os.get() : spare;
        lapack::gemm_square(n, X, X, dst);
        spare = X;
        X = dst;
        if (shape != kFull) fix(X, sq - k);
    }
    if (sq == 0) std::copy(P, P + nn, os.get());
    T* r = os.get();
    if (shape == kUpper)  // the other triangle exactly 0 (gemm leaves 0, unless an overflow made 0 * inf)
        for (lint j = 0; j < n; ++j) std::fill(r + j * n + j + 1, r + (j + 1) * n, T(0));
    else if (shape == kLower)
        for (lint j = 1; j < n; ++j) std::fill(r + j * n, r + j * n + j, T(0));
    return 0;
}

nb::object expm(nb::handle xo) {
    Input x(xo, "expm", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "expm");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(b, cost(kExpm, double(n) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            return expm_one<T>(x, off[0], n, res + i * n * n, s);
        });
        if (f) raise(b, f, "expm", "x");
        return to_array(out);
    });
}

}  // namespace

void bind_expm(nb::module_& m) {
    m.def("expm", named("expm", &expm), nb::arg().none());
    // For measurements only: move the estimator threshold (docs/design/linalg.md §7.7).
    m.def("_set_expm_estimate_min", [](std::int64_t n) { kExpmEstimateMin = n; });
}

}  // namespace sd::linalg
