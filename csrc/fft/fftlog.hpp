// FFTLog, the fast Hankel transform of logarithmically spaced samples (docs/design/fft.md §3.9, milestone
// FT5): fht, ifht and fhtoffset with scipy's definitions (Hamilton 2000, as scipy.fft's _fftlog_backend).
//
// The transform of n reals a_j (spacing dln, order mu, offset lnkr, bias q, j_c = (n - 1) / 2):
//
//     v_j = a_j e^(-q (j - j_c) dln),   V = the half spectrum of the real DFT of v,
//     W_m = U_m V_m / n (m = 0 .. n / 2),   w = the inverse real DFT of the Hermitian W (unnormalised),
//     A_j = w_(n-1-j) e^(-q ((j - j_c) dln + lnkr)),
//     U_m = 2^q Gamma(x+ + i y_m) / Gamma(x- - i y_m) e^(2 i y_m (ln 2 - lnkr)),
//     x+- = (mu + 1 +- q) / 2,   y_m = pi m / (n dln),
//
// and ifht divides by conj U_m in place of the product, with the reciprocal biases (e^(q ((j - j_c) dln +
// lnkr)) on the input, e^(q (j - j_c) dln) on the output). U_0 and, for even n, U_(n/2) enter through their
// real parts only (their spectral entries are real), as scipy forces. A pole of Gamma(x+) or of Gamma(x-)
// alone at m = 0 makes U_0 infinite or zero: the transform (fht) or its inverse (ifht) is singular there
// and that coefficient is set to 0, scipy's repair, without a warning (docs/api/fft.md §1.4). Poles of both
// give the finite limit 2^q (-1)^(j+k) k! / j! (x+ = -j, x- = -k).
//
// The coefficients (decision FT58). A coefficient is e^Z with Z = log Gamma(x+ + iy) - log Gamma(x- - iy) +
// q ln 2 + 2 i y (ln 2 - lnkr), and its imaginary part is large: 2 arg Gamma(x + iy) grows like 2 y ln y
// (about 1600 at y = 157, the largest y of n = 4096 and dln = 0.01). A double Z carries an absolute error of
// at least half an ulp of that, which is a relative error of hundreds of u in the coefficient. So Z is
// formed in double-double, reduced modulo 2 pi i there, and only its exponential rounded:
//   log Gamma(w) mod 2 pi i: Re w < 1/2 by reflection, log pi - log sin(pi w) - log Gamma(1 - w); Re w >= 1/2
//   and |w| < 16 by the shift log Gamma(w + N) - log(w (w + 1) .. (w + N - 1)), Re(w + N) >= 16; then
//   Stirling's series with eight terms, whose remainder is below 2^9 |B_18| / (18 17 16^17) < 2^-61 for
//   |w| >= 16 and |arg w| <= pi / 2. The large terms ((w - 1/2) log w - w + log sqrt(2 pi)) are
//   double-double, the series (below 1/192) is double. Only exp(Z) matters, so every branch of a logarithm
//   is as good as another.
//   log z (double-double): L0 + i t0 = log|z| + i arg z from libm, then log z = L0 + i t0 + log(1 + e) with
//   1 + e = z e^(-L0 - i t0) formed with the double-double exponential and sine and cosine below.
//   e^x: reduced by (64 m + i) ln 2 / 64 to |r| <= ln 2 / 128, the table's 2^(i/64) times the Taylor series of
//   r (double-double through r^4, double beyond): relatively within about 2^-96. sin, cos: reduced by j pi / 64
//   to |r| <= pi / 128, the Taylor series (double-double through r^5 and r^6, double beyond) and the addition
//   formulas with the table's (sin, cos)(j pi / 64): within about 2^-100. The tables are made once by the
//   series forms (reductions by ln 2 and 2^10, by pi / 2 and 8, and longer series).
//   log sin(pi w) for |Im w| > 20 is pi |b| - ln 2 + i sgn(b) (pi / 2 - pi f), f = Re w mod 2 (the rest is
//   below 2^-180); else sin(pi f) cosh(pi b) + i cos(pi f) sinh(pi b) and its double-double logarithm.
// The coefficient U_m / n (fht) or 1 / (n conj U_m) (ifht) is e^(+-Re Z - ln n) (cos, sin)(Im Z mod 2 pi),
// each part rounded once: within u/2 of its value plus the double-double errors (2^-60 of |U_m| while the
// arguments stay below 2^36). With q = 0 one log Gamma serves both (Z = 2 i Im log Gamma(x + iy)) and |U_m| = 1
// exactly in the formula.
//
// The biases: e^(s k + c) with k = j - j_c, s = -+q dln and c the offset term formed exactly as
// double-doubles, rounded by exp_round (Tang's method: a table of 2^(i/64) in double-double and a degree-6
// polynomial on |r| <= ln 2 / 128): within u/2 + 2^-59 relatively. With q = 0 there are none (the
// transform skips them).
//
// The plan (the coefficients and biases of one (n, dln, mu, offset, bias, direction)) is kept in a cache of
// its own, as Hamilton's FFTLog keeps its fhti table: an LRU of at most kFhtCachePlans plans and
// kFhtCacheBytes bytes. A plan's values depend on its key alone (each coefficient on its own m), so the
// threads that compute them change no bit.
//
// fhtoffset (scipy's low-ringing offset): lnkr + (arg - round(arg)) dln with arg = (ln 2 - lnkr) / dln +
// (Im log Gamma(x+ + iy) + Im log Gamma(x- + iy)) / pi, y = pi / (2 dln), round half to even: arg in
// double-double (a branch of log Gamma moves arg by an even integer, which the rounding removes exactly),
// the result rounded once.
//
// Error (docs/design/fft.md §4.2): with eps_F the real transforms' contract ((kappa L(n) + c0) u, or the
// Bluestein form of the inner length) and the factors above, ||A_hat - A|| <= (2 eps_F + c_h u) ||b_out||_inf
// max_m |U_m| ||v|| + floor (c_h = 8: the biases' u/2 + 2^-59 and their products' u on both sides, the
// coefficient's u/2, the FMA product's 2u). The proof is the composition of the two real transforms'
// contracts, each a sqrt(n) times an isometry between ||.|| and the Hermitian norm, with the diagonal maps
// between them (fftlog.cpp).
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "fft/ddmath.hpp"

namespace sd::fft {

namespace lg {

using dd::DC;
using dd::DD;

inline constexpr DD kLn2 = {0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56};
inline constexpr DD kPi = {0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53};
inline constexpr DD kHalfPi = {0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54};
inline constexpr DD kTwoPi = {0x1.921fb54442d18p+2, 0x1.1a62633145c07p-52};
inline constexpr DD kLnPi = {0x1.250d048e7a1bdp+0, 0x1.7abf2ad8d5088p-57};
inline constexpr DD kHalfLog2Pi = {0x1.d67f1c864beb5p-1, -0x1.65b5a1b7ff5dfp-55};
inline constexpr double kTwoOverPi = 0x1.45f306dc9c883p-1;
inline constexpr double kInvTwoPi = 0x1.45f306dc9c883p-3;
inline constexpr double kStirlingR = 16.0;

inline DD dd_of(double x) { return {x, 0.0}; }
inline DD ldexp_dd(DD a, int e) { return {std::ldexp(a.hi, e), std::ldexp(a.lo, e)}; }
inline DD abs_dd(DD a) { return a.hi < 0 ? dd::neg(a) : a; }
inline DD add_d(DD a, double b) { return dd::add(a, dd_of(b)); }

// a / b for double-doubles: two quotient digits and a correction, within about 2^-104 relatively.
inline DD ddiv(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    DD r = dd::sub(a, dd::mul(b, q1));
    const double q2 = r.hi / b.hi;
    r = dd::sub(r, dd::mul(b, q2));
    const double q3 = r.hi / b.hi;
    return dd::add(dd::quick_two_sum(q1, q2), dd_of(q3));
}

// The series forms, which build the tables of the fast ones below. e^x: reduced by k ln 2 (ln 2 to 2^-107),
// then by 2^10, the Taylor series to the 9th power and ten squarings of (1 + s): relatively within about
// 2^-97. sin, cos: reduced by pi / 2 and 8, the Taylor series to the 19th power and three doublings: within
// about 2^-100.
inline DD exp_series(DD x) {
    const double k = std::nearbyint(x.hi / kLn2.hi);
    const DD r = ldexp_dd(dd::sub(x, dd::mul(kLn2, k)), -10);  // |r| <= 3.4e-4
    DD s = dd_of(1.0);
    for (int j = 9; j >= 2; --j) s = add_d(dd::div(dd::mul(r, s), static_cast<double>(j)), 1.0);
    s = dd::mul(r, s);  // e^r - 1; the first term left out is below 2^-115 of it
    for (int i = 0; i < 10; ++i) s = dd::add(dd::mul(s, 2.0), dd::mul(s, s));  // (1 + s)^2 - 1
    const int ki = static_cast<int>(k);
    return ldexp_dd(ldexp_dd(add_d(s, 1.0), ki / 2), ki - ki / 2);
}

inline void sincos_series(DD th, DD& s, DD& c) {
    const double j = std::nearbyint(th.hi * kTwoOverPi);
    const DD r = ldexp_dd(dd::sub(th, dd::mul(kHalfPi, j)), -3);  // |r| <= pi / 32 + tiny
    const DD x2 = dd::mul(r, r);
    DD ps = dd_of(1.0), pc = dd_of(1.0);
    for (int k = 9; k >= 1; --k) {  // through r^19 / 19! and r^18 / 18!
        ps = dd::sub(dd_of(1.0), dd::div(dd::mul(x2, ps), static_cast<double>((2 * k) * (2 * k + 1))));
        pc = dd::sub(dd_of(1.0), dd::div(dd::mul(x2, pc), static_cast<double>((2 * k - 1) * (2 * k))));
    }
    DD ss = dd::mul(r, ps), cc = pc;
    for (int i = 0; i < 3; ++i) {
        const DD s2 = dd::mul(dd::mul(ss, cc), 2.0);
        cc = dd::sub(dd::mul(cc, cc), dd::mul(ss, ss));
        ss = s2;
    }
    switch (static_cast<int64_t>(j) & 3) {
        case 0: s = ss, c = cc; break;
        case 1: s = cc, c = dd::neg(ss); break;
        case 2: s = dd::neg(ss), c = dd::neg(cc); break;
        default: s = dd::neg(cc), c = ss; break;
    }
}

// The tables: 2^(i/64), i = 0 .. 63, and (sin, cos)(j pi / 64), j = 0 .. 127, double-double.
struct Tables {
    DD exp2[64];
    DD sin[128], cos[128];
    Tables() {
        for (int i = 0; i < 64; ++i) exp2[i] = exp_series(dd::mul(kLn2, i / 64.0));
        for (int j = 0; j < 128; ++j)
            sincos_series(dd::div(dd::mul(kPi, static_cast<double>(j)), 64.0), sin[j], cos[j]);
    }
};
inline const Tables& tables() {
    static const Tables tab;
    return tab;
}

inline constexpr DD kInv6 = {0x1.5555555555555p-3, 0x1.5555555555555p-57};
inline constexpr DD kInv24 = {0x1.5555555555555p-5, 0x1.5555555555555p-59};
inline constexpr DD kInv120 = {0x1.1111111111111p-7, 0x1.1111111111111p-63};
inline constexpr DD kInv720 = {0x1.6c16c16c16c17p-10, -0x1.f49f49f49f49fp-65};
inline constexpr DD kLn2Over64 = {0x1.62e42fefa39efp-7, 0x1.abc9e3b39803fp-62};
inline constexpr DD kPiOver64 = {0x1.921fb54442d18p-5, 0x1.1a62633145c07p-59};

// e^x (|x| < 2^40), relatively within about 2^-96: x = (64 m + i) ln 2 / 64 + r, |r| <= ln 2 / 128, and
// e^r - 1 = r + r^2 / 2 + r^3 / 6 + r^4 / 24 in double-double plus the terms through r^11 (below 4e-14,
// each next one below 2^-118) in double.
inline DD exp_dd(DD x) {
    if (x.hi > 709.8) return {INFINITY, 0.0};
    if (x.hi < -745.2) return {0.0, 0.0};
    const double K = std::nearbyint(x.hi * (64.0 / kLn2.hi));
    const DD r = dd::sub(x, dd::mul(kLn2Over64, K));
    const double q = r.hi;
    const double tail =
        q * q * q * q * q *
        (1.0 / 120 + q * (1.0 / 720 + q * (1.0 / 5040 + q * (1.0 / 40320 + q * (1.0 / 362880 +
                                                                                q * (1.0 / 3628800 + q / 39916800))))));
    DD t = dd::add(dd::mul(kInv24, r), kInv6);
    t = add_d(dd::mul(t, r), 0.5);
    t = add_d(dd::mul(t, r), 1.0);
    const DD sm1 = add_d(dd::mul(t, r), tail);  // e^r - 1
    const int64_t Ki = static_cast<int64_t>(K);
    const int64_t i = ((Ki % 64) + 64) % 64;
    const int m = static_cast<int>((Ki - i) / 64);
    const DD& b = tables().exp2[i];
    const DD e = dd::add(b, dd::mul(b, sm1));
    return ldexp_dd(ldexp_dd(e, m / 2), m - m / 2);
}

// (sin, cos) of theta (|theta| < 2^42), within about 2^-100: theta = j pi / 64 + r, |r| <= pi / 128, the
// Taylor series of r in double-double through r^5 and r^6 and in double beyond (through r^13 and r^14; the
// terms left in double are below 1e-15, the first left out below 2^-120), and the addition formulas with the
// table.
inline void sincos_dd(DD th, DD& s, DD& c) {
    const double j = std::nearbyint(th.hi * 0x1.45f306dc9c883p+4);  // 64 / pi
    const DD r = dd::sub(th, dd::mul(kPiOver64, j));
    const double q = r.hi, q2 = q * q;
    const double ts =
        q2 * q2 * q2 * q * (-1.0 / 5040 + q2 * (1.0 / 362880 + q2 * (-1.0 / 39916800 + q2 * (1.0 / 6227020800.0))));
    const double tc = q2 * q2 * q2 * q2 *
                      (1.0 / 40320 + q2 * (-1.0 / 3628800 + q2 * (1.0 / 479001600 + q2 * (-1.0 / 87178291200.0))));
    const DD x2 = dd::mul(r, r);
    DD a = dd::sub(dd::mul(kInv120, x2), kInv6);
    a = add_d(dd::mul(a, x2), 1.0);
    const DD sr = add_d(dd::mul(a, r), ts);
    DD b = dd::sub(kInv24, dd::mul(kInv720, x2));
    b = add_d(dd::mul(b, x2), -0.5);
    const DD cr = add_d(add_d(dd::mul(b, x2), tc), 1.0);
    const int64_t jj = ((static_cast<int64_t>(j) % 128) + 128) % 128;
    const Tables& tab = tables();
    const DD& S = tab.sin[jj];
    const DD& C = tab.cos[jj];
    s = dd::add(dd::mul(S, cr), dd::mul(C, sr));
    c = dd::sub(dd::mul(C, cr), dd::mul(S, sr));
}

// theta reduced modulo 2 pi into about [-pi, pi] (|theta| < 2^42).
inline DD reduce_2pi(DD th) {
    const double k = std::nearbyint(th.hi * kInvTwoPi);
    return dd::sub(th, dd::mul(kTwoPi, k));
}

// x - round(x) in [-1/2, 1/2], rounding half to even (x a normalised double-double, |x| < 2^52). A tie has
// x.lo = 0 (x.hi = fl(x)), where nearbyint already picks the even neighbour.
inline DD frac_even(DD x) {
    const double r = std::nearbyint(x.hi);
    const DD f = dd::add(dd_of(x.hi - r), dd_of(x.lo));  // x.hi - r is exact
    if (f.hi > 0.5 || (f.hi == 0.5 && f.lo > 0.0)) return add_d(f, -1.0);
    if (f.hi < -0.5 || (f.hi == -0.5 && f.lo < 0.0)) return add_d(f, 1.0);
    return f;
}

// log z for z != 0 (any branch is as good: only e^log z is used), within about 2^-97 absolutely.
inline DC clog(DC z) {
    const double L0 = std::log(std::hypot(z.re.hi, z.im.hi));
    const double t0 = std::atan2(z.im.hi, z.re.hi);
    const DD E = exp_dd(dd_of(-L0));
    DD s, c;
    sincos_dd(dd_of(t0), s, c);
    const DC q = dd::cmul(z, DC{dd::mul(E, c), dd::neg(dd::mul(E, s))});  // z e^(-L0 - i t0) = 1 + e
    const DC e = {add_d(q.re, -1.0), q.im};
    const DC e2 = dd::cmul(e, e);  // log(1 + e) = e - e^2 / 2 + O(e^3), |e| < 2^-45
    return {
        dd::add(dd_of(L0), dd::sub(e.re, dd::mul(e2.re, 0.5))), dd::add(dd_of(t0), dd::sub(e.im, dd::mul(e2.im, 0.5)))
    };
}

// Stirling's series for |w| >= 16, Re w > 0: (w - 1/2) log w - w + log sqrt(2 pi) + sum_(k=1)^8 c_k / w^(2k-1).
inline DC stirling(DC w) {
    const DC lw = clog(w);
    const DC m = dd::cmul(DC{add_d(w.re, -0.5), w.im}, lw);
    DC r = {dd::add(dd::sub(m.re, w.re), kHalfLog2Pi), dd::sub(m.im, w.im)};
    constexpr double c[8] = {1.0 / 12.0,   -1.0 / 360.0,      1.0 / 1260.0, -1.0 / 1680.0,
                             1.0 / 1188.0, -691.0 / 360360.0, 1.0 / 156.0,  -3617.0 / 122400.0};  // fmt: skip
    const double x = w.re.hi, y = w.im.hi, d = x * x + y * y;
    const double zr = x / d, zi = -y / d;  // 1 / w
    const double qr = zr * zr - zi * zi, qi = 2.0 * zr * zi;  // 1 / w^2
    double pr = c[7], pi = 0.0;
    for (int k = 6; k >= 0; --k) {
        const double tr = pr * qr - pi * qi + c[k];
        pi = pr * qi + pi * qr;
        pr = tr;
    }
    const double sr = pr * zr - pi * zi, si = pr * zi + pi * zr;  // below 1/192 in size
    r.re = add_d(r.re, sr);
    r.im = add_d(r.im, si);
    return r;
}

// log Gamma(w) mod 2 pi i for Re w >= 1/2 (about).
inline DC lngamma_right(DC w) {
    const double a = w.re.hi, b = w.im.hi;
    if (a * a + b * b >= kStirlingR * kStirlingR) return stirling(w);
    const int N = static_cast<int>(std::ceil(kStirlingR - a));
    DC p = w, v = w;
    for (int k = 1; k < N; ++k) {
        v.re = add_d(v.re, 1.0);
        p = dd::cmul(p, v);
    }
    v.re = add_d(v.re, 1.0);  // w + N
    return dd::csub(stirling(v), clog(p));
}

// log sin(pi w) mod 2 pi i for w off the poles.
inline DC lnsinpi(DC w) {
    const double r2 = 2.0 * std::nearbyint(w.re.hi * 0.5);
    const DD f = dd::add(dd_of(w.re.hi - r2), dd_of(w.re.lo));  // Re w mod 2, exactly
    if (std::fabs(w.im.hi) > 20.0) {
        DD im = dd::sub(kHalfPi, dd::mul(kPi, f));
        if (w.im.hi < 0) im = dd::neg(im);
        return {dd::sub(dd::mul(kPi, abs_dd(w.im)), kLn2), im};
    }
    DD s, c;
    sincos_dd(dd::mul(kPi, f), s, c);
    const DD pb = dd::mul(kPi, w.im);
    const DD E = exp_dd(pb);
    const DD Ei = ddiv(dd_of(1.0), E);
    const DD ch = dd::mul(dd::add(E, Ei), 0.5);
    DD sh;
    if (std::fabs(pb.hi) < 1.0) {  // sinh x = x (1 + x^2 / 3! (1 + x^2 / (4 5) (...))), through x^29 / 29!
        const DD x2 = dd::mul(pb, pb);
        DD p = dd_of(1.0);
        for (int k = 14; k >= 1; --k)
            p = add_d(dd::div(dd::mul(x2, p), static_cast<double>((2 * k) * (2 * k + 1))), 1.0);
        sh = dd::mul(pb, p);
    } else {
        sh = dd::mul(dd::sub(E, Ei), 0.5);
    }
    return clog(DC{dd::mul(s, ch), dd::mul(c, sh)});
}

// log Gamma(w) mod 2 pi i for w off the poles (|w| < 2^36).
inline DC lngamma(DC w) {
    if (w.re.hi >= 0.5) return lngamma_right(w);
    const DC r = lngamma_right(DC{dd::sub(dd_of(1.0), w.re), dd::neg(w.im)});
    const DC ls = lnsinpi(w);
    return {dd::sub(dd::sub(kLnPi, ls.re), r.re), dd::sub(dd::neg(ls.im), r.im)};
}

// e^x rounded to a double, relatively within u/2 + 2^-59 (Tang's method: 2^(i/64) from the table and a
// degree-6 polynomial in double on |r| <= ln 2 / 128).
inline double exp_round(DD x) {
    if (x.hi > 709.8) return INFINITY;
    if (x.hi < -745.2) return 0.0;
    const double K = std::nearbyint(x.hi * (64.0 / kLn2.hi));
    const DD r = dd::sub(x, dd::mul(kLn2Over64, K));  // |r| <= ln 2 / 128 + tiny
    const double q = r.hi;
    const double p = q + q * q * (0.5 + q * (1.0 / 6 + q * (1.0 / 24 + q * (1.0 / 120 + q * (1.0 / 720))))) + r.lo;
    const int64_t Ki = static_cast<int64_t>(K);
    const int64_t i = ((Ki % 64) + 64) % 64;
    const int m = static_cast<int>((Ki - i) / 64);
    const DD& t = tables().exp2[i];
    const double v = t.hi + std::fma(t.hi, p, t.lo + t.lo * p);
    return std::ldexp(std::ldexp(v, m / 2), m - m / 2);
}

}  // namespace lg

// The key of an FFTLog plan: the length, the parameters (compared bit for bit) and the direction.
struct FhtKey {
    int64_t n = 0;
    double dln = 0.0, mu = 0.0, offset = 0.0, bias = 0.0;
    bool inverse = false;
};

// The coefficients and biases of one key. coef: n / 2 + 1 interleaved complex, U_m / n (fht) or
// 1 / (n conj U_m) (ifht), and at m = 0 and (even n) m = n / 2 the reals Re U_m / n and 1 / (n Re U_m) (scipy
// drops Im U_m there before either); bin, bout: the input and output biases (n each; empty when the bias is
// 0). singular: a coefficient was set to 0 for a pole.
class FhtPlan {
public:
    explicit FhtPlan(const FhtKey& key);
    const FhtKey& key() const { return key_; }
    const double* coef() const { return coef_.data(); }
    const double* bin() const { return bin_.empty() ? nullptr : bin_.data(); }
    const double* bout() const { return bout_.empty() ? nullptr : bout_.data(); }
    bool singular() const { return singular_; }
    int64_t bytes() const { return static_cast<int64_t>((coef_.size() + bin_.size() + bout_.size()) * sizeof(double)); }

private:
    FhtKey key_;
    std::vector<double> coef_, bin_, bout_;
    bool singular_ = false;
};

inline constexpr int kFhtCachePlans = 16;
inline constexpr int64_t kFhtCacheBytes = int64_t{128} << 20;

// The plan of key, from its cache or made now (holding the GIL on entry; released while a plan is made).
std::shared_ptr<const FhtPlan> get_fht_plan(const FhtKey& key);

// scipy's fhtoffset (see above).
double fht_offset(double dln, double mu, double initial, double bias);

}  // namespace sd::fft
