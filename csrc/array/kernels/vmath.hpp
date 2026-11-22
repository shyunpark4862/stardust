// float64 exp, expm1, log1p and cos in NEON lanes (docs/design/array.md §5.4), each with a
// scalar twin of the same IEEE operations in the same order, so an element's value never
// depends on its position or the call's length (f(x)[i] == f(x[i]), any split, any thread).
// Arguments the lanes do not cover (overflow and underflow ranges, NaN, the domain's edges,
// subnormals of log) take libm in that lane. Special values are C99 Annex F's, -0 included.
//
// exp: x = (k + j/128) log 2 + r, |r| <= log 2 / 256; e^x = 2^k 2^(j/128) (1 + p(r)) with
// 2^(j/128) a double and its relative tail (the table, Arb), p the degree-5 Taylor polynomial
// (truncation r^6/720 < 2^-60): ARM optimized-routines' exp, also stats/vmath.hpp's.
// expm1: the same reduction, e^x - 1 = (s - 1) + s t, s = 2^(k + j/128), t = tail + p(r):
// s - 1 is exact where it matters (s in [1/2, 2], Sterbenz) and fma rounds the sum once, so
// the result keeps its relative accuracy near 0 (|x| < 2^-54 returns x).
// log (the core of log1p; sd.log itself stays on vvlog, as fast as this one):
// x = 2^k z, z in [0.684, 1.371) cut into 128 intervals of the bit pattern, the one that
// holds 1 centred on it; c = 1/invc near the interval's centre (invc a double, c = 1 for the
// middle one), log x = k log 2 + log c + log1p(r), r = z invc - 1 (one fma, |r| < 2^-8),
// log c a double-double (Arb), log1p(r) = r + r^2 q(r), q Taylor to r^5 (truncation
// r^8/8). k log 2 + log c is exact (both multiples of 2^-42) and + r error-free (Fast2Sum).
// log1p: u = 1 + x and its rounding error e (TwoSum) enter as r = z invc - 1 + e 2^-k invc.
// cos: x = n pi/2 + r, r a double-double (Cody-Waite with pi/2 in four parts, the first
// two of 33 bits, so the steps that cancel are exact) for |x| <= 2^20, libm beyond; then
// FreeBSD's __kernel_sin and __kernel_cos (minimax to 2^-58) at r, both, picked and signed by
// n mod 4. sin itself stays on vvsin (2.5x faster than these lanes, 3 ULP), tan on libm: a
// quotient of these lanes' sin and cos was 2 ULP and 1.4x faster than libm (3 ULP), made
// 1 ULP (double-double quotient) no faster.
// Measured against Arb (random arguments over the whole range, dense near 0, 1 and the
// multiples of pi/2, 1-2 x 10^5 each), at most / share not correctly rounded: exp 1 ULP /
// 0.07% (vvexp 1 / 20%), expm1 1 / 1.9% (libm 1 / 1.1%), log 1 / 1.1% (vvlog 2 / 37%), log1p
// 1 / 0.05% (libm 1 / 0.05%), cos 1 / 2.2% (libm 1 / 2.9%).
// ns per element, one thread (vForce or libm): exp 0.71 (1.10), expm1 1.1 (3.7), log1p 1.3
// (2.5), cos 1.7-1.9 (3.6-5.5).
#pragma once

#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace sd::vmath {

inline double as_double(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}
inline std::uint64_t as_bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

// ---- exp and expm1 ---------------------------------------------------------

inline constexpr double kExpLn2HiN = 0x1.62e42fefa39efp-8;
inline constexpr double kExpLn2LoN = 0x1.abc9e3b39803fp-63;
inline constexpr double kExpInvLn2N = 0x1.71547652b82fep+7;
inline constexpr double kExpShift = 0x1.8p52;
inline constexpr double kExpMax = 708.0;  // e^+-708 and 2^(k + j/128) stay normal
// 2^(j/128) = s (1 + tail): {bits of tail, bits of s - (j << 45)} (Arb, rounded to nearest)
// clang-format off
alignas(64) inline constexpr std::uint64_t kExpTable[2 * 128] = {
    0x0000000000000000ULL, 0x3ff0000000000000ULL, 0x3c9b3b4f1a88bf6eULL, 0x3feff63da9fb3335ULL,
    0xbc7160139cd8dc5dULL, 0x3fefec9a3e778061ULL, 0xbc905e7a108766d1ULL, 0x3fefe315e86e7f85ULL,
    0x3c8cd2523567f613ULL, 0x3fefd9b0d3158574ULL, 0xbc8bce8023f98efaULL, 0x3fefd06b29ddf6deULL,
    0x3c60f74e61e6c861ULL, 0x3fefc74518759bc8ULL, 0x3c90a3e45b33d399ULL, 0x3fefbe3ecac6f383ULL,
    0x3c979aa65d837b6dULL, 0x3fefb5586cf9890fULL, 0x3c8eb51a92fdeffcULL, 0x3fefac922b7247f7ULL,
    0x3c3ebe3d702f9cd1ULL, 0x3fefa3ec32d3d1a2ULL, 0xbc6a033489906e0bULL, 0x3fef9b66affed31bULL,
    0xbc9556522a2fbd0eULL, 0x3fef9301d0125b51ULL, 0xbc5080ef8c4eea55ULL, 0x3fef8abdc06c31ccULL,
    0xbc91c923b9d5f416ULL, 0x3fef829aaea92de0ULL, 0x3c80d3e3e95c55afULL, 0x3fef7a98c8a58e51ULL,
    0xbc801b15eaa59348ULL, 0x3fef72b83c7d517bULL, 0xbc8f1ff055de323dULL, 0x3fef6af9388c8deaULL,
    0x3c8b898c3f1353bfULL, 0x3fef635beb6fcb75ULL, 0xbc96d99c7611eb26ULL, 0x3fef5be084045cd4ULL,
    0x3c9aecf73e3a2f60ULL, 0x3fef54873168b9aaULL, 0xbc8fe782cb86389dULL, 0x3fef4d5022fcd91dULL,
    0x3c8a6f4144a6c38dULL, 0x3fef463b88628cd6ULL, 0x3c807a05b0e4047dULL, 0x3fef3f49917ddc96ULL,
    0x3c968efde3a8a894ULL, 0x3fef387a6e756238ULL, 0x3c875e18f274487dULL, 0x3fef31ce4fb2a63fULL,
    0x3c80472b981fe7f2ULL, 0x3fef2b4565e27cddULL, 0xbc96b87b3f71085eULL, 0x3fef24dfe1f56381ULL,
    0x3c82f7e16d09ab31ULL, 0x3fef1e9df51fdee1ULL, 0xbc3d219b1a6fbffaULL, 0x3fef187fd0dad990ULL,
    0x3c8b3782720c0ab4ULL, 0x3fef1285a6e4030bULL, 0x3c6e149289cecb8fULL, 0x3fef0cafa93e2f56ULL,
    0x3c834d754db0abb6ULL, 0x3fef06fe0a31b715ULL, 0x3c864201e2ac744cULL, 0x3fef0170fc4cd831ULL,
    0x3c8fdd395dd3f84aULL, 0x3feefc08b26416ffULL, 0xbc86a3803b8e5b04ULL, 0x3feef6c55f929ff1ULL,
    0xbc924aedcc4b5068ULL, 0x3feef1a7373aa9cbULL, 0xbc9907f81b512d8eULL, 0x3feeecae6d05d866ULL,
    0xbc71d1e83e9436d2ULL, 0x3feee7db34e59ff7ULL, 0xbc991919b3ce1b15ULL, 0x3feee32dc313a8e5ULL,
    0x3c859f48a72a4c6dULL, 0x3feedea64c123422ULL, 0xbc9312607a28698aULL, 0x3feeda4504ac801cULL,
    0xbc58a78f4817895bULL, 0x3feed60a21f72e2aULL, 0xbc7c2c9b67499a1bULL, 0x3feed1f5d950a897ULL,
    0x3c4363ed60c2ac11ULL, 0x3feece086061892dULL, 0x3c9666093b0664efULL, 0x3feeca41ed1d0057ULL,
    0x3c6ecce1daa10379ULL, 0x3feec6a2b5c13cd0ULL, 0x3c93ff8e3f0f1230ULL, 0x3feec32af0d7d3deULL,
    0x3c7690cebb7aafb0ULL, 0x3feebfdad5362a27ULL, 0x3c931dbdeb54e077ULL, 0x3feebcb299fddd0dULL,
    0xbc8f94340071a38eULL, 0x3feeb9b2769d2ca7ULL, 0xbc87deccdc93a349ULL, 0x3feeb6daa2cf6642ULL,
    0xbc78dec6bd0f385fULL, 0x3feeb42b569d4f82ULL, 0xbc861246ec7b5cf6ULL, 0x3feeb1a4ca5d920fULL,
    0x3c93350518fdd78eULL, 0x3feeaf4736b527daULL, 0x3c7b98b72f8a9b05ULL, 0x3feead12d497c7fdULL,
    0x3c9063e1e21c5409ULL, 0x3feeab07dd485429ULL, 0x3c34c7855019c6eaULL, 0x3feea9268a5946b7ULL,
    0x3c9432e62b64c035ULL, 0x3feea76f15ad2148ULL, 0xbc8ce44a6199769fULL, 0x3feea5e1b976dc09ULL,
    0xbc8c33c53bef4da8ULL, 0x3feea47eb03a5585ULL, 0xbc845378892be9aeULL, 0x3feea34634ccc320ULL,
    0xbc93cedd78565858ULL, 0x3feea23882552225ULL, 0x3c5710aa807e1964ULL, 0x3feea155d44ca973ULL,
    0xbc93b3efbf5e2228ULL, 0x3feea09e667f3bcdULL, 0xbc6a12ad8734b982ULL, 0x3feea012750bdabfULL,
    0xbc6367efb86da9eeULL, 0x3fee9fb23c651a2fULL, 0xbc80dc3d54e08851ULL, 0x3fee9f7df9519484ULL,
    0xbc781f647e5a3ecfULL, 0x3fee9f75e8ec5f74ULL, 0xbc86ee4ac08b7db0ULL, 0x3fee9f9a48a58174ULL,
    0xbc8619321e55e68aULL, 0x3fee9feb564267c9ULL, 0x3c909ccb5e09d4d3ULL, 0x3feea0694fde5d3fULL,
    0xbc7b32dcb94da51dULL, 0x3feea11473eb0187ULL, 0x3c94ecfd5467c06bULL, 0x3feea1ed0130c132ULL,
    0x3c65ebe1abd66c55ULL, 0x3feea2f336cf4e62ULL, 0xbc88a1c52fb3cf42ULL, 0x3feea427543e1a12ULL,
    0xbc9369b6f13b3734ULL, 0x3feea589994cce13ULL, 0xbc805e843a19ff1eULL, 0x3feea71a4623c7adULL,
    0xbc94d450d872576eULL, 0x3feea8d99b4492edULL, 0x3c90ad675b0e8a00ULL, 0x3feeaac7d98a6699ULL,
    0x3c8db72fc1f0eab4ULL, 0x3feeace5422aa0dbULL, 0xbc65b6609cc5e7ffULL, 0x3feeaf3216b5448cULL,
    0x3c7bf68359f35f44ULL, 0x3feeb1ae99157736ULL, 0xbc93091fa71e3d83ULL, 0x3feeb45b0b91ffc6ULL,
    0xbc5da9b88b6c1e29ULL, 0x3feeb737b0cdc5e5ULL, 0xbc6c23f97c90b959ULL, 0x3feeba44cbc8520fULL,
    0xbc92434322f4f9aaULL, 0x3feebd829fde4e50ULL, 0xbc85ca6cd7668e4bULL, 0x3feec0f170ca07baULL,
    0x3c71affc2b91ce27ULL, 0x3feec49182a3f090ULL, 0x3c6dd235e10a73bbULL, 0x3feec86319e32323ULL,
    0xbc87c50422622263ULL, 0x3feecc667b5de565ULL, 0x3c8b1c86e3e231d5ULL, 0x3feed09bec4a2d33ULL,
    0xbc91bbd1d3bcbb15ULL, 0x3feed503b23e255dULL, 0x3c90cc319cee31d2ULL, 0x3feed99e1330b358ULL,
    0x3c8469846e735ab3ULL, 0x3feede6b5579fdbfULL, 0xbc82dfcd978e9db4ULL, 0x3feee36bbfd3f37aULL,
    0x3c8c1a7792cb3387ULL, 0x3feee89f995ad3adULL, 0xbc907b8f4ad1d9faULL, 0x3feeee07298db666ULL,
    0xbc55c3d956dcaebaULL, 0x3feef3a2b84f15fbULL, 0xbc90a40e3da6f640ULL, 0x3feef9728de5593aULL,
    0xbc68d6f438ad9334ULL, 0x3feeff76f2fb5e47ULL, 0xbc91eee26b588a35ULL, 0x3fef05b030a1064aULL,
    0x3c74ffd70a5fddcdULL, 0x3fef0c1e904bc1d2ULL, 0xbc91bdfbfa9298acULL, 0x3fef12c25bd71e09ULL,
    0x3c736eae30af0cb3ULL, 0x3fef199bdd85529cULL, 0x3c8ee3325c9ffd94ULL, 0x3fef20ab5fffd07aULL,
    0x3c84e08fd10959acULL, 0x3fef27f12e57d14bULL, 0x3c63cdaf384e1a67ULL, 0x3fef2f6d9406e7b5ULL,
    0x3c676b2c6c921968ULL, 0x3fef3720dcef9069ULL, 0xbc808a1883ccb5d2ULL, 0x3fef3f0b555dc3faULL,
    0xbc8fad5d3ffffa6fULL, 0x3fef472d4a07897cULL, 0xbc900dae3875a949ULL, 0x3fef4f87080d89f2ULL,
    0x3c74a385a63d07a7ULL, 0x3fef5818dcfba487ULL, 0xbc82919e2040220fULL, 0x3fef60e316c98398ULL,
    0x3c8e5a50d5c192acULL, 0x3fef69e603db3285ULL, 0x3c843a59ac016b4bULL, 0x3fef7321f301b460ULL,
    0xbc82d52107b43e1fULL, 0x3fef7c97337b9b5fULL, 0xbc892ab93b470dc9ULL, 0x3fef864614f5a129ULL,
    0x3c74b604603a88d3ULL, 0x3fef902ee78b3ff6ULL, 0x3c83c5ec519d7271ULL, 0x3fef9a51fbc74c83ULL,
    0xbc8ff7128fd391f0ULL, 0x3fefa4afa2a490daULL, 0xbc8dae98e223747dULL, 0x3fefaf482d8e67f1ULL,
    0x3c8ec3bc41aa2008ULL, 0x3fefba1bee615a27ULL, 0x3c842b94c3a9eb32ULL, 0x3fefc52b376bba97ULL,
    0x3c8a64a931d185eeULL, 0x3fefd0765b6e4540ULL, 0xbc8e37bae43be3edULL, 0x3fefdbfdad9cbe14ULL,
    0x3c77893b4d91cd9dULL, 0x3fefe7c1819e90d8ULL, 0x3c5305c14160cc89ULL, 0x3feff3c22b8f71f1ULL,
};
// clang-format on

struct ExpParts {
    double scale, t;  // e^x = scale (1 + t)
};

inline ExpParts exp_parts(double x) {
    const double z = x * kExpInvLn2N + kExpShift;
    const double kd = z - kExpShift;
    const std::uint64_t ki = as_bits(z);
    double r = std::fma(kd, -kExpLn2HiN, x);
    r = std::fma(kd, -kExpLn2LoN, r);
    const std::uint64_t j = ki & 127;
    const double tail = as_double(kExpTable[2 * j]);
    const double scale = as_double(kExpTable[2 * j + 1] + (ki << 45));
    const double r2 = r * r;
    const double p23 = std::fma(r, 1.0 / 6, 0.5);
    const double p45 = std::fma(r, 1.0 / 120, 1.0 / 24);
    double t = std::fma(r2, p23, tail + r);
    t = std::fma(r2 * r2, p45, t);
    return {scale, t};
}

struct ExpPartsV {
    float64x2_t scale, t;
};

inline ExpPartsV exp_parts(float64x2_t x) {
    const float64x2_t z = vaddq_f64(vmulq_f64(x, vdupq_n_f64(kExpInvLn2N)), vdupq_n_f64(kExpShift));
    const float64x2_t kd = vsubq_f64(z, vdupq_n_f64(kExpShift));
    const uint64x2_t ki = vreinterpretq_u64_f64(z);
    float64x2_t r = vfmaq_f64(x, kd, vdupq_n_f64(-kExpLn2HiN));
    r = vfmaq_f64(r, kd, vdupq_n_f64(-kExpLn2LoN));
    const std::uint64_t j0 = vgetq_lane_u64(ki, 0) & 127, j1 = vgetq_lane_u64(ki, 1) & 127;
    const uint64x2_t e0 = vld1q_u64(&kExpTable[2 * j0]), e1 = vld1q_u64(&kExpTable[2 * j1]);
    const float64x2_t tail = vreinterpretq_f64_u64(vzip1q_u64(e0, e1));
    const float64x2_t scale = vreinterpretq_f64_u64(vaddq_u64(vzip2q_u64(e0, e1), vshlq_n_u64(ki, 45)));
    const float64x2_t r2 = vmulq_f64(r, r);
    const float64x2_t p23 = vfmaq_f64(vdupq_n_f64(0.5), r, vdupq_n_f64(1.0 / 6));
    const float64x2_t p45 = vfmaq_f64(vdupq_n_f64(1.0 / 24), r, vdupq_n_f64(1.0 / 120));
    float64x2_t t = vfmaq_f64(vaddq_f64(tail, r), r2, p23);
    t = vfmaq_f64(t, vmulq_f64(r2, r2), p45);
    return {scale, t};
}

inline double exp1(double x) {
    if (!(std::fabs(x) <= kExpMax)) return std::exp(x);
    const ExpParts e = exp_parts(x);
    return std::fma(e.scale, e.t, e.scale);
}

inline float64x2_t exp_lanes(float64x2_t x) {
    const ExpPartsV e = exp_parts(x);
    return vfmaq_f64(e.scale, e.scale, e.t);
}

inline constexpr double kTiny = 0x1p-54;  // below it expm1(x) and log1p(x) round to x

// Below it expm1 is its Taylor polynomial to x^8 (truncation x^9/9! < 2^-58 x), summed onto x
// by one fma: near 0 the table form's cancellation between s - 1 and s t cost up to 2 ULP.
inline constexpr double kExpm1Poly = 0x1p-5;
// Below it s < 1/2, s - 1 is no longer exact, and e^x - 1 (Sterbenz-free but in [-1, -1/2])
// is closer: at most 0.75 ULP against 1.
inline constexpr double kExpm1Neg = -0x1.62e42fefa39efp-1;  // -log 2

inline double expm1_poly(double x) {
    const double x2 = x * x;
    const double a = std::fma(x, 1.0 / 6, 0.5);
    const double b = std::fma(x, 1.0 / 120, 1.0 / 24);
    const double c = std::fma(x, 1.0 / 5040, 1.0 / 720);
    const double p = std::fma(x2 * x2, std::fma(x2, 1.0 / 40320, c), std::fma(x2, b, a));
    return std::fma(x2, p, x);
}
inline float64x2_t expm1_poly(float64x2_t x) {
    const float64x2_t x2 = vmulq_f64(x, x);
    const float64x2_t a = vfmaq_f64(vdupq_n_f64(0.5), x, vdupq_n_f64(1.0 / 6));
    const float64x2_t b = vfmaq_f64(vdupq_n_f64(1.0 / 24), x, vdupq_n_f64(1.0 / 120));
    const float64x2_t c = vfmaq_f64(vdupq_n_f64(1.0 / 720), x, vdupq_n_f64(1.0 / 5040));
    const float64x2_t p = vfmaq_f64(vfmaq_f64(a, x2, b), vmulq_f64(x2, x2), vfmaq_f64(c, x2, vdupq_n_f64(1.0 / 40320)));
    return vfmaq_f64(x, x2, p);
}

inline double expm1_1(double x) {
    if (!(std::fabs(x) <= kExpMax)) return std::expm1(x);
    if (std::fabs(x) < kTiny) return x;
    if (std::fabs(x) < kExpm1Poly) return expm1_poly(x);
    const ExpParts e = exp_parts(x);
    if (x < kExpm1Neg) return std::fma(e.scale, e.t, e.scale) - 1.0;
    return std::fma(e.scale, e.t, e.scale - 1.0);
}

inline float64x2_t expm1_lanes(float64x2_t x) {
    const ExpPartsV e = exp_parts(x);
    float64x2_t y = vfmaq_f64(vsubq_f64(e.scale, vdupq_n_f64(1.0)), e.scale, e.t);
    const uint64x2_t neg = vcltq_f64(x, vdupq_n_f64(kExpm1Neg));
    y = vbslq_f64(neg, vsubq_f64(vfmaq_f64(e.scale, e.scale, e.t), vdupq_n_f64(1.0)), y);
    const uint64x2_t small = vcaltq_f64(x, vdupq_n_f64(kExpm1Poly));
    if ((vgetq_lane_u64(small, 0) | vgetq_lane_u64(small, 1)) != 0) y = vbslq_f64(small, expm1_poly(x), y);
    return vbslq_f64(vcaltq_f64(x, vdupq_n_f64(kTiny)), x, y);
}

// ---- log and log1p ---------------------------------------------------------

// t = bits(x) - kLogOff: bits 45-51 the interval, bits 52-63 the exponent k
inline constexpr std::uint64_t kLogOff = 0x3fe5f00000000000ULL;
inline constexpr double kLn2Hi = 0x1.62e42fefa3800p-1;  // k * kLn2Hi exact for |k| < 2^11
inline constexpr double kLn2Lo = 0x1.ef35793c76730p-45;
struct LogEntry {
    std::uint64_t invc, logc_hi, logc_lo, pad;
};
// {1/c, log c as a double-double} per interval (Arb, rounded to nearest)
alignas(64) inline constexpr LogEntry kLogTable[128] = {
    {0x3ff745d1745d1746ULL, 0xbfd7fafa3bd81000ULL, 0xbd3471b79bf6d4cbULL},
    {0x3ff724287f46debcULL, 0xbfd79e26687d0000ULL, 0x3d330a0168817444ULL},
    {0x3ff702e05c0b8170ULL, 0xbfd741d876c68000ULL, 0x3d313c7b5b11cfa7ULL},
    {0x3ff6e1f76b4337c7ULL, 0xbfd6e60ee6af2000ULL, 0x3d3a3556a0f7749eULL},
    {0x3ff6c16c16c16c17ULL, 0xbfd68ac83e9c7000ULL, 0x3d37acd66c548a30ULL},
    {0x3ff6a13cd1537290ULL, 0xbfd630030b3ab000ULL, 0x3d2dbc23e731ae00ULL},
    {0x3ff6816816816817ULL, 0xbfd5d5bddf596000ULL, 0x3d09de2a08a465dcULL},
    {0x3ff661ec6a5122f9ULL, 0xbfd57bf753c8d000ULL, 0xbd1fadadee5d40efULL},
    {0x3ff642c8590b2164ULL, 0xbfd522ae0738a000ULL, 0xbd2eba708164c759ULL},
    {0x3ff623fa77016240ULL, 0xbfd4c9e09e173000ULL, 0x3d2e18891b0ad8a4ULL},
    {0x3ff6058160581606ULL, 0xbfd4718dc271c000ULL, 0xbd3071d8fb4c14c5ULL},
    {0x3ff5e75bb8d015e7ULL, 0xbfd419b423d5f000ULL, 0x3d3ce7a9226de3ecULL},
    {0x3ff5c9882b931057ULL, 0xbfd3c25277333000ULL, 0xbd183454b606bd5cULL},
    {0x3ff5ac056b015ac0ULL, 0xbfd36b6776be1000ULL, 0xbd115ecdb0f177c8ULL},
    {0x3ff58ed2308158edULL, 0xbfd314f1e1d36000ULL, 0x3d28e5bad3213cb8ULL},
    {0x3ff571ed3c506b3aULL, 0xbfd2bef07cdc9000ULL, 0xbd2aa5ba4a5004f4ULL},
    {0x3ff5555555555555ULL, 0xbfd269621134e000ULL, 0x3d31ba1f10522625ULL},
    {0x3ff5390948f40febULL, 0xbfd214456d0ec000ULL, 0x3d3cac5428b728a3ULL},
    {0x3ff51d07eae2f815ULL, 0xbfd1bf99635a7000ULL, 0x3d31ade9575c2125ULL},
    {0x3ff5015015015015ULL, 0xbfd16b5ccbad0000ULL, 0x3d3232a9042d74bfULL},
    {0x3ff4e5e0a72f0539ULL, 0xbfd1178e8227e000ULL, 0xbd31e9b8ce2d07f2ULL},
    {0x3ff4cab88725af6eULL, 0xbfd0c42d67616000ULL, 0xbd270d4b163ceae9ULL},
    {0x3ff4afd6a052bf5bULL, 0xbfd07138604d6000ULL, 0x3d3e70124e912b17ULL},
    {0x3ff49539e3b2d067ULL, 0xbfd01eae5626c000ULL, 0xbd3a44ecfade85aeULL},
    {0x3ff47ae147ae147bULL, 0xbfcf991c6cb3c000ULL, 0x3d390b84cd7cc834ULL},
    {0x3ff460cbc7f5cf9aULL, 0xbfcef5ade4dd0000ULL, 0x3ccad11565bb8e11ULL},
    {0x3ff446f86562d9fbULL, 0xbfce530effe72000ULL, 0x3d3fdafbb13f7c18ULL},
    {0x3ff42d6625d51f87ULL, 0xbfcdb13db0d48000ULL, 0xbd32813a847527e6ULL},
    {0x3ff4141414141414ULL, 0xbfcd1037f2656000ULL, 0x3d08527e75b6f6e4ULL},
    {0x3ff3fb013fb013fbULL, 0xbfcc6ffbc6f00000ULL, 0xbd3ee128d3a69d43ULL},
    {0x3ff3e22cbce4a902ULL, 0xbfcbd087383be000ULL, 0x3d2d5844595412b6ULL},
    {0x3ff3c995a47babe7ULL, 0xbfcb31d8575bc000ULL, 0xbd3c75de562a63cbULL},
    {0x3ff3b13b13b13b14ULL, 0xbfca93ed3c8ae000ULL, 0x3d286a4350562169ULL},
    {0x3ff3991c2c187f63ULL, 0xbfc9f6c40708a000ULL, 0x3d333aa94bcd3f43ULL},
    {0x3ff3813813813814ULL, 0xbfc95a5adcf70000ULL, 0xbd08262858a0ff6fULL},
    {0x3ff3698df3de0748ULL, 0xbfc8beafeb390000ULL, 0x3d071154aae92cd1ULL},
    {0x3ff3521cfb2b78c1ULL, 0xbfc823c16551a000ULL, 0xbd1e02db9a631e83ULL},
    {0x3ff33ae45b57bcb2ULL, 0xbfc7898d85444000ULL, 0xbd38e81be3dbaf3fULL},
    {0x3ff323e34a2b10bfULL, 0xbfc6f0128b756000ULL, 0xbd3571d90d31ef0fULL},
    {0x3ff30d190130d190ULL, 0xbfc6574ebe8c2000ULL, 0x3d398d1d34f0f462ULL},
    {0x3ff2f684bda12f68ULL, 0xbfc5bf406b544000ULL, 0x3d128023eb68981cULL},
    {0x3ff2e025c04b8097ULL, 0xbfc527e5e4a1c000ULL, 0x3d34e61b8d4b411dULL},
    {0x3ff2c9fb4d812ca0ULL, 0xbfc4913d8333c000ULL, 0x3d353a43558124c4ULL},
    {0x3ff2b404ad012b40ULL, 0xbfc3fb45a5992000ULL, 0xbd319313c0cae559ULL},
    {0x3ff29e4129e4129eULL, 0xbfc365fcb015a000ULL, 0x3d3fd720afb9691bULL},
    {0x3ff288b01288b013ULL, 0xbfc2d1610c868000ULL, 0xbd03d0eccb81b4a1ULL},
    {0x3ff27350b8812735ULL, 0xbfc23d712a49c000ULL, 0xbd100aa38fd3df5cULL},
    {0x3ff25e22708092f1ULL, 0xbfc1aa2b7e240000ULL, 0x3d31ad48dde3b366ULL},
    {0x3ff2492492492492ULL, 0xbfc1178e8227e000ULL, 0xbd21e778ce2d07f2ULL},
    {0x3ff23456789abcdfULL, 0xbfc08598b59e4000ULL, 0x3d27e5fd7009902cULL},
    {0x3ff21fb78121fb78ULL, 0xbfbfe89139dbc000ULL, 0xbd356494d82f7a82ULL},
    {0x3ff20b470c67c0d9ULL, 0xbfbec739830a0000ULL, 0xbd31267ba80cdd10ULL},
    {0x3ff1f7047dc11f70ULL, 0xbfbda72763844000ULL, 0xbd1a79401fa71733ULL},
    {0x3ff1e2ef3b3fb874ULL, 0xbfbc885801bc4000ULL, 0xbd263f51c65aacd3ULL},
    {0x3ff1cf06ada2811dULL, 0xbfbb6ac88dad4000ULL, 0xbd3b1cbff50225c7ULL},
    {0x3ff1bb4a4046ed29ULL, 0xbfba4e7640b1c000ULL, 0x3d0e4336b94407c8ULL},
    {0x3ff1a7b9611a7b96ULL, 0xbfb9335e5d594000ULL, 0xbd230f5c3abd47daULL},
    {0x3ff19453808ca29cULL, 0xbfb8197e2f410000ULL, 0x3d3c102460d20041ULL},
    {0x3ff1811811811812ULL, 0xbfb700d30aeac000ULL, 0xbced068da99ded32ULL},
    {0x3ff16e0689427379ULL, 0xbfb5e95a4d978000ULL, 0xbd31ccace1d17171ULL},
    {0x3ff15b1e5f75270dULL, 0xbfb4d3115d208000ULL, 0x3cf53e2582f4e1efULL},
    {0x3ff1485f0e0acd3bULL, 0xbfb3bdf5a7d20000ULL, 0x3d31a1e0ad125895ULL},
    {0x3ff135c81135c811ULL, 0xbfb2aa04a4470000ULL, 0xbd37a16ba8b1cb41ULL},
    {0x3ff12358e75d3033ULL, 0xbfb1973bd1464000ULL, 0xbd3560a154f930b3ULL},
    {0x3ff1111111111111ULL, 0xbfb08598b59e4000ULL, 0x3d17e9dd7009902cULL},
    {0x3ff0fef010fef011ULL, 0xbfaeea31c0068000ULL, 0xbd3c3de83606d891ULL},
    {0x3ff0ecf56be69c90ULL, 0xbfaccb73cddd8000ULL, 0xbd3967c36e09f5feULL},
    {0x3ff0db20a88f4696ULL, 0xbfaaaef2d0fb0000ULL, 0xbd21085a353bb42eULL},
    {0x3ff0c9714fbcda3bULL, 0xbfa894aa149f8000ULL, 0xbd39a55a8be97661ULL},
    {0x3ff0b7e6ec259dc8ULL, 0xbfa67c94f2d48000ULL, 0xbd3db2a0827cca0cULL},
    {0x3ff0a6810a6810a7ULL, 0xbfa466aed42e0000ULL, 0x3d2c073375bdfd28ULL},
    {0x3ff0953f39010954ULL, 0xbfa252f32f8d0000ULL, 0xbd28401ae021b67bULL},
    {0x3ff0842108421084ULL, 0xbfa0415d89e78000ULL, 0x3d3ddfc7f461c516ULL},
    {0x3ff073260a47f7c6ULL, 0xbf9c63d2ec150000ULL, 0x3d354a3ce030a687ULL},
    {0x3ff0624dd2f1a9fcULL, 0xbf98492528c90000ULL, 0x3d2a9dba325a0c34ULL},
    {0x3ff05197f7d73404ULL, 0xbf9432a925980000ULL, 0xbd097739928637feULL},
    {0x3ff0410410410410ULL, 0xbf90205658930000ULL, 0xbd360dd27c8e8417ULL},
    {0x3ff03091b51f5e1aULL, 0xbf882448a3880000ULL, 0xbd34506412c584e0ULL},
    {0x3ff0204081020408ULL, 0xbf80101575880000ULL, 0xbd3bcd251998b506ULL},
    {0x3ff0101010101010ULL, 0xbf70080559580000ULL, 0xbd2164afcb31c67bULL},
    {0x3ff0000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x3fefc07f01fc07f0ULL, 0x3f7fe02a6b100000ULL, 0x3d19e63f0dda40e4ULL},
    {0x3fef81f81f81f820ULL, 0x3f8fc0a8b0fc0000ULL, 0x3cde1e7cf6d3a69cULL},
    {0x3fef44659e4a4271ULL, 0x3f97b91b07d60000ULL, 0xbd33b685b602ace4ULL},
    {0x3fef07c1f07c1f08ULL, 0x3f9f829b0e780000ULL, 0x3d297c267c7e09e4ULL},
    {0x3feecc07b301ecc0ULL, 0x3fa39e87b9fe8000ULL, 0x3d3eb3d480ad9015ULL},
    {0x3fee9131abf0b767ULL, 0x3fa77458f6330000ULL, 0xbd31807ce586af09ULL},
    {0x3fee573ac901e574ULL, 0x3fab42dd71198000ULL, 0xbd1c8d7ae5d6704cULL},
    {0x3fee1e1e1e1e1e1eULL, 0x3faf0a30c0118000ULL, 0xbd2d579e83368e91ULL},
    {0x3fede5d6e3f8868aULL, 0x3fb16536eea38000ULL, 0xbd1472de768fa309ULL},
    {0x3fedae6076b981dbULL, 0x3fb341d7961bc000ULL, 0x3d31cfb299837610ULL},
    {0x3fed77b654b82c34ULL, 0x3fb51b073f060000ULL, 0x3d383ba9278e686aULL},
    {0x3fed41d41d41d41dULL, 0x3fb6f0d28ae58000ULL, 0xbd34b2241b664613ULL},
    {0x3fed0cb58f6ec074ULL, 0x3fb8c345d6318000ULL, 0x3d3b22b5acb42a66ULL},
    {0x3fecd85689039b0bULL, 0x3fba926d3a4ac000ULL, 0x3d3561c50bd22a9cULL},
    {0x3feca4b3055ee191ULL, 0x3fbc5e548f5bc000ULL, 0x3d1d0c97585fbe06ULL},
    {0x3fec71c71c71c71cULL, 0x3fbe27076e2b0000ULL, 0xbd2a2c2c2af0003cULL},
    {0x3fec3f8f01c3f8f0ULL, 0x3fbfec9131dc0000ULL, 0xbd354455d1ae6607ULL},
    {0x3fec0e070381c0e0ULL, 0x3fc0d77e7cd08000ULL, 0x3d3cb6cd2ee2f482ULL},
    {0x3febdd2b899406f7ULL, 0x3fc1b72ad52f6000ULL, 0x3d2e86041811a396ULL},
    {0x3febacf914c1bad0ULL, 0x3fc29552f8200000ULL, 0xbd35bd67f4471dfcULL},
    {0x3feb7d6c3dda338bULL, 0x3fc371fc201e8000ULL, 0x3d3eea079b2d8abcULL},
    {0x3feb4e81b4e81b4fULL, 0x3fc44d2b6ccb8000ULL, 0xbd171f416135783cULL},
    {0x3feb2036406c80d9ULL, 0x3fc526e5e3a1c000ULL, 0xbd3790aa37fc5238ULL},
    {0x3feaf286bca1af28ULL, 0x3fc5ff3070a7a000ULL, 0xbd38546f183bebf2ULL},
    {0x3feac5701ac5701bULL, 0x3fc6d60fe719e000ULL, 0xbd3bc91557134767ULL},
    {0x3fea98ef606a63beULL, 0x3fc7ab890210e000ULL, 0xbd2be51072534a58ULL},
    {0x3fea6d01a6d01a6dULL, 0x3fc87fa06520c000ULL, 0x3d322130401202fcULL},
    {0x3fea41a41a41a41aULL, 0x3fc9525a9cf46000ULL, 0xbd3294937d9f158fULL},
    {0x3fea16d3f97a4b02ULL, 0x3fca23bc1fe2c000ULL, 0xbd353d6d91dc9f0bULL},
    {0x3fe9ec8e951033d9ULL, 0x3fcaf3c94e80c000ULL, 0xbcb92e633fcd9066ULL},
    {0x3fe9c2d14ee4a102ULL, 0x3fcbc286742d8000ULL, 0x3d39a873f39d121cULL},
    {0x3fe999999999999aULL, 0x3fcc8ff7c79aa000ULL, 0xbd27814f689f8434ULL},
    {0x3fe970e4f80cb872ULL, 0x3fcd5c216b4fc000ULL, 0xbd21b0d1bbca681bULL},
    {0x3fe948b0fcd6e9e0ULL, 0x3fce27076e2b0000ULL, 0xbd3a302c2af0003cULL},
    {0x3fe920fb49d0e229ULL, 0x3fcef0adcbdc6000ULL, 0xbd2b2a179c86af24ULL},
    {0x3fe8f9c18f9c18faULL, 0x3fcfb9186d5e4000ULL, 0xbd0d6b2aab993c87ULL},
    {0x3fe8d3018d3018d3ULL, 0x3fd0402594b4d000ULL, 0x3cf037b89ef42d7fULL},
    {0x3fe8acb90f6bf3aaULL, 0x3fd0a324e2739000ULL, 0x3d0c4dee7ef4030eULL},
    {0x3fe886e5f0abb04aULL, 0x3fd1058bf9ae5000ULL, 0xbd34affd817d52cdULL},
    {0x3fe8618618618618ULL, 0x3fd1675cababa000ULL, 0x3d383c0e731f55c4ULL},
    {0x3fe83c977ab2beddULL, 0x3fd1c898c169a000ULL, 0xbd381260e5c62affULL},
    {0x3fe8181818181818ULL, 0x3fd22941fbcf8000ULL, 0xbd3a6876f5eb0963ULL},
    {0x3fe7f405fd017f40ULL, 0x3fd2895a13de8000ULL, 0x3d3a917ad24c13f0ULL},
    {0x3fe7d05f417d05f4ULL, 0x3fd2e8e2bae12000ULL, 0xbd26791e99b72bd8ULL},
    {0x3fe7ad2208e0ecc3ULL, 0x3fd347dd9a988000ULL, 0xbd25522dd4c58092ULL},
    {0x3fe78a4c8178a4c8ULL, 0x3fd3a64c55694000ULL, 0x3d37a81cbcd735d0ULL},
    {0x3fe767dce434a9b1ULL, 0x3fd404308686a000ULL, 0x3d3f8f043049f7d3ULL},
};

// log1p(r) - r = r^2 q(r), |r| < 2^-8
inline double log_q(double r, double r2) {
    const double a = std::fma(r, 1.0 / 3, -0.5);
    const double b = std::fma(r, 1.0 / 5, -0.25);
    const double c = std::fma(r, 1.0 / 7, -1.0 / 6);
    return std::fma(r2 * r2, c, std::fma(r2, b, a));
}
inline float64x2_t log_q(float64x2_t r, float64x2_t r2) {
    const float64x2_t a = vfmaq_f64(vdupq_n_f64(-0.5), r, vdupq_n_f64(1.0 / 3));
    const float64x2_t b = vfmaq_f64(vdupq_n_f64(-0.25), r, vdupq_n_f64(1.0 / 5));
    const float64x2_t c = vfmaq_f64(vdupq_n_f64(-1.0 / 6), r, vdupq_n_f64(1.0 / 7));
    return vfmaq_f64(vfmaq_f64(a, r2, b), vmulq_f64(r2, r2), c);
}

// log u for a positive normal u; `e` (bits(u)'s interval and exponent already split) adds
// e_scaled = e 2^-k to z before the reduction (log1p's rounding error of 1 + x; 0 for log).
template <bool Corr>
inline double log_from(std::uint64_t ub, double e) {
    const std::uint64_t t = ub - kLogOff;
    const LogEntry& c = kLogTable[(t >> 45) & 127];
    const std::int64_t k = static_cast<std::int64_t>(t) >> 52;
    const double z = as_double(ub - (static_cast<std::uint64_t>(k) << 52));
    const double invc = as_double(c.invc), lch = as_double(c.logc_hi);
    double r = std::fma(z, invc, -1.0);
    if constexpr (Corr) r = std::fma(e * as_double(static_cast<std::uint64_t>(1023 - k) << 52), invc, r);
    const double kd = static_cast<double>(k);
    const double a = kd * kLn2Hi;
    const double w = a + lch;  // exact: both multiples of 2^-42 below 2^10
    const double hi = w + r;
    const double lo = (w - hi) + r;
    const double r2 = r * r;
    const double low = std::fma(kd, kLn2Lo, as_double(c.logc_lo));
    return hi + std::fma(r2, log_q(r, r2), lo + low);
}

template <bool Corr>
inline float64x2_t log_from(uint64x2_t ub, float64x2_t e) {
    const uint64x2_t t = vsubq_u64(ub, vdupq_n_u64(kLogOff));
    const LogEntry& c0 = kLogTable[(vgetq_lane_u64(t, 0) >> 45) & 127];
    const LogEntry& c1 = kLogTable[(vgetq_lane_u64(t, 1) >> 45) & 127];
    const int64x2_t k = vshrq_n_s64(vreinterpretq_s64_u64(t), 52);
    const float64x2_t z = vreinterpretq_f64_u64(vsubq_u64(ub, vshlq_n_u64(vreinterpretq_u64_s64(k), 52)));
    const uint64x2_t a0 = vld1q_u64(&c0.invc), a1 = vld1q_u64(&c1.invc);
    const float64x2_t invc = vreinterpretq_f64_u64(vzip1q_u64(a0, a1));
    const float64x2_t lch = vreinterpretq_f64_u64(vzip2q_u64(a0, a1));
    const float64x2_t lcl = vreinterpretq_f64_u64(vcombine_u64(vld1_u64(&c0.logc_lo), vld1_u64(&c1.logc_lo)));
    float64x2_t r = vfmaq_f64(vdupq_n_f64(-1.0), z, invc);
    if constexpr (Corr) {
        const float64x2_t two_mk = vreinterpretq_f64_s64(vshlq_n_s64(vsubq_s64(vdupq_n_s64(1023), k), 52));
        r = vfmaq_f64(r, vmulq_f64(e, two_mk), invc);
    }
    const float64x2_t kd = vcvtq_f64_s64(k);
    const float64x2_t a = vmulq_f64(kd, vdupq_n_f64(kLn2Hi));
    const float64x2_t w = vaddq_f64(a, lch);
    const float64x2_t hi = vaddq_f64(w, r);
    const float64x2_t lo = vaddq_f64(vsubq_f64(w, hi), r);
    const float64x2_t r2 = vmulq_f64(r, r);
    const float64x2_t low = vfmaq_f64(lcl, kd, vdupq_n_f64(kLn2Lo));
    return vaddq_f64(hi, vfmaq_f64(vaddq_f64(lo, low), r2, log_q(r, r2)));
}

// Lanes log_from covers: positive normal finite x (unsigned compare: negatives wrap high).
inline bool log_ok(std::uint64_t b) { return b - 0x0010000000000000ULL < 0x7fe0000000000000ULL; }
inline uint64x2_t log_ok(uint64x2_t b) {
    return vcltq_u64(vsubq_u64(b, vdupq_n_u64(0x0010000000000000ULL)), vdupq_n_u64(0x7fe0000000000000ULL));
}

inline double log1(double x) { return log_ok(as_bits(x)) ? log_from<false>(as_bits(x), 0.0) : std::log(x); }

inline float64x2_t log_lanes(float64x2_t x) { return log_from<false>(vreinterpretq_u64_f64(x), vdupq_n_f64(0.0)); }

// log1p's lanes: -1 < x < 2^1000, |x| >= kTiny (NaN fails)
inline constexpr double kLog1pMax = 0x1p1000;  // 2^-k stays normal for u = 1 + x below it
inline bool log1p_ok(double x) { return x > -1.0 && x < kLog1pMax; }

inline double log1p_1(double x) {
    if (!log1p_ok(x)) return std::log1p(x);
    if (std::fabs(x) < kTiny) return x;
    const double u = 1.0 + x;
    const double v = u - x;
    const double e = (1.0 - v) + (x - (u - v));  // 1 + x - u exactly (TwoSum)
    return log_from<true>(as_bits(u), e);
}

inline float64x2_t log1p_lanes(float64x2_t x) {
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t u = vaddq_f64(one, x);
    const float64x2_t v = vsubq_f64(u, x);
    const float64x2_t e = vaddq_f64(vsubq_f64(one, v), vsubq_f64(x, vsubq_f64(u, v)));
    const float64x2_t y = log_from<true>(vreinterpretq_u64_f64(u), e);
    return vbslq_f64(vcaltq_f64(x, vdupq_n_f64(kTiny)), x, y);
}

// ---- cos ------------------------------------------------------------------

inline constexpr double kTrigMax = 0x1p20;  // |x| below it: n < 2^20, so n P1 and n P2 are exact
inline constexpr double kInvPio2 = 0x1.45f306dc9c883p-1;
// pi/2 = P1 + P2 + P3 + P4 (Arb): P1 and P2 33 bits each, P3 and P4 rounded to nearest
inline constexpr double kPio2_1 = 0x1.921fb54400000p+0;
inline constexpr double kPio2_2 = 0x1.0b4611a600000p-34;
inline constexpr double kPio2_3 = 0x1.3198a2e037073p-69;
inline constexpr double kPio2_4 = 0x1.129024e088a68p-123;
// fdlibm's minimax coefficients (FreeBSD k_sin.c, k_cos.c): |poly - sin| < 2^-58 relative
inline constexpr double kS1 = -1.66666666666666324348e-01, kS2 = 8.33333333332248946124e-03,
                        kS3 = -1.98412698298579493134e-04, kS4 = 2.75573137070700676789e-06,
                        kS5 = -2.50507602534068634195e-08, kS6 = 1.58969099521155010221e-10;
inline constexpr double kC1 = 4.16666666666666019037e-02, kC2 = -1.38888888888741095749e-03,
                        kC3 = 2.48015872894767294178e-05, kC4 = -2.75573143513906633035e-07,
                        kC5 = 2.08757232129817482790e-09, kC6 = -1.13596475577881948265e-11;

// x = n pi/2 + hi + lo, |hi| <~ pi/4; q = n mod 4 in the low bits.
struct Reduced {
    double hi, lo;
    std::uint64_t q;
};

inline Reduced reduce(double x) {
    const double z = std::fma(x, kInvPio2, kExpShift);
    const double n = z - kExpShift;
    const double r1 = std::fma(-n, kPio2_1, x);  // exact
    const double r2 = std::fma(-n, kPio2_2, r1);
    const double e2 = std::fma(-n, kPio2_2, r1 - r2);  // r2's rounding error, exactly
    const double hi = std::fma(-n, kPio2_3, r2);
    const double lo = std::fma(-n, kPio2_4, std::fma(-n, kPio2_3, r2 - hi) + e2);
    return {hi, lo, as_bits(z)};
}

// sin(x + y) and cos(x + y), |x| <~ pi/4, |y| < ulp(x): FreeBSD's __kernel_sin, __kernel_cos
inline double sin_k(double x, double y) {
    const double z = x * x, w = z * z;
    const double r = std::fma(z, std::fma(z, kS4, kS3), kS2) + z * w * std::fma(z, kS6, kS5);
    const double v = z * x;
    return x - ((z * (0.5 * y - v * r) - y) - v * kS1);
}
inline double cos_k(double x, double y) {
    const double z = x * x, w = z * z;
    const double r = z * std::fma(z, std::fma(z, kC3, kC2), kC1) + w * w * std::fma(z, std::fma(z, kC6, kC5), kC4);
    const double hz = 0.5 * z;
    const double u = 1.0 - hz;
    return u + (((1.0 - u) - hz) + (z * r - x * y));
}

struct ReducedV {
    float64x2_t hi, lo;
    uint64x2_t q;
};

inline ReducedV reduce(float64x2_t x) {
    const float64x2_t z = vfmaq_f64(vdupq_n_f64(kExpShift), x, vdupq_n_f64(kInvPio2));
    const float64x2_t mn = vsubq_f64(vdupq_n_f64(kExpShift), z);  // -n
    const float64x2_t r1 = vfmaq_f64(x, mn, vdupq_n_f64(kPio2_1));
    const float64x2_t r2 = vfmaq_f64(r1, mn, vdupq_n_f64(kPio2_2));
    const float64x2_t e2 = vfmaq_f64(vsubq_f64(r1, r2), mn, vdupq_n_f64(kPio2_2));
    const float64x2_t hi = vfmaq_f64(r2, mn, vdupq_n_f64(kPio2_3));
    const float64x2_t lo =
        vfmaq_f64(vaddq_f64(vfmaq_f64(vsubq_f64(r2, hi), mn, vdupq_n_f64(kPio2_3)), e2), mn, vdupq_n_f64(kPio2_4));
    return {hi, lo, vreinterpretq_u64_f64(z)};
}

inline float64x2_t sin_k(float64x2_t x, float64x2_t y) {
    const float64x2_t z = vmulq_f64(x, x), w = vmulq_f64(z, z);
    const float64x2_t r = vaddq_f64(
        vfmaq_f64(vdupq_n_f64(kS2), z, vfmaq_f64(vdupq_n_f64(kS3), z, vdupq_n_f64(kS4))),
        vmulq_f64(vmulq_f64(z, w), vfmaq_f64(vdupq_n_f64(kS5), z, vdupq_n_f64(kS6)))
    );
    const float64x2_t v = vmulq_f64(z, x);
    const float64x2_t t = vsubq_f64(vmulq_f64(vdupq_n_f64(0.5), y), vmulq_f64(v, r));
    return vsubq_f64(x, vsubq_f64(vsubq_f64(vmulq_f64(z, t), y), vmulq_f64(v, vdupq_n_f64(kS1))));
}
inline float64x2_t cos_k(float64x2_t x, float64x2_t y) {
    const float64x2_t z = vmulq_f64(x, x), w = vmulq_f64(z, z);
    const float64x2_t a = vfmaq_f64(vdupq_n_f64(kC1), z, vfmaq_f64(vdupq_n_f64(kC2), z, vdupq_n_f64(kC3)));
    const float64x2_t b = vfmaq_f64(vdupq_n_f64(kC4), z, vfmaq_f64(vdupq_n_f64(kC5), z, vdupq_n_f64(kC6)));
    const float64x2_t r = vaddq_f64(vmulq_f64(z, a), vmulq_f64(vmulq_f64(w, w), b));
    const float64x2_t hz = vmulq_f64(vdupq_n_f64(0.5), z);
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t u = vsubq_f64(one, hz);
    return vaddq_f64(u, vaddq_f64(vsubq_f64(vsubq_f64(one, u), hz), vsubq_f64(vmulq_f64(z, r), vmulq_f64(x, y))));
}

inline double flip(double v, std::uint64_t sign) { return as_double(as_bits(v) ^ sign); }

inline bool trig_ok(double x) { return std::fabs(x) <= kTrigMax; }  // NaN fails
inline uint64x2_t trig_ok(float64x2_t x) { return vcaleq_f64(x, vdupq_n_f64(kTrigMax)); }

inline double cos1(double x) {
    if (!trig_ok(x)) return std::cos(x);
    const Reduced r = reduce(x);
    const double s = sin_k(r.hi, r.lo), c = cos_k(r.hi, r.lo);
    return flip(r.q & 1 ? s : c, ((r.q + 1) & 2) << 62);
}
inline float64x2_t cos_lanes(float64x2_t x) {
    const ReducedV r = reduce(x);
    const float64x2_t s = sin_k(r.hi, r.lo), c = cos_k(r.hi, r.lo);
    const uint64x2_t odd = vtstq_u64(r.q, vdupq_n_u64(1));
    const uint64x2_t sign = vshlq_n_u64(vandq_u64(vaddq_u64(r.q, vdupq_n_u64(1)), vdupq_n_u64(2)), 62);
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vbslq_f64(odd, s, c)), sign));
}
// ---- whole arrays ------------------------------------------------------------

// run's loop one pair a step (cos: its many constants gained nothing from two pairs a step)
template <typename Lanes, typename OkV, typename One>
inline void run_pairs(double* out, const double* x, std::int64_t n, Lanes lanes, OkV ok, One one) {
    std::int64_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t v = vld1q_f64(x + i);
        const uint64x2_t good = ok(v);
        if ((vgetq_lane_u64(good, 0) & vgetq_lane_u64(good, 1)) != 0) [[likely]] {
            vst1q_f64(out + i, lanes(v));
        } else {
            const double a = vgetq_lane_f64(v, 0), b = vgetq_lane_f64(v, 1);
            out[i] = one(a);
            out[i + 1] = one(b);
        }
    }
    if (i < n) out[i] = one(x[i]);
}

// out[i] = f(x[i]) for n contiguous elements (out may be x): Lanes(v) on pairs whose both
// elements pass Ok, else One per element (One also takes the odd last one).
template <typename Lanes, typename OkV, typename One>
inline void run(double* out, const double* x, std::int64_t n, Lanes lanes, OkV ok, One one) {
    std::int64_t i = 0;
    for (; i + 4 <= n; i += 4) {  // two pairs a step: one branch for both
        const float64x2_t v0 = vld1q_f64(x + i), v1 = vld1q_f64(x + i + 2);
        const uint64x2_t good = vandq_u64(ok(v0), ok(v1));
        if ((vgetq_lane_u64(good, 0) & vgetq_lane_u64(good, 1)) != 0) [[likely]] {
            const float64x2_t y0 = lanes(v0), y1 = lanes(v1);
            vst1q_f64(out + i, y0);
            vst1q_f64(out + i + 2, y1);
        } else {
            double t[4];
            vst1q_f64(t, v0);
            vst1q_f64(t + 2, v1);
            for (int k = 0; k < 4; ++k) out[i + k] = one(t[k]);
        }
    }
    for (; i < n; ++i) out[i] = one(x[i]);
}

inline void exp(double* out, const double* x, std::int64_t n) {
    run(
        out, x, n, [](float64x2_t v) { return exp_lanes(v); },
        [](float64x2_t v) { return vcaleq_f64(v, vdupq_n_f64(kExpMax)); }, [](double v) { return exp1(v); }
    );
}

inline void expm1(double* out, const double* x, std::int64_t n) {
    run(
        out, x, n, [](float64x2_t v) { return expm1_lanes(v); },
        [](float64x2_t v) { return vcaleq_f64(v, vdupq_n_f64(kExpMax)); }, [](double v) { return expm1_1(v); }
    );
}

inline void log1p(double* out, const double* x, std::int64_t n) {
    run(
        out, x, n, [](float64x2_t v) { return log1p_lanes(v); },
        [](float64x2_t v) { return vandq_u64(vcgtq_f64(v, vdupq_n_f64(-1.0)), vcltq_f64(v, vdupq_n_f64(kLog1pMax))); },
        [](double v) { return log1p_1(v); }
    );
}

inline void cos(double* out, const double* x, std::int64_t n) {
    run_pairs(
        out, x, n, [](float64x2_t v) { return cos_lanes(v); }, [](float64x2_t v) { return trig_ok(v); },
        [](double v) { return cos1(v); }
    );
}
}  // namespace sd::vmath
