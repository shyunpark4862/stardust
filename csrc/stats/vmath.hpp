// Elementwise exp for the stats chunk kernels (stats/chunk.hpp), in NEON lanes with a
// scalar twin for the odd last element: the same IEEE operations in the same order, so an
// element's value does not depend on its position or the call's length (f(x)[i] == f(x[i])).
// 0.6 ns an element against vvexp's 1.0 on this machine (M4 Pro), and within 0.5 ULP (vvexp 1).
//
// exp: x = (k + j/128) log 2 + r, |r| <= log 2 / 256, e^x = 2^k 2^(j/128) (1 + p(r)) with
// 2^(j/128) a double and its relative tail (a table made with Arb), p the degree-5 Taylor
// polynomial (its truncation r^6/720 < 2^-60). The algorithm of ARM's optimized-routines exp;
// relative error at most 0.52 ULP (1.04 in units of 2^-53) on the normal range. Arguments
// outside [-708, 708] (and NaN) take libm's exp.
//
// (A log of the same kind, fdlibm's table-free form vectorized, ran at vvlog's speed here:
// the chunk kernels keep vForce's log.)
#pragma once

#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace sd::stats::vm {

inline constexpr double kExpLn2HiN = 0x1.62e42fefa39efp-8;
inline constexpr double kExpLn2LoN = 0x1.abc9e3b39803fp-63;
inline constexpr double kExpInvLn2N = 0x1.71547652b82fep+7;
// 2^(j/128): {relative tail, bits of the double - (j << 45)}, the second ready for the exponent.
alignas(64) inline constexpr std::uint64_t kExpTable[2 * 128] = {
    0x0000000000000000ULL, 0x3ff0000000000000ULL, 0x3c9b3b4f1a88bf6eULL, 0x3feff63da9fb3335ULL, 0xbc7160139cd8dc5dULL,
    0x3fefec9a3e778061ULL, 0xbc905e7a108766d1ULL, 0x3fefe315e86e7f85ULL, 0x3c8cd2523567f613ULL, 0x3fefd9b0d3158574ULL,
    0xbc8bce8023f98efaULL, 0x3fefd06b29ddf6deULL, 0x3c60f74e61e6c861ULL, 0x3fefc74518759bc8ULL, 0x3c90a3e45b33d399ULL,
    0x3fefbe3ecac6f383ULL, 0x3c979aa65d837b6dULL, 0x3fefb5586cf9890fULL, 0x3c8eb51a92fdeffcULL, 0x3fefac922b7247f7ULL,
    0x3c3ebe3d702f9cd1ULL, 0x3fefa3ec32d3d1a2ULL, 0xbc6a033489906e0bULL, 0x3fef9b66affed31bULL, 0xbc9556522a2fbd0eULL,
    0x3fef9301d0125b51ULL, 0xbc5080ef8c4eea55ULL, 0x3fef8abdc06c31ccULL, 0xbc91c923b9d5f416ULL, 0x3fef829aaea92de0ULL,
    0x3c80d3e3e95c55afULL, 0x3fef7a98c8a58e51ULL, 0xbc801b15eaa59348ULL, 0x3fef72b83c7d517bULL, 0xbc8f1ff055de323dULL,
    0x3fef6af9388c8deaULL, 0x3c8b898c3f1353bfULL, 0x3fef635beb6fcb75ULL, 0xbc96d99c7611eb26ULL, 0x3fef5be084045cd4ULL,
    0x3c9aecf73e3a2f60ULL, 0x3fef54873168b9aaULL, 0xbc8fe782cb86389dULL, 0x3fef4d5022fcd91dULL, 0x3c8a6f4144a6c38dULL,
    0x3fef463b88628cd6ULL, 0x3c807a05b0e4047dULL, 0x3fef3f49917ddc96ULL, 0x3c968efde3a8a894ULL, 0x3fef387a6e756238ULL,
    0x3c875e18f274487dULL, 0x3fef31ce4fb2a63fULL, 0x3c80472b981fe7f2ULL, 0x3fef2b4565e27cddULL, 0xbc96b87b3f71085eULL,
    0x3fef24dfe1f56381ULL, 0x3c82f7e16d09ab31ULL, 0x3fef1e9df51fdee1ULL, 0xbc3d219b1a6fbffaULL, 0x3fef187fd0dad990ULL,
    0x3c8b3782720c0ab4ULL, 0x3fef1285a6e4030bULL, 0x3c6e149289cecb8fULL, 0x3fef0cafa93e2f56ULL, 0x3c834d754db0abb6ULL,
    0x3fef06fe0a31b715ULL, 0x3c864201e2ac744cULL, 0x3fef0170fc4cd831ULL, 0x3c8fdd395dd3f84aULL, 0x3feefc08b26416ffULL,
    0xbc86a3803b8e5b04ULL, 0x3feef6c55f929ff1ULL, 0xbc924aedcc4b5068ULL, 0x3feef1a7373aa9cbULL, 0xbc9907f81b512d8eULL,
    0x3feeecae6d05d866ULL, 0xbc71d1e83e9436d2ULL, 0x3feee7db34e59ff7ULL, 0xbc991919b3ce1b15ULL, 0x3feee32dc313a8e5ULL,
    0x3c859f48a72a4c6dULL, 0x3feedea64c123422ULL, 0xbc9312607a28698aULL, 0x3feeda4504ac801cULL, 0xbc58a78f4817895bULL,
    0x3feed60a21f72e2aULL, 0xbc7c2c9b67499a1bULL, 0x3feed1f5d950a897ULL, 0x3c4363ed60c2ac11ULL, 0x3feece086061892dULL,
    0x3c9666093b0664efULL, 0x3feeca41ed1d0057ULL, 0x3c6ecce1daa10379ULL, 0x3feec6a2b5c13cd0ULL, 0x3c93ff8e3f0f1230ULL,
    0x3feec32af0d7d3deULL, 0x3c7690cebb7aafb0ULL, 0x3feebfdad5362a27ULL, 0x3c931dbdeb54e077ULL, 0x3feebcb299fddd0dULL,
    0xbc8f94340071a38eULL, 0x3feeb9b2769d2ca7ULL, 0xbc87deccdc93a349ULL, 0x3feeb6daa2cf6642ULL, 0xbc78dec6bd0f385fULL,
    0x3feeb42b569d4f82ULL, 0xbc861246ec7b5cf6ULL, 0x3feeb1a4ca5d920fULL, 0x3c93350518fdd78eULL, 0x3feeaf4736b527daULL,
    0x3c7b98b72f8a9b05ULL, 0x3feead12d497c7fdULL, 0x3c9063e1e21c5409ULL, 0x3feeab07dd485429ULL, 0x3c34c7855019c6eaULL,
    0x3feea9268a5946b7ULL, 0x3c9432e62b64c035ULL, 0x3feea76f15ad2148ULL, 0xbc8ce44a6199769fULL, 0x3feea5e1b976dc09ULL,
    0xbc8c33c53bef4da8ULL, 0x3feea47eb03a5585ULL, 0xbc845378892be9aeULL, 0x3feea34634ccc320ULL, 0xbc93cedd78565858ULL,
    0x3feea23882552225ULL, 0x3c5710aa807e1964ULL, 0x3feea155d44ca973ULL, 0xbc93b3efbf5e2228ULL, 0x3feea09e667f3bcdULL,
    0xbc6a12ad8734b982ULL, 0x3feea012750bdabfULL, 0xbc6367efb86da9eeULL, 0x3fee9fb23c651a2fULL, 0xbc80dc3d54e08851ULL,
    0x3fee9f7df9519484ULL, 0xbc781f647e5a3ecfULL, 0x3fee9f75e8ec5f74ULL, 0xbc86ee4ac08b7db0ULL, 0x3fee9f9a48a58174ULL,
    0xbc8619321e55e68aULL, 0x3fee9feb564267c9ULL, 0x3c909ccb5e09d4d3ULL, 0x3feea0694fde5d3fULL, 0xbc7b32dcb94da51dULL,
    0x3feea11473eb0187ULL, 0x3c94ecfd5467c06bULL, 0x3feea1ed0130c132ULL, 0x3c65ebe1abd66c55ULL, 0x3feea2f336cf4e62ULL,
    0xbc88a1c52fb3cf42ULL, 0x3feea427543e1a12ULL, 0xbc9369b6f13b3734ULL, 0x3feea589994cce13ULL, 0xbc805e843a19ff1eULL,
    0x3feea71a4623c7adULL, 0xbc94d450d872576eULL, 0x3feea8d99b4492edULL, 0x3c90ad675b0e8a00ULL, 0x3feeaac7d98a6699ULL,
    0x3c8db72fc1f0eab4ULL, 0x3feeace5422aa0dbULL, 0xbc65b6609cc5e7ffULL, 0x3feeaf3216b5448cULL, 0x3c7bf68359f35f44ULL,
    0x3feeb1ae99157736ULL, 0xbc93091fa71e3d83ULL, 0x3feeb45b0b91ffc6ULL, 0xbc5da9b88b6c1e29ULL, 0x3feeb737b0cdc5e5ULL,
    0xbc6c23f97c90b959ULL, 0x3feeba44cbc8520fULL, 0xbc92434322f4f9aaULL, 0x3feebd829fde4e50ULL, 0xbc85ca6cd7668e4bULL,
    0x3feec0f170ca07baULL, 0x3c71affc2b91ce27ULL, 0x3feec49182a3f090ULL, 0x3c6dd235e10a73bbULL, 0x3feec86319e32323ULL,
    0xbc87c50422622263ULL, 0x3feecc667b5de565ULL, 0x3c8b1c86e3e231d5ULL, 0x3feed09bec4a2d33ULL, 0xbc91bbd1d3bcbb15ULL,
    0x3feed503b23e255dULL, 0x3c90cc319cee31d2ULL, 0x3feed99e1330b358ULL, 0x3c8469846e735ab3ULL, 0x3feede6b5579fdbfULL,
    0xbc82dfcd978e9db4ULL, 0x3feee36bbfd3f37aULL, 0x3c8c1a7792cb3387ULL, 0x3feee89f995ad3adULL, 0xbc907b8f4ad1d9faULL,
    0x3feeee07298db666ULL, 0xbc55c3d956dcaebaULL, 0x3feef3a2b84f15fbULL, 0xbc90a40e3da6f640ULL, 0x3feef9728de5593aULL,
    0xbc68d6f438ad9334ULL, 0x3feeff76f2fb5e47ULL, 0xbc91eee26b588a35ULL, 0x3fef05b030a1064aULL, 0x3c74ffd70a5fddcdULL,
    0x3fef0c1e904bc1d2ULL, 0xbc91bdfbfa9298acULL, 0x3fef12c25bd71e09ULL, 0x3c736eae30af0cb3ULL, 0x3fef199bdd85529cULL,
    0x3c8ee3325c9ffd94ULL, 0x3fef20ab5fffd07aULL, 0x3c84e08fd10959acULL, 0x3fef27f12e57d14bULL, 0x3c63cdaf384e1a67ULL,
    0x3fef2f6d9406e7b5ULL, 0x3c676b2c6c921968ULL, 0x3fef3720dcef9069ULL, 0xbc808a1883ccb5d2ULL, 0x3fef3f0b555dc3faULL,
    0xbc8fad5d3ffffa6fULL, 0x3fef472d4a07897cULL, 0xbc900dae3875a949ULL, 0x3fef4f87080d89f2ULL, 0x3c74a385a63d07a7ULL,
    0x3fef5818dcfba487ULL, 0xbc82919e2040220fULL, 0x3fef60e316c98398ULL, 0x3c8e5a50d5c192acULL, 0x3fef69e603db3285ULL,
    0x3c843a59ac016b4bULL, 0x3fef7321f301b460ULL, 0xbc82d52107b43e1fULL, 0x3fef7c97337b9b5fULL, 0xbc892ab93b470dc9ULL,
    0x3fef864614f5a129ULL, 0x3c74b604603a88d3ULL, 0x3fef902ee78b3ff6ULL, 0x3c83c5ec519d7271ULL, 0x3fef9a51fbc74c83ULL,
    0xbc8ff7128fd391f0ULL, 0x3fefa4afa2a490daULL, 0xbc8dae98e223747dULL, 0x3fefaf482d8e67f1ULL, 0x3c8ec3bc41aa2008ULL,
    0x3fefba1bee615a27ULL, 0x3c842b94c3a9eb32ULL, 0x3fefc52b376bba97ULL, 0x3c8a64a931d185eeULL, 0x3fefd0765b6e4540ULL,
    0xbc8e37bae43be3edULL, 0x3fefdbfdad9cbe14ULL, 0x3c77893b4d91cd9dULL, 0x3fefe7c1819e90d8ULL, 0x3c5305c14160cc89ULL,
    0x3feff3c22b8f71f1ULL,
};

inline constexpr double kExpShift = 0x1.8p52;
inline constexpr double kExpMax = 708.0;  // e^+-708 and the scale 2^(k + j/128) are normal

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

// One element, |x| <= kExpMax (the lanes' arithmetic).
inline double exp_core(double x) {
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
    return std::fma(scale, t, scale);
}

inline float64x2_t exp_core(float64x2_t x) {
    const float64x2_t z = vaddq_f64(vmulq_f64(x, vdupq_n_f64(kExpInvLn2N)), vdupq_n_f64(kExpShift));
    const float64x2_t kd = vsubq_f64(z, vdupq_n_f64(kExpShift));
    const uint64x2_t ki = vreinterpretq_u64_f64(z);
    float64x2_t r = vfmaq_f64(x, kd, vdupq_n_f64(-kExpLn2HiN));
    r = vfmaq_f64(r, kd, vdupq_n_f64(-kExpLn2LoN));
    const std::uint64_t j0 = vgetq_lane_u64(ki, 0) & 127, j1 = vgetq_lane_u64(ki, 1) & 127;
    const float64x2_t tail =
        vreinterpretq_f64_u64(vcombine_u64(vld1_u64(&kExpTable[2 * j0]), vld1_u64(&kExpTable[2 * j1])));
    const uint64x2_t sb = vcombine_u64(vld1_u64(&kExpTable[2 * j0 + 1]), vld1_u64(&kExpTable[2 * j1 + 1]));
    const float64x2_t scale = vreinterpretq_f64_u64(vaddq_u64(sb, vshlq_n_u64(ki, 45)));
    const float64x2_t r2 = vmulq_f64(r, r);
    const float64x2_t p23 = vfmaq_f64(vdupq_n_f64(0.5), r, vdupq_n_f64(1.0 / 6));
    const float64x2_t p45 = vfmaq_f64(vdupq_n_f64(1.0 / 24), r, vdupq_n_f64(1.0 / 120));
    float64x2_t t = vfmaq_f64(vaddq_f64(tail, r), r2, p23);
    t = vfmaq_f64(t, vmulq_f64(r2, r2), p45);
    return vfmaq_f64(scale, scale, t);
}

inline double exp1(double x) { return std::fabs(x) <= kExpMax ? exp_core(x) : std::exp(x); }

// out[i] = e^x[i]; out may be x.
inline void exp(double* out, const double* x, int n) {
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t v = vld1q_f64(x + i);
        const uint64x2_t in = vcaleq_f64(v, vdupq_n_f64(kExpMax));  // false for NaN
        float64x2_t y = exp_core(vbslq_f64(in, v, vdupq_n_f64(0.0)));
        if ((vgetq_lane_u64(in, 0) & vgetq_lane_u64(in, 1)) == 0) [[unlikely]] {
            double a[2];
            vst1q_f64(a, y);
            if (vgetq_lane_u64(in, 0) == 0) a[0] = std::exp(vgetq_lane_f64(v, 0));
            if (vgetq_lane_u64(in, 1) == 0) a[1] = std::exp(vgetq_lane_f64(v, 1));
            y = vld1q_f64(a);
        }
        vst1q_f64(out + i, y);
    }
    if (i < n) out[i] = exp1(x[i]);
}

}  // namespace sd::stats::vm
