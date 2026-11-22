// The library's own mixed-radix complex FFT (docs/design/fft.md §3.3): a Stockham transform with NEON
// butterflies of radix 2, 3, 4, 5, 7, 11 and (milestone FT6, decision FT64) 13, 17, 19, 23, 29 and 31, for
// every length whose prime factors are at most 31. Other lengths go through Bluestein (fft/bluestein.hpp),
// whose inner transform is one of these. next_fast_len's set stays the 11-smooth lengths (decision FT18).
//
// Layout. Input and output are interleaved complex128 (re, im, re, im, ...: sd's complex128 arrays
// and the axis driver's line buffers); the stages in between work on split buffers (all real parts,
// then all imaginary parts), so every NEON operation of a butterfly does two useful lanes (two values
// of the same position in two neighbouring sub-transforms) and no lane is shuffled. The first stage
// reads interleaved data with vld2q (de-interleaving on the load), the last writes interleaved data
// with vst2q: no separate (de)interleaving pass.
//
// Stockham ordering (no bit reversal). With L the length of the current sub-transforms, s = n / L
// their number (the stride) and m = L / r: stage input x and output y are related by
//     y[q + s (r p + k)] = w_L^(p k) * sum_j x[q + s (p + j m)] w_r^(j k),   0 <= q < s, 0 <= p < m,
// and the next stage runs with L / r and s r. The first stage (s = 1) runs the p loop inside (two
// neighbouring p per NEON register, its outputs interleaved by vst2q / vst3q / vst4q); the others run
// the q loop inside, where the twiddle is one broadcast pair per p. A loop of odd length ends with one
// position computed in both lanes of a register (the same operations on the same values) of which the
// first lane is stored: the same bits as in a pair, so a result never depends on whether its position
// was paired.
//
// Radices, in this order (decision FT35): 4 as often as it goes, one 2 if the power of two is odd,
// then 3, 5, 7, 11, 13, 17, 19, 23, 29, 31. Inverse transforms swap the real and imaginary parts on the first load and on the
// last store (swap(z) = i conj(z), so swap(DFT(swap(x))) is the DFT with the sign +): the same
// arithmetic as the forward transform, which is all the error analysis below needs.
//
// Arithmetic. A complex product is (ar wr - ai wi, ar wi + ai wr) with each part one product and one
// fused multiply-add (written out: the build has -ffp-contract=off). The odd radices p use the real
// form of the DFT of length p: with h = (p - 1) / 2, t_j = a_j + a_(p-j) and d_j = a_j - a_(p-j),
//     y_0 = a_0 + t_1 + ... + t_h,
//     A_k = a_0 + sum_j cos(2 pi jk / p) t_j,   B_k = sum_j sin(2 pi jk / p) d_j   (FMA chains, j = 1 .. h),
//     y_k = A_k - i B_k,   y_(p-k) = A_k + i B_k   (k = 1 .. h),
// with the constants rounded once to nearest (OddConst). Radix 2 and 4 need no multiplication (-i is an
// exact swap and negation). The result is a function of n alone (no choice depends on alignment,
// timing, the batch or values).
//
// Error bound (decision FT31; docs/design/fft.md §4.2, proof sketch there). A stage is the butterflies
// and then the twiddles; exactly it is sqrt(r) times a unitary map. tests/fft/proof.py propagates a
// first-order bound on every rounding of each butterfly as written here (the constants' exact errors
// included) and takes the 2-norm of the resulting nonnegative matrix: the computed butterflies satisfy
// ||b_hat - b|| <= eta_b sqrt(r) ||a|| with eta_b / u = 1.414 (r = 2), 4.000 (4), 5.768 (3), 9.912 (5),
// 13.669 (7), 22.875 (11). The twiddle product adds 2u (Jeannerod, Kornerup, Louvet and Muller 2017:
// the FMA complex product) and the twiddle's own error mu <= 1.5u (twiddle.hpp), so a stage's eta per
// log2 r is 4.914, 3.750, 5.847, 5.776, 6.116 and 7.624 u, all below kappa = 8: the length-n transform
// is within (kappa log2 n + O(u)) u ||y|| <= kappa L(n) u ||y||. The radices 13 to 31 need more, 8.221
// (13), 10.012, 10.930, 12.786, 15.679 and 16.826 u (31) per log2 r, so a length with one of them is held to
// kappa = 17 (decision FT64: still well inside the Bluestein bound those lengths had before, which carries
// rho ~ 2 and twice the levels); lengths with a prime factor above 31 go to Bluestein.
//
// Four-step (decision FT41, milestone FT6; Bailey 1990). A length n >= kFourStepMin is split once as
// n = n1 n2 (n1 the largest divisor of n at most sqrt n: n alone decides it). With j = j1 n2 + j2 and
// k = k1 + n1 k2,
//     y[k1 + n1 k2] = sum_j2 w_n2^(j2 k2) [w_n^(j2 k1) sum_j1 x[j1 n2 + j2] w_n1^(j1 k1)]:
// the column pass gathers kFourStepCols neighbouring columns (stride n2) at a time, transforms them (length
// n1), multiplies by the twiddles w_n^(j2 k1) (one full table, decision FT39, laid out by column block; none
// where j2 k1 = 0, as the stages skip p = 0) and writes them by column block into the work buffer; the row
// pass gathers kFourStepRows rows at a time from the column blocks, transforms them (length n2) and scatters
// them into y (stride n1). The blocks of each pass are independent tasks (run_tasks: the pool), so one long
// line runs on every thread, and the split and the operations depend on n alone: the bits are the same at
// any thread count. The inverse swaps the parts on the gather and on the last scatter, as above. The error
// count is the stages' (one butterfly stage and one twiddle product per level; the twiddle pass takes the
// place of the last column stage's twiddles), so the bound kappa L(n) u ||y|| holds as written, with the
// radices of n1 then those of n2.
//
// Two-line kernel (milestone FT6): the same stages with the two lanes of a register holding one position
// of two lines, for the lines of an axis other than the last (fft/driver.cpp); every lane goes through the
// operations the one-line kernel applies to that position, so the bits are the same (detail::Lines).
#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <vector>

#include "fft/twiddle.hpp"

namespace sd::fft {

// Runs fn(t) for t in [0, ntasks): the pool's parallel_for in the library (fft/plan.cpp), a plain loop in
// tools/fft_probe_kernels.cpp. The tasks are independent, so the threads change no bit. task_ns: the
// time of one task on one core (the pool's wake-up rule).
void run_tasks(int64_t ntasks, const std::function<void(int64_t)>& fn, int64_t task_ns);

// fn(i0, i1) over [0, count) in chunks of kChunk entries, the chunks on the pool when there are two or more
// (inside a task the pool runs them in order). For the O(n) passes of a long line (the real pass, Bluestein's
// chirps, the scaling): every entry is computed alone, so the chunks change no bit.
inline constexpr int64_t kChunk = int64_t{1} << 14;
template <class Fn>
inline void run_chunks(int64_t count, Fn&& fn) {
    if (count <= kChunk) {
        if (count > 0) fn(int64_t{0}, count);
        return;
    }
    run_tasks(
        (count + kChunk - 1) / kChunk, [&](int64_t t) { fn(t * kChunk, std::min(count, (t + 1) * kChunk)); }, kChunk
    );
}

// The four-step's threshold n4 and blocks (measured, milestone FT6: below 2^15 a split line loses more on one
// thread than it gains on ten).
inline constexpr int64_t kFourStepMin = int64_t{1} << 15;
inline constexpr int64_t kFourStepCols = 16;  // columns per block of the column pass
inline constexpr int64_t kFourStepRows = 16;  // rows per block of the row pass

namespace detail {

using V = float64x2_t;

struct CV {
    V re, im;  // two complex numbers in split form
};

inline CV cadd(CV a, CV b) { return {vaddq_f64(a.re, b.re), vaddq_f64(a.im, b.im)}; }
inline CV csub(CV a, CV b) { return {vsubq_f64(a.re, b.re), vsubq_f64(a.im, b.im)}; }
inline CV mul_neg_i(CV a) { return {a.im, vnegq_f64(a.re)}; }  // -i a, exact
inline CV cmul(CV a, V wr, V wi) {
    return {vfmsq_f64(vmulq_f64(a.re, wr), a.im, wi), vfmaq_f64(vmulq_f64(a.re, wi), a.im, wr)};
}

// cos and sin of 2 pi m / p, m = 0 .. p - 1, each rounded to nearest (computed with Arb at 300 bits;
// tests/fft/proof.py checks them and uses their exact errors).
template <int P>
struct OddConst;

template <>
struct OddConst<3> {
    static constexpr double c[3] = {0x1p+0, -0x1p-1, -0x1p-1};
    static constexpr double s[3] = {0.0, 0x1.bb67ae8584caap-1, -0x1.bb67ae8584caap-1};
};

template <>
struct OddConst<5> {
    static constexpr double c[5] = {
        0x1p+0, 0x1.3c6ef372fe950p-2, -0x1.9e3779b97f4a8p-1, -0x1.9e3779b97f4a8p-1, 0x1.3c6ef372fe950p-2
    };
    static constexpr double s[5] = {
        0.0, 0x1.e6f0e134454ffp-1, 0x1.2cf2304755a5ep-1, -0x1.2cf2304755a5ep-1, -0x1.e6f0e134454ffp-1
    };
};

template <>
struct OddConst<7> {
    static constexpr double c[7] = {
        0x1p+0,
        0x1.3f3a0e28bedd1p-1,
        -0x1.c7b90e3024582p-3,
        -0x1.cd4bca9cb5c71p-1,
        -0x1.cd4bca9cb5c71p-1,
        -0x1.c7b90e3024582p-3,
        0x1.3f3a0e28bedd1p-1,
    };
    static constexpr double s[7] = {
        0.0,
        0x1.904c37505de4bp-1,
        0x1.f329c0558e969p-1,
        0x1.bc4c04d71abc1p-2,
        -0x1.bc4c04d71abc1p-2,
        -0x1.f329c0558e969p-1,
        -0x1.904c37505de4bp-1,
    };
};

template <>
struct OddConst<11> {
    static constexpr double c[11] = {
        0x1p+0,
        0x1.aeb8c8764f0bap-1,
        0x1.a9628d9c712b6p-2,
        -0x1.2375f640f44dbp-3,
        -0x1.4f49e7f775887p-1,
        -0x1.eb42a9bcd5057p-1,
        -0x1.eb42a9bcd5057p-1,
        -0x1.4f49e7f775887p-1,
        -0x1.2375f640f44dbp-3,
        0x1.a9628d9c712b6p-2,
        0x1.aeb8c8764f0bap-1,
    };
    static constexpr double s[11] = {
        0.0,
        0x1.14cedf8bb580bp-1,
        0x1.d1bb48eee2c13p-1,
        0x1.fac9e043842efp-1,
        0x1.82f19bb3a28a1p-1,
        0x1.207e7fd768dbfp-2,
        -0x1.207e7fd768dbfp-2,
        -0x1.82f19bb3a28a1p-1,
        -0x1.fac9e043842efp-1,
        -0x1.d1bb48eee2c13p-1,
        -0x1.14cedf8bb580bp-1,
    };
};

template <>
struct OddConst<13> {
    static constexpr double c[13] = {
        0x1p+0,
        0x1.c55a7e00740e9p-1,
        0x1.22d961ea71119p-1,
        0x1.edb7debaa3ed8p-4,
        -0x1.6b1d8b2365da1p-2,
        -0x1.7f3ccd0032e0cp-1,
        -0x1.f11f493053d00p-1,
        -0x1.f11f493053d00p-1,
        -0x1.7f3ccd0032e0cp-1,
        -0x1.6b1d8b2365da1p-2,
        0x1.edb7debaa3ed8p-4,
        0x1.22d961ea71119p-1,
        0x1.c55a7e00740e9p-1,
    };
    static constexpr double s[13] = {
        0.0,
        0x1.dbe064267c47cp-2,
        0x1.a55e242a4c3d2p-1,
        0x1.fc44566966769p-1,
        0x1.deba72ef20147p-1,
        0x1.5384d024c2f84p-1,
        0x1.ea1e54bc48dbfp-3,
        -0x1.ea1e54bc48dbfp-3,
        -0x1.5384d024c2f84p-1,
        -0x1.deba72ef20147p-1,
        -0x1.fc44566966769p-1,
        -0x1.a55e242a4c3d2p-1,
        -0x1.dbe064267c47cp-2,
    };
};

template <>
struct OddConst<17> {
    static constexpr double c[17] = {
        0x1p+0,
        0x1.dd6d000370991p-1,
        0x1.7a5f6075d4884p-1,
        0x1.c86fa2b2883cdp-2,
        0x1.79ee63259b75ep-4,
        -0x1.183b1c61f0d01p-2,
        -0x1.348c86ed5f1bbp-1,
        -0x1.b34fa910ea3b9p-1,
        -0x1.f7484007faef3p-1,
        -0x1.f7484007faef3p-1,
        -0x1.b34fa910ea3b9p-1,
        -0x1.348c86ed5f1bbp-1,
        -0x1.183b1c61f0d01p-2,
        0x1.79ee63259b75ep-4,
        0x1.c86fa2b2883cdp-2,
        0x1.7a5f6075d4884p-1,
        0x1.dd6d000370991p-1,
    };
    static constexpr double s[17] = {
        0.0,
        0x1.71e955d8e7cdcp-2,
        0x1.58eea2a9d6da3p-1,
        0x1.ca52d7c9e640bp-1,
        0x1.fdd0deb564b22p-1,
        0x1.ec746923c349fp-1,
        0x1.9895b6c9a05f6p-1,
        0x1.0d8884363dd80p-1,
        0x1.7851aacd6c6b4p-3,
        -0x1.7851aacd6c6b4p-3,
        -0x1.0d8884363dd80p-1,
        -0x1.9895b6c9a05f6p-1,
        -0x1.ec746923c349fp-1,
        -0x1.fdd0deb564b22p-1,
        -0x1.ca52d7c9e640bp-1,
        -0x1.58eea2a9d6da3p-1,
        -0x1.71e955d8e7cdcp-2,
    };
};

template <>
struct OddConst<19> {
    static constexpr double c[19] = {
        0x1p+0,
        0x1.e442285231be1p-1,
        0x1.940a398f9cd23p-1,
        0x1.180996c77c8cap-1,
        0x1.f6c118574c83ep-3,
        -0x1.523eb8420f5f5p-4,
        -0x1.9b560b9f596eap-2,
        -0x1.5ac4a670a1cffp-1,
        -0x1.c24a622e3e9f9p-1,
        -0x1.f90459484f2b2p-1,
        -0x1.f90459484f2b2p-1,
        -0x1.c24a622e3e9f9p-1,
        -0x1.5ac4a670a1cffp-1,
        -0x1.9b560b9f596eap-2,
        -0x1.523eb8420f5f5p-4,
        0x1.f6c118574c83ep-3,
        0x1.180996c77c8cap-1,
        0x1.940a398f9cd23p-1,
        0x1.e442285231be1p-1,
    };
    static constexpr double s[19] = {
        0.0,
        0x1.4c7e04850cfaap-2,
        0x1.3a7a16b394423p-1,
        0x1.aca115aae3de4p-1,
        0x1.f0553b4de2e18p-1,
        0x1.fe40529a542aap-1,
        0x1.d4e03dd110b08p-1,
        0x1.78b0cdee73e0fp-1,
        0x1.e75ec0ded7beep-2,
        0x1.5116f7f2d58c5p-3,
        -0x1.5116f7f2d58c5p-3,
        -0x1.e75ec0ded7beep-2,
        -0x1.78b0cdee73e0fp-1,
        -0x1.d4e03dd110b08p-1,
        -0x1.fe40529a542aap-1,
        -0x1.f0553b4de2e18p-1,
        -0x1.aca115aae3de4p-1,
        -0x1.3a7a16b394423p-1,
        -0x1.4c7e04850cfaap-2,
    };
};

template <>
struct OddConst<23> {
    static constexpr double c[23] = {
        0x1p+0,
        0x1.ed037ea3d2dbbp-1,
        0x1.b57675cf309eep-1,
        0x1.5d779b07cfef7p-1,
        0x1.d71b4a0c5a6c8p-2,
        0x1.a0ad8bd1e2882p-3,
        -0x1.17855b599f3b9p-4,
        -0x1.56eaae597c776p-2,
        -0x1.2742a4a775cfbp-1,
        -0x1.8d2a07c16d46fp-1,
        -0x1.d59cb83ef99bcp-1,
        -0x1.fb3b3035aa6cdp-1,
        -0x1.fb3b3035aa6cdp-1,
        -0x1.d59cb83ef99bcp-1,
        -0x1.8d2a07c16d46fp-1,
        -0x1.2742a4a775cfbp-1,
        -0x1.56eaae597c776p-2,
        -0x1.17855b599f3b9p-4,
        0x1.a0ad8bd1e2882p-3,
        0x1.d71b4a0c5a6c8p-2,
        0x1.5d779b07cfef7p-1,
        0x1.b57675cf309eep-1,
        0x1.ed037ea3d2dbbp-1,
    };
    static constexpr double s[23] = {
        0.0,
        0x1.14459ad2be466p-2,
        0x1.0a06e851db7cap-1,
        0x1.763021aaa15dap-1,
        0x1.c698e42f47b09p-1,
        0x1.f54a827142577p-1,
        0x1.fece70dfd3efbp-1,
        0x1.e270060999288p-1,
        0x1.a249e0b897ca9p-1,
        0x1.431df5838f7efp-1,
        0x1.97f6748e524b2p-2,
        0x1.16de8a4564f0ap-3,
        -0x1.16de8a4564f0ap-3,
        -0x1.97f6748e524b2p-2,
        -0x1.431df5838f7efp-1,
        -0x1.a249e0b897ca9p-1,
        -0x1.e270060999288p-1,
        -0x1.fece70dfd3efbp-1,
        -0x1.f54a827142577p-1,
        -0x1.c698e42f47b09p-1,
        -0x1.763021aaa15dap-1,
        -0x1.0a06e851db7cap-1,
        -0x1.14459ad2be466p-2,
    };
};

template <>
struct OddConst<29> {
    static constexpr double c[29] = {
        0x1p+0,
        0x1.f4079c06c0992p-1,
        0x1.d0adb9b447ccfp-1,
        0x1.979982a38e65ap-1,
        0x1.4b76371208a62p-1,
        0x1.dfa67657e7608p-2,
        0x1.11f2f2e2f1e3bp-2,
        0x1.bb81853a18977p-5,
        -0x1.4b545c0234a71p-3,
        -0x1.7b057f20bf2e4p-2,
        -0x1.1f53e93956dbfp-1,
        -0x1.73b5ae5db4e10p-1,
        -0x1.b6b5fbd9f7255p-1,
        -0x1.e532cbe45c954p-1,
        -0x1.fcffa67b61650p-1,
        -0x1.fcffa67b61650p-1,
        -0x1.e532cbe45c954p-1,
        -0x1.b6b5fbd9f7255p-1,
        -0x1.73b5ae5db4e10p-1,
        -0x1.1f53e93956dbfp-1,
        -0x1.7b057f20bf2e4p-2,
        -0x1.4b545c0234a71p-3,
        0x1.bb81853a18977p-5,
        0x1.11f2f2e2f1e3bp-2,
        0x1.dfa67657e7608p-2,
        0x1.4b76371208a62p-1,
        0x1.979982a38e65ap-1,
        0x1.d0adb9b447ccfp-1,
        0x1.f4079c06c0992p-1,
    };
    static constexpr double s[29] = {
        0.0,
        0x1.b8426c12812bcp-3,
        0x1.adf7689c97b70p-2,
        0x1.35d9650d47852p-1,
        0x1.863a1ada0cfa6p-1,
        0x1.c45bb0d10918cp-1,
        0x1.ed566cb3dcba1p-1,
        0x1.ff3fc588e859dp-1,
        0x1.f941537248537p-1,
        0x1.dba2d62cb789fp-1,
        0x1.a7c6da34af89fp-1,
        0x1.601a24ba81342p-1,
        0x1.07f6acd7cdce2p-1,
        0x1.46f6faf5fcb72p-2,
        0x1.badb02034d9ffp-4,
        -0x1.badb02034d9ffp-4,
        -0x1.46f6faf5fcb72p-2,
        -0x1.07f6acd7cdce2p-1,
        -0x1.601a24ba81342p-1,
        -0x1.a7c6da34af89fp-1,
        -0x1.dba2d62cb789fp-1,
        -0x1.f941537248537p-1,
        -0x1.ff3fc588e859dp-1,
        -0x1.ed566cb3dcba1p-1,
        -0x1.c45bb0d10918cp-1,
        -0x1.863a1ada0cfa6p-1,
        -0x1.35d9650d47852p-1,
        -0x1.adf7689c97b70p-2,
        -0x1.b8426c12812bcp-3,
    };
};

template <>
struct OddConst<31> {
    static constexpr double c[31] = {
        0x1p+0,
        0x1.f584f2ce43b84p-1,
        0x1.d681a366a00fap-1,
        0x1.a43b1b1379affp-1,
        0x1.60c045a2e9729p-1,
        0x1.0ed45eea3b09fp-1,
        0x1.63a3fcfaca412p-2,
        0x1.361fc440b478fp-3,
        -0x1.9eeb01776b57dp-5,
        -0x1.00ab0eb2d7d94p-2,
        -0x1.c2f6af3928a8ep-2,
        -0x1.3965f49174d13p-1,
        -0x1.847bf1d5146ccp-1,
        -0x1.bfaa5c136b224p-1,
        -0x1.e884f0cc22cccp-1,
        -0x1.fd5f830f860f9p-1,
        -0x1.fd5f830f860f9p-1,
        -0x1.e884f0cc22cccp-1,
        -0x1.bfaa5c136b224p-1,
        -0x1.847bf1d5146ccp-1,
        -0x1.3965f49174d13p-1,
        -0x1.c2f6af3928a8ep-2,
        -0x1.00ab0eb2d7d94p-2,
        -0x1.9eeb01776b57dp-5,
        0x1.361fc440b478fp-3,
        0x1.63a3fcfaca412p-2,
        0x1.0ed45eea3b09fp-1,
        0x1.60c045a2e9729p-1,
        0x1.a43b1b1379affp-1,
        0x1.d681a366a00fap-1,
        0x1.f584f2ce43b84p-1,
    };
    static constexpr double s[31] = {
        0.0,
        0x1.9c4266041ca8fp-3,
        0x1.93d20572ca90bp-2,
        0x1.247d447a27216p-1,
        0x1.73180a4b0d300p-1,
        0x1.b2818007c19dfp-1,
        0x1.e0210c26a6e6fp-1,
        0x1.fa18852c3e08ap-1,
        0x1.ff57c5208ccf9p-1,
        0x1.efa7cddb128fap-1,
        0x1.cbad095f50378p-1,
        0x1.94e08eb13c451p-1,
        0x1.4d80b1ad9ccf6p-1,
        0x1.f0f2ff6705becp-2,
        0x1.328c3f1b322cbp-2,
        0x1.9e62aca53c49fp-4,
        -0x1.9e62aca53c49fp-4,
        -0x1.328c3f1b322cbp-2,
        -0x1.f0f2ff6705becp-2,
        -0x1.4d80b1ad9ccf6p-1,
        -0x1.94e08eb13c451p-1,
        -0x1.cbad095f50378p-1,
        -0x1.efa7cddb128fap-1,
        -0x1.ff57c5208ccf9p-1,
        -0x1.fa18852c3e08ap-1,
        -0x1.e0210c26a6e6fp-1,
        -0x1.b2818007c19dfp-1,
        -0x1.73180a4b0d300p-1,
        -0x1.247d447a27216p-1,
        -0x1.93d20572ca90bp-2,
        -0x1.9c4266041ca8fp-3,
    };
};

template <int R>
struct Butterfly {
    // The odd radices: the real form of the header comment.
    static inline void run(CV* a) {
        constexpr int H = (R - 1) / 2;
        CV t[H + 1], d[H + 1];
        for (int j = 1; j <= H; ++j) {
            t[j] = cadd(a[j], a[R - j]);
            d[j] = csub(a[j], a[R - j]);
        }
        CV y[R];
        y[0] = a[0];
        for (int j = 1; j <= H; ++j) y[0] = cadd(y[0], t[j]);
        for (int k = 1; k <= H; ++k) {
            CV A = a[0];
            for (int j = 1; j <= H; ++j) {
                const V c = vdupq_n_f64(OddConst<R>::c[(j * k) % R]);
                A = {vfmaq_f64(A.re, t[j].re, c), vfmaq_f64(A.im, t[j].im, c)};
            }
            const V s1 = vdupq_n_f64(OddConst<R>::s[k % R]);
            CV B = {vmulq_f64(d[1].re, s1), vmulq_f64(d[1].im, s1)};
            for (int j = 2; j <= H; ++j) {
                const V s = vdupq_n_f64(OddConst<R>::s[(j * k) % R]);
                B = {vfmaq_f64(B.re, d[j].re, s), vfmaq_f64(B.im, d[j].im, s)};
            }
            y[k] = {vaddq_f64(A.re, B.im), vsubq_f64(A.im, B.re)};  // A - i B
            y[R - k] = {vsubq_f64(A.re, B.im), vaddq_f64(A.im, B.re)};  // A + i B
        }
        for (int k = 0; k < R; ++k) a[k] = y[k];
    }
};

template <>
struct Butterfly<2> {
    static inline void run(CV* a) {
        const CV t = a[0];
        a[0] = cadd(t, a[1]);
        a[1] = csub(t, a[1]);
    }
};

template <>
struct Butterfly<4> {
    static inline void run(CV* a) {
        const CV t0 = cadd(a[0], a[2]), t1 = csub(a[0], a[2]);
        const CV t2 = cadd(a[1], a[3]), t3 = mul_neg_i(csub(a[1], a[3]));
        a[0] = cadd(t0, t2);
        a[2] = csub(t0, t2);
        a[1] = cadd(t1, t3);
        a[3] = csub(t1, t3);
    }
};

// Loads and stores. Swap: the inverse transform's exchange of real and imaginary parts.
template <bool Swap>
inline CV load_il2(const double* x) {  // two neighbouring complex numbers of interleaved data
    const float64x2x2_t v = vld2q_f64(x);
    return Swap ? CV{v.val[1], v.val[0]} : CV{v.val[0], v.val[1]};
}
template <bool Swap>
inline CV load_il1(const double* x) {  // one complex number in both lanes
    const V r = vld1q_dup_f64(x), i = vld1q_dup_f64(x + 1);
    return Swap ? CV{i, r} : CV{r, i};
}
template <bool Swap>
inline void store_il2(double* y, CV a) {
    const float64x2x2_t v = Swap ? float64x2x2_t{{a.im, a.re}} : float64x2x2_t{{a.re, a.im}};
    vst2q_f64(y, v);
}
template <bool Swap>
inline void store_il1(double* y, CV a) {  // the first lane
    y[0] = vgetq_lane_f64(Swap ? a.im : a.re, 0);
    y[1] = vgetq_lane_f64(Swap ? a.re : a.im, 0);
}
inline CV load_split2(const double* re, const double* im, int64_t i) { return {vld1q_f64(re + i), vld1q_f64(im + i)}; }
inline CV load_split1(const double* re, const double* im, int64_t i) {
    return {vld1q_dup_f64(re + i), vld1q_dup_f64(im + i)};
}
inline void store_split2(double* re, double* im, int64_t i, CV a) {
    vst1q_f64(re + i, a.re);
    vst1q_f64(im + i, a.im);
}
inline void store_split1(double* re, double* im, int64_t i, CV a) {
    re[i] = vgetq_lane_f64(a.re, 0);
    im[i] = vgetq_lane_f64(a.im, 0);
}

// A transform of length R in one stage (n = R): interleaved in, interleaved out.
template <int R, bool Swap>
void stage_single(const double* x, double* y) {
    CV a[R];
    for (int j = 0; j < R; ++j) a[j] = load_il1<Swap>(x + 2 * j);
    Butterfly<R>::run(a);
    for (int k = 0; k < R; ++k) store_il1<Swap>(y + 2 * k, a[k]);
}

// The first stage (s = 1, length L = n): interleaved input, split output; two neighbouring p per
// register, the last one alone when m is odd. Twiddles tw[(k - 1) m + p] (re and im arrays).
template <int R, bool Swap>
void stage_first(const double* x, double* yr, double* yi, int64_t m, const double* twr, const double* twi) {
    int64_t p = 0;
    for (; p + 1 < m; p += 2) {
        CV a[R];
        for (int j = 0; j < R; ++j) a[j] = load_il2<Swap>(x + 2 * (p + j * m));
        Butterfly<R>::run(a);
        for (int k = 1; k < R; ++k) {
            a[k] = cmul(a[k], vld1q_f64(twr + (k - 1) * m + p), vld1q_f64(twi + (k - 1) * m + p));
        }
        if constexpr (R == 2) {
            vst2q_f64(yr + 2 * p, (float64x2x2_t{{a[0].re, a[1].re}}));
            vst2q_f64(yi + 2 * p, (float64x2x2_t{{a[0].im, a[1].im}}));
        } else if constexpr (R == 3) {
            vst3q_f64(yr + 3 * p, (float64x2x3_t{{a[0].re, a[1].re, a[2].re}}));
            vst3q_f64(yi + 3 * p, (float64x2x3_t{{a[0].im, a[1].im, a[2].im}}));
        } else if constexpr (R == 4) {
            vst4q_f64(yr + 4 * p, (float64x2x4_t{{a[0].re, a[1].re, a[2].re, a[3].re}}));
            vst4q_f64(yi + 4 * p, (float64x2x4_t{{a[0].im, a[1].im, a[2].im, a[3].im}}));
        } else {
            for (int k = 0; k < R; ++k) {
                yr[R * p + k] = vgetq_lane_f64(a[k].re, 0);
                yi[R * p + k] = vgetq_lane_f64(a[k].im, 0);
                yr[R * p + R + k] = vgetq_lane_f64(a[k].re, 1);
                yi[R * p + R + k] = vgetq_lane_f64(a[k].im, 1);
            }
        }
    }
    if (p < m) {
        CV a[R];
        for (int j = 0; j < R; ++j) a[j] = load_il1<Swap>(x + 2 * (p + j * m));
        Butterfly<R>::run(a);
        for (int k = 1; k < R; ++k) {
            a[k] = cmul(a[k], vld1q_dup_f64(twr + (k - 1) * m + p), vld1q_dup_f64(twi + (k - 1) * m + p));
        }
        for (int k = 0; k < R; ++k) store_split1(yr, yi, R * p + k, a[k]);
    }
}

// A later stage (s > 1): split input; split output, or interleaved output (OutIL, the last stage:
// m = 1, no twiddles). Twiddles tw[p (R - 1) + k - 1], one broadcast pair per p (p = 0 has none).
template <int R, bool OutIL, bool Swap>
void stage_inner(
    const double* xr,
    const double* xi,
    double* yr,
    double* yi,
    double* yil,
    int64_t s,
    int64_t m,
    const double* twr,
    const double* twi
) {
    for (int64_t p = 0; p < m; ++p) {
        V wr[R], wi[R];
        if (p != 0) {
            for (int k = 1; k < R; ++k) {
                wr[k] = vdupq_n_f64(twr[p * (R - 1) + k - 1]);
                wi[k] = vdupq_n_f64(twi[p * (R - 1) + k - 1]);
            }
        }
        auto finish = [&](CV* a) {
            Butterfly<R>::run(a);
            if (p != 0) {
                for (int k = 1; k < R; ++k) a[k] = cmul(a[k], wr[k], wi[k]);
            }
        };
        int64_t q = 0;
        for (; q + 1 < s; q += 2) {
            CV a[R];
            for (int j = 0; j < R; ++j) a[j] = load_split2(xr, xi, q + s * (p + j * m));
            finish(a);
            for (int k = 0; k < R; ++k) {
                const int64_t at = q + s * (R * p + k);
                if constexpr (OutIL) {
                    store_il2<Swap>(yil + 2 * at, a[k]);
                } else {
                    store_split2(yr, yi, at, a[k]);
                }
            }
        }
        if (q < s) {
            CV a[R];
            for (int j = 0; j < R; ++j) a[j] = load_split1(xr, xi, q + s * (p + j * m));
            finish(a);
            for (int k = 0; k < R; ++k) {
                const int64_t at = q + s * (R * p + k);
                if constexpr (OutIL) {
                    store_il1<Swap>(yil + 2 * at, a[k]);
                } else {
                    store_split1(yr, yi, at, a[k]);
                }
            }
        }
    }
}

// ---- The two-line kernel (milestone FT6): the stages above with the two lanes of a register holding the same
// position of two lines (a, b), every position in turn, so each lane goes through exactly the operations the
// one-line kernel applies to that position (the same butterflies, the same twiddles, the first stage's p = 0
// twiddle 1 included and the later stages' p = 0 skipped): a line's bits do not depend on its partner or on
// which kernel ran it. The lines are read from and written to strided memory (an axis other than the last),
// position i of line a at a + i * stride (doubles); the stages in between work on split buffers of pairs.

// Two complex numbers, one from each of two interleaved lines, as one split pair (lane 0: a, lane 1: b).
template <bool Swap>
inline CV load_pair(const double* a, const double* b) {
    const V va = vld1q_f64(a), vb = vld1q_f64(b);
    const CV v = {vzip1q_f64(va, vb), vzip2q_f64(va, vb)};
    return Swap ? CV{v.im, v.re} : v;
}

// Position i of lines a and b (strided, interleaved complex128) as a pair; Adj: b = a + 2 (neighbouring lines).
template <bool Swap, bool Adj>
inline CV load_lines(const double* a, const double* b, int64_t off) {
    if constexpr (Adj) {
        const float64x2x2_t v = vld2q_f64(a + off);
        return Swap ? CV{v.val[1], v.val[0]} : CV{v.val[0], v.val[1]};
    } else {
        return load_pair<Swap>(a + off, b + off);
    }
}

// The output side: times s (one rounding; the factor of the normalisation) and conjugated when asked, then
// stored to position off of lines a and b (b may be a: the same values twice).
template <bool Swap, bool Adj>
inline void store_lines(double* a, double* b, int64_t off, CV v, double s, bool conj) {
    if (s != 1.0) v = {vmulq_n_f64(v.re, s), vmulq_n_f64(v.im, s)};
    if (Swap) v = {v.im, v.re};
    if (conj) v.im = vnegq_f64(v.im);
    if constexpr (Adj) {
        vst2q_f64(a + off, (float64x2x2_t{{v.re, v.im}}));
    } else {
        vst1q_f64(a + off, vzip1q_f64(v.re, v.im));
        vst1q_f64(b + off, vzip2q_f64(v.re, v.im));
    }
}

// Split pair buffers: position i at 2 i of the re and im arrays (two lanes each).
inline CV load_pairbuf(const double* re, const double* im, int64_t i) {
    return {vld1q_f64(re + 2 * i), vld1q_f64(im + 2 * i)};
}
inline void store_pairbuf(double* re, double* im, int64_t i, CV v) {
    vst1q_f64(re + 2 * i, v.re);
    vst1q_f64(im + 2 * i, v.im);
}

struct Lines {  // the two lines of a pass: input and output pointers and position strides (doubles)
    const double *xa, *xb;
    double *ya, *yb;
    int64_t xs, ys;
    double scale;
    bool conj;
};

template <int R, bool Swap, bool Adj>
void pair_single(const Lines& L) {
    CV a[R];
    for (int j = 0; j < R; ++j) a[j] = load_lines<Swap, Adj>(L.xa, L.xb, j * L.xs);
    Butterfly<R>::run(a);
    for (int k = 0; k < R; ++k) store_lines<Swap, Adj>(L.ya, L.yb, k * L.ys, a[k], L.scale, L.conj);
}

// The first stage (s = 1): lines in, pair buffer out; every p takes its twiddles (p = 0: the table's 1).
template <int R, bool Swap, bool Adj>
void pair_first(const Lines& L, double* yr, double* yi, int64_t m, const double* twr, const double* twi) {
    for (int64_t p = 0; p < m; ++p) {
        CV a[R];
        for (int j = 0; j < R; ++j) a[j] = load_lines<Swap, Adj>(L.xa, L.xb, (p + j * m) * L.xs);
        Butterfly<R>::run(a);
        for (int k = 1; k < R; ++k) {
            a[k] = cmul(a[k], vld1q_dup_f64(twr + (k - 1) * m + p), vld1q_dup_f64(twi + (k - 1) * m + p));
        }
        for (int k = 0; k < R; ++k) store_pairbuf(yr, yi, R * p + k, a[k]);
    }
}

// A middle stage: pair buffers in and out, twiddles from p = 1 on.
template <int R>
void pair_inner(
    const double* xr,
    const double* xi,
    double* yr,
    double* yi,
    int64_t s,
    int64_t m,
    const double* twr,
    const double* twi
) {
    for (int64_t p = 0; p < m; ++p) {
        V wr[R], wi[R];
        if (p != 0) {
            for (int k = 1; k < R; ++k) {
                wr[k] = vdupq_n_f64(twr[p * (R - 1) + k - 1]);
                wi[k] = vdupq_n_f64(twi[p * (R - 1) + k - 1]);
            }
        }
        for (int64_t q = 0; q < s; ++q) {
            CV a[R];
            for (int j = 0; j < R; ++j) a[j] = load_pairbuf(xr, xi, q + s * (p + j * m));
            Butterfly<R>::run(a);
            if (p != 0) {
                for (int k = 1; k < R; ++k) a[k] = cmul(a[k], wr[k], wi[k]);
            }
            for (int k = 0; k < R; ++k) store_pairbuf(yr, yi, q + s * (R * p + k), a[k]);
        }
    }
}

// The last stage (m = 1, no twiddles): pair buffer in, lines out.
template <int R, bool Swap, bool Adj>
void pair_last(const double* xr, const double* xi, const Lines& L, int64_t s) {
    for (int64_t q = 0; q < s; ++q) {
        CV a[R];
        for (int j = 0; j < R; ++j) a[j] = load_pairbuf(xr, xi, q + s * j);
        Butterfly<R>::run(a);
        for (int k = 0; k < R; ++k) store_lines<Swap, Adj>(L.ya, L.yb, (q + s * k) * L.ys, a[k], L.scale, L.conj);
    }
}

}  // namespace detail

// The radices (decision FT35; milestone FT6 added the primes 13 to 31, decision FT64).
inline constexpr int kOddRadices[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};

// The radices of a mixed-radix plan of length n (the order of the header comment), or an empty list
// when n has a prime factor above 31 (or n = 1, which needs no stage).
inline std::vector<int> mixed_radices(int64_t n) {
    std::vector<int> out;
    if (n < 2) return out;
    int64_t rest = n;
    int twos = 0;
    while (rest % 2 == 0) rest /= 2, ++twos;
    for (; twos >= 2; twos -= 2) out.push_back(4);
    if (twos == 1) out.push_back(2);
    for (int p : kOddRadices) {
        while (rest % p == 0) rest /= p, out.push_back(p);
    }
    if (rest != 1) out.clear();
    return out;
}

// True when n is 31-smooth (n >= 1): the lengths the mixed radix transforms directly.
inline bool is_mixed_length(int64_t n) { return n == 1 || !mixed_radices(n).empty(); }

// The four-step's n1 for a 31-smooth n >= kFourStepMin: the largest divisor of n at most sqrt n.
inline int64_t four_step_n1(int64_t n) {
    int64_t best = 1;
    for (int64_t d = 2; d <= n / d; ++d) {
        if (n % d == 0) best = d;
    }
    return best;
}

namespace detail {

// A four-step task's buffer: up to kKeep doubles the thread's own (one per slot and nesting depth, kept from
// call to call: a fresh allocation per task costs page faults the threads serialise on, decision FT50),
// a larger one allocated for the task. A task runs on one thread from start to end.
class FsBuffer {
public:
    FsBuffer(int slot, int64_t count) {
        constexpr int64_t kKeep = int64_t{1} << 17;
        struct Kept {
            double* mem[6] = {};
            int64_t cap[6] = {};
            ~Kept() {
                for (double* p : mem) std::free(p);
            }
        };
        thread_local Kept kept;
        if (count <= kKeep && slot < 6) {
            if (kept.cap[slot] < count) {
                double* const p = alloc(count);  // (may throw: the old buffer stays)
                std::free(kept.mem[slot]);
                kept.mem[slot] = p, kept.cap[slot] = count;
            }
            p_ = kept.mem[slot];
        } else {
            own_ = alloc(count);
            p_ = own_;
        }
    }
    ~FsBuffer() { std::free(own_); }
    FsBuffer(const FsBuffer&) = delete;
    FsBuffer& operator=(const FsBuffer&) = delete;
    double* get() const { return p_; }

private:
    static double* alloc(int64_t count) {
        void* p = std::aligned_alloc(128, (static_cast<size_t>(count) * sizeof(double) + 127) / 128 * 128);
        if (p == nullptr) throw std::bad_alloc();
        return static_cast<double*>(p);
    }
    double* p_ = nullptr;
    double* own_ = nullptr;
};

}  // namespace detail

// The plan of one 31-smooth length: the radices and each stage's twiddle table, or (n >= kFourStepMin) the
// four-step's two plans and its twiddle table. Read-only after construction, so any number of threads may
// run it at once.
class CfftPlan {
public:
    explicit CfftPlan(int64_t n, int depth = 0) : n_(n), depth_(depth) {
        if (n >= kFourStepMin) {
            const int64_t n1 = four_step_n1(n), n2 = n / n1;
            col_ = std::make_unique<CfftPlan>(n1, depth + 1);
            row_ = std::make_unique<CfftPlan>(n2, depth + 1);
            radices_ = col_->radices_;
            radices_.insert(radices_.end(), row_->radices_.begin(), row_->radices_.end());
            ftw_.resize(static_cast<size_t>(2 * n));
            for (int64_t j0 = 0; j0 < n2; j0 += kFourStepCols) {  // by column block, as run_four_step reads it
                const int64_t g = std::min(kFourStepCols, n2 - j0);
                for (int64_t k1 = 0; k1 < n1; ++k1) {
                    for (int64_t c = 0; c < g; ++c) {
                        const Twiddle w = twiddle((j0 + c) * k1, n);
                        const size_t at = static_cast<size_t>(2 * (j0 * n1 + k1 * g + c));
                        ftw_[at] = w.re, ftw_[at + 1] = w.im;
                    }
                }
            }
            return;
        }
        radices_ = mixed_radices(n);
        int64_t s = 1;
        for (size_t i = 0; i < radices_.size(); ++i) {
            const int r = radices_[i];
            const int64_t L = n / s, m = L / r;
            Table t;
            t.r = r, t.s = s, t.m = m;
            const bool has = i == 0 ? m > 1 || radices_.size() > 1 : m > 1;
            if (has) {
                t.re.resize(static_cast<size_t>(m * (r - 1)));
                t.im.resize(t.re.size());
                for (int64_t p = 0; p < m; ++p) {
                    for (int k = 1; k < r; ++k) {
                        const Twiddle w = twiddle(p * k * s, n);  // w_L^(p k) = e^(-2 pi i p k s / n)
                        const size_t at =
                            i == 0 ? static_cast<size_t>((k - 1) * m + p) : static_cast<size_t>(p * (r - 1) + k - 1);
                        t.re[at] = w.re, t.im[at] = w.im;
                    }
                }
            }
            stages_.push_back(std::move(t));
            s *= r;
        }
    }

    int64_t n() const { return n_; }
    const std::vector<int>& radices() const { return radices_; }
    bool four_step() const { return col_ != nullptr; }
    int64_t n1() const { return col_ ? col_->n_ : 0; }

    // y = DFT(x) (the sign -; inverse: the sign +, no 1/n), x and y interleaved complex128 of n
    // entries each, not overlapping; work: 2n doubles, overlapping neither.
    void execute(const double* x, double* y, double* work, bool inverse) const {
        if (inverse) {
            run<true>(x, y, work);
        } else {
            run<false>(x, y, work);
        }
    }
    void forward(const double* x, double* y, double* work) const { run<false>(x, y, work); }

    // The two-line kernel (detail::Lines; not for a four-step plan): lines L.xa, L.xb (L.xb may be L.xa: one
    // line) transformed into L.ya, L.yb, times L.scale and conjugated with L.conj, each line's bits those of
    // execute() followed by that scaling. The input is read whole before any output is written, so the lines
    // may be transformed in place. work: 8n doubles.
    void execute_lines(const detail::Lines& L, double* work, bool inverse) const {
        const bool adj = L.xb == L.xa + 2 && L.yb == L.ya + 2;
        if (inverse) {
            adj ? run_lines<true, true>(L, work) : run_lines<true, false>(L, work);
        } else {
            adj ? run_lines<false, true>(L, work) : run_lines<false, false>(L, work);
        }
    }

    // Bytes the plan holds (its twiddle tables).
    int64_t bytes() const {
        int64_t b = static_cast<int64_t>(sizeof(*this) + ftw_.size() * sizeof(double));
        for (const Table& t : stages_) b += static_cast<int64_t>(2 * t.re.size() * sizeof(double) + sizeof(Table));
        if (col_) b += col_->bytes() + row_->bytes();
        return b;
    }

private:
    struct Table {
        int r = 0;
        int64_t s = 0, m = 0;
        std::vector<double> re, im;
    };

    // The line stride (doubles) of a four-step block buffer of lines of `len` complex numbers: a multiple of
    // 16 doubles, off the powers of two (the lines of a block in different cache sets).
    static int64_t fs_stride(int64_t len) { return (2 * len + 15) / 16 * 16 + 16; }

    // The four-step of the header comment. work (2n doubles) holds the twiddled column transforms by column
    // block: block j0 (columns j0 .. j0 + g - 1) at 2 j0 n1, its n1 rows of g entries one after the other (the
    // twiddle table is laid out alike), so the column pass writes and reads sequentially; the row pass gathers
    // a block of rows from every column block.
    template <bool Swap>
    [[gnu::noinline]] void run_four_step(const double* x, double* y, double* work) const {
        const int64_t n1 = col_->n_, n2 = row_->n_, Gc = kFourStepCols, Gr = kFourStepRows;
        const int64_t ldb = fs_stride(n1), ldd = fs_stride(n2);
        const int slot = 3 * depth_;
        double* const T = work;
        const double* const tw = ftw_.data();
        const CfftPlan& col = *col_;
        const CfftPlan& row = *row_;
        // the columns j0 .. j0 + g - 1: gather, transform, twiddle, write their block of T
        auto columns = [&](int64_t b) {
            const int64_t j0 = b * Gc, g = std::min(Gc, n2 - j0);
            const detail::FsBuffer bb(slot, Gc * ldb), cb(slot + 1, Gc * ldb), wb(slot + 2, 2 * n1);
            double* const B = bb.get();
            double* const C = cb.get();
            for (int64_t j1 = 0; j1 < n1; ++j1) {
                const double* src = x + 2 * (j1 * n2 + j0);
                for (int64_t c = 0; c < g; ++c) {
                    const float64x2_t v = vld1q_f64(src + 2 * c);
                    vst1q_f64(B + c * ldb + 2 * j1, Swap ? vextq_f64(v, v, 1) : v);
                }
            }
            for (int64_t c = 0; c < g; ++c) col.run<false>(B + c * ldb, C + c * ldb, wb.get());
            double* const Tb = T + 2 * j0 * n1;
            const double* const wbk = tw + 2 * j0 * n1;
            for (int64_t c = 0; c < g; ++c) vst1q_f64(Tb + 2 * c, vld1q_f64(C + c * ldb));  // k1 = 0: no twiddle
            for (int64_t k1 = 1; k1 < n1; ++k1) {
                double* const dst = Tb + 2 * k1 * g;
                const double* const w = wbk + 2 * k1 * g;
                int64_t c = 0;
                if (j0 == 0) {  // j2 = 0: no twiddle
                    vst1q_f64(dst, vld1q_f64(C + 2 * k1));
                    c = 1;
                }
                for (; c + 1 < g; c += 2) {
                    const detail::CV a = detail::load_pair<false>(C + c * ldb + 2 * k1, C + (c + 1) * ldb + 2 * k1);
                    const float64x2x2_t wv = vld2q_f64(w + 2 * c);
                    detail::store_il2<false>(dst + 2 * c, detail::cmul(a, wv.val[0], wv.val[1]));
                }
                if (c < g) {
                    const detail::CV a = detail::load_il1<false>(C + c * ldb + 2 * k1);
                    const detail::CV r = detail::cmul(a, vld1q_dup_f64(w + 2 * c), vld1q_dup_f64(w + 2 * c + 1));
                    detail::store_il1<false>(dst + 2 * c, r);
                }
            }
        };
        // the rows k0 .. k0 + g - 1: gather from every column block, transform, scatter into y
        auto rows = [&](int64_t b) {
            const int64_t k0 = b * Gr, g = std::min(Gr, n1 - k0);
            const detail::FsBuffer db(slot, Gr * ldd), eb(slot + 1, Gr * ldd), wb(slot + 2, 2 * n2);
            double* const D = db.get();
            double* const E = eb.get();
            for (int64_t j0 = 0; j0 < n2; j0 += Gc) {
                const int64_t gj = std::min(Gc, n2 - j0);
                const double* const src = T + 2 * (j0 * n1 + k0 * gj);
                for (int64_t c = 0; c < g; ++c) {
                    for (int64_t e = 0; e < gj; ++e)
                        vst1q_f64(D + c * ldd + 2 * (j0 + e), vld1q_f64(src + 2 * (c * gj + e)));
                }
            }
            for (int64_t c = 0; c < g; ++c) row.run<false>(D + c * ldd, E + c * ldd, wb.get());
            for (int64_t k2 = 0; k2 < n2; ++k2) {
                double* const dst = y + 2 * (k0 + n1 * k2);
                for (int64_t c = 0; c < g; ++c) {
                    const float64x2_t v = vld1q_f64(E + c * ldd + 2 * k2);
                    vst1q_f64(dst + 2 * c, Swap ? vextq_f64(v, v, 1) : v);
                }
            }
        };
        const auto ns = [](int64_t len, int64_t g) {  // one block's time on one core, roughly
            int64_t l = 1;
            while ((int64_t{1} << l) < len) ++l;
            return g * len * l / 4;
        };
        run_tasks((n2 + Gc - 1) / Gc, columns, ns(n1, Gc));
        run_tasks((n1 + Gr - 1) / Gr, rows, ns(n2, Gr));
    }

    template <bool Swap, bool Adj>
    void run_lines(const detail::Lines& L, double* work) const {
        const int64_t n = n_;
        const int S = static_cast<int>(stages_.size());
        if (S == 0) {  // n = 1
            const detail::CV v = detail::load_lines<false, Adj>(L.xa, L.xb, 0);
            detail::store_lines<false, Adj>(L.ya, L.yb, 0, v, L.scale, L.conj);
            return;
        }
        if (S == 1) {
            switch (stages_[0].r) {
                case 2: detail::pair_single<2, Swap, Adj>(L); break;
                case 3: detail::pair_single<3, Swap, Adj>(L); break;
                case 4: detail::pair_single<4, Swap, Adj>(L); break;
                case 5: detail::pair_single<5, Swap, Adj>(L); break;
                case 7: detail::pair_single<7, Swap, Adj>(L); break;
                case 11: detail::pair_single<11, Swap, Adj>(L); break;
                case 13: detail::pair_single<13, Swap, Adj>(L); break;
                case 17: detail::pair_single<17, Swap, Adj>(L); break;
                case 19: detail::pair_single<19, Swap, Adj>(L); break;
                case 23: detail::pair_single<23, Swap, Adj>(L); break;
                case 29: detail::pair_single<29, Swap, Adj>(L); break;
                default: detail::pair_single<31, Swap, Adj>(L); break;
            }
            return;
        }
        double* const A[2] = {work, work + 2 * n};
        double* const B[2] = {work + 4 * n, work + 6 * n};
        const double* in_r = nullptr;
        const double* in_i = nullptr;
        for (int i = 0; i < S; ++i) {
            const Table& t = stages_[static_cast<size_t>(i)];
            double* const* out = (i % 2 == 0) ? A : B;
            const double* wr = t.re.data();
            const double* wi = t.im.data();
            if (i == 0) {
                switch (t.r) {
                    case 2: detail::pair_first<2, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 3: detail::pair_first<3, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 4: detail::pair_first<4, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 5: detail::pair_first<5, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 7: detail::pair_first<7, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 11: detail::pair_first<11, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 13: detail::pair_first<13, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 17: detail::pair_first<17, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 19: detail::pair_first<19, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 23: detail::pair_first<23, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    case 29: detail::pair_first<29, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                    default: detail::pair_first<31, Swap, Adj>(L, out[0], out[1], t.m, wr, wi); break;
                }
            } else if (i == S - 1) {
                switch (t.r) {
                    case 2: detail::pair_last<2, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 3: detail::pair_last<3, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 4: detail::pair_last<4, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 5: detail::pair_last<5, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 7: detail::pair_last<7, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 11: detail::pair_last<11, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 13: detail::pair_last<13, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 17: detail::pair_last<17, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 19: detail::pair_last<19, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 23: detail::pair_last<23, Swap, Adj>(in_r, in_i, L, t.s); break;
                    case 29: detail::pair_last<29, Swap, Adj>(in_r, in_i, L, t.s); break;
                    default: detail::pair_last<31, Swap, Adj>(in_r, in_i, L, t.s); break;
                }
            } else {
                switch (t.r) {
                    case 2: detail::pair_inner<2>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 3: detail::pair_inner<3>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 4: detail::pair_inner<4>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 5: detail::pair_inner<5>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 7: detail::pair_inner<7>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 11: detail::pair_inner<11>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 13: detail::pair_inner<13>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 17: detail::pair_inner<17>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 19: detail::pair_inner<19>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 23: detail::pair_inner<23>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    case 29: detail::pair_inner<29>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                    default: detail::pair_inner<31>(in_r, in_i, out[0], out[1], t.s, t.m, wr, wi); break;
                }
            }
            in_r = out[0], in_i = out[1];
        }
    }

    template <bool Swap>
    void run(const double* x, double* y, double* work) const {
        if (col_) {
            run_four_step<Swap>(x, y, work);
            return;
        }
        const int64_t n = n_;
        const int S = static_cast<int>(stages_.size());
        if (S == 0) {  // n = 1
            y[0] = x[0], y[1] = x[1];
            return;
        }
        if (S == 1) {
            switch (stages_[0].r) {
                case 2: detail::stage_single<2, Swap>(x, y); break;
                case 3: detail::stage_single<3, Swap>(x, y); break;
                case 4: detail::stage_single<4, Swap>(x, y); break;
                case 5: detail::stage_single<5, Swap>(x, y); break;
                case 7: detail::stage_single<7, Swap>(x, y); break;
                case 11: detail::stage_single<11, Swap>(x, y); break;
                case 13: detail::stage_single<13, Swap>(x, y); break;
                case 17: detail::stage_single<17, Swap>(x, y); break;
                case 19: detail::stage_single<19, Swap>(x, y); break;
                case 23: detail::stage_single<23, Swap>(x, y); break;
                case 29: detail::stage_single<29, Swap>(x, y); break;
                default: detail::stage_single<31, Swap>(x, y); break;
            }
            return;
        }
        double* const A[2] = {y, y + n};  // the output's memory as a split buffer
        double* const B[2] = {work, work + n};  // the work buffer
        const double* in_r = nullptr;
        const double* in_i = nullptr;
        for (int i = 0; i < S; ++i) {
            const Table& t = stages_[static_cast<size_t>(i)];
            double* const* out = ((S - 2 - i) % 2 == 0) ? B : A;  // the last stage reads B
            if (i == 0) {
                first<Swap>(t, x, out[0], out[1]);
            } else if (i == S - 1) {
                inner<true, Swap>(t, in_r, in_i, nullptr, nullptr, y);
            } else {
                inner<false, false>(t, in_r, in_i, out[0], out[1], nullptr);
            }
            in_r = out[0], in_i = out[1];
        }
    }

    template <bool Swap>
    static void first(const Table& t, const double* x, double* yr, double* yi) {
        const double* wr = t.re.data();
        const double* wi = t.im.data();
        switch (t.r) {
            case 2: detail::stage_first<2, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 3: detail::stage_first<3, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 4: detail::stage_first<4, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 5: detail::stage_first<5, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 7: detail::stage_first<7, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 11: detail::stage_first<11, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 13: detail::stage_first<13, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 17: detail::stage_first<17, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 19: detail::stage_first<19, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 23: detail::stage_first<23, Swap>(x, yr, yi, t.m, wr, wi); break;
            case 29: detail::stage_first<29, Swap>(x, yr, yi, t.m, wr, wi); break;
            default: detail::stage_first<31, Swap>(x, yr, yi, t.m, wr, wi); break;
        }
    }

    template <bool OutIL, bool Swap>
    static void inner(const Table& t, const double* xr, const double* xi, double* yr, double* yi, double* yil) {
        const double* wr = t.re.data();
        const double* wi = t.im.data();
        switch (t.r) {
            case 2: detail::stage_inner<2, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 3: detail::stage_inner<3, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 4: detail::stage_inner<4, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 5: detail::stage_inner<5, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 7: detail::stage_inner<7, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 11: detail::stage_inner<11, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 13: detail::stage_inner<13, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 17: detail::stage_inner<17, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 19: detail::stage_inner<19, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 23: detail::stage_inner<23, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            case 29: detail::stage_inner<29, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
            default: detail::stage_inner<31, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
        }
    }

    int64_t n_;
    int depth_;  // the four-step's nesting depth (its tasks' buffer slots)
    std::vector<int> radices_;
    std::vector<Table> stages_;
    std::unique_ptr<CfftPlan> col_, row_;  // the four-step: the plans of n1 and n2
    std::vector<double> ftw_;  // the four-step: w_n^(j2 k1) at k1 n2 + j2, interleaved
};

}  // namespace sd::fft
