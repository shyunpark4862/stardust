// The library's own mixed-radix complex FFT (docs/design/fft.md §3.3): a Stockham transform with NEON
// butterflies of radix 2, 3, 4, 5, 7 and 11, for every length whose prime factors are at most 11 (the
// 11-smooth lengths: next_fast_len's set, decision FT18). Other lengths go through Bluestein
// (fft/bluestein.hpp), whose inner transform is one of these.
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
// then 3, 5, 7, 11. Inverse transforms swap the real and imaginary parts on the first load and on the
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
// is within (kappa log2 n + O(u)) u ||y|| <= kappa L(n) u ||y||. Radix 13 would need 8.22 u: lengths
// with a prime factor above 11 go to Bluestein (decision FT36).
#pragma once

#include <arm_neon.h>

#include <cstdint>
#include <vector>

#include "fft/twiddle.hpp"

namespace sd::fft {

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
            y[k] = {vaddq_f64(A.re, B.im), vsubq_f64(A.im, B.re)};      // A - i B
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

}  // namespace detail

// The radices of a mixed-radix plan of length n (the order of the header comment), or an empty list
// when n has a prime factor above 11 (or n = 1, which needs no stage).
inline std::vector<int> mixed_radices(int64_t n) {
    std::vector<int> out;
    if (n < 2) return out;
    int64_t rest = n;
    int twos = 0;
    while (rest % 2 == 0) rest /= 2, ++twos;
    for (; twos >= 2; twos -= 2) out.push_back(4);
    if (twos == 1) out.push_back(2);
    for (int p : {3, 5, 7, 11}) {
        while (rest % p == 0) rest /= p, out.push_back(p);
    }
    if (rest != 1) out.clear();
    return out;
}

// True when n is 11-smooth (n >= 1): the lengths the mixed radix transforms directly.
inline bool is_mixed_length(int64_t n) { return n == 1 || !mixed_radices(n).empty(); }

// The plan of one 11-smooth length: the radices and each stage's twiddle table. Read-only after
// construction, so any number of threads may run it at once.
class CfftPlan {
public:
    explicit CfftPlan(int64_t n) : n_(n), radices_(mixed_radices(n)) {
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

    // Bytes the plan holds (its twiddle tables).
    int64_t bytes() const {
        int64_t b = static_cast<int64_t>(sizeof(*this));
        for (const Table& t : stages_) b += static_cast<int64_t>(2 * t.re.size() * sizeof(double) + sizeof(Table));
        return b;
    }

private:
    struct Table {
        int r = 0;
        int64_t s = 0, m = 0;
        std::vector<double> re, im;
    };

    template <bool Swap>
    void run(const double* x, double* y, double* work) const {
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
                default: detail::stage_single<11, Swap>(x, y); break;
            }
            return;
        }
        double* const A[2] = {y, y + n};        // the output's memory as a split buffer
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
            default: detail::stage_first<11, Swap>(x, yr, yi, t.m, wr, wi); break;
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
            default: detail::stage_inner<11, OutIL, Swap>(xr, xi, yr, yi, yil, t.s, t.m, wr, wi); break;
        }
    }

    int64_t n_;
    std::vector<int> radices_;
    std::vector<Table> stages_;
};

}  // namespace sd::fft
