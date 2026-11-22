// The discrete cosine and sine transforms of types 1-4 (docs/design/fft.md §3.5, milestone FT4): one line
// of N reals on the real FFT of fft/plan.hpp. vDSP's DCT is single precision only (§3.1), so every type is
// this library's own code. The definitions are scipy's (norm='backward', unnormalised):
//
//     DCT-I    y_k = x_0 + (-1)^k x_(N-1) + 2 sum_(n=1)^(N-2) x_n cos(pi k n / (N - 1))       (N >= 2)
//     DCT-II   y_k = 2 sum x_n cos(pi k (2n + 1) / 2N)
//     DCT-III  y_k = x_0 + 2 sum_(n>=1) x_n cos(pi n (2k + 1) / 2N)
//     DCT-IV   y_k = 2 sum x_n cos(pi (2k + 1) (2n + 1) / 4N)
//     DST-I    y_k = 2 sum x_n sin(pi (k + 1) (n + 1) / (N + 1))
//     DST-II   y_k = 2 sum x_n sin(pi (k + 1) (2n + 1) / 2N)
//     DST-III  y_k = (-1)^k x_(N-1) + 2 sum_(n<=N-2) x_n sin(pi (2k + 1) (n + 1) / 2N)
//     DST-IV   y_k = 2 sum x_n sin(pi (2k + 1) (2n + 1) / 4N)
//
// The algorithms (each depends on N alone):
//   DCT-I    the real FFT of the even extension v = (x_0 .. x_(N-1), x_(N-2) .. x_1), length 2 (N - 1);
//            y_k = Re V_k (the real kind of 2 (N - 1)).
//   DST-I    the real FFT of the odd extension (0, x_0 .. x_(N-1), 0, -x_(N-1) .. -x_0), length 2 (N + 1);
//            y_k = -Im V_(k+1) (the real kind of 2 (N + 1)).
//   DCT-II   Makhoul's order v = (x_0, x_2, .., x_3, x_1) (the even entries forward, the odd ones backward),
//            the real FFT V of length N, y_k = 2 Re(e^(-i pi k / 2N) V_k): with (c, s) = (cos, sin)(pi k / 2N),
//            y_k = 2 (c Re V_k + s Im V_k) and y_(N-k) = 2 (s Re V_k - c Im V_k) from one V_k.
//   DCT-III  its reverse: V_0 = x_0, V_k = (x_k - i x_(N-k)) e^(+i pi k / 2N) (a Hermitian half spectrum;
//            x_(N-k) = x_k at k = N / 2), the unnormalised inverse real FFT v of length N, y_2j = v_j,
//            y_(2j+1) = v_(N-1-j).
//   DCT-IV, even N: z_j = (x_2j + i x_(N-1-2j)) w_j with w_j = e^(-i pi (8j + 1) / 8N), the complex FFT Z of
//            length N / 2, u_p = Z_p w_p, y_2p = 2 Re u_p and y_(N-1-2p) = -2 Im u_p. (The phase
//            pi (2k + 1)(2n + 1) / 4N at k = 2p, n = 2j is 2 pi p j / (N / 2) plus the two halves
//            pi (8p + 1) / 8N + pi (8j + 1) / 8N; the odd entries' cosines are the sines of that phase.)
//   DCT-IV, odd N: one real FFT of length N on a signed permutation of x (dct4_odd_tables). With
//            a = 2k + 1, b = 2n + 1 and 1 / 8N = alpha / 8 + beta / N (alpha N + 8 beta = 1; N is odd), the
//            phase 2 pi ab / 8N is A + B with A = 2 pi alpha ab / 8 and B = 2 pi beta ab / N. cos A and
//            sin A are chi1(alpha ab) / sqrt 2 and chi1 chi4(alpha ab) / sqrt 2 for the multiplicative
//            characters chi1 (+1 at r = 1, 7 mod 8, -1 at 3, 5) and chi4 (+1 at r = 1 mod 4, -1 at 3), so
//                y_k = sqrt 2 chi1(alpha a) sum_n chi1(b) x_n (cos B - chi4(alpha a) chi4(b) sin B).
//            Put u_j = chi1(b) x_n at j = b mod N when chi4(b) = 1 and at j = -b mod N when chi4(b) = -1
//            (b -> 2N - b keeps chi4 and is an involution, so j is a bijection) and U = DFT_N(u):
//                y_k = sqrt 2 chi1(alpha a) (Re U_l + chi4(alpha a) Im U_l),   l = beta a mod N,
//            with U_l = conj U_(N-l) above N / 2 (the real FFT's half spectrum; the odd real path of N).
//   DST-II, DST-III, DST-IV  the DCT of the same type with exact sign changes and reversals:
//            DST-II(x)_k = DCT-II((-1)^n x_n)_(N-1-k), DST-III(x)_k = (-1)^k DCT-III(x_(N-1-n))_k,
//            DST-IV(x)_k = (-1)^k DCT-IV(x_(N-1-n))_k. So a DST's bits are those of its DCT (§4.5).
//
// Normalisation and orthogonalize (docs/design/fft.md §2.6): the result is multiplied by one factor s
// (1, 1/N', fl(1/fl(sqrt N')) with N' the logical length 2 (N - 1), 2 (N + 1) or 2N); orthogonalize
// multiplies the input's x_0 (DCT-I: and x_(N-1); DCT-III) by fl(sqrt 2) before the transform and divides
// the result's y_0 (DCT-I: and y_(N-1); DCT-II) by fl(sqrt 2) after the factor, each one rounding of its
// own (the DSTs' x_(N-1), y_(N-1) through the reversal). DCT-IV and DST-I have none.
//
// Error (docs/design/fft.md §4.2: ||y_hat - y|| <= (kappa L(M) + c2) u ||T|| ||x|| + tau, M = 2N + 2). The
// real FFT inside holds its own contract on the whole spectrum ((kappa L(n) + c0) u sqrt(n) ||v||,
// decision FT45), and each map from that spectrum to y is a multiple of an isometry on Hermitian spectra
// no larger than ||T|| / sqrt(n) ||v||: DCT-I takes the real parts of the first N entries (||v|| <= sqrt 2
// ||x||, ||T|| <= 2 sqrt(N - 1)), DST-I the imaginary parts of N entries whose mirror images carry the same
// error (so half the whole spectrum's), DCT-II 2 Re(w_k V_k) (2 times an isometry, ||T|| = 2 sqrt N), DCT-III
// the inverse with ||V||_H <= sqrt 2 ||x||, the odd DCT-IV sqrt 2 times an isometry (||T|| = sqrt(2N)), the
// even DCT-IV the complex FFT of N / 2 between two unit diagonals. The inner length is at most M and its
// level count at most L(M) - 1, which leaves kappa u, and the pre- and post-processing (one twiddle
// product, 2u + mu = 3.5u, a sum and a sqrt 2 product) stays inside c2 = 8 with it. An inner length with a
// radix of 13 to 31 takes kappa = 17 (decision FT64), one with a prime factor above 31 runs Bluestein: the
// bound takes its rho and inner length (decision FT53).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "fft/plan.hpp"
#include "fft/rfft.hpp"
#include "fft/twiddle.hpp"

namespace sd::fft {

inline constexpr double kSqrt2 = 0x1.6a09e667f3bcdp+0;  // fl(sqrt 2)

// The Dct2 kind's table: (cos, sin)(pi k / 2N), k = 0 .. N / 2, interleaved.
inline std::vector<double> dct2_twiddles(int64_t n) {
    std::vector<double> w(static_cast<size_t>(2 * (n / 2 + 1)));
    for (int64_t k = 0; k <= n / 2; ++k) {
        const Twiddle t = twiddle(k, 4 * n);
        w[static_cast<size_t>(2 * k)] = t.re, w[static_cast<size_t>(2 * k + 1)] = -t.im;
    }
    return w;
}

// The even Dct4 kind's table: w_j = e^(-i pi (8j + 1) / 8N), j = 0 .. N / 2 - 1, interleaved.
inline std::vector<double> dct4_twiddles(int64_t n) {
    std::vector<double> w(static_cast<size_t>(n));
    for (int64_t j = 0; j < n / 2; ++j) {
        const Twiddle t = twiddle(8 * j + 1, 16 * n);
        w[static_cast<size_t>(2 * j)] = t.re, w[static_cast<size_t>(2 * j + 1)] = t.im;
    }
    return w;
}

// The odd Dct4 kind's tables (2N entries): entry j < N is (n << 1) | [chi1(b) = -1], the source of u_j; entry
// N + k is (l' << 2) | [chi1(alpha a) = -1] << 1 | [sigma = -1] with l' = min(l, N - l) and sigma the sign of
// Im U_l' in y_k (chi4(alpha a), flipped when l > N / 2).
inline std::vector<int64_t> dct4_odd_tables(int64_t n) {
    auto chi1 = [](int64_t r) { return (r & 7) == 1 || (r & 7) == 7 ? 1 : -1; };
    auto chi4 = [](int64_t r) { return (r & 3) == 1 ? 1 : -1; };
    const int64_t alpha = n & 7;  // n^-1 mod 8: the odd squares are 1 mod 8
    // beta = (1 - alpha n) / 8 (exact) reduced mod n: 8^-1 mod n
    __int128 br = ((static_cast<__int128>(1) - static_cast<__int128>(alpha) * n) / 8) % n;
    if (br < 0) br += n;
    const int64_t beta = static_cast<int64_t>(br);
    std::vector<int64_t> t(static_cast<size_t>(2 * n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t b = 2 * i + 1;
        const int64_t bm = b % n;
        const int64_t j = chi4(b) == 1 ? bm : (n - bm) % n;
        t[static_cast<size_t>(j)] = (i << 1) | (chi1(b) < 0 ? 1 : 0);
    }
    for (int64_t k = 0; k < n; ++k) {
        const int64_t a = 2 * k + 1;
        const int64_t l = static_cast<int64_t>(static_cast<__int128>(beta) * (a % n) % n);
        const int64_t aa = (alpha * (a & 7)) & 7;
        int sigma = chi4(aa);
        int64_t lp = l;
        if (2 * l > n) lp = n - l, sigma = -sigma;
        t[static_cast<size_t>(n + k)] = (lp << 2) | (chi1(aa) < 0 ? 2 : 0) | (sigma < 0 ? 1 : 0);
    }
    return t;
}

// The transform a line takes: the type actually computed (idct's and idst's 2 and 3 swapped), DCT or DST,
// and orthogonalize.
struct R2r {
    int type = 2;
    bool sine = false;
    bool ortho = false;
};

// The kind and length of the plan a transform of N points runs (fft/plan.hpp).
inline Kind r2r_kind(const R2r& r) { return r.type == 1 ? Kind::Real : r.type == 4 ? Kind::Dct4 : Kind::Dct2; }
inline int64_t r2r_plan_length(const R2r& r, int64_t n) {
    if (r.type != 1) return n;
    return r.sine ? 2 * (n + 1) : 2 * (n - 1);
}

// The doubles of r2r_line's tmp.
inline int64_t r2r_tmp_doubles(const R2r& r, int64_t n) {
    if (r.type == 1) return 4 * n + 8;
    if (n % 2 == 0) return r.type == 4 ? 2 * n : 3 * n + 2;
    return 6 * n + 2;
}

namespace detail {

// V (n / 2 + 1 interleaved complex) = the half spectrum of the n reals v by the inner plan of a Dct2 kind
// (the real kind for even n, the complex kind and the Hermitian part for odd n: the real path of
// fft/driver.cpp). tmp: 4n doubles (odd n).
inline void real_forward(const Plan& inner, const double* v, double* V, double* tmp, double* work) {
    const int64_t n = inner.n();
    if (inner.kind() == Kind::Real) {
        inner.r2c(v, V, work);
        return;
    }
    double* const c = tmp;
    double* const t = tmp + 2 * n;
    run_chunks(n, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) c[2 * i] = v[i], c[2 * i + 1] = 0.0;
    });
    inner.execute(c, t, work, false);
    hermitian_part(t, V, n, n / 2 + 1);
}

// v (n reals) = the unnormalised inverse of the Hermitian half spectrum V (n / 2 + 1 entries; the imaginary
// parts of V_0 and, for even n, V_(n/2) dropped). tmp: n doubles (even n) or 4n (odd n).
inline void real_inverse(const Plan& inner, const double* V, double* v, double* tmp, double* work) {
    const int64_t n = inner.n();
    if (inner.kind() == Kind::Real) {
        inner.c2r<false>(V, v, tmp, work);
        return;
    }
    double* const e = tmp;
    double* const t = tmp + 2 * n;
    e[0] = V[0], e[1] = 0.0;
    run_chunks(n / 2, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0 + 1; i <= i1; ++i) {
            const double re = V[2 * i], im = V[2 * i + 1];
            e[2 * i] = re, e[2 * i + 1] = im;
            e[2 * (n - i)] = re, e[2 * (n - i) + 1] = -im;
        }
    });
    inner.execute(e, t, work, true);
    run_chunks(n, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) v[i] = t[2 * i];
    });
}

inline void dct1(const Plan& p, int64_t n, const double* x, double* y, double* tmp, double* work) {
    double* const v = tmp;              // 2n - 2
    double* const X = tmp + 2 * n + 2;  // n complex
    run_chunks(n, [&](int64_t i0, int64_t i1) {
        std::memcpy(v + i0, x + i0, static_cast<size_t>(i1 - i0) * sizeof(double));
        for (int64_t i = i0; i < std::min(i1, n - 2); ++i) v[n + i] = x[n - 2 - i];
    });
    p.r2c(v, X, work);
    run_chunks(n, [&](int64_t k0, int64_t k1) {
        for (int64_t k = k0; k < k1; ++k) y[k] = X[2 * k];
    });
}

inline void dst1(const Plan& p, int64_t n, const double* x, double* y, double* tmp, double* work) {
    double* const v = tmp;              // 2n + 2
    double* const X = tmp + 2 * n + 2;  // n + 2 complex
    v[0] = 0.0, v[n + 1] = 0.0;
    run_chunks(n, [&](int64_t i0, int64_t i1) {
        std::memcpy(v + 1 + i0, x + i0, static_cast<size_t>(i1 - i0) * sizeof(double));
        for (int64_t i = i0; i < i1; ++i) v[n + 2 + i] = -x[n - 1 - i];
    });
    p.r2c(v, X, work);
    run_chunks(n, [&](int64_t k0, int64_t k1) {
        for (int64_t k = k0; k < k1; ++k) y[k] = -X[2 * k + 3];
    });
}

inline void dct2(const Plan& p, int64_t n, const double* x, double* y, double* tmp, double* work) {
    double* const v = tmp;          // n
    double* const V = tmp + n;      // n / 2 + 1 complex
    double* const t = tmp + 2 * n + 2;  // 4n (odd n)
    run_chunks((n + 1) / 2, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) v[i] = x[2 * i];
        for (int64_t i = i0; i < std::min(i1, n / 2); ++i) v[n - 1 - i] = x[2 * i + 1];
    });
    real_forward(*p.half(), v, V, t, work);
    const double* const w = p.table();
    y[0] = 2.0 * V[0];
    run_chunks(n / 2, [&](int64_t k0, int64_t k1) {
        for (int64_t k = k0 + 1; k <= k1; ++k) {
            const double c = w[2 * k], s = w[2 * k + 1], re = V[2 * k], im = V[2 * k + 1];
            y[k] = 2.0 * std::fma(s, im, c * re);
            if (2 * k != n) y[n - k] = 2.0 * std::fma(-c, im, s * re);
        }
    });
}

inline void dct3(const Plan& p, int64_t n, const double* x, double* y, double* tmp, double* work) {
    double* const V = tmp;              // n / 2 + 1 complex
    double* const v = tmp + n + 2;      // n
    double* const t = tmp + 2 * n + 2;  // n (even n) or 4n (odd n)
    const double* const w = p.table();
    V[0] = x[0], V[1] = 0.0;
    run_chunks(n / 2, [&](int64_t k0, int64_t k1) {
        for (int64_t k = k0 + 1; k <= k1; ++k) {
            const double c = w[2 * k], s = w[2 * k + 1], a = x[k], b = x[n - k];
            V[2 * k] = std::fma(s, b, c * a);
            V[2 * k + 1] = std::fma(-c, b, s * a);
        }
    });
    real_inverse(*p.half(), V, v, t, work);
    run_chunks((n + 1) / 2, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) y[2 * i] = v[i];
        for (int64_t i = i0; i < std::min(i1, n / 2); ++i) y[2 * i + 1] = v[n - 1 - i];
    });
}

inline void dct4(const Plan& p, int64_t n, const double* x, double* y, double* tmp, double* work) {
    if (n % 2 == 0) {
        const int64_t h = n / 2;
        double* const z = tmp;      // h complex
        double* const Z = tmp + n;  // h complex
        const double* const w = p.table();
        run_chunks(h, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; ++j) {
                const double a = x[2 * j], b = x[n - 1 - 2 * j], wr = w[2 * j], wi = w[2 * j + 1];
                z[2 * j] = std::fma(-b, wi, a * wr);
                z[2 * j + 1] = std::fma(b, wr, a * wi);
            }
        });
        p.half()->execute(z, Z, work, false);
        run_chunks(h, [&](int64_t q0, int64_t q1) {
            for (int64_t q = q0; q < q1; ++q) {
                const double a = Z[2 * q], b = Z[2 * q + 1], wr = w[2 * q], wi = w[2 * q + 1];
                y[2 * q] = 2.0 * std::fma(-b, wi, a * wr);
                y[n - 1 - 2 * q] = -2.0 * std::fma(b, wr, a * wi);
            }
        });
        return;
    }
    const int64_t* const tab = p.index();
    double* const c = tmp;              // n complex
    double* const t = tmp + 2 * n;      // n complex
    double* const U = tmp + 4 * n;      // (n + 1) / 2 complex
    run_chunks(n, [&](int64_t j0, int64_t j1) {
        for (int64_t j = j0; j < j1; ++j) {
            const int64_t e = tab[j];
            const double v = x[e >> 1];
            c[2 * j] = (e & 1) ? -v : v, c[2 * j + 1] = 0.0;
        }
    });
    p.half()->execute(c, t, work, false);
    hermitian_part(t, U, n, (n + 1) / 2);
    run_chunks(n, [&](int64_t k0, int64_t k1) {
        for (int64_t k = k0; k < k1; ++k) {
            const int64_t e = tab[n + k];
            const int64_t l = e >> 2;
            const double re = U[2 * l], im = U[2 * l + 1];
            y[k] = ((e & 2) ? -kSqrt2 : kSqrt2) * ((e & 1) ? re - im : re + im);
        }
    });
}

inline void reverse(double* x, int64_t n) {
    for (int64_t i = 0, j = n - 1; i < j; ++i, --j) {
        const double t = x[i];
        x[i] = x[j], x[j] = t;
    }
}

inline void negate_odd(double* x, int64_t n) {
    for (int64_t i = 1; i < n; i += 2) x[i] = -x[i];
}

}  // namespace detail

// y (n doubles) times s, one rounding each (by chunks: a long line's on the pool).
inline void scale_reals(double* y, int64_t n, double s) {
    if (s == 1.0) return;
    run_chunks(n, [&](int64_t k0, int64_t k1) {
        for (int64_t k = k0; k < k1; ++k) y[k] *= s;
    });
}


// y (n doubles) = the transform r of the n doubles x times the factor s, with orthogonalize's sqrt 2 when
// r.ortho. plan: the plan of (r2r_kind(r), r2r_plan_length(r, n)); x is overwritten (the sign changes,
// reversals and sqrt 2 of the input); tmp: r2r_tmp_doubles(r, n) doubles; work: the plan's work_doubles().
// None of x, y, tmp, work overlap.
inline void r2r_line(
    const Plan& plan, const R2r& r, int64_t n, double s, double* x, double* y, double* tmp, double* work
) {
    if (r.type == 1) {
        if (r.sine) {
            detail::dst1(plan, n, x, y, tmp, work);
            scale_reals(y, n, s);
            return;
        }
        if (r.ortho) x[0] *= kSqrt2, x[n - 1] *= kSqrt2;
        detail::dct1(plan, n, x, y, tmp, work);
        scale_reals(y, n, s);
        if (r.ortho) y[0] /= kSqrt2, y[n - 1] /= kSqrt2;
        return;
    }
    if (r.sine) {  // the DCT's input: DST-II (-1)^n x_n, DST-III and DST-IV the reversed x
        if (r.type == 2) {
            detail::negate_odd(x, n);
        } else {
            detail::reverse(x, n);
        }
    }
    if (r.type == 2) {
        detail::dct2(plan, n, x, y, tmp, work);
    } else if (r.type == 3) {
        if (r.ortho) x[0] *= kSqrt2;
        detail::dct3(plan, n, x, y, tmp, work);
    } else {
        detail::dct4(plan, n, x, y, tmp, work);
    }
    scale_reals(y, n, s);
    if (r.ortho && r.type == 2) y[0] /= kSqrt2;
    if (r.sine) {  // DST-II the reversed result, DST-III and DST-IV (-1)^k y_k
        if (r.type == 2) {
            detail::reverse(y, n);
        } else {
            detail::negate_odd(y, n);
        }
    }
}

}  // namespace sd::fft
