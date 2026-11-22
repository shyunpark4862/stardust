// Double-double arithmetic for the plans' kernels that must be accurate to far below one rounding
// (docs/design/fft.md §3.3, §4.2): Bluestein's chirp and the transform of its convolution kernel.
//
// A value is hi + lo with |lo| <= ulp(hi) / 2. Sums and products are Dekker's and Knuth's error-free
// transformations (std::fma for the product), so each operation is within about 2^-104 of its result
// (relatively for products, of the larger operand for sums). The plans use these values only through
// norms (the kernel's transform and the chirp enter the bound as a whole, docs/design/fft.md §4.2),
// where errors of 2^-100 are invisible next to the u = 2^-53 of the doubles that are finally stored.
//
// cis(t, N) = e^(-2 pi i t / N) for integers 0 <= t < N: t is folded onto an angle in [0, pi/4] by exact
// integer arithmetic (as twiddle.hpp), the angle is (pi/4) (r / N) formed as a double-double, and sin
// and cos are their Taylor series to the 31st power (the first term left out is below 2^-110 of the
// result on [0, pi/4]), summed by Horner's rule in double-double. CisTable answers many t of one N with
// two tables of about sqrt(N) entries each (t = t1 B + t0: one double-double complex product).
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::fft::dd {

struct DD {
    double hi = 0.0, lo = 0.0;
};

struct DC {
    DD re, im;
};

inline DD quick_two_sum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}

inline DD two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

inline DD add(DD a, DD b) {
    DD s = two_sum(a.hi, b.hi);
    const DD t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline DD neg(DD a) { return {-a.hi, -a.lo}; }
inline DD sub(DD a, DD b) { return add(a, neg(b)); }

inline DD mul(DD a, DD b) {
    const double p = a.hi * b.hi;
    const double e = std::fma(a.hi, b.hi, -p);
    return quick_two_sum(p, e + (a.hi * b.lo + a.lo * b.hi));
}

inline DD mul(DD a, double b) {
    const double p = a.hi * b;
    const double e = std::fma(a.hi, b, -p);
    return quick_two_sum(p, e + a.lo * b);
}

// a / d for a double d (exact integers in the series below)
inline DD div(DD a, double d) {
    const double q1 = a.hi / d;
    const double p = q1 * d;
    const double pe = std::fma(q1, d, -p);
    const double r = ((a.hi - p) - pe) + a.lo;
    return quick_two_sum(q1, r / d);
}

inline DC cadd(DC a, DC b) { return {add(a.re, b.re), add(a.im, b.im)}; }
inline DC csub(DC a, DC b) { return {sub(a.re, b.re), sub(a.im, b.im)}; }
inline DC cmul(DC a, DC b) { return {sub(mul(a.re, b.re), mul(a.im, b.im)), add(mul(a.re, b.im), mul(a.im, b.re))}; }
inline DC conj(DC a) { return {a.re, neg(a.im)}; }
inline DC mul_neg_i(DC a) { return {a.im, neg(a.re)}; }  // -i a

// pi/4 as a double-double
inline constexpr double kQuarterPiHi = 0x1.921fb54442d18p-1;
inline constexpr double kQuarterPiLo = 0x1.1a62633145c07p-55;

// (sin, cos) of theta = (pi/4)(num / den), 0 <= num <= den < 2^53
inline void octant_sincos(int64_t num, int64_t den, DD& s, DD& c) {
    if (num == 0) {
        s = {0.0, 0.0};
        c = {1.0, 0.0};
        return;
    }
    const double dn = static_cast<double>(num), dd = static_cast<double>(den);
    const double q = dn / dd;
    const double e = std::fma(-q, dd, dn) / dd;  // num - q den is exact
    const DD frac = quick_two_sum(q, e);
    const DD th = mul(frac, DD{kQuarterPiHi, kQuarterPiLo});
    const DD x2 = mul(th, th);
    constexpr int K = 15;  // terms through theta^31 / 31! and theta^30 / 30!
    DD ps{1.0, 0.0}, pc{1.0, 0.0};
    for (int k = K; k >= 1; --k) {
        ps = sub(DD{1.0, 0.0}, div(mul(x2, ps), static_cast<double>((2 * k) * (2 * k + 1))));
        pc = sub(DD{1.0, 0.0}, div(mul(x2, pc), static_cast<double>((2 * k - 1) * (2 * k))));
    }
    s = mul(th, ps);
    c = pc;
}

// e^(-2 pi i t / N), 0 <= t < N, N < 2^53 / 8
inline DC cis(int64_t t, int64_t N) {
    const int64_t o = (8 * t) / N;
    const int64_t r = 8 * t - o * N;
    DD as, ac;  // (sin, cos) of the angle measured from the quarter turn q
    int64_t q;
    if (o % 2 == 0) {
        q = o / 2;
        octant_sincos(r, N, as, ac);
    } else {
        q = (o + 1) / 2;
        DD bs, bc;
        octant_sincos(N - r, N, bs, bc);
        as = neg(bs), ac = bc;
    }
    DD c, s;  // cos and sin of 2 pi t / N
    switch (q & 3) {
        case 0: c = ac, s = as; break;
        case 1: c = neg(as), s = ac; break;
        case 2: c = neg(ac), s = neg(as); break;
        default: c = as, s = neg(ac); break;
    }
    return {c, neg(s)};
}

// cis(t, N) for many t of one N: coarse[t / B] * fine[t % B].
class CisTable {
public:
    explicit CisTable(int64_t N) : N_(N) {
        B_ = static_cast<int64_t>(std::ceil(std::sqrt(static_cast<double>(N))));
        if (B_ < 1) B_ = 1;
        fine_.resize(static_cast<size_t>(B_));
        for (int64_t t = 0; t < B_; ++t) fine_[static_cast<size_t>(t)] = cis(t % N, N);
        const int64_t nc = N / B_ + 1;
        coarse_.resize(static_cast<size_t>(nc));
        for (int64_t t = 0; t < nc; ++t) coarse_[static_cast<size_t>(t)] = cis((t * B_) % N, N);
    }
    DC operator()(int64_t t) const {
        const DC& c = coarse_[static_cast<size_t>(t / B_)];
        const DC& f = fine_[static_cast<size_t>(t % B_)];
        if (t % B_ == 0) return c;
        if (t < B_) return f;
        return cmul(c, f);
    }
    int64_t N() const { return N_; }

private:
    int64_t N_, B_ = 1;
    std::vector<DC> fine_, coarse_;
};

// The forward DFT (sign -) of x (length m, 5-smooth: factors 2, 3, 4, 5) in double-double: recursive
// decimation in time, out-of-place into y. w: the CisTable of m.
void dft(const DC* x, int64_t stride, DC* y, int64_t m, int64_t top, const CisTable& w);

}  // namespace sd::fft::dd
