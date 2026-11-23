// Expansions: exact sums of float64s (docs/design/spatial_geometry.md §3.1, the second stage of the
// predicates). An expansion here is ANY list of doubles h[0..n) whose exact sum is the value it stands
// for; n = 0 is zero. The routines keep the lists short and (in practice) sorted by magnitude and
// non-overlapping, as Shewchuk's do, but nothing below relies on that:
//
//   (E1) every routine preserves the exact sum. It is built from two error-free transformations only:
//        two_sum (Knuth; exact for every pair of doubles whose sum does not overflow) and two_prod
//        (p = fl(ab), e = fma(a, b, -p); exact when ab does not overflow and ab - p is representable).
//        fast_two_sum, which needs |a| >= |b|, is not used, so no ordering property is needed for (E1).
//   (E2) sign() certifies the sign from the list itself: with top = h[n-1] and r^ the float sum of the
//        other |h[i]|, |sum of the others| <= r <= r^ / (1 - u)^(n - 2) < r^ (1 + 2^-21) for n <= 2^20,
//        so |top| > fl(r^ (1 + 2^-20)) >= r^ (1 + 2^-21) gives sign(top). When that fails it distills
//        the list (one two_sum pass, which preserves the sum) and tries again; after kPasses it answers
//        "undecided" (2) and the caller takes the wide-integer path. So a wrong sign is impossible
//        whatever the list looks like, and "undecided" costs time only.
//
// The caller's promise (predicates.hpp, Range::safe): no intermediate value overflows, and every exact
// intermediate value is a multiple of 2^-1074 (so a product's rounding error is representable and
// two_prod is exact). Inside that promise the lists these routines produce from sorted non-overlapping
// inputs are the ones Shewchuk's grow / scale / fast-expansion-sum produce (the same operations on the
// same values; a two_sum where he uses fast_two_sum returns the same pair when his precondition holds),
// which is why sign() nearly always certifies at once.
//
// Sizes are static: add(Ex<N>, Ex<M>) is an Ex<N + M>, mul(Ex<N>, double) an Ex<2N>, mul(Ex<N>, Ex<M>)
// an Ex<2NM>; the largest one the predicates build is 1,152 doubles (insphere), all on the stack.
#pragma once

#include <cmath>

namespace sd::spatial {

inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bv = s - a;
    const double av = s - bv;
    e = (a - av) + (b - bv);
}

// a - b = d + e exactly.
inline void two_diff(double a, double b, double& d, double& e) {
    d = a - b;
    const double bv = a - d;
    const double av = d + bv;
    e = (a - av) + (bv - b);
}

inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
    e = std::fma(a, b, -p);
}

namespace expansion {

// h = e + sf * f (sf = 1 or -1); returns the length (<= ne + nf). h must not overlap e or f. The terms
// are taken in order of magnitude (a merge of the two lists) and accumulated by two_sum; zeros are dropped.
inline int sum(const double* e, int ne, const double* f, int nf, double sf, double* h) {
    int i = 0, j = 0, k = 0;
    auto next = [&]() -> double {
        if (j >= nf) return e[i++];
        if (i >= ne) return sf * f[j++];
        return std::fabs(e[i]) <= std::fabs(f[j]) ? e[i++] : sf * f[j++];
    };
    if (ne + nf == 0) return 0;
    double q = next();
    while (i < ne || j < nf) {
        double s, t;
        two_sum(q, next(), s, t);
        if (t != 0.0) h[k++] = t;
        q = s;
    }
    if (q != 0.0) h[k++] = q;
    return k;
}

// h = e * b; returns the length (<= 2 ne). h must not overlap e.
inline int scale(const double* e, int ne, double b, double* h) {
    if (ne == 0 || b == 0.0) return 0;
    int k = 0;
    double q, t;
    two_prod(e[0], b, q, t);
    if (t != 0.0) h[k++] = t;
    for (int i = 1; i < ne; ++i) {
        double p1, p0, s;
        two_prod(e[i], b, p1, p0);
        two_sum(q, p0, s, t);
        if (t != 0.0) h[k++] = t;
        two_sum(p1, s, q, t);
        if (t != 0.0) h[k++] = t;
    }
    if (q != 0.0) h[k++] = q;
    return k;
}

inline constexpr int kPasses = 6;

// The sign of the sum of h[0..n): -1, 0, 1, or 2 when kPasses distillations did not certify it (E2).
// Rewrites h. n <= 2^20.
inline int sign(double* h, int n) {
    for (int pass = 0;; ++pass) {
        if (n == 0) return 0;
        const double top = h[n - 1];
        double r = 0.0;
        for (int i = 0; i + 1 < n; ++i) r += std::fabs(h[i]);
        if (std::fabs(top) > r * (1.0 + 0x1p-20)) return top > 0.0 ? 1 : -1;
        if (pass == kPasses) return 2;
        double s = h[0];
        int k = 0;
        for (int i = 1; i < n; ++i) {
            double s2, t;
            two_sum(s, h[i], s2, t);
            if (t != 0.0) h[k++] = t;
            s = s2;
        }
        if (s != 0.0) h[k++] = s;
        n = k;
    }
}

}  // namespace expansion

// An expansion of at most N doubles.
template <int N>
struct Ex {
    int n = 0;
    double v[N];
};

inline Ex<2> mul(double a, double b) {
    Ex<2> r;
    double p, e;
    two_prod(a, b, p, e);
    if (e != 0.0) r.v[r.n++] = e;
    if (p != 0.0) r.v[r.n++] = p;
    return r;
}

template <int N, int M>
inline Ex<N + M> add(const Ex<N>& a, const Ex<M>& b) {
    Ex<N + M> r;
    r.n = expansion::sum(a.v, a.n, b.v, b.n, 1.0, r.v);
    return r;
}

template <int N, int M>
inline Ex<N + M> sub(const Ex<N>& a, const Ex<M>& b) {
    Ex<N + M> r;
    r.n = expansion::sum(a.v, a.n, b.v, b.n, -1.0, r.v);
    return r;
}

template <int N>
inline Ex<2 * N> mul(const Ex<N>& a, double b) {
    Ex<2 * N> r;
    r.n = expansion::scale(a.v, a.n, b, r.v);
    return r;
}

// a * b, the scaled copies of a summed one component of b at a time.
template <int N, int M>
inline Ex<2 * N * M> mul(const Ex<N>& a, const Ex<M>& b) {
    Ex<2 * N * M> r;
    double other[2 * N * M], t[2 * N];
    double* acc = r.v;
    double* tmp = other;
    int na = 0;
    for (int j = 0; j < b.n; ++j) {
        const int nt = expansion::scale(a.v, a.n, b.v[j], t);
        const int ns = expansion::sum(acc, na, t, nt, 1.0, tmp);
        double* swap = acc;
        acc = tmp;
        tmp = swap;
        na = ns;
    }
    if (acc != r.v)
        for (int i = 0; i < na; ++i) r.v[i] = acc[i];
    r.n = na;
    return r;
}

template <int N>
inline int sign(Ex<N>& a) {
    return expansion::sign(a.v, a.n);
}

}  // namespace sd::spatial
