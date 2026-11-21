// The Gauss-Legendre nodes and weights of fixed_quad (docs/design/integrate.md §3.7, decision
// C42): correctly rounded doubles.
//
// n <= 100: the table of tools/gen_gauss_legendre.py (Arb, each value settled and rounded).
// n > 100: Newton's method on P_n, first in double from the asymptotic guess
// cos(pi (i - 1/4) / (n + 1/2)), then twice in double-double (about 106 bits; P_n and P_n' by the
// three-term recurrence (j + 1) P_{j+1} = (2j + 1) x P_j - j P_{j-1} and
// P_n' = n (x P_n - P_{n-1}) / (x^2 - 1)), and the weight 2 / ((1 - x^2) P_n'(x)^2) at the final
// node, in double-double too. These come back as (hi, lo) pairs; stardust/integrate/_gauss.py
// rounds them and settles a value too close to a rounding boundary for the double-double's error
// by exact arithmetic. Each node is computed on its own (the nodes in parallel above 2^15
// recurrence steps of work), so the values do not depend on the thread count.
#include "integrate/gauss_table.hpp"
#include "integrate/integrate.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace sd::integrate {

namespace nb = nanobind;

namespace {

// ---- double-double: hi + lo, |lo| <= ulp(hi) / 2 (the error-free transformations)

struct DD {
    double hi, lo;
};

inline DD two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
inline DD quick_two_sum(double a, double b) {  // |a| >= |b|
    const double s = a + b;
    return {s, b - (s - a)};
}
inline DD two_prod(double a, double b) {
    const double p = a * b;
    return {p, std::fma(a, b, -p)};
}
inline DD add(DD a, DD b) {
    const DD s = two_sum(a.hi, b.hi);
    const DD t = two_sum(a.lo, b.lo);
    const DD r = quick_two_sum(s.hi, s.lo + t.hi);
    return quick_two_sum(r.hi, r.lo + t.lo);
}
inline DD neg(DD a) { return {-a.hi, -a.lo}; }
inline DD sub(DD a, DD b) { return add(a, neg(b)); }
inline DD mul(DD a, DD b) {
    DD p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}
inline DD mul(DD a, double b) {
    DD p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    return quick_two_sum(p.hi, p.lo);
}
inline DD div(DD a, DD b) {  // one correction from the exact remainder: ~2^-104 relative
    const double q1 = a.hi / b.hi;
    DD r = sub(a, mul(b, q1));
    const double q2 = r.hi / b.hi;
    r = sub(r, mul(b, q2));
    const double q3 = r.hi / b.hi;
    const DD q = quick_two_sum(q1, q2);
    return add(q, DD{q3, 0.});
}
inline DD div(DD a, double b) { return div(a, DD{b, 0.}); }

// P_n(x) and P_{n-1}(x), n >= 1
struct Pair {
    DD p, q;
};

Pair legendre_dd(int n, DD x) {
    DD q{1., 0.}, p = x;
    for (int j = 1; j < n; ++j) {
        const DD next =
            div(sub(mul(mul(x, p), 2. * j + 1.), mul(q, static_cast<double>(j))), static_cast<double>(j + 1));
        q = p;
        p = next;
    }
    return {p, q};
}

DD derivative_dd(int n, DD x, const Pair& v) {  // P_n'(x)
    return div(mul(sub(mul(x, v.p), v.q), static_cast<double>(n)), sub(mul(x, x), DD{1., 0.}));
}

double newton_double(int n, double x) {
    for (int it = 0; it < 100; ++it) {
        double q = 1., p = x;
        for (int j = 1; j < n; ++j) {
            const double next = ((2. * j + 1.) * x * p - j * q) / (j + 1.);
            q = p;
            p = next;
        }
        const double dp = n * (x * p - q) / (x * x - 1.);
        const double step = p / dp;
        x -= step;
        if (std::abs(step) <= 0x1p-52 * std::abs(x)) break;
    }
    return x;
}

// The i-th nonnegative node of n points in ascending order (i = 0 is the smallest) and its
// weight, as double-doubles.
void node_dd(int n, int i, DD& x, DD& w) {
    const int half = (n + 1) / 2;
    if (n % 2 == 1 && i == 0) {
        x = {0., 0.};
    } else {
        const int r = half - i;  // the r-th largest zero
        const double guess = std::cos(std::numbers::pi * (r - 0.25) / (n + 0.5));
        x = {newton_double(n, guess), 0.};
        for (int it = 0; it < 2; ++it) {
            const Pair v = legendre_dd(n, x);
            x = sub(x, div(v.p, derivative_dd(n, x, v)));
        }
    }
    const Pair v = legendre_dd(n, x);
    const DD d = derivative_dd(n, x, v);
    w = div(DD{2., 0.}, mul(sub(DD{1., 0.}, mul(x, x)), mul(d, d)));
}

nb::object vector_out(const std::vector<double>& v) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(v.size())};
    NDArray a = new_array(DType::Float64, shape);
    double* p = reinterpret_cast<double*>(a.data);
    for (std::size_t k = 0; k < v.size(); ++k) p[k] = v[k];
    return wrap(std::move(a));
}

// legendre_table(n), 1 <= n <= 100: (x, w) of the ceil(n/2) nonnegative nodes, ascending
nb::object legendre_table(int n) {
    std::vector<double> x, w;
    for (int k = kGaussOffset[n]; k < kGaussOffset[n + 1]; ++k) {
        x.push_back(kGaussNodes[k]);
        w.push_back(kGaussWeights[k]);
    }
    return nb::make_tuple(vector_out(x), vector_out(w));
}

// legendre_dd(n), n >= 1: (x_hi, x_lo, w_hi, w_lo) of the ceil(n/2) nonnegative nodes, ascending
nb::object legendre_dd_arrays(int n) {
    const int half = (n + 1) / 2;
    std::vector<double> xh(half), xl(half), wh(half), wl(half);
    auto one = [&](std::int64_t i) {
        DD x, w;
        node_dd(n, static_cast<int>(i), x, w);
        xh[i] = x.hi, xl[i] = x.lo, wh[i] = w.hi, wl[i] = w.lo;
    };
    const std::int64_t work = static_cast<std::int64_t>(half) * n;
    {
        parallel::Unlocked unlocked(work >= kParallelThreshold);
        if (work >= kParallelThreshold && parallel::available()) parallel::parallel_for(half, one);
        else
            for (int i = 0; i < half; ++i) one(i);
    }
    return nb::make_tuple(vector_out(xh), vector_out(xl), vector_out(wh), vector_out(wl));
}

}  // namespace

void bind_gauss(nb::module_& m) {
    m.attr("GAUSS_TABLE_MAX") = kGaussTableMax;
    m.def("legendre_table", &legendre_table, nb::arg("n"));
    m.def("legendre_dd", &legendre_dd_arrays, nb::arg("n"));
}

}  // namespace sd::integrate
