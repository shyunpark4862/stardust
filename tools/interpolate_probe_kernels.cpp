// The linear-kernel candidates of docs/design/interpolate.md §3.2 and §9 question 1, measured for
// tools/interpolate_probe.py (section "kernels"): the properties each one keeps on random intervals and
// the time of a numpy.interp-like loop (binary search + kernel) per candidate.
//
//   a  the chosen kernel (decision IP4): t = (x - x0) / (x1 - x0); t == 0 -> a, t == 1 -> b; opposite
//      signs t b + (1 - t) a; otherwise a + t (b - a), clamped at b (C++20 std::lerp's construction)
//   b  numpy 2.5.4's wheel: fma(s, x - x0, a), s = (b - a) / (x1 - x0)
//   c  fma(t, b - a, a)
//   d  (1 - t) a + t b
//   e  a computed without branches: both forms, then selects (fcsel); the same bits as a
//
// Compiled by the probe with stardust's flags (-O3 -mcpu=native -ffp-contract=off): every fma is the
// one written. Usage: kernels props | kernels time
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

volatile double g_sink = 0.0;  // the timed loops' results are read, so the compiler keeps them

inline double lerp_a(double x0, double x1, double a, double b, double x) {
    double t = (x - x0) / (x1 - x0);
    if (t == 0.0) return a;
    if (t == 1.0) return b;
    if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0)) return t * b + (1.0 - t) * a;
    double r = a + t * (b - a);
    if ((t > 1.0) == (b > a)) return b < r ? r : b;
    return r < b ? r : b;
}

inline double lerp_b(double x0, double x1, double a, double b, double x) {
    double s = (b - a) / (x1 - x0);
    return std::fma(s, x - x0, a);
}

inline double lerp_c(double x0, double x1, double a, double b, double x) {
    double t = (x - x0) / (x1 - x0);
    return std::fma(t, b - a, a);
}

inline double lerp_d(double x0, double x1, double a, double b, double x) {
    double t = (x - x0) / (x1 - x0);
    return (1.0 - t) * a + t * b;
}

inline double lerp_e(double x0, double x1, double a, double b, double x) {
    double t = (x - x0) / (x1 - x0);
    double r1 = t * b + (1.0 - t) * a;
    double r2 = a + t * (b - a);
    double up = std::fmax(r2, b), down = std::fmin(r2, b);
    r2 = ((t > 1.0) == (b > a)) ? up : down;  // bitwise tests below: no short-circuit branches
    bool opposite = ((a <= 0) & (b >= 0)) | ((a >= 0) & (b <= 0));
    double r = opposite ? r1 : r2;
    r = (t == 1.0) ? b : r;
    return (t == 0.0) ? a : r;
}

using Kernel = double (*)(double, double, double, double, double);
constexpr int NK = 5;
const Kernel KERNELS[NK] = {lerp_a, lerp_b, lerp_c, lerp_d, lerp_e};
const char* NAMES[NK] = {"a", "b", "c", "d", "e"};

// properties on intervals [x0, x1] with values a, b: exact ends, a == b, monotone in x, within [a, b]
void props() {
    std::mt19937_64 g(20261123);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::uniform_int_distribution<int> ex(-6, 6);
    const int intervals = 3000;
    long ends[NK] = {}, equal[NK] = {}, mono[NK] = {}, range[NK] = {}, points = 0, e_differs = 0;
    for (int it = 0; it < intervals; ++it) {
        double x0 = nd(g) * std::ldexp(1.0, ex(g));
        double x1 = x0 + std::fabs(nd(g)) * std::ldexp(1.0, ex(g)) + 1e-300;
        double a = nd(g) * std::ldexp(1.0, ex(g)), b = nd(g) * std::ldexp(1.0, ex(g));
        if (it % 10 == 0) b = a;                                   // equal values
        if (it % 10 == 1) b = a * (1 + std::ldexp(nd(g), -40));    // nearly equal
        if (it % 10 == 2) b = std::copysign(std::fabs(b), -a);     // opposite signs
        std::vector<double> xs;
        double lo = x0, hi = x1;
        xs.push_back(x0);
        xs.push_back(x1);
        for (int k = 0; k < 40; ++k) {
            lo = std::nextafter(lo, INFINITY);
            hi = std::nextafter(hi, -INFINITY);
            xs.push_back(lo);
            xs.push_back(hi);
        }
        for (int k = 0; k < 200; ++k) xs.push_back(x0 + (x1 - x0) * ud(g));
        std::sort(xs.begin(), xs.end());
        xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
        xs.erase(std::remove_if(xs.begin(), xs.end(), [&](double v) { return v < x0 || v > x1; }), xs.end());
        points += (long)xs.size();
        for (double x : xs) {
            double va = lerp_a(x0, x1, a, b, x), ve = lerp_e(x0, x1, a, b, x);
            e_differs += std::memcmp(&va, &ve, sizeof(double)) != 0;
        }
        for (int k = 0; k < NK; ++k) {
            Kernel f = KERNELS[k];
            if (f(x0, x1, a, b, x0) != a || f(x0, x1, a, b, x1) != b) ++ends[k];
            double prev = f(x0, x1, a, b, xs[0]);
            bool m = true, r = true, e = true;
            for (double x : xs) {
                double v = f(x0, x1, a, b, x);
                if (a == b && v != a) e = false;
                if (v < std::min(a, b) || v > std::max(a, b)) r = false;
                if ((b >= a && v < prev) || (b < a && v > prev)) m = false;
                prev = v;
            }
            equal[k] += (a == b && !e);
            mono[k] += !m;
            range[k] += !r;
        }
    }
    std::printf(
        "intervals %d, points %ld (the 40 doubles next to each end and 200 uniform points each)\n", intervals, points
    );
    std::printf("kernel  ends_wrong  equal_wrong  not_monotone  out_of_range (intervals)\n");
    for (int k = 0; k < NK; ++k)
        std::printf("  %s     %8ld  %11ld  %12ld  %12ld\n", NAMES[k], ends[k], equal[k], mono[k], range[k]);
    std::printf("points where e's bits differ from a's: %ld\n", e_differs);
}

// sd.interp's loop: binary search for x[i] <= v < x[i+1] (clamped to the ends), then the kernel
template <int K>
double run(
    const std::vector<double>& xp, const std::vector<double>& fp, const std::vector<double>& x, std::vector<double>& out
) {
    const size_t n = xp.size();
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < x.size(); ++i) {
        double v = x[i];
        if (v <= xp[0]) {
            out[i] = fp[0];
            continue;
        }
        if (v >= xp[n - 1]) {
            out[i] = fp[n - 1];
            continue;
        }
        size_t j = (size_t)(std::upper_bound(xp.begin(), xp.end(), v) - xp.begin()) - 1;
        out[i] = KERNELS[K](xp[j], xp[j + 1], fp[j], fp[j + 1], v);
    }
    auto t1 = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (double v : out) sum += v;
    g_sink = g_sink + sum;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double check_cost(const std::vector<double>& xp) {
    auto t0 = std::chrono::steady_clock::now();
    bool ok = std::isfinite(xp[0]);
    for (size_t i = 1; i < xp.size(); ++i) ok &= (xp[i] >= xp[i - 1]) & std::isfinite(xp[i]);
    auto t1 = std::chrono::steady_clock::now();
    volatile bool sink = ok;
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void timing() {
    std::mt19937_64 g(20261124);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::normal_distribution<double> nd;
    for (size_t n : {size_t(1000), size_t(1000000)}) {
        std::vector<double> xp(n), fp(n);
        for (size_t i = 0; i < n; ++i) xp[i] = (double)i / (double)(n - 1), fp[i] = nd(g);
        std::vector<double> x(1000000), out(x.size());
        for (double& v : x) v = ud(g);
        for (int sorted = 0; sorted < 2; ++sorted) {
            if (sorted) std::sort(x.begin(), x.end());
            double best[NK] = {1e300, 1e300, 1e300, 1e300, 1e300};
            for (int rep = 0; rep < 5; ++rep) {
                best[0] = std::min(best[0], run<0>(xp, fp, x, out));
                best[1] = std::min(best[1], run<1>(xp, fp, x, out));
                best[2] = std::min(best[2], run<2>(xp, fp, x, out));
                best[3] = std::min(best[3], run<3>(xp, fp, x, out));
                best[4] = std::min(best[4], run<4>(xp, fp, x, out));
            }
            std::printf(
                "n = %7zu, 10^6 %s points: a %.2f  b %.2f  c %.2f  d %.2f  e %.2f ms (best of 5)\n", n,
                sorted ? "sorted" : "random", best[0], best[1], best[2], best[3], best[4]
            );
        }
        double c = 1e300;
        for (int rep = 0; rep < 5; ++rep) c = std::min(c, check_cost(xp));
        std::printf("n = %7zu: checking xp finite and nondecreasing %.3f ms\n", n, c);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "props") == 0) props();
    else if (argc > 1 && std::strcmp(argv[1], "time") == 0) timing();
    else {
        std::fprintf(stderr, "usage: kernels props|time\n");
        return 2;
    }
    return 0;
}
