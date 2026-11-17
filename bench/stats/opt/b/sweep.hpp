// Sweep + timing helpers for chunk kernels against the scalar definitions.
#pragma once
#include "stats/engine.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
namespace sw {
using namespace sd::stats;
inline std::mt19937_64 rng(12345);
inline double uni(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
inline double logu(double a, double b) { return std::exp(uni(std::log(a), std::log(b))); }
__attribute__((noinline)) inline double p1(double v) { static volatile double z = 0; return v + z; }

// err in contract units: mixed -> |d| / max(1, |ref|), else relative (subnormal ref: absolute / 2^-1022).
inline double err_of(double got, double ref, bool mixed) {
    if (got == ref || (std::isnan(got) && std::isnan(ref))) return 0.0;
    if (std::isnan(got) != std::isnan(ref) || std::isinf(got) || std::isinf(ref)) return INFINITY;
    const double d = std::fabs(got - ref);
    if (mixed) return d / std::max(1.0, std::fabs(ref));
    return d / std::max(std::fabs(ref), 0x1p-1022);
}

// run a chunk kernel over arrays (NP parameter arrays), same = all params constant
template <std::size_t NP, class K>
void run(K kernel, const std::vector<double>& x, const std::array<std::vector<double>, NP>& p, bool same, std::vector<double>& out) {
    const int n = (int)x.size();
    out.resize(n);
    for (int off = 0; off < n; off += kChunkSize) {
        const int m = std::min(kChunkSize, n - off);
        Chunk<NP> c;
        c.n = m; c.x = x.data() + off;
        for (std::size_t k = 0; k < NP; ++k) { c.p[k] = p[k].data() + off; c.same[k] = same; }
        kernel(c, out.data() + off);
    }
}
struct Worst { double e = 0; int i = -1; };
inline FILE* dump = nullptr;  // points whose difference is above 1e-15, for the Arb check
template <std::size_t NP, class K, class Sc>
Worst sweep(const char* name, K kernel, Sc scalar, const std::vector<double>& x, const std::array<std::vector<double>, NP>& p, bool mixed, bool same = false) {
    std::vector<double> out;
    run<NP>(kernel, x, p, same, out);
    Worst w; long nd = 0, nbig = 0;
    std::vector<std::pair<double, int>> big;
    for (size_t i = 0; i < x.size(); ++i) {
        const double ref = scalar(i);
        const double e = err_of(out[i], ref, mixed);
        if (out[i] != ref) ++nd;
        if (e > w.e) { w.e = e; w.i = (int)i; }
        if (e > 1e-15) { ++nbig; big.push_back({e, (int)i}); }
    }
    printf("%-22s n=%zu differing %5.1f%% max %.3g (above 1e-15: %ld)", name, x.size(), 100.0 * nd / x.size(), w.e, nbig);
    if (w.i >= 0) { printf("  at x=%.17g p=", x[w.i]); for (auto& v : p) printf("%.17g ", v[w.i]); }
    printf("\n");
    if (dump) {
        std::sort(big.begin(), big.end(), [](auto& a, auto& b) { return a.first > b.first; });
        // the worst 300 and a spread of the rest
        for (size_t j = 0; j < big.size(); j += (j < 300 ? 1 : std::max<size_t>(1, big.size() / 300))) {
            const int i = big[j].second;
            fprintf(dump, "%s,%.17g", name, x[i]);
            for (std::size_t k = 0; k < 2; ++k) fprintf(dump, ",%.17g", k < NP ? p[k][i] : 0.0);
            fprintf(dump, ",%.17g,%.17g\n", out[i], scalar(i));
        }
    }
    return w;
}
template <std::size_t NP, class K>
double timeit(const char* name, K kernel, const std::vector<double>& x, const std::array<std::vector<double>, NP>& p) {
    std::vector<double> out; double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t = std::chrono::steady_clock::now();
        run<NP>(kernel, x, p, true, out);
        best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t).count() / x.size());
    }
    printf("%-34s %7.1f ns\n", name, best);
    return best;
}
template <class F>
double timesc(const char* name, F f, const std::vector<double>& x) {
    std::vector<double> o(x.size()); double best = 1e18;
    for (int r = 0; r < 5; ++r) {
        auto t = std::chrono::steady_clock::now();
        for (size_t i = 0; i < x.size(); ++i) o[i] = f(x[i]);
        best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t).count() / x.size());
    }
    printf("%-34s %7.1f ns\n", name, best);
    return best;
}
}  // namespace sw
