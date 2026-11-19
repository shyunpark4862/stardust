// Accuracy and speed of ckd::log_dd (stats/chunk_dd.hpp) against special::dd::log, and of
// vForce's atan against libm's.   Build: bench/stats/opt/a/build.sh log_dd
#include "stats/chunk_dd.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
#include <random>
using namespace sd::stats;
namespace dd = sd::special::dd;
int main() {
    std::mt19937_64 g(1);
    const int N = 1 << 20;
    std::vector<double> x(N), xl(N, 0.0), oh(N), ol(N);
    std::uniform_real_distribution<double> U(0, 1);
    double worst = 0;
    for (int mode = 0; mode < 6; ++mode) {
        for (int i = 0; i < N; ++i) {
            double u = U(g);
            if (mode == 0) x[i] = std::exp(-10 + 20 * u);
            else if (mode == 1) x[i] = 1 + (u - 0.5) * 1e-3;
            else if (mode == 2) x[i] = std::exp(-692 + 1384 * u);
            else if (mode == 3) x[i] = 1 + (u - 0.5) * std::pow(2.0, -52 * U(g));
            else if (mode == 4) x[i] = 0.7 + 0.9 * u;
            else x[i] = std::ldexp(1 + u, (i % 2 ? -999 : 998));
            xl[i] = (mode % 2) ? x[i] * 0x1p-53 * (U(g) - 0.5) : 0.0;
            double h = x[i] + xl[i];
            xl[i] -= (h - x[i]);
            x[i] = h;
        }
        for (int b = 0; b < N; b += 1024) ckd::log_dd(x.data() + b, xl.data() + b, oh.data() + b, ol.data() + b, 1024);
        {
            long bad = 0;
            for (int i = 0; i < N; ++i) {
                double a, b2;
                ckd::log_dd_one(x[i], xl[i], a, b2);
                bad += std::memcmp(&a, &oh[i], 8) != 0 || std::memcmp(&b2, &ol[i], 8) != 0;
            }
            if (bad) printf("LANES DIFFER FROM SCALAR: %ld\n", bad);
        }
        double mx = 0;
        int wi = 0;
        for (int i = 0; i < N; ++i) {
            auto r = dd::log(dd::DD{x[i], xl[i]});
            auto dv = (dd::DD{oh[i], ol[i]} - r);
            double e = std::fabs((dv.hi + dv.lo) / r.hi);
            if (!(e <= mx) && r.hi != 0) {
                mx = e;
                wi = i;
            }
        }
        printf("mode %d max rel diff %.3g (2^%.1f) at x=%.17g\n", mode, mx, std::log2(mx), x[wi]);
        worst = std::max(worst, mx);
    }
    // the guard
    double sp[] = {0.0,   -1.0,     5e-324, 0x1p-1000, 0x1.fffffp-1000, 0x1p-999, 0x1p999, 0x1.ffffffp999, 0x1p1000,
                   1e308, INFINITY, NAN,    1.0},
           z[13] = {0}, a[13], b[13];
    ckd::log_dd(sp, z, a, b, 13);
    for (int i = 0; i < 13; ++i) printf("%g -> %.17g %g\n", sp[i], a[i], b[i]);
    double best = 1e9;
    for (int r = 0; r < 9; ++r) {
        auto t = std::chrono::steady_clock::now();
        for (int b = 0; b < N; b += 1024) ckd::log_dd(x.data() + b, xl.data() + b, oh.data() + b, ol.data() + b, 1024);
        best =
            std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t).count() / N);
    }
    printf("log_dd %.2f ns\n", best);
    // atan
    double mxa = 0;
    for (int rep = 0; rep < 4; ++rep) {
        for (int i = 0; i < N; ++i) {
            double u = U(g);
            x[i] = (i % 2 ? -1 : 1) * (rep == 0   ? std::exp(-690 + 1380 * u)
                                       : rep == 1 ? 10 * u
                                       : rep == 2 ? std::exp(-5 + 10 * u)
                                                  : 1 + (u - .5) * .5);
        }
        for (int b = 0; b < N; b += 1024) ckd::atan(oh.data() + b, x.data() + b, 1024);
        for (int i = 0; i < N; ++i) {
            double r = std::atan(x[i]);
            mxa = std::max(mxa, std::fabs(oh[i] - r) / (std::fabs(r) * 0x1p-52));
        }
    }
    printf("vvatan vs libm atan: max %.2f ULP\n", mxa);
}
