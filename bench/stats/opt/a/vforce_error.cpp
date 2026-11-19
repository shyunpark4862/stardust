// The relative error of the vForce functions the chunk kernels use (stats/chunk.hpp's kVfExp
// and kVfLog, stats/chunk_dd.hpp's kPow, kTanh, kAtanh, kAtan), in kUlp = 2^-53, against
// double-double references, with libm's on the same points for comparison (libm measures
// 1.0 for exp, log and pow: the references are sound).
//     exp:   num::exp_full                    log:   special::dd::log
//     pow:   exp_full(dd::log(x) y)           tanh:  -m / (2 + m), m = expm1(-2x) (extreme_detail::expm1_any)
//     atanh: log((1 + s) / (1 - s)) / 2 (the series below 1e-5)
//     atan:  no double-double reference here: the distance to libm's atan (itself within 1 ULP)
// (tanh: over 2.5e8 points the maximum was 5.27.)
// Build: bench/stats/opt/a/build.sh vforce_error
#include "stats/dist/extreme.hpp"

#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <random>
#include <vector>

using namespace sd::stats;
using num::DD;

namespace {
std::mt19937_64 g(3);
std::uniform_real_distribution<double> U(0.0, 1.0);
const int N = 1 << 22;
std::vector<double> x(N), y(N), o(N);

double units(double got, DD ref) { return std::fabs((DD{got, 0.0} - ref).hi) / (std::fabs(ref.hi) * 0x1p-53); }
template <class F>
void run(F f) {
    for (int b = 0; b < N; b += 1024) f(o.data() + b, x.data() + b, y.data() + b, 1024);
}
}  // namespace

int main() {
    for (int mode = 0; mode < 4; ++mode) {
        for (int i = 0; i < N; ++i)
            x[i] = mode == 0   ? std::exp(-30 + 60 * U(g))
                   : mode == 1 ? 0.5 + 1.5 * U(g)
                   : mode == 2 ? std::exp(-700 + 1400 * U(g))
                               : 0.9 + 0.2 * U(g);
        run([](double* out, const double* a, const double*, int n) { vvlog(out, a, &n); });
        double mv = 0, ml = 0;
        for (int i = 0; i < N; ++i) {
            const DD r = num::dd::log(x[i]);
            mv = std::max(mv, units(o[i], r)), ml = std::max(ml, units(std::log(x[i]), r));
        }
        std::printf("log   mode %d: vForce %.3f, libm %.3f\n", mode, mv, ml);
        for (int i = 0; i < N; ++i)
            x[i] = mode == 0 ? -30 + 60 * U(g) : mode == 1 ? -1 + 2 * U(g) : mode == 2 ? -700 + 1400 * U(g) : -5 * U(g);
        run([](double* out, const double* a, const double*, int n) { vvexp(out, a, &n); });
        mv = ml = 0;
        for (int i = 0; i < N; ++i) {
            const DD r = num::exp_full(DD{x[i], 0.0});
            mv = std::max(mv, units(o[i], r)), ml = std::max(ml, units(std::exp(x[i]), r));
        }
        std::printf("exp   mode %d: vForce %.3f, libm %.3f\n", mode, mv, ml);
    }
    for (int mode = 0; mode < 6; ++mode) {
        for (int i = 0; i < N; ++i) {
            const double u = U(g), v = U(g);
            if (mode == 0) x[i] = 0.05 + 0.95 * u, y[i] = 0.05 + 10 * v;
            else if (mode == 1)
                x[i] = std::exp(-5 + 10 * u),
                y[i] = std::fabs(std::log(x[i])) * std::exp(-7 + 14 * v) > 700 ? 1.0 : std::exp(-7 + 14 * v);
            else if (mode == 2) x[i] = 1 + (u - 0.5) * 1e-3, y[i] = std::exp(14 * v);
            else if (mode == 3) x[i] = std::exp(-600 + 1200 * u), y[i] = (v - 0.5) * 2.0;
            else if (mode == 4) x[i] = 0.2 + 0.8 * u, y[i] = 3.0;
            else x[i] = 1 + (u - 0.5) * 1e-8, y[i] = (v < 0.5 ? -1 : 1) * std::exp(20 * U(g));
        }
        run([](double* out, const double* a, const double* e, int n) { vvpow(out, e, a, &n); });
        double mv = 0, ml = 0;
        for (int i = 0; i < N; ++i) {
            const DD r = num::exp_full(num::dd::log(x[i]) * y[i]);
            mv = std::max(mv, units(o[i], r)), ml = std::max(ml, units(std::pow(x[i], y[i]), r));
        }
        std::printf("pow   mode %d: vForce %.3f, libm %.3f\n", mode, mv, ml);
    }
    for (int mode = 0; mode < 4; ++mode) {
        for (int i = 0; i < N; ++i) {
            const double u = U(g);
            x[i] = mode == 0 ? 0.35 * u : mode == 1 ? 20 * u : mode == 2 ? std::pow(10.0, -300 * u) : 0.2 + 0.6 * u;
        }
        run([](double* out, const double* a, const double*, int n) { vvtanh(out, a, &n); });
        double mv = 0, ml = 0;
        for (int i = 0; i < N; ++i) {
            const DD m = extreme_detail::expm1_any(DD{-2 * x[i], 0.0});
            const DD r = -(m / (m + 2.0));
            mv = std::max(mv, units(o[i], r)), ml = std::max(ml, units(std::tanh(x[i]), r));
        }
        std::printf("tanh  mode %d: vForce %.3f, libm %.3f\n", mode, mv, ml);
    }
    for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < N; ++i) {
            const double u = U(g);
            x[i] = mode == 0 ? u / 3 : mode == 1 ? std::pow(10.0, -300 * u) : -u / 3;
        }
        run([](double* out, const double* a, const double*, int n) { vvatanh(out, a, &n); });
        double mv = 0, ml = 0;
        for (int i = 0; i < N; ++i) {
            DD r;
            if (std::fabs(x[i]) < 1e-5) {
                const DD s{x[i], 0.0}, s2 = s * s;
                r = s + s * s2 * DD{1.0 / 3, 1.8503717077085941e-17} + s * s2 * s2 * 0.2;
            } else {
                r = num::dd::log(num::dd::two_sum(1.0, x[i]) / num::dd::two_sum(1.0, -x[i])) * 0.5;
            }
            mv = std::max(mv, units(o[i], r)), ml = std::max(ml, units(std::atanh(x[i]), r));
        }
        std::printf("atanh mode %d: vForce %.3f, libm %.3f\n", mode, mv, ml);
    }
    double ma = 0;
    for (int mode = 0; mode < 4; ++mode) {
        for (int i = 0; i < N; ++i) {
            const double u = U(g);
            x[i] = (i % 2 ? -1 : 1) * (mode == 0   ? std::exp(-690 + 1380 * u)
                                       : mode == 1 ? 10 * u
                                       : mode == 2 ? std::exp(-5 + 10 * u)
                                                   : 1 + (u - 0.5) * 0.5);
        }
        run([](double* out, const double* a, const double*, int n) { vvatan(out, a, &n); });
        for (int i = 0; i < N; ++i)
            ma = std::max(ma, std::fabs(o[i] - std::atan(x[i])) / (std::fabs(std::atan(x[i])) * 0x1p-53));
    }
    std::printf("atan: vForce within %.3f of libm's\n", ma);
}
