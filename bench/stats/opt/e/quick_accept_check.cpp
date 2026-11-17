// Group E: the plain-double first test of the Poisson (PTRS) and binomial (BTRS) rejections
// (stats/sample_dist.hpp, quick_accept) against the double-double log pmf it stands in for:
// the plain value must lie within the margin the test allows, for every k a draw can propose.
// Prints the largest |plain - exact| / margin (must stay below 1).
//
//   $(cat <scratchpad>/cc.txt) bench/stats/opt/e/quick_accept_check.cpp -o qa && ./qa
#include "stats/sample.hpp"
#include "stats/sample_dist.hpp"

#include <cstdio>
#include <random>

using namespace sd::stats;

int main() {
    std::mt19937_64 g(7);
    double worst = 0.0;
    long long n = 0;
    for (double mu : {10.0, 10.5, 12.3, 37.0, 100.0, 1e3, 12345.6, 1e6, 1e9, 1e12, 1e15, 1e18, 1e300}) {
        const smp::PoissonPrep s = smp::poisson_prep(mu);
        std::normal_distribution<double> N(0.0, 1.0);
        for (int i = 0; i < 200000; ++i) {
            const double k = std::floor(std::fabs(mu + 8.0 * std::sqrt(mu) * N(g)));
            const double klm = k * s.log_mu, lf = smp::log_fact(k);
            const double ld = (klm - s.mu) - lf, mag = std::fabs(klm) + s.mu + lf + 2.0 * k + 1.0;
            const num::DD l = Poisson::log_pmf_dd(k, mu);
            const double margin = 1e-11 + 0x1p-48 * mag;
            worst = std::max(worst, std::fabs((ld - l.hi) - l.lo) / margin);
            ++n;
        }
    }
    std::printf("Poisson:  %lld proposals, largest |plain - exact| / margin = %.3f\n", n, worst);
    worst = 0.0, n = 0;
    for (double nn : {20.0, 37.0, 100.0, 1e4, 123457.0, 1e8, 1e12, 1e15})
        for (double p : {0.5, 0.3, 0.01, 1e-3, 1e-6, 1e-10}) {
            if (nn * p < 10.0) continue;
            const smp::BinomPrep s = smp::binom_prep(nn, p, num::dd::two_sum(1.0, -p));
            std::normal_distribution<double> N(0.0, 1.0);
            const double sd = std::sqrt(nn * p * (1.0 - p));
            const num::DD lm = binom_detail::log_pmf(s.m, s.n, s.p, s.q);
            for (int i = 0; i < 100000; ++i) {
                const double k = std::min(nn, std::floor(std::fabs(nn * p + 8.0 * sd * N(g))));
                const double lfk = smp::log_fact(k) + smp::log_fact(s.n - k), t = (k - s.m) * s.lpq;
                const double ld = (s.lfm - lfk) + t, mag = s.lfm + lfk + std::fabs(t) + 4.0 * s.n + 1.0;
                const num::DD d = binom_detail::log_pmf(k, s.n, s.p, s.q) - lm;
                const double margin = 1e-11 + 0x1p-48 * mag;
                worst = std::max(worst, std::fabs((ld - d.hi) - d.lo) / margin);
                ++n;
            }
        }
    std::printf("Binomial: %lld proposals, largest |plain - exact| / margin = %.3f\n", n, worst);
    return 0;
}
