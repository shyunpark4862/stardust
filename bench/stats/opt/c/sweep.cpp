// Group C (discrete distributions): the fast paths against the precise scalar definitions.
//
// Each case draws parameters and k over the method's whole range (bulk, both tails, the
// switch points of the fast path, tiny and huge parameters), runs the method's chunk kernel
// and the scalar definition on the same inputs, and prints the largest difference in
// contract units (relative; for the log methods relative to max(1, |value|)), the share of
// elements the fast path took, and the worst input. The precise side is good to ~1e-16, so
// a fast path must stay within 2e-15.
//
//     cd <repo> && $(cat <scratchpad>/cc.txt) bench/stats/opt/c/sweep.cpp -o /tmp/sweep -framework Accelerate
//     /tmp/sweep
#include "stats/dist/discrete_more.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace sd::stats;

static std::mt19937_64 gen(12345);
static double unif(double a, double b) { return std::uniform_real_distribution<double>(a, b)(gen); }
static double logunif(double a, double b) { return std::exp(unif(std::log(a), std::log(b))); }
// a probability parameter: mid, tiny, or near 1
static double prob() {
    const double u = unif(0, 1);
    if (u < 0.5) return unif(0.01, 0.99);
    if (u < 0.75) return logunif(1e-12, 0.01);
    return 1.0 - logunif(1e-12, 0.01);
}
// a deviation in sds: bulk, tails, far tails
static double zdev() {
    const double u = unif(0, 1);
    if (u < 0.5) return unif(-3, 3);
    if (u < 0.85) return unif(-8, 8);
    return unif(-40, 40);
}

struct Worst {
    double d = 0.0, x = 0, p[3] = {0, 0, 0}, fast = 0, ref = 0;
    long n = 0, nfast = 0;
};

// kernel(Chunk<NP>, out) against scalar(x, p...): NP parameters drawn by draw(x, p).
template <std::size_t NP, class Kernel, class Scalar, class Draw>
void sweep(const char* name, bool log_method, long count, Kernel&& kernel, Scalar&& scalar, Draw&& draw) {
    Worst w;
    constexpr int B = 512;
    double x[B], p[NP][B], out[B];
    for (long done = 0; done < count; done += B) {
        for (int i = 0; i < B; ++i) {
            double q[3] = {0, 0, 0};
            draw(x[i], q);
            for (std::size_t k = 0; k < NP; ++k) p[k][i] = q[k];
        }
        Chunk<NP> c{B, x, {}, {}};
        for (std::size_t k = 0; k < NP; ++k) c.p[k] = p[k], c.same[k] = false;
        kernel(c, out);
        for (int i = 0; i < B; ++i) {
            double q[3] = {0, 0, 0};
            for (std::size_t k = 0; k < NP; ++k) q[k] = p[k][i];
            const double ref = scalar(x[i], q);
            ++w.n;
            if (out[i] == ref || (out[i] != out[i] && ref != ref)) continue;
            ++w.nfast;  // differs: the fast path took it (an equal fast result is not counted)
            const double scale = log_method ? std::max(1.0, std::fabs(ref)) : std::fabs(ref);
            const double d = std::fabs(out[i] - ref) / scale;
            if (!(d <= w.d)) w = Worst{d, x[i], {q[0], q[1], q[2]}, out[i], ref, w.n, w.nfast};
        }
    }
    std::printf(
        "%-28s max diff %.2e  (differing %5.1f%% of %ld)  worst at x=%.17g p=(%.17g, %.17g, %.17g) fast=%.17g "
        "ref=%.17g\n",
        name, w.d, 100.0 * w.nfast / w.n, w.n, w.x, w.p[0], w.p[1], w.p[2], w.fast, w.ref
    );
}

int main(int argc, char** argv) {
    const long N = argc > 1 ? std::atol(argv[1]) : 400000;

    // binomial: n from 26 to 2^53, k = mean + z sd, and k placed at the switch points
    auto draw_binom = [](double& x, double* q) {
        const double n = std::floor(logunif(20, 0x1p53)), p = prob();
        const double u = unif(0, 1), sd = std::sqrt(n * p * (1 - p));
        double k = std::floor(n * p + zdev() * sd);
        if (u < 0.05) k = std::floor(n * p * (1 + unif(-0.15, 0.15)));  // around |t| = 1/8
        if (u > 0.95) k = unif(0, 1) < 0.5 ? std::floor(unif(0, 30)) : n - std::floor(unif(0, 30));  // around 13
        x = std::min(std::max(k, 0.0), n);
        q[0] = n, q[1] = p;
    };
    sweep<2>(
        "binomial pmf", false, N, [](const Chunk<2>& c, double* o) { Binomial::pmf_kernel(c, o); },
        [](double x, const double* q) { return Binomial::pmf(x, q[0], q[1]); }, draw_binom
    );
    sweep<2>(
        "binomial logpmf", true, N, [](const Chunk<2>& c, double* o) { Binomial::logpmf_kernel(c, o); },
        [](double x, const double* q) { return Binomial::logpmf(x, q[0], q[1]); }, draw_binom
    );

    auto draw_nbinom = [](double& x, double* q) {
        const double n = unif(0, 1) < 0.3 ? std::floor(logunif(5, 1e15)) : logunif(5, 1e15), p = prob();
        const double mean = n * (1 - p) / p, sd = std::sqrt(n * (1 - p)) / p, u = unif(0, 1);
        double k = std::floor(mean + zdev() * sd);
        if (u < 0.05) k = std::floor(mean * (1 + unif(-0.15, 0.15)));
        if (u > 0.95) k = std::floor(unif(0, 30));
        x = std::min(std::max(k, 0.0), 0x1p60);
        q[0] = n, q[1] = p;
    };
    sweep<2>(
        "negative binomial pmf", false, N, [](const Chunk<2>& c, double* o) { NegativeBinomial::pmf_kernel(c, o); },
        [](double x, const double* q) { return NegativeBinomial::pmf(x, q[0], q[1]); }, draw_nbinom
    );
    sweep<2>(
        "negative binomial logpmf", true, N,
        [](const Chunk<2>& c, double* o) { NegativeBinomial::logpmf_kernel(c, o); },
        [](double x, const double* q) { return NegativeBinomial::logpmf(x, q[0], q[1]); }, draw_nbinom
    );

    auto draw_pois = [](double& x, double* q) {
        const double mu = logunif(1e-3, 1e18), u = unif(0, 1);
        double k = std::floor(mu + zdev() * std::sqrt(mu));
        if (u < 0.05) k = std::floor(mu * (1 + unif(-0.15, 0.15)));
        if (u > 0.95) k = std::floor(unif(0, 30));
        x = std::max(k, 0.0);
        q[0] = mu;
    };
    sweep<1>(
        "poisson pmf", false, N, [](const Chunk<1>& c, double* o) { Poisson::pmf_kernel(c, o); },
        [](double x, const double* q) { return Poisson::pmf(x, q[0]); }, draw_pois
    );
    sweep<1>(
        "poisson logpmf", true, N, [](const Chunk<1>& c, double* o) { Poisson::logpmf_kernel(c, o); },
        [](double x, const double* q) { return Poisson::logpmf(x, q[0]); }, draw_pois
    );

    // geometric: p over (0, 1), k over the quantiles (both tails) and the first few
    auto draw_geom = [](double& x, double* q) {
        const double p = prob(), u = unif(0, 1);
        double k = std::ceil(std::log(logunif(1e-300, 1)) / std::log1p(-p));
        if (u < 0.2) k = std::floor(unif(1, 20));
        if (u > 0.9) k = std::ceil(std::log(logunif(0.4, 0.6)) / std::log1p(-p));  // around c = 1/2
        x = std::max(k, 1.0);
        q[0] = p;
    };
    sweep<1>(
        "geometric pmf", false, N, [](const Chunk<1>& c, double* o) { Geometric::pmf_kernel(c, o); },
        [](double x, const double* q) { return Geometric::pmf(x, q[0]); }, draw_geom
    );
    sweep<1>(
        "geometric logpmf (same bits)", true, N, [](const Chunk<1>& c, double* o) { Geometric::logpmf_kernel(c, o); },
        [](double x, const double* q) { return Geometric::logpmf(x, q[0]); }, draw_geom
    );
    sweep<1>(
        "geometric logcdf", true, N, [](const Chunk<1>& c, double* o) { Geometric::logcdf_kernel(c, o); },
        [](double x, const double* q) { return Geometric::logcdf(x, q[0]); }, draw_geom
    );
    // the inverses' shortcut must return what the search returns: 0 differing
    auto draw_geom_inv = [](double& x, double* q) {
        const double u = unif(0, 1);
        q[0] = prob();
        x = u < 0.6 ? unif(0, 1) : (u < 0.8 ? logunif(1e-300, 1) : 1.0 - logunif(1e-17, 1));
        if (u > 0.97) {  // on a step of the cdf, and a few ULP around it
            const double k = std::floor(logunif(1, 1e6));
            x = Geometric::cdf(k, q[0]) * (1.0 + std::floor(unif(-4, 5)) * 0x1p-52);
        }
    };
    sweep<1>(
        "geometric icdf (same value)", false, N, [](const Chunk<1>& c, double* o) { Geometric::icdf_chunk(c, o); },
        [](double x, const double* q) { return Geometric::icdf(x, q[0]); }, draw_geom_inv
    );
    sweep<1>(
        "geometric iccdf (same value)", false, N, [](const Chunk<1>& c, double* o) { Geometric::iccdf_chunk(c, o); },
        [](double x, const double* q) { return Geometric::iccdf(x, q[0]); },
        [&](double& x, double* q) { draw_geom_inv(x, q); }
    );

    auto draw_logser = [](double& x, double* q) {
        const double u = unif(0, 1), p = prob();
        double k = std::floor(logunif(1, 1e4));
        if (u < 0.5) k = std::max(1.0, std::floor(std::log(logunif(1e-300, 1)) / std::log(p)));
        x = k;
        q[0] = p;
    };
    sweep<1>(
        "logseries pmf", false, N, [](const Chunk<1>& c, double* o) { LogSeries::pmf_kernel(c, o); },
        [](double x, const double* q) { return LogSeries::pmf(x, q[0]); }, draw_logser
    );
    sweep<1>(
        "logseries logpmf", true, N, [](const Chunk<1>& c, double* o) { LogSeries::logpmf_kernel(c, o); },
        [](double x, const double* q) { return LogSeries::logpmf(x, q[0]); }, draw_logser
    );

    // hypergeometric: totals from 121 to 2^53, K and n over (0, N), k around the mean
    auto draw_hyper = [](double& x, double* q) {
        const double N = std::floor(logunif(121, 0x1p53));
        const double K = std::floor(N * prob()), n = std::floor(N * prob());
        const double lo = std::max(0.0, n - (N - K)), hi = std::min(n, K);
        const double mean = n * (K / N), sd = std::sqrt(mean * ((N - K) / N) * ((N - n) / std::max(N - 1, 1.0)));
        double k = std::floor(mean + zdev() * sd);
        const double u = unif(0, 1);
        if (u < 0.05) k = std::floor(mean * (1 + unif(-0.15, 0.15)));
        if (u > 0.95) k = (unif(0, 1) < 0.5 ? lo + std::floor(unif(0, 30)) : hi - std::floor(unif(0, 30)));
        x = std::min(std::max(k, lo), hi);
        q[0] = N, q[1] = K, q[2] = n;
    };
    sweep<3>(
        "hypergeometric pmf", false, N, [](const Chunk<3>& c, double* o) { Hypergeometric::pmf_kernel(c, o); },
        [](double x, const double* q) { return Hypergeometric::pmf(x, q[0], q[1], q[2]); }, draw_hyper
    );
    sweep<3>(
        "hypergeometric logpmf", true, N, [](const Chunk<3>& c, double* o) { Hypergeometric::logpmf_kernel(c, o); },
        [](double x, const double* q) { return Hypergeometric::logpmf(x, q[0], q[1], q[2]); }, draw_hyper
    );

    // the log tails by one vForce log (disc::log_tail_kernel) against the scalar definitions:
    // k over the whole support, far tails included (those fall back: same bits)
    auto wide = [](auto&& draw) {
        return [draw](double& x, double* q) {
            draw(x, q);
            if (unif(0, 1) < 0.2) x = std::floor(x * logunif(1e-3, 1e3));
        };
    };
    sweep<2>(
        "binomial logcdf", true, N / 4, [](const Chunk<2>& c, double* o) { Binomial::logcdf_kernel(c, o); },
        [](double x, const double* q) { return Binomial::logcdf(x, q[0], q[1]); }, wide(draw_binom)
    );
    sweep<2>(
        "binomial logccdf", true, N / 4, [](const Chunk<2>& c, double* o) { Binomial::logccdf_kernel(c, o); },
        [](double x, const double* q) { return Binomial::logccdf(x, q[0], q[1]); }, wide(draw_binom)
    );
    sweep<2>(
        "negative binomial logcdf", true, N / 4,
        [](const Chunk<2>& c, double* o) { NegativeBinomial::logcdf_kernel(c, o); },
        [](double x, const double* q) { return NegativeBinomial::logcdf(x, q[0], q[1]); }, wide(draw_nbinom)
    );
    sweep<2>(
        "negative binomial logccdf", true, N / 4,
        [](const Chunk<2>& c, double* o) { NegativeBinomial::logccdf_kernel(c, o); },
        [](double x, const double* q) { return NegativeBinomial::logccdf(x, q[0], q[1]); }, wide(draw_nbinom)
    );
    sweep<1>(
        "poisson logcdf", true, N / 4, [](const Chunk<1>& c, double* o) { Poisson::logcdf_kernel(c, o); },
        [](double x, const double* q) { return Poisson::logcdf(x, q[0]); }, wide(draw_pois)
    );
    sweep<1>(
        "poisson logccdf", true, N / 4, [](const Chunk<1>& c, double* o) { Poisson::logccdf_kernel(c, o); },
        [](double x, const double* q) { return Poisson::logccdf(x, q[0]); }, wide(draw_pois)
    );
    return 0;
}
