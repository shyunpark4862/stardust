// Group C: the scalar definitions of the sum-based cdfs and of every inverse on a fixed set
// of inputs, one value per line (hex floats), to compare two versions of the headers:
//
//     clang++ -I<old headers> ... dump.cpp -o dump_old;  clang++ ... dump.cpp -o dump_new
//     ./dump_old > old.txt; ./dump_new > new.txt; python bench/stats/opt/c/dump_compare.py old.txt new.txt
//
// Lines: "<case> <method> <value>". The cdfs of the hypergeometric and beta-binomial changed
// algorithm (the double-double recurrence); the inverses must agree up to the contract's
// backward error (dump_compare.py checks an inverse that differs against the cdf at both).
#include "stats/dist/discrete_more.hpp"

#include <cstdio>
#include <random>

using namespace sd::stats;

static std::mt19937_64 gen(777);
static double unif(double a, double b) { return std::uniform_real_distribution<double>(a, b)(gen); }
static double logunif(double a, double b) { return std::exp(unif(std::log(a), std::log(b))); }
static double tail_p() {
    const double u = unif(0, 1);
    return u < 0.6 ? unif(0, 1) : (u < 0.8 ? logunif(1e-300, 1) : 1.0 - logunif(1e-16, 1));
}

template <class D, class... P>
void values(const char* name, double x, P... p) {
    std::printf("%s x=%a cdf %a\n%s x=%a ccdf %a\n%s x=%a logcdf %a\n%s x=%a logccdf %a\n", name, x, D::cdf(x, p...), name, x,
                D::ccdf(x, p...), name, x, D::logcdf(x, p...), name, x, D::logccdf(x, p...));
}
template <class D, class... P>
void inverses(const char* name, double t, P... p) {
    const double a = D::icdf(t, p...), b = D::iccdf(t, p...);
    const double lt = std::log(t);
    std::printf("%s t=%a icdf %a %a\n%s t=%a iccdf %a %a\n", name, t, a, D::cdf(a, p...), name, t, b, D::ccdf(b, p...));
    std::printf("%s t=%a ilogcdf %a %a\n%s t=%a ilogccdf %a %a\n", name, lt, D::ilogcdf(lt, p...), 0.0, name, lt,
                D::ilogccdf(lt, p...), 0.0);
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 3000;
    char name[200];
    for (int i = 0; i < n; ++i) {
        {  // hypergeometric beyond the exact totals
            const double N = std::floor(logunif(121, i % 4 == 0 ? 3e6 : 3e4));
            const double K = std::floor(unif(0, N + 1)), m = std::floor(unif(0, N + 1));
            const double lo = Hypergeometric::lo_of(N, K, m), hi = Hypergeometric::hi_of(N, K, m);
            std::snprintf(name, sizeof name, "hyper(%.0f,%.0f,%.0f)", N, K, m);
            values<Hypergeometric>(name, std::floor(unif(lo, hi + 1)), N, K, m);
            values<Hypergeometric>(name, Hypergeometric::icdf(tail_p(), N, K, m), N, K, m);
            inverses<Hypergeometric>(name, tail_p(), N, K, m);
        }
        {  // beta-binomial: unimodal and not (a + b <= 2)
            const double nn = std::floor(logunif(1, i % 4 == 0 ? 3e5 : 3e3));
            const double a = logunif(1e-2, 1e3), b = i % 3 == 0 ? logunif(1e-2, 1.5) : logunif(1e-2, 1e3);
            std::snprintf(name, sizeof name, "betabinom(%.0f,%a,%a)", nn, a, b);
            values<BetaBinomial>(name, std::floor(unif(0, nn + 1)), nn, a, b);
            values<BetaBinomial>(name, BetaBinomial::icdf(tail_p(), nn, a, b), nn, a, b);
            inverses<BetaBinomial>(name, tail_p(), nn, a, b);
        }
        {
            const double nn = std::floor(logunif(1, 1e12)), p = unif(0, 1) < 0.5 ? unif(0, 1) : logunif(1e-9, 1);
            std::snprintf(name, sizeof name, "binom(%.0f,%a)", nn, p);
            inverses<Binomial>(name, tail_p(), nn, p);
            const double mu = logunif(1e-3, 1e12);
            std::snprintf(name, sizeof name, "poisson(%a)", mu);
            inverses<Poisson>(name, tail_p(), mu);
            const double r = logunif(1e-2, 1e8);
            std::snprintf(name, sizeof name, "nbinom(%a,%a)", r, p);
            inverses<NegativeBinomial>(name, tail_p(), r, p);
            std::snprintf(name, sizeof name, "geom(%a)", p);
            inverses<Geometric>(name, tail_p(), p);
            const double pl = unif(0, 1) < 0.5 ? unif(0, 1) : 1.0 - logunif(1e-6, 1);
            std::snprintf(name, sizeof name, "logser(%a)", pl);
            inverses<LogSeries>(name, tail_p(), pl);
            const double az = 1.0 + logunif(1e-2, 20);
            std::snprintf(name, sizeof name, "zipf(%a)", az);
            inverses<Zipf>(name, tail_p(), az);
        }
    }
    return 0;
}
