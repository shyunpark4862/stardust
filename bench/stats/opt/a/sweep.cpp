// Fast paths of group A (the closed-form continuous distributions) against their precise
// scalar definitions: see sweep.hpp. Build and run: bench/stats/opt/a/build.sh sweep && ./sweep
// (`./sweep Weibull` for one distribution, `--quick` for a tenth of the points).
#include "sweep.hpp"
#include "stats/dist/normal.hpp"
#include "stats/dist/lognormal.hpp"
#include "stats/dist/exponential.hpp"
#include "stats/dist/uniform.hpp"
#include "stats/dist/laplace.hpp"
#include "stats/dist/cauchy.hpp"
#include "stats/dist/weibull.hpp"
#include "stats/dist/extreme.hpp"
#include "stats/dist/lifetime.hpp"
using namespace sw;
using P1 = std::vector<std::array<double, 1>>;
using P2 = std::vector<std::array<double, 2>>;
using P3 = std::vector<std::array<double, 3>>;
int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--quick") g_quick = true;
        else g_only = argv[i];
    }
    const P2 loc = {{1.5, 2.0},     {0.0, 1.0},   {-3.0, 0.01},     {1e5, 1.0},    {-1e-3, 1e-8},
                    {1e200, 1e190}, {0.3, 1e150}, {1e-300, 1e-305}, {7.0, 3e-320}, {-2.0, 1e307}};
    const P1 sc = {{2.0}, {1.0}, {1e-3}, {1e5}, {1e-300}, {1e300}, {3e-320}, {0.7}};
    const P2 wb = {{1.7, 2.0},    {1.0, 1.0},   {0.5, 3.0},  {0.05, 1.0},  {8.0, 1e-3}, {300.0, 2.0}, {1e4, 1e10},
                   {2.5, 1e-300}, {0.3, 1e250}, {1e-3, 1.0}, {7.7e8, 3.0}, {3e9, 1.0},  {1e15, 2.0}};
    SW_M(2, Normal, pdf, kRel, kX, loc);
    SW_M(2, Normal, logpdf, kMixed, kX, loc);
    SW_M(2, Normal, cdf, kRel, kX, loc);
    SW_M(2, Normal, ccdf, kRel, kX, loc);
    SW_M(2, Normal, logcdf, kRel, kX, loc);
    SW_M(2, Normal, logccdf, kRel, kX, loc);
    SW_M(2, Normal, icdf, kInv, kP, loc);
    SW_M(2, Normal, iccdf, kInv, kP, loc);
    SW_M(1, Exponential, pdf, kRel, kX, sc);
    SW_M(1, Exponential, logpdf, kMixed, kX, sc);
    SW_M(1, Exponential, cdf, kRel, kX, sc);
    SW_M(1, Exponential, ccdf, kRel, kX, sc);
    SW_M(1, Exponential, logcdf, kRel, kX, sc);
    SW_M(1, Exponential, iccdf, kInv, kP, sc);
    SW_M(2, Laplace, pdf, kRel, kX, loc);
    SW_M(2, Laplace, logpdf, kMixed, kX, loc);
    SW_M(2, Laplace, cdf, kRel, kX, loc);
    SW_M(2, Laplace, ccdf, kRel, kX, loc);
    SW_M(2, Laplace, logcdf, kRel, kX, loc);
    SW_M(2, Laplace, logccdf, kRel, kX, loc);
    SW_M(2, Laplace, icdf, kInv, kP, loc);
    SW_M(2, Laplace, iccdf, kInv, kP, loc);
    SW_M(2, Weibull, pdf, kRel, kX, wb);
    SW_M(2, Weibull, logpdf, kMixed, kX, wb);
    SW_M(2, Weibull, cdf, kRel, kX, wb);
    SW_M(2, Weibull, ccdf, kRel, kX, wb);
    SW_M(2, Weibull, logcdf, kRel, kX, wb);
    SW_M(2, Weibull, logccdf, kRel, kX, wb);
    SW_M(2, Weibull, icdf, kInv, kP, wb);
    SW_M(2, Weibull, iccdf, kInv, kP, wb);
    SW_M(2, Weibull, ilogccdf, kInv, kLogP, wb);
    SW_M(2, Logistic, pdf, kRel, kX, loc);
    SW_M(2, Logistic, logpdf, kMixed, kX, loc);
    SW_M(2, Logistic, cdf, kRel, kX, loc);
    SW_M(2, Logistic, ccdf, kRel, kX, loc);
    SW_M(2, Logistic, logcdf, kRel, kX, loc);
    SW_M(2, Logistic, logccdf, kRel, kX, loc);
    SW_M(2, Logistic, icdf, kInv, kP, loc);
    SW_M(2, Logistic, iccdf, kInv, kP, loc);
    SW_M(2, Gumbel, pdf, kRel, kX, loc);
    SW_M(2, Gumbel, logpdf, kMixed, kX, loc);
    SW_M(2, Gumbel, cdf, kRel, kX, loc);
    SW_M(2, Gumbel, ccdf, kRel, kX, loc);
    SW_M(2, Gumbel, logcdf, kRel, kX, loc);
    SW_M(2, Gumbel, logccdf, kRel, kX, loc);
    SW_M(2, Gumbel, icdf, kInv, kP, loc);
    SW_M(2, Gumbel, iccdf, kInv, kP, loc);
    const P3 gv = {{0.2, 1.5, 2.0},   {-0.3, 0.0, 1.0},        {0.0, 0.5, 2.0},  {1.0, -3.0, 0.01},
                   {-1.5, 1e5, 1.0},  {3.0, 0.0, 1e150},       {1e-5, 1.0, 2.0}, {-1e-12, 0.0, 1.0},
                   {25.0, 2.0, 3.0},  {-40.0, 1e-300, 1e-305}, {0.5, 0.0, 1.0},  {1.29e-9, 9.5, 1.0},
                   {-2e-9, 0.0, 3.0}, {1e-11, 1.0, 1.0},       {4e-8, -2.0, 0.5}};
    SW_M(3, GEV, pdf, kRel, kX, gv);
    SW_M(3, GEV, logpdf, kMixed, kX, gv);
    SW_M(3, GEV, cdf, kRel, kX, gv);
    SW_M(3, GEV, ccdf, kRel, kX, gv);
    SW_M(3, GEV, logcdf, kRel, kX, gv);
    SW_M(3, GEV, logccdf, kRel, kX, gv);
    const P2 pa = {{3.0, 1.0},   {1.0, 2.0},  {0.5, 1e-3},   {0.05, 1.0}, {40.0, 1e5}, {1e4, 3.0}, {2.5, 1e-300},
                   {0.3, 1e250}, {1e-3, 1.0}, {7.0, 3e-320}, {6e8, 2.0},  {5e9, 1.0},  {1e14, 3.0}};
    SW_M(2, Pareto, pdf, kRel, kX, pa);
    SW_M(2, Pareto, logpdf, kMixed, kX, pa);
    SW_M(2, Pareto, cdf, kRel, kX, pa);
    SW_M(2, Pareto, ccdf, kRel, kX, pa);
    SW_M(2, Pareto, logcdf, kRel, kX, pa);
    SW_M(2, Pareto, logccdf, kRel, kX, pa);
    SW_M(2, Pareto, icdf, kInv, kP, pa);
    SW_M(2, Pareto, iccdf, kInv, kP, pa);
    SW_M(1, Rayleigh, pdf, kRel, kX, sc);
    SW_M(1, Rayleigh, logpdf, kMixed, kX, sc);
    SW_M(1, Rayleigh, cdf, kRel, kX, sc);
    SW_M(1, Rayleigh, ccdf, kRel, kX, sc);
    SW_M(1, Rayleigh, logcdf, kRel, kX, sc);
    SW_M(1, Rayleigh, logccdf, kRel, kX, sc);
    SW_M(1, Rayleigh, icdf, kInv, kP, sc);
    SW_M(1, Rayleigh, iccdf, kInv, kP, sc);
    const P2 ln = {{0.5, 0.8},    {0.0, 1.0},  {5.0, 0.1},  {-3.0, 0.01}, {100.0, 3.0},
                   {-600.0, 2.0}, {0.0, 30.0}, {2.0, 1e-3}, {650.0, 5.0}, {0.3, 1e-6}};
    const P2 un = {{-1.0, 3.0},        {0.0, 1.0},        {1e-300, 2e-300}, {-1e308, 1e308},
                   {5.0, 5.000000001}, {-3e-320, 7e-310}, {1e10, 1e300}};
    SW_M(2, LogNormal, pdf, kRel, kX, ln);
    SW_M(2, LogNormal, logpdf, kMixed, kX, ln);
    SW_M(2, LogNormal, cdf, kRel, kX, ln);
    SW_M(2, LogNormal, ccdf, kRel, kX, ln);
    SW_M(2, LogNormal, logcdf, kRel, kX, ln);
    SW_M(2, LogNormal, logccdf, kRel, kX, ln);
    SW_M(2, LogNormal, icdf, kInv, kP, ln);
    SW_M(2, LogNormal, iccdf, kInv, kP, ln);
    SW_M(2, Cauchy, pdf, kRel, kX, loc);
    SW_M(2, Cauchy, logpdf, kMixed, kX, loc);
    SW_M(2, Cauchy, cdf, kRel, kX, loc);
    SW_M(2, Cauchy, ccdf, kRel, kX, loc);
    SW_M(2, Cauchy, logcdf, kRel, kX, loc);
    SW_M(2, Cauchy, logccdf, kRel, kX, loc);
    SW_M(2, Uniform, icdf, kInv, kP, un);
    SW_M(2, Uniform, iccdf, kInv, kP, un);
    SW_M(3, GEV, icdf, kInv, kP, gv);
    SW_M(3, GEV, iccdf, kInv, kP, gv);
    std::printf("worst %.3g; %d methods outside 2e-15 or with differing bits\n", g_worst, g_bad);
    return g_bad != 0;
}
