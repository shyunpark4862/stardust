// Group B inverses on a fixed point set: `method,p,p0,p1,x` per line. Build it against the tree
// and against the headers before the change (git show HEAD:csrc/stats/dist/{gamma,beta,t_f,
// lifetime}.hpp into <dir>/stats/dist/, with -I<dir> before -Icsrc), run both and give the two
// outputs to inverse_check.py, which compares them and judges both by Arb's backward error.
//
//   $(cat <scratchpad>/cc.txt) -Ibench/stats/opt/b bench/stats/opt/b/inverse_sweep.cpp -o inv_new -framework Accelerate
//   ./inv_new 4000 > new.csv; ./inv_old 4000 > old.csv
//   .venv/bin/python bench/stats/opt/b/inverse_check.py new.csv old.csv
#include "stats/dist/t_f.hpp"
#include "stats/dist/lifetime.hpp"
#include "sweep.hpp"
using namespace sw;
int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 4000;
    for (int i = 0; i < N; ++i) {
        const int k = i % 5;
        const double p = k == 0 ? uni(0, 1) : k == 1 ? logu(1e-12, 0.5) : k == 2 ? 1 - logu(1e-12, 0.5) : k == 3 ? logu(1e-250, 1e-10) : uni(0.4, 0.6);
        const double lp = i % 2 ? std::log(p) : -logu(1e-8, 30);
        const double a = i % 3 == 0 ? logu(0.01, 7) : (i % 3 == 1 ? uni(0.03, 7) : 0.5 * (1 + (int)uni(0, 13)));
        const double b = (i / 3) % 3 == 0 ? logu(0.01, 7) : ((i / 3) % 3 == 1 ? uni(0.03, 7) : 0.5 * (1 + (int)uni(0, 13)));
        const double ag = i % 4 == 3 ? uni(0.1, 31) : 0.5 * (1 + (int)uni(0, 62)), s = i % 2 ? 1.0 : logu(1e-3, 1e3);
#define OUT(name, P, p0, p1v, val) printf("%s,%.17g,%.17g,%.17g,%.17g\n", name, P, p0, p1v, val)
        OUT("beta_icdf", p, a, b, Beta::icdf(p, a, b)); OUT("beta_iccdf", p, a, b, Beta::iccdf(p, a, b));
        OUT("beta_ilogcdf", lp, a, b, Beta::ilogcdf(lp, a, b)); OUT("beta_ilogccdf", lp, a, b, Beta::ilogccdf(lp, a, b));
        OUT("f_icdf", p, 2 * a, 2 * b, F::icdf(p, 2 * a, 2 * b)); OUT("f_iccdf", p, 2 * a, 2 * b, F::iccdf(p, 2 * a, 2 * b));
        OUT("f_ilogcdf", lp, 2 * a, 2 * b, F::ilogcdf(lp, 2 * a, 2 * b)); OUT("f_ilogccdf", lp, 2 * a, 2 * b, F::ilogccdf(lp, 2 * a, 2 * b));
        OUT("t_icdf", p, 2 * a, 0.0, StudentT::icdf(p, 2 * a)); OUT("t_ilogcdf", lp, 2 * a, 0.0, StudentT::ilogcdf(lp, 2 * a));
        OUT("gamma_icdf", p, ag, s, Gamma::icdf(p, ag, s)); OUT("gamma_iccdf", p, ag, s, Gamma::iccdf(p, ag, s));
        OUT("gamma_ilogcdf", lp, ag, s, Gamma::ilogcdf(lp, ag, s)); OUT("gamma_ilogccdf", lp, ag, s, Gamma::ilogccdf(lp, ag, s));
        OUT("chi2_icdf", p, 2 * ag, 0.0, ChiSquared::icdf(p, 2 * ag)); OUT("chi2_iccdf", p, 2 * ag, 0.0, ChiSquared::iccdf(p, 2 * ag));
        OUT("invgamma_icdf", p, ag, s, InverseGamma::icdf(p, ag, s)); OUT("invgamma_iccdf", p, ag, s, InverseGamma::iccdf(p, ag, s));
    }
}
