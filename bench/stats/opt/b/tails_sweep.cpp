// Group B tails: cdf, ccdf, logcdf, logccdf of the gamma family (finite sums for 2a an integer),
// the beta (Boost's betainc for shapes <= 6) and F's log tails, against the double-double
// definitions (igam, ibeta and the log tails from their pieces). Prints the largest difference
// per method (relative) and dumps the points above 1e-15 for arb_check.py.
//
//   $(cat <scratchpad>/cc.txt) -Ibench/stats/opt/b bench/stats/opt/b/tails_sweep.cpp -o tails_sweep -framework Accelerate
//   ./tails_sweep 300000 dump.csv && .venv/bin/python bench/stats/opt/b/arb_check.py dump.csv
#include "stats/dist/t_f.hpp"
#include "stats/dist/lifetime.hpp"
#include "sweep.hpp"
using namespace sw;
namespace sp = sd::special;
struct Acc { const char* name; bool mixed; double w = 0; long n = 0, nd = 0; double wx = 0, wa = 0, wb = 0; };
static FILE* dumpf;
static void put(Acc& A, double x, double p0, double p1v, double got, double ref) {
    const double e = err_of(got, ref, A.mixed); ++A.n; if (got != ref) ++A.nd;
    if (e > A.w) A.w = e, A.wx = x, A.wa = p0, A.wb = p1v;
    if (e > 1e-15 && dumpf && (A.n % 7 == 0 || e > 3e-15)) fprintf(dumpf, "%s,%.17g,%.17g,%.17g,%.17g,%.17g\n", A.name, x, p0, p1v, got, ref);
}
static void rep(const Acc& A) { printf("%-18s n=%ld differing %5.1f%% max %.3g at x=%.17g p=%.17g %.17g\n", A.name, A.n, 100.0 * A.nd / A.n, A.w, A.wx, A.wa, A.wb); }
int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 300000;
    dumpf = argc > 2 ? fopen(argv[2], "w") : nullptr;
    Acc g[4] = {{"gamma_cdf", false}, {"gamma_ccdf", false}, {"gamma_logcdf", false}, {"gamma_logccdf", false}};
    Acc ig[4] = {{"invgamma_cdf", false}, {"invgamma_ccdf", false}, {"invgamma_logcdf", false}, {"invgamma_logccdf", false}};
    Acc be[4] = {{"beta_cdf", false}, {"beta_ccdf", false}, {"beta_logcdf", false}, {"beta_logccdf", false}};
    Acc ff[2] = {{"f_logcdf", false}, {"f_logccdf", false}};
    for (int i = 0; i < N; ++i) {
        // gamma family: integer and half-integer a (the finite sum), a few others
        const double a = i % 5 == 4 ? uni(0.1, 30) : 0.5 * (1 + (int)uni(0, 62));
        const double s = i % 3 == 0 ? 1.0 : (i % 3 == 1 ? 2.0 : logu(1e-3, 1e3));
        const int k = i % 4;
        const double y = k == 0 ? a + logu(1e-6, 700) : k == 1 ? uni(0, 700) : k == 2 ? a * (1 + uni(-1, 3) / std::sqrt(a)) : a * std::exp(uni(-1e-3, 1e-3));
        if (y > 0) {
            const double x = y * s; const num::DD yd = num::ratio(x, s);
            put(g[0], x, a, s, Gamma::cdf(x, a, s), gamma_detail::tail(a, yd, false));
            put(g[1], x, a, s, Gamma::ccdf(x, a, s), gamma_detail::tail(a, yd, true));
            put(g[2], x, a, s, Gamma::logcdf(x, a, s), gamma_detail::log_tail(a, yd, false));
            put(g[3], x, a, s, Gamma::logccdf(x, a, s), gamma_detail::log_tail(a, yd, true));
            const double xi = s / y; const num::DD yi = num::ratio(s, xi);
            if (std::isfinite(yi.hi) && yi.hi > 0) {
                put(ig[0], xi, a, s, InverseGamma::cdf(xi, a, s), gamma_detail::tail(a, yi, true));
                put(ig[1], xi, a, s, InverseGamma::ccdf(xi, a, s), gamma_detail::tail(a, yi, false));
                put(ig[2], xi, a, s, InverseGamma::logcdf(xi, a, s), gamma_detail::log_tail(a, yi, true));
                put(ig[3], xi, a, s, InverseGamma::logccdf(xi, a, s), gamma_detail::log_tail(a, yi, false));
            }
        }
        // beta: shapes up to 6 (Boost), a few beyond the switch
        const double ba = i % 3 == 0 ? logu(1e-4, 6.5) : (i % 3 == 1 ? uni(0, 6.5) : 0.5 * (1 + (int)uni(0, 12)));
        const double bb = (i / 3) % 3 == 0 ? logu(1e-4, 6.5) : ((i / 3) % 3 == 1 ? uni(0, 6.5) : 0.5 * (1 + (int)uni(0, 12)));
        const double m = ba / (ba + bb);
        const int kk = i % 6;
        const double xb = kk == 0 ? logu(1e-300, 1) : kk == 1 ? 1 - logu(1e-17, 1) : kk == 2 ? uni(0, 1) : kk == 3 ? m + uni(-3, 3) * std::sqrt(m * (1 - m) / (ba + bb + 1)) : kk == 4 ? m * (1 + uni(-1e-3, 1e-3)) : logu(1e-12, 1);
        if (xb > 0 && xb < 1) {
            const num::DD X{xb, 0.0}, Y = num::dd::two_sum(1.0, -xb);
            const auto r = sp::ibeta_detail::ibeta(ba, bb, xb);
            put(be[0], xb, ba, bb, Beta::cdf(xb, ba, bb), r.i);
            put(be[1], xb, ba, bb, Beta::ccdf(xb, ba, bb), r.ic);
            put(be[2], xb, ba, bb, Beta::logcdf(xb, ba, bb), beta_detail::log_tail(ba, bb, X, Y, false));
            put(be[3], xb, ba, bb, Beta::logccdf(xb, ba, bb), beta_detail::log_tail(ba, bb, X, Y, true));
        }
        // F: dfn, dfd up to 13
        const double d1 = 2 * ba * (i % 11 == 0 ? 1.05 : 1), d2 = 2 * bb;
        const double xf = kk == 0 ? logu(1e-300, 1e300) : kk == 1 ? uni(0, 5) : kk == 2 ? logu(1e-3, 1e3) : kk == 3 ? std::exp(uni(-3, 3) * std::sqrt(1 / ba + 1 / bb)) : kk == 4 ? uni(0.9, 1.1) : logu(1e-12, 1e12);
        {
            const auto gq = sp::distr_detail::f_arg(d1, d2, xf);
            put(ff[0], xf, d1, d2, F::logcdf(xf, d1, d2), beta_detail::log_tail(0.5 * d1, 0.5 * d2, gq.x, gq.y, false));
            put(ff[1], xf, d1, d2, F::logccdf(xf, d1, d2), beta_detail::log_tail(0.5 * d1, 0.5 * d2, gq.x, gq.y, true));
        }
    }
    for (auto& A : g) rep(A); for (auto& A : ig) rep(A); for (auto& A : be) rep(A); for (auto& A : ff) rep(A);
}
