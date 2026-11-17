// The noncentral distributions' paths of the bulk against the precise definitions they hand
// over to (docs/design/stats.md §3.4): the largest difference in the contract's units over
// random parameters of the guaranteed region (tools/stats_ref.py NC_RANGE) and points in the
// bulk, both tails and across the paths' own limits, and the share of points the bulk takes.
//
//   cdf, ccdf   special's *_lite            against the far-end walk (ncx2_s, ncf_s, nct_s)
//   logpdf      log_density_bulk            against the mixture from its peak / the integral
//   F pdf       pdf_bulk (the powers)       against exp of the mixture's log
//   logcdf, logccdf  log_tail_bulk          against log_tail on the wide sums (mixed bound)
//   icdf, iccdf Newton on the bulk's sums   against the bracketing solver on the wide tails,
//               in units of max(1, kappa), kappa = tail / (|x| f)
//
//   $(cat <scratchpad>/cc.txt) bench/stats/opt/d/sweep.cpp -o /tmp/sweep && /tmp/sweep [n]
#include "stats/dist/noncentral.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>

using namespace sd::stats;
namespace nd = sd::special::nc_detail;

struct Worst {
    double e = 0.0, a[4] = {0, 0, 0, 0};
    long fast = 0, total = 0;
    void see(double err, const double* p) {
        if (err > e) {
            e = err;
            for (int k = 0; k < 4; ++k) a[k] = p[k];
        }
    }
    void print(const char* name) const {
        std::printf("%-28s bulk share %5.1f%%  worst %.3g  at %.17g %.17g %.17g %.17g\n", name,
                    total ? 100.0 * fast / total : 0.0, e, a[0], a[1], a[2], a[3]);
    }
};
static double rel(double f, double r) { return std::fabs(f - r) / std::fabs(r); }
static double mixed(sd::stats::num::DD f, sd::stats::num::DD r) {
    return std::fabs((f.hi - r.hi) + (f.lo - r.lo)) / std::max(1.0, std::fabs(r.hi));
}

int main(int argc, char** argv) {
    const long n = argc > 1 ? std::atol(argv[1]) : 200000;
    std::mt19937_64 g(20261116);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto lu = [&](double lo, double hi) { return std::exp(std::log(lo) + U(g) * std::log(hi / lo)); };
    // a point of a positive variable about its middle m: by the quantile, near m, far tails
    auto point = [&](long i, double m, double q) {
        const double u = U(g);
        switch (i % 4) {
            case 0: return q;
            case 1: return m * std::exp(4.0 * (u - 0.5));
            case 2: return m * std::exp(60.0 * (u - 0.7));
            default: return m * (1.0 + 0.2 * (u - 0.5));
        }
    };
    Worst c[2], f[2], t[2], pc, pf, pt, ic[2], jf[2], it[2], pfd, lc[2], lf[2], lt[2];
    auto mixed1 = [](double v, double r) { return std::fabs(v - r) / std::max(1.0, std::fabs(r)); };
    for (long i = 0; i < n; ++i) {
        const bool small = i % 2 == 0;  // half the parameters where the bulk's paths live
        {  // chi-square
            const double df = small ? lu(1e-2, 50.0) : lu(1e-2, 1e4), nc = small ? lu(1e-3, 50.0) : lu(1e-3, 1e4);
            const double u = U(g), q = NoncentralChiSquared::icdf(u, df, nc);
            const double a[4] = {df, nc, point(i, df + nc, q), 0.0};
            if (a[2] > 0.0 && std::isfinite(a[2])) {
                for (int up = 0; up < 2; ++up) {
                    double v;
                    ++c[up].total;
                    if (!nd::ncx2_lite(df, nc, a[2], up, v)) continue;
                    ++c[up].fast;
                    const double r = std::min(1.0, sd::special::value(nd::ncx2_s(df, nc, a[2], up)));
                    if (r >= 1e-300) c[up].see(rel(v, r), a);
                    double lv;
                    ++lc[up].total;
                    if (ncd::log_tail_bulk([&](bool u_, double& w) { return nd::ncx2_lite(df, nc, a[2], u_, w); }, up, lv)) {
                        ++lc[up].fast;
                        lc[up].see(mixed1(lv, ncd::log_tail([&](bool u_) { return NoncentralChiSquared::tail(a[2], df, nc, u_); }, up)), a);
                    }
                }
                num::DD l;
                ++pc.total;
                if (NoncentralChiSquared::log_density_bulk(a[2], df, nc, l)) {
                    ++pc.fast;
                    pc.see(mixed(l, NoncentralChiSquared::series(a[2], df, nc).log), a);
                }
            }
            for (int up = 0; up < 2; ++up) {
                const double p = i % 3 == 0 ? std::exp(-700.0 * U(g) * U(g)) : u;
                const double b[4] = {df, nc, p, 0.0};
                const ncd::Target tg = ncd::target_p(p, up);
                const double x0 = nd::ncx2_start(df, nc, up ? 1.0 - p : p);
                double x;
                ++ic[up].total;
                if (!(p > 0.0 && p < 1.0) ||
                    !ncd::newton([&](double x_, bool s, double& v, double& d) { return nd::ncx2_lite(df, nc, x_, s, v, &d); }, tg, x0,
                                 true, x))
                    continue;
                ++ic[up].fast;
                const double r = ncd::invert([&](double x_, bool s) { return NoncentralChiSquared::tail(x_, df, nc, s); }, tg, x0, true);
                const double kappa = std::min(p, 1.0 - p) / (r * NoncentralChiSquared::pdf(r, df, nc));
                ic[up].see(rel(x, r) / std::max(1.0, kappa), b);
            }
        }
        {  // F
            const double dfn = small ? lu(1e-2, 50.0) : lu(1e-2, 1e4), dfd = small ? lu(1e-2, 100.0) : lu(1e-2, 1e4);
            const double nc = small ? lu(1e-3, 50.0) : lu(1e-3, 1e4);
            const double u = U(g), q = NoncentralF::icdf(u, dfn, dfd, nc);
            const double a[4] = {dfn, dfd, nc, point(i, (dfn + nc) / dfn, q)};
            if (a[3] > 0.0 && std::isfinite(a[3])) {
                for (int up = 0; up < 2; ++up) {
                    double v;
                    ++f[up].total;
                    if (!nd::ncf_lite(dfn, dfd, nc, a[3], up, v)) continue;
                    ++f[up].fast;
                    const double r = std::min(1.0, sd::special::value(nd::ncf_s(dfn, dfd, nc, a[3], up)));
                    if (r >= 1e-300) f[up].see(rel(v, r), a);
                    double lv;
                    ++lf[up].total;
                    if (ncd::log_tail_bulk([&](bool u_, double& w) { return nd::ncf_lite(dfn, dfd, nc, a[3], u_, w); }, up, lv)) {
                        ++lf[up].fast;
                        lf[up].see(mixed1(lv, ncd::log_tail([&](bool u_) { return NoncentralF::tail(a[3], dfn, dfd, nc, u_); }, up)), a);
                    }
                }
                double pv;
                ++pfd.total;
                if (NoncentralF::pdf_bulk(a[3], dfn, dfd, nc, pv)) {
                    ++pfd.fast;
                    pfd.see(rel(pv, num::exp_dd(NoncentralF::series(a[3], dfn, dfd, nc).log)), a);
                }
                num::DD l;
                ++pf.total;
                if (NoncentralF::log_density_bulk(a[3], dfn, dfd, nc, l)) {
                    ++pf.fast;
                    pf.see(mixed(l, NoncentralF::series(a[3], dfn, dfd, nc).log), a);
                }
            }
            for (int up = 0; up < 2; ++up) {
                const double p = i % 3 == 0 ? std::exp(-700.0 * U(g) * U(g)) : u;
                const double b[4] = {dfn, dfd, nc, p};
                const ncd::Target tg = ncd::target_p(p, up);
                const double cc = (dfn + 2.0 * nc) / (dfn + nc), k = (dfn + nc) / cc;
                const double x0 = cc * k / dfn * sd::special::Fdtri::f(k, dfd, up ? 1.0 - p : p);
                double x;
                ++jf[up].total;
                if (!(p > 0.0 && p < 1.0) ||
                    !ncd::newton([&](double x_, bool s, double& v, double& d) { return nd::ncf_lite(dfn, dfd, nc, x_, s, v, &d); }, tg,
                                 x0, true, x))
                    continue;
                ++jf[up].fast;
                const double r = ncd::invert([&](double x_, bool s) { return NoncentralF::tail(x_, dfn, dfd, nc, s); }, tg, x0, true);
                const double kappa = std::min(p, 1.0 - p) / (r * NoncentralF::pdf(r, dfn, dfd, nc));
                jf[up].see(rel(x, r) / std::max(1.0, kappa), b);
            }
        }
        {  // t
            const double df = small ? lu(1e-1, 100.0) : lu(1e-1, 1e5);
            const double nc = (small ? lu(1e-3, 8.0) : lu(1e-3, 100.0)) * (U(g) < 0.5 ? -1.0 : 1.0);
            const double u = U(g), q = NoncentralT::icdf(u, df, nc);
            double x;
            switch (i % 4) {
                case 0: x = q; break;
                case 1: x = nc + (std::fabs(nc) + 1.0) * std::sinh(8.0 * (U(g) - 0.5)); break;
                case 2: x = std::sinh(80.0 * (U(g) - 0.5)); break;
                default: x = (U(g) - 0.5) * 2.0 * std::exp(-30.0 * U(g)); break;  // about 0
            }
            const double a[4] = {df, nc, x, 0.0};
            if (std::isfinite(x)) {
                for (int up = 0; up < 2; ++up) {
                    double v;
                    ++t[up].total;
                    if (!nd::nct_lite(df, nc, x, up, v)) continue;
                    ++t[up].fast;
                    const double r = sd::special::value(nd::nct_s(df, nc, x, up));
                    if (r >= 1e-300) t[up].see(rel(v, r), a);
                    double lv;
                    ++lt[up].total;
                    if (ncd::log_tail_bulk([&](bool u_, double& w) { return nd::nct_lite(df, nc, x, u_, w); }, up, lv)) {
                        ++lt[up].fast;
                        lt[up].see(mixed1(lv, ncd::log_tail([&](bool u_) { return NoncentralT::tail(x, df, nc, u_); }, up)), a);
                    }
                }
                num::DD l;
                ++pt.total;
                if (NoncentralT::log_density_bulk(x, df, nc, l)) {
                    ++pt.fast;
                    pt.see(mixed(l, nct_detail::log_moment(1, df, nc, x)), a);
                }
            }
            for (int up = 0; up < 2; ++up) {
                const double p = i % 3 == 0 ? std::exp(-700.0 * U(g) * U(g)) : u;
                const double b[4] = {df, nc, p, 0.0};
                if (!(p > 0.0 && p < 1.0)) continue;
                ++it[up].total;
                // the method itself (its start is inside it) against the solver alone
                const double got = NoncentralT::inv(p, df, nc, up);
                const double r = NoncentralT::inv_precise(p, df, nc, up);
                if (got == r) continue;  // (the same path, or the same bits)
                ++it[up].fast;
                const double kappa = std::min(p, 1.0 - p) / (std::fabs(r) * NoncentralT::pdf(r, df, nc));
                it[up].see(rel(got, r) / std::max(1.0, kappa), b);
            }
        }
    }
    c[0].print("chi-square cdf"), c[1].print("chi-square ccdf"), pc.print("chi-square logpdf");
    lc[0].print("chi-square logcdf"), lc[1].print("chi-square logccdf");
    ic[0].print("chi-square icdf"), ic[1].print("chi-square iccdf");
    f[0].print("F cdf"), f[1].print("F ccdf"), pf.print("F logpdf"), pfd.print("F pdf (powers)");
    lf[0].print("F logcdf"), lf[1].print("F logccdf");
    jf[0].print("F icdf"), jf[1].print("F iccdf");
    t[0].print("t cdf"), t[1].print("t ccdf"), pt.print("t logpdf");
    lt[0].print("t logcdf"), lt[1].print("t logccdf");
    it[0].print("t icdf (differing points)"), it[1].print("t iccdf (differing points)");
}
