// Group B densities: the chunk kernels (dens::kernel, csrc/stats/dist/gamma.hpp) against the
// double-double definitions (pdf_dd, logpdf_dd) over the whole input range: bulk, both tails,
// the ends, tiny and huge shapes. Prints the largest difference per method in contract units
// and, with a file name, dumps the points that differ by more than 1e-15 for arb_check.py
// (which says which side is right). Then times kernel and one-element calls.
//
//   $(cat <scratchpad>/cc.txt) -Ibench/stats/opt/b bench/stats/opt/b/density_sweep.cpp -o density_sweep -framework Accelerate
//   ./density_sweep 400000 dump.csv && .venv/bin/python bench/stats/opt/b/arb_check.py dump.csv
//   ./density_sweep 100000            (the timings too)
#include "stats/dist/gamma.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/dist/lifetime.hpp"
#include "sweep.hpp"
using namespace sw;
static double pick(double bulk_lo, double bulk_hi) { return uni(bulk_lo, bulk_hi); }
int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 400000;
    if (argc > 2) dump = fopen(argv[2], "w");
    std::vector<double> x(N), a(N), s(N), df(N), xb(N), b(N), xt(N), xf(N), xi(N);
    auto shape = [&](int i) {
        switch (i % 7) {
            case 0: return logu(1e-6, 1e6);
            case 1: return logu(1e-300, 1e-3);
            case 2: return logu(100, 1e15);
            case 3: return (double)(1 + i % 23) * 0.5;
            default: return uni(0.01, 14);
        }
    };
    for (int i = 0; i < N; ++i) {
        a[i] = shape(i);
        b[i] = shape(i / 7 + i);
        s[i] = i % 2 ? logu(1e-3, 1e3) : (i % 4 ? 1.0 : logu(1e-290, 1e290));
        const int k = i % 6;
        double y = k == 0   ? logu(1e-300, 1e3)
                   : k == 1 ? a[i] * std::exp(uni(-3, 3))
                   : k == 2 ? uni(0, 50)
                   : k == 3 ? logu(1e-8, 10)
                   : k == 4 ? a[i] * (1 + uni(-3, 3) / std::sqrt(a[i]))
                            : uni(0, 3 * a[i] + 5);
        x[i] = y * s[i];
        xi[i] = s[i] / y;
        df[i] = 2 * a[i];
        double m = a[i] / (a[i] + b[i]);
        xb[i] = k == 0   ? logu(1e-300, 1)
                : k == 1 ? 1 - logu(1e-17, 1)
                : k == 2 ? uni(0, 1)
                : k == 3 ? std::min(1.0, std::max(0.0, m + uni(-3, 3) * std::sqrt(m * (1 - m) / (a[i] + b[i] + 1))))
                : k == 4 ? logu(1e-10, 0.5)
                         : uni(0.4, 0.6);
        xt[i] = (k == 0   ? logu(1e-300, 1e300)
                 : k == 1 ? uni(-5, 5)
                 : k == 2 ? logu(1e-170, 1e-140)
                 : k == 3 ? logu(1e-3, 1e3)
                 : k == 4 ? uni(-40, 40)
                          : logu(1e140, 1e160)) *
                (i % 2 ? 1 : -1);
        xf[i] = k == 0   ? logu(1e-300, 1e300)
                : k == 1 ? uni(0, 5)
                : k == 2 ? logu(1e-3, 1e3)
                : k == 3 ? std::exp(uni(-3, 3) * std::sqrt(1 / a[i] + 1 / b[i]))
                : k == 4 ? uni(0.5, 1.5)
                         : logu(1e-12, 1e12);
        if (i % 1000 == 7) {
            double sp = (i % 3 == 0) ? 0.0 : (i % 3 == 1 ? -1.0 : INFINITY);
            x[i] = xi[i] = xf[i] = sp;
            xt[i] = i % 3 ? INFINITY : -INFINITY;
            xb[i] = i % 3 == 0 ? 0.0 : (i % 3 == 1 ? 1.0 : 2.0);
        }
        if (i % 1000 == 8) {
            x[i] = xi[i] = xb[i] = xf[i] = 5e-324;
            xt[i] = 0.0;
        }
    }
    std::array<std::vector<double>, 2> P{a, s}, PB{a, b}, PF{df, b};
    std::array<std::vector<double>, 1> Q{df};
    sweep<2>("gamma_pdf", Gamma::pdf_chunk, [&](size_t i) { return Gamma::pdf_dd(x[i], a[i], s[i]); }, x, P, false);
    sweep<2>(
        "gamma_logpdf", Gamma::logpdf_chunk, [&](size_t i) { return Gamma::logpdf_dd(x[i], a[i], s[i]); }, x, P, true
    );
    sweep<1>(
        "chi2_pdf", ChiSquared::pdf_chunk, [&](size_t i) { return Gamma::pdf_dd(x[i], 0.5 * df[i], 2.0); }, x, Q, false
    );
    sweep<1>(
        "chi2_logpdf", ChiSquared::logpdf_chunk, [&](size_t i) { return Gamma::logpdf_dd(x[i], 0.5 * df[i], 2.0); }, x,
        Q, true
    );
    sweep<2>(
        "invgamma_pdf", InverseGamma::pdf_chunk, [&](size_t i) { return InverseGamma::pdf_dd(xi[i], a[i], s[i]); }, xi,
        P, false
    );
    sweep<2>(
        "invgamma_logpdf", InverseGamma::logpdf_chunk,
        [&](size_t i) { return InverseGamma::logpdf_dd(xi[i], a[i], s[i]); }, xi, P, true
    );
    sweep<2>("beta_pdf", Beta::pdf_chunk, [&](size_t i) { return Beta::pdf_dd(xb[i], a[i], b[i]); }, xb, PB, false);
    sweep<2>(
        "beta_logpdf", Beta::logpdf_chunk, [&](size_t i) { return Beta::logpdf_dd(xb[i], a[i], b[i]); }, xb, PB, true
    );
    sweep<1>("t_pdf", StudentT::pdf_chunk, [&](size_t i) { return StudentT::pdf_dd(xt[i], df[i]); }, xt, Q, false);
    sweep<1>(
        "t_logpdf", StudentT::logpdf_chunk, [&](size_t i) { return StudentT::logpdf_dd(xt[i], df[i]); }, xt, Q, true
    );
    sweep<2>("f_pdf", F::pdf_chunk, [&](size_t i) { return F::pdf_dd(xf[i], df[i], b[i]); }, xf, PF, false);
    sweep<2>("f_logpdf", F::logpdf_chunk, [&](size_t i) { return F::logpdf_dd(xf[i], df[i], b[i]); }, xf, PF, true);
    if (argc > 2) return 0;
    // timing at the bench's parameters
    const size_t M = 100000;
    std::vector<double> u(M), xg(M), xc(M), xbb(M), xtt(M), xff(M), xii(M);
    for (size_t i = 0; i < M; ++i) {
        u[i] = uni(0, 1);
        xg[i] = Gamma::icdf(u[i], 2.5, 3.0);
        xc[i] = ChiSquared::icdf(u[i], 4.5);
        xbb[i] = Beta::icdf(u[i], 2.5, 0.7);
        xtt[i] = StudentT::icdf(u[i], 3.5);
        xff[i] = F::icdf(u[i], 3.0, 11.0);
        xii[i] = InverseGamma::icdf(u[i], 4.5, 2.0);
    }
    auto C = [&](double v) { return std::vector<double>(M, v); };
#define TM(D, NP, X, PARR, ...)                                                              \
    timesc(#D " pdf one-element", [](double v) { return D::pdf(v, __VA_ARGS__); }, X);       \
    timeit<NP>(#D " pdf chunk", D::pdf_chunk, X, PARR);                                      \
    timesc(#D " logpdf one-element", [](double v) { return D::logpdf(v, __VA_ARGS__); }, X); \
    timeit<NP>(#D " logpdf chunk", D::logpdf_chunk, X, PARR);
    TM(Gamma, 2, xg, (std::array<std::vector<double>, 2>{C(2.5), C(3.0)}), p1(2.5), p1(3.0))
    TM(ChiSquared, 1, xc, (std::array<std::vector<double>, 1>{C(4.5)}), p1(4.5))
    TM(InverseGamma, 2, xii, (std::array<std::vector<double>, 2>{C(4.5), C(2.0)}), p1(4.5), p1(2.0))
    TM(Beta, 2, xbb, (std::array<std::vector<double>, 2>{C(2.5), C(0.7)}), p1(2.5), p1(0.7))
    TM(StudentT, 1, xtt, (std::array<std::vector<double>, 1>{C(3.5)}), p1(3.5))
    TM(F, 2, xff, (std::array<std::vector<double>, 2>{C(3.0), C(11.0)}), p1(3.0), p1(11.0))
}
