// special's ndtri against the Newton-refined double-double quantile (normal_detail::quantile_z),
// relative error of z in kUlp, by region of p (|z| >= 1/4: nearer 0 the reference's own
// absolute 1e-16 dominates). Build: bench/stats/opt/a/build.sh ndtri_error
#include "stats/dist/normal.hpp"
#include <random>
#include <cstdio>
using namespace sd::stats;
int main() {
    std::mt19937_64 g(9);
    std::uniform_real_distribution<double> U(0, 1);
    double mx[4] = {0, 0, 0, 0}; // z>=-0.75 & p<=.5 ; p>.5 ; -1<=z<-0.75 ; z<-1
    for (int i = 0; i < 4000000; ++i) {
        double u = U(g);
        double p = i % 4 == 0   ? u
                   : i % 4 == 1 ? std::pow(10.0, -300 * u * u)
                   : i % 4 == 2 ? 0.2 + 0.6 * u
                                : 1 - std::pow(10.0, -16 * u);
        if (!(p > 0 && p < 1)) continue;
        double z = normal_detail::ndtri(p);
        auto r = normal_detail::quantile_z(p, false);
        if (std::fabs(r.hi) < 0.25) continue;
        double e = std::fabs((z - r.hi) - r.lo) / (std::fabs(r.hi) * 0x1p-53);
        int k = p > 0.5 ? 1 : z >= -0.75 ? 0 : z >= -1 ? 2 : 3;
        if (e > mx[k]) mx[k] = e;
    }
    printf(
        "ndtri rel err (kUlp): lower z>=-0.75: %.2f; p>1/2: %.2f; -1<=z<-0.75: %.2f; z<-1: %.2f\n", mx[0], mx[1], mx[2],
        mx[3]
    );
}
