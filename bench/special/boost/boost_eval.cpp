// Evaluate or time one Boost.Math function on inputs from a binary file.
//   boost_eval eval NAME in.bin out.bin
//   boost_eval time NAME in.bin            -> prints ns per element (best of 5)
// in.bin: int64 n, int64 nin, then nin arrays of n doubles. Errors -> NaN (ignore_error policy).
#define BOOST_MATH_STANDALONE
#include <boost/math/policies/policy.hpp>
namespace bp = boost::math::policies;
using Pol = bp::policy<
    bp::domain_error<bp::ignore_error>,
    bp::pole_error<bp::ignore_error>,
    bp::overflow_error<bp::ignore_error>,
    bp::underflow_error<bp::ignore_error>,
    bp::evaluation_error<bp::ignore_error>,
    bp::rounding_error<bp::ignore_error>,
    bp::indeterminate_result_error<bp::ignore_error>,
    bp::promote_double<false>>;
#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/special_functions/logit.hpp>
#include <boost/math/special_functions/logistic_sigmoid.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace bm = boost::math;
using F = std::function<double(const double*)>;
static const Pol pol;
static const double NaN = std::numeric_limits<double>::quiet_NaN();

static double ek(double m, bool second) {  // ellipk / ellipe: parameter m = k^2, 0 <= m <= 1 only
    if (!(m >= 0.0 && m <= 1.0)) return NaN;
    return second ? bm::ellint_2(std::sqrt(m), pol) : bm::ellint_1(std::sqrt(m), pol);
}
static int as_int(double v) { return (v == std::floor(v) && std::fabs(v) < 2e9) ? static_cast<int>(v) : INT32_MIN; }

static std::map<std::string, std::pair<int, F>> table() {
    std::map<std::string, std::pair<int, F>> t;
    t["expit"] = {1, [](const double* a) { return bm::logistic_sigmoid(a[0], pol); }};
    t["logit"] = {1, [](const double* a) { return bm::logit(a[0], pol); }};
    t["erf"] = {1, [](const double* a) { return bm::erf(a[0], pol); }};
    t["erfc"] = {1, [](const double* a) { return bm::erfc(a[0], pol); }};
    t["erfinv"] = {1, [](const double* a) { return bm::erf_inv(a[0], pol); }};
    t["erfcinv"] = {1, [](const double* a) { return bm::erfc_inv(a[0], pol); }};
    t["gamma"] = {1, [](const double* a) { return bm::tgamma(a[0], pol); }};
    t["gammaln"] = {1, [](const double* a) { return bm::lgamma(a[0], pol); }};
    t["digamma"] = {1, [](const double* a) { return bm::digamma(a[0], pol); }};
    t["polygamma"] = {2, [](const double* a) {
                          int n = as_int(a[0]);
                          return n < 0 ? NaN : bm::polygamma(n, a[1], pol);
                      }};
    t["beta"] = {2, [](const double* a) { return bm::beta(a[0], a[1], pol); }};
    t["zeta"] = {1, [](const double* a) { return bm::zeta(a[0], pol); }};
    t["gammainc"] = {2, [](const double* a) { return bm::gamma_p(a[0], a[1], pol); }};
    t["gammaincc"] = {2, [](const double* a) { return bm::gamma_q(a[0], a[1], pol); }};
    t["gammaincinv"] = {2, [](const double* a) { return bm::gamma_p_inv(a[0], a[1], pol); }};
    t["gammainccinv"] = {2, [](const double* a) { return bm::gamma_q_inv(a[0], a[1], pol); }};
    t["betainc"] = {3, [](const double* a) { return bm::ibeta(a[0], a[1], a[2], pol); }};
    t["betaincc"] = {3, [](const double* a) { return bm::ibetac(a[0], a[1], a[2], pol); }};
    t["betaincinv"] = {3, [](const double* a) { return bm::ibeta_inv(a[0], a[1], a[2], pol); }};
    t["betainccinv"] = {3, [](const double* a) { return bm::ibetac_inv(a[0], a[1], a[2], pol); }};
    t["expi"] = {1, [](const double* a) { return bm::expint(a[0], pol); }};
    t["expn"] = {2, [](const double* a) {
                     int n = as_int(a[0]);
                     return n < 0 ? NaN : bm::expint(n, a[1], pol);
                 }};
    t["ellipk"] = {1, [](const double* a) { return ek(a[0], false); }};
    t["ellipe"] = {1, [](const double* a) { return ek(a[0], true); }};
    t["j0"] = {1, [](const double* a) { return bm::cyl_bessel_j(0, a[0], pol); }};
    t["j1"] = {1, [](const double* a) { return bm::cyl_bessel_j(1, a[0], pol); }};
    t["y0"] = {1, [](const double* a) { return bm::cyl_neumann(0, a[0], pol); }};
    t["y1"] = {1, [](const double* a) { return bm::cyl_neumann(1, a[0], pol); }};
    t["i0"] = {1, [](const double* a) { return bm::cyl_bessel_i(0, a[0], pol); }};
    t["i1"] = {1, [](const double* a) { return bm::cyl_bessel_i(1, a[0], pol); }};
    t["k0"] = {1, [](const double* a) { return bm::cyl_bessel_k(0, a[0], pol); }};
    t["k1"] = {1, [](const double* a) { return bm::cyl_bessel_k(1, a[0], pol); }};
    t["jv"] = {2, [](const double* a) { return bm::cyl_bessel_j(a[0], a[1], pol); }};
    t["yv"] = {2, [](const double* a) { return bm::cyl_neumann(a[0], a[1], pol); }};
    t["iv"] = {2, [](const double* a) { return bm::cyl_bessel_i(a[0], a[1], pol); }};
    t["kv"] = {2, [](const double* a) { return bm::cyl_bessel_k(a[0], a[1], pol); }};
    t["airy_ai"] = {1, [](const double* a) { return bm::airy_ai(a[0], pol); }};
    t["airy_aip"] = {1, [](const double* a) { return bm::airy_ai_prime(a[0], pol); }};
    t["airy_bi"] = {1, [](const double* a) { return bm::airy_bi(a[0], pol); }};
    t["airy_bip"] = {1, [](const double* a) { return bm::airy_bi_prime(a[0], pol); }};
    t["hyp1f1"] = {3, [](const double* a) { return bm::hypergeometric_1F1(a[0], a[1], a[2], pol); }};
    t["lambertw"] = {1, [](const double* a) { return bm::lambert_w0(a[0], pol); }};
    t["lambertw_m1"] = {1, [](const double* a) { return bm::lambert_wm1(a[0], pol); }};
    return t;
}

static double safe(const F& f, const double* a) {
    try {
        return f(a);
    } catch (...) { return NaN; }
}

int main(int argc, char** argv) {
    auto t = table();
    if (argc >= 2 && std::string(argv[1]) == "list") {
        for (auto& [k, v] : t) std::printf("%s\n", k.c_str());
        return 0;
    }
    const std::string mode = argv[1], name = argv[2];
    FILE* fi = std::fopen(argv[3], "rb");
    std::int64_t n, nin;
    std::fread(&n, 8, 1, fi);
    std::fread(&nin, 8, 1, fi);
    std::vector<std::vector<double>> x(nin, std::vector<double>(n));
    for (auto& v : x) std::fread(v.data(), 8, n, fi);
    std::fclose(fi);
    const F& f = t.at(name).second;
    std::vector<double> out(n);
    auto run = [&] {
        double a[4];
        for (std::int64_t i = 0; i < n; ++i) {
            for (std::int64_t k = 0; k < nin; ++k) a[k] = x[k][i];
            out[i] = safe(f, a);
        }
    };
    if (mode == "eval") {
        run();
        FILE* fo = std::fopen(argv[4], "wb");
        std::fwrite(out.data(), 8, n, fo);
        std::fclose(fo);
        return 0;
    }
    double best = 1e300;
    for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        run();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / n);
    }
    double s = 0;
    for (double v : out) s += v == v ? 0.0 : 1.0;
    std::printf("%.3f\n", best);
    return 0;
}
