// Boost.Math as a speed comparator (bench/speed/boost.py builds and loads it).
//
// The same functions, arguments and error policy as the accuracy comparator
// (bench/accuracy/boost_comparator.cpp), arranged for timing: the name is resolved once per call
// of the entry point, and the loop over the elements calls the Boost function directly (one
// template instance per function, inlined), so the time is Boost's own plus a loop. A
// distribution with scalar parameters is constructed once per call, as sd's distribution object
// is; with array parameters it is constructed per element, as sd does then.
//
//   sdb_special(name, nargs, args, n, out)                      scipy.special's names
//   sdb_dist(dist, method, params, nparams, x, n, out)          scalar parameters
//   sdb_dist_arrays(dist, method, params, nparams, x, n, out)   parameter arrays of n
//
// Each returns 0, or -1 for an unknown name or method, -2 for a wrong argument count.

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>

#include <boost/math/distributions.hpp>
#include <boost/math/special_functions.hpp>

namespace bm = boost::math;
using namespace boost::math::policies;

using Pol = policy<
    domain_error<ignore_error>,
    pole_error<ignore_error>,
    overflow_error<ignore_error>,
    underflow_error<ignore_error>,
    evaluation_error<ignore_error>,
    rounding_error<ignore_error>,
    indeterminate_result_error<ignore_error>,
    assert_undefined<false>>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const Pol pol;

inline bool integer(double k) { return std::isfinite(k) && k == std::floor(k); }
inline bool n_int(double v) { return integer(v) && v >= 0 && v < 2147483647.0; }

using Loop = void (*)(const double* const* a, long n, double* o);

template <class F>
void loop1(const double* const* a, long n, double* o) {
    const double* x = a[0];
    for (long i = 0; i < n; ++i) {
        try {
            o[i] = F::f(x[i]);
        } catch (...) { o[i] = kNaN; }
    }
}
template <class F>
void loop2(const double* const* a, long n, double* o) {
    const double *x = a[0], *y = a[1];
    for (long i = 0; i < n; ++i) {
        try {
            o[i] = F::f(x[i], y[i]);
        } catch (...) { o[i] = kNaN; }
    }
}
template <class F>
void loop3(const double* const* a, long n, double* o) {
    const double *x = a[0], *y = a[1], *z = a[2];
    for (long i = 0; i < n; ++i) {
        try {
            o[i] = F::f(x[i], y[i], z[i]);
        } catch (...) { o[i] = kNaN; }
    }
}
template <class F>
void loop4(const double* const* a, long n, double* o) {
    const double *x = a[0], *y = a[1], *z = a[2], *w = a[3];
    for (long i = 0; i < n; ++i) {
        try {
            o[i] = F::f(x[i], y[i], z[i], w[i]);
        } catch (...) { o[i] = kNaN; }
    }
}

using NormalD = bm::normal_distribution<double, Pol>;
using LogisticD = bm::logistic_distribution<double, Pol>;
using StudentD = bm::students_t_distribution<double, Pol>;
using Chi2D = bm::chi_squared_distribution<double, Pol>;
using FisherD = bm::fisher_f_distribution<double, Pol>;
using BinomD = bm::binomial_distribution<double, Pol>;
using NBinomD = bm::negative_binomial_distribution<double, Pol>;
using PoissonD = bm::poisson_distribution<double, Pol>;
using NcChi2D = bm::non_central_chi_squared_distribution<double, Pol>;
using NcFD = bm::non_central_f_distribution<double, Pol>;
using NcTD = bm::non_central_t_distribution<double, Pol>;

// One struct per function: f(...) with scipy.special's argument order.
#define SP1(N, E)                               \
    struct N {                                  \
        static double f(double a) { return E; } \
    };
#define SP2(N, E)                                         \
    struct N {                                            \
        static double f(double a, double b) { return E; } \
    };
#define SP3(N, E)                                                   \
    struct N {                                                      \
        static double f(double a, double b, double c) { return E; } \
    };
#define SP4(N, E)                                                             \
    struct N {                                                                \
        static double f(double a, double b, double c, double d) { return E; } \
    };

SP1(Gamma, bm::tgamma(a, pol))
SP1(Gammaln, bm::lgamma(a, pol))
SP2(Beta, bm::beta(a, b, pol))
SP1(Digamma, bm::digamma(a, pol))
SP2(Polygamma, n_int(a) ? bm::polygamma(int(a), b, pol) : kNaN)
SP2(Gammainc, bm::gamma_p(a, b, pol))
SP2(Gammaincc, bm::gamma_q(a, b, pol))
SP2(Gammaincinv, bm::gamma_p_inv(a, b, pol))
SP2(Gammainccinv, bm::gamma_q_inv(a, b, pol))
SP3(Betainc, bm::ibeta(a, b, c, pol))
SP3(Betaincc, bm::ibetac(a, b, c, pol))
SP3(Betaincinv, bm::ibeta_inv(a, b, c, pol))
SP3(Betainccinv, bm::ibetac_inv(a, b, c, pol))
SP1(Erf, bm::erf(a, pol))
SP1(Erfc, bm::erfc(a, pol))
SP1(Erfinv, bm::erf_inv(a, pol))
SP1(Erfcinv, bm::erfc_inv(a, pol))
SP1(Expi, bm::expint(a, pol))
SP2(Expn, n_int(a) ? bm::expint(unsigned(a), b, pol) : kNaN)
SP2(Jv, bm::cyl_bessel_j(a, b, pol))
SP2(Yv, bm::cyl_neumann(a, b, pol))
SP2(Iv, bm::cyl_bessel_i(a, b, pol))
SP2(Kv, bm::cyl_bessel_k(a, b, pol))
SP1(J0, bm::cyl_bessel_j(0.0, a, pol))
SP1(J1, bm::cyl_bessel_j(1.0, a, pol))
SP1(Y0, bm::cyl_neumann(0.0, a, pol))
SP1(Y1, bm::cyl_neumann(1.0, a, pol))
SP1(I0, bm::cyl_bessel_i(0.0, a, pol))
SP1(I1, bm::cyl_bessel_i(1.0, a, pol))
SP1(K0, bm::cyl_bessel_k(0.0, a, pol))
SP1(K1, bm::cyl_bessel_k(1.0, a, pol))
SP1(AiryAi, bm::airy_ai(a, pol))
SP1(AiryAip, bm::airy_ai_prime(a, pol))
SP1(AiryBi, bm::airy_bi(a, pol))
SP1(AiryBip, bm::airy_bi_prime(a, pol))
SP1(LambertWm1, bm::lambert_wm1(a, pol))
SP1(Zeta, bm::zeta(a, pol))
SP3(Hyp1f1, bm::hypergeometric_1F1(a, b, c, pol))
SP1(Ndtr, bm::cdf(NormalD(), a))
SP1(LogNdtr, bm::logcdf(NormalD(), a))
SP1(Ndtri, bm::quantile(NormalD(), a))
SP1(Expit, bm::cdf(LogisticD(), a))
SP1(LogExpit, bm::logcdf(LogisticD(), a))
SP1(Logit, bm::quantile(LogisticD(), a))
SP2(Stdtr, bm::cdf(StudentD(a), b))
SP2(Stdtrit, bm::quantile(StudentD(a), b))
SP2(Chdtr, bm::cdf(Chi2D(a), b))
SP2(Chdtrc, bm::cdf(bm::complement(Chi2D(a), b)))
SP2(Chdtri, bm::quantile(bm::complement(Chi2D(a), b)))
SP3(Fdtr, bm::cdf(FisherD(a, b), c))
SP3(Fdtrc, bm::cdf(bm::complement(FisherD(a, b), c)))
SP3(Fdtri, bm::quantile(FisherD(a, b), c))
SP3(Bdtr, integer(a) ? bm::cdf(BinomD(b, c), a) : kNaN)
SP3(Bdtrc, integer(a) ? bm::cdf(bm::complement(BinomD(b, c), a)) : kNaN)
SP3(Nbdtr, integer(a) ? bm::cdf(NBinomD(b, c), a) : kNaN)
SP3(Nbdtrc, integer(a) ? bm::cdf(bm::complement(NBinomD(b, c), a)) : kNaN)
SP2(Pdtr, integer(a) ? bm::cdf(PoissonD(b), a) : kNaN)
SP2(Pdtrc, integer(a) ? bm::cdf(bm::complement(PoissonD(b), a)) : kNaN)
SP3(Chndtr, bm::cdf(NcChi2D(a, b), c))  // sd's order (df, nc, x), not scipy's (x, df, nc)
SP3(Chndtrc, bm::cdf(bm::complement(NcChi2D(a, b), c)))
SP3(Chndtrix, bm::quantile(NcChi2D(a, b), c))
SP4(Ncfdtr, bm::cdf(NcFD(a, b, c), d))
SP4(Ncfdtrc, bm::cdf(bm::complement(NcFD(a, b, c), d)))
SP4(Ncfdtri, bm::quantile(NcFD(a, b, c), d))
SP3(Nctdtr, bm::cdf(NcTD(a, b), c))
SP3(Nctdtrc, bm::cdf(bm::complement(NcTD(a, b), c)))
SP3(Nctdtrit, bm::quantile(NcTD(a, b), c))

const std::unordered_map<std::string, std::pair<int, Loop>>& specials() {
    static const std::unordered_map<std::string, std::pair<int, Loop>> t = {
        {"gamma", {1, loop1<Gamma>}},
        {"gammaln", {1, loop1<Gammaln>}},
        {"beta", {2, loop2<Beta>}},
        {"digamma", {1, loop1<Digamma>}},
        {"polygamma", {2, loop2<Polygamma>}},
        {"gammainc", {2, loop2<Gammainc>}},
        {"gammaincc", {2, loop2<Gammaincc>}},
        {"gammaincinv", {2, loop2<Gammaincinv>}},
        {"gammainccinv", {2, loop2<Gammainccinv>}},
        {"betainc", {3, loop3<Betainc>}},
        {"betaincc", {3, loop3<Betaincc>}},
        {"betaincinv", {3, loop3<Betaincinv>}},
        {"betainccinv", {3, loop3<Betainccinv>}},
        {"erf", {1, loop1<Erf>}},
        {"erfc", {1, loop1<Erfc>}},
        {"erfinv", {1, loop1<Erfinv>}},
        {"erfcinv", {1, loop1<Erfcinv>}},
        {"expi", {1, loop1<Expi>}},
        {"expn", {2, loop2<Expn>}},
        {"jv", {2, loop2<Jv>}},
        {"yv", {2, loop2<Yv>}},
        {"iv", {2, loop2<Iv>}},
        {"kv", {2, loop2<Kv>}},
        {"j0", {1, loop1<J0>}},
        {"j1", {1, loop1<J1>}},
        {"y0", {1, loop1<Y0>}},
        {"y1", {1, loop1<Y1>}},
        {"i0", {1, loop1<I0>}},
        {"i1", {1, loop1<I1>}},
        {"k0", {1, loop1<K0>}},
        {"k1", {1, loop1<K1>}},
        {"airy_ai", {1, loop1<AiryAi>}},
        {"airy_aip", {1, loop1<AiryAip>}},
        {"airy_bi", {1, loop1<AiryBi>}},
        {"airy_bip", {1, loop1<AiryBip>}},
        {"lambertw_m1", {1, loop1<LambertWm1>}},
        {"zeta", {1, loop1<Zeta>}},
        {"hyp1f1", {3, loop3<Hyp1f1>}},
        {"ndtr", {1, loop1<Ndtr>}},
        {"log_ndtr", {1, loop1<LogNdtr>}},
        {"ndtri", {1, loop1<Ndtri>}},
        {"expit", {1, loop1<Expit>}},
        {"log_expit", {1, loop1<LogExpit>}},
        {"logit", {1, loop1<Logit>}},
        {"stdtr", {2, loop2<Stdtr>}},
        {"stdtrit", {2, loop2<Stdtrit>}},
        {"chdtr", {2, loop2<Chdtr>}},
        {"chdtrc", {2, loop2<Chdtrc>}},
        {"chdtri", {2, loop2<Chdtri>}},
        {"fdtr", {3, loop3<Fdtr>}},
        {"fdtrc", {3, loop3<Fdtrc>}},
        {"fdtri", {3, loop3<Fdtri>}},
        {"bdtr", {3, loop3<Bdtr>}},
        {"bdtrc", {3, loop3<Bdtrc>}},
        {"nbdtr", {3, loop3<Nbdtr>}},
        {"nbdtrc", {3, loop3<Nbdtrc>}},
        {"pdtr", {2, loop2<Pdtr>}},
        {"pdtrc", {2, loop2<Pdtrc>}},
        {"chndtr", {3, loop3<Chndtr>}},
        {"chndtrc", {3, loop3<Chndtrc>}},
        {"chndtrix", {3, loop3<Chndtrix>}},
        {"ncfdtr", {4, loop4<Ncfdtr>}},
        {"ncfdtrc", {4, loop4<Ncfdtrc>}},
        {"ncfdtri", {4, loop4<Ncfdtri>}},
        {"nctdtr", {3, loop3<Nctdtr>}},
        {"nctdtrc", {3, loop3<Nctdtrc>}},
        {"nctdtrit", {3, loop3<Nctdtrit>}},
    };
    return t;
}

// ---------------------------------------------------------------- distributions

enum Method { kPdf, kLogpdf, kCdf, kCcdf, kLogcdf, kLogccdf, kIcdf, kIccdf, kBad };

Method method_of(const std::string& m, bool discrete) {
    if (m == (discrete ? "pmf" : "pdf")) return kPdf;
    if (m == (discrete ? "logpmf" : "logpdf")) return kLogpdf;
    if (m == "cdf") return kCdf;
    if (m == "ccdf") return kCcdf;
    if (m == "logcdf") return kLogcdf;
    if (m == "logccdf") return kLogccdf;
    if (!discrete && m == "icdf") return kIcdf;  // Boost's discrete quantiles round otherwise
    if (!discrete && m == "iccdf") return kIccdf;
    return kBad;
}

template <Method M, class D>
inline double eval(const D& d, double x) {
    if constexpr (M == kPdf) return bm::pdf(d, x);
    if constexpr (M == kLogpdf) return bm::logpdf(d, x);
    if constexpr (M == kCdf) return bm::cdf(d, x);
    if constexpr (M == kCcdf) return bm::cdf(bm::complement(d, x));
    if constexpr (M == kLogcdf) return bm::logcdf(d, x);
    if constexpr (M == kLogccdf) return bm::logcdf(bm::complement(d, x));
    if constexpr (M == kIcdf) return bm::quantile(d, x);
    if constexpr (M == kIccdf) return bm::quantile(bm::complement(d, x));
    return kNaN;
}

// a distribution: its parameter count, whether discrete (evaluated at integer points only,
// shifted by `shift`: sd's geometric counts trials, Boost's failures), and make(params)
struct Dist {
    int np;
    bool discrete;
    double shift;
    // loop over x with scalar parameters p, or with parameter arrays pa (n each)
    void (*scalar)(Method, const double* p, double shift, bool discrete, const double* x, long n, double* o);
    void (*arrays)(Method, const double* const* pa, double shift, bool discrete, const double* x, long n, double* o);
};

template <Method M, class D>
void run_scalar(const D& d, double shift, bool discrete, const double* x, long n, double* o) {
    for (long i = 0; i < n; ++i) {
        double v = x[i] - shift;
        try {
            o[i] = (discrete && M != kIcdf && M != kIccdf && !integer(v)) ? kNaN : eval<M>(d, v);
        } catch (...) { o[i] = kNaN; }
    }
}

template <class Make>
struct Ops {
    static void scalar(Method m, const double* p, double shift, bool disc, const double* x, long n, double* o) {
        auto d = Make::make(p);
        switch (m) {
            case kPdf: return run_scalar<kPdf>(d, shift, disc, x, n, o);
            case kLogpdf: return run_scalar<kLogpdf>(d, shift, disc, x, n, o);
            case kCdf: return run_scalar<kCdf>(d, shift, disc, x, n, o);
            case kCcdf: return run_scalar<kCcdf>(d, shift, disc, x, n, o);
            case kLogcdf: return run_scalar<kLogcdf>(d, shift, disc, x, n, o);
            case kLogccdf: return run_scalar<kLogccdf>(d, shift, disc, x, n, o);
            case kIcdf: return run_scalar<kIcdf>(d, shift, disc, x, n, o);
            case kIccdf: return run_scalar<kIccdf>(d, shift, disc, x, n, o);
            default: return;
        }
    }
    template <Method M>
    static void arr(const double* const* pa, int np, double shift, bool disc, const double* x, long n, double* o) {
        double p[4];
        for (long i = 0; i < n; ++i) {
            for (int j = 0; j < np; ++j) p[j] = pa[j][i];
            double v = x[i] - shift;
            try {
                o[i] = (disc && M != kIcdf && M != kIccdf && !integer(v)) ? kNaN : eval<M>(Make::make(p), v);
            } catch (...) { o[i] = kNaN; }
        }
    }
    static void arrays(Method m, const double* const* pa, double shift, bool disc, const double* x, long n, double* o) {
        const int np = Make::np;
        switch (m) {
            case kPdf: return arr<kPdf>(pa, np, shift, disc, x, n, o);
            case kLogpdf: return arr<kLogpdf>(pa, np, shift, disc, x, n, o);
            case kCdf: return arr<kCdf>(pa, np, shift, disc, x, n, o);
            case kCcdf: return arr<kCcdf>(pa, np, shift, disc, x, n, o);
            case kLogcdf: return arr<kLogcdf>(pa, np, shift, disc, x, n, o);
            case kLogccdf: return arr<kLogccdf>(pa, np, shift, disc, x, n, o);
            case kIcdf: return arr<kIcdf>(pa, np, shift, disc, x, n, o);
            case kIccdf: return arr<kIccdf>(pa, np, shift, disc, x, n, o);
            default: return;
        }
    }
};

#define DIST(N, NP, E)                                  \
    struct N {                                          \
        static constexpr int np = NP;                   \
        static auto make(const double* p) { return E; } \
    };

// sd's parameters (tests/stats/refdata.CLASSES order), as the accuracy comparator maps them
DIST(MkNormal, 2, (bm::normal_distribution<double, Pol>(p[0], p[1])))
DIST(MkStudentT, 1, (bm::students_t_distribution<double, Pol>(p[0])))
DIST(MkChiSquared, 1, (bm::chi_squared_distribution<double, Pol>(p[0])))
DIST(MkF, 2, (bm::fisher_f_distribution<double, Pol>(p[0], p[1])))
DIST(MkGamma, 2, (bm::gamma_distribution<double, Pol>(p[0], p[1])))
DIST(MkBeta, 2, (bm::beta_distribution<double, Pol>(p[0], p[1])))
DIST(MkExponential, 1, (bm::exponential_distribution<double, Pol>(1.0 / p[0])))
DIST(MkLognormal, 2, (bm::lognormal_distribution<double, Pol>(p[0], p[1])))
DIST(MkUniform, 2, (bm::uniform_distribution<double, Pol>(p[0], p[1])))
DIST(MkCauchy, 2, (bm::cauchy_distribution<double, Pol>(p[0], p[1])))
DIST(MkLaplace, 2, (bm::laplace_distribution<double, Pol>(p[0], p[1])))
DIST(MkLogistic, 2, (bm::logistic_distribution<double, Pol>(p[0], p[1])))
DIST(MkWeibull, 2, (bm::weibull_distribution<double, Pol>(p[0], p[1])))
DIST(MkGumbel, 2, (bm::extreme_value_distribution<double, Pol>(p[0], p[1])))
DIST(MkPareto, 2, (bm::pareto_distribution<double, Pol>(p[1], p[0])))
DIST(MkRayleigh, 1, (bm::rayleigh_distribution<double, Pol>(p[0])))
DIST(MkInverseGamma, 2, (bm::inverse_gamma_distribution<double, Pol>(p[0], p[1])))
DIST(MkNcChi2, 2, (bm::non_central_chi_squared_distribution<double, Pol>(p[0], p[1])))
DIST(MkNcT, 2, (bm::non_central_t_distribution<double, Pol>(p[0], p[1])))
DIST(MkNcF, 3, (bm::non_central_f_distribution<double, Pol>(p[0], p[1], p[2])))
DIST(MkBinomial, 2, (bm::binomial_distribution<double, Pol>(p[0], p[1])))
DIST(MkBernoulli, 1, (bm::bernoulli_distribution<double, Pol>(p[0])))
DIST(MkPoisson, 1, (bm::poisson_distribution<double, Pol>(p[0])))
DIST(MkNegBinomial, 2, (bm::negative_binomial_distribution<double, Pol>(p[0], p[1])))
DIST(MkGeometric, 1, (bm::geometric_distribution<double, Pol>(p[0])))

template <class Make>
Dist dist(bool discrete = false, double shift = 0.0) {
    return {Make::np, discrete, shift, &Ops<Make>::scalar, &Ops<Make>::arrays};
}

const std::unordered_map<std::string, Dist>& dists() {
    static const std::unordered_map<std::string, Dist> t = {
        {"normal", dist<MkNormal>()},
        {"studentt", dist<MkStudentT>()},
        {"chisquared", dist<MkChiSquared>()},
        {"f", dist<MkF>()},
        {"gamma", dist<MkGamma>()},
        {"beta", dist<MkBeta>()},
        {"exponential", dist<MkExponential>()},
        {"lognormal", dist<MkLognormal>()},
        {"uniform", dist<MkUniform>()},
        {"cauchy", dist<MkCauchy>()},
        {"laplace", dist<MkLaplace>()},
        {"logistic", dist<MkLogistic>()},
        {"weibull", dist<MkWeibull>()},
        {"gumbel", dist<MkGumbel>()},
        {"pareto", dist<MkPareto>()},
        {"rayleigh", dist<MkRayleigh>()},
        {"inversegamma", dist<MkInverseGamma>()},
        {"noncentralchisquared", dist<MkNcChi2>()},
        {"noncentralt", dist<MkNcT>()},
        {"noncentralf", dist<MkNcF>()},
        {"binomial", dist<MkBinomial>(true)},
        {"bernoulli", dist<MkBernoulli>(true)},
        {"poisson", dist<MkPoisson>(true)},
        {"negativebinomial", dist<MkNegBinomial>(true)},
        {"geometric", dist<MkGeometric>(true, 1.0)},
    };
    return t;
}

}  // namespace

extern "C" {

int sdb_special(const char* name, int nargs, const double* const* args, long n, double* out) {
    auto it = specials().find(name);
    if (it == specials().end()) return -1;
    if (nargs != it->second.first) return -2;
    it->second.second(args, n, out);
    return 0;
}

int sdb_dist(
    const char* name, const char* method, const double* params, int nparams, const double* x, long n, double* out
) {
    auto it = dists().find(name);
    if (it == dists().end()) return -1;
    const Dist& d = it->second;
    if (nparams != d.np) return -2;
    Method m = method_of(method, d.discrete);
    if (m == kBad) return -1;
    d.scalar(m, params, d.shift, d.discrete, x, n, out);
    return 0;
}

int sdb_dist_arrays(
    const char* name, const char* method, const double* const* params, int nparams, const double* x, long n, double* out
) {
    auto it = dists().find(name);
    if (it == dists().end()) return -1;
    const Dist& d = it->second;
    if (nparams != d.np) return -2;
    Method m = method_of(method, d.discrete);
    if (m == kBad) return -1;
    d.arrays(m, params, d.shift, d.discrete, x, n, out);
    return 0;
}

// the names this build knows: specials, then distributions as "dist:"
const char* sdb_names() {
    static std::string names;
    if (names.empty()) {
        for (auto& kv : specials()) names += kv.first + " ";
        for (auto& kv : dists()) names += kv.first + ": ";
    }
    return names.c_str();
}

}  // extern "C"
