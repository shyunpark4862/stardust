// Boost.Math as a comparator of the accuracy tables (bench/accuracy/boost.py builds and loads it).
//
// One entry point over double arrays: sd_boost(name, nargs, args, n, out) evaluates the named
// function element by element and returns 0, or -1 for a name it does not know. Boost's own
// default policy, except that errors return NaN or inf instead of throwing (a comparator must
// answer every point; the measurement leaves NaN out and counts it). Discrete distributions are
// evaluated at integer points only (Boost's binomial, Poisson and negative binomial cdfs are
// defined for real k and are another quantity in between): elsewhere NaN.

#include <cmath>
#include <cstring>
#include <functional>
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
    assert_undefined<false>>;  // Cauchy's moments: NaN

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

using Fn = std::function<double(const double*)>;  // the arguments of one point

bool integer(double k) { return std::isfinite(k) && k == std::floor(k); }

// the methods of a continuous distribution d at x (the last argument)
template <class D>
double cont(const std::string& m, const D& d, double x) {
    if (m == "pdf") return bm::pdf(d, x);
    if (m == "logpdf") return bm::logpdf(d, x);
    if (m == "cdf") return bm::cdf(d, x);
    if (m == "ccdf") return bm::cdf(bm::complement(d, x));
    if (m == "logcdf") return bm::logcdf(d, x);
    if (m == "logccdf") return bm::logcdf(bm::complement(d, x));
    if (m == "icdf") return bm::quantile(d, x);
    if (m == "iccdf") return bm::quantile(bm::complement(d, x));
    return kNaN;
}

template <class D>
double moment(const std::string& m, const D& d) {
    if (m == "mean") return bm::mean(d);
    if (m == "variance") return bm::variance(d);
    if (m == "standard_deviation") return bm::standard_deviation(d);
    if (m == "skewness") return bm::skewness(d);
    if (m == "kurtosis") return bm::kurtosis_excess(d);
    if (m == "median") return bm::median(d);
    if (m == "mode") return bm::mode(d);
    return kNaN;
}

// a discrete distribution: pmf, logpmf, cdf, ccdf, logcdf, logccdf at integer k only
template <class D>
double disc(const std::string& m, const D& d, double k) {
    if (!integer(k)) return kNaN;
    if (m == "pmf") return bm::pdf(d, k);
    if (m == "logpmf") return bm::logpdf(d, k);
    return cont(m, d, k);
}

bool is_moment(const std::string& m) {
    return m == "mean" || m == "variance" || m == "standard_deviation" || m == "skewness" || m == "kurtosis" ||
           m == "median" || m == "mode";
}

// distribution name -> (number of parameters, evaluator(method, params, x))
using DistFn = std::function<double(const std::string&, const double*, bool, double)>;

template <class Make>
DistFn continuous(Make make) {
    return [make](const std::string& m, const double* p, bool mom, double x) {
        auto d = make(p);
        return mom ? moment(m, d) : cont(m, d, x);
    };
}

template <class Make>
DistFn discrete(Make make, double shift = 0) {
    return [make, shift](const std::string& m, const double* p, bool mom, double x) {
        auto d = make(p);
        if (mom) {
            double v = moment(m, d);
            return (m == "mean" || m == "median" || m == "mode") ? v + shift : v;
        }
        return disc(m, d, x - shift);
    };
}

const std::unordered_map<std::string, std::pair<int, DistFn>>& dists() {
    static const std::unordered_map<std::string, std::pair<int, DistFn>> t = {
        {"normal", {2, continuous([](const double* p) { return bm::normal_distribution<double, Pol>(p[0], p[1]); })}},
        {"studentt", {1, continuous([](const double* p) { return bm::students_t_distribution<double, Pol>(p[0]); })}},
        {"chisquared",
         {1, continuous([](const double* p) { return bm::chi_squared_distribution<double, Pol>(p[0]); })}},
        {"f", {2, continuous([](const double* p) { return bm::fisher_f_distribution<double, Pol>(p[0], p[1]); })}},
        {"gamma", {2, continuous([](const double* p) { return bm::gamma_distribution<double, Pol>(p[0], p[1]); })}},
        {"beta", {2, continuous([](const double* p) { return bm::beta_distribution<double, Pol>(p[0], p[1]); })}},
        {"exponential",
         {1, continuous([](const double* p) { return bm::exponential_distribution<double, Pol>(1.0 / p[0]); })}},
        {"lognormal",
         {2, continuous([](const double* p) { return bm::lognormal_distribution<double, Pol>(p[0], p[1]); })}},
        {"uniform", {2, continuous([](const double* p) { return bm::uniform_distribution<double, Pol>(p[0], p[1]); })}},
        {"cauchy", {2, continuous([](const double* p) { return bm::cauchy_distribution<double, Pol>(p[0], p[1]); })}},
        {"laplace", {2, continuous([](const double* p) { return bm::laplace_distribution<double, Pol>(p[0], p[1]); })}},
        {"logistic",
         {2, continuous([](const double* p) { return bm::logistic_distribution<double, Pol>(p[0], p[1]); })}},
        {"weibull", {2, continuous([](const double* p) { return bm::weibull_distribution<double, Pol>(p[0], p[1]); })}},
        {"gumbel",
         {2, continuous([](const double* p) { return bm::extreme_value_distribution<double, Pol>(p[0], p[1]); })}},
        {"pareto", {2, continuous([](const double* p) { return bm::pareto_distribution<double, Pol>(p[1], p[0]); })}},
        {"rayleigh", {1, continuous([](const double* p) { return bm::rayleigh_distribution<double, Pol>(p[0]); })}},
        {"inversegamma",
         {2, continuous([](const double* p) { return bm::inverse_gamma_distribution<double, Pol>(p[0], p[1]); })}},
        {"noncentralchisquared", {2, continuous([](const double* p) {
                                      return bm::non_central_chi_squared_distribution<double, Pol>(p[0], p[1]);
                                  })}},
        {"noncentralt",
         {2, continuous([](const double* p) { return bm::non_central_t_distribution<double, Pol>(p[0], p[1]); })}},
        {"noncentralf", {3, continuous([](const double* p) {
                             return bm::non_central_f_distribution<double, Pol>(p[0], p[1], p[2]);
                         })}},
        {"binomial", {2, discrete([](const double* p) { return bm::binomial_distribution<double, Pol>(p[0], p[1]); })}},
        {"bernoulli", {1, discrete([](const double* p) { return bm::bernoulli_distribution<double, Pol>(p[0]); })}},
        {"poisson", {1, discrete([](const double* p) { return bm::poisson_distribution<double, Pol>(p[0]); })}},
        {"negativebinomial",
         {2, discrete([](const double* p) { return bm::negative_binomial_distribution<double, Pol>(p[0], p[1]); })}},
        // sd's geometric counts trials (support from 1), Boost's failures: k - 1
        {"geometric",
         {1, discrete([](const double* p) { return bm::geometric_distribution<double, Pol>(p[0]); }, 1.0)}},
    };
    return t;
}

// scipy.special's names, arguments in scipy's order
const std::unordered_map<std::string, std::pair<int, Fn>>& specials() {
    static const Pol pol;
    static const bm::normal_distribution<double, Pol> nd;
    static const bm::logistic_distribution<double, Pol> lg;
    auto n_int = [](double v) { return integer(v) && v >= 0 && v < 2147483647.0; };
    static const std::unordered_map<std::string, std::pair<int, Fn>> t = {
        {"gamma", {1, [](const double* a) { return bm::tgamma(a[0], pol); }}},
        {"gammaln", {1, [](const double* a) { return bm::lgamma(a[0], pol); }}},
        {"beta", {2, [](const double* a) { return bm::beta(a[0], a[1], pol); }}},
        {"digamma", {1, [](const double* a) { return bm::digamma(a[0], pol); }}},
        {"polygamma",
         {2, [n_int](const double* a) { return n_int(a[0]) ? bm::polygamma(int(a[0]), a[1], pol) : kNaN; }}},
        {"gammainc", {2, [](const double* a) { return bm::gamma_p(a[0], a[1], pol); }}},
        {"gammaincc", {2, [](const double* a) { return bm::gamma_q(a[0], a[1], pol); }}},
        {"gammaincinv", {2, [](const double* a) { return bm::gamma_p_inv(a[0], a[1], pol); }}},
        {"gammainccinv", {2, [](const double* a) { return bm::gamma_q_inv(a[0], a[1], pol); }}},
        {"betainc", {3, [](const double* a) { return bm::ibeta(a[0], a[1], a[2], pol); }}},
        {"betaincc", {3, [](const double* a) { return bm::ibetac(a[0], a[1], a[2], pol); }}},
        {"betaincinv", {3, [](const double* a) { return bm::ibeta_inv(a[0], a[1], a[2], pol); }}},
        {"betainccinv", {3, [](const double* a) { return bm::ibetac_inv(a[0], a[1], a[2], pol); }}},
        {"erf", {1, [](const double* a) { return bm::erf(a[0], pol); }}},
        {"erfc", {1, [](const double* a) { return bm::erfc(a[0], pol); }}},
        {"erfinv", {1, [](const double* a) { return bm::erf_inv(a[0], pol); }}},
        {"erfcinv", {1, [](const double* a) { return bm::erfc_inv(a[0], pol); }}},
        {"expi", {1, [](const double* a) { return bm::expint(a[0], pol); }}},
        {"expn", {2, [n_int](const double* a) { return n_int(a[0]) ? bm::expint(unsigned(a[0]), a[1], pol) : kNaN; }}},
        {"jv", {2, [](const double* a) { return bm::cyl_bessel_j(a[0], a[1], pol); }}},
        {"yv", {2, [](const double* a) { return bm::cyl_neumann(a[0], a[1], pol); }}},
        {"iv", {2, [](const double* a) { return bm::cyl_bessel_i(a[0], a[1], pol); }}},
        {"kv", {2, [](const double* a) { return bm::cyl_bessel_k(a[0], a[1], pol); }}},
        {"j0", {1, [](const double* a) { return bm::cyl_bessel_j(0.0, a[0], pol); }}},
        {"j1", {1, [](const double* a) { return bm::cyl_bessel_j(1.0, a[0], pol); }}},
        {"y0", {1, [](const double* a) { return bm::cyl_neumann(0.0, a[0], pol); }}},
        {"y1", {1, [](const double* a) { return bm::cyl_neumann(1.0, a[0], pol); }}},
        {"i0", {1, [](const double* a) { return bm::cyl_bessel_i(0.0, a[0], pol); }}},
        {"i1", {1, [](const double* a) { return bm::cyl_bessel_i(1.0, a[0], pol); }}},
        {"k0", {1, [](const double* a) { return bm::cyl_bessel_k(0.0, a[0], pol); }}},
        {"k1", {1, [](const double* a) { return bm::cyl_bessel_k(1.0, a[0], pol); }}},
        {"airy_ai", {1, [](const double* a) { return bm::airy_ai(a[0], pol); }}},
        {"airy_aip", {1, [](const double* a) { return bm::airy_ai_prime(a[0], pol); }}},
        {"airy_bi", {1, [](const double* a) { return bm::airy_bi(a[0], pol); }}},
        {"airy_bip", {1, [](const double* a) { return bm::airy_bi_prime(a[0], pol); }}},
        {"lambertw_m1", {1, [](const double* a) { return bm::lambert_wm1(a[0], pol); }}},
        {"zeta", {1, [](const double* a) { return bm::zeta(a[0], pol); }}},
        {"hyp1f1", {3, [](const double* a) { return bm::hypergeometric_1F1(a[0], a[1], a[2], pol); }}},
        {"ndtr", {1, [](const double* a) { return bm::cdf(nd, a[0]); }}},
        {"log_ndtr", {1, [](const double* a) { return bm::logcdf(nd, a[0]); }}},
        {"ndtri", {1, [](const double* a) { return bm::quantile(nd, a[0]); }}},
        {"expit", {1, [](const double* a) { return bm::cdf(lg, a[0]); }}},
        {"log_expit", {1, [](const double* a) { return bm::logcdf(lg, a[0]); }}},
        {"logit", {1, [](const double* a) { return bm::quantile(lg, a[0]); }}},
        {"stdtr", {2, [](const double* a) { return bm::cdf(bm::students_t_distribution<double, Pol>(a[0]), a[1]); }}},
        {"stdtrit",
         {2, [](const double* a) { return bm::quantile(bm::students_t_distribution<double, Pol>(a[0]), a[1]); }}},
        {"chdtr", {2, [](const double* a) { return bm::cdf(bm::chi_squared_distribution<double, Pol>(a[0]), a[1]); }}},
        {"chdtrc",
         {2,
          [](const double* a) {
              return bm::cdf(bm::complement(bm::chi_squared_distribution<double, Pol>(a[0]), a[1]));
          }}},
        {"chdtri",
         {2,
          [](const double* a) {
              return bm::quantile(bm::complement(bm::chi_squared_distribution<double, Pol>(a[0]), a[1]));
          }}},
        {"fdtr",
         {3, [](const double* a) { return bm::cdf(bm::fisher_f_distribution<double, Pol>(a[0], a[1]), a[2]); }}},
        {"fdtrc",
         {3,
          [](const double* a) {
              return bm::cdf(bm::complement(bm::fisher_f_distribution<double, Pol>(a[0], a[1]), a[2]));
          }}},
        {"fdtri",
         {3, [](const double* a) { return bm::quantile(bm::fisher_f_distribution<double, Pol>(a[0], a[1]), a[2]); }}},
        {"bdtr",
         {3,
          [](const double* a) {
              return integer(a[0]) ? bm::cdf(bm::binomial_distribution<double, Pol>(a[1], a[2]), a[0]) : kNaN;
          }}},
        {"bdtrc",
         {3,
          [](const double* a) {
              return integer(a[0]) ? bm::cdf(bm::complement(bm::binomial_distribution<double, Pol>(a[1], a[2]), a[0]))
                                   : kNaN;
          }}},
        {"nbdtr",
         {3,
          [](const double* a) {
              return integer(a[0]) ? bm::cdf(bm::negative_binomial_distribution<double, Pol>(a[1], a[2]), a[0]) : kNaN;
          }}},
        {"nbdtrc",
         {3,
          [](const double* a) {
              return integer(a[0])
                         ? bm::cdf(bm::complement(bm::negative_binomial_distribution<double, Pol>(a[1], a[2]), a[0]))
                         : kNaN;
          }}},
        {"pdtr",
         {2,
          [](const double* a) {
              return integer(a[0]) ? bm::cdf(bm::poisson_distribution<double, Pol>(a[1]), a[0]) : kNaN;
          }}},
        {"pdtrc",
         {2,
          [](const double* a) {
              return integer(a[0]) ? bm::cdf(bm::complement(bm::poisson_distribution<double, Pol>(a[1]), a[0])) : kNaN;
          }}},
        {"chndtr",
         {3,
          [](const double* a) {
              return bm::cdf(bm::non_central_chi_squared_distribution<double, Pol>(a[0], a[1]), a[2]);
          }}},
        {"chndtrc",
         {3,
          [](const double* a) {
              return bm::cdf(bm::complement(bm::non_central_chi_squared_distribution<double, Pol>(a[0], a[1]), a[2]));
          }}},
        {"chndtrix",
         {3,
          [](const double* a) {
              return bm::quantile(bm::non_central_chi_squared_distribution<double, Pol>(a[0], a[1]), a[2]);
          }}},
        {"ncfdtr",
         {4,
          [](const double* a) {
              return bm::cdf(bm::non_central_f_distribution<double, Pol>(a[0], a[1], a[2]), a[3]);
          }}},
        {"ncfdtrc",
         {4,
          [](const double* a) {
              return bm::cdf(bm::complement(bm::non_central_f_distribution<double, Pol>(a[0], a[1], a[2]), a[3]));
          }}},
        {"ncfdtri",
         {4,
          [](const double* a) {
              return bm::quantile(bm::non_central_f_distribution<double, Pol>(a[0], a[1], a[2]), a[3]);
          }}},
        {"nctdtr",
         {3, [](const double* a) { return bm::cdf(bm::non_central_t_distribution<double, Pol>(a[0], a[1]), a[2]); }}},
        {"nctdtrc",
         {3,
          [](const double* a) {
              return bm::cdf(bm::complement(bm::non_central_t_distribution<double, Pol>(a[0], a[1]), a[2]));
          }}},
        {"nctdtrit", {3, [](const double* a) {
                          return bm::quantile(bm::non_central_t_distribution<double, Pol>(a[0], a[1]), a[2]);
                      }}},
    };
    return t;
}

}  // namespace

extern "C" {

// 0, or -1 for an unknown name, -2 for a wrong argument count. A distribution's method is
// "dist:method" with its parameters first and then x (none for a moment).
int sd_boost(const char* name, int nargs, const double** args, long n, double* out) {
    std::string s(name);
    auto colon = s.find(':');
    if (colon != std::string::npos) {
        auto it = dists().find(s.substr(0, colon));
        if (it == dists().end()) return -1;
        std::string m = s.substr(colon + 1);
        int np = it->second.first;
        bool mom = is_moment(m);
        if (nargs != np + (mom ? 0 : 1)) return -2;
        double p[4];
        for (long i = 0; i < n; ++i) {
            for (int j = 0; j < np; ++j) p[j] = args[j][i];
            try {  // a few internals throw whatever the policy (the non-central chi-squared mode's gamma_p_derivative)
                out[i] = it->second.second(m, p, mom, mom ? 0.0 : args[np][i]);
            } catch (...) { out[i] = kNaN; }
        }
        return 0;
    }
    auto it = specials().find(s);
    if (it == specials().end()) return -1;
    if (nargs != it->second.first) return -2;
    double a[4];
    for (long i = 0; i < n; ++i) {
        for (int j = 0; j < nargs; ++j) a[j] = args[j][i];
        try {
            out[i] = it->second.second(a);
        } catch (...) { out[i] = kNaN; }
    }
    return 0;
}

// the names this build knows, separated by spaces (specials, then distributions as "dist:")
const char* sd_boost_names() {
    static std::string names;
    if (names.empty()) {
        for (auto& kv : specials()) names += kv.first + " ";
        for (auto& kv : dists()) names += kv.first + ": ";
    }
    return names.c_str();
}

}  // extern "C"
