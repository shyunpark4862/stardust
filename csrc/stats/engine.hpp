// The distribution engine (docs/design/stats.md §3): one elementwise Op per
// (distribution, method), made from a distribution struct by a template, and
// bound through the same n-ary dispatcher as sd.special (bind/nary_bind.hpp).
//
// A distribution is a struct of static functions:
//
//     struct Normal {
//         static constexpr char name[] = "normal";
//         static constexpr std::array<const char*, 2> params{"mu", "sigma"};
//         static bool valid(double mu, double sigma);       // the parameter domain
//         static double cdf(double x, double mu, double sigma);
//         ...                                               // one per method below
//     };
//
// DistOp<D, M> is the Op of method M: inputs (params..., x) for the methods that
// take an input, (params...) for the others; the binding names it
// "<distribution>_<method>" in stardust._core.stats. It returns NaN where the
// parameters are outside their domain (T8) or the input is NaN, so a method is
// only ever called with valid parameters and a non-NaN input.
#pragma once

#include "special/common.hpp"
#include "special/cost.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace sd::stats {

using special::kInf;
using special::kNaN;
using special::Real;

// "<a>_<b>" at compile time.
template <std::size_t A, std::size_t B>
constexpr std::array<char, A + B> join_name(const char (&a)[A], const char (&b)[B]) {
    std::array<char, A + B> r{};  // (A - 1) + 1 + (B - 1) + 1 characters
    std::size_t k = 0;
    for (std::size_t i = 0; i + 1 < A; ++i) r[k++] = a[i];
    r[k++] = '_';
    for (std::size_t i = 0; i < B; ++i) r[k++] = b[i];
    return r;
}

// ---- methods ------------------------------------------------------------------
//
// Each names itself, says whether it takes an input (and the input's name in
// messages), and calls the distribution's function, with a default where the
// method follows from others.

namespace method {

#define SD_STATS_INPUT_METHOD(Tag, fname, argname)                                \
    struct Tag {                                                                  \
        static constexpr char name[] = #fname;                                    \
        static constexpr bool input = true;                                       \
        static constexpr const char* arg = argname;                               \
        template <class D, class... P>                                            \
        static double call(double x, P... p) { return D::fname(x, p...); }        \
    };

SD_STATS_INPUT_METHOD(Pdf, pdf, "x")
SD_STATS_INPUT_METHOD(Logpdf, logpdf, "x")
SD_STATS_INPUT_METHOD(Pmf, pmf, "k")  // discrete distributions (T9, T25)
SD_STATS_INPUT_METHOD(Logpmf, logpmf, "k")
SD_STATS_INPUT_METHOD(Cdf, cdf, "x")
SD_STATS_INPUT_METHOD(Logcdf, logcdf, "x")
SD_STATS_INPUT_METHOD(Ccdf, ccdf, "x")
SD_STATS_INPUT_METHOD(Logccdf, logccdf, "x")
SD_STATS_INPUT_METHOD(Icdf, icdf, "p")
SD_STATS_INPUT_METHOD(Iccdf, iccdf, "p")
SD_STATS_INPUT_METHOD(Ilogcdf, ilogcdf, "logp")
SD_STATS_INPUT_METHOD(Ilogccdf, ilogccdf, "logp")
#undef SD_STATS_INPUT_METHOD

#define SD_STATS_PARAM_METHOD(Tag, fname)                                         \
    struct Tag {                                                                  \
        static constexpr char name[] = #fname;                                    \
        static constexpr bool input = false;                                      \
        template <class D, class... P>                                            \
        static double call(P... p) { return D::fname(p...); }                     \
    };

SD_STATS_PARAM_METHOD(Mean, mean)
SD_STATS_PARAM_METHOD(Variance, variance)
SD_STATS_PARAM_METHOD(Skewness, skewness)
SD_STATS_PARAM_METHOD(Kurtosis, kurtosis)
SD_STATS_PARAM_METHOD(Mode, mode)
SD_STATS_PARAM_METHOD(Entropy, entropy)
SD_STATS_PARAM_METHOD(SupportLo, support_lo)
SD_STATS_PARAM_METHOD(SupportHi, support_hi)
#undef SD_STATS_PARAM_METHOD

// The standard deviation from the variance (sqrt halves a relative error).
struct StandardDeviation {
    static constexpr char name[] = "standard_deviation";
    static constexpr bool input = false;
    template <class D, class... P>
    static double call(P... p) {
        if constexpr (requires { D::standard_deviation(p...); }) return D::standard_deviation(p...);
        else return std::sqrt(D::variance(p...));
    }
};

// The median: icdf(1/2) unless the distribution has a closed form.
struct Median {
    static constexpr char name[] = "median";
    static constexpr bool input = false;
    template <class D, class... P>
    static double call(P... p) {
        if constexpr (requires { D::median(p...); }) return D::median(p...);
        else return D::icdf(0.5, p...);
    }
};

}  // namespace method

// ---- the Op -------------------------------------------------------------------

// Per-element cost in ns for the parallel split (special/cost.hpp): the
// distribution's own estimate when it gives one, else a middling default.
template <class D, class M>
constexpr double cost_of() {
    if constexpr (requires { D::template cost<M>(); }) return D::template cost<M>();
    else return 20.0;
}

template <class D, class M, class = std::make_index_sequence<D::params.size()>>
struct DistOp;

template <class D, class M, std::size_t... I>
struct DistOp<D, M, std::index_sequence<I...>> : Real {
    template <std::size_t> using P = double;
    static constexpr auto name_buf = join_name(D::name, M::name);
    static constexpr const char* name = name_buf.data();
    static constexpr int nin = static_cast<int>(M::input) + static_cast<int>(sizeof...(I));
    static constexpr int nout = 1;
    static constexpr std::array<const char*, nin> args = [] {
        std::array<const char*, nin> a{};
        std::size_t k = 0;
        for (const char* s : D::params) a[k++] = s;
        if constexpr (M::input) a[k] = M::arg;
        return a;
    }();
    static constexpr double cost_ns = cost_of<D, M>();

    static double f(P<I>... p, double x)
        requires M::input
    {
        if (!D::valid(p...) || std::isnan(x)) return kNaN;
        return M::template call<D>(x, p...);
    }
    static double f(P<I>... p)
        requires(!M::input)
    {
        if (!D::valid(p...)) return kNaN;
        return M::template call<D>(p...);
    }
};

}  // namespace sd::stats
