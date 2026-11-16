// The statistics area: the submodule stardust._core.stats (docs/design/stats.md
// §1.4, §3). Every distribution method is an elementwise function
// "<distribution>_<method>" with inputs (params..., x) or (params...), bound
// through the n-ary dispatcher like sd.special (bind/nary_bind.hpp); the
// distribution classes of stardust.stats call them.
#include "bind/stats_bind.hpp"

#include "bind/nary_bind.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/cauchy.hpp"
#include "stats/dist/discrete_basic.hpp"
#include "stats/dist/discrete_more.hpp"
#include "stats/dist/exponential.hpp"
#include "stats/dist/extreme.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/laplace.hpp"
#include "stats/dist/lifetime.hpp"
#include "stats/dist/lognormal.hpp"
#include "stats/dist/noncentral.hpp"
#include "stats/dist/normal.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/dist/uniform.hpp"
#include "stats/dist/weibull.hpp"
#include "stats/engine.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace sd {

namespace {

template <class D, class... M>
void def_methods(nb::module_& m) {
    (nary_bind::def<stats::DistOp<D, M>>(m), ...);
}

// Every method of a continuous distribution (T9; sample, T7, is bound in bind/stats_sample.cpp).
template <class D>
void def_continuous(nb::module_& m) {
    using namespace stats::method;
    def_methods<D, Pdf, Logpdf, Cdf, Logcdf, Ccdf, Logccdf, Icdf, Iccdf, Ilogcdf, Ilogccdf, Mean, Variance,
                StandardDeviation, Skewness, Kurtosis, Median, Mode, Entropy, SupportLo, SupportHi>(m);
}

// Every method of a discrete distribution: pmf and logpmf in place of pdf and logpdf (T25).
template <class D>
void def_discrete(nb::module_& m) {
    using namespace stats::method;
    def_methods<D, Pmf, Logpmf, Cdf, Logcdf, Ccdf, Logccdf, Icdf, Iccdf, Ilogcdf, Ilogccdf, Mean, Variance,
                StandardDeviation, Skewness, Kurtosis, Median, Mode, Entropy, SupportLo, SupportHi>(m);
}

}  // namespace

void bind_stats(nb::module_& core) {
    nb::module_ m = core.def_submodule("stats", "stardust statistics native core");
    // T4: the continuous basic 12 (docs/api/stats.md §2.1)
    def_continuous<stats::Normal>(m);
    def_continuous<stats::StudentT>(m);
    def_continuous<stats::ChiSquared>(m);
    def_continuous<stats::F>(m);
    def_continuous<stats::Gamma>(m);
    def_continuous<stats::Beta>(m);
    def_continuous<stats::Exponential>(m);
    def_continuous<stats::Uniform>(m);
    def_continuous<stats::Laplace>(m);
    def_continuous<stats::Cauchy>(m);
    def_continuous<stats::LogNormal>(m);
    def_continuous<stats::Weibull>(m);
    // T6: extreme value and lifetime 6 (docs/api/stats.md §2.1)
    def_continuous<stats::Logistic>(m);
    def_continuous<stats::Gumbel>(m);
    def_continuous<stats::GEV>(m);
    def_continuous<stats::Pareto>(m);
    def_continuous<stats::Rayleigh>(m);
    def_continuous<stats::InverseGamma>(m);
    // T6: noncentral 3
    def_continuous<stats::NoncentralChiSquared>(m);
    def_continuous<stats::NoncentralT>(m);
    def_continuous<stats::NoncentralF>(m);
    // T5: the discrete 10 (docs/api/stats.md §2.1)
    def_discrete<stats::Binomial>(m);
    def_discrete<stats::Poisson>(m);
    def_discrete<stats::Geometric>(m);
    def_discrete<stats::NegativeBinomial>(m);
    def_discrete<stats::Hypergeometric>(m);
    def_discrete<stats::Bernoulli>(m);
    def_discrete<stats::DiscreteUniform>(m);
    def_discrete<stats::BetaBinomial>(m);
    def_discrete<stats::LogSeries>(m);
    def_discrete<stats::Zipf>(m);
    bind_stats_sample(m);  // T7
}

}  // namespace sd
