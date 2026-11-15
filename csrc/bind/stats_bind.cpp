// The statistics area: the submodule stardust._core.stats (docs/design/stats.md
// §1.4, §3). Every distribution method is an elementwise function
// "<distribution>_<method>" with inputs (params..., x) or (params...), bound
// through the n-ary dispatcher like sd.special (bind/nary_bind.hpp); the
// distribution classes of stardust.stats call them.
#include "bind/stats_bind.hpp"

#include "bind/nary_bind.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/cauchy.hpp"
#include "stats/dist/exponential.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/laplace.hpp"
#include "stats/dist/lognormal.hpp"
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

// Every method of a continuous distribution (T9; sample is T7).
template <class D>
void def_continuous(nb::module_& m) {
    using namespace stats::method;
    def_methods<D, Pdf, Logpdf, Cdf, Logcdf, Ccdf, Logccdf, Icdf, Iccdf, Ilogcdf, Ilogccdf, Mean, Variance,
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
}

}  // namespace sd
