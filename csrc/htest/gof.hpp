// The tests of goodness of fit to a given distribution (milestone H2, docs/design/htest.md §3.3):
// chisquare and power_divergence, D'Agostino's skewtest, kurtosistest and normaltest,
// jarque_bera, shapiro, anderson, ks_1samp and cramervonmises, as lane kernels of
// stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_gof(nanobind::module_& m);

}  // namespace sd::htest
