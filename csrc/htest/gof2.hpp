// The tests of goodness of fit between samples (milestone H4, docs/design/htest.md §3.5): ks_2samp,
// cramervonmises_2samp, anderson_ksamp and epps_singleton_2samp, as lane kernels of
// stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_gof2(nanobind::module_& m);

}  // namespace sd::htest
