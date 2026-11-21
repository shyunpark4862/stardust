// The contingency tables (milestone H6, docs/design/htest.md §3.7): chi2_contingency,
// fisher_exact (2 x 2 and r x c), barnard_exact, boschloo_exact, mcnemar, cochrans_q and the
// estimates odds_ratio and relative_risk's companion, as lane kernels of stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_contingency(nanobind::module_& m);

}  // namespace sd::htest
