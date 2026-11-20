// The tests of means and variances (milestone H1, docs/design/htest.md §3.2): the t tests, the
// one-way analyses of variance and the tests of equal variances, as lane kernels of
// stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_parametric(nanobind::module_& m);

}  // namespace sd::htest
