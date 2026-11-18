// Quantile regression's numerical core (docs/design/regression.md §3.6): the Frisch-Newton
// interior point method, the crossover to an optimal basis, simplex pivots to a certified
// optimal vertex, the vertex rule of non-unique solutions and the covariances of R's
// summary.rq ('iid', 'nid', 'ker').
#pragma once

#include <nanobind/nanobind.h>

namespace sd::regression {

void bind_quantile(nanobind::module_& m);

}  // namespace sd::regression
