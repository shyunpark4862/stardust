// minimize's iterations (docs/design/optimize.md §3.5): Nelder-Mead (nelder_mead.cpp), BFGS
// (bfgs.cpp) and L-BFGS-B (lbfgsb.cpp), each the same steps as scipy 1.18.1 (O9), on a function
// computed by a Python callback.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::optimize {

void bind_nelder_mead(nanobind::module_& m);
void bind_bfgs(nanobind::module_& m);
void bind_lbfgsb(nanobind::module_& m);

}  // namespace sd::optimize
