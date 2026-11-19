// minimize's iterations (docs/design/optimize.md §3.5): Nelder-Mead (nelder_mead.cpp), BFGS
// (bfgs.cpp), L-BFGS-B (lbfgsb.cpp), Newton-CG (newton_cg.cpp), trust-ncg and trust-exact
// (trust.cpp), each the same steps as scipy 1.18.1 (O9), on a function computed by a Python
// callback.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::optimize {

void bind_nelder_mead(nanobind::module_& m);
void bind_bfgs(nanobind::module_& m);
void bind_lbfgsb(nanobind::module_& m);
void bind_newton_cg(nanobind::module_& m);
void bind_trust(nanobind::module_& m);

}  // namespace sd::optimize
