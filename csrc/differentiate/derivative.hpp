// sd.differentiate's derivative, jacobian and hessian (docs/design/integrate.md §3.8; scipy 1.18.1's
// differentiate/_differentiate.py): the bindings of the submodule stardust._core.differentiate.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::differentiate {

void bind_derivative(nanobind::module_& m);

}  // namespace sd::differentiate
