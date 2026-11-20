// root's iterations (docs/design/optimize.md §3.6): MINPACK's hybrd and hybrj (hybrd.cpp) on a
// system computed by a Python callback; 'lm' is lsq.cpp's lmder.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::optimize {

void bind_hybr(nanobind::module_& m);

}  // namespace sd::optimize
