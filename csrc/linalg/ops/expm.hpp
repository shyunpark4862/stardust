#pragma once

#include <nanobind/nanobind.h>

namespace sd::linalg {

void bind_expm(nanobind::module_& m);

}  // namespace sd::linalg
