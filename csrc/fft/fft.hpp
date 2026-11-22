// The fft area's native functions (docs/design/fft.md §6): one bind_* per source file.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::fft {

void bind_basic(nanobind::module_& m);
void bind_helper(nanobind::module_& m);
void bind_r2r(nanobind::module_& m);
void bind_fftlog(nanobind::module_& m);

}  // namespace sd::fft
