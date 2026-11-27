// The embed area's native core: one bind_* per file, called by bind/embed_bind.cpp (docs/design/embed.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::embed {

void bind_prototype(nanobind::module_& m);  // milestone EM0's measurements (prototype.cpp); EM1, EM6, EM7 remove it

}  // namespace sd::embed
