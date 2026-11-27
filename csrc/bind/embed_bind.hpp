// The embed area: the submodule stardust._core.embed (docs/design/embed.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_embed(nanobind::module_& core);

}  // namespace sd
