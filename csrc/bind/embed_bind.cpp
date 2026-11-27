// The embed area: the submodule stardust._core.embed (docs/design/embed.md §6).
#include "bind/embed_bind.hpp"

#include "embed/embed.hpp"

namespace nb = nanobind;

namespace sd {

void bind_embed(nb::module_& core) {
    nb::module_ m = core.def_submodule("embed", "stardust embed native core");
    embed::bind_prototype(m);
}

}  // namespace sd
