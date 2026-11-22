// The spatial area: the submodule stardust._core.spatial (docs/design/spatial.md §6).
#include "bind/spatial_bind.hpp"

#include "spatial/spatial.hpp"

namespace nb = nanobind;

namespace sd {

void bind_spatial(nb::module_& core) {
    nb::module_ m = core.def_submodule("spatial", "stardust spatial native core");
    spatial::bind_distance(m);
    spatial::bind_kdtree(m);
    spatial::bind_neighbors(m);
    spatial::bind_query(m);
    spatial::bind_dual(m);
    spatial::bind_sphere(m);
}

}  // namespace sd
