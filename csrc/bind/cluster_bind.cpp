// The cluster area: the submodule stardust._core.cluster (docs/design/cluster.md §6).
#include "bind/cluster_bind.hpp"

#include "cluster/cluster.hpp"

namespace nb = nanobind;

namespace sd {

void bind_cluster(nb::module_& core) {
    nb::module_ m = core.def_submodule("cluster", "stardust cluster native core");
    cluster::bind_hierarchy(m);
    cluster::bind_prototype(m);
}

}  // namespace sd
