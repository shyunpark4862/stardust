// The optimize area: the submodule stardust._core.optimize (docs/design/optimize.md §6).
#include "bind/optimize_bind.hpp"

#include "optimize/lsq.hpp"
#include "optimize/scalar.hpp"

namespace nb = nanobind;

namespace sd {

void bind_optimize(nb::module_& core) {
    nb::module_ m = core.def_submodule("optimize", "stardust optimize native core");
    optimize::bind_lsq(m);
    optimize::bind_scalar(m);
    optimize::bind_batch(m);
}

}  // namespace sd
