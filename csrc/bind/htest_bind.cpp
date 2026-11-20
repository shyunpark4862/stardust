// The hypothesis tests' area: the submodule stardust._core.htest (docs/design/htest.md §6).
#include "bind/htest_bind.hpp"

#include "htest/parametric.hpp"

namespace nb = nanobind;

namespace sd {

void bind_htest(nb::module_& core) {
    nb::module_ m = core.def_submodule("htest", "stardust hypothesis tests native core");
    htest::bind_parametric(m);
}

}  // namespace sd
