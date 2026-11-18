// The regression area: the submodule stardust._core.regression (docs/design/regression.md §5).
#include "bind/regression_bind.hpp"

#include "regression/glm.hpp"
#include "regression/linear.hpp"

namespace nb = nanobind;

namespace sd {

void bind_regression(nb::module_& core) {
    nb::module_ m = core.def_submodule("regression", "stardust regression native core");
    regression::bind_linear(m);
    regression::bind_glm(m);
}

}  // namespace sd
