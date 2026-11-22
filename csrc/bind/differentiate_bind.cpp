// The numerical differentiation area: the submodule stardust._core.differentiate
// (docs/design/integrate.md §6.3).
#include "bind/differentiate_bind.hpp"

#include "differentiate/accurate.hpp"
#include "differentiate/derivative.hpp"

namespace nb = nanobind;

namespace sd {

void bind_differentiate(nb::module_& core) {
    nb::module_ m = core.def_submodule("differentiate", "stardust differentiate native core");
    differentiate::bind_derivative(m);
    differentiate::bind_accurate(m);
}

}  // namespace sd
