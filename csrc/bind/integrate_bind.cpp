// The numerical integration area: the submodule stardust._core.integrate
// (docs/design/integrate.md §6.3).
#include "bind/integrate_bind.hpp"

#include "integrate/integrate.hpp"

namespace nb = nanobind;

namespace sd {

void bind_integrate(nb::module_& core) {
    nb::module_ m = core.def_submodule("integrate", "stardust integrate native core");
    integrate::bind_samples(m);
    integrate::bind_gauss(m);
    integrate::bind_quadpack(m);
    integrate::bind_tanhsinh(m);
    integrate::bind_adaptive(m);
}

}  // namespace sd
