// The interpolate area: the submodule stardust._core.interpolate (docs/design/interpolate.md §6). sd.interp
// is the array area's function and is bound with the array (array/ops/interp.cpp).
#include "bind/interpolate_bind.hpp"

#include "interpolate/interpolate.hpp"

namespace nb = nanobind;

namespace sd {

void bind_interpolate(nb::module_& core) {
    nb::module_ m = core.def_submodule("interpolate", "stardust interpolate native core");
    interpolate::bind_linear(m);
    interpolate::bind_cubic(m);
    interpolate::bind_ppoly(m);
    interpolate::bind_bspline(m);
    interpolate::bind_polyint(m);
    interpolate::bind_aaa(m);
    interpolate::bind_grid(m);
    interpolate::bind_rbf(m);
    interpolate::bind_scattered(m);
}

}  // namespace sd
