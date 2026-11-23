// The smooth area: the submodule stardust._core.smooth (docs/design/smooth.md §6).
#include "bind/smooth_bind.hpp"

#include "smooth/smooth.hpp"

namespace nb = nanobind;

namespace sd {

void bind_smooth(nb::module_& core) {
    nb::module_ m = core.def_submodule("smooth", "stardust smooth native core");
    smooth::bind_local(m);
    smooth::bind_loess(m);
    smooth::bind_savgol(m);
    smooth::bind_spline(m);
    smooth::bind_fitpack(m);
    smooth::bind_kreg(m);
    smooth::bind_kde(m);
    smooth::bind_bandwidth(m);
}

}  // namespace sd
