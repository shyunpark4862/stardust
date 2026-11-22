// The fft area: the submodule stardust._core.fft (docs/design/fft.md §6).
#include "bind/fft_bind.hpp"

#include "fft/fft.hpp"

namespace nb = nanobind;

namespace sd {

void bind_fft(nb::module_& core) {
    nb::module_ m = core.def_submodule("fft", "stardust fft native core");
    fft::bind_basic(m);
    fft::bind_helper(m);
    fft::bind_r2r(m);
    fft::bind_fftlog(m);
}

}  // namespace sd
