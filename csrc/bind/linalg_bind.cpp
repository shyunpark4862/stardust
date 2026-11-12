// The linalg area: the submodule stardust._core.linalg (DESIGN.md §3.2).
#include "bind/linalg_bind.hpp"

#include "linalg/batch.hpp"
#include "linalg/cost.hpp"
#include "linalg/errors.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pack.hpp"
#include "linalg/pyrefs.hpp"
#include "linalg/ops/basic.hpp"
#include "linalg/ops/expm.hpp"
#include "linalg/ops/factor.hpp"
#include "linalg/ops/lu.hpp"
#include "linalg/ops/nonsym.hpp"
#include "linalg/ops/norms.hpp"
#include "linalg/ops/qr.hpp"
#include "linalg/ops/special.hpp"
#include "linalg/ops/spectral.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace sd {

void bind_linalg(nb::module_& core) {
    nb::module_ m = core.def_submodule("linalg", "stardust linalg native core");
    sd::linalg::register_error_translator();
    sd::linalg::bind_setup(m);
    sd::linalg::bind_lu(m);
    sd::linalg::bind_basic(m);
    sd::linalg::bind_norms(m);
    sd::linalg::bind_factor(m);
    sd::linalg::bind_qr(m);
    sd::linalg::bind_spectral(m);
    sd::linalg::bind_nonsym(m);
    sd::linalg::bind_special(m);
    sd::linalg::bind_expm(m);
    // For measurements only: move the split constants (docs/design/linalg.md §7.7).
    m.def("_set_tuning", [](double min_ns, double task_ns) {
        sd::linalg::kParallelMinNs = min_ns;
        sd::linalg::kTaskNs = task_ns;
    });
    // For measurements only: the size from which one matrix is packed on the pool
    // (docs/design/linalg.md §5.6).
    m.def("_set_pack_parallel_min", [](std::int64_t n) { sd::linalg::detail::kPackParallelMin = n; });
    m.def("_set_lu_task_cap", [](bool on) { sd::linalg::kLuTaskCapOn = on; });
}

}  // namespace sd
