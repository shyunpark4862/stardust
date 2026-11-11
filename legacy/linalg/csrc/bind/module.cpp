// The stardust_linalg._core extension module.
#include "core/batch.hpp"
#include "core/errors.hpp"
#include "core/outblock.hpp"
#include "core/pyrefs.hpp"
#include "ops/basic.hpp"
#include "ops/expm.hpp"
#include "ops/factor.hpp"
#include "ops/lu.hpp"
#include "ops/nonsym.hpp"
#include "ops/norms.hpp"
#include "ops/qr.hpp"
#include "ops/special.hpp"
#include "ops/spectral.hpp"

#include <nanobind/nanobind.h>

NB_MODULE(_core, m) {
    m.doc() = "stardust-linalg native core";
    sdla::load_sdnp_refs();
    sdla::register_error_translator();
    sdla::bind_setup(m);
    sdla::bind_outblock(m);
    sdla::bind_lu(m);
    sdla::bind_basic(m);
    sdla::bind_norms(m);
    sdla::bind_factor(m);
    sdla::bind_qr(m);
    sdla::bind_spectral(m);
    sdla::bind_nonsym(m);
    sdla::bind_special(m);
    sdla::bind_expm(m);
    // For experiments/m9_threshold only: move the split constants (M9).
    m.def("_set_tuning", [](double min_ns, double task_ns) {
        sdla::kParallelMinNs = min_ns;
        sdla::kTaskNs = task_ns;
    });
}
