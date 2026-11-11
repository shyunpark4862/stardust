// The stardust_linalg._core extension module.
#include "core/errors.hpp"
#include "core/outblock.hpp"
#include "core/pyrefs.hpp"
#include "ops/basic.hpp"
#include "ops/factor.hpp"
#include "ops/lu.hpp"
#include "ops/norms.hpp"
#include "ops/qr.hpp"

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
}
