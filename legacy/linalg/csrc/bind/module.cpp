// The stardust_linalg._core extension module.
//
// M0: an empty module, so that the build, the Accelerate link and the import
// path are in place. The public functions arrive from M1 on (ROADMAP.md).
#include <nanobind/nanobind.h>

NB_MODULE(_core, m) {
    m.doc() = "stardust-linalg native core";
}
