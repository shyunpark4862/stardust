// Python entry point of the native core (`stardust._core`).

#include "bind/array_bind.hpp"
#include "bind/linalg_bind.hpp"
#include "bind/pyutil.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/creation.hpp"
#include "array/ops/matmul.hpp"
#include "array/ops/reduction.hpp"
#include "array/ops/shape.hpp"
#include "array/ops/sort.hpp"
#include "array/ops/ufunc.hpp"
#include "parallel/pool.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <Accelerate/Accelerate.h>

namespace nb = nanobind;

namespace {

// Calls into Accelerate once so a broken link or a wrong ILP64 setup fails
// loudly at import-test time rather than at the first BLAS call
// (docs/design/array.md §1.3).
double accelerate_smoke_test() {
    const double x[] = {1.0, 2.0, 3.0};
    const double y[] = {4.0, 5.0, 6.0};
    return cblas_ddot(3, x, 1, y, 1);  // 1*4 + 2*5 + 3*6 = 32
}

nb::dict build_info() {
    nb::dict info;
    info["compiler"] = std::string(__VERSION__);
    info["cplusplus"] = static_cast<long>(__cplusplus);
#if defined(__ARM_NEON)
    info["neon"] = true;
#else
    info["neon"] = false;
#endif
#if defined(__ARM_FEATURE_COMPLEX)
    info["fcma"] = true;
#else
    info["fcma"] = false;
#endif
    info["blas_ilp64"] = sizeof(__LAPACK_int) == 8;
    info["accelerate_ddot"] = accelerate_smoke_test();
    return info;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "stardust native core";
    // nanobind reports objects still alive when its module is torn down. A
    // user's global dict holding sd.float64 at interpreter exit (finalization
    // order is not ours to control) triggers it without any real leak, so the
    // check is off; tests/test_m1_ndarray.py measures refcounts directly instead.
    nb::set_leak_warnings(false);
    sd::register_dtypes(m);  // first: other bindings use the dtype objects
    sd::bind_ndarray(m);
    sd::bind_creation(m);
    sd::bind_copy(m);
    sd::bind_shape(m);
    sd::bind_ufuncs(m);
    sd::bind_reductions(m);
    sd::bind_sort(m);
    sd::bind_matmul(m);
    m.def("_build_info", &build_info, "Toolchain facts captured at compile time.");
    // docs/api/array.md §0.3. Results never depend on the thread count
    // (docs/design/array.md §10.5).
    m.def("set_num_threads", [](nb::handle n) {
        const std::int64_t v = sd::parse_int(n, "n");
        if (v < 1 || v > 1024) throw nb::value_error("the number of threads must be between 1 and 1024");
        sd::parallel::set_num_threads(static_cast<int>(v));
    }, nb::arg());
    m.def("get_num_threads", &sd::parallel::num_threads);
    sd::bind_linalg(m);  // last: it looks up the array objects bound above
}
