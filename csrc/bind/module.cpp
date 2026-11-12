// Python entry point of the native core (`stardust._core`).

#include "bind/array_bind.hpp"
#include "bind/errors.hpp"
#include "bind/frame_bind.hpp"
#include "bind/linalg_bind.hpp"
#include "bind/special_bind.hpp"
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
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

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

// A check of the pool itself: task t writes t * t into its own slot, so a
// task that runs twice or never shows up in the result. The result must not
// depend on the thread count.
std::vector<std::int64_t> pool_smoke_test(std::int64_t ntasks) {
    std::vector<std::int64_t> out(static_cast<std::size_t>(ntasks));
    sd::parallel::parallel_for(ntasks, [&](std::int64_t t) { out[static_cast<std::size_t>(t)] = t * t; });
    return out;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "stardust native core";
    // nanobind reports objects still alive when its module is torn down. A
    // user's global dict holding sd.float64 at interpreter exit (finalization
    // order is not ours to control) triggers it without any real leak, so the
    // check is off; tests/array/test_ndarray.py measures refcounts directly instead.
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
    m.def("_pool_smoke_test", &pool_smoke_test, nb::arg("ntasks"));
    // API.md §3. Results never depend on the thread count
    // (docs/design/array.md §10.5).
    m.def("set_num_threads", [](nb::handle n) {
        sd::call_named("set_num_threads", [&] {
            constexpr const char* kRange = "'n' must be between 1 and 1024";
            std::int64_t v = 0;
            try {
                v = sd::parse_int(n, "n");
            } catch (const std::overflow_error&) {  // out of 1..1024 like any other int: ValueError (API.md §3)
                throw nb::value_error(kRange);
            }
            if (v < 1 || v > 1024) throw nb::value_error(kRange);
            sd::parallel::set_num_threads(static_cast<int>(v));
        });
    }, nb::arg().none());
    m.def("get_num_threads", &sd::parallel::num_threads);
    sd::bind_linalg(m);  // after the array: it looks up the array objects bound above
    sd::bind_frame(m);
    sd::bind_special(m);  // after the array, like linalg
}
