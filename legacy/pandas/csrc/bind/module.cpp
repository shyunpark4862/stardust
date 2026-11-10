// Python entry point of the native core (`stardust_pandas._core`).

#include "parallel/pool.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

nb::dict build_info() {
    nb::dict info;
    info["compiler"] = std::string(__VERSION__);
    info["cplusplus"] = static_cast<long>(__cplusplus);
#if defined(__ARM_NEON)
    info["neon"] = true;
#else
    info["neon"] = false;
#endif
    return info;
}

// M0 check of the copied pool: task t writes t * t into its own slot, so a
// task that runs twice or never shows up in the result. The result must not
// depend on the thread count.
std::vector<std::int64_t> pool_smoke_test(std::int64_t ntasks) {
    std::vector<std::int64_t> out(static_cast<std::size_t>(ntasks));
    sd::parallel::parallel_for(ntasks, [&](std::int64_t t) { out[static_cast<std::size_t>(t)] = t * t; });
    return out;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "stardust-pandas native core";
    m.def("_build_info", &build_info, "Toolchain facts captured at compile time.");
    m.def("_pool_smoke_test", &pool_smoke_test, nb::arg("ntasks"));
    // Results never depend on the thread count (DESIGN.md §2.6).
    m.def("set_num_threads", [](std::int64_t n) {
        if (n < 1 || n > 1024) throw nb::value_error("the number of threads must be between 1 and 1024");
        sd::parallel::set_num_threads(static_cast<int>(n));
    }, nb::arg("n"));
    m.def("get_num_threads", &sd::parallel::num_threads);
}
