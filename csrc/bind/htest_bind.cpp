// The hypothesis tests' area: the submodule stardust._core.htest (docs/design/htest.md §6).
#include "bind/htest_bind.hpp"

#include "htest/contingency.hpp"
#include "htest/correlation.hpp"
#include "htest/gof.hpp"
#include "htest/gof2.hpp"
#include "htest/multicomp.hpp"
#include "htest/parametric.hpp"
#include "htest/proportion.hpp"
#include "htest/rank.hpp"

namespace nb = nanobind;

namespace sd {

void bind_htest(nb::module_& core) {
    nb::module_ m = core.def_submodule("htest", "stardust hypothesis tests native core");
    htest::bind_parametric(m);
    htest::bind_gof(m);
    htest::bind_rank(m);
    htest::bind_gof2(m);
    htest::bind_correlation(m);
    htest::bind_contingency(m);
    htest::bind_proportion(m);
    htest::bind_multicomp(m);
}

}  // namespace sd
