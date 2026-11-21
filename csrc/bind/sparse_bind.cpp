// The sparse area: the submodule stardust._core.sparse (docs/design/sparse.md §6).
#include "bind/sparse_bind.hpp"

#include "sparse/direct.hpp"
#include "sparse/format.hpp"

namespace nb = nanobind;

namespace sd {

void bind_sparse(nb::module_& core) {
    nb::module_ m = core.def_submodule("sparse", "stardust sparse arrays native core");
    sparse::bind_direct(m);
    sparse::bind_canonical(m);
    sparse::bind_convert(m);
    sparse::bind_dok(m);
    sparse::bind_spmv(m);
    sparse::bind_binop(m);
    sparse::bind_spgemm(m);
    sparse::bind_reduce(m);
}

}  // namespace sd
