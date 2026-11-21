#include "linalg/input.hpp"

#include "bind/pyutil.hpp"

#include <cstring>
#include <utility>
#include <string>

namespace sd::linalg {

namespace {

std::string where(const Input& x) { return std::string(x.func()) + "(): '" + x.arg() + "'"; }

// A sparse array of sd.sparse given to a linalg function (docs/design/sparse.md §3.10): the
// message names the sd.sparse function to use, once that function exists, else the dense
// conversion. Only the error path looks; other calls are unchanged.
std::string sparse_hint(nb::handle obj, const char* func) {
    if (!is_sparse_array(obj)) return "";
    static const std::pair<const char*, const char*> table[] = {
        {"solve", "spsolve"},
        {"solve_triangular", "spsolve_triangular"},
        {"lu", "splu"},
        {"lu_solve", "splu"},
        {"inv", "inv"},
        {"expm", "expm"},
        {"eigh", "eigsh"},
        {"eigvalsh", "eigsh"},
        {"eig", "eigs"},
        {"eigvals", "eigs"},
        {"svd", "svds"},
        {"svdvals", "svds"},
        {"matrix_norm", "norm"},
        {"vector_norm", "norm"},
        {"lstsq", "lsqr"},
        {"kron", "kron"},
        {"kronsum", "kronsum"},
        {"block_diag", "block_diag"},
        {"bandwidth", "spbandwidth"},
    };  // fmt: skip
    nb::object sparse = nb::module_::import_("stardust.sparse");
    const std::string f = func;
    for (const auto& [name, target] : table) {
        if (f != name || !nb::hasattr(sparse, target)) continue;
        if (f == "lstsq") {
            if (!nb::hasattr(sparse, "lsmr")) break;
            return "; for a sparse array use sd.sparse.lsqr or sd.sparse.lsmr";
        }
        return std::string("; for a sparse array use sd.sparse.") + target;
    }
    return "; a sparse array has no " + f + ": use " + f + "(A.toarray())";
}

}  // namespace

Input::Input(nb::handle obj, const char* func, const char* arg) : func_(func), arg_(arg) {
    if (!is_ndarray(obj)) {
        throw nb::type_error((std::string(func) + "(): '" + arg + "' must be an sd.ndarray, not " + py_type_name(obj) +
                              sparse_hint(obj, func))
                                 .c_str());
    }
    x_ = &as_ndarray(obj);
}

void require_matrix(const Input& x) {
    if (x.ndim() < 2)
        throw nb::value_error((where(x) + " must have at least 2 dimensions, got " + std::to_string(x.ndim())).c_str());
}

void require_square(const Input& x) {
    require_matrix(x);
    if (x.shape(-1) != x.shape(-2))
        throw nb::value_error((where(x) + " must be square in its last two dimensions, got " +
                               std::to_string(x.shape(-2)) + "x" + std::to_string(x.shape(-1)))
                                  .c_str());
}

}  // namespace sd::linalg
