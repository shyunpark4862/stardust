#include "linalg/input.hpp"

#include "bind/pyutil.hpp"

#include <cstring>
#include <string>

namespace sd::linalg {

namespace {

std::string where(const Input& x) { return std::string(x.func()) + "(): '" + x.arg() + "'"; }

}  // namespace

Input::Input(nb::handle obj, const char* func, const char* arg) : func_(func), arg_(arg) {
    if (!is_ndarray(obj)) {
        throw nb::type_error(
            (std::string(func) + "(): '" + arg + "' must be an sd.ndarray, not " + py_type_name(obj)).c_str()
        );
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
