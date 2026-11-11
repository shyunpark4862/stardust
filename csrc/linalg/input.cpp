#include "linalg/input.hpp"

#include "linalg/pyrefs.hpp"

#include <cstring>
#include <string>

namespace sd::linalg {

namespace {

DType parse_format(const char* f) {
    if (std::strcmp(f, "?") == 0) return DType::Bool;
    if (std::strcmp(f, "q") == 0) return DType::Int64;
    if (std::strcmp(f, "d") == 0) return DType::Float64;
    if (std::strcmp(f, "Zd") == 0) return DType::Complex128;
    // sd.ndarray only exports these four (docs/design/array.md §12.3).
    throw nb::type_error((std::string("unexpected buffer format '") + f + "'").c_str());
}

std::string where(const Input& x) { return std::string(x.func()) + "(): '" + x.arg() + "'"; }

}  // namespace

Input::Input(nb::handle obj, const char* func, const char* arg) : func_(func), arg_(arg) {
    if (!PyObject_TypeCheck(obj.ptr(), reinterpret_cast<PyTypeObject*>(refs().ndarray_type))) {
        throw nb::type_error((std::string(func) + "(): '" + arg + "' must be a stardust.ndarray, not " +
                              Py_TYPE(obj.ptr())->tp_name)
                                 .c_str());
    }
    if (PyObject_GetBuffer(obj.ptr(), &view_, PyBUF_RECORDS_RO) != 0) throw nb::python_error();
    try {
        dtype_ = parse_format(view_.format);
    } catch (...) {
        PyBuffer_Release(&view_);
        throw;
    }
}

Input::~Input() { PyBuffer_Release(&view_); }

void require_matrix(const Input& x) {
    if (x.ndim() < 2)
        throw nb::value_error(
            (where(x) + " must have at least 2 dimensions, got " + std::to_string(x.ndim())).c_str());
}

void require_square(const Input& x) {
    require_matrix(x);
    if (x.shape(-1) != x.shape(-2))
        throw nb::value_error((where(x) + " must be square in its last two dimensions, got " +
                               std::to_string(x.shape(-2)) + "x" + std::to_string(x.shape(-1)))
                                  .c_str());
}

}  // namespace sd::linalg
