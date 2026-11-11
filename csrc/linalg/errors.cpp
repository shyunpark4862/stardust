#include "linalg/errors.hpp"

#include "linalg/pyrefs.hpp"

namespace sd::linalg {

void register_error_translator() {
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const LinAlgError& e) {
            PyObject* type = refs().linalg_error ? refs().linalg_error : PyExc_ValueError;
            PyErr_SetString(type, e.what());
        }
    });
}

}  // namespace sd::linalg
