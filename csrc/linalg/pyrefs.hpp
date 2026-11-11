// Python objects the core uses on every call, looked up once (docs/design/linalg.md §3.4).
#pragma once

#include <nanobind/nanobind.h>

#include <initializer_list>

namespace sd::linalg {

namespace nb = nanobind;

// The result namedtuples defined in _results.py, in the order _setup receives them.
enum class Result { Slogdet, LU, QR, QRPivoted, Lstsq, Eigh, SVD, Eig, Schur, Hessenberg, LDL, Count };

struct PyRefs {
    // From the array bindings on _core itself, at import of _core (docs/design/linalg.md §3.6).
    PyObject* ndarray_type = nullptr;
    PyObject* asarray = nullptr;
    PyObject* get_num_threads = nullptr;
    PyObject* dtypes[4] = {};  // sd.bool, sd.int64, sd.float64, sd.complex128 (DType order)
    // From stardust.linalg's Python files, via _core._setup().
    PyObject* linalg_error = nullptr;
    PyObject* results[static_cast<int>(Result::Count)] = {};
};

// The references are deliberately never released: they live until the
// process ends, and releasing them during interpreter shutdown would be unsafe.
PyRefs& refs();

void load_sdnp_refs(nb::module_& core);
void bind_setup(nb::module_& m);

// Builds a result namedtuple from its fields (in field order).
nb::object make_result(Result r, std::initializer_list<nb::handle> fields);

}  // namespace sd::linalg
