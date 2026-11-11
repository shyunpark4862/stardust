// Python objects the core uses on every call, looked up once (ARCHITECTURE.md §1).
#pragma once

#include <nanobind/nanobind.h>

#include <initializer_list>

namespace sdla {

namespace nb = nanobind;

// The result namedtuples defined in _results.py, in the order _setup receives them.
enum class Result { Slogdet, LU, QR, QRPivoted, Lstsq, Eigh, SVD, Eig, Schur, Hessenberg, LDL, Count };

struct PyRefs {
    // From stardust_numpy, at import of _core.
    PyObject* ndarray_type = nullptr;
    PyObject* asarray = nullptr;
    PyObject* get_num_threads = nullptr;
    PyObject* dtypes[4] = {};  // sdnp.bool, sdnp.int64, sdnp.float64, sdnp.complex128 (DType order)
    // From stardust_linalg's Python files, via _core._setup().
    PyObject* linalg_error = nullptr;
    PyObject* results[static_cast<int>(Result::Count)] = {};
};

// The references are deliberately never released: they live until the
// process ends, and releasing them during interpreter shutdown would be unsafe.
PyRefs& refs();

void load_sdnp_refs();
void bind_setup(nb::module_& m);

// Builds a result namedtuple from its fields (in field order).
nb::object make_result(Result r, std::initializer_list<nb::handle> fields);

}  // namespace sdla
