// Python objects the core uses on every call, looked up once (ARCHITECTURE.md §1).
#pragma once

#include <nanobind/nanobind.h>

namespace sdla {

namespace nb = nanobind;

struct PyRefs {
    // From stardust_numpy, at import of _core.
    PyObject* ndarray_type = nullptr;
    PyObject* asarray = nullptr;
    PyObject* get_num_threads = nullptr;
    // From stardust_linalg's Python files, via _core._setup().
    PyObject* linalg_error = nullptr;
    PyObject* slogdet_result = nullptr;
};

// The references are deliberately never released: they live until the
// process ends, and releasing them during interpreter shutdown would be unsafe.
PyRefs& refs();

void load_sdnp_refs();
void bind_setup(nb::module_& m);

}  // namespace sdla
