// Python objects the core uses on every call, looked up once (docs/design/linalg.md §3.4).
#pragma once

#include <nanobind/nanobind.h>

#include <initializer_list>

namespace sd::linalg {

namespace nb = nanobind;

// The result namedtuples defined in stardust/_results.py (LINALG), in the order _setup receives them.
enum class Result { Slogdet, LU, QR, QRPivoted, Lstsq, Eigh, SVD, Eig, Schur, Hessenberg, LDL, Bandwidth, Count };

struct PyRefs {
    // From stardust.linalg's Python files, via _core._setup().
    PyObject* linalg_error = nullptr;
    PyObject* results[static_cast<int>(Result::Count)] = {};
};

// The references are deliberately never released: they live until the
// process ends, and releasing them during interpreter shutdown would be unsafe.
PyRefs& refs();

void bind_setup(nb::module_& m);

// Builds a result namedtuple from its fields (in field order).
nb::object make_result(Result r, std::initializer_list<nb::handle> fields);

}  // namespace sd::linalg
