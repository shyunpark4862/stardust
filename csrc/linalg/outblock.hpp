// Result memory: allocated here, exported through the buffer protocol, and
// wrapped by sd.asarray without a copy: the boundary path in use until linalg
// builds NDArray results directly (docs/design/linalg.md §3.6).
#pragma once

#include "linalg/types.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sd::linalg {

namespace nb = nanobind;

// A freshly allocated result array, not yet handed to Python.
struct Out {
    nb::object block;                   // the Python object that owns the memory
    std::byte* data = nullptr;
    std::vector<std::int64_t> strides;  // bytes, one per axis

    template <class T>
    T* as() const {
        return reinterpret_cast<T*>(data);
    }
};

// (batch..., m, n): every matrix column-major, batch axes outside in C order (docs/design/linalg.md §4).
Out alloc_matrices(const std::vector<std::int64_t>& batch, std::int64_t m, std::int64_t n, DType dt);
// (batch..., m): every vector contiguous, batch axes outside in C order.
Out alloc_vectors(const std::vector<std::int64_t>& batch, std::int64_t m, DType dt);
// Any shape in C order (results that are not matrices: reductions, scalars per batch element).
Out alloc_c(const std::vector<std::int64_t>& shape, DType dt);

// The sd.ndarray viewing the block (through the cached sd.asarray).
nb::object to_sdnp(const Out& out);

// A 0-dimensional result is a Python scalar (docs/api/linalg.md §1.6).
nb::object py_scalar(double v);
nb::object py_scalar(cplx v);
nb::object py_scalar(std::int64_t v);

void bind_outblock(nb::module_& m);

}  // namespace sd::linalg
