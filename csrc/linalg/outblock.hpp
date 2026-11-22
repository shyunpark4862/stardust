// Result memory: an NDArray with a fresh Buffer, laid out the way LAPACK
// writes it, handed to Python as the array it is (docs/design/linalg.md §3.3).
#pragma once

#include "array/ndarray.hpp"
#include "linalg/types.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sd::linalg {

namespace nb = nanobind;

// A freshly allocated result array, not yet handed to Python.
struct Out {
    NDArray arr;                        // owns the memory (a fresh Buffer)
    std::byte* data = nullptr;
    std::vector<std::int64_t> strides;  // bytes, one per axis
    bool zeroed = false;                // the memory starts as zeros (alloc_matrices(..., zeroed))

    template <class T>
    T* as() const {
        return reinterpret_cast<T*>(data);
    }
};

// (batch..., m, n): every matrix column-major, batch axes outside in C order (docs/design/linalg.md §4).
// `zeroed`: zeroed memory wanted (results that are mostly zero: block_diag, kronsum); granted for
// large blocks only, as Out::zeroed says -- otherwise the caller writes the zeros.
Out alloc_matrices(
    const std::vector<std::int64_t>& batch, std::int64_t m, std::int64_t n, DType dt, bool zeroed = false
);
// (batch..., m): every vector contiguous, batch axes outside in C order.
Out alloc_vectors(const std::vector<std::int64_t>& batch, std::int64_t m, DType dt);
// Any shape in C order (results that are not matrices: reductions, scalars per batch element).
Out alloc_c(const std::vector<std::int64_t>& shape, DType dt);

// The result as an sd.ndarray.
nb::object to_array(const Out& out);

// A 0-dimensional result is a Python scalar (docs/api/linalg.md §1.6).
nb::object py_scalar(double v);
nb::object py_scalar(cplx v);
nb::object py_scalar(std::int64_t v);


}  // namespace sd::linalg
