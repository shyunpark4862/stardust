// Result memory: allocated here, exported through the buffer protocol, and
// wrapped by sdnp.asarray without a copy (ARCHITECTURE.md §3.3, DESIGN.md §2.9).
#pragma once

#include "core/types.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sdla {

namespace nb = nanobind;

// A freshly allocated result array, not yet handed to Python.
struct Out {
    nb::object block;                  // the Python object that owns the memory
    std::byte* data = nullptr;
    std::vector<std::int64_t> strides; // bytes, one per axis
};

// (batch..., m, n): every matrix column-major, batch axes outside in C order (DESIGN.md §2.3).
Out alloc_matrices(const std::vector<std::int64_t>& batch, std::int64_t m, std::int64_t n, bool cplx);
// (batch..., m): every vector contiguous, batch axes outside in C order.
Out alloc_vectors(const std::vector<std::int64_t>& batch, std::int64_t m, bool cplx);
// (batch...): one number per batch element, C order. Needs batch.size() >= 1.
Out alloc_scalars(const std::vector<std::int64_t>& batch, bool cplx);

// The sdnp.ndarray viewing the block (through the cached sdnp.asarray).
nb::object to_sdnp(const Out& out);

// A 0-dimensional result is a Python scalar (API.md §0.1, Z1).
nb::object py_scalar(double v);
nb::object py_scalar(cplx v);

void bind_outblock(nb::module_& m);

}  // namespace sdla
