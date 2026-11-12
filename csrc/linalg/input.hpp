// Reading an argument: an sd.ndarray taken directly as the NDArray it is
// (docs/design/linalg.md §3.2); the names only serve error messages.
#pragma once

#include "array/ndarray.hpp"
#include "linalg/types.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace sd::linalg {

namespace nb = nanobind;

class Input {
public:
    // Raises TypeError unless `obj` is an sd.ndarray (docs/api/linalg.md §1.2).
    // `func` and `arg` only name things in error messages. The array is
    // borrowed: the caller's argument keeps it alive for the whole call.
    Input(nb::handle obj, const char* func, const char* arg);
    Input(const Input&) = delete;
    Input& operator=(const Input&) = delete;

    DType dtype() const { return x_->dtype; }
    int ndim() const { return x_->ndim; }
    std::int64_t shape(int d) const { return x_->shape[d < 0 ? d + x_->ndim : d]; }
    std::int64_t stride(int d) const { return x_->strides[d < 0 ? d + x_->ndim : d]; }  // bytes
    const std::byte* data() const { return x_->data; }

    // The leading `ndim() - core` axes: the batch part when the last `core`
    // axes are one matrix (core = 2) or one vector (core = 1).
    std::span<const std::int64_t> batch_shape(int core) const { return {x_->shape.data(), size_t(x_->ndim - core)}; }
    std::span<const std::int64_t> batch_strides(int core) const { return {x_->strides.data(), size_t(x_->ndim - core)}; }

    std::span<const std::int64_t> shape_span() const { return x_->shape_span(); }
    std::span<const std::int64_t> strides_span() const { return x_->strides_span(); }

    const char* func() const { return func_; }
    const char* arg() const { return arg_; }

private:
    const NDArray* x_;
    const char* func_;
    const char* arg_;
};

// Checks shared by the matrix functions; all raise ValueError (docs/api/linalg.md §1.2).
void require_matrix(const Input& x);  // ndim >= 2
void require_square(const Input& x);  // ndim >= 2 and the last two axes equal

}  // namespace sd::linalg
