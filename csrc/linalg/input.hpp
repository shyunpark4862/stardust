// Reading an argument: an sd.ndarray seen through the buffer protocol, the
// boundary path in use until linalg takes NDArray directly
// (docs/design/linalg.md §3.6).
#pragma once

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
    // `func` and `arg` only name things in error messages.
    Input(nb::handle obj, const char* func, const char* arg);
    ~Input();
    Input(const Input&) = delete;
    Input& operator=(const Input&) = delete;

    DType dtype() const { return dtype_; }
    int ndim() const { return view_.ndim; }
    std::int64_t shape(int d) const { return view_.shape[d < 0 ? d + view_.ndim : d]; }
    std::int64_t stride(int d) const { return view_.strides[d < 0 ? d + view_.ndim : d]; }  // bytes
    const std::byte* data() const { return static_cast<const std::byte*>(view_.buf); }

    // The leading `ndim() - core` axes: the batch part when the last `core`
    // axes are one matrix (core = 2) or one vector (core = 1).
    std::span<const Py_ssize_t> batch_shape(int core) const { return {view_.shape, size_t(view_.ndim - core)}; }
    std::span<const Py_ssize_t> batch_strides(int core) const { return {view_.strides, size_t(view_.ndim - core)}; }

    std::span<const Py_ssize_t> shape_span() const { return {view_.shape, size_t(view_.ndim)}; }
    std::span<const Py_ssize_t> strides_span() const { return {view_.strides, size_t(view_.ndim)}; }

    const char* func() const { return func_; }
    const char* arg() const { return arg_; }

private:
    Py_buffer view_{};
    DType dtype_ = DType::Float64;
    const char* func_;
    const char* arg_;
};

// Checks shared by the matrix functions; all raise ValueError (docs/api/linalg.md §1.2).
void require_matrix(const Input& x);  // ndim >= 2
void require_square(const Input& x);  // ndim >= 2 and the last two axes equal

}  // namespace sd::linalg
