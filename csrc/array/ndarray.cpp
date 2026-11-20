#include "array/ndarray.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>

namespace sd {

std::int64_t NDArray::size() const {
    std::int64_t n = 1;
    for (int d = 0; d < ndim; ++d) n *= shape[d];
    return n;
}

// An axis of extent 1 is never stepped over, so its stride is irrelevant to
// contiguity; an empty array is trivially contiguous (numpy uses the same
// "relaxed strides" rule).
void NDArray::update_flags() {
    if (size() == 0) {
        c_contiguous = f_contiguous = true;
        return;
    }
    std::int64_t expect = itemsize();
    c_contiguous = true;
    for (int d = ndim - 1; d >= 0; --d) {
        if (shape[d] == 1) continue;
        if (strides[d] != expect) {
            c_contiguous = false;
            break;
        }
        expect *= shape[d];
    }
    expect = itemsize();
    f_contiguous = true;
    for (int d = 0; d < ndim; ++d) {
        if (shape[d] == 1) continue;
        if (strides[d] != expect) {
            f_contiguous = false;
            break;
        }
        expect *= shape[d];
    }
}

std::string shape_str(std::span<const std::int64_t> shape) {
    std::string out = "(";
    for (size_t i = 0; i < shape.size(); ++i) out += (i ? ", " : "") + std::to_string(shape[i]);
    return out + (shape.size() == 1 ? ",)" : ")");
}

namespace {

// checked_size without the 0-d check: the number of dimensions, negative
// extents and the byte size's overflow.
std::int64_t checked_count(std::span<const std::int64_t> shape, DType dtype) {
    if (shape.size() > static_cast<size_t>(kMaxNdim))
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    const std::int64_t limit = std::numeric_limits<std::int64_t>::max() / itemsize(dtype);
    std::int64_t n = 1;
    for (std::int64_t extent : shape) {
        if (extent < 0) throw nb::value_error("negative dimensions are not allowed");
        if (extent != 0 && n > limit / extent) throw nb::value_error("array is too big");
        n *= extent;
    }
    return n;
}

}  // namespace

std::int64_t checked_size(std::span<const std::int64_t> shape, DType dtype) {
    if (shape.empty()) throw nb::value_error("0-dimensional arrays are not supported (stardust has no 0-d arrays)");
    return checked_count(shape, dtype);
}

namespace {

// Allocates a contiguous array whose axes are laid out in memory in the order
// `order` (slowest-varying axis first).
NDArray allocate(DType dtype, std::span<const std::int64_t> shape, std::span<const int> order, bool zeroed) {
    const std::int64_t n = checked_size(shape, dtype);
    NDArray a;
    a.dtype = dtype;
    a.ndim = static_cast<int>(shape.size());
    std::copy(shape.begin(), shape.end(), a.shape.begin());
    a.buffer = Buffer::allocate(n * itemsize(dtype), zeroed);
    a.data = a.buffer->data();
    std::int64_t stride = itemsize(dtype);
    for (int k = a.ndim - 1; k >= 0; --k) {
        a.strides[order[k]] = stride;
        stride *= std::max<std::int64_t>(shape[order[k]], 1);
    }
    a.update_flags();
    return a;
}

}  // namespace

NDArray new_array(DType dtype, std::span<const std::int64_t> shape, bool zeroed) {
    std::array<int, kMaxNdim> order;
    std::iota(order.begin(), order.end(), 0);
    return allocate(dtype, shape, {order.data(), shape.size()}, zeroed);
}

NDArray new_array_like(const NDArray& proto, DType dtype, bool zeroed) {
    // Sort axes by decreasing |stride|; ties keep the original (C) order. A
    // stride of 0 (a broadcast axis) carries no layout information, as in the
    // iterator (docs/design/array.md §4.2): such an axis keeps its C place,
    // and only the other axes are sorted among the places they hold.
    std::array<int, kMaxNdim> order;
    std::iota(order.begin(), order.begin() + proto.ndim, 0);
    std::array<int, kMaxNdim> slots;  // the places of the axes with a stride
    int live = 0;
    for (int d = 0; d < proto.ndim; ++d)
        if (proto.strides[d] != 0) slots[live++] = d;
    std::array<int, kMaxNdim> axes = slots;
    // Insertion sort (stable): at most 32 axes, and std::stable_sort would
    // allocate a scratch buffer on every call (docs/design/array.md §11).
    for (int i = 1; i < live; ++i)
        for (int j = i; j > 0 && std::llabs(proto.strides[axes[j]]) > std::llabs(proto.strides[axes[j - 1]]); --j)
            std::swap(axes[j], axes[j - 1]);
    for (int i = 0; i < live; ++i) order[slots[i]] = axes[i];
    return allocate(dtype, proto.shape_span(), {order.data(), static_cast<size_t>(proto.ndim)}, zeroed);
}

NDArray new_array_strided(
    DType dtype, std::span<const std::int64_t> shape, std::span<const std::int64_t> strides, std::int64_t nbytes
) {
    // The checks of new_array before anything is copied into the fixed
    // kMaxNdim-long shape and strides. A 0-d shape is allowed here: sd.linalg
    // computes a scalar result in one before handing out a Python scalar.
    checked_count(shape, dtype);
    NDArray a;
    a.dtype = dtype;
    a.ndim = static_cast<int>(shape.size());
    std::copy(shape.begin(), shape.end(), a.shape.begin());
    std::copy(strides.begin(), strides.end(), a.strides.begin());
    a.buffer = Buffer::allocate(nbytes, false);
    a.data = a.buffer->data();
    a.update_flags();
    return a;
}

}  // namespace sd
