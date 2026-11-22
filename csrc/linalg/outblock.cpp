#include "linalg/outblock.hpp"

#include "base/buffer.hpp"

#include <algorithm>

namespace sd::linalg {

namespace {

constexpr std::int64_t kZeroedMin = std::int64_t(1) << 20;  // bytes

// The layout is linalg's (column-major matrices, batch axes outside in C
// order: docs/design/linalg.md §4); the memory is a Buffer like every array's,
// 128-byte aligned, so LAPACK's 16-byte alignment needs hold.
Out alloc(
    const std::vector<std::int64_t>& shape,
    std::vector<std::int64_t> strides,
    std::int64_t nbytes,
    DType dt,
    bool zeroed = false
) {
    Out out;
    // Zeroed memory only for large blocks (calloc gives them as fresh zero pages, which cost no
    // more than writing the zeros); a small one would be zeroed by calloc's memset, and fresh
    // pages for a medium one made small block_diag and kronsum calls slower. out.zeroed tells.
    zeroed = zeroed && nbytes >= kZeroedMin;
    out.zeroed = zeroed;
    if (zeroed) {  // new_array_strided's array on zeroed memory
        checked_size(shape, dt);
        NDArray& a = out.arr;
        a.dtype = dt;
        a.ndim = static_cast<int>(shape.size());
        std::copy(shape.begin(), shape.end(), a.shape.begin());
        std::copy(strides.begin(), strides.end(), a.strides.begin());
        a.buffer = Buffer::allocate(nbytes, true);
        a.data = a.buffer->data();
        a.update_flags();
    } else {
        out.arr = new_array_strided(dt, shape, strides, nbytes);
    }
    out.data = out.arr.data;
    out.strides = std::move(strides);
    return out;
}

// The checks new_array makes on a shape, with the same messages: at most kMaxNdim
// axes, and a byte count within int64. An empty axis empties the array, but the
// other extents are still checked (numpy's rule). With the check passed, every
// stride and the total below are products that fit.
void check_shape(const std::vector<std::int64_t>& shape, DType dt) {
    if (shape.size() > static_cast<std::size_t>(kMaxNdim))
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    std::int64_t nbytes = itemsize(dt);
    for (std::int64_t extent : shape)
        if (extent != 0 && __builtin_mul_overflow(nbytes, extent, &nbytes)) throw nb::value_error("array is too big");
}

std::int64_t fill_c_strides(
    const std::vector<std::int64_t>& shape, std::size_t count, std::vector<std::int64_t>& strides, std::int64_t block
) {
    std::int64_t s = block;
    for (std::size_t d = count; d-- > 0;) {
        strides[d] = s;
        s *= shape[d];
    }
    return s;
}

}  // namespace

Out alloc_matrices(const std::vector<std::int64_t>& batch, std::int64_t m, std::int64_t n, DType dt, bool zeroed) {
    const std::int64_t isz = itemsize(dt);
    std::vector<std::int64_t> shape = batch, strides(batch.size() + 2);
    shape.push_back(m);
    shape.push_back(n);
    check_shape(shape, dt);
    strides[batch.size()] = isz;
    strides[batch.size() + 1] = isz * m;
    const std::int64_t total = fill_c_strides(shape, batch.size(), strides, isz * m * n);
    return alloc(shape, std::move(strides), total, dt, zeroed);
}

Out alloc_vectors(const std::vector<std::int64_t>& batch, std::int64_t m, DType dt) {
    std::vector<std::int64_t> shape = batch;
    shape.push_back(m);
    return alloc_c(shape, dt);
}

Out alloc_c(const std::vector<std::int64_t>& shape, DType dt) {
    check_shape(shape, dt);
    std::vector<std::int64_t> strides(shape.size());
    const std::int64_t total = fill_c_strides(shape, shape.size(), strides, itemsize(dt));
    return alloc(shape, std::move(strides), total, dt);
}

nb::object to_array(const Out& out) { return nb::cast(out.arr); }

nb::object py_scalar(double v) { return nb::steal(PyFloat_FromDouble(v)); }
nb::object py_scalar(cplx v) { return nb::steal(PyComplex_FromDoubles(v.real(), v.imag())); }
nb::object py_scalar(std::int64_t v) { return nb::steal(PyLong_FromLongLong(v)); }


}  // namespace sd::linalg
