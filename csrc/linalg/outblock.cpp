#include "linalg/outblock.hpp"

namespace sd::linalg {

namespace {

// The layout is linalg's (column-major matrices, batch axes outside in C
// order: docs/design/linalg.md §4); the memory is a Buffer like every array's,
// 128-byte aligned, so LAPACK's 16-byte alignment needs hold.
Out alloc(const std::vector<std::int64_t>& shape, std::vector<std::int64_t> strides, std::int64_t nbytes, DType dt) {
    Out out;
    out.arr = new_array_strided(dt, shape, strides, nbytes);
    out.data = out.arr.data;
    out.strides = std::move(strides);
    return out;
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

Out alloc_matrices(const std::vector<std::int64_t>& batch, std::int64_t m, std::int64_t n, DType dt) {
    const std::int64_t isz = itemsize(dt);
    std::vector<std::int64_t> shape = batch, strides(batch.size() + 2);
    shape.push_back(m);
    shape.push_back(n);
    strides[batch.size()] = isz;
    strides[batch.size() + 1] = isz * m;
    const std::int64_t total = fill_c_strides(shape, batch.size(), strides, isz * m * n);
    return alloc(shape, std::move(strides), total, dt);
}

Out alloc_vectors(const std::vector<std::int64_t>& batch, std::int64_t m, DType dt) {
    std::vector<std::int64_t> shape = batch;
    shape.push_back(m);
    return alloc_c(shape, dt);
}

Out alloc_c(const std::vector<std::int64_t>& shape, DType dt) {
    std::vector<std::int64_t> strides(shape.size());
    const std::int64_t total = fill_c_strides(shape, shape.size(), strides, itemsize(dt));
    return alloc(shape, std::move(strides), total, dt);
}

nb::object to_array(const Out& out) { return nb::cast(out.arr); }

nb::object py_scalar(double v) { return nb::steal(PyFloat_FromDouble(v)); }
nb::object py_scalar(cplx v) { return nb::steal(PyComplex_FromDoubles(v.real(), v.imag())); }
nb::object py_scalar(std::int64_t v) { return nb::steal(PyLong_FromLongLong(v)); }


}  // namespace sd::linalg
