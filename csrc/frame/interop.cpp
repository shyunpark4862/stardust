#include "frame/interop.hpp"

#include "array/ndarray.hpp"
#include "array/ops/reduction.hpp"
#include "bind/pyutil.hpp"

#include <cstring>
#include <string>

namespace sd::frame {

namespace {

NDArray view_of(const Column& col, bool writeable) {
    NDArray a;
    a.buffer = col.values;
    a.data = col.values->data() + col.offset * itemsize(col.dtype);
    a.dtype = col.dtype;
    a.ndim = 1;
    a.shape[0] = col.length;
    a.strides[0] = itemsize(col.dtype);
    a.update_flags();
    a.writeable = writeable;
    return a;
}

}  // namespace

nb::object to_array_view(const Column& col, bool writeable) {
    if (col.dtype == DType::String) throw nb::type_error("a string column has no array (the array area has no string dtype)");
    return wrap(view_of(col, writeable));
}

Column from_array(nb::handle obj) {
    const NDArray& a = require_ndarray(obj, "data");
    if (a.ndim != 1) throw nb::value_error(("expected a 1-D array, got " + std::to_string(a.ndim) + " dimensions").c_str());
    if (a.dtype == DType::Complex128) throw nb::type_error("a column cannot hold complex128 (the frame has no complex dtype)");
    Column col = make_fixed(a.dtype, a.shape[0]);
    const std::int64_t size = itemsize(a.dtype);
    std::byte* out = col.values->data();
    if (a.strides[0] == size) {
        std::memcpy(out, a.data, static_cast<std::size_t>(a.shape[0] * size));
    } else {
        for (std::int64_t i = 0; i < a.shape[0]; ++i) std::memcpy(out + i * size, a.data + i * a.strides[0], static_cast<std::size_t>(size));
    }
    return col;
}

nb::object columns_to_array(const std::vector<Column>& cols, std::int64_t n) {
    const DType d = cols.empty() ? DType::Float64 : cols[0].dtype;
    const std::int64_t size = itemsize(d), k = static_cast<std::int64_t>(cols.size());
    const std::int64_t shape[2] = {n, k}, strides[2] = {size, n * size};
    NDArray out = new_array_strided(d, shape, strides, n * k * size);
    for (std::int64_t j = 0; j < k; ++j) {
        const Column& c = cols[j];
        if (c.dtype != d || c.null_count != 0 || c.length != n) throw nb::value_error("columns_to_array: one dtype, no NA, n rows");
        std::memcpy(out.data + j * n * size, c.values->data() + c.offset * size, static_cast<std::size_t>(n * size));
    }
    return wrap(std::move(out));
}

double array_sum_float64(const Column& col) {
    const nb::object view = wrap(view_of(col, /*writeable=*/false));
    return nb::cast<double>(sd::sum(view, nb::none(), dtype_object(DType::Float64), /*keepdims=*/false));
}

}  // namespace sd::frame
