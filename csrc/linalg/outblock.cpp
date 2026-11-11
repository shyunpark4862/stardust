#include "linalg/outblock.hpp"

#include "linalg/pyrefs.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <new>

namespace sd::linalg {

namespace {

struct Free {
    void operator()(std::byte* p) const { std::free(p); }
};

// The Python-visible owner of one result's memory.
struct OutBlock {
    std::unique_ptr<std::byte, Free> data;
    std::int64_t nbytes = 0;
    std::vector<Py_ssize_t> shape, strides;
    DType dtype = DType::Float64;
};

// The exact formats sd.ndarray exports, so sd.asarray shares instead of copying
// (docs/design/array.md §12.3).
const char* format_of(DType d) {
    switch (d) {
        case DType::Bool: return "?";
        case DType::Int64: return "q";
        case DType::Float64: return "d";
        default: return "Zd";
    }
}

int block_getbuffer(PyObject* self, Py_buffer* view, int flags) {
    OutBlock& b = *nb::inst_ptr<OutBlock>(self);
    if ((flags & PyBUF_STRIDES) != PyBUF_STRIDES) {
        PyErr_SetString(PyExc_BufferError, "stardust.linalg result blocks are exported with strides only");
        view->obj = nullptr;
        return -1;
    }
    view->buf = b.data.get();
    view->obj = Py_NewRef(self);  // keeps the block alive while exported
    view->len = b.nbytes;
    view->readonly = 0;
    view->itemsize = itemsize(b.dtype);
    view->format = (flags & PyBUF_FORMAT) ? const_cast<char*>(format_of(b.dtype)) : nullptr;
    view->ndim = static_cast<int>(b.shape.size());
    view->shape = b.shape.data();
    view->strides = b.strides.data();
    view->suboffsets = nullptr;
    view->internal = nullptr;
    return 0;
}

PyType_Slot block_slots[] = {{Py_bf_getbuffer, reinterpret_cast<void*>(block_getbuffer)}, {0, nullptr}};

// Allocates a block with the given shape and strides; `nbytes` spans all elements.
Out alloc(const std::vector<std::int64_t>& shape, std::vector<std::int64_t> strides, std::int64_t nbytes, DType dt) {
    OutBlock b;
    b.dtype = dt;
    b.nbytes = nbytes;
    b.shape.assign(shape.begin(), shape.end());
    b.strides.assign(strides.begin(), strides.end());
    // 128-byte aligned (as the array's own buffers, docs/design/array.md §3.2); size rounded up
    // to the alignment.
    const std::size_t bytes = static_cast<std::size_t>(std::max<std::int64_t>(nbytes, 1) + 127) / 128 * 128;
    b.data.reset(static_cast<std::byte*>(std::aligned_alloc(128, bytes)));
    if (!b.data) throw std::bad_alloc();
    std::byte* data = b.data.get();
    Out out;
    out.block = nb::cast(std::move(b), nb::rv_policy::move);
    out.data = data;
    out.strides = std::move(strides);
    return out;
}

// C-order strides for the leading `count` axes of `shape`, starting from the
// byte size of one trailing block. Returns the total byte size.
std::int64_t fill_c_strides(const std::vector<std::int64_t>& shape, std::size_t count,
                            std::vector<std::int64_t>& strides, std::int64_t block) {
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

nb::object to_sdnp(const Out& out) {
    PyObject* args[] = {out.block.ptr()};
    PyObject* r = PyObject_Vectorcall(refs().asarray, args, 1, nullptr);
    if (!r) throw nb::python_error();
    return nb::steal(r);
}

nb::object py_scalar(double v) { return nb::steal(PyFloat_FromDouble(v)); }
nb::object py_scalar(cplx v) { return nb::steal(PyComplex_FromDoubles(v.real(), v.imag())); }
nb::object py_scalar(std::int64_t v) { return nb::steal(PyLong_FromLongLong(v)); }

void bind_outblock(nb::module_& m) {
    nb::class_<OutBlock>(m, "_OutBlock", nb::type_slots(block_slots));
}

}  // namespace sd::linalg
