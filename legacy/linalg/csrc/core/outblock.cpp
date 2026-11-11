#include "core/outblock.hpp"

#include "core/pyrefs.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <new>

namespace sdla {

namespace {

struct Free {
    void operator()(std::byte* p) const { std::free(p); }
};

// The Python-visible owner of one result's memory.
struct OutBlock {
    std::unique_ptr<std::byte, Free> data;
    std::int64_t nbytes = 0;
    std::vector<Py_ssize_t> shape, strides;
    bool cplx = false;
};

int block_getbuffer(PyObject* self, Py_buffer* view, int flags) {
    OutBlock& b = *nb::inst_ptr<OutBlock>(self);
    if ((flags & PyBUF_STRIDES) != PyBUF_STRIDES) {
        PyErr_SetString(PyExc_BufferError, "stardust_linalg result blocks are exported with strides only");
        view->obj = nullptr;
        return -1;
    }
    view->buf = b.data.get();
    view->obj = Py_NewRef(self);  // keeps the block alive while exported
    view->len = b.nbytes;
    view->readonly = 0;
    view->itemsize = b.cplx ? 16 : 8;
    // The exact formats stardust-numpy exports, so it shares instead of copying.
    view->format = (flags & PyBUF_FORMAT) ? const_cast<char*>(b.cplx ? "Zd" : "d") : nullptr;
    view->ndim = static_cast<int>(b.shape.size());
    view->shape = b.shape.data();
    view->strides = b.strides.data();
    view->suboffsets = nullptr;
    view->internal = nullptr;
    return 0;
}

PyType_Slot block_slots[] = {{Py_bf_getbuffer, reinterpret_cast<void*>(block_getbuffer)}, {0, nullptr}};

// Allocates a block with the given shape and strides; `nbytes` spans all elements.
Out alloc(std::vector<std::int64_t> shape, std::vector<std::int64_t> strides, std::int64_t nbytes, bool cplx) {
    OutBlock b;
    b.cplx = cplx;
    b.nbytes = nbytes;
    b.shape.assign(shape.begin(), shape.end());
    b.strides.assign(strides.begin(), strides.end());
    // 128-byte aligned (as stardust-numpy's own buffers); size rounded up to the alignment.
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

// C-order strides for the batch axes, starting from the size of one item.
std::int64_t fill_batch_strides(const std::vector<std::int64_t>& batch, std::vector<std::int64_t>& strides,
                                std::int64_t item) {
    std::int64_t s = item;
    for (std::size_t d = batch.size(); d-- > 0;) {
        strides[d] = s;
        s *= batch[d];
    }
    return s;  // total bytes
}

}  // namespace

Out alloc_matrices(const std::vector<std::int64_t>& batch, std::int64_t m, std::int64_t n, bool cplx) {
    const std::int64_t isz = cplx ? 16 : 8;
    std::vector<std::int64_t> shape = batch, strides(batch.size() + 2);
    shape.push_back(m);
    shape.push_back(n);
    strides[batch.size()] = isz;
    strides[batch.size() + 1] = isz * m;
    const std::int64_t total = fill_batch_strides(batch, strides, isz * m * n);
    return alloc(std::move(shape), std::move(strides), total, cplx);
}

Out alloc_vectors(const std::vector<std::int64_t>& batch, std::int64_t m, bool cplx) {
    const std::int64_t isz = cplx ? 16 : 8;
    std::vector<std::int64_t> shape = batch, strides(batch.size() + 1);
    shape.push_back(m);
    strides[batch.size()] = isz;
    const std::int64_t total = fill_batch_strides(batch, strides, isz * m);
    return alloc(std::move(shape), std::move(strides), total, cplx);
}

Out alloc_scalars(const std::vector<std::int64_t>& batch, bool cplx) {
    const std::int64_t isz = cplx ? 16 : 8;
    std::vector<std::int64_t> strides(batch.size());
    const std::int64_t total = fill_batch_strides(batch, strides, isz);
    return alloc(batch, std::move(strides), total, cplx);
}

nb::object to_sdnp(const Out& out) {
    PyObject* args[] = {out.block.ptr()};
    PyObject* r = PyObject_Vectorcall(refs().asarray, args, 1, nullptr);
    if (!r) throw nb::python_error();
    return nb::steal(r);
}

nb::object py_scalar(double v) { return nb::steal(PyFloat_FromDouble(v)); }
nb::object py_scalar(cplx v) { return nb::steal(PyComplex_FromDoubles(v.real(), v.imag())); }

void bind_outblock(nb::module_& m) {
    nb::class_<OutBlock>(m, "_OutBlock", nb::type_slots(block_slots));
}

}  // namespace sdla
