// DLPack (docs/api/array.md §3.5; T0b): from_dlpack, and ndarray.__dlpack__ /
// __dlpack_device__. CPU only, the four array dtypes only.
//
// The structs are the DLPack 1.0 ABI (dlpack.h, Apache-2.0), written out
// here: the capsule "dltensor_versioned" holds a DLManagedTensorVersioned,
// the legacy capsule "dltensor" a DLManagedTensor. A consumer renames the
// capsule to "used_..." and calls the deleter when it is done.

#include "array/ops/dlpack.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/ops/copy.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace sd {

namespace {

// ------------------------------------------------------------ the ABI

struct DLDevice {
    std::int32_t device_type;
    std::int32_t device_id;
};
struct DLDataType {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};
struct DLTensor {
    void* data;
    DLDevice device;
    std::int32_t ndim;
    DLDataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;  // in elements; null means C-contiguous
    std::uint64_t byte_offset;
};
struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor*);
};
struct DLPackVersion {
    std::uint32_t major, minor;
};
struct DLManagedTensorVersioned {
    DLPackVersion version;
    void* manager_ctx;
    void (*deleter)(DLManagedTensorVersioned*);
    std::uint64_t flags;
    DLTensor dl_tensor;
};

constexpr std::int32_t kDLCPU = 1;
constexpr std::uint8_t kDLInt = 0, kDLFloat = 2, kDLComplex = 5, kDLBool = 6;
constexpr std::uint64_t kFlagReadOnly = 1u << 0, kFlagIsCopied = 1u << 1;
constexpr std::uint32_t kMajor = 1, kMinor = 0;

DLDataType to_dl(DType d) {
    switch (d) {
        case DType::Bool: return {kDLBool, 8, 1};
        case DType::Int64: return {kDLInt, 64, 1};
        case DType::Float64: return {kDLFloat, 64, 1};
        case DType::Complex128: return {kDLComplex, 128, 1};
        default: throw nb::type_error("DLPack export supports the array dtypes only");
    }
}

std::optional<DType> from_dl(DLDataType t) {
    if (t.lanes != 1) return std::nullopt;
    if (t.code == kDLBool && t.bits == 8) return DType::Bool;
    if (t.code == kDLInt && t.bits == 64) return DType::Int64;
    if (t.code == kDLFloat && t.bits == 64) return DType::Float64;
    if (t.code == kDLComplex && t.bits == 128) return DType::Complex128;
    return std::nullopt;
}

std::string dl_name(DLDataType t) {
    static const char* codes[] = {"int", "uint", "float", "opaque", "bfloat", "complex", "bool"};
    const std::string code = t.code < 7 ? codes[t.code] : "code " + std::to_string(t.code);
    return code + std::to_string(t.bits) + (t.lanes != 1 ? "x" + std::to_string(t.lanes) : "");
}

// --------------------------------------------------------------- export

// What a capsule keeps alive: a view of the array (its buffer and base) and
// the shape/strides the DLTensor points at.
struct Export {
    NDArray keep;
    std::vector<std::int64_t> shape, strides;
};

void fill_tensor(DLTensor& t, Export& e) {
    const NDArray& a = e.keep;
    e.shape.assign(a.shape_span().begin(), a.shape_span().end());
    for (int d = 0; d < a.ndim; ++d) e.strides.push_back(a.strides[d] / a.itemsize());
    t.data = a.data;
    t.device = {kDLCPU, 0};
    t.ndim = a.ndim;
    t.dtype = to_dl(a.dtype);
    t.shape = e.shape.data();
    t.strides = e.strides.data();
    t.byte_offset = 0;
}

// Deleters may run on any thread: the kept view holds Python references.
template <typename M>
void delete_export(M* m) {
    PyGILState_STATE gil = PyGILState_Ensure();
    delete static_cast<Export*>(m->manager_ctx);
    delete m;
    PyGILState_Release(gil);
}

// A capsule nobody consumed still owns its tensor.
void versioned_capsule_destructor(PyObject* cap) {
    if (PyCapsule_IsValid(cap, "dltensor_versioned")) {
        auto* m = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(cap, "dltensor_versioned"));
        if (m && m->deleter) m->deleter(m);
    }
}
void legacy_capsule_destructor(PyObject* cap) {
    if (PyCapsule_IsValid(cap, "dltensor")) {
        auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
        if (m && m->deleter) m->deleter(m);
    }
}

bool cpu_device(nb::handle dev) {
    if (dev.is_none()) return true;
    if (!PyTuple_Check(dev.ptr()) || nb::len(dev) != 2) return false;
    return nb::cast<std::int64_t>(dev[0]) == kDLCPU && nb::cast<std::int64_t>(dev[1]) == 0;
}

nb::object dlpack(nb::handle self, nb::handle stream, nb::handle max_version, nb::handle dl_device, nb::handle copy_obj) {
    const NDArray& a = as_ndarray(self);
    if (!stream.is_none()) throw nb::buffer_error("'stream' must be None for CPU arrays");
    if (!cpu_device(dl_device)) throw nb::buffer_error("only the CPU device (1, 0) is supported");
    const std::optional<bool> copy = parse_copy(copy_obj);
    bool versioned = false;
    if (!max_version.is_none()) {
        if (!PyTuple_Check(max_version.ptr()) || nb::len(max_version) != 2)
            throw nb::type_error("'max_version' must be None or a (major, minor) tuple");
        versioned = nb::cast<std::int64_t>(max_version[0]) >= 1;
    }
    auto* e = new Export{copy == true ? cast_copy(a, a.dtype) : view_of(self), {}, {}};
    const bool read_only = !e->keep.writeable;
    if (versioned) {
        auto* m = new DLManagedTensorVersioned{};
        m->version = {kMajor, kMinor};
        m->manager_ctx = e;
        m->deleter = &delete_export<DLManagedTensorVersioned>;
        m->flags = (read_only ? kFlagReadOnly : 0) | (copy == true ? kFlagIsCopied : 0);
        fill_tensor(m->dl_tensor, *e);
        PyObject* cap = PyCapsule_New(m, "dltensor_versioned", &versioned_capsule_destructor);
        if (!cap) {
            m->deleter(m);
            throw nb::python_error();
        }
        return nb::steal(cap);
    }
    // The legacy struct cannot say "read-only" (numpy refuses the same way).
    if (read_only) {
        delete e;
        throw nb::buffer_error("cannot export a read-only array with the legacy DLPack format; pass max_version=(1, 0)");
    }
    auto* m = new DLManagedTensor{};
    m->manager_ctx = e;
    m->deleter = &delete_export<DLManagedTensor>;
    fill_tensor(m->dl_tensor, *e);
    PyObject* cap = PyCapsule_New(m, "dltensor", &legacy_capsule_destructor);
    if (!cap) {
        m->deleter(m);
        throw nb::python_error();
    }
    return nb::steal(cap);
}

// --------------------------------------------------------------- import

// The owner a consumed tensor hangs on (the new array's base): its capsule
// destructor calls the producer's deleter.
void owner_versioned(PyObject* cap) {
    auto* m = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(cap, "sd_dlpack_owner"));
    if (m && m->deleter) m->deleter(m);
}
void owner_legacy(PyObject* cap) {
    auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "sd_dlpack_owner"));
    if (m && m->deleter) m->deleter(m);
}

nb::object from_dlpack(nb::handle x, nb::handle device, nb::handle copy_obj) {
    const std::optional<bool> copy = parse_copy(copy_obj);
    if (!device.is_none() && !cpu_device(device)) throw nb::value_error("'device' must be None or the CPU device (1, 0)");
    if (!nb::hasattr(x, "__dlpack__"))
        throw nb::type_error(("'x' must support the DLPack protocol (__dlpack__), not " + py_type_name(x)).c_str());
    if (nb::hasattr(x, "__dlpack_device__")) {
        const nb::object dev = x.attr("__dlpack_device__")();
        if (!cpu_device(dev)) throw nb::buffer_error("only CPU arrays can be imported");
    }
    nb::object cap;
    try {
        nb::object c = copy ? nb::object(nb::bool_(*copy)) : nb::object(nb::none());
        cap = x.attr("__dlpack__")(nb::arg("max_version") = nb::make_tuple(kMajor, kMinor), nb::arg("copy") = c);
    } catch (nb::python_error& err) {
        if (!err.matches(PyExc_TypeError)) throw;
        cap = x.attr("__dlpack__")();  // a producer from before DLPack 1.0 takes no keywords
    }

    // Take ownership first, so every error below still frees the tensor.
    DLTensor* t = nullptr;
    bool read_only = false, copied = false;
    nb::object owner;
    if (PyCapsule_IsValid(cap.ptr(), "dltensor_versioned")) {
        auto* m = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(cap.ptr(), "dltensor_versioned"));
        PyCapsule_SetName(cap.ptr(), "used_dltensor_versioned");
        PyObject* own = PyCapsule_New(m, "sd_dlpack_owner", &owner_versioned);
        if (!own) {  // we own the tensor now: free it before failing
            m->deleter(m);
            throw nb::python_error();
        }
        owner = nb::steal(own);
        if (m->version.major > kMajor)
            throw nb::buffer_error(("DLPack version " + std::to_string(m->version.major) + "." +
                                    std::to_string(m->version.minor) + " is newer than the supported 1.x")
                                       .c_str());
        t = &m->dl_tensor;
        read_only = m->flags & kFlagReadOnly;
        copied = m->flags & kFlagIsCopied;  // the producer already made the copy= asked for
    } else if (PyCapsule_IsValid(cap.ptr(), "dltensor")) {
        auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), "dltensor"));
        PyCapsule_SetName(cap.ptr(), "used_dltensor");
        PyObject* own = PyCapsule_New(m, "sd_dlpack_owner", &owner_legacy);
        if (!own) {
            m->deleter(m);
            throw nb::python_error();
        }
        owner = nb::steal(own);
        t = &m->dl_tensor;
    } else {
        throw nb::type_error("__dlpack__ did not return a DLPack capsule");
    }

    if (t->device.device_type != kDLCPU) throw nb::buffer_error("only CPU arrays can be imported");
    const std::optional<DType> d = from_dl(t->dtype);
    if (!d) throw nb::type_error(("DLPack dtype " + dl_name(t->dtype) + " is not an array dtype (bool, int64, float64, complex128)").c_str());
    if (t->ndim == 0) throw nb::value_error("0-dimensional arrays are not supported (stardust has no 0-d arrays)");
    if (t->ndim > kMaxNdim)
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());

    NDArray a;
    a.dtype = *d;
    a.ndim = t->ndim;
    std::int64_t stride = 1;
    for (int k = t->ndim - 1; k >= 0; --k) {
        a.shape[k] = t->shape[k];
        a.strides[k] = (t->strides ? t->strides[k] : stride) * a.itemsize();
        stride *= t->shape[k];
    }
    a.data = static_cast<std::byte*>(t->data) + t->byte_offset;
    a.writeable = !read_only;
    a.base = owner;
    a.update_flags();

    // As with the buffer protocol (§3.3): shared only when naturally aligned,
    // and a bool tensor only when every byte is 0 or 1. Otherwise a copy, made
    // byte-wise in C order (bool bytes normalized to 0/1).
    const auto align = static_cast<std::uintptr_t>(a.itemsize() == 16 ? 8 : a.itemsize());
    bool aligned = reinterpret_cast<std::uintptr_t>(a.data) % align == 0;
    for (int k = 0; k < a.ndim; ++k) aligned = aligned && a.strides[k] % static_cast<std::int64_t>(align) == 0;
    auto for_each = [&](auto&& fn) {  // fn(i, element pointer), i in C order
        std::vector<std::int64_t> at(a.ndim, 0);
        for (std::int64_t i = 0, n = a.size(); i < n; ++i) {
            std::int64_t off = 0;
            for (int k = 0; k < a.ndim; ++k) off += at[k] * a.strides[k];
            fn(i, a.data + off);
            for (int k = a.ndim - 1; k >= 0; --k) {
                if (++at[k] < a.shape[k]) break;
                at[k] = 0;
            }
        }
    };
    bool valid = true;
    if (a.dtype == DType::Bool) {
        std::uint8_t any = 0;
        for_each([&](std::int64_t, const std::byte* p) { any |= static_cast<std::uint8_t>(*p); });
        valid = any <= 1;
    }
    if (!aligned || !valid) {
        if (copy == false)
            throw nb::value_error(valid ? "copy=False, but this tensor is not aligned and can only be copied"
                                        : "copy=False, but this bool tensor holds bytes other than 0 and 1");
        NDArray c = new_array(a.dtype, a.shape_span());
        const std::int64_t item = a.itemsize();
        for_each([&](std::int64_t i, const std::byte* p) {
            if (a.dtype == DType::Bool) reinterpret_cast<std::uint8_t*>(c.data)[i] = *p != std::byte{0};
            else std::memcpy(c.data + i * item, p, static_cast<size_t>(item));
        });
        return wrap(std::move(c));
    }
    if (copy == true && !copied) return wrap(cast_copy(a, a.dtype));
    return wrap(std::move(a));
}

}  // namespace

void bind_dlpack(nb::module_& m, nb::class_<NDArray>& cls) {
    using nb::arg;
    m.def("from_dlpack", named("from_dlpack", &from_dlpack), arg().none(), nb::kw_only(), arg("device").none() = nb::none(),
          arg("copy").none() = nb::none());
    cls.def("__dlpack__", named("__dlpack__", &dlpack), nb::kw_only(), arg("stream").none() = nb::none(),
            arg("max_version").none() = nb::none(), arg("dl_device").none() = nb::none(),
            arg("copy").none() = nb::none());
    cls.def("__dlpack_device__", [](const NDArray&) { return nb::make_tuple(kDLCPU, 0); });
}

}  // namespace sd
