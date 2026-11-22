// DLPack (docs/api/array.md §3.5; T0b): from_dlpack, and ndarray.__dlpack__ /
// __dlpack_device__. CPU only, the four array dtypes only.
//
// The structs are the DLPack 1.0 ABI (dlpack.h, Apache-2.0), written out
// here: the capsule "dltensor_versioned" holds a DLManagedTensorVersioned,
// the legacy capsule "dltensor" a DLManagedTensor. A consumer renames the
// capsule to "used_..." and calls the deleter when it is done.

#include "array/ops/dlpack.hpp"
#include "array/ops/creation.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/ops/copy.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <climits>
#include <cstring>
#include <optional>
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

// A device: None, or a (device_type, device_id) tuple of two ints (the int
// rule of index_value, so no bool). nullopt when it has another form; `bad`
// is then what to name in the message.
std::optional<bool> cpu_device(nb::handle dev, std::string& bad) {
    if (dev.is_none()) return true;
    if (!PyTuple_Check(dev.ptr()) || PyTuple_GET_SIZE(dev.ptr()) != 2) {
        bad = py_type_name(dev);
        return std::nullopt;
    }
    std::int64_t v[2];
    for (Py_ssize_t i = 0; i < 2; ++i) {
        nb::handle item = PyTuple_GET_ITEM(dev.ptr(), i);
        switch (index_value(item, v[i])) {
            case IndexResult::Ok: break;
            case IndexResult::Overflow: return false;  // an int, just not the CPU's
            case IndexResult::NotInt: bad = "a tuple holding " + py_type_name(item); return std::nullopt;
        }
    }
    return v[0] == kDLCPU && v[1] == 0;
}

nb::object dlpack(
    nb::handle self, nb::handle stream, nb::handle max_version, nb::handle dl_device, nb::handle copy_obj
) {
    const NDArray& a = as_ndarray(self);
    if (!stream.is_none()) throw nb::buffer_error("'stream' must be None for CPU arrays");
    std::string bad;
    const std::optional<bool> cpu = cpu_device(dl_device, bad);
    if (!cpu)
        throw nb::type_error(
            ("'dl_device' must be None or a (device_type, device_id) tuple of ints, not " + bad).c_str()
        );
    if (!*cpu) throw nb::buffer_error("only the CPU device (1, 0) is supported");
    const std::optional<bool> copy = parse_copy(copy_obj);
    bool versioned = false;
    if (!max_version.is_none()) {
        const char* kVersion = "'max_version' must be None or a (major, minor) tuple of ints, not ";
        if (!PyTuple_Check(max_version.ptr()) || PyTuple_GET_SIZE(max_version.ptr()) != 2)
            throw nb::type_error((kVersion + py_type_name(max_version)).c_str());
        std::int64_t v[2];
        for (Py_ssize_t i = 0; i < 2; ++i) {
            nb::handle item = PyTuple_GET_ITEM(max_version.ptr(), i);
            switch (index_value(item, v[i])) {
                case IndexResult::Ok: break;
                case IndexResult::Overflow: v[i] = INT64_MAX; break;  // a version past any we know
                case IndexResult::NotInt:
                    throw nb::type_error((kVersion + ("a tuple holding " + py_type_name(item))).c_str());
            }
        }
        versioned = v[0] >= 1;
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
        throw nb::buffer_error(
            "cannot export a read-only array with the legacy DLPack format; pass max_version=(1, 0)"
        );
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
    {
        std::string bad;
        const std::optional<bool> cpu = cpu_device(device, bad);
        if (!cpu) throw nb::type_error(("'device' must be None or the CPU device (1, 0), not " + bad).c_str());
        if (!*cpu) throw nb::value_error("'device' must be None or the CPU device (1, 0)");
    }
    // A numpy array: the view its __dlpack__ would describe, without the two calls
    // and the capsules (240 ns against numpy's own from_dlpack at 114).
    if (nb::object r = share_numpy_array(x, std::nullopt, copy); r.is_valid()) return r;
    // Each attribute looked up once, the names, the keyword names and the version
    // tuple made once, and the call a vectorcall (the generic attribute and keyword
    // path: 252 ns per import of a numpy array, numpy's own from_dlpack 114).
    static PyObject* const s_dlpack = PyUnicode_InternFromString("__dlpack__");
    static PyObject* const s_device = PyUnicode_InternFromString("__dlpack_device__");
    static PyObject* const s_kwnames = Py_BuildValue("(ss)", "max_version", "copy");
    static PyObject* const s_version = Py_BuildValue("(ii)", kMajor, kMinor);
    PyObject* raw = nullptr;
    if (PyObject_GetOptionalAttr(x.ptr(), s_dlpack, &raw) < 0) throw nb::python_error();
    if (!raw)
        throw nb::type_error(("'x' must support the DLPack protocol (__dlpack__), not " + py_type_name(x)).c_str());
    const nb::object method = nb::steal(raw);
    if (PyObject_GetOptionalAttr(x.ptr(), s_device, &raw) < 0) throw nb::python_error();
    if (raw) {
        const nb::object dev_method = nb::steal(raw);
        PyObject* dev = PyObject_CallNoArgs(dev_method.ptr());
        if (!dev) throw nb::python_error();
        std::string bad;
        const std::optional<bool> cpu = cpu_device(nb::steal(dev), bad);
        if (!cpu)
            throw nb::type_error(
                ("__dlpack_device__ must return a (device_type, device_id) tuple of ints, not " + bad).c_str()
            );
        if (!*cpu) throw nb::buffer_error("only CPU arrays can be imported");
    }
    nb::object cap;
    {
        PyObject* args[] = {s_version, copy ? (*copy ? Py_True : Py_False) : Py_None};
        PyObject* c = PyObject_Vectorcall(method.ptr(), args, 0, s_kwnames);
        if (c) {
            cap = nb::steal(c);
        } else {
            if (!PyErr_ExceptionMatches(PyExc_TypeError)) throw nb::python_error();
            PyErr_Clear();
            PyObject* c0 = PyObject_CallNoArgs(method.ptr());  // a producer from before DLPack 1.0 takes no keywords
            if (!c0) throw nb::python_error();
            cap = nb::steal(c0);
        }
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
    if (!d)
        throw nb::type_error(
            ("DLPack dtype " + dl_name(t->dtype) + " is not an array dtype (bool, int64, float64, complex128)").c_str()
        );
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
        std::int64_t i = 0;
        for_each_element<1>(a.shape_span(), {a.data}, {a.strides.data()}, [&](std::byte* p) { fn(i++, p); });
    };
    bool valid = true;
    if (a.dtype == DType::Bool) {
        // The OR of all bytes is at most 1 exactly when every byte is 0 or 1
        // (as for a buffer, §3.3): a flat branch-free scan when contiguous.
        // Bool is one byte, so its threshold is 8 times the array's.
        const std::int64_t n = a.size();
        std::uint8_t any = 0;
        const parallel::Unlocked unlocked(n >= 8 * kParallelThreshold);
        if (a.c_contiguous) {
            const auto* q = reinterpret_cast<const std::uint8_t*>(a.data);
            for (std::int64_t i = 0; i < n; ++i) any |= q[i];
        } else {
            for_each([&](std::int64_t, const std::byte* p) { any |= static_cast<std::uint8_t>(*p); });
        }
        valid = any <= 1;
    }
    if (!aligned || !valid) {
        if (copy == false)
            throw nb::value_error(
                valid ? "copy=False, but this tensor is not aligned and can only be copied"
                      : "copy=False, but this bool tensor holds bytes other than 0 and 1"
            );
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
    m.def(
        "from_dlpack", named("from_dlpack", &from_dlpack), nb::sig("def from_dlpack(x, /, *, device=None, copy=None)"),
        arg().none(), nb::kw_only(), arg("device").none() = nb::none(), arg("copy").none() = nb::none()
    );
    cls.def(
        "__dlpack__", named("__dlpack__", &dlpack), nb::kw_only(), arg("stream").none() = nb::none(),
        arg("max_version").none() = nb::none(), arg("dl_device").none() = nb::none(), arg("copy").none() = nb::none()
    );
    cls.def("__dlpack_device__", [](const NDArray&) { return nb::make_tuple(kDLCPU, 0); });
}

}  // namespace sd
