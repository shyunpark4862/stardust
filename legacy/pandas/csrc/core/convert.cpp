#include "core/convert.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sd {

namespace {

// What a Python value is, for dtype inference.
enum class Kind : std::uint8_t { NA, Bool, Int, Float, Str };

std::string type_name(PyObject* o) { return Py_TYPE(o)->tp_name; }

Kind classify(PyObject* o, PyObject* na) {
    if (o == Py_None || o == na) return Kind::NA;
    if (PyBool_Check(o)) return Kind::Bool;  // before int: bool is an int subclass
    if (PyLong_Check(o)) return Kind::Int;
    if (PyFloat_Check(o)) return Kind::Float;  // includes numpy.float64
    if (PyUnicode_Check(o)) return Kind::Str;  // includes numpy.str_
    // Other numpy scalars (numpy.int32, numpy.bool, numpy.float32, ...).
    const char* name = Py_TYPE(o)->tp_name;
    if (std::strncmp(name, "numpy.", 6) == 0) {
        if (std::strcmp(name, "numpy.bool") == 0) return Kind::Bool;
        if (PyIndex_Check(o)) return Kind::Int;
        if (Py_TYPE(o)->tp_as_number && Py_TYPE(o)->tp_as_number->nb_float) return Kind::Float;
    }
    throw nb::type_error(("unsupported value of type '" + type_name(o) + "'").c_str());
}

std::int64_t int_value(PyObject* o) {
    nb::object index = nb::steal(PyNumber_Index(o));  // numpy integers -> int
    if (!index.is_valid()) throw nb::python_error();
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(index.ptr(), &overflow);
    if (overflow != 0) throw std::overflow_error("int " + std::string(nb::repr(index).c_str()) + " does not fit in int64");
    if (v == -1 && PyErr_Occurred()) throw nb::python_error();
    return v;
}

double float_value(PyObject* o) {
    double d = PyFloat_AsDouble(o);
    if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
    return d;
}

const char* kind_name(Kind k) {
    switch (k) {
    case Kind::NA: return "NA";
    case Kind::Bool: return "bool";
    case Kind::Int: return "int";
    case Kind::Float: return "float";
    case Kind::Str: return "str";
    }
    return "?";
}

[[noreturn]] void cannot_store(PyObject* o, DType target) {
    throw nb::type_error(
        ("cannot store a value of type '" + type_name(o) + "' in a column of dtype " + dtype_name(target)).c_str());
}

}  // namespace

Column from_list(nb::handle seq, nb::handle na, std::optional<DType> dtype) {
    nb::object fast = nb::steal(PySequence_Fast(seq.ptr(), "expected a sequence"));
    if (!fast.is_valid()) throw nb::python_error();
    const std::int64_t n = PySequence_Fast_GET_SIZE(fast.ptr());
    PyObject** items = PySequence_Fast_ITEMS(fast.ptr());

    // Pass 1: classify every value. Numbers are read here into `scratch`
    // (int64 or double bits), strings only measured.
    std::vector<Kind> kinds(static_cast<std::size_t>(n));
    std::vector<std::int64_t> scratch(static_cast<std::size_t>(n));
    bool seen[5] = {};
    std::int64_t nulls = 0, nbytes = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        PyObject* o = items[i];
        Kind k = classify(o, na.ptr());
        kinds[i] = k;
        seen[static_cast<int>(k)] = true;
        switch (k) {
        case Kind::NA: ++nulls; break;
        case Kind::Bool: scratch[i] = PyObject_IsTrue(o); break;
        case Kind::Int: scratch[i] = int_value(o); break;
        case Kind::Float: {
            double d = float_value(o);
            std::memcpy(&scratch[i], &d, sizeof d);
            break;
        }
        case Kind::Str: {
            Py_ssize_t size = 0;
            if (PyUnicode_AsUTF8AndSize(o, &size) == nullptr) throw nb::python_error();
            nbytes += size;
            break;
        }
        }
    }
    const bool b = seen[int(Kind::Bool)], in = seen[int(Kind::Int)], f = seen[int(Kind::Float)],
               s = seen[int(Kind::Str)];

    DType target;
    if (dtype) {
        target = *dtype;
    } else {
        // There is no object dtype, so bool and str do not mix with anything
        // (API.md §10). int with float is float64, checked for exactness below.
        auto mixed = [&](Kind k) {
            std::string others;
            for (Kind o : {Kind::Bool, Kind::Int, Kind::Float, Kind::Str})
                if (o != k && seen[int(o)]) others += std::string(others.empty() ? "" : ", ") + kind_name(o);
            throw nb::type_error((std::string(kind_name(k)) + " values cannot be mixed with " + others +
                                  " values (there is no object dtype)")
                                     .c_str());
        };
        if (b && (in || f || s)) mixed(Kind::Bool);
        if (s && (in || f)) mixed(Kind::Str);
        target = s ? DType::String : b ? DType::Bool : f ? DType::Float64 : in ? DType::Int64 : DType::Float64;
    }

    // Pass 2: fill the column.
    Column col;
    std::uint8_t* bits = nullptr;
    if (target == DType::String) {
        col.dtype = DType::String;
        col.length = n;
        col.values = Buffer::allocate(nbytes, /*zeroed=*/false);
        col.offsets = Buffer::allocate((n + 1) * 8, /*zeroed=*/false);
    } else {
        col = make_fixed(target, n);
    }
    if (nulls > 0) {
        bits = add_validity(col);
        col.null_count = nulls;
    }
    std::byte* out = col.values->data();
    std::int64_t* offs = col.offsets ? reinterpret_cast<std::int64_t*>(col.offsets->data()) : nullptr;
    std::int64_t pos = 0;
    if (offs) offs[0] = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        PyObject* o = items[i];
        const Kind k = kinds[i];
        if (k == Kind::NA) bitmap::clear(bits, i);
        switch (target) {
        case DType::Bool:
            if (k != Kind::NA && k != Kind::Bool) cannot_store(o, target);
            reinterpret_cast<std::uint8_t*>(out)[i] = static_cast<std::uint8_t>(k == Kind::NA ? 0 : scratch[i]);
            break;
        case DType::Int64: {
            std::int64_t v = 0;  // NA rows hold 0 (ARCHITECTURE.md §2.4)
            if (k == Kind::Int) {
                v = scratch[i];
            } else if (k == Kind::Float) {
                double d;
                std::memcpy(&d, &scratch[i], sizeof d);
                if (!double_is_exact_int(d))
                    throw nb::value_error(("float " + std::string(nb::repr(nb::handle(o)).c_str()) +
                                           " cannot be converted to int64 without loss")
                                              .c_str());
                v = static_cast<std::int64_t>(d);
            } else if (k != Kind::NA) {
                cannot_store(o, target);
            }
            reinterpret_cast<std::int64_t*>(out)[i] = v;
            break;
        }
        case DType::Float64: {
            double d = 0.0;
            if (k == Kind::Float) {
                std::memcpy(&d, &scratch[i], sizeof d);
            } else if (k == Kind::Int) {
                if (!int_is_exact_double(scratch[i]))
                    throw nb::value_error(("int " + std::to_string(scratch[i]) +
                                           " cannot be represented exactly as float64")
                                              .c_str());
                d = static_cast<double>(scratch[i]);
            } else if (k != Kind::NA) {
                cannot_store(o, target);
            }
            reinterpret_cast<double*>(out)[i] = d;
            break;
        }
        case DType::String:
            if (k == Kind::Str) {
                Py_ssize_t size = 0;
                const char* utf8 = PyUnicode_AsUTF8AndSize(o, &size);  // cached by pass 1
                std::memcpy(out + pos, utf8, static_cast<std::size_t>(size));
                pos += size;
            } else if (k != Kind::NA) {
                cannot_store(o, target);
            }
            offs[i + 1] = pos;  // NA rows are empty strings (ARCHITECTURE.md §2.4)
            break;
        }
    }
    return col;
}

namespace {

template <class To, class From>
void widen(To* out, const void* data, std::int64_t n, std::int64_t stride) {
    const From* in = static_cast<const From*>(data);
    for (std::int64_t i = 0; i < n; ++i) out[i] = static_cast<To>(in[i * stride]);
}

}  // namespace

Column from_numpy(const nb::ndarray<nb::ro, nb::ndim<1>, nb::device::cpu>& arr) {
    const auto n = static_cast<std::int64_t>(arr.shape(0));
    const std::int64_t stride = arr.stride(0);  // in elements
    const void* data = arr.data();
    const nb::dlpack::dtype dt = arr.dtype();
    using code = nb::dlpack::dtype_code;
    auto unsupported = [&]() -> Column {
        throw nb::type_error(("unsupported numpy dtype (code " + std::to_string(dt.code) + ", " +
                              std::to_string(dt.bits) + " bits)")
                                 .c_str());
    };
    if (dt.lanes != 1) return unsupported();

    if (dt.code == static_cast<std::uint8_t>(code::Bool) && dt.bits == 8) {
        Column col = make_fixed(DType::Bool, n);
        widen<std::uint8_t, std::uint8_t>(reinterpret_cast<std::uint8_t*>(col.values->data()), data, n, stride);
        return col;
    }
    if (dt.code == static_cast<std::uint8_t>(code::Int)) {
        Column col = make_fixed(DType::Int64, n);
        auto* out = reinterpret_cast<std::int64_t*>(col.values->data());
        switch (dt.bits) {
        case 8: widen<std::int64_t, std::int8_t>(out, data, n, stride); break;
        case 16: widen<std::int64_t, std::int16_t>(out, data, n, stride); break;
        case 32: widen<std::int64_t, std::int32_t>(out, data, n, stride); break;
        case 64: widen<std::int64_t, std::int64_t>(out, data, n, stride); break;
        default: return unsupported();
        }
        return col;
    }
    if (dt.code == static_cast<std::uint8_t>(code::UInt)) {
        Column col = make_fixed(DType::Int64, n);
        auto* out = reinterpret_cast<std::int64_t*>(col.values->data());
        switch (dt.bits) {
        case 8: widen<std::int64_t, std::uint8_t>(out, data, n, stride); break;
        case 16: widen<std::int64_t, std::uint16_t>(out, data, n, stride); break;
        case 32: widen<std::int64_t, std::uint32_t>(out, data, n, stride); break;
        case 64: {
            const auto* in = static_cast<const std::uint64_t*>(data);
            for (std::int64_t i = 0; i < n; ++i) {
                const std::uint64_t v = in[i * stride];
                if (v > static_cast<std::uint64_t>(INT64_MAX))
                    throw std::overflow_error("uint64 value " + std::to_string(v) + " does not fit in int64");
                out[i] = static_cast<std::int64_t>(v);
            }
            break;
        }
        default: return unsupported();
        }
        return col;
    }
    if (dt.code == static_cast<std::uint8_t>(code::Float)) {
        Column col = make_fixed(DType::Float64, n);
        auto* out = reinterpret_cast<double*>(col.values->data());
        switch (dt.bits) {
        case 16: widen<double, _Float16>(out, data, n, stride); break;
        case 32: widen<double, float>(out, data, n, stride); break;
        case 64: widen<double, double>(out, data, n, stride); break;
        default: return unsupported();
        }
        return col;
    }
    return unsupported();
}

nb::list to_list(const Column& col, nb::handle na) {
    PyObject* list = PyList_New(col.length);
    if (list == nullptr) throw nb::python_error();
    nb::list result = nb::steal<nb::list>(list);
    for (std::int64_t i = 0; i < col.length; ++i) {
        PyObject* item;
        if (!col.is_valid(i)) {
            item = Py_NewRef(na.ptr());
        } else {
            switch (col.dtype) {
            case DType::Bool: item = PyBool_FromLong(col.data<std::uint8_t>()[i]); break;
            case DType::Int64: item = PyLong_FromLongLong(col.data<std::int64_t>()[i]); break;
            case DType::Float64: item = PyFloat_FromDouble(col.data<double>()[i]); break;
            case DType::String: {
                std::string_view sv = col.string_at(i);
                item = PyUnicode_DecodeUTF8(sv.data(), static_cast<Py_ssize_t>(sv.size()), "strict");
                break;
            }
            default: item = nullptr;
            }
            if (item == nullptr) throw nb::python_error();
        }
        PyList_SET_ITEM(list, i, item);
    }
    return result;
}

namespace {

// A capsule that keeps a Buffer alive for as long as a numpy array uses it.
nb::capsule buffer_owner(const std::shared_ptr<Buffer>& buf) {
    auto* holder = new std::shared_ptr<Buffer>(buf);
    return nb::capsule(holder, [](void* p) noexcept { delete static_cast<std::shared_ptr<Buffer>*>(p); });
}

nb::dlpack::dtype numpy_dtype(DType dt) {
    switch (dt) {
    case DType::Bool: return nb::dtype<bool>();
    case DType::Int64: return nb::dtype<std::int64_t>();
    case DType::Float64: return nb::dtype<double>();
    case DType::String: break;
    }
    throw nb::type_error("a string column has no numpy view");
}

}  // namespace

nb::object to_numpy_view(const Column& col) {
    const nb::dlpack::dtype dt = numpy_dtype(col.dtype);
    const std::byte* ptr = col.values->data() + col.offset * item_size(col.dtype);
    return nb::ndarray<nb::numpy, nb::ro>(ptr, {static_cast<std::size_t>(col.length)}, buffer_owner(col.values), {}, dt)
        .cast();
}

nb::object validity_mask(const Column& col) {
    auto buf = Buffer::allocate(col.length, /*zeroed=*/false);
    auto* out = reinterpret_cast<std::uint8_t*>(buf->data());
    const std::vector<std::uint8_t> m = valid_bytes(col);
    std::memcpy(out, m.data(), m.size());
    return nb::ndarray<nb::numpy, bool>(out, {static_cast<std::size_t>(col.length)}, buffer_owner(buf)).cast();
}

}  // namespace sd
