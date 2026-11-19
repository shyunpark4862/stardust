#include "bind/pyutil.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace sd {

namespace {

// Borrowed: the module attributes (sd.bool, ...) own the five objects, and
// nothing in _core can run after the module itself is torn down.
PyObject* g_dtype_objects[kNumDTypes] = {};

}  // namespace

void register_dtypes(nb::module_& m) {
    auto cls = nb::class_<PyDType>(m, "DType")
                   .def("__repr__", [](const PyDType& d) { return dtype_name(d.value); })
                   .def("__str__", [](const PyDType& d) { return dtype_name(d.value); });
    use_short_type_name(cls);

    for (DType d : {DType::Bool, DType::Int64, DType::Float64, DType::Complex128, DType::String}) {
        nb::object obj = nb::cast(PyDType{d});
        m.attr(dtype_name(d)) = obj;
        g_dtype_objects[static_cast<int>(d)] = obj.ptr();
    }
}

nb::handle dtype_object(DType d) { return g_dtype_objects[static_cast<int>(d)]; }

std::optional<DType> parse_any_dtype(nb::handle o) {
    if (o.is_none()) return std::nullopt;
    for (int i = 0; i < kNumDTypes; ++i)
        if (o.ptr() == g_dtype_objects[i]) return static_cast<DType>(i);
    if (PyUnicode_Check(o.ptr())) {
        Py_ssize_t len = 0;
        const char* s = PyUnicode_AsUTF8AndSize(o.ptr(), &len);
        if (s == nullptr) throw nb::python_error();
        if (auto d = dtype_from_name(std::string_view(s, static_cast<std::size_t>(len)))) return d;
    }
    return std::nullopt;  // the caller raises, naming what it accepts
}

std::string describe_dtype_arg(nb::handle o) {
    if (const std::optional<DType> d = parse_any_dtype(o); d && !PyUnicode_Check(o.ptr()))
        return std::string("sd.") + dtype_name(*d);  // a dtype the area does not have: sd.string
    if (PyUnicode_Check(o.ptr())) return nb::repr(o).c_str();
    if (PyType_Check(o.ptr())) {  // a type by its full name (numpy.int64: not to be read as sd.int64); builtins bare
        const std::string module = nb::cast<std::string>(nb::str(nb::getattr(o, "__module__", nb::str("builtins"))));
        const std::string name = nb::cast<std::string>(nb::str(nb::getattr(o, "__qualname__")));
        return module == "builtins" ? name : module + "." + name;
    }
    return py_type_name(o);
}

void use_short_type_name(nb::handle type) {
    auto* t = reinterpret_cast<PyTypeObject*>(type.ptr());
    // tp_name points into the type's own copy of its full name (freed with
    // the type), so pointing past the last dot keeps that ownership.
    if (const char* dot = std::strrchr(t->tp_name, '.')) t->tp_name = dot + 1;
}

bool parse_bool(nb::handle o, const char* what) {
    if (o.ptr() == Py_True) return true;
    if (o.ptr() == Py_False) return false;
    throw nb::type_error((std::string("'") + what + "' must be True or False, not " + py_type_name(o)).c_str());
}

IndexResult index_value(nb::handle o, std::int64_t& out) {
    PyObject* p = o.ptr();
    if (PyBool_Check(p)) return IndexResult::NotInt;  // bool is an int subclass, but not an int argument
    nb::object owned;
    if (!PyLong_Check(p)) {
        // sd.ndarray has __index__ only to refuse it with its own message.
        if (!PyIndex_Check(p) || is_ndarray(o)) return IndexResult::NotInt;
        PyObject* r = PyNumber_Index(p);  // numpy.int64 -> int; numpy.bool_ refuses
        if (r == nullptr) {
            if (!PyErr_ExceptionMatches(PyExc_TypeError)) throw nb::python_error();
            PyErr_Clear();
            return IndexResult::NotInt;
        }
        owned = nb::steal(r);
        p = r;
    }
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(p, &overflow);
    if (overflow != 0) return IndexResult::Overflow;
    if (v == -1 && PyErr_Occurred()) throw nb::python_error();
    out = v;
    return IndexResult::Ok;
}

namespace {

// PyFloat_AsDouble, with Python's bare OverflowError for an int beyond the double range
// turned into ours: "'what' is out of range" as for any other argument out of range (the
// caller's wrapper adds "func(): ").
double as_double(PyObject* p, const char* what) {
    const double v = PyFloat_AsDouble(p);
    if (v == -1.0 && PyErr_Occurred()) {
        if (!PyErr_ExceptionMatches(PyExc_OverflowError)) throw nb::python_error();
        PyErr_Clear();
        if (what == nullptr) throw std::overflow_error("Python int too large to convert to float64");
        throw std::overflow_error(std::string("'") + what + "' is out of range");
    }
    return v;
}

}  // namespace

std::optional<double> real_value(nb::handle o, const char* what) {
    PyObject* p = o.ptr();
    if (PyFloat_Check(p) || (PyLong_Check(p) && !PyBool_Check(p))) return as_double(p, what);
    std::int64_t i = 0;
    switch (index_value(o, i)) {
        case IndexResult::Ok: return static_cast<double>(i);
        case IndexResult::Overflow: {  // a numpy uint64 beyond int64: still a number
            nb::object r = nb::steal(PyNumber_Index(p));
            if (!r.is_valid()) throw nb::python_error();
            return as_double(r.ptr(), what);
        }
        case IndexResult::NotInt: return std::nullopt;
    }
    __builtin_unreachable();
}

std::optional<DType> parse_dtype(nb::handle o) {
    if (o.is_none()) return std::nullopt;
    const std::optional<DType> d = parse_any_dtype(o);
    if (!d || !is_array_dtype(*d))
        throw nb::type_error(("'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128 (or their names), not " +
                              describe_dtype_arg(o))
                                 .c_str());
    return d;
}

std::string parse_choice(nb::handle o, const char* what, std::span<const char* const> choices) {
    std::string listed;
    size_t i = 0;
    for (const char* c : choices) {
        if (i > 0) listed += i + 1 == choices.size() ? " or " : ", ";
        listed += std::string("'") + c + "'";
        ++i;
    }
    const std::string head = std::string("'") + what + "' must be " + listed + ", not ";
    if (!nb::isinstance<nb::str>(o)) throw nb::type_error((head + py_type_name(o)).c_str());
    const std::string s = nb::borrow<nb::str>(o).c_str();
    for (const char* c : choices)
        if (s == c) return s;
    throw nb::value_error((head + nb::repr(o).c_str()).c_str());
}

std::optional<bool> parse_copy(nb::handle o) {
    if (o.is_none()) return std::nullopt;
    if (o.ptr() == Py_True) return true;
    if (o.ptr() == Py_False) return false;
    throw nb::type_error(("'copy' must be True, False or None, not " + py_type_name(o)).c_str());
}

std::int64_t parse_int(nb::handle o, const char* what) {
    std::int64_t v = 0;
    switch (index_value(o, v)) {
        case IndexResult::Ok: return v;
        case IndexResult::Overflow:  // an int that must become an int64 (API.md §2)
            throw std::overflow_error(std::string("'") + what + "' is out of range");
        case IndexResult::NotInt: break;
    }
    throw nb::type_error((std::string("'") + what + "' must be an int, not " + py_type_name(o)).c_str());
}

int normalize_axis(std::int64_t axis, int ndim) {
    if (axis < -ndim || axis >= ndim)
        throw nb::value_error(("axis " + std::to_string(axis) + " is out of bounds for an array of dimension " +
                               std::to_string(ndim))
                                  .c_str());
    return static_cast<int>(axis < 0 ? axis + ndim : axis);
}

std::vector<int> parse_axes(nb::handle o, int ndim, const char* what) {
    // An int, or a tuple or list of ints, as in sd.linalg (docs/api/array.md §1.3).
    auto as_int = [&](nb::handle item) {
        std::int64_t v = 0;
        switch (index_value(item, v)) {
            case IndexResult::Ok: return normalize_axis(v, ndim);
            case IndexResult::Overflow: throw std::overflow_error(std::string("'") + what + "' is out of range");
            case IndexResult::NotInt: break;
        }
        throw nb::type_error(
            (std::string("'") + what + "' must be an int or a tuple or list of ints, not " + py_type_name(item)).c_str()
        );
    };
    std::vector<int> axes;
    if (PyTuple_Check(o.ptr()) || PyList_Check(o.ptr())) {
        for (nb::handle item : o) axes.push_back(as_int(item));
    } else {
        axes.push_back(as_int(o));
    }
    for (size_t i = 0; i < axes.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (axes[i] == axes[j]) throw nb::value_error("repeated axis");
    return axes;
}

NDArray& require_ndarray(nb::handle o, const char* what) {
    if (!is_ndarray(o))
        throw nb::type_error((std::string("'") + what + "' must be an sd.ndarray, not " + py_type_name(o)).c_str());
    return as_ndarray(o);
}

NDArray view_of(nb::handle x) {
    const NDArray& a = as_ndarray(x);
    NDArray v = a;
    v.base = a.base.is_valid() ? a.base : nb::borrow(x);
    return v;
}

ShapeArg parse_shape_arg(nb::handle o) {
    ShapeArg s;
    std::int64_t v = 0;
    if (!PyTuple_Check(o.ptr())) {
        if (index_value(o, v) == IndexResult::Ok) {
            s.small[0] = v, s.n = 1;
            return s;
        }
    } else if (PyTuple_GET_SIZE(o.ptr()) <= kMaxNdim) {
        const Py_ssize_t m = PyTuple_GET_SIZE(o.ptr());
        bool ok = true;
        for (Py_ssize_t i = 0; i < m && ok; ++i) {
            ok = index_value(PyTuple_GET_ITEM(o.ptr(), i), v) == IndexResult::Ok;
            s.small[static_cast<std::size_t>(i)] = v;
        }
        if (ok) {
            s.n = static_cast<std::size_t>(m);
            return s;
        }
    }
    s.big = parse_shape(o);  // anything unusual, errors included: the general parser
    s.n = s.big.size();
    return s;
}

std::vector<std::int64_t> parse_shape(nb::handle o) {
    auto one = [](nb::handle item, std::int64_t& v) {  // false: not an int
        switch (index_value(item, v)) {
            case IndexResult::Ok: return true;
            case IndexResult::Overflow: throw std::overflow_error("'shape' is out of range");
            case IndexResult::NotInt: return false;
        }
        __builtin_unreachable();
    };
    std::int64_t v = 0;
    if (!PyTuple_Check(o.ptr())) {
        if (one(o, v)) return {v};
        throw nb::type_error(("'shape' must be an int or a tuple of ints, not " + py_type_name(o)).c_str());
    }
    std::vector<std::int64_t> shape;
    for (nb::handle item : nb::borrow<nb::tuple>(o)) {
        if (!one(item, v))
            throw nb::type_error(
                ("'shape' must be an int or a tuple of ints, not a tuple holding " + py_type_name(item)).c_str()
            );
        shape.push_back(v);
    }
    return shape;
}

}  // namespace sd
