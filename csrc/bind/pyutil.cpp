#include "bind/pyutil.hpp"

#include <stdexcept>
#include <string>

namespace sd {

namespace {

// Borrowed: the module attributes (sd.bool, ...) own the five objects, and
// nothing in _core can run after the module itself is torn down.
PyObject* g_dtype_objects[kNumDTypes] = {};

}  // namespace

void register_dtypes(nb::module_& m) {
    nb::class_<PyDType>(m, "DType")
        .def("__repr__", [](const PyDType& d) { return dtype_name(d.value); })
        .def("__str__", [](const PyDType& d) { return dtype_name(d.value); });

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
    if (PyUnicode_Check(o.ptr()) || PyType_Check(o.ptr())) return nb::repr(o).c_str();
    return Py_TYPE(o.ptr())->tp_name;
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

std::optional<bool> parse_copy(nb::handle o) {
    if (o.is_none()) return std::nullopt;
    if (o.ptr() == Py_True) return true;
    if (o.ptr() == Py_False) return false;
    throw nb::type_error("'copy' must be True, False or None");
}

std::int64_t parse_int(nb::handle o, const char* what) {
    if (!PyLong_Check(o.ptr()) || PyBool_Check(o.ptr()))
        throw nb::type_error((std::string("'") + what + "' must be an int, not " + Py_TYPE(o.ptr())->tp_name).c_str());
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(o.ptr(), &overflow);
    if (overflow != 0)  // an int that must become an int64 (API.md §2)
        throw std::overflow_error(std::string("'") + what + "' is out of range");
    return v;
}

int normalize_axis(std::int64_t axis, int ndim) {
    if (axis < -ndim || axis >= ndim)
        throw nb::value_error(("axis " + std::to_string(axis) + " is out of bounds for an array of dimension " +
                               std::to_string(ndim)).c_str());
    return static_cast<int>(axis < 0 ? axis + ndim : axis);
}

std::vector<int> parse_axes(nb::handle o, int ndim, const char* what) {
    // An int, or a tuple or list of ints, as in sd.linalg (docs/api/array.md §1.3).
    auto as_int = [&](nb::handle item) {
        if (!PyLong_Check(item.ptr()) || PyBool_Check(item.ptr()))
            throw nb::type_error((std::string("'") + what + "' must be an int or a tuple or list of ints, not " +
                                  Py_TYPE(item.ptr())->tp_name)
                                     .c_str());
        return normalize_axis(parse_int(item, what), ndim);
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

std::vector<std::int64_t> parse_shape(nb::handle o) {
    if (PyLong_Check(o.ptr()) && !PyBool_Check(o.ptr())) return {parse_int(o, "shape")};
    if (!PyTuple_Check(o.ptr()))
        throw nb::type_error(("'shape' must be an int or a tuple of ints, not " + py_type_name(o)).c_str());
    std::vector<std::int64_t> shape;
    for (nb::handle item : nb::borrow<nb::tuple>(o)) {
        if (!PyLong_Check(item.ptr()) || PyBool_Check(item.ptr()))
            throw nb::type_error((std::string("'shape' must be an int or a tuple of ints, not a tuple holding ") +
                                  Py_TYPE(item.ptr())->tp_name)
                                     .c_str());
        shape.push_back(parse_int(item, "shape"));
    }
    return shape;
}

}  // namespace sd
