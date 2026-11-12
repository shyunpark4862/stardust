#include "linalg/args.hpp"

#include "bind/pyutil.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace sd::linalg {

std::int64_t parse_int(nb::handle o, const char* func, const char* what) {
    if (!PyLong_Check(o.ptr()) || PyBool_Check(o.ptr()))
        throw nb::type_error(
            (std::string(func) + "(): '" + what + "' must be an int, not " + Py_TYPE(o.ptr())->tp_name).c_str());
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(o.ptr(), &overflow);
    if (overflow) throw nb::value_error((std::string(func) + "(): '" + what + "' is out of range").c_str());
    return v;
}

std::optional<DType> parse_dtype(nb::handle o, const char* func) {
    if (o.is_none()) return std::nullopt;
    for (int i = 0; i < kNumArrayDTypes; ++i)
        if (o.ptr() == dtype_object(static_cast<DType>(i)).ptr()) return static_cast<DType>(i);
    throw nb::type_error((std::string(func) +
                          "(): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128, not " +
                          Py_TYPE(o.ptr())->tp_name)
                             .c_str());
}

int normalize_axis(std::int64_t axis, int ndim, const char* func) {
    if (axis < -ndim || axis >= ndim)
        throw nb::value_error((std::string(func) + "(): axis " + std::to_string(axis) +
                               " is out of bounds for an array of dimension " + std::to_string(ndim))
                                  .c_str());
    return static_cast<int>(axis < 0 ? axis + ndim : axis);
}

std::vector<int> parse_axes(nb::handle o, int ndim, const char* func) {
    std::vector<int> axes;
    if (o.is_none()) {
        for (int d = 0; d < ndim; ++d) axes.push_back(d);
        return axes;
    }
    if (PyTuple_Check(o.ptr())) {
        for (nb::handle item : nb::borrow<nb::tuple>(o)) axes.push_back(normalize_axis(parse_int(item, func, "axis"), ndim, func));
    } else {
        axes.push_back(normalize_axis(parse_int(o, func, "axis"), ndim, func));
    }
    std::sort(axes.begin(), axes.end());
    if (std::adjacent_find(axes.begin(), axes.end()) != axes.end())
        throw nb::value_error((std::string(func) + "(): repeated axis").c_str());
    return axes;
}

// `rtol=` (docs/api/linalg.md §1.1): None -> max(m, n) * eps; else a real number >= 0.
double parse_rtol(nb::handle o, std::int64_t m, std::int64_t n, const char* func) {
    if (o.is_none()) return static_cast<double>(std::max<std::int64_t>({m, n, 1})) * 2.220446049250313e-16;
    if (PyBool_Check(o.ptr()) || !(PyLong_Check(o.ptr()) || PyFloat_Check(o.ptr())))
        throw nb::type_error((std::string(func) + "(): 'rtol' must be None, an int or a float").c_str());
    const double v = PyFloat_AsDouble(o.ptr());
    if (v == -1.0 && PyErr_Occurred()) throw nb::python_error();
    if (!(v >= 0) || std::isinf(v))
        throw nb::value_error((std::string(func) + "(): 'rtol' must be a finite number >= 0").c_str());
    return v;
}

}  // namespace sd::linalg
