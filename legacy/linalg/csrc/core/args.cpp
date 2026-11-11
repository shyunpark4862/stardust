#include "core/args.hpp"

#include "core/pyrefs.hpp"

#include <algorithm>
#include <string>

namespace sdla {

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
    for (int i = 0; i < 4; ++i)
        if (o.ptr() == refs().dtypes[i]) return static_cast<DType>(i);
    throw nb::type_error((std::string(func) +
                          "(): 'dtype' must be sdnp.bool, sdnp.int64, sdnp.float64 or sdnp.complex128, not " +
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

}  // namespace sdla
