#include "linalg/args.hpp"

#include "bind/pyutil.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace sd::linalg {

std::int64_t parse_int(nb::handle o, const char* func, const char* what) {
    std::int64_t v = 0;
    switch (index_value(o, v)) {  // operator.index semantics, as in the array area (bind/pyutil.hpp)
        case IndexResult::Ok: return v;
        case IndexResult::Overflow:  // an int that must become an int64 (API.md §2)
            throw std::overflow_error(std::string(func) + "(): '" + what + "' is out of range");
        case IndexResult::NotInt: break;
    }
    throw nb::type_error((std::string(func) + "(): '" + what + "' must be an int, not " + py_type_name(o)).c_str());
}

std::optional<DType> parse_dtype(nb::handle o, const char* func) {
    if (o.is_none()) return std::nullopt;
    if (const std::optional<DType> d = parse_any_dtype(o); d && is_array_dtype(*d)) return d;
    throw nb::type_error((std::string(func) +
                          "(): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128 (or their names), not " +
                          describe_dtype_arg(o))
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
    auto one = [&](nb::handle item) {  // the array area's message (bind/pyutil.cpp parse_axes)
        std::int64_t v = 0;
        switch (index_value(item, v)) {
            case IndexResult::Ok: return normalize_axis(v, ndim, func);
            case IndexResult::Overflow: throw std::overflow_error(std::string(func) + "(): 'axis' is out of range");
            case IndexResult::NotInt: break;
        }
        throw nb::type_error((std::string(func) + "(): 'axis' must be an int or a tuple or list of ints, not " +
                              py_type_name(item))
                                 .c_str());
    };
    if (PyTuple_Check(o.ptr()) || PyList_Check(o.ptr())) {  // a list like a tuple (docs/api/linalg.md §3)
        nb::object seq = nb::steal(PySequence_Tuple(o.ptr()));
        if (!seq.is_valid()) throw nb::python_error();
        for (nb::handle item : nb::borrow<nb::tuple>(seq)) axes.push_back(one(item));
    } else {
        axes.push_back(one(o));
    }
    std::sort(axes.begin(), axes.end());
    if (std::adjacent_find(axes.begin(), axes.end()) != axes.end())
        throw nb::value_error((std::string(func) + "(): repeated axis").c_str());
    return axes;
}

// `rtol=` (docs/api/linalg.md §1.1): None -> max(m, n) * eps; else a real number >= 0.
double parse_rtol(nb::handle o, std::int64_t m, std::int64_t n, const char* func) {
    if (o.is_none()) return static_cast<double>(std::max<std::int64_t>({m, n, 1})) * 2.220446049250313e-16;
    const std::optional<double> r = real_value(o, "rtol");
    if (!r)
        throw nb::type_error(
            (std::string(func) + "(): 'rtol' must be None, an int or a float, not " + py_type_name(o)).c_str());
    const double v = *r;
    if (!(v >= 0) || std::isinf(v))
        throw nb::value_error((std::string(func) + "(): 'rtol' must be a finite number >= 0").c_str());
    return v;
}

}  // namespace sd::linalg
