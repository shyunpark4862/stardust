// The argument checks the transforms of sd.fft share (docs/api/fft.md §1; docs/design/fft.md §2.4, §2.5):
// norm, n, axis, s and axes with the messages of API.md §2.1, the normalisation factors and the result
// shapes. basic.cpp (the complex, real and Hermitian transforms) and r2r.cpp (the DCTs and DSTs, milestone
// FT4) use them; milestone FT4 moved them here from basic.cpp unchanged.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"

namespace sd::fft {

namespace nb = nanobind;

enum class Norm { Backward, Ortho, Forward };

inline Norm parse_norm(nb::handle o) {
    if (PyUnicode_Check(o.ptr())) {
        const char* s = PyUnicode_AsUTF8(o.ptr());
        if (s != nullptr) {
            if (std::strcmp(s, "backward") == 0) return Norm::Backward;
            if (std::strcmp(s, "ortho") == 0) return Norm::Ortho;
            if (std::strcmp(s, "forward") == 0) return Norm::Forward;
        } else {
            PyErr_Clear();
        }
    }
    parse_choice(o, "norm", {"backward", "ortho", "forward"});  // raises the error
    return Norm::Backward;
}

// The factor a transform of length n is multiplied by (docs/design/fft.md §2.5): 1, 1/n, or
// fl(1/fl(sqrt(n))).
inline double norm_factor(Norm norm, int64_t n, bool inverse) {
    const double dn = static_cast<double>(n);
    switch (norm) {
        case Norm::Ortho: return 1.0 / std::sqrt(dn);
        case Norm::Backward: return inverse ? 1.0 / dn : 1.0;
        default: return inverse ? 1.0 : 1.0 / dn;
    }
}

inline int64_t parse_length(nb::handle o) {  // n: None (-1) or an int >= 1
    if (o.is_none()) return -1;
    int64_t v = 0;
    switch (index_value(o, v)) {
        case IndexResult::NotInt: throw nb::type_error(("'n' must be an int or None, not " + py_type_name(o)).c_str());
        case IndexResult::Overflow: throw std::overflow_error("'n' is out of range");
        case IndexResult::Ok: break;
    }
    if (v < 1) throw nb::value_error(("'n' must be at least 1, not " + std::to_string(v)).c_str());
    return v;
}

inline int parse_axis(nb::handle o, int ndim) {
    int64_t v = 0;
    switch (index_value(o, v)) {
        case IndexResult::NotInt: throw nb::type_error(("'axis' must be an int, not " + py_type_name(o)).c_str());
        case IndexResult::Overflow: throw std::overflow_error("'axis' is out of range");
        case IndexResult::Ok: break;
    }
    return normalize_axis(v, ndim);
}

struct Args {
    const NDArray* x;
    int64_t n;  // -1: None
    int axis;
    Norm norm;
};

inline Args parse_args(nb::handle xh, nb::handle nh, nb::handle axh, nb::handle normh) {
    const NDArray& x = require_ndarray(xh, "x");
    if (x.dtype == DType::String) throw nb::type_error("'x' must be numeric, not a string array");
    const int64_t n = parse_length(nh);
    const int axis = parse_axis(axh, x.ndim);
    return {&x, n, axis, parse_norm(normh)};
}

// n, or the axis's length when n is None (a length of 0 is an error).
inline int64_t length_or_axis(const Args& a) {
    if (a.n >= 0) return a.n;
    const int64_t m = a.x->shape[static_cast<size_t>(a.axis)];
    if (m == 0) throw nb::value_error("cannot transform an axis of length 0");
    return m;
}

inline NDArray new_result(const NDArray& x, int axis, int64_t len, DType dtype) {
    std::vector<int64_t> shape(x.shape.begin(), x.shape.begin() + x.ndim);
    shape[static_cast<size_t>(axis)] = len;
    return new_array(dtype, shape);
}

// The normalisation factor of an n-dimensional transform: as norm_factor with N = the product of the
// lengths (a double; exact below 2^53).
inline double norm_factor_nd(Norm norm, const std::vector<int64_t>& lengths, bool inverse) {
    double dn = 1.0;
    for (const int64_t v : lengths) dn *= static_cast<double>(v);
    switch (norm) {
        case Norm::Ortho: return 1.0 / std::sqrt(dn);
        case Norm::Backward: return inverse ? 1.0 / dn : 1.0;
        default: return inverse ? 1.0 : 1.0 / dn;
    }
}

// s: None (empty, given = false), or an int or a tuple or list of ints, each at least 1 or -1 (the
// input's length along that axis; docs/design/fft.md §2.4).
inline std::vector<int64_t> parse_s(nb::handle o, bool& given) {
    given = !o.is_none();
    std::vector<int64_t> s;
    if (!given) return s;
    auto one = [&](nb::handle item) {
        int64_t v = 0;
        switch (index_value(item, v)) {
            case IndexResult::NotInt:
                throw nb::type_error(
                    ("'s' must be an int or a tuple or list of ints, not " + py_type_name(item)).c_str()
                );
            case IndexResult::Overflow: throw std::overflow_error("'s' is out of range");
            case IndexResult::Ok: break;
        }
        if (v < 1 && v != -1) {
            throw nb::value_error(("the entries of 's' must be at least 1 (or -1), not " + std::to_string(v)).c_str());
        }
        s.push_back(v);
    };
    if (PyTuple_Check(o.ptr()) || PyList_Check(o.ptr())) {
        for (nb::handle item : o) one(item);
    } else {
        one(o);
    }
    return s;
}

struct NdArgs {
    const NDArray* x;
    std::vector<int64_t> s;  // as given (-1 kept), or empty when None
    bool s_given;
    std::vector<int> axes;  // in [0, ndim), distinct, not empty
    Norm norm;
};

inline NdArgs parse_nd(nb::handle xh, nb::handle sh, nb::handle axh, nb::handle normh) {
    const NDArray& x = require_ndarray(xh, "x");
    if (x.dtype == DType::String) throw nb::type_error("'x' must be numeric, not a string array");
    NdArgs a{&x, {}, false, {}, Norm::Backward};
    a.s = parse_s(sh, a.s_given);
    const int nd = x.ndim;
    if (axh.is_none()) {
        if (a.s_given) {  // the last len(s) axes (decision FT14)
            const int64_t k = static_cast<int64_t>(a.s.size());
            if (k > nd) {
                const std::string msg =
                    "'s' has " + std::to_string(k) + " entries for an array of dimension " + std::to_string(nd);
                throw nb::value_error(msg.c_str());
            }
            for (int64_t i = 0; i < k; ++i) a.axes.push_back(static_cast<int>(nd - k + i));
        } else {
            for (int d = 0; d < nd; ++d) a.axes.push_back(d);
        }
    } else {
        a.axes = parse_axes(axh, nd, "axes");
        if (a.s_given && a.s.size() != a.axes.size()) {
            const std::string msg = "'s' and 'axes' must have the same length, not " + std::to_string(a.s.size()) +
                                    " and " + std::to_string(a.axes.size());
            throw nb::value_error(msg.c_str());
        }
    }
    if (a.axes.empty()) throw nb::value_error(axh.is_none() ? "'s' must not be empty" : "'axes' must not be empty");
    a.norm = parse_norm(normh);
    return a;
}

// The transform length along axes[i]: s[i], the input's length when s is None or s[i] is -1 (0 is an
// error then).
inline int64_t nd_length(const NdArgs& a, size_t i) {
    if (a.s_given && a.s[i] != -1) return a.s[i];
    const int64_t m = a.x->shape[static_cast<size_t>(a.axes[i])];
    if (m == 0) throw nb::value_error("cannot transform an axis of length 0");
    return m;
}

inline std::vector<int64_t> shape_of(const NDArray& x) {
    return std::vector<int64_t>(x.shape.begin(), x.shape.begin() + x.ndim);
}

// The result's shape: x's with len[i] along axes[i] (and with `last_len` along the last of axes when
// given, for the real transforms' half spectrum).
inline std::vector<int64_t> nd_shape(const NDArray& x, const std::vector<int>& axes, const std::vector<int64_t>& len) {
    std::vector<int64_t> shape = shape_of(x);
    for (size_t i = 0; i < axes.size(); ++i) shape[static_cast<size_t>(axes[i])] = len[i];
    return shape;
}

inline bool any_zero(const std::vector<int64_t>& shape) {
    for (const int64_t v : shape)
        if (v == 0) return true;
    return false;
}

}  // namespace sd::fft
