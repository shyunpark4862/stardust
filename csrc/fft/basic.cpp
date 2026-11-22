// fft and ifft of sd.fft (docs/api/fft.md §2; docs/design/fft.md §2.2, §2.4, §2.5; milestone FT1), and
// the plan cache's tools for the tests (_plan_info, _cache_state, _cache_clear, _twiddles).
//
//   _c2c(x, n, axis, norm, inverse) -> the complex transform of every line of x along axis
//
// The arguments are checked here, in the order of the signature (x, n, axis, norm), with the messages of
// API.md §2.1; the Python functions (src/stardust/fft/_basic.py) only fix the call shape.
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "fft/driver.hpp"
#include "fft/fft.hpp"
#include "fft/plan.hpp"
#include "fft/twiddle.hpp"

namespace nb = nanobind;

namespace sd::fft {

namespace {

enum class Norm { Backward, Ortho, Forward };

Norm parse_norm(nb::handle o) {
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
double norm_factor(Norm norm, int64_t n, bool inverse) {
    const double dn = static_cast<double>(n);
    switch (norm) {
        case Norm::Ortho: return 1.0 / std::sqrt(dn);
        case Norm::Backward: return inverse ? 1.0 / dn : 1.0;
        default: return inverse ? 1.0 : 1.0 / dn;
    }
}

int64_t parse_length(nb::handle o) {  // n: None (-1) or an int >= 1
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

int parse_axis(nb::handle o, int ndim) {
    int64_t v = 0;
    switch (index_value(o, v)) {
        case IndexResult::NotInt: throw nb::type_error(("'axis' must be an int, not " + py_type_name(o)).c_str());
        case IndexResult::Overflow: throw std::overflow_error("'axis' is out of range");
        case IndexResult::Ok: break;
    }
    return normalize_axis(v, ndim);
}

nb::object c2c(nb::handle xh, nb::handle nh, nb::handle axh, nb::handle normh, bool inverse) {
    return call_named(inverse ? "ifft" : "fft", [&]() -> nb::object {
        const NDArray& x = require_ndarray(xh, "x");
        if (x.dtype == DType::String) throw nb::type_error("'x' must be numeric, not a string array");
        int64_t n = parse_length(nh);
        const int axis = parse_axis(axh, x.ndim);
        const Norm norm = parse_norm(normh);
        if (n < 0) {
            n = x.shape[static_cast<size_t>(axis)];
            if (n == 0) throw nb::value_error("cannot transform an axis of length 0");
        }
        std::vector<int64_t> shape(x.shape.begin(), x.shape.begin() + x.ndim);
        shape[static_cast<size_t>(axis)] = n;
        NDArray out = new_array(DType::Complex128, shape);
        c2c_axis(x, axis, n, inverse, norm_factor(norm, n, inverse), out);
        return wrap(std::move(out));
    });
}

nb::dict plan_info(int64_t n) {
    const std::shared_ptr<const Plan> p = get_plan(Kind::Complex, n);
    nb::dict d;
    d["n"] = n;
    d["path"] = p->bluestein() ? "bluestein" : "mixed";
    d["bytes"] = p->bytes();
    d["cost"] = p->cost();
    if (p->bluestein()) {
        d["m"] = p->blue()->m();
        d["rho"] = p->blue()->rho();
        d["radices"] = p->blue()->inner().radices();
    } else {
        d["radices"] = p->mixed()->radices();
    }
    return d;
}

nb::dict cache_state_py() {
    const CacheState s = cache_state();
    nb::dict d;
    d["plans"] = s.plans;
    d["bytes"] = s.bytes;
    d["lengths"] = s.lengths;
    d["max_plans"] = kCachePlans;
    d["max_bytes"] = kCacheBytes;
    return d;
}

// The twiddles e^(-2 pi i k / n), k = 0 .. n - 1, as the plans compute them (twiddle.hpp): (n, 2) float64.
nb::object twiddles(int64_t n) {
    if (n < 1) throw nb::value_error("_twiddles(): n must be at least 1");
    NDArray out = new_array(DType::Float64, std::vector<int64_t>{n, 2});
    double* d = reinterpret_cast<double*>(out.data);
    for (int64_t k = 0; k < n; ++k) {
        const Twiddle w = twiddle(k, n);
        d[2 * k] = w.re, d[2 * k + 1] = w.im;
    }
    return wrap(std::move(out));
}

}  // namespace

void bind_basic(nb::module_& m) {
    m.def(
        "_c2c", &c2c, nb::arg("x").none(), nb::arg("n").none(), nb::arg("axis").none(), nb::arg("norm").none(),
        nb::arg("inverse")
    );  // fmt: skip
    m.def("_plan_info", &plan_info, nb::arg("n"));
    m.def("_cache_state", &cache_state_py);
    m.def("_cache_clear", [] { cache_clear(); });
    m.def("_twiddles", &twiddles, nb::arg("n"));
}

}  // namespace sd::fft
