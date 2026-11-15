#include "array/ops/creation.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/ops/copy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace sd {

namespace {

// New arrays are contiguous, so filling them is a flat loop over memory
// regardless of their axis order.
template <typename T>
void fill_flat(NDArray& a, T value) {
    T* p = reinterpret_cast<T*>(a.data);
    std::fill(p, p + a.size(), value);
}

void fill_one(NDArray& a) {
    visit_dtype(a.dtype, [&]<typename T>(std::type_identity<T>) { fill_flat(a, cast_value<T>(true)); });
}

void fill_scalar(NDArray& a, nb::handle value, DType value_dtype) {
    visit_dtype(a.dtype, [&]<typename T>(std::type_identity<T>) { fill_flat(a, scalar_from_python<T>(value, value_dtype)); });
}

DType require_scalar(nb::handle o, const char* what) {
    if (auto d = scalar_dtype(o)) return *d;
    throw nb::type_error(
        (std::string("'") + what + "' must be a bool, int, float or complex, not " + py_type_name(o)).c_str());
}

// arange's arguments: real numbers only (docs/api/array.md §3.1).
DType require_real(nb::handle o, const char* what) {
    const std::optional<DType> d = scalar_dtype(o);
    if (d && *d != DType::Complex128) return *d;
    throw nb::type_error((std::string("'") + what + "' must be a bool, int or float, not " + py_type_name(o)).c_str());
}

// A bool or int argument that fits int64 (arange decides by this whether int
// arguments may be read as floats, docs/api/array.md §3.1).
bool fits_int64(nb::handle o) {
    std::int64_t v = 0;
    return index_value(o, v) != IndexResult::Overflow;  // NotInt: a bool or numpy.bool_
}

}  // namespace

nb::object zeros(nb::handle shape, nb::handle dtype) {
    return wrap(new_array(parse_dtype(dtype).value_or(DType::Float64), parse_shape_arg(shape).span(), /*zeroed=*/true));
}

nb::object ones(nb::handle shape, nb::handle dtype) {
    NDArray a = new_array(parse_dtype(dtype).value_or(DType::Float64), parse_shape_arg(shape).span());
    fill_one(a);
    return wrap(std::move(a));
}

nb::object empty(nb::handle shape, nb::handle dtype) {
    return wrap(new_array(parse_dtype(dtype).value_or(DType::Float64), parse_shape_arg(shape).span()));
}

nb::object full(nb::handle shape, nb::handle fill_value, nb::handle dtype) {
    const DType kind = require_scalar(fill_value, "fill_value");
    const DType target = parse_dtype(dtype).value_or(kind);
    check_castable(kind, target);
    NDArray a = new_array(target, parse_shape_arg(shape).span());
    fill_scalar(a, fill_value, kind);
    return wrap(std::move(a));
}

nb::object zeros_like(nb::handle x, nb::handle dtype) {
    const NDArray& p = require_ndarray(x);
    return wrap(new_array_like(p, parse_dtype(dtype).value_or(p.dtype), /*zeroed=*/true));
}

nb::object ones_like(nb::handle x, nb::handle dtype) {
    const NDArray& p = require_ndarray(x);
    NDArray a = new_array_like(p, parse_dtype(dtype).value_or(p.dtype));
    fill_one(a);
    return wrap(std::move(a));
}

nb::object empty_like(nb::handle x, nb::handle dtype) {
    const NDArray& p = require_ndarray(x);
    return wrap(new_array_like(p, parse_dtype(dtype).value_or(p.dtype)));
}

// Without dtype= the fill value meets x's dtype implicitly: upward only,
// like x[...] = fill_value (full_like(int array, 1.5) is a TypeError). With
// dtype= the conversion is explicit, as in full (docs/api/array.md §1.5).
nb::object full_like(nb::handle x, nb::handle fill_value, nb::handle dtype) {
    const NDArray& p = require_ndarray(x);
    const DType kind = require_scalar(fill_value, "fill_value");
    const std::optional<DType> requested = parse_dtype(dtype);
    const DType target = requested.value_or(p.dtype);
    if (requested) check_castable(kind, target);
    else check_assignable(kind, target);
    NDArray a = new_array_like(p, target);
    fill_scalar(a, fill_value, kind);
    return wrap(std::move(a));
}

// Values are start + i*step, computed in int64 when every argument is an int
// and in float64 otherwise, then converted to `dtype` by the explicit rules.
nb::object arange(nb::handle start_obj, nb::handle stop_obj, nb::handle step_obj, nb::handle dtype_obj) {
    const std::optional<DType> dtype = parse_dtype(dtype_obj);
    nb::object zero = nb::int_(0);
    nb::handle start = stop_obj.is_none() ? zero : start_obj;
    nb::handle stop = stop_obj.is_none() ? start_obj : stop_obj;

    DType kinds[3];
    nb::handle args[3] = {start, stop, step_obj};
    const char* names[3] = {"start", stop_obj.is_none() ? "start" : "stop", "step"};  // as the caller wrote them
    for (int i = 0; i < 3; ++i) kinds[i] = require_real(args[i], names[i]);
    // All ints: computed exactly in int64. With a float or complex dtype= an
    // int need not become an int64 (§9.1 #12 of DESIGN.md): when one is
    // outside int64, the arguments are read as floats instead.
    bool integral = std::max({kinds[0], kinds[1], kinds[2]}) <= DType::Int64;
    if (integral && dtype && (*dtype == DType::Float64 || *dtype == DType::Complex128))
        integral = fits_int64(start) && fits_int64(stop) && fits_int64(step_obj);

    if (integral) {
        const std::int64_t a = scalar_from_python<std::int64_t>(start, kinds[0]);
        const std::int64_t b = scalar_from_python<std::int64_t>(stop, kinds[1]);
        const std::int64_t s = scalar_from_python<std::int64_t>(step_obj, kinds[2]);
        if (s == 0) throw nb::value_error("'step' must not be zero");
        const __int128 span = static_cast<__int128>(b) - a;
        __int128 count = 0;  // up to 2^64 - 1: arange(-2**63, 2**63 - 1) does not fit int64
        if (s > 0 && span > 0) count = (span + s - 1) / s;
        if (s < 0 && span < 0) count = (-span - s - 1) / -static_cast<__int128>(s);
        if (count > std::numeric_limits<std::int64_t>::max())
            throw nb::value_error("the result would have too many elements");
        const auto n = static_cast<std::int64_t>(count);
        NDArray out = new_array(dtype.value_or(DType::Int64), std::vector<std::int64_t>{n});
        visit_dtype(out.dtype, [&]<typename T>(std::type_identity<T>) {
            T* p = reinterpret_cast<T*>(out.data);
            for (std::int64_t i = 0; i < n; ++i) p[i] = cast_value<T>(static_cast<std::int64_t>(a + static_cast<__int128>(i) * s));
        });
        return wrap(std::move(out));
    }

    const double a = scalar_from_python<double>(start, kinds[0]);
    const double b = scalar_from_python<double>(stop, kinds[1]);
    const double s = scalar_from_python<double>(step_obj, kinds[2]);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(s))
        throw nb::value_error("'start', 'stop' and 'step' must be finite (not nan or inf)");
    if (s == 0) throw nb::value_error("'step' must not be zero");
    const double len = std::ceil((b - a) / s);
    if (len >= 9.2e18) throw nb::value_error("the result would have too many elements");
    const std::int64_t n = len > 0 ? static_cast<std::int64_t>(len) : 0;
    NDArray out = new_array(dtype.value_or(DType::Float64), std::vector<std::int64_t>{n});
    visit_dtype(out.dtype, [&]<typename T>(std::type_identity<T>) {
        T* p = reinterpret_cast<T*>(out.data);
        for (std::int64_t i = 0; i < n; ++i) p[i] = cast_value<T>(a + static_cast<double>(i) * s);
    });
    return wrap(std::move(out));
}

// Same arithmetic as numpy (y = i*step + start, last element pinned to stop)
// so that results agree bit for bit.
nb::object linspace(nb::handle start, nb::handle stop, nb::handle num_obj, nb::handle dtype_obj, nb::handle endpoint_obj) {
    const DType ks = require_scalar(start, "start");
    const DType ke = require_scalar(stop, "stop");
    const std::int64_t num = parse_int(num_obj, "num");
    if (num < 0) throw nb::value_error("'num' must be non-negative");
    const bool endpoint = parse_bool(endpoint_obj, "endpoint");

    const DType inferred = promote(ks, ke) == DType::Complex128 ? DType::Complex128 : DType::Float64;
    const DType target = parse_dtype(dtype_obj).value_or(inferred);
    check_castable(inferred, target);
    NDArray out = new_array(target, std::vector<std::int64_t>{num});

    auto compute = [&]<typename C>(std::type_identity<C>) {
        const C s = scalar_from_python<C>(start, ks);
        const C e = scalar_from_python<C>(stop, ke);
        const C delta = e - s;
        const std::int64_t div = endpoint ? num - 1 : num;
        const C step = div > 0 ? delta / static_cast<double>(div) : C(0);
        visit_dtype(target, [&]<typename T>(std::type_identity<T>) {
            T* p = reinterpret_cast<T*>(out.data);
            for (std::int64_t i = 0; i < num; ++i) {
                const double fi = static_cast<double>(i);
                C y;
                if (div <= 0) y = fi * delta + s;
                else if (step == C(0)) y = (fi / static_cast<double>(div)) * delta + s;  // step underflowed
                else y = fi * step + s;
                if (endpoint && num > 1 && i == num - 1) y = e;
                p[i] = cast_value<T>(y);
            }
        });
    };
    if (inferred == DType::Complex128) compute(std::type_identity<complex128>{});
    else compute(std::type_identity<double>{});
    return wrap(std::move(out));
}

nb::object eye(nb::handle n_rows_obj, nb::handle n_cols_obj, nb::handle k_obj, nb::handle dtype_obj) {
    const std::int64_t n = parse_int(n_rows_obj, "n_rows");
    const std::int64_t m = n_cols_obj.is_none() ? n : parse_int(n_cols_obj, "n_cols");
    const std::int64_t k = parse_int(k_obj, "k");
    NDArray out = new_array(parse_dtype(dtype_obj).value_or(DType::Float64), std::vector<std::int64_t>{n, m}, true);
    visit_dtype(out.dtype, [&]<typename T>(std::type_identity<T>) {
        T* p = reinterpret_cast<T*>(out.data);
        for (std::int64_t i = std::max<std::int64_t>(0, -k); i < n && i + k < m; ++i) p[i * m + i + k] = cast_value<T>(true);
    });
    return wrap(std::move(out));
}

void bind_creation(nb::module_& m) {
    using nb::arg;
    m.def("asarray", named("asarray", &asarray), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none(), arg("copy").none() = nb::none());
    m.def("zeros", named("zeros", &zeros), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none());
    m.def("ones", named("ones", &ones), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none());
    m.def("empty", named("empty", &empty), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none());
    m.def("full", named("full", &full), arg().none(), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none());
    m.def("zeros_like", named("zeros_like", &zeros_like), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none());
    m.def("ones_like", named("ones_like", &ones_like), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none());
    m.def("empty_like", named("empty_like", &empty_like), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none());
    m.def("full_like", named("full_like", &full_like), arg().none(), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none());
    m.def("arange", named("arange", &arange), arg().none(), arg().none() = nb::none(), arg().none() = 1, nb::kw_only(),
          arg("dtype").none() = nb::none());
    m.def("linspace", named("linspace", &linspace), arg().none(), arg().none(), arg().none(), nb::kw_only(), arg("dtype").none() = nb::none(),
          arg("endpoint").none() = true);
    m.def("eye", named("eye", &eye), arg().none(), arg().none() = nb::none(), nb::kw_only(), arg("k").none() = 0, arg("dtype").none() = nb::none());
}

}  // namespace sd
