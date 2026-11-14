// quantile, median, nanquantile, nanmedian (docs/api/array.md §7.10).
//
// Every lane (the elements of one output position, over the reduced axes) is
// copied as float64 into a row of a scratch matrix, where it is selected in
// place. The positions and the interpolation are numpy's, line by line, so
// the results are numpy's bits (docs/design/array.md §6.10): numpy 2.x
// _QuantileMethods, _get_indexes, _get_gamma, _lerp and _median.

#include "array/ops/quantile.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/sort.hpp"
#include "array/ops/pieces.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace sd {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------- methods

enum class Method {
    InvertedCdf,
    AveragedInvertedCdf,
    ClosestObservation,
    InterpolatedInvertedCdf,
    Hazen,
    Weibull,
    Linear,
    MedianUnbiased,
    NormalUnbiased,
    Lower,
    Higher,
    Midpoint,
    Nearest,
    Median,  // median(): numpy's mean of the middle one or two, not a quantile method
};

struct MethodName {
    std::string_view name;
    Method method;
};

constexpr MethodName kMethods[] = {
    {"inverted_cdf", Method::InvertedCdf},
    {"averaged_inverted_cdf", Method::AveragedInvertedCdf},
    {"closest_observation", Method::ClosestObservation},
    {"interpolated_inverted_cdf", Method::InterpolatedInvertedCdf},
    {"hazen", Method::Hazen},
    {"weibull", Method::Weibull},
    {"linear", Method::Linear},
    {"median_unbiased", Method::MedianUnbiased},
    {"normal_unbiased", Method::NormalUnbiased},
    {"lower", Method::Lower},
    {"higher", Method::Higher},
    {"midpoint", Method::Midpoint},
    {"nearest", Method::Nearest},
};

Method parse_method(nb::handle o) {
    if (!nb::isinstance<nb::str>(o))
        throw nb::type_error(("'method' must be a str, not " + py_type_name(o)).c_str());
    const std::string_view s = nb::borrow<nb::str>(o).c_str();
    for (const MethodName& m : kMethods)
        if (m.name == s) return m.method;
    std::string names;
    for (const MethodName& m : kMethods) {
        if (!names.empty()) names += ", ";
        names += "'" + std::string(m.name) + "'";
    }
    throw nb::value_error(("'method' must be one of " + names + ", not '" + std::string(s) + "'").c_str());
}

// --------------------------------------------------------------- positions
//
// What one quantile reads from a sorted lane of n > 0 values: element `lo`
// alone (lerp false), or elements lo and hi blended by gamma.

struct Pick {
    std::int64_t lo = 0, hi = 0;
    double gamma = 0;
    bool lerp = false;
};

// numpy's _compute_virtual_index: n*q + (alpha + q*(1 - alpha - beta)) - 1.
double virtual_index(double n, double q, double alpha, double beta) {
    return n * q + (alpha + q * (1 - alpha - beta)) - 1;
}

// Python's float % 2 (the sign of the divisor), as numpy's np.floor(index) % 2.
double py_mod2(double v) { return v - 2 * std::floor(v / 2); }

// _discrete_interpolation_to_boundaries: previous (when `take_previous`) or
// previous + 1, and never below 0.
std::int64_t to_boundary(double index, bool odd_rule) {
    const double previous = std::floor(index);
    const double gamma = index - previous;
    const bool take_previous = gamma == 0 && (!odd_rule || py_mod2(previous) == 1);
    const auto r = static_cast<std::int64_t>(take_previous ? previous : previous + 1);
    return r < 0 ? 0 : r;
}

Pick pick(Method m, std::int64_t count, double q) {
    const double n = static_cast<double>(count);
    Pick p;
    auto take = [&](std::int64_t k) {
        p.lo = p.hi = k;
        return p;
    };
    double v = 0;  // the virtual index of an interpolating method
    switch (m) {
        case Method::Lower: return take(static_cast<std::int64_t>(std::floor((n - 1) * q)));
        case Method::Higher: return take(static_cast<std::int64_t>(std::ceil((n - 1) * q)));
        case Method::Nearest: return take(static_cast<std::int64_t>(std::nearbyint((n - 1) * q)));  // half to even, as np.around
        case Method::InvertedCdf: return take(to_boundary(n * q - 1, false));
        case Method::ClosestObservation: return take(to_boundary(n * q - 1 - 0.5, true));
        case Method::AveragedInvertedCdf: v = n * q - 1; break;
        case Method::InterpolatedInvertedCdf: v = virtual_index(n, q, 0, 1); break;
        case Method::Hazen: v = virtual_index(n, q, 0.5, 0.5); break;
        case Method::Weibull: v = virtual_index(n, q, 0, 0); break;
        case Method::Linear: v = (n - 1) * q; break;
        case Method::MedianUnbiased: v = virtual_index(n, q, 1 / 3.0, 1 / 3.0); break;
        case Method::NormalUnbiased: v = virtual_index(n, q, 3 / 8.0, 3 / 8.0); break;
        case Method::Midpoint: v = 0.5 * (std::floor((n - 1) * q) + std::ceil((n - 1) * q)); break;
        case Method::Median: break;  // handled by the caller
    }
    // _get_indexes. Past the end both neighbours are the last element, and
    // the index numpy then subtracts in _get_gamma is -1, not n - 1.
    double previous = std::floor(v);
    if (v >= n - 1) {
        previous = -1;
        p.lo = p.hi = count - 1;
    } else if (v < 0) {
        previous = 0;
        p.lo = p.hi = 0;
    } else {
        p.lo = static_cast<std::int64_t>(previous);
        p.hi = p.lo + 1;
    }
    double gamma = v - previous;
    if (m == Method::AveragedInvertedCdf) gamma = gamma == 0 ? 0.5 : 1.0;
    if (m == Method::Midpoint) gamma = std::floor(v) == v ? 0.0 : 0.5;
    p.gamma = gamma;
    p.lerp = true;
    return p;
}

// numpy's _lerp, in its order of operations.
double lerp(double a, double b, double t) {
    const double diff = b - a;
    return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

// ---------------------------------------------------------------- one lane

struct Job {
    Method method;
    bool skip_nan;
    const std::vector<double>* qs;  // empty for median
    std::int64_t n;                 // lane length before NaNs are dropped
    std::int64_t lanes;             // output positions; the result row stride
};

// Lanes up to this long are sorted (the NEON sort) rather than selected:
// measured per element, sorting 16 / 256 / 1000 / 4096 values took 2.4 / 4.1
// / 5.6 / 7.3 ns against 8.3 / 8.0 / 7.8 / 7.5 ns for nth_element, which wins
// above (6.9 ns at 1e6 against 13.9 ns).
constexpr std::int64_t kSortSelect = 4096;

// Puts the order statistics at the (ascending, distinct) positions ks[0..count)
// in place, within p[lo..hi): the middle position first, then each half
// only inside its side of it -- about n log(count) work instead of the
// n * count of selecting left to right. A position at either end of its
// range is just the smallest or largest of the range, a plain scan.
// Everything compares in the total order -0.0 < +0.0 (TotalLess), the order
// the sorts use too, so every path picks the same value, sign included.
void select(double* p, std::int64_t lo, std::int64_t hi, const std::int64_t* ks, int count) {
    if (count == 0) return;
    if (count > 32) {  // many positions: sorting the range is cheaper
        sort_float64_total(p + lo, hi - lo, false);
        return;
    }
    const int mid = count / 2;
    const std::int64_t k = ks[mid];
    if (k == lo) std::iter_swap(p + k, std::min_element(p + lo, p + hi, TotalLess{}));
    else if (k == hi - 1) std::iter_swap(p + k, std::max_element(p + lo, p + hi, TotalLess{}));
    else std::nth_element(p + lo, p + k, p + hi, TotalLess{});
    select(p, lo, k, ks, mid);
    select(p, k + 1, hi, ks + mid + 1, count - mid - 1);
}

// The order statistics at ks: a short lane is sorted; a long lane that is
// the only one is sorted in parallel pieces when the pool is free (10
// threads, 1e6 values: 4.1 ms against 6.9 ms selecting); otherwise selected.
void order_statistics(double* p, std::int64_t n, bool only_lane, const std::int64_t* ks, int count) {
    if (n <= kSortSelect) sort_float64_total(p, n, false);
    else if (only_lane && n >= 2 * (1 << 16) && parallel::available()) sort_float64_total(p, n, true);
    else select(p, 0, n, ks, count);
}

void run_lane(const Job& job, double* p, double* out) {
    std::int64_t n = job.n;
    const std::int64_t results = job.qs ? static_cast<std::int64_t>(job.qs->size()) : 1;
    auto fill = [&](double v) {
        for (std::int64_t j = 0; j < results; ++j) out[j * job.lanes] = v;
    };
    if (job.skip_nan) {
        n = std::remove_if(p, p + n, [](double v) { return std::isnan(v); }) - p;
    } else if (std::any_of(p, p + n, [](double v) { return std::isnan(v); })) {
        fill(kNaN);  // numpy: a lane with a NaN gives NaN
        return;
    }
    if (n == 0) {
        fill(kNaN);
        return;
    }

    if (job.method == Method::Median) {
        // numpy's _median: mean() of the middle one or two, a sum from +0.0
        // divided by the count.
        const std::int64_t h = n / 2;
        if (n % 2) {
            const std::int64_t ks[] = {h};
            order_statistics(p, n, job.lanes == 1, ks, 1);
            out[0] = (0.0 + p[h]) / 1.0;
        } else {
            const std::int64_t ks[] = {h - 1, h};
            order_statistics(p, n, job.lanes == 1, ks, 2);
            out[0] = (0.0 + p[h - 1] + p[h]) / 2.0;
        }
        return;
    }

    std::vector<Pick> picks(results);
    std::vector<std::int64_t> ks;
    ks.reserve(2 * results);
    for (std::int64_t j = 0; j < results; ++j) {
        picks[j] = pick(job.method, n, (*job.qs)[j]);
        ks.push_back(picks[j].lo);
        ks.push_back(picks[j].hi);
    }
    std::sort(ks.begin(), ks.end());
    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    order_statistics(p, n, job.lanes == 1, ks.data(), static_cast<int>(ks.size()));
    for (std::int64_t j = 0; j < results; ++j) {
        const Pick& k = picks[j];
        out[j * job.lanes] = k.lerp ? lerp(p[k.lo], p[k.hi], k.gamma) : p[k.lo];
    }
}

// ---------------------------------------------------------------- arguments

// q: a Python int or float, or a 1-D real array; every value in [0, 1].
// `scalar` tells whether the result has a q axis.
std::vector<double> parse_q(nb::handle o, bool& scalar) {
    std::vector<double> qs;
    if (is_ndarray(o)) {
        const NDArray& a = as_ndarray(o);
        if (a.dtype == DType::Complex128) throw nb::type_error("'q' must be real, not complex128");
        if (a.ndim > 1) throw nb::value_error("'q' must be a scalar or 1-D");
        scalar = false;
        NDArray d = new_array(DType::Float64, a.shape_span());
        assign(d, a);
        const double* v = reinterpret_cast<const double*>(d.data);
        qs.assign(v, v + d.size());
    } else if (const std::optional<double> v = real_value(o)) {
        scalar = true;
        qs.push_back(*v);
    } else {
        throw nb::type_error(("'q' must be a float or an ndarray, not " + py_type_name(o)).c_str());
    }
    for (double q : qs)
        if (!(q >= 0 && q <= 1)) throw nb::value_error("'q' must be in the range [0, 1]");
    return qs;
}

nb::object compute(nb::handle x_obj, const std::vector<double>* qs, bool q_scalar, nb::handle axis_obj, Method method,
                   bool skip_nan, bool keepdims) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) throw nb::type_error("not supported for complex128");
    std::array<bool, kMaxNdim> reduced{};
    if (axis_obj.is_none()) {
        for (int d = 0; d < x.ndim; ++d) reduced[d] = true;
    } else {
        for (int d : parse_axes(axis_obj, x.ndim)) reduced[d] = true;
    }

    // x with the kept axes first and the reduced ones last, copied as a
    // C-ordered float64 matrix: one lane per row.
    NDArray view = x;
    std::vector<std::int64_t> shape;
    std::int64_t lanes = 1, n = 1;
    int k = 0;
    for (int pass = 0; pass < 2; ++pass)
        for (int d = 0; d < x.ndim; ++d) {
            if (reduced[d] != (pass == 1)) continue;
            view.shape[k] = x.shape[d];
            view.strides[k] = x.strides[d];
            ++k;
            shape.push_back(x.shape[d]);
            (pass ? n : lanes) *= x.shape[d];
        }
    view.update_flags();
    NDArray rows = new_array(DType::Float64, shape);
    assign(rows, view);

    // The result: q's axis, then x's shape without (or with 1 on) the reduced axes.
    std::vector<std::int64_t> out_shape;
    if (!q_scalar) out_shape.push_back(static_cast<std::int64_t>(qs->size()));
    for (int d = 0; d < x.ndim; ++d) {
        if (!reduced[d]) out_shape.push_back(x.shape[d]);
        else if (keepdims) out_shape.push_back(1);
    }
    // No 0-d arrays: a scalar result is computed into one element and returned as a float.
    const bool scalar_result = out_shape.empty();
    if (scalar_result) out_shape.push_back(1);
    NDArray out = new_array(DType::Float64, out_shape);

    const Job job{method, skip_nan, qs, n, lanes};
    double* base = reinterpret_cast<double*>(rows.data);
    double* res = reinterpret_cast<double*>(out.data);
    auto lane_range = [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) run_lane(job, base + i * n, res + i);
    };
    // Each lane is independent and selected in its own row: any split gives
    // the same bits.
    for_pieces(lanes, kParallelTask / std::max<std::int64_t>(n, 1), lanes * n,
               [&](std::int64_t, std::int64_t begin, std::int64_t end) { lane_range(begin, end); });
    if (scalar_result) return to_python(res[0]);
    return wrap(std::move(out));
}

template <bool SkipNan>
nb::object quantile_impl(nb::handle x, nb::handle q_obj, nb::handle axis, nb::handle method_obj, nb::handle keepdims_obj) {
    require_ndarray(x);
    bool q_scalar = true;
    const std::vector<double> qs = parse_q(q_obj, q_scalar);
    const Method method = parse_method(method_obj);
    const bool keepdims = parse_bool(keepdims_obj, "keepdims");
    return compute(x, &qs, q_scalar, axis, method, SkipNan, keepdims);
}

template <bool SkipNan>
nb::object median_impl(nb::handle x, nb::handle axis, nb::handle keepdims_obj) {
    require_ndarray(x);
    const bool keepdims = parse_bool(keepdims_obj, "keepdims");
    return compute(x, nullptr, true, axis, Method::Median, SkipNan, keepdims);
}

}  // namespace

void bind_quantile(nb::module_& m) {
    using nb::arg;
    const auto axis = arg("axis").none() = nb::none();
    const auto keepdims = arg("keepdims").none() = false;
    const auto method = arg("method").none() = "linear";
    m.def("quantile", named("quantile", &quantile_impl<false>), arg().none(), arg().none(), nb::kw_only(), axis, method,
          keepdims);
    m.def("nanquantile", named("nanquantile", &quantile_impl<true>), arg().none(), arg().none(), nb::kw_only(), axis,
          method, keepdims);
    m.def("median", named("median", &median_impl<false>), arg().none(), nb::kw_only(), axis, keepdims);
    m.def("nanmedian", named("nanmedian", &median_impl<true>), arg().none(), nb::kw_only(), axis, keepdims);
}

}  // namespace sd
