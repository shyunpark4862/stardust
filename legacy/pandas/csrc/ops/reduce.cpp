#include "ops/reduce.hpp"

#include "ops/pairwise.hpp"

#include <nanobind/nanobind.h>

#include <arm_neon.h>

#include <cmath>
#include <cstring>
#include <string>

namespace nb = nanobind;

namespace sd {

namespace {

Column one_value(DType dt) { return make_fixed(dt, 1); }

Column na_value(DType dt) {
    Column out;
    if (dt == DType::String) {
        out.dtype = DType::String;
        out.length = 1;
        out.values = Buffer::allocate(0, false);
        out.offsets = Buffer::allocate(16, true);
    } else {
        out = make_fixed(dt, 1);
        std::memset(out.values->data(), 0, static_cast<std::size_t>(item_size(dt)));
    }
    add_validity(out)[0] = 0;
    out.null_count = 1;
    return out;
}

template <class T>
Column scalar(DType dt, T v) {
    Column out = one_value(dt);
    *reinterpret_cast<T*>(out.values->data()) = v;
    return out;
}

// The value of a row as double for float sums (NA rows hold 0, §2.4).
double as_double(const Column& c, std::int64_t i) {
    switch (c.dtype) {
    case DType::Int64: return static_cast<double>(c.data<std::int64_t>()[i]);
    case DType::Float64: return c.data<double>()[i];
    default: return static_cast<double>(c.data<std::uint8_t>()[i]);
    }
}

// The mean of the rows with a value; count > 0. int64 sums exactly in
// int128 (no overflow, no order dependence), then divides (API.md §15).
double mean_of(const Column& c, std::int64_t count) {
    if (c.dtype == DType::Int64) {
        __int128 total = 0;
        const auto* v = c.data<std::int64_t>();
        for (std::int64_t i = 0; i < c.length; ++i) total += v[i];  // NA rows hold 0
        return static_cast<double>(total) / static_cast<double>(count);
    }
    if (c.dtype == DType::Bool) {
        std::int64_t trues = 0;
        const auto* v = c.data<std::uint8_t>();
        for (std::int64_t i = 0; i < c.length; ++i) trues += v[i];
        return static_cast<double>(trues) / static_cast<double>(count);
    }
    const double* v = c.data<double>();
    return float_sum(c.length, [v](std::int64_t i) { return v[i]; }) / static_cast<double>(count);
}

template <bool Max>
Column float_minmax(const Column& c) {
    const double* v = c.data<double>();
    const std::int64_t n = c.length;
    if (n == 0) return na_value(DType::Float64);
    if (!c.validity) {  // no NA: four vector accumulators, two lanes each
        double best = v[0];
        std::int64_t i = 0;
        if (n >= 8) {
            float64x2_t acc[4];
            for (int k = 0; k < 4; ++k) acc[k] = vld1q_f64(v + 2 * k);
            for (i = 8; i + 8 <= n; i += 8)
                for (int k = 0; k < 4; ++k) {
                    const float64x2_t x = vld1q_f64(v + i + 2 * k);
                    acc[k] = Max ? vmaxq_f64(acc[k], x) : vminq_f64(acc[k], x);
                }
            for (int k = 0; k < 4; ++k)
                for (int lane = 0; lane < 2; ++lane) best = ieee_minmax<Max>(best, acc[k][lane]);
        }
        for (; i < n; ++i) best = ieee_minmax<Max>(best, v[i]);
        return scalar(DType::Float64, best);
    }
    std::int64_t i = 0;
    while (i < n && !c.is_valid(i)) ++i;
    if (i == n) return na_value(DType::Float64);
    double best = v[i];
    for (++i; i < n; ++i)
        if (c.is_valid(i)) best = ieee_minmax<Max>(best, v[i]);
    return scalar(DType::Float64, best);
}

// int64 and bool: the value itself, typed, so the loop has no dtype switch.
template <bool Max, class T>
Column plain_minmax(const Column& c) {
    const T* v = c.data<T>();
    const std::int64_t n = c.length;
    std::int64_t i = 0;
    while (i < n && !c.is_valid(i)) ++i;
    if (i == n) return na_value(c.dtype);
    T best = v[i];
    if (!c.validity) {
        for (++i; i < n; ++i) best = Max ? (v[i] > best ? v[i] : best) : (v[i] < best ? v[i] : best);
    } else {
        for (++i; i < n; ++i)
            if (c.is_valid(i)) best = Max ? (v[i] > best ? v[i] : best) : (v[i] < best ? v[i] : best);
    }
    return scalar(c.dtype, best);
}

template <bool Max>
Column min_max(const Column& c) {
    switch (c.dtype) {
    case DType::Float64: return float_minmax<Max>(c);
    case DType::Int64: return plain_minmax<Max, std::int64_t>(c);
    case DType::Bool: return plain_minmax<Max, std::uint8_t>(c);
    case DType::String: break;
    }
    std::int64_t at = -1;  // strings: the winning row, returned as a slice of the column
    for (std::int64_t i = 0; i < c.length; ++i)
        if (c.is_valid(i) && (at < 0 || (Max ? c.string_at(i) > c.string_at(at) : c.string_at(i) < c.string_at(at)))) at = i;
    if (at < 0) return na_value(c.dtype);
    return c.slice(at, 1);
}

[[noreturn]] void unsupported(std::string_view op, DType dt) {
    throw nb::type_error(("cannot compute " + std::string(op) + " of a " + dtype_name(dt) + " column (API.md §15)").c_str());
}

}  // namespace

DType reduction_dtype(std::string_view op, DType dt) {
    if (op == "count") return DType::Int64;
    const bool keeps = op == "min" || op == "max" || op == "first" || op == "last";
    if (keeps) return dt;
    const bool is_var = op == "var" || op == "std";
    if (op != "sum" && op != "mean" && !is_var) throw nb::value_error(("unknown reduction '" + std::string(op) + "'").c_str());
    if (dt == DType::String || (dt == DType::Bool && is_var)) unsupported(op, dt);
    if (op == "sum") return dt == DType::Float64 ? DType::Float64 : DType::Int64;
    return DType::Float64;
}

Column na_scalar(DType dt) { return na_value(dt); }

Column reduce(std::string_view op, const Column& c, bool skipna, std::int64_t ddof) {
    const std::int64_t count = c.length - c.null_count;
    if (op == "count") return scalar<std::int64_t>(DType::Int64, count);
    if (op == "first" || op == "last") throw nb::value_error("first/last are groupby reductions");

    const DType result = reduction_dtype(op, c.dtype);  // the dtype of the answer, NA included
    const bool is_sum = op == "sum", is_mean = op == "mean", is_var = op == "var" || op == "std";
    const bool is_min = op == "min", is_max = op == "max";
    if (is_var && ddof < 0) throw nb::value_error("ddof must be >= 0");
    if (!skipna && c.null_count > 0) return na_value(result);

    if (is_sum) {
        if (c.dtype == DType::Float64) {
            const double* v = c.data<double>();
            return scalar(DType::Float64, float_sum(c.length, [v](std::int64_t i) { return v[i]; }));
        }
        std::uint64_t total = 0;  // int64 wraps (API.md §3); NA rows hold 0
        if (c.dtype == DType::Int64) {
            const auto* v = c.data<std::int64_t>();
            for (std::int64_t i = 0; i < c.length; ++i) total += static_cast<std::uint64_t>(v[i]);
        } else {
            const auto* v = c.data<std::uint8_t>();
            for (std::int64_t i = 0; i < c.length; ++i) total += v[i];
        }
        return scalar(DType::Int64, static_cast<std::int64_t>(total));
    }
    if (is_min || is_max) return is_max ? min_max<true>(c) : min_max<false>(c);
    if (count == 0) return na_value(DType::Float64);
    const double mean = mean_of(c, count);
    if (is_mean) return scalar(DType::Float64, mean);

    if (count - ddof <= 0) return na_value(DType::Float64);  // too few values (API.md §15)
    const double ssd = float_sum(c.length, [&](std::int64_t i) {
        if (!c.is_valid(i)) return 0.0;
        const double d = as_double(c, i) - mean;
        return d * d;
    });
    const double var = ssd / static_cast<double>(count - ddof);
    return scalar(DType::Float64, op == "std" ? std::sqrt(var) : var);
}

}  // namespace sd
