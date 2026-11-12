#include "frame/ops/reduce.hpp"

#include "frame/interop.hpp"
#include "frame/ops/pairwise.hpp"
#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::frame {

namespace {

Column one_value(DType dt) { return make_fixed(dt, 1); }

// part(lo, hi) for every for_chunks chunk, then the parts folded in chunk
// order: fold(fold(fold(first, part0), part1), ...). The chunks depend on
// n alone, so the result does not depend on the thread count.
template <class T, class Part, class Fold>
T over_chunks(std::int64_t n, T first, Part part, Fold fold) {
    std::vector<T> parts(static_cast<std::size_t>(num_chunks(n)), first);
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        parts[static_cast<std::size_t>(lo / kChunkRows)] = part(lo, hi);
    });
    T acc = first;
    for (const T& p : parts) acc = fold(acc, p);
    return acc;
}

Column na_value(DType dt) {
    Column out;
    if (dt == DType::String) {
        out.dtype = DType::String;
        out.length = 1;
        out.values = Buffer::allocate(0, false);
        out.offsets = Buffer::allocate(16, true);
    } else {
        out = make_fixed(dt, 1);
        std::memset(out.values->data(), 0, static_cast<std::size_t>(itemsize(dt)));
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

// The value of a row as double for float sums (NA rows hold 0, docs/design/frame.md §4.5).
double as_double(const Column& c, std::int64_t i) {
    switch (c.dtype) {
    case DType::Int64: return static_cast<double>(c.data<std::int64_t>()[i]);
    case DType::Float64: return c.data<double>()[i];
    default: return static_cast<double>(c.data<std::uint8_t>()[i]);
    }
}

// The mean of the rows with a value; count > 0. int64 is summed as float64 the
// way the array area sums it (NA rows hold 0), then divided (docs/api/frame.md §6.3).
double mean_of(const Column& c, std::int64_t count) {
    if (c.dtype == DType::Int64)  // the array area's float64 sum, so both areas agree to the bit (DESIGN.md §9.1 #11)
        return array_sum_float64(c) / static_cast<double>(count);
    if (c.dtype == DType::Bool) {
        const auto* v = c.data<std::uint8_t>();
        const std::int64_t trues = over_chunks<std::int64_t>(
            c.length, 0,
            [v](std::int64_t lo, std::int64_t hi) {
                std::int64_t t = 0;
                for (std::int64_t i = lo; i < hi; ++i) t += v[i];
                return t;
            },
            [](std::int64_t a, std::int64_t b) { return a + b; });
        return static_cast<double>(trues) / static_cast<double>(count);
    }
    const double* v = c.data<double>();
    return float_sum(c.length, [v](std::int64_t i) { return v[i]; }) / static_cast<double>(count);
}

// The IEEE minimum or maximum of rows [lo, hi) (identity when there are
// none). IEEE 754-2019 minimum/maximum is associative and commutative (NaN
// wins, -0.0 < 0.0), so chunk results folded in any grouping give the same
// bits.
template <bool Max>
double float_minmax_range(const double* v, const std::uint8_t* m, std::int64_t lo, std::int64_t hi) {
    const double identity = Max ? -INFINITY : INFINITY;
    double best = identity;
    if (m == nullptr) {  // no NA: four vector accumulators, two lanes each
        std::int64_t i = lo;
        if (hi - lo >= 8) {
            float64x2_t acc[4];
            for (int k = 0; k < 4; ++k) acc[k] = vld1q_f64(v + lo + 2 * k);
            for (i = lo + 8; i + 8 <= hi; i += 8)
                for (int k = 0; k < 4; ++k) {
                    const float64x2_t x = vld1q_f64(v + i + 2 * k);
                    acc[k] = Max ? vmaxq_f64(acc[k], x) : vminq_f64(acc[k], x);
                }
            for (int k = 0; k < 4; ++k)
                for (int lane = 0; lane < 2; ++lane) best = ieee_minmax<Max>(best, acc[k][lane]);
        }
        for (; i < hi; ++i) best = ieee_minmax<Max>(best, v[i]);
        return best;
    }
    // With NA (docs/design/frame.md §7.4): blocks of rows are copied with NA rows replaced by the
    // identity (+inf for min, -inf for max) -- a branch-free select -- and
    // folded with the same vector instruction.
    constexpr std::int64_t kBlock = 1024;
    double buf[kBlock];
    float64x2_t acc[4];
    for (auto& a : acc) a = vdupq_n_f64(identity);
    for (std::int64_t b = lo; b < hi; b += kBlock) {
        const std::int64_t len = std::min(kBlock, hi - b);
        const std::uint8_t* mb = m + b;
        const double* vb = v + b;
        for (std::int64_t k = 0; k < len; ++k) buf[k] = mb[k] ? vb[k] : identity;
        std::int64_t k = 0;
        for (; k + 8 <= len; k += 8)  // four accumulators: four independent chains
            for (int a = 0; a < 4; ++a)
                acc[a] = Max ? vmaxq_f64(acc[a], vld1q_f64(buf + k + 2 * a)) : vminq_f64(acc[a], vld1q_f64(buf + k + 2 * a));
        for (; k < len; ++k) acc[0] = vsetq_lane_f64(ieee_minmax<Max>(vgetq_lane_f64(acc[0], 0), buf[k]), acc[0], 0);
    }
    for (auto& a : acc) best = ieee_minmax<Max>(ieee_minmax<Max>(best, vgetq_lane_f64(a, 0)), vgetq_lane_f64(a, 1));
    return best;
}

template <bool Max>
Column float_minmax(const Column& c) {
    const double* v = c.data<double>();
    const std::int64_t n = c.length;
    if (n == 0 || c.null_count == n) return na_value(DType::Float64);
    const std::vector<std::uint8_t> m = c.validity ? valid_bytes(c) : std::vector<std::uint8_t>{};
    const std::uint8_t* mp = c.validity ? m.data() : nullptr;
    const double best = over_chunks<double>(
        n, Max ? -INFINITY : INFINITY,
        [&](std::int64_t lo, std::int64_t hi) { return float_minmax_range<Max>(v, mp, lo, hi); },
        [](double a, double b) { return ieee_minmax<Max>(a, b); });
    return scalar(DType::Float64, best);
}

// int64 and bool: the value itself, typed, so the loop has no dtype switch.
template <bool Max, class T>
Column plain_minmax(const Column& c) {
    const T* v = c.data<T>();
    const std::int64_t n = c.length;
    if (c.null_count == n) return na_value(c.dtype);
    struct Best {
        bool any;
        T value;
    };
    auto pick = [](Best a, Best b) -> Best {
        if (!a.any) return b;
        if (!b.any) return a;
        return {true, Max ? (b.value > a.value ? b.value : a.value) : (b.value < a.value ? b.value : a.value)};
    };
    const Best best = over_chunks<Best>(
        n, Best{false, T{}},
        [&](std::int64_t lo, std::int64_t hi) {
            Best b{false, T{}};
            if (!c.validity) {
                b = {true, v[lo]};
                for (std::int64_t i = lo + 1; i < hi; ++i) b.value = Max ? (v[i] > b.value ? v[i] : b.value) : (v[i] < b.value ? v[i] : b.value);
            } else {
                for (std::int64_t i = lo; i < hi; ++i)
                    if (c.is_valid(i)) b = pick(b, Best{true, v[i]});
            }
            return b;
        },
        pick);
    return scalar(c.dtype, best.value);
}

template <bool Max>
Column min_max(const Column& c) {
    switch (c.dtype) {
    case DType::Float64: return float_minmax<Max>(c);
    case DType::Int64: return plain_minmax<Max, std::int64_t>(c);
    case DType::Bool: return plain_minmax<Max, std::uint8_t>(c);
    case DType::String: break;
    case DType::Complex128: break;  // not a frame dtype
    }
    // Strings: the winning row (the first of equal ones), returned as a
    // slice of the column.
    auto better = [&](std::int64_t a, std::int64_t b) {  // b replaces a
        if (b < 0) return a;
        if (a < 0) return b;
        return (Max ? c.string_at(b) > c.string_at(a) : c.string_at(b) < c.string_at(a)) ? b : a;
    };
    const std::int64_t at = over_chunks<std::int64_t>(
        c.length, -1,
        [&](std::int64_t lo, std::int64_t hi) {
            std::int64_t w = -1;
            for (std::int64_t i = lo; i < hi; ++i)
                if (c.is_valid(i)) w = better(w, i);
            return w;
        },
        better);
    if (at < 0) return na_value(c.dtype);
    return c.slice(at, 1);
}

[[noreturn]] void unsupported(std::string_view op, DType dt) {
    throw nb::type_error(("cannot compute " + std::string(op) + " of a " + dtype_name(dt) + " column").c_str());
}

}  // namespace

DType reduction_dtype(std::string_view op, DType dt) {
    if (op == "count") return DType::Int64;
    const bool keeps = op == "min" || op == "max" || op == "first" || op == "last";
    if (keeps) return dt;
    const bool is_var = op == "var" || op == "std";
    if (op != "sum" && op != "mean" && !is_var) throw nb::value_error(("unknown reduction '" + std::string(op) + "'").c_str());
    if (dt == DType::String) unsupported(op, dt);  // bool's std/var are of 0 and 1, as sd.std on bool
    if (op == "sum") return dt == DType::Float64 ? DType::Float64 : DType::Int64;
    return DType::Float64;
}

Column na_scalar(DType dt) { return na_value(dt); }

Column reduce(std::string_view op, const Column& c, bool skipna, double correction) {
    const std::int64_t count = c.length - c.null_count;
    if (op == "count") return scalar<std::int64_t>(DType::Int64, count);
    if (op == "first" || op == "last") throw nb::value_error("first/last are groupby reductions");

    const DType result = reduction_dtype(op, c.dtype);  // the dtype of the answer, NA included
    const bool is_sum = op == "sum", is_mean = op == "mean";
    const bool is_min = op == "min", is_max = op == "max";
    if (!skipna && c.null_count > 0) return na_value(result);

    if (is_sum) {
        if (c.dtype == DType::Float64) {
            const double* v = c.data<double>();
            return scalar(DType::Float64, float_sum(c.length, [v](std::int64_t i) { return v[i]; }));
        }
        // int64 wraps (docs/api/frame.md §1.5); NA rows hold 0. Wrapping addition is
        // associative, so the chunk order does not even matter.
        const bool is_int = c.dtype == DType::Int64;
        const auto* vi = c.data<std::int64_t>();
        const auto* vb = c.data<std::uint8_t>();
        const std::uint64_t total = over_chunks<std::uint64_t>(
            c.length, 0,
            [=](std::int64_t lo, std::int64_t hi) {
                std::uint64_t t = 0;
                if (is_int)
                    for (std::int64_t i = lo; i < hi; ++i) t += static_cast<std::uint64_t>(vi[i]);
                else
                    for (std::int64_t i = lo; i < hi; ++i) t += vb[i];
                return t;
            },
            [](std::uint64_t a, std::uint64_t b) { return a + b; });
        return scalar(DType::Int64, static_cast<std::int64_t>(total));
    }
    if (is_min || is_max) return is_max ? min_max<true>(c) : min_max<false>(c);
    if (count == 0) return na_value(DType::Float64);
    const double mean = mean_of(c, count);
    if (is_mean) return scalar(DType::Float64, mean);

    if (static_cast<double>(count) - correction <= 0) return na_value(DType::Float64);  // too few values (docs/api/frame.md §6.4)
    // NA rows add 0.0. The validity is a byte per row (docs/design/frame.md §4.4), so this is a
    // select, not a branch; the sum's order and bits are as before.
    const std::vector<std::uint8_t> m = valid_bytes(c);
    const double ssd = float_sum(c.length, [&](std::int64_t i) {
        const double d = as_double(c, i) - mean;
        return m[i] ? d * d : 0.0;
    }, 4);
    const double var = ssd / (static_cast<double>(count) - correction);
    return scalar(DType::Float64, op == "std" ? std::sqrt(var) : var);
}

}  // namespace sd::frame
