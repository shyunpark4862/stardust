#include "core/column.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <type_traits>

namespace nb = nanobind;

namespace sd {

const char* dtype_name(DType dt) {
    switch (dt) {
    case DType::Bool: return "bool";
    case DType::Int64: return "int64";
    case DType::Float64: return "float64";
    case DType::String: return "string";
    }
    return "?";
}

DType parse_dtype(std::string_view name) {
    if (name == "bool") return DType::Bool;
    if (name == "int64") return DType::Int64;
    if (name == "float64") return DType::Float64;
    if (name == "string") return DType::String;
    throw nb::value_error(("unknown dtype '" + std::string(name) + "'").c_str());
}

Column make_fixed(DType dtype, std::int64_t length) {
    Column col;
    col.dtype = dtype;
    col.length = length;
    col.values = Buffer::allocate(length * item_size(dtype), /*zeroed=*/false);
    return col;
}

std::uint8_t* add_validity(Column& col) {
    const std::int64_t nbytes = bitmap::num_bytes(col.length);
    col.validity = Buffer::allocate(nbytes, /*zeroed=*/false);
    auto* bits = reinterpret_cast<std::uint8_t*>(col.validity->data());
    std::memset(bits, 0xff, static_cast<std::size_t>(nbytes));
    return bits;
}

std::shared_ptr<Buffer> rebased_validity(const Column& col) {
    if (!col.validity) return nullptr;
    auto out = Buffer::allocate(bitmap::num_bytes(col.length), /*zeroed=*/false);
    bitmap::copy_bits(col.validity_bits(), col.offset, col.length, reinterpret_cast<std::uint8_t*>(out->data()));
    return out;
}

std::shared_ptr<Buffer> and_validity(const Column& a, const Column& b, int scalar_side, std::int64_t n) {
    const Column* rows[2] = {&a, &b};
    std::shared_ptr<Buffer> out;
    for (int side = 1; side <= 2; ++side) {
        const Column& c = *rows[side - 1];
        if (!c.validity) continue;
        if (side == scalar_side) {  // an NA scalar makes every row NA
            out = Buffer::allocate(bitmap::num_bytes(n), /*zeroed=*/true);
            return out;
        }
        if (!out) {
            out = rebased_validity(c);
        } else {
            auto other = rebased_validity(c);
            auto* o = reinterpret_cast<std::uint8_t*>(out->data());
            const auto* p = reinterpret_cast<const std::uint8_t*>(other->data());
            for (std::int64_t j = 0; j < bitmap::num_bytes(n); ++j) o[j] &= p[j];
        }
    }
    return out;
}

std::vector<std::uint8_t> valid_bytes(const Column& col) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(col.length), 1);
    if (auto bits = rebased_validity(col))
        bitmap::expand(reinterpret_cast<const std::uint8_t*>(bits->data()), col.length, out.data());
    return out;
}

void finish_fixed(Column& col) {
    col.null_count = 0;
    if (!col.validity) return;
    const auto* bits = col.validity_bits();
    col.null_count = col.length - bitmap::count_set(bits, 0, col.length);
    if (col.null_count == 0) {
        col.validity.reset();
        return;
    }
    // Branch-free: AND every value with all-ones (valid) or zero (NA), the
    // bitmap first expanded to one byte per row (M10) so the loop vectorizes.
    std::vector<std::uint8_t> m(static_cast<std::size_t>(col.length));
    bitmap::expand(bits, col.length, m.data());
    // `n` is a local: a store through a uint8_t pointer may alias anything, so
    // a loop bound read from col.length would be reloaded every iteration and
    // the loop would not vectorize (M10).
    const std::int64_t n = col.length;
    auto zero = [&](auto* v) {
        using T = std::remove_reference_t<decltype(*v)>;
        for (std::int64_t i = 0; i < n; ++i) v[i] &= static_cast<T>(0 - static_cast<T>(m[i]));
    };
    if (item_size(col.dtype) == 1) zero(reinterpret_cast<std::uint8_t*>(col.values->data()));
    else zero(reinterpret_cast<std::int64_t*>(col.values->data()));  // int64 and float64: 0.0 is all-zero bits
}

Column Column::slice(std::int64_t start, std::int64_t len) const {
    if (start < 0 || len < 0 || start + len > length) throw nb::index_error("column slice out of range");
    Column out = *this;
    out.offset = offset + start;
    out.length = len;
    if (validity) {
        out.null_count = len - bitmap::count_set(validity_bits(), out.offset, len);
        if (out.null_count == 0) out.validity.reset();
    }
    return out;
}

namespace {

// The source's validity, re-based to start at row 0 of a fresh column.
void copy_validity(const Column& src, Column& dst) {
    dst.null_count = src.null_count;
    if (!src.validity) return;
    std::uint8_t* bits = add_validity(dst);
    for (std::int64_t i = 0; i < src.length; ++i)
        if (!src.is_valid(i)) bitmap::clear(bits, i);
}

template <class To, class From, class Exact>
Column convert_values(const Column& src, DType target, Exact exact, const char* what) {
    Column out = make_fixed(target, src.length);
    copy_validity(src, out);
    const From* in = src.data<From>();
    To* o = reinterpret_cast<To*>(out.values->data());
    for (std::int64_t i = 0; i < src.length; ++i) {
        // NA rows hold 0 (ARCHITECTURE.md §2.4), which converts exactly.
        if (src.is_valid(i) && !exact(in[i])) {
            std::string v = std::is_same_v<From, double> ? nb::repr(nb::float_(in[i])).c_str()
                                                         : std::to_string(static_cast<std::int64_t>(in[i]));
            throw nb::value_error((std::string(what) + " " + v + " cannot be converted to " + dtype_name(target) +
                                   " without loss").c_str());
        }
        o[i] = static_cast<To>(in[i]);
    }
    return out;
}

}  // namespace

Column Column::convert(DType target) const {
    if (target == dtype) return *this;
    if (dtype == DType::Int64 && target == DType::Float64)
        return convert_values<double, std::int64_t>(*this, target, int_is_exact_double, "int");
    if (dtype == DType::Float64 && target == DType::Int64)
        return convert_values<std::int64_t, double>(*this, target, double_is_exact_int, "float");
    throw nb::type_error((std::string("cannot convert ") + dtype_name(dtype) + " to " + dtype_name(target) +
                          " here; only lossless int64 <-> float64 conversions are allowed (use astype)")
                             .c_str());
}

}  // namespace sd
