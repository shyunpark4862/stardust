#include "core/column.hpp"

#include <nanobind/nanobind.h>

#include <cmath>
#include <cstring>
#include <string>

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
