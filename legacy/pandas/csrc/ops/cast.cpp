#include "ops/cast.hpp"

#include <nanobind/nanobind.h>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace nb = nanobind;

namespace sd {

namespace {

// A fixed-width result from a fixed-width input, keeping the validity.
// f(value) runs on rows with a value only; NA rows get 0.
template <class To, class From, class F>
Column convert_fixed(const Column& col, DType target, F f) {
    Column out = make_fixed(target, col.length);
    out.validity = rebased_validity(col);
    out.null_count = col.null_count;
    const From* in = col.data<From>();
    To* o = reinterpret_cast<To*>(out.values->data());
    for (std::int64_t i = 0; i < col.length; ++i) o[i] = col.is_valid(i) ? f(in[i]) : To{};
    return out;
}

// --- to string ---

// Python's repr(float) (= str(float)): the shortest digits that round-trip,
// written fixed when the decimal exponent is in (-4, 16] and as d.ddde+XX
// otherwise, with ".0" added to integral fixed values.
std::string python_float_str(double d) {
    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d < 0 ? "-inf" : "inf";
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);  // shortest round-trip
    std::string_view sci(buf, static_cast<std::size_t>(res.ptr - buf));               // e.g. -1.2345e+20
    std::string out;
    if (sci.front() == '-') {
        out += '-';
        sci.remove_prefix(1);
    }
    const std::size_t e = sci.find('e');
    std::string digits;
    for (char c : sci.substr(0, e))
        if (c != '.') digits += c;
    const int exp10 = std::atoi(std::string(sci.substr(e + 1)).c_str());
    const int decpt = exp10 + 1;  // value = 0.<digits> * 10^decpt
    const int nd = static_cast<int>(digits.size());
    if (decpt <= -4 || decpt > 16) {
        out += digits[0];
        if (nd > 1) out += "." + digits.substr(1);
        const int x = decpt - 1;
        out += x < 0 ? "e-" : "e+";
        std::string mag = std::to_string(x < 0 ? -x : x);
        if (mag.size() < 2) mag = "0" + mag;
        out += mag;
    } else if (decpt <= 0) {
        out += "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
    } else if (decpt >= nd) {
        out += digits + std::string(static_cast<std::size_t>(decpt - nd), '0') + ".0";
    } else {
        out += digits.substr(0, static_cast<std::size_t>(decpt)) + "." + digits.substr(static_cast<std::size_t>(decpt));
    }
    return out;
}

template <class From, class F>
Column to_string(const Column& col, F format) {
    Column out;
    out.dtype = DType::String;
    out.length = col.length;
    out.null_count = col.null_count;
    out.validity = rebased_validity(col);
    out.offsets = Buffer::allocate((col.length + 1) * 8, /*zeroed=*/false);
    auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
    std::string bytes;
    offs[0] = 0;
    const From* in = col.data<From>();
    for (std::int64_t i = 0; i < col.length; ++i) {
        if (col.is_valid(i)) bytes += format(in[i]);
        offs[i + 1] = static_cast<std::int64_t>(bytes.size());
    }
    out.values = Buffer::allocate(static_cast<std::int64_t>(bytes.size()), /*zeroed=*/false);
    std::memcpy(out.values->data(), bytes.data(), bytes.size());
    return out;
}

// --- from string: strict ASCII grammars (API.md §12) ---

bool is_digit(char c) { return c >= '0' && c <= '9'; }

[[noreturn]] void cannot_parse(std::string_view s, DType target) {
    nb::str repr = nb::repr(nb::str(s.data(), s.size()));
    throw nb::value_error(("cannot convert string " + std::string(repr.c_str()) + " to " + dtype_name(target)).c_str());
}

std::int64_t parse_int(std::string_view s) {
    std::string_view body = s;
    if (!body.empty() && (body[0] == '+' || body[0] == '-')) body.remove_prefix(1);
    if (body.empty()) cannot_parse(s, DType::Int64);
    for (char c : body)
        if (!is_digit(c)) cannot_parse(s, DType::Int64);
    std::int64_t v = 0;
    const char* first = s.data() + (s[0] == '+' ? 1 : 0);  // from_chars takes '-' but not '+'
    auto res = std::from_chars(first, s.data() + s.size(), v);
    if (res.ec == std::errc::result_out_of_range) cannot_parse(s, DType::Int64);
    return v;
}

// [+-]? ( digits [. digits?] | . digits ) ( [eE] [+-]? digits )?   or   [+-]? (nan | inf | infinity)
bool float_grammar(std::string_view s) {
    std::size_t i = 0, n = s.size();
    if (i < n && (s[i] == '+' || s[i] == '-')) ++i;
    std::string rest;
    for (std::size_t k = i; k < n; ++k) rest += static_cast<char>(s[k] >= 'A' && s[k] <= 'Z' ? s[k] - 'A' + 'a' : s[k]);
    if (rest == "nan" || rest == "inf" || rest == "infinity") return true;
    std::size_t digits = 0;
    while (i < n && is_digit(s[i])) ++i, ++digits;
    if (i < n && s[i] == '.') {
        ++i;
        while (i < n && is_digit(s[i])) ++i, ++digits;
    }
    if (digits == 0) return false;
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < n && (s[i] == '+' || s[i] == '-')) ++i;
        std::size_t exp_digits = 0;
        while (i < n && is_digit(s[i])) ++i, ++exp_digits;
        if (exp_digits == 0) return false;
    }
    return i == n;
}

double parse_float(std::string_view s) {
    if (!float_grammar(s)) cannot_parse(s, DType::Float64);
    const char* first = s.data() + (s[0] == '+' ? 1 : 0);
    double v = 0;
    auto res = std::from_chars(first, s.data() + s.size(), v);
    if (res.ec == std::errc::result_out_of_range) {
        // Overflow and deep underflow: like Python's float(), give +-inf or
        // +-0.0 (and subnormals) by the correctly rounded strtod.
        std::string copy(s);
        v = std::strtod(copy.c_str(), nullptr);
    }
    return v;
}

template <class To, class F>
Column from_string(const Column& col, DType target, F parse) {
    Column out = make_fixed(target, col.length);
    out.validity = rebased_validity(col);
    out.null_count = col.null_count;
    To* o = reinterpret_cast<To*>(out.values->data());
    for (std::int64_t i = 0; i < col.length; ++i) o[i] = col.is_valid(i) ? parse(col.string_at(i)) : To{};
    return out;
}

}  // namespace

Column astype(const Column& col, DType target) {
    using I = std::int64_t;
    using B = std::uint8_t;
    const DType from = col.dtype;
    if (from == target) return col;
    switch (target) {
    case DType::Float64:
        if (from == DType::Int64) return convert_fixed<double, I>(col, target, [](I x) { return static_cast<double>(x); });
        if (from == DType::Bool) return convert_fixed<double, B>(col, target, [](B x) { return x ? 1.0 : 0.0; });
        return from_string<double>(col, target, parse_float);
    case DType::Int64:
        if (from == DType::Float64)
            return convert_fixed<I, double>(col, target, [](double x) {
                // Truncates toward zero; there is no int for NaN, inf or
                // values outside [-2^63, 2^63).
                if (!(x >= -9223372036854775808.0 && x < 9223372036854775808.0))
                    throw nb::value_error(("cannot convert float " + python_float_str(x) + " to int64").c_str());
                return static_cast<I>(x);
            });
        if (from == DType::Bool) return convert_fixed<I, B>(col, target, [](B x) { return static_cast<I>(x); });
        return from_string<I>(col, target, parse_int);
    case DType::Bool:
        if (from == DType::Int64) return convert_fixed<B, I>(col, target, [](I x) -> B { return x != 0; });
        if (from == DType::Float64) return convert_fixed<B, double>(col, target, [](double x) -> B { return x != 0; });
        return from_string<B>(col, target, [](std::string_view s) -> B {
            if (s == "True") return 1;
            if (s == "False") return 0;
            cannot_parse(s, DType::Bool);
        });
    case DType::String:
        if (from == DType::Int64) return to_string<I>(col, [](I x) { return std::to_string(x); });
        if (from == DType::Float64) return to_string<double>(col, python_float_str);
        return to_string<B>(col, [](B x) { return std::string(x ? "True" : "False"); });
    }
    throw nb::value_error("unknown dtype");
}

}  // namespace sd
