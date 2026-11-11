#include "ops/cast.hpp"

#include "ops/parse.hpp"
#include "parallel/pool.hpp"

#include <nanobind/nanobind.h>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace nb = nanobind;

namespace sd {

namespace {

using parallel::for_chunks;
using parallel::kChunkRows;

// A fixed-width result from a fixed-width input, keeping the validity.
// f(value) runs on rows with a value only; NA rows get 0. ok(value) says
// whether the value converts at all; the first row (in row order) that does
// not is reported through `bad`, and the chunks stop early.
template <class To, class From, class F, class Ok>
Column convert_fixed(const Column& col, DType target, F f, Ok ok, parallel::FirstRow& bad) {
    Column out = make_fixed(target, col.length);
    out.validity = rebased_validity(col);
    out.null_count = col.null_count;
    const From* in_ = col.data<From>();
    To* o_ = reinterpret_cast<To*>(out.values->data());
    const std::vector<std::uint8_t> m_ = col.validity ? valid_bytes(col) : std::vector<std::uint8_t>{};
    const bool any_na = col.validity != nullptr;
    for_chunks(col.length, [&](std::int64_t lo, std::int64_t hi) {
        if (bad.failed_before(lo)) return;
        const From* in = in_;  // locals, not captures (see selection.cpp isna)
        To* o = o_;
        const std::uint8_t* m = any_na ? m_.data() : nullptr;
        bool all_ok = true;
        for (std::int64_t i = lo; i < hi; ++i) all_ok &= !(m == nullptr || m[i]) || ok(in[i]);
        if (!all_ok) {
            for (std::int64_t i = lo; i < hi; ++i)
                if ((m == nullptr || m[i]) && !ok(in[i])) {
                    bad.offer(i);
                    return;
                }
        }
        if (m == nullptr) {
            for (std::int64_t i = lo; i < hi; ++i) o[i] = f(in[i]);
        } else {
            for (std::int64_t i = lo; i < hi; ++i) o[i] = m[i] ? f(in[i]) : To{};
        }
    });
    return out;
}

template <class To, class From, class F>
Column convert_fixed(const Column& col, DType target, F f) {
    parallel::FirstRow never;
    return convert_fixed<To, From>(col, target, f, [](From) { return true; }, never);
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
    offs[0] = 0;
    // Each chunk formats into its own string with chunk-relative offsets;
    // then the chunks are placed one after another, in chunk order.
    const std::int64_t nchunks = parallel::num_chunks(col.length);
    std::vector<std::string> text(static_cast<std::size_t>(nchunks));
    const From* in = col.data<From>();
    for_chunks(col.length, [&](std::int64_t lo, std::int64_t hi) {
        std::string& bytes = text[static_cast<std::size_t>(lo / kChunkRows)];
        for (std::int64_t i = lo; i < hi; ++i) {
            if (col.is_valid(i)) bytes += format(in[i]);
            offs[i + 1] = static_cast<std::int64_t>(bytes.size());
        }
    }, 16);
    std::vector<std::int64_t> start(static_cast<std::size_t>(nchunks) + 1, 0);
    for (std::size_t c = 0; c < text.size(); ++c) start[c + 1] = start[c] + static_cast<std::int64_t>(text[c].size());
    out.values = Buffer::allocate(start.back(), /*zeroed=*/false);
    for_chunks(col.length, [&](std::int64_t lo, std::int64_t hi) {
        const std::size_t c = static_cast<std::size_t>(lo / kChunkRows);
        std::memcpy(out.values->data() + start[c], text[c].data(), text[c].size());
        for (std::int64_t i = lo; i < hi; ++i) offs[i + 1] += start[c];
    });
    return out;
}

// --- from string: strict ASCII grammars (API.md §12) ---

[[noreturn]] void cannot_parse(std::string_view s, DType target) {
    nb::str repr = nb::repr(nb::str(s.data(), s.size()));
    throw nb::value_error(("cannot convert string " + std::string(repr.c_str()) + " to " + dtype_name(target)).c_str());
}

// parse(text, value) returns false for text that is not a value of the
// target. The error names the first such row, after the parallel part.
template <class To, class Parse>
Column from_string(const Column& col, DType target, Parse parse) {
    Column out = make_fixed(target, col.length);
    out.validity = rebased_validity(col);
    out.null_count = col.null_count;
    To* o = reinterpret_cast<To*>(out.values->data());
    parallel::FirstRow bad;
    for_chunks(col.length, [&](std::int64_t lo, std::int64_t hi) {
        if (bad.failed_before(lo)) return;
        for (std::int64_t i = lo; i < hi; ++i) {
            To v{};
            if (col.is_valid(i) && !parse(col.string_at(i), v)) {
                bad.offer(i);
                return;
            }
            o[i] = v;
        }
    }, 8);
    if (bad.any()) cannot_parse(col.string_at(bad.row()), target);
    return out;
}

bool parse_int(std::string_view s, std::int64_t& v) { return text::parse_int64(s, v) == 1; }
bool parse_float(std::string_view s, double& d) { return text::parse_double(s, d); }
bool parse_bool(std::string_view s, std::uint8_t& b) {
    if (s == "True") b = 1;
    else if (s == "False") b = 0;
    else return false;
    return true;
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
        return from_string<double>(col, target, [](std::string_view t, double& v) { return parse_float(t, v); });
    case DType::Int64:
        if (from == DType::Float64) {
            // Truncates toward zero; there is no int for NaN, inf or values
            // outside [-2^63, 2^63).
            parallel::FirstRow bad;
            Column out = convert_fixed<I, double>(
                col, target, [](double x) { return static_cast<I>(x); },
                [](double x) { return x >= -9223372036854775808.0 && x < 9223372036854775808.0; }, bad);
            if (bad.any())
                throw nb::value_error(("cannot convert float " + python_float_str(col.data<double>()[bad.row()]) +
                                       " to int64").c_str());
            return out;
        }
        if (from == DType::Bool) return convert_fixed<I, B>(col, target, [](B x) { return static_cast<I>(x); });
        return from_string<I>(col, target, [](std::string_view t, I& v) { return parse_int(t, v); });
    case DType::Bool:
        if (from == DType::Int64) return convert_fixed<B, I>(col, target, [](I x) -> B { return x != 0; });
        if (from == DType::Float64) return convert_fixed<B, double>(col, target, [](double x) -> B { return x != 0; });
        return from_string<B>(col, target, [](std::string_view t, B& v) { return parse_bool(t, v); });
    case DType::String:
        if (from == DType::Int64) return to_string<I>(col, [](I x) { return std::to_string(x); });
        if (from == DType::Float64) return to_string<double>(col, python_float_str);
        return to_string<B>(col, [](B x) { return std::string(x ? "True" : "False"); });
    }
    throw nb::value_error("unknown dtype");
}

}  // namespace sd
