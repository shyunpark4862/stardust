// The strict text grammars of API.md §12, shared by astype (M2) and
// read_csv (M9) so that both read "12", "nan" or " 1" the same way.
#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace sd::text {

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

// [+-]? digits, ASCII only.
inline bool int_grammar(std::string_view s) {
    if (!s.empty() && (s[0] == '+' || s[0] == '-')) s.remove_prefix(1);
    if (s.empty()) return false;
    for (char c : s)
        if (!is_digit(c)) return false;
    return true;
}

// An int64 from text in int_grammar; false when it is out of range.
inline bool to_int(std::string_view s, std::int64_t& v) {
    const char* first = s.data() + (s[0] == '+' ? 1 : 0);  // from_chars takes '-' but not '+'
    return std::from_chars(first, s.data() + s.size(), v).ec == std::errc{};
}

// [+-]? ( digits [. digits?] | . digits ) ( [eE] [+-]? digits )?   or   [+-]? (nan | inf | infinity), any case
// Whether `s` is `word` (lowercase) ignoring ASCII case, without copying.
inline bool equals_lower(std::string_view s, std::string_view word) {
    if (s.size() != word.size()) return false;
    for (std::size_t k = 0; k < s.size(); ++k)
        if ((s[k] >= 'A' && s[k] <= 'Z' ? s[k] - 'A' + 'a' : s[k]) != word[k]) return false;
    return true;
}

inline bool float_grammar(std::string_view s) {
    std::size_t i = 0, n = s.size();
    if (i < n && (s[i] == '+' || s[i] == '-')) ++i;
    const std::string_view rest = s.substr(i);
    if (equals_lower(rest, "nan") || equals_lower(rest, "inf") || equals_lower(rest, "infinity")) return true;
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

// A double from text in float_grammar, correctly rounded; overflow and deep
// underflow give +-inf and +-0.0 (and subnormals) as Python's float() does.
inline double to_double(std::string_view s) {
    const char* first = s.data() + (s[0] == '+' ? 1 : 0);
    double v = 0;
    auto res = std::from_chars(first, s.data() + s.size(), v);
    if (res.ec == std::errc::result_out_of_range) {
        std::string copy(s);
        v = std::strtod(copy.c_str(), nullptr);
    }
    return v;
}

}  // namespace sd::text
