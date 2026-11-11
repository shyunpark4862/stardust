#include "io/csv.hpp"

#include "ops/parse.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstring>
#include <string_view>
#include <unordered_set>

namespace nb = nanobind;

namespace sd {

namespace {

// Where a field's text is in the input. Quoted fields point inside the
// quotes; kEscaped says the text still holds "" pairs to turn into ".
struct Field {
    std::int64_t start;
    std::int64_t length;
    std::uint8_t flags;
};
constexpr std::uint8_t kQuoted = 1, kEscaped = 2;

[[noreturn]] void csv_error(std::int64_t line, const std::string& what) {
    throw nb::value_error(("read_csv: line " + std::to_string(line) + ": " + what).c_str());
}

// RFC 4180 tokenizer: one record per call (false at the end of the input).
// `line` is the 1-based line where the next record starts; quoted fields may
// span lines.
class Tokenizer {
public:
    Tokenizer(const char* p, std::size_t n, char sep) : p_(p), n_(n), sep_(sep) {
        if (n_ >= 3 && std::memcmp(p_, "\xEF\xBB\xBF", 3) == 0) pos_ = 3;  // UTF-8 byte order mark
    }

    std::int64_t line() const { return line_; }

    bool next(std::vector<Field>& out) {
        out.clear();
        if (pos_ >= n_) return false;
        while (true) {
            if (pos_ < n_ && p_[pos_] == '"') {
                quoted(out);
            } else {
                const std::size_t start = pos_;
                while (pos_ < n_ && p_[pos_] != sep_ && p_[pos_] != '\n') {
                    if (p_[pos_] == '"') csv_error(line_, "a quote inside an unquoted field");
                    ++pos_;
                }
                std::size_t end = pos_;
                if (end > start && p_[end - 1] == '\r' && pos_ < n_ && p_[pos_] == '\n') --end;  // a \r\n line end
                if (end > start && p_[end - 1] == '\r' && pos_ == n_) --end;
                out.push_back({static_cast<std::int64_t>(start), static_cast<std::int64_t>(end - start), 0});
            }
            if (pos_ >= n_) return true;  // the last record need not end with a newline
            if (p_[pos_] == sep_) {
                ++pos_;
                if (pos_ >= n_) {  // "a," at the very end: one more, empty field
                    out.push_back({static_cast<std::int64_t>(pos_), 0, 0});
                    return true;
                }
                continue;
            }
            ++pos_;  // '\n'
            ++line_;
            return true;
        }
    }

private:
    void quoted(std::vector<Field>& out) {
        const std::int64_t open_line = line_;
        const std::size_t start = ++pos_;
        std::uint8_t flags = kQuoted;
        while (true) {
            const void* q = std::memchr(p_ + pos_, '"', n_ - pos_);
            if (!q) csv_error(open_line, "a quoted field is never closed");
            const std::size_t at = static_cast<const char*>(q) - p_;
            for (std::size_t k = pos_; k < at; ++k) line_ += p_[k] == '\n';
            if (at + 1 < n_ && p_[at + 1] == '"') {  // "" is one quote
                flags |= kEscaped;
                pos_ = at + 2;
                continue;
            }
            out.push_back({static_cast<std::int64_t>(start), static_cast<std::int64_t>(at - start), flags});
            pos_ = at + 1;
            break;
        }
        if (pos_ < n_ && p_[pos_] == '\r' && pos_ + 1 < n_ && p_[pos_ + 1] == '\n') ++pos_;
        if (pos_ < n_ && p_[pos_] != sep_ && p_[pos_] != '\n')
            csv_error(line_, "a closing quote must be followed by the separator or the end of the line");
    }

    const char* p_;
    std::size_t n_, pos_ = 0;
    char sep_;
    std::int64_t line_ = 1;
};

// A field's text, with "" pairs made " (only escaped fields need a copy).
std::string_view text_of(const char* base, const Field& f, std::string& scratch) {
    std::string_view raw(base + f.start, static_cast<std::size_t>(f.length));
    if (!(f.flags & kEscaped)) return raw;
    scratch.clear();
    for (std::size_t i = 0; i < raw.size(); ++i) {
        scratch += raw[i];
        if (raw[i] == '"') ++i;  // skip the second quote of the pair
    }
    return scratch;
}

bool valid_utf8(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        if (len == 0 || i + len > s.size()) return false;
        std::uint32_t cp = len == 1 ? c : c & (0x7F >> len);
        for (std::size_t k = 1; k < len; ++k) {
            const auto cc = static_cast<unsigned char>(s[i + k]);
            if ((cc >> 6) != 0x2) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        // No overlong forms, no surrogates, nothing past U+10FFFF.
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) || cp > 0x10FFFF ||
            (cp >= 0xD800 && cp <= 0xDFFF))
            return false;
        i += len;
    }
    return true;
}

class ColumnBuilder {
public:
    ColumnBuilder(const char* base, const std::vector<Field>& fields, const std::vector<std::int64_t>& lines,
                  const std::unordered_set<std::string>& na, const std::string& name)
        : base_(base), f_(fields), lines_(lines), na_(na), name_(name) {}

    // API.md §19: NA is the unquoted empty field and the na_values texts.
    bool is_na(std::size_t i) {
        const Field& f = f_[i];
        if (f.length == 0 && !(f.flags & kQuoted)) return true;
        return !na_.empty() && na_.count(std::string(text(i)));
    }

    std::string_view text(std::size_t i) { return text_of(base_, f_[i], scratch_); }

    // int64, then float64, then bool, then string (API.md §19).
    DType infer() {
        bool all_int = true, all_float = true, all_bool = true, any = false;
        for (std::size_t i = 0; i < f_.size() && (all_int || all_float || all_bool); ++i) {
            if (is_na(i)) continue;
            any = true;
            const std::string_view s = text(i);
            if (text::int_grammar(s)) {
                std::int64_t v;
                if (!text::to_int(s, v)) {  // beyond int64: neither int64 nor (without loss) float64
                    all_int = all_float = false;
                } else if (!int_is_exact_double(v)) {
                    all_float = false;  // fine as int64; as float64 it would change
                }
            } else {
                all_int = false;
                if (!text::float_grammar(s)) all_float = false;
            }
            if (s != "True" && s != "False") all_bool = false;
        }
        if (!any) return DType::Float64;  // every value NA: float64 (API.md §10)
        if (all_int) return DType::Int64;
        if (all_float) return DType::Float64;
        if (all_bool) return DType::Bool;
        return DType::String;
    }

    Column build(DType dt) {
        const std::int64_t n = static_cast<std::int64_t>(f_.size());
        if (dt == DType::String) return build_strings();
        Column col = make_fixed(dt, n);
        std::uint8_t* bits = add_validity(col);
        for (std::int64_t i = 0; i < n; ++i) {
            if (is_na(i)) {
                bitmap::clear(bits, i);
                continue;
            }
            const std::string_view s = text(i);
            switch (dt) {
            case DType::Int64: {
                std::int64_t v;
                if (!text::int_grammar(s) || !text::to_int(s, v)) fail(i, s, dt);
                reinterpret_cast<std::int64_t*>(col.values->data())[i] = v;
                break;
            }
            case DType::Float64:
                if (!text::float_grammar(s)) fail(i, s, dt);
                reinterpret_cast<double*>(col.values->data())[i] = text::to_double(s);
                break;
            case DType::Bool:
                if (s != "True" && s != "False") fail(i, s, dt);
                reinterpret_cast<std::uint8_t*>(col.values->data())[i] = s == "True";
                break;
            case DType::String: break;
            }
        }
        finish_fixed(col);
        return col;
    }

private:
    Column build_strings() {
        const std::int64_t n = static_cast<std::int64_t>(f_.size());
        Column col;
        col.dtype = DType::String;
        col.length = n;
        col.offsets = Buffer::allocate((n + 1) * 8, false);
        std::uint8_t* bits = add_validity(col);
        auto* offs = reinterpret_cast<std::int64_t*>(col.offsets->data());
        offs[0] = 0;
        std::int64_t total = 0;
        for (std::int64_t i = 0; i < n; ++i) total += is_na(i) ? 0 : static_cast<std::int64_t>(text(i).size());
        col.values = Buffer::allocate(total, false);
        auto* out = reinterpret_cast<char*>(col.values->data());
        std::int64_t pos = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            if (is_na(i)) {
                bitmap::clear(bits, i);
            } else {
                const std::string_view s = text(i);
                if (!valid_utf8(s)) csv_error(lines_[i], "column '" + name_ + "' is not valid UTF-8");
                std::memcpy(out + pos, s.data(), s.size());
                pos += static_cast<std::int64_t>(s.size());
            }
            offs[i + 1] = pos;
        }
        col.null_count = n - bitmap::count_set(bits, 0, n);
        if (col.null_count == 0) col.validity.reset();
        return col;
    }

    [[noreturn]] void fail(std::int64_t i, std::string_view s, DType dt) {
        csv_error(lines_[i], "column '" + name_ + "': cannot read '" + std::string(s) + "' as " + dtype_name(dt));
    }

    const char* base_;
    const std::vector<Field>& f_;
    const std::vector<std::int64_t>& lines_;
    const std::unordered_set<std::string>& na_;
    const std::string& name_;
    std::string scratch_;
};

}  // namespace

std::pair<std::vector<std::string>, std::vector<Column>> read_csv(const char* data, std::size_t size,
                                                                   const CsvOptions& opt) {
    Tokenizer tok(data, size, opt.sep);
    std::vector<Field> rec;
    std::string scratch;

    std::vector<std::string> names = opt.names;
    if (opt.header) {
        const std::int64_t line = tok.line();
        if (!tok.next(rec)) csv_error(line, "the file is empty but header=True");
        if (names.empty()) {
            for (const Field& f : rec) names.emplace_back(text_of(data, f, scratch));
        } else if (names.size() != rec.size()) {
            csv_error(line, std::to_string(names.size()) + " names for a header of " + std::to_string(rec.size()) + " fields");
        }
    }
    if (names.empty()) throw nb::value_error("read_csv: header=False needs names= (API.md §19)");
    {
        std::unordered_set<std::string> seen;
        for (const auto& k : names)
            if (!seen.insert(k).second) throw nb::value_error(("read_csv: column name '" + k + "' repeats (API.md §6)").c_str());
    }

    // Which file column feeds each output column.
    std::vector<std::string> out_names = opt.usecols.empty() ? names : opt.usecols;
    std::vector<std::int64_t> source(out_names.size());
    for (std::size_t j = 0; j < out_names.size(); ++j) {
        auto it = std::find(names.begin(), names.end(), out_names[j]);
        if (it == names.end()) throw nb::key_error(("read_csv: no column '" + out_names[j] + "'").c_str());
        source[j] = it - names.begin();
    }

    std::vector<std::vector<Field>> fields(out_names.size());
    std::vector<std::int64_t> lines;
    const std::size_t ncols = names.size();
    while (opt.nrows < 0 || static_cast<std::int64_t>(lines.size()) < opt.nrows) {
        const std::int64_t line = tok.line();
        if (!tok.next(rec)) break;
        if (rec.size() != ncols)
            csv_error(line, "expected " + std::to_string(ncols) + " fields, got " + std::to_string(rec.size()));
        lines.push_back(line);
        for (std::size_t j = 0; j < source.size(); ++j) fields[j].push_back(rec[source[j]]);
    }

    const std::unordered_set<std::string> na(opt.na_values.begin(), opt.na_values.end());
    std::vector<Column> cols;
    for (std::size_t j = 0; j < out_names.size(); ++j) {
        ColumnBuilder b(data, fields[j], lines, na, out_names[j]);
        auto forced = opt.dtypes.find(out_names[j]);
        const DType dt = forced != opt.dtypes.end() ? forced->second : opt.all_dtype ? *opt.all_dtype : b.infer();
        cols.push_back(b.build(dt));
    }
    return {out_names, cols};
}

}  // namespace sd
