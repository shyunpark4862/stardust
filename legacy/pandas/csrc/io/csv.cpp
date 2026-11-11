#include "io/csv.hpp"

#include "ops/parse.hpp"

#include <nanobind/nanobind.h>

#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <string_view>
#include <unordered_set>

namespace nb = nanobind;

namespace sd {

namespace {

// Where a field's text is in the input. Quoted fields point inside the
// quotes; kEscaped says the text still holds "" pairs to turn into ".
struct Field {  // 16 bytes: a million rows of fields are many megabytes to write (M10: was 24)
    std::int64_t start;
    std::uint32_t length;
    std::uint32_t flags;
};
constexpr std::uint8_t kQuoted = 1, kEscaped = 2;

[[noreturn]] void csv_error(std::int64_t line, const std::string& what) {
    throw nb::value_error(("read_csv: line " + std::to_string(line) + ": " + what).c_str());
}

// The 1-based line of byte `offset`: errors are rare, so lines are counted
// only when one happens (M10; M9 counted them for every record).
std::int64_t line_at(const char* p, std::int64_t offset) {
    std::int64_t line = 1;
    for (std::int64_t i = 0; i < offset; ++i) line += p[i] == '\n';
    return line;
}

// A 64-bit mask with bit i set where byte i of the 64 equals c: four NEON
// compares, each byte ANDed with its bit's weight, then pairwise adds fold
// 64 bytes into 8.
std::uint64_t mask64(const uint8x16_t v[4], std::uint8_t c) {
    static const uint8x16_t weight = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t k = vdupq_n_u8(c);
    uint8x16_t m[4];
    for (int j = 0; j < 4; ++j) m[j] = vandq_u8(vceqq_u8(v[j], k), weight);
    uint8x16_t s = vpaddq_u8(vpaddq_u8(m[0], m[1]), vpaddq_u8(m[2], m[3]));
    s = vpaddq_u8(s, s);
    return vgetq_lane_u64(vreinterpretq_u64_u8(s), 0);
}

// Bit i of the result is the XOR of bits 0..i of x: a carry-less multiply
// by all ones (PMULL). Applied to the quote mask, it is 1 exactly on the
// bytes inside quotes (a "" pair toggles twice and stays inside).
std::uint64_t prefix_xor(std::uint64_t x) {
    return vgetq_lane_u64(vreinterpretq_u64_p128(vmull_p64(x, ~std::uint64_t{0})), 0);
}

// RFC 4180 tokenizer over a structural index (M10, the simdjson/simdcsv
// idea). Each 64-byte block becomes three bitmasks -- quotes, separators,
// line feeds -- and prefix_xor of the quotes marks the bytes inside quotes.
// Separators and line feeds outside quotes are the only field ends, and the
// set bits of that mask are visited one by one: no per-byte branch. Quote
// rules are then checked per field, through the quote mask, only for the
// fields that hold a quote.
class Tokenizer {
public:
    Tokenizer(const char* p, std::size_t n, char sep) : p_(p), n_(static_cast<std::int64_t>(n)), sep_(sep) {
        if (n_ >= 3 && std::memcmp(p_, "\xEF\xBB\xBF", 3) == 0) field_start_ = 3;  // UTF-8 byte order mark
        quotes_.reserve(static_cast<std::size_t>(n_ / 64 + 1));
    }

    // The byte where the next record starts (for error messages).
    std::int64_t offset() const { return field_start_; }

    // The next record's fields, each handed to sink(index, field) as it is
    // found (M10: no per-record vector in between). Returns the number of
    // fields, or -1 at the end of the input.
    template <class Sink>
    std::int64_t next(Sink&& sink) {
        if (done_ || field_start_ >= n_) return -1;
        const std::int64_t record = field_start_;
        for (std::int64_t count = 0;; ++count) {
            const std::int64_t at = next_structural();
            if (at < 0) {  // the end of the input ends the last record
                if (inside_) csv_error(line_at(p_, record), "a quoted field is never closed");
                sink(count, field(field_start_, n_, record));
                field_start_ = n_;
                done_ = true;
                return count + 1;
            }
            sink(count, field(field_start_, at, record));
            field_start_ = at + 1;
            if (p_[at] == '\n') return count + 1;
        }
    }

private:
    // The next separator or line feed outside quotes, or -1.
    std::int64_t next_structural() {
        while (bits_ == 0) {
            if (block_ * 64 >= n_) return -1;
            scan_block();
        }
        const std::int64_t at = (block_ - 1) * 64 + __builtin_ctzll(bits_);
        bits_ &= bits_ - 1;
        return at;
    }

    void scan_block() {
        const std::int64_t base = block_ * 64;
        std::uint8_t tail[64];
        const std::uint8_t* src = reinterpret_cast<const std::uint8_t*>(p_ + base);
        if (base + 64 > n_) {  // the last, partial block: padded with zero bytes
            std::memset(tail, 0, 64);
            std::memcpy(tail, src, static_cast<std::size_t>(n_ - base));
            src = tail;
        }
        uint8x16_t v[4];
        for (int j = 0; j < 4; ++j) v[j] = vld1q_u8(src + 16 * j);
        const std::uint64_t q = mask64(v, '"');
        const std::uint64_t in = prefix_xor(q) ^ (inside_ ? ~std::uint64_t{0} : 0);
        inside_ = (in >> 63) & 1;
        quotes_.push_back(q);
        bits_ = (mask64(v, static_cast<std::uint8_t>(sep_)) | mask64(v, '\n')) & ~in;
        if (base < 3 && field_start_ == 3) bits_ &= ~std::uint64_t{7};  // never inside the BOM
        ++block_;
    }

    bool any_quote(std::int64_t from, std::int64_t to) const {  // a quote in [from, to)?
        for (std::int64_t w = from >> 6; w <= (to - 1) >> 6 && from < to; ++w) {
            std::uint64_t m = quotes_[w];
            const std::int64_t lo = std::max(from, w * 64) - w * 64, hi = std::min(to, w * 64 + 64) - w * 64;
            m &= (hi == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << hi) - 1) & ~((std::uint64_t{1} << lo) - 1);
            if (m) return true;
        }
        return false;
    }

    // The field in bytes [from, to), checked against the quoting rules.
    Field field(std::int64_t from, std::int64_t to, std::int64_t record) {
        if (to > from && p_[to - 1] == '\r') --to;  // a \r\n line end (or a last \r at the end of the input)
        if (to - from >= (std::int64_t{1} << 32)) csv_error(line_at(p_, record), "a field of 4 GB or more");
        if (!any_quote(from, to)) return {from, static_cast<std::uint32_t>(to - from), 0};
        if (p_[from] != '"') csv_error(line_at(p_, record), "a quote inside an unquoted field");
        if (to - from < 2 || p_[to - 1] != '"')
            csv_error(line_at(p_, record), "a closing quote must be followed by the separator or the end of the line");
        std::uint8_t flags = kQuoted;
        for (std::int64_t i = from + 1; i < to - 1; ++i) {
            if (p_[i] != '"') continue;
            if (i + 1 >= to - 1 || p_[i + 1] != '"')
                csv_error(line_at(p_, record), "a closing quote must be followed by the separator or the end of the line");
            flags |= kEscaped;  // "" is one quote
            ++i;
        }
        return {from + 1, static_cast<std::uint32_t>(to - from - 2), flags};
    }

    const char* p_;
    std::int64_t n_;
    char sep_;
    std::int64_t field_start_ = 0, block_ = 0;
    std::uint64_t bits_ = 0;
    bool inside_ = false, done_ = false;
    std::vector<std::uint64_t> quotes_;  // the quote mask of every block scanned so far
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
    ColumnBuilder(const char* base, const std::vector<Field>& fields, const std::vector<std::int64_t>& starts,
                  const std::unordered_set<std::string>& na, const std::string& name)
        : base_(base), f_(fields), starts_(starts), na_(na), name_(name) {}

    // API.md §19: NA is the unquoted empty field and the na_values texts.
    bool is_na(std::size_t i) {
        const Field& f = f_[i];
        if (f.length == 0 && !(f.flags & kQuoted)) return true;
        return !na_.empty() && na_.count(std::string(text(i)));
    }

    std::string_view text(std::size_t i) { return text_of(base_, f_[i], scratch_); }

    // Infers the dtype (int64, then float64, then bool, then string; API.md
    // §19) and builds the column in the same pass over the fields (M10: the
    // values parsed while inferring are kept, not parsed again). ints are
    // parsed until a field is not an int; then the ints so far are converted
    // to double -- exactly, or the column is text -- and doubles are parsed.
    Column infer_build() {
        enum class Kind { Int, Float, Text };
        const std::int64_t n = static_cast<std::int64_t>(f_.size());
        Kind kind = Kind::Int;
        bool any = false;
        std::vector<std::uint8_t> valid(static_cast<std::size_t>(n));
        std::vector<std::int64_t> iv(static_cast<std::size_t>(n), 0);
        std::vector<double> dv;
        for (std::int64_t i = 0; i < n && kind != Kind::Text; ++i) {
            if (is_na(i)) continue;
            valid[i] = 1;
            any = true;
            const std::string_view s = text(i);
            std::int64_t v;
            if (kind == Kind::Int) {
                const int r = text::parse_int64(s, v);
                if (r == 1) {
                    iv[i] = v;
                    continue;
                }
                if (r == -1) {  // beyond int64: text (API.md §19)
                    kind = Kind::Text;
                    continue;
                }
                double d;
                if (!text::parse_double(s, d)) {
                    kind = Kind::Text;
                    continue;
                }
                kind = Kind::Float;  // the ints so far become doubles, if exactly
                dv.assign(static_cast<std::size_t>(n), 0.0);
                for (std::int64_t k = 0; k < i && kind == Kind::Float; ++k) {
                    if (!valid[k]) continue;
                    if (!int_is_exact_double(iv[k])) kind = Kind::Text;
                    dv[k] = static_cast<double>(iv[k]);
                }
                if (kind == Kind::Float) dv[i] = d;
                continue;
            }
            const int r = text::parse_int64(s, v);
            if (r != 0) {  // an int in a float column: exactly, or text
                if (r < 0 || !int_is_exact_double(v)) kind = Kind::Text;
                else dv[i] = static_cast<double>(v);
            } else if (!text::parse_double(s, dv[i])) {
                kind = Kind::Text;
            }
        }
        if (!any) return build(DType::Float64);  // every value NA: float64 (API.md §10)
        if (kind == Kind::Text) return build(all_bool() ? DType::Bool : DType::String);
        const DType dt = kind == Kind::Int ? DType::Int64 : DType::Float64;
        Column col = make_fixed(dt, n);
        if (dt == DType::Int64) std::memcpy(col.values->data(), iv.data(), static_cast<std::size_t>(n) * 8);
        else std::memcpy(col.values->data(), dv.data(), static_cast<std::size_t>(n) * 8);
        std::uint8_t* bits = add_validity(col);
        for (std::int64_t i = 0; i < n; ++i)
            if (!valid[i]) bitmap::clear(bits, i);
        finish_fixed(col);
        return col;
    }

    bool all_bool() {
        for (std::int64_t i = 0, n = static_cast<std::int64_t>(f_.size()); i < n; ++i) {
            if (is_na(i)) continue;
            const std::string_view s = text(i);
            if (s != "True" && s != "False") return false;
        }
        return true;
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
                if (text::parse_int64(s, v) != 1) fail(i, s, dt);
                reinterpret_cast<std::int64_t*>(col.values->data())[i] = v;
                break;
            }
            case DType::Float64:
                if (!text::parse_double(s, reinterpret_cast<double*>(col.values->data())[i])) fail(i, s, dt);
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
                if (!valid_utf8(s)) csv_error(line_at(base_, starts_[i]), "column '" + name_ + "' is not valid UTF-8");
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
        csv_error(line_at(base_, starts_[i]), "column '" + name_ + "': cannot read '" + std::string(s) + "' as " + dtype_name(dt));
    }

    const char* base_;
    const std::vector<Field>& f_;
    const std::vector<std::int64_t>& starts_;  // the byte where each row's record starts
    const std::unordered_set<std::string>& na_;
    const std::string& name_;
    std::string scratch_;
};

}  // namespace

std::pair<std::vector<std::string>, std::vector<Column>> read_csv(const char* data, std::size_t size,
                                                                   const CsvOptions& opt) {
    Tokenizer tok(data, size, opt.sep);
    std::vector<Field> rec;
    auto collect = [&rec](std::int64_t, const Field& f) { rec.push_back(f); };
    std::string scratch;

    std::vector<std::string> names = opt.names;
    if (opt.header) {
        if (tok.next(collect) < 0) csv_error(1, "the file is empty but header=True");
        const std::int64_t line = 1;
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
    std::vector<std::int64_t> starts;
    const std::size_t ncols = names.size();
    // slot[i]: the output column that file column i feeds, or -1.
    std::vector<std::int64_t> slot(ncols, -1);
    for (std::size_t j = 0; j < source.size(); ++j) slot[source[j]] = static_cast<std::int64_t>(j);
    const std::int64_t width = static_cast<std::int64_t>(ncols);
    auto place = [&](std::int64_t i, const Field& f) {
        if (i < width && slot[i] >= 0) fields[slot[i]].push_back(f);
    };
    while (opt.nrows < 0 || static_cast<std::int64_t>(starts.size()) < opt.nrows) {
        const std::int64_t start = tok.offset();
        const std::int64_t count = tok.next(place);
        if (count < 0) break;
        if (count != width)
            csv_error(line_at(data, start), "expected " + std::to_string(ncols) + " fields, got " + std::to_string(count));
        starts.push_back(start);
    }

    const std::unordered_set<std::string> na(opt.na_values.begin(), opt.na_values.end());
    std::vector<Column> cols;
    for (std::size_t j = 0; j < out_names.size(); ++j) {
        ColumnBuilder b(data, fields[j], starts, na, out_names[j]);
        auto forced = opt.dtypes.find(out_names[j]);
        if (forced != opt.dtypes.end()) cols.push_back(b.build(forced->second));
        else if (opt.all_dtype) cols.push_back(b.build(*opt.all_dtype));
        else cols.push_back(b.infer_build());
    }
    return {out_names, cols};
}

}  // namespace sd
