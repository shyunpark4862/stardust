#include "frame/io/csv.hpp"

#include "frame/ops/parse.hpp"
#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <optional>
#include <cstring>
#include <string_view>
#include <unordered_set>

namespace nb = nanobind;

namespace sd::frame {

namespace {

// Where a field's text is in the input. Quoted fields point inside the
// quotes; kEscaped says the text still holds "" pairs to turn into ".
struct Field {  // 16 bytes, was 24: a million rows of fields are many megabytes to write (docs/design/frame.md §10.3)
    std::int64_t start;
    std::uint32_t length;
    std::uint32_t flags;
};
constexpr std::uint8_t kQuoted = 1, kEscaped = 2;

[[noreturn]] void csv_error(std::int64_t line, const std::string& what) {
    throw nb::value_error(("line " + std::to_string(line) + ": " + what).c_str());
}

// The 1-based line of byte `offset`: errors are rare, so lines are counted
// only when one happens, not for every record (docs/design/frame.md §10.3).
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

// RFC 4180 tokenizer over a structural index (the simdjson/simdcsv idea,
// docs/design/frame.md §10.2). Each 64-byte block becomes three bitmasks -- quotes, separators,
// line feeds -- and prefix_xor of the quotes marks the bytes inside quotes.
// Separators and line feeds outside quotes are the only field ends, and the
// set bits of that mask are visited one by one: no per-byte branch. Quote
// rules are then checked per field, through the quote mask, only for the
// fields that hold a quote.
//
// A tokenizer starts at byte `begin`, which must be a record start (each
// chunk of a file gets its own tokenizer, docs/design/frame.md §10.6); blocks are counted from
// there. Only a tokenizer at byte 0 skips a byte order mark.
class Tokenizer {
public:
    Tokenizer(const char* p, std::size_t n, char sep, std::int64_t begin = 0)
        : p_(p), n_(static_cast<std::int64_t>(n)), sep_(sep), origin_(begin), field_start_(begin) {
        if (begin == 0 && n_ >= 3 && std::memcmp(p_, "\xEF\xBB\xBF", 3) == 0) {  // UTF-8 byte order mark
            field_start_ = 3;
            bom_ = true;
        }
    }

    // The byte where the next record starts (for error messages).
    std::int64_t offset() const { return field_start_; }

    // The next record's fields, each handed to sink(index, field) as it is
    // found (no per-record vector in between, docs/design/frame.md §10.3). Returns the number of
    // fields, or -1 at the end of the input.
    template <class Sink>
    std::int64_t next(Sink&& sink) {
        if (done_ || field_start_ >= n_) return -1;
        const std::int64_t record = field_start_;
        for (std::int64_t count = 0;; ++count) {
            const std::int64_t at = next_structural();
            if (at < 0) {  // the end of the input ends the last record
                if (inside_) csv_error(line_at(p_, record), "a quoted field is never closed");
                sink(count, field(field_start_, n_, record, false));
                field_start_ = n_;
                done_ = true;
                return count + 1;
            }
            const bool line_end = p_[at] == '\n';
            sink(count, field(field_start_, at, record, line_end));
            field_start_ = at + 1;
            if (line_end) return count + 1;
        }
    }

private:
    // The next separator or line feed outside quotes, or -1.
    std::int64_t next_structural() {
        while (bits_ == 0) {
            if (origin_ + block_ * 64 >= n_) return -1;
            scan_block();
        }
        const std::int64_t at = origin_ + (block_ - 1) * 64 + __builtin_ctzll(bits_);
        bits_ &= bits_ - 1;
        return at;
    }

    void scan_block() {
        const std::int64_t base = origin_ + block_ * 64;
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
        if (bom_ && block_ == 0) bits_ &= ~std::uint64_t{7};  // never inside the BOM
        ++block_;
    }

    bool any_quote(std::int64_t from, std::int64_t to) const {  // a quote in [from, to)?
        from -= origin_;  // block numbers count from the origin
        to -= origin_;
        for (std::int64_t w = from >> 6; w <= (to - 1) >> 6 && from < to; ++w) {
            std::uint64_t m = quotes_[w];
            const std::int64_t lo = std::max(from, w * 64) - w * 64, hi = std::min(to, w * 64 + 64) - w * 64;
            m &= (hi == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << hi) - 1) & ~((std::uint64_t{1} << lo) - 1);
            if (m) return true;
        }
        return false;
    }

    // The field in bytes [from, to), checked against the quoting rules.
    // `line_end`: the field ends at a line feed, so a \r just before it is
    // the \r\n line end. Anywhere else a \r is a byte of the value
    // (docs/api/frame.md §10.2: line ends are \n and \r\n only); after a closing quote it is an
    // error like any other byte.
    Field field(std::int64_t from, std::int64_t to, std::int64_t record, bool line_end) {
        if (line_end && to > from && p_[to - 1] == '\r') --to;
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
    std::int64_t origin_;  // block 0 starts here
    std::int64_t field_start_, block_ = 0;
    std::uint64_t bits_ = 0;
    bool inside_ = false, done_ = false, bom_ = false;
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

// --- the file in chunks (docs/design/frame.md §10.6) ---------------------
//
// The body (after the header) is cut into chunks of kChunkBytes when the
// pool can run in parallel (else one chunk: the result is the same).
//  1. Each chunk counts its quotes. Whether a byte is inside quotes is the
//     parity of the quotes before it -- exactly what the serial tokenizer
//     tracks -- so the running parity gives every chunk's state at its start.
//  2. A chunk owns the records that start inside it: it finds its first
//     record start (after a line feed outside quotes), tokenizes from there
//     and finishes the last record even past its end.
//  3. Each column is built chunk by chunk, straight into the final buffers;
//     the chunks' inferred types are then merged (the rules of
//     docs/api/frame.md §10.4 hold for the whole column).
// Errors are the serial ones: the first chunk with a tokenizing error
// reports it; then columns in order, the first bad row of a column.

constexpr std::int64_t kChunkBytes = std::int64_t{1} << 19;

bool odd_quotes(const char* p, std::int64_t from, std::int64_t to) {
    std::int64_t count = 0;
    for (std::int64_t i = from; i < to; ++i) count += p[i] == '"';
    return count & 1;
}

// The first record start in [from, size) when byte `from` is inside quotes
// or not; size when there is none.
std::int64_t first_record(const char* p, std::int64_t size, std::int64_t from, bool inside) {
    if (p[from - 1] == '\n' && !inside) return from;  // a line feed is no quote: its state is from's
    for (std::int64_t i = from; i < size; ++i) {
        if (p[i] == '"') inside = !inside;
        else if (p[i] == '\n' && !inside) return i + 1;
    }
    return size;
}

struct Chunk {
    std::vector<std::vector<Field>> fields;  // per output column, one per record
    std::vector<std::int64_t> starts;        // the byte where each record starts
    std::string error;                       // the chunk's tokenizing error, if any
};

class ColumnReader {
public:
    ColumnReader(const char* base, const std::vector<Chunk>& chunks, const std::vector<std::int64_t>& row0,
                 std::size_t column, const std::unordered_set<std::string>& na, const std::string& name)
        : base_(base), chunks_(chunks), row0_(row0), j_(column), na_(na), name_(name), n_(row0.back()) {}

    // docs/api/frame.md §10.4: the column inferred (int64, then float64, then bool, then
    // string), or read as `forced`.
    Column read(std::optional<DType> forced) {
        if (forced) return build(*forced);
        enum Kind : std::uint8_t { Int, Float, Text };
        Column col = make_fixed(DType::Int64, n_);  // int64 and double values share these 8 bytes per row
        std::vector<std::uint8_t> valid(static_cast<std::size_t>(n_));
        std::vector<std::uint8_t> kinds(chunks_.size()), any(chunks_.size());
        auto* iv = reinterpret_cast<std::int64_t*>(col.values->data());
        auto* dv = reinterpret_cast<double*>(col.values->data());
        each_chunk([&](std::size_t c, const std::vector<Field>& f, std::int64_t r0) {
            std::string scratch;
            Kind kind = Int;
            for (std::size_t i = 0; i < f.size() && kind != Text; ++i) {
                const std::int64_t row = r0 + static_cast<std::int64_t>(i);
                if (is_na(f[i], scratch)) {
                    valid[row] = 0;
                    iv[row] = 0;
                    continue;
                }
                valid[row] = 1;
                any[c] = 1;
                const std::string_view s = text_of(base_, f[i], scratch);
                std::int64_t v;
                if (kind == Int) {
                    const int r = text::parse_int64(s, v);
                    if (r == 1) {
                        iv[row] = v;
                        continue;
                    }
                    double d;
                    if (r == -1 || !text::parse_double(s, d)) {  // beyond int64, or no number: text
                        kind = Text;
                        continue;
                    }
                    kind = Float;  // the chunk's ints so far become doubles, if exactly
                    for (std::int64_t k = r0; k < row && kind == Float; ++k) {
                        if (!valid[k]) continue;
                        if (!int_is_exact_double(iv[k])) kind = Text;
                        else dv[k] = static_cast<double>(iv[k]);
                    }
                    dv[row] = d;
                    continue;
                }
                const int r = text::parse_int64(s, v);
                if (r != 0) {  // an int in a float column: exactly, or text
                    if (r < 0 || !int_is_exact_double(v)) kind = Text;
                    else dv[row] = static_cast<double>(v);
                } else if (!text::parse_double(s, dv[row])) {
                    kind = Text;
                }
            }
            kinds[c] = kind;
        });
        if (std::find(any.begin(), any.end(), 1) == any.end()) return build(DType::Float64);  // all NA: float64 (docs/api/frame.md §10.4)
        Kind kind = Int;
        for (std::uint8_t k : kinds) kind = std::max(kind, static_cast<Kind>(k));
        if (kind == Float) {  // chunks of ints become doubles, if exactly
            std::atomic<bool> inexact{false};
            each_chunk([&](std::size_t c, const std::vector<Field>& f, std::int64_t r0) {
                if (kinds[c] != Int) return;
                for (std::int64_t row = r0; row < r0 + static_cast<std::int64_t>(f.size()); ++row) {
                    if (!valid[row]) continue;
                    if (!int_is_exact_double(iv[row])) inexact = true;
                    dv[row] = static_cast<double>(iv[row]);
                }
            });
            if (inexact) kind = Text;
        }
        if (kind == Text) return build(all_bool() ? DType::Bool : DType::String);
        if (kind == Float) col.dtype = DType::Float64;
        set_validity(col, valid);
        return col;
    }

private:
    template <class F>
    void each_chunk(F f) {
        parallel::parallel_for(static_cast<std::int64_t>(chunks_.size()), [&](std::int64_t c) {
            const auto k = static_cast<std::size_t>(c);
            f(k, chunks_[k].fields[j_], row0_[k]);
        });
    }

    // docs/api/frame.md §10.3: NA is the unquoted empty field and the na_values texts.
    bool is_na(const Field& f, std::string& scratch) const {
        if (f.length == 0 && !(f.flags & kQuoted)) return true;
        return !na_.empty() && na_.count(std::string(text_of(base_, f, scratch)));
    }

    const Field& field_of(std::int64_t row, std::int64_t& start) const {
        const auto c = static_cast<std::size_t>(std::upper_bound(row0_.begin(), row0_.end(), row) - row0_.begin() - 1);
        start = chunks_[c].starts[static_cast<std::size_t>(row - row0_[c])];
        return chunks_[c].fields[j_][static_cast<std::size_t>(row - row0_[c])];
    }

    bool all_bool() {
        std::atomic<bool> other{false};
        each_chunk([&](std::size_t, const std::vector<Field>& f, std::int64_t) {
            std::string scratch;
            for (std::size_t i = 0; i < f.size() && !other.load(std::memory_order_relaxed); ++i) {
                if (is_na(f[i], scratch)) continue;
                const std::string_view s = text_of(base_, f[i], scratch);
                if (s != "True" && s != "False") other = true;
            }
        });
        return !other;
    }

    // valid[0, n) as the column's bitmap; then the NA count and zeroed NA rows.
    void set_validity(Column& col, const std::vector<std::uint8_t>& valid) {
        col.validity = Buffer::allocate(bitmap::num_bytes(n_), false);
        auto* bits = reinterpret_cast<std::uint8_t*>(col.validity->data());
        for_chunks(n_, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t b = lo / 8; b < bitmap::num_bytes(hi); ++b) {
                unsigned v = 0;
                for (std::int64_t i = b * 8; i < std::min(hi, b * 8 + 8); ++i) v |= static_cast<unsigned>(valid[i]) << (i & 7);
                bits[b] = static_cast<std::uint8_t>(v);
            }
        });
        finish_fixed(col);
    }

    Column build(DType dt) {
        if (dt == DType::String) return build_strings();
        Column col = make_fixed(dt, n_);
        std::vector<std::uint8_t> valid(static_cast<std::size_t>(n_));
        parallel::FirstRow bad;
        each_chunk([&](std::size_t, const std::vector<Field>& f, std::int64_t r0) {
            std::string scratch;
            for (std::size_t i = 0; i < f.size(); ++i) {
                const std::int64_t row = r0 + static_cast<std::int64_t>(i);
                if (bad.failed_before(row)) return;
                const bool na = is_na(f[i], scratch);
                valid[row] = !na;
                if (na) continue;
                const std::string_view s = text_of(base_, f[i], scratch);
                bool ok = true;
                switch (dt) {
                case DType::Int64: ok = text::parse_int64(s, reinterpret_cast<std::int64_t*>(col.values->data())[row]) == 1; break;
                case DType::Float64: ok = text::parse_double(s, reinterpret_cast<double*>(col.values->data())[row]); break;
                case DType::Bool:
                    ok = s == "True" || s == "False";
                    reinterpret_cast<std::uint8_t*>(col.values->data())[row] = s == "True";
                    break;
                case DType::String: break;
                case DType::Complex128: break;  // not a frame dtype
                }
                if (!ok) {
                    bad.offer(row);
                    return;
                }
            }
        });
        if (bad.any()) {
            std::int64_t start;
            std::string scratch;
            const std::string s(text_of(base_, field_of(bad.row(), start), scratch));
            csv_error(line_at(base_, start), "column '" + name_ + "': cannot read '" + s + "' as " + dtype_name(dt));
        }
        set_validity(col, valid);
        return col;
    }

    Column build_strings() {
        Column col;
        col.dtype = DType::String;
        col.length = n_;
        col.offsets = Buffer::allocate((n_ + 1) * 8, false);
        auto* offs = reinterpret_cast<std::int64_t*>(col.offsets->data());
        offs[0] = 0;
        std::vector<std::uint8_t> valid(static_cast<std::size_t>(n_));
        std::vector<std::int64_t> bytes(chunks_.size() + 1, 0);  // per chunk, then where each starts
        parallel::FirstRow bad;
        // Pass 1: each chunk's bytes, checking UTF-8.
        each_chunk([&](std::size_t c, const std::vector<Field>& f, std::int64_t r0) {
            std::string scratch;
            std::int64_t pos = 0;
            for (std::size_t i = 0; i < f.size(); ++i) {
                const std::int64_t row = r0 + static_cast<std::int64_t>(i);
                if (bad.failed_before(row)) return;
                if (is_na(f[i], scratch)) continue;
                const std::string_view s = text_of(base_, f[i], scratch);
                if (!valid_utf8(s)) {
                    bad.offer(row);
                    return;
                }
                valid[row] = 1;
                pos += static_cast<std::int64_t>(s.size());
            }
            bytes[c + 1] = pos;
        });
        if (bad.any()) {
            std::int64_t start;
            field_of(bad.row(), start);
            csv_error(line_at(base_, start), "column '" + name_ + "' is not valid UTF-8");
        }
        for (std::size_t c = 0; c < chunks_.size(); ++c) bytes[c + 1] += bytes[c];
        col.values = Buffer::allocate(bytes.back(), false);
        auto* out = reinterpret_cast<char*>(col.values->data());
        // Pass 2: the bytes and the final offsets, from the chunk's start.
        each_chunk([&](std::size_t c, const std::vector<Field>& f, std::int64_t r0) {
            std::string scratch;
            std::int64_t pos = bytes[c];
            for (std::size_t i = 0; i < f.size(); ++i) {
                const std::int64_t row = r0 + static_cast<std::int64_t>(i);
                if (valid[row]) {
                    const std::string_view s = text_of(base_, f[i], scratch);
                    std::memcpy(out + pos, s.data(), s.size());
                    pos += static_cast<std::int64_t>(s.size());
                }
                offs[row + 1] = pos;
            }
        });
        col.validity = Buffer::allocate(bitmap::num_bytes(n_), false);
        auto* bits = reinterpret_cast<std::uint8_t*>(col.validity->data());
        std::int64_t count = 0;
        for (std::int64_t i = 0; i < n_; ++i) count += valid[i];
        for_chunks(n_, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t b = lo / 8; b < bitmap::num_bytes(hi); ++b) {
                unsigned v = 0;
                for (std::int64_t i = b * 8; i < std::min(hi, b * 8 + 8); ++i) v |= static_cast<unsigned>(valid[i]) << (i & 7);
                bits[b] = static_cast<std::uint8_t>(v);
            }
        });
        col.null_count = n_ - count;
        if (col.null_count == 0) col.validity.reset();
        return col;
    }

    const char* base_;
    const std::vector<Chunk>& chunks_;
    const std::vector<std::int64_t>& row0_;  // the first row of each chunk; back(): the row count
    std::size_t j_;
    const std::unordered_set<std::string>& na_;
    const std::string& name_;
    std::int64_t n_;
};

}  // namespace

std::pair<std::vector<std::string>, std::vector<Column>> read_csv(const char* data, std::size_t size,
                                                                   const CsvOptions& opt) {
    // Nothing below touches Python objects: other Python threads run
    // meanwhile (docs/design/frame.md §10.6).
    Unlocked unlocked(static_cast<std::int64_t>(size));
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
    if (names.empty()) throw nb::value_error("'names' is required when header=False");
    {
        std::unordered_set<std::string> seen;
        for (const auto& k : names)
            if (!seen.insert(k).second) throw nb::value_error(("column name '" + k + "' repeats").c_str());
    }

    // Which file column feeds each output column.
    std::vector<std::string> out_names = opt.usecols.empty() ? names : opt.usecols;
    std::vector<std::int64_t> source(out_names.size());
    for (std::size_t j = 0; j < out_names.size(); ++j) {
        auto it = std::find(names.begin(), names.end(), out_names[j]);
        if (it == names.end()) throw nb::key_error(("'usecols' names a column that is not in the file: '" + out_names[j] + "'").c_str());
        source[j] = it - names.begin();
    }
    const std::size_t ncols = names.size();
    std::vector<std::int64_t> slot(ncols, -1);  // slot[i]: the output column file column i feeds, or -1
    for (std::size_t j = 0; j < source.size(); ++j) slot[source[j]] = static_cast<std::int64_t>(j);
    const std::int64_t width = static_cast<std::int64_t>(ncols);

    // The chunks. One chunk -- the serial reader -- with nrows (records are
    // read in order, up to nrows) and when the pool cannot run in parallel:
    // the result does not depend on the chunks, so they may follow the
    // thread count (docs/design/frame.md §11.5, as for factorize).
    const auto n = static_cast<std::int64_t>(size);
    const std::int64_t body = tok.offset();
    const bool serial = opt.nrows >= 0 || !parallel::available();
    const std::int64_t nchunks = serial ? 1 : std::max<std::int64_t>(1, (n - body + kChunkBytes - 1) / kChunkBytes);
    auto chunk_begin = [&](std::int64_t c) { return std::min(n, body + c * kChunkBytes); };
    auto chunk_end = [&](std::int64_t c) { return c + 1 == nchunks ? n : chunk_begin(c + 1); };
    std::vector<std::uint8_t> odd(static_cast<std::size_t>(nchunks));
    if (nchunks > 1)
        parallel::parallel_for(nchunks, [&](std::int64_t c) { odd[c] = odd_quotes(data, chunk_begin(c), chunk_end(c)); });
    std::vector<std::uint8_t> inside(static_cast<std::size_t>(nchunks), 0);  // inside quotes where chunk c starts
    for (std::int64_t c = 1; c < nchunks; ++c) inside[c] = inside[c - 1] ^ odd[c - 1];

    std::vector<Chunk> chunks(static_cast<std::size_t>(nchunks));
    parallel::parallel_for(nchunks, [&](std::int64_t c) {
        Chunk& ch = chunks[static_cast<std::size_t>(c)];
        ch.fields.resize(out_names.size());
        const std::int64_t end = chunk_end(c);
        const std::int64_t first = c == 0 ? body : first_record(data, n, chunk_begin(c), inside[c]);
        if (first >= end || first >= n) return;
        try {
            Tokenizer t(data, size, opt.sep, first);
            auto place = [&](std::int64_t i, const Field& f) {
                if (i < width && slot[i] >= 0) ch.fields[slot[i]].push_back(f);
            };
            while (opt.nrows < 0 || static_cast<std::int64_t>(ch.starts.size()) < opt.nrows) {
                const std::int64_t start = t.offset();
                if (start >= end) break;  // the next chunk's record
                const std::int64_t count = t.next(place);
                if (count < 0) break;
                if (count != width)
                    csv_error(line_at(data, start), "expected " + std::to_string(ncols) + " fields, got " + std::to_string(count));
                ch.starts.push_back(start);
            }
        } catch (const std::exception& e) {
            ch.error = e.what();
        }
    });
    for (const Chunk& ch : chunks)  // the first error in file order
        if (!ch.error.empty()) throw nb::value_error(ch.error.c_str());
    std::vector<std::int64_t> row0(static_cast<std::size_t>(nchunks) + 1, 0);
    for (std::int64_t c = 0; c < nchunks; ++c) row0[c + 1] = row0[c] + static_cast<std::int64_t>(chunks[c].starts.size());

    const std::unordered_set<std::string> na(opt.na_values.begin(), opt.na_values.end());
    std::vector<Column> cols;
    for (std::size_t j = 0; j < out_names.size(); ++j) {
        ColumnReader reader(data, chunks, row0, j, na, out_names[j]);
        std::optional<DType> forced;
        if (auto it = opt.dtypes.find(out_names[j]); it != opt.dtypes.end()) forced = it->second;
        else if (opt.all_dtype) forced = opt.all_dtype;
        cols.push_back(reader.read(forced));
    }
    return {out_names, cols};
}

}  // namespace sd::frame
