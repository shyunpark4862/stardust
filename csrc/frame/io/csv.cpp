#include "frame/io/csv.hpp"

#include "frame/ops/parse.hpp"
#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <functional>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd::frame {

namespace {

// Where a field's text is in the input. Quoted fields point inside the
// quotes; kEscaped says the text still holds "" pairs to turn into ".
struct Field {  // 16 bytes: a million rows of fields are many megabytes to write (docs/design/frame.md §10.3)
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
        // any_quote looks only inside this record's fields: the quote masks of
        // the blocks before it are dropped now and then, so a long input keeps
        // a few masks, not one per block (docs/design/frame.md §10.6).
        const std::int64_t keep = (record - origin_) >> 6;
        if (keep - quotes_base_ >= kKeepQuoteBlocks) {
            quotes_.erase(quotes_.begin(), quotes_.begin() + (keep - quotes_base_));
            quotes_base_ = keep;
        }
        for (std::int64_t count = 0;; ++count) {
            const std::int64_t at = next_structural();
            if (at < 0) {  // the end of the input ends the last record
                // An odd quote count runs to the end of the input. If the field
                // holding the open quote does not start with one, the mistake is
                // the same as with an even count: a quote inside an unquoted field.
                if (inside_)
                    csv_error(
                        line_at(p_, record),
                        p_[field_start_] == '"' ? "a quoted field is never closed" : "a quote inside an unquoted field"
                    );
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
        // The zero padding of a partial block is not input: sep="\0" would match it.
        if (base + 64 > n_) bits_ &= (std::uint64_t{1} << (n_ - base)) - 1;
        if (bom_ && block_ == 0) bits_ &= ~std::uint64_t{7};  // never inside the BOM
        ++block_;
    }

    bool any_quote(std::int64_t from, std::int64_t to) const {  // a quote in [from, to)?
        from -= origin_;  // block numbers count from the origin
        to -= origin_;
        for (std::int64_t w = from >> 6; w <= (to - 1) >> 6 && from < to; ++w) {
            std::uint64_t m = quotes_[static_cast<std::size_t>(w - quotes_base_)];
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
                csv_error(
                    line_at(p_, record), "a closing quote must be followed by the separator or the end of the line"
                );
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
    static constexpr std::int64_t kKeepQuoteBlocks = 1024;  // masks dropped at most this many at a time
    std::vector<std::uint64_t> quotes_;  // the quote masks of blocks quotes_base_, quotes_base_ + 1, ...
    std::int64_t quotes_base_ = 0;
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
    for (std::uint64_t w; i + 8 <= s.size(); i += 8) {  // ASCII eight bytes at a time
        std::memcpy(&w, s.data() + i, 8);
        if (w & 0x8080808080808080u) break;
    }
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
//  3. While a chunk is tokenized, every field goes straight into its
//     column's Part of the chunk: a number, or text once the chunk's column
//     holds text. No field positions are kept. The chunks' inferred types
//     are then merged column by column (the rules of docs/api/frame.md
//     §10.4 hold for the whole column), and a chunk whose kind the merge
//     changes to text is read again.
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

// Bytes that are not valid UTF-8 as \xNN (Python's repr of bytes), the rest
// as they are: a message must be valid UTF-8 to become a str.
std::string printable(std::string_view s) {
    std::string out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t len = 1;
        const auto c = static_cast<unsigned char>(s[i]);
        if (c >= 0x80) {
            len = (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
            if (len == 0 || i + len > s.size() || !valid_utf8(s.substr(i, len))) {
                static const char* hex = "0123456789abcdef";
                out += "\\x";
                out += hex[c >> 4];
                out += hex[c & 15];
                ++i;
                continue;
            }
        }
        out.append(s.substr(i, len));
        i += len;
    }
    return out;
}

// The na_values texts, found by a string_view (no std::string per field).
struct TextHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};
class NaSet {
public:
    template <class It>
    NaSet(It first, It last) : set_(first, last) {
        for (const std::string& t : set_)
            lengths_ |= t.size() < 64 ? std::uint64_t{1} << t.size() : 0, long_ |= t.size() >= 64;
    }
    bool empty() const { return set_.empty(); }
    // Most fields have a length no na_values text has: no hash for them.
    bool contains(std::string_view s) const {
        if (s.size() < 64 ? !((lengths_ >> s.size()) & 1) : !long_) return false;
        return set_.find(s) != set_.end();
    }

private:
    std::unordered_set<std::string, TextHash, std::equal_to<>> set_;
    std::uint64_t lengths_ = 0;  // bit k: a text of k bytes
    bool long_ = false;          // a text of 64 bytes or more
};

// One output column's rows of one chunk, built while the chunk is tokenized
// (docs/design/frame.md §10.6).
struct Part {
    enum Kind : std::uint8_t { Int, Float, Text };
    Kind kind = Int;        // inferred: what the chunk's values are so far
    bool any = false;       // a value that is not NA
    bool all_bool = true;   // Text: every value is True or False
    std::vector<std::uint8_t> valid;  // per row: 1 a value, 0 NA
    std::vector<std::int64_t> num;    // Int, Float, forced int64 and float64: int64 or double bits per row
    std::vector<std::uint8_t> flag;   // forced bool: 0 or 1 per row
    std::string bytes;                // Text and forced string: the values back to back
    std::vector<std::int64_t> ends;   // where each row's bytes end in `bytes`
    std::int64_t bad = -1;  // the first row that cannot be read as the forced dtype, or that is not valid UTF-8
    std::string bad_text;   // a forced number or bool: that row's text
};

struct Chunk {
    std::int64_t first = 0;            // the byte where its first record starts
    std::vector<std::int64_t> starts;  // the byte where each record starts
    std::vector<Part> parts;           // per output column
    std::string error;                 // the chunk's tokenizing error, if any
};

// The 1-based line of a row of the file's body: where that record starts.
std::int64_t line_of(const char* base, const std::vector<Chunk>& chunks, std::size_t c, std::int64_t row) {
    return line_at(base, chunks[c].starts[static_cast<std::size_t>(row)]);
}

// Reads the chunks: tokenizes their records and hands every field of an
// output column straight to that column's Part of the chunk.
class ChunkReader {
public:
    ChunkReader(
        const char* data,
        std::size_t size,
        char sep,
        const std::vector<std::int64_t>& slot,
        const std::vector<std::optional<DType>>& forced,
        const NaSet& na
    )
        : data_(data),
          size_(size),
          sep_(sep),
          slot_(slot),
          forced_(forced),
          na_(na),
          width_(static_cast<std::int64_t>(slot.size())) {}

    // The records of `ch` from byte `from`, as long as each starts before
    // `end` and fewer than `limit` have been read in all (`read`, counted
    // across chunks). Returns false once the input has no more records.
    bool read(Chunk& ch, Tokenizer& t, std::int64_t end, std::int64_t limit, std::int64_t& read) {
        ch.parts.resize(forced_.size());
        std::string scratch;
        while (read < limit) {
            const std::int64_t start = t.offset();
            if (start >= end) return true;  // the next chunk's record
            const auto row = static_cast<std::int64_t>(ch.starts.size());
            const std::int64_t count = t.next([&](std::int64_t i, const Field& f) {
                if (i < width_ && slot_[i] >= 0) add(ch, static_cast<std::size_t>(slot_[i]), f, row, scratch);
            });
            if (count < 0) return false;
            if (count != width_)
                csv_error(
                    line_at(data_, start),
                    "expected " + std::to_string(width_) + " fields, got " + std::to_string(count)
                );
            ch.starts.push_back(start);
            ++read;
            if (row == 0) reserve(ch, (end - start) / std::max<std::int64_t>(1, t.offset() - start) + 1);
        }
        return false;
    }

    // Rows [0, rows) of column j of `ch` as text (Part::bytes and ends), read
    // again from the input: the chunk's numbers so far, once the column
    // turns out to hold text.
    void to_text(Chunk& ch, std::size_t j, std::int64_t rows) {
        Part& p = ch.parts[j];
        p.num.clear();
        p.num.shrink_to_fit();
        p.bytes.clear();
        p.ends.clear();
        p.kind = Part::Text;
        p.all_bool = true;
        if (rows == 0) return;
        Tokenizer t(data_, size_, sep_, ch.first);
        std::string scratch;
        for (std::int64_t row = 0; row < rows; ++row)
            t.next([&](std::int64_t i, const Field& f) {
                if (i >= width_ || slot_[i] != static_cast<std::int64_t>(j)) return;
                if (p.valid[static_cast<std::size_t>(row)]) append_text(p, text_of(data_, f, scratch), row);
                else p.ends.push_back(static_cast<std::int64_t>(p.bytes.size()));
            });
    }

private:
    // Room for about `rows` rows (from the first record's length), so the
    // Parts do not grow step by step; any number of rows still fits.
    static void reserve(Chunk& ch, std::int64_t rows) {
        const auto r = static_cast<std::size_t>(rows);
        ch.starts.reserve(r);
        for (Part& p : ch.parts) {
            p.valid.reserve(r);
            if (p.kind == Part::Text) {
                p.ends.reserve(r);
                p.bytes.reserve(p.bytes.size() * r);
            } else {
                p.num.reserve(r);
            }
        }
    }

    // docs/api/frame.md §10.3: NA is the unquoted empty field and the na_values texts.
    bool is_na(const Field& f, std::string& scratch) const {
        if (f.length == 0 && !(f.flags & kQuoted)) return true;
        return !na_.empty() && na_.contains(text_of(data_, f, scratch));
    }

    static void append_text(Part& p, std::string_view s, std::int64_t row) {
        if (p.bad < 0 && !valid_utf8(s)) p.bad = row;
        if (p.all_bool && s != "True" && s != "False") p.all_bool = false;
        p.bytes.append(s);
        p.ends.push_back(static_cast<std::int64_t>(p.bytes.size()));
    }

    static std::int64_t bits_of(double d) {
        std::int64_t b;
        std::memcpy(&b, &d, sizeof b);
        return b;
    }

    void add(Chunk& ch, std::size_t j, const Field& f, std::int64_t row, std::string& scratch) {
        Part& p = ch.parts[j];
        if (forced_[j]) return add_forced(p, *forced_[j], f, row, scratch);
        // docs/api/frame.md §10.4: int64, then float64, then text (bool or string, decided per column).
        if (is_na(f, scratch)) {
            p.valid.push_back(0);
            if (p.kind == Part::Text) p.ends.push_back(static_cast<std::int64_t>(p.bytes.size()));
            else p.num.push_back(0);
            return;
        }
        p.valid.push_back(1);
        p.any = true;
        const std::string_view s = text_of(data_, f, scratch);
        std::int64_t v;
        double d;
        switch (p.kind) {
            case Part::Int:
                if (text::parse_int64(s, v) == 1) {
                    p.num.push_back(v);
                    return;
                }
                if (text::parse_double(s, d)) {  // an int beyond int64 is float syntax
                    p.kind = Part::Float;  // the chunk's ints so far become doubles, rounded as in list inference
                    for (std::size_t k = 0; k < p.num.size(); ++k)
                        if (p.valid[k]) p.num[k] = bits_of(static_cast<double>(p.num[k]));
                    p.num.push_back(bits_of(d));
                    return;
                }
                break;  // no number: text
            case Part::Float:
                // A float column: an int is rounded to float64 (beyond int64 too: it is float syntax),
                // as list inference rounds an int next to a float (docs/api/frame.md §2.2, §10.4).
                if (text::parse_int64(s, v) == 1) {
                    p.num.push_back(bits_of(static_cast<double>(v)));
                    return;
                }
                if (text::parse_double(s, d)) {
                    p.num.push_back(bits_of(d));
                    return;
                }
                break;
            case Part::Text: return append_text(p, s, row);
        }
        to_text(ch, j, row);
        append_text(p, s, row);
    }

    void add_forced(Part& p, DType dt, const Field& f, std::int64_t row, std::string& scratch) {
        if (p.bad >= 0) return;  // the column cannot be read: it raises (docs/design/frame.md §10.6)
        const bool na = is_na(f, scratch);
        p.valid.push_back(!na);
        const std::string_view s = na ? std::string_view{} : text_of(data_, f, scratch);
        bool ok = true;
        switch (dt) {
            case DType::Int64: {
                std::int64_t v = 0;
                ok = na || text::parse_int64(s, v) == 1;
                p.num.push_back(na ? 0 : v);
                break;
            }
            case DType::Float64: {
                double d = 0.0;
                ok = na || text::parse_double(s, d);
                p.num.push_back(na ? 0 : bits_of(d));
                break;
            }
            case DType::Bool:
                ok = na || s == "True" || s == "False";
                p.flag.push_back(!na && s == "True");
                break;
            case DType::String:
                if (!na && !valid_utf8(s)) p.bad = row;
                p.bytes.append(s);
                p.ends.push_back(static_cast<std::int64_t>(p.bytes.size()));
                return;
            case DType::Complex128: break;  // not a frame dtype
        }
        if (!ok) {
            p.bad = row;
            p.bad_text = s;
        }
    }

    const char* data_;
    std::size_t size_;
    char sep_;
    const std::vector<std::int64_t>& slot_;  // slot_[i]: the output column file column i feeds, or -1
    const std::vector<std::optional<DType>>& forced_;
    const NaSet& na_;
    std::int64_t width_;
};

// Builds the columns from the chunks' Parts, one column at a time, freeing
// each column's Parts as it goes: the memory is the result's plus one
// column's Parts (docs/design/frame.md §10.6).
class ColumnBuilder {
public:
    ColumnBuilder(const char* base, std::vector<Chunk>& chunks, ChunkReader& reader)
        : base_(base), chunks_(chunks), reader_(reader), row0_(chunks.size() + 1, 0) {
        for (std::size_t c = 0; c < chunks.size(); ++c)
            row0_[c + 1] = row0_[c] + static_cast<std::int64_t>(chunks[c].starts.size());
        n_ = row0_.back();
    }

    Column build(std::size_t j, std::optional<DType> forced, const std::string& name) {
        j_ = j;
        if (forced) {
            if (auto bad = first_bad()) {
                const auto [c, row] = *bad;
                const Part& p = chunks_[c].parts[j];
                if (*forced == DType::String)
                    csv_error(line_of(base_, chunks_, c, row), "column '" + name + "' is not valid UTF-8");
                csv_error(
                    line_of(base_, chunks_, c, row),
                    "column '" + name + "': cannot read '" + printable(p.bad_text) + "' as " + dtype_name(*forced)
                );
            }
            return *forced == DType::String ? strings() : fixed(*forced, *forced == DType::Bool);
        }
        bool any = false;
        Part::Kind kind = Part::Int;
        for (Chunk& ch : chunks_) {
            any |= ch.parts[j].any;
            kind = std::max(kind, ch.parts[j].kind);
        }
        if (!any) return all_na();  // all NA: float64 (docs/api/frame.md §10.4)
        if (kind == Part::Float) {  // chunks of ints become doubles, rounded
            each_part([](Part& p) {
                if (p.kind != Part::Int) return;
                for (std::size_t k = 0; k < p.num.size(); ++k)
                    if (p.valid[k]) {
                        const double d = static_cast<double>(p.num[k]);
                        std::memcpy(&p.num[k], &d, sizeof d);
                    }
            });
        }
        if (kind != Part::Text) return fixed(kind == Part::Float ? DType::Float64 : DType::Int64, false);
        bool all_bool = true;
        for (Chunk& ch : chunks_) {
            const Part& p = ch.parts[j];
            all_bool &= p.kind == Part::Text ? p.all_bool : !p.any;
        }
        if (all_bool) return bools();
        // A string column: the chunks that held numbers only are read again as text.
        parallel::parallel_for(static_cast<std::int64_t>(chunks_.size()), [&](std::int64_t c) {
            Chunk& ch = chunks_[static_cast<std::size_t>(c)];
            Part& p = ch.parts[j];
            if (p.kind != Part::Text) reader_.to_text(ch, j, static_cast<std::int64_t>(ch.starts.size()));
        });
        if (auto bad = first_bad())
            csv_error(line_of(base_, chunks_, bad->first, bad->second), "column '" + name + "' is not valid UTF-8");
        return strings();
    }

private:
    template <class F>
    void each_part(F f) {
        parallel::parallel_for(static_cast<std::int64_t>(chunks_.size()), [&](std::int64_t c) {
            f(chunks_[static_cast<std::size_t>(c)].parts[j_]);
        });
    }

    template <class F>
    void each_chunk(F f) {
        parallel::parallel_for(static_cast<std::int64_t>(chunks_.size()), [&](std::int64_t c) {
            const auto k = static_cast<std::size_t>(c);
            f(chunks_[k].parts[j_], row0_[k]);
        });
    }

    // (chunk, row) of the column's first bad row in file order, if any.
    std::optional<std::pair<std::size_t, std::int64_t>> first_bad() const {
        for (std::size_t c = 0; c < chunks_.size(); ++c)
            if (chunks_[c].parts[j_].bad >= 0) return std::pair{c, chunks_[c].parts[j_].bad};
        return std::nullopt;
    }

    static void release(Part& p) { p = Part{}; }

    // valid[0, n) as the column's bitmap; then the NA count and zeroed NA rows.
    void set_validity(Column& col, const std::vector<std::uint8_t>& valid) {
        col.validity = Buffer::allocate(bitmap::num_bytes(n_), false);
        auto* bits = reinterpret_cast<std::uint8_t*>(col.validity->data());
        for_chunks(n_, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t b = lo / 8; b < bitmap::num_bytes(hi); ++b) {
                unsigned v = 0;
                for (std::int64_t i = b * 8; i < std::min(hi, b * 8 + 8); ++i)
                    v |= static_cast<unsigned>(valid[i]) << (i & 7);
                bits[b] = static_cast<std::uint8_t>(v);
            }
        });
        finish_fixed(col);
    }

    Column all_na() {
        Column col = make_fixed(DType::Float64, n_);
        std::memset(col.values->data(), 0, static_cast<std::size_t>(n_ * 8));
        set_validity(col, std::vector<std::uint8_t>(static_cast<std::size_t>(n_), 0));
        each_part(release);
        return col;
    }

    // int64 or float64 from Part::num, or bool from Part::flag.
    Column fixed(DType dt, bool from_flags) {
        Column col = make_fixed(dt, n_);
        std::vector<std::uint8_t> valid(static_cast<std::size_t>(n_));
        std::byte* out = col.values->data();
        const std::int64_t size = itemsize(dt);
        each_chunk([&](Part& p, std::int64_t r0) {
            const std::int64_t rows = static_cast<std::int64_t>(p.valid.size());
            const void* src = from_flags ? static_cast<const void*>(p.flag.data()) : p.num.data();
            if (rows > 0) {
                std::memcpy(out + r0 * size, src, static_cast<std::size_t>(rows * size));
                std::memcpy(valid.data() + r0, p.valid.data(), static_cast<std::size_t>(rows));
            }
            release(p);
        });
        set_validity(col, valid);
        return col;
    }

    // An inferred bool column: every value is True or False (the Parts of
    // text), the other Parts are all NA.
    Column bools() {
        Column col = make_fixed(DType::Bool, n_);
        std::vector<std::uint8_t> valid(static_cast<std::size_t>(n_));
        auto* out = reinterpret_cast<std::uint8_t*>(col.values->data());
        each_chunk([&](Part& p, std::int64_t r0) {
            std::int64_t from = 0;
            for (std::size_t i = 0; i < p.valid.size(); ++i) {
                const std::int64_t row = r0 + static_cast<std::int64_t>(i);
                valid[row] = p.valid[i];
                out[row] = 0;
                if (p.kind == Part::Text) {
                    const std::int64_t to = p.ends[i];
                    out[row] = p.valid[i] && to - from == 4;  // "True"; the others are "False"
                    from = to;
                }
            }
            release(p);
        });
        set_validity(col, valid);
        return col;
    }

    Column strings() {
        Column col;
        col.dtype = DType::String;
        col.length = n_;
        col.offsets = Buffer::allocate((n_ + 1) * 8, false);
        auto* offs = reinterpret_cast<std::int64_t*>(col.offsets->data());
        offs[0] = 0;
        std::vector<std::int64_t> at(chunks_.size() + 1, 0);  // where each chunk's bytes start
        for (std::size_t c = 0; c < chunks_.size(); ++c)
            at[c + 1] = at[c] + static_cast<std::int64_t>(chunks_[c].parts[j_].bytes.size());
        col.values = Buffer::allocate(at.back(), false);
        auto* out = reinterpret_cast<char*>(col.values->data());
        std::vector<std::uint8_t> valid(static_cast<std::size_t>(n_));
        parallel::parallel_for(static_cast<std::int64_t>(chunks_.size()), [&](std::int64_t c) {
            const auto k = static_cast<std::size_t>(c);
            Part& p = chunks_[k].parts[j_];
            const std::int64_t r0 = row0_[k];
            std::memcpy(out + at[k], p.bytes.data(), p.bytes.size());
            for (std::size_t i = 0; i < p.ends.size(); ++i) {
                offs[r0 + static_cast<std::int64_t>(i) + 1] = at[k] + p.ends[i];
                valid[r0 + static_cast<std::int64_t>(i)] = p.valid[i];
            }
            release(p);
        });
        col.validity = Buffer::allocate(bitmap::num_bytes(n_), false);
        auto* bits = reinterpret_cast<std::uint8_t*>(col.validity->data());
        std::int64_t count = 0;
        for (std::int64_t i = 0; i < n_; ++i) count += valid[i];
        for_chunks(n_, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t b = lo / 8; b < bitmap::num_bytes(hi); ++b) {
                unsigned v = 0;
                for (std::int64_t i = b * 8; i < std::min(hi, b * 8 + 8); ++i)
                    v |= static_cast<unsigned>(valid[i]) << (i & 7);
                bits[b] = static_cast<std::uint8_t>(v);
            }
        });
        col.null_count = n_ - count;
        if (col.null_count == 0) col.validity.reset();
        return col;
    }

    const char* base_;
    std::vector<Chunk>& chunks_;
    ChunkReader& reader_;
    std::vector<std::int64_t> row0_;  // the first row of each chunk; back(): the row count
    std::int64_t n_ = 0;
    std::size_t j_ = 0;
};

}  // namespace

std::int64_t csv_records_end(const char* data, std::size_t size, std::int64_t records) {
    if (records <= 0) return 0;
    bool inside = false;
    std::int64_t seen = 0;
    for (std::size_t i = 0; i < size; ++i) {
        if (data[i] == '"') inside = !inside;
        else if (data[i] == '\n' && !inside && ++seen == records) return static_cast<std::int64_t>(i) + 1;
    }
    return -1;
}

std::pair<std::vector<std::string>, std::vector<Column>> read_csv(
    const char* data, std::size_t size, const CsvOptions& opt
) {
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
            for (const Field& f : rec) {
                names.emplace_back(text_of(data, f, scratch));
                // docs/api/frame.md §10.2: the names become str, so they must be UTF-8 as values are
                if (!valid_utf8(names.back())) csv_error(line, "column names are not valid UTF-8");
            }
        } else if (names.size() != rec.size()) {
            csv_error(
                line, std::to_string(names.size()) + " names for a header of " + std::to_string(rec.size()) + " fields"
            );
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
        if (it == names.end())
            throw nb::key_error(("'usecols' names a column that is not in the file: '" + out_names[j] + "'").c_str());
        source[j] = it - names.begin();
    }
    const std::size_t ncols = names.size();
    std::vector<std::int64_t> slot(ncols, -1);  // slot[i]: the output column file column i feeds, or -1
    for (std::size_t j = 0; j < source.size(); ++j) slot[source[j]] = static_cast<std::int64_t>(j);
    std::vector<std::optional<DType>> forced(out_names.size());
    for (std::size_t j = 0; j < out_names.size(); ++j) {
        if (auto it = opt.dtypes.find(out_names[j]); it != opt.dtypes.end()) forced[j] = it->second;
        else if (opt.all_dtype) forced[j] = opt.all_dtype;
    }
    const NaSet na(opt.na_values.begin(), opt.na_values.end());
    ChunkReader reader(data, size, opt.sep, slot, forced, na);

    // The chunks. With nrows (records are read in order, up to nrows) and
    // when the pool cannot run in parallel, one tokenizer reads the body in
    // order, starting a new chunk at the first record past each kChunkBytes;
    // otherwise the chunks are read in parallel. The result does not depend
    // on the chunks, so they may follow the thread count (docs/design/frame.md
    // §11.5, as for factorize).
    const auto n = static_cast<std::int64_t>(size);
    const std::int64_t body = tok.offset();
    const std::int64_t limit = opt.nrows >= 0 ? opt.nrows : INT64_MAX;
    std::vector<Chunk> chunks;
    // A task's std::bad_alloc is rethrown as is (a MemoryError), not taken
    // for an error in the file.
    if (opt.nrows >= 0 || !parallel::available()) {
        Tokenizer t(data, size, opt.sep, body);
        std::int64_t read = 0;
        for (bool more = true; more;) {
            Chunk& ch = chunks.emplace_back();
            ch.first = t.offset();
            try {
                more = reader.read(ch, t, std::min(n, ch.first + kChunkBytes), limit, read) && t.offset() < n;
            } catch (const std::bad_alloc&) { throw; } catch (const std::exception& e) {
                ch.error = e.what();
                more = false;
            }
        }
    } else {
        const std::int64_t nchunks = std::max<std::int64_t>(1, (n - body + kChunkBytes - 1) / kChunkBytes);
        auto chunk_begin = [&](std::int64_t c) { return std::min(n, body + c * kChunkBytes); };
        auto chunk_end = [&](std::int64_t c) { return c + 1 == nchunks ? n : chunk_begin(c + 1); };
        std::vector<std::uint8_t> odd(static_cast<std::size_t>(nchunks));
        if (nchunks > 1)
            parallel::parallel_for(nchunks, [&](std::int64_t c) {
                odd[c] = odd_quotes(data, chunk_begin(c), chunk_end(c));
            });
        std::vector<std::uint8_t> inside(static_cast<std::size_t>(nchunks), 0);  // inside quotes where chunk c starts
        for (std::int64_t c = 1; c < nchunks; ++c) inside[c] = inside[c - 1] ^ odd[c - 1];
        chunks.resize(static_cast<std::size_t>(nchunks));
        parallel::parallel_for(nchunks, [&](std::int64_t c) {
            Chunk& ch = chunks[static_cast<std::size_t>(c)];
            ch.parts.resize(out_names.size());
            const std::int64_t end = chunk_end(c);
            ch.first = c == 0 ? body : first_record(data, n, chunk_begin(c), inside[c]);
            if (ch.first >= end || ch.first >= n) return;
            try {
                Tokenizer t(data, size, opt.sep, ch.first);
                std::int64_t read = 0;
                reader.read(ch, t, end, INT64_MAX, read);
            } catch (const std::bad_alloc&) { throw; } catch (const std::exception& e) {
                ch.error = e.what();
            }
        });
    }
    for (const Chunk& ch : chunks)  // the first error in file order
        if (!ch.error.empty()) throw nb::value_error(ch.error.c_str());

    ColumnBuilder builder(data, chunks, reader);
    std::vector<Column> cols;
    for (std::size_t j = 0; j < out_names.size(); ++j) cols.push_back(builder.build(j, forced[j], out_names[j]));
    return {out_names, cols};
}

}  // namespace sd::frame
