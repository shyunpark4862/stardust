#include "frame/io/csv.hpp"

#include "frame/ops/parse.hpp"
#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <arm_neon.h>
#include <unistd.h>

#include <cerrno>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
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

// --- the file in chunks (docs/design/frame.md §10.6) ---------------------
//
// The body (after the header) is cut into byte chunks.
//  1. Each chunk counts its quotes (NEON). Whether a byte is inside quotes
//     is the parity of the quotes before it, so the running parity gives
//     every chunk's state at its start: no guessing.
//  2. A chunk owns the records that start inside it: it finds its first
//     record start (after a line feed outside quotes) and indexes its
//     records from there (index_records), the last one even past its end.
//     The index is the byte of every field end, `width` per record; a chunk
//     with quotes then checks the quoting rules field by field.
//  3. Each output column of the chunk is then parsed in one loop over its
//     fields (parse_column): per-column loops, no per-field dispatch on the
//     column. The index is dropped once the chunk is parsed, so memory is
//     the result plus one index per running chunk.
//  4. The chunks' inferred kinds are merged column by column (the rules of
//     docs/api/frame.md §10.4 hold for the whole column); a chunk whose kind
//     the merge changes to text is indexed and read again.
// Errors are the serial ones: the first chunk with a tokenizing error
// reports it (within a chunk: the first in record and field order); then
// columns in order, the first bad row of a column.
//
// The chunks depend on the file size and the thread count: the result does
// not depend on them (docs/design/frame.md §11.5, as for factorize).

constexpr std::int64_t kChunkBytes = std::int64_t{1} << 19;     // the most bytes per chunk
constexpr std::int64_t kMinChunkBytes = std::int64_t{1} << 16;  // the least, when the file is cut finer

// The bytes per chunk: 512 KB, or for a file of less than 32 of those, the
// file in 32 chunks (no smaller than 64 KB) so that every thread gets
// several (a 2 MB file was 4 chunks: at most 4 threads, unevenly).
std::int64_t chunk_bytes(std::int64_t body, bool parallel) {
    if (!parallel) return kChunkBytes;
    return std::clamp(body / 32, kMinChunkBytes, kChunkBytes);
}

// Whether [from, to) holds an odd number of quotes: 16 bytes per step, each
// lane toggled by its quotes.
bool odd_quotes(const char* p, std::int64_t from, std::int64_t to) {
    const auto* s = reinterpret_cast<const std::uint8_t*>(p);
    const uint8x16_t q = vdupq_n_u8('"');
    uint8x16_t acc = vdupq_n_u8(0);
    std::int64_t i = from;
    for (; i + 16 <= to; i += 16) acc = veorq_u8(acc, vceqq_u8(vld1q_u8(s + i), q));
    unsigned odd = vaddvq_u8(vandq_u8(acc, vdupq_n_u8(1))) & 1;
    for (; i < to; ++i) odd ^= s[i] == '"';
    return odd;
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

// A chunk's records, tokenized: the byte ending each field (separator, line
// feed, or the end of the input), `width` per record.
//
// Once a quote is seen, every field also gets the quotes it holds, from a
// running count of the quote bits (a popcount per field): 0 none, 1 two
// (a quoted field without "" if they are its first and last bytes), 2 more
// or an odd number (a quoted field with "" pairs, or an error). Only fields
// with quotes are then checked, and only "" fields copied.
struct RecordIndex {
    std::unique_ptr<std::int64_t[]> ends;  // not zeroed: written before read
    std::unique_ptr<std::uint8_t[]> qf;    // per field, once `quotes`: kNoQuote, kTwoQuotes or kMoreQuotes
    std::int64_t size = 0, cap = 0;
    std::int64_t records = 0;
    bool quotes = false;  // a quote somewhere in the bytes read
    bool ascii = true;    // no byte >= 0x80 in the bytes read: every text is valid UTF-8

    static constexpr std::uint8_t kNoQuote = 0, kTwoQuotes = 1, kMoreQuotes = 2;

    std::int64_t operator[](std::int64_t k) const { return ends[static_cast<std::size_t>(k)]; }
    std::uint8_t quote_flag(std::int64_t k) const { return quotes ? qf[static_cast<std::size_t>(k)] : kNoQuote; }
    // Room for `need` ends in all.
    void reserve(std::int64_t need) {
        if (need <= cap) return;
        const std::int64_t to = std::max(need, cap * 2);
        std::unique_ptr<std::int64_t[]> grown(new std::int64_t[static_cast<std::size_t>(to)]);
        if (size > 0) std::memcpy(grown.get(), ends.get(), static_cast<std::size_t>(size) * sizeof(std::int64_t));
        ends = std::move(grown);
        if (qf) {
            std::unique_ptr<std::uint8_t[]> g(new std::uint8_t[static_cast<std::size_t>(to)]);
            if (size > 0) std::memcpy(g.get(), qf.get(), static_cast<std::size_t>(size));
            qf = std::move(g);
        }
        cap = to;
    }
    // The first quote: the fields so far hold none.
    void start_quotes() {
        qf.reset(new std::uint8_t[static_cast<std::size_t>(cap)]);
        std::memset(qf.get(), 0, static_cast<std::size_t>(size));
        quotes = true;
    }
    static std::uint8_t flag_of(std::int64_t nquotes) {
        return nquotes == 0 ? kNoQuote : nquotes == 2 ? kTwoQuotes : kMoreQuotes;
    }
};

// Where a field's text is: inside the quotes of a quoted field; `escaped`:
// it still holds "" pairs to turn into ".
struct FieldText {
    std::int64_t from, to;
    bool quoted, escaped;
};

class Reader {
public:
    Reader(
        const char* data,
        std::size_t size,
        char sep,
        std::int64_t width,
        const std::vector<std::int64_t>& source,
        const std::vector<std::optional<DType>>& forced,
        const NaSet& na
    )
        : p_(data),
          n_(static_cast<std::int64_t>(size)),
          sep_(sep),
          width_(width),
          source_(source),
          forced_(forced),
          na_(na),
          has_na_values_(!na.empty()) {}

    // Indexes the records that start in [first, end), at most `limit`: the
    // last one runs on past `end` to its line end. Throws the first
    // tokenizing error of those records, as a serial tokenizer meets it.
    void index_records(std::int64_t first, std::int64_t end, std::int64_t limit, RecordIndex& ix) const {
        ix.size = 0;
        ix.records = 0;
        ix.quotes = false;
        ix.qf.reset();
        if (first >= n_ || first >= end || limit <= 0) return;
        const std::int64_t W = width_;
        ix.reserve((std::min(end, n_) - first) / 8 + 128);
        std::int64_t k = 0;            // fields indexed
        std::int64_t count = 0;        // fields of the current record
        std::int64_t rec = first;      // the current record's first byte
        std::int64_t prev = first - 1;  // the byte that ended the field before
        bool inside = false;
        std::int64_t qbase = 0;     // quotes before the current block
        std::int64_t qlast = 0;     // quotes before the current field
        uint8x16_t high = vdupq_n_u8(0);
        auto finish = [&] {
            ix.size = k;
            ix.ascii = (vmaxvq_u8(high) & 0x80) == 0;
        };
        for (std::int64_t base = first;; base += 64) {
            if (base >= n_) {  // the end of the input ends the last record
                if (inside) {
                    finish();
                    check_quotes(first, ix, k, rec);
                    const std::int64_t fs = prev + 1;
                    csv_error(
                        line_at(p_, rec),
                        p_[fs] == '"' ? "a quoted field is never closed" : "a quote inside an unquoted field"
                    );
                }
                if (count > 0 || prev + 1 < n_) {
                    ix.size = k;
                    push(ix, k, n_);
                    if (ix.quotes) ix.qf[static_cast<std::size_t>(k - 1)] = RecordIndex::flag_of(qbase - qlast);
                    big_field(first, ix, k, rec, prev, n_);
                    ++count;
                    if (count != W) count_error(first, ix, k, rec, count);
                    ++ix.records;
                }
                finish();
                return;
            }
            std::uint8_t tail[64];
            const auto* src = reinterpret_cast<const std::uint8_t*>(p_ + base);
            const bool partial = base + 64 > n_;
            if (partial) {  // the last, partial block: padded with zero bytes
                std::memset(tail, 0, 64);
                std::memcpy(tail, src, static_cast<std::size_t>(n_ - base));
                src = tail;
            }
            uint8x16_t v[4];
            for (int j = 0; j < 4; ++j) v[j] = vld1q_u8(src + 16 * j);
            high = vmaxq_u8(high, vmaxq_u8(vmaxq_u8(v[0], v[1]), vmaxq_u8(v[2], v[3])));
            const std::uint64_t q = mask64(v, '"');
            std::uint64_t in = 0;
            if (q | static_cast<std::uint64_t>(inside)) {
                in = prefix_xor(q) ^ (inside ? ~std::uint64_t{0} : 0);
                inside = (in >> 63) & 1;
            }
            const std::uint64_t lf = mask64(v, '\n') & ~in;
            std::uint64_t bits = (mask64(v, static_cast<std::uint8_t>(sep_)) & ~in) | lf;
            // The zero padding of a partial block is not input: sep="\0" would match it.
            if (partial) bits &= (std::uint64_t{1} << (n_ - base)) - 1;
            if (ix.cap < k + 64) {
                ix.size = k;
                ix.reserve(k + 64);
            }
            if (q && !ix.quotes) {
                ix.size = k;
                ix.start_quotes();
            }
            std::int64_t* out = ix.ends.get();
            std::uint8_t* qf = ix.qf.get();
            while (bits) {
                const int b = __builtin_ctzll(bits);
                bits &= bits - 1;
                const std::int64_t at = base + b;
                if (qf) {  // the quotes of this field: those before `at` less those before the field
                    const std::int64_t here = qbase + __builtin_popcountll(q & ((std::uint64_t{1} << b) - 1));
                    qf[k] = RecordIndex::flag_of(here - qlast);
                    qlast = here;
                }
                out[k++] = at;
                if (at - prev > (std::int64_t{1} << 32)) [[unlikely]]
                    big_field(first, ix, k, rec, prev, at);
                prev = at;
                ++count;
                if ((lf >> b) & 1) {  // a record ends
                    if (count != W) [[unlikely]] {
                        finish();
                        count_error(first, ix, k, rec, count);
                    }
                    count = 0;
                    rec = at + 1;
                    if (++ix.records == limit || rec >= end) {
                        finish();
                        return;
                    }
                }
            }
            qbase += __builtin_popcountll(q);
        }
    }

    // The quoting rules for every field of `ix` (a chunk with quotes).
    void check_quotes(std::int64_t first, const RecordIndex& ix) const { check_quotes(first, ix, ix.size, first); }

    // Field k of `ix` (field ends; the first field starts at `first`).
    FieldText field(std::int64_t first, const RecordIndex& ix, std::int64_t k, bool record_end) const {
        std::int64_t from = k == 0 ? first : ix[k - 1] + 1;
        std::int64_t to = ix[k];
        // A field ending a record at a line feed: a \r just before it is the \r\n line end.
        if (record_end && to < n_ && to > from && p_[to - 1] == '\r') --to;
        if (const std::uint8_t qf = ix.quote_flag(k)) {  // checked by check_quotes: "..." (with "" pairs: 2)
            return {from + 1, to - 1, true, qf == RecordIndex::kMoreQuotes};
        }
        return {from, to, false, false};
    }

    // The line of record `row` of a chunk (for error messages): indexes it again.
    std::int64_t line_of_row(std::int64_t first, std::int64_t row) const {
        if (row == 0) return line_at(p_, first);
        RecordIndex ix;
        index_records(first, n_, row, ix);
        return line_at(p_, ix[row * width_ - 1] + 1);
    }

    // The `rows` records from `first` again (records already read: no error).
    void reindex(std::int64_t first, std::int64_t rows, RecordIndex& ix) const { index_records(first, n_, rows, ix); }

    const char* data() const { return p_; }

    // One output column's rows of one chunk.
    struct Part {
        // Inferred: what the chunk's values are. Bool: every value True or
        // False (kept as flags, not as text).
        enum Kind : std::uint8_t { Int, Float, Bool, Text };
        Kind kind = Int;
        bool any = false;  // a value that is not NA
        std::int64_t rows = 0;
        std::int64_t nna = 0;             // NA rows
        std::vector<std::uint8_t> valid;  // per row, 1 a value, 0 NA; empty while there is no NA
        std::vector<std::int64_t> num;    // Int, Float, forced int64 and float64: int64 or double bits per row
        std::vector<std::uint8_t> flag;   // Bool and forced bool: 0 or 1 per row
        std::string bytes;                // Text and forced string: the values back to back
        std::vector<std::int64_t> ends;   // where each row's bytes end in `bytes`
        std::int64_t bad = -1;  // the first row that cannot be read as the forced dtype, or that is not valid UTF-8
        std::string bad_text;   // a forced number or bool: that row's text

        bool is_valid(std::int64_t r) const { return valid.empty() || valid[static_cast<std::size_t>(r)]; }
        void set_na(std::int64_t r) {
            if (valid.empty()) valid.assign(static_cast<std::size_t>(rows), 1);
            valid[static_cast<std::size_t>(r)] = 0;
            ++nna;
        }
    };

    // Output column j's Part of a chunk from its index (`rows` records).
    void parse_column(std::size_t j, std::int64_t first, const RecordIndex& ix, Part& p, bool as_text = false) const {
        p = Part{};
        p.rows = ix.records;
        if (p.rows == 0) return;
        if (forced_[j]) return parse_forced(*forced_[j], j, first, ix, p);
        if (as_text) return parse_text(j, first, ix, p, false);
        const std::int64_t W = width_, i = source_[j], R = p.rows;
        const bool last = i == W - 1;
        p.num.resize(static_cast<std::size_t>(R));
        std::int64_t* num = p.num.data();
        std::string scratch;
        std::int64_t r = 0;
        // docs/api/frame.md §10.4: int64, then float64, then text (bool or string, decided per column).
        for (; r < R; ++r) {
            const FieldText f = field(first, ix, r * W + i, last);
            if (is_na(f, scratch)) {
                p.set_na(r);
                num[r] = 0;
                continue;
            }
            const bool first_value = !p.any;
            p.any = true;
            const std::string_view s = text(f, scratch);
            std::int64_t v;
            if (text::parse_int64(s, v, begin_of(f)) == 1) {
                num[r] = v;
                continue;
            }
            double d;
            if (!text::parse_double(s, d, begin_of(f))) {  // no number: bools, or text
                if (first_value && (s == "True" || s == "False")) return parse_bools(j, first, ix, p, r);
                return parse_text(j, first, ix, p, false);
            }
            // An int beyond int64 is float syntax. The chunk's ints so far
            // become doubles, rounded as in list inference.
            for (std::int64_t k = 0; k < r; ++k)
                if (p.is_valid(k)) num[k] = bits_of(static_cast<double>(num[k]));
            num[r++] = bits_of(d);
            p.kind = Part::Float;
            break;
        }
        for (; r < R; ++r) {
            const FieldText f = field(first, ix, r * W + i, last);
            if (is_na(f, scratch)) {
                p.set_na(r);
                num[r] = 0;
                continue;
            }
            const std::string_view s = text(f, scratch);
            double d;
            // A float column: an int is rounded to float64, as list inference
            // rounds an int next to a float (docs/api/frame.md §2.2, §10.4). parse_double
            // reads an int as exactly that, except "-0": the int 0, so +0.0.
            if (!text::parse_double(s, d, begin_of(f))) return parse_text(j, first, ix, p, false);
            if (d == 0.0 && std::signbit(d) && text::int_grammar(s)) d = 0.0;
            num[r] = bits_of(d);
        }
    }

private:
    static std::int64_t bits_of(double d) {
        std::int64_t b;
        std::memcpy(&b, &d, sizeof b);
        return b;
    }

    static void push(RecordIndex& ix, std::int64_t& k, std::int64_t at) {
        ix.size = k;
        ix.reserve(k + 1);
        ix.ends[static_cast<std::size_t>(k++)] = at;
    }

    // A field of 4 GB or more (field k - 1, ending at `at`) is an error,
    // after the quoting errors of the fields before it.
    void big_field(
        std::int64_t first, RecordIndex& ix, std::int64_t k, std::int64_t rec, std::int64_t prev, std::int64_t at
    ) const {
        std::int64_t to = at;
        if (to < n_ && p_[to] == '\n' && to > prev + 1 && p_[to - 1] == '\r') --to;
        if (to - (prev + 1) < (std::int64_t{1} << 32)) return;
        ix.size = k;
        check_quotes(first, ix, k - 1, rec);
        csv_error(line_at(p_, rec), "a field of 4 GB or more");
    }

    // A record of `count` fields: the quoting errors of the fields read so
    // far come first (the serial tokenizer checks each field as it ends).
    [[noreturn]] void count_error(
        std::int64_t first, RecordIndex& ix, std::int64_t k, std::int64_t rec, std::int64_t count
    ) const {
        ix.size = k;
        check_quotes(first, ix, k, rec);
        csv_error(line_at(p_, rec), "expected " + std::to_string(width_) + " fields, got " + std::to_string(count));
    }

    // The quoting rules for fields [0, k) of `ix` (when it has quotes); `rec`
    // is unused but for the call sites' symmetry.
    void check_quotes(std::int64_t first, const RecordIndex& ix, std::int64_t k, std::int64_t /*rec*/) const {
        if (!ix.quotes) return;
        std::int64_t record = first;
        for (std::int64_t f = 0; f < k; ++f) {
            const std::int64_t from = f == 0 ? first : ix[f - 1] + 1;
            std::int64_t to = ix[f];
            const bool line_end = to < n_ && p_[to] == '\n';
            if (line_end && to > from && p_[to - 1] == '\r') --to;
            if (const std::uint8_t flag = ix.qf[static_cast<std::size_t>(f)]) {
                if (p_[from] != '"') csv_error(line_at(p_, record), "a quote inside an unquoted field");
                if (to - from < 2 || p_[to - 1] != '"')
                    csv_error(
                        line_at(p_, record), "a closing quote must be followed by the separator or the end of the line"
                    );
                // Two quotes, the first and the last byte: nothing in between to check.
                for (std::int64_t i = from + 1; flag == RecordIndex::kMoreQuotes && i < to - 1; ++i) {
                    if (p_[i] != '"') continue;
                    if (i + 1 >= to - 1 || p_[i + 1] != '"')
                        csv_error(
                            line_at(p_, record),
                            "a closing quote must be followed by the separator or the end of the line"
                        );
                    ++i;  // "" is one quote
                }
            }
            if (line_end) record = ix[f] + 1;
        }
    }

    // A field's text, with "" pairs made " (only escaped fields need a copy).
    std::string_view text(const FieldText& f, std::string& scratch) const {
        std::string_view raw(p_ + f.from, static_cast<std::size_t>(f.to - f.from));
        if (!f.escaped) return raw;
        scratch.clear();
        for (std::size_t i = 0; i < raw.size(); ++i) {
            scratch += raw[i];
            if (raw[i] == '"') ++i;  // skip the second quote of the pair
        }
        return scratch;
    }

    // docs/api/frame.md §10.3: NA is the unquoted empty field and the na_values texts.
    [[gnu::always_inline]] bool is_na(const FieldText& f, std::string& scratch) const {
        if (f.to == f.from && !f.quoted) return true;
        return has_na_values_ && na_.contains(text(f, scratch));
    }

    // Readable memory before a field's text, for the number parsers' wide
    // reads: the input, or the copy of an escaped field.
    const char* begin_of(const FieldText& f) const { return f.escaped ? nullptr : p_; }

    // Column j of the chunk from row r0, its first value, True or False: bools
    // while every value is one; text from the first that is not.
    void parse_bools(std::size_t j, std::int64_t first, const RecordIndex& ix, Part& p, std::int64_t r0) const {
        const std::int64_t W = width_, i = source_[j], R = p.rows;
        const bool last = i == W - 1;
        p.kind = Part::Bool;
        p.num = {};
        p.flag.assign(static_cast<std::size_t>(R), 0);  // rows before r0 are NA
        std::string scratch;
        for (std::int64_t r = r0; r < R; ++r) {
            const FieldText f = field(first, ix, r * W + i, last);
            if (is_na(f, scratch)) {
                p.set_na(r);
                continue;
            }
            const std::string_view s = text(f, scratch);
            if (s == "True") p.flag[static_cast<std::size_t>(r)] = 1;
            else if (s != "False") return parse_text(j, first, ix, p, false);
        }
    }

    // Column j of the chunk as text (inferred: `forced` false; or a forced string column).
    void parse_text(std::size_t j, std::int64_t first, const RecordIndex& ix, Part& p, bool /*forced*/) const {
        const std::int64_t W = width_, i = source_[j], R = p.rows;
        const bool last = i == W - 1;
        p.kind = Part::Text;
        p.num = {};
        p.flag = {};
        p.valid.clear();
        p.nna = 0;
        p.any = false;
        // The bytes of every field bound the text's (quotes and "" pairs only shrink it).
        std::int64_t total = 0;
        for (std::int64_t r = 0; r < R; ++r) {
            const std::int64_t k = r * W + i;
            total += ix[k] - (k == 0 ? first : ix[k - 1] + 1);
        }
        // 16 bytes of room past the end: short texts are copied 16 bytes at a time.
        p.bytes.resize(static_cast<std::size_t>(total + 16));
        p.ends.resize(static_cast<std::size_t>(R));
        char* out = p.bytes.data();
        std::int64_t at = 0;
        std::string scratch;
        for (std::int64_t r = 0; r < R; ++r) {
            const FieldText f = field(first, ix, r * W + i, last);
            if (is_na(f, scratch)) {
                p.set_na(r);
            } else {
                const std::string_view s = text(f, scratch);
                p.any = true;
                if (p.bad < 0 && !ix.ascii && !valid_utf8(s)) p.bad = r;
                if (s.size() <= 16 && !f.escaped && f.from + 16 <= n_) {  // no call to memcpy for a short text
                    std::uint64_t a, b;
                    std::memcpy(&a, s.data(), 8);
                    std::memcpy(&b, s.data() + 8, 8);
                    std::memcpy(out + at, &a, 8);
                    std::memcpy(out + at + 8, &b, 8);
                } else {
                    std::memcpy(out + at, s.data(), s.size());
                }
                at += static_cast<std::int64_t>(s.size());
            }
            p.ends[static_cast<std::size_t>(r)] = at;
        }
        p.bytes.resize(static_cast<std::size_t>(at));
    }

    void parse_forced(DType dt, std::size_t j, std::int64_t first, const RecordIndex& ix, Part& p) const {
        if (dt == DType::String) return parse_text(j, first, ix, p, true);
        const std::int64_t W = width_, i = source_[j], R = p.rows;
        const bool last = i == W - 1;
        if (dt == DType::Bool) p.flag.resize(static_cast<std::size_t>(R));
        else p.num.resize(static_cast<std::size_t>(R));
        std::string scratch;
        for (std::int64_t r = 0; r < R; ++r) {
            const FieldText f = field(first, ix, r * W + i, last);
            if (is_na(f, scratch)) {
                p.set_na(r);
                if (dt == DType::Bool) p.flag[static_cast<std::size_t>(r)] = 0;
                else p.num[static_cast<std::size_t>(r)] = 0;
                continue;
            }
            const std::string_view s = text(f, scratch);
            bool ok = true;
            switch (dt) {
                case DType::Int64: {
                    std::int64_t v = 0;
                    ok = text::parse_int64(s, v, begin_of(f)) == 1;
                    p.num[static_cast<std::size_t>(r)] = v;
                    break;
                }
                case DType::Float64: {
                    double d = 0.0;
                    ok = text::parse_double(s, d, begin_of(f));
                    p.num[static_cast<std::size_t>(r)] = bits_of(d);
                    break;
                }
                case DType::Bool:
                    ok = s == "True" || s == "False";
                    p.flag[static_cast<std::size_t>(r)] = s == "True";
                    break;
                default: break;
            }
            if (!ok) {  // the column cannot be read: it raises (docs/design/frame.md §10.6)
                p.bad = r;
                p.bad_text = s;
                return;
            }
        }
    }

    const char* p_;
    std::int64_t n_;
    char sep_;
    std::int64_t width_;                       // fields per record
    const std::vector<std::int64_t>& source_;  // source_[j]: the file column output column j reads
    const std::vector<std::optional<DType>>& forced_;
    const NaSet& na_;
    bool has_na_values_;
};

using Part = Reader::Part;

struct Chunk {
    std::int64_t first = 0;   // the byte where its first record starts
    std::int64_t rows = 0;    // its records
    std::vector<Part> parts;  // per output column
    std::string error;        // the chunk's tokenizing error, if any
};

// Builds the columns from the chunks' Parts, one column at a time, freeing
// each column's Parts as it goes: the memory is the result's plus one
// column's Parts (docs/design/frame.md §10.6).
class ColumnBuilder {
public:
    ColumnBuilder(std::vector<Chunk>& chunks, const Reader& reader)
        : chunks_(chunks), reader_(reader), row0_(chunks.size() + 1, 0) {
        for (std::size_t c = 0; c < chunks.size(); ++c) row0_[c + 1] = row0_[c] + chunks[c].rows;
        n_ = row0_.back();
    }

    Column build(std::size_t j, std::optional<DType> forced, const std::string& name) {
        j_ = j;
        if (forced) {
            if (auto bad = first_bad()) {
                const auto [c, row] = *bad;
                const Part& p = chunks_[c].parts[j];
                const std::int64_t line = reader_.line_of_row(chunks_[c].first, row);
                if (*forced == DType::String) csv_error(line, "column '" + name + "' is not valid UTF-8");
                csv_error(
                    line, "column '" + name + "': cannot read '" + printable(p.bad_text) + "' as " + dtype_name(*forced)
                );
            }
            return *forced == DType::String ? strings() : fixed(*forced, *forced == DType::Bool);
        }
        // Chunks of numbers, of bools and of text (chunks of NA only count as none).
        bool any = false, num = false, flt = false, bol = false, txt = false;
        for (Chunk& ch : chunks_) {
            const Part& p = ch.parts[j];
            if (!p.any) continue;
            any = true;
            num |= p.kind == Part::Int || p.kind == Part::Float;
            flt |= p.kind == Part::Float;
            bol |= p.kind == Part::Bool;
            txt |= p.kind == Part::Text;
        }
        if (!any) return all_na();  // all NA: float64 (docs/api/frame.md §10.4)
        if (!txt && !bol && flt) {  // chunks of ints become doubles, rounded
            each_part([](Part& p) {
                if (p.kind != Part::Int) return;
                for (std::size_t k = 0; k < p.num.size(); ++k)
                    if (p.is_valid(static_cast<std::int64_t>(k))) {
                        const double d = static_cast<double>(p.num[k]);
                        std::memcpy(&p.num[k], &d, sizeof d);
                    }
            });
        }
        if (!txt && !bol) return fixed(flt ? DType::Float64 : DType::Int64, false);
        if (!txt && !num) return bools();  // every value True or False
        // A string column: the chunks that held numbers or bools only are read again as text.
        parallel_tasks(static_cast<std::int64_t>(chunks_.size()), [&](std::int64_t c) {
            Chunk& ch = chunks_[static_cast<std::size_t>(c)];
            Part& p = ch.parts[j];
            if (p.kind == Part::Text || ch.rows == 0) return;
            RecordIndex ix;
            reader_.reindex(ch.first, ch.rows, ix);
            reader_.parse_column(j, ch.first, ix, p, true);
        });
        if (auto bad = first_bad())
            csv_error(
                reader_.line_of_row(chunks_[bad->first].first, bad->second), "column '" + name + "' is not valid UTF-8"
            );
        return strings();
    }

private:
    template <class F>
    void each_part(F f) {
        parallel_tasks(static_cast<std::int64_t>(chunks_.size()), [&](std::int64_t c) {
            f(chunks_[static_cast<std::size_t>(c)].parts[j_]);
        });
    }

    template <class F>
    void each_chunk(F f) {
        parallel_tasks(static_cast<std::int64_t>(chunks_.size()), [&](std::int64_t c) {
            const auto k = static_cast<std::size_t>(c);
            f(chunks_[k].parts[j_], row0_[k], k);
        });
    }

    // (chunk, row) of the column's first bad row in file order, if any.
    std::optional<std::pair<std::size_t, std::int64_t>> first_bad() const {
        for (std::size_t c = 0; c < chunks_.size(); ++c)
            if (chunks_[c].parts[j_].bad >= 0) return std::pair{c, chunks_[c].parts[j_].bad};
        return std::nullopt;
    }

    static void release(Part& p) { p = Part{}; }

    // The column's validity bitmap from the Parts (none without NA) and its
    // NA count. The Parts hold 0 in their NA rows already (docs/design/frame.md §4.5).
    void set_validity(Column& col) {
        std::int64_t nna = 0;
        for (const Chunk& ch : chunks_) nna += ch.parts[j_].nna;
        col.null_count = nna;
        col.validity.reset();
        if (nna == 0) return;
        col.validity = Buffer::allocate(bitmap::num_bytes(n_), true);
        auto* bits = reinterpret_cast<std::uint8_t*>(col.validity->data());
        // Serially: a chunk's rows need not start on a byte boundary.
        for (std::size_t c = 0; c < chunks_.size(); ++c) {
            const Part& p = chunks_[c].parts[j_];
            put_bits(bits, row0_[c], p.rows, p.valid.empty() ? nullptr : p.valid.data());
        }
    }

    // Bits [r0, r0 + rows) of a zeroed bitmap from one byte per row (null: all set).
    static void put_bits(std::uint8_t* bits, std::int64_t r0, std::int64_t rows, const std::uint8_t* valid) {
        std::int64_t r = 0;
        for (; r < rows && ((r0 + r) & 7); ++r)
            if (!valid || valid[r]) bitmap::set(bits, r0 + r);
        for (; r + 8 <= rows; r += 8) {
            std::uint8_t b = 0xff;
            if (valid) {
                std::uint64_t w;
                std::memcpy(&w, valid + r, 8);  // bytes 0 or 1: gather bit 0 of each byte
                b = static_cast<std::uint8_t>(((w & 0x0101010101010101ULL) * 0x0102040810204080ULL) >> 56);
            }
            bits[(r0 + r) >> 3] = b;
        }
        for (; r < rows; ++r)
            if (!valid || valid[r]) bitmap::set(bits, r0 + r);
    }

    Column all_na() {
        Column col = make_fixed(DType::Float64, n_);
        std::memset(col.values->data(), 0, static_cast<std::size_t>(n_ * 8));
        col.validity = Buffer::allocate(bitmap::num_bytes(n_), true);
        col.null_count = n_;
        each_part(release);
        return col;
    }

    // int64 or float64 from Part::num, or bool from Part::flag.
    Column fixed(DType dt, bool from_flags) {
        Column col = make_fixed(dt, n_);
        set_validity(col);  // before the values: it reads the Parts' NA rows; finish_fixed zeroes nothing yet
        std::byte* out = col.values->data();
        const std::int64_t size = itemsize(dt);
        each_chunk([&](Part& p, std::int64_t r0, std::size_t) {
            const void* src = from_flags ? static_cast<const void*>(p.flag.data()) : p.num.data();
            if (p.rows > 0) std::memcpy(out + r0 * size, src, static_cast<std::size_t>(p.rows * size));
            release(p);
        });
        return col;
    }

    // An inferred bool column: the Parts of bools, the others all NA.
    Column bools() {
        Column col = make_fixed(DType::Bool, n_);
        set_validity(col);
        auto* out = reinterpret_cast<std::uint8_t*>(col.values->data());
        each_chunk([&](Part& p, std::int64_t r0, std::size_t) {
            if (p.kind == Part::Bool) std::memcpy(out + r0, p.flag.data(), static_cast<std::size_t>(p.rows));
            else std::memset(out + r0, 0, static_cast<std::size_t>(p.rows));
            release(p);
        });
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
        std::int64_t nna = 0;
        for (const Chunk& ch : chunks_) nna += ch.parts[j_].nna;
        if (nna > 0) {
            col.validity = Buffer::allocate(bitmap::num_bytes(n_), true);
            auto* bits = reinterpret_cast<std::uint8_t*>(col.validity->data());
            for (std::size_t c = 0; c < chunks_.size(); ++c) {
                const Part& p = chunks_[c].parts[j_];
                put_bits(bits, row0_[c], p.rows, p.valid.empty() ? nullptr : p.valid.data());
            }
        }
        col.null_count = nna;
        auto* out = reinterpret_cast<char*>(col.values->data());
        each_chunk([&](Part& p, std::int64_t r0, std::size_t c) {
            const std::int64_t base = at[c];
            if (!p.bytes.empty()) std::memcpy(out + base, p.bytes.data(), p.bytes.size());
            for (std::int64_t i = 0; i < p.rows; ++i) offs[r0 + i + 1] = base + p.ends[static_cast<std::size_t>(i)];
            release(p);
        });
        return col;
    }

    std::vector<Chunk>& chunks_;
    const Reader& reader_;
    std::vector<std::int64_t> row0_;  // the first row of each chunk; back(): the row count
    std::int64_t n_ = 0;
    std::size_t j_ = 0;
};

}  // namespace

// Counts records (line feeds outside quotes) in a block of a file read in
// pieces: from the state `inside` at the block's start, up to `records`.
CsvScan csv_scan_records(const char* data, std::size_t size, std::int64_t records, bool inside) {
    CsvScan out{-1, 0, inside};
    if (records <= 0) {
        out.end = 0;
        return out;
    }
    const auto n = static_cast<std::int64_t>(size);
    for (std::int64_t base = 0; base < n; base += 64) {
        std::uint8_t tail[64];
        const auto* src = reinterpret_cast<const std::uint8_t*>(data + base);
        if (base + 64 > n) {
            std::memset(tail, 0, 64);
            std::memcpy(tail, src, static_cast<std::size_t>(n - base));
            src = tail;
        }
        uint8x16_t v[4];
        for (int j = 0; j < 4; ++j) v[j] = vld1q_u8(src + 16 * j);
        const std::uint64_t in = prefix_xor(mask64(v, '"')) ^ (out.inside ? ~std::uint64_t{0} : 0);
        out.inside = (in >> 63) & 1;
        std::uint64_t lf = mask64(v, '\n') & ~in;
        if (base + 64 > n) lf &= (std::uint64_t{1} << (n - base)) - 1;
        const std::int64_t c = __builtin_popcountll(lf);
        if (out.seen + c >= records) {
            for (std::int64_t need = records - out.seen; need > 1; --need) lf &= lf - 1;
            out.end = base + __builtin_ctzll(lf) + 1;
            out.seen = records;
            return out;
        }
        out.seen += c;
    }
    return out;
}

std::optional<FileBytes> read_fd(int fd, std::int64_t size) {
    constexpr std::int64_t kPiece = std::int64_t{1} << 22;
    const Unlocked unlocked(size);
    FileBytes out;
    // Uninitialized room: every byte kept is written by a read.
    out.data.reset(new char[static_cast<std::size_t>(std::max<std::int64_t>(size, 1))]);
    char* buf = out.data.get();
    const std::int64_t pieces = (size + kPiece - 1) / kPiece;
    std::vector<std::int64_t> got(static_cast<std::size_t>(pieces), 0);
    std::atomic<bool> failed{false};
    parallel_tasks(
        pieces,
        [&](std::int64_t t) {
            const std::int64_t from = t * kPiece, to = std::min(size, from + kPiece);
            std::int64_t at = from;
            while (at < to) {
                const ssize_t r = ::pread(fd, buf + at, static_cast<std::size_t>(to - at), static_cast<off_t>(at));
                if (r < 0 && errno == EINTR) continue;
                if (r < 0) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                if (r == 0) break;  // the file shrank: it ends here
                at += r;
            }
            got[static_cast<std::size_t>(t)] = at - from;
        },
        kPiece / 10  // a copy from the page cache: ~0.1 ns a byte
    );
    if (failed.load()) return std::nullopt;
    out.size = 0;  // the bytes up to the first short piece
    bool shrank = false;
    for (std::int64_t t = 0; t < pieces && !shrank; ++t) {
        out.size += got[static_cast<std::size_t>(t)];
        shrank = got[static_cast<std::size_t>(t)] < std::min(size, (t + 1) * kPiece) - t * kPiece;
    }
    if (shrank) return out;
    // The file may have grown since it was opened: read the rest.
    std::vector<char> rest;
    char more[1 << 16];
    for (off_t at = static_cast<off_t>(out.size);;) {
        const ssize_t r = ::pread(fd, more, sizeof more, at);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return std::nullopt;
        if (r == 0) break;
        rest.insert(rest.end(), more, more + r);
        at += r;
    }
    if (!rest.empty()) {
        std::unique_ptr<char[]> grown(new char[static_cast<std::size_t>(out.size) + rest.size()]);
        std::memcpy(grown.get(), buf, static_cast<std::size_t>(out.size));
        std::memcpy(grown.get() + out.size, rest.data(), rest.size());
        out.data = std::move(grown);
        out.size += static_cast<std::int64_t>(rest.size());
    }
    return out;
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
    std::vector<std::optional<DType>> forced(out_names.size());
    for (std::size_t j = 0; j < out_names.size(); ++j) {
        if (auto it = opt.dtypes.find(out_names[j]); it != opt.dtypes.end()) forced[j] = it->second;
        else if (opt.all_dtype) forced[j] = opt.all_dtype;
    }
    const NaSet na(opt.na_values.begin(), opt.na_values.end());
    const Reader reader(data, size, opt.sep, static_cast<std::int64_t>(names.size()), source, forced, na);

    // The chunks: their first records from the quote parity before them.
    const auto n = static_cast<std::int64_t>(size);
    const std::int64_t body = tok.offset();
    const bool parallel = parallel::available();
    const std::int64_t step = chunk_bytes(n - body, parallel);
    const std::int64_t nchunks = std::max<std::int64_t>(1, (n - body + step - 1) / step);
    auto chunk_begin = [&](std::int64_t c) { return std::min(n, body + c * step); };
    auto chunk_end = [&](std::int64_t c) { return c + 1 == nchunks ? n : chunk_begin(c + 1); };
    std::vector<std::uint8_t> odd(static_cast<std::size_t>(nchunks), 0);
    if (nchunks > 1)
        parallel_tasks(  // ~0.05 ns a byte
            nchunks - 1, [&](std::int64_t c) { odd[c] = odd_quotes(data, chunk_begin(c), chunk_end(c)); }, step / 20
        );
    std::vector<Chunk> chunks(static_cast<std::size_t>(nchunks));
    {
        bool inside = false;  // inside quotes where chunk c starts
        for (std::int64_t c = 0; c < nchunks; ++c) {
            Chunk& ch = chunks[static_cast<std::size_t>(c)];
            ch.first = c == 0 ? body : (chunk_begin(c) >= n ? n : first_record(data, n, chunk_begin(c), inside));
            inside ^= odd[static_cast<std::size_t>(c)];
        }
    }
    // A chunk: index its records, check their quoting, parse every output column.
    // A task's std::bad_alloc is rethrown as is (a MemoryError), not taken
    // for an error in the file.
    auto read_chunk = [&](std::int64_t c, std::int64_t limit) {
        Chunk& ch = chunks[static_cast<std::size_t>(c)];
        ch.parts.resize(out_names.size());
        const std::int64_t end = chunk_end(c);
        if (ch.first >= end || ch.first >= n) return;
        try {
            RecordIndex ix;
            reader.index_records(ch.first, end, limit, ix);
            reader.check_quotes(ch.first, ix);
            ch.rows = ix.records;
            for (std::size_t j = 0; j < out_names.size(); ++j) reader.parse_column(j, ch.first, ix, ch.parts[j]);
        } catch (const std::bad_alloc&) { throw; } catch (const std::exception& e) {
            ch.error = e.what();
        }
    };
    if (opt.nrows >= 0) {
        // Records in order, up to nrows: chunk by chunk until enough are read
        // (read_csv hands over only the bytes of those records, so this is
        // rare; see csv_scan_records).
        std::int64_t left = opt.nrows;
        for (std::int64_t c = 0; c < nchunks; ++c) {
            if (left <= 0) {
                chunks.resize(static_cast<std::size_t>(c));
                break;
            }
            read_chunk(c, left);
            left -= chunks[static_cast<std::size_t>(c)].rows;
            if (!chunks[static_cast<std::size_t>(c)].error.empty()) break;
        }
    } else {
        // ~1 ns a byte per thread (40 MB of mixed columns: 34 ms on one)
        parallel_tasks(nchunks, [&](std::int64_t c) { read_chunk(c, INT64_MAX); }, step);
    }
    for (const Chunk& ch : chunks)  // the first error in file order
        if (!ch.error.empty()) throw nb::value_error(ch.error.c_str());
    for (Chunk& ch : chunks)
        if (ch.parts.size() != out_names.size()) ch.parts.resize(out_names.size());

    ColumnBuilder builder(chunks, reader);
    std::vector<Column> cols;
    for (std::size_t j = 0; j < out_names.size(); ++j) cols.push_back(builder.build(j, forced[j], out_names[j]));
    return {out_names, cols};
}

}  // namespace sd::frame
