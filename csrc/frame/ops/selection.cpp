#include "frame/ops/selection.hpp"

#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <atomic>
#include <climits>
#include <new>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd::frame {

namespace {

const std::int64_t* positions_of(const Column& positions) {
    if (positions.dtype != DType::Int64 || positions.null_count != 0)
        throw nb::type_error("positions must be int64 without NA");
    return positions.data<std::int64_t>();
}


std::size_t chunk_of(std::int64_t lo) { return static_cast<std::size_t>(lo / kChunkRows); }

// Per-chunk totals turned into starts: start[c] is the sum over the chunks
// before c, start.back() the whole sum. Summed in chunk order, so the layout
// is the same for any thread count.
template <class Count>
std::vector<std::int64_t> chunk_starts(std::int64_t n, Count count, std::int64_t cost = 1) {
    std::vector<std::int64_t> start(static_cast<std::size_t>(num_chunks(n)) + 1, 0);
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) { start[chunk_of(lo) + 1] = count(lo, hi); }, cost);
    for (std::size_t c = 1; c < start.size(); ++c) start[c] += start[c - 1];
    return start;
}

// The rows of a fixed-width column at `pos` (-1: NA) in one pass, no
// branch on the data: each row reads its value (row 0 for -1), keeps it
// where the position is valid and the source row has a value, and sets its
// validity bit; the bitmap is written a 64-bit word at a time. Returns the
// NA count, or -1 when a position is out of range (then the output is
// dropped). NA rows get 0, as every NA row (docs/design/frame.md §4.5):
// source NA rows hold 0 already.
template <class T>
std::int64_t gather(const Column& col, const std::int64_t* pos, std::int64_t n, T* out, std::uint64_t* words) {
    const T* in_ = col.data<T>();
    const std::uint8_t* src = col.validity_bits();
    const std::int64_t off = col.offset;
    const auto length = static_cast<std::uint64_t>(col.length);
    std::vector<std::int64_t> nna(static_cast<std::size_t>(num_chunks(n)), 0);
    std::atomic<bool> bad_any{false};
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {  // chunks are whole 64-bit words of the bitmap
        const T* in = in_;  // locals: see isna
        T* o = out;
        std::uint64_t* w = words;
        std::uint64_t bad = 0;
        std::int64_t na = 0;
        for (std::int64_t b = lo; b < hi; b += 64) {
            std::uint64_t m = 0;
            const std::int64_t e = std::min(hi, b + 64);
            for (std::int64_t i = b; i < e; ++i) {
                const std::int64_t p = pos[i];
                bad |= static_cast<std::uint64_t>(p + 1) > length;  // -1 .. length - 1
                const bool has = p >= 0;
                const std::int64_t q = has ? p : 0;
                const bool valid = has && (!src || bitmap::get(src, off + q));
                const T v = in[q];
                o[i] = valid ? v : T{};
                m |= static_cast<std::uint64_t>(valid) << (i - b);
            }
            w[b >> 6] = m;
            na += (e - b) - __builtin_popcountll(m);
        }
        nna[static_cast<std::size_t>(lo / kChunkRows)] = na;
        if (bad) bad_any.store(true, std::memory_order_relaxed);
    });
    if (bad_any.load()) return -1;
    std::int64_t total = 0;
    for (std::int64_t k : nna) total += k;
    return total;
}

}  // namespace

namespace {
Column take_fixed(const Column& col, const Column& positions);
}  // namespace

Column take(const Column& col, const Column& positions) {
    if (col.dtype != DType::String) return take_fixed(col, positions);
    const std::int64_t* pos = positions_of(positions);
    const std::int64_t n = positions.length;
    std::atomic<bool> any_missing{false}, out_of_range{false};
    const std::int64_t length = col.length;
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        bool missing = false, bad = false;
        for (std::int64_t i = lo; i < hi; ++i) {
            bad |= pos[i] < -1 || pos[i] >= length;
            missing |= pos[i] == -1;
        }
        if (missing) any_missing = true;
        if (bad) out_of_range = true;
    });
    if (out_of_range) throw nb::index_error("take: position out of range");
    Column out;
    out.dtype = col.dtype;
    out.length = n;
    std::uint8_t* bits = nullptr;
    auto make_bits = [&] {
        if (col.validity || any_missing) bits = add_validity(out);
    };
    if (col.dtype == DType::String) {
        out.offsets = Buffer::allocate((n + 1) * 8, /*zeroed=*/false);
        make_bits();
        auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
        auto size = [&](std::int64_t i) {
            return pos[i] < 0 ? 0 : static_cast<std::int64_t>(col.string_at(pos[i]).size());
        };
        const std::vector<std::int64_t> start = chunk_starts(n, [&](std::int64_t lo, std::int64_t hi) {
            std::int64_t t = 0;
            for (std::int64_t i = lo; i < hi; ++i) t += size(i);
            return t;
        });
        out.values = Buffer::allocate(start.back(), /*zeroed=*/false);
        auto* bytes = reinterpret_cast<char*>(out.values->data());
        offs[0] = 0;
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            std::int64_t at = start[chunk_of(lo)];
            for (std::int64_t i = lo; i < hi; ++i) {
                if (pos[i] < 0) {
                    bitmap::clear(bits, i);
                } else {
                    std::string_view sv = col.string_at(pos[i]);
                    std::memcpy(bytes + at, sv.data(), sv.size());
                    at += static_cast<std::int64_t>(sv.size());
                    if (bits && !col.is_valid(pos[i])) bitmap::clear(bits, i);
                }
                offs[i + 1] = at;
            }
        });
        out.null_count = bits ? n - bitmap::count_set(bits, 0, n) : 0;
        if (out.null_count == 0) out.validity.reset();
        return out;
    }
    return out;
}

namespace {

Column take_fixed(const Column& col, const Column& positions) {
    const std::int64_t* pos = positions_of(positions);
    const std::int64_t n = positions.length;
    Column out;
    out.dtype = col.dtype;
    out.length = n;
    out.values = Buffer::allocate(n * itemsize(col.dtype), /*zeroed=*/false);
    if (n == 0) return out;
    if (col.length == 0) {  // only -1 can be valid: every row NA
        for (std::int64_t i = 0; i < n; ++i)
            if (pos[i] != -1) throw nb::index_error("take: position out of range");
        std::memset(out.values->data(), 0, static_cast<std::size_t>(n * itemsize(col.dtype)));
        out.validity = Buffer::allocate(bitmap::num_bytes(n), /*zeroed=*/true);
        out.null_count = n;
        return out;
    }
    // The bitmap in whole words (rows past n are never read); dropped when nothing is NA.
    auto validity = Buffer::allocate(((n + 63) / 64) * 8, /*zeroed=*/false);
    auto* words = reinterpret_cast<std::uint64_t*>(validity->data());
    std::int64_t nna = 0;
    switch (col.dtype) {
        case DType::Bool: nna = gather(col, pos, n, reinterpret_cast<std::uint8_t*>(out.values->data()), words); break;
        case DType::Int64: nna = gather(col, pos, n, reinterpret_cast<std::int64_t*>(out.values->data()), words); break;
        case DType::Float64: nna = gather(col, pos, n, reinterpret_cast<double*>(out.values->data()), words); break;
        case DType::String:
        case DType::Complex128: break;  // not here / not a frame dtype
    }
    if (nna < 0) throw nb::index_error("take: position out of range");
    out.null_count = nna;
    if (nna > 0) out.validity = std::move(validity);
    return out;
}

}  // namespace

namespace {

// Branch-free compaction of rows [lo, hi) to `out`: every row is written at
// `k`, and `k` moves on only when the row is kept, so the loop has no
// data-dependent branch. The loop stops at the chunk's last kept row, so
// every write lands on one of this chunk's own output slots: the chunks run
// in parallel next to each other with no slot written twice.
template <class Get>
void compact_rows(const std::uint8_t* keep, std::int64_t lo, std::int64_t hi, Get get, auto* out) {
    std::int64_t last = hi - 1;
    while (last >= lo && !keep[last]) --last;
    std::int64_t k = 0;
    for (std::int64_t i = lo; i <= last; ++i) {
        out[k] = get(i);
        k += keep[i];
    }
}

// Kept rows of a mask before each chunk (see chunk_starts).
std::vector<std::int64_t> kept_starts(const std::uint8_t* keep, std::int64_t n) {
    return chunk_starts(n, [keep](std::int64_t lo, std::int64_t hi) {
        std::int64_t k = 0;
        for (std::int64_t i = lo; i < hi; ++i) k += keep[i];
        return k;
    });
}

template <class T>
void compact(const T* in, const std::uint8_t* keep, std::int64_t n, const std::vector<std::int64_t>& start, T* out) {
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        compact_rows(keep, lo, hi, [in](std::int64_t i) { return in[i]; }, out + start[chunk_of(lo)]);
    });
}

// bytes[0, n) of 0/1 into bits [0, n), eight rows per output byte.
void pack_bits(const std::uint8_t* bytes, std::int64_t n, std::uint8_t* bits) {
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        for (std::int64_t j = lo / 8; j < bitmap::num_bytes(hi); ++j) {
            unsigned b = 0;
            for (std::int64_t i = j * 8; i < std::min(hi, j * 8 + 8); ++i)
                b |= static_cast<unsigned>(bytes[i]) << (i & 7);
            bits[j] = static_cast<std::uint8_t>(b);
        }
    });
}

}  // namespace

Column nonzero(const Column& mask) {
    if (mask.dtype != DType::Bool) throw nb::type_error("a mask must be bool");
    const auto* m = mask.data<std::uint8_t>();  // NA rows hold 0 (docs/design/frame.md §4.5): never kept
    const std::int64_t n = mask.length;
    const std::vector<std::int64_t> start = kept_starts(m, n);
    Column out = make_fixed(DType::Int64, start.back());
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        compact_rows(m, lo, hi, [](std::int64_t i) { return i; }, o + start[chunk_of(lo)]);
    });
    return out;
}

Column filter(const Column& col, const Column& mask) {
    if (mask.dtype != DType::Bool || mask.length != col.length)
        throw nb::type_error("filter: a bool mask of the column's length");
    // Strings: take() copies only the kept rows. Two branch-free string
    // compactions were measured and lost (docs/design/frame.md §7.3).
    if (col.dtype == DType::String) return take(col, nonzero(mask));
    const std::int64_t n = col.length;
    const std::uint8_t* keep = mask.data<std::uint8_t>();  // NA rows hold 0 (docs/design/frame.md §4.5): never kept
    const std::vector<std::int64_t> start = kept_starts(keep, n);
    const std::int64_t count = start.back();
    Column out = make_fixed(col.dtype, count);
    switch (col.dtype) {
        case DType::Bool:
            compact(col.data<std::uint8_t>(), keep, n, start, reinterpret_cast<std::uint8_t*>(out.values->data()));
            break;
        case DType::Int64:
            compact(col.data<std::int64_t>(), keep, n, start, reinterpret_cast<std::int64_t*>(out.values->data()));
            break;
        case DType::Float64:
            compact(col.data<double>(), keep, n, start, reinterpret_cast<double*>(out.values->data()));
            break;
        case DType::String: break;
        case DType::Complex128: break;  // not a frame dtype
    }
    if (col.validity) {  // compact the validity as bytes, then pack it into bits
        std::vector<std::uint8_t> valid = valid_bytes(col), kept(static_cast<std::size_t>(count));
        compact(valid.data(), keep, n, start, kept.data());
        out.validity = Buffer::allocate(bitmap::num_bytes(count), /*zeroed=*/false);
        pack_bits(kept.data(), count, reinterpret_cast<std::uint8_t*>(out.validity->data()));
        finish_fixed(out);
    }
    return out;
}

std::vector<Column> filter_columns(const std::vector<Column>& cols, const Column& mask, const Column& positions) {
    if (mask.dtype != DType::Bool) throw nb::type_error("filter: a bool mask");
    const std::int64_t n = mask.length;
    for (const Column& c : cols)
        if (c.length != n) throw nb::type_error("filter: a bool mask of the column's length");
    const std::uint8_t* keep = mask.data<std::uint8_t>();  // NA rows hold 0: never kept
    std::vector<Column> out(cols.size());
    // Numbers and bools without NA are compacted chunk by chunk, all
    // columns at once (one task per column and chunk); the others (strings,
    // columns with NA) one after another, each split by take() or filter()
    // itself (as one task, a string column ran on one thread). The mask's
    // per-chunk counts are counted once for every column.
    std::vector<std::size_t> plain, other;
    for (std::size_t j = 0; j < cols.size(); ++j)
        (cols[j].dtype != DType::String && !cols[j].validity ? plain : other).push_back(j);
    // Up to 2/3 of the rows kept: gathering the kept rows by position reads
    // and writes less than compacting every row (a 100-column frame with
    // half its rows kept: 1.6 ms instead of 2.8 on one thread).
    const std::int64_t count = positions.length;
    const bool gather = count * 3 <= n * 2;
    const std::int64_t* pos = positions.data<std::int64_t>();
    std::vector<std::int64_t> start;
    if (!plain.empty() && !gather) start = kept_starts(keep, n);
    for (std::size_t j : plain) out[j] = make_fixed(cols[j].dtype, count);
    for (std::size_t j : other)
        out[j] = cols[j].dtype == DType::String ? take(cols[j], positions) : filter(cols[j], mask);
    const std::int64_t C = num_chunks(gather ? count : n);
    const auto nplain = static_cast<std::int64_t>(plain.size()) * C;
    auto task = [&](std::int64_t t) {
        const std::size_t j = plain[static_cast<std::size_t>(t / C)];
        const std::int64_t c = t % C, lo = c * kChunkRows, hi = std::min(gather ? count : n, lo + kChunkRows);
        const Column& col = cols[j];
        auto run = [&](const auto* in, auto* o) {
            if (gather) {
                for (std::int64_t k = lo; k < hi; ++k) o[k] = in[pos[k]];  // positions from nonzero: in range
                return;
            }
            compact_rows(keep, lo, hi, [in](std::int64_t i) { return in[i]; }, o + start[static_cast<std::size_t>(c)]);
        };
        std::byte* o = out[j].values->data();
        switch (col.dtype) {
            case DType::Bool: run(col.data<std::uint8_t>(), reinterpret_cast<std::uint8_t*>(o)); break;
            case DType::Int64: run(col.data<std::int64_t>(), reinterpret_cast<std::int64_t*>(o)); break;
            case DType::Float64: run(col.data<double>(), reinterpret_cast<double*>(o)); break;
            default: break;
        }
    };
    const std::int64_t ntasks = nplain;
    const std::int64_t work = (gather ? count : n) * static_cast<std::int64_t>(plain.size());
    if (work < kParallelThreshold) {
        for (std::int64_t t = 0; t < ntasks; ++t) task(t);
    } else {
        run_unlocked(ntasks, task, work);
    }
    return out;
}

Column isna(const Column& col, bool want_na) {
    Column out = make_fixed(DType::Bool, col.length);
    auto* o = reinterpret_cast<std::uint8_t*>(out.values->data());
    const std::uint8_t flip = want_na ? 1 : 0;  // isna is 1 - valid, notna is valid
    auto bits = rebased_validity(col);
    if (!bits) {
        const Unlocked unlocked(col.length);
        std::memset(o, flip ^ 1, static_cast<std::size_t>(col.length));
        return out;
    }
    const auto* b = reinterpret_cast<const std::uint8_t*>(bits->data());
    for_chunks(col.length, [b, o, flip](std::int64_t lo, std::int64_t hi) {
        // Locals, not captures: a closure's captures live in the closure
        // object, and a store through a uint8_t pointer may alias them, so
        // they would be reloaded every row and the loop would not vectorize
        // (docs/design/frame.md §7.6).
        std::uint8_t* out = o + lo;
        const std::int64_t len = hi - lo;
        bitmap::expand(b + lo / 8, len, out);  // eight rows per lookup (docs/design/frame.md §4.4)
        if (flip)
            for (std::int64_t i = 0; i < len; ++i) out[i] ^= 1;
    });
    return out;
}

Column isnan(const Column& col) {
    Column out = make_fixed(DType::Bool, col.length);
    out.validity = rebased_validity(col);
    auto* o = reinterpret_cast<std::uint8_t*>(out.values->data());
    if (col.dtype != DType::Float64) {
        std::memset(o, 0, static_cast<std::size_t>(col.length));
    } else {
        const double* x = col.data<double>();
        for_chunks(col.length, [x, o](std::int64_t lo, std::int64_t hi) {
            const double* xx = x;  // locals: see isna
            std::uint8_t* oo = o;
            for (std::int64_t i = lo; i < hi; ++i) oo[i] = xx[i] != xx[i];  // NaN is the one value unequal to itself
        });
    }
    finish_fixed(out);  // NA rows hold 0 (docs/design/frame.md §4.5)
    return out;
}

Column fill_na(const Column& col, const Column& value) {
    if (value.dtype != col.dtype || value.length != 1 || value.null_count != 0)
        throw nb::type_error("fill value must be one value of the column's dtype");
    if (col.null_count == 0) return col;
    if (col.dtype == DType::String) {
        const std::string_view fill = value.string_at(0);
        Column out;
        out.dtype = DType::String;
        out.length = col.length;
        out.offsets = Buffer::allocate((col.length + 1) * 8, /*zeroed=*/false);
        auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
        auto row = [&](std::int64_t i) { return col.is_valid(i) ? col.string_at(i) : fill; };
        const std::vector<std::int64_t> start = chunk_starts(col.length, [&](std::int64_t lo, std::int64_t hi) {
            std::int64_t t = 0;
            for (std::int64_t i = lo; i < hi; ++i) t += static_cast<std::int64_t>(row(i).size());
            return t;
        });
        out.values = Buffer::allocate(start.back(), /*zeroed=*/false);
        auto* bytes = reinterpret_cast<char*>(out.values->data());
        offs[0] = 0;
        for_chunks(col.length, [&](std::int64_t lo, std::int64_t hi) {
            std::int64_t at = start[chunk_of(lo)];
            for (std::int64_t i = lo; i < hi; ++i) {
                std::string_view sv = row(i);
                std::memcpy(bytes + at, sv.data(), sv.size());
                at += static_cast<std::int64_t>(sv.size());
                offs[i + 1] = at;
            }
        });
        return out;
    }
    // One select per row, no branch: the chunk's validity expanded to a
    // byte per row (a bitmap starting on a byte), or read bit by bit.
    Column out = make_fixed(col.dtype, col.length);
    const std::uint8_t* bits = col.validity_bits();
    const std::int64_t off = col.offset;
    auto run = [&](const auto* in_, auto fill, auto* out_) {
        for_chunks(col.length, [=](std::int64_t lo, std::int64_t hi) {
            const auto* in = in_;  // locals: see isna
            auto* o = out_;
            const std::int64_t len = hi - lo;
            if ((off + lo) % 8 == 0) {
                std::uint8_t m[kChunkRows];
                bitmap::expand(bits + (off + lo) / 8, len, m);
                for (std::int64_t i = 0; i < len; ++i) o[lo + i] = m[i] ? in[lo + i] : fill;
            } else {
                for (std::int64_t i = lo; i < hi; ++i) o[i] = bitmap::get(bits, off + i) ? in[i] : fill;
            }
        });
    };
    switch (col.dtype) {
        case DType::Bool:
            run(col.data<std::uint8_t>(), value.data<std::uint8_t>()[0],
                reinterpret_cast<std::uint8_t*>(out.values->data()));
            break;
        case DType::Int64:
            run(col.data<std::int64_t>(), value.data<std::int64_t>()[0],
                reinterpret_cast<std::int64_t*>(out.values->data()));
            break;
        case DType::Float64:
            run(col.data<double>(), value.data<double>()[0], reinterpret_cast<double*>(out.values->data()));
            break;
        default: break;
    }
    return out;
}

bool equals(const Column& a, const Column& b) {
    if (a.dtype != b.dtype || a.length != b.length || a.null_count != b.null_count) return false;
    auto same = [&](std::int64_t i) {
        if (a.is_valid(i) != b.is_valid(i)) return false;
        if (!a.is_valid(i)) return true;
        if (a.dtype == DType::String) return a.string_at(i) == b.string_at(i);
        const std::int64_t size = itemsize(a.dtype);
        return std::memcmp(
                   a.values->data() + (a.offset + i) * size, b.values->data() + (b.offset + i) * size,
                   static_cast<std::size_t>(size)
               ) == 0;
    };
    std::atomic<bool> differ{false};
    for_chunks(a.length, [&](std::int64_t lo, std::int64_t hi) {
        for (std::int64_t i = lo; i < hi && !differ.load(std::memory_order_relaxed); ++i)
            if (!same(i)) differ = true;
    });
    return !differ;
}

namespace {

// Gives `col` values and validity Buffers that nothing else holds, starting
// at offset 0, copying its rows when needed. Afterwards writes cannot be
// seen through any other object.
void detach(Column& col, bool need_validity) {
    const bool shared = col.values.use_count() > 1 || (col.validity && col.validity.use_count() > 1);
    const bool add_bits = need_validity && !col.validity;
    if (!shared && !(add_bits && col.offset != 0)) {
        if (add_bits) add_validity(col);
        return;
    }
    const std::int64_t size = itemsize(col.dtype);
    auto values = Buffer::allocate(col.length * size, /*zeroed=*/false);
    std::memcpy(values->data(), col.values->data() + col.offset * size, static_cast<std::size_t>(col.length * size));
    auto validity = rebased_validity(col);
    col.values = values;
    col.validity = validity;
    col.offset = 0;
    if (add_bits) add_validity(col);
}

const std::int64_t* checked_positions(const Column& positions, std::int64_t length) {
    const std::int64_t* pos = positions_of(positions);
    for (std::int64_t i = 0; i < positions.length; ++i)
        if (pos[i] < 0 || pos[i] >= length) throw nb::index_error("set_rows: position out of range");
    return pos;
}

}  // namespace

// Runs with the GIL held from start to end, even for many rows: it may
// write `col`'s Buffers in place (when nothing else holds them), and a
// kernel of another Python thread must not start reading them meanwhile.
// Holding the GIL is what makes every other call see the column either
// before or after the write (the snapshot rule, docs/design/frame.md §5.3).
void set_rows(Column& col, const Column& positions, const Column& values) {
    if (values.dtype != col.dtype) throw nb::type_error("set_rows: values must have the column's dtype");
    const bool broadcast = values.length == 1;
    if (!broadcast && values.length != positions.length) throw nb::value_error("set_rows: one value per position");
    const std::int64_t* pos = checked_positions(positions, col.length);
    const std::int64_t n = positions.length;
    if (n == 0) return;

    if (col.dtype == DType::String) {
        // Variable width: build new Buffers. src[row] is the value index that
        // replaces the row, or -1 to keep it (the last write to a row wins).
        std::vector<std::int64_t> src(static_cast<std::size_t>(col.length), -1);
        for (std::int64_t k = 0; k < n; ++k) src[pos[k]] = broadcast ? 0 : k;
        Column out;
        out.dtype = DType::String;
        out.length = col.length;
        out.offsets = Buffer::allocate((col.length + 1) * 8, /*zeroed=*/false);
        std::uint8_t* bits = add_validity(out);
        auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
        offs[0] = 0;
        auto row = [&](std::int64_t i) -> std::pair<bool, std::string_view> {
            if (src[i] < 0) return {col.is_valid(i), col.string_at(i)};
            return {values.is_valid(src[i]), values.string_at(src[i])};
        };
        for (std::int64_t i = 0; i < col.length; ++i) {
            auto [valid, sv] = row(i);
            offs[i + 1] = offs[i] + (valid ? static_cast<std::int64_t>(sv.size()) : 0);
        }
        out.values = Buffer::allocate(offs[col.length], /*zeroed=*/false);
        for (std::int64_t i = 0; i < col.length; ++i) {
            auto [valid, sv] = row(i);
            if (valid) std::memcpy(out.values->data() + offs[i], sv.data(), sv.size());
            else bitmap::clear(bits, i);  // NA rows stay empty (docs/design/frame.md §4.5)
        }
        out.null_count = col.length - bitmap::count_set(bits, 0, col.length);
        if (out.null_count == 0) out.validity.reset();
        col = out;
        return;
    }

    detach(col, values.null_count > 0);
    const std::int64_t size = itemsize(col.dtype);
    std::byte* v = col.values->data() + col.offset * size;
    auto* bits = col.validity ? reinterpret_cast<std::uint8_t*>(col.validity->data()) : nullptr;
    const std::byte* in = values.values->data() + values.offset * size;
    for (std::int64_t k = 0; k < n; ++k) {
        const std::int64_t j = broadcast ? 0 : k;
        const std::int64_t row = col.offset + pos[k];
        if (values.is_valid(j)) {
            std::memcpy(v + pos[k] * size, in + j * size, static_cast<std::size_t>(size));
            if (bits) bitmap::set(bits, row);
        } else {
            std::memset(v + pos[k] * size, 0, static_cast<std::size_t>(size));
            bitmap::clear(bits, row);
        }
    }
    if (bits) {
        col.null_count = col.length - bitmap::count_set(bits, col.offset, col.length);
        if (col.null_count == 0) col.validity.reset();
    }
}

Column concat(const std::vector<Column>& cols) {
    if (cols.empty()) throw nb::value_error("concat needs at least one column");
    const DType dt = cols[0].dtype;
    std::int64_t n = 0, nulls = 0;
    for (const Column& c : cols) {
        if (c.dtype != dt) throw nb::type_error("concat: columns must have one dtype");
        n += c.length;
        nulls += c.null_count;
    }
    Column out;
    out.dtype = dt;
    out.length = n;
    std::uint8_t* bits = nulls ? add_validity(out) : nullptr;
    Unlocked unlocked(n);  // serial copies, without the GIL (docs/design/frame.md §11.4)
    if (dt == DType::String) {
        out.offsets = Buffer::allocate((n + 1) * 8, /*zeroed=*/false);
        auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
        offs[0] = 0;
        std::int64_t row = 0;
        for (const Column& c : cols)
            for (std::int64_t i = 0; i < c.length; ++i, ++row)
                offs[row + 1] = offs[row] + static_cast<std::int64_t>(c.string_at(i).size());
        out.values = Buffer::allocate(offs[n], /*zeroed=*/false);
        row = 0;
        for (const Column& c : cols)
            for (std::int64_t i = 0; i < c.length; ++i, ++row) {
                std::string_view sv = c.string_at(i);
                std::memcpy(out.values->data() + offs[row], sv.data(), sv.size());
            }
    } else {
        // The copies in chunks of rows, on the pool for long parts (each
        // part's chunks write their own bytes).
        const std::int64_t size = itemsize(dt);
        out.values = Buffer::allocate(n * size, /*zeroed=*/false);
        std::int64_t row = 0;
        for (const Column& c : cols) {
            std::byte* to = out.values->data() + row * size;
            const std::byte* from = c.values->data() + c.offset * size;
            for_chunks(c.length, [=](std::int64_t lo, std::int64_t hi) {
                std::memcpy(to + lo * size, from + lo * size, static_cast<std::size_t>((hi - lo) * size));
            });
            row += c.length;
        }
    }
    if (bits) {
        std::int64_t row = 0;
        for (const Column& c : cols)
            for (std::int64_t i = 0; i < c.length; ++i, ++row)
                if (!c.is_valid(i)) bitmap::clear(bits, row);
        out.null_count = nulls;
    }
    return out;
}

Column concat_ranges(
    const std::vector<std::int64_t>& starts,
    const std::vector<std::int64_t>& steps,
    const std::vector<std::int64_t>& lengths
) {
    if (starts.size() != steps.size() || starts.size() != lengths.size())
        throw nb::value_error("concat_ranges: one start, step and length per range");
    std::int64_t n = 0;
    for (std::int64_t len : lengths) n += len;
    Column out = make_fixed(DType::Int64, n);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    std::int64_t at = 0;
    for (std::size_t r = 0; r < starts.size(); ++r) {
        const auto a = static_cast<std::uint64_t>(starts[r]), d = static_cast<std::uint64_t>(steps[r]);
        std::int64_t* w = o + at;
        for_chunks(lengths[r], [=](std::int64_t lo, std::int64_t hi) {
            // Locals, not captures: int64 stores may alias the closure's
            // uint64 members, which would be reloaded every row (see isna).
            std::int64_t* ww = w;
            const std::uint64_t aa = a, dd = d;
            for (std::int64_t i = lo; i < hi; ++i)
                ww[i] = static_cast<std::int64_t>(aa + static_cast<std::uint64_t>(i) * dd);
        });
        at += lengths[r];
    }
    return out;
}

Column arange(std::int64_t start, std::int64_t step, std::int64_t n) {
    Column out = make_fixed(DType::Int64, n);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    for_chunks(n, [=](std::int64_t lo, std::int64_t hi) {
        std::int64_t* w = o;  // locals: see isna
        // In uint64: i * step may overflow int64 even when start + i * step does not.
        const auto a = static_cast<std::uint64_t>(start), d = static_cast<std::uint64_t>(step);
        for (std::int64_t i = lo; i < hi; ++i) w[i] = static_cast<std::int64_t>(a + static_cast<std::uint64_t>(i) * d);
    });
    return out;
}

std::pair<Column, std::int64_t> wrap_positions(const Column& positions, std::int64_t n) {
    const std::int64_t* pos = positions_of(positions);
    const std::int64_t len = positions.length;
    Column out = make_fixed(DType::Int64, len);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    parallel::FirstRow bad;
    for_chunks(len, [=, &bad](std::int64_t lo, std::int64_t hi) {
        const std::int64_t* p = pos;  // locals: see isna
        std::int64_t* w = o;
        const std::int64_t size = n;
        for (std::int64_t i = lo; i < hi; ++i) {
            const std::int64_t v = p[i];
            if (v < -size || v >= size) {
                bad.offer(i);
                return;
            }
            w[i] = v < 0 ? v + size : v;
        }
    });
    return {std::move(out), bad.any() ? bad.row() : -1};
}

Column full(const Column& value, std::int64_t n) {
    if (value.length != 1 || n < 0) throw nb::value_error("full: one value and a length >= 0");
    const bool na = value.null_count != 0;
    Column out;
    if (value.dtype == DType::String) {
        const std::string_view s = na ? std::string_view{} : value.string_at(0);
        const auto size = static_cast<std::int64_t>(s.size());
        if (size != 0 && n > INT64_MAX / size) throw std::bad_alloc();
        out.dtype = DType::String;
        out.length = n;
        out.values = Buffer::allocate(size * n, /*zeroed=*/false);
        out.offsets = Buffer::allocate((n + 1) * 8, /*zeroed=*/false);
        auto* bytes = reinterpret_cast<char*>(out.values->data());
        auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
        offs[0] = 0;
        for_chunks(n, [=](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t i = lo; i < hi; ++i) {
                std::memcpy(bytes + i * size, s.data(), static_cast<std::size_t>(size));
                offs[i + 1] = (i + 1) * size;
            }
        });
    } else {
        out = make_fixed(value.dtype, n);
        const std::int64_t size = itemsize(value.dtype);
        std::byte* o = out.values->data();
        std::byte one[8] = {};  // NA rows hold 0 (docs/design/frame.md §4.5)
        if (!na) std::memcpy(one, value.values->data() + value.offset * size, static_cast<std::size_t>(size));
        if (size == 1) {
            const Unlocked unlocked(n);
            std::memset(o, static_cast<int>(one[0]), static_cast<std::size_t>(n));
        } else {
            std::int64_t word;
            std::memcpy(&word, one, 8);
            for_chunks(n, [=](std::int64_t lo, std::int64_t hi) {
                auto* w = reinterpret_cast<std::int64_t*>(o);
                for (std::int64_t i = lo; i < hi; ++i) w[i] = word;
            });
        }
    }
    if (na && n > 0) {
        out.validity = Buffer::allocate(bitmap::num_bytes(n), /*zeroed=*/true);
        out.null_count = n;
    }
    return out;
}

}  // namespace sd::frame
