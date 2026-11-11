#include "ops/selection.hpp"

#include "parallel/pool.hpp"

#include <nanobind/nanobind.h>

#include <atomic>
#include <cstring>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd {

namespace {

const std::int64_t* positions_of(const Column& positions) {
    if (positions.dtype != DType::Int64 || positions.null_count != 0)
        throw nb::type_error("positions must be int64 without NA");
    return positions.data<std::int64_t>();
}

using parallel::for_chunks;
using parallel::kChunkRows;

std::size_t chunk_of(std::int64_t lo) { return static_cast<std::size_t>(lo / kChunkRows); }

// Per-chunk totals turned into starts: start[c] is the sum over the chunks
// before c, start.back() the whole sum. Summed in chunk order, so the layout
// is the same for any thread count.
template <class Count>
std::vector<std::int64_t> chunk_starts(std::int64_t n, Count count, std::int64_t cost = 1) {
    std::vector<std::int64_t> start(static_cast<std::size_t>(parallel::num_chunks(n)) + 1, 0);
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) { start[chunk_of(lo) + 1] = count(lo, hi); }, cost);
    for (std::size_t c = 1; c < start.size(); ++c) start[c] += start[c - 1];
    return start;
}

template <class T>
void gather(const Column& col, const std::int64_t* pos, std::int64_t n, Column& out, std::uint8_t* bits) {
    const T* in_ = col.data<T>();
    T* o_ = reinterpret_cast<T*>(out.values->data());
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {  // chunks are multiples of 64 rows: tasks never share a bitmap byte
        const T* in = in_;  // locals: see isna
        T* o = o_;
        for (std::int64_t i = lo; i < hi; ++i) {
            const std::int64_t p = pos[i];
            if (p < 0) {
                o[i] = T{};
                bitmap::clear(bits, i);
            } else {
                o[i] = in[p];
                if (bits && !col.is_valid(p)) bitmap::clear(bits, i);
            }
        }
    });
}

}  // namespace

Column take(const Column& col, const Column& positions) {
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
        auto size = [&](std::int64_t i) { return pos[i] < 0 ? 0 : static_cast<std::int64_t>(col.string_at(pos[i]).size()); };
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
    out.values = Buffer::allocate(n * item_size(col.dtype), /*zeroed=*/false);
    make_bits();
    switch (col.dtype) {
    case DType::Bool: gather<std::uint8_t>(col, pos, n, out, bits); break;
    case DType::Int64: gather<std::int64_t>(col, pos, n, out, bits); break;
    case DType::Float64: gather<double>(col, pos, n, out, bits); break;
    case DType::String: break;
    }
    finish_fixed(out);
    return out;
}

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
            for (std::int64_t i = j * 8; i < std::min(hi, j * 8 + 8); ++i) b |= static_cast<unsigned>(bytes[i]) << (i & 7);
            bits[j] = static_cast<std::uint8_t>(b);
        }
    });
}

}  // namespace

Column nonzero(const Column& mask) {
    if (mask.dtype != DType::Bool) throw nb::type_error("a mask must be bool");
    const auto* m = mask.data<std::uint8_t>();  // NA rows hold 0 (ARCHITECTURE.md §2.4): never kept
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
    if (mask.dtype != DType::Bool || mask.length != col.length) throw nb::type_error("filter: a bool mask of the column's length");
    // Strings: take() copies only the kept rows. Two branch-free string
    // compactions were measured and lost (M10 explainer §6).
    if (col.dtype == DType::String) return take(col, nonzero(mask));
    const std::int64_t n = col.length;
    const std::uint8_t* keep = mask.data<std::uint8_t>();  // NA rows hold 0 (ARCHITECTURE.md §2.4): never kept
    const std::vector<std::int64_t> start = kept_starts(keep, n);
    const std::int64_t count = start.back();
    Column out = make_fixed(col.dtype, count);
    switch (col.dtype) {
    case DType::Bool: compact(col.data<std::uint8_t>(), keep, n, start, reinterpret_cast<std::uint8_t*>(out.values->data())); break;
    case DType::Int64: compact(col.data<std::int64_t>(), keep, n, start, reinterpret_cast<std::int64_t*>(out.values->data())); break;
    case DType::Float64: compact(col.data<double>(), keep, n, start, reinterpret_cast<double*>(out.values->data())); break;
    case DType::String: break;
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

Column isna(const Column& col, bool want_na) {
    Column out = make_fixed(DType::Bool, col.length);
    auto* o = reinterpret_cast<std::uint8_t*>(out.values->data());
    const std::uint8_t flip = want_na ? 1 : 0;  // isna is 1 - valid, notna is valid
    auto bits = rebased_validity(col);
    if (!bits) {
        std::memset(o, flip ^ 1, static_cast<std::size_t>(col.length));
        return out;
    }
    const auto* b = reinterpret_cast<const std::uint8_t*>(bits->data());
    for_chunks(col.length, [b, o, flip](std::int64_t lo, std::int64_t hi) {
        // Locals, not captures: a closure's captures live in the closure
        // object, and a store through a uint8_t pointer may alias them, so
        // they would be reloaded every row and the loop would not vectorize
        // (the M10 aliasing problem again).
        std::uint8_t* out = o + lo;
        const std::int64_t len = hi - lo;
        bitmap::expand(b + lo / 8, len, out);  // eight rows per lookup (M10)
        if (flip)
            for (std::int64_t i = 0; i < len; ++i) out[i] ^= 1;
    });
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
    const std::int64_t size = item_size(col.dtype);
    Column out = make_fixed(col.dtype, col.length);
    std::byte* o = out.values->data();
    const std::byte* in = col.values->data() + col.offset * size;
    const std::byte* fill = value.values->data() + value.offset * size;
    for_chunks(col.length, [&](std::int64_t lo, std::int64_t hi) {
        std::memcpy(o + lo * size, in + lo * size, static_cast<std::size_t>((hi - lo) * size));
        for (std::int64_t i = lo; i < hi; ++i)
            if (!col.is_valid(i)) std::memcpy(o + i * size, fill, static_cast<std::size_t>(size));
    });
    return out;
}

bool equals(const Column& a, const Column& b) {
    if (a.dtype != b.dtype || a.length != b.length || a.null_count != b.null_count) return false;
    auto same = [&](std::int64_t i) {
        if (a.is_valid(i) != b.is_valid(i)) return false;
        if (!a.is_valid(i)) return true;
        if (a.dtype == DType::String) return a.string_at(i) == b.string_at(i);
        const std::int64_t size = item_size(a.dtype);
        return std::memcmp(a.values->data() + (a.offset + i) * size, b.values->data() + (b.offset + i) * size,
                           static_cast<std::size_t>(size)) == 0;
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
    const std::int64_t size = item_size(col.dtype);
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
// before or after the write (the snapshot rule, API.md §20).
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
            else bitmap::clear(bits, i);  // NA rows stay empty (ARCHITECTURE.md §2.4)
        }
        out.null_count = col.length - bitmap::count_set(bits, 0, col.length);
        if (out.null_count == 0) out.validity.reset();
        col = out;
        return;
    }

    detach(col, values.null_count > 0);
    const std::int64_t size = item_size(col.dtype);
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
    parallel::Unlocked unlocked(n);  // serial copies, without the GIL (API.md §20)
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
        const std::int64_t size = item_size(dt);
        out.values = Buffer::allocate(n * size, /*zeroed=*/false);
        std::int64_t row = 0;
        for (const Column& c : cols) {
            std::memcpy(out.values->data() + row * size, c.values->data() + c.offset * size,
                        static_cast<std::size_t>(c.length * size));
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

Column arange(std::int64_t start, std::int64_t step, std::int64_t n) {
    Column out = make_fixed(DType::Int64, n);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    for_chunks(n, [=](std::int64_t lo, std::int64_t hi) {
        std::int64_t* w = o;  // locals: see isna
        const std::int64_t a = start, d = step;
        for (std::int64_t i = lo; i < hi; ++i) w[i] = a + i * d;
    });
    return out;
}

}  // namespace sd
