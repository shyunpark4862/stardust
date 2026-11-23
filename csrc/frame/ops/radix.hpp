// Stable LSD radix sort on 64-bit keys and ordering the values of a column
// (docs/design/frame.md §9.3, §9.4), shared by sort_values and groupby's
// group order.
#pragma once

#include "frame/column.hpp"
#include "frame/ops/hash.hpp"
#include "frame/parallel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

namespace sd::frame {

// An unsigned 64-bit key whose order is the value order (docs/api/frame.md §8.2).
// int64: flip the sign bit, so negatives come first. float64: -0.0 becomes
// 0.0 and every NaN one NaN (both above +inf once mapped); then a positive
// float gets its sign bit set and a negative one all bits flipped -- the
// classic map from IEEE order to unsigned integer order.
inline std::uint64_t radix_key(const Column& c, std::int64_t i) {
    switch (c.dtype) {
        case DType::Int64: return static_cast<std::uint64_t>(c.data<std::int64_t>()[i]) ^ (std::uint64_t{1} << 63);
        case DType::Bool: return c.data<std::uint8_t>()[i];
        case DType::Float64: {
            const std::uint64_t bits = float_key(c.data<double>()[i]);  // NaN and -0.0 normalized (hash.hpp)
            return (bits >> 63) ? ~bits : bits | (std::uint64_t{1} << 63);
        }
        case DType::String: break;
        case DType::Complex128: break;  // not a frame dtype
    }
    return 0;
}

// The first 8 bytes of a string as a big-endian number (shorter strings
// padded with zero bytes): comparing prefixes compares the strings' first 8
// bytes in code point order (UTF-8 byte order).
inline std::uint64_t string_prefix(std::string_view s) {
    std::uint8_t b[8] = {};
    std::memcpy(b, s.data(), std::min<std::size_t>(8, s.size()));
    std::uint64_t k = 0;
    for (int j = 0; j < 8; ++j) k = (k << 8) | b[j];
    return k;
}

// Stable LSD radix sort of `rows` by 64-bit keys, least significant byte
// first. `digits` (eight by default) bounds the bytes that can differ: the
// caller sets fewer only from what the keys are (bools: one byte; ranks
// below a count of distinct values: the bytes of that count).
//
// One read of the keys first (scan_keys) finds what the data needs:
//  - keys already in order (non-decreasing): nothing moves;
//  - keys in reverse order (non-increasing): the runs of equal keys are
//    reversed as wholes, each keeping its own order (reverse_runs);
//  - otherwise only the bytes that differ between keys get a pass: either
//    the bytes of (key - min) up to the highest set one, or the bytes where
//    some keys differ, whichever is fewer. A byte that is the same in every
//    key orders nothing, so a pass over it would leave every row in place.
// Every path ends in the one stable order, so the data only sets the cost
// (int64 keys 0..10^6 take 3 passes instead of 8, an already sorted column
// one read).
//
// In parallel (docs/design/frame.md §9.5): each pass counts the digit per
// chunk of positions, gives every (digit, chunk) pair its place -- digit by
// digit, and within a digit chunk by chunk -- and then each chunk scatters
// its elements to its own places. An element's place is exactly the one the
// serial pass gives it, so the order is the same for any thread count.
//
// Serial: when the sort is not split -- one thread, or fewer keys than
// kParallelThreshold (2^17) -- all the histograms come from one read of the
// keys, with no per-chunk tables.

// From this many keys a serial sort may use digits wider than a byte
// (plan_passes): below it, clearing and summing 2,048-entry histograms
// costs more than the passes it saves.
inline constexpr std::int64_t kWideRadixKeys = std::int64_t{1} << 15;

// What one read of the keys tells the sort.
struct KeyScan {
    bool nondecreasing = true;
    bool nonincreasing = true;
    std::uint64_t lo = ~std::uint64_t{0}, hi = 0;          // min and max
    std::uint64_t ones = 0, zeros = ~std::uint64_t{0};      // OR and AND of all keys
};

inline KeyScan scan_range(const std::uint64_t* k, std::int64_t lo, std::int64_t hi) {
    KeyScan s;
    std::uint64_t mn = ~std::uint64_t{0}, mx = 0, o = 0, a = ~std::uint64_t{0};
    bool up = true, down = true;
    std::uint64_t prev = k[lo];
    for (std::int64_t j = lo; j < hi; ++j) {
        const std::uint64_t x = k[j];
        up &= prev <= x;
        down &= prev >= x;
        prev = x;
        mn = std::min(mn, x);
        mx = std::max(mx, x);
        o |= x;
        a &= x;
    }
    s.nondecreasing = up;
    s.nonincreasing = down;
    s.lo = mn;
    s.hi = mx;
    s.ones = o;
    s.zeros = a;
    return s;
}

inline KeyScan scan_keys(const std::vector<std::uint64_t>& keys) {
    const auto n = static_cast<std::int64_t>(keys.size());
    const std::uint64_t* k = keys.data();
    const std::size_t C = static_cast<std::size_t>(num_chunks(n));
    std::vector<KeyScan> part(C);
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        part[static_cast<std::size_t>(lo / kChunkRows)] = scan_range(k, lo, hi);
    });
    KeyScan s;
    for (std::size_t c = 0; c < C; ++c) {
        const KeyScan& p = part[c];
        s.nondecreasing &= p.nondecreasing;
        s.nonincreasing &= p.nonincreasing;
        if (c > 0) {  // the chunk boundary
            const std::uint64_t a = k[static_cast<std::int64_t>(c) * kChunkRows - 1],
                                b = k[static_cast<std::int64_t>(c) * kChunkRows];
            s.nondecreasing &= a <= b;
            s.nonincreasing &= a >= b;
        }
        s.lo = std::min(s.lo, p.lo);
        s.hi = std::max(s.hi, p.hi);
        s.ones |= p.ones;
        s.zeros &= p.zeros;
    }
    return s;
}

// keys non-increasing: the stable ascending order is the runs of equal keys
// from last to first, each run in its own order.
inline void reverse_runs(std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows) {
    const auto n = static_cast<std::int64_t>(keys.size());
    std::vector<std::uint64_t> keys2(static_cast<std::size_t>(n));
    std::vector<std::int64_t> rows2(static_cast<std::size_t>(n));
    std::int64_t at = 0;
    for (std::int64_t b = n; b > 0;) {
        std::int64_t a = b - 1;
        while (a > 0 && keys[a - 1] == keys[b - 1]) --a;
        for (std::int64_t j = a; j < b; ++j, ++at) {
            keys2[at] = keys[j];
            rows2[at] = rows[j];
        }
        b = a;
    }
    keys.swap(keys2);
    rows.swap(rows2);
}

// The passes a sort needs: the shift and width (bits) of each digit, low
// first, and the offset subtracted from every key before its digits are read.
struct RadixPlan {
    int shifts[8];
    int widths[8];
    int npass = 0;
    std::uint64_t offset = 0;
};

// Byte digits: either the bytes of (key - min) up to its highest set one or
// the bytes where some keys differ, whichever is fewer. `wide` (a serial
// sort of many keys) also weighs digits of up to 11 bits over the bits of
// (key - min): a pass costs about the same up to 11 bits (1M keys: 1.5 ms
// for 8, 9, 10 or 11), so keys spanning 17-22 bits take 2 passes, not 3. Not
// for a split sort: 2,048 output streams per task overflowed the caches
// (docs/design/frame.md §9.5).
inline RadixPlan plan_passes(const KeyScan& s, int digits, bool wide = false) {
    RadixPlan by_mask, by_range;
    const std::uint64_t differ = s.ones ^ s.zeros;  // bits that are not the same in every key
    for (int d = 0; d < digits; ++d)
        if ((differ >> (8 * d)) & 0xff) {
            by_mask.shifts[by_mask.npass] = 8 * d;
            by_mask.widths[by_mask.npass++] = 8;
        }
    const std::uint64_t span = s.hi - s.lo;
    by_range.offset = s.lo;
    const int bits = span == 0 ? 0 : 64 - __builtin_clzll(span);
    const int max_width = wide ? 11 : 8;
    const int npass = (bits + max_width - 1) / max_width;
    for (int d = 0, at = 0; d < npass; ++d) {
        // Bytes; wide digits: the bits left, spread evenly over the passes left.
        const int w = wide ? (bits - at + (npass - d) - 1) / (npass - d) : 8;
        by_range.shifts[d] = at;
        by_range.widths[d] = w;
        at += w;
    }
    by_range.npass = npass;
    return by_range.npass <= by_mask.npass ? by_range : by_mask;
}

inline void radix_passes_serial(std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows, const RadixPlan& p) {
    const std::size_t n = keys.size();
    const std::uint64_t off = p.offset;
    std::vector<std::int64_t> count(static_cast<std::size_t>(p.npass) << 11, 0);  // 2^11 per pass
    for (std::uint64_t k : keys) {
        const std::uint64_t x = k - off;
        for (int d = 0; d < p.npass; ++d)
            ++count[(static_cast<std::size_t>(d) << 11) + ((x >> p.shifts[d]) & ((1u << p.widths[d]) - 1))];
    }
    std::vector<std::uint64_t> keys2(n);
    std::vector<std::int64_t> rows2(n);
    std::vector<std::int64_t> next_(std::size_t{1} << 11);
    for (int d = 0; d < p.npass; ++d) {
        const int shift = p.shifts[d];
        const std::uint64_t mask = (std::uint64_t{1} << p.widths[d]) - 1;
        const std::int64_t* c = count.data() + (static_cast<std::size_t>(d) << 11);
        std::int64_t* next = next_.data();  // a local pointer: the row stores below cannot alias it
        std::int64_t at = 0;
        for (std::uint64_t b = 0; b <= mask; ++b) {
            next[b] = at;
            at += c[b];
        }
        const std::uint64_t* k = keys.data();
        const std::int64_t* r = rows.data();
        std::uint64_t* k2 = keys2.data();
        std::int64_t* r2 = rows2.data();
        for (std::size_t j = 0; j < n; ++j) {
            const std::int64_t to = next[((k[j] - off) >> shift) & mask]++;
            k2[to] = k[j];
            r2[to] = r[j];
        }
        keys.swap(keys2);
        rows.swap(rows2);
    }
}

// The serial sort on its own (a task's or a short run's sort).
inline void radix_sort_serial(std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows, int digits = 8) {
    if (keys.size() < 2) return;
    const KeyScan s = scan_range(keys.data(), 0, static_cast<std::int64_t>(keys.size()));
    if (s.nondecreasing) return;
    if (s.nonincreasing) {
        reverse_runs(keys, rows);
        return;
    }
    radix_passes_serial(keys, rows, plan_passes(s, digits, keys.size() >= kWideRadixKeys));
}

// The number of byte digits that holds every value up to `max_key`
// (at least one).
inline int digits_for(std::uint64_t max_key) {
    int d = 1;
    while (d < 8 && (max_key >> (8 * d)) != 0) ++d;
    return d;
}

inline void radix_sort(std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows, int digits = 8) {
    const auto n = static_cast<std::int64_t>(keys.size());
    if (n < 2) return;
    // Split from the threshold on: below it the serial sort's wider digits
    // (fewer passes) beat byte passes on the pool (10^5 keys: 0.44 ms against 0.57).
    const bool split = parallel::available() && n >= kParallelThreshold;
    const Unlocked unlocked(split ? 0 : n * 2);  // the GIL rule of the passes (docs/design/frame.md §9.5)
    const KeyScan s = split ? scan_keys(keys) : scan_range(keys.data(), 0, n);
    if (s.nondecreasing) return;
    if (s.nonincreasing) {
        const Unlocked u(n);
        reverse_runs(keys, rows);
        return;
    }
    const RadixPlan p = plan_passes(s, digits, !split && n >= kWideRadixKeys);
    if (!split) {
        radix_passes_serial(keys, rows, p);
        return;
    }
    const std::size_t C = static_cast<std::size_t>(num_chunks(n));
    std::vector<std::int64_t> at(C * 256);
    std::vector<std::uint64_t> keys2(static_cast<std::size_t>(n));
    std::vector<std::int64_t> rows2(static_cast<std::size_t>(n));
    const std::uint64_t off = p.offset;
    for (int d = 0; d < p.npass; ++d) {
        const int shift = p.shifts[d];
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            // Four histograms, summed at the end: runs of one digit would
            // otherwise chain every increment on the one before it.
            std::uint32_t h[4][256] = {};
            const std::uint64_t* k = keys.data();
            std::int64_t j = lo;
            for (; j + 4 <= hi; j += 4)
                for (int u = 0; u < 4; ++u) ++h[u][((k[j + u] - off) >> shift) & 0xff];
            for (; j < hi; ++j) ++h[0][((k[j] - off) >> shift) & 0xff];
            std::int64_t* out = at.data() + lo / kChunkRows * 256;
            for (int b = 0; b < 256; ++b) out[b] = std::int64_t{h[0][b]} + h[1][b] + h[2][b] + h[3][b];
        });
        std::int64_t pos = 0;
        for (std::size_t b = 0; b < 256; ++b)
            for (std::size_t c = 0; c < C; ++c) {
                const std::int64_t t = at[c * 256 + b];
                at[c * 256 + b] = pos;
                pos += t;
            }
        for_chunks(
            n,
            [&](std::int64_t lo, std::int64_t hi) {
                // A local copy of this chunk's places: the int64 row stores below
                // could alias an array on the heap, which would make every
                // increment wait for the stores before it.
                std::int64_t next[256];
                std::copy_n(at.data() + lo / kChunkRows * 256, 256, next);
                const std::uint64_t* k = keys.data();
                const std::int64_t* r = rows.data();
                std::uint64_t* k2 = keys2.data();
                std::int64_t* r2 = rows2.data();
                for (std::int64_t j = lo; j < hi; ++j) {
                    const std::int64_t to = next[((k[j] - off) >> shift) & 0xff]++;
                    k2[to] = k[j];
                    r2[to] = r[j];
                }
            },
            2
        );
        keys.swap(keys2);
        rows.swap(rows2);
    }
}

// The 8 bytes of `s` from byte `depth` on, as string_prefix reads the first
// 8 (zero past the end).
inline std::uint64_t string_key_at(std::string_view s, std::size_t depth) {
    return s.size() <= depth ? 0 : string_prefix(s.substr(depth));
}

// Runs of equal keys at least this long go one radix level deeper (the next
// 8 bytes); shorter ones are ordered by a comparison sort. Either way the
// run ends up in the one stable order, so this only sets the cost.
inline constexpr std::int64_t kStringRadixRun = 128;

// rows[0, len) agree on their first `depth` bytes (zero-padded): orders them
// by the rest, stably. Serial, for a task or a short run: an MSD radix
// sort of 8 more bytes while the run is long and some string still has
// bytes left, a comparison sort otherwise.
inline void order_string_run(const Column& c, std::int64_t* rows, std::int64_t len, std::size_t depth) {
    if (len < 2) return;
    bool more = false;  // a string longer than `depth`: a deeper level can tell rows apart
    if (len >= kStringRadixRun)
        for (std::int64_t j = 0; j < len && !more; ++j) more = c.string_at(rows[j]).size() > depth;
    if (!more) {
        std::stable_sort(rows, rows + len, [&](std::int64_t x, std::int64_t y) {
            return c.string_at(x) < c.string_at(y);
        });
        return;
    }
    std::vector<std::uint64_t> keys(static_cast<std::size_t>(len));
    std::vector<std::int64_t> sub(rows, rows + len);
    for (std::int64_t j = 0; j < len; ++j) keys[j] = string_key_at(c.string_at(sub[j]), depth);
    radix_sort_serial(keys, sub);
    std::copy(sub.begin(), sub.end(), rows);
    for (std::int64_t a = 0; a < len;) {
        std::int64_t b = a + 1;
        while (b < len && keys[b] == keys[a]) ++b;
        order_string_run(c, rows + a, b - a, depth + 8);
        a = b;
    }
}

// `rows`, sorted by `keys` (each row's 8 bytes from depth - 8): orders every
// run of equal keys by the bytes from `depth` on. A run is ordered by the
// chunk where it starts (runs are independent of each other); a run long
// enough to split across threads is left to the caller's thread, which
// sorts it one level deeper with a parallel radix sort and recurses.
inline void order_string_runs(
    const Column& c, const std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows, std::size_t depth
) {
    const auto n = static_cast<std::int64_t>(rows.size());
    std::vector<std::vector<std::pair<std::int64_t, std::int64_t>>> long_runs(static_cast<std::size_t>(num_chunks(n)));
    for_chunks(
        n,
        [&](std::int64_t lo, std::int64_t hi) {
            std::int64_t a = lo;
            while (a > 0 && a < n && keys[a] == keys[a - 1]) ++a;  // a run that started in an earlier chunk is not ours
            while (a < hi) {
                std::int64_t b = a + 1;
                while (b < n && keys[b] == keys[a]) ++b;
                if (b - a >= kParallelThreshold) long_runs[static_cast<std::size_t>(lo / kChunkRows)].push_back({a, b});
                else order_string_run(c, rows.data() + a, b - a, depth);
                a = b;
            }
        },
        8
    );
    for (const auto& chunk : long_runs)
        for (auto [a, b] : chunk) {
            const std::int64_t len = b - a;
            std::vector<std::int64_t> sub(rows.begin() + a, rows.begin() + b);
            std::vector<std::uint64_t> sub_keys(static_cast<std::size_t>(len));
            std::atomic<bool> more{false};
            for_chunks(
                len,
                [&](std::int64_t lo, std::int64_t hi) {
                    bool m = false;
                    for (std::int64_t j = lo; j < hi; ++j) {
                        const std::string_view s = c.string_at(sub[j]);
                        sub_keys[j] = string_key_at(s, depth);
                        m |= s.size() > depth;
                    }
                    if (m) more = true;
                },
                4
            );
            if (more) {
                radix_sort(sub_keys, sub);
                order_string_runs(c, sub_keys, sub, depth + 8);
            } else {  // no bytes left: the rows differ at most in trailing NUL bytes
                const Unlocked unlocked(len);
                std::stable_sort(sub.begin(), sub.end(), [&](std::int64_t x, std::int64_t y) {
                    return c.string_at(x) < c.string_at(y);
                });
            }
            std::copy(sub.begin(), sub.end(), rows.begin() + a);
        }
}

// `rows` (with values, no NA) ordered by the column's values, ascending.
// Numbers and bools: radix sort on radix_key (bools: one digit). Strings:
// radix sort on the 8-byte prefix, then each run of equal prefixes by the
// next 8 bytes, and so on (an MSD radix sort), with a comparison sort only
// for short runs (comparing every pair of strings cost half of a string
// groupby, docs/design/frame.md §9.4).
// Rows with equal values keep their order.
inline void order_rows(const Column& c, std::vector<std::int64_t>& rows) {
    const auto n = static_cast<std::int64_t>(rows.size());
    std::vector<std::uint64_t> keys(rows.size());
    if (c.dtype != DType::String) {
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t j = lo; j < hi; ++j) keys[j] = radix_key(c, rows[j]);
        });
        radix_sort(keys, rows, c.dtype == DType::Bool ? 1 : 8);
        return;
    }
    for_chunks(
        n,
        [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t j = lo; j < hi; ++j) keys[j] = string_prefix(c.string_at(rows[j]));
        },
        4
    );
    radix_sort(keys, rows);
    order_string_runs(c, keys, rows, 8);
}

// Positions 0 .. n-1 stably sorted by key[i] in [0, nkeys): perm lists them
// key by key, and key k's positions are perm[start[k], start[k + 1]).
//
// A counting sort in two levels, so it runs in parallel without a counter
// per (chunk, key) -- that would be 61 x 630k counters for 1M rows and 630k
// groups (docs/design/frame.md §9.6):
//  1. Keys fall into B buckets of `width` consecutive keys. Each chunk of
//     positions counts its positions per bucket; they are scattered to their
//     buckets, bucket by bucket and within a bucket chunk by chunk, so every
//     bucket lists its positions in order.
//  2. Each bucket then counting-sorts its own positions by key, on its own.
// The result is the one a serial counting sort gives, so with one thread
// it is just that.
inline void counting_sort(
    const std::int64_t* key,
    std::int64_t n,
    std::int64_t nkeys,
    std::vector<std::int64_t>& start,
    std::vector<std::int64_t>& perm
) {
    start.assign(static_cast<std::size_t>(nkeys + 1), 0);
    perm.resize(static_cast<std::size_t>(n));
    if (n == 0) return;
    if (n < kParallelThreshold || !parallel::available()) {
        // One thread: the plain counting sort. The order is the same; the
        // two levels only pay off in parallel (docs/design/frame.md §9.6,
        // §11.5).
        Unlocked unlocked(n);
        for (std::int64_t i = 0; i < n; ++i) ++start[static_cast<std::size_t>(key[i]) + 1];
        for (std::int64_t k = 0; k < nkeys; ++k)
            start[static_cast<std::size_t>(k) + 1] += start[static_cast<std::size_t>(k)];
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::int64_t i = 0; i < n; ++i)
            perm[static_cast<std::size_t>(fill[static_cast<std::size_t>(key[i])]++)] = i;
        return;
    }
    // `width` is a power of two, so a key's bucket is a shift (a division
    // per position cost more than the whole serial counting sort).
    const std::int64_t buckets = 256;
    int wbits = 0;
    while ((std::int64_t{1} << wbits) * buckets < nkeys) ++wbits;
    const std::int64_t width = std::int64_t{1} << wbits;
    const std::int64_t B = std::max<std::int64_t>(1, (nkeys + width - 1) / width);
    const std::size_t C = static_cast<std::size_t>(num_chunks(n));
    const auto Bs = static_cast<std::size_t>(B);
    std::vector<std::int64_t> at(C * Bs, 0);
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t* cnt = at.data() + lo / kChunkRows * B;
        for (std::int64_t i = lo; i < hi; ++i) ++cnt[key[i] >> wbits];
    });
    std::vector<std::int64_t> bstart(Bs + 1, 0);
    std::int64_t pos = 0;
    for (std::size_t b = 0; b < Bs; ++b) {
        bstart[b] = pos;
        for (std::size_t k = 0; k < C; ++k) {
            const std::int64_t t = at[k * Bs + b];
            at[k * Bs + b] = pos;
            pos += t;
        }
    }
    bstart[Bs] = pos;
    std::vector<std::int64_t> by_bucket(static_cast<std::size_t>(n));
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t* next = at.data() + lo / kChunkRows * B;
        for (std::int64_t i = lo; i < hi; ++i) by_bucket[static_cast<std::size_t>(next[key[i] >> wbits]++)] = i;
    });
    parallel::parallel_for(B, [&](std::int64_t b) {
        const std::int64_t k0 = std::min(nkeys, b * width), k1 = std::min(nkeys, (b + 1) * width);
        const std::int64_t r0 = bstart[static_cast<std::size_t>(b)], r1 = bstart[static_cast<std::size_t>(b) + 1];
        // Local counts: start[k0] belongs to the bucket before (it ends
        // there), so a bucket writes start only at (k0, k1].
        std::vector<std::int64_t> fill(static_cast<std::size_t>(k1 - k0), 0);
        for (std::int64_t j = r0; j < r1; ++j)
            ++fill[static_cast<std::size_t>(key[by_bucket[static_cast<std::size_t>(j)]] - k0)];
        std::int64_t at_pos = r0;
        for (std::int64_t k = k0; k < k1; ++k) {
            const std::int64_t c = fill[static_cast<std::size_t>(k - k0)];
            fill[static_cast<std::size_t>(k - k0)] = at_pos;
            at_pos += c;
            start[static_cast<std::size_t>(k) + 1] = at_pos;
        }
        for (std::int64_t j = r0; j < r1; ++j) {
            const std::int64_t i = by_bucket[static_cast<std::size_t>(j)];
            perm[static_cast<std::size_t>(fill[static_cast<std::size_t>(key[i] - k0)]++)] =
                i;  // stable: order within a key
        }
    });
}

}  // namespace sd::frame
