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

// Stable LSD radix sort of `rows` by 64-bit keys: `digits` passes of one
// byte, least significant first (eight by default). The caller sets fewer
// digits only from what the keys are, never from their values: bools (one
// byte) and ranks below a count of distinct values (the bytes of that
// count). A higher byte that is the same in every key -- 0, or 0xff for a
// reversed key -- orders nothing, so leaving it out gives the same order.
// Every pass of the `digits` runs, whatever the data (no skipping of
// constant digits).
//
// In parallel (docs/design/frame.md §9.5): each pass counts the digit per chunk of positions,
// gives every (digit, chunk) pair its place -- digit by digit, and within a
// digit chunk by chunk -- and then each chunk scatters its elements to its
// own places. An element's place is exactly the one the serial pass gives
// it, so the order is the same for any thread count (docs/design/frame.md §11.5).
//
// Serial: when no pass would be split -- one thread, or too few keys for
// the scatter to reach the threshold -- a serial sort runs instead: all
// eight histograms in one read of the keys, no per-chunk tables. The same
// stable sort, so the same order; at 1k-10k keys it is 25-29% faster than
// eight counting passes (docs/design/frame.md §9.5).
inline void radix_sort_serial(std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows, int digits = 8) {
    const std::size_t n = keys.size();
    std::vector<std::array<std::int64_t, 256>> count(static_cast<std::size_t>(digits));
    for (auto& h : count) h.fill(0);
    for (std::uint64_t k : keys)
        for (int d = 0; d < digits; ++d) ++count[d][(k >> (8 * d)) & 0xff];
    std::vector<std::uint64_t> keys2(n);
    std::vector<std::int64_t> rows2(n);
    for (int d = 0; d < digits; ++d) {
        std::int64_t next[256];  // local: the row stores below cannot alias it
        next[0] = 0;
        for (int b = 1; b < 256; ++b) next[b] = next[b - 1] + count[d][b - 1];
        const int shift = 8 * d;
        const std::uint64_t* k = keys.data();
        const std::int64_t* r = rows.data();
        std::uint64_t* k2 = keys2.data();
        std::int64_t* r2 = rows2.data();
        for (std::size_t j = 0; j < n; ++j) {
            const std::int64_t to = next[(k[j] >> shift) & 0xff]++;
            k2[to] = k[j];
            r2[to] = r[j];
        }
        keys.swap(keys2);
        rows.swap(rows2);
    }
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
    if (n == 0) return;
    if (!parallel::available() || n * 2 < kParallelThreshold) {  // 2: the scatter's cost below
        Unlocked unlocked(n * 2);  // the GIL rule of the passes it replaces (docs/design/frame.md §9.5)
        radix_sort_serial(keys, rows, digits);
        return;
    }
    const std::size_t C = static_cast<std::size_t>(num_chunks(n));
    std::vector<std::int64_t> at(C * 256);
    std::vector<std::uint64_t> keys2(static_cast<std::size_t>(n));
    std::vector<std::int64_t> rows2(static_cast<std::size_t>(n));
    for (int d = 0; d < digits; ++d) {
        const int shift = 8 * d;
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            // Four histograms, summed at the end: runs of one digit would
            // otherwise chain every increment on the one before it.
            std::uint32_t h[4][256] = {};
            const std::uint64_t* k = keys.data();
            std::int64_t j = lo;
            for (; j + 4 <= hi; j += 4)
                for (int u = 0; u < 4; ++u) ++h[u][(k[j + u] >> shift) & 0xff];
            for (; j < hi; ++j) ++h[0][(k[j] >> shift) & 0xff];
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
                    const std::int64_t to = next[(k[j] >> shift) & 0xff]++;
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
