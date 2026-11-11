// Stable LSD radix sort on 64-bit keys (M7) and ordering the values of a
// column (M10), shared by sort_values and groupby's group order.
#pragma once

#include "core/column.hpp"
#include "ops/hash.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace sd {

// An unsigned 64-bit key whose order is the value order (API.md §17).
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

// Stable LSD radix sort of `rows` by 64-bit keys: eight passes of one byte,
// least significant first. Every pass runs, whatever the data (no skipping
// of constant digits).
//
// In parallel (M11): each pass counts the digit per chunk of positions,
// gives every (digit, chunk) pair its place -- digit by digit, and within a
// digit chunk by chunk -- and then each chunk scatters its elements to its
// own places. An element's place is exactly the one the serial pass gives
// it, so the order is the same for any thread count (DESIGN.md §2.6).
inline void radix_sort(std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows) {
    const auto n = static_cast<std::int64_t>(keys.size());
    if (n == 0) return;
    const std::size_t C = static_cast<std::size_t>(parallel::num_chunks(n));
    std::vector<std::int64_t> at(C * 256);
    std::vector<std::uint64_t> keys2(static_cast<std::size_t>(n));
    std::vector<std::int64_t> rows2(static_cast<std::size_t>(n));
    for (int d = 0; d < 8; ++d) {
        const int shift = 8 * d;
        parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            // Four histograms, summed at the end: runs of one digit would
            // otherwise chain every increment on the one before it.
            std::uint32_t h[4][256] = {};
            const std::uint64_t* k = keys.data();
            std::int64_t j = lo;
            for (; j + 4 <= hi; j += 4)
                for (int u = 0; u < 4; ++u) ++h[u][(k[j + u] >> shift) & 0xff];
            for (; j < hi; ++j) ++h[0][(k[j] >> shift) & 0xff];
            std::int64_t* out = at.data() + lo / parallel::kChunkRows * 256;
            for (int b = 0; b < 256; ++b) out[b] = std::int64_t{h[0][b]} + h[1][b] + h[2][b] + h[3][b];
        });
        std::int64_t pos = 0;
        for (std::size_t b = 0; b < 256; ++b)
            for (std::size_t c = 0; c < C; ++c) {
                const std::int64_t t = at[c * 256 + b];
                at[c * 256 + b] = pos;
                pos += t;
            }
        parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            // A local copy of this chunk's places: the int64 row stores below
            // could alias an array on the heap, which would make every
            // increment wait for the stores before it.
            std::int64_t next[256];
            std::copy_n(at.data() + lo / parallel::kChunkRows * 256, 256, next);
            const std::uint64_t* k = keys.data();
            const std::int64_t* r = rows.data();
            std::uint64_t* k2 = keys2.data();
            std::int64_t* r2 = rows2.data();
            for (std::int64_t j = lo; j < hi; ++j) {
                const std::int64_t to = next[(k[j] >> shift) & 0xff]++;
                k2[to] = k[j];
                r2[to] = r[j];
            }
        }, 2);
        keys.swap(keys2);
        rows.swap(rows2);
    }
}

// `rows` (with values, no NA) ordered by the column's values, ascending.
// Numbers and bools: radix sort on radix_key. Strings: radix sort on the
// 8-byte prefix, then a comparison sort only inside runs of equal prefixes
// (M10: comparing every pair of strings cost half of a string groupby).
// Rows with equal values keep their order.
inline void order_rows(const Column& c, std::vector<std::int64_t>& rows) {
    const auto n = static_cast<std::int64_t>(rows.size());
    std::vector<std::uint64_t> keys(rows.size());
    if (c.dtype != DType::String) {
        parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t j = lo; j < hi; ++j) keys[j] = radix_key(c, rows[j]);
        });
        radix_sort(keys, rows);
        return;
    }
    parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        for (std::int64_t j = lo; j < hi; ++j) keys[j] = string_prefix(c.string_at(rows[j]));
    }, 4);
    radix_sort(keys, rows);
    // Runs of equal prefixes: the rest of the bytes decides. A run is sorted
    // by the chunk where it starts (runs are independent of each other).
    parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t a = lo;
        while (a > 0 && a < n && keys[a] == keys[a - 1]) ++a;  // a run that started in an earlier chunk is not ours
        while (a < hi) {
            std::int64_t b = a + 1;
            while (b < n && keys[b] == keys[a]) ++b;
            if (b - a > 1)
                std::stable_sort(rows.begin() + a, rows.begin() + b,
                                 [&](std::int64_t x, std::int64_t y) { return c.string_at(x) < c.string_at(y); });
            a = b;
        }
    }, 8);
}

// Positions 0 .. n-1 stably sorted by key[i] in [0, nkeys): perm lists them
// key by key, and key k's positions are perm[start[k], start[k + 1]).
//
// A counting sort in two levels, so it runs in parallel without a counter
// per (chunk, key) -- that would be 61 x 630k counters for 1M rows and 630k
// groups (M11):
//  1. Keys fall into B buckets of `width` consecutive keys. Each chunk of
//     positions counts its positions per bucket; they are scattered to their
//     buckets, bucket by bucket and within a bucket chunk by chunk, so every
//     bucket lists its positions in order.
//  2. Each bucket then counting-sorts its own positions by key, on its own.
// The result is the one a serial counting sort gives, so with one thread
// it is just that.
inline void counting_sort(const std::int64_t* key, std::int64_t n, std::int64_t nkeys, std::vector<std::int64_t>& start,
                   std::vector<std::int64_t>& perm) {
    start.assign(static_cast<std::size_t>(nkeys + 1), 0);
    perm.resize(static_cast<std::size_t>(n));
    if (n == 0) return;
    if (n < parallel::kThreshold || !parallel::available()) {
        // One thread: the plain counting sort. The order is the same; the
        // two levels only pay off in parallel (user decision 2026-11-11,
        // DESIGN.md §2.6).
        parallel::Unlocked unlocked(n);
        for (std::int64_t i = 0; i < n; ++i) ++start[static_cast<std::size_t>(key[i]) + 1];
        for (std::int64_t k = 0; k < nkeys; ++k) start[static_cast<std::size_t>(k) + 1] += start[static_cast<std::size_t>(k)];
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::int64_t i = 0; i < n; ++i) perm[static_cast<std::size_t>(fill[static_cast<std::size_t>(key[i])]++)] = i;
        return;
    }
    // `width` is a power of two, so a key's bucket is a shift (a division
    // per position cost more than the whole serial counting sort).
    const std::int64_t buckets = 256;
    int wbits = 0;
    while ((std::int64_t{1} << wbits) * buckets < nkeys) ++wbits;
    const std::int64_t width = std::int64_t{1} << wbits;
    const std::int64_t B = std::max<std::int64_t>(1, (nkeys + width - 1) / width);
    const std::size_t C = static_cast<std::size_t>(parallel::num_chunks(n));
    const auto Bs = static_cast<std::size_t>(B);
    std::vector<std::int64_t> at(C * Bs, 0);
    parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t* cnt = at.data() + lo / parallel::kChunkRows * B;
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
    parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t* next = at.data() + lo / parallel::kChunkRows * B;
        for (std::int64_t i = lo; i < hi; ++i) by_bucket[static_cast<std::size_t>(next[key[i] >> wbits]++)] = i;
    });
    parallel::parallel_for(B, [&](std::int64_t b) {
        const std::int64_t k0 = std::min(nkeys, b * width), k1 = std::min(nkeys, (b + 1) * width);
        const std::int64_t r0 = bstart[static_cast<std::size_t>(b)], r1 = bstart[static_cast<std::size_t>(b) + 1];
        // Local counts: start[k0] belongs to the bucket before (it ends
        // there), so a bucket writes start only at (k0, k1].
        std::vector<std::int64_t> fill(static_cast<std::size_t>(k1 - k0), 0);
        for (std::int64_t j = r0; j < r1; ++j) ++fill[static_cast<std::size_t>(key[by_bucket[static_cast<std::size_t>(j)]] - k0)];
        std::int64_t at_pos = r0;
        for (std::int64_t k = k0; k < k1; ++k) {
            const std::int64_t c = fill[static_cast<std::size_t>(k - k0)];
            fill[static_cast<std::size_t>(k - k0)] = at_pos;
            at_pos += c;
            start[static_cast<std::size_t>(k) + 1] = at_pos;
        }
        for (std::int64_t j = r0; j < r1; ++j) {
            const std::int64_t i = by_bucket[static_cast<std::size_t>(j)];
            perm[static_cast<std::size_t>(fill[static_cast<std::size_t>(key[i] - k0)]++)] = i;  // stable: order within a key
        }
    });
}

}  // namespace sd
