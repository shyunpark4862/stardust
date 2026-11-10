#include "ops/hash.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <string_view>
#include <type_traits>

namespace nb = nanobind;

namespace sd {

namespace {

// murmur3's 64-bit finalizer: every input bit affects every output bit, so
// the low bits that pick a slot are well mixed even for keys 0, 1, 2, ...
std::uint64_t fmix64(std::uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

// FNV-1a over the bytes, then fmix64 (FNV's low bits are weak on their own).
// A textbook hash, one byte per step; M10 measures a faster one.
std::uint64_t hash_bytes(std::string_view s) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return fmix64(h);
}

// The key of a float: its bits, with every NaN made one NaN and -0.0 made
// 0.0, so equal keys have equal bits (API.md §1).
std::uint64_t float_key(double d) {
    if (d != d) return 0x7ff8000000000000ULL;
    if (d == 0.0) return 0;
    std::uint64_t bits;
    std::memcpy(&bits, &d, sizeof bits);
    return bits;
}

// Open addressing with linear probing (DESIGN.md §2.5, R3). A slot holds a
// code (-1 when empty) next to the key's hash, so a probe reads one slot and
// compares hashes before keys; for 8-byte keys the slot holds the key itself
// and the probe never leaves the table. The table doubles when half full;
// slots keep their hashes, so growing never hashes a key again.
struct Slot {
    std::uint64_t word;  // the key itself for 8-byte keys, else the key's hash
    std::int64_t code;
};

template <bool HasNA, class Key, class KeyOf, class Hash>
Factorized factorize_with(const Column& col, KeyOf key_of, Hash hash) {
    constexpr bool kInline = std::is_same_v<Key, std::uint64_t>;
    using S = Slot;
    const std::int64_t n = col.length;
    Factorized out;
    out.codes = make_fixed(DType::Int64, n);
    auto* codes = reinterpret_cast<std::int64_t*>(out.codes.values->data());
    std::vector<std::int64_t> first;
    std::vector<Key> keys;  // strings only: the key of each code, for comparing
    std::vector<S> slots(16, S{0, -1});
    std::uint64_t mask = slots.size() - 1;
    std::int64_t na_code = -1, in_table = 0;

    auto grow = [&] {
        std::vector<S> old(slots.size() * 2, S{0, -1});
        old.swap(slots);
        mask = slots.size() - 1;
        for (const S& e : old) {
            if (e.code < 0) continue;
            std::uint64_t h;
            if constexpr (kInline) h = hash(e.word);
            else h = e.word;
            std::uint64_t s = h & mask;
            while (slots[s].code >= 0) s = (s + 1) & mask;
            slots[s] = e;
        }
    };

    for (std::int64_t i = 0; i < n; ++i) {
        if constexpr (HasNA) {
            if (!col.is_valid(i)) {  // all NA rows are one key
                if (na_code < 0) {
                    na_code = static_cast<std::int64_t>(first.size());
                    first.push_back(i);
                    if constexpr (!kInline) keys.push_back(Key{});
                }
                codes[i] = na_code;
                continue;
            }
        }
        const Key k = key_of(i);
        const std::uint64_t h = hash(k);
        std::uint64_t s = h & mask;
        while (true) {
            S& e = slots[s];
            if (e.code < 0) {  // a new key
                const auto code = static_cast<std::int64_t>(first.size());
                if constexpr (kInline) e = S{k, code};
                else e = S{h, code};
                first.push_back(i);
                if constexpr (!kInline) keys.push_back(k);
                codes[i] = code;
                if (++in_table * 2 > static_cast<std::int64_t>(slots.size())) grow();
                break;
            }
            bool hit;
            if constexpr (kInline) hit = e.word == k;
            else hit = e.word == h && keys[e.code] == k;
            if (hit) {
                codes[i] = e.code;
                break;
            }
            s = (s + 1) & mask;
        }
    }
    out.ngroups = static_cast<std::int64_t>(first.size());
    out.first = make_fixed(DType::Int64, out.ngroups);
    std::copy(first.begin(), first.end(), reinterpret_cast<std::int64_t*>(out.first.values->data()));
    return out;
}

// Picks the loop without NA checks when the column has no NA (a property
// of the column, DESIGN.md §2.1).
template <class Key, class KeyOf, class Hash>
Factorized factorize_any(const Column& col, KeyOf key_of, Hash hash) {
    return col.validity ? factorize_with<true, Key>(col, key_of, hash) : factorize_with<false, Key>(col, key_of, hash);
}

}  // namespace

Factorized factorize(const Column& col) {
    switch (col.dtype) {
    case DType::Bool: {
        const auto* v = col.data<std::uint8_t>();
        return factorize_any<std::uint64_t>(col, [v](std::int64_t i) -> std::uint64_t { return v[i]; }, fmix64);
    }
    case DType::Int64: {
        const auto* v = col.data<std::int64_t>();
        return factorize_any<std::uint64_t>(col, [v](std::int64_t i) { return static_cast<std::uint64_t>(v[i]); }, fmix64);
    }
    case DType::Float64: {
        const auto* v = col.data<double>();
        return factorize_any<std::uint64_t>(col, [v](std::int64_t i) { return float_key(v[i]); }, fmix64);
    }
    case DType::String:
        return factorize_any<std::string_view>(col, [&col](std::int64_t i) { return col.string_at(i); }, hash_bytes);
    }
    throw nb::type_error("unknown dtype");
}

Factorized factorize_rows(const std::vector<Column>& cols) {
    if (cols.empty()) throw nb::value_error("factorize_rows needs at least one column");
    if (cols.size() == 1) return factorize(cols[0]);
    const std::int64_t n = cols[0].length;
    for (const Column& c : cols)
        if (c.length != n) throw nb::value_error("columns must have the same length");

    // acc holds the tuple of codes so far as one number below acc_n.
    Factorized acc = factorize(cols[0]);
    std::int64_t acc_n = acc.ngroups;
    for (std::size_t j = 1; j < cols.size(); ++j) {
        Factorized next = factorize(cols[j]);
        if (next.ngroups != 0 && acc_n > (std::int64_t{1} << 62) / next.ngroups) {
            // The product would overflow: compact acc to dense codes first.
            acc = factorize(acc.codes);
            acc_n = acc.ngroups;
        }
        auto* a = reinterpret_cast<std::int64_t*>(acc.codes.values->data());
        const auto* b = next.codes.data<std::int64_t>();
        for (std::int64_t i = 0; i < n; ++i) a[i] = a[i] * next.ngroups + b[i];
        acc_n *= next.ngroups;
    }
    return factorize(acc.codes);  // dense, in order of first appearance
}

std::pair<Column, Column> group_counts(const Column& codes, std::int64_t ngroups) {
    Column counts = make_fixed(DType::Int64, ngroups);
    Column last = make_fixed(DType::Int64, ngroups);
    auto* cnt = reinterpret_cast<std::int64_t*>(counts.values->data());
    auto* lst = reinterpret_cast<std::int64_t*>(last.values->data());
    std::fill(cnt, cnt + ngroups, 0);
    const auto* c = codes.data<std::int64_t>();
    for (std::int64_t i = 0; i < codes.length; ++i) {
        ++cnt[c[i]];
        lst[c[i]] = i;
    }
    return {counts, last};
}

Column stable_order(const Column& counts, bool ascending) {
    const std::int64_t n = counts.length;
    Column out = make_fixed(DType::Int64, n);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    std::iota(o, o + n, std::int64_t{0});
    const auto* v = counts.data<std::int64_t>();
    // A comparison sort for now; M7 brings radix sort.
    if (ascending) std::stable_sort(o, o + n, [v](std::int64_t x, std::int64_t y) { return v[x] < v[y]; });
    else std::stable_sort(o, o + n, [v](std::int64_t x, std::int64_t y) { return v[x] > v[y]; });
    return out;
}

}  // namespace sd
