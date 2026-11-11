// Hashing: the key table, factorize (values -> dense int codes) and the
// label lookup engine of Index (DESIGN.md §2.5). factorize is the base of
// unique, duplicated and value_counts (M3), and later groupby and merge.
#pragma once

#include "core/column.hpp"

#include <cstring>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

namespace sd {

// murmur3's 64-bit finalizer: every input bit affects every output bit, so
// the low bits that pick a slot are well mixed even for keys 0, 1, 2, ...
struct Fmix64 {
    std::uint64_t operator()(std::uint64_t k) const {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }
};

// FNV-1a over the bytes, then fmix64 (FNV's low bits are weak on their own).
// A textbook hash, one byte per step; M10 measures a faster one.
struct HashBytes {
    std::uint64_t operator()(std::string_view s) const {
        std::uint64_t h = 0xcbf29ce484222325ULL;
        for (unsigned char c : s) {
            h ^= c;
            h *= 0x100000001b3ULL;
        }
        return Fmix64{}(h);
    }
};

// The key of a float: its bits, with every NaN made one NaN and -0.0 made
// 0.0, so equal keys have equal bits (API.md §1).
inline std::uint64_t float_key(double d) {
    if (d != d) return 0x7ff8000000000000ULL;
    if (d == 0.0) return 0;
    std::uint64_t bits;
    std::memcpy(&bits, &d, sizeof bits);
    return bits;
}

// Open addressing with linear probing (DESIGN.md §2.5, R3): key -> code.
// A slot holds a code (-1 when empty) next to one word: the key itself for
// 8-byte keys, so a probe never leaves the table, or else the key's hash,
// compared before the key. The table doubles when half full; slots keep
// their hashes, so growing never hashes a string again.
template <class Key, class Hash>
class KeyTable {
public:
    static constexpr bool kInline = std::is_same_v<Key, std::uint64_t>;

    // The code of `k`, adding it as `new_code` when absent (then .second is true).
    std::pair<std::int64_t, bool> insert(Key k, std::int64_t new_code) {
        const std::uint64_t h = Hash{}(k);
        std::uint64_t s = h & mask_;
        while (true) {
            Slot& e = slots_[s];
            if (e.code < 0) {
                e = Slot{word_of(k, h), new_code};
                if constexpr (!kInline) {
                    if (static_cast<std::int64_t>(keys_.size()) <= new_code) keys_.resize(new_code + 1);
                    keys_[new_code] = k;
                }
                if (++size_ * 2 > static_cast<std::int64_t>(slots_.size())) grow();
                return {new_code, true};
            }
            if (matches(e, k, h)) return {e.code, false};
            s = (s + 1) & mask_;
        }
    }

    // The code of `k`, or -1.
    std::int64_t find(Key k) const {
        const std::uint64_t h = Hash{}(k);
        std::uint64_t s = h & mask_;
        while (true) {
            const Slot& e = slots_[s];
            if (e.code < 0) return -1;
            if (matches(e, k, h)) return e.code;
            s = (s + 1) & mask_;
        }
    }

private:
    struct Slot {
        std::uint64_t word;
        std::int64_t code;
    };

    static std::uint64_t word_of(Key k, std::uint64_t h) {
        if constexpr (kInline) return k;
        else return h;
    }

    bool matches(const Slot& e, Key k, std::uint64_t h) const {
        if constexpr (kInline) return e.word == k;
        else return e.word == h && keys_[e.code] == k;
    }

    void grow() {
        std::vector<Slot> old(slots_.size() * 2, Slot{0, -1});
        old.swap(slots_);
        mask_ = slots_.size() - 1;
        for (const Slot& e : old) {
            if (e.code < 0) continue;
            std::uint64_t h;
            if constexpr (kInline) h = Hash{}(e.word);
            else h = e.word;
            std::uint64_t s = h & mask_;
            while (slots_[s].code >= 0) s = (s + 1) & mask_;
            slots_[s] = e;
        }
    }

    std::vector<Slot> slots_ = std::vector<Slot>(16, Slot{0, -1});
    std::uint64_t mask_ = 15;
    std::int64_t size_ = 0;
    std::vector<Key> keys_;  // string keys by code, for comparing
};

struct Factorized {
    Column codes;  // int64, one per row: 0, 1, ... in order of first appearance
    Column first;  // int64, per code: the row where it first appears
    std::int64_t ngroups = 0;
};

// Codes for the rows of one column. Keys follow API.md §1 and §13: NaN
// equals NaN, 0.0 equals -0.0, and all NA rows share one code.
Factorized factorize(const Column& col);

// Codes for rows of several columns taken together (a row's key is the
// tuple of its values). Each column is factorized, the codes are combined
// into one int64 (mixed radix), and that is factorized again so the codes
// are dense and in order of first appearance.
Factorized factorize_rows(const std::vector<Column>& cols);

// Per code: how many rows have it, and the last row that has it.
std::pair<Column, Column> group_counts(const Column& codes, std::int64_t ngroups);

// Positions that order `counts` by value (descending unless `ascending`),
// keeping the original order among equal values (a stable sort).
Column stable_order(const Column& counts, bool ascending);

// Whether any row of `col` has the key of the one-row `value` (API.md §13,
// §14: NaN matches NaN, NA matches NA). The dtypes must be the same.
bool contains_key(const Column& col, const Column& value);

// The lookup engine of an Index (ARCHITECTURE.md §4): the labels (int64 or
// string, no NA) in a key table, built once and kept by the Index.
class LabelTable {
public:
    explicit LabelTable(const Column& labels);

    bool is_unique() const { return ngroups_ == labels_.length; }

    // Per target row: the code of its label in this table, or -1 when absent
    // (also for NA targets). The target dtype must match.
    Column lookup(const Column& targets) const;

    // Per target row: the first position holding its label, or -1.
    Column first_positions(const Column& targets) const;

    // For each target row in order, every position holding its label (in
    // position order), concatenated. Absent labels contribute nothing.
    Column all_positions(const Column& targets) const;

private:
    Column labels_;  // kept alive: string keys point into its bytes
    std::vector<std::int64_t> codes_;  // per position
    std::vector<std::int64_t> first_;  // per code
    std::int64_t ngroups_ = 0;
    std::unique_ptr<KeyTable<std::uint64_t, Fmix64>> ints_;
    std::unique_ptr<KeyTable<std::string_view, HashBytes>> strings_;
    // Positions grouped by code (built on first use by all_positions).
    mutable std::vector<std::int64_t> group_start_, group_rows_;
};

}  // namespace sd
