// Hashing: the key table, factorize (values -> dense int codes) and the
// label lookup engine of Index (DESIGN.md §2.5). factorize is the base of
// unique, duplicated and value_counts (M3), and later groupby and merge.
#pragma once

#include "core/column.hpp"

#include <arm_neon.h>

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

// Strings (M10): eight bytes per step. Each 8-byte word is folded into the
// state with a 64x64 -> 128-bit multiply whose two halves are XORed (the
// "mum" mix of the wyhash family): one multiply spreads every input bit over
// the whole word. The last partial word is read byte-safely; the length is
// mixed in so "a" and "a\0" differ; fmix64 finishes, so the low bits (the
// Swiss tag) and the group bits are well mixed. M3 used FNV-1a, one byte and
// one multiply per step.
struct HashBytes {
    static std::uint64_t mum(std::uint64_t a, std::uint64_t b) {
        const unsigned __int128 r = static_cast<unsigned __int128>(a) * b;
        return static_cast<std::uint64_t>(r) ^ static_cast<std::uint64_t>(r >> 64);
    }
    std::uint64_t operator()(std::string_view s) const {
        constexpr std::uint64_t k0 = 0xa0761d6478bd642fULL, k1 = 0xe7037ed1a0b428dbULL;
        const char* p = s.data();
        std::size_t n = s.size();
        std::uint64_t h = k0 ^ (static_cast<std::uint64_t>(n) * k1);
        for (; n >= 8; p += 8, n -= 8) {
            std::uint64_t w;
            std::memcpy(&w, p, 8);
            h = mum(w ^ k1, h ^ k0);
        }
        if (n > 0) {
            std::uint64_t w = 0;
            std::memcpy(&w, p, n);
            h = mum(w ^ k1, h ^ k0);
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
class LinearTable {
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

// A Swiss table (M10): the open-addressing table of Abseil and Rust's
// hashbrown. Slots come in groups of 16 with one control byte each: the
// low 7 bits of the hash (the tag) or kEmpty. A lookup loads a group's 16
// control bytes and compares all of them with the tag in one NEON
// instruction; only slots whose tag matches are checked for the key. So
// "is it in this slot or the next?" -- the branch that linear probing
// mispredicts (M3) -- becomes bit operations on a mask, and a miss usually
// ends at the first group, which has an empty byte. Groups are probed
// linearly; the table doubles when 7/8 full. Same interface as LinearTable.
template <class Key, class Hash>
class SwissTable {
public:
    static constexpr bool kInline = std::is_same_v<Key, std::uint64_t>;

    std::pair<std::int64_t, bool> insert(Key k, std::int64_t new_code) {
        const std::uint64_t h = Hash{}(k);
        const std::uint8_t tag = static_cast<std::uint8_t>(h & 0x7f);
        for (std::uint64_t g = (h >> 7) & gmask_;; g = (g + 1) & gmask_) {
            const uint8x16_t ctrl = vld1q_u8(ctrl_.data() + g * 16);
            for (std::uint64_t m = nibbles(vceqq_u8(ctrl, vdupq_n_u8(tag))); m; m = clear_nibble(m)) {
                const Slot& e = slots_[g * 16 + (__builtin_ctzll(m) >> 2)];
                if (matches(e, k, h)) return {e.code, false};
            }
            const std::uint64_t empty = nibbles(vcltzq_s8(vreinterpretq_s8_u8(ctrl)));  // kEmpty has the top bit set
            if (empty) {
                const std::uint64_t at = g * 16 + (__builtin_ctzll(empty) >> 2);
                ctrl_[at] = tag;
                slots_[at] = Slot{word_of(k, h), new_code};
                if constexpr (!kInline) {
                    if (static_cast<std::int64_t>(keys_.size()) <= new_code) keys_.resize(new_code + 1);
                    keys_[new_code] = k;
                }
                if (++size_ * 8 > static_cast<std::int64_t>(slots_.size()) * 7) grow();
                return {new_code, true};
            }
        }
    }

    std::int64_t find(Key k) const {
        const std::uint64_t h = Hash{}(k);
        const std::uint8_t tag = static_cast<std::uint8_t>(h & 0x7f);
        for (std::uint64_t g = (h >> 7) & gmask_;; g = (g + 1) & gmask_) {
            const uint8x16_t ctrl = vld1q_u8(ctrl_.data() + g * 16);
            for (std::uint64_t m = nibbles(vceqq_u8(ctrl, vdupq_n_u8(tag))); m; m = clear_nibble(m)) {
                const Slot& e = slots_[g * 16 + (__builtin_ctzll(m) >> 2)];
                if (matches(e, k, h)) return e.code;
            }
            if (nibbles(vcltzq_s8(vreinterpretq_s8_u8(ctrl)))) return -1;
        }
    }

private:
    static constexpr std::uint8_t kEmpty = 0x80;
    struct Slot {
        std::uint64_t word;  // the key itself for 8-byte keys, else the key's hash
        std::int64_t code;
    };

    // A 16-byte compare result (0x00 or 0xFF per byte) as a 64-bit mask with
    // four bits per byte: shift each 16-bit lane right by 4 and narrow.
    static std::uint64_t nibbles(uint8x16_t cmp) {
        return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4)), 0);
    }
    static std::uint64_t clear_nibble(std::uint64_t m) { return m & ~(std::uint64_t{0xF} << (__builtin_ctzll(m) & ~3u)); }

    static std::uint64_t word_of(Key k, std::uint64_t h) {
        if constexpr (kInline) return k;
        else return h;
    }
    bool matches(const Slot& e, Key k, std::uint64_t h) const {
        if constexpr (kInline) return e.word == k;
        else return e.word == h && keys_[e.code] == k;
    }

    void grow() {
        std::vector<std::uint8_t> old_ctrl(ctrl_.size() * 2, kEmpty);
        std::vector<Slot> old_slots(slots_.size() * 2);
        old_ctrl.swap(ctrl_);
        old_slots.swap(slots_);
        gmask_ = slots_.size() / 16 - 1;
        for (std::size_t i = 0; i < old_slots.size(); ++i) {
            if (old_ctrl[i] & kEmpty) continue;
            const Slot& e = old_slots[i];
            std::uint64_t h;
            if constexpr (kInline) h = Hash{}(e.word);
            else h = e.word;
            for (std::uint64_t g = (h >> 7) & gmask_;; g = (g + 1) & gmask_) {
                const std::uint64_t empty = nibbles(vcltzq_s8(vreinterpretq_s8_u8(vld1q_u8(ctrl_.data() + g * 16))));
                if (empty) {
                    const std::uint64_t at = g * 16 + (__builtin_ctzll(empty) >> 2);
                    ctrl_[at] = old_ctrl[i];
                    slots_[at] = e;
                    break;
                }
            }
        }
    }

    std::vector<std::uint8_t> ctrl_ = std::vector<std::uint8_t>(32, kEmpty);
    std::vector<Slot> slots_ = std::vector<Slot>(32);
    std::uint64_t gmask_ = 1;  // groups - 1
    std::int64_t size_ = 0;
    std::vector<Key> keys_;
};

// The table everything uses (M10: Swiss, after measuring it against LinearTable).
template <class Key, class Hash>
using KeyTable = SwissTable<Key, Hash>;

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

// Dense codes for the values of one "build" column (NA rows get -1), and the
// codes of another column's values looked up in them (-1: NA or absent).
// The build side of a hash join (M10): only it is inserted; the probe side
// only looks up. Keys follow factorize: NaN = NaN, 0.0 = -0.0.
class ValueCodes {
public:
    explicit ValueCodes(const Column& build);
    const std::vector<std::int64_t>& codes() const { return codes_; }
    std::int64_t ngroups() const { return ngroups_; }
    std::vector<std::int64_t> lookup(const Column& probe) const;

private:
    Column build_;  // kept alive: string keys point into its bytes
    std::vector<std::int64_t> codes_;
    std::int64_t ngroups_ = 0;
    std::unique_ptr<KeyTable<std::uint64_t, Fmix64>> words_;
    std::unique_ptr<KeyTable<std::string_view, HashBytes>> strings_;
};

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
