// Hashing: the key table, factorize (values -> dense int codes) and the
// label lookup engine of Index (DESIGN.md §2.5). factorize is the base of
// unique, duplicated and value_counts (M3), and later groupby and merge.
#pragma once

#include "core/column.hpp"
#include "parallel/pool.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
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

    SwissTable() { allocate(32); }
    SwissTable(SwissTable&&) noexcept = default;
    SwissTable& operator=(SwissTable&&) noexcept = default;

    std::pair<std::int64_t, bool> insert(Key k, std::int64_t new_code) { return insert(k, Hash{}(k), new_code); }
    std::int64_t find(Key k) const { return find(k, Hash{}(k)); }

    // Room for `n` keys without growing (M11: a partition knows how many
    // rows it has, an upper bound on its keys). Only on an empty table.
    // Costs almost nothing up front: the control bytes come zeroed from
    // calloc (fresh pages, touched only when used) and the slots are left
    // uninitialized, read only where a control byte says "full".
    void reserve(std::int64_t n) {
        std::size_t slots = 32;
        while (static_cast<std::int64_t>(slots) * 7 < n * 8) slots *= 2;
        if (slots > capacity()) allocate(slots);
    }

    // The same with the key's hash already computed (M11: the partitioned
    // factorize hashes every row once, up front).
    std::pair<std::int64_t, bool> insert(Key k, std::uint64_t h, std::int64_t new_code) {
        const std::uint8_t tag = tag_of(h);
        for (std::uint64_t g = (h >> 7) & gmask_;; g = (g + 1) & gmask_) {
            const uint8x16_t ctrl = vld1q_u8(ctrl_.get() + g * 16);
            for (std::uint64_t m = nibbles(vceqq_u8(ctrl, vdupq_n_u8(tag))); m; m = clear_nibble(m)) {
                const Slot& e = slots_[g * 16 + (__builtin_ctzll(m) >> 2)];
                if (matches(e, k, h)) return {e.code, false};
            }
            const std::uint64_t empty = nibbles(vceqzq_u8(ctrl));
            if (empty) {
                const std::uint64_t at = g * 16 + (__builtin_ctzll(empty) >> 2);
                ctrl_[at] = tag;
                slots_[at] = Slot{word_of(k, h), new_code};
                if constexpr (!kInline) {
                    if (static_cast<std::int64_t>(keys_.size()) <= new_code) keys_.resize(new_code + 1);
                    keys_[new_code] = k;
                }
                if (++size_ * 8 > static_cast<std::int64_t>(capacity()) * 7) grow();
                return {new_code, true};
            }
        }
    }

    std::int64_t find(Key k, std::uint64_t h) const {
        const std::uint8_t tag = tag_of(h);
        for (std::uint64_t g = (h >> 7) & gmask_;; g = (g + 1) & gmask_) {
            const uint8x16_t ctrl = vld1q_u8(ctrl_.get() + g * 16);
            for (std::uint64_t m = nibbles(vceqq_u8(ctrl, vdupq_n_u8(tag))); m; m = clear_nibble(m)) {
                const Slot& e = slots_[g * 16 + (__builtin_ctzll(m) >> 2)];
                if (matches(e, k, h)) return e.code;
            }
            if (nibbles(vceqzq_u8(ctrl))) return -1;
        }
    }

private:
    // Control bytes: 0 is empty (so calloc'd memory is an empty table);
    // a full slot holds 0x80 | the low 7 bits of the hash (the tag).
    static std::uint8_t tag_of(std::uint64_t h) { return static_cast<std::uint8_t>(0x80 | (h & 0x7f)); }
    struct Slot {
        std::uint64_t word;  // the key itself for 8-byte keys, else the key's hash
        std::int64_t code;
    };
    struct Free {
        void operator()(void* p) const { std::free(p); }
    };

    std::size_t capacity() const { return static_cast<std::size_t>(gmask_ + 1) * 16; }

    void allocate(std::size_t slots) {
        ctrl_.reset(static_cast<std::uint8_t*>(std::calloc(slots, 1)));
        slots_.reset(static_cast<Slot*>(std::malloc(slots * sizeof(Slot))));
        if (!ctrl_ || !slots_) throw std::bad_alloc();
        gmask_ = slots / 16 - 1;
    }

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
        const std::size_t old_n = capacity();
        std::unique_ptr<std::uint8_t[], Free> old_ctrl = std::move(ctrl_);
        std::unique_ptr<Slot[], Free> old_slots = std::move(slots_);
        allocate(old_n * 2);
        for (std::size_t i = 0; i < old_n; ++i) {
            if (old_ctrl[i] == 0) continue;
            const Slot& e = old_slots[i];
            std::uint64_t h;
            if constexpr (kInline) h = Hash{}(e.word);
            else h = e.word;
            for (std::uint64_t g = (h >> 7) & gmask_;; g = (g + 1) & gmask_) {
                const std::uint64_t empty = nibbles(vceqzq_u8(vld1q_u8(ctrl_.get() + g * 16)));
                if (empty) {
                    const std::uint64_t at = g * 16 + (__builtin_ctzll(empty) >> 2);
                    ctrl_[at] = old_ctrl[i];
                    slots_[at] = e;
                    break;
                }
            }
        }
    }

    std::unique_ptr<std::uint8_t[], Free> ctrl_;
    std::unique_ptr<Slot[], Free> slots_;
    std::uint64_t gmask_ = 0;  // groups - 1
    std::int64_t size_ = 0;
    std::vector<Key> keys_;
};

// The table everything uses (M10: Swiss, after measuring it against LinearTable).
template <class Key, class Hash>
using KeyTable = SwissTable<Key, Hash>;

// factorize split by hash (M11, user decision 2026-11-11). The top bits of
// a key's hash pick one of P partitions; each partition is factorized by
// its own Swiss table, in parallel, then the codes are renumbered by the row
// where each key first appears. So the codes are exactly those of one
// serial pass over the rows (0, 1, ... in order of first appearance) for
// any thread count, and each table is small enough to stay in cache.
//
// P is 256 from kThreshold rows up when the pool can run in parallel, and
// 1 otherwise -- then build() is the plain serial pass. The codes do not
// depend on P at all, so P may follow the thread count (user decision
// 2026-11-11: one thread keeps the serial speed; DESIGN.md §2.6).
// Equal keys share a hash, so they always land in one partition. NA rows
// are kept out of the tables: all of them share one code (API.md §13).
template <class Key, class Hash>
class Partitioned {
public:
    // Factorizes rows [0, n): key_of(i) is row i's key, valid(i) false for
    // NA rows. Writes codes[0, n) and returns the first row of each code.
    template <class KeyOf, class Valid>
    std::vector<std::int64_t> build(std::int64_t n, KeyOf key_of, Valid valid, std::int64_t* codes);

    // The code of `k`, or -1 when it is not a key of the build rows.
    std::int64_t find(Key k) const {
        const std::uint64_t h = Hash{}(k);
        if (shift_ == 64) return tables_[0].find(k, h);  // one partition: its codes are the codes
        const std::int64_t local = tables_[static_cast<std::size_t>(h >> shift_)].find(k, h);
        return local < 0 ? -1 : global_[static_cast<std::size_t>(h >> shift_)][static_cast<std::size_t>(local)];
    }

    std::int64_t ngroups() const { return ngroups_; }
    std::int64_t na_code() const { return na_code_; }  // -1 without NA rows

private:
    int shift_ = 64;  // partition = hash >> shift_ (64: one partition)
    std::vector<SwissTable<Key, Hash>> tables_;
    std::vector<std::vector<std::int64_t>> global_;  // per partition: local code -> code
    std::int64_t ngroups_ = 0;
    std::int64_t na_code_ = -1;
};

template <class Key, class Hash>
template <class KeyOf, class Valid>
std::vector<std::int64_t> Partitioned<Key, Hash>::build(std::int64_t n, KeyOf key_of, Valid valid, std::int64_t* codes) {
    using parallel::for_chunks;
    using parallel::kChunkRows;
    // 8-byte keys are their own word and rehashing one is a few cycles;
    // strings keep their hash, computed once (rehashing would read the bytes again).
    constexpr bool kInline = std::is_same_v<Key, std::uint64_t>;
    constexpr int kBits = 8;
    const int bits = n >= parallel::kThreshold && parallel::available() ? kBits : 0;
    const std::size_t P = std::size_t{1} << bits;
    shift_ = 64 - bits;
    tables_ = std::vector<SwissTable<Key, Hash>>(P);
    global_.assign(P, {});
    if (P == 1) {  // the serial pass: codes in order of first appearance, NA rows sharing one
        SwissTable<Key, Hash>& table = tables_[0];
        std::vector<std::int64_t> first;
        parallel::Unlocked unlocked(n);
        for (std::int64_t i = 0; i < n; ++i) {
            if (!valid(i)) {
                if (na_code_ < 0) {
                    na_code_ = static_cast<std::int64_t>(first.size());
                    first.push_back(i);
                }
                codes[i] = na_code_;
                continue;
            }
            auto [code, added] = table.insert(key_of(i), static_cast<std::int64_t>(first.size()));
            if (added) first.push_back(i);
            codes[i] = code;
        }
        ngroups_ = static_cast<std::int64_t>(first.size());
        return first;
    }
    auto part_of = [this](std::uint64_t h) -> std::size_t { return shift_ == 64 ? 0 : static_cast<std::size_t>(h >> shift_); };
    const std::size_t C = static_cast<std::size_t>(parallel::num_chunks(n));
    auto chunk = [](std::int64_t lo) { return static_cast<std::size_t>(lo / kChunkRows); };

    // 1. Count rows per (chunk, partition); strings also keep their hash.
    std::vector<std::uint64_t> hashes(kInline ? 0 : static_cast<std::size_t>(n));
    std::vector<std::int64_t> count(C * P, 0);
    parallel::FirstRow first_na;
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t* cnt = count.data() + chunk(lo) * P;
        std::int64_t na = -1;
        for (std::int64_t i = lo; i < hi; ++i) {
            if (!valid(i)) {
                if (na < 0) na = i;
                continue;
            }
            const std::uint64_t h = Hash{}(key_of(i));
            if constexpr (!kInline) hashes[static_cast<std::size_t>(i)] = h;
            ++cnt[part_of(h)];
        }
        if (na >= 0) first_na.offer(na);
    }, 4);
    // 2. Where each (partition, chunk) starts: partition by partition, and
    // within one, chunk by chunk, so every partition lists its rows in row
    // order -- the order a serial pass would meet them.
    std::vector<std::int64_t> pstart(P + 1, 0);
    std::vector<std::int64_t> at(C * P);
    std::int64_t pos = 0;
    for (std::size_t p = 0; p < P; ++p) {
        pstart[p] = pos;
        for (std::size_t c = 0; c < C; ++c) {
            at[c * P + p] = pos;
            pos += count[c * P + p];
        }
    }
    pstart[P] = pos;
    // 3. Scatter (word, row) entries to their partitions: the key for 8-byte
    // keys, the hash for strings. A partition then reads its entries front
    // to back instead of jumping across the rows (M11: reading keys and
    // hashes through row numbers made one thread 2-4x slower).
    struct Entry {
        std::uint64_t word;
        std::int64_t row;
    };
    std::vector<Entry> entries(static_cast<std::size_t>(pos));
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t* next = at.data() + chunk(lo) * P;
        for (std::int64_t i = lo; i < hi; ++i) {
            if (!valid(i)) {
                codes[i] = -1;
                continue;
            }
            std::uint64_t w, h;
            if constexpr (kInline) {
                w = key_of(i);
                h = Hash{}(w);
            } else {
                w = h = hashes[static_cast<std::size_t>(i)];
            }
            const std::int64_t j = next[part_of(h)]++;
            entries[static_cast<std::size_t>(j)] = Entry{w, i};
            codes[i] = j;  // where row i's entry went, until step 6
        }
    });
    // 4. Each partition: local codes in its row order (written over the
    // entry's word), and its first rows, which are in row order too.
    std::vector<std::vector<std::int64_t>> firsts(P);
    parallel::parallel_for(static_cast<std::int64_t>(P), [&](std::int64_t pt) {
        const auto p = static_cast<std::size_t>(pt);
        SwissTable<Key, Hash>& table = tables_[p];
        table.reserve(pstart[p + 1] - pstart[p]);  // never grows: no rehashing, no allocation in the loop
        std::vector<std::int64_t>& f = firsts[p];
        f.reserve(static_cast<std::size_t>(pstart[p + 1] - pstart[p]));
        for (std::int64_t j = pstart[p]; j < pstart[p + 1]; ++j) {
            Entry& e = entries[static_cast<std::size_t>(j)];
            std::pair<std::int64_t, bool> r;
            if constexpr (kInline) r = table.insert(e.word, Hash{}(e.word), static_cast<std::int64_t>(f.size()));
            else r = table.insert(key_of(e.row), e.word, static_cast<std::int64_t>(f.size()));
            if (r.second) f.push_back(e.row);
            e.word = static_cast<std::uint64_t>(r.first);
        }
    });
    // 5. The code of a key is the number of first rows before its own
    // first row. Each row chunk gathers the first rows that fall in it (a
    // slice of every partition's sorted list) and numbers them in row order.
    // A task writes only its own chunk's part of every array, so no two
    // threads write one cache line (scattered writes from all partitions at
    // once made 10 threads barely 3x faster than one).
    const std::int64_t na_first = first_na.any() ? first_na.row() : -1;
    std::vector<std::int64_t> fstart(C + 1, 0);
    // bound[p * (C + 1) + c]: where chunk c's rows start in firsts[p]. One
    // walk down each sorted list (binary searches per chunk and partition
    // mispredicted enough to cost 7 ms at 630k keys).
    std::vector<std::int64_t> bound(P * (C + 1));
    parallel::parallel_for(static_cast<std::int64_t>(P), [&](std::int64_t pt) {
        const auto p = static_cast<std::size_t>(pt);
        const auto& f = firsts[p];
        std::int64_t* b = bound.data() + p * (C + 1);
        std::size_t l = 0;
        for (std::size_t c = 0; c <= C; ++c) {
            const std::int64_t row = static_cast<std::int64_t>(c) * kChunkRows;
            while (l < f.size() && f[l] < row) ++l;
            b[c] = static_cast<std::int64_t>(l);
        }
    });
    auto slice = [&](std::size_t p, std::int64_t lo, std::int64_t) {  // firsts[p] within chunk [lo, hi)
        const std::int64_t* b = bound.data() + p * (C + 1) + chunk(lo);
        return std::pair{b[0], b[1]};
    };
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t k = na_first >= lo && na_first < hi;
        for (std::size_t p = 0; p < P; ++p) {
            auto [a, b] = slice(p, lo, hi);
            k += b - a;
        }
        fstart[chunk(lo) + 1] = k;
    });
    for (std::size_t c = 0; c < C; ++c) fstart[c + 1] += fstart[c];
    ngroups_ = fstart[C];
    std::vector<std::int64_t> first(static_cast<std::size_t>(ngroups_));
    for (std::size_t p = 0; p < P; ++p) global_[p].resize(firsts[p].size());
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        // rank[i - lo]: 1 + the code of first row i, 0 for other rows
        std::vector<std::int64_t> rank(static_cast<std::size_t>(hi - lo), 0);
        for (std::size_t p = 0; p < P; ++p) {
            auto [a, b] = slice(p, lo, hi);
            for (auto l = a; l < b; ++l) rank[static_cast<std::size_t>(firsts[p][static_cast<std::size_t>(l)] - lo)] = 1;
        }
        if (na_first >= lo && na_first < hi) rank[static_cast<std::size_t>(na_first - lo)] = 1;
        // Branch-free numbering: first rows are often too dense and too
        // random for a branch to predict.
        std::int64_t g = fstart[chunk(lo)];
        std::int64_t* rk = rank.data();
        for (std::int64_t i = 0, len = hi - lo; i < len; ++i) {
            const std::int64_t is = rk[i];
            g += is;
            rk[i] = g & -is;
        }
        for (std::size_t p = 0; p < P; ++p) {
            auto [a, b] = slice(p, lo, hi);
            for (auto l = a; l < b; ++l) {
                const std::int64_t row = firsts[p][static_cast<std::size_t>(l)];
                const std::int64_t code = rk[row - lo] - 1;
                global_[p][static_cast<std::size_t>(l)] = code;
                first[static_cast<std::size_t>(code)] = row;
            }
        }
        if (na_first >= lo && na_first < hi) first[static_cast<std::size_t>(rk[na_first - lo] - 1)] = na_first;
        if (na_first >= lo && na_first < hi) na_code_ = rank[static_cast<std::size_t>(na_first - lo)] - 1;
    }, 4);
    // 6. Entries' local codes -> codes (each partition reads its own entries
    // front to back), then each row reads its entry, in row order.
    parallel::parallel_for(static_cast<std::int64_t>(P), [&](std::int64_t pt) {
        const auto p = static_cast<std::size_t>(pt);
        const std::vector<std::int64_t>& g = global_[p];
        for (std::int64_t j = pstart[p]; j < pstart[p + 1]; ++j) {
            Entry& e = entries[static_cast<std::size_t>(j)];
            e.word = static_cast<std::uint64_t>(g[static_cast<std::size_t>(e.word)]);
        }
    });
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        for (std::int64_t i = lo; i < hi; ++i)
            codes[i] = codes[i] < 0 ? na_code_ : static_cast<std::int64_t>(entries[static_cast<std::size_t>(codes[i])].word);
    });
    return first;
}

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
    std::int64_t na_code_ = -1;  // the build's NA code, taken out of the numbering
    std::unique_ptr<Partitioned<std::uint64_t, Fmix64>> words_;
    std::unique_ptr<Partitioned<std::string_view, HashBytes>> strings_;
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
    std::unique_ptr<Partitioned<std::uint64_t, Fmix64>> ints_;
    std::unique_ptr<Partitioned<std::string_view, HashBytes>> strings_;
    // Positions grouped by code (built on first use by all_positions).
    mutable std::vector<std::int64_t> group_start_, group_rows_;
};

}  // namespace sd
