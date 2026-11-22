#include "frame/ops/hash.hpp"

#include "frame/ops/radix.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <limits>
#include <numeric>
#include <string_view>
#include <string>
#include <type_traits>

namespace nb = nanobind;

namespace sd::frame {

namespace {

template <class Key, class Hash, class KeyOf>
Factorized factorize_with(const Column& col, KeyOf key_of) {
    const std::int64_t n = col.length;
    Factorized out;
    out.codes = make_fixed(DType::Int64, n);
    auto* codes = reinterpret_cast<std::int64_t*>(out.codes.values->data());
    Partitioned<Key, Hash> table;
    std::vector<std::int64_t> first;
    // The loop without NA checks when the column has no NA (a property of
    // the column, docs/design/frame.md §4.4).
    if (col.validity) first = table.build(n, key_of, [&col](std::int64_t i) { return col.is_valid(i); }, codes);
    else first = table.build(n, key_of, [](std::int64_t) { return true; }, codes);
    out.ngroups = table.ngroups();
    out.first = make_fixed(DType::Int64, out.ngroups);
    std::copy(first.begin(), first.end(), reinterpret_cast<std::int64_t*>(out.first.values->data()));
    return out;
}

// Row i's key as an offset into the direct table (value - lo).
template <class T>
auto direct_keys(const Column& col, std::int64_t lo) {
    return [v = col.data<T>(), lo](std::int64_t i) { return static_cast<std::int64_t>(v[i]) - lo; };
}

// The direct factorize, with or without NA rows (`valid`) and over keys
// `key(i)` in [0, span).
template <class KeyOf, class Valid>
Factorized direct_with(std::int64_t n, std::int64_t span, KeyOf key, Valid valid, std::vector<std::int32_t>* code_of) {
    Factorized out;
    out.codes = make_fixed(DType::Int64, n);
    auto* codes = reinterpret_cast<std::int64_t*>(out.codes.values->data());
    std::vector<std::int64_t> first;
    std::vector<std::int32_t> table;  // 1 + code per key, 0: not seen
    // Many keys (a span of a quarter of the rows or more): serial, as the
    // parallel passes below were slower than the one pass with many keys
    // (an atomic minimum on most rows: 1M rows over 1M keys took 5.8 ms at 10
    // threads against 1.3 ms for one).
    const bool many = span * 4 >= n;  // keys often new: no branch on it
    if (many || n < kParallelThreshold || !parallel::available()) {
        // One pass: a key's first row gives it the next code.
        // Branch-free: whether a row's key is new is as good as random for
        // many keys (0.63 of 1M rows over 1M keys), and a mispredicted
        // branch cost more than the table access. Every row writes its
        // slot and the next entry of `first`; only a new key moves past it.
        Unlocked unlocked(n);
        table.assign(static_cast<std::size_t>(span), 0);
        first.resize(static_cast<std::size_t>(n) + 1);
        std::int32_t* t = table.data();
        std::int64_t* f = first.data();
        std::int64_t nf = 0, na_code = -1;
        for (std::int64_t i = 0; i < n; ++i) {
            if (!valid(i)) {
                if (na_code < 0) {
                    na_code = nf;
                    f[nf++] = i;
                }
                codes[i] = na_code;
                continue;
            }
            std::int32_t& slot = t[key(i)];
            const std::int32_t c = slot;
            if (!many) {  // few keys: a new one is rare, the branch predicted
                if (c == 0) [[unlikely]] {
                    f[nf++] = i;
                    slot = static_cast<std::int32_t>(nf);
                    codes[i] = nf - 1;
                } else {
                    codes[i] = c - 1;
                }
                continue;
            }
            const std::int64_t is_new = c == 0;
            f[nf] = i;
            nf += is_new;
            const std::int32_t code = is_new ? static_cast<std::int32_t>(nf) : c;
            slot = code;
            codes[i] = code - 1;
        }
        first.resize(static_cast<std::size_t>(nf));
    } else {
        // In parallel, with the codes of the serial pass, and no branch on
        // the data:
        //  1. head[k]: the first row of key k, an atomic minimum over the
        //     rows (LDSMIN; the result does not depend on the order);
        //  2. a bitmap of the first rows (head[key(i)] == i) and its count
        //     per 64-row word;
        //  3. prefix sums of the counts: a first row's code is the number of
        //     first rows before it (its rank in the bitmap);
        //  4. every row: code = rank(head[key(i)]); a first row also writes
        //     first[code]. NA rows share the code of the first NA row.
        constexpr std::int32_t kNone = INT32_MAX;
        std::vector<std::int32_t> head(static_cast<std::size_t>(span));
        std::int32_t* h = head.data();
        for_chunks(span, [&](std::int64_t lo, std::int64_t hi) { std::fill(h + lo, h + hi, kNone); });
        parallel::FirstRow first_na;
        for_chunks(
            n,
            [&](std::int64_t lo, std::int64_t hi) {
                std::int64_t na = -1;
                for (std::int64_t i = lo; i < hi; ++i) {
                    if (!valid(i)) {
                        if (na < 0) na = i;
                        continue;
                    }
                    std::int32_t* slot = h + key(i);
                    // Read first: once every key has a small row the table's
                    // cache lines stay shared and no row writes.
                    if (static_cast<std::int32_t>(i) < __atomic_load_n(slot, __ATOMIC_RELAXED))
                        __atomic_fetch_min(slot, static_cast<std::int32_t>(i), __ATOMIC_RELAXED);
                }
                if (na >= 0) first_na.offer(na);
            },
            4
        );
        const std::int64_t na_first = first_na.any() ? first_na.row() : -1;
        const std::int64_t W = (n + 63) / 64;
        std::vector<std::uint64_t> bits(static_cast<std::size_t>(W));
        std::vector<std::int64_t> before(static_cast<std::size_t>(W) + 1, 0);  // first rows before word w
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {  // chunks are whole words (kChunkRows % 64 == 0)
            for (std::int64_t w = lo / 64; w * 64 < hi; ++w) {
                std::uint64_t m = 0;
                for (std::int64_t i = w * 64, e = std::min(hi, w * 64 + 64); i < e; ++i) {
                    const bool is_first = valid(i) ? h[key(i)] == i : i == na_first;
                    m |= static_cast<std::uint64_t>(is_first) << (i & 63);
                }
                bits[static_cast<std::size_t>(w)] = m;
                before[static_cast<std::size_t>(w) + 1] = __builtin_popcountll(m);
            }
        });
        for (std::int64_t w = 0; w < W; ++w)
            before[static_cast<std::size_t>(w) + 1] += before[static_cast<std::size_t>(w)];
        first.resize(static_cast<std::size_t>(before[static_cast<std::size_t>(W)]));
        auto rank = [b = bits.data(), p = before.data()](std::int64_t r) {
            return p[r >> 6] + __builtin_popcountll(b[r >> 6] & ((std::uint64_t{1} << (r & 63)) - 1));
        };
        const std::int64_t na_code = na_first >= 0 ? rank(na_first) : -1;
        if (na_first >= 0) first[static_cast<std::size_t>(na_code)] = na_first;
        std::int64_t* f = first.data();
        for_chunks(
            n,
            [&](std::int64_t lo, std::int64_t hi) {
                for (std::int64_t i = lo; i < hi; ++i) {
                    if (!valid(i)) {
                        codes[i] = na_code;
                        continue;
                    }
                    const std::int64_t r = h[key(i)];
                    const std::int64_t c = rank(r);
                    codes[i] = c;
                    if (r == i) f[c] = i;
                }
            },
            2
        );
        if (code_of) {  // 1 + code per key, 0 for keys that never appear
            table.resize(static_cast<std::size_t>(span));
            std::int32_t* t = table.data();
            for_chunks(span, [&](std::int64_t lo, std::int64_t hi) {
                for (std::int64_t k = lo; k < hi; ++k)
                    t[k] = h[k] == kNone ? 0 : static_cast<std::int32_t>(rank(h[k]) + 1);
            });
        }
    }
    out.ngroups = static_cast<std::int64_t>(first.size());
    out.first = make_fixed(DType::Int64, out.ngroups);
    std::copy(first.begin(), first.end(), reinterpret_cast<std::int64_t*>(out.first.values->data()));
    if (code_of) code_of->swap(table);
    return out;
}

}  // namespace

std::optional<DirectRange> direct_range(const Column& col) {
    const std::int64_t n = col.length;
    if (n == 0 || n > INT32_MAX - 1) return std::nullopt;
    if (col.dtype == DType::Bool) return DirectRange{0, 2};
    if (col.dtype != DType::Int64) return std::nullopt;
    const auto* v = col.data<std::int64_t>();
    const std::size_t C = static_cast<std::size_t>(num_chunks(n));
    std::vector<std::int64_t> mins(C), maxs(C);
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t a = v[lo], b = v[lo];
        for (std::int64_t i = lo; i < hi; ++i) {
            a = std::min(a, v[i]);
            b = std::max(b, v[i]);
        }
        mins[static_cast<std::size_t>(lo / kChunkRows)] = a;
        maxs[static_cast<std::size_t>(lo / kChunkRows)] = b;
    });
    const std::int64_t lo = *std::min_element(mins.begin(), mins.end());
    const std::int64_t hi = *std::max_element(maxs.begin(), maxs.end());
    const std::uint64_t span =
        static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo) + 1;  // no overflow as uint64
    if (span == 0 || span > static_cast<std::uint64_t>(direct_span_limit(n))) return std::nullopt;
    return DirectRange{lo, static_cast<std::int64_t>(span)};
}

Factorized factorize_direct(const Column& col, DirectRange r, std::vector<std::int32_t>* code_of) {
    const std::int64_t n = col.length;
    auto run = [&](auto key) {
        if (col.null_count > 0)
            return direct_with(n, r.span, key, [&col](std::int64_t i) { return col.is_valid(i); }, code_of);
        return direct_with(n, r.span, key, [](std::int64_t) { return true; }, code_of);
    };
    if (col.dtype == DType::Bool) return run(direct_keys<std::uint8_t>(col, r.lo));
    return run(direct_keys<std::int64_t>(col, r.lo));
}

std::optional<Column> unique_sorted(const Column& col) {
    const std::optional<DirectRange> r = direct_range(col);
    if (!r) return std::nullopt;
    const std::int64_t n = col.length, lo = r->lo, span = r->span;
    // One bit per possible key: set the keys that appear, then read them
    // out in order. The values are the keys themselves, so no row is chosen
    // to stand for a key.
    std::vector<std::uint64_t> seen(static_cast<std::size_t>((span + 63) / 64), 0);
    std::uint64_t* w = seen.data();
    const std::uint8_t* bits = col.validity_bits();
    const std::int64_t off = col.offset;
    const bool nulls = col.null_count > 0;
    {
        const Unlocked unlocked(n);
        auto mark = [&](auto value) {
            for (std::int64_t i = 0; i < n; ++i) {
                const std::uint64_t k = static_cast<std::uint64_t>(value(i) - lo);
                // An NA row (value 0, in range) sets no bit.
                const std::uint64_t set = nulls ? std::uint64_t{bitmap::get(bits, off + i)} : 1;
                w[k >> 6] |= set << (k & 63);
            }
        };
        if (col.dtype == DType::Bool)
            mark([v = col.data<std::uint8_t>()](std::int64_t i) { return std::int64_t{v[i]}; });
        else mark([v = col.data<std::int64_t>()](std::int64_t i) { return v[i]; });
    }
    std::int64_t count = 0;
    for (std::uint64_t x : seen) count += __builtin_popcountll(x);
    const std::int64_t total = count + (nulls ? 1 : 0);  // NA once, last
    Column out = make_fixed(col.dtype, total);
    {
        const Unlocked unlocked(span);
        std::int64_t at = 0;
        auto emit = [&](auto* o) {
            using T = std::remove_reference_t<decltype(*o)>;
            for (std::size_t j = 0; j < seen.size(); ++j)
                for (std::uint64_t x = seen[j]; x; x &= x - 1)
                    o[at++] = static_cast<T>(lo + static_cast<std::int64_t>(j) * 64 + __builtin_ctzll(x));
            if (nulls) o[at] = T{};
        };
        if (col.dtype == DType::Bool) emit(reinterpret_cast<std::uint8_t*>(out.values->data()));
        else emit(reinterpret_cast<std::int64_t*>(out.values->data()));
    }
    if (nulls) {
        std::uint8_t* b = add_validity(out);
        bitmap::clear(b, total - 1);
        out.null_count = 1;
    }
    return out;
}

Factorized factorize(const Column& col) {
    using U = std::uint64_t;
    if (auto r = direct_range(col)) return factorize_direct(col, *r);
    switch (col.dtype) {
        case DType::Bool: {
            const auto* v = col.data<std::uint8_t>();
            return factorize_with<U, Fmix64>(col, [v](std::int64_t i) -> U { return v[i]; });
        }
        case DType::Int64: {
            const auto* v = col.data<std::int64_t>();
            return factorize_with<U, Fmix64>(col, [v](std::int64_t i) { return static_cast<U>(v[i]); });
        }
        case DType::Float64: {
            const auto* v = col.data<double>();
            return factorize_with<U, Fmix64>(col, [v](std::int64_t i) { return float_key(v[i]); });
        }
        case DType::String:
            return factorize_with<std::string_view, HashBytes>(col, [&col](std::int64_t i) {
                return col.string_at(i);
            });
        case DType::Complex128: break;  // not a frame dtype
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
        const std::int64_t m = next.ngroups;
        for_chunks(n, [=](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t i = lo; i < hi; ++i) a[i] = a[i] * m + b[i];
        });
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
    Unlocked unlocked(codes.length);  // serial, without the GIL (docs/design/frame.md §11.4)
    for (std::int64_t i = 0; i < codes.length; ++i) {
        ++cnt[c[i]];
        lst[c[i]] = i;
    }
    return {counts, last};
}

Column stable_order(const Column& counts, bool ascending) {
    const std::int64_t n = counts.length;
    // The counts are >= 0, so as uint64 they keep their order; descending
    // flips every bit (docs/design/frame.md §9.3). The stable radix sort keeps
    // equal counts in order of first appearance, for any thread count.
    std::vector<std::uint64_t> keys(static_cast<std::size_t>(n));
    std::vector<std::int64_t> rows(static_cast<std::size_t>(n));
    const auto* v = counts.data<std::int64_t>();
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        for (std::int64_t i = lo; i < hi; ++i) {
            const auto k = static_cast<std::uint64_t>(v[i]);
            keys[i] = ascending ? k : ~k;
            rows[i] = i;
        }
    });
    radix_sort(keys, rows);
    Column out = make_fixed(DType::Int64, n);
    const Unlocked unlocked(n);
    std::copy(rows.begin(), rows.end(), reinterpret_cast<std::int64_t*>(out.values->data()));
    return out;
}

}  // namespace sd::frame

namespace sd::frame {

bool contains_key(const Column& col, const Column& value) {
    if (col.dtype != value.dtype || value.length != 1)
        throw nb::type_error("contains_key: one value of the column's dtype");
    if (!value.is_valid(0)) return col.null_count > 0;
    auto same = [&](std::int64_t i) {
        if (!col.is_valid(i)) return false;
        switch (col.dtype) {
            case DType::Bool: return col.data<std::uint8_t>()[i] == value.data<std::uint8_t>()[0];
            case DType::Int64: return col.data<std::int64_t>()[i] == value.data<std::int64_t>()[0];
            case DType::Float64: return float_key(col.data<double>()[i]) == float_key(value.data<double>()[0]);
            case DType::String: return col.string_at(i) == value.string_at(0);
            case DType::Complex128: break;  // not a frame dtype
        }
        return false;
    };
    std::atomic<bool> found{false};
    for_chunks(col.length, [&](std::int64_t lo, std::int64_t hi) {
        for (std::int64_t i = lo; i < hi && !found.load(std::memory_order_relaxed); ++i)
            if (same(i)) found = true;
    });
    return found;
}

namespace {

// The code of every probe row in `table` (-1 for NA rows and keys it does
// not have), codes above `skip` moved down by one (docs/design/frame.md §8.8).
// Dispatched once per call -- dtype, NA or not -- so the row loop is one
// find and a store: asking "valid? string? which dtype?" on every row was
// half of a one-thread 1M-row merge. Everything the loop reads is a local
// of the task, so the int64 stores cannot alias it (docs/design/frame.md §7.6).
template <class Key, class Hash, class KeyOf>
void find_rows(
    const Partitioned<Key, Hash>& table, const Column& probe, KeyOf key_of, std::int64_t skip, std::int64_t* out
) {
    const bool nulls = probe.null_count > 0;
    for_chunks(
        probe.length,
        [&](std::int64_t lo, std::int64_t hi) {
            const Partitioned<Key, Hash>& t = table;
            const KeyOf key = key_of;
            const std::int64_t s = skip;
            std::int64_t* o = out;
            if (!nulls) {
                for (std::int64_t i = lo; i < hi; ++i) {
                    const std::int64_t c = t.find(key(i));
                    o[i] = c - (c > s);
                }
            } else {
                const std::uint8_t* bits = probe.validity_bits();
                const std::int64_t off = probe.offset;
                for (std::int64_t i = lo; i < hi; ++i) {
                    if (!bitmap::get(bits, off + i)) {
                        o[i] = -1;
                        continue;
                    }
                    const std::int64_t c = t.find(key(i));
                    o[i] = c - (c > s);
                }
            }
        },
        8
    );
}

// Row i's string from pointers held by the closure (not reloaded through
// the Column after every store).
auto string_keys(const Column& c) {
    return [offs = c.string_offsets(), chars = reinterpret_cast<const char*>(c.values->data())](std::int64_t i) {
        return std::string_view(chars + offs[i], static_cast<std::size_t>(offs[i + 1] - offs[i]));
    };
}

// No code is above it: nothing moves.
constexpr std::int64_t kNoSkip = std::numeric_limits<std::int64_t>::max();

}  // namespace

LabelTable::LabelTable(const Column& labels) : labels_(labels) {
    if (labels.null_count != 0) throw nb::value_error("index labels cannot be NA");
    const std::int64_t n = labels.length;
    codes_.resize(static_cast<std::size_t>(n));
    auto all = [](std::int64_t) { return true; };
    if (labels.dtype == DType::Int64) {
        ints_ = std::make_unique<Partitioned<std::uint64_t, Fmix64>>();
        const auto* v = labels.data<std::int64_t>();
        first_ = ints_->build(n, [v](std::int64_t i) { return static_cast<std::uint64_t>(v[i]); }, all, codes_.data());
        ngroups_ = ints_->ngroups();
    } else if (labels.dtype == DType::String) {
        strings_ = std::make_unique<Partitioned<std::string_view, HashBytes>>();
        first_ = strings_->build(n, [this](std::int64_t i) { return labels_.string_at(i); }, all, codes_.data());
        ngroups_ = strings_->ngroups();
    } else {
        throw nb::type_error("index labels must be int64 or string");
    }
}

Column LabelTable::lookup(const Column& targets) const {
    if (targets.dtype != labels_.dtype)
        throw nb::type_error((std::string("cannot look up ") + dtype_name(targets.dtype) + " labels in an " +
                              dtype_name(labels_.dtype) + " index")
                                 .c_str());
    Column out = make_fixed(DType::Int64, targets.length);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    if (ints_) {
        const auto* v = targets.data<std::int64_t>();
        find_rows(*ints_, targets, [v](std::int64_t i) { return static_cast<std::uint64_t>(v[i]); }, kNoSkip, o);
    } else {
        find_rows(*strings_, targets, string_keys(targets), kNoSkip, o);
    }
    return out;
}

Column LabelTable::first_positions(const Column& targets) const {
    Column out = lookup(targets);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    for_chunks(out.length, [&](std::int64_t lo, std::int64_t hi) {
        for (std::int64_t i = lo; i < hi; ++i)
            if (o[i] >= 0) o[i] = first_[o[i]];
    });
    return out;
}

Column LabelTable::all_positions(const Column& targets) const {
    // The CSR is built with the GIL held (nothing before the end of this
    // block releases it): two Python threads must not build it at once. The
    // lookups after it read only.
    if (group_start_.empty()) {  // counting sort of positions by code: CSR groups
        group_start_.assign(static_cast<std::size_t>(ngroups_ + 1), 0);
        for (std::int64_t c : codes_) ++group_start_[c + 1];
        for (std::int64_t g = 0; g < ngroups_; ++g) group_start_[g + 1] += group_start_[g];
        group_rows_.resize(codes_.size());
        std::vector<std::int64_t> fill(group_start_.begin(), group_start_.end() - 1);
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(codes_.size()); ++i) group_rows_[fill[codes_[i]]++] = i;
    }
    Column codes = lookup(targets);
    Unlocked unlocked(targets.length);
    const auto* c = codes.data<std::int64_t>();
    std::int64_t total = 0;
    for (std::int64_t i = 0; i < codes.length; ++i)
        if (c[i] >= 0) total += group_start_[c[i] + 1] - group_start_[c[i]];
    Column out = make_fixed(DType::Int64, total);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    std::int64_t k = 0;
    for (std::int64_t i = 0; i < codes.length; ++i) {
        if (c[i] < 0) continue;
        for (std::int64_t j = group_start_[c[i]]; j < group_start_[c[i] + 1]; ++j) o[k++] = group_rows_[j];
    }
    return out;
}

}  // namespace sd::frame

namespace sd::frame {

namespace {

std::uint64_t word_key(const Column& c, std::int64_t i) {
    switch (c.dtype) {
        case DType::Int64: return static_cast<std::uint64_t>(c.data<std::int64_t>()[i]);
        case DType::Float64: return float_key(c.data<double>()[i]);
        default: return c.data<std::uint8_t>()[i];
    }
}

}  // namespace

ValueCodes::ValueCodes(const Column& build) : build_(build) {
    const std::int64_t n = build.length;
    if ((direct_ = direct_range(build))) {
        Factorized f = factorize_direct(build, *direct_, &code_of_);
        codes_.assign(f.codes.data<std::int64_t>(), f.codes.data<std::int64_t>() + n);
        ngroups_ = f.ngroups;
        if (build.null_count > 0) {  // the NA code out of the numbering, as below
            const auto* first = f.first.data<std::int64_t>();
            std::int64_t na = -1;
            for (std::int64_t g = 0; g < f.ngroups && na < 0; ++g)
                if (!build.is_valid(first[g])) na = g;
            for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
                for (std::int64_t i = lo; i < hi; ++i) {
                    std::int64_t& c = codes_[static_cast<std::size_t>(i)];
                    c = c == na ? -1 : c > na ? c - 1 : c;
                }
            });
            std::int32_t* t = code_of_.data();
            for_chunks(direct_->span, [&](std::int64_t lo, std::int64_t hi) {
                for (std::int64_t k = lo; k < hi; ++k) t[k] -= t[k] > na + 1;  // 1 + code: codes above na move down
            });
            --ngroups_;
            na_code_ = na;
        }
        return;
    }
    codes_.resize(static_cast<std::size_t>(n));
    auto valid = [this](std::int64_t i) { return build_.is_valid(i); };
    if (build.dtype == DType::String) {
        strings_ = std::make_unique<Partitioned<std::string_view, HashBytes>>();
        strings_->build(n, [this](std::int64_t i) { return build_.string_at(i); }, valid, codes_.data());
        ngroups_ = strings_->ngroups();
    } else {
        words_ = std::make_unique<Partitioned<std::uint64_t, Fmix64>>();
        words_->build(n, [this](std::int64_t i) { return word_key(build_, i); }, valid, codes_.data());
        ngroups_ = words_->ngroups();
    }
    // NA rows join nothing: -1, not the shared NA code (docs/api/frame.md §9.1).
    if (build.null_count > 0) {
        const std::int64_t na = strings_ ? strings_->na_code() : words_->na_code();
        // The NA code is a code like the others, numbered by its first row;
        // the codes after it move down by one so the codes stay dense.
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t i = lo; i < hi; ++i) {
                std::int64_t& c = codes_[static_cast<std::size_t>(i)];
                c = c == na ? -1 : c > na ? c - 1 : c;
            }
        });
        --ngroups_;
        na_code_ = na;
    }
}

std::vector<std::int64_t> ValueCodes::lookup(const Column& probe) const {
    if (probe.dtype != build_.dtype) throw nb::type_error("lookup: the dtypes differ");
    std::vector<std::int64_t> out(static_cast<std::size_t>(probe.length));
    if (direct_) {  // the direct table: one read per row, no branch (an out-of-range key reads slot 0 and drops it)
        const std::int32_t* t = code_of_.data();
        const std::uint64_t span = static_cast<std::uint64_t>(direct_->span);
        const std::int64_t lo = direct_->lo;
        std::int64_t* o = out.data();
        const std::uint8_t* bits = probe.validity_bits();
        const std::int64_t off = probe.offset;
        auto run = [&](auto value) {
            for_chunks(
                probe.length,
                [&, value](std::int64_t a, std::int64_t b) {
                    for (std::int64_t i = a; i < b; ++i) {
                        const std::uint64_t k = static_cast<std::uint64_t>(value(i)) - static_cast<std::uint64_t>(lo);
                        const bool in = k < span && (!bits || bitmap::get(bits, off + i));
                        o[i] = (in ? t[in ? k : 0] : 0) - 1;
                    }
                },
                2
            );
        };
        if (probe.dtype == DType::Bool)
            run([v = probe.data<std::uint8_t>()](std::int64_t i) { return std::int64_t{v[i]}; });
        else run([v = probe.data<std::int64_t>()](std::int64_t i) { return v[i]; });
        return out;
    }
    const std::int64_t skip = na_code_ >= 0 ? na_code_ : kNoSkip;  // the same renumbering as the build rows
    std::int64_t* o = out.data();
    switch (probe.dtype) {
        case DType::String: find_rows(*strings_, probe, string_keys(probe), skip, o); break;
        case DType::Int64: {
            const auto* v = probe.data<std::int64_t>();
            find_rows(*words_, probe, [v](std::int64_t i) { return static_cast<std::uint64_t>(v[i]); }, skip, o);
            break;
        }
        case DType::Float64: {
            const auto* v = probe.data<double>();
            find_rows(*words_, probe, [v](std::int64_t i) { return float_key(v[i]); }, skip, o);
            break;
        }
        case DType::Bool: {
            const auto* v = probe.data<std::uint8_t>();
            find_rows(*words_, probe, [v](std::int64_t i) { return std::uint64_t{v[i]}; }, skip, o);
            break;
        }
        case DType::Complex128: break;  // not a frame dtype
    }
    return out;
}

}  // namespace sd::frame
