#include "frame/ops/hash.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <atomic>
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

}  // namespace

Factorized factorize(const Column& col) {
    using U = std::uint64_t;
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
        return factorize_with<std::string_view, HashBytes>(col, [&col](std::int64_t i) { return col.string_at(i); });
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
    Column out = make_fixed(DType::Int64, n);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    std::iota(o, o + n, std::int64_t{0});
    const auto* v = counts.data<std::int64_t>();
    Unlocked unlocked(n);
    // A comparison sort: it orders one count per distinct value, not the rows.
    if (ascending) std::stable_sort(o, o + n, [v](std::int64_t x, std::int64_t y) { return v[x] < v[y]; });
    else std::stable_sort(o, o + n, [v](std::int64_t x, std::int64_t y) { return v[x] > v[y]; });
    return out;
}

}  // namespace sd::frame

namespace sd::frame {

bool contains_key(const Column& col, const Column& value) {
    if (col.dtype != value.dtype || value.length != 1) throw nb::type_error("contains_key: one value of the column's dtype");
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
void find_rows(const Partitioned<Key, Hash>& table, const Column& probe, KeyOf key_of, std::int64_t skip,
               std::int64_t* out) {
    const bool nulls = probe.null_count > 0;
    for_chunks(probe.length, [&](std::int64_t lo, std::int64_t hi) {
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
    }, 8);
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
    // The cache is built with the GIL held: two Python threads must not
    // build it at once. The lookups after it read only.
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
    const std::int64_t skip = na_code_ >= 0 ? na_code_ : kNoSkip;  // the same renumbering as the build rows
    std::int64_t* o = out.data();
    switch (probe.dtype) {
    case DType::String:
        find_rows(*strings_, probe, string_keys(probe), skip, o);
        break;
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
