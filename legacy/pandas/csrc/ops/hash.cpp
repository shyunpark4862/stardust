#include "ops/hash.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <string_view>
#include <string>
#include <type_traits>

namespace nb = nanobind;

namespace sd {

namespace {

template <bool HasNA, class Key, class Hash, class KeyOf>
Factorized factorize_with(const Column& col, KeyOf key_of) {
    const std::int64_t n = col.length;
    Factorized out;
    out.codes = make_fixed(DType::Int64, n);
    auto* codes = reinterpret_cast<std::int64_t*>(out.codes.values->data());
    std::vector<std::int64_t> first;
    KeyTable<Key, Hash> table;
    std::int64_t na_code = -1;
    for (std::int64_t i = 0; i < n; ++i) {
        if constexpr (HasNA) {
            if (!col.is_valid(i)) {  // all NA rows are one key, kept out of the table
                if (na_code < 0) {
                    na_code = static_cast<std::int64_t>(first.size());
                    first.push_back(i);
                }
                codes[i] = na_code;
                continue;
            }
        }
        auto [code, added] = table.insert(key_of(i), static_cast<std::int64_t>(first.size()));
        if (added) first.push_back(i);
        codes[i] = code;
    }
    out.ngroups = static_cast<std::int64_t>(first.size());
    out.first = make_fixed(DType::Int64, out.ngroups);
    std::copy(first.begin(), first.end(), reinterpret_cast<std::int64_t*>(out.first.values->data()));
    return out;
}

// Picks the loop without NA checks when the column has no NA (a property
// of the column, DESIGN.md §2.1).
template <class Key, class Hash, class KeyOf>
Factorized factorize_any(const Column& col, KeyOf key_of) {
    return col.validity ? factorize_with<true, Key, Hash>(col, key_of) : factorize_with<false, Key, Hash>(col, key_of);
}

}  // namespace

Factorized factorize(const Column& col) {
    using U = std::uint64_t;
    switch (col.dtype) {
    case DType::Bool: {
        const auto* v = col.data<std::uint8_t>();
        return factorize_any<U, Fmix64>(col, [v](std::int64_t i) -> U { return v[i]; });
    }
    case DType::Int64: {
        const auto* v = col.data<std::int64_t>();
        return factorize_any<U, Fmix64>(col, [v](std::int64_t i) { return static_cast<U>(v[i]); });
    }
    case DType::Float64: {
        const auto* v = col.data<double>();
        return factorize_any<U, Fmix64>(col, [v](std::int64_t i) { return float_key(v[i]); });
    }
    case DType::String:
        return factorize_any<std::string_view, HashBytes>(col, [&col](std::int64_t i) { return col.string_at(i); });
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

namespace sd {

bool contains_key(const Column& col, const Column& value) {
    if (col.dtype != value.dtype || value.length != 1) throw nb::type_error("contains_key: one value of the column's dtype");
    if (!value.is_valid(0)) return col.null_count > 0;
    for (std::int64_t i = 0; i < col.length; ++i) {
        if (!col.is_valid(i)) continue;
        bool same = false;
        switch (col.dtype) {
        case DType::Bool: same = col.data<std::uint8_t>()[i] == value.data<std::uint8_t>()[0]; break;
        case DType::Int64: same = col.data<std::int64_t>()[i] == value.data<std::int64_t>()[0]; break;
        case DType::Float64: same = float_key(col.data<double>()[i]) == float_key(value.data<double>()[0]); break;
        case DType::String: same = col.string_at(i) == value.string_at(0); break;
        }
        if (same) return true;
    }
    return false;
}

LabelTable::LabelTable(const Column& labels) : labels_(labels) {
    if (labels.null_count != 0) throw nb::value_error("index labels cannot be NA");
    const std::int64_t n = labels.length;
    codes_.resize(static_cast<std::size_t>(n));
    auto add = [&](auto& table, auto key, std::int64_t i) {
        auto [code, added] = table.insert(key, ngroups_);
        if (added) {
            first_.push_back(i);
            ++ngroups_;
        }
        codes_[i] = code;
    };
    if (labels.dtype == DType::Int64) {
        ints_ = std::make_unique<KeyTable<std::uint64_t, Fmix64>>();
        const auto* v = labels.data<std::int64_t>();
        for (std::int64_t i = 0; i < n; ++i) add(*ints_, static_cast<std::uint64_t>(v[i]), i);
    } else if (labels.dtype == DType::String) {
        strings_ = std::make_unique<KeyTable<std::string_view, HashBytes>>();
        for (std::int64_t i = 0; i < n; ++i) add(*strings_, labels_.string_at(i), i);
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
    for (std::int64_t i = 0; i < targets.length; ++i) {
        if (!targets.is_valid(i)) o[i] = -1;
        else if (ints_) o[i] = ints_->find(static_cast<std::uint64_t>(targets.data<std::int64_t>()[i]));
        else o[i] = strings_->find(targets.string_at(i));
    }
    return out;
}

Column LabelTable::first_positions(const Column& targets) const {
    Column out = lookup(targets);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    for (std::int64_t i = 0; i < out.length; ++i)
        if (o[i] >= 0) o[i] = first_[o[i]];
    return out;
}

Column LabelTable::all_positions(const Column& targets) const {
    if (group_start_.empty()) {  // counting sort of positions by code: CSR groups
        group_start_.assign(static_cast<std::size_t>(ngroups_ + 1), 0);
        for (std::int64_t c : codes_) ++group_start_[c + 1];
        for (std::int64_t g = 0; g < ngroups_; ++g) group_start_[g + 1] += group_start_[g];
        group_rows_.resize(codes_.size());
        std::vector<std::int64_t> fill(group_start_.begin(), group_start_.end() - 1);
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(codes_.size()); ++i) group_rows_[fill[codes_[i]]++] = i;
    }
    Column codes = lookup(targets);
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

}  // namespace sd

namespace sd {

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
    if (build.dtype == DType::String) strings_ = std::make_unique<KeyTable<std::string_view, HashBytes>>();
    else words_ = std::make_unique<KeyTable<std::uint64_t, Fmix64>>();
    for (std::int64_t i = 0; i < n; ++i) {
        if (!build.is_valid(i)) {
            codes_[i] = -1;
            continue;
        }
        auto [code, added] = strings_ ? strings_->insert(build_.string_at(i), ngroups_) : words_->insert(word_key(build, i), ngroups_);
        ngroups_ += added;
        codes_[i] = code;
    }
}

std::vector<std::int64_t> ValueCodes::lookup(const Column& probe) const {
    if (probe.dtype != build_.dtype) throw nb::type_error("lookup: the dtypes differ");
    std::vector<std::int64_t> out(static_cast<std::size_t>(probe.length));
    for (std::int64_t i = 0; i < probe.length; ++i)
        out[i] = !probe.is_valid(i) ? -1 : strings_ ? strings_->find(probe.string_at(i)) : words_->find(word_key(probe, i));
    return out;
}

}  // namespace sd
