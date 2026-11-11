#include "ops/sort.hpp"

#include "ops/hash.hpp"
#include "ops/radix.hpp"
#include "parallel/pool.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>

namespace nb = nanobind;

namespace sd {

namespace {

// Stable sort of `order` (positions into `key`) by `key`.
// `order` split into rows with a value and NA rows, each in its order. In
// parallel: per chunk, count; then each chunk copies to its own places.
void split_na(const Column& key, const std::vector<std::int64_t>& order, std::vector<std::int64_t>& valid_rows,
              std::vector<std::int64_t>& na_rows) {
    const auto n = static_cast<std::int64_t>(order.size());
    if (!key.validity) {
        valid_rows = order;
        return;
    }
    const std::size_t C = static_cast<std::size_t>(parallel::num_chunks(n));
    std::vector<std::int64_t> valid_at(C + 1, 0), na_at(C + 1, 0);
    parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t k = 0;
        for (std::int64_t j = lo; j < hi; ++j) k += key.is_valid(order[j]);
        valid_at[lo / parallel::kChunkRows + 1] = k;
        na_at[lo / parallel::kChunkRows + 1] = (hi - lo) - k;
    });
    for (std::size_t c = 0; c < C; ++c) {
        valid_at[c + 1] += valid_at[c];
        na_at[c + 1] += na_at[c];
    }
    valid_rows.resize(static_cast<std::size_t>(valid_at[C]));
    na_rows.resize(static_cast<std::size_t>(na_at[C]));
    parallel::for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t v = valid_at[lo / parallel::kChunkRows], a = na_at[lo / parallel::kChunkRows];
        for (std::int64_t j = lo; j < hi; ++j) {
            const std::int64_t r = order[j];
            if (key.is_valid(r)) valid_rows[v++] = r;
            else na_rows[a++] = r;
        }
    });
}

void sort_by(const Column& key, bool ascending, bool na_last, std::vector<std::int64_t>& order) {
    std::vector<std::int64_t> valid_rows, na_rows;
    split_na(key, order, valid_rows, na_rows);
    const auto nv = static_cast<std::int64_t>(valid_rows.size());

    if (key.dtype == DType::String) {
        // Ranks of the distinct strings stand in for the strings, so that a
        // reversed direction is just reversed ranks.
        Factorized f = factorize(key);
        const auto* code = f.codes.data<std::int64_t>();
        std::vector<std::int64_t> distinct;
        for (std::int64_t g = 0; g < f.ngroups; ++g)
            if (key.is_valid(f.first.data<std::int64_t>()[g])) distinct.push_back(f.first.data<std::int64_t>()[g]);
        order_rows(key, distinct);
        std::vector<std::uint64_t> rank(static_cast<std::size_t>(f.ngroups));
        for (std::size_t r = 0; r < distinct.size(); ++r) rank[code[distinct[r]]] = r;
        std::vector<std::uint64_t> keys(valid_rows.size());
        parallel::for_chunks(nv, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t j = lo; j < hi; ++j) {
                const std::uint64_t k = rank[code[valid_rows[j]]];
                keys[j] = ascending ? k : ~k;
            }
        });
        radix_sort(keys, valid_rows);
    } else {
        std::vector<std::uint64_t> keys(valid_rows.size());
        parallel::for_chunks(nv, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t j = lo; j < hi; ++j) {
                const std::uint64_t k = radix_key(key, valid_rows[j]);
                keys[j] = ascending ? k : ~k;  // the reverse order; equal keys stay equal, so ties keep their order
            }
        });
        radix_sort(keys, valid_rows);
    }

    std::size_t at = 0;
    auto put = [&](const std::vector<std::int64_t>& part) {
        std::copy(part.begin(), part.end(), order.begin() + static_cast<std::ptrdiff_t>(at));
        at += part.size();
    };
    if (na_last) {
        put(valid_rows);
        put(na_rows);
    } else {
        put(na_rows);
        put(valid_rows);
    }
}

}  // namespace

Column argsort(const std::vector<Column>& keys, const std::vector<bool>& ascending, const std::vector<bool>& na_last) {
    if (keys.empty() || ascending.size() != keys.size() || na_last.size() != keys.size())
        throw nb::value_error("argsort: one direction and one NA position per key");
    const std::int64_t n = keys[0].length;
    for (const Column& k : keys)
        if (k.length != n) throw nb::value_error("argsort: keys must have the same length");
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), std::int64_t{0});
    // Several keys: sort by the last key first, then each earlier key with a
    // stable sort, so the first key decides and later keys break its ties.
    for (std::size_t j = keys.size(); j-- > 0;) sort_by(keys[j], ascending[j], na_last[j], order);
    Column out = make_fixed(DType::Int64, n);
    std::copy(order.begin(), order.end(), reinterpret_cast<std::int64_t*>(out.values->data()));
    return out;
}

}  // namespace sd
