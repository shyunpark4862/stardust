#include "ops/sort.hpp"

#include "ops/hash.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>

namespace nb = nanobind;

namespace sd {

namespace {

// An unsigned 64-bit key whose order is the value order (API.md §17).
// int64: flip the sign bit, so negatives come first. float64: -0.0 becomes
// 0.0 and every NaN one NaN (both above +inf once mapped); then a positive
// float gets its sign bit set and a negative one all bits flipped -- the
// classic map from IEEE order to unsigned integer order.
std::uint64_t radix_key(const Column& c, std::int64_t i) {
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

// Stable LSD radix sort of `rows` by 64-bit keys: eight passes of one byte,
// least significant first, all eight histograms counted in one pre-pass.
// Every pass runs, whatever the data (no skipping of constant digits).
void radix_sort(std::vector<std::uint64_t>& keys, std::vector<std::int64_t>& rows) {
    const std::size_t n = keys.size();
    std::vector<std::array<std::int64_t, 256>> count(8);
    for (auto& h : count) h.fill(0);
    for (std::uint64_t k : keys)
        for (int d = 0; d < 8; ++d) ++count[d][(k >> (8 * d)) & 0xff];
    std::vector<std::uint64_t> keys2(n);
    std::vector<std::int64_t> rows2(n);
    for (int d = 0; d < 8; ++d) {
        std::array<std::int64_t, 256> start{};
        for (int b = 1; b < 256; ++b) start[b] = start[b - 1] + count[d][b - 1];
        for (std::size_t j = 0; j < n; ++j) {
            const auto b = (keys[j] >> (8 * d)) & 0xff;
            const std::int64_t to = start[b]++;
            keys2[to] = keys[j];
            rows2[to] = rows[j];
        }
        keys.swap(keys2);
        rows.swap(rows2);
    }
}

// Stable sort of `order` (positions into `key`) by `key`.
void sort_by(const Column& key, bool ascending, bool na_last, std::vector<std::int64_t>& order) {
    const std::size_t n = order.size();
    std::vector<std::int64_t> valid_rows, na_rows;
    valid_rows.reserve(n);
    for (std::int64_t r : order) (key.is_valid(r) ? valid_rows : na_rows).push_back(r);

    std::vector<std::uint64_t> keys(valid_rows.size());
    if (key.dtype == DType::String) {
        // Ranks of the distinct strings (code point order) stand in for the
        // strings: only the distinct values meet a comparison sort.
        Factorized f = factorize(key);
        const auto* code = f.codes.data<std::int64_t>();
        const auto* first = f.first.data<std::int64_t>();
        std::vector<std::int64_t> by_value(static_cast<std::size_t>(f.ngroups));
        std::iota(by_value.begin(), by_value.end(), std::int64_t{0});
        std::sort(by_value.begin(), by_value.end(), [&](std::int64_t a, std::int64_t b) {
            const bool va = key.is_valid(first[a]), vb = key.is_valid(first[b]);
            if (!va || !vb) return va && !vb;  // the NA code (if any) goes last; NA rows are apart anyway
            return key.string_at(first[a]) < key.string_at(first[b]);
        });
        std::vector<std::uint64_t> rank(static_cast<std::size_t>(f.ngroups));
        for (std::size_t r = 0; r < by_value.size(); ++r) rank[by_value[r]] = r;
        for (std::size_t j = 0; j < valid_rows.size(); ++j) keys[j] = rank[code[valid_rows[j]]];
    } else {
        for (std::size_t j = 0; j < valid_rows.size(); ++j) keys[j] = radix_key(key, valid_rows[j]);
    }
    if (!ascending)  // the reverse order; equal keys stay equal, so ties keep their order
        for (auto& k : keys) k = ~k;
    radix_sort(keys, valid_rows);

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
