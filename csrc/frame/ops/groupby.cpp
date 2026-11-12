#include "frame/ops/groupby.hpp"

#include "frame/ops/hash.hpp"
#include "frame/ops/pairwise.hpp"
#include "frame/ops/radix.hpp"
#include "frame/ops/reduce.hpp"
#include "frame/ops/selection.hpp"
#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>

namespace nb = nanobind;

namespace sd::frame {

Groups::Groups(const Column& codes, std::int64_t ngroups) : ngroups_(ngroups) {
    if (codes.dtype != DType::Int64 || codes.null_count) throw nb::type_error("group codes must be int64 without NA");
    counting_sort(codes.data<std::int64_t>(), codes.length, ngroups, start_, rows_);
}

// f(g_lo, g_hi) over the groups, cut into tasks of whole groups with about
// kChunkRows rows each. Each group is reduced on its own, so where the cuts
// fall changes nothing in the result.
template <class F>
void Groups::for_groups(F f) const {
    const std::int64_t n = static_cast<std::int64_t>(rows_.size());
    if (n < kParallelThreshold) {
        f(std::int64_t{0}, ngroups_);
        return;
    }
    std::vector<std::int64_t> cuts{0};
    for (std::int64_t r = kChunkRows; r < n; r += kChunkRows) {
        const auto g = static_cast<std::int64_t>(std::lower_bound(start_.begin(), start_.end() - 1, r) - start_.begin());
        if (g > cuts.back()) cuts.push_back(g);
    }
    if (ngroups_ > cuts.back()) cuts.push_back(ngroups_);
    parallel::parallel_for(static_cast<std::int64_t>(cuts.size()) - 1,
                           [&](std::int64_t t) { f(cuts[static_cast<std::size_t>(t)], cuts[static_cast<std::size_t>(t) + 1]); });
}

Column Groups::sizes() const {
    Column out = make_fixed(DType::Int64, ngroups_);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    for (std::int64_t g = 0; g < ngroups_; ++g) o[g] = start_[g + 1] - start_[g];
    return out;
}

namespace {

double row_double(const Column& c, std::int64_t i) {
    switch (c.dtype) {
    case DType::Int64: return static_cast<double>(c.data<std::int64_t>()[i]);
    case DType::Float64: return c.data<double>()[i];
    default: return static_cast<double>(c.data<std::uint8_t>()[i]);
    }
}

// Whether row a beats row b for min (Max false) or max (Max true); both have
// values. Floats follow IEEE 754-2019 minimum/maximum like the column
// reductions (docs/design/frame.md §7.4): NaN wins, -0.0 < 0.0.
template <bool Max>
bool beats(const Column& c, std::int64_t a, std::int64_t b) {
    switch (c.dtype) {
    case DType::Int64: return Max ? c.data<std::int64_t>()[a] > c.data<std::int64_t>()[b] : c.data<std::int64_t>()[a] < c.data<std::int64_t>()[b];
    case DType::Bool: return Max ? c.data<std::uint8_t>()[a] > c.data<std::uint8_t>()[b] : c.data<std::uint8_t>()[a] < c.data<std::uint8_t>()[b];
    case DType::String: return Max ? c.string_at(a) > c.string_at(b) : c.string_at(a) < c.string_at(b);
    case DType::Float64: {
        const double x = c.data<double>()[a], y = c.data<double>()[b];
        if (std::isnan(y)) return false;  // the first NaN stays
        if (std::isnan(x)) return true;
        const double m = ieee_minmax<Max>(x, y);
        return std::memcmp(&m, &x, sizeof m) == 0 && std::memcmp(&x, &y, sizeof x) != 0;
    }
    case DType::Complex128: break;  // not a frame dtype
    }
    return false;
}

}  // namespace

Column Groups::reduce(std::string_view op, const Column& col, bool skipna, double correction) const {
    const DType result = reduction_dtype(op, col.dtype);
    const bool is_var = op == "var" || op == "std";
    const std::int64_t G = ngroups_;
    auto count_of = [&](std::int64_t g) {
        std::int64_t k = 0;
        for (std::int64_t j = start_[g]; j < start_[g + 1]; ++j) k += col.is_valid(rows_[j]);
        return k;
    };

    // min, max, first, last pick a row per group (-1: none); take() then
    // brings its value, of any dtype, NA when there is none.
    if (op == "min" || op == "max" || op == "first" || op == "last") {
        Column winners = make_fixed(DType::Int64, G);
        auto* w = reinterpret_cast<std::int64_t*>(winners.values->data());
        for_groups([&](std::int64_t g_lo, std::int64_t g_hi) {
        for (std::int64_t g = g_lo; g < g_hi; ++g) {
            const std::int64_t b = start_[g], e = start_[g + 1];
            std::int64_t at = -1;
            if (op == "first" || op == "last") {
                const bool fwd = op == "first";
                for (std::int64_t k = 0; k < e - b; ++k) {
                    const std::int64_t r = rows_[fwd ? b + k : e - 1 - k];
                    if (!skipna || col.is_valid(r)) {
                        at = col.is_valid(r) ? r : -1;  // skipna=False: the first row, NA or not
                        break;
                    }
                }
            } else {
                bool any_na = false;
                for (std::int64_t j = b; j < e; ++j) {
                    const std::int64_t r = rows_[j];
                    if (!col.is_valid(r)) {
                        any_na = true;
                        continue;
                    }
                    if (at < 0 || (op == "max" ? beats<true>(col, r, at) : beats<false>(col, r, at))) at = r;
                }
                if (!skipna && any_na) at = -1;
            }
            w[g] = at;
        }
        });
        return take(col, winners);
    }

    Column out = make_fixed(result, G);
    // NA per group as a byte (tasks may meet inside a bitmap byte), packed
    // into the bitmap at the end.
    std::vector<std::uint8_t> valid(static_cast<std::size_t>(G), 1);
    for_groups([&](std::int64_t g_lo, std::int64_t g_hi) {
    for (std::int64_t g = g_lo; g < g_hi; ++g) {
        const std::int64_t b = start_[g], n = start_[g + 1] - b;
        const std::int64_t count = count_of(g);
        auto set_na = [&] { valid[static_cast<std::size_t>(g)] = 0; };
        auto get = [&](std::int64_t k) { return row_double(col, rows_[b + k]); };  // NA rows hold 0
        if (op == "count") {
            reinterpret_cast<std::int64_t*>(out.values->data())[g] = count;
            continue;
        }
        if (!skipna && count < n) {
            set_na();
            continue;
        }
        if (op == "sum") {
            if (result == DType::Float64) {
                reinterpret_cast<double*>(out.values->data())[g] = float_sum(n, get);
            } else {
                std::uint64_t total = 0;  // int64 wraps (docs/api/frame.md §1.5); bool counts True
                for (std::int64_t k = 0; k < n; ++k) {
                    const std::int64_t r = rows_[b + k];
                    total += col.dtype == DType::Int64 ? static_cast<std::uint64_t>(col.data<std::int64_t>()[r])
                                                       : col.data<std::uint8_t>()[r];
                }
                reinterpret_cast<std::int64_t*>(out.values->data())[g] = static_cast<std::int64_t>(total);
            }
            continue;
        }
        if (count == 0 || (is_var && static_cast<double>(count) - correction <= 0)) {
            set_na();
            continue;
        }
        double mean;
        if (col.dtype == DType::Int64) {  // exact, as the column mean (docs/api/frame.md §6.3)
            __int128 total = 0;
            for (std::int64_t k = 0; k < n; ++k) total += col.data<std::int64_t>()[rows_[b + k]];
            mean = static_cast<double>(total) / static_cast<double>(count);
        } else if (col.dtype == DType::Bool) {
            std::int64_t trues = 0;
            for (std::int64_t k = 0; k < n; ++k) trues += col.data<std::uint8_t>()[rows_[b + k]];
            mean = static_cast<double>(trues) / static_cast<double>(count);
        } else {
            mean = float_sum(n, get) / static_cast<double>(count);
        }
        double value = mean;
        if (is_var) {
            const double ssd = float_sum(n, [&](std::int64_t k) {
                const std::int64_t r = rows_[b + k];
                if (!col.is_valid(r)) return 0.0;
                const double d = row_double(col, r) - mean;
                return d * d;
            });
            value = ssd / (static_cast<double>(count) - correction);
            if (op == "std") value = std::sqrt(value);
        }
        reinterpret_cast<double*>(out.values->data())[g] = value;
    }
    });
    std::uint8_t* bits = add_validity(out);
    for (std::int64_t g = 0; g < G; ++g)
        if (!valid[static_cast<std::size_t>(g)]) bitmap::clear(bits, g);
    finish_fixed(out);
    return out;
}

Column sort_groups(const std::vector<Column>& keys) {
    const std::int64_t n = keys.empty() ? 0 : keys[0].length;
    // Sort the distinct values of each key, not the groups: give every group
    // the rank of its key value (0, 1, ... in key order), then order the
    // groups by rank with a stable counting sort, last key first (an LSD
    // radix sort). The distinct values are ordered by order_rows (a radix
    // sort; docs/design/frame.md §9.2).
    std::vector<std::int64_t> order(static_cast<std::size_t>(n)), next(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), std::int64_t{0});
    for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
        const Column& k = *it;
        Factorized f = factorize(k);  // dense codes over the groups; equal keys share a code
        const auto* code = f.codes.data<std::int64_t>();
        const auto* first = f.first.data<std::int64_t>();
        // The distinct values in order (radix sort): values first, then
        // the NA code last (docs/api/frame.md §7.2).
        std::vector<std::int64_t> valued, na_code;
        for (std::int64_t g = 0; g < f.ngroups; ++g) (k.is_valid(first[g]) ? valued : na_code).push_back(first[g]);
        order_rows(k, valued);
        std::vector<std::int64_t> rank(static_cast<std::size_t>(f.ngroups));
        std::int64_t r = 0;
        for (std::int64_t row : valued) rank[code[row]] = r++;
        for (std::int64_t row : na_code) rank[code[row]] = r++;
        // Stable sort of `order` by rank (keeps the order of earlier passes).
        std::vector<std::int64_t> key(static_cast<std::size_t>(n)), start, perm;
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t j = lo; j < hi; ++j) key[j] = rank[code[order[j]]];
        });
        counting_sort(key.data(), n, f.ngroups, start, perm);
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t j = lo; j < hi; ++j) next[j] = order[perm[j]];
        });
        order.swap(next);
    }
    Column out = make_fixed(DType::Int64, n);
    std::copy(order.begin(), order.end(), reinterpret_cast<std::int64_t*>(out.values->data()));
    return out;
}

}  // namespace sd::frame
