#include "frame/ops/groupby.hpp"

#include "base/int_sum.hpp"

#include "frame/ops/hash.hpp"
#include "frame/ops/pairwise.hpp"
#include "frame/ops/radix.hpp"
#include "frame/ops/reduce.hpp"
#include "frame/ops/selection.hpp"
#include "frame/ops/sort.hpp"
#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <type_traits>

namespace nb = nanobind;

namespace sd::frame {

Groups::Groups(const Column& codes, std::int64_t ngroups) : ngroups_(ngroups) {
    if (codes.dtype != DType::Int64 || codes.null_count) throw nb::type_error("group codes must be int64 without NA");
    const std::int64_t n = codes.length;
    const std::int64_t* c = codes.data<std::int64_t>();
    // Codes in order of first appearance that never go down (0, 0, 1, 1, 1,
    // 2, ...: keys already grouped, as in a column sorted by key): every
    // group is a run of rows, found by its boundaries, and no row moves. A
    // counting sort of such codes was the slowest case (each count and
    // each place waits for the one before: 3 ms for 1M sorted rows).
    std::atomic<bool> runs{true};
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        bool ok = true;
        for (std::int64_t i = std::max<std::int64_t>(lo, 1); i < hi; ++i)
            ok &= c[i] - c[i - 1] <= 1 && c[i] >= c[i - 1];
        if (!ok) runs.store(false, std::memory_order_relaxed);
    });
    if (runs.load() && (n == 0 || c[0] == 0)) {
        contiguous_ = true;
        start_.assign(static_cast<std::size_t>(ngroups) + 1, n);
        start_[0] = 0;
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t i = std::max<std::int64_t>(lo, 1); i < hi; ++i)
                if (c[i] != c[i - 1]) start_[static_cast<std::size_t>(c[i])] = i;
        });
        return;
    }
    counting_sort(c, n, ngroups, start_, rows_);
}

// f(g_lo, g_hi) over the groups, cut into tasks of whole groups with about
// kChunkRows rows each. Each group is reduced on its own, so where the cuts
// fall changes nothing in the result.
template <class F>
void Groups::for_groups(F f) const {
    const std::int64_t n = start_.back();
    if (n < kParallelThreshold) {
        f(std::int64_t{0}, ngroups_);
        return;
    }
    std::vector<std::int64_t> cuts{0};
    for (std::int64_t r = kChunkRows; r < n; r += kChunkRows) {
        const auto g =
            static_cast<std::int64_t>(std::lower_bound(start_.begin(), start_.end() - 1, r) - start_.begin());
        if (g > cuts.back()) cuts.push_back(g);
    }
    if (ngroups_ > cuts.back()) cuts.push_back(ngroups_);
    run_unlocked(
        static_cast<std::int64_t>(cuts.size()) - 1,
        [&](std::int64_t t) { f(cuts[static_cast<std::size_t>(t)], cuts[static_cast<std::size_t>(t) + 1]); },
        4 * n  // rows read through the layout: a few times a plain pass
    );
}

Column Groups::sizes() const {
    Column out = make_fixed(DType::Int64, ngroups_);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    for (std::int64_t g = 0; g < ngroups_; ++g) o[g] = start_[g + 1] - start_[g];
    return out;
}


namespace {

// Row j of the layout: rows_[j], or j itself when the codes were already in
// order (Groups::contiguous_: every group a run of rows).
struct Indirect {
    const std::int64_t* rows;
    std::int64_t operator()(std::int64_t j) const { return rows[j]; }
};
struct Identity {
    std::int64_t operator()(std::int64_t j) const { return j; }
};

// The winner of a group for min (Max false) or max: values `v`, rows
// row(b) .. row(e - 1), validity `valid(r)`. Ties keep the first row; for
// floats IEEE 754-2019 minimum/maximum (NaN wins, the first NaN stays;
// -0.0 < 0.0), as the column reductions (docs/design/frame.md §7.4). -1 for
// a group without a value, or with an NA when !skipna.
template <bool Max, class T, class Row, class Valid>
std::int64_t winner(const T* v, std::int64_t b, std::int64_t e, Row row, Valid valid, bool skipna) {
    std::int64_t at = -1;
    T best{};
    bool any_na = false;
    for (std::int64_t j = b; j < e; ++j) {
        const std::int64_t r = row(j);
        if (!valid(r)) {
            any_na = true;
            continue;
        }
        const T x = v[r];
        bool take;
        if (at < 0) {
            take = true;
        } else if constexpr (std::is_same_v<T, double>) {
            if (std::isnan(best)) take = false;  // the first NaN stays
            else if (std::isnan(x)) take = true;
            else {
                const double m = ieee_minmax<Max>(x, best);
                take = std::memcmp(&m, &x, sizeof m) == 0 && std::memcmp(&x, &best, sizeof x) != 0;
            }
        } else {
            take = Max ? x > best : x < best;
        }
        if (take) {
            at = r;
            best = x;
        }
    }
    return !skipna && any_na ? -1 : at;
}

}  // namespace

Column Groups::reduce(std::string_view op, const Column& col, bool skipna, double correction) const {
    if (contiguous_) return reduce_with(op, col, skipna, correction, Identity{});
    return reduce_with(op, col, skipna, correction, Indirect{rows_.data()});
}

std::vector<Column> Groups::reduce_columns(
    std::string_view op, const std::vector<Column>& cols, bool skipna, double correction
) const {
    for (const Column& c : cols) reduction_dtype(op, c.dtype);  // the first column that cannot raises, in order
    std::vector<Column> out(cols.size());
    const std::int64_t n = start_.back(), m = static_cast<std::int64_t>(cols.size());
    if (m < 2 || n >= kParallelThreshold || n * m < kParallelThreshold || !parallel::available()) {
        for (std::size_t j = 0; j < cols.size(); ++j) out[j] = reduce(op, cols[j], skipna, correction);
        return out;
    }
    run_unlocked(
        m,
        [&](std::int64_t j) {
            out[static_cast<std::size_t>(j)] = reduce(op, cols[static_cast<std::size_t>(j)], skipna, correction);
        },
        n * m
    );
    return out;
}

template <class Row>
Column Groups::reduce_with(std::string_view op, const Column& col, bool skipna, double correction, Row row) const {
    const DType result = reduction_dtype(op, col.dtype);
    const bool is_var = op == "var" || op == "std";
    const std::int64_t G = ngroups_;
    const std::int64_t* start = start_.data();
    const std::uint8_t* bits = col.validity_bits();
    const std::int64_t off = col.offset;
    const bool nulls = col.null_count > 0;
    auto valid = [bits, off, nulls](std::int64_t r) { return !nulls || bitmap::get(bits, off + r); };
    auto count_of = [&](std::int64_t g) {
        if (!nulls) return start[g + 1] - start[g];
        std::int64_t k = 0;
        for (std::int64_t j = start[g]; j < start[g + 1]; ++j) k += valid(row(j));
        return k;
    };

    // min, max, first, last pick a row per group (-1: none); take() then
    // brings its value, of any dtype, NA when there is none.
    if (op == "min" || op == "max" || op == "first" || op == "last") {
        Column winners = make_fixed(DType::Int64, G);
        auto* w = reinterpret_cast<std::int64_t*>(winners.values->data());
        const bool first_last = op == "first" || op == "last", is_max = op == "max";
        for_groups([&](std::int64_t g_lo, std::int64_t g_hi) {
            for (std::int64_t g = g_lo; g < g_hi; ++g) {
                const std::int64_t b = start[g], e = start[g + 1];
                std::int64_t at = -1;
                if (first_last) {
                    const bool fwd = op == "first";
                    for (std::int64_t k = 0; k < e - b; ++k) {
                        const std::int64_t r = row(fwd ? b + k : e - 1 - k);
                        if (!skipna || valid(r)) {
                            at = valid(r) ? r : -1;  // skipna=False: the first row, NA or not
                            break;
                        }
                    }
                } else {
                    switch (col.dtype) {
                        case DType::Float64:
                            at = is_max ? winner<true>(col.data<double>(), b, e, row, valid, skipna)
                                        : winner<false>(col.data<double>(), b, e, row, valid, skipna);
                            break;
                        case DType::Int64:
                            at = is_max ? winner<true>(col.data<std::int64_t>(), b, e, row, valid, skipna)
                                        : winner<false>(col.data<std::int64_t>(), b, e, row, valid, skipna);
                            break;
                        case DType::Bool:
                            at = is_max ? winner<true>(col.data<std::uint8_t>(), b, e, row, valid, skipna)
                                        : winner<false>(col.data<std::uint8_t>(), b, e, row, valid, skipna);
                            break;
                        default: {  // strings: by code point
                            bool any_na = false;
                            for (std::int64_t j = b; j < e; ++j) {
                                const std::int64_t r = row(j);
                                if (!valid(r)) {
                                    any_na = true;
                                    continue;
                                }
                                if (at < 0 || (is_max ? col.string_at(r) > col.string_at(at)
                                                      : col.string_at(r) < col.string_at(at)))
                                    at = r;
                            }
                            if (!skipna && any_na) at = -1;
                        }
                    }
                }
                w[g] = at;
            }
        });
        return take(col, winners);
    }

    Column out = make_fixed(result, G);
    // NA per group as a byte (tasks may meet inside a bitmap byte), packed
    // into the bitmap at the end.
    std::vector<std::uint8_t> na(static_cast<std::size_t>(G), 0);
    auto* od = reinterpret_cast<double*>(out.values->data());
    auto* oi = reinterpret_cast<std::int64_t*>(out.values->data());
    const double* vf = col.dtype == DType::Float64 ? col.data<double>() : nullptr;
    const std::int64_t* vi = col.dtype == DType::Int64 ? col.data<std::int64_t>() : nullptr;
    const std::uint8_t* vb = col.dtype == DType::Bool ? col.data<std::uint8_t>() : nullptr;
    const bool is_count = op == "count", is_sum = op == "sum";
    for_groups([&](std::int64_t g_lo, std::int64_t g_hi) {
        for (std::int64_t g = g_lo; g < g_hi; ++g) {
            const std::int64_t b = start[g], n = start[g + 1] - b;
            const std::int64_t count = count_of(g);
            if (is_count) {
                oi[g] = count;
                continue;
            }
            if (!skipna && count < n) {
                na[static_cast<std::size_t>(g)] = 1;
                continue;
            }
            if (is_sum) {
                if (vf) {  // NA rows hold 0: the sum of the group's rows, as the column sum
                    od[g] = float_sum(n, [vf, b, row](std::int64_t k) { return vf[row(b + k)]; });
                } else {
                    std::uint64_t total = 0;  // int64 wraps (docs/api/frame.md §1.5); bool counts True
                    if (vi)
                        for (std::int64_t k = 0; k < n; ++k) total += static_cast<std::uint64_t>(vi[row(b + k)]);
                    else
                        for (std::int64_t k = 0; k < n; ++k) total += vb[row(b + k)];
                    oi[g] = static_cast<std::int64_t>(total);
                }
                continue;
            }
            if (count == 0 || (is_var && static_cast<double>(count) - correction <= 0)) {
                na[static_cast<std::size_t>(g)] = 1;
                continue;
            }
            double mean;
            if (vb) {
                std::int64_t trues = 0;
                for (std::int64_t k = 0; k < n; ++k) trues += vb[row(b + k)];
                mean = static_cast<double>(trues) / static_cast<double>(count);
            } else if (vi) {  // the exact sum, as the column mean (base/int_sum.hpp)
                IntSum s;
                for (std::int64_t k = 0; k < n; ++k) s.add(vi[row(b + k)]);  // NA rows hold 0
                mean = sd::mean_of(s, count);
            } else {
                mean =
                    float_sum(n, [vf, b, row](std::int64_t k) { return vf[row(b + k)]; }) / static_cast<double>(count);
            }
            double value = mean;
            if (is_var) {  // one fma per square, as the column var
                auto x = [&](std::int64_t r) {
                    return vf ? vf[r] : vi ? static_cast<double>(vi[r]) : static_cast<double>(vb[r]);
                };
                const double ssd = nulls ? squares_sum(
                                               n,
                                               [&](std::int64_t k) {
                                                   const std::int64_t r = row(b + k);
                                                   return valid(r) ? x(r) - mean : 0.0;
                                               }
                                           )
                                         : squares_sum(n, [&](std::int64_t k) { return x(row(b + k)) - mean; });
                value = ssd / (static_cast<double>(count) - correction);
                if (op == "std") value = std::sqrt(value);
            }
            od[g] = value;
        }
    });
    std::int64_t nna = 0;
    for (std::uint8_t x : na) nna += x;
    if (nna > 0) {
        std::uint8_t* obits = add_validity(out);
        for (std::int64_t g = 0; g < G; ++g)
            if (na[static_cast<std::size_t>(g)]) bitmap::clear(obits, g);
        finish_fixed(out);
    }
    return out;
}

Column sort_groups(const std::vector<Column>& keys) {
    const std::int64_t n = keys.empty() ? 0 : keys[0].length;
    // One number or bool key: the stable sort of the rows by value, NA last,
    // is the order (equal values -- -0.0 and 0.0, NaNs -- keep their order,
    // as the ranks below would): one radix sort instead of factorize, ranks
    // and a counting sort.
    if (keys.size() == 1 && keys[0].dtype != DType::String) {
        if (std::optional<Column> order = argsort(keys, {true}, {true})) return *order;
        return arange(0, 1, n);
    }
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
        {
            const Unlocked unlocked(f.ngroups);
            for (std::int64_t g = 0; g < f.ngroups; ++g) (k.is_valid(first[g]) ? valued : na_code).push_back(first[g]);
        }
        order_rows(k, valued);
        std::vector<std::int64_t> rank(static_cast<std::size_t>(f.ngroups));
        {
            const Unlocked unlocked(f.ngroups);
            std::int64_t r = 0;
            for (std::int64_t row : valued) rank[code[row]] = r++;
            for (std::int64_t row : na_code) rank[code[row]] = r++;
        }
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
