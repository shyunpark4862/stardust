#include "frame/ops/join.hpp"

#include "frame/ops/hash.hpp"
#include "frame/ops/radix.hpp"
#include "frame/ops/selection.hpp"
#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <atomic>
#include <optional>
#include <string>

namespace nb = nanobind;

namespace sd::frame {

namespace {

// One integer key whose right values span a small range and never repeat
// (each key one right row), and an inner, left or outer join: the right
// rows sit in a direct table by key, and one pass over the left keys reads
// each row's match -- no codes, no grouping of the right rows. The result
// is exactly the general path's: left order, at most one match per left
// row, outer's unmatched right rows after them in right order. nullopt when
// the keys do not qualify (the general path then runs).
std::optional<std::pair<Column, Column>> unique_direct_join(
    const Column& left, const Column& right, std::string_view how
) {
    if (how == "right") return std::nullopt;
    const std::optional<DirectRange> r = direct_range(right);
    if (!r) return std::nullopt;
    const std::int64_t nl = left.length, nr = right.length, lo = r->lo;
    const auto span = static_cast<std::uint64_t>(r->span);
    std::vector<std::int64_t> table(r->span + 1, 0);  // 1 + the right row of key lo + k; slot `span`: no row
    {
        const Unlocked unlocked(nr);
        auto value = [&right](std::int64_t i) {
            return right.dtype == DType::Bool ? std::int64_t{right.data<std::uint8_t>()[i]}
                                              : right.data<std::int64_t>()[i];
        };
        for (std::int64_t i = 0; i < nr; ++i) {
            if (!right.is_valid(i)) continue;  // NA matches nothing
            std::int64_t& slot = table[static_cast<std::size_t>(value(i) - lo)];
            if (slot != 0) return std::nullopt;  // a repeated right key
            slot = i + 1;
        }
    }
    const bool inner = how == "inner", outer = how == "outer";
    const std::int64_t* t = table.data();
    const std::uint8_t* bits = left.validity_bits();
    const std::int64_t off = left.offset;
    // The right row of left row i, or -1 (one read, no branch: a key out of
    // range or NA reads the empty last slot).
    auto match = [&](auto value) {
        return [=](std::int64_t i) {
            const std::uint64_t k = static_cast<std::uint64_t>(value(i)) - static_cast<std::uint64_t>(lo);
            const bool in = k < span && (!bits || bitmap::get(bits, off + i));
            return t[in ? k : span] - 1;
        };
    };
    auto run = [&](auto row_of) -> std::pair<Column, Column> {
        std::vector<std::uint8_t> probed(outer ? static_cast<std::size_t>(nr) : 0, 0);
        std::vector<std::int64_t> matched;  // inner: per left row, its right row or -1
        const std::size_t C = static_cast<std::size_t>(num_chunks(nl));
        std::vector<std::int64_t> at(C + 1, 0);
        Column lout, rout;
        if (inner) {
            matched.resize(static_cast<std::size_t>(nl));
            for_chunks(
                nl,
                [&](std::int64_t a, std::int64_t b) {
                    std::int64_t k = 0;
                    for (std::int64_t i = a; i < b; ++i) {
                        const std::int64_t m = row_of(i);
                        matched[static_cast<std::size_t>(i)] = m;
                        k += m >= 0;
                    }
                    at[static_cast<std::size_t>(a / kChunkRows) + 1] = k;
                },
                2
            );
            for (std::size_t c = 0; c < C; ++c) at[c + 1] += at[c];
            const std::int64_t total = at[C];
            lout = make_fixed(DType::Int64, total);
            rout = make_fixed(DType::Int64, total);
            auto* lo_ = reinterpret_cast<std::int64_t*>(lout.values->data());
            auto* ro = reinterpret_cast<std::int64_t*>(rout.values->data());
            for_chunks(nl, [&](std::int64_t a, std::int64_t b) {
                std::int64_t k = at[static_cast<std::size_t>(a / kChunkRows)];
                std::int64_t spare[2];  // where a row without a match writes: not the next chunk's first slot
                for (std::int64_t i = a; i < b; ++i) {
                    const std::int64_t m = matched[static_cast<std::size_t>(i)];
                    std::int64_t* l = m >= 0 ? lo_ + k : spare;
                    std::int64_t* r = m >= 0 ? ro + k : spare + 1;
                    *l = i;
                    *r = m;
                    k += m >= 0;
                }
            });
            return {lout, rout};
        }
        // left and outer: every left row once, in order; outer then the right
        // rows no left row matched, in right order.
        std::vector<std::int64_t> extra_at;
        std::int64_t extra = 0;
        const std::size_t CB = static_cast<std::size_t>(num_chunks(nr));
        auto* pr = probed.data();
        std::vector<std::int64_t> right_rows(static_cast<std::size_t>(nl));
        std::int64_t* rr = right_rows.data();
        for_chunks(
            nl,
            [&](std::int64_t a, std::int64_t b) {
                for (std::int64_t i = a; i < b; ++i) {
                    const std::int64_t m = row_of(i);
                    rr[i] = m;
                    if (outer && m >= 0) std::atomic_ref<std::uint8_t>(pr[m]).store(1, std::memory_order_relaxed);
                }
            },
            2
        );
        if (outer) {
            extra_at.assign(CB + 1, 0);
            for_chunks(nr, [&](std::int64_t a, std::int64_t b) {
                std::int64_t k = 0;
                for (std::int64_t i = a; i < b; ++i) k += !pr[i];
                extra_at[static_cast<std::size_t>(a / kChunkRows) + 1] = k;
            });
            for (std::size_t c = 0; c < CB; ++c) extra_at[c + 1] += extra_at[c];
            extra = extra_at[CB];
        }
        const std::int64_t total = nl + extra;
        lout = make_fixed(DType::Int64, total);
        rout = make_fixed(DType::Int64, total);
        auto* lo_ = reinterpret_cast<std::int64_t*>(lout.values->data());
        auto* ro = reinterpret_cast<std::int64_t*>(rout.values->data());
        for_chunks(nl, [&](std::int64_t a, std::int64_t b) {
            for (std::int64_t i = a; i < b; ++i) {
                lo_[i] = i;
                ro[i] = rr[i];
            }
        });
        if (outer)
            for_chunks(nr, [&](std::int64_t a, std::int64_t b) {
                std::int64_t k = nl + extra_at[static_cast<std::size_t>(a / kChunkRows)];
                for (std::int64_t i = a; i < b; ++i)
                    if (!pr[i]) {
                        lo_[k] = -1;
                        ro[k++] = i;
                    }
            });
        return {lout, rout};
    };
    if (left.dtype == DType::Bool)
        return run(match([v = left.data<std::uint8_t>()](std::int64_t i) { return std::int64_t{v[i]}; }));
    return run(match([v = left.data<std::int64_t>()](std::int64_t i) { return v[i]; }));
}

// Several integer keys whose right values span small ranges with a small
// product: one int64 key per row, the offsets of the keys in mixed radix
// (x0 - lo0) * span1 * ... + (x1 - lo1) * ... A left row with any key
// outside the right ranges matches nothing (-1); a row with any key NA is
// NA. The same matches as the keys taken together. nullopt when the keys do
// not qualify.
std::optional<std::pair<Column, Column>> composite_keys(
    const std::vector<Column>& left, const std::vector<Column>& right
) {
    std::vector<DirectRange> ranges;
    std::int64_t total = 1;
    const std::int64_t limit = direct_span_limit(right[0].length);
    for (const Column& c : right) {
        const std::optional<DirectRange> r = direct_range(c);
        if (!r || r->span > limit / total) return std::nullopt;
        ranges.push_back(*r);
        total *= r->span;
    }
    auto combine = [&](const std::vector<Column>& cols) {
        const std::int64_t n = cols[0].length;
        Column out = make_fixed(DType::Int64, n);
        auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
        bool any_na = false;
        for (const Column& c : cols) any_na |= c.null_count > 0;
        std::uint8_t* bits = any_na ? add_validity(out) : nullptr;
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {  // chunks are whole bitmap bytes
            for (std::int64_t i = lo; i < hi; ++i) {
                std::int64_t acc = 0;
                bool in = true, valid = true;
                for (std::size_t j = 0; j < cols.size(); ++j) {
                    const Column& c = cols[j];
                    const std::int64_t v =
                        c.dtype == DType::Bool ? std::int64_t{c.data<std::uint8_t>()[i]} : c.data<std::int64_t>()[i];
                    const std::uint64_t k = static_cast<std::uint64_t>(v) - static_cast<std::uint64_t>(ranges[j].lo);
                    in &= k < static_cast<std::uint64_t>(ranges[j].span);
                    valid &= c.is_valid(i);
                    acc = acc * ranges[j].span + static_cast<std::int64_t>(k);
                }
                o[i] = valid ? (in ? acc : -1) : 0;
                if (!valid) bitmap::clear(bits, i);
            }
        });
        finish_fixed(out);
        return out;
    };
    return std::pair{combine(left), combine(right)};
}

}  // namespace

std::pair<Column, Column> join_indices(
    const std::vector<Column>& left, const std::vector<Column>& right, std::string_view how
) {
    if (how != "inner" && how != "left" && how != "right" && how != "outer")
        throw nb::value_error(("how must be inner, left, right or outer, got '" + std::string(how) + "'").c_str());
    if (left.empty() || left.size() != right.size()) throw nb::value_error("one right key per left key");
    for (std::size_t j = 0; j < left.size(); ++j)
        if (left[j].dtype != right[j].dtype)
            throw nb::type_error((std::string("join keys must have one dtype, got ") + dtype_name(left[j].dtype) +
                                  " and " + dtype_name(right[j].dtype))
                                     .c_str());

    // Build on the right keys, probe with the left ones, rather than
    // factorizing both sides together (docs/design/frame.md §9.7). With several keys, each column gets its own
    // codes and the tuple becomes one number, code = code * n_j + code_j;
    // when that could pass 2^62 the codes so far are compacted through a
    // table built on the right side, the left side mapped through the same
    // table. A -1 anywhere (NA, or a left value the right never has) means
    // the row matches nothing.
    if (left.size() == 1) {
        if (auto fast = unique_direct_join(left[0], right[0], how)) return *fast;
    } else if (auto keys = composite_keys(left, right)) {  // several small integer keys: one key
        return join_indices({keys->first}, {keys->second}, how);
    }
    const std::int64_t nl = left[0].length, nr = right[0].length;
    std::vector<std::int64_t> lc, rc;
    std::int64_t G = 0;
    auto compact = [&] {  // right codes -> dense, left codes through the same table
        auto as_column = [](const std::vector<std::int64_t>& v) {
            Column c = make_fixed(DType::Int64, static_cast<std::int64_t>(v.size()));
            std::uint8_t* bits = add_validity(c);
            const Unlocked unlocked(static_cast<std::int64_t>(v.size()));
            for (std::size_t i = 0; i < v.size(); ++i) {
                reinterpret_cast<std::int64_t*>(c.values->data())[i] = v[i] < 0 ? 0 : v[i];
                if (v[i] < 0) bitmap::clear(bits, static_cast<std::int64_t>(i));
            }
            finish_fixed(c);
            return c;
        };
        ValueCodes table(as_column(rc));
        lc = table.lookup(as_column(lc));
        rc = table.codes();
        G = table.ngroups();
    };
    for (std::size_t j = 0; j < left.size(); ++j) {
        ValueCodes table(right[j]);
        std::vector<std::int64_t> l = table.lookup(left[j]);
        const std::vector<std::int64_t>& r = table.codes();
        const std::int64_t nj = table.ngroups();
        if (j == 0) {
            lc = std::move(l);
            rc = r;
            G = nj;
            continue;
        }
        if (nj != 0 && G > (std::int64_t{1} << 62) / nj) compact();
        for_chunks(nl, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t i = lo; i < hi; ++i) lc[i] = (lc[i] < 0 || l[i] < 0) ? -1 : lc[i] * nj + l[i];
        });
        for_chunks(nr, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t i = lo; i < hi; ++i) rc[i] = (rc[i] < 0 || r[i] < 0) ? -1 : rc[i] * nj + r[i];
        });
        G *= nj;
    }
    if (left.size() > 1) compact();  // dense codes for the CSR below

    // "probe" rows lead the output in their order; "build" rows are found by
    // code. A right join is a left join with the sides swapped.
    const bool swap = how == "right";
    if (swap) {  // the left rows become the build side: give them dense codes of their own
        std::vector<std::int64_t> l_dense(lc.size(), -1), r_dense(rc.size(), -1);
        std::vector<std::int64_t> remap(static_cast<std::size_t>(G), -1);
        std::int64_t g2 = 0;
        Unlocked unlocked(nl + nr);  // serial: numbering in order of first appearance
        for (std::size_t i = 0; i < lc.size(); ++i)
            if (lc[i] >= 0) {
                if (remap[lc[i]] < 0) remap[lc[i]] = g2++;
                l_dense[i] = remap[lc[i]];
            }
        for (std::size_t i = 0; i < rc.size(); ++i)
            if (rc[i] >= 0) r_dense[i] = remap[rc[i]];
        lc.swap(r_dense);  // probe = right, build = left
        rc.swap(l_dense);
        G = g2;
    }
    const std::int64_t np = swap ? nr : nl, nb_ = swap ? nl : nr;
    const std::int64_t* pcode = lc.data();
    const std::int64_t* bcode = rc.data();
    const bool keep_probe = how != "inner";  // left, right (swapped), outer keep unmatched probe rows
    const bool keep_build = how == "outer";

    // Build rows grouped by code, in their order (counting sort, CSR); rows
    // with no code (-1) go to an extra last key and are left out.
    std::vector<std::int64_t> start, rows;
    {
        std::vector<std::int64_t> key(static_cast<std::size_t>(nb_));
        for_chunks(nb_, [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t i = lo; i < hi; ++i) key[i] = bcode[i] < 0 ? G : bcode[i];
        });
        counting_sort(key.data(), nb_, G + 1, start, rows);
    }

    // Per code c, at c + 1: where its build rows start in `rows` and how many
    // there are; at 0, for probe rows without a code: none. One read of 16
    // bytes per probe row, no branch on whether it has a code.
    struct Span {
        std::int64_t begin, count;
    };
    std::vector<Span> span(static_cast<std::size_t>(G) + 1);
    span[0] = {0, 0};
    for (std::int64_t c = 0; c < G; ++c)
        span[static_cast<std::size_t>(c) + 1] = {
            start[static_cast<std::size_t>(c)],
            start[static_cast<std::size_t>(c) + 1] - start[static_cast<std::size_t>(c)]
        };
    rows.push_back(-1);  // read (and dropped) for a probe row without a match
    const Span* sp = span.data() + 1;  // sp[c] for c >= -1
    const std::int64_t* brows = rows.data();

    // Count per chunk of probe rows, then each chunk fills its own part.
    // probed: codes some probe row has (several rows may set one flag).
    std::vector<std::uint8_t> probed(static_cast<std::size_t>(G), 0);
    const std::size_t CP = static_cast<std::size_t>(num_chunks(np));
    std::vector<std::int64_t> pstart(CP + 1, 0);
    for_chunks(np, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t t = 0;
        for (std::int64_t p = lo; p < hi; ++p) {
            const std::int64_t m = sp[pcode[p]].count;
            if (keep_build && pcode[p] >= 0)
                std::atomic_ref<std::uint8_t>(probed[pcode[p]]).store(1, std::memory_order_relaxed);
            t += keep_probe ? std::max<std::int64_t>(m, 1) : m;
        }
        pstart[lo / kChunkRows + 1] = t;
    });
    for (std::size_t c = 0; c < CP; ++c) pstart[c + 1] += pstart[c];
    auto unprobed = [&](std::int64_t i) { return bcode[i] < 0 || !probed[bcode[i]]; };
    const std::size_t CB = static_cast<std::size_t>(num_chunks(nb_));
    std::vector<std::int64_t> bstart(CB + 1, pstart[CP]);
    if (keep_build) {
        for_chunks(nb_, [&](std::int64_t lo, std::int64_t hi) {
            std::int64_t t = 0;
            for (std::int64_t i = lo; i < hi; ++i) t += unprobed(i);
            bstart[lo / kChunkRows + 1] = t;
        });
        for (std::size_t c = 0; c < CB; ++c) bstart[c + 1] += bstart[c];
    }
    const std::int64_t total = bstart[CB];
    Column pout = make_fixed(DType::Int64, total), bout = make_fixed(DType::Int64, total);
    auto* po = reinterpret_cast<std::int64_t*>(pout.values->data());
    auto* bo = reinterpret_cast<std::int64_t*>(bout.values->data());
    for_chunks(np, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t k = pstart[lo / kChunkRows];
        std::int64_t spare[2];  // where a dropped row writes: not the next chunk's first slot
        for (std::int64_t p = lo; p < hi; ++p) {
            const Span e = sp[pcode[p]];
            if (e.count <= 1) [[likely]] {  // no match or one: no loop
                const bool kept = keep_probe || e.count;
                *(kept ? po + k : spare) = p;
                *(kept ? bo + k : spare + 1) = brows[e.count ? e.begin : static_cast<std::int64_t>(rows.size()) - 1];
                k += kept;
                continue;
            }
            for (std::int64_t j = e.begin; j < e.begin + e.count; ++j, ++k) {
                po[k] = p;
                bo[k] = brows[j];
            }
        }
    });
    if (keep_build)
        for_chunks(nb_, [&](std::int64_t lo, std::int64_t hi) {
            std::int64_t k = bstart[lo / kChunkRows];
            for (std::int64_t i = lo; i < hi; ++i)
                if (unprobed(i)) {
                    po[k] = -1;
                    bo[k] = i;
                    ++k;
                }
        });
    return swap ? std::pair<Column, Column>{bout, pout} : std::pair<Column, Column>{pout, bout};
}

bool is_arange(const Column& positions) {
    const std::int64_t n = positions.length;
    if (positions.dtype != DType::Int64 || positions.null_count) return false;
    const std::int64_t* v = positions.data<std::int64_t>();
    std::atomic<bool> same{true};
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        bool ok = true;
        for (std::int64_t i = lo; i < hi; ++i) ok &= v[i] == i;
        if (!ok) same.store(false, std::memory_order_relaxed);
    });
    return same.load();
}

}  // namespace sd::frame
