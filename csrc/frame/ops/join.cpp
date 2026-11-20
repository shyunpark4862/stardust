#include "frame/ops/join.hpp"

#include "frame/ops/hash.hpp"
#include "frame/ops/radix.hpp"
#include "frame/ops/selection.hpp"
#include "frame/parallel.hpp"

#include <nanobind/nanobind.h>

#include <atomic>
#include <string>

namespace nb = nanobind;

namespace sd::frame {

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

    // Count per chunk of probe rows, then each chunk fills its own part.
    // probed: codes some probe row has (several rows may set one flag).
    std::vector<std::uint8_t> probed(static_cast<std::size_t>(G), 0);
    auto matches = [&](std::int64_t p) { return pcode[p] < 0 ? 0 : start[pcode[p] + 1] - start[pcode[p]]; };
    const std::size_t CP = static_cast<std::size_t>(num_chunks(np));
    std::vector<std::int64_t> pstart(CP + 1, 0);
    for_chunks(np, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t t = 0;
        for (std::int64_t p = lo; p < hi; ++p) {
            const std::int64_t m = matches(p);
            if (keep_build && pcode[p] >= 0)
                std::atomic_ref<std::uint8_t>(probed[pcode[p]]).store(1, std::memory_order_relaxed);
            t += m > 0 ? m : (keep_probe ? 1 : 0);
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
        for (std::int64_t p = lo; p < hi; ++p) {
            const std::int64_t b0 = pcode[p] < 0 ? 0 : start[pcode[p]], b1 = pcode[p] < 0 ? 0 : start[pcode[p] + 1];
            if (b1 > b0) {
                for (std::int64_t j = b0; j < b1; ++j, ++k) {
                    po[k] = p;
                    bo[k] = rows[j];
                }
            } else if (keep_probe) {
                po[k] = p;
                bo[k] = -1;
                ++k;
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

}  // namespace sd::frame
