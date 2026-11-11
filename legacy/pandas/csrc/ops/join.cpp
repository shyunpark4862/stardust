#include "ops/join.hpp"

#include "ops/hash.hpp"
#include "ops/selection.hpp"

#include <nanobind/nanobind.h>

#include <string>

namespace nb = nanobind;

namespace sd {

std::pair<Column, Column> join_indices(const std::vector<Column>& left, const std::vector<Column>& right,
                                       std::string_view how) {
    if (how != "inner" && how != "left" && how != "right" && how != "outer")
        throw nb::value_error(("how must be inner, left, right or outer, got '" + std::string(how) + "'").c_str());
    if (left.empty() || left.size() != right.size()) throw nb::value_error("one right key per left key");
    for (std::size_t j = 0; j < left.size(); ++j)
        if (left[j].dtype != right[j].dtype)
            throw nb::type_error((std::string("join keys must have one dtype, got ") + dtype_name(left[j].dtype) +
                                  " and " + dtype_name(right[j].dtype) + " (API.md §18)")
                                     .c_str());

    // Build on the right keys, probe with the left ones (M10; M8 factorized
    // both sides together). With several keys, each column gets its own
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
        for (std::int64_t i = 0; i < nl; ++i) lc[i] = (lc[i] < 0 || l[i] < 0) ? -1 : lc[i] * nj + l[i];
        for (std::int64_t i = 0; i < nr; ++i) rc[i] = (rc[i] < 0 || r[i] < 0) ? -1 : rc[i] * nj + r[i];
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

    // Build rows grouped by code, in their order (counting sort, CSR).
    std::vector<std::int64_t> start(static_cast<std::size_t>(G + 1), 0), rows;
    for (std::int64_t i = 0; i < nb_; ++i)
        if (bcode[i] >= 0) ++start[bcode[i] + 1];
    for (std::int64_t g = 0; g < G; ++g) start[g + 1] += start[g];
    rows.resize(static_cast<std::size_t>(start[G]));
    {
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::int64_t i = 0; i < nb_; ++i)
            if (bcode[i] >= 0) rows[fill[bcode[i]]++] = i;
    }

    // Count, then fill.
    std::vector<std::uint8_t> probed(static_cast<std::size_t>(G), 0);  // codes some probe row has
    std::int64_t total = 0;
    for (std::int64_t p = 0; p < np; ++p) {
        const std::int64_t m = pcode[p] < 0 ? 0 : start[pcode[p] + 1] - start[pcode[p]];
        if (pcode[p] >= 0) probed[pcode[p]] = 1;
        total += m > 0 ? m : (keep_probe ? 1 : 0);
    }
    if (keep_build)
        for (std::int64_t i = 0; i < nb_; ++i) total += bcode[i] < 0 || !probed[bcode[i]];
    Column pout = make_fixed(DType::Int64, total), bout = make_fixed(DType::Int64, total);
    auto* po = reinterpret_cast<std::int64_t*>(pout.values->data());
    auto* bo = reinterpret_cast<std::int64_t*>(bout.values->data());
    std::int64_t k = 0;
    for (std::int64_t p = 0; p < np; ++p) {
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
    if (keep_build)
        for (std::int64_t i = 0; i < nb_; ++i)
            if (bcode[i] < 0 || !probed[bcode[i]]) {
                po[k] = -1;
                bo[k] = i;
                ++k;
            }
    return swap ? std::pair<Column, Column>{bout, pout} : std::pair<Column, Column>{pout, bout};
}

}  // namespace sd
