#include "ops/join.hpp"

#include "ops/hash.hpp"
#include "ops/selection.hpp"

#include <nanobind/nanobind.h>

#include <string>

namespace nb = nanobind;

namespace sd {

namespace {

// Whether each of rows [begin, begin + n) of the key columns holds an NA.
std::vector<std::uint8_t> any_na(const std::vector<Column>& keys, std::int64_t n) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(n), 0);
    for (const Column& k : keys)
        if (k.validity)
            for (std::int64_t i = 0; i < n; ++i) out[i] |= !k.is_valid(i);
    return out;
}

}  // namespace

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

    // One hash table for both sides: factorize left keys and right keys
    // together, so equal keys get one code whichever side they are on.
    const std::int64_t nl = left[0].length, nr = right[0].length;
    std::vector<Column> both;
    for (std::size_t j = 0; j < left.size(); ++j) both.push_back(concat({left[j], right[j]}));
    Factorized f = factorize_rows(both);
    const auto* code = f.codes.data<std::int64_t>();
    std::vector<std::uint8_t> na_l = any_na(left, nl), na_r = any_na(right, nr);

    // "probe" rows lead the output in their order; "build" rows are found by
    // code. A right join is a left join with the sides swapped.
    const bool swap = how == "right";
    const std::int64_t np = swap ? nr : nl, nb_ = swap ? nl : nr;
    const std::int64_t* pcode = swap ? code + nl : code;
    const std::int64_t* bcode = swap ? code : code + nl;
    const std::vector<std::uint8_t>& pna = swap ? na_r : na_l;
    const std::vector<std::uint8_t>& bna = swap ? na_l : na_r;
    const bool keep_probe = how != "inner";  // left, right (swapped), outer keep unmatched probe rows
    const bool keep_build = how == "outer";

    // Build rows grouped by code, in their order (counting sort, CSR).
    const std::int64_t G = f.ngroups;
    std::vector<std::int64_t> start(static_cast<std::size_t>(G + 1), 0), rows;
    for (std::int64_t i = 0; i < nb_; ++i)
        if (!bna[i]) ++start[bcode[i] + 1];
    for (std::int64_t g = 0; g < G; ++g) start[g + 1] += start[g];
    rows.resize(static_cast<std::size_t>(start[G]));
    {
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::int64_t i = 0; i < nb_; ++i)
            if (!bna[i]) rows[fill[bcode[i]]++] = i;
    }

    // Count, then fill.
    std::vector<std::uint8_t> probed(static_cast<std::size_t>(G), 0);  // codes some probe row has
    std::int64_t total = 0;
    for (std::int64_t p = 0; p < np; ++p) {
        const std::int64_t m = pna[p] ? 0 : start[pcode[p] + 1] - start[pcode[p]];
        if (!pna[p]) probed[pcode[p]] = 1;
        total += m > 0 ? m : (keep_probe ? 1 : 0);
    }
    if (keep_build)
        for (std::int64_t i = 0; i < nb_; ++i) total += bna[i] || !probed[bcode[i]];
    Column pout = make_fixed(DType::Int64, total), bout = make_fixed(DType::Int64, total);
    auto* po = reinterpret_cast<std::int64_t*>(pout.values->data());
    auto* bo = reinterpret_cast<std::int64_t*>(bout.values->data());
    std::int64_t k = 0;
    for (std::int64_t p = 0; p < np; ++p) {
        const std::int64_t b0 = pna[p] ? 0 : start[pcode[p]], b1 = pna[p] ? 0 : start[pcode[p] + 1];
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
            if (bna[i] || !probed[bcode[i]]) {
                po[k] = -1;
                bo[k] = i;
                ++k;
            }
    return swap ? std::pair<Column, Column>{bout, pout} : std::pair<Column, Column>{pout, bout};
}

}  // namespace sd
