// Orders of a point set for the triangulations (docs/design/spatial_geometry.md §3.3, §3.4): the
// lexicographic order with the equal points found, Hilbert keys, and the BRIO insertion order.
//
// Everything here is a function of the coordinates alone (no random state, no thread count):
//   - lex_order: the indices sorted by (x, y[, z], index); -0.0 equals 0.0. Equal points are neighbours
//     and the first of a run is its representative (the smallest index, SPT96).
//   - hilbert_key: the position of a grid cell along the Hilbert curve of 2^bits cells per axis
//     (Skilling's transpose form). Cells that follow each other on the curve share a face.
//   - brio_order: the distinct points in rounds whose sizes double (a point's round comes from
//     splitmix64 of its lexicographic rank, so about half the points fall in the last round, a quarter
//     in the one before, ...), each round along the Hilbert curve of the bounding box with 2^21 cells per
//     axis. The rounding of a cell coordinate changes the insertion order only, never a result.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::spatial {

inline constexpr int kHilbertBits = 21;
// The first BRIO round holds at most about this many points.
inline constexpr std::int64_t kBrioFirstRound = 64;

inline bool lex_less(const double* p, const double* q, int d) {
    for (int k = 0; k < d; ++k)
        if (p[k] != q[k]) return p[k] < q[k];
    return false;
}

inline bool same_point(const double* p, const double* q, int d) {
    for (int k = 0; k < d; ++k)
        if (p[k] != q[k]) return false;
    return true;
}

// The indices 0..n-1 sorted by (coordinates, index).
inline std::vector<std::int64_t> lex_order(const double* pts, std::int64_t n, int d) {
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [=](std::int64_t i, std::int64_t j) {
        const double *p = pts + i * d, *q = pts + j * d;
        if (lex_less(p, q, d)) return true;
        if (lex_less(q, p, d)) return false;
        return i < j;
    });
    return order;
}

// rep[i]: the smallest index with point i's coordinates. Returns the representatives in lexicographic
// order (the distinct points).
inline std::vector<std::int64_t> representatives(
    const double* pts, int d, const std::vector<std::int64_t>& order, std::vector<std::int64_t>& rep
) {
    std::vector<std::int64_t> distinct;
    rep.assign(order.size(), 0);
    for (std::size_t k = 0; k < order.size(); ++k) {
        const std::int64_t i = order[k];
        if (k > 0 && same_point(pts + i * d, pts + distinct.back() * d, d)) {
            rep[static_cast<std::size_t>(i)] = distinct.back();
        } else {
            rep[static_cast<std::size_t>(i)] = i;
            distinct.push_back(i);
        }
    }
    return distinct;
}

// The Hilbert index of the cell x[0..d) (each below 2^bits; d = 2 or 3, d * bits <= 64).
inline std::uint64_t hilbert_key(const std::uint32_t* cell, int d, int bits) {
    std::uint32_t x[3] = {cell[0], cell[1], d == 3 ? cell[2] : 0};
    const std::uint32_t top = std::uint32_t{1} << (bits - 1);
    for (std::uint32_t q = top; q > 1; q >>= 1) {
        const std::uint32_t p = q - 1;
        for (int i = 0; i < d; ++i) {
            if (x[i] & q) {
                x[0] ^= p;
            } else {
                const std::uint32_t t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
    }
    for (int i = 1; i < d; ++i) x[i] ^= x[i - 1];
    std::uint32_t t = 0;
    for (std::uint32_t q = top; q > 1; q >>= 1)
        if (x[d - 1] & q) t ^= q - 1;
    for (int i = 0; i < d; ++i) x[i] ^= t;
    std::uint64_t key = 0;
    for (int b = bits - 1; b >= 0; --b)
        for (int i = 0; i < d; ++i) key = (key << 1) | ((x[i] >> b) & 1);
    return key;
}

// The cell of x in [lo, hi] cut into 2^bits cells (the halves keep hi - lo finite).
inline std::uint32_t grid_cell(double x, double lo, double hi, int bits) {
    const double width = 0.5 * hi - 0.5 * lo;
    if (!(width > 0.0)) return 0;
    const double t = (0.5 * x - 0.5 * lo) / width * static_cast<double>(std::uint32_t{1} << bits);
    const std::uint32_t last = (std::uint32_t{1} << bits) - 1;
    if (!(t > 0.0)) return 0;
    return t >= static_cast<double>(last) ? last : static_cast<std::uint32_t>(t);
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// The number of BRIO rounds for m points: the first has about m / 2^(rounds - 1) <= kBrioFirstRound.
inline int brio_rounds(std::int64_t m) {
    int r = 1;
    while ((m >> (r - 1)) > kBrioFirstRound) ++r;
    return r;
}

// The insertion order of the distinct points (given in lexicographic order): rounds from the smallest
// to the largest, each sorted by (Hilbert key, lexicographic rank).
inline std::vector<std::int64_t> brio_order(const double* pts, int d, const std::vector<std::int64_t>& distinct) {
    const std::int64_t m = static_cast<std::int64_t>(distinct.size());
    std::vector<std::int64_t> out(static_cast<std::size_t>(m));
    if (m == 0) return out;
    double lo[3], hi[3];
    for (int k = 0; k < d; ++k) lo[k] = hi[k] = pts[distinct[0] * d + k];
    for (std::int64_t r = 1; r < m; ++r)
        for (int k = 0; k < d; ++k) {
            const double v = pts[distinct[static_cast<std::size_t>(r)] * d + k];
            lo[k] = std::min(lo[k], v);
            hi[k] = std::max(hi[k], v);
        }
    struct Item {
        int round;
        std::uint64_t key;
        std::int64_t rank;
    };
    const int rounds = brio_rounds(m);
    std::vector<Item> items(static_cast<std::size_t>(m));
    for (std::int64_t r = 0; r < m; ++r) {
        const double* p = pts + distinct[static_cast<std::size_t>(r)] * d;
        std::uint32_t cell[3] = {0, 0, 0};
        for (int k = 0; k < d; ++k) cell[k] = grid_cell(p[k], lo[k], hi[k], kHilbertBits);
        const int zeros = std::countr_zero(splitmix64(static_cast<std::uint64_t>(r)) | (std::uint64_t{1} << 63));
        const int round = rounds - 1 - std::min(zeros, rounds - 1);  // 0 is inserted first
        items[static_cast<std::size_t>(r)] = {round, hilbert_key(cell, d, kHilbertBits), r};
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.round != b.round) return a.round < b.round;
        if (a.key != b.key) return a.key < b.key;
        return a.rank < b.rank;
    });
    for (std::int64_t r = 0; r < m; ++r)
        out[static_cast<std::size_t>(r)] = distinct[static_cast<std::size_t>(items[static_cast<std::size_t>(r)].rank)];
    return out;
}

}  // namespace sd::spatial
