// Sparse times sparse (docs/design/sparse.md §2.6, §3.2).
//
//   _spgemm(a_indptr, a_indices, a_data, b_indptr, b_indices, b_data, n_minor, /, *, swap)
//       Gustavson's row by row product of canonical compressed arrays of one dtype (int64,
//       float64 or complex128): output line i is the sum, over the entries (k, a) of line i of the
//       outer operand in ascending k, of a times line k of the inner operand. n_minor is the length
//       of the output lines. Each output entry C_ij is the sum over k ascending of the products,
//       added one by one from zero (sparse/arith.hpp: fma for float64), as scipy's csr_matmat. CSR
//       @ CSR passes (A, B); CSC @ CSC passes (B, A) with swap=True, so that the product is still
//       A's value times B's value. Exact zeros of the result are dropped (scipy), the lines are
//       sorted (canonical).
//
// Two passes over fixed line ranges (by size alone): the count of distinct positions per line, then
// the values; a dense accumulator of the line length per thread, and a bit set of the positions met
// that gives them in ascending order without a sort (scipy's csr_matmat leaves its lines unsorted).
// The results do not depend on the thread count.
#include "array/kernels/simd_sort.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/arith.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;

nb::object readonly(NDArray&& a) {
    a.writeable = false;
    return wrap(std::move(a));
}

// A thread's accumulator of one output line (n_minor values) and the marks of the positions met:
// a mark equals the line's stamp once the position is in the line. Stamps only grow, so the
// workspace is reused across lines and calls without clearing.
// The positions met in a line as a bit set (a word per 64 positions) with a summary bit per word:
// setting one is two writes, and taking them out in ascending order scans the n_minor / 4096
// summary words and visits only the set bits (count trailing zeros), clearing them. Cheaper than
// sorting each line's positions (on one thread the sort was about 60% of the product, SP2) while
// the summary is short against the line (spgemm_typed's choice); the order is exact either way,
// so the result is the same.
struct PositionSet {
    std::vector<std::uint64_t> word, summary;
    void reserve(int64_t n) {
        word.assign(static_cast<std::size_t>((n + 63) >> 6), 0);
        summary.assign(static_cast<std::size_t>((n + 4095) >> 12), 0);
    }
    void insert(int64_t j) {
        word[static_cast<std::size_t>(j >> 6)] |= std::uint64_t(1) << (j & 63);
        summary[static_cast<std::size_t>(j >> 12)] |= std::uint64_t(1) << ((j >> 6) & 63);
    }
    // f(j) for every position in ascending order; the set is empty after.
    template <class F>
    void drain(F&& f) {
        for (std::size_t a = 0; a < summary.size(); ++a) {
            std::uint64_t s = summary[a];
            if (!s) continue;
            summary[a] = 0;
            while (s) {
                const int64_t wi = (static_cast<int64_t>(a) << 6) | __builtin_ctzll(s);
                s &= s - 1;
                std::uint64_t w = word[static_cast<std::size_t>(wi)];
                word[static_cast<std::size_t>(wi)] = 0;
                while (w) {
                    f((wi << 6) | __builtin_ctzll(w));
                    w &= w - 1;
                }
            }
        }
    }
};

template <class T>
struct Workspace {
    std::vector<std::uint64_t> mark;
    std::vector<T> acc;
    std::vector<int64_t> cols;
    PositionSet positions;
    std::uint64_t epoch = 0;
};

template <class T>
Workspace<T>& workspace(int64_t n_minor, bool bits) {
    thread_local Workspace<T> ws;
    if (static_cast<int64_t>(ws.mark.size()) < n_minor) {
        ws.mark.resize(static_cast<std::size_t>(n_minor), 0);
        ws.acc.resize(static_cast<std::size_t>(n_minor));
    }
    // empty between calls (every line drains it); exactly n_minor wide, so the scan is no longer
    if (bits && static_cast<int64_t>(ws.positions.word.size()) != ((n_minor + 63) >> 6)) ws.positions.reserve(n_minor);
    return ws;
}

// The distinct positions of a line, ascending: NEON sorting networks past a few dozen (the array
// area's sort; distinct values, so its instability cannot show).
inline void sort_positions(int64_t* p, int64_t n) {
    if (n < 32) std::sort(p, p + n);
    else simd_sort::sort<simd_sort::I64>(p, n);
}

template <class T>
nb::tuple spgemm_typed(
    const NDArray& a_ptr,
    const NDArray& a_ix,
    const NDArray& a_v,
    const NDArray& b_ptr,
    const NDArray& b_ix,
    const NDArray& b_v,
    int64_t n_minor,
    bool swap
) {
    const int64_t n_lines = a_ptr.shape[0] - 1;
    const int64_t* ap = i64(a_ptr);
    const int64_t* ai = i64(a_ix);
    const int64_t* bp = i64(b_ptr);
    const int64_t* bi = i64(b_ix);
    const T* av = typed<T>(a_v);
    const T* bv = typed<T>(b_v);
    bool bits = false;  // how a line's positions come out in order (set below)
    // line r of the product: emit(j, value) for its nonzero entries in ascending j
    auto product_line = [&](int64_t r, Workspace<T>& ws, auto&& emit) {
        const std::uint64_t stamp = ++ws.epoch;
        ws.cols.clear();
        for (int64_t p = ap[r]; p < ap[r + 1]; ++p) {
            const int64_t k = ai[p];
            const T a = av[p];
            for (int64_t q = bp[k]; q < bp[k + 1]; ++q) {
                const int64_t j = bi[q];
                T& s = ws.acc[static_cast<std::size_t>(j)];
                std::uint64_t& m = ws.mark[static_cast<std::size_t>(j)];
                if (m != stamp) {
                    m = stamp;
                    if (bits) ws.positions.insert(j);
                    else ws.cols.push_back(j);
                    s = T{};
                }
                s = swap ? madd(bv[q], a, s) : madd(a, bv[q], s);
            }
        }
        auto out = [&](int64_t j) {
            const T s = ws.acc[static_cast<std::size_t>(j)];
            if (is_nonzero(s)) emit(j, s);
        };
        if (bits) {
            ws.positions.drain(out);
        } else {
            sort_positions(ws.cols.data(), static_cast<int64_t>(ws.cols.size()));
            for (const int64_t j : ws.cols) out(j);
        }
    };
    // the work of each line: the entries of B it visits
    std::vector<int64_t> cost(static_cast<std::size_t>(n_lines) + 1, 0);
    for (int64_t r = 0; r < n_lines; ++r) {
        int64_t c = 0;
        for (int64_t p = ap[r]; p < ap[r + 1]; ++p) c += bp[ai[p] + 1] - bp[ai[p]];
        cost[static_cast<std::size_t>(r) + 1] = cost[static_cast<std::size_t>(r)] + c;
    }
    const int64_t flops = cost[static_cast<std::size_t>(n_lines)];
    // the bit set while its summary scan (n_minor / 4096 words a line) is short against the line
    bits = (n_minor >> 12) <= 4 * (flops / std::max<int64_t>(n_lines, 1)) + 16;
    const std::vector<int64_t> bounds = line_bounds(cost.data(), n_lines, kChunk);
    const int64_t nt = static_cast<int64_t>(bounds.size()) - 1;
    const int64_t work = flops + n_lines + n_minor;
    // pass 1: distinct positions per line (an upper bound of the stored entries)
    std::vector<int64_t> upper(static_cast<std::size_t>(n_lines) + 1, 0);
    for_tasks(nt, work, [&](int64_t t) {
        Workspace<T>& ws = workspace<T>(n_minor, bits);
        for (int64_t r = bounds[t]; r < bounds[t + 1]; ++r) {
            const std::uint64_t stamp = ++ws.epoch;
            int64_t c = 0;
            for (int64_t p = ap[r]; p < ap[r + 1]; ++p) {
                const int64_t k = ai[p];
                for (int64_t q = bp[k]; q < bp[k + 1]; ++q) {
                    std::uint64_t& m = ws.mark[static_cast<std::size_t>(bi[q])];
                    if (m != stamp) {
                        m = stamp;
                        ++c;
                    }
                }
            }
            upper[static_cast<std::size_t>(r) + 1] = c;
        }
    });
    for (int64_t r = 0; r < n_lines; ++r) upper[static_cast<std::size_t>(r) + 1] += upper[static_cast<std::size_t>(r)];
    const int64_t cap = upper[static_cast<std::size_t>(n_lines)];
    NDArray indices = new_index(cap);
    const int64_t cap_shape[] = {cap};
    NDArray data = new_array(dtype_of<T>, cap_shape);
    int64_t* ox = i64(indices);
    T* ov = typed<T>(data);
    std::vector<int64_t> kept(static_cast<std::size_t>(n_lines), 0);
    // pass 2: the values, sorted lines, zeros dropped (each line from its upper-bound offset)
    for_tasks(nt, work, [&](int64_t t) {
        Workspace<T>& ws = workspace<T>(n_minor, bits);
        for (int64_t r = bounds[t]; r < bounds[t + 1]; ++r) {
            int64_t w = upper[static_cast<std::size_t>(r)];
            product_line(r, ws, [&](int64_t j, T v) {
                ox[w] = j;
                ov[w] = v;
                ++w;
            });
            kept[static_cast<std::size_t>(r)] = w - upper[static_cast<std::size_t>(r)];
        }
    });
    NDArray indptr = new_index(n_lines + 1);
    int64_t* op = i64(indptr);
    op[0] = 0;
    for (int64_t r = 0; r < n_lines; ++r) op[r + 1] = op[r] + kept[static_cast<std::size_t>(r)];
    const int64_t total = op[n_lines];
    if (total < cap) {  // zeros were dropped: close the gaps, in line order
        const parallel::Unlocked unlocked(cap >= kParallelThreshold);
        for (int64_t r = 0; r < n_lines; ++r) {
            const int64_t src = upper[static_cast<std::size_t>(r)], n = kept[static_cast<std::size_t>(r)];
            if (src != op[r]) {
                std::memmove(ox + op[r], ox + src, static_cast<std::size_t>(n) * sizeof(int64_t));
                std::memmove(static_cast<void*>(ov + op[r]), ov + src, static_cast<std::size_t>(n) * sizeof(T));
            }
        }
        indices.shape[0] = total;
        data.shape[0] = total;
    }
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(data)));
}

nb::tuple spgemm(
    nb::handle a_ptr_h,
    nb::handle a_ix_h,
    nb::handle a_v_h,
    nb::handle b_ptr_h,
    nb::handle b_ix_h,
    nb::handle b_v_h,
    int64_t n_minor,
    bool swap
) {
    const NDArray& a_ptr = vector_arg(a_ptr_h, "_spgemm", DType::Int64);
    const NDArray& a_ix = vector_arg(a_ix_h, "_spgemm", DType::Int64);
    const NDArray& a_v = vector_arg(a_v_h, "_spgemm");
    const NDArray& b_ptr = vector_arg(b_ptr_h, "_spgemm", DType::Int64);
    const NDArray& b_ix = vector_arg(b_ix_h, "_spgemm", DType::Int64);
    const NDArray& b_v = vector_arg(b_v_h, "_spgemm", a_v.dtype);
    return visit_dtype(a_v.dtype, [&]<class T>(std::type_identity<T>) -> nb::tuple {
        if constexpr (std::is_same_v<T, bool>) throw nb::type_error("_spgemm(): no bool loop");
        else return spgemm_typed<T>(a_ptr, a_ix, a_v, b_ptr, b_ix, b_v, n_minor, swap);
    });
}

}  // namespace

void bind_spgemm(nb::module_& m) {
    m.def(
        "_spgemm", &spgemm, nb::arg("a_indptr"), nb::arg("a_indices"), nb::arg("a_data"), nb::arg("b_indptr"),
        nb::arg("b_indices"), nb::arg("b_data"), nb::arg("n_minor"), nb::kw_only(), nb::arg("swap") = false
    );
}

}  // namespace sd::sparse
