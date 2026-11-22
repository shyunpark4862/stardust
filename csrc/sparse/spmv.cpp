// Sparse times dense and the sums along an axis (docs/design/sparse.md §2.6, §3.2, §3.3).
//
//   _spmv(indptr, indices, data, x, /, *, n_minor, scatter, transpose_out)
//       The product of compressed arrays (lines indptr, positions indices, values data) and the
//       dense x (1-D or 2-D, any strides, the dtype of data: int64, float64 or complex128).
//       Gather (scatter=False): output row r is line r, sum over its entries p of
//       data[p] * x[indices[p], :] (CSR @ x, CSC^T @ x). Scatter: output row indices[p] gets
//       data[p] * x[line, :] (CSC @ x, CSR^T @ x, x @ CSR). n_minor is the line length: the rows
//       of x (gather) or of the output (scatter).
//       Result: (n_rows,) for a 1-D x, (n_rows, k) for a 2-D x, or (k, n_rows) with
//       transpose_out (x @ A of a 2-D x).
//   _line_sum(indptr, indices, data, /, *, n_minor, scatter, count)
//       The sum over each line (gather) or over each position (scatter) of the values; with
//       count > 0 the mean, the sum divided by count. bool and int64 sum exactly in 128 bits:
//       the sum wraps to int64 (as the array's), the mean is the exact sum rounded once and
//       divided once (base/int_sum.hpp's rule). float64 and complex128 sum in order (one piece of
//       kLine terms at a time, sparse/arith.hpp).
//       track (both): False when the caller knows no position holds more than kLine entries
//       (_minor_over): the scatter skips its per-position counts.
//   _minor_over(indices, n_minor, limit, /)
//       Whether a position occurs more than limit times (a scatter output of more than kLine terms).
//   _exact_mean(data, count, /)
//       The mean of bool or int64 values over count elements: the exact sum rounded once, divided
//       once (NaN for count 0).
//
// Every output element is one sum in ascending index order (sparse/arith.hpp), computed by one
// task: gather splits the lines into fixed ranges of about 2^14 entries; scatter runs in line order,
// on the calling thread or (SP8) on the pool by blocks of outputs, each block taking its terms in
// line order (scatter_blocks). So the bits never depend on the thread count.
#include "base/int_sum.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/arith.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;
using i128 = __int128;

constexpr int64_t kBlockCols = 16;

// ---- one output element: ordered_sum (sparse/arith.hpp) ----

// The same for kb columns at once (acc[0, kb)): step(p, acc) adds entry p's kb terms. K > 0 fixes
// kb at compile time (a full block of contiguous columns: the loop runs in NEON lanes; the order
// of each element's terms is the same).
template <int K, class T, class Step>
inline void ordered_sums(int64_t b, int64_t e, int64_t kb, const Step& step, T* out, std::vector<T>& parts) {
    if constexpr (K > 0) kb = K;
    const int64_t n = e - b;
    T acc[kBlockCols];
    if (n <= kLine) {
        for (int64_t c = 0; c < kb; ++c) acc[c] = T{};
        for (int64_t p = b; p < e; ++p) step(p, acc);
        for (int64_t c = 0; c < kb; ++c) out[c] = acc[c];
        return;
    }
    const int64_t nch = (n + kLine - 1) / kLine;
    parts.resize(static_cast<std::size_t>(nch * kb));
    for (int64_t ch = 0; ch < nch; ++ch) {
        for (int64_t c = 0; c < kb; ++c) acc[c] = T{};
        const int64_t pe = std::min(e, b + (ch + 1) * kLine);
        for (int64_t p = b + ch * kLine; p < pe; ++p) step(p, acc);
        for (int64_t c = 0; c < kb; ++c) parts[static_cast<std::size_t>(c * nch + ch)] = acc[c];
    }
    for (int64_t c = 0; c < kb; ++c) out[c] = tree_sum(parts.data() + c * nch, nch);
}

// Runs fn(lo, hi, parts) over every line, in tasks of about 2^14 units (entries times columns)
// cut by indptr alone.
template <class T, class F>
void over_lines(int64_t n_lines, const int64_t* ip, int64_t k, const F& fn) {
    const int64_t nnz = ip[n_lines] - ip[0];
    const std::vector<int64_t> bounds =
        line_bounds(ip, n_lines, std::max<int64_t>(1, kChunk / std::max<int64_t>(k, 1)));
    const int64_t nt = static_cast<int64_t>(bounds.size()) - 1;
    for_tasks(nt, (nnz + n_lines) * std::max<int64_t>(k, 1), [&](int64_t t) {
        std::vector<T> parts;
        fn(bounds[t], bounds[t + 1], parts);
    });
}

// The scatter on the calling thread, in line order: output row j (k values at y + j * ys, column
// stride yc; zeroed here) accumulates its terms in the order of the lines. step(p, line, yrow)
// adds entry p's k terms into yrow. An output past kLine terms (floats) is cut into pieces of kLine
// like a line: a 16-bit count per output says when a piece is full. track=False: the caller knows
// that no position holds more than kLine entries (_minor_over), and the counts are skipped.
template <class T, class Step>
void scatter_serial(
    int64_t n_lines,
    int64_t n_out,
    const int64_t* ip,
    const int64_t* ix,
    int64_t k,
    T* y,
    int64_t ys,
    int64_t yc,
    bool track,
    const Step& step
) {
    const int64_t nnz = ip[n_lines] - ip[0];
    const parallel::Unlocked unlocked((nnz + n_out) * k >= kParallelThreshold);
    for (int64_t j = 0; j < n_out; ++j)
        for (int64_t c = 0; c < k; ++c) y[j * ys + c * yc] = T{};
    constexpr bool floating = std::is_same_v<T, double> || std::is_same_v<T, complex128>;
    if (!floating || nnz <= kLine || !track) {  // no output has more than kLine terms
        // outputs past the cache are updated at random: fetch the one 16 entries ahead (its whole
        // row; X @ A of 16 rows at nnz 10^7 ran in a third of the time). In cache the fetch only
        // costs (A.T @ x at nnz 10^4-10^7: 20% more).
        const int64_t pend = ip[n_lines];
        const int64_t row_bytes = k * yc * static_cast<int64_t>(sizeof(T));
        if (n_out * row_bytes <= (int64_t(8) << 20)) {
            for (int64_t r = 0; r < n_lines; ++r)
                for (int64_t p = ip[r]; p < ip[r + 1]; ++p) step(p, r, y + ix[p] * ys);
            return;
        }
        for (int64_t r = 0; r < n_lines; ++r)
            for (int64_t p = ip[r]; p < ip[r + 1]; ++p) {
                if (p + 16 < pend) {
                    const char* ahead = reinterpret_cast<const char*>(y + ix[p + 16] * ys);
                    __builtin_prefetch(ahead, 1);
                    if (row_bytes > 64) __builtin_prefetch(ahead + row_bytes - 1, 1);
                }
                step(p, r, y + ix[p] * ys);
            }
        return;
    }
    std::vector<std::uint16_t> cnt(static_cast<std::size_t>(n_out), 0);
    std::unordered_map<int64_t, std::vector<T>> pieces;  // per long output: k values per full piece
    for (int64_t r = 0; r < n_lines; ++r)
        for (int64_t p = ip[r]; p < ip[r + 1]; ++p) {
            const int64_t j = ix[p];
            T* yr = y + j * ys;
            step(p, r, yr);
            if (++cnt[static_cast<std::size_t>(j)] == kLine) {
                std::vector<T>& v = pieces[j];
                for (int64_t c = 0; c < k; ++c) {
                    v.push_back(yr[c * yc]);
                    yr[c * yc] = T{};
                }
                cnt[static_cast<std::size_t>(j)] = 0;
            }
        }
    for (auto& [j, v] : pieces) {
        T* yr = y + j * ys;
        if (cnt[static_cast<std::size_t>(j)])
            for (int64_t c = 0; c < k; ++c) v.push_back(yr[c * yc]);
        const int64_t nch = static_cast<int64_t>(v.size()) / k;
        for (int64_t c = 0; c < k; ++c) yr[c * yc] = tree_sum(v.data() + c, nch, k);
    }
}

// The same scatter on the pool, with the same terms in the same order for every output: blocks of
// 2^shift outputs (cache-sized), and tasks over the lines (fixed ranges of about 2^14 entries, cut
// by indptr alone). Each task sorts its entries by output block (counting, stable) into items --
// make(p, line) gives an entry's item: its output j and what its term needs, so the second pass
// reads them in order --, then each block applies its items (apply(item, yrow)) in task order, that
// is line order, exactly as scatter_serial does for those outputs, long outputs cut the same way.
// So the result is the serial one, bit for bit, for any thread count. Returns false (nothing done)
// when there are too few blocks to share.
template <class T, class Make, class Apply>
bool scatter_blocks(
    int64_t n_lines,
    int64_t n_out,
    const int64_t* ip,
    const int64_t* ix,
    int64_t k,
    T* y,
    int64_t ys,
    int64_t yc,
    bool track,
    const Make& make,
    const Apply& apply
) {
    using Item = decltype(make(int64_t(0), int64_t(0)));
    int shift = 6;
    while (shift < 12 && (int64_t(2) << shift) * k <= 4096) ++shift;  // a block holds <= 4096 values
    if (n_out <= 0) return false;
    const int64_t nb = ((n_out - 1) >> shift) + 1;
    if (nb < 4) return false;
    const int64_t width = int64_t(1) << shift;
    const std::vector<int64_t> bounds = line_bounds(ip, n_lines, kChunk);
    const int64_t nt = static_cast<int64_t>(bounds.size()) - 1;
    const int64_t nnz = ip[n_lines] - ip[0];
    const parallel::Unlocked unlocked(true);
    std::unique_ptr<int64_t[]> off(new int64_t[static_cast<std::size_t>(nt * nb)]());
    parallel::parallel_for(nt, [&](int64_t t) {
        int64_t* c = off.get() + t * nb;
        for (int64_t p = ip[bounds[t]]; p < ip[bounds[t + 1]]; ++p) ++c[ix[p] >> shift];
    });
    std::vector<int64_t> start(static_cast<std::size_t>(nb + 1));
    int64_t run = 0;
    for (int64_t b = 0; b < nb; ++b) {
        start[b] = run;
        for (int64_t t = 0; t < nt; ++t) {
            const int64_t c = off[t * nb + b];
            off[t * nb + b] = run;
            run += c;
        }
    }
    start[nb] = run;
    std::unique_ptr<Item[]> items(new Item[static_cast<std::size_t>(std::max<int64_t>(nnz, 1))]);
    parallel::parallel_for(nt, [&](int64_t t) {
        int64_t* o = off.get() + t * nb;
        for (int64_t r = bounds[t]; r < bounds[t + 1]; ++r)
            for (int64_t p = ip[r]; p < ip[r + 1]; ++p) items[o[ix[p] >> shift]++] = make(p, r);
    });
    constexpr bool floating = std::is_same_v<T, double> || std::is_same_v<T, complex128>;
    const bool count = floating && nnz > kLine && track;
    parallel::parallel_for(nb, [&](int64_t b) {
        const int64_t j0 = b << shift, j1 = std::min(n_out, j0 + width);
        for (int64_t j = j0; j < j1; ++j)
            for (int64_t c = 0; c < k; ++c) y[j * ys + c * yc] = T{};
        const Item* it = items.get();
        if (!count) {
            for (int64_t q = start[b]; q < start[b + 1]; ++q) apply(it[q], y + it[q].j * ys);
            return;
        }
        std::vector<std::uint16_t> cnt(static_cast<std::size_t>(j1 - j0), 0);
        std::unordered_map<int64_t, std::vector<T>> pieces;
        for (int64_t q = start[b]; q < start[b + 1]; ++q) {
            const int64_t j = it[q].j;
            T* yr = y + j * ys;
            apply(it[q], yr);
            if (++cnt[static_cast<std::size_t>(j - j0)] == kLine) {
                std::vector<T>& v = pieces[j];
                for (int64_t c = 0; c < k; ++c) {
                    v.push_back(yr[c * yc]);
                    yr[c * yc] = T{};
                }
                cnt[static_cast<std::size_t>(j - j0)] = 0;
            }
        }
        for (auto& [j, v] : pieces) {
            T* yr = y + j * ys;
            if (cnt[static_cast<std::size_t>(j - j0)])
                for (int64_t c = 0; c < k; ++c) v.push_back(yr[c * yc]);
            const int64_t nch = static_cast<int64_t>(v.size()) / k;
            for (int64_t c = 0; c < k; ++c) yr[c * yc] = tree_sum(v.data() + c, nch, k);
        }
    });
    return true;
}

// The scatter: on the pool from 2^16 units of work (entries times columns), else in line order on
// the calling thread (step(p, line, yrow)). Same bits either way.
template <class T, class Step, class Make, class Apply>
void scatter_any(
    int64_t n_lines,
    int64_t n_out,
    const int64_t* ip,
    const int64_t* ix,
    int64_t k,
    T* y,
    int64_t ys,
    int64_t yc,
    bool track,
    const Step& step,
    const Make& make,
    const Apply& apply
) {
    const int64_t nnz = ip[n_lines] - ip[0];
    if ((nnz + n_out) * k >= 2 * kParallelThreshold && parallel::available() &&
        scatter_blocks<T>(n_lines, n_out, ip, ix, k, y, ys, yc, track, make, apply))
        return;
    scatter_serial<T>(n_lines, n_out, ip, ix, k, y, ys, yc, track, step);
}

// Items of scatter_blocks: an output, and the two factors of its term (a product) or the term.
template <class T>
struct ProductItem {
    int64_t j;
    T a, x;
};
template <class T>
struct TermItem {
    int64_t j;
    T a;
};
struct EntryItem {
    int64_t j, p, line;
};

// ---- _spmv ----

int64_t elements(int64_t stride_bytes, int64_t item) {
    if (stride_bytes % item) throw nb::value_error("_spmv(): strides must be whole elements");
    return stride_bytes / item;
}

template <class T>
nb::object spmv_typed(
    const NDArray& indptr,
    const NDArray& indices,
    const NDArray& data,
    const NDArray& x,
    int64_t n_minor,
    bool scatter,
    bool transpose_out,
    bool track
) {
    const int64_t n_lines = indptr.shape[0] - 1, n_out = n_minor;
    const bool two = x.ndim == 2;
    const int64_t k = two ? x.shape[1] : 1;
    const int64_t item = sizeof(T);
    const int64_t xr = elements(x.strides[0], item), xc = two ? elements(x.strides[1], item) : 1;
    const int64_t rows = scatter ? n_out : n_lines;
    NDArray out;
    if (!two) {
        const int64_t shape[] = {rows};
        out = new_array(dtype_of<T>, shape);
    } else if (transpose_out) {
        const int64_t shape[] = {k, rows};
        out = new_array(dtype_of<T>, shape);
    } else {
        const int64_t shape[] = {rows, k};
        out = new_array(dtype_of<T>, shape);
    }
    T* o = typed<T>(out);
    const int64_t os_r = transpose_out ? 1 : k, os_c = transpose_out ? rows : 1;
    const T* xp = reinterpret_cast<const T*>(x.data);
    auto run_gather = [&](int64_t lines, const int64_t* lp, const int64_t* li, const T* lv) {
        const int64_t pend = lp[lines];
        if (k == 1) {
            over_lines<T>(lines, lp, 1, [&](int64_t lo, int64_t hi, std::vector<T>& parts) {
                line_dots<T>(lo, hi, lp, li, lv, xp, xr, o, os_r, parts);
            });
            return;
        }
        over_lines<T>(lines, lp, k, [&](int64_t lo, int64_t hi, std::vector<T>& parts) {
            T acc_out[kBlockCols];
            for (int64_t r = lo; r < hi; ++r)
                for (int64_t c0 = 0; c0 < k; c0 += kBlockCols) {
                    const int64_t kb = std::min(kBlockCols, k - c0);
                    if (kb == kBlockCols && xc == 1) {
                        // the rows of x are gathered at random: fetch the one 8 entries ahead
                        auto step = [lv, li, xp, xr, c0, pend](int64_t p, T* acc) {
                            if (p + 8 < pend) {
                                const T* ahead = xp + li[p + 8] * xr + c0;
                                __builtin_prefetch(ahead);
                                __builtin_prefetch(ahead + kBlockCols - 1);
                            }
                            const T a = lv[p];
                            const T* row = xp + li[p] * xr + c0;
                            for (int c = 0; c < kBlockCols; ++c) acc[c] = madd(a, row[c], acc[c]);
                        };
                        ordered_sums<kBlockCols, T>(lp[r], lp[r + 1], kb, step, acc_out, parts);
                    } else {
                        auto step = [lv, li, xp, xr, xc, c0, kb](int64_t p, T* acc) {
                            const T a = lv[p];
                            const T* row = xp + li[p] * xr + c0 * xc;
                            for (int64_t c = 0; c < kb; ++c) acc[c] = madd(a, row[c * xc], acc[c]);
                        };
                        ordered_sums<0, T>(lp[r], lp[r + 1], kb, step, acc_out, parts);
                    }
                    for (int64_t c = 0; c < kb; ++c) o[r * os_r + (c0 + c) * os_c] = acc_out[c];
                }
        });
    };
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const T* v = typed<T>(data);
    if (!scatter) {
        run_gather(n_lines, ip, ix, v);
    } else if (k == 1) {
        auto step = [v, xp, xr](int64_t p, int64_t line, T* yr) { yr[0] = madd(v[p], xp[line * xr], yr[0]); };
        auto make = [v, xp, xr, ix](int64_t p, int64_t line) { return ProductItem<T>{ix[p], v[p], xp[line * xr]}; };
        auto apply = [](const ProductItem<T>& t, T* yr) { yr[0] = madd(t.a, t.x, yr[0]); };
        scatter_any<T>(n_lines, n_out, ip, ix, 1, o, os_r, os_c, track, step, make, apply);
    } else {
        // the k values of an output row are contiguous while summing (a transposed result: a work
        // array, transposed at the end -- the same values)
        std::vector<T> work;
        T* y = o;
        if (transpose_out) {
            work.resize(static_cast<std::size_t>(rows * k));
            y = work.data();
        }
        // the rows of x as contiguous k values (x @ A hands x's transpose, whose rows are columns
        // of x: each term would touch k cache lines): a copy, the same values
        std::vector<T> xbuf;
        const T* xs = xp;
        int64_t xsr = xr, xsc = xc;
        if (xc != 1) {
            xbuf.resize(static_cast<std::size_t>(n_lines * k));
            T* b = xbuf.data();
            constexpr int64_t kTile = 64;
            for_chunks(
                n_lines,
                [&](int64_t, int64_t lo, int64_t hi) {
                    for (int64_t l0 = lo; l0 < hi; l0 += kTile) {
                        const int64_t l1 = std::min(hi, l0 + kTile);
                        for (int64_t c = 0; c < k; ++c)
                            for (int64_t l = l0; l < l1; ++l) b[l * k + c] = xp[l * xr + c * xc];
                    }
                },
                std::max<int64_t>(kTile, kChunk / k), n_lines * k
            );
            xs = b;
            xsr = k;
            xsc = 1;
        }
        auto step = [v, xs, xsr, xsc, k](int64_t p, int64_t line, T* yr) {
            const T a = v[p];
            const T* row = xs + line * xsr;
            if (k == kBlockCols && xsc == 1) {
                for (int c = 0; c < kBlockCols; ++c) yr[c] = madd(a, row[c], yr[c]);
                return;
            }
            for (int64_t c = 0; c < k; ++c) yr[c] = madd(a, row[c * xsc], yr[c]);
        };
        auto make = [ix](int64_t p, int64_t line) { return EntryItem{ix[p], p, line}; };
        auto apply = [&step](const EntryItem& t, T* yr) { step(t.p, t.line, yr); };
        scatter_any<T>(n_lines, n_out, ip, ix, k, y, k, 1, track, step, make, apply);
        if (transpose_out) {
            constexpr int64_t kTile = 64;
            for_chunks(
                rows,
                [&](int64_t, int64_t lo, int64_t hi) {
                    for (int64_t j0 = lo; j0 < hi; j0 += kTile) {
                        const int64_t j1 = std::min(hi, j0 + kTile);
                        for (int64_t c = 0; c < k; ++c)
                            for (int64_t j = j0; j < j1; ++j) o[j * os_r + c * os_c] = y[j * k + c];
                    }
                },
                std::max<int64_t>(kTile, kChunk / k), rows * k
            );
        }
    }
    return wrap(std::move(out));
}

nb::object spmv(
    nb::handle indptr_h,
    nb::handle indices_h,
    nb::handle data_h,
    nb::handle x_h,
    int64_t n_minor,
    bool scatter,
    bool transpose_out,
    bool track
) {
    const NDArray& indptr = vector_arg(indptr_h, "_spmv", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_spmv", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_spmv");
    const NDArray& x = require_ndarray(x_h, "x");
    if (x.dtype != data.dtype || data.dtype == DType::Bool || (x.ndim != 1 && x.ndim != 2))
        throw nb::type_error("_spmv(): wrong operands");
    const int64_t n_lines = indptr.shape[0] - 1;
    if (x.shape[0] != (scatter ? n_lines : n_minor)) throw nb::value_error("_spmv(): lengths differ");
    return visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) -> nb::object {
        if constexpr (std::is_same_v<T, bool>) return nb::none();
        else return spmv_typed<T>(indptr, indices, data, x, n_minor, scatter, transpose_out, track);
    });
}

// ---- _line_sum ----

template <class T, class Acc, class R>
nb::object line_sum_typed(
    const NDArray& indptr,
    const NDArray& indices,
    const NDArray& data,
    int64_t n_out,
    bool scatter,
    int64_t count,
    bool track
) {
    const int64_t n_lines = indptr.shape[0] - 1;
    const int64_t rows = scatter ? n_out : n_lines;
    const int64_t shape[] = {rows};
    NDArray out = new_array(dtype_of<R>, shape);
    R* o = typed<R>(out);
    auto finish = [count](Acc s) -> R {
        if constexpr (std::is_same_v<Acc, i128>) {
            if (count > 0) return static_cast<R>(static_cast<double>(s) / static_cast<double>(count));  // IntSum's rule
            return static_cast<R>(static_cast<int64_t>(static_cast<std::uint64_t>(s)));  // wraps
        } else {
            if (count > 0) return s / static_cast<double>(count);
            return s;
        }
    };
    auto term = [](const T& a) -> Acc {
        if constexpr (std::is_same_v<T, bool>) return a ? Acc(1) : Acc(0);
        else return Acc(a);
    };
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const T* v = typed<T>(data);
    auto run_gather = [&](int64_t lines, const int64_t* lp, const T* lv) {
        over_lines<Acc>(lines, lp, 1, [&](int64_t lo, int64_t hi, std::vector<Acc>& parts) {
            auto step = [lv, &term](int64_t p, Acc acc) { return plus(acc, term(lv[p])); };
            for (int64_t r = lo; r < hi; ++r) o[r] = finish(ordered_sum<Acc>(lp[r], lp[r + 1], step, parts));
        });
    };
    if (!scatter) {
        run_gather(n_lines, ip, v);
    } else if constexpr (std::is_same_v<Acc, R>) {
        auto step = [v, &term](int64_t p, int64_t, Acc* yr) { yr[0] = plus(yr[0], term(v[p])); };
        auto make = [v, ix, &term](int64_t p, int64_t) { return TermItem<Acc>{ix[p], term(v[p])}; };
        auto apply = [](const TermItem<Acc>& t, Acc* yr) { yr[0] = plus(yr[0], t.a); };
        scatter_any<Acc>(n_lines, n_out, ip, ix, 1, o, 1, 1, track, step, make, apply);
        if (count > 0) {
            const parallel::Unlocked unlocked(n_out >= kParallelThreshold);
            for (int64_t j = 0; j < n_out; ++j) o[j] = finish(o[j]);
        }
    } else {
        std::vector<Acc> y(static_cast<std::size_t>(n_out));
        auto step = [v, &term](int64_t p, int64_t, Acc* yr) { yr[0] = plus(yr[0], term(v[p])); };
        auto make = [v, ix, &term](int64_t p, int64_t) { return TermItem<Acc>{ix[p], term(v[p])}; };
        auto apply = [](const TermItem<Acc>& t, Acc* yr) { yr[0] = plus(yr[0], t.a); };
        scatter_any<Acc>(n_lines, n_out, ip, ix, 1, y.data(), 1, 1, track, step, make, apply);
        const parallel::Unlocked unlocked(n_out >= kParallelThreshold);
        for (int64_t j = 0; j < n_out; ++j) o[j] = finish(y[static_cast<std::size_t>(j)]);
    }
    return wrap(std::move(out));
}

nb::object line_sum(
    nb::handle indptr_h,
    nb::handle indices_h,
    nb::handle data_h,
    int64_t n_minor,
    bool scatter,
    int64_t count,
    bool track
) {
    const int64_t n_out = n_minor;
    const NDArray& indptr = vector_arg(indptr_h, "_line_sum", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_line_sum", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_line_sum");
    return visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) -> nb::object {
        if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int64_t>) {
            if (count > 0) return line_sum_typed<T, i128, double>(indptr, indices, data, n_out, scatter, count, track);
            return line_sum_typed<T, i128, int64_t>(indptr, indices, data, n_out, scatter, 0, track);
        } else {
            return line_sum_typed<T, T, T>(indptr, indices, data, n_out, scatter, count, track);
        }
    });
}

double exact_mean(nb::handle data_h, int64_t count) {
    const NDArray& data = vector_arg(data_h, "_exact_mean");
    IntSum s;
    const int64_t n = data.shape[0];
    {
        const parallel::Unlocked unlocked(n >= kParallelThreshold);
        if (data.dtype == DType::Bool) {
            int64_t trues = 0;
            const bool* b = typed<bool>(data);
            for (int64_t i = 0; i < n; ++i) trues += b[i];
            s.add(trues);
        } else if (data.dtype == DType::Int64) {
            s.add_range(i64(data), n);
        } else {
            throw nb::type_error("_exact_mean(): bool or int64 only");
        }
    }
    return mean_of(s, count);
}

// Whether some position of [0, n_minor) occurs more than `limit` times in `indices` (counts
// saturate at limit + 1; limit < 65535).
bool minor_over(nb::handle indices_h, int64_t n_minor, int64_t limit) {
    const NDArray& indices = vector_arg(indices_h, "_minor_over", DType::Int64);
    const int64_t nnz = indices.shape[0];
    if (nnz <= limit) return false;
    const int64_t* ix = i64(indices);
    std::vector<std::uint16_t> cnt(static_cast<std::size_t>(n_minor), 0);
    const parallel::Unlocked unlocked(nnz >= kParallelThreshold);
    for (int64_t p = 0; p < nnz; ++p)
        if (++cnt[static_cast<std::size_t>(ix[p])] > limit) return true;
    return false;
}

}  // namespace

void bind_spmv(nb::module_& m) {
    m.def(
        "_spmv", &spmv, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("x"), nb::kw_only(),
        nb::arg("n_minor"), nb::arg("scatter"), nb::arg("transpose_out") = false, nb::arg("track") = true
    );
    m.def(
        "_line_sum", &line_sum, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::kw_only(),
        nb::arg("n_minor"), nb::arg("scatter"), nb::arg("count") = 0, nb::arg("track") = true
    );
    m.def("_minor_over", &minor_over);
    m.def("_exact_mean", &exact_mean);
}

}  // namespace sd::sparse
