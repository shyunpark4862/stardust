// The builders of sd.sparse (docs/design/sparse.md §3.4, milestone SP3): the Kronecker product,
// stacking, and the triangle filters.
//
//   _kron(a_indptr, a_indices, a_data, b_indptr, b_indices, b_data, b_cols, /)
//       The canonical CSR arrays of kron(A, B) from the CSR arrays of A (m x n) and B (p x q):
//       row i*p + r holds a_ij * b_rl at column j*q + l for every stored a_ij of row i and b_rl
//       of row r (stored zeros give stored zeros). Both data arrays have the loop dtype the
//       caller chose by the array's multiply rule; the product is the array's (ops::Multiply).
//   _stack_minor(parts, offsets, /)
//       Blocks side by side along the minor axis: `parts` a list of (indptr, indices, data) with
//       the same number of lines and the same dtype, `offsets` the minor offset of each block.
//       Line r of the result is line r of each block in block order, shifted: canonical again.
//   _stack_major(parts, /)
//       Blocks one after the other along the major axis (the lines of each block in turn).
//   _band_coo(major, minor, data, k, /, *, lower)
//       The COO entries on or below (lower: col <= row + k) or on or above (col >= row + k) the
//       k-th diagonal, order and repeats kept; None when every entry stays.
//   _band_compressed(indptr, indices, data, k, /, *, lower, by_rows)
//       The same for canonical CSR (by_rows) or CSC arrays: (row, col, data) of the kept entries
//       in line order (the order of their tocoo()), new arrays. A line keeps a prefix or a suffix
//       of its sorted entries, found by counting (short lines) or a binary search.
//   _kronsum(a_indptr, a_indices, a_data, b_indptr, b_indices, b_data, /)
//       The canonical CSR arrays of kron(I_n, A) + kron(B, I_m) from the CSR arrays of the square
//       A (m x m) and B (n x n), one dtype: row i*m + p holds A's row p in block i and b_ij at
//       column j*m + p; a position both hold (a_pp and b_ii) is add_value(a, b) (A's first, as
//       the sum of the two COO parts); zeros (and -0.0) are not stored. Rows are filled straight,
//       in fixed ranges of rows.
//   _diags(n_rows, n_cols, offsets, diagonals, /)
//       The canonical CSR arrays of diags_array: diagonal j (a contiguous 1-D array of the
//       result's dtype, of length 1 -- the whole diagonal -- or at least the diagonal's) at
//       offset offsets[j] (distinct), its zeros (and -0.0) not stored; None when the rows times
//       the diagonals far exceed the entries (the caller builds through COO then).
//
// Every output element is written by one task and tasks are cut by sizes alone, so the results
// do not depend on the thread count.
#include "array/kernels/scalar_ops.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;

nb::object readonly(NDArray&& a) {
    a.writeable = false;
    return wrap(std::move(a));
}

NDArray new_values(DType dtype, int64_t n) {
    const int64_t shape[] = {n};
    return new_array(dtype, shape);
}

int64_t lines_per_task(int64_t line_len) { return std::max<int64_t>(1, kChunk / std::max<int64_t>(line_len, 1)); }

// ---- _kron ----

[[noreturn]] void overflow() {
    PyErr_SetString(PyExc_OverflowError, "_kron(): the result's shape is out of range");
    throw nb::python_error();
}

nb::tuple kron(
    nb::handle ap_h, nb::handle ai_h, nb::handle av_h, nb::handle bp_h, nb::handle bi_h, nb::handle bv_h, int64_t q
) {
    const NDArray& ap_a = vector_arg(ap_h, "_kron", DType::Int64);
    const NDArray& ai_a = vector_arg(ai_h, "_kron", DType::Int64);
    const NDArray& av_a = vector_arg(av_h, "_kron");
    const NDArray& bp_a = vector_arg(bp_h, "_kron", DType::Int64);
    const NDArray& bi_a = vector_arg(bi_h, "_kron", DType::Int64);
    const NDArray& bv_a = vector_arg(bv_h, "_kron", av_a.dtype);
    const int64_t m = ap_a.shape[0] - 1, p = bp_a.shape[0] - 1;
    const int64_t* ap = i64(ap_a);
    const int64_t* ai = i64(ai_a);
    const int64_t* bp = i64(bp_a);
    const int64_t* bi = i64(bi_a);
    // The caller checks the result's shape (kron(): the result's shape is out of range); these
    // guard the index arithmetic below all the same.
    int64_t rows = 0, top = 0;
    if (__builtin_mul_overflow(m, p, &rows) || rows == std::numeric_limits<int64_t>::max()) overflow();
    for (int64_t s = 0; s < ai_a.shape[0]; ++s) top = std::max(top, ai[s]);
    if (ai_a.shape[0] > 0 && (__builtin_mul_overflow(top, q, &top) || __builtin_add_overflow(top, q - 1, &top)))
        overflow();
    NDArray indptr = new_indptr(rows);
    int64_t* op = i64(indptr);
    op[0] = 0;
    for (int64_t i = 0; i < m; ++i)
        for (int64_t r = 0; r < p; ++r) {
            const int64_t R = i * p + r;
            int64_t cnt = 0;
            if (__builtin_mul_overflow(ap[i + 1] - ap[i], bp[r + 1] - bp[r], &cnt) ||
                __builtin_add_overflow(op[R], cnt, &op[R + 1]))
                throw nb::value_error("array is too big");  // more stored entries than an int64 counts
        }
    const int64_t total = op[rows];
    NDArray indices = new_index(total);
    NDArray data = new_values(av_a.dtype, total);
    int64_t* ox = i64(indices);
    visit_dtype(av_a.dtype, [&]<class T>(std::type_identity<T>) {
        if constexpr (std::is_same_v<T, bool>) {
            throw nb::type_error("_kron(): no bool loop");
        } else {
            const T* av = typed<T>(av_a);
            const T* bv = typed<T>(bv_a);
            T* ov = typed<T>(data);
            for_chunks(
                rows,
                [&](int64_t, int64_t lo, int64_t hi) {
                    for (int64_t R = lo; R < hi; ++R) {
                        const int64_t i = R / p, r = R % p;
                        int64_t w = op[R];
                        for (int64_t s = ap[i]; s < ap[i + 1]; ++s)
                            for (int64_t t = bp[r]; t < bp[r + 1]; ++t) {
                                ox[w] = ai[s] * q + bi[t];
                                ov[w] = ops::Multiply::f(av[s], bv[t]);
                                ++w;
                            }
                    }
                },
                lines_per_task(rows > 0 ? total / rows : 0), total + rows
            );
        }
    });
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(data)));
}

// ---- stacking ----

struct Part {
    const int64_t* ip;
    const int64_t* ix;
    const std::byte* v;
    int64_t lines, nnz;
};

std::vector<Part> parts_arg(nb::handle parts_h, const char* func, DType& dtype) {
    std::vector<Part> parts;
    bool first = true;
    for (nb::handle t : parts_h) {
        const NDArray& ip = vector_arg(t[0], func, DType::Int64);
        const NDArray& ix = vector_arg(t[1], func, DType::Int64);
        const NDArray& v = vector_arg(t[2], func);
        if (first) dtype = v.dtype;
        else if (v.dtype != dtype) throw nb::type_error((std::string(func) + "(): dtypes differ").c_str());
        first = false;
        parts.push_back({i64(ip), i64(ix), v.data, ip.shape[0] - 1, ix.shape[0]});
    }
    if (parts.empty()) throw nb::value_error((std::string(func) + "(): no parts").c_str());
    return parts;
}

nb::tuple stack_minor(nb::handle parts_h, nb::handle offsets_h) {
    DType dtype = DType::Float64;
    const std::vector<Part> parts = parts_arg(parts_h, "_stack_minor", dtype);
    std::vector<int64_t> off;
    for (nb::handle v : offsets_h) off.push_back(nb::cast<int64_t>(v));
    const int64_t lines = parts[0].lines, size = itemsize(dtype);
    int64_t nnz = 0;
    for (const Part& p : parts) {
        if (p.lines != lines) throw nb::value_error("_stack_minor(): line counts differ");
        nnz += p.nnz;
    }
    NDArray indptr = new_indptr(lines);
    int64_t* op = i64(indptr);
    op[0] = 0;
    for (int64_t r = 0; r < lines; ++r) {
        int64_t c = 0;
        for (const Part& p : parts) c += p.ip[r + 1] - p.ip[r];
        op[r + 1] = op[r] + c;
    }
    NDArray indices = new_index(nnz);
    NDArray data = new_values(dtype, nnz);
    int64_t* ox = i64(indices);
    for_chunks(
        lines,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r) {
                int64_t w = op[r];
                for (std::size_t b = 0; b < parts.size(); ++b) {
                    const Part& p = parts[b];
                    const int64_t s = p.ip[r], e = p.ip[r + 1];
                    for (int64_t k = s; k < e; ++k) ox[w + k - s] = p.ix[k] + off[b];
                    std::memcpy(data.data + w * size, p.v + s * size, static_cast<std::size_t>((e - s) * size));
                    w += e - s;
                }
            }
        },
        lines_per_task(lines > 0 ? nnz / lines : 0), nnz + lines
    );
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(data)));
}

nb::tuple stack_major(nb::handle parts_h) {
    DType dtype = DType::Float64;
    const std::vector<Part> parts = parts_arg(parts_h, "_stack_major", dtype);
    const int64_t size = itemsize(dtype);
    int64_t lines = 0, nnz = 0;
    for (const Part& p : parts) {
        lines += p.lines;
        nnz += p.nnz;
    }
    NDArray indptr = new_indptr(lines);
    NDArray indices = new_index(nnz);
    NDArray data = new_values(dtype, nnz);
    int64_t* op = i64(indptr);
    int64_t* ox = i64(indices);
    // where each part starts in the result, then the copies in fixed chunks of entries (the
    // indices of a part shift by nothing: its lines start at its base)
    std::vector<int64_t> base(parts.size() + 1, 0), first(parts.size() + 1, 0);
    for (std::size_t b = 0; b < parts.size(); ++b) {
        base[b + 1] = base[b] + (parts[b].ip[parts[b].lines] - parts[b].ip[0]);
        first[b + 1] = first[b] + parts[b].lines;
    }
    for (std::size_t b = 0; b < parts.size(); ++b) {
        const Part& p = parts[b];
        const int64_t shift = base[b] - p.ip[0];
        for_chunks(p.lines + 1, [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi && r < p.lines; ++r) op[first[b] + r] = p.ip[r] + shift;
        });
        const int64_t n = base[b + 1] - base[b];
        for_chunks(
            n,
            [&](int64_t, int64_t lo, int64_t hi) {
                std::memcpy(ox + base[b] + lo, p.ix + p.ip[0] + lo, static_cast<std::size_t>((hi - lo) * 8));
                std::memcpy(
                    data.data + (base[b] + lo) * size, p.v + (p.ip[0] + lo) * size,
                    static_cast<std::size_t>((hi - lo) * size)
                );
            },
            1 << 16
        );
    }
    op[lines] = base[parts.size()];
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(data)));
}

// ---- _band_coo ----

nb::object band_coo(nb::handle major_h, nb::handle minor_h, nb::handle data_h, int64_t k, bool lower) {
    const NDArray& major = vector_arg(major_h, "_band_coo", DType::Int64);
    const NDArray& minor = vector_arg(minor_h, "_band_coo", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_band_coo");
    const int64_t nnz = minor.shape[0], size = data.itemsize();
    if (major.shape[0] != nnz) throw nb::value_error("_band_coo(): lengths differ");
    const int64_t* ma = i64(major);
    const int64_t* mi = i64(minor);
    // col <= row + k (lower) or col >= row + k (upper), without overflow: col - row vs k
    auto keep = [&](int64_t e) { return lower ? mi[e] - ma[e] <= k : mi[e] - ma[e] >= k; };
    const int64_t nt = num_chunks(nnz);
    std::vector<int64_t> start(static_cast<std::size_t>(nt + 1), 0);
    for_chunks(nnz, [&](int64_t t, int64_t lo, int64_t hi) {
        int64_t c = 0;
        for (int64_t e = lo; e < hi; ++e) c += keep(e);
        start[static_cast<std::size_t>(t + 1)] = c;
    });
    for (int64_t t = 0; t < nt; ++t) start[static_cast<std::size_t>(t + 1)] += start[static_cast<std::size_t>(t)];
    const int64_t total = start[static_cast<std::size_t>(nt)];
    if (total == nnz) return nb::none();
    NDArray out_r = new_index(total);
    NDArray out_c = new_index(total);
    NDArray out_v = new_values(data.dtype, total);
    int64_t* orow = i64(out_r);
    int64_t* oc = i64(out_c);
    for_chunks(nnz, [&](int64_t t, int64_t lo, int64_t hi) {
        int64_t w = start[static_cast<std::size_t>(t)];
        for (int64_t e = lo; e < hi; ++e)
            if (keep(e)) {
                orow[w] = ma[e];
                oc[w] = mi[e];
                std::memcpy(out_v.data + w * size, data.data + e * size, static_cast<std::size_t>(size));
                ++w;
            }
    });
    return nb::make_tuple(readonly(std::move(out_r)), readonly(std::move(out_c)), wrap(std::move(out_v)));
}

// ---- _band_compressed ----

// t + k saturated to the int64 range.
inline int64_t add_sat(int64_t t, int64_t k) {
    int64_t r;
    if (__builtin_add_overflow(t, k, &r))
        return k > 0 ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int64_t>::min();
    return r;
}

// t - k saturated to the int64 range.
inline int64_t sub_sat(int64_t t, int64_t k) {
    int64_t r;
    if (__builtin_sub_overflow(t, k, &r))
        return k < 0 ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int64_t>::min();
    return r;
}

// How many of the sorted ix[0, n) are <= t.
inline int64_t count_le(const int64_t* ix, int64_t n, int64_t t) {
    if (n > 32) return std::upper_bound(ix, ix + n, t) - ix;
    int64_t c = 0;
    for (int64_t p = 0; p < n; ++p) c += ix[p] <= t;
    return c;
}

nb::tuple band_compressed(
    nb::handle indptr_h, nb::handle indices_h, nb::handle data_h, int64_t k, bool lower, bool by_rows
) {
    const NDArray& indptr = vector_arg(indptr_h, "_band_compressed", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_band_compressed", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_band_compressed");
    const int64_t lines = indptr.shape[0] - 1, nnz = indices.shape[0];
    if (lines < 0 || data.shape[0] != nnz) throw nb::value_error("_band_compressed(): lengths differ");
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    // line l keeps ix <= t(l) (a prefix) or ix >= t(l) (a suffix):
    //   CSR, row l: lower col <= l + k (prefix), upper col >= l + k (suffix);
    //   CSC, column l: lower row >= l - k (suffix), upper row <= l - k (prefix).
    // cut[l]: where the kept prefix ends or the kept suffix starts.
    const bool prefix = lower == by_rows;
    std::vector<int64_t> cut(static_cast<std::size_t>(lines));
    const std::vector<int64_t> bounds = line_bounds(ip, lines, kChunk);
    const int64_t nt = static_cast<int64_t>(bounds.size()) - 1;
    std::vector<int64_t> start(static_cast<std::size_t>(nt + 1), 0);
    for_tasks(nt, nnz + lines, [&](int64_t t) {
        int64_t c = 0;
        for (int64_t l = bounds[t]; l < bounds[t + 1]; ++l) {
            const int64_t s = ip[l], e = ip[l + 1];
            const int64_t th = by_rows ? add_sat(l, k) : sub_sat(l, k);
            int64_t q;
            if (prefix) q = s + count_le(ix + s, e - s, th);
            else q = th == std::numeric_limits<int64_t>::min() ? s : s + count_le(ix + s, e - s, th - 1);
            cut[static_cast<std::size_t>(l)] = q;
            c += prefix ? q - s : e - q;
        }
        start[static_cast<std::size_t>(t + 1)] = c;
    });
    for (int64_t t = 0; t < nt; ++t) start[static_cast<std::size_t>(t + 1)] += start[static_cast<std::size_t>(t)];
    const int64_t total = start[static_cast<std::size_t>(nt)];
    NDArray out_major = new_index(total);
    NDArray out_minor = new_index(total);
    NDArray out_v = new_values(data.dtype, total);
    int64_t* oma = i64(out_major);
    int64_t* omi = i64(out_minor);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        T* ov = typed<T>(out_v);
        for_tasks(nt, nnz + lines, [&](int64_t t) {
            int64_t w = start[static_cast<std::size_t>(t)];
            for (int64_t l = bounds[t]; l < bounds[t + 1]; ++l) {
                const int64_t q = cut[static_cast<std::size_t>(l)];
                const int64_t a = prefix ? ip[l] : q, b = prefix ? q : ip[l + 1];
                for (int64_t p = a; p < b; ++p, ++w) {
                    oma[w] = l;
                    omi[w] = ix[p];
                    ov[w] = v[p];
                }
            }
        });
    });
    nb::object major = readonly(std::move(out_major)), minor = readonly(std::move(out_minor));
    return by_rows ? nb::make_tuple(major, minor, wrap(std::move(out_v)))
                   : nb::make_tuple(minor, major, wrap(std::move(out_v)));
}

// ---- _kronsum ----

nb::tuple kronsum(
    nb::handle a_ptr_h, nb::handle a_ix_h, nb::handle a_v_h, nb::handle b_ptr_h, nb::handle b_ix_h, nb::handle b_v_h
) {
    const NDArray& a_ptr = vector_arg(a_ptr_h, "_kronsum", DType::Int64);
    const NDArray& a_ix = vector_arg(a_ix_h, "_kronsum", DType::Int64);
    const NDArray& a_v = vector_arg(a_v_h, "_kronsum");
    const NDArray& b_ptr = vector_arg(b_ptr_h, "_kronsum", DType::Int64);
    const NDArray& b_ix = vector_arg(b_ix_h, "_kronsum", DType::Int64);
    const NDArray& b_v = vector_arg(b_v_h, "_kronsum", a_v.dtype);
    const int64_t m = a_ptr.shape[0] - 1, n = b_ptr.shape[0] - 1;
    if (m < 0 || n < 0 || a_ix.shape[0] != a_v.shape[0] || b_ix.shape[0] != b_v.shape[0])
        throw nb::value_error("_kronsum(): lengths differ");
    const int64_t* ap = i64(a_ptr);
    const int64_t* ai = i64(a_ix);
    const int64_t* bp = i64(b_ptr);
    const int64_t* bi = i64(b_ix);
    int64_t rows = 0;
    if (__builtin_mul_overflow(m, n, &rows)) overflow();
    // the entries before overlaps and zeros (saturated: it only sizes the tasks)
    const double upper_d =
        static_cast<double>(ap[m]) * static_cast<double>(n) + static_cast<double>(bp[n]) * static_cast<double>(m);
    const int64_t upper = upper_d < 9.0e18 ? static_cast<int64_t>(upper_d) : int64_t(9000000000000000000);
    NDArray indptr = new_indptr(rows);
    int64_t* op = i64(indptr);
    const int64_t per_row = std::max<int64_t>(1, (upper / std::max<int64_t>(rows, 1)));
    const int64_t chunk = lines_per_task(per_row);
    const int64_t nt = num_chunks(rows, chunk);
    std::vector<int64_t> start(static_cast<std::size_t>(nt + 1), 0);
    NDArray indices, data;
    visit_dtype(a_v.dtype, [&]<class T>(std::type_identity<T>) {
        const T* av = typed<T>(a_v);
        const T* bv = typed<T>(b_v);
        // row r = i*m + p: emit(col, value) in ascending col
        auto row = [&](int64_t r, auto&& emit) {
            const int64_t i = r / m, p = r - i * m;
            const int64_t base = i * m;
            int64_t qb = bp[i];
            const int64_t qe = bp[i + 1];
            for (; qb < qe && bi[qb] < i; ++qb)  // B's entries left of the diagonal block
                if (is_nonzero(bv[qb])) emit(bi[qb] * m + p, bv[qb]);
            const bool bdiag = qb < qe && bi[qb] == i;
            bool placed = !bdiag;
            for (int64_t k = ap[p]; k < ap[p + 1]; ++k) {
                const int64_t c = ai[k];
                if (!placed && c >= p) {
                    if (c == p) {
                        const T v = add_value(av[k], bv[qb]);
                        if (is_nonzero(v)) emit(base + c, v);
                        placed = true;
                        continue;
                    }
                    if (is_nonzero(bv[qb])) emit(base + p, bv[qb]);
                    placed = true;
                }
                if (is_nonzero(av[k])) emit(base + c, av[k]);
            }
            if (!placed && is_nonzero(bv[qb])) emit(base + p, bv[qb]);
            if (bdiag) ++qb;
            for (; qb < qe; ++qb)  // right of it
                if (is_nonzero(bv[qb])) emit(bi[qb] * m + p, bv[qb]);
        };
        const int64_t work = upper + rows;  // both below 2^63 - 2^60
        for_chunks(
            rows,
            [&](int64_t t, int64_t lo, int64_t hi) {
                int64_t c = 0;
                for (int64_t r = lo; r < hi; ++r) row(r, [&](int64_t, T) { ++c; });
                start[static_cast<std::size_t>(t + 1)] = c;
            },
            chunk, work
        );
        for (int64_t t = 0; t < nt; ++t) start[static_cast<std::size_t>(t + 1)] += start[static_cast<std::size_t>(t)];
        const int64_t total = start[static_cast<std::size_t>(nt)];
        indices = new_index(total);
        data = new_values(a_v.dtype, total);
        int64_t* ox = i64(indices);
        T* ov = typed<T>(data);
        op[0] = 0;
        for_chunks(
            rows,
            [&](int64_t t, int64_t lo, int64_t hi) {
                int64_t w = start[static_cast<std::size_t>(t)];
                for (int64_t r = lo; r < hi; ++r) {
                    row(r, [&](int64_t col, T v) {
                        ox[w] = col;
                        ov[w] = v;
                        ++w;
                    });
                    op[r + 1] = w;
                }
            },
            chunk, work
        );
    });
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(data)));
}

// ---- _diags ----

nb::object diags(int64_t m, int64_t n, nb::handle offsets_h, nb::handle diagonals_h) {
    struct Diag {
        int64_t off;
        const NDArray* a;
        int64_t r0, len;  // the rows [r0, r0 + len) it crosses
    };
    std::vector<Diag> ds;
    nb::object oseq = nb::borrow(offsets_h), dseq = nb::borrow(diagonals_h);
    const std::size_t nd = nb::len(oseq);
    if (nb::len(dseq) != nd || nd == 0) throw nb::value_error("_diags(): lengths differ");
    DType dtype = DType::String;
    int64_t stored = 0;  // the entries, zeros included
    for (std::size_t j = 0; j < nd; ++j) {
        const NDArray& a = vector_arg(dseq[j], "_diags");
        if (j == 0) dtype = a.dtype;
        else if (a.dtype != dtype) throw nb::type_error("_diags(): the diagonals' dtypes differ");
        const int64_t off = nb::cast<int64_t>(oseq[j]);
        const int64_t len = std::max<int64_t>(std::min({m + off, n - off, m, n}), 0);
        if (len > 0 && a.shape[0] != 1 && a.shape[0] < len) throw nb::value_error("_diags(): short diagonal");
        ds.push_back({off, &a, std::max<int64_t>(-off, 0), len});
        stored += len;
    }
    std::sort(ds.begin(), ds.end(), [](const Diag& x, const Diag& y) { return x.off < y.off; });
    const int64_t kd = static_cast<int64_t>(nd);
    if (m > 0 && kd > 4 * (stored / m + 1)) return nb::none();  // most diagonals miss most rows
    // row i holds the diagonals with -i <= off <= n - 1 - i, in offset order (columns increase);
    // the entries before row lo: each diagonal's rows below lo
    auto before = [&](int64_t lo) {
        int64_t c = 0;
        for (const Diag& d : ds) c += std::clamp<int64_t>(lo - d.r0, 0, d.len);
        return c;
    };
    NDArray indptr = new_indptr(m);
    NDArray indices = new_index(stored);
    NDArray out = new_values(dtype, stored);
    int64_t* ip = i64(indptr);
    int64_t* ox = i64(indices);
    const int64_t rows_per_task = std::max<int64_t>(1, kChunk / kd);
    const int64_t nt = num_chunks(m, rows_per_task);
    std::vector<char> zero(static_cast<std::size_t>(std::max<int64_t>(nt, 1)), 0);
    visit_dtype(dtype, [&]<class T>(std::type_identity<T>) {
        std::vector<const T*> val(nd);
        std::vector<int64_t> step(nd);
        for (std::size_t j = 0; j < nd; ++j) {
            step[j] = ds[j].a->shape[0] == 1 ? 0 : 1;
            val[j] = typed<T>(*ds[j].a);
        }
        T* ov = typed<T>(out);
        for_chunks(
            m,
            [&](int64_t t, int64_t lo, int64_t hi) {
                int64_t w = before(lo);
                // [jlo, jhi): the diagonals crossing row i (both bounds only fall as i grows)
                int64_t jlo = kd, jhi = kd;
                while (jlo > 0 && ds[jlo - 1].off >= -lo) --jlo;
                while (jhi > 0 && ds[jhi - 1].off > n - 1 - lo) --jhi;
                bool z = false;
                for (int64_t i = lo; i < hi; ++i) {
                    while (jlo > 0 && ds[jlo - 1].off >= -i) --jlo;
                    while (jhi > 0 && ds[jhi - 1].off > n - 1 - i) --jhi;
                    ip[i] = w;
                    for (int64_t j = jlo; j < jhi; ++j) {
                        const T v = val[j][(i - ds[j].r0) * step[j]];
                        ox[w] = i + ds[j].off;
                        ov[w] = v;
                        z |= !is_nonzero(v);
                        ++w;
                    }
                }
                zero[static_cast<std::size_t>(t)] = z;
            },
            rows_per_task, std::min(m, std::numeric_limits<int64_t>::max() / kd) * kd
        );
    });
    ip[m] = stored;
    bool zeros = false;
    for (char z : zero) zeros = zeros || z;
    return nb::make_tuple(readonly(std::move(indptr)), readonly(std::move(indices)), wrap(std::move(out)), zeros);
}

}  // namespace

void bind_construct(nb::module_& m) {
    m.def("_diags", &diags);
    m.def("_kronsum", &kronsum);
    m.def(
        "_band_compressed", &band_compressed, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("k"),
        nb::kw_only(), nb::arg("lower"), nb::arg("by_rows")
    );
    m.def("_kron", &kron);
    m.def("_stack_minor", &stack_minor);
    m.def("_stack_major", &stack_major);
    m.def(
        "_band_coo", &band_coo, nb::arg("major"), nb::arg("minor"), nb::arg("data"), nb::arg("k"), nb::kw_only(),
        nb::arg("lower")
    );
}

}  // namespace sd::sparse
