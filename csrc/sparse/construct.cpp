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
    const int64_t rows = m * p;
    NDArray indptr = new_index(rows + 1);
    int64_t* op = i64(indptr);
    op[0] = 0;
    for (int64_t i = 0; i < m; ++i)
        for (int64_t r = 0; r < p; ++r) {
            const int64_t R = i * p + r;
            op[R + 1] = op[R] + (ap[i + 1] - ap[i]) * (bp[r + 1] - bp[r]);
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
    NDArray indptr = new_index(lines + 1);
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
    NDArray indptr = new_index(lines + 1);
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

}  // namespace

void bind_construct(nb::module_& m) {
    m.def("_kron", &kron);
    m.def("_stack_minor", &stack_minor);
    m.def("_stack_major", &stack_major);
    m.def(
        "_band_coo", &band_coo, nb::arg("major"), nb::arg("minor"), nb::arg("data"), nb::arg("k"), nb::kw_only(),
        nb::arg("lower")
    );
}

}  // namespace sd::sparse
