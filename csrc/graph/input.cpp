// The graph area's input conversion (docs/design/graph.md §2.1, §2.7, §3.1; milestone GR1).
//
//   _dense_csr(a, null_value, use_null, nan_null, inf_null, values)
//       a (n, m) C-contiguous sd.ndarray of bool, int64 or float64 -> (indptr, indices, data or None): the
//       canonical CSR of the entries kept, each entry read as a float64 x and left out when use_null and
//       x == null_value, nan_null and x is NaN, or inf_null and x is +-inf. The graph rule of a dense
//       matrix is (0, True, True, True); csgraph_from_dense passes its arguments. values: float64 data
//       (scipy's conversion of the whole matrix first), else only the pattern. as_int64 (a bool or int64
//       matrix; maximum_flow's capacities): the values as int64 data instead of float64.
//   _to_dense(indptr, indices, data, n, null_value)
//       -> (n, n) float64: null_value everywhere, the stored values at their places (csgraph_to_dense).
//
// Both run by blocks of rows fixed by the size alone: one pass counts each row's entries, a prefix sum
// places the rows, a second pass writes them. The same bytes for any thread count.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph.hpp"
#include "graph/graph_mod.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::graph {

namespace {

using std::int64_t;

constexpr int64_t kBlockWork = 1 << 15;  // the entries one task of a row-block pass scans

[[noreturn]] void wrong(const char* func, const char* what) {
    throw nb::type_error((std::string(func) + "(): wrong " + what).c_str());
}

const NDArray& vec(nb::handle o, const char* func, DType dtype, const char* what) {
    if (!is_ndarray(o)) wrong(func, what);
    const NDArray& a = as_ndarray(o);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != dtype) wrong(func, what);
    return a;
}

template <class F>
void run_tasks(int64_t ntasks, bool split, F&& fn) {
    if (!split || ntasks <= 1) {
        for (int64_t t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

nb::object new_vector(DType t, int64_t n) {
    const int64_t shape[] = {n};
    return wrap(new_array(t, shape));
}

// Whether an entry is kept, without branches (the counting pass vectorizes; the entries are random).
template <class T>
struct Keep {
    double null;
    bool use_null, nan_null, inf_null;
    bool operator()(T v) const {
        const double x = static_cast<double>(v);
        bool drop = use_null & (x == null);
        if constexpr (std::is_same_v<T, double>)
            drop = drop | (nan_null & (x != x)) | (inf_null & (std::fabs(x) == INFINITY));
        return !drop;
    }
};

template <class T, class V>
nb::object dense_csr_of(const NDArray& a, const Keep<T>& keep, bool values) {
    constexpr DType kValues = std::is_same_v<V, double> ? DType::Float64 : DType::Int64;
    const int64_t n = a.shape[0], m = a.shape[1];
    const T* A = reinterpret_cast<const T*>(a.data);
    const int64_t rows_per = std::max<int64_t>(1, kBlockWork / std::max<int64_t>(m, 1));
    const int64_t ntasks = (n + rows_per - 1) / rows_per;
    const bool split = n * m >= kParallelWork;
    nb::object op = new_vector(DType::Int64, n + 1);
    int64_t* P = reinterpret_cast<int64_t*>(as_ndarray(op).data);
    P[0] = 0;
    {
        const parallel::Unlocked unlocked(split);
        run_tasks(ntasks, split, [&](int64_t t) {
            for (int64_t i = t * rows_per, e = std::min(n, (t + 1) * rows_per); i < e; ++i) {
                const T* row = A + i * m;
                int64_t c = 0;
                for (int64_t j = 0; j < m; ++j) c += keep(row[j]);
                P[i + 1] = c;
            }
        });
        for (int64_t i = 0; i < n; ++i) P[i + 1] += P[i];
    }
    const int64_t nnz = P[n];
    nb::object oi = new_vector(DType::Int64, nnz);
    nb::object od = values ? new_vector(kValues, nnz) : nb::none();
    int64_t* I = reinterpret_cast<int64_t*>(as_ndarray(oi).data);
    V* D = values ? reinterpret_cast<V*>(as_ndarray(od).data) : nullptr;
    {
        const parallel::Unlocked unlocked(split);
        run_tasks(ntasks, split, [&](int64_t t) {
            // A row is compacted without branches into a scratch row (every entry written, the position
            // advanced only for a kept one), then copied to its place.
            std::vector<int64_t> si(static_cast<size_t>(m));
            std::vector<V> sv(values ? static_cast<size_t>(m) : 0);
            for (int64_t i = t * rows_per, e = std::min(n, (t + 1) * rows_per); i < e; ++i) {
                const T* row = A + i * m;
                int64_t k = 0;
                if (values) {
                    for (int64_t j = 0; j < m; ++j) {
                        si[static_cast<size_t>(k)] = j;
                        sv[static_cast<size_t>(k)] = static_cast<V>(row[j]);
                        k += keep(row[j]);
                    }
                    std::memcpy(D + P[i], sv.data(), static_cast<size_t>(k) * sizeof(V));
                } else {
                    for (int64_t j = 0; j < m; ++j) {
                        si[static_cast<size_t>(k)] = j;
                        k += keep(row[j]);
                    }
                }
                std::memcpy(I + P[i], si.data(), static_cast<size_t>(k) * sizeof(int64_t));
            }
        });
    }
    return nb::make_tuple(op, oi, od);
}

nb::object dense_csr(
    nb::handle ah, double null, bool use_null, bool nan_null, bool inf_null, bool values, bool as_int64
) {
    const char* func = "_dense_csr";
    if (!is_ndarray(ah)) wrong(func, "matrix");
    const NDArray& a = as_ndarray(ah);
    if (a.ndim != 2 || !a.c_contiguous) wrong(func, "matrix");
    if (as_int64 && a.dtype == DType::Float64) wrong(func, "matrix");
    switch (a.dtype) {
        case DType::Bool:
            if (as_int64) return dense_csr_of<bool, int64_t>(a, Keep<bool>{null, use_null, nan_null, inf_null}, values);
            return dense_csr_of<bool, double>(a, Keep<bool>{null, use_null, nan_null, inf_null}, values);
        case DType::Int64:
            if (as_int64)
                return dense_csr_of<int64_t, int64_t>(a, Keep<int64_t>{null, use_null, nan_null, inf_null}, values);
            return dense_csr_of<int64_t, double>(a, Keep<int64_t>{null, use_null, nan_null, inf_null}, values);
        case DType::Float64:
            return dense_csr_of<double, double>(a, Keep<double>{null, use_null, nan_null, inf_null}, values);
        default: wrong(func, "matrix");
    }
}

nb::object to_dense(nb::handle ip, nb::handle ix, nb::handle dh, int64_t n, double null) {
    const View g = view_arg(ip, ix, dh, n, "_to_dense");
    if (!g.data) wrong("_to_dense", "data");
    const int64_t shape[] = {n, n};
    nb::object out = wrap(new_array(DType::Float64, shape));
    double* O = reinterpret_cast<double*>(as_ndarray(out).data);
    const int64_t rows_per = std::max<int64_t>(1, kBlockWork / std::max<int64_t>(n, 1));
    const int64_t ntasks = (n + rows_per - 1) / rows_per;
    const bool split = n * n >= kParallelWork;
    const parallel::Unlocked unlocked(split);
    run_tasks(ntasks, split, [&](int64_t t) {
        for (int64_t i = t * rows_per, e = std::min(n, (t + 1) * rows_per); i < e; ++i) {
            double* row = O + i * n;
            std::fill(row, row + n, null);
            for (int64_t k = g.indptr[i]; k < g.indptr[i + 1]; ++k) row[g.indices[k]] = g.data[k];
        }
    });
    return out;
}

}  // namespace

View view_arg(nb::handle ip, nb::handle ix, nb::handle dh, int64_t n, const char* func) {
    const NDArray& p = vec(ip, func, DType::Int64, "indptr");
    const NDArray& i = vec(ix, func, DType::Int64, "indices");
    if (n < 0 || p.shape[0] != n + 1) wrong(func, "indptr");
    View g;
    g.n = n;
    g.indptr = reinterpret_cast<const int64_t*>(p.data);
    g.indices = reinterpret_cast<const int64_t*>(i.data);
    if (g.indptr[0] != 0 || g.indptr[n] != i.shape[0]) wrong(func, "indptr");
    if (!dh.is_none()) {
        const NDArray& d = vec(dh, func, DType::Float64, "data");
        if (d.shape[0] != i.shape[0]) wrong(func, "data");
        g.data = reinterpret_cast<const double*>(d.data);
    }
    return g;
}

Transposed transpose(
    const View& g, const NDArray& indptr, const NDArray& indices, const NDArray& values, bool weights
) {
    Transposed t;
    t.arrays = sparse::transpose(g.n, g.n, indptr, indices, values);
    t.view = View{
        g.n, sparse::i64(t.arrays.indptr), sparse::i64(t.arrays.indices),
        weights ? reinterpret_cast<const double*>(t.arrays.data.data) : nullptr
    };
    return t;
}

bool symmetric(const View& g, bool weights, bool no_diagonal) {
    // Row u's entries, met for u ascending, are the mirrors of the next unseen entries of their columns'
    // rows when the pattern is symmetric (a canonical row v lists its u ascending): one cursor per row.
    const int64_t n = g.n;
    std::vector<int64_t> cursor(g.indptr, g.indptr + n);
    for (int64_t u = 0; u < n; ++u) {
        for (int64_t k = g.indptr[u], e = g.indptr[u + 1]; k < e; ++k) {
            const int64_t v = g.indices[k];
            const int64_t c = cursor[static_cast<size_t>(v)]++;
            if (c >= g.indptr[v + 1] || g.indices[c] != u || (no_diagonal && v == u)) return false;
            if (weights && g.data) {
                const double a = g.data[k], w = undirected_weight(a, g.data[c]);
                if (std::memcmp(&a, &w, sizeof(double)) != 0 && !(a != a && w != w)) return false;
            }
        }
    }
    return true;  // every entry used one place of its column's row and none ran past it: all are used
}

void bind_input(nb::module_& m) {
    m.def(
        "_dense_csr", &dense_csr, nb::arg("a"), nb::arg("null_value"), nb::arg("use_null"), nb::arg("nan_null"),
        nb::arg("inf_null"), nb::arg("values"), nb::arg("as_int64") = false
    );
    m.def(
        "_to_dense", &to_dense, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("n"),
        nb::arg("null_value")
    );
}

}  // namespace sd::graph
