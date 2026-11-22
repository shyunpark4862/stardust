// The ordering and the laplacian (docs/design/graph.md §2.5, §3.6; milestone GR4).
//
//   _rcm(indptr, indices, n, symmetric_mode) -> int64 (n,)
//       The reverse Cuthill-McKee permutation of a square graph's pattern (decision GR52). The graph read
//       is the pattern itself (symmetric_mode) or the union of its pattern and its transpose's. A node's
//       degree is its row length in that pattern plus 1 when its diagonal entry is stored (a self loop
//       has two ends: scipy's count). The components are searched breadth first in the order of their
//       seeds, each seed the unvisited node first in the order (degree, index); a node's unvisited
//       neighbours join the queue in the order (degree, index). The whole order is then reversed.
//       Serial: one pass over the pattern (two with the transpose).
//   _laplacian_csr(indptr, indices, data, n, normed, out_degree, symmetrized)
//       -> (indptr, indices, data, diag)
//       The laplacian of a CSR matrix with int64 or float64 data (float64 when normed) as canonical CSR
//       (decisions GR50, GR53, GR54): the matrix M is the input or, symmetrized, A + A^T with sd.sparse's
//       sum rule (an exact zero sum is not stored). A node's degree is the sum of its column of M
//       (out_degree or symmetrized: its row), the diagonal left out, the terms added one by one from
//       zero in ascending index order (int64 wraps). The result holds -M(i, j) at M's off-diagonal
//       entries (a NaN keeps its sign: scipy's product by -1) and the degree at every diagonal place;
//       normed: -((M(i, j) / w(i)) / w(j)) with
//       w = sqrt(degree) (1 for degree 0), and 1 on the diagonal (0 where the degree is 0). diag is the
//       degrees (normed: w), in the data's dtype.
//   _laplacian_dense(a, normed, out_degree, symmetrized) -> (L, diag)
//       The same of a C-contiguous (n, n) int64 or float64 matrix: every entry is a value (a matrix
//       function: 0 is 0, not "no edge"), so -0.0 stands where A holds 0.0; symmetrized adds A^T first.
//
// Every row, column and degree is one fixed sum computed by one task: the row blocks and column blocks
// change nothing, so the bytes do not depend on the thread count (§3.8).
#include "array/kernels/scalar_ops.hpp"
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph.hpp"
#include "graph/graph_mod.hpp"
#include "parallel/pool.hpp"
#include "sparse/format.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace nb = nanobind;

namespace sd::graph {

namespace {

using std::int64_t;

constexpr int64_t kBlockWork = 1 << 15;  // the entries one task of a row-block pass scans

[[noreturn]] void wrong(const char* func, const char* what) {
    throw nb::type_error((std::string(func) + "(): wrong " + what).c_str());
}

const NDArray& vec(nb::handle o, const char* func, const char* what) {
    if (!is_ndarray(o)) wrong(func, what);
    const NDArray& a = as_ndarray(o);
    if (a.ndim != 1 || !a.c_contiguous) wrong(func, what);
    return a;
}

nb::object new_vector(DType t, int64_t n) {
    const int64_t shape[] = {n};
    return wrap(new_array(t, shape));
}

template <class T>
T* ptr(nb::object& o) {
    return reinterpret_cast<T*>(as_ndarray(o).data);
}

// Runs fn(lo, hi) over [0, n) in blocks of `per` (a size fixed by the input alone), on the pool when split.
template <class F>
void blocks(int64_t n, int64_t per, bool split, F&& fn) {
    per = std::max<int64_t>(per, 1);
    const int64_t ntasks = (n + per - 1) / per;
    if (!split || ntasks <= 1) {
        for (int64_t t = 0; t < ntasks; ++t) fn(t * per, std::min(n, (t + 1) * per));
        return;
    }
    parallel::parallel_for(ntasks, [&](int64_t t) { fn(t * per, std::min(n, (t + 1) * per)); });
}

// ---- the arithmetic of the two dtypes ----

template <class T>
inline T add(T a, T b) {
    if constexpr (std::is_same_v<T, int64_t>) return ops::wrap_add(a, b);
    else return a + b;
}

// -a, a NaN kept as it is (scipy's `data *= -1`: a product returns its NaN operand unchanged).
template <class T>
inline T neg(T a) {
    if constexpr (std::is_same_v<T, int64_t>) return static_cast<int64_t>(0 - static_cast<std::uint64_t>(a));
    else return a != a ? a : -a;
}

// ---- reverse Cuthill-McKee ----

// The neighbours of node i in the union of two canonical rows (ascending, each once), fn(j) for each.
template <class F>
inline void each_union(const int64_t* a, const int64_t* ae, const int64_t* b, const int64_t* be, F&& fn) {
    while (a < ae && b < be) {
        if (*a < *b) fn(*a++);
        else if (*b < *a) fn(*b++);
        else {
            fn(*a++);
            ++b;
        }
    }
    while (a < ae) fn(*a++);
    while (b < be) fn(*b++);
}

nb::object rcm(nb::handle ip, nb::handle ix, int64_t n, bool symmetric_mode) {
    const View g = view_arg(ip, ix, nb::none(), n, "_rcm");
    std::optional<sparse::Compressed> t;
    bool both = false;  // the union of the pattern and its transpose's, unless they are the same
    if (!symmetric_mode) {
        const parallel::Unlocked unlocked(n + g.nnz() >= kParallelWork);
        both = !symmetric(g, false);
    }
    if (both) t.emplace(sparse::transpose(n, n, as_ndarray(ip), as_ndarray(ix), as_ndarray(ix)));
    nb::object out = new_vector(DType::Int64, n);
    int64_t* order = ptr<int64_t>(out);
    {
        const parallel::Unlocked unlocked(n + g.nnz() >= kParallelWork);
        const int64_t* tp = t ? sparse::i64(t->indptr) : nullptr;
        const int64_t* tx = t ? sparse::i64(t->indices) : nullptr;
        auto neighbours = [&](int64_t i, auto&& fn) {
            const int64_t* a = g.indices + g.indptr[i];
            const int64_t* ae = g.indices + g.indptr[i + 1];
            if (!t) {
                for (; a < ae; ++a) fn(*a);
                return;
            }
            each_union(a, ae, tx + tp[i], tx + tp[i + 1], fn);
        };
        // The degrees: the row length of the pattern read, a stored diagonal counted twice.
        std::vector<int64_t> degree(static_cast<size_t>(n), 0);
        int64_t max_degree = 0;
        for (int64_t i = 0; i < n; ++i) {
            int64_t d = 0;
            neighbours(i, [&](int64_t j) { d += 1 + (j == i); });
            degree[static_cast<size_t>(i)] = d;
            max_degree = std::max(max_degree, d);
        }
        // The seed order: the nodes by (degree, index), a counting sort (stable, so ties keep index order).
        std::vector<int64_t> start(static_cast<size_t>(max_degree) + 2, 0), seeds(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) ++start[static_cast<size_t>(degree[static_cast<size_t>(i)]) + 1];
        for (size_t d = 1; d < start.size(); ++d) start[d] += start[d - 1];
        for (int64_t i = 0; i < n; ++i)
            seeds[static_cast<size_t>(start[static_cast<size_t>(degree[static_cast<size_t>(i)])]++)] = i;
        std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
        auto by_degree = [&](int64_t a, int64_t b) {
            const int64_t da = degree[static_cast<size_t>(a)], db = degree[static_cast<size_t>(b)];
            return da < db || (da == db && a < b);
        };
        int64_t found = 0;
        for (int64_t s : seeds) {
            if (found == n) break;
            if (seen[static_cast<size_t>(s)]) continue;
            seen[static_cast<size_t>(s)] = 1;
            order[found++] = s;
            for (int64_t head = found - 1; head < found; ++head) {
                const int64_t i = order[head], first = found;
                neighbours(i, [&](int64_t j) {
                    if (!seen[static_cast<size_t>(j)]) {
                        seen[static_cast<size_t>(j)] = 1;
                        order[found++] = j;
                    }
                });
                std::sort(order + first, order + found, by_degree);
            }
        }
        std::reverse(order, order + n);
    }
    return out;
}

// ---- the laplacian of a CSR matrix ----

// A matrix of canonical rows read in place, its values of type T: row i is entries ip[i] .. ie[i] - 1 (a
// CSR's ie is ip + 1; the symmetric sum leaves room after its rows).
template <class T>
struct Rows {
    int64_t n = 0;
    const int64_t* ip = nullptr;
    const int64_t* ie = nullptr;
    const int64_t* ix = nullptr;
    const T* v = nullptr;
};

// A + A^T as canonical rows (an exact zero sum not stored: sd.sparse's sum rule), with the transpose t: row i
// starts where the rows before it could at most end (ip[i] = the entries of rows < i in A and A^T) and ends
// at ie[i], so one merge per row writes it (no counting pass first).
template <class T>
struct SymSum {
    std::vector<int64_t> ip, ie;
    std::unique_ptr<int64_t[]> ix;  // room for every entry of A and A^T, left uninitialized
    std::unique_ptr<T[]> v;
};

template <class T>
void symmetric_sum(const Rows<T>& a, const Rows<T>& t, bool split, SymSum<T>& s) {
    const int64_t n = a.n;
    s.ip.assign(static_cast<size_t>(n) + 1, 0);
    auto merge = [&](int64_t i, auto&& emit) {
        int64_t p = a.ip[i], pe = a.ie[i], q = t.ip[i], qe = t.ie[i];
        while (p < pe || q < qe) {
            int64_t j;
            T x;
            if (q >= qe || (p < pe && a.ix[p] < t.ix[q])) {
                j = a.ix[p];
                x = a.v[p++];
            } else if (p >= pe || t.ix[q] < a.ix[p]) {
                j = t.ix[q];
                x = t.v[q++];
            } else {
                j = a.ix[p];
                x = add(a.v[p++], t.v[q++]);
            }
            if (x != T(0) || x != x) emit(j, x);
        }
    };
    for (int64_t i = 0; i <= n; ++i) s.ip[static_cast<size_t>(i)] = a.ip[i] + t.ip[i];
    s.ie.resize(static_cast<size_t>(n));
    const size_t room = static_cast<size_t>(s.ip[static_cast<size_t>(n)]);
    s.ix.reset(new int64_t[room]);
    s.v.reset(new T[room]);
    const int64_t per = std::max<int64_t>(1, kBlockWork * n / std::max<int64_t>(1, 2 * a.ip[n] + n));
    blocks(n, per, split, [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) {
            size_t k = static_cast<size_t>(s.ip[static_cast<size_t>(i)]);
            merge(i, [&](int64_t j, T x) {
                s.ix[k] = j;
                s.v[k++] = x;
            });
            s.ie[static_cast<size_t>(i)] = static_cast<int64_t>(k);
        }
    });
}

template <class T>
nb::object laplacian_csr_of(
    nb::handle iph, nb::handle ixh, nb::handle dh, int64_t n, bool normed, bool out_degree, bool symmetrized
) {
    constexpr DType kT = std::is_same_v<T, double> ? DType::Float64 : DType::Int64;
    const NDArray& ipa = as_ndarray(iph);
    const NDArray& ixa = as_ndarray(ixh);
    const NDArray& da = as_ndarray(dh);
    Rows<T> a{n, sparse::i64(ipa), sparse::i64(ipa) + 1, sparse::i64(ixa), reinterpret_cast<const T*>(da.data)};
    const int64_t nnz = a.ip[n];
    const bool split = nnz + n >= kParallelWork;
    std::optional<sparse::Compressed> tr;
    if (symmetrized) tr.emplace(sparse::transpose(n, n, ipa, ixa, da));
    nb::object op = new_vector(DType::Int64, n + 1);
    nb::object odiag = new_vector(normed ? DType::Float64 : kT, n);
    int64_t* P = ptr<int64_t>(op);
    SymSum<T> sum;
    Rows<T> m = a;
    std::vector<T> degree(static_cast<size_t>(n));
    std::vector<double> w;
    {
        const parallel::Unlocked unlocked(split);
        if (tr) {
            const Rows<T> t{
                n, sparse::i64(tr->indptr), sparse::i64(tr->indptr) + 1, sparse::i64(tr->indices),
                reinterpret_cast<const T*>(tr->data.data)
            };
            symmetric_sum(a, t, split, sum);
            m = Rows<T>{n, sum.ip.data(), sum.ie.data(), sum.ix.get(), sum.v.get()};
        }
        const int64_t mnnz = m.ip[n];
        const int64_t per = std::max<int64_t>(1, kBlockWork * n / std::max<int64_t>(1, mnnz + n));
        if (out_degree || symmetrized) {
            blocks(n, per, split, [&](int64_t lo, int64_t hi) {
                for (int64_t i = lo; i < hi; ++i) {
                    T acc = T(0);
                    for (int64_t k = m.ip[i]; k < m.ie[i]; ++k)
                        if (m.ix[k] != i) acc = add(acc, m.v[k]);
                    degree[static_cast<size_t>(i)] = acc;
                }
            });
        } else {
            // a column's terms arrive in ascending row order: one sum from zero per column
            std::fill(degree.begin(), degree.end(), T(0));
            for (int64_t i = 0; i < n; ++i)
                for (int64_t k = m.ip[i]; k < m.ie[i]; ++k) {
                    const int64_t j = m.ix[k];
                    if (j != i) degree[static_cast<size_t>(j)] = add(degree[static_cast<size_t>(j)], m.v[k]);
                }
        }
        if constexpr (std::is_same_v<T, double>) {
            if (normed) {
                w.resize(static_cast<size_t>(n));
                for (int64_t i = 0; i < n; ++i) {
                    const double d = degree[static_cast<size_t>(i)];
                    w[static_cast<size_t>(i)] = d == 0 ? 1.0 : std::sqrt(d);
                }
            }
        }
        // the rows: M's off-diagonal entries and the diagonal place
        P[0] = 0;
        blocks(n, per, split, [&](int64_t lo, int64_t hi) {
            for (int64_t i = lo; i < hi; ++i) {
                int64_t c = 1;
                for (int64_t k = m.ip[i]; k < m.ie[i]; ++k) c += m.ix[k] != i;
                P[i + 1] = c;
            }
        });
        for (int64_t i = 0; i < n; ++i) P[i + 1] += P[i];
    }
    const int64_t lnnz = P[n];
    nb::object oi = new_vector(DType::Int64, lnnz);
    nb::object ov = new_vector(normed ? DType::Float64 : kT, lnnz);
    int64_t* I = ptr<int64_t>(oi);
    {
        const parallel::Unlocked unlocked(split);
        const int64_t per = std::max<int64_t>(1, kBlockWork * n / std::max<int64_t>(1, lnnz));
        if (normed) {
            if constexpr (std::is_same_v<T, double>) {
                double* V = ptr<double>(ov);
                double* D = ptr<double>(odiag);
                blocks(n, per, split, [&](int64_t lo, int64_t hi) {
                    for (int64_t i = lo; i < hi; ++i) {
                        int64_t dst = P[i];
                        bool placed = false;
                        const double wi = w[static_cast<size_t>(i)];
                        const double one = degree[static_cast<size_t>(i)] == 0 ? 0.0 : 1.0;
                        for (int64_t k = m.ip[i]; k < m.ie[i]; ++k) {
                            const int64_t j = m.ix[k];
                            if (j == i) continue;
                            if (!placed && j > i) {
                                I[dst] = i;
                                V[dst++] = one;
                                placed = true;
                            }
                            I[dst] = j;
                            V[dst++] = neg((m.v[k] / wi) / w[static_cast<size_t>(j)]);
                        }
                        if (!placed) {
                            I[dst] = i;
                            V[dst] = one;
                        }
                        D[i] = wi;
                    }
                });
            }
        } else {
            T* V = ptr<T>(ov);
            T* D = ptr<T>(odiag);
            blocks(n, per, split, [&](int64_t lo, int64_t hi) {
                for (int64_t i = lo; i < hi; ++i) {
                    int64_t dst = P[i];
                    bool placed = false;
                    const T di = degree[static_cast<size_t>(i)];
                    for (int64_t k = m.ip[i]; k < m.ie[i]; ++k) {
                        const int64_t j = m.ix[k];
                        if (j == i) continue;
                        if (!placed && j > i) {
                            I[dst] = i;
                            V[dst++] = di;
                            placed = true;
                        }
                        I[dst] = j;
                        V[dst++] = neg(m.v[k]);
                    }
                    if (!placed) {
                        I[dst] = i;
                        V[dst] = di;
                    }
                    D[i] = di;
                }
            });
        }
    }
    return nb::make_tuple(op, oi, ov, odiag);
}

nb::object laplacian_csr(
    nb::handle ip, nb::handle ix, nb::handle dh, int64_t n, bool normed, bool out_degree, bool symmetrized
) {
    const char* func = "_laplacian_csr";
    const View g = view_arg(ip, ix, nb::none(), n, func);
    const NDArray& d = vec(dh, func, "data");
    if (d.shape[0] != g.nnz()) wrong(func, "data");
    if (d.dtype == DType::Float64) return laplacian_csr_of<double>(ip, ix, dh, n, normed, out_degree, symmetrized);
    if (d.dtype == DType::Int64 && !normed)
        return laplacian_csr_of<int64_t>(ip, ix, dh, n, normed, out_degree, symmetrized);
    wrong(func, "data");
}

// ---- the laplacian of a dense matrix ----

template <class T>
nb::object laplacian_dense_of(const NDArray& a, bool normed, bool out_degree, bool symmetrized) {
    constexpr DType kT = std::is_same_v<T, double> ? DType::Float64 : DType::Int64;
    constexpr int64_t kRows = 8;  // rows summed side by side (each its own chain, in order)
    const int64_t n = a.shape[0];
    const T* A = reinterpret_cast<const T*>(a.data);
    const int64_t shape[] = {n, n};
    nb::object out = wrap(new_array(normed ? DType::Float64 : kT, shape));
    nb::object odiag = new_vector(normed ? DType::Float64 : kT, n);
    T* L = ptr<T>(out);
    const bool split = n * n >= kParallelWork;
    const int64_t rows_per = std::max<int64_t>(kRows, kBlockWork / std::max<int64_t>(n, 1) / kRows * kRows);
    {
        const parallel::Unlocked unlocked(split);
        // M: A itself, or A + A^T written in L's place first (its diagonal is never read)
        const T* M = A;
        if (symmetrized) {
            constexpr int64_t kTile = 64;
            const int64_t nt = (n + kTile - 1) / kTile;
            blocks(nt, std::max<int64_t>(1, rows_per / kTile), split, [&](int64_t lo, int64_t hi) {
                for (int64_t bi = lo; bi < hi; ++bi)
                    for (int64_t bj = 0; bj < nt; ++bj)
                        for (int64_t i = bi * kTile, ie = std::min(n, i + kTile); i < ie; ++i)
                            for (int64_t j = bj * kTile, je = std::min(n, j + kTile); j < je; ++j)
                                L[i * n + j] = add(A[i * n + j], A[j * n + i]);
            });
            M = L;
        }
        std::vector<T> degree(static_cast<size_t>(n));
        const bool rows = out_degree || symmetrized;
        // acc[r] = the sum of row i0 + r of M (r < k), diagonal left out, from zero in ascending column
        // order; kRows rows side by side
        auto row_sums = [&](int64_t i0, int64_t k, T* acc) {
            if (k == kRows) {
                for (int64_t j = 0; j < n; ++j)
                    for (int64_t r = 0; r < kRows; ++r) {
                        const T x = M[(i0 + r) * n + j];
                        if (j != i0 + r) acc[r] = add(acc[r], x);
                    }
            } else {
                for (int64_t r = 0; r < k; ++r)
                    for (int64_t j = 0; j < n; ++j) {
                        const T x = M[(i0 + r) * n + j];
                        if (j != i0 + r) acc[r] = add(acc[r], x);
                    }
            }
        };
        const bool pool = parallel::available();
        if (!normed && !pool) {
            // One thread: one pass reads each entry once, adds it to its degree and writes it negated (each
            // degree is the same sum in the same order as in the separate passes below, which the pool
            // splits better).
            T* D = ptr<T>(odiag);
            if (rows) {
                // kRows rows summed side by side, then written while they are still in the cache
                blocks(n, rows_per, split, [&](int64_t lo, int64_t hi) {
                    for (int64_t i0 = lo; i0 < hi; i0 += kRows) {
                        const int64_t k = std::min(kRows, hi - i0);
                        T acc[kRows] = {};
                        row_sums(i0, k, acc);
                        for (int64_t r = 0; r < k; ++r) {
                            const int64_t i = i0 + r;
                            const T* src = M + i * n;
                            T* row = L + i * n;
                            for (int64_t j = 0; j < n; ++j) row[j] = neg(src[j]);
                            row[i] = acc[r];
                            D[i] = acc[r];
                        }
                    }
                });
            } else {
                // the column sums over the rows in order (an accumulator per column), each row negated as
                // it is added: each column's sum from zero in ascending row order, as in the column blocks
                const int64_t width = n;
                std::vector<T> sums(static_cast<size_t>(width), T(0));
                T* acc = sums.data();
                auto run = [](const T* __restrict src, T* __restrict dst, T* __restrict sum, int64_t len) {
                    for (int64_t c = 0; c < len; ++c) {
                        sum[c] = add(sum[c], src[c]);
                        dst[c] = neg(src[c]);
                    }
                };
                for (int64_t i = 0; i < n; ++i) {  // the diagonal entry is not added
                    const T* src = M + i * n;
                    T* dst = L + i * n;
                    run(src, dst, acc, i);
                    dst[i] = neg(src[i]);
                    run(src + i + 1, dst + i + 1, acc + i + 1, width - i - 1);
                }
                for (int64_t c = 0; c < width; ++c) {
                    L[c * n + c] = acc[c];
                    D[c] = acc[c];
                }
            }
        } else if (rows) {
            // the row sums, each from zero in ascending column order, kRows rows at a time
            blocks(n, rows_per, split, [&](int64_t lo, int64_t hi) {
                for (int64_t i0 = lo; i0 < hi; i0 += kRows) {
                    const int64_t k = std::min(kRows, hi - i0);
                    T acc[kRows] = {};
                    row_sums(i0, k, acc);
                    for (int64_t r = 0; r < k; ++r) degree[static_cast<size_t>(i0 + r)] = acc[r];
                }
            });
        } else {
            // the column sums, each from zero in ascending row order, by blocks of columns
            constexpr int64_t kCols = 32;
            blocks(n, kCols, split, [&](int64_t lo, int64_t hi) {
                T acc[kCols];
                const int64_t width = hi - lo;
                std::fill(acc, acc + width, T(0));
                for (int64_t i = 0; i < n; ++i) {
                    const T* row = M + i * n + lo;
                    if (i >= lo && i < hi) {
                        for (int64_t c = 0; c < width; ++c)
                            if (lo + c != i) acc[c] = add(acc[c], row[c]);
                    } else {
                        for (int64_t c = 0; c < width; ++c) acc[c] = add(acc[c], row[c]);
                    }
                }
                for (int64_t c = 0; c < width; ++c) degree[static_cast<size_t>(lo + c)] = acc[c];
            });
        }
        // the entries, from M (in place when M is L)
        if (normed) {
            if constexpr (std::is_same_v<T, double>) {
                std::vector<double> w(static_cast<size_t>(n));
                double* D = ptr<double>(odiag);
                for (int64_t i = 0; i < n; ++i) {
                    const double d = degree[static_cast<size_t>(i)];
                    w[static_cast<size_t>(i)] = d == 0 ? 1.0 : std::sqrt(d);
                    D[i] = w[static_cast<size_t>(i)];
                }
                blocks(n, rows_per, split, [&](int64_t lo, int64_t hi) {
                    for (int64_t i = lo; i < hi; ++i) {
                        const double* src = M + i * n;
                        double* row = L + i * n;
                        const double wi = w[static_cast<size_t>(i)];
                        for (int64_t j = 0; j < n; ++j) row[j] = neg((src[j] / wi) / w[static_cast<size_t>(j)]);
                        row[i] = degree[static_cast<size_t>(i)] == 0 ? 0.0 : 1.0;
                    }
                });
            }
        } else if (pool) {
            T* D = ptr<T>(odiag);
            blocks(n, rows_per, split, [&](int64_t lo, int64_t hi) {
                for (int64_t i = lo; i < hi; ++i) {
                    const T* src = M + i * n;
                    T* row = L + i * n;
                    for (int64_t j = 0; j < n; ++j) row[j] = neg(src[j]);
                    row[i] = degree[static_cast<size_t>(i)];
                    D[i] = degree[static_cast<size_t>(i)];
                }
            });
        }
    }
    return nb::make_tuple(out, odiag);
}

nb::object laplacian_dense(nb::handle ah, bool normed, bool out_degree, bool symmetrized) {
    const char* func = "_laplacian_dense";
    if (!is_ndarray(ah)) wrong(func, "matrix");
    const NDArray& a = as_ndarray(ah);
    if (a.ndim != 2 || !a.c_contiguous || a.shape[0] != a.shape[1]) wrong(func, "matrix");
    if (a.dtype == DType::Float64) return laplacian_dense_of<double>(a, normed, out_degree, symmetrized);
    if (a.dtype == DType::Int64 && !normed) return laplacian_dense_of<int64_t>(a, normed, out_degree, symmetrized);
    wrong(func, "matrix");
}

}  // namespace

void bind_ordering(nb::module_& m) {
    m.def("_rcm", &rcm, nb::arg("indptr"), nb::arg("indices"), nb::arg("n"), nb::arg("symmetric_mode"));
    m.def(
        "_laplacian_csr", &laplacian_csr, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("n"),
        nb::arg("normed"), nb::arg("out_degree"), nb::arg("symmetrized")
    );
    m.def(
        "_laplacian_dense", &laplacian_dense, nb::arg("a"), nb::arg("normed"), nb::arg("out_degree"),
        nb::arg("symmetrized")
    );
}

}  // namespace sd::graph
