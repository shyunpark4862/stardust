// Prototypes of sd.embed's kernels for milestone EM0's measurements (docs/design/embed.md §5, the five
// "시제품" items), timed and checked by tools/embed_proto.py. Not public: the private names _proto_* of _core.embed;
// the milestones that implement these paths (EM1 the Gram and the eigensolver, EM6 the Barnes-Hut tree, EM7 the
// UMAP optimizer) replace them by the area's kernels and remove this file.
//
// _proto_gram(X, form) -> (p, p)
//     X^T X of a C (n, p) matrix (the caller centres it), the full symmetric matrix. form 0: one Accelerate dsyrk
//     over all rows (Accelerate's own threads); form 1: the own fixed-order NEON kernel (tasks of 4 x 4 column
//     blocks of the upper triangle, each summing all rows in row order with two-lane FMA accumulators: the same
//     bits for any thread count); form 2: dsyrk on row chunks of 2^16 rows fixed by size, called one after
//     another and added in chunk order (thread independent); form 3: dsyrk on one row chunk per sd thread, the
//     chunks called side by side on the pool and added in thread order (D-33: the bits follow the thread count).
// _proto_eigh(C, form) -> (p,) ascending eigenvalues of a symmetric C. form 0: Accelerate dsyevd ('N'); form 1:
//     the own cyclic Jacobi method (fixed sweep order, one thread; eigenvalues only).
// _proto_bh(Y, form, theta, single) -> (forces (n, 2), sum_q, build_ns, walk_ns)
//     t-SNE's Barnes-Hut repulsion in 2-D (van der Maaten 2014, sklearn's _barnes_hut_tsne): for every point the
//     sum over the tree's nodes it accepts (max width / distance < theta, or a leaf) of N q^2 (y - c), q =
//     1 / (1 + |y - c|^2), and Z = sum q N. form 0: the sorted tree of decision EM38 (Morton keys of 2^21 cells
//     per axis in the exact bounding box, a radix sort by (key, index), the quadtree built from the sorted runs,
//     centres of mass summed in child order); form 1: an insertion quadtree like sklearn's (points inserted one
//     by one in input order, the centres of mass updated along the path). The walk is the same for both: every
//     point on its own (tasks of points), children in a fixed order; Z is the chunk sum of the per-point sums.
//     single: float coordinates, centres and forces (sklearn's float32), Z in double.
// _proto_umap(head, tail, eps, Y0, n_epochs, a, b, form, k0, k1, nsr, gamma, alpha0) -> Y
//     UMAP's layout optimisation on a fuzzy graph given as COO edges (both directions, sorted by head) with
//     umap-learn's epochs_per_sample. form 0: umap-learn's sequential SGD (optimize_layout_euclidean: edge by
//     edge, the head and tail moved in place, the negative samples move the head; one thread); form 1: the
//     synchronous batch of decision EM40 with SGD (every gradient of an epoch at the epoch's start coordinates,
//     gathered per vertex: edges by head in CSR order, by tail in CSC order, each edge's attraction computed at
//     both ends, the negative samples at the head; then Y += alpha g); form 2: the same batch with uwot's Adam
//     (beta1 0.5, beta2 0.9, eps 1e-7, bias-corrected, alpha decaying linearly). Negative sample s of edge i in
//     epoch e is element (1 + e) 2^40 + i nsr + s of the Philox stream of key (k0, k1). Forms 1 and 2 run the
//     vertices in tasks and are thread independent.
#include "array/ndarray.hpp"
#include "base/keyorder.hpp"
#include "bind/pyutil.hpp"
#include "embed/embed.hpp"
#include "parallel/pool.hpp"
#include "random/philox.hpp"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sd::embed {

namespace nb = nanobind;

namespace {

using std::int64_t;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

const NDArray& f64(nb::handle o, const char* what, int ndim) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Float64 || !a.c_contiguous || a.ndim != ndim)
        throw nb::value_error("the prototype needs C-contiguous float64 arrays of the documented rank");
    return a;
}

const NDArray& i64(nb::handle o, const char* what) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Int64 || !a.c_contiguous || a.ndim != 1)
        throw nb::value_error("the prototype needs C-contiguous 1-D int64 arrays");
    return a;
}

NDArray out_f64(std::initializer_list<int64_t> shape) {
    std::vector<int64_t> s(shape);
    return new_array(DType::Float64, s, true);
}

// ---------------------------------------------------------------- (1)-(2) the Gram and the eigensolver

void gram_neon(const double* X, int64_t n, int64_t p, double* G) {
    constexpr int64_t B = 4;
    const int64_t nb_ = (p + B - 1) / B;
    std::vector<std::pair<int64_t, int64_t>> tiles;
    for (int64_t I = 0; I < nb_; ++I)
        for (int64_t J = I; J < nb_; ++J) tiles.push_back({I, J});
    auto tile = [&](int64_t t) {
        const int64_t i0 = tiles[static_cast<size_t>(t)].first * B, j0 = tiles[static_cast<size_t>(t)].second * B;
        const int64_t ni = std::min(B, p - i0), nj = std::min(B, p - j0);
        if (ni == B && nj == B) {
            // acc[a][h]: column i0 + a against columns j0 + 2h, j0 + 2h + 1 (two lanes)
            float64x2_t acc[4][2];
            for (auto& r : acc) r[0] = r[1] = vdupq_n_f64(0.0);
            for (int64_t r = 0; r < n; ++r) {
                const double* x = X + r * p;
                const float64x2_t y0 = vld1q_f64(x + j0), y1 = vld1q_f64(x + j0 + 2);
                for (int a = 0; a < 4; ++a) {
                    const float64x2_t xa = vdupq_n_f64(x[i0 + a]);
                    acc[a][0] = vfmaq_f64(acc[a][0], xa, y0);
                    acc[a][1] = vfmaq_f64(acc[a][1], xa, y1);
                }
            }
            for (int a = 0; a < 4; ++a) {
                double v[4];
                vst1q_f64(v, acc[a][0]);
                vst1q_f64(v + 2, acc[a][1]);
                for (int c = 0; c < 4; ++c) G[(i0 + a) * p + j0 + c] = v[c];
            }
        } else {
            for (int64_t a = 0; a < ni; ++a)
                for (int64_t c = 0; c < nj; ++c) {
                    double s = 0.0;
                    for (int64_t r = 0; r < n; ++r) s = std::fma(X[r * p + i0 + a], X[r * p + j0 + c], s);
                    G[(i0 + a) * p + j0 + c] = s;
                }
        }
    };
    parallel::parallel_for(static_cast<int64_t>(tiles.size()), tile, n * 16);
    for (int64_t i = 0; i < p; ++i)
        for (int64_t j = 0; j < i; ++j) G[i * p + j] = G[j * p + i];
}

void syrk_rows(const double* X, int64_t m, int64_t p, double* C) {  // C = X^T X (upper), m rows
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, p, m, 1.0, X, p, 0.0, C, p);
}

nb::object proto_gram(nb::handle Xo, int form) {
    const NDArray& Xa = f64(Xo, "X", 2);
    const int64_t n = Xa.shape[0], p = Xa.shape[1];
    const double* X = reinterpret_cast<const double*>(Xa.data);
    NDArray Ga = out_f64({p, p});
    double* G = reinterpret_cast<double*>(Ga.data);
    {
        nb::gil_scoped_release nogil;
        if (form == 0) {
            syrk_rows(X, n, p, G);
        } else if (form == 1) {
            gram_neon(X, n, p, G);
        } else if (form == 2 || form == 3) {
            const int64_t T = form == 2 ? std::max<int64_t>(1, (n + 65535) / 65536)
                                        : std::max<int64_t>(1, std::min<int64_t>(parallel::num_threads(), n));
            const int64_t L = (n + T - 1) / T;
            std::vector<double> parts(static_cast<size_t>(T * p * p), 0.0);
            auto chunk = [&](int64_t t) {
                const int64_t r0 = t * L, r1 = std::min(n, r0 + L);
                if (r1 > r0) syrk_rows(X + r0 * p, r1 - r0, p, parts.data() + t * p * p);
            };
            if (form == 3 && T > 1) parallel::parallel_for(T, chunk, L * p * p);
            else
                for (int64_t t = 0; t < T; ++t) chunk(t);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t q = 0; q < p * p; ++q)
                    G[q] = t == 0 ? parts[static_cast<size_t>(q)] : G[q] + parts[static_cast<size_t>(t * p * p + q)];
        } else {
            throw nb::value_error("form must be 0 to 3");
        }
        if (form != 1)
            for (int64_t i = 0; i < p; ++i)
                for (int64_t j = 0; j < i; ++j) G[i * p + j] = G[j * p + i];
    }
    return wrap(std::move(Ga));
}

void jacobi_eig(std::vector<double>& A, int64_t p, double* w) {
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0, diag = 0.0;
        for (int64_t i = 0; i < p; ++i) {
            diag += A[static_cast<size_t>(i * p + i)] * A[static_cast<size_t>(i * p + i)];
            for (int64_t j = i + 1; j < p; ++j)
                off += A[static_cast<size_t>(i * p + j)] * A[static_cast<size_t>(i * p + j)];
        }
        if (off <= 1e-32 * diag) break;
        for (int64_t i = 0; i < p - 1; ++i)
            for (int64_t j = i + 1; j < p; ++j) {
                const double aij = A[static_cast<size_t>(i * p + j)];
                if (aij == 0.0) continue;
                const double aii = A[static_cast<size_t>(i * p + i)], ajj = A[static_cast<size_t>(j * p + j)];
                const double theta = (ajj - aii) / (2.0 * aij);
                const double t = std::copysign(1.0, theta) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int64_t k = 0; k < p; ++k) {  // rows i and j, then columns i and j (A symmetric, full)
                    const double aik = A[static_cast<size_t>(i * p + k)], ajk = A[static_cast<size_t>(j * p + k)];
                    A[static_cast<size_t>(i * p + k)] = c * aik - s * ajk;
                    A[static_cast<size_t>(j * p + k)] = s * aik + c * ajk;
                }
                for (int64_t k = 0; k < p; ++k) {
                    const double aki = A[static_cast<size_t>(k * p + i)], akj = A[static_cast<size_t>(k * p + j)];
                    A[static_cast<size_t>(k * p + i)] = c * aki - s * akj;
                    A[static_cast<size_t>(k * p + j)] = s * aki + c * akj;
                }
            }
    }
    for (int64_t i = 0; i < p; ++i) w[i] = A[static_cast<size_t>(i * p + i)];
    std::sort(w, w + p);
}

nb::object proto_eigh(nb::handle Co, int form) {
    const NDArray& Ca = f64(Co, "C", 2);
    const int64_t p = Ca.shape[0];
    if (Ca.shape[1] != p) throw nb::value_error("C must be square");
    std::vector<double> A(reinterpret_cast<const double*>(Ca.data), reinterpret_cast<const double*>(Ca.data) + p * p);
    NDArray wa = out_f64({p});
    double* w = reinterpret_cast<double*>(wa.data);
    {
        nb::gil_scoped_release nogil;
        if (form == 0) {
            __LAPACK_int nn = p, lda = p, info = 0, lwork = -1, liwork = -1, iq = 0;
            double q = 0;
            char jobz = 'N', uplo = 'L';
            dsyevd_(&jobz, &uplo, &nn, A.data(), &lda, w, &q, &lwork, &iq, &liwork, &info);
            lwork = static_cast<__LAPACK_int>(q);
            liwork = std::max<__LAPACK_int>(1, iq);
            std::vector<double> wk(static_cast<size_t>(lwork));
            std::vector<__LAPACK_int> iw(static_cast<size_t>(liwork));
            dsyevd_(&jobz, &uplo, &nn, A.data(), &lda, w, wk.data(), &lwork, iw.data(), &liwork, &info);
        } else {
            jacobi_eig(A, p, w);
        }
    }
    return wrap(std::move(wa));
}

// ---------------------------------------------------------------- (3) Barnes-Hut in 2-D

template <class T>
struct Node {
    T cx = 0, cy = 0;  // centre of mass
    T x0 = 0, y0 = 0, w = 0;  // the cell: corner and width
    int64_t count = 0;
    int64_t child[4] = {-1, -1, -1, -1};
    int64_t point = -1;  // a leaf's first point (sorted tree: the run start; insertion tree: the point)
};

inline std::uint64_t spread21(std::uint64_t v) {  // the 21 low bits of v to the even bits
    v &= 0x1fffff;
    v = (v | (v << 16)) & 0x0000ffff0000ffffULL;
    v = (v | (v << 8)) & 0x00ff00ff00ff00ffULL;
    v = (v | (v << 4)) & 0x0f0f0f0f0f0f0f0fULL;
    v = (v | (v << 2)) & 0x3333333333333333ULL;
    v = (v | (v << 1)) & 0x5555555555555555ULL;
    return v;
}

template <class T>
struct SortedTree {
    std::vector<Node<T>> nodes;
    std::vector<int64_t> order;  // the points sorted by (key, index)
    std::vector<std::uint64_t> keys;
    const T* Y = nullptr;

    // node over the sorted run [a, b) at Morton level `level` (0: the root cell)
    int64_t build(int64_t a, int64_t b, int level, T x0, T y0, T w) {
        const int64_t id = static_cast<int64_t>(nodes.size());
        nodes.emplace_back();
        nodes[static_cast<size_t>(id)].x0 = x0, nodes[static_cast<size_t>(id)].y0 = y0,
        nodes[static_cast<size_t>(id)].w = w;
        nodes[static_cast<size_t>(id)].count = b - a;
        if (b - a == 1 || level == 21 || keys[static_cast<size_t>(a)] == keys[static_cast<size_t>(b - 1)]) {
            T sx = 0, sy = 0;
            for (int64_t r = a; r < b; ++r) {
                const int64_t i = order[static_cast<size_t>(r)];
                sx += Y[2 * i], sy += Y[2 * i + 1];
            }
            Node<T>& nd = nodes[static_cast<size_t>(id)];
            nd.cx = sx / static_cast<T>(b - a), nd.cy = sy / static_cast<T>(b - a), nd.point = a;
            return id;
        }
        const int shift = 2 * (20 - level);
        int64_t s = a;
        const T h = w / 2;
        T sx = 0, sy = 0;
        for (int q = 0; q < 4; ++q) {
            int64_t e = s;
            while (e < b && static_cast<int>((keys[static_cast<size_t>(e)] >> shift) & 3) == q) ++e;
            if (e > s) {
                const T cx0 = x0 + ((q & 1) ? h : 0), cy0 = y0 + ((q & 2) ? h : 0);
                const int64_t c = build(s, e, level + 1, cx0, cy0, h);
                nodes[static_cast<size_t>(id)].child[q] = c;
                const Node<T>& cn = nodes[static_cast<size_t>(c)];
                sx += cn.cx * static_cast<T>(cn.count), sy += cn.cy * static_cast<T>(cn.count);
            }
            s = e;
        }
        Node<T>& nd = nodes[static_cast<size_t>(id)];
        nd.cx = sx / static_cast<T>(b - a), nd.cy = sy / static_cast<T>(b - a);
        return id;
    }

    void make(const T* Yp, int64_t n) {
        Y = Yp;
        T lo[2] = {Y[0], Y[1]}, hi[2] = {Y[0], Y[1]};
        for (int64_t i = 0; i < n; ++i)
            for (int d = 0; d < 2; ++d) lo[d] = std::min(lo[d], Y[2 * i + d]), hi[d] = std::max(hi[d], Y[2 * i + d]);
        const T w = std::max(hi[0] - lo[0], hi[1] - lo[1]) * T(1.000001) + T(1e-12);
        std::vector<std::pair<std::uint64_t, int64_t>> ki(static_cast<size_t>(n)), tmp(static_cast<size_t>(n));
        const double cells = 2097152.0;
        for (int64_t i = 0; i < n; ++i) {
            const auto cx = static_cast<std::uint64_t>(std::min(cells - 1, std::floor((Y[2 * i] - lo[0]) / w * cells)));
            const auto cy =
                static_cast<std::uint64_t>(std::min(cells - 1, std::floor((Y[2 * i + 1] - lo[1]) / w * cells)));
            ki[static_cast<size_t>(i)] = {spread21(cx) | (spread21(cy) << 1), i};
        }
        // LSD radix sort of the 42-bit keys, 14 bits per pass (stable: the index order kept among equal keys)
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<int64_t> cnt((1 << 14) + 1, 0);
            const int sh = 14 * pass;
            for (auto& e : ki) ++cnt[((e.first >> sh) & 0x3fff) + 1];
            for (size_t v = 1; v < cnt.size(); ++v) cnt[v] += cnt[v - 1];
            for (auto& e : ki) tmp[static_cast<size_t>(cnt[(e.first >> sh) & 0x3fff]++)] = e;
            ki.swap(tmp);
        }
        order.resize(static_cast<size_t>(n));
        keys.resize(static_cast<size_t>(n));
        for (int64_t r = 0; r < n; ++r)
            order[static_cast<size_t>(r)] = ki[static_cast<size_t>(r)].second,
            keys[static_cast<size_t>(r)] = ki[static_cast<size_t>(r)].first;
        nodes.clear();
        nodes.reserve(static_cast<size_t>(2 * n));
        build(0, n, 0, lo[0], lo[1], w);
    }
};

template <class T>
struct InsertTree {  // sklearn's _QuadTree: points inserted one by one, centres of mass updated on the way down
    std::vector<Node<T>> nodes;
    const T* Y = nullptr;

    int64_t add(T x0, T y0, T w) {
        nodes.emplace_back();
        Node<T>& nd = nodes.back();
        nd.x0 = x0, nd.y0 = y0, nd.w = w;
        return static_cast<int64_t>(nodes.size()) - 1;
    }

    void insert(int64_t i) {
        const T px = Y[2 * i], py = Y[2 * i + 1];
        int64_t c = 0;
        for (int depth = 0;; ++depth) {
            if (nodes[static_cast<size_t>(c)].count == 0) {
                Node<T>& nd = nodes[static_cast<size_t>(c)];
                nd.count = 1, nd.point = i, nd.cx = px, nd.cy = py;
                return;
            }
            {
                Node<T>& nd = nodes[static_cast<size_t>(c)];
                const T k = static_cast<T>(nd.count);
                nd.cx = (nd.cx * k + px) / (k + 1), nd.cy = (nd.cy * k + py) / (k + 1);
            }
            const Node<T> cur = nodes[static_cast<size_t>(c)];
            const bool leaf = cur.child[0] < 0 && cur.child[1] < 0 && cur.child[2] < 0 && cur.child[3] < 0;
            const T h = cur.w / 2;
            if (leaf) {
                const int64_t j = cur.point;
                if ((Y[2 * j] == px && Y[2 * j + 1] == py) || depth > 40) {  // a duplicate: one leaf
                    ++nodes[static_cast<size_t>(c)].count;
                    return;
                }
                const int qj = (Y[2 * j] >= cur.x0 + h ? 1 : 0) | (Y[2 * j + 1] >= cur.y0 + h ? 2 : 0);
                const int64_t cc = add(cur.x0 + ((qj & 1) ? h : 0), cur.y0 + ((qj & 2) ? h : 0), h);
                Node<T>& child = nodes[static_cast<size_t>(cc)];
                child.count = cur.count, child.point = j, child.cx = Y[2 * j], child.cy = Y[2 * j + 1];
                nodes[static_cast<size_t>(c)].child[qj] = cc;
                nodes[static_cast<size_t>(c)].point = -1;
            }
            ++nodes[static_cast<size_t>(c)].count;
            const int q = (px >= cur.x0 + h ? 1 : 0) | (py >= cur.y0 + h ? 2 : 0);
            const int64_t next = nodes[static_cast<size_t>(c)].child[q];
            if (next < 0) {
                const int64_t cc = add(cur.x0 + ((q & 1) ? h : 0), cur.y0 + ((q & 2) ? h : 0), h);
                Node<T>& child = nodes[static_cast<size_t>(cc)];
                child.count = 1, child.point = i, child.cx = px, child.cy = py;
                nodes[static_cast<size_t>(c)].child[q] = cc;
                return;
            }
            c = next;
        }
    }

    void make(const T* Yp, int64_t n) {
        Y = Yp;
        T lo[2] = {Y[0], Y[1]}, hi[2] = {Y[0], Y[1]};
        for (int64_t i = 0; i < n; ++i)
            for (int d = 0; d < 2; ++d) lo[d] = std::min(lo[d], Y[2 * i + d]), hi[d] = std::max(hi[d], Y[2 * i + d]);
        const T w = std::max(hi[0] - lo[0], hi[1] - lo[1]) * T(1.000001) + T(1e-12);
        nodes.clear();
        nodes.reserve(static_cast<size_t>(4 * n));
        add(lo[0], lo[1], w);
        for (int64_t i = 0; i < n; ++i) insert(i);
    }
};

// The repulsion of point i: sum over accepted nodes of N q^2 (y - c); returns sum N q (Z's part).
template <class T>
double walk(const std::vector<Node<T>>& nodes, T px, T py, T theta2, T* f) {
    int64_t stack[256];
    int top = 0;
    stack[top++] = 0;
    double z = 0.0;
    T fx = 0, fy = 0;
    while (top > 0) {
        const Node<T>& nd = nodes[static_cast<size_t>(stack[--top])];
        if (nd.count == 0) continue;
        const T dx = px - nd.cx, dy = py - nd.cy, d2 = dx * dx + dy * dy;
        const bool leaf = nd.child[0] < 0 && nd.child[1] < 0 && nd.child[2] < 0 && nd.child[3] < 0;
        if (leaf || nd.w * nd.w < theta2 * d2) {
            if (d2 == 0) continue;  // the point itself (or its duplicates): no force
            const T q = T(1) / (T(1) + d2), nq = static_cast<T>(nd.count) * q;
            z += static_cast<double>(nq);
            fx += nq * q * dx, fy += nq * q * dy;
            continue;
        }
        for (int c = 3; c >= 0; --c)
            if (nd.child[c] >= 0) stack[top++] = nd.child[c];
    }
    f[0] = fx, f[1] = fy;
    return z;
}

template <class T>
nb::object bh_run(const double* Yd, int64_t n, int form, double theta) {
    std::vector<T> Y(static_cast<size_t>(2 * n));
    for (int64_t q = 0; q < 2 * n; ++q) Y[static_cast<size_t>(q)] = static_cast<T>(Yd[q]);
    NDArray Fa = out_f64({n, 2});
    double* F = reinterpret_cast<double*>(Fa.data);
    double zsum = 0.0;
    int64_t build_ns = 0, walk_ns = 0;
    {
        nb::gil_scoped_release nogil;
        const int64_t t0 = now_ns();
        SortedTree<T> st;
        InsertTree<T> it;
        if (form == 0) st.make(Y.data(), n);
        else it.make(Y.data(), n);
        const std::vector<Node<T>>& nodes = form == 0 ? st.nodes : it.nodes;
        const int64_t t1 = now_ns();
        std::vector<double> zi(static_cast<size_t>(n));
        const T theta2 = static_cast<T>(theta * theta);
        constexpr int64_t kTask = 256;
        const int64_t ntask = (n + kTask - 1) / kTask;
        parallel::parallel_for(
            ntask,
            [&](int64_t t) {
                for (int64_t i = t * kTask; i < std::min(n, (t + 1) * kTask); ++i) {
                    T f[2];
                    zi[static_cast<size_t>(i)] =
                        walk(nodes, Y[static_cast<size_t>(2 * i)], Y[static_cast<size_t>(2 * i + 1)], theta2, f);
                    F[2 * i] = static_cast<double>(f[0]), F[2 * i + 1] = static_cast<double>(f[1]);
                }
            },
            kTask * 4000
        );
        keyorder::pairwise_chunks(
            keyorder::chunks_of(n), 1,
            [&](int64_t c, double* out) {
                double s = 0.0;
                for (int64_t i = c * keyorder::kChunk; i < std::min(n, (c + 1) * keyorder::kChunk); ++i)
                    s += zi[static_cast<size_t>(i)];
                out[0] = s;
            },
            &zsum
        );
        const int64_t t2 = now_ns();
        build_ns = t1 - t0, walk_ns = t2 - t1;
    }
    return nb::make_tuple(wrap(std::move(Fa)), zsum, build_ns, walk_ns);
}

nb::object proto_bh(nb::handle Yo, int form, double theta, bool single) {
    const NDArray& Ya = f64(Yo, "Y", 2);
    if (Ya.shape[1] != 2) throw nb::value_error("Y must be (n, 2)");
    const auto* Y = reinterpret_cast<const double*>(Ya.data);
    return single ? bh_run<float>(Y, Ya.shape[0], form, theta) : bh_run<double>(Y, Ya.shape[0], form, theta);
}

// ---------------------------------------------------------------- (4) the UMAP layout

inline double clip4(double v) { return v > 4.0 ? 4.0 : (v < -4.0 ? -4.0 : v); }

nb::object proto_umap(
    nb::handle head_o,
    nb::handle tail_o,
    nb::handle eps_o,
    nb::handle Y0o,
    int64_t n_epochs,
    double a,
    double b,
    int form,
    int64_t k0,
    int64_t k1,
    int64_t nsr,
    double gamma,
    double alpha0
) {
    const NDArray& ha = i64(head_o, "head");
    const NDArray& ta = i64(tail_o, "tail");
    const NDArray& ea = f64(eps_o, "epochs_per_sample", 1);
    const NDArray& Ya0 = f64(Y0o, "Y0", 2);
    const int64_t m = ha.shape[0], n = Ya0.shape[0], dim = Ya0.shape[1];
    const auto* head = reinterpret_cast<const int64_t*>(ha.data);
    const auto* tail = reinterpret_cast<const int64_t*>(ta.data);
    const auto* eps = reinterpret_cast<const double*>(ea.data);
    NDArray Ya = out_f64({n, dim});
    double* Y = reinterpret_cast<double*>(Ya.data);
    std::copy(reinterpret_cast<const double*>(Ya0.data), reinterpret_cast<const double*>(Ya0.data) + n * dim, Y);
    {
        nb::gil_scoped_release nogil;
        const random::Key key = keyorder::key_of(k0, k1);
        std::vector<double> next(eps, eps + m), eneg(static_cast<size_t>(m)), next_neg(static_cast<size_t>(m));
        for (int64_t i = 0; i < m; ++i)
            eneg[static_cast<size_t>(i)] = eps[i] / static_cast<double>(nsr),
            next_neg[static_cast<size_t>(i)] = eneg[static_cast<size_t>(i)];
        auto attract = [&](const double* cur, const double* oth, double* g) {  // the head's gradient step direction
            double d2 = 0.0;
            for (int64_t d = 0; d < dim; ++d) d2 += (cur[d] - oth[d]) * (cur[d] - oth[d]);
            double coeff = 0.0;
            if (d2 > 0.0) coeff = -2.0 * a * b * std::pow(d2, b - 1.0) / (a * std::pow(d2, b) + 1.0);
            for (int64_t d = 0; d < dim; ++d) g[d] = clip4(coeff * (cur[d] - oth[d]));
        };
        auto repel = [&](const double* cur, const double* oth, double* g) {
            double d2 = 0.0;
            for (int64_t d = 0; d < dim; ++d) d2 += (cur[d] - oth[d]) * (cur[d] - oth[d]);
            double coeff = 0.0;
            if (d2 > 0.0) coeff = 2.0 * gamma * b / ((0.001 + d2) * (a * std::pow(d2, b) + 1.0));
            for (int64_t d = 0; d < dim; ++d) g[d] = coeff > 0.0 ? clip4(coeff * (cur[d] - oth[d])) : 0.0;
        };
        if (form == 0) {
            random::Stream st(key);
            double g[16];
            for (int64_t e = 0; e < n_epochs; ++e) {
                const double alpha = alpha0 * (1.0 - static_cast<double>(e) / static_cast<double>(n_epochs));
                for (int64_t i = 0; i < m; ++i) {
                    if (next[static_cast<size_t>(i)] > static_cast<double>(e)) continue;
                    double* cur = Y + head[i] * dim;
                    double* oth = Y + tail[i] * dim;
                    attract(cur, oth, g);
                    for (int64_t d = 0; d < dim; ++d) cur[d] += g[d] * alpha, oth[d] -= g[d] * alpha;
                    next[static_cast<size_t>(i)] += eps[i];
                    const auto nneg = static_cast<int64_t>(
                        (static_cast<double>(e) - next_neg[static_cast<size_t>(i)]) / eneg[static_cast<size_t>(i)]
                    );
                    for (int64_t s = 0; s < nneg; ++s) {
                        const int64_t k = keyorder::draw(
                            st,
                            keyorder::element(
                                1 + static_cast<std::uint64_t>(e), static_cast<std::uint64_t>(i * nsr + s)
                            ),
                            n
                        );
                        if (k == head[i]) continue;
                        repel(cur, Y + k * dim, g);
                        for (int64_t d = 0; d < dim; ++d) cur[d] += g[d] * alpha;
                    }
                    next_neg[static_cast<size_t>(i)] += static_cast<double>(nneg) * eneg[static_cast<size_t>(i)];
                }
            }
        } else {
            // CSR by head (the edges come sorted by head) and the edges sorted by tail (stable): the gather orders
            std::vector<int64_t> hp(static_cast<size_t>(n + 1), 0), tp(static_cast<size_t>(n + 1), 0),
                by_tail(static_cast<size_t>(m));
            for (int64_t i = 0; i < m; ++i)
                ++hp[static_cast<size_t>(head[i] + 1)], ++tp[static_cast<size_t>(tail[i] + 1)];
            for (int64_t v = 0; v < n; ++v)
                hp[static_cast<size_t>(v + 1)] += hp[static_cast<size_t>(v)],
                    tp[static_cast<size_t>(v + 1)] += tp[static_cast<size_t>(v)];
            {
                std::vector<int64_t> fill(tp.begin(), tp.end() - 1);
                for (int64_t i = 0; i < m; ++i) by_tail[static_cast<size_t>(fill[static_cast<size_t>(tail[i])]++)] = i;
            }
            std::vector<double> G(static_cast<size_t>(n * dim)), Y0(static_cast<size_t>(n * dim));
            std::vector<double> M1(static_cast<size_t>(n * dim), 0.0), M2(static_cast<size_t>(n * dim), 0.0);
            std::vector<int64_t> nneg(static_cast<size_t>(m));
            std::vector<char> active(static_cast<size_t>(m));
            const double b1 = 0.5, b2 = 0.9, aeps = 1e-7;
            constexpr int64_t kTask = 512;
            const int64_t ntask = (n + kTask - 1) / kTask;
            for (int64_t e = 0; e < n_epochs; ++e) {
                const double alpha = alpha0 * (1.0 - static_cast<double>(e) / static_cast<double>(n_epochs));
                for (int64_t i = 0; i < m; ++i) {  // the epoch's schedule (exact, the same as form 0's)
                    active[static_cast<size_t>(i)] = next[static_cast<size_t>(i)] <= static_cast<double>(e);
                    nneg[static_cast<size_t>(i)] = 0;
                    if (!active[static_cast<size_t>(i)]) continue;
                    next[static_cast<size_t>(i)] += eps[i];
                    nneg[static_cast<size_t>(i)] = static_cast<int64_t>(
                        (static_cast<double>(e) - next_neg[static_cast<size_t>(i)]) / eneg[static_cast<size_t>(i)]
                    );
                    next_neg[static_cast<size_t>(i)] +=
                        static_cast<double>(nneg[static_cast<size_t>(i)]) * eneg[static_cast<size_t>(i)];
                }
                std::copy(Y, Y + n * dim, Y0.begin());
                parallel::parallel_for(
                    ntask,
                    [&](int64_t t) {
                        random::Stream st(key);
                        double g[16], acc[16];
                        for (int64_t v = t * kTask; v < std::min(n, (t + 1) * kTask); ++v) {
                            for (int64_t d = 0; d < dim; ++d) acc[d] = 0.0;
                            const double* yv = Y0.data() + v * dim;
                            for (int64_t q = hp[static_cast<size_t>(v)]; q < hp[static_cast<size_t>(v + 1)]; ++q) {
                                if (!active[static_cast<size_t>(q)]) continue;
                                attract(yv, Y0.data() + tail[q] * dim, g);
                                for (int64_t d = 0; d < dim; ++d) acc[d] += g[d];
                                for (int64_t s = 0; s < nneg[static_cast<size_t>(q)]; ++s) {
                                    const int64_t k = keyorder::draw(
                                        st,
                                        keyorder::element(
                                            1 + static_cast<std::uint64_t>(e), static_cast<std::uint64_t>(q * nsr + s)
                                        ),
                                        n
                                    );
                                    if (k == v) continue;
                                    repel(yv, Y0.data() + k * dim, g);
                                    for (int64_t d = 0; d < dim; ++d) acc[d] += g[d];
                                }
                            }
                            for (int64_t r = tp[static_cast<size_t>(v)]; r < tp[static_cast<size_t>(v + 1)]; ++r) {
                                const int64_t q = by_tail[static_cast<size_t>(r)];
                                if (!active[static_cast<size_t>(q)]) continue;
                                attract(Y0.data() + head[q] * dim, yv, g);  // the head's step; the tail moves by -g
                                for (int64_t d = 0; d < dim; ++d) acc[d] -= g[d];
                            }
                            double* yo = Y + v * dim;
                            if (form == 1) {
                                for (int64_t d = 0; d < dim; ++d) yo[d] = yv[d] + alpha * acc[d];
                            } else {
                                const double tt = static_cast<double>(e + 1);
                                const double scale =
                                    alpha * std::sqrt(1.0 - std::pow(b2, tt)) / (1.0 - std::pow(b1, tt));
                                const double epsc = aeps * std::sqrt(1.0 - std::pow(b2, tt));
                                for (int64_t d = 0; d < dim; ++d) {
                                    double& m1 = M1[static_cast<size_t>(v * dim + d)];
                                    double& m2 = M2[static_cast<size_t>(v * dim + d)];
                                    m1 += (1.0 - b1) * (acc[d] - m1);
                                    m2 += (1.0 - b2) * (acc[d] * acc[d] - m2);
                                    yo[d] = yv[d] + scale * m1 / (std::sqrt(m2) + epsc);
                                }
                            }
                        }
                    },
                    kTask * 2000
                );
            }
        }
    }
    return wrap(std::move(Ya));
}

}  // namespace

void bind_prototype(nb::module_& m) {
    m.def("_proto_gram", &proto_gram, nb::arg("X"), nb::arg("form"));
    m.def("_proto_eigh", &proto_eigh, nb::arg("C"), nb::arg("form"));
    m.def("_proto_bh", &proto_bh, nb::arg("Y"), nb::arg("form"), nb::arg("theta"), nb::arg("single"));
    m.def(
        "_proto_umap", &proto_umap, nb::arg("head"), nb::arg("tail"), nb::arg("eps"), nb::arg("Y0"),
        nb::arg("n_epochs"), nb::arg("a"), nb::arg("b"), nb::arg("form"), nb::arg("k0"), nb::arg("k1"), nb::arg("nsr"),
        nb::arg("gamma"), nb::arg("alpha0")
    );
}

}  // namespace sd::embed
