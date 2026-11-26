// Fuzzy c-means (docs/design/cluster.md §2.3, §3.5, §4.3; decision CL27 and milestone CL2's in §1.1): Bezdek's
// alternating updates on sd's squared distance (partition.hpp: sqd, the difference form, on the data scaled by CL43's
// power of two; the rows arrive in key order, CL7).
//
//   memberships  of a point to centres at squared distances D_1 .. D_k: u_j = r_j^q / sum_l r_l^q with
//                r_l = min D / D_l and q = 1 / (m - 1) (r^q = r for m = 2), the ratios to the nearest centre so
//                nothing overflows; a point on centres (D_j = 0) belongs to those centres equally.
//   centres      the weighted means sum_i u_ij^m x_i / sum_i u_ij^m by CL48's chunk sums.
//   iteration    from the start's centres: memberships; then until the largest change of a membership is below
//                tol (at most max_iter times): centres, memberships. The objective is sum_ij u_ij^m D_ij.
//
// _core.cluster:
//   fuzzy_cmeans(X, order, k, m, init, C0, max_iter, tol, key0, key1)
//       -> (memberships (n, k) in key order, centres (k, p), objective, n_iter, converged)
//       init 1 'random' (k points drawn uniformly without replacement: element key of the point, start 0 of the
//       stream; CL41), 2 the array C0
//   fuzzy_memberships(X, C, m) -> (m, k)      the memberships of rows X to centres C (predict_proba)
#include "cluster/cluster.hpp"
#include "cluster/partition.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sd::cluster {

namespace {

constexpr int64_t kTask = 512;

// the memberships of rows X (n, p) to centres C (k, p) on one grid: U (n, k); returns the largest |U - old| when
// old is given
double memberships(
    const double* X, int64_t n, int64_t p, const double* C, int64_t k, double m, double* U, const double* old = nullptr
) {
    const double q = 1.0 / (m - 1.0);
    const int64_t nt = (n + kTask - 1) / kTask;
    constexpr int kP = spatial::kPanel;
    const int64_t np = (k + kP - 1) / kP;
    std::vector<double> panels(static_cast<size_t>(np * p * kP), 0.0);
    for (int64_t b = 0; b < np; ++b)
        spatial::pack_panel(C + b * kP * p, p, std::min<int64_t>(kP, k - b * kP), p, panels.data() + b * p * kP);
    std::vector<double> change(static_cast<size_t>(nt), 0.0);
    auto body = [&](int64_t t) {
        std::vector<double> Dall(static_cast<size_t>(4 * k));
        double buf[4 * kP];
        double ch = 0.0;
        const int64_t i0 = t * kTask, i1 = std::min(n, (t + 1) * kTask);
        for (int64_t g = i0; g < i1; g += 4) {
            const int R = static_cast<int>(std::min<int64_t>(4, i1 - g));
            const double* xr[4];
            for (int r = 0; r < 4; ++r) xr[r] = X + (g + std::min(r, R - 1)) * p;
            for (int64_t b = 0; b < np; ++b) {  // spatial's tile kernel: sqd's sums
                spatial::tile<spatial::SqOp, 4, false>(xr, panels.data() + b * p * kP, nullptr, p, buf);
                for (int r = 0; r < R; ++r)
                    for (int64_t c = 0; c < std::min<int64_t>(kP, k - b * kP); ++c) {
                        const int64_t j = b * kP + c;
                        double v = buf[r * kP + c];
                        if (!(v >= spatial::kTiny)) v = spatial::sq_finish<false>(v, xr[r], C + j * p, nullptr, p);
                        Dall[static_cast<size_t>(r * k + j)] = v;
                    }
            }
            for (int r = 0; r < R; ++r) {
                const int64_t i = g + r;
                const double* D = Dall.data() + r * k;
                double mn = kInfD;
                int64_t zeros = 0;
                for (int64_t j = 0; j < k; ++j) mn = std::min(mn, D[j]), zeros += D[j] == 0.0;
                double* u = U + i * k;
                if (zeros) {
                    const double share = 1.0 / static_cast<double>(zeros);
                    for (int64_t j = 0; j < k; ++j) u[j] = D[j] == 0.0 ? share : 0.0;
                } else {
                    double s = 0.0;
                    for (int64_t j = 0; j < k; ++j) {
                        const double rr = mn / D[j];
                        u[j] = q == 1.0 ? rr : std::pow(rr, q);
                        s += u[j];
                    }
                    for (int64_t j = 0; j < k; ++j) u[j] /= s;
                }
                if (old)
                    for (int64_t j = 0; j < k; ++j) ch = std::max(ch, std::fabs(u[j] - old[i * k + j]));
            }
        }
        change[static_cast<size_t>(t)] = ch;
    };
    if (nt > 1 && n * k * p >= kSplitWork) parallel::parallel_for(nt, body, kTask * k * (p + 20));
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    return *std::max_element(change.begin(), change.end());
}

inline double powm(double u, double m) { return m == 2.0 ? u * u : std::pow(u, m); }

// the centres from memberships U by CL48's chunk sums of u^m x and u^m
void centres(const double* X, int64_t n, int64_t p, const double* U, int64_t k, double m, double* C) {
    const int64_t width = k * p + k;
    std::vector<double> tot(static_cast<size_t>(width));
    pairwise_chunks(
        chunks_of(n), width,
        [&](int64_t c, double* out) {
            std::fill(out, out + width, 0.0);
            for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i)
                for (int64_t j = 0; j < k; ++j) {
                    const double w = powm(U[i * k + j], m);
                    out[k * p + j] += w;
                    for (int64_t d = 0; d < p; ++d) out[j * p + d] += w * X[i * p + d];
                }
        },
        tot.data(), kChunk * k * p
    );
    for (int64_t j = 0; j < k; ++j)
        for (int64_t d = 0; d < p; ++d)
            if (tot[static_cast<size_t>(k * p + j)] > 0.0)
                C[j * p + d] = tot[static_cast<size_t>(j * p + d)] / tot[static_cast<size_t>(k * p + j)];
}

double objective(const double* X, int64_t n, int64_t p, const double* U, const double* C, int64_t k, double m) {
    double t = 0.0;
    pairwise_chunks(
        chunks_of(n), 1,
        [&](int64_t c, double* out) {
            double s = 0.0;
            for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i)
                for (int64_t j = 0; j < k; ++j) s += powm(U[i * k + j], m) * sqd(X + i * p, C + j * p, p);
            out[0] = s;
        },
        &t, kChunk * k * p
    );
    return t;
}

nb::object fuzzy_py(
    nb::handle Xo,
    nb::handle ordo,
    int64_t k,
    double m,
    int init,
    nb::handle C0o,
    int64_t max_iter,
    double tol,
    int64_t key0,
    int64_t key1
) {
    int64_t n, p, no;
    const double* X0 = matrix_arg(Xo, "X", n, p);
    const int64_t* order = index_arg(ordo, "order", no);
    if (no != n || k < 1 || k > n || !(m > 1.0)) throw nb::value_error("fuzzy_cmeans");
    const double* C0 = nullptr;
    if (init == 2) {
        int64_t kk, pp;
        C0 = matrix_arg(C0o, "C0", kk, pp);
        if (kk != k || pp != p) throw nb::value_error("C0 must be (k, p)");
    }
    std::vector<double> U(static_cast<size_t>(n * k)), V(static_cast<size_t>(n * k)), C(static_cast<size_t>(k * p));
    double obj = 0.0;
    int64_t it = 0;
    bool converged = false;
    int e = 0;
    {
        const parallel::Unlocked unlocked(n * p >= kSplitWork);
        e = scale_exponent(std::max(max_abs_of(X0, n * p), C0 ? max_abs_of(C0, k * p) : 0.0));
        std::vector<double> Xs(static_cast<size_t>(n * p));
        for (int64_t r = 0; r < n; ++r)
            for (int64_t d = 0; d < p; ++d)
                Xs[static_cast<size_t>(r * p + d)] =
                    e ? spatial::scaled(X0[order[r] * p + d], e) : X0[order[r] * p + d];
        const double* X = Xs.data();
        if (C0) {
            for (int64_t q = 0; q < k * p; ++q) C[static_cast<size_t>(q)] = e ? spatial::scaled(C0[q], e) : C0[q];
        } else {
            // k points drawn uniformly without replacement: the k largest uniforms of element (0, key)
            random::Stream st(key_of(key0, key1));
            std::vector<double> u(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i)
                u[static_cast<size_t>(i)] = uniform(st, element(0, static_cast<std::uint64_t>(i)));
            std::vector<int64_t> idx(static_cast<size_t>(n));
            std::iota(idx.begin(), idx.end(), int64_t{0});
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t a, int64_t b) {
                const double x = u[static_cast<size_t>(a)], y = u[static_cast<size_t>(b)];
                return x != y ? x > y : a < b;
            });
            for (int64_t j = 0; j < k; ++j)
                std::copy(
                    X + idx[static_cast<size_t>(j)] * p, X + (idx[static_cast<size_t>(j)] + 1) * p, C.begin() + j * p
                );
        }
        memberships(X, n, p, C.data(), k, m, U.data());
        for (it = 1; it <= max_iter; ++it) {
            centres(X, n, p, U.data(), k, m, C.data());
            const double change = memberships(X, n, p, C.data(), k, m, V.data(), U.data());
            U.swap(V);
            if (change < tol) {
                converged = true;
                break;
            }
        }
        it = std::min(it, max_iter);
        obj = objective(X, n, p, U.data(), C.data(), k, m);
        if (e) {
            obj = std::ldexp(obj, -2 * e);
            for (auto& c : C) c = spatial::scaled(c, -e);
        }
    }
    return nb::make_tuple(wrap(float_array(U, n, k)), wrap(float_array(C, k, p)), obj, it, converged);
}

nb::object fuzzy_memberships_py(nb::handle Xo, nb::handle Co, double m) {
    int64_t n, p, k, pc;
    const double* X = matrix_arg(Xo, "X", n, p);
    const double* C = matrix_arg(Co, "C", k, pc);
    if (pc != p) throw nb::value_error("C");
    std::vector<double> U(static_cast<size_t>(n * k));
    {
        const parallel::Unlocked unlocked(n * k * p >= kSplitWork);
        const int e = scale_exponent(std::max(max_abs_of(X, n * p), max_abs_of(C, k * p)));
        std::vector<double> Xs, Cs;
        const double* Xg = X;
        const double* Cg = C;
        if (e) Xs = scaled_copy(X, n * p, e), Cs = scaled_copy(C, k * p, e), Xg = Xs.data(), Cg = Cs.data();
        memberships(Xg, n, p, Cg, k, m, U.data());
    }
    return wrap(float_array(U, n, k));
}

}  // namespace

void bind_fuzzy(nb::module_& m) {
    m.def(
        "fuzzy_cmeans", &fuzzy_py, nb::arg("X"), nb::arg("order"), nb::arg("k"), nb::arg("m"), nb::arg("init"),
        nb::arg("C0").none(), nb::arg("max_iter"), nb::arg("tol"), nb::arg("key0"), nb::arg("key1")
    );
    m.def("fuzzy_memberships", &fuzzy_memberships_py, nb::arg("X"), nb::arg("C"), nb::arg("m"));
}

}  // namespace sd::cluster
