// RBFInterpolator of sd.interpolate (docs/design/interpolate.md §3.8, §6; decisions IP25, IP29, IP90-IP96): the
// kernels, the saddle-point system and its LDL^T solve, the evaluation, and the local systems of neighbors=k.
//
//   kernels      scipy's eight, phi(r) with r = epsilon ||x - y|| (decision IP91: epsilon times the distance; scipy
//                takes the distance of epsilon x and epsilon y): linear -r, thin_plate_spline r^2 log r (0 at r = 0),
//                cubic r^3, quintic -r^5, multiquadric -sqrt(r^2 + 1), inverse_multiquadric 1 / sqrt(r^2 + 1),
//                inverse_quadratic 1 / (r^2 + 1), gaussian exp(-r^2); products in the order (r r) r, ((r r)(r r)) r;
//                log and exp are array/kernels/vmath.hpp's (decision IP100), so two lanes give one entry's bits.
//                The distance is sd.spatial's euclidean pair kernel (spatial/metric.hpp: the fma chain of the squared
//                differences, the careful path for a tiny or huge sum), so a pair has the bits cdist gives it
//   tail         the monomials of degree <= `degree` (scipy's order, given as the (q, m) powers) of the shifted and
//                scaled coordinates xhat = (x - shift) / scale, shift = 0.5 max + 0.5 min, scale = (max - min) / 2 over
//                the points of the system (a zero scale is 1), as scipy; a monomial is the product of its factors
//                one by one in the order of the coordinates
//   system       [[K + diag(smoothing), P], [P^T, 0]] [a; b] = [d; 0], N = n + q, built in the lower triangle
//                (column-major; the matrix is symmetric) column by column in parallel, every entry its own kernel
//   checks       before the solve, the rank of P (n x q) as numpy's matrix_rank (singular values by gesdd, the
//                tolerance s_max max(n, q) eps): a rank below q is reported (scipy finds it only after a failed
//                solve and otherwise returns values of size 1e11; decision IP92)
//   solve        Bunch-Kaufman (lapack sytrf, lower), then dsytrs's steps written out (ldlt_solve, decision IP101)
//                per column of the N x s right side (complex values are two real columns: the matrix is real);
//                sytrf's exact zero pivot, or a solution that is not finite, is reported singular. scipy's dsysv is
//                the same factorization
//   evaluation   per point, sum_j phi(r_j) c_j + sum_k mono_k b_k: one fma per term in the order of the data points,
//                then the monomials (fixed by the sizes, so by no split); points split into pieces in parallel, four
//                points side by side in NEON lanes (eval_lanes; the system's columns two entries at a time)
//   neighbors    the k nearest data points of each point (the caller's sd.spatial.KDTree.query: sorted by (distance,
//                index), so ties go to the smaller index) as a sorted index set; the points are sorted by their set
//                (then by their number), each distinct set is solved once on its own points (its own shift and
//                scale) and its points evaluated with it, the sets in parallel. A point's value depends on its set
//                alone: not on the other points, their order, or the thread count
//
// Private bindings (the Python layer checks the arguments; values (n, s) float64 or complex128):
//   rbf_duplicate(y, smoothing) -> (i, j)   two equal data points both with zero smoothing (j the smallest such
//                                            second point, i the first equal to it), or (-1, -1)
//   rbf_fit(y, d, smoothing, kernel, epsilon, powers) -> (coeffs (N, s), shift, scale, status, rank)
//       status 0 solved, 1 P's rank below q (rank), 2 sytrf's exact zero pivot, 3 a coefficient not finite
//   rbf_matrix(y, smoothing, kernel, epsilon, powers) -> the computed system (N, N), both triangles (for the tests)
//   rbf_eval(x, y, coeffs, kernel, epsilon, powers, shift, scale) -> (Q, s)
//   rbf_local(x, y, d, smoothing, kernel, epsilon, powers, idx) -> (out (Q, s), status, point, rank)
//       idx (Q, k) the neighbours of each point; status as rbf_fit's for the failing set of the smallest point number

#include "interpolate/interpolate.hpp"

#include "array/kernels/vmath.hpp"
#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "linalg/lapack.hpp"
#include "parallel/pool.hpp"
#include "spatial/metric.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sd::interpolate {

namespace nb = nanobind;

namespace {

using linalg::lint;
namespace lapack = linalg::lapack;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kEps = 0x1p-52;  // numpy's finfo(float64).eps (matrix_rank's tolerance)

enum Kernel : int {
    kLinear = 0,
    kThinPlate,
    kCubic,
    kQuintic,
    kMultiquadric,
    kInverseMultiquadric,
    kInverseQuadratic,
    kGaussian,
    kKernels
};

// log and exp are the array area's NEON kernels' scalar twins (array/kernels/vmath.hpp: within 1 ULP, the
// contract's assumption L), so the evaluation's lanes (phi_lanes) give the bits of this scalar form (decision IP100).
template <int K>
inline double phi(double r) {
    if constexpr (K == kLinear) return -r;
    else if constexpr (K == kThinPlate) return r == 0 ? 0.0 : (r * r) * vmath::log1(r);
    else if constexpr (K == kCubic) return (r * r) * r;
    else if constexpr (K == kQuintic) {
        const double r2 = r * r;
        return -((r2 * r2) * r);
    } else if constexpr (K == kMultiquadric) return -std::sqrt(r * r + 1.0);
    else if constexpr (K == kInverseMultiquadric) return 1.0 / std::sqrt(r * r + 1.0);
    else if constexpr (K == kInverseQuadratic) return 1.0 / (r * r + 1.0);
    else return vmath::exp1(-(r * r));
}

// phi of two r at once: the same operations as phi (so the same bits); the lanes vmath's log and exp do not cover
// (r = 0, subnormal or not finite for log; |r^2| > 708 for exp) take phi.
template <int K>
inline float64x2_t phi_lanes(float64x2_t r) {
    if constexpr (K == kLinear) return vnegq_f64(r);
    else if constexpr (K == kCubic) return vmulq_f64(vmulq_f64(r, r), r);
    else if constexpr (K == kQuintic) {
        const float64x2_t r2 = vmulq_f64(r, r);
        return vnegq_f64(vmulq_f64(vmulq_f64(r2, r2), r));
    } else if constexpr (K == kMultiquadric) return vnegq_f64(vsqrtq_f64(vaddq_f64(vmulq_f64(r, r), vdupq_n_f64(1.0))));
    else if constexpr (K == kInverseMultiquadric)
        return vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(vaddq_f64(vmulq_f64(r, r), vdupq_n_f64(1.0))));
    else if constexpr (K == kInverseQuadratic)
        return vdivq_f64(vdupq_n_f64(1.0), vaddq_f64(vmulq_f64(r, r), vdupq_n_f64(1.0)));
    else {
        const uint64x2_t ok = K == kThinPlate ? vmath::log_ok(vreinterpretq_u64_f64(r))
                                              : vcaleq_f64(vmulq_f64(r, r), vdupq_n_f64(vmath::kExpMax));
        if ((vgetq_lane_u64(ok, 0) & vgetq_lane_u64(ok, 1)) == 0) [[unlikely]] {
            const double v[2] = {phi<K>(vgetq_lane_f64(r, 0)), phi<K>(vgetq_lane_f64(r, 1))};
            return vld1q_f64(v);
        }
        if constexpr (K == kThinPlate) return vmulq_f64(vmulq_f64(r, r), vmath::log_lanes(r));
        else return vmath::exp_lanes(vnegq_f64(vmulq_f64(r, r)));
    }
}

// f(std::integral_constant<int, K>{}) for the kernel id k (checked by the caller)
template <typename F>
void with_kernel(int k, F&& f) {
    switch (k) {
        case kLinear: f(std::integral_constant<int, kLinear>{}); break;
        case kThinPlate: f(std::integral_constant<int, kThinPlate>{}); break;
        case kCubic: f(std::integral_constant<int, kCubic>{}); break;
        case kQuintic: f(std::integral_constant<int, kQuintic>{}); break;
        case kMultiquadric: f(std::integral_constant<int, kMultiquadric>{}); break;
        case kInverseMultiquadric: f(std::integral_constant<int, kInverseMultiquadric>{}); break;
        case kInverseQuadratic: f(std::integral_constant<int, kInverseQuadratic>{}); break;
        default: f(std::integral_constant<int, kGaussian>{}); break;
    }
}

// sd.spatial's euclidean distance of two points (the pair kernel cdist uses)
inline double distance(const double* a, const double* b, std::int64_t m) {
    double s = 0.0;
    spatial::pair_scalar<spatial::SqOp, false>(a, b, nullptr, m, &s);
    return spatial::euclid_finish<false>(s, a, b, nullptr, m);
}

// fn(begin, end) over [0, total) in pieces of at least `grain`, on the pool when the work (units of a float64
// operation) is large enough; the pieces depend on the sizes only (as polyint.cpp's).
template <typename F>
void split_range(std::int64_t total, std::int64_t work, std::int64_t grain, const F& fn) {
    if (total <= 0) return;
    if (work >= kParallelThreshold && parallel::available() && total > grain) {
        const std::int64_t per_unit = std::max<std::int64_t>(1, work / total);
        const std::int64_t per = std::max(grain, kParallelTask * 4 / per_unit);
        const std::int64_t tasks = (total + per - 1) / per;
        if (tasks > 1) {
            parallel::parallel_for(
                tasks, [&](std::int64_t t) { fn(t * per, std::min(total, (t + 1) * per)); }, work_ns(work / tasks)
            );
            return;
        }
    }
    const parallel::Unlocked unlocked(work >= kParallelThreshold);
    fn(0, total);
}

// Lanes whose sum of squares spatial::needs_care leaves alone: kTiny <= s <= kHuge, or s NaN.
inline uint64x2_t sum_fine(float64x2_t s, float64x2_t tiny, float64x2_t huge) {
    const uint64x2_t in = vandq_u64(vcgeq_f64(s, tiny), vcleq_f64(s, huge));
    const uint64x2_t nan = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(s, s))));
    return vorrq_u64(in, nan);
}

// ---------------------------------------------------------------- arguments

struct Mat {  // a C-contiguous (rows, cols) float64 array
    const double* p;
    std::int64_t rows, cols;
};

Mat mat_arg(nb::handle o, const char* name) {
    const NDArray& a = require_ndarray(o, name);
    if (a.ndim != 2 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error((std::string("'") + name + "' must be a C-contiguous 2-D float64 array").c_str());
    return {reinterpret_cast<const double*>(a.data), a.shape[0], a.shape[1]};
}

const double* vec_arg(nb::handle o, const char* name, std::int64_t n) {
    const NDArray& a = require_ndarray(o, name);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[0] != n)
        throw nb::value_error((std::string("'") + name + "' must be a C-contiguous 1-D float64 array of length " +
                               std::to_string(n))
                                  .c_str());
    return reinterpret_cast<const double*>(a.data);
}

struct Ints {
    const std::int64_t* p;
    std::int64_t rows, cols;
};

Ints ints_arg(nb::handle o, const char* name) {
    const NDArray& a = require_ndarray(o, name);
    if (a.ndim != 2 || !a.c_contiguous || a.dtype != DType::Int64)
        throw nb::value_error((std::string("'") + name + "' must be a C-contiguous 2-D int64 array").c_str());
    return {reinterpret_cast<const std::int64_t*>(a.data), a.shape[0], a.shape[1]};
}

// (n, s) float64 or complex128 values as doubles: D per row (s, or 2 s)
struct Cols {
    const double* p;
    std::int64_t n, ncol, D;
    DType dtype;
};

Cols cols_arg(nb::handle o, const char* name, std::int64_t rows) {
    const NDArray& a = require_ndarray(o, name);
    if (a.ndim != 2 || !a.c_contiguous || a.shape[0] != rows)
        throw nb::value_error((std::string("'") + name + "' must be a C-contiguous (n, s) array").c_str());
    if (a.dtype != DType::Float64 && a.dtype != DType::Complex128)
        throw nb::type_error((std::string("'") + name + "' must be float64 or complex128").c_str());
    const std::int64_t per = a.dtype == DType::Complex128 ? 2 : 1;
    return {reinterpret_cast<const double*>(a.data), a.shape[0], a.shape[1], a.shape[1] * per, a.dtype};
}

int kernel_arg(int kernel) {
    if (kernel < 0 || kernel >= kKernels) throw nb::value_error("'kernel' must be a kernel id in [0, 8)");
    return kernel;
}

// ---------------------------------------------------------------- the system

// The polynomial tail's geometry of the points y[rows[0..n)] (rows null: 0..n): shift and scale per axis.
void shift_scale(
    const double* y, const std::int64_t* rows, std::int64_t n, std::int64_t m, double* shift, double* scale
) {
    for (std::int64_t l = 0; l < m; ++l) {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (std::int64_t i = 0; i < n; ++i) {
            const double v = y[(rows ? rows[i] : i) * m + l];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        if (n == 0) lo = hi = 0.0;
        // (hi + lo) / 2 and (hi - lo) / 2 as scipy, without overflow: the halves first (exact but in the subnormals)
        shift[l] = 0.5 * hi + 0.5 * lo;
        const double s = 0.5 * hi - 0.5 * lo;
        scale[l] = s == 0.0 ? 1.0 : s;
    }
}

// The monomials of x: out[k] = prod_l xhat_l^powers[k][l], xhat_l = (x_l - shift_l) / scale_l
inline void monomials(
    const double* x,
    std::int64_t m,
    const std::int64_t* pw,
    std::int64_t q,
    const double* shift,
    const double* scale,
    double* xhat,
    double* out
) {
    for (std::int64_t l = 0; l < m; ++l) xhat[l] = (x[l] - shift[l]) / scale[l];
    for (std::int64_t k = 0; k < q; ++k) {
        double v = 1.0;
        for (std::int64_t l = 0; l < m; ++l)
            for (std::int64_t e = 0; e < pw[k * m + l]; ++e) v *= xhat[l];
        out[k] = v;
    }
}

// The lower triangle of the system of the points y[rows[0..n)] (rows null: 0..n) into A (N x N column-major,
// N = n + q). Parallel by columns when `par` (each entry is its own kernel, so any split gives the same bits).
template <int K>
void fill_system(
    const double* y,
    const std::int64_t* rows,
    std::int64_t n,
    std::int64_t m,
    const double* smooth,
    double eps,
    const std::int64_t* pw,
    std::int64_t q,
    const double* shift,
    const double* scale,
    double* A,
    bool par
) {
    const std::int64_t N = n + q;
    auto row = [&](std::int64_t i) { return y + (rows ? rows[i] : i) * m; };
    // The points coordinate-major (ys[l n + i]) for the lanes: two entries of a column at a time, the operations of
    // the scalar entry (distance, eps r, phi; spatial's careful path in its lane), so the same bits.
    const bool lanes = m <= spatial::kBlock;
    std::vector<double> ys(lanes ? static_cast<std::size_t>(n * m) : 0);
    if (lanes)
        for (std::int64_t i = 0; i < n; ++i)
            for (std::int64_t l = 0; l < m; ++l) ys[l * n + i] = row(i)[l];
    const float64x2_t veps = vdupq_n_f64(eps), tiny = vdupq_n_f64(spatial::kTiny), huge = vdupq_n_f64(spatial::kHuge);
    auto column = [&](std::int64_t j, double* xhat, double* mono) {
        double* col = A + j * N;
        if (j < n) {
            const double* yj = row(j);
            std::int64_t i = j;
            if (lanes) {
                for (; i + 2 <= n; i += 2) {
                    float64x2_t sq = vdupq_n_f64(0.0);
                    for (std::int64_t l = 0; l < m; ++l) {
                        const float64x2_t z = vsubq_f64(vld1q_f64(ys.data() + l * n + i), vdupq_n_f64(yj[l]));
                        sq = vfmaq_f64(sq, z, z);
                    }
                    float64x2_t d = vsqrtq_f64(sq);
                    const uint64x2_t fine = sum_fine(sq, tiny, huge);
                    if ((vgetq_lane_u64(fine, 0) & vgetq_lane_u64(fine, 1)) == 0) [[unlikely]] {
                        const double dv[2] = {distance(row(i), yj, m), distance(row(i + 1), yj, m)};
                        d = vld1q_f64(dv);
                    }
                    vst1q_f64(col + i, phi_lanes<K>(vmulq_f64(veps, d)));
                }
            }
            for (; i < n; ++i) col[i] = phi<K>(eps * distance(row(i), yj, m));
            col[j] = col[j] + smooth[rows ? rows[j] : j];
            monomials(yj, m, pw, q, shift, scale, xhat, mono);
            for (std::int64_t k = 0; k < q; ++k) col[n + k] = mono[k];
        } else {
            for (std::int64_t i = j; i < N; ++i) col[i] = 0.0;
        }
    };
    const std::int64_t work = n * n / 2 * (2 * m + 24) + N * q * (m + 4);
    if (par && work >= kParallelThreshold && parallel::available() && N > 64) {
        const std::int64_t per = 32, tasks = (N + per - 1) / per;
        parallel::parallel_for(
            tasks,
            [&](std::int64_t t) {
                std::vector<double> xhat(m), mono(q);
                for (std::int64_t j = t * per; j < std::min(N, (t + 1) * per); ++j) column(j, xhat.data(), mono.data());
            },
            work_ns(work / tasks)
        );
        return;
    }
    const parallel::Unlocked unlocked(work >= kParallelThreshold);
    std::vector<double> xhat(m), mono(q);
    for (std::int64_t j = 0; j < N; ++j) column(j, xhat.data(), mono.data());
}

// numpy's matrix_rank of P (the system's rows n.. of columns 0..n): singular values by gesdd, tolerance
// s_max max(n, q) eps
std::int64_t tail_rank(const double* A, std::int64_t n, std::int64_t q, linalg::Scratch& sc) {
    if (q == 0) return 0;
    const std::int64_t N = n + q;
    std::vector<double> P(static_cast<std::size_t>(n * q)), s(static_cast<std::size_t>(std::min(n, q)));
    for (std::int64_t k = 0; k < q; ++k)
        for (std::int64_t i = 0; i < n; ++i) P[i + k * n] = A[(n + k) + i * N];
    double du = 0.0, dvt = 0.0;
    if (lapack::gesdd('N', n, q, P.data(), std::max<lint>(1, n), s.data(), &du, 1, &dvt, 1, sc) != 0)
        return -1;  // no convergence: reported as a rank failure
    double smax = 0.0;
    for (const double v : s) smax = std::max(smax, v);
    const double tol = smax * static_cast<double>(std::max(n, q)) * kEps;
    std::int64_t rank = 0;
    for (const double v : s) rank += v > tol ? 1 : 0;
    return rank;
}

// sum_{i in [lo, N)} A[i] b[i] (four partial sums by i mod 4, combined (s0 + s1) + (s2 + s3): fixed by the size)
inline double dot_tail(const double* a, const double* b, std::int64_t lo, std::int64_t N) {
    double s[4] = {0.0, 0.0, 0.0, 0.0};
    std::int64_t i = lo;
    for (; i + 4 <= N; i += 4)
        for (int r = 0; r < 4; ++r) s[r] = std::fma(a[i + r], b[i + r], s[r]);
    for (int r = 0; i < N; ++i, ++r) s[r] = std::fma(a[i], b[i], s[r]);
    return (s[0] + s[1]) + (s[2] + s[3]);
}

// Solves A x = b in place with sytrf's lower Bunch-Kaufman factor of A (N x N column-major, LAPACK's 1-based ipiv:
// positive for a 1 x 1 pivot, both rows of a 2 x 2 one negative): LAPACK dsytrs's steps (L D y = b with the
// interchanges, then L^T x = y), written out because the library call cost more than the solve at the small sizes
// of neighbors= (3.5 us for N = 53, decision IP101). Higham's theorem 11.3 bounds it as dsytrs.
void ldlt_solve(const double* A, std::int64_t N, const lint* ipiv, double* b) {
    for (std::int64_t k = 0; k < N;) {
        const double* ak = A + k * N;
        if (ipiv[k] > 0) {
            const std::int64_t kp = ipiv[k] - 1;
            if (kp != k) std::swap(b[k], b[kp]);
            const double bk = b[k];
            for (std::int64_t i = k + 1; i < N; ++i) b[i] = b[i] - ak[i] * bk;
            b[k] = b[k] * (1.0 / ak[k]);
            k += 1;
        } else {
            const std::int64_t kp = -ipiv[k] - 1;
            if (kp != k + 1) std::swap(b[k + 1], b[kp]);
            const double* ak1 = ak + N;
            const double b0 = b[k], b1 = b[k + 1];
            for (std::int64_t i = k + 2; i < N; ++i) b[i] = b[i] - ak[i] * b0;
            for (std::int64_t i = k + 2; i < N; ++i) b[i] = b[i] - ak1[i] * b1;
            const double akm1k = ak[k + 1], akm1 = ak[k] / akm1k, akk = ak1[k + 1] / akm1k;
            const double denom = akm1 * akk - 1.0;
            const double bkm1 = b0 / akm1k, bkk = b1 / akm1k;
            b[k] = (akk * bkm1 - bkk) / denom;
            b[k + 1] = (akm1 * bkk - bkm1) / denom;
            k += 2;
        }
    }
    for (std::int64_t k = N - 1; k >= 0;) {
        if (ipiv[k] > 0) {
            b[k] = b[k] - dot_tail(A + k * N, b, k + 1, N);
            const std::int64_t kp = ipiv[k] - 1;
            if (kp != k) std::swap(b[k], b[kp]);
            k -= 1;
        } else {
            b[k] = b[k] - dot_tail(A + k * N, b, k + 1, N);
            b[k - 1] = b[k - 1] - dot_tail(A + (k - 1) * N, b, k + 1, N);
            const std::int64_t kp = -ipiv[k] - 1;
            if (kp != k) std::swap(b[k], b[kp]);
            k -= 2;
        }
    }
}

// Factors A (N x N, lower) and solves for the values (n rows of D doubles, row stride ld) into coef (N x D
// row-major). Returns the status (0 solved, 2 an exact zero pivot, 3 not finite).
int solve_system(
    double* A,
    std::int64_t n,
    std::int64_t q,
    const double* vals,
    const std::int64_t* rows,
    std::int64_t D,
    double* coef,
    linalg::Scratch& sc
) {
    const std::int64_t N = n + q;
    std::vector<lint> ipiv(static_cast<std::size_t>(std::max<std::int64_t>(N, 1)));
    if (N == 0) return 0;
    if (lapack::sytrf(N, A, N, ipiv.data(), sc) > 0) return 2;
    std::vector<double> B(static_cast<std::size_t>(N * D));
    for (std::int64_t c = 0; c < D; ++c) {
        for (std::int64_t i = 0; i < n; ++i) B[i + c * N] = vals[(rows ? rows[i] : i) * D + c];
        for (std::int64_t i = n; i < N; ++i) B[i + c * N] = 0.0;
    }
    for (std::int64_t c = 0; c < D; ++c) ldlt_solve(A, N, ipiv.data(), B.data() + c * N);
    bool finite = true;
    for (std::int64_t i = 0; i < N; ++i)
        for (std::int64_t c = 0; c < D; ++c) {
            const double v = B[i + c * N];
            finite = finite && std::isfinite(v);
            coef[i * D + c] = v;
        }
    return finite ? 0 : 3;
}

// ---------------------------------------------------------------- evaluation

constexpr std::int64_t kLaneCols = 8;  // eval_lanes' accumulators: at most 8 value columns (doubles) per point

// eval_points for four points at a time (two NEON pairs): per data point j the four distances, phi and the fma into
// each point's sums run side by side, so four chains of fma are in flight instead of one. Every point gets the
// operations of eval_points' scalar loop in the same order (the distance's fma chain, sqrt, eps r, phi; the sums in
// the order of the data points, then the monomials), so the same bits; a distance that needs spatial's careful path
// takes it in its lane. Returns the first point left for the scalar loop (fewer than four remain).
template <int K>
std::int64_t eval_lanes(
    const double* X,
    std::int64_t i0,
    std::int64_t i1,
    const std::int64_t* which,
    std::int64_t m,
    const double* y,
    const std::int64_t* rows,
    std::int64_t n,
    const double* coef,
    std::int64_t D,
    double eps,
    const std::int64_t* pw,
    std::int64_t q,
    const double* shift,
    const double* scale,
    double* out,
    double* xhat,
    double* mono
) {
    constexpr int G = 4;
    double xs[spatial::kBlock * G];  // the four points' coordinates, coordinate-major
    const float64x2_t veps = vdupq_n_f64(eps), tiny = vdupq_n_f64(spatial::kTiny), huge = vdupq_n_f64(spatial::kHuge);
    std::int64_t t = i0;
    for (; t + G <= i1; t += G) {
        std::int64_t idx[G];
        for (int g = 0; g < G; ++g) idx[g] = which ? which[t + g] : t + g;
        for (std::int64_t l = 0; l < m; ++l)
            for (int g = 0; g < G; ++g) xs[l * G + g] = X[idx[g] * m + l];
        float64x2_t acc[2 * kLaneCols];
        for (std::int64_t c = 0; c < 2 * D; ++c) acc[c] = vdupq_n_f64(0.0);
        for (std::int64_t j = 0; j < n; ++j) {
            const double* yj = y + (rows ? rows[j] : j) * m;
            float64x2_t s0 = vdupq_n_f64(0.0), s1 = s0;
            for (std::int64_t l = 0; l < m; ++l) {
                const float64x2_t yl = vdupq_n_f64(yj[l]);
                const float64x2_t z0 = vsubq_f64(vld1q_f64(xs + l * G), yl),
                                  z1 = vsubq_f64(vld1q_f64(xs + l * G + 2), yl);
                s0 = vfmaq_f64(s0, z0, z0);
                s1 = vfmaq_f64(s1, z1, z1);
            }
            float64x2_t d0 = vsqrtq_f64(s0), d1 = vsqrtq_f64(s1);
            const uint64x2_t fine0 = sum_fine(s0, tiny, huge), fine1 = sum_fine(s1, tiny, huge);
            if ((vgetq_lane_u64(fine0, 0) & vgetq_lane_u64(fine0, 1) & vgetq_lane_u64(fine1, 0) &
                 vgetq_lane_u64(fine1, 1)) == 0) [[unlikely]] {
                double dv[G];
                for (int g = 0; g < G; ++g) dv[g] = distance(X + idx[g] * m, yj, m);
                d0 = vld1q_f64(dv);
                d1 = vld1q_f64(dv + 2);
            }
            const float64x2_t f0 = phi_lanes<K>(vmulq_f64(veps, d0)), f1 = phi_lanes<K>(vmulq_f64(veps, d1));
            const double* cj = coef + j * D;
            for (std::int64_t c = 0; c < D; ++c) {
                const float64x2_t cc = vdupq_n_f64(cj[c]);
                acc[2 * c] = vfmaq_f64(acc[2 * c], f0, cc);
                acc[2 * c + 1] = vfmaq_f64(acc[2 * c + 1], f1, cc);
            }
        }
        for (int g = 0; g < G; ++g) {
            double a[kLaneCols];
            for (std::int64_t c = 0; c < D; ++c) a[c] = acc[2 * c + g / 2][g % 2];
            monomials(X + idx[g] * m, m, pw, q, shift, scale, xhat, mono);
            for (std::int64_t k = 0; k < q; ++k) {
                const double* ck = coef + (n + k) * D;
                for (std::int64_t c = 0; c < D; ++c) a[c] = std::fma(mono[k], ck[c], a[c]);
            }
            std::memcpy(out + idx[g] * D, a, static_cast<std::size_t>(D) * sizeof(double));
        }
    }
    return t;
}

// The values at the points X[i0..i1) (rows of m) into out (rows of D): the data points y[rows[0..n)] (rows null:
// 0..n), coefficients coef (N x D row-major). acc, xhat, mono: room for D, m and q doubles.
template <int K>
void eval_points(
    const double* X,
    std::int64_t i0,
    std::int64_t i1,
    const std::int64_t* which,
    std::int64_t m,
    const double* y,
    const std::int64_t* rows,
    std::int64_t n,
    const double* coef,
    std::int64_t D,
    double eps,
    const std::int64_t* pw,
    std::int64_t q,
    const double* shift,
    const double* scale,
    double* out,
    double* acc,
    double* xhat,
    double* mono
) {
    std::int64_t t0 = i0;
    if (m <= spatial::kBlock && D <= kLaneCols)
        t0 = eval_lanes<K>(X, i0, i1, which, m, y, rows, n, coef, D, eps, pw, q, shift, scale, out, xhat, mono);
    for (std::int64_t t = t0; t < i1; ++t) {
        const std::int64_t i = which ? which[t] : t;
        const double* x = X + i * m;
        for (std::int64_t c = 0; c < D; ++c) acc[c] = 0.0;
        for (std::int64_t j = 0; j < n; ++j) {
            const double f = phi<K>(eps * distance(x, y + (rows ? rows[j] : j) * m, m));
            const double* cj = coef + j * D;
            for (std::int64_t c = 0; c < D; ++c) acc[c] = std::fma(f, cj[c], acc[c]);
        }
        monomials(x, m, pw, q, shift, scale, xhat, mono);
        for (std::int64_t k = 0; k < q; ++k) {
            const double* ck = coef + (n + k) * D;
            for (std::int64_t c = 0; c < D; ++c) acc[c] = std::fma(mono[k], ck[c], acc[c]);
        }
        std::memcpy(out + i * D, acc, static_cast<std::size_t>(D) * sizeof(double));
    }
}

// ---------------------------------------------------------------- bindings

nb::object rbf_duplicate(nb::handle yo, nb::handle so) {
    const Mat Y = mat_arg(yo, "y");
    const std::int64_t n = Y.rows, m = Y.cols;
    const double* sm = vec_arg(so, "smoothing", n);
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), std::int64_t{0});
    auto less = [&](std::int64_t a, std::int64_t b) {
        for (std::int64_t l = 0; l < m; ++l) {
            const double u = Y.p[a * m + l], v = Y.p[b * m + l];
            if (u < v) return true;
            if (v < u) return false;
        }
        return a < b;
    };
    auto equal = [&](std::int64_t a, std::int64_t b) {
        for (std::int64_t l = 0; l < m; ++l)
            if (!(Y.p[a * m + l] == Y.p[b * m + l])) return false;
        return true;
    };
    {
        const parallel::Unlocked unlocked(n * m >= kParallelThreshold);
        std::sort(order.begin(), order.end(), less);
    }
    std::int64_t bi = -1, bj = -1;
    for (std::int64_t s = 0; s < n;) {
        std::int64_t e = s + 1;
        while (e < n && equal(order[s], order[e])) ++e;
        std::int64_t first = -1, second = -1;  // the run is in index order: its first two zero-smoothing points
        for (std::int64_t t = s; t < e && second < 0; ++t)
            if (sm[order[t]] == 0.0) (first < 0 ? first : second) = order[t];
        if (second >= 0 && (bj < 0 || second < bj)) bi = first, bj = second;
        s = e;
    }
    return nb::make_tuple(bi, bj);
}

nb::object rbf_fit(nb::handle yo, nb::handle dol, nb::handle so, int kernel, double eps, nb::handle po) {
    const Mat Y = mat_arg(yo, "y");
    const std::int64_t n = Y.rows, m = Y.cols;
    const Cols V = cols_arg(dol, "d", n);
    const double* sm = vec_arg(so, "smoothing", n);
    const Ints Pw = ints_arg(po, "powers");
    if (Pw.cols != m) throw nb::value_error("'powers' must have one column per coordinate");
    kernel_arg(kernel);
    const std::int64_t q = Pw.rows, N = n + q, D = V.D;
    const std::int64_t cshape[2] = {N, V.ncol}, mshape[1] = {m};
    NDArray coef = new_array(V.dtype, cshape, true), sh = new_array(DType::Float64, mshape),
            scl = new_array(DType::Float64, mshape);
    auto* shift = reinterpret_cast<double*>(sh.data);
    auto* scale = reinterpret_cast<double*>(scl.data);
    shift_scale(Y.p, nullptr, n, m, shift, scale);
    int status = 0;
    std::int64_t rank = q;
    {
        std::vector<double> A(static_cast<std::size_t>(N * N));
        with_kernel(kernel, [&](auto kc) {
            fill_system<decltype(kc)::value>(Y.p, nullptr, n, m, sm, eps, Pw.p, q, shift, scale, A.data(), true);
        });
        const parallel::Unlocked unlocked(N * N * (N + D) >= kParallelThreshold);
        linalg::Scratch sc;
        rank = tail_rank(A.data(), n, q, sc);
        if (rank < q) status = 1;
        else status = solve_system(A.data(), n, q, V.p, nullptr, D, reinterpret_cast<double*>(coef.data), sc);
    }
    return nb::make_tuple(wrap(std::move(coef)), wrap(std::move(sh)), wrap(std::move(scl)), status, rank);
}

nb::object rbf_matrix(nb::handle yo, nb::handle so, int kernel, double eps, nb::handle po) {
    const Mat Y = mat_arg(yo, "y");
    const std::int64_t n = Y.rows, m = Y.cols;
    const double* sm = vec_arg(so, "smoothing", n);
    const Ints Pw = ints_arg(po, "powers");
    if (Pw.cols != m) throw nb::value_error("'powers' must have one column per coordinate");
    kernel_arg(kernel);
    const std::int64_t q = Pw.rows, N = n + q;
    std::vector<double> shift(m), scale(m);
    shift_scale(Y.p, nullptr, n, m, shift.data(), scale.data());
    const std::int64_t shape[2] = {N, N};
    NDArray out = new_array(DType::Float64, shape);
    auto* A = reinterpret_cast<double*>(out.data);
    with_kernel(kernel, [&](auto kc) {
        fill_system<decltype(kc)::value>(Y.p, nullptr, n, m, sm, eps, Pw.p, q, shift.data(), scale.data(), A, true);
    });
    for (std::int64_t j = 0; j < N; ++j)  // the upper triangle from the lower (symmetric: A[i + j N] = A[j + i N])
        for (std::int64_t i = 0; i < j; ++i) A[i + j * N] = A[j + i * N];
    return wrap(std::move(out));
}

nb::object rbf_eval(
    nb::handle xo, nb::handle yo, nb::handle co, int kernel, double eps, nb::handle po, nb::handle sho, nb::handle sco
) {
    const Mat X = mat_arg(xo, "x");
    const Mat Y = mat_arg(yo, "y");
    const std::int64_t n = Y.rows, m = Y.cols, Q = X.rows;
    if (X.cols != m) throw nb::value_error("'x' and 'y' must have one number of coordinates");
    const Ints Pw = ints_arg(po, "powers");
    if (Pw.cols != m) throw nb::value_error("'powers' must have one column per coordinate");
    kernel_arg(kernel);
    const std::int64_t q = Pw.rows;
    const Cols C = cols_arg(co, "coeffs", n + q);
    const double* shift = vec_arg(sho, "shift", m);
    const double* scale = vec_arg(sco, "scale", m);
    const std::int64_t D = C.D;
    const std::int64_t shape[2] = {Q, C.ncol};
    NDArray out = new_array(C.dtype, shape);
    auto* o = reinterpret_cast<double*>(out.data);
    if (Q == 0 || D == 0) return wrap(std::move(out));
    with_kernel(kernel, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        split_range(Q, Q * (n + q) * (2 * m + 24 + 2 * D), 8, [&](std::int64_t b, std::int64_t e) {
            std::vector<double> acc(D), xhat(m), mono(q);
            eval_points<K>(
                X.p, b, e, nullptr, m, Y.p, nullptr, n, C.p, D, eps, Pw.p, q, shift, scale, o, acc.data(), xhat.data(),
                mono.data()
            );
        });
    });
    return wrap(std::move(out));
}

nb::object rbf_local(
    nb::handle xo, nb::handle yo, nb::handle dol, nb::handle so, int kernel, double eps, nb::handle po, nb::handle io
) {
    const Mat X = mat_arg(xo, "x");
    const Mat Y = mat_arg(yo, "y");
    const std::int64_t n = Y.rows, m = Y.cols, Q = X.rows;
    if (X.cols != m) throw nb::value_error("'x' and 'y' must have one number of coordinates");
    const Cols V = cols_arg(dol, "d", n);
    const double* sm = vec_arg(so, "smoothing", n);
    const Ints Pw = ints_arg(po, "powers");
    if (Pw.cols != m) throw nb::value_error("'powers' must have one column per coordinate");
    const Ints I = ints_arg(io, "idx");
    if (I.rows != Q || I.cols < 1) throw nb::value_error("'idx' must be (Q, k) with k >= 1");
    kernel_arg(kernel);
    const std::int64_t k = I.cols, q = Pw.rows, N = k + q, D = V.D;
    for (std::int64_t t = 0; t < Q * k; ++t)
        if (I.p[t] < 0 || I.p[t] >= n) throw nb::value_error("'idx' must hold data point numbers");
    const std::int64_t shape[2] = {Q, V.ncol};
    NDArray out = new_array(V.dtype, shape);
    auto* o = reinterpret_cast<double*>(out.data);
    int status = 0;
    std::int64_t bad_point = -1, bad_rank = q;
    if (Q > 0) {
        // each point's set, sorted; the points ordered by (set, number); the runs of one set
        std::vector<std::int64_t> sets(I.p, I.p + Q * k), order(static_cast<std::size_t>(Q)), starts;
        split_range(Q, Q * k * 8, 256, [&](std::int64_t b, std::int64_t e) {
            for (std::int64_t i = b; i < e; ++i) std::sort(sets.begin() + i * k, sets.begin() + (i + 1) * k);
        });
        std::iota(order.begin(), order.end(), std::int64_t{0});
        {
            const parallel::Unlocked unlocked(Q * k >= kParallelThreshold);
            const std::int64_t* S = sets.data();
            std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
                for (std::int64_t t = 0; t < k; ++t)
                    if (S[a * k + t] != S[b * k + t]) return S[a * k + t] < S[b * k + t];
                return a < b;
            });
            for (std::int64_t t = 0; t < Q; ++t)
                if (t == 0 || !std::equal(S + order[t] * k, S + order[t] * k + k, S + order[t - 1] * k))
                    starts.push_back(t);
            starts.push_back(Q);
        }
        const std::int64_t G = static_cast<std::int64_t>(starts.size()) - 1;
        parallel::FirstRow first;  // the smallest point number of a failing set
        std::vector<int> gstatus(static_cast<std::size_t>(G), 0);
        std::vector<std::int64_t> grank(static_cast<std::size_t>(G), q);
        with_kernel(kernel, [&](auto kc) {
            constexpr int K = decltype(kc)::value;
            const std::int64_t per_set = N * N * N / 3 + k * k * (m + 12) + N * D * 4;
            const std::int64_t per_point = N * (2 * m + 24 + 2 * D);
            split_range(G, G * per_set + Q * per_point, 1, [&](std::int64_t gb, std::int64_t ge) {
                linalg::Scratch sc;
                std::vector<double> A(static_cast<std::size_t>(N * N)), coef(static_cast<std::size_t>(N * D));
                std::vector<double> ys(static_cast<std::size_t>(k * m)), shift(m), scale(m), acc(D), xhat(m), mono(q);
                for (std::int64_t g = gb; g < ge; ++g) {
                    const std::int64_t* rows = sets.data() + order[starts[g]] * k;
                    for (std::int64_t i = 0; i < k; ++i)
                        std::memcpy(ys.data() + i * m, Y.p + rows[i] * m, static_cast<std::size_t>(m) * sizeof(double));
                    shift_scale(ys.data(), nullptr, k, m, shift.data(), scale.data());
                    std::vector<double> sl(static_cast<std::size_t>(k));
                    for (std::int64_t i = 0; i < k; ++i) sl[i] = sm[rows[i]];
                    fill_system<K>(
                        ys.data(), nullptr, k, m, sl.data(), eps, Pw.p, q, shift.data(), scale.data(), A.data(), false
                    );
                    const std::int64_t r = tail_rank(A.data(), k, q, sc);
                    int st = r < q ? 1 : solve_system(A.data(), k, q, V.p, rows, D, coef.data(), sc);
                    if (st != 0) {
                        gstatus[g] = st;
                        grank[g] = r;
                        first.offer(order[starts[g]]);
                        for (std::int64_t t = starts[g]; t < starts[g + 1]; ++t)
                            std::fill(o + order[t] * D, o + (order[t] + 1) * D, kNaN);
                        continue;
                    }
                    eval_points<K>(
                        X.p, starts[g], starts[g + 1], order.data(), m, ys.data(), nullptr, k, coef.data(), D, eps,
                        Pw.p, q, shift.data(), scale.data(), o, acc.data(), xhat.data(), mono.data()
                    );
                }
            });
        });
        if (first.any()) {
            bad_point = first.row();
            for (std::int64_t g = 0; g < G; ++g)
                if (order[starts[g]] == bad_point) status = gstatus[g], bad_rank = grank[g];
        }
    }
    return nb::make_tuple(wrap(std::move(out)), status, bad_point, bad_rank);
}

}  // namespace

void bind_rbf(nb::module_& m) {
    m.def("rbf_duplicate", &rbf_duplicate, nb::arg("y").none(), nb::arg("smoothing").none());
    m.def(
        "rbf_fit", &rbf_fit, nb::arg("y").none(), nb::arg("d").none(), nb::arg("smoothing").none(), nb::arg("kernel"),
        nb::arg("epsilon"), nb::arg("powers").none()
    );
    m.def(
        "rbf_matrix", &rbf_matrix, nb::arg("y").none(), nb::arg("smoothing").none(), nb::arg("kernel"),
        nb::arg("epsilon"), nb::arg("powers").none()
    );
    m.def(
        "rbf_eval", &rbf_eval, nb::arg("x").none(), nb::arg("y").none(), nb::arg("coeffs").none(), nb::arg("kernel"),
        nb::arg("epsilon"), nb::arg("powers").none(), nb::arg("shift").none(), nb::arg("scale").none()
    );
    m.def(
        "rbf_local", &rbf_local, nb::arg("x").none(), nb::arg("y").none(), nb::arg("d").none(),
        nb::arg("smoothing").none(), nb::arg("kernel"), nb::arg("epsilon"), nb::arg("powers").none(),
        nb::arg("idx").none()
    );
}

}  // namespace sd::interpolate
