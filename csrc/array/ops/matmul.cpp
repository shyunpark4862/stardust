// matmul (docs/design/array.md §9).
//
// Shapes follow the Array API: (..., m, k) @ (..., k, n) -> (..., m, n), the
// leading "batch" axes broadcast, and a 1-D operand is a row (left) or a
// column (right) vector whose axis is dropped from the result. vector @
// vector is a Python scalar (docs/design/array.md §2.2).
//
// float64/complex128 call Accelerate's BLAS (gemm, gemv for a single row or
// column, dot for a single element). BLAS reads a matrix given a pointer, a
// "leading dimension" and a transpose flag, so any matrix whose rows or whose
// columns are contiguous is passed as is; only other layouts are copied
// first. int64 (and bool, computed in int64: docs/design/array.md §9.1) use
// our own i-k-j loop.

#include "array/ops/matmul.hpp"

#include "accelerate/blas.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/elementwise.hpp"
#include "array/ops/copy.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"


#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sd {

namespace {

using blas_int = blas::Int;

// A rows x cols matrix in memory: element (i, j) at p + i*rs + j*cs bytes.
struct Mat {
    const std::byte* p;
    std::int64_t rows, cols, rs, cs;
};

// How BLAS should read `a`, if it can without a copy: row-major with leading
// dimension ld, or -- when the columns are the contiguous direction -- as the
// transpose of a row-major cols x rows matrix.
struct BlasView {
    blas::Trans trans;
    blas_int ld;
};

std::optional<BlasView> blas_view(const Mat& a, std::int64_t item) {
    auto unit = [&](std::int64_t s, std::int64_t extent) { return extent == 1 || s == item; };
    auto lead = [&](std::int64_t s, std::int64_t extent, std::int64_t inner) -> std::optional<blas_int> {
        if (extent == 1) return std::max<std::int64_t>(inner, 1);
        if (s <= 0 || s % item != 0 || s / item < std::max<std::int64_t>(inner, 1)) return std::nullopt;
        return s / item;
    };
    if (unit(a.cs, a.cols))
        if (auto ld = lead(a.rs, a.rows, a.cols)) return BlasView{CblasNoTrans, *ld};
    if (unit(a.rs, a.rows))
        if (auto ld = lead(a.cs, a.cols, a.rows)) return BlasView{CblasTrans, *ld};
    return std::nullopt;
}

// A contiguous row-major copy of `a`, for layouts BLAS cannot read.
template <typename T>
std::unique_ptr<T[]> packed(const Mat& a) {
    auto buf = std::make_unique_for_overwrite<T[]>(static_cast<size_t>(std::max<std::int64_t>(a.rows * a.cols, 1)));
    for (std::int64_t i = 0; i < a.rows; ++i)
        for (std::int64_t j = 0; j < a.cols; ++j) buf[i * a.cols + j] = load_as<T>(a.p + i * a.rs + j * a.cs);
    return buf;
}

// A vector (n elements, stride s bytes) as a BLAS pointer + increment.
template <typename T>
const T* blas_vector(const std::byte* p, std::int64_t n, std::int64_t s, blas_int& inc, std::unique_ptr<T[]>& keep) {
    const std::int64_t item = sizeof(T);
    if (n == 1 || (s > 0 && s % item == 0)) {
        inc = n == 1 ? 1 : static_cast<blas_int>(s / item);
        return reinterpret_cast<const T*>(p);
    }
    keep = packed<T>({p, 1, n, 0, s});  // zero or negative stride: copy
    inc = 1;
    return keep.get();
}

// ------------------------------------------------------------ BLAS kernels

template <typename T>
void blas_matmul(const Mat& a, const Mat& b, T* c) {
    const std::int64_t m = a.rows, k = a.cols, n = b.cols;
    if (m == 0 || n == 0) return;
    if (k == 0) {
        std::fill(c, c + m * n, T(0));
        return;
    }
    if (m == 1 && n == 1) {  // dot
        blas_int ia, ib;
        std::unique_ptr<T[]> ka, kb;
        const T* x = blas_vector<T>(a.p, k, a.cs, ia, ka);
        const T* y = blas_vector<T>(b.p, k, b.rs, ib, kb);
        blas::dot<T>(k, x, ia, y, ib, c);
        return;
    }
    if (n == 1 || m == 1) {  // gemv: one row times a matrix, or a matrix times one column
        const bool col = n == 1;
        const Mat& mat = col ? a : b;                       // the matrix operand
        const std::byte* vp = col ? b.p : a.p;              // the vector operand
        const std::int64_t vs = col ? b.rs : a.cs;
        Mat mt = mat;
        if (!col) mt = {b.p, b.cols, b.rows, b.cs, b.rs};  // y = B^T a: read B as its transpose
        blas_int inc;
        std::unique_ptr<T[]> kv, km;
        const T* x = blas_vector<T>(vp, k, vs, inc, kv);
        std::optional<BlasView> v = blas_view(mt, sizeof(T));
        const T* ap = reinterpret_cast<const T*>(mt.p);
        if (!v) {
            km = packed<T>(mt);
            ap = km.get();
            v = BlasView{CblasNoTrans, static_cast<blas_int>(std::max<std::int64_t>(mt.cols, 1))};
        }
        // gemv takes the *stored* matrix: rows x cols as given, or cols x rows when transposed.
        const blas_int sr = v->trans == CblasNoTrans ? mt.rows : mt.cols;
        const blas_int sc = v->trans == CblasNoTrans ? mt.cols : mt.rows;
        blas::gemv<T>(v->trans, sr, sc, ap, v->ld, x, inc, c);
        return;
    }
    std::unique_ptr<T[]> ka, kb;
    std::optional<BlasView> va = blas_view(a, sizeof(T)), vb = blas_view(b, sizeof(T));
    const T* ap = reinterpret_cast<const T*>(a.p);
    const T* bp = reinterpret_cast<const T*>(b.p);
    if (!va) {
        ka = packed<T>(a);
        ap = ka.get();
        va = BlasView{CblasNoTrans, static_cast<blas_int>(k)};
    }
    if (!vb) {
        kb = packed<T>(b);
        bp = kb.get();
        vb = BlasView{CblasNoTrans, static_cast<blas_int>(n)};
    }
    blas::gemm<T>(va->trans, vb->trans, m, n, k, ap, va->ld, bp, vb->ld, c, n);
}

// int64: i-k-j order, so the innermost loop walks a row of b and a row of c
// (docs/design/array.md §9.3). Products and sums wrap. Scalar multiplies: NEON has no
// 64-bit integer vector multiply.
void int_matmul(const Mat& a, const Mat& b, std::int64_t* c) {
    const std::int64_t m = a.rows, k = a.cols, n = b.cols;
    for (std::int64_t i = 0; i < m; ++i) {
        std::int64_t* crow = c + i * n;
        std::fill(crow, crow + n, 0);
        for (std::int64_t p = 0; p < k; ++p) {
            const std::int64_t aip = load_as<std::int64_t>(a.p + i * a.rs + p * a.cs);
            const std::byte* brow = b.p + p * b.rs;
            std::int64_t j = 0;
            if (b.cs == sizeof(std::int64_t)) {
                // Unrolled by 4 (docs/design/array.md §9.3):
                // four independent multiply-adds per iteration keep the
                // multiplier busy, and a longer loop body no longer depends on
                // where the linker happens to place it (the 7-instruction
                // version ran 1.7x slower when it straddled a 64-byte fetch block).
                const std::int64_t* bp = reinterpret_cast<const std::int64_t*>(brow);
                for (; j + 4 <= n; j += 4) {
                    crow[j] = ops::wrap_add(crow[j], ops::wrap_mul(aip, bp[j]));
                    crow[j + 1] = ops::wrap_add(crow[j + 1], ops::wrap_mul(aip, bp[j + 1]));
                    crow[j + 2] = ops::wrap_add(crow[j + 2], ops::wrap_mul(aip, bp[j + 2]));
                    crow[j + 3] = ops::wrap_add(crow[j + 3], ops::wrap_mul(aip, bp[j + 3]));
                }
            }
            for (; j < n; ++j)
                crow[j] = ops::wrap_add(crow[j], ops::wrap_mul(aip, load_as<std::int64_t>(brow + j * b.cs)));
        }
    }
}

}  // namespace

// Batched BLAS on the pool (docs/design/array.md §9.4):
// matrices of at most kSmallBlas multiply-adds, grouped into tasks of about
// kBlasTaskWork. Accelerate ran a square dgemm on one thread up to 256
// (CPU time / wall time 1.00) and on several from 288 (1.62), measured.
inline constexpr std::int64_t kSmallBlas = 256 * 256 * 256;
inline constexpr std::int64_t kBlasTaskWork = 1 << 15;

nb::object matmul(nb::handle x1, nb::handle x2) {
    const NDArray& a0 = require_ndarray(x1, "x1");
    const NDArray& b0 = require_ndarray(x2, "x2");
    const DType d = promote(promote(a0.dtype, b0.dtype), DType::Int64);  // bool @ bool is int64: docs/design/array.md §9.1
    const NDArray a = a0.dtype == d ? a0 : cast_copy(a0, d);
    const NDArray b = b0.dtype == d ? b0 : cast_copy(b0, d);
    const bool a_vec = a.ndim == 1, b_vec = b.ndim == 1;
    const std::int64_t m = a_vec ? 1 : a.shape[a.ndim - 2], k = a.shape[a.ndim - 1];
    const std::int64_t kb = b_vec ? b.shape[0] : b.shape[b.ndim - 2], n = b_vec ? 1 : b.shape[b.ndim - 1];
    if (k != kb)
        throw nb::value_error(("the last axis of 'x1' (" + std::to_string(k) +
                               ") must match the second-to-last axis of 'x2' (" + std::to_string(kb) + ")")
                                  .c_str());

    // Batch axes: everything before the matrix axes, broadcast together.
    const int a_batch = a_vec ? 0 : a.ndim - 2, b_batch = b_vec ? 0 : b.ndim - 2;
    const std::span<const std::int64_t> spans[] = {{a.shape.data(), static_cast<size_t>(a_batch)},
                                                   {b.shape.data(), static_cast<size_t>(b_batch)}};
    const std::vector<std::int64_t> batch = broadcast_shapes(spans);
    const int bd = static_cast<int>(batch.size());
    std::vector<std::int64_t> full = batch;  // batch + (m, n), what we compute into
    full.push_back(m);
    full.push_back(n);
    if (full.size() > static_cast<size_t>(kMaxNdim))
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    NDArray out = new_array(d, full);

    auto batch_stride = [&](const NDArray& x, int nbatch, int dd) {  // stride of broadcast batch axis dd
        const int xd = dd - (bd - nbatch);
        return xd < 0 || x.shape[xd] == 1 ? 0 : x.strides[xd];
    };
    std::int64_t count = 1;
    for (std::int64_t e : batch) count *= e;
    std::array<std::int64_t, kMaxNdim> idx{};
    const std::byte* pa = a.data;
    const std::byte* pb = b.data;
    const std::int64_t a_rs = a_vec ? 0 : a.strides[a.ndim - 2], a_cs = a.strides[a.ndim - 1];
    const std::int64_t b_rs = b_vec ? b.strides[0] : b.strides[b.ndim - 2], b_cs = b_vec ? 0 : b.strides[b.ndim - 1];
    std::vector<std::pair<const std::byte*, const std::byte*>> starts(count);  // (a, b) of each batch
    for (std::int64_t t = 0; t < count; ++t) {
        starts[t] = {pa, pb};
        for (int dd = bd - 1; dd >= 0; --dd) {  // C-order odometer over the batch axes
            pa += batch_stride(a, a_batch, dd);
            pb += batch_stride(b, b_batch, dd);
            if (++idx[dd] < batch[dd]) break;
            pa -= batch_stride(a, a_batch, dd) * batch[dd];
            pb -= batch_stride(b, b_batch, dd) * batch[dd];
            idx[dd] = 0;
        }
    }
    const std::int64_t item = out.itemsize();
    if (d == DType::Int64) {
        // Our own loop: tasks of whole output rows, about kParallelTask multiply-adds
        // each (rows of c are independent, so the bits do not depend on the cut).
        const std::int64_t rows = std::max<std::int64_t>(1, kParallelTask / std::max<std::int64_t>(k * n, 1));
        const std::int64_t per_batch = (m + rows - 1) / rows;
        parallel::parallel_for(count * per_batch, [&](std::int64_t task) {
            const std::int64_t t = task / per_batch, r0 = (task % per_batch) * rows, r1 = std::min(m, r0 + rows);
            const Mat am{starts[t].first + r0 * a_rs, r1 - r0, k, a_rs, a_cs}, bm{starts[t].second, k, n, b_rs, b_cs};
            int_matmul(am, bm, reinterpret_cast<std::int64_t*>(out.data + (t * m + r0) * n * item));
        });
    } else {
        auto one_batch = [&](std::int64_t t) {
            const Mat am{starts[t].first, m, k, a_rs, a_cs}, bm{starts[t].second, k, n, b_rs, b_cs};
            std::byte* pc = out.data + t * m * n * item;
            visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
                if constexpr (ops::is_float<T> || ops::is_cplx<T>) blas_matmul(am, bm, reinterpret_cast<T*>(pc));
            });
        };
        // BLAS: Accelerate threads internally on large matrices; their batches
        // go one after another so the two never compete. Small ones it runs on
        // the calling thread: those batches go to the pool, several per task
        // (docs/design/array.md §9.4). Each matrix is still one BLAS call of
        // its own, so the bits do not change.
        const std::int64_t work = m * n * k;
        if (count > 1 && work <= kSmallBlas && parallel::available()) {
            const std::int64_t per_task = std::max<std::int64_t>(1, kBlasTaskWork / std::max<std::int64_t>(work, 1));
            parallel::parallel_for((count + per_task - 1) / per_task, [&](std::int64_t task) {
                for (std::int64_t t = task * per_task; t < std::min(count, (task + 1) * per_task); ++t) one_batch(t);
            });
        } else {
            // On the calling thread (Accelerate may thread inside): a call of real
            // size lets other Python threads run meanwhile, as cov's BLAS call does.
            // BLAS and the packing touch no Python objects.
            std::optional<nb::gil_scoped_release> release;
            if (count * work >= kBlasTaskWork) release.emplace();
            for (std::int64_t t = 0; t < count; ++t) one_batch(t);
        }
    }

    // Drop the vector axes; the result is contiguous, so this only renames.
    std::vector<std::int64_t> shape = batch;
    if (!a_vec) shape.push_back(m);
    if (!b_vec) shape.push_back(n);
    if (shape.empty()) return load_python(d, out.data);  // vector @ vector
    out.ndim = static_cast<int>(shape.size());
    std::int64_t stride = out.itemsize();
    for (int dd = out.ndim - 1; dd >= 0; --dd) {
        out.shape[dd] = shape[dd];
        out.strides[dd] = stride;
        stride *= std::max<std::int64_t>(shape[dd], 1);
    }
    out.update_flags();
    return wrap(std::move(out));
}

void bind_matmul(nb::module_& m) { m.def("matmul", named("matmul", &matmul), nb::arg().none(), nb::arg().none()); }

}  // namespace sd
