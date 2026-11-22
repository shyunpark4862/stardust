// The axis driver (driver.hpp; docs/design/fft.md §3.7).
#include "fft/driver.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "parallel/pool.hpp"

namespace sd::fft {

namespace {

struct FreeDeleter {
    void operator()(double* p) const { std::free(p); }
};
using Buffer = std::unique_ptr<double[], FreeDeleter>;

Buffer doubles(int64_t count) {  // 128-byte aligned (a cache line; alignment never changes a value)
    const size_t bytes = (static_cast<size_t>(std::max<int64_t>(count, 1)) * sizeof(double) + 127) / 128 * 128;
    void* p = std::aligned_alloc(128, bytes);
    if (p == nullptr) throw std::bad_alloc();
    return Buffer(static_cast<double*>(p));
}

void scale_line(double* y, int64_t count, double s) {  // count doubles times s, one rounding each
    const float64x2_t v = vdupq_n_f64(s);
    int64_t i = 0;
    for (; i + 2 <= count; i += 2) vst1q_f64(y + i, vmulq_f64(vld1q_f64(y + i), v));
    for (; i < count; ++i) y[i] *= s;
}

// Entry j of a line of dtype t at p as (re, im).
inline void read_entry(DType t, const std::byte* p, double* to) {
    switch (t) {
        case DType::Complex128: std::memcpy(to, p, 16); return;
        case DType::Float64: std::memcpy(to, p, 8); break;
        case DType::Int64: {
            int64_t v;
            std::memcpy(&v, p, 8);
            to[0] = static_cast<double>(v);
            break;
        }
        default: to[0] = *reinterpret_cast<const bool*>(p) ? 1.0 : 0.0; break;
    }
    to[1] = 0.0;
}

}  // namespace

void c2c_axis(const NDArray& x, int axis, int64_t n, bool inverse, double scale, NDArray& out) {
    const int nd = x.ndim;
    std::vector<int64_t> dims, xs, os;  // the other axes, in order
    int64_t lines = 1;
    for (int d = 0; d < nd; ++d) {
        if (d == axis) continue;
        dims.push_back(x.shape[static_cast<size_t>(d)]);
        xs.push_back(x.strides[static_cast<size_t>(d)]);
        os.push_back(out.strides[static_cast<size_t>(d)]);
        lines *= x.shape[static_cast<size_t>(d)];
    }
    if (lines == 0) return;
    const std::shared_ptr<const Plan> plan = get_plan(Kind::Complex, n);
    const int64_t nin = x.shape[static_cast<size_t>(axis)];
    const int64_t xstride = x.strides[static_cast<size_t>(axis)];
    const int64_t ostride = out.strides[static_cast<size_t>(axis)];
    const int64_t cnt = std::min(nin, n);
    const DType dt = x.dtype;
    const bool direct = dt == DType::Complex128 && xstride == 16 && ostride == 16 && nin >= n;
    const bool xcontig = dt == DType::Complex128 && xstride == 16;
    const bool ocontig = ostride == 16;
    const int64_t block = direct ? 1 : std::clamp<int64_t>(kBlockDoubles / std::max<int64_t>(4 * n, 1), 1, kBlockLines);

    const int64_t cost = plan->cost();
    const double total = static_cast<double>(lines) * static_cast<double>(cost);
    int64_t per_task = std::max<int64_t>((kTaskWork + cost - 1) / cost, 1);
    per_task = (per_task + block - 1) / block * block;
    const int64_t ntasks = (lines + per_task - 1) / per_task;
    const bool big = total >= static_cast<double>(kParallelWork);
    const int64_t work_doubles = plan->work_doubles();
    const std::byte* const xbase = x.data;
    std::byte* const obase = out.data;
    const int64_t nother = static_cast<int64_t>(dims.size());

    auto task = [&](int64_t t) {
        const int64_t l0 = t * per_task, l1 = std::min(lines, l0 + per_task);
        // the multi-index of line l0 (the last other axis fastest) and its offsets
        std::vector<int64_t> idx(static_cast<size_t>(nother), 0);
        int64_t rest = l0, xoff = 0, ooff = 0;
        for (int64_t d = nother - 1; d >= 0; --d) {
            const int64_t ext = dims[static_cast<size_t>(d)];
            idx[static_cast<size_t>(d)] = rest % ext;
            rest /= ext;
            xoff += idx[static_cast<size_t>(d)] * xs[static_cast<size_t>(d)];
            ooff += idx[static_cast<size_t>(d)] * os[static_cast<size_t>(d)];
        }
        auto advance = [&] {
            for (int64_t d = nother - 1; d >= 0; --d) {
                const size_t u = static_cast<size_t>(d);
                xoff += xs[u], ooff += os[u];
                if (++idx[u] < dims[u]) return;
                xoff -= xs[u] * dims[u], ooff -= os[u] * dims[u];
                idx[u] = 0;
            }
        };
        const Buffer work = doubles(work_doubles);
        if (direct) {
            for (int64_t l = l0; l < l1; ++l) {
                double* y = reinterpret_cast<double*>(obase + ooff);
                plan->execute(reinterpret_cast<const double*>(xbase + xoff), y, work.get(), inverse);
                if (scale != 1.0) scale_line(y, 2 * n, scale);
                advance();
            }
            return;
        }
        const Buffer in = doubles(block * 2 * n), res = doubles(block * 2 * n);
        const std::byte* xp[kBlockLines];
        std::byte* op[kBlockLines];
        for (int64_t l = l0; l < l1; l += block) {
            const int64_t g = std::min(block, l1 - l);
            for (int64_t k = 0; k < g; ++k) {
                xp[k] = xbase + xoff, op[k] = obase + ooff;
                advance();
            }
            // gather
            if (xcontig) {
                for (int64_t k = 0; k < g; ++k) std::memcpy(in.get() + k * 2 * n, xp[k], static_cast<size_t>(16 * cnt));
            } else if (dt == DType::Complex128) {
                for (int64_t i = 0; i < cnt; ++i) {
                    for (int64_t k = 0; k < g; ++k) std::memcpy(in.get() + k * 2 * n + 2 * i, xp[k] + i * xstride, 16);
                }
            } else {
                for (int64_t i = 0; i < cnt; ++i) {
                    for (int64_t k = 0; k < g; ++k) read_entry(dt, xp[k] + i * xstride, in.get() + k * 2 * n + 2 * i);
                }
            }
            if (cnt < n) {
                for (int64_t k = 0; k < g; ++k) {
                    std::memset(in.get() + k * 2 * n + 2 * cnt, 0, static_cast<size_t>(16 * (n - cnt)));
                }
            }
            // transform
            for (int64_t k = 0; k < g; ++k) {
                double* y = res.get() + k * 2 * n;
                plan->execute(in.get() + k * 2 * n, y, work.get(), inverse);
                if (scale != 1.0) scale_line(y, 2 * n, scale);
            }
            // scatter
            if (ocontig) {
                for (int64_t k = 0; k < g; ++k) std::memcpy(op[k], res.get() + k * 2 * n, static_cast<size_t>(16 * n));
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t k = 0; k < g; ++k) std::memcpy(op[k] + i * ostride, res.get() + k * 2 * n + 2 * i, 16);
                }
            }
        }
    };

    const parallel::Unlocked unlocked(big);
    if (big && ntasks > 1) {
        const int64_t task_ns = static_cast<int64_t>(std::min(1e12, static_cast<double>(per_task) * cost / 5.0));
        parallel::parallel_for(ntasks, task, task_ns);
    } else {
        for (int64_t t = 0; t < ntasks; ++t) task(t);
    }
}

}  // namespace sd::fft
