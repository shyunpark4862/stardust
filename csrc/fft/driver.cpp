// The axis driver (driver.hpp; docs/design/fft.md §3.7).
#include "fft/driver.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <array>
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

// The stride (doubles) of a line of `count` doubles in a task's block buffer: a multiple of 16 (the lines
// stay 128-byte aligned) and never a multiple of a large power of two, so the lines of a block do not
// fall into the same cache sets (a length of 2^k would otherwise put every line of the block there).
int64_t line_stride(int64_t count) { return (count + 15) / 16 * 16 + 16; }

// One of a task's buffers. Up to kScratchKeep doubles it is the thread's own scratch, kept from task to
// task and call to call (a fresh allocation per task costs page faults, which the threads of a call
// serialise on; milestone FT3); a larger one is allocated for the task. A task runs on one thread from
// start to end and tasks do not nest, so a slot has one user at a time.
inline constexpr int64_t kScratchKeep = int64_t{1} << 16;  // doubles per slot: 512 KB
enum Slot : int { kWork = 0, kIn = 1, kRes = 2, kTmp = 3, kSlots = 4 };

struct Scratch {
    double* p = nullptr;
    Buffer own;
    double* get() const { return p; }
};

Scratch scratch(Slot slot, int64_t count) {
    thread_local std::array<Buffer, kSlots> mem;
    thread_local std::array<int64_t, kSlots> cap{};
    Scratch s;
    if (count <= kScratchKeep) {
        if (cap[slot] < count) {
            mem[slot] = doubles(count);
            cap[slot] = count;
        }
        s.p = mem[slot].get();
    } else {
        s.own = doubles(count);
        s.p = s.own.get();
    }
    return s;
}

Scratch maybe_scratch(bool need, Slot slot, int64_t count) {  // a task takes only the buffers its path uses
    return need ? scratch(slot, count) : Scratch();
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

// Entry j of a real line (bool, int64 or float64) at p as a double.
inline double read_real(DType t, const std::byte* p) {
    switch (t) {
        case DType::Float64: {
            double v;
            std::memcpy(&v, p, 8);
            return v;
        }
        case DType::Int64: {
            int64_t v;
            std::memcpy(&v, p, 8);
            return static_cast<double>(v);
        }
        default: return *reinterpret_cast<const bool*>(p) ? 1.0 : 0.0;
    }
}

// The lines of a block: one on the direct path, else as many as fit a scratch slot (line_doubles: the longer
// of a line's input and output buffers), at most kBlockLines and at most lines / kBlockShare (a small call
// still makes tasks for every thread).
int64_t block_lines(bool direct, int64_t line_doubles, int64_t lines) {
    if (direct) return 1;
    const int64_t fit = kScratchKeep / line_stride(line_doubles);
    return std::clamp<int64_t>(std::min(fit, (lines + kBlockShare - 1) / kBlockShare), 1, kBlockLines);
}

// The lines of a task (decisions FT40, FT50): enough for kTaskWork units of work and for at most kMaxTasks
// tasks, a multiple of the block. The sizes alone decide it, and a line's bits never depend on it.
int64_t lines_per_task(int64_t lines, int64_t cost, int64_t block) {
    int64_t per = std::max<int64_t>({(kTaskWork + cost - 1) / cost, (lines + kMaxTasks - 1) / kMaxTasks, 1});
    return (per + block - 1) / block * block;
}

// The lines of x along `axis` and their partners in out, split into tasks of consecutive lines as
// c2c_axis does (sizes alone decide). make_task() gives a task's callable, run(xp, op, g) on g <= block
// lines at the input and output pointers xp[k], op[k], which owns the task's buffers.
template <class MakeTask>
void drive_lines(const NDArray& x, int axis, const NDArray& out, int64_t cost, int64_t block, MakeTask make_task) {
    const int nd = x.ndim;
    std::vector<int64_t> dims, xs, os;
    int64_t lines = 1;
    for (int d = 0; d < nd; ++d) {
        if (d == axis) continue;
        dims.push_back(x.shape[static_cast<size_t>(d)]);
        xs.push_back(x.strides[static_cast<size_t>(d)]);
        os.push_back(out.strides[static_cast<size_t>(d)]);
        lines *= x.shape[static_cast<size_t>(d)];
    }
    if (lines == 0) return;
    const double total = static_cast<double>(lines) * static_cast<double>(cost);
    const int64_t per_task = lines_per_task(lines, cost, block);
    const int64_t ntasks = (lines + per_task - 1) / per_task;
    const bool big = total >= static_cast<double>(kParallelWork);
    const std::byte* const xbase = x.data;
    std::byte* const obase = out.data;
    const int64_t nother = static_cast<int64_t>(dims.size());

    auto task = [&](int64_t t) {
        const int64_t l0 = t * per_task, l1 = std::min(lines, l0 + per_task);
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
        auto run = make_task();
        const std::byte* xp[kBlockLines];
        std::byte* op[kBlockLines];
        for (int64_t l = l0; l < l1; l += block) {
            const int64_t g = std::min(block, l1 - l);
            for (int64_t k = 0; k < g; ++k) {
                xp[k] = xbase + xoff, op[k] = obase + ooff;
                advance();
            }
            run(xp, op, g);
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


// y (count interleaved complex) times s, conjugated when Conj: one rounding per part (the conjugation is
// exact, so its order with the scaling does not matter).
void finish_line(double* y, int64_t count, double s, bool conj) {
    if (s != 1.0) scale_line(y, 2 * count, s);
    if (conj) {
        for (int64_t k = 0; k < count; ++k) y[2 * k + 1] = -y[2 * k + 1];
    }
}

// The Hermitian projection of an odd-length complex spectrum t (n entries) into y (count <= n entries:
// k = 0 .. count - 1): y_0 = Re t_0 + 0i, y_k = (t_k + conj t_(n-k)) / 2.
void hermitian_part(const double* t, double* y, int64_t n, int64_t count) {
    y[0] = t[0], y[1] = 0.0;
    const float64x2_t half = vdupq_n_f64(0.5);
    const float64x2_t conj = {1.0, -1.0};
    for (int64_t k = 1; k < count; ++k) {
        const float64x2_t a = vld1q_f64(t + 2 * k);
        const float64x2_t b = vmulq_f64(vld1q_f64(t + 2 * (n - k)), conj);
        vst1q_f64(y + 2 * k, vmulq_f64(vaddq_f64(a, b), half));
    }
}

// y_(n-k) = conj y_k for k = 1 .. (n - 1) / 2: the whole spectrum from its first n / 2 + 1 entries.
void mirror(double* y, int64_t n) {
    for (int64_t k = 1; 2 * k < n; ++k) {
        y[2 * (n - k)] = y[2 * k];
        y[2 * (n - k) + 1] = -y[2 * k + 1];
    }
}

}  // namespace

void r2c_axis(const NDArray& x, int axis, int64_t n, bool full, bool conj, double scale, NDArray& out) {
    if (out.size() == 0) return;  // no line: no plan either (n may be any length)
    const bool even = n % 2 == 0;
    const std::shared_ptr<const Plan> plan = get_plan(even ? Kind::Real : Kind::Complex, n);
    const int64_t nin = x.shape[static_cast<size_t>(axis)];
    const int64_t xstride = x.strides[static_cast<size_t>(axis)];
    const int64_t ostride = out.strides[static_cast<size_t>(axis)];
    const int64_t cnt = std::min(nin, n);
    const int64_t half = n / 2 + 1;
    const int64_t nout = full ? n : half;
    const DType dt = x.dtype;
    const bool xdirect = even && dt == DType::Float64 && xstride == 8 && nin >= n;
    const bool xcontig = dt == DType::Float64 && xstride == 8;
    const bool odirect = ostride == 16;
    const int64_t block = block_lines(xdirect && odirect, std::max(n, 2 * nout), out.size() / nout);
    const int64_t work_doubles = plan->work_doubles();
    const int64_t ldi = line_stride(n), ldo = line_stride(2 * nout);

    auto make_task = [&] {
        return [&, work = scratch(kWork, work_doubles), in = maybe_scratch(!xdirect, kIn, block * ldi),
                res = maybe_scratch(!odirect, kRes, block * ldo),
                tmp = maybe_scratch(!even, kTmp, 4 * n)](const std::byte* const* xp, std::byte* const* op, int64_t g) {
            if (!xdirect) {  // gather: float64 values, cut or zero-padded to n
                double* const b = in.get();
                if (xcontig) {
                    for (int64_t k = 0; k < g; ++k) std::memcpy(b + k * ldi, xp[k], static_cast<size_t>(8 * cnt));
                } else {
                    for (int64_t i = 0; i < cnt; ++i) {
                        for (int64_t k = 0; k < g; ++k) b[k * ldi + i] = read_real(dt, xp[k] + i * xstride);
                    }
                }
                if (cnt < n) {
                    for (int64_t k = 0; k < g; ++k)
                        std::memset(b + k * ldi + cnt, 0, static_cast<size_t>(8 * (n - cnt)));
                }
            }
            for (int64_t k = 0; k < g; ++k) {
                const double* src = xdirect ? reinterpret_cast<const double*>(xp[k]) : in.get() + k * ldi;
                double* y = odirect ? reinterpret_cast<double*>(op[k]) : res.get() + k * ldo;
                if (even) {
                    plan->r2c(src, y, work.get());
                } else {
                    double* const c = tmp.get();
                    double* const t = tmp.get() + 2 * n;
                    for (int64_t i = 0; i < n; ++i) c[2 * i] = src[i], c[2 * i + 1] = 0.0;
                    plan->execute(c, t, work.get(), false);
                    hermitian_part(t, y, n, half);
                }
                if (full) mirror(y, n);
                finish_line(y, nout, scale, conj);
            }
            if (!odirect) {  // scatter
                const double* const r = res.get();
                for (int64_t i = 0; i < nout; ++i) {
                    for (int64_t k = 0; k < g; ++k) std::memcpy(op[k] + i * ostride, r + k * ldo + 2 * i, 16);
                }
            }
        };
    };
    drive_lines(x, axis, out, plan->cost() + n, block, make_task);
}

void c2r_axis(const NDArray& x, int axis, int64_t n, bool conj, double scale, NDArray& out) {
    if (out.size() == 0) return;  // no line: no plan either
    const bool even = n % 2 == 0;
    const std::shared_ptr<const Plan> plan = get_plan(even ? Kind::Real : Kind::Complex, n);
    const int64_t nin = x.shape[static_cast<size_t>(axis)];
    const int64_t xstride = x.strides[static_cast<size_t>(axis)];
    const int64_t ostride = out.strides[static_cast<size_t>(axis)];
    const int64_t half = n / 2 + 1;
    const int64_t cnt = std::min(nin, half);
    const DType dt = x.dtype;
    const bool xdirect = dt == DType::Complex128 && xstride == 16 && nin >= half;
    const bool xcontig = dt == DType::Complex128 && xstride == 16;
    const bool odirect = ostride == 8;
    const int64_t block = block_lines(xdirect && odirect, std::max(2 * half, n), out.size() / n);
    const int64_t work_doubles = plan->work_doubles();
    const int64_t ldi = line_stride(2 * half), ldo = line_stride(n);

    auto make_task = [&] {
        return [&, work = scratch(kWork, work_doubles), in = maybe_scratch(!xdirect, kIn, block * ldi),
                res = maybe_scratch(!odirect, kRes, block * ldo),
                tmp = scratch(kTmp, even ? n : 4 * n)](const std::byte* const* xp, std::byte* const* op, int64_t g) {
            if (!xdirect) {  // gather: complex values, cut or zero-padded to n / 2 + 1
                double* const b = in.get();
                if (xcontig) {
                    for (int64_t k = 0; k < g; ++k) std::memcpy(b + k * ldi, xp[k], static_cast<size_t>(16 * cnt));
                } else {
                    for (int64_t i = 0; i < cnt; ++i) {
                        for (int64_t k = 0; k < g; ++k) read_entry(dt, xp[k] + i * xstride, b + k * ldi + 2 * i);
                    }
                }
                if (cnt < half) {
                    for (int64_t k = 0; k < g; ++k) {
                        std::memset(b + k * ldi + 2 * cnt, 0, static_cast<size_t>(16 * (half - cnt)));
                    }
                }
            }
            for (int64_t k = 0; k < g; ++k) {
                const double* src = xdirect ? reinterpret_cast<const double*>(xp[k]) : in.get() + k * ldi;
                double* y = odirect ? reinterpret_cast<double*>(op[k]) : res.get() + k * ldo;
                if (even) {
                    if (conj) {
                        plan->c2r<true>(src, y, tmp.get(), work.get());
                    } else {
                        plan->c2r<false>(src, y, tmp.get(), work.get());
                    }
                } else {  // the Hermitian extension (x_0's imaginary part dropped), one complex inverse
                    double* const e = tmp.get();
                    double* const t = tmp.get() + 2 * n;
                    e[0] = src[0], e[1] = 0.0;
                    for (int64_t i = 1; i < half; ++i) {
                        const double re = src[2 * i], im = conj ? -src[2 * i + 1] : src[2 * i + 1];
                        e[2 * i] = re, e[2 * i + 1] = im;
                        e[2 * (n - i)] = re, e[2 * (n - i) + 1] = -im;
                    }
                    plan->execute(e, t, work.get(), true);
                    for (int64_t i = 0; i < n; ++i) y[i] = t[2 * i];
                }
                if (scale != 1.0) scale_line(y, n, scale);
            }
            if (!odirect) {  // scatter
                const double* const r = res.get();
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t k = 0; k < g; ++k) std::memcpy(op[k] + i * ostride, r + k * ldo + i, 8);
                }
            }
        };
    };
    drive_lines(x, axis, out, plan->cost() + n, block, make_task);
}

void c2c_axis(
    const NDArray& x, int axis, int64_t n, bool inverse, double scale, NDArray& out, bool conj_in, bool conj_out
) {
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
    const bool in_place = x.data == out.data;
    const bool direct = dt == DType::Complex128 && xstride == 16 && ostride == 16 && nin >= n && !in_place && !conj_in;
    const bool xcontig = dt == DType::Complex128 && xstride == 16;
    const bool ocontig = ostride == 16;
    const int64_t block = block_lines(direct, 2 * n, lines);

    const int64_t cost = plan->cost();
    const double total = static_cast<double>(lines) * static_cast<double>(cost);
    const int64_t per_task = lines_per_task(lines, cost, block);
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
        const Scratch work = scratch(kWork, work_doubles);
        if (direct) {
            for (int64_t l = l0; l < l1; ++l) {
                double* y = reinterpret_cast<double*>(obase + ooff);
                plan->execute(reinterpret_cast<const double*>(xbase + xoff), y, work.get(), inverse);
                finish_line(y, n, scale, conj_out);
                advance();
            }
            return;
        }
        const int64_t ld = line_stride(2 * n);
        const Scratch in = scratch(kIn, block * ld), res = scratch(kRes, block * ld);
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
                for (int64_t k = 0; k < g; ++k) std::memcpy(in.get() + k * ld, xp[k], static_cast<size_t>(16 * cnt));
            } else if (dt == DType::Complex128) {
                for (int64_t i = 0; i < cnt; ++i) {
                    for (int64_t k = 0; k < g; ++k) std::memcpy(in.get() + k * ld + 2 * i, xp[k] + i * xstride, 16);
                }
            } else {
                for (int64_t i = 0; i < cnt; ++i) {
                    for (int64_t k = 0; k < g; ++k) read_entry(dt, xp[k] + i * xstride, in.get() + k * ld + 2 * i);
                }
            }
            if (conj_in) {
                for (int64_t k = 0; k < g; ++k) {
                    double* const b = in.get() + k * ld;
                    for (int64_t i = 0; i < cnt; ++i) b[2 * i + 1] = -b[2 * i + 1];
                }
            }
            if (cnt < n) {
                for (int64_t k = 0; k < g; ++k) {
                    std::memset(in.get() + k * ld + 2 * cnt, 0, static_cast<size_t>(16 * (n - cnt)));
                }
            }
            // transform
            for (int64_t k = 0; k < g; ++k) {
                double* y = res.get() + k * ld;
                plan->execute(in.get() + k * ld, y, work.get(), inverse);
                finish_line(y, n, scale, conj_out);
            }
            // scatter
            if (ocontig) {
                for (int64_t k = 0; k < g; ++k) std::memcpy(op[k], res.get() + k * ld, static_cast<size_t>(16 * n));
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t k = 0; k < g; ++k) std::memcpy(op[k] + i * ostride, res.get() + k * ld + 2 * i, 16);
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
