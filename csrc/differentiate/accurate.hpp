// The accurate difference Jacobian (docs/design/integrate.md §6.1, decisions C3, C15): the 5-point
// differences of sd.optimize's approx_fprime and check_grad (their default) and of
// sd.regression.nls's Jacobian at the solution (regression G70, G88); stardust/differentiate/
// _differences.py documents the method and keeps the Python entry point (Differences). The loop
// runs here so that a call of g costs the call and little around it: the points, the stencil
// sums, the rounding levels and the step search work on plain buffers, and only g itself is
// Python (this file is included once, by bind/differentiate_bind.cpp).
//
// g(theta) returns the m values at theta: a float64 sd.ndarray of shape (m,), or a Python float
// when m = 1; anything else goes through the Python callable `check` (which converts it or
// raises). theta is a list of p floats, or a new float64 sd.ndarray (array_x). Each returned
// array is read once, in place, right after the call (the stencil's sum, the largest magnitudes
// and the finiteness test in one pass), so a g that returns one buffer it overwrites each time
// gets the same Jacobian; the central stencil's arrays at -h and +h are kept (referenced) for
// second(), as the Python loop kept them.
//
// The passes over the m values run on the pool from 2^16 values, in chunks fixed by m alone;
// they are elementwise, or reductions that are exact (maxima, flags) or linalg's fast sums
// (chunked by the length only). So every bit, and every step the search chooses, is the same
// on any thread count (guardrail d: the steps are chosen by comparisons of these norms and
// maxima).
#pragma once

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "linalg/fastsum.hpp"
#include "linalg/input.hpp"
#include "parallel/pool.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace sd::differentiate {

namespace accurate_detail {

namespace nb = nanobind;
using vec = std::vector<double>;
using lint = std::int64_t;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// the steps: powers of two from 2^-8 |x_j| down to 2^-30 |x_j| (_differences.py)
constexpr int kFdStart = -8, kFdEnd = -30;
constexpr double kNoiseFactor = 16.0;
constexpr double kU = 0x1p-53;
constexpr double kRoundingTol = 0x1p-26;
constexpr double kGrowTol = 0x1p-36;
constexpr double kSettled = 0x1p-20;
// the largest start of a search, as an exponent: x_j = 0's 2^-8 of a variable of size 1
// (frexp(1) = 0.5 * 2^1, so 2^(1 - 8) = 2^-7)
constexpr int kCapExp = 1 + kFdStart;

// The passes over m values: chunks of kTask values on the pool from kParallelMin values.
constexpr lint kTask = lint(1) << 15;
constexpr lint kParallelMin = lint(1) << 16;

// body(i0, i1, t) over [0, m) in the chunks; t is the chunk's index (0 when serial: one chunk).
template <class Body>
void chunks(lint m, const Body& body) {
    if (m >= kParallelMin && parallel::available()) {
        const lint nt = (m + kTask - 1) / kTask;
        // a pass costs about 1 ns per value: few long tasks (4 at 10^5) wake every worker at once
        parallel::parallel_for(nt, [&](std::int64_t t) { body(t * kTask, std::min(m, (t + 1) * kTask), t); }, kTask);
    } else {
        body(0, m, 0);
    }
}
inline lint nchunks(lint m) { return m >= kParallelMin ? (m + kTask - 1) / kTask : 1; }

// The 2-norm of v[0..n): the fast sum of squares, rescaled when it under- or overflows.
inline double norm2(const double* v, lint n) {
    if (n == 0) return 0.0;
    const double s = linalg::fast::sum_squares(n, v, 1);
    const double r = linalg::fast::norm_from_squares(s, n);
    if (r >= 0) return r;
    double mx = 0;
    bool nan = false;
    for (lint i = 0; i < n; ++i) {
        const double a = std::abs(v[i]);
        if (std::isnan(a)) nan = true;
        else mx = std::max(mx, a);
    }
    if (nan) return kNaN;
    if (mx == 0 || std::isinf(mx)) return mx;
    double t = 0;
    for (lint i = 0; i < n; ++i) {
        const double q = v[i] / mx;
        t += q * q;
    }
    return mx * std::sqrt(t);
}

// The largest of |a[i]| and of |a[i] - b[i]| (b may be null: then only the first), NaN
// propagated as sd.max does (exact, so any order).
inline void max_abs2(const double* a, const double* b, lint m, double& ma, double& md) {
    const lint nt = nchunks(m);
    std::vector<double> pa(static_cast<std::size_t>(nt), 0.0), pd(static_cast<std::size_t>(nt), 0.0);
    chunks(m, [&](lint i0, lint i1, lint t) {
        double xa = 0, xd = 0;
        bool na = false, nd = false;
        for (lint i = i0; i < i1; ++i) {
            const double u = std::abs(a[i]);
            na = na || std::isnan(u);
            xa = std::max(xa, u);
            if (b) {
                const double v = std::abs(a[i] - b[i]);
                nd = nd || std::isnan(v);
                xd = std::max(xd, v);
            }
        }
        pa[std::size_t(t)] = na ? kNaN : xa;
        pd[std::size_t(t)] = nd ? kNaN : xd;
    });
    ma = md = 0;
    for (lint t = 0; t < nt; ++t) {
        const double u = pa[std::size_t(t)], v = pd[std::size_t(t)];
        ma = std::isnan(u) || std::isnan(ma) ? kNaN : std::max(ma, u);
        md = std::isnan(v) || std::isnan(md) ? kNaN : std::max(md, v);
    }
}

inline int frexp_exp(double v) {
    int e = 0;
    std::frexp(v, &e);
    return e;
}

// The stencils: central, and one-sided at a bound or where g is not finite on one side
// (offsets in steps h, weights; the denominator is 12 h).
struct Stencil {
    int count;
    int offs[5];
    double wts[5];
    int lo, hi;
    double wsum;
};
inline constexpr Stencil kStencils[3] = {
    {4, {-2, -1, 1, 2, 0}, {1.0, -8.0, 8.0, -1.0, 0.0}, -2, 2, 18.0},
    {5, {0, 1, 2, 3, 4}, {-25.0, 48.0, -36.0, 16.0, -3.0}, 0, 4, 128.0},
    {5, {0, -1, -2, -3, -4}, {25.0, -48.0, 36.0, -16.0, 3.0}, -4, 0, 128.0},
};

// The m values of one call of g: the array it returned, read in place (obj keeps it alive), or
// a copy (a float).
struct Values {
    nb::object obj;
    const std::byte* data = nullptr;
    std::int64_t stride = 0;  // bytes
    vec own;
    double at(lint i) const {
        double v;
        std::memcpy(&v, data + i * stride, sizeof v);
        return v;
    }
    // contiguous doubles (the data, or a copy into scratch)
    const double* contiguous(lint m, vec& scratch) const {
        if (stride == static_cast<std::int64_t>(sizeof(double))) return reinterpret_cast<const double*>(data);
        scratch.resize(static_cast<std::size_t>(m));
        for (lint i = 0; i < m; ++i) scratch[std::size_t(i)] = at(i);
        return scratch.data();
    }
};

}  // namespace accurate_detail

class AccurateDiff {
public:
    using vec = accurate_detail::vec;
    using lint = accurate_detail::lint;
    using Values = accurate_detail::Values;

    AccurateDiff(
        nanobind::object g, lint p, vec lb, vec ub, nanobind::object weights, bool array_x, nanobind::object check
    )
        : g_(std::move(g)),
          check_(std::move(check)),
          p_(p),
          lb_(std::move(lb)),
          ub_(std::move(ub)),
          array_x_(array_x),
          steps_(static_cast<std::size_t>(p), accurate_detail::kNaN),
          central_(static_cast<std::size_t>(p)),
          noise_(static_cast<std::size_t>(p), 0.0) {
        namespace nb = nanobind;
        if (static_cast<lint>(lb_.size()) != p_ || static_cast<lint>(ub_.size()) != p_)
            throw nb::value_error("Differences: the bounds must have p values");
        if (!weights.is_none()) {
            linalg::Input w(weights, "Differences", "weights");
            if (w.dtype() != DType::Float64 || w.ndim() != 1)
                throw nb::type_error("Differences: weights must be a 1-dimensional float64 array");
            has_w_ = true;
            w_.resize(static_cast<std::size_t>(w.shape(0)));
            for (lint i = 0; i < w.shape(0); ++i)
                std::memcpy(&w_[std::size_t(i)], w.data() + i * w.stride(0), sizeof(double));
        }
    }

    // The steps per variable (None: not chosen yet).
    nanobind::list steps() const {
        nanobind::list out;
        for (double h : steps_) {
            if (std::isnan(h)) out.append(nanobind::none());
            else out.append(h);
        }
        return out;
    }
    void set_steps(nanobind::handle seq) {
        namespace nb = nanobind;
        nb::sequence s = nb::borrow<nb::sequence>(seq);
        if (static_cast<lint>(nb::len(s)) != p_) throw nb::value_error("Differences: steps must have p values");
        for (lint j = 0; j < p_; ++j) {
            nb::object v = s[static_cast<std::size_t>(j)];
            steps_[static_cast<std::size_t>(j)] = v.is_none() ? accurate_detail::kNaN : nb::cast<double>(v);
        }
    }
    bool too_narrow() const { return too_narrow_; }
    nanobind::list noise() const {
        nanobind::list out;
        for (double v : noise_) out.append(v);
        return out;
    }

    // (J, bad): the (m, p) Jacobian at x (g0 = g(x)), rows scaled by the weights, with the steps
    // chosen at the first call; bad is the first variable whose column no stencil gives (J is
    // None then). zero_noise: a column within 16 times its rounding level is set to 0 (G88).
    nanobind::tuple jacobian(const vec& x, nanobind::handle g0o, bool zero_noise) {
        namespace nb = nanobind;
        using namespace accurate_detail;
        check_x(x);
        load_g0(g0o);
        const lint m = m_;
        for (auto& c : central_) {
            c.set = false;
            c.fm = Values{};
            c.fp = Values{};
        }
        too_narrow_ = false;
        std::vector<vec> cols(static_cast<std::size_t>(p_));
        xw_ = x;
        for (lint j = 0; j < p_; ++j) {
            bool ok = false, fits = false;
            const auto uj = static_cast<std::size_t>(j);
            vec& col = cols[uj];
            col.resize(static_cast<std::size_t>(m));
            if (!std::isnan(steps_[uj])) {
                ok = column(j, steps_[uj], col.data());
                fits = fits_;
            }
            if (!ok) {
                double h = kNaN;
                ok = choose_step(j, col.data(), h);
                steps_[uj] = h;
                fits = fits || fits_;
            }
            if (!ok) {
                too_narrow_ = !fits;
                return nb::make_tuple(nb::none(), j);
            }
        }
        // the weighted columns; with zero_noise a column within 16 times its rounding level is 0:
        // kept, a noise column has a direction of its own and can take the place of an identified
        // parameter's column (a exp(-k t) + c at a large k)
        vec keep(static_cast<std::size_t>(p_), 1.0);
        for (lint j = 0; j < p_; ++j) {
            vec& col = cols[std::size_t(j)];
            if (has_w_)
                chunks(m, [&](lint i0, lint i1, lint) {
                    for (lint i = i0; i < i1; ++i) col[std::size_t(i)] = col[std::size_t(i)] * w_[std::size_t(i)];
                });
            if (zero_noise && norm2(col.data(), m) <= kNoiseFactor * noise_[std::size_t(j)]) keep[std::size_t(j)] = 0.0;
        }
        const std::int64_t shape[2] = {m, p_};
        NDArray out = new_array(DType::Float64, shape);
        double* o = reinterpret_cast<double*>(out.data);
        chunks(m, [&](lint i0, lint i1, lint) {
            for (lint i = i0; i < i1; ++i)
                for (lint j = 0; j < p_; ++j) {
                    const double v = cols[std::size_t(j)][std::size_t(i)];
                    o[i * p_ + j] = zero_noise ? v * keep[std::size_t(j)] : v;
                }
        });
        release();
        return nb::make_tuple(wrap(std::move(out)), nb::none());
    }

    // scipy's approx_fprime with steps eps (one per variable): (g(x + h_j e_j) - g0) /
    // ((x_j + h_j) - x_j), h_j = sqrt(eps) sign(x_j) max(1, |x_j|) where x_j + eps_j rounds to
    // x_j. Returns (m, p).
    nanobind::object forward(const vec& x, nanobind::handle g0o, const vec& eps) {
        namespace nb = nanobind;
        check_x(x);
        load_g0(g0o);
        if (static_cast<lint>(eps.size()) != p_) throw nb::value_error("Differences: eps must have p values");
        const lint m = m_;
        const std::int64_t shape[2] = {m, p_};
        NDArray out = new_array(DType::Float64, shape);
        double* o = reinterpret_cast<double*>(out.data);
        Values v;
        xw_ = x;
        for (lint j = 0; j < p_; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            const double t = x[uj];
            double h = eps[uj];
            if ((t + h) - t == 0) h = 0x1p-26 * (t >= 0 ? 1.0 : -1.0) * std::max(1.0, std::abs(t));
            xw_[uj] = t + h;
            const double dx = (t + h) - t;
            call(v);
            xw_[uj] = t;
            for (lint i = 0; i < m; ++i) o[i * p_ + j] = (v.at(i) - g0_[std::size_t(i)]) / dx;
        }
        return wrap(std::move(out));
    }

    // S = sum_i c_i d^2 g_i / dx^2 over the variables idx (0 elsewhere), by central differences
    // with the Jacobian's steps (its kept values for the diagonal); None when a point is outside
    // the bounds or not finite. Returns a list of p lists.
    nanobind::object second(const vec& x, nanobind::handle g0o, nanobind::handle co, const std::vector<lint>& idx) {
        namespace nb = nanobind;
        using namespace accurate_detail;
        check_x(x);
        load_g0(g0o);
        const lint m = m_;
        vec c(static_cast<std::size_t>(m));
        {
            linalg::Input ci(co, "Differences", "c");
            if (ci.dtype() != DType::Float64 || ci.ndim() != 1 || ci.shape(0) != m)
                throw nb::value_error("Differences.second: c must be float64 of g's length");
            for (lint i = 0; i < m; ++i) std::memcpy(&c[std::size_t(i)], ci.data() + i * ci.stride(0), sizeof(double));
        }
        vec hs(static_cast<std::size_t>(p_));
        for (lint j = 0; j < p_; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            if (central_[uj].set) hs[uj] = central_[uj].h;
            else if (!std::isnan(steps_[uj])) hs[uj] = steps_[uj];
            else hs[uj] = std::ldexp(1.0, frexp_exp(x[uj] != 0 ? std::abs(x[uj]) : 1.0) + kFdStart - 2);
        }
        for (lint j : idx) {
            const auto uj = static_cast<std::size_t>(j);
            if (j < 0 || j >= p_) throw nb::index_error("Differences.second: index out of range");
            if (!(lb_[uj] <= x[uj] - hs[uj] && x[uj] + hs[uj] <= ub_[uj])) return nb::none();
        }
        vec scratch;
        auto dotc = [&](const double* v) { return m ? linalg::fast::dot(m, c.data(), 1, v, 1) : 0.0; };
        auto dotv = [&](const Values& v) { return dotc(v.contiguous(m, scratch)); };
        const double c0 = dotc(g0_.data());
        Values v;
        xw_ = x;
        // phi(j, sj, k, sk): c . g(x + sj h_j e_j + sk h_k e_k) (k < 0: none); false when g is
        // not finite there
        auto phi = [&](lint j, double sj, lint k, double sk, double& out) {
            const auto uj = static_cast<std::size_t>(j);
            xw_[uj] = x[uj] + sj * hs[uj];
            if (k >= 0) xw_[std::size_t(k)] = x[std::size_t(k)] + sk * hs[std::size_t(k)];
            call(v);
            xw_[uj] = x[uj];
            if (k >= 0) xw_[std::size_t(k)] = x[std::size_t(k)];
            if (!finite(v)) return false;
            out = dotv(v);
            return true;
        };
        std::vector<vec> S(static_cast<std::size_t>(p_), vec(static_cast<std::size_t>(p_), 0.0));
        for (std::size_t a = 0; a < idx.size(); ++a) {
            const lint j = idx[a];
            const auto uj = static_cast<std::size_t>(j);
            double fm, fp;
            if (central_[uj].set) {
                fm = dotv(central_[uj].fm);
                fp = dotv(central_[uj].fp);
            } else if (!phi(j, -1.0, -1, 0.0, fm) || !phi(j, 1.0, -1, 0.0, fp)) {
                return nb::none();
            }
            S[uj][uj] = (fp - 2 * c0 + fm) / (hs[uj] * hs[uj]);
            for (std::size_t b = 0; b < a; ++b) {
                const lint k = idx[b];
                const auto uk = static_cast<std::size_t>(k);
                double pp, pm, mp, mm;
                if (!phi(j, 1.0, k, 1.0, pp) || !phi(j, 1.0, k, -1.0, pm) || !phi(j, -1.0, k, 1.0, mp) ||
                    !phi(j, -1.0, k, -1.0, mm))
                    return nb::none();
                const double val = (pp - pm - mp + mm) / (4 * hs[uj] * hs[uk]);
                S[uj][uk] = val;
                S[uk][uj] = val;
            }
        }
        nb::list out;
        for (const vec& row : S) {
            nb::list r;
            for (double t : row) r.append(t);
            out.append(r);
        }
        return out;
    }

private:
    struct First {
        bool use = false;
        bool ok = false;
        const vec* col = nullptr;
        bool fits = false;
    };
    struct Central {
        bool set = false;
        double h = 0;
        Values fm, fp;
    };

    nanobind::object g_, check_;
    lint p_;
    vec lb_, ub_;
    bool array_x_;
    bool has_w_ = false;
    vec w_;
    vec steps_;
    std::vector<Central> central_;
    vec noise_;
    bool fits_ = false;
    bool too_narrow_ = false;
    bool settled_ = false;
    lint m_ = 0;
    vec g0_;
    vec xw_;   // the point g is called at
    vec acc_;  // the stencil's sum
    vec big_;  // the values' largest magnitudes (then weighted)
    vec tmp_;  // a weighted column
    Values vals_[5];

    // The references to g's last arrays let go (the central ones stay for second()).
    void release() {
        for (Values& v : vals_) v = Values{};
    }

    void check_x(const vec& x) const {
        if (static_cast<lint>(x.size()) != p_) throw nanobind::value_error("Differences: x must have p values");
    }

    void load_g0(nanobind::handle g0o) {
        namespace nb = nanobind;
        linalg::Input in(g0o, "Differences", "g0");
        if (in.dtype() != DType::Float64 || in.ndim() != 1)
            throw nb::type_error("Differences: g0 must be a 1-dimensional float64 array");
        m_ = in.shape(0);
        if (has_w_ && static_cast<lint>(w_.size()) != m_)
            throw nb::value_error("Differences: the weights must have g's length");
        g0_.resize(static_cast<std::size_t>(m_));
        for (lint i = 0; i < m_; ++i) std::memcpy(&g0_[std::size_t(i)], in.data() + i * in.stride(0), sizeof(double));
        acc_.resize(static_cast<std::size_t>(m_));
        big_.resize(static_cast<std::size_t>(m_));
        tmp_.resize(static_cast<std::size_t>(m_));
    }

    nanobind::object arg() const {
        namespace nb = nanobind;
        if (array_x_) {
            const std::int64_t shape[1] = {p_};
            NDArray a = new_array(DType::Float64, shape);
            if (p_) std::memcpy(a.data, xw_.data(), static_cast<std::size_t>(p_) * sizeof(double));
            return wrap(std::move(a));
        }
        nb::list a;
        for (double v : xw_) a.append(v);
        return a;
    }

    // g(xw_) into v (the returned array referenced, or a float copied).
    void call(Values& v) {
        namespace nb = nanobind;
        nb::object o = g_(arg());
        const lint m = m_;
        if (m == 1 && PyFloat_Check(o.ptr())) {
            v.own.assign(1, PyFloat_AS_DOUBLE(o.ptr()));
            v.obj = nb::object();
            v.data = reinterpret_cast<const std::byte*>(v.own.data());
            v.stride = sizeof(double);
            return;
        }
        auto fits = [&](nb::handle h) {
            if (!is_ndarray(h)) return false;
            const NDArray& a = as_ndarray(h);
            return a.dtype == DType::Float64 && a.ndim == 1 && a.shape[0] == m;
        };
        if (!fits(o)) {
            if (check_.is_none()) throw nb::type_error("Differences: g must return a float64 array of g0's length");
            o = check_(o);
            if (!fits(o)) throw nb::type_error("Differences: check must return a float64 array of g0's length");
        }
        const NDArray& a = as_ndarray(o);
        v.data = a.data;
        v.stride = a.strides[0];
        v.obj = std::move(o);
    }

    // Whether every value is finite (the exponent bits tested as integers).
    bool finite(const Values& v) const {
        using namespace accurate_detail;
        constexpr std::uint64_t kExp = 0x7ff0000000000000ull;
        const lint nt = nchunks(m_);
        std::vector<std::uint8_t> bad(static_cast<std::size_t>(nt), 0);
        chunks(m_, [&](lint i0, lint i1, lint t) {
            std::uint64_t b = 0;
            for (lint i = i0; i < i1; ++i) {
                std::uint64_t u;
                std::memcpy(&u, v.data + i * v.stride, sizeof u);
                b |= static_cast<std::uint64_t>((u & kExp) == kExp);
            }
            bad[std::size_t(t)] = b != 0;
        });
        for (std::uint8_t b : bad)
            if (b) return false;
        return true;
    }

    // dg/dx_j (unweighted) into out by the first feasible 5-point stencil with step h; false
    // when none gives one. Keeps the central stencil's values for second() and sets noise_[j],
    // the weighted 2-norm of the column's rounding level ((sum |w_o|) / (12 h) u
    // max_o |g_i(x + o h e_j)|), and fits_ (some stencil fitted within the bounds).
    bool column(lint j, double h, double* out) {
        using namespace accurate_detail;
        constexpr std::uint64_t kExp = 0x7ff0000000000000ull;
        const auto uj = static_cast<std::size_t>(j);
        const double t = xw_[uj];
        const lint m = m_;
        const lint nt = nchunks(m);
        std::vector<std::uint8_t> bad(static_cast<std::size_t>(nt), 0);
        double* acc = acc_.data();
        double* big = big_.data();
        fits_ = false;
        for (int kind = 0; kind < 3; ++kind) {
            const Stencil& st = kStencils[kind];
            if (!(lb_[uj] <= t + st.lo * h && t + st.hi * h <= ub_[uj])) continue;
            fits_ = true;
            bool ok = true;
            // each value is added into the sum (in the order of the offsets) and the largest
            // magnitudes as it comes, with its finiteness test; a value that is not finite
            // ends the stencil (its later points are not evaluated)
            for (int q = 0; q < st.count && ok; ++q) {
                const int o = st.offs[q];
                const double w = st.wts[q];
                const std::byte* d;
                std::int64_t s;
                if (o == 0) {
                    d = reinterpret_cast<const std::byte*>(g0_.data());
                    s = sizeof(double);
                } else {
                    xw_[uj] = t + o * h;
                    call(vals_[q]);
                    xw_[uj] = t;
                    d = vals_[q].data;
                    s = vals_[q].stride;
                }
                const bool first = q == 0;
                chunks(m, [&](lint i0, lint i1, lint c) {
                    std::uint64_t b = 0;
                    if (s == static_cast<std::int64_t>(sizeof(double))) {
                        const double* v = reinterpret_cast<const double*>(d);
                        for (lint i = i0; i < i1; ++i) {
                            const double vi = v[i];
                            std::uint64_t u;
                            std::memcpy(&u, &vi, sizeof u);
                            b |= static_cast<std::uint64_t>((u & kExp) == kExp);
                            if (first) {
                                acc[i] = w * vi;
                                big[i] = std::abs(vi);
                            } else {
                                acc[i] = acc[i] + w * vi;
                                big[i] = std::max(big[i], std::abs(vi));
                            }
                        }
                    } else {
                        for (lint i = i0; i < i1; ++i) {
                            double vi;
                            std::memcpy(&vi, d + i * s, sizeof vi);
                            std::uint64_t u;
                            std::memcpy(&u, &vi, sizeof u);
                            b |= static_cast<std::uint64_t>((u & kExp) == kExp);
                            if (first) {
                                acc[i] = w * vi;
                                big[i] = std::abs(vi);
                            } else {
                                acc[i] = acc[i] + w * vi;
                                big[i] = std::max(big[i], std::abs(vi));
                            }
                        }
                    }
                    bad[std::size_t(c)] = b != 0;
                });
                for (std::uint8_t b : bad) ok = ok && !b;
            }
            if (!ok) continue;
            const double den = 12.0 * h;
            const double* wr = has_w_ ? w_.data() : nullptr;
            chunks(m, [&](lint i0, lint i1, lint) {
                for (lint i = i0; i < i1; ++i) {
                    if (wr) big[i] = big[i] * wr[i];
                    out[i] = acc[i] / den;
                }
            });
            noise_[uj] = m ? st.wsum / (12.0 * h) * kU * norm2(big, m) : 0.0;
            if (kind == 0) {  // the values at -h and +h, kept for second()
                Central& c = central_[uj];
                c.set = true;
                c.h = h;
                std::swap(c.fm, vals_[1]);
                std::swap(c.fp, vals_[2]);
            }
            return true;
        }
        return false;
    }

    // The 2-norm of a column weighted as the Jacobian (noise_'s scale).
    double wnorm(const double* col) {
        using namespace accurate_detail;
        if (!has_w_) return norm2(col, m_);
        double* b = tmp_.data();
        chunks(m_, [&](lint i0, lint i1, lint) {
            for (lint i = i0; i < i1; ++i) b[i] = col[i] * w_[std::size_t(i)];
        });
        return norm2(b, m_);
    }

    // noise_[j] (of the column just computed) relative to the column's size: inf for 0.
    double rounding(const double* col, lint j) {
        const double size = wnorm(col);
        return size > 0 ? noise_[std::size_t(j)] / size : accurate_detail::kInf;
    }

    // The step of variable j (_differences.py's choose_step): the column into out and its
    // step h; false when no stencil gives one.
    bool choose_step(lint j, double* out, double& h) {
        using namespace accurate_detail;
        const auto uj = static_cast<std::size_t>(j);
        const lint m = m_;
        const int base = frexp_exp(xw_[uj] != 0 ? std::abs(xw_[uj]) : 1.0);
        const int e0 = base + kFdStart;
        vec col0(static_cast<std::size_t>(m));
        const bool ok0 = column(j, std::ldexp(1.0, e0), col0.data());
        const First start{true, ok0, &col0, fits_};
        if (!ok0 || e0 >= kCapExp) return descend(j, e0, start, true, out, h);
        const double r0 = rounding(col0.data(), j);
        if (r0 <= kGrowTol) {
            bool ok = descend(j, e0, start, true, out, h);
            if (ok && !settled_) from_cap(j, out, h);
            return ok;
        }
        const bool fits0 = fits_;
        int e = 0;
        if (grow(j, e0, col0.data(), e)) {
            if (descend(j, e, First{}, true, out, h)) return true;
        }
        bool ok;
        if (r0 <= kRoundingTol) ok = descend(j, e0, First{}, true, out, h);  // the plain search from the start
        else ok = descend(j, e0, First{}, false, out, h);
        fits_ = fits_ || fits0;
        return ok;
    }

    // The search from 2^kCapExp in place of (out, h) when its estimates agree (_from_cap).
    void from_cap(lint j, double* out, double& h) {
        const bool fits = fits_;
        vec alt(static_cast<std::size_t>(m_));
        double halt = accurate_detail::kNaN;
        const bool ok = descend(j, accurate_detail::kCapExp, First{}, true, alt.data(), halt);
        if (ok && settled_) {
            fits_ = fits_ || fits;
            std::copy(alt.begin(), alt.end(), out);
            h = halt;
            return;
        }
        fits_ = fits;
    }

    // The exponent e of the step to start from when the start 2^e0 is at rounding level
    // (_grow); false when no step leaves rounding level.
    bool grow(lint j, int e0, const double* col0, int& eout) {
        using namespace accurate_detail;
        const lint m = m_;
        const double r = rounding(col0, j);
        long jump;
        if (std::isinf(r)) jump = kCapExp - e0;
        else jump = static_cast<long>(std::ceil(std::log2(r / kRoundingTol)));
        int e = static_cast<int>(std::min<long>(kCapExp, e0 + std::max<long>(jump, 2)));
        vec col(static_cast<std::size_t>(m)), nxt(static_cast<std::size_t>(m)), diff(static_cast<std::size_t>(m));
        if (!column(j, std::ldexp(1.0, e), col.data())) return false;
        double lvl = noise_[std::size_t(j)];
        while (e + 2 <= kCapExp) {
            if (!column(j, std::ldexp(1.0, e + 2), nxt.data())) break;
            const double nlvl = noise_[std::size_t(j)];
            e += 2;
            for (lint i = 0; i < m; ++i) diff[std::size_t(i)] = nxt[std::size_t(i)] - col[std::size_t(i)];
            const bool shows = wnorm(diff.data()) > lvl + nlvl;
            std::swap(col, nxt);
            lvl = nlvl;
            if (shows) break;
        }
        const double size = wnorm(col.data());
        if (!(size > 0 && lvl / size * kNoiseFactor < 1)) return false;
        eout = e;
        return true;
    }

    // The search from 2^e0 down by factors of 4 (12 steps): the estimate where consecutive ones
    // stop getting closer by more than 8 times (_descend). first: the column at 2^e0 when just
    // computed (use; ok: a stencil gave one; its fits). strict: the second estimate is not taken
    // over the first when it is rounding noise and the first is not. Sets settled_.
    bool descend(lint j, int e0, const First& first, bool strict, double* out, double& hbest) {
        using namespace accurate_detail;
        const lint m = m_;
        bool have_best = false, have_prev = false, prev_ok = false, fits = false;
        double dprev = kInf;
        vec col(static_cast<std::size_t>(m)), prev(static_cast<std::size_t>(m));
        hbest = kNaN;
        for (int e = e0; e > e0 + kFdEnd - kFdStart - 1; e -= 2) {
            const double h = std::ldexp(1.0, e);
            bool ok;
            if (e == e0 && first.use) {  // the start's column, computed by the caller
                ok = first.ok;
                if (ok) std::copy(first.col->begin(), first.col->end(), col.begin());
                fits_ = first.fits;
            } else {
                ok = column(j, h, col.data());
            }
            fits = fits || fits_;
            if (!ok) {
                if (have_best) break;
                continue;
            }
            if (!have_prev) {
                std::copy(col.begin(), col.end(), out);
                hbest = h;
                have_best = true;
                prev_ok = strict && rounding(col.data(), j) * kNoiseFactor < 1;
            } else {
                if (prev_ok && dprev == kInf && rounding(col.data(), j) * kNoiseFactor >= 1) break;
                double sc, md;
                max_abs2(col.data(), prev.data(), m, sc, md);
                const double dd = sc > 0 ? md / sc : 0.0;
                if (dd >= dprev / 8) break;
                std::copy(col.begin(), col.end(), out);
                hbest = h;
                dprev = dd;
                if (dd == 0) break;
            }
            std::swap(prev, col);
            have_prev = true;
        }
        fits_ = fits;
        settled_ = dprev <= kSettled;
        return have_best;
    }
};

inline void bind_accurate(nanobind::module_& m) {
    namespace nb = nanobind;
    nb::class_<AccurateDiff>(m, "AccurateDiff")
        .def(
            nb::init<
                nb::object, std::int64_t, std::vector<double>, std::vector<double>, nb::object, bool, nb::object>(),
            nb::arg("g"), nb::arg("p"), nb::arg("lb"), nb::arg("ub"), nb::arg("weights").none(), nb::arg("array_x"),
            nb::arg("check").none()
        )
        .def_prop_rw("steps", &AccurateDiff::steps, &AccurateDiff::set_steps)
        .def_prop_ro("too_narrow", &AccurateDiff::too_narrow)
        .def_prop_ro("noise", &AccurateDiff::noise)
        .def("jacobian", &AccurateDiff::jacobian, nb::arg("x"), nb::arg("g0"), nb::arg("zero_noise") = false)
        .def("forward", &AccurateDiff::forward, nb::arg("x"), nb::arg("g0"), nb::arg("eps"))
        .def("second", &AccurateDiff::second, nb::arg("x"), nb::arg("g0"), nb::arg("c"), nb::arg("idx"));
}

}  // namespace sd::differentiate
