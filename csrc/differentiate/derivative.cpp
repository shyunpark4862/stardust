// derivative, jacobian and hessian (docs/design/integrate.md §3.8, decisions C26-C38; scipy 1.18.1's
// differentiate/_differentiate.py), the same steps and evaluation points as scipy (C5).
//
// The method is scipy's: a difference formula of order 2n (n = (order + 1) // 2 terms) repeated
// with the step divided by the step factor c, the error estimate the change between two
// iterations. The central stencil is x + h t_i at the nodes -1/c^(n-1), ..., -1, 0, 1, ...,
// 1/c^(n-1); the first iteration evaluates its 2n points, each next one the two new innermost
// points (the step h/c in place of h) and keeps the rest. The one-sided stencil (step_direction
// != 0) is x +- h t_i at 0, 1, 1/d, ..., 1/d^(2n-1) with d = c**0.5: 2n points first, then two.
// The weights are the exact ones correctly rounded (C28; stardust/differentiate/_weights.py),
// so df may differ from scipy's (LAPACK-solved weights) in the last digits; the points are
// computed with scipy's expressions and are scipy's to the bit.
//
// The element's tests after each iteration, in scipy's order: converged (0) when error <
// atol + rtol |df|, else a non-finite x or df (-3, df NaN), else the error grown tenfold over
// the last iteration's (-1); max_iter reached is -2. Points and tests are elementwise and each
// element's arithmetic is the same on any thread (its sum in a fixed order: the central one over
// the pairs of nodes, C38; the one-sided one in the stencil's order).
//
// derivative: the elementwise family's loop (csrc/elementwise/loop.hpp; optimize O38): the first
// call f(x) on all elements (k,), then each iteration one call on the elements in progress,
// (k, p) with p = 2n then 2 (C26), the user's args as (k, 1). jacobian: f takes (m, N) and
// returns (k, N) or (N,) (C27); the columns of each call are the variables' stencil points with
// the other variables at x (scipy's xph), and like scipy's (preserve_shape) every column is
// evaluated in each iteration until every entry stopped. hessian: the jacobian of the gradient
// that an inner jacobian computes at every column of the outer's calls (rtol / 100 inside), the
// counts summed as scipy sums them.
#include "differentiate/derivative.hpp"

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "elementwise/loop.hpp"

#include <nanobind/stl/string.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace sd::differentiate {

namespace nb = nanobind;

namespace {

using namespace sd::elementwise;

constexpr int sErrorIncrease = -1;  // scipy's _EERRORINCREASE (derivative's -1)

// The stencil's constants. The powers are scipy's expressions: numpy's c ** arange(n) and
// d ** arange(2n) (libm pow) in the first iteration, Python's c ** (n - 1) and c ** 0.5 (pow).
struct Stencil {
    std::size_t n;
    double c, d, cn;
    vec pc, pd, wc, wr;

    Stencil(std::size_t terms, double factor, vec central, vec one_sided)
        : n(terms),
          c(factor),
          d(std::pow(factor, 0.5)),
          cn(std::pow(factor, static_cast<double>(terms - 1))),
          pc(terms),
          pd(2 * terms),
          wc(std::move(central)),
          wr(std::move(one_sided)) {
        for (std::size_t k = 0; k < n; ++k) pc[k] = std::pow(c, static_cast<double>(k));
        for (std::size_t k = 0; k < 2 * n; ++k) pd[k] = std::pow(d, static_cast<double>(k));
    }
    std::size_t width() const { return 2 * n + 1; }
    std::size_t points(bool first) const { return first ? 2 * n : 2; }
};

// The stencil's side: npsign(step_direction); NaN has none (its element's points are NaN and it
// stops with -3 after the first iteration).
enum Side : int { kLeft = -1, kCentral = 0, kRight = 1, kNone = 2 };

inline int side_of(double dir) {
    const double s = npsign(dir);
    return std::isnan(s) ? kNone : static_cast<int>(s);
}

// The slot of f(x) in an element's stencil values: the centre (central), the first (one-sided).
inline std::size_t fx_slot(const Stencil& S, int side) { return side == kLeft || side == kRight ? 0 : S.n; }

// The points of one element (x, its step h, its side) in this iteration, in scipy's order:
// central x - h/c^(n-1), ..., x - h, x + h, ..., x + h/c^(n-1) first, then x -+ h/c^(n-1);
// one-sided x +- h/d^k (k < 2n) first, then x +- h/c^(n-1), x +- (h/d)/c^(n-1).
inline void points(const Stencil& S, int side, double x, double h, bool first, double* out) {
    const std::size_t n = S.n;
    if (side == kNone) {
        for (std::size_t q = 0, p = S.points(first); q < p; ++q) out[q] = kNaN;
        return;
    }
    if (side == kCentral) {
        if (first) {
            for (std::size_t q = 0; q < n; ++q) {
                out[q] = x + -(h / S.pc[n - 1 - q]);
                out[n + q] = x + h / S.pc[q];
            }
        } else {
            out[0] = x + -h / S.cn;
            out[1] = x + h / S.cn;
        }
        return;
    }
    const bool right = side == kRight;
    if (first) {
        for (std::size_t q = 0; q < 2 * n; ++q) {
            const double hr = h / S.pd[q];
            out[q] = right ? x + hr : x - hr;
        }
    } else {
        const double h0 = h / S.cn, h1 = h / S.d / S.cn;
        out[0] = right ? x + h0 : x - h0;
        out[1] = right ? x + h1 : x - h1;
    }
}

// The one-sided sum: the weights times the values, in the stencil's order.
inline double dot(const vec& w, const double* v) {
    double s = w[0] * v[0];
    for (std::size_t i = 1; i < w.size(); ++i) s += w[i] * v[i];
    return s;
}

// The central sum over the pairs of nodes -t, t (the weights are exactly antisymmetric, C28):
// sum_q w_q (f(x - h t_q) - f(x + h t_q)), outermost pair first. The same value as scipy's
// fc @ wc in exact arithmetic; each difference of two close values is exact (Sterbenz), so the
// cancellation that the plain sum suffers when f(x) is much larger than f'(x) h is gone: on the
// collection the largest relative error falls from 3.2e-10 to 6.2e-11 (log at 1e4) and more
// iterations stop where scipy's do (docs/design/integrate.md §3.8, §7.8).
inline double central(const Stencil& S, const double* v) {
    const std::size_t n = S.n;
    double s = S.wc[n - 1] * (v[n - 1] - v[n + 1]);
    for (std::size_t q = n - 1; q-- > 0;) s += S.wc[q] * (v[q] - v[2 * n - q]);
    return s;
}

// One element's estimate and error.
struct Estimate {
    double df = kNaN, err = kNaN;
};

// The update of one element after a call: val(q) the function at its q-th point. v holds its
// stencil's values: central [f(x - h t) for the n left nodes from the innermost out, f(x), the n
// right ones from the outermost in] (scipy's fc); one-sided [f(x), f at the 2n nodes from the
// outermost in] (scipy's fo). The new innermost values enter and the outermost leave. Returns
// the status (sInProgress to go on); h is the step of this iteration (the caller divides it).
template <class Val>
int update(
    const Stencil& S,
    int side,
    double x,
    double h,
    double* v,
    Val&& val,
    bool first,
    Estimate& e,
    double atol,
    double rtol
) {
    const std::size_t n = S.n;
    double s;
    if (side == kLeft || side == kRight) {
        if (first) {
            for (std::size_t q = 0; q < 2 * n; ++q) v[1 + q] = val(q);
        } else {
            for (std::size_t q = 1; q + 1 < 2 * n; ++q) v[q] = v[q + 2];
            v[2 * n - 1] = val(0);
            v[2 * n] = val(1);
        }
        s = dot(S.wr, v);
    } else {
        if (first) {
            for (std::size_t q = 0; q < n; ++q) {
                v[q] = val(q);
                v[n + 1 + q] = val(n + q);
            }
        } else {
            for (std::size_t q = n - 1; q > 0; --q) v[q] = v[q - 1];
            v[0] = val(0);
            for (std::size_t q = n + 1; q < 2 * n; ++q) v[q] = v[q + 1];
            v[2 * n] = val(1);
        }
        s = central(S, v);
        // The centre's weight is 0: in the first iteration f(x) enters as 0 f(x), NaN when f(x) is
        // not finite (scipy's fc @ wc); after it scipy's centre slot holds an earlier point's
        // value, which was finite or the element stopped with -3.
        if (first) s += 0.0 * v[n];
    }
    double df = s / h;
    if (side == kLeft) df = -df;
    if (side == kNone) df = kNaN;
    const double last = e.err;
    e.err = std::abs(df - e.df);
    e.df = df;
    if (e.err < atol + rtol * std::abs(df)) return sConverged;
    if (!(std::isfinite(x) && std::isfinite(df))) {
        e.df = kNaN;
        return sValueErr;
    }
    if (e.err > last * 10) return sErrorIncrease;
    return sInProgress;
}

inline double first_step(double h) { return h > 0 ? h : kNaN; }  // scipy: h0 <= 0 is NaN

// ---- derivative: elementwise

nb::object derivative_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle args,
    nb::handle x,
    nb::handle initial_step,
    nb::handle step_direction,
    nb::handle wc,
    nb::handle wr,
    std::int64_t terms,
    double factor,
    double atol,
    double rtol,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const vec xs = read(x), hs = read(initial_step), ds = read(step_direction);
        const Stencil S(static_cast<std::size_t>(terms), factor, read(wc), read(wr));
        const std::size_t n = xs.size(), w = S.width();
        CallPoints F(f, check, args, n);
        vec fx;
        F.all(xs, fx);  // scipy's first evaluation (eim._initialize): every element at x
        struct E {  // one element's state, together in memory
            double x, h;
            Estimate e;
            int side;
        };
        std::vector<E> el(n);
        vec V(n * w);
        for (std::size_t i = 0; i < n; ++i) {
            el[i] = {xs[i], first_step(hs[i]), Estimate{}, side_of(ds[i])};
            V[i * w + fx_slot(S, el[i].side)] = fx[i];
        }
        Progress P(n);
        std::int64_t nit = 0, nfev = 1;
        std::vector<std::uint32_t> on;
        while (nit < max_iter && !P.active.empty()) {
            const bool first = nit == 0;
            const std::size_t k = P.active.size(), p = S.points(first);
            F.start(k, p);
            for_weighted(k, p, [&](std::size_t r0, std::size_t r1) {
                vec pts(p);
                for (std::size_t r = r0; r < r1; ++r) {
                    const E& e = el[static_cast<std::size_t>(P.active[r])];
                    points(S, e.side, e.x, e.h, first, pts.data());
                    for (std::size_t q = 0; q < p; ++q) F.put(r, q, pts[q]);
                }
            });
            F.run(P.active.data());
            nfev += static_cast<std::int64_t>(p);
            nit += 1;
            advance_weighted(P, on, p, [&](std::size_t r) {
                const auto i = static_cast<std::size_t>(P.active[r]);
                E& e = el[i];
                const int st = update(
                    S, e.side, e.x, e.h, &V[i * w], [&](std::size_t q) { return F.value(r, q); }, first, e.e, atol, rtol
                );
                e.h /= S.c;
                if (st == sInProgress) return true;
                P.status[i] = st;
                P.stopped(i, nit, nfev);
                return false;
            });
        }
        P.finish(nit, nfev);
        vec df(n), err(n);
        for (std::size_t i = 0; i < n; ++i) {
            df[i] = el[i].e.df;
            err[i] = el[i].e.err;
        }
        return nb::make_tuple(out(df), out(err), out(P.status), out(P.nit), out(P.nfev));
    });
}

// ---- jacobian and hessian: f takes (m, N), each column a point

// The user's vectorized function: f(X) for X (m, N) returns (k, N), or (N,) for a scalar
// function. The first call decides which (jacobian) unless the scalar form is required
// (hessian); a value of another shape goes through check(value, (m, N), expect), which
// returns a float64 array of an accepted shape or raises.
class Vectorized {
public:
    static constexpr std::int64_t kEither = -1, kScalar = -2;

    Vectorized(nb::handle f, nb::handle check, std::int64_t expect) : f_(f), check_(check), expect_(expect) {}

    double* start(std::size_t m, std::size_t N) {
        m_ = static_cast<std::int64_t>(m);
        n_ = static_cast<std::int64_t>(N);
        const std::int64_t shape[2] = {m_, n_};
        NDArray a = new_array(DType::Float64, shape);
        double* buf = reinterpret_cast<double*>(a.data);
        arr_ = wrap(std::move(a));
        return buf;
    }

    void run() {
        nb::object r = f_(arr_);
        arr_ = nb::object();
        if (!accepted(r)) r = check_(r, nb::make_tuple(m_, n_), expect_);
        const NDArray& v = as_ndarray(r);
        if (expect_ == kEither) expect_ = v.ndim == 1 ? kScalar : v.shape[0];
        vp_ = v.data;
        s0_ = v.strides[0];
        s1_ = v.ndim == 2 ? v.strides[1] : 0;
        res_ = std::move(r);
    }

    bool scalar() const { return expect_ == kScalar; }
    std::size_t outputs() const { return scalar() ? 1 : static_cast<std::size_t>(expect_); }
    double value(std::size_t i, std::size_t col) const {
        double v;
        const std::int64_t off = scalar() ? static_cast<std::int64_t>(col) * s0_
                                          : static_cast<std::int64_t>(i) * s0_ + static_cast<std::int64_t>(col) * s1_;
        std::memcpy(&v, vp_ + off, sizeof v);
        return v;
    }
    // the evaluations an entry's count gains from a call of p points per column
    std::int64_t count(std::size_t, std::size_t p) const { return static_cast<std::int64_t>(p); }

private:
    bool accepted(nb::handle r) const {
        if (!is_ndarray(r)) return false;
        const NDArray& v = as_ndarray(r);
        if (v.dtype != DType::Float64) return false;
        const bool one = v.ndim == 1 && v.shape[0] == n_, two = v.ndim == 2 && v.shape[1] == n_;
        if (expect_ == kEither) return one || two;
        if (expect_ == kScalar) return one;
        return two && v.shape[0] == expect_;
    }

    nb::handle f_, check_;
    std::int64_t expect_, m_ = 0, n_ = 0;
    nb::object arr_, res_;
    const std::byte* vp_ = nullptr;
    std::int64_t s0_ = 0, s1_ = 0;
};

// A jacobian's entries, (k, m, B) in C order: k outputs, m variables, B base points.
struct Entries {
    std::size_t k = 0;
    bool scalar = false;
    vec df, err;
    ivec status, nit, nfev;
};

// The jacobian of ev's function at the B points x0 (m, B) (C order): column j B + b of a call is
// a point of variable j at base point b, the other variables at x0[:, b]. h0 and dir: one step
// and one side per variable.
template <class Eval>
Entries jacobian_run(
    Eval& ev,
    std::size_t m,
    std::size_t B,
    const double* x0,
    const vec& h0,
    const vec& dir,
    const Stencil& S,
    double atol,
    double rtol,
    std::int64_t max_iter
) {
    const std::size_t C = m * B, w = S.width();
    double* X = ev.start(m, C);  // the first call (scipy's eim._initialize): each column at its base point
    for (std::size_t c = 0; c < m; ++c)
        for (std::size_t col = 0; col < C; ++col) X[c * C + col] = x0[c * B + col % B];
    ev.run();
    Entries o;
    o.k = ev.outputs();
    o.scalar = ev.scalar();
    const std::size_t E = o.k * C;
    struct Col {
        double x, h;
        int side;
    };
    std::vector<Col> cols(C);
    for (std::size_t col = 0; col < C; ++col) {
        const std::size_t j = col / B;
        cols[col] = {x0[col], first_step(h0[j]), side_of(dir[j])};  // x0[j B + b] = x0[col]
    }
    std::vector<Estimate> est(E);
    vec V(E * w);
    ivec acc(E);
    for (std::size_t e = 0; e < E; ++e) {
        const std::size_t i = e / C, col = e % C;
        V[e * w + fx_slot(S, cols[col].side)] = ev.value(i, col);
        acc[e] = ev.count(e, 1);
    }
    Progress P(E);
    std::int64_t nit = 0;
    std::vector<std::uint32_t> on;
    vec pts;
    while (nit < max_iter && !P.active.empty()) {
        const bool first = nit == 0;
        const std::size_t p = S.points(first), N = C * p;
        pts.resize(N);
        for (std::size_t col = 0; col < C; ++col)
            points(S, cols[col].side, cols[col].x, cols[col].h, first, &pts[col * p]);
        X = ev.start(m, N);
        for (std::size_t c = 0; c < m; ++c) {
            double* row = X + c * N;
            for (std::size_t col = 0; col < C; ++col) {
                const double base = x0[c * B + col % B];
                double* dst = row + col * p;
                if (col / B == c) {
                    for (std::size_t q = 0; q < p; ++q) dst[q] = pts[col * p + q];
                } else {
                    for (std::size_t q = 0; q < p; ++q) dst[q] = base;
                }
            }
        }
        ev.run();
        nit += 1;
        advance(P, on, [&](std::size_t r) {
            const auto e = static_cast<std::size_t>(P.active[r]);
            const std::size_t i = e / C, col = e % C;
            const Col& cl = cols[col];
            acc[e] += ev.count(e, p);
            const int st = update(
                S, cl.side, cl.x, cl.h, &V[e * w], [&](std::size_t q) { return ev.value(i, col * p + q); }, first,
                est[e], atol, rtol
            );
            if (st == sInProgress) return true;
            P.status[e] = st;
            P.stopped(e, nit, acc[e]);
            return false;
        });
        for (Col& cl : cols) cl.h /= S.c;  // scipy divides every entry's step, stopped or not
    }
    for (std::int64_t e : P.active) {
        const auto u = static_cast<std::size_t>(e);
        P.status[u] = sConvErr;
        P.stopped(u, nit, acc[u]);
    }
    o.df.resize(E);
    o.err.resize(E);
    for (std::size_t e = 0; e < E; ++e) {
        o.df[e] = est[e].df;
        o.err[e] = est[e].err;
    }
    o.status = std::move(P.status);
    o.nit = std::move(P.nit);
    o.nfev = std::move(P.nfev);
    return o;
}

nb::object entries_out(const Entries& o) {
    return nb::make_tuple(
        out(o.df), out(o.err), out(o.status), out(o.nit), out(o.nfev), static_cast<std::int64_t>(o.k), o.scalar
    );
}

nb::object jacobian_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle x,
    nb::handle initial_step,
    nb::handle step_direction,
    nb::handle wc,
    nb::handle wr,
    std::int64_t terms,
    double factor,
    double atol,
    double rtol,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const vec x0 = read(x), h0 = read(initial_step), dir = read(step_direction);
        const Stencil S(static_cast<std::size_t>(terms), factor, read(wc), read(wr));
        Vectorized ev(f, check, Vectorized::kEither);
        return entries_out(jacobian_run(ev, x0.size(), 1, x0.data(), h0, dir, S, atol, rtol, max_iter));
    });
}

// hessian's outer function: at the outer call's columns X (m, N), the gradient of f by the inner
// jacobian (central, rtol / 100) at the N points; the counts are scipy's: an outer entry (i, j)
// gains the inner evaluations of entry i at the columns of variable j (summed over its points).
class Gradient {
public:
    Gradient(Vectorized& f, const vec& h0, const Stencil& S, double atol, double rtol, std::int64_t max_iter)
        : f_(f), h0_(h0), dir_(h0.size(), 0.0), S_(S), atol_(atol), rtol_(rtol), max_iter_(max_iter) {}

    double* start(std::size_t m, std::size_t N) {
        m_ = m;
        n_ = N;
        X_.assign(m * N, 0.0);
        return X_.data();
    }
    void run() {
        r_ = jacobian_run(f_, m_, n_, X_.data(), h0_, dir_, S_, atol_, rtol_, max_iter_);
        const std::size_t p = n_ / m_;
        counts_.assign(m_ * m_, 0);
        for (std::size_t i = 0; i < m_; ++i)
            for (std::size_t j = 0; j < m_; ++j) {
                std::int64_t s = 0;
                for (std::size_t q = 0; q < p; ++q) s += r_.nfev[i * n_ + j * p + q];
                counts_[i * m_ + j] = s;
            }
    }
    bool scalar() const { return false; }
    std::size_t outputs() const { return m_; }
    double value(std::size_t i, std::size_t col) const { return r_.df[i * n_ + col]; }
    std::int64_t count(std::size_t e, std::size_t) const { return counts_[e]; }

private:
    Vectorized& f_;
    const vec& h0_;
    vec dir_;
    const Stencil& S_;
    double atol_, rtol_;
    std::int64_t max_iter_;
    std::size_t m_ = 0, n_ = 0;
    vec X_;
    Entries r_;
    ivec counts_;
};

nb::object hessian_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle x,
    nb::handle initial_step,
    nb::handle wc,
    nb::handle wr,
    std::int64_t terms,
    double factor,
    double atol,
    double rtol,
    std::int64_t max_iter
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const vec x0 = read(x), h0 = read(initial_step), dir(x0.size(), 0.0);
        const Stencil S(static_cast<std::size_t>(terms), factor, read(wc), read(wr));
        Vectorized ev(f, check, Vectorized::kScalar);
        Gradient g(ev, h0, S, atol, rtol / 100, max_iter);
        return entries_out(jacobian_run(g, x0.size(), 1, x0.data(), h0, dir, S, atol, rtol, max_iter));
    });
}

}  // namespace

void bind_derivative(nb::module_& m) {
    m.def(
        "derivative_solve", &derivative_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"), nb::arg("args"),
        nb::arg("x"), nb::arg("initial_step"), nb::arg("step_direction"), nb::arg("wc"), nb::arg("wr"),
        nb::arg("terms"), nb::arg("factor"), nb::arg("atol"), nb::arg("rtol"), nb::arg("max_iter")
    );
    m.def(
        "jacobian_solve", &jacobian_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"), nb::arg("x"),
        nb::arg("initial_step"), nb::arg("step_direction"), nb::arg("wc"), nb::arg("wr"), nb::arg("terms"),
        nb::arg("factor"), nb::arg("atol"), nb::arg("rtol"), nb::arg("max_iter")
    );
    m.def(
        "hessian_solve", &hessian_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"), nb::arg("x"),
        nb::arg("initial_step"), nb::arg("wc"), nb::arg("wr"), nb::arg("terms"), nb::arg("factor"), nb::arg("atol"),
        nb::arg("rtol"), nb::arg("max_iter")
    );
}

}  // namespace sd::differentiate
