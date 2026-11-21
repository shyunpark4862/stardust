// The loop of the elementwise iterative methods (docs/design/integrate.md §6.2, decisions C4, C16;
// scipy 1.18.1's _lib/_elementwise_iterative_method.py): every element of a batch is solved on its
// own, and one iteration evaluates the user's function once on the points of the elements still in
// progress (in their order; optimize O38), then updates each of them. Shared by sd.optimize's
// find_root, find_minimum, bracket_root and bracket_minimum (csrc/optimize/batch.cpp) and the
// calculus area's derivative, tanhsinh and nsum.
//
// The parts:
//   - for_elements: a pass over the elements in progress, split over the threads from
//     kParallel elements (without the GIL);
//   - gather: the user's args (one value per element, O39) at the elements in progress;
//   - Call: one call of the user's function on k points, one per element ((k,), the optimize
//     functions) or p per element ((k, p) in C order with args (k, 1), the calculus functions;
//     p may change from one call to the next, and an element with fewer points fills the rest
//     of its row, with NaN as scipy does), its values checked;
//   - Progress: the elements in progress and the status, iteration and evaluation counts of the
//     ones that stopped (scipy's eim._loop counts);
//   - advance: the update of the elements in progress, then the compression to the ones that
//     go on, in their order;
//   - read and out: the vectors of the Python layer's float64 and int64 1-D arrays; npsign,
//     npmin, npmax: numpy's sign, minimum and maximum, which scipy's array steps use.
// Each element's state is a structure of its algorithm (in memory together, so that a pass reads
// it in order). Each element's arithmetic is the same on any thread and the elements that go on
// keep their order, so the results are the same for any thread count.
#pragma once

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace sd::elementwise {

namespace nb = nanobind;

using vec = std::vector<double>;
using ivec = std::vector<std::int64_t>;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// scipy's status codes (eim._ECONVERGED ... _EINPUTERR, _ECONTINUE). -4 (_ECALLBACK) is not
// used: there is no callback (O37); nsum gives -4 its own meaning (docs/design/integrate.md §2.2).
enum Status : int {
    sConverged = 0,
    sSignErr = -1,
    sConvErr = -2,
    sValueErr = -3,
    sInputErr = -5,
    sInProgress = 1,
};

inline double npsign(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return x == 0 ? 0.0 : kNaN;
}

// numpy's minimum and maximum: NaN when either is.
inline double npmin(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return kNaN;
    return a <= b ? a : b;
}
inline double npmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return kNaN;
    return a >= b ? a : b;
}

// A float64 1-D sd.ndarray argument (from the Python layer) as a vector.
inline vec read(nb::handle o) {
    const NDArray& a = as_ndarray(o);
    vec v(static_cast<std::size_t>(a.shape[0]));
    for (std::size_t i = 0; i < v.size(); ++i)
        std::memcpy(&v[i], a.data + static_cast<std::int64_t>(i) * a.strides[0], sizeof(double));
    return v;
}

inline nb::object out(const vec& v) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(v.size())};
    NDArray a = new_array(DType::Float64, shape);
    if (!v.empty()) std::memcpy(a.data, v.data(), v.size() * sizeof(double));
    return wrap(std::move(a));
}

inline nb::object out(const ivec& v) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(v.size())};
    NDArray a = new_array(DType::Int64, shape);
    if (!v.empty()) std::memcpy(a.data, v.data(), v.size() * sizeof(std::int64_t));
    return wrap(std::move(a));
}

// From this many elements in progress the passes over them (the next points, the arguments'
// gathering, the updates) split over the threads, in tasks of kTask elements: an element's work
// is a few ns (docs/design/optimize.md §3.3.1).
constexpr std::size_t kParallel = std::size_t{1} << 15;
constexpr std::size_t kTask = std::size_t{1} << 13;

// fn(r0, r1) over [0, k) in pieces (no Python objects inside)
template <class Fn>
void for_elements(std::size_t k, Fn&& fn) {
    if (k >= kParallel && parallel::available()) {
        const auto tasks = static_cast<std::int64_t>((k + kTask - 1) / kTask);
        parallel::parallel_for(tasks, [&](std::int64_t t) {
            const std::size_t r0 = static_cast<std::size_t>(t) * kTask;
            fn(r0, std::min(k, r0 + kTask));
        });
    } else {
        parallel::Unlocked u(k >= kParallel);
        fn(std::size_t{0}, k);
    }
}

// dst[q] = src[e(q)] for q < k (items of isz bytes; src's stride st), e(q) = idx[q] mod n (idx[q]
// < 2 n; q when idx is null). The values are copied as they are, so any split gives the same.
inline void gather(
    std::byte* dst,
    const std::byte* src,
    std::int64_t st,
    std::int64_t isz,
    const std::int64_t* idx,
    std::int64_t n,
    std::int64_t k
) {
    for_elements(static_cast<std::size_t>(k), [=](std::size_t q0, std::size_t q1) {
        const auto b = static_cast<std::int64_t>(q0), c = static_cast<std::int64_t>(q1);
        auto elem = [=](std::int64_t q) {
            const std::int64_t e = idx ? idx[q] : q;
            return e >= n ? e - n : e;
        };
        if (isz == 8) {
            auto* out = reinterpret_cast<std::uint64_t*>(dst);
            for (std::int64_t q = b; q < c; ++q) {
                std::uint64_t v;
                std::memcpy(&v, src + elem(q) * st, 8);
                out[q] = v;
            }
        } else {
            for (std::int64_t q = b; q < c; ++q)
                std::memcpy(dst + q * isz, src + elem(q) * st, static_cast<std::size_t>(isz));
        }
    });
}

// One call of the user's function on the points of k elements. start(k) prepares one point per
// element (a new float64 sd.ndarray (k,), each arg (k,)), start(k, p) p points per element ((k, p)
// in C order, each arg (k, 1)). put writes a point (into the array and a copy kept here: the
// function may change its argument), run(idx) calls f(x, *args) with each of the user's args
// (1-D, one value per element, O39) gathered at the elements idx[r] mod n (all of them when idx is
// null) into a new array, and value reads f's values. A returned value that is not a float64
// sd.ndarray of the argument's shape goes through check(value, k) for (k,) and
// check(value, (k, p)) for (k, p), which returns one or raises.
class Call {
public:
    Call(nb::handle f, nb::handle check, nb::handle args, std::size_t n)
        : f_(f), check_(check), n_(static_cast<std::int64_t>(n)) {
        for (nb::handle a : nb::borrow<nb::tuple>(args)) args_.push_back(&as_ndarray(a));
    }

    void start(std::size_t k) {
        k_ = static_cast<std::int64_t>(k);
        p_ = 1;
        rows_ = false;
        x_.resize(k);
        const std::int64_t shape[1] = {k_};
        NDArray a = new_array(DType::Float64, shape);
        buf_ = reinterpret_cast<double*>(a.data);
        arr_ = wrap(std::move(a));
    }
    void start(std::size_t k, std::size_t p) {
        k_ = static_cast<std::int64_t>(k);
        p_ = static_cast<std::int64_t>(p);
        rows_ = true;
        x_.resize(k * p);
        const std::int64_t shape[2] = {k_, p_};
        NDArray a = new_array(DType::Float64, shape);
        buf_ = reinterpret_cast<double*>(a.data);
        arr_ = wrap(std::move(a));
    }
    void put(std::size_t r, double x) {
        x_[r] = x;
        buf_[r] = x;
    }
    void put(std::size_t r, std::size_t j, double x) { put(r * static_cast<std::size_t>(p_) + j, x); }
    double x(std::size_t r) const { return x_[r]; }
    double x(std::size_t r, std::size_t j) const { return x_[r * static_cast<std::size_t>(p_) + j]; }

    void run(const std::int64_t* idx) {
        if (k_ == 0) return;
        nb::object r;
        if (args_.empty()) {
            r = f_(arr_);
        } else {
            nb::tuple call = nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(args_.size() + 1)));
            PyTuple_SET_ITEM(call.ptr(), 0, arr_.release().ptr());
            const std::int64_t shape[2] = {k_, 1};
            const std::span<const std::int64_t> sh(shape, rows_ ? 2 : 1);
            for (std::size_t j = 0; j < args_.size(); ++j) {
                const NDArray& src = *args_[j];
                NDArray a = new_array(src.dtype, sh);
                // the element of q: idx[q] < 2 n (bracket_root's sides), so mod n is one subtraction
                gather(a.data, src.data, src.strides[0], src.itemsize(), idx, n_, k_);
                PyTuple_SET_ITEM(call.ptr(), static_cast<Py_ssize_t>(j + 1), wrap(std::move(a)).release().ptr());
            }
            PyObject* out = PyObject_Call(f_.ptr(), call.ptr(), nullptr);
            if (!out) throw nb::python_error();
            r = nb::steal(out);
        }
        arr_ = nb::object();
        bool ok = false;
        if (is_ndarray(r)) {
            const NDArray& v = as_ndarray(r);
            ok = v.dtype == DType::Float64 && v.ndim == (rows_ ? 2 : 1) && v.shape[0] == k_ &&
                 (!rows_ || v.shape[1] == p_);
        }
        if (!ok) r = rows_ ? check_(r, nb::make_tuple(k_, p_)) : check_(r, k_);
        const NDArray& v = as_ndarray(r);
        vp_ = v.data;
        vs_ = v.strides[0];
        vs1_ = rows_ ? v.strides[1] : 0;
        res_ = std::move(r);
    }
    double value(std::size_t r) const {
        double v;
        std::memcpy(&v, vp_ + static_cast<std::int64_t>(r) * vs_, sizeof v);
        return v;
    }
    double value(std::size_t r, std::size_t j) const {
        double v;
        std::memcpy(&v, vp_ + static_cast<std::int64_t>(r) * vs_ + static_cast<std::int64_t>(j) * vs1_, sizeof v);
        return v;
    }

    // f at the points xs (all of them, one per element), into fx.
    void all(const vec& xs, vec& fx) {
        start(xs.size());
        for (std::size_t r = 0; r < xs.size(); ++r) put(r, xs[r]);
        run(nullptr);
        fx.resize(xs.size());
        for (std::size_t r = 0; r < xs.size(); ++r) fx[r] = value(r);
    }

private:
    nb::handle f_, check_;
    std::int64_t n_;
    std::vector<const NDArray*> args_;
    nb::object arr_, res_;
    double* buf_ = nullptr;
    vec x_;
    const std::byte* vp_ = nullptr;
    std::int64_t vs_ = 0, vs1_ = 0, k_ = 0, p_ = 1;
    bool rows_ = false;
};

// The elements in progress (in order) and the bookkeeping of the ones that stopped.
struct Progress {
    std::vector<std::int64_t> active;
    ivec status, nit, nfev;

    explicit Progress(std::size_t n) : status(n, sInProgress), nit(n, 0), nfev(n, 0) {
        active.resize(n);
        for (std::size_t i = 0; i < n; ++i) active[i] = static_cast<std::int64_t>(i);
    }

    void stopped(std::size_t i, std::int64_t it, std::int64_t fev) {
        nit[i] = it;
        nfev[i] = fev;
    }

    // max_iter reached: the rest stop with -2.
    void finish(std::int64_t it, std::int64_t fev) {
        for (std::int64_t i : active) {
            const auto u = static_cast<std::size_t>(i);
            status[u] = sConvErr;
            nit[u] = it;
            nfev[u] = fev;
        }
        active.clear();
    }
};

// The update of the elements in progress: goes_on(r) for the r-th of them (it updates the element
// and records it when it stops), then the ones that go on stay in progress, in their order.
template <class Fn>
void advance(Progress& P, std::vector<std::uint32_t>& on, Fn&& goes_on) {
    const std::size_t k = P.active.size();
    std::size_t w = 0;
    if (k >= kParallel && parallel::available()) {
        on.resize(k);
        for_elements(k, [&](std::size_t r0, std::size_t r1) {
            for (std::size_t r = r0; r < r1; ++r) on[r] = goes_on(r);
        });
        for (std::size_t r = 0; r < k; ++r)
            if (on[r]) P.active[w++] = P.active[r];
    } else {
        parallel::Unlocked u(k >= kParallel);
        for (std::size_t r = 0; r < k; ++r)
            if (goes_on(r)) P.active[w++] = P.active[r];
    }
    P.active.resize(w);
}

}  // namespace sd::elementwise
