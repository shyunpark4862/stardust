// quad_vec and cubature (docs/design/integrate.md §3.3, §3.4, decisions C5, C94-C107): scipy
// 1.18.1's Python integrators integrate/_quad_vec.py and integrate/_cubature.py (with
// integrate/_rules), the same evaluation points and steps, with the user's function a Python
// callback (the GIL held).
//
// The arithmetic is scipy's, that is numpy's and Python's: no fused multiply-add anywhere (the
// build turns contraction off), every elementwise step in scipy's order, the sums numpy's (a sum
// along the points is numpy's pairwise sum when the reduction runs along memory, a running sum per
// value otherwise; integrate/numpy_ops.hpp), the 2-norm numpy's BLAS dot product, the heaps
// Python's heapq on the same comparisons (ties and NaN included), Python's min and max.
//
//   - quad_vec_solve: scipy's quad_vec. An infinite range becomes (0, 1) or (-1, 1) (SemiInfiniteFunc,
//     DoubleInfiniteFunc: points with |t| below sqrt(DBL_MIN) are 0 without a call), the break
//     points map with them and split the first intervals; each rule (gk21, gk15, trapezoid) gives
//     an integral, its error estimate in the norm ('2' or 'max', abs for a scalar f) and the
//     rounding error; up to 128 intervals of the largest errors are halved per round; the
//     intervals' integrals are kept in scipy's LRU cache of 100e6 bytes (sys.getsizeof of the
//     first integral per entry), an evicted one is computed again when its interval is halved
//     (21 or 15 more calls), as scipy does; the trapezoid rule's halving calls each point once
//     (scipy's lru_cache).
//   - cubature_solve: scipy's cubature with the rules gk21, gk15 (Cartesian products of the 1-D
//     Gauss-Kronrod rules, the lower rule scipy's roots_legendre(10) and (7) to the bit) and
//     genz-malik (degree 7 with the degree 5 rule). Per region two calls: the rule's nodes, then
//     the nodes of both rules (the error estimate |higher - lower|); the argument is (n, ndim) in
//     scipy's layout (C order, Fortran order for genz-malik, whose nodes scipy transposes), and
//     the sum along the points follows the layout of f's value as numpy's reduction does (C100). The region of the largest
//     max|error| is split in 2^ndim halves until every value's error is within atol + rtol
//     |estimate| or max_subdivisions; the initial regions are a plain list (scipy pops from it
//     without heapify). Infinite limits are scipy's transforms, but (-inf, b] calls f at
//     x = b - (1 - t)/t, the transform scipy documents (scipy's code calls f at -b + (1 - t)/t: the
//     integral over [-b, inf), C101); break points with two or more doubly infinite variables map
//     as scipy means them (scipy's inverse raises there, C102); a value that is not finite at the
//     end is status 3, not converged (scipy reports 'converged', C103).
#include "integrate/integrate.hpp"
#include "integrate/numpy_ops.hpp"

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iterator>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sd::integrate {

namespace nb = nanobind;

namespace {

using vec = std::vector<double>;
using numpy::maxabs;
using numpy::norm2_inplace;
using numpy::np_sum;
using numpy::PageBuffer;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kTiny = std::numeric_limits<double>::min();  // sys.float_info.min

// Python's min(a, b) and max(a, b) of floats: b replaces a only when b < a (b > a)
inline double pymin(double a, double b) { return b < a ? b : a; }
inline double pymax(double a, double b) { return b > a ? b : a; }

// ---- Python's heapq (the C module's siftdown, siftup, heappush, heappop) on a vector, lt the
// items' "<"

template <class T, class Lt>
void sift_down(std::vector<T>& h, std::size_t start, std::size_t pos, const Lt& lt) {
    T item = std::move(h[pos]);
    while (pos > start) {
        const std::size_t parent = (pos - 1) >> 1;
        if (!lt(item, h[parent])) break;
        h[pos] = std::move(h[parent]);
        pos = parent;
    }
    h[pos] = std::move(item);
}

template <class T, class Lt>
void sift_up(std::vector<T>& h, std::size_t pos, const Lt& lt) {
    const std::size_t end = h.size(), start = pos, limit = end >> 1;
    T item = std::move(h[pos]);
    while (pos < limit) {
        std::size_t child = 2 * pos + 1;
        if (child + 1 < end && !lt(h[child], h[child + 1])) ++child;
        h[pos] = std::move(h[child]);
        pos = child;
    }
    h[pos] = std::move(item);
    sift_down(h, start, pos, lt);
}

template <class T, class Lt>
void heappush(std::vector<T>& h, T item, const Lt& lt) {
    h.push_back(std::move(item));
    sift_down(h, 0, h.size() - 1, lt);
}

template <class T, class Lt>
T heappop(std::vector<T>& h, const Lt& lt) {
    T last = std::move(h.back());
    h.pop_back();
    if (h.empty()) return last;
    T ret = std::move(h[0]);
    h[0] = std::move(last);
    sift_up(h, 0, lt);
    return ret;
}

template <class T, class Lt>
void heapify(std::vector<T>& h, const Lt& lt) {
    for (std::size_t i = h.size() / 2; i-- > 0;) sift_up(h, i, lt);
}

// The order heappop would give the items of the heap h, as indices into h (h unchanged).
template <class T, class Lt>
std::vector<std::size_t> pop_order(const std::vector<T>& h, const Lt& lt) {
    std::vector<std::size_t> idx(h.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    auto ilt = [&](std::size_t a, std::size_t b) { return lt(h[a], h[b]); };
    std::vector<std::size_t> out;
    out.reserve(idx.size());
    while (!idx.empty()) out.push_back(heappop(idx, ilt));
    return out;
}

std::string shape_text(const std::vector<std::int64_t>& s, bool scalar) {
    if (scalar) return "a scalar";
    std::string t = "(";
    for (std::size_t i = 0; i < s.size(); ++i) t += (i ? ", " : "") + std::to_string(s[i]);
    return t + (s.size() == 1 ? ",)" : ")");
}

nb::object readonly(NDArray&& a) {
    a.writeable = false;
    return wrap(std::move(a));
}

// A float64 array f returned: its data read through its strides (an sd.ndarray, or any object of
// the buffer protocol with format 'd', a numpy array; the buffer is released at the end).
class ArrayRead {
public:
    ArrayRead() = default;
    ArrayRead(const ArrayRead&) = delete;
    ArrayRead& operator=(const ArrayRead&) = delete;
    ~ArrayRead() { release(); }

    // true if o is such an array with at least one dimension
    bool open(nb::handle o) {
        release();
        if (is_ndarray(o)) {
            const NDArray& a = as_ndarray(o);
            if (a.dtype != DType::Float64 || a.ndim < 1) return false;
            data = a.data;
            shape.assign(a.shape.begin(), a.shape.begin() + a.ndim);
            strides.assign(a.strides.begin(), a.strides.begin() + a.ndim);
            return true;
        }
        if (!PyObject_CheckBuffer(o.ptr())) return false;
        if (PyObject_GetBuffer(o.ptr(), &buf_, PyBUF_RECORDS_RO) != 0) {
            PyErr_Clear();
            return false;
        }
        held_ = true;
        const char* f = buf_.format ? buf_.format : "B";
        const bool dbl = buf_.itemsize == 8 && (std::strcmp(f, "d") == 0 || std::strcmp(f, "<d") == 0 ||
                                                std::strcmp(f, "=d") == 0 || std::strcmp(f, "@d") == 0);
        if (!dbl || buf_.ndim < 1) {
            release();
            return false;
        }
        data = static_cast<const std::byte*>(buf_.buf);
        shape.assign(buf_.shape, buf_.shape + buf_.ndim);
        strides.assign(buf_.strides, buf_.strides + buf_.ndim);
        return true;
    }

    std::int64_t size() const {
        std::int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    // the elements in C order from dimension `from` on, at the leading index i (from = 1), or all
    // (from = 0)
    void read(double* out, int from = 0, std::int64_t i = 0) const {
        const std::byte* base = data + (from ? i * strides[0] : 0);
        const int nd = static_cast<int>(shape.size());
        if (nd == from) {
            std::memcpy(out, base, sizeof(double));
            return;
        }
        if (nd == from + 1) {
            const std::int64_t n = shape[static_cast<std::size_t>(from)], st = strides[static_cast<std::size_t>(from)];
            if (st == 8) std::memcpy(out, base, static_cast<std::size_t>(n) * sizeof(double));
            else
                for (std::int64_t q = 0; q < n; ++q) std::memcpy(out + q, base + q * st, sizeof(double));
            return;
        }
        std::int64_t k[64] = {};
        std::int64_t n = 1;
        for (int d = from; d < nd; ++d) n *= shape[static_cast<std::size_t>(d)];
        for (std::int64_t q = 0; q < n; ++q) {
            const std::byte* p = base;
            for (int d = from; d < nd; ++d) p += k[d] * strides[static_cast<std::size_t>(d)];
            std::memcpy(out + q, p, sizeof(double));
            for (int d = nd - 1; d >= from; --d) {
                if (++k[d] < shape[static_cast<std::size_t>(d)]) break;
                k[d] = 0;
            }
        }
    }

    const std::byte* data = nullptr;
    std::vector<std::int64_t> shape, strides;

private:
    void release() {
        if (held_) PyBuffer_Release(&buf_);
        held_ = false;
    }
    Py_buffer buf_{};
    bool held_ = false;
};

// =====================================================================================
// quad_vec

// _quad_vec.py's Gauss-Kronrod nodes and weights (its decimal literals; the 21-point rule's are
// QUADPACK's dqk21, the 15-point dqk15's)
constexpr double kX21[21] = {
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0,
    -0.148874338981631210884826001129720,
    -0.294392862701460198131126603103866,
    -0.433395394129247190799265943165784,
    -0.562757134668604683339000099272694,
    -0.679409568299024406234327365114874,
    -0.780817726586416897063717578345042,
    -0.865063366688984510732096688423493,
    -0.930157491355708226001207180059508,
    -0.973906528517171720077964012084452,
    -0.995657163025808080735527280689003,
};
constexpr double kW10[10] = {
    0.066671344308688137593568809893332, 0.149451349150580593145776339657697, 0.219086362515982043995534934228163,
    0.269266719309996355091226921569469, 0.295524224714752870173892994651338, 0.295524224714752870173892994651338,
    0.269266719309996355091226921569469, 0.219086362515982043995534934228163, 0.149451349150580593145776339657697,
    0.066671344308688137593568809893332,
};
constexpr double kV21[21] = {
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390, 0.054755896574351996031381300244580,
    0.075039674810919952767043140916190, 0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077958109831074, 0.134709217311473325928054001771707, 0.142775938577060080797094273138717,
    0.147739104901338491374841515972068, 0.149445554002916905664936468389821, 0.147739104901338491374841515972068,
    0.142775938577060080797094273138717, 0.134709217311473325928054001771707, 0.123491976262065851077958109831074,
    0.109387158802297641899210590325805, 0.093125454583697605535065465083366, 0.075039674810919952767043140916190,
    0.054755896574351996031381300244580, 0.032558162307964727478818972459390, 0.011694638867371874278064396062192,
};
constexpr double kX15[15] = {
    0.991455371120812639206854697526329,  0.949107912342758524526189684047851,  0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,  0.586087235467691130294144838258730,  0.405845151377397166906606412076961,
    0.207784955007898467600689403773245,  0.000000000000000000000000000000000,  -0.207784955007898467600689403773245,
    -0.405845151377397166906606412076961, -0.586087235467691130294144838258730, -0.741531185599394439863864773280788,
    -0.864864423359769072789712788640926, -0.949107912342758524526189684047851, -0.991455371120812639206854697526329,
};
constexpr double kW7[7] = {
    0.129484966168869693270611432679082, 0.279705391489276667901467771423780, 0.381830050505118944950369775488975,
    0.417959183673469387755102040816327, 0.381830050505118944950369775488975, 0.279705391489276667901467771423780,
    0.129484966168869693270611432679082,
};
constexpr double kV15[15] = {
    0.022935322010529224963732008058970, 0.063092092629978553290700663189204, 0.104790010322250183839876322541518,
    0.140653259715525918745189590510238, 0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
    0.204432940075298892414161999234649, 0.209482141084727828012999174891714, 0.204432940075298892414161999234649,
    0.190350578064785409913256402421014, 0.169004726639267902826583426598550, 0.140653259715525918745189590510238,
    0.104790010322250183839876322541518, 0.063092092629978553290700663189204, 0.022935322010529224963732008058970,
};

enum Rule { kGk21 = 1, kGk15 = 2, kTrapezoid = 3 };
enum Norm { kNorm2 = 0, kNormMax = 1 };
enum Transform { kFinite = 0, kSemi, kDouble };

// The integrand quad_vec's rules see: the user's f at a Python float (through scipy's transform of
// an infinite range), its value a scalar or an array of one shape (the first value decides).
struct VecFn {
    nb::handle f, check;
    int transform = kFinite;
    double start = 0.0;
    int sgn = 1;
    double tmin = std::pow(kTiny, 0.5);  // the transforms' threshold, sys.float_info.min**0.5

    bool known = false, scalar = false, np_scalar = false;
    std::vector<std::int64_t> dims;
    std::size_t m = 0;
    ArrayRead rd;

    // f's value at x into out (m doubles once known; the first value sets m)
    void call(double x, vec& out) {
        nb::object r = f(x);
        if (PyFloat_Check(r.ptr())) {
            if (!PyFloat_CheckExact(r.ptr())) np_scalar = true;
            settle(true, {});
            out.assign(1, PyFloat_AS_DOUBLE(r.ptr()));
            return;
        }
        if (!rd.open(r)) {
            if (!PyLong_CheckExact(r.ptr())) np_scalar = true;
            r = check(r);
            if (PyFloat_Check(r.ptr())) {
                settle(true, {});
                out.assign(1, PyFloat_AS_DOUBLE(r.ptr()));
                return;
            }
            rd.open(r);  // check gives a float or a float64 sd.ndarray of one dimension or more
        }
        settle(false, rd.shape);
        out.resize(m);
        rd.read(out.data());
    }

    void settle(bool sc, const std::vector<std::int64_t>& shape) {
        if (!known) {
            std::int64_t n = 1;
            for (auto s : shape) n *= s;
            if (!sc && n == 0)
                throw nb::value_error(
                    ("'f' must return a number or a non-empty array, not shape " + shape_text(shape, false)).c_str()
                );
            known = true;
            scalar = sc;
            dims = shape;
            m = sc ? 1 : static_cast<std::size_t>(n);
            return;
        }
        if (sc != scalar || (!sc && shape != dims))
            throw nb::value_error(("'f' must return the same shape at every point, not " + shape_text(dims, scalar) +
                                   " and then " + shape_text(shape, sc))
                                      .c_str());
    }

    // the (transformed) integrand at t: true with out filled, false for the transforms' 0 at
    // |t| < tmin (no call)
    bool operator()(double t, vec& out) {
        if (transform == kFinite) {
            call(t, out);
            return true;
        }
        if (transform == kSemi) {
            if (t < tmin) return false;
            const double x = start + (sgn * (1 - t)) / t;
            call(x, out);
            for (auto& v : out) v = (sgn * (v / t)) / t;
            return true;
        }
        if (std::abs(t) < tmin) return false;
        const double x = (1 - std::abs(t)) / t;
        call(x, out);
        for (auto& v : out) v = (v / t) / t;
        return true;
    }
};

// One rule on one interval: the integral (m values; empty: the value is 0 and f's shape is not
// known yet), its error estimate and rounding error.
struct RuleOut {
    vec ig;
    double err = 0.0, rnd = 0.0;
};

struct QuadVec {
    VecFn fn;
    int rule = kGk21, norm = kNorm2;
    PageBuffer scratch;
    vec work;  // n_nodes x m values
    std::vector<char> zero;  // the nodes the transform gave 0
    std::vector<vec> vals;  // one value per node
    std::int64_t nfev = 0;

    // the norm of m values (abs of a scalar f's value)
    double norm_of(const double* x) {
        if (fn.scalar) return std::abs(x[0]);
        if (norm == kNormMax) return maxabs(x, fn.m);
        double* b = scratch.get(fn.m);
        std::memcpy(b, x, fn.m * sizeof(double));
        return norm2_inplace(b, fn.m);
    }
    template <class G>
    double norm_by(G&& g) {  // the norm of the m values g(j), computed into the page-aligned buffer
        const std::size_t m = fn.m;
        if (fn.scalar) return std::abs(g(std::size_t{0}));
        double* b = scratch.get(m);
        for (std::size_t j = 0; j < m; ++j) b[j] = g(j);
        if (norm == kNormMax) return maxabs(b, m);
        return norm2_inplace(b, m);
    }

    // the values of n nodes in vals (zero[i]: 0, filled once m is known); false when none was
    // called and f's shape is still unknown
    template <class X>
    bool evaluate(std::size_t n, X&& node) {
        if (vals.size() < n) vals.resize(n);
        zero.assign(n, 0);
        for (std::size_t i = 0; i < n; ++i)
            if (!fn(node(i), vals[i])) zero[i] = 1;
        if (!fn.known) return false;
        for (std::size_t i = 0; i < n; ++i)
            if (zero[i]) vals[i].assign(fn.m, 0.0);
        return true;
    }

    // _quadrature_gk on [a, b] with nodes x, Gauss weights w (at the odd nodes), Kronrod weights v
    void gk(double a, double b, const double* x, const double* w, const double* v, std::size_t n, RuleOut& out) {
        const double c = 0.5 * (a + b), h = 0.5 * (b - a);
        nfev += static_cast<std::int64_t>(n);
        if (!evaluate(n, [&](std::size_t i) { return c + h * x[i]; })) {
            out.ig.clear();  // all 0 (scipy: the Python float 0.0)
            out.err = out.rnd = 0.0;
            return;
        }
        const std::size_t m = fn.m;
        vec sk(m, 0.0), ska(m, 0.0), sg(m, 0.0), sda(m, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double vv = v[i];
            const double* ff = vals[i].data();
            for (std::size_t j = 0; j < m; ++j) {
                sk[j] += vv * ff[j];
                ska[j] += vv * std::abs(ff[j]);
            }
        }
        for (std::size_t i = 0; i < n / 2; ++i) {
            const double* ff = vals[2 * i + 1].data();
            for (std::size_t j = 0; j < m; ++j) sg[j] += w[i] * ff[j];
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double* ff = vals[i].data();
            for (std::size_t j = 0; j < m; ++j) sda[j] += v[i] * std::abs(ff[j] - sk[j] / 2.0);
        }
        double err = norm_by([&](std::size_t j) { return (sk[j] - sg[j]) * h; });
        const double dabs = norm_by([&](std::size_t j) { return sda[j] * h; });
        if (dabs != 0 && err != 0) err = dabs * pymin(1.0, std::pow(200 * err / dabs, 1.5));
        const double s = (50 * kEps) * h;
        const double rnd = norm_by([&](std::size_t j) { return s * ska[j]; });
        if (rnd > kTiny) err = pymax(err, rnd);
        out.ig.resize(m);
        for (std::size_t j = 0; j < m; ++j) out.ig[j] = h * sk[j];
        out.err = err;
        out.rnd = rnd;
    }

    // _quadrature_trapezoid on [x1, x2]: f at x1, x2, then the midpoint
    void trapezoid(double x1, double x2, RuleOut& out, std::vector<std::pair<double, vec>>* cache) {
        const double x3 = 0.5 * (x1 + x2);
        const double xs[3] = {x1, x2, x3};
        if (!cache) nfev += 3;
        // scipy's lru_cache of the subdivision: a point seen in it is not called again
        auto node_value = [&](std::size_t i, vec& out_v) -> bool {
            if (cache)
                for (auto& [x, v] : *cache)
                    if (x == xs[i]) {
                        if (v.empty()) return false;
                        out_v = v;
                        return true;
                    }
            const bool ok = fn(xs[i], out_v);
            if (cache) {
                ++nfev;
                cache->emplace_back(xs[i], ok ? out_v : vec());
            }
            return ok;
        };
        if (vals.size() < 3) vals.resize(3);
        zero.assign(3, 0);
        for (std::size_t i = 0; i < 3; ++i)
            if (!node_value(i, vals[i])) zero[i] = 1;
        if (!fn.known) {
            out.ig.clear();
            out.err = out.rnd = 0.0;
            return;
        }
        const std::size_t m = fn.m;
        for (std::size_t i = 0; i < 3; ++i)
            if (zero[i]) vals[i].assign(m, 0.0);
        const double* f1 = vals[0].data();
        const double* f2 = vals[1].data();
        const double* f3 = vals[2].data();
        const double q = 0.25 * (x2 - x1);
        vec s2(m), s1(m);
        for (std::size_t j = 0; j < m; ++j) s2[j] = q * ((f1[j] + 2 * f3[j]) + f2[j]);
        const double n1 = norm_of(f1), n3 = norm_of(f3), n2 = norm_of(f2);
        out.rnd = ((0.25 * std::abs(x2 - x1)) * ((n1 + 2 * n3) + n2)) * 2e-16;
        const double hh = 0.5 * (x2 - x1);
        for (std::size_t j = 0; j < m; ++j) s1[j] = hh * (f1[j] + f2[j]);
        out.err = (1.0 / 3.0) * norm_by([&](std::size_t j) { return s1[j] - s2[j]; });
        out.ig = std::move(s2);
    }

    void apply(double a, double b, RuleOut& out, std::vector<std::pair<double, vec>>* cache = nullptr) {
        if (rule == kGk21) gk(a, b, kX21, kW10, kV21, 21, out);
        else if (rule == kGk15) gk(a, b, kX15, kW7, kV15, 15, out);
        else trapezoid(a, b, out, cache);
    }
};

// scipy's LRUDict of the intervals' integrals, keyed by the interval's ends (as Python floats:
// -0.0 is 0.0); an empty vector is a 0 integral of the shape still unknown
class Lru {
public:
    explicit Lru(std::int64_t max) : max_(max) {}
    void set(double x1, double x2, vec v) {
        const Key k{x1 + 0.0, x2 + 0.0};
        auto it = map_.find(k);
        if (it != map_.end()) {
            it->second->second = std::move(v);
            items_.splice(items_.end(), items_, it->second);
            return;
        }
        items_.emplace_back(k, std::move(v));
        map_.emplace(k, std::prev(items_.end()));
        if (static_cast<std::int64_t>(items_.size()) > max_) {
            map_.erase(items_.front().first);
            items_.pop_front();
        }
    }
    bool pop(double x1, double x2, vec& out) {
        auto it = map_.find(Key{x1 + 0.0, x2 + 0.0});
        if (it == map_.end()) return false;
        out = std::move(it->second->second);
        items_.erase(it->second);
        map_.erase(it);
        return true;
    }
    const vec* get(double x1, double x2) const {
        auto it = map_.find(Key{x1 + 0.0, x2 + 0.0});
        return it == map_.end() ? nullptr : &it->second->second;
    }

private:
    using Key = std::pair<double, double>;
    struct Hash {
        std::size_t operator()(const Key& k) const {
            std::uint64_t a, b;
            std::memcpy(&a, &k.first, 8);
            std::memcpy(&b, &k.second, 8);
            return std::hash<std::uint64_t>()(a * 0x9E3779B97F4A7C15ULL ^ b);
        }
    };
    std::int64_t max_;
    std::list<std::pair<Key, vec>> items_;
    std::unordered_map<Key, std::list<std::pair<Key, vec>>::iterator, Hash> map_;
};

// a heap item: scipy's tuple (-err, x1, x2), compared as Python compares tuples
struct Item {
    double ne, x1, x2;
};
inline bool item_lt(const Item& p, const Item& q) {
    if (!(p.ne == q.ne)) return p.ne < q.ne;
    if (!(p.x1 == q.x1)) return p.x1 < q.x1;
    if (!(p.x2 == q.x2)) return p.x2 < q.x2;
    return false;
}

// a + b elementwise into a (b empty: zeros)
void add_into(vec& a, const vec& b) {
    if (b.empty()) {
        for (auto& v : a) v = v + 0.0;
        return;
    }
    for (std::size_t j = 0; j < a.size(); ++j) a[j] += b[j];
}

nb::object quad_vec_solve(
    const char* fname,
    nb::callable f,
    nb::callable check,
    double a,
    double b,
    double epsabs,
    double epsrel,
    int norm,
    std::int64_t limit,
    std::vector<double> points,
    bool has_points,
    int quadrature
) {
    return call_named(fname, [&]() -> nb::object {
        QuadVec q;
        q.fn.f = f;
        q.fn.check = check;
        q.norm = norm;
        double final_sign = 1.0;
        // scipy's transforms of an infinite range (the recursive call on (0, 1) or (-1, 1))
        const bool fa = std::isfinite(a), fb = std::isfinite(b);
        if (!(fa && fb)) {
            if (quadrature == 0) quadrature = kGk15;
            if (fa || fb) {
                q.fn.transform = kSemi;
                q.fn.start = fa ? a : b;
                const double infty = fa ? b : a;
                q.fn.sgn = infty < 0 ? -1 : 1;
                for (auto& p : points) {
                    const double z = q.fn.sgn * (p - q.fn.start) + 1;
                    p = z == 0 ? kInf : 1 / z;
                }
                if (!fa) final_sign = -1.0;
                a = 0.0;
                b = 1.0;
            } else {
                q.fn.transform = kDouble;
                std::vector<double> pts{0.0};
                for (double p : points) pts.push_back((p < 0 ? -1.0 : 1.0) / (std::abs(p) + 1));
                points = std::move(pts);
                has_points = true;
                if (b < a) final_sign = -1.0;
                if (a != b) {
                    a = -1.0;
                    b = 1.0;
                } else {
                    a = b = 1.0;
                }
            }
        } else if (quadrature == 0) {
            quadrature = kGk21;
        }
        q.rule = quadrature;

        // the initial intervals
        std::vector<std::pair<double, double>> initial;
        if (!has_points) {
            initial.emplace_back(a, b);
        } else {
            std::stable_sort(points.begin(), points.end(), [](double x, double y) { return x < y; });
            double prev = a;
            for (double p : points) {
                if (!(a < p && p < b) || p == prev) continue;
                initial.emplace_back(prev, p);
                prev = p;
            }
            initial.emplace_back(prev, b);
        }

        vec gi;
        double ge = 0.0, re = 0.0;
        bool first = true, gi_zero = false;  // gi_zero: the first integrals were 0 of an unknown shape
        std::unique_ptr<Lru> cache;
        std::vector<Item> heap;
        RuleOut r;
        for (auto [x1, x2] : initial) {
            q.apply(x1, x2, r);
            if (first) {
                std::int64_t size;  // sys.getsizeof of the first integral (scipy's cache_count)
                if (r.ig.empty()) size = 24;
                else if (q.fn.scalar) size = q.fn.np_scalar ? 32 : 24;
                else
                    size =
                        96 + 16 * static_cast<std::int64_t>(q.fn.dims.size()) + 8 * static_cast<std::int64_t>(q.fn.m);
                cache = std::make_unique<Lru>(100000000 / size);
                gi = r.ig;
                gi_zero = r.ig.empty();
                ge = r.err;
                re = r.rnd;
                first = false;
            } else {
                if (gi.empty() && !r.ig.empty()) gi.assign(r.ig.size(), 0.0);  // 0.0 + the first array
                add_into(gi, r.ig);
                ge += r.err;
                re += r.rnd;
            }
            cache->set(x1, x2, r.ig);
            heap.push_back(Item{-r.err, x1, x2});
        }
        (void)gi_zero;
        heapify(heap, item_lt);
        if (!q.fn.known) throw nb::value_error("'f' was never called");  // the last initial interval always calls f
        if (gi.empty()) gi.assign(q.fn.m, 0.0);

        const std::size_t m = q.fn.m;
        auto norm_gi = [&]() { return q.norm_of(gi.data()); };
        int ier = 1;
        RuleOut r1, r2, r0;
        std::vector<std::pair<double, vec>> lru;
        struct Job {
            double old_err, a, b;
            bool have;
            vec old_int;
        };
        std::vector<Job> jobs;
        vec dint(m);
        while (!heap.empty() && static_cast<std::int64_t>(heap.size()) < limit) {
            double tol = pymax(epsabs, epsrel * norm_gi());
            jobs.clear();
            double err_sum = 0;
            for (int j = 0; j < 128; ++j) {
                if (heap.empty()) break;
                if (j > 0 && err_sum > ge - tol / 8) break;
                const Item it = heappop(heap, item_lt);
                Job jb{-it.ne, it.x1, it.x2, false, {}};
                jb.have = cache->pop(it.x1, it.x2, jb.old_int);
                err_sum += -it.ne;
                jobs.push_back(std::move(jb));
            }
            for (Job& jb : jobs) {  // scipy's _subdivide_interval, in order
                const double c = 0.5 * (jb.a + jb.b);
                lru.clear();
                auto* lc = q.rule == kTrapezoid ? &lru : nullptr;
                q.apply(jb.a, c, r1, lc);
                q.apply(c, jb.b, r2, lc);
                if (!jb.have) {
                    q.apply(jb.a, jb.b, r0, lc);
                    jb.old_int = r0.ig;
                }
                for (std::size_t k = 0; k < m; ++k) {
                    const double s1 = r1.ig.empty() ? 0.0 : r1.ig[k], s2 = r2.ig.empty() ? 0.0 : r2.ig[k],
                                 o = jb.old_int.empty() ? 0.0 : jb.old_int[k];
                    dint[k] = s1 + s2 - o;
                }
                const double derr = r1.err + r2.err - jb.old_err;
                const double dround = r1.rnd + r2.rnd;
                for (std::size_t k = 0; k < m; ++k) gi[k] += dint[k];
                ge += derr;
                re += dround;
                cache->set(jb.a, c, r1.ig);
                heappush(heap, Item{-r1.err, jb.a, c}, item_lt);
                cache->set(c, jb.b, r2.ig);
                heappush(heap, Item{-r2.err, c, jb.b}, item_lt);
            }
            if (heap.size() >= 2) {
                tol = pymax(epsabs, epsrel * norm_gi());
                if (ge < tol / 8) {
                    ier = 0;
                    break;
                }
                if (ge < re) {
                    ier = 2;
                    break;
                }
            }
            if (!(std::isfinite(ge) && std::isfinite(re))) {
                ier = 3;
                break;
            }
        }

        // the result: the integral (with the transform's sign), the error, the intervals from the
        // largest error down (heappop's order), their integrals (NaN when evicted from the cache)
        const double err = ge + re;
        nb::object integral;
        if (q.fn.scalar) {
            integral = nb::float_(gi[0] * final_sign);
        } else {
            NDArray out = new_array(DType::Float64, q.fn.dims);
            auto* o = reinterpret_cast<double*>(out.data);
            for (std::size_t k = 0; k < m; ++k) o[k] = gi[k] * final_sign;
            integral = readonly(std::move(out));
        }
        const auto order = pop_order(heap, item_lt);
        const auto n = static_cast<std::int64_t>(order.size());
        const std::int64_t s2[2] = {n, 2}, s1[1] = {n};
        std::vector<std::int64_t> sv{n};
        if (!q.fn.scalar) sv.insert(sv.end(), q.fn.dims.begin(), q.fn.dims.end());
        NDArray ends = new_array(DType::Float64, s2), vals = new_array(DType::Float64, sv),
                errs = new_array(DType::Float64, s1);
        auto* e = reinterpret_cast<double*>(ends.data);
        auto* v = reinterpret_cast<double*>(vals.data);
        auto* rr = reinterpret_cast<double*>(errs.data);
        for (std::size_t k = 0; k < order.size(); ++k) {
            const Item& it = heap[order[k]];
            e[2 * k] = it.x1;
            e[2 * k + 1] = it.x2;
            rr[k] = -it.ne;
            const vec* got = cache->get(it.x1, it.x2);
            for (std::size_t j = 0; j < m; ++j)
                v[k * m + j] = got ? (got->empty() ? 0.0 : (*got)[j] * final_sign) : kNaN;
        }
        return nb::make_tuple(
            integral, err, ier, q.nfev, readonly(std::move(ends)), readonly(std::move(vals)), readonly(std::move(errs))
        );
    });
}

// =====================================================================================
// cubature

// scipy's roots_legendre(10) and roots_legendre(7) (scipy.special 1.18.1, to the bit): the lower
// rules of its Gauss-Kronrod quadratures (up to 1 ulp off the correctly rounded nodes and 19 ulp
// off the weights; C99)
constexpr double kGL10x[10] = {
    -0x1.f2a3e062af2d8p-1, -0x1.bae995e9cb2f2p-1, -0x1.5bdb9228de198p-1, -0x1.bbcc009016adcp-2, -0x1.30e507891e279p-3,
    0x1.30e507891e279p-3,  0x1.bbcc009016adcp-2,  0x1.5bdb9228de198p-1,  0x1.bae995e9cb2f2p-1,  0x1.f2a3e062af2d8p-1,
};
constexpr double kGL10w[10] = {
    0x1.1115f8b62dc0cp-4, 0x1.32138c878efebp-3, 0x1.c0b059d00bc33p-3, 0x1.13baa7a559bffp-2, 0x1.2e9de7014d6f1p-2,
    0x1.2e9de7014d6f1p-2, 0x1.13baa7a559bffp-2, 0x1.c0b059d00bc33p-3, 0x1.32138c878efebp-3, 0x1.1115f8b62dc0cp-4,
};
constexpr double kGL7x[7] = {
    -0x1.e5f178e7c622ap-1, -0x1.7ba9f9be3a1d6p-1, -0x1.9f95df119fd62p-2, 0x0.0p+0,
    0x1.9f95df119fd62p-2,  0x1.7ba9f9be3a1d6p-1,  0x1.e5f178e7c622ap-1,
};
constexpr double kGL7w[7] = {
    0x1.092f69f826d55p-3, 0x1.1e6b1713d8642p-2, 0x1.86fe74ee32b40p-2, 0x1.abfd7e03c2fa4p-2,
    0x1.86fe74ee32b40p-2, 0x1.1e6b1713d8642p-2, 0x1.092f69f826d55p-3,
};

enum CubeRule { kCubeGk21 = 0, kCubeGk15 = 1, kGenzMalik = 2 };

// A fixed rule on [-1, 1]^d: nodes (n x d, C order) and weights
struct Fixed {
    std::size_t n = 0;
    vec nodes, weights;
};

// _cartesian_product of d copies of a 1-D rule (the first variable slowest) and the products of
// the weights (np.prod along the row: in order)
Fixed product_rule(const double* x, const double* w, std::size_t k, std::size_t d) {
    Fixed r;
    r.n = 1;
    for (std::size_t i = 0; i < d; ++i) r.n *= k;
    r.nodes.resize(r.n * d);
    r.weights.resize(r.n);
    std::vector<std::size_t> idx(d, 0);
    for (std::size_t p = 0; p < r.n; ++p) {
        double wp = 1.0;
        for (std::size_t i = 0; i < d; ++i) {
            r.nodes[p * d + i] = x[idx[i]];
            wp *= w[idx[i]];
        }
        r.weights[p] = wp;
        for (std::size_t i = d; i-- > 0;) {
            if (++idx[i] < k) break;
            idx[i] = 0;
        }
    }
    return r;
}

// _distinct_permutations of `items` (the next lexicographic permutation from the sorted ones)
void distinct_permutations(vec items, vec& nodes, std::size_t& n) {
    std::sort(items.begin(), items.end());
    do {
        nodes.insert(nodes.end(), items.begin(), items.end());
        ++n;
    } while (std::next_permutation(items.begin(), items.end()));
}

// GenzMalikCubature(d): the degree 7 rule (higher) and the degree 5 rule (lower), scipy's nodes in
// its order and its weights by Python's arithmetic (an int quotient correctly rounded)
void genz_malik(std::size_t d, Fixed& hi, Fixed& lo) {
    const double l2 = std::sqrt(9.0 / 70.0), l3 = std::sqrt(9.0 / 10.0), l4 = std::sqrt(9.0 / 10.0),
                 l5 = std::sqrt(9.0 / 19.0);
    auto fill = [&](Fixed& r, bool with5) {
        r.n = 0;
        r.nodes.clear();
        auto one = [&](std::initializer_list<double> head) {
            vec it(d, 0.0);
            std::size_t i = 0;
            for (double v : head) it[i++] = v;
            distinct_permutations(it, r.nodes, r.n);
        };
        r.nodes.insert(r.nodes.end(), d, 0.0);
        ++r.n;
        one({l2});
        one({-l2});
        one({l3});
        one({-l3});
        one({l4, l4});
        one({l4, -l4});
        one({-l4, -l4});
        if (with5) {
            const std::size_t m = std::size_t{1} << d;
            for (std::size_t p = 0; p < m; ++p) {  // itertools.product((l5, -l5), repeat=d)
                for (std::size_t i = 0; i < d; ++i) r.nodes.push_back(((p >> (d - 1 - i)) & 1) ? -l5 : l5);
                ++r.n;
            }
        }
    };
    fill(hi, true);
    fill(lo, false);
    const double two = std::ldexp(1.0, static_cast<int>(d));  // 2**d
    const auto nd = static_cast<double>(d);
    const double w1 = two * (12824.0 - 9120.0 * nd + 400.0 * nd * nd) / 19683.0;  // exact numerators
    const double w2 = two * 980.0 / 6561.0;
    const double w3 = two * (1820.0 - 400.0 * nd) / 19683.0;
    const double w4 = two * (200.0 / 19683.0);
    const double w5 = 6859.0 / 19683.0;
    hi.weights.clear();
    hi.weights.push_back(w1);
    hi.weights.insert(hi.weights.end(), 2 * d, w2);
    hi.weights.insert(hi.weights.end(), 2 * d, w3);
    hi.weights.insert(hi.weights.end(), 2 * (d - 1) * d, w4);
    hi.weights.insert(hi.weights.end(), std::size_t{1} << d, w5);
    const double u1 = two * (729.0 - 950.0 * nd + 50.0 * nd * nd) / 729.0;
    const double u2 = two * (245.0 / 486.0);
    const double u3 = two * (265.0 - 100.0 * nd) / 1458.0;
    const double u4 = two * (25.0 / 729.0);
    lo.weights.clear();
    lo.weights.push_back(u1);
    lo.weights.insert(lo.weights.end(), 2 * d, u2);
    lo.weights.insert(lo.weights.end(), 2 * d, u3);
    lo.weights.insert(lo.weights.end(), 2 * (d - 1) * d, u4);
}

struct Region {
    vec a, b, est, err;
    double key;  // np.max(np.abs(err))
};
inline bool region_lt(const Region& p, const Region& q) { return p.key > q.key; }

struct Cubature {
    nb::handle f, check;
    std::size_t d = 0;
    Fixed hi, both;  // the rule; both rules' nodes with the weights (w, -w_lower): the error rule
    // the transform of infinite limits
    bool transform = false;
    std::vector<int> kind;  // per variable: 0 finite, 1 [start, inf), 2 (-inf, b] (x = b - (1-t)/t), 3 (-inf, inf)
    vec start;
    std::size_t num_inf = 0;

    bool known = false;
    std::vector<std::int64_t> dims;  // f's value shape per point
    std::size_t m = 0;
    bool scalar = false;
    std::int64_t nfev = 0;
    bool fortran = false;  // genz-malik: the argument in Fortran order
    ArrayRead rd;
    vec fx, tmp, x_;

    // f at the n nodes t (n x d), values times the transform's Jacobian into fx (n x m); false:
    // the sum along the points runs as a running sum per value (numpy's reduction off memory)
    bool call(const double* t, std::size_t n) {
        const std::int64_t shape[2] = {static_cast<std::int64_t>(n), static_cast<std::int64_t>(d)};
        // scipy's layout of the argument: C order for the product rules, Fortran order for
        // genz-malik (its nodes are a transpose); f's values follow it in elementwise code, and
        // their layout decides numpy's order of the sum
        const std::int64_t cs[2] = {static_cast<std::int64_t>(d) * 8, 8}, fs[2] = {8, static_cast<std::int64_t>(n) * 8};
        NDArray arr = new_array_strided(DType::Float64, shape, fortran ? fs : cs, static_cast<std::int64_t>(n * d * 8));
        x_.resize(n * d);
        double* x = x_.data();
        std::memcpy(x, t, n * d * sizeof(double));
        vec jac;
        if (transform) {
            jac.resize(n);
            for (std::size_t p = 0; p < n; ++p) {
                double prod = 1.0;
                for (std::size_t i = 0; i < d; ++i) {
                    const double ti = t[p * d + i];
                    double& xi = x[p * d + i];
                    switch (kind[i]) {
                        case 1: xi = start[i] + (1 - ti) / ti; break;
                        case 2: xi = -(start[i] + (1 - ti) / ti); break;  // start = -b
                        case 3: xi = (1 - std::abs(ti)) / ti; break;
                        default: continue;
                    }
                    prod *= ti * ti;
                }
                jac[p] = 1 / prod;
            }
        }
        auto* xa = reinterpret_cast<double*>(arr.data);
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t i = 0; i < d; ++i) xa[fortran ? i * n + p : p * d + i] = x[p * d + i];
        nb::object r = f(wrap(std::move(arr)));
        nfev += static_cast<std::int64_t>(n);
        if (!(rd.open(r) && rd.shape[0] == static_cast<std::int64_t>(n))) {
            r = check(r, nb::make_tuple(static_cast<std::int64_t>(n), static_cast<std::int64_t>(d)));
            rd.open(r);  // a float64 sd.ndarray (n, ...)
        }
        std::vector<std::int64_t> s(rd.shape.begin() + 1, rd.shape.end());
        std::int64_t per = 1;
        for (auto v : s) per *= v;
        if (!known) {
            if (per == 0)
                throw nb::value_error(
                    ("'f' must return at least one value per point, not shape " + shape_text(rd.shape, false)).c_str()
                );
            known = true;
            dims = s;
            scalar = s.empty();
            m = static_cast<std::size_t>(per);
        } else if (s != dims) {
            std::vector<std::int64_t> want{static_cast<std::int64_t>(n)};
            want.insert(want.end(), dims.begin(), dims.end());
            throw nb::value_error(("'f' must return the same shape per point at every call: " +
                                   shape_text(want, false) + " here, not " + shape_text(rd.shape, false))
                                      .c_str());
        }
        fx.resize(n * m);
        for (std::size_t p = 0; p < n; ++p) rd.read(fx.data() + p * m, 1, static_cast<std::int64_t>(p));
        if (transform)
            for (std::size_t p = 0; p < n; ++p)
                for (std::size_t j = 0; j < m; ++j) fx[p * m + j] = fx[p * m + j] * jac[p];
        // the layout numpy's product and sum follow: along memory when the points' axis has the
        // smallest stride of the axes that are longer than 1 and not broadcast
        if (m == 1) return true;
        const std::int64_t s0 = std::abs(rd.strides[0]);
        bool inner = s0 != 0, other = false;
        for (std::size_t k = 1; k < rd.shape.size(); ++k) {
            if (rd.shape[k] <= 1 || rd.strides[k] == 0) continue;
            other = true;
            if (std::abs(rd.strides[k]) <= s0) inner = false;
        }
        return inner && other;
    }

    // _apply_fixed_rule of rule r on [a, b]: est (m values); abs_value: the error rule's |sum|
    void apply(const Fixed& r, const vec& a, const vec& b, vec& est, bool abs_value) {
        vec half(d);
        double prod = 1.0;
        for (std::size_t i = 0; i < d; ++i) {
            const double len = b[i] - a[i];
            half[i] = len * 0.5;
            prod *= len;
        }
        const double scale = prod / std::ldexp(1.0, static_cast<int>(d));
        tmp.resize(r.n * d);
        for (std::size_t p = 0; p < r.n; ++p)
            for (std::size_t i = 0; i < d; ++i) tmp[p * d + i] = (r.nodes[p * d + i] + 1) * half[i] + a[i];
        const bool along = call(tmp.data(), r.n);
        est.assign(m, 0.0);
        vec w(r.n);
        for (std::size_t p = 0; p < r.n; ++p) w[p] = r.weights[p] * scale;
        if (along) {
            vec col(r.n);
            for (std::size_t j = 0; j < m; ++j) {
                for (std::size_t p = 0; p < r.n; ++p) col[p] = w[p] * fx[p * m + j];
                est[j] = np_sum(col.data(), r.n);
            }
        } else {
            for (std::size_t p = 0; p < r.n; ++p)
                for (std::size_t j = 0; j < m; ++j) est[j] += w[p] * fx[p * m + j];
        }
        if (abs_value)
            for (auto& v : est) v = std::abs(v);
    }

    Region region(vec a, vec b) {
        Region g{std::move(a), std::move(b), {}, {}, 0.0};
        apply(hi, g.a, g.b, g.est, false);
        apply(both, g.a, g.b, g.err, true);
        g.key = maxabs(g.err.data(), m);
        return g;
    }
};

// _split_subregion: the 2^d halves (or the parts at `at`) in scipy's order
std::vector<std::pair<vec, vec>> split(const vec& a, const vec& b, const vec& at) {
    const std::size_t d = a.size(), n = std::size_t{1} << d;
    std::vector<std::pair<vec, vec>> out(n, {vec(d), vec(d)});
    for (std::size_t p = 0; p < n; ++p)
        for (std::size_t i = 0; i < d; ++i) {
            const bool hi = (p >> (d - 1 - i)) & 1;
            out[p].first[i] = hi ? at[i] : a[i];
            out[p].second[i] = hi ? b[i] : at[i];
        }
    return out;
}

nb::object cubature_solve(
    const char* fname,
    nb::callable f,
    nb::callable check,
    std::vector<double> a,
    std::vector<double> b,
    int rule,
    double rtol,
    double atol,
    std::int64_t max_subdivisions,
    std::vector<std::vector<double>> points
) {
    return call_named(fname, [&]() -> nb::object {
        Cubature c;
        c.f = f;
        c.check = check;
        const std::size_t d = a.size();
        c.d = d;
        Fixed lower;
        if (rule == kGenzMalik) {
            genz_malik(d, c.hi, lower);
            c.fortran = true;
        } else {
            const bool k21 = rule == kCubeGk21;
            c.hi = product_rule(k21 ? kX21 : kX15, k21 ? kV21 : kV15, k21 ? 21 : 15, d);
            lower = product_rule(k21 ? kGL10x : kGL7x, k21 ? kGL10w : kGL7w, k21 ? 10 : 7, d);
        }
        c.both.n = c.hi.n + lower.n;
        c.both.nodes = c.hi.nodes;
        c.both.nodes.insert(c.both.nodes.end(), lower.nodes.begin(), lower.nodes.end());
        c.both.weights = c.hi.weights;
        for (double w : lower.weights) c.both.weights.push_back(-w);

        // flipped limits (scipy's sign), then the transform of infinite limits
        int flips = 0;
        for (std::size_t i = 0; i < d; ++i)
            if (a[i] > b[i]) {
                ++flips;
                std::swap(a[i], b[i]);
            }
        const double sign = flips % 2 ? -1.0 : 1.0;
        c.kind.assign(d, 0);
        c.start.assign(d, 0.0);
        for (std::size_t i = 0; i < d; ++i) {
            if (a[i] == -kInf && b[i] == kInf) c.kind[i] = 3;
            else if (b[i] == kInf) c.kind[i] = 1, c.start[i] = a[i];
            else if (a[i] == -kInf) c.kind[i] = 2, c.start[i] = -b[i];
            if (c.kind[i]) ++c.num_inf;
        }
        if (c.num_inf) {
            c.transform = true;
            for (auto& p : points)  // _InfiniteLimitsTransform.inv, each variable as scipy means it
                for (std::size_t i = 0; i < d; ++i) {
                    const double x = p[i];
                    switch (c.kind[i]) {
                        case 1: p[i] = 1 / (x - c.start[i] + 1); break;
                        case 2: p[i] = 1 / ((-x) - c.start[i] + 1); break;
                        case 3: p[i] = x == 0 ? kInf : 1 / (x + (x > 0 ? 1.0 : -1.0)); break;
                        default: break;
                    }
                }
            for (std::size_t i = 0; i < d; ++i) {
                if (c.kind[i] == 3) a[i] = -1.0, b[i] = 1.0;
                else if (c.kind[i]) a[i] = 0.0, b[i] = 1.0;
            }
            points.emplace_back(d, 0.0);  // the origin, the transforms' problem point
        }

        // _split_region_at_points
        std::vector<std::pair<vec, vec>> initial{{a, b}};
        for (const auto& p : points) {
            if (std::any_of(p.begin(), p.end(), [](double v) { return std::isinf(v); })) continue;
            std::vector<std::pair<vec, vec>> next;
            for (auto& [ak, bk] : initial) {
                const bool at_a = std::equal(p.begin(), p.end(), ak.begin()),
                           at_b = std::equal(p.begin(), p.end(), bk.begin());
                bool inside = !at_a && !at_b;
                for (std::size_t i = 0; i < d && inside; ++i)
                    if (!(ak[i] <= p[i] && p[i] <= bk[i])) inside = false;
                if (!inside) {
                    next.emplace_back(std::move(ak), std::move(bk));
                    continue;
                }
                for (auto& [l, r] : split(ak, bk, p)) {
                    bool zero = false;
                    for (std::size_t i = 0; i < d; ++i) zero = zero || l[i] == r[i];
                    if (!zero) next.emplace_back(std::move(l), std::move(r));
                }
            }
            initial = std::move(next);
        }

        std::vector<Region> regions;
        vec est, err;  // empty: scipy's 0.0 before the first region
        for (auto& [ak, bk] : initial) {
            Region g = c.region(std::move(ak), std::move(bk));
            if (est.empty()) {
                est.assign(c.m, 0.0);
                err.assign(c.m, 0.0);
            }
            for (std::size_t j = 0; j < c.m; ++j) {
                est[j] += g.est[j];
                err[j] += g.err[j];
            }
            regions.push_back(std::move(g));
        }
        auto unconverged = [&]() {
            for (std::size_t j = 0; j < est.size(); ++j)
                if (err[j] > atol + rtol * std::abs(est[j])) return true;
            return false;
        };
        std::int64_t subdivisions = 0;
        bool success = true;
        while (unconverged()) {
            Region k = heappop(regions, region_lt);
            for (std::size_t j = 0; j < c.m; ++j) {
                est[j] -= k.est[j];
                err[j] -= k.err[j];
            }
            vec mid(d);
            for (std::size_t i = 0; i < d; ++i) mid[i] = (k.a[i] + k.b[i]) / 2;
            for (auto& [l, r] : split(k.a, k.b, mid)) {
                Region g = c.region(std::move(l), std::move(r));
                for (std::size_t j = 0; j < c.m; ++j) {
                    est[j] += g.est[j];
                    err[j] += g.err[j];
                }
                heappush(regions, std::move(g), region_lt);
            }
            if (++subdivisions >= max_subdivisions) {
                success = false;
                break;
            }
        }
        int status = 0;
        if (!success) {
            status = 1;
        } else {
            for (std::size_t j = 0; j < est.size(); ++j)
                if (!std::isfinite(est[j]) || !std::isfinite(err[j])) status = 3;
        }

        const std::size_t m = c.m;
        nb::object integral, error;
        if (c.scalar || !c.known) {
            integral = nb::float_(sign * (est.empty() ? 0.0 : est[0]));
            error = nb::float_(err.empty() ? 0.0 : err[0]);
        } else {
            NDArray ve = new_array(DType::Float64, c.dims), vr = new_array(DType::Float64, c.dims);
            auto* pe = reinterpret_cast<double*>(ve.data);
            auto* pr = reinterpret_cast<double*>(vr.data);
            for (std::size_t j = 0; j < m; ++j) {
                pe[j] = sign * est[j];
                pr[j] = err[j];
            }
            integral = readonly(std::move(ve));
            error = readonly(std::move(vr));
        }
        const auto order = pop_order(regions, region_lt);
        const auto n = static_cast<std::int64_t>(order.size());
        const std::int64_t sb[3] = {n, 2, static_cast<std::int64_t>(d)};
        std::vector<std::int64_t> sv{n};
        if (!c.scalar) sv.insert(sv.end(), c.dims.begin(), c.dims.end());
        NDArray bounds = new_array(DType::Float64, sb), vals = new_array(DType::Float64, sv),
                errs = new_array(DType::Float64, sv);
        auto* pb = reinterpret_cast<double*>(bounds.data);
        auto* pv = reinterpret_cast<double*>(vals.data);
        auto* pr = reinterpret_cast<double*>(errs.data);
        for (std::size_t k = 0; k < order.size(); ++k) {
            const Region& g = regions[order[k]];
            for (std::size_t i = 0; i < d; ++i) {
                pb[(2 * k) * d + i] = g.a[i];
                pb[(2 * k + 1) * d + i] = g.b[i];
            }
            for (std::size_t j = 0; j < m; ++j) {
                pv[k * m + j] = sign * g.est[j];
                pr[k * m + j] = g.err[j];
            }
        }
        return nb::make_tuple(
            integral, error, status, subdivisions, c.nfev, readonly(std::move(bounds)), readonly(std::move(vals)),
            readonly(std::move(errs))
        );
    });
}

}  // namespace

void bind_adaptive(nb::module_& m) {
    m.def(
        "quad_vec_solve", &quad_vec_solve, nb::arg("fn"), nb::arg("f"), nb::arg("check"), nb::arg("a"), nb::arg("b"),
        nb::arg("epsabs"), nb::arg("epsrel"), nb::arg("norm"), nb::arg("limit"), nb::arg("points"),
        nb::arg("has_points"), nb::arg("quadrature")
    );
    m.def(
        "cubature_solve", &cubature_solve, nb::arg("fn"), nb::arg("f"), nb::arg("check"), nb::arg("a"), nb::arg("b"),
        nb::arg("rule"), nb::arg("rtol"), nb::arg("atol"), nb::arg("max_subdivisions"), nb::arg("points")
    );
}

}  // namespace sd::integrate
