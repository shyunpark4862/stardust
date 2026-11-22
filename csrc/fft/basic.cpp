// The transforms of sd.fft (docs/api/fft.md §2, §3; docs/design/fft.md §2.2-§2.5, §3.4, §3.7): fft and
// ifft (milestone FT1; real input by the real path since FT2, decision FT16), rfft, irfft, hfft and ihfft
// (FT2), the n-dimensional transforms (FT3), and the plan cache's tools for the tests (_plan_info,
// _cache_state, _cache_clear, _twiddles).
//
//   _c2c(x, n, axis, norm, inverse) -> fft, ifft: complex input by the complex transform of every line,
//                                      real input by the real transform's whole spectrum
//   _r2c(x, n, axis, norm, name)    -> rfft, ihfft: the half spectrum of real input
//   _c2r(x, n, axis, norm, name)    -> irfft, hfft: the real inverse of a Hermitian half spectrum
//   _c2cn(x, s, axes, norm, inverse, name) -> fft2, ifft2, fftn, ifftn
//   _r2cn(x, s, axes, norm, name)          -> rfft2, rfftn, ihfft2, ihfftn
//   _c2rn(x, s, axes, norm, name)          -> irfft2, irfftn, hfft2, hfftn
//
// The arguments are checked here, in the order of the signature (x, n or s, axis or axes, norm), with the
// messages of API.md §2.1; the Python functions (src/stardust/fft/_basic.py) only fix the call shape.
//
// An n-dimensional transform is one pass of the axis driver per axis, from the last of `axes` to the first
// (decision FT25; numpy's order): rfftn's first pass is the real one (rfft along the last of `axes`),
// irfftn's real pass comes last (the complex passes over the other axes, then irfft), so a non-Hermitian
// input loses the parts numpy and scipy drop. A pass writes a new complex128 array in C order when it
// changes its axis's length or reads the input, and otherwise transforms the previous pass's array in
// place (decision FT47). The normalisation is one factor of N = the product of the lengths, applied by
// the last pass (one rounding per part, as in one dimension; decision FT48).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "fft/args.hpp"
#include "fft/driver.hpp"
#include "fft/fft.hpp"
#include "fft/plan.hpp"
#include "fft/twiddle.hpp"

namespace nb = nanobind;

namespace sd::fft {

namespace {

nb::object c2c(nb::handle xh, nb::handle nh, nb::handle axh, nb::handle normh, bool inverse) {
    return call_named(inverse ? "ifft" : "fft", [&]() -> nb::object {
        const Args a = parse_args(xh, nh, axh, normh);
        const int64_t n = length_or_axis(a);
        NDArray out = new_result(*a.x, a.axis, n, DType::Complex128);
        const double s = norm_factor(a.norm, n, inverse);
        if (a.x->dtype == DType::Complex128) {
            c2c_axis(*a.x, a.axis, n, inverse, s, out);
        } else {  // real input: the real path's whole spectrum (decision FT16); ifft is its conjugate
            r2c_axis(*a.x, a.axis, n, true, inverse, s, out);
        }
        return wrap(std::move(out));
    });
}

// rfft (the forward direction) and ihfft (conj(rfft) with the inverse direction's factor; numpy's
// definition).
nb::object r2c(nb::handle xh, nb::handle nh, nb::handle axh, nb::handle normh, const std::string& name) {
    const bool h = name == "ihfft";
    return call_named(h ? "ihfft" : "rfft", [&]() -> nb::object {
        const Args a = parse_args(xh, nh, axh, normh);
        if (a.x->dtype == DType::Complex128) throw nb::type_error("'x' must be real, not complex128");
        const int64_t n = length_or_axis(a);
        NDArray out = new_result(*a.x, a.axis, n / 2 + 1, DType::Complex128);
        r2c_axis(*a.x, a.axis, n, false, h, norm_factor(a.norm, n, h), out);
        return wrap(std::move(out));
    });
}

// irfft (the inverse direction) and hfft (irfft of conj(x) with the forward direction's factor; numpy's
// definition). The default n is 2 (m - 1).
nb::object c2r(nb::handle xh, nb::handle nh, nb::handle axh, nb::handle normh, const std::string& name) {
    const bool h = name == "hfft";
    return call_named(h ? "hfft" : "irfft", [&]() -> nb::object {
        const Args a = parse_args(xh, nh, axh, normh);
        int64_t n = a.n;
        if (n < 0) {
            const int64_t m = a.x->shape[static_cast<size_t>(a.axis)];
            if (m == 0) throw nb::value_error("cannot transform an axis of length 0");
            if (m == 1) throw nb::value_error("the default 'n' = 2 (m - 1) is 0 for an axis of length m = 1; give 'n'");
            n = 2 * (m - 1);
        }
        NDArray out = new_result(*a.x, a.axis, n, DType::Float64);
        c2r_axis(*a.x, a.axis, n, h, norm_factor(a.norm, n, !h), out);
        return wrap(std::move(out));
    });
}

// ---- the n-dimensional transforms (milestone FT3)

// The complex passes over axes[i] for i = last, last - 1, .., 0 (back to front) on cur, which already
// holds the earlier passes' result (own: an array of this call, which a pass that keeps its axis's
// length transforms in place) or is the input (own = false). The last pass multiplies by `scale` (with
// conj_out, conjugates); the first reads its input conjugated with conj_in.
NDArray complex_passes(
    NDArray cur,
    bool own,
    const std::vector<int>& axes,
    const std::vector<int64_t>& len,
    int last,
    bool inverse,
    double scale,
    bool conj_in,
    bool conj_out
) {
    for (int i = last; i >= 0; --i) {
        const int ax = axes[static_cast<size_t>(i)];
        const int64_t n = len[static_cast<size_t>(i)];
        const double s = i == 0 ? scale : 1.0;
        const bool ci = conj_in && i == last, co = conj_out && i == 0;
        if (own && cur.shape[static_cast<size_t>(ax)] == n) {
            c2c_axis(cur, ax, n, inverse, s, cur, ci, co);
        } else {
            NDArray out = new_result(cur, ax, n, DType::Complex128);
            c2c_axis(cur, ax, n, inverse, s, out, ci, co);
            cur = std::move(out);
            own = true;
        }
    }
    return cur;
}


// fftn and ifftn (fft2, ifft2): the complex transform over axes; real input takes the real path in the
// first pass (the whole spectrum, decision FT16; ifftn: its conjugate).
nb::object c2cn(nb::handle xh, nb::handle sh, nb::handle axh, nb::handle normh, bool inverse, const std::string& name) {
    return call_named(name.c_str(), [&]() -> nb::object {
        const NdArgs a = parse_nd(xh, sh, axh, normh);
        const NDArray& x = *a.x;
        const int k = static_cast<int>(a.axes.size());
        std::vector<int64_t> len(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i) len[static_cast<size_t>(i)] = nd_length(a, static_cast<size_t>(i));
        const std::vector<int64_t> shape = nd_shape(x, a.axes, len);
        if (any_zero(shape)) return wrap(new_array(DType::Complex128, shape));  // no line to transform
        const double scale = norm_factor_nd(a.norm, len, inverse);
        if (x.dtype == DType::Complex128) {
            return wrap(complex_passes(x, false, a.axes, len, k - 1, inverse, scale, false, false));
        }
        const int ax = a.axes.back();
        NDArray first = new_result(x, ax, len.back(), DType::Complex128);
        r2c_axis(x, ax, len.back(), true, inverse, k == 1 ? scale : 1.0, first);
        return wrap(complex_passes(std::move(first), true, a.axes, len, k - 2, inverse, scale, false, false));
    });
}

// rfftn (forward) and ihfftn (conj(rfftn) with the inverse direction's factor; scipy's definition): rfft
// along the last of axes, then the forward complex passes over the others (rfft2, ihfft2 alike).
nb::object r2cn(nb::handle xh, nb::handle sh, nb::handle axh, nb::handle normh, const std::string& name) {
    const bool h = name == "ihfftn" || name == "ihfft2";
    return call_named(name.c_str(), [&]() -> nb::object {
        const NdArgs a = parse_nd(xh, sh, axh, normh);
        const NDArray& x = *a.x;
        if (x.dtype == DType::Complex128) throw nb::type_error("'x' must be real, not complex128");
        const int k = static_cast<int>(a.axes.size());
        std::vector<int64_t> len(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i) len[static_cast<size_t>(i)] = nd_length(a, static_cast<size_t>(i));
        const double scale = norm_factor_nd(a.norm, len, h);
        std::vector<int64_t> half = len;
        half.back() = len.back() / 2 + 1;
        const std::vector<int64_t> shape = nd_shape(x, a.axes, half);
        if (any_zero(shape)) return wrap(new_array(DType::Complex128, shape));
        const int ax = a.axes.back();
        NDArray first = new_result(x, ax, half.back(), DType::Complex128);
        r2c_axis(x, ax, len.back(), false, h && k == 1, k == 1 ? scale : 1.0, first);
        return wrap(complex_passes(std::move(first), true, a.axes, len, k - 2, false, scale, false, h));
    });
}

// irfftn (inverse) and hfftn (irfftn of conj(x) with the forward direction's factor; scipy's definition):
// the inverse complex passes over all but the last of axes on the input cut or zero-padded to
// n / 2 + 1 entries along the last, then irfft along the last (default n = 2 (m - 1)).
nb::object c2rn(nb::handle xh, nb::handle sh, nb::handle axh, nb::handle normh, const std::string& name) {
    const bool h = name == "hfftn" || name == "hfft2";
    return call_named(name.c_str(), [&]() -> nb::object {
        const NdArgs a = parse_nd(xh, sh, axh, normh);
        const NDArray& x = *a.x;
        const int k = static_cast<int>(a.axes.size());
        std::vector<int64_t> len(static_cast<size_t>(k));
        for (int i = 0; i + 1 < k; ++i) len[static_cast<size_t>(i)] = nd_length(a, static_cast<size_t>(i));
        const int ax = a.axes.back();
        const int64_t m = x.shape[static_cast<size_t>(ax)];
        if (a.s_given && a.s.back() != -1) {
            len.back() = a.s.back();
        } else if (a.s_given) {  // -1: the input's length (numpy and scipy)
            if (m == 0) throw nb::value_error("cannot transform an axis of length 0");
            len.back() = m;
        } else {
            if (m == 0) throw nb::value_error("cannot transform an axis of length 0");
            if (m == 1) {
                throw nb::value_error("the default s[-1] = 2 (m - 1) is 0 for an axis of length m = 1; give 's'");
            }
            len.back() = 2 * (m - 1);
        }
        const int64_t n = len.back();
        const double scale = norm_factor_nd(a.norm, len, !h);
        const std::vector<int64_t> shape = nd_shape(x, a.axes, len);
        if (any_zero(shape)) return wrap(new_array(DType::Float64, shape));
        NDArray out = new_array(DType::Float64, shape);
        if (k == 1) {
            c2r_axis(x, ax, n, h, scale, out);
            return wrap(std::move(out));
        }
        // the first complex pass reads x cut to (or zero-padded to) n / 2 + 1 entries along ax
        const int64_t half = n / 2 + 1, keep = std::min(m, half);
        NDArray xv = NDArray::meta_of(x);
        xv.shape[static_cast<size_t>(ax)] = keep;
        const int a0 = a.axes[static_cast<size_t>(k - 2)];
        std::vector<int64_t> s1 = shape_of(x);
        s1[static_cast<size_t>(ax)] = half;
        s1[static_cast<size_t>(a0)] = len[static_cast<size_t>(k - 2)];
        NDArray cur = new_array(DType::Complex128, s1, keep < half);
        NDArray ov = NDArray::meta_of(cur);
        ov.shape[static_cast<size_t>(ax)] = keep;
        c2c_axis(xv, a0, len[static_cast<size_t>(k - 2)], true, 1.0, ov, h, false);
        cur = complex_passes(std::move(cur), true, a.axes, len, k - 3, true, 1.0, false, false);
        c2r_axis(cur, ax, n, false, scale, out);
        return wrap(std::move(out));
    });
}

// The plan of a complex length n, or with real (n even) of the real length n: its path, size and cost.
nb::dict plan_info(int64_t n, bool real) {
    if (n < 1 || (real && n % 2 != 0)) throw nb::value_error("_plan_info(): n must be at least 1 (even if real)");
    const std::shared_ptr<const Plan> p = get_plan(real ? Kind::Real : Kind::Complex, n);
    nb::dict d;
    d["n"] = n;
    d["kind"] = real ? "real" : "complex";
    d["path"] = p->bluestein() ? "bluestein" : "mixed";
    d["bytes"] = p->bytes();
    d["cost"] = p->cost();
    if (p->bluestein()) {
        d["m"] = p->blue()->m();
        d["rho"] = p->blue()->rho();
        d["radices"] = p->blue()->inner().radices();
        d["n1"] = p->blue()->inner().n1();
    } else {
        d["radices"] = p->mixed()->radices();
        d["n1"] = p->mixed()->n1();  // the four-step's split n = n1 n2 (0: none; milestone FT6)
    }
    return d;
}

nb::dict cache_state_py() {
    const CacheState s = cache_state();
    nb::dict d;
    d["plans"] = s.plans;
    d["bytes"] = s.bytes;
    d["lengths"] = s.lengths;
    d["max_plans"] = kCachePlans;
    d["max_bytes"] = kCacheBytes;
    return d;
}

// The twiddles e^(-2 pi i k / n), k = 0 .. n - 1, as the plans compute them (twiddle.hpp): (n, 2) float64.
nb::object twiddles(int64_t n) {
    if (n < 1) throw nb::value_error("_twiddles(): n must be at least 1");
    NDArray out = new_array(DType::Float64, std::vector<int64_t>{n, 2});
    double* d = reinterpret_cast<double*>(out.data);
    for (int64_t k = 0; k < n; ++k) {
        const Twiddle w = twiddle(k, n);
        d[2 * k] = w.re, d[2 * k + 1] = w.im;
    }
    return wrap(std::move(out));
}

}  // namespace

void bind_basic(nb::module_& m) {
    m.def(
        "_c2c", &c2c, nb::arg("x").none(), nb::arg("n").none(), nb::arg("axis").none(), nb::arg("norm").none(),
        nb::arg("inverse")
    );  // fmt: skip
    m.def(
        "_r2c", &r2c, nb::arg("x").none(), nb::arg("n").none(), nb::arg("axis").none(), nb::arg("norm").none(),
        nb::arg("name")
    );  // fmt: skip
    m.def(
        "_c2r", &c2r, nb::arg("x").none(), nb::arg("n").none(), nb::arg("axis").none(), nb::arg("norm").none(),
        nb::arg("name")
    );  // fmt: skip
    m.def(
        "_c2cn", &c2cn, nb::arg("x").none(), nb::arg("s").none(), nb::arg("axes").none(), nb::arg("norm").none(),
        nb::arg("inverse"), nb::arg("name")
    );  // fmt: skip
    m.def(
        "_r2cn", &r2cn, nb::arg("x").none(), nb::arg("s").none(), nb::arg("axes").none(), nb::arg("norm").none(),
        nb::arg("name")
    );  // fmt: skip
    m.def(
        "_c2rn", &c2rn, nb::arg("x").none(), nb::arg("s").none(), nb::arg("axes").none(), nb::arg("norm").none(),
        nb::arg("name")
    );  // fmt: skip
    m.def("_plan_info", &plan_info, nb::arg("n"), nb::arg("real") = false);
    m.def("_cache_state", &cache_state_py);
    m.def("_cache_clear", [] { cache_clear(); });
    m.def("_twiddles", &twiddles, nb::arg("n"));
}

}  // namespace sd::fft
