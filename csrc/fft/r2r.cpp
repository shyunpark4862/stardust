// The discrete cosine and sine transforms of sd.fft (docs/api/fft.md §4; docs/design/fft.md §2.6, §3.5;
// milestone FT4): dct, idct, dst, idst and the n-dimensional dctn, idctn, dstn, idstn.
//
//   _r2r(x, type, n, axis, norm, orthogonalize, inverse, sine, name)  -> dct, idct, dst, idst
//   _r2rn(x, type, s, axes, norm, orthogonalize, inverse, sine, name) -> dctn, idctn, dstn, idstn
//
// The arguments are checked here in the order of the signature (x, type, n or s, axis or axes, norm,
// orthogonalize) with the messages of API.md §2.1. idct and idst compute the inverse type (2 <-> 3; 1 and 4
// are their own inverses) with the inverse direction's factor; N' (the logical length the factor counts)
// is 2 (N - 1) for DCT-I, 2 (N + 1) for DST-I and 2N otherwise, the product over the axes for the
// n-dimensional ones. orthogonalize=None means norm == 'ortho'. A complex input transforms its real and
// imaginary parts separately (decision FT28): the driver reads a float64 view of each part and writes the
// same part of the complex128 result, so dct(x) is dct(x.real) + 1j dct(x.imag) bit for bit.
//
// An n-dimensional transform is one pass of the axis driver per axis, from the last of `axes` to the first
// (as the other transforms, decision FT25). The first pass reads the input, a pass that keeps its axis's
// length transforms the previous pass's array in place, and the last pass writes the result; the factor
// is applied once, by the last pass (decision FT48's rule), so dctn over one axis is dct bit for bit.
#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/stl/string.h>

#include "fft/args.hpp"
#include "fft/driver.hpp"
#include "fft/fft.hpp"
#include "fft/r2r.hpp"

namespace nb = nanobind;

namespace sd::fft {

namespace {

int parse_type(nb::handle o) {
    int64_t v = 0;
    switch (index_value(o, v)) {
        case IndexResult::NotInt: throw nb::type_error(("'type' must be an int, not " + py_type_name(o)).c_str());
        case IndexResult::Overflow: throw std::overflow_error("'type' is out of range");
        case IndexResult::Ok: break;
    }
    if (v < 1 || v > 4) throw nb::value_error(("'type' must be 1, 2, 3 or 4, not " + std::to_string(v)).c_str());
    return static_cast<int>(v);
}

bool parse_orthogonalize(nb::handle o, Norm norm) {
    if (o.is_none()) return norm == Norm::Ortho;
    if (o.ptr() == Py_True) return true;
    if (o.ptr() == Py_False) return false;
    throw nb::type_error(("'orthogonalize' must be True, False or None, not " + py_type_name(o)).c_str());
}

// The transform a call computes: idct and idst swap types 2 and 3.
R2r spec(int type, bool sine, bool ortho, bool inverse) {
    R2r r;
    r.type = inverse && (type == 2 || type == 3) ? 5 - type : type;
    r.sine = sine;
    r.ortho = ortho;
    return r;
}

// The logical length N' of a transform of n points (docs/design/fft.md §2.5).
int64_t logical_length(const R2r& r, int64_t n) {
    if (r.type == 1) return r.sine ? 2 * (n + 1) : 2 * (n - 1);
    return 2 * n;
}

void check_length(const R2r& r, int64_t n) {
    if (r.type == 1 && !r.sine && n < 2) {
        throw nb::value_error(("type=1 needs at least 2 points, not " + std::to_string(n)).c_str());
    }
}

// A float64 view of the real (part 0) or imaginary (part 1) parts of a complex128 array.
NDArray part_view(const NDArray& a, int part) {
    NDArray v = NDArray::meta_of(a);
    v.dtype = DType::Float64;
    v.data = a.data + 8 * part;
    v.update_flags();
    return v;
}

nb::object r2r(
    nb::handle xh,
    nb::handle th,
    nb::handle nh,
    nb::handle axh,
    nb::handle normh,
    nb::handle oh,
    bool inverse,
    bool sine,
    const std::string& name
) {
    return call_named(name.c_str(), [&]() -> nb::object {
        const NDArray& x = require_ndarray(xh, "x");
        if (x.dtype == DType::String) throw nb::type_error("'x' must be numeric, not a string array");
        const int type = parse_type(th);
        const Args a = parse_args(xh, nh, axh, normh);
        const R2r r = spec(type, sine, parse_orthogonalize(oh, a.norm), inverse);
        const int64_t n = length_or_axis(a);
        check_length(r, n);
        const bool cplx = x.dtype == DType::Complex128;
        NDArray out = new_result(x, a.axis, n, cplx ? DType::Complex128 : DType::Float64);
        const double s = norm_factor(a.norm, logical_length(r, n), inverse);
        if (!cplx) {
            r2r_axis(x, a.axis, n, r, s, out);
        } else {
            for (int part = 0; part < 2; ++part) {
                const NDArray xv = part_view(x, part);
                NDArray ov = part_view(out, part);
                r2r_axis(xv, a.axis, n, r, s, ov);
            }
        }
        return wrap(std::move(out));
    });
}

// The passes of one real array x (or a float64 view of a complex array's parts) over axes[k-1] .. axes[0],
// the last writing into out (float64, or a view of the complex result's parts).
void r2r_passes(
    const NDArray& x,
    const std::vector<int>& axes,
    const std::vector<int64_t>& len,
    const R2r& r,
    double scale,
    NDArray& out
) {
    const int k = static_cast<int>(axes.size());
    NDArray cur = x;
    bool own = false;
    for (int i = k - 1; i >= 0; --i) {
        const int ax = axes[static_cast<size_t>(i)];
        const int64_t n = len[static_cast<size_t>(i)];
        if (i == 0) {
            r2r_axis(cur, ax, n, r, scale, out);
        } else if (own && cur.shape[static_cast<size_t>(ax)] == n) {
            r2r_axis(cur, ax, n, r, 1.0, cur);
        } else {
            NDArray next = new_result(cur, ax, n, DType::Float64);
            r2r_axis(cur, ax, n, r, 1.0, next);
            cur = std::move(next);
            own = true;
        }
    }
}

nb::object r2rn(
    nb::handle xh,
    nb::handle th,
    nb::handle sh,
    nb::handle axh,
    nb::handle normh,
    nb::handle oh,
    bool inverse,
    bool sine,
    const std::string& name
) {
    return call_named(name.c_str(), [&]() -> nb::object {
        const NDArray& xin = require_ndarray(xh, "x");
        if (xin.dtype == DType::String) throw nb::type_error("'x' must be numeric, not a string array");
        const int type = parse_type(th);
        const NdArgs a = parse_nd(xh, sh, axh, normh);
        const NDArray& x = *a.x;
        const R2r r = spec(type, sine, parse_orthogonalize(oh, a.norm), inverse);
        const int k = static_cast<int>(a.axes.size());
        std::vector<int64_t> len(static_cast<size_t>(k)), logical(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i) {
            const int64_t n = nd_length(a, static_cast<size_t>(i));
            check_length(r, n);
            len[static_cast<size_t>(i)] = n;
            logical[static_cast<size_t>(i)] = logical_length(r, n);
        }
        const bool cplx = x.dtype == DType::Complex128;
        const std::vector<int64_t> shape = nd_shape(x, a.axes, len);
        NDArray out = new_array(cplx ? DType::Complex128 : DType::Float64, shape);
        if (any_zero(shape)) return wrap(std::move(out));  // no line to transform
        const double s = norm_factor_nd(a.norm, logical, inverse);
        if (!cplx) {
            r2r_passes(x, a.axes, len, r, s, out);
        } else {
            for (int part = 0; part < 2; ++part) {
                NDArray ov = part_view(out, part);
                r2r_passes(part_view(x, part), a.axes, len, r, s, ov);
            }
        }
        return wrap(std::move(out));
    });
}

}  // namespace

void bind_r2r(nb::module_& m) {
    m.def(
        "_r2r", &r2r, nb::arg("x").none(), nb::arg("type").none(), nb::arg("n").none(), nb::arg("axis").none(),
        nb::arg("norm").none(), nb::arg("orthogonalize").none(), nb::arg("inverse"), nb::arg("sine"), nb::arg("name")
    );  // fmt: skip
    m.def(
        "_r2rn", &r2rn, nb::arg("x").none(), nb::arg("type").none(), nb::arg("s").none(), nb::arg("axes").none(),
        nb::arg("norm").none(), nb::arg("orthogonalize").none(), nb::arg("inverse"), nb::arg("sine"), nb::arg("name")
    );  // fmt: skip
}

}  // namespace sd::fft
