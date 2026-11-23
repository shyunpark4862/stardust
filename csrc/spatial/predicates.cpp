// Private bindings of the geometric predicates for the tests and the timings
// (docs/design/spatial_geometry.md §7.3; SPT8). Nothing here is public API: the triangulations call
// csrc/spatial/predicates.hpp directly.
//
//   _predicate(name, P, mode) -> int64 (m,)
//       name: 'orient2d' (P is (m, 3, 2)), 'orient3d' (m, 4, 3), 'incircle' (m, 4, 2), 'insphere' (m, 5, 3),
//             'incircle_sos', 'insphere_sos'.
//       mode: 'auto' (the *_any function: the range of each row's points picks the path), 'fast' (the plain
//             function: the rows must satisfy the range promise), 'filter' (stage A alone), 'expansion'
//             (stage B alone), 'wide' (stage W alone). A stage that does not decide gives 2. The *_sos
//             names take 'auto', 'fast' and 'wide'.
//   _predicate_stages(name, P) -> (filter, expansion, wide, outside)
//       how many rows each stage of the 'auto' path decides (outside: rows outside the range promise).
//   _det(name, P, mode) -> float64 (m,)
//       name: 'det2' ((m, 3, 2)), 'det3' ((m, 4, 3)). mode: 'auto', 'fast' (NaN when not certified), 'wide'.
//   _predicate_constants() -> dict
//   _range(P) -> (hi, lo) of a float64 array, or None when every element is 0.
//   _wide_eval(x) -> str     (X0 X1 - X2 X3) X4 + X5^3 in hex, X = x / 2^e, e the lowest set bit of the six.
//   _wide_round(x, shift) -> float     RN(x0 x1 x2 2^shift) through WideInt::to_double.
//   _expansion_eval(x) -> float64 list   the expansion of (x0 x1 - x2 x3) (x4 x5 + x6 x7) (its exact sum is the value).
//   _lex_order(P) -> (order, rep)        P (n, d) float64, d = 2 or 3.
//   _hilbert_keys(cells, bits) -> int64  cells (m, d) int64.
//   _brio_order(P) -> int64 (distinct,)
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "spatial/predicates.hpp"
#include "spatial/sfc.hpp"
#include "spatial/spatial.hpp"

#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sd::spatial {

namespace nb = nanobind;

namespace {

const NDArray& f64_array(nb::handle h, int ndim, const char* func) {
    if (!is_ndarray(h)) throw nb::type_error((std::string(func) + "(): expected an sd.ndarray").c_str());
    const NDArray& a = as_ndarray(h);
    if (a.ndim != ndim || a.dtype != DType::Float64 || !a.c_contiguous)
        throw nb::type_error((std::string(func) + "(): wrong layout").c_str());
    return a;
}

nb::object new_array_of(DType t, std::initializer_list<std::int64_t> shape) {
    const std::vector<std::int64_t> s(shape);
    return wrap(new_array(t, s));
}

enum class Mode { Auto, Fast, Filter, Expansion, Wide };

Mode mode_of(const std::string& s) {
    if (s == "auto") return Mode::Auto;
    if (s == "fast") return Mode::Fast;
    if (s == "filter") return Mode::Filter;
    if (s == "expansion") return Mode::Expansion;
    if (s == "wide") return Mode::Wide;
    throw nb::value_error("_predicate(): unknown mode");
}

struct Shape {
    int points, dim, degree;
};

Shape shape_of(const std::string& name) {
    if (name == "orient2d" || name == "det2") return {3, 2, 2};
    if (name == "orient3d" || name == "det3") return {4, 3, 3};
    if (name == "incircle" || name == "incircle_sos") return {4, 2, 4};
    if (name == "insphere" || name == "insphere_sos") return {5, 3, 5};
    throw nb::value_error("_predicate(): unknown name");
}

const NDArray& rows_of(nb::handle h, const Shape& s, const char* func) {
    const NDArray& a = f64_array(h, 3, func);
    if (a.shape[1] != s.points || a.shape[2] != s.dim)
        throw nb::value_error((std::string(func) + "(): wrong shape").c_str());
    return a;
}

int call(const std::string& name, Mode mode, const double* p) {
    if (name == "orient2d") {
        const double *a = p, *b = p + 2, *c = p + 4;
        switch (mode) {
            case Mode::Auto: return orient2d_any(a, b, c);
            case Mode::Fast: return orient2d(a, b, c);
            case Mode::Filter: return orient2d_filter(a, b, c);
            case Mode::Expansion: return orient2d_expansion(a, b, c);
            case Mode::Wide: return orient2d_wide(a, b, c);
        }
    } else if (name == "orient3d") {
        const double *a = p, *b = p + 3, *c = p + 6, *d = p + 9;
        switch (mode) {
            case Mode::Auto: return orient3d_any(a, b, c, d);
            case Mode::Fast: return orient3d(a, b, c, d);
            case Mode::Filter: return orient3d_filter(a, b, c, d);
            case Mode::Expansion: return orient3d_expansion(a, b, c, d);
            case Mode::Wide: return orient3d_wide(a, b, c, d);
        }
    } else if (name == "incircle") {
        const double *a = p, *b = p + 2, *c = p + 4, *d = p + 6;
        switch (mode) {
            case Mode::Auto: return incircle_any(a, b, c, d);
            case Mode::Fast: return incircle(a, b, c, d);
            case Mode::Filter: return incircle_filter(a, b, c, d);
            case Mode::Expansion: return incircle_expansion(a, b, c, d);
            case Mode::Wide: return incircle_wide(a, b, c, d);
        }
    } else if (name == "insphere") {
        const double *a = p, *b = p + 3, *c = p + 6, *d = p + 9, *e = p + 12;
        switch (mode) {
            case Mode::Auto: return insphere_any(a, b, c, d, e);
            case Mode::Fast: return insphere(a, b, c, d, e);
            case Mode::Filter: return insphere_filter(a, b, c, d, e);
            case Mode::Expansion: return insphere_expansion(a, b, c, d, e);
            case Mode::Wide: return insphere_wide(a, b, c, d, e);
        }
    } else if (name == "incircle_sos") {
        const double *a = p, *b = p + 2, *c = p + 4, *d = p + 6;
        switch (mode) {
            case Mode::Auto: return incircle_sos_any(a, b, c, d);
            case Mode::Fast: return incircle_sos(a, b, c, d);
            case Mode::Wide: return detail::incircle_tie(a, b, c, d, incircle_wide, orient2d_wide);
            default: break;
        }
    } else if (name == "insphere_sos") {
        const double *a = p, *b = p + 3, *c = p + 6, *d = p + 9, *e = p + 12;
        switch (mode) {
            case Mode::Auto: return insphere_sos_any(a, b, c, d, e);
            case Mode::Fast: return insphere_sos(a, b, c, d, e);
            case Mode::Wide: return detail::insphere_tie(a, b, c, d, e, insphere_wide, orient3d_wide);
            default: break;
        }
    }
    throw nb::value_error("_predicate(): this name does not take this mode");
}

nb::object predicate(const std::string& name, nb::handle ph, const std::string& mode_name) {
    const Shape s = shape_of(name);
    const NDArray& P = rows_of(ph, s, "_predicate");
    const Mode mode = mode_of(mode_name);
    const std::int64_t m = P.shape[0], step = s.points * s.dim;
    nb::object out = new_array_of(DType::Int64, {m});
    auto* o = reinterpret_cast<std::int64_t*>(as_ndarray(out).data);
    const auto* p = reinterpret_cast<const double*>(P.data);
    if (m > 0) (void)call(name, mode, p);  // a wrong name / mode pair throws before the loop
    if (name == "orient2d" && mode == Mode::Fast) {  // the hot loops, without the dispatch: for the timings
        for (std::int64_t i = 0; i < m; ++i) o[i] = orient2d(p + 6 * i, p + 6 * i + 2, p + 6 * i + 4);
    } else if (name == "orient3d" && mode == Mode::Fast) {
        for (std::int64_t i = 0; i < m; ++i)
            o[i] = orient3d(p + 12 * i, p + 12 * i + 3, p + 12 * i + 6, p + 12 * i + 9);
    } else if (name == "incircle" && mode == Mode::Fast) {
        for (std::int64_t i = 0; i < m; ++i) o[i] = incircle(p + 8 * i, p + 8 * i + 2, p + 8 * i + 4, p + 8 * i + 6);
    } else if (name == "insphere" && mode == Mode::Fast) {
        for (std::int64_t i = 0; i < m; ++i)
            o[i] = insphere(p + 15 * i, p + 15 * i + 3, p + 15 * i + 6, p + 15 * i + 9, p + 15 * i + 12);
    } else {
        for (std::int64_t i = 0; i < m; ++i) o[i] = call(name, mode, p + step * i);
    }
    return out;
}

nb::object predicate_stages(const std::string& name, nb::handle ph) {
    const Shape s = shape_of(name);
    if (name.ends_with("_sos") || name.starts_with("det")) throw nb::value_error("_predicate_stages(): unknown name");
    const NDArray& P = rows_of(ph, s, "_predicate_stages");
    const std::int64_t m = P.shape[0], step = s.points * s.dim;
    const auto* p = reinterpret_cast<const double*>(P.data);
    std::int64_t filter = 0, exp = 0, wide = 0, outside = 0;
    for (std::int64_t i = 0; i < m; ++i) {
        Range r;
        r.add(p + step * i, step);
        if (!r.safe(s.degree)) {
            ++outside;
            continue;
        }
        if (call(name, Mode::Filter, p + step * i) != 2) ++filter;
        else if (call(name, Mode::Expansion, p + step * i) != 2) ++exp;
        else ++wide;
    }
    return nb::make_tuple(filter, exp, wide, outside);
}

nb::object det(const std::string& name, nb::handle ph, const std::string& mode_name) {
    if (name != "det2" && name != "det3") throw nb::value_error("_det(): unknown name");
    const Shape s = shape_of(name);
    const NDArray& P = rows_of(ph, s, "_det");
    const Mode mode = mode_of(mode_name);
    if (mode == Mode::Filter || mode == Mode::Expansion) throw nb::value_error("_det(): unknown mode");
    const std::int64_t m = P.shape[0];
    nb::object out = new_array_of(DType::Float64, {m});
    auto* o = reinterpret_cast<double*>(as_ndarray(out).data);
    const auto* p = reinterpret_cast<const double*>(P.data);
    if (name == "det2") {
        for (std::int64_t i = 0; i < m; ++i) {
            const double *a = p + 6 * i, *b = a + 2, *c = a + 4;
            o[i] = mode == Mode::Auto   ? det2_any(a, b, c)
                   : mode == Mode::Fast ? det2_fast(a, b, c)
                                        : det2_wide(a, b, c);
        }
    } else {
        for (std::int64_t i = 0; i < m; ++i) {
            const double *a = p + 12 * i, *b = a + 3, *c = a + 6, *d = a + 9;
            o[i] = mode == Mode::Auto   ? det3_any(a, b, c, d)
                   : mode == Mode::Fast ? det3_fast(a, b, c, d)
                                        : det3_wide(a, b, c, d);
        }
    }
    return out;
}

nb::object predicate_constants() {
    nb::dict d;
    d["orient2d"] = kOrient2dFilter;
    d["orient3d"] = kOrient3dFilter;
    d["incircle"] = kIncircleFilter;
    d["insphere"] = kInsphereFilter;
    d["value_accept"] = kValueAccept;
    d["hilbert_bits"] = kHilbertBits;
    d["brio_first_round"] = kBrioFirstRound;
    d["wide_capacity"] = WideInt::kCap;
    d["expansion_passes"] = expansion::kPasses;
    return d;
}

nb::object range_of(nb::handle h) {
    if (!is_ndarray(h)) throw nb::type_error("_range(): expected an sd.ndarray");
    const NDArray& a = as_ndarray(h);
    if (a.dtype != DType::Float64 || !a.c_contiguous) throw nb::type_error("_range(): wrong layout");
    Range r;
    r.add(reinterpret_cast<const double*>(a.data), a.size());
    if (r.hi == INT_MIN) return nb::none();
    return nb::make_tuple(r.hi, r.lo);
}

const double* vector_of(nb::handle h, std::int64_t n, const char* func) {
    const NDArray& a = f64_array(h, 1, func);
    if (a.shape[0] != n) throw nb::value_error((std::string(func) + "(): wrong length").c_str());
    return reinterpret_cast<const double*>(a.data);
}

std::string wide_eval(nb::handle h) {
    const double* x = vector_of(h, 6, "_wide_eval");
    Range r;
    r.add(x, 6);
    if (r.hi == INT_MIN) return "0x0";
    WideInt v[6];
    for (int i = 0; i < 6; ++i) v[i] = WideInt::scaled(x[i], r.lo);
    const WideInt a = mul(sub(mul(v[0], v[1]), mul(v[2], v[3])), v[4]);
    return add(a, mul(mul(v[5], v[5]), v[5])).hex();
}

double wide_round(nb::handle h, std::int64_t shift) {
    const double* x = vector_of(h, 3, "_wide_round");
    Range r;
    r.add(x, 3);
    if (r.hi == INT_MIN) return 0.0;
    const WideInt p = mul(mul(WideInt::scaled(x[0], r.lo), WideInt::scaled(x[1], r.lo)), WideInt::scaled(x[2], r.lo));
    return p.to_double(3 * std::int64_t{r.lo} + shift);
}

nb::object expansion_eval(nb::handle h) {
    const double* x = vector_of(h, 8, "_expansion_eval");
    auto e = mul(sub(mul(x[0], x[1]), mul(x[2], x[3])), add(mul(x[4], x[5]), mul(x[6], x[7])));
    nb::list out;
    for (int i = 0; i < e.n; ++i) out.append(e.v[i]);
    const int s = sign(e);
    return nb::make_tuple(out, s);
}

const NDArray& points_of(nb::handle h, const char* func) {
    const NDArray& a = f64_array(h, 2, func);
    if (a.shape[1] != 2 && a.shape[1] != 3) throw nb::value_error((std::string(func) + "(): wrong shape").c_str());
    return a;
}

nb::object vector_out(const std::vector<std::int64_t>& v) {
    nb::object out = new_array_of(DType::Int64, {static_cast<std::int64_t>(v.size())});
    auto* o = reinterpret_cast<std::int64_t*>(as_ndarray(out).data);
    for (std::size_t i = 0; i < v.size(); ++i) o[i] = v[i];
    return out;
}

nb::object lex_order_py(nb::handle h) {
    const NDArray& P = points_of(h, "_lex_order");
    const int d = static_cast<int>(P.shape[1]);
    const auto* p = reinterpret_cast<const double*>(P.data);
    const std::vector<std::int64_t> order = lex_order(p, P.shape[0], d);
    std::vector<std::int64_t> rep;
    representatives(p, d, order, rep);
    return nb::make_tuple(vector_out(order), vector_out(rep));
}

nb::object brio_order_py(nb::handle h) {
    const NDArray& P = points_of(h, "_brio_order");
    const int d = static_cast<int>(P.shape[1]);
    const auto* p = reinterpret_cast<const double*>(P.data);
    const std::vector<std::int64_t> order = lex_order(p, P.shape[0], d);
    std::vector<std::int64_t> rep;
    const std::vector<std::int64_t> distinct = representatives(p, d, order, rep);
    return vector_out(brio_order(p, d, distinct));
}

nb::object hilbert_keys(nb::handle h, int bits) {
    if (!is_ndarray(h)) throw nb::type_error("_hilbert_keys(): expected an sd.ndarray");
    const NDArray& C = as_ndarray(h);
    if (C.ndim != 2 || C.dtype != DType::Int64 || !C.c_contiguous || (C.shape[1] != 2 && C.shape[1] != 3))
        throw nb::type_error("_hilbert_keys(): wrong layout");
    const int d = static_cast<int>(C.shape[1]);
    if (bits < 1 || bits > 21) throw nb::value_error("_hilbert_keys(): bits must be in [1, 21]");
    const auto* c = reinterpret_cast<const std::int64_t*>(C.data);
    std::vector<std::int64_t> keys(static_cast<std::size_t>(C.shape[0]));
    for (std::int64_t i = 0; i < C.shape[0]; ++i) {
        std::uint32_t cell[3] = {0, 0, 0};
        for (int k = 0; k < d; ++k) {
            const std::int64_t v = c[i * d + k];
            if (v < 0 || v >= (std::int64_t{1} << bits)) throw nb::value_error("_hilbert_keys(): cell out of range");
            cell[k] = static_cast<std::uint32_t>(v);
        }
        keys[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(hilbert_key(cell, d, bits));
    }
    return vector_out(keys);
}

}  // namespace

void bind_predicates(nb::module_& m) {
    m.def("_predicate", &predicate, nb::arg("name"), nb::arg("P"), nb::arg("mode") = "auto");
    m.def("_predicate_stages", &predicate_stages, nb::arg("name"), nb::arg("P"));
    m.def("_det", &det, nb::arg("name"), nb::arg("P"), nb::arg("mode") = "auto");
    m.def("_predicate_constants", &predicate_constants);
    m.def("_range", &range_of, nb::arg("P"));
    m.def("_wide_eval", &wide_eval, nb::arg("x"));
    m.def("_wide_round", &wide_round, nb::arg("x"), nb::arg("shift"));
    m.def("_expansion_eval", &expansion_eval, nb::arg("x"));
    m.def("_lex_order", &lex_order_py, nb::arg("P"));
    m.def("_hilbert_keys", &hilbert_keys, nb::arg("cells"), nb::arg("bits"));
    m.def("_brio_order", &brio_order_py, nb::arg("P"));
}

}  // namespace sd::spatial
