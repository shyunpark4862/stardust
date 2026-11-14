// can_cast, result_type, isdtype, finfo, iinfo, and tensordot
// (docs/api/array.md §2.1, §9; T0b).

#include "array/ops/dtype_info.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/ops/matmul.hpp"
#include "array/ops/shape.hpp"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <string>
#include <vector>

namespace sd {

namespace {

constexpr const char* kDTypes = "sd.bool, sd.int64, sd.float64 or sd.complex128 (or their names)";

// A dtype argument (the object or its name, API.md §1), or, when
// `arrays` is set, an array (its dtype). Array dtypes only: sd.string is a
// TypeError. The message is the one every dtype argument uses.
DType dtype_arg(nb::handle o, const char* what, bool arrays = true) {
    if (arrays && is_ndarray(o)) return as_ndarray(o).dtype;
    const std::optional<DType> d = parse_any_dtype(o);
    if (!d || !is_array_dtype(*d))
        throw nb::type_error(("'" + std::string(what) + "' must be " + kDTypes + (arrays ? " or an sd.ndarray" : "") +
                              ", not " + describe_dtype_arg(o))
                                 .c_str());
    return *d;
}

// Implicit conversion goes up the order bool < int64 < float64 < complex128
// only (docs/design/array.md §2.5).
bool can_cast(nb::handle from, nb::handle to) { return dtype_arg(from, "from_") <= dtype_arg(to, "to"); }

// One dtype per kind, so a Python scalar promotes exactly as an array of its
// kind would: the result is the largest of the arguments.
nb::object result_type(nb::args args) {
    if (args.size() == 0) throw nb::value_error("at least one array, dtype or scalar is required");
    DType r = DType::Bool;
    for (nb::handle h : args) {
        if (is_ndarray(h) || parse_any_dtype(h)) {
            r = promote(r, dtype_arg(h, "arrays_and_dtypes"));
        } else if (const std::optional<DType> s = scalar_dtype(h)) {
            r = promote(r, *s);
        } else {
            throw nb::type_error(("'arrays_and_dtypes' must be sd.ndarray, dtypes or Python scalars, not " + py_type_name(h)).c_str());
        }
    }
    return nb::borrow(dtype_object(r));
}

// isdtype(dtype, kind): kind is a dtype, one of the Array API's kind names,
// or a tuple of them.
bool is_kind(DType d, nb::handle kind) {
    if (const std::optional<DType> k = parse_any_dtype(kind)) {  // a dtype (or its name): equality
        if (!is_array_dtype(*k)) return false;
        return d == *k;
    }
    if (!nb::isinstance<nb::str>(kind))
        throw nb::type_error(("'kind' must be a dtype, a str or a tuple of them, not " + py_type_name(kind)).c_str());
    const std::string s = nb::borrow<nb::str>(kind).c_str();
    if (s == "bool") return d == DType::Bool;
    if (s == "signed integer" || s == "integral") return d == DType::Int64;
    if (s == "unsigned integer") return false;
    if (s == "real floating") return d == DType::Float64;
    if (s == "complex floating") return d == DType::Complex128;
    if (s == "numeric") return d != DType::Bool;
    throw nb::value_error(("'kind' must be 'bool', 'signed integer', 'unsigned integer', 'integral', 'real floating', "
                           "'complex floating' or 'numeric', not '" + s + "'")
                              .c_str());
}

bool isdtype(nb::handle dtype_obj, nb::handle kind) {
    const DType d = dtype_arg(dtype_obj, "dtype", /*arrays=*/false);
    if (PyTuple_Check(kind.ptr())) {
        for (nb::handle k : kind)
            if (is_kind(d, k)) return true;
        return false;
    }
    return is_kind(d, kind);
}

nb::object results(const char* name) { return nb::module_::import_("stardust._results").attr(name); }

// finfo of float64; complex128 describes its float64 parts (Array API).
nb::object finfo(nb::handle type) {
    const DType d = dtype_arg(type, "type");
    if (d != DType::Float64 && d != DType::Complex128)
        throw nb::type_error((std::string("'type' must be float64 or complex128, not ") + dtype_name(d)).c_str());
    return results("FinfoResult")(64, DBL_EPSILON, DBL_MAX, -DBL_MAX, DBL_MIN, dtype_object(DType::Float64));
}

nb::object iinfo(nb::handle type) {
    const DType d = dtype_arg(type, "type");
    if (d != DType::Int64) throw nb::type_error((std::string("'type' must be int64, not ") + dtype_name(d)).c_str());
    return results("IinfoResult")(64, INT64_MAX, INT64_MIN, dtype_object(DType::Int64));
}

// ------------------------------------------------------------- tensordot

std::vector<int> axis_list(nb::handle o, int ndim) {
    std::vector<int> axes;
    if (!PyTuple_Check(o.ptr()) && !PyList_Check(o.ptr()))
        throw nb::type_error(("each entry of 'axes' must be a sequence of ints, not " + py_type_name(o)).c_str());
    for (nb::handle h : o) {
        const int d = normalize_axis(parse_int(h, "axes"), ndim);
        if (std::find(axes.begin(), axes.end(), d) != axes.end()) throw nb::value_error("'axes' has a repeated axis");
        axes.push_back(d);
    }
    return axes;
}

nb::object tensordot(nb::handle x1, nb::handle x2, nb::handle axes_obj) {
    const NDArray& a = require_ndarray(x1, "x1");
    const NDArray& b = require_ndarray(x2, "x2");
    std::vector<int> ca, cb;  // contracted axes of x1 and x2, paired in order
    std::int64_t n = 0;
    if (index_value(axes_obj, n) == IndexResult::Ok) {
        if (n < 0) throw nb::value_error("'axes' must be non-negative");
        if (n > a.ndim || n > b.ndim) throw nb::value_error("'axes' is larger than an array's number of dimensions");
        for (int i = 0; i < n; ++i) {
            ca.push_back(a.ndim - static_cast<int>(n) + i);
            cb.push_back(i);
        }
    } else if ((PyTuple_Check(axes_obj.ptr()) || PyList_Check(axes_obj.ptr())) && nb::len(axes_obj) == 2) {
        ca = axis_list(axes_obj[0], a.ndim);
        cb = axis_list(axes_obj[1], b.ndim);
        if (ca.size() != cb.size()) throw nb::value_error("the two axis lists of 'axes' must have the same length");
    } else {
        throw nb::type_error(("'axes' must be an int or a pair of int sequences, not " + py_type_name(axes_obj)).c_str());
    }
    for (size_t i = 0; i < ca.size(); ++i)
        if (a.shape[ca[i]] != b.shape[cb[i]])
            throw nb::value_error(("shape mismatch: axis " + std::to_string(ca[i]) + " of x1 has length " +
                                   std::to_string(a.shape[ca[i]]) + ", axis " + std::to_string(cb[i]) + " of x2 has " +
                                   std::to_string(b.shape[cb[i]]))
                                      .c_str());
    // x1 -> (free..., contracted...) as (M, K); x2 -> (contracted..., free...) as (K, N); one matmul.
    auto is_in = [](const std::vector<int>& v, int d) { return std::find(v.begin(), v.end(), d) != v.end(); };
    std::vector<int> pa, pb;
    std::vector<std::int64_t> out_shape;
    std::int64_t m = 1, k = 1, nn = 1;
    for (int d = 0; d < a.ndim; ++d)
        if (!is_in(ca, d)) {
            pa.push_back(d);
            out_shape.push_back(a.shape[d]);
            m *= a.shape[d];
        }
    for (int d : ca) {
        pa.push_back(d);
        k *= a.shape[d];
    }
    for (int d : cb) pb.push_back(d);
    for (int d = 0; d < b.ndim; ++d)
        if (!is_in(cb, d)) {
            pb.push_back(d);
            out_shape.push_back(b.shape[d]);
            nn *= b.shape[d];
        }
    auto permuted = [](nb::handle x, const std::vector<int>& order) {
        NDArray v = view_of(x);
        const NDArray& s = as_ndarray(x);
        for (size_t d = 0; d < order.size(); ++d) {
            v.shape[d] = s.shape[order[d]];
            v.strides[d] = s.strides[order[d]];
        }
        v.update_flags();
        return wrap(std::move(v));
    };
    const std::int64_t sa[] = {m, k}, sb[] = {k, nn};
    nb::object a2 = reshape(permuted(x1, pa), sa, std::nullopt);
    nb::object b2 = reshape(permuted(x2, pb), sb, std::nullopt);
    nb::object c = matmul(a2, b2);
    if (out_shape.empty()) {  // everything contracted: a scalar (no 0-d arrays)
        const NDArray& r = as_ndarray(c);
        return load_python(r.dtype, r.data);
    }
    return reshape(c, out_shape, std::nullopt);
}

}  // namespace

void bind_dtype_info(nb::module_& m) {
    using nb::arg;
    m.def("can_cast", named("can_cast", &can_cast), arg().none(), arg().none());
    m.def("result_type", named("result_type", &result_type));
    m.def("isdtype", named("isdtype", &isdtype), arg().none(), arg().none());
    m.def("finfo", named("finfo", &finfo), arg().none());
    m.def("iinfo", named("iinfo", &iinfo), arg().none());
    m.def("tensordot", named("tensordot", &tensordot), arg().none(), arg().none(), nb::kw_only(), arg("axes").none() = 2);
}

}  // namespace sd
