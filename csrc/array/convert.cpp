#include "array/convert.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace sd {

void check_castable(DType from, DType to) {
    if (from == DType::Complex128 && (to == DType::Int64 || to == DType::Float64)) {
        std::string msg = std::string("cannot convert complex128 to ") + dtype_name(to) +
                          ": the imaginary part would be lost (take the real part explicitly)";
        throw nb::type_error(msg.c_str());
    }
}

namespace {

// numpy's abstract scalar types (numpy.bool, numpy.integer, numpy.floating,
// numpy.complexfloating), looked up once numpy has been imported -- a numpy
// scalar cannot exist before that, and stardust never imports numpy itself.
// numpy.float64 and numpy.complex128 subclass float and complex, so they are
// taken before this is reached.
struct NumpyScalarTypes {
    PyObject* types[4] = {};  // in DType order; owned for the process lifetime
    bool ready = false;
};

NumpyScalarTypes& numpy_scalar_types() {
    static NumpyScalarTypes t;
    if (t.ready) return t;
    PyObject* np = PyImport_GetModule(nb::str("numpy").ptr());  // only if already imported
    if (np == nullptr) {
        PyErr_Clear();
        return t;
    }
    const char* names[4] = {"bool", "integer", "floating", "complexfloating"};
    for (int k = 0; k < 4; ++k) {
        t.types[k] = PyObject_GetAttrString(np, names[k]);
        if (t.types[k] == nullptr) PyErr_Clear();
    }
    Py_DECREF(np);
    t.ready = true;
    return t;
}

// A numpy scalar of one of the four kinds -> its dtype (docs/api/array.md §6.2).
// numpy.timedelta64 is a numpy.integer without __index__: not an int.
std::optional<DType> numpy_scalar_dtype(PyObject* p) {
    if (std::strncmp(Py_TYPE(p)->tp_name, "numpy.", 6) != 0) return std::nullopt;
    // an array is no scalar: without this sd.asarray(ndarray) paid four isinstance calls (37 ns of 175)
    if (std::strcmp(Py_TYPE(p)->tp_name + 6, "ndarray") == 0) return std::nullopt;
    const NumpyScalarTypes& t = numpy_scalar_types();
    for (int k = 0; k < 4; ++k) {
        if (t.types[k] == nullptr) continue;
        const int is = PyObject_IsInstance(p, t.types[k]);
        if (is < 0) throw nb::python_error();
        if (is == 0) continue;
        if (k == 1 && !PyIndex_Check(p)) return std::nullopt;
        return static_cast<DType>(k);
    }
    return std::nullopt;
}

}  // namespace

std::optional<DType> scalar_dtype(nb::handle o) {
    PyObject* p = o.ptr();
    if (PyBool_Check(p)) return DType::Bool;  // before PyLong: bool is an int subclass
    if (PyLong_Check(p)) return DType::Int64;
    if (PyFloat_Check(p)) return DType::Float64;
    if (PyComplex_Check(p)) return DType::Complex128;
    return numpy_scalar_dtype(p);
}

namespace {

std::int64_t py_to_int64(PyObject* p) {
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(p, &overflow);
    if (overflow != 0) throw std::overflow_error("Python int too large to convert to int64");
    if (v == -1 && PyErr_Occurred()) throw nb::python_error();
    return v;
}

double py_to_double(PyObject* p) {
    double v = PyFloat_AsDouble(p);  // works for int too
    if (v == -1.0 && PyErr_Occurred()) {
        if (!PyErr_ExceptionMatches(PyExc_OverflowError)) throw nb::python_error();
        PyErr_Clear();  // an int beyond the double range: ours, so that the function's name is added
        throw std::overflow_error("Python int too large to convert to float64");
    }
    return v;
}

}  // namespace

template <typename T>
T scalar_from_python(nb::handle o, DType src) {
    PyObject* p = o.ptr();
    if constexpr (std::is_same_v<T, bool>) {
        int truth = PyObject_IsTrue(p);  // x != 0 for every numeric type (nan -> true)
        if (truth < 0) throw nb::python_error();
        return truth != 0;
    } else {
        switch (src) {
            case DType::Bool: {
                if (PyBool_Check(p)) return cast_value<T>(p == Py_True);
                const int truth = PyObject_IsTrue(p);  // numpy.bool
                if (truth < 0) throw nb::python_error();
                return cast_value<T>(truth != 0);
            }
            case DType::Int64: {
                if constexpr (std::is_same_v<T, std::int64_t>) return py_to_int64(p);
                else return cast_value<T>(py_to_double(p));
            }
            case DType::Float64: return cast_value<T>(py_to_double(p));
            case DType::Complex128: {
                if constexpr (is_complex_v<T>) {
                    Py_complex c = PyComplex_AsCComplex(p);
                    if (c.real == -1.0 && PyErr_Occurred()) throw nb::python_error();
                    return T(c.real, c.imag);
                } else {
                    check_castable(DType::Complex128, dtype_of<T>);  // always throws
                    __builtin_unreachable();
                }
            }
            case DType::String: break;  // scalar_dtype() never gives it: not a Python scalar's dtype
        }
        __builtin_unreachable();
    }
}

template bool scalar_from_python<bool>(nb::handle, DType);
template std::int64_t scalar_from_python<std::int64_t>(nb::handle, DType);
template double scalar_from_python<double>(nb::handle, DType);
template complex128 scalar_from_python<complex128>(nb::handle, DType);

nb::object to_python(bool v) { return nb::bool_(v); }
nb::object to_python(std::int64_t v) { return nb::steal(PyLong_FromLongLong(v)); }
nb::object to_python(double v) { return nb::steal(PyFloat_FromDouble(v)); }
nb::object to_python(complex128 v) { return nb::steal(PyComplex_FromDoubles(v.real(), v.imag())); }

nb::object load_python(DType d, const std::byte* p) {
    return visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        T v;
        std::memcpy(&v, p, sizeof(T));
        return to_python(v);
    });
}

}  // namespace sd
