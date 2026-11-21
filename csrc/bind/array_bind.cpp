// The Python `ndarray` type: attributes, tolist, repr hooks, and the buffer
// protocol export that lets numpy (or memoryview) read our memory directly.

#include "bind/array_bind.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/indexing.hpp"
#include "array/ops/matmul.hpp"
#include "array/ops/dlpack.hpp"
#include "array/ops/reduction.hpp"
#include "array/ops/shape.hpp"
#include "array/ops/ufunc.hpp"

#include <string>
#include <type_traits>

namespace sd {

namespace {

static_assert(sizeof(Py_ssize_t) == sizeof(std::int64_t), "shape/strides are handed out as Py_ssize_t*");

// PEP 3118 format codes of our dtypes ('q' = int64, 'Zd' = complex double).
const char* const kFormats[kNumArrayDTypes] = {"?", "q", "d", "Zd"};

int getbuffer(PyObject* self, Py_buffer* view, int flags) {
    const NDArray& a = *nb::inst_ptr<NDArray>(self);
    auto fail = [&](const char* msg) {
        PyErr_SetString(PyExc_BufferError, msg);
        view->obj = nullptr;
        return -1;
    };
    if ((flags & PyBUF_WRITABLE) && !a.writeable) return fail("array is read-only");
    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS && !a.c_contiguous) return fail("array is not C-contiguous");
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS && !a.f_contiguous) return fail("array is not F-contiguous");
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS && !a.c_contiguous && !a.f_contiguous)
        return fail("array is not contiguous");
    const bool with_strides = (flags & PyBUF_STRIDES) == PyBUF_STRIDES;
    if (!with_strides && !a.c_contiguous) return fail("array is not C-contiguous; request strides");

    view->buf = a.data;
    view->obj = Py_NewRef(self);  // keeps the array (and so its Buffer) alive while exported
    view->len = a.nbytes();
    view->readonly = !a.writeable;
    view->itemsize = a.itemsize();
    view->format = (flags & PyBUF_FORMAT) ? const_cast<char*>(kFormats[static_cast<int>(a.dtype)]) : nullptr;
    view->ndim = a.ndim;
    view->shape = (flags & PyBUF_ND) == PyBUF_ND
                      ? reinterpret_cast<Py_ssize_t*>(const_cast<std::int64_t*>(a.shape.data()))
                      : nullptr;
    view->strides = with_strides ? reinterpret_cast<Py_ssize_t*>(const_cast<std::int64_t*>(a.strides.data())) : nullptr;
    view->suboffsets = nullptr;
    view->internal = nullptr;
    return 0;
}

// The plain attributes and len() as type slots, without nanobind's
// function dispatch in between (x.shape 34 ns against numpy's 18, len(x) 17
// against 7). They cannot fail except for memory.
const NDArray& self_array(PyObject* self) { return *nb::inst_ptr<NDArray>(nb::handle(self)); }

PyObject* get_shape(PyObject* self, void*) {
    const NDArray& a = self_array(self);
    PyObject* t = PyTuple_New(a.ndim);
    if (!t) return nullptr;
    for (int d = 0; d < a.ndim; ++d) {
        PyObject* v = PyLong_FromLongLong(a.shape[d]);
        if (!v) {
            Py_DECREF(t);
            return nullptr;
        }
        PyTuple_SET_ITEM(t, d, v);
    }
    return t;
}
PyObject* get_strides(PyObject* self, void*) {
    const NDArray& a = self_array(self);
    PyObject* t = PyTuple_New(a.ndim);
    if (!t) return nullptr;
    for (int d = 0; d < a.ndim; ++d) {
        PyObject* v = PyLong_FromLongLong(a.strides[d]);
        if (!v) {
            Py_DECREF(t);
            return nullptr;
        }
        PyTuple_SET_ITEM(t, d, v);
    }
    return t;
}
PyObject* get_itemsize(PyObject* self, void*) { return PyLong_FromLongLong(self_array(self).itemsize()); }
PyObject* get_nbytes(PyObject* self, void*) { return PyLong_FromLongLong(self_array(self).nbytes()); }
PyObject* get_base(PyObject* self, void*) {
    const NDArray& a = self_array(self);
    return Py_NewRef(a.base.is_valid() ? a.base.ptr() : Py_None);
}
PyObject* get_ndim(PyObject* self, void*) { return PyLong_FromLong(self_array(self).ndim); }
PyObject* get_size(PyObject* self, void*) { return PyLong_FromLongLong(self_array(self).size()); }
PyObject* get_dtype(PyObject* self, void*) { return Py_NewRef(dtype_object(self_array(self).dtype).ptr()); }
Py_ssize_t array_len(PyObject* self) { return static_cast<Py_ssize_t>(self_array(self).shape[0]); }  // never 0-d

// Read-only, with the message of the other attributes (Python properties).
int read_only(PyObject*, PyObject* value, void* name) {
    PyErr_Format(
        PyExc_AttributeError,
        value ? "property '%s' of 'ndarray' object has no setter" : "property '%s' of 'ndarray' object has no deleter",
        static_cast<const char*>(name)
    );
    return -1;
}

PyGetSetDef ndarray_getset[] = {
    {"shape", get_shape, read_only, nullptr, const_cast<char*>("shape")},
    {"ndim", get_ndim, read_only, nullptr, const_cast<char*>("ndim")},
    {"size", get_size, read_only, nullptr, const_cast<char*>("size")},
    {"dtype", get_dtype, read_only, nullptr, const_cast<char*>("dtype")},
    {"strides", get_strides, read_only, nullptr, const_cast<char*>("strides")},
    {"itemsize", get_itemsize, read_only, nullptr, const_cast<char*>("itemsize")},
    {"nbytes", get_nbytes, read_only, nullptr, const_cast<char*>("nbytes")},
    {"base", get_base, read_only, nullptr, const_cast<char*>("base")},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyType_Slot ndarray_slots[] = {
    {Py_bf_getbuffer, reinterpret_cast<void*>(getbuffer)},
    {Py_tp_getset, reinterpret_cast<void*>(ndarray_getset)},
    {Py_sq_length, reinterpret_cast<void*>(array_len)},
    {Py_mp_length, reinterpret_cast<void*>(array_len)},
    {0, nullptr},
};

nb::object tolist(const NDArray& a, int depth, const std::byte* p) {
    if (depth == a.ndim) return load_python(a.dtype, p);
    nb::list out;
    for (std::int64_t i = 0; i < a.shape[depth]; ++i) out.append(tolist(a, depth + 1, p + i * a.strides[depth]));
    return out;
}

// Like tolist, but along axes longer than 2*edge keeps only the first and
// last `edge` items with an Ellipsis between them. Lets repr show a huge
// array without materialising it as Python objects.
nb::object summary(const NDArray& a, int depth, const std::byte* p, std::int64_t edge) {
    if (depth == a.ndim) return load_python(a.dtype, p);
    nb::list out;
    const std::int64_t n = a.shape[depth];
    auto add = [&](std::int64_t i) { out.append(summary(a, depth + 1, p + i * a.strides[depth], edge)); };
    if (edge > 0 && n > 2 * edge) {
        for (std::int64_t i = 0; i < edge; ++i) add(i);
        out.append(nb::ellipsis());
        for (std::int64_t i = n - edge; i < n; ++i) add(i);
    } else {
        for (std::int64_t i = 0; i < n; ++i) add(i);
    }
    return out;
}

// What iter(x) returns: walks axis 0, yielding x[0], x[1], ...
struct ArrayIterator {
    nb::object array;
    std::int64_t next = 0, stop = 0;
};

// ---- operators: thin wrappers over the elementwise functions --------------
// An operand we do not take (a list, a numpy array, None) gives
// NotImplemented, so Python can try the other side and then raise TypeError.

nb::object not_implemented() { return nb::borrow(Py_NotImplemented); }

// == and != cannot give NotImplemented: when both sides do, Python falls back
// to identity and `x == [1, 2]` would quietly be False. So an operand we do
// not take is first offered to the other side's comparison (a type that knows
// sd.ndarray answers), and if it declines this is a TypeError, like
// `x < [1, 2]` and `x + [1, 2]` (docs/api/array.md §10.3).
template <typename Op>
nb::object op_equality(nb::handle self, nb::handle other) {
    constexpr bool eq = std::is_same_v<Op, ops::Equal>;
    if (is_operand(other)) return binary<Op>(self, other, nb::none(), eq ? "==" : "!=");
    const std::string refused = std::string("'") + (eq ? "==" : "!=") + "' not supported between instances of ";
    if (richcmpfunc cmp = Py_TYPE(other.ptr())->tp_richcompare) {
        nb::object r = nb::steal(cmp(other.ptr(), self.ptr(), eq ? Py_EQ : Py_NE));  // == and != reflect to themselves
        if (!r.is_valid()) {
            // The other side's refusal names the operands in its own order
            // ('Series' and 'ndarray'); ours below keeps the user's order.
            // Any other error is the other side's to tell.
            nb::python_error err;
            if (!err.matches(PyExc_TypeError) || std::string(nb::str(err.value()).c_str()).rfind(refused, 0) != 0)
                throw err;
        } else if (r.ptr() != Py_NotImplemented) {
            return r;
        }
    }
    throw nb::type_error((refused + "'" + py_type_name(self) + "' and '" + py_type_name(other) + "'").c_str());
}

// Operators keep Python's message form, with the operator's symbol and no
// function name (bind/errors.hpp): "unsupported operand dtypes for <: ...".
template <typename Op>
void def_compare(nb::class_<NDArray>& cls, const char* method, const char* symbol) {
    cls.def(
        method,
        [symbol](nb::handle self, nb::handle other) {
            return is_operand(other) ? binary<Op>(self, other, nb::none(), symbol) : not_implemented();
        },
        nb::arg().none()
    );
}

// x op= y is op(x, y, out=x): same rules as out= (upward conversion only,
// overlap-safe: docs/design/array.md §12.2), same object back.
template <typename Op>
void def_arith(
    nb::class_<NDArray>& cls,
    const char* fwd,
    const char* refl,
    const char* inplace,
    const char* symbol,
    const char* inplace_symbol
) {
    def_compare<Op>(cls, fwd, symbol);
    cls.def(
        refl,
        [symbol](nb::handle self, nb::handle other) {  // other <op> self
            return is_operand(other) ? binary<Op>(other, self, nb::none(), symbol) : not_implemented();
        },
        nb::arg().none()
    );
    cls.def(
        inplace,
        [inplace_symbol](nb::handle self, nb::handle other) {
            return is_operand(other) ? binary<Op>(self, other, self, inplace_symbol) : not_implemented();
        },
        nb::arg().none()
    );
}

template <typename Op>
void def_unary(nb::class_<NDArray>& cls, const char* method, const char* symbol) {
    cls.def(method, [symbol](nb::handle self) { return unary<Op>(self, nb::none(), symbol); });
}

nb::object format_call(const char* fn, nb::handle self) {
    return nb::module_::import_("stardust._format").attr(fn)(self);
}

}  // namespace

void bind_ndarray(nb::module_& m) {
    auto cls =
        nb::class_<NDArray>(m, "ndarray", nb::type_slots(ndarray_slots))
            .def_prop_ro("T", named(".T", &transpose_2d))
            .def_prop_ro("mT", named(".mT", &matrix_transpose))
            .def(
                "tolist", [](const NDArray& a) { return tolist(a, 0, a.data); }, nb::sig("def tolist(self, /)")
            )
            .def(
                "astype", named("astype", &astype), nb::sig("def astype(self, dtype, /, *, copy=True)"),
                nb::arg().none(), nb::kw_only(), nb::arg("copy").none() = true
            )
            .def(
                "copy",
                [](const NDArray& a) { return call_named("copy", [&] { return wrap(cast_copy(a, a.dtype)); }); },
                nb::sig("def copy(self, /)")
            )
            .def(
                "reshape", named("reshape", nb::overload_cast<nb::handle, nb::handle, nb::handle>(&reshape)),
                nb::sig("def reshape(self, shape, /, *, copy=None)"), nb::arg().none(), nb::kw_only(),
                nb::arg("copy").none() = nb::none()
            )
            // .none(): x[None] is newaxis, and nanobind rejects None by default.
            // Reductions as methods (docs/design/array.md §12.2): the same functions, x as self.
            // Each nb::sig is the module function's with self for x (no type annotations).
            .def(
                "sum", named("sum", &sum), nb::sig("def sum(self, /, *, axis=None, dtype=None, keepdims=False)"),
                nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("dtype").none() = nb::none(),
                nb::arg("keepdims").none() = false
            )
            .def(
                "prod", named("prod", &prod), nb::sig("def prod(self, /, *, axis=None, dtype=None, keepdims=False)"),
                nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("dtype").none() = nb::none(),
                nb::arg("keepdims").none() = false
            )
            .def(
                "min", named("min", &min), nb::sig("def min(self, /, *, axis=None, keepdims=False)"), nb::kw_only(),
                nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false
            )
            .def(
                "max", named("max", &max), nb::sig("def max(self, /, *, axis=None, keepdims=False)"), nb::kw_only(),
                nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false
            )
            .def(
                "mean", named("mean", &mean), nb::sig("def mean(self, /, *, axis=None, keepdims=False)"), nb::kw_only(),
                nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false
            )
            .def(
                "var", named("var", &var), nb::sig("def var(self, /, *, axis=None, correction=0, keepdims=False)"),
                nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("correction").none() = 0,
                nb::arg("keepdims").none() = false
            )
            .def(
                "std", named("std", &stddev), nb::sig("def std(self, /, *, axis=None, correction=0, keepdims=False)"),
                nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("correction").none() = 0,
                nb::arg("keepdims").none() = false
            )
            .def(
                "argmin", named("argmin", &argmin), nb::sig("def argmin(self, /, *, axis=None, keepdims=False)"),
                nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false
            )
            .def(
                "argmax", named("argmax", &argmax), nb::sig("def argmax(self, /, *, axis=None, keepdims=False)"),
                nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false
            )
            .def(
                "any", named("any", &any), nb::sig("def any(self, /, *, axis=None, keepdims=False)"), nb::kw_only(),
                nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false
            )
            .def(
                "all", named("all", &all), nb::sig("def all(self, /, *, axis=None, keepdims=False)"), nb::kw_only(),
                nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false
            )
            // There is no constructor (docs/api/array.md §10). Without this
            // __init__, nanobind says "stardust._core.ndarray: no constructor defined!".
            .def(
                "__init__",
                [](NDArray*, nb::args, nb::kwargs) {
                    throw nb::type_error(
                        "ndarray(): there is no constructor; use asarray() or another creation function"
                    );
                }
            )
            .def("__getitem__", &getitem, nb::arg().none())
            .def("__setitem__", &setitem, nb::arg().none(), nb::arg().none())
            // Python's message for a container without deletion, as for a tuple
            // (docs/api/array.md §5.3). Without it, `del x[0]` said "AttributeError: __delitem__".
            .def(
                "__delitem__",
                [](const NDArray&, nb::handle) {
                    throw nb::type_error("'ndarray' object doesn't support item deletion");
                },
                nb::arg().none()
            )
            .def(
                "__iter__",
                [](nb::handle self) { return ArrayIterator{nb::borrow(self), 0, as_ndarray(self).shape[0]}; }
            )
            // Without this, `v in x` would quietly fall back to iterating x.
            .def(
                "__contains__",
                [](const NDArray&, nb::handle) -> bool {
                    throw nb::type_error("`in` is not supported for arrays; compare with == and use any()");
                }
            )
            .def("_summary", [](const NDArray& a, std::int64_t edge) { return summary(a, 0, a.data, edge); })
            .def(
                "_flags",
                [](const NDArray& a) {
                    nb::dict d;
                    d["c_contiguous"] = a.c_contiguous;
                    d["f_contiguous"] = a.f_contiguous;
                    d["writeable"] = a.writeable;
                    return d;
                }
            )
            .def("_set_readonly", [](NDArray& a) { a.writeable = false; })
            .def("__repr__", [](nb::handle self) { return format_call("array_repr", self); })
            .def("__str__", [](nb::handle self) { return format_call("array_str", self); })
            // An array is never a scalar, whatever its size
            // (docs/design/array.md §12.2). All four hooks must exist: without
            // __int__/__float__, int()/float() fall back to the buffer protocol
            // and try to parse the raw bytes as text.
            .def(
                "__bool__",
                [](const NDArray&) -> bool {
                    throw nb::type_error("the truth value of an array is ambiguous; use any() or all()");
                }
            )
            .def("__int__", [](const NDArray&) -> int { throw nb::type_error("an array cannot be converted to int"); })
            .def(
                "__float__",
                [](const NDArray&) -> double { throw nb::type_error("an array cannot be converted to float"); }
            )
            .def(
                "__complex__",
                [](const NDArray&) -> double { throw nb::type_error("an array cannot be converted to complex"); }
            )
            .def("__index__", [](const NDArray&) -> int {
                throw nb::type_error("an array cannot be used as an index");
            });
    using namespace ops;
    def_arith<Add>(cls, "__add__", "__radd__", "__iadd__", "+", "+=");
    def_arith<Subtract>(cls, "__sub__", "__rsub__", "__isub__", "-", "-=");
    def_arith<Multiply>(cls, "__mul__", "__rmul__", "__imul__", "*", "*=");
    def_arith<Divide>(cls, "__truediv__", "__rtruediv__", "__itruediv__", "/", "/=");
    def_arith<FloorDivide>(cls, "__floordiv__", "__rfloordiv__", "__ifloordiv__", "//", "//=");
    def_arith<Remainder>(cls, "__mod__", "__rmod__", "__imod__", "%", "%=");
    def_arith<Pow>(cls, "__pow__", "__rpow__", "__ipow__", "**", "**=");
    def_arith<BitwiseAnd>(cls, "__and__", "__rand__", "__iand__", "&", "&=");
    def_arith<BitwiseOr>(cls, "__or__", "__ror__", "__ior__", "|", "|=");
    def_arith<BitwiseXor>(cls, "__xor__", "__rxor__", "__ixor__", "^", "^=");
    def_arith<LeftShift>(cls, "__lshift__", "__rlshift__", "__ilshift__", "<<", "<<=");
    def_arith<RightShift>(cls, "__rshift__", "__rrshift__", "__irshift__", ">>", ">>=");
    // Python reflects comparisons itself (x > y becomes y < x).
    cls.def("__eq__", &op_equality<Equal>, nb::arg().none());
    cls.def("__ne__", &op_equality<NotEqual>, nb::arg().none());
    def_compare<Less>(cls, "__lt__", "<");
    def_compare<LessEqual>(cls, "__le__", "<=");
    def_compare<Greater>(cls, "__gt__", ">");
    def_compare<GreaterEqual>(cls, "__ge__", ">=");
    // @ takes arrays only (no Python scalars). x @= y needs the result to
    // have x's shape and converts upward only, like every in-place operator
    // (docs/design/array.md §9.1).
    cls.def(
        "__matmul__",
        [](nb::handle self, nb::handle other) { return is_ndarray(other) ? matmul(self, other) : not_implemented(); },
        nb::arg().none()
    );
    cls.def(
        "__rmatmul__",
        [](nb::handle self, nb::handle other) { return is_ndarray(other) ? matmul(other, self) : not_implemented(); },
        nb::arg().none()
    );
    cls.def(
        "__imatmul__",
        [](nb::handle self, nb::handle other) -> nb::object {
            if (!is_ndarray(other)) return not_implemented();
            nb::object r = matmul(self, other);
            NDArray& x = as_ndarray(self);
            if (!is_ndarray(r) || !std::ranges::equal(as_ndarray(r).shape_span(), x.shape_span()))
                throw nb::value_error("x @= y needs a result with the shape of x");
            assign_checked(x, as_ndarray(r));
            return nb::borrow(self);
        },
        nb::arg().none()
    );
    def_unary<Negative>(cls, "__neg__", "unary -");
    def_unary<Positive>(cls, "__pos__", "unary +");
    def_unary<Abs>(cls, "__abs__", "abs()");
    def_unary<BitwiseInvert>(cls, "__invert__", "unary ~");

    // Arrays are mutable containers: unhashable, like lists. (Defining
    // __eq__ would also do this; being explicit documents it.)
    cls.attr("__hash__") = nb::none();
    use_short_type_name(cls);
    // numpy, seeing `np_array + sd_array`, would convert our array through
    // the buffer protocol and return a numpy result. None tells numpy's
    // ufuncs to step aside, so mixing the two raises TypeError
    // (docs/design/array.md §12.2).
    cls.attr("__array_ufunc__") = nb::none();

    // nanobind's properties have no name, so assigning one said "property ''
    // of 'ndarray' object has no setter". __set_name__ gives each its name.
    for (const char* name : {"T", "mT"}) cls.attr(name).attr("__set_name__")(cls, name);

    bind_dlpack(m, cls);  // from_dlpack, __dlpack__, __dlpack_device__

    auto it_cls = nb::class_<ArrayIterator>(m, "_ArrayIterator")
                      .def("__iter__", [](nb::handle self) { return nb::borrow(self); })
                      .def("__next__", [](ArrayIterator& it) {
                          if (it.next >= it.stop) throw nb::stop_iteration();
                          return item_along_axis0(it.array, it.next++);
                      });
    use_short_type_name(it_cls);
}

}  // namespace sd
