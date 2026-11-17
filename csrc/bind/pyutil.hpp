// Glue between Python objects and the C++ core: dtype objects, argument
// parsing shared by many functions, and wrapping NDArray as a Python object.
// The parsers' messages name the argument ('axis') but not the function: the
// binding adds that (bind/errors.hpp).
#pragma once

#include "array/ndarray.hpp"
#include "bind/errors.hpp"  // py_type_name

#include <nanobind/nanobind.h>

#include <initializer_list>
#include <optional>
#include <array>
#include <span>
#include <string>
#include <vector>

namespace sd {

namespace nb = nanobind;

// Python-side dtype object. Exactly five instances exist (sd.bool, ...,
// sd.string), so dtypes compare by identity.
struct PyDType {
    DType value;
};

void register_dtypes(nb::module_& m);
nb::handle dtype_object(DType d);

// A dtype object or its name -> its DType (any of the five); None and
// anything else -> nullopt. Python types like `float` are not dtypes (API.md §1).
std::optional<DType> parse_any_dtype(nb::handle o);

// What a rejected dtype argument was, for messages: a dtype object by its
// name (sd.string), a string or a type as itself ('float32', <class 'float'>),
// anything else by its type's name.
std::string describe_dtype_arg(nb::handle o);

// Gives a bound class the short type name Python's own messages use
// ("unsupported operand type(s) for +: 'ndarray' and 'list'", "unhashable
// type: 'ndarray'", pickle, missing attributes). nanobind names the type
// "stardust._core.ndarray" in tp_name; classes written in Python have just
// "ndarray" there (API.md §2.1). __module__ and repr(type) are unchanged.
void use_short_type_name(nb::handle type);

// A bool option: True or False only. 1, None and numpy.bool_ are TypeErrors
// ("'keepdims' must be True or False, not int", API.md §2.1).
bool parse_bool(nb::handle o, const char* what);

// An int argument by operator.index semantics (API.md §2): a Python int or an
// object with __index__ (numpy.int64, ...); bool, numpy.bool_ and sd.ndarray
// are not ints. NotInt and Overflow (outside int64) let the caller name what
// it accepts.
enum class IndexResult { Ok, NotInt, Overflow };
IndexResult index_value(nb::handle o, std::int64_t& out);

// A real-number argument (correction=, ord=, rtol=): a Python float (numpy.float64
// is one), or an int by the rule above. nullopt for anything else; an int beyond the
// double range is OverflowError "'what' is out of range" (what: the argument's name).
std::optional<double> real_value(nb::handle o, const char* what = nullptr);

// An array `dtype=` argument: None -> nullopt; otherwise one of the four
// array dtypes, as an object or a name (API.md §1). TypeError otherwise,
// sd.string included.
std::optional<DType> parse_dtype(nb::handle o);

// A string option: one of `choices`. The message lists them the same way
// for a wrong type (TypeError, "not NoneType") and a wrong value (ValueError,
// "not 'x'"): "'side' must be 'left' or 'right', not ..." (API.md §2.1).
std::string parse_choice(nb::handle o, const char* what, std::span<const char* const> choices);
inline std::string parse_choice(nb::handle o, const char* what, std::initializer_list<const char*> choices) {
    return parse_choice(o, what, std::span<const char* const>(choices.begin(), choices.size()));
}

// `copy=` argument: None / True / False.
std::optional<bool> parse_copy(nb::handle o);

// `shape` argument: an int or a tuple of ints (docs/design/array.md §2.8).
std::vector<std::int64_t> parse_shape(nb::handle o);

// The same, without a heap allocation for up to kMaxNdim axes (zeros(3) and the
// other creation functions: the vector was a fifth of a tiny call). A longer
// tuple is parsed by parse_shape, so its errors are the same.
struct ShapeArg {
    std::array<std::int64_t, kMaxNdim> small;
    std::vector<std::int64_t> big;
    std::size_t n = 0;
    std::span<const std::int64_t> span() const { return big.empty() && n <= small.size() ? std::span<const std::int64_t>(small.data(), n) : std::span<const std::int64_t>(big); }
};
ShapeArg parse_shape_arg(nb::handle o);

// An int argument (index_value), e.g. `num` or `k`. TypeError for anything
// else, OverflowError outside int64 (API.md §2).
std::int64_t parse_int(nb::handle o, const char* what);

// Maps axis in [-ndim, ndim) to [0, ndim); ValueError otherwise.
int normalize_axis(std::int64_t axis, int ndim);

// `axis` argument: an int, or a tuple or list of ints (docs/design/array.md §2.8), each
// normalized against ndim. Repeated axes are a ValueError. `what` names the
// argument in messages (permute_dims calls it "axes").
std::vector<int> parse_axes(nb::handle o, int ndim, const char* what = "axis");

inline bool is_ndarray(nb::handle o) { return nb::isinstance<NDArray>(o); }
inline NDArray& as_ndarray(nb::handle o) { return *nb::inst_ptr<NDArray>(o); }
// as_ndarray, with a TypeError naming the argument when `o` is not an array.
NDArray& require_ndarray(nb::handle o, const char* what = "x");
inline nb::object wrap(NDArray&& a) { return nb::cast(std::move(a)); }

// A new array object over the same memory as array `x`, ready to have its
// data/shape/strides edited (call update_flags() afterwards). Its `base` is
// the final owner of the memory: views of views never form chains.
NDArray view_of(nb::handle x);

}  // namespace sd
