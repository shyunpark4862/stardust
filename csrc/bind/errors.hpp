// The public function's name in error messages (API.md §2.1).
//
// A message raised by a public function starts with "name(): ". Helpers and
// kernels reached from several functions write the message without a name;
// the binding of each public function adds its own:
//
//     m.def("zeros", named("zeros", &zeros), ...);          // a function pointer
//     return call_named(func, [&] { return kernel(...); });  // a name passed down from Python
//
// Operators, indexing and the other Python protocols are bound without a name
// and keep Python's own message form.
#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <utility>

namespace sd {

namespace nb = nanobind;

// "name(): " + msg, or msg unchanged when it already starts with that. A name
// starting with '.' is a property: ".T: " + msg. A null or empty name adds nothing.
std::string with_name(const char* func, const char* msg);

// A TypeError raised for an sd.sparse array given where an sd.ndarray is required
// (require_ndarray): rethrow_named names the function and adds what to use for a sparse
// array, sparse_hint(func) (docs/design/sparse.md §3.10, decision SP67). Anywhere else it is
// a plain TypeError with its message.
struct SparseOperand : nb::builtin_exception {
    explicit SparseOperand(const std::string& msg)
        : nb::builtin_exception(nb::exception_type::type_error, msg.c_str()) {}
};

// "; for a sparse array use A.sum()" (the sparse method, operator or function for array function
// `func`), else "; a sparse array has no func: use func(A.toarray())". In bind/pyutil.cpp.
std::string sparse_hint(const std::string& func);

// Inside a catch block: rethrows the exception being handled with `func`'s
// name in front of its message (TypeError, ValueError, IndexError, KeyError,
// BufferError, OverflowError and LinAlgError keep their type). Anything
// else -- a Python error already set, MemoryError, nanobind's own control
// flow -- is rethrown as it is.
[[noreturn]] void rethrow_named(const char* func);

// The name of `o`'s type, for "..., not list" in messages: the type's __name__ without a
// module path ("ndarray", "list", not "stardust._core.ndarray"), except numpy's types,
// which keep "numpy." ("numpy.bool", "numpy.int64", "numpy.ndarray": not to be read as
// Python's bool or sd's int64 and ndarray; user decision 2026-11-19, API.md §2.1). Here
// rather than in pyutil.hpp so the frame area can use it without the array headers.
std::string py_type_name(nb::handle o);

template <class F>
decltype(auto) call_named(const char* func, F&& f) {
    try {
        return std::forward<F>(f)();
    } catch (...) { rethrow_named(func); }
}

// A function pointer bound as the public function `func`: same signature, its
// error messages named. `func` must outlive the module (a string literal or Op::name).
template <class R, class... A>
auto named(const char* func, R (*f)(A...)) {
    return [func, f](A... args) -> R {
        try {
            return f(std::forward<A>(args)...);
        } catch (...) { rethrow_named(func); }
    };
}

}  // namespace sd
