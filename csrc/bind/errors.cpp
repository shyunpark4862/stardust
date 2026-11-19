#include "bind/errors.hpp"

#include "linalg/errors.hpp"

#include <cstring>
#include <new>
#include <stdexcept>

namespace sd {

std::string py_type_name(nb::handle o) {
    // tp_name is "module.Name" for static types (numpy.int64), "Name" for heap types: the
    // short name, except numpy's types, which keep their public module (numpy.bool,
    // numpy.float64, numpy.ndarray, numpy.random.Generator: the module without its private
    // parts, numpy.random._generator; not to be read as Python's bool or sd's ndarray; user
    // decision 2026-11-19, API.md §2.1). The one helper of every area (Python's too:
    // _core._type_name).
    PyTypeObject* t = Py_TYPE(o.ptr());
    const char* name = t->tp_name;
    const char* dot = std::strrchr(name, '.');
    std::string out = dot ? dot + 1 : name;
    PyObject* mod = PyObject_GetAttrString(reinterpret_cast<PyObject*>(t), "__module__");
    if (mod == nullptr) {
        PyErr_Clear();
        return out;
    }
    if (PyUnicode_Check(mod)) {
        const char* m = PyUnicode_AsUTF8(mod);
        if (m == nullptr) PyErr_Clear();
        else if (std::strncmp(m, "numpy", 5) == 0 && (m[5] == '\0' || m[5] == '.')) {
            std::string pub;  // the module's parts up to the first private one
            for (const char *p = m, *e; *p != '\0'; p = *e ? e + 1 : e) {
                e = std::strchr(p, '.');
                if (e == nullptr) e = p + std::strlen(p);
                if (*p == '_') break;
                pub += (pub.empty() ? "" : ".") + std::string(p, e);
            }
            out = pub + "." + out;
        }
    }
    Py_DECREF(mod);
    return out;
}

std::string with_name(const char* func, const char* msg) {
    if (func == nullptr || *func == '\0') return msg;
    const std::string prefix = std::string(func) + (func[0] == '.' ? ": " : "(): ");
    if (std::strncmp(msg, prefix.c_str(), prefix.size()) == 0) return msg;
    return prefix + msg;
}

void rethrow_named(const char* func) {
    try {
        throw;
    } catch (const nb::builtin_exception& e) {
        switch (e.type()) {
            case nb::exception_type::type_error:
            case nb::exception_type::value_error:
            case nb::exception_type::index_error:
            case nb::exception_type::key_error:
            case nb::exception_type::buffer_error:
                throw nb::builtin_exception(e.type(), with_name(func, e.what()).c_str());
            default:
                throw;  // stop_iteration, next_overload, ...: control flow, not messages
        }
    } catch (const linalg::LinAlgError& e) {
        throw linalg::LinAlgError(with_name(func, e.what()));
    } catch (const std::overflow_error& e) {
        throw std::overflow_error(with_name(func, e.what()));
    } catch (const std::bad_alloc& e) {  // MemoryError; Buffer::allocate's says why
        const bool plain = std::strcmp(e.what(), "std::bad_alloc") == 0;
        PyErr_SetString(PyExc_MemoryError, with_name(func, plain ? "out of memory" : e.what()).c_str());
        throw nb::python_error();
    }
}

}  // namespace sd
