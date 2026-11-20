#include "linalg/pyrefs.hpp"

#include "array/ndarray.hpp"

#include <string>
#include <vector>

namespace sd::linalg {

PyRefs& refs() {
    static PyRefs r;
    return r;
}

namespace {

PyObject* keep(nb::handle h) {
    h.inc_ref();
    return h.ptr();
}

}  // namespace

void bind_setup(nb::module_& m) {
    // Called once by stardust/linalg/__init__.py with the types defined in Python;
    // `results` lists the namedtuples in the order of the Result enum.
    m.def("_setup", [](nb::handle linalg_error, nb::tuple results) {
        if (results.size() != static_cast<size_t>(Result::Count))
            throw nb::value_error("_setup: wrong number of result types");
        refs().linalg_error = keep(linalg_error);
        for (size_t i = 0; i < results.size(); ++i) refs().results[i] = keep(results[i]);
    });
}

nb::object make_result(Result r, std::initializer_list<nb::handle> fields) {
    // The array fields are read-only (API.md §2): a result object's fields are views of one
    // computation, never edited in place; .copy() gives a writable array.
    std::vector<PyObject*> args;
    for (nb::handle f : fields) {
        NDArray* a = nullptr;
        if (nb::try_cast<NDArray*>(f, a) && a != nullptr) a->writeable = false;
        args.push_back(f.ptr());
    }
    PyObject* o = PyObject_Vectorcall(refs().results[static_cast<int>(r)], args.data(), args.size(), nullptr);
    if (!o) throw nb::python_error();
    return nb::steal(o);
}

}  // namespace sd::linalg
