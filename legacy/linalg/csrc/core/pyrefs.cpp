#include "core/pyrefs.hpp"

#include <string>
#include <vector>

namespace sdla {

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

void load_sdnp_refs() {
    nb::module_ sdnp = nb::module_::import_("stardust_numpy");
    refs().ndarray_type = keep(sdnp.attr("ndarray"));
    refs().asarray = keep(sdnp.attr("asarray"));
    refs().get_num_threads = keep(sdnp.attr("get_num_threads"));
    const char* names[4] = {"bool", "int64", "float64", "complex128"};
    for (int i = 0; i < 4; ++i) refs().dtypes[i] = keep(sdnp.attr(names[i]));
}

void bind_setup(nb::module_& m) {
    // Called once by stardust_linalg/__init__.py with the types defined in Python;
    // `results` lists the namedtuples in the order of the Result enum.
    m.def("_setup", [](nb::handle linalg_error, nb::tuple results) {
        if (results.size() != static_cast<size_t>(Result::Count))
            throw nb::value_error("_setup: wrong number of result types");
        refs().linalg_error = keep(linalg_error);
        for (size_t i = 0; i < results.size(); ++i) refs().results[i] = keep(results[i]);
    });
}

nb::object make_result(Result r, std::initializer_list<nb::handle> fields) {
    std::vector<PyObject*> args;
    for (nb::handle f : fields) args.push_back(f.ptr());
    PyObject* o = PyObject_Vectorcall(refs().results[static_cast<int>(r)], args.data(), args.size(), nullptr);
    if (!o) throw nb::python_error();
    return nb::steal(o);
}

}  // namespace sdla
