#include "core/pyrefs.hpp"

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
}

void bind_setup(nb::module_& m) {
    // Called once by stardust_linalg/__init__.py with the types defined in Python.
    m.def("_setup", [](nb::handle linalg_error, nb::handle slogdet_result) {
        refs().linalg_error = keep(linalg_error);
        refs().slogdet_result = keep(slogdet_result);
    });
}

}  // namespace sdla
