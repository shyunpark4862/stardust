// The special-function area: the submodule stardust._core.special
// (docs/design/special.md §1.4). Every function is elementwise and goes
// through the array's n-ary dispatcher (array/ops/ufunc.hpp, nary): inputs
// positional-only, out= keyword-only, and no out= when there are several
// results (docs/api/special.md §1).
#include "bind/special_bind.hpp"

#include "array/iter.hpp"
#include "array/ops/ufunc.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "special/cost.hpp"
#include "special/kernels/chunked.hpp"
#include "special/ops/airy.hpp"
#include "special/ops/bessel.hpp"
#include "special/ops/bessel01.hpp"
#include "special/ops/ellip.hpp"
#include "special/ops/erf.hpp"
#include "special/ops/expint.hpp"
#include "special/ops/gamma.hpp"
#include "special/ops/hyper.hpp"
#include "special/ops/lambertw.hpp"
#include "special/ops/incbeta.hpp"
#include "special/ops/incgamma.hpp"
#include "special/ops/loggamma.hpp"
#include "special/ops/logistic.hpp"
#include "special/ops/testing.hpp"
#include "special/ops/wofz.hpp"
#include "special/reduce.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>
#include <utility>

namespace nb = nanobind;

namespace sd {

namespace {

template <std::size_t> using handle_t = nb::handle;

// An Op is either a scalar definition f (the n-ary loop) or a chunk kernel
// (NEON + vForce, special/kernels/chunked.hpp).
template <typename Op>
constexpr const LoopTable& table_of() {
    if constexpr (requires { &Op::chunk; }) return special::chunked_table<Op>;
    else return nary_table<Op>;
}

template <typename Op>
nb::object call(std::span<const nb::handle> objs, nb::handle out) {
    return run_nary(table_of<Op>(), Op::nout, Op::args, objs, out, special::weight_of(Op::cost_ns));
}

// One elementwise function: Op::nin positional inputs, then out= when it has
// one result.
template <typename Op>
void def_nary(nb::module_& m) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        if constexpr (Op::nout == 1) {
            m.def(
                Op::name,
                [](handle_t<I>... xs, nb::handle out) {
                    return call_named(Op::name, [&] {
                        const nb::handle objs[] = {xs...};
                        return call<Op>(objs, out);
                    });
                },
                ((void)I, nb::arg().none())..., nb::kw_only(), nb::arg("out").none() = nb::none());
        } else {
            m.def(
                Op::name,
                [](handle_t<I>... xs) {
                    return call_named(Op::name, [&] {
                        const nb::handle objs[] = {xs...};
                        return call<Op>(objs, nb::none());
                    });
                },
                ((void)I, nb::arg().none())...);
        }
    }(std::make_index_sequence<Op::nin>{});
}

template <typename... Ops>
void def_naries(nb::module_& m) { (def_nary<Ops>(m), ...); }

// A complex128 operand: a complex sd.ndarray or a Python complex (S8: the functions
// that take complex choose their complex loop by the input, not by promotion).
bool is_complex_operand(nb::handle o) {
    if (is_ndarray(o)) return as_ndarray(o).dtype == DType::Complex128;
    return PyComplex_Check(o.ptr());
}

// A one-input function with a real Op and a complex Op of the same name (S8, §12).
template <typename Real, typename Complex>
void def_real_complex(nb::module_& m) {
    m.def(
        Real::name,
        [](nb::handle x, nb::handle out) {
            return call_named(Real::name, [&] {
                const nb::handle objs[] = {x};
                return is_complex_operand(x) ? call<Complex>(objs, out) : call<Real>(objs, out);
            });
        },
        nb::arg().none(), nb::kw_only(), nb::arg("out").none() = nb::none());
}

// For tests: how many tasks the iterator cuts a contiguous float64
// elementwise call of n elements into, at `weight` adds per element
// (one input, one output). Depends on n and weight only, never on the
// thread count.
std::int64_t parallel_tasks(std::int64_t n, std::int64_t weight) {
    const std::int64_t shape[] = {n};
    const std::int64_t strides[] = {8};
    const IterOperand ops[] = {{nullptr, shape, strides}, {nullptr, shape, strides}};
    return NdIter(shape, ops).plan(Split::Elementwise, weight).tasks;
}

}  // namespace

void bind_special(nb::module_& core) {
    nb::module_ m = core.def_submodule("special", "stardust special-function native core");
    using namespace special;
    def_naries<Expit, Logit, LogExpit, Xlogy, Xlog1py, Entr, RelEntr>(m);  // S1, docs/api/special.md §5
    def_real_complex<Erf, ErfComplex>(m);  // S2, §2; complex S8, §12
    def_real_complex<Erfc, ErfcComplex>(m);
    def_real_complex<Erfcx, ErfcxComplex>(m);
    def_naries<Erfinv, Erfcinv, Ndtr, LogNdtr, Ndtri>(m);  // S2, §2
    def_naries<ErfcSelect, ErfcEstrin>(m);  // S2 measurement variants (special/ops/erf.hpp)
    def_naries<Gamma, Rgamma, Gammaln, Digamma, Polygamma, Beta, Betaln, Binom>(m);  // S3, §3
    def_naries<Gammainc, Gammaincc, Gammaincinv, Gammainccinv>(m);  // S4, §3
    def_naries<Betainc, Betaincc, Betaincinv, Betainccinv>(m);  // S4, §3
    def_naries<Expi, Expn, Ellipk, Ellipe>(m);  // S5, §7, §10
    def_naries<EllipkAgm, EllipeCarlson>(m);  // S5 measurement variants (the design's methods)
    def_naries<I0, I0e, I1, I1e, K0, K0e, K1, K1e, J0, J1, Y0, Y1>(m);  // S6, §8.1
    def_naries<Jv, Yv, Iv, Kv, Jve, Yve, Ive, Kve>(m);  // S6, §8.2
    def_naries<Hyp1f1, Hyp2f1>(m);  // S7, §11
    def_naries<Wofz, Loggamma>(m);  // S8, §12
    // lambertw(x, /, *, k=0, out=None): k is an int; 0 or -1 for a real x (§11), any for a
    // complex one (§12)
    m.def(
        "lambertw",
        [](nb::handle x, nb::handle k, nb::handle out) {
            return call_named("lambertw", [&] {
                if (!PyLong_Check(k.ptr()) || PyBool_Check(k.ptr())) throw nb::type_error("'k' must be an int");
                if (is_complex_operand(x)) {
                    const long kv = PyLong_AsLong(k.ptr());
                    if (kv == -1 && PyErr_Occurred()) throw nb::python_error();
                    const nb::object kc = nb::steal(PyComplex_FromDoubles(static_cast<double>(kv), 0.0));
                    const nb::handle objs[] = {x, kc};
                    return call<LambertwComplex>(objs, out);
                }
                const long kv = PyLong_AsLong(k.ptr());
                if (kv == -1 && PyErr_Occurred()) PyErr_Clear();
                if ((kv != 0 && kv != -1) || PyErr_Occurred()) throw nb::value_error("'k' must be 0 or -1 for a real argument");
                const nb::handle objs[] = {x};
                return kv == 0 ? call<Lambertw0>(objs, out) : call<LambertwM1>(objs, out);
            });
        },
        nb::arg().none(), nb::kw_only(), nb::arg("k").none() = 0, nb::arg("out").none() = nb::none());
    // airy(x, /) -> AiryResult(ai, aip, bi, bip) (§9; stardust._results)
    m.def(
        "airy",
        [](nb::handle x) {
            return call_named("airy", [&] {
                const nb::handle objs[] = {x};
                const nb::object t = call<Airy>(objs, nb::none());
                return nb::module_::import_("stardust._results").attr("AiryResult")(*t);
            });
        },
        nb::arg().none());
    def_naries<GammaLanczos>(m);  // S3 measurement variant (gamma candidate A)
    // zeta(x, q=None, /, *, out=None): Riemann with one argument, Hurwitz with two (§7)
    m.def(
        "zeta",
        [](nb::handle x, nb::handle q, nb::handle out) {
            return call_named("zeta", [&] {
                if (q.is_none()) {
                    const nb::handle objs[] = {x};
                    return call<Zeta>(objs, out);
                }
                const nb::handle objs[] = {x, q};
                return call<Hurwitz>(objs, out);
            });
        },
        nb::arg().none(), nb::arg().none() = nb::none(), nb::kw_only(), nb::arg("out").none() = nb::none());
    // S5 reductions (§6): the array's max and pairwise sum (special/reduce.cpp)
    m.def(
        "logsumexp",
        [](nb::handle x, nb::handle axis, nb::handle keepdims) {
            return call_named("logsumexp", [&] { return logsumexp(x, axis, keepdims); });
        },
        nb::arg().none(), nb::kw_only(), nb::arg("axis").none() = nb::none(), nb::arg("keepdims").none() = false);
    m.def(
        "softmax", [](nb::handle x, nb::handle axis) { return call_named("softmax", [&] { return softmax(x, axis); }); },
        nb::arg().none(), nb::kw_only(), nb::arg("axis").none() = nb::none());
    m.def(
        "log_softmax",
        [](nb::handle x, nb::handle axis) { return call_named("log_softmax", [&] { return log_softmax(x, axis); }); },
        nb::arg().none(), nb::kw_only(), nb::arg("axis").none() = nb::none());
    def_naries<Fma3, Sum4, Powers4, CfTan>(m);  // engine tests (special/ops/testing.hpp)
    m.def("_parallel_tasks", &parallel_tasks, nb::arg("n"), nb::arg("weight"));
    m.def("_weight", [](const std::string& name) -> std::int64_t {
        std::int64_t w = 0;
        auto check = [&]<typename Op>() { if (name == Op::name) w = weight_of(Op::cost_ns); };
        [&]<typename... Ops>() { (check.template operator()<Ops>(), ...); }
            .template operator()<Expit, Logit, LogExpit, Xlogy, Xlog1py, Entr, RelEntr, Erf, Erfc, Erfcx, Erfinv,
                                 Erfcinv, Ndtr, LogNdtr, Ndtri, Gamma, Rgamma, Gammaln, Digamma, Polygamma,
                                 Beta, Betaln, Binom, Zeta, Gammainc, Gammaincc,
                                 Gammaincinv, Gammainccinv, Betainc, Betaincc, Betaincinv, Betainccinv, Expi, Expn, Ellipk, Ellipe, I0, I0e, I1, I1e, K0, K0e, K1, K1e, J0, J1, Y0,
                                 Y1, Jv, Yv, Iv, Kv, Jve, Yve, Ive, Kve, Airy, Lambertw0, Hyp1f1, Hyp2f1, Wofz, Loggamma, Fma3, Sum4, Powers4, CfTan>();
        if (name == "zeta_hurwitz") w = weight_of(Hurwitz::cost_ns);  // zeta's two-argument form
        if (name == "lambertw_m1") w = weight_of(LambertwM1::cost_ns);  // lambertw(k=-1)
        // S8: the complex loops (reference-file names: erf_complex, lambertw_c1, ...)
        if (name == "erf_complex") w = weight_of(ErfComplex::cost_ns);
        if (name == "erfc_complex") w = weight_of(ErfcComplex::cost_ns);
        if (name == "erfcx_complex") w = weight_of(ErfcxComplex::cost_ns);
        if (name.rfind("lambertw_c", 0) == 0) w = weight_of(LambertwComplex::cost_ns);
        if (w == 0) throw nb::key_error(name.c_str());
        return w;
    }, nb::arg("name"));
}

}  // namespace sd
