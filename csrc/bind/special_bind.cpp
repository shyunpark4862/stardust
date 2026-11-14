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
#include "special/ops/erf.hpp"
#include "special/ops/gamma.hpp"
#include "special/ops/incbeta.hpp"
#include "special/ops/incgamma.hpp"
#include "special/ops/logistic.hpp"
#include "special/ops/testing.hpp"

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
    def_naries<Erf, Erfc, Erfcx, Erfinv, Erfcinv, Ndtr, LogNdtr, Ndtri>(m);  // S2, §2
    def_naries<ErfcSelect, ErfcEstrin>(m);  // S2 measurement variants (special/ops/erf.hpp)
    def_naries<Gamma, Rgamma, Gammaln, Digamma, Polygamma, Beta, Betaln, Binom>(m);  // S3, §3
    def_naries<Gammainc, Gammaincc, Gammaincinv, Gammainccinv>(m);  // S4, §3
    def_naries<Betainc, Betaincc, Betaincinv, Betainccinv>(m);  // S4, §3
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
    def_naries<Fma3, Sum4, Powers4, CfTan>(m);  // engine tests (special/ops/testing.hpp)
    m.def("_parallel_tasks", &parallel_tasks, nb::arg("n"), nb::arg("weight"));
    m.def("_weight", [](const std::string& name) -> std::int64_t {
        std::int64_t w = 0;
        auto check = [&]<typename Op>() { if (name == Op::name) w = weight_of(Op::cost_ns); };
        [&]<typename... Ops>() { (check.template operator()<Ops>(), ...); }
            .template operator()<Expit, Logit, LogExpit, Xlogy, Xlog1py, Entr, RelEntr, Erf, Erfc, Erfcx, Erfinv,
                                 Erfcinv, Ndtr, LogNdtr, Ndtri, Gamma, Rgamma, Gammaln, Digamma, Polygamma,
                                 Beta, Betaln, Binom, Zeta, Gammainc, Gammaincc,
                                 Gammaincinv, Gammainccinv, Betainc, Betaincc, Betaincinv, Betainccinv, Fma3, Sum4, Powers4, CfTan>();
        if (name == "zeta_hurwitz") w = weight_of(Hurwitz::cost_ns);  // zeta's two-argument form
        if (w == 0) throw nb::key_error(name.c_str());
        return w;
    }, nb::arg("name"));
}

}  // namespace sd
