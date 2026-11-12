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

// One elementwise function: Op::nin positional inputs, then out= when it has
// one result.
template <typename Op>
void def_nary(nb::module_& m) {
    constexpr std::int64_t weight = special::weight_of(Op::cost_ns);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        if constexpr (Op::nout == 1) {
            m.def(
                Op::name,
                [](handle_t<I>... xs, nb::handle out) {
                    return call_named(Op::name, [&] {
                        const nb::handle objs[] = {xs...};
                        return nary<Op>(objs, out, weight);
                    });
                },
                ((void)I, nb::arg().none())..., nb::kw_only(), nb::arg("out").none() = nb::none());
        } else {
            m.def(
                Op::name,
                [](handle_t<I>... xs) {
                    return call_named(Op::name, [&] {
                        const nb::handle objs[] = {xs...};
                        return nary<Op>(objs, nb::none(), weight);
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
    def_naries<Fma3, Sum4, Powers4, CfTan>(m);  // engine tests (special/ops/testing.hpp)
    m.def("_parallel_tasks", &parallel_tasks, nb::arg("n"), nb::arg("weight"));
    m.def("_weight", [](const std::string& name) -> std::int64_t {
        if (name == Fma3::name) return weight_of(Fma3::cost_ns);
        if (name == Sum4::name) return weight_of(Sum4::cost_ns);
        if (name == Powers4::name) return weight_of(Powers4::cost_ns);
        if (name == CfTan::name) return weight_of(CfTan::cost_ns);
        throw nb::key_error(name.c_str());
    }, nb::arg("name"));
}

}  // namespace sd
