// Binding an elementwise Op as a Python function through the array's n-ary
// dispatcher (array/ops/ufunc.hpp, run_nary): Op::nin positional-only inputs,
// then out= (keyword-only) when the Op has one result. Shared by the special
// and stats areas (docs/api/special.md §1, docs/api/stats.md §1).
#pragma once

#include "array/ops/ufunc.hpp"
#include "bind/errors.hpp"
#include "special/cost.hpp"
#include "special/kernels/chunked.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <span>
#include <utility>

namespace sd::nary_bind {

namespace nb = nanobind;

template <std::size_t> using handle_t = nb::handle;

// An Op is either a scalar definition f (the n-ary loop) or a chunk kernel
// (NEON + vForce, special/kernels/chunked.hpp).
template <typename Op>
constexpr const LoopTable& table_of() {
    if constexpr (requires { &Op::chunk; }) return special::chunked_table<Op>;
    else if constexpr (requires { Op::table; })
        return Op::table != nullptr ? *Op::table : nary_table<Op>;  // stats' chunk kernels
    else return nary_table<Op>;
}

template <typename Op>
nb::object call(std::span<const nb::handle> objs, nb::handle out) {
    return run_nary(table_of<Op>(), Op::nout, Op::args, objs, out, special::weight_of(Op::cost_ns));
}

// m.def(Op::name, ...): the inputs, then out= when there is one result.
template <typename Op>
void def(nb::module_& m) {
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
                ((void)I, nb::arg().none())..., nb::kw_only(), nb::arg("out").none() = nb::none()
            );
        } else {
            m.def(
                Op::name,
                [](handle_t<I>... xs) {
                    return call_named(Op::name, [&] {
                        const nb::handle objs[] = {xs...};
                        return call<Op>(objs, nb::none());
                    });
                },
                ((void)I, nb::arg().none())...
            );
        }
    }(std::make_index_sequence<Op::nin>{});
}

}  // namespace sd::nary_bind
