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
#include <cstdint>
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

// The weight of a call: the Op's cost, or, for an Op whose cost depends on its scalar
// parameters (stats' discrete distributions on a narrow support: Op::cost_for), the cost for
// the parameters given, when every parameter is a Python float or int.
template <typename Op>
std::int64_t weight_for(std::span<const nb::handle> objs) {
    if constexpr (requires {
                      Op::memo_cost_ns;
                      Op::kParams;
                  }) {
        if constexpr (Op::memo_cost_ns > 0.0) {
            double p[Op::kParams];
            for (std::size_t k = 0; k < Op::kParams; ++k) {
                PyObject* o = objs[k].ptr();
                if (PyFloat_CheckExact(o)) p[k] = PyFloat_AS_DOUBLE(o);
                else if (PyLong_CheckExact(o)) {
                    p[k] = PyLong_AsDouble(o);
                    if (p[k] == -1.0 && PyErr_Occurred()) {
                        PyErr_Clear();
                        return special::weight_of(Op::cost_ns);
                    }
                } else return special::weight_of(Op::cost_ns);
            }
            return special::weight_of(Op::cost_for(p));
        }
    }
    return special::weight_of(Op::cost_ns);
}

template <typename Op>
nb::object call(std::span<const nb::handle> objs, nb::handle out) {
    return run_nary(table_of<Op>(), Op::nout, Op::args, objs, out, weight_for<Op>(objs));
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
