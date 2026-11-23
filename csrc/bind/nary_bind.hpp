// Binding an elementwise Op as a Python function through the array's n-ary
// dispatcher (array/ops/ufunc.hpp, run_nary): Op::nin positional-only inputs,
// then out= (keyword-only) when the Op has one result. Shared by the special
// and stats areas (docs/api/special.md §1, docs/api/stats.md §1).
#pragma once

#include "array/ops/ufunc.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"
#include "bind/pyutil.hpp"
#include "bind/errors.hpp"
#include "special/cost.hpp"
#include "special/kernels/chunked.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
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

// The cost of a call per element in ns: the Op's cost, or, for an Op whose cost depends on
// its scalar parameters (stats' discrete distributions on a narrow support: Op::cost_for),
// the cost for the parameters given, when every parameter is a Python float or int.
template <typename Op>
double cost_for(std::span<const nb::handle> objs) {
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
                        return Op::cost_ns;
                    }
                } else return Op::cost_ns;
            }
            return Op::cost_for(p);
        }
    }
    return Op::cost_ns;
}

// The number of elements of the call's broadcast shape, from its array operands and out=
// (scalars broadcast): the largest extent on each axis, a 0 anywhere giving 0. Only an
// estimate for the weight (a mismatch of shapes raises later, in run_nary).
inline std::int64_t broadcast_size(std::span<const nb::handle> objs, nb::handle out) {
    Dims dims;  // from the last axis
    dims.fill(1);
    int ndim = 0;
    auto take = [&](nb::handle o) {
        if (!is_ndarray(o)) return;
        const NDArray& a = as_ndarray(o);
        for (int i = 0; i < a.ndim; ++i) {
            const std::int64_t x = a.shape[a.ndim - 1 - i];
            std::int64_t& e = dims[i];
            e = x == 0 || e == 0 ? 0 : std::max(e, x);
        }
        ndim = std::max(ndim, a.ndim);
    };
    for (nb::handle o : objs) take(o);
    take(out);
    std::int64_t n = 1;
    for (int d = 0; d < ndim; ++d) n *= dims[d];
    return n;
}

// The weight of a call (the work per element in adds, docs/design/special.md §3.3). It is
// capped at special::kMaxWeight, which sets the task size of a heavy function (16 elements);
// whether the call runs without the GIL is decided by the work before the cap (API.md §3):
// cost x elements against the parallel threshold. A heavy call below the threshold at the
// capped weight but above it at its own (jv(1e300, 1e300) on 127 elements: 6 s with the GIL
// held) gets the weight that just reaches the threshold, threshold / elements, so it runs
// without the GIL in tasks of about elements / 8. The weight never changes a result.
template <typename Op>
std::int64_t weight_for(std::span<const nb::handle> objs, nb::handle out) {
    const double ns = cost_for<Op>(objs);
    const std::int64_t w = special::weight_of(ns);
    if (w < special::kMaxWeight) return w;
    const std::int64_t n = broadcast_size(objs, out);
    if (n == 0 || n * w >= kParallelThreshold) return w;
    if (ns / special::kAddNs * static_cast<double>(n) < static_cast<double>(kParallelThreshold)) return w;
    return (kParallelThreshold + n - 1) / n;
}

// Whether a scalar call of the Op (v: its inputs) runs without the GIL: the rule of every call
// (API.md §3), the work of one element against the parallel threshold, as weight_for gives it to
// a one-element call. Known when compiling for most Ops, so their scalar calls pay nothing for
// it. An Op whose cost depends on its scalar parameters (stats' discrete distributions: a narrow
// support answered from the memo in a few ns, a wide one in tens of us) decides with the cost of
// the parameters given, so that a call much shorter than releasing the GIL keeps it.
template <typename Op>
inline constexpr bool kScalarUnlocked = Op::cost_ns / special::kAddNs >= static_cast<double>(kParallelThreshold);
template <typename Op>
bool scalar_unlocked(const double* v) {
    if constexpr (!kScalarUnlocked<Op>) {
        return false;
    } else if constexpr (requires { Op::memo_cost_ns; }) {
        if constexpr (Op::memo_cost_ns > 0.0)
            return Op::cost_for(v) / special::kAddNs >= static_cast<double>(kParallelThreshold);
        return true;
    } else {
        return true;
    }
}

// The call. The commonest scalar call, every input a Python float and no out=, takes a shorter
// way: the Op's float64 loop (the one run_nary selects for those inputs, so the same bits) on
// one element from the stack, its result a Python float as run_nary's. run_nary's argument
// handling and 0-d arrays were about 30 ns of a 60 ns erf(x). Anything else (ints, complex,
// arrays, out=) is run_nary.
template <typename Op>
nb::object call(std::span<const nb::handle> objs, nb::handle out) {
    if constexpr (Op::nout == 1) {
        const LoopEntry& e = table_of<Op>()[static_cast<std::size_t>(DType::Float64)];
        if (out.is_none() && e.loop != nullptr && e.out == DType::Float64) {
            double v[Op::nin + 1];
            bool floats = true;
            for (int k = 0; k < Op::nin && floats; ++k) {
                PyObject* o = objs[static_cast<std::size_t>(k)].ptr();
                floats = PyFloat_CheckExact(o);
                if (floats) v[k] = PyFloat_AS_DOUBLE(o);
            }
            if (floats) {
                std::byte* args[Op::nin + 1];
                std::int64_t st[Op::nin + 1];
                for (int k = 0; k <= Op::nin; ++k) {
                    args[k] = reinterpret_cast<std::byte*>(&v[k]);
                    st[k] = sizeof(double);
                }
                {
                    const parallel::Unlocked unlocked(scalar_unlocked<Op>(v));
                    e.loop(args, 1, st, nullptr);
                }
                return nb::steal(PyFloat_FromDouble(v[Op::nin]));
            }
        }
    }
    return run_nary(table_of<Op>(), Op::nout, Op::args, objs, out, weight_for<Op>(objs, out));
}

// The Python signature, "def name(a, b, /, *, out=None)" from Op::args (API.md §2.1: the
// inputs named, still positional-only), kept for the module's lifetime.
template <typename Op>
const char* signature() {
    static const std::string s = [] {
        std::string r = std::string("def ") + Op::name + "(";
        for (std::size_t i = 0; i < Op::args.size(); ++i) r += std::string(i ? ", " : "") + Op::args[i];
        r += Op::nout == 1 ? ", /, *, out=None)" : ", /)";
        return r;
    }();
    return s.c_str();
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
                nb::sig(signature<Op>()), ((void)I, nb::arg().none())..., nb::kw_only(),
                nb::arg("out").none() = nb::none()
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
                nb::sig(signature<Op>()), ((void)I, nb::arg().none())...
            );
        }
    }(std::make_index_sequence<Op::nin>{});
}

}  // namespace sd::nary_bind
