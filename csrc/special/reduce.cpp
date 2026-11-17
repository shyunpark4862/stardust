// logsumexp, softmax, log_softmax (docs/api/special.md §6, docs/design/special.md §3.5; S5).
//
// Two passes over the array's own reductions: m = max(x) (array/ops/reduction.cpp,
// NaN propagates), then the pairwise sum of exp(x - m) -- deterministic for any
// thread count (docs/design/array.md §6.2). The elementwise steps between them
// are the Ops below, run through the n-ary dispatcher with broadcasting
// (keepdims=True keeps m and the sum broadcastable against x).
//
// Accuracy (S5 user decision): logsumexp and log_softmax |error| <=
// 1e-14 max(1, |value|), softmax relative 1e-14 per element. x - m rounds to
// d = hi + lo (TwoSum, exact), and exp(d) = exp(hi) (1 + lo): rounding d itself
// would cost |d| 2^-53 relative in exp -- 7.8e-14 at d = -700.
//
// scipy's conventions on infinities: logsumexp and log_softmax replace an
// infinite max by 0 (so logsumexp([inf, 1]) = inf, log_softmax = [nan, -inf]);
// softmax does not (softmax([inf, 1]) = [nan, nan], all -inf gives nan).
#include "special/reduce.hpp"

#include "array/ndarray.hpp"
#include "array/ops/reduction.hpp"
#include "array/ops/ufunc.hpp"
#include "bind/pyutil.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/gamma_core.hpp"

#include <array>
#include <cmath>

namespace sd {

namespace special {

namespace reduce_ops {

// m where finite, else 0 (scipy's shift for logsumexp, log_softmax).
struct FiniteOrZero : Real {
    static constexpr const char* name = "_finite_or_zero";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"m"};
    static constexpr double cost_ns = 0.3;
    static double f(double m) { return std::isfinite(m) ? m : 0.0; }
};

// exp(x - m) with x - m = hi + lo exact.
struct ExpDiff : Real {
    static constexpr const char* name = "_exp_diff";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"x", "m"};
    static constexpr double cost_ns = 2.5;
    static double f(double x, double m) {
        const TwoSum d = two_sum(x, -m);
        const double e = std::exp(d.s);
        if (!std::isfinite(d.s) || e == 0.0) return e;  // inf - inf = NaN, exp(-inf) = 0: no correction
        return e + e * d.err;
    }
};

// m + log(s)
struct LseCombine : Real {
    static constexpr const char* name = "_lse_combine";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"m", "s"};
    static constexpr double cost_ns = 2.0;
    static double f(double m, double s) { return m + std::log(s); }
};

// (x - m) - log s with x - m = hi + lo: (hi - log s) + lo
struct LogSoftmaxCombine : Real {
    static constexpr const char* name = "_log_softmax_combine";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"x", "m", "logs"};
    static constexpr double cost_ns = 1.0;
    static double f(double x, double m, double logs) {
        const TwoSum d = two_sum(x, -m);
        if (!std::isfinite(d.s)) return d.s - logs;
        return (d.s - logs) + d.err;
    }
};

struct Divide : Real {
    static constexpr const char* name = "_divide";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"e", "s"};
    static constexpr double cost_ns = 0.5;
    static double f(double e, double s) { return e / s; }
};

struct Log : Real {
    static constexpr const char* name = "_log";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"s"};
    static constexpr double cost_ns = 2.0;
    static double f(double s) { return std::log(s); }
};

template <typename Op, typename... H>
nb::object ew(H... hs) {
    const nb::handle objs[] = {hs...};
    return nary<Op>(objs, nb::none(), weight_of(Op::cost_ns));
}

}  // namespace reduce_ops

using namespace reduce_ops;

nb::object logsumexp(nb::handle x, nb::handle axis, nb::handle keepdims) {
    const NDArray& a = require_ndarray(x);
    if (a.size() == 0) return ew<Log>(sd::sum(x, axis, nb::none(), keepdims));  // log(0) = -inf
    const nb::object m = ew<FiniteOrZero>(sd::max(x, axis, nb::bool_(true)));
    const nb::object s = sd::sum(ew<ExpDiff>(x, m), axis, nb::none(), keepdims);
    // m again in the result's shape (a max pass is cheap next to the exp pass)
    return ew<LseCombine>(ew<FiniteOrZero>(sd::max(x, axis, keepdims)), s);
}

nb::object softmax(nb::handle x, nb::handle axis) {
    const NDArray& a = require_ndarray(x);
    if (a.size() == 0) return ew<ExpDiff>(x, nb::float_(0.0));  // an empty float64 array
    const nb::object m = sd::max(x, axis, nb::bool_(true));
    const nb::object e = ew<ExpDiff>(x, m);
    return ew<Divide>(e, sd::sum(e, axis, nb::none(), nb::bool_(true)));
}

nb::object log_softmax(nb::handle x, nb::handle axis) {
    const NDArray& a = require_ndarray(x);
    if (a.size() == 0) return ew<ExpDiff>(x, nb::float_(0.0));
    const nb::object m = ew<FiniteOrZero>(sd::max(x, axis, nb::bool_(true)));
    const nb::object logs = ew<Log>(sd::sum(ew<ExpDiff>(x, m), axis, nb::none(), nb::bool_(true)));
    return ew<LogSoftmaxCombine>(x, m, logs);
}

}  // namespace special

}  // namespace sd
