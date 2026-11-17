// Boost.Math where it is no loss (user decision 2026-11-14; docs/design/special.md
// §5.14, bench/special/boost/routing.md).
//
// A region goes to Boost only if Boost keeps the 1e-14 contract there on every
// reference point and is not slower (within 10%, one thread). Everything else
// stays ours: the special values (NaN, infinities, zeros, poles) are decided
// before any Boost call, and a Boost result outside the range it was measured
// accurate in (subnormal, zero, infinite, NaN) falls back to our code.
//
// The headers are third_party/boost_math (tools/vendor_boost_math.py), built in
// standalone mode. The policy turns every Boost error into a returned value
// (NaN, inf or 0) and keeps double arithmetic in double. One error is not a value:
// an evaluation error, which Boost raises when a series or a root search reaches its
// iteration limit (1e6 terms, 200 steps). Ignored, it would hand back the partial sum or
// the last iterate; here it throws, and call() below turns that into NaN (a bound that is
// reached gives NaN, docs/design/stats.md 1.3).
#pragma once

#include <boost/math/policies/policy.hpp>

namespace sd::special::bst {
namespace bp = boost::math::policies;
using Policy = bp::policy<bp::domain_error<bp::ignore_error>, bp::pole_error<bp::ignore_error>,
                          bp::overflow_error<bp::ignore_error>, bp::underflow_error<bp::ignore_error>,
                          bp::denorm_error<bp::ignore_error>, bp::evaluation_error<bp::user_error>,
                          bp::rounding_error<bp::ignore_error>, bp::indeterminate_result_error<bp::ignore_error>,
                          bp::promote_float<false>, bp::promote_double<false>>;
struct NotEvaluated {};  // thrown for Boost's evaluation error
}  // namespace sd::special::bst

namespace boost::math::policies {
// Boost's hook for evaluation_error<user_error> (declared in its error_handling.hpp).
template <class T>
T user_evaluation_error(const char*, const char*, const T&) {
    throw sd::special::bst::NotEvaluated{};
}
}  // namespace boost::math::policies

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/expint.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/special_functions/zeta.hpp>

#include <cmath>
#include <limits>

namespace sd::special::bst {

namespace bm = boost::math;
inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// A Boost result we keep: finite and normal (or an exact zero where the caller
// says zero is possible). Anything else goes back to our code.
inline bool normal(double v) { return std::fabs(v) >= 0x1p-1022 && std::fabs(v) <= 0x1.fffffffffffffp+1023; }

// Each call: the value, or NaN if Boost threw anyway (it should not under this policy).
template <typename F>
inline double call(F f) {
    try {
        return f();
    } catch (...) {
        return kNaN;
    }
}

inline double tgamma(double x) { return call([&] { return bm::tgamma(x, Policy()); }); }
inline double digamma(double x) { return call([&] { return bm::digamma(x, Policy()); }); }
inline double beta(double a, double b) { return call([&] { return bm::beta(a, b, Policy()); }); }
inline double zeta(double x) { return call([&] { return bm::zeta(x, Policy()); }); }
inline double gamma_p_inv(double a, double p) { return call([&] { return bm::gamma_p_inv(a, p, Policy()); }); }
inline double gamma_q_inv(double a, double q) { return call([&] { return bm::gamma_q_inv(a, q, Policy()); }); }
inline double ibeta(double a, double b, double x) { return call([&] { return bm::ibeta(a, b, x, Policy()); }); }
inline double ibetac(double a, double b, double x) { return call([&] { return bm::ibetac(a, b, x, Policy()); }); }
inline double ibeta_inv(double a, double b, double p) { return call([&] { return bm::ibeta_inv(a, b, p, Policy()); }); }
inline double ibetac_inv(double a, double b, double q) { return call([&] { return bm::ibetac_inv(a, b, q, Policy()); }); }
inline double expint(double x) { return call([&] { return bm::expint(x, Policy()); }); }
inline double i(double v, double x) { return call([&] { return bm::cyl_bessel_i(v, x, Policy()); }); }
inline double k(double v, double x) { return call([&] { return bm::cyl_bessel_k(v, x, Policy()); }); }
inline double w0(double x) { return call([&] { return bm::lambert_w0(x, Policy()); }); }
inline double wm1(double x) { return call([&] { return bm::lambert_wm1(x, Policy()); }); }
inline double hyp1f1(double a, double b, double x) {
    return call([&] { return bm::hypergeometric_1F1(a, b, x, Policy()); });
}

}  // namespace sd::special::bst
