// The statistics area: the submodule stardust._core.stats (docs/design/stats.md
// §1.4, §3). Every distribution method is an elementwise function
// "<distribution>_<method>" with inputs (params..., x) or (params...), bound
// through the n-ary dispatcher like sd.special (bind/nary_bind.hpp); the
// distribution classes of stardust.stats call them.
#include "bind/stats_bind.hpp"

#include "bind/nary_bind.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/cauchy.hpp"
#include "stats/dist/discrete_basic.hpp"
#include "stats/dist/discrete_more.hpp"
#include "stats/dist/exponential.hpp"
#include "stats/dist/extreme.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/laplace.hpp"
#include "stats/dist/lifetime.hpp"
#include "stats/dist/lognormal.hpp"
#include "stats/dist/noncentral.hpp"
#include "stats/dist/normal.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/dist/uniform.hpp"
#include "stats/dist/weibull.hpp"
#include "stats/engine.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace nb = nanobind;

namespace sd {

namespace {

// One method bound as nary_bind::def does, with a shorter way for the commonest call: every
// input a Python float and no out= (a distribution object's scalar call). The op's float64
// loop then runs on one element from the stack, the loop run_nary would pick for those inputs
// (the same bits); run_nary's argument handling took most of a scalar call (80 ns of 110).
// Anything else (ints, arrays, out=) is nary_bind's call.
template <class Op>
nb::object call_fast(std::span<const nb::handle> objs, nb::handle out) {
    if (out.is_none()) {
        double v[Op::nin + 1];
        bool floats = true;
        for (int k = 0; k < Op::nin; ++k) {
            PyObject* o = objs[static_cast<std::size_t>(k)].ptr();
            floats = floats && PyFloat_CheckExact(o);
            if (!floats) break;
            v[k] = PyFloat_AS_DOUBLE(o);
        }
        const Loop loop = nary_bind::table_of<Op>()[static_cast<std::size_t>(DType::Float64)].loop;
        if (floats && loop != nullptr) {
            std::byte* args[Op::nin + 1];
            std::int64_t st[Op::nin + 1];
            for (int k = 0; k <= Op::nin; ++k) {
                args[k] = reinterpret_cast<std::byte*>(&v[k]);
                st[k] = sizeof(double);
            }
            loop(args, 1, st, nullptr);
            return nb::steal(PyFloat_FromDouble(v[Op::nin]));
        }
    }
    return nary_bind::call<Op>(objs, out);
}
template <class Op>
void def_fast(nb::module_& m) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        m.def(
            Op::name,
            [](nary_bind::handle_t<I>... xs, nb::handle out) {
                return call_named(Op::name, [&] {
                    const nb::handle objs[] = {xs...};
                    return call_fast<Op>(objs, out);
                });
            },
            nb::sig(nary_bind::signature<Op>()), ((void)I, nb::arg().none())..., nb::kw_only(),
            nb::arg("out").none() = nb::none()
        );
    }(std::make_index_sequence<Op::nin>{});
}

template <class D, class... M>
void def_methods(nb::module_& m) {
    (def_fast<stats::DistOp<D, M>>(m), ...);
}

// Every method of a continuous distribution (T9; sample, T7, is bound in bind/stats_sample.cpp).
template <class D>
void def_continuous(nb::module_& m) {
    using namespace stats::method;
    def_methods<
        D, Pdf, Logpdf, Cdf, Logcdf, Ccdf, Logccdf, Icdf, Iccdf, Ilogcdf, Ilogccdf, Mean, Variance, StandardDeviation,
        Skewness, Kurtosis, Median, Mode, Entropy, SupportLo, SupportHi>(m);
}

// Every method of a discrete distribution: pmf and logpmf in place of pdf and logpdf (T25).
template <class D>
void def_discrete(nb::module_& m) {
    using namespace stats::method;
    def_methods<
        D, Pmf, Logpmf, Cdf, Logcdf, Ccdf, Logccdf, Icdf, Iccdf, Ilogcdf, Ilogccdf, Mean, Variance, StandardDeviation,
        Skewness, Kurtosis, Median, Mode, Entropy, SupportLo, SupportHi>(m);
}

}  // namespace

void bind_stats(nb::module_& core) {
    nb::module_ m = core.def_submodule("stats", "stardust statistics native core");
    // T4: the continuous basic 12 (docs/api/stats.md §2.1)
    def_continuous<stats::Normal>(m);
    def_continuous<stats::StudentT>(m);
    def_continuous<stats::ChiSquared>(m);
    def_continuous<stats::F>(m);
    def_continuous<stats::Gamma>(m);
    def_continuous<stats::Beta>(m);
    def_continuous<stats::Exponential>(m);
    def_continuous<stats::Uniform>(m);
    def_continuous<stats::Laplace>(m);
    def_continuous<stats::Cauchy>(m);
    def_continuous<stats::LogNormal>(m);
    def_continuous<stats::Weibull>(m);
    // T6: extreme value and lifetime 6 (docs/api/stats.md §2.1)
    def_continuous<stats::Logistic>(m);
    def_continuous<stats::Gumbel>(m);
    def_continuous<stats::GEV>(m);
    def_continuous<stats::Pareto>(m);
    def_continuous<stats::Rayleigh>(m);
    def_continuous<stats::InverseGamma>(m);
    // T6: noncentral 3
    def_continuous<stats::NoncentralChiSquared>(m);
    def_continuous<stats::NoncentralT>(m);
    def_continuous<stats::NoncentralF>(m);
    // T5: the discrete 10 (docs/api/stats.md §2.1)
    def_discrete<stats::Binomial>(m);
    def_discrete<stats::Poisson>(m);
    def_discrete<stats::Geometric>(m);
    def_discrete<stats::NegativeBinomial>(m);
    def_discrete<stats::Hypergeometric>(m);
    def_discrete<stats::Bernoulli>(m);
    def_discrete<stats::DiscreteUniform>(m);
    def_discrete<stats::BetaBinomial>(m);
    def_discrete<stats::LogSeries>(m);
    def_discrete<stats::Zipf>(m);
    bind_stats_sample(m);  // T7
    bind_stats_desc(m);    // T8
    bind_stats_mv(m);      // T9
    bind_stats_mvcdf(m);   // T9
    bind_stats_matrix(m);  // T10
}

}  // namespace sd
