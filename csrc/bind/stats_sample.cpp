// The samples of the distributions (T7, docs/design/stats.md §4): "<distribution>_sample"
// in stardust._core.stats, (k0, k1, pos, shape, params...) -> float64 array of `shape`.
// The Python side (stardust.stats._distribution) checks the arguments, broadcasts the
// parameters to their common shape P (the trailing axes of `shape`: shape = S + P) and
// reserves the stream range; element i takes the parameters at i mod |P| and the stream
// element pos + i (csrc/stats/sample.hpp).
#include "bind/stats_bind.hpp"

#include "array/convert.hpp"
#include "array/ops/pieces.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "random/philox.hpp"
#include "random/random.hpp"
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
#include "stats/sample.hpp"
#include "stats/sample_dist.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace nb = nanobind;

namespace sd {

namespace {

// The byte offset of C-order element i of an array (a broadcast view may have zero strides).
std::int64_t offset_of(const NDArray& a, std::int64_t i) {
    std::int64_t off = 0;
    for (int d = a.ndim - 1; d >= 0; --d) {
        off += (i % a.shape[d]) * a.strides[d];
        i /= a.shape[d];
    }
    return off;
}

// Work per element for the pool's threshold (in "elements"): a sample is a few to a few
// hundred ns.
template <class D>
constexpr std::int64_t sample_work() {
    if constexpr (requires { stats::smp::Custom<D>::kWork; }) return stats::smp::Custom<D>::kWork;
    else return 64;
}

// The parameters a sample is drawn for: the domain, and a limit at an infinite parameter
// (stats/engine.hpp: the t at df = inf), which the distribution's sampler draws itself.
template <class D, class... P>
bool in_domain(P... p) {
    if constexpr (stats::has_limit<D, P...>) {
        if (D::at_limit(p...)) return true;
    }
    return D::valid(p...);
}

template <class D, class Seq = std::make_index_sequence<D::params.size()>>
struct Sampler;

template <class D, std::size_t... I>
struct Sampler<D, std::index_sequence<I...>> {
    template <std::size_t>
    using Arr = const NDArray&;

    static nb::object call(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, nb::handle shape_obj, Arr<I>... p) {
        const std::array<const NDArray*, sizeof...(I)> ps{&p...};
        for (const NDArray* a : ps)
            if (a->dtype != DType::Float64) throw nb::type_error("sample: the parameters must be float64 arrays");
        NDArray out = new_array(DType::Float64, parse_shape_arg(shape_obj).span());
        double* o = reinterpret_cast<double*>(out.data);
        const std::int64_t n = out.size();
        const std::int64_t np = ps[0]->size();  // |P|, the parameters' common size
        if (n == 0) return wrap(std::move(out));
        if (np == 0 || n % np != 0) throw nb::value_error("sample: the parameters do not fit the shape");
        bool scalar = true;  // one parameter set for every element (the usual call)
        for (const NDArray* a : ps)
            for (int d = 0; d < a->ndim; ++d) scalar = scalar && (a->strides[d] == 0 || a->shape[d] == 1);
        auto param = [&](const NDArray& a, std::int64_t j) {
            return *reinterpret_cast<const double*>(a.data + (scalar ? 0 : offset_of(a, j)));
        };
        using C = stats::smp::Custom<D>;
        constexpr bool has_prep = requires(double x) { C::prep((static_cast<void>(I), x)...); };
        const random::Key key{k0, k1};
        {
            nb::gil_scoped_release release;
            for_pieces(n, kParallelTask, n * sample_work<D>(), [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
                random::Stream s(key);
                auto at = [&](std::int64_t i) { return pos + static_cast<std::uint64_t>(i); };
                if (scalar) {  // one parameter set: checked (and prepared) once
                    const std::array<double, sizeof...(I)> v{param(p, 0)...};
                    if (!in_domain<D>(v[I]...)) {
                        for (std::int64_t i = i0; i < i1; ++i) o[i] = stats::kNaN;
                        return;
                    }
                    if constexpr (has_prep) {
                        const auto st = C::prep(v[I]...);
                        bool extra = false;  // the method draws extra words (for this parameter set)
                        if constexpr (requires { C::kExtra; }) extra = true;
                        if constexpr (requires { C::extra(st); }) extra = C::extra(st);
                        if (extra) {
                            // their first extra blocks four elements at a time
                            std::array<std::uint64_t, 16> blk;
                            std::uint64_t es[4], fw[4];
                            std::int64_t at4 = 0;
                            int have = 0;
                            stats::smp::each_first(key, pos, i0, i1, [&](std::int64_t i, std::uint64_t first) {
                                if (have == 0) at4 = i;
                                es[have] = at(i), fw[have] = first;
                                if (++have < 4) return;
                                random::philox4(es, key, blk, 1);
                                for (int q = 0; q < 4; ++q) {
                                    stats::smp::Words w(s, es[q], fw[q], blk.data() + 4 * q);
                                    o[at4 + q] = C::draw(w, st);
                                }
                                have = 0;
                            });
                            for (int q = 0; q < have; ++q) {
                                stats::smp::Words w(s, es[q], fw[q]);
                                o[at4 + q] = C::draw(w, st);
                            }
                        } else {
                            stats::smp::each_first(key, pos, i0, i1, [&](std::int64_t i, std::uint64_t first) {
                                stats::smp::Words w(s, at(i), first);
                                o[i] = C::draw(w, st);
                            });
                        }
                    } else {
                        stats::smp::each_first(key, pos, i0, i1, [&](std::int64_t i, std::uint64_t first) {
                            stats::smp::Words w(s, at(i), first);
                            o[i] = stats::smp::sample<D>(w, v[I]...);
                        });
                    }
                    return;
                }
                stats::smp::each_first(key, pos, i0, i1, [&](std::int64_t i, std::uint64_t first) {
                    const std::int64_t j = i % np;
                    const std::array<double, sizeof...(I)> v{param(p, j)...};
                    if (!in_domain<D>(v[I]...)) {
                        o[i] = stats::kNaN;
                        return;
                    }
                    stats::smp::Words w(s, at(i), first);
                    if constexpr (has_prep) o[i] = C::draw(w, C::prep(v[I]...));
                    else o[i] = stats::smp::sample<D>(w, v[I]...);
                });
            });
        }
        return wrap(std::move(out));
    }

    // One sample for scalar parameters (a Python float: no arrays), element pos of the
    // stream; the same value as element 0 of call() at that position.
    template <std::size_t>
    using Num = double;
    static double one(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, Num<I>... p) {
        if (!in_domain<D>(p...)) return stats::kNaN;
        random::Stream s({k0, k1});
        stats::smp::Words w(s, pos);
        using C = stats::smp::Custom<D>;
        if constexpr (requires { C::prep(p...); }) return C::draw(w, C::prep(p...));
        else return stats::smp::sample<D>(w, p...);
    }
    // ... with the element reserved from the generator's state here (sample()'s scalar call:
    // no Python lock, no key tuple; the element is taken whatever the parameters, as before)
    // The parameters as the distribution's tuple of Python floats (its _scalar): unpacked here,
    // a third of the call's cost when Python spread them as arguments.
    static double one_state(random::State& st, nb::handle params) {
        PyObject* t = params.ptr();
        if (!PyTuple_Check(t) || PyTuple_GET_SIZE(t) != static_cast<Py_ssize_t>(sizeof...(I)))
            throw nb::type_error("sample: the parameters must be a tuple of floats");
        double v[sizeof...(I) + 1];
        for (std::size_t q = 0; q < sizeof...(I); ++q) {
            PyObject* o = PyTuple_GET_ITEM(t, static_cast<Py_ssize_t>(q));
            v[q] = PyFloat_CheckExact(o) ? PyFloat_AS_DOUBLE(o) : PyFloat_AsDouble(o);
            if (v[q] == -1.0 && PyErr_Occurred()) throw nb::python_error();
        }
        const random::State::Taken r = st.reserve(1);
        return one(r.k0, r.k1, r.pos, v[I]...);
    }
};

template <class... D>
void def_samples(nb::module_& m) {
    (m.def((std::string(D::name) + "_sample").c_str(), &Sampler<D>::call), ...);
    (m.def((std::string(D::name) + "_sample1").c_str(), &Sampler<D>::one), ...);
    (m.def((std::string(D::name) + "_sample1s").c_str(), &Sampler<D>::one_state), ...);
}

}  // namespace

void bind_stats_sample(nb::module_& m) {
    using namespace stats;
    def_samples<
        Normal, StudentT, ChiSquared, F, Gamma, Beta, Exponential, Uniform, Laplace, Cauchy, LogNormal, Weibull,
        Logistic, Gumbel, GEV, Pareto, Rayleigh, InverseGamma, NoncentralChiSquared, NoncentralT, NoncentralF, Binomial,
        Poisson, Geometric, NegativeBinomial, Hypergeometric, Bernoulli, DiscreteUniform, BetaBinomial, LogSeries,
        Zipf>(m);
}

}  // namespace sd
