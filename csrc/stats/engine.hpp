// The distribution engine (docs/design/stats.md §3): one elementwise Op per
// (distribution, method), made from a distribution struct by a template, and
// bound through the same n-ary dispatcher as sd.special (bind/nary_bind.hpp).
//
// A distribution is a struct of static functions:
//
//     struct Normal {
//         static constexpr char name[] = "normal";
//         static constexpr std::array<const char*, 2> params{"mu", "sigma"};
//         static bool valid(double mu, double sigma);       // the parameter domain
//         static double cdf(double x, double mu, double sigma);
//         ...                                               // one per method below
//     };
//
// DistOp<D, M> is the Op of method M: inputs (params..., x) for the methods that
// take an input, (params...) for the others; the binding names it
// "<distribution>_<method>" in stardust._core.stats. It returns NaN where the
// parameters are outside their domain (T8) or the input is NaN, so a method is
// only ever called with valid parameters and a non-NaN input.
#pragma once

#include "accelerate/vforce.hpp"
#include "array/kernels/elementwise.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "stats/cost_table.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace sd::stats {

using special::kInf;
using special::kNaN;
using special::Real;

// ---- chunk kernels (the fast paths; docs/design/stats.md §3.4) -------------------------------
//
// A distribution may give a method a chunk kernel beside its scalar definition:
//
//     static void cdf_chunk(const Chunk<2>& c, double* out);
//
// c.x and c.p[k] are contiguous arrays of c.n <= kChunkSize elements (a broadcast scalar
// parameter fills its array; c.same[k] says so, for constants worth computing once). The
// kernel writes every out[i]. It only ever sees valid parameters (not at a limit) and a
// non-NaN x: the engine gives the other elements a good element's values in the kernel's
// copy of the chunk and puts their own values (NaN, a limit's) afterwards. A kernel is the
// method's definition for every layout: a single element is a chunk of n = 1, and its
// building blocks (ck::exp: array/kernels/vmath.hpp; log, pow, ...: accelerate/vforce.hpp; elementwise loops)
// do not depend on an element's position, so f(x)[i] == f(x[i]) on any thread count.
//
// The usual shape: the result in plain double for the whole chunk (vForce), a bound on its
// relative error per element from the formula's condition, and the scalar definition (the
// double-double one) for the elements whose bound passes kFastBudget or that are special
// (ends of the support, infinities). The contract is 1e-14 (T10); the fast path keeps to
// 2e-15 (user decision 2026-11-16: speed inside the contract).
inline constexpr int kChunkSize = static_cast<int>(vforce_detail::kChunk);  // 1024
inline constexpr double kUlp = 0x1p-53;            // unit roundoff
inline constexpr double kFastBudget = 16.0;        // in kUlp: 1.8e-15

template <std::size_t NP>
struct Chunk {
    int n;
    const double* x;
    std::array<const double*, NP> p;
    std::array<bool, NP> same;
};

// "<a>_<b>" at compile time.
template <std::size_t A, std::size_t B>
constexpr std::array<char, A + B> join_name(const char (&a)[A], const char (&b)[B]) {
    std::array<char, A + B> r{};  // (A - 1) + 1 + (B - 1) + 1 characters
    std::size_t k = 0;
    for (std::size_t i = 0; i + 1 < A; ++i) r[k++] = a[i];
    r[k++] = '_';
    for (std::size_t i = 0; i < B; ++i) r[k++] = b[i];
    return r;
}

// ---- methods ------------------------------------------------------------------
//
// Each names itself, says whether it takes an input (and the input's name in
// messages), and calls the distribution's function, with a default where the
// method follows from others.

namespace method {

#define SD_STATS_INPUT_METHOD(Tag, fname, argname)                                                   \
    struct Tag {                                                                                     \
        static constexpr char name[] = #fname;                                                       \
        static constexpr bool input = true;                                                          \
        static constexpr const char* arg = argname;                                                  \
        template <class D, class... P>                                                               \
        static double call(double x, P... p) {                                                       \
            return D::fname(x, p...);                                                                \
        }                                                                                            \
        template <class D, class C>                                                                  \
        static constexpr bool chunked = requires(const C& c, double* o) { D::fname##_chunk(c, o); }; \
        template <class D, class C>                                                                  \
        static void run_chunk(const C& c, double* o) {                                               \
            if constexpr (chunked<D, C>) D::fname##_chunk(c, o);                                     \
        }                                                                                            \
    };

SD_STATS_INPUT_METHOD(Pdf, pdf, "x")
SD_STATS_INPUT_METHOD(Logpdf, logpdf, "x")
SD_STATS_INPUT_METHOD(Pmf, pmf, "k")  // discrete distributions (T9, T25)
SD_STATS_INPUT_METHOD(Logpmf, logpmf, "k")
SD_STATS_INPUT_METHOD(Cdf, cdf, "x")
SD_STATS_INPUT_METHOD(Logcdf, logcdf, "x")
SD_STATS_INPUT_METHOD(Ccdf, ccdf, "x")
SD_STATS_INPUT_METHOD(Logccdf, logccdf, "x")
SD_STATS_INPUT_METHOD(Icdf, icdf, "p")
SD_STATS_INPUT_METHOD(Iccdf, iccdf, "p")
SD_STATS_INPUT_METHOD(Ilogcdf, ilogcdf, "logp")
SD_STATS_INPUT_METHOD(Ilogccdf, ilogccdf, "logp")
#undef SD_STATS_INPUT_METHOD

#define SD_STATS_PARAM_METHOD(Tag, fname)      \
    struct Tag {                               \
        static constexpr char name[] = #fname; \
        static constexpr bool input = false;   \
        template <class D, class... P>         \
        static double call(P... p) {           \
            return D::fname(p...);             \
        }                                      \
    };

SD_STATS_PARAM_METHOD(Mean, mean)
SD_STATS_PARAM_METHOD(Variance, variance)
SD_STATS_PARAM_METHOD(Skewness, skewness)
SD_STATS_PARAM_METHOD(Kurtosis, kurtosis)
SD_STATS_PARAM_METHOD(Mode, mode)
SD_STATS_PARAM_METHOD(Entropy, entropy)
SD_STATS_PARAM_METHOD(SupportLo, support_lo)
SD_STATS_PARAM_METHOD(SupportHi, support_hi)
#undef SD_STATS_PARAM_METHOD

// The standard deviation from the variance (sqrt halves a relative error).
struct StandardDeviation {
    static constexpr char name[] = "standard_deviation";
    static constexpr bool input = false;
    template <class D, class... P>
    static double call(P... p) {
        if constexpr (requires { D::standard_deviation(p...); }) return D::standard_deviation(p...);
        else return std::sqrt(D::variance(p...));
    }
};

// The median: icdf(1/2) unless the distribution has a closed form.
struct Median {
    static constexpr char name[] = "median";
    static constexpr bool input = false;
    template <class D, class... P>
    static double call(P... p) {
        if constexpr (requires { D::median(p...); }) return D::median(p...);
        else return D::icdf(0.5, p...);
    }
};

}  // namespace method

// ---- the Op -------------------------------------------------------------------

// Per-element cost in ns for the parallel split (special/cost.hpp): the measured
// one (stats/cost_table.hpp, from the official speed tables), else the distribution's
// own estimate when it gives one, else a middling default. A cost several times too
// high cuts a chunk kernel's work into tasks of a few dozen elements, and ten threads
// then run slower than one.
//
// The table has the methods with an input. Of the parameter-only ones, those that sum over the
// support, search it or integrate (the entropy and the median of the discrete and the
// noncentral distributions, the noncentral mode) take microseconds per element and more: they
// count 1 us, a floor. The discrete mode and moments and the gamma family's entropy (digamma,
// log-gamma) count as the heaviest measured functions, kMaxWeight adds. At the default 20 ns a
// call of a few hundred such elements ran on one thread holding the GIL (200 negative
// binomial entropies took 36 s on any thread count). Only the split depends on these values,
// never a result.
template <class D>
constexpr bool is_discrete = requires { &D::pmf; };
constexpr bool name_is(const char* s, const char* t) {
    while (*s && *s == *t) ++s, ++t;
    return *s == *t;
}
constexpr bool name_starts(const char* s, const char* prefix) {
    while (*prefix && *s == *prefix) ++s, ++prefix;
    return *prefix == 0;
}
inline constexpr double kIterativeNs = 1000.0;
inline constexpr double kHeavyNs = static_cast<double>(special::kMaxWeight) * special::kAddNs;
template <class D, class M>
constexpr double param_cost() {
    using namespace method;
    constexpr bool noncentral = name_starts(D::name, "noncentral");
    constexpr bool gamma_family =
        name_is(D::name, "gamma") || name_is(D::name, "chisquared") || name_is(D::name, "inversegamma");
    constexpr bool moment = std::is_same_v<M, Mean> || std::is_same_v<M, Variance> ||
                            std::is_same_v<M, StandardDeviation> || std::is_same_v<M, Skewness> ||
                            std::is_same_v<M, Kurtosis>;
    if constexpr (std::is_same_v<M, Entropy> || std::is_same_v<M, Median>) {
        if constexpr (is_discrete<D> || noncentral) return kIterativeNs;
        else if constexpr (gamma_family && std::is_same_v<M, Entropy>) return kHeavyNs;
        else return 0.0;
    } else if constexpr (std::is_same_v<M, Mode>) {
        return noncentral ? kIterativeNs : (is_discrete<D> ? kHeavyNs : 0.0);
    } else if constexpr (moment) {
        return is_discrete<D> ? kHeavyNs : 0.0;
    } else {
        return 0.0;
    }
}
template <class D, class M>
constexpr double cost_of() {
    if (const double ns = cost_table::find(D::name, M::name); ns > 0.0) return ns;
    if constexpr (requires { D::template cost<M>(); }) return D::template cost<M>();
    else if constexpr (!M::input && param_cost<D, M>() > 0.0) return param_cost<D, M>();
    else return 20.0;
}

// Whether the per-thread memo of values at integers (stats/discrete.hpp, KMemo) will answer
// most elements of a call with these scalar parameters: the central 99.8% of the
// distribution spans at most kMemoSpan integers (a quarter of the memo's 1024 slots, so keys
// rarely collide). The answer for the last parameters is kept per thread: a distribution
// object calls with the same ones again and again.
inline constexpr double kMemoSpan = 256.0;
template <class D, class... P>
bool memo_narrow(P... p) {
    thread_local double last[sizeof...(P)];
    thread_local bool have = false, narrow = false;
    const double cur[] = {static_cast<double>(p)...};
    if (have && std::memcmp(last, cur, sizeof cur) == 0) return narrow;
    bool r = false;
    if (D::valid(p...)) {
        const double lo = D::icdf(1e-3, p...), hi = D::icdf(1.0 - 1e-3, p...);
        r = std::isfinite(lo) && std::isfinite(hi) && hi - lo <= kMemoSpan;
    }
    std::memcpy(last, cur, sizeof cur);
    have = true;
    narrow = r;
    return r;
}

// A probability or a density (pdf, pmf, cdf, ccdf) is never negative, so never -0.0 either:
// a formula may round to it (-(tiny) * (tiny), a product with a negative factor that
// underflows). v + 0.0 turns -0.0 into +0.0 and changes no other value. The log methods keep
// their sign: log of a probability that rounds to 1 is -0.0, a limit from below.
template <class M>
inline constexpr bool kNonNegative = std::is_same_v<M, method::Pdf> || std::is_same_v<M, method::Pmf> ||
                                     std::is_same_v<M, method::Cdf> || std::is_same_v<M, method::Ccdf>;

template <class D, class M, class = std::make_index_sequence<D::params.size()>>
struct DistOp;

// ---- limits at a parameter (docs/api/stats.md §2, user decisions 2026-11-19) -----------------
//
// A distribution may be another one at some parameter values: the limit of an infinite
// parameter that valid() leaves out (the t at df = inf is the normal), a point mass (the
// negative binomial's n = 0), or, inside valid(), a shape too small for the distribution's
// own functions (Beta's and BetaBinomial's below 2^-1021: two atoms, T50; the engine asks
// at_limit first, so valid() need not leave them out). It then says where, and gives every
// method's value there,
//
//     static bool at_limit(double df);                           // inf, a point mass, a tiny shape
//     template <class M> static double limit_in(double x, double df);   // methods with an input
//     template <class M> static double limit_par(double df);            // the others
//
// usually from the other distribution's own Op (method_of), or from a scaled one (scaled_of:
// X = Y / c). The engine asks at_limit before valid, in the scalar and the chunk paths alike.
template <class D, class... P>
constexpr bool has_limit = requires(P... p) { D::at_limit(p...); };

template <std::size_t>
using Dbl = double;

// The loop of a method with a chunk kernel. args: the parameters, x, out.
//
// Each element is one of: good (valid parameters, not at a limit, x not NaN: the kernel's
// value), at a limit (the limit's value, NaN for a NaN x) or bad (NaN). The kernel only sees
// good elements: a bad or limit element gets the parameters and x of a good one in the
// kernel's copy of the chunk, and its own value afterwards. A kernel falls back to the
// scalar definition for the elements its fast path leaves, and the scalar definition
// assumes valid() (a Pareto with scale <= 0 never ends there); an element's kernel value
// depends on that element alone, so the good elements keep their bits.
//
// The common case takes a shorter way: every element good (scalar parameters checked once per
// call, then only a branch-free scan of x for NaNs; parameter arrays checked by one branch-free
// pass), the kernel reads the operands in place and writes the output in place when they are
// contiguous and the output does not overlap an input. A broadcast scalar parameter fills its
// chunk buffer once per call. The kernel computes the same values either way (elementwise).
template <class D, std::size_t NP, std::size_t... I>
inline bool all_good_params(const Chunk<NP>& c, std::index_sequence<I...>) {
    bool ok = true;
    for (int i = 0; i < c.n; ++i) {
        bool limit = false;
        if constexpr (has_limit<D, Dbl<I>...>) limit = D::at_limit(c.p[I][i]...);
        ok &= !limit & D::valid(c.p[I][i]...) & (c.x[i] == c.x[i]);
    }
    return ok;
}
inline bool no_nan(const double* x, int n) {
    bool ok = true;
    for (int i = 0; i < n; ++i) ok &= x[i] == x[i];
    return ok;
}
// Whether [a, a + n) and [b, b + n) (doubles) are apart.
inline bool apart(const double* a, const double* b, int n) { return a + n <= b || b + n <= a; }

template <class D, class M, std::size_t... I>
void chunk_loop_impl(std::byte* const* args, std::int64_t n, const std::int64_t* s, std::index_sequence<I...>) {
    constexpr std::size_t NP = sizeof...(I);
    constexpr bool kLimits = has_limit<D, Dbl<I>...>;
    using namespace vforce_detail;
    enum : unsigned char { kGood, kBad, kLimit };
    double xb[kChunk], ob[kChunk], pb[NP][kChunk];
    unsigned char state[kChunk];
    [[maybe_unused]] double lim[kLimits ? kChunk : 1];
    const bool all_same = ((s[I] == 0) && ...);
    const int first = static_cast<int>(std::min<std::int64_t>(kChunk, n));
    // a broadcast scalar parameter: its buffer filled once (the slow path below never writes it)
    ((s[I] == 0 ? (void)std::fill(pb[I], pb[I] + first, *reinterpret_cast<const double*>(args[I])) : void()), ...);
    bool same_good = false;  // scalar parameters, valid and not at a limit
    if (all_same) {
        bool limit = false;
        if constexpr (kLimits) limit = D::at_limit(pb[I][0]...);
        same_good = !limit && D::valid(pb[I][0]...);
    }
    const bool out_direct = s[NP + 1] == sizeof(double);
    for (std::int64_t off = 0; off < n; off += kChunk) {
        const int m = static_cast<int>(std::min<std::int64_t>(kChunk, n - off));
        Chunk<NP> c{
            m,
            gather(args[NP] + off * s[NP], s[NP], m, xb),
            {(s[I] == 0 ? static_cast<const double*>(pb[I]) : gather(args[I] + off * s[I], s[I], m, pb[I]))...},
            {(s[I] == 0)...}
        };
        // the NaNs, and for a probability or a density + 0.0 (kNonNegative) in the same pass
        constexpr double zero = kNonNegative<M> ? 0.0 : -0.0;  // v + -0.0 is v for every v
        std::byte* o = args[NP + 1] + off * s[NP + 1];
        const bool fast = all_same ? same_good && no_nan(c.x, m) : all_good_params<D>(c, std::index_sequence<I...>{});
        if (fast) {
            double* dst = ob;
            if (out_direct) {
                double* od = reinterpret_cast<double*>(o);
                if (apart(od, c.x, m) && ((s[I] == 0 || apart(od, c.p[I], m)) && ...)) dst = od;
            }
            M::template run_chunk<D, Chunk<NP>>(c, dst);
            if constexpr (kNonNegative<M>)
                for (int i = 0; i < m; ++i) dst[i] += zero;
            if (dst == ob) {
                if (out_direct) std::memcpy(o, ob, static_cast<std::size_t>(m) * sizeof(double));
                else scatter(ob, o, s[NP + 1], m);
            }
            continue;
        }
        int good = -1, bad = 0;
        auto classify = [&](int i, bool limit, bool valid) {
            const bool nan = c.x[i] != c.x[i];
            state[i] = limit ? (nan ? kBad : kLimit) : (valid && !nan ? kGood : kBad);
            if (state[i] == kGood) good = good < 0 ? i : good;
            else ++bad;
        };
        if (all_same) {
            bool limit = false;
            if constexpr (kLimits) limit = D::at_limit(c.p[I][0]...);
            const bool valid = !limit && D::valid(c.p[I][0]...);
            for (int i = 0; i < m; ++i) classify(i, limit, valid);
        } else {
            for (int i = 0; i < m; ++i) {
                bool limit = false;
                if constexpr (kLimits) limit = D::at_limit(c.p[I][i]...);
                classify(i, limit, !limit && D::valid(c.p[I][i]...));
            }
        }
        if constexpr (kLimits) {  // the limits: another distribution's values (before the copy below)
            if (bad > 0)
                for (int i = 0; i < m; ++i)
                    if (state[i] == kLimit) lim[i] = D::template limit_in<M>(c.x[i], c.p[I][i]...) + zero;
        }
        if (good >= 0) {
            if (bad > 0) {  // the kernel's copy: every element a good one
                auto own = [&](const double*& p, double* buf) {
                    if (p != buf) std::memcpy(buf, p, static_cast<std::size_t>(m) * sizeof(double));
                    p = buf;
                };
                own(c.x, xb);
                ((c.same[I] ? void() : own(c.p[I], pb[I])), ...);
                for (int i = 0; i < m; ++i) {
                    if (state[i] == kGood) continue;
                    xb[i] = xb[good];
                    ((c.same[I] ? void() : void(pb[I][i] = pb[I][good])), ...);
                }
            }
            M::template run_chunk<D, Chunk<NP>>(c, ob);
        }
        for (int i = 0; i < m; ++i) {
            if (state[i] == kGood) ob[i] += zero;
            else if constexpr (kLimits) ob[i] = state[i] == kLimit ? lim[i] : kNaN;
            else ob[i] = kNaN;
        }
        if (s[NP + 1] == sizeof(double)) std::memcpy(o, ob, static_cast<std::size_t>(m) * sizeof(double));
        else scatter(ob, o, s[NP + 1], m);
    }
}
template <class D, class M>
void chunk_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    chunk_loop_impl<D, M>(args, n, s, std::make_index_sequence<D::params.size()>{});
}
template <class D, class M>
inline constexpr LoopTable chunk_table = {
    LoopEntry{}, LoopEntry{}, LoopEntry{&chunk_loop<D, M>, DType::Float64}, LoopEntry{}
};

// Method M of distribution T at one point, with the bits T's own method gives a Python
// float: through its chunk kernel (a chunk of one) when it has one, else its Op.
template <class T, class M, class... A>
double method_of(double x, A... a)
    requires M::input
{
    constexpr std::size_t NP = sizeof...(A);
    if constexpr (M::template chunked<T, Chunk<NP>>) {
        double pv[NP] = {a...};
        double xv = x, out = 0.0;
        std::byte* args[NP + 2];
        std::int64_t st[NP + 2];
        for (std::size_t k = 0; k < NP; ++k) args[k] = reinterpret_cast<std::byte*>(&pv[k]);
        args[NP] = reinterpret_cast<std::byte*>(&xv);
        args[NP + 1] = reinterpret_cast<std::byte*>(&out);
        for (std::size_t k = 0; k < NP + 2; ++k) st[k] = sizeof(double);
        chunk_loop<T, M>(args, 1, st, nullptr);
        return out;
    } else {
        return DistOp<T, M>::f(a..., x);
    }
}
template <class T, class M, class... A>
double method_of(A... a)
    requires(!M::input)
{
    return DistOp<T, M>::f(a...);
}

// Method M of X = Y / c (c > 0 finite), Y of distribution T with parameters a.
template <class T, class M, class... A>
double scaled_of(double c, double x, A... a)
    requires M::input
{
    if constexpr (std::is_same_v<M, method::Pdf>) return c * method_of<T, M>(c * x, a...);
    else if constexpr (std::is_same_v<M, method::Logpdf>) return std::log(c) + method_of<T, M>(c * x, a...);
    else if constexpr (std::is_same_v<M, method::Cdf> || std::is_same_v<M, method::Ccdf> ||
                       std::is_same_v<M, method::Logcdf> || std::is_same_v<M, method::Logccdf>)
        return method_of<T, M>(c * x, a...);
    else return method_of<T, M>(x, a...) / c;  // the four inverses
}
template <class T, class M, class... A>
double scaled_of(double c, A... a)
    requires(!M::input)
{
    if constexpr (std::is_same_v<M, method::Variance>) return method_of<T, M>(a...) / c / c;
    else if constexpr (std::is_same_v<M, method::Skewness> || std::is_same_v<M, method::Kurtosis>)
        return method_of<T, M>(a...);
    else if constexpr (std::is_same_v<M, method::Entropy>) return method_of<T, M>(a...) - std::log(c);
    else return method_of<T, M>(a...) / c;  // mean, standard deviation, median, mode, the support
}

template <class D, class M, std::size_t... I>
struct DistOp<D, M, std::index_sequence<I...>> : Real {
    template <std::size_t> using P = double;
    static constexpr auto name_buf = join_name(D::name, M::name);
    static constexpr const char* name = name_buf.data();
    static constexpr int nin = static_cast<int>(M::input) + static_cast<int>(sizeof...(I));
    static constexpr int nout = 1;
    static constexpr std::array<const char*, nin> args = [] {
        std::array<const char*, nin> a{};
        std::size_t k = 0;
        for (const char* s : D::params) a[k++] = s;
        if constexpr (M::input) a[k] = M::arg;
        return a;
    }();
    static constexpr double cost_ns = cost_of<D, M>();
    // The cost when the memo of values at integers answers the call (a memo row of
    // cost_table: a discrete distribution on a narrow support; negative without one), and
    // the cost for scalar parameters p (nary_bind picks the weight per call with it: a
    // narrow call costs 1-2 ns per element, a wide one up to tens of us, and one weight for
    // both made the narrow calls split into tasks so small that 10 threads ran slower).
    static constexpr std::size_t kParams = sizeof...(I);
    static constexpr double memo_cost_ns = cost_table::find(D::name, M::name, true);
    static double cost_for(const double* p) {
        if constexpr (memo_cost_ns > 0.0) {
            if (memo_narrow<D>(p[I]...)) return memo_cost_ns;
        }
        return cost_ns;
    }
    // the method's chunk kernel, when the distribution has one (nary_bind::table_of)
    static constexpr const LoopTable* table = []() -> const LoopTable* {
        if constexpr (M::input) {
            if constexpr (M::template chunked<D, Chunk<sizeof...(I)>>) return &chunk_table<D, M>;
        }
        return nullptr;
    }();

    static double f(P<I>... p, double x)
        requires M::input
    {
        if constexpr (has_limit<D, P<I>...>) {
            if (D::at_limit(p...)) {
                if (std::isnan(x)) return kNaN;
                if constexpr (kNonNegative<M>) return D::template limit_in<M>(x, p...) + 0.0;
                else return D::template limit_in<M>(x, p...);
            }
        }
        if (!D::valid(p...) || std::isnan(x)) return kNaN;
        if constexpr (kNonNegative<M>) return M::template call<D>(x, p...) + 0.0;
        else return M::template call<D>(x, p...);
    }
    static double f(P<I>... p)
        requires(!M::input)
    {
        if constexpr (has_limit<D, P<I>...>) {
            if (D::at_limit(p...)) return signed_zero(D::template limit_par<M>(p...), p...);
        }
        if (!D::valid(p...)) return kNaN;
        return signed_zero(M::template call<D>(p...), p...);
    }
    // A variance or a standard deviation is never negative, nor a mean, median or mode on a
    // support in [0, inf): never -0.0 either (Poisson(mu=-0.0), whose formulas give it). v + 0.0
    // turns -0.0 into +0.0 and changes no other value, as kNonNegative.
    static double signed_zero(double v, P<I>... p)
        requires(!M::input)
    {
        if constexpr (std::is_same_v<M, method::Variance> || std::is_same_v<M, method::StandardDeviation>) {
            return v + 0.0;
        } else if constexpr (std::is_same_v<M, method::Mean> || std::is_same_v<M, method::Median> ||
                             std::is_same_v<M, method::Mode>) {
            return v == 0.0 && DistOp<D, method::SupportLo>::f(p...) >= 0.0 ? 0.0 : v;
        } else {
            return v;
        }
    }
};

}  // namespace sd::stats
