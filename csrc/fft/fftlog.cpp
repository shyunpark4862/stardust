// FFTLog (fftlog.hpp; docs/api/fft.md §6; docs/design/fft.md §3.9; milestone FT5): the plans and their cache,
// fhtoffset, and the native functions
//
//   _fht(a, dln, mu, offset, bias, inverse)        -> fht, ifht (the arguments checked in Python)
//   _fhtoffset(dln, mu, initial, bias)             -> fhtoffset
//   _fht_plan(n, dln, mu, offset, bias, inverse)   the plan's coefficients and biases (tests)
//   _fht_cache_state(), _fht_cache_clear()
//
// Error (the contract of fftlog.hpp). Write v = B_in a (exact), X = F v (the real DFT's half spectrum),
// y = R(U' X) with U' = U / n and R the unnormalised inverse real DFT of a Hermitian half spectrum, and
// A = B_out P y (P the reversal). ||F v||_H = sqrt(n) ||v|| and ||R W|| = sqrt(n) ||W||_H, so ||y|| <= M ||v||
// with M = max |U_m|. The computed v has ||v_hat - v|| <= e1 ||v|| (e1 = the bias's u/2 + 2^-59 and the
// product's u/2 per entry), the real transform X_hat = F v_hat + E1 with ||E1||_H <= eps_F sqrt(n) ||v_hat||,
// the coefficient products W_hat_m = U'_m X_hat_m (1 + d_m), |d_m| <= u/2 + 2u + 2^-59 (the coefficient's
// rounding and the FMA product, Jeannerod, Kornerup, Louvet and Muller 2017), the inverse transform
// y_hat = R W_hat + E2 with ||E2|| <= eps_F sqrt(n) ||W_hat||_H, and the output products e1 again. Summing,
// ||A_hat - A|| <= ||B_out||_inf M ||v|| (2 eps_F + 2 e1 + max |d_m| + O(u^2)), inside (2 eps_F + 8u) of the
// contract. The floor collects the gradual underflow of both transforms (tau each, times M and ||B_out||) and
// of the three products (2^-1075 per entry).
#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

#include <nanobind/stl/vector.h>

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "fft/args.hpp"
#include "fft/driver.hpp"
#include "fft/fft.hpp"
#include "fft/fftlog.hpp"
#include "parallel/pool.hpp"

namespace nb = nanobind;

namespace sd::fft {

namespace {

using lg::DC;
using lg::DD;

bool same_key(const FhtKey& a, const FhtKey& b) {
    return a.n == b.n && a.inverse == b.inverse && std::memcmp(&a.dln, &b.dln, 8) == 0 &&
           std::memcmp(&a.mu, &b.mu, 8) == 0 && std::memcmp(&a.offset, &b.offset, 8) == 0 &&
           std::memcmp(&a.bias, &b.bias, 8) == 0;
}

// A nonpositive integer (a pole of Gamma): x = hi + lo normalised, so an integer below 2^53 has lo = 0.
bool is_pole(DD x) { return x.lo == 0.0 && x.hi <= 0.0 && x.hi == std::floor(x.hi); }

// fn(i) for i in [0, count), split into chunks of `per` on the pool (each i is computed alone: the threads
// change no bit). per_ns: the time of one i, for the pool's wake-up rule.
template <class Fn>
void for_each(int64_t count, int64_t per, int64_t per_ns, Fn fn) {
    const int64_t ntasks = (count + per - 1) / per;
    if (ntasks > 1) {
        parallel::parallel_for(
            ntasks,
            [&](int64_t t) {
                const int64_t i1 = std::min(count, (t + 1) * per);
                for (int64_t i = t * per; i < i1; ++i) fn(i);
            },
            per * per_ns
        );  // fmt: skip
    } else {
        for (int64_t i = 0; i < count; ++i) fn(i);
    }
}

// (mu + 1 + sign q) / 2 as a double-double (the sum of three doubles; exact when it is an integer).
DD half_sum(double mu, double q) { return lg::ldexp_dd(dd::add(dd::two_sum(mu, 1.0), lg::dd_of(q)), -1); }

// The rounded parts of e^e (cos t, sin t), t already reduced; with real, of e^e cos t alone (fht's real
// entries) or, with recip, of e^e / cos t (ifht's: 1 / (n Re U), since U's imaginary part is dropped first).
void round_cis(DD e, DD t, bool real, bool recip, double* out) {
    const DD E = lg::exp_dd(e);
    DD s, c;
    lg::sincos_dd(t, s, c);
    out[1] = 0.0;
    if (real && recip) {
        out[0] = c.hi == 0.0 ? INFINITY : lg::ddiv(E, c).hi;  // e^e finite: |cos t| >= 2^-110 or exactly 0
        if (std::isinf(E.hi)) out[0] = std::copysign(INFINITY, c.hi);
        return;
    }
    if (std::isinf(E.hi)) {
        out[0] = c.hi == 0.0 ? 0.0 : std::copysign(INFINITY, c.hi);
        if (!real) out[1] = s.hi == 0.0 ? 0.0 : std::copysign(INFINITY, s.hi);
        return;
    }
    out[0] = dd::mul(E, c).hi;
    if (!real) out[1] = dd::mul(E, s).hi;
}

}  // namespace

FhtPlan::FhtPlan(const FhtKey& key) : key_(key) {
    const int64_t n = key.n, h = n / 2 + 1;
    const double q = key.bias;
    coef_.assign(static_cast<size_t>(2 * h), 0.0);
    const DD xp = half_sum(key.mu, q), xm = half_sum(key.mu, -q);
    const DD ndln = dd::mul(lg::dd_of(key.dln), static_cast<double>(n));  // n dln, exactly
    const DD delta = lg::ddiv(lg::kPi, ndln);  // y_m = m pi / (n dln)
    const DD lin = lg::ddiv(dd::sub(lg::kLn2, lg::dd_of(key.offset)), ndln);  // 2 y_m (ln 2 - lnkr) = 2 pi m lin
    const DD qln2 = dd::mul(lg::kLn2, q);
    const DD lnn = lg::clog(DC{lg::dd_of(static_cast<double>(n)), lg::dd_of(0.0)}).re;
    const bool inv = key.inverse;

    // coef_m = e^(+-Re Z - ln n) (cos, sin)(Im Z), Z = log Gamma(x+ + i y) - conj log Gamma(x- + i y) + q ln 2 +
    // 2 pi i m lin
    auto coefficient = [&](int64_t m) {
        double* out = coef_.data() + 2 * m;
        DD re, im;
        if (m == 0 && (is_pole(xp) || is_pole(xm))) {
            if (!(is_pole(xp) && is_pole(xm))) {  // U_0 infinite (fht singular) or 0 (ifht singular): 0 both ways
                singular_ = true;
                return;
            }
            // Gamma(-j + e) / Gamma(-k + e) -> (-1)^(j + k) k! / j!
            const double j = -xp.hi, k = -xm.hi;
            re = dd::sub(
                lg::lngamma(DC{lg::dd_of(k + 1.0), lg::dd_of(0.0)}).re,
                lg::lngamma(DC{lg::dd_of(j + 1.0), lg::dd_of(0.0)}).re
            );  // fmt: skip
            re = dd::add(re, qln2);
            im = std::fmod(j + k, 2.0) == 0.0 ? lg::dd_of(0.0) : lg::kPi;
        } else {
            const DD y = dd::mul(delta, static_cast<double>(m));
            const DC zp = lg::lngamma(DC{xp, y});
            if (q == 0.0) {
                re = lg::dd_of(0.0);
                im = dd::mul(zp.im, 2.0);
            } else {
                const DC zm = lg::lngamma(DC{xm, y});
                re = dd::add(dd::sub(zp.re, zm.re), qln2);
                im = dd::add(zp.im, zm.im);
            }
            im = dd::add(im, dd::mul(lg::kTwoPi, lg::frac_even(dd::mul(lin, static_cast<double>(m)))));
        }
        const bool real = m == 0 || 2 * m == n;  // the real spectral entries take Re U only (scipy forces it)
        round_cis(dd::sub(inv ? dd::neg(re) : re, lnn), lg::reduce_2pi(im), real, inv, out);
    };
    for_each(h, 16, 2000, coefficient);

    if (q != 0.0) {
        const DD s = dd::mul(lg::dd_of(q), key.dln), c = dd::mul(lg::dd_of(q), key.offset);  // exact products
        bin_.resize(static_cast<size_t>(n));
        bout_.resize(static_cast<size_t>(n));
        for_each(n, 4096, 20, [&](int64_t j) {
            const double k = 0.5 * static_cast<double>(2 * j - (n - 1));  // j - j_c, exactly
            const DD sk = dd::mul(s, k);
            // fht: e^(-q k dln) in, e^(-q (k dln + lnkr)) out; ifht: e^(q (k dln + lnkr)) in, e^(q k dln) out
            const DD ein = inv ? dd::add(sk, c) : dd::neg(sk);
            const DD eout = inv ? sk : dd::neg(dd::add(sk, c));
            bin_[static_cast<size_t>(j)] = lg::exp_round(ein);
            bout_[static_cast<size_t>(j)] = lg::exp_round(eout);
        });
    }
}

// ---------------------------------------------------------------- the cache

namespace {

struct FhtEntry {
    FhtKey key;
    std::shared_ptr<const FhtPlan> plan;
    int64_t bytes;
    uint64_t used;
};

std::mutex g_fht_mutex;
std::vector<FhtEntry> g_fht_entries;
uint64_t g_fht_clock = 0;
int64_t g_fht_bytes = 0;

FhtEntry* find_fht(const FhtKey& key) {
    for (FhtEntry& e : g_fht_entries) {
        if (same_key(e.key, key)) return &e;
    }
    return nullptr;
}

}  // namespace

std::shared_ptr<const FhtPlan> get_fht_plan(const FhtKey& key) {
    {
        const std::lock_guard<std::mutex> lock(g_fht_mutex);
        if (FhtEntry* e = find_fht(key)) {
            e->used = ++g_fht_clock;
            return e->plan;
        }
    }
    std::shared_ptr<const FhtPlan> made;
    {
        const parallel::GilRelease release;
        made = std::make_shared<const FhtPlan>(key);
    }
    const int64_t b = made->bytes();
    if (b > kFhtCacheBytes) return made;
    std::shared_ptr<const FhtPlan> dropped;  // freed after the lock is released
    const std::lock_guard<std::mutex> lock(g_fht_mutex);
    if (FhtEntry* e = find_fht(key)) {
        e->used = ++g_fht_clock;
        dropped = std::move(made);
        return e->plan;
    }
    while (!g_fht_entries.empty() &&
           (static_cast<int>(g_fht_entries.size()) >= kFhtCachePlans || g_fht_bytes + b > kFhtCacheBytes)) {
        auto lru =
            std::min_element(g_fht_entries.begin(), g_fht_entries.end(), [](const FhtEntry& u, const FhtEntry& v) {
                return u.used < v.used;
            });
        g_fht_bytes -= lru->bytes;
        g_fht_entries.erase(lru);
    }
    g_fht_entries.push_back({key, made, b, ++g_fht_clock});
    g_fht_bytes += b;
    return made;
}

double fht_offset(double dln, double mu, double initial, double bias) {
    const DD xp = half_sum(mu, bias), xm = half_sum(mu, -bias);
    const DD y = lg::ddiv(lg::kPi, lg::dd_of(2.0 * dln));  // pi / (2 dln)
    const DC zp = lg::lngamma(DC{xp, y}), zm = lg::lngamma(DC{xm, y});
    const DD arg = dd::add(
        lg::ddiv(dd::sub(lg::kLn2, lg::dd_of(initial)), lg::dd_of(dln)), lg::ddiv(dd::add(zp.im, zm.im), lg::kPi)
    );  // fmt: skip
    const DD f = lg::frac_even(arg);
    return dd::add(dd::mul(f, dln), lg::dd_of(initial)).hi;
}

namespace {

nb::object fht(nb::handle ah, double dln, double mu, double offset, double bias, bool inverse) {
    const NDArray& a = require_ndarray(ah, inverse ? "A" : "a");
    const int axis = a.ndim - 1;
    const int64_t n = a.shape[static_cast<size_t>(axis)];
    if (n < 1) throw nb::value_error("cannot transform an axis of length 0");
    NDArray out = new_result(a, axis, n, DType::Float64);
    if (out.size() == 0) return wrap(std::move(out));
    FhtKey key;
    key.n = n, key.dln = dln, key.mu = mu, key.offset = offset, key.bias = bias, key.inverse = inverse;
    const std::shared_ptr<const FhtPlan> plan = get_fht_plan(key);
    fht_axis(a, axis, *plan, out);
    return wrap(std::move(out));
}

nb::dict fht_plan_py(int64_t n, double dln, double mu, double offset, double bias, bool inverse) {
    if (n < 1) throw nb::value_error("_fht_plan(): n must be at least 1");
    FhtKey key;
    key.n = n, key.dln = dln, key.mu = mu, key.offset = offset, key.bias = bias, key.inverse = inverse;
    const std::shared_ptr<const FhtPlan> p = get_fht_plan(key);
    nb::dict d;
    const int64_t h = n / 2 + 1;
    NDArray c = new_array(DType::Complex128, std::vector<int64_t>{h});
    std::memcpy(c.data, p->coef(), static_cast<size_t>(16 * h));
    d["coef"] = wrap(std::move(c));
    for (const char* name : {"bin", "bout"}) {
        const double* v = std::strcmp(name, "bin") == 0 ? p->bin() : p->bout();
        if (v == nullptr) {
            d[name] = nb::none();
        } else {
            NDArray b = new_array(DType::Float64, std::vector<int64_t>{n});
            std::memcpy(b.data, v, static_cast<size_t>(8 * n));
            d[name] = wrap(std::move(b));
        }
    }
    d["singular"] = p->singular();
    d["bytes"] = p->bytes();
    return d;
}

nb::dict fht_cache_state() {
    const std::lock_guard<std::mutex> lock(g_fht_mutex);
    std::vector<const FhtEntry*> order;
    for (const FhtEntry& e : g_fht_entries) order.push_back(&e);
    std::sort(order.begin(), order.end(), [](const FhtEntry* u, const FhtEntry* v) { return u->used < v->used; });
    std::vector<int64_t> lengths;
    for (const FhtEntry* e : order) lengths.push_back(e->key.n);
    nb::dict d;
    d["plans"] = static_cast<int64_t>(g_fht_entries.size());
    d["bytes"] = g_fht_bytes;
    d["lengths"] = lengths;
    d["max_plans"] = kFhtCachePlans;
    d["max_bytes"] = kFhtCacheBytes;
    return d;
}

void fht_cache_clear() {
    std::vector<FhtEntry> old;
    const std::lock_guard<std::mutex> lock(g_fht_mutex);
    old.swap(g_fht_entries);
    g_fht_bytes = 0;
}

}  // namespace

void bind_fftlog(nb::module_& m) {
    m.def(
        "_fht", &fht, nb::arg("a").none(), nb::arg("dln"), nb::arg("mu"), nb::arg("offset"), nb::arg("bias"),
        nb::arg("inverse")
    );  // fmt: skip
    m.def("_fhtoffset", &fht_offset, nb::arg("dln"), nb::arg("mu"), nb::arg("initial"), nb::arg("bias"));
    m.def(
        "_fht_plan", &fht_plan_py, nb::arg("n"), nb::arg("dln"), nb::arg("mu"), nb::arg("offset"), nb::arg("bias"),
        nb::arg("inverse")
    );  // fmt: skip
    m.def("_fht_cache_state", &fht_cache_state);
    m.def("_fht_cache_clear", &fht_cache_clear);
}

}  // namespace sd::fft
