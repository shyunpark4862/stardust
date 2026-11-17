// Fast path against precise definition, for the chunk kernels of group A (closed-form
// continuous distributions). For every (distribution, method) with a chunk kernel:
//   * the largest difference between D::m_chunk and the scalar D::m, in contract units
//     (relative; logpdf: over max(1, |value|); a subnormal value: over DBL_MIN), across
//     parameter sets and inputs covering the bulk, both tails, the ends of the support,
//     special values and a wide log-uniform range;
//   * that an element computed alone (n = 1) and with per-element parameters (same = false)
//     has the same bits as in a full chunk;
//   * ns per element of the kernel and of the scalar definition on bulk inputs.
#pragma once
#include "stats/engine.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace sw {
using sd::stats::Chunk;
using sd::stats::kChunkSize;
inline double g_worst = 0.0;
inline int g_bad = 0;
inline bool g_quick = false;
inline std::string g_only;

enum Kind { kRel, kMixed, kInv };
enum In { kX, kP, kLogP };

inline bool same_bits(double a, double b) { return std::memcmp(&a, &b, 8) == 0 || (a != a && b != b); }
inline double units(double fast, double ref, Kind kind) {
    if (same_bits(fast, ref) || fast == ref) return 0.0;
    if (std::isnan(fast) || std::isnan(ref) || std::isinf(fast) || std::isinf(ref)) return 1e300;
    double den = std::fabs(ref);
    if (kind == kMixed) den = std::max(1.0, den);
    den = std::max(den, std::numeric_limits<double>::min());
    return std::fabs(fast - ref) / den;
}

template <std::size_t NP>
struct Method {
    std::string name;
    std::function<double(double, const double*)> scalar;
    std::function<void(const Chunk<NP>&, double*)> chunk;
    Kind kind;
    In in;
};

template <std::size_t NP>
void run_chunks(const Method<NP>& m, const std::vector<double>& x, const std::array<double, NP>& p, bool same,
                std::vector<double>& out) {
    const int N = static_cast<int>(x.size());
    out.resize(x.size());
    static double pb[8][kChunkSize];
    for (std::size_t k = 0; k < NP; ++k) std::fill(pb[k], pb[k] + kChunkSize, p[k]);
    for (int b = 0; b < N; b += kChunkSize) {
        Chunk<NP> c;
        c.n = std::min(kChunkSize, N - b);
        c.x = x.data() + b;
        for (std::size_t k = 0; k < NP; ++k) c.p[k] = pb[k], c.same[k] = same;
        m.chunk(c, out.data() + b);
    }
}

// inputs: quantile-spread x (through icdf), wide log-uniform values of both signs, specials
template <std::size_t NP>
std::vector<double> inputs(In in, const std::function<double(double, const double*)>& icdf,
                           const std::array<double, NP>& p, std::mt19937_64& g, int n, bool bulk) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> x;
    const double inf = std::numeric_limits<double>::infinity();
    auto prob = [&](int i) {
        const double u = U(g);
        if (bulk) return 0.001 + 0.998 * u;
        switch (i % 6) {
            case 0: return u;
            case 1: return std::pow(10.0, -320.0 * u * u);       // lower tail
            case 2: return 1.0 - std::pow(10.0, -17.0 * u);      // upper tail
            case 3: return 0.5 + (u - 0.5) * std::pow(10.0, -16.0 * U(g));
            case 4: return std::ldexp(std::floor(u * 64.0), -6);  // dyadic points
            default: return u < 0.5 ? 0.25 + 1e-3 * (U(g) - 0.5) : 0.75 + 1e-3 * (U(g) - 0.5);
        }
    };
    for (int i = 0; i < n; ++i) {
        const double pr = prob(i);
        if (in == kP) {
            x.push_back(pr);
        } else if (in == kLogP) {
            const double u = U(g);
            if (bulk) x.push_back(std::log(pr));
            else if (i % 4 == 0) x.push_back(std::log(pr));
            else if (i % 4 == 1) x.push_back(-std::pow(10.0, -20.0 + 24.0 * u));
            else if (i % 4 == 2) x.push_back(-0.6931471805599453 * (1.0 + (u - 0.5) * std::pow(10.0, -15.0 * U(g))));
            else x.push_back(-745.0 * u);
        } else {
            double v;
            if (bulk || i % 3 != 2) {
                v = (!bulk && i % 6 == 4) ? icdf(pr, p.data()) : icdf(pr, p.data());
                if (!bulk && i % 9 == 0) v = std::nextafter(v, U(g) < 0.5 ? -inf : inf);
            } else {
                const double mag = std::pow(10.0, -310.0 + 620.0 * U(g));
                v = (U(g) < 0.5 ? -mag : mag);
                if (i % 2) v = icdf(0.5, p.data()) + (U(g) < 0.5 ? -1 : 1) * std::pow(10.0, -20.0 + 28.0 * U(g)) * std::max(1e-300, std::fabs(icdf(0.75, p.data()) - icdf(0.25, p.data())));
            }
            x.push_back(v);
        }
    }
    if (!bulk) {
        const double sp[] = {0.0, -0.0, 1.0, -1.0, 0.5, 0.25, 0.75, inf, -inf, 5e-324, -5e-324, 2.2250738585072014e-308,
                             1.7976931348623157e308, -1.7976931348623157e308, 1e-300, 1e300, -1e300, 2.0, -2.0,
                             -0.6931471805599453, -0.6931471805599454, -0.6931471805599452, -1.3862943611198906, -40.0, -700.0, -745.2, -1e5};
        for (double v : sp) x.push_back(v);
        for (std::size_t k = 0; k < NP; ++k) {
            x.push_back(p[k]);
            x.push_back(std::nextafter(p[k], inf));
            x.push_back(std::nextafter(p[k], -inf));
            x.push_back(-p[k]);
        }
        if (in == kX) {
            for (double q : {0.0, 1.0}) {
                const double e = icdf(q, p.data());
                x.push_back(e);
                x.push_back(std::nextafter(e, inf));
                x.push_back(std::nextafter(e, -inf));
            }
        }
    }
    return x;
}

template <std::size_t NP>
void sweep(const Method<NP>& m, const std::function<double(double, const double*)>& icdf,
           const std::vector<std::array<double, NP>>& params, int npts = 40000) {
    if (!g_only.empty() && m.name.find(g_only) == std::string::npos) return;
    std::mt19937_64 g(12345);
    double worst = 0.0, wx = 0.0, wf = 0.0, wr = 0.0;
    std::array<double, NP> wp{};
    long total = 0, bits_bad = 0;
    std::vector<double> out, out2;
    if (g_quick) npts = 4000;
    for (const auto& p : params) {
        const std::vector<double> x = inputs<NP>(m.in, icdf, p, g, npts, false);
        run_chunks(m, x, p, true, out);
        run_chunks(m, x, p, false, out2);
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (x[i] != x[i]) continue;
            const double ref = m.scalar(x[i], p.data());
            double u = units(out[i], ref, m.kind);
            if (m.kind == kInv && u > 2e-16 && u < 1e299) {
                // the contract of an inverse: |dx| <= 1e-14 max(|x|, |p dx/dp|) (kappa = |p x'(p) / x|)
                const double h = 1e-7 * std::fabs(x[i]);
                double hi = x[i] + h, lo = x[i] - h;
                if (m.in == kP) hi = std::min(hi, 1.0); else hi = std::min(hi, 0.0);
                const double slope = (m.scalar(hi, p.data()) - m.scalar(lo, p.data())) / (hi - lo);
                const double den = std::max(std::fabs(ref), std::fabs(x[i] * slope));
                if (std::isfinite(den) && den > 0) u = std::fabs(out[i] - ref) / den;
            }
            if (u > worst) worst = u, wx = x[i], wf = out[i], wr = ref, wp = p;
            // alone, and with per-element parameters
            double one;
            Chunk<NP> c;
            c.n = 1, c.x = &x[i];
            for (std::size_t k = 0; k < NP; ++k) c.p[k] = &p[k], c.same[k] = (i % 2 == 0);
            m.chunk(c, &one);
            if (!same_bits(one, out[i]) || !same_bits(out2[i], out[i])) {
                if (++bits_bad <= 3) std::printf("   BITS %s x=%.17g chunk=%.17g alone=%.17g varying=%.17g\n", m.name.c_str(), x[i], out[i], one, out2[i]);
            }
            ++total;
        }
    }
    // speed on bulk inputs, first parameter set
    std::mt19937_64 g2(7);
    const std::vector<double> xb = inputs<NP>(m.in, icdf, params[0], g2, 100 * kChunkSize, true);
    double tf = 1e9, ts = 1e9;
    std::vector<double> o(xb.size());
    for (int r = 0; r < 7; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        run_chunks(m, xb, params[0], true, o);
        tf = std::min(tf, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / xb.size());
    }
    static volatile double zero = 0.0;
    std::array<double, NP> pv = params[0];
    for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < xb.size(); ++i) {
            if ((i & 1023) == 0) for (auto& v : pv) v += zero;
            o[i] = m.scalar(xb[i], pv.data());
        }
        ts = std::min(ts, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / xb.size());
    }
    // share of bulk elements that differ from the scalar definition is not the fallback share; report time only
    std::printf("%-22s pts %7ld  max diff %9.3g %s  scalar %6.1f ns  chunk %6.1f ns%s", m.name.c_str(), total, worst,
                worst <= 2e-15 ? "ok " : "BAD", ts, tf, bits_bad ? "  BITS DIFFER" : "");
    if (worst > 2e-16) {
        std::printf("   [worst x=%.17g fast=%.17g ref=%.17g p=", wx, wf, wr);
        for (double v : wp) std::printf("%g ", v);
        std::printf("]");
    }
    std::printf("\n");
    g_worst = std::max(g_worst, worst);
    if (worst > 2e-15 || bits_bad) ++g_bad;
}
}  // namespace sw

#define SW_M(NP, D, m, kind, in, ...)                                                             \
    sw::sweep<NP>(sw::Method<NP>{#D "." #m, SW_S##NP(D, m), [](const sw::Chunk<NP>& c, double* o) { sd::stats::D::m##_chunk(c, o); }, kind, in}, SW_S##NP(D, icdf), __VA_ARGS__)
#define SW_S1(D, m) [](double x, const double* p) { return sd::stats::D::m(x, p[0]); }
#define SW_S2(D, m) [](double x, const double* p) { return sd::stats::D::m(x, p[0], p[1]); }
#define SW_S3(D, m) [](double x, const double* p) { return sd::stats::D::m(x, p[0], p[1], p[2]); }
