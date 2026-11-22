// The plans, Bluestein's transform and the plan cache (docs/design/fft.md §3.3, §3.6; plan.hpp,
// bluestein.hpp, ddmath.hpp).
#include "fft/plan.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

#include "fft/ddmath.hpp"
#include "parallel/pool.hpp"

namespace sd::fft {

// ---------------------------------------------------------------- double-double DFT

namespace dd {

void dft(const DC* x, int64_t stride, DC* y, int64_t m, int64_t top, const CisTable& w) {
    if (m == 1) {
        y[0] = x[0];
        return;
    }
    const int r = m % 4 == 0 ? 4 : m % 2 == 0 ? 2 : m % 3 == 0 ? 3 : 5;
    const int64_t mr = m / r;
    for (int j = 0; j < r; ++j) dft(x + j * stride, stride * r, y + j * mr, mr, top, w);
    const int64_t step = top / m;  // W_m^t = cis(t step, top)
    DC wr[5][5];                   // W_r^(j k) for the odd radices
    if (r == 3 || r == 5) {
        for (int j = 0; j < r; ++j) {
            for (int k = 0; k < r; ++k) wr[j][k] = w(((j * k) % r) * (top / r));
        }
    }
    DC z[5];
    for (int64_t k1 = 0; k1 < mr; ++k1) {
        for (int j = 0; j < r; ++j) {
            const DC v = y[j * mr + k1];
            z[j] = (j == 0 || k1 == 0) ? v : cmul(v, w((j * k1 * step) % top));
        }
        if (r == 2) {
            y[k1] = cadd(z[0], z[1]);
            y[mr + k1] = csub(z[0], z[1]);
        } else if (r == 4) {
            const DC t0 = cadd(z[0], z[2]), t1 = csub(z[0], z[2]);
            const DC t2 = cadd(z[1], z[3]), t3 = mul_neg_i(csub(z[1], z[3]));
            y[k1] = cadd(t0, t2);
            y[mr + k1] = cadd(t1, t3);
            y[2 * mr + k1] = csub(t0, t2);
            y[3 * mr + k1] = csub(t1, t3);
        } else {
            for (int k2 = 0; k2 < r; ++k2) {
                DC acc = z[0];
                for (int j = 1; j < r; ++j) acc = cadd(acc, cmul(z[j], wr[j][k2]));
                y[k2 * mr + k1] = acc;
            }
        }
    }
}

}  // namespace dd

// ---------------------------------------------------------------- Bluestein

int64_t smooth5_at_least(int64_t target) {
    int64_t best = INT64_MAX;
    for (int64_t p5 = 1;; p5 *= 5) {
        for (int64_t p35 = p5;; p35 *= 3) {
            int64_t v = p35;
            while (v < target) v *= 2;
            best = std::min(best, v);
            if (p35 >= target) break;
        }
        if (p5 >= target) break;
    }
    return best;
}

namespace {

using detail::CV;

// y_j = x_j w_j for j < count (the FMA complex product of cfft.hpp); SwapIn / SwapOut exchange the
// real and imaginary parts of x / of the product. y may be x.
template <bool SwapIn, bool SwapOut>
void cmul_line(const double* x, const double* w, double* y, int64_t count) {
    int64_t j = 0;
    for (; j + 1 < count; j += 2) {
        const float64x2x2_t wv = vld2q_f64(w + 2 * j);
        const CV a = detail::load_il2<SwapIn>(x + 2 * j);
        detail::store_il2<SwapOut>(y + 2 * j, detail::cmul(a, wv.val[0], wv.val[1]));
    }
    if (j < count) {
        const CV a = detail::load_il1<SwapIn>(x + 2 * j);
        const CV r = detail::cmul(a, vld1q_dup_f64(w + 2 * j), vld1q_dup_f64(w + 2 * j + 1));
        detail::store_il1<SwapOut>(y + 2 * j, r);
    }
}

}  // namespace

BluesteinPlan::BluesteinPlan(int64_t n) : n_(n), m_(smooth5_at_least(2 * n - 1)), inner_(m_) {
    const int64_t m = m_;
    const dd::CisTable cw(2 * n);
    std::vector<dd::DC> b(static_cast<size_t>(m)), f(static_cast<size_t>(m));
    chirp_.resize(static_cast<size_t>(2 * n));
    for (int64_t l = 0; l < n; ++l) {
        const auto sq = static_cast<unsigned __int128>(l) * static_cast<unsigned __int128>(l);
        const auto t = static_cast<int64_t>(sq % static_cast<unsigned __int128>(2 * n));
        const dd::DC w = cw(t);  // e^(-pi i l^2 / n) = e^(-2 pi i (l^2 mod 2n) / 2n)
        chirp_[static_cast<size_t>(2 * l)] = w.re.hi;
        chirp_[static_cast<size_t>(2 * l + 1)] = w.im.hi;
        b[static_cast<size_t>(l)] = dd::conj(w);
        if (l > 0) b[static_cast<size_t>(m - l)] = dd::conj(w);
    }
    {
        const dd::CisTable cm(m);
        dd::dft(b.data(), 1, f.data(), m, m, cm);
    }
    beta_.resize(static_cast<size_t>(2 * m));
    double big = 0.0;
    const double dm = static_cast<double>(m);
    for (int64_t k = 0; k < m; ++k) {
        const dd::DC& v = f[static_cast<size_t>(k)];
        beta_[static_cast<size_t>(2 * k)] = dd::div(v.re, dm).hi;
        beta_[static_cast<size_t>(2 * k + 1)] = dd::div(v.im, dm).hi;
        big = std::max(big, std::hypot(v.re.hi, v.im.hi));
    }
    rho_ = big / std::sqrt(static_cast<double>(n)) * (1.0 + 0x1p-40);  // rounded up
}

template <bool Swap>
void BluesteinPlan::run(const double* x, double* y, double* work) const {
    const int64_t n = n_, m = m_;
    double* const a = work;
    double* const b = work + 2 * m;
    double* const w = work + 4 * m;
    cmul_line<Swap, false>(x, chirp_.data(), a, n);
    std::memset(a + 2 * n, 0, static_cast<size_t>(2 * (m - n)) * sizeof(double));
    inner_.execute(a, b, w, false);
    cmul_line<false, false>(b, beta_.data(), b, m);
    inner_.execute(b, a, w, true);
    cmul_line<false, Swap>(a, chirp_.data(), y, n);
}

void BluesteinPlan::execute(const double* x, double* y, double* work, bool inverse) const {
    if (inverse) {
        run<true>(x, y, work);
    } else {
        run<false>(x, y, work);
    }
}

// ---------------------------------------------------------------- plans

namespace {

int64_t levels(int64_t n) {  // ceil(log2 n), 1 at least
    int64_t l = 0;
    while ((int64_t{1} << l) < n) ++l;
    return std::max<int64_t>(l, 1);
}

}  // namespace

Plan::Plan(Kind kind, int64_t n) : kind_(kind), n_(n) {
    if (is_mixed_length(n)) {
        mixed_ = std::make_unique<CfftPlan>(n);
        cost_ = n * levels(n);
    } else {
        blue_ = std::make_unique<BluesteinPlan>(n);
        const int64_t m = blue_->m();
        cost_ = 2 * m * levels(m) + 4 * m;
    }
}

// ---------------------------------------------------------------- the cache

namespace {

struct Entry {
    Kind kind;
    int64_t n;
    std::shared_ptr<const Plan> plan;
    int64_t bytes;
    uint64_t used;
};

std::mutex g_mutex;
std::vector<Entry> g_entries;
uint64_t g_clock = 0;
int64_t g_bytes = 0;

Entry* find(Kind kind, int64_t n) {
    for (Entry& e : g_entries) {
        if (e.kind == kind && e.n == n) return &e;
    }
    return nullptr;
}

}  // namespace

std::shared_ptr<const Plan> get_plan(Kind kind, int64_t n) {
    {
        const std::lock_guard<std::mutex> lock(g_mutex);
        if (Entry* e = find(kind, n)) {
            e->used = ++g_clock;
            return e->plan;
        }
    }
    std::shared_ptr<const Plan> made;
    {
        const parallel::GilRelease release;
        made = std::make_shared<const Plan>(kind, n);
    }
    const int64_t b = made->bytes();
    if (b > kCacheBytes) return made;
    std::shared_ptr<const Plan> dropped;  // freed after the lock is released
    const std::lock_guard<std::mutex> lock(g_mutex);
    if (Entry* e = find(kind, n)) {  // another thread made it meanwhile: keep the first
        e->used = ++g_clock;
        dropped = std::move(made);
        return e->plan;
    }
    while (!g_entries.empty() && (static_cast<int>(g_entries.size()) >= kCachePlans || g_bytes + b > kCacheBytes)) {
        auto lru = std::min_element(g_entries.begin(), g_entries.end(), [](const Entry& u, const Entry& v) {
            return u.used < v.used;
        });
        g_bytes -= lru->bytes;
        g_entries.erase(lru);
    }
    g_entries.push_back({kind, n, made, b, ++g_clock});
    g_bytes += b;
    return made;
}

CacheState cache_state() {
    const std::lock_guard<std::mutex> lock(g_mutex);
    CacheState s;
    std::vector<const Entry*> order;
    for (const Entry& e : g_entries) order.push_back(&e);
    std::sort(order.begin(), order.end(), [](const Entry* u, const Entry* v) { return u->used < v->used; });
    for (const Entry* e : order) s.lengths.push_back(e->n);
    s.plans = static_cast<int64_t>(g_entries.size());
    s.bytes = g_bytes;
    return s;
}

void cache_clear() {
    std::vector<Entry> old;
    {
        const std::lock_guard<std::mutex> lock(g_mutex);
        old.swap(g_entries);
        g_bytes = 0;
    }
}

}  // namespace sd::fft
