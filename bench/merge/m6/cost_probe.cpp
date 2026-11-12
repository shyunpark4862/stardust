// What DESIGN §4.4 rules 3 and 11 would cost the array area, measured on bare
// kernels before M6 changes anything (one thread, 1M elements, best of 30):
//   rule 3: int64 vs float64 comparison, rounded (now) vs exact (the frame's)
//   rule 11: mean of int64, double sum (now) vs exact int128 sums
//
//   clang++ -std=c++20 -O3 -mcpu=native -ffp-contract=off bench/merge/m6/cost_probe.cpp -o build/cost_probe && build/cost_probe
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using I = std::int64_t;
static volatile double sink;
static volatile std::uint8_t bsink;

template <class F>
double best_us(F&& f, int reps = 30) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        best = us < best ? us : best;
    }
    return best;
}

// The frame's three-way comparison (csrc/frame/ops/elementwise.cpp).
[[gnu::noinline]] int compare_slow(I i, double d) {
    if (d != d) return 2;
    if (d >= 9223372036854775808.0) return -1;
    if (d < -9223372036854775808.0) return 1;
    const double td = std::trunc(d);
    const auto t = static_cast<I>(td);
    if (i != t) return i < t ? -1 : 1;
    const double frac = d - td;
    return frac > 0 ? -1 : frac < 0 ? 1 : 0;
}

[[gnu::noinline]] void less_rounded(const I* a, const double* b, std::uint8_t* o, I n) {
    for (I k = 0; k < n; ++k) o[k] = static_cast<double>(a[k]) < b[k];
}

// Exact, vector-friendly: the plain comparison for |i| <= 2^53 (exact there),
// then a second pass fixes the rare lanes outside.
[[gnu::noinline]] void less_exact(const I* a, const double* b, std::uint8_t* o, I n) {
    constexpr I lim = I{1} << 53;
    std::uint64_t outside = 0;
    for (I k = 0; k < n; ++k) {
        o[k] = static_cast<double>(a[k]) < b[k];
        outside |= static_cast<std::uint64_t>(a[k] + lim) > static_cast<std::uint64_t>(2 * lim);
    }
    if (outside)
        for (I k = 0; k < n; ++k)
            if (a[k] < -lim || a[k] > lim) o[k] = compare_slow(a[k], b[k]) == -1;
}

// Exact, two passes: min/max of the ints first (vectorizes), then the plain
// comparison when they are all within +-2^53, else the slow path per lane.
[[gnu::noinline]] void less_exact2(const I* a, const double* b, std::uint8_t* o, I n) {
    constexpr I lim = I{1} << 53;
    I lo = 0, hi = 0;
    for (I k = 0; k < n; ++k) {
        lo = a[k] < lo ? a[k] : lo;
        hi = a[k] > hi ? a[k] : hi;
    }
    if (lo >= -lim && hi <= lim) {
        less_rounded(a, b, o, n);
        return;
    }
    for (I k = 0; k < n; ++k)
        o[k] = (a[k] < -lim || a[k] > lim) ? compare_slow(a[k], b[k]) == -1 : static_cast<double>(a[k]) < b[k];
}

// numpy-style pairwise sum of the int64 values converted to double.
double pairwise(const I* a, I n) {
    if (n <= 128) {
        double r[8] = {};
        I i = 0;
        for (; i + 8 <= n; i += 8)
            for (int j = 0; j < 8; ++j) r[j] += static_cast<double>(a[i + j]);
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) s += static_cast<double>(a[i]);
        return s;
    }
    I n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

double mean_double(const I* a, I n) { return (0.0 + pairwise(a, n)) / static_cast<double>(n); }

double mean_int128(const I* a, I n) {  // the frame's loop
    __int128 t = 0;
    for (I i = 0; i < n; ++i) t += a[i];
    return static_cast<double>(t) / static_cast<double>(n);
}

// Exact and SIMD-friendly: high and low 32-bit halves summed in int64 lanes.
// Each half is < 2^32 in magnitude, so up to 2^31 terms cannot overflow.
double mean_split(const I* a, I n) {
    I hi = 0;
    std::uint64_t lo = 0;
    for (I i = 0; i < n; ++i) {
        hi += a[i] >> 32;
        lo += static_cast<std::uint32_t>(a[i]);
    }
    const __int128 t = (static_cast<__int128>(hi) << 32) + lo;
    return static_cast<double>(t) / static_cast<double>(n);
}

// Axis 0 of a (rows, cols) C-order matrix: one output per column, the rows
// added into a row of accumulators (how a reduction over the outer axis runs).
void axis0_double(const I* a, I rows, I cols, double* out) {
    std::memset(out, 0, sizeof(double) * cols);
    for (I r = 0; r < rows; ++r)
        for (I c = 0; c < cols; ++c) out[c] += static_cast<double>(a[r * cols + c]);
}

void axis0_split(const I* a, I rows, I cols, double* out, std::vector<I>& hi, std::vector<std::uint64_t>& lo) {
    hi.assign(cols, 0);
    lo.assign(cols, 0);
    for (I r = 0; r < rows; ++r)
        for (I c = 0; c < cols; ++c) {
            hi[c] += a[r * cols + c] >> 32;
            lo[c] += static_cast<std::uint32_t>(a[r * cols + c]);
        }
    for (I c = 0; c < cols; ++c) out[c] = static_cast<double>((static_cast<__int128>(hi[c]) << 32) + lo[c]) / rows;
}

int main() {
    const I n = 1'000'000;
    std::mt19937_64 g(0);
    std::vector<I> small(n), big(n);
    std::vector<double> f(n);
    for (I k = 0; k < n; ++k) {
        small[k] = static_cast<I>(g() % 2'000'001) - 1'000'000;
        big[k] = static_cast<I>(g());  // most are outside +-2^53
        f[k] = static_cast<double>(static_cast<I>(g() % 2'000'001) - 1'000'000) + 0.5;
    }
    std::vector<std::uint8_t> o(n);
    std::printf("rule 3: int64 < float64, 1M elements (us)\n");
    std::printf("  rounded (now)                 %8.1f\n", best_us([&] { less_rounded(small.data(), f.data(), o.data(), n); bsink = o[n / 2]; }));
    std::printf("  exact, ints within 2^53       %8.1f\n", best_us([&] { less_exact(small.data(), f.data(), o.data(), n); bsink = o[n / 2]; }));
    std::printf("  exact 2-pass, within 2^53     %8.1f\n", best_us([&] { less_exact2(small.data(), f.data(), o.data(), n); bsink = o[n / 2]; }));
    std::printf("  exact 2-pass, mostly outside  %8.1f\n", best_us([&] { less_exact2(big.data(), f.data(), o.data(), n); bsink = o[n / 2]; }));
    std::printf("  exact, ints mostly outside    %8.1f\n", best_us([&] { less_exact(big.data(), f.data(), o.data(), n); bsink = o[n / 2]; }));
    std::printf("rule 11: mean(int64), 1M elements (us)\n");
    std::printf("  double pairwise (now)         %8.1f\n", best_us([&] { sink = mean_double(small.data(), n); }));
    std::printf("  int128 scalar (frame)         %8.1f\n", best_us([&] { sink = mean_int128(small.data(), n); }));
    std::printf("  int64 hi/lo halves            %8.1f\n", best_us([&] { sink = mean_split(small.data(), n); }));
    const I rows = 1000, cols = 1000;
    std::vector<double> out(cols);
    std::vector<I> hi;
    std::vector<std::uint64_t> lo;
    std::printf("rule 11: mean(int64, axis=0) of (1000, 1000) (us)\n");
    std::printf("  double accumulators (now)     %8.1f\n", best_us([&] { axis0_double(small.data(), rows, cols, out.data()); sink = out[7]; }));
    std::printf("  int64 hi/lo accumulators      %8.1f\n", best_us([&] { axis0_split(small.data(), rows, cols, out.data(), hi, lo); sink = out[7]; }));
}
