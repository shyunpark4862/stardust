// Timings of sd.smooth's kernel candidates (docs/design/smooth.md §3.1, §3.2), compiled by tools/smooth_probe.py
// with sd's flags (-O3 -mcpu=native -ffp-contract=off). Each line of the output is "<name> <value> <unit>".
//
//   lowess   the one-pass closed form of the local line (five tricube moments of a window in d = (x_j - x0) / h),
//            two accumulator lanes in a fixed pairing: n = 10^4 sorted points, the window k = 6667 nearest, every
//            point evaluated; ns per window point
//   loess    one local quadratic fit in two variables (six terms) on a window of q points: Householder QR of the
//            weighted design against the 21 moments and a 6 x 6 Cholesky; ns per window point
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// The local line at x0 on the sorted window [l, r): (c s0 - b s1) / (a c - b^2).
double local_line(const double* x, const double* y, std::size_t l, std::size_t r, double x0, double h) {
    double a[2] = {0, 0}, b[2] = {0, 0}, c[2] = {0, 0}, s0[2] = {0, 0}, s1[2] = {0, 0};
    const double inv = 1.0 / h;
    std::size_t j = l;
    for (; j + 2 <= r; j += 2) {
        for (int lane = 0; lane < 2; ++lane) {
            const double d = (x[j + lane] - x0) * inv;
            const double ad = std::fabs(d);
            const double t = 1.0 - ad * ad * ad;
            const double v = ad < 1.0 ? t * t * t : 0.0;
            const double vd = v * d;
            a[lane] += v;
            b[lane] += vd;
            c[lane] += vd * d;
            s0[lane] += v * y[j + lane];
            s1[lane] += vd * y[j + lane];
        }
    }
    for (; j < r; ++j) {
        const double d = (x[j] - x0) * inv;
        const double ad = std::fabs(d);
        const double t = 1.0 - ad * ad * ad;
        const double v = ad < 1.0 ? t * t * t : 0.0;
        a[0] += v;
        b[0] += v * d;
        c[0] += v * d * d;
        s0[0] += v * y[j];
        s1[0] += v * d * y[j];
    }
    const double A = a[0] + a[1], B = b[0] + b[1], C = c[0] + c[1], S0 = s0[0] + s0[1], S1 = s1[0] + s1[1];
    return (C * S0 - B * S1) / (A * C - B * B);
}

void lowess() {
    const std::size_t n = 10000, k = 6667;
    std::mt19937_64 g(20261123);
    std::uniform_real_distribution<double> u(0.0, 10.0);
    std::normal_distribution<double> z(0.0, 0.3);
    std::vector<double> x(n), y(n), out(n);
    for (auto& v : x) v = u(g);
    std::sort(x.begin(), x.end());
    for (std::size_t i = 0; i < n; ++i) y[i] = std::sin(x[i]) + z(g);
    double best = 1e300;
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now();
        std::size_t l = 0;
        for (std::size_t i = 0; i < n; ++i) {
            while (l + k < n && x[l + k] - x[i] < x[i] - x[l]) ++l;
            const double h = std::max(x[i] - x[l], x[l + k - 1] - x[i]);
            out[i] = local_line(x.data(), y.data(), l, l + k, x[i], h);
        }
        best = std::min(best, now() - t0);
    }
    std::printf("lowess_closed_form %.3f ns/window-point\n", best * 1e9 / (double(n) * double(k)));
    std::printf(
        "lowess_closed_form_fit %.1f ms (n = 10^4, k = 6667, one fit of every point, one thread)\n", best * 1e3
    );
    std::printf("lowess_checksum %.17g\n", out[n / 2]);
}

constexpr int T = 6;

// Householder QR of the q x 6 weighted design in place; the intercept of the least-squares solution.
double fit_qr(std::vector<double>& A, std::vector<double>& rhs, std::size_t q) {
    for (int c = 0; c < T; ++c) {
        double norm = 0;
        for (std::size_t i = c; i < q; ++i) norm += A[i * T + c] * A[i * T + c];
        norm = std::sqrt(norm);
        const double alpha = A[c * T + c] > 0 ? -norm : norm;
        const double v0 = A[c * T + c] - alpha;
        double vv = v0 * v0;
        for (std::size_t i = c + 1; i < q; ++i) vv += A[i * T + c] * A[i * T + c];
        if (vv == 0) continue;
        for (int cc = c + 1; cc <= T; ++cc) {
            double* col = cc < T ? nullptr : rhs.data();
            double dot = cc < T ? v0 * A[c * T + cc] : v0 * rhs[c];
            for (std::size_t i = c + 1; i < q; ++i) dot += A[i * T + c] * (col ? col[i] : A[i * T + cc]);
            const double f = 2 * dot / vv;
            if (col) {
                col[c] -= f * v0;
                for (std::size_t i = c + 1; i < q; ++i) col[i] -= f * A[i * T + c];
            } else {
                A[c * T + cc] -= f * v0;
                for (std::size_t i = c + 1; i < q; ++i) A[i * T + cc] -= f * A[i * T + c];
            }
        }
        A[c * T + c] = alpha;
    }
    double sol[T];
    for (int c = T - 1; c >= 0; --c) {
        double s = rhs[c];
        for (int cc = c + 1; cc < T; ++cc) s -= A[c * T + cc] * sol[cc];
        sol[c] = s / A[c * T + c];
    }
    return sol[0];
}

// The 21 moments and the right side in one pass, then a 6 x 6 Cholesky.
double fit_moments(const double* u1, const double* u2, const double* v, const double* y, std::size_t q) {
    double M[T][T] = {}, b[T] = {};
    for (std::size_t i = 0; i < q; ++i) {
        const double p[T] = {1.0, u1[i], u2[i], u1[i] * u1[i], u1[i] * u2[i], u2[i] * u2[i]};
        for (int a = 0; a < T; ++a) {
            const double wa = v[i] * p[a];
            b[a] += wa * y[i];
            for (int c = a; c < T; ++c) M[a][c] += wa * p[c];
        }
    }
    for (int a = 0; a < T; ++a) {
        for (int c = 0; c <= a; ++c) {
            double s = M[c][a];
            for (int k = 0; k < c; ++k) s -= M[a][k] * M[c][k];
            M[a][c] = a == c ? std::sqrt(s) : s / M[c][c];
        }
    }
    double z[T], sol[T];
    for (int a = 0; a < T; ++a) {
        double s = b[a];
        for (int k = 0; k < a; ++k) s -= M[a][k] * z[k];
        z[a] = s / M[a][a];
    }
    for (int a = T - 1; a >= 0; --a) {
        double s = z[a];
        for (int k = a + 1; k < T; ++k) s -= M[k][a] * sol[k];
        sol[a] = s / M[a][a];
    }
    return sol[0];
}

void loess() {
    for (std::size_t q : {64u, 750u, 7500u}) {
        std::mt19937_64 g(20261123 + q);
        std::uniform_real_distribution<double> u(-0.7, 0.7);
        std::normal_distribution<double> z(0.0, 0.3);
        std::vector<double> u1(q), u2(q), v(q), y(q);
        for (std::size_t i = 0; i < q; ++i) {
            u1[i] = u(g);
            u2[i] = u(g);
            const double r = std::sqrt(u1[i] * u1[i] + u2[i] * u2[i]);
            const double t = 1.0 - r * r * r;
            v[i] = t * t * t;
            y[i] = std::sin(u1[i]) + u2[i] * u2[i] + z(g);
        }
        const int reps = int(2000000 / q) + 1;
        std::vector<double> A(q * T), rhs(q);
        double tq = 1e300, tm = 1e300, rq = 0, rm = 0;
        for (int outer = 0; outer < 3; ++outer) {
            double t0 = now();
            for (int rep = 0; rep < reps; ++rep) {
                for (std::size_t i = 0; i < q; ++i) {
                    const double s = std::sqrt(v[i]);
                    const double p[T] = {1.0, u1[i], u2[i], u1[i] * u1[i], u1[i] * u2[i], u2[i] * u2[i]};
                    for (int a = 0; a < T; ++a) A[i * T + a] = s * p[a];
                    rhs[i] = s * y[i];
                }
                rq = fit_qr(A, rhs, q);
            }
            tq = std::min(tq, (now() - t0) / reps);
            t0 = now();
            for (int rep = 0; rep < reps; ++rep) rm = fit_moments(u1.data(), u2.data(), v.data(), y.data(), q);
            tm = std::min(tm, (now() - t0) / reps);
        }
        std::printf("loess_qr_q%zu %.3f ns/window-point\n", q, tq * 1e9 / double(q));
        std::printf("loess_moments_q%zu %.3f ns/window-point\n", q, tm * 1e9 / double(q));
        std::printf("loess_difference_q%zu %.3g (QR intercept - moment intercept)\n", q, rq - rm);
    }
}

}  // namespace

int main() {
    lowess();
    loess();
    return 0;
}
