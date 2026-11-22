// Timing probes of sd.spatial's kernels (docs/design/spatial.md §3.1, §3.2, §3.5; decision SPT22),
// run by tools/spatial_probe.py (sections kernels, ddot, gemm). Built with sd's flags
// (-O3 -mcpu=native -ffp-contract=off) and Accelerate; one process, the calling thread unless
// a section says otherwise.
//
//   kernels DIR   the two pair orders of tools/spatial_probe_pair_kernel.hpp in a cdist-like loop
//                 (XB in blocks that stay in L2, each block against every XA row), one thread,
//                 best of several runs, per n; checks that the batched kernels give the one-pair
//                 kernels' bits, and dumps small cases into DIR for the probe to compare with
//                 sd.linalg.vector_norm(x - y)
//   ddot          whether cblas_ddot(z, z) depends on where z is in memory: the same values at
//                 32 addresses 128 bytes apart (and at page-aligned ones), per length
//   gemm          the screening product of §3.2 (C = A B^T, 256 x 2048 tiles) called from T
//                 threads at once, T = 1, 2, 4, 6, 10: the aggregate rate (AMX is shared by a
//                 cluster's cores)
#include "spatial_probe_pair_kernel.hpp"

#include <Accelerate/Accelerate.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace sd::spatial::proto;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<double> normal(std::int64_t count, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::normal_distribution<double> d;
    std::vector<double> v(static_cast<std::size_t>(count));
    for (auto& x : v) x = d(g);
    return v;
}

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// Best wall time of fn over runs that together take at least `budget` seconds (at least 3 runs).
template <class F>
double best_time(F&& fn, double budget = 0.4) {
    double best = 1e300, total = 0.;
    for (int r = 0; r < 3 || total < budget; ++r) {
        const auto t0 = Clock::now();
        fn();
        const double t = seconds_since(t0);
        best = std::min(best, t);
        total += t;
        if (r > 200) break;
    }
    return best;
}

std::int64_t block_rows(std::int64_t n) {
    // XB rows per block: about 192 KiB of doubles, a multiple of 8, at least 8
    std::int64_t b = (192 * 1024 / 8) / std::max<std::int64_t>(n, 1);
    b = std::max<std::int64_t>(8, b - b % 8);
    return b;
}

// cdist in NormOrder with P pairs per pass (n >= 16); below 16 terms the tile kernel is the order.
template <int P>
void cdist_norm(const double* xa, std::int64_t ma, const double* xb, std::int64_t mb, std::int64_t n, double* out) {
    std::vector<double> zbuf(static_cast<std::size_t>(n > kBlasSqMin ? n : 1));
    const std::int64_t bj = block_rows(n);
    for (std::int64_t j0 = 0; j0 < mb; j0 += bj) {
        const std::int64_t j1 = std::min(mb, j0 + bj);
        for (std::int64_t i = 0; i < ma; ++i) {
            const double* x = xa + i * n;
            for (std::int64_t j = j0; j < j1; j += P) {
                const double* y[P];
                for (int p = 0; p < P; ++p) y[p] = xb + (j + p) * n;
                double s[P];
                norm_order<P>(x, y, n, s, zbuf.data());
                for (int p = 0; p < P; ++p) out[i * mb + j + p] = std::sqrt(s[p]);
            }
        }
    }
}

// cdist in TileOrder (ma a multiple of 4, mb of 8), panels packed per block (the packing timed).
void cdist_tile(const double* xa, std::int64_t ma, const double* xb, std::int64_t mb, std::int64_t n, double* out) {
    const std::int64_t bj = block_rows(n);
    std::vector<double> panels(static_cast<std::size_t>(bj * n));
    double tile[32];
    for (std::int64_t j0 = 0; j0 < mb; j0 += bj) {
        const std::int64_t j1 = std::min(mb, j0 + bj);
        for (std::int64_t j = j0; j < j1; j += 8)
            pack_panel(xb + j * n, n, std::min<std::int64_t>(8, j1 - j), n, panels.data() + (j - j0) * n);
        for (std::int64_t i = 0; i < ma; i += 4) {
            const double* x[4] = {xa + i * n, xa + (i + 1) * n, xa + (i + 2) * n, xa + (i + 3) * n};
            for (std::int64_t j = j0; j < j1; j += 8) {
                tile_4x8(x, panels.data() + (j - j0) * n, n, tile);
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 8; ++c) out[(i + r) * mb + j + c] = std::sqrt(tile[r * 8 + c]);
            }
        }
    }
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

void write(const std::string& path, const std::vector<double>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(v.data(), sizeof(double), v.size(), f);
    std::fclose(f);
}

int kernels(const char* dir) {
    std::printf("kernels: one thread; best of >= 3 runs; times in ms; ratio = NormOrder / TileOrder\n");
    std::printf(
        "%6s %6s %6s %10s %10s %10s %7s  %s\n", "n", "mA", "mB", "norm(1x2)", "norm(1x1)", "tile(4x8)", "ratio",
        "checks"
    );
    const std::int64_t ns[] = {3, 8, 15, 16, 17, 24, 31, 32, 48, 64, 96, 100, 128, 256, 512, 513, 1024, 2048, 4096};
    for (std::int64_t n : ns) {
        const std::int64_t m = n <= 512 ? 1024 : (n <= 2048 ? 512 : 256);
        const auto xa = normal(m * n, 1000 + n), xb = normal(m * n, 2000 + n);
        std::vector<double> o2(m * m), o1(m * m), ot(m * m);
        double t2 = 0., t1 = 0.;
        if (n >= 16) {
            t2 = best_time([&] { cdist_norm<2>(xa.data(), m, xb.data(), m, n, o2.data()); });
            t1 = best_time([&] { cdist_norm<1>(xa.data(), m, xb.data(), m, n, o1.data()); });
        }
        const double tt = best_time([&] { cdist_tile(xa.data(), m, xb.data(), m, n, ot.data()); });
        // checks: the batched kernels against the one-pair kernels, bit for bit
        bool ok_norm = n < 16 || same_bits(o1, o2);
        bool ok_tile = true;
        std::int64_t same_as_norm = 0;
        for (std::int64_t i = 0; i < m; i += 7)
            for (std::int64_t j = 0; j < m; j += 5) {
                const double s = tile_order(xa.data() + i * n, xb.data() + j * n, n);
                ok_tile &= std::sqrt(s) == ot[i * m + j];
                if (n >= 16) same_as_norm += ot[i * m + j] == o1[i * m + j];
            }
        char checks[160];
        std::snprintf(
            checks, sizeof checks, "norm 1x2 == 1x1: %s; tile == one-pair: %s", ok_norm ? "yes" : "NO",
            ok_tile ? "yes" : "NO"
        );
        if (n >= 16)
            std::printf(
                "%6lld %6lld %6lld %10.3f %10.3f %10.3f %7.2f  %s\n", (long long)n, (long long)m, (long long)m,
                t2 * 1e3, t1 * 1e3, tt * 1e3, std::min(t1, t2) / tt, checks
            );
        else
            std::printf(
                "%6lld %6lld %6lld %10s %10s %10.3f %7s  %s (n < 16: one order)\n", (long long)n, (long long)m,
                (long long)m, "-", "-", tt * 1e3, "1.00", checks
            );
        (void)same_as_norm;
        std::fflush(stdout);
        // a small dump for the bit check against sd.linalg.vector_norm
        const std::int64_t k = 16;
        std::vector<double> sa(xa.begin(), xa.begin() + k * n), sb(xb.begin(), xb.begin() + k * n), dn(k * k),
            dt(k * k);
        if (n >= 16) cdist_norm<2>(sa.data(), k, sb.data(), k, n, dn.data());
        cdist_tile(sa.data(), k, sb.data(), k, n, dt.data());
        if (n < 16) dn = dt;
        const std::string base = std::string(dir) + "/n" + std::to_string(n);
        write(base + ".xa", sa);
        write(base + ".xb", sb);
        write(base + ".norm", dn);
        write(base + ".tile", dt);
    }
    return 0;
}

int ddot() {
    std::printf("ddot: cblas_ddot(z, z) of the same values at 32 addresses 128 bytes apart and 8 page-aligned ones\n");
    std::printf("%7s %18s %18s\n", "n", "distinct (128 B)", "distinct (page)");
    const std::int64_t ns[] = {512, 896, 897, 1024, 2048, 2049, 4096, 10000, 16384};
    for (std::int64_t n : ns) {
        const auto z = normal(n, 77 + n);
        const std::size_t page = 16384;
        const std::size_t bytes = (static_cast<std::size_t>(n) * 8 + 32 * 128 + page) * 9;
        void* raw = nullptr;
        if (posix_memalign(&raw, page, bytes)) return 1;
        auto* base = static_cast<unsigned char*>(raw);
        std::set<double> a128, apage;
        for (int k = 0; k < 32; ++k) {
            auto* p = reinterpret_cast<double*>(base + 128 * k);
            std::memcpy(p, z.data(), n * 8);
            a128.insert(cblas_ddot(n, p, 1, p, 1));
        }
        const std::size_t stride = ((static_cast<std::size_t>(n) * 8 + page - 1) / page) * page;
        for (int k = 0; k < 8; ++k) {
            auto* p = reinterpret_cast<double*>(base + stride * k);
            std::memcpy(p, z.data(), n * 8);
            apage.insert(cblas_ddot(n, p, 1, p, 1));
        }
        std::printf("%7lld %18zu %18zu\n", (long long)n, a128.size(), apage.size());
        std::free(raw);
    }
    return 0;
}

int gemm() {
    const char* mode = std::getenv("VECLIB_MAXIMUM_THREADS");
    std::printf(
        "gemm: C (256 x 2048) = A (256 x n) B^T (2048 x n), from T threads at once; VECLIB_MAXIMUM_THREADS=%s\n",
        mode ? mode : "(unset)"
    );
    std::printf("%5s %4s %12s %14s %10s\n", "n", "T", "ms per call", "GFLOP/s total", "speedup");
    const int mrows = 256, ncols = 2048;
    for (int n : {16, 64, 512}) {
        double base_rate = 0.;
        for (int threads : {1, 2, 4, 6, 10}) {
            const int calls = n <= 64 ? 200 : 40;
            std::vector<std::vector<double>> as, bs, cs;
            for (int t = 0; t < threads; ++t) {
                as.push_back(normal(mrows * n, 10 + t));
                bs.push_back(normal(ncols * n, 20 + t));
                cs.emplace_back(static_cast<std::size_t>(mrows) * ncols);
            }
            double best = 1e300;
            for (int rep = 0; rep < 3; ++rep) {
                const auto t0 = Clock::now();
                std::vector<std::thread> pool;
                for (int t = 0; t < threads; ++t)
                    pool.emplace_back([&, t] {
                        for (int c = 0; c < calls; ++c)
                            cblas_dgemm(
                                CblasRowMajor, CblasNoTrans, CblasTrans, mrows, ncols, n, 1.0, as[t].data(), n,
                                bs[t].data(), n, 0.0, cs[t].data(), ncols
                            );
                    });
                for (auto& th : pool) th.join();
                best = std::min(best, seconds_since(t0));
            }
            const double flops = 2.0 * mrows * ncols * n * calls * threads;
            const double rate = flops / best / 1e9;
            if (threads == 1) base_rate = rate;
            std::printf("%5d %4d %12.4f %14.1f %10.2f\n", n, threads, best / calls * 1e3, rate, rate / base_rate);
            std::fflush(stdout);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s kernels DIR | ddot | gemm\n", argv[0]);
        return 2;
    }
    const std::string what = argv[1];
    if (what == "kernels" && argc == 3) return kernels(argv[2]);
    if (what == "ddot") return ddot();
    if (what == "gemm") return gemm();
    std::fprintf(stderr, "unknown section %s\n", argv[1]);
    return 2;
}
