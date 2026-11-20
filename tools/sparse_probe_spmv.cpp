// The cost of int64 indices in CSR SpMV and SpMM (docs/design/sparse.md §2.2): the same kernel
// instantiated with int32 and with int64 `indices` / `indptr`, on 1 thread and on 10 threads.
//
// Built and run by tools/sparse_probe.py (section spmv). By hand:
//   clang++ -std=c++20 -O3 -mcpu=native -ffp-contract=off -o spmv tools/sparse_probe_spmv.cpp
//   ./spmv [dump_dir]
//
// The kernel is the order of §2.6: y_i = sum_j a_ij x_j over the row's stored entries in ascending
// j, one std::fma per term starting from 0. SpMM (k = 16, X row-major) runs the same sum for every
// output element, the k lanes side by side. The 10-thread runs split the rows into fixed ranges of
// about 2^14 stored entries (from indptr only) on a small persistent spinning pool, so every row
// is summed by one task in the serial order and both index widths give the same bits as 1 thread.
//
// Matrices: random n x n with exactly r entries per row (r = 3, 10, 100; distinct columns drawn
// uniformly, then sorted) at nnz = 1e5, 1e6, 1e7, and the 2-D 5-point Laplacian on a 1000 x 1000
// grid (n = 1e6, nnz ~5e6). x is uniform in [-1, 1).
//
// Timing: int32 and int64 runs alternate in rounds (so a burst of noise on the shared machine
// hits both), each round is one call; reported are the median and the best over the rounds.
// The ratio is int64 / int32 of the medians.
//
// With dump_dir, the matrices of nnz <= 1e6 and their x, y are written there as raw little-endian
// files (<name>.indptr/.indices int64, .data/.x/.y float64, .meta text) for tools/sparse_probe.py, which
// compares y with scipy's csr_array @ x bit for bit.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

struct SplitMix {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }  // [0, 1)
};

struct Csr {
    std::string name;
    std::int64_t n = 0;
    std::vector<std::int64_t> indptr, indices;
    std::vector<double> data;
};

Csr random_matrix(std::int64_t nnz, int r, std::uint64_t seed) {
    Csr a;
    a.n = nnz / r;
    a.name = "random_r" + std::to_string(r) + "_nnz" + std::to_string(nnz);
    SplitMix g{seed};
    a.indptr.assign(static_cast<std::size_t>(a.n + 1), 0);
    a.indices.reserve(static_cast<std::size_t>(a.n * r));
    a.data.reserve(static_cast<std::size_t>(a.n * r));
    std::vector<std::int64_t> row;
    for (std::int64_t i = 0; i < a.n; ++i) {
        row.clear();
        while (static_cast<int>(row.size()) < r) {
            std::int64_t j = static_cast<std::int64_t>(g.next() % static_cast<std::uint64_t>(a.n));
            if (std::find(row.begin(), row.end(), j) == row.end()) row.push_back(j);
        }
        std::sort(row.begin(), row.end());
        for (auto j : row) {
            a.indices.push_back(j);
            a.data.push_back(2.0 * g.uniform() - 1.0);
        }
        a.indptr[static_cast<std::size_t>(i + 1)] = static_cast<std::int64_t>(a.indices.size());
    }
    return a;
}

Csr laplacian2d(std::int64_t grid) {
    Csr a;
    a.n = grid * grid;
    a.name = "laplacian2d_" + std::to_string(grid);
    a.indptr.assign(static_cast<std::size_t>(a.n + 1), 0);
    for (std::int64_t i = 0; i < a.n; ++i) {
        std::int64_t x = i % grid, y = i / grid;
        auto add = [&](std::int64_t j, double v) {
            a.indices.push_back(j);
            a.data.push_back(v);
        };
        if (y > 0) add(i - grid, -1.0);
        if (x > 0) add(i - 1, -1.0);
        add(i, 4.0);
        if (x < grid - 1) add(i + 1, -1.0);
        if (y < grid - 1) add(i + grid, -1.0);
        a.indptr[static_cast<std::size_t>(i + 1)] = static_cast<std::int64_t>(a.indices.size());
    }
    return a;
}

template <class I>
struct CsrI {
    std::int64_t n;
    std::vector<I> indptr, indices;
    const std::vector<double>* data;
};

template <class I>
CsrI<I> narrow(const Csr& a) {
    CsrI<I> b{a.n, {}, {}, &a.data};
    b.indptr.assign(a.indptr.begin(), a.indptr.end());
    b.indices.assign(a.indices.begin(), a.indices.end());
    return b;
}

template <class I>
void spmv_rows(const CsrI<I>& a, const double* x, double* y, std::int64_t r0, std::int64_t r1) {
    const I* ip = a.indptr.data();
    const I* ix = a.indices.data();
    const double* d = a.data->data();
    for (std::int64_t i = r0; i < r1; ++i) {
        double acc = 0.0;
        for (I p = ip[i]; p < ip[i + 1]; ++p) acc = std::fma(d[p], x[ix[p]], acc);
        y[i] = acc;
    }
}

constexpr int K = 16;

template <class I>
void spmm_rows(const CsrI<I>& a, const double* x, double* y, std::int64_t r0, std::int64_t r1) {
    const I* ip = a.indptr.data();
    const I* ix = a.indices.data();
    const double* d = a.data->data();
    for (std::int64_t i = r0; i < r1; ++i) {
        double acc[K] = {};
        for (I p = ip[i]; p < ip[i + 1]; ++p) {
            const double v = d[p];
            const double* xr = x + static_cast<std::int64_t>(ix[p]) * K;
            for (int c = 0; c < K; ++c) acc[c] = std::fma(v, xr[c], acc[c]);
        }
        std::memcpy(y + i * K, acc, sizeof acc);
    }
}

// A persistent spinning pool: run(ntasks, f) has the caller and the workers pull task numbers
// from one counter until none are left.
class Pool {
public:
    explicit Pool(int nthreads) {
        for (int t = 1; t < nthreads; ++t) workers_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        stop_.store(true);
        gen_.fetch_add(1);
        for (auto& w : workers_) w.join();
    }
    void run(std::int64_t ntasks, const std::function<void(std::int64_t)>& f) {
        f_ = &f;
        ntasks_ = ntasks;
        next_.store(0);
        done_.store(0);
        gen_.fetch_add(1, std::memory_order_release);
        work();
        while (done_.load(std::memory_order_acquire) < ntasks) {}
    }

private:
    void work() {
        for (;;) {
            std::int64_t t = next_.fetch_add(1);
            if (t >= ntasks_) return;
            (*f_)(t);
            done_.fetch_add(1, std::memory_order_release);
        }
    }
    void loop() {
        std::uint64_t seen = gen_.load();
        for (;;) {
            std::uint64_t g;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) {}
            seen = g;
            if (stop_.load()) return;
            work();
        }
    }
    std::vector<std::thread> workers_;
    std::atomic<std::uint64_t> gen_{0};
    std::atomic<std::int64_t> next_{0}, done_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(std::int64_t)>* f_ = nullptr;
    std::int64_t ntasks_ = 0;
};

// Row ranges of about 2^14 stored entries each, from indptr only.
std::vector<std::int64_t> ranges(const Csr& a) {
    std::vector<std::int64_t> cut{0};
    const std::int64_t step = 1 << 14;
    std::int64_t target = step;
    for (std::int64_t i = 1; i <= a.n; ++i) {
        if (a.indptr[static_cast<std::size_t>(i)] >= target || i == a.n) {
            cut.push_back(i);
            target = a.indptr[static_cast<std::size_t>(i)] + step;
        }
    }
    return cut;
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Stat {
    double median, best;
};

Stat stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front()};
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

void write_raw(const std::string& path, const void* p, std::size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(p, 1, bytes, f);
    std::fclose(f);
}

void measure(const Csr& a, Pool& pool, const char* dump_dir) {
    auto a32 = narrow<std::int32_t>(a);
    auto a64 = narrow<std::int64_t>(a);
    const auto cut = ranges(a);
    const std::int64_t ntasks = static_cast<std::int64_t>(cut.size()) - 1;
    const std::int64_t nnz = a.indptr.back();

    SplitMix g{12345};
    // SpMM only up to n = 2e6 rows: X and the three outputs are 128 bytes a row each
    const bool mm = a.n <= 2000000;
    std::vector<double> x(static_cast<std::size_t>(a.n)), xk(static_cast<std::size_t>(mm ? a.n * K : 0));
    for (auto& v : x) v = 2.0 * g.uniform() - 1.0;
    for (auto& v : xk) v = 2.0 * g.uniform() - 1.0;
    std::vector<double> y32(x.size()), y64(x.size()), y32p(x.size()), y64p(x.size());
    std::vector<double> z32(xk.size()), z64(xk.size()), z64p(xk.size());

    // rounds: enough for ~0.4 s per (kernel, width, threads) at the first estimate, 7..400
    double t0 = now();
    spmv_rows(a64, x.data(), y64.data(), 0, a.n);
    double one = std::max(now() - t0, 1e-6);
    int rounds = static_cast<int>(std::clamp(0.4 / one, 7.0, 400.0));
    int rounds_mm = std::clamp(rounds / 10, 5, 60);

    auto serial = [&](auto& m, std::vector<double>& y) {
        double s = now();
        spmv_rows(m, x.data(), y.data(), 0, a.n);
        return now() - s;
    };
    auto parallel = [&](auto& m, std::vector<double>& y) {
        std::function<void(std::int64_t)> f = [&](std::int64_t t) {
            spmv_rows(m, x.data(), y.data(), cut[static_cast<std::size_t>(t)], cut[static_cast<std::size_t>(t + 1)]);
        };
        double s = now();
        pool.run(ntasks, f);
        return now() - s;
    };
    auto serial_mm = [&](auto& m, std::vector<double>& z) {
        double s = now();
        spmm_rows(m, xk.data(), z.data(), 0, a.n);
        return now() - s;
    };
    auto parallel_mm = [&](auto& m, std::vector<double>& z) {
        std::function<void(std::int64_t)> f = [&](std::int64_t t) {
            spmm_rows(m, xk.data(), z.data(), cut[static_cast<std::size_t>(t)], cut[static_cast<std::size_t>(t + 1)]);
        };
        double s = now();
        pool.run(ntasks, f);
        return now() - s;
    };

    std::vector<double> s32, s64, p32, p64, m32, m64, q32, q64;
    for (int r = 0; r < rounds; ++r) {
        s32.push_back(serial(a32, y32));
        s64.push_back(serial(a64, y64));
        p32.push_back(parallel(a32, y32p));
        p64.push_back(parallel(a64, y64p));
    }
    for (int r = 0; mm && r < rounds_mm; ++r) {
        m32.push_back(serial_mm(a32, z32));
        m64.push_back(serial_mm(a64, z64));
        q32.push_back(parallel_mm(a32, z64p));
        q64.push_back(parallel_mm(a64, z64p));
    }
    const bool bits = same_bits(y32, y64) && same_bits(y64, y64p) && same_bits(y32p, y64) && same_bits(z32, z64) &&
                      same_bits(z64, z64p);
    auto S32 = stats(s32), S64 = stats(s64), P32 = stats(p32), P64 = stats(p64);
    if (!mm) m32 = m64 = q32 = q64 = {std::nan("")};
    auto M32 = stats(m32), M64 = stats(m64), Q32 = stats(q32), Q64 = stats(q64);
    std::printf(
        "%-26s n=%9lld nnz=%9lld | SpMV 1T  i32 %8.3f (%8.3f) i64 %8.3f (%8.3f) ratio %.3f | 10T i32 %7.3f i64 %7.3f "
        "ratio %.3f | SpMM16 1T i32 %8.2f i64 %8.2f ratio %.3f | 10T i32 %7.2f i64 %7.2f ratio %.3f | bits %s | "
        "rounds %d/%d\n",
        a.name.c_str(), static_cast<long long>(a.n), static_cast<long long>(nnz), S32.median * 1e3, S32.best * 1e3,
        S64.median * 1e3, S64.best * 1e3, S64.median / S32.median, P32.median * 1e3, P64.median * 1e3,
        P64.median / P32.median, M32.median * 1e3, M64.median * 1e3, M64.median / M32.median, Q32.median * 1e3,
        Q64.median * 1e3, Q64.median / Q32.median, bits ? "same" : "DIFFER", rounds, rounds_mm
    );
    std::fflush(stdout);

    if (dump_dir && nnz <= 1000000) {
        std::string p = std::string(dump_dir) + "/" + a.name;
        write_raw(p + ".indptr", a.indptr.data(), a.indptr.size() * 8);
        write_raw(p + ".indices", a.indices.data(), a.indices.size() * 8);
        write_raw(p + ".data", a.data.data(), a.data.size() * 8);
        write_raw(p + ".x", x.data(), x.size() * 8);
        write_raw(p + ".y", y64.data(), y64.size() * 8);
        write_raw(p + ".xk", xk.data(), xk.size() * 8);
        write_raw(p + ".zk", z64.data(), z64.size() * 8);
        FILE* f = std::fopen((p + ".meta").c_str(), "w");
        if (f) {
            std::fprintf(f, "%lld %lld %d\n", static_cast<long long>(a.n), static_cast<long long>(nnz), K);
            std::fclose(f);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* dump_dir = argc > 1 ? argv[1] : nullptr;
    Pool pool(10);
    std::printf("times in ms: median (best) over alternating rounds; ratio = int64 / int32 medians\n");
    for (std::int64_t nnz : {100000LL, 1000000LL, 10000000LL}) {
        for (int r : {3, 10, 100}) {
            Csr a =
                random_matrix(nnz, r, 20261120ull + static_cast<std::uint64_t>(nnz) + static_cast<std::uint64_t>(r));
            measure(a, pool, dump_dir);
        }
    }
    measure(laplacian2d(300), pool, dump_dir);
    measure(laplacian2d(1000), pool, dump_dir);
    return 0;
}
