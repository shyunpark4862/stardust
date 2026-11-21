// Probe of Accelerate's Sparse Solvers (Solve.h) for sd.sparse (docs/design/sparse.md §1.4, §3.6):
// which factorization / ordering pairs give the same bits run after run, how singular matrices are
// reported, and how fast and large the direct factorizations are.
//
// Built and run by tools/sparse_probe.py (sections det, conc, sing, speed; each in Accelerate's
// default thread mode and with VECLIB_MAXIMUM_THREADS=1). By hand:
//   clang++ -std=c++20 -O3 -mcpu=native -ffp-contract=off -o probe tools/sparse_probe_accelerate.cpp \
//       -framework Accelerate
//   ./probe det|conc|sing|speed|iter
//
// Sections (one line per case, tab-free, prefixed by the section name):
//   det    (DET_FROM=kind:size resumes at that matrix) every factorization x ordering on every matrix kind and size: REPS factor + solve runs in
//          one process; "distinct" counts the different solution byte patterns over the runs,
//          "resolve" the patterns of three solves with the first run's factor, "maxdiff" the
//          largest normwise relative difference to the first run, "relres" ||b - Ax|| / ||b||
//          (least squares: ||A^T r|| / ||A^T b||), "fact" the mean factor time, "peak" the peak
//          bytes Accelerate allocated through the options' malloc for the factor.
//   conc   the same factor + solve on 4 std::threads at once, 5 runs each, against the serial bytes
//          (`conc AMD` runs only the cases of one ordering).
//   sing   structurally and numerically singular matrices: status, zeroTolerance 0 vs 1e-14, inf or
//          NaN in the solution; and messages the reportError callback receives.
//   speed  best-of-3 factor and solve times of LU/AMD, LU/Metis, LU/MTMetis and Cholesky/AMD on
//          2-D and 3-D Laplacians (the matrices of bench/speed/cases/sparse.py's splu cases).
//   iter   best-of-3 times of Accelerate's iterative solvers on the grids of the iterative speed
//          cases (bench/speed/cases/sparse.py, SP5) and their settings: CG (rtol 1e-10), restarted
//          GMRES and DQGMRES (20 vectors, rtol 1e-8), LSMR (Fong-Saunders test, atol = btol =
//          1e-8), from x0 = 0 (so Accelerate's test relative to the first residual is scipy's,
//          relative to ||b||); the true relative residual ||b - Ax|| / ||b|| of the result.
//
// The reportError callback is always set: with NULL, a parameter error stops the process with
// __builtin_trap().

#include <Accelerate/Accelerate.h>
#include <malloc/malloc.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using cplx = std::complex<double>;

// ---------------------------------------------------------------- allocation and error hooks

std::atomic<long> g_bytes{0}, g_peak{0};

void* counting_malloc(size_t n) {
    void* p = std::malloc(n);
    if (p) {
        long now = g_bytes.fetch_add(static_cast<long>(malloc_size(p))) + static_cast<long>(malloc_size(p));
        long old = g_peak.load();
        while (now > old && !g_peak.compare_exchange_weak(old, now)) {}
    }
    return p;
}

void counting_free(void* p) {
    if (p) g_bytes.fetch_sub(static_cast<long>(malloc_size(p)));
    std::free(p);
}

thread_local std::string t_error;

void report_error(const char* message) { t_error = message; }

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

long max_rss_mb() {
    rusage u{};
    getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss / (1024 * 1024);  // bytes on macOS
}

// ---------------------------------------------------------------- matrices (CSC, int32 rows)

template <class T>
struct Csc {
    std::string kind;
    int m = 0, n = 0;
    std::vector<long> cs;
    std::vector<int> ri;
    std::vector<T> v;
    bool symmetric = false;  // the values are symmetric (or Hermitian)
};

struct SplitMix {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    double sym() { return 2.0 * uniform() - 1.0; }
    int below(int k) { return static_cast<int>(next() % static_cast<std::uint64_t>(k)); }
};

template <class T>
struct Triplets {
    std::vector<int> r, c;
    std::vector<T> v;
    void add(int i, int j, T x) {
        r.push_back(i);
        c.push_back(j);
        v.push_back(x);
    }
};

// CSC with sorted rows; duplicates summed in input order.
template <class T>
Csc<T> from_triplets(std::string kind, int m, int n, const Triplets<T>& t) {
    Csc<T> a;
    a.kind = std::move(kind);
    a.m = m;
    a.n = n;
    std::vector<long> order(t.r.size());
    for (size_t k = 0; k < order.size(); ++k) order[k] = static_cast<long>(k);
    std::stable_sort(order.begin(), order.end(), [&](long x, long y) {
        return t.c[x] != t.c[y] ? t.c[x] < t.c[y] : t.r[x] < t.r[y];
    });
    a.cs.assign(static_cast<size_t>(n + 1), 0);
    for (size_t q = 0; q < order.size(); ++q) {
        long k = order[q];
        bool dup = q > 0 && t.c[order[q - 1]] == t.c[k] && t.r[order[q - 1]] == t.r[k];
        if (dup) {
            a.v.back() += t.v[k];
        } else {
            a.ri.push_back(t.r[k]);
            a.v.push_back(t.v[k]);
            a.cs[static_cast<size_t>(t.c[k] + 1)]++;
        }
    }
    for (int j = 0; j < n; ++j) a.cs[static_cast<size_t>(j + 1)] += a.cs[static_cast<size_t>(j)];
    return a;
}

Csc<double> laplacian2d(int g, bool neumann = false, double shift = 0.0) {
    Triplets<double> t;
    for (int y = 0; y < g; ++y)
        for (int x = 0; x < g; ++x) {
            int i = y * g + x, deg = 0;
            auto nb = [&](int xx, int yy) {
                if (xx < 0 || yy < 0 || xx >= g || yy >= g) return;
                t.add(i, yy * g + xx, -1.0);
                ++deg;
            };
            nb(x - 1, y), nb(x + 1, y), nb(x, y - 1), nb(x, y + 1);
            t.add(i, i, (neumann ? deg : 4.0) + shift);
        }
    auto a = from_triplets(neumann ? (shift > 0 ? "nearsing2d" : "neumann2d") : "lap2d", g * g, g * g, t);
    a.symmetric = true;
    return a;
}

Csc<double> laplacian3d(int g) {
    Triplets<double> t;
    for (int z = 0; z < g; ++z)
        for (int y = 0; y < g; ++y)
            for (int x = 0; x < g; ++x) {
                int i = (z * g + y) * g + x;
                auto nb = [&](int xx, int yy, int zz) {
                    if (xx < 0 || yy < 0 || zz < 0 || xx >= g || yy >= g || zz >= g) return;
                    t.add(i, (zz * g + yy) * g + xx, -1.0);
                };
                nb(x - 1, y, z), nb(x + 1, y, z), nb(x, y - 1, z), nb(x, y + 1, z), nb(x, y, z - 1), nb(x, y, z + 1);
                t.add(i, i, 6.0);
            }
    auto a = from_triplets("lap3d", g * g * g, g * g * g, t);
    a.symmetric = true;
    return a;
}

// Diagonally dominant 5-point convection-diffusion: off-diagonals sum to 3.3 < 4.
Csc<double> convdiff2d(int g) {
    Triplets<double> t;
    for (int y = 0; y < g; ++y)
        for (int x = 0; x < g; ++x) {
            int i = y * g + x;
            if (y > 0) t.add(i, i - g, -1.1);
            if (x > 0) t.add(i, i - 1, -0.6);
            t.add(i, i, 4.0);
            if (x < g - 1) t.add(i, i + 1, -0.8);
            if (y < g - 1) t.add(i, i + g, -0.8);
        }
    return from_triplets("convdiff2d", g * g, g * g, t);
}

// r random off-diagonals per row in [-1, 1), diagonal = |row| sum + 1. The columns are drawn from
// a window of half-width `window` around the diagonal (0: the whole row); without one, a random
// pattern of n ~ 1e5 fills its factors to an almost dense matrix (tens of GB).
Csc<double> random_dd(int n, int r, std::uint64_t seed, int window = 0) {
    SplitMix g{seed};
    Triplets<double> t;
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int k = 0; k < r; ++k) {
            int j = window ? std::clamp(i - window + g.below(2 * window + 1), 0, n - 1) : g.below(n);
            double v = g.sym();
            if (j == i) continue;
            t.add(i, j, v);
            s += std::fabs(v);
        }
        t.add(i, i, s + 1.0);
    }
    return from_triplets("random_dd", n, n, t);
}

// Unsymmetric band of half-width bw, diagonally dominant.
Csc<double> banded(int n, int bw, std::uint64_t seed) {
    SplitMix g{seed};
    Triplets<double> t;
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int j = std::max(0, i - bw); j <= std::min(n - 1, i + bw); ++j)
            if (j != i) {
                double v = g.sym();
                t.add(i, j, v);
                s += std::fabs(v);
            }
        t.add(i, i, s + 0.5);
    }
    return from_triplets("banded", n, n, t);
}

// Dense b x b random blocks on the diagonal (pivoting needed: no dominance).
Csc<double> blockdiag(int n, int b, std::uint64_t seed) {
    SplitMix g{seed};
    Triplets<double> t;
    for (int s = 0; s < n; s += b)
        for (int i = s; i < std::min(n, s + b); ++i)
            for (int j = s; j < std::min(n, s + b); ++j) t.add(i, j, g.sym());
    return from_triplets("blockdiag", n, n, t);
}

// Saddle point [[L, B^T], [B, 0]]: L the 2-D Laplacian on g x g, B with 2 entries per row and
// g*g/4 rows (row i touches grid nodes within 100 of 4i). Symmetric indefinite with a zero
// diagonal block.
Csc<double> saddle(int g, std::uint64_t seed) {
    SplitMix r{seed};
    int n1 = g * g, m1 = n1 / 4;
    auto L = laplacian2d(g);
    Triplets<double> t;
    for (int j = 0; j < n1; ++j)
        for (long p = L.cs[j]; p < L.cs[j + 1]; ++p) t.add(L.ri[p], j, L.v[p]);
    for (int i = 0; i < m1; ++i)
        for (int k = 0; k < 2; ++k) {  // grid nodes near 4i: a local coupling keeps the fill bounded
            int j = std::clamp(4 * i - 100 + r.below(201), 0, n1 - 1);
            double v = 0.5 + r.uniform();
            t.add(n1 + i, j, v);
            t.add(j, n1 + i, v);
        }
    auto a = from_triplets("saddle", n1 + m1, n1 + m1, t);
    a.symmetric = true;
    return a;
}

// Overdetermined (3n/2) x n: an identity on top for full column rank, 4 random entries a column
// (rows from a window of half-width `window` around 3j/2 when it is not 0, as random_dd).
Csc<double> rect(int n, std::uint64_t seed, int window = 0) {
    SplitMix g{seed};
    int m = n + n / 2;
    Triplets<double> t;
    for (int j = 0; j < n; ++j) {
        t.add(j, j, 1.0);
        const int c = j + j / 2;
        for (int k = 0; k < 4; ++k)
            t.add(window ? std::clamp(c - window + g.below(2 * window + 1), 0, m - 1) : g.below(m), j, g.sym());
    }
    return from_triplets("rect", m, n, t);
}

// A complex shifted 2-D Laplacian: Helmholtz-like L - (0.5 - 0.3i) I with an unsymmetric
// imaginary convection term.
Csc<cplx> complex2d(int g) {
    Triplets<cplx> t;
    for (int y = 0; y < g; ++y)
        for (int x = 0; x < g; ++x) {
            int i = y * g + x;
            if (y > 0) t.add(i, i - g, cplx(-1.0, 0.2));
            if (x > 0) t.add(i, i - 1, cplx(-1.0, 0.1));
            t.add(i, i, cplx(3.5, 0.3));
            if (x < g - 1) t.add(i, i + 1, cplx(-1.0, -0.1));
            if (y < g - 1) t.add(i, i + g, cplx(-1.0, -0.2));
        }
    return from_triplets("complex2d", g * g, g * g, t);
}

template <class T>
Csc<T> lower_triangle(const Csc<T>& a) {
    Csc<T> l;
    l.kind = a.kind;
    l.m = a.m;
    l.n = a.n;
    l.symmetric = true;
    l.cs.assign(static_cast<size_t>(a.n + 1), 0);
    for (int j = 0; j < a.n; ++j) {
        for (long p = a.cs[j]; p < a.cs[j + 1]; ++p)
            if (a.ri[p] >= j) {
                l.ri.push_back(a.ri[p]);
                l.v.push_back(a.v[p]);
            }
        l.cs[static_cast<size_t>(j + 1)] = static_cast<long>(l.ri.size());
    }
    return l;
}

// ---------------------------------------------------------------- factor and solve

const char* fact_name(SparseFactorization_t f) {
    switch (f) {
        case SparseFactorizationCholesky: return "Cholesky";
        case SparseFactorizationLDLT: return "LDLT";
        case SparseFactorizationLDLTUnpivoted: return "LDLTUnpivoted";
        case SparseFactorizationLDLTSBK: return "LDLTSBK";
        case SparseFactorizationLDLTTPP: return "LDLTTPP";
        case SparseFactorizationQR: return "QR";
        case SparseFactorizationCholeskyAtA: return "CholeskyAtA";
        case SparseFactorizationLU: return "LU";
        case SparseFactorizationLUUnpivoted: return "LUUnpivoted";
        case SparseFactorizationLUSPP: return "LUSPP";
        case SparseFactorizationLUTPP: return "LUTPP";
        default: return "?";
    }
}

const char* order_name(SparseOrder_t o) {
    switch (o) {
        case SparseOrderDefault: return "Default";
        case SparseOrderAMD: return "AMD";
        case SparseOrderMetis: return "Metis";
        case SparseOrderCOLAMD: return "COLAMD";
        case SparseOrderMTMetis: return "MTMetis";
        default: return "?";
    }
}

bool symmetric_type(SparseFactorization_t f) {
    return f == SparseFactorizationCholesky || f == SparseFactorizationLDLT || f == SparseFactorizationLDLTUnpivoted ||
           f == SparseFactorizationLDLTSBK || f == SparseFactorizationLDLTTPP;
}

SparseSymbolicFactorOptions symbolic(SparseOrder_t o) {
    SparseSymbolicFactorOptions so{};
    so.control = SparseDefaultControl;
    so.orderMethod = o;
    so.order = nullptr;
    so.ignoreRowsAndColumns = nullptr;
    so.malloc = counting_malloc;
    so.free = counting_free;
    so.reportError = report_error;  // never NULL
    return so;
}

SparseNumericFactorOptions numeric(double zero_tol = 0.0) {
    SparseNumericFactorOptions no{};
    no.control = SparseDefaultControl;
    no.scalingMethod = SparseScalingDefault;
    no.scaling = nullptr;
    no.pivotTolerance = 0.1;
    no.zeroTolerance = zero_tol;
    return no;
}

std::uint64_t hash_bytes(const void* p, size_t bytes) {
    const unsigned char* c = static_cast<const unsigned char*>(p);
    std::uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) h = (h ^ c[i]) * 1099511628211ull;
    return h;
}

template <class T>
double absval(T v) {
    return std::abs(v);
}

// One factor + solve, the matrix held as the factorization wants it.
struct Outcome {
    int status = 0;
    std::vector<double> x;  // real, or complex as (re, im) pairs
    std::vector<std::vector<double>> resolves;
    double fact_ms = 0, solve_ms = 0;
    long peak = 0;
    std::string error;
};

template <class T>
struct Problem {
    Csc<T> full, lower;
    std::vector<T> b;
};

template <class T>
Problem<T> make_problem(Csc<T> a) {
    Problem<T> p;
    p.full = std::move(a);
    if (p.full.symmetric) p.lower = lower_triangle(p.full);
    p.b.resize(static_cast<size_t>(p.full.m));
    for (int i = 0; i < p.full.m; ++i) {
        if constexpr (std::is_same_v<T, double>) p.b[static_cast<size_t>(i)] = std::sin(0.37 * i) + 0.1;
        else p.b[static_cast<size_t>(i)] = cplx(std::sin(0.37 * i), std::cos(0.21 * i));
    }
    return p;
}

template <class T>
Outcome factor_solve(
    const Problem<T>& p, SparseFactorization_t f, SparseOrder_t o, int resolves = 0, double zero_tol = 0.0
) {
    Outcome out;
    const bool sym = symmetric_type(f);
    const Csc<T>& a = sym ? p.lower : p.full;
    t_error.clear();
    g_bytes = 0;
    g_peak = 0;
    const int m = a.m, n = a.n;
    std::vector<T> x(static_cast<size_t>(std::max(m, n)));
    double t0 = now(), t1;
    if constexpr (std::is_same_v<T, double>) {
        SparseMatrix_Double A{};
        A.structure.rowCount = m;
        A.structure.columnCount = n;
        A.structure.columnStarts = const_cast<long*>(a.cs.data());
        A.structure.rowIndices = const_cast<int*>(a.ri.data());
        A.structure.attributes = SparseAttributes_t{};
        A.structure.attributes.kind = sym ? SparseSymmetric : SparseOrdinary;
        A.structure.attributes.triangle = SparseLowerTriangle;
        A.structure.blockSize = 1;
        A.data = const_cast<double*>(a.v.data());
        auto F = SparseFactor(f, A, symbolic(o), numeric(zero_tol));
        t1 = now();
        out.status = static_cast<int>(F.status);
        if (F.status == SparseStatusOK) {
            auto solve = [&](std::vector<double>& xs) {
                xs.assign(static_cast<size_t>(n), 0.0);
                std::vector<double> bx(p.b.begin(), p.b.end());
                DenseMatrix_Double B{m, 1, m, SparseAttributes_t{}, bx.data()};
                DenseMatrix_Double X{n, 1, n, SparseAttributes_t{}, xs.data()};
                if (f == SparseFactorizationCholeskyAtA) {  // a factor of A^T A: solve A^T A x = A^T b
                    // in place in max(m, n) rows
                    const int k = std::max(m, n);
                    std::vector<double> w(static_cast<size_t>(k), 0.0);
                    for (int j = 0; j < n; ++j) {
                        double c = 0;
                        for (long q = a.cs[j]; q < a.cs[j + 1]; ++q) c += a.v[q] * p.b[static_cast<size_t>(a.ri[q])];
                        w[static_cast<size_t>(j)] = c;
                    }
                    DenseMatrix_Double W{k, 1, k, SparseAttributes_t{}, w.data()};
                    SparseSolve(F, W);
                    xs.assign(w.begin(), w.begin() + n);
                } else if (m == n) {
                    SparseSolve(F, B);
                    xs.assign(bx.begin(), bx.begin() + n);
                } else {
                    SparseSolve(F, B, X);
                }
            };
            double s0 = now();
            solve(out.x);
            out.solve_ms = (now() - s0) * 1e3;
            for (int r = 0; r < resolves; ++r) {
                out.resolves.emplace_back();
                solve(out.resolves.back());
            }
        }
        SparseCleanup(F);
    } else {
        SparseMatrix_Complex_Double A{};
        A.structure.rowCount = m;
        A.structure.columnCount = n;
        A.structure.columnStarts = const_cast<long*>(a.cs.data());
        A.structure.rowIndices = const_cast<int*>(a.ri.data());
        A.structure.attributes = SparseAttributesComplex_t{};
        A.structure.attributes.kind = SparseOrdinary;
        A.structure.blockSize = 1;
        A.data = const_cast<cplx*>(a.v.data());
        auto F = SparseFactor(f, A, symbolic(o), numeric(zero_tol));
        t1 = now();
        out.status = static_cast<int>(F.status);
        if (F.status == SparseStatusOK) {
            auto solve = [&](std::vector<double>& xs) {
                std::vector<cplx> bx(p.b.begin(), p.b.end());
                DenseMatrix_Complex_Double B{m, 1, m, SparseAttributesComplex_t{}, bx.data()};
                SparseSolve(F, B);
                xs.resize(static_cast<size_t>(2 * n));
                std::memcpy(xs.data(), bx.data(), sizeof(cplx) * static_cast<size_t>(n));
            };
            double s0 = now();
            solve(out.x);
            out.solve_ms = (now() - s0) * 1e3;
            for (int r = 0; r < resolves; ++r) {
                out.resolves.emplace_back();
                solve(out.resolves.back());
            }
        }
        SparseCleanup(F);
    }
    out.fact_ms = (t1 - t0) * 1e3;
    out.peak = g_peak.load();
    out.error = t_error;
    return out;
}

// ||b - Ax|| / ||b|| for square A, ||A^T r|| / ||A^T b|| for rectangular A (A^H for complex).
template <class T>
double relres(const Problem<T>& p, const std::vector<double>& xs) {
    const Csc<T>& a = p.full;
    std::vector<T> x(static_cast<size_t>(a.n));
    std::memcpy(x.data(), xs.data(), sizeof(T) * static_cast<size_t>(a.n));
    std::vector<T> r(p.b);
    for (int j = 0; j < a.n; ++j)
        for (long q = a.cs[j]; q < a.cs[j + 1]; ++q)
            r[static_cast<size_t>(a.ri[q])] -= a.v[q] * x[static_cast<size_t>(j)];
    if (a.m == a.n) {
        double rn = 0, bn = 0;
        for (int i = 0; i < a.m; ++i) {
            rn += std::norm(r[static_cast<size_t>(i)]);
            bn += std::norm(p.b[static_cast<size_t>(i)]);
        }
        return std::sqrt(rn / bn);
    }
    double gn = 0, hn = 0;
    for (int j = 0; j < a.n; ++j) {
        T gr{}, gb{};
        for (long q = a.cs[j]; q < a.cs[j + 1]; ++q) {
            T c = a.v[q];
            if constexpr (!std::is_same_v<T, double>) c = std::conj(c);
            gr += c * r[static_cast<size_t>(a.ri[q])];
            gb += c * p.b[static_cast<size_t>(a.ri[q])];
        }
        gn += std::norm(gr);
        hn += std::norm(gb);
    }
    return std::sqrt(gn / hn);
}

double normwise_diff(const std::vector<double>& x, const std::vector<double>& y) {
    double d = 0, s = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        d += (x[i] - y[i]) * (x[i] - y[i]);
        s += x[i] * x[i];
    }
    return s > 0 ? std::sqrt(d / s) : std::sqrt(d);
}

bool finite_all(const std::vector<double>& x) {
    return std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); });
}

const char* mode() {
    const char* e = std::getenv("VECLIB_MAXIMUM_THREADS");
    return e ? e : "default";
}

// ---------------------------------------------------------------- det

// DET_FROM=kind:size starts the det section at that matrix (to resume a run).
bool det_started(const std::string& kind, const char* size) {
    static bool started = std::getenv("DET_FROM") == nullptr;
    if (!started && kind + ":" + size == std::getenv("DET_FROM")) started = true;
    return started;
}

template <class T>
void det_case(const Problem<T>& p, const char* size, SparseFactorization_t f, SparseOrder_t o, int max_reps) {
    if (symmetric_type(f) && !p.full.symmetric) return;
    if (!det_started(p.full.kind, size)) return;
    std::set<std::uint64_t> fh, rh;
    std::vector<double> x0;
    double md = 0, tf = 0, ts = 0, res = NAN;
    int reps = 0, status = 0;
    long peak = 0;
    std::string err;
    double budget_start = now();
    for (int r = 0; r < max_reps; ++r) {
        Outcome out = factor_solve(p, f, o, r == 0 ? 3 : 0);
        status = out.status;
        err = out.error;
        if (status != SparseStatusOK) break;
        ++reps;
        tf += out.fact_ms;
        ts += out.solve_ms;
        peak = std::max(peak, out.peak);
        fh.insert(hash_bytes(out.x.data(), out.x.size() * 8));
        if (r == 0) {
            x0 = out.x;
            res = relres(p, out.x);
            for (auto& y : out.resolves) {
                rh.insert(hash_bytes(y.data(), y.size() * 8));
                md = std::max(md, normwise_diff(x0, y));
            }
        } else {
            md = std::max(md, normwise_diff(x0, out.x));
        }
        // keep a case near 40 s: fewer runs for slow factorizations, never fewer than 2
        if (r >= 1 && (now() - budget_start) * (r + 2) / (r + 1) > 40.0) break;
    }
    if (status != SparseStatusOK) {
        std::printf(
            "DET mode=%s kind=%s size=%s n=%d nnz=%ld fact=%s order=%s status=%d error=\"%s\"\n", mode(),
            p.full.kind.c_str(), size, p.full.n, p.full.cs.back(), fact_name(f), order_name(o), status, err.c_str()
        );
    } else {
        std::printf(
            "DET mode=%s kind=%s size=%s n=%d nnz=%ld fact=%s order=%s status=0 reps=%d distinct=%zu resolve=%zu "
            "maxdiff=%.1e relres=%.1e fact_ms=%.2f solve_ms=%.2f peak_mb=%.1f\n",
            mode(), p.full.kind.c_str(), size, p.full.n, p.full.cs.back(), fact_name(f), order_name(o), reps, fh.size(),
            rh.size(), md, res, tf / reps, ts / reps, peak / 1048576.0
        );
    }
    std::fflush(stdout);
}

template <class T>
void det_matrix(
    const Problem<T>& p,
    const char* size,
    int reps,
    bool symmetric_types,
    bool lu_types,
    bool qr_types,
    bool colamd_lu = false
) {
    const SparseOrder_t nd[] = {SparseOrderAMD, SparseOrderMetis, SparseOrderMTMetis};
    if (symmetric_types)
        for (auto f :
             {SparseFactorizationCholesky, SparseFactorizationLDLT, SparseFactorizationLDLTUnpivoted,
              SparseFactorizationLDLTSBK, SparseFactorizationLDLTTPP})
            for (auto o : nd) det_case(p, size, f, o, reps);
    if (lu_types) {
        for (auto f :
             {SparseFactorizationLU, SparseFactorizationLUSPP, SparseFactorizationLUTPP,
              SparseFactorizationLUUnpivoted})
            for (auto o : nd) det_case(p, size, f, o, reps);
        det_case(p, size, SparseFactorizationLU, SparseOrderDefault, reps);
        if (colamd_lu) det_case(p, size, SparseFactorizationLU, SparseOrderCOLAMD, 1);
    }
    if (qr_types)
        for (auto f : {SparseFactorizationQR, SparseFactorizationCholeskyAtA})
            for (auto o : {SparseOrderCOLAMD, SparseOrderAMD, SparseOrderMetis}) det_case(p, size, f, o, reps);
    if (det_started(p.full.kind, size))
        std::printf(
            "RSS mode=%s after kind=%s size=%s max_rss_mb=%ld\n", mode(), p.full.kind.c_str(), size, max_rss_mb()
        );
}

void run_det() {
    const int R = 20;
    // small (n ~ 3.6e3) and medium (n ~ 9e4) for every kind
    struct Size {
        const char* name;
        int g2, g3, n;
    };
    // (the 3-D Laplacian's medium is n = 2.7e4: its factors at 9e4 take gigabytes)
    for (Size s : {Size{"small", 60, 15, 3600}, Size{"medium", 300, 30, 90000}}) {
        det_matrix(make_problem(laplacian2d(s.g2)), s.name, R, true, true, true);
        det_matrix(make_problem(laplacian3d(s.g3)), s.name, R, true, true, false);
        det_matrix(make_problem(convdiff2d(s.g2)), s.name, R, false, true, true, true);
        const int window = s.n > 10000 ? 100 : 0;
        det_matrix(make_problem(random_dd(s.n, 5, 11, window)), s.name, R, false, true, true);
        det_matrix(make_problem(banded(s.n, 5, 12)), s.name, R, false, true, false);
        det_matrix(make_problem(blockdiag(s.n, 8, 13)), s.name, R, false, true, false);
        det_matrix(make_problem(saddle(s.g2, 14)), s.name, R, true, true, false);
        det_matrix(make_problem(laplacian2d(s.g2, true, 1e-8)), s.name, R, true, true, false);
        det_matrix(make_problem(rect(s.n, 15, window)), s.name, R, false, false, true);
        det_matrix(make_problem(complex2d(s.g2)), s.name, R, false, true, false);
    }
    // large (n ~ 4.9e5): the 2-D Laplacian and convection-diffusion, the pairs that matter
    auto lap = make_problem(laplacian2d(700));
    for (auto f : {SparseFactorizationCholesky, SparseFactorizationLDLT})
        for (auto o : {SparseOrderAMD, SparseOrderMTMetis}) det_case(lap, "large", f, o, R);
    for (auto o : {SparseOrderAMD, SparseOrderMetis, SparseOrderMTMetis})
        det_case(lap, "large", SparseFactorizationLU, o, R);
    det_case(lap, "large", SparseFactorizationQR, SparseOrderCOLAMD, R);
    auto cd = make_problem(convdiff2d(700));
    for (auto o : {SparseOrderAMD, SparseOrderMetis, SparseOrderMTMetis})
        det_case(cd, "large", SparseFactorizationLU, o, R);
    auto cx = make_problem(complex2d(700));
    det_case(cx, "large", SparseFactorizationLU, SparseOrderAMD, R);
    std::printf("RSS mode=%s end max_rss_mb=%ld\n", mode(), max_rss_mb());
}

// ---------------------------------------------------------------- conc

std::string g_filter;

template <class T>
void conc_case(const Problem<T>& p, SparseFactorization_t f, SparseOrder_t o) {
    if (!g_filter.empty() && g_filter != order_name(o)) return;
    Outcome ref = factor_solve(p, f, o);
    const std::uint64_t h0 = hash_bytes(ref.x.data(), ref.x.size() * 8);
    std::atomic<int> same{0}, total{0};
    std::vector<std::thread> ts;
    std::mutex mu;
    std::set<std::uint64_t> seen;
    double worst = 0;  // the largest relres of the concurrent runs
    for (int t = 0; t < 4; ++t)
        ts.emplace_back([&] {
            for (int r = 0; r < 5; ++r) {
                Outcome out = factor_solve(p, f, o);
                std::uint64_t h = hash_bytes(out.x.data(), out.x.size() * 8);
                double rr = out.status == 0 ? relres(p, out.x) : NAN;
                total++;
                if (h == h0) same++;
                std::lock_guard<std::mutex> lk(mu);
                seen.insert(h);
                worst = std::isnan(rr) || std::isnan(worst) ? NAN : std::max(worst, rr);
            }
        });
    for (auto& t : ts) t.join();
    std::printf(
        "CONC mode=%s kind=%s n=%d fact=%s order=%s threads=4 runs=%d same_as_serial=%d distinct=%zu "
        "relres_serial=%.1e "
        "relres_worst=%.1e\n",
        mode(), p.full.kind.c_str(), p.full.n, fact_name(f), order_name(o), total.load(), same.load(), seen.size(),
        relres(p, ref.x), worst
    );
    std::fflush(stdout);
}

// filter: only the cases whose ordering name is this (empty: all)
void run_conc(const std::string& filter) {
    g_filter = filter;
    auto lap = make_problem(laplacian2d(300));
    auto cd = make_problem(convdiff2d(300));
    auto rd = make_problem(random_dd(90000, 5, 11, 100));
    auto cx = make_problem(complex2d(300));
    auto rc = make_problem(rect(90000, 15, 100));
    for (auto o : {SparseOrderAMD, SparseOrderMetis, SparseOrderMTMetis}) {
        conc_case(lap, SparseFactorizationLU, o);
        conc_case(cd, SparseFactorizationLU, o);
        conc_case(rd, SparseFactorizationLU, o);
        conc_case(cx, SparseFactorizationLU, o);
    }
    conc_case(lap, SparseFactorizationCholesky, SparseOrderAMD);
    conc_case(cd, SparseFactorizationQR, SparseOrderCOLAMD);
    conc_case(rc, SparseFactorizationQR, SparseOrderCOLAMD);
}

// ---------------------------------------------------------------- sing

void sing_line(const char* what, const Problem<double>& p, SparseFactorization_t f, SparseOrder_t o, double zt) {
    Outcome out = factor_solve(p, f, o, 0, zt);
    std::printf(
        "SING %s n=%d fact=%s order=%s zeroTol=%g status=%d", what, p.full.n, fact_name(f), order_name(o), zt,
        out.status
    );
    if (out.status == 0) {
        size_t nonfinite =
            static_cast<size_t>(std::count_if(out.x.begin(), out.x.end(), [](double v) { return !std::isfinite(v); }));
        std::printf(" nonfinite=%zu/%zu", nonfinite, out.x.size());
        if (out.x.size() <= 3) {
            std::printf(" x=");
            for (double v : out.x) std::printf("%.17g ", v);
        }
        if (finite_all(out.x)) std::printf(" relres=%.2e", relres(p, out.x));
    }
    if (!out.error.empty()) std::printf(" error=\"%s\"", out.error.c_str());
    std::printf("\n");
    std::fflush(stdout);
}

Problem<double> small_problem(
    const char* kind, int n, std::vector<long> cs, std::vector<int> ri, std::vector<double> v, bool symmetric
) {
    Csc<double> a;
    a.kind = kind;
    a.m = a.n = n;
    a.cs = std::move(cs);
    a.ri = std::move(ri);
    a.v = std::move(v);
    a.symmetric = symmetric;
    auto p = make_problem(a);
    for (int i = 0; i < n; ++i) p.b[static_cast<size_t>(i)] = 1.0;
    return p;
}

void run_sing() {
    const SparseFactorization_t lus[] = {
        SparseFactorizationLU, SparseFactorizationLUTPP, SparseFactorizationLUSPP, SparseFactorizationLUUnpivoted,
        SparseFactorizationQR
    };
    // structurally singular: column 2 empty
    auto s3 = small_problem("struct3", 3, {0, 1, 2, 2}, {0, 1}, {1, 1}, false);
    for (auto f : lus)
        sing_line("structural", s3, f, f == SparseFactorizationQR ? SparseOrderCOLAMD : SparseOrderAMD, 0);
    // structurally singular, larger: a random_dd with one column emptied
    {
        auto a = random_dd(2000, 5, 21);
        Triplets<double> t;
        for (int j = 0; j < a.n; ++j)
            for (long q = a.cs[j]; q < a.cs[j + 1]; ++q)
                if (j != 777) t.add(a.ri[q], j, a.v[q]);
        auto p = make_problem(from_triplets("struct2000", 2000, 2000, t));
        sing_line("structural", p, SparseFactorizationLU, SparseOrderAMD, 0);
        sing_line("structural", p, SparseFactorizationQR, SparseOrderCOLAMD, 0);
    }
    // numerically singular rank 1: [[1, 2], [2, 4]]
    auto r1 = small_problem("rank1_2x2", 2, {0, 2, 4}, {0, 1, 0, 1}, {1, 2, 2, 4}, true);
    for (auto f : lus)
        for (double zt : {0.0, 1e-14})
            sing_line("numeric", r1, f, f == SparseFactorizationQR ? SparseOrderCOLAMD : SparseOrderAMD, zt);
    sing_line("numeric", r1, SparseFactorizationCholesky, SparseOrderAMD, 0);
    sing_line("numeric", r1, SparseFactorizationLDLT, SparseOrderAMD, 0);
    // exactly singular, structurally full: the Neumann Laplacian (rows sum to 0 exactly)
    auto nm = make_problem(laplacian2d(60, true));
    for (double zt : {0.0, 1e-14}) {
        sing_line("neumann", nm, SparseFactorizationLU, SparseOrderAMD, zt);
        sing_line("neumann", nm, SparseFactorizationQR, SparseOrderCOLAMD, zt);
    }
    sing_line("neumann", nm, SparseFactorizationCholesky, SparseOrderAMD, 0);
    // two equal columns in a random matrix
    {
        // column 11 replaced by a copy of column 10
        auto a = random_dd(2000, 5, 22);
        Triplets<double> w;
        for (int j = 0; j < a.n; ++j)
            for (long q = a.cs[j]; q < a.cs[j + 1]; ++q)
                if (j != 11) w.add(a.ri[q], j, a.v[q]);
        for (long q = a.cs[10]; q < a.cs[11]; ++q) w.add(a.ri[q], 11, a.v[q]);
        auto p = make_problem(from_triplets("dupcol2000", 2000, 2000, w));
        for (double zt : {0.0, 1e-14}) {
            sing_line("dupcol", p, SparseFactorizationLU, SparseOrderAMD, zt);
            sing_line("dupcol", p, SparseFactorizationQR, SparseOrderCOLAMD, zt);
        }
    }
    // near singular (Neumann + 1e-8 I): status 0 and a finite, large solution expected
    auto ns = make_problem(laplacian2d(60, true, 1e-8));
    sing_line("nearsing", ns, SparseFactorizationLU, SparseOrderAMD, 0);
    // parameter errors through reportError
    auto lap = make_problem(laplacian2d(10));
    sing_line("param", lap, SparseFactorizationLU, SparseOrderCOLAMD, 0);
    auto rc = make_problem(rect(100, 3));
    {
        Outcome out = factor_solve(rc, SparseFactorizationLU, SparseOrderAMD);
        std::printf("SING param rectangular LU status=%d error=\"%s\"\n", out.status, out.error.c_str());
        Outcome c = factor_solve(rc, SparseFactorizationCholeskyAtA, SparseOrderAMD);
        std::printf("SING param rectangular CholeskyAtA/AMD status=%d error=\"%s\"\n", c.status, c.error.c_str());
    }
    std::fflush(stdout);
}

// ---------------------------------------------------------------- speed

template <class T>
void speed_case(const Problem<T>& p, SparseFactorization_t f, SparseOrder_t o) {
    double bf = 1e30, bs = 1e30;
    long peak = 0;
    int status = 0;
    for (int r = 0; r < 3; ++r) {
        Outcome out = factor_solve(p, f, o);
        status = out.status;
        if (status) break;
        bf = std::min(bf, out.fact_ms);
        bs = std::min(bs, out.solve_ms);
        peak = std::max(peak, out.peak);
        if (out.fact_ms > 20000) break;
    }
    std::printf(
        "SPEED mode=%s kind=%s n=%d nnz=%ld fact=%s order=%s status=%d factor_ms=%.2f solve_ms=%.2f "
        "peak_mb=%.1f max_rss_mb=%ld\n",
        mode(), p.full.kind.c_str(), p.full.n, p.full.cs.back(), fact_name(f), order_name(o), status, bf, bs,
        peak / 1048576.0, max_rss_mb()
    );
    std::fflush(stdout);
}

void run_speed() {
    for (int g : {100, 300, 700, 1000}) {
        auto p = make_problem(laplacian2d(g));
        for (auto o : {SparseOrderAMD, SparseOrderMetis, SparseOrderMTMetis}) speed_case(p, SparseFactorizationLU, o);
        speed_case(p, SparseFactorizationCholesky, SparseOrderAMD);
    }
    for (int g : {22, 45, 60}) {
        auto p = make_problem(laplacian3d(g));
        for (auto o : {SparseOrderAMD, SparseOrderMetis, SparseOrderMTMetis}) speed_case(p, SparseFactorizationLU, o);
        speed_case(p, SparseFactorizationCholesky, SparseOrderAMD);
    }
    for (int g : {300, 700}) {
        auto p = make_problem(convdiff2d(g));
        speed_case(p, SparseFactorizationLU, SparseOrderAMD);
    }
}

// ---------------------------------------------------------------- iter (SP5)

enum class Iter { CG, GMRES, DQGMRES, LSMR };

const char* iter_name(Iter k) {
    switch (k) {
        case Iter::CG: return "CG";
        case Iter::GMRES: return "GMRES(20)";
        case Iter::DQGMRES: return "DQGMRES(20)";
        case Iter::LSMR: return "LSMR";
    }
    return "?";
}

void iter_case(const Problem<double>& p, const char* grid, Iter k) {
    const Csc<double>& a = p.full;
    SparseMatrix_Double A{};
    A.structure.rowCount = a.m;
    A.structure.columnCount = a.n;
    A.structure.columnStarts = const_cast<long*>(a.cs.data());
    A.structure.rowIndices = const_cast<int*>(a.ri.data());
    A.structure.attributes = SparseAttributes_t{};
    A.structure.attributes.kind = SparseOrdinary;
    A.structure.blockSize = 1;
    A.data = const_cast<double*>(a.v.data());
    SparseIterativeMethod method;
    if (k == Iter::CG) {
        SparseCGOptions o{};
        o.reportError = report_error;
        o.maxIterations = 20000;
        o.rtol = 1e-10;
        method = SparseConjugateGradient(o);
    } else if (k == Iter::LSMR) {
        SparseLSMROptions o{};
        o.reportError = report_error;
        o.convergenceTest = SparseLSMRCTFongSaunders;
        o.atol = 1e-8;
        o.btol = 1e-8;
        o.conditionLimit = 1e8;
        o.maxIterations = 20000;
        method = SparseLSMR(o);
    } else {
        SparseGMRESOptions o{};
        o.reportError = report_error;
        o.variant = k == Iter::GMRES ? SparseVariantGMRES : SparseVariantDQGMRES;
        o.nvec = 20;
        o.maxIterations = 40000;
        o.rtol = 1e-8;
        method = SparseGMRES(o);
    }
    double best = 1e30;
    int status = 0;
    std::vector<double> x;
    for (int r = 0; r < 3; ++r) {
        std::vector<double> bx(p.b.begin(), p.b.end());
        x.assign(static_cast<size_t>(a.n), 0.0);
        DenseVector_Double B{a.m, bx.data()};
        DenseVector_Double X{a.n, x.data()};
        t_error.clear();
        const double t0 = now();
        status = static_cast<int>(SparseSolve(method, A, B, X));
        best = std::min(best, (now() - t0) * 1e3);
        if (best > 20000) break;
    }
    std::printf(
        "ITER mode=%s grid=%s n=%d method=%s status=%d ms=%.2f relres=%.3g%s%s\n", mode(), grid, a.n, iter_name(k),
        status, best, relres(p, x), t_error.empty() ? "" : " error=", t_error.c_str()
    );
    std::fflush(stdout);
}

void run_iter() {
    struct Grid {
        const char* name;
        int g, dim;
    };
    for (const Grid& gr :
         {Grid{"lap2d/100", 100, 2}, Grid{"lap2d/300", 300, 2}, Grid{"lap2d/1000", 1000, 2}, Grid{"lap3d/22", 22, 3},
          Grid{"lap3d/45", 45, 3}}) {
        auto p = make_problem(gr.dim == 2 ? laplacian2d(gr.g) : laplacian3d(gr.g));
        for (Iter k : {Iter::CG, Iter::GMRES, Iter::DQGMRES, Iter::LSMR}) {
            if (gr.g == 1000 && k != Iter::CG) continue;  // (the speed cases run CG and BiCGSTAB there)
            iter_case(p, gr.name, k);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string what = argc > 1 ? argv[1] : "det";
    if (what == "det") run_det();
    if (what == "conc") run_conc(argc > 2 ? argv[2] : "");
    if (what == "sing") run_sing();
    if (what == "speed") run_speed();
    if (what == "iter") run_iter();
    return 0;
}
