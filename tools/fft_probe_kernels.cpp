// The C++ side of tools/fft_probe.py (docs/design/fft.md §3.1, §3.2, §7.4): vDSP's DFT probed through
// the SDK header's functions (never its binary), and the prototype of the library's own path
// (csrc/fft/cfft.hpp, rfft.hpp) timed against it. Every command prints tab-separated lines that
// the Python side parses; nothing here writes into the repository.
//
//   paths MAXN          own path vs vDSP, end to end from sd's layouts, per length class (FT22)
//   lengths KIND A B    the lengths A..B whose setup is not NULL (KIND zop zrop ilv dct)
//   big CAP             zop / zrop / Interleaved setups of f 2^k up to CAP points
//   det                 bit determinism of zop and zrop: repeat, misaligned, threads, in place,
//                       recreated, sharing `previous`, and inverse vs conj(forward(conj))
//   setup CAP           setup time and memory (malloc's bytes in use) per class up to CAP points
//   cpu                 CPU time / wall time of one execution (does vDSP thread inside?)
//   split               zop's execution alone, with the (de)interleaving, and Interleaved
//   dump KIND N IN OUT  the own and vDSP outputs of the input file IN (KIND c2c or r2c)
//   twiddle N OUT       csrc/fft/twiddle.hpp's e^(-2 pi i k / n), k = 0 .. n - 1, into OUT
//
// Built by tools/fft_probe.py with sd's flags (-O3 -mcpu=native -ffp-contract=off). No setup is
// created whose estimate (csrc/fft/vdsp.hpp) passes 512 MB; the Python side also watches the
// process's memory and kills it above its cap.
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "fft/cfft.hpp"
#include "fft/rfft.hpp"
#include "fft/vdsp.hpp"

using namespace sd::fft;

// The library runs a four-step's tasks on its pool (fft/plan.cpp); the probe runs them in order.
void sd::fft::run_tasks(int64_t ntasks, const std::function<void(int64_t)>& fn, int64_t) {
    for (int64_t t = 0; t < ntasks; ++t) fn(t);
}

namespace {

constexpr int64_t kMaxSetupBytes = int64_t{512} << 20;

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// The bytes malloc holds for the process (every zone): a setup's memory is the difference across
// its creation (the process's footprint is not, since freed pages are reused).
double malloc_mb() {
    malloc_statistics_t st;
    malloc_zone_statistics(nullptr, &st);
    return static_cast<double>(st.size_in_use) / 1048576.0;
}

uint64_t g_state = 0x9e3779b97f4a7c15ULL;
double uniform() {  // [-1, 1), xorshift64*
    g_state ^= g_state >> 12, g_state ^= g_state << 25, g_state ^= g_state >> 27;
    return static_cast<double>((g_state * 2685821657736338717ULL) >> 11) * 0x1p-52 - 1.0;
}

struct Buf {  // 128-byte aligned doubles
    double* p = nullptr;
    explicit Buf(size_t n) {
        if (posix_memalign(reinterpret_cast<void**>(&p), 128, std::max<size_t>(n, 2) * sizeof(double))) std::abort();
        std::memset(p, 0, std::max<size_t>(n, 2) * sizeof(double));
    }
    ~Buf() { free(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
};

// Best time per call (us) of each function, timed in alternating blocks of at least 1 ms.
std::vector<double> best_us(const std::vector<std::function<void()>>& fns, int blocks = 9) {
    std::vector<int> reps(fns.size(), 1);
    for (size_t i = 0; i < fns.size(); ++i) {
        fns[i]();  // warm up
        for (;;) {
            const double t0 = now();
            for (int r = 0; r < reps[i]; ++r) fns[i]();
            if (now() - t0 >= 1e-3 || reps[i] >= (1 << 24)) break;
            reps[i] *= 2;
        }
    }
    std::vector<double> best(fns.size(), 1e30);
    for (int b = 0; b < blocks; ++b) {
        for (size_t i = 0; i < fns.size(); ++i) {
            const double t0 = now();
            for (int r = 0; r < reps[i]; ++r) fns[i]();
            best[i] = std::min(best[i], (now() - t0) / reps[i] * 1e6);
        }
    }
    return best;
}

double rel_diff(const double* a, const double* b, size_t len) {
    double num = 0, den = 0;
    for (size_t i = 0; i < len; ++i) num += (a[i] - b[i]) * (a[i] - b[i]), den += b[i] * b[i];
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

const int64_t kClasses[] = {1, 3, 5, 15};

// ---------------------------------------------------------------- paths (FT22)

void paths(int64_t maxn) {
    for (int kind = 0; kind < 2; ++kind) {
        const VdspKind vk = kind == 0 ? VdspKind::c2c : VdspKind::r2c;
        for (int64_t f : kClasses) {
            for (int k = 0; (f << k) <= maxn; ++k) {
                const int64_t n = f << k;
                if (n < 16 || !vdsp_accepts(vk, n) || vdsp_setup_bytes(n) > kMaxSetupBytes) continue;
                const size_t in_len = kind == 0 ? 2 * n : n, out_len = kind == 0 ? 2 * n : n + 2;
                Buf x(in_len), y_own(out_len), y_vdsp(out_len), y_ilv(out_len), work(2 * n);
                for (size_t i = 0; i < in_len; ++i) x.p[i] = uniform();
                VdspPair vd(vk, n);
                if (!vd.ok()) {
                    std::printf(
                        "paths\t%s\t%lld\t%d\t%lld\tno-setup\n", kind ? "r2c" : "c2c", (long long)f, k, (long long)n
                    );
                    continue;
                }
                double own_us, vdsp_us, ilv_us = -1, diff;
                if (kind == 0) {
                    CfftPlan own(n);
                    own.forward(x.p, y_own.p, work.p);
                    vd.c2c_forward(x.p, y_vdsp.p, work.p);
                    diff = rel_diff(y_own.p, y_vdsp.p, out_len);
                    std::vector<std::function<void()>> fns = {
                        [&] { own.forward(x.p, y_own.p, work.p); }, [&] { vd.c2c_forward(x.p, y_vdsp.p, work.p); }
                    };
                    VdspInterleaved il(n);
                    if (n <= 4096 && il.ok()) fns.push_back([&] { il.forward(x.p, y_ilv.p); });
                    const auto t = best_us(fns);
                    own_us = t[0], vdsp_us = t[1];
                    if (t.size() > 2) ilv_us = t[2];
                } else {
                    RfftPlan own(n);
                    own.forward(x.p, y_own.p, work.p);
                    vd.r2c_forward(x.p, y_vdsp.p, work.p);
                    diff = rel_diff(y_own.p, y_vdsp.p, out_len);
                    const auto t = best_us(
                        {[&] { own.forward(x.p, y_own.p, work.p); }, [&] { vd.r2c_forward(x.p, y_vdsp.p, work.p); }}
                    );
                    own_us = t[0], vdsp_us = t[1];
                }
                std::printf(
                    "paths\t%s\t%lld\t%d\t%lld\t%.4f\t%.4f\t%.4f\t%.3e\n", kind ? "r2c" : "c2c", (long long)f, k,
                    (long long)n, own_us, vdsp_us, ilv_us, diff
                );
                std::fflush(stdout);
            }
        }
    }
}

// ---------------------------------------------------------------- lengths, big

void* create(const std::string& kind, int64_t n) {
    const auto len = static_cast<vDSP_Length>(n);
    if (kind == "zop") return vDSP_DFT_zop_CreateSetupD(nullptr, len, vDSP_DFT_FORWARD);
    if (kind == "zrop") return n % 2 ? nullptr : vDSP_DFT_zrop_CreateSetupD(nullptr, len, vDSP_DFT_FORWARD);
    if (kind == "ilv") {
        return vDSP_DFT_Interleaved_CreateSetupD(nullptr, len, vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_ComplextoComplex);
    }
    if (kind == "dct") return vDSP_DCT_CreateSetup(nullptr, len, vDSP_DCT_II);
    std::abort();
}

void destroy(const std::string& kind, void* s) {
    if (kind == "zop" || kind == "zrop") vDSP_DFT_DestroySetupD(static_cast<vDSP_DFT_SetupD>(s));
    if (kind == "ilv") vDSP_DFT_Interleaved_DestroySetupD(static_cast<vDSP_DFT_Interleaved_SetupD>(s));
    if (kind == "dct") vDSP_DFT_DestroySetup(static_cast<vDSP_DFT_Setup>(s));
}

// One line per accepted length, printed at once (a crash loses nothing before it). The length being
// tried goes to stderr first, so a crash names it.
void lengths(const std::string& kind, int64_t first, int64_t last) {
    for (int64_t n = first; n <= last; ++n) {
        std::fprintf(stderr, "%lld\n", (long long)n);
        void* s = create(kind, n);
        if (!s) continue;
        destroy(kind, s);
        std::printf("lengths\t%s\t%lld\n", kind.c_str(), (long long)n);
        std::fflush(stdout);
    }
}

void big(int64_t cap) {
    const int64_t fs[] = {1, 3, 5, 9, 15, 25};
    for (const std::string kind : {"zop", "zrop", "ilv"}) {
        for (int64_t f : fs) {
            for (int k = 0; (f << k) <= cap; ++k) {
                const int64_t n = f << k;
                if (vdsp_setup_bytes(n) > kMaxSetupBytes) break;
                void* s = create(kind, n);
                std::printf("big\t%s\t%lld\t%d\t%lld\t%d\n", kind.c_str(), (long long)f, k, (long long)n, s != nullptr);
                if (s) destroy(kind, s);
                std::fflush(stdout);
            }
        }
    }
}

// ---------------------------------------------------------------- det

struct Job {
    vDSP_DFT_SetupD st;
    const double *ir, *ii;
    double *outr, *outi;
};

void* run_job(void* p) {
    auto* j = static_cast<Job*>(p);
    for (int r = 0; r < 50; ++r) vDSP_DFT_ExecuteD(j->st, j->ir, j->ii, j->outr, j->outi);
    return nullptr;
}

bool same(const double* a, const double* b, int64_t len) { return std::memcmp(a, b, len * sizeof(double)) == 0; }

void det() {
    const int64_t lens[] = {1024, 4096, 3072, 15360, 20480, 40960, 1 << 16, 1 << 20};
    for (int kind = 0; kind < 2; ++kind) {
        const VdspKind vk = kind == 0 ? VdspKind::c2c : VdspKind::r2c;
        for (int64_t n : lens) {
            const int64_t h = kind == 0 ? n : n / 2;  // the split arrays' length
            VdspPair vd(vk, n);
            if (!vd.ok()) continue;
            vDSP_DFT_SetupD st = vd.forward_setup();
            Buf base(4 * h + 64), o1(2 * h), o2(2 * h);
            double *ir = base.p, *ii = base.p + h + 8;
            for (int64_t i = 0; i < h; ++i) ir[i] = uniform(), ii[i] = uniform();
            vDSP_DFT_ExecuteD(st, ir, ii, o1.p, o1.p + h);
            vDSP_DFT_ExecuteD(st, ir, ii, o2.p, o2.p + h);
            const bool rep = same(o1.p, o2.p, 2 * h);
            Buf mb(2 * h + 16), m3(2 * h + 16);  // 8-byte misaligned copies
            std::memcpy(mb.p + 1, ir, h * 8), std::memcpy(mb.p + h + 3, ii, h * 8);
            vDSP_DFT_ExecuteD(st, mb.p + 1, mb.p + h + 3, m3.p + 1, m3.p + h + 5);
            const bool mis = same(o1.p, m3.p + 1, h) && same(o1.p + h, m3.p + h + 5, h);
            constexpr int kThreads = 10;
            pthread_t th[kThreads];
            Job jobs[kThreads];
            std::vector<std::vector<double>> outs(kThreads, std::vector<double>(2 * h));
            for (int t = 0; t < kThreads; ++t) {
                jobs[t] = {st, ir, ii, outs[t].data(), outs[t].data() + h};
                pthread_create(&th[t], nullptr, run_job, &jobs[t]);
            }
            bool thr = true;
            for (int t = 0; t < kThreads; ++t) {
                pthread_join(th[t], nullptr);
                thr = thr && same(outs[t].data(), o1.p, 2 * h);
            }
            Buf ip(2 * h);
            std::memcpy(ip.p, ir, h * 8), std::memcpy(ip.p + h, ii, h * 8);
            vDSP_DFT_ExecuteD(st, ip.p, ip.p + h, ip.p, ip.p + h);
            const bool inplace = same(ip.p, o1.p, 2 * h);
            Buf o3(2 * h), o4(2 * h);
            bool recreated, shared;
            {
                VdspPair again(vk, n);
                vDSP_DFT_ExecuteD(again.forward_setup(), ir, ii, o3.p, o3.p + h);
                recreated = same(o3.p, o1.p, 2 * h);
                VdspPair other(vk, 2 * n);
                vDSP_DFT_SetupD s2 = vk == VdspKind::c2c
                                         ? vDSP_DFT_zop_CreateSetupD(other.forward_setup(), n, vDSP_DFT_FORWARD)
                                         : vDSP_DFT_zrop_CreateSetupD(other.forward_setup(), n, vDSP_DFT_FORWARD);
                vDSP_DFT_ExecuteD(s2, ir, ii, o4.p, o4.p + h);
                shared = s2 && same(o4.p, o1.p, 2 * h);
                if (s2) vDSP_DFT_DestroySetupD(s2);
            }
            int conjrel = -1;  // inverse(x) == conj(forward(conj x)), complex only
            if (kind == 0) {
                Buf a(2 * h), b(2 * h), c(2 * h);
                vDSP_DFT_ExecuteD(vd.inverse_setup(), ir, ii, a.p, a.p + h);
                for (int64_t i = 0; i < h; ++i) c.p[i] = ir[i], c.p[h + i] = -ii[i];
                vDSP_DFT_ExecuteD(st, c.p, c.p + h, b.p, b.p + h);
                conjrel = 1;
                for (int64_t i = 0; i < h; ++i) {
                    if (a.p[i] != b.p[i] || a.p[h + i] != -b.p[h + i]) {
                        conjrel = 0;
                        break;
                    }
                }
            }
            std::printf(
                "det\t%s\t%lld\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", kind ? "zrop" : "zop", (long long)n, rep, mis, thr,
                inplace, recreated, shared, conjrel
            );
            std::fflush(stdout);
        }
    }
}

// ---------------------------------------------------------------- setup, cpu, split

void setup(int64_t cap) {
    for (int kind = 0; kind < 2; ++kind) {
        const VdspKind vk = kind == 0 ? VdspKind::c2c : VdspKind::r2c;
        for (int64_t f : kClasses) {
            for (int k = 0; (f << k) <= cap; ++k) {
                const int64_t n = f << k;
                if (n < 16 || !vdsp_accepts(vk, n) || vdsp_setup_bytes(n) > kMaxSetupBytes) continue;
                const auto len = static_cast<vDSP_Length>(n);
                const double m0 = malloc_mb(), t0 = now();
                vDSP_DFT_SetupD a = vk == VdspKind::c2c ? vDSP_DFT_zop_CreateSetupD(nullptr, len, vDSP_DFT_FORWARD)
                                                        : vDSP_DFT_zrop_CreateSetupD(nullptr, len, vDSP_DFT_FORWARD);
                const double t1 = now(), m1 = malloc_mb();
                vDSP_DFT_SetupD b = vk == VdspKind::c2c ? vDSP_DFT_zop_CreateSetupD(a, len, vDSP_DFT_INVERSE)
                                                        : vDSP_DFT_zrop_CreateSetupD(a, len, vDSP_DFT_INVERSE);
                const double t2 = now(), m2 = malloc_mb();
                int64_t own_bytes;
                double own_us;
                {
                    const double t3 = now();
                    if (kind == 0) {
                        CfftPlan p(n);
                        own_bytes = p.bytes();
                    } else {
                        RfftPlan p(n);
                        own_bytes = p.bytes();
                    }
                    own_us = (now() - t3) * 1e6;
                }
                std::printf(
                    "setup\t%s\t%lld\t%d\t%lld\t%.2f\t%.4f\t%.2f\t%.4f\t%.2f\t%.4f\n", kind ? "zrop" : "zop",
                    (long long)f, k, (long long)n, (t1 - t0) * 1e6, m1 - m0, (t2 - t1) * 1e6, m2 - m1, own_us,
                    own_bytes / 1048576.0
                );
                std::fflush(stdout);
                vDSP_DFT_DestroySetupD(b);
                vDSP_DFT_DestroySetupD(a);
            }
        }
    }
}

double cpu_seconds() {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6;
}

void cpu() {
    const int64_t lens[] = {1 << 16, 1 << 20, 15 << 16};
    for (int64_t n : lens) {
        VdspPair vd(VdspKind::c2c, n);
        Buf a(4 * n);
        for (int64_t i = 0; i < 2 * n; ++i) a.p[i] = uniform();
        const int reps = n >= (1 << 20) ? 10 : 200;
        const double w0 = now(), c0 = cpu_seconds();
        for (int r = 0; r < reps; ++r) vDSP_DFT_ExecuteD(vd.forward_setup(), a.p, a.p + n, a.p + 2 * n, a.p + 3 * n);
        const double w = now() - w0, c = cpu_seconds() - c0;
        std::printf("cpu\t%lld\t%.1f\t%.3f\n", (long long)n, w / reps * 1e6, c / w);
        std::fflush(stdout);
    }
}

void split() {
    const int64_t lens[] = {1024, 4096, 3072, 15360, 20480, 1 << 16, 1 << 20};
    for (int64_t n : lens) {
        VdspPair vd(VdspKind::c2c, n);
        VdspInterleaved il(n);
        Buf c(2 * n), o(2 * n), re(n), im(n), ore(n), oim(n), work(2 * n);
        for (int64_t j = 0; j < 2 * n; ++j) c.p[j] = uniform();
        std::vector<std::function<void()>> fns = {
            [&] { vDSP_DFT_ExecuteD(vd.forward_setup(), re.p, im.p, ore.p, oim.p); },
            [&] { vd.c2c_forward(c.p, o.p, work.p); },
        };
        if (il.ok()) fns.push_back([&] { il.forward(c.p, o.p); });
        const auto t = best_us(fns);
        std::printf("split\t%lld\t%.4f\t%.4f\t%.4f\n", (long long)n, t[0], t[1], t.size() > 2 ? t[2] : -1.0);
        std::fflush(stdout);
    }
}

// ---------------------------------------------------------------- dump

void dump(const std::string& kind, int64_t n, const char* in, const char* out) {
    const bool c2c = kind == "c2c";
    const size_t in_len = c2c ? 2 * n : n, out_len = c2c ? 2 * n : n + 2;
    Buf x(in_len), y_own(out_len), y_vdsp(out_len), work(2 * n);
    FILE* fi = std::fopen(in, "rb");
    if (!fi || std::fread(x.p, sizeof(double), in_len, fi) != in_len) std::abort();
    std::fclose(fi);
    bool have_vdsp = vdsp_accepts(c2c ? VdspKind::c2c : VdspKind::r2c, n);
    if (c2c) {
        CfftPlan(n).forward(x.p, y_own.p, work.p);
        if (have_vdsp) VdspPair(VdspKind::c2c, n).c2c_forward(x.p, y_vdsp.p, work.p);
    } else {
        RfftPlan(n).forward(x.p, y_own.p, work.p);
        if (have_vdsp) VdspPair(VdspKind::r2c, n).r2c_forward(x.p, y_vdsp.p, work.p);
    }
    FILE* fo = std::fopen(out, "wb");
    std::fwrite(y_own.p, sizeof(double), out_len, fo);
    std::fwrite(y_vdsp.p, sizeof(double), out_len, fo);
    std::fclose(fo);
    std::printf("dump\t%s\t%lld\t%d\n", kind.c_str(), (long long)n, have_vdsp);
}

void twiddles(int64_t n, const char* out) {
    std::vector<double> w(static_cast<size_t>(2 * n));
    for (int64_t k = 0; k < n; ++k) {
        const Twiddle t = twiddle(k, n);
        w[static_cast<size_t>(2 * k)] = t.re, w[static_cast<size_t>(2 * k + 1)] = t.im;
    }
    FILE* fo = std::fopen(out, "wb");
    std::fwrite(w.data(), sizeof(double), w.size(), fo);
    std::fclose(fo);
}

}  // namespace

int main(int argc, char** argv) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (argc < 2) return 2;
    const std::string cmd = argv[1];
    if (cmd == "paths") paths(argc > 2 ? std::atoll(argv[2]) : (1 << 20));
    else if (cmd == "lengths") lengths(argv[2], std::atoll(argv[3]), std::atoll(argv[4]));
    else if (cmd == "big") big(std::atoll(argv[2]));
    else if (cmd == "det") det();
    else if (cmd == "setup") setup(std::atoll(argv[2]));
    else if (cmd == "cpu") cpu();
    else if (cmd == "split") split();
    else if (cmd == "dump") dump(argv[2], std::atoll(argv[3]), argv[4], argv[5]);
    else if (cmd == "twiddle") twiddles(std::atoll(argv[2]), argv[3]);
    else return 2;
    return 0;
}
