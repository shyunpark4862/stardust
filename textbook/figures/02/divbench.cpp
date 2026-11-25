// Division and gcd timings for chapter 2.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <time.h>
#include <random>
#include <algorithm>
using i64 = int64_t; using u64 = uint64_t;
static double now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }
static inline u64 mulhi(u64 a, u64 b) { return (u64)(((unsigned __int128)a * b) >> 64); }
__attribute__((noinline)) void floor_sdiv(const i64* a, i64 b, i64* q, size_t n) {
  for (size_t i = 0; i < n; ++i) { i64 x = a[i]; i64 t = x / b; if ((x % b != 0) && ((x < 0) != (b < 0))) --t; q[i] = t; }
}
struct Magic { u64 m; int l; };
// 63-bit numerators (after the sign fold u = a ^ (a >> 63) < 2^63): m = ceil(2^(63+l)/d) < 2^64.
static Magic magic63(u64 d) { int l = 64 - __builtin_clzll(d - 1); unsigned __int128 two = (unsigned __int128)1 << (63 + l);
  return { (u64)(two / d + (two % d != 0)), l }; }
__attribute__((noinline)) void floor_magic63(const i64* a, Magic mg, i64* q, size_t n) {
  for (size_t i = 0; i < n; ++i) { u64 s = (u64)(a[i] >> 63), u = (u64)a[i] ^ s; q[i] = (i64)((mulhi(u, mg.m) >> (mg.l - 1)) ^ s); }
}
static Magic magic(u64 d) {  // d >= 2: m' = ceil(2^(64+l)/d) - 2^64, l = ceil(log2 d)
  int l = 64 - __builtin_clzll(d - 1);
  unsigned __int128 two = (unsigned __int128)1 << (64 + l);
  unsigned __int128 m = two / d + (two % d != 0);
  return { (u64)(m - ((unsigned __int128)1 << 64)), l };
}
__attribute__((noinline)) void floor_magic(const i64* a, Magic mg, i64* q, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    u64 s = (u64)(a[i] >> 63), u = (u64)a[i] ^ s;
    u64 t = mulhi(u, mg.m);
    u64 qq = (t + ((u - t) >> 1)) >> (mg.l - 1);
    q[i] = (i64)(qq ^ s);
  }
}
__attribute__((noinline)) void floor_vdiv(const i64* a, const i64* b, i64* q, size_t n) {
  for (size_t i = 0; i < n; ++i) { i64 x = a[i], y = b[i]; i64 t = x / y; if ((x % y != 0) && ((x < 0) != (y < 0))) --t; q[i] = t; }
}
static inline u64 euclid(u64 a, u64 b) { while (b) { u64 r = a % b; a = b; b = r; } return a; }
static inline u64 binary_gcd(u64 a, u64 b) {
  if (a == 0) return b; if (b == 0) return a;
  int k = __builtin_ctzll(a | b); a >>= __builtin_ctzll(a);
  while (b) { b >>= __builtin_ctzll(b); if (a > b) std::swap(a, b); b -= a; }
  return a << k;
}
int main(int argc, char** argv) {
  const size_t n = 1 << 20; const int reps = 50;
  std::mt19937_64 g(12345);
  std::vector<i64> a(n), b(n), q1(n), q2(n);
  for (auto& x : a) x = (i64)(g() >> 1) - (i64)(1ULL << 62);
  for (auto& x : b) { do x = (i64)(g() % 2000) - 1000; while (x == 0 || x == -1); }
  volatile i64 dv = 7; i64 d = dv;
  auto bench = [&](auto f) { double best = 1e9; for (int r = 0; r < reps; ++r) { double t0 = now(); f(); best = std::min(best, now() - t0); } return best / n * 1e9; };
  for (i64 dd : {3LL, 7LL, 10LL, 1000000007LL}) {
    volatile i64 vd = dd; d = vd; Magic mg = magic((u64)d);
    double t1 = bench([&]{ floor_sdiv(a.data(), d, q1.data(), n); });
    double t2 = bench([&]{ floor_magic(a.data(), mg, q2.data(), n); });
    size_t bad = 0; for (size_t i = 0; i < n; ++i) bad += q1[i] != q2[i];
    Magic m6 = magic63((u64)d); double t4 = bench([&]{ floor_magic63(a.data(), m6, q2.data(), n); });
    for (size_t i = 0; i < n; ++i) bad += q1[i] != q2[i];
    printf("d=%lld sdiv %.3f ns  magic65 %.3f ns  magic63 %.3f ns  mismatches %zu  (m63=%#llx l=%d)\n", (long long)d, t1, t2, t4, bad, (unsigned long long)m6.m, m6.l);
  }
  double t3 = bench([&]{ floor_vdiv(a.data(), b.data(), q1.data(), n); });
  printf("vector divisor sdiv %.3f ns\n", t3);
  // edge check of magic floor on extremes
  for (i64 dd : {2LL, 3LL, 7LL, 10LL, 641LL, 1LL << 40, (1LL << 62) + 1, INT64_MAX}) { Magic mg = magic63(dd); i64 xs[] = {INT64_MIN, INT64_MIN + 1, -1, 0, 1, INT64_MAX, -dd, dd, -dd - 1}; i64 out[9], ref[9];
    floor_magic63(xs, mg, out, 9); floor_sdiv(xs, dd, ref, 9); for (int i = 0; i < 9; ++i) if (out[i] != ref[i]) printf("EDGE MISMATCH d=%lld x=%lld\n", (long long)dd, (long long)xs[i]); }
  // gcd
  const size_t m = 1 << 18; std::vector<u64> x(m), y(m); for (size_t i = 0; i < m; ++i) { x[i] = g(); y[i] = g(); }
  volatile u64 sink = 0;
  auto gb = [&](auto f) { double best = 1e9; for (int r = 0; r < 20; ++r) { double t0 = now(); u64 s = 0; for (size_t i = 0; i < m; ++i) s += f(x[i], y[i]); sink = s; best = std::min(best, now() - t0); } return best / m * 1e9; };
  auto libcxx = [](u64 a, u64 b) { if (a < b) std::swap(a, b); if (b == 0) return a; a %= b; if (a == 0) return b; return binary_gcd(a, b); };
  printf("gcd libc++-style %.1f ns\n", gb(libcxx));
  printf("gcd euclid %.1f ns  binary %.1f ns  std::gcd %.1f ns\n", gb(euclid), gb(binary_gcd), gb([](u64 p, u64 q) { return std::gcd(p, q); }));
  size_t badg = 0; for (size_t i = 0; i < m; ++i) badg += euclid(x[i], y[i]) != binary_gcd(x[i], y[i]); printf("gcd mismatches %zu\n", badg);
  // average Euclid steps
  double steps = 0; for (size_t i = 0; i < m; ++i) { u64 p = x[i], q = y[i]; while (q) { u64 r = p % q; p = q; q = r; steps++; } } printf("mean euclid steps %.2f (theory 12 ln2/pi^2 ln 2^64 = %.2f)\n", steps / m, 12 * 0.6931471805599453 / (3.141592653589793 * 3.141592653589793) * 64 * 0.6931471805599453);
}
