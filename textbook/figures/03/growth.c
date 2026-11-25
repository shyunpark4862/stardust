// Measured growth (Section 3.1): insertion sort (n^2) and bottom-up merge sort (n log n) on random
// doubles, and the time of one dependent load in a random cycle through an array of a given size (L1, L2,
// or only memory: the pointer-chasing latency).
// clang -O2 growth.c -o growth; ./growth > growth.dat
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }
static unsigned long long s = 88172645463325252ULL;
static double rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (s >> 11) * 0x1p-53; }
static void insertion(double* a, int n) {
  for (int i = 1; i < n; ++i) { double v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; --j; } a[j + 1] = v; }
}
static void merge_sort(double* a, double* tmp, int n) {
  for (int w = 1; w < n; w *= 2) {
    for (int lo = 0; lo < n; lo += 2 * w) {
      int mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;
      while (i < mid && j < hi) tmp[k++] = a[j] < a[i] ? a[j++] : a[i++];
      while (i < mid) tmp[k++] = a[i++];
      while (j < hi) tmp[k++] = a[j++];
    }
    memcpy(a, tmp, sizeof(double) * n);
  }
}
int main(void) {
  printf("# sort: n insertion_s merge_s\n");
  for (int e = 4; e <= 20; ++e) {
    int n = 1 << e;
    double* src = malloc(8 * n); double* a = malloc(8 * n); double* t = malloc(8 * n);
    for (int i = 0; i < n; ++i) src[i] = rnd();
    double best_i = -1, best_m = 1e30;
    int reps = n <= 4096 ? 50 : (n <= 65536 ? 5 : 3);
    for (int r = 0; r < reps; ++r) {
      memcpy(a, src, 8 * n); double t0 = now(); merge_sort(a, t, n); double dt = now() - t0; if (dt < best_m) best_m = dt;
    }
    if (n <= (1 << 16)) {
      best_i = 1e30;
      for (int r = 0; r < (n <= 4096 ? 20 : 2); ++r) { memcpy(a, src, 8 * n); double t0 = now(); insertion(a, n); double dt = now() - t0; if (dt < best_i) best_i = dt; }
    }
    printf("%d %.6e %.6e\n", n, best_i, best_m);
    free(src); free(a); free(t);
  }
  printf("\n# chase: bytes ns_per_access\n");
  for (long bytes = 4096; bytes <= (1L << 30); bytes *= 2) {
    long n = bytes / 8; long* next = malloc(bytes); long* perm = malloc(bytes);
    for (long i = 0; i < n; ++i) perm[i] = i;
    for (long i = n - 1; i > 0; --i) { long j = (long)(rnd() * (i + 1)); long t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (long i = 0; i < n; ++i) next[perm[i]] = perm[(i + 1) % n];  // one random cycle through all slots
    long steps = 1L << 24; double best = 1e30; volatile long sink = 0;
    for (int k = 0; k < 3; ++k) {
      long p = perm[0]; double t0 = now();
      for (long r = 0; r < steps; ++r) p = next[p];
      double dt = (now() - t0) / steps * 1e9; if (dt < best) best = dt; sink += p;
    }
    printf("%ld %.4f\n", bytes, best);
    free(next); free(perm);
  }
  return 0;
}
