# 배열 영역 전체 측정 (numpy 대비)

환경: arm64 macOS, Python 3.15.0, numpy 2.5.4, stardust 86839a1 + 작업 중 변경, 원소 1,000,000개 기준, 10라운드, 2026-11-15 09:20, 걸린 시간 46분

배율 = numpy 시간 ÷ stardust 시간 (1보다 크면 stardust가 빠르다). numpy는 1스레드(BLAS만 여러 스레드).
시간은 7번 반복 중 최솟값을 다시 라운드 사이 최솟값으로 잡았다. **굵은 배율**은 1스레드에서 numpy보다 느린 줄이다.

- 케이스 380개. 1스레드에서 numpy보다 느린 것 85개, 10스레드에서도 느린 것 34개.
- 결과가 numpy와 다른 것: 0개

## 1스레드에서 numpy보다 느린 것 (배율 낮은 순)

| 묶음 | 케이스 | numpy | sd 1T | 배율 1T | sd 10T | 배율 10T |
|---|---|---:|---:|---:|---:|---:|
| convert | `asarray numpy f64 .T` | 12 ns | 146 ns | 0.09 | 145 ns | 0.09 |
| convert | `asarray numpy f64 (view)` | 12 ns | 137 ns | 0.09 | 133 ns | 0.09 |
| convert | `asarray numpy f64 strided` | 12 ns | 134 ns | 0.09 | 132 ns | 0.09 |
| convert | `numpy.asarray(sd array)` | 13 ns | 106 ns | 0.12 | 105 ns | 0.12 |
| tiny | `finfo(float64)` | 111 ns | 377 ns | 0.29 | 361 ns | 0.31 |
| convert | `from_dlpack numpy` | 114 ns | 252 ns | 0.45 | 248 ns | 0.46 |
| index | `f2[:, 5] (column)` | 48 ns | 103 ns | 0.47 | 102 ns | 0.47 |
| index | `f[1000:2000] (view)` | 43 ns | 88 ns | 0.49 | 87 ns | 0.49 |
| index | `f[::2] (view)` | 43 ns | 87 ns | 0.50 | 87 ns | 0.50 |
| tiny | `sm[1:, ::2]` | 51 ns | 101 ns | 0.51 | 101 ns | 0.51 |
| index | `f2[5] (row view)` | 42 ns | 71 ns | 0.59 | 71 ns | 0.59 |
| tiny | `sm[1]` | 42 ns | 71 ns | 0.60 | 71 ns | 0.60 |
| tiny | `sm.T` | 39 ns | 65 ns | 0.61 | 64 ns | 0.61 |
| tiny | `asarray(s3) (no-op)` | 12 ns | 20 ns | 0.63 | 20 ns | 0.63 |
| tiny | `zeros(3)` | 60 ns | 86 ns | 0.70 | 86 ns | 0.70 |
| tiny | `empty(3)` | 60 ns | 85 ns | 0.71 | 83 ns | 0.72 |
| creation | `empty(N)` | 105 ns | 137 ns | 0.77 | 130 ns | 0.81 |
| manip | `unstack(f2)` | 46.9 µs | 60.5 µs | 0.78 | 60.3 µs | 0.78 |
| unary i64 | `sign` | 97.4 µs | 125.3 µs | 0.78 | 29.5 µs | 3.30 |
| unary i64 | `abs` | 94.1 µs | 118.9 µs | 0.79 | 28.9 µs | 3.26 |
| manip | `diff(f)` | 106.0 µs | 131.0 µs | 0.81 | 30.3 µs | 3.49 |
| tiny | `sqrt(s3)` | 121 ns | 146 ns | 0.83 | 147 ns | 0.83 |
| tiny | `s3 + t3s` | 152 ns | 181 ns | 0.84 | 182 ns | 0.83 |
| tiny | `s3 < t3s` | 148 ns | 177 ns | 0.84 | 180 ns | 0.82 |
| tiny | `exp(s3)` | 124 ns | 148 ns | 0.84 | 152 ns | 0.82 |
| binary layout | `ft + f2 (mixed order)` | 456.7 µs | 534.1 µs | 0.86 | 67.0 µs | 6.82 |
| matmul | `vecdot(f2, g2)` | 130.6 µs | 152.6 µs | 0.86 | 22.4 µs | 5.83 |
| unary f64 | `acosh` | 2.49 ms | 2.90 ms | 0.86 | 317.7 µs | 7.85 |
| manip | `astype(f, M.complex128)` | 158.0 µs | 182.2 µs | 0.87 | 59.7 µs | 2.64 |
| binary mixed | `i + 1 (scalar)` | 94.2 µs | 108.4 µs | 0.87 | 29.3 µs | 3.21 |
| size 1e3 | `argsort(a) (1e3)` | 7.6 µs | 8.5 µs | 0.89 | 8.6 µs | 0.89 |
| binary mid | `fm + gm (1e4)` | 1.7 µs | 1.8 µs | 0.90 | 1.9 µs | 0.87 |
| assign | `gz[idx] = 1.0 (scatter 1e5)` | 95.6 µs | 105.5 µs | 0.91 | 80.4 µs | 1.19 |
| search | `take(f2, rows, axis=0)` | 111.5 µs | 122.3 µs | 0.91 | 30.8 µs | 3.62 |
| tiny | `len(s3)` | 7 ns | 7 ns | 0.93 | 7 ns | 0.93 |
| convert | `asarray list 3` | 134 ns | 143 ns | 0.93 | 142 ns | 0.94 |
| assign | `gz[idxn] = 1.0 (scatter 1e6)` | 960.1 µs | 1.03 ms | 0.93 | 812.1 µs | 1.18 |
| unary c128 | `square` | 266.0 µs | 282.6 µs | 0.94 | 65.2 µs | 4.08 |
| stats | `nanquantile(fn, q5)` | 11.39 ms | 12.04 ms | 0.95 | 3.89 ms | 2.93 |
| binary mixed | `c * f` | 608.4 µs | 642.7 µs | 0.95 | 110.6 µs | 5.50 |
| tiny | `concat([s3, t3s])` | 241 ns | 254 ns | 0.95 | 260 ns | 0.93 |
| tiny | `s3.shape` | 18 ns | 19 ns | 0.95 | 19 ns | 0.96 |
| manip | `tile(v, (1000,))` | 71.9 µs | 75.4 µs | 0.95 | 29.0 µs | 2.48 |
| binary c128 | `add` | 325.5 µs | 341.2 µs | 0.95 | 122.6 µs | 2.65 |
| convert | `asarray numpy f32` | 88.8 µs | 92.5 µs | 0.96 | 29.4 µs | 3.03 |
| manip | `roll(f2, 1, axis=1)` | 159.9 µs | 166.3 µs | 0.96 | 32.5 µs | 4.92 |
| assign | `x[1:] += x[:-1] (overlap)` | 237.6 µs | 246.8 µs | 0.96 | 99.3 µs | 2.39 |
| manip | `concat([f2, g2], axis=1)` | 342.6 µs | 354.7 µs | 0.97 | 76.8 µs | 4.46 |
| size 1e3 | `i + j (1e3)` | 285 ns | 294 ns | 0.97 | 294 ns | 0.97 |
| reduce axis | `sum(fst) (strided)` | 149.7 µs | 154.2 µs | 0.97 | 26.4 µs | 5.68 |
| reduce axis | `mean(f2, axis=0)` | 129.3 µs | 131.7 µs | 0.98 | 32.9 µs | 3.94 |
| reduce axis | `sum(f2, axis=0)` | 128.3 µs | 130.5 µs | 0.98 | 31.4 µs | 4.08 |
| reduce axis | `max(f2, axis=0)` | 128.0 µs | 130.0 µs | 0.99 | 31.7 µs | 4.04 |
| binary i64 | `bitwise_left_shift` | 147.1 µs | 149.2 µs | 0.99 | 37.6 µs | 3.91 |
| creation | `full(N, 2.5)` | 57.1 µs | 57.9 µs | 0.99 | 57.8 µs | 0.99 |
| manip | `concat([f, g])` | 187.7 µs | 190.0 µs | 0.99 | 69.9 µs | 2.68 |
| matmul | `matmul batch 1000 x 8x8` | 70.6 µs | 71.4 µs | 0.99 | 14.0 µs | 5.06 |
| size 1e7 | `a * 2.0 (1e7)` | 1.30 ms | 1.31 ms | 0.99 | 716.5 µs | 1.81 |
| binary c128 | `multiply` | 319.6 µs | 323.0 µs | 0.99 | 123.9 µs | 2.58 |
| creation | `full_like(f, 2.5)` | 57.3 µs | 57.9 µs | 0.99 | 57.0 µs | 1.00 |
| search | `searchsorted(srt, f)` | 53.45 ms | 53.95 ms | 0.99 | 6.33 ms | 8.44 |
| size 1e3 | `a + b (1e3)` | 286 ns | 288 ns | 0.99 | 290 ns | 0.99 |
| size 1e7 | `sum(a) (1e7)` | 1.19 ms | 1.20 ms | 0.99 | 468.6 µs | 2.54 |
| binary i64 | `add` | 145.8 µs | 146.8 µs | 0.99 | 37.4 µs | 3.90 |
| binary i64 | `bitwise_or` | 146.1 µs | 147.0 µs | 0.99 | 37.0 µs | 3.95 |
| binary i64 | `bitwise_xor` | 146.1 µs | 146.9 µs | 0.99 | 37.2 µs | 3.92 |
| unary i64 | `negative` | 94.6 µs | 95.1 µs | 1.00 | 29.0 µs | 3.26 |
| size 1e7 | `a + b (1e7)` | 2.01 ms | 2.02 ms | 1.00 | 1.14 ms | 1.76 |
| assign | `fw[::2] = 1.0 (strided fill)` | 121.9 µs | 122.4 µs | 1.00 | 19.3 µs | 6.32 |
| size 1e5 | `copy (1e5)` | 9.5 µs | 9.6 µs | 1.00 | 5.4 µs | 1.78 |
| creation | `asarray(copy=True)` | 91.1 µs | 91.3 µs | 1.00 | 33.9 µs | 2.69 |
| unary f64 | `ceil` | 93.9 µs | 94.2 µs | 1.00 | 28.8 µs | 3.26 |
| convert | `asarray numpy f32 rows m[:,1:1001]` | 137.1 µs | 137.4 µs | 1.00 | 27.0 µs | 5.08 |
| size 1e7 | `i + j (1e7)` | 1.95 ms | 1.96 ms | 1.00 | 1.15 ms | 1.70 |
| creation | `zeros(N)` | 28.1 µs | 28.1 µs | 1.00 | 28.1 µs | 1.00 |
| matmul | `matmul 512` | 321.2 µs | 321.8 µs | 1.00 | 321.7 µs | 1.00 |
| unary f64 | `negative` | 132.1 µs | 132.2 µs | 1.00 | 28.9 µs | 4.58 |
| unary f64 | `expm1` | 1.86 ms | 1.86 ms | 1.00 | 205.9 µs | 9.02 |
| convert | `asarray numpy >f8` | 94.3 µs | 94.3 µs | 1.00 | 30.2 µs | 3.12 |
| unary f64 | `log1p` | 2.48 ms | 2.48 ms | 1.00 | 272.1 µs | 9.10 |
| size 1e5 | `i + j (1e5)` | 14.2 µs | 14.3 µs | 1.00 | 5.5 µs | 2.59 |
| binary f64 | `minimum` | 145.9 µs | 145.9 µs | 1.00 | 37.0 µs | 3.94 |
| unary i64 | `square` | 254.2 µs | 254.2 µs | 1.00 | 35.5 µs | 7.15 |
| binary i64 | `bitwise_right_shift` | 149.5 µs | 149.5 µs | 1.00 | 37.8 µs | 3.96 |
| unary f64 | `square` | 94.4 µs | 94.4 µs | 1.00 | 28.9 µs | 3.27 |

## 전체

| 묶음 | 케이스 | numpy | sd 1T | 배율 1T | sd 10T | 배율 10T | 결과 확인 |
|---|---|---:|---:|---:|---:|---:|---|
| creation | `zeros(N)` | 28.1 µs | 28.1 µs | **1.00** | 28.1 µs | 1.00 | ok |
| creation | `ones(N)` | 57.5 µs | 56.8 µs | 1.01 | 56.9 µs | 1.01 | ok |
| creation | `empty(N)` | 105 ns | 137 ns | **0.77** | 130 ns | 0.81 | ok |
| creation | `full(N, 2.5)` | 57.1 µs | 57.9 µs | **0.99** | 57.8 µs | 0.99 | ok |
| creation | `zeros_like(f)` | 57.3 µs | 28.1 µs | 2.03 | 28.1 µs | 2.03 | ok |
| creation | `ones_like(f)` | 57.2 µs | 57.0 µs | 1.00 | 57.3 µs | 1.00 | ok |
| creation | `empty_like(f)` | 144 ns | 131 ns | 1.10 | 129 ns | 1.12 | ok |
| creation | `full_like(f, 2.5)` | 57.3 µs | 57.9 µs | **0.99** | 57.0 µs | 1.00 | ok |
| creation | `arange(N)` | 86.5 µs | 86.0 µs | 1.01 | 85.8 µs | 1.01 | ok |
| creation | `linspace(0.0, 1.0, N)` | 458.2 µs | 122.6 µs | 3.74 | 123.4 µs | 3.71 | ok |
| creation | `eye(1000)` | 35.4 µs | 32.3 µs | 1.10 | 34.1 µs | 1.04 | ok |
| creation | `meshgrid(v, v)` | 142.7 µs | 129.4 µs | 1.10 | 129.0 µs | 1.11 | ok |
| creation | `tril(f2)` | 346.1 µs | 182.9 µs | 1.89 | 145.8 µs | 2.37 | ok |
| creation | `triu(f2)` | 350.5 µs | 164.2 µs | 2.13 | 111.1 µs | 3.16 | ok |
| creation | `zeros((1000, 1000))` | 28.3 µs | 28.1 µs | 1.01 | 28.2 µs | 1.00 | ok |
| creation | `arange(N, float64)` | 232.7 µs | 121.4 µs | 1.92 | 123.7 µs | 1.88 | ok |
| creation | `asarray(copy=True)` | 91.1 µs | 91.3 µs | **1.00** | 33.9 µs | 2.69 | ok |
| convert | `asarray numpy f64 (view)` | 12 ns | 137 ns | **0.09** | 133 ns | 0.09 | ok |
| convert | `asarray numpy f64 strided` | 12 ns | 134 ns | **0.09** | 132 ns | 0.09 | ok |
| convert | `asarray numpy f64 .T` | 12 ns | 146 ns | **0.09** | 145 ns | 0.09 | ok |
| convert | `asarray numpy f32` | 88.8 µs | 92.5 µs | **0.96** | 29.4 µs | 3.03 | ok |
| convert | `asarray numpy f32 .T (np keeps F)` | 132.0 µs | 129.4 µs | 1.02 | 26.6 µs | 4.97 | ok |
| convert | `asarray numpy f32 .T (np C order)` | 402.2 µs | 130.2 µs | 3.09 | 26.5 µs | 15.16 | ok |
| convert | `asarray numpy f32 rows m[:,1:1001]` | 137.1 µs | 137.4 µs | **1.00** | 27.0 µs | 5.08 | ok |
| convert | `asarray numpy >f8` | 94.3 µs | 94.3 µs | **1.00** | 30.2 µs | 3.12 | ok |
| convert | `asarray numpy >f8 2-D [::2]` | 107.6 µs | 105.7 µs | 1.02 | 30.0 µs | 3.59 | ok |
| convert | `asarray numpy i32` | 79.2 µs | 77.7 µs | 1.02 | 29.4 µs | 2.69 | ok |
| convert | `asarray list 3` | 134 ns | 143 ns | **0.93** | 142 ns | 0.94 | ok |
| convert | `asarray list 1e4` | 140.0 µs | 63.9 µs | 2.19 | 63.9 µs | 2.19 | ok |
| convert | `from_dlpack numpy` | 114 ns | 252 ns | **0.45** | 248 ns | 0.46 | ok |
| convert | `numpy.asarray(sd array)` | 13 ns | 106 ns | **0.12** | 105 ns | 0.12 | ok |
| manip | `reshape(f, (1000, 1000))` | 275 ns | 140 ns | 1.97 | 141 ns | 1.95 | ok |
| manip | `reshape(ft, (N,))` | 462.5 µs | 310.5 µs | 1.49 | 52.4 µs | 8.83 | ok |
| manip | `permute_dims(t3, (2, 0, 1))` | 180 ns | 162 ns | 1.11 | 161 ns | 1.12 | ok |
| manip | `matrix_transpose(f2)` | 250 ns | 84 ns | 2.97 | 84 ns | 2.99 | ok |
| manip | `moveaxis(t3, 0, -1)` | 1.1 µs | 185 ns | 5.89 | 185 ns | 5.87 | ok |
| manip | `squeeze(M.reshape(f, (1, N)), axis=0)` | 422 ns | 246 ns | 1.71 | 246 ns | 1.72 | ok |
| manip | `expand_dims(f, axis=0)` | 610 ns | 97 ns | 6.27 | 97 ns | 6.27 | ok |
| manip | `broadcast_to(v, (1000, 1000))` | 993 ns | 113 ns | 8.77 | 111 ns | 8.91 | ok |
| manip | `broadcast_arrays(f2, v)` | 1.7 µs | 225 ns | 7.67 | 223 ns | 7.76 | ok |
| manip | `concat([f, g])` | 187.7 µs | 190.0 µs | **0.99** | 69.9 µs | 2.68 | ok |
| manip | `concat([f2, g2], axis=1)` | 342.6 µs | 354.7 µs | **0.97** | 76.8 µs | 4.46 | ok |
| manip | `stack([f, g])` | 191.0 µs | 188.7 µs | 1.01 | 69.9 µs | 2.73 | ok |
| manip | `flip(f)` | 179 ns | 82 ns | 2.18 | 82 ns | 2.18 | ok |
| manip | `roll(f, 1)` | 95.7 µs | 95.2 µs | 1.01 | 34.6 µs | 2.76 | ok |
| manip | `roll(f2, 1, axis=1)` | 159.9 µs | 166.3 µs | **0.96** | 32.5 µs | 4.92 | ok |
| manip | `repeat(f, 2)` | 496.6 µs | 204.1 µs | 2.43 | 61.2 µs | 8.11 | ok |
| manip | `tile(v, (1000,))` | 71.9 µs | 75.4 µs | **0.95** | 29.0 µs | 2.48 | ok |
| manip | `unstack(f2)` | 46.9 µs | 60.5 µs | **0.78** | 60.3 µs | 0.78 | ok |
| manip | `astype(f, M.int64)` | 267.0 µs | 129.2 µs | 2.07 | 28.6 µs | 9.32 | ok |
| manip | `astype(i, M.float64)` | 248.9 µs | 98.1 µs | 2.54 | 28.8 µs | 8.64 | ok |
| manip | `astype(f, M.complex128)` | 158.0 µs | 182.2 µs | **0.87** | 59.7 µs | 2.64 | ok |
| manip | `astype(b, M.float64)` | 247.6 µs | 151.2 µs | 1.64 | 27.2 µs | 9.09 | ok |
| manip | `astype(ft, M.float64)` | 94.3 µs | 94.1 µs | 1.00 | 34.0 µs | 2.77 | ok |
| manip | `ravel(ft) (copy)` | 426.6 µs | 306.6 µs | 1.39 | 51.2 µs | 8.34 | ok |
| manip | `diff(f)` | 106.0 µs | 131.0 µs | **0.81** | 30.3 µs | 3.49 | ok |
| binary f64 | `add` | 199.1 µs | 149.7 µs | 1.33 | 36.9 µs | 5.39 | ok |
| binary f64 | `subtract` | 149.3 µs | 146.1 µs | 1.02 | 36.8 µs | 4.06 | ok |
| binary f64 | `multiply` | 146.5 µs | 146.0 µs | 1.00 | 37.3 µs | 3.93 | ok |
| binary f64 | `divide` | 199.6 µs | 146.4 µs | 1.36 | 36.4 µs | 5.48 | ok |
| binary f64 | `floor_divide` | 6.43 ms | 6.03 ms | 1.07 | 671.6 µs | 9.57 | ok |
| binary f64 | `remainder` | 6.48 ms | 5.61 ms | 1.15 | 611.9 µs | 10.59 | ok |
| binary f64 | `pow` | 4.61 ms | 2.64 ms | 1.74 | 293.1 µs | 15.71 | ok |
| binary f64 | `atan2` | 12.50 ms | 1.87 ms | 6.69 | 211.8 µs | 59.02 | ok |
| binary f64 | `maximum` | 200.6 µs | 147.6 µs | 1.36 | 37.5 µs | 5.35 | ok |
| binary f64 | `minimum` | 145.9 µs | 145.9 µs | **1.00** | 37.0 µs | 3.94 | ok |
| binary f64 | `copysign` | 252.7 µs | 146.6 µs | 1.72 | 37.0 µs | 6.83 | ok |
| binary f64 | `hypot` | 1.48 ms | 1.23 ms | 1.20 | 139.8 µs | 10.55 | ok |
| binary f64 | `logaddexp` | 10.28 ms | 9.86 ms | 1.04 | 1.06 ms | 9.69 | ok |
| binary f64 | `nextafter` | 5.16 ms | 647.5 µs | 7.97 | 76.6 µs | 67.41 | ok |
| binary f64 | `equal` | 200.7 µs | 134.9 µs | 1.49 | 25.1 µs | 8.00 | ok |
| binary f64 | `less` | 200.8 µs | 134.0 µs | 1.50 | 25.0 µs | 8.04 | ok |
| binary f64 | `greater_equal` | 200.4 µs | 134.0 µs | 1.50 | 24.9 µs | 8.05 | ok |
| binary bool | `logical_and` | 25.3 µs | 17.9 µs | 1.42 | 6.2 µs | 4.09 | ok |
| binary bool | `logical_or` | 25.6 µs | 17.9 µs | 1.43 | 6.2 µs | 4.15 | ok |
| binary bool | `logical_xor` | 18.4 µs | 17.9 µs | 1.03 | 6.2 µs | 2.96 | ok |
| binary i64 | `add` | 145.8 µs | 146.8 µs | **0.99** | 37.4 µs | 3.90 | ok |
| binary i64 | `multiply` | 268.2 µs | 267.4 µs | 1.00 | 39.5 µs | 6.78 | ok |
| binary i64 | `floor_divide` | 739.6 µs | 596.0 µs | 1.24 | 75.8 µs | 9.76 | ok |
| binary i64 | `remainder` | 880.0 µs | 696.5 µs | 1.26 | 83.4 µs | 10.55 | ok |
| binary i64 | `maximum` | 149.9 µs | 147.2 µs | 1.02 | 37.4 µs | 4.01 | ok |
| binary i64 | `equal` | 137.3 µs | 135.4 µs | 1.01 | 24.8 µs | 5.53 | ok |
| binary i64 | `bitwise_and` | 147.6 µs | 147.4 µs | 1.00 | 37.6 µs | 3.93 | ok |
| binary i64 | `bitwise_or` | 146.1 µs | 147.0 µs | **0.99** | 37.0 µs | 3.95 | ok |
| binary i64 | `bitwise_xor` | 146.1 µs | 146.9 µs | **0.99** | 37.2 µs | 3.92 | ok |
| binary i64 | `bitwise_left_shift` | 147.1 µs | 149.2 µs | **0.99** | 37.6 µs | 3.91 | ok |
| binary i64 | `bitwise_right_shift` | 149.5 µs | 149.5 µs | **1.00** | 37.8 µs | 3.96 | ok |
| binary i64 | `pow(i, 2)` | 261.0 µs | 250.7 µs | 1.04 | 35.5 µs | 7.35 | ok |
| binary c128 | `add` | 325.5 µs | 341.2 µs | **0.95** | 122.6 µs | 2.65 | ok |
| binary c128 | `multiply` | 319.6 µs | 323.0 µs | **0.99** | 123.9 µs | 2.58 | ok |
| binary c128 | `divide` | 4.41 ms | 1.62 ms | 2.73 | 181.5 µs | 24.30 | ok |
| binary c128 | `equal` | 275.9 µs | 271.2 µs | 1.02 | 88.3 µs | 3.12 | ok |
| binary c128 | `pow` | 35.50 ms | 34.51 ms | 1.03 | 3.73 ms | 9.53 | ok |
| binary mixed | `f * i (cast)` | 437.0 µs | 220.3 µs | 1.98 | 44.6 µs | 9.80 | ok |
| binary mixed | `f * i + g` | 607.8 µs | 432.8 µs | 1.40 | 142.9 µs | 4.25 | ok |
| binary mixed | `f + 1.0 (scalar)` | 94.1 µs | 94.1 µs | 1.00 | 29.5 µs | 3.19 | ok |
| binary mixed | `i + 1 (scalar)` | 94.2 µs | 108.4 µs | **0.87** | 29.3 µs | 3.21 | ok |
| binary mixed | `c * f` | 608.4 µs | 642.7 µs | **0.95** | 110.6 µs | 5.50 | ok |
| binary layout | `f2 + v (broadcast)` | 240.2 µs | 108.7 µs | 2.21 | 29.3 µs | 8.19 | ok |
| binary layout | `f2 + v[:, None] (broadcast col)` | 217.4 µs | 105.0 µs | 2.07 | 29.5 µs | 7.37 | ok |
| binary layout | `ft + 1.0 (transposed)` | 94.7 µs | 94.4 µs | 1.00 | 29.1 µs | 3.25 | ok |
| binary layout | `ft + f2 (mixed order)` | 456.7 µs | 534.1 µs | **0.86** | 67.0 µs | 6.82 | ok |
| binary layout | `fst + 1.0 (strided)` | 252.2 µs | 250.2 µs | 1.01 | 38.5 µs | 6.54 | ok |
| binary layout | `fst + f (strided + contiguous)` | 262.2 µs | 256.2 µs | 1.02 | 70.9 µs | 3.70 | ok |
| binary mid | `fm + gm (1e4)` | 1.7 µs | 1.8 µs | **0.90** | 1.9 µs | 0.87 | ok |
| binary mid | `fm * 2.0 (1e4)` | 1.1 µs | 986 ns | 1.10 | 939 ns | 1.15 | ok |
| unary f64 | `negative` | 132.1 µs | 132.2 µs | **1.00** | 28.9 µs | 4.58 | ok |
| unary f64 | `positive` | 247.9 µs | 94.0 µs | 2.64 | 28.8 µs | 8.60 | ok |
| unary f64 | `abs` | 94.2 µs | 94.0 µs | 1.00 | 28.9 µs | 3.26 | ok |
| unary f64 | `square` | 94.4 µs | 94.4 µs | **1.00** | 28.9 µs | 3.27 | ok |
| unary f64 | `sqrt` | 242.6 µs | 239.3 µs | 1.01 | 34.9 µs | 6.95 | ok |
| unary f64 | `sign` | 298.2 µs | 218.3 µs | 1.37 | 33.3 µs | 8.96 | ok |
| unary f64 | `exp` | 1.49 ms | 1.07 ms | 1.40 | 121.6 µs | 12.24 | ok |
| unary f64 | `expm1` | 1.86 ms | 1.86 ms | **1.00** | 205.9 µs | 9.02 | ok |
| unary f64 | `log` | 3.14 ms | 1.08 ms | 2.90 | 123.2 µs | 25.50 | ok |
| unary f64 | `log1p` | 2.48 ms | 2.48 ms | **1.00** | 272.1 µs | 9.10 | ok |
| unary f64 | `log2` | 3.44 ms | 1.03 ms | 3.33 | 118.0 µs | 29.15 | ok |
| unary f64 | `log10` | 3.56 ms | 1.08 ms | 3.28 | 123.4 µs | 28.83 | ok |
| unary f64 | `sin` | 3.81 ms | 804.6 µs | 4.73 | 93.1 µs | 40.93 | ok |
| unary f64 | `cos` | 3.73 ms | 3.68 ms | 1.01 | 399.5 µs | 9.33 | ok |
| unary f64 | `tan` | 4.16 ms | 4.12 ms | 1.01 | 444.5 µs | 9.35 | ok |
| unary f64 | `atan` | 7.63 ms | 1.12 ms | 6.81 | 127.0 µs | 60.05 | ok |
| unary f64 | `sinh` | 2.61 ms | 1.02 ms | 2.56 | 116.3 µs | 22.44 | ok |
| unary f64 | `cosh` | 2.48 ms | 958.5 µs | 2.58 | 109.7 µs | 22.58 | ok |
| unary f64 | `tanh` | 2.22 ms | 1.20 ms | 1.86 | 135.0 µs | 16.43 | ok |
| unary f64 | `asinh` | 4.78 ms | 3.57 ms | 1.34 | 394.0 µs | 12.13 | ok |
| unary f64 | `floor` | 94.3 µs | 94.1 µs | 1.00 | 28.9 µs | 3.27 | ok |
| unary f64 | `ceil` | 93.9 µs | 94.2 µs | **1.00** | 28.8 µs | 3.26 | ok |
| unary f64 | `trunc` | 94.4 µs | 94.1 µs | 1.00 | 29.1 µs | 3.24 | ok |
| unary f64 | `round` | 95.5 µs | 94.2 µs | 1.01 | 28.8 µs | 3.32 | ok |
| unary f64 | `isnan` | 71.7 µs | 71.6 µs | 1.00 | 18.5 µs | 3.87 | ok |
| unary f64 | `isinf` | 71.5 µs | 71.5 µs | 1.00 | 18.5 µs | 3.87 | ok |
| unary f64 | `isfinite` | 71.8 µs | 71.5 µs | 1.00 | 18.4 µs | 3.90 | ok |
| unary f64 | `signbit` | 71.6 µs | 71.5 µs | 1.00 | 18.4 µs | 3.88 | ok |
| unary f64 | `reciprocal` | 124.5 µs | 123.6 µs | 1.01 | 29.1 µs | 4.28 | ok |
| unary f64 | `asin` | 5.34 ms | 936.8 µs | 5.70 | 107.1 µs | 49.82 | ok |
| unary f64 | `acos` | 6.94 ms | 1.27 ms | 5.45 | 143.7 µs | 48.33 | ok |
| unary f64 | `atanh` | 5.34 ms | 2.18 ms | 2.46 | 240.9 µs | 22.19 | ok |
| unary f64 | `acosh` | 2.49 ms | 2.90 ms | **0.86** | 317.7 µs | 7.85 | ok |
| unary f64 | `sign (both signs)` | 3.54 ms | 218.4 µs | 16.21 | 33.2 µs | 106.70 | ok |
| unary f64 | `exp strided` | 1.49 ms | 1.26 ms | 1.19 | 142.4 µs | 10.46 | ok |
| unary f64 | `sqrt transposed` | 248.1 µs | 244.3 µs | 1.02 | 34.9 µs | 7.11 | ok |
| unary f64 | `exp (1e4)` | 15.1 µs | 10.9 µs | 1.39 | 10.9 µs | 1.39 | ok |
| unary c128 | `abs` | 1.63 ms | 1.27 ms | 1.29 | 143.0 µs | 11.38 | ok |
| unary c128 | `exp` | 8.19 ms | 7.14 ms | 1.15 | 771.8 µs | 10.61 | ok |
| unary c128 | `sqrt` | 2.23 ms | 1.54 ms | 1.45 | 175.1 µs | 12.74 | ok |
| unary c128 | `log` | 22.73 ms | 21.04 ms | 1.08 | 2.23 ms | 10.19 | ok |
| unary c128 | `sin` | 10.34 ms | 9.40 ms | 1.10 | 1.01 ms | 10.25 | ok |
| unary c128 | `conj` | 195.5 µs | 194.8 µs | 1.00 | 65.8 µs | 2.97 | ok |
| unary c128 | `real` | 84 ns | 66 ns | 1.28 | 66 ns | 1.29 | ok |
| unary c128 | `imag` | 84 ns | 65 ns | 1.30 | 65 ns | 1.29 | ok |
| unary c128 | `square` | 266.0 µs | 282.6 µs | **0.94** | 65.2 µs | 4.08 | ok |
| unary c128 | `isnan` | 320.9 µs | 202.8 µs | 1.58 | 28.9 µs | 11.10 | ok |
| unary i64 | `abs` | 94.1 µs | 118.9 µs | **0.79** | 28.9 µs | 3.26 | ok |
| unary i64 | `negative` | 94.6 µs | 95.1 µs | **1.00** | 29.0 µs | 3.26 | ok |
| unary i64 | `square` | 254.2 µs | 254.2 µs | **1.00** | 35.5 µs | 7.15 | ok |
| unary i64 | `sign` | 97.4 µs | 125.3 µs | **0.78** | 29.5 µs | 3.30 | ok |
| unary i64 | `bitwise_invert` | 115.4 µs | 94.3 µs | 1.22 | 29.2 µs | 3.95 | ok |
| unary bool | `logical_not` | 18.7 µs | 16.8 µs | 1.11 | 5.7 µs | 3.25 | ok |
| select | `clip(f, 0.6, 1.2)` | 253.8 µs | 95.6 µs | 2.66 | 29.2 µs | 8.69 | ok |
| select | `clip(i, -10, 10)` | 135.0 µs | 131.3 µs | 1.03 | 29.8 µs | 4.53 | ok |
| select | `where(b, f, g)` | 274.9 µs | 215.1 µs | 1.28 | 51.2 µs | 5.37 | ok |
| select | `where(b, f, 0.0)` | 261.6 µs | 156.3 µs | 1.67 | 27.6 µs | 9.48 | ok |
| reduce f64 | `sum` | 112.9 µs | 111.0 µs | 1.02 | 22.9 µs | 4.93 | ok |
| reduce f64 | `prod` | 781.4 µs | 108.1 µs | 7.23 | 19.6 µs | 39.85 | ok |
| reduce f64 | `min` | 75.2 µs | 71.7 µs | 1.05 | 20.8 µs | 3.62 | ok |
| reduce f64 | `max` | 75.3 µs | 71.6 µs | 1.05 | 20.8 µs | 3.62 | ok |
| reduce f64 | `mean` | 113.4 µs | 111.2 µs | 1.02 | 22.9 µs | 4.95 | ok |
| reduce f64 | `var` | 444.0 µs | 313.6 µs | 1.42 | 55.2 µs | 8.05 | ok |
| reduce f64 | `std` | 448.0 µs | 313.5 µs | 1.43 | 55.3 µs | 8.10 | ok |
| reduce f64 | `argmin` | 550.6 µs | 480.8 µs | 1.15 | 57.3 µs | 9.62 | ok |
| reduce f64 | `argmax` | 550.3 µs | 472.9 µs | 1.16 | 57.2 µs | 9.61 | ok |
| reduce i64 | `sum` | 72.6 µs | 72.0 µs | 1.01 | 20.6 µs | 3.52 | ok |
| reduce i64 | `min` | 76.4 µs | 71.7 µs | 1.07 | 19.9 µs | 3.83 | ok |
| reduce i64 | `max` | 76.6 µs | 71.6 µs | 1.07 | 19.9 µs | 3.84 | ok |
| reduce i64 | `mean` | 361.3 µs | 205.1 µs | 1.76 | 31.0 µs | 11.66 | ok |
| reduce i64 | `var` | 929.3 µs | 524.9 µs | 1.77 | 96.6 µs | 9.62 | ok |
| reduce i64 | `argmax` | 125.5 µs | 122.7 µs | 1.02 | 23.3 µs | 5.38 | ok |
| reduce c128 | `sum` | 191.3 µs | 175.4 µs | 1.09 | 40.4 µs | 4.73 | ok |
| reduce c128 | `prod` | 3.87 ms | 278.6 µs | 13.90 | 36.4 µs | 106.31 | ok |
| reduce c128 | `mean` | 191.5 µs | 174.3 µs | 1.10 | 40.5 µs | 4.73 | ok |
| reduce c128 | `var` | 1.06 ms | 490.1 µs | 2.17 | 86.4 µs | 12.32 | ok |
| reduce bool | `any` | 488 ns | 109 ns | 4.50 | 111 ns | 4.38 | ok |
| reduce bool | `all` | 479 ns | 108 ns | 4.45 | 111 ns | 4.30 | ok |
| reduce bool | `sum(b)` | 155.6 µs | 132.6 µs | 1.17 | 22.0 µs | 7.06 | ok |
| reduce axis | `sum(f2, axis=0)` | 128.3 µs | 130.5 µs | **0.98** | 31.4 µs | 4.08 | ok |
| reduce axis | `sum(f2, axis=1)` | 118.7 µs | 114.4 µs | 1.04 | 24.2 µs | 4.91 | ok |
| reduce axis | `mean(f2, axis=0)` | 129.3 µs | 131.7 µs | **0.98** | 32.9 µs | 3.94 | ok |
| reduce axis | `mean(f2, axis=1)` | 119.8 µs | 115.6 µs | 1.04 | 25.3 µs | 4.74 | ok |
| reduce axis | `max(f2, axis=0)` | 128.0 µs | 130.0 µs | **0.99** | 31.7 µs | 4.04 | ok |
| reduce axis | `max(f2, axis=1)` | 83.5 µs | 75.4 µs | 1.11 | 19.8 µs | 4.21 | ok |
| reduce axis | `var(f2, axis=0)` | 646.8 µs | 446.6 µs | 1.45 | 91.0 µs | 7.11 | ok |
| reduce axis | `var(f2, axis=1)` | 587.8 µs | 322.8 µs | 1.82 | 60.9 µs | 9.65 | ok |
| reduce axis | `argmax(f2, axis=0)` | 1.01 ms | 948.0 µs | 1.07 | 108.2 µs | 9.34 | ok |
| reduce axis | `argmax(f2, axis=1)` | 528.6 µs | 464.9 µs | 1.14 | 57.2 µs | 9.24 | ok |
| reduce axis | `sum(t3, axis=(0, 2))` | 170.2 µs | 168.1 µs | 1.01 | 41.0 µs | 4.16 | ok |
| reduce axis | `sum(ft) (transposed)` | 112.9 µs | 111.1 µs | 1.02 | 22.9 µs | 4.93 | ok |
| reduce axis | `sum(ft, axis=0)` | 118.8 µs | 114.5 µs | 1.04 | 24.2 µs | 4.92 | ok |
| reduce axis | `sum(fst) (strided)` | 149.7 µs | 154.2 µs | **0.97** | 26.4 µs | 5.68 | ok |
| reduce axis | `sum(i2, axis=0)` | 278.7 µs | 130.2 µs | 2.14 | 31.5 µs | 8.86 | ok |
| reduce mid | `sum(fm) (1e4)` | 1.7 µs | 1.2 µs | 1.41 | 1.2 µs | 1.40 | ok |
| reduce mid | `var(fm) (1e4)` | 8.8 µs | 3.3 µs | 2.64 | 3.3 µs | 2.63 | ok |
| scan | `cumulative_sum(f)` | 2.14 ms | 383.8 µs | 5.59 | 69.9 µs | 30.65 | ok |
| scan | `cumulative_prod(g)` | 2.36 ms | 423.2 µs | 5.58 | 74.0 µs | 31.91 | ok |
| scan | `cumulative_sum(i)` | 1.51 ms | 264.1 µs | 5.70 | 86.1 µs | 17.49 | ok |
| scan | `cumulative_sum(f2, axis=0)` | 2.97 ms | 618.7 µs | 4.80 | 343.7 µs | 8.64 | ok |
| scan | `cumulative_sum(f2, axis=1)` | 2.26 ms | 531.8 µs | 4.26 | 64.7 µs | 35.00 | ok |
| scan | `cumulative_sum(c)` | 3.23 ms | 679.9 µs | 4.75 | 170.0 µs | 19.01 | ok |
| nan | `nansum` | 1.27 ms | 157.3 µs | 8.10 | 25.0 µs | 51.06 | ok |
| nan | `nanprod` | 2.01 ms | 115.6 µs | 17.41 | 19.5 µs | 103.29 | ok |
| nan | `nanmin` | 75.9 µs | 71.7 µs | 1.06 | 20.6 µs | 3.67 | ok |
| nan | `nanmax` | 75.9 µs | 71.7 µs | 1.06 | 20.7 µs | 3.66 | ok |
| nan | `nanmean` | 1.45 ms | 230.9 µs | 6.29 | 39.1 µs | 37.10 | ok |
| nan | `nanvar` | 3.31 ms | 540.7 µs | 6.12 | 79.5 µs | 41.66 | ok |
| nan | `nanstd` | 3.33 ms | 541.2 µs | 6.15 | 79.5 µs | 41.89 | ok |
| nan | `nanargmin` | 1.72 ms | 557.8 µs | 3.09 | 71.4 µs | 24.13 | ok |
| nan | `nanargmax` | 1.71 ms | 557.8 µs | 3.07 | 71.5 µs | 23.95 | ok |
| nan | `nancumulative_sum` | 3.31 ms | 402.6 µs | 8.23 | 71.7 µs | 46.18 | ok |
| nan | `nancumulative_prod` | 3.55 ms | 415.4 µs | 8.54 | 73.3 µs | 48.44 | ok |
| nan | `nansum(f2-like, axis=0)` | 1.29 ms | 131.4 µs | 9.80 | 32.9 µs | 39.20 | ok |
| count | `count_nonzero(b)` | 23.4 µs | 22.8 µs | 1.03 | 5.9 µs | 3.96 | ok |
| count | `count_nonzero(f)` | 248.0 µs | 93.9 µs | 2.64 | 15.8 µs | 15.69 | ok |
| count | `count_nonzero(f2, axis=0)` | 623.5 µs | 333.5 µs | 1.87 | 59.8 µs | 10.42 | ok |
| stats | `quantile(f, q5)` | 13.78 ms | 13.20 ms | 1.04 | 3.20 ms | 4.31 | ok |
| stats | `quantile(f, 0.5)` | 7.02 ms | 6.59 ms | 1.06 | 3.20 ms | 2.19 | ok |
| stats | `median(f)` | 7.31 ms | 6.58 ms | 1.11 | 3.19 ms | 2.29 | ok |
| stats | `median(f2, axis=0)` | 8.09 ms | 5.79 ms | 1.40 | 644.2 µs | 12.55 | ok |
| stats | `median(f2, axis=1)` | 7.51 ms | 5.49 ms | 1.37 | 628.0 µs | 11.96 | ok |
| stats | `nanquantile(fn, q5)` | 11.39 ms | 12.04 ms | **0.95** | 3.89 ms | 2.93 | ok |
| stats | `nanmedian(fn)` | 7.94 ms | 6.64 ms | 1.20 | 3.89 ms | 2.04 | ok |
| stats | `cov(1000 x 1000)` | 2.18 ms | 1.77 ms | 1.23 | 1.51 ms | 1.44 | ok |
| stats | `cov(20 x 1e5)` | 3.99 ms | 2.45 ms | 1.63 | 2.13 ms | 1.88 | ok |
| stats | `corrcoef(1000 x 1000)` | 3.00 ms | 2.04 ms | 1.47 | 1.56 ms | 1.92 | ok |
| stats | `bincount(iv)` | 997.5 µs | 351.0 µs | 2.84 | 84.3 µs | 11.84 | ok |
| stats | `bincount(iv, weights)` | 1.06 ms | 468.4 µs | 2.27 | 381.3 µs | 2.79 | ok |
| stats | `histogram(f, 100)` | 4.03 ms | 1.64 ms | 2.46 | 220.4 µs | 18.28 | ok |
| stats | `histogram(f, 100, range)` | 4.27 ms | 1.14 ms | 3.73 | 164.7 µs | 25.93 | ok |
| sort | `sort(f)` | 27.24 ms | 14.40 ms | 1.89 | 4.02 ms | 6.77 | ok |
| sort | `sort(f, stable=False)` | 27.23 ms | 14.43 ms | 1.89 | 4.02 ms | 6.77 | ok |
| sort | `sort(half zeros)` | 15.40 ms | 14.95 ms | 1.03 | 3.40 ms | 4.53 | ok |
| sort | `sort(i)` | 9.79 ms | 2.44 ms | 4.01 | 1.01 ms | 9.72 | ok |
| sort | `sort(i wide)` | 20.95 ms | 6.34 ms | 3.31 | 2.44 ms | 8.59 | ok |
| sort | `sort(f2, axis=1)` | 13.83 ms | 6.40 ms | 2.16 | 716.7 µs | 19.29 | ok |
| sort | `sort(f2, axis=0)` | 14.41 ms | 7.00 ms | 2.06 | 957.7 µs | 15.04 | ok |
| sort | `sort(f, descending)` | 27.47 ms | 14.61 ms | 1.88 | 4.05 ms | 6.78 | ok |
| sort | `argsort(f)` | 68.13 ms | 10.27 ms | 6.64 | 4.34 ms | 15.70 | ok |
| sort | `argsort(f, stable=False)` | 68.71 ms | 10.26 ms | 6.69 | 4.35 ms | 15.80 | ok |
| sort | `argsort(i)` | 36.47 ms | 2.76 ms | 13.23 | 1.27 ms | 28.80 | ok |
| sort | `argsort(i, stable=False)` | 36.41 ms | 2.76 ms | 13.17 | 1.27 ms | 28.57 | ok |
| sort | `argsort(f2, axis=1)` | 31.32 ms | 8.74 ms | 3.58 | 932.7 µs | 33.57 | ok |
| sort | `sort(fm) (1e4)` | 182.4 µs | 85.1 µs | 2.14 | 84.4 µs | 2.16 | ok |
| sort | `argsort(fm) (1e4)` | 187.5 µs | 88.9 µs | 2.11 | 89.0 µs | 2.11 | ok |
| unique | `unique_values(iv)` | 5.49 ms | 2.83 ms | 1.94 | 1.06 ms | 5.17 | ok |
| unique | `unique_counts(iv)` | 9.27 ms | 4.17 ms | 2.22 | 2.45 ms | 3.78 | ok |
| unique | `unique_inverse(iv)` | 39.17 ms | 4.21 ms | 9.31 | 2.50 ms | 15.68 | ok |
| unique | `unique_all(iv)` | 79.30 ms | 4.22 ms | 18.78 | 2.49 ms | 31.82 | ok |
| unique | `unique_values(f)` | 27.56 ms | 14.59 ms | 1.89 | 4.21 ms | 6.54 | ok |
| search | `searchsorted(srt, keys)` | 9.35 ms | 4.37 ms | 2.14 | 768.5 µs | 12.16 | ok |
| search | `searchsorted(srt, f)` | 53.45 ms | 53.95 ms | **0.99** | 6.33 ms | 8.44 | ok |
| search | `nonzero(b)` | 343.3 µs | 324.2 µs | 1.06 | 48.3 µs | 7.11 | ok |
| search | `nonzero(f2 > 0.5)` | 4.54 ms | 459.5 µs | 9.89 | 78.3 µs | 58.05 | ok |
| search | `take(f, idx)` | 149.5 µs | 68.6 µs | 2.18 | 11.5 µs | 13.02 | ok |
| search | `take(f2, rows, axis=0)` | 111.5 µs | 122.3 µs | **0.91** | 30.8 µs | 3.62 | ok |
| search | `take_along_axis(f2, idx2, axis=1)` | 2.04 ms | 438.8 µs | 4.65 | 70.2 µs | 29.07 | ok |
| index | `f[1000:2000] (view)` | 43 ns | 88 ns | **0.49** | 87 ns | 0.49 | ok |
| index | `f[::2] (view)` | 43 ns | 87 ns | **0.50** | 87 ns | 0.50 | ok |
| index | `f2[5] (row view)` | 42 ns | 71 ns | **0.59** | 71 ns | 0.59 | ok |
| index | `f2[:, 5] (column)` | 48 ns | 103 ns | **0.47** | 102 ns | 0.47 | ok |
| index | `f2[10, 20] (scalar)` | 33 ns | 25 ns | 1.36 | 25 ns | 1.36 | ok |
| index | `f[idx] (gather 1e5)` | 72.5 µs | 68.8 µs | 1.05 | 11.5 µs | 6.33 | ok |
| index | `f[idxn] (gather 1e6)` | 727.8 µs | 692.3 µs | 1.05 | 83.5 µs | 8.72 | ok |
| index | `f[b] (mask)` | 3.34 ms | 340.3 µs | 9.82 | 47.7 µs | 70.12 | ok |
| index | `f2[rows] (fancy rows)` | 158.5 µs | 121.9 µs | 1.30 | 30.8 µs | 5.14 | ok |
| index | `f2[f2 > 0.5] (2-D mask)` | 3.49 ms | 417.7 µs | 8.36 | 69.4 µs | 50.33 | ok |
| index | `s3[1] (tiny scalar)` | 28 ns | 23 ns | 1.20 | 23 ns | 1.20 | ok |
| assign | `gz[idx] = 1.0 (scatter 1e5)` | 95.6 µs | 105.5 µs | **0.91** | 80.4 µs | 1.19 | stmt |
| assign | `gz[idxn] = 1.0 (scatter 1e6)` | 960.1 µs | 1.03 ms | **0.93** | 812.1 µs | 1.18 | stmt |
| assign | `fw[b] = 0.0 (mask)` | 2.94 ms | 407.3 µs | 7.22 | 55.1 µs | 53.38 | stmt |
| assign | `fw[:] = 1.0 (fill)` | 61.9 µs | 61.7 µs | 1.00 | 28.5 µs | 2.17 | stmt |
| assign | `fw[:] = g (copy)` | 93.5 µs | 93.5 µs | 1.00 | 33.4 µs | 2.80 | stmt |
| assign | `fw[::2] = 1.0 (strided fill)` | 121.9 µs | 122.4 µs | **1.00** | 19.3 µs | 6.32 | stmt |
| assign | `x += g (in place)` | 164.5 µs | 164.1 µs | 1.00 | 29.0 µs | 5.67 | ok |
| assign | `x[1:] += x[:-1] (overlap)` | 237.6 µs | 246.8 µs | **0.96** | 99.3 µs | 2.39 | stmt |
| assign | `x *= 1.0000001` | 111.7 µs | 111.2 µs | 1.00 | 28.8 µs | 3.87 | ok |
| matmul | `matmul 512` | 321.2 µs | 321.8 µs | **1.00** | 321.7 µs | 1.00 | ok |
| matmul | `matmul 1000` | 2.01 ms | 2.01 ms | 1.00 | 2.01 ms | 1.00 | ok |
| matmul | `matmul(f2, v) (gemv)` | 13.6 µs | 13.5 µs | 1.01 | 13.5 µs | 1.01 | ok |
| matmul | `matmul(v, v) (dot)` | 644 ns | 277 ns | 2.32 | 276 ns | 2.34 | ok |
| matmul | `matmul batch 1000 x 8x8` | 70.6 µs | 71.4 µs | **0.99** | 14.0 µs | 5.06 | ok |
| matmul | `matmul int 200` | 2.55 ms | 1.55 ms | 1.64 | 215.2 µs | 11.87 | ok |
| matmul | `matmul(ft, f2) (transposed)` | 1.30 ms | 1.30 ms | 1.00 | 1.30 ms | 1.00 | ok |
| matmul | `vecdot(f2, g2)` | 130.6 µs | 152.6 µs | **0.86** | 22.4 µs | 5.83 | ok |
| matmul | `tensordot(t3, t3, 2)` | 379.6 µs | 373.7 µs | 1.02 | 374.6 µs | 1.01 | ok |
| size 1e3 | `a + b (1e3)` | 286 ns | 288 ns | **0.99** | 290 ns | 0.99 | ok |
| size 1e3 | `a * 2.0 (1e3)` | 357 ns | 288 ns | 1.24 | 291 ns | 1.23 | ok |
| size 1e3 | `exp(a) (1e3)` | 1.6 µs | 1.2 µs | 1.35 | 1.2 µs | 1.35 | ok |
| size 1e3 | `sin(a) (1e3)` | 2.5 µs | 949 ns | 2.65 | 954 ns | 2.64 | ok |
| size 1e3 | `sum(a) (1e3)` | 663 ns | 189 ns | 3.51 | 193 ns | 3.44 | ok |
| size 1e3 | `max(a) (1e3)` | 582 ns | 139 ns | 4.17 | 142 ns | 4.10 | ok |
| size 1e3 | `var(a) (1e3)` | 3.5 µs | 471 ns | 7.49 | 479 ns | 7.36 | ok |
| size 1e3 | `cumulative_sum(a) (1e3)` | 2.4 µs | 701 ns | 3.35 | 703 ns | 3.34 | ok |
| size 1e3 | `a < b (1e3)` | 353 ns | 266 ns | 1.33 | 269 ns | 1.31 | ok |
| size 1e3 | `a[a > 1.0] (1e3)` | 2.2 µs | 830 ns | 2.66 | 831 ns | 2.66 | ok |
| size 1e3 | `where(a > 1.0, a, b) (1e3)` | 985 ns | 581 ns | 1.69 | 586 ns | 1.68 | ok |
| size 1e3 | `sort(a) (1e3)` | 14.0 µs | 5.6 µs | 2.51 | 5.6 µs | 2.51 | ok |
| size 1e3 | `argsort(a) (1e3)` | 7.6 µs | 8.5 µs | **0.89** | 8.6 µs | 0.89 | ok |
| size 1e3 | `astype int (1e3)` | 525 ns | 204 ns | 2.58 | 206 ns | 2.55 | ok |
| size 1e3 | `copy (1e3)` | 211 ns | 209 ns | 1.01 | 210 ns | 1.01 | ok |
| size 1e3 | `i + j (1e3)` | 285 ns | 294 ns | **0.97** | 294 ns | 0.97 | ok |
| size 1e3 | `sum(i) (1e3)` | 620 ns | 143 ns | 4.34 | 146 ns | 4.26 | ok |
| size 1e3 | `sort(i) (1e3)` | 12.2 µs | 7.0 µs | 1.74 | 7.1 µs | 1.72 | ok |
| size 1e3 | `median(a) (1e3)` | 5.6 µs | 4.6 µs | 1.22 | 4.6 µs | 1.23 | ok |
| size 1e3 | `unique_values(i) (1e3)` | 24.9 µs | 7.0 µs | 3.53 | 7.1 µs | 3.48 | ok |
| size 1e5 | `a + b (1e5)` | 20.2 µs | 20.0 µs | 1.01 | 5.6 µs | 3.63 | ok |
| size 1e5 | `a * 2.0 (1e5)` | 9.8 µs | 9.7 µs | 1.01 | 5.2 µs | 1.90 | ok |
| size 1e5 | `exp(a) (1e5)` | 148.8 µs | 106.4 µs | 1.40 | 16.3 µs | 9.12 | ok |
| size 1e5 | `sin(a) (1e5)` | 364.7 µs | 80.2 µs | 4.55 | 13.0 µs | 28.15 | ok |
| size 1e5 | `sum(a) (1e5)` | 12.2 µs | 11.4 µs | 1.06 | 4.7 µs | 2.59 | ok |
| size 1e5 | `max(a) (1e5)` | 8.1 µs | 7.2 µs | 1.12 | 3.8 µs | 2.10 | ok |
| size 1e5 | `var(a) (1e5)` | 48.2 µs | 32.0 µs | 1.50 | 10.2 µs | 4.72 | ok |
| size 1e5 | `cumulative_sum(a) (1e5)` | 210.4 µs | 38.1 µs | 5.52 | 11.7 µs | 17.93 | ok |
| size 1e5 | `a < b (1e5)` | 20.1 µs | 13.4 µs | 1.50 | 5.0 µs | 4.05 | ok |
| size 1e5 | `a[a > 1.0] (1e5)` | 260.5 µs | 41.3 µs | 6.30 | 15.8 µs | 16.46 | ok |
| size 1e5 | `where(a > 1.0, a, b) (1e5)` | 44.0 µs | 29.3 µs | 1.50 | 10.8 µs | 4.06 | ok |
| size 1e5 | `sort(a) (1e5)` | 2.25 ms | 1.15 ms | 1.95 | 1.15 ms | 1.95 | ok |
| size 1e5 | `argsort(a) (1e5)` | 5.12 ms | 843.0 µs | 6.08 | 842.7 µs | 6.08 | ok |
| size 1e5 | `astype int (1e5)` | 27.4 µs | 12.9 µs | 2.12 | 4.9 µs | 5.58 | ok |
| size 1e5 | `copy (1e5)` | 9.5 µs | 9.6 µs | **1.00** | 5.4 µs | 1.78 | ok |
| size 1e5 | `i + j (1e5)` | 14.2 µs | 14.3 µs | **1.00** | 5.5 µs | 2.59 | ok |
| size 1e5 | `sum(i) (1e5)` | 7.9 µs | 7.3 µs | 1.09 | 3.9 µs | 2.03 | ok |
| size 1e5 | `sort(i) (1e5)` | 1.06 ms | 236.2 µs | 4.48 | 236.4 µs | 4.47 | ok |
| size 1e5 | `median(a) (1e5)` | 574.7 µs | 553.5 µs | 1.04 | 566.3 µs | 1.01 | ok |
| size 1e5 | `unique_values(i) (1e5)` | 861.3 µs | 229.7 µs | 3.75 | 229.9 µs | 3.75 | ok |
| size 1e7 | `a + b (1e7)` | 2.01 ms | 2.02 ms | **1.00** | 1.14 ms | 1.76 | ok |
| size 1e7 | `a * 2.0 (1e7)` | 1.30 ms | 1.31 ms | **0.99** | 716.5 µs | 1.81 | ok |
| size 1e7 | `exp(a) (1e7)` | 14.86 ms | 10.65 ms | 1.40 | 1.23 ms | 12.10 | ok |
| size 1e7 | `sin(a) (1e7)` | 37.99 ms | 8.01 ms | 4.74 | 1.07 ms | 35.57 | ok |
| size 1e7 | `sum(a) (1e7)` | 1.19 ms | 1.20 ms | **0.99** | 468.6 µs | 2.54 | ok |
| size 1e7 | `max(a) (1e7)` | 888.8 µs | 869.3 µs | 1.02 | 370.3 µs | 2.40 | ok |
| size 1e7 | `var(a) (1e7)` | 4.99 ms | 3.35 ms | 1.49 | 946.5 µs | 5.27 | ok |
| size 1e7 | `cumulative_sum(a) (1e7)` | 21.96 ms | 4.21 ms | 5.22 | 1.61 ms | 13.66 | ok |
| size 1e7 | `a < b (1e7)` | 2.01 ms | 1.36 ms | 1.47 | 796.9 µs | 2.52 | ok |
| size 1e7 | `a[a > 1.0] (1e7)` | 36.16 ms | 4.41 ms | 8.21 | 1.16 ms | 31.16 | ok |
| size 1e7 | `where(a > 1.0, a, b) (1e7)` | 4.58 ms | 3.28 ms | 1.40 | 1.90 ms | 2.40 | ok |
| size 1e7 | `sort(a) (1e7)` | 316.18 ms | 176.94 ms | 1.79 | 50.35 ms | 6.28 | ok |
| size 1e7 | `argsort(a) (1e7)` | 903.39 ms | 85.57 ms | 10.56 | 35.27 ms | 25.61 | ok |
| size 1e7 | `astype int (1e7)` | 2.70 ms | 1.32 ms | 2.05 | 717.0 µs | 3.77 | ok |
| size 1e7 | `copy (1e7)` | 1.19 ms | 1.19 ms | 1.00 | 610.0 µs | 1.94 | ok |
| size 1e7 | `i + j (1e7)` | 1.95 ms | 1.96 ms | **1.00** | 1.15 ms | 1.70 | ok |
| size 1e7 | `sum(i) (1e7)` | 874.6 µs | 869.4 µs | 1.01 | 369.7 µs | 2.37 | ok |
| size 1e7 | `sort(i) (1e7)` | 96.79 ms | 24.41 ms | 3.96 | 9.45 ms | 10.24 | ok |
| size 1e7 | `median(a) (1e7)` | 78.19 ms | 75.87 ms | 1.03 | 40.69 ms | 1.92 | ok |
| size 1e7 | `unique_values(i) (1e7)` | 83.42 ms | 23.52 ms | 3.55 | 8.60 ms | 9.70 | ok |
| tiny | `zeros(3)` | 60 ns | 86 ns | **0.70** | 86 ns | 0.70 | ok |
| tiny | `empty(3)` | 60 ns | 85 ns | **0.71** | 83 ns | 0.72 | ok |
| tiny | `asarray(s3) (no-op)` | 12 ns | 20 ns | **0.63** | 20 ns | 0.63 | ok |
| tiny | `s3 + t3s` | 152 ns | 181 ns | **0.84** | 182 ns | 0.83 | ok |
| tiny | `s3 * 2.0` | 235 ns | 207 ns | 1.14 | 210 ns | 1.12 | ok |
| tiny | `i3 + 1` | 273 ns | 214 ns | 1.28 | 217 ns | 1.26 | ok |
| tiny | `sqrt(s3)` | 121 ns | 146 ns | **0.83** | 147 ns | 0.83 | ok |
| tiny | `exp(s3)` | 124 ns | 148 ns | **0.84** | 152 ns | 0.82 | ok |
| tiny | `s3 < t3s` | 148 ns | 177 ns | **0.84** | 180 ns | 0.82 | ok |
| tiny | `sum(s3)` | 541 ns | 90 ns | 5.98 | 93 ns | 5.85 | ok |
| tiny | `max(s3)` | 518 ns | 90 ns | 5.77 | 92 ns | 5.66 | ok |
| tiny | `mean(s3)` | 875 ns | 89 ns | 9.83 | 92 ns | 9.53 | ok |
| tiny | `sum(sm, axis=0)` | 762 ns | 186 ns | 4.09 | 190 ns | 4.02 | ok |
| tiny | `sort(s3)` | 293 ns | 135 ns | 2.18 | 135 ns | 2.18 | ok |
| tiny | `reshape(sm, (4, 3))` | 265 ns | 144 ns | 1.85 | 142 ns | 1.87 | ok |
| tiny | `sm.T` | 39 ns | 65 ns | **0.61** | 64 ns | 0.61 | ok |
| tiny | `sm[1]` | 42 ns | 71 ns | **0.60** | 71 ns | 0.60 | ok |
| tiny | `sm[1:, ::2]` | 51 ns | 101 ns | **0.51** | 101 ns | 0.51 | ok |
| tiny | `concat([s3, t3s])` | 241 ns | 254 ns | **0.95** | 260 ns | 0.93 | ok |
| tiny | `where(s3 > 1, s3, t3s)` | 506 ns | 406 ns | 1.25 | 409 ns | 1.24 | ok |
| tiny | `matmul(sm.T, sm)` | 420 ns | 309 ns | 1.36 | 304 ns | 1.38 | ok |
| tiny | `result_type(f, i)` | 55 ns | 46 ns | 1.19 | 46 ns | 1.19 | ok |
| tiny | `can_cast(int64, float64)` | 88 ns | 26 ns | 3.37 | 26 ns | 3.36 | ok |
| tiny | `finfo(float64)` | 111 ns | 377 ns | **0.29** | 361 ns | 0.31 | ok |
| tiny | `s3.shape` | 18 ns | 19 ns | **0.95** | 19 ns | 0.96 | ok |
| tiny | `len(s3)` | 7 ns | 7 ns | **0.93** | 7 ns | 0.93 | ok |
| tiny | `float(s3[0])` | 36 ns | 28 ns | 1.29 | 29 ns | 1.28 | ok |
