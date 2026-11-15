# 배열 영역 전체 측정 (numpy 대비)

환경: arm64 macOS, Python 3.15.0, numpy 2.5.4, stardust 86839a1 + 작업 중 변경, 원소 1,000,000개 기준, 10라운드, 2026-11-15 10:12, 걸린 시간 46분

배율 = numpy 시간 ÷ stardust 시간 (1보다 크면 stardust가 빠르다). numpy는 1스레드(BLAS만 여러 스레드).
시간은 7번 반복 중 최솟값을 다시 라운드 사이 최솟값으로 잡았다. **굵은 배율**은 1스레드에서 numpy보다 느린 줄이다.

- 케이스 380개. 1스레드에서 numpy보다 느린 것 73개, 10스레드에서도 느린 것 34개.
- 결과가 numpy와 다른 것: 0개

## 1스레드에서 numpy보다 느린 것 (배율 낮은 순)

| 묶음 | 케이스 | numpy | sd 1T | 배율 1T | sd 10T | 배율 10T |
|---|---|---:|---:|---:|---:|---:|
| convert | `asarray numpy f64 .T` | 12 ns | 147 ns | 0.08 | 148 ns | 0.08 |
| convert | `asarray numpy f64 (view)` | 12 ns | 139 ns | 0.09 | 136 ns | 0.09 |
| convert | `asarray numpy f64 strided` | 12 ns | 134 ns | 0.09 | 135 ns | 0.09 |
| convert | `numpy.asarray(sd array)` | 12 ns | 110 ns | 0.11 | 108 ns | 0.11 |
| index | `f2[:, 5] (column)` | 48 ns | 103 ns | 0.46 | 103 ns | 0.46 |
| index | `f[1000:2000] (view)` | 43 ns | 90 ns | 0.48 | 89 ns | 0.48 |
| index | `f[::2] (view)` | 43 ns | 89 ns | 0.48 | 88 ns | 0.49 |
| tiny | `sm[1:, ::2]` | 51 ns | 102 ns | 0.51 | 101 ns | 0.51 |
| convert | `from_dlpack numpy` | 116 ns | 225 ns | 0.51 | 226 ns | 0.51 |
| tiny | `sm.T` | 38 ns | 68 ns | 0.56 | 67 ns | 0.57 |
| index | `f2[5] (row view)` | 42 ns | 72 ns | 0.59 | 72 ns | 0.59 |
| tiny | `sm[1]` | 43 ns | 72 ns | 0.59 | 72 ns | 0.59 |
| tiny | `asarray(s3) (no-op)` | 12 ns | 20 ns | 0.62 | 20 ns | 0.62 |
| tiny | `zeros(3)` | 61 ns | 91 ns | 0.66 | 90 ns | 0.67 |
| tiny | `empty(3)` | 59 ns | 86 ns | 0.69 | 90 ns | 0.66 |
| manip | `unstack(f2)` | 46.7 µs | 62.4 µs | 0.75 | 62.2 µs | 0.75 |
| unary i64 | `sign` | 96.5 µs | 125.0 µs | 0.77 | 30.8 µs | 3.13 |
| tiny | `exp(s3)` | 124 ns | 154 ns | 0.81 | 155 ns | 0.80 |
| tiny | `sqrt(s3)` | 121 ns | 149 ns | 0.81 | 150 ns | 0.81 |
| binary c128 | `add` | 320.4 µs | 388.3 µs | 0.83 | 124.1 µs | 2.58 |
| tiny | `s3 < t3s` | 148 ns | 179 ns | 0.83 | 184 ns | 0.81 |
| tiny | `s3 + t3s` | 151 ns | 179 ns | 0.84 | 183 ns | 0.82 |
| creation | `empty(N)` | 110 ns | 130 ns | 0.85 | 128 ns | 0.86 |
| size 1e3 | `argsort(a) (1e3)` | 7.6 µs | 8.9 µs | 0.86 | 8.9 µs | 0.86 |
| matmul | `vecdot(f2, g2)` | 131.2 µs | 152.9 µs | 0.86 | 22.3 µs | 5.90 |
| unary f64 | `acosh` | 2.49 ms | 2.90 ms | 0.86 | 317.7 µs | 7.84 |
| assign | `gz[idx] = 1.0 (scatter 1e5)` | 95.7 µs | 105.6 µs | 0.91 | 80.5 µs | 1.19 |
| search | `take(f2, rows, axis=0)` | 111.4 µs | 122.0 µs | 0.91 | 32.4 µs | 3.44 |
| tiny | `concat([s3, t3s])` | 238 ns | 260 ns | 0.92 | 261 ns | 0.91 |
| binary mid | `fm + gm (1e4)` | 1.7 µs | 1.8 µs | 0.93 | 1.7 µs | 0.96 |
| assign | `gz[idxn] = 1.0 (scatter 1e6)` | 960.4 µs | 1.03 ms | 0.94 | 795.0 µs | 1.21 |
| tiny | `len(s3)` | 7 ns | 7 ns | 0.94 | 7 ns | 0.94 |
| unary c128 | `square` | 266.0 µs | 282.6 µs | 0.94 | 67.5 µs | 3.94 |
| binary i64 | `bitwise_left_shift` | 146.9 µs | 155.0 µs | 0.95 | 39.3 µs | 3.74 |
| tiny | `s3.shape` | 18 ns | 19 ns | 0.95 | 19 ns | 0.97 |
| manip | `tile(v, (1000,))` | 72.1 µs | 76.0 µs | 0.95 | 30.5 µs | 2.36 |
| manip | `concat([f2, g2], axis=1)` | 333.9 µs | 350.1 µs | 0.95 | 78.4 µs | 4.26 |
| manip | `roll(f2, 1, axis=1)` | 155.7 µs | 162.7 µs | 0.96 | 33.7 µs | 4.62 |
| assign | `x[1:] += x[:-1] (overlap)` | 236.4 µs | 246.8 µs | 0.96 | 103.0 µs | 2.30 |
| stats | `nanquantile(fn, q5)` | 11.51 ms | 12.02 ms | 0.96 | 3.85 ms | 2.99 |
| convert | `asarray list 3` | 137 ns | 143 ns | 0.96 | 144 ns | 0.95 |
| reduce axis | `sum(fst) (strided)` | 149.6 µs | 155.6 µs | 0.96 | 27.6 µs | 5.42 |
| size 1e3 | `i + j (1e3)` | 284 ns | 294 ns | 0.96 | 298 ns | 0.95 |
| creation | `zeros((1000, 1000))` | 28.3 µs | 29.3 µs | 0.96 | 29.4 µs | 0.96 |
| creation | `zeros(N)` | 28.1 µs | 29.1 µs | 0.96 | 28.1 µs | 1.00 |
| creation | `full_like(f, 2.5)` | 57.5 µs | 59.6 µs | 0.96 | 56.7 µs | 1.01 |
| manip | `astype(f, M.complex128)` | 158.1 µs | 162.6 µs | 0.97 | 60.1 µs | 2.63 |
| binary i64 | `add` | 144.4 µs | 148.4 µs | 0.97 | 39.3 µs | 3.67 |
| unary i64 | `abs` | 94.1 µs | 96.3 µs | 0.98 | 30.1 µs | 3.12 |
| convert | `asarray numpy >f8` | 91.1 µs | 92.7 µs | 0.98 | 31.9 µs | 2.85 |
| reduce axis | `mean(f2, axis=0)` | 129.3 µs | 131.5 µs | 0.98 | 33.3 µs | 3.88 |
| reduce axis | `max(f2, axis=0)` | 127.9 µs | 130.1 µs | 0.98 | 31.9 µs | 4.01 |
| reduce axis | `sum(f2, axis=0)` | 128.4 µs | 130.3 µs | 0.99 | 32.0 µs | 4.02 |
| matmul | `matmul batch 1000 x 8x8` | 70.5 µs | 71.5 µs | 0.99 | 13.8 µs | 5.11 |
| binary c128 | `multiply` | 322.0 µs | 326.6 µs | 0.99 | 124.2 µs | 2.59 |
| convert | `asarray numpy f32` | 88.9 µs | 90.0 µs | 0.99 | 31.0 µs | 2.87 |
| binary f64 | `maximum` | 146.7 µs | 148.4 µs | 0.99 | 39.2 µs | 3.74 |
| size 1e3 | `a + b (1e3)` | 285 ns | 287 ns | 0.99 | 289 ns | 0.99 |
| binary i64 | `bitwise_xor` | 144.6 µs | 145.8 µs | 0.99 | 39.5 µs | 3.67 |
| size 1e7 | `sum(a) (1e7)` | 1.19 ms | 1.20 ms | 0.99 | 477.8 µs | 2.49 |
| assign | `fw[::2] = 1.0 (strided fill)` | 122.5 µs | 123.3 µs | 0.99 | 22.1 µs | 5.53 |
| size 1e3 | `copy (1e3)` | 209 ns | 210 ns | 0.99 | 212 ns | 0.98 |
| size 1e7 | `i + j (1e7)` | 1.94 ms | 1.95 ms | 0.99 | 1.15 ms | 1.68 |
| size 1e5 | `i + j (1e5)` | 14.2 µs | 14.3 µs | 0.99 | 5.9 µs | 2.42 |
| binary f64 | `subtract` | 145.8 µs | 146.5 µs | 1.00 | 39.1 µs | 3.73 |
| unary f64 | `isnan` | 71.5 µs | 71.8 µs | 1.00 | 21.2 µs | 3.38 |
| search | `searchsorted(srt, f)` | 53.51 ms | 53.72 ms | 1.00 | 6.36 ms | 8.41 |
| matmul | `matmul(f2, v) (gemv)` | 13.6 µs | 13.6 µs | 1.00 | 13.7 µs | 0.99 |
| creation | `arange(N)` | 85.7 µs | 85.8 µs | 1.00 | 85.5 µs | 1.00 |
| unary f64 | `floor` | 93.9 µs | 94.0 µs | 1.00 | 30.2 µs | 3.11 |
| unary f64 | `expm1` | 1.86 ms | 1.86 ms | 1.00 | 206.1 µs | 9.01 |
| assign | `fw[:] = g (copy)` | 93.5 µs | 93.6 µs | 1.00 | 34.2 µs | 2.74 |
| unary f64 | `log1p` | 2.47 ms | 2.48 ms | 1.00 | 272.1 µs | 9.09 |

## 전체

| 묶음 | 케이스 | numpy | sd 1T | 배율 1T | sd 10T | 배율 10T | 결과 확인 |
|---|---|---:|---:|---:|---:|---:|---|
| creation | `zeros(N)` | 28.1 µs | 29.1 µs | **0.96** | 28.1 µs | 1.00 | ok |
| creation | `ones(N)` | 57.2 µs | 56.7 µs | 1.01 | 56.9 µs | 1.00 | ok |
| creation | `empty(N)` | 110 ns | 130 ns | **0.85** | 128 ns | 0.86 | ok |
| creation | `full(N, 2.5)` | 57.1 µs | 56.8 µs | 1.01 | 56.9 µs | 1.00 | ok |
| creation | `zeros_like(f)` | 57.3 µs | 28.2 µs | 2.03 | 28.2 µs | 2.03 | ok |
| creation | `ones_like(f)` | 57.4 µs | 56.7 µs | 1.01 | 56.8 µs | 1.01 | ok |
| creation | `empty_like(f)` | 144 ns | 130 ns | 1.11 | 127 ns | 1.13 | ok |
| creation | `full_like(f, 2.5)` | 57.5 µs | 59.6 µs | **0.96** | 56.7 µs | 1.01 | ok |
| creation | `arange(N)` | 85.7 µs | 85.8 µs | **1.00** | 85.5 µs | 1.00 | ok |
| creation | `linspace(0.0, 1.0, N)` | 446.9 µs | 122.6 µs | 3.64 | 122.5 µs | 3.65 | ok |
| creation | `eye(1000)` | 36.3 µs | 32.1 µs | 1.13 | 32.3 µs | 1.13 | ok |
| creation | `meshgrid(v, v)` | 142.7 µs | 127.9 µs | 1.12 | 132.7 µs | 1.08 | ok |
| creation | `tril(f2)` | 349.1 µs | 175.8 µs | 1.99 | 146.0 µs | 2.39 | ok |
| creation | `triu(f2)` | 349.5 µs | 167.1 µs | 2.09 | 112.0 µs | 3.12 | ok |
| creation | `zeros((1000, 1000))` | 28.3 µs | 29.3 µs | **0.96** | 29.4 µs | 0.96 | ok |
| creation | `arange(N, float64)` | 227.9 µs | 93.0 µs | 2.45 | 92.8 µs | 2.46 | ok |
| creation | `asarray(copy=True)` | 90.7 µs | 90.5 µs | 1.00 | 34.4 µs | 2.64 | ok |
| convert | `asarray numpy f64 (view)` | 12 ns | 139 ns | **0.09** | 136 ns | 0.09 | ok |
| convert | `asarray numpy f64 strided` | 12 ns | 134 ns | **0.09** | 135 ns | 0.09 | ok |
| convert | `asarray numpy f64 .T` | 12 ns | 147 ns | **0.08** | 148 ns | 0.08 | ok |
| convert | `asarray numpy f32` | 88.9 µs | 90.0 µs | **0.99** | 31.0 µs | 2.87 | ok |
| convert | `asarray numpy f32 .T (np keeps F)` | 133.3 µs | 127.5 µs | 1.05 | 27.7 µs | 4.82 | ok |
| convert | `asarray numpy f32 .T (np C order)` | 443.6 µs | 130.9 µs | 3.39 | 27.4 µs | 16.20 | ok |
| convert | `asarray numpy f32 rows m[:,1:1001]` | 132.9 µs | 130.5 µs | 1.02 | 28.0 µs | 4.75 | ok |
| convert | `asarray numpy >f8` | 91.1 µs | 92.7 µs | **0.98** | 31.9 µs | 2.85 | ok |
| convert | `asarray numpy >f8 2-D [::2]` | 108.1 µs | 105.7 µs | 1.02 | 31.1 µs | 3.48 | ok |
| convert | `asarray numpy i32` | 79.0 µs | 77.0 µs | 1.03 | 31.3 µs | 2.53 | ok |
| convert | `asarray list 3` | 137 ns | 143 ns | **0.96** | 144 ns | 0.95 | ok |
| convert | `asarray list 1e4` | 137.5 µs | 62.3 µs | 2.21 | 62.3 µs | 2.21 | ok |
| convert | `from_dlpack numpy` | 116 ns | 225 ns | **0.51** | 226 ns | 0.51 | ok |
| convert | `numpy.asarray(sd array)` | 12 ns | 110 ns | **0.11** | 108 ns | 0.11 | ok |
| manip | `reshape(f, (1000, 1000))` | 275 ns | 146 ns | 1.89 | 145 ns | 1.89 | ok |
| manip | `reshape(ft, (N,))` | 503.5 µs | 373.3 µs | 1.35 | 56.7 µs | 8.88 | ok |
| manip | `permute_dims(t3, (2, 0, 1))` | 180 ns | 165 ns | 1.09 | 164 ns | 1.10 | ok |
| manip | `matrix_transpose(f2)` | 250 ns | 87 ns | 2.87 | 86 ns | 2.91 | ok |
| manip | `moveaxis(t3, 0, -1)` | 1.1 µs | 188 ns | 5.81 | 188 ns | 5.82 | ok |
| manip | `squeeze(M.reshape(f, (1, N)), axis=0)` | 427 ns | 251 ns | 1.70 | 248 ns | 1.72 | ok |
| manip | `expand_dims(f, axis=0)` | 615 ns | 99 ns | 6.21 | 99 ns | 6.23 | ok |
| manip | `broadcast_to(v, (1000, 1000))` | 956 ns | 114 ns | 8.40 | 120 ns | 8.00 | ok |
| manip | `broadcast_arrays(f2, v)` | 1.7 µs | 235 ns | 7.03 | 226 ns | 7.32 | ok |
| manip | `concat([f, g])` | 187.4 µs | 187.1 µs | 1.00 | 70.6 µs | 2.65 | ok |
| manip | `concat([f2, g2], axis=1)` | 333.9 µs | 350.1 µs | **0.95** | 78.4 µs | 4.26 | ok |
| manip | `stack([f, g])` | 191.9 µs | 189.4 µs | 1.01 | 70.5 µs | 2.72 | ok |
| manip | `flip(f)` | 184 ns | 83 ns | 2.22 | 83 ns | 2.21 | ok |
| manip | `roll(f, 1)` | 96.8 µs | 94.7 µs | 1.02 | 35.4 µs | 2.73 | ok |
| manip | `roll(f2, 1, axis=1)` | 155.7 µs | 162.7 µs | **0.96** | 33.7 µs | 4.62 | ok |
| manip | `repeat(f, 2)` | 494.7 µs | 187.6 µs | 2.64 | 61.6 µs | 8.03 | ok |
| manip | `tile(v, (1000,))` | 72.1 µs | 76.0 µs | **0.95** | 30.5 µs | 2.36 | ok |
| manip | `unstack(f2)` | 46.7 µs | 62.4 µs | **0.75** | 62.2 µs | 0.75 | ok |
| manip | `astype(f, M.int64)` | 264.0 µs | 126.4 µs | 2.09 | 30.1 µs | 8.76 | ok |
| manip | `astype(i, M.float64)` | 250.1 µs | 101.1 µs | 2.47 | 30.0 µs | 8.34 | ok |
| manip | `astype(f, M.complex128)` | 158.1 µs | 162.6 µs | **0.97** | 60.1 µs | 2.63 | ok |
| manip | `astype(b, M.float64)` | 245.9 µs | 150.6 µs | 1.63 | 28.9 µs | 8.51 | ok |
| manip | `astype(ft, M.float64)` | 93.8 µs | 91.9 µs | 1.02 | 34.4 µs | 2.72 | ok |
| manip | `ravel(ft) (copy)` | 457.2 µs | 331.5 µs | 1.38 | 53.7 µs | 8.51 | ok |
| manip | `diff(f)` | 106.6 µs | 105.0 µs | 1.02 | 33.2 µs | 3.21 | ok |
| binary f64 | `add` | 177.1 µs | 150.8 µs | 1.17 | 39.0 µs | 4.54 | ok |
| binary f64 | `subtract` | 145.8 µs | 146.5 µs | **1.00** | 39.1 µs | 3.73 | ok |
| binary f64 | `multiply` | 147.3 µs | 146.5 µs | 1.01 | 39.0 µs | 3.77 | ok |
| binary f64 | `divide` | 147.4 µs | 146.1 µs | 1.01 | 39.0 µs | 3.78 | ok |
| binary f64 | `floor_divide` | 6.43 ms | 5.99 ms | 1.07 | 680.0 µs | 9.45 | ok |
| binary f64 | `remainder` | 6.43 ms | 5.50 ms | 1.17 | 616.0 µs | 10.44 | ok |
| binary f64 | `pow` | 4.59 ms | 2.63 ms | 1.74 | 292.3 µs | 15.72 | ok |
| binary f64 | `atan2` | 12.52 ms | 1.87 ms | 6.70 | 212.0 µs | 59.09 | ok |
| binary f64 | `maximum` | 146.7 µs | 148.4 µs | **0.99** | 39.2 µs | 3.74 | ok |
| binary f64 | `minimum` | 147.0 µs | 146.2 µs | 1.01 | 39.4 µs | 3.73 | ok |
| binary f64 | `copysign` | 252.4 µs | 146.8 µs | 1.72 | 39.1 µs | 6.46 | ok |
| binary f64 | `hypot` | 1.48 ms | 1.22 ms | 1.21 | 139.8 µs | 10.59 | ok |
| binary f64 | `logaddexp` | 10.27 ms | 9.83 ms | 1.05 | 1.07 ms | 9.63 | ok |
| binary f64 | `nextafter` | 5.11 ms | 646.8 µs | 7.90 | 76.8 µs | 66.50 | ok |
| binary f64 | `equal` | 200.9 µs | 135.3 µs | 1.48 | 26.8 µs | 7.49 | ok |
| binary f64 | `less` | 200.5 µs | 136.1 µs | 1.47 | 26.8 µs | 7.48 | ok |
| binary f64 | `greater_equal` | 200.6 µs | 135.5 µs | 1.48 | 26.5 µs | 7.57 | ok |
| binary bool | `logical_and` | 25.6 µs | 17.9 µs | 1.43 | 6.7 µs | 3.82 | ok |
| binary bool | `logical_or` | 25.3 µs | 17.9 µs | 1.42 | 6.7 µs | 3.78 | ok |
| binary bool | `logical_xor` | 18.6 µs | 17.9 µs | 1.04 | 6.7 µs | 2.75 | ok |
| binary i64 | `add` | 144.4 µs | 148.4 µs | **0.97** | 39.3 µs | 3.67 | ok |
| binary i64 | `multiply` | 268.2 µs | 268.2 µs | 1.00 | 41.2 µs | 6.51 | ok |
| binary i64 | `floor_divide` | 743.2 µs | 604.0 µs | 1.23 | 76.0 µs | 9.78 | ok |
| binary i64 | `remainder` | 886.5 µs | 672.0 µs | 1.32 | 83.4 µs | 10.62 | ok |
| binary i64 | `maximum` | 201.4 µs | 146.5 µs | 1.37 | 39.6 µs | 5.09 | ok |
| binary i64 | `equal` | 135.7 µs | 135.5 µs | 1.00 | 26.7 µs | 5.08 | ok |
| binary i64 | `bitwise_and` | 148.0 µs | 146.0 µs | 1.01 | 39.4 µs | 3.76 | ok |
| binary i64 | `bitwise_or` | 201.3 µs | 146.1 µs | 1.38 | 39.5 µs | 5.10 | ok |
| binary i64 | `bitwise_xor` | 144.6 µs | 145.8 µs | **0.99** | 39.5 µs | 3.67 | ok |
| binary i64 | `bitwise_left_shift` | 146.9 µs | 155.0 µs | **0.95** | 39.3 µs | 3.74 | ok |
| binary i64 | `bitwise_right_shift` | 148.5 µs | 148.4 µs | 1.00 | 39.3 µs | 3.78 | ok |
| binary i64 | `pow(i, 2)` | 261.2 µs | 251.2 µs | 1.04 | 35.9 µs | 7.28 | ok |
| binary c128 | `add` | 320.4 µs | 388.3 µs | **0.83** | 124.1 µs | 2.58 | ok |
| binary c128 | `multiply` | 322.0 µs | 326.6 µs | **0.99** | 124.2 µs | 2.59 | ok |
| binary c128 | `divide` | 4.40 ms | 1.63 ms | 2.70 | 182.5 µs | 24.12 | ok |
| binary c128 | `equal` | 275.1 µs | 272.9 µs | 1.01 | 91.4 µs | 3.01 | ok |
| binary c128 | `pow` | 35.65 ms | 34.44 ms | 1.04 | 3.72 ms | 9.58 | ok |
| binary mixed | `f * i (cast)` | 426.9 µs | 227.9 µs | 1.87 | 44.8 µs | 9.52 | ok |
| binary mixed | `f * i + g` | 592.3 µs | 426.0 µs | 1.39 | 147.4 µs | 4.02 | ok |
| binary mixed | `f + 1.0 (scalar)` | 94.1 µs | 94.0 µs | 1.00 | 30.8 µs | 3.06 | ok |
| binary mixed | `i + 1 (scalar)` | 131.5 µs | 96.0 µs | 1.37 | 30.4 µs | 4.32 | ok |
| binary mixed | `c * f` | 665.3 µs | 631.7 µs | 1.05 | 111.9 µs | 5.95 | ok |
| binary layout | `f2 + v (broadcast)` | 227.8 µs | 108.4 µs | 2.10 | 31.3 µs | 7.27 | ok |
| binary layout | `f2 + v[:, None] (broadcast col)` | 205.5 µs | 105.1 µs | 1.96 | 31.1 µs | 6.62 | ok |
| binary layout | `ft + 1.0 (transposed)` | 94.3 µs | 94.1 µs | 1.00 | 30.4 µs | 3.10 | ok |
| binary layout | `ft + f2 (mixed order)` | 452.4 µs | 437.9 µs | 1.03 | 67.9 µs | 6.66 | ok |
| binary layout | `fst + 1.0 (strided)` | 253.2 µs | 250.2 µs | 1.01 | 38.8 µs | 6.52 | ok |
| binary layout | `fst + f (strided + contiguous)` | 263.8 µs | 256.9 µs | 1.03 | 73.2 µs | 3.60 | ok |
| binary mid | `fm + gm (1e4)` | 1.7 µs | 1.8 µs | **0.93** | 1.7 µs | 0.96 | ok |
| binary mid | `fm * 2.0 (1e4)` | 1.1 µs | 1.0 µs | 1.07 | 967 ns | 1.12 | ok |
| unary f64 | `negative` | 131.8 µs | 131.1 µs | 1.01 | 30.2 µs | 4.36 | ok |
| unary f64 | `positive` | 247.3 µs | 94.0 µs | 2.63 | 30.2 µs | 8.20 | ok |
| unary f64 | `abs` | 94.3 µs | 94.1 µs | 1.00 | 30.2 µs | 3.12 | ok |
| unary f64 | `square` | 94.4 µs | 93.9 µs | 1.00 | 30.2 µs | 3.13 | ok |
| unary f64 | `sqrt` | 242.9 µs | 238.1 µs | 1.02 | 35.3 µs | 6.88 | ok |
| unary f64 | `sign` | 298.2 µs | 219.0 µs | 1.36 | 33.4 µs | 8.92 | ok |
| unary f64 | `exp` | 1.49 ms | 1.07 ms | 1.39 | 121.4 µs | 12.25 | ok |
| unary f64 | `expm1` | 1.86 ms | 1.86 ms | **1.00** | 206.1 µs | 9.01 | ok |
| unary f64 | `log` | 3.14 ms | 1.08 ms | 2.90 | 123.1 µs | 25.53 | ok |
| unary f64 | `log1p` | 2.47 ms | 2.48 ms | **1.00** | 272.1 µs | 9.09 | ok |
| unary f64 | `log2` | 3.44 ms | 1.03 ms | 3.33 | 117.8 µs | 29.20 | ok |
| unary f64 | `log10` | 3.50 ms | 1.08 ms | 3.23 | 123.1 µs | 28.41 | ok |
| unary f64 | `sin` | 3.81 ms | 801.7 µs | 4.75 | 93.6 µs | 40.71 | ok |
| unary f64 | `cos` | 3.75 ms | 3.68 ms | 1.02 | 399.7 µs | 9.39 | ok |
| unary f64 | `tan` | 4.15 ms | 4.12 ms | 1.01 | 444.5 µs | 9.35 | ok |
| unary f64 | `atan` | 7.62 ms | 1.12 ms | 6.83 | 126.8 µs | 60.06 | ok |
| unary f64 | `sinh` | 2.61 ms | 1.02 ms | 2.56 | 116.3 µs | 22.44 | ok |
| unary f64 | `cosh` | 2.48 ms | 958.1 µs | 2.58 | 109.5 µs | 22.61 | ok |
| unary f64 | `tanh` | 2.22 ms | 1.19 ms | 1.86 | 135.0 µs | 16.44 | ok |
| unary f64 | `asinh` | 4.79 ms | 3.59 ms | 1.33 | 394.0 µs | 12.15 | ok |
| unary f64 | `floor` | 93.9 µs | 94.0 µs | **1.00** | 30.2 µs | 3.11 | ok |
| unary f64 | `ceil` | 94.1 µs | 94.0 µs | 1.00 | 30.2 µs | 3.12 | ok |
| unary f64 | `trunc` | 94.3 µs | 93.9 µs | 1.00 | 30.2 µs | 3.12 | ok |
| unary f64 | `round` | 94.7 µs | 94.2 µs | 1.01 | 30.2 µs | 3.14 | ok |
| unary f64 | `isnan` | 71.5 µs | 71.8 µs | **1.00** | 21.2 µs | 3.38 | ok |
| unary f64 | `isinf` | 71.5 µs | 71.4 µs | 1.00 | 21.2 µs | 3.37 | ok |
| unary f64 | `isfinite` | 71.7 µs | 71.6 µs | 1.00 | 21.3 µs | 3.37 | ok |
| unary f64 | `signbit` | 72.5 µs | 71.4 µs | 1.01 | 21.3 µs | 3.40 | ok |
| unary f64 | `reciprocal` | 124.6 µs | 124.3 µs | 1.00 | 30.8 µs | 4.05 | ok |
| unary f64 | `asin` | 5.34 ms | 935.5 µs | 5.70 | 107.2 µs | 49.78 | ok |
| unary f64 | `acos` | 6.94 ms | 1.27 ms | 5.47 | 143.6 µs | 48.32 | ok |
| unary f64 | `atanh` | 5.34 ms | 2.18 ms | 2.45 | 240.7 µs | 22.17 | ok |
| unary f64 | `acosh` | 2.49 ms | 2.90 ms | **0.86** | 317.7 µs | 7.84 | ok |
| unary f64 | `sign (both signs)` | 3.54 ms | 218.9 µs | 16.17 | 33.7 µs | 105.13 | ok |
| unary f64 | `exp strided` | 1.49 ms | 1.26 ms | 1.18 | 142.4 µs | 10.45 | ok |
| unary f64 | `sqrt transposed` | 246.3 µs | 241.3 µs | 1.02 | 35.6 µs | 6.93 | ok |
| unary f64 | `exp (1e4)` | 15.1 µs | 10.9 µs | 1.39 | 10.9 µs | 1.39 | ok |
| unary c128 | `abs` | 1.63 ms | 1.26 ms | 1.29 | 143.0 µs | 11.39 | ok |
| unary c128 | `exp` | 8.19 ms | 7.13 ms | 1.15 | 772.1 µs | 10.60 | ok |
| unary c128 | `sqrt` | 2.23 ms | 1.57 ms | 1.42 | 174.4 µs | 12.79 | ok |
| unary c128 | `log` | 22.72 ms | 21.05 ms | 1.08 | 2.22 ms | 10.21 | ok |
| unary c128 | `sin` | 10.35 ms | 9.38 ms | 1.10 | 1.02 ms | 10.18 | ok |
| unary c128 | `conj` | 194.7 µs | 194.4 µs | 1.00 | 67.8 µs | 2.87 | ok |
| unary c128 | `real` | 84 ns | 67 ns | 1.26 | 67 ns | 1.27 | ok |
| unary c128 | `imag` | 85 ns | 67 ns | 1.27 | 67 ns | 1.27 | ok |
| unary c128 | `square` | 266.0 µs | 282.6 µs | **0.94** | 67.5 µs | 3.94 | ok |
| unary c128 | `isnan` | 321.1 µs | 203.0 µs | 1.58 | 30.2 µs | 10.64 | ok |
| unary i64 | `abs` | 94.1 µs | 96.3 µs | **0.98** | 30.1 µs | 3.12 | ok |
| unary i64 | `negative` | 94.6 µs | 94.2 µs | 1.00 | 30.1 µs | 3.14 | ok |
| unary i64 | `square` | 253.7 µs | 253.6 µs | 1.00 | 36.1 µs | 7.03 | ok |
| unary i64 | `sign` | 96.5 µs | 125.0 µs | **0.77** | 30.8 µs | 3.13 | ok |
| unary i64 | `bitwise_invert` | 94.6 µs | 94.0 µs | 1.01 | 30.3 µs | 3.13 | ok |
| unary bool | `logical_not` | 18.7 µs | 12.0 µs | 1.56 | 6.3 µs | 2.97 | ok |
| select | `clip(f, 0.6, 1.2)` | 253.7 µs | 109.8 µs | 2.31 | 30.7 µs | 8.26 | ok |
| select | `clip(i, -10, 10)` | 134.4 µs | 130.5 µs | 1.03 | 31.4 µs | 4.29 | ok |
| select | `where(b, f, g)` | 275.3 µs | 216.0 µs | 1.27 | 54.3 µs | 5.07 | ok |
| select | `where(b, f, 0.0)` | 261.3 µs | 157.1 µs | 1.66 | 31.2 µs | 8.37 | ok |
| reduce f64 | `sum` | 112.8 µs | 111.2 µs | 1.01 | 24.0 µs | 4.69 | ok |
| reduce f64 | `prod` | 788.4 µs | 108.4 µs | 7.28 | 21.5 µs | 36.61 | ok |
| reduce f64 | `min` | 75.3 µs | 71.6 µs | 1.05 | 22.6 µs | 3.33 | ok |
| reduce f64 | `max` | 75.3 µs | 71.7 µs | 1.05 | 22.6 µs | 3.34 | ok |
| reduce f64 | `mean` | 113.4 µs | 111.1 µs | 1.02 | 24.1 µs | 4.71 | ok |
| reduce f64 | `var` | 478.1 µs | 312.3 µs | 1.53 | 57.7 µs | 8.29 | ok |
| reduce f64 | `std` | 444.8 µs | 313.0 µs | 1.42 | 57.6 µs | 7.73 | ok |
| reduce f64 | `argmin` | 559.4 µs | 494.7 µs | 1.13 | 56.9 µs | 9.84 | ok |
| reduce f64 | `argmax` | 558.2 µs | 471.2 µs | 1.18 | 57.3 µs | 9.74 | ok |
| reduce i64 | `sum` | 72.6 µs | 71.8 µs | 1.01 | 22.3 µs | 3.25 | ok |
| reduce i64 | `min` | 76.5 µs | 71.6 µs | 1.07 | 22.0 µs | 3.48 | ok |
| reduce i64 | `max` | 76.4 µs | 71.6 µs | 1.07 | 22.0 µs | 3.48 | ok |
| reduce i64 | `mean` | 360.9 µs | 204.6 µs | 1.76 | 31.1 µs | 11.61 | ok |
| reduce i64 | `var` | 923.2 µs | 520.7 µs | 1.77 | 99.9 µs | 9.25 | ok |
| reduce i64 | `argmax` | 125.4 µs | 122.4 µs | 1.02 | 24.4 µs | 5.15 | ok |
| reduce c128 | `sum` | 193.5 µs | 174.9 µs | 1.11 | 44.8 µs | 4.32 | ok |
| reduce c128 | `prod` | 3.93 ms | 278.0 µs | 14.13 | 36.4 µs | 107.80 | ok |
| reduce c128 | `mean` | 192.0 µs | 174.8 µs | 1.10 | 44.8 µs | 4.28 | ok |
| reduce c128 | `var` | 1.06 ms | 489.4 µs | 2.17 | 91.2 µs | 11.66 | ok |
| reduce bool | `any` | 490 ns | 108 ns | 4.52 | 112 ns | 4.37 | ok |
| reduce bool | `all` | 487 ns | 108 ns | 4.50 | 111 ns | 4.38 | ok |
| reduce bool | `sum(b)` | 155.5 µs | 133.2 µs | 1.17 | 22.2 µs | 7.01 | ok |
| reduce axis | `sum(f2, axis=0)` | 128.4 µs | 130.3 µs | **0.99** | 32.0 µs | 4.02 | ok |
| reduce axis | `sum(f2, axis=1)` | 118.9 µs | 113.9 µs | 1.04 | 25.0 µs | 4.75 | ok |
| reduce axis | `mean(f2, axis=0)` | 129.3 µs | 131.5 µs | **0.98** | 33.3 µs | 3.88 | ok |
| reduce axis | `mean(f2, axis=1)` | 119.9 µs | 115.3 µs | 1.04 | 26.2 µs | 4.57 | ok |
| reduce axis | `max(f2, axis=0)` | 127.9 µs | 130.1 µs | **0.98** | 31.9 µs | 4.01 | ok |
| reduce axis | `max(f2, axis=1)` | 83.4 µs | 75.7 µs | 1.10 | 21.8 µs | 3.83 | ok |
| reduce axis | `var(f2, axis=0)` | 634.4 µs | 426.1 µs | 1.49 | 90.4 µs | 7.02 | ok |
| reduce axis | `var(f2, axis=1)` | 582.4 µs | 322.4 µs | 1.81 | 62.5 µs | 9.32 | ok |
| reduce axis | `argmax(f2, axis=0)` | 1.02 ms | 930.8 µs | 1.09 | 108.8 µs | 9.34 | ok |
| reduce axis | `argmax(f2, axis=1)` | 529.9 µs | 477.0 µs | 1.11 | 57.7 µs | 9.19 | ok |
| reduce axis | `sum(t3, axis=(0, 2))` | 170.4 µs | 160.8 µs | 1.06 | 41.3 µs | 4.13 | ok |
| reduce axis | `sum(ft) (transposed)` | 112.8 µs | 111.1 µs | 1.02 | 24.1 µs | 4.69 | ok |
| reduce axis | `sum(ft, axis=0)` | 118.8 µs | 114.3 µs | 1.04 | 25.0 µs | 4.75 | ok |
| reduce axis | `sum(fst) (strided)` | 149.6 µs | 155.6 µs | **0.96** | 27.6 µs | 5.42 | ok |
| reduce axis | `sum(i2, axis=0)` | 278.7 µs | 130.4 µs | 2.14 | 31.9 µs | 8.74 | ok |
| reduce mid | `sum(fm) (1e4)` | 1.7 µs | 1.2 µs | 1.42 | 1.2 µs | 1.41 | ok |
| reduce mid | `var(fm) (1e4)` | 8.8 µs | 3.3 µs | 2.66 | 3.3 µs | 2.65 | ok |
| scan | `cumulative_sum(f)` | 2.18 ms | 384.5 µs | 5.67 | 69.9 µs | 31.20 | ok |
| scan | `cumulative_prod(g)` | 2.30 ms | 421.6 µs | 5.45 | 73.5 µs | 31.24 | ok |
| scan | `cumulative_sum(i)` | 1.52 ms | 264.3 µs | 5.75 | 85.5 µs | 17.77 | ok |
| scan | `cumulative_sum(f2, axis=0)` | 2.91 ms | 618.2 µs | 4.71 | 345.0 µs | 8.44 | ok |
| scan | `cumulative_sum(f2, axis=1)` | 2.31 ms | 528.5 µs | 4.38 | 65.1 µs | 35.51 | ok |
| scan | `cumulative_sum(c)` | 3.18 ms | 647.4 µs | 4.91 | 175.2 µs | 18.13 | ok |
| nan | `nansum` | 1.28 ms | 157.4 µs | 8.15 | 25.8 µs | 49.72 | ok |
| nan | `nanprod` | 2.03 ms | 115.5 µs | 17.60 | 21.1 µs | 96.47 | ok |
| nan | `nanmin` | 75.8 µs | 71.7 µs | 1.06 | 22.6 µs | 3.36 | ok |
| nan | `nanmax` | 75.9 µs | 71.7 µs | 1.06 | 22.5 µs | 3.37 | ok |
| nan | `nanmean` | 1.46 ms | 230.8 µs | 6.31 | 39.8 µs | 36.60 | ok |
| nan | `nanvar` | 3.34 ms | 541.0 µs | 6.17 | 80.7 µs | 41.35 | ok |
| nan | `nanstd` | 3.35 ms | 540.3 µs | 6.20 | 80.6 µs | 41.54 | ok |
| nan | `nanargmin` | 1.71 ms | 554.8 µs | 3.08 | 71.2 µs | 24.01 | ok |
| nan | `nanargmax` | 1.72 ms | 564.9 µs | 3.04 | 71.1 µs | 24.18 | ok |
| nan | `nancumulative_sum` | 3.31 ms | 404.2 µs | 8.18 | 71.8 µs | 46.06 | ok |
| nan | `nancumulative_prod` | 3.57 ms | 415.9 µs | 8.59 | 73.2 µs | 48.84 | ok |
| nan | `nansum(f2-like, axis=0)` | 1.28 ms | 131.6 µs | 9.75 | 32.8 µs | 39.15 | ok |
| count | `count_nonzero(b)` | 23.4 µs | 22.9 µs | 1.02 | 6.2 µs | 3.80 | ok |
| count | `count_nonzero(f)` | 248.1 µs | 93.8 µs | 2.64 | 15.8 µs | 15.66 | ok |
| count | `count_nonzero(f2, axis=0)` | 623.6 µs | 333.9 µs | 1.87 | 59.0 µs | 10.56 | ok |
| stats | `quantile(f, q5)` | 13.92 ms | 13.35 ms | 1.04 | 3.16 ms | 4.40 | ok |
| stats | `quantile(f, 0.5)` | 7.15 ms | 6.58 ms | 1.09 | 3.15 ms | 2.27 | ok |
| stats | `median(f)` | 7.32 ms | 6.65 ms | 1.10 | 3.16 ms | 2.32 | ok |
| stats | `median(f2, axis=0)` | 8.14 ms | 5.68 ms | 1.43 | 629.3 µs | 12.94 | ok |
| stats | `median(f2, axis=1)` | 7.57 ms | 5.34 ms | 1.42 | 612.3 µs | 12.37 | ok |
| stats | `nanquantile(fn, q5)` | 11.51 ms | 12.02 ms | **0.96** | 3.85 ms | 2.99 | ok |
| stats | `nanmedian(fn)` | 7.97 ms | 6.54 ms | 1.22 | 3.84 ms | 2.08 | ok |
| stats | `cov(1000 x 1000)` | 2.16 ms | 1.78 ms | 1.21 | 1.52 ms | 1.42 | ok |
| stats | `cov(20 x 1e5)` | 4.01 ms | 2.44 ms | 1.64 | 2.14 ms | 1.88 | ok |
| stats | `corrcoef(1000 x 1000)` | 2.99 ms | 2.03 ms | 1.47 | 1.56 ms | 1.91 | ok |
| stats | `bincount(iv)` | 1.00 ms | 350.7 µs | 2.86 | 83.9 µs | 11.96 | ok |
| stats | `bincount(iv, weights)` | 1.07 ms | 467.9 µs | 2.28 | 378.9 µs | 2.82 | ok |
| stats | `histogram(f, 100)` | 4.02 ms | 1.64 ms | 2.46 | 220.6 µs | 18.23 | ok |
| stats | `histogram(f, 100, range)` | 4.27 ms | 1.14 ms | 3.73 | 162.8 µs | 26.21 | ok |
| sort | `sort(f)` | 27.23 ms | 14.02 ms | 1.94 | 3.97 ms | 6.86 | ok |
| sort | `sort(f, stable=False)` | 27.14 ms | 14.01 ms | 1.94 | 3.97 ms | 6.84 | ok |
| sort | `sort(half zeros)` | 15.38 ms | 14.82 ms | 1.04 | 3.37 ms | 4.56 | ok |
| sort | `sort(i)` | 9.78 ms | 2.46 ms | 3.98 | 1.02 ms | 9.55 | ok |
| sort | `sort(i wide)` | 20.90 ms | 6.33 ms | 3.30 | 2.30 ms | 9.09 | ok |
| sort | `sort(f2, axis=1)` | 13.81 ms | 6.27 ms | 2.20 | 703.8 µs | 19.63 | ok |
| sort | `sort(f2, axis=0)` | 14.40 ms | 6.88 ms | 2.09 | 948.6 µs | 15.18 | ok |
| sort | `sort(f, descending)` | 27.45 ms | 14.21 ms | 1.93 | 3.99 ms | 6.88 | ok |
| sort | `argsort(f)` | 68.63 ms | 10.26 ms | 6.69 | 4.33 ms | 15.84 | ok |
| sort | `argsort(f, stable=False)` | 68.40 ms | 10.26 ms | 6.66 | 4.33 ms | 15.78 | ok |
| sort | `argsort(i)` | 36.62 ms | 2.74 ms | 13.36 | 1.25 ms | 29.41 | ok |
| sort | `argsort(i, stable=False)` | 36.60 ms | 2.75 ms | 13.31 | 1.24 ms | 29.42 | ok |
| sort | `argsort(f2, axis=1)` | 31.41 ms | 8.74 ms | 3.59 | 909.8 µs | 34.52 | ok |
| sort | `sort(fm) (1e4)` | 183.1 µs | 85.5 µs | 2.14 | 84.6 µs | 2.16 | ok |
| sort | `argsort(fm) (1e4)` | 182.8 µs | 88.7 µs | 2.06 | 88.9 µs | 2.06 | ok |
| unique | `unique_values(iv)` | 5.47 ms | 2.84 ms | 1.93 | 1.05 ms | 5.20 | ok |
| unique | `unique_counts(iv)` | 9.30 ms | 4.15 ms | 2.24 | 2.42 ms | 3.85 | ok |
| unique | `unique_inverse(iv)` | 38.85 ms | 4.18 ms | 9.29 | 2.47 ms | 15.71 | ok |
| unique | `unique_all(iv)` | 79.91 ms | 4.21 ms | 18.98 | 2.48 ms | 32.28 | ok |
| unique | `unique_values(f)` | 27.47 ms | 14.23 ms | 1.93 | 4.17 ms | 6.59 | ok |
| search | `searchsorted(srt, keys)` | 9.32 ms | 4.36 ms | 2.14 | 770.1 µs | 12.10 | ok |
| search | `searchsorted(srt, f)` | 53.51 ms | 53.72 ms | **1.00** | 6.36 ms | 8.41 | ok |
| search | `nonzero(b)` | 343.0 µs | 324.7 µs | 1.06 | 48.5 µs | 7.07 | ok |
| search | `nonzero(f2 > 0.5)` | 4.59 ms | 462.0 µs | 9.93 | 81.2 µs | 56.50 | ok |
| search | `take(f, idx)` | 149.6 µs | 68.9 µs | 2.17 | 11.7 µs | 12.84 | ok |
| search | `take(f2, rows, axis=0)` | 111.4 µs | 122.0 µs | **0.91** | 32.4 µs | 3.44 | ok |
| search | `take_along_axis(f2, idx2, axis=1)` | 1.94 ms | 440.4 µs | 4.41 | 75.1 µs | 25.87 | ok |
| index | `f[1000:2000] (view)` | 43 ns | 90 ns | **0.48** | 89 ns | 0.48 | ok |
| index | `f[::2] (view)` | 43 ns | 89 ns | **0.48** | 88 ns | 0.49 | ok |
| index | `f2[5] (row view)` | 42 ns | 72 ns | **0.59** | 72 ns | 0.59 | ok |
| index | `f2[:, 5] (column)` | 48 ns | 103 ns | **0.46** | 103 ns | 0.46 | ok |
| index | `f2[10, 20] (scalar)` | 33 ns | 25 ns | 1.32 | 25 ns | 1.32 | ok |
| index | `f[idx] (gather 1e5)` | 73.4 µs | 68.9 µs | 1.06 | 11.6 µs | 6.30 | ok |
| index | `f[idxn] (gather 1e6)` | 728.3 µs | 691.9 µs | 1.05 | 82.2 µs | 8.86 | ok |
| index | `f[b] (mask)` | 3.37 ms | 340.0 µs | 9.92 | 47.7 µs | 70.74 | ok |
| index | `f2[rows] (fancy rows)` | 158.0 µs | 122.3 µs | 1.29 | 32.3 µs | 4.89 | ok |
| index | `f2[f2 > 0.5] (2-D mask)` | 3.50 ms | 417.9 µs | 8.38 | 72.3 µs | 48.45 | ok |
| index | `s3[1] (tiny scalar)` | 28 ns | 23 ns | 1.20 | 23 ns | 1.21 | ok |
| assign | `gz[idx] = 1.0 (scatter 1e5)` | 95.7 µs | 105.6 µs | **0.91** | 80.5 µs | 1.19 | stmt |
| assign | `gz[idxn] = 1.0 (scatter 1e6)` | 960.4 µs | 1.03 ms | **0.94** | 795.0 µs | 1.21 | stmt |
| assign | `fw[b] = 0.0 (mask)` | 2.93 ms | 410.2 µs | 7.14 | 55.7 µs | 52.57 | stmt |
| assign | `fw[:] = 1.0 (fill)` | 62.4 µs | 62.1 µs | 1.01 | 29.6 µs | 2.11 | stmt |
| assign | `fw[:] = g (copy)` | 93.5 µs | 93.6 µs | **1.00** | 34.2 µs | 2.74 | stmt |
| assign | `fw[::2] = 1.0 (strided fill)` | 122.5 µs | 123.3 µs | **0.99** | 22.1 µs | 5.53 | stmt |
| assign | `x += g (in place)` | 165.3 µs | 164.2 µs | 1.01 | 31.2 µs | 5.29 | ok |
| assign | `x[1:] += x[:-1] (overlap)` | 236.4 µs | 246.8 µs | **0.96** | 103.0 µs | 2.30 | stmt |
| assign | `x *= 1.0000001` | 111.7 µs | 111.1 µs | 1.01 | 30.2 µs | 3.70 | ok |
| matmul | `matmul 512` | 320.3 µs | 319.4 µs | 1.00 | 320.4 µs | 1.00 | ok |
| matmul | `matmul 1000` | 2.02 ms | 2.01 ms | 1.00 | 2.01 ms | 1.00 | ok |
| matmul | `matmul(f2, v) (gemv)` | 13.6 µs | 13.6 µs | **1.00** | 13.7 µs | 0.99 | ok |
| matmul | `matmul(v, v) (dot)` | 654 ns | 324 ns | 2.02 | 321 ns | 2.04 | ok |
| matmul | `matmul batch 1000 x 8x8` | 70.5 µs | 71.5 µs | **0.99** | 13.8 µs | 5.11 | ok |
| matmul | `matmul int 200` | 2.56 ms | 1.55 ms | 1.65 | 216.3 µs | 11.83 | ok |
| matmul | `matmul(ft, f2) (transposed)` | 1.31 ms | 1.30 ms | 1.01 | 1.30 ms | 1.01 | ok |
| matmul | `vecdot(f2, g2)` | 131.2 µs | 152.9 µs | **0.86** | 22.3 µs | 5.90 | ok |
| matmul | `tensordot(t3, t3, 2)` | 377.9 µs | 374.5 µs | 1.01 | 374.7 µs | 1.01 | ok |
| size 1e3 | `a + b (1e3)` | 285 ns | 287 ns | **0.99** | 289 ns | 0.99 | ok |
| size 1e3 | `a * 2.0 (1e3)` | 359 ns | 288 ns | 1.25 | 292 ns | 1.23 | ok |
| size 1e3 | `exp(a) (1e3)` | 1.7 µs | 1.2 µs | 1.35 | 1.2 µs | 1.35 | ok |
| size 1e3 | `sin(a) (1e3)` | 2.5 µs | 953 ns | 2.64 | 955 ns | 2.63 | ok |
| size 1e3 | `sum(a) (1e3)` | 664 ns | 186 ns | 3.56 | 191 ns | 3.47 | ok |
| size 1e3 | `max(a) (1e3)` | 583 ns | 140 ns | 4.18 | 142 ns | 4.11 | ok |
| size 1e3 | `var(a) (1e3)` | 3.6 µs | 468 ns | 7.67 | 477 ns | 7.53 | ok |
| size 1e3 | `cumulative_sum(a) (1e3)` | 2.4 µs | 709 ns | 3.42 | 711 ns | 3.41 | ok |
| size 1e3 | `a < b (1e3)` | 352 ns | 267 ns | 1.32 | 270 ns | 1.30 | ok |
| size 1e3 | `a[a > 1.0] (1e3)` | 2.2 µs | 845 ns | 2.61 | 843 ns | 2.62 | ok |
| size 1e3 | `where(a > 1.0, a, b) (1e3)` | 1.0 µs | 586 ns | 1.72 | 591 ns | 1.70 | ok |
| size 1e3 | `sort(a) (1e3)` | 14.0 µs | 5.6 µs | 2.51 | 5.6 µs | 2.50 | ok |
| size 1e3 | `argsort(a) (1e3)` | 7.6 µs | 8.9 µs | **0.86** | 8.9 µs | 0.86 | ok |
| size 1e3 | `astype int (1e3)` | 523 ns | 206 ns | 2.54 | 207 ns | 2.52 | ok |
| size 1e3 | `copy (1e3)` | 209 ns | 210 ns | **0.99** | 212 ns | 0.98 | ok |
| size 1e3 | `i + j (1e3)` | 284 ns | 294 ns | **0.96** | 298 ns | 0.95 | ok |
| size 1e3 | `sum(i) (1e3)` | 629 ns | 142 ns | 4.41 | 146 ns | 4.30 | ok |
| size 1e3 | `sort(i) (1e3)` | 12.2 µs | 7.1 µs | 1.71 | 7.1 µs | 1.73 | ok |
| size 1e3 | `median(a) (1e3)` | 5.7 µs | 4.7 µs | 1.22 | 4.6 µs | 1.23 | ok |
| size 1e3 | `unique_values(i) (1e3)` | 24.8 µs | 7.2 µs | 3.44 | 7.2 µs | 3.46 | ok |
| size 1e5 | `a + b (1e5)` | 20.2 µs | 20.1 µs | 1.01 | 5.9 µs | 3.44 | ok |
| size 1e5 | `a * 2.0 (1e5)` | 9.8 µs | 9.7 µs | 1.02 | 5.7 µs | 1.72 | ok |
| size 1e5 | `exp(a) (1e5)` | 149.0 µs | 106.7 µs | 1.40 | 16.3 µs | 9.15 | ok |
| size 1e5 | `sin(a) (1e5)` | 363.8 µs | 80.3 µs | 4.53 | 12.8 µs | 28.42 | ok |
| size 1e5 | `sum(a) (1e5)` | 12.2 µs | 11.4 µs | 1.07 | 5.1 µs | 2.39 | ok |
| size 1e5 | `max(a) (1e5)` | 8.1 µs | 7.2 µs | 1.12 | 4.2 µs | 1.90 | ok |
| size 1e5 | `var(a) (1e5)` | 48.3 µs | 32.0 µs | 1.51 | 11.0 µs | 4.40 | ok |
| size 1e5 | `cumulative_sum(a) (1e5)` | 209.6 µs | 38.4 µs | 5.46 | 12.1 µs | 17.37 | ok |
| size 1e5 | `a < b (1e5)` | 20.0 µs | 13.4 µs | 1.49 | 5.2 µs | 3.85 | ok |
| size 1e5 | `a[a > 1.0] (1e5)` | 252.4 µs | 41.4 µs | 6.10 | 16.2 µs | 15.61 | ok |
| size 1e5 | `where(a > 1.0, a, b) (1e5)` | 44.0 µs | 29.3 µs | 1.50 | 11.4 µs | 3.87 | ok |
| size 1e5 | `sort(a) (1e5)` | 2.25 ms | 1.13 ms | 1.98 | 1.13 ms | 1.99 | ok |
| size 1e5 | `argsort(a) (1e5)` | 5.12 ms | 843.1 µs | 6.08 | 843.3 µs | 6.07 | ok |
| size 1e5 | `astype int (1e5)` | 27.5 µs | 13.1 µs | 2.09 | 5.4 µs | 5.10 | ok |
| size 1e5 | `copy (1e5)` | 9.6 µs | 9.5 µs | 1.00 | 5.7 µs | 1.67 | ok |
| size 1e5 | `i + j (1e5)` | 14.2 µs | 14.3 µs | **0.99** | 5.9 µs | 2.42 | ok |
| size 1e5 | `sum(i) (1e5)` | 7.9 µs | 7.3 µs | 1.10 | 4.3 µs | 1.86 | ok |
| size 1e5 | `sort(i) (1e5)` | 1.06 ms | 237.6 µs | 4.47 | 237.3 µs | 4.47 | ok |
| size 1e5 | `median(a) (1e5)` | 569.7 µs | 559.2 µs | 1.02 | 578.6 µs | 0.98 | ok |
| size 1e5 | `unique_values(i) (1e5)` | 871.1 µs | 230.8 µs | 3.78 | 230.5 µs | 3.78 | ok |
| size 1e7 | `a + b (1e7)` | 2.01 ms | 2.01 ms | 1.00 | 1.14 ms | 1.76 | ok |
| size 1e7 | `a * 2.0 (1e7)` | 1.36 ms | 1.31 ms | 1.04 | 717.8 µs | 1.89 | ok |
| size 1e7 | `exp(a) (1e7)` | 14.86 ms | 10.69 ms | 1.39 | 1.23 ms | 12.07 | ok |
| size 1e7 | `sin(a) (1e7)` | 38.09 ms | 8.01 ms | 4.76 | 1.07 ms | 35.53 | ok |
| size 1e7 | `sum(a) (1e7)` | 1.19 ms | 1.20 ms | **0.99** | 477.8 µs | 2.49 | ok |
| size 1e7 | `max(a) (1e7)` | 890.0 µs | 873.3 µs | 1.02 | 373.4 µs | 2.38 | ok |
| size 1e7 | `var(a) (1e7)` | 5.02 ms | 3.34 ms | 1.50 | 962.8 µs | 5.21 | ok |
| size 1e7 | `cumulative_sum(a) (1e7)` | 21.45 ms | 4.20 ms | 5.11 | 1.61 ms | 13.32 | ok |
| size 1e7 | `a < b (1e7)` | 2.01 ms | 1.36 ms | 1.47 | 793.9 µs | 2.53 | ok |
| size 1e7 | `a[a > 1.0] (1e7)` | 36.17 ms | 4.40 ms | 8.21 | 1.17 ms | 30.98 | ok |
| size 1e7 | `where(a > 1.0, a, b) (1e7)` | 4.58 ms | 3.28 ms | 1.39 | 1.91 ms | 2.40 | ok |
| size 1e7 | `sort(a) (1e7)` | 316.13 ms | 170.96 ms | 1.85 | 49.79 ms | 6.35 | ok |
| size 1e7 | `argsort(a) (1e7)` | 907.71 ms | 85.49 ms | 10.62 | 35.32 ms | 25.70 | ok |
| size 1e7 | `astype int (1e7)` | 2.70 ms | 1.30 ms | 2.08 | 717.5 µs | 3.77 | ok |
| size 1e7 | `copy (1e7)` | 1.18 ms | 1.18 ms | 1.00 | 610.9 µs | 1.94 | ok |
| size 1e7 | `i + j (1e7)` | 1.94 ms | 1.95 ms | **0.99** | 1.15 ms | 1.68 | ok |
| size 1e7 | `sum(i) (1e7)` | 869.0 µs | 857.5 µs | 1.01 | 372.6 µs | 2.33 | ok |
| size 1e7 | `sort(i) (1e7)` | 97.52 ms | 24.51 ms | 3.98 | 9.40 ms | 10.37 | ok |
| size 1e7 | `median(a) (1e7)` | 78.92 ms | 75.55 ms | 1.04 | 40.32 ms | 1.96 | ok |
| size 1e7 | `unique_values(i) (1e7)` | 83.47 ms | 23.73 ms | 3.52 | 8.51 ms | 9.81 | ok |
| tiny | `zeros(3)` | 61 ns | 91 ns | **0.66** | 90 ns | 0.67 | ok |
| tiny | `empty(3)` | 59 ns | 86 ns | **0.69** | 90 ns | 0.66 | ok |
| tiny | `asarray(s3) (no-op)` | 12 ns | 20 ns | **0.62** | 20 ns | 0.62 | ok |
| tiny | `s3 + t3s` | 151 ns | 179 ns | **0.84** | 183 ns | 0.82 | ok |
| tiny | `s3 * 2.0` | 247 ns | 209 ns | 1.18 | 212 ns | 1.16 | ok |
| tiny | `i3 + 1` | 285 ns | 216 ns | 1.32 | 216 ns | 1.32 | ok |
| tiny | `sqrt(s3)` | 121 ns | 149 ns | **0.81** | 150 ns | 0.81 | ok |
| tiny | `exp(s3)` | 124 ns | 154 ns | **0.81** | 155 ns | 0.80 | ok |
| tiny | `s3 < t3s` | 148 ns | 179 ns | **0.83** | 184 ns | 0.81 | ok |
| tiny | `sum(s3)` | 548 ns | 90 ns | 6.10 | 92 ns | 5.96 | ok |
| tiny | `max(s3)` | 527 ns | 91 ns | 5.79 | 92 ns | 5.73 | ok |
| tiny | `mean(s3)` | 888 ns | 89 ns | 9.99 | 91 ns | 9.73 | ok |
| tiny | `sum(sm, axis=0)` | 751 ns | 190 ns | 3.95 | 194 ns | 3.87 | ok |
| tiny | `sort(s3)` | 293 ns | 135 ns | 2.17 | 136 ns | 2.16 | ok |
| tiny | `reshape(sm, (4, 3))` | 263 ns | 149 ns | 1.77 | 148 ns | 1.78 | ok |
| tiny | `sm.T` | 38 ns | 68 ns | **0.56** | 67 ns | 0.57 | ok |
| tiny | `sm[1]` | 43 ns | 72 ns | **0.59** | 72 ns | 0.59 | ok |
| tiny | `sm[1:, ::2]` | 51 ns | 102 ns | **0.51** | 101 ns | 0.51 | ok |
| tiny | `concat([s3, t3s])` | 238 ns | 260 ns | **0.92** | 261 ns | 0.91 | ok |
| tiny | `where(s3 > 1, s3, t3s)` | 514 ns | 413 ns | 1.24 | 414 ns | 1.24 | ok |
| tiny | `matmul(sm.T, sm)` | 422 ns | 316 ns | 1.34 | 315 ns | 1.34 | ok |
| tiny | `result_type(f, i)` | 55 ns | 47 ns | 1.17 | 47 ns | 1.18 | ok |
| tiny | `can_cast(int64, float64)` | 88 ns | 28 ns | 3.20 | 28 ns | 3.17 | ok |
| tiny | `finfo(float64)` | 110 ns | 20 ns | 5.57 | 20 ns | 5.61 | ok |
| tiny | `s3.shape` | 18 ns | 19 ns | **0.95** | 19 ns | 0.97 | ok |
| tiny | `len(s3)` | 7 ns | 7 ns | **0.94** | 7 ns | 0.94 | ok |
| tiny | `float(s3[0])` | 37 ns | 28 ns | 1.30 | 28 ns | 1.28 | ok |
