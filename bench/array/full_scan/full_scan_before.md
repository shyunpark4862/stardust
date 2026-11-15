# 배열 영역 전체 측정 (numpy 대비)

환경: arm64 macOS, Python 3.15.0, numpy 2.5.4, stardust 86839a1 + 작업 중 변경, 원소 1,000,000개 기준, 10라운드, 2026-11-15 08:15, 걸린 시간 46분

배율 = numpy 시간 ÷ stardust 시간 (1보다 크면 stardust가 빠르다). numpy는 1스레드(BLAS만 여러 스레드).
시간은 7번 반복 중 최솟값을 다시 라운드 사이 최솟값으로 잡았다. **굵은 배율**은 1스레드에서 numpy보다 느린 줄이다.

- 케이스 380개. 1스레드에서 numpy보다 느린 것 95개, 10스레드에서도 느린 것 38개.
- 결과가 numpy와 다른 것: 0개

## 1스레드에서 numpy보다 느린 것 (배율 낮은 순)

| 묶음 | 케이스 | numpy | sd 1T | 배율 1T | sd 10T | 배율 10T |
|---|---|---:|---:|---:|---:|---:|
| convert | `asarray numpy f64 .T` | 12 ns | 149 ns | 0.08 | 148 ns | 0.08 |
| convert | `asarray numpy f64 (view)` | 12 ns | 135 ns | 0.09 | 134 ns | 0.09 |
| convert | `asarray numpy f64 strided` | 12 ns | 134 ns | 0.09 | 134 ns | 0.09 |
| convert | `numpy.asarray(sd array)` | 12 ns | 107 ns | 0.12 | 106 ns | 0.12 |
| convert | `asarray numpy f32 .T (np keeps F)` | 136.8 µs | 870.4 µs | 0.16 | 365.0 µs | 0.37 |
| assign | `fw[:] = 1.0 (fill)` | 62.9 µs | 248.0 µs | 0.25 | 35.8 µs | 1.76 |
| reduce i64 | `argmax` | 125.4 µs | 493.3 µs | 0.25 | 59.2 µs | 2.12 |
| tiny | `finfo(float64)` | 111 ns | 368 ns | 0.30 | 376 ns | 0.29 |
| select | `clip(i, -10, 10)` | 134.9 µs | 376.0 µs | 0.36 | 48.7 µs | 2.77 |
| binary i64 | `pow(i, 2)` | 261.3 µs | 686.3 µs | 0.38 | 80.0 µs | 3.27 |
| tiny | `len(s3)` | 7 ns | 17 ns | 0.39 | 17 ns | 0.39 |
| convert | `asarray numpy >f8 2-D [::2]` | 108.4 µs | 270.4 µs | 0.40 | 36.3 µs | 2.98 |
| manip | `repeat(f, 2)` | 496.8 µs | 1.21 ms | 0.41 | 760.0 µs | 0.65 |
| convert | `from_dlpack numpy` | 115 ns | 255 ns | 0.45 | 258 ns | 0.45 |
| convert | `asarray numpy f32 .T (np C order)` | 433.3 µs | 934.6 µs | 0.46 | 366.7 µs | 1.18 |
| index | `f2[:, 5] (column)` | 48 ns | 103 ns | 0.47 | 102 ns | 0.47 |
| index | `f[1000:2000] (view)` | 43 ns | 89 ns | 0.48 | 86 ns | 0.50 |
| index | `f[::2] (view)` | 43 ns | 88 ns | 0.49 | 88 ns | 0.49 |
| count | `count_nonzero(b)` | 23.4 µs | 47.1 µs | 0.50 | 9.3 µs | 2.52 |
| tiny | `sm[1:, ::2]` | 51 ns | 100 ns | 0.51 | 100 ns | 0.51 |
| tiny | `s3.shape` | 18 ns | 34 ns | 0.52 | 34 ns | 0.52 |
| convert | `asarray numpy f32 rows m[:,1:1001]` | 141.6 µs | 260.1 µs | 0.54 | 36.0 µs | 3.93 |
| tiny | `sm.T` | 38 ns | 65 ns | 0.58 | 64 ns | 0.59 |
| tiny | `sm[1]` | 42 ns | 71 ns | 0.59 | 71 ns | 0.60 |
| index | `f2[5] (row view)` | 42 ns | 71 ns | 0.60 | 71 ns | 0.60 |
| tiny | `zeros(3)` | 60 ns | 100 ns | 0.61 | 100 ns | 0.60 |
| tiny | `empty(3)` | 60 ns | 98 ns | 0.61 | 98 ns | 0.61 |
| tiny | `asarray(s3) (no-op)` | 12 ns | 20 ns | 0.63 | 20 ns | 0.63 |
| matmul | `vecdot(f2, g2)` | 131.9 µs | 200.1 µs | 0.66 | 28.1 µs | 4.70 |
| creation | `empty(N)` | 110 ns | 148 ns | 0.74 | 145 ns | 0.75 |
| manip | `unstack(f2)` | 46.6 µs | 61.1 µs | 0.76 | 60.5 µs | 0.77 |
| unary f64 | `isinf` | 71.6 µs | 93.6 µs | 0.76 | 21.1 µs | 3.40 |
| unary f64 | `isfinite` | 72.0 µs | 93.6 µs | 0.77 | 20.9 µs | 3.44 |
| unary i64 | `sign` | 96.8 µs | 124.4 µs | 0.78 | 31.3 µs | 3.10 |
| assign | `x[1:] += x[:-1] (overlap)` | 237.5 µs | 299.7 µs | 0.79 | 103.5 µs | 2.29 |
| tiny | `sqrt(s3)` | 123 ns | 147 ns | 0.83 | 148 ns | 0.83 |
| tiny | `exp(s3)` | 126 ns | 151 ns | 0.84 | 153 ns | 0.83 |
| tiny | `s3 < t3s` | 149 ns | 177 ns | 0.84 | 180 ns | 0.83 |
| tiny | `s3 + t3s` | 150 ns | 177 ns | 0.85 | 181 ns | 0.83 |
| unary c128 | `square` | 266.2 µs | 310.8 µs | 0.86 | 67.3 µs | 3.96 |
| unary f64 | `acosh` | 2.49 ms | 2.91 ms | 0.86 | 317.9 µs | 7.85 |
| size 1e3 | `argsort(a) (1e3)` | 7.6 µs | 8.9 µs | 0.86 | 8.8 µs | 0.86 |
| assign | `gz[idx] = 1.0 (scatter 1e5)` | 95.6 µs | 105.4 µs | 0.91 | 80.3 µs | 1.19 |
| binary layout | `ft + f2 (mixed order)` | 498.8 µs | 549.7 µs | 0.91 | 61.0 µs | 8.18 |
| binary mixed | `c * f` | 581.9 µs | 639.8 µs | 0.91 | 111.7 µs | 5.21 |
| search | `take(f2, rows, axis=0)` | 111.5 µs | 122.4 µs | 0.91 | 32.3 µs | 3.46 |
| select | `where(b, f, 0.0)` | 261.7 µs | 286.4 µs | 0.91 | 39.6 µs | 6.61 |
| unary f64 | `negative` | 120.6 µs | 132.0 µs | 0.91 | 30.3 µs | 3.98 |
| convert | `asarray list 3` | 131 ns | 142 ns | 0.92 | 142 ns | 0.92 |
| assign | `gz[idxn] = 1.0 (scatter 1e6)` | 960.9 µs | 1.03 ms | 0.94 | 807.7 µs | 1.19 |
| binary mid | `fm + gm (1e4)` | 1.6 µs | 1.7 µs | 0.95 | 1.8 µs | 0.90 |
| stats | `nanquantile(fn, q5)` | 11.49 ms | 12.12 ms | 0.95 | 3.85 ms | 2.99 |
| select | `where(b, f, g)` | 275.5 µs | 289.5 µs | 0.95 | 42.6 µs | 6.47 |
| manip | `roll(f2, 1, axis=1)` | 160.3 µs | 168.0 µs | 0.95 | 33.7 µs | 4.76 |
| search | `searchsorted(srt, f)` | 53.42 ms | 55.58 ms | 0.96 | 6.39 ms | 8.36 |
| manip | `astype(f, M.complex128)` | 156.3 µs | 162.3 µs | 0.96 | 59.8 µs | 2.62 |
| size 1e3 | `i + j (1e3)` | 284 ns | 293 ns | 0.97 | 292 ns | 0.97 |
| tiny | `concat([s3, t3s])` | 244 ns | 252 ns | 0.97 | 257 ns | 0.95 |
| manip | `tile(v, (1000,))` | 72.3 µs | 74.7 µs | 0.97 | 30.1 µs | 2.40 |
| manip | `concat([f2, g2], axis=1)` | 344.9 µs | 356.2 µs | 0.97 | 78.5 µs | 4.39 |
| convert | `asarray numpy i32` | 79.2 µs | 81.5 µs | 0.97 | 31.7 µs | 2.50 |
| reduce axis | `sum(fst) (strided)` | 149.7 µs | 153.9 µs | 0.97 | 27.6 µs | 5.41 |
| reduce axis | `mean(f2, axis=0)` | 129.3 µs | 131.8 µs | 0.98 | 32.8 µs | 3.94 |
| matmul | `matmul batch 1000 x 8x8` | 70.5 µs | 71.7 µs | 0.98 | 13.9 µs | 5.09 |
| creation | `zeros(N)` | 28.3 µs | 28.8 µs | 0.98 | 28.9 µs | 0.98 |
| reduce axis | `var(f2, axis=0)` | 633.0 µs | 642.8 µs | 0.98 | 111.8 µs | 5.66 |
| assign | `fw[::2] = 1.0 (strided fill)` | 122.6 µs | 124.5 µs | 0.99 | 21.4 µs | 5.73 |
| reduce axis | `sum(f2, axis=0)` | 128.5 µs | 130.4 µs | 0.99 | 31.4 µs | 4.09 |
| unary c128 | `conj` | 193.8 µs | 196.7 µs | 0.99 | 67.7 µs | 2.86 |
| reduce axis | `max(f2, axis=0)` | 128.2 µs | 130.0 µs | 0.99 | 31.7 µs | 4.05 |
| size 1e7 | `sum(a) (1e7)` | 1.19 ms | 1.20 ms | 0.99 | 478.4 µs | 2.48 |
| binary c128 | `multiply` | 320.4 µs | 324.2 µs | 0.99 | 126.4 µs | 2.54 |
| creation | `arange(N)` | 85.9 µs | 86.7 µs | 0.99 | 86.4 µs | 0.99 |
| binary f64 | `maximum` | 146.5 µs | 147.8 µs | 0.99 | 38.7 µs | 3.78 |
| convert | `asarray numpy f32` | 93.8 µs | 94.6 µs | 0.99 | 31.5 µs | 2.98 |
| unary f64 | `square` | 94.0 µs | 94.7 µs | 0.99 | 30.4 µs | 3.09 |
| binary f64 | `divide` | 145.5 µs | 146.4 µs | 0.99 | 38.5 µs | 3.78 |
| manip | `concat([f, g])` | 190.3 µs | 191.2 µs | 1.00 | 71.1 µs | 2.68 |
| unary f64 | `ceil` | 94.0 µs | 94.4 µs | 1.00 | 30.3 µs | 3.10 |
| size 1e3 | `a + b (1e3)` | 285 ns | 286 ns | 1.00 | 290 ns | 0.98 |
| size 1e7 | `i + j (1e7)` | 1.96 ms | 1.97 ms | 1.00 | 1.15 ms | 1.70 |
| size 1e3 | `copy (1e3)` | 208 ns | 208 ns | 1.00 | 209 ns | 0.99 |
| binary f64 | `multiply` | 145.0 µs | 145.3 µs | 1.00 | 38.4 µs | 3.78 |
| binary i64 | `equal` | 135.8 µs | 136.1 µs | 1.00 | 26.6 µs | 5.10 |
| unary i64 | `bitwise_invert` | 94.4 µs | 94.6 µs | 1.00 | 30.2 µs | 3.12 |
| unary f64 | `isnan` | 71.6 µs | 71.7 µs | 1.00 | 21.4 µs | 3.34 |
| matmul | `matmul 512` | 321.3 µs | 321.7 µs | 1.00 | 321.5 µs | 1.00 |
| creation | `asarray(copy=True)` | 94.0 µs | 94.0 µs | 1.00 | 34.6 µs | 2.72 |
| unary f64 | `log1p` | 2.48 ms | 2.48 ms | 1.00 | 272.4 µs | 9.09 |
| size 1e7 | `a * 2.0 (1e7)` | 1.31 ms | 1.31 ms | 1.00 | 717.1 µs | 1.83 |
| size 1e5 | `i + j (1e5)` | 14.3 µs | 14.3 µs | 1.00 | 5.9 µs | 2.42 |
| size 1e5 | `copy (1e5)` | 9.5 µs | 9.6 µs | 1.00 | 5.7 µs | 1.67 |
| unary f64 | `expm1` | 1.86 ms | 1.86 ms | 1.00 | 206.2 µs | 9.01 |
| manip | `stack([f, g])` | 191.6 µs | 191.7 µs | 1.00 | 71.0 µs | 2.70 |
| binary i64 | `multiply` | 267.5 µs | 267.5 µs | 1.00 | 40.5 µs | 6.60 |

## 전체

| 묶음 | 케이스 | numpy | sd 1T | 배율 1T | sd 10T | 배율 10T | 결과 확인 |
|---|---|---:|---:|---:|---:|---:|---|
| creation | `zeros(N)` | 28.3 µs | 28.8 µs | **0.98** | 28.9 µs | 0.98 | ok |
| creation | `ones(N)` | 59.0 µs | 58.0 µs | 1.02 | 57.9 µs | 1.02 | ok |
| creation | `empty(N)` | 110 ns | 148 ns | **0.74** | 145 ns | 0.75 | ok |
| creation | `full(N, 2.5)` | 59.6 µs | 58.5 µs | 1.02 | 57.6 µs | 1.03 | ok |
| creation | `zeros_like(f)` | 59.0 µs | 28.5 µs | 2.07 | 28.9 µs | 2.04 | ok |
| creation | `ones_like(f)` | 59.1 µs | 58.8 µs | 1.00 | 59.5 µs | 0.99 | ok |
| creation | `empty_like(f)` | 145 ns | 130 ns | 1.12 | 131 ns | 1.11 | ok |
| creation | `full_like(f, 2.5)` | 60.1 µs | 59.1 µs | 1.02 | 59.2 µs | 1.02 | ok |
| creation | `arange(N)` | 85.9 µs | 86.7 µs | **0.99** | 86.4 µs | 0.99 | ok |
| creation | `linspace(0.0, 1.0, N)` | 463.2 µs | 126.0 µs | 3.68 | 124.3 µs | 3.73 | ok |
| creation | `eye(1000)` | 37.2 µs | 33.3 µs | 1.12 | 33.9 µs | 1.10 | ok |
| creation | `meshgrid(v, v)` | 149.2 µs | 132.0 µs | 1.13 | 132.9 µs | 1.12 | ok |
| creation | `tril(f2)` | 353.1 µs | 183.3 µs | 1.93 | 141.7 µs | 2.49 | ok |
| creation | `triu(f2)` | 353.1 µs | 167.4 µs | 2.11 | 112.4 µs | 3.14 | ok |
| creation | `zeros((1000, 1000))` | 29.4 µs | 28.6 µs | 1.03 | 28.9 µs | 1.02 | ok |
| creation | `arange(N, float64)` | 240.6 µs | 93.6 µs | 2.57 | 93.7 µs | 2.57 | ok |
| creation | `asarray(copy=True)` | 94.0 µs | 94.0 µs | **1.00** | 34.6 µs | 2.72 | ok |
| convert | `asarray numpy f64 (view)` | 12 ns | 135 ns | **0.09** | 134 ns | 0.09 | ok |
| convert | `asarray numpy f64 strided` | 12 ns | 134 ns | **0.09** | 134 ns | 0.09 | ok |
| convert | `asarray numpy f64 .T` | 12 ns | 149 ns | **0.08** | 148 ns | 0.08 | ok |
| convert | `asarray numpy f32` | 93.8 µs | 94.6 µs | **0.99** | 31.5 µs | 2.98 | ok |
| convert | `asarray numpy f32 .T (np keeps F)` | 136.8 µs | 870.4 µs | **0.16** | 365.0 µs | 0.37 | ok |
| convert | `asarray numpy f32 .T (np C order)` | 433.3 µs | 934.6 µs | **0.46** | 366.7 µs | 1.18 | ok |
| convert | `asarray numpy f32 rows m[:,1:1001]` | 141.6 µs | 260.1 µs | **0.54** | 36.0 µs | 3.93 | ok |
| convert | `asarray numpy >f8` | 95.6 µs | 94.5 µs | 1.01 | 32.3 µs | 2.96 | ok |
| convert | `asarray numpy >f8 2-D [::2]` | 108.4 µs | 270.4 µs | **0.40** | 36.3 µs | 2.98 | ok |
| convert | `asarray numpy i32` | 79.2 µs | 81.5 µs | **0.97** | 31.7 µs | 2.50 | ok |
| convert | `asarray list 3` | 131 ns | 142 ns | **0.92** | 142 ns | 0.92 | ok |
| convert | `asarray list 1e4` | 140.4 µs | 62.8 µs | 2.24 | 62.8 µs | 2.24 | ok |
| convert | `from_dlpack numpy` | 115 ns | 255 ns | **0.45** | 258 ns | 0.45 | ok |
| convert | `numpy.asarray(sd array)` | 12 ns | 107 ns | **0.12** | 106 ns | 0.12 | ok |
| manip | `reshape(f, (1000, 1000))` | 276 ns | 140 ns | 1.97 | 140 ns | 1.97 | ok |
| manip | `reshape(ft, (N,))` | 492.0 µs | 401.1 µs | 1.23 | 54.4 µs | 9.04 | ok |
| manip | `permute_dims(t3, (2, 0, 1))` | 179 ns | 162 ns | 1.11 | 162 ns | 1.11 | ok |
| manip | `matrix_transpose(f2)` | 250 ns | 84 ns | 2.98 | 84 ns | 2.99 | ok |
| manip | `moveaxis(t3, 0, -1)` | 1.1 µs | 187 ns | 5.75 | 184 ns | 5.85 | ok |
| manip | `squeeze(M.reshape(f, (1, N)), axis=0)` | 427 ns | 243 ns | 1.76 | 242 ns | 1.77 | ok |
| manip | `expand_dims(f, axis=0)` | 608 ns | 99 ns | 6.14 | 98 ns | 6.20 | ok |
| manip | `broadcast_to(v, (1000, 1000))` | 966 ns | 112 ns | 8.63 | 112 ns | 8.63 | ok |
| manip | `broadcast_arrays(f2, v)` | 1.7 µs | 225 ns | 7.41 | 226 ns | 7.39 | ok |
| manip | `concat([f, g])` | 190.3 µs | 191.2 µs | **1.00** | 71.1 µs | 2.68 | ok |
| manip | `concat([f2, g2], axis=1)` | 344.9 µs | 356.2 µs | **0.97** | 78.5 µs | 4.39 | ok |
| manip | `stack([f, g])` | 191.6 µs | 191.7 µs | **1.00** | 71.0 µs | 2.70 | ok |
| manip | `flip(f)` | 179 ns | 82 ns | 2.19 | 82 ns | 2.20 | ok |
| manip | `roll(f, 1)` | 96.4 µs | 94.7 µs | 1.02 | 35.6 µs | 2.71 | ok |
| manip | `roll(f2, 1, axis=1)` | 160.3 µs | 168.0 µs | **0.95** | 33.7 µs | 4.76 | ok |
| manip | `repeat(f, 2)` | 496.8 µs | 1.21 ms | **0.41** | 760.0 µs | 0.65 | ok |
| manip | `tile(v, (1000,))` | 72.3 µs | 74.7 µs | **0.97** | 30.1 µs | 2.40 | ok |
| manip | `unstack(f2)` | 46.6 µs | 61.1 µs | **0.76** | 60.5 µs | 0.77 | ok |
| manip | `astype(f, M.int64)` | 267.3 µs | 126.9 µs | 2.11 | 30.2 µs | 8.86 | ok |
| manip | `astype(i, M.float64)` | 250.5 µs | 107.5 µs | 2.33 | 30.2 µs | 8.29 | ok |
| manip | `astype(f, M.complex128)` | 156.3 µs | 162.3 µs | **0.96** | 59.8 µs | 2.62 | ok |
| manip | `astype(b, M.float64)` | 249.1 µs | 152.4 µs | 1.63 | 29.2 µs | 8.53 | ok |
| manip | `astype(ft, M.float64)` | 94.4 µs | 94.1 µs | 1.00 | 34.5 µs | 2.73 | ok |
| manip | `ravel(ft) (copy)` | 471.4 µs | 331.5 µs | 1.42 | 54.6 µs | 8.64 | ok |
| manip | `diff(f)` | 106.0 µs | 104.6 µs | 1.01 | 33.0 µs | 3.22 | ok |
| binary f64 | `add` | 201.1 µs | 150.4 µs | 1.34 | 38.4 µs | 5.24 | ok |
| binary f64 | `subtract` | 147.4 µs | 145.5 µs | 1.01 | 38.4 µs | 3.84 | ok |
| binary f64 | `multiply` | 145.0 µs | 145.3 µs | **1.00** | 38.4 µs | 3.78 | ok |
| binary f64 | `divide` | 145.5 µs | 146.4 µs | **0.99** | 38.5 µs | 3.78 | ok |
| binary f64 | `floor_divide` | 6.49 ms | 6.08 ms | 1.07 | 672.3 µs | 9.65 | ok |
| binary f64 | `remainder` | 6.53 ms | 5.61 ms | 1.16 | 613.3 µs | 10.64 | ok |
| binary f64 | `pow` | 4.61 ms | 2.65 ms | 1.74 | 293.5 µs | 15.72 | ok |
| binary f64 | `atan2` | 12.83 ms | 1.90 ms | 6.76 | 212.1 µs | 60.50 | ok |
| binary f64 | `maximum` | 146.5 µs | 147.8 µs | **0.99** | 38.7 µs | 3.78 | ok |
| binary f64 | `minimum` | 147.3 µs | 146.0 µs | 1.01 | 38.7 µs | 3.80 | ok |
| binary f64 | `copysign` | 252.6 µs | 146.3 µs | 1.73 | 38.4 µs | 6.58 | ok |
| binary f64 | `hypot` | 1.48 ms | 1.24 ms | 1.20 | 139.7 µs | 10.62 | ok |
| binary f64 | `logaddexp` | 10.35 ms | 9.94 ms | 1.04 | 1.07 ms | 9.70 | ok |
| binary f64 | `nextafter` | 5.29 ms | 647.8 µs | 8.17 | 76.6 µs | 69.12 | ok |
| binary f64 | `equal` | 201.1 µs | 135.2 µs | 1.49 | 26.4 µs | 7.60 | ok |
| binary f64 | `less` | 200.8 µs | 135.5 µs | 1.48 | 26.5 µs | 7.58 | ok |
| binary f64 | `greater_equal` | 200.6 µs | 135.5 µs | 1.48 | 26.5 µs | 7.58 | ok |
| binary bool | `logical_and` | 25.6 µs | 17.9 µs | 1.43 | 6.7 µs | 3.83 | ok |
| binary bool | `logical_or` | 25.6 µs | 17.9 µs | 1.43 | 6.6 µs | 3.86 | ok |
| binary bool | `logical_xor` | 18.4 µs | 17.9 µs | 1.03 | 6.7 µs | 2.75 | ok |
| binary i64 | `add` | 146.7 µs | 145.3 µs | 1.01 | 38.5 µs | 3.81 | ok |
| binary i64 | `multiply` | 267.5 µs | 267.5 µs | **1.00** | 40.5 µs | 6.60 | ok |
| binary i64 | `floor_divide` | 735.3 µs | 590.4 µs | 1.25 | 76.1 µs | 9.66 | ok |
| binary i64 | `remainder` | 878.8 µs | 669.6 µs | 1.31 | 84.0 µs | 10.46 | ok |
| binary i64 | `maximum` | 146.2 µs | 145.7 µs | 1.00 | 38.7 µs | 3.78 | ok |
| binary i64 | `equal` | 135.8 µs | 136.1 µs | **1.00** | 26.6 µs | 5.10 | ok |
| binary i64 | `bitwise_and` | 147.9 µs | 145.4 µs | 1.02 | 38.5 µs | 3.84 | ok |
| binary i64 | `bitwise_or` | 147.7 µs | 144.6 µs | 1.02 | 38.7 µs | 3.82 | ok |
| binary i64 | `bitwise_xor` | 146.3 µs | 145.3 µs | 1.01 | 38.6 µs | 3.79 | ok |
| binary i64 | `bitwise_left_shift` | 150.0 µs | 145.9 µs | 1.03 | 38.9 µs | 3.86 | ok |
| binary i64 | `bitwise_right_shift` | 148.8 µs | 148.4 µs | 1.00 | 39.0 µs | 3.82 | ok |
| binary i64 | `pow(i, 2)` | 261.3 µs | 686.3 µs | **0.38** | 80.0 µs | 3.27 | ok |
| binary c128 | `add` | 404.4 µs | 326.4 µs | 1.24 | 126.0 µs | 3.21 | ok |
| binary c128 | `multiply` | 320.4 µs | 324.2 µs | **0.99** | 126.4 µs | 2.54 | ok |
| binary c128 | `divide` | 4.40 ms | 1.63 ms | 2.71 | 183.0 µs | 24.06 | ok |
| binary c128 | `equal` | 275.6 µs | 273.2 µs | 1.01 | 91.0 µs | 3.03 | ok |
| binary c128 | `pow` | 35.70 ms | 34.70 ms | 1.03 | 3.73 ms | 9.58 | ok |
| binary mixed | `f * i (cast)` | 431.6 µs | 222.4 µs | 1.94 | 44.2 µs | 9.76 | ok |
| binary mixed | `f * i + g` | 591.5 µs | 423.4 µs | 1.40 | 154.0 µs | 3.84 | ok |
| binary mixed | `f + 1.0 (scalar)` | 94.2 µs | 94.0 µs | 1.00 | 31.1 µs | 3.03 | ok |
| binary mixed | `i + 1 (scalar)` | 132.7 µs | 95.2 µs | 1.39 | 30.7 µs | 4.33 | ok |
| binary mixed | `c * f` | 581.9 µs | 639.8 µs | **0.91** | 111.7 µs | 5.21 | ok |
| binary layout | `f2 + v (broadcast)` | 228.4 µs | 109.4 µs | 2.09 | 31.6 µs | 7.23 | ok |
| binary layout | `f2 + v[:, None] (broadcast col)` | 193.3 µs | 105.7 µs | 1.83 | 31.2 µs | 6.19 | ok |
| binary layout | `ft + 1.0 (transposed)` | 94.6 µs | 94.0 µs | 1.01 | 30.6 µs | 3.09 | ok |
| binary layout | `ft + f2 (mixed order)` | 498.8 µs | 549.7 µs | **0.91** | 61.0 µs | 8.18 | ok |
| binary layout | `fst + 1.0 (strided)` | 253.3 µs | 250.6 µs | 1.01 | 39.3 µs | 6.44 | ok |
| binary layout | `fst + f (strided + contiguous)` | 260.7 µs | 259.0 µs | 1.01 | 73.5 µs | 3.55 | ok |
| binary mid | `fm + gm (1e4)` | 1.6 µs | 1.7 µs | **0.95** | 1.8 µs | 0.90 | ok |
| binary mid | `fm * 2.0 (1e4)` | 1.0 µs | 984 ns | 1.06 | 969 ns | 1.07 | ok |
| unary f64 | `negative` | 120.6 µs | 132.0 µs | **0.91** | 30.3 µs | 3.98 | ok |
| unary f64 | `positive` | 247.9 µs | 94.1 µs | 2.63 | 30.2 µs | 8.21 | ok |
| unary f64 | `abs` | 132.3 µs | 94.1 µs | 1.41 | 30.6 µs | 4.33 | ok |
| unary f64 | `square` | 94.0 µs | 94.7 µs | **0.99** | 30.4 µs | 3.09 | ok |
| unary f64 | `sqrt` | 247.2 µs | 245.2 µs | 1.01 | 35.4 µs | 6.97 | ok |
| unary f64 | `sign` | 299.1 µs | 219.1 µs | 1.37 | 34.1 µs | 8.77 | ok |
| unary f64 | `exp` | 1.49 ms | 1.07 ms | 1.39 | 121.6 µs | 12.24 | ok |
| unary f64 | `expm1` | 1.86 ms | 1.86 ms | **1.00** | 206.2 µs | 9.01 | ok |
| unary f64 | `log` | 3.14 ms | 1.09 ms | 2.89 | 123.2 µs | 25.50 | ok |
| unary f64 | `log1p` | 2.48 ms | 2.48 ms | **1.00** | 272.4 µs | 9.09 | ok |
| unary f64 | `log2` | 3.44 ms | 1.03 ms | 3.33 | 117.8 µs | 29.21 | ok |
| unary f64 | `log10` | 3.53 ms | 1.09 ms | 3.25 | 123.3 µs | 28.62 | ok |
| unary f64 | `sin` | 3.81 ms | 802.6 µs | 4.75 | 93.3 µs | 40.88 | ok |
| unary f64 | `cos` | 3.74 ms | 3.69 ms | 1.02 | 399.3 µs | 9.38 | ok |
| unary f64 | `tan` | 4.16 ms | 4.12 ms | 1.01 | 444.0 µs | 9.37 | ok |
| unary f64 | `atan` | 7.63 ms | 1.12 ms | 6.84 | 127.0 µs | 60.10 | ok |
| unary f64 | `sinh` | 2.61 ms | 1.02 ms | 2.55 | 116.3 µs | 22.43 | ok |
| unary f64 | `cosh` | 2.48 ms | 959.5 µs | 2.58 | 109.8 µs | 22.57 | ok |
| unary f64 | `tanh` | 2.22 ms | 1.19 ms | 1.86 | 135.0 µs | 16.41 | ok |
| unary f64 | `asinh` | 4.79 ms | 3.60 ms | 1.33 | 394.2 µs | 12.15 | ok |
| unary f64 | `floor` | 94.1 µs | 94.0 µs | 1.00 | 30.4 µs | 3.10 | ok |
| unary f64 | `ceil` | 94.0 µs | 94.4 µs | **1.00** | 30.3 µs | 3.10 | ok |
| unary f64 | `trunc` | 94.5 µs | 94.1 µs | 1.00 | 30.4 µs | 3.11 | ok |
| unary f64 | `round` | 95.1 µs | 94.0 µs | 1.01 | 30.5 µs | 3.12 | ok |
| unary f64 | `isnan` | 71.6 µs | 71.7 µs | **1.00** | 21.4 µs | 3.34 | ok |
| unary f64 | `isinf` | 71.6 µs | 93.6 µs | **0.76** | 21.1 µs | 3.40 | ok |
| unary f64 | `isfinite` | 72.0 µs | 93.6 µs | **0.77** | 20.9 µs | 3.44 | ok |
| unary f64 | `signbit` | 71.6 µs | 71.5 µs | 1.00 | 21.5 µs | 3.33 | ok |
| unary f64 | `reciprocal` | 124.8 µs | 124.4 µs | 1.00 | 31.0 µs | 4.03 | ok |
| unary f64 | `asin` | 5.34 ms | 936.7 µs | 5.71 | 107.2 µs | 49.86 | ok |
| unary f64 | `acos` | 6.95 ms | 1.28 ms | 5.45 | 143.9 µs | 48.32 | ok |
| unary f64 | `atanh` | 5.33 ms | 2.18 ms | 2.45 | 240.8 µs | 22.15 | ok |
| unary f64 | `acosh` | 2.49 ms | 2.91 ms | **0.86** | 317.9 µs | 7.85 | ok |
| unary f64 | `sign (both signs)` | 3.55 ms | 219.2 µs | 16.17 | 34.1 µs | 103.98 | ok |
| unary f64 | `exp strided` | 1.49 ms | 1.26 ms | 1.19 | 142.6 µs | 10.45 | ok |
| unary f64 | `sqrt transposed` | 248.2 µs | 247.0 µs | 1.00 | 35.7 µs | 6.96 | ok |
| unary f64 | `exp (1e4)` | 15.1 µs | 10.9 µs | 1.39 | 10.9 µs | 1.39 | ok |
| unary c128 | `abs` | 1.64 ms | 1.27 ms | 1.29 | 143.0 µs | 11.45 | ok |
| unary c128 | `exp` | 8.19 ms | 7.14 ms | 1.15 | 773.3 µs | 10.59 | ok |
| unary c128 | `sqrt` | 2.23 ms | 1.56 ms | 1.43 | 177.0 µs | 12.60 | ok |
| unary c128 | `log` | 22.71 ms | 20.80 ms | 1.09 | 2.23 ms | 10.21 | ok |
| unary c128 | `sin` | 10.27 ms | 9.37 ms | 1.10 | 1.01 ms | 10.17 | ok |
| unary c128 | `conj` | 193.8 µs | 196.7 µs | **0.99** | 67.7 µs | 2.86 | ok |
| unary c128 | `real` | 85 ns | 65 ns | 1.30 | 65 ns | 1.30 | ok |
| unary c128 | `imag` | 85 ns | 65 ns | 1.30 | 65 ns | 1.30 | ok |
| unary c128 | `square` | 266.2 µs | 310.8 µs | **0.86** | 67.3 µs | 3.96 | ok |
| unary c128 | `isnan` | 320.9 µs | 203.4 µs | 1.58 | 30.4 µs | 10.55 | ok |
| unary i64 | `abs` | 132.4 µs | 97.0 µs | 1.37 | 30.3 µs | 4.37 | ok |
| unary i64 | `negative` | 95.5 µs | 94.5 µs | 1.01 | 30.3 µs | 3.15 | ok |
| unary i64 | `square` | 254.2 µs | 254.1 µs | 1.00 | 36.1 µs | 7.05 | ok |
| unary i64 | `sign` | 96.8 µs | 124.4 µs | **0.78** | 31.3 µs | 3.10 | ok |
| unary i64 | `bitwise_invert` | 94.4 µs | 94.6 µs | **1.00** | 30.2 µs | 3.12 | ok |
| unary bool | `logical_not` | 18.7 µs | 12.0 µs | 1.55 | 6.2 µs | 3.01 | ok |
| select | `clip(f, 0.6, 1.2)` | 253.9 µs | 96.9 µs | 2.62 | 30.7 µs | 8.26 | ok |
| select | `clip(i, -10, 10)` | 134.9 µs | 376.0 µs | **0.36** | 48.7 µs | 2.77 | ok |
| select | `where(b, f, g)` | 275.5 µs | 289.5 µs | **0.95** | 42.6 µs | 6.47 | ok |
| select | `where(b, f, 0.0)` | 261.7 µs | 286.4 µs | **0.91** | 39.6 µs | 6.61 | ok |
| reduce f64 | `sum` | 112.9 µs | 110.0 µs | 1.03 | 24.3 µs | 4.64 | ok |
| reduce f64 | `prod` | 804.8 µs | 109.4 µs | 7.35 | 21.6 µs | 37.23 | ok |
| reduce f64 | `min` | 75.2 µs | 71.7 µs | 1.05 | 22.7 µs | 3.31 | ok |
| reduce f64 | `max` | 75.3 µs | 71.7 µs | 1.05 | 22.7 µs | 3.31 | ok |
| reduce f64 | `mean` | 113.4 µs | 110.1 µs | 1.03 | 24.4 µs | 4.64 | ok |
| reduce f64 | `var` | 444.6 µs | 311.5 µs | 1.43 | 58.0 µs | 7.67 | ok |
| reduce f64 | `std` | 480.9 µs | 311.6 µs | 1.54 | 58.0 µs | 8.30 | ok |
| reduce f64 | `argmin` | 559.7 µs | 494.8 µs | 1.13 | 57.3 µs | 9.76 | ok |
| reduce f64 | `argmax` | 559.6 µs | 483.5 µs | 1.16 | 57.1 µs | 9.80 | ok |
| reduce i64 | `sum` | 72.5 µs | 71.9 µs | 1.01 | 22.5 µs | 3.23 | ok |
| reduce i64 | `min` | 76.6 µs | 71.7 µs | 1.07 | 22.0 µs | 3.49 | ok |
| reduce i64 | `max` | 76.5 µs | 71.6 µs | 1.07 | 21.9 µs | 3.49 | ok |
| reduce i64 | `mean` | 361.1 µs | 203.7 µs | 1.77 | 31.0 µs | 11.66 | ok |
| reduce i64 | `var` | 926.8 µs | 518.4 µs | 1.79 | 100.5 µs | 9.23 | ok |
| reduce i64 | `argmax` | 125.4 µs | 493.3 µs | **0.25** | 59.2 µs | 2.12 | ok |
| reduce c128 | `sum` | 190.6 µs | 174.8 µs | 1.09 | 45.0 µs | 4.24 | ok |
| reduce c128 | `prod` | 4.01 ms | 277.5 µs | 14.44 | 36.7 µs | 109.37 | ok |
| reduce c128 | `mean` | 192.4 µs | 174.9 µs | 1.10 | 45.0 µs | 4.27 | ok |
| reduce c128 | `var` | 1.07 ms | 488.8 µs | 2.18 | 91.6 µs | 11.66 | ok |
| reduce bool | `any` | 484 ns | 109 ns | 4.45 | 112 ns | 4.32 | ok |
| reduce bool | `all` | 478 ns | 107 ns | 4.46 | 111 ns | 4.33 | ok |
| reduce bool | `sum(b)` | 156.4 µs | 133.3 µs | 1.17 | 22.2 µs | 7.05 | ok |
| reduce axis | `sum(f2, axis=0)` | 128.5 µs | 130.4 µs | **0.99** | 31.4 µs | 4.09 | ok |
| reduce axis | `sum(f2, axis=1)` | 118.8 µs | 114.2 µs | 1.04 | 25.1 µs | 4.73 | ok |
| reduce axis | `mean(f2, axis=0)` | 129.3 µs | 131.8 µs | **0.98** | 32.8 µs | 3.94 | ok |
| reduce axis | `mean(f2, axis=1)` | 119.9 µs | 115.4 µs | 1.04 | 26.3 µs | 4.56 | ok |
| reduce axis | `max(f2, axis=0)` | 128.2 µs | 130.0 µs | **0.99** | 31.7 µs | 4.05 | ok |
| reduce axis | `max(f2, axis=1)` | 83.5 µs | 75.4 µs | 1.11 | 21.9 µs | 3.81 | ok |
| reduce axis | `var(f2, axis=0)` | 633.0 µs | 642.8 µs | **0.98** | 111.8 µs | 5.66 | ok |
| reduce axis | `var(f2, axis=1)` | 578.8 µs | 322.2 µs | 1.80 | 63.2 µs | 9.15 | ok |
| reduce axis | `argmax(f2, axis=0)` | 985.2 µs | 955.3 µs | 1.03 | 108.7 µs | 9.07 | ok |
| reduce axis | `argmax(f2, axis=1)` | 529.7 µs | 474.3 µs | 1.12 | 57.6 µs | 9.20 | ok |
| reduce axis | `sum(t3, axis=(0, 2))` | 170.3 µs | 168.4 µs | 1.01 | 41.1 µs | 4.15 | ok |
| reduce axis | `sum(ft) (transposed)` | 112.9 µs | 110.1 µs | 1.02 | 24.3 µs | 4.64 | ok |
| reduce axis | `sum(ft, axis=0)` | 118.8 µs | 114.3 µs | 1.04 | 25.1 µs | 4.73 | ok |
| reduce axis | `sum(fst) (strided)` | 149.7 µs | 153.9 µs | **0.97** | 27.6 µs | 5.41 | ok |
| reduce axis | `sum(i2, axis=0)` | 278.6 µs | 130.6 µs | 2.13 | 31.5 µs | 8.85 | ok |
| reduce mid | `sum(fm) (1e4)` | 1.7 µs | 1.2 µs | 1.42 | 1.2 µs | 1.41 | ok |
| reduce mid | `var(fm) (1e4)` | 8.8 µs | 3.3 µs | 2.64 | 3.3 µs | 2.63 | ok |
| scan | `cumulative_sum(f)` | 2.20 ms | 389.3 µs | 5.65 | 69.2 µs | 31.81 | ok |
| scan | `cumulative_prod(g)` | 2.49 ms | 423.2 µs | 5.88 | 73.7 µs | 33.78 | ok |
| scan | `cumulative_sum(i)` | 1.56 ms | 264.2 µs | 5.91 | 85.4 µs | 18.29 | ok |
| scan | `cumulative_sum(f2, axis=0)` | 3.03 ms | 621.7 µs | 4.87 | 344.1 µs | 8.79 | ok |
| scan | `cumulative_sum(f2, axis=1)` | 2.37 ms | 536.5 µs | 4.41 | 65.2 µs | 36.30 | ok |
| scan | `cumulative_sum(c)` | 3.28 ms | 657.9 µs | 4.98 | 170.1 µs | 19.28 | ok |
| nan | `nansum` | 1.28 ms | 156.9 µs | 8.17 | 26.2 µs | 49.01 | ok |
| nan | `nanprod` | 2.01 ms | 115.6 µs | 17.41 | 21.3 µs | 94.54 | ok |
| nan | `nanmin` | 75.9 µs | 71.7 µs | 1.06 | 22.7 µs | 3.35 | ok |
| nan | `nanmax` | 75.9 µs | 71.7 µs | 1.06 | 22.7 µs | 3.34 | ok |
| nan | `nanmean` | 1.46 ms | 230.4 µs | 6.33 | 39.7 µs | 36.73 | ok |
| nan | `nanvar` | 3.33 ms | 540.3 µs | 6.16 | 80.6 µs | 41.25 | ok |
| nan | `nanstd` | 3.36 ms | 540.3 µs | 6.21 | 80.6 µs | 41.66 | ok |
| nan | `nanargmin` | 1.72 ms | 566.0 µs | 3.04 | 71.1 µs | 24.22 | ok |
| nan | `nanargmax` | 1.71 ms | 563.3 µs | 3.04 | 71.3 µs | 24.02 | ok |
| nan | `nancumulative_sum` | 3.35 ms | 402.7 µs | 8.32 | 72.1 µs | 46.51 | ok |
| nan | `nancumulative_prod` | 3.67 ms | 416.7 µs | 8.82 | 72.9 µs | 50.39 | ok |
| nan | `nansum(f2-like, axis=0)` | 1.28 ms | 131.4 µs | 9.75 | 32.7 µs | 39.17 | ok |
| count | `count_nonzero(b)` | 23.4 µs | 47.1 µs | **0.50** | 9.3 µs | 2.52 | ok |
| count | `count_nonzero(f)` | 248.0 µs | 93.9 µs | 2.64 | 15.9 µs | 15.63 | ok |
| count | `count_nonzero(f2, axis=0)` | 624.6 µs | 334.1 µs | 1.87 | 59.4 µs | 10.52 | ok |
| stats | `quantile(f, q5)` | 14.02 ms | 13.68 ms | 1.02 | 3.15 ms | 4.44 | ok |
| stats | `quantile(f, 0.5)` | 7.29 ms | 6.74 ms | 1.08 | 3.16 ms | 2.31 | ok |
| stats | `median(f)` | 7.32 ms | 6.74 ms | 1.09 | 3.16 ms | 2.31 | ok |
| stats | `median(f2, axis=0)` | 8.15 ms | 5.69 ms | 1.43 | 632.9 µs | 12.87 | ok |
| stats | `median(f2, axis=1)` | 7.58 ms | 5.38 ms | 1.41 | 616.4 µs | 12.30 | ok |
| stats | `nanquantile(fn, q5)` | 11.49 ms | 12.12 ms | **0.95** | 3.85 ms | 2.99 | ok |
| stats | `nanmedian(fn)` | 8.03 ms | 6.76 ms | 1.19 | 3.86 ms | 2.08 | ok |
| stats | `cov(1000 x 1000)` | 2.16 ms | 1.77 ms | 1.22 | 1.51 ms | 1.44 | ok |
| stats | `cov(20 x 1e5)` | 4.01 ms | 2.44 ms | 1.64 | 2.15 ms | 1.87 | ok |
| stats | `corrcoef(1000 x 1000)` | 3.02 ms | 2.03 ms | 1.48 | 1.55 ms | 1.94 | ok |
| stats | `bincount(iv)` | 1.01 ms | 350.9 µs | 2.87 | 84.3 µs | 11.93 | ok |
| stats | `bincount(iv, weights)` | 1.07 ms | 467.8 µs | 2.28 | 377.6 µs | 2.83 | ok |
| stats | `histogram(f, 100)` | 4.02 ms | 1.64 ms | 2.45 | 220.6 µs | 18.22 | ok |
| stats | `histogram(f, 100, range)` | 4.26 ms | 1.14 ms | 3.73 | 162.8 µs | 26.18 | ok |
| sort | `sort(f)` | 27.22 ms | 14.06 ms | 1.94 | 3.97 ms | 6.85 | ok |
| sort | `sort(f, stable=False)` | 27.24 ms | 14.05 ms | 1.94 | 3.97 ms | 6.86 | ok |
| sort | `sort(half zeros)` | 15.44 ms | 14.95 ms | 1.03 | 3.38 ms | 4.57 | ok |
| sort | `sort(i)` | 9.80 ms | 2.46 ms | 3.98 | 1.03 ms | 9.51 | ok |
| sort | `sort(i wide)` | 20.96 ms | 6.33 ms | 3.31 | 2.43 ms | 8.64 | ok |
| sort | `sort(f2, axis=1)` | 13.82 ms | 6.30 ms | 2.20 | 704.4 µs | 19.62 | ok |
| sort | `sort(f2, axis=0)` | 14.41 ms | 6.91 ms | 2.09 | 950.4 µs | 15.16 | ok |
| sort | `sort(f, descending)` | 27.49 ms | 14.22 ms | 1.93 | 3.99 ms | 6.89 | ok |
| sort | `argsort(f)` | 68.75 ms | 10.26 ms | 6.70 | 4.36 ms | 15.76 | ok |
| sort | `argsort(f, stable=False)` | 68.67 ms | 10.26 ms | 6.69 | 4.39 ms | 15.64 | ok |
| sort | `argsort(i)` | 36.92 ms | 2.78 ms | 13.29 | 1.25 ms | 29.44 | ok |
| sort | `argsort(i, stable=False)` | 36.93 ms | 2.77 ms | 13.34 | 1.26 ms | 29.35 | ok |
| sort | `argsort(f2, axis=1)` | 31.68 ms | 8.34 ms | 3.80 | 891.7 µs | 35.53 | ok |
| sort | `sort(fm) (1e4)` | 183.2 µs | 85.5 µs | 2.14 | 85.5 µs | 2.14 | ok |
| sort | `argsort(fm) (1e4)` | 182.8 µs | 88.9 µs | 2.06 | 88.9 µs | 2.06 | ok |
| unique | `unique_values(iv)` | 5.47 ms | 2.83 ms | 1.93 | 1.04 ms | 5.25 | ok |
| unique | `unique_counts(iv)` | 9.31 ms | 4.24 ms | 2.19 | 2.49 ms | 3.73 | ok |
| unique | `unique_inverse(iv)` | 39.50 ms | 4.24 ms | 9.32 | 2.48 ms | 15.90 | ok |
| unique | `unique_all(iv)` | 81.25 ms | 4.24 ms | 19.17 | 2.49 ms | 32.67 | ok |
| unique | `unique_values(f)` | 27.81 ms | 14.24 ms | 1.95 | 4.18 ms | 6.65 | ok |
| search | `searchsorted(srt, keys)` | 9.33 ms | 4.45 ms | 2.10 | 769.4 µs | 12.13 | ok |
| search | `searchsorted(srt, f)` | 53.42 ms | 55.58 ms | **0.96** | 6.39 ms | 8.36 | ok |
| search | `nonzero(b)` | 343.1 µs | 324.7 µs | 1.06 | 48.4 µs | 7.09 | ok |
| search | `nonzero(f2 > 0.5)` | 4.58 ms | 463.4 µs | 9.87 | 81.2 µs | 56.36 | ok |
| search | `take(f, idx)` | 149.5 µs | 68.8 µs | 2.17 | 11.6 µs | 12.94 | ok |
| search | `take(f2, rows, axis=0)` | 111.5 µs | 122.4 µs | **0.91** | 32.3 µs | 3.46 | ok |
| search | `take_along_axis(f2, idx2, axis=1)` | 2.08 ms | 441.0 µs | 4.72 | 74.9 µs | 27.81 | ok |
| index | `f[1000:2000] (view)` | 43 ns | 89 ns | **0.48** | 86 ns | 0.50 | ok |
| index | `f[::2] (view)` | 43 ns | 88 ns | **0.49** | 88 ns | 0.49 | ok |
| index | `f2[5] (row view)` | 42 ns | 71 ns | **0.60** | 71 ns | 0.60 | ok |
| index | `f2[:, 5] (column)` | 48 ns | 103 ns | **0.47** | 102 ns | 0.47 | ok |
| index | `f2[10, 20] (scalar)` | 33 ns | 25 ns | 1.34 | 25 ns | 1.34 | ok |
| index | `f[idx] (gather 1e5)` | 73.3 µs | 68.5 µs | 1.07 | 11.5 µs | 6.36 | ok |
| index | `f[idxn] (gather 1e6)` | 728.8 µs | 692.1 µs | 1.05 | 83.3 µs | 8.75 | ok |
| index | `f[b] (mask)` | 3.38 ms | 338.6 µs | 9.99 | 48.1 µs | 70.29 | ok |
| index | `f2[rows] (fancy rows)` | 158.1 µs | 122.5 µs | 1.29 | 32.0 µs | 4.94 | ok |
| index | `f2[f2 > 0.5] (2-D mask)` | 3.50 ms | 417.7 µs | 8.39 | 72.5 µs | 48.34 | ok |
| index | `s3[1] (tiny scalar)` | 27 ns | 22 ns | 1.23 | 23 ns | 1.21 | ok |
| assign | `gz[idx] = 1.0 (scatter 1e5)` | 95.6 µs | 105.4 µs | **0.91** | 80.3 µs | 1.19 | stmt |
| assign | `gz[idxn] = 1.0 (scatter 1e6)` | 960.9 µs | 1.03 ms | **0.94** | 807.7 µs | 1.19 | stmt |
| assign | `fw[b] = 0.0 (mask)` | 3.00 ms | 406.8 µs | 7.38 | 55.8 µs | 53.81 | stmt |
| assign | `fw[:] = 1.0 (fill)` | 62.9 µs | 248.0 µs | **0.25** | 35.8 µs | 1.76 | stmt |
| assign | `fw[:] = g (copy)` | 93.6 µs | 93.5 µs | 1.00 | 34.1 µs | 2.74 | stmt |
| assign | `fw[::2] = 1.0 (strided fill)` | 122.6 µs | 124.5 µs | **0.99** | 21.4 µs | 5.73 | stmt |
| assign | `x += g (in place)` | 164.0 µs | 163.9 µs | 1.00 | 31.1 µs | 5.28 | ok |
| assign | `x[1:] += x[:-1] (overlap)` | 237.5 µs | 299.7 µs | **0.79** | 103.5 µs | 2.29 | stmt |
| assign | `x *= 1.0000001` | 111.7 µs | 111.2 µs | 1.00 | 30.6 µs | 3.65 | ok |
| matmul | `matmul 512` | 321.3 µs | 321.7 µs | **1.00** | 321.5 µs | 1.00 | ok |
| matmul | `matmul 1000` | 2.01 ms | 2.01 ms | 1.00 | 2.01 ms | 1.00 | ok |
| matmul | `matmul(f2, v) (gemv)` | 13.6 µs | 13.5 µs | 1.01 | 13.6 µs | 1.00 | ok |
| matmul | `matmul(v, v) (dot)` | 650 ns | 275 ns | 2.36 | 269 ns | 2.41 | ok |
| matmul | `matmul batch 1000 x 8x8` | 70.5 µs | 71.7 µs | **0.98** | 13.9 µs | 5.09 | ok |
| matmul | `matmul int 200` | 2.56 ms | 1.56 ms | 1.64 | 215.5 µs | 11.87 | ok |
| matmul | `matmul(ft, f2) (transposed)` | 1.31 ms | 1.30 ms | 1.01 | 1.29 ms | 1.01 | ok |
| matmul | `vecdot(f2, g2)` | 131.9 µs | 200.1 µs | **0.66** | 28.1 µs | 4.70 | ok |
| matmul | `tensordot(t3, t3, 2)` | 377.3 µs | 373.4 µs | 1.01 | 373.5 µs | 1.01 | ok |
| size 1e3 | `a + b (1e3)` | 285 ns | 286 ns | **1.00** | 290 ns | 0.98 | ok |
| size 1e3 | `a * 2.0 (1e3)` | 359 ns | 286 ns | 1.25 | 291 ns | 1.23 | ok |
| size 1e3 | `exp(a) (1e3)` | 1.7 µs | 1.2 µs | 1.35 | 1.2 µs | 1.35 | ok |
| size 1e3 | `sin(a) (1e3)` | 2.5 µs | 949 ns | 2.65 | 953 ns | 2.64 | ok |
| size 1e3 | `sum(a) (1e3)` | 663 ns | 188 ns | 3.53 | 192 ns | 3.45 | ok |
| size 1e3 | `max(a) (1e3)` | 585 ns | 140 ns | 4.17 | 142 ns | 4.11 | ok |
| size 1e3 | `var(a) (1e3)` | 3.4 µs | 466 ns | 7.35 | 477 ns | 7.18 | ok |
| size 1e3 | `cumulative_sum(a) (1e3)` | 2.4 µs | 709 ns | 3.40 | 709 ns | 3.40 | ok |
| size 1e3 | `a < b (1e3)` | 352 ns | 265 ns | 1.33 | 269 ns | 1.31 | ok |
| size 1e3 | `a[a > 1.0] (1e3)` | 2.2 µs | 842 ns | 2.63 | 846 ns | 2.61 | ok |
| size 1e3 | `where(a > 1.0, a, b) (1e3)` | 989 ns | 776 ns | 1.28 | 784 ns | 1.26 | ok |
| size 1e3 | `sort(a) (1e3)` | 14.1 µs | 5.6 µs | 2.52 | 5.6 µs | 2.51 | ok |
| size 1e3 | `argsort(a) (1e3)` | 7.6 µs | 8.9 µs | **0.86** | 8.8 µs | 0.86 | ok |
| size 1e3 | `astype int (1e3)` | 524 ns | 203 ns | 2.58 | 207 ns | 2.53 | ok |
| size 1e3 | `copy (1e3)` | 208 ns | 208 ns | **1.00** | 209 ns | 0.99 | ok |
| size 1e3 | `i + j (1e3)` | 284 ns | 293 ns | **0.97** | 292 ns | 0.97 | ok |
| size 1e3 | `sum(i) (1e3)` | 627 ns | 142 ns | 4.41 | 146 ns | 4.30 | ok |
| size 1e3 | `sort(i) (1e3)` | 12.3 µs | 7.1 µs | 1.72 | 7.2 µs | 1.72 | ok |
| size 1e3 | `median(a) (1e3)` | 5.5 µs | 4.6 µs | 1.19 | 4.7 µs | 1.19 | ok |
| size 1e3 | `unique_values(i) (1e3)` | 24.8 µs | 7.2 µs | 3.43 | 7.2 µs | 3.43 | ok |
| size 1e5 | `a + b (1e5)` | 20.3 µs | 20.1 µs | 1.01 | 5.9 µs | 3.42 | ok |
| size 1e5 | `a * 2.0 (1e5)` | 9.8 µs | 9.7 µs | 1.02 | 5.7 µs | 1.73 | ok |
| size 1e5 | `exp(a) (1e5)` | 149.0 µs | 107.0 µs | 1.39 | 16.2 µs | 9.17 | ok |
| size 1e5 | `sin(a) (1e5)` | 364.3 µs | 80.2 µs | 4.54 | 12.8 µs | 28.54 | ok |
| size 1e5 | `sum(a) (1e5)` | 12.2 µs | 11.4 µs | 1.07 | 5.1 µs | 2.38 | ok |
| size 1e5 | `max(a) (1e5)` | 8.1 µs | 7.2 µs | 1.12 | 4.3 µs | 1.88 | ok |
| size 1e5 | `var(a) (1e5)` | 48.2 µs | 31.9 µs | 1.51 | 11.0 µs | 4.39 | ok |
| size 1e5 | `cumulative_sum(a) (1e5)` | 212.3 µs | 38.4 µs | 5.53 | 12.2 µs | 17.38 | ok |
| size 1e5 | `a < b (1e5)` | 20.0 µs | 13.4 µs | 1.50 | 5.1 µs | 3.90 | ok |
| size 1e5 | `a[a > 1.0] (1e5)` | 258.8 µs | 41.3 µs | 6.27 | 16.4 µs | 15.81 | ok |
| size 1e5 | `where(a > 1.0, a, b) (1e5)` | 44.0 µs | 36.7 µs | 1.20 | 11.8 µs | 3.74 | ok |
| size 1e5 | `sort(a) (1e5)` | 2.27 ms | 1.14 ms | 2.00 | 1.13 ms | 2.00 | ok |
| size 1e5 | `argsort(a) (1e5)` | 5.10 ms | 843.3 µs | 6.04 | 842.5 µs | 6.05 | ok |
| size 1e5 | `astype int (1e5)` | 27.4 µs | 13.2 µs | 2.08 | 5.4 µs | 5.09 | ok |
| size 1e5 | `copy (1e5)` | 9.5 µs | 9.6 µs | **1.00** | 5.7 µs | 1.67 | ok |
| size 1e5 | `i + j (1e5)` | 14.3 µs | 14.3 µs | **1.00** | 5.9 µs | 2.42 | ok |
| size 1e5 | `sum(i) (1e5)` | 7.9 µs | 7.3 µs | 1.09 | 4.4 µs | 1.81 | ok |
| size 1e5 | `sort(i) (1e5)` | 1.06 ms | 236.8 µs | 4.49 | 237.0 µs | 4.49 | ok |
| size 1e5 | `median(a) (1e5)` | 583.5 µs | 569.8 µs | 1.02 | 588.3 µs | 0.99 | ok |
| size 1e5 | `unique_values(i) (1e5)` | 873.1 µs | 230.9 µs | 3.78 | 230.4 µs | 3.79 | ok |
| size 1e7 | `a + b (1e7)` | 2.02 ms | 2.01 ms | 1.00 | 1.15 ms | 1.76 | ok |
| size 1e7 | `a * 2.0 (1e7)` | 1.31 ms | 1.31 ms | **1.00** | 717.1 µs | 1.83 | ok |
| size 1e7 | `exp(a) (1e7)` | 14.85 ms | 10.68 ms | 1.39 | 1.23 ms | 12.04 | ok |
| size 1e7 | `sin(a) (1e7)` | 37.97 ms | 8.01 ms | 4.74 | 1.07 ms | 35.47 | ok |
| size 1e7 | `sum(a) (1e7)` | 1.19 ms | 1.20 ms | **0.99** | 478.4 µs | 2.48 | ok |
| size 1e7 | `max(a) (1e7)` | 881.6 µs | 858.5 µs | 1.03 | 373.3 µs | 2.36 | ok |
| size 1e7 | `var(a) (1e7)` | 5.01 ms | 3.34 ms | 1.50 | 963.9 µs | 5.19 | ok |
| size 1e7 | `cumulative_sum(a) (1e7)` | 22.34 ms | 4.23 ms | 5.27 | 1.61 ms | 13.85 | ok |
| size 1e7 | `a < b (1e7)` | 2.01 ms | 1.36 ms | 1.47 | 800.9 µs | 2.51 | ok |
| size 1e7 | `a[a > 1.0] (1e7)` | 36.31 ms | 4.39 ms | 8.26 | 1.17 ms | 31.12 | ok |
| size 1e7 | `where(a > 1.0, a, b) (1e7)` | 4.57 ms | 3.95 ms | 1.16 | 1.66 ms | 2.76 | ok |
| size 1e7 | `sort(a) (1e7)` | 317.92 ms | 170.68 ms | 1.86 | 49.84 ms | 6.38 | ok |
| size 1e7 | `argsort(a) (1e7)` | 906.50 ms | 85.63 ms | 10.59 | 35.28 ms | 25.69 | ok |
| size 1e7 | `astype int (1e7)` | 2.70 ms | 1.31 ms | 2.05 | 718.6 µs | 3.76 | ok |
| size 1e7 | `copy (1e7)` | 1.18 ms | 1.18 ms | 1.00 | 611.7 µs | 1.94 | ok |
| size 1e7 | `i + j (1e7)` | 1.96 ms | 1.97 ms | **1.00** | 1.15 ms | 1.70 | ok |
| size 1e7 | `sum(i) (1e7)` | 866.5 µs | 866.3 µs | 1.00 | 372.2 µs | 2.33 | ok |
| size 1e7 | `sort(i) (1e7)` | 97.55 ms | 24.53 ms | 3.98 | 9.44 ms | 10.34 | ok |
| size 1e7 | `median(a) (1e7)` | 79.15 ms | 76.75 ms | 1.03 | 40.41 ms | 1.96 | ok |
| size 1e7 | `unique_values(i) (1e7)` | 83.88 ms | 23.68 ms | 3.54 | 8.56 ms | 9.80 | ok |
| tiny | `zeros(3)` | 60 ns | 100 ns | **0.61** | 100 ns | 0.60 | ok |
| tiny | `empty(3)` | 60 ns | 98 ns | **0.61** | 98 ns | 0.61 | ok |
| tiny | `asarray(s3) (no-op)` | 12 ns | 20 ns | **0.63** | 20 ns | 0.63 | ok |
| tiny | `s3 + t3s` | 150 ns | 177 ns | **0.85** | 181 ns | 0.83 | ok |
| tiny | `s3 * 2.0` | 244 ns | 205 ns | 1.19 | 209 ns | 1.16 | ok |
| tiny | `i3 + 1` | 278 ns | 216 ns | 1.29 | 217 ns | 1.28 | ok |
| tiny | `sqrt(s3)` | 123 ns | 147 ns | **0.83** | 148 ns | 0.83 | ok |
| tiny | `exp(s3)` | 126 ns | 151 ns | **0.84** | 153 ns | 0.83 | ok |
| tiny | `s3 < t3s` | 149 ns | 177 ns | **0.84** | 180 ns | 0.83 | ok |
| tiny | `sum(s3)` | 544 ns | 89 ns | 6.10 | 91 ns | 5.95 | ok |
| tiny | `max(s3)` | 526 ns | 89 ns | 5.90 | 91 ns | 5.76 | ok |
| tiny | `mean(s3)` | 862 ns | 88 ns | 9.79 | 91 ns | 9.52 | ok |
| tiny | `sum(sm, axis=0)` | 758 ns | 189 ns | 4.01 | 192 ns | 3.96 | ok |
| tiny | `sort(s3)` | 292 ns | 134 ns | 2.17 | 135 ns | 2.16 | ok |
| tiny | `reshape(sm, (4, 3))` | 263 ns | 142 ns | 1.85 | 142 ns | 1.85 | ok |
| tiny | `sm.T` | 38 ns | 65 ns | **0.58** | 64 ns | 0.59 | ok |
| tiny | `sm[1]` | 42 ns | 71 ns | **0.59** | 71 ns | 0.60 | ok |
| tiny | `sm[1:, ::2]` | 51 ns | 100 ns | **0.51** | 100 ns | 0.51 | ok |
| tiny | `concat([s3, t3s])` | 244 ns | 252 ns | **0.97** | 257 ns | 0.95 | ok |
| tiny | `where(s3 > 1, s3, t3s)` | 506 ns | 406 ns | 1.25 | 412 ns | 1.23 | ok |
| tiny | `matmul(sm.T, sm)` | 417 ns | 311 ns | 1.34 | 311 ns | 1.34 | ok |
| tiny | `result_type(f, i)` | 55 ns | 45 ns | 1.22 | 45 ns | 1.22 | ok |
| tiny | `can_cast(int64, float64)` | 88 ns | 26 ns | 3.40 | 26 ns | 3.40 | ok |
| tiny | `finfo(float64)` | 111 ns | 368 ns | **0.30** | 376 ns | 0.29 | ok |
| tiny | `s3.shape` | 18 ns | 34 ns | **0.52** | 34 ns | 0.52 | ok |
| tiny | `len(s3)` | 7 ns | 17 ns | **0.39** | 17 ns | 0.39 | ok |
| tiny | `float(s3[0])` | 36 ns | 28 ns | 1.32 | 28 ns | 1.32 | ok |
