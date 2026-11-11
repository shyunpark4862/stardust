# 벤치마크 기준선 (M0)

옛 세 패키지를 `legacy/`의 마지막 커밋에서 빌드해 한자리에서 잰 값이다. 행마다 여러 라운드 중 가장 빠른 값.

```
python 3.15.0 arm64 numpy 2.5.4 scipy 1.18.1 pandas 3.1.1 polars 2.1.0
legacy/numpy 3c70608
legacy/linalg de5ebc7
legacy/pandas d8b6b65
```

## array (17줄)

- numpy / legacy: 기하평균 5.37배, 17줄 중 legacy가 빠름 16, 비슷 0, 느림 1

| 행 | legacy | legacy-1 | numpy |
|---|---:|---:|---:|
| f + g | 39.8µs | 148.4µs | 202.7µs |
| f * i + g  (casts) | 154.9µs | 397.1µs | 634.5µs |
| ft + 1.0  (transposed) | 31.0µs | 91.2µs | 128.4µs |
| maximum(f, g) | 39.7µs | 153.7µs | 199.4µs |
| exp(f) | 121.6µs | 1.08ms | 1.51ms |
| sin(f) | 92.1µs | 813.2µs | 3.83ms |
| sum(f) | 23.9µs | 113.2µs | 114.8µs |
| sum(m, axis=1) | 24.9µs | 117.7µs | 120.7µs |
| max(f) | 22.4µs | 69.2µs | 73.8µs |
| var(f) | 57.2µs | 318.0µs | 478.9µs |
| x[idx]  gather | 80.5µs | 733.1µs | 756.6µs |
| x[idx] = 1.0  scatter | 806.6µs | 1.04ms | 953.6µs |
| sort(f) | 4.03ms | 14.16ms | 26.62ms |
| sort(s)  2000 rows | 1.42ms | 12.60ms | 28.21ms |
| int64 matmul 300^3 | 783.9µs | 5.16ms | 8.47ms |
| tiny a + b | 180ns | 174ns | 158ns |
| tiny x[1, 2] | 25ns | 25ns | 33ns |

## linalg (258줄)

- ref / legacy: 기하평균 3.06배, 248줄 중 legacy가 빠름 189, 비슷 49, 느림 10

| 행 | legacy | legacy-1 | ref |
|---|---:|---:|---:|
| solve · (3,3) | 471ns | 474ns | 2.1µs |
| inv · (3,3) | 454ns | 465ns | 1.7µs |
| det · (3,3) | 173ns | 173ns | 1.0µs |
| slogdet · (3,3) | 318ns | 309ns | 1.5µs |
| vecdot rows · (3,3) | 395ns | 385ns | 318ns |
| outer · (3,3) | 329ns | 330ns | 544ns |
| trace · (3,3) | 54ns | 53ns | 558ns |
| diagonal · (3,3) | 265ns | 266ns | 472ns |
| matrix_power 3 · (3,3) | 441ns | 436ns | 1.1µs |
| vector_norm rows · (3,3) | 466ns | 481ns | 1.2µs |
| matrix_norm fro · (3,3) | 78ns | 77ns | 904ns |
| matrix_norm 1 · (3,3) | 68ns | 66ns | 1.4µs |
| matrix_norm 2 · (3,3) | 1.6µs | 1.6µs | 6.4µs |
| solve_triangular · (3,3) | 376ns | 370ns | 4.6µs |
| cholesky · (3,3) | 322ns | 321ns | 1.6µs |
| cholesky_solve · (3,3) | 373ns | 372ns | 3.8µs |
| lu · (3,3) | 1.1µs | 1.1µs | 1.6µs |
| lu_solve · (3,3) | 508ns | 513ns | 3.9µs |
| qr · (3,3) | 1.2µs | 1.3µs | 6.1µs |
| qr complete · (3,3) | 1.2µs | 1.2µs | 6.2µs |
| qr_pivoted · (3,3) | 1.6µs | 1.6µs | 2.3µs |
| lstsq · (3,3) | 2.6µs | 2.6µs | 5.1µs |
| eigh · (3,3) | 1.6µs | 1.5µs | 2.7µs |
| eigvalsh · (3,3) | 788ns | 789ns | 2.0µs |
| eigh(a, b) · (3,3) | 1.8µs | 1.8µs | 7.0µs |
| svd · (3,3) | 3.5µs | 3.6µs | 5.0µs |
| svdvals · (3,3) | 1.8µs | 1.7µs | 3.7µs |
| matrix_rank · (3,3) | 1.6µs | 1.6µs | 6.5µs |
| pinv · (3,3) | 3.0µs | 2.9µs | 10.5µs |
| cond · (3,3) | 1.6µs | 1.6µs | 4.9µs |
| eig · (3,3) | 5.0µs | 4.8µs | 6.6µs |
| eigvals · (3,3) | 2.0µs | 2.0µs | 3.8µs |
| schur · (3,3) | 2.7µs | 2.8µs | 7.8µs |
| hessenberg · (3,3) | 1.4µs | 1.4µs | 8.8µs |
| ldl · (3,3) | 1.3µs | 1.3µs | 18.3µs |
| solve_tridiagonal · (3,3) | 423ns | 415ns | 7.6µs |
| solve_banded 2,2 · (3,3) | 1.3µs | 1.3µs | 9.9µs |
| solve_banded 1,1 · (3,3) | 384ns | 382ns | 7.6µs |
| solve_sylvester · (3,3) | 4.3µs | 4.4µs | 21.3µs |
| kron with 3x3 · (3,3) | 382ns | 373ns | 4.9µs |
| cross rows · (3,3) | 442ns | 419ns | 6.6µs |
| expm · (3,3) | 1.1µs | 1.1µs | 1.4µs |
| solve · (32,32) | 4.5µs | 4.4µs | 7.5µs |
| inv · (32,32) | 6.8µs | 6.4µs | 10.2µs |
| det · (32,32) | 3.7µs | 3.5µs | 4.8µs |
| slogdet · (32,32) | 3.9µs | 3.8µs | 5.4µs |
| vecdot rows · (32,32) | 712ns | 707ns | 504ns |
| outer · (32,32) | 435ns | 425ns | 966ns |
| trace · (32,32) | 62ns | 61ns | 567ns |
| diagonal · (32,32) | 274ns | 274ns | 491ns |
| matrix_power 3 · (32,32) | 1.8µs | 1.8µs | 2.6µs |
| vector_norm rows · (32,32) | 1.2µs | 1.2µs | 1.7µs |
| matrix_norm fro · (32,32) | 498ns | 492ns | 1.2µs |
| matrix_norm 1 · (32,32) | 429ns | 426ns | 1.9µs |
| matrix_norm 2 · (32,32) | 23.4µs | 23.5µs | 30.0µs |
| solve_triangular · (32,32) | 839ns | 840ns | 5.0µs |
| cholesky · (32,32) | 1.4µs | 1.4µs | 3.0µs |
| cholesky_solve · (32,32) | 979ns | 978ns | 4.7µs |
| lu · (32,32) | 5.7µs | 5.6µs | 6.1µs |
| lu_solve · (32,32) | 1.1µs | 1.1µs | 4.3µs |
| qr · (32,32) | 8.8µs | 8.9µs | 16.5µs |
| qr complete · (32,32) | 8.9µs | 8.8µs | 16.3µs |
| qr_pivoted · (32,32) | 11.1µs | 11.0µs | 12.0µs |
| lstsq · (32,32) | 61.7µs | 60.7µs | 64.7µs |
| eigh · (32,32) | 45.3µs | 42.7µs | 45.3µs |
| eigvalsh · (32,32) | 16.2µs | 16.0µs | 18.0µs |
| eigh(a, b) · (32,32) | 53.2µs | 51.9µs | 59.0µs |
| svd · (32,32) | 69.6µs | 68.6µs | 71.6µs |
| svdvals · (32,32) | 23.5µs | 23.5µs | 26.9µs |
| matrix_rank · (32,32) | 23.3µs | 23.1µs | 29.7µs |
| pinv · (32,32) | 72.4µs | 69.0µs | 79.6µs |
| cond · (32,32) | 23.4µs | 23.3µs | 28.3µs |
| eig · (32,32) | 187.4µs | 185.5µs | 191.3µs |
| eigvals · (32,32) | 66.9µs | 66.5µs | 69.0µs |
| schur · (32,32) | 96.8µs | 95.4µs | 102.2µs |
| hessenberg · (32,32) | 12.7µs | 12.4µs | 21.0µs |
| ldl · (32,32) | 3.8µs | 3.7µs | 45.1µs |
| solve_tridiagonal · (32,32) | 703ns | 675ns | 7.8µs |
| solve_banded 2,2 · (32,32) | 2.2µs | 2.1µs | 10.7µs |
| solve_banded 1,1 · (32,32) | 676ns | 672ns | 7.9µs |
| solve_sylvester · (32,32) | 222.6µs | 288.3µs | 321.5µs |
| kron with 3x3 · (32,32) | 2.6µs | 2.5µs | 21.1µs |
| cross rows · (32,32) | 549ns | 545ns | 6.9µs |
| expm · (32,32) | 19.4µs | 19.2µs | 21.5µs |
| solve · (300,300) | 324.5µs | 323.4µs | 323.3µs |
| inv · (300,300) | 616.5µs | 613.5µs | 644.5µs |
| det · (300,300) | 290.2µs | 289.8µs | 298.6µs |
| slogdet · (300,300) | 299.4µs | 291.7µs | 298.9µs |
| vecdot rows · (300,300) | 6.1µs | 19.6µs | 11.6µs |
| outer · (300,300) | 6.4µs | 6.5µs | 29.7µs |
| trace · (300,300) | 154ns | 153ns | 599ns |
| diagonal · (300,300) | 308ns | 310ns | 596ns |
| matrix_power 3 · (300,300) | 219.0µs | 217.5µs | 162.8µs |
| vector_norm rows · (300,300) | 9.3µs | 36.7µs | 24.2µs |
| matrix_norm fro · (300,300) | 8.7µs | 38.0µs | 22.1µs |
| matrix_norm 1 · (300,300) | 8.3µs | 32.1µs | 23.8µs |
| matrix_norm 2 · (300,300) | 2.77ms | 2.67ms | 2.90ms |
| solve_triangular · (300,300) | 35.3µs | 35.1µs | 30.2µs |
| cholesky · (300,300) | 111.3µs | 108.8µs | 120.2µs |
| cholesky_solve · (300,300) | 49.6µs | 48.4µs | 67.0µs |
| lu · (300,300) | 339.0µs | 341.1µs | 308.2µs |
| lu_solve · (300,300) | 40.7µs | 41.6µs | 35.9µs |
| qr · (300,300) | 1.28ms | 1.28ms | 1.38ms |
| qr complete · (300,300) | 1.30ms | 1.29ms | 1.38ms |
| qr_pivoted · (300,300) | 1.64ms | 1.63ms | 1.62ms |
| lstsq · (300,300) | 4.32ms | 4.25ms | 4.32ms |
| eigh · (300,300) | 3.53ms | 3.54ms | 3.53ms |
| eigvalsh · (300,300) | 2.11ms | 2.07ms | 2.10ms |
| eigh(a, b) · (300,300) | 4.17ms | 4.18ms | 4.29ms |
| svd · (300,300) | 5.29ms | 5.26ms | 5.35ms |
| svdvals · (300,300) | 2.68ms | 2.73ms | 2.77ms |
| matrix_rank · (300,300) | 2.75ms | 2.67ms | 2.78ms |
| pinv · (300,300) | 5.49ms | 5.41ms | 5.51ms |
| cond · (300,300) | 2.69ms | 2.66ms | 2.78ms |
| eig · (300,300) | 23.83ms | 23.13ms | 24.45ms |
| eigvals · (300,300) | 14.12ms | 13.77ms | 14.30ms |
| schur · (300,300) | 16.40ms | 16.07ms | 16.82ms |
| hessenberg · (300,300) | 2.24ms | 2.23ms | 2.20ms |
| ldl · (300,300) | 503.4µs | 504.8µs | 932.5µs |
| solve_tridiagonal · (300,300) | 2.9µs | 2.7µs | 10.5µs |
| solve_banded 2,2 · (300,300) | 11.0µs | 11.0µs | 20.6µs |
| solve_banded 1,1 · (300,300) | 2.8µs | 2.8µs | 10.7µs |
| solve_sylvester · (300,300) | 24.34ms | 39.22ms | 43.10ms |
| kron with 3x3 · (300,300) | 171.9µs | 170.8µs | 1.26ms |
| cross rows · (300,300) | 1.6µs | 1.6µs | 9.5µs |
| expm · (300,300) | 1.97ms | 2.04ms | 2.54ms |
| solve · (1000,1000) | 3.09ms | 3.53ms | 4.29ms |
| inv · (1000,1000) | 6.96ms | 7.45ms | 8.74ms |
| det · (1000,1000) | 2.76ms | 3.25ms | 3.18ms |
| slogdet · (1000,1000) | 2.78ms | 3.26ms | 3.10ms |
| vecdot rows · (1000,1000) | 26.7µs | 173.9µs | 134.8µs |
| outer · (1000,1000) | 67.5µs | 67.4µs | 329.5µs |
| trace · (1000,1000) | 433ns | 429ns | 900ns |
| diagonal · (1000,1000) | 496ns | 499ns | 839ns |
| matrix_power 3 · (1000,1000) | 4.81ms | 4.70ms | 4.21ms |
| vector_norm rows · (1000,1000) | 52.0µs | 398.6µs | 230.4µs |
| matrix_norm fro · (1000,1000) | 65.2µs | 412.4µs | 246.0µs |
| matrix_norm 1 · (1000,1000) | 58.1µs | 381.5µs | 266.6µs |
| matrix_norm 2 · (1000,1000) | 34.05ms | 34.71ms | 36.59ms |
| solve_triangular · (1000,1000) | 254.4µs | 358.9µs | 217.8µs |
| cholesky · (1000,1000) | 1.43ms | 1.42ms | 1.99ms |
| cholesky_solve · (1000,1000) | 346.3µs | 462.4µs | 840.2µs |
| lu · (1000,1000) | 3.35ms | 3.66ms | 3.88ms |
| lu_solve · (1000,1000) | 401.4µs | 400.7µs | 266.4µs |
| qr · (1000,1000) | 16.39ms | 16.28ms | 18.97ms |
| qr complete · (1000,1000) | 16.38ms | 16.20ms | 18.88ms |
| qr_pivoted · (1000,1000) | 21.19ms | 21.99ms | 22.41ms |
| lstsq · (1000,1000) | 47.13ms | 48.36ms | 48.62ms |
| eigh · (1000,1000) | 49.72ms | 47.84ms | 49.63ms |
| eigvalsh · (1000,1000) | 30.51ms | 29.77ms | 30.11ms |
| eigh(a, b) · (1000,1000) | 58.02ms | 58.35ms | 63.34ms |
| svd · (1000,1000) | 64.37ms | 65.28ms | 67.87ms |
| svdvals · (1000,1000) | 34.29ms | 35.41ms | 36.49ms |
| matrix_rank · (1000,1000) | 33.97ms | 35.27ms | 36.33ms |
| pinv · (1000,1000) | 66.65ms | 66.78ms | 67.34ms |
| cond · (1000,1000) | 33.22ms | 34.43ms | 35.48ms |
| eig · (1000,1000) | 309.36ms | 302.61ms | 307.94ms |
| eigvals · (1000,1000) | 174.55ms | 172.68ms | 172.95ms |
| schur · (1000,1000) | 211.66ms | 202.23ms | 214.07ms |
| hessenberg · (1000,1000) | 31.79ms | 31.34ms | 32.79ms |
| ldl · (1000,1000) | 6.07ms | 5.95ms | 8.36ms |
| solve_tridiagonal · (1000,1000) | 8.4µs | 7.9µs | 19.9µs |
| solve_banded 2,2 · (1000,1000) | 34.5µs | 34.6µs | 45.4µs |
| solve_banded 1,1 · (1000,1000) | 8.4µs | 7.9µs | 18.1µs |
| solve_sylvester · (1000,1000) | 312.56ms | 503.69ms | 979.13ms |
| kron with 3x3 · (1000,1000) | 1.69ms | 1.89ms | 14.05ms |
| cross rows · (1000,1000) | 4.5µs | 4.6µs | 15.8µs |
| expm · (1000,1000) | 40.27ms | 39.86ms | 44.93ms |
| solve · (1000,3,3) | 19.5µs | 140.3µs | 260.8µs |
| inv · (1000,3,3) | 19.3µs | 138.1µs | 230.5µs |
| det · (1000,3,3) | 14.3µs | 100.2µs | 134.6µs |
| slogdet · (1000,3,3) | 16.6µs | 116.6µs | 127.9µs |
| vecdot rows · (1000,3,3) | 5.6µs | 21.4µs | 11.6µs |
| outer · (1000,3,3) | 6.6µs | 6.6µs | 18.4µs |
| trace · (1000,3,3) | 4.1µs | 5.5µs | 5.9µs |
| diagonal · (1000,3,3) | 3.6µs | 3.6µs | 2.6µs |
| matrix_power 3 · (1000,3,3) | 14.9µs | 96.6µs | 78.6µs |
| vector_norm rows · (1000,3,3) | 8.4µs | 41.4µs | 19.0µs |
| matrix_norm fro · (1000,3,3) | 11.7µs | 27.5µs | 9.0µs |
| matrix_norm 1 · (1000,3,3) | 6.6µs | 12.9µs | 39.0µs |
| matrix_norm 2 · (1000,3,3) | 156.7µs | 1.02ms | 1.31ms |
| solve_triangular · (1000,3,3) | 8.6µs | 39.4µs | 3.41ms |
| cholesky · (1000,3,3) | 8.0µs | 30.4µs | 55.3µs |
| cholesky_solve · (1000,3,3) | 11.9µs | 45.8µs | 2.64ms |
| lu · (1000,3,3) | 18.4µs | 115.2µs | 111.2µs |
| lu_solve · (1000,3,3) | 14.6µs | 69.6µs | - |
| qr · (1000,3,3) | 43.5µs | 311.2µs | 400.1µs |
| qr complete · (1000,3,3) | 44.9µs | 303.2µs | 399.3µs |
| qr_pivoted · (1000,3,3) | 57.0µs | 409.5µs | 406.3µs |
| lstsq · (1000,3,3) | 221.1µs | 1.51ms | 1.50ms |
| eigh · (1000,3,3) | 96.1µs | 758.3µs | 806.6µs |
| eigvalsh · (1000,3,3) | 58.8µs | 401.2µs | 431.9µs |
| eigh(a, b) · (1000,3,3) | 125.3µs | 1.01ms | 6.04ms |
| svd · (1000,3,3) | 283.2µs | 1.81ms | 1.87ms |
| svdvals · (1000,3,3) | 159.6µs | 1.01ms | 1.28ms |
| matrix_rank · (1000,3,3) | 158.7µs | 1.02ms | 1.33ms |
| pinv · (1000,3,3) | 286.5µs | 1.87ms | 1.97ms |
| cond · (1000,3,3) | 157.0µs | 1.02ms | 1.31ms |
| eig · (1000,3,3) | 539.4µs | 2.97ms | 3.02ms |
| eigvals · (1000,3,3) | 200.3µs | 1.30ms | 1.33ms |
| schur · (1000,3,3) | 224.9µs | 1.46ms | 7.08ms |
| hessenberg · (1000,3,3) | 79.5µs | 380.0µs | 7.96ms |
| ldl · (1000,3,3) | 21.9µs | 123.4µs | - |
| solve_tridiagonal · (1000,3,3) | 7.5µs | 37.4µs | - |
| solve_banded 2,2 · (1000,3,3) | 926.0µs | 842.8µs | - |
| solve_banded 1,1 · (1000,3,3) | 9.3µs | 37.0µs | - |
| solve_sylvester · (1000,3,3) | 658.4µs | 4.11ms | 20.81ms |
| kron with 3x3 · (1000,3,3) | 8.7µs | 39.3µs | 155.6µs |
| cross rows · (1000,3,3) | 8.0µs | 15.3µs | 24.5µs |
| expm · (1000,3,3) | 114.2µs | 891.2µs | 860.6µs |
| solve · (100,32,32) | 142.3µs | 417.1µs | 556.7µs |
| inv · (100,32,32) | 358.7µs | 623.3µs | 824.6µs |
| det · (100,32,32) | 136.7µs | 364.9µs | 378.5µs |
| slogdet · (100,32,32) | 143.4µs | 377.5µs | 377.7µs |
| vecdot rows · (100,32,32) | 8.8µs | 38.3µs | 18.9µs |
| outer · (100,32,32) | 5.1µs | 11.9µs | 41.7µs |
| trace · (100,32,32) | 2.5µs | 2.5µs | 2.7µs |
| diagonal · (100,32,32) | 2.4µs | 2.3µs | 2.8µs |
| matrix_power 3 · (100,32,32) | 35.7µs | 139.0µs | 78.7µs |
| vector_norm rows · (100,32,32) | 14.6µs | 73.7µs | 42.6µs |
| matrix_norm fro · (100,32,32) | 9.5µs | 44.3µs | 25.5µs |
| matrix_norm 1 · (100,32,32) | 8.6µs | 38.6µs | 43.8µs |
| matrix_norm 2 · (100,32,32) | 262.5µs | 2.26ms | 2.44ms |
| solve_triangular · (100,32,32) | 11.1µs | 49.3µs | 391.7µs |
| cholesky · (100,32,32) | 19.2µs | 110.4µs | 144.6µs |
| cholesky_solve · (100,32,32) | 13.7µs | 61.4µs | 361.0µs |
| lu · (100,32,32) | 133.1µs | 468.1µs | 409.7µs |
| lu_solve · (100,32,32) | 14.2µs | 58.3µs | - |
| qr · (100,32,32) | 113.4µs | 804.9µs | 972.9µs |
| qr complete · (100,32,32) | 114.2µs | 805.0µs | 979.1µs |
| qr_pivoted · (100,32,32) | 138.8µs | 1.01ms | 989.1µs |
| lstsq · (100,32,32) | 701.9µs | 6.07ms | 6.12ms |
| eigh · (100,32,32) | 511.1µs | 4.33ms | 4.40ms |
| eigvalsh · (100,32,32) | 191.3µs | 1.59ms | 1.68ms |
| eigh(a, b) · (100,32,32) | 633.3µs | 5.08ms | 6.27ms |
| svd · (100,32,32) | 846.9µs | 6.76ms | 7.44ms |
| svdvals · (100,32,32) | 259.0µs | 2.27ms | 2.54ms |
| matrix_rank · (100,32,32) | 261.9µs | 2.28ms | 2.52ms |
| pinv · (100,32,32) | 817.3µs | 6.91ms | 7.51ms |
| cond · (100,32,32) | 259.7µs | 2.27ms | 2.51ms |
| eig · (100,32,32) | 3.66ms | 19.33ms | 21.04ms |
| eigvals · (100,32,32) | 821.4µs | 6.88ms | 7.04ms |
| schur · (100,32,32) | 1.19ms | 9.66ms | 10.43ms |
| hessenberg · (100,32,32) | 179.8µs | 1.15ms | 2.08ms |
| ldl · (100,32,32) | 44.2µs | 262.5µs | - |
| solve_tridiagonal · (100,32,32) | 7.8µs | 27.2µs | - |
| solve_banded 2,2 · (100,32,32) | 179.8µs | 172.8µs | - |
| solve_banded 1,1 · (100,32,32) | 7.7µs | 26.2µs | - |
| solve_sylvester · (100,32,32) | 4.45ms | 30.78ms | 32.79ms |
| kron with 3x3 · (100,32,32) | 33.4µs | 205.1µs | 1.53ms |
| cross rows · (100,32,32) | 8.4µs | 16.2µs | 42.5µs |
| expm · (100,32,32) | 551.1µs | 1.85ms | 2.16ms |
| vecdot · (1000000,) | 30.0µs | 170.4µs | 35.3µs |
| vector_norm · (1000000,) | 63.9µs | 379.5µs | 238.1µs |
| vector_norm inf · (1000000,) | 46.0µs | 270.8µs | 177.9µs |
| solve_tridiagonal · n=100000 | 826.4µs | 767.3µs | 977.9µs |
| solve_banded 2,2 · n=100000 | 3.32ms | 3.32ms | 3.37ms |
| cross · (1000000,3) | 527.0µs | 3.80ms | 5.55ms |

## frame (49줄)

- pandas / legacy: 기하평균 4.64배, 49줄 중 legacy가 빠름 44, 비슷 4, 느림 1
- polars-1 / legacy: 기하평균 2.24배, 49줄 중 legacy가 빠름 38, 비슷 4, 느림 7
- polars / legacy: 기하평균 1.78배, 49줄 중 legacy가 빠름 29, 비슷 4, 느림 16

| 행 | legacy | legacy-1 | pandas | polars-1 | polars |
|---|---:|---:|---:|---:|---:|
| s + s · 10 | 464ns | 460ns | 10.0µs | 749ns | 743ns |
| s.sum() · 10 | 179ns | 173ns | 3.5µs | 83ns | 82ns |
| df["a"] · 10 | 178ns | 178ns | 3.3µs | 303ns | 307ns |
| df[df["a"] > 0] · 10 | 1.8µs | 1.7µs | 42.7µs | 25.2µs | 33.0µs |
| float + float · 1000 | 706ns | 704ns | 10.4µs | 948ns | 949ns |
| float sum · 1000 | 316ns | 310ns | 3.9µs | 242ns | 241ns |
| mask filter · 1000 | 2.7µs | 2.7µs | 29.7µs | 2.9µs | 2.9µs |
| nunique int, 1k keys · 1000 | 3.1µs | 3.2µs | 9.5µs | 5.0µs | 16.7µs |
| nunique int, n keys · 1000 | 3.3µs | 3.2µs | 9.7µs | 5.0µs | 17.8µs |
| nunique string · 1000 | 9.2µs | 9.2µs | 20.0µs | 5.4µs | 5.4µs |
| groupby int 1k sum · 1000 | 28.8µs | 29.7µs | 95.3µs | 125.8µs | 319.0µs |
| groupby int, n keys, sum · 1000 | 29.5µs | 28.9µs | 97.9µs | 126.9µs | 327.3µs |
| groupby string sum · 1000 | 21.8µs | 21.9µs | 120.1µs | 111.0µs | 438.0µs |
| groupby int + string sum · 1000 | 70.4µs | 69.6µs | 234.2µs | 210.6µs | 628.6µs |
| merge n x n/10 inner · 1000 | 10.2µs | 10.0µs | 224.3µs | 39.1µs | 178.1µs |
| sort int64 · 1000 | 15.5µs | 14.9µs | 37.6µs | 4.7µs | 17.6µs |
| sort string · 1000 | 28.3µs | 29.7µs | 204.3µs | 28.2µs | 42.0µs |
| read_csv floats · 1000 | 30.2µs | 30.6µs | 121.5µs | 178.7µs | 252.5µs |
| read_csv mixed · 1000 | 65.9µs | 65.9µs | 187.4µs | 248.8µs | 324.2µs |
| float + float · 100000 | 15.6µs | 21.1µs | 27.8µs | 21.7µs | 21.0µs |
| float sum · 100000 | 8.8µs | 8.7µs | 25.3µs | 7.8µs | 7.8µs |
| mask filter · 100000 | 102.5µs | 103.0µs | 332.1µs | 33.6µs | 33.6µs |
| nunique int, 1k keys · 100000 | 198.1µs | 196.6µs | 151.7µs | 485.3µs | 315.8µs |
| nunique int, n keys · 100000 | 582.0µs | 580.8µs | 582.3µs | 727.7µs | 405.7µs |
| nunique string · 100000 | 1.44ms | 1.46ms | 1.46ms | 779.1µs | 777.8µs |
| groupby int 1k sum · 100000 | 352.0µs | 353.5µs | 403.9µs | 766.3µs | 781.5µs |
| groupby int, n keys, sum · 100000 | 2.78ms | 2.87ms | 2.80ms | 3.69ms | 1.99ms |
| groupby string sum · 100000 | 2.10ms | 2.13ms | 5.92ms | 2.59ms | 1.70ms |
| groupby int + string sum · 100000 | 6.30ms | 6.42ms | 13.14ms | 10.45ms | 2.92ms |
| merge n x n/10 inner · 100000 | 686.2µs | 930.5µs | 743.6µs | 635.6µs | 575.8µs |
| sort int64 · 100000 | 767.1µs | 1.49ms | 5.92ms | 724.6µs | 393.7µs |
| sort string · 100000 | 2.90ms | 3.57ms | 34.06ms | 7.84ms | 1.59ms |
| read_csv floats · 100000 | 990.1µs | 2.40ms | 2.49ms | 2.34ms | 1.64ms |
| read_csv mixed · 100000 | 1.95ms | 6.43ms | 9.02ms | 5.85ms | 4.10ms |
| float + float · 1000000 | 44.2µs | 175.5µs | 201.5µs | 514.8µs | 205.5µs |
| float sum · 1000000 | 15.0µs | 89.1µs | 201.0µs | 85.8µs | 85.0µs |
| mask filter · 1000000 | 137.9µs | 1.01ms | 4.05ms | 303.0µs | 303.5µs |
| nunique int, 1k keys · 1000000 | 973.5µs | 2.26ms | 1.48ms | 4.70ms | 2.45ms |
| nunique int, n keys · 1000000 | 3.35ms | 7.90ms | 8.19ms | 9.85ms | 2.95ms |
| nunique string · 1000000 | 3.45ms | 12.94ms | 19.13ms | 13.95ms | 14.27ms |
| groupby int 1k sum · 1000000 | 1.83ms | 3.77ms | 3.37ms | 7.52ms | 2.80ms |
| groupby int, n keys, sum · 1000000 | 12.43ms | 34.54ms | 31.66ms | 45.73ms | 10.59ms |
| groupby string sum · 1000000 | 8.42ms | 21.11ms | 77.35ms | 35.27ms | 7.23ms |
| groupby int + string sum · 1000000 | 20.04ms | 72.30ms | 165.11ms | 167.41ms | 25.02ms |
| merge n x n/10 inner · 1000000 | 2.43ms | 13.09ms | 7.03ms | 8.11ms | 2.59ms |
| sort int64 · 1000000 | 2.67ms | 14.41ms | 92.16ms | 9.99ms | 3.15ms |
| sort string · 1000000 | 8.56ms | 39.36ms | 555.95ms | 65.00ms | 11.94ms |
| read_csv floats · 1000000 | 4.89ms | 25.25ms | 5.68ms | 23.37ms | 5.19ms |
| read_csv mixed · 1000000 | 12.51ms | 66.23ms | 88.38ms | 57.70ms | 11.87ms |

