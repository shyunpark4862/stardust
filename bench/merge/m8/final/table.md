## 배열, 기본 스레드

- 17줄, new/legacy 기하평균 **0.997** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 0

## 배열, 1스레드

- 17줄, new/legacy 기하평균 **1.053** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 4, 10% 넘게 빨라진 줄 0

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| f + g | 148.2µs | 184.9µs | 1.25 |
| f * i + g  (casts) | 393.8µs | 437.4µs | 1.11 |
| ft + 1.0  (transposed) | 86.7µs | 123.5µs | 1.42 |
| maximum(f, g) | 160.6µs | 191.7µs | 1.19 |

## linalg, 기본 스레드

- 258줄, new/legacy 기하평균 **0.952** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 2, 10% 넘게 빨라진 줄 37

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| solve_triangular · (300,300) | 35.8µs | 46.5µs | 1.30 |
| solve_triangular · (100,32,32) | 11.2µs | 13.7µs | 1.23 |
| solve · (3,3) | 469ns | 341ns | 0.73 |
| inv · (3,3) | 457ns | 329ns | 0.72 |
| vecdot rows · (3,3) | 389ns | 269ns | 0.69 |
| outer · (3,3) | 323ns | 196ns | 0.61 |
| diagonal · (3,3) | 262ns | 155ns | 0.59 |
| matrix_power 3 · (3,3) | 446ns | 330ns | 0.74 |
| vector_norm rows · (3,3) | 482ns | 367ns | 0.76 |
| solve_triangular · (3,3) | 378ns | 253ns | 0.67 |
| cholesky · (3,3) | 323ns | 198ns | 0.61 |
| cholesky_solve · (3,3) | 371ns | 247ns | 0.66 |
| lu · (3,3) | 1.1µs | 757ns | 0.69 |
| lu_solve · (3,3) | 520ns | 375ns | 0.72 |
| qr · (3,3) | 1.3µs | 1.0µs | 0.80 |
| qr complete · (3,3) | 1.3µs | 1.0µs | 0.80 |
| qr_pivoted · (3,3) | 1.6µs | 1.3µs | 0.77 |
| lstsq · (3,3) | 2.5µs | 2.3µs | 0.89 |
| eigh · (3,3) | 1.5µs | 1.3µs | 0.88 |
| eigvalsh · (3,3) | 799ns | 675ns | 0.84 |
| eigh(a, b) · (3,3) | 1.7µs | 1.5µs | 0.88 |
| hessenberg · (3,3) | 1.4µs | 1.2µs | 0.83 |
| ldl · (3,3) | 1.3µs | 979ns | 0.76 |
| solve_tridiagonal · (3,3) | 421ns | 281ns | 0.67 |
| solve_banded 1,1 · (3,3) | 380ns | 260ns | 0.68 |
| kron with 3x3 · (3,3) | 379ns | 262ns | 0.69 |
| cross rows · (3,3) | 423ns | 301ns | 0.71 |
| expm · (3,3) | 1.1µs | 967ns | 0.90 |
| vecdot rows · (32,32) | 709ns | 586ns | 0.83 |
| outer · (32,32) | 442ns | 298ns | 0.67 |
| diagonal · (32,32) | 275ns | 158ns | 0.58 |
| matrix_power 3 · (32,32) | 1.8µs | 1.6µs | 0.91 |
| lu_solve · (32,32) | 1.1µs | 931ns | 0.88 |
| solve_tridiagonal · (32,32) | 693ns | 529ns | 0.76 |
| solve_banded 1,1 · (32,32) | 660ns | 498ns | 0.75 |
| cross rows · (32,32) | 541ns | 433ns | 0.80 |
| diagonal · (300,300) | 304ns | 206ns | 0.68 |
| diagonal · (1000,1000) | 490ns | 397ns | 0.81 |
| solve_banded 1,1 · (1000,3,3) | 10.6µs | 7.5µs | 0.70 |

## linalg, 1스레드

- 258줄, new/legacy 기하평균 **0.951** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 3, 10% 넘게 빨라진 줄 44

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| solve_triangular · (300,300) | 35.4µs | 58.1µs | 1.64 |
| solve_triangular · (1000,1000) | 358.2µs | 486.8µs | 1.36 |
| solve_triangular · (100,32,32) | 49.4µs | 59.1µs | 1.20 |
| solve · (3,3) | 469ns | 342ns | 0.73 |
| inv · (3,3) | 453ns | 332ns | 0.73 |
| vecdot rows · (3,3) | 386ns | 269ns | 0.70 |
| outer · (3,3) | 329ns | 197ns | 0.60 |
| diagonal · (3,3) | 269ns | 156ns | 0.58 |
| matrix_power 3 · (3,3) | 447ns | 327ns | 0.73 |
| vector_norm rows · (3,3) | 484ns | 368ns | 0.76 |
| solve_triangular · (3,3) | 378ns | 248ns | 0.66 |
| cholesky · (3,3) | 318ns | 199ns | 0.63 |
| cholesky_solve · (3,3) | 375ns | 246ns | 0.65 |
| lu · (3,3) | 1.1µs | 751ns | 0.67 |
| lu_solve · (3,3) | 515ns | 368ns | 0.71 |
| qr · (3,3) | 1.3µs | 1.0µs | 0.78 |
| qr complete · (3,3) | 1.3µs | 1.0µs | 0.78 |
| qr_pivoted · (3,3) | 1.6µs | 1.3µs | 0.77 |
| lstsq · (3,3) | 2.6µs | 2.3µs | 0.88 |
| eigh · (3,3) | 1.5µs | 1.3µs | 0.84 |
| eigvalsh · (3,3) | 800ns | 660ns | 0.82 |
| eigh(a, b) · (3,3) | 1.7µs | 1.5µs | 0.87 |
| svd · (3,3) | 3.5µs | 3.2µs | 0.90 |
| eigvals · (3,3) | 2.2µs | 1.8µs | 0.83 |
| hessenberg · (3,3) | 1.4µs | 1.2µs | 0.82 |
| ldl · (3,3) | 1.3µs | 953ns | 0.73 |
| solve_tridiagonal · (3,3) | 422ns | 282ns | 0.67 |
| solve_banded 2,2 · (3,3) | 1.3µs | 1.1µs | 0.85 |
| solve_banded 1,1 · (3,3) | 382ns | 258ns | 0.68 |
| solve_sylvester · (3,3) | 4.4µs | 4.0µs | 0.90 |
| kron with 3x3 · (3,3) | 383ns | 264ns | 0.69 |
| cross rows · (3,3) | 434ns | 302ns | 0.70 |
| expm · (3,3) | 1.1µs | 954ns | 0.87 |
| vecdot rows · (32,32) | 715ns | 583ns | 0.82 |
| outer · (32,32) | 425ns | 302ns | 0.71 |
| diagonal · (32,32) | 277ns | 158ns | 0.57 |
| vector_norm rows · (32,32) | 1.2µs | 1.1µs | 0.90 |
| cholesky · (32,32) | 1.4µs | 1.3µs | 0.90 |
| cholesky_solve · (32,32) | 982ns | 822ns | 0.84 |
| lu_solve · (32,32) | 1.1µs | 904ns | 0.85 |
| ldl · (32,32) | 3.7µs | 3.3µs | 0.89 |
| solve_tridiagonal · (32,32) | 679ns | 508ns | 0.75 |
| solve_banded 1,1 · (32,32) | 677ns | 496ns | 0.73 |
| cross rows · (32,32) | 560ns | 431ns | 0.77 |
| diagonal · (300,300) | 314ns | 208ns | 0.66 |
| diagonal · (1000,1000) | 502ns | 384ns | 0.77 |
| diagonal · (1000,3,3) | 3.6µs | 3.2µs | 0.90 |

## linalg 대 numpy/scipy (기본 스레드)

- 248줄: 통합판이 10% 넘게 빠름 192, 비슷 47, 느림 9. ref/new 기하평균 3.21

## frame, 기본 스레드

- 49줄, new/legacy 기하평균 **0.998** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 2, 10% 넘게 빨라진 줄 3

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| s.sum() · 10 | 181ns | 227ns | 1.25 |
| float sum · 1000 | 314ns | 354ns | 1.13 |
| mask filter · 1000 | 2.6µs | 2.4µs | 0.91 |
| mask filter · 100000 | 102.4µs | 73.9µs | 0.72 |
| mask filter · 1000000 | 136.9µs | 107.3µs | 0.78 |

## frame, 1스레드

- 49줄, new/legacy 기하평균 **0.974** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 1, 10% 넘게 빨라진 줄 4

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| s.sum() · 10 | 183ns | 228ns | 1.25 |
| mask filter · 1000 | 2.7µs | 2.5µs | 0.90 |
| sort int64 · 1000 | 15.0µs | 13.2µs | 0.87 |
| mask filter · 100000 | 103.3µs | 73.1µs | 0.71 |
| mask filter · 1000000 | 1.02ms | 715.6µs | 0.70 |

## frame 대 pandas (기본 스레드)

- 49줄: 통합판이 10% 넘게 빠름 44, 비슷 4, 느림 1. ref/new 기하평균 4.64

