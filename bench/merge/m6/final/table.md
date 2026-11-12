## 배열, 기본 스레드

- 17줄, new/legacy 기하평균 **0.993** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 0

## 배열, 1스레드

- 17줄, new/legacy 기하평균 **0.994** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 1

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| f + g | 198.0µs | 144.4µs | 0.73 |

## linalg, 기본 스레드

- 258줄, new/legacy 기하평균 **0.951** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 2, 10% 넘게 빨라진 줄 41

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| lu_solve · (1000,3,3) | 14.6µs | 17.4µs | 1.19 |
| matrix_norm fro · (100,32,32) | 9.5µs | 11.3µs | 1.18 |
| solve · (3,3) | 475ns | 342ns | 0.72 |
| inv · (3,3) | 459ns | 330ns | 0.72 |
| vecdot rows · (3,3) | 380ns | 261ns | 0.69 |
| outer · (3,3) | 324ns | 199ns | 0.62 |
| diagonal · (3,3) | 262ns | 153ns | 0.58 |
| matrix_power 3 · (3,3) | 440ns | 313ns | 0.71 |
| vector_norm rows · (3,3) | 474ns | 359ns | 0.76 |
| solve_triangular · (3,3) | 369ns | 258ns | 0.70 |
| cholesky · (3,3) | 316ns | 198ns | 0.63 |
| cholesky_solve · (3,3) | 363ns | 247ns | 0.68 |
| lu · (3,3) | 1.1µs | 760ns | 0.70 |
| lu_solve · (3,3) | 502ns | 367ns | 0.73 |
| qr · (3,3) | 1.3µs | 982ns | 0.78 |
| qr complete · (3,3) | 1.3µs | 988ns | 0.78 |
| qr_pivoted · (3,3) | 1.6µs | 1.2µs | 0.75 |
| lstsq · (3,3) | 2.5µs | 2.3µs | 0.91 |
| eigh · (3,3) | 1.5µs | 1.3µs | 0.89 |
| eigvalsh · (3,3) | 767ns | 674ns | 0.88 |
| eigh(a, b) · (3,3) | 1.7µs | 1.6µs | 0.90 |
| svd · (3,3) | 3.5µs | 3.1µs | 0.90 |
| hessenberg · (3,3) | 1.4µs | 1.2µs | 0.86 |
| ldl · (3,3) | 1.3µs | 985ns | 0.75 |
| solve_tridiagonal · (3,3) | 409ns | 286ns | 0.70 |
| solve_banded 1,1 · (3,3) | 369ns | 253ns | 0.69 |
| kron with 3x3 · (3,3) | 380ns | 263ns | 0.69 |
| cross rows · (3,3) | 421ns | 302ns | 0.72 |
| expm · (3,3) | 1.1µs | 985ns | 0.89 |
| vecdot rows · (32,32) | 707ns | 589ns | 0.83 |
| outer · (32,32) | 429ns | 289ns | 0.68 |
| diagonal · (32,32) | 270ns | 157ns | 0.58 |
| vector_norm rows · (32,32) | 1.2µs | 1.1µs | 0.90 |
| solve_triangular · (32,32) | 828ns | 726ns | 0.88 |
| cholesky_solve · (32,32) | 948ns | 855ns | 0.90 |
| lu_solve · (32,32) | 1.1µs | 927ns | 0.88 |
| solve_tridiagonal · (32,32) | 661ns | 522ns | 0.79 |
| solve_banded 1,1 · (32,32) | 621ns | 483ns | 0.78 |
| cross rows · (32,32) | 556ns | 427ns | 0.77 |
| diagonal · (300,300) | 312ns | 202ns | 0.65 |
| diagonal · (1000,1000) | 496ns | 398ns | 0.80 |
| solve_tridiagonal · (1000,3,3) | 10.5µs | 7.6µs | 0.72 |
| solve_banded 1,1 · (1000,3,3) | 10.1µs | 7.0µs | 0.70 |

## linalg, 1스레드

- 258줄, new/legacy 기하평균 **0.945** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 45

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| solve · (3,3) | 460ns | 340ns | 0.74 |
| inv · (3,3) | 452ns | 326ns | 0.72 |
| vecdot rows · (3,3) | 384ns | 262ns | 0.68 |
| outer · (3,3) | 328ns | 202ns | 0.62 |
| diagonal · (3,3) | 264ns | 155ns | 0.59 |
| matrix_power 3 · (3,3) | 448ns | 313ns | 0.70 |
| vector_norm rows · (3,3) | 482ns | 362ns | 0.75 |
| solve_triangular · (3,3) | 387ns | 249ns | 0.64 |
| cholesky · (3,3) | 324ns | 198ns | 0.61 |
| cholesky_solve · (3,3) | 370ns | 245ns | 0.66 |
| lu · (3,3) | 1.1µs | 737ns | 0.68 |
| lu_solve · (3,3) | 509ns | 361ns | 0.71 |
| qr · (3,3) | 1.2µs | 989ns | 0.79 |
| qr complete · (3,3) | 1.2µs | 991ns | 0.81 |
| qr_pivoted · (3,3) | 1.6µs | 1.2µs | 0.78 |
| eigh · (3,3) | 1.5µs | 1.3µs | 0.86 |
| eigvalsh · (3,3) | 783ns | 653ns | 0.83 |
| eigh(a, b) · (3,3) | 1.7µs | 1.5µs | 0.87 |
| svd · (3,3) | 3.5µs | 3.1µs | 0.88 |
| schur · (3,3) | 2.7µs | 2.4µs | 0.87 |
| hessenberg · (3,3) | 1.4µs | 1.2µs | 0.85 |
| ldl · (3,3) | 1.3µs | 945ns | 0.72 |
| solve_tridiagonal · (3,3) | 412ns | 278ns | 0.67 |
| solve_banded 2,2 · (3,3) | 1.3µs | 1.1µs | 0.83 |
| solve_banded 1,1 · (3,3) | 374ns | 247ns | 0.66 |
| kron with 3x3 · (3,3) | 370ns | 263ns | 0.71 |
| cross rows · (3,3) | 417ns | 298ns | 0.71 |
| expm · (3,3) | 1.1µs | 945ns | 0.85 |
| vecdot rows · (32,32) | 706ns | 579ns | 0.82 |
| outer · (32,32) | 454ns | 301ns | 0.66 |
| diagonal · (32,32) | 286ns | 157ns | 0.55 |
| matrix_power 3 · (32,32) | 1.8µs | 1.6µs | 0.90 |
| vector_norm rows · (32,32) | 1.2µs | 1.1µs | 0.90 |
| solve_triangular · (32,32) | 850ns | 701ns | 0.82 |
| cholesky · (32,32) | 1.4µs | 1.2µs | 0.86 |
| cholesky_solve · (32,32) | 1.0µs | 798ns | 0.80 |
| lu_solve · (32,32) | 1.1µs | 891ns | 0.81 |
| ldl · (32,32) | 3.7µs | 3.3µs | 0.88 |
| solve_tridiagonal · (32,32) | 714ns | 506ns | 0.71 |
| solve_banded 2,2 · (32,32) | 2.2µs | 2.0µs | 0.90 |
| solve_banded 1,1 · (32,32) | 670ns | 473ns | 0.71 |
| kron with 3x3 · (32,32) | 2.6µs | 2.4µs | 0.91 |
| cross rows · (32,32) | 548ns | 421ns | 0.77 |
| diagonal · (300,300) | 314ns | 204ns | 0.65 |
| diagonal · (1000,1000) | 498ns | 399ns | 0.80 |

## linalg 대 numpy/scipy (기본 스레드)

- 248줄: 통합판이 10% 넘게 빠름 191, 비슷 48, 느림 9. ref/new 기하평균 3.23

## frame, 기본 스레드

- 49줄, new/legacy 기하평균 **0.991** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 1, 10% 넘게 빨라진 줄 2

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| float + float · 1000000 | 38.0µs | 42.6µs | 1.12 |
| mask filter · 100000 | 101.8µs | 73.0µs | 0.72 |
| mask filter · 1000000 | 138.0µs | 104.9µs | 0.76 |

## frame, 1스레드

- 49줄, new/legacy 기하평균 **0.964** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 6

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| mask filter · 1000 | 2.6µs | 2.3µs | 0.88 |
| sort int64 · 1000 | 15.1µs | 13.1µs | 0.87 |
| float + float · 100000 | 20.8µs | 15.3µs | 0.74 |
| mask filter · 100000 | 102.9µs | 72.9µs | 0.71 |
| float + float · 1000000 | 202.1µs | 175.4µs | 0.87 |
| mask filter · 1000000 | 1.02ms | 707.4µs | 0.69 |

## frame 대 pandas (기본 스레드)

- 49줄: 통합판이 10% 넘게 빠름 44, 비슷 4, 느림 1. ref/new 기하평균 4.69

