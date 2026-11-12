## 배열, 기본 스레드

- 17줄, new/legacy 기하평균 **0.992** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 0

## 배열, 1스레드

- 17줄, new/legacy 기하평균 **1.040** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 3, 10% 넘게 빨라진 줄 0

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| f + g | 136.6µs | 182.5µs | 1.34 |
| f * i + g  (casts) | 380.9µs | 440.9µs | 1.16 |
| maximum(f, g) | 146.8µs | 188.3µs | 1.28 |

## linalg, 기본 스레드

- 258줄, new/legacy 기하평균 **0.952** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 1, 10% 넘게 빨라진 줄 40

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| eig · (100,32,32) | 3.02ms | 3.74ms | 1.24 |
| solve · (3,3) | 473ns | 344ns | 0.73 |
| inv · (3,3) | 455ns | 331ns | 0.73 |
| vecdot rows · (3,3) | 383ns | 269ns | 0.70 |
| outer · (3,3) | 322ns | 199ns | 0.62 |
| diagonal · (3,3) | 266ns | 152ns | 0.57 |
| matrix_power 3 · (3,3) | 443ns | 329ns | 0.74 |
| vector_norm rows · (3,3) | 474ns | 368ns | 0.78 |
| solve_triangular · (3,3) | 372ns | 253ns | 0.68 |
| cholesky · (3,3) | 324ns | 196ns | 0.60 |
| cholesky_solve · (3,3) | 369ns | 251ns | 0.68 |
| lu · (3,3) | 1.1µs | 716ns | 0.64 |
| lu_solve · (3,3) | 510ns | 377ns | 0.74 |
| qr · (3,3) | 1.3µs | 991ns | 0.78 |
| qr complete · (3,3) | 1.3µs | 1.0µs | 0.78 |
| qr_pivoted · (3,3) | 1.6µs | 1.2µs | 0.77 |
| lstsq · (3,3) | 2.5µs | 2.3µs | 0.90 |
| eigh · (3,3) | 1.5µs | 1.3µs | 0.86 |
| eigvalsh · (3,3) | 789ns | 675ns | 0.86 |
| eigh(a, b) · (3,3) | 1.7µs | 1.5µs | 0.86 |
| hessenberg · (3,3) | 1.4µs | 1.2µs | 0.81 |
| ldl · (3,3) | 1.3µs | 931ns | 0.71 |
| solve_tridiagonal · (3,3) | 419ns | 278ns | 0.66 |
| solve_banded 1,1 · (3,3) | 383ns | 252ns | 0.66 |
| kron with 3x3 · (3,3) | 383ns | 263ns | 0.69 |
| cross rows · (3,3) | 432ns | 299ns | 0.69 |
| expm · (3,3) | 1.1µs | 951ns | 0.87 |
| vecdot rows · (32,32) | 701ns | 594ns | 0.85 |
| outer · (32,32) | 431ns | 299ns | 0.70 |
| diagonal · (32,32) | 272ns | 158ns | 0.58 |
| matrix_power 3 · (32,32) | 1.8µs | 1.6µs | 0.90 |
| solve_triangular · (32,32) | 820ns | 722ns | 0.88 |
| lu_solve · (32,32) | 1.0µs | 916ns | 0.89 |
| ldl · (32,32) | 3.7µs | 3.3µs | 0.90 |
| solve_tridiagonal · (32,32) | 678ns | 526ns | 0.78 |
| solve_banded 1,1 · (32,32) | 649ns | 492ns | 0.76 |
| cross rows · (32,32) | 542ns | 432ns | 0.80 |
| diagonal · (300,300) | 301ns | 210ns | 0.70 |
| diagonal · (1000,1000) | 489ns | 398ns | 0.81 |
| diagonal · (1000,3,3) | 3.6µs | 3.2µs | 0.90 |
| solve_banded 1,1 · (1000,3,3) | 10.5µs | 7.3µs | 0.69 |

## linalg, 1스레드

- 258줄, new/legacy 기하평균 **0.948** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 44

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| solve · (3,3) | 464ns | 344ns | 0.74 |
| inv · (3,3) | 454ns | 331ns | 0.73 |
| vecdot rows · (3,3) | 383ns | 267ns | 0.70 |
| outer · (3,3) | 326ns | 202ns | 0.62 |
| diagonal · (3,3) | 266ns | 152ns | 0.57 |
| matrix_power 3 · (3,3) | 436ns | 325ns | 0.74 |
| vector_norm rows · (3,3) | 474ns | 363ns | 0.77 |
| solve_triangular · (3,3) | 376ns | 251ns | 0.67 |
| cholesky · (3,3) | 316ns | 198ns | 0.63 |
| cholesky_solve · (3,3) | 362ns | 252ns | 0.70 |
| lu · (3,3) | 1.1µs | 713ns | 0.67 |
| lu_solve · (3,3) | 510ns | 375ns | 0.74 |
| qr · (3,3) | 1.2µs | 981ns | 0.81 |
| qr complete · (3,3) | 1.2µs | 976ns | 0.79 |
| qr_pivoted · (3,3) | 1.6µs | 1.2µs | 0.78 |
| lstsq · (3,3) | 2.6µs | 2.3µs | 0.88 |
| eigh · (3,3) | 1.5µs | 1.3µs | 0.86 |
| eigvalsh · (3,3) | 783ns | 671ns | 0.86 |
| eigh(a, b) · (3,3) | 1.7µs | 1.5µs | 0.86 |
| svd · (3,3) | 3.5µs | 3.1µs | 0.88 |
| schur · (3,3) | 2.7µs | 2.3µs | 0.88 |
| hessenberg · (3,3) | 1.4µs | 1.1µs | 0.81 |
| ldl · (3,3) | 1.3µs | 933ns | 0.73 |
| solve_tridiagonal · (3,3) | 409ns | 278ns | 0.68 |
| solve_banded 2,2 · (3,3) | 1.3µs | 1.1µs | 0.83 |
| solve_banded 1,1 · (3,3) | 378ns | 250ns | 0.66 |
| kron with 3x3 · (3,3) | 368ns | 264ns | 0.72 |
| cross rows · (3,3) | 424ns | 299ns | 0.70 |
| expm · (3,3) | 1.1µs | 956ns | 0.88 |
| vecdot rows · (32,32) | 708ns | 580ns | 0.82 |
| outer · (32,32) | 429ns | 291ns | 0.68 |
| diagonal · (32,32) | 274ns | 156ns | 0.57 |
| vector_norm rows · (32,32) | 1.2µs | 1.1µs | 0.91 |
| solve_triangular · (32,32) | 842ns | 696ns | 0.83 |
| cholesky · (32,32) | 1.4µs | 1.3µs | 0.89 |
| cholesky_solve · (32,32) | 977ns | 816ns | 0.84 |
| lu_solve · (32,32) | 1.0µs | 900ns | 0.86 |
| ldl · (32,32) | 3.7µs | 3.3µs | 0.90 |
| solve_tridiagonal · (32,32) | 691ns | 514ns | 0.74 |
| solve_banded 1,1 · (32,32) | 675ns | 478ns | 0.71 |
| cross rows · (32,32) | 542ns | 432ns | 0.80 |
| diagonal · (300,300) | 311ns | 205ns | 0.66 |
| solve_triangular · (300,300) | 45.7µs | 35.8µs | 0.78 |
| diagonal · (1000,1000) | 486ns | 408ns | 0.84 |

## linalg 대 numpy/scipy (기본 스레드)

- 248줄: 통합판이 10% 넘게 빠름 191, 비슷 48, 느림 9. ref/new 기하평균 3.23
