## 배열, 기본 스레드

- 17줄, new/legacy 기하평균 **0.997** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 0

## 배열, 1스레드

- 17줄, new/legacy 기하평균 **0.988** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 1

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| tiny x[1, 2] | 25ns | 22ns | 0.89 |

## linalg, 기본 스레드

- 258줄, new/legacy 기하평균 **0.944** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 45

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| solve · (3,3) | 471ns | 337ns | 0.72 |
| inv · (3,3) | 454ns | 329ns | 0.73 |
| vecdot rows · (3,3) | 382ns | 264ns | 0.69 |
| outer · (3,3) | 321ns | 196ns | 0.61 |
| diagonal · (3,3) | 268ns | 153ns | 0.57 |
| matrix_power 3 · (3,3) | 432ns | 319ns | 0.74 |
| vector_norm rows · (3,3) | 468ns | 355ns | 0.76 |
| solve_triangular · (3,3) | 370ns | 248ns | 0.67 |
| cholesky · (3,3) | 319ns | 196ns | 0.61 |
| cholesky_solve · (3,3) | 365ns | 244ns | 0.67 |
| lu · (3,3) | 1.1µs | 714ns | 0.66 |
| lu_solve · (3,3) | 508ns | 373ns | 0.73 |
| qr · (3,3) | 1.2µs | 985ns | 0.80 |
| qr complete · (3,3) | 1.2µs | 975ns | 0.79 |
| qr_pivoted · (3,3) | 1.6µs | 1.2µs | 0.75 |
| eigh · (3,3) | 1.5µs | 1.3µs | 0.88 |
| eigvalsh · (3,3) | 775ns | 676ns | 0.87 |
| eigh(a, b) · (3,3) | 1.7µs | 1.5µs | 0.89 |
| schur · (3,3) | 2.6µs | 2.4µs | 0.90 |
| hessenberg · (3,3) | 1.4µs | 1.1µs | 0.81 |
| ldl · (3,3) | 1.3µs | 918ns | 0.71 |
| solve_tridiagonal · (3,3) | 410ns | 278ns | 0.68 |
| solve_banded 2,2 · (3,3) | 1.3µs | 1.2µs | 0.88 |
| solve_banded 1,1 · (3,3) | 372ns | 249ns | 0.67 |
| kron with 3x3 · (3,3) | 376ns | 262ns | 0.70 |
| cross rows · (3,3) | 423ns | 296ns | 0.70 |
| expm · (3,3) | 1.1µs | 964ns | 0.90 |
| vecdot rows · (32,32) | 702ns | 586ns | 0.83 |
| outer · (32,32) | 426ns | 309ns | 0.73 |
| diagonal · (32,32) | 273ns | 151ns | 0.55 |
| solve_triangular · (32,32) | 826ns | 717ns | 0.87 |
| cholesky · (32,32) | 1.4µs | 1.3µs | 0.89 |
| cholesky_solve · (32,32) | 964ns | 824ns | 0.86 |
| lu_solve · (32,32) | 1.0µs | 922ns | 0.90 |
| ldl · (32,32) | 3.7µs | 3.3µs | 0.89 |
| solve_tridiagonal · (32,32) | 702ns | 510ns | 0.73 |
| solve_banded 1,1 · (32,32) | 657ns | 492ns | 0.75 |
| cross rows · (32,32) | 540ns | 425ns | 0.79 |
| diagonal · (300,300) | 307ns | 202ns | 0.66 |
| slogdet · (1000,1000) | 2.84ms | 2.55ms | 0.90 |
| diagonal · (1000,1000) | 493ns | 387ns | 0.79 |
| solve_tridiagonal · (1000,3,3) | 17.5µs | 9.4µs | 0.54 |
| solve_banded 1,1 · (1000,3,3) | 10.7µs | 7.3µs | 0.69 |
| lu · (100,32,32) | 141.1µs | 127.8µs | 0.91 |
| eig · (100,32,32) | 3.68ms | 3.30ms | 0.90 |

## linalg, 1스레드

- 258줄, new/legacy 기하평균 **0.934** (1보다 작으면 통합판이 빠르다)
- 10% 넘게 느려진 줄 0, 10% 넘게 빨라진 줄 59

| 행 | legacy | new | 비율 |
|---|---:|---:|---:|
| solve · (3,3) | 458ns | 340ns | 0.74 |
| inv · (3,3) | 441ns | 328ns | 0.74 |
| vecdot rows · (3,3) | 386ns | 254ns | 0.66 |
| outer · (3,3) | 323ns | 195ns | 0.61 |
| trace · (3,3) | 53ns | 47ns | 0.89 |
| diagonal · (3,3) | 268ns | 149ns | 0.56 |
| matrix_power 3 · (3,3) | 436ns | 306ns | 0.70 |
| vector_norm rows · (3,3) | 476ns | 344ns | 0.72 |
| matrix_norm fro · (3,3) | 76ns | 69ns | 0.90 |
| matrix_norm 1 · (3,3) | 68ns | 60ns | 0.88 |
| matrix_norm 2 · (3,3) | 1.6µs | 1.5µs | 0.89 |
| solve_triangular · (3,3) | 368ns | 241ns | 0.65 |
| cholesky · (3,3) | 314ns | 194ns | 0.62 |
| cholesky_solve · (3,3) | 363ns | 243ns | 0.67 |
| lu · (3,3) | 1.1µs | 657ns | 0.58 |
| lu_solve · (3,3) | 503ns | 358ns | 0.71 |
| qr · (3,3) | 1.3µs | 949ns | 0.75 |
| qr complete · (3,3) | 1.3µs | 963ns | 0.75 |
| qr_pivoted · (3,3) | 1.6µs | 1.2µs | 0.73 |
| lstsq · (3,3) | 2.6µs | 2.2µs | 0.88 |
| eigh · (3,3) | 1.5µs | 1.3µs | 0.84 |
| eigvalsh · (3,3) | 792ns | 648ns | 0.82 |
| eigh(a, b) · (3,3) | 1.8µs | 1.5µs | 0.84 |
| svd · (3,3) | 3.6µs | 3.1µs | 0.85 |
| svdvals · (3,3) | 1.8µs | 1.6µs | 0.91 |
| cond · (3,3) | 1.6µs | 1.5µs | 0.91 |
| eig · (3,3) | 5.0µs | 4.2µs | 0.84 |
| eigvals · (3,3) | 2.0µs | 1.6µs | 0.81 |
| schur · (3,3) | 2.7µs | 2.1µs | 0.80 |
| hessenberg · (3,3) | 1.4µs | 1.1µs | 0.77 |
| ldl · (3,3) | 1.3µs | 901ns | 0.68 |
| solve_tridiagonal · (3,3) | 410ns | 272ns | 0.66 |
| solve_banded 2,2 · (3,3) | 1.3µs | 1.1µs | 0.81 |
| solve_banded 1,1 · (3,3) | 374ns | 237ns | 0.63 |
| solve_sylvester · (3,3) | 4.3µs | 3.6µs | 0.84 |
| kron with 3x3 · (3,3) | 372ns | 256ns | 0.69 |
| cross rows · (3,3) | 427ns | 286ns | 0.67 |
| expm · (3,3) | 1.1µs | 886ns | 0.82 |
| vecdot rows · (32,32) | 704ns | 572ns | 0.81 |
| outer · (32,32) | 431ns | 289ns | 0.67 |
| trace · (32,32) | 61ns | 56ns | 0.91 |
| diagonal · (32,32) | 275ns | 154ns | 0.56 |
| vector_norm rows · (32,32) | 1.2µs | 1.0µs | 0.87 |
| matrix_norm 2 · (32,32) | 23.4µs | 21.2µs | 0.90 |
| solve_triangular · (32,32) | 842ns | 666ns | 0.79 |
| cholesky · (32,32) | 1.4µs | 1.2µs | 0.82 |
| cholesky_solve · (32,32) | 985ns | 773ns | 0.79 |
| lu_solve · (32,32) | 1.1µs | 862ns | 0.81 |
| cond · (32,32) | 23.6µs | 21.4µs | 0.91 |
| ldl · (32,32) | 3.7µs | 3.1µs | 0.84 |
| solve_tridiagonal · (32,32) | 680ns | 492ns | 0.72 |
| solve_banded 2,2 · (32,32) | 2.2µs | 1.9µs | 0.89 |
| solve_banded 1,1 · (32,32) | 637ns | 481ns | 0.75 |
| cross rows · (32,32) | 542ns | 425ns | 0.78 |
| diagonal · (300,300) | 307ns | 202ns | 0.66 |
| solve_tridiagonal · (300,300) | 2.8µs | 2.5µs | 0.90 |
| solve_banded 1,1 · (300,300) | 2.7µs | 2.5µs | 0.90 |
| cross rows · (300,300) | 1.6µs | 1.4µs | 0.91 |
| diagonal · (1000,1000) | 478ns | 396ns | 0.83 |

## linalg 대 numpy/scipy (기본 스레드)

- 248줄: 통합판이 10% 넘게 빠름 189, 비슷 50, 느림 9. ref/new 기하평균 3.21
