# T0b 측정: Array API 채우기, bincount·histogram

- 실행: `.venv/bin/python bench/array/t0b_array_api.py` (2026-11-15, Apple Silicon, numpy 2.5.4)
- 호출당 시간(5회 중 최선). `np/sd`가 1보다 크면 stardust가 빠르다. 10스레드 열은 실행마다 흔들린다.
- 배열: `v`, `w` 1e6 float64, `m` 1000×1000, `s` 정렬된 1e5, `big` 1e7.

| 경우 | sd 1T | sd 10T | numpy | np/sd 1T | np/sd 10T |
|---|---:|---:|---:|---:|---:|
| `copysign(v, w)` | 0.1552 ms | 0.0730 ms | 0.2660 ms | 1.71× | 3.64× |
| `hypot(v, w)` | 1.2704 ms | 0.2191 ms | 1.5313 ms | 1.21× | 6.99× |
| `logaddexp(v, w)` | 15.3557 ms | 2.0152 ms | 15.7821 ms | 1.03× | 7.83× |
| `nextafter(v, w)` | 5.5436 ms | 0.6839 ms | 5.5559 ms | 1.00× | 8.12× |
| `reciprocal(v)` | 0.1283 ms | 0.0458 ms | 0.1289 ms | 1.00× | 2.81× |
| `reciprocal(z complex)` | 0.9528 ms | 0.1709 ms | 3.7851 ms | 3.97× | 22.15× |
| `signbit(v)` | 0.0753 ms | 0.0368 ms | 0.0750 ms | 1.00× | 2.03× |
| `roll(v, 12345)` | 0.1027 ms | 0.0607 ms | 0.1077 ms | 1.05× | 1.77× |
| `roll(m, 7, axis=(0,1))` | 0.1484 ms | 0.0710 ms | 0.1585 ms | 1.07× | 2.23× |
| `repeat(m, 2, axis=1)` | 0.5347 ms | 0.1097 ms | 0.5376 ms | 1.01× | 4.90× |
| `tile(m, (2, 2))` | 0.5116 ms | 0.1966 ms | 0.5840 ms | 1.14× | 2.97× |
| `tril(m)` | 0.1968 ms | 0.1936 ms | 0.3697 ms | 1.88× | 1.91× |
| `diff(v)` | 0.1097 ms | 0.0615 ms | 0.1117 ms | 1.02× | 1.82× |
| `diff(m, axis=0)` | 0.1037 ms | 0.0570 ms | 0.1031 ms | 0.99× | 1.81× |
| `meshgrid(x1, y1)` | 0.1390 ms | 0.1392 ms | 0.1576 ms | 1.13× | 1.13× |
| `take(v, idx)` | 0.7435 ms | 0.1671 ms | 1.8769 ms | 2.52× | 11.23× |
| `take_along_axis(m, am, axis=1)` | 1.6327 ms | 1.6256 ms | 2.3578 ms | 1.44× | 1.45× |
| `nonzero(nz 1e6)` | 2.3217 ms | 2.3698 ms | 3.7966 ms | 1.64× | 1.60× |
| `count_nonzero(big 1e7)` | 2.3678 ms | 0.9091 ms | 2.5707 ms | 1.09× | 2.83× |
| `searchsorted(s 1e5, v 1e6)` | 42.6056 ms | 5.7996 ms | 81.0543 ms | 1.90× | 13.98× |
| `bincount(ints 1e6)` | 0.8853 ms | 0.8720 ms | 1.0441 ms | 1.18× | 1.20× |
| `histogram(v, 100)` | 2.7756 ms | 2.7572 ms | 4.2037 ms | 1.51× | 1.52× |
| `histogram(v, 100, weights)` | 3.0462 ms | 3.0009 ms | 4.3794 ms | 1.44× | 1.46× |
| `tensordot(a3, b2, axes=2)` | 0.2317 ms | 0.2325 ms | 0.2316 ms | 1.00× | 1.00× |
| `flip(small) (call cost)` | 0.0001 ms | 0.0001 ms | 0.0002 ms | 2.09× | 2.14× |
| `hypot(1.0, 2.0) (scalar call)` | 0.0001 ms | 0.0001 ms | 0.0003 ms | 5.93× | 6.28× |

## 읽는 법
- **원소별 6개**: 다른 ufunc와 같은 디스패처라서 10스레드로 나뉜다(`logaddexp`, `nextafter` 8배). 1스레드는 numpy와 같다(같은 libm 공식). complex `reciprocal`은 1스레드도 3.9배다(numpy의 complex 루프가 느리다).
- **`repeat`, `meshgrid`**: 처음에는 gather(인덱스 배열)와 stride 0 뷰의 `assign`으로 만들어 numpy의 0.76배, 0.43배였다. 행 단위로 직접 채우는 루프로 바꿔 1.0배, 1.13배가 되었다. `repeat`은 출력의 겹치지 않는 부분으로 나누어 병렬이다.
- **`searchsorted`**: 찾을 값을 나누어 병렬로 찾는다(10스레드 14배). 1스레드도 2배다(numpy는 매번 앞 결과를 힌트로 쓰는 분기가 많다).
- **`take`**: 고급 인덱싱의 gather(§7.3)다. 1스레드 2.4배, 10스레드 10배.
- **`histogram`, `bincount`**: numpy의 블록 구조를 그대로 따라서 직렬이다(가중치 합의 비트를 맞추려고). 그래도 1.2–1.5배다.
- **`tensordot`**: 거의 전부 BLAS다. numpy와 같다.
- 호출 비용: `flip(small)` 0.1µs(numpy 0.2µs), 스칼라 `hypot(1.0, 2.0)` 0.05µs(numpy 0.3µs).
