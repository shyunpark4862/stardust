# T0b 측정: Array API 채우기, bincount·histogram

- 실행: `.venv/bin/python bench/array/t0b_array_api.py` (2026-11-15, 최적화 검토 뒤, Apple Silicon, numpy 2.5.4)
- 호출당 시간(5회 중 최선). `np/sd`가 1보다 크면 stardust가 빠르다. 10스레드 열은 실행마다 흔들린다.
- 배열: `v`, `w` 1e6 float64, `m` 1000×1000, `s` 정렬된 1e5, `big` 1e7.

| 경우 | sd 1T | sd 10T | numpy | np/sd 1T | np/sd 10T |
|---|---:|---:|---:|---:|---:|
| `copysign(v, w)` | 0.1587 ms | 0.0674 ms | 0.2629 ms | 1.66× | 3.90× |
| `hypot(v, w)` | 1.2757 ms | 0.2006 ms | 1.5335 ms | 1.20× | 7.64× |
| `logaddexp(v, w)` | 15.5188 ms | 1.9268 ms | 15.7462 ms | 1.01× | 8.17× |
| `nextafter(v, w)` | 0.6712 ms | 0.1231 ms | 5.5652 ms | 8.29× | 45.22× |
| `reciprocal(v)` | 0.1292 ms | 0.0459 ms | 0.1285 ms | 0.99× | 2.80× |
| `reciprocal(z complex)` | 0.9649 ms | 0.1616 ms | 3.7955 ms | 3.93× | 23.49× |
| `signbit(v)` | 0.0755 ms | 0.0338 ms | 0.0753 ms | 1.00× | 2.23× |
| `roll(v, 12345)` | 0.1058 ms | 0.0479 ms | 0.1066 ms | 1.01× | 2.23× |
| `roll(m, 7, axis=(0,1))` | 0.1510 ms | 0.0641 ms | 0.1669 ms | 1.11× | 2.60× |
| `repeat(m, 2, axis=1)` | 0.5175 ms | 0.1044 ms | 0.5280 ms | 1.02× | 5.06× |
| `tile(m, (2, 2))` | 0.6249 ms | 0.1838 ms | 0.6334 ms | 1.01× | 3.45× |
| `tril(m)` | 0.2006 ms | 0.2083 ms | 0.3738 ms | 1.86× | 1.79× |
| `diff(v)` | 0.1425 ms | 0.0555 ms | 0.1207 ms | 0.85× | 2.18× |
| `diff(m, axis=0)` | 0.1389 ms | 0.0495 ms | 0.1050 ms | 0.76× | 2.12× |
| `meshgrid(x1, y1)` | 0.1395 ms | 0.1402 ms | 0.1579 ms | 1.13× | 1.13× |
| `take(v, idx)` | 0.7391 ms | 0.1596 ms | 1.8507 ms | 2.50× | 11.59× |
| `take_along_axis(m, am, axis=1)` | 0.4586 ms | 0.1364 ms | 2.3443 ms | 5.11× | 17.18× |
| `nonzero(nz 1e6)` | 0.3763 ms | 0.0789 ms | 3.8330 ms | 10.19× | 48.59× |
| `count_nonzero(big 1e7)` | 1.0986 ms | 0.4152 ms | 2.5559 ms | 2.33× | 6.16× |
| `searchsorted(s 1e5, v 1e6)` | 43.0046 ms | 5.7418 ms | 81.6435 ms | 1.90× | 14.22× |
| `bincount(ints 1e6)` | 0.3671 ms | 0.1096 ms | 1.0350 ms | 2.82× | 9.44× |
| `histogram(v, 100)` | 2.1960 ms | 0.4043 ms | 4.2522 ms | 1.94× | 10.52× |
| `histogram(v, 100, weights)` | 2.1614 ms | 0.4056 ms | 4.5091 ms | 2.09× | 11.12× |
| `tensordot(a3, b2, axes=2)` | 0.2334 ms | 0.2323 ms | 0.2314 ms | 0.99× | 1.00× |
| `flip(small) (call cost)` | 0.0001 ms | 0.0001 ms | 0.0002 ms | 2.05× | 2.08× |
| `hypot(1.0, 2.0) (scalar call)` | 0.0001 ms | 0.0001 ms | 0.0003 ms | 5.76× | 6.22× |

## 읽는 법
- 2026-11-15 최적화 검토(A–C) 뒤의 값이다. 전의 값과의 비교는 아래에 있다.
- **원소별 6개**: 다른 ufunc와 같은 디스패처라서 10스레드로 나뉜다. `nextafter`는 비트 연산으로 직접 계산한다(libm과 같은 결과, 8배). complex `reciprocal`은 1스레드도 3.9배다.
- **`nonzero`**: 조각마다 세고 분기 없이 병렬로 채운다. 1스레드 10배, 10스레드 49배.
- **`take_along_axis`**: 결과를 원소별로 돌며 `x_view + index * stride`를 읽는다. 1스레드 5배, 10스레드 17배.
- **`histogram`, `bincount`**: 복사가 없고, 개수는 구간별로, 가중치 합은 65536 블록별로 병렬이다(비트는 numpy와 같다). 10스레드 9–11배.
- **`searchsorted`**: 찾을 값을 나누어 병렬로 찾는다(10스레드 14배).
- **`repeat`, `meshgrid`**: 행 단위로 직접 채운다. `diff`는 `subtract`이고, 따로 재면 numpy와 같다(이 표의 0.8배는 측정마다 흔들린다).
- **`tensordot`**: 거의 전부 BLAS다. numpy와 같다.

## 최적화 검토 전후 (1스레드 / 10스레드)
| 경우 | 전 | 후 |
|---|---|---|
| `nextafter(v, w)` | 5.6 / 0.67 ms | 0.67 / 0.12 ms |
| `nonzero(nz 1e6)` | 2.3 / 2.4 ms (1e7: 23 ms) | 0.38 / 0.08 ms (1e7: 3.8 ms) |
| `take_along_axis(m, am, axis=1)` | 1.6 / 1.6 ms | 0.46 / 0.14 ms |
| `count_nonzero(big 1e7)` | 2.4 / 0.81 ms | 1.1 / 0.42 ms |
| `bincount` (1e7) | 9.5 / 9.2 ms | 3.7 / 1.1 ms |
| `histogram(v, 100)` | 2.3 / 2.4 ms | 2.2 / 0.40 ms |
| `histogram(v, 100, weights)` | 2.6 / 2.6 ms | 2.2 / 0.41 ms |
