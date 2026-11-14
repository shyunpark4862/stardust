# T0 측정: 분위수, NaN 리덕션, cov·corrcoef

- 실행: `.venv/bin/python bench/array/t0_statistics.py` (2026-11-15, Apple Silicon, numpy 2.5.4)
- 호출당 시간(5회 중 최선). numpy는 이 함수들에서 한 스레드다. `np/sd`가 1보다 크면 stardust가 빠르다.
- 10스레드 열은 실행마다 흔들린다(같은 날 앞선 실행에서 `nanmax(mn, axis=1)` 0.026ms, `nansum(bign)` 1.5ms).

| 경우 | sd 1T | sd 10T | numpy | np/sd 1T | np/sd 10T |
|---|---:|---:|---:|---:|---:|
| `quantile(v 1e6, 0.5)` | 6.155 ms | 5.926 ms | 8.474 ms | 1.38× | 1.43× |
| `quantile(v 1e6, [.1,.5,.9])` | 11.730 ms | 11.828 ms | 12.697 ms | 1.08× | 1.07× |
| `quantile(small 100, 0.3)` | 0.000 ms | 0.000 ms | 0.012 ms | 28.77× | 27.62× |
| `quantile(m, 0.5, axis=0)` | 8.305 ms | 1.160 ms | 9.280 ms | 1.12× | 8.00× |
| `quantile(m, [.1,.5,.9], axis=1)` | 13.594 ms | 1.887 ms | 13.318 ms | 0.98× | 7.06× |
| `quantile(v, 0.5, nearest)` | 5.672 ms | 5.703 ms | 7.802 ms | 1.38× | 1.37× |
| `median(v 1e6)` | 5.845 ms | 5.979 ms | 7.750 ms | 1.33× | 1.30× |
| `median(m, axis=1)` | 7.933 ms | 1.128 ms | 8.248 ms | 1.04× | 7.31× |
| `nanmedian(vn 1e6)` | 5.249 ms | 5.294 ms | 8.822 ms | 1.68× | 1.67× |
| `nanmedian(mn, axis=1)` | 7.923 ms | 1.091 ms | 15.322 ms | 1.93× | 14.04× |
| `nanquantile(mn, [.1,.5,.9], axis=0)` | 13.685 ms | 2.073 ms | 30.536 ms | 2.23× | 14.73× |
| `nansum(bign 1e7)` | 4.215 ms | 2.485 ms | 16.677 ms | 3.96× | 6.71× |
| `nanprod(mn, axis=0)` | 0.415 ms | 0.412 ms | 1.519 ms | 3.66× | 3.69× |
| `nanmean(bign 1e7)` | 10.042 ms | 3.017 ms | 18.359 ms | 1.83× | 6.09× |
| `nanmean(mn, axis=1)` | 1.128 ms | 0.559 ms | 1.674 ms | 1.48× | 2.99× |
| `nanvar(bign 1e7)` | 20.088 ms | 5.251 ms | 41.667 ms | 2.07× | 7.93× |
| `nanstd(mn, axis=0)` | 2.379 ms | 0.852 ms | 4.174 ms | 1.75× | 4.90× |
| `nanmin(bign 1e7)` | 1.172 ms | 0.712 ms | 1.172 ms | 1.00× | 1.65× |
| `nanmax(mn, axis=1)` | 0.081 ms | 0.081 ms | 0.090 ms | 1.11× | 1.12× |
| `nanargmin(bign 1e7)` | 10.448 ms | 2.948 ms | 20.950 ms | 2.01× | 7.11× |
| `nancumulative_sum(vn 1e6)` | 0.700 ms | 0.445 ms | 3.784 ms | 5.40× | 8.50× |
| `cov(obs 20x1e5)` | 2.888 ms | 2.700 ms | 4.381 ms | 1.52× | 1.62× |
| `cov(wide 300x3000)` | 1.179 ms | 1.389 ms | 1.199 ms | 1.02× | 0.86× |
| `corrcoef(wide 300x3000)` | 1.203 ms | 1.391 ms | 1.283 ms | 1.07× | 0.92× |

## 읽는 법
- **분위수**: 레인이 여럿이면 레인마다 병렬이다(1000×1000에서 10스레드 7–15배). 레인 하나(`axis=None`)는 직렬이다. 그래도 numpy의 1.1–1.7배다(`nth_element`, 가운데 위치부터 나누어 고르기).
- `quantile(small 100)`: numpy는 Python 단계가 많아 12µs이고, stardust는 0.4µs다.
- **NaN 리덕션**: numpy는 NaN을 바꾼 복사본을 Python에서 여러 번 만든다. stardust는 원소별 커널 하나로 복사본과 마스크를 함께 만든다(`nanvar`는 두 번째 패스에서 NaN을 건너뛰어 복사본을 하나만 쓴다). 1스레드 1.5–5배다.
- `nanmin`·`nanmax`는 복사 없이 FMINNM/FMAXNM으로 접는다. numpy와 같은 속도다(둘 다 메모리 대역폭).
- **cov·corrcoef**: 시간의 대부분이 행렬곱(BLAS)이다. 10스레드 열이 1스레드보다 느린 것은 복사(`assign`)와 풀 호출의 비용이다. numpy와 거의 같다.
