# T0 측정: 분위수, NaN 리덕션, cov·corrcoef

- 실행: `.venv/bin/python bench/array/t0_statistics.py` (2026-11-15, 최적화 검토 뒤, Apple Silicon, numpy 2.5.4)
- 호출당 시간(5회 중 최선). numpy는 이 함수들에서 한 스레드다. `np/sd`가 1보다 크면 stardust가 빠르다.
- 10스레드 열은 실행마다 흔들린다(같은 날 앞선 실행에서 `nanmax(mn, axis=1)` 0.026ms, `nansum(bign)` 1.5ms).

| 경우 | sd 1T | sd 10T | numpy | np/sd 1T | np/sd 10T |
|---|---:|---:|---:|---:|---:|
| `quantile(v 1e6, 0.5)` | 6.005 ms | 3.943 ms | 8.493 ms | 1.41× | 2.15× |
| `quantile(v 1e6, [.1,.5,.9])` | 11.882 ms | 4.011 ms | 12.725 ms | 1.07× | 3.17× |
| `quantile(small 100, 0.3)` | 0.001 ms | 0.001 ms | 0.012 ms | 21.98× | 21.34× |
| `quantile(m, 0.5, axis=0)` | 6.165 ms | 0.881 ms | 9.141 ms | 1.48× | 10.37× |
| `quantile(m, [.1,.5,.9], axis=1)` | 5.833 ms | 0.828 ms | 13.256 ms | 2.27× | 16.01× |
| `quantile(v, 0.5, nearest)` | 5.769 ms | 3.872 ms | 7.789 ms | 1.35× | 2.01× |
| `median(v 1e6)` | 6.011 ms | 3.883 ms | 7.730 ms | 1.29× | 1.99× |
| `median(m, axis=1)` | 5.758 ms | 0.812 ms | 8.129 ms | 1.41× | 10.01× |
| `nanmedian(vn 1e6)` | 5.384 ms | 4.414 ms | 8.654 ms | 1.61× | 1.96× |
| `nanmedian(mn, axis=1)` | 6.047 ms | 0.825 ms | 15.185 ms | 2.51× | 18.41× |
| `nanquantile(mn, [.1,.5,.9], axis=0)` | 6.428 ms | 0.906 ms | 29.182 ms | 4.54× | 32.20× |
| `nansum(bign 1e7)` | 1.662 ms | 0.649 ms | 16.154 ms | 9.72× | 24.88× |
| `nanprod(mn, axis=0)` | 0.137 ms | 0.043 ms | 1.428 ms | 10.43× | 33.44× |
| `nanmean(bign 1e7)` | 2.696 ms | 1.085 ms | 17.895 ms | 6.64× | 16.49× |
| `nanmean(mn, axis=1)` | 0.249 ms | 0.075 ms | 1.623 ms | 6.53× | 21.65× |
| `nanvar(bign 1e7)` | 5.886 ms | 1.853 ms | 39.899 ms | 6.78× | 21.54× |
| `nanstd(mn, axis=0)` | 0.827 ms | 0.224 ms | 3.893 ms | 4.71× | 17.35× |
| `nanmin(bign 1e7)` | 0.996 ms | 0.492 ms | 1.008 ms | 1.01× | 2.05× |
| `nanmax(mn, axis=1)` | 0.080 ms | 0.041 ms | 0.089 ms | 1.12× | 2.16× |
| `nanargmin(bign 1e7)` | 6.270 ms | 1.211 ms | 20.638 ms | 3.29× | 17.05× |
| `nancumulative_sum(vn 1e6)` | 0.427 ms | 0.125 ms | 3.729 ms | 8.72× | 29.74× |
| `cov(obs 20x1e5)` | 2.565 ms | 2.288 ms | 4.142 ms | 1.62× | 1.81× |
| `cov(wide 300x3000)` | 0.948 ms | 0.916 ms | 1.160 ms | 1.22× | 1.27× |
| `corrcoef(wide 300x3000)` | 0.985 ms | 0.975 ms | 1.266 ms | 1.28× | 1.30× |

## 읽는 법
- 2026-11-15 최적화 검토(A–C) 뒤의 값이다. 전의 값과의 비교는 아래에 있다.
- **분위수**: 레인이 여럿이면 레인마다 병렬이다(10스레드 10–32배). 4096개 이하의 레인은 NEON 정렬로, 그보다 긴 레인은 `nth_element`로 고른다. 레인 하나(`axis=None`)는 풀이 비어 있으면 조각 병렬 정렬이다(10스레드 2–3배).
- `quantile(small 100)`: numpy는 Python 단계가 많아 12µs이고, stardust는 0.5µs다.
- **NaN 리덕션**: 복사본도 마스크도 만들지 않는다. 원래 커널이 NaN을 중립값으로 읽는다(docs/design/array.md §6.11). numpy는 복사본을 Python에서 여러 번 만든다. 1스레드 3–10배.
- `nanmin`·`nanmax`는 FMINNM/FMAXNM으로 접는다. numpy와 같은 속도다(둘 다 메모리 대역폭).
- **cov·corrcoef**: 복사와 중심화를 한 패스로 하고, 곱은 모양에 따라 `syrk`(절반의 계산) 또는 `gemm`이다. 아래 삼각은 타일 단위로 채운다(docs/design/array.md §6.12). 1.2–1.8배(작은 행렬은 9배).

## 최적화 검토 전후 (1스레드 / 10스레드)
| 경우 | 전 | 후 |
|---|---|---|
| `nansum(bign 1e7)` | 3.8 / 1.6 ms | 1.7 / 0.65 ms |
| `nanmean(bign 1e7)` | 8.9 / 2.4 ms | 2.7 / 1.1 ms |
| `nanvar(bign 1e7)` | 22.5 / 4.1 ms | 5.9 / 1.9 ms |
| `nanargmin(bign 1e7)` | 10.7 / 2.1 ms | 6.3 / 1.2 ms |
| `nancumulative_sum(vn 1e6)` | 0.70 / 0.45 ms | 0.43 / 0.13 ms |
| `quantile(v 1e6, [.1,.5,.9])` | 11.7 / 11.8 ms | 11.9 / 4.0 ms |
| `median(m, axis=1)` | 7.9 / 1.1 ms | 5.8 / 0.81 ms |
| `cov(wide 300x3000)` | 1.11 / 1.39 ms | 0.95 / 0.92 ms |
| `cov(obs 20x1e5)` | 2.71 / 2.35 ms | 2.57 / 2.29 ms |
| `corrcoef(wide 300x3000)` | 1.13 / 1.40 ms | 0.99 / 0.98 ms |
| `cov` (2000×200) | 3.2 / 3.2 ms (1000×1000에서 잼) | 2.3 / 1.5 ms |
