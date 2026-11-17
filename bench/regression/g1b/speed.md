# G1b 속도: HAC·군집 공분산, 진단, summary()

`bench/regression/g1b/speed.py`로 쟀다(2026-11-18, Apple M4 Pro, statsmodels 0.15.0, numpy 2.5.4). 시간은 ms, 여러 번 가운데 가장 빠른 값이다(원래 출력은 `speed.log`). 입력은 각 라이브러리의 배열로 미리 만들었다(statsmodels의 상수 열 포함). 오차는 AR(1)(계수 0.5)이다. sd의 스레드는 `sd.set_num_threads`, numpy BLAS(statsmodels 밑)는 threadpoolctl로 정했다.

- fit: `ols(y, X)`와 statsmodels `OLS(y, X).fit()`, 둘 다 `bse`·`pvalues`를 읽는다.
- HAC: `cov_type='HAC'`(기본 지연 수 floor(4(n/100)^(2/9)): 10⁵에서 18, 10⁶에서 30), statsmodels는 같은 `maxlags`와 `use_t=True`.
- 군집: 군집 1,000개(int64 레이블), statsmodels는 `cov_kwds={'groups': g}, use_t=True`.
- 진단: 적합 뒤 지렛대, 내적 스튜던트화 잔차, Cook 거리, Durbin–Watson, Jarque–Bera, Breusch–Pagan. statsmodels는 `OLSInfluence`의 `hat_matrix_diag`, `resid_studentized_internal`, `cooks_distance`, `durbin_watson(wresid)`, `jarque_bera(wresid)`, `het_breuschpagan(resid, exog)`. 외적 스튜던트화 잔차는 sd만 쟀다(statsmodels의 `resid_studentized_external`은 행을 하나씩 뺀 적합을 n번 한다).
- summary: 적합 뒤 `summary()`(statsmodels는 `str(summary())`).

| (n, k) | 스레드 | 경우 | sd | statsmodels | 배 |
|---|---:|---|---:|---:|---:|
| (100,000, 10) | 1 | fit | 9.04 | 10.2 | 1.1 |
| (100,000, 10) | 1 | HAC (18) | 14.8 | 17.6 | 1.2 |
| (100,000, 10) | 1 | 군집 (1,000) | 12.3 | 13.5 | 1.1 |
| (100,000, 10) | 1 | fit + 진단 | 25.5 | 30.2 | 1.2 |
| (100,000, 10) | 1 | fit + summary() | 11.0 | 17.9 | 1.6 |
| (100,000, 10) | 1 | fit + 외적 스튜던트화 | 14.5 | | |
| (100,000, 10) | 10 | fit | 3.83 | 10.4 | 2.7 |
| (100,000, 10) | 10 | HAC (18) | 7.77 | 17.8 | 2.3 |
| (100,000, 10) | 10 | 군집 (1,000) | 6.90 | 13.7 | 2.0 |
| (100,000, 10) | 10 | fit + 진단 | 13.1 | 30.2 | 2.3 |
| (100,000, 10) | 10 | fit + summary() | 4.39 | 18.0 | 4.1 |
| (100,000, 10) | 10 | fit + 외적 스튜던트화 | 8.80 | | |
| (1,000,000, 50) | 1 | fit | 465 | 935 | 2.0 |
| (1,000,000, 50) | 1 | HAC (30) | 871 | 1,500 | 1.7 |
| (1,000,000, 50) | 1 | 군집 (1,000) | 659 | 1,200 | 1.8 |
| (1,000,000, 50) | 1 | fit + 진단 | 1,370 | 2,040 | 1.5 |
| (1,000,000, 50) | 1 | fit + summary() | 486 | 1,040 | 2.1 |
| (1,000,000, 50) | 1 | fit + 외적 스튜던트화 | 887 | | |
| (1,000,000, 50) | 10 | fit | 239 | 940 | 3.9 |
| (1,000,000, 50) | 10 | HAC (30) | 489 | 1,510 | 3.1 |
| (1,000,000, 50) | 10 | 군집 (1,000) | 420 | 1,200 | 2.9 |
| (1,000,000, 50) | 10 | fit + 진단 | 860 | 2,040 | 2.4 |
| (1,000,000, 50) | 10 | fit + summary() | 242 | 1,040 | 4.3 |
| (1,000,000, 50) | 10 | fit + 외적 스튜던트화 | 619 | | |

- 모든 경우에 statsmodels보다 빠르다: 1스레드 1.1–2.1배, 10스레드 2.0–4.3배.
- HAC와 군집이 고전형 적합에 더하는 시간의 대부분은 얇은 Q(`orgqr`, QR 한 번 꼴)다. HAC의 지연 합은 지연마다 행렬곱을 하지 않고 Bartlett 필터 V를 만들어 UᵀV 한 번으로 구한다. (10⁶, 50)에서 HAC(지연 30)는 군집보다 1스레드 212 ms, 10스레드 69 ms 더 걸리고, 이것이 필터와 UᵀV다.
- 진단의 시간은 적합, 지렛대를 위한 QR과 `orgqr`(적합과 같은 열로 한 번 더), Breusch–Pagan의 보조 회귀(같은 크기의 적합 한 번)다. 지렛대를 적합 때 늘 만들면 모든 적합이 `orgqr`만큼 느려지므로 처음 읽을 때 만든다. Durbin–Watson과 Jarque–Bera의 합도 처음 읽을 때 따로 더해서 적합의 시간에 들어가지 않는다((10⁵, 10) 1스레드에서 적합 안에서 더하면 9.0 → 10.3 ms).
- summary()는 적합에 Durbin–Watson·Jarque–Bera의 합과 조건수만 더한다.
