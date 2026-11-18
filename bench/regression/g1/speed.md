# G1 속도: sd.regression.ols (최적화 8772322 전 측정)

> 이 파일의 수치는 회귀 커널 최적화(커밋 8772322) 전에 잰 값이다. 그 뒤의 수치는 다시 재야 한다(설계 문서의 일부 값은 그 최적화 때 따로 빠르게 잰 것이다).

`bench/regression/g1/speed.py`로 쟀다(2026-11-18, Apple M4 Pro, statsmodels 0.15.0, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1). 시간은 ms, 여러 번 가운데 가장 빠른 값이다. 입력은 각 라이브러리의 배열로 미리 만들었다(statsmodels의 상수 열 포함). sd의 스레드는 `sd.set_num_threads`, numpy BLAS(statsmodels, scikit-learn 밑)는 threadpoolctl로 정했다.

- fit: `sd.regression.ols(y, X)`(커널 전부: 계수, 개선 1번, 적합값·잔차, 제곱합, 공분산), statsmodels `OLS(y, X).fit()`(기본 `pinv`), scikit-learn `LinearRegression().fit(X, y)`.
- +inference: fit 뒤 `params`, `bse`, `pvalues`, `rsquared`를 읽는다(둘 다 처음 읽을 때 계산한다). scikit-learn에는 추론이 없다.

| (n, k) | 스레드 | sd fit | sd +inference | statsmodels fit | statsmodels +inference | scikit-learn fit |
|---|---:|---:|---:|---:|---:|---:|
| (1,000, 10) | 1 | 0.105 | 0.113 | 0.189 | 0.235 | 0.244 |
| (1,000, 10) | 10 | 0.104 | 0.112 | 0.186 | 0.227 | 0.239 |
| (100,000, 10) | 1 | 9.01 | 9.08 | 10.3 | 10.5 | 5.99 |
| (100,000, 10) | 10 | 3.8 | 3.9 | 10.1 | 10.6 | 6.09 |
| (1,000,000, 50) | 1 | 469 | 467 | 937 | 938 | 330 |
| (1,000,000, 50) | 10 | 236 | 236 | 944 | 945 | 330 |
| (10,000, 1000) | 1 | 214 | 211 | 439 | 441 | 204 |
| (10,000, 1000) | 10 | 186 | 186 | 443 | 435 | 207 |

배치: (50, 3) 회귀 1만 개, fit + inference.

| 스레드 | sd 한 번 호출 | sd Python 반복 | statsmodels Python 반복 |
|---:|---:|---:|---:|
| 1 | 37.0 | 128 | 770 |
| 10 | 4.82 | 129 | 770 |

- statsmodels보다 모든 크기에서 빠르다: 1스레드 1.1–2.0배, 10스레드 1.7–4.0배. 배치는 statsmodels 반복의 21배(1스레드), 160배(10스레드)다.
- scikit-learn(scipy `lstsq`, 그 밑은 scipy의 OpenBLAS)보다는 1스레드의 (10⁵, 10)에서 1.5배, (10⁶, 50)에서 1.4배 느리다. 10스레드에서는 (10⁵, 10) 1.6배, (10⁶, 50) 1.4배 빠르고, (10⁴, 1000)은 같다. scikit-learn의 시간은 threadpoolctl의 제한과 상관없이 같았다.
- sd의 시간 가운데 QR(Accelerate `geqrf`)이 가장 크다. (10⁶, 50)에서 같은 행렬의 R만 구하는 scipy(OpenBLAS) `qr(mode='r')`이 215 ms, sd.linalg.qr(Q도 만든다)이 364 ms였다(10스레드). 나머지는 double-double 행 합 세 번(처음 잔차, 개선 1번, 마지막)과 반사 적용이다. 개선 한 번은 적합 시간의 10–25%다(docs/design/regression.md §3.1).
- 입력 검사(NaN·inf)와 꾸리기는 X의 메모리 순서대로 읽는다. 열 순서로 읽던 첫 구현은 행 우선 (10⁶, 50)에서 그 검사만으로 QR보다 오래 걸렸다(431 → 236 ms, 10스레드).
