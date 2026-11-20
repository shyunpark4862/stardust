# sd.regression G7a 속도

`.venv/bin/python bench/regression/g7a/speed.py`가 만든다(`--render`는 results.json에서 표만 다시 쓴다). 검정만 잰다: 적합은 미리 한다. 같은 함수가 없으면 —. statsmodels는 numpy의 BLAS 스레드, R은 기본 BLAS다. 다른 작업이 함께 돈 기계에서 잰 값이라 수 % 흔들린다.

| 경우 (n, k) | sd 1스레드 ms | sd 10스레드 ms | statsmodels ms | R ms |
|---|---:|---:|---:|---:|
| t_test (10000, 10) | 0.0202 | 0.0193 | 0.0402 | — |
| wald_test (10000, 10) | 0.0331 | 0.0317 | 0.0605 | 0.726 |
| compare_f_test (10000, 10) | 0.0333 | 0.0327 | 0.0204 | 0.292 |
| compare_lr_test (10000, 10) | 0.0328 | 0.0288 | 0.019 | 0.229 |
| anova_typ1 (10000, 10) | 0.0489 | 0.0453 | — | 0.228 |
| anova_typ3 (10000, 10) | 0.245 | 0.246 | — | 8.91 |
| vif (10000, 10) | 0.138 | 0.0545 | 18.3 | 0.668 |
| dffits (10000, 10) | 0.636 | 0.627 | 1.36e+04 | 0.544 |
| dfbetas (10000, 10) | 0.898 | 0.844 | 1.3e+04 | 2.32 |
| white_test (10000, 10) | 6.58 | 5.32 | 12.8 | — |
| goldfeld_quandt (10000, 10) | 1.08 | 1.06 | 1.36 | 1.07 |
| reset_test (10000, 10) | 1.15 | 1.16 | 1.9 | 5.36 |
| glm influence (10000, 10) | 1.09 | 0.989 | 2.64 | — |
| t_test (100000, 10) | 0.0189 | 0.0189 | 0.0395 | — |
| wald_test (100000, 10) | 0.0303 | 0.032 | 0.0589 | 4.56 |
| compare_f_test (100000, 10) | 0.172 | 0.0791 | 0.0202 | 0.575 |
| compare_lr_test (100000, 10) | 0.172 | 0.0782 | 0.0185 | 1.42 |
| anova_typ1 (100000, 10) | 0.0462 | 0.0439 | — | 0.514 |
| anova_typ3 (100000, 10) | 0.238 | 0.228 | — | 132 |
| vif (100000, 10) | 1.3 | 0.294 | 161 | 4.48 |
| dffits (100000, 10) | 4.94 | 4.53 | — | 12.6 |
| dfbetas (100000, 10) | 7.96 | 6.13 | — | 38.7 |
| white_test (100000, 10) | 71.6 | 37.8 | 91.7 | — |
| goldfeld_quandt (100000, 10) | 10.3 | 10.4 | 11 | 22.2 |
| reset_test (100000, 10) | 11 | 4.23 | 16.6 | 72.8 |
| glm influence (100000, 10) | 9.43 | 7.3 | 23.6 | — |

## bench/speed·bench/accuracy에 넣을 경우 (그 체계의 주인이 넣는다)

- 속도(`bench/speed`의 regression 경우로; statsmodels·R 비교자는 위와 같다): (n, k) = (10⁴, 10), (10⁵, 10)에서
  `t_test`(두 줄), `wald_test`(두 줄), `compare_f_test`·`compare_lr_test`(작은 모형 열 하나), `anova`(Type I, Type III,
  모형 셋), `vif`, `dffits`, `dfbetas`, `white_test`, `goldfeld_quandt`, `reset_test`, GLM(이항)의 `dfbetas`; 배치 1만 개의
  (50, 3)에서 `t_test`·`wald_test`·`vif`·`dfbetas`. statsmodels의 `OLSInfluence.dffits`·`dfbetas`는 행마다 다시 적합해
  (n = 10⁴에서 14 s) 10⁴에서만 잰다.
- 정확도(`bench/accuracy`의 regression; 참값 tests/regression/ref/g7a.json): 문제마다 `t_test`의 effect·sd·t·p,
  `wald_test`, `compare_f_test`, `compare_lr_test`, Type I·III 표, `vif`, `dffits`, `dfbetas`, `white_test`,
  `goldfeld_quandt`, `reset_test`, GLM의 `t_test`·`wald_test`·`compare_lr_test`·Type I·`leverage`·`cooks_distance`·
  `dffits`·`dfbetas`. 비교자는 statsmodels(같은 이름의 함수)와 R(lmtest, car, stats; tools/gen_regression_g7a_r.py와
  같은 호출).
