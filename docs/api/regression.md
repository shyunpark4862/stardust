# sd.regression API

> 상태: **설계 초안** (2026-11-18), 구현 전. 시그니처는 안이고, 이름·기본값·결과 필드는 마일스톤마다 확정한다([docs/design/regression.md](../design/regression.md) §8).

## §1. 공통 규칙 (안)
- 입력은 위치로만(`y`, `X`), 옵션은 키워드로만. 별칭은 없다.
- `y`는 길이 n의 1차원, `X`는 (n,) 또는 (n, k)다. `sd.ndarray`, 실수의 시퀀스, `sd.frame`의 `Series`·`DataFrame`을 받는다. frame이면 계수에 컬럼 이름이 붙는다.
- float64로 계산한다(bool·int64는 바꿔서). complex는 `TypeError`다.
- **절편은 기본으로 넣는다**(사용자 결정 2026-11-18). `intercept=False`로 끈다.
- NaN은 `nan_policy`('propagate', 'omit', 'raise')로 다룬다(sd.stats의 기술통계와 같다).
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 경고는 내지 않는다. 수렴·특이성 같은 상태는 결과의 필드다.
- 에러는 함수 이름으로 시작한다(`ols(): 'X' must have 100 rows like 'y', not 99`).
- 정확도: 닫힌 꼴은 Arb 참값과 비교해 잰 값을 적는다(계약의 꼴은 조건수를 넣은 한계, G1에서 확정). 반복 방법은 최적성 조건의 잔차와 statsmodels·scikit-learn과의 차이를 적는다.

## §2. 선형 (G1)
```
ols(y, X, /, *, intercept=True, cov_type='nonrobust', rtol=None, nan_policy='propagate')
wls(y, X, /, *, weights, intercept=True, cov_type='nonrobust', rtol=None, nan_policy='propagate')
gls(y, X, /, *, sigma, intercept=True, nan_policy='propagate')
linregress(x, y, /)                       # -> LinregressResult(slope, intercept, rvalue, pvalue, stderr, intercept_stderr)
```
- 결과(이름은 statsmodels의 것, 결정 2026-11-18): `params bse tvalues pvalues conf_int(alpha=0.05) cov_params()`, `fittedvalues resid nobs df_model df_resid ssr ess rsquared rsquared_adj fvalue f_pvalue llf aic bic scale`, `rank condition_number`, `predict(X_new)`, `predict_interval(X_new, *, alpha=0.05, kind='mean')`, `summary()`
- 절편은 `params[0]`이다(추론 필드도 같은 자리). 필드는 입력과 상관없이 `sd.ndarray`이고, 계수 이름은 `param_names`(`'const'`, frame의 컬럼 이름 또는 `'x1'…`)다.
- 가중치 0인 행은 적합과 `nobs`·자유도에서 빠지고 적합값·잔차만 계산한다(R과 같다). 음수 가중치는 `ValueError`다.
- 랭크 부족: 앞 열들로 설명되는 열(피벗 없는 QR의 |R_jj| ≤ rtol·‖x_j‖, `rtol` 기본 max(n, k)·eps)의 계수·표준오차·t·p는 NaN이다(R과 같은 선택; R의 허용오차는 1e-7). 적합값·잔차·예측은 정상이다.
- `cov_type`: `'nonrobust'`, `'HC0'`–`'HC3'`, `'HAC'`(Newey–West), `'cluster'`(군집 레이블을 받는다; 인자 이름은 G1에서 정한다).
- 진단: 지렛대, 스튜던트화 잔차, Cook 거리, Durbin–Watson, Jarque–Bera, Breusch–Pagan.
- 배치: `y` (..., n), `X` (..., n, k)이면 결과 필드도 앞에 배치 모양이 붙는다. 배치에서 `nan_policy='omit'`은 `ValueError`다.
- 다항식 적합은 따로 없다: 거듭제곱 열을 만들어 `ols`를 부른다.

## §3. 일반화 선형 (G2)
```
glm(y, X, /, *, family, link=None, weights=None, offset=None, exposure=None, intercept=True,
    max_iter=100, nan_policy='propagate')
```
- `family`(안): `'gaussian'`, `'binomial'`, `'poisson'`, `'gamma'`, `'inverse_gaussian'`, `'negative_binomial'`. `link`가 None이면 정준 링크.
- 결과(안): §2의 추론 필드 + `deviance pearson_chi2 null_deviance llr llr_pvalue converged n_iter`, `resid_deviance resid_pearson resid_response`

## §4. 정규화 (G3)
```
ridge(y, X, /, *, alpha, intercept=True)              # alpha는 스칼라 또는 1차원(여러 값을 한 번에)
lasso(y, X, /, *, alpha, intercept=True, max_iter=…, tol=…)
elastic_net(y, X, /, *, alpha, l1_ratio=0.5, intercept=True, max_iter=…, tol=…)
```
- 경로와 교차검증의 꼴은 G3에서 정한다.
- 결과(안): `params intercept_ … predict(X_new) converged n_iter dual_gap`

## §5. 강건 (G4)
```
rlm(y, X, /, *, norm='huber', intercept=True, max_iter=…)
```

## §6. 분위 (G5)
```
quantreg(y, X, /, *, q=0.5, intercept=True)
```

## §7. 비선형 최소제곱 (G6)
- `curve_fit`/`least_squares` 꼴의 시그니처는 G6에서 정한다.

## §8. 다른 라이브러리와의 대응 (안)
| 하는 일 | statsmodels | scikit-learn | scipy | sd.regression |
|---|---|---|---|---|
| 최소제곱 | `OLS(y, add_constant(X)).fit()` | `LinearRegression().fit(X, y)` | `stats.linregress` (단순) | `ols(y, X)` |
| 가중 | `WLS` | `fit(..., sample_weight=)` | | `wls(y, X, weights=)` |
| GLM | `GLM(y, X, family=)` | `LogisticRegression`, `PoissonRegressor` 등(정규화 포함) | | `glm(y, X, family=)` |
| 정규화 | `OLS.fit_regularized` | `Ridge`, `Lasso`, `ElasticNet`, `…CV` | | `ridge`, `lasso`, `elastic_net` |
| 강건 | `RLM` | `HuberRegressor`, `TheilSenRegressor` | `stats.theilslopes`, `siegelslopes` | `rlm` |
| 분위 | `QuantReg` | `QuantileRegressor` | | `quantreg` |
| 비선형 | | | `optimize.curve_fit`, `least_squares` | G6 |
