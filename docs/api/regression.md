# sd.regression API

> 상태: **G1a 구현** (2026-11-18): §2의 `ols wls gls linregress`, 고전형과 HC0–HC3 공분산, 랭크 부족, 배치. HAC·군집 공분산, 진단, `summary()`는 G1b다. §3부터는 설계 초안이고, 이름·기본값·결과 필드는 마일스톤마다 확정한다([docs/design/regression.md](../design/regression.md) §8).

## §1. 공통 규칙
- 입력은 위치로만(`y`, `X`), 옵션은 키워드로만. 별칭은 없다.
- `y`는 (..., n), `X`는 (..., n, k) 또는 (n,)이다(1차원 `X`는 설명변수 하나). `sd.ndarray`, 실수의 시퀀스, `sd.frame`의 `Series`·`DataFrame`을 받는다. frame의 결측값은 NaN이 되어 `nan_policy`를 따른다. 0차원(Python 스칼라)은 `ValueError`다.
- float64로 계산한다(bool·int64는 바꿔서). complex는 `TypeError`다(`ols(): 'X' must be real, not a complex128 array`).
- **절편은 기본으로 넣는다**(사용자 결정 2026-11-18). `intercept=False`로 끈다. 절편이 있는지는 이 옵션만 보고 정한다. statsmodels처럼 `X`의 상수 열을 찾아 절편으로 보지 않는다(절편 없이 상수 열을 주면 R²는 비중심이다).
- **NaN** (`nan_policy`, sd.stats의 기술통계와 같은 세 값과 같은 메시지): `'propagate'`(기본)는 그 회귀의 모든 값이 NaN이다(`nobs`·`df_model`·`df_resid`도 NaN, `rank`는 0). `'omit'`은 `y`, `X`의 행, 가중치 가운데 NaN이 있는 행을 빼고(적합값·잔차도 남은 행만), `'raise'`는 `ValueError`(`ols(): the input contains NaN (nan_policy='raise')`)다.
- inf는 `nan_policy`와 상관없이 `ValueError`다(`ols(): 'X' contains infinity`; 사용자 결정 2026-11-18).
- **배치**: 앞 축을 배치로 받아 같은 모양의 회귀를 한 번에 푼다. `y`, `X`, `weights`, `sigma`의 배치 모양은 서로 브로드캐스트한다(`X` 하나에 `y` 여럿도 된다). 결과 필드의 앞에 배치 모양이 붙는다. 배치에서 `nan_policy='omit'`은 `ValueError`다. 배치의 결과는 회귀 하나씩 부른 결과와 비트까지 같다.
- 결과 필드는 입력과 상관없이 `sd.ndarray`다. 회귀 하나의 스칼라는 Python 수다(`rank`는 `int`, 나머지는 `float`; statsmodels처럼 `nobs`·`df_model`·`df_resid`도 `float`). 배치면 `rank`는 int64, 나머지는 float64 배열이다.
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 경고는 내지 않는다.
- 에러는 함수 이름으로 시작한다(`ols(): 'X' must have 100 rows like 'y', not 99`). 옵션의 메시지는 다른 영역과 같은 꼴이다(`ols(): 'rtol' must be a finite number >= 0`, `ols(): 'cov_type' must be 'nonrobust', 'HC0', 'HC1', 'HC2' or 'HC3', not 'x'`).
- **정확도** (G15, 사용자 결정 2026-11-18): 계수 벡터의 상대오차(노름) ≤ 1e-14를 κ(X) ≤ 10¹¹에서 약속한다(잔차가 ‖Xβ‖의 100배까지 잰 값). 반복 개선을 필요할 때까지 최대 3번 한다. 그 밖은 잰 값이다: κ 10¹²에서 1.6e-12, 10¹³에서 3.4e-9, 10¹⁴에서 2.5e-7(설계 §3.1). 표준오차는 R에서 계산하므로 상대오차가 κ·u 정도다. 닫힌 꼴은 Arb 참값과 비교한다.

## §2. 선형 (G1)
```
ols(y, X, /, *, intercept=True, cov_type='nonrobust', rtol=None, nan_policy='propagate')
wls(y, X, /, *, weights, intercept=True, cov_type='nonrobust', rtol=None, nan_policy='propagate')
gls(y, X, /, *, sigma, intercept=True, cov_type='nonrobust', rtol=None, nan_policy='propagate')
linregress(x, y, /)                       # -> LinregressResult(slope, intercept, rvalue, pvalue, stderr, intercept_stderr)
```
- 결과는 `OLSResult`, `WLSResult`, `GLSResult`이고 필드 이름은 statsmodels의 것이다(G10).
  - 계수와 추론: `params bse tvalues pvalues`, `conf_int(alpha=0.05)`(…, k, 2: 아래·위 끝), `cov_params()`(…, k, k)
  - 적합: `fittedvalues resid nobs df_model df_resid ssr ess centered_tss uncentered_tss rsquared rsquared_adj fvalue f_pvalue llf aic bic scale`
  - 상태: `rank condition_number param_names cov_type`
  - 예측: `predict(X_new, /)`, `predict_interval(X_new, /, *, alpha=0.05, kind='mean')`(…, m, 2: 아래·위 끝; `kind='observation'`이면 새 관측의 구간)
- 절편은 `params[0]`이다(추론 필드도 같은 자리). 계수 이름 `param_names`는 절편 `'const'`, frame이면 컬럼 이름(`Series`는 그 이름), 그 밖에는 `'x1'…'xk'`다.
- 추론 필드는 처음 읽을 때 계산한다. p값은 t의 두 꼬리 2·`StudentT(df_resid).ccdf(|t|)`, 구간은 `iccdf`, F의 p값은 `F(df_model, df_resid).ccdf`다(sd.stats).
- `rsquared`, `ess`는 R² < 1/2이면 ESS를 직접 더한 값으로, 그 밖은 1 − SSR/TSS로 계산한다(R² ≈ 0에서도 상대 정확도). TSS는 절편이 있으면 `centered_tss`, 없으면 `uncentered_tss`다. `rsquared_adj`는 1 − (nobs − 절편 수)/df_resid·(1 − R²)다. 절편만 남은 모형은 ESS와 R²가 0이고 `fvalue`, `f_pvalue`는 NaN이다(statsmodels와 같다).
- `llf`는 정규 우도의 최댓값이다(statsmodels와 같다; WLS는 ½Σlog w를, GLS는 −½log|Σ|를 더한다). `aic` = −2·llf + 2·rank, `bic` = −2·llf + log(nobs)·rank다.
- `condition_number`는 쓰인 열(가중·백색화한, 절편 열 포함)의 2-노름 조건수다(R의 특이값으로). 겹친 열은 들어가지 않는다.
- **`cov_type`**: `'nonrobust'`(σ̂²(XᵀX)⁻¹), `'HC0'`–`'HC3'`(statsmodels와 같은 샌드위치와 보정). HC에서도 p값과 구간은 t(`df_resid`)로 계산하고(사용자 결정 2026-11-18; R의 coeftest와 같고, statsmodels의 `use_t=True`와 같다), `fvalue`는 절편을 뺀 계수 전부의 Wald 검정 F(분자 자유도는 그 계수 수, 분모는 `df_resid`)다. `'HAC'`(Newey–West)와 `'cluster'`(인자 이름은 G1b에서 정한다)는 G1b다.
- **가중치** (`wls`): 0인 행은 적합과 `nobs`·자유도에서 빠지고 적합값·잔차만 계산한다(R의 `lm`과 같다; statsmodels는 `nobs`에 센다). 음수는 `ValueError`(`wls(): 'weights' must be >= 0`), NaN은 `nan_policy`를 따른다. 가중치가 모두 1이면 `ols`와 비트까지 같다.
- **GLS**: `sigma`는 (…, n, n)인 오차 공분산이고 아래 삼각만 읽는다. Cholesky Σ = LLᵀ로 L⁻¹y, L⁻¹X를 만들어 푼다. `fittedvalues`·`resid`는 원래 척도(y − Xβ)이고, `ssr`과 TSS는 백색화한 값이다(statsmodels와 같다). 양정치가 아니면 `sd.linalg.LinAlgError`(`gls(): 'sigma' is not positive definite`)다. `nan_policy='omit'`은 `sigma`의 i행이나 i열에 NaN이 있으면 관측 i도 뺀다.
- **랭크 부족** (G9, G21): 피벗 없는 QR에서 |R_jj| ≤ rtol·‖x_j‖인 열(가중·백색화한 열)을 겹친다고 보고 빼서 다시 분해한다(R `dqrdc2`와 같은 선택: 앞 열이 남는다). `rtol`의 기본은 max(nobs, k)·eps다(R의 허용오차는 `rtol=1e-7`). 겹친 열의 계수·표준오차·t·p는 NaN이고, `cov_params()`의 그 행과 열도 NaN이다. 나머지 값은 그 열을 뺀 적합과 비트까지 같다. 적합값·잔차·예측은 정상이고(예측에서 겹친 열은 쓰지 않는다), `rank`는 남은 열 수, `df_model`·`df_resid`·`aic`·`bic`는 `rank`로 센다. 절편과 X의 상수 열이 겹치면 절편이 남는다. 열이 행보다 많으면 앞의 nobs개 열까지만 남는다.
- **완전 적합** (잔차가 정확히 0): `bse` 0, `tvalues` ±inf(계수가 0이면 NaN), `pvalues` 0, `llf` +inf다(IEEE의 극한; statsmodels는 잔차가 반올림 수준이라 아주 큰 t와 아주 작은 p를 낸다). `df_resid`가 0이면 `scale`은 inf(또는 0/0이면 NaN)다.
- **`linregress`**: scipy와 같은 꼴(G18). 결과는 Python `float`, `x`와 `y`가 (…, n)이면 배치 모양의 배열이다. NaN은 전파한다. scipy처럼 `x`가 모두 같으면 모든 값이 NaN이고, `y`가 모두 같으면 `rvalue`, `pvalue`, `stderr`, `intercept_stderr`가 NaN이다(기울기 0, 절편 ȳ). 점이 한 직선 위에 있으면 `pvalue`는 0이다(scipy는 1e-20을 더해 9e-11 같은 값을 낸다). `x`가 정확히 같을 때만 기울기가 없다고 본다(`rtol=0`).
- 진단(지렛대, 스튜던트화 잔차, Cook 거리, Durbin–Watson, Jarque–Bera, Breusch–Pagan)과 `summary()`는 G1b다.
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
