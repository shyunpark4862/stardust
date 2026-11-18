# sd.regression API

> 상태: **G1–G3 구현** (2026-11-18): §2의 `ols wls gls linregress`, 고전형·HC0–HC3·HAC·군집 공분산, 랭크 부족, 배치, 진단, `summary(*, alpha=0.05)`; §3의 `glm`; §4의 `ridge lasso elastic_net`, 경로, `ridge_cv lasso_cv elastic_net_cv`, 정규화 GLM `penalized_glm`. §5부터는 설계 초안이고, 이름·기본값·결과 필드는 마일스톤마다 확정한다([docs/design/regression.md](../design/regression.md) §8).

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
ols(y, X, /, *, intercept=True, cov_type='nonrobust', maxlags=None, groups=None, rtol=None, nan_policy='propagate')
wls(y, X, /, *, weights, intercept=True, cov_type='nonrobust', maxlags=None, groups=None, rtol=None, nan_policy='propagate')
gls(y, X, /, *, sigma, intercept=True, cov_type='nonrobust', maxlags=None, groups=None, rtol=None, nan_policy='propagate')
linregress(x, y, /)                       # -> LinregressResult(slope, intercept, rvalue, pvalue, stderr, intercept_stderr)
```
- 결과는 `OLSResult`, `WLSResult`, `GLSResult`이고 필드 이름은 statsmodels의 것이다(G10).
  - 계수와 추론: `params bse tvalues pvalues`, `conf_int(alpha=0.05)`(…, k, 2: 아래·위 끝), `cov_params()`(…, k, k)
  - 적합: `fittedvalues resid nobs df_model df_resid ssr ess centered_tss uncentered_tss rsquared rsquared_adj fvalue f_pvalue llf aic bic scale`
  - 상태: `rank condition_number param_names cov_type`
  - 예측: `predict(X_new, /)`, `predict_interval(X_new, /, *, alpha=0.05, kind='mean')`(…, m, 2: 아래·위 끝; `kind='observation'`이면 새 관측의 구간)
  - 진단: `leverage resid_studentized_internal resid_studentized_external cooks_distance durbin_watson`, `jarque_bera()`(→ `JarqueBeraResult(statistic, pvalue)`), `breusch_pagan(*, studentize=True)`(→ `BreuschPaganResult(lm, lm_pvalue, fvalue, f_pvalue)`, statsmodels `het_breuschpagan`의 필드와 순서)
  - `summary(*, alpha=0.05)`: 글(`str`)
- 절편은 `params[0]`이다(추론 필드도 같은 자리). 계수 이름 `param_names`는 절편 `'const'`, frame이면 컬럼 이름(`Series`는 그 이름), 그 밖에는 `'x1'…'xk'`다.
- 추론 필드는 처음 읽을 때 계산한다. p값은 t의 두 꼬리 2·`StudentT(df_resid).ccdf(|t|)`, 구간은 `iccdf`, F의 p값은 `F(df_model, df_resid).ccdf`다(sd.stats).
- `rsquared`, `ess`는 R² < 1/2이면 ESS를 직접 더한 값으로, 그 밖은 1 − SSR/TSS로 계산한다(R² ≈ 0에서도 상대 정확도). TSS는 절편이 있으면 `centered_tss`, 없으면 `uncentered_tss`다. `rsquared_adj`는 1 − (nobs − 절편 수)/df_resid·(1 − R²)다. 절편만 남은 모형은 ESS와 R²가 0이고 `fvalue`, `f_pvalue`는 NaN이다(statsmodels와 같다).
- `llf`는 정규 우도의 최댓값이다(statsmodels와 같다; WLS는 ½Σlog w를, GLS는 −½log|Σ|를 더한다). `aic` = −2·llf + 2·rank, `bic` = −2·llf + log(nobs)·rank다.
- `condition_number`는 쓰인 열(가중·백색화한, 절편 열 포함)의 2-노름 조건수다(R의 특이값으로). 겹친 열은 들어가지 않는다.
- **`cov_type`**: `'nonrobust'`(σ̂²(XᵀX)⁻¹), `'HC0'`–`'HC3'`(statsmodels와 같은 샌드위치와 보정), `'HAC'`, `'cluster'`. 강건 공분산에서도 p값과 구간은 t(`df_resid`)로 계산하고(사용자 결정 2026-11-18; R의 coeftest와 같고, statsmodels의 `use_t=True`와 같다), `fvalue`는 절편을 뺀 계수 전부의 Wald 검정 F(분자 자유도는 그 계수 수, 분모는 `df_resid`)다. 군집만 자유도가 다르다(아래).
  - **`'HAC'`** (Newey–West, G26): 지연 수는 `maxlags`(0 이상의 int)이고, None이면 floor(4(nobs/100)^(2/9))다(statsmodels의 `S_hac_simple`과 같은 식). 커널은 Bartlett(가중치 1 − l/(L+1))이고 소표본 보정은 하지 않는다(statsmodels의 기본 `use_correction=False`, R `sandwich::NeweyWest(adjust=FALSE)`와 같다; Stata의 `newey`는 n/(n−k)를 곱한다). 자동 대역폭 선택은 없다. 행은 주어진 순서가 시간 순서다. `maxlags`를 다른 `cov_type`과 주면 `ValueError`(`ols(): 'maxlags' is only used when cov_type='HAC'`)다.
  - **`'cluster'`** (일원 군집, G27): `groups`는 행마다 군집 레이블이다. int·bool·float의 `sd.ndarray`(배치 축이 있으면 회귀마다 다른 레이블), 정수·실수·문자열의 시퀀스, frame `Series`를 받고, 같은지만 본다. 소표본 보정 G/(G−1)·(nobs−1)/(nobs−rank)를 곱하고, t의 p값·구간·`predict_interval`과 F 검정의 분모 자유도는 G − 1이다(Stata와 같고, statsmodels의 `use_t=True`와 같다; statsmodels의 `get_prediction`은 이때도 `df_resid`를 쓴다). `df_resid`는 그대로 nobs − rank다. G는 적합에 쓰인 행(가중치 > 0)에 나오는 군집 수다. 군집이 하나면 공분산은 NaN이다. 군집 수 − 1이 검정할 계수 수보다 작으면 공분산이 특이해서 `fvalue`가 NaN이다. `groups` 없이 `'cluster'`면 `ValueError`(`ols(): 'groups' is required when cov_type='cluster'`), 다른 `cov_type`에 `groups`를 주면 `ValueError`(`ols(): 'groups' is only used when cov_type='cluster'`), 길이가 다르면 `ValueError`(`ols(): 'groups' must have 100 values like 'y', not 99`)다. 레이블이 없는 행(float의 NaN, 시퀀스의 None·NaN, frame의 NA)은 NaN 데이터처럼 `nan_policy`를 따른다.
  - 가중치 0인 행은 HAC의 지연 합과 군집 합에서도 빠진다(적합에 쓰인 행이 그 순서로 이어진다). 그래서 0인 행이 있는 WLS는 그 행을 뺀 WLS와 비트까지 같다(statsmodels는 그 행을 점수 0인 시점으로 두고, 그 행에만 있는 군집도 센다).
- **가중치** (`wls`): 0인 행은 적합과 `nobs`·자유도에서 빠지고 적합값·잔차만 계산한다(R의 `lm`과 같다; statsmodels는 `nobs`에 센다). 음수는 `ValueError`(`wls(): 'weights' must be >= 0`), NaN은 `nan_policy`를 따른다. 가중치가 모두 1이면 `ols`와 비트까지 같다.
- **GLS**: `sigma`는 (…, n, n)인 오차 공분산이고 아래 삼각만 읽는다. Cholesky Σ = LLᵀ로 L⁻¹y, L⁻¹X를 만들어 푼다. `fittedvalues`·`resid`는 원래 척도(y − Xβ)이고, `ssr`과 TSS는 백색화한 값이다(statsmodels와 같다). 양정치가 아니면 `sd.linalg.LinAlgError`(`gls(): 'sigma' is not positive definite`)다. `nan_policy='omit'`은 `sigma`의 i행이나 i열에 NaN이 있으면 관측 i도 뺀다.
- **랭크 부족** (G9, G21): 피벗 없는 QR에서 |R_jj| ≤ rtol·‖x_j‖인 열(가중·백색화한 열)을 겹친다고 보고 빼서 다시 분해한다(R `dqrdc2`와 같은 선택: 앞 열이 남는다). `rtol`의 기본은 max(nobs, k)·eps다(R의 허용오차는 `rtol=1e-7`). 겹친 열의 계수·표준오차·t·p는 NaN이고, `cov_params()`의 그 행과 열도 NaN이다. 나머지 값은 그 열을 뺀 적합과 비트까지 같다. 적합값·잔차·예측은 정상이고(예측에서 겹친 열은 쓰지 않는다), `rank`는 남은 열 수, `df_model`·`df_resid`·`aic`·`bic`는 `rank`로 센다. 절편과 X의 상수 열이 겹치면 절편이 남는다. 열이 행보다 많으면 앞의 nobs개 열까지만 남는다.
- **완전 적합** (잔차가 정확히 0): `bse` 0, `tvalues` ±inf(계수가 0이면 NaN), `pvalues` 0, `llf` +inf다(IEEE의 극한; statsmodels는 잔차가 반올림 수준이라 아주 큰 t와 아주 작은 p를 낸다). `df_resid`가 0이면 `scale`은 inf(또는 0/0이면 NaN)다.
- **`linregress`**: scipy와 같은 꼴(G18). 결과는 Python `float`, `x`와 `y`가 (…, n)이면 배치 모양의 배열이다. NaN은 전파한다. scipy처럼 `x`가 모두 같으면 모든 값이 NaN이고, `y`가 모두 같으면 `rvalue`, `pvalue`, `stderr`, `intercept_stderr`가 NaN이다(기울기 0, 절편 ȳ). 점이 한 직선 위에 있으면 `pvalue`는 0이다(scipy는 1e-20을 더해 9e-11 같은 값을 낸다). `x`가 정확히 같을 때만 기울기가 없다고 본다(`rtol=0`).
- **진단** (G17, G28, G29): 모두 커널이 푸는 백색화한 문제의 것이다(WLS는 √w y와 √w X, GLS는 L⁻¹y와 L⁻¹X; 강건 공분산과 같다). ẽ는 백색화한 잔차(OLS는 `resid`, WLS는 √w·`resid`, GLS는 L⁻¹`resid`)다.
  - `leverage`: 지렛대 h_t(hat 행렬의 대각, …, n). 적합의 QR에서 얇은 Q를 만들어 h_t = ‖q_t‖²로 구한다(XᵀX를 만들지 않는다). 처음 읽을 때 같은 열로 QR을 한 번 더 한다.
  - `resid_studentized_internal`: ẽ_t / (s√(1 − h_t)), s² = `scale`(statsmodels의 이름, R의 `rstandard`). `resid_studentized_external`: 행 t를 뺀 척도 s_(t)로 나눈 값으로, r√((df − 1)/(df − r²))로 계산한다(r은 내적, df는 `df_resid`; R의 `rstudent`).
  - `cooks_distance`: r_t²/rank · h_t/(1 − h_t)(R의 `cooks.distance`; statsmodels의 `cooks_distance[0]`은 rank 대신 X의 열 수로 나눠 겹친 열이 있을 때만 다르다).
  - `durbin_watson`: Σ(ẽ_t − ẽ_{t−1})² / Σẽ_t², 적합에 쓰인 행의 순서로(double-double 합).
  - `jarque_bera()`: n/6·(S² + (K − 3)²/4), S·K는 ẽ의 중심 적률(n으로 나눈)로 만든 왜도·첨도, p값은 자유도 2의 카이제곱이다(statsmodels의 `jarque_bera(wresid)`와 같다; 절편이 있으면 ẽ의 평균은 0이다).
  - `breusch_pagan(*, studentize=True)`: ẽ²를 상수와 백색화한 설계의 열(OLS는 X; WLS는 √w와 √w X; GLS는 L⁻¹1과 L⁻¹X)에 회귀하는 보조 회귀다. `studentize=True`(Koenker, statsmodels·R lmtest의 기본)는 lm = n·R², `False`(원래 판)는 lm = ESS/(2σ̂⁴), σ̂² = `ssr`/n다. lm의 p값은 자유도가 보조 회귀의 `df_model`인 카이제곱이고, `fvalue`·`f_pvalue`는 보조 회귀의 F 검정이다. 보조 회귀는 모형의 절편과 상관없이 상수를 넣는다(statsmodels의 `het_breuschpagan`은 주어진 설계를 그대로 쓰므로 절편이 있는 모형에서 같다). 보조 회귀의 열이 겹치면 G1의 규칙대로 빠지고 자유도도 줄어든다.
  - 가중치 0인 행: `leverage`, 두 스튜던트화 잔차, `cooks_distance`는 가중 식대로 0이고, `durbin_watson`, `jarque_bera()`, `breusch_pagan()`의 합에서는 빠진다(그 행을 뺀 WLS와 같다).
  - 겹친 열: 그 열을 뺀 적합과 같다. 완전 적합(잔차가 정확히 0): 스튜던트화 잔차, Cook 거리, `durbin_watson`, `jarque_bera()`, `breusch_pagan()`은 0/0이라 NaN이고 `leverage`는 정상이다. NaN이 있는 회귀(propagate)는 모두 NaN이다. 배치는 필드가 배치 모양을 갖는다(`jarque_bera()`·`breusch_pagan()`의 각 값도).
- **`summary(*, alpha=0.05)`** (G13): statsmodels의 OLS summary와 같은 꼴의 글이다. 계수 표의 구간은 1 − alpha다. 위에 모형(OLS·WLS·GLS), 방법, 관측 수, 자유도, 공분산 종류, R²·수정 R², F와 p값, 로그우도, AIC, BIC; 가운데에 계수 표(`param_names`, 계수, 표준오차, t, P>|t|, 95% 구간); 아래에 Durbin–Watson, Jarque–Bera와 그 p값, 조건수(`condition_number`); 끝에 공분산 종류의 설명(HAC는 지연 수, 군집은 군집 수와 자유도)과 겹친 열의 이름이 있다. 숫자의 꼴은 statsmodels와 같다(R² `%.3f`, 계수 소수 4자리 등). 폭은 78자이고 계수 이름이 길면 넓어진다. 날짜·시각은 넣지 않는다. 배치 결과는 `ValueError`(`summary(): takes one regression, not a batch of shape (7,)`)다. `repr`는 한 줄 요약 그대로다.
- 다항식 적합은 따로 없다: 거듭제곱 열을 만들어 `ols`를 부른다.

- `df_resid_inference`: p값과 구간에 쓰는 t의 자유도(F의 분모 자유도). 보통 `df_resid`, `cov_type='cluster'`이면 군집 수 − 1이다.

## §3. 일반화 선형 (G2)
```
glm(y, X, /, *, family, link=None, var_weights=None, freq_weights=None, trials=None, offset=None, exposure=None, intercept=True, scale=None, cov_type='nonrobust', maxlags=None, groups=None, max_iter=100, tol=1e-08, rtol=None, nan_policy='propagate')
```
- 결과는 `GLMResult`이고 필드 이름은 statsmodels의 것이다(G10). §1의 공통 규칙(입력, 절편, NaN, inf, 배치, 스레드, 에러의 꼴)을 따른다.
- **분포족과 링크** (G31): `family`는 `'gaussian'`, `'binomial'`, `'poisson'`, `'gamma'`이고 `link`는 분포족마다 다음 가운데 하나다. None이면 정준 링크(맨 앞)다.

  | `family` | `link` | 분산 V(μ) | `y` |
  |---|---|---|---|
  | `'gaussian'` | `'identity'`, `'log'` | 1 | 실수 |
  | `'binomial'` | `'logit'`, `'probit'`, `'cloglog'`, `'log'` | μ(1 − μ) | 0 또는 1, `trials`가 있으면 [0, 1]의 비율 |
  | `'poisson'` | `'log'`, `'identity'`, `'sqrt'` | μ | ≥ 0 (정수가 아니어도 된다) |
  | `'gamma'` | `'inverse'`, `'log'`, `'identity'` | μ² | > 0 |

  - 이름은 문자열로만 받는다(`family='binomial', link='probit'`; statsmodels처럼 객체를 받지 않는다). 다른 이름은 `ValueError`(`glm(): 'link' must be None, 'logit', 'probit', 'cloglog' or 'log' for the binomial family, not 'identity'`)다.
  - `y`가 분포족의 범위 밖이면 가중치와 상관없이 `ValueError`다(`glm(): 'y' must be 0 or 1 for the binomial family without 'trials'`; R과 같다).
- **가중치** (G36): `var_weights`(분산이 V(μ)/w, R의 `weights`)와 `freq_weights`(행마다 같은 관측의 수). 둘 다 ≥ 0이고 정수가 아니어도 된다. `freq_weights`는 행을 그 수만큼 반복한 적합과 같다: `nobs`는 적합에 쓰인 행의 `freq_weights` 합이고 `df_resid`, `llf`, `aic`, `bic`, 강건 공분산도 반복한 자료의 값이다(statsmodels의 `nobs`는 행 수이고 이 합은 `wnobs`다). 가중치(`var_weights`·`freq_weights`·`trials`의 곱)가 0인 행은 적합과 `nobs`·자유도에서 빠지고 적합값·잔차만 계산한다(G20; statsmodels는 `var_weights`가 0인 행을 `nobs`에 센다). 음수는 `ValueError`, NaN은 `nan_policy`를 따른다.
- **이항 반응** (G35): `trials`(> 0, 이항 분포족만)를 주면 `y`는 성공의 비율이다. 시행 수는 R의 사전 가중치처럼 분산과 편차에 곱해지고, `llf`에는 log C(n, ny)가 들어간다(statsmodels의 (성공, 실패) 두 열 반응과 같은 값).
- **오프셋과 노출**: η = Xβ + `offset` + log(`exposure`)다. `exposure`는 > 0이고 log 링크에서만 받는다(statsmodels와 같다). `predict`·`predict_interval`에도 새 행의 `offset`·`exposure`를 준다(없으면 0).
- **척도** (G34): `scale=None`은 이항·포아송 1, 정규·감마 Pearson χ²/`df_resid`다. `'X2'`는 Pearson χ²/`df_resid`, `'dev'`는 편차/`df_resid`, 양수는 그 값으로 고정한다(이항·포아송에도 준다: 과대산포의 준우도).
- **검정 분포** (G38, G39): 척도가 고정이면(이항·포아송의 기본, 또는 수로 준 `scale`) 정규분포(z), 추정하면(정규·감마의 기본, `'X2'`, `'dev'`) t(`df_resid`)다(R의 `summary.glm`과 같다). 강건 공분산(G39)은 HC0–HC3·HAC에서 척도가 고정인 이항·포아송이면 정규분포(R의 `coeftest`·Stata와 같고, statsmodels의 `use_t=False`와 같다), 정규·감마와 척도를 추정하는 이항·포아송(`'X2'`, `'dev'`)이면 t(`df_resid`)이고, 군집은 분포족과 상관없이 t(G − 1)다(G27). `df_resid_inference`는 그 자유도이고 정규분포면 inf다. `summary()`의 표 머리는 `z`·`P>|z|` 또는 `t`·`P>|t|`다.
- **결과 필드**:
  - 계수와 추론: `params bse tvalues pvalues`, `conf_int(alpha=0.05)`, `cov_params()`(척도 × (XᵀWX)⁻¹, W는 기대 정보(Fisher)의 가중치), `df_resid_inference`
  - 적합: `fittedvalues`(μ), `linear_predictor`(η), `resid_response`(y − μ, `y`의 척도), `resid_pearson`((y − μ)√(m/V(μ)), m = `var_weights` × `trials`), `resid_deviance`(sign(y − μ)√(m·d(y, μ))), `resid_working`((y − μ)g′(μ))
  - 적합도: `deviance null_deviance pearson_chi2 scale llf llnull llr llr_pvalue aic bic nobs df_model df_resid`
  - 상태: `rank condition_number converged n_iter separation param_names family link cov_type`
  - 예측: `predict(X_new, /, *, offset=None, exposure=None, which='mean')`(`'linear'`이면 η), `predict_interval(X_new, /, *, offset=None, exposure=None, alpha=0.05, which='mean')`: η ± q·se(η)를 역링크로 옮긴 구간(…, m, 2; 역링크가 감소하는 inverse 링크에서도 아래·위 순서)
  - `summary(*, alpha=0.05)`: statsmodels의 GLM summary와 같은 꼴의 글(날짜·시각은 넣지 않는다)
- `llf`는 statsmodels의 `loglike`(상수항 포함, 척도는 `scale`)다. 정규·identity는 척도를 편차/`nobs`로 둔 집중 로그우도라 `ols`의 `llf`와 같다. `aic` = −2·`llf` + 2·`rank`, `bic` = −2·`llf` + log(`nobs`)·`rank`(statsmodels의 `bic_llf`; 편차로 만든 `bic_deviance`는 두지 않는다). 척도는 모수로 세지 않는다(statsmodels와 같다; R의 AIC는 정규·감마에서 1을 더 센다).
- 귀무 모형은 절편만 있는 모형이다(오프셋이 있으면 그 오프셋과 함께 다시 적합한다). `intercept=False`이면 귀무 모형은 모수가 없는 η = `offset`이다(R의 `glm`; statsmodels는 이때도 상수를 적합한다). `null_deviance`, `llnull`(같은 척도, 정규·identity는 집중 로그우도)이 그 값이고, `llr` = 2(`llf` − `llnull`), `llr_pvalue`는 자유도 `df_model`의 카이제곱이다.
- **`cov_type`** (G37): §2와 같은 이름과 옵션이다. 강건 공분산은 점수 s_i = x_i·m_i(y_i − μ_i)μ′_i/V(μ_i)의 샌드위치이고, 빵은 관측 정보(로그우도의 헤시안)의 역행렬이다(statsmodels와 같다; 정준 링크에서는 기대 정보와 같다). `'HC1'`은 `'HC0'`에 nobs/(nobs − rank)를, `'HC2'`·`'HC3'`는 √W X의 지렛대 h로 1/(1 − h)·1/(1 − h)²를 곱한다(§2의 OLS와 같은 뜻; statsmodels의 GLM은 `'HC1'`–`'HC3'`에도 `'HC0'`을 낸다). `freq_weights`는 반복한 행이 각각 점수를 내는 것으로 센다(HC·HAC, 군집은 같은 군집의 합).
- **수렴** (G32): IRLS가 편차의 상대 변화 |D − D_old|/(|D| + 0.1) < `tol`(R의 `glm.fit`과 같은 판정, 기본 1e-8)에 닿으면 `converged=True`다. 판정 전의 반복은 정규방정식(Cholesky; 조건이 나쁘거나 G21이 열을 뺄 때, 또 행 수 × 열 수가 512 미만이면 QR)으로 푼다(G40, G43). 판정 뒤에도 계수가 반올림보다 크게 움직이는 동안 QR로 반복을 이어 가고 마지막에 반복 개선을 한 최소제곱을 한 번 더 풀므로(정준이 아닌 링크는 이 둘이 관측 정보의 뉴턴 걸음이고 얇은 Q로 푼다, G42; 판정 전에도 Fisher 걸음이 직전의 1/2 아래로 줄지 않는 반복이 3번 이어지면 그때부터 뉴턴 걸음이다, G44), 계수는 그 판정의 정밀도가 아니라 최대우도 추정에 반올림 수준으로 가깝다(설계 §3.4). `n_iter`는 최소제곱을 푼 횟수다(마지막 풀이 포함; R은 그 풀이를 세지 않는다). `max_iter`(≥ 1)에 닿으면 마지막 값을 내고 `converged=False`, `n_iter`는 `max_iter`다. 경고는 없다.
- **분리** (G33, 이항과 로그 링크의 포아송): 최대우도 추정이 존재하지 않으면(완전·준완전 분리, 0만 있는 포아송 군 등) `separation=True`이고 `params`·`bse`·`tvalues`·`pvalues`·`cov_params()`는 NaN이다. 경계로 가는 행의 `fittedvalues`는 y(0 또는 1, 포아송은 0), `linear_predictor`는 ±inf, 잔차는 0이다. 나머지 행은 그 행들만의 적합(가능도 최대화의 극한)이고 `deviance`·`llf`·`pearson_chi2`도 그 극한의 값이다. 판정은 설계 §3.4의 증명서다. 이항·포아송 밖이나 다른 링크(경계에서 유한한 최대우도 추정, 예: 포아송 identity에서 μ = 0)는 분리가 아니다.
- 겹친 열(G9, G21)·`rtol`·배치·`nan_policy`는 §2와 같다. 겹친 열의 계수와 추론은 NaN이고 `rank`, `df_model`, `df_resid`, `aic`, `bic`는 남은 열로 센다.
- **정확도** (잰 값, 설계 §3.4): 계수 벡터의 상대오차(노름)는 Arb 참값과 견줘 분포족·링크 12개와 가중치·시행 수·노출·절편 없음·`scale='dev'`의 무작위 문제 17개에서 최대 1.1e-16이다. √W X의 조건수 κ가 클 때는 정준 링크(로지스틱·포아송 log) κ ≤ 10¹⁰에서 1.3e-15, 10¹²에서 9e-12, 정준이 아닌 링크 8개(probit, cloglog, 이항 log, 포아송 identity·sqrt, 감마 log·identity, 정규 log) κ ≤ 10⁸에서 3.5e-15, 10¹⁰에서 4e-14다. 같은 17개에서 표준오차·편차·Pearson χ²·로그우도·강건 표준오차는 1e-15 안, p값은 2e-13 안이다. 표준오차는 G1처럼 R에서 계산하므로 상대오차가 κ·u 정도다.

## §4. 정규화 (G3)
```
ridge(y, X, /, *, alpha, weights=None, intercept=True, standardize=False, nan_policy='propagate')
lasso(y, X, /, *, alpha, weights=None, intercept=True, standardize=False, max_iter=1000, tol=0.0001, nan_policy='propagate')
elastic_net(y, X, /, *, alpha, l1_ratio=0.5, weights=None, intercept=True, standardize=False, max_iter=1000, tol=0.0001, nan_policy='propagate')
ridge_cv(y, X, /, *, rng=None, cv=5, alphas=100, eps=0.001, select='min', weights=None, intercept=True, standardize=False, nan_policy='propagate')
lasso_cv(y, X, /, *, rng, cv=5, alphas=100, eps=0.001, select='min', weights=None, intercept=True, standardize=False, max_iter=1000, tol=0.0001, nan_policy='propagate')
elastic_net_cv(y, X, /, *, rng, l1_ratio=0.5, cv=5, alphas=100, eps=0.001, select='min', weights=None, intercept=True, standardize=False, max_iter=1000, tol=0.0001, nan_policy='propagate')
penalized_glm(y, X, /, *, family, alpha, l1_ratio=0.5, weights=None, intercept=True, standardize=False, max_iter=100, tol=1e-08, nan_policy='propagate')
```
- §1의 공통 규칙(입력, 절편, NaN, inf, 배치, 스레드, 에러의 꼴)을 따른다. 정규화 GLM의 이름은 `penalized_glm`, 결과는 `PenalizedGLMResult`다(G49).
- **목적함수** (G45, scikit-learn과 같다): (1/(2·sw))·Σ wᵢ(yᵢ − b₀ − xᵢβ)² + alpha·(l1_ratio·‖Sβ‖₁ + (1 − l1_ratio)/2·‖Sβ‖²). sw = Σw이고, 가중치가 없으면 wᵢ = 1(sw = n)이다. `ridge`는 l1_ratio = 0, `lasso`는 1이다. 절편 b₀은 벌점 밖이다(가중 평균으로 중심화한다). S는 `standardize=False`이면 I, `True`이면 열의 표준편차의 대각이다(아래).
  - 가중치는 scikit-learn의 `sample_weight`와 같은 뜻이다(scikit-learn은 가중치를 합이 n이 되게 다시 맞추므로 같은 목적함수다). 0인 행은 적합과 `nobs`에서 빠지고 적합값·잔차만 계산한다(G20). 음수는 `ValueError`, NaN은 `nan_policy`를 따른다.
  - scikit-learn과의 대응: `Lasso`·`ElasticNet`·`enet_path`의 `alpha`는 같다. `Ridge`는 Σw r² + alpha‖β‖²을 풀므로 그 `alpha`는 여기의 sw배다. `LogisticRegression`의 `C`는 1/(sw·alpha), `PoissonRegressor`의 `alpha`는 l1_ratio = 0에서 같다.
- **`standardize=True`** (glmnet과 같다): 표준화한 계수 sⱼβⱼ에 벌점을 주고 원래 척도의 계수를 돌려준다. sⱼ는 모표준편차 √(Σw(xⱼ − x̄ⱼ)²/sw)이고, 절편이 없으면 √(Σw xⱼ²/sw)다(glmnet의 규칙으로 알고 있지만 R이 이 환경에 없어 확인하지 못했다). 그래서 X의 열에 상수를 곱해도 예측은 바뀌지 않는다.
- 상수 열(절편이 없으면 0인 열)은 적합에서 빠지고 그 계수는 0이다.
- **`alpha`**: 0 이상의 유한한 수, 또는 그런 수의 1차원 배열(경로, G46). 배열이면 큰 값부터 앞의 해에서 시작해(따뜻한 시작) 모두 풀고, 결과는 준 순서다. 그때 회귀 하나의 필드마다 배치 축 뒤, 자기 축 앞에 길이 L의 축이 붙는다: `params` (..., L, k), 스칼라 필드 (..., L), `fittedvalues`·`resid` (..., L, n), `predict` (..., L, m). 결과의 `alpha`는 준 값(수 또는 배열)이다. `ridge`·`lasso`·`elastic_net`의 경로의 각 해는 같은 alpha로 따로 푼 것과 비트까지 같다(lasso·엘라스틱 넷은 연마한 해의 활성 집합과 부호가 같으면; 아래 연마). 정규화 GLM은 시작점에 따라 반올림 수준에서 다를 수 있다.
- **`ridge`의 풀이** (G52): 정규방정식 XᵀX를 만들지 않고 중심화한 설계의 QR과 R의 SVD 하나로 모든 alpha를 푼다(설계 §3.5). 해의 조건수가 크면 double-double 잔차로 개선한다(G51).
- **결과** (G48, 예측 중심): `RidgeResult`, `LassoResult`, `ElasticNetResult`, `PenalizedGLMResult`와 교차검증의 `RidgeCVResult`, `LassoCVResult`, `ElasticNetCVResult`(각각 앞의 것에 교차검증 필드를 더한 것).
  - 공통: `params`(절편 `params[0]`), `param_names`(§2와 같다), `alpha`, `l1_ratio`, `standardize`, `nobs`(가중치가 0보다 큰 행 수, `float`), `df_model`, `predict(X_new, /)`
  - 선형(`ridge`·`lasso`·`elastic_net`): `fittedvalues`, `resid`(y − 적합값, double-double로 한 번 반올림), `ssr`(Σw r², 적합에 쓰인 행), `rsquared`(1 − `ssr`/TSS, TSS는 가중 평균에 대한 Σw(y − ȳ)²이고 절편이 없으면 Σw y²: `ols`와 같은 규칙. 절편이 있으면 scikit-learn의 `score(X, y, sample_weight=w)`와 같다)
  - 좌표 하강(`lasso`·`elastic_net`·정규화 GLM): `converged`, `n_iter`, `dual_gap`. `ridge`는 닫힌 꼴이라 이 셋이 없다.
  - 정규화 GLM: `family`, `separation`, `fittedvalues`(μ), `linear_predictor`(η), `resid_response`(y − μ), `deviance`, `null_deviance`(절편만의 모형, 절편이 없으면 η = 0), `deviance_ratio`(1 − `deviance`/`null_deviance`, glmnet의 `dev.ratio`), `predict(X_new, /, *, which='mean')`(`'linear'`이면 η)
  - 표준오차·p값·구간 필드와 `summary()`는 없다(G48, G52).
- **`df_model`** (유효 자유도, 절편은 세지 않는다): 0 아닌 계수의 집합 A에 대해 tr Z_A(Z_AᵀΩZ_A + alpha(1 − l1_ratio)I)⁻¹Z_AᵀΩ다(Z는 중심화한 열, `standardize`면 표준화한 열, Ω = diag(w/sw); 정규화 GLM은 Ω에 작업 가중치를 곱한다). lasso(l1_ratio = 1)는 0 아닌 계수의 수, ridge(A가 모든 열)는 Σ s²/(s² + alpha)(s는 Ω^½Z의 특이값)다. 엘라스틱 넷의 이 식은 Zou–Hastie(2005)의 것이다(glmnet의 `df`는 0 아닌 계수의 수).
- **`lasso`·`elastic_net`의 수렴** (scikit-learn과 같은 판정): 열 순서의 좌표 하강이고, 쌍대 간격이 `tol` × Σω(y − ȳ)²(절편이 없으면 Σω y²) 이하이면 수렴이다. scikit-learn의 tol·‖y‖²(중심화·가중한 y, 그 목적함수의 n배 척도)와 같은 뜻이다. 간격은 가장 큰 계수 변화가 `tol` × 가장 큰 계수 이하인 전체 스윕 뒤에 잰다(scikit-learn과 같다). `n_iter`는 스윕 수(전체 스윕과 0 아닌 계수만의 스윕을 모두 센다), `max_iter`는 그 상한이다(alpha마다). 상한에 닿으면 마지막 값을 내고 `converged=False`다(G32). 경고는 없다.
  - 수렴한 해는 KKT 연마를 거친다: 그 해의 0 아닌 계수와 부호로 정상 조건의 연립방정식을 풀고, 부호가 그대로이고 0인 계수마다 |zⱼᵀΩr| ≤ alpha·l1_ratio이면 그 해를 낸다(아니면 좌표 하강을 더 작은 허용오차로 이어 간 뒤 다시 연마하고, 끝내 통과하지 못하면 좌표 하강의 해를 낸다). 그래서 계수는 `tol`의 정밀도가 아니라 정확한 해에 반올림 수준으로 가깝고(아래 정확도), 해가 0인 계수는 정확히 0이다.
  - `dual_gap`은 목적함수 단위의 쌍대 간격이다(scikit-learn의 `dual_gap_`은 sw배 큰 척도). 연마한 해에서는 반올림 수준이고 그 크기의 음수일 수도 있다.
- **정규화 GLM** (G47): `family`는 `'binomial'`(logit 링크, `y`는 0 또는 1)과 `'poisson'`(log 링크, `y` ≥ 0)이다. 정준 링크만이다. 목적함수는 편차/(2·sw) + 같은 벌점이다(glmnet과 같다; 가중 편차는 Σwᵢdᵢ). 바깥은 IRLS(현재 계수에서 이차 근사)이고 안쪽은 같은 좌표 하강이다. 목적함수 F의 상대 변화 |F − F_old|/(|F| + 0.1)가 `tol`(기본 1e-8, `glm`과 같은 꼴) 아래면 수렴이다. 목적함수가 오르면 걸음을 반씩 줄인다. 수렴한 해는 0 아닌 계수와 부호를 고정한 뉴턴법으로 연마한다(선형과 같은 KKT 검사). `n_iter`는 IRLS 반복 수이고 `max_iter`(기본 100)가 그 상한이다. `dual_gap`은 돌려준 계수에서의 마지막 이차 근사 문제의 쌍대 간격(목적함수 단위)이다. y가 모두 0(이항은 모두 0 또는 모두 1)이면 유한한 해가 없어 계수가 NaN이고 `converged=False`다. 링크 인자와 `offset`·`trials`는 없다(G52).
  - **alpha = 0** (G50): 벌점이 없으면 `glm`의 적합 그대로다(`var_weights=weights`; 같은 `max_iter`·`tol`). 분리는 `glm`의 증명서로 판정하고(G33), 그때 `params`는 NaN, `separation=True`, `linear_predictor`·`fittedvalues`·`deviance`는 `glm`의 극한값(경계 행의 η = ±inf, μ = 0 또는 1)이며 `dual_gap`은 NaN이다. 분리가 없으면 `params`·`deviance`·`n_iter`·`converged`·`df_model`이 `glm`의 것이다(겹친 열도 G9대로 NaN). 경로에서는 alpha가 0인 자리만 그렇다. alpha > 0이면 목적함수가 계수와 함께 커져 분리돼도 해가 유한하므로 판정하지 않고 `separation`은 False다.
- **교차검증** (G46): k겹으로 alpha를 고르고 모든 데이터로 다시 적합한다.
  - `cv`는 겹 수(2 이상, 행 수 이하; 기본 5, scikit-learn의 기본과 같다; glmnet은 10)다. 겹은 `rng`(`sd.random.default_rng(seed)`의 Generator, `sample`처럼 꼭 준다: `lasso_cv`·`elastic_net_cv`는 키워드 인자에 기본값이 없고, `ridge_cv`는 `cv='gcv'`일 때만 없어도 된다)의 `rng.permutation(n)`로 나눈다: 행 i의 겹은 (0, 1, …, n − 1 mod cv)를 그 순열로 섞은 값이다(glmnet의 `sample(rep(1:K, length = n))`과 같은 꼴, 겹 크기의 차이는 1 이하). 행은 `nan_policy='omit'`이 뺀 뒤의 행이다. 가중치 0인 행도 겹을 받지만 적합과 오차에는 들지 않는다. 배치면 모든 회귀가 같은 겹을 쓴다(배치는 같은 시드의 새 Generator로 하나씩 부른 것과 비트까지 같다).
  - `alphas`: int이면 그 길이의 격자 alpha_max·`eps`^(i/(L − 1))(i = 0 … L − 1; scikit-learn의 `LassoCV`와 같은 격자, alpha_max는 해가 0이 되는 가장 작은 alpha, 1e-15 아래면 1e-15), 배열이면 그 값을 큰 것부터 정렬한 것이다. `ridge_cv`의 격자는 l1_ratio = 1e-3으로 계산한 alpha_max에서 시작한다(glmnet의 ridge 규칙; scikit-learn은 l1_ratio = 0의 자동 격자가 없다). `elastic_net_cv`의 l1_ratio = 0은 `alphas`를 배열로 줘야 한다.
  - 겹마다 나머지로 경로 전체를 적합하고(배치로 함께 푼다), 그 겹의 가중 평균 제곱오차 Σw(y − ŷ)²/Σw를 잰다. `cv_error`는 겹들의 평균(scikit-learn의 `mse_path_.mean(axis=1)`과 같다), `cv_se`는 겹들의 표준편차(자유도 cv − 1)/√cv다.
  - `alpha_min`은 `cv_error`가 가장 작은 alpha(같으면 큰 쪽), `alpha_1se`는 `cv_error` ≤ 최솟값 + 그 점의 `cv_se`인 가장 큰 alpha다(1-SE 규칙). `select='min'`(기본, scikit-learn과 같다)이면 `alpha_min`, `'1se'`면 `alpha_1se`로 다시 적합한다(glmnet의 `predict`는 `lambda.1se`가 기본). 결과의 `alpha`가 고른 값이고, 나머지 필드는 그 alpha의 적합(`ridge`·`lasso`·`elastic_net`을 그 alpha로 부른 것과 비트까지 같다)이다.
  - 교차검증 필드: `alphas` (..., L), `cv_error` (..., L), `cv_se` (..., L), `alpha_min`, `alpha_1se`
  - `elastic_net_cv`는 l1_ratio 하나를 받는다(scikit-learn의 `ElasticNetCV`처럼 여러 값을 함께 고르지 않는다, G52).
  - `ridge_cv(cv='gcv')`: 일반화 교차검증(Golub–Heath–Wahba) GCV = (`ssr`/sw)/(1 − tr H/`nobs`)², tr H = `df_model` + 절편 수. 모든 데이터의 한 SVD로 닫힌 꼴이다. `cv_se`와 `alpha_1se`는 NaN이고 `select='1se'`는 `ValueError`다.
- **정확도** (잰 값, 설계 §3.5): Arb 참값(정해진 활성 집합의 KKT 연립방정식을 정확히 풀고 0인 계수의 KKT 부등식으로 증명한 해)과 견준 계수 벡터의 상대오차(노름)는 조건이 좋은 문제(n = 60 – 20,000)에서 lasso·엘라스틱 넷·ridge 1.7e-15 안, 정규화 GLM 3e-16 안이다. 중심화한 설계의 조건수가 10²–10⁶이면 lasso·엘라스틱 넷 8.6e-15 안이다(연마가 개선한 칸은 3e-16 안). 조건이 나쁘면 ridge와 lasso·엘라스틱 넷의 연마는 double-double 잔차로 개선하고(G51), ridge는 조건수 10¹⁰, alpha 1e-18(최소제곱 가까이)까지 1.7e-15 안이다. scikit-learn은 기본 `tol=1e-4`에서 4e-6–2e-4(조건수 10²–10⁶에서 2e-4–0.34), `tol=1e-12`에서 5e-14–1e-12다.
- **에러**: `lasso(): 'alpha' must be finite and >= 0`, `elastic_net(): 'l1_ratio' must be between 0 and 1 (inclusive)`, `lasso_cv(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not int`, `lasso_cv(): 'cv' must be <= the number of rows (30), not 31`, `ridge_cv(): 'cv' must be an int or 'gcv', not 'x'`, `penalized_glm(): 'family' must be 'binomial' or 'poisson', not 'gamma'`, `penalized_glm(): 'y' must be 0 or 1 for the binomial family`.

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
| 정규화 | `OLS.fit_regularized` | `Ridge`, `Lasso`, `ElasticNet`, `enet_path`, `…CV` | | `ridge`, `lasso`, `elastic_net`(`alpha`에 배열이면 경로), `ridge_cv`, `lasso_cv`, `elastic_net_cv` |
| 정규화 GLM | `GLM.fit_regularized` | `LogisticRegression`(penalty), `PoissonRegressor` | | `penalized_glm` |
| 강건 | `RLM` | `HuberRegressor`, `TheilSenRegressor` | `stats.theilslopes`, `siegelslopes` | `rlm` |
| 분위 | `QuantReg` | `QuantileRegressor` | | `quantreg` |
| 비선형 | | | `optimize.curve_fit`, `least_squares` | G6 |
