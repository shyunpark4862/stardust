# 회귀 영역 설계 (sd.regression)

> 상태: **설계 초안, G0 완료** (2026-11-18). 큰 결정은 사용자가 정했다(§1.1). 나머지는 안이고, 마일스톤마다 시작 전에 물을 것은 §8에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

`sd.regression`(회귀분석)의 확정 사항과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md)에 있다. 공개 함수와 시그니처의 안은 [docs/api/regression.md](../api/regression.md)에 있다.
- 바닥은 `sd.linalg`(QR, Cholesky, SVD, 삼각 풀이)와 `sd.stats`(t·F·카이제곱 분포의 꼬리와 역함수)다. 데이터는 `sd.ndarray`나 `sd.frame`의 컬럼으로 받는다.
- 비교 대상은 statsmodels(추론), scikit-learn(예측·정규화), scipy(`linregress`, `curve_fit`·`least_squares`), R(`lm`, `glm`, `quantreg`)이다. statsmodels 0.15.0과 scikit-learn 1.9.1을 테스트 의존성으로 쓴다(G0).

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-18)
| 번호 | 결정 |
|---|---|
| G1 목적 | 추론(계수의 표준오차·t·p·신뢰구간, 적합도, 진단; statsmodels가 하는 일)과 예측(정규화, 새 X의 예측; scikit-learn이 하는 일)을 **동등하게** 다룬다 |
| G2 모델 범위 | 선형(OLS·WLS·GLS, 강건 표준오차), 일반화 선형(GLM), 정규화(ridge·lasso·elastic net), 강건·분위·비선형 최소제곱. 순서는 마일스톤으로 나눈다 (§7) |
| G3 API | 함수 + 결과 객체: `r = sd.regression.ols(y, X)` 뒤 `r.params`, `r.predict(X_new)`. 배열(과 frame 컬럼)로 받는다. R식 공식 문자열(`"y ~ x1 + x2"`)과 scikit-learn식 추정기 클래스(`fit`/`predict`)는 두지 않는다 |
| G4 위치 | 새 최상위 영역 `sd.regression` |
| G5 절편 | 기본으로 넣는다. `intercept=False`로 끈다(statsmodels는 넣지 않고, scikit-learn과 R은 넣는다) |
| G6 추론 | 회귀에 딸린 검정(계수의 t, 모형의 F, 신뢰구간, 우도비)은 결과 객체에 둔다. 분포는 `sd.stats`의 것을 쓴다. 독립 검정 함수(t검정 등)는 여전히 나중이다(stats.md §7) |
| G7 정확도 | 닫힌 꼴이 있는 것(OLS 계수·표준오차·p값 등)은 다른 영역처럼 Arb 참값과 비교한다. 반복 방법(GLM, lasso, 분위)은 최적성 조건의 잔차와 statsmodels·scikit-learn의 값으로 확인한다. 두 패키지는 테스트 의존성으로만 둔다 |
| G8 대조 패키지 (G0) | statsmodels·scikit-learn을 pyproject의 test 의존성에 넣고 설치한다. Python 3.15에서 설치되지 않으면 다른 Python으로 두 패키지의 결과를 한 번 만들어 테스트용 고정 데이터로 둔다 |
| G9 랭크 부족 (G1) | R처럼: 열 피벗 QR로 독립인 열만 풀고, 겹치는 열의 계수·표준오차·t·p는 NaN이다. 적합값·잔차·예측은 정상이다. 절편과 상수 열의 중복도 같다 |
| G10 결과 필드 (G1) | statsmodels의 이름(`params bse tvalues pvalues rsquared …`). 절편은 `params[0]`이고 추론 필드도 같은 자리다 |
| G11 강건 표준오차 (G1) | HC0–HC3, HAC(Newey–West), 군집 강건 |
| G12 polyfit (G1) | 따로 두지 않는다. 다항식 적합은 거듭제곱 열로 `ols`를 부른다. 배열 영역에도 두지 않는다 |
| G13 summary (G1) | `summary()`는 계수 표(계수·표준오차·t·p·구간)와 R²·F·AIC 등을 담은 글을 돌려준다. `repr`는 짧은 요약이다 |
| G14 배치 (G1) | 넣는다: 앞 축을 배치로 받고(`y` (..., n), `X` (..., n, k)) 결과 필드도 배치 모양이다. 배치에서 `nan_policy='omit'`은 행 수가 달라지므로 `ValueError`다 |
| G15 반복 개선 (G1) | 항상 한 번 보정한다: QR 해의 잔차를 double-double로 계산해 같은 QR로 증강 시스템([I X; Xᵀ 0]) 꼴의 보정을 푼다(QR을 다시 하지 않는다). 계약은 "κ(X) ≤ 한계까지 계수 벡터의 상대오차 1e-14, 그 밖은 잰 값"이고, 한계값은 G1에서 재서 다시 확정한다 |
| G16 NaN (G1) | `nan_policy`의 기본은 'propagate'(결과 값이 모두 NaN). 'omit'은 NaN이 있는 행을 빼고, 'raise'는 `ValueError` |
| G17 진단 (G1) | 지렛대(hat), 스튜던트화 잔차, Cook 거리와 Durbin–Watson, Jarque–Bera, Breusch–Pagan. `summary()`에는 Durbin–Watson과 Jarque–Bera를 넣는다(statsmodels와 같다) |
| G18 linregress (G1) | scipy와 같은 꼴: `linregress(x, y)`(인자 순서도 scipy), 결과는 `slope intercept rvalue pvalue stderr intercept_stderr` namedtuple |
| G19 이름 (G1) | 결과 필드는 입력과 상관없이 `sd.ndarray`다. 계수 이름은 `param_names`(절편 `'const'`, frame이면 컬럼 이름, 배열이면 `'x1'…'xk'`)에 두고 `summary()`가 쓴다 |
| G20 가중치 0 (G1) | R의 `lm`과 같다: 적합에 쓰지 않고 `nobs`·잔차 자유도에서 뺀다. 적합값과 잔차는 계산한다. 음수 가중치는 `ValueError`, NaN 가중치는 `nan_policy`를 따른다 |
| G21 겹치는 열의 선택 (G1) | 피벗 없는 QR에서 \|R_jj\| ≤ rtol·‖x_j‖인 열을 겹친다고 보고 빼서 다시 분해한다(R `dqrdc2`와 같은 선택: 앞 열이 남는다). `rtol`의 기본은 sd.linalg와 같은 max(n, k)·eps이고 `rtol=`로 바꾼다(R의 1e-7과 같게 하려면 `rtol=1e-7`) |

### 1.2 범위 (안)
- **선형**: `ols`, `wls`(가중치), `gls`(알려진 오차 공분산), 단순회귀 `linregress`(scipy와 같은 결과, G18). 다항식 적합은 `ols`로 한다(G12). 공분산 추정은 고전형, HC0–HC3, HAC(Newey–West), 군집 강건(G11)
- **일반화 선형**: `glm(y, X, family=…, link=…)`. 분포족은 정규, 이항(로짓·프로빗·cloglog), 포아송(로그), 감마(역수·로그), 역가우스, 음이항(고정된 α). 오프셋과 노출(exposure), 사전 가중치
- **정규화**: `ridge`(여러 α를 한 번에), `lasso`, `elastic_net`, 정규화 경로, 교차검증으로 α 고르기(§8 G3 질문). 정규화 GLM(로지스틱 L1·L2)은 §8 G3 질문
- **강건**: `rlm`(M 추정: Huber, Tukey bisquare 등, 척도는 MAD), §8 G4 질문으로 Theil–Sen·Siegel
- **분위**: `quantreg(y, X, q=0.5)`. 선형계획으로 정확한 해(§3.6)와 표준오차
- **비선형 최소제곱**: `curve_fit`/`least_squares` 꼴(Levenberg–Marquardt, 범위 제약). 사용자 함수는 Python 호출이다(§3.7)
- **없는 것** (안): 공식 문자열과 범주형 변수 자동 처리(더미 코딩은 사용자가 만든다), 추정기 클래스, 혼합 효과 모형, 시계열 모형(ARIMA 등), 패널·도구변수(linearmodels), 생존 분석, 베이즈 회귀, 신경망·트리 같은 비모수 예측 모형, 결과의 그림

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 입력은 위치로(`y`, `X`), 옵션은 키워드로만. 별칭이 없다. float64 실수로 계산하고(bool·int64는 바꿔서) complex는 `TypeError`다.
- 같은 입력이면 스레드 수와 상관없이 같은 비트다.
- **모든 호출은 정해진 시간 안에 끝난다**(stats.md §1.3). 반복 방법에는 근거 있는 반복 상한을 둔다. 상한에 닿았을 때 무엇을 돌려줄지는 §8 G2 질문이다(stats의 규칙은 NaN이고, statsmodels·scikit-learn은 경고와 함께 마지막 값을 준다).
- 경고를 내지 않는다(special과 같다). 수렴, 특이성, 분리(separation) 같은 상태는 결과 객체의 필드로 알린다(§2).
- 참값은 Arb만 쓴다. mpmath는 쓰지 않는다.
- 영역 간 일관성이 먼저다. 자유도 인자 이름(`correction`), NaN 처리(`nan_policy`), 결과의 namedtuple·객체 규칙은 기존 영역을 따른다(§8 G1 질문).

---

## 2. 입력과 결과 (안)

### 2.1 입력
- `y`: 길이 n의 1차원. `X`: (n,) 또는 (n, k). `X`가 1차원이면 설명변수 하나다.
- `sd.frame.DataFrame`/`Series`도 받는다. 그때 계수에 컬럼 이름이 붙는다(절편은 `'const'`; 이름 붙은 결과의 꼴은 G1 구현 때 정한다). 받는 쪽은 `to_array`로 바꾸고 이름만 들고 간다.
- 절편(G5)은 X 앞에 1의 열을 붙인 것과 같다. X에 이미 상수 열이 있으면 그 열의 계수가 NaN이다(랭크 부족 규칙 G9).
- **NaN** (G16): `nan_policy`('propagate' 기본, 'omit', 'raise'; stats의 기술통계와 같다). 'omit'은 y나 X의 행 가운데 NaN이 있는 행을 뺀다.
- **배치** (G14): linalg처럼 앞 축을 배치로 받아 같은 모양의 회귀 여러 개를 한 번에 푼다(`y` (..., n), `X` (..., n, k)). 작은 회귀를 많이 푸는 경우(그룹별 회귀, 이동창 회귀)에 statsmodels·scikit-learn보다 크게 빠를 수 있는 곳이다.

### 2.2 결과 객체 (OLS를 예로)
- 계수와 추론: `params`, `bse`(표준오차), `tvalues`, `pvalues`, `conf_int(alpha=0.05)`, `cov_params()`
- 적합: `fittedvalues`, `resid`, `nobs`, `df_model`, `df_resid`, `ssr`, `ess`, `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`, `llf`, `aic`, `bic`, `scale`(σ̂²)
- 예측: `predict(X_new)`(점예측), `predict_interval(X_new, alpha=…, kind='mean' | 'observation')`(평균의 신뢰구간과 예측구간)
- 상태: `rank`, `condition_number`, 반복 방법은 `converged`, `n_iter`
- 진단 (G17): 지렛대(hat 값), 스튜던트화 잔차, Cook 거리, Durbin–Watson, Jarque–Bera, Breusch–Pagan.
- 이름은 statsmodels의 것이다(G10: `params`, `bse`, `rsquared`). scikit-learn의 이름(`coef_`, `intercept_`)은 쓰지 않는다. 절편은 `params[0]`이다.
- `summary()` (G13): 계수 표와 적합도·진단을 담은 글. `repr`는 짧은 요약.

---

## 3. 수치 방법 (안)

### 3.1 OLS
- **QR**: X(절편 열 포함)를 Householder QR로 분해한다(LAPACK `geqrf`, sd.linalg와 같은 호출). 계수는 R β = Qᵀy의 삼각 풀이다. 정규방정식(XᵀX)은 조건수를 제곱하므로 쓰지 않는다.
- **정확도를 올리는 단계** (G15, 재서 정한다): QR로 얻은 β에서 잔차 r = y − Xβ를 double-double로 계산하고, 같은 QR로 보정 Δβ를 풀어 한 번 더한다(혼합 정밀도 반복 개선, Björck). 잔차가 작은 문제에서는 β의 오차가 κ(X)·u 수준에서 거의 u 수준으로 내려간다. 큰 잔차 문제에서는 κ²·u 항이 남으므로, 보정을 증강 시스템([I X; Xᵀ 0])에 대해 해야 한다. 비용과 이득을 재서 정한다(§8 G1 질문).
- **계약의 꼴** (안): 계수 하나하나에 상대 1e-14를 약속할 수 없다(조건수가 큰 X에서는 문제 자체가 민감하다). special의 역함수처럼 조건수를 넣은 꼴로 정한다: ‖Δβ‖/‖β‖ ≤ c·u·(κ + κ²·‖r‖/(‖X‖‖β‖)) 같은 섭동 한계를 기준으로 잰 값을 적는다. 보장 영역(κ 범위)은 G1에서 정한다.
- **랭크 부족** (G9): R처럼 겹치는 열의 계수를 NaN으로 두고 나머지를 푼다. 고르는 규칙은 G21(피벗 없는 QR의 |R_jj|, 앞 열이 남는다).
- **표준오차**: cov = σ̂²(RᵀR)⁻¹. (RᵀR)⁻¹은 R⁻¹을 삼각 역행렬로 구해 R⁻¹R⁻ᵀ로 만든다. 지렛대 h_ii = ‖Q의 i행‖²이다(HC2·HC3, 진단에 쓴다).
- **p값과 구간**: t의 두꼬리 p = 2·`StudentT(df).ccdf(|t|)`(1 − cdf가 아니므로 아주 작은 p도 상대 정확도), F의 p = `F(dfn, dfd).ccdf(F)`, 구간은 `icdf`. 모두 sd.stats의 보장 영역에서 상대 1e-14다.
- **R²**: 1 − SSR/TSS는 적합이 좋을 때 상쇄가 없지만, 적합이 나쁠 때(R² ≈ 0)는 ESS/TSS로 계산해 상대 정확도를 지킨다(안). SSR·TSS는 보상 합.

### 3.2 WLS, GLS
- WLS: 행마다 √w를 곱해 OLS로 푼다. 가중치 0인 행은 적합과 nobs·자유도에서 빼고 적합값·잔차만 계산한다(R과 같다, G20). statsmodels(0.15.0)는 그 행을 `nobs`에 세므로 자유도와 표준오차가 다르다(60행 중 2행이 0이면 df 57 대 55, 표준오차 1.8% 차이; tools/regref.py로 확인).
- GLS: 오차 공분산 Σ의 Cholesky L로 L⁻¹y, L⁻¹X를 만들어 OLS로 푼다. Σ를 추정하는 FGLS(AR(1) 등)는 범위 밖(안).

### 3.3 강건 표준오차
- HC0–HC3: 샌드위치 (XᵀX)⁻¹ Xᵀ diag(ω) X (XᵀX)⁻¹을 QR로 계산한다(R⁻¹(Qᵀ diag(ω) Q)R⁻ᵀ 꼴, XᵀX를 만들지 않는다).
- HAC(Newey–West, Bartlett 커널; 지연 수의 기본값과 소표본 보정은 statsmodels와 맞춘다), 군집 강건(군집 레이블, 소표본 보정 G/(G−1)·(n−1)/(n−k)) (G11).

### 3.4 GLM
- IRLS(반복 가중 최소제곱): 매 반복이 WLS 한 번이다. 시작값은 statsmodels·R과 같은 규칙(μ₀ = (y + ȳ)/2 등)으로 두고, 수렴은 편차(deviance)의 상대 변화로 판정한다. 반복 상한은 안으로 100이다.
- 편차와 로그우도는 분포족마다 상쇄 없는 식으로 계산한다(예: 포아송 편차 2Σ[y log(y/μ) − (y − μ)]는 y/μ ≈ 1에서 log1p로). 링크와 역링크는 sd.special의 `expit`, `logit`, `log1p`, `expm1`, `ndtr`, `ndtri`를 쓴다.
- 척도: 이항·포아송은 1, 그 밖은 Pearson χ²/df(statsmodels 기본)와 편차/df 가운데 §8 G2 질문.
- **완전 분리**(로지스틱에서 계수가 무한대로 가는 경우): 감지해서 상태 필드로 알리고 계수는 NaN 또는 inf로 둔다(§8 G2 질문). statsmodels는 에러나 경고, R은 경고와 아주 큰 계수다.
- 추론: Wald 표준오차(피셔 정보), 우도비 검정(`llr`, `llr_pvalue`), 편차·Pearson 잔차.

### 3.5 정규화
- **목적함수의 꼴**: scikit-learn과 같게 (1/(2n))‖y − Xβ‖² + α(ρ‖β‖₁ + (1 − ρ)/2‖β‖²)로 할지, glmnet처럼 표준화와 λ 경로를 기본으로 할지 §8 G3 질문. 절편은 벌점에서 뺀다.
- **ridge**: X의 SVD 한 번으로 모든 α의 해를 얻는다(β(α) = V diag(s/(s² + α)) Uᵀy). 일반화 교차검증(GCV)도 같은 SVD로 닫힌 꼴이다.
- **lasso, elastic net**: 좌표 하강(공분산 갱신, 활성 집합), 수렴은 쌍대 간격(duality gap)으로 판정한다(scikit-learn과 같은 기준). 경로는 큰 α에서 작은 α로 따뜻한 시작(warm start).
- **교차검증**: k겹으로 α를 고른다. 겹 나누기는 `sd.random`의 `rng`를 받는다(재현 가능).
- 정규화 해에는 고전적 표준오차가 없다. 추론 필드는 두지 않는다(G1의 "동등"은 모델마다 그 모델에 맞는 쪽을 뜻한다고 본다, §8 G3 질문).

### 3.6 강건과 분위
- **RLM**: M 추정을 IRLS로 푼다. 척도는 MAD(정규 일관 상수 1.4826), Huber의 제안 2는 §8 G4 질문. 표준오차는 Huber의 샌드위치(statsmodels의 H1).
- **분위 회귀**: 목적함수 Σ ρ_q(y − Xβ)는 선형계획이다. statsmodels의 IRLS 근사는 정확한 해를 주지 않는다. 안은 Frisch–Newton 내부점법(Koenker–Portnoy, R `quantreg`의 `fn`)으로 풀고 마지막에 기저를 찾아 정확한 꼭짓점 해를 낸다. 표준오차는 커널 샌드위치(Hendricks–Koenker)와 순위 기반 구간 가운데 §8 G5 질문.

### 3.7 비선형 최소제곱
- 사용자 함수 f(x, *θ)를 Python에서 부른다. 한 반복에 함수 호출(과 수치 야코비안을 위한 호출)이 필요하므로 GIL을 쥔 채 돈다. 계산의 무게는 사용자 함수에 있다.
- 방법은 Levenberg–Marquardt(MINPACK `lmder`의 신뢰 영역 꼴)와, 범위 제약이 있으면 신뢰 영역 반사(TRF, scipy `least_squares`의 기본) 가운데 §8 G6 질문. 결과에는 공분산(σ̂²(JᵀJ)⁻¹)과 추론을 둔다(G6).

---

## 4. 성능 (안)
- 기준은 statsmodels와 scikit-learn(1·10스레드), 크기는 (n, k) = (10³, 10), (10⁵, 10), (10⁶, 50), (10⁴, 1000)과 배치(1만 개의 (50, 3))다.
- QR과 삼각 풀이는 Accelerate LAPACK(sd.linalg와 같다)이고, 잔차·합·지렛대처럼 행을 도는 부분은 기존 스레드풀과 GIL 규칙(API.md §3)을 따른다.
- 배치(§2.1)를 넣으면 작은 회귀 여러 개는 linalg의 `run_batch`처럼 행렬마다 한 태스크로 나눈다.

---

## 5. 층과 파일 (안)
- `src/stardust/regression/`: 공개 함수, 입력 정리(frame 컬럼, 절편, NaN), 결과 객체, 추론(sd.stats 호출)
- `csrc/regression/`: QR 기반 풀이와 반복 개선, 지렛대·샌드위치, IRLS의 한 걸음, 좌표 하강, 분위 회귀의 내부점법 — 루프가 무거운 곳만 C++. 선형대수 호출은 `csrc/linalg/lapack.hpp`의 래퍼를 쓴다.
- `tools/regref.py`: Arb 참값(정확한 유리수 또는 고정밀 구간으로 최소제곱 풀기, 표준오차와 p값까지)
- `tests/regression/`, `bench/regression/`

---

## 6. 참값과 검증 (안)
- **닫힌 꼴(OLS·WLS·GLS·ridge)**: Arb의 행렬 연산(`arb_mat`)으로 정규방정식을 고정밀로 풀어 참값을 만든다(고정밀에서는 조건수 제곱이 문제가 아니다). 계수, 표준오차, t, p, R², F, 구간, 예측구간 모두.
- **반복 방법**: 최적성 조건(GLM은 점수 방정식 Xᵀ W (z − Xβ) = 0, lasso는 KKT 조건과 쌍대 간격, 분위는 선형계획의 쌍대 실행 가능성)의 잔차를 재고, 같은 문제의 statsmodels·scikit-learn·R 결과와 비교한다. 해가 유일하지 않은 경우(분위 회귀의 여러 꼭짓점)는 목적함수 값으로 비교한다.
- 시험 문제: NIST StRD의 선형 회귀 인증 데이터 11개(Norris, Pontius, NoInt1–2, Filip, Longley, Wampler1–5; `tests/regression/data/`, 읽기는 `tests/regression/nist.py`)는 공인된 참값(15자리)과 극단적인 조건수를 함께 준다.
- **참값 도구** (G0, `tools/regref.py`): 입력(double, 또는 10진 문자열과 거듭제곱을 Arb로 주는 함수)을 정확한 구간으로 받아 정규방정식을 구간 산술로 푼다. 고정밀에서는 조건수의 제곱이 비트를 더 쓸 뿐 틀리게 하지 않는다. 모든 값의 구간이 2⁻¹¹⁰ 상대까지 좁아질 때까지 정밀도를 두 배씩 올린다. 내는 값: 계수, 표준오차, t, p, 신뢰구간, 적합값, 잔차, 지렛대, Cook 거리, SSR·ESS·TSS, R²·수정 R², F와 p, 척도, 로그우도·AIC·BIC, HC0–HC3 표준오차, Durbin–Watson, Jarque–Bera와 p. 가중치 0인 행은 R처럼 뺀다.
  - 잔차가 정확히 0인 완전 적합(Wampler1·2)은 어떤 유한 정밀도로도 0임을 확정할 수 없다. 1024비트에서도 잔차 제곱합의 구간이 0을 품으면 완전 적합으로 보고 계수·적합값과 0인 값들만 낸다(t·p·구간 등은 정의되지 않는다).
  - 확인: NIST 11개 모두 인증값(계수, 표준오차, 잔차 표준편차, R²)과 1e-14 안이다. 무작위 문제에서 statsmodels와 1e-15 수준으로 맞는다(Cook 거리는 statsmodels 쪽이 2.4e-14 틀린다). 테스트는 `tests/regression/test_reference_tool.py`.
- **기존 패키지의 성적** (G0, 2026-11-18): 사용자가 실제로 주는 double 입력(10진 값을 double로, 거듭제곱 열도 double로)에 대한 Arb 참값과 계수를 비교했다(최대 상대오차).

  | 문제 | κ(X) | statsmodels 0.15.0 | scikit-learn 1.9.1 | numpy·sd.linalg `lstsq` | QR + 개선 1번(정규식 보정, 모형) |
  |---|---:|---:|---:|---:|---:|
  | Norris | 8.6e2 | 1.1e-12 | 5.6e-14 | 9.1e-13 | 2.1e-15 |
  | Pontius | 1.4e13 | 7.8e-8 | 6.6e2 | 7.8e-8 | 3.6e-14 |
  | Filip | 1.8e15 | 1.1 | 1.0 | 1.1 | 1.4e-8 |
  | Longley | 4.9e9 | 1.4e-11 | 1.2e-14 | 1.4e-11 | 4.3e-12 |
  | Wampler1 | 6.4e6 | 2.9e-10 | 0.65 | 2.9e-10 | 2.7e-11 |
  | Wampler2 | 6.4e6 | 4.1e-11 | 3.4 | 4.2e-11 | 6.9e-16 |
  | Wampler4 | 6.4e6 | 3.9e-9 | 0.65 | 3.6e-9 | 3.8e-9 |
  | Wampler5 | 6.4e6 | 3.8e-7 | 0.65 | 4.0e-7 | 3.8e-7 |

  - Filip은 세 패키지 모두 자릿수가 하나도 맞지 않는다. scikit-learn은 다항식 문제(Pontius, Wampler)에서 크게 틀린다.
  - 마지막 열은 G15를 미리 본 모형이다(numpy QR, 잔차를 정확히, 보정은 R⁻¹Qᵀr). 잔차가 작은 문제(Pontius, Wampler2, Filip)는 크게 좋아지지만, 잔차가 큰 문제(Longley, Wampler4·5)는 그대로다: 오차의 κ²·‖r‖ 항이 남기 때문이다(§3.1). G1의 보정은 증강 시스템이나 보정된 반정규식(CSNE)을 여러 번 하는 꼴이어야 하고, κ²u ≥ 1인 Filip은 그래도 한계가 있다. G1에서 재서 제안한다.

---

## 7. 마일스톤 (안)
| 마일스톤 | 내용 |
|---|---|
| G0 준비 | statsmodels·scikit-learn을 테스트 의존성으로, `tools/regref.py`(Arb 참값), NIST StRD 데이터 (완료 2026-11-18) |
| G1 선형 | 결과 객체의 공통 틀, `ols wls gls linregress`, 추론·예측·구간, 강건 표준오차, 랭크 부족, 진단, 배치 |
| G2 GLM | 분포족·링크, IRLS, 분리 감지, 우도비, 편차·Pearson 잔차 |
| G3 정규화 | `ridge lasso elastic_net`, 경로, 교차검증 |
| G4 강건 | `rlm`(M 추정) |
| G5 분위 | `quantreg`(내부점법, 정확한 해), 표준오차 |
| G6 비선형 | `curve_fit`/`least_squares` 꼴 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 아래 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.

---

## 8. 마일스톤을 시작하기 전에 물을 것
- **G0**: 정했다(2026-11-18, G8).
- **G1**: 정했다(2026-11-18, G9–G21). 남은 것: 정확도 계약의 κ 한계값(G15, G1에서 재서 다시 확정).
- **G2**: 분포족·링크 목록, 반복 상한에 닿았을 때(NaN / 마지막 값 + `converged=False`), 척도 추정, 분리의 처리, 음이항의 α를 추정할지
- **G3**: 목적함수의 꼴과 표준화(scikit-learn / glmnet), α의 이름(`alpha`·`l1_ratio` / `lambda`), 교차검증의 범위, 정규화 GLM(로지스틱), 추론 필드를 둘지
- **G4**: M 추정 함수 목록, 척도 추정, Theil–Sen·Siegel(scipy의 `theilslopes`, `siegelslopes`)을 넣을지
- **G5**: 표준오차 방법, 여러 분위를 한 번에 받을지
- **G6**: 이름과 API(scipy의 `curve_fit`과 `least_squares` 가운데 어느 꼴), 방법, 야코비안을 사용자가 줄 수 있게 할지, 범위 제약
