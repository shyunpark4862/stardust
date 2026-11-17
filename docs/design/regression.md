# 회귀 영역 설계 (sd.regression)

> 상태: **G1 구현** (2026-11-18): 선형 모형의 수치 핵심(§2, §3.1–§3.3). 큰 결정은 사용자가 정했다(§1.1). 나머지는 안이고, 마일스톤마다 시작 전에 물을 것은 §8에 있다. §3.1의 정확도 계약은 사용자 결정을 기다리는 제안이다.
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
| G15 반복 개선 (G1) | 필요할 때까지, 최대 3번 보정한다(사용자 결정 2026-11-18; 처음의 "항상 한 번"을 측정 뒤 바꿨다): QR 해의 잔차를 double-double로 계산해 같은 QR로 증강 시스템([I X; Xᵀ 0]) 꼴의 보정을 푼다. 보정이 다음 단계에서 의미가 없어지면 멈춘다(§3.1). 계약은 "κ(X) ≤ 10¹¹이면 계수 벡터의 상대오차 ≤ 1e-14, 그 밖은 잰 값"이다 |
| G22 완전 적합 (G1) | 잔차가 정확히 0이면 극한값: `bse` 0, t ±inf, p 0, `llf` +inf(사용자 결정 2026-11-18) |
| G23 강건 공분산의 분포 (G1) | HC(와 G1b의 HAC·군집)에서도 p값과 구간은 t(`df_resid`), 모형 검정은 F(q, `df_resid`)다(R의 coeftest·Stata와 같다; statsmodels의 기본은 정규분포) (사용자 결정 2026-11-18) |
| G24 inf (G1) | y·X·가중치에 inf가 있으면 `nan_policy`와 상관없이 `ValueError`다(statsmodels·R과 같다) (사용자 결정 2026-11-18) |
| G25 G1a의 세부 (사용자 확인 2026-11-18) | NaN이 있는 회귀(propagate)는 `nobs`·`df_model`·`df_resid`가 NaN이고 `rank`는 0이다. 절편 여부는 `intercept` 옵션만 본다(상수 열을 찾지 않는다: 절편 없이 상수 열을 주면 R²는 비중심, R과 같다). `gls`의 `sigma`는 (…, n, n) 행렬만 받고 `cov_type`·`rtol`은 ols와 같다. 이름 `uncentered_tss`, 결과 클래스 `OLSResult`·`WLSResult`·`GLSResult`. `linregress`는 배치를 받고, 점이 한 직선 위에 있으면 p = 0이다(scipy는 1e-20을 더해 약 9e-11) |
| G26 HAC (G1b) | `cov_type='HAC'`, 지연 수는 `maxlags=`(기본 floor(4(n/100)^(2/9)), Newey–West 1994), Bartlett 커널. 자동 대역폭 선택은 하지 않는다 (사용자 결정 2026-11-18) |
| G27 군집 (G1b) | `cov_type='cluster', groups=…`(일원 군집만). 소표본 보정 G/(G−1)·(n−1)/(n−k), t의 자유도 G−1(Stata와 같다) (사용자 결정 2026-11-18) |
| G28 스튜던트화 잔차 (G1b) | 내적 `resid_studentized_internal`과 외적 `resid_studentized_external`을 둘 다 둔다(statsmodels의 이름) (사용자 결정 2026-11-18) |
| G29 Breusch–Pagan (G1b) | Koenker의 스튜던트화 판이 기본(statsmodels·R lmtest와 같다), `studentize=False`로 원래 판 (사용자 결정 2026-11-18) |
| G30 G1b의 세부 (사용자 확인 2026-11-18) | 가중치 0인 행의 지렛대·스튜던트화 잔차·Cook 거리는 0(가중 공식의 값, statsmodels와 같다). Breusch–Pagan 보조 회귀는 늘 상수항을 넣는다. 군집 레이블은 float도 받고 NaN은 결측이다. 완전 적합에서 스튜던트화 잔차·Cook·DW·JB·BP는 NaN(0/0). 추론의 자유도는 공개 필드 `df_resid_inference`(군집이면 G−1). `summary(*, alpha=0.05)`, 배치는 에러 |
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

## 2. 입력과 결과

### 2.1 입력
- `y`: (…, n). `X`: (…, n, k) 또는 (n,)(설명변수 하나). `sd.ndarray`, 실수의 시퀀스, `sd.frame`의 `Series`·`DataFrame`. frame은 `to_array(dtype=float64, na_value=NaN)`로 바꾸고 컬럼 이름만 들고 간다(`param_names`). bool·int64는 float64로, complex는 `TypeError`다.
- 절편(G5)은 커널이 X를 꾸릴 때 앞에 붙이는 1의 열이다(Python에서 배열을 이어 붙이지 않는다). 절편이 있는지는 `intercept`만 본다. statsmodels는 X의 상수 열을 찾아 절편으로 보지만(`k_constant`), 여기서는 X에 상수 열이 있으면 그 열이 겹치는 열이다(G9).
- **NaN** (G16): `'propagate'`는 커널이 회귀마다 NaN을 보고 그 회귀의 결과를 모두 NaN으로 채운다(`rank` 0). `'omit'`은 Python에서 행을 빼고(배치에서는 `ValueError`), `'raise'`는 Python에서 검사한다. inf는 커널이 찾아 언제나 `ValueError`다(statsmodels의 `MissingDataError`, R의 `NA/NaN/Inf in 'x'`처럼 에러다). X는 메모리 순서로 훑고(행 우선 X는 행 밖), NaN과 inf는 유한하지 않은 값이 있을 때만 다시 가른다.
- **배치** (G14): `y`, `X`, `weights`(GLS는 백색화한 1의 열)의 배치 모양을 linalg의 `Batch`로 브로드캐스트하고, linalg의 `run_batch`로 회귀 하나를 태스크 하나로 돈다. 회귀 하나의 계산은 어디서 돌든 같으므로 배치는 회귀를 하나씩 부른 것과 비트까지 같고, 스레드 수와도 상관없다. 회귀 하나가 크면 그 안의 행 합(§3.1)을 스레드풀로 나누는데, 나누는 덩어리는 열 수로만 정해지고 부분합을 순서대로 더하므로 이것도 비트가 같다. 배치 안에서는 풀이 중첩되지 않아 행 합이 한 스레드로 돈다(결과는 같다).

### 2.2 결과 객체
- `OLSResult`, `WLSResult`, `GLSResult`(같은 틀). 커널은 회귀마다 `params`, 공분산(고전형은 (RᵀR)⁻¹, 강건형은 샌드위치), 쓰인 열의 R, 적합값, 잔차, 쓰인 열 표시, 스칼라 13개(nobs, rank, SSR, 중심·비중심 TSS, 직접 더한 중심·비중심 ESS, Σlog w, Wald F와 그 자유도, NaN 표시, 군집 수, HAC 지연 수)를 낸다. 나머지 필드는 Python에서 처음 읽을 때 sd 배열 연산과 sd.stats로 계산해 둔다(statsmodels의 `cache_readonly`와 같은 방식).
- 결과는 커널에 준 입력(백색화한 y·X, 가중치, GLS의 백색화한 1의 열, 군집 번호)을 참조로 들고 있다. 진단은 처음 읽을 때 이것으로 커널을 다시 부른다(복사하지 않는다).
  - `leverage`: 커널의 두 번째 방식(`hat_only`)이 적합과 같은 열 선택·같은 QR을 하고 `orgqr`로 얇은 Q를 만들어 h_t = ‖q_t‖²를 낸다. Q는 반올림 수준에서 직교이므로 1 − h도 κ와 상관없이 절대오차가 u 정도다(R로 R⁻ᵀx_t를 풀면 κu). 적합 자체에는 Q를 만드는 비용(QR 한 번 꼴)을 더하지 않으려고 따로 부른다.
  - `durbin_watson`, `jarque_bera()`: 커널 `resid_stats`가 백색화한 잔차를 적합에 쓰인 행만 순서대로 모아 Σ(ẽ_t − ẽ_{t−1})²와 중심 적률의 합 Σd^p(p = 2, 3, 4, d = ẽ − 평균)을 double-double로 더한다. 덩어리는 n으로만 정해지고 순서대로 더하므로 스레드 수와 상관없는 비트다.
  - `breusch_pagan()`: 보조 회귀(ẽ²를 상수와 백색화한 설계의 열에)를 같은 커널로 푼다(가중치 0인 행은 0/1 가중치로 뺀다). 보조 회귀의 R², ESS, F는 G1a의 필드 그대로다.
  - 스튜던트화 잔차와 Cook 거리는 위의 것과 `scale`로 sd 배열 연산이다. 외적 스튜던트화 잔차는 r√((df − 1)/(df − r²))로 한 번에 구한다(statsmodels는 행을 하나씩 뺀 적합 n번).
- 이름은 statsmodels의 것이다(G10). `centered_tss`(언제나 중심화)와 `uncentered_tss`를 둘 다 둔다(statsmodels와 같다; 절편이 없으면 R²는 비중심 TSS를 쓴다).
- `conf_int`와 `predict_interval`은 (…, k, 2), (…, m, 2) 배열이다(statsmodels의 `conf_int`와 같은 모양).
- `summary()`(G13)는 회귀 하나의 글이다. 배치에서는 `ValueError`다(`nan_policy='omit'`의 배치 에러와 같은 꼴). 배치의 한 원소를 고르는 방법(결과의 인덱싱)은 없다.

---

## 3. 수치 방법

### 3.1 OLS
- **QR**: 쓰이는 행(가중치 > 0)을 √w로 백색화한 설계(절편 열 포함)를 열 우선으로 꾸려 LAPACK `geqrf`(sd.linalg와 같은 Accelerate)로 분해한다. 처음 해는 R β = (Qᵀ b)₁의 삼각 풀이다. 정규방정식은 쓰지 않는다.
- **랭크 부족** (G9, G21): 피벗 없는 `geqrf` 뒤 |R_jj| ≤ rtol·‖x_j‖(x_j는 백색화한 열, ‖x_j‖은 분해 전의 2-노름)인 첫 열을 빼고, 그 뒤 열들만 다시 꾸려 앞 j개 반사의 Qᵀ를 적용하고(`ormqr`) 남은 블록을 `geqrf`한다. R `dqrdc2`의 선택과 같다(겹친 열은 반사를 만들지 않으므로, 뒤 열의 판정이 그 열의 잡음 방향에 오염되지 않는다). 겹친 열이 8개를 넘으면 그 뒤는 `dqrdc2`처럼 한 열씩(`larfg`·`larf`) 처리해 비용을 QR 한 번 꼴로 묶는다. 하나라도 뺐으면 남은 열로 처음부터 `geqrf`를 한 번 더 해서, 결과가 그 열을 뺀 적합과 비트까지 같다. 열이 쓰인 행보다 많으면 앞 nobs개만 남는다. `rtol`의 기본은 max(nobs, k)·eps(sd.linalg와 같다)다. NIST 11개는 기본 `rtol`에서 모두 완전 랭크다(Filip 포함).
- **반복 개선** (G15): 증강 시스템 [I X; XᵀW 0][r; β] = [y; 0]에 대한 Björck(1967)의 개선을 같은 QR로 한다. 한 번에:
  1. 잔차의 두 블록 f = y − r − Xβ와 g = −XᵀW r을 double-double로 더한다(Ogita–Rump–Oishi의 Dot2: 곱은 `fma`로 정확히, 합은 two-sum으로). X는 백색화하기 전의 입력을 읽으므로 √w의 반올림은 전처리(QR)에만 들어가고 고정점에는 들어가지 않는다.
  2. h = R⁻ᵀg, d = Qᵀ(√w·f), δβ = R⁻¹(d₁ − h), δr̃ = Q[h; d₂], r += δr̃/√w, β += δβ.
  - 처음 r은 y − Xβ₀를 double-double로 구해 반올림한 값이고, 그 나머지가 첫 f다. 개선 횟수는 상수다(데이터에 따라 멈추지 않는다).
  - Q의 적용은 Accelerate `dormqr` 대신 직접 한다: 벡터 하나에 부른 `dormqr`가 개선 세 번을 도는 적합 시간의 60%를 썼다((10⁵, 10), 10스레드). 반사 하나에 한 번 훑으며 그 반사를 적용하고 다음 반사의 내적을 만든다. 내적은 m으로만 정한 4,096행 덩어리로 나눠 순서대로 더한다(스레드 수와 상관없는 비트, 2¹⁵행부터 스레드풀).
- **잰 값** (`bench/regression/g1/refine.py`, 결과 `refine.json`): Arb 참값(tools/regref.py, 같은 double 입력)에 대한 계수 벡터의 상대오차(노름). NIST 11개는 사용자가 주는 입력(10진 값과 거듭제곱 열을 double로 한 번 반올림)이다. 0번은 QR 해 그대로다.

  | 문제 | κ(X) | 0번 | 1번 | 2번 | 표준오차 (1번) |
  |---|---:|---:|---:|---:|---:|
  | Norris | 8.6e2 | 2e-13 | 9e-17 | 9e-17 | 4e-16 |
  | Pontius | 1.4e13 | 2e-13 | 4e-17 | 4e-17 | 5e-16 |
  | NoInt1 | 1 | 1e-17 | 1e-17 | 1e-17 | 0 |
  | NoInt2 | 1 | 2e-16 | 3e-17 | 3e-17 | 0 |
  | Filip | 1.8e15 | 4e-8 | 2e-13 | 1e-17 | 7e-8 |
  | Longley | 4.9e9 | 6e-13 | 2e-17 | 2e-17 | 2e-13 |
  | Wampler1 | 6.4e6 | 1e-10 | 0 | 0 | 0 (완전 적합) |
  | Wampler2 | 6.4e6 | 9e-15 | 4e-17 | 4e-17 | 0.8 (아래) |
  | Wampler3 | 6.4e6 | 2e-10 | 0 | 0 | 3e-14 |
  | Wampler4 | 6.4e6 | 2e-9 | 0 | 0 | 3e-14 |
  | Wampler5 | 6.4e6 | 2e-7 | 0 | 0 | 3e-14 |

  - 3번, 4번은 2번과 같다. G0의 정규식 보정(R⁻¹Qᵀr)이 고치지 못한 큰 잔차 문제(Longley, Wampler4·5)도 1번에 u 수준이 된다. Filip(κ²u > 1)은 1번에 2e-13, 2번에 1e-17이다.
  - 무작위 문제: X = U diag(s) Vᵀ(n = 100, k = 8, 절편 없음, s는 1에서 1/κ까지 등비), β는 표준정규, 잔차는 X에 직교하고 ‖r‖ = ρ‖Xβ‖. 씨앗 3개의 최댓값, 칸마다 ρ = 10⁻¹⁰ / 1 / 100(R² ≈ 1, 1/2, 10⁻⁴)이다.

  | κ | 0번 | 1번 | 2번 | 3번 | 4번 | 표준오차 (1번) |
  |---:|---|---|---|---|---|---:|
  | 1e0 | 6e-16 / 4e-16 / 1e-14 | 4e-17 / 5e-17 / 5e-17 | 같음 | 같음 | 같음 | 6e-14 |
  | 1e2 | 2e-15 / 4e-14 / 1e-12 | 5e-17 / 4e-17 / 5e-17 | 같음 | 같음 | 같음 | 6e-13 |
  | 1e4 | 6e-14 / 3e-10 / 4e-8 | 7e-17 / 5e-17 / 5e-17 | 같음 | 같음 | 같음 | 9e-13 |
  | 1e5 | 4e-12 / 2e-8 / 3e-6 | 5e-17 / 8e-17 / 5e-17 | 같음 | 같음 | 같음 | 9e-13 |
  | 1e6 | 4e-12 / 5e-7 / 1e-4 | 4e-17 / 4e-17 / 3e-16 | 4e-17 / 4e-17 / 5e-17 | 같음 | 같음 | 3e-11 |
  | 1e7 | 7e-11 / 2e-4 / 2e-2 | 5e-17 / 6e-15 / 5e-13 | 5e-17 / 5e-17 / 6e-17 | 같음 | 같음 | 9e-11 |
  | 1e8 | 2e-10 / 1e-2 / 4e-2 | 6e-17 / 2e-12 / 1e-10 | 6e-17 / 5e-17 / 1e-16 | 같음 | 같음 | 1e-9 |
  | 1e9 | 1e-8 / 1e-1 / 3e0 | 6e-15 / 2e-9 / 7e-9 | 5e-17 / 1e-15 / 1e-14 | 5e-17 / 5e-16 / 2e-16 | 같음 | 2e-8 |
  | 1e10 | 4e-8 / 1e0 / 5e0 | 4e-13 / 2e-7 / 3e-7 | 8e-17 / 4e-13 / 6e-13 | 8e-17 / 1e-15 / 5e-16 | 같음 | 3e-7 |
  | 1e11 | 5e-6 / 1e0 / 2e0 | 2e-11 / 2e-6 / 1e-6 | 2e-16 / 5e-11 / 2e-11 | 6e-17 / 8e-16 / 1e-15 | 같음 | 9e-7 |
  | 1e12 | 9e-5 / 7e0 / 1e0 | 8e-9 / 2e-4 / 9e-6 | 3e-14 / 2e-8 / 3e-9 | 6e-17 / 2e-12 / 8e-14 | 6e-17 / 1e-14 / 9e-16 | 1e-5 |
  | 1e13 | 1e-2 / 5e0 / 4e0 | 3e-6 / 8e-4 / 5e-4 | 2e-8 / 1e-5 / 2e-6 | 9e-12 / 3e-9 / 5e-10 | 2e-14 / 2e-11 / 9e-13 | 2e-4 |
  | 1e14 | 6e-1 / 3e0 / 4e0 | 3e-4 / 9e-4 / 7e-3 | 8e-6 / 4e-5 / 7e-5 | 1e-8 / 3e-8 / 2e-7 | 1e-10 / 7e-10 / 1e-9 | 8e-4 |
  | 1e16 | 2e0 / 1e0 / 2e1 | 2e-1 / 4e-1 / 5e-2 | 9e-1 / 7e-1 / 2e-1 | 6e-1 / 4e-1 / 5e-3 | 1e-1 / 7e-2 / 3e-3 | 3e-1 |

  - 한 번에 오차가 대략 κ·u배(그 0.01–1배) 준다. 처음 오차가 κu + κ²uρ 꼴이므로, s번 뒤는 대략 (κu)ˢ(κu + κ²uρ)다. 큰 잔차(ρ ≥ 1)에서 1e-14가 되는 κ의 한계는 1번 10⁶, 2번 10⁸(10⁹은 ρ = 100에서 1e-14 경계), 3번 10¹¹, 4번 10¹²이다. 작은 잔차(ρ = 10⁻¹⁰)는 1번 10⁹, 2번 10¹¹, 3번 10¹²다. 성분별 최대 상대오차도 같은 한계 안에서 1e-14 아래다(1번, κ = 10⁶, ρ = 100에서 6e-15).
  - κ ≈ 10¹⁶(κu ≈ 1)은 개선이 수렴하지 않는다. 그런 X는 기본 `rtol`에서 열 하나가 겹친다고 판정된다(rank 7/8).
- **개선 한 번의 비용** (`bench/regression/g1/refine_cost.py`, 개선 없는 적합 대비): (10³, 10) 0.018 ms / 0.091 ms, (10⁵, 10) 1.8 / 7.4 ms(10스레드 0.33 / 3.4 ms), (10⁴, 200) 3.3 / 17.7 ms(10스레드 2.0 / 14.1 ms). 한 번에 적합 시간의 10–25%다.
- **결정** (G15, 사용자 2026-11-18): 1번 고정이면 κ ≤ 10⁶까지만 1e-14였다(큰 잔차에서 κ 10⁸ 1e-10, 10¹⁰ 3e-7). 그래서 **필요할 때까지 최대 3번** 보정한다.
  - **멈추는 규칙**: 한 번에 오차가 약 κu배 줄므로, 보정 d 뒤에 남은 오차는 약 ‖d‖·κu다. ‖d‖∞·16·κ̂u ≤ 2⁻⁵⁴‖β‖∞이면 멈춘다. κ̂는 LAPACK `trcon`이 R에서 어림한 1-노름 조건수다(작은 배수 안; 16이 그 여유). 판정은 데이터에만 달려 있어 스레드 수와 무관하게 같은 비트다. 첫 보정은 늘 한다.
  - **계약**: "κ(X) ≤ 10¹¹이면 계수 벡터의 상대오차(노름) ≤ 1e-14". κ는 쓰인 열(가중·백색화, 절편 포함)의 2-노름 조건수(`condition_number`)이고, 잔차가 ‖Xβ‖의 100배까지(R² ≥ 10⁻⁴) 잰 값이다. 무작위 문제(κ별 9개, 3가지 잔차)의 최대: κ 10⁹ 5.4e-16, 10¹⁰ 1.4e-15, 10¹¹ 1.1e-15, 그 밖은 잰 값으로 10¹² 1.6e-12, 10¹³ 3.4e-9, 10¹⁴ 2.5e-7. NIST 11개는 모두 1.3e-17 아래다(Filip도; 1번 고정에서는 1.6e-13).
  - **비용**: 조건이 좋은 문제는 1번과 같다((10⁵, 10) 9.3 ms, 개선 없이 7.4 ms). κ = 10⁹인 (10⁵, 10)은 두 번째 보정까지 해서 12.1 ms다.
- **표준오차**: cov = σ̂²(RᵀR)⁻¹(LAPACK `potri`, R에서). β를 개선해도 R은 그대로이므로 상대오차가 대략 κ·u다(표의 마지막 열: κ 10⁶ 3e-11, 10⁸ 1e-9, 10¹⁰ 3e-7, Filip 7e-8). 잔차가 반올림 수준인 거의 완전한 적합(Wampler2: double 입력의 참 SSR이 아주 작다)은 β를 double로 반올림한 것만으로 잔차가 바뀌어 표준오차의 상대 정확도가 없다(값 자체는 1e-16 수준). σ̂²는 double-double 잔차로 더하므로, 잔차가 크면 β의 반올림은 SSR에 2차로만 들어간다.
- **적합값과 잔차**: 마지막 β로 모든 행(가중치 0인 행도)의 Xβ와 y − Xβ를 double-double로 한 번 더 구해 반올림한다. 같은 합으로 SSR = Σw r², 중심 TSS = Σw(y − ȳ)², 비중심 TSS = Σw y², ESS = Σw(ŷ − ȳ)²와 Σw ŷ²를 double-double로 더한다(ȳ는 가중 평균, double-double). `predict`도 같은 Dot2이고 열 순서도 같아서, 적합에 쓴 X로 부르면 `fittedvalues`와 비트까지 같다.
- **p값과 구간**: t의 두꼬리 p = 2·`StudentT(df).ccdf(|t|)`(1 − cdf가 아니므로 아주 작은 p도 상대 정확도), F의 p = `F(dfn, dfd).ccdf(F)`, 구간은 `iccdf`. 모두 sd.stats의 보장 영역에서 상대 1e-14다.
- **R²**: ESS < SSR(R² < 1/2)이면 ESS/TSS(직접 더한 ESS), 그 밖은 1 − SSR/TSS다. 어느 쪽도 상쇄가 2배를 넘지 않는다. `rsquared_adj`의 1 − R²도 같은 쪽에서 구한다.
- **완전 적합**: SSR이 정확히 0이면 IEEE대로 `bse` 0, t ±inf, p 0, `llf` +inf다. statsmodels는 잔차가 반올림 수준이라 bse ~1e-14, t ~1e13–1e15, p ~1e-90–1e-103을 낸다(numpy 경고와 함께; x = 0..9, y = 1 + 2x + 3x²에서 확인). R은 이 환경에 없어 확인하지 못했다. 계약은 정하지 않았다(§8).

### 3.2 WLS, GLS
- WLS: 가중치 0인 행은 적합과 nobs·자유도에서 빼고 적합값·잔차만 계산한다(R과 같다, G20). statsmodels(0.15.0)는 그 행을 `nobs`에 세므로 자유도와 표준오차가 다르다(60행 중 2행이 0이면 df 57 대 55, 표준오차 1.8% 차이; tools/regref.py로 확인). 가중치가 모두 1이면 OLS와 비트까지 같고, 0인 행이 있는 WLS는 그 행을 뺀 WLS와 비트까지 같다. `llf`에 ½Σlog w(쓰인 행)를 더한다.
- GLS: Σ = LLᵀ(`sd.linalg.cholesky`, 아래 삼각을 읽는다)로 L⁻¹y, L⁻¹X, L⁻¹1을 `solve_triangular`로 만들고, 백색화한 1의 열을 절편 열로(그리고 TSS의 중심화에) 써서 같은 커널로 푼다(statsmodels 0.15의 `dtrtri(cholesky(Σ))`와 같은 백색화). 적합값·잔차는 원래 척도의 Xβ, y − Xβ를 같은 Dot2로 다시 구한다. 중심 TSS는 백색화한 y를 백색화한 1의 열에 사영한 나머지다(statsmodels의 GLS `centered_tss`와 같다). `llf`에서 ½log|Σ| = Σlog L_ii를 뺀다. Σ = I면 OLS, 대각 Σ면 가중치 1/σ_ii의 WLS와 1e-13 안에서 같다.
  - GLS의 정확도는 백색화한 문제에 대한 것이다(L⁻¹의 반올림은 문제 자체를 κ(L)·u만큼 바꾼다).
  - 이 환경의 scipy 1.18.1은 n ≥ 33에서 `scipy.linalg.cholesky(lower=True)`가 틀린 값을 낸다(위 삼각이 0이 아니고 LLᵀ ≠ Σ). 그 함수를 쓰는 statsmodels의 GLS도 그 크기부터 틀리므로, 테스트는 n = 30에서 statsmodels와 비교한다.
- Σ를 추정하는 FGLS(AR(1) 등)는 범위 밖(안).

### 3.3 강건 표준오차
- HC0–HC3: 얇은 Q(`orgqr`)로 지렛대 h_i = ‖Q_i‖²(HC2·HC3만), ω_i = ẽ_i²·(1, n/(n−k), 1/(1−h_i), 1/(1−h_i)²)(ẽ는 백색화한 잔차), 고기 Qᵀdiag(ω)Q를 `syrk`로, 공분산 R⁻¹(Qᵀ diag(ω) Q)R⁻ᵀ를 `trsm` 두 번으로 만든다(XᵀX를 만들지 않는다). HC1의 보정은 nobs/df_resid다. 가중치 0인 행은 빠진다. Arb 참값(regref의 `bse_HC0`–`bse_HC3`)과 1e-13 안이다(κ가 작을 때).
- HC에서도 p값과 구간은 **t(df_resid)**다(G23; R의 `lmtest::coeftest`·Stata와 같다. statsmodels의 기본은 정규분포 `use_t=False`라 `use_t=True`와 같은 값이다). `fvalue`는 절편을 뺀 계수 전부의 Wald 통계량 β₂ᵀV₂₂⁻¹β₂/q이고 p값은 F(q, df_resid)다(statsmodels의 `f_test`와 같다).
- 강건 공분산은 모두 점수 U = diag(ẽ) Q(얇은 Q의 행 q_t에 백색화한 잔차를 곱한 것)에서 고기 S를 만들고 R⁻¹ S R⁻ᵀ로 바꾼다. XᵀX와 (XᵀX)⁻¹을 만들지 않는다(statsmodels는 `pinv`의 (XᵀX)⁻¹을 양쪽에 곱한다).
- **HAC** (G26): S = UᵀU + Σ_l w_l (U_{l:}ᵀU_{:−l} + 그 전치), w_l = 1 − l/(L+1)(Bartlett). 지연 L개의 행렬곱 대신 V_t = Σ_l w_l u_{t−l}(열마다 길이 L의 필터)을 만들어 지연 합 전체를 UᵀV 한 번으로 구한다. V는 4,096행 덩어리로 만들고(열마다 스레드풀) 덩어리의 `dgemm`을 순서대로 더하므로, 메모리는 덩어리 하나이고 비트는 스레드 수와 상관없다. 지연 수의 기본은 floor(4(nobs/100)^(2/9))로 statsmodels와 같은 double 연산이다. 소표본 보정은 하지 않는다: statsmodels의 `cov_type='HAC'`은 `use_correction=False`가 기본이고(True면 n/(n−k)), R `sandwich::NeweyWest`도 `adjust=FALSE`가 기본이다(R의 기본 지연 수는 자동 대역폭이고 사전 백색화를 해서 여기와 다르다). Stata의 `newey`는 n/(n−k)를 곱한다.
- **군집** (G27): S = Σ_g s_g s_gᵀ, s_g는 군집 g의 점수 합(행 순서대로). 군집 수 × 열 수가 2²²(32 MB) 이하이면 G × k 행렬에 행을 순서대로 흩뿌려 더하고(열마다 스레드풀) `syrk` 한 번, 그보다 크면 행을 군집별로 안정 정렬해 군집 덩어리마다 합을 만들고 `syrk`를 누적한다. 두 길의 군집 합은 같은 순서로 더하므로 같은 값이다. 보정은 G/(G−1)·(n−1)/(n−k)(k는 rank)다. G는 적합에 쓰인 행에 나오는 군집 수이고, 하나면 공분산은 NaN이다.
  - 자유도: t와 F의 분모는 G − 1이다. statsmodels(`use_t=True`)와 Stata가 같다(statsmodels는 이때 `df_resid_inference`를 두고 `df_resid`는 그대로; 여기도 `df_resid`는 그대로다). statsmodels의 `get_prediction`은 군집에서도 `df_resid`를 쓰지만 여기의 `predict_interval`은 `conf_int`와 같은 G − 1이다. statsmodels의 k는 X의 열 수라 겹친 열이 있을 때만 다르다. statsmodels의 기본(`use_t=False`)은 정규분포다.
  - 레이블은 Python에서 0..G_tot−1의 번호로 바꾼다(정수·실수 배열은 `unique_inverse`, 문자열은 처음 나온 순서, frame `Series`는 frame의 `factorize`). 없는 레이블은 −1이고 커널이 NaN 데이터로 본다. 커널은 회귀마다 쓰인 행에 나오는 번호를 처음 나온 순서로 다시 매기므로, 결과는 번호를 매긴 방법과 상관없다.
- **가중치 0** (G20): 0인 행은 Q에 행이 없으므로 HAC의 지연은 적합에 쓰인 행을 이어서 세고, 그 행에만 있는 군집은 G에 들지 않는다. 그래서 0인 행이 있는 WLS의 강건 공분산도 그 행을 뺀 WLS와 비트까지 같다. statsmodels(와 R의 `sandwich`)는 그 행을 점수 0인 시점·관측으로 남겨서, 지연의 짝과 G가 다르다.
- **Wald F** (강건형 모두): 절편을 뺀 계수 b₂의 b₂ᵀV₂₂⁻¹b₂/q를 Q의 좌표에서 계산한다. 절편이 첫 열이므로 R⁻¹의 아래 행들은 [0, R₂₂⁻¹]이고, V₂₂ = c·R₂₂⁻¹S₂₂R₂₂⁻ᵀ이므로 b₂ᵀV₂₂⁻¹b₂ = zᵀS₂₂⁻¹z/c, z = R₂₂b₂다. 공분산(조건수가 κ(X)²과 고기의 조건수의 곱)의 역행렬을 만들지 않는다. κ = 10⁵인 군집 문제(ols_cond)에서 V₂₂를 Cholesky로 풀면 F의 상대오차가 2.4e-6, 이 식은 2.7e-11이다. S₂₂가 특이하면(군집 수 − 1 < q) NaN이다.
- **정확도** (Arb 참값 `tests/regression/ref/linear.json`, 같은 double 입력): 관측별 값은 max|차| / max|참값|, 나머지는 상대오차다. Wampler2(잔차가 반올림 수준)와 Wampler1(완전 적합)은 뺐다.

  | 문제 | κ | 지렛대 | 내적 | 외적 | Cook | DW | JB | BP | HAC bse | 군집 bse | 군집 p | 군집 F·p |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | ols_int | 1.6 | 4e-16 | 2e-16 | 2e-16 | 7e-16 | 1e-16 | 7e-16 | 2e-16 | 4e-16 | 5e-16 | 2e-15 | 2e-15 |
  | ols_noint | 1.5 | 1e-16 | 2e-16 | 3e-16 | 5e-16 | 8e-17 | 3e-16 | 8e-16 | 2e-16 | 3e-17 | 4e-16 | 7e-16 |
  | wls_zero | 1.8 | 3e-16 | 3e-16 | 4e-16 | 4e-16 | 3e-17 | 1e-15 | 7e-16 | 2e-16 | 4e-16 | 1e-15 | 2e-15 |
  | wls_noint | 1.8 | 1e-16 | 2e-16 | 2e-16 | 3e-16 | 1e-16 | 2e-15 | 8e-16 | 2e-16 | 2e-16 | 9e-16 | 2e-15 |
  | ols_cond | 1.0e5 | 3e-12 | 6e-14 | 6e-14 | 5e-13 | 5e-17 | 1e-15 | 8e-12 | 4e-13 | 2e-13 | 7e-17 | 3e-11 |
  | ols_kappa11 | 1.0e11 | 4e-6 | 2e-7 | 2e-7 | 1e-6 | 6e-14 | 2e-12 | 6e-7 | 8e-7 | | | |
  | Norris | 8.6e2 | 4e-16 | 4e-14 | 5e-14 | 8e-14 | 3e-14 | 4e-13 | 2e-13 | 2e-14 | | | |
  | Pontius | 1.4e13 | 8e-16 | 2e-13 | 2e-13 | 4e-13 | 5e-15 | 7e-13 | 4e-13 | 2e-14 | | | |
  | NoInt1 | 1 | 3e-16 | 4e-16 | 5e-16 | 6e-16 | 1e-16 | 7e-16 | 2e-14 | 1e-16 | | | |
  | NoInt2 | 1 | 5e-16 | 3e-16 | 5e-15 | 2e-15 | 1e-16 | 4e-16 | 5e-15 | 4e-16 | | | |
  | Filip | 1.8e15 | 4e-8 | 4e-8 | 4e-8 | 4e-8 | 4e-9 | 7e-8 | 2e-8 | 2e-7 | | | |
  | Longley | 4.9e9 | 3e-13 | 1e-12 | 1e-12 | 3e-12 | 1e-16 | 7e-15 | 1e-12 | 1e-12 | | | |
  | Wampler3–5 | 6.4e6 | 2e-14 | 1e-14 | 1e-14 | 5e-14 | 2e-17 | 2e-16 | 8e-16 | 6e-14 | | | |

  - 조건이 좋은 문제는 모두 u의 몇 배다. 오차는 대략 κu를 따른다: 지렛대는 X의 열공간 자체가 κu만큼 움직이므로(Pontius처럼 조건수가 열의 척도에서만 오면 그렇지 않다), 잔차에서 오는 값은 β의 반올림이 작은 잔차에 |x||β|·u로 들어가므로(Norris 4e-14) 그렇다. 테스트의 기준은 max(1e-13, 10κu)다.
  - statsmodels와는 HAC·군집의 bse·p·F, 지렛대, 두 스튜던트화 잔차, Cook 거리, DW, JB, BP 모두 1e-11 안에서 같다(OLS, WLS, GLS; `tests/regression/test_robust_diagnostics.py`).
- **속도** (`bench/regression/g1b/speed.md`, statsmodels 대비): (10⁵, 10)에서 HAC 1.2배(1스레드)·2.3배(10스레드), 군집(1,000개) 1.1·2.0배, 적합과 진단 1.2·2.3배, (10⁶, 50)에서 HAC 1.7·3.1배, 군집 1.8·2.9배, 적합과 진단 1.5·2.4배 빠르다. 진단의 시간은 지렛대를 위한 QR·`orgqr` 한 번과 Breusch–Pagan의 보조 회귀(적합 한 번)이고, 고전형 적합의 시간은 진단이 없을 때와 같다.

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
- **참값 도구** (G0, `tools/regref.py`): 입력(double, 또는 10진 문자열과 거듭제곱을 Arb로 주는 함수)을 정확한 구간으로 받아 정규방정식을 구간 산술로 푼다. 고정밀에서는 조건수의 제곱이 비트를 더 쓸 뿐 틀리게 하지 않는다. 모든 값의 구간이 2⁻¹¹⁰ 상대까지 좁아질 때까지 정밀도를 두 배씩 올린다. 내는 값: 계수, 표준오차, t, p, 신뢰구간, 적합값, 잔차, 지렛대, 내적·외적 스튜던트화 잔차(외적은 행을 뺀 척도의 정의대로), Cook 거리, SSR·ESS·TSS, R²·수정 R², F와 p, 척도, 로그우도·AIC·BIC, HC0–HC3·HAC(지연 수를 받는다) 표준오차, 군집 레이블을 주면 군집 표준오차·p·Wald F와 p, Durbin–Watson, Jarque–Bera(중심 적률)와 p, Breusch–Pagan(스튜던트화 판과 원래 판, F 판). 가중치 0인 행은 R처럼 뺀다.
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
- **G1**: 정했다(2026-11-18, G9–G30).
- **G2**: 분포족·링크 목록, 반복 상한에 닿았을 때(NaN / 마지막 값 + `converged=False`), 척도 추정, 분리의 처리, 음이항의 α를 추정할지
- **G3**: 목적함수의 꼴과 표준화(scikit-learn / glmnet), α의 이름(`alpha`·`l1_ratio` / `lambda`), 교차검증의 범위, 정규화 GLM(로지스틱), 추론 필드를 둘지
- **G4**: M 추정 함수 목록, 척도 추정, Theil–Sen·Siegel(scipy의 `theilslopes`, `siegelslopes`)을 넣을지
- **G5**: 표준오차 방법, 여러 분위를 한 번에 받을지
- **G6**: 이름과 API(scipy의 `curve_fit`과 `least_squares` 가운데 어느 꼴), 방법, 야코비안을 사용자가 줄 수 있게 할지, 범위 제약
