# 회귀 영역 설계 (sd.regression)

> 상태: **G1–G6 구현·승인** (2026-11-19): 선형 모형, GLM, 정규화, 강건·분위 회귀, 비선형 최소제곱의 수치 핵심(§2, §3.1–§3.7). 큰 결정은 사용자가 정했다(§1.1). 그 뒤의 사이트 작업과 코드 리뷰에서 찾은 문제와 그 사용자 결정 G83–G116(2026-11-19–20)을 반영했다. 나머지는 안이고, 마일스톤마다 시작 전에 물을 것은 §8에 있다.
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
| G31 분포족 (G2) | 정규(identity·log), 이항(logit·probit·cloglog·log), 포아송(log·identity·sqrt), 감마(inverse·log·identity). 역가우스·음이항·Tweedie는 넣지 않는다 (사용자 결정 2026-11-18) |
| G32 미수렴 (G2) | 반복 상한(`max_iter=`, 기본 100)에 닿으면 마지막 값을 내고 `converged=False`, `n_iter`는 상한이다. 경고는 없다 (사용자 결정 2026-11-18). 2026-11-19: 상한이 판정과 연마가 끝나기 전에 멈췄을 때만 False다(마침 상한에서 끝나면 `n_iter` = 상한이고 True; G93) |
| G33 완전 분리 (G2) | 감지하면 계수·표준오차·t·p를 NaN으로 두고 `separation=True`. 적합값 μ는 0·1로 낼 수 있다. 최대우도 추정이 존재하지 않으므로 "맞는 값 또는 NaN"을 따른다 (사용자 결정 2026-11-18) |
| G34 척도 (G2) | 정규·감마의 척도 기본은 Pearson χ²/df(statsmodels·R과 같다). `scale=`로 편차/df나 고정값. 이항·포아송은 1 (사용자 결정 2026-11-18) |
| G35 이항 반응 (G2) | y는 0/1이거나 [0, 1]의 비율이고, 비율이면 `trials=`(시행 수)를 준다. (성공, 실패) 두 열은 받지 않는다 (사용자 결정 2026-11-18) |
| G36 가중치 (G2) | `var_weights=`(분산을 1/w로, R의 사전 가중치)와 `freq_weights=`(관측의 반복 수)를 이름으로 나눠 받는다(statsmodels와 같은 뜻) (사용자 결정 2026-11-18) |
| G37 GLM의 강건 표준오차 (G2) | G1과 같은 `cov_type`(HC0–HC3, HAC, 군집)을 받는다 (사용자 결정 2026-11-18) |
| G38 GLM의 검정 분포 (G2) | R과 같다: 척도가 고정인 분포족(이항·포아송)은 정규분포(z), 추정하는 분포족(정규·감마)은 t(df_resid). 강건 공분산은 G23·G27을 따른다 (사용자 결정 2026-11-18) |
| G39 강건 GLM의 분포 (G2) | 척도가 고정인 분포족(이항·포아송)은 강건 공분산(HC·HAC)에서도 z다(R의 coeftest·Stata와 같다). 군집이면 t(G−1). 정규·감마는 t. G38의 "강건은 G23·G27을 따른다"를 이렇게 고친다 (사용자 결정 2026-11-18) |
| G40 GLM의 중간 반복 (G2) | 중간 반복은 정규방정식(XᵀWX의 Cholesky; 불안정하면 QR)으로 풀고, 수렴 뒤의 연마와 마지막 풀이는 QR + 반복 개선이다. 최종 정확도는 마지막 풀이가 정한다 (사용자 결정 2026-11-18) |
| G41 GLM의 세부 (사용자 확인 2026-11-18) | 이항·포아송도 `scale=`(준우도)을 받고 그때는 t, Pearson 척도의 이름은 `'X2'`. 정규·log에서 y ≤ 0인 행은 평균에서 시작(평균 ≤ 0이면 `ValueError`). `trials`는 > 0만, trials 없는 이항의 y는 0/1, `exposure`는 log 링크에서만. 필드 `linear_predictor`·`llnull`·`resid_working`, `resid_anscombe`는 없다 |
| G42 비정준 링크의 마지막 걸음 (G2) | 연마와 마지막 풀이의 뉴턴 걸음을 얇은 Q로 푼다(관측 정보 가중치가 음수인 행이 있어도 정규방정식 꼴로 풀지 않는다). 비정준 링크에서만 QR 한 번의 비용이 더 들고, 정확도는 정준 링크와 같은 수준이 목표다 (사용자 결정 2026-11-18) |
| G43 작은 문제의 중간 반복 (G2) | 행 수 × 열 수 < 512이면 중간 반복도 QR로 푼다(재서 QR이 더 빨랐다; 정확도는 같고 결과 비트만 경로에 따라 다를 수 있다) (사용자 확인 2026-11-18) |
| G44 느린 수렴 (G2) | 수렴 판정 전에도 수렴비가 느린 것이 보이면 관측 정보의 뉴턴 걸음(G42의 얇은 Q 풀이)으로 바꾼다. 판정은 데이터에만 달린 수렴비 기준이다 (사용자 결정 2026-11-18) |
| G45 정규화의 목적함수 (G3) | scikit-learn과 같다: (1/2n)‖y − Xβ‖² + alpha·(l1_ratio‖β‖₁ + (1 − l1_ratio)/2‖β‖²), 절편은 벌점 밖. 인자 이름 `alpha`·`l1_ratio`. 기본은 표준화 없음(scikit-learn과 같은 해), `standardize=True`면 glmnet처럼 표준화한 계수에 벌점을 주고 원래 척도로 돌려준다 (사용자 결정 2026-11-18) |
| G46 경로와 교차검증 (G3) | `alpha`에 1차원 배열을 주면 경로 전체를 따뜻한 시작으로 한 번에 푼다. `ridge_cv`·`lasso_cv`·`elastic_net_cv`(k겹, 최소 오차와 1-SE 규칙 둘 다 보고, 겹 나누기는 `rng=`), ridge는 GCV도 (사용자 결정 2026-11-18) |
| G47 정규화 GLM (G3) | 이항·포아송을 넣는다(`glm`과 같은 family). 바깥 IRLS + 안쪽 좌표 하강(glmnet). 함수 이름은 구현 때 묻는다 (사용자 결정 2026-11-18) |
| G48 정규화 결과 (G3) | 예측 중심: `params`(절편 `params[0]`), `predict`, 잔차·R², 유효 자유도(ridge는 tr H, lasso는 0 아닌 계수 수), `converged`·`n_iter`·쌍대 간격. 표준오차·p값 필드는 두지 않는다 (사용자 결정 2026-11-18) |
| G49 정규화 GLM의 이름 (G3) | `penalized_glm`(결과 `PenalizedGLMResult`) (사용자 결정 2026-11-18) |
| G50 alpha = 0의 정규화 GLM (G3) | 벌점이 없으면 `glm`과 같은 규칙: 분리 증명서로 판정해 계수를 NaN으로 두고 `separation=True`(G33). alpha > 0이면 해가 유한하므로 판정하지 않는다 (사용자 결정 2026-11-18) |
| G51 ridge의 개선 (G3) | 개선 잔차를 double-double로 구한다(G1의 OLS와 같은 정확도 수준) (사용자 결정 2026-11-18). lasso·엘라스틱 넷의 KKT 연마의 개선도 같다 (사용자 결정 2026-11-18) |
| G52 G3의 세부 (사용자 확인 2026-11-18) | 경로 축은 배치 축 뒤(`params` (…, L, k)). 가중치 인자 `weights`. ridge에는 `converged`·`n_iter`·`dual_gap`이 없다. `ssr` 필드. CV의 `select='min'|'1se'`(기본 min), ridge 기본 격자는 glmnet 규칙, `elastic_net_cv`는 l1_ratio 하나, 배치는 같은 겹을 공유. 정규화 GLM은 정준 링크만, offset·trials 없음, l1_ratio 기본 0.5. `summary()` 없음. ridge는 정확도를 위해 QR + SVD(정규방정식을 쓰지 않음) |
| G53 강건 추정 방식 (G4) | `rlm(method='M')`(기본, 최소제곱에서 시작하는 IRLS, statsmodels RLM과 같은 결과)과 `method='MM'`(S 추정에서 시작해 bisquare M으로, R robustbase lmrob의 방식; 무작위 부분표본이라 `rng=` 필수) (사용자 결정 2026-11-18) |
| G54 ψ 함수 (G4) | Huber, Tukey bisquare, Hampel. 조율 상수의 기본은 Huber 1.345·bisquare 4.685(정규 효율 95%의 표준값)와 Hampel (2, 4, 8)(statsmodels 기본; 정규 효율 0.990) (사용자 결정 2026-11-18) |
| G55 척도 (G4) | M 추정은 반복마다 MAD(1.4826 × 잔차 MAD)를 갱신한다(statsmodels 기본). `scale_est='huber'`로 Huber의 제안 2 (사용자 결정 2026-11-18) |
| G56 Theil–Sen·Siegel (G4) | `theilslopes(y, x)`, `siegelslopes(y, x)`를 scipy와 같은 결과 꼴로 둔다. 중앙값은 정확히 구하므로 scipy와 같은 값이어야 한다 (사용자 결정 2026-11-18) |
| G57 rlm의 추론 (G4) | M은 Huber의 H1(statsmodels 기본), MM은 lmrob의 샌드위치. 분포는 t(df_resid)(R의 요약과 같다; statsmodels는 z) (사용자 결정 2026-11-18) |
| G58 MM의 큰 n (G4) | n > 2000이면 robustbase의 큰 n 방식(구간별 부분표본 후 병합)을 쓴다. 작은 n은 그대로 (사용자 결정 2026-11-18) |
| G59 G4의 세부 (사용자 확인 2026-11-18) | `norm`·`scale_est`의 기본은 None(M은 huber·MAD, MM은 bisquare·S 척도; MM에 huber·hampel이나 `scale_est`를 주면 `ValueError`). MM의 상수는 lmrob 값(1.54764, 4.685061), M의 bisquare 기본은 statsmodels 값(4.685). 제안 2의 d = 2.5 고정. `tol`은 GLM과 같은 상대 기준, `max_iter` 기본 50. `weights` 없음. 완전 적합은 척도 0. 부분표본 수 500 고정. `method`·`norm`은 공개 필드 |
| G60 분위 회귀의 풀이 (G5) | Frisch–Newton 내부점법으로 풀고 마지막에 기저(잔차 0인 k행)를 찾아 정확한 꼭짓점 해를 낸다. 모든 n에서 같은 방법이다(크기에 따라 경로를 바꾸지 않는다) (사용자 결정 2026-11-18) |
| G61 분위 회귀의 표준오차 (G5) | 기본 `'nid'`(Hendricks–Koenker 샌드위치, Hall–Sheather 대역폭; R summary.rq의 큰 n 기본), 옵션 `'iid'`(Koenker–Bassett)와 `'ker'`(Powell 커널). p값과 구간은 t(n − k)(R과 같다) (사용자 결정 2026-11-18) |
| G62 여러 분위 (G5) | `q`에 1차원 배열을 주면 결과에 분위 축이 생긴다(G52의 경로 축과 같은 자리). 앞 분위의 해에서 따뜻하게 시작한다 (사용자 결정 2026-11-18) |
| G63 유일하지 않은 해 (G5) | 최적 꼭짓점 하나를 결정적으로 내고 `nonunique=True`로 알린다(목적함수 값은 유일하다). 어느 꼭짓점인지의 규칙은 문서에 적는다 (사용자 결정 2026-11-18) |
| G64 분위 회귀의 수렴 (G5) | `converged`는 돌려준 꼭짓점이 최적성 증명서(기저 행의 쌍대 변수가 [q − 1, q] 안)를 갖췄는지다. 내부점법이 `max_iter`에 닿아도 심플렉스가 증명서에 닿으면 True다 (사용자 결정 2026-11-19) |
| G65 분위 회귀의 정의되지 않는 표준오차 (G5) | 희소도가 정의되지 않으면(완전 적합, q가 0이나 1에 아주 가까울 때) `bse`는 NaN이다. OLS의 G22처럼 0으로 두지 않는다 (사용자 결정 2026-11-19) |
| G66 10진 q의 동률 (G5) | 최적 기저의 쌍대 변수가 경계에서 32u 안이면 동률로 보고 `nonunique=True`다(n = 10, q = 0.3처럼 double의 nq가 정수에서 반올림만큼 벗어난 경우) (사용자 결정 2026-11-19) |
| G67 summary.rq의 eps (G5) | 'nid'의 f_i 분모와 'iid'의 pz(\|r_i\| < eps)에 쓰는 eps는 2⁻²⁶·s, s = 적합에 쓰인 행의 max\|y_i\|(`nan_policy='omit'` 뒤; s = 0이면 1). R의 절대 eps는 작은 단위의 y에서 표준오차를 틀리게 하고, 상대 eps로는 y에 c를 곱하면 `params`·`bse`가 c배다 (사용자 결정 2026-11-19) |
| G68 비선형의 이름과 꼴 (G6) | `nls(f, x, y, /, *, p0, ...)`, 결과는 `NLSResult`(G10의 결과 객체; R의 이름). 모형 함수는 scipy `curve_fit`의 꼴 f(x, *θ)다. `least_squares` 꼴의 일반 풀이는 두지 않는다 (사용자 결정 2026-11-19) |
| G69 비선형의 방법 (G6) | 범위 제약이 없으면 Levenberg–Marquardt(MINPACK `lmder`의 신뢰 영역 꼴), 있으면 신뢰 영역 반사(TRF). scipy `curve_fit`의 기본 규칙과 같고 `method=`로 고를 수 있다 (사용자 결정 2026-11-19) |
| G70 비선형의 야코비안 (G6) | 사용자가 `jac`을 줄 수 있다. 없으면 반복은 2-point 차분, 해에서 공분산과 다듬기에 쓰는 야코비안은 더 정확한 차분(중앙 차분과 Richardson 등)이다 (사용자 결정 2026-11-19) |
| G71 사용자 함수의 입력 (G6) | f에는 x를 sd 배열(float64)로, θ를 float 인자들로 넘긴다 (사용자 결정 2026-11-19) |
| G72 비선형의 가중치와 척도 (G6) | `weights=`만 둔다(G20과 같은 뜻: 0이면 적합과 `nobs`에서 빠진다). 공분산은 늘 σ̂²(JᵀWJ)⁻¹다(R `nls`와 같다). scipy의 `sigma`는 weights = 1/sigma²에 해당한다 (사용자 결정 2026-11-19) |
| G73 비선형의 수렴과 정확도 (G6) | scipy와 같은 ftol·xtol·gtol로 멈춘 뒤, 정밀 차분(또는 사용자) 야코비안의 가우스–뉴턴 걸음을 계수가 반올림보다 크게 움직이는 동안 이어 간다(G32의 rlm·glm과 같은 정신) (사용자 결정 2026-11-19) |
| G74 비유한 함수값 (G6) | 시험점에서 f가 NaN이나 inf를 내면 그 걸음을 거절하고 신뢰 영역을 줄인다. p0에서 비유한하면 `ValueError` (사용자 결정 2026-11-19) |
| G75 x의 모양 (G6) | x는 (n,) 또는 (n, d)(첫 축이 관측)이고 f는 (n,)을 돌려준다. `nan_policy='omit'`은 y나 x가 NaN인 행을 뺀다 (사용자 결정 2026-11-19) |
| G76 비선형의 겹친 모수 (G6) | `params`는 반복이 멈춘 값 그대로 숫자로 두고, 추론(`bse`, t, p, 구간, `cov_params()`의 그 행과 열)만 NaN이다 (사용자 결정 2026-11-19) |
| G77 비선형의 rtol 기본값 (G6) | 차분 야코비안이면 2⁻³⁰, 사용자 `jac`이면 선형 모형과 같은 max(n, p)·u (사용자 결정 2026-11-19) |
| G78 비선형의 llf·aic·bic (G6) | `NLSResult`에 `llf`, `aic`, `bic`를 둔다. ols·wls와 같은 식과 모수 수다: 정규 우도의 최댓값 −nobs/2·(log 2π + log(ssr/nobs) + 1) + ½Σlog w(쓰인 행), aic = −2·llf + 2·rank, bic = −2·llf + log(nobs)·rank(척도는 세지 않는다). R `logLik.nls`와 같은 llf이고, R의 `AIC`·`BIC`는 σ를 세어 2, log(nobs)만큼 크다 (사용자 결정 2026-11-19) |
| G79 비선형의 차분 비용 (G6) | 차분 야코비안의 다듬기 비용(f 호출이 scipy `curve_fit`의 몇 배)은 그대로 둔다. 다듬기를 끄는 옵션은 두지 않는다. 사용자 `jac`이 그 비용의 대부분을 없앤다 (사용자 결정 2026-11-19) |
| G80 정준 링크의 연마 반복 (G2) | 정준 링크의 연마 반복도 정규방정식(XᵀWX의 Cholesky)으로 푼다. 다만 κ(U)²u ≤ 10⁻¹⁰(κ는 중간 반복과 같은 `trcon` 어림)일 때만이고, 아니면 QR이다. 정준이 아닌 링크(G42의 얇은 Q)는 그대로다. 마지막 풀이는 QR + 반복 개선이라 정확도는 거기서 정해진다. G40의 "연마는 QR"을 이렇게 고친다 (사용자 결정 2026-11-19) |
| G81 정규화 로지스틱의 μ (G3) | `penalized_glm`(이항)의 μ와 1 − μ를 `glm`의 logit 링크와 같은 sd.special `expit` 커널로 구한다. 같은 η면 두 함수의 μ가 비트까지 같다 (사용자 결정 2026-11-19) |
| G82 MM의 S 고정점 (G4) | S 추정의 마지막 고정점 반복에서 매번 M 단계와 같은 뉴턴 걸음(추정방정식 Σψ(r_i/s(β))x_i, M-척도 s(β)의 β 의존 포함)을 먼저 해 보고, ‖F‖를 줄일 때만 받는다(아니면 가중 최소제곱 한 걸음). 고정점은 같다. 테스트의 S 기준값은 Arb로 증명한 고정점이다(`tools/regref.py`의 `s_estimate`) (사용자 결정 2026-11-19) |
| G83 x=None과 omit (G4) | `theilslopes`·`siegelslopes`에서 `x`가 None이고 `nan_policy='omit'`이면 빠진 점을 뺀 뒤에도 남은 점의 x는 원래 위치다(scipy는 다시 번호를 매긴다: [scipy와 다름]). 남은 점은 원래 자리에 있기 때문이다 (사용자 결정 2026-11-19) |
| G84 0의 부호 (전체) | 회귀의 모든 함수에서 정확히 0인 계수는 +0.0이다(−0.0을 내지 않는다; `theilslopes`의 규칙). 결과를 만들 때 계수에 +0.0을 더한다 (사용자 결정 2026-11-19) |
| G85 TSS가 0 (G1, G3) | y가 모두 같아 TSS가 정확히 0이면 `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`는 NaN(0/0), `ess`는 0이다(ols·wls·gls, 강건 공분산의 F도). 정규화의 `rsquared`·`deviance_ratio`도 같다. statsmodels의 −inf·−4는 따르지 않는다. 계수는 계약 정확도 안의 값 그대로다(`linregress`만 기울기 +0, 절편 y₀) (사용자 결정 2026-11-19) |
| G86 입력의 복사 (전체) | 나중에 계산하는 필드가 읽는 입력(`y`, `X`, `weights`)은 적합할 때 복사한다. 넘긴 배열을 나중에 고쳐도 결과가 바뀌지 않는다. 메모리는 X 하나만큼 더 든다(API §1) (사용자 결정 2026-11-19) |
| G87 ridge_cv의 격자 (G3) | `ridge_cv`의 `eps` 기본은 1e-6이다(`lasso_cv`·`elastic_net_cv`는 1e-3 그대로). 1e-3이면 격자 끝이 lasso의 alpha_max라 최적 alpha가 격자 끝에 걸렸다 (사용자 결정 2026-11-19) |
| G88 nls의 잡음 열 (G6) | 차분 야코비안의 열의 노름이 그 차분의 반올림 수준(5점 공식의 Σ|w|/(12h)·u·‖f‖, 16배의 여유) 아래면 그 열은 0으로 보고 겹친 모수로 낸다. 식별되는 모수가 잡음 열 대신 빠지지 않게 한다 (사용자 결정 2026-11-19) |
| G89 nls의 넘친 ssr (G6) | 판정을 통과했어도 `ssr`이 유한하지 않으면 `converged=False`다([scipy와 다름]) (사용자 결정 2026-11-19) |
| G90 DataFrame의 예측 (전체) | 적합과 예측이 모두 `DataFrame`이면 열 이름과 순서가 같아야 하고, 아니면 `ValueError`(scikit-learn과 같다). 한쪽만 `DataFrame`이면 위치로 맞춘다(scikit-learn은 경고; 여기는 경고 없음) (사용자 결정 2026-11-19) |
| G91 예측의 inf (전체) | `X_new`에 inf가 있으면 `ValueError`(`'X_new' contains infinity`, 적합의 입력과 같은 규칙). NaN은 그 행의 예측이 NaN (사용자 결정 2026-11-19) |
| G92 상수 y (G1) | 절편이 있고 y가 적합에 쓰인 행(가중치 > 0)에서 정확히 모두 같으면 ols·wls·gls(배치 포함)는 정확한 해 (y₀, 0, …, 0)를 낸다(겹친 열은 G9·G21대로 NaN). 적합값 y₀, 잔차 y − y₀, SSR·TSS·ESS 0이라 완전 적합(G22: `bse` 0)이고 R²·F는 G85의 NaN이다. GLS는 절편 열이 모형에 있어 백색화한 문제에서도 같은 해다. 절편이 없으면 해가 자명하지 않아 하지 않는다 (사용자 결정 2026-11-19) |
| G93 수렴과 연마 (전체) | `converged`는 수렴 판정을 통과했고 그 뒤의 연마(반올림 수준까지의 반복)도 끝났을 때만 True다. 연마가 반복 상한에 잘리면 False다: `glm`(IRLS의 연마; 상한에 잘린 반복 뒤 마지막 풀이만 판정을 통과한 경우도 False), `penalized_glm`(alpha = 0 포함; 다시 돈 좌표 하강이 `max_iter`에 닿아 연마하지 못하면 False), `lasso`·`elastic_net`(같다), `rlm` M·MM(MM은 S 추정의 다듬기와 고정점 반복이 끝났는지도 본다), `nls`(다듬기의 f 호출이 `nfev`에 들고 `max_nfev`에서 남은 만큼만 쓴다; 해의 공분산용 야코비안은 그 밖이다). `quantreg`의 `converged`는 증명서(G64)라 그대로다. `n_iter` ≤ `max_iter`는 모든 함수에서 성립한다: `penalized_glm`의 l1_ratio = 0은 처음 뉴턴 걸음을 `max_iter`로 자르고 남은 만큼 IRLS를 돈다(뉴턴이 상한에 잘리면 그 반복점에서 이어 간다; 예전에는 귀무 모형을 냈다) (사용자 결정 2026-11-19) |
| G94 상수 y의 확장 (G92) | G92의 정확한 해를 `glm` 정규·identity(절편, offset 없음)와 `rlm` M·MM(절편)에도 쓴다: (y₀, 0, …, 0), 적합값 y₀, 척도 0. 그래서 기울기의 t는 0/0 = NaN이고 반올림이 남긴 기울기(1e-33)로 p ≈ 0이 나오지 않는다. 0/0인 요약값은 NaN이다: `quantreg`의 `prsquared`, GLM의 `pseudo_rsquared()`·`llr`(llf = llnull = +inf). 다른 GLM 분포족은 정확한 해가 자명할 때만(정규·log, 감마: 적합이 정확하면 G97). `nls`는 모형을 모르므로 하지 않고, 0/0인 곳만 NaN이다 (사용자 결정 2026-11-19) |
| G95 df_resid = 0과 완전 적합 (전체) | `df_resid` = 0이면 추론(`scale`, `bse`, t, p, 구간, `cov_params()`, 강건 공분산도)이 0/0이라 NaN이다(ols·wls·gls·`linregress`(점 2개; scipy 1.18.1도 NaN), 척도를 추정하는 `glm`, `rlm`, `nls`; `quantreg`는 원래 그렇다). 척도가 고정인 GLM(이항·포아송)은 0/0이 아니라 그대로다. 완전 적합의 판정은 `rlm`·`quantreg`와 같은 허용오차로 통일한다: 쓰인 행마다 \|r_i\| ≤ (k + 16)u(\|y_i\| + ‖x_i‖₁‖β‖∞)이면 SSR을 0으로 둔다(잔차 필드는 계산한 값 그대로). df_resid > 0이면 G22대로 `bse` 0, t ±inf, p 0, `llf` +inf (사용자 결정 2026-11-19) |
| G96 빈 입력 (전체) | 행이 없으면(n = 0, `nan_policy='omit'` 뒤 남은 행이 없음, 가중치가 모두 0) 결과는 NaN이고 `nobs` 0, `converged` False, `n_iter` 0이다(sd.stats의 빈 입력 규칙): ols·wls·gls·glm·정규화(교차검증 포함)·rlm(MM 포함; 예전에는 getrf의 `RuntimeError`)·quantreg·nls. 배치에서는 그 회귀만 NaN이다. `df_model`·`df_resid`도 NaN이다(−1, 0이 아니다). `linregress`·`theilslopes`·`siegelslopes`는 scipy의 꼴을 지켜 점이 2개보다 적으면 `ValueError`(`linregress(): 'x' and 'y' must have length at least 2`)다 (사용자 결정 2026-11-19) |
| G97 φ = 0의 로그우도 (G2) | 척도를 추정하는 GLM이 완전 적합이라 φ = 0이면 `llf`는 +inf다(OLS의 G22와 같다; `aic`·`bic`는 −inf). `llnull`은 같은 φ = 0에서 귀무 모형도 정확하면 +inf, 아니면 −inf다(우도의 극한) (사용자 결정 2026-11-19) |
| G98 읽기 전용 설정 필드 (전체) | 결과의 설정 필드(`param_names`, `cov_type`, `family`, `link`, `alpha`, `l1_ratio`, `standardize`, `method`, `norm`, `q`)는 모든 결과 클래스에서 읽기 전용 속성이다. `quantreg`의 배열 `q`는 복사본이다 (사용자 결정 2026-11-19) |
| G99 conf_int (전체) | `conf_int(*, alpha=0.05)`: `alpha`는 키워드로만 받는다(모든 결과) (사용자 결정 2026-11-19) |
| G100 GLM의 pseudo R² (G2) | `GLMResult.pseudo_rsquared(*, kind='cs')`: statsmodels의 이름과 기본(Cox–Snell, `summary()`가 쓰는 값). `kind='mcf'`는 McFadden. 0/0이면 NaN (사용자 결정 2026-11-19) |
| G101 모수 없는 모형 (전체) | X에 열이 없고 `intercept=False`이면 모든 함수에서 `ValueError`(`ols(): the model has no parameters (no columns in 'X' and intercept=False)`; `nls`는 `'p0' is empty`)다(`rlm`·`quantreg`가 하던 것) (사용자 결정 2026-11-19) |
| G102 linregress의 nan_policy (G1) | `linregress(x, y, /, *, nan_policy='propagate')`: `theilslopes`의 규칙('omit'은 배치에서 `ValueError`, inf는 언제나 `ValueError`) (사용자 결정 2026-11-19) |
| G103 강건·분위·비선형의 예측 구간 (G4–G6) | `rlm`, `quantreg`, `nls`에 `predict_interval(X_new, /, *, alpha=0.05)`을 둔다. 적합값(평균의 위치)의 신뢰구간만이다: x β ± t·se, se² = x C xᵀ(`nls`는 해에서의 야코비안으로 델타 방법). 새 관측의 예측 구간은 두지 않는다: 강건 척도는 오차분포의 표준편차가 아니고, 분위 하나는 조건부 분포의 퍼짐을 말하지 않으며, 비선형 적합은 새 관측의 오차분포를 주지 않기 때문이다 (사용자 결정 2026-11-19) |
| G104 nls의 예측 입력 (G6) | `nls`의 `predict`·`predict_interval`도 G90(적합과 예측이 모두 `DataFrame`이면 열 이름과 순서가 같아야 한다)과 G91(`x_new`의 inf는 `ValueError`)을 따른다 (사용자 결정 2026-11-19) |
| G105 열 하나의 y (전체) | 열이 하나인 frame `DataFrame`의 y는 그 열(n,)로 읽는다. sd 배열 (n, 1)은 배치 규칙 그대로(행 하나짜리 회귀 n개)이지만, X의 행 수가 맞지 않으면 메시지가 그 경우를 짚는다(`'X' must have 1 row like 'y', not 100 ('y' of shape (100, 1) is a batch of regressions with 1 row each; ...)`; "1 rows"의 문법도 고친다) (사용자 결정 2026-11-19) |
| G106 alpha = 0의 정규화 (G3) | `ridge`·`lasso`·`elastic_net`의 alpha = 0은 최소 노름의 최소제곱 해다(G110이 lasso·엘라스틱 넷의 값을 정한다: ridge(alpha=0)과 같은 해). 겹친 열은 최소 노름으로 나뉜다. G50(정규화 GLM의 alpha = 0은 `glm`의 규칙: 겹친 열 NaN, 분리 증명서)과 다른 것은, 선형 정규화의 결과가 예측 중심이라(G48) 경로를 따라 계수가 끊기지 않는 것이 겹친 열의 NaN보다 쓸모 있기 때문이다 (사용자 결정 2026-11-19) |
| G107 빈 겹 (G3) | 교차검증에서 시험 행의 가중치가 모두 0인 겹은 `cv_error`의 평균과 `cv_se`에서 뺀다(잴 오차가 없다). 모든 겹이 비는 것은 가중치가 모두 0인 경우뿐이고, 그때는 G96의 NaN 결과다 (사용자 결정 2026-11-19) |
| G108 군집 하나와 상수 y (G1) | `cov_type='cluster'`에 군집이 하나이고 y가 상수이면, G92의 공분산 0이 아니라 군집 하나의 NaN이 이긴다 (사용자 결정 2026-11-19) |
| G109 nls의 완전 적합 (G6) | `nls`에서 `ssr`이 정확히 0이고 `df_resid` > 0이면 `bse`는 0 그대로이고 `tvalues`·`pvalues`는 NaN이다(θ/0의 ±inf가 가짜 유의성을 보이지 않게; 모형을 모르므로 상수 y의 a + b·x에도 반올림 크기의 b가 남는다). 구간은 `bse` 0이라 [θ, θ]이고, `predict_interval`의 폭도 0이다. ols의 G22(t ±inf, p 0, 구간 [θ, θ])와는 t·p만 다르다: 선형 모형은 완전 적합의 해가 정확해 t의 극한이 뜻이 있지만, 비선형 반복이 멈춘 점은 그렇지 않다 (사용자 결정 2026-11-19) |
| G110 lasso·엘라스틱 넷의 alpha = 0 (G3) | `lasso`·`elastic_net`의 alpha = 0은 최소 노름의 최소제곱 해이고 `ridge(alpha=0)`과 비트까지 같다(같은 SVD, p > n도). `converged` True, `n_iter` 0, `dual_gap` 0이고, `df_model`은 0 아닌 계수의 수다(G115). 경로나 교차검증의 겹에 0이 있으면 그 자리가 그 값이다. 최소제곱 해가 유일하지 않으면 lasso 경로의 alpha → 0+ 극한(최소 L1 노름 쪽)과 다를 수 있다(경로가 0에서 끊길 수 있다). 좌표 하강은 그런 문제에서 최소점 가운데 시작점에 달린 하나에 멈춘다 (사용자 결정 2026-11-19) |
| G111 GLM의 상수 y (G2) | 절편이 있고 offset이 없으며 y가 쓰인 행에서 정확히 같을 때(y₀), g(y₀)가 유한하고 V(y₀) > 0이면 모든 분포족·링크에서 정확한 해 (g(y₀), 0, …, 0), μ = y₀다(G94의 정규·identity를 넓힌다). 편차·Pearson χ²·귀무 편차 0, `llnull` = `llf`. 척도를 추정하면 완전 적합의 규칙(척도 0, 기울기의 p NaN, `llf` +inf)이고, 척도가 고정인 이항·포아송은 그 공분산 그대로라 기울기의 t 0, p 1이다(가짜 유의성은 없다; 척도가 고정이라 완전 적합의 0/0이 생기지 않는다). g(y₀)가 무한하면(이항 모두 0·1, 포아송 log 모두 0) 분리·경계의 규칙 그대로다 (사용자 결정 2026-11-19) |
| G112 McFadden의 pseudo R² (G2) | `pseudo_rsquared(kind='mcf')`는 `llf`가 무한하면(척도 0의 완전 적합) NaN이다(`llnull`이 0이어도 NaN). Cox–Snell은 극한 1을 그대로 낸다 (사용자 결정 2026-11-19) |
| G113 설정 필드의 복사 (전체) | 리스트나 배열인 설정 필드(`param_names`, 배열의 `alpha`·`q`)는 읽을 때마다 새 복사본(리스트는 새 리스트, 배열은 새 sd 배열)이다. 돌려받은 것을 고쳐도 결과가 바뀌지 않는다(G98의 읽기 전용을 내용까지) (2026-11-19, R88) |
| G114 GLM의 경계 상수 y (G2) | 절편이 있고 offset이 없으며 y가 쓰인 행에서 정확히 같을 때(y₀) g(y₀)가 유한한데 y₀가 분포족 범위의 경계(V(y₀) = 0)이면, 곧 포아송 identity·sqrt의 모두 0과 이항 log의 모두 1(비율도)이면, 정확한 해 (g(y₀), 0, …, 0) = (0, …, 0), μ = y₀다: `converged` True, `n_iter` 0, `separation` False, 쓰인 행의 잔차 0, 편차·Pearson χ²·귀무 편차 0, `llf` = `llnull` = 0. V(y₀) = 0이라 정보가 없으므로 공분산 전체가 NaN이고 `bse`·t·p·`conf_int()`·`predict_interval()`도 NaN이다. 반복하지 않는다(IRLS는 μ → 0으로 선형으로만 다가가 identity는 `max_iter`에 닿아 `converged` False와 1e-31을, sqrt는 1e-36을 남겼다). 겹친 열은 G9대로 NaN이다. 척도를 `'X2'`·`'dev'`로 추정하면 척도 0이라 `llf`는 G97의 +inf다. g(y₀)가 무한한 경계(포아송 log의 모두 0, 이항 logit·probit·cloglog의 모두 0·1, 이항 log의 모두 0)는 분리의 규칙(G33) 그대로다. G111을 경계로 넓힌다 (사용자 결정 2026-11-20) |
| G115 lasso·엘라스틱 넷 alpha = 0의 df_model (G3) | `lasso`·`elastic_net`의 alpha = 0에서 `df_model`은 0 아닌 계수의 수다(alpha > 0의 lasso와 같은 뜻; l2 = 0이라 엘라스틱 넷의 식 a − l2·tr H⁻¹도 a다). ridge의 tr H(= 랭크)는 열과 그 사본이 1을 나눠 셌다. 계수는 그대로 `ridge(alpha=0)`과 비트까지 같다. 경로, 교차검증의 겹과 다시 적합도 같다. G110의 "ridge의 것"을 바꾼다 (사용자 결정 2026-11-20) |
| G116 좌표 하강의 정확한 0 (G3) | lasso·엘라스틱 넷·정규화 GLM의 좌표 하강은 soft threshold에 기울기의 반올림 크기만큼 여유를 둔다: \|z_j\| − l1 ≤ 64u·√(d_j·vᵀΩv)(연마의 KKT 검사와 같은 여유 `kKktAbs`)이면 계수는 정확히 0이다. 활성 열의 사본처럼 기울기가 정확히 l1에 놓인 열이 반올림으로 1e-15를 받지 않아 `df_model`도 바르게 센다. l1 = 0(l1_ratio = 0)에는 여유가 없다. 그 크기보다 큰 계수는 그대로다(잘 떨어진 해는 비트까지 같다). 정확히 겹친 열의 lasso는 같은 부호로 나눈 모든 해가 최소점이라, 경로의 따뜻한 시작이 사본에 반올림이 아닌 크기의 계수를 줄 수는 있다(G110의 시작점에 달린 최소점) (사용자 결정 2026-11-20) |
| G16 NaN (G1) | `nan_policy`의 기본은 'propagate'(결과 값이 모두 NaN). 'omit'은 NaN이 있는 행을 빼고, 'raise'는 `ValueError` |
| G17 진단 (G1) | 지렛대(hat), 스튜던트화 잔차, Cook 거리와 Durbin–Watson, Jarque–Bera, Breusch–Pagan. `summary()`에는 Durbin–Watson과 Jarque–Bera를 넣는다(statsmodels와 같다) |
| G18 linregress (G1) | scipy와 같은 꼴: `linregress(x, y)`(인자 순서도 scipy), 결과는 `slope intercept rvalue pvalue stderr intercept_stderr` namedtuple |
| G19 이름 (G1) | 결과 필드는 입력과 상관없이 `sd.ndarray`다. 계수 이름은 `param_names`(절편 `'const'`, frame이면 컬럼 이름, 배열이면 `'x1'…'xk'`)에 두고 `summary()`가 쓴다 |
| G20 가중치 0 (G1) | R의 `lm`과 같다: 적합에 쓰지 않고 `nobs`·잔차 자유도에서 뺀다. 적합값과 잔차는 계산한다. 음수 가중치는 `ValueError`, NaN 가중치는 `nan_policy`를 따른다 |
| G21 겹치는 열의 선택 (G1) | 피벗 없는 QR에서 \|R_jj\| ≤ rtol·‖x_j‖인 열을 겹친다고 보고 빼서 다시 분해한다(R `dqrdc2`와 같은 선택: 앞 열이 남는다). `rtol`의 기본은 sd.linalg와 같은 max(n, k)·eps이고 `rtol=`로 바꾼다(R의 1e-7과 같게 하려면 `rtol=1e-7`) |

### 1.2 범위 (안)
- **선형**: `ols`, `wls`(가중치), `gls`(알려진 오차 공분산), 단순회귀 `linregress`(scipy와 같은 결과, G18). 다항식 적합은 `ols`로 한다(G12). 공분산 추정은 고전형, HC0–HC3, HAC(Newey–West), 군집 강건(G11)
- **일반화 선형**: `glm(y, X, family=…, link=…)`. 분포족은 정규(identity·log), 이항(logit·probit·cloglog·log), 포아송(log·identity·sqrt), 감마(inverse·log·identity)다(G31). 오프셋과 노출(exposure), 사전 가중치(`var_weights`)와 빈도 가중치(`freq_weights`)
- **정규화**: `ridge`(여러 α를 한 번에), `lasso`, `elastic_net`, 정규화 경로, 교차검증으로 α 고르기(`ridge_cv`·`lasso_cv`·`elastic_net_cv`, ridge는 GCV도), 정규화 GLM `penalized_glm`(이항·포아송) (G45–G52, §3.5)
- **강건**: `rlm`(M 추정: Huber, Tukey bisquare, Hampel, 척도는 MAD 또는 Huber의 제안 2; MM 추정), Theil–Sen·Siegel의 `theilslopes`·`siegelslopes` (G53–G57, §3.6)
- **분위**: `quantreg(y, X, q=0.5)`. 내부점법과 심플렉스로 최적성 증명서가 있는 꼭짓점(G60), 유일하지 않으면 사전식 최소점(G63), 여러 분위(G62), R `summary.rq`의 'nid'·'iid'·'ker' 표준오차(G61) (§3.6)
- **비선형 최소제곱**: `nls`(scipy `curve_fit` 꼴, G68). 범위 제약이 없으면 MINPACK `lmder`의 Levenberg–Marquardt, 있으면 신뢰 영역 반사(G69), 해에서 정확한 야코비안의 가우스–뉴턴(느리면 뉴턴) 다듬기(G73). 사용자 함수는 Python 호출이다(§3.7)
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
- 결과는 커널에 준 입력(백색화한 y·X, 가중치, GLS의 백색화한 1의 열, 군집 번호)을 적합할 때 복사해 들고 있다(G86: 넘긴 배열을 나중에 고쳐도 결과가 그대로다). 진단은 처음 읽을 때 이것으로 커널을 다시 부른다. 2의 거듭제곱으로 척도를 바꿔 적합한 경우(§3.1)에는 바꾼 y·X다.
- 설정 필드(`param_names`, `cov_type`, `family`, `link`, `alpha`, `l1_ratio`, `standardize`, `method`, `norm`, `q`)는 읽기 전용 속성이다(G98).
  - `leverage`: 커널의 두 번째 방식(`hat_only`)이 적합과 같은 열 선택·같은 QR을 하고 `orgqr`로 얇은 Q를 만들어 h_t = ‖q_t‖²를 낸다. Q는 반올림 수준에서 직교이므로 1 − h도 κ와 상관없이 절대오차가 u 정도다(R로 R⁻ᵀx_t를 풀면 κu). 적합 자체에는 Q를 만드는 비용(QR 한 번 꼴)을 더하지 않으려고 따로 부른다.
  - `durbin_watson`, `jarque_bera()`: 커널 `resid_stats`가 백색화한 잔차를 적합에 쓰인 행만 순서대로 모아 Σ(ẽ_t − ẽ_{t−1})²와 중심 적률의 합 Σd^p(p = 2, 3, 4, d = ẽ − 평균)을 double-double로 더한다. 덩어리는 n으로만 정해지고 순서대로 더하므로 스레드 수와 상관없는 비트다.
  - `breusch_pagan()`: 보조 회귀(ẽ²를 상수와 백색화한 설계의 열에)를 같은 커널로 푼다(가중치 0인 행은 0/1 가중치로 뺀다). 보조 회귀의 R², ESS, F는 G1a의 필드 그대로다.
  - 스튜던트화 잔차와 Cook 거리는 위의 것과 `scale`로 sd 배열 연산이다. 외적 스튜던트화 잔차는 r√((df − 1)/(df − r²))로 한 번에 구한다(statsmodels는 행을 하나씩 뺀 적합 n번).
- `GLMResult`(G2)도 같은 틀(공통 부분은 `_RegressionResult`: 계수의 추론, 예측의 설계 검사)이다. 커널 `glm_fit`은 회귀마다 `params`, 공분산(고전형은 척도를 곱하기 전의 (XᵀWX)⁻¹, 강건형은 샌드위치), R, μ, η, 쓰인 열 표시, 스칼라 14개(nobs, rank, 편차, 귀무 편차, Pearson χ², 척도, llf, llnull, 반복 수, 수렴, 분리, NaN 표시, 군집 수, HAC 지연 수)와 행마다의 잔차 네 가지(반응, Pearson, 편차, 작업)를 낸다. 진단(지렛대 등)은 GLM에 두지 않는다.
- 정규화의 결과(G48: `RidgeResult`, `LassoResult`, `ElasticNetResult`, 교차검증의 `…CVResult`, `PenalizedGLMResult`)는 `_RegressionResult`를 쓰지 않는다(추론이 없다; `summary()`도 없다, G52). 커널 `penalized_path`가 alpha마다 `params`와 스칼라 13개(nobs, sw, 반복 수, 쌍대 간격, 수렴, 유효 자유도, NaN 표시, 잔차 제곱합(GLM은 편차), TSS(GLM은 귀무 편차), 시험 오차와 시험 가중치 합, 연마 표시, GCV)를 내고, 적합값과 잔차는 처음 읽을 때 `apply`(Dot2)로 구한다. `ssr`도 그 잔차에서 더한다(Gram의 잔차 제곱합은 R²가 1에 가까우면 상쇄된다). 경로의 축은 배치 축 뒤, 필드 자신의 축 앞이다.
- `RLMResult`(G4)도 `_RegressionResult`의 틀이다(계수의 추론, 예측). 커널 `rlm_fit`이 내는 것은 §3.6에 있다. 공분산은 커널이 마지막 값(H1 또는 avar1)으로 내므로 `cov_type`은 `'H1'`·`'avar1'`이고 척도를 다시 곱하지 않는다. Theil–Sen·Siegel의 결과는 namedtuple(`TheilslopesResult`, `SiegelslopesResult`)이다(G56).
- 이름은 statsmodels의 것이다(G10). `centered_tss`(언제나 중심화)와 `uncentered_tss`를 둘 다 둔다(statsmodels와 같다; 절편이 없으면 R²는 비중심 TSS를 쓴다).
- `conf_int`와 `predict_interval`은 (…, k, 2), (…, m, 2) 배열이다(statsmodels의 `conf_int`와 같은 모양).
- `summary()`(G13)는 회귀 하나의 글이다. 배치에서는 `ValueError`다(`nan_policy='omit'`의 배치 에러와 같은 꼴). 배치의 한 원소를 고르는 방법(결과의 인덱싱)은 없다.

---

## 3. 수치 방법

### 3.1 OLS
- **QR**: 쓰이는 행(가중치 > 0)을 √w로 백색화한 설계(절편 열 포함)를 열 우선으로 꾸려 LAPACK `geqrf`(sd.linalg와 같은 Accelerate)로 분해한다. 처음 해는 R β = (Qᵀ b)₁의 삼각 풀이다. 정규방정식은 쓰지 않는다. G21의 판정에 쓰는 열의 노름은 열마다 `dnrm2` 한 번이고, 큰 문제에서는 열들을 스레드풀에 나눈다(열 하나는 늘 한 호출이라 스레드 수와 상관없는 비트).
- **랭크 부족** (G9, G21): 피벗 없는 `geqrf` 뒤 |R_jj| ≤ rtol·‖x_j‖(x_j는 백색화한 열, ‖x_j‖은 분해 전의 2-노름)인 첫 열을 빼고, 그 뒤 열들만 다시 꾸려 앞 j개 반사의 Qᵀ를 적용하고(`ormqr`) 남은 블록을 `geqrf`한다. R `dqrdc2`의 선택과 같다(겹친 열은 반사를 만들지 않으므로, 뒤 열의 판정이 그 열의 잡음 방향에 오염되지 않는다). 겹친 열이 8개를 넘으면 그 뒤는 `dqrdc2`처럼 한 열씩(`larfg`·`larf`) 처리해 비용을 QR 한 번 꼴로 묶는다. 하나라도 뺐으면 남은 열로 처음부터 `geqrf`를 한 번 더 해서, 결과가 그 열을 뺀 적합과 비트까지 같다. 열이 쓰인 행보다 많으면 앞 nobs개만 남는다. `rtol`의 기본은 max(nobs, k)·eps(sd.linalg와 같다)다. NIST 11개는 기본 `rtol`에서 모두 완전 랭크다(Filip 포함).
- **반복 개선** (G15): 증강 시스템 [I X; XᵀW 0][r; β] = [y; 0]에 대한 Björck(1967)의 개선을 같은 QR로 한다. 한 번에:
  1. 잔차의 두 블록 f = y − r − Xβ와 g = −XᵀW r을 double-double로 더한다(Ogita–Rump–Oishi의 Dot2: 곱은 `fma`로 정확히, 합은 two-sum으로). X는 백색화하기 전의 입력을 읽으므로 √w의 반올림은 전처리(QR)에만 들어가고 고정점에는 들어가지 않는다.
     - 행 우선 X에서는 Xβ를 행을 따라 계산한다: 네 행을 NEON 레인 넷에 두고 각 레인이 자기 행의 열 합을 열 순서대로 한다. g의 열 합은 X의 배치와 상관없이 네 열을 레인에 두고 각 레인이 자기 열의 행 합을 행 순서대로 한다. 레인마다 스칼라 식과 같은 연산을 같은 순서로 하므로 결과는 비트까지 같다(행 우선 (10⁶, 50)에서 Xβ 45 → 19 ms, g 58 → 34 ms). GLM·rlm의 Dot2 Xβ와 정규방정식 풀이의 double Xβ도 같은 방식(네 행씩)이다.
  2. h = R⁻ᵀg, d = Qᵀ(√w·f), δβ = R⁻¹(d₁ − h), δr̃ = Q[h; d₂], r += δr̃/√w, β += δβ.
  - 처음 r은 y − Xβ₀를 double-double로 구해 반올림한 값이고, 그 나머지가 첫 f다. 개선 횟수는 아래의 멈춤 규칙(최대 3번)을 따른다.
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
  - κ ≈ 10¹⁶(κu ≈ 1)은 개선이 수렴하지 않는다. 그런 X는 기본 `rtol`에서 대개 열 하나가 겹친다고 판정된다(`refine.log`의 9개 중 8개가 rank 7/8, `k1e16-rho100-0`은 8/8).
- **개선 한 번의 비용** (`bench/regression/g1/refine_cost.py`, 개선 없는 적합 대비): (10³, 10) 0.018 ms / 0.091 ms, (10⁵, 10) 1.8 / 7.4 ms(10스레드 0.33 / 3.4 ms), (10⁴, 200) 3.3 / 17.7 ms(10스레드 2.0 / 14.1 ms). 한 번에 적합 시간의 10–25%다.
- **결정** (G15, 사용자 2026-11-18): 1번 고정이면 κ ≤ 10⁶까지만 1e-14였다(큰 잔차에서 κ 10⁸ 1e-10, 10¹⁰ 3e-7). 그래서 **필요할 때까지 최대 3번** 보정한다.
  - **멈추는 규칙**: 한 번에 오차가 약 κu배 줄므로, 보정 d 뒤에 남은 오차는 약 ‖d‖·κu다. ‖d‖∞·16·κ̂u ≤ 2⁻⁵⁴‖β‖∞이면 멈춘다. κ̂는 LAPACK `trcon`이 R에서 어림한 1-노름 조건수다(작은 배수 안; 16이 그 여유). 판정은 데이터에만 달려 있어 스레드 수와 무관하게 같은 비트다. 첫 보정은 늘 한다.
  - **계약**: "κ(X) ≤ 10¹¹이면 계수 벡터의 상대오차(노름) ≤ 1e-14". κ는 쓰인 열(가중·백색화, 절편 포함)의 2-노름 조건수(`condition_number`)이고, 잔차가 ‖Xβ‖의 100배까지(R² ≥ 10⁻⁴) 잰 값이다. 무작위 문제(κ별 9개, 3가지 잔차)의 최대: κ 10⁹ 5.4e-16, 10¹⁰ 1.4e-15, 10¹¹ 1.1e-15, 그 밖은 잰 값으로 10¹² 1.6e-12, 10¹³ 3.4e-9, 10¹⁴ 2.5e-7. NIST 11개는 모두 1.3e-17 아래다(Filip도; 1번 고정에서는 1.6e-13).
  - **비용**: 조건이 좋은 문제는 1번과 같다((10⁵, 10) 9.3 ms, 개선 없이 7.4 ms). κ = 10⁹인 (10⁵, 10)은 두 번째 보정까지 해서 12.1 ms다.
- **표준오차**: cov = σ̂²(RᵀR)⁻¹(LAPACK `potri`, R에서). β를 개선해도 R은 그대로이므로 상대오차가 대략 κ·u다(표의 마지막 열: κ 10⁶ 3e-11, 10⁸ 1e-9, 10¹⁰ 3e-7, Filip 7e-8). 잔차가 반올림 수준인 거의 완전한 적합(Wampler2: double 입력의 참 SSR이 아주 작다)은 β를 double로 반올림한 것만으로 잔차가 바뀌어 표준오차의 상대 정확도가 없다(값 자체는 1e-16 수준). σ̂²는 double-double 잔차로 더하므로, 잔차가 크면 β의 반올림은 SSR에 2차로만 들어간다.
- **적합값과 잔차**: 마지막 β로 모든 행(가중치 0인 행도)의 Xβ와 y − Xβ를 double-double로 한 번 더 구해 반올림한다. 같은 합으로 SSR = Σw r², 중심 TSS = Σw(y − ȳ)², 비중심 TSS = Σw y², ESS = Σw(ŷ − ȳ)²와 Σw ŷ²를 double-double로 더한다(ȳ는 가중 평균, double-double). `predict`도 같은 Dot2이고 열 순서도 같아서, 적합에 쓴 X로 부르면 `fittedvalues`와 비트까지 같다.
- **p값과 구간**: t의 두꼬리 p = 2·`StudentT(df).ccdf(|t|)`(1 − cdf가 아니므로 아주 작은 p도 상대 정확도), F의 p = `F(dfn, dfd).ccdf(F)`, 구간은 `iccdf`. 모두 sd.stats의 보장 영역에서 상대 1e-14다.
- **R²**: ESS < SSR(R² < 1/2)이면 ESS/TSS(직접 더한 ESS), 그 밖은 1 − SSR/TSS다. 어느 쪽도 상쇄가 2배를 넘지 않는다. `rsquared_adj`의 1 − R²도 같은 쪽에서 구한다.
- **완전 적합**: SSR이 정확히 0이면 IEEE대로 `bse` 0, t ±inf, p 0, `llf` +inf다. statsmodels는 잔차가 반올림 수준이라 bse ~1e-14, t ~1e13–1e15, p ~1e-90–1e-103을 낸다(numpy 경고와 함께; x = 0..9, y = 1 + 2x + 3x²에서 확인). R(4.6.2)의 `summary.lm`도 같은 문제에서 "essentially perfect fit: summary may be unreliable" 경고와 함께 표준오차 9e-15·5e-15·5e-16, p 1e-96–1e-109, F 2.8e32를 내고, 잔차가 반올림 수준이라 `rstandard`·`cooks.distance`·`dwtest`(2.19)·`bptest`도 유한한 값이다. 계약은 정하지 않았다(§8).
  - **극단 척도의 t** (R90, 2026-11-19): `tvalues`는 적합의 단위에서 β′/√diag(C′)로 구한다(데이터 단위의 계수나 `bse`가 넘치거나 아래로 넘쳐도 t, p, `linregress`의 r·p는 맞다; 기울기의 부호도 t에서 읽는다).
  - **판정의 허용오차** (G95, 2026-11-19): 위의 예처럼 QR의 잔차는 정확히 0이 되지 않는다. 그래서 완전 적합은 `rlm`·`quantreg`와 같은 기준으로 정한다: 쓰인 행마다 \|r_t\| ≤ (kk + 16)u(\|y_t\| + ‖x_t‖₁‖β‖∞)(kk는 쓰인 열 수, 커널이 적합의 단위로 잰다)이면 SSR을 0으로 두고 강건 공분산의 점수도 0으로 둔다. 판정은 처음 벗어나는 행에서 멈추므로 보통 첫 행 하나의 비용이다. 잔차 필드(`resid`)는 계산한 값 그대로이고, 진단(스튜던트화 잔차, Cook 거리, DW, JB, BP)은 SSR이 0이면 백색화 잔차를 0으로 보아 0/0의 NaN이다. `df_resid` = 0이면 이 판정과 상관없이 `scale`과 추론 전부가 NaN이다(0/0, 강건 공분산도).
- **극단적인 척도** (2026-11-19): 열이나 y의 크기가 [2⁻²⁵⁶, 2²⁵⁶] 밖이면(예: x·1e160, x·1e-170, y·1e-165) (RᵀR)⁻¹의 원소나 잔차의 제곱이 double을 벗어나 추론이 조용히 틀렸다(t 90 대 108, `bse` inf와 t 0, `ssr` 0과 t inf). 그래서 Python 층이 그런 열과 y를 2의 거듭제곱 f로 나눠 [1/2, 1)로 옮긴 문제를 커널에 준다(회귀마다, 열마다; 정확한 연산이라 같은 범위의 데이터는 비트까지 그대로다). `params`·R·적합값·잔차는 돌려놓고, 제곱합과 공분산은 적합의 단위로 두었다가 필드를 만들 때 한 번에 곱한다: `bse` = √diag(C′)·f_y/f_x, t = β′/√diag(C′), `conf_int()`의 두 끝은 (β′ ∓ q√diag(C′))·f_y/f_x(끝마다 부호를 지킨 채 ±inf로 넘친다; β ± q·`bse`로 구하면 계수와 `bse`가 함께 넘칠 때 inf − inf = NaN이었다), `llf`는 log SSR′ + 2 log f_y, R²·F·진단은 비라서 단위가 없다. 그래서 1e±300 범위에서 추론이 척도 등변이다. 참값 자체가 double 밖이면(`ssr`이 1e-331) 그 필드만 0이나 inf다.
- **빈 입력** (G96): 쓰인 행이 없으면 커널이 NaN 결과에 `nobs` 0을 낸다(`df_model`·`df_resid`도 NaN).
- **예측의 열** (2026-11-19): `predict`는 커널 `apply`에 계수와 함께 열 표시(모형의 열 1, 겹친 열 0)를 준다. 모형의 열은 계수가 0이어도 항에 들어가므로 `X_new`의 NaN이 예측의 NaN이 되고, 겹친 열은 빠진다(예전에는 계수가 정확히 0인 열을 건너뛰어 그 열의 NaN이 사라졌다; 정규화의 0인 계수도 같다). `predict_interval`도 같은 열 표시로 X를 고쳐 분산을 구하므로 두 함수가 같은 행에서 NaN이다.

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
  - 레이블은 Python에서 0..G_tot−1의 번호로 바꾼다(정수·실수 배열은 `unique_inverse`, 문자열은 처음 나온 순서, frame `Series`는 frame의 `factorize`). 없는 레이블은 −1이고 커널이 NaN 데이터로 본다.
  - 배치의 번호는 모든 회귀의 레이블에 걸쳐 매겨지므로(G_tot), 커널은 G_tot이 그 회귀의 행 수보다 크면 그 회귀의 번호만 정렬해 촘촘한 번호로 바꾼 뒤 처음 나온 순서의 지역 번호를 매긴다(2026-11-19; 예전에는 회귀마다 G_tot 크기의 표를 만들어 비용과 메모리가 배치 전체의 레이블 수에 비례했다). 지역 번호는 같으므로 합의 순서와 비트도 같다.
- **빈도 가중치(GLM)의 HC2·HC3** (2026-11-19): 빈도 f_i인 행은 같은 관측 f_i개이고 그 하나하나의 지렛대는 h_i/f_i다(반복한 자료의 값). 그래서 ω_i의 1/(1 − h)·1/(1 − h)²에 h_i/f_i를 쓴다(예전에는 h_i). HC0·HC1·HAC는 그대로다.
- **HAC의 기본 지연 수** (2026-11-19): floor(4(nobs/100)^(2/9))의 nobs는 문서대로 `nobs`다: 쓰인 행의 수, GLM이면 빈도 가중치의 합(예전에는 행 수). 빈도 가중치가 있을 때 HAC의 지연 합은 행 순서의 이웃끼리이고 반복한 자료의 값이 아니다(statsmodels의 관례; 반복한 자료라면 같은 행의 사본들이 지연 1의 이웃이 된다). 커널은 회귀마다 쓰인 행에 나오는 번호를 처음 나온 순서로 다시 매기므로, 결과는 번호를 매긴 방법과 상관없다.
- **가중치 0** (G20): 0인 행은 Q에 행이 없으므로 HAC의 지연은 적합에 쓰인 행을 이어서 세고, 그 행에만 있는 군집은 G에 들지 않는다. 그래서 0인 행이 있는 WLS의 강건 공분산도 그 행을 뺀 WLS와 비트까지 같다. statsmodels는 그 행을 점수 0인 시점·관측으로 남겨서, 지연의 짝과 G가 다르다. R의 `sandwich`(3.1.3)도 그 행을 남기는데, n에 그 행을 세므로 HC0부터 공분산이 (n₀/n)²배(n₀는 쓰인 행 수)이고, HC1의 자유도도 그 행을 세며, HC2·HC3는 그 행이 빠진 `hatvalues`를 길이가 다른 채로 되풀이해 쓴다(경고). 그 행을 뺀 R의 적합과는 HC0–HC3·HAC·군집 모두 1e-15 안에서 같다.
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
  - R(4.6.2 `lm`, sandwich 3.1.3, lmtest 0.9.40; `tests/regression/test_against_r.py`, 고정 데이터 `tests/regression/ref/regression_r.json`, `tools/gen_regression_r.py`)과는 조건이 좋은 문제 7개(절편 없음, 상수 열, 겹친 열, 가중치 0)에서 계수·표준오차·R²·F·잔차 4e-15, `hatvalues`·`rstandard`·`rstudent`·`cooks.distance` 4e-15, `vcovHC`(HC0–HC3)·`NeweyWest(lag, prewhite = FALSE, adjust = FALSE)`·`vcovCL(type = "HC1")`의 표준오차 4e-15, `coeftest`의 p값(군집은 `df = G − 1`) 1e-13, `dwtest` 5e-16, `bptest` 8e-15 안이다. R의 `AIC`는 σ를 모수로 세어 여기보다 2 크다. `bptest`의 기본 보조 회귀는 모형의 설계 그대로라 절편이 없는 모형에서는 상수가 없고(자유도 rank − 1), `varformula = ~ X`로 상수를 넣으면 여기와 같다. 가중 적합의 `bptest`는 원래 잔차로 가중 보조 회귀를 하는 다른 통계량이다. `dwtest`는 가중 적합이면 에러다("weighted regressions are not supported").
  - 조건이 나쁘면 R 쪽이 틀린다(Arb 참값과 비교): κ = 10⁸(n = 100, 절편 없음)에서 R `lm`의 계수가 3e-9, `vcovHC`·`NeweyWest`·`vcovCL`의 표준오차가 0.01–0.18 틀리고(빵과 고기를 XᵀX 꼴로 곱해 κ²u), 여기는 계수 0, 표준오차 7e-10 안이다. 5차 다항식(κ = 2.6×10⁵, 절편 포함)에서는 R의 계수 3e-12, 강건 표준오차 4e-8–1e-7, 여기는 1e-13이다.
- **속도** (`bench/regression/g1b/speed.md`, statsmodels 대비; 최적화 8772322 전 측정): (10⁵, 10)에서 HAC 1.2배(1스레드)·2.3배(10스레드), 군집(1,000개) 1.1·2.0배, 적합과 진단 1.2·2.3배, (10⁶, 50)에서 HAC 1.7·3.1배, 군집 1.8·2.9배, 적합과 진단 1.5·2.4배 빠르다. 진단의 시간은 지렛대를 위한 QR·`orgqr` 한 번과 Breusch–Pagan의 보조 회귀(적합 한 번)이고, 고전형 적합의 시간은 진단이 없을 때와 같다.

### 3.4 GLM
- **IRLS** (`csrc/regression/glm.cpp`): 매 반복은 작업 반응 z = η − offset + (y − μ)/μ′을 가중치 f·m·μ′²/V(μ)로 푸는 가중 최소제곱 한 번이다(f = `freq_weights`, m = `var_weights` × `trials`). QR로 푸는 반복은 G1의 최소제곱 핵심(`csrc/regression/lsq.hpp`의 `fit_one`, `linear.cpp`)을 그대로 부르므로 QR과 겹친 열의 판정(G21, 그 반복의 가중치로)이 `ols`와 같다. 그 풀이는 입력 검사·제곱합·공분산을 건너뛰고 계수와 모든 행의 Xβ(Dot2)만 낸다. μ′이 0이거나 가중치가 유한하지 않은 행은 그 반복의 최소제곱에서 빠진다(R의 `good`). 항등 링크는 z = y − offset을 그대로 쓴다(정규·identity가 `ols`·`wls`와 비트까지 같다).
- **중간 반복은 정규방정식** (G40): 편차의 판정 전 반복은 XᵀWX b = XᵀWz를 Cholesky(`potrf`)와 삼각 풀이 둘로 푼다. XᵀWX와 XᵀWz는 n으로만 정한 4,096행 덩어리마다 √W X의 행을 이어 꾸려 `dsyrk`·`dgemv`로 만들고(덩어리는 스레드풀에), 덩어리의 결과를 순서대로 더한다(스레드 수와 상관없는 비트). η = Xβ + offset은 그냥 double로 더한다(중간 반복의 η 오차는 다음 최소제곱에 2차로만 들어간다).
  - **정준 링크의 연마 반복** (G80): 판정 뒤의 연마 반복도 κ(U)²u ≤ 10⁻¹⁰이면 정규방정식이다. 다만 계수 b가 아니라 걸음 d를 푼다: XᵀWX d = XᵀWρ(ρ = (y − μ)/μ′, 작업 잔차), 새 계수는 b + d. 정규방정식의 상대오차가 걸음에만 붙으므로 새 계수의 오차는 κ(XᵀWX)u‖d‖ + u‖b‖로 QR의 연마와 같은 수준이다. η는 QR의 연마처럼 Dot2로 구한다(마지막 풀이의 작업 반응과 가중치, 곧 공분산이 마지막 반복의 η에서 나온다). 마지막 풀이는 그대로 QR과 반복 개선이라 계수의 정확도가 거기서 정해진다. 문턱을 넘으면(κ(√W X) ≳ 10³) 연마는 QR이다. 정준이 아닌 링크의 연마는 G42의 뉴턴 걸음 그대로다. Arb 참값과의 오차는 위 `refine.py`의 문제(κ = 1–10³)에서 계수 1.3e-16, 표준오차 7.8e-15 안으로 모든 반복이 QR인 것과 같은 자릿수다. (10⁶, 50) 로지스틱 1스레드 1,131 → 812 ms, 10스레드 517 → 307 ms(빠르게 잰 값).
  - **겹친 열**은 G21이 정한다. Cholesky 인자 U는 같은 열 순서의 QR의 R과 부호만 다르므로, G21의 판정 |R_jj| ≤ rtol·‖√W x_j‖를 U에 그대로 적용한다: 어떤 열이 |U_jj| ≤ 2·rtol·‖√W x_j‖이면(2는 인자의 반올림 여유) 그 반복은 QR로 풀고 그 QR이 열을 뺀다. 정규방정식은 마지막 QR이 남긴 열(QR이 없었으면 모든 열)만 쓴다. 연마 반복과 마지막 풀이가 다시 QR로 판정하고, 결과의 겹친 열은 마지막 풀이의 것이다. 기본 `rtol`(max(n, k)·u)에서는 아래의 조건수 판정이 먼저 걸리고, R처럼 `rtol=1e-7`을 주면 이 판정이 걸린다.
  - **QR로 돌아가는 판정**: `potrf`가 실패하거나, LAPACK `trcon`으로 어림한 Cholesky 인자의 조건수 κ(U)로 κ(XᵀWX)·u = κ(U)²·2⁻⁵³ > 10⁻⁴이면 그 반복은 QR이다. 정규방정식의 해는 상대오차가 κ(XᵀWX)·u 정도다. 걸음의 상대오차 e는 이차 수렴 앞에서 반복마다 오차를 e배로만 줄이는데, 편차의 판정(tol 1e-8)에 필요한 계수 정확도가 √tol = 1e-4 정도이므로 e ≤ 1e-4이면 반복 수가 QR과 같다(잰 반복 수도 같다). κ(√W X) ≳ 10⁶이면 QR이다. 행 수 × 열 수가 512 미만이면 처음부터 QR이다(G43; 그 크기에서는 QR이 Gram 행렬을 꾸리는 비용보다 싸다: (50, 3) 로지스틱이 QR 18.7 µs, Cholesky 19.9 µs; (200, 3)은 같고 (500, 5)는 Cholesky가 13% 빠르다). 최종 정확도는 QR과 반복 개선인 연마·마지막 풀이가 정한다(아래 표: Cholesky로 바꿔도 Arb 참값과의 오차가 그대로다).
- **시작값**: R의 `family$initialize`다. 이항 (m y + 1/2)/(m + 1), 포아송 y + 1/10, 정규·감마 y이고 η₀ = g(μ₀), μ는 η₀에서 다시 구한다(`glm.fit`과 같다). statsmodels의 (y + ȳ)/2(이항 (y + 1/2)/2)는 쓰지 않는다: 0만 있는 포아송에서 log 0이 된다. 정규·log 링크에서 y ≤ 0인 행은 가중 평균 ȳ에서 시작하고, ȳ ≤ 0이면 `ValueError`다(R은 y ≤ 0이 하나라도 있으면 에러다).
- **수렴**: |D − D_old|/(|D| + 0.1) < `tol`(R `glm.fit`의 판정과 기본 1e-8)이면 `converged=True`다. 척도와 상관없는 상대 판정이라 R을 따랐다(statsmodels는 편차의 절대 변화 1e-8). 그러나 편차의 변화는 계수 오차의 제곱이라, 이 판정에서 멈추면 계수는 √tol 남짓까지만 맞는다(statsmodels와의 차이가 정준이 아닌 링크에서 1e-6–1e-10인 까닭; 아래 표). 그래서 판정 뒤에도 계수가 반올림보다 크게 움직이는 동안 반복을 이어 간다(연마):
  - 마지막 두 걸음의 비 r = ‖d_t‖∞/‖d_{t−1}‖∞로, 정준 링크는 다음다음 걸음 ‖d_t‖r²(마지막 풀이가 뉴턴 걸음이므로), 정준이 아닌 링크는 다음 걸음 ‖d_t‖r이 2⁻⁵⁰‖β‖∞ 아래면 멈춘다(정준이 아닌 링크의 마지막 풀이도 뉴턴 걸음이지만, 그 걸음의 정확도가 관측 정보에 달려 있어 r²로 멈추면 이항 log의 κ = 10⁶ 문제가 9e-15, r로 멈추면 9e-16이다). ‖d_t‖ 자체가 그 아래이거나 r > 1/2(줄지 않음: 반올림의 바닥)이어도 멈춘다. 판정은 데이터에만 달려 있어 스레드 수와 상관없다.
  - 정준이 아닌 링크에서 Fisher 득점법은 선형으로 수렴한다(잰 비: probit 0.04, 이항 log 0.15, 포아송 identity 0.1). 그래서 연마 반복과 마지막 풀이는 관측 정보의 뉴턴 걸음이고, 얇은 Q로 푼다(G42; 아래 "뉴턴 걸음").
  - **느린 Fisher 득점법** (G44): 판정 전에도, 걸음이 직전 걸음의 1/2 아래로 줄지 않는 반복(반감하지 않은, 계수가 있는 상태의 반복)이 3번 이어지면 남은 반복을 뉴턴 걸음(같은 얇은 Q 풀이, 걸음 반감은 그대로)으로 한다(정준이 아닌 링크만; 정준이면 두 걸음이 같다). 최대 근처의 Fisher 득점법은 걸음이 수렴비 ρ = ‖I − I_E⁻¹I_O‖배씩 줄므로, 비가 1/2 이상이면 반올림까지 50번이 넘는다. 먼 곳에서는 방향이 꺾여 비가 한 번 1/2을 넘는 일이 흔하지만, 수렴하는 적합에서 세 번 이어지지는 않는다(잰 수렴비 0.04–0.15; 정확도 표들과 속도 표의 적합에서 판정이 들지 않아 반복과 결과가 판정이 없을 때와 비트까지 같다). 판정은 계수의 걸음에만 달려 있어 스레드 수와 배치와 상관없다. 판정 전의 뉴턴 걸음은 개선하지 않는다(판정 전 반복과 같다).
  - 정준이 아닌 링크에서 편차의 판정 직후의 걸음은 Fisher의 것이라 그다음 뉴턴 걸음이 더 클 수 있다. 그래서 그 반복에서는 비의 기준을 새로 시작한다(뉴턴 걸음 둘의 비부터 쓴다).
  - QR로 푸는 연마 반복은 반복 개선을 한다(마지막 풀이와 같은 최대 3번, G15의 멈춤 규칙). 그 앞의 반복은 하지 않는다(아래 측정).
  - 경계에서 유한한 최대우도 추정(포아송 identity에서 μ = 0인 행 등)은 걸음 반감으로 경계에 다가가므로 계수가 편차 판정의 정밀도까지만 맞는다.
  - **수렴의 뜻** (G93, 2026-11-19): `converged`는 판정을 통과했고 연마가 위의 멈춤 규칙 가운데 하나로 끝났을 때만 True다. 연마 중에 `max_iter`(마지막 풀이에 하나를 남긴 반복 수)에 닿으면 False다. 예전에는 상한에 잘린 반복 뒤의 마지막 풀이가 편차 판정만 통과해도 True였는데, 그 계수는 판정의 정밀도(√tol)까지만 맞는다. 걸음 반감이 막혀 멈추면(경계) 판정의 결과 그대로다.
- **상수 y와 완전 적합** (G94, G95, G97, G111, G114, 2026-11-19): 절편이 있고 offset이 없으며 y가 쓰인 행에서 정확히 같고 g(y₀)가 유한하며 V(y₀) > 0이면 커널이 정확한 해 (g(y₀), 0, …, 0), μ = y₀, η = g(y₀), 쓰인 행의 잔차 0, 편차·Pearson χ² 0, 강건 공분산 0(군집 하나면 NaN, G108), 귀무 편차 0, `llf`는 μ = y₀에서(척도를 추정하면 G97의 +inf), `llnull` = `llf`를 낸다(반복의 반올림은 기울기 1e-32, 척도 1e-66과 가짜 t를 남겼다). 공분산은 마지막 풀이의 (XᵀWX)⁻¹ 그대로라 척도가 고정인 분포족에서는 기울기의 t가 0, p가 1이다. 일반의 완전 적합은 정규·identity의 SSR이 G95의 허용오차로 0이 된다(마지막 풀이가 G1의 커널이다). 척도를 추정하는 분포족에서 φ = 0이면 `llf`는 +inf, `llnull`은 귀무 편차가 0이면 +inf 아니면 −inf다(G97; 예전에는 0/0의 NaN). `df_resid` = 0이면 추정하는 척도와 그 공분산은 NaN이다(G95; 척도가 고정이면 그대로). y₀가 범위의 경계(V(y₀) = 0: 포아송 identity·sqrt의 0, 이항 log의 1)이면(G114) 커널은 IRLS를 돌지 않고 시작점의 가중치로 마지막 풀이만 해 겹친 열을 정한 뒤 같은 정확한 해를 내고, 공분산 전체를 NaN으로, 쓰인 행의 잔차를 모두 0(분리된 행과 같다)으로 둔다(`n_iter` 0, `converged` True). 쓰이지 않은 행의 y가 y₀와 다르면 그 잔차는 극한(Pearson·편차 ±inf)이다.
- **빈 입력** (G96): 쓰인 행(가중치 > 0)이 없으면 NaN 결과에 `nobs` 0, `converged` False, `n_iter` 0이다(예전에는 `converged` True, `n_iter` 3, `df_model` −1).
- **걸음 반감**: 새 계수의 μ·η가 유효 범위 밖이거나(R의 `validmu`·`valideta`: 이항 0 < μ < 1, 포아송·감마 μ > 0, 포아송 identity·sqrt와 감마 inverse·identity η > 0), 편차가 유한하지 않거나, 편차가 오르면((D − D_old)/(|D| + 0.1) ≥ `tol`, glm2(Marschner 2011)의 규칙; R의 `glm.fit`은 앞의 두 경우만 반감한다) 직전 계수 쪽으로 반씩, 최대 64번 줄인다. 첫 반복처럼 직전 계수가 없으면(시작은 η₀ = g(μ₀)라 어떤 β의 Xβ도 아니다) 같은 가중치로 그 η를 최소제곱한 계수가 유효하면 그쪽으로, 아니면 η 자체를 반씩 시작 쪽으로 줄인다(R은 이때 에러다). 반감한 반복에서도 수렴을 판정한다(glm2와 같다; 분리에서 반감이 끝없이 이어져 `max_iter`까지 가는 것을 막는다).
- **뉴턴 걸음** (G42, 정준이 아닌 링크의 연마 반복과 마지막 풀이): c = f·m·(y − μ)(μ″/V − μ′²V′/V²)라 하면 관측 정보는 Xᵀ(W − C)X다(C = diag(c); 정준 링크는 c = 0). 뉴턴 걸음은 상태 η₀(그 반복의 η − offset, 가중치·c·작업 반응을 구한 값)에 대해 XᵀW r + XᵀC(Xβ − η₀) = 0, r = z − Xβ를 푸는 것이고, G1의 증강 시스템 개선을 이 꼴로 넓혀 같은 QR로 푼다.
  - 개선의 보정에서 정규방정식 블록은 Rᵀ(I − T)R, T = Qᵀ diag(c/w) Q다. T는 얇은 Q(`orgqr`, QR 한 번 꼴의 비용: 정준이 아닌 링크에서만)에서 고정한 4,096행 덩어리의 `dgemm`으로 만들고, I − T의 Cholesky로 y = (I − T)⁻¹(d₁ − R⁻ᵀg), δβ = R⁻¹y, δr̃ = Q[d₁ − y; d₂]다(T = 0이면 G1의 보정과 같다). 잔차 g는 −XᵀW r − XᵀC(Xβ − η₀)를 double-double로 더한다. Gram 행렬 XᵀCX를 만들지 않으므로 κ(X)²가 들어오지 않고, I − T의 조건수(관측 정보와 기대 정보의 비)만 멈춤 규칙의 κ에 곱한다. I − T가 양정치가 아니면(최대 근처가 아니면) 그 풀이는 Fisher 걸음이다.
  - 잔차의 기준을 Xβ₀가 아니라 η₀로 둔다: η₀는 Xβ₀를 반올림한 값이라, Xβ₀를 쓰면 그 반올림이 C를 거쳐 점수에 들어가 κ²로 커진다(probit, κ = 10⁶에서 정확한 산술로도 3e-13; η₀를 쓰면 4e-16).
  - 시작값은 Fisher 해에서 만든 β₀ + R⁻¹(I − T)⁻¹(c₁ − Rβ₀)이고 개선이 고친다(최대 3번, G15의 규칙).
  - 관측 정보의 가중치가 모두 양수이면 그 가중치의 최소제곱이 뉴턴 걸음이지만, 음수인 행이 흔하고(이항 log, 포아송·감마 identity) 한 길로 두는 편이 단순해서 늘 위의 방법을 쓴다.
- **마지막 풀이**: 마지막 반복의 가중치와 작업 반응으로 G1의 전체 풀이(반복 개선 최대 3번, 공분산)를 한 번 더 한다. X는 처음에 검사했고 작업 반응과 가중치는 만들 때 유한하므로 입력 검사는 건너뛴다(rlm의 시작 최소제곱도 같다). 그 계수가 `params`이고(정준이 아닌 링크는 같은 QR의 뉴턴 걸음), (RᵀR)⁻¹ = (XᵀWX)⁻¹에 척도를 곱한 것이 공분산이다(기대 정보; R·statsmodels와 같다). R과 statsmodels는 마지막 반복 앞의 가중치로 공분산을 만들므로 마지막 걸음만큼 다르다(아래 표: 정준 링크에서 표준오차 1e-10–5e-9). 차이는 그쪽 마지막 걸음의 크기에 비례하므로 문제에 따라 더 크다(공부 시간·합격 20행 로지스틱에서 statsmodels 기본 `tol` 1e-8이면 6.9×10⁻⁶, `tol` 1e-14이면 5×10⁻¹¹). 마지막 풀이의 계수가 유효하지 않거나 편차를 올리면 마지막 반복의 계수를 쓴다. `n_iter`는 마지막 풀이까지 센 최소제곱의 수다(R 4.6.2의 기본 판정에서 센 반복 수보다 정준 링크는 1–2, 정준이 아닌 링크는 3–4 크다: 연마와 마지막 풀이; R의 계수가 판정의 정밀도까지만 맞는 대신이다). `max_iter`는 그 수의 상한이다(G32).
- **중간 반복의 풀이와 반복 개선의 위치** (`bench/regression/g2/refine.py`, 결과 `refine.md`): X = U diag(s) Vᵀ(n = 200, k = 6, s는 1에서 1/κ까지 등비, Xβ의 크기 1)인 로지스틱·포아송 회귀의 계수 상대오차(노름, Arb 참값, 씨앗 3개의 최대). glm은 판정 전 반복이 정규방정식, 연마·마지막 풀이가 QR과 개선이다(G80 전의 측정).

  | 분포족 | κ | glm | 판정 전 반복도 QR | 모든 반복을 개선 | 마지막 풀이 개선 없음 | 표준오차 |
  |---|---:|---:|---:|---:|---:|---:|
  | 이항 | 1e0 | 7.8e-17 | 9.1e-17 | 9.1e-17 | 3.6e-16 | 2.4e-16 |
  | 이항 | 1e6 | 2.2e-16 | 2.2e-16 | 5.7e-17 | 1.5e-11 | 4.9e-12 |
  | 이항 | 1e8 | 2.5e-16 | 2.5e-16 | 2.5e-16 | 2.7e-9 | 6.6e-10 |
  | 이항 | 1e10 | 1.1e-15 | 1.1e-15 | 4.5e-16 | 5.0e-7 | 1.3e-7 |
  | 이항 | 1e12 | 4.4e-12 | 4.4e-12 | 7.5e-12 | 1.9e-5 | 1.2e-6 |
  | 포아송 | 1e0 | 5.8e-17 | 6.4e-17 | 6.4e-17 | 4.9e-16 | 3.6e-16 |
  | 포아송 | 1e6 | 7.6e-17 | 7.6e-17 | 7.6e-17 | 3.6e-12 | 2.6e-12 |
  | 포아송 | 1e8 | 5.8e-17 | 5.8e-17 | 5.8e-17 | 7.3e-10 | 4.3e-10 |
  | 포아송 | 1e10 | 2.8e-16 | 2.8e-16 | 1.2e-16 | 8.3e-8 | 3.9e-8 |
  | 포아송 | 1e12 | 9.0e-12 | 9.0e-12 | 4.0e-12 | 9.3e-6 | 2.5e-6 |

  - glm은 κ ≤ 10¹⁰에서 1.1e-15 안이고, 판정 전 반복을 QR로 풀거나 모든 반복을 개선하는 것과 같은 자릿수다(반복 수도 같다). 마지막 풀이는 마지막 반복의 가중치에서 한 걸음을 더 가므로, 그 앞의 반복이 κ²u 꼴로 틀려도 이차로 지워진다. 로지스틱 적합의 시간(1스레드)은 glm이 (10⁵, 10) 42 ms·(10⁶, 50) 1,364 ms, 판정 전 반복도 QR이면 54·2,377 ms, 모든 반복을 개선하면 71·2,947 ms다(최적화 전 측정; G80 전이라 연마는 QR이다).
  - 연마 반복을 개선하지 않고 마지막 풀이만 개선하면 κ = 10¹⁰에서 4.6e-14, 10¹²에서 1.5e-9이고, 연마의 개선이 더하는 시간은 6–8%다. 마지막 풀이를 개선하지 않으면 κu 꼴로 나빠진다(10⁸에서 3e-9).
- **정준이 아닌 링크의 정확도** (`bench/regression/g2/noncanonical.py`, 결과 `noncanonical.md`): X = U diag(s) Vᵀ(n = 200, 열 5개와 절편, s는 1에서 1/κ까지 등비), 링크마다 μ가 범위 안에 있게 고른 η의 문제, 씨앗 2개의 최대. 계수의 상대오차(노름, Arb 참값). 괄호는 같은 적합을 연마와 마지막 풀이도 Fisher 걸음으로 한 것이다.

  | 링크 | κ = 10⁴ | 10⁶ | 10⁸ | 10¹⁰ |
  |---|---:|---:|---:|---:|
  | 이항 probit | 1.9e-16 (1.9e-15) | 3.8e-16 (6.0e-14) | 3.1e-16 (5.8e-11) | 3.8e-15 (3.6e-9) |
  | 이항 cloglog | 3.0e-16 (2.8e-14) | 5.5e-16 (9.2e-13) | 1.2e-16 (2.6e-11) | 2.1e-15 (2.3e-9) |
  | 이항 log | 1.2e-15 (4.5e-6) | 9.0e-16 (1.2e-5) | 3.5e-15 (2.1e-5) | 2.1e-14 (2.4e-5) |
  | 포아송 identity | 1.3e-16 (8.3e-9) | 1.5e-16 (8.1e-9) | 4.2e-17 (7.9e-9) | 4.3e-14 (1.5e-8) |
  | 포아송 sqrt | 1.3e-16 (4.4e-16) | 2.0e-16 (2.0e-14) | 1.4e-16 (5.5e-12) | 4.7e-15 (7.5e-10) |
  | 감마 log | 4.1e-17 (3.3e-13) | 9.1e-17 (2.6e-13) | 3.4e-17 (2.5e-10) | 4.3e-15 (8.1e-9) |
  | 감마 identity | 8.5e-17 (1.0e-8) | 1.1e-16 (1.4e-7) | 4.3e-18 (1.4e-7) | 3.2e-14 (1.1e-7) |
  | 정규 log | 1.4e-16 (1.6e-15) | 1.7e-16 (2.4e-13) | 3.6e-17 (8.3e-12) | 4.2e-16 (9.9e-10) |
  | (정준) 이항 logit | 2.4e-16 | 2.4e-16 | 8.8e-16 | 1.3e-15 |
  | (정준) 포아송 log | 1.8e-16 | 1.0e-16 | 6.5e-17 | 8.6e-17 |

  - κ ≤ 10⁸에서 3.5e-15 안으로 정준 링크와 같은 수준이고, 10¹⁰에서 4e-16–4e-14다(정준 1e-15 안). 10¹⁰의 남은 차이는 반올림한 η₀에서 구한 점수의 감도(관측 정보의 가중치 W − C로 사영한 η의 반올림)다. Fisher 걸음의 연마는 κ와 함께 나빠지고, 수렴이 느린 링크(이항 log, 감마·포아송 identity)는 κ와 상관없이 편차 판정의 정밀도(1e-8–1e-5)에 머문다.
  - 비용(`noncanonical.md`, 같은 적합을 처음부터 Fisher 걸음으로 한 것 대비): probit (10⁵, 10) 1스레드 +18%·10스레드 +30%, (10⁶, 50) +34%·+57%; 감마 log +7%·+11%, +13%·+32%. 연마 반복과 마지막 풀이마다 `orgqr` 한 번과 T의 `dgemm`(덩어리마다 스레드풀, 순서대로 합)이다(반복은 1번 줄어든다). probit은 분리 증명서의 Fisher 걸음 한 번이 더 들어 있다. 정준 링크는 그대로다(±2%).
  - 느린 Fisher 득점법(G44): `refine.py` 꼴의 감마 log 문제(평평한 방향에서 수렴비 약 0.9)는 Fisher 걸음만으로는 편차가 계수보다 먼저 평평해져 판정을 계수가 1e-5–1e-3 틀린 채로 통과하고(21–49번), 연마만 뉴턴이면 맞는 값에 닿지만 31번이 든다(κ = 10⁶). 판정이 들면 κ = 10⁴–10¹⁰에서 9–15번에 3.8e-14 안이다(`noncanonical.md`). statsmodels는 이 문제에서 500번에 수렴하지 않고, R의 `glm.fit`도 Fisher 득점법이다: R 4.6.2는 κ = 10⁴, 10⁶의 이 문제에서 기본 판정(epsilon 1e-8)으로 27번에 멈춰 계수가 1e-4–2e-3 다르고, epsilon 1e-15로도 83–94번에 5e-9–9e-9 다르다.
  - 표준오차는 G1처럼 R에서 계산하므로 상대오차가 κu 정도다(10⁸에서 7e-10).
- **편차와 로그우도** (상쇄 없이): 단위 편차 d(y, μ)는
  - 정규 (y − μ)², 포아송 2μ·φ(r)(y = 0이면 2μ), 감마 2·ψ(r), 이항 2[μφ(r₁) + (1 − μ)φ(r₂)](y = 0이면 −2 log(1 − μ), y = 1이면 −2 log μ)이고, r = (y − μ)/μ, r₁ = (y − μ)/μ, r₂ = (μ − y)/(1 − μ), ψ(r) = r − log1p(r), φ(r) = (1 + r)log1p(r) − r이다.
  - ψ는 [−1/2, 1]에서 s = r/(2 + r)의 atanh 급수(ψ = rs − 2s³(1/3 + s²/5 + …), |s| ≤ 1/3라 17항)로, 밖에서는 r − log1p(r)(상쇄 3.6배 이하)로 계산한다. φ는 [−1/2, 1]에서 r² − (1 + r)ψ(r)(상쇄 4배 이하), 밖에서는 x log x − r(x = y/μ)다. y log(y/μ) − (y − μ)를 그대로 계산하면 μ ≈ y에서 두 항이 상쇄된다(statsmodels의 `resid_deviance`가 잔차 1e-3에서 2e-11 상대오차를 내는 까닭).
  - 이항은 μ와 1 − μ를 따로 정확히 구한다: logit은 sd.special `expit(η)`와 `expit(−η)`, probit은 `ndtr(η)`와 `ndtr(−η)`(φ(η)는 η²를 h + l로 정확히 나눠 e^{−h/2}(1 − l/2)), cloglog는 1 − μ = exp(−e^η), μ = −expm1(−e^η), log는 1 − μ = −expm1(η)다. log μ는 μ < 1/2이면 log μ, 아니면 log1p(−(1 − μ))다(1 − μ의 로그도 같다). 시작값의 g(μ₀)는 `logit`·`ndtri`다.
  - 로그우도는 statsmodels의 `loglike`(상수항 포함)다: 포아송 v/φ·(y log μ − μ − lgamma(y + 1)), 감마 w log(w y/μ) − w y/μ − lgamma(w) − log y(w = v/φ), 이항 v[log C(n, ny) + ny log μ + n(1 − y)log(1 − μ)], 정규 −½[v(y − μ)²/φ + log(φ/v) + log 2π]. μ와 상관없는 lgamma 항은 한 번만 계산하고 값이 정확히 0인 경우(y = 0·1, ny = 0·n)는 lgamma를 부르지 않는다. 정규·identity는 φ = 편차/nobs의 집중 로그우도(statsmodels와 같고 `ols`의 `llf`와 같다)다.
  - 모든 합은 n으로만 정한 4,096행 덩어리를 double-double로 더해 순서대로 합친다(스레드 수와 상관없는 비트). 귀무 모형의 합(닫힌 꼴의 Σf m y, Σf m(1 − y), Σf m와 귀무 편차), `nobs`(Σf), 정규·log의 시작값(가중 평균)도 같다. 정규화(§3.5)·rlm·분위 회귀의 행 합(가중치 합, 귀무 모형의 평균, naive 갱신의 vᵀΩv, TSQR 아래 블록의 잔차 제곱합, 제안 2의 척도 기울기, 'ker'의 평균과 분산)도 같은 꼴이다. 비선형 최소제곱(§3.7)의 노름과 내적은 한 스레드에서 행 순서로 한 번에 더한다(MINPACK·scipy의 걸음을 옮긴 부분이라 그대로 두었다; 스레드 수와는 상관없다).
- **귀무 모형**: 절편이 있고 오프셋이 없으면 μ = Σ f m y/Σ f m(닫힌 꼴, 링크와 상관없다), 오프셋이 있으면 절편만의 IRLS(같은 커널), 절편이 없으면 η = offset(R의 `glm`; statsmodels는 이때도 상수를 적합한다). `llnull`은 같은 척도의 로그우도이고, 정규·identity는 귀무 모형의 집중 로그우도라 `llr` = n log(SSR₀/SSR)이다(statsmodels의 `llnull`은 여기서 전체 모형의 척도를 쓰므로 `llf`와 정의가 어긋난다).
- **강건 공분산** (G37): 점수 s_i = x_i·m_i(y_i − μ_i)μ′_i/V(μ_i)는 마지막 풀이의 백색화한 잔차 √w(z − Xβ)와 얇은 Q의 행으로 G1의 커널이 만드는 점수 그대로다(척도는 빵과 고기에서 약분된다). 빵은 관측 정보의 역행렬이다(statsmodels가 `model.hessian(observed=True)`로 그렇게 한다; R의 `sandwich`는 기대 정보). 관측 정보 = Rᵀ(I − T)R, T = Qᵀ diag(c/w) Q라서, 빵은 R⁻¹(I − T)⁻¹R⁻ᵀ이고 I − T의 Cholesky로 고기를 양쪽에서 푼다(XᵀX를 만들지 않는다; I − T가 양정치가 아니면, 곧 최대가 아니면 NaN). 정준 링크는 c = 0이라 G1과 같은 계산이다. HC1의 n/(n − k)와 군집의 (n − 1)/(n − k)의 n은 `nobs`(Σf)다. `freq_weights`가 있으면 HC·HAC의 점수를 √f로 나눈다(복제한 행이 각자 점수를 내는 것과 같다; statsmodels와 같다). 군집 합은 나누지 않는다. GLM에는 Wald F가 없다.
- **분리** (G33, 이항의 네 링크와 log 링크의 포아송): 최대우도 추정이 없으면 계수가 무한대로 가는 방향 b가 있다(경계 행 y = 1에서 xᵢᵀb ≥ 0, y = 0에서 ≤ 0, 그 밖의 행에서 = 0, 하나라도 엄격히). 판정은 두 증명서로 한다.
  1. **존재 증명서**: 마지막 풀이의 걸음 d(β_t에서의 Fisher 걸음, XᵀWX d = 점수; 마지막 풀이가 뉴턴 걸음이면 같은 가중치로 Fisher 걸음을 한 번 더 개선해 푼다: 분리가 가능한 이항 probit·cloglog·log에서 QR 한 번이 더 든다)가 모든 경계 행에서 |xᵢᵀd| < |yᵢ − μᵢ|/(2μ′ᵢ)이면 최대우도 추정이 있다. 점수 방정식에서 Σ xᵢ m_i μ′ᵢ/Vᵢ[(yᵢ − μᵢ) − μ′ᵢ xᵢᵀd] = 0이고 이 조건이면 경계 행의 계수가 모두 그 행의 부호를 그대로 가지므로, 경계 행의 엄격히 양인 결합이 0이 된다(Stiemke의 대안 정리; 그러면 위의 b는 없다). 1/2는 d를 반올림으로 구한 여유다. 로지스틱이면 조건은 |xᵢᵀd| < 1/(2(1 − μᵢ)) 꼴이라 μ가 반올림 수준으로 0·1에 가까워도(분리 없는 강한 효과) 통과한다.
  2. **방향 증명서**: 1을 통과하지 못한 경계 행 S에 대해, d를 나머지 적합 행의 X의 영공간에 사영한 b(그 행들의 Xd를 X에 최소제곱한 계수를 빼서; 같은 커널)가 S의 모든 행에서 sᵢxᵢᵀb > √u·Σ|x_ij d_j|(사영 전의 d로 잰 크기)이면 S는 무한대로 가는 행이다. 기준을 사영한 b 자신의 크기로 두면 안 된다: 나머지 행이 영공간을 남기지 않으면 b는 반올림 잡음이고, 잡음끼리 견주면 우연히 통과한다(예전에는 `max_iter`가 작아 마지막 걸음이 클 때 그래서 분리가 아닌 문제를 분리로 냈다: 공부 시간·합격 20행 자료, `max_iter=2`에서 `separation=True`, 편차 15.57로 최대우도의 16.06보다 작았다). 그렇지 않은 행은 S에서 빼고 다시 사영한다. 남는 S가 없으면 분리가 아니다(느린 수렴, `max_iter`에 닿은 반복 등; 결과는 그대로 낸다). 분리된 행을 뺀 나머지의 다시 적합은 본 적합과 `max_iter`를 나눠 쓴다(`n_iter` ≤ `max_iter`, G32).
  - S가 증명되면 S를 경계(μ = y)에 두고 나머지 행만 다시 IRLS로 적합하고(가능도 최대화의 극한: 나머지 행에서 계수가 정해지지 않는 방향은 겹친 열이 된다), 그 적합에 1–2를 되풀이한다(최대 8번). 결과의 μ·η·편차·Pearson χ²·`llf`는 그 극한이고 계수와 추론은 NaN이다(G33). `n_iter`는 다시 적합한 반복도 센다.
  - 반복 횟수의 임계값은 쓰지 않는다. 판정을 가르는 것은 정리의 조건뿐이고, 남은 상수 둘(1/2, √u)은 반올림의 여유다. LP(Konis 2007)를 풀지 않고 IRLS가 이미 만든 걸음을 증명서로 쓴다. 이항 log 링크는 η < 0의 제약 때문에 y = 1인 행이 등식 행이고, 포아송 identity·sqrt나 감마처럼 경계에서도 유한한 최대우도 추정이 있는 경우(예: μ = 0인 0 개수)는 분리가 아니다.
- **추론** (G38, G39): 척도가 고정이면 정규분포, 추정하면 t(`df_resid`)다. 강건 공분산은 HC0–HC3·HAC에서 척도가 고정인 이항·포아송이면 정규분포, 그 밖은 t(`df_resid`)이고, 군집은 t(G − 1)다. p값은 `Normal().ccdf`·`StudentT(df).ccdf`의 두 꼬리, 구간은 `iccdf`다.
- **정확도** (Arb 참값 `tests/regression/ref/glm.json`, `tools/regref.py`의 `glm`): 참값 도구는 double 입력에서 점수 방정식을 Arb의 뉴턴(관측 헤시안)으로 풀고, 그 점 둘레의 상자에 Krawczyk 연산자를 적용해 상자 안에 근이 하나뿐임을 증명한 뒤(구간 뉴턴), 그 구간으로 나머지 값(표준오차, 편차, Pearson χ², 척도, 로그우도, 귀무 모형, 강건 표준오차, p값)을 2⁻¹¹⁰ 상대까지 좁힌다. 문제는 분포족·링크 12개와 이항 비율(시행 수), 포아송 오프셋·노출, 감마의 두 가중치(0 하나), 절편 없음, `scale='dev'`(n = 60). `bench/regression/g2/accuracy.py`의 결과:

  | 문제 | `n_iter` | 계수 (노름) | 표준오차 | p값 | 편차·Pearson χ²·척도 | llf·llnull·귀무 편차 | 강건 표준오차 (HC0–HC3, HAC, 군집) |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | gaussian_identity | 3 | 4.9e-17 | 1.5e-16 | 8.2e-15 | 6.6e-17 | 2.4e-16 | 5.1e-16 |
  | gaussian_log | 8 | 5.4e-17 | 1.4e-16 | 2.5e-15 | 1.0e-16 | 8.9e-17 | 3.0e-16 |
  | binomial_logit | 6 | 7.9e-17 | 7.9e-17 | 1.7e-15 | 1.8e-17 | 2.0e-17 | 5.4e-16 |
  | binomial_probit | 7 | 1.7e-17 | 2.7e-16 | 2.7e-15 | 1.6e-16 | 1.4e-17 | 8.5e-16 |
  | binomial_cloglog | 10 | 2.2e-17 | 1.6e-16 | 3.5e-16 | 7.9e-17 | 7.9e-17 | 2.6e-16 |
  | binomial_log | 14 | 1.1e-16 | 4.5e-16 | 1.4e-14 | 4.7e-17 | 3.0e-17 | 9.5e-16 |
  | poisson_log | 7 | 8.4e-17 | 1.4e-16 | 3.7e-15 | 4.1e-17 | 7.4e-17 | 3.9e-16 |
  | poisson_identity | 11 | 9.9e-17 | 2.7e-16 | 3.0e-14 | 1.5e-16 | 1.5e-16 | 7.4e-16 |
  | poisson_sqrt | 8 | 7.4e-17 | 1.9e-16 | 1.6e-13 | 1.6e-16 | 7.9e-17 | 6.9e-16 |
  | gamma_inverse | 7 | 8.1e-17 | 1.5e-16 | 9.6e-15 | 1.2e-16 | 9.1e-17 | 4.5e-16 |
  | gamma_log | 9 | 3.9e-17 | 1.4e-16 | 4.8e-15 | 7.7e-17 | 7.5e-17 | 6.5e-16 |
  | gamma_identity | 17 | 4.3e-17 | 1.4e-16 | 2.8e-15 | 1.5e-16 | 8.5e-17 | 3.1e-16 |
  | binomial_trials | 6 | 4.5e-17 | 1.3e-16 | 6.2e-15 | 2.2e-16 | 1.6e-16 | 8.3e-16 |
  | poisson_exposure | 7 | 6.1e-17 | 1.8e-16 | 1.9e-15 | 1.7e-16 | 3.4e-17 | 4.5e-16 |
  | gamma_weights | 8 | 3.8e-17 | 1.9e-16 | 1.0e-14 | 1.4e-16 | 1.5e-16 | 7.5e-16 |
  | logit_noint | 7 | 8.8e-17 | 1.8e-16 | 1.1e-15 | 8.6e-17 | 4.4e-17 | 6.0e-16 |
  | gamma_dev | 8 | 1.0e-16 | 1.9e-16 | 4.3e-15 | 2.2e-16 | 9.9e-17 | 6.2e-16 |
  | 최대 | | 1.1e-16 | 4.5e-16 | 1.6e-13 | 2.2e-16 | 2.4e-16 | 9.5e-16 |

  - 이 17개(n = 60)는 행 수 × 열 수가 512 미만이라 판정 전 반복도 QR이다. 정규방정식의 길은 n = 200, k = 6의 Arb 문제(`test_inner_normal_equations_against_arb`; κ ≤ 10⁸에서 정준 링크 1e-15, probit·감마 log 5e-15 안)와 위의 반복 개선 표, QR 길과의 비교(`test_inner_normal_equations`)로 확인한다.
  - statsmodels(0.15.0)와는 정준 링크에서 계수 1e-11, 정준이 아닌 링크에서 1e-6 안이다(`tests/regression/test_glm.py`). 정준이 아닌 링크의 차이는 statsmodels가 편차의 절대 변화로 멈춰 계수가 덜 수렴한 것이다(Arb와 비교한 위 표가 근거). 표준오차는 위의 공분산 가중치 차이로 1e-10–5e-9 다르다.
  - R(4.6.2 `glm`, sandwich 3.1.3, lmtest 0.9.40; `tests/regression/test_against_r.py`): 분포족·링크 12개와 시행 수, 오프셋(절편 없음 포함), 사전 가중치(0 포함), 절편 없음, quasipoisson·quasibinomial(여기의 `scale='X2'`) 21문제(n = 60). R의 `glm.fit`을 여기의 해에서 한 번 돌리면(`start = params`, `maxit = 1`) 계수가 그 자리에 머물고(1.2e-15), 표준오차·편차·Pearson χ²·척도·적합값 1.6e-15, `summary.glm`의 p값 1.8e-13, 귀무 편차 1e-15 안이다. 정준 링크의 `vcovHC`(HC0–HC3)·`NeweyWest`·`vcovCL`은 1.6e-15, `coeftest`의 p값(군집은 `df = G − 1`) 1.2e-12 안이다. R의 기본 판정(epsilon 1e-8)으로 끝까지 적합하면 편차 판정에서 멈추므로 계수가 정준 링크 1e-7 안, 정준이 아닌 링크 4e-7–2e-4 다르고, 표준오차는 마지막 반복 앞의 가중치 때문에 1e-4까지 다르다(epsilon 1e-13으로도 정준이 아닌 링크의 계수가 1e-10–1e-6 다르다).
- **statsmodels·R과 다른 점** (확인한 것):
  - statsmodels GLM의 `cov_type='HC1'`·`'HC2'`·`'HC3'`는 모두 HC0을 낸다(보정을 하지 않는다). 여기는 G1의 OLS와 같은 뜻이다(HC2·HC3는 √W X의 지렛대; R `sandwich`와 같은 정의).
  - statsmodels의 `get_prediction`은 감소하는 역링크(감마 inverse)에서 구간의 아래·위가 뒤바뀐다. 여기는 순서대로 낸다.
  - statsmodels의 `nobs`는 행 수다(`freq_weights`의 합은 `wnobs`; `df_resid`는 같다). `var_weights`가 0인 행도 센다.
  - statsmodels의 `resid_response`는 이항 시행 수가 있으면 성공 수의 척도다. 여기는 R처럼 `y`의 척도다.
  - R의 AIC는 정규·감마에서 척도를 모수로 하나 더 세고, R의 `logLik`은 그 척도를 편차/Σw로 둔다(여기서 `scale=deviance/nobs`로 적합하면 같은 값이다). 사전 가중치가 있으면 R의 감마 `logLik`은 Σwᵢ log f(yᵢ; μᵢ, φ)로 가중치를 밀도의 로그에 곱하고(여기와 statsmodels는 모양 wᵢ/φ), 가중치 0인 행이 있는 정규 분포족의 R AIC는 log 0 때문에 Inf다(`lm`의 `logLik`은 그 행을 빼서 유한하다). 여기는 statsmodels를 따랐다. 이항(시행 수의 log C(n, ny) 포함)·포아송(사전 가중치 포함)의 `llf`·`aic`는 R과 1e-14 안에서 같다.
  - R `sandwich`의 빵은 기대 정보라 정준이 아닌 링크의 강건 표준오차가 여기(관측 정보)와 다르다(위 21문제에서 3–32%; R의 값은 기대 정보의 샌드위치와 1e-12 안에서 같다). 사전 가중치가 0인 행이 있으면 §3.3의 OLS와 같은 n의 차이가 있다.
  - `lmtest::coeftest`는 glm이면 분포족과 상관없이 정규분포다(`df = Inf`가 기본). 여기는 G39대로 정규·감마와 척도를 추정하는 이항·포아송이 t(`df_resid`)다(z로 다시 계산하면 R의 p값과 1.2e-12 안에서 같다).
  - 분리: R은 경고("fitted probabilities numerically 0 or 1 occurred", 완전 분리면 "algorithm did not converge"도)와 함께 큰 계수와 표준오차를 낸다(x = 1..10, y = [x > 5.5]의 로지스틱에서 계수 (−246, 45), 표준오차 (3.4e5, 6.1e4), 25번, `converged=FALSE`, 편차 7.9e-10; x = 5에 0과 1이 함께 있는 준완전 분리는 20번에 수렴으로 끝난다). 0만 있는 군의 포아송 log는 경고 없이 수렴으로 끝난다(계수 21.3, 표준오차 6.3e3). 여기는 셋 모두 `separation=True`이고 계수가 NaN이며, 편차는 R이 다가가는 극한(0, 5.5451774, 3.9121534)이다(G33).
  - R은 시작값이 유효하지 않으면 에러다: 정규·log에서 y ≤ 0이 있으면 "cannot find valid starting values: please specify some", 첫 걸음이 범위를 벗어나면(포아송 identity에서 μ가 0에 가까운 자료) "no valid set of coefficients has been found: please supply starting values". 여기는 둘 다 적합하고, 그 해를 R의 시작값으로 주면 R도 같은 해에 머문다(1e-15). 경계에 해가 있는 이항 log는 R이 "algorithm stopped at boundary value"로 멈추고(편차가 1e-9 상대만큼 크다), 여기는 경계에 더 가깝다.
- **속도** (`bench/regression/g2/speed.md`; 최적화 8772322 전 측정): (10⁵, 10)·(10⁶, 50)의 로지스틱·포아송에서 statsmodels보다 1스레드 1.2–3.1배, 10스레드 3.9–6.8배 빠르다(반복은 statsmodels보다 1–2번 많다: 연마와 마지막 풀이). 판정 전 반복의 정규방정식(G40)이 QR보다 시간을 1스레드 21–46%, 10스레드 45–64% 줄였다(최적화 전 측정; 그 뒤 G80과 행 패스의 NEON으로 더 줄었다). scikit-learn은 lbfgs가 1.9–23배, newton-cholesky가 1.4–7.7배 빠르지만 표준오차가 없고 계수가 lbfgs 1e-4–1e-2, newton-cholesky 1e-7–2e-6 다르다. 작은 로지스틱 10,000개((50, 3))의 배치는 statsmodels 반복문보다 1스레드 18배, 10스레드 141배 빠르다.

### 3.5 정규화
- **목적함수와 좌표** (G45): (1/(2·sw))·Σwᵢ(yᵢ − b₀ − xᵢβ)² + alpha·(ρ‖Sβ‖₁ + (1 − ρ)/2·‖Sβ‖²), sw = Σw, ρ = `l1_ratio`, S는 `standardize`면 열의 모표준편차, 아니면 I다. 커널(`csrc/regression/penalized.cpp`)은 중심화·척도화한 좌표 zᵢⱼ = (xᵢⱼ − cⱼ)/sⱼ, bⱼ = sⱼβⱼ, ω = w/sw에서 ½Σω(v − zb)² + l1‖b‖₁ + l2/2‖b‖²(v = y − c_y, l1 = alpha·ρ, l2 = alpha·(1 − ρ))를 푼다. 절편이 없으면 c = 0, c_y = 0이다.
- **입력 준비**: 가중치가 0보다 큰 행이 적합의 행이다(G20). 한 번 훑어 가중 평균(고정한 덩어리의 double 합)과 상수 열(모든 값이 첫 행과 같은 열; 절편이 없으면 0인 열)을 찾고, 상수 열은 적합에서 뺀다(계수 0). 이어 √w[(x − c), y − c_y, 1]의 Gram을 `dsyrk`로 만든다(행 덩어리는 열 수로만 정하고, 고정한 수의 슬롯이 덩어리를 차례로 누적한 뒤 슬롯을 순서대로 더한다). 마지막 열 S = Σw(x − c)로 G − SSᵀ/sw를 만들면 정확히 중심화한 Gram이고, 평균의 반올림은 남지 않는다(평균도 c + S/sw로 고친다). 척도와 Gram·Zᵀv·vᵀv가 이것에서 나온다. 열이 1,024개를 넘으면(아래 naive 갱신) 대각과 합만 한 번에 더한다.
- **ridge**: 중심화·백색화·척도화한 설계 Ω^½Z(m × k)를 QR로 분해한다. 4,096행 이상(또는 4k행 이상)의 블록이 4개 이상이면 TSQR이다: 블록마다 QR(스레드풀), 블록의 R을 쌓아 다시 QR한다(블록은 크기로만 정해 스레드 수와 상관없는 비트). R의 SVD(`gesdd`) R = U diag(s) Vᵀ와 d = Qᵀv로 모든 alpha에서 b = V diag(s/(s² + alpha)) Uᵀd₁, 잔차 제곱합 ‖d₂‖² + Σ(alpha/(s² + alpha))²(Uᵀd₁)², 유효 자유도 Σs²/(s² + alpha), GCV가 닫힌 꼴이다. XᵀX를 만들지 않는다. max(m, k)·2u·s₁ 아래의 특이값은 0으로 보아(alpha = 0에서 최소 노름 해) 그 방향의 계수는 0이다. 행이 열보다 적으면 Ω^½Z의 SVD를 직접 한다.
  - 해의 조건수 (s₁² + alpha)/(s²_min + alpha)가 64k를 넘으면 최대 3번 개선한다(G51). 잔차 rᵢ = (yᵢ − c_y) − γ′ − Σ zᵢⱼbⱼ는 G1의 Dot2처럼 double-double로 구한다: yᵢ − c_y와 xᵢⱼ − cⱼ는 two-sum으로 정확히, zᵢⱼ = (xᵢⱼ − cⱼ)/sⱼ와 모든 곱·합은 double-double이고, 기울기 Σw zⱼ r과 Σw r도 double-double로 더해 한 번 반올림한다(덩어리는 크기로만 정한다). 보정 (ZᵀΩZ + alpha·I)⁻¹g는 같은 SVD로 푼다. 그래서 고정점이 double로 반올림한 설계가 아니라 정확한 문제의 해이고, 한 번에 오차가 그 조건수 × u배 준다. 조건수가 그 아래면 개선하지 않으므로 조건이 좋은 문제의 비용은 그대로다.
- **lasso·엘라스틱 넷의 좌표 하강**: 열 순서로 도는 순환 좌표 하강이다(난수 순서는 없다). 열이 1,024개 이하이면 공분산 갱신(위의 Gram; 좌표 하나에 O(k), 기울기 c − Gb를 갱신), 그보다 많으면 naive 갱신(Ω^½Z를 열 우선으로 꾸리고 잔차를 갱신)이다. 열 우선으로 꾸릴 때(ridge의 QR도) 4,096행 덩어리를 64행 타일로 열마다 이어 쓴다: 한 행씩 쓰면 같은 행의 원소가 덩어리 길이(2의 거듭제곱, 32 KB)만큼 떨어져 몇 개의 캐시 집합에 몰려 서로 밀어낸다(원소의 계산은 같아 비트가 같다; ridge (10⁵, 100) 1스레드 124 → 90 ms). glmnet의 활성 집합 순환과 같다: 전체 스윕 한 번 뒤 0 아닌 계수만 스윕하고, 그 변화가 작아지면 다시 전체 스윕이다.
  - **수렴** (scikit-learn의 판정): 가장 큰 계수 변화가 `tol` × 가장 큰 계수 이하인 전체 스윕 뒤(와 시작할 때) 쌍대 간격을 재고, 간격 ≤ `tol` × Σωv²이면 수렴이다. 이것은 scikit-learn의 tol·‖y‖²(그 목적함수는 이것의 sw배)와 같은 판정이다. 간격은 scikit-learn의 꼴(L1 항이 있으면 쌍대점을 척도 s = min(1, l1/‖Zᵀ Ω r − l2 b‖∞)로 맞춘 꼴 A, ridge는 꼴 B, alpha = 0은 기울기의 제곱 노름)이지만 vᵀΩv 크기의 항이 상쇄되지 않게 고쳐 쓴다: 꼴 A는 ½R²(1 − s)² + ½l2‖b‖²(1 + s²) + l1‖b‖₁ − s·D(R² = Σωr², D = Σωr(v − r)는 bᵀZᵀΩr과 절편 항), 꼴 B는 ‖ZᵀΩr − l2b‖²/(2l2) + bᵀZᵀΩr − D다. `n_iter`는 스윕 수(전체와 활성 집합 스윕을 모두 센다)이고 `max_iter`가 alpha마다의 상한이다(G32).
  - scikit-learn의 기본 `tol=1e-4`에서 계수는 대략 √(간격/μ)까지만 맞는다(μ는 활성 집합의 이차 항의 가장 작은 고윳값; 잰 값 1e-6–1e-4, 조건이 나쁘면 0.3까지, `bench/regression/g3/accuracy.md`). 그래서 아래의 연마를 한다(G2의 연마와 같은 생각: 판정의 정밀도가 아니라 해에 반올림 수준으로).
- **KKT 연마**: 0 아닌 계수의 집합 A와 그 부호 σ로 정상 조건은 연립방정식 (Z_AᵀΩZ_A + l2·I)b_A = Z_AᵀΩv − l1·σ다. Gram 블록의 Cholesky로 풀고, LAPACK이 그 인자에서 어림한 1-노름 조건수(차원 a배까지 2-노름 조건수보다 클 수 있다)가 64a를 넘으면 데이터에서 다시 구한 잔차로 최대 3번 개선한다(ridge와 같은 double-double 잔차 패스, G51). 부호가 모두 그대로이고, A 밖의 열마다 |zⱼᵀΩr| ≤ l1(1 + 10⁻⁹) + 64u√(dⱼ vᵀΩv)이면(dⱼ = zⱼᵀΩzⱼ; 반올림 여유) 그 해를 쓴다. L1 항이 없으면(l1_ratio = 0) 모든 열이 A이고 부호는 보지 않는다.
  - 연마는 간격 판정을 통과했을 때, 그리고 좌표 하강 도중 두 스윕 연달아 모든 계수의 부호가 같을 때(활성 집합이 자리를 잡았을 때, 부호 꼴마다 한 번) 해 본다. 도중의 연마가 통과하면 그 해의 간격으로 수렴을 판정한다(좌표 하강은 조건이 나쁜 활성 집합에서 선형으로만 수렴하는데, 연마는 그것을 한 번에 푼다: (10⁴, 1000)의 경로에서 스윕 수가 줄었다).
  - 간격 판정을 통과했는데 연마가 실패하면(alpha_max 바로 아래처럼 아직 활성 집합이 다르다) 허용오차를 10⁻⁶배로 줄여 좌표 하강을 이어 간 뒤 다시 연마한다(최대 2번). 끝내 통과하지 못하면(예: 똑같은 열이 둘이라 해가 하나가 아니고 Gram 블록이 특이하다) 좌표 하강의 해를 낸다.
  - 연마한 해는 A와 부호에만 달려 있다. 그래서 경로의 해와 같은 alpha의 단독 적합, 그리고 alpha를 어떤 순서로 줘도 비트까지 같다(테스트 `test_path_equals_single_fits`). 해가 0인 계수는 정확히 0이다.
  - 연마한 해의 정확도: 개선하지 않으면 Gram 블록의 반올림이 정한다(잰 값 8.6e-15 안). 개선은 ridge와 같은 double-double 잔차 패스(G51)라 고정점이 정확한 문제의 해이고, 개선한 해는 3e-16 안이다(설계의 조건수 10⁶까지). 개선은 조건수 어림이 문턱을 넘을 때만 돌므로 조건이 좋은 문제와 경로의 비용은 그대로다.
- **경로** (G46): alpha를 큰 것부터 풀고 앞의 해에서 시작한다(따뜻한 시작). 결과는 준 순서다. 공분산 갱신이면 alpha 하나의 비용은 행 수와 상관없다(스윕 O(k·a), 연마 O(a³)).
- **교차검증** (G46): 겹은 가중치로 나타낸다: 겹 q의 학습 가중치는 w·[fold ≠ q], 시험 가중치는 w·[fold = q]다. 그래서 cv개의 적합이 배치 하나이고(`run_batch`로 겹마다 태스크 하나), 각 적합은 그 행만으로 한 적합과 같다(G20의 가중치 0 규칙). 시험 행의 오차는 경로의 모든 alpha에 대해 시험 행의 덩어리마다 `dgemm`으로 예측을 만들어 Σw(y − ŷ)²를 순서대로 더한다. 격자는 모든 데이터에서 한 번 정한다(scikit-learn의 `LassoCV`와 같다). 고른 alpha로 모든 데이터를 다시 적합하고(GCV는 같은 SVD의 그 alpha 해를 그대로 쓴다), 결과는 그 alpha의 단독 적합과 비트까지 같다.
  - 겹의 순열은 `rng.permutation(n)` 하나다(sd.random: 스레드 수와 상관없다). 배치의 모든 회귀가 같은 겹을 쓴다.
  - `cv_se`는 겹 오차의 표준편차/√cv다(glmnet의 `cvsd`는 겹 크기의 가중 평균을 쓴다; 겹 크기가 같으면 같다).
  - **빈 겹** (G107, 2026-11-19): 시험 행의 가중치가 모두 0인 겹은 잴 오차가 없어 평균과 표준편차에서 뺀다(그 수 c′로 평균, 표준편차/√c′). 커널은 그 겹의 시험 오차 합을 0으로 두고 `dgemm`을 부르지 않는다(예전에는 M = 0인 `dgemm`이 프로세스를 끝냈다). 모든 겹이 비면 가중치가 모두 0이고 G96의 NaN 결과(`nobs` 0)다. 데이터가 없으면(n = 0) 격자도 NaN이고 결과는 같은 NaN이다.
- **열이 많은 경로의 메모리** (2026-11-19): 활성 집합이 비면 Gram 블록을 만들지 않는다(예전에는 열 0개의 `dsyrk`, ldc = 0이 프로세스를 끝냈다: p > 1,024에서 해가 모두 0인 lasso, 그리고 그때의 `lasso_cv`·`elastic_net_cv`는 늘). 정규화 GLM의 naive 경로(열 1,024개 초과)는 연마의 KKT 검사와 마지막 값(쌍대 간격, 유효 자유도)에 (p + 3)²의 작업 Gram 대신 열마다의 합(가중 평균, 기울기, 대각)과 활성 열의 Gram만 만든다(메모리 O(np); p = 4,000에서 128 MB가 사라졌다).
- **alpha = 0** (G106, G110, 2026-11-19): `ridge`·`lasso`·`elastic_net`의 alpha = 0은 최소 노름의 최소제곱 해다. lasso·엘라스틱 넷은 Python 층이 그 자리만 ridge 커널(alpha 0의 SVD 풀이)로 다시 풀어 갈아 끼운다(경로, 교차검증의 겹과 시험 오차, 다시 적합 모두, 배치의 alpha도; `converged` True, `n_iter` 0, `dual_gap` 0, `df_model`은 0 아닌 계수의 수, G115). 커널은 alpha = 0에서 좌표 하강을 하지 않는다(경로는 큰 alpha부터라 0은 끝에 있어 따뜻한 시작에 쓰이지 않는다; 모든 alpha가 0이면 Gram도 만들지 않는다). X (200, 400)의 `lasso(alpha=0)`이 9.5 ms에서 5.7 ms로 `ridge(alpha=0)`(5.7 ms)과 같아졌다. 좌표 하강은 해가 유일하지 않은 설계(겹친 열, p > n)에서 시작점에 달린 최소점에 멈추고 반복도 많아서다. 정규화 GLM의 alpha = 0(G50)과 다르다.
- **정규화 GLM** (G47, 이름 `penalized_glm`·`PenalizedGLMResult`는 G49): 이항 logit, 포아송 log(정준 링크)만이다(링크 인자, `offset`·`trials`는 없다, G52). 목적함수 F = 편차/(2·sw) + 벌점(가중 편차 Σwᵢdᵢ; glmnet과 같다)을 IRLS로 줄인다: 현재 계수에서 작업 가중치 W = w·V(μ)와 작업 반응 z = η + (y − μ)/V(μ)의 이차 근사를 같은 좌표 하강으로 푼다(안쪽의 쌍대 간격 10⁻⁹ 상대, 스윕 1,000번 상한). μ는 `glm`의 logit 링크와 같은 sd.special `expit` 커널로 μ = expit(η), 1 − μ = expit(−η)를 구하고 V = μ(1 − μ)다(G81: 같은 η면 `glm`과 같은 μ 비트이고, 결과의 `fittedvalues`도 같은 `expit`이다). 단위 편차 2·softplus(±η)의 log(1 + e^−|η|)는 둘 가운데 작은 쪽 m = expit(−|η|)로 −log1p(−m)이다. 반복마다 한 번 훑어 [√W x̂, √W, √W η, u](x̂ = (x − c)/s, u = √(w/V)(y − μ))의 Gram을 만들면 W로 중심화한 Gram, 기울기 Σw(x̂ − ĉ)(y − μ), Σw(y − μ), 작업 반응의 합이 모두 그것에서 나온다(열이 1,024개를 넘으면 naive 갱신을 위해 꾸린다). F가 오르면((F − F_old)/(|F| + 0.1) ≥ `tol`) 걸음을 반씩 줄인다(최대 64번). |F − F_old|/(|F| + 0.1) < `tol`(기본 1e-8)이면 수렴이다.
  - 연마: A와 부호를 고정한 매끄러운 문제를 뉴턴법으로 푼다(헤시안 X_AᵀWX_A/sw + l2·I는 같은 Gram, 절편은 W로 중심화한 좌표라 블록이 나뉜다; 걸음 반감도 같다). 걸음이 4u × 계수 크기 아래로 줄면 멈춘다(걸음이 줄지 않으면 √u × 크기 아래일 때만 통과). 부호와 A 밖 열의 KKT(위와 같은 꼴)를 검사하고 통과하면 쓴다. 실패하면 안쪽 허용오차를 10⁻⁶배로 줄여 IRLS를 이어 간다(최대 2번).
  - L1 항이 없으면(l1_ratio = 0) 매끄러운 문제이므로 처음부터 뉴턴법이고(정확한 안쪽 풀이의 IRLS와 같다), 실패할 때만 IRLS다. `n_iter`는 IRLS(또는 뉴턴) 반복 수다. 처음 뉴턴 걸음은 `max_iter`로 자르고, 상한에 잘리면 그 반복점(목적함수가 줄어든 점)에서 남은 반복으로 IRLS를 이어 간다(G93; 예전에는 걸음 수가 `max_iter`를 넘을 수 있었고, 실패하면 귀무 모형에서 다시 시작해 상한이 작으면 귀무 모형을 냈다).
  - 수렴(G93): 연마가 실패해 다시 돈 IRLS가 `max_iter`에 닿아 연마하지 못하면 `converged`는 False다(선형의 `lasso`·`elastic_net`도 같다).
  - y가 모두 경계(이항 0 또는 1, 포아송 0)면 `null_deviance`는 0이다(μ = ȳ가 y 자체; `glm`과 같다, 예전에는 NaN).
  - `dual_gap`은 돌려준 계수에서의 이차 근사 문제의 쌍대 간격(목적함수 단위)이다. 해에서 이차 근사와 목적함수의 기울기가 같으므로 최적성의 척도이지만, 목적함수 값의 차이의 상한은 아니다.
  - 귀무 모형은 가중 평균 ȳ의 절편만이다(절편이 없으면 η = 0). ȳ가 0(이항은 0이나 1)이면 유한한 해가 없어 계수가 NaN이고 `converged=False`다.
  - **alpha = 0** (G50): 벌점이 없으면 Python 층이 `glm`(`var_weights=weights`, 같은 `max_iter`·`tol`)을 불러 alpha가 0인 자리의 `params`, 편차, 귀무 편차, 반복 수, 수렴, `df_model`을 그것으로 바꾼다. 분리 판정은 `glm` 커널의 증명서(§3.4) 그대로이고(코드를 되풀이하지 않는다), 분리면 `params`가 NaN, `separation=True`, η·μ는 `glm`의 극한값, `dual_gap`은 NaN이다. 분리가 없으면 결과가 `glm`과 같다(비트까지). alpha > 0이면 목적함수가 계수의 크기와 함께 커지므로 분리돼도 유한한 해가 있어(L1만이어도) 판정하지 않고 `separation`은 False다. 커널은 alpha = 0에서도 돌고(경로의 따뜻한 시작은 그대로) 그 값은 버린다.
  - 진단(편차 잔차 등)과 강건 공분산은 두지 않는다.
- **유효 자유도** (`df_model`): tr Z_A(Z_AᵀΩZ_A + l2·I)⁻¹Z_AᵀΩ = a − l2·tr H⁻¹(H는 연마의 Gram 블록, `potri`). l2 = 0이면 a(lasso, Zou–Hastie–Tibshirani 2007), A가 모든 열이면 ridge의 Σs²/(s² + l2)이다. 엘라스틱 넷의 이 식은 Zou–Hastie(2005)의 것이다(glmnet의 `df`는 0 아닌 계수의 수). GLM은 Ω에 작업 가중치를 곱한다. lasso·엘라스틱 넷의 alpha = 0은 l2 = 0이라 a, 곧 ridge 해의 0 아닌 계수의 수다(G115; Python 층이 센다).
- **결정성**: 행을 도는 모든 합은 크기로만 정한 덩어리를 순서대로 더하고(Gram은 고정 슬롯), 좌표 순서는 열 순서다. 회귀 하나의 계산은 배치 안이든 밖이든 같으므로 배치와 반복문이 비트까지 같고, 1스레드와 10스레드도 같다(테스트로 확인; 7만 행의 lasso 경로, ridge, lasso_cv, 로지스틱).
- **정확도** (`bench/regression/g3/accuracy.py`, 결과 `accuracy.md`): `tools/regref.py`의 `penalized`가 sd 해의 활성 집합과 부호로 정상 조건을 Arb에서 정확히 풀고(GLM은 뉴턴법과 Krawczyk 검사로 근 하나를 증명), 부호와 0인 계수마다의 엄격한 KKT 부등식으로 그 해가 최소점임을 증명한다. 계수 벡터의 상대오차(노름):

  | 문제 | sd | scikit-learn `tol=1e-4` | `tol=1e-12` |
  |---|---:|---:|---:|
  | lasso·엘라스틱 넷 (n = 60, 가중치·0인 가중치·표준화·절편 없음·상수 열, 8개) | 6.9e-16 – 1.5e-15 | 3.6e-6 – 7.8e-5 | 5.2e-14 – 1.4e-12 |
  | ridge (3개) | 5.1e-16 – 1.2e-15 | 7.1e-16 – 1.2e-15 | 5.1e-16 – 1.3e-15 |
  | 로지스틱·포아송, l1_ratio 1, 0.5, 0 (7개) | 1.0e-16 – 2.8e-16 | 5.1e-7 – 1.6e-4 | 3.2e-15 – 1.9e-12 |
  | 조건수 10⁶의 lasso / 엘라스틱 넷(연마가 개선) / ridge alpha 1e-8 | 7.6e-17 / 5.3e-17 / 1.4e-17 | 7.2e-5 / 6.0e-2 / 2.5e-9 | 7.4e-13 / 7.8e-10 / 2.6e-12 |

  - 조건수를 바꾼 문제(n = 200, k = 8)는 lasso·엘라스틱 넷 κ ≤ 10⁶에서 8.6e-15 안이다: 연마가 개선한 칸은 3e-16 안(double-double 잔차, G51; double 잔차로는 2.4e-14였다), 개선이 돌지 않은 칸(κ = 10², lasso alpha_max/20)이 8.6e-15다. scikit-learn의 기본 `tol`은 κ = 10²–10⁶에서 1.8e-4–0.34다. ridge는 alpha 1e-3, 1e-8 모두 κ ≤ 10⁶에서 2e-16 안이고(double-double 잔차의 개선, G51; 그 전의 double 잔차로는 alpha 1e-8, κ ≥ 10⁴에서 1.2e-12), 최소제곱에 가까운 κ = 10⁸–10¹⁰, alpha 1e-12–1e-18에서 1.7e-15 안이다(G1의 OLS 수준).
  - 행 수: n = 1,000과 20,000(k = 5)에서 lasso·엘라스틱 넷·ridge 5.6e-16 안, 로지스틱 L1·포아송 L2 1.9e-16 안이다.
  - 테스트(`tests/regression/test_penalized.py`)의 기준은 4e-15(개선하는 조건수 10⁶의 세 문제는 4e-16)다. scikit-learn과는 `tol`을 1e-14로 조이면 lasso·엘라스틱 넷 1e-9, ridge 1e-12, 로지스틱 1e-7(saga), 포아송 1e-10 안에서 같다.
- **scikit-learn·glmnet과 다른 점** (확인한 것과 못 한 것):
  - scikit-learn의 `Ridge`는 Σw r² + alpha‖β‖²이라 그 alpha는 여기의 sw배이고, `LogisticRegression`의 `C`는 1/(sw·alpha)다. `dual_gap_`은 scikit-learn 1.9.1에서 n_samples로 나눈 값이라(`enet_path`) 여기의 `dual_gap`과 같은 단위다. `Lasso`·`ElasticNet`의 기본은 Gram을 쓰지 않는다(`precompute=False`); 여기는 열 1,024개까지 Gram이다(glmnet은 500개 미만).
  - scikit-learn의 `n_iter_`은 전체 스윕(epoch)이고 간격 안전 선별(gap safe screening)을 한다. 여기는 활성 집합 스윕도 세고 선별은 없다.
  - scikit-learn의 `LassoCV`는 기본으로 섞지 않은 연속 겹(`KFold`)이다. 여기는 `rng`의 무작위 겹이다(glmnet과 같은 꼴). 같은 겹을 주면 `mse_path_`의 평균, `alpha_`, 계수가 1e-9 안에서 같다.
  - scikit-learn의 liblinear는 절편에도 벌점을 준다(다른 문제). saga와 lbfgs의 기본 `tol`은 계수를 1e-5–4e-4 다르게 둔다(`bench/regression/g3/speed.md`).
  - glmnet: λ가 여기의 alpha, glmnet의 alpha가 여기의 l1_ratio다. 이항·포아송과 lasso는 같은 목적함수다. 정규 분포족의 glmnet은 y를 모표준편차 s_y(가중; 절편이 없으면 √(Σw y²/sw))로 나눠 풀고 되돌리므로 L2 항이 λ(1 − α)/s_y다: glmnet(λ, α)의 해는 여기의 L1 계수 λα, L2 계수 λ(1 − α)/s_y인 해(alpha = λα + λ(1 − α)/s_y, l1_ratio = λα/alpha)와 같고, s_y = 1인 y에서만 같은 목적함수다(설명서의 목적함수와도 다르다; R의 `?glmnet`이 비교하려면 y를 표준화하라고 권하는 까닭).
  - R glmnet 5.2(`thresh = 1e-20`; `tests/regression/test_against_r.py`)와는 lasso(표준화 여부, 가중치), 위 대응의 엘라스틱 넷·ridge, 이항·포아송 정규화 GLM(l1_ratio 1, 0.5, 0, 표준화 여부, 가중치)의 40개 적합에서 계수가 4e-11 안(glmnet의 수렴 판정)이고 여기의 목적함수 값이 glmnet 해의 값보다 크지 않다. glmnet의 `df`는 0 아닌 계수 수와 같고 `dev.ratio`는 `rsquared`·`deviance_ratio`와 같다. 기본 경로의 첫 λ(alpha 1과 0)는 `lasso_cv`·`ridge_cv`의 격자 첫 값과 1e-13 안에서 같다(절편 있음).
  - glmnet은 기본으로 표준화하고(여기는 G45대로 하지 않는다), 기본 겹 수 10, λ 격자의 최소 비율 1e-4(n > p) 또는 1e-2(여기는 scikit-learn의 `eps=1e-3`), `cv.glmnet`의 `predict` 기본은 `lambda.1se`(여기는 `select='min'`), 엘라스틱 넷의 `df`는 0 아닌 계수의 수다. R(glmnet 5.2)로 확인했다(tests/regression/test_against_r.py): 절편이 있을 때의 sⱼ(모표준편차)와 ridge 격자의 규칙(l1_ratio = 1e-3로 계산한 λ_max)은 glmnet과 같다. 절편이 없으면 glmnet은 중심화한 모표준편차로, 여기는 RMS로 나눈다(사용자 결정 2026-11-18: RMS 유지). 정규 분포의 엘라스틱 넷·ridge는 glmnet이 y를 표준화하므로 s_y = 1일 때만 같다(api §4).
- **속도** (`bench/regression/g3/speed.py`, 결과 `speed.md`, scikit-learn 대비 '배' = scikit-learn 시간/sd 시간; 최적화 8772322 전 측정):

  | (n, k) | 스레드 | ridge (Cholesky / svd) | lasso | 경로 100개 | lasso_cv 5겹 | ridge_cv GCV | 로지스틱 L1 (saga / liblinear) | 로지스틱 L2 (lbfgs / newton-cholesky) |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | (10⁵, 10) | 1 | 0.2 / 0.82 | 0.8 | 2.6 | 1.2 | 18 | 3.2 / 0.72 | 0.27 / 0.29 |
  | (10⁵, 10) | 10 | 0.69 / 2.9 | 1.2 | 3.8 | 3.7 | 63 | 17 / 3.7 | 1.4 / 1.5 |
  | (10⁵, 100) | 1 | 0.10 / 1.2 | 1.5 | 1.7 | 1.1 | 7.1 | 7.6 / 1.7 | 0.13 / 0.39 |
  | (10⁵, 100) | 10 | 0.52 / 6.1 | 4.8 | 4.6 | 3.8 | 36 | 28 / 6.4 | 0.51 / 1.5 |
  | (10⁴, 1000) | 1 | 0.092 / 1.3 | 1.2 | 0.71 | 0.64 | 9.0 | 4.8 / 1.3 | 0.038 / 0.61 |
  | (10⁴, 1000) | 10 | 0.10 / 1.4 | 1.7 | 0.78 | 1.6 | 9.4 | 7.3 / 1.7 | 0.052 / 0.86 |

  - ridge는 QR과 SVD라 XᵀX의 Cholesky(scikit-learn의 기본)보다 1스레드 5–11배 느리고 같은 SVD 해법과는 비슷하다. 그 대신 여러 alpha와 GCV가 분해 하나로 나온다.
  - (10⁴, 1000)의 경로와 CV가 1스레드에서 느린 것은 0 아닌 계수가 수백 개인 alpha마다의 연마(Cholesky, a³/3)다.
  - 로지스틱 L2는 newton-cholesky와 같은 뉴턴법이지만 반복마다 Gram을 다시 만들고 Dot2로 η를 구해 1스레드 0.29–0.61배다(최적화 전 측정; 그 뒤 η의 NEON과 G81의 `expit`로 (10⁵, 20) 로지스틱이 1스레드 43 → 25 ms). lbfgs는 더 빠르지만 계수가 기본 `tol`에서 3e-5–4e-4 다르다.

### 3.6 강건과 분위
- **커널** (`csrc/regression/robust.cpp`): 회귀 하나가 태스크 하나다(G1의 `run_batch`). 시작은 G1의 최소제곱 핵심 그대로다(QR과 반복 개선, G21의 겹친 열, (XᵀX)⁻¹ = (RᵀR)⁻¹). 결과는 계수, 공분산(H1 또는 avar1), 시작의 R, 적합값·잔차(Dot2), 쓰인 열 표시, 스칼라 10개(nobs, rank, 척도, 반복 수, 수렴, NaN 표시, S의 반복 수와 수렴, 그리고 공통 결과 코드가 묻는 군집 수·지연 수 자리의 NaN), 가중치, MM의 S 계수다.
- **M** (G53, G55): IRLS의 한 반복은 직전 잔차의 가중치 w(r_i/s)로 y를 가중 최소제곱하고, 새 잔차로 척도를 다시 구한다(statsmodels `RLM.fit`의 순서와 같다). 그래서 고정점은 추정방정식 Σ ψ(r_i/s(β)) x_i = 0과 척도 규칙 s(β)를 함께 푼 해이고, statsmodels와 같은 해다.
  - 판정 전의 최소제곱은 GLM의 중간 반복과 같은 정규방정식(덩어리의 `dsyrk`, Cholesky, G40; 조건이 나쁘거나 G21이 열을 빼면 QR)이고, 판정 뒤와 마지막은 반복 개선한 QR이다(`lsq.hpp`의 `fit_one`을 lean으로).
  - **뉴턴 걸음**: IRLS는 선형으로 수렴하고, MAD 척도에서는 그 비율이 0.7에 이르기도 한다(n = 61, Huber에서 statsmodels는 tol 1e-15까지 89번). 그래서 목적함수의 판정 뒤, 그리고 판정 전이라도 걸음이 직전의 1/2 아래로 줄지 않는 반복이 3번 이어지면(GLM의 G44와 같은 판정), F(β) = Σ ψ(z_i) x_i의 뉴턴 걸음 β + s M⁻¹F, M = XᵀΨ′X + (XᵀΨ′z)(∇s)ᵀ로 간다. ∇s는 MAD면 가운데 행의 −sign(r_m) x_m/C(짝수 n이면 두 행의 평균), 제안 2면 −Σ_안 r_i x_i/(s(2T − n_밖 d²)), 고정 척도면 0이다. F는 열마다 double-double로 더하고, M은 k × k LU로 푼다. 걸음은 ‖F‖가 줄고 새 척도가 원래의 1/2–2배 안일 때만 받고, 아니면 그 반복은 개선한 QR의 IRLS다. 척도 조건은 ‖F‖만으로는 막지 못하는 걸음을 막는다: 작은 z에서 ψ(z) ≈ z라 척도가 커지기만 해도 ‖F‖가 줄고, 재하강 ψ는 모든 행을 c 밖으로 보내도 F = 0이다. 멈추는 규칙은 GLM과 같다(걸음 ≤ 2⁻⁵⁰‖β‖, 또는 두 걸음의 비율로 내다본 나머지가 그 아래, 또는 걸음이 더 줄지 않을 때). Arb 문제 19개에서 IRLS만으로는 12–50번이고 두 문제가 `max_iter` 50에 닿아 계수 오차가 7×10⁻¹⁰·9×10⁻¹¹이며, 뉴턴 걸음을 넣으면 6–19번에 반올림 수준이다.
  - **반올림 수준의 완전 적합**: 행의 절반 넘게가 한 직선(평면) 위에 있으면 IRLS는 그 행들의 잔차와 척도를 함께 0으로 다가가게 할 뿐 정확히 0에 닿지 못한다(β가 반올림된 값이라). 척도가 반올림 수준이 되면 r_i/s가 반올림 잡음이라 ψ·가중치·뉴턴 모형이 뜻을 잃고, 예전에는 그 잡음에서 받은 뉴턴 걸음이 엉뚱한 점으로 뛰었다(x = 1..10, y = 1 + 2x, y₃ = 50, y₇ = −20에서 [1.971, 1.820], 척도 0.936, `converged=True`). 그래서 반복마다 척도를 정하는 행들의 잔차가 분위 회귀의 기준 (k + 16)u(\|y_i\| + ‖x_i‖₁‖β‖∞) 안에 드는지 본다(MAD는 절반 넘게, 제안 2는 0이 아닌 행이 2T/d² 이하, S의 M-척도는 T 이하; 먼저 가장 큰 기준으로 r만 한 번 훑는다). 들면 척도를 0으로 두고 멈추며, 계수는 그 행들만의 최소제곱(가중치 1/0, 개선한 QR)이다. 이는 정확히 0인 완전 적합(API §5)과 같은 결과이고 그 극한이다. MM은 S의 3–4단계에서 같은 판정을 한다. 제안 2의 척도 풀이도 안쪽 행이 모두 정확히 0이면(근이 없으면) 0을 낸다(예전에는 마지막 t의 1/√t, 위 예에서 17.8).
  - **MAD**: 0을 중심으로 \|r_i\|/0.6744897501960817의 중앙값을 선택(`nth_element`)으로 구한다(statsmodels는 정렬). 짝수 n은 아래 절반의 최댓값과의 평균이다(numpy의 `median`과 같은 (a + b)/2).
  - **제안 2**: t = 1/s²에서 f(t) = Σ min(r_i² t, d²)/2는 오목하고 증가하는 조각별 일차함수이므로, t = 0에서 시작한 뉴턴 걸음 t ← (T − n_밖 d²/2)/(Σ_안 r_i²/2)는 근까지 단조롭게 올라가 근에서 멈춘다(보통 3–5번, 합은 double-double). T = nobs·((nobs − rank)/nobs)·E χ(Z), E χ(Z) = d² + (1 − d²)Φ(d) − 1/2 − dφ(d)는 statsmodels의 식이다(Φ는 `erfc`로).
- **MM** (G53): robustbase 0.99.7 `lmrob`의 fast-S(Salibian-Barrera–Yohai 2006)와 M 단계다. R의 `lmrob.control()` 기본값과 같다(`nResample` 500, `tuning.chi` 1.54764, `bb` 0.5, `best.r.s` 2, `k.fast.s` 1, `k.max` 200, `refine.tol` 1e-7, `solve.tol` 1e-7, `mts` 1000, `tuning.psi` 4.685061, `cov` `.vcov.avar1`). 쓰인 열(시작 최소제곱이 남긴 열)을 행 우선으로 꾸려 쓴다.
  1. 후보 500개를 따로 계산한다(한 회귀면 4개씩 묶은 태스크로 스레드풀): 기본 부분집합(robustbase의 "nonsingular" 표본처럼 선형 독립인 행만, 재직교화한 그람-슈미트로 판정), LU로 그 해, robustbase의 MAD(1.4826·median\|r\|)에서 한 걸음(척도 한 걸음 s ← s√(Σρ̃(r/s)/T), T = (n − p)/2, bisquare χ 가중치의 가중 최소제곱). 후보마다 Philox의 자기 원소에서 뽑으므로 계산 순서와 상관없다.
  2. robustbase의 선택을 후보 순서대로: 남긴 것 중 가장 큰 척도 W에서 Σρ̃(r/W) < T인 후보만(척도가 W보다 작다) 척도를 구해(아래, 상대 10⁻¹⁰) W를 바꾼다. 척도를 구하는 후보는 보통 십여 개다(500개 모두에 구하면 (10⁵, 10)에서 4배 느리다).
  3. 남긴 2개를 robustbase처럼 상대 변화 10⁻⁷까지 다듬고(척도 한 걸음 + 가중 최소제곱) 척도가 작은 쪽을 고른다. 그 뒤 고정점(β가 자기 잔차의 정확한 M-척도로 만든 가중치의 가중 최소제곱)까지 더 반복하고, 마지막 하나는 개선한 QR이다. 가중 최소제곱만의 반복은 비율 0.8–0.9로 선형이라 50–90번 걸렸다. 그래서 반복마다 먼저 추정방정식 F(β) = Σψ(r_i/s(β))x_i(ψ ∝ ρ̃′, s(β)는 정확한 M-척도)의 뉴턴 걸음 β + s M⁻¹F를 해 보고(G82), ‖F‖를 줄이고 새 척도가 1/2–2배 안일 때만 받는다(M의 뉴턴과 같다). 아니면 그 반복은 정규방정식의 가중 최소제곱이다. M은 M 단계의 뉴턴과 같은 XᵀΨ′X + (XᵀΨ′z)(∇s)ᵀ이고, M-척도의 기울기는 Σρ̃(r_i/s) = T를 미분한(Σρ̃′(z_i)(x_i + z_i∇s) = 0) ∇s = −Σψ(z_i)x_i/Σψ(z_i)z_i다(고정점에서는 0). 예전 코드는 여기에 s를 한 번 더 곱해 s ≠ 1인 자료에서 뉴턴 걸음의 이 항이 틀렸다(고정점에서는 두 합이 0이라 결과는 같고, 고정점 가까이에서 시작하는 이 단계에서는 반복 수도 거의 같았다). 고정점은 그대로이고, 새 잔차는 Dot2라 F의 값이 정확하다. 멈춤 규칙은 그대로다(걸음 ≤ 2⁻⁴⁶‖β‖, 또는 두 걸음의 비로 내다본 나머지가 그 아래). (10⁵, 10)에서 2–3단계의 S 반복 합이 53 → 27번, R 문제 다섯 개에서 36–77 → 18–39번이고, Arb로 증명한 고정점(`tools/regref.py`의 `s_estimate`: (β, s)의 뉴턴과 Krawczyk 판정)까지의 거리가 R 문제 네 개(n ≤ 500)에서 3.2×10⁻¹⁵–2.5×10⁻¹⁴ → 1.9×10⁻¹⁷–1.2×10⁻¹⁶이다(계수 벡터의 노름 상대). MM 전체는 (10⁵, 10) 1스레드 255 → 195 ms, 10스레드 66 → 49 ms(빠르게 잰 값).
  4. M 단계: bisquare(c = 4.685061)로 S 척도를 고정한 M의 IRLS(뉴턴 걸음 포함).
  - **M-척도**: g(s) = Σρ̃(r_i/s) − T는 s → 0에서 (0이 아닌 잔차 수) − T, s → ∞에서 −T로 감소한다. 0이 아닌 잔차가 T 이하이면 0(robustbase의 완전 적합)이고, 아니면 부호로 아는 괄호 (0, ∞) 안에서 g와 g′를 한 번에 더하는 뉴턴(괄호를 벗어나면 이분, 끝이 없으면 2배·1/2배)으로 반올림까지 푼다.
  - 난수: 회귀마다 Python이 `rng`에서 64비트 두 단어를 받아 키로 쓴다. 후보 j는 원소 j·2³² + t의 첫 단어로 행을 뽑고(Lemire의 방법, 기각은 그 원소의 여분 단어; sd.random과 같은 규칙), 이미 고른 행이거나 독립이 아니면 다음 t로 간다(`mts` 1000번까지).
  - **큰 n** (G58): n > 2000(`fast.s.large.n`)이고 p < 400이면 robustbase의 `fast_s_large_n`(Salibian-Barrera–Yohai 2006의 큰 n 방식)이다. C 소스가 설치본에 없어 R의 `lmrob.control()` 값과 `trace.lev = 4`의 추적 출력으로 단계를 확인했다: 행 2,000개를 비복원으로 뽑아(Philox 원소 2⁶³ + t의 부분 Fisher–Yates) 400개씩 5묶음으로 나누고, 묶음마다 1–2단계를 그 400행에서 한다(묶음마다 후보 500개, T = (400 − p)/2; 묶음 g의 후보 j는 원소 (g·500 + j)·2³² + t). 남은 10개는 뽑은 2,000행 전체에서 자기 묶음의 척도로 한 걸음 개선하고(R의 출력도 "1 refinement"; `k.m_s`는 M-S 추정의 값이라 이 단계에 쓰이지 않는다) 그 2,000행의 M-척도로 2개를 고른다(지금 남긴 것보다 작을 때만 바꾼다). 그 뒤 3–4단계는 모든 행에서 같다. 모든 행의 작은 n 탐색과 S 척도가 같았고(`bench/regression/g4/speed.md`의 세 문제에서 12자리, 테스트의 n = 6,000 지렛대 25%·40%에서 1e-10 안), (10⁵, 10)에서 7.5배(1스레드)·4배(10스레드) 빠르다.
- **추론** (G57):
  - M의 H1: statsmodels `RLMResults.bcov_scaled`의 식 그대로이고 (XᵀX)⁻¹은 시작의 (RᵀR)⁻¹(statsmodels는 `pinv`)이다. ψ′·ψ²의 합은 double-double이다. R `MASS`의 `summary.rlm`은 같은 식에서 var ψ′를 n − 1로 나눈다(K가 1 + p var/(n m²)·n/(n − 1)): R과의 비교는 그 비를 곱해 1e-14 안이다.
  - MM의 avar1: robustbase 0.99.7의 `.vcov.avar1` 소스와 같은 식이다. A = s(XᵀΨ′X)⁻¹, a = A·Xᵀ(ψ′(z)z)/mean(χ′(z₀)z₀), V = A(XᵀΨ²X)A − (a(AXᵀ(ψχ))ᵀ + (AXᵀ(ψχ))aᵀ)/n + mean(χ(z₀)² − b²)aaᵀ/n(z₀ = S 잔차/s, χ = ρ̃). 이는 MM 추정의 영향함수에 S 척도의 영향을 더한 β̂ − β ≈ Σ[Ax_iψ_i − a(χ_i − b)/n]의 공분산이다(Croux, Dhaene, Hoorelbeke 2003; 위 식의 b 항은 robustbase가 쓴 꼴 그대로). 합은 시작의 R로 얇은 Q 좌표(q_i = R⁻ᵀx_i)에서 만들고 V = R⁻¹V_Q R⁻ᵀ로 되돌린다: XᵀΨ′X를 직접 만들어 역행렬을 구하면 κ(X)²u라 스택 손실(κ ≈ 10³)에서 8×10⁻¹⁴ 틀리고(robustbase도 같은 크기), Q 좌표에서는 7×10⁻¹⁶이다. 음의 고윳값이 있으면 robustbase의 `posdefify`(0으로 두고 대각을 원래 값으로 다시 맞춘다)를 `syevd`로 한다.
- **Theil–Sen·Siegel** (G56, `csrc/regression/slopes.cpp`): API §5. 쌍의 기울기는 scipy가 만드는 double 그대로이고, 순위 통계량을 정확히 고른다.
  - Theil–Sen의 순위는 중앙값(짝수면 둘)과 Sen 구간의 두 끝, 최대 4개다. 기울기가 2²³개 이하면 한 배열에 모아 순위 순서로 `nth_element`, 그보다 많으면 훑기다: 고정 키 Philox로 쌍 2¹⁶개를 뽑아 순위마다 피벗 lo, hi를 표본의 위치 ± (4√m + 4)에서 고른다. 한 번 훑으며(행 덩어리 128개, 크기로만 정한다) 순위마다 lo보다 작은·같은, hi보다 작은·같은 값의 수를 세고, 사이 값의 예상 수가 2²² 이하면 사이 값을 모두, 넘으면 쌍의 해시로 고른 부분표본(약 2¹⁶개)을 모은다. 답이 피벗과 같으면 끝, 사이에 있고 모두 모았으면 선택, 부분표본이면 그것으로 피벗을 좁혀 다시 훑는다. 표본이 빗나가 답이 피벗 밖이면 이미 아는 경계와 그 피벗 사이를 정확한 수로 다시 훑는다. 비교는 순서를 보존하는 정수 키(−0은 +0)로 한다. n = 10⁴(5×10⁷쌍)은 1번, 10⁵(5×10⁹쌍)은 2–3번 훑는다(두 번째 훑기 뒤 사이 값이 약 5×10⁶개로 모으기 한계 2²² 근처라, 넘으면 한 번 더 좁힌다).
  - 무한대 기울기(차의 넘침)가 있으면 Sen 구간의 순위를 유한한 기울기 수로 다시 정해 고른다(scipy의 nt).
  - 0의 부호는 +0으로 낸다: scipy의 −0·+0은 정렬이 같은 값 사이에서 무엇을 앞에 두는지에 달려 있다.
  - **기울기가 없는 쌍** (2026-11-19): 차가 넘쳐 inf/inf = NaN인 쌍은 x가 같은 쌍처럼 기울기가 없다고 보고 빼며 N에도 세지 않는다(scipy는 `nanmedian`·`isfinite`로 같은 쌍을 뺀다; 예전에는 정렬 키가 NaN을 큰 값으로 넣어 `high_slope`가 NaN이었다). 그런 쌍은 x의 범위가 넘칠 때만 생기므로, 그때만 O(n²)로 센다. Siegel도 점마다 그 쌍을 뺀다.
- **2026-11-19의 고침** (G93–G96):
  - `converged`(G93): M의 IRLS는 판정 뒤의 연마가 멈춤 규칙으로 끝나야(또는 척도가 0이어야) True이고 `max_iter`에 잘리면 False다. MM은 그에 더해 S 추정의 다듬기(10⁻⁷)와 고정점 반복이 끝나야 한다(S의 반복 수와 수렴은 따로 내지 않고 이것에만 쓴다).
  - 정확한 시작(G94, G95): 절편이 있고 y가 정확히 같으면 해는 (y₀, 0, …, 0)이고 척도 0, 반복 0, `converged` True다(MM은 그것이 S 계수, M 단계 없음). n = rank이면 최소제곱의 해가 정확하므로 같은 규칙이다(MM은 예전에 S의 척도 걸음이 T = 0으로 나눠 모든 값이 NaN이었다). 그때 `df_resid` = 0이라 공분산은 NaN이다. Huber의 제안 2는 T ≤ 0(n ≤ rank)이면 척도 0이다(예전에는 inf).
  - 열이 없으면(rank 0) 반복할 것이 없다: 척도는 잔차(y)의 척도(M은 MAD 또는 제안 2, MM은 S의 M-척도), `n_iter` 0, `converged` True다(`quantreg`와 같다). n = 0은 G96의 NaN 결과다(MM은 예전에 `getrf`가 `RuntimeError`와 LAPACK의 stderr 출력을 냈다).
- **분위 회귀의 2026-11-19 고침**:
  - 꼭짓점의 풀이(`refined`): 보정이 u‖v‖∞ 아래면 바뀐 것으로 세지 않고, 끝에 그 크기 아래의 성분을 0으로 둔 해가 기저 행의 잔차의 최댓값을 늘리지 않으면 그것을 쓴다. double-double 잔차는 큰 성분만 보므로, 참값이 0인 꼭짓점 좌표(상수 y의 기울기 등)에 1e-63이 남았다. 그래서 상수 y의 해가 정확히 (y₀, 0, …, 0)이다(G94).
  - 'ker'의 완전 적합: 모든 잔차가 eps(G67) 안이면 희소도를 정할 수 없어 `bse`는 NaN이다(G65; 예전에는 커널 폭이 반올림 잡음이라 1e-16이었다).
  - n = 0은 G96의 NaN 결과이고 `converged` False, `n_iter` 0이다(예전에는 True). 열이 없으면(rank 0) `converged` True, `n_iter` 0(`rlm`과 같다).
- **정확도와 대조** (`bench/regression/g4/accuracy.md`; 테스트 `tests/regression/test_robust.py`)
  - Arb(`tools/regref.py`의 `rlm`): 해의 ψ 조각과 척도의 행(MAD의 가운데 행, 제안 2의 안·밖)을 고정한 매끄러운 방정식을 뉴턴으로 풀고, 그 조각과 순서가 상자 전체에서 확실한지 확인한 뒤 Krawczyk 판정으로 근을 증명한다(MM은 S 계수와 척도를 고정한 M 단계; 그 S는 `s_estimate`로 증명한 고정점을 반올림한 값이고, sd의 S가 그것과 원소마다 9.4×10⁻¹⁶, 척도 2.2×10⁻¹⁶ 안이다). 19개 문제(ψ 3 × 척도 2 × 짝·홀 n, 절편 없음, n = 400, 스택 손실 ψ 3, MM 2)에서 계수 벡터 1.1×10⁻¹⁶, 척도 1.4×10⁻¹⁵, 표준오차 1.2×10⁻¹⁵, 가중치(절대) 1.1×10⁻¹⁵ 안이다. 척도의 오차는 가운데 잔차의 상쇄(\|r_m\|가 \|y\|보다 작다)에서 온다.
  - statsmodels(tol 1e-15, maxiter 1000, `HuberScale(d=2.5, tol=1e-15)`): ψ 3 × 척도 2에서 계수 2.3×10⁻¹⁵, 척도 2.1×10⁻¹⁵, 표준오차 1.3×10⁻¹⁵, 가중치 4.7×10⁻¹⁵ 안. statsmodels의 기본값(목적함수의 절대 변화 1e-8)과는 그쪽의 수렴 오차만큼(10⁻⁵ 이하, 속도 표의 문제에서 3×10⁻¹⁵–2.4×10⁻¹⁴) 다르다.
  - R(4.6.2, `tests/regression/ref/robust_r.json`, `tools/gen_robust_r.py`; R은 테스트 의존성이 아니라 결과를 한 번 만들어 둔다): `MASS::rlm(scale.est='proposal 2', k2=2.5, acc=1e-15)`와 계수 4.1×10⁻¹⁵, 척도 1.7×10⁻¹⁴ 안, 표준오차는 var ψ′의 n − 1을 맞추면 10⁻¹⁴ 안이다. `MASS`의 MAD(0.6745)와는 척도 1.0–1.9×10⁻⁵, 계수 3×10⁻⁷–3×10⁻⁶ 다르다. `lmrob`의 M 단계를 S 시작(sd의 부분표본이 닿은 S를 `s_estimate`로 증명한 고정점)에서 `rel.tol` 1e-15로 부르면 계수 1.1×10⁻¹⁵, `.vcov.avar1`은 7.2×10⁻¹⁵(스택 손실은 robustbase 쪽 κ²u로 1.5×10⁻¹³) 안이다. `lmrob` 자체(자기 부분표본, S의 `refine.tol` 1e-7; 큰 n 문제 n = 5,000 포함)와는 계수 1.1×10⁻⁹–1.2×10⁻⁷, 척도 1.3×10⁻⁸–1.1×10⁻⁵다. lmrob의 `scale`은 마지막 고정점 걸음의 값이라 정확한 M-척도와 그 정도 다르고(같은 잔차의 M-척도는 R의 `find_scale`과 10⁻¹⁰ 안에서 같다), 그래서 sd보다 작을 때도 있다.
  - scipy: 무작위·동률·격자·평평한 자료, n = 2–5,000, 두 방법, 여러 `alpha`에서 비트까지 같다(0의 부호 빼고). 작은 n에서 쌍의 정확한 유리수 기울기의 중앙값과 4ulp 안, 기울기가 double로 정확한 정수 자료에서는 같다. 훑기 알고리즘은 테스트가 한계를 낮춰(모으기 한계 40–5,000) 작은 문제로도 돌려 scipy와 비트까지 같음을 확인한다.
- **속도** (`bench/regression/g4/speed.md`; 최적화 8772322 전 측정): M(Huber, bisquare)은 statsmodels RLM의 기본값보다 (10⁵, 10)에서 3.2–3.3배(1스레드)·8.2–8.7배(10스레드), (10⁶, 50)에서 8.3–9.2배·29–31배 빠르다(statsmodels는 반복마다 `pinv`). MM(큰 n 방식)은 (10⁴, 5) 63 ms·35 ms, (10⁵, 10) 240 ms·65 ms, (10⁶, 10) 1.88 s·0.42 s(1·10스레드; 최적화 전 측정)이고 R `lmrob`(92 ms, 398 ms, 2.65 s)보다 1.4–1.7배·2.7–6.2배 빠르다. lmrob과의 계수 차이는 10⁻⁸ 수준이다. Theil–Sen은 scipy보다 n = 10³에서 11배, 10⁴에서 36배·138배, Siegel은 2.3–2.7배(1스레드)·16–23배(10스레드) 빠르고, n = 10⁵(scipy는 메모리가 모자란다)은 Theil–Sen 37 s·4.2 s, Siegel 68 s·7.9 s다.
- **분위 회귀** (G60–G63, `csrc/regression/quantile.cpp`): 목적함수 Σ m_i ρ_q(y_i − x_iβ)는 선형계획이고, 해는 꼭짓점(잔차가 0인 k행이 정하는 점)이다. 회귀 하나가 태스크 하나이고(G1의 `run_batch`), 한 회귀의 분위들은 준 순서로 푼다.
  1. 검사와 최소제곱: G1의 최소제곱 핵심이 겹친 열(G9, G21), R, (RᵀR)⁻¹(공분산), 내부점법의 시작을 준다.
  2. 쓰인 열(절편 먼저)을 행 우선으로 꾸리고, y와 x가 비트까지 같은 행(0의 부호는 빼고)을 한 행과 개수 m_i로 합친다. 행의 해시를 윗비트로 묶음에 나누고(안정한 계수 정렬이라 묶음 안은 행 순서) 묶음마다 표로 찾으므로 대표 행은 늘 가장 앞 행이다. 정수 반응과 더미 설계의 동률이 만드는 퇴화(잔차가 0인 행이 k개보다 많은 꼭짓점)를 대부분 없앤다.
  3. **내부점법** (첫 분위): quantreg `rq.fit.fnb`의 lp_fnm 그대로의 쌍대 선형계획 max yᵀa, Xᵀa = (1 − q)Xᵀm, 0 ≤ a ≤ m이다. Mehrotra의 예측–보정, 경계까지 0.99995의 걸음, 중심화 μ(g/μ)³/(2n). 뉴턴 행렬 XᵀQX는 n으로만 정한 4,096행 덩어리의 `dsyrk`를 순서대로 더하고 Cholesky로 푼다. lp_fnm과 다른 곳은 셋이다.
     - 시작 계수는 최소제곱의 절편을 그 잔차의 q분위수만큼 옮긴 점이다(lp_fnm은 최소제곱 그대로). 끝 분위에서 반복이 준다: (10⁵, 10)의 네 잡음(정규, t₂, 이분산, 지수)에서 q = 0.01은 최소제곱 그대로면 100번(상한)·100·100·26번, 옮기면 37·100·65·25번이고, q = 0.1·0.25·0.5·0.9의 합은 309번과 293번, n = 20–400의 작은 문제 600개는 최대 16번과 12번이다.
     - 쌍대 여유를 z = max(r, 0) + δ, w = max(−r, 0) + δ(δ는 평균 \|r\|의 10⁻³)로 시작한다(z − w = r이라 쌍대 실행 가능하다). lp_fnm은 z = max(r, 0)이고 r이 정확히 0인 행만 r에 0.001을 더하는데, 그러면 시작이 쌍대 실행 불가능하고, 반올림 수준의 잔차(옮긴 시작의 분위 행)가 q = 1/(z/x + w/s) ≈ 10¹⁵을 만들어 뉴턴 행렬이 특이해진다. 첫 걸음의 Xᵀdx가 10⁻²만큼 어긋나고 간격이 그 어긋남에서 멈춘다: 옮긴 시작에 lp_fnm의 여유를 쓰면 위의 작은 문제 600개 가운데 15%가 40–83번 돈다.
     - 뉴턴 방정식은 조립한 행렬로 두 번 보정해 푼다(Xᵀdx = 0이 반올림 수준으로 남아 반복이 실행 가능하다).
     - 멈춤: 상대 쌍대 간격(cᵀx − bᵀy + uᵀw)/\|yᵀλ\| ≤ `tol`(기본 1e-5), `max_iter`, 또는 Cholesky가 실패할 때. 어디서 멈추든 아래 심플렉스가 정확한 꼭짓점을 찾으므로 `tol`은 속도만 바꾼다. (10⁵·10⁶, 10)에서 1e-4–1e-9를 재서 1e-5가 가장 빨랐다(피벗 1–16번).
  4. **교차**: 내부점법의 계수에서 행을 |r_i| 순서(같으면 행 순서; 선택 알고리즘으로 앞부분만 정렬)로 보며 재직교화한 그람–슈미트로 앞 행들과 독립인 행(직교 성분 > rtol‖x_i‖, rtol은 G21과 같다)을 k개 모은다. 그 기저에서 바로 심플렉스로 간다(반복 수의 기준은 없다).
  5. **심플렉스** (Barrodale–Roberts 1973의 걸음): 꼭짓점의 기저 B에서 β는 X_Bβ = y_B의 QR 풀이와 double-double 잔차의 보정(최대 3번, G15와 같은 꼴)이다. 쌍대 변수는 X_Bᵀλ = −g, g = Σ_{i∉B} ψ_q(r_i)m_i x_i이고 λ_j ∈ [m_j(q − 1), m_j q]이면 최적(증명서)이다. g는 Ogita–Rump–Oishi의 Sum2로 더하고 행의 부호가 바뀔 때만 고치며, 끝에서는 처음부터 다시 더해 증명서를 확인한다. 기저 행 하나를 떼는 모서리 가운데 축소 비용이 가장 음수인 것(Dantzig)을 따라, 잔차가 0을 지나는 점 t_i = r_i/(x_i d)와 기울기의 증가 m_i\|x_i d\|를 모아 기울기가 0 이상이 되는 점까지 간다(정확한 직선 탐색; 고른 행이 기저에 들고 그 앞의 행들은 부호가 바뀐다). 그 점은 정규 표본(1,024개)으로 정한 문턱에서 한 번 나누어 찾고, 아니면 이분 선택이다(한 걸음에 평균 수백 개의 점을 지난다). 피벗 하나는 행을 한 번 훑는 것(잔차와 x_i d를 네 행씩 FMA로)이고 (10⁵, 10)에서 1스레드 0.6 ms다.
     - 잔차는 (k + 16)u(\|y_i\| + ‖x_i‖₁‖β‖∞) 안이면 0으로 보고 이전의 부호 쪽을 유지한다. \|x_i d\|도 같은 꼴의 기준 밑이면 0이다.
     - 퇴화(0인 잔차가 막는 걸음 0의 피벗)도 같은 규칙으로 하고, 기저와 부호들의 해시로 같은 상태가 다시 나오면 목적함수가 줄 때까지 Bland 규칙(들어갈 변수와 나갈 행 모두 가장 작은 번호)으로 한다. 그래서 순환하지 않는다.
     - 피벗은 4n + 64k + 1000번까지(안전 상한; 잰 문제에서는 교차 뒤 0–35번)이고 닿으면 `converged=False`다.
  6. **유일하지 않은 해와 꼭짓점 규칙** (G63): 최적 기저의 λ로 최소점 집합 F는 축소 비용이 양수인 변수를 0에 둔 집합, 곧 부호 조건 {r_i ≥ 0 (위쪽 행), r_i ≤ 0 (아래쪽 행), r_j = 0 (λ_j가 상자 안인 기저 행)}이다(상보 여유: 어떤 최적 쌍대 해로도 F가 이렇게 정해진다). λ_j가 경계(32u·max(m_j, \|λ_j\|) 안)에 있는 기저 행이 없으면 F는 한 점이다. 있으면 그 부호 조건 안에서 사전식 목적(`params[0]`, 그다음 `params[1]`, …; 성분은 64u‖d‖∞ 밑이면 0) 쪽 모서리만 Bland 규칙으로 따라가 사전식 최소점에 가고(F에서 나가지 않도록 첫 꺾이는 점에서 멈춘다), 길이가 있는 걸음이 한 번이라도 있었거나 사전식 최대 쪽으로 길이가 있는 걸음이 있으면 `nonunique`다. F의 사전식 최소점은 하나로 정해지므로 시작점과 상관없다. 교차가 내부점법에서 닿는 꼭짓점을 규칙으로 하면 앞 분위의 기저에서 시작하는 분위 배열(G62)이 따로 부른 것과 다른 꼭짓점을 낼 수 있다. 절편만이면 type-1 표본 분위수다.
  7. **정규 기저**: 잔차가 0인 행 가운데 행 순서로 처음의 독립인 k행을 기저로 다시 골라 β를 푼다. 같은 꼭짓점이면 길(교차, 앞 분위, 스레드 수)과 상관없이 비트까지 같다.
  8. **공분산** (G61): R `summary.rq`(quantreg 6.1)의 식이다(API §6). 다만 eps는 2⁻²⁶·max\|y_i\|다(G67; R은 y의 단위 그대로의 2⁻²⁶): y에 c를 곱하면 `params`·`bse`가 c배이고, 테스트가 c = 10⁻⁹, 10⁻³, 10³, 10⁹에서 세 공분산 모두 10⁻¹³ 안을 확인한다('iid'는 잔차의 30%가 0인 문제 포함). 'ker'에는 절대적인 상수가 없다. 'nid'의 q ± h 적합은 그 분위의 상태에서 시작하는 심플렉스다. 분위 배열이면 이 적합들(분위마다 둘)이 서로 독립이므로, 앞 분위에서 이어 가는 사슬이 그 분위들에 닿은 뒤 스레드풀의 태스크로 돌린다(태스크 안은 한 스레드, 태스크는 한 번에 메모리로 정한 수까지; 꼭짓점의 상태를 복사해 시작하므로 계산은 제자리에서 할 때와 같고 비트도 같다). (10⁵, 10)의 분위 아홉 개, 10스레드 421 → 279 ms. 샌드위치 q(1 − q)H⁻¹JH⁻¹은 N = R₀R_f⁻¹R_f⁻ᵀ(R₀은 X의, R_f는 √f X의 QR)의 q(1 − q)NᵀN으로 만든다. 'iid'의 중앙값 회귀도 같은 풀이(내부점법, 교차, 심플렉스, 규칙)다.
  - **여러 분위** (G62): 둘째 분위부터 앞 분위의 기저와 부호에서 심플렉스를 시작한다(꼭짓점은 q에 따라 바뀌지 않고 최적성만 바뀐다). 분위 간격 0.1, (10⁵, 10)에서 분위마다 73–106번의 피벗이다(속도 표).
  - **정확도와 대조** (`bench/regression/g5/accuracy.md`; 테스트 `tests/regression/test_quantreg.py`):
    - 정확성: 모든 문제에서 돌려준 기저를 유리수로 다시 풀었다(`tools/regref.py`의 `quantreg`: python-flint의 `fmpq_mat`). 정확한 꼭짓점과 잔차·목적함수, 그리고 증명서: 잔차가 0인 행이 기저뿐이면 λ_B를 바로 풀고, 더 있으면(퇴화) 같은 x의 행을 합친 정확한 선형계획으로 여유 ε(λ와 상자 끝의 거리)의 최댓값을 구한다. 최적 ⇔ ε ≥ 0, 유일 ⇔ ε > 0(상대 내부는 선형 사상과 바뀐다: ri(A·상자) = A·ri(상자)). 78개 적합에서 모두 증명서가 성립하고, 계수 8.6×10⁻¹⁷, 목적함수 1.9×10⁻¹⁶, 기저 행의 잔차 6.0×10⁻¹⁷(\|y_i\| + \|x_i\|\|β\|에 대해) 안이며, `nonunique`가 정확한 판정과 모두 같다(경계에서 10⁻¹⁴ 안의 쌍대 변수를 동률로 보는 것 포함).
    - R(4.6.2, quantreg 6.1, `tests/regression/ref/quantreg_r.json`, `tools/gen_quantreg_r.py`; 결과를 한 번 만들어 둔다): `rq(method = "br")`와 유일한 해에서 계수 1.0×10⁻¹⁴(engel q = 0.75: R 쪽의 오차, sd는 정확한 값과 5.4×10⁻¹⁷), 유일하지 않은 해(짝수 개의 중앙값, 더미 설계)에서 목적함수가 같다. 짝수 개의 중앙값에서는 둘 다 아래쪽 값이고, dummies60의 q = 0.75에서 br은 (4, 1, 3), sd는 사전식 최소점 (3, 2, 4)다. br의 "Solution may be nonunique" 경고는 유일한 최적점 두 곳(dummies60·grid40의 q = 0.5; 정확한 판정으로 유일)에서도 나온다. `fn`의 계수는 내부점법의 근사로 2×10⁻¹⁴–1.8×10⁻⁸ 다르고 목적함수는 sd보다 작지 않다. `summary.rq`(eps에 max\|y_i\|를 곱한 판; R의 함수 본문에서 그 한 줄만 바꿔 부른다. y/max\|y_i\|로 부르고 표준오차에 곱하면 y를 나눈 반올림이 'nid'를 4×10⁻¹³ 움직이고 정수 자료의 동률을 깬다): 'nid'(Hall–Sheather와 Bofinger) 7.8×10⁻¹⁴, sd의 꼭짓점에서 'ker' 4.0×10⁻¹⁴, 'iid' 1.3×10⁻¹⁴ 안이다. 'iid'의 희소도를 주는 중앙값 회귀(순서 통계량을 등간격 x에)가 유일하지 않은 세 경우(stackloss q = 0.75, tails500 q = 0.99 Bofinger, dummies60 q = 0.75)만 R의 br과 다른 꼭짓점이라 7.6×10⁻²–0.12 다르다(dummies60은 R의 기울기가 0이라 표준오차 0).
    - statsmodels(`QuantReg.fit`의 기본값, IRLS): 정확한 꼭짓점에서 계수가 1.2×10⁻⁷–9.6×10⁻²(n = 50에서 가장 크다), 목적함수가 2.2×10⁻¹⁰–1.0×10⁻⁴ 크다. 표준오차는 다른 추정량(Epanechnikov 커널과 자기 대역폭의 샌드위치)이라 비교하지 않았다.
  - **속도** (`bench/regression/g5/speed.md`; 최적화 8772322 전 측정): 표준오차 'iid'(적합과 작은 중앙값 회귀)로 R `rq.fit(method = "fn")`보다 (10⁵, 10)에서 1.5배(1스레드)·5.8배(10스레드), (10⁶, 10)에서 1.4배·6.4배 빠르고 (10³, 5)에서는 같다(0.48 ms와 0.52 ms). br보다는 (10⁵, 10)에서 50배·190배 빠르다. 기본 'nid'까지는 R의 `fn` + `summary.rq(se = "nid")`보다 2.4–2.5배·8.2–9.3배, statsmodels보다 (10⁵, 10)에서 2.5배·9.4배 빠르다. 내부점법의 반복 수는 R과 거의 같고(11–12번과 13–16번) 시간의 대부분이다. 분위 아홉 개(0.1–0.9, (10⁵, 10), `cov_type='iid'`)는 528 ms·181 ms로 분위마다 따로 부른 것(610 ms·160 ms)보다 1스레드에서 13% 빠르고 10스레드에서 13% 느리며, R fn 아홉 번(1.34 s)보다 2.5배·7.4배 빠르다.

### 3.7 비선형 최소제곱
- **나눔** (G68–G75): 반복은 C++ 커널(`csrc/regression/nls.cpp`의 `nls_solve`)이고, 잔차 r(θ) = √w (f(x, θ) − y)(쓰인 행)와 사용자 야코비안은 Python 콜백(`src/stardust/regression/_nls.py`)이다. 사용자 함수가 Python이라 루프는 GIL을 쥔 채 돌고, m·p²이 2×10⁵ 플롭 꼴(linalg의 10 µs 규칙)을 넘는 QR만 GIL을 놓는다. 다듬기, 정확한 야코비안, 공분산은 Python에서 sd 배열 연산과 선형 모형의 커널(`fit`)로 한다. 반복의 p 크기 계산을 Python으로 하면(lmpar·qrsolv의 Givens 회전) 반복마다 0.1 ms 남짓이 들어 작은 문제에서 scipy의 Fortran보다 몇 배 느려서 C++로 옮겼다. 회귀 하나만 받는다(배치 없음).
- **야코비안 하나의 m × p 일** 은 Householder QR 한 번(LAPACK `geqrf`, Qᵀr은 `ormqr`)이다: J = Q₀R₀. 그 뒤는 모두 p 크기다. lmder의 열 피벗 QR(qrfac)은 R₀의 피벗 QR과 같고(Q₀가 열의 노름과 내적을 바꾸지 않는다), TRF의 [J d; D]의 SVD는 [R₀ d; D](2p × p, `gesdd`)의 SVD이며 Uᵀ[r; 0]의 앞 p개는 Q₀ᵀr에서 온다. 기울기 Jᵀr = R₀ᵀ(Q₀ᵀr), 이차 모형의 값 ‖J_h s‖ = ‖R_h s‖도 그렇다.
- **'lm'** (G69): MINPACK `lmder`(Moré 1978)를 0부터 세는 C++로 옮겼다(lmpar, qrsolv, qrfac; factor 100, mode 1: 척도 D는 야코비안 열 노름의 누적 최댓값). 차분이면 lmdif의 `fdjac2`(epsfcn = u: h = √u·|θ_j|, 0이면 √u; 몫에는 실제로 더해진 (θ_j + h) − θ_j를 쓴다)이고 nfev가 그 호출을 센다(lmdif와 같다). 멈춤은 lmder의 info 1–8 그대로다(5: `max_nfev`; 6–8의 기계 정밀도 판은 ftol·xtol·gtol의 판으로 센다). scipy lm과의 비교(다듬기를 끈 sd, `bench/regression/g6/accuracy.md`)에서 LRE와 nfev가 거의 같다(Misra1a 시작점 2: nfev 38에서 간격 고르기와 해의 야코비안 25번을 빼면 13, scipy 13).
- **'trf'** (G69): scipy 1.18.1 `least_squares`의 `trf_bounds`('exact' 부분문제, x_scale = 1, 손실 함수 없음)를 같은 순서로 옮겼다: Coleman–Li 척도 v·dv, 신뢰 영역 Δ₀ = ‖θ₀/√v‖, `solve_lsq_trust_region`(SVD와 α의 뉴턴 10번), `select_step`(경계까지 줄인 걸음 θ배, 반사 걸음, 기울기 걸음 가운데 이차 모형이 가장 작은 것), `make_strictly_feasible(rstep=0)`, `update_tr_radius`, `check_termination`. 경계가 모두 무한이면 `trf_no_bounds`와 같은 반복이다. 2점 차분은 'lm'과 같은 h이고 위 경계를 넘으면 뒤로 뺀다(scipy의 h = √u·max(1, |θ|)와 다르다: 영역 안에서 같은 규칙을 쓰려고 lmder의 것을 따랐다).
- **유한하지 않은 시험점** (G74): 'lm'은 잔차 노름을 +inf로 둔다. 그러면 lmder 자신의 규칙(0.1·‖f₁‖ ≥ ‖f‖)이 걸음을 거절하고 Δ를 1/10로, μ를 10배로 한다. 'trf'는 scipy처럼 Δ = ‖걸음‖/4로 하고 같은 반복 안에서 다시 푼다(Δ가 0이 되면 xtol로 멈춘다). 시작점이 유한하지 않으면 `ValueError`다.
- **정확한 야코비안** (G70): 5점 중앙 차분 (f(θ − 2h) − 8f(θ − h) + 8f(θ + h) − f(θ + 2h))/(12h), h는 2의 거듭제곱(그래서 θ ± kh가 대개 정확하다). 간격을 |θ_j|에 비례해 정하면(2⁻¹⁰|θ_j|) 모수의 척도가 |θ_j|와 다를 때 틀린다: 위치 모수(가우스의 중심 b₄ = 101, 폭 23; Eckerle4의 b₃ = 451, 폭 4.6)는 h/폭이 커서 절단 오차가 3×10⁻⁸·7×10⁻⁵이다. 그래서 모수마다 처음 한 번 간격을 고른다: 2⁻⁸|θ_j|에서 4배씩 줄이며(절단 오차는 256배, 반올림 오차는 4배씩 바뀐다) 이웃한 추정의 차이가 직전 차이의 1/8보다 작은 동안 이어 가고, 마지막 추정을 쓴다. 그 간격은 다듬기와 해의 야코비안에 그대로 쓴다(국소 척도는 해 근처에서 바뀌지 않는다). 2⁻¹⁰에서 시작하면 고른 간격이 한 단계 작아져(첫 쌍은 늘 받으므로) Lanczos3의 계수가 10⁻⁹로 나빠졌다. 해석적 야코비안과 열마다 상대 1.4×10⁻¹¹ 안(NIST 27개; 고정 간격은 2⁻¹⁰ 6.9×10⁻⁵, 2⁻¹⁴ 1.1×10⁻⁹)이고, 고르는 데 모수마다 3–5단계(12–20번 호출; NIST 27개에서 잰 값)가 든다. Richardson 보외(h와 h/2의 5점 차분)는 고른 간격에서 반올림이 앞서 이득이 없었다. 경계나 유한하지 않은 값 때문에 중앙 차분을 쓸 수 없으면 한쪽 5점 차분 (−25f₀ + 48f₁ − 36f₂ + 16f₃ − 3f₄)/(12h)다.
- **다듬기** (G73): 수렴한 반복점 θ₀에서, 정확한 야코비안 J의 가우스–뉴턴 걸음 d = argmin‖r + Jd‖를 G1의 커널(`fit`: QR, 겹친 열, 증강 시스템의 반복 개선 최대 3번)로 푼다. 고정점은 Jᵀr = 0이므로, 그 정확도는 J의 정확도와 r의 반올림만이 정한다(반복의 2점 차분 J는 10⁻⁸ 정확도라 그 고정점은 대략 10⁻⁸κ²‖r‖/‖J‖만큼 틀리고, 반복은 ftol에서 멈춘다).
  - 멈춤: 걸음의 성분별 상대 크기 s_k = max_j |d_j|/|θ_j|가 2⁻⁵⁰ 밑, 또는 비 q = s_k/s_{k−1}로 내다본 나머지 s_k q/(1 − q)가 2⁻⁵⁰ 밑, 또는 s_k ≤ 2⁻⁴⁰에서 q > 1/2(반올림의 바닥: Hahn1의 jac은 4×10⁻¹⁵에서 비가 1 근처를 오간다). 걸음이 커지면(q ≥ 1) 그 걸음을 받지 않는다. 시험점이 유한하지 않거나 경계 밖이거나 제곱합이 1/16 넘게 오르면(걸음이 2⁻³⁰ 밑이 아닐 때) 멈춘다. 마지막 걸음이 2⁻⁴⁵ 넘으면 그 점의 야코비안을 다시 구한다(공분산은 돌려준 θ의 야코비안이다). 모두 10걸음까지다.
  - **뉴턴으로 바꾸기**: 가우스–뉴턴은 비 ρ((JᵀJ)⁻¹S), S = Σ c_i ∇²f_i(c = w(f − y))로 선형 수렴한다. ENSO(잔차 표준편차 2.2)와 Thurber는 ρ ≈ 0.64–0.67이라 반복이 멈춘 10⁻³–10⁻⁵에서 10걸음 뒤에도 10⁻⁶이었고, 시작점 1의 ENSO처럼 처음 걸음이 커지기도 한다(국소적으로 가우스–뉴턴이 발산하는 방향). 그래서 (a) 남은 가우스–뉴턴 걸음 수 log(나머지/2⁻⁵⁰)/log(1/q)가 남은 걸음 한도보다 많거나 그 호출 수가 S의 비용 + 두 걸음보다 많을 때, 또는 (b) 걸음이 줄지 않을 때(2⁻⁴⁰ 위에서) S를 한 번 구해 뉴턴 걸음 (I + (JᵀJ)⁻¹S)dₙ = d로 바꾼다(GLM의 G44: 피셔 득점법이 느리면 관측 정보의 뉴턴 걸음). (JᵀJ)⁻¹은 같은 커널의 공분산이고, 그래서 dₙ = 0 ⇔ d = 0이라 고정점은 그대로다. S는 φ(t) = Σ c_i f_i(t)의 2차 중앙 차분이다(대각은 야코비안의 θ ± h 값을 다시 쓰고, 비대각은 4점: 2p(p − 1)번 호출, `jac`이 있으면 2p²번). S의 상대 정확도는 10⁻⁷ 정도이고 수렴 속도만 바꾼다. 뉴턴 걸음이 실패하면(특이, 유한하지 않음, 제곱합 증가) 가우스–뉴턴으로 돌아간다. 경계에 걸린 모수는 S에서도 뺀다.
  - 경계(TRF): 반복이 끝난 점에서 scipy `find_active_constraints`(rtol = xtol)의 거리 안이고 제곱합의 기울기 Jᵀr가 바깥을 향하는 모수를 경계값으로 맞춘다(그 점에서 f가 유한하고 제곱합이 1/16 넘게 오르지 않을 때). 그 모수는 다듬기에서 고정하고(열을 빼고 푼다), 다른 모수의 걸음이 경계를 넘으면 멈춘다. scipy는 경계에서 10⁻¹⁰·max(1, |b|) 안쪽 값(0.5000000000000001 등)을 낸다.
- **다듬기와 max_nfev** (G93, 2026-11-19): 다듬기의 f 호출은 `nfev`에 들고 `max_nfev`에서 반복이 쓰고 남은 만큼만 쓴다. 걸음마다 그 비용(새 점 1번 + 정확한 야코비안: 차분이면 4p번, `jac`이면 0번)을, 뉴턴으로 바꿀 때는 S의 비용(2p²번)을 먼저 보고, 모자라면 멈춘다. 걸음이 아직 필요한데 예산이 모자라 멈추면 `converged`는 False다. 해의 공분산에 쓰는 야코비안(반복이 끝난 점의 값과 정확한 야코비안)은 결과에 늘 필요하므로 그 예산 밖이다. 그래서 `max_nfev`는 해를 찾는 호출(반복과 다듬기)의 상한이고, 공분산용 야코비안의 호출은 그 위에 더해져 `nfev`에 센다 (사용자 결정 2026-11-19). 그 수는 값 1번과, 차분이면 간격을 처음 고르는 호출이다: 잰 예에서 모수마다 12–16번(값 호출을 빼고, 지수 감쇠 p = 2 14번, 감쇠 + 상수 p = 3 13.3번, 로지스틱 p = 4 13번, 가우스 봉우리 p = 3 16번, Misra1a 12번; NIST 27개는 12–20번)이다. 후보 간격 12개 × 중앙 차분 4번 = 모수마다 48번은 중앙 차분만의 상한이고 전체의 상한이 아니다: 경계나 유한하지 않은 값 때문에 한쪽 차분으로 물러서면 후보마다 그 호출이 더해져 48번을 넘을 수 있다(후보마다 많아야 12번). `jac`이면 값 1번이다.
- **빈 입력과 자유도** (G95, G96): 쓰인 행이 없으면(모두 NaN을 뺀 뒤, 가중치가 모두 0) f를 부르지 않고 NaN 결과에 `nobs` 0이다. `df_resid` = 0이면 `scale`이 0/0이라 추론이 NaN이다(예전에는 SSR/0 = inf 또는 NaN). 완전 적합의 정확한 해는 모형을 모르므로 하지 않는다(G94). `ssr`이 정확히 0이고 `df_resid` > 0이면 `bse` 0, t·p NaN, 구간 [θ, θ]다(G109).
- **예측 구간** (G103, G104): `predict_interval`은 델타 방법이다: x_new의 행에서 해의 야코비안 J_new(사용자 `jac`, 또는 해에서 고른 간격의 5점 차분; G88의 잡음 열 규칙도 같다)로 se² = J_new C J_newᵀ. NaN이 있는 행(x_new나 그 f)은 차분에서 빼고 그 행의 구간만 NaN이다(R100). `predict`·`predict_interval`의 `x_new`는 G90·G91을 따른다.
- **입력** (2026-11-19): 음수 가중치와 inf는 NaN 정책과 상관없이 NaN이 있는 행에서도 `ValueError`다(가중치를 먼저 본다, wls·glm과 같다). `weights`의 메시지는 wls의 것이다(스칼라 `not a scalar`, 길이 `not 39`). `bounds`의 길이가 2가 아니면 `ValueError`, `jac`이 스칼라를 내면 f처럼 `ValueError`다.
- **llf·aic·bic** (G78): ols·wls의 식 그대로다(척도를 모수로 세지 않는다). 가중치가 있으면 ½Σlog w(쓰인 행)를 더한다. R `logLik.nls`(가중 잔차의 제곱합과 Σlog w)와 고정 자료 6개(가중, 가중치 0 포함)에서 10⁻¹³ 안이고, R의 `AIC`·`BIC`는 df = p + 1이라 정확히 2, log(nobs) 크다. 모수에 대해 선형인 모형을 `nls`로 풀면 `ols`·`wls`의 `llf`·`aic`·`bic`와 10⁻¹³ 안에서 같다.
- **공분산과 겹친 모수** (G72, G9): 해에서 −r을 J에 회귀하는 `fit` 한 번이 (RᵀR)⁻¹, R, 쓰인 열을 준다(G21의 판정 |R_jj| ≤ rtol·‖J_j‖). 겹친 모수는 추론이 NaN이고 `params`는 그대로다(G76; f에 넣어야 하므로). 차분 야코비안의 기본 rtol은 2⁻³⁰이다(G77): f = abx(a와 b가 곱으로만 들어간다)를 차분으로 풀면 기본 max(n, p)·u에서는 |R_bb|/‖J_b‖ ≈ 10⁻¹²라 겹친 열로 보지 않아 `bse`가 1.9×10¹⁰, 1.3×10¹¹이 되었다. 이 f = abx는 R `nls`가 QR의 tol 1e-7에서 "singular gradient" 에러를 낸다. NIST 27개의 가장 작은 |R_jj|/‖J_j‖는 Bennett5의 4.9×10⁻⁵다(R `nls`도 Bennett5는 푼다).
- **정확도** (`bench/regression/g6/accuracy.md`; 테스트 `tests/regression/test_nls.py`):
  - Arb 참값(`tools/regref.py`의 `nls`: 같은 double 자료로, 정확한 야코비안의 가우스–뉴턴을 320비트로 걸음이 2⁻¹²⁰ 밑이 될 때까지; 표준오차는 그 점의 σ̂²(JᵀJ)⁻¹): NIST 27개 × 시작점 2개에서 해석적 `jac`이면 계수의 상대오차가 최대 3.1×10⁻¹³(Lanczos1–3, MGH10: κ(J) ~ 10⁴–10⁸), 중앙값 1×10⁻¹⁵, 표준오차 5.2×10⁻¹², 중앙값 1–2×10⁻¹⁴이다. 차분이면 계수 3.0×10⁻¹⁰(Lanczos3, MGH10), 중앙값 3–6×10⁻¹³, 표준오차 9.1×10⁻¹⁰, 중앙값 1–2×10⁻¹²이다. 인증값(10진 자료, 11자리)과는 LRE가 계수 9.5(차분)·10.3(`jac`) 이상이다.
  - Lanczos1(인증 SSR 1.4×10⁻²⁵)은 잔차 10⁻¹³이 f의 반올림(u·|f| ≈ 5×10⁻¹⁶)의 수백 배밖에 안 되어 표준오차가 3–4자리다. double 자료의 참값조차 10진 자료의 인증값과 4×10⁻⁴ 다르다(G1의 Wampler2와 같은 사정).
  - 다듬기를 끄면(반복이 멈춘 점) LRE 중앙값이 6.4–6.8로 scipy lm(6.4–6.6)과 같다. ENSO·Thurber·MGH09는 3–5자리다. scipy trf·R nls·nlsLM도 중앙값 5.9–6.8이다. 표준오차는 2점 차분의 야코비안(MINPACK의 마지막 반복의 R, R의 `numericDeriv`)이라 scipy 5.0–5.3, R 6.1–6.3이다.
  - 실패(수렴하지 않거나 다른 점): sd는 시작점 1의 BoxBOD(차분: b₂가 커서 야코비안이 특이한 정류점, `jac`이면 성공), Bennett5(`max_nfev` 1,200과 300), MGH09(`jac`: 기본 `max_nfev` 400; 차분은 2,000이라 성공). scipy lm은 시작점 1에서 4개가 실패하고 MGH17은 다른 점에서 멈춘다. R nls는 시작점 1에서 11개, 2에서 2개가 실패하고(기본 `maxiter` 50과 `minFactor`), nlsLM은 시작점 1에서 7개, 2에서 2개가 다른 점(LRE 0–1)이다.
  - R(4.6.2, `tests/regression/ref/nls_r.json`, `tools/gen_nls_r.py`; R은 테스트 의존성이 아니라 결과를 한 번 만들어 둔다): `deriv()`의 해석적 기울기와 `nls.control(tol = 1e-10)`의 R `nls`와 계수 1.3×10⁻¹¹(twod는 R의 tol 1e-8이라 4.4×10⁻¹⁰, port의 경계 문제는 4.9×10⁻⁹), 표준오차 1.1×10⁻¹¹, σ̂·제곱합 2×10⁻¹⁵, 자유도는 같다(가중치 0인 행을 n에서 빼는 것 포함; R `summary.nls`의 n = Σ[w > 0]). 차이는 R의 상대 오프셋 판정의 정밀도다.
  - scipy: 경계 문제 세 개에서 sd의 제곱합이 scipy 이하이고 계수·표준오차가 2×10⁻⁵ 안(scipy trf의 멈춤 정밀도), `weights=1/sigma²`가 `curve_fit(sigma=)`과 10⁻⁶ 안이다.
- **비용과 속도** (G79, `bench/regression/g6/speed.md`): 차분이면 f 호출이 scipy lm의 5–7배다(간격 고르기 ≈ 16p, 야코비안마다 4p, 가우스–뉴턴 2–4걸음). 1스레드 시간은 scipy의 4–13배(n = 10⁶, p = 10에서 13.3 s와 0.99 s)이고, 모형의 sd 연산이 나뉘는 10스레드에서는 n = 10⁶에서 0.87–2.0배다. 해석적 `jac`이면 nfev 7–13(scipy 4–6)이고 n = 10⁶에서 scipy jac과 같거나 1.4–1.5배 빠르다. R `nls`(1스레드)보다는 n = 10⁶, p = 2·4에서 2.1배(10스레드) 빠르고 p = 10에서 1.5배 느리다.

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
- 비선형(G6): NIST StRD의 비선형 회귀 27개(`tests/regression/data/nls/`, 읽기와 모형·해석적 야코비안은 `tests/regression/nist_nls.py`; 인증값 11자리, 시작점 2개)와, 같은 double 자료의 Arb 참값(`tools/regref.py`의 `nls`: 정확한 야코비안의 가우스–뉴턴을 320비트로)이다(§3.7).
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
| G4 강건 | `rlm`(M·MM 추정), `theilslopes`, `siegelslopes` |
| G5 분위 | `quantreg`(내부점법, 정확한 해), 표준오차 |
| G6 비선형 | `nls`(`curve_fit` 꼴), LM·TRF, 다듬기, 추론 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 아래 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.

---

## 8. 마일스톤을 시작하기 전에 물을 것
- **G0**: 정했다(2026-11-18, G8).
- **G1**: 정했다(2026-11-18, G9–G30).
- **G2**: 정했다(2026-11-18, G31–G44).
- **G3**: 정했다(2026-11-18, G45–G52).
- **G4**: 정했다(2026-11-18, G53–G59).
- **G5**: 정했다(2026-11-18, G60–G63).
- **G6**: 정했다(2026-11-19, G68–G79).
