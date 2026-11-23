# sd.smooth API

> 상태: **확정** (2026-11-23; 승인(위임)). §2–§6의 공개 이름 스물아홉이 모두 구현되어 있다: 산점도 평활(§2: `lowess LowessResult loess LoessResult`), 스플라인(§3: `make_lsq_spline make_smoothing_spline make_splrep make_splprep generate_knots smoothing_spline SmoothingSplineResult pspline PSplineResult`), 핵 회귀(§4: `kernel_regression KernelRegressionResult bw_dpill`), 핵밀도추정(§5: `kde KDEResult DensityResult bw_scott bw_silverman bw_nrd0 bw_nrd bw_sj bw_isj bw_cv_ls bw_cv_ml`), Savitzky–Golay(§6: `savgol_coeffs savgol_filter`). 시그니처는 scipy 1.18.1(`scipy.interpolate`의 평활 스플라인, `scipy.stats.gaussian_kde`, `scipy.signal`의 Savitzky–Golay), statsmodels 0.15.0(`nonparametric`), scikit-learn 1.9.1(`KernelDensity`), R 4.6.2(`stats`, `KernSmooth`)의 이름과 뜻에 stardust의 규칙과 SM1–SM142의 결정을 입힌 것이다.

- 설계와 결정의 근거는 [docs/design/smooth.md](../design/smooth.md)(§1.1 SM1–SM142), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/smooth/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 함수(`name(...)`), 결과 클래스(`Name`), 결과의 메서드(`Name.method(...)`)다. 블록의 모듈 수준 이름이 `sd.smooth.__all__` 전부이고, 시그니처가 블록과 같아야 한다. 결과의 필드는 절마다의 표에 있다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양 (SM2, SM4, SM5)
- **이름공간은 평평하다**: 모든 공개 이름이 `sd.smooth`에 바로 있다. scipy가 `interpolate`·`stats`·`signal`에, statsmodels가 `nonparametric`에 나눠 둔 것이 한곳에 있다. 하위 모듈은 없다.
- **맞추는 함수는 소문자 이름이고 `<이름>Result`를 돌려준다**: `lowess` → `LowessResult`, `loess` → `LoessResult`, `smoothing_spline` → `SmoothingSplineResult`, `pspline` → `PSplineResult`, `kernel_regression` → `KernelRegressionResult`, `kde` → `KDEResult`. 회귀의 `ols` → `OLSResult`와 같은 꼴이다. 맞춘 뒤 새 점에서 평가하는 일은 결과의 메서드(`predict`, `pdf`)다.
- **scipy 이름의 스플라인 함수는 `sd.interpolate.BSpline`을 돌려준다**: `make_lsq_spline`, `make_smoothing_spline`, `make_splrep`, `make_splprep`(→ `(BSpline, u)`), `generate_knots`(→ 매듭 배열의 리스트). 보간의 `make_interp_spline`과 같은 꼴이고 평가·도함수·적분은 `BSpline`의 메서드다.
- **값만 돌려주는 함수**: `savgol_coeffs`, `savgol_filter`, `bw_*`(float, 또는 여러 차원의 (p,) 배열).
- **인자의 순서는 (x, y)다** (SM5): 여러 설명변수는 (X, y)다. scipy의 스플라인, R의 `lowess`·`smooth.spline`·`ksmooth`, 보간 영역과 같다. `[statsmodels와 다름]`: statsmodels의 `lowess(endog, exog)`와 `KernelReg(endog, exog, …)`는 (y, x)다(§8의 대응표).
- **데이터는 위치로만, 옵션은 키워드로만** 받는다. 별칭이 없다: `kernel`, `criterion`, `family`, `mode`, `method`의 문자열은 소문자 정식 이름만 받는다. 틀린 호출 모양은 부른 이름으로 시작한다(API.md §2.1): `lowess(): ...`, 결과의 메서드는 `predict(): ...`.

### §1.2 입력 (SM6, SM9)
- **받는 것**: API.md §5의 상위 함수 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `Series`; 여러 변수를 받는 `loess`·`kernel_regression`·`kde`·`bw_scott`·`bw_silverman`·`bw_cv_ls`·`bw_cv_ml`은 `DataFrame`도, 열이 변수다). 계산은 float64다. bool·정수는 float64로 바꾼다. 복소수는 Savitzky–Golay만 받는다(§6). 다른 함수의 복소수는 `TypeError`다.
- **정렬과 같은 x**: 정렬되지 않은 x와 같은 x의 반복을 받는다. 결과의 배열(`fittedvalues`, `resid`, `leverage`)은 입력의 순서다. 결과는 입력의 순서와 무관하다(같은 자료를 다른 순서로 주면 같은 값이 그 순서로 나온다).
- **`nan_policy`** (SM9): 회귀·통계와 같은 세 값이다. `'propagate'`(기본): x나 y에 NaN이 있으면 결과의 모든 값이 NaN이다. `'omit'`: x, y, 무게 가운데 NaN이 있는 점을 빼고 맞춘다. `fittedvalues`, `resid`, `leverage`의 그 자리는 NaN이고 `nobs`는 뺀 뒤의 수다. `'raise'`: `ValueError`. ±inf는 언제나 `ValueError`다. 평가점(`predict`, `pdf`)의 NaN은 그 자리의 결과가 NaN이다. Savitzky–Golay는 `nan_policy`가 없다: 배열의 원소별 필터라 NaN이 그 점이 든 창의 출력으로 번진다(§6).
- **무게** (SM6): 새 함수(`lowess`, `loess`, `smoothing_spline`, `pspline`, `kernel_regression`, `kde`)는 `weights=`이고 뜻은 Σwᵢrᵢ²(회귀의 `wls`와 같다; `kde`는 점의 확률 무게)다. scipy 이름의 스플라인 함수는 scipy의 이름 `w=`와 scipy의 뜻이다: `make_smoothing_spline`은 Σwᵢrᵢ², `make_lsq_spline`·`make_splrep`·`make_splprep`·`generate_knots`는 Σ(wᵢrᵢ)²(FITPACK의 `w`는 1/σ다). 무게는 유한한 0 이상의 수이고 양수가 하나 이상이어야 한다(`ValueError`). 무게가 0인 점은 맞춤에 들지 않지만 그 점의 `fittedvalues`는 계산한다.

### §1.3 회귀형 결과의 공통 필드 (SM4, SM7)
`LowessResult`, `LoessResult`, `SmoothingSplineResult`, `PSplineResult`, `KernelRegressionResult`가 함께 가진다.

| 필드·메서드 | 뜻 |
|---|---|
| `fittedvalues` | 자료점의 적합값 (n,)(입력의 순서) |
| `resid` | `y - fittedvalues` |
| `nobs` | 맞춤에 든 점의 수(`'omit'`이 뺀 점은 세지 않는다) |
| `df` | 유효 자유도: 모자 행렬 L의 대각합. 강건 적합은 마지막 반복의 무게로 정한 선형 평활기의 대각합이다 |
| `leverage` | L의 대각 (n,) |
| `scale` | 잔차 척도 σ̂: `loess`는 √(RSS/δ₁)(R과 같다), 그 밖은 √(RSS/(nobs − df))(0 이하의 분모는 NaN; 스플라인의 배치는 열마다) |
| 조절 모수 | `frac`·`span`·`lam`·`bw` 등 고른 값(절마다) |
| `predict(x_new, /)` | 새 점의 값(같은 정의의 평가) |
| `predict_interval(x_new, /, *, alpha=0.05)` | 평균의 점별 구간 (m, 2): ŷ(x₀) ± t·σ̂‖l(x₀)‖₂(무게가 있으면 √(Σlⱼ²/wⱼ)). 편향은 넣지 않는다. 분위수의 자유도는 `loess`에서 δ₁²/δ₂(R과 같다; δ₂가 없는 nobs > 4096은 정규 분위수), 그 밖은 nobs − df. 강건 적합은 마지막 반복의 무게를 고정한 어림이다(R과 같다) |
| `summary()` | API.md §2.3의 상자: 머리에 방법, `nobs`, `df`, 조절 모수, 기준 값, `scale`. 표는 없다 |

- 배열 필드는 쓰기 금지이고 처음 읽을 때 계산하는 필드도 있다(API.md §2.2). 결과는 `predict`를 위해 자료의 사본을 쥔다(회귀의 규칙과 같다).
- `repr`는 한 줄이다(API.md §2.3): `LowessResult(nobs=200, df=4.98321, frac=0.666667, it=3)`.
- 반복으로 고른 조절 모수(GCV·CV·AICc의 최소화)는 `status`와 `converged`가 찾기가 구간의 끝에서 끝났는지를 말한다(API.md §2.2: 에러가 아니다).

### §1.4 결정성과 정확도 (SM3)
- **같은 입력이면 스레드 수와 상관없이 같은 비트다**. 평가점 하나는 언제나 같은 커널과 같은 합 순서로 계산하고, 나누기는 크기로만 정한다. 빠른 길과 느린 길이 있는 곳(KDE의 직접 합과 트리, 정렬된 창과 KD 트리)은 두 길이 같은 정의의 같은 비트를 낸다. 구간화한 FFT 밀도(`density(binned=True)`)만 다른 값이고 사용자가 인자로 고른다. 길을 고르는 기준은 모양뿐이고 값이나 스레드 수가 아니다.
- **정확도 계약**: 조절 모수(창, 대역폭, λ, 매듭)를 주어진 것으로 둔 정확한 추정량과 견준다(docs/design/smooth.md §4). 국소 적합은 κ·u·Σ|lⱼ||yⱼ|의 배수, 스플라인 계수는 u·κ(M)·max|c|의 배수, 밀도는 상대 오차(꼬리 포함), Savitzky–Golay 계수는 u·max|h|의 배수다. 반복으로 고른 조절 모수는 정해진 절차의 결과이고 참 최소점과의 거리는 약속하지 않는다.
- **모든 호출은 정해진 시간 안에 끝난다**: 반복이 있는 곳(강건 반복, λ·대역폭 찾기, FITPACK의 근 찾기)은 입력과 무관한 상한이 있다.

### §1.5 에러와 경고
- API.md §2의 종류와 꼴이다. 경고는 내지 않는다. 다른 라이브러리가 경고하거나 조용히 넘어가는 곳은 이렇다.

  | 다른 라이브러리 | sd |
  |---|---|
  | scipy `savgol_filter`의 복소수(`ComplexWarning`, 허수부를 버린다) | 실수부와 허수부를 따로 거른 complex128 |
  | scipy `make_splrep`의 닿지 않는 `s`(경고와 그때의 스플라인) | `ValueError` |
  | scipy `splrep`의 반복 상한(경고) | 없음: 근 찾기가 이분으로 이어 가 언제나 끝난다 |
  | scipy `make_smoothing_spline`의 GCV 실패(`ValueError: Seems like the problem is ill-posed`) | 풀린다(척도와 무관한 λ 구간, 띠 QR) |
  | scipy `savgol_coeffs`의 높은 차수(조용히 틀린 계수) | 정확한 계수 |
  | statsmodels `lowess`의 NaN(조용히 빼서 결과의 줄이 준다) | `nan_policy` |
  | statsmodels의 0 나누기(`RuntimeWarning`) | 정의된 값(§2) 또는 NaN |
  | R `loess`의 랭크 부족(경고와 유사역행렬) | 낮은 차수의 적합과 `n_rank_deficient` |

- 할당 전에 `MemoryError`인 곳(API.md §2): `density`의 출력이 물리 메모리보다 클 때, 구간화 길의 격자와 FFT 배열이 물리 메모리의 1/4보다 클 때, Savitzky–Golay 계수의 작업 공간이 물리 메모리보다 클 때다. `loess`의 δ₂는 양수 무게의 점이 4096개 이하일 때만 계산하므로(134 MB 이하) 이 검사가 없다.

### §1.6 다른 라이브러리와 다른 점 (모음)
- 공통: (x, y) 순서, 결과 객체, 입력 순서의 결과, `nan_policy`, 경고 없음, 스레드와 무관한 비트.
- `lowess`: x의 단위와 무관한 퇴화 규칙(statsmodels는 절댓값 10⁻¹²로 막는다), 무게를 자르지 않는다(R은 0.001h, 0.999h에서 자른다), 반복은 언제나 `it`번(R은 일찍 멈춘다), `delta`의 기본 0(R은 범위의 0.01배).
- `loess`: 언제나 직접 계산(R의 기본은 KD 트리 꼭짓점의 혼합), 설명변수 수의 상한 없음(R은 넷), 랭크 부족은 낮은 차수의 적합, `family='symmetric'`의 통계량은 마지막 적합의 것(R은 첫 적합과 의사값).
- 스플라인: GCV가 x의 단위와 무관하다, 같은 x를 묶는다, 정렬되지 않은 x를 받는다, λ = inf를 받는다, R의 `spar`는 없다.
- 핵밀도추정: 자료의 꼴 (n, p)(scipy는 (p, n)), 수로 준 `bw`는 대역폭 자체(scipy는 표본 표준편차에 곱하는 인수), 가우스가 아닌 핵의 `bw`는 핵의 표준편차(R과 같다; sklearn·statsmodels는 지지집합의 반폭).
- Savitzky–Golay: 정확한 계수, 복소수, NaN의 번짐.

## §2. 산점도 평활
```
lowess(x, y, /, *, frac=0.6666666666666666, it=3, delta=0.0, weights=None, nan_policy='propagate')
LowessResult
LowessResult.predict(x_new, /)
LowessResult.predict_interval(x_new, /, *, alpha=0.05)
LowessResult.summary()
loess(X, y, /, *, span=0.75, degree=2, family='gaussian', iterations=4, normalize=True, weights=None, nan_policy='propagate')
LoessResult
LoessResult.predict(X_new, /)
LoessResult.predict_interval(X_new, /, *, alpha=0.05)
LoessResult.summary()
```
- **`lowess`** (기준은 statsmodels): 평가점 x₀의 창은 가장 가까운 k = min(n, max(2, int(frac·n + 10⁻¹⁰)))개의 점이고 반경 h는 k번째로 가까운 거리다. 무게는 tricube (1 − (|xⱼ − x₀|/h)³)³(거리가 h 이상이면 0)에 강건 무게와 `weights`를 곱한 것이고, 적합은 무게 붙은 국소 직선이다. `frac`은 [0, 1]의 수, `it`은 0 이상의 int, `delta`는 0 이상의 유한한 수다.
  - **결과는 자료의 함수다**: 거리가 h인 점의 무게가 0이라 같은 거리의 점 가운데 어느 것이 창에 드는지가 값을 바꾸지 않는다. 같은 x가 여러 번 나와도, 입력의 순서가 달라도 같은 값이다.
  - **퇴화**: 양수 무게의 점이 모두 같은 x이면 그 점들의 무게 평균이고, 그 밖에는 언제나 직선이다. h = 0(창의 점이 모두 x₀와 같다)이면 x₀와 같은 점들의 무게 평균이다(R과 같다). `[statsmodels와 다름]` `[R과 다름]`: statsmodels는 무게 붙은 제곱합을 절댓값 10⁻¹²로 막아 x의 단위가 값을 바꾼다: 시험 자료 random/200의 x에 10⁻⁸을 곱하면 `frac=0.1, it=0`의 적합이 0.34만큼 달라진다. R은 √(제곱합)이 범위의 0.001배 이하이면 국소 평균으로 바꾼다.
  - **강건 반복**: 잔차의 중앙 절댓값 m = median|rᵢ|에서 bisquare 무게 (1 − (r/6m)²)²(|r| ≥ 6m은 0)다. m = 0이면 잔차가 0인 점의 무게가 1이고 그 밖은 0이다(statsmodels와 같다). 양수 무게의 점이 둘보다 적은 창에서는 그 평가점의 값이 y 그대로다(statsmodels와 같다; `predict`에서는 NaN). 반복은 언제나 `it`번이다(`[R과 다름]`: R은 m이 평균 잔차의 10⁻⁷배보다 작으면 멈춘다).
  - **`delta`**: 마지막으로 적합한 점에서 x가 `delta` 안인 점은 건너뛰고(그 안의 가장 먼 점에서 다시 적합한다) 건너뛴 점은 두 적합 사이의 선형 보간이다(statsmodels·R과 같은 규칙). 기본 0은 모든 점에서 적합한다(SM8; `[R과 다름]`: R의 기본은 범위의 0.01배다). 큰 n에서 빠른 길이 필요하면 `delta`를 준다.
  - `[R과 다름]`: R은 거리가 0.001h 이하인 점의 무게를 1로, 0.999h 이상을 0으로 자른다. sd와 statsmodels는 자르지 않는다. 그래서 R과는 적합값의 크기에 비해 10⁻⁹ 크기로 다르다(시험 자료에서 가장 큰 차가 10⁻⁷ 아래다).
  - **`predict(x_new)`**: 새 점마다 자료의 가장 가까운 k개로 같은 국소 적합을 한다. 강건 무게는 맞출 때의 마지막 것이다(statsmodels의 `xvals`와 같다). 자료 범위 밖도 계산한다(국소 직선의 연장). `x_new`는 어떤 모양이든 받고 수 하나는 float를 돌려준다. `predict_interval`의 ‖l‖은 그 점의 모자 행의 √(Σlⱼ²/wⱼ)이다.
  - **무게가 0인 점**은 맞춤에 들지 않는다(회귀의 `wls`와 같다): `nobs`, `df`, 강건 반복의 중앙값에서 빠지고 `leverage`는 0이다. 그 점의 `fittedvalues`는 계산한다. `scale`은 √(Σwᵢrᵢ²/(nobs − df))이고 nobs − df ≤ 0이면 NaN이다. 양수 무게의 점이 둘보다 적어 y를 돌려준 점의 `leverage`는 1이다.
  - **결정성**: 자료를 (x, y, 무게, 색인)으로 정렬해 계산하므로 같은 점들을 다른 순서로 주면 같은 비트가 그 순서로 나온다. x에 2의 거듭제곱을 곱해도(`delta`도 함께) 같은 비트다. x의 범위가 double을 넘치면 `ValueError`다.
- **`loess`** (기준은 R): `X`는 (n,) 또는 (n, p), `degree`는 0, 1, 2, `family`는 `'gaussian'`(최소제곱) 또는 `'symmetric'`(bisquare 강건 반복; 적합이 `iterations`번). 창의 점 수는 q = floor(n·span + 10⁻⁵)이고 반경은 q번째로 가까운 거리, span > 1이면 가장 먼 거리의 √span배다(R의 계산과 같다; R의 문서는 span^(1/p)라고 적지만 R은 제곱 거리에 span을 곱한다, SM29). `normalize=True`는 p > 1일 때 변수마다 10% 절사 표준편차로 나눈다(R과 같다).
  - 거리는 유클리드 거리이고 이웃은 (거리, 색인) 순서로 정해진다. p = 1의 `loess(degree=1, span=f, family='gaussian')`와 `lowess(frac=f, it=0)`는 창의 점 수가 같으면 같은 비트다.
  - `[R과 다름]` 언제나 평가점마다 국소 적합을 직접 한다(R의 `loess.control(surface='direct', statistics='exact')`와 같은 값). R의 기본 `surface='interpolate'`는 KD 트리의 꼭짓점에서만 적합하고 그 사이를 3차 혼합으로 채우는 어림이라, 시험 자료 random/200(잡음의 표준편차 0.3)의 기본 적합(2차, span 0.75)이 0.12만큼 다르고 자료 범위 밖에서 `NA`다. `surface`, `cell`, `statistics`는 없다.
  - `[R과 다름]` 설명변수의 수에 상한이 없다(R은 넷). 국소 다항식의 항 수(2차 (p + 1)(p + 2)/2, 1차 p + 1, 0차 1)보다 q가 작으면 `ValueError`다(R도 에러다). 창의 점이 낮은 차원에 놓여 국소 계의 랭크가 모자라면 1변수는 창의 서로 다른 x가 받치는 차수로 낮추고, 여러 변수는 QR에서 앞 열들에 기대는 열(남은 노름이 max(m, t)·eps·‖열‖ 이하)을 빼고 맞춘다. 그런 평가점은 `n_rank_deficient`에 센다(R은 경고와 유사역행렬). 양수 무게의 점이 하나뿐인 창은 그 점의 y, 하나도 없으면 자료점은 자기 y, 새 점은 NaN이다.
  - `family='symmetric'`의 강건 무게는 R의 것이다: cmad = 6·median|rᵢ|(양수 무게의 점)에서 |r| ≤ 0.001·cmad이면 1, |r| > 0.999·cmad이면 0, 그 사이는 (1 − (r/cmad)²)²이고 cmad가 0에 가까우면(가장 작은 정규 수 아래) 모두 1이다. `lowess`의 무게(statsmodels: 자르지 않는다)와 다르다.
  - 통계량: `df`(tr L, R의 `trace.hat`), `one_delta`(δ₁ = tr((I − L)ᵀ(I − L))), `two_delta`(δ₂ = tr(((I − L)ᵀ(I − L))²); 처음 읽을 때 계산하고, 양수 무게의 점이 4096개를 넘으면 NaN이며 `predict_interval`의 분위수가 정규 분포다), `enp`(R의 식 δ₁ + 2·df − nobs = tr(LᵀL); `df`와 다르다), `scale` = √(Σwᵢrᵢ²/δ₁). 합은 양수 무게의 점에 대한 것이다.
  - `[R과 다름]` `family='symmetric'`의 `df`, `leverage`, δ₁, δ₂, `scale`은 마지막 적합의 평활기(마지막 강건 무게를 고정한 L)의 것이다(`lowess`와 같은 정의). R은 이것들을 첫 적합(강건 무게 없음)에서 재고 `s`를 의사값의 잔차로 잰다. 적합값과 `predict`는 R과 같다.
  - **`predict_interval`**: ŷ ± t·`scale`·√(Σlⱼ²/wⱼ)이고 t는 자유도 δ₁²/δ₂의 분위수다(R의 `predict(se=TRUE)`의 `se.fit`과 같은 양).
  - **정확도**: 국소 계의 조건수 κ가 2³⁶ 이하인 평가점에서 정확한 적합과 16·κ·u·Σ|lⱼ||yⱼ| 안이다. 좁은 덩어리 밖으로 2차를 외삽하는 평가점처럼 κ가 그보다 크면 좌표 x − x₀의 반올림이 κ²배로 커진다(R도 같다).

| `LowessResult`만의 필드 | 뜻 |
|---|---|
| `frac`, `k`, `it`, `delta` | 받은 값과 창의 점 수 |
| `robust_weights` | 마지막 강건 무게 (n,)(`it=0`이면 모두 1) |

| `LoessResult`만의 필드 | 뜻 |
|---|---|
| `span`, `q`, `degree`, `family` | 받은 값과 창의 점 수 |
| `enp`, `one_delta`, `two_delta` | R의 이름의 통계량(`enp`는 δ₁ + 2·df − nobs) |
| `robust_weights`, `n_rank_deficient` | 마지막 강건 무게, 랭크가 모자란 평가점의 수 |

## §3. 스플라인
```
make_lsq_spline(x, y, t, /, *, k=3, w=None, axis=0)
make_smoothing_spline(x, y, /, *, w=None, lam=None, axis=0)
make_splrep(x, y, /, *, w=None, xb=None, xe=None, k=3, s=0.0, t=None, nest=None, bc_type=None)
make_splprep(x, /, *, w=None, u=None, ub=None, ue=None, k=3, s=0.0, t=None, nest=None, bc_type=None)
generate_knots(x, y, /, *, w=None, xb=None, xe=None, k=3, s=0.0, nest=None, bc_type=None)
smoothing_spline(x, y, /, *, lam=None, df=None, criterion='gcv', weights=None, axis=0, nan_policy='propagate')
SmoothingSplineResult
SmoothingSplineResult.predict(x_new, /)
SmoothingSplineResult.predict_interval(x_new, /, *, alpha=0.05)
SmoothingSplineResult.summary()
pspline(x, y, /, *, n_segments=None, k=3, diff_order=2, lam=None, df=None, criterion='gcv', xb=None, xe=None, weights=None, axis=0, nan_policy='propagate')
PSplineResult
PSplineResult.predict(x_new, /)
PSplineResult.predict_interval(x_new, /, *, alpha=0.05)
PSplineResult.summary()
```
- **평활 스플라인** (`make_smoothing_spline`, `smoothing_spline`): Σᵢ wᵢ(yᵢ − f(xᵢ))² + λ∫f″(x)² dx를 가장 작게 하는 자연 3차 스플라인이다. 매듭은 scipy와 같다(서로 다른 x 전부, 양 끝이 4겹; 계수 n + 2개). **λ의 단위는 scipy의 것이다**(x의 단위의 세제곱에 비례한다). λ = 0은 자연 경계의 보간 스플라인, λ = inf는 무게 붙은 최소제곱 직선이다(`[scipy와 다름]`: scipy는 inf를 받지 않는다). `lam=None`이면 `criterion`으로 고른다.
  - **R과의 환산**: R `smooth.spline`의 `lambda`는 x를 [0, 1]로 옮기고 무게를 평균 1로 맞춘 문제의 것이라 λ_sd = λ_R·(xₙ − x₁)³·(Σw/n)다. `smooth.spline(all.knots=TRUE, lambda=λ_R)`와 같은 곡선이다. R의 `spar`는 없다(자료에 달린 다른 눈금이다; `[R과 다름]`).
  - **`df=`**: tr L(λ) = df가 되는 λ를 찾는다(2 < df ≤ 무게가 양수인 서로 다른 x의 수). `lam`과 함께 주면 `ValueError`다.
  - **`criterion`**: `'gcv'`(n·RSS_w/(n − tr L)²)와 `'cv'`(Σwᵢ((yᵢ − f̂ᵢ)/(1 − Lᵢᵢ))²/n). λ는 척도와 무관한 고정 구간에서 고른 격자 53점 뒤 Brent로 찾고, 극소가 구간의 끝이면 그 끝이 답이며 `converged=False`다(docs/design/smooth.md §3.3). `[scipy와 다름]`: scipy는 λ를 (0, n)에서 찾아서 x의 단위가 바뀌면 다른 곡선을 고르고(n = 200의 시험 자료에서 x에 0.01, 10, 100, 1000을 곱하면 적합이 1.20, 0.038, 0.39, 0.71만큼 달라진다), n = 10⁴의 고르지 않은 x에서 `ValueError`다.
  - `[scipy와 다름]` 같은 x가 여러 번 나와도 되고 정렬되어 있지 않아도 된다: 같은 x의 점들을 무게 합과 무게 평균의 한 점으로 묶는다(벌점 문제가 정확히 같다; 허용오차로 묶지 않는다). 서로 다른 x가 다섯보다 적으면 `ValueError`다.
  - `make_smoothing_spline`은 `nan_policy`가 없어 NaN과 ±inf가 `ValueError`이고, 무게는 0 이상이면 된다(`[scipy와 다름]`: scipy는 양수만 받는다).
- **`make_lsq_spline(x, y, t)`**: 주어진 매듭 `t`의 무게 붙은 최소제곱 스플라인(Σ(wᵢrᵢ)²)이다. x는 t[k] ≤ x ≤ t[−k] 안이어야 한다(scipy와 같다). `[scipy와 다름]`: 정렬되지 않은 x와 같은 x를 받는다(scipy는 줄지 않는 x만). Schoenberg–Whitney 조건은 양수 무게의 점으로 정확히 판정해 깨지면 `LinAlgError`다(scipy의 QR 길은 같은 입력에서 경고 없이 NaN 계수를 낸다). `method`와 `check_finite`는 없다.
- **`make_splrep`·`make_splprep`·`generate_knots`** (SM12): FITPACK의 뜻이다: Σ(wᵢ(yᵢ − g(xᵢ)))² ≤ s인 가장 매끄러운(k차 도함수의 뜀의 제곱합이 가장 작은) k차 스플라인. `s = 0`은 보간이고 `make_interp_spline(x, y, k=k)`와 같은 비트다(그때 `t`, `w`, `nest`를 주면 `ValueError`; scipy와 같다). 매듭 `t`를 주면 그 매듭의 벌점 적합이다. x(`make_splprep`은 `u`)는 줄지 않아야 하고(정렬하지 않는다: 매듭을 놓는 규칙이 순서에 기댄다; scipy와 같다) `k`는 1–5(FITPACK의 차수)다. `bc_type`은 `None`, `'not-a-knot'`(`None`과 같다), `'periodic'`이다. `make_splprep`은 곡선(첫 축이 좌표인 (d, m) 자료)이고 `u`가 없으면 현의 길이로 매개변수를 만들며 `(BSpline, u)`를 돌려준다. `generate_knots`는 매듭을 늘려 가는 과정의 매듭 배열을 차례로 담은 리스트다. 매듭을 놓는 규칙과 멈춤 규칙(|Σ(w r)² − s| ≤ 0.001·s)은 scipy 1.18.1의 것이다.
  - `s`가 닿지 않으면(매듭 수가 `nest`에 닿았다) `ValueError`다(`[scipy와 다름]`: scipy는 경고와 함께 그때의 스플라인을 준다).
  - 옛 계열(`splrep`, `splprep`, `UnivariateSpline`)은 없다. 두 계열은 같은 `s`에서 다른 매듭과 다른 스플라인을 낸다(시험 자료 random/200의 s = 12에서 매듭 89개와 73개, 두 곡선의 차 0.35). 대응표는 §8이다.
- **`pspline`** (SM13): 고른 간격의 매듭(`[xb, xe]`를 `n_segments`개 구간으로 나눈다; 기본 `xb`, `xe`는 x의 최솟값과 최댓값, 기본 `n_segments`는 min(n/4, 40)을 올림한 값) 위의 k차 B-스플라인에 계수의 `diff_order`차 차분 벌점 λ‖D c‖²를 준 것이다(Eilers–Marx). 매듭 수가 n과 무관해 큰 n에서 빠르다. `lam`, `df`, `criterion`은 평활 스플라인과 같다. λ = inf는 `diff_order − 1`차 다항식의 최소제곱이다.
- **배치**: `y`의 `axis` 축이 x에 맞는 축이고 나머지 축의 열마다 같은 분해로 푼다(결과의 `BSpline`이 같은 배치를 가진다). `smoothing_spline`과 `pspline`의 λ 찾기는 배치 전체의 기준 합이다(열마다 따로 고르려면 열마다 부른다). 그 결과의 `scale`은 열마다(배치 모양의 배열), `gcv`·`cv`는 열의 합, `leverage`와 `df`는 열에 공통이고, `fittedvalues`·`resid`는 `y`의 모양이다.
- 결과 `SmoothingSplineResult`·`PSplineResult`의 `predict(x_new)`는 `spline(x_new)`와 같은 비트다.
- **작업 정밀도에서 특이한 계**: 다른 점들보다 훨씬 가까이 모인 x(간격의 비가 10⁸쯤인 덩어리)에서 큰 λ의 계는 자릿수가 남지 않는다. 쌓은 행렬의 열을 맞춘 조건수 추정이 2⁴³을 넘으면 `LinAlgError`이고(`make_lsq_spline`, `pspline`도 같다), λ 찾기와 `df=`는 지렛대를 믿을 수 있는 λ만 후보로 둔다(docs/design/smooth.md §1.1의 SM58). 같은 x로 묶으면 피한다.

| `SmoothingSplineResult`만의 필드 | 뜻 |
|---|---|
| `spline` | 맞춘 `sd.interpolate.BSpline` |
| `lam` | 쓴 λ(scipy의 단위) |
| `gcv`, `cv` | 두 기준의 값(고르지 않은 기준도 계산한다) |
| `criterion`, `status`, `converged` | 고른 기준과 찾기의 상태(`lam`이나 `df`를 주면 `status`는 0, `converged`는 `True`) |

`PSplineResult`는 위와 같고 `n_segments`, `k`, `diff_order`가 더 있다.

## §4. 핵 회귀
```
kernel_regression(X, y, /, *, degree=1, bw='cv_ls', kernel='gaussian', weights=None, nan_policy='propagate')
KernelRegressionResult
KernelRegressionResult.predict(X_new, /)
KernelRegressionResult.predict_interval(X_new, /, *, alpha=0.05)
KernelRegressionResult.gradient(X_new, /)
KernelRegressionResult.summary()
bw_dpill(x, y, /)
```
- `degree=0`은 Nadaraya–Watson(국소 상수), `degree=1`은 국소 선형이다(statsmodels의 `'lc'`, `'ll'`; 기본은 statsmodels처럼 국소 선형). 핵은 변수마다의 가우스 핵의 곱이고 `bw`는 변수마다의 핵의 표준편차다. `kernel`은 `'gaussian'`만이다.
- **`bw`** (SM17): 양수(모든 변수에 같은 값), (p,) 배열, `'cv_ls'`(하나씩 빼는 최소제곱 교차검증; statsmodels의 기본), `'aicc'`(Hurvich–Simonoff–Tsai의 보정 AIC; statsmodels의 `'aic'`), `'dpill'`(p = 1이고 `degree=1`일 때의 직접 대입; `KernSmooth::dpill`의 식을 구간화 없이 계산한다; 무게를 받지 않는다). 고른 값은 `result.bw`다. 찾기는 고정된 시작점과 고정 상한의 결정적 절차다: 변수는 s = log(h/h_ref)이고 h_ref = 1.06·σ̂·nobs^(−1/(4 + p))(σ̂는 무게 붙은 표준편차, statsmodels의 시작점), 구간은 |s| ≤ log 64다. p = 1은 격자 33점 뒤 Brent(|Δs| ≤ 10⁻⁶, 60번), p ≥ 2는 Nelder–Mead(처음 단체의 변 0.5, 모든 꼭짓점이 10⁻⁵와 10⁻⁹ 안이면 멈춘다, 200·p번)다. 기준은 CV(h) = Σwᵢ(yᵢ − ŷ₋ᵢ)²/nobs와 AICc(h) = log(RSS/nobs) + (1 + tr L/nobs)/(1 − (tr L + 2)/nobs)(분모가 0 이하이면 +inf)다. 변수의 퍼짐이 0이거나 점이 둘보다 적으면 `ValueError`다.
- `[statsmodels와 다름]`: 인자가 (X, y)이고 결과가 객체다. `var_type`이 없다(연속 변수만). 국소 선형의 `'aicc'`는 국소 선형의 tr L을 쓴다(statsmodels의 `aic_hurvich`는 `reg_type='ll'`에서도 Nadaraya–Watson의 대각합을 쓴다; 국소 상수와 `cv_ls`는 같은 값이다). 무게를 받는다(statsmodels는 없다). 모든 무게가 언더플로하는 평가점(자료에서 아주 먼 점)에서도 0/0이 아니라 무게의 비가 정하는 값(가장 가까운 점들의 적합)을 낸다. 고른 대역폭은 기준이 평평하면 statsmodels와 유효숫자 몇 자리에서 다를 수 있다(찾는 절차가 다르다). sd가 고른 점의 기준 값은 statsmodels가 고른 점의 값보다 크지 않다.
- `[R과 다름]`: `ksmooth`의 `bandwidth`는 핵의 사분위수가 ±0.25·bandwidth인 눈금이고(가우스 핵의 표준편차는 0.3706506·bandwidth) 핵을 4 표준편차에서 자른다. sd의 `bw`는 표준편차이고 자르지 않는다(정확도에 닿지 않는 증명된 자르기만 한다).
- **모양** (SM91): `X`가 1차원이면 `bw`는 float이고 `predict`·`gradient`는 어떤 모양의 x_new든 받아 같은 모양을 돌려준다(수 하나는 float). `X`가 (n, p)이면 `bw`는 (p,) 배열이고(p = 1이면 float) `X_new`는 (m, p) 또는 점 하나 (p,)다. `DataFrame`은 열이 변수다.
- **`gradient(X_new)`**: `degree=1`의 국소 기울기 β_l/h_l, (m, p)다(statsmodels `fit`의 둘째 반환값). `degree=0`이면 `ValueError`다.
- **`bw_dpill(x, y)`**: Ruppert–Sheather–Wand의 국소 선형 대역폭(1변수)이다. `KernSmooth::dpill`은 격자 401점의 구간화로 어림하므로 같은 값이 아니다(시험 자료 넷에서 0.31% 안). x와 y는 유한해야 하고(NaN도 `ValueError`; `nan_policy`가 없다), 자료점이 여섯보다 적거나, 퍼짐이 0이거나, 블록의 4차 적합이 랭크를 잃거나, 추정이 0이거나 유한하지 않으면 `ValueError`다(`[R과 다름]`: R은 `NA`와 경고).

| `KernelRegressionResult`만의 필드 | 뜻 |
|---|---|
| `bw` | 쓴 대역폭 (p,)(1변수는 float) |
| `degree`, `kernel` | 받은 값 |
| `criterion` | 고른 규칙(`'cv_ls'`, `'aicc'`, `'dpill'`; 수로 준 `bw`는 `None`) |
| `criterion_value`, `status`, `converged` | 고른 기준의 값과 찾기의 상태(수로 준 `bw`와 `'dpill'`은 NaN, 0, `True`). `status`는 0(구간 안), −1·1(작은·큰 끝에서 끝났다; `converged=False`), 2(평가 상한) |

- 결과의 `fittedvalues`와 `leverage`는 하나씩 뺀 적합에서 자기 항을 되돌려 넣어 구한다: Lᵢᵢ = wᵢh′/(E + wᵢh′), ŷᵢ = (1 − Lᵢᵢ)ŷ₋ᵢ + Lᵢᵢyᵢ. `predict`를 자료점에서 부른 값과 끝자리가 다를 수 있다(둘 다 계약 안이다). 국소 선형의 국소 설계가 랭크가 모자라면(모든 무게가 한 x에 있거나 여러 변수가 한 직선 위에 있을 때) 빠진 방향의 기울기는 0이다(국소 평균). 무게가 0인 점은 맞춤에 들지 않고 `leverage`가 0이다.

## §5. 핵밀도추정
```
kde(data, /, *, bw='scott', adjust=1.0, kernel='gaussian', weights=None, bounds=None, nan_policy='propagate')
KDEResult
KDEResult.pdf(x, /)
KDEResult.logpdf(x, /)
KDEResult.cdf(x, /)
KDEResult.sample(shape=None, /, *, rng)
KDEResult.density(*, n=None, cut=3.0, lower=None, upper=None, binned=True)
KDEResult.marginal(dims, /)
DensityResult
bw_scott(data, /, *, weights=None)
bw_silverman(data, /, *, weights=None)
bw_nrd0(x, /)
bw_nrd(x, /)
bw_sj(x, /, *, method='ste', nb=1000)
bw_isj(x, /, *, n_grid=16384)
bw_cv_ls(data, /, *, nb=None)
bw_cv_ml(data, /)
```
- **자료의 꼴**: `data`는 (n,) 또는 (n, p)다(점이 행). 평가점은 (m,) 또는 (m, p)다. `[scipy와 다름]`: `gaussian_kde`는 (p, n)이다(점이 열). sd의 다른 영역과 statsmodels·sklearn이 (n, p)다.
- **추정량**: f̂(x) = Σⱼ wⱼ K_H(x − xⱼ)(무게는 합이 1이 되게 맞춘다). 가우스 핵은 공분산 H의 정규 밀도다.
- **`bw`** (SM18–SM20):
  - 규칙의 이름: `'scott'`(기본)과 `'silverman'`은 scipy의 뜻이다: H = f²·Σ̂(Σ̂는 무게 붙은 표본 공분산, 자유도 보정 있음), f = n_eff^(−1/(p + 4)) 또는 (n_eff(p + 2)/4)^(−1/(p + 4)), n_eff = (Σw)²/Σw². `'nrd0'`, `'nrd'`, `'sj'`, `'isj'`는 1차원의 규칙이다(아래의 `bw_*`). `'cv_ls'`, `'cv_ml'`은 변수마다의 대역폭(대각 H)을 교차검증으로 고른다.
  - 수: 양수 하나는 핵의 표준편차다(여러 차원이면 모든 변수에 같은 값). (p,) 배열은 변수마다의 표준편차(대각 H), (p, p) 배열은 H(대칭 양정치)다. `[scipy와 다름]`: scipy의 수는 표본 표준편차에 곱하는 인수다. R, statsmodels, sklearn처럼 수는 대역폭 그 자체이고, `adjust`가 고른 대역폭에 곱하는 인수다(R·statsmodels와 같은 이름).
  - 같은 이름의 규칙이 라이브러리마다 다른 수다(시험 자료 normal/200(표본 표준편차 1.226)의 `'scott'`: scipy 0.4249, sklearn 0.3466, statsmodels 0.3865, R `bw.nrd` 0.3895). sd의 이름마다 식이 하나다(아래).
- **`kernel`** (SM18): `'gaussian'`(모든 차원)과 1차원의 `'epanechnikov'`, `'rectangular'`, `'triangular'`, `'biweight'`, `'cosine'`이다. **`bw`는 언제나 핵의 표준편차다**(R과 같다). `[sklearn·statsmodels와 다름]`: 두 라이브러리는 가우스가 아닌 핵의 대역폭이 지지집합의 반폭이다(대역폭 1의 Epanechnikov 핵이 가운데에서 3/4다; R과 sd는 3/(4√5)). 그 대역폭 b는 sd의 `bw` = b/√5(Epanechnikov), b/√3(rectangular, sklearn `'tophat'`), b/√6(triangular, sklearn `'linear'`), b/√7(biweight)다. sklearn의 `'cosine'`은 cos(πz/2)·π/4 꼴의 다른 핵이고 sd의 `'cosine'`은 R의 (1 + cos(πz/a))/(2a)다.
- **`pdf`·`logpdf`**: 정의 그대로의 합이다(계약 안의 증명된 자르기만 한다). `pdf`의 상대 오차는 (8 + 2p + log₂n + D₀)u 이하이고(D₀는 가장 가까운 자료점까지의 대역폭 단위 제곱 거리), `logpdf`의 절대 오차는 (8 + 2p + log₂n)u·max(1, |logpdf|) 이하다. 전체 공분산 H는 백색화의 반올림이 κ_A = √cond₂(H)배로 커지므로 두 한계에 κ_A를 곱한다(scipy도 같다). `logpdf`는 밀도가 언더플로하는 꼬리에서도 상대 정확도를 갖는다(가장 큰 항을 지수 밖으로 뺀다). 대역폭이 double의 끝(2⁻¹⁰⁶⁰ 같은)이어도 정규화가 넘치지 않는다. 결과는 평가점마다 하나다(1차원은 어떤 모양이든, 여러 차원은 (m, p) 또는 점 하나 (p,)).
- **`cdf`**: 1차원의 Σwⱼ Φ((x − xⱼ)/h)다(가우스 핵; 다른 핵은 그 핵의 누적분포; `bounds`가 있으면 반사한 추정량의 누적분포). 절대 오차는 (8 + log₂n)u 이하다. 여러 차원은 `ValueError`다.
- **`sample(shape, *, rng)`**: 점을 무게대로 뽑고 핵의 잡음을 더한다. `rng`(`sd.random.Generator`)는 꼭 있어야 한다(sd.stats의 분포와 같다). `shape=None`은 하나다(1차원은 float, 여러 차원은 (p,)). 결과의 모양은 `shape + (p,)`(1차원은 `shape`)다. 무게대로 점을 뽑은 뒤 잡음을 뽑는다. `bounds`가 있으면 구간 안으로 접은 점을 돌려준다(여러 번 반사; 밀도는 경계마다 한 번 반사이므로 경계 너머 두 배 넘게 나간 질량만큼 다르다).
- **`density(*, n=None, cut=3.0, lower=None, upper=None, binned=True)`** (SM26): 고른 격자 위의 값 `DensityResult(x, density)`다. 격자는 축마다 [min − cut·h, max + cut·h](`lower`, `upper`가 끝을 바꾸고 `bounds`가 자른다)의 n점이다(R·statsmodels와 같다). `n=None`은 1차원 512, 2차원 128, 3차원 64점이다(3차원 512³은 작업 배열이 4 GB를 넘는다). `binned=True`(기본)는 선형 구간화와 FFT 합성곱의 어림이고 격자점의 절대 오차가 δ²/(8√(2π)h³) + φ(8.3)/h 이하다(δ는 격자 간격; docs/design/smooth.md §4.5). 꼬리의 상대 오차는 크다. `binned=False`는 격자점마다 `pdf`다. 여러 차원(p ≤ 3)은 `n`이 축마다의 점 수인 `DensityResult(x, density)`이고 `x`가 축마다의 격자의 튜플, `density`가 (n, …, n) 배열이다. 가우스가 아닌 핵의 구간화 길은 그 핵의 표본으로 같은 일을 한다. 여러 차원의 구간화는 축마다의 다중선형이고 한계는 Σ_l δ_l²(H⁻¹)_ll·K(0)/8 + K(0)·exp(−8.3²/2)이다. `[R과 다름]`: R은 청한 `n`을 2의 거듭제곱으로 올려 계산하고 보간한다. sd는 1차원에서 n이 512보다 작으면 512점에서 계산해 선형 보간하고(R과 같다; 한계에 보간의 δ²/(8√(2π)h³)이 더해진다) 그 밖은 청한 n점 그대로다. 작업 배열이 물리 메모리의 1/4을 넘으면 `MemoryError`다.
- **`marginal(dims)`**: 고른 차원(int 또는 int의 시퀀스)의 주변 밀도의 `KDEResult`다(가우스 핵에서 H의 부분 행렬).
- **`bounds`** (SM21): 1차원에서 `(lo, hi)`(한쪽은 ±inf)를 주면 경계 반사다: f̂(x) + f̂(2·lo − x) + f̂(2·hi − x)이고 구간 밖은 0이다. 자료가 구간 밖에 있으면 `ValueError`다. 설치된 비교 대상에는 없다(statsmodels의 `clip`은 자료를 버리는 다른 뜻이다).
- 퇴화한 자료(표본 공분산이 양정치가 아니다: 점이 낮은 차원에 놓이거나 모두 같다)에서 규칙이 대역폭 0을 내면 `LinAlgError`다(scipy와 같다). (p, p)의 `bw`는 정확히 대칭이어야 하고(`ValueError`) 양정치가 아니면 `LinAlgError`다. `'scott'`·`'silverman'` 밖의 규칙은 무게를 받지 않는다(`ValueError`; `[R과 다름]`: R `density`는 무게를 버린 대역폭을 경고와 함께 쓴다). 1차원 규칙이나 가우스가 아닌 핵을 여러 차원에 주면 `ValueError`다.
- 같은 자료를 다른 순서로 주면 같은 비트다. 여러 변수의 가우스 합은 점이 4096개 이상이고 평가점이 16개 이상이며 자르기 반경이 백색화한 자료의 축마다 범위의 1/16 이하이면 spatial의 KD 트리가 고른 후보에서, 아니면 모든 점에서 구하고 두 길은 같은 비트다(길은 모양으로만 정하고 속도만 바뀐다).
- **대역폭 함수** (SM19; 모두 자료만 받고 핵의 표준편차를 돌려준다. 여러 차원을 받는 것은 (p,)):

  | 함수 | 식 | 출처 |
  |---|---|---|
  | `bw_scott` | n_eff^(−1/(p + 4))·σ̂(변수마다) | scipy의 `'scott'`(1차원은 `gaussian_kde`의 √H) |
  | `bw_silverman` | (n_eff(p + 2)/4)^(−1/(p + 4))·σ̂ | scipy의 `'silverman'` |
  | `bw_nrd0` | 0.9·min(σ̂, IQR/1.34)·n^(−1/5)(둘 다 0이면 σ̂, |x₁|, 1의 차례) | R `bw.nrd0` |
  | `bw_nrd` | 1.06·min(σ̂, IQR/1.34)·n^(−1/5) | R `bw.nrd` |
  | `bw_sj` | Sheather–Jones, `method='ste'`(방정식 풀기) 또는 `'dpi'`(직접 대입), 쌍 합은 구간 `nb`개 | R `bw.SJ` |
  | `bw_isj` | Botev–Grotowski–Kroese의 고정점(격자 `n_grid`점의 구간화와 DCT-II) | Botev 등(2010) |
  | `bw_cv_ls` | ∫f̂² − (2/n)Σf̂₋ᵢ(xᵢ)를 가장 작게 하는 h(`nb=None`은 정확한 쌍 합, 수는 R처럼 구간 개수) | statsmodels `cv_ls`, R `bw.ucv` |
  | `bw_cv_ml` | Σlog f̂₋ᵢ(xᵢ)를 가장 크게 하는 h | statsmodels `cv_ml` |

  - 사분위수는 배열의 `quantile`(선형 보간, R의 type 7)이다. `[statsmodels와 다름]`: statsmodels의 `bw_scott`·`bw_silverman`은 1.349로 나누는 다른 식이다(§8).
  - 최소화와 근 찾기는 `bw_nrd`의 값의 [1/64, 64]배 구간(`bw_sj`는 R의 hmax = 1.144·scale·n^(−1/5) 둘레)에서 고른 격자 33점과 Brent다. 근과 최소점은 같은 방정식·기준의 Arb 값과 상대 10⁻⁸ 안이고(잰 값 1.3×10⁻¹³) 닫힌 꼴은 상대 8u 안이다.
  - `bw_sj`: R `bw.SJ`의 구간 개수(n > nb/2이면 구간화한 표)와 식이다. `'ste'`는 격자의 첫 부호 바뀜을 10⁻¹³까지 푼다. `[R과 다름]`: R의 `uniroot`는 0.1·lower의 허용오차에서 멈춰 값이 조금 다르다(시험 자료에서 0.3% 안). R의 구간 (int)(x/d)는 0에서 세므로 자료를 옮기면 값이 조금 바뀐다(R과 같다).
  - `bw_isj`: Botev의 `kde.m`처럼 [min − 범위/2, max + 범위/2]의 격자, N은 서로 다른 값의 수, t ∈ (0, 0.1]의 가장 작은 근이다(구간화는 선형). 근이 없으면 `ValueError`이고 메시지가 `bw_sj`·`bw_nrd0`을 짚는다.
  - `bw_cv_ls(nb=None)`은 statsmodels `cv_ls`의 기준을 정확한 쌍 합으로, `nb`를 주면 R `bw.ucv`의 구간 개수와 식(교차항을 n²으로 나눈다)으로 계산한다. `[statsmodels와 다름]` `[R과 다름]`: 고른 h는 격자의 가장 작은 기준이라 statsmodels의 `fmin`이나 R `bw.ucv`의 `optimize`([0.1·hmax, hmax], 허용오차 0.1·lower)가 멈춘 국소 최소와 다를 수 있다(시험 자료 weighted/40의 `nb=1000`에서 R 0.585, sd 0.796이고 sd의 기준 값이 더 작다). 기준이 구간의 끝에서 가장 작으면 그 끝(h_ref/64 또는 64·h_ref)을 돌려준다: 같은 값이 많은 이산적인 자료의 `cv_ls`는 h → 0으로 가서 아래 끝이 나온다.

| `KDEResult`의 필드 | 뜻 |
|---|---|
| `bw` | 1차원은 핵의 표준편차(float), 여러 차원은 √diag(H) (p,) |
| `covariance` | H (p, p) |
| `factor` | 규칙의 인수 f(수로 준 `bw`는 NaN) |
| `neff`, `nobs`, `ndim` | 유효 표본 수, 점의 수, 차원 |
| `kernel`, `bounds`, `weights` | 받은 값(무게는 합이 1로 맞춘 것) |

`DensityResult`는 namedtuple `(x, density)`다.

## §6. Savitzky–Golay
```
savgol_coeffs(window_length, polyorder, /, *, deriv=0, delta=1.0, pos=None, use='conv')
savgol_filter(x, window_length, polyorder, /, *, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
```
- 뜻은 scipy와 같다(SM24): 창 안의 점에 `polyorder`차 다항식을 최소제곱으로 맞춰 자리 `pos`(기본은 가운데; 짝수 창은 반 칸 어긋난 가운데)에서의 `deriv`차 도함수를 내는 선형 필터다. `use='conv'`는 뒤집은 순서(합성곱용), `'dot'`은 그대로다. `deriv > polyorder`이면 계수가 모두 0이다. `delta`는 표본 간격이다(도함수를 delta^deriv로 나눈다).
- `window_length`는 1 이상의 정수, `polyorder`와 `deriv`는 0 이상의 정수, `pos`는 `None` 또는 0 ≤ `pos` < `window_length`의 정수다(SM71; scipy의 문서와 같다. scipy가 에러 없이 받는 float의 `pos`는 `TypeError`). `delta`는 유한하고 0이 아닌 수다(음수도 된다; SM73).
- `mode`는 `'interp'`(끝의 `window_length // 2`점은 끝 창에 맞춘 다항식의 값), `'mirror'`, `'nearest'`, `'constant'`(`cval`), `'wrap'`이다(창 밖의 표본은 `scipy.ndimage`의 규칙으로 읽는다).
- 결과는 `x`와 같은 모양의 float64(복소수 입력은 complex128) 배열이다. 정수·bool·numpy의 float32는 float64로 계산한다. 축의 길이가 0이거나 다른 축의 길이가 0이면 빈 결과다. 창이 축보다 길어도 `'interp'`가 아니면 된다(창 밖은 `mode`의 규칙으로 읽는다).
- 출력 하나는 그 창과 계수만으로 정해진다: 스레드 수와 배열의 배치와 무관하게 같은 비트이고, 여러 줄을 한 번에 거른 결과의 한 줄은 그 줄만 거른 것과 같은 비트다(SM75). `'interp'`의 끝 출력은 끝 창에 맞춘 다항식의 값을 한 번 반올림한 것이고, 끝 창에 inf가 있으면 NaN이다(SM72).
- `[scipy와 다름]` **계수가 정확하다** (SM25): 정확한 유리수 계수와의 차가 계수의 최댓값의 8u 이하다(`window_length ≤ 1001`의 모든 `polyorder`). scipy는 Vandermonde 행렬의 최소제곱으로 계수를 구해 높은 차수에서 다른 필터가 된다: `savgol_coeffs(101, 8)`의 합이 1 + 8.9×10⁻⁴, `(101, 10)`·`(51, 45)`·`(301, 30)`은 계수 전체가 틀린다. 에러도 경고도 없다.
- `deriv > polyorder`이면 `savgol_filter`의 출력은 0이고, 창에 NaN이나 inf가 있으면 NaN이다(아래 NaN 규칙과 같다; SM76).
- `[scipy와 다름]` 복소수 입력은 실수부와 허수부를 따로 거른 complex128이다(scipy는 경고와 함께 허수부를 버린다). 정수와 bool은 float64다(scipy와 같다).
- `[scipy와 다름]` NaN은 어느 `mode`에서나 그 점이 든 창의 출력으로 번진다. scipy의 `'interp'`는 끝 창에 NaN이 있으면 `ValueError`이고 다른 `mode`는 번진다.
- 에러: `polyorder ≥ window_length`, `'interp'`에서 창이 축보다 길 때, `pos`가 범위 밖일 때 `ValueError`(scipy와 같다). 음수 `deriv`, 0이거나 유한하지 않은 `delta`, 0차원 `x`도 `ValueError`다(scipy는 앞의 둘에 조용히 다른 값을 낸다). 계수의 작업 공간 16·`window_length`·(`polyorder` + 1) 바이트가 물리 메모리를 넘으면 할당 전에 `MemoryError`다(SM74).

## §7. 결과 객체
| 이름 | 만드는 함수 | 이 결과만의 필드와 메서드 |
|---|---|---|
| `LowessResult` | `lowess` | `frac`, `k`, `it`, `delta`, `robust_weights` |
| `LoessResult` | `loess` | `span`, `q`, `degree`, `family`, `enp`, `one_delta`, `two_delta`, `robust_weights`, `n_rank_deficient` |
| `SmoothingSplineResult` | `smoothing_spline` | `spline`, `lam`, `gcv`, `cv`, `criterion`, `status`, `converged` |
| `PSplineResult` | `pspline` | 위와 같고 `n_segments`, `k`, `diff_order` |
| `KernelRegressionResult` | `kernel_regression` | `bw`, `degree`, `kernel`, `criterion`, `criterion_value`, `status`, `converged`, `gradient` |
| `KDEResult` | `kde` | `bw`, `covariance`, `factor`, `neff`, `nobs`, `ndim`, `kernel`, `bounds`, `weights`, `pdf logpdf cdf sample density marginal` |
| `DensityResult` | `KDEResult.density` | namedtuple `(x, density)` |

- 회귀형 결과(앞의 다섯)의 공통 필드는 §1.3이다. `KDEResult`는 분포 객체의 메서드 이름(`pdf`, `logpdf`, `cdf`, `sample`)을 쓰고(SM22) `summary()`가 없다(분포와 같다). `repr`는 한 줄이다: `KDEResult(nobs=200, ndim=1, bw=0.307569, kernel='gaussian')`.
- 결과 클래스는 사용자가 만들지 않는다(생성자는 공개 API가 아니다).

## §8. 다른 라이브러리의 이름 (대응표)
| 다른 라이브러리 | sd.smooth | 다른 점 |
|---|---|---|
| statsmodels `lowess(endog, exog, frac, it, delta, xvals, is_sorted, missing, return_sorted)` | `lowess(x, y, *, frac, it, delta)` → `LowessResult` | 인자 순서, `xvals` → `predict(x_new)`, `missing` → `nan_policy`, `is_sorted`·`return_sorted`는 없다(결과는 입력 순서). 정렬된 표는 `x`로 정렬해 만든다 |
| R `lowess(x, y, f, iter, delta)` | `lowess(x, y, *, frac=f, it=iter, delta=delta)` | `delta`의 기본(R은 범위의 0.01배) |
| R `loess(y ~ x1 + x2, span, degree, family, normalize)` | `loess(X, y, *, span, degree, family, normalize)` | 언제나 직접 계산. `predict(m, se=TRUE)` → `predict_interval` |
| scipy `make_smoothing_spline(x, y, w, lam)` | 같은 이름, 그리고 진단이 있는 `smoothing_spline` | λ = inf, 같은 x와 정렬되지 않은 x, 단위와 무관한 GCV |
| R `smooth.spline(x, y, w, df, lambda, cv, all.knots=TRUE)` | `smoothing_spline(x, y, *, lam, df, criterion)` | λ_sd = λ_R·(xₙ − x₁)³·(Σw/n), `cv=TRUE` → `criterion='cv'`, `spar`는 없다. 매듭을 줄이는 R의 기본(`all.knots=FALSE`)에 가까운 일은 `pspline` |
| `mgcv::gam(y ~ s(x, bs="ps"))` | `pspline(x, y)` | 매듭과 기본 λ 기준이 다르다(GCV는 같은 식) |
| scipy `make_lsq_spline(x, y, t, k, w, method)` | `make_lsq_spline(x, y, t, *, k, w)` | `method`, `check_finite`는 없다 |
| scipy `make_splrep`, `make_splprep`, `generate_knots` | 같은 이름 | 닿지 않는 `s`는 `ValueError` |
| scipy `splrep(x, y, w, s)`·`splev` | `make_splrep(x, y, w=w, s=s)`와 결과의 호출 | 같은 `s`에서 매듭이 다르다(새 계열의 규칙). `splrep`의 무게가 있을 때의 기본 s = m − √(2m)은 주어야 한다 |
| scipy `splprep` | `make_splprep` | 위와 같다 |
| scipy `UnivariateSpline(x, y, w, k, s)` | `make_splrep(x, y, w=w, k=k, s=s)` | 기본 `s`가 m이 아니라 0이다 |
| scipy `LSQUnivariateSpline(x, y, t_interior, w, k)` | `make_lsq_spline(x, y, t, w=w, k=k)` | `t`는 양 끝을 k + 1겹으로 붙인 전체 매듭이다 |
| scipy `SmoothBivariateSpline`, `LSQBivariateSpline`, `SmoothSphereBivariateSpline`, `bisplrep` | 없음(나중; SM23) | 흩어진 점의 평활은 `sd.interpolate.RBFInterpolator(smoothing=)` |
| statsmodels `KernelReg(endog, exog, var_type, reg_type, bw)` | `kernel_regression(X, y, *, degree, bw)` | 인자 순서, `reg_type='lc'·'ll'` → `degree=0·1`, `bw='aic'` → `'aicc'`, `var_type`은 없다(연속만), `fit(X_new)` → `predict`·`gradient` |
| R `ksmooth(x, y, kernel='normal', bandwidth)` | `kernel_regression(x, y, degree=0, bw=0.3706506 * bandwidth)` | sd는 핵을 자르지 않는다. `kernel='box'`는 없다 |
| `KernSmooth::locpoly`, `dpill` | `kernel_regression(degree=1)`, `bw_dpill` | 구간화하지 않는다 |
| scipy `gaussian_kde(dataset, bw_method, weights)` | `kde(data, *, bw, weights)` | 자료가 (n, p), 수로 준 `bw`는 대역폭(scipy의 수 f는 1차원에서 `bw=f·σ̂`, 여러 차원에서 `bw=f²·Σ̂`의 (p, p) 행렬), `evaluate`·`__call__` → `pdf`, `logpdf` → `logpdf`, `resample` → `sample(rng=)`, `integrate_box_1d` → `cdf`의 차, `set_bandwidth` → 다시 `kde`. `integrate_box`·`integrate_gaussian`·`integrate_kde`는 없다(나중) |
| statsmodels `KDEUnivariate(x).fit(kernel, bw, fft, gridsize, adjust, cut, clip)` | `kde(x, *, bw, kernel, adjust).density(n=gridsize, cut=cut, binned=fft)` | 가우스가 아닌 핵의 `bw`는 반폭(위), `clip`은 없다(`bounds`가 다른 뜻) |
| statsmodels `KDEMultivariate(data, var_type, bw)` | `kde(data, *, bw)` | `bw='normal_reference'`는 1.06·σ̂·n^(−1/(p + 4))(σ̂는 n으로 나눈 것)이라 sd의 규칙에 없다(수로 준다), `'cv_ml'`·`'cv_ls'`는 같은 이름 |
| sklearn `KernelDensity(bandwidth, kernel).fit(X).score_samples(Y)` | `kde(X, *, bw, kernel).logpdf(Y)` | 기본 대역폭(sklearn은 1.0), 가우스가 아닌 핵의 눈금(위), `'tophat'` → `'rectangular'`, `'linear'` → `'triangular'`, `'exponential'`은 없다, `atol`·`rtol`의 어림은 없다(언제나 계약 안의 합) |
| R `density(x, bw, adjust, kernel, weights, n, from, to, cut)` | `kde(x, *, bw, adjust, kernel, weights).density(n=n, lower=from, upper=to, cut=cut)` | 기본 대역폭(R은 `nrd0`, sd는 `'scott'`), `n`의 올림(위) |
| R `bw.nrd0`, `bw.nrd`, `bw.SJ`, `bw.ucv` | `bw_nrd0`, `bw_nrd`, `bw_sj`, `bw_cv_ls(nb=)` | `bw.bcv`와 `KernSmooth::dpik`는 없다 |
| statsmodels `bw_scott`, `bw_silverman`, `bw_normal_reference` | 없음 | 식이 다르다(1.059·min(σ̂, IQR/1.349)·n^(−1/5) 등): 수로 준다 |
| scipy `savgol_coeffs`, `savgol_filter` | 같은 이름 | 정확한 계수, 복소수, NaN |
