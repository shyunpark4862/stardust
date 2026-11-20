# sd.htest API

> 상태: **초안** (H0, 2026-11-20; H1 구현 2026-11-20). 공개된 검정은 H1의 아홉 함수(§2)다. 나머지는 모듈 `sd.htest`, 결과 객체의 바탕, 공통 입력 층(H0) 위에 마일스톤마다 더한다. 아래 시그니처는 scipy 1.18.1의 `scipy.stats`와 statsmodels 0.15.0의 `statsmodels.stats`의 이름과 인자 이름(H3, H5)에 stardust의 규칙과 H6–H15의 결정을 입힌 안이다. 함수는 마일스톤 H1–H9에서 하나씩 공개된다(docs/design/htest.md §8). 마일스톤에서 정할 것은 그 자리에 "(H<n>에서 정한다)"로 적고 §12에 추천과 함께 모았다. sd.regression과 같은 값을 계산하는 곳은 §13에 있다.

- 설계와 결정의 근거는 [docs/design/htest.md](../design/htest.md)(§1.1 H1–H33), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/htest/test_api_surface.py`가 읽는다. 블록의 이름이 계획한 공개 이름 전부이고, 공개된 함수는 시그니처가 블록과 같아야 한다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- 함수 이름과 인자 이름은 scipy의 것이다. statsmodels에만 있는 함수(다중 비교 보정, 비율, McNemar, Cochran Q)는 statsmodels의 이름이다(H5). 같은 개념이 scipy에 있으면 scipy의 이름을 쓴다(`alternative`의 `'less'`·`'greater'`). 다른 영역에 이미 있는 개념은 그 영역의 꼴이 먼저다: 신뢰구간은 regression의 `conf_int(*, alpha=0.05)`(H16).
- **데이터는 위치로만, 옵션은 키워드로만** 받는다(H3). 시그니처의 꼴은 sd.stats 기술통계와 같다: 데이터, `/`, `*`, `axis=None`, 그 검정의 옵션(scipy의 순서), `nan_policy='propagate'`, `keepdims=False`. 여러 표본을 받는 함수(`*samples`)는 표본 뒤가 모두 키워드다.
- **별칭이 없다**: scipy 결과의 `correlation`(= `statistic`), MGC의 `stat`, statsmodels의 줄임 방법 이름(`'hs'`, `'b'`), 다른 대소문자의 방법 이름을 받지 않는다. scipy의 `random_state`·`seed`는 없고 무작위는 `rng`뿐이다.
- 자유도는 결과 필드의 이름이 정해져 있다(H10, H19): 카이제곱·t의 자유도는 `df`(`ttest_*`, `chi2_contingency`의 scipy `dof`, `cochrans_q`), F 꼴 검정의 두 자유도는 `df_num`·`df_denom`이다. `chisquare`·`power_divergence`의 인자 `ddof`는 이름이 그대로다: 기대 빈도를 맞추는 데 쓴 모수의 수라 sd.stats의 `correction`(분산의 자유도에서 빼는 수)과 뜻이 다르다.
- 반복 상한은 `max_iter`다(`multipletests`의 statsmodels `maxiter`; optimize의 O14와 같다).
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1): `ttest_ind(): unexpected keyword argument 'equal_variance'`. 결과의 메서드(`conf_int()`)는 작은 메서드라 Python의 꼴이다.

### §1.2 데이터, `axis`, 배치
- **데이터**(표본, 표, 개수의 배열)는 sd.stats 기술통계와 같은 입력이다(API.md §5, docs/api/stats.md §4): `sd.ndarray`, 수의 시퀀스, 실수 스칼라(원소 하나), numpy 배열, frame의 `Series`·`DataFrame`. float64로 바꿔 계산하고(bool·int64도), complex는 `TypeError`다. frame의 결측은 NaN이 되어 `nan_policy`를 따른다. 메시지도 sd.stats의 것이다: `ttest_ind(): 'a' must be an sd.ndarray, a sequence of numbers or a frame Series or DataFrame, not dict`.
- **`axis`**(H9): 기본값은 None이다(sd.stats 기술통계와 같다; scipy는 대개 0). None이면 표본마다 전체를 펼쳐 검정 하나다. int나 int의 tuple·리스트면 그 축을 따라 표본을 읽고 나머지 축이 배치다. 중복된 축은 `ttest_ind(): repeated axis`, 범위 밖은 `ttest_ind(): axis 2 is out of bounds for an array of dimension 2`(`ValueError`)다.
  - 여러 표본은 표본 축 밖의 축끼리 브로드캐스트하고(먼저 앞에 길이 1의 축을 붙여 차원 수를 맞춘다; scipy와 같다), 표본 축의 길이는 달라도 된다. 맞지 않으면 `ttest_ind(): shapes could not be broadcast together outside the sample axes: (3, 5), (2, 7)`(`ValueError`)다.
  - 짝지은 표본(`ttest_rel`, `wilcoxon(x, y)`, 상관)은 표본 축의 길이도 같아야 한다: `ttest_rel(): the paired samples must agree in size, not 5 and 6`(axis=None), `... in length along the sample axes, not 5 and 6`.
  - `keepdims=True`면 결과에 표본 축이 길이 1로 남는다.
  - 배치의 줄(lane)은 서로 독립인 검정이다. 줄 하나의 계산은 배치 안에서나 혼자서나 같은 비트다.
- **표를 받는 함수**(분할표, `page_trend_test`의 `data`, `cochrans_q`의 `x`, `mcnemar`의 `table`)는 `axis`가 없다. 마지막 축들이 표이고 앞의 축이 배치다(H9): `chi2_contingency`에 (10, 2, 3) 배열을 주면 2 × 3 표 10개의 검정이다.
- **개수**: 표와 비율의 `count`·`nobs`는 음이 아닌 정수 값이어야 한다(정수 값의 float는 받는다). 아니면 `fisher_exact(): 'table' must hold non-negative integer counts`(`ValueError`)다. 스칼라 개수(`binomtest`의 `k`·`n`)는 int 인자의 규칙(API.md §2: `operator.index`, bool은 아니다)이고 scipy도 정수만 받는다.

### §1.3 `nan_policy`, 무한대, 퇴화한 입력
- **`nan_policy`**: `'propagate'`(NaN이 있는 줄의 결과가 NaN), `'omit'`(줄마다 NaN을 빼고 남은 크기로; 짝지은 표본은 쌍을 뺀다), `'raise'`(NaN이 있으면 `ValueError`: `ttest_ind(): the input contains NaN (nan_policy='raise')`). 메시지는 sd.stats의 꼴이다(T41): `ttest_ind(): 'nan_policy' must be 'propagate', 'omit' or 'raise', not 'skip'`.
- **무한대**: 순위 검정은 ±inf를 순서대로 다룬다. 적률을 쓰는 검정은 sd.stats의 규칙이다(합이 유한하지 않으면 보정 없이, inf − inf는 NaN).
- **퇴화한 입력**(H11): 관측이 너무 적은 줄(t 검정의 n < 2, `skewtest`의 n < 8, `kurtosistest`의 n < 5, `shapiro`의 n < 3 등), 분산이 0인 줄, 표본이 빈 줄의 결과는 NaN이다. 경고는 없다(API.md §2). 배치의 다른 줄은 그대로 계산한다. scipy가 inf와 0을 내는 곳(분산 0의 t 검정: `ttest_1samp([1, 1, 1], 0)`은 scipy에서 statistic inf, pvalue 0)도 NaN이다(H24, API.md §2.2의 "정할 수 없는 값은 NaN"; `[scipy와 다름]`). 분산 0의 판정은 sd.stats `skew`의 규칙이다(m2 ≤ (ε·평균)², H20). 검정마다의 최소 크기는 그 마일스톤의 절에 적는다.

### §1.4 옵션
- **`alternative`**: `'two-sided'`, `'less'`, `'greater'`만이다. 메시지는 `mannwhitneyu(): 'alternative' must be 'two-sided', 'less' or 'greater', not 'larger'`다.
- **`method`와 `'auto'`**(H12): scipy가 `method='auto'`를 두는 검정(`ks_1samp ks_2samp mannwhitneyu wilcoxon ansari kendalltau cramervonmises_2samp page_trend_test`)은 scipy와 같은 규칙으로 방법을 고르고, 결과의 `method` 필드에 실제로 쓴 방법(`'exact'`, `'asymptotic'`, `'approx'` 등 그 검정의 이름)을 적는다. scipy는 `page_trend_test`만 적는다. 배치에서는 줄마다 고를 수 있다(배치의 `method`의 꼴은 H3에서 정한다).
- **정확한 방법의 상한**(H13): `method='exact'`를 고른 호출의 작업량이 그 검정의 문서화한 상한을 넘으면 `ValueError`다. 메시지는 다른 길을 알려 준다: `ks_2samp(): method='exact' needs about 4.1e+12 operations for this input, more than its cap of 1e+10; use method='asymptotic'`(재표본이 있는 검정은 `or a PermutationMethod`). `'auto'`는 상한 안에서만 정확을 고른다. 그래서 모든 호출이 정해진 시간 안에 끝난다. 상한의 값은 검정마다 그 마일스톤에서 정한다.
- **`rng`**: 무작위를 쓰는 계산(재표본, 몬테카를로 p값)은 `rng=`에 `sd.random.Generator`를 꼭 받는다. 없거나 다른 것이면 `permutation_test(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType`(`TypeError`; optimize의 O31과 같다). 시드 없이 무작위로 도는 기본값은 없다(sd.random의 R3). 같은 `rng` 상태와 같은 입력이면 스레드 수와 상관없이 같은 비트다.
- 문자열 옵션은 정확한 이름만 받는다(대소문자를 바꾸지 않는다). 수 옵션은 int와 float(numpy 정수도 int)다.

### §1.5 결과 객체 (H6, H7, H10, H12, H16, H18, H19, H21)
- 검정마다 결과 클래스가 하나이고 이름은 `<함수>Result`다: `ttest_ind` → `TtestIndResult`, `pearsonr` → `PearsonrResult`, `chi2_contingency` → `Chi2ContingencyResult`(H6). 공통 바탕 클래스를 쓴다. 클래스는 내보내지 않는다(`sd.htest`의 공개 이름은 함수뿐이다). `isinstance`가 아니라 필드로 읽는다.
- **예외: 두 영역이 같은 검정을 돌려주는 클래스**(H18). 같은 검정은 같은 형이다: Jarque–Bera의 결과 클래스는 htest의 것이고, `sd.regression`이 내보내는 공개 이름 `JarqueBeraResult`가 그 클래스를 가리킨다(`sd.htest.jarque_bera(x)`와 `r.jarque_bera()`가 같은 형을 돌려준다; 지금의 namedtuple은 H2에서 이 클래스로 바뀐다). 그 밖의 결과는 H6대로 내보내지 않는다.
- **필드는 읽기 전용 속성이다**(regression의 G98, optimize의 `OptimizeResult`와 같다): 대입하면 `AttributeError`다. 배열 필드는 쓰기 금지 배열이고 읽을 때 복사하지 않는다(H21, API.md §2.2, regression의 G120; 고치려면 `.copy()`). 리스트인 설정 필드는 읽을 때마다 새 리스트다(G113). 그 호출에서 주지 않는 필드(근사가 아닐 때의 `wilcoxon`의 `zstatistic`)는 `AttributeError`라 `hasattr()`로 안다.
- 검정 하나면 필드는 Python 수다(`statistic`·`pvalue`·`df`는 float, 개수는 int, `method`는 str). 배치면 배치 모양의 sd 배열이다.
- **튜플 풀기**(H7): scipy의 결과가 풀리는 검정만 scipy와 같은 개수와 순서로 풀린다(`stat, p = sd.htest.ttest_ind(a, b)`; `chi2_contingency`는 넷, `median_test`도 넷). 인덱스(`r[0]`)와 `len(r)`도 그 튜플의 것이다. scipy의 결과가 풀리지 않는 검정(`binomtest`, `cramervonmises`, `alexandergovern`, `barnard_exact`, `somersd`, `page_trend_test`, 재표본 결과 등)은 풀리지 않는다(`TypeError: cannot unpack non-iterable BinomtestResult object`). statsmodels의 튜플을 돌려주는 함수는 §8·§9에 적는다.
- **신뢰구간**(H16): regression의 꼴인 메서드 `r.conf_int(alpha=0.05)`다(G99; scipy의 `confidence_interval(confidence_level=0.95)`가 아니다). `alpha`는 키워드로만 받는 유의수준이고 구간은 1 − alpha다. 돌려주는 것은 아래 끝과 위 끝의 (…, 2) float64 배열이다: 검정 하나면 (2,), 배치면 (배치 모양, 2), `tukey_hsd`처럼 비교가 여럿이면 (k, k, 2). 부를 때마다 새 배열이다(regression의 `conf_int()`처럼 쓸 수 있다, G120). `alpha`가 (0, 1) 밖이면 `conf_int(): 'alpha' must be between 0 and 1 (exclusive)`(`ValueError`), 수가 아니면 `conf_int(): 'alpha' must be an int or a float, not str`(`TypeError`)다(regression과 같은 메시지). 구간의 방향은 그 검정의 `alternative`를 따른다(scipy와 같다: 한쪽이면 한 끝이 ±inf).
- `repr`는 한 줄 요약이다: `TtestIndResult(statistic=2.13, pvalue=0.0412, df=27)`, 배치는 `TtestIndResult(batch shape (10,))`. `summary()`는 없다.

### §1.6 에러
- API.md §2의 종류와 꼴이다. 공개 함수의 에러는 `ttest_ind(): `로 시작하고, 인자는 작은따옴표 안에 쓴다. 경고는 내지 않는다. scipy가 경고(`SmallSampleWarning`, `ConstantInputWarning`, 근사가 부정확하다는 경고)를 내는 곳은 NaN이거나(퇴화한 입력) 문서다(근사의 품질).

### §1.7 scipy·statsmodels와 다른 점 (모든 검정에 걸친 것)
- `axis`의 기본값 None(scipy는 대개 0), 데이터는 위치로만, 옵션은 키워드로만, 별칭 없음, `random_state` 없음과 `rng` 필수, 결과 클래스의 이름(`TtestIndResult`; scipy는 `TtestResult`·`SignificanceResult`를 나눠 쓴다), `dof` → `df`, 모든 `'auto'` 결과의 `method` 필드, 퇴화한 입력의 NaN과 경고 없음, `method='exact'`의 상한. 값의 차이는 반올림 크기이고 정확도 계약 안이다(API.md §4 규칙 3, docs/design/htest.md §4).
- 신뢰구간은 regression의 `conf_int(*, alpha=0.05)`이고 scipy의 `confidence_interval(confidence_level=0.95)`가 아니다(H16, `[scipy와 다름]`: 영역 사이의 통일이 먼저다). 돌려주는 것도 `ConfidenceInterval(low, high)`가 아니라 (…, 2) 배열이다. 대응은 §11의 표다.
- H1에서 scipy와 다른 것(§2): `df`는 언제나 float(H22), `ttest_ind_from_stats`의 `df`·`conf_int()`(H23), `f_oneway`의 `df_num`·`df_denom`(H27), `levene`의 `df_num`·`df_denom`과 `bartlett`·`fligner`·`alexandergovern`의 `df`(H29), 표준오차 0과 상수 집단의 NaN(H24, H11), Welch의 nᵢ < 2에서 `df` NaN, `alexandergovern`의 n < 2에서 에러 대신 NaN, 표본이 둘 미만인 `bartlett`·`levene`·`fligner`의 `TypeError`, `ttest_ind`의 `method`는 None만(H9까지).

## §2. 모수: 평균과 분산 (H1)
```
ttest_1samp(a, popmean, /, *, axis=None, alternative='two-sided', nan_policy='propagate', keepdims=False)
ttest_rel(a, b, /, *, axis=None, alternative='two-sided', nan_policy='propagate', keepdims=False)
ttest_ind(a, b, /, *, axis=None, equal_var=True, alternative='two-sided', trim=0, method=None, nan_policy='propagate', keepdims=False)
ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, /, *, equal_var=True, alternative='two-sided')
f_oneway(*samples, axis=None, equal_var=True, nan_policy='propagate', keepdims=False)
alexandergovern(*samples, axis=None, nan_policy='propagate', keepdims=False)
bartlett(*samples, axis=None, nan_policy='propagate', keepdims=False)
levene(*samples, axis=None, center='median', proportiontocut=0.05, nan_policy='propagate', keepdims=False)
fligner(*samples, axis=None, center='median', proportiontocut=0.05, nan_policy='propagate', keepdims=False)
```
- `ttest_1samp`·`ttest_rel`·`ttest_ind`: 결과 `statistic pvalue df`, 둘로 풀린다, `conf_int(alpha=0.05)`(평균, 평균의 차, 절삭 평균의 차의 구간; scipy의 `confidence_interval`과 같은 정의: 추정값 ∓ t 분위수 × 표준오차, 한쪽 대립가설은 한 끝이 ±inf). `df`는 언제나 float다(H22).
- `ttest_1samp`의 `popmean`은 실수이거나 결과의 모양(배치 모양)으로 브로드캐스트되는 배열이다. scipy처럼 표본 축을 길이 1로 남긴 모양도 받는다. 아니면 `ttest_1samp(): 'popmean' cannot be broadcast to the result's shape (4, 5)`(`ValueError`)다.
- `ttest_rel`: 차 a − b의 한 표본 검정이다. 차는 double로 반올림하지 않고 double-double로 든다(3·10⁹와 10⁹의 짝에서 scipy·R은 t가 1.1e-9 틀린다). `nan_policy='omit'`은 NaN이 있는 쌍을 뺀다.
- `ttest_ind`: `equal_var=True`는 Student(원소 하나의 표본은 분산 0; scipy와 같다), `False`는 Welch, `trim`(0 이상 0.5 미만, 아니면 `ttest_ind(): 'trim' must be >= 0 and < 0.5`)은 Yuen의 절삭 t다(scipy의 g = int(n·trim), 윈저화 분산). `method`는 지금 None만 받고(`ttest_ind(): 'method' must be None, not str`, `TypeError`), 재표본 객체(`PermutationMethod`, `MonteCarloMethod`)는 H9에서 더한다.
- `ttest_ind_from_stats`: 요약값(평균, 보정 1의 표준편차, 크기)에서의 t 검정이다. 입력은 실수나 서로 브로드캐스트되는 배열이고, 모두 실수면 결과의 필드가 Python float다. 결과는 `ttest_ind`과 같은 `statistic pvalue df`와 `conf_int(alpha=0.05)`이다(H23, `[scipy와 다름]`: scipy의 결과는 `statistic pvalue`뿐이다). 둘로 풀린다. 브로드캐스트되지 않으면 `ttest_ind_from_stats(): the arguments could not be broadcast together: (2,), (3,)`(`ValueError`)다.
- `f_oneway`: 결과 `statistic pvalue df_num df_denom`(H27, H19; scipy는 `statistic pvalue`), 둘로 풀린다. `equal_var=False`는 Welch의 분산분석(H26; R `oneway.test(var.equal = FALSE)`)이고 `df_denom`이 정수가 아니다. 집단 더미의 OLS와 `sd.regression`의 `anova` 표(항 하나)와 같은 F·p값이지만 비트까지 같지는 않다: `anova`는 설계 행렬만 받아 집단 레이블이 없어 적합의 QR에서 계산하므로, 두 값은 정확도 계약 안에서 같다(p값은 같은 `F(df_num, df_denom).ccdf` 호출; §13).
- `ttest_ind`(Student)의 t·p는 집단 더미 하나의 OLS에서 그 계수의 t·p와 같은 값이다(부호는 더미의 방향; §13).
- `alexandergovern`: 결과 `statistic pvalue df`(`df`는 k − 1, H29; scipy에는 없다), 풀리지 않는다(scipy와 같다).
- `bartlett`·`levene`·`fligner`: 결과는 `bartlett`·`fligner`가 `statistic pvalue df`(k − 1), `levene`이 `statistic pvalue df_num df_denom`(k − 1, N − k)이다(H29, H19; scipy의 결과에는 자유도가 없다). 둘로 풀리고(scipy와 같다, H7) 자유도는 튜플에 들지 않는다. `center`는 `'median'`, `'mean'`, `'trimmed'`이고 `proportiontocut`(0 이상 0.5 미만)은 `'trimmed'`일 때 쓴다(scipy `trim_mean`의 g = int(p·n)). 중앙값과 평균은 double-double이라 짝수 n의 두 가운데 값의 편차가 정확히 같은 동점으로 남는다(scipy·R은 중앙값을 double로 반올림해 동점이 갈린다: 정의는 같고, `fligner`의 p값이 반올림 때문에 6.4e-9–2.9e-7 다를 수 있다).
- 표본이 둘 미만이면 다섯 다표본 함수 모두 `f_oneway(): at least two samples are required, not 1`(`TypeError`, H30)다. scipy는 `f_oneway`·`alexandergovern`이 `TypeError`, `bartlett`·`levene`·`fligner`가 `ValueError`다. 표본 인자의 이름은 메시지에서 `'samples[1]'`이다.
- 실수 스칼라 표본은 원소 하나의 표본이다(H25, sd.stats의 D-30).
- **퇴화한 입력**(H11, H24, H20): 모든 필드가 NaN인 곳: 빈 표본, `nan_policy='propagate'`의 NaN(scipy와 같다), 유한하지 않은 값이 평균이나 제곱합을 무한하게 만드는 줄. t·p만 NaN인 곳: 한 표본 검정의 n < 2(`df` = n − 1), Student의 n₁ + n₂ − 2 ≤ 0, 표준오차 0(관련된 표본이 모두 상수; `conf_int()`는 폭 0의 (추정값, 추정값)). Welch에서 nᵢ < 2면 `df`도 NaN이다(H31; scipy는 1). 통계량·p가 NaN인 곳: `f_oneway`의 빈 집단, N ≤ k, 모든 집단이 상수(SSW = 0; scipy는 inf와 0 또는 NaN), Welch·`alexandergovern`·`bartlett`의 nᵢ < 2나 상수 집단(H28)(scipy `bartlett`은 inf와 0, `alexandergovern`의 n < 2는 `ValueError`), `levene`의 빈 집단이나 모든 집단의 편차가 상수, `fligner`의 빈 집단이나 모든 점수가 같은 줄. `[scipy와 다름]`인 것은 표준오차 0과 상수 집단의 NaN, Welch의 `df`, `alexandergovern`의 에러 대신 NaN이다.
- 값이 아주 크거나 작아도(가장 큰 크기가 2⁵⁰⁰ 이상이거나 2⁻⁴⁰⁰ 아래) 통계량은 그대로다: 그런 줄은 2의 거듭제곱 단위로 계산한다(docs/design/htest.md §3.2).

## §3. 적합도 1: 분포를 정한 검정 (H2)
```
chisquare(f_obs, f_exp=None, /, *, axis=None, ddof=0, sum_check=True, nan_policy='propagate', keepdims=False)
power_divergence(f_obs, f_exp=None, /, *, axis=None, ddof=0, lambda_=None, nan_policy='propagate', keepdims=False)
skewtest(a, /, *, axis=None, alternative='two-sided', nan_policy='propagate', keepdims=False)
kurtosistest(a, /, *, axis=None, alternative='two-sided', nan_policy='propagate', keepdims=False)
normaltest(a, /, *, axis=None, nan_policy='propagate', keepdims=False)
jarque_bera(x, /, *, axis=None, nan_policy='propagate', keepdims=False)
shapiro(x, /, *, axis=None, nan_policy='propagate', keepdims=False)
anderson(x, /, *, axis=None, dist=None, method='interpolate', nan_policy='propagate', keepdims=False)
ks_1samp(x, cdf, /, *, axis=None, alternative='two-sided', method='auto', nan_policy='propagate', keepdims=False)
kstest(rvs, cdf, /, *, axis=None, alternative='two-sided', method='auto', nan_policy='propagate', keepdims=False)   # 둘지 H2에서 정한다
cramervonmises(rvs, cdf, /, *, axis=None, nan_policy='propagate', keepdims=False)
```
- 결과는 모두 `statistic pvalue`이다. `chisquare power_divergence skewtest kurtosistest normaltest jarque_bera shapiro anderson ks_1samp kstest`는 둘로 풀리고, `cramervonmises`는 풀리지 않는다(scipy와 같다).
- `chisquare`·`power_divergence`: `ddof`는 기대 빈도를 맞춘 모수의 수다(§1.1). `lambda_`는 수나 scipy의 이름(`'pearson'`, `'log-likelihood'`, `'freeman-tukey'`, `'mod-log-likelihood'`, `'neyman'`, `'cressie-read'`)이다. `sum_check`는 관측과 기대의 합이 scipy의 상대 허용오차 안에서 같은지 본다.
- `jarque_bera`(H17, H18, H20): 커널은 htest의 것이고 `sd.regression`의 `r.jarque_bera()`가 그것을 부른다. 통계량은 평균을 뺀 2·3·4차 합(double-double)에서 n/6·(S² + (K − 3)²/4), p값은 `ChiSquared(df=2).ccdf(jb)`다(닫힌 꼴 e^(−JB/2)는 그와 1 ulp 다를 수 있어 쓰지 않는다). 그래서 `sd.htest.jarque_bera(r.resid)`와 OLS의 `r.jarque_bera()`는 비트까지 같고 같은 형(`JarqueBeraResult`, `sd.regression`이 내보내는 이름; H18)을 돌려준다. 예외는 완전 적합이다: regression은 잔차를 0으로 보아 NaN을 내고(G95, G117), `jarque_bera(r.resid)`는 반올림 크기의 잔차로 계산한다. WLS·GLS는 백색화한 잔차의 값이다(regression이 `wresid`를 두면 그 쓰인 행을 넘긴다). 분산 0의 판정은 sd.stats `skew`의 규칙이다(m2 ≤ (ε·평균)²이면 NaN; H20). 커널은 H2에서 regression의 `resid_stats`를 옮겨 오고, regression의 골든은 H20이 바꾸는 퇴화한 경우 말고는 그대로다.
- `shapiro`: Royston(1995) AS R94의 W와 p값이다(scipy·R과 같은 알고리즘). n < 3은 NaN이다(§1.3).
- `anderson`: scipy 1.17부터의 꼴만이다(`statistic pvalue`; 옛 `critical_values`, `significance_level`, `fit_result`는 없다). `dist=None`은 정규 분포다. 받는 분포 족과 그 이름(sd.stats의 분포 클래스 또는 scipy의 문자열)은 H2에서 정한다. `method`는 `'interpolate'`(Stephens의 표 보간)이고 `MonteCarloMethod`는 H9에서 더한다. scipy 1.18은 `method`를 주지 않으면 경고와 함께 옛 꼴을 낸다. 여기서는 기본값이 `'interpolate'`다(`[scipy와 다름]`).
- `ks_1samp`·`kstest`·`cramervonmises`의 `cdf`: sd.stats의 분포 객체(`sd.stats.Normal(mu=0, sigma=1)`; 그 `cdf`를 쓴다)나 `cdf(x)`를 돌려주는 호출 가능한 객체다. scipy의 분포 이름 문자열(`'norm'`)과 `args=()`는 받지 않는다(optimize의 O12와 같다; H2에서 정한다).
- `ks_1samp`: `method`는 `'auto'`, `'exact'`, `'approx'`, `'asymp'`(scipy의 이름)이다. 결과에 `statistic_location`, `statistic_sign`(scipy와 같다), `method`(H12)가 있다.
- `kstest`는 한 표본·두 표본을 고르는 scipy의 감싸개다. 둘지는 H2에서 정한다(추천: 두지 않는다).

## §4. 순위 검정 (H3)
```
mannwhitneyu(x, y, /, *, axis=None, use_continuity=True, alternative='two-sided', method='auto', nan_policy='propagate', keepdims=False)
ranksums(x, y, /, *, axis=None, alternative='two-sided', nan_policy='propagate', keepdims=False)
wilcoxon(x, y=None, /, *, axis=None, zero_method='wilcox', correction=False, alternative='two-sided', method='auto', nan_policy='propagate', keepdims=False)
kruskal(*samples, axis=None, nan_policy='propagate', keepdims=False)
friedmanchisquare(*samples, axis=None, nan_policy='propagate', keepdims=False)
brunnermunzel(x, y, /, *, axis=None, alternative='two-sided', distribution='t', nan_policy='propagate', keepdims=False)
ansari(x, y, /, *, axis=None, alternative='two-sided', method='auto', nan_policy='propagate', keepdims=False)
mood(x, y, /, *, axis=None, alternative='two-sided', nan_policy='propagate', keepdims=False)
median_test(*samples, ties='below', correction=True, lambda_=1, nan_policy='propagate')
page_trend_test(data, /, *, ranked=False, predicted_ranks=None, method='auto')
```
- 결과는 `statistic pvalue`이고 둘로 풀린다. 다른 것: `median_test`는 `statistic pvalue median table`으로 넷으로 풀린다. `page_trend_test`는 `statistic pvalue method`이고 풀리지 않는다(scipy와 같다).
- `method`: `mannwhitneyu`는 `'auto'`, `'asymptotic'`, `'exact'`, `wilcoxon`은 `'auto'`, `'exact'`, `'asymptotic'`, `ansari`는 `'auto'`, `'exact'`, `'asymptotic'`, `page_trend_test`는 `'auto'`, `'exact'`, `'asymptotic'`이다. 결과의 `method`에 쓴 방법을 적는다(H12). `'auto'`가 정확을 고르는 문턱은 scipy 1.18.1 그대로다(docs/design/htest.md §3.1의 표; H3에서 정한다). 재표본 객체는 H9에서 더한다.
- `wilcoxon`: `y`가 없으면 `x`가 차다. `zero_method`는 `'wilcox'`, `'pratt'`, `'zsplit'`이다. 정규 근사로 계산했을 때만 `zstatistic` 필드가 있다(H3에서 정한다: scipy 1.18.1은 사용자가 `method='asymptotic'`을 준 때만 둔다).
- 동점이 있을 때의 `method='exact'`는 H3에서 정한다(추천: 관측한 중간 순위의 조건부 정확 분포, `[scipy와 다름]`).
- `median_test`·`page_trend_test`의 배치: `page_trend_test`의 `data`는 표를 받는 함수의 규칙이다(마지막 두 축이 (m, n), 앞이 배치). `median_test`는 scipy처럼 `axis`가 없다(배치는 H3에서 정한다).

## §5. 적합도 2: 표본끼리 (H4)
```
ks_2samp(data1, data2, /, *, axis=None, alternative='two-sided', method='auto', nan_policy='propagate', keepdims=False)
cramervonmises_2samp(x, y, /, *, axis=None, method='auto', nan_policy='propagate', keepdims=False)
anderson_ksamp(samples, /, *, variant='midrank', method=None)
epps_singleton_2samp(x, y, /, *, axis=None, t=(0.4, 0.8), nan_policy='propagate', keepdims=False)
```
- `ks_2samp`: 결과 `statistic pvalue statistic_location statistic_sign method`, 둘로 풀린다. `method`는 `'auto'`, `'exact'`, `'asymp'`이다.
- `cramervonmises_2samp`: 결과 `statistic pvalue method`, 풀리지 않는다. `method`는 `'auto'`, `'exact'`, `'asymptotic'`이다.
- `anderson_ksamp`: scipy 1.17부터의 `variant` 꼴만(`'midrank'`, `'right'`, `'continuous'`)이다. 옛 `midrank`와 `critical_values`는 없다. scipy 1.18.1은 `variant`를 주지 않으면 경고하고 옛 꼴을 낸다. 여기서는 기본값이 `'midrank'`다(`[scipy와 다름]`; H4에서 정한다). 결과 `statistic pvalue`. scipy 문서는 `variant`를 주면 풀리지 않는다고 적지만 1.18.1의 결과는 둘로 풀린다(어느 쪽을 따를지 H4에서 정한다). `method=None`은 표의 보간, `PermutationMethod`는 H9다.
- `epps_singleton_2samp`: 결과 `statistic pvalue`, 둘로 풀린다.

## §6. 상관 (H5)
```
pearsonr(x, y, /, *, axis=None, alternative='two-sided', method=None, nan_policy='propagate', keepdims=False)
spearmanrho(x, y, /, *, axis=None, alternative='two-sided', method=None, nan_policy='propagate', keepdims=False)
spearmanr(a, b=None, /, *, axis=None, alternative='two-sided', nan_policy='propagate')   # 둘지 H5에서 정한다
pointbiserialr(x, y, /, *, axis=None, nan_policy='propagate', keepdims=False)
kendalltau(x, y, /, *, axis=None, method='auto', variant='b', alternative='two-sided', nan_policy='propagate', keepdims=False)
weightedtau(x, y, /, *, axis=None, rank=True, weigher=None, additive=True, nan_policy='propagate', keepdims=False)
somersd(x, y=None, /, *, alternative='two-sided')
chatterjeexi(x, y, /, *, axis=None, y_continuous=False, method='asymptotic', nan_policy='propagate', keepdims=False)
```
- 결과는 `statistic pvalue`(별칭 `correlation`은 없다)이고 둘로 풀린다. 다른 것: `pearsonr`은 `conf_int(alpha=0.05)`(Fisher z; 재표본 구간은 H9), `kendalltau`는 `method` 필드(H12), `somersd`는 `statistic pvalue table`이고 풀리지 않는다.
- `pearsonr`의 r·p값은 `sd.regression.linregress(x, y)`의 `rvalue`·`pvalue`와 같은 검정이고 같은 커널(1 − r²을 잔차에서; docs/design/htest.md §3.6)이라 같은 값이다(§13).
- `pearsonr`의 `nan_policy`·`keepdims`는 scipy 1.18.1에 없다(`[scipy와 다름]`: 모든 표본 함수의 꼴, §1.1).
- `spearmanr`는 열들의 상관 행렬을 내는 scipy의 옛 꼴이다(`axis`는 변수의 축). `spearmanrho`는 1.18의 새 꼴(브로드캐스트, 배치)이다. 둘 다 둘지는 H5에서 정한다.
- `kendalltau`: `method`는 `'auto'`, `'asymptotic'`, `'exact'`, `variant`는 `'b'`, `'c'`다.
- `weightedtau`: scipy의 `pvalue`는 언제나 NaN이다(scipy가 정의하지 않는다). 그대로 둘지는 H5에서 정한다.
- `chatterjeexi`: `method`는 `'asymptotic'`이고 재표본은 H9다.

## §7. 분할표 (H6)
```
chi2_contingency(observed, /, *, correction=True, lambda_=None, method=None)
fisher_exact(table, /, *, alternative='two-sided', method=None)
barnard_exact(table, /, *, alternative='two-sided', pooled=True, n=32)
boschloo_exact(table, /, *, alternative='two-sided', n=32)
mcnemar(table, /, *, exact=True, correction=True)
cochrans_q(x, /)
odds_ratio(table, /, *, kind='conditional')   # 넣을지 H6에서 정한다
relative_risk(exposed_cases, exposed_total, control_cases, control_total, /)   # 넣을지 H6에서 정한다
```
- 표는 마지막 축들이고 앞이 배치다(§1.2).
- `chi2_contingency`: 결과 `statistic pvalue df expected_freq`(scipy의 `dof`는 `df`, H10), 넷으로 풀린다. `df`는 float다(H6에서 정한다). 소수의 개수(가중 표)를 받을지는 H6에서 정한다.
- `fisher_exact`: 결과 `statistic pvalue`, 둘로 풀린다. 2 × 2는 초기하 분포(양쪽 p값은 scipy·R의 1 + 10⁻⁷ 규칙), r × c의 기본 방법은 H6에서 정한다(scipy는 시드 없는 몬테카를로; 추천: R `fisher.test`의 정확한 망 알고리즘과 상한, `[scipy와 다름]`). scipy의 기본값 `alternative=None`은 `'two-sided'`와 같다. 여기서는 기본값을 `'two-sided'`로 적는다.
- `barnard_exact`·`boschloo_exact`: 결과 `statistic pvalue`, 풀리지 않는다. 2 × 2만이다. `n`의 뜻(분할의 처음 격자 크기로만 쓸지)은 H6에서 정한다.
- `mcnemar`(statsmodels): 결과 `statistic pvalue`, 풀리지 않는다(statsmodels의 `Bunch`). `exact=True`는 이항 정확 검정, `False`는 카이제곱(1)과 `correction`.
- `cochrans_q`(statsmodels): `x`는 (N, k)의 0·1 값이다. 결과 `statistic pvalue df`, 풀리지 않는다. statsmodels의 `return_object`는 없다(언제나 결과 객체).
- `odds_ratio`·`relative_risk`(`scipy.stats.contingency`): 검정이 아니라 신뢰구간이 있는 추정이다. 결과는 scipy의 필드(`odds_ratio`는 `statistic`, `relative_risk`는 `relative_risk exposed_cases exposed_total control_cases control_total`)와 `conf_int(alpha=0.05)`이다(`odds_ratio`의 `conf_int`는 scipy의 `confidence_interval`처럼 `alternative`도 받는다: `conf_int(*, alpha=0.05, alternative='two-sided')`).

## §8. 비율과 이항 (H7)
```
binomtest(k, n, /, *, p=0.5, alternative='two-sided')
quantile_test(x, /, *, axis=None, q=0.0, p=0.5, alternative='two-sided', nan_policy='propagate', keepdims=False)
poisson_means_test(k1, n1, k2, n2, /, *, diff=0, alternative='two-sided')
proportions_ztest(count, nobs, /, *, value=None, alternative='two-sided', prop_var=None)
proportions_chisquare(count, nobs, /, *, value=None)
proportion_confint(count, nobs, /, *, alpha=0.05, method='normal', alternative='two-sided')
test_proportions_2indep(count1, nobs1, count2, nobs2, /, *, value=None, method=None, compare='diff', alternative='two-sided', correction=True)
confint_proportions_2indep(count1, nobs1, count2, nobs2, /, *, method=None, compare='diff', alpha=0.05, correction=True)
```
- `binomtest`: 결과 `k n alternative statistic pvalue proportion_estimate`와 메서드 `conf_int(*, alpha=0.05, method='exact')`(scipy의 `proportion_ci(confidence_level=0.95, method='exact')`; 방법 `'exact'`, `'wilson'`, `'wilsoncc'`은 scipy와 같다; H16), 풀리지 않는다. `k`·`n`은 int, `p`는 [0, 1]의 실수다.
- `quantile_test`: 결과 `statistic statistic_type pvalue`와 `conf_int(alpha=0.05)`, 풀리지 않는다. scipy에 없는 `nan_policy`를 둔다(§1.1).
- `poisson_means_test`: 결과 `statistic pvalue`, 둘로 풀린다.
- statsmodels의 함수(H5): 신뢰수준은 statsmodels와 같은 `alpha=0.05`(H16), `alternative`는 `'larger'`·`'smaller'`가 아니라 `'greater'`·`'less'`, `proportions_ztest`의 `prop_var=False`는 None, 결과는 튜플 대신 결과 객체다. 각각 H7에서 정한다(§12의 추천).
  - `proportions_ztest`: 결과 `statistic pvalue`, 둘로 풀린다(statsmodels의 튜플 순서).
  - `proportions_chisquare`: 결과 `statistic pvalue table expected_freq`. statsmodels는 `(chi2, p, (table, expected))`를 돌려준다. 풀기의 꼴은 H7에서 정한다.
  - `proportion_confint`: 아래 끝과 위 끝의 (…, 2) 배열을 돌려준다(`conf_int()`과 같은 꼴, H16; 구간 하나면 statsmodels의 `(low, upp)`처럼 `low, upp = ...`로 풀린다). `method`는 `'normal'`, `'agresti_coull'`, `'exact'`(statsmodels `'beta'`, Clopper–Pearson), `'wilson'`, `'wilsoncc'`, `'jeffreys'`, `'binom_test'`의 안이다(이름은 H7에서 정한다).
  - `test_proportions_2indep`: 결과는 statsmodels의 필드(`statistic pvalue compare method diff ratio odds_ratio variance alternative value prop1_null prop2_null`)이고 둘로 풀린다. `return_results`는 없다. `method=None`과 기본 방법은 statsmodels 0.15.0의 것으로 고정한다(차 `'agresti-caffo'`, 비 `'log-adjusted'`, 오즈비 `'logit-adjusted'`).
  - `confint_proportions_2indep`: (…, 2) 배열(`proportion_confint`과 같은 꼴). 기본 방법은 0.15.0의 것(차 `'newcomb'`, 비 `'log-adjusted'`, 오즈비 `'logit-adjusted'`).

## §9. 다중 비교와 보정 (H8)
```
tukey_hsd(*samples, equal_var=True)
dunnett(*samples, control, alternative='two-sided')
multipletests(pvals, /, *, alpha=0.05, method='holm-sidak', max_iter=1, axis=None)
false_discovery_control(ps, /, *, axis=None, method='bh')
combine_pvalues(pvalues, /, *, axis=None, method='fisher', weights=None, nan_policy='propagate', keepdims=False)
```
- `tukey_hsd`: 결과 `statistic pvalue`((k, k) 배열)와 `conf_int(alpha=0.05)`, 풀리지 않는다. `equal_var=False`는 Games–Howell이다(scipy 1.18.1).
- `dunnett`: `control`은 키워드로만이다(scipy와 같다). 결과 `statistic pvalue`와 `conf_int(alpha=0.05)`, 풀리지 않는다. 다변량 t의 cdf가 결정적이라(sd.stats의 `MultivariateT`) `rng`가 없다(`[scipy와 다름]`; H8에서 정한다).
- `multipletests`(statsmodels): 방법은 statsmodels의 긴 이름 열 개(`'bonferroni'`, `'sidak'`, `'holm-sidak'`, `'holm'`, `'simes-hochberg'`, `'hommel'`, `'fdr_bh'`, `'fdr_by'`, `'fdr_tsbh'`, `'fdr_tsbky'`)만이다. 결과 `reject pvalues_corrected alpha_sidak alpha_bonferroni`이고 statsmodels의 튜플 순서로 넷으로 풀린다(필드 이름은 H8에서 정한다). `is_sorted`·`returnsorted`는 없다. `axis=None`이면 모든 p값이 한 묶음이다.
- `multipletests`·`false_discovery_control`은 회귀 계수 여럿의 p값(`r.pvalues`, 배치면 (…, k))과 계획 중인 `sd.regression`의 단계적 변수 선택이 내는 p값을 보정하는 짝이다: `multipletests(r.pvalues, axis=-1)`. 다만 같은 자료로 고르고 검정한 선택 뒤의 p값은 명목 수준을 지키지 않고, 다중 검정 보정도 그 선택의 편향을 고치지 못한다(§13).
- `false_discovery_control`: 보정한 p값의 배열을 돌려준다(결과 객체가 아니다; scipy와 같다). `method`는 `'bh'`, `'by'`. `axis=None`이면 모든 원소가 한 묶음이고 결과는 입력의 모양이다(sd.stats의 `rankdata`와 같다; scipy는 펼친 1차원을 돌려준다).
- `combine_pvalues`: 결과 `statistic pvalue`, 둘로 풀린다. `method`는 `'fisher'`, `'pearson'`, `'tippett'`, `'stouffer'`, `'mudholkar_george'`.
- statsmodels의 `fdrcorrection`은 두지 않는다(H8에서 정한다).

## §10. 재표본과 몬테카를로 (H9)
```
permutation_test(data, statistic, /, *, permutation_type='independent', vectorized=None, n_resamples=9999, batch=None, alternative='two-sided', axis=None, rng=None)
monte_carlo_test(data, rvs, statistic, /, *, vectorized=None, n_resamples=9999, batch=None, alternative='two-sided', axis=None, rng)
bootstrap(data, statistic, /, *, n_resamples=9999, batch=None, vectorized=None, paired=False, axis=None, alternative='two-sided', method='BCa', rng)
PermutationMethod(n_resamples=9999, batch=None, *, rng=None)
MonteCarloMethod(n_resamples=9999, batch=None, rvs=None, *, rng)
BootstrapMethod(n_resamples=9999, batch=None, method='BCa', *, rng)
bws_test(x, y, /, *, axis=None, alternative='two-sided', method=None)
multiscale_graphcorr(x, y, /, *, compute_distance=None, reps=1000, workers=1, is_twosamp=False, rng)   # 넣을지 H9에서 정한다
goodness_of_fit(dist, data, /, *, known_params=None, fit_params=None, guessed_params=None, statistic='ad', n_mc_samples=9999, rng)   # 넣을지 H9에서 정한다
```
- `permutation_test`: 가능한 순열이 `n_resamples` 이하이면 모두 나열하고(결정적, `rng`를 쓰지 않는다), 그보다 많으면 `rng`가 꼭 있어야 한다(없으면 `TypeError`). 결과 `statistic pvalue null_distribution`, 풀리지 않는다.
- `monte_carlo_test`·`bootstrap`·`MonteCarloMethod`·`BootstrapMethod`는 언제나 무작위라 `rng`가 꼭 있다(키워드, 기본값 없음). `monte_carlo_test`의 `rvs`가 `rng`를 받는 꼴(`rvs(size, *, rng)`)은 H9에서 정한다.
- `bootstrap`: 결과 `bootstrap_distribution standard_error`와 메서드 `conf_int(*, alpha=0.05)`(H16: scipy는 호출의 `confidence_level`로 만든 필드 `confidence_interval`이다; 여기서는 들고 있는 재표본 분포에서 `alpha`마다 percentile·basic·BCa 구간을 만든다). scipy의 `bootstrap_result`(앞의 결과에 잇기)는 두지 않는 안이다(H9에서 정한다).
- `axis=None`이면 표본마다 펼치고 사용자 `statistic`은 `axis=-1`로 부른다(H9의 결정에 맞춘 안; scipy는 `axis=0`).
- `PermutationMethod`·`MonteCarloMethod`·`BootstrapMethod`는 앞 마일스톤의 검정들의 `method=`에 넘기는 설정 객체다(scipy의 이름; `random_state` 없음). 공개 클래스다.
- `bws_test`: 결과 `statistic pvalue null_distribution`, 풀리지 않는다. 기본 `method=None`은 가능하면 모든 순열을 나열하고 아니면 `method=PermutationMethod(rng=...)`를 요구한다(H9에서 정한다).
- `multiscale_graphcorr`: scipy의 `random_state`는 `rng`, `compute_distance=None`은 유클리드 거리. 결과 `statistic pvalue mgc_dict`(scipy의 `stat` 별칭은 없다).
- `goodness_of_fit`: `dist`는 sd.stats의 분포 클래스(H9에서 정한다). 결과 `fit_result statistic pvalue null_distribution`.

## §11. scipy·statsmodels 이름 → sd 이름
같은 이름은 적지 않는다(`scipy.stats.ttest_ind` → `sd.htest.ttest_ind`처럼 §2–§10의 모든 함수는 scipy·statsmodels의 이름 그대로다).

| scipy·statsmodels | sd.htest | 이유 |
|---|---|---|
| 결과의 `dof`(`chi2_contingency`) | `df` | H10 |
| 결과의 `correlation`, MGC의 `stat` | 없음(`statistic`) | 별칭 없음 |
| `random_state=`, `seed=` | `rng=`(sd.random의 Generator) | H3 |
| `multipletests(maxiter=)` | `max_iter=` | optimize O14와 같은 이름 |
| `multipletests` 결과 튜플 `(reject, pvals_corrected, alphacSidak, alphacBonf)` | 필드 `reject pvalues_corrected alpha_sidak alpha_bonferroni` | H8에서 정한다 |
| 결과의 `confidence_interval(confidence_level=0.95)` → `ConfidenceInterval(low, high)` | `conf_int(*, alpha=0.05)` → (…, 2) 배열 | H16(regression의 G99) |
| `binomtest(...).proportion_ci(confidence_level=0.95, method=)` | `conf_int(*, alpha=0.05, method=)` | H16 |
| `bootstrap(confidence_level=0.95)`와 필드 `confidence_interval` | 결과의 `conf_int(*, alpha=0.05)` | H16 |
| statsmodels의 `alpha`(`proportion_confint` 등) | 그대로 `alpha` | H16 |
| F 꼴 자유도(statsmodels `df_num`·`df_denom`, R `parameter`) | `df_num`, `df_denom` | H19 |
| `ttest_ind_from_stats`의 결과 `statistic pvalue` | `statistic pvalue df`와 `conf_int()` | H23 |
| `f_oneway`의 결과 `statistic pvalue` | `statistic pvalue df_num df_denom` | H27 |
| `levene`의 결과 `statistic pvalue` | `statistic pvalue df_num df_denom` | H29 |
| `bartlett`·`fligner`·`alexandergovern`의 결과 `statistic pvalue` | `statistic pvalue df` | H29 |
| `alternative='larger'`, `'smaller'` | `'greater'`, `'less'` | H3·H5(scipy의 이름) |
| `proportion_confint(method='beta')` | `method='exact'` | H7에서 정한다 |
| `chi2_contingency(...).dof`, `cochrans_q` 결과의 `df` | `df` | H10 |
| scipy의 결과 형 `TtestResult`, `SignificanceResult` 등 | 함수마다 `<함수>Result`(내보내지 않음) | H6 |
| `kstest` | 미정 | H2에서 정한다 |
| `statsmodels.stats.multitest.fdrcorrection` | 없음(`multipletests(method='fdr_bh')`·`false_discovery_control`) | H8에서 정한다 |
| `statsmodels` `binom_test` | 없음(`binomtest`) | 같은 검정 |

## §12. 마일스톤별 남은 질문 (추천을 첫째에)
docs/design/htest.md §9와 같은 목록이다. 결정이 나면 이 문서와 그 문서를 함께 고친다.

- **H1**: 정했다(H22–H33, docs/design/htest.md §1.1).
- **H2**: `kstest`를 두지 않는다(추천) / 둔다. `cdf` 인자는 sd.stats 분포 객체와 호출 가능한 객체(추천) / 문자열 이름도. `anderson`의 `dist`는 sd.stats의 분포 클래스(추천; `gumbel_l`은 없다) / scipy의 문자열. `anderson`의 기본 `method='interpolate'`(추천). Shapiro–Wilk는 AS R94의 정의(추천).
- **H3**: 동점이 있을 때 `'exact'`는 조건부 정확 분포(추천) / scipy처럼 동점이 없는 분포 / `ValueError`. `'auto'`의 문턱은 scipy 그대로(추천). `ranksums`를 둔다(추천). `wilcoxon`의 `zstatistic`은 정규 근사로 계산한 때마다(추천; scipy는 사용자가 `'asymptotic'`을 준 때만). 배치의 `method` 필드는 줄마다의 문자열 배열(추천) / 줄이 모두 같을 때만 str. `median_test`의 배치: `axis`를 더한다(추천) / scipy처럼 없다.
- **H4**: `ks_2samp`의 동점: scipy의 정확 분포(추천, H4 전에 R `psmirnov`와 다시 대조) / R처럼 조건부. `cramervonmises_2samp`의 `'exact'`를 동적 계획으로 크게(추천). `anderson_ksamp`의 기본 `variant='midrank'`(추천)와 풀기: scipy 1.18.1의 실제 동작(둘로 풀린다; 추천) / scipy 문서(풀리지 않는다).
- **H5**: `spearmanr`와 `spearmanrho` 둘 다(추천). Spearman 정확 분포는 넣지 않는다(추천). `weightedtau`는 `pvalue` NaN 그대로(추천).
- **H6**: r × c `fisher_exact`의 기본은 정확한 망 알고리즘과 상한(추천). `barnard_exact`·`boschloo_exact`의 `n`은 처음 격자 크기로만(추천). `odds_ratio`·`relative_risk`를 넣는다(추천). `chi2_contingency`는 소수의 개수도 받는다(추천). `mcnemar`의 기본 `exact=True`(statsmodels 그대로; 추천). `chi2_contingency`의 `df`는 float(추천, H10과 `ttest`의 `df`와 같은 형) / int.
- **H7**: 방법 이름은 scipy에 있는 것은 scipy의 이름(`'exact'`, `'wilson'`, `'wilsoncc'`)과 statsmodels에만 있는 것(추천) / statsmodels 그대로(`'beta'`). 기본값은 statsmodels의 `'normal'`(추천). `prop_var`는 None 또는 실수(추천). `proportions_chisquare`는 `(statistic, pvalue, (table, expected_freq))`의 셋으로 풀린다(statsmodels 그대로; 추천) / 넷의 필드만. `proportion_confint`·`confint_proportions_2indep`은 `conf_int()`과 같은 (…, 2) 배열(추천; H16) / statsmodels처럼 (low, upp) 튜플. 기본 방법은 statsmodels 0.15.0의 것으로 고정(추천). `proportions_chisquare`에 Yates 보정은 없다(추천).
- **H8**: `false_discovery_control` + `multipletests`, `fdrcorrection` 없음(추천). `multipletests`의 기본 `'holm-sidak'`(추천). 필드 이름 `reject pvalues_corrected alpha_sidak alpha_bonferroni`(추천). `fdr_gbs`는 넣지 않는다(추천). 스튜던트화 범위 분포는 htest 안에(H14로 정했다). `dunnett`에 `rng`가 없다(추천).
- **sd.regression과의 경계**: H16–H21로 정했다(§13; docs/design/htest.md §1.4). 남은 것은 regression 쪽의 결정이다(docs/design/regression.md §8의 "htest와의 관계"): 새 회귀 검정(모형 비교의 F·LR·Wald, White, Goldfeld–Quandt, RESET)의 결과를 H6·H7·H11·H19·H21의 꼴로 둘지, `breusch_pagan()`에 자유도 필드를 더할지(H19라면 카이제곱의 `df`), 백색화한 잔차 `wresid`를 결과에 둘지, 단계적 선택에 보정 옵션을 둘지(둔다면 `multipletests`의 방법 이름과 같은 커널).
- **H9**: 설정 객체는 scipy의 이름과 `rng` 필수(추천). `monte_carlo_test`의 `rvs(size, *, rng)`(추천). `bootstrap_result`를 두지 않는다(추천). 재표본 함수의 `axis=None`(표본마다 펼치기; 추천) / scipy의 0. `multiscale_graphcorr` 넣는다(추천). `goodness_of_fit`은 `anderson`과 같은 족의 최우추정으로(추천). `bws_test`의 기본(나열 아니면 `rng` 요구; 추천). R 꾸러미(`coin` 등)는 마일스톤마다 묻는다(H15).

## §13. sd.regression과 겹치는 곳
표본·표를 받는 검정은 `sd.htest`, 적합한 모형을 받는 검정과 진단(`breusch_pagan()`, `durbin_watson`, 계획 중인 VIF·DFFITS·DFBETAS·White·Goldfeld–Quandt·RESET, 모형 비교의 F·LR·Wald 검정과 `anova` 표)은 `sd.regression`의 것이다. 두 영역이 함께 계산하는 값의 커널은 htest에 있고 regression이 부른다(H17). 같은 검정은 같은 형을 돌려준다(H18).

| 같은 값 | sd.htest | sd.regression | 같음의 정도 |
|---|---|---|---|
| Jarque–Bera | `jarque_bera(x)` | `r.jarque_bera()`(백색화한 잔차, 가중치 0인 행은 빠진다) | 한 커널(htest)과 같은 p값 호출 `ChiSquared(df=2).ccdf`: `jarque_bera(r.resid)` = OLS의 `r.jarque_bera()`, 비트까지, 같은 형(H17, H18). 예외: 완전 적합(regression은 NaN). WLS·GLS는 백색화한 잔차(regression의 `wresid`가 생기면 그것) |
| 일원 분산분석의 F | `f_oneway(*groups)` | 집단 더미의 OLS(`fvalue`, `f_pvalue`), 계획 중인 `anova` 표(항 하나) | 같은 F·p이지만 정확도 계약 안에서만 같다: `anova`는 설계 행렬만 받아 집단 레이블이 없고 QR에서 계산한다. p값은 같은 `F(df_num, df_denom).ccdf` 호출 |
| 두 집단의 t | `ttest_ind(a, b)` | 집단 더미 하나의 OLS에서 그 계수의 `tvalues`·`pvalues` | 같은 검정(Student), 계약 안에서 같다(부호는 더미의 방향) |
| 상관의 검정 | `pearsonr(x, y)` | `linregress(x, y)`의 `rvalue`·`pvalue`(기울기 = 0의 t) | 같은 검정, 1 − r²을 잔차에서 계산하는 같은 방법. H17에 따라 커널은 htest(H5)에 두고 `linregress`가 부른다(그때 regression의 골든이 바뀌는 경우가 있으면 따로 묻는다) |
| p값의 꼬리 | 모든 검정 | `pvalues`, `f_pvalue`, `lm_pvalue`, 모형 비교의 F·LR·Wald | sd.stats 분포의 `ccdf`·`logccdf`를 같은 인자로 부른다(같은 비트) |
| 신뢰구간 | `conf_int(*, alpha=0.05)` | `conf_int(*, alpha=0.05)`(G99) | 같은 이름, 같은 인자, 같은 메시지, 같은 (…, 2) 끝 순서(H16) |
| 다중 검정 보정 | `multipletests`, `false_discovery_control` | 계수 여럿의 `pvalues`, 계획 중인 단계적 변수 선택의 p값 | `multipletests(r.pvalues, axis=-1)`(배치 (…, k)도). 선택 뒤의 p값은 명목 수준을 지키지 않고, 보정도 선택의 편향을 고치지 못한다 |
| 사후 비교 | `tukey_hsd`, `dunnett` | 계획 중인 `anova` 표 | 분산분석 뒤의 쌍 비교는 htest의 것이다 |

- `durbin_watson`은 p값이 없는 통계량(속성)이고 scipy에 검정이 없어 htest에 짝이 없다. `breusch_pagan()`·White·Goldfeld–Quandt·RESET은 모형의 설계 행렬이 필요해 regression에만 있다. 잔차의 다른 정규성 검정(Shapiro–Wilk, Anderson–Darling, D'Agostino)은 잔차를 htest에 넘긴다.
- regression의 지금 결과(`JarqueBeraResult`, `BreuschPaganResult`)와 htest의 결정:

| | 지금의 regression | 결정 |
|---|---|---|
| 형 | 두 namedtuple(`stardust._results`), `sd.regression`이 내보낸다 | Jarque–Bera는 htest의 결과 클래스가 되고 `sd.regression.JarqueBeraResult`가 그 클래스를 가리킨다(H18, H2에서). `BreuschPaganResult`는 htest에 짝이 없어 regression이 정한다 |
| 풀기 | JB 둘(scipy와 같다), BP 넷(statsmodels `het_breuschpagan`의 순서) | H7과 맞는다. htest의 JB 클래스도 둘로 풀린다 |
| 필드 이름 | BP는 `lm lm_pvalue fvalue f_pvalue`(statsmodels) | scipy에 없는 결과는 statsmodels의 이름(H5)이라 맞는다 |
| 자유도 | BP에 자유도 필드가 없다 | 둔다면 카이제곱은 `df`, F는 `df_num`·`df_denom`(H19; regression이 정한다) |
| 퇴화한 입력 | 완전 적합은 0/0 NaN, BP 자유도 ≤ 0은 NaN, 경고 없음 | H11과 맞는다. 분산 0의 판정은 sd.stats `skew`의 규칙 하나로(H20): regression의 `jarque_bera()`도 m2 ≤ (ε·평균)²이면 NaN이 된다(그런 퇴화한 경우만 값이 바뀐다) |
| 배열 | 쓰기 금지, 복사 없음(G120) | 같다(H21) |

