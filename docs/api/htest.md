# sd.htest API

> 상태: **초안** (H0, 2026-11-20; H1·H2 구현 2026-11-20, H3·H4·H5·H6·H7 구현 2026-11-21). 공개된 검정은 H1의 아홉 함수(§2), H2의 열 함수(§3), H3의 열 함수(§4), H4의 네 함수(§5), H5의 여덟 함수(§6), H6의 여덟 함수(§7), H7의 여덟 함수(§8)다. 나머지는 모듈 `sd.htest`, 결과 객체의 바탕, 공통 입력 층(H0) 위에 마일스톤마다 더한다. 아래 시그니처는 scipy 1.18.1의 `scipy.stats`와 statsmodels 0.15.0의 `statsmodels.stats`의 이름과 인자 이름(H3, H5)에 stardust의 규칙과 H6–H15의 결정을 입힌 안이다. 함수는 마일스톤 H1–H9에서 하나씩 공개된다(docs/design/htest.md §8). 마일스톤에서 정할 것은 그 자리에 "(H<n>에서 정한다)"로 적고 §12에 추천과 함께 모았다. sd.regression과 같은 값을 계산하는 곳은 §13에 있다.

- 설계와 결정의 근거는 [docs/design/htest.md](../design/htest.md)(§1.1 H1–H101), 공통 규칙은 [API.md](../../API.md)다.
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
- **`method`와 `'auto'`**(H12, H46): scipy가 `method='auto'`를 두는 검정(`ks_1samp ks_2samp mannwhitneyu wilcoxon ansari kendalltau cramervonmises_2samp page_trend_test`)은 scipy와 같은 문턱으로 방법을 고르고, 결과의 `method` 필드에 실제로 쓴 방법(`'exact'`, `'asymptotic'`, `'approx'` 등 그 검정의 이름)을 적는다. scipy는 `page_trend_test`만 적는다. 배치에서는 줄마다 고르고(scipy는 동점 같은 조건을 배치 전체에서 본다), `method`는 줄마다의 이름을 배치 모양으로 중첩한 리스트다(H48: `[['exact', 'asymptotic'], ...]`; sd 배열에 문자열 형이 없다). 리스트는 읽을 때마다 새 리스트다(G113). `nan_policy='propagate'`로 NaN이 된 줄의 `method`는 그 줄의 나머지 값으로 고른 방법이다(H53).
- **정확한 방법의 상한**(H13): `method='exact'`를 고른 호출의 작업량이 그 검정의 문서화한 상한을 넘으면 `ValueError`다. 메시지는 다른 길을 알려 준다: `ks_2samp(): method='exact' needs about 4.1e+12 operations for this input, more than its cap of 1e+09; use method='asymp'`(재표본이 있는 검정은 `or a PermutationMethod`). `'auto'`는 상한 안에서만 정확을 고른다. 그래서 모든 호출이 정해진 시간 안에 끝난다. 상한의 값은 검정마다 그 마일스톤에서 정한다(순위 검정은 §4).
- **`rng`**: 무작위를 쓰는 계산(재표본, 몬테카를로 p값)은 `rng=`에 `sd.random.Generator`를 꼭 받는다. 없거나 다른 것이면 `permutation_test(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType`(`TypeError`; optimize의 O31과 같다). 시드 없이 무작위로 도는 기본값은 없다(sd.random의 R3). 같은 `rng` 상태와 같은 입력이면 스레드 수와 상관없이 같은 비트다.
- 문자열 옵션은 정확한 이름만 받는다(대소문자를 바꾸지 않는다). 수 옵션은 int와 float(numpy 정수도 int)다.

### §1.5 결과 객체 (H6, H7, H10, H12, H16, H18, H19, H21)
- 검정마다 결과 클래스가 하나이고 이름은 `<함수>Result`다: `ttest_ind` → `TtestIndResult`, `pearsonr` → `PearsonrResult`, `chi2_contingency` → `Chi2ContingencyResult`(H6). 공통 바탕 클래스를 쓴다. 클래스는 내보내지 않는다(`sd.htest`의 공개 이름은 함수뿐이다). `isinstance`가 아니라 필드로 읽는다.
- **예외: 두 영역이 같은 검정을 돌려주는 클래스**(H18). 같은 검정은 같은 형이다: Jarque–Bera의 결과 클래스는 htest의 것이고, `sd.regression`이 내보내는 공개 이름 `JarqueBeraResult`가 그 클래스를 가리킨다(`sd.htest.jarque_bera(x)`와 `r.jarque_bera()`가 같은 형을 돌려준다; H39, regression G165: namedtuple이 아니어서 튜플 비교·해시·`_asdict`·`_replace`·`_fields`가 없고 `(statistic, pvalue)` 풀기는 그대로다). 그 밖의 결과는 H6대로 내보내지 않는다.
- **필드는 읽기 전용 속성이다**(regression의 G98, optimize의 `OptimizeResult`와 같다): 대입하면 `AttributeError`다. 배열 필드는 쓰기 금지 배열이고 읽을 때 복사하지 않는다(H21, API.md §2.2, regression의 G120; 고치려면 `.copy()`). 리스트인 필드(설정, 배치의 `method`)는 읽을 때마다 새 리스트다(G113). 그 호출에서 주지 않는 필드(정규 근사를 쓰지 않은 `mannwhitneyu`·`wilcoxon`의 `zstatistic`, `distribution='normal'`인 `brunnermunzel`의 `df`)는 `AttributeError`라 `hasattr()`로 안다.
- 검정 하나면 필드는 Python 수다(`statistic`·`pvalue`·`df`는 float, 개수는 int, `method`는 str). 배치면 배치 모양의 sd 배열이다(`method`는 중첩 리스트, H48).
- **튜플 풀기**(H7): scipy의 결과가 풀리는 검정만 scipy와 같은 개수와 순서로 풀린다(`stat, p = sd.htest.ttest_ind(a, b)`; `chi2_contingency`는 넷, `median_test`도 넷). 인덱스(`r[0]`)와 `len(r)`도 그 튜플의 것이다. scipy의 결과가 풀리지 않는 검정(`binomtest`, `cramervonmises`, `alexandergovern`, `barnard_exact`, `somersd`, `page_trend_test`, 재표본 결과 등)은 풀리지 않는다(`TypeError: cannot unpack non-iterable BinomtestResult object`). statsmodels의 튜플을 돌려주는 함수는 §8·§9에 적는다.
- **신뢰구간**(H16): regression의 꼴인 메서드 `r.conf_int(alpha=0.05)`다(G99; scipy의 `confidence_interval(confidence_level=0.95)`가 아니다). `alpha`는 키워드로만 받는 유의수준이고 구간은 1 − alpha다. 돌려주는 것은 아래 끝과 위 끝의 (…, 2) float64 배열이다: 검정 하나면 (2,), 배치면 (배치 모양, 2), `tukey_hsd`처럼 비교가 여럿이면 (k, k, 2). 부를 때마다 새 배열이다(regression의 `conf_int()`처럼 쓸 수 있다, G120). `alpha`가 (0, 1) 밖이면 `conf_int(): 'alpha' must be between 0 and 1 (exclusive)`(`ValueError`), 수가 아니면 `conf_int(): 'alpha' must be an int or a float, not str`(`TypeError`)다(regression과 같은 메시지). 구간의 방향은 그 검정의 `alternative`를 따른다(scipy와 같다: 한쪽이면 한 끝이 ±inf).
- `repr`는 한 줄 요약이다: `TtestIndResult(statistic=2.13, pvalue=0.0412, df=27)`, 배치는 `TtestIndResult(batch shape (10,))`. 배열 필드는 짧은 1차원이면 값(`statistic=[2.13]`, `keepdims=True`), 그 밖에는 모양(`table=<array of shape (2, 3)>`)이다(H76).
- **`summary()`**(H74–H76): 회귀 상자의 꼴인 글(`str`)이다(API.md §2.3). 제목은 검정의 이름(scipy·R·statsmodels의 이름; 갈래는 괄호: `Two-sample t-test (Welch)`, `Kendall rank correlation test (tau-b)`)이고, 머리의 왼쪽은 필드의 수(`Statistic:`, `P-value:`, `Df:`·`Df Num:`·`Df Denom:`, 그 밖의 수 필드), 오른쪽은 호출의 설명(`Alternative:`, `Method:`, `No. Observations:`(표본 크기, 여럿이면 `25, 24`; 다섯 이상이면 합과 `No. Samples:`), 옵션(`Center:`, `Trim:`, `Lambda:` 등))이다. 표본 크기는 셀 수 있을 때만 적는다(`nan_policy='omit'`에서 NaN을 뺀 수는 따로 세지 않아 적지 않는다; 빈도의 검정은 `No. Categories:`).
  - 구간이 있는 결과(H16)는 `summary(*, alpha=0.05)`이고 구간이 한 줄의 표다(`mean`, `mean difference`, `correlation`; 열 이름은 분위수, 한쪽 검정은 `[0.000 0.950]`·`[0.050 1.000]`이고 끝은 ±inf나 ±1). 구간이 없는 결과는 `summary()`이고 `alpha`를 주면 `summary(): unexpected keyword argument 'alpha'`다.
  - 검정 하나의 배열 필드(`median_test`·`somersd`의 `table`)는 표로 보인다(행 `[i]`, 열 `[j]`; 20을 넘는 행·열은 앞뒤 5개와 `...`).
  - 배치는 레인마다 한 줄인 표다: 열은 레인마다의 필드(`statistic pvalue df` 등)와 구간이고, 레인마다 다른 `method`는 열, 모두 같으면 머리다. 레인마다의 표(배치의 `table`)는 보이지 않는다. 20줄이 넘으면 앞뒤 5줄과 `...` 줄이다.

  ```text
                            Two-sample t-test (Welch)
  ==============================================================================
  Statistic:                     -0.9266  Alternative:                 two-sided
  P-value:                         0.359  No. Observations:               25, 24
  Df:                              46.87
  ==============================================================================
                       [0.025     0.975]
  ------------------------------------------------------------------------------
  mean difference      -0.790      0.292
  ==============================================================================

                           Two-sample t-test (Student)
  ==============================================================================
  Batch shape:                      (3,)  No. Observations:               20, 22
  Alternative:                 two-sided
  ==============================================================================
              statistic     pvalue         df     [0.025     0.975]
  ------------------------------------------------------------------------------
  [0]             0.761      0.451         40     -0.454      1.003
  [1]            -1.169      0.249         40     -1.234      0.329
  [2]            -1.278      0.209         40     -0.829      0.187
  ==============================================================================
  ```

### §1.6 에러
- API.md §2의 종류와 꼴이다. 공개 함수의 에러는 `ttest_ind(): `로 시작하고, 인자는 작은따옴표 안에 쓴다. 경고는 내지 않는다. scipy가 경고(`SmallSampleWarning`, `ConstantInputWarning`, 근사가 부정확하다는 경고)를 내는 곳은 NaN이거나(퇴화한 입력) 문서다(근사의 품질).

### §1.7 scipy·statsmodels와 다른 점 (모든 검정에 걸친 것)
- `axis`의 기본값 None(scipy는 대개 0), 데이터는 위치로만, 옵션은 키워드로만, 별칭 없음, `random_state` 없음과 `rng` 필수, 결과 클래스의 이름(`TtestIndResult`; scipy는 `TtestResult`·`SignificanceResult`를 나눠 쓴다), `dof` → `df`, 모든 `'auto'` 결과의 `method` 필드, 퇴화한 입력의 NaN과 경고 없음, `method='exact'`의 상한. 값의 차이는 반올림 크기이고 정확도 계약 안이다(API.md §4 규칙 3, docs/design/htest.md §4).
- 신뢰구간은 regression의 `conf_int(*, alpha=0.05)`이고 scipy의 `confidence_interval(confidence_level=0.95)`가 아니다(H16, `[scipy와 다름]`: 영역 사이의 통일이 먼저다). 돌려주는 것도 `ConfidenceInterval(low, high)`가 아니라 (…, 2) 배열이다. 대응은 §11의 표다.
- H3에서 scipy와 다른 것(§4): 동점이 있을 때의 `'exact'`(H45), 정규 근사의 `zstatistic`(H47), 줄마다의 `method`(H48), `median_test`의 `axis`(H49)와 정확한 중앙값, `kruskal`·`friedmanchisquare`·`brunnermunzel`의 `df`, 퇴화한 입력의 NaN, `wilcoxon`의 정확한 차.
- H4에서 scipy와 다른 것(§5): 동점이 있는 `ks_2samp`의 `'exact'`(H55)와 `cramervonmises_2samp`의 `'exact'`(조건부 분포, H58), `epps_singleton_2samp`의 `df`(H59), `cramervonmises_2samp`의 `'exact'`가 큰 표본까지(H56), `anderson_ksamp`의 기본 `variant='midrank'`(H57), `ks_2samp`·`cramervonmises_2samp`의 `method` 필드, 퇴화한 입력의 NaN(scipy는 `ValueError`), `ks_2samp`의 정확 계산이 넘치지 않는 것(scipy는 큰 표본에서 경고와 함께 점근으로).
- H5에서 scipy와 다른 것(§6): `pearsonr`·`spearmanrho`의 p값을 1 − r²에서 상쇄 없이(scipy는 |r| ≈ 1에서 틀린다), `pearsonr`의 `nan_policy`·`keepdims`, 거의 상수인 표본(H20)·쌍이 2개 미만·inf의 NaN(scipy는 경고와 값, `ValueError`), `spearmanr`의 기본 `axis=None`, `spearmanr`의 n = 2에서 p = 1(`spearmanrho`와 같다; scipy의 옛 꼴은 NaN), `kendalltau`의 n = 2 정규 근사(scipy는 0/0의 NaN), `weightedtau`의 `nan_policy='propagate'`가 NaN(scipy는 NaN을 가장 큰 값으로 순위 매긴다), `somersd`의 S = 0이 NaN(scipy는 p = 0)·NaN과 2차원 `x`에 준 `y`의 `ValueError`·두 표본의 표가 10⁷칸까지, `chatterjeexi`의 x의 동점을 위치로 가르기(scipy는 `argsort`가 임의로), 별칭 `correlation` 없음(H65).
- H6에서 scipy와 다른 것(§7): r × c `fisher_exact`가 정확한 망 알고리즘(H77; scipy는 시드 없는 몬테카를로)과 그 작업량 상한, `barnard_exact`·`boschloo_exact`의 `n`이 처음 격자 크기로만(H78)이고 p값이 참 상한, Barnard의 정확한 기각역, `chi2_contingency`의 `df`가 float(H82)·0인 행·열의 NaN·n차원 표 없음(3차원 이상은 배치)·Cressie–Read의 0인 칸의 극한, 표의 배치(모든 분할표 함수), `odds_ratio`의 정수 값 float 표, `mcnemar`의 b + c = 0 NaN(statsmodels는 inf와 0), `cochrans_q`의 0·1 값만.
- H7에서 scipy·statsmodels와 다른 것(§8): 방법과 `alternative`의 이름(H85, H86), `proportion_confint`의 한쪽 구간의 방향(H95), `prop_var`(H87), `return_results` 없음(H88), `binomtest`의 `proportion_estimate` 없음(H96), 구간을 배열로(H94), 인자 범위의 `ValueError`와 `nobs = 0`의 NaN(H97), 차의 점수 검정의 참 최우추정(H98: statsmodels의 삼차식 버그), E-test의 모든 결과의 합(H99), `quantile_test`의 `nan_policy`와 스칼라 `q`·`p`(H100), `'binom_test'` 구간의 정의(H101), 0인 표준오차의 NaN(H24), 개수의 배치(statsmodels의 함수).
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
cramervonmises(rvs, cdf, /, *, axis=None, nan_policy='propagate', keepdims=False)
```
- 결과는 모두 `statistic pvalue`이고, `cramervonmises` 말고는 둘로 풀린다(scipy와 같다). 다른 필드: `chisquare`·`power_divergence`의 `df`(k − 1 − `ddof`; scipy의 결과에는 없다. H29처럼 튜플에 들지 않는다, H43), `ks_1samp`의 `statistic_location`, `statistic_sign`(scipy와 같다; 검정 하나면 int, 배치면 int64 배열, NaN인 줄은 0), `method`.
- `kstest`(한 표본·두 표본을 고르는 scipy의 감싸개)는 없다(H34): 한 표본은 `ks_1samp`, 두 표본은 `ks_2samp`(H4)이다.
- `chisquare`·`power_divergence`: `ddof`는 기대 빈도를 맞춘 모수의 수다(§1.1; int, 음수도 받는다). `f_exp=None`이면 기대 빈도는 관측의 평균(double-double)이고, 주면 `f_obs`와 브로드캐스트되는 배열이다(짝지은 표본이라 `nan_policy='omit'`은 쌍을 뺀다). `lambda_`는 None(Pearson), 실수, scipy의 이름(`'pearson'` 1, `'log-likelihood'` 0, `'freeman-tukey'` −1/2, `'mod-log-likelihood'` −1, `'neyman'` −2, `'cressie-read'` 2/3)이다. `chisquare(sum_check=True)`와 `power_divergence`(scipy처럼 언제나)는 `f_exp`를 주면 각 표본의 두 합이 상대 2⁻²⁶(scipy의 √ε) 안에서 같은지 보고, 아니면 `ValueError`다. 관측이 0인 칸은 그 극한이다: λ > −1이면 유한한 값, λ ≤ −1이면 inf(`[scipy와 다름]`: scipy는 −1 < λ < 0과 λ < −1에서 0·inf의 NaN). 자유도가 0 이하이면 p값은 NaN이다.
- `skewtest`(n < 8은 NaN), `kurtosistest`(n < 5는 NaN), `normaltest`(n < 8은 NaN): scipy의 D'Agostino·Anscombe–Glynn 변환이다. 표본 왜도가 정확히 0이면 `skewtest`의 통계량은 0, p값은 1이다(`[scipy와 다름]`: scipy 1.18.1은 그 경우 y = 0을 1로 바꿔 0이 아닌 값을 낸다). `normaltest`의 p값은 `ChiSquared(df=2).ccdf`다.
- `jarque_bera`(H17, H18, H20, H38): 커널은 htest의 것이고 `sd.regression`의 `r.jarque_bera()`가 그것을 부른다. 통계량은 평균을 뺀 2·3·4차 합(double-double)에서 n/6·(S² + (K − 3)²/4), p값은 `ChiSquared(df=2).ccdf(jb)`다(닫힌 꼴 e^(−JB/2)는 그와 1 ulp 다를 수 있어 쓰지 않는다). 그래서 `sd.htest.jarque_bera(r.resid)`와 OLS의 `r.jarque_bera()`는 비트까지 같다. 예외는 완전 적합이다: regression은 잔차를 0으로 보아 NaN을 내고(G95, G117), `jarque_bera(r.resid)`는 반올림 크기의 잔차로 계산한다. WLS·GLS는 백색화한 잔차의 값이다(regression이 `wresid`를 두면 그 쓰인 행을 넘긴다). 분산 0의 판정은 sd.stats `skew`의 규칙이다(m2 ≤ (ε·평균)²이면 NaN; H20). 결과는 `sd.regression.JarqueBeraResult`로 공개된 htest의 클래스다(H18, H39; regression G165).
- `shapiro`(H37): Royston(1995) AS R94의 W와 p값이다(scipy·R과 같은 알고리즘, 같은 상수). n < 3은 NaN이다. n = 3은 scipy처럼 W를 [0.75, 1]로 자르고 정확한 분포의 p값이다. n의 위 한계는 없다(scipy는 n > 5000에서 경고, R은 에러).
- `anderson`(H36): scipy 1.17부터의 꼴만이다(`statistic pvalue`; 옛 `critical_values`, `significance_level`, `fit_result`는 없다). `dist`는 sd.stats의 분포 클래스다: None과 `sd.stats.Normal`(평균과 표준편차(n − 1)), `sd.stats.Exponential`(평균; 위치 0), `sd.stats.Logistic`, `sd.stats.Gumbel`(최댓값의 Gumbel, scipy의 `'gumbel_r'`; 둘 다 최우추정을 끝자리까지 푼다). scipy의 문자열 이름, 클래스가 아닌 분포 객체, 다른 클래스는 받지 않는다(scipy `'weibull_min'`의 Weibull은 위치를 포함한 최우추정과 함께 H9의 `goodness_of_fit`에서 더한다, H40). `method`는 `'interpolate'`이고 기본값이다(`[scipy와 다름]`: scipy 1.18은 주지 않으면 경고와 함께 옛 꼴을 낸다). p값은 scipy의 표(Stephens)와 `np.interp` 그대로라 표의 수준 범위(정규·지수 0.01–0.15, 로지스틱 0.005–0.25, Gumbel 0.01–0.25) 밖으로 나가지 않는다. `MonteCarloMethod`는 H9에서 더한다. n < 2와 상수 표본은 NaN이다.
- `ks_1samp`·`cramervonmises`의 `cdf`(H35): sd.stats의 분포 객체(`sd.stats.Normal(mu=0, sigma=1)`; 그 `cdf`)나 `cdf(x)`를 돌려주는 호출 가능한 객체다. 정렬한 표본을 배치의 모양(…, n)으로 한 번 넘기고, 같은 모양의 수 배열을 받아야 한다. scipy의 분포 이름 문자열(`'norm'`)과 `args=()`는 받지 않는다(optimize의 O12와 같다).
- `ks_1samp`: `method`는 `'auto'`, `'exact'`, `'approx'`, `'asymp'`(scipy의 이름과 뜻)이다. 양쪽 검정은 `'auto'`가 언제나 `'exact'`(scipy와 같다)이고, 한쪽 검정은 `method`와 상관없이 Smirnov의 정확 분포다(scipy와 같다). 결과의 `method`는 쓴 분포다(H12; 한쪽 검정은 `'exact'`). `'exact'`의 양쪽 분포는 n·D²가 6.5보다 작고 n이 약 1000 이하인 곳에서 double-double의 Durbin 행렬, 그 위의 꼬리는 Smirnov의 정확한 한쪽 꼬리의 두 배(빠뜨린 두 꼬리의 확률은 1e-17 아래)이고, 행렬이 상한(곱셈 4·10⁷번)을 넘는 큰 n의 가운데는 Pelz–Good 근사다(scipy·Simard–L'Ecuyer와 같은 선택이며 계약이 느슨한 곳; docs/design/htest.md §4). 계산량이 언제나 유한해 `ValueError`는 없다(H42).
- `cramervonmises`: p값은 Csörgő–Faraway의 유한 표본 근사식의 값이고 scipy처럼 0 아래는 0이다. 1 − cdf라 절대 오차가 약 1e-16이다(H41). n < 2는 NaN이다.

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
median_test(*samples, axis=None, ties='below', correction=True, lambda_=1, nan_policy='propagate', keepdims=False)
page_trend_test(data, /, *, ranked=False, predicted_ranks=None, method='auto')
```
- **결과**: 모두 `statistic pvalue`이고 `page_trend_test` 말고는 둘로 풀린다(scipy와 같다). 다른 필드:
  - `mannwhitneyu`·`wilcoxon`: `method`, 정규 근사를 쓴 때의 `zstatistic`(H47; scipy 1.18.1은 사용자가 `method='asymptotic'`을 준 때만 둔다). 배치에서는 한 줄이라도 정규 근사면 있고 정확한 줄은 NaN이다. `wilcoxon`의 양쪽 검정은 scipy처럼 −|z|다.
  - `ansari`·`page_trend_test`: `method`.
  - `kruskal`·`friedmanchisquare`: `df`(k − 1), `brunnermunzel`: `distribution='t'`일 때 Satterthwaite의 `df`(H51, H29·H43의 꼴: scipy의 결과에는 없고 튜플에 들지 않는다).
  - `median_test`: `statistic pvalue median table df`이고 넷(`statistic pvalue median table`)으로 풀린다. `table`은 (…, 2, k)의 int64 배열(첫 줄이 중앙값 위, 둘째 줄이 아래), `df`는 k − 1이다.
- **순위**: 중간 순위는 정확하다(값을 정렬하고 동점 t개가 위치 b부터면 (2b + t + 1)/2). 순위 합, U, T⁺, AB, L은 정확한 정수(의 반)이고, 정확 검정은 정수로 견준다. ±inf는 순서대로 순위를 받는다.
- **`method`**: `mannwhitneyu`·`wilcoxon`·`ansari`·`page_trend_test`는 `'auto'`, `'exact'`, `'asymptotic'`이다. 결과의 `method`에 쓴 방법을 적는다(H12). `'auto'`의 문턱은 scipy 1.18.1의 것이다(H46): `mannwhitneyu`는 한 표본이 8 이하이고 동점이 없으면 정확, `wilcoxon`은 차가 50개 이하이고 동점과 0이 없으면 정확이고 있으면 13개 이하에서 정확(scipy는 같은 분포를 부호의 나열로 계산한다), `ansari`는 두 표본이 55 미만이고 동점이 없으면 정확, `page_trend_test`는 n ≤ 8이고 m ≤ 12(n = 3이면 m ≤ 20)이면 정확이다. 재표본 객체는 H9에서 더한다.
- **동점이 있을 때의 `'exact'`**(H45, `[scipy와 다름]`): 관측한 중간 순위의 조건부 정확 분포다(순위들의 모든 배치, `wilcoxon`은 0이 아닌 차의 모든 부호를 같은 확률로 센다). scipy 1.18.1의 `'exact'`는 동점이 없는 분포를 쓴다(통계량을 내림·올림해 넣는다: 그 데이터의 귀무분포가 아니다). R의 `wilcox.test(exact = TRUE)`는 같은 조건부 분포다. 동점이 있으면 분포가 대칭이 아닐 수 있어(동점 묶음이 거울처럼 놓이지 않는다) 양쪽 p값은 min(1, 2·min(P(S ≤ s), P(S ≥ s)))다(동점이 없으면 scipy의 2·P(U ≥ max(U₁, U₂))와 같다). `page_trend_test`의 `'auto'`는 scipy처럼 모양만 보므로 동점이 있는 작은 표도 정확(조건부)이다.
- **정확한 방법의 상한**(H13): 작업 10⁹번 또는 동시에 드는 개수 10⁷개다. 넘으면 `'exact'`는 `ValueError`(`mannwhitneyu(): method='exact' needs about 3.2e+13 operations for this input, more than its cap of 1e+09; use method='asymptotic'`), `'auto'`는 정규 근사다. 동점이 없으면 셈은 128비트 정수(가우스 이항계수, 부분합)이고 전체 개수가 2¹²⁴ 아래인 곳이다: `mannwhitneyu`는 한 표본 8개에 합 약 17만 개까지, 두 표본이 같은 크기면 합 127개까지, `wilcoxon`은 차 124개까지, `ansari`는 합 127개까지. 그 위와 동점이 있는 조건부 분포는 double-double의 셈이다(단계마다 양수의 덧셈이라 상대 오차가 단계 수 × 2⁻¹⁰⁴). 상한 안의 크기는 `wilcoxon`이 차 약 1250개, 동점이 있는 `mannwhitneyu`·`ansari`가 (값의 수) × (작은 표본의 크기) × (합의 범위) ≤ 10⁹이다(두 표본 각 25개에 0.3–0.7 ms, 각 40개에 2–4 ms).
- `mannwhitneyu`: `statistic`은 x의 U₁(x가 y보다 큰 쌍의 수, 동점은 반)이다. 정규 근사는 scipy의 동점 분산 보정과 `use_continuity`의 0.5다.
- `ranksums`: scipy가 legacy로 표시한 함수이지만 둔다(H50). z = (R₁ − n₁(N + 1)/2)/√(n₁n₂(N + 1)/12)이고 동점·연속성 보정이 없다(scipy와 같다). 보정이 있는 검정은 `mannwhitneyu`다.
- `wilcoxon`: `y`가 없으면 `x`가 차다. 차 x − y는 double-double로 정확하게 든다(1e16과 작은 값의 차가 반올림되어 동점이 생기거나 순서가 바뀌지 않는다; scipy는 double로 뺀다). `zero_method`는 `'wilcox'`(0을 뺀다), `'pratt'`(0까지 순위를 매기고 0의 순위는 빼놓는다), `'zsplit'`(0의 순위를 T⁺와 T⁻에 반씩)이다. R의 `wilcox.test`는 0을 Pratt처럼 다룬다. `statistic`은 양쪽 검정이면 min(T⁺, T⁻), 한쪽이면 T⁺다. 0이 아닌 차가 없으면(`'wilcox'`) NaN이다.
- `kruskal`·`friedmanchisquare`: scipy의 동점 보정이 든 카이제곱(k − 1)이다. H와 Friedman 통계량은 정수인 순위 합의 편차에서 계산한다(scipy·R의 ΣRᵢ²/nᵢ − 3(N + 1)은 상쇄한다). `friedmanchisquare`는 표본이 셋 이상, 같은 길이(짝)이고 `nan_policy='omit'`은 NaN이 든 블록을 뺀다. 표본이 모자라면 `kruskal(): at least two samples are required, not 1`, `friedmanchisquare(): at least three samples are required, not 2`(`TypeError`, H30의 꼴)다. 값이 모두 같으면 NaN이다.
- `brunnermunzel`: 통계량은 scipy의 것(y가 클수록 양수), p값은 −W에서 t(Satterthwaite의 자유도) 또는 정규다. 두 표본의 위치 분산이 모두 0이면(완전 분리, 모두 동점) 통계량·p·`df`가 NaN이다(`[scipy와 다름]`: scipy는 ±inf와 경고; H24의 규칙, H52). 표본 하나가 2 미만이어도 NaN이다.
- `ansari`: AB = x의 min(r, N + 1 − r)의 합. 'greater'는 AB의 아래 꼬리다(scipy와 같다: x의 척도가 크면 AB가 작다). 정규 근사의 분산은 scipy의 것(동점이 있으면 점수 제곱합의 꼴)이다.
- `mood`: scipy의 Mood 제곱 순위 검정(동점이면 Mielke의 T와 분산)의 z와 정규 p값이다. N < 3이면 NaN이다.
- `median_test`(H49, `[scipy와 다름]`: `axis`·`keepdims`·배치가 있다): 모은 표본의 중앙값(짝수 개면 가운데 두 값의 정확한 평균; scipy는 double로 반올림해 그 반올림과 같은 값을 "같다"로 센다), 그 위·아래 개수의 2 × k 표(`ties`: 같은 값을 `'below'`, `'above'`, `'ignore'`), 표의 scipy `chi2_contingency`(k = 2이고 `correction`이면 Yates, `lambda_`는 `power_divergence`의 이름이나 수)다. 행이나(`'ignore'`면) 열이 빈 표는 통계량·p가 NaN이다(H52; scipy는 `ValueError`). `median`은 그 중앙값을 double로 반올림한 값이다. NaN인 줄(propagate)의 표는 0이다(`ks_1samp`의 `statistic_sign`과 같다).
- `page_trend_test`: `data`는 (…, m, n)이고 마지막 두 축이 표(m 블록 × n 처리), 앞 축이 배치다(H9). L = Σⱼ pⱼCⱼ(Cⱼ는 행마다의 중간 순위의 열 합, p는 `predicted_ranks`, 기본 1..n). p값은 한쪽(L이 그만큼 이상)이다. `ranked=True`면 `data`를 이미 순위로 본다(1..n 안이어야 하고, `'exact'`에는 정수나 반정수여야 한다). 정확 분포는 행마다의 순열 분포를 합성곱한 것이고 동점이 없으면 (m, n)마다 한 번 만들어 둔다. 정규 근사는 scipy의 것(동점 보정 없음)이다. NaN이 있으면 `page_trend_test(): 'data' must not contain NaN`, m < 2나 n < 3이면 `page_trend_test(): 'data' must be at least a 2 x 3 table (2 blocks, 3 treatments), not 3 x 2`(`ValueError`)다.
- **퇴화한 입력**(H11, H52): 빈 표본, 모든 값이 같아 분산이 0인 줄, propagate의 NaN은 NaN이다(경고 없음). scipy가 0과 1을 내는 곳(`wilcoxon(zeros)`), inf를 내는 곳(`brunnermunzel`의 완전 분리), `ValueError`인 곳(`median_test`의 빈 행)도 NaN이다.
- scipy·R과의 값의 차이(docs/design/htest.md §7): scipy의 `ansari` 정확 p값은 N = 45–108에서 6.5e-9–2.6e-7 틀리고(`gscale`), `wilcoxon`의 정확한 꼬리는 1 − cdf라 n = 60의 2⁻⁵⁹에서 0이다. 여기서는 정수의 셈이다.

## §5. 적합도 2: 표본끼리 (H4)
```
ks_2samp(data1, data2, /, *, axis=None, alternative='two-sided', method='auto', nan_policy='propagate', keepdims=False)
cramervonmises_2samp(x, y, /, *, axis=None, method='auto', nan_policy='propagate', keepdims=False)
anderson_ksamp(samples, /, *, variant='midrank', method=None)
epps_singleton_2samp(x, y, /, *, axis=None, t=(0.4, 0.8), nan_policy='propagate', keepdims=False)
```
- **결과**: 모두 `statistic pvalue`이고 `cramervonmises_2samp` 말고는 둘로 풀린다(scipy와 같다; `anderson_ksamp`은 H57). 다른 필드: `ks_2samp`의 `statistic_location`·`statistic_sign`(scipy와 같다; 검정 하나면 int, 배치면 int64 배열, NaN인 줄은 0)과 `method`, `cramervonmises_2samp`의 `method`(H12; 배치면 줄마다의 중첩 리스트, H48), `epps_singleton_2samp`의 `df`(H59).
- `ks_2samp`: D는 두 경험 분포 함수의 차의 최댓값(`'greater'`는 D⁺ = max(F₁ − F₂), `'less'`는 D⁻)이고 1/lcm(n₁, n₂)의 정확한 배수를 한 번 반올림한 값이다. `statistic_location`은 D에 처음 닿는 값, `statistic_sign`은 D⁺면 +1, D⁻면 −1이다(양쪽은 D⁻ > D⁺일 때만 −1). `method`는 `'auto'`, `'exact'`, `'asymp'`다. `'auto'`는 max(n₁, n₂) ≤ 10000이면 정확이다(scipy의 `MAX_AUTO_N`; scipy는 정확 계산이 double을 넘치면 경고와 함께 점근으로 넘어가지만 여기서는 확률로 세어 넘치지 않으므로 언제나 정확이다). 점근은 scipy의 것(양쪽은 한 표본 `kstwo`를 n = round(n₁n₂/(n₁ + n₂))에서, 한쪽은 Hodges의 식)이다. 정확한 방법의 상한은 작업 10⁹번이고(H13; `'auto'`의 범위 안은 언제나 상한 안이다), 넘으면 `ks_2samp(): method='exact' needs about … more than its cap of 1e+09; use method='asymp'`(`ValueError`)다. 빈 표본은 NaN이다(scipy는 `ValueError`).
- **동점이 있을 때의 `'exact'`**(H55, `[scipy와 다름]`): 관측한 동점의 조건부 정확 분포다(라벨의 모든 배치를 같은 확률로 세고, 경험 분포 함수는 같은 값의 묶음 끝에서만 본다). R 4.3부터의 `ks.test(exact = TRUE)`(`psmirnov`)와 같은 분포다. scipy 1.18.1의 `'exact'`는 동점을 무시한 분포라 값이 다르다: x = [1, 2, 2, 3, 3, 3], y = [2, 3, 4, 4, 5]에서 sd와 R은 0.108225(= 25/231), scipy는 0.238이다. 동점이 없으면 같은 분포다. `'auto'`가 정확을 고르면 동점이 있어도 이 조건부 분포다.
- `cramervonmises_2samp`: T = U/(kN) − (4k − 1)/(6N)(k = n_xn_y, N = n_x + n_y, U는 중간 순위의 Anderson 식 10)이고 정확한 유리수를 한 번 반올림한 값이다. `method`는 `'auto'`(max(n_x, n_y) ≤ 20이면 정확, scipy의 문턱; H56), `'exact'`, `'asymptotic'`이다. `'exact'`는 동적 계획이라 scipy의 문턱을 넘어 표본마다 약 100개까지 쓸 수 있다(H56, `[scipy와 다름]`: scipy의 `'exact'`는 모든 부분합을 나열해 40 × 35에 몇 초다). 상한은 합친 목록 항목 10⁹개와 동시에 드는 항목 10⁷개이고, 넘으면 `method='asymptotic'`을 알려 주는 `ValueError`다. 동점이 있으면 `'exact'`는 관측한 동점의 조건부 정확 분포다(H58, `[scipy와 다름]`: scipy는 동점이 없는 분포에 중간 순위의 U를 넣는다). 점근 p값은 scipy의 것(극한 분포의 1 − cdf, 0 아래는 0)이라 절대 오차가 약 1e-16이다(H60, H41과 같다). 표본 하나가 2 미만이면 NaN이다.
- `anderson_ksamp`: `samples`는 표본(각각 펼친다) 둘 이상의 리스트나 튜플이다(H61; scipy가 받는 2차원 배열은 받지 않는다. 아니면 `anderson_ksamp(): 'samples' must be a list or tuple of samples, not numpy.ndarray`, 하나면 `anderson_ksamp(): at least two samples are required, not 1`, 둘 다 `TypeError`; H30의 꼴). scipy 1.17부터의 `variant` 꼴만(`'midrank'`(동점을 다루는 Scholz–Stephens의 A2akN), `'right'`, `'continuous'`)이고 옛 `midrank`와 `critical_values`는 없다. 기본값은 `'midrank'`다(H57, `[scipy와 다름]`: scipy 1.18.1은 `variant`를 주지 않으면 경고하고 옛 꼴을 낸다). 결과 `statistic pvalue`는 둘로 풀린다(H57: scipy 1.18.1의 실제 동작; scipy의 문서는 풀리지 않는다고 적는다). `statistic`은 표준화한 (A2kN − (k − 1))/σ_N, `pvalue`는 Scholz–Stephens 표의 보간(scipy와 같다: 표의 범위 위는 0.25, 아래는 0.001에서 자른다; scipy는 그때 경고한다)이다. `axis`·`nan_policy`·배치는 없고(scipy와 같다) NaN이 있으면 결과가 NaN이다. 빈 표본, 서로 다른 값이 하나뿐, 모두 4개 미만은 NaN이다(scipy는 `ValueError`). `method=None`만 받고(아니면 `anderson_ksamp(): 'method' must be None, not str`, `TypeError`) `PermutationMethod`는 H9에서 더한다.
- `epps_singleton_2samp`: scipy의 정의(합친 표본의 IQR/2로 나눈 점 t에서의 경험 특성함수, 편향 공분산의 의사역행렬)이고 표본이 25 미만이면 작은 표본 보정을 곱한다. p값의 자유도는 scipy처럼 공분산의 의사역행렬의 계수(보통 2·len(t))이고 결과의 `df`(float)다(H59, `[scipy와 다름]`: scipy의 결과에는 없다; H29·H43·H51의 꼴로 튜플에 들지 않는다). `t`는 양의 유한한 수 하나나 그 1차원 시퀀스다(아니면 `epps_singleton_2samp(): 't' must hold positive finite numbers`, `ValueError`). 표본 하나가 5 미만, 유한하지 않은 값, IQR이 0이면 NaN이다(H11; scipy는 경고).
- scipy·R과의 값의 차이(docs/design/htest.md §7): scipy의 점근 `ks_2samp`은 n > 140의 `kstwo`(Pelz–Good)라 3e-7쯤, 점근 `cramervonmises_2samp`의 꼬리는 급수를 1e-7에서 멈춰 그만큼, 오프셋 10⁸의 `epps_singleton_2samp`은 1e-9 틀린다. R의 `ks.test`는 정확 분포가 sd와 1e-15 안이고 깊은 꼬리에서 0이다.

## §6. 상관 (H5)
```
pearsonr(x, y, /, *, axis=None, alternative='two-sided', method=None, nan_policy='propagate', keepdims=False)
spearmanrho(x, y, /, *, axis=None, alternative='two-sided', method=None, nan_policy='propagate', keepdims=False)
spearmanr(a, b=None, /, *, axis=None, alternative='two-sided', nan_policy='propagate')
pointbiserialr(x, y, /, *, axis=None, nan_policy='propagate', keepdims=False)
kendalltau(x, y, /, *, axis=None, method='auto', variant='b', alternative='two-sided', nan_policy='propagate', keepdims=False)
weightedtau(x, y, /, *, axis=None, rank=True, weigher=None, additive=True, nan_policy='propagate', keepdims=False)
somersd(x, y=None, /, *, alternative='two-sided')
chatterjeexi(x, y, /, *, axis=None, y_continuous=False, method='asymptotic', nan_policy='propagate', keepdims=False)
```
- **결과**: 모두 `statistic pvalue`이고 `somersd` 말고는 둘로 풀린다(scipy와 같다). 별칭 `correlation`은 없다(H65). 다른 것: `pearsonr`은 `conf_int(alpha=0.05)`(Fisher z; 재표본 구간은 H9), `kendalltau`는 `method` 필드(H12; 배치면 줄마다, H48), `somersd`는 `statistic pvalue table`이고 풀리지 않는다(`table`은 int64 배열).
- 짝지은 표본이라 x와 y는 표본 축의 길이가 같아야 한다(§1.2). `nan_policy='omit'`은 NaN이 든 쌍을 뺀다. 쌍이 2개 미만이면 NaN이다(scipy의 `pearsonr`은 `ValueError`, 다른 함수는 경고와 NaN).
- `pearsonr`: r = Sxy/√(Sxx·Syy)를 double-double의 합에서 계산하고(값마다 x와 y를 따로 2의 거듭제곱 단위로 옮겨 1e±300도 넘치지 않는다), p값은 scipy의 r의 베타 분포(양쪽은 I_q((n − 2)/2, 1/2), q = 1 − r²)다. q를 반올림한 r에서 만들지 않고 r² < 1/2이면 double-double r에서, 그 위는 중심화한 y를 x에 사영한 잔차에서 만든다. 그래서 |r| ≈ 1에서도 p값이 맞다(`[scipy와 다름]`: scipy는 y = x + 10⁻⁷·잡음(n = 10)에서 0.11, 10⁻⁹에서 p = 0(참 1.2e-74) 틀린다; docs/design/htest.md §7). `conf_int`는 scipy의 Fisher z 꼴 tanh(atanh r ∓ q_α/√(n − 3))(한쪽이면 다른 끝이 ∓1, n ≤ 3이면 (−1, 1))이고 atanh r은 1 − |r| = q/(1 + |r|)에서 계산한다. 표본 하나가 상수이면(분산 0의 판정은 H20: m2 ≤ (ε·평균)²; scipy는 정확히 같은 값만 상수로 보고 거의 상수이면 경고와 함께 값을 낸다) 또는 inf가 있으면 NaN이다. n = 2는 r = ±1, p = 1(scipy의 약속)이다. `method`는 None만 받는다(아니면 `pearsonr(): 'method' must be None, not str`, `TypeError`; 재표본은 H9). `nan_policy`·`keepdims`는 scipy 1.18.1의 `pearsonr`에 없다(`[scipy와 다름]`: 모든 표본 함수의 꼴, §1.1).
  - **`sd.corrcoef`와의 차이**(H66): `pearsonr`의 r은 htest의 정확한 커널(double-double)이고 `sd.corrcoef`(array 영역, numpy 꼴)는 numpy와 같은 계산이다. 둘 다 자기 정확도 계약 안이라 같은 데이터에서 끝자리가 다를 수 있다(무작위 문제에서 상대 1e-13 안, 비트까지 같지 않은 곳이 있다). 정확한 값이 필요하면 `pearsonr`을 쓴다.
  - scipy의 `linregress`는 sd에 없다(regression G188). 기울기 = 0의 t 검정은 `sd.regression.ols(y, x).pvalues[1]`이고 이 p값과 같은 검정이다(계약 안에서 같고 비트는 같지 않다; §13).
- `pointbiserialr`: `pearsonr`의 통계량과 양쪽 p값이다(x가 두 값의 변수). 구간은 없다(scipy와 같다).
- `spearmanrho`(H62): scipy 1.18의 새 꼴이다(짝지은 표본의 브로드캐스트와 배치). 통계량은 정확한 중간 순위의 `pearsonr`이고 p값은 그 베타 분포(t 근사)다. 정확 분포는 없다(H63; scipy에도 없다. R `cor.test(method = 'spearman')`의 기본은 n < 1290에서 동점이 없으면 AS 89의 정확 분포라 작은 표본에서 다르다: n = 10에서 1.5배; `exact = FALSE`로 같은 값이다). ±inf는 순서대로 순위를 받는다. `method`는 None만 받는다(H9).
- `spearmanr`(H62): scipy의 옛 꼴이다. `axis`는 변수의 축이다: 0이면 `a`의 열마다, 1이면 행마다, None(기본, H73·H9; scipy는 0)이면 `a`를 펼친 하나가 변수다. `b`를 주면 같은 규칙의 변수를 더한다. 2차원보다 큰 입력은 `ValueError`, `axis`는 None·0·1만이다. 변수가 둘이면 `statistic`·`pvalue`가 수, 셋 이상이면 (k, k) 배열이고(대각은 1과 0) (i, j) 원소는 `spearmanrho(변수 i, 변수 j)`와 같은 비트다. 배치는 없다(`axis`가 변수의 축이라; `repr`는 `SpearmanrResult(batch shape (k, k))`). `nan_policy`: `'propagate'`면 NaN이 든 변수의 행과 열이 NaN, `'omit'`이면 쌍마다 NaN이 든 관측을 뺀다(scipy의 mstats와 같다), `'raise'`. n = 2의 p값은 1이다(`spearmanrho`와 같다; scipy의 옛 꼴은 NaN). 변수가 하나면 `spearmanr(): needs at least 2 variables to compare, not 1`(`ValueError`)다.
- `kendalltau`: `variant`는 `'b'`(동점 보정)·`'c'`(Stuart), `method`는 `'auto'`, `'exact'`, `'asymptotic'`이다. 일치·불일치 쌍은 병합 정렬로 세는 정확한 정수이고 τ는 그 몫을 한 번 반올림한 값이다. `'auto'`는 scipy의 문턱(H46: 동점이 없고 n ≤ 33이거나 불일치·일치 쌍이 하나 이하면 정확)이다. 정확 분포는 역순 쌍의 수(Mahonian 수)의 꼬리다(n ≤ 33은 128비트 정수, 그 위는 양수만 더하는 double-double 확률; double 범위 밑의 꼬리도 서브노멀까지 맞다). 동점이 있는 `'exact'`는 `kendalltau(): method='exact' needs samples without ties; use method='asymptotic'`(`ValueError`; scipy와 같은 규칙, H68: x와 y 모두의 동점에 조건부인 정확 분포는 일반적으로 다루기 어렵다), 작업 n(c + 1)이 10⁹를 넘으면(H13; τ ≈ 0에서 n ≈ 1500) `kendalltau(): method='exact' needs about … use method='asymptotic'`(`ValueError`)다. 정규 근사는 scipy의 동점 분산이고 n = 2에서도 값을 낸다(`[scipy와 다름]`: scipy는 분산의 0/0 항으로 NaN). 한 표본이 상수이면 NaN이다. R `cor.test(method = 'kendall', exact = TRUE)`는 n ≤ 33에서 2e-9까지 맞고 그 위는 1 − cdf라 깊은 꼬리에서 틀린다(docs/design/htest.md §7).
- `weightedtau`: Vigna의 가중 τ(scipy와 같은 알고리즘, 합은 double-double)다. `pvalue`는 언제나 NaN이다(H64; scipy가 정의하지 않는다). `rank`: True(두 사전식 순위 (x, y)·(y, x)의 τ의 평균), None((x, y)의 순위), False(원소의 위치), 또는 x·y와 짝지은 순위의 표본(scipy처럼 그 값의 순서만 쓴다; 유한한 수여야 한다: `weightedtau(): 'rank' must hold finite numbers`). `weigher`는 None(쌍곡선 1/(r + 1)) 또는 순위 r(0이 가장 중요)의 호출 가능한 객체(순위 0..n − 1에서 한 번씩 부른다), `additive`는 교환의 가중치를 두 가중치의 합(True)이나 곱으로 한다. NaN은 `nan_policy`를 따른다(H70, `[scipy와 다름]`: scipy는 NaN이 든 표본을 NaN이 가장 큰 순위로 바꿔 값을 낸다).
- `somersd`: Somers의 D(Y|X)다. `x`와 `y`가 두 순위(1차원, 같은 길이)이면 교차표(x의 서로 다른 값이 행, y의 값이 열, 오름차순)를 만들고, `y`가 None이면 2차원 `x`가 표다(음이 아닌 정수 개수, 0이 아닌 칸이 둘 이상: scipy와 같은 검사; 2차원 `x`에 `y`를 주면 `somersd(): 'y' must be None when 'x' is a table`(`ValueError`), scipy는 `y`를 무시한다). 일치·불일치의 가중합과 점근 표준오차의 합은 표의 누적합에서 정확한 정수다(scipy는 칸마다 블록을 다시 더한다). p값은 Z = (P − Q)/√(4S)의 정규 근사(scipy)다. 표의 행이나 열이 하나뿐이면 NaN, S = 0이면 p값이 NaN이다(H69, `[scipy와 다름]`: scipy는 Z = inf, p = 0). NaN이 있으면 `somersd(): 'x' and 'y' must not contain NaN`(`ValueError`)이고, 두 표본의 표가 10⁷칸을 넘으면 `somersd(): the table of … distinct values has more cells than its cap of 1e+07`(`ValueError`; H72: 표가 결과의 필드라 메모리가 정해져야 한다)이다. `axis`·배치·`nan_policy`는 없다(scipy와 같다).
- `chatterjeexi`: Chatterjee의 ξ(y가 x의 함수인 정도; 대칭이 아니다)와 독립성의 한쪽 정규 근사 p값(큰 ξ)이다. ξ는 정확한 유리수를 한 번 반올림하고, 귀무 분산(`y_continuous=False`면 scipy `_xi_std`의 식, True면 2/5)의 합은 정수다. x의 동점은 위치로 가른다(H71, `[scipy와 다름]`: scipy는 정렬이 임의로 가른다; Chatterjee는 무작위로 가르기를 권한다). y가 상수이면 NaN이다. `method`는 `'asymptotic'`만이고(`chatterjeexi(): 'method' must be 'asymptotic', not 'auto'`) 재표본은 H9다.
- scipy·R과의 값의 차이(docs/design/htest.md §7): scipy의 `pearsonr` p값은 |r| ≈ 1에서 0.11–1, 깊은 꼬리(10⁻²²⁰)에서 1.2e-12, `chatterjeexi`의 꼬리 p값 1.6e-12, 동점이 있는 `weightedtau` 1e-14 틀린다. R `cor.test`는 같은 Pearson의 손실(eps = 10⁻⁹에서 5.6e10배)과 오프셋의 r 2e-9, Kendall 정확 분포의 1 − cdf(n = 60에서 6.8e-14, 참 2.4e-82)다.

## §7. 분할표 (H6)
```
chi2_contingency(observed, /, *, correction=True, lambda_=None, method=None)
fisher_exact(table, /, *, alternative='two-sided', method=None)
barnard_exact(table, /, *, alternative='two-sided', pooled=True, n=32)
boschloo_exact(table, /, *, alternative='two-sided', n=32)
mcnemar(table, /, *, exact=True, correction=True)
cochrans_q(x, /)
odds_ratio(table, /, *, kind='conditional')
relative_risk(exposed_cases, exposed_total, control_cases, control_total, /)
```
- 표는 마지막 두 축이고 앞의 축이 배치다(§1.2, H9): (10, 2, 3)은 2 × 3 표 10개다. scipy의 `chi2_contingency`가 받는 n차원 표(3원 이상의 표)는 없다(`[scipy와 다름]`: 3차원 이상의 입력은 배치다). `nan_policy`는 없다: 개수는 음이 아닌 유한한 값이어야 하고(정확한 검정은 정수 값; 아니면 `fisher_exact(): 'table' must hold non-negative integer counts`, `ValueError`), 빈 표는 `fisher_exact(): 'table' must be at least a 1 x 1 table, not 0 x 2`, 2차원이 아니면 `chi2_contingency(): 'observed' must have at least 2 dimensions (the table's), not 1`(`ValueError`)이다. 표 하나는 배치 안에서나 혼자서나 같은 비트다.
- **결과**: `chi2_contingency`는 `statistic pvalue df expected_freq`이고 넷으로 풀린다(scipy의 `dof`는 `df`, H10; `df`는 float, H82; `expected_freq`는 (…, r, c)의 쓰기 금지 배열). `fisher_exact`는 `statistic pvalue`이고 둘로 풀린다. `barnard_exact`·`boschloo_exact`·`mcnemar`는 `statistic pvalue`, `cochrans_q`는 `statistic pvalue df`이고 풀리지 않는다(scipy의 dataclass, statsmodels의 Bunch). `odds_ratio`는 `statistic`과 `conf_int(*, alpha=0.05, alternative='two-sided')`, `relative_risk`는 `relative_risk exposed_cases exposed_total control_cases control_total`과 `conf_int(*, alpha=0.05)`이고 풀리지 않는다(검정이 아니라 `pvalue`가 없다). `method`는 None만 받는다(`chi2_contingency(): 'method' must be None, not str`, `TypeError`; 재표본은 H9).
- `chi2_contingency`: 기댓값 E = R_i C_j / N(정확한 double-double에서 한 번 반올림), 자유도 (r − 1)(c − 1)이다. 자유도가 1이고 `correction=True`면 Yates의 보정(각 칸을 E 쪽으로 min(1/2, |O − E|)), 통계량은 `power_divergence`의 것(`lambda_`는 그 이름이나 수, None은 Pearson)이다. 정수가 아닌 개수(가중 표)도 받는다(H80, scipy와 같다). 자유도 0(행이나 열이 하나)은 통계량 0, p = 1(scipy와 같다). 0인 행·열(기댓값 0)은 통계량·p가 NaN이다(H11, `[scipy와 다름]`: scipy는 `ValueError`). 0인 칸은 `power_divergence`처럼 극한이다(λ ≤ −1이면 +inf, p = 0; scipy는 NaN).
- `fisher_exact`: 2 × 2 표의 통계량은 표본 오즈비 ad/bc(bc = 0이면 inf), p값은 첫 칸의 초기하 분포에서 scipy의 규칙이다(양쪽: 관측 표의 확률 × (1 + 10⁻¹⁴)보다 작은 반대편 표; 최빈값과 같으면 1; R `fisher.test`는 1 + 10⁻⁷). 0인 행·열은 NaN과 p = 1(scipy). r × c 표(H77, `[scipy와 다름]`: scipy의 기본은 시드 없는 몬테카를로)는 정확한 망 알고리즘의 p값이다: 주변합이 같은 모든 표 가운데 관측 표의 확률 × (1 + 10⁻⁷) 이하인 표의 확률 합(R `fisher.test`의 규칙). 통계량은 scipy처럼 관측 표의 확률이다. 작업량이 상한(2·10⁸)을 넘으면 `fisher_exact(): the exact r x c test needs more than its cap of 2e+08 operations for this table; use chi2_contingency`(`ValueError`, H13)이고 상한에 닿는 데 몇 초다. R의 기본 workspace가 거부하는 4 × 5(N = 100) 같은 표도 정확한 값을 낸다(R은 workspace를 늘리면 7초, 1.7e-10 다른 값; docs/design/htest.md §7). 0인 행·열을 뺀 뒤 행이나 열이 하나면 (1, 1)이다. r × c 표의 `alternative`는 `'two-sided'`만이다(`fisher_exact(): 'alternative' must be 'two-sided' for a table that is not 2 x 2, not 'less'`; scipy는 None만). scipy의 기본값 `alternative=None`은 `'two-sided'`와 같고 여기서는 `'two-sided'`로 적는다.
- `barnard_exact`·`boschloo_exact`: 2 × 2 표만이고(`barnard_exact(): 'table' must be a 2 x 2 table, not 2 x 3`) 두 열이 두 이항 표본이다(scipy의 방향). `n`은 1 이상의 int이고 상한 찾기의 처음 격자 크기로만 쓴다(H78, `[scipy와 다름]`: scipy는 `shgo`의 표본 수라 결과가 `n`에 따라 다르고 상한을 놓칠 수 있다). p값은 `n`과 상관없이 장애 모수에 대한 참 상한이다(상대 2⁻⁴⁸ 안으로 증명한 값). Barnard의 기각역은 Wald 통계량을 정확한 유리수로 견준다(`[scipy와 다름]`: scipy는 반올림한 통계량을 견주어 |T| = |T_obs|인 거울 표를 빠뜨릴 수 있다: [[7, 12], [8, 3]]에서 9.5e-4 낮다). Boschloo의 통계량은 한쪽 Fisher p값이고 기각역은 그것이 관측의 (1 + 10⁻¹³)배 이하인 표(scipy의 허용오차), 양쪽은 작은 쪽의 두 배다. 0인 열은 Barnard (NaN, 1), Boschloo (NaN, NaN)(scipy). 표가 (n₁ + 1)(n₂ + 1) = 10⁷개를 넘으면 p값이 NaN이다(통계량은 낸다).
- `mcnemar`(statsmodels): 2 × 2 표의 b = table[0, 1], c = table[1, 0]이다. `exact=True`(기본, H81)는 통계량 min(b, c)와 이항(b + c, 1/2) 꼬리의 두 배(1로 자른다), `exact=False`는 (|b − c| − correction)²/(b + c)와 카이제곱(1)이다(statsmodels의 식: b = c에서도 1을 뺀다; R `mcnemar.test`는 b = c를 보정하지 않는다). b + c = 0의 카이제곱 꼴은 NaN이다(H11, `[statsmodels와 다름]`: inf와 0).
- `cochrans_q`(statsmodels): `x`는 (…, N, k)의 0·1 값(블록 × 처리)이다(다른 값은 `cochrans_q(): 'x' must hold 0 and 1 values (the successes)`; statsmodels는 가장 큰 값을 성공으로 본다). 결과 `statistic pvalue df`(k − 1)다. statsmodels의 `return_object`는 없다(언제나 결과 객체). 모든 블록이 모두 0이나 모두 1이면 NaN이다.
- `odds_ratio`(`scipy.stats.contingency`, H79): `kind='conditional'`은 Fisher의 비중심 초기하 분포의 조건부 최우추정(첫 칸이 받침의 끝이면 0이나 inf), `'sample'`은 ad/bc다. `conf_int(*, alpha=0.05, alternative='two-sided')`는 scipy의 `confidence_interval(confidence_level, alternative)`이다(조건부: 정확 구간, 표본: Woolf 구간; 한쪽은 아래 끝 0이나 위 끝 inf). 0인 행·열은 NaN과 (0, inf)다. 정수 값의 float 표를 받는다(scipy는 정수 dtype만). 배치가 있다.
- `relative_risk`(`scipy.stats.contingency`, H79): 네 개수는 int 인자다(scipy처럼; 사례 수는 0 이상, 총수는 1 이상, 사례 수 ≤ 총수: `relative_risk(): 'exposed_cases' must not exceed 'exposed_total'`). 추정값은 정확한 몫을 한 번 반올림한 값, `conf_int(*, alpha=0.05)`는 Katz 구간이고 0인 사례 수의 끝은 scipy의 것((0, NaN), (NaN, inf), (NaN, NaN))이다.
- scipy·R과의 값의 차이(docs/design/htest.md §7): scipy의 `barnard_exact`는 거울 표를 빠뜨리거나(9.5e-4) `shgo`가 상한을 놓친다(1.2e-7), `chi2_contingency`는 Cressie–Read의 0인 칸이 NaN이고 O ≈ E의 G가 1.6e-11 틀린다, 조건부 오즈비의 구간은 `brentq`의 허용오차로 8e-12다. R `fisher.test`의 r × c p값은 큰 표에서 1.7e-10, 조건부 오즈비와 구간은 `uniroot`의 허용오차로 6e-4까지 다르다.

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
- **개수**(H97): `count`·`nobs`(와 `binomtest`의 `k`·`n`)는 음이 아닌 정수 값이고 `count ≤ nobs`다(`proportions_ztest(): 'count' must not exceed 'nobs'`, `... 'count' must hold non-negative integer counts`, `ValueError`). statsmodels의 다섯 함수는 개수를 브로드캐스트한다: 결과는 그 모양의 배치이고(`repr`는 `batch shape`), 모두 스칼라면 검정 하나(Python 수)다. `proportions_ztest`·`proportions_chisquare`는 마지막 축이 표본이고 앞 축이 배치다(스칼라는 표본 하나). `nobs = 0`인 줄은 NaN이다(H11). 정수가 아닌 개수는 받지 않는다(statsmodels는 `'binom_test'` 밖에서 받는다).
- `binomtest`: `k`·`n`은 int(0 ≤ k ≤ n, n ≥ 1), `p`는 [0, 1]의 실수다. 결과 `k n alternative statistic pvalue`(scipy의 `proportion_estimate`는 `statistic`과 같은 값이라 두지 않는다: 별칭 없음, H96)와 메서드 `conf_int(*, alpha=0.05, method='exact')`(scipy의 `proportion_ci(confidence_level=0.95, method='exact')`; `'exact'`(Clopper–Pearson), `'wilson'`, `'wilsoncc'`; H16), 풀리지 않는다. 구간은 검정의 `alternative`를 따른다(`'less'`는 [0, 위 끝], `'greater'`는 [아래 끝, 1]). 양쪽 p값은 scipy 1.18.1의 규칙(반대편에서 pmf ≤ (1 + 10⁻⁷)·pmf(k)인 결과; R `binom.test`도 같다)이고 비교는 double-double log pmf와 정확한 pn으로 한다. p = 0·1은 점질량이다. scipy는 n = 10⁵에서 5.9e-14 틀린다.
- `quantile_test`: `q`는 실수, `p`는 (0, 1)의 실수다(scipy 1.18.1은 그 밖에서 NaN; `ValueError`). 결과 `statistic statistic_type pvalue`(scipy처럼 float: T1 = #(x ≤ q)와 형 1, 또는 T2 = #(x < q)와 형 2)와 `conf_int(*, alpha=0.05)`(순서통계량 x[ppf(a) − 1], x[isf(a)]; 표본 밖이면 NaN, 한쪽 검정의 열린 쪽은 ±inf), 풀리지 않는다. scipy에 없는 `nan_policy`를 두고 `q`·`p`는 스칼라다(배치는 `axis`로; H100). 빈 표본은 NaN이다. 구간은 호출 때의 표본으로 만든다(결과가 복사해 둔다).
- `poisson_means_test`: `k1`·`k2`는 int(≥ 0), `n1`·`n2`는 양의 유한한 실수, `diff`는 0 이상이다. 결과 `statistic pvalue`, 둘로 풀린다. p값은 T가 관측만큼 극단인 모든 결과의 귀무 확률이다(H99, `[scipy와 다름]`: scipy는 두 합을 [ppf(1e-10), ppf(1 − 1e-16)]로 자르고 격자와 관측의 T를 다른 식으로 계산해 결과를 빠뜨릴 수 있다: 깊은 꼬리에서 4.9e-9–9000배, 관측과 같은 극단의 결과를 빠뜨린 곳 6.6e-4 낮다). 귀무 추정 λ̂₂ ≤ 0이면 (0, 1)이다(scipy).
- statsmodels의 함수(H5): 신뢰수준은 statsmodels와 같은 `alpha=0.05`(H16), `alternative`는 `'larger'`·`'smaller'`가 아니라 `'greater'`·`'less'`(H86), 결과는 튜플 대신 결과 객체나 배열이다. 기본 방법은 statsmodels 0.15.0의 것으로 고정한다(H90).
  - `proportions_ztest`: 결과 `statistic pvalue`, 둘로 풀린다(statsmodels의 튜플 순서). 표본 하나는 `value`가 꼭 있고 분산은 표본 비율(또는 `prop_var`)의 것, 둘은 합동 비율의 것이다. `prop_var`는 None이나 [0, 1]의 실수다(H87; statsmodels의 `False`는 받지 않는다, 0은 그 값이다). 셋 이상의 표본은 `ValueError`다(statsmodels는 `NotImplementedError`). 표준오차 0은 NaN이다(H24; statsmodels는 ±inf와 0).
  - `proportions_chisquare`: 결과 `statistic pvalue df table expected_freq`이고 statsmodels처럼 `(statistic, pvalue, (table, expected_freq))`의 셋으로 풀린다(H93). `table`은 (…, k, 2)의 (count, nobs − count)다. `value=None`이면 같은 비율의 검정으로 `chi2_contingency(table, correction=False)`와 비트까지 같고(`df` = k − 1; 0인 열은 NaN, 표본 하나는 (0, 1)), `value`([0, 1])가 있으면 그 비율에 대한 검정이다(`df` = k). Yates 보정은 없다(H92; R `prop.test(correct = FALSE)`).
  - `proportion_confint`: 아래 끝과 위 끝의 (…, 2) 배열을 돌려준다(`conf_int()`과 같은 꼴, H94; 구간 하나면 (2,)이라 statsmodels의 `(low, upp)`처럼 `low, upp = ...`로 풀린다). `method`(H85)는 `'normal'`(기본), `'agresti_coull'`, `'exact'`(statsmodels `'beta'`, Clopper–Pearson), `'wilson'`, `'wilsoncc'`(scipy의 것, 연속성 보정), `'jeffreys'`, `'binom_test'`다. 양쪽 구간은 꼬리마다 alpha/2(`'binom_test'`는 양쪽 p값이라 alpha 그대로), 한쪽은 그 꼬리에 alpha다. `'less'`는 [0, 위 끝], `'greater'`는 [아래 끝, 1]이다(H95, `[statsmodels와 다름]`: statsmodels의 `'larger'`가 [0, 위 끝]이다). `'normal'`·`'agresti_coull'`·`'wilson'`은 [0, 1]로 자른다(statsmodels). `'binom_test'`는 count/nobs를 품은 채택 영역의 끝이다(H101; p값이 단조가 아니라 statsmodels의 `brentq`는 바깥의 다른 교차점에 닿기도 한다; 한쪽은 Clopper–Pearson).
  - `test_proportions_2indep`: 결과는 statsmodels의 필드(`statistic pvalue compare method diff ratio odds_ratio variance alternative value`, `method='score'`면 `prop1_null prop2_null`; 다른 방법에서는 `AttributeError`)이고 둘로 풀린다(H88). `return_results`는 없다. `compare`는 `'diff'`, `'ratio'`, `'odds-ratio'`(H89; `'or'`는 받지 않는다), `method=None`은 0.15.0의 기본(차 `'agresti-caffo'`, 비 `'log-adjusted'`, 오즈비 `'logit-adjusted'`), 차의 방법은 `'wald'`, `'agresti-caffo'`, `'score'`(`'newcomb'`은 구간만), 비는 `'log'`, `'log-adjusted'`, `'score'`, 오즈비는 `'logit'`, `'logit-adjusted'`, `'logit-smoothed'`, `'score'`다. `value`(None: 0 또는 1)는 차가 (−1, 1), 비·오즈비가 (0, ∞)다(H97). 차의 점수 검정은 참 제한 최우추정을 쓴다(H98, `[statsmodels와 다름]`: statsmodels 0.15.0의 삼차식이 틀려 귀무값이 0이 아니면 다른 값이다). 0인 개수의 log는 IEEE의 끝이다(statsmodels와 같다).
  - `confint_proportions_2indep`: (…, 2) 배열(`proportion_confint`과 같은 꼴). 기본 방법은 0.15.0의 것(차 `'newcomb'`, 비 `'log-adjusted'`, 오즈비 `'logit-adjusted'`)이고 차는 `'wald'`, `'agresti-caffo'`, `'newcomb'`, `'score'`, 비는 `'log'`, `'log-adjusted'`, `'score'`(Koopman), 오즈비는 `'logit'`, `'logit-adjusted'`, `'logit-smoothed'`, `'score'`다. 점수 구간은 그 검정을 뒤집고(|z| = z_{α/2}), z가 닿지 않는 쪽은 정의역의 끝이다(차 ±1, 오즈비 0·inf; statsmodels의 `brentq`는 에러). 차의 점수 구간은 H98 때문에 statsmodels와 다르다.

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
| `proportion_confint(method='beta')` | `method='exact'` | H85 |
| `proportion_confint(alternative='larger')`([0, 위 끝]) | `alternative='less'` | H95 |
| `proportions_ztest(prop_var=False)` | `prop_var=None` | H87 |
| `test_proportions_2indep(return_results=False)` | 없음(언제나 결과 객체, 둘로 풀린다) | H88 |
| `test_proportions_2indep(compare='or')` | `compare='odds-ratio'` | H89 |
| `binomtest(...).proportion_estimate` | 없음(`statistic`) | H96 |
| `proportions_chisquare` → `(chi2, p, (table, expected))` | 같은 셋으로 풀리는 결과 객체(`df` 필드를 더한다) | H93 |
| `proportion_confint`·`confint_proportions_2indep` → `(low, upp)` | (…, 2) 배열(풀면 `low, upp`) | H94 |
| `chi2_contingency(...).dof`, `cochrans_q` 결과의 `df` | `df` | H10 |
| scipy의 결과 형 `TtestResult`, `SignificanceResult` 등 | 함수마다 `<함수>Result`(내보내지 않음) | H6 |
| `kstest` | 없음(`ks_1samp`, `ks_2samp`) | H34 |
| 분포 인자의 이름 문자열(`'norm'`)과 `args=()` | sd.stats 분포 객체나 호출 가능한 `cdf` | H35 |
| `anderson(dist='norm')`(문자열), `method` 없으면 옛 꼴 | `dist=sd.stats.Normal`(클래스), 기본 `method='interpolate'` | H36 |
| `power_divergence`·`chisquare`의 결과 `statistic pvalue` | `statistic pvalue df` | H43(H29의 꼴) |
| `statsmodels.stats.multitest.fdrcorrection` | 없음(`multipletests(method='fdr_bh')`·`false_discovery_control`) | H8에서 정한다 |
| `statsmodels` `binom_test` | 없음(`binomtest`) | 같은 검정 |
| `mannwhitneyu`·`wilcoxon`의 `zstatistic`(`method='asymptotic'`을 준 때만) | 정규 근사를 쓴 때마다 | H47 |
| 동점이 있을 때 `method='exact'`(동점이 없는 분포) | 중간 순위의 조건부 정확 분포 | H45 |
| `kruskal`·`friedmanchisquare`·`brunnermunzel`의 결과 `statistic pvalue` | `statistic pvalue df` | H51 |
| r × c `fisher_exact`(시드 없는 몬테카를로) | 정확한 망 알고리즘(R `fisher.test`의 규칙)과 작업량 상한 | H77 |
| `barnard_exact`·`boschloo_exact`의 `n`(`shgo`의 표본 수) | 처음 격자 크기로만, p값은 참 상한 | H78 |
| `odds_ratio(...).confidence_interval(confidence_level, alternative)` | `conf_int(*, alpha=0.05, alternative='two-sided')` | H16, H79 |
| `relative_risk(...).confidence_interval(confidence_level)` | `conf_int(*, alpha=0.05)` | H16, H79 |
| `chi2_contingency`의 결과 `dof`(int), n차원 표 | `df`(float), 2차원 표와 앞 축의 배치 | H10, H82, H9 |
| statsmodels `cochrans_q(x, return_object=True)` | `cochrans_q(x)`(언제나 결과 객체) | §1.5 |
| `median_test`(`axis` 없음) | `axis`, `keepdims`, 배치, 결과의 `df` | H49 |
| 배치의 `method`(`page_trend_test`: 배치가 없다) | 줄마다의 이름의 중첩 리스트 | H48 |
| `ks_2samp(method='exact')`의 동점(동점이 없는 분포) | 관측한 동점의 조건부 정확 분포(R `psmirnov`) | H55 |
| `cramervonmises_2samp(method='exact')`(작은 표본의 나열) | 동적 계획, 큰 표본까지(`'auto'` 문턱 20은 그대로) | H56 |
| `anderson_ksamp(samples)`(`variant`가 없으면 경고와 옛 꼴) | 기본 `variant='midrank'`, `statistic pvalue`가 둘로 풀린다 | H57 |
| `ks_2samp`·`cramervonmises_2samp`의 결과(`method` 없음) | `method` 필드 | H12 |
| `epps_singleton_2samp`의 결과 `statistic pvalue` | `statistic pvalue df` | H59 |
| 결과의 `correlation`(상관 검정 모두) | 없음(`statistic`) | H65 |
| `spearmanr(a, b=None, axis=0)` | `axis=None`(펼친 하나가 변수; 열마다는 `axis=0`) | H73 |
| `weightedtau`의 NaN(가장 큰 순위로 바꾼다) | `nan_policy`(`'propagate'`면 NaN) | H70 |
| `somersd`의 S = 0(Z = inf, p = 0) | p값 NaN | H69 |
| `pearsonr`의 상수 판정(정확히 같은 값) | 분산 0의 규칙 m2 ≤ (ε·평균)² | H20 |

## §12. 마일스톤별 남은 질문 (추천을 첫째에)
docs/design/htest.md §9와 같은 목록이다. 결정이 나면 이 문서와 그 문서를 함께 고친다.

- **H1**: 정했다(H22–H33, docs/design/htest.md §1.1).
- **H2**: 정했다(H34–H44, docs/design/htest.md §1.1).
- **H3**: 정했다(H45–H54, docs/design/htest.md §1.1).
- **H4**: 정했다(H55–H61, docs/design/htest.md §1.1).
- **H5**: 정했다(H62–H73, docs/design/htest.md §1.1).
- **H6**: 정했다(H77–H84, docs/design/htest.md §1.1).
- **H7**: 정했다(H85–H101, docs/design/htest.md §1.1).
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
| 상관의 검정 | `pearsonr(x, y)` | `ols(y, x)`의 기울기의 `pvalues[1]`(기울기 = 0의 t) | 같은 검정, 다른 계산(`ols`는 QR 적합의 t, `pearsonr`은 1 − r²을 상쇄 없이 만든 베타 분포): 계약 안에서 같다. scipy `linregress`의 짝이던 `sd.regression.linregress`는 뺐다(regression G188, H67) |
| p값의 꼬리 | 모든 검정 | `pvalues`, `f_pvalue`, `lm_pvalue`, 모형 비교의 F·LR·Wald | sd.stats 분포의 `ccdf`·`logccdf`를 같은 인자로 부른다(같은 비트) |
| 신뢰구간 | `conf_int(*, alpha=0.05)` | `conf_int(*, alpha=0.05)`(G99) | 같은 이름, 같은 인자, 같은 메시지, 같은 (…, 2) 끝 순서(H16) |
| 다중 검정 보정 | `multipletests`, `false_discovery_control` | 계수 여럿의 `pvalues`, 계획 중인 단계적 변수 선택의 p값 | `multipletests(r.pvalues, axis=-1)`(배치 (…, k)도). 선택 뒤의 p값은 명목 수준을 지키지 않고, 보정도 선택의 편향을 고치지 못한다 |
| 사후 비교 | `tukey_hsd`, `dunnett` | 계획 중인 `anova` 표 | 분산분석 뒤의 쌍 비교는 htest의 것이다 |

- `durbin_watson`은 p값이 없는 통계량(속성)이고 scipy에 검정이 없어 htest에 짝이 없다. `breusch_pagan()`·White·Goldfeld–Quandt·RESET은 모형의 설계 행렬이 필요해 regression에만 있다. 잔차의 다른 정규성 검정(Shapiro–Wilk, Anderson–Darling, D'Agostino)은 잔차를 htest에 넘긴다.
- regression의 지금 결과(`JarqueBeraResult`, `BreuschPaganResult`)와 htest의 결정:

| | 지금의 regression | 결정 |
|---|---|---|
| 형 | 두 namedtuple(`stardust._results`), `sd.regression`이 내보낸다 | Jarque–Bera는 htest의 결과 클래스가 되고 `sd.regression.JarqueBeraResult`가 그 클래스를 가리킨다(H18). H2에서 커널을 옮기고(H38) 클래스를 바꿨다(H39, regression G165). `BreuschPaganResult`는 htest에 짝이 없어 regression이 정한다 |
| 풀기 | JB 둘(scipy와 같다), BP 넷(statsmodels `het_breuschpagan`의 순서) | H7과 맞는다. htest의 JB 클래스도 둘로 풀린다 |
| 필드 이름 | BP는 `lm lm_pvalue fvalue f_pvalue`(statsmodels) | scipy에 없는 결과는 statsmodels의 이름(H5)이라 맞는다 |
| 자유도 | BP에 자유도 필드가 없다 | 둔다면 카이제곱은 `df`, F는 `df_num`·`df_denom`(H19; regression이 정한다) |
| 퇴화한 입력 | 완전 적합은 0/0 NaN, BP 자유도 ≤ 0은 NaN, 경고 없음 | H11과 맞는다. 분산 0의 판정은 sd.stats `skew`의 규칙 하나로(H20): regression의 `jarque_bera()`도 m2 ≤ (ε·평균)²이면 NaN이 된다(그런 퇴화한 경우만 값이 바뀐다) |
| 배열 | 쓰기 금지, 복사 없음(G120) | 같다(H21) |

