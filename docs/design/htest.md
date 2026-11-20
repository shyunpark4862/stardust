# 가설검정 영역 설계 (sd.htest)

> 상태: **H0 완료·승인, H1 완료, 승인 대기** (2026-11-20). 큰 결정(§1.1 H1–H33)은 사용자가 정했다. H0(준비)과 H1(평균과 분산)의 결과는 §3.2, §4, §7, §8에 있다. H2–H10은 시작 전이고, 나머지는 안이다. 시작 전에 정할 것은 §9에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

`sd.htest`(가설검정: 검정통계량, p값, 신뢰구간)의 확정 사항과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처의 안은 [docs/api/htest.md](../api/htest.md)다(H0).
- 비교 대상은 scipy 1.18.1의 `scipy.stats`(검정, `scipy.stats.contingency`), statsmodels 0.15.0의 `statsmodels.stats`(`multitest`, `proportion`, `contingency_tables`), R 4.6.2의 `stats` 꾸러미(`t.test`, `wilcox.test`, `ks.test`, `fisher.test`, `prop.test`, `binom.test`, `p.adjust`, `mcnemar.test` 등)다.
- 바닥은 `sd.stats`의 분포(p값과 신뢰구간의 cdf·ccdf·icdf), `sd.special`, `sd.stats.rankdata`, 기술통계의 보상 합 커널([stats.md](stats.md) §5), `sd.optimize`(신뢰구간을 뒤집는 근 찾기, 장애 모수의 최대화, 분포의 최우추정), `sd.random`(재표본)이다.
- 이 영역은 [stats.md](stats.md) §7의 "검정은 나중에 따로"를 잇는다. 위치는 `sd.stats.tests`에서 `sd.htest`로 바뀌었다(H1).

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-20)
| 번호 | 결정 |
|---|---|
| H1 위치와 이름 | 새 최상위 모듈 `sd.htest`다. `sd.stats.tests`가 아니다. stats.md의 T2(하위 모듈 `sd.stats.tests`, 상관도 그곳)를 바꾼다(stats.md §7에 적었다). 상관(Pearson·Spearman·Kendall 등)의 계수와 p값도 여기 둔다. T3의 경계(p값을 내거나 귀무분포를 쓰면 검정, 아니면 stats; 검정이 stats를 부르는 방향만)는 그대로다 (사용자 결정 2026-11-20) |
| H2 범위 | 첫 판에 scipy.stats의 가설검정 전부를 마일스톤으로 나눠 넣는다: 모수(t, F·분산분석, 분산의 동질성, 카이제곱), 비모수 순위 검정, 적합도, 분할표, 상관, 재표본(`permutation_test`, `bootstrap`, `monte_carlo_test`)과 그 밖(`binomtest`, `poisson_means_test`, `page_trend_test`, `median_test` 등). 목록은 scipy 1.18.1의 공개 API에서 뽑았다(§1.2). `sd.stats`에 이미 있는 기술통계와 검정이 아닌 것은 뺀다 (사용자 결정 2026-11-20) |
| H3 API | scipy의 함수 이름과 인자 이름을 따르고 stardust의 규칙을 쓴다: 입력은 위치로, 옵션은 키워드로만, 별칭 없음, stardust의 메시지, `nan_policy`, `axis`와 배치는 sd.stats 기술통계의 규칙, 무작위는 `rng=`(sd.random·sd.stats와 같다). 결과는 `statistic`, `pvalue`, scipy에 있으면 신뢰구간, 자유도 등을 담은 객체이고, 필드는 읽기 전용이며 복사 규칙은 regression의 G98·G113(와 G120, API.md §2.2)을 따른다. 결과의 세부 꼴은 H6–H12와 §2.2다 (사용자 결정 2026-11-20) |
| H4 정확도 | 검정통계량과 p값은 다른 영역처럼 Arb 참값에 대한 정확도 계약을 갖는다: 보장 영역에서 상대 1e-14, 느슨한 영역은 잰 값을 문서화한다. scipy·R과의 비트 동일은 목표가 아니다. 정확한 귀무분포(작은 표본의 순위 검정, 정확 분할표 검정, Kolmogorov–Smirnov의 정확한 분포)는 Arb나 정확한 유리수 참값으로 확인한다. p값은 sd.stats·sd.special의 분포를 쓴다. scipy·R·statsmodels와의 비교는 정확도 체계(`bench/accuracy`, 비교 패키지 열)와 속도 체계(`bench/speed`)에서 한다 (§4, §7) (사용자 결정 2026-11-20) |
| H5 statsmodels 두 묶음 | scipy의 검정에 더해 statsmodels.stats의 두 묶음을 넣는다. (1) 다중 비교 보정: `multipletests`(bonferroni, sidak, holm-sidak, holm, simes-hochberg, hommel, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky)와 `fdrcorrection`(scipy `false_discovery_control`과 겹친다; 한 가지 API의 안은 §3.9). (2) 비율과 이항: `proportions_ztest`, `proportions_chisquare`, `proportion_confint`(모든 방법), 두 독립 비율의 차·비·오즈비 검정과 구간(`test_proportions_2indep`, `confint_proportions_2indep`), McNemar(`mcnemar`), Cochran Q(`cochrans_q`). TOST 동등성 검정과 검정력 분석은 지금은 뺀다(나중 후보, §1.2). 비교 대상은 statsmodels와 R(`prop.test`, `binom.test`, `p.adjust`, `mcnemar.test`)이다. 이름은 scipy에 같은 개념이 있으면 scipy의 이름, statsmodels에만 있으면 statsmodels의 이름에 stardust의 규칙을 쓴다(이름의 질문은 §9) (사용자 결정 2026-11-20) |
| H6 결과 객체 | 검정마다 읽기 전용 결과 클래스 하나이고, 모두 공통 바탕 클래스 위에 있다. 규칙은 regression·optimize의 결과와 같다: 필드는 읽기 전용 속성이다(배열 필드는 H21). 이름은 `<함수>Result`(`TtestIndResult`)다. 클래스는 공개 이름으로 내보내지 않는다(예외는 H18) (사용자 결정 2026-11-20) |
| H7 튜플 풀기 | scipy의 결과가 풀리는 곳만 scipy와 같은 개수로 풀린다(대개 `statistic, pvalue`; `chi2_contingency`는 넷). 그 밖은 필드로만 읽는다 (사용자 결정 2026-11-20) |
| H8 신뢰구간 | ~~scipy에 있는 곳은 scipy의 꼴인 메서드 `r.confidence_interval(confidence_level=0.95)`다~~ — H16이 바꿨다 (사용자 결정 2026-11-20) |
| H9 `axis` | 기본값은 None이다(sd.stats 기술통계처럼 전체를 표본 하나로). 배치는 `axis`로 한다. 표를 받는 함수(분할표 등)는 마지막 축들이 표이고 앞 축이 배치다 (사용자 결정 2026-11-20) |
| H10 자유도 | 자유도는 어디서나 `df`다(scipy `chi2_contingency`의 `dof`도 `df`). `chisquare`의 인자 `ddof`는 이름을 그대로 둔다(뜻이 다르다: 기대 빈도를 맞춘 모수의 수). F 꼴 검정의 이름은 H19가 다듬는다 (사용자 결정 2026-11-20) |
| H11 퇴화한 입력 | 관측이 너무 적거나 분산이 0인 입력의 결과는 NaN이고 경고는 없다. 배치에서는 다른 줄이 그대로 계산된다 (사용자 결정 2026-11-20) |
| H12 `'auto'`의 기록 | `method='auto'`는 실제로 쓴 방법(`'exact'`, `'asymptotic'` 등)을 결과의 `method` 필드에 적는다 (사용자 결정 2026-11-20) |
| H13 정확한 방법의 상한 | `method='exact'`의 작업량이 문서화한 상한을 넘으면 `ValueError`다(메시지가 `'asymptotic'`이나 순열 방법을 알려 준다). 모든 호출은 정해진 시간 안에 끝난다 (사용자 결정 2026-11-20) |
| H14 새 귀무분포 | 검정에만 필요한 새 귀무분포(Kolmogorov–Smirnov, 스튜던트화 범위, Fisher의 비중심 초기하)는 처음에 htest 안에만 둔다(sd.stats의 공개 분포가 아니다) (사용자 결정 2026-11-20) |
| H15 R 꾸러미 | R 대조 꾸러미(`coin`, `Exact`, `kSamples` 등)는 마일스톤마다 이름·CRAN 출처·크기를 적어 묻고 설치한다. 지금은 설치하지 않는다(base R만) (사용자 결정 2026-11-20) |
| H16 신뢰구간 (H8을 바꾼다) | 신뢰구간은 어디서나 regression의 꼴인 `r.conf_int(alpha=0.05)`다(G99: `alpha`는 키워드로만, (…, 2)의 아래·위 끝). scipy의 `confidence_interval(confidence_level=)`은 쓰지 않는다: 영역 사이의 통일이 먼저다. 신뢰수준을 받는 함수의 인자도 `alpha`다. scipy와의 차이는 docs/api/htest.md §11의 표에 적는다 (사용자 결정 2026-11-20) |
| H17 같은 값의 커널 | 두 영역이 함께 계산하는 값의 커널은 htest에 있다. Jarque–Bera와 그에 딸린 적률 검정은 htest의 것이고 regression이 htest의 커널을 부른다. p값은 regression이 지금 하는 대로 `ChiSquared(df=2).ccdf`다. 그래서 `htest.jarque_bera(x)`와 `r.jarque_bera()`는 같은 비트다. 커널(regression의 `resid_stats`의 그 부분)은 H2에서 옮기고, regression의 골든은 그대로여야 한다 (사용자 결정 2026-11-20) |
| H18 같은 검정은 같은 형 | Jarque–Bera의 결과 클래스는 htest의 것이고, regression의 공개 이름 `JarqueBeraResult`가 그 같은 클래스를 가리킨다(이름이 그대로라 깨지는 것이 없다). 그 밖의 htest 결과는 H6대로 내보내지 않는 클래스다. 이 예외는 문서에 적는다 (사용자 결정 2026-11-20) |
| H19 자유도의 이름 (H10을 다듬는다) | F 꼴 검정은 `df_num`과 `df_denom`, 카이제곱과 t는 `df`다 (사용자 결정 2026-11-20) |
| H20 분산 0의 규칙 | 분산 0의 판정은 어디서나 하나다: sd.stats `skew`의 규칙(m2 ≤ (ε·평균)²이면 NaN). regression의 `jarque_bera()`도 그렇다. regression의 결과는 그런 퇴화한 경우에만 바뀌고, 커널을 옮길 때 그런 경우가 있으면 그것만 골든을 다시 기록한다 (사용자 결정 2026-11-20) |
| H21 결과 배열 (H6을 고친다) | 배열 필드는 regression·optimize처럼 복사하지 않는 쓰기 금지 배열이다. 리스트인 설정 필드는 읽을 때마다 새 복사본이다(G113) (사용자 결정 2026-11-20) |
| H22 자유도의 형 | `df`(와 `df_num`·`df_denom`)는 언제나 float다. scipy의 Student t처럼 정수인 곳도 float다(Welch의 자유도는 정수가 아니다) (사용자 결정 2026-11-20) |
| H23 요약값의 t | `ttest_ind_from_stats`의 결과에 `df`와 `conf_int()`를 더한다. `ttest_ind`과 같은 필드다(scipy의 결과는 `statistic pvalue`뿐이다) (사용자 결정 2026-11-20) |
| H24 표준오차 0 | 표준오차가 0인 t 검정의 t와 p는 NaN이다(scipy는 inf와 0). API.md §2.2의 "정할 수 없는 값은 NaN"과 같다 (사용자 결정 2026-11-20) |
| H25 실수 스칼라 표본 | 표본 인자의 실수 스칼라는 원소 하나의 표본이다(sd.stats의 규칙 D-30). 결과는 퇴화한 입력의 NaN이다 (사용자 결정 2026-11-20) |
| H26 Welch 분산분석 | Welch의 일원 분산분석(R `oneway.test(var.equal=FALSE)`)을 넣는다. 이름은 scipy 1.18.1의 것인 `f_oneway(*samples, equal_var=False)`다 (사용자 결정 2026-11-20) |
| H27 분산분석의 자유도 | `f_oneway`의 결과에 `df_num`과 `df_denom`을 둔다(H19의 이름; scipy의 결과는 `statistic pvalue`뿐이다) (사용자 결정 2026-11-20) |
| H28 분산 0인 집단 | H24의 NaN을 다표본 검정에도 쓴다: `f_oneway`의 모든 집단이 상수(SSW = 0; scipy는 inf와 0 또는 NaN), Bartlett·Alexander–Govern·Welch 분산분석의 상수 집단(scipy `bartlett`은 inf와 0)은 통계량과 p값이 NaN이다(H11, 판정은 H20). 두 집단의 F는 t²라 H24와 같은 규칙이다 (사용자 결정 2026-11-20) |
| H29 분산의 동질성과 Alexander–Govern의 자유도 | `levene`에 `df_num`·`df_denom`, `bartlett`·`fligner`·`alexandergovern`에 `df`(k − 1)를 둔다(H10, H19의 이름; scipy의 결과에는 없다). 풀기는 scipy와 같다(H7): 새 필드는 튜플에 들지 않는다 (사용자 결정 2026-11-20) |
| H30 표본이 둘 미만 | 다표본 함수 다섯(`f_oneway alexandergovern bartlett levene fligner`)은 표본이 둘 미만이면 모두 `TypeError`다(scipy의 `bartlett levene fligner`는 `ValueError`) (사용자 결정 2026-11-20) |
| H31 Welch의 작은 표본 | Welch t에서 표본 하나의 크기가 2 미만이면 `df`도 NaN이다(scipy는 1) (사용자 결정 2026-11-20) |
| H32 H1의 속도 숙제 | 1스레드의 double-double 훑기(NEON), `levene`의 병렬 중앙값 선택, `fligner` 정렬의 병합을 H10에서 한다(§8) (사용자 결정 2026-11-20) |
| H33 Levene의 R 대조 | R 대조는 `car::leveneTest`로 한다(car 3.1.5, regression의 G7a가 설치했다; H15) (사용자 결정 2026-11-20) |

### 1.2 범위 (안)
목록은 scipy 1.18.1 `scipy.stats` 모듈 문서의 "Hypothesis Tests and related functions" 절과 공개 이름 전체(`dir(scipy.stats)`)를 대조해 만들었다. "출처"의 sm은 statsmodels 0.15.0이다. 마일스톤은 §8이다. "재표본(H9)"은 그 검정의 `method=`에 재표본 객체를 받는 것을 H9에서 더한다는 뜻이다.

**모수: 평균과 분산 (마일스톤 H1)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `ttest_1samp(a, popmean, /, *, ...)` | scipy | t 분포. `statistic pvalue df`, `conf_int()` |
| `ttest_rel(a, b, /, *, ...)` | scipy | 같다 |
| `ttest_ind(a, b, /, *, equal_var, trim, method, ...)` | scipy | Student·Welch, `trim`이면 Yuen. 재표본(H9) |
| `ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, /, *, ...)` | scipy | 같다(요약값에서; `df`와 `conf_int()`는 H23) |
| `f_oneway(*samples, equal_var, ...)` | scipy | F 분포, `equal_var=False`면 Welch 분산분석(H26). `statistic pvalue df_num df_denom`(H27) |
| `alexandergovern(*samples, ...)` | scipy | 정규화 변환의 합 → 카이제곱 |
| `bartlett(*samples, ...)` | scipy | 카이제곱 |
| `levene(*samples, center, proportiontocut, ...)` | scipy | F(중앙값·평균·절삭 평균 중심) |
| `fligner(*samples, center, proportiontocut, ...)` | scipy | 정규 점수의 카이제곱 |

**적합도 1: 분포를 정한 검정 (마일스톤 H2)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `chisquare(f_obs, f_exp=None, /, *, ddof, sum_check, ...)` | scipy | 카이제곱 |
| `power_divergence(f_obs, f_exp=None, /, *, ddof, lambda_, ...)` | scipy | Cressie–Read 족, 카이제곱 |
| `skewtest`, `kurtosistest`, `normaltest` | scipy | D'Agostino의 정규 변환, 정규·카이제곱(2) |
| `jarque_bera(x, /, *, ...)` | scipy | 카이제곱(2): p = e^(−JB/2) |
| `shapiro(x, /, *, ...)` | scipy | Royston(1995) AS R94의 W와 정규 근사 |
| `anderson(x, /, *, dist, method)` | scipy | 1.17부터의 새 꼴(`statistic pvalue`; 표의 보간 또는 몬테카를로). 옛 꼴(`critical_values`, `significance_level`, `fit_result`)은 두지 않는다 |
| `ks_1samp(x, cdf, /, *, alternative, method, ...)` | scipy | 정확(Durbin·Pomeranz, Smirnov), 근사, 점근(Kolmogorov). `statistic_location statistic_sign` |
| `kstest(rvs, cdf, /, *, ...)` | scipy | 한 표본·두 표본을 고르는 감싸개(§9 H2: 둘지) |
| `cramervonmises(rvs, cdf, /, *, ...)` | scipy | Csörgő–Faraway의 유한 표본 분포 |

**순위 검정 (마일스톤 H3)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `mannwhitneyu(x, y, /, *, use_continuity, alternative, method, ...)` | scipy | 정확(동점 없는 작은 표본), 정규 근사(동점 보정). 재표본(H9) |
| `ranksums(x, y, /, *, ...)` | scipy | 정규 근사(scipy는 legacy로 표시, §9 H3) |
| `wilcoxon(x, y=None, /, *, zero_method, correction, alternative, method, ...)` | scipy | 정확, 정규 근사(`zstatistic`). 재표본(H9) |
| `kruskal(*samples, ...)` | scipy | 카이제곱(동점 보정) |
| `friedmanchisquare(*samples, ...)` | scipy | 카이제곱 |
| `brunnermunzel(x, y, /, *, alternative, distribution, ...)` | scipy | t 또는 정규 |
| `ansari(x, y, /, *, alternative, method, ...)` | scipy | 정확(AS 93), 정규 근사. 재표본(H9) |
| `mood(x, y, /, *, alternative, ...)` | scipy | 정규 근사 |
| `median_test(*samples, ties, correction, lambda_, ...)` | scipy | 2 × k 분할표의 카이제곱(`median table` 필드) |
| `page_trend_test(data, /, *, ranked, predicted_ranks, method)` | scipy | 정확(재귀), 정규 근사(`method` 필드) |

**적합도 2: 표본끼리 (마일스톤 H4)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `ks_2samp(data1, data2, /, *, alternative, method, ...)` | scipy | 정확(격자 경로), 점근 |
| `cramervonmises_2samp(x, y, /, *, method, ...)` | scipy | 정확(Burr의 셈), 점근 |
| `anderson_ksamp(samples, /, *, variant, method)` | scipy | Scholz–Stephens 표의 보간. 재표본(H9). 1.17부터의 `variant` 꼴만(옛 `midrank`, `critical_values` 없음) |
| `epps_singleton_2samp(x, y, /, *, t, ...)` | scipy | 특성함수의 카이제곱(4) |

**상관 (마일스톤 H5)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `pearsonr(x, y, /, *, alternative, method, ...)` | scipy | 베타 분포(정확한 정규 이론), `conf_int()`(Fisher z). 재표본(H9) |
| `spearmanrho(x, y, /, *, alternative, method, ...)` | scipy | t 근사. 1.18의 새 꼴(브로드캐스트). 재표본(H9) |
| `spearmanr(a, b=None, /, *, ...)` | scipy | 열들의 상관 행렬 꼴(§9 H5: 둘지) |
| `pointbiserialr(x, y, /, *, ...)` | scipy | `pearsonr`과 같다 |
| `kendalltau(x, y, /, *, method, variant, alternative, ...)` | scipy | 정확(동점 없음: 역순 쌍의 분포), 정규 근사(동점 분산). τ-b, τ-c |
| `weightedtau(x, y, /, *, rank, weigher, additive, ...)` | scipy | 통계량만(scipy의 `pvalue`는 언제나 NaN; §9 H5) |
| `somersd(x, y=None, /, *, alternative)` | scipy | 점근 표준오차의 정규 근사(`table` 필드) |
| `chatterjeexi(x, y, /, *, y_continuous, method, ...)` | scipy | 정규 근사. 재표본(H9) |

**분할표 (마일스톤 H6)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `chi2_contingency(observed, /, *, correction, lambda_, method)` | scipy | 카이제곱(1자유도면 Yates), Cressie–Read. `expected_freq`. 재표본(H9) |
| `fisher_exact(table, /, *, alternative, method)` | scipy | 2 × 2는 초기하. r × c는 §3.7(scipy는 시드 없는 몬테카를로) |
| `barnard_exact(table, /, *, alternative, pooled, n)` | scipy | 장애 모수의 상한(§3.7) |
| `boschloo_exact(table, /, *, alternative, n)` | scipy | 같다 |
| `mcnemar(table, /, *, exact, correction)` | sm | 이항 정확 또는 카이제곱(연속성 보정) |
| `cochrans_q(x, /, *, ...)` | sm | 카이제곱(`df`) |
| `odds_ratio(table, /, *, kind)`, `relative_risk(...)` | scipy(`stats.contingency`) | 검정이 아니고 신뢰구간이 있는 추정(§9 H6: 넣을지) |

**비율과 이항 (마일스톤 H7)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `binomtest(k, n, /, *, p, alternative)` | scipy | 정확(양쪽은 1 + 10⁻⁷ 규칙), `conf_int(method=)`(exact, wilson, wilsoncc; scipy의 `proportion_ci`, H16) |
| `quantile_test(x, /, *, q, p, alternative)` | scipy | 이항, `conf_int()` |
| `poisson_means_test(k1, n1, k2, n2, /, *, diff, alternative)` | scipy | E-test(이중 합) |
| `proportions_ztest(count, nobs, /, *, value, alternative, prop_var)` | sm | 정규 |
| `proportions_chisquare(count, nobs, /, *, value)` | sm | 카이제곱(`table`, `expected_freq`) |
| `proportion_confint(count, nobs, /, *, ..., method, alternative)` | sm | normal, agresti_coull, beta(Clopper–Pearson), wilson, jeffreys, binom_test (방법 이름은 §9 H7) |
| `test_proportions_2indep(count1, nobs1, count2, nobs2, /, *, value, method, compare, alternative, correction)` | sm | 차(wald, agresti-caffo, score), 비(log, log-adjusted, score), 오즈비(logit, logit-adjusted, logit-smoothed, score) |
| `confint_proportions_2indep(count1, nobs1, count2, nobs2, /, *, method, compare, ..., correction)` | sm | 차(wald, agresti-caffo, newcomb, score), 비(log, log-adjusted, score), 오즈비(logit, logit-adjusted, score) |

**다중 비교와 보정 (마일스톤 H8)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `tukey_hsd(*samples, equal_var)` | scipy | 스튜던트화 범위 분포, `conf_int()` |
| `dunnett(*samples, control, alternative, rng)` | scipy | 다변량 t의 cdf(`sd.stats.MultivariateT`), `conf_int()` |
| `multipletests(pvals, /, *, alpha, method, max_iter, ...)` | sm | 열 방법(§3.9) |
| `false_discovery_control(ps, /, *, axis, method)` | scipy | bh, by |
| `combine_pvalues(pvalues, /, *, method, weights, ...)` | scipy | fisher, pearson, tippett, stouffer, mudholkar_george |

**재표본과 몬테카를로 (마일스톤 H9)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `permutation_test(data, statistic, /, *, permutation_type, vectorized, n_resamples, batch, alternative, axis, rng)` | scipy | 전부 나열(가능하면) 또는 무작위 순열, `null_distribution` |
| `monte_carlo_test(data, rvs, statistic, /, *, ...)` | scipy | 같다 |
| `bootstrap(data, statistic, /, *, n_resamples, batch, vectorized, paired, axis, alternative, method, rng)` | scipy | percentile, basic, BCa. `conf_int()`(H16), `bootstrap_distribution`, `standard_error` |
| `PermutationMethod`, `MonteCarloMethod`, `BootstrapMethod` | scipy | 앞 검정들의 `method=`에 넘기는 설정 객체(§9 H9) |
| `bws_test(x, y, /, *, alternative, method)` | scipy | Baumgartner–Weiss–Schindler, 순열 |
| `multiscale_graphcorr(x, y, /, *, ...)` | scipy | MGC, 순열(§9 H9: 넣을지) |
| `goodness_of_fit(dist, data, /, *, ...)` | scipy | 모수를 맞춘 몬테카를로 적합도(§9 H9: 분포 맞추기가 필요하다) |

**빼는 것**
- `sd.regression`에 이미 있다: `linregress`(G18), `theilslopes`, `siegelslopes`(G56). OLS 결과의 `jarque_bera()`·`breusch_pagan()` 진단도 regression에 있다. `htest.jarque_bera`와 같은 커널을 쓴다(값이 같아야 한다).
- 기술통계(sd.stats §4)와 검정이 아닌 것: `describe skew kurtosis moment zscore rankdata sem iqr mode entropy differential_entropy`, 평균의 변형·절삭 통계(`gmean`, `trim_mean`, `tvar` …), `kstat`, `lmoment`, `variation`, `quantile`, `expectile`, `estimated_cdf`, `binned_statistic*`, 원형 통계, 변환(`boxcox`, `yeojohnson`, `obrientransform`, `sigmaclip`, `trimboth`, `zmap` …), 거리(`wasserstein_distance`, `energy_distance`), `fit`, `ecdf`, `sobol_indices`, 그림 검정(`probplot`, `ppcc_*`), `bayes_mvs`, `mvsdist`, `scipy.stats.contingency`의 `association`·`crosstab`·`expected_freq`·`margins`, `tiecorrect`(검정 안의 보조), `scipy.stats.mstats`(마스크 배열 판).
- 생존 분석: `logrank`(scipy의 `CensoredData`가 없다).
- 검정력: scipy `power`(모의실험), statsmodels의 `power_proportions_2indep`, `samplesize_*`, `proportion_effectsize` — 나중 후보(H5).
- TOST 동등성 검정: statsmodels `tost_proportions_2indep`, `binom_tost`, `proportions_ztost` — 나중 후보(H5).
- statsmodels의 겹치는 것: `binom_test`(= `binomtest`), `score_test_proportions_2indep`(= `test_proportions_2indep(method='score')`), `fdrcorrection_twostage`(= `multipletests`의 fdr_tsbh·fdr_tsbky), `fdrcorrection`(§3.9의 안: 두지 않는다; §9 H8). 나중 후보: `proportions_chisquare_allpairs`, `proportions_chisquare_pairscontrol`, `multinomial_proportions_confint`, `multipletests`의 `fdr_gbs`(statsmodels가 "다른 꾸러미와 대조하지 않았다"고 적은 방법).
- 일반 z 검정(statsmodels `ztest`)은 scipy에 없고 H5의 두 묶음도 아니므로 넣지 않는다.

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다. 별칭이 없다(scipy 결과의 `correlation`, MGC의 `stat` 같은 같은 값의 둘째 이름도 두지 않는다). float64 실수로 계산하고(bool·int64는 바꿔서) complex는 `TypeError`다.
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 무작위가 드는 함수는 같은 `rng` 상태에서 스레드 수와 상관없이 같은 비트다.
- **모든 호출은 정해진 시간 안에 끝난다**(stats.md §1.3). 정확한 분포의 동적 계획, 무한 급수, 장애 모수의 최대화, 근 찾기에는 근거 있는 상한을 둔다. 상한에 닿으면 조용히 틀린 값이 아니라 NaN이다(원소별 값의 규칙, API.md §2.2). 정확한 방법을 사용자가 고른 경우(`method='exact'`)에 작업량이 상한을 넘으면 `ValueError`로 알린다(H13).
- 경고를 내지 않는다. scipy가 경고(`ConstantInputWarning`, `SmallSampleWarning`, 근사가 부정확하다는 경고)를 내는 곳은 NaN이나 문서다.
- 참값은 Arb(python-flint)와 정확한 유리수(Python의 int·`fractions`)만 쓴다. mpmath는 쓰지 않는다.
- 영역 간 일관성이 먼저다: 자유도 인자 이름(`chisquare`의 `ddof`는 뜻이 달라 그대로, 결과의 자유도는 `df`; H10), `nan_policy`, `axis`, 메시지의 꼴(`ttest_ind(): …`), `rng`, 결과의 읽기 전용 규칙은 기존 영역을 따른다.
- 반올림이 아닌 동작(기본값, `'auto'`가 고르는 방법, 양쪽 p값의 정의, 연속성 보정, 동점 보정)은 scipy·statsmodels와 같다(API.md §4의 규칙 2). 다르게 정한 곳은 `[scipy와 다름]`으로 이유와 함께 적는다.


### 1.4 다른 영역과의 관계: sd.regression
- **경계**: 표본·표를 받는 검정은 htest, 적합한 모형을 받는 검정과 진단은 regression이다. regression에는 이미 `jarque_bera()`, `breusch_pagan()`, `durbin_watson`이 있고, regression의 G7a·G7b(변수 선택, `anova` 표와 F·LR·Wald의 모형 비교, VIF·DFFITS·DFBETAS·White·Goldfeld–Quandt·RESET; docs/design/regression.md §3.8)도 regression의 것이다. 모형을 받는 검정은 설계 행렬이 필요해 htest에 두지 않는다. 잔차의 다른 정규성 검정(Shapiro–Wilk, Anderson–Darling, D'Agostino)은 잔차를 htest에 넘긴다. htest의 분산 동질성 검정(Bartlett, Levene, Fligner)은 집단을 받는 다른 검정이라 겹치지 않는다.
- **같은 값의 커널은 htest에 있다**(H17), **같은 검정은 같은 형이다**(H18). 표는 docs/api/htest.md §13이다.
  - Jarque–Bera: 커널(평균을 뺀 2·3·4차 합, double-double; 지금 `csrc/regression/linear.cpp`의 `resid_stats`의 그 부분)은 H2에서 htest로 옮기고 regression의 `r.jarque_bera()`가 부른다. p값은 두 쪽 모두 `ChiSquared(df=2).ccdf(jb)`다: 닫힌 꼴 e^(−JB/2)는 그와 1 ulp 다를 수 있다(jb = 1.3에서 0.5220457767610162와 0.522045776761016; regression 설계자가 찾았다). 그래서 `htest.jarque_bera(r.resid)`와 OLS의 `r.jarque_bera()`는 같은 비트이고 같은 형(`JarqueBeraResult`)이다. regression의 골든은 H20이 바꾸는 퇴화한 경우 말고 그대로여야 한다(옮긴 뒤 `tools/golden.py check regression`).
  - 예외 하나: 완전 적합(G95, G117)에서 regression은 잔차를 정확히 0으로 보아 NaN을 내고, `htest.jarque_bera(r.resid)`는 반올림 크기의 잔차로 값을 낸다.
  - WLS·GLS: `r.jarque_bera()`는 백색화한 잔차(쓰인 행의 √w·`resid`, GLS는 L⁻¹`resid`; 가중치 0인 행은 빠진다)의 값이다. `resid`는 원래 척도라 지금은 htest에 그대로 넘길 수 없다. regression이 `wresid`(statsmodels의 이름)를 두면 그 쓰인 행을 넘겨 같은 비트가 된다(regression §8의 질문).
  - 일원 분산분석: 집단 더미의 OLS로 만든 `anova` 표(항 하나)와 `htest.f_oneway`는 같은 F·p이지만 비트까지 같지는 않다: `anova`는 설계 행렬만 받아 집단 레이블이 없으므로 적합의 QR에서 계산하고, `f_oneway`는 집단 제곱합에서 계산한다. 통계량은 두 쪽의 정확도 계약 안에서 같고 p값은 같은 `F(df_num, df_denom).ccdf` 호출이다. 주인은 htest(`f_oneway`)다. 두 집단의 `ttest_ind`과 더미 계수의 t도 계약 안에서 같다.
  - Pearson: `pearsonr(x, y)`의 r·p와 `linregress(x, y)`의 `rvalue`·`pvalue`는 같은 검정이다. 1 − r²의 커널은 H17에 따라 H5에서 htest에 두고 `linregress`가 부른다(그때 regression의 골든이 바뀌는 경우가 있으면 따로 묻는다).
  - p값의 꼬리: 두 영역 모두 sd.stats 분포의 `ccdf`·`logccdf`를 부른다(같은 인자면 같은 비트).
  - 신뢰구간: 두 영역 모두 `conf_int(*, alpha=0.05)`이고 같은 검사와 메시지(`'alpha' must be between 0 and 1 (exclusive)`)다(H16, G99).
  - 다중 검정 보정: `multipletests`(H8 마일스톤)는 계수 여럿의 `r.pvalues`와 단계적 변수 선택의 p값을 보정하는 짝이다(`multipletests(r.pvalues, axis=-1)`, 배치 (…, k)). 다만 선택 뒤의 p값은 같은 자료로 고르고 검정했기 때문에 명목 수준을 지키지 않고, 보정도 선택의 편향을 고치지 못한다(regression 설계자의 주의; 두 영역의 문서에 함께 적는다). 분산분석 뒤의 쌍 비교는 `tukey_hsd`·`dunnett`(H8)다.
- **결과 바탕 클래스의 안정성**: regression G7a의 검정 결과 클래스가 `_TestResult`를 상속하므로 그 API(`_FIELDS`, `_UNPACK`, `_CI`, `_make`, 생성자)를 안정되게 유지한다(docs/design/regression.md G163).
- **regression의 지금 결과**: `JarqueBeraResult`·`BreuschPaganResult`는 `stardust._results`의 namedtuple이고 `sd.regression`이 내보낸다(D-25). Jarque–Bera는 H2에서 htest의 결과 클래스가 되고 `sd.regression.JarqueBeraResult`가 그 클래스를 가리킨다(H18). 그 클래스는 지금처럼 둘로 풀리고 인덱스·`len`도 같다. namedtuple에만 있는 것(`==`의 튜플 비교, `_asdict`, `_replace`, `_fields`)과 골든이 결과를 적는 꼴이 바뀌는지는 H2에서 확인하고, 바뀌면 그 차이를 묻는다. `BreuschPaganResult`는 htest에 짝이 없어 regression이 정한다(자유도 필드를 둔다면 H19의 `df`). 퇴화한 입력의 경고 없는 NaN(완전 적합의 0/0, BP 자유도 ≤ 0)은 H11과 같고, 분산 0의 판정은 H20으로 sd.stats `skew`의 규칙 하나다(regression의 `jarque_bera()`는 그런 퇴화한 경우에만 값이 바뀐다). regression의 코드는 H0에서 바꾸지 않았다.

---

## 2. 입력과 결과

### 2.1 입력
- **데이터**(표본, 표, 개수): API.md §5의 데이터 입력이다. `sd.ndarray`, 실수의 시퀀스, numpy 배열, frame의 `Series`(`DataFrame`은 여러 열을 받는 곳만)를 받고 float64 배열로 바꾼다. frame의 결측은 NaN이 되어 `nan_policy`를 따른다. 실수 스칼라는 길이 1의 표본이 아니라 스칼라 인자(`popmean`, `k`, `n`)에만 받는다.
- **개수**(`binomtest`의 k·n, 분할표, 비율의 count·nobs): 음이 아닌 정수 값이어야 한다. scipy처럼 정수가 아닌 값은 `ValueError`, 정수 값의 float는 받는다(§9 H6: 분할표의 소수 기댓값).
- **`axis`와 배치**(H3, sd.stats 기술통계의 규칙 T28·T37): `axis`는 None(전체를 하나의 표본으로), int, int의 tuple·리스트다. 축을 따라 표본을 읽고 나머지 축은 배치다. 두 표본 이상을 받는 검정은 배치 축끼리 브로드캐스트하고 표본 축의 길이는 달라도 된다(scipy의 `_axis_nan_policy` 규칙과 같다). `keepdims`를 둔다. 기본값은 None이다(H9).
  - 배치는 줄(lane)마다 독립인 검정이다. 줄 하나의 계산은 어디서 돌든 같으므로 배치는 줄을 하나씩 부른 것과 비트까지 같다.
  - 축이 정해진 입력(분할표 (r, c), `page_trend_test`의 (m, n), `cochrans_q`의 (N, k), `chi2_contingency`)은 마지막 축들이 표이고 앞 축이 배치다(H9).
- **`nan_policy`**: `'propagate'`(NaN이 있는 줄은 결과가 NaN), `'omit'`(줄마다 NaN을 빼고 남은 크기로; 짝지은 검정은 쌍을 뺀다), `'raise'`(`ValueError`). 메시지는 T41의 꼴이다. 줄마다 남은 크기가 달라도 배치로 계산한다(scipy는 마스크 배열로 느리다).
- **무한대**: 순위 검정은 ±inf를 순서대로 다룬다(순위가 정해진다). 적률을 쓰는 검정은 sd.stats의 규칙(합이 유한하지 않으면 보정 없이; inf − inf는 NaN)이다.
- **`alternative`**: `'two-sided'`, `'less'`, `'greater'`만이다. statsmodels의 `'larger'`, `'smaller'`는 받지 않는다(H5: scipy에 있는 개념은 scipy의 이름; §9 H7). 메시지는 T41의 꼴이다.
- **무작위**(H3): 무작위가 드는 계산(재표본, 몬테카를로 p값, `dunnett`의 다변량 t를 scipy처럼 QMC로 할 때, r × c Fisher의 몬테카를로)은 `rng=`(sd.random의 `Generator`)를 꼭 받는다. 없으면 O31의 꼴인 `TypeError`(`'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType`)다. scipy처럼 시드 없이 무작위로 도는 기본값은 없다(sd.random의 R3: 전역 상태가 없다).
- **사용자 함수**(`ks_1samp`·`cramervonmises`의 `cdf`, 재표본의 `statistic`, `monte_carlo_test`의 `rvs`): Python 호출이다. 콜백 동안 GIL을 쥐고(optimize §2.1), 그 사이의 계산은 C++이다. 분포 인자는 sd.stats의 분포 객체(`sd.stats.Normal(mu=0, sigma=1)`)나 `cdf(x)`를 돌려주는 호출 가능한 객체를 받는다. scipy의 분포 이름 문자열('norm')과 `args=()`는 받지 않는다(O12와 같다; §9 H2).

### 2.2 결과 객체 (H6–H8, H10, H12)
- 검정마다 결과 클래스가 있고 공통 바탕 `_TestResult`를 쓴다(H6). 정의는 `src/stardust/htest/_result.py`다: 클래스를 내보내지 않으므로(H6) 모든 영역의 공개 결과 형이 있는 `stardust._results`(D-25)가 아니라 optimize의 `_result.py`처럼 영역 안에 둔다. 하위 클래스는 필드 목록(`_FIELDS`), 풀기 순서(`_UNPACK`), 신뢰구간 유무(`_CI`)만 적고, 속성·풀기·`conf_int`은 바탕이 만든다. 예외: Jarque–Bera의 클래스는 `sd.regression.JarqueBeraResult`로 공개된다(H18).
- 필드: `statistic`, `pvalue`와 검정마다 scipy·statsmodels의 필드(`df`, `zstatistic`, `statistic_location`, `statistic_sign`, `expected_freq`, `table`, `null_distribution` 등). 별칭은 없다(§1.3). 자유도는 카이제곱·t가 `df`, F 꼴이 `df_num`·`df_denom`이다(H10, H19). `method='auto'`가 있는 검정은 쓴 방법을 `method`에 적는다(H12).
- 필드는 읽기 전용 속성이다(G98; 대입하면 `AttributeError`). 배열 필드는 쓰기 금지 배열이고 읽을 때 복사하지 않는다(H21; G120, O103, API.md §2.2). 리스트인 설정 필드는 읽을 때마다 새 리스트다(G113). 검정이 쓰는 생성자 `_make`는 새 배열을 그대로 받고, 공개 꼴의 생성자는 받은 배열을 한 번 복사한다(O103). 그 호출에서 주지 않는 필드는 `AttributeError`다(`hasattr()`로 안다).
- 스칼라 검정의 필드는 Python `float`(개수는 `int`), 배치는 배치 모양의 sd 배열이다(기술통계와 같다). `df`(와 `df_num`·`df_denom`)는 Welch처럼 정수가 아닐 수 있어 언제나 float다(H22).
- **신뢰구간**(H16): regression의 꼴인 메서드 `conf_int(*, alpha=0.05)`이고(G99) 아래·위 끝의 (…, 2) float64 배열을 돌려준다(검정 하나면 (2,); 부를 때마다 새 배열이라 쓸 수 있다, G120). `binomtest`는 scipy `proportion_ci`의 `method`를 받아 `conf_int(*, alpha=0.05, method='exact')`, `odds_ratio`는 `alternative`도 받는다. 신뢰수준을 호출에서 받는 함수(`proportion_confint`, `confint_proportions_2indep`)의 인자도 `alpha=0.05`이고(statsmodels와 같은 이름), `bootstrap`은 호출의 `confidence_level` 대신 결과의 `conf_int(alpha=)`다. 검사와 메시지는 regression의 것과 같다(`conf_int(): 'alpha' must be between 0 and 1 (exclusive)`).
- **튜플 풀기**(H7): scipy의 결과가 풀리는 곳(`stat, p = ttest_ind(...)`; `chi2_contingency`·`median_test`는 넷)은 같은 개수와 순서로 풀리고 `r[0]`, `len(r)`도 그 튜플의 것이다. scipy가 풀지 않는 결과(`binomtest`, `cramervonmises`, `somersd`, 재표본 결과 등)는 풀리지 않는다. statsmodels가 튜플을 돌려주는 함수(`proportions_ztest`, `proportion_confint`, `multipletests` 등)는 결과 객체로 바꾸되 같은 순서로 풀리게 하는 안이다(§9 H7·H8).
- `repr`는 한 줄 요약이다(`TtestIndResult(statistic=2.13, pvalue=0.0412, df=27)`, 배치는 `TtestIndResult(batch shape (10,))`). `summary()`는 두지 않는다.
- 공통 입력 층은 `src/stardust/htest/_common.py`다(H0): 데이터는 sd.stats 기술통계의 변환 함수를 그대로 쓰고(같은 메시지), `Samples`가 `axis`·배치·`keepdims`·`nan_policy`(짝지은 표본은 쌍을 뺀다)를, `Tables`가 표의 배치와 개수 검사를 맡는다. 옵션 검사(`alternative`, `method`와 `resolved`, `check_exact_work`, `rng`, `alpha`, int·실수 인자)는 다른 영역과 같은 메시지 꼴이다.

---

## 3. 수치 방법 (안)
scipy가 정확하지 않거나 느린 곳과 그것을 넘는 방법을 묶음마다 적는다. 잰 값은 이 초안을 쓰며 scipy 1.18.1에서 본 것이고(Apple M4 Pro, 1스레드), H0의 기준선에서 다시 잰다.

### 3.1 공통
- **p값의 꼬리**: p값은 언제나 작은 쪽 꼬리를 직접 계산한다(`ccdf`는 1 − cdf가 아니다; sd.stats의 `ccdf`·`logccdf`). 양쪽 p값은 연속 분포에서 2·min(cdf, ccdf)를 1로 자른 값이다(scipy와 같다). 대칭 분포(t, 정규)는 2·ccdf(|t|)다. 2⁻¹⁰²² 아래로 내려가는 p값도 서브노멀까지 값을 낸다(0이 아니다).
- **통계량의 조건수**: p값의 오차는 통계량의 오차가 κ_p = |s·∂p/∂s / p|배로 커진 것과 분포 함수의 오차의 합이다. 깊은 꼬리에서 κ_p는 크다(정규에서 z²쯤). 그래서 계약은 special의 역함수처럼 κ_p를 넣은 꼴이다(§4).
- **통계량을 정확하게**: 평균·분산·제곱합은 sd.stats 기술통계의 커널(평균을 hi + lo로, TwoSum 보상 합; stats.md §5)을 쓴다. 상쇄가 본질인 통계량은 식을 바꿔 쓴다(묶음마다 아래). 순위는 `sd.stats.rankdata`의 커널(정확한 중간 순위: 반정수라 double로 정확하다)이다.
- **정확한 검정의 "적어도 그만큼 극단"**: 비교는 통계량의 정수 꼴로 한다(U, T⁺, 역순 쌍의 수, KS의 격자 간격 k/lcm(n₁, n₂), CvM의 정수 합). 수학적으로 같은 두 값이 반올림으로 갈려 한 칸을 빼거나 넣는 일이 없다. 이산 분포의 양쪽 p값은 scipy의 규칙 그대로다. `binomtest`는 반대편에서 확률이 관측의 확률 × (1 + 10⁻⁷) 이하인 결과를 더한다(R `binom.test`와 같다). 2 × 2 `fisher_exact`는 허용오차가 10⁻¹⁴다: 관측의 확률이 최빈값의 확률과 상대 10⁻¹⁴ 안이면 p = 1이고, 아니면 반대편에서 관측의 확률 × (1 + 10⁻¹⁴)보다 작은(최빈값 아래쪽은 이하인) 결과를 더한다. R의 `fisher.test`(r × c도)는 1 + 10⁻⁷이다(H0에서 scipy 1.18.1 소스로 확인했다).
- **정확한 분포는 확률로 쌓는다**: 개수(조합의 수)는 n이 조금만 커도 double을 넘는다(C(2000, 1000) ≈ 10⁶⁰⁰). 그래서 점화식을 양수의 볼록 결합으로 고쳐 확률을 직접 쌓는다(예: Mann–Whitney f(m, n, u) = m/(m+n)·f(m−1, n, u−n) + n/(m+n)·f(m, n−1, u)). 항이 모두 양수이고 뺄셈이 없어 상대오차가 점화의 깊이 × ε로 묶이고, 꼬리의 합도 양수의 합이다. 서브노멀 아래의 꼬리는 log 쪽(log-sum-exp)으로 쌓는다.
- **연속성 보정과 동점 보정**: scipy·statsmodels의 정의 그대로다(Mann–Whitney의 0.5, `wilcoxon(correction=True)`의 0.5, Yates의 min(0.5, |O − E|), McNemar의 (|b − c| − 1)², 순위 검정의 동점 분산 보정).
- **`'auto'`의 선택은 scipy 그대로다**(API.md §4 규칙 2). scipy 1.18.1 소스에서 읽은 문턱:

| 함수 | `'auto'`가 정확을 고르는 조건 (scipy 1.18.1) |
|---|---|
| `mannwhitneyu` | 한 표본의 크기가 8 이하이고 동점이 없다 |
| `wilcoxon` | 동점·0이 없으면 n ≤ 50. 있으면 n ≤ 13에서 모든 부호의 나열(순열), 그 위는 점근 |
| `ansari` | m < 55, n < 55이고 동점이 없다 |
| `kendalltau` | 동점이 없고 n ≤ 33이거나 min(역순 쌍, 순 쌍) ≤ 1 |
| `ks_1samp` | 언제나 정확 쪽(양쪽은 정확, 한쪽은 Smirnov) |
| `ks_2samp` | max(n₁, n₂) ≤ 10000(`MAX_AUTO_N`); 정확이 실패하면 점근 |
| `cramervonmises_2samp` | max(n₁, n₂) ≤ 20 |
| `page_trend_test` | scipy의 `_choose_method`(H3에서 옮긴다) |
| `fisher_exact` r × c | 언제나 시드 없는 몬테카를로(`MonteCarloMethod()`) — §3.7 |

  - 정확한 계산이 싼 곳(동점이 있는 작은 표본, 문턱 바로 위)에서 `'auto'`를 넓힐지는 묻는다(§9 H3). 추천은 scipy 그대로이고 `'exact'`를 고르면 더 넓게 쓸 수 있게 한다.
- **동점이 있을 때의 `'exact'`**: scipy의 `mannwhitneyu`·`ansari`의 `'exact'`는 "동점을 보정하지 않는다"(동점이 없는 분포를 쓴다: p값이 그 데이터의 귀무분포가 아니다), `kendalltau`의 `'exact'`는 동점이면 `ValueError`다. 여기서는 동점이 있으면 `'exact'`가 조건부 순열 분포(관측한 중간 순위의 정확한 분포)를 쓰는 안이다(`[scipy와 다름]`; §9 H3).
- **퇴화한 입력**: 상수 표본, 0인 분산, 너무 작은 표본(scipy가 NaN과 경고를 내는 곳: `skewtest` n < 8, `kurtosistest` n < 5, `shapiro` n < 3)은 NaN이다(경고 없음). scipy의 `ttest_1samp([1, 1, 1], 0)`은 statistic inf, pvalue 0이고 여기서는 NaN이다(H24, API.md §2.2의 "정할 수 없는 값은 NaN"; `[scipy와 다름]`). 분산 0의 판정은 sd.stats `skew`의 규칙(H20)이다.

### 3.2 모수: 평균과 분산 (H1, 구현)
코드는 `src/stardust/htest/_parametric.py`(인자 검사와 표본의 배치)와 `csrc/htest/`(`moments.hpp`의 줄 적률, `parametric.cpp`의 검정)다. 검정 하나는 C++ 호출 하나다(`stardust._core.htest`의 `ttest`, `ttest_stats`, `kgroups`).
- **줄의 적률**(`moments.hpp`): 줄(배치의 검정 하나의 표본)마다 개수, 평균, 편차 제곱합을 double-double로 낸다. 첫 번째 훑기는 합(Neumaier의 보상 합에 낮은 부분을 더한 것), 두 번째는 double-double 평균에서의 (x − m)²이다. 편차는 x − m.hi를 TwoSum으로 정확히 빼고 m.lo를 더한 double-double이고 그 제곱도 double-double이다. 합은 원소 j를 j mod 4번째 합에 넣는 네 개의 합을 순서대로 합친다(지연이 한 사슬에 묶이지 않게). 일은 줄마다 8192원소의 고정된 덩어리이고 스레드풀에서 돌며 덩어리를 순서대로 합친다: 비트는 스레드 수와 상관없고, 긴 줄 하나(`axis=None`)도 병렬이다. 줄 하나는 배치 안에서나 혼자서나 같은 비트다.
  - 평균과 제곱합이 double-double이라 상쇄가 본질인 통계량이 자릿수를 지킨다: 오프셋 10⁸–10⁹의 평균의 차(t), 집단 평균과 전체 평균의 차(F, Levene), 거의 같은 분산의 비(Bartlett).
  - **단위**: H1의 통계량은 모든 표본을 같은 수로 곱해도 그대로다. 가장 큰 크기가 2⁵⁰⁰ 이상(편차의 제곱이 넘친다)이거나 2⁻⁴⁰⁰ 아래(제곱이 정규 수 밖으로 나간다)인 줄은 그 크기의 지수 e로 2^e 단위에서 계산하고(값 × 2⁻ᵉ, 정확), 추정값과 표준오차는 끝에 원래 단위로 돌린다. 1e±300으로 곱한 데이터의 통계량은 1e-14, p값은 1e-13 안에서 같다.
  - **분산 0**(H20): 표본의 m2 = 제곱합/n이 (ε·평균)² 이하이면 상수다(ε = 2⁻⁵²). 원소 하나의 표본도 상수다.
- **t 검정**: `ttest_1samp`은 (m − μ₀)/√(s²/n), `ttest_rel`은 차 a − b를 double-double로 든 한 표본 검정(두 큰 값의 차가 double로 반올림되지 않는다; scipy는 a − b를 double로 만든다), `ttest_ind`은 scipy의 `_equal_var_ttest_denom`(원소 하나의 표본은 분산 0)과 `_unequal_var_ttest_denom`(Welch 자유도)이다. 평균의 차는 double-double끼리 뺀다. Student의 합동 분산은 (SS₁ + SS₂)/(n₁ + n₂ − 2)다. p값은 `StudentT(df)`의 `ccdf`·`cdf`를 C++에서 그 분포의 메서드로 부른다(`stats::method_of`: Python의 `StudentT(df=df).ccdf(t)`와 같은 비트). 양쪽은 2·ccdf(|t|)다.
  - **Yuen의 절삭**(`trim`): scipy처럼 g = int(n·trim)(double 곱), 가운데 h = n − 2g개의 순서통계량의 평균과, 양 끝 g개를 x₍g+1₎·x₍n−g₎로 바꾼 표본의 제곱합(윈저화)을 h − 1로 나눈 분산이다. 순서통계량은 선택 두 번(`nth_element`)으로 찾는다(O(n); scipy는 정렬한다).
  - **신뢰구간**(`conf_int`): scipy의 정의(추정값 ∓ q·se, q는 `StudentT.iccdf`; 한쪽은 한 끝이 ±inf)다. 추정값은 double-double이다. 표준오차가 0이면(H24) 폭 0의 구간 (추정값, 추정값)이다(scipy와 regression의 `conf_int`와 같다).
  - **퇴화**: 표본이 비거나(모든 필드 NaN) 한 표본 검정의 n < 2(t·p NaN, `df` = n − 1), Student의 n₁ + n₂ − 2 ≤ 0, Welch의 nᵢ < 2(`df`도 NaN, H31; scipy는 `df` = 1), 표준오차 0(관련된 표본이 모두 상수: t·p NaN, H24), propagate의 NaN(모든 필드 NaN, scipy와 같다), 무한대(평균이나 제곱합이 유한하지 않으면 NaN).
- **`ttest_ind_from_stats`**: 같은 식을 요약값에서 계산하고 `df`와 `conf_int()`를 낸다(H23). 평균의 차는 double-double이다. 두 표준편차가 0이면 NaN(H24), nᵢ ≤ 0이면 NaN, Welch는 nᵢ ≤ 1이면 NaN이다.
- **분산분석**(`f_oneway`): F = (SSB/(k − 1))/(SSW/(N − k)), SSB = Σnᵢ(mᵢ − m)²의 차는 double-double 평균끼리, SSW = ΣSSᵢ다. 집단 하나가 비거나 N ≤ k면 NaN이다(`df_num`·`df_denom`은 k − 1, N − k로 둔다). 모든 집단이 상수이면(SSW = 0) NaN이다(H28): scipy는 집단 평균이 다르면 inf와 p = 0, 같으면 NaN을 낸다. 표준오차 0의 t가 NaN인 것(H24)과 같은 규칙이다(두 집단의 F는 t²다). p값은 `F(df_num, df_denom).ccdf`(regression의 `f_pvalue`와 같은 호출)다.
- **Welch 분산분석**(`equal_var=False`, H26): scipy와 R의 식, wᵢ = nᵢ/sᵢ², 가중 평균 m_w = Σwᵢmᵢ/Σwᵢ(분자와 분모 모두 double-double: 반올림한 W로 나누면 m_w가 W의 반올림 × 평균의 크기만큼 움직인다), F = [Σwᵢ(mᵢ − m_w)²/(k − 1)]/[1 + 2(k − 2)T/(k² − 1)], `df_denom` = (k² − 1)/(3T), T = Σ(1 − wᵢ/W)²/(nᵢ − 1). 1 − wᵢ/W는 다른 가중치의 합 / W로 계산한다(한 가중치가 지배하면 1 − wᵢ/W가 상쇄된다). nᵢ < 2나 상수 집단은 NaN이다.
- **`alexandergovern`**: scipy의 식이다. 가중 평균은 Welch처럼 Σ(mᵢ/seᵢ²)/Σ(1/seᵢ²)를 double-double로 한다: scipy의 정규화한 가중치 Σwᵢmᵢ는 가중치의 반올림이 평균의 크기만큼 커져 오프셋 10⁸에서 통계량이 2.7e-8 틀린다. Hill의 변환은 scipy와 같은 순서이고 log(1 + t²/v)는 log1p다. nᵢ < 2(scipy는 `ValueError`)나 상수 집단(scipy와 같은 판정이 아니라 H20)은 NaN이다.
- **`bartlett`**: T = [(N − k)log s_p² − Σ(nᵢ − 1)log sᵢ²]/C에서 dᵢ = sᵢ²/s_p² − 1이라 하면 Σ(nᵢ − 1)dᵢ = 0이므로 분자는 Σ(nᵢ − 1)(dᵢ − log1p(dᵢ))다. 항이 모두 음이 아니어서 상쇄가 없다(설계 초안의 log1p((s_p² − sᵢ²)/sᵢ²)는 부호가 섞인 항의 합이라 이 꼴로 바꿨다). sᵢ², s_p², dᵢ는 double-double, dᵢ − log1p(dᵢ)는 sd.special의 double-double `log1pmx`다. 거의 같은 분산(차 10⁻⁹)의 T = 1.594e-15가 상대 0으로 맞는다(scipy 53배, R 음수). 상수 집단(scipy: T = inf, p = 0)이나 nᵢ < 2는 NaN이다(H11, H28). 결과의 `df`는 k − 1이다(H29).
- **`levene`**: 중심은 double-double이다: 중앙값(가운데 값, 짝수 n은 두 가운데 값의 정확한 반), 평균(double-double 평균), 절삭 평균(scipy `trim_mean`의 g = int(p·n)과 가운데 평균). z = |x − c|도 double-double이고 그 일원 분산분석이다(위의 `anova`). 중앙값은 선택(`nth_element`)이다. 집단 하나가 비면 NaN, z가 모든 집단에서 상수면 NaN이다. 결과에 `df_num`(k − 1)·`df_denom`(N − k)이 있다(H29).
- **`fligner`**: aᵢⱼ = |x − cᵢ|(double-double, levene과 같은 중심)를 모든 집단에 걸쳐 정렬하고(값의 정확한 같음으로 동점; 큰 줄은 고정된 조각을 스레드풀에서 정렬하고 짝으로 병합), 중간 순위 r의 점수 Φ⁻¹(1/2 + r/(2(N + 1)))를 그 인자를 만들지 않고 계산한다: r ≤ (N + 1)/2면 q = r/(2(N + 1)) ≤ 1/4에서 AS241의 가운데 꼴 q·R(q), 그 위는 −ndtri((N + 1 − r)/(2(N + 1)))다. 각 인자는 정확한 정수의 몫 하나만 반올림한다(scipy는 r/(2(N + 1)) + 1/2을 double로 만든다). 통계량은 Σnᵢ(Āᵢ − ā)²/V², 평균과 V²는 double-double이다. `df`는 k − 1이다(H29). 중앙값이 double-double이라 짝수 n의 두 가운데 값의 편차가 정확히 같은 동점으로 남는다: scipy와 R은 중앙값을 double로 반올림해 그 동점이 갈리고 순위가 1/2 움직인다(n = 500–1500의 집단에서 p가 6.4e-9–2.9e-7 틀린다).
- **p값의 꼬리와 regression**: 모든 p값은 sd.stats 분포의 메서드다(`StudentT`, `F`, `ChiSquared`; §3.1). `f_oneway`의 F와 집단 더미 OLS의 `fvalue`, 두 집단 `ttest_ind`과 더미 계수의 t는 계약 안에서 같다(tests/htest/test_parametric.py가 1e-13로 확인한다; 비트는 아니다, §1.4).

### 3.3 적합도 1 (H2)
- **카이제곱과 Cressie–Read**: Pearson의 Σ(O − E)²/E는 항이 양수라 좋다. 로그 우도비 G = 2ΣO log(O/E)는 O ≈ E이면 항이 서로 상쇄한다(ΣO = ΣE라 일차 항이 지워진다). scipy는 그대로 더해 O − E = 10⁻³, E = 10⁶에서 G가 상대 6e-9 틀렸다(이 초안에서 Arb와 잼: 9.99999996e-12, 참값 1.000000002e-11). 항마다 O·log(O/E) − (O − E) = E·[(1 + δ)log1p(δ) − δ](δ = (O − E)/E, 언제나 ≥ 0)로 바꾸고 δ가 작으면 급수로 계산한다. 일반 λ의 2/(λ(λ + 1))ΣO[(O/E)^λ − 1]도 같은 방법이다(expm1(λ·log1p(δ))에서 일차 항을 뺀 꼴). λ = 0, −1(Neyman, 수정 로그 우도비)은 극한식이다. `sum_check`는 scipy의 상대 허용오차 그대로다.
- **D'Agostino의 셋**: `skewtest`·`kurtosistest`의 변환은 세제곱근과 로그에 음수 밑이 나올 수 있다(scipy는 부호를 따로 다룬다). 같은 식을 cbrt로, 꼬리 p값은 정규의 `ccdf`로 계산한다. `normaltest`의 K² = Z₁² + Z₂²는 카이제곱(2)이라 p = e^(−K²/2)이다. `jarque_bera`의 p값은 `ChiSquared(df=2).ccdf`다(H17: regression과 같은 호출이라 같은 비트; 닫힌 꼴 e^(−JB/2)는 1 ulp 다를 수 있다. scipy는 `chi2.sf`).
- **`shapiro`**: Royston의 AS R94(scipy와 R의 swilk)를 옮긴다. 계수 a_i는 기대 순서통계량의 다항 근사이므로 W는 "정확한 Shapiro–Wilk의 W"가 아니라 이 알고리즘이 정의한 값이다. 계약은 그 정의에 대한 정확도다(§4). n이 크면 W가 1에 가까워 log(1 − W)의 1 − W가 자릿수를 잃는다: 1 − W = [SST − (Σa_i x_(i))²]/SST를 double-double로 직접 계산한다. scipy와 R은 W를 double로 만든 뒤 1 − W를 한다(이 초안에서 n = 50에서 scipy와 R의 W가 상대 7e-16, p가 1.8e-14 다름).
- **`anderson`**: A² = −n − Σ(2i − 1)/n·[log F(x_(i)) + log(1 − F(x_(n+1−i)))]의 두 로그를 분포의 `logcdf`·`logccdf`로 계산한다(꼬리의 표본에서 log(1 − F)를 1 − F로 만들지 않는다). 모수 추정: 정규·지수는 닫힌 꼴, 로지스틱·Gumbel·Weibull은 scipy처럼 최우추정(`sd.optimize.root_scalar`로 점수 방정식). p값의 `'interpolate'`는 scipy의 표(Stephens)와 보간식 그대로다(계약은 그 보간에 대한 정확도). 분포 족을 어떻게 받을지는 §9 H2.
- **`ks_1samp`**: 양쪽 정확 분포는 Simard–L'Ecuyer(2011)의 선택(작은 n은 Durbin 행렬(Marsaglia–Tsang–Wang)이나 Pomeranz, 큰 n은 Pelz–Good 점근과 Kolmogorov 분포)이다. scipy의 `kstwo`도 같은 선택이다. Durbin 행렬은 원소가 유리수이고 d가 double(정확한 유리수)이라 작은 n의 참값은 정확한 유리수로 만든다(§7). 행렬의 거듭제곱은 지수를 따로 들어(넘침 방지) double-double로 곱하는 안이다. 한쪽은 Birnbaum–Tingey의 유한 합(양수 항, log 쪽에서 합)과 큰 n의 점근이다. 정확도는 H0의 기준선에서 scipy·R(`ks.test(exact=TRUE)`)과 함께 잰다.
- **`cramervonmises`**: 통계량은 정렬한 F(x_(i))에서 보상 합이다. p값은 Csörgő–Faraway(1996)의 유한 표본 분포(점근 분포 + 1/n 보정)이고, 점근 분포는 Bessel `kv` 급수(scipy와 같은 식)다. 급수의 항 수에 상한과 꼬리 상한을 둔다.

### 3.4 순위 검정 (H3)
- **Mann–Whitney U**: 정확 분포는 §3.1의 확률 점화식(크기 min(n₁, n₂)·n₁n₂/2의 배열, 대칭을 써서 절반)이다. 정규 근사는 동점 분산 보정과 연속성 보정. p값의 꼬리는 `log_ndtr`·`ndtr(−|z|)`(scipy는 n = 200 완전 분리에서 4.8e-67을 내고, 참값 대비 상대 4.3e-14다; §7).
- **Wilcoxon 부호 순위**: T⁺의 분포는 ∏(1 + z^i)/2ⁿ이고, g_k(t) = (g_{k−1}(t) + g_{k−1}(t − k))/2로 쌓는다(반으로 나누기는 정확하고 덧셈은 양수끼리). `zero_method`('wilcox', 'pratt', 'zsplit')와 동점이면 정규 근사(`zstatistic`)이고, scipy처럼 n ≤ 13이면 모든 부호를 나열한다(2¹³ = 8192, 정수 꼴 통계량으로 센다).
- **Ansari–Bradley**: AS 93(scipy의 `gscale`)의 분포를 같은 꼴의 점화식으로 쌓는다. **Mood**: 정규 근사(동점 보정은 scipy의 식).
- **Kruskal–Wallis, Friedman**: 순위 합의 카이제곱과 동점 보정(`tiecorrect`의 식). 정확 분포는 scipy에 없어 두지 않는다.
- **Brunner–Munzel**: 합친 순위와 집단 안 순위에서 분산을 만들고 t(Satterthwaite 자유도) 또는 정규다. 분산이 0이면 scipy는 경고와 함께 ±inf나 NaN을 낸다: 규칙은 §3.1의 퇴화한 입력이다.
- **`median_test`**: 2 × k 표를 만들고 §3.7의 `chi2_contingency`다(`ties`, `correction`, `lambda_`).
- **`page_trend_test`**: 정확 분포는 scipy의 재귀(Odiase–Granell)와 같은 분포를 확률로 쌓는다.
- 동점이 있는 작은 표본의 정확 p값(조건부 순열 분포)은 순위가 중간 순위(반정수)라 통계량을 2배 한 정수로 셀 수 있다. 다항식의 곱(생성함수)으로 같은 확률 점화식에 넣는다(§3.1, §9 H3).

### 3.5 적합도 2: 표본끼리 (H4)
- **`ks_2samp`**: 정확 분포는 격자 경로의 수(Hodges 1957)다. 경계 |i/n₁ − j/n₂| < d 안의 경로를 확률(각 걸음 n₁/(n₁ + n₂) 꼴로 정규화)로 세면 넘침이 없다. scipy는 n ≤ 10⁴에서 빠르다(이 초안에서 n = 10⁴에 2 ms). 정확도와 동점이 있을 때의 처리(R 4.3부터 `psmirnov`는 동점의 조건부 정확 분포)는 H4 시작 전에 scipy·R과 대조해 정한다(§9 H4). 통계량 D는 정수 격자 간격으로 비교한다(§3.1).
- **`cramervonmises_2samp`**: 정확 분포는 U = n₁Σ(r_i − i)² + n₂Σ(s_j − j)²(정수)의 분포를 격자 경로 위에서 동적 계획으로 센다(Burr 1963). scipy는 모든 분할을 나열해 max(n₁, n₂) ≤ 20으로 묶어 둔다: 동적 계획은 그보다 훨씬 큰 n에서도 정확 분포를 낼 수 있다(문턱은 scipy 그대로, `'exact'`를 고르면 더 크게; §9 H4).
- **`anderson_ksamp`**: Scholz–Stephens(1987)의 통계량(midrank, right, continuous)과 표의 보간(0.1%–25%에서 자름; scipy 그대로).
- **`epps_singleton_2samp`**: 특성함수의 표본 공분산과 그 의사역행렬(`sd.linalg.pinv`), 작은 표본 보정, 카이제곱(4).

### 3.6 상관 (H5)
- **`pearsonr`**: r 자체보다 1 − r²(와 1 ∓ r)이 p값을 정한다: p = I_{1−r²}((n − 2)/2, 1/2)(t = r√((n − 2)/(1 − r²))의 꼬리). scipy는 r을 만든 뒤 1 − r²을 계산해 r ≈ 1에서 p값이 크게 틀린다. 이 초안에서 잰 값(n = 10, y = x + ε·잡음, 정확한 유리수로 r을 구해 참 p와 비교): ε = 10⁻⁴에서 상대 3.1e-6, 10⁻⁶에서 0.015, 10⁻⁷에서 4.4(참 p 3.2e-62, scipy 1.7e-61). 1 − r²은 중심화한 y를 중심화한 x에 사영한 잔차 e = y_c − (x_c·y_c/x_c·x_c)x_c로 ‖e‖²/‖y_c‖²를 double-double로 계산한다(회귀의 R²와 같은 방법; regression §3.1). 신뢰구간은 Fisher z = artanh r을 atanh(r) = ½log1p(2r/(1 − r))로, 1 − r도 같은 잔차에서 얻는다.
- **`spearmanrho`·`spearmanr`**: 순위의 Pearson 상관과 t 근사(scipy 그대로; scipy에는 정확 분포가 없다. R의 `cor.test(method='spearman')`은 AS 89의 정확 분포를 쓴다: 넣을지는 §9 H5).
- **`kendalltau`**: 순 쌍·역순 쌍은 병합 정렬로 O(n log n)이다(scipy도 그렇다). 정확 분포는 역순 쌍의 수(Mahonian 수)의 분포를 g_n(k) = (1/n)Σ_{j<n} g_{n−1}(k − j)로 쌓는다(미끄러지는 창의 합; 양수의 평균). scipy는 "n이 크면 정밀도를 잃을 수 있다"고 적는다. τ-c와 동점 분산은 scipy의 식. scipy의 배치는 줄마다 Python 고리라 느리다(이 초안에서 10⁴ 줄 × 30에 0.98 s).
- **`weightedtau`**: 쌍곡선 가중치와 Vigna의 O(n log n) 알고리즘(scipy와 같다). p값은 scipy가 정의하지 않는다(언제나 NaN).
- **`somersd`**: 분할표에서 일치·불일치 쌍과 점근 표준오차(ASE). **`chatterjeexi`**: 순위와 Chatterjee의 점근 분산(`y_continuous`).

### 3.7 분할표 (H6)
- **`chi2_contingency`**: 기댓값 E = 행합·열합/N은 정확한 정수 곱에서 한 번 나눈다. 자유도 1이면 Yates의 보정(scipy: |O − E|를 min(0.5, |O − E|)만큼 줄인다). `lambda_`는 §3.3의 Cressie–Read다.
- **`fisher_exact` 2 × 2**: 초기하 분포는 sd.stats의 `Hypergeometric`(전체 120 이하면 정확한 128비트 개수, 그 위는 double-double 비율 점화식; stats.md §3.2)이다. 양쪽 p값은 scipy의 규칙(§3.1: 허용오차 10⁻¹⁴, 최빈값과 같으면 1; R은 1 + 10⁻⁷)이고, 그 문턱의 비교는 pmf의 비를 정확한 log 차로 한다. 통계량(표본 오즈비)은 scipy의 정의(무조건 MLE가 아닌 표본 오즈비 ad/bc).
- **`fisher_exact` r × c**: scipy는 시드 없는 몬테카를로(`random_table`)라 같은 입력에서 p값이 매번 다르다. 여기서는 rng 없이 무작위를 쓸 수 없으므로(§2.1) 기본값을 정해야 한다. 안: R의 `fisher.test`처럼 Mehta–Patel의 망 알고리즘(FEXACT)으로 정확한 p값을 내고, 작업량이 상한을 넘으면 `method=MonteCarloMethod(rng=...)`를 쓰라는 `ValueError`(§9 H6).
- **`barnard_exact`·`boschloo_exact`**: p값은 장애 모수 π에 대한 상한 sup_π P_π(T ≥ t_obs)이고, P_π는 π의 다항식(차수 N)이다. scipy는 `n`(기본 32)개 점의 격자에서 시작하는 `shgo`(전역 최적화)라 상한을 놓쳐도 알 수 없다(이 초안의 세 표에서는 n = 512와 1e-13 안). 여기서는 구간마다 다항식의 Bernstein 계수로 위 경계를 두고 경계가 최댓값 후보보다 작은 구간을 버리는 분할(가지치기)로 최댓값을 상대 1e-14 안에서 찾는다. 참값은 같은 분할을 Arb 구간으로 한 엄밀한 상한이다(§7). `n`의 뜻을 어떻게 둘지는 §9 H6이다.
- **`mcnemar`**: `exact=True`면 b + c번의 이항(1/2) 양쪽 p값(`binomtest`의 커널), 아니면 카이제곱(1)과 연속성 보정. **`cochrans_q`**: Q 통계량과 카이제곱(k − 1).
- **`odds_ratio`**(조건부 MLE와 구간)는 Fisher의 비중심 초기하 분포가 필요하다(sd.stats에 없다; §9 H6).

### 3.8 비율과 이항 (H7)
- **`binomtest`**: 꼬리는 `sd.special.bdtr`·`bdtrc`(정확한 꼬리). 양쪽 p값은 관측 쪽의 반대편에서 pmf ≤ pmf(k)·(1 + 10⁻⁷)인 경계를 이분으로 찾는다(scipy·R의 규칙). `proportion_ci`의 'exact'(Clopper–Pearson)은 `betaincinv`, 'wilson'·'wilsoncc'는 닫힌 꼴(이차식의 근을 상쇄 없는 꼴로)이다.
- **`quantile_test`**: 이항 꼬리와 순서통계량 구간. **`poisson_means_test`**: Krishnamoorthy–Thomson의 E-test는 두 포아송의 이중 무한 합이다. scipy는 경험적인 범위로 자른다. 여기서는 잘라 낸 꼬리의 상한(포아송 꼬리)을 계산해 상대 2⁻⁶⁰ 아래가 될 때까지 넓힌다(상한 있음, §1.3).
- **`proportions_ztest`·`proportions_chisquare`**: statsmodels의 식(`prop_var`, 하나·여럿의 비율, `value`). 정규·카이제곱의 작은 쪽 꼬리.
- **`proportion_confint`**: normal(Wald), agresti_coull, beta(Clopper–Pearson: `betaincinv`), wilson, jeffreys(Beta(k + ½, n − k + ½)의 분위수), binom_test(양쪽 이항 검정을 뒤집기: statsmodels는 `brentq`로 수치 역함수; 여기서는 `sd.optimize.root_scalar`와 단조성으로 구간의 끝을 1e-14로). `alternative`의 한쪽 구간.
- **`test_proportions_2indep`·`confint_proportions_2indep`**: statsmodels 0.15.0의 정의(차의 wald·agresti-caffo·newcomb·score, 비의 log·log-adjusted·score, 오즈비의 logit·logit-adjusted·logit-smoothed·score; Miettinen–Nurminen의 `correction`). score 구간은 점수 통계량을 뒤집는 근 찾기(statsmodels는 `brentq`와 경계 근처의 실패를 경고로 처리)이고, Miettinen–Nurminen의 제한된 MLE는 삼차 방정식의 닫힌 꼴이다(삼각함수 꼴의 근에서 상쇄가 있는 곳은 H7에서 Arb로 확인한다). statsmodels가 "실험 중, 기본값이 바뀔 수 있다"고 적은 함수라 기본 방법은 0.15.0의 것(차 agresti-caffo(검정)·newcomb(구간), 비 log-adjusted, 오즈비 logit-adjusted)으로 고정해 적는다.

### 3.9 다중 비교와 보정 (H8)
- **보정의 한 API(안)**: 같은 일을 하는 함수가 셋이다: scipy `false_discovery_control(ps, *, axis=0, method='bh')`(보정한 p값 배열; bh, by), statsmodels `multipletests(pvals, alpha, method='hs', ...)`(기각 여부, 보정 p값, Šidák·Bonferroni의 보정 α), `fdrcorrection(pvals, alpha, method='indep')`(fdr_bh·fdr_by의 기각과 보정 p값).
  - `false_discovery_control`: scipy의 이름과 뜻 그대로 둔다(scipy에 있는 개념). 보정한 p값만 돌려준다(배열; 결과 객체가 아니다).
  - `multipletests(pvals, /, *, alpha=0.05, method='holm-sidak', max_iter=1, axis=None)`: statsmodels에만 있는 FWER 보정과 두 단계 FDR을 맡는다. 방법은 statsmodels의 긴 이름 열 개만 받는다(`'b'`, `'hs'` 같은 줄임은 별칭이라 두지 않는다). `maxiter`는 O14의 꼴인 `max_iter`다. `is_sorted`·`returnsorted`는 두지 않는다(정렬은 커널이 한다). 결과는 `reject pvalues_corrected alpha_sidak alpha_bonferroni`의 객체이고 statsmodels의 튜플 순서로 풀린다(이름은 §9 H8).
  - `fdrcorrection`은 두지 않는다: `multipletests(method='fdr_bh'|'fdr_by')`와 `false_discovery_control`이 같은 값을 낸다(별칭 없음).
  - 세 이름이 한 커널(`csrc/htest/multitest.hpp`)을 쓴다: `false_discovery_control(method='bh')`와 `multipletests(method='fdr_bh').pvalues_corrected`는 비트까지 같다.
- **계산**: 정렬과 누적 최소·최대(단계 내림·올림). Šidák의 1 − (1 − p)^m은 −expm1(m·log1p(−p))로 계산한다(작은 p에서 상쇄가 없다; statsmodels는 그대로 계산한다). Hommel은 O(m²)의 정의식 대신 O(m log m)의 알고리즘(Meijer–Krebs–Goeman 2019)으로 R의 `p.adjust`와 같은 값을 내는 안이다. 보정 p값은 1에서 자른다. 두 단계 FDR(fdr_tsbh, fdr_tsbky)의 보정 p값은 `alpha`에 따라 달라진다(statsmodels가 적은 대로).
- **`combine_pvalues`**: Fisher의 −2Σlog p는 카이제곱(2k)의 위꼬리이고 이것은 포아송 cdf의 닫힌 꼴이다(log 쪽에서 합). Stouffer는 Σw_iΦ⁻¹(1 − p_i)를 `ndtri`의 꼬리 대칭으로(p가 작을 때 1 − p를 만들지 않는다). Pearson, Tippett(1 − (1 − p_min)^k를 −expm1 꼴로), Mudholkar–George(t 근사).
- **`tukey_hsd`**: 스튜던트화 범위 분포의 cdf·ccdf가 필요하다(sd.stats에 없다). 이중 적분(정규 범위의 분포를 카이 분포로 섞기)을 적응 Gauss–Kronrod로 계산하고 Arb의 구간 적분으로 확인한다. 공개 분포로 둘지는 §9 H8. 신뢰구간은 그 icdf다.
- **`dunnett`**: 다변량 t의 cdf는 sd.stats의 `MultivariateT.cdf`(d ≤ 3은 1-D 적분으로 1e-14, d ≥ 4는 고정 시드의 결정적 QMC, `atol` 1e-6; T29)를 쓴다. 그래서 scipy와 달리 `rng` 없이도 결정적이다. d ≥ 4의 p값은 느슨한 계약이다(§4). 신뢰구간은 cdf를 뒤집는 근 찾기다.

### 3.10 재표본과 몬테카를로 (H9)
- **`permutation_test`**: 가능한 순열의 수가 `n_resamples` 이하이면 모두 나열한다(scipy와 같다; 결과가 결정적이고 `rng`는 쓰지 않는다). 그보다 많으면 `rng`로 뽑는다. p값은 scipy의 정의((b + 1)/(n + 1)의 몬테카를로 꼴, 정확한 나열은 b/n; 같은 값의 비교에 scipy의 상대 허용오차 `eps·100`쯤을 둔다: H9에서 scipy 소스로 확인). 순열은 C++에서 Philox 스트림으로 만들고(sd.random), 사용자 `statistic`은 `batch`개씩 묶은 배열로 부른다(`vectorized`).
- **`bootstrap`**: percentile, basic, BCa. BCa의 가속 계수는 잭나이프(n번의 statistic 호출)이고, 보정한 분위수는 `ndtr`·`ndtri`다. 분위수는 sd.quantile의 정의(scipy의 `np.percentile` 보간과 같은 것)를 쓴다.
- **`monte_carlo_test`**: 사용자 `rvs(size, ...)`로 귀무 표본을 만든다. scipy의 `rvs`는 numpy 생성기를 받는 호출이다: 여기서는 `rvs`가 `rng`를 받는 꼴(sd.stats 분포의 `sample(shape, rng=rng)`와 같은 규약)로 정한다(§9 H9).
- **`method=` 객체**: H1–H8의 검정 가운데 scipy가 `PermutationMethod`·`MonteCarloMethod`·`BootstrapMethod`를 받는 것(`ttest_ind`, `pearsonr`, `spearmanrho`, `mannwhitneyu`, `wilcoxon`, `ansari`, `chatterjeexi`, `anderson`, `anderson_ksamp`, `chi2_contingency`, `fisher_exact`)에 같은 객체를 받는다.
- **`goodness_of_fit`**: 분포 족의 모수를 데이터에 맞추고(최우추정), 맞춘 분포에서 몬테카를로로 같은 맞추기를 되풀이한다. sd.stats에는 `fit`이 없다(stats.md §1.2). 넣으려면 족마다 최우추정이 필요하다(§9 H9).
- **`multiscale_graphcorr`**: 거리 행렬, 국소 상관의 지도, 순열 p값. O(n²) 메모리와 n²·순열 수의 계산이다(§9 H9).
- 재표본의 결과는 같은 `rng` 상태와 같은 입력이면 스레드 수와 상관없이 비트까지 같다: 순열·표본은 고정된 덩어리로 나눈 스트림에서 만든다(sd.random의 `spawn` 규칙).

---

## 4. 정확도 계약 (안)
- **통계량**: 입력 double을 정확한 값으로 보고 Arb로 계산한 통계량에 대해 상대 1e-14다(보장 영역). 순위·개수에서 나오는 통계량(U, T⁺, 순위 합, 역순 쌍, 분할표의 개수)은 정확한 값이다. 보장 영역은 재고 나서 묻는다(T24처럼): 기본안은 합이 넘치지 않는 유한한 데이터, n ≤ 10⁷, 분모가 0이 아닌 곳이고, 상쇄를 식으로 지운 통계량(§3.2–§3.6의 t, G, Bartlett, 1 − r²)은 조건수에 상관없이다.
- **p값**: 정확한 통계량에서 Arb로 계산한 p값에 대해 |Δp/p| ≤ 1e-14·max(1, κ_p)다(κ_p는 §3.1의 통계량에 대한 조건수; 정확한 정수 통계량이면 κ_p 항이 없다). 서브노멀 p값은 절대 45·2⁻¹⁰⁷⁴다(special과 같다). 정확한 귀무분포의 p값은 Arb나 정확한 유리수 참값에 대해 상대 1e-14다.
- **정의한 근사**: 점근 근사, 정규화 변환(D'Agostino, Royston, Hill), 보간 표(`anderson`, `anderson_ksamp`)의 p값은 "그 근사식의 값"에 대한 계약이다. 근사가 참 귀무분포와 얼마나 다른지(실제 1종 오류율)는 계약이 아니고, H0–H9에서 정확 분포나 몬테카를로로 재서 문서화한다(scipy가 경고로 알리는 곳과 같은 정보).
- **신뢰구간**: 끝점은 그 구간을 정의하는 식(분위수, 뒤집은 검정)에 대해 icdf와 같은 기준(|Δx/x| ≤ 1e-14·max(1, κ))이다.
- **느슨한 영역**(잰 값을 문서화): `dunnett`의 d ≥ 4(다변량 t의 QMC, `atol` 1e-6; T29), 재표본과 몬테카를로의 p값(무작위; 계약은 같은 `rng`면 같은 비트와 정의식뿐), Barnard·Boschloo의 큰 N(분할의 상한에 닿으면 NaN), Kolmogorov–Smirnov의 정확 분포가 큰 n에서 점근으로 넘어가는 곳(Simard–L'Ecuyer의 선택 경계).
- **값을 내지 못하면 NaN**: 상한(§1.3)에 닿거나 남는 반올림의 상한이 계약을 넘는 것을 보이지 못하면 NaN이다(stats의 T10 규칙).
- scipy·R·statsmodels와의 차이는 계약이 아니라 정확도 표의 비교 패키지 열로 보인다(H4). 비트 동일은 목표가 아니다.
- **H1에서 잰 것**(tests/htest/test_parametric.py가 지킨다): 67문제(t 32, 분산분석 10, Alexander–Govern 4, Bartlett 6, Levene 7, Fligner 8) 모두 계약 안이다. 통계량은 최대 3.8e-16(오프셋 10⁸–3·10⁹, 거의 같은 분산을 포함해), `df`·`df_num`·`df_denom`은 2.1e-16, p값은 max(1, κ_p)로 나눈 상대오차가 최대 3.8e-16(가장 큰 상대오차는 Alexander–Govern의 꼬리 5.8e-14, κ_p = 240), 신뢰구간의 끝은 추정값과 반폭의 크기에 대해 1e-15 안(끝 자체가 상쇄로 작은 `ttest_1samp/small`의 아래 끝은 상대 7.8e-15)이다. 통계량의 보장 영역은 유한한 데이터 전부다(가장 큰 크기가 2⁵⁰⁰ 이상이거나 2⁻⁴⁰⁰ 아래인 줄은 2의 거듭제곱 단위로 계산한다, §3.2). 입력이 서브노멀이면 입력 자체가 이미 자릿수를 잃은 값이다.

---

## 5. 성능 (안)
- scipy의 검정은 `_axis_nan_policy` 장식자를 지나 호출 하나의 고정 비용이 크다(이 초안에서 n = 10: `pearsonr` 80 µs, `kendalltau` 105 µs, `mannwhitneyu` 149 µs, `fisher_exact` 2 × 2 180 µs, `ttest_ind` 208 µs, `wilcoxon` 242 µs, `barnard_exact` 756 µs, `boschloo_exact` 2.1 ms). 작은 표본에서는 이 비용을 줄이는 것이 이득의 대부분이다: 입력 정리는 기술통계의 길(stats §5의 줄 행렬)을 쓰고, 검정 하나의 계산은 C++ 한 번 호출이다.
- **배치**: 줄마다 독립이므로 줄을 스레드풀에 나눈다(stats의 기술통계와 같다). scipy의 배치는 벡터화된 곳(t 검정: 10⁴ 줄 × 30에 2.9 ms)과 줄마다 Python 고리인 곳(`kendalltau` 985 ms, `mannwhitneyu` 16 ms)이 섞여 있다. 정확 분포가 필요한 배치는 같은 (n₁, n₂)의 분포를 호출 안에서 한 번만 만든다.
- **정렬**: 순위 검정은 줄마다 정렬 하나(sd.stats의 rankdata 커널). Kendall은 병합 정렬의 역순 셈.
- **정확 분포의 비용**: Mann–Whitney O(n₁n₂·min(n₁, n₂)), Wilcoxon O(n³), Kendall O(n³), KS 두 표본 O(n₁n₂), CvM 두 표본 O(n₁n₂·U의 범위). 메모리 상한을 두고 넘으면 `'auto'`는 점근으로(scipy가 고르는 곳은 그대로), `'exact'`는 `ValueError`다.
- **재표본**: 순열·표본 만들기는 C++(병렬, 결정적)이고 시간의 대부분은 사용자 `statistic`이다. 기준은 같은 `n_resamples`에서 scipy 대비 사용자 함수를 뺀 오버헤드와, 내장 통계량(t, 평균의 차 등)을 C++로 계산하는 빠른 길(§9 H9)이다.
- 크기: n = 10, 100, 10⁴, 10⁶(정확 분포가 있는 검정은 문턱 근처까지), 배치 10⁴ 줄 × 30, 분할표 2 × 2·5 × 5·N = 10⁴. 측정은 1·10스레드. 비교 대상은 scipy, statsmodels(H7·H8), R(정확도 측정과 같은 함수)이다. 사례는 벤치 세션이 `bench/speed/cases/htest.py`에 넣는다(O101과 같다; 이쪽은 목록을 넘긴다).

---

## 6. 층과 파일 (안)
- `src/stardust/htest/`: `__init__.py`(공개 이름), `_common.py`(입력 정리: 데이터 변환, `axis`·배치·`keepdims`, `nan_policy`, `alternative`, `rng`, 메시지; `Samples(scan=False)`는 NaN을 C++ 커널에 맡긴다), `_parametric.py`(H1, 구현), `_gof.py`(H2·H4), `_rank.py`(H3), `_correlation.py`(H5), `_contingency.py`(H6), `_proportion.py`(H7), `_multitest.py`(H8), `_resampling.py`(H9). 결과 클래스는 `_result.py`(공통 바탕 `_TestResult`; §2.2)에 둔다.
- `csrc/htest/`: `moments.hpp`(H1, 구현: 줄 행렬의 double-double 적률, 고정 덩어리의 스레드풀 분할, 단위, 분산 0의 판정, double-double 키의 정렬), `parametric.cpp`·`parametric.hpp`(H1, 구현: t 검정, 분산분석, 분산의 동질성), `ranks.hpp`(정렬, 중간 순위, 동점 표; `rankdata`의 커널을 같이 쓴다), `exact/`(`mwu.hpp`, `wilcoxon.hpp`, `kendall.hpp`, `ansari.hpp`, `page.hpp`, `ks.hpp`(Durbin·Pomeranz·Smirnov·Kolmogorov·두 표본 격자 경로), `cvm.hpp`, `fexact.hpp`(r × c 망 알고리즘), `nuisance.hpp`(Barnard·Boschloo의 Bernstein 분할)), `divergence.hpp`(Cressie–Read), `swilk.hpp`(AS R94), `studentized_range.hpp`, `multitest.hpp`, `resample.hpp`(순열·부트스트랩 표본의 스트림). 바인딩은 `csrc/bind/htest_bind.cpp`(`stardust._core.htest`; H1부터).
- 분포 함수는 `csrc/stats/`의 것을 C++에서 바로 부른다(Python을 거치지 않는다). 새 귀무분포(Kolmogorov–Smirnov, 스튜던트화 범위, Fisher의 비중심 초기하)는 처음에 `csrc/htest/` 안에 두고, sd.stats·sd.special에 공개하지 않는다(H14).
- `tests/htest/`(마일스톤마다 `test_<묶음>.py`, `test_api_surface.py`가 docs/api/htest.md를 읽는다), 문제 모음 `tests/htest/problems.py`, 참값 `tests/htest/ref/`, 대조 데이터 `tests/htest/data/`, 도구 `tools/htest_ref.py`(Arb와 정확한 유리수를 한 도구에)·`tools/gen_htest_scipy.py`·`tools/gen_htest_r.py`, 기준선 `bench/htest/h0/`(§7).

---

## 7. 참값과 검증
H0에서 만든 것(2026-11-20). 문제 모음은 `tests/htest/problems.py`, 참값은 `tests/htest/ref/`(80 KB), 대조 데이터는 `tests/htest/data/`(132 KB), 만드는 도구는 `tools/`에 있다. 검사는 `tests/htest/test_reference_data.py`다.

- **문제 모음** (`tests/htest/problems.py`): H1–H8의 모든 검정에 281개다(H1 44, H2 55, H3 39, H4 15, H5 24, H6 26, H7 54, H8 24). 문제 하나는 호출 하나(scipy·statsmodels의 함수 이름, 위치 입력, scipy의 키워드 옵션)이고, 검정이 만나는 입력의 종류를 덮는다: 작은 표본(정확 분포), 큰 표본(10³–3×10³), 동점과 0, 깊은 꼬리(p ≪ 10⁻¹⁶, 10⁻³⁰⁰까지), 큰 오프셋(평균 10⁸–10⁹: 통계량의 상쇄), 퇴화한 입력(상수 표본), 배치(axis=-1). 입력은 numpy PCG64의 고정 시드로 만들고 동점이 필요하면 반올림한다. 모든 데이터·참값 파일은 문제마다 입력의 SHA-256(`digest`)을 적어 두어, numpy의 스트림이 바뀌면 조용히 다른 입력이 되지 않고 테스트가 실패한다.
- **참값 도구** (`tools/htest_ref.py`): Arb(python-flint)와 정확한 유리수(Python의 int·`fractions`)를 한 도구에 두었다(계획의 `tools/htest_exact.py`를 합쳤다). 입력 double은 정확한 유리수다. 데이터의 유리 함수인 통계량(평균, 제곱합, 순위, 개수, t², r², F, U, T⁺, Kendall의 역순 쌍, KS의 격자 간격, 카이제곱)은 `Fraction`으로 정확히 계산해 한 번 반올림한다. p값은
  - 정수로 센 정확한 귀무분포: Mann–Whitney(가우스 이항계수의 계수), Wilcoxon 부호 순위(부분합의 수), Kendall(Mahonian 수), 두 표본 KS(격자 경로), Fisher의 2 × 2(초기하)와 r × c(주변합이 같은 모든 표), 이항(p는 그 double의 유리수; 항을 저장하지 않고 흘려 보내 n = 10⁵에서도 메모리가 작다);
  - Arb의 공: 정칙화 불완전 베타(t, F, 상관), 정칙화 불완전 감마(카이제곱), erfc·erfcinv(정규), 한 표본 KS의 정확 분포(Marsaglia–Tsang–Wang의 행렬 거듭제곱, Arb 행렬)와 한쪽(Birnbaum–Tingey의 양수 항 합), Kolmogorov의 극한(꼬리 상한을 공에 더한다).
  - Arb 값은 `tools/arbref.py`의 `rigorous()`로 상대 2⁻¹¹⁰까지 정착시킨다(정밀도를 두 배씩 올린다; "두 정밀도가 맞으면" 같은 어림이 없다). 결과는 double-double [hi, lo]다. 문제마다 시간 제한(SIGALRM), 10프로세스, 워커는 50문제마다 교체(`maxtasksperchild`), mpmath는 쓰지 않는다.
  - 양쪽 규칙의 허용오차는 그 double의 정확한 유리수다: 이항 1 + 10⁻⁷(scipy·R), scipy 2 × 2 Fisher 1 + 10⁻¹⁴(§3.1), r × c Fisher는 R `fisher.test`의 1 + 10⁻⁷.
  - 첫 참값(17묶음, 138문제; 10프로세스로 2초, 프로세스 RSS 합의 최대 0.3 GB): `ttest`(1표본·짝·Student·Welch·요약값 21), `f_oneway`(Welch 포함 5), `bartlett` 3, `power_divergence`(카이제곱과 Cressie–Read 여섯 이름·λ = 0.3, 14), `jarque_bera` 4, `ks_1samp`(정확·approx·asymp·한쪽 8), `mannwhitneyu`(정확·정규 8), `wilcoxon`(정확·정규·세 0 처리 11), `ks_2samp`(정확 6), `kendalltau`(정확·정규·τ-c 6), `pearsonr`(Fisher z 구간 포함 8), `spearmanrho` 4, `fisher_exact`(2 × 2 7과 3 × 3 1), `chi2_contingency` 5, `binomtest` 9, `combine_pvalues`(Fisher·Pearson·Tippett·Stouffer 6), `multipletests`(보정 p값 여덟 방법과 `false_discovery_control` 12). 참값은 scipy가 그 문제에서 고른 방법을 따르고 그 이름을 `method`에 적는다(H12). 다음 마일스톤은 `FAMILIES`에 같은 꼴로 묶음을 더한다.
  - 도구 자체의 검사(`test_reference_data.py`): 개수의 합(C(m + n, m), 2ⁿ, n!)과 대칭, 같은 크기의 두 표본 KS 경로 수와 Gnedenko–Korolyuk의 닫힌 꼴, n = 1·2의 Kolmogorov·Smirnov 분포의 닫힌 꼴, 이항과 Fisher의 규칙.
- **scipy·statsmodels 대조** (`tools/gen_htest_scipy.py` → `data/scipy_h1.json`…`scipy_h8.json`, `data/statsmodels.json`): 281문제 모두를 scipy 1.18.1·statsmodels 0.15.0의 기본값으로 돌려 결과의 모든 필드(`statistic pvalue df dof zstatistic statistic_location statistic_sign expected_freq median table method …`), 신뢰구간(95%; `binomtest`는 `proportion_ci`의 세 방법), 튜플로 풀리는 개수, 낸 경고의 종류, 또는 낸 예외를 적었다. r × c `fisher_exact`는 scipy의 기본이 시드 없는 몬테카를로라 적지 않고, `dunnett`은 `rng=12345`로 적었다.
- **R 대조** (`tools/gen_htest_r.py` → `data/r.json`): R 4.6.2 base `stats`(H15)와 Levene의 `car::leveneTest`(H33, car 3.1.5)로 같은 검정이 있는 169문제(H0 145, H1에서 24 더; `t.test oneway.test bartlett.test fligner.test chisq.test shapiro.test ks.test wilcox.test kruskal.test friedman.test ansari.test mood.test cor.test fisher.test mcnemar.test binom.test prop.test TukeyHSD p.adjust`). R의 정의가 scipy와 다른 곳(ks.test의 정확 분포가 n < 100·mn < 10⁴에서만이고 두 표본은 동점의 조건부, wilcox.test는 n < 50에서 정확, cor.test의 Spearman은 AS 89의 정확 분포, fisher.test의 오즈비는 조건부 MLE)은 R의 값 그대로 적고 비교가 그것을 가린다.
- **H0에서 본 것** (scipy·R 대 참값, 상대오차; `test_reference_data.py`가 기록된 값으로 지킨다):

  | 문제 | 참값 | scipy | R |
  |---|---|---|---|
  | `bartlett` 거의 같은 분산(n = 400 × 3, 차 10⁻⁹) | T = 1.594e-15 | 8.5e-14 (53배) | −8.5e-14 (음수) |
  | `power_divergence` G, O − E = ±10⁻³, E = 10⁶ | 2.0000001899898e-12 | 9.9e-8 | (없음) |
  | `pearsonr` y = x + 10⁻⁴·잡음(n = 10)의 p | 1.1885e-34 | 5.4e-7 | 2.4e-7 |
  | 같은 꼴, 10⁻⁶ | 1.1884e-50 | 8.3e-3 | 1.4e-3 |
  | 같은 꼴, 10⁻⁷ | 1.1884e-58 | 0.11 | 0.088 |
  | `ttest_1samp` 평균 10⁸ + 0.1, μ₀ = 10⁸(n = 200)의 t | 0.51438337273546 | 1.7e-8 | 1.8e-8 |
  | `ttest_ind` 평균 10⁸ + 0.5 대 10⁸의 t·p | 3.05469003894 | 3.4e-8, p 3.2e-7 | 1.1e-8, p 1.0e-7 |
  | `ttest_rel` 10⁹ 근처 짝의 차의 t | 545.84321315810 | 0 | 1.2e-6, p 1.2e-4 |
  | `f_oneway` 평균 10⁸(n = 40 × 3)의 F | 2.89184567295 | 1.5e-16 | 1.6e-8 |
  | `ks_1samp` n = 80의 정확 p | 1.7142e-21 | 5.5e-15 | 0(1 − cdf) |
  | `wilcoxon` n = 300 정규 근사의 p | 6.1449e-51 | 1.0e-14 | 0 |
  | `ttest_1samp([1, 1, 1], 0)` | NaN(H11) | inf, 0 | 에러 |

  - 그 밖에 scipy는 모든 참값과 상대 10⁻¹² 안이다. 계약(10⁻¹⁴·max(1, κ_p))에 가까운 곳: `ks_1samp` n = 1000의 정확 p 2.6e-13, `ks_2samp` 정확 한쪽 7.1e-14(p = 1.7e-3), `binomtest` n = 10⁵ 5.9e-14, `mannwhitneyu` 완전 분리 4.3e-14(p = 4.8e-67, z² ≈ 300), `chi2_contingency` 큰 표 7.5e-14(p = 1.5e-197), Stouffer 5.3e-14(p = 1.2e-106). 정수 통계량의 정확 분포(`mannwhitneyu`, `wilcoxon`, `kendalltau`, `fisher_exact`, `binomtest`의 작은 n)는 scipy와 R이 참값과 0–3e-16 안이다. 보정 p값(`multipletests` 여덟 방법, `false_discovery_control`, R `p.adjust`)은 모두 1e-15 안이다.
  - scipy 1.18.1의 2 × 2 `fisher_exact`는 양쪽 규칙의 허용오차가 10⁻¹⁴다(이 초안이 적었던 10⁻⁷은 `binomtest`와 R의 것이다; §3.1을 고쳤다).
  - scipy의 `wilcoxon`은 `method='auto'`로 n ≤ 50에서 정확, 동점·0이 있으면 n ≤ 13에서 부호의 나열(순열 검정), 그 밖은 정규 근사를 고르고, `zstatistic` 필드는 사용자가 `method='asymptotic'`을 준 때만 둔다(`'auto'`가 근사를 골라도 없다; §9 H3).
  - `anderson_ksamp`은 문서에 "`variant`를 주면 풀리지 않는다"고 적었지만 1.18.1의 결과는 둘로 풀린다(§9 H4).
- **H1에서 더한 것** (2026-11-20):
  - 문제 37개(H1은 76개, 모두 313개): 큰 오프셋(Welch t, Welch 분산분석, Alexander–Govern, Bartlett, Levene, Fligner, 3·10⁹와 10⁹의 짝 t), 상수 표본(H24: `ttest_1samp/offset-constant`, `ttest_ind/constant-both`, `ttest_ind_from_stats/zero-std`, `f_oneway/constant-groups`), 원소 하나의 표본(Student), Yuen(Welch, 동점), 깊은 꼬리의 한쪽 t, 두 집단 F(= t²), Welch 분산분석의 큰 표본·꼬리, Levene·Fligner의 절삭(p = 0.2)·꼬리·큰 표본, 배치 다섯. scipy·R 대조 데이터는 그 문제들만 더해졌다.
  - 참값 6묶음 67문제(10프로세스로 10초, RSS 합의 최대 1.0 GB): `ttest` 32(Yuen 포함, 모든 t 검정의 `conf_int`: alpha = 0.05, t 분위수는 Arb의 근을 부호 바뀜으로 2⁻¹⁴⁰까지 확인), `f_oneway` 10, `alexandergovern` 4(새), `bartlett` 6, `levene` 7(새: 중심과 z가 정확한 유리수라 W도 정확), `fligner` 8(새: 정확한 중간 순위, Arb의 점수). 모든 문제에 p값의 조건수 `kappa`(통계량과 계산한 자유도에 대한 |x ∂p/∂x|/p의 합; Arb의 중앙 차분)를 적었다. 퇴화한 입력의 참값은 H24·H11의 규칙이다(`ttest_1samp/constant`의 `df`는 NaN에서 2로 바뀌었다).
  - 도구의 검사(`test_reference_data.py`): 두 집단 F = t²와 같은 p, `trim=0`의 Yuen = Student, k = 2의 Welch 분산분석 = Welch t²와 같은 `df`, 두 집단 Levene = 편차의 Student t², 짝수 n의 정확한 중앙값과 절삭 평균, Fligner 점수의 대칭. scipy·R의 신뢰구간도 참값과 견준다(`CI_KNOWN`).
  - scipy·R·statsmodels가 참값에서 10⁻¹²보다 벗어나는 H1의 곳(상대오차; sd는 모두 계약 안이다):

  | 문제 | 참값 | scipy | R | statsmodels |
  |---|---|---|---|---|
  | `ttest_rel` 3·10⁹와 10⁹의 짝(n = 3)의 t·p | t = 1.654e12, p = 3.655e-25 | 1.1e-9, 2.2e-9 | 1.1e-9, 2.2e-9 | |
  | `ttest_ind` Welch, 평균 10⁸ + 0.5 대 10⁸의 t·p | 1.07720234 | 2.0e-8, 3.4e-8 | 1.9e-8, 3.2e-8 | 5.5e-8, 9.2e-8 |
  | `ttest_ind` Student 오프셋(H0)의 신뢰구간 끝 | | 9.8e-8, 2.1e-8 | 3.3e-8, 7.9e-9 | |
  | `ttest_rel` 오프셋(H0)의 신뢰구간 끝 | | 맞음 | 3.9e-7 | |
  | `f_oneway` Welch, 오프셋 10⁸의 F·p | 2.15807454 | 5.6e-8, 1.1e-7 | 7.9e-9, 1.6e-8 | 5.6e-8, 1.1e-7 |
  | `f_oneway` 오프셋(H0, Student) | 2.89184567295 | 맞음 | 1.6e-8 | 6.5e-9, 1.8e-8 |
  | `alexandergovern` 오프셋 10⁸의 A·p | 6.14093501 | 2.7e-8, 8.4e-8 | (없음) | |
  | `bartlett` 오프셋 10⁸의 T·p | 24.2874942 | 맞음 | 2.6e-9, 3.2e-8 | |
  | `levene` 오프셋 10⁸의 W·p | 5.52641933 | 1.5e-11, 7.4e-11 | 8.4e-9, 4.2e-8(`car::leveneTest`) | |
  | `fligner` 짝수 n의 큰 표본(1500, 1200, 1000)의 통계량·p | 149.678241, p = 3.146e-33 | 8.5e-11, 6.4e-9 | 같다 | |
  | `fligner` 꼬리(500 × 2) | 264.950783, p = 1.430e-59 | 2.2e-9, 2.9e-7 | 같다 | |
  | 상수 표본: `ttest_ind`, `ttest_ind_from_stats`, `f_oneway` | NaN(H24, H11) | −inf·0, −inf·0, inf·0 | 에러(t.test), inf·0(oneway.test) | 값을 낸다 |

  - statsmodels(`weightstats.ttest_ind`, `oneway.anova_oneway`)는 H0의 오프셋 문제에서도 t·p 3.4e-8·3.2e-7, F·p 6.5e-9·1.8e-8 틀린다. Welch t 오프셋의 신뢰구간 끝은 scipy 2.4e-8·7.1e-9, R 2.2e-8·7.7e-9다.
  - Fligner의 차이는 중앙값의 반올림이다: 짝수 n의 두 가운데 값은 정확한 중앙값에서 같은 거리라 같은 순위(동점)인데, scipy와 R은 중앙값을 double로 반올림해 그 동점을 가르고 두 순위를 1/2 옮긴다(둘이 서로 1e-15 안에서 같고 참값과 다르다). sd는 중앙값을 double-double로 두어 동점이 남는다. Levene의 1.5e-11도 같은 중앙값의 반올림이다.
  - Levene의 R 대조(H33, `car::leveneTest`; tools/gen_htest_r.py, `data/r.json`에 7문제): 세 중심(R의 `median`, `mean`, `mean(trim = p)`) 모두 통계량·p·`Df`가 참값과 맞고(3e-14 안), 오프셋 10⁸에서만 통계량 8.4e-9, p 4.2e-8 틀린다(R의 중앙값도 double로 반올림한다; scipy는 1.5e-11).
  - R의 `var.test`(두 분산의 F 비)는 scipy에 짝이 없어 H1에 없다(scipy의 분산 검정은 `bartlett levene fligner`다).
- **H1 속도** (`bench/htest/h1/speed.py` → `speed.md`, `speed.json`; scipy는 1스레드, sd는 1·10스레드): 10스레드에서 모든 경우에 scipy보다 빠르다. n = 10에서 10–40배(호출 하나가 8–13 µs, scipy 0.12–0.41 ms), 10³에서 2–21배, 10⁶에서 1.0–6.4배(t 0.65–1.3 ms, scipy 1.6–3.3 ms; `levene` 23 ms 대 25 ms, `fligner` 95 ms 대 344 ms, Yuen 21 ms 대 138 ms), 배치 10⁴ × 30에서 2.4–5.7배. 1스레드에서는 10⁶과 배치에서 scipy보다 느리다(t 10⁶ 5.0 ms 대 1.6 ms): double-double 적률이 원소마다 3번 훑기(최대 크기, 합, 제곱합)에 4–5 ns다. `levene`·`fligner`의 긴 줄은 집단마다의 중앙값 선택(`nth_element`)과 병합의 마지막 단계가 한 스레드다. 둘 다 H10의 후보다.
- **속도 기준선** (`bench/htest/h0/speed.py` → `speed.md`, `speed.json`): scipy·statsmodels의 대표 검정 41개를 n = 10, 10³, 10⁶과 배치 10⁴ × 30에서 1스레드로 쟀다. 작은 표본의 고정 비용은 0.06–0.25 ms(`chisquare` 0.07, `pearsonr` 0.08, `ttest_ind` 0.19, `wilcoxon` 0.21 ms; `tukey_hsd`는 18 ms, `cramervonmises_2samp`의 정확 분포 2.2 ms, `boschloo_exact` 2.0 ms)이고, 10⁶에서는 정렬하는 검정이 150–340 ms(`mannwhitneyu` 196, `kruskal` 297, `brunnermunzel` 339, `chatterjeexi` 310), `weightedtau`가 2.6 s, 적률 검정이 2–20 ms다. 배치에서 줄마다 Python 고리인 `kendalltau`는 950 ms, `ks_2samp`은 88 ms다.
- **정확도·속도 체계에 넣는 계획** (두 체계는 다른 세션의 파일이라 H0은 고치지 않았다):
  - `bench/accuracy/sources.py`: `AREAS`에 `"htest"`를 더하고 `sweeps()`에 `if "htest" in areas: from . import htest; out += htest.sweeps()`.
  - 새 파일 `bench/accuracy/htest.py`: `tests/htest/ref/<묶음>.json`마다 `Custom("htest", 묶음, run, compare)`. `run`은 문제마다 `sd.htest`의 함수를 불러(`problems.py`의 입력, scipy 키워드를 sd 키워드로 바꾸는 표 하나) 필드마다의 상대오차를 `common.region()`으로 모은다. 영역(region)은 `statistic`과 `pvalue`(그리고 `df`·`df_num`·`df_denom`, `conf_int`)이고 계약은 §4다: 통계량 1e-14, p값 1e-14·max(1, κ_p)(정수 통계량이면 κ_p 항 없이), 정의한 근사는 그 식에 대한 1e-14. κ_p는 참값 도구가 함께 적는다(H1부터 `kappa` 필드). `compare`는 `common.compare_libs`로 scipy·statsmodels(실행 중에 부른다)와 R(`tests/htest/data/r.json`의 고정 값; R이 있으면 `rsession`으로 다시 부를 수 있다)의 같은 영역을 잰다.
  - `bench/accuracy/tables.py`: `HEAD["htest"] = "sd.htest"`, `INTRO["htest"]`(참값의 출처와 계약을 적은 한 문단). 표는 `bench/htest/accuracy.md`.
  - `bench/accuracy/compare.py`: 할 것 없음(`LIBS`에 statsmodels가 이미 있다).
  - `bench/speed/registry.py`: `AREAS`에 `"htest"`. 새 파일 `bench/speed/cases/htest.py`: `bench/htest/h0/speed.py`의 사례를 `Case(area="htest", group=마일스톤, name=검정, size, sd=…, refs=(Ref("scipy", …), Ref("statsmodels", …), RRef("R", "t.test(x, y)", …)))`로 옮기고, `TITLES`에 마일스톤의 제목. 마일스톤마다 공개된 함수의 사례만 더한다(O101과 같다: 벤치 세션이 넣고 이쪽은 목록을 넘긴다).
  - `tools/golden.py`: `AREAS`에 `"htest"`, 새 파일 `tests/golden/cases_htest.py`(H1의 첫 공개 함수부터).
  - H1의 속도 사례(`bench/speed/cases/htest.py`, group "H1"; sd는 1·10스레드, refs는 scipy와 R): `ttest_1samp`(R `t.test(x, mu = 0)`), `ttest_rel`(`t.test(x, y, paired = TRUE)`), `ttest_ind` Student·Welch(`t.test(x, y, var.equal = TRUE/FALSE)`)·`trim=0.2`(scipy만), `ttest_ind_from_stats`(scipy만, 스칼라와 10⁴개 배열), `f_oneway` 세 집단 Student·Welch(`oneway.test(y ~ g, var.equal = TRUE/FALSE)`), `alexandergovern`(scipy만), `bartlett`(`bartlett.test`), `levene` 세 중심(R `car::leveneTest(y, g, center = ...)`, H33), `fligner`(`fligner.test`, 중앙값 중심). 크기는 bench/htest/h1/speed.py와 같다: 표본마다 n = 10, 10³, 10⁶과 배치 10⁴ × 30(R은 배치 없이).
  - H1의 정확도 사례(`bench/accuracy/htest.py`): 묶음 `ttest f_oneway alexandergovern bartlett levene fligner`, 영역 `statistic pvalue df df_num df_denom conf_int`, 비교 열은 scipy·statsmodels(`weightstats.ttest_ind`, `oneway.anova_oneway`)·R(`data/r.json`).
  - H1의 골든 사례(`tests/golden/cases_htest.py`; 결과 객체는 모든 필드와 `conf_int()`, `conf_int(alpha=0.1)`을 적는다): 아홉 함수 각각 고정 데이터 셋(작은 표본, 오프셋 10⁸, 동점, 상수 표본, 원소 하나, NaN의 세 `nan_policy`, 빈 표본) × 옵션(`alternative` 셋, `equal_var` 둘, `trim` 0·0.2, `center` 셋과 `proportiontocut` 0.2, `keepdims`), 배치(axis=-1, 축 둘의 tuple), 긴 줄(n = 10⁵, 여러 스레드의 커널), 스케일 1e±300, `popmean` 배열, `ttest_ind_from_stats`의 브로드캐스트.
- **아직 계획인 것**: 장애 모수의 상한(Barnard·Boschloo, H6)은 Arb 구간 산술의 분할로 엄밀한 sup을 구한다(§3.7). 정의한 근사(AS R94, D'Agostino, Hill, 보간 표)는 같은 정의식을 Arb로 계산한다(상수는 scipy·R 소스의 십진 상수 그대로). 근사의 품질(점근·근사 p값의 실제 1종 오류율)은 정확 분포(작은 n)와 몬테카를로(큰 n, sd.random)로 재서 표로 둔다(계약이 아니다, §4). 공개 함수마다 `tools/golden.py`에 고정 입력의 결과를 더한다(API.md §4 규칙 1).

---

## 8. 마일스톤
| 마일스톤 | 내용 |
|---|---|
| H0 준비 (완료, 승인 대기 2026-11-20) | API 문서 docs/api/htest.md(모든 공개 함수의 시그니처, scipy·statsmodels 이름 → sd 이름의 대응표, 마일스톤별 질문), 결과 객체의 틀(`_TestResult`, `conf_int`), 공통 입력 층(`axis`·배치·`nan_policy`·`alternative`·`rng`·`method`·정확한 방법의 상한), 참값 도구(Arb·정확한 유리수), scipy·statsmodels·R 대조 도구와 데이터, 속도 기준선, 정확도·속도 체계에 넣는 계획. 결과는 §2.2·§7 |
| H1 평균과 분산 (완료, 승인 대기 2026-11-20) | `ttest_1samp ttest_rel ttest_ind ttest_ind_from_stats f_oneway alexandergovern bartlett levene fligner`(Welch 분산분석은 `f_oneway(equal_var=False)`). 결정 H22–H27, 방법 §3.2, 정확도 §4, 참값·비교·속도 §7 |
| H2 적합도 1 | `chisquare power_divergence skewtest kurtosistest normaltest jarque_bera shapiro anderson ks_1samp kstest cramervonmises` |
| H3 순위 검정 | `mannwhitneyu ranksums wilcoxon kruskal friedmanchisquare brunnermunzel ansari mood median_test page_trend_test` |
| H4 적합도 2 | `ks_2samp cramervonmises_2samp anderson_ksamp epps_singleton_2samp` |
| H5 상관 | `pearsonr spearmanrho spearmanr pointbiserialr kendalltau weightedtau somersd chatterjeexi` |
| H6 분할표 | `chi2_contingency fisher_exact`(2 × 2, r × c) `barnard_exact boschloo_exact mcnemar cochrans_q`(와 §9 H6의 `odds_ratio relative_risk`) |
| H7 비율과 이항 | `binomtest quantile_test poisson_means_test proportions_ztest proportions_chisquare proportion_confint test_proportions_2indep confint_proportions_2indep` |
| H8 다중 비교와 보정 | `tukey_hsd dunnett multipletests false_discovery_control combine_pvalues` |
| H9 재표본 | `permutation_test monte_carlo_test bootstrap`, `PermutationMethod MonteCarloMethod BootstrapMethod`와 앞 검정들의 `method=`, `bws_test`(와 §9 H9의 `multiscale_graphcorr goodness_of_fit`) |
| H10 최적화와 일관성 점검 | 모든 공개 함수의 속도를 크기마다 scipy·statsmodels·R과 견주어 고치기(H1에서 넘어온 것, H32: 1스레드의 double-double 적률 훑기를 NEON으로, `levene`의 중앙값 선택을 병렬로, `fligner` 정렬의 병합 단계를 병렬로), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다. H1–H8은 서로 기대는 것이 적어 순서를 바꿀 수 있다(H6의 `median_test`가 쓰는 `chi2_contingency`는 H3이 먼저면 H3에서 만든다).

---

## 9. 마일스톤을 시작하기 전에 물을 것
추천을 첫째에 둔다.

- **H0**: 모두 정했다(§1.1 H6–H15).
- **regression과의 경계**: H16–H21로 정했다(§1.1, §1.4). 남은 것은 regression 쪽의 결정이다(docs/design/regression.md §8의 "htest와의 관계"): 새 회귀 검정의 결과 꼴(H6·H7·H11·H19·H21에 맞출지, 내보낼지), `breusch_pagan()`의 자유도 필드, `wresid`, 단계적 선택의 보정 옵션(둔다면 `multipletests`의 방법 이름과 같은 커널). htest 쪽은 H2에서 Jarque–Bera의 커널과 클래스를 옮길 때 namedtuple에만 있던 동작과 골든의 꼴을 확인해 묻는다.
- **H1**: 모두 정했다(§1.1 H22–H33).
- **H2**: `kstest` 감싸개: 두지 않는다(추천: `ks_1samp`·`ks_2samp`을 고르기만 하는 감싸개이고, 문자열 분포 이름과 `N`으로 표본을 만드는 옛 꼴이 섞여 있다) / scipy처럼 둔다. 분포 인자: sd.stats 분포 객체와 호출 가능한 `cdf`(추천) / scipy의 문자열 이름도. `anderson`의 `dist`: sd.stats의 분포 클래스(`dist=sd.stats.Normal`; 추천. `gumbel_l`은 sd.stats에 없어 빠진다) / scipy의 문자열(별칭 'gumbel', 'extreme1' 없이). `anderson`의 `method` 기본값 `'interpolate'`(추천; scipy 1.18은 주지 않으면 경고하고 옛 꼴을 낸다) / 꼭 주게. Shapiro–Wilk는 AS R94의 정의(추천) / 정확한 기대 순서통계량의 계수(다른 검정이 된다).
- **H3**: 동점이 있을 때 `'exact'`: 조건부 순열 분포(추천, `[scipy와 다름]`) / scipy처럼 동점이 없는 분포 / `ValueError`. `'auto'`의 문턱: scipy 그대로(추천) / 정확이 싼 곳까지 넓히기. `ranksums`(scipy는 legacy, `mannwhitneyu`를 권한다): 둔다(추천, 같은 커널) / 뺀다. `wilcoxon`의 `zstatistic` 필드는 정규 근사로 계산한 때마다 둔다(추천; scipy 1.18.1은 사용자가 `method='asymptotic'`을 준 때만이고 `'auto'`가 근사를 골라도 없다: §7) / scipy 그대로. 배치의 `method` 필드(H12): 줄마다의 문자열 배열(추천) / 줄이 모두 같을 때만 str. `median_test`에 `axis`(배치)를 더한다(추천) / scipy처럼 없다.
- **H4**: `ks_2samp`의 동점: scipy의 정확 분포(동점 무시) 그대로(추천, H4 시작 전에 scipy 소스·R `psmirnov`와 대조하고 다시 묻는다) / R 4.3처럼 조건부 정확. `cramervonmises_2samp`의 `'exact'`를 동적 계획으로 큰 n까지(추천; `'auto'` 문턱은 20 그대로). `anderson_ksamp`의 기본 `variant='midrank'`(추천; scipy 1.18.1은 주지 않으면 경고하고 옛 꼴)와 풀기: scipy 1.18.1의 실제 동작(둘로 풀린다; 추천) / scipy 문서(풀리지 않는다).
- **H5**: `spearmanr`(열들의 상관 행렬 꼴)과 `spearmanrho`(1.18의 새 꼴): 둘 다(추천: scipy에 둘 다 있고 동작이 다르다) / `spearmanrho`만 / `spearmanr`만. Spearman의 정확 분포(R의 AS 89): 넣지 않는다(추천, scipy에 없다) / `method='exact'`로 더한다. `weightedtau`: 넣되 `pvalue`는 NaN(scipy 그대로; 추천) / 결과에 `pvalue` 필드를 두지 않는다 / 뺀다. 결과의 `correlation` 별칭: 두지 않는다(추천).
- **H6**: r × c `fisher_exact`의 기본: 정확한 망 알고리즘(R `fisher.test`; 추천, `[scipy와 다름]`: scipy는 시드 없는 몬테카를로) / `method`를 꼭 주게(`MonteCarloMethod(rng=...)`). `barnard_exact`·`boschloo_exact`의 `n`: 받되 분할의 처음 격자 크기로만 쓴다(결과는 `n`과 상관없이 참 상한; 추천) / scipy처럼 정확도를 정하는 인자로. `odds_ratio`·`relative_risk`(scipy.stats.contingency, 신뢰구간이 있는 추정): 넣는다(추천: H7의 비·오즈비 구간과 한 묶음이고 조건부 오즈비는 Fisher 검정과 같은 분포) / 뺀다. 소수의 개수(가중 표): scipy처럼 받는 곳은 받는다(`chi2_contingency`, 추천) / 정수만. `mcnemar`의 기본 `exact=True`(statsmodels 그대로; 추천) / R `mcnemar.test`처럼 카이제곱. `chi2_contingency`의 `df`는 float(추천: H10의 다른 `df`와 같은 형) / scipy처럼 int.
- **H7** (statsmodels 이름의 질문): 
  1. (H16으로 정했다: `proportion_confint`·`confint_proportions_2indep`의 신뢰수준은 `alpha=0.05`.)
  2. `proportion_confint`의 방법 이름: scipy에 있는 것은 scipy의 이름(`'exact'`(Clopper–Pearson; statsmodels `'beta'`), `'wilson'`, 더해서 scipy의 `'wilsoncc'`)과 statsmodels에만 있는 것(`'normal'`, `'agresti_coull'`, `'jeffreys'`, `'binom_test'`)(추천) / statsmodels 이름 그대로(`'beta'`). 기본값: statsmodels의 `'normal'`(추천, 규칙 2) / scipy `proportion_ci`의 `'exact'`.
  3. `alternative`: `'less'`·`'greater'`(추천, H5) / statsmodels의 `'smaller'`·`'larger'`.
  4. `proportions_ztest`의 `prop_var=False`(False 또는 실수): None 또는 실수(추천, bool과 수가 섞인 인자를 피한다) / 그대로.
  5. `test_proportions_2indep`의 `return_results`: 두지 않는다(언제나 결과 객체, (statistic, pvalue)로 풀린다; 추천). 결과 필드는 statsmodels의 것(`statistic pvalue compare method diff ratio odds_ratio variance alternative value prop1_null prop2_null`).
  6. `compare`의 값: statsmodels의 `'diff'`, `'ratio'`, `'odds-ratio'` 그대로(추천).
  7. 기본 방법: statsmodels 0.15.0의 것으로 고정(추천; statsmodels가 바뀌어도 따라가지 않는다고 적는다) / 매번 statsmodels를 따른다.
  8. `binom_test`(statsmodels): 두지 않는다(`binomtest`와 같다; 추천).
  9. `proportions_chisquare`의 Yates 보정(R `prop.test`의 `correct=TRUE`): statsmodels처럼 없다(추천, R과는 `correct=FALSE`로 견준다) / `correction=`을 더한다.
- **H8**: 보정의 API(§3.9): `false_discovery_control`(scipy) + `multipletests`(statsmodels, 열 방법), `fdrcorrection` 없음(추천) / `multipletests` 하나로(scipy의 것도 뺀다) / 셋 다. `multipletests`의 기본 방법: statsmodels의 `'holm-sidak'`(추천, 규칙 2) / R `p.adjust`의 `'holm'`. 결과 필드 이름: `reject pvalues_corrected alpha_sidak alpha_bonferroni`(추천; statsmodels 튜플의 `pvals_corrected`, `alphacSidak`, `alphacBonf`를 stardust의 이름 꼴로) / statsmodels의 이름 그대로. `multipletests`에 `axis`를 둔다(추천, `false_discovery_control`과 같게; statsmodels는 1차원만). `fdr_gbs`: 넣지 않는다(추천, statsmodels가 대조하지 않았다고 적은 방법) / 넣는다. 스튜던트화 범위 분포는 htest 안에 둔다(H14). `dunnett`의 `rng`: 받지 않는다(다변량 t의 cdf가 결정적; 추천, `[scipy와 다름]`) / scipy처럼 받는다.
- **H9**: `method=` 설정 객체: sd의 `PermutationMethod(n_resamples=9999, batch=None, *, rng)`, `MonteCarloMethod(..., rvs=None, *, rng)`, `BootstrapMethod(..., method='BCa', *, rng)`(추천: scipy의 이름, `rng` 필수, `random_state` 없음) / 문자열 `method='permutation'`과 `n_resamples`·`rng` 키워드. `monte_carlo_test`의 `rvs` 규약: `rvs(size, *, rng)`(추천) / scipy처럼 `rvs(size)`(사용자가 생성기를 닫아 넣는다). 내장 통계량의 빠른 길(평균의 차 등을 C++로): H10에서(추천) / H9에서. `multiscale_graphcorr`: 넣는다(H2의 "전부"; 추천) / 뺀다(O(n²) 메모리, `workers`). `goodness_of_fit`: sd.stats 분포 일부(정규, 지수, 로지스틱, Gumbel, Weibull 등 `anderson`과 같은 족)의 최우추정으로 넣는다(추천) / 분포 맞추기(`fit`)를 sd.stats에 먼저 더한다 / 뺀다. `bws_test`의 기본(scipy는 시드 없는 `PermutationMethod()`): 가능하면 모든 순열을 나열하고, 그보다 크면 `rng`를 꼭 받는다(추천). 재표본 함수의 `axis` 기본값: None(표본마다 펼치고 `statistic`은 `axis=-1`로 부른다; H9에 맞춘 추천) / scipy의 0. `bootstrap`의 `bootstrap_result`(앞 결과에 잇기): 두지 않는다(추천) / 둔다. R 꾸러미(`coin` 등)는 이 마일스톤에서 묻는다(H15).
- **H10**: 범위는 O97처럼 모든 공개 함수를 크기마다 견주는 것(추천). 코드 리뷰는 사용자가 따로 하는지(O96 2부처럼) 그때 묻는다.
