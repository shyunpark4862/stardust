# 가설검정 영역 설계 (sd.htest)

> 상태: **설계 초안** (2026-11-20). 큰 결정(§1.1 H1–H5)은 사용자가 정했다. 마일스톤(§8 H0–H10)은 모두 시작 전이고, 나머지는 안이다. 시작 전에 정할 것은 §9에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

`sd.htest`(가설검정: 검정통계량, p값, 신뢰구간)의 확정 사항과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처의 안은 H0에서 docs/api/htest.md로 쓴다.
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
| H3 API | scipy의 함수 이름과 인자 이름을 따르고 stardust의 규칙을 쓴다: 입력은 위치로, 옵션은 키워드로만, 별칭 없음, stardust의 메시지, `nan_policy`, `axis`와 배치는 sd.stats 기술통계의 규칙, 무작위는 `rng=`(sd.random·sd.stats와 같다). 결과는 `statistic`, `pvalue`, scipy에 있으면 신뢰구간, 자유도 등을 담은 객체이고, 필드는 읽기 전용이며 복사 규칙은 regression의 G98·G113(와 G120, API.md §2.2)을 따른다. 결과의 세부 꼴은 §2.2의 안이고 §9 H0에서 묻는다 (사용자 결정 2026-11-20) |
| H4 정확도 | 검정통계량과 p값은 다른 영역처럼 Arb 참값에 대한 정확도 계약을 갖는다: 보장 영역에서 상대 1e-14, 느슨한 영역은 잰 값을 문서화한다. scipy·R과의 비트 동일은 목표가 아니다. 정확한 귀무분포(작은 표본의 순위 검정, 정확 분할표 검정, Kolmogorov–Smirnov의 정확한 분포)는 Arb나 정확한 유리수 참값으로 확인한다. p값은 sd.stats·sd.special의 분포를 쓴다. scipy·R·statsmodels와의 비교는 정확도 체계(`bench/accuracy`, 비교 패키지 열)와 속도 체계(`bench/speed`)에서 한다 (§4, §7) (사용자 결정 2026-11-20) |
| H5 statsmodels 두 묶음 | scipy의 검정에 더해 statsmodels.stats의 두 묶음을 넣는다. (1) 다중 비교 보정: `multipletests`(bonferroni, sidak, holm-sidak, holm, simes-hochberg, hommel, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky)와 `fdrcorrection`(scipy `false_discovery_control`과 겹친다; 한 가지 API의 안은 §3.9). (2) 비율과 이항: `proportions_ztest`, `proportions_chisquare`, `proportion_confint`(모든 방법), 두 독립 비율의 차·비·오즈비 검정과 구간(`test_proportions_2indep`, `confint_proportions_2indep`), McNemar(`mcnemar`), Cochran Q(`cochrans_q`). TOST 동등성 검정과 검정력 분석은 지금은 뺀다(나중 후보, §1.2). 비교 대상은 statsmodels와 R(`prop.test`, `binom.test`, `p.adjust`, `mcnemar.test`)이다. 이름은 scipy에 같은 개념이 있으면 scipy의 이름, statsmodels에만 있으면 statsmodels의 이름에 stardust의 규칙을 쓴다(이름의 질문은 §9) (사용자 결정 2026-11-20) |

### 1.2 범위 (안)
목록은 scipy 1.18.1 `scipy.stats` 모듈 문서의 "Hypothesis Tests and related functions" 절과 공개 이름 전체(`dir(scipy.stats)`)를 대조해 만들었다. "출처"의 sm은 statsmodels 0.15.0이다. 마일스톤은 §8이다. "재표본(H9)"은 그 검정의 `method=`에 재표본 객체를 받는 것을 H9에서 더한다는 뜻이다.

**모수: 평균과 분산 (마일스톤 H1)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `ttest_1samp(a, popmean, /, *, ...)` | scipy | t 분포. `statistic pvalue df`, `confidence_interval()` |
| `ttest_rel(a, b, /, *, ...)` | scipy | 같다 |
| `ttest_ind(a, b, /, *, equal_var, trim, method, ...)` | scipy | Student·Welch, `trim`이면 Yuen. 재표본(H9) |
| `ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, /, *, ...)` | scipy | 같다(요약값에서) |
| `f_oneway(*samples, equal_var, ...)` | scipy | F 분포, `equal_var=False`면 Welch 분산분석 |
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
| `pearsonr(x, y, /, *, alternative, method, ...)` | scipy | 베타 분포(정확한 정규 이론), `confidence_interval()`(Fisher z). 재표본(H9) |
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
| `binomtest(k, n, /, *, p, alternative)` | scipy | 정확(양쪽은 1 + 10⁻⁷ 규칙), `proportion_ci()`(exact, wilson, wilsoncc) |
| `quantile_test(x, /, *, q, p, alternative)` | scipy | 이항, `confidence_interval()` |
| `poisson_means_test(k1, n1, k2, n2, /, *, diff, alternative)` | scipy | E-test(이중 합) |
| `proportions_ztest(count, nobs, /, *, value, alternative, prop_var)` | sm | 정규 |
| `proportions_chisquare(count, nobs, /, *, value)` | sm | 카이제곱(`table`, `expected_freq`) |
| `proportion_confint(count, nobs, /, *, ..., method, alternative)` | sm | normal, agresti_coull, beta(Clopper–Pearson), wilson, jeffreys, binom_test (방법 이름은 §9 H7) |
| `test_proportions_2indep(count1, nobs1, count2, nobs2, /, *, value, method, compare, alternative, correction)` | sm | 차(wald, agresti-caffo, score), 비(log, log-adjusted, score), 오즈비(logit, logit-adjusted, logit-smoothed, score) |
| `confint_proportions_2indep(count1, nobs1, count2, nobs2, /, *, method, compare, ..., correction)` | sm | 차(wald, agresti-caffo, newcomb, score), 비(log, log-adjusted, score), 오즈비(logit, logit-adjusted, score) |

**다중 비교와 보정 (마일스톤 H8)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `tukey_hsd(*samples, equal_var)` | scipy | 스튜던트화 범위 분포, `confidence_interval()` |
| `dunnett(*samples, control, alternative, rng)` | scipy | 다변량 t의 cdf(`sd.stats.MultivariateT`), `confidence_interval()` |
| `multipletests(pvals, /, *, alpha, method, max_iter, ...)` | sm | 열 방법(§3.9) |
| `false_discovery_control(ps, /, *, axis, method)` | scipy | bh, by |
| `combine_pvalues(pvalues, /, *, method, weights, ...)` | scipy | fisher, pearson, tippett, stouffer, mudholkar_george |

**재표본과 몬테카를로 (마일스톤 H9)**
| 함수 | 출처 | p값과 결과(안) |
|---|---|---|
| `permutation_test(data, statistic, /, *, permutation_type, vectorized, n_resamples, batch, alternative, axis, rng)` | scipy | 전부 나열(가능하면) 또는 무작위 순열, `null_distribution` |
| `monte_carlo_test(data, rvs, statistic, /, *, ...)` | scipy | 같다 |
| `bootstrap(data, statistic, /, *, n_resamples, batch, vectorized, paired, axis, confidence_level, alternative, method, rng)` | scipy | percentile, basic, BCa. `confidence_interval`, `bootstrap_distribution`, `standard_error` |
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
- **모든 호출은 정해진 시간 안에 끝난다**(stats.md §1.3). 정확한 분포의 동적 계획, 무한 급수, 장애 모수의 최대화, 근 찾기에는 근거 있는 상한을 둔다. 상한에 닿으면 조용히 틀린 값이 아니라 NaN이다(원소별 값의 규칙, API.md §2.2). 정확한 방법을 사용자가 고른 경우(`method='exact'`)에 작업량이 상한을 넘으면 `ValueError`로 알린다(§9 H0).
- 경고를 내지 않는다. scipy가 경고(`ConstantInputWarning`, `SmallSampleWarning`, 근사가 부정확하다는 경고)를 내는 곳은 NaN이나 문서다.
- 참값은 Arb(python-flint)와 정확한 유리수(Python의 int·`fractions`)만 쓴다. mpmath는 쓰지 않는다.
- 영역 간 일관성이 먼저다: 자유도 인자 이름(`correction`이 아니라 scipy·statsmodels의 `ddof`가 나오는 곳은 §9 H0), `nan_policy`, `axis`, 메시지의 꼴(`ttest_ind(): …`), `rng`, 결과의 읽기 전용 규칙은 기존 영역을 따른다.
- 반올림이 아닌 동작(기본값, `'auto'`가 고르는 방법, 양쪽 p값의 정의, 연속성 보정, 동점 보정)은 scipy·statsmodels와 같다(API.md §4의 규칙 2). 다르게 정한 곳은 `[scipy와 다름]`으로 이유와 함께 적는다.

---

## 2. 입력과 결과

### 2.1 입력
- **데이터**(표본, 표, 개수): API.md §5의 데이터 입력이다. `sd.ndarray`, 실수의 시퀀스, numpy 배열, frame의 `Series`(`DataFrame`은 여러 열을 받는 곳만)를 받고 float64 배열로 바꾼다. frame의 결측은 NaN이 되어 `nan_policy`를 따른다. 실수 스칼라는 길이 1의 표본이 아니라 스칼라 인자(`popmean`, `k`, `n`)에만 받는다.
- **개수**(`binomtest`의 k·n, 분할표, 비율의 count·nobs): 음이 아닌 정수 값이어야 한다. scipy처럼 정수가 아닌 값은 `ValueError`, 정수 값의 float는 받는다(§9 H6: 분할표의 소수 기댓값).
- **`axis`와 배치**(H3, sd.stats 기술통계의 규칙 T28·T37): `axis`는 None(전체를 하나의 표본으로), int, int의 tuple·리스트다. 축을 따라 표본을 읽고 나머지 축은 배치다. 두 표본 이상을 받는 검정은 배치 축끼리 브로드캐스트하고 표본 축의 길이는 달라도 된다(scipy의 `_axis_nan_policy` 규칙과 같다). `keepdims`를 둔다. 기본값은 §9 H0의 질문이다(추천: sd.stats처럼 None).
  - 배치는 줄(lane)마다 독립인 검정이다. 줄 하나의 계산은 어디서 돌든 같으므로 배치는 줄을 하나씩 부른 것과 비트까지 같다.
  - 축이 정해진 입력(분할표 (r, c), `page_trend_test`의 (m, n), `cochrans_q`의 (N, k), `chi2_contingency`)은 마지막 축들이 표이고 앞 축이 배치다(regression의 `theilslopes`처럼; §9 H0).
- **`nan_policy`**: `'propagate'`(NaN이 있는 줄은 결과가 NaN), `'omit'`(줄마다 NaN을 빼고 남은 크기로; 짝지은 검정은 쌍을 뺀다), `'raise'`(`ValueError`). 메시지는 T41의 꼴이다. 줄마다 남은 크기가 달라도 배치로 계산한다(scipy는 마스크 배열로 느리다).
- **무한대**: 순위 검정은 ±inf를 순서대로 다룬다(순위가 정해진다). 적률을 쓰는 검정은 sd.stats의 규칙(합이 유한하지 않으면 보정 없이; inf − inf는 NaN)이다.
- **`alternative`**: `'two-sided'`, `'less'`, `'greater'`만이다. statsmodels의 `'larger'`, `'smaller'`는 받지 않는다(H5: scipy에 있는 개념은 scipy의 이름; §9 H7). 메시지는 T41의 꼴이다.
- **무작위**(H3): 무작위가 드는 계산(재표본, 몬테카를로 p값, `dunnett`의 다변량 t를 scipy처럼 QMC로 할 때, r × c Fisher의 몬테카를로)은 `rng=`(sd.random의 `Generator`)를 꼭 받는다. 없으면 O31의 꼴인 `TypeError`(`'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType`)다. scipy처럼 시드 없이 무작위로 도는 기본값은 없다(sd.random의 R3: 전역 상태가 없다).
- **사용자 함수**(`ks_1samp`·`cramervonmises`의 `cdf`, 재표본의 `statistic`, `monte_carlo_test`의 `rvs`): Python 호출이다. 콜백 동안 GIL을 쥐고(optimize §2.1), 그 사이의 계산은 C++이다. 분포 인자는 sd.stats의 분포 객체(`sd.stats.Normal(mu=0, sigma=1)`)나 `cdf(x)`를 돌려주는 호출 가능한 객체를 받는다. scipy의 분포 이름 문자열('norm')과 `args=()`는 받지 않는다(O12와 같다; §9 H2).

### 2.2 결과 객체 (안)
- 검정마다 결과 클래스가 있고 공통 기반(`_TestResult`)을 쓴다. 정의는 모든 영역의 결과 형이 있는 `stardust._results`다(D-25). `sd.htest`가 내보낼지는 §9 H0이다(지금 결과 형을 내보내는 영역은 stats와 regression).
- 필드: `statistic`, `pvalue`와 검정마다 scipy·statsmodels의 필드(`df`, `zstatistic`, `statistic_location`, `statistic_sign`, `expected_freq`, `table`, `null_distribution`, `method`(page_trend_test) 등). 별칭은 없다(§1.3). 이름이 갈리는 곳(scipy `chi2_contingency`의 `dof`와 다른 곳의 `df`)은 §9 H0이다.
- 필드는 읽기 전용 속성이다(G98; 대입하면 `AttributeError`). 배열 필드는 쓰기 금지 뷰이고 복사하지 않는다(G120, O103, API.md §2.2). 리스트인 설정 필드는 읽을 때마다 새 리스트다(G113). 결과를 만드는 공개 생성자는 받은 배열을 한 번 복사한다(O103).
- 스칼라 검정의 필드는 Python `float`(개수는 `int`), 배치는 배치 모양의 sd 배열이다(기술통계와 같다). `df`는 Welch처럼 정수가 아닐 수 있어 언제나 float다(§9 H1).
- **신뢰구간**(안): scipy처럼 메서드 `confidence_interval(*, confidence_level=0.95)`이고 `ConfidenceInterval(low, high)` namedtuple(배치면 배치 모양의 쓰기 금지 배열 둘)을 돌려준다. `binomtest`는 scipy의 `proportion_ci(*, confidence_level=0.95, method='exact')`다. regression의 `conf_int(*, alpha=0.05)`((…, k, 2) 배열)과 이름·인자가 다르다: §9 H0에서 묻는다.
- **튜플 풀기**(안): scipy의 결과가 풀리는 곳(`stat, p = ttest_ind(...)`; `chi2_contingency`는 넷)은 같은 개수로 풀린다. scipy가 풀지 않는 결과(`binomtest`, `cramervonmises`, `somersd`, 재표본 결과 등)는 풀리지 않는다(API.md §4 규칙 2: 사용자 코드가 기댄다). statsmodels가 튜플을 돌려주는 함수(`proportions_ztest`, `proportion_confint`, `multipletests` 등)는 결과 객체로 바꾸되 같은 순서로 풀리게 한다(§9 H7·H8).
- `repr`는 짧은 요약이다. `summary()`는 두지 않는다.
- **`method='auto'`가 고른 방법**: scipy는 `page_trend_test`만 결과에 적는다. 모든 `'auto'` 검정의 결과에 고른 방법을 둘지는 §9 H0이다.

---

## 3. 수치 방법 (안)
scipy가 정확하지 않거나 느린 곳과 그것을 넘는 방법을 묶음마다 적는다. 잰 값은 이 초안을 쓰며 scipy 1.18.1에서 본 것이고(Apple M4 Pro, 1스레드), H0의 기준선에서 다시 잰다.

### 3.1 공통
- **p값의 꼬리**: p값은 언제나 작은 쪽 꼬리를 직접 계산한다(`ccdf`는 1 − cdf가 아니다; sd.stats의 `ccdf`·`logccdf`). 양쪽 p값은 연속 분포에서 2·min(cdf, ccdf)를 1로 자른 값이다(scipy와 같다). 대칭 분포(t, 정규)는 2·ccdf(|t|)다. 2⁻¹⁰²² 아래로 내려가는 p값도 서브노멀까지 값을 낸다(0이 아니다).
- **통계량의 조건수**: p값의 오차는 통계량의 오차가 κ_p = |s·∂p/∂s / p|배로 커진 것과 분포 함수의 오차의 합이다. 깊은 꼬리에서 κ_p는 크다(정규에서 z²쯤). 그래서 계약은 special의 역함수처럼 κ_p를 넣은 꼴이다(§4).
- **통계량을 정확하게**: 평균·분산·제곱합은 sd.stats 기술통계의 커널(평균을 hi + lo로, TwoSum 보상 합; stats.md §5)을 쓴다. 상쇄가 본질인 통계량은 식을 바꿔 쓴다(묶음마다 아래). 순위는 `sd.stats.rankdata`의 커널(정확한 중간 순위: 반정수라 double로 정확하다)이다.
- **정확한 검정의 "적어도 그만큼 극단"**: 비교는 통계량의 정수 꼴로 한다(U, T⁺, 역순 쌍의 수, KS의 격자 간격 k/lcm(n₁, n₂), CvM의 정수 합). 수학적으로 같은 두 값이 반올림으로 갈려 한 칸을 빼거나 넣는 일이 없다. 이산 분포의 양쪽 p값은 scipy·R의 규칙 그대로다: 확률이 관측의 확률 × (1 + 10⁻⁷) 이하인 결과를 모두 더한다(`binomtest`, 2 × 2 `fisher_exact`).
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
- **퇴화한 입력**: 상수 표본, 0인 분산, 너무 작은 표본(scipy가 NaN과 경고를 내는 곳: `skewtest` n < 8, `kurtosistest` n < 5, `shapiro` n < 3)은 NaN이다(경고 없음). scipy의 `ttest_1samp([1, 1, 1], 0)`은 statistic inf, pvalue 0이다. API.md §2.2("정할 수 없는 값은 NaN": 표준오차가 0인 회귀의 t·p값)와 맞추려면 NaN이다(`[scipy와 다름]`; §9 H1).

### 3.2 모수: 평균과 분산 (H1)
- **t 검정**: 평균의 차 x̄ − μ₀(와 x̄₁ − x̄₂)는 hi + lo 평균에서 double-double로 뺀다. 분산은 보상 합의 중심 제곱합이다. Welch 자유도 ν = (v₁ + v₂)²/(v₁²/(n₁ − 1) + v₂²/(n₂ − 1))는 비가 같은 크기의 항이라 상쇄가 없다. p값은 `StudentT(df).ccdf`(|t|)의 두 배, df = inf는 정규(T31). 신뢰구간은 `StudentT.icdf`다. `trim`(Yuen)은 정렬 뒤 윈저화 분산이다. `ttest_rel`은 차의 한 표본 검정이고 차는 정확히 뺀 double이 아니라 double-double로 든다(두 큰 값의 차가 반올림되는 것을 막는다; scipy는 a − b를 double로 만든다).
- **분산분석**: 집단 평균과 전체 평균은 hi + lo, SSB = Σn_i(m_i − m)²와 SSW는 두 번 훑기다. scipy의 `f_oneway`도 평균을 빼고 계산해 평균 10⁸·퍼짐 1의 데이터에서 1.6e-16로 좋다(이 초안에서 잼). 이득은 속도다(배치, §5). Welch 분산분석(`equal_var=False`)은 scipy의 식이다.
- **`bartlett`**: T = [(N − k)log s_p² − Σ(n_i − 1)log s_i²]/C는 분산이 거의 같으면 큰 두 항이 상쇄된다. Σ(n_i − 1)log(s_p²/s_i²)로 바꾸고 log(s_p²/s_i²)를 log1p((s_p² − s_i²)/s_i²)로, 차는 double-double로 계산한다.
- **`levene`, `fligner`**: 중심(중앙값, 평균, 절삭 평균)에서의 절댓값 편차에 분산분석을 한다. 중앙값은 sd.quantile의 정확한 선택이다. Fligner의 정규 점수 Φ⁻¹(1/2 + r/(2(N + 1)))는 `ndtri`이고, 점수의 평균·분산은 보상 합이다.
- **`alexandergovern`**: Hill의 정규화 변환(t에서 z로 가는 다항식)과 카이제곱이다. 변환식은 scipy와 같은 순서로 쓰고 Arb로 확인한다.

### 3.3 적합도 1 (H2)
- **카이제곱과 Cressie–Read**: Pearson의 Σ(O − E)²/E는 항이 양수라 좋다. 로그 우도비 G = 2ΣO log(O/E)는 O ≈ E이면 항이 서로 상쇄한다(ΣO = ΣE라 일차 항이 지워진다). scipy는 그대로 더해 O − E = 10⁻³, E = 10⁶에서 G가 상대 6e-9 틀렸다(이 초안에서 Arb와 잼: 9.99999996e-12, 참값 1.000000002e-11). 항마다 O·log(O/E) − (O − E) = E·[(1 + δ)log1p(δ) − δ](δ = (O − E)/E, 언제나 ≥ 0)로 바꾸고 δ가 작으면 급수로 계산한다. 일반 λ의 2/(λ(λ + 1))ΣO[(O/E)^λ − 1]도 같은 방법이다(expm1(λ·log1p(δ))에서 일차 항을 뺀 꼴). λ = 0, −1(Neyman, 수정 로그 우도비)은 극한식이다. `sum_check`는 scipy의 상대 허용오차 그대로다.
- **D'Agostino의 셋**: `skewtest`·`kurtosistest`의 변환은 세제곱근과 로그에 음수 밑이 나올 수 있다(scipy는 부호를 따로 다룬다). 같은 식을 cbrt로, 꼬리 p값은 정규의 `ccdf`로 계산한다. `normaltest`의 K² = Z₁² + Z₂²는 카이제곱(2)이라 p = e^(−K²/2)이다. `jarque_bera`도 p = e^(−JB/2)다(닫힌 꼴; scipy는 `chi2.sf`).
- **`shapiro`**: Royston의 AS R94(scipy와 R의 swilk)를 옮긴다. 계수 a_i는 기대 순서통계량의 다항 근사이므로 W는 "정확한 Shapiro–Wilk의 W"가 아니라 이 알고리즘이 정의한 값이다. 계약은 그 정의에 대한 정확도다(§4). n이 크면 W가 1에 가까워 log(1 − W)의 1 − W가 자릿수를 잃는다: 1 − W = [SST − (Σa_i x_(i))²]/SST를 double-double로 직접 계산한다. scipy와 R은 W를 double로 만든 뒤 1 − W를 한다(이 초안에서 n = 50에서 scipy와 R의 W가 상대 7e-16, p가 1.8e-14 다름).
- **`anderson`**: A² = −n − Σ(2i − 1)/n·[log F(x_(i)) + log(1 − F(x_(n+1−i)))]의 두 로그를 분포의 `logcdf`·`logccdf`로 계산한다(꼬리의 표본에서 log(1 − F)를 1 − F로 만들지 않는다). 모수 추정: 정규·지수는 닫힌 꼴, 로지스틱·Gumbel·Weibull은 scipy처럼 최우추정(`sd.optimize.root_scalar`로 점수 방정식). p값의 `'interpolate'`는 scipy의 표(Stephens)와 보간식 그대로다(계약은 그 보간에 대한 정확도). 분포 족을 어떻게 받을지는 §9 H2.
- **`ks_1samp`**: 양쪽 정확 분포는 Simard–L'Ecuyer(2011)의 선택(작은 n은 Durbin 행렬(Marsaglia–Tsang–Wang)이나 Pomeranz, 큰 n은 Pelz–Good 점근과 Kolmogorov 분포)이다. scipy의 `kstwo`도 같은 선택이다. Durbin 행렬은 원소가 유리수이고 d가 double(정확한 유리수)이라 작은 n의 참값은 정확한 유리수로 만든다(§7). 행렬의 거듭제곱은 지수를 따로 들어(넘침 방지) double-double로 곱하는 안이다. 한쪽은 Birnbaum–Tingey의 유한 합(양수 항, log 쪽에서 합)과 큰 n의 점근이다. 정확도는 H0의 기준선에서 scipy·R(`ks.test(exact=TRUE)`)과 함께 잰다.
- **`cramervonmises`**: 통계량은 정렬한 F(x_(i))에서 보상 합이다. p값은 Csörgő–Faraway(1996)의 유한 표본 분포(점근 분포 + 1/n 보정)이고, 점근 분포는 Bessel `kv` 급수(scipy와 같은 식)다. 급수의 항 수에 상한과 꼬리 상한을 둔다.

### 3.4 순위 검정 (H3)
- **Mann–Whitney U**: 정확 분포는 §3.1의 확률 점화식(크기 min(n₁, n₂)·n₁n₂/2의 배열, 대칭을 써서 절반)이다. 정규 근사는 동점 분산 보정과 연속성 보정. p값의 꼬리는 `log_ndtr`·`ndtr(−|z|)`(scipy는 이 초안에서 n = 200 완전 분리에서 4.8e-67을 낸다; 이 값의 정확도는 H0에서 잰다).
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
- **`fisher_exact` 2 × 2**: 초기하 분포는 sd.stats의 `Hypergeometric`(전체 120 이하면 정확한 128비트 개수, 그 위는 double-double 비율 점화식; stats.md §3.2)이다. 양쪽 p값은 1 + 10⁻⁷ 규칙(scipy·R과 같다)이고, 그 문턱의 비교는 pmf의 비를 정확한 log 차로 한다. 통계량(표본 오즈비)은 scipy의 정의(무조건 MLE가 아닌 표본 오즈비 ad/bc).
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
- `src/stardust/htest/`: `__init__.py`(공개 이름), `_common.py`(입력 정리: 데이터 변환, `axis`·배치·`keepdims`, `nan_policy`, `alternative`, `rng`, 메시지), `_parametric.py`(H1), `_gof.py`(H2·H4), `_rank.py`(H3), `_correlation.py`(H5), `_contingency.py`(H6), `_proportion.py`(H7), `_multitest.py`(H8), `_resampling.py`(H9). 결과 클래스는 `stardust._results`(D-25)에 둔다.
- `csrc/htest/`: `lanes.hpp`(줄 행렬과 배치·스레드, 기술통계의 `desc_*` 커널과 같은 꼴), `moments.hpp`(t·F·분산 검정의 통계량; sd.stats의 보상 합을 같이 쓴다), `ranks.hpp`(정렬, 중간 순위, 동점 표; `rankdata`의 커널을 같이 쓴다), `exact/`(`mwu.hpp`, `wilcoxon.hpp`, `kendall.hpp`, `ansari.hpp`, `page.hpp`, `ks.hpp`(Durbin·Pomeranz·Smirnov·Kolmogorov·두 표본 격자 경로), `cvm.hpp`, `fexact.hpp`(r × c 망 알고리즘), `nuisance.hpp`(Barnard·Boschloo의 Bernstein 분할)), `divergence.hpp`(Cressie–Read), `swilk.hpp`(AS R94), `studentized_range.hpp`, `multitest.hpp`, `resample.hpp`(순열·부트스트랩 표본의 스트림). 바인딩은 `csrc/bind/htest_bind.cpp`(`stardust._core.htest`).
- 분포 함수는 `csrc/stats/`의 것을 C++에서 바로 부른다(Python을 거치지 않는다). 새 귀무분포(Kolmogorov–Smirnov, 스튜던트화 범위, Fisher의 비중심 초기하)는 처음에 `csrc/htest/` 안에 두고, sd.stats·sd.special에 공개할지는 묻는다(§9 H0).
- `tests/htest/`(마일스톤마다 `test_<묶음>.py`, `test_api_surface.py`가 docs/api/htest.md를 읽는다), 참값 `tests/htest/ref/`, 도구 `tools/htest_ref.py`·`tools/htest_exact.py`·`tools/gen_htest_r.py`(§7).

---

## 7. 참값과 검증 (안)
- **Arb 참값** (`tools/htest_ref.py`, `tools/arbref.py`를 쓴다): 입력 double을 정확한 값으로 Arb에 넣어 통계량을 계산하고(평균, 제곱합, 로그, 순위의 함수), 그 통계량에서 p값을 Arb의 분포 함수(불완전 감마·베타, erfc, Bessel)로 계산한다. 규칙은 stats와 같다(stats.md §9): 공의 반지름이 2⁻¹¹⁰ 아래로 정착한 값만 쓰고, 자른 급수·적분은 꼬리 상한을 공에 더하고, 점마다 시간 제한, 10프로세스, 워커는 500점마다 교체, 입력점은 고정(`--resample`로 새로 뽑는다), 실행 중 RSS를 감시한다(mpmath는 쓰지 않는다).
- **정확한 유리수 참값** (`tools/htest_exact.py`): 조합으로 정의되는 귀무분포는 Python의 int·`fractions`로 정확히 센다: Mann–Whitney·Wilcoxon·Kendall·Ansari·Page의 개수, 두 표본 KS·CvM의 격자 경로 수, 초기하·이항의 꼬리(확률이 double인 이항은 그 double의 정확한 유리수), Durbin 행렬의 거듭제곱(한 표본 KS, d는 정확한 유리수). 크기가 커서 유리수가 감당하지 못하는 곳은 Arb로 넘긴다.
- **장애 모수의 상한**: Barnard·Boschloo는 Arb 구간 산술의 분할로 엄밀한 sup을 구한다(§3.7).
- **정의한 근사의 참값**: AS R94, D'Agostino, Hill의 변환, 보간 표는 같은 정의식을 Arb로 계산한다(상수는 scipy·R 소스의 십진 상수 그대로).
- **비교 대상**(H4): 정확도 체계 `bench/accuracy`에 htest를 더하고 비교 패키지 열을 둔다: scipy 1.18.1, statsmodels 0.15.0(H7·H8), R 4.6.2 base `stats`(`t.test`, `var.test`, `bartlett.test`, `fligner.test`, `ansari.test`, `mood.test`, `wilcox.test`, `kruskal.test`, `friedman.test`, `ks.test`, `shapiro.test`, `chisq.test`, `fisher.test`, `mcnemar.test`, `prop.test`, `binom.test`, `poisson.test`, `cor.test`, `oneway.test`, `ptukey`, `p.adjust`; 다변량 t는 이미 있는 mvtnorm). 이 기계의 R에는 그 밖의 꾸러미(`coin`, `Exact`, `exact2x2`, `kSamples`, `goftest`, `BrunnerMunzel`, `PMCMRplus`)가 없다(설치할지는 §9 H0). `bench/accuracy`와 `tests`는 테스트 체계 세션의 파일이라 넣는 방법은 그 세션과 맞춘다.
- **고정 대조 데이터** (`tools/gen_htest_r.py`): R이 없는 기계에서도 테스트가 돌도록 R의 결과를 `tests/htest/data/`에 고정해 둔다(regression의 `gen_regression_r.py`와 같다).
- **근사의 품질**: 점근·근사 p값의 실제 1종 오류율과 정확 분포와의 차이를 정확 분포(작은 n)와 몬테카를로(큰 n, sd.random)로 재서 표로 둔다(계약이 아니다, §4).
- **골든**: 공개 함수마다 `tools/golden.py`에 고정 입력의 결과를 더한다(자기 비트를 지킨다, API.md §4 규칙 1).

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 |
|---|---|
| H0 준비 | API 문서 docs/api/htest.md(모든 공개 함수의 시그니처, scipy·statsmodels 이름 → sd 이름의 대응표, 마일스톤별 질문), 결과 객체의 틀(`_TestResult`, `ConfidenceInterval`), 공통 입력 층(`axis`·배치·`nan_policy`·`alternative`·`rng`), 참값 도구의 틀(Arb·정확한 유리수), R 대조 도구, scipy·statsmodels·R 기준선(정확도·속도) |
| H1 평균과 분산 | `ttest_1samp ttest_rel ttest_ind ttest_ind_from_stats f_oneway alexandergovern bartlett levene fligner` |
| H2 적합도 1 | `chisquare power_divergence skewtest kurtosistest normaltest jarque_bera shapiro anderson ks_1samp kstest cramervonmises` |
| H3 순위 검정 | `mannwhitneyu ranksums wilcoxon kruskal friedmanchisquare brunnermunzel ansari mood median_test page_trend_test` |
| H4 적합도 2 | `ks_2samp cramervonmises_2samp anderson_ksamp epps_singleton_2samp` |
| H5 상관 | `pearsonr spearmanrho spearmanr pointbiserialr kendalltau weightedtau somersd chatterjeexi` |
| H6 분할표 | `chi2_contingency fisher_exact`(2 × 2, r × c) `barnard_exact boschloo_exact mcnemar cochrans_q`(와 §9 H6의 `odds_ratio relative_risk`) |
| H7 비율과 이항 | `binomtest quantile_test poisson_means_test proportions_ztest proportions_chisquare proportion_confint test_proportions_2indep confint_proportions_2indep` |
| H8 다중 비교와 보정 | `tukey_hsd dunnett multipletests false_discovery_control combine_pvalues` |
| H9 재표본 | `permutation_test monte_carlo_test bootstrap`, `PermutationMethod MonteCarloMethod BootstrapMethod`와 앞 검정들의 `method=`, `bws_test`(와 §9 H9의 `multiscale_graphcorr goodness_of_fit`) |
| H10 최적화와 일관성 점검 | 모든 공개 함수의 속도를 크기마다 scipy·statsmodels·R과 견주어 고치기, 메시지·시그니처의 영역 간 일관성, 코드 리뷰 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다. H1–H8은 서로 기대는 것이 적어 순서를 바꿀 수 있다(H6의 `median_test`가 쓰는 `chi2_contingency`는 H3이 먼저면 H3에서 만든다).

---

## 9. 마일스톤을 시작하기 전에 물을 것
추천을 첫째에 둔다.

- **H0 (모든 마일스톤에 걸친 것)**
  1. 결과 객체의 꼴: 검정마다 결과 클래스(공통 기반 `_TestResult`, 읽기 전용, `stardust._results`)(추천) / 결과 클래스 하나(`TestResult`)에 필드를 더하기 / namedtuple(기술통계의 `DescribeResult`처럼).
  2. 결과 클래스의 이름: DESIGN의 `<함수>Result` 규칙(`TtestIndResult`, `PearsonrResult`; 추천) / scipy의 이름(`TtestResult`를 셋이 같이 쓰고 여러 함수가 `SignificanceResult`). `sd.htest`가 결과 형을 내보낼지(추천: 내보내지 않는다; 지금 내보내는 것은 stats·regression뿐이다, D-25).
  3. 튜플 풀기: scipy가 푸는 곳만 같은 개수로(추천, API.md §4 규칙 2) / 모두 (statistic, pvalue)로 / 풀지 않는다.
  4. 신뢰구간: scipy의 `confidence_interval(*, confidence_level=0.95)` → `ConfidenceInterval(low, high)`(추천; regression의 `conf_int(alpha=)`와 다른 이름임을 API.md §6에 적는다) / regression의 `conf_int(*, alpha=0.05)` → (…, 2) 배열로 맞추기.
  5. `axis`의 기본값: None(sd.stats 기술통계와 같다; 추천) / scipy의 0(함수마다 0 또는 None). 표를 받는 함수(분할표, `page_trend_test`, `cochrans_q`)는 마지막 축들이 표이고 앞이 배치(추천).
  6. 필드 이름의 통일: `chi2_contingency`의 `dof`를 다른 곳처럼 `df`로(추천, O4처럼 영역 안의 같은 개념은 같은 이름) / scipy 그대로.
  7. 자유도 인자: `chisquare`·`power_divergence`의 `ddof`(기술통계는 `correction`) — `ddof` 그대로(추천: 뜻이 다르다. 여기서는 기대 빈도를 맞춘 모수 수다) / `correction`으로.
  8. 퇴화한 입력과 작은 표본: NaN, 경고 없음(추천; D-32와 같은 원리) / scipy처럼(일부 `ValueError`).
  9. `'auto'`가 고른 방법을 모든 결과에 `method` 필드로 둘지: 둔다(추천) / scipy처럼 `page_trend_test`만.
  10. `'exact'`의 작업량이 상한을 넘을 때: `ValueError`(추천) / 점근으로 조용히.
  11. 새 귀무분포(Kolmogorov–Smirnov `kstwo`·`kstwobign`, 스튜던트화 범위, Fisher 비중심 초기하)의 공개: 처음에는 htest 안에만(추천) / sd.stats의 분포로(T6의 꼴, 같은 계약) / sd.special의 함수로(`kolmogorov`, `smirnov`, scipy 이름).
  12. R 꾸러미 설치(`coin`, `Exact`, `exact2x2`, `kSamples`, `goftest`, `BrunnerMunzel`, `PMCMRplus`): 필요한 마일스톤에서 하나씩 묻는다(추천) / base R만 / 지금 모두.
- **H1**: `ttest_*`의 `df`를 언제나 float로(추천) / scipy처럼 Student는 int. 표준오차 0의 t·p(scipy inf·0): NaN(추천, API.md §2.2) / scipy처럼. `ttest_ind`의 `method`(재표본)는 H9에서 더한다(추천). `f_oneway`의 `equal_var=False`(Welch 분산분석, scipy 1.18.1에 있다)를 넣는다(추천).
- **H2**: `kstest` 감싸개: 두지 않는다(추천: `ks_1samp`·`ks_2samp`을 고르기만 하는 감싸개이고, 문자열 분포 이름과 `N`으로 표본을 만드는 옛 꼴이 섞여 있다) / scipy처럼 둔다. 분포 인자: sd.stats 분포 객체와 호출 가능한 `cdf`(추천) / scipy의 문자열 이름도. `anderson`의 `dist`: sd.stats의 분포 클래스(`dist=sd.stats.Normal`; 추천. `gumbel_l`은 sd.stats에 없어 빠진다) / scipy의 문자열(별칭 'gumbel', 'extreme1' 없이). `anderson`의 `method` 기본값 `'interpolate'`(추천; scipy 1.18은 주지 않으면 경고하고 옛 꼴을 낸다) / 꼭 주게. Shapiro–Wilk는 AS R94의 정의(추천) / 정확한 기대 순서통계량의 계수(다른 검정이 된다).
- **H3**: 동점이 있을 때 `'exact'`: 조건부 순열 분포(추천, `[scipy와 다름]`) / scipy처럼 동점이 없는 분포 / `ValueError`. `'auto'`의 문턱: scipy 그대로(추천) / 정확이 싼 곳까지 넓히기. `ranksums`(scipy는 legacy, `mannwhitneyu`를 권한다): 둔다(추천, 같은 커널) / 뺀다. `wilcoxon`의 `zstatistic` 필드는 근사일 때만 있다(scipy와 같다; 추천).
- **H4**: `ks_2samp`의 동점: scipy의 정확 분포(동점 무시) 그대로(추천, H4 시작 전에 scipy 소스·R `psmirnov`와 대조하고 다시 묻는다) / R 4.3처럼 조건부 정확. `cramervonmises_2samp`의 `'exact'`를 동적 계획으로 큰 n까지(추천; `'auto'` 문턱은 20 그대로).
- **H5**: `spearmanr`(열들의 상관 행렬 꼴)과 `spearmanrho`(1.18의 새 꼴): 둘 다(추천: scipy에 둘 다 있고 동작이 다르다) / `spearmanrho`만 / `spearmanr`만. Spearman의 정확 분포(R의 AS 89): 넣지 않는다(추천, scipy에 없다) / `method='exact'`로 더한다. `weightedtau`: 넣되 `pvalue`는 NaN(scipy 그대로; 추천) / 결과에 `pvalue` 필드를 두지 않는다 / 뺀다. 결과의 `correlation` 별칭: 두지 않는다(추천).
- **H6**: r × c `fisher_exact`의 기본: 정확한 망 알고리즘(R `fisher.test`; 추천, `[scipy와 다름]`: scipy는 시드 없는 몬테카를로) / `method`를 꼭 주게(`MonteCarloMethod(rng=...)`). `barnard_exact`·`boschloo_exact`의 `n`: 받되 분할의 처음 격자 크기로만 쓴다(결과는 `n`과 상관없이 참 상한; 추천) / scipy처럼 정확도를 정하는 인자로. `odds_ratio`·`relative_risk`(scipy.stats.contingency, 신뢰구간이 있는 추정): 넣는다(추천: H7의 비·오즈비 구간과 한 묶음이고 조건부 오즈비는 Fisher 검정과 같은 분포) / 뺀다. 소수의 개수(가중 표): scipy처럼 받는 곳은 받는다(`chi2_contingency`, 추천) / 정수만. `mcnemar`의 기본 `exact=True`(statsmodels 그대로; 추천) / R `mcnemar.test`처럼 카이제곱.
- **H7** (statsmodels 이름의 질문): 
  1. `proportion_confint`의 신뢰수준: `confidence_level=0.95`(추천, H0-4의 꼴) / statsmodels의 `alpha=0.05`. `confint_proportions_2indep`도 같다.
  2. `proportion_confint`의 방법 이름: scipy에 있는 것은 scipy의 이름(`'exact'`(Clopper–Pearson; statsmodels `'beta'`), `'wilson'`, 더해서 scipy의 `'wilsoncc'`)과 statsmodels에만 있는 것(`'normal'`, `'agresti_coull'`, `'jeffreys'`, `'binom_test'`)(추천) / statsmodels 이름 그대로(`'beta'`). 기본값: statsmodels의 `'normal'`(추천, 규칙 2) / scipy `proportion_ci`의 `'exact'`.
  3. `alternative`: `'less'`·`'greater'`(추천, H5) / statsmodels의 `'smaller'`·`'larger'`.
  4. `proportions_ztest`의 `prop_var=False`(False 또는 실수): None 또는 실수(추천, bool과 수가 섞인 인자를 피한다) / 그대로.
  5. `test_proportions_2indep`의 `return_results`: 두지 않는다(언제나 결과 객체, (statistic, pvalue)로 풀린다; 추천). 결과 필드는 statsmodels의 것(`statistic pvalue compare method diff ratio odds_ratio variance alternative value prop1_null prop2_null`).
  6. `compare`의 값: statsmodels의 `'diff'`, `'ratio'`, `'odds-ratio'` 그대로(추천).
  7. 기본 방법: statsmodels 0.15.0의 것으로 고정(추천; statsmodels가 바뀌어도 따라가지 않는다고 적는다) / 매번 statsmodels를 따른다.
  8. `binom_test`(statsmodels): 두지 않는다(`binomtest`와 같다; 추천).
  9. `proportions_chisquare`의 Yates 보정(R `prop.test`의 `correct=TRUE`): statsmodels처럼 없다(추천, R과는 `correct=FALSE`로 견준다) / `correction=`을 더한다.
- **H8**: 보정의 API(§3.9): `false_discovery_control`(scipy) + `multipletests`(statsmodels, 열 방법), `fdrcorrection` 없음(추천) / `multipletests` 하나로(scipy의 것도 뺀다) / 셋 다. `multipletests`의 기본 방법: statsmodels의 `'holm-sidak'`(추천, 규칙 2) / R `p.adjust`의 `'holm'`. 결과 필드 이름: `reject pvalues_corrected alpha_sidak alpha_bonferroni`(추천; statsmodels 튜플의 `pvals_corrected`, `alphacSidak`, `alphacBonf`를 stardust의 이름 꼴로) / statsmodels의 이름 그대로. `multipletests`에 `axis`를 둔다(추천, `false_discovery_control`과 같게; statsmodels는 1차원만). `fdr_gbs`: 넣지 않는다(추천, statsmodels가 대조하지 않았다고 적은 방법) / 넣는다. 스튜던트화 범위 분포를 sd.stats의 `StudentizedRange`로 공개할지(H0-11). `dunnett`의 `rng`: 받지 않는다(다변량 t의 cdf가 결정적; 추천, `[scipy와 다름]`) / scipy처럼 받는다.
- **H9**: `method=` 설정 객체: sd의 `PermutationMethod(n_resamples=9999, batch=None, *, rng)`, `MonteCarloMethod(..., rvs=None, *, rng)`, `BootstrapMethod(..., method='BCa', *, rng)`(추천: scipy의 이름, `rng` 필수, `random_state` 없음) / 문자열 `method='permutation'`과 `n_resamples`·`rng` 키워드. `monte_carlo_test`의 `rvs` 규약: `rvs(size, *, rng)`(추천) / scipy처럼 `rvs(size)`(사용자가 생성기를 닫아 넣는다). 내장 통계량의 빠른 길(평균의 차 등을 C++로): H10에서(추천) / H9에서. `multiscale_graphcorr`: 넣는다(H2의 "전부"; 추천) / 뺀다(O(n²) 메모리, `workers`). `goodness_of_fit`: sd.stats 분포 일부(정규, 지수, 로지스틱, Gumbel, Weibull 등 `anderson`과 같은 족)의 최우추정으로 넣는다(추천) / 분포 맞추기(`fit`)를 sd.stats에 먼저 더한다 / 뺀다. `bws_test`의 기본(scipy는 시드 없는 `PermutationMethod()`): 가능하면 모든 순열을 나열하고, 그보다 크면 `rng`를 꼭 받는다(추천).
- **H10**: 범위는 O97처럼 모든 공개 함수를 크기마다 견주는 것(추천). 코드 리뷰는 사용자가 따로 하는지(O96 2부처럼) 그때 묻는다.
