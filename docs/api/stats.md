# sd.stats API

> 상태: **T7 구현** (2026-11-16): 연속 분포 12개(T4), 이산 분포 10개(T5), 극값·수명 6개와 비중심 3개(T6), 모든 분포의 `sample`(T7). §2.1의 나머지는 초안이다. 결정과 내부 구조는 [docs/design/stats.md](../design/stats.md) (T1–T23).

## §1. 공통 규칙
- 입력은 위치로만(`/` 앞), 옵션은 키워드로만(`*` 뒤). 별칭은 없다.
- 피연산자는 `sd.ndarray`와 Python 스칼라. float64로 계산하고(bool·int64는 바꿔서), complex는 `TypeError`다. 모두 Python 스칼라면 결과도 Python `float`다.
- 분포의 매개변수가 정의역 밖이면 그 원소의 결과는 NaN이다. 에러는 없다.
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 표본도 같은 `rng` 상태에서는 같다(docs/api/random.md).
- 정확도: 보장 영역에서 참값 대비 상대 1e-14(special과 같은 계약). 분포마다 보장 영역과 잰 값은 해당 절에 적는다.
- 검정과 상관은 `sd.stats.tests`에 있다(§6).

## §2. 단변량 분포
```
X.pdf(x, /)          X.logpdf(x, /)        # 이산 분포는 pmf, logpmf
X.cdf(x, /)          X.logcdf(x, /)
X.ccdf(x, /)         X.logccdf(x, /)
X.icdf(p, /)         X.iccdf(p, /)
X.ilogcdf(logp, /)   X.ilogccdf(logp, /)
X.support()
X.mean()  X.variance()  X.standard_deviation()  X.skewness()  X.kurtosis()
X.median()  X.mode()  X.entropy()
X.sample(shape=(), /, *, rng)
```
- **매개변수**는 분포를 만들 때 키워드로만 준다(T19): `sd.stats.Normal(mu=0.0, sigma=2.0)`. 위치 인자는 `TypeError`다. 기본값은 위치 0·척도 1에만 있고(`mu=0`, `sigma=1`, `scale=1`), 모양 매개변수와 `Uniform(*, a, b)`는 꼭 줘야 한다.
- 매개변수는 Python 실수(bool·int 포함)나 complex가 아닌 `sd.ndarray`다. 그 밖의 형은 `TypeError`, 서로 브로드캐스트할 수 없는 모양은 만들 때 `ValueError`다. 매개변수는 읽기 전용 속성이다(`X.mu`는 준 값 그대로). `repr`은 `Normal(mu=0.0, sigma=2.0)` 꼴이다.
- 이름 규칙(T18): 위치가 있는 분포는 `mu`·`sigma`(코시·라플라스의 `sigma`는 표준편차가 아니라 척도), 로그정규는 밑 정규의 `mu`·`sigma`. [0, ∞)의 분포는 모양 + `scale`(rate가 아니다). 모양 매개변수는 scipy 이름(`a b c df dfn dfd`).
- **입력을 받는 메서드**(`pdf`부터 `ilogccdf`까지)는 원소별이고, 매개변수와 입력이 서로 브로드캐스트한다. special처럼 `out=`을 받는다(T22). **매개변수만의 메서드**(모멘트, `median`, `mode`, `entropy`, `support`)는 매개변수의 브로드캐스트 모양을 돌려주고 `out=`이 없다. 모든 피연산자가 Python 스칼라면 결과는 Python `float`다. `support()`는 `(lo, hi)` 튜플이다.
- 매개변수가 정의역 밖이거나 NaN이면 그 원소의 모든 메서드가 NaN이다(T8). 입력이 NaN이면 NaN이다.
- 지지집합 밖: `pdf`는 0(`logpdf`는 −inf), `cdf`는 0 또는 1. `icdf`·`iccdf`는 [0, 1] 밖이면 NaN이고 0과 1에서 지지집합의 끝이다. `ilogcdf`·`ilogccdf`는 양수면 NaN이고 0과 −inf에서 끝이다.
- `ccdf`, `logccdf`는 위꼬리에서도 상대 정확도다(1 − cdf가 아니다). `logcdf`, `logccdf`는 꼬리가 언더플로하는 곳에서도 값이 있다(예: `Normal().logcdf(-1e5)`).
- `kurtosis()`는 **초과 첨도**다(정규 = 0, T23).
- `mode()`는 밀도가 가장 큰 점이다. 그 점이 지지집합의 끝이어도, 거기서 밀도가 발산해도 그 점이다(감마 a < 1 → 0). 하나로 정해지지 않으면 NaN이다(균등, 베타 a, b < 1 또는 a = b = 1; T21).
- 모멘트가 정의되지 않으면 NaN, 무한이면 inf다. 평균이 +∞로 발산하면(X ≥ 0인 F의 dfd ≤ 2) inf다. 분산 이상은 분산이 유한하고 그 모멘트가 발산하면 inf, 분산이 유한하지 않으면 NaN이다(t의 df ≤ 1 평균, 코시의 모든 모멘트는 NaN).
- `entropy()`는 미분 엔트로피이고 단위는 nat다.
- 정확도(T20): 보장 영역에서 상대 1e-14. 단 `logpdf`와 `entropy`는 |오차| ≤ 1e-14·max(1, |값|)이다(값이 0을 지난다). 역함수 넷은 special의 역함수처럼 |Δx/x| ≤ 1e-14·max(1, κ)이고, κ는 그 메서드의 입력에 대한 상대 조건수다(`icdf(p)`는 |p / (x f(x))|, `ilogcdf(logp)`는 |logp·p / (x f(x))|).
- **이산 분포**(T25): `pdf`·`logpdf` 대신 `pmf`·`logpmf`이고 나머지 메서드는 같다. 정수가 아닌 k에서 `pmf`는 0, `cdf`·`ccdf`와 그 log는 floor(k)에서의 값이다. `icdf(p)`는 cdf(k) ≥ p인 가장 작은 k, `iccdf(p)`는 ccdf(k) ≤ p인 가장 작은 k이다(scipy의 ppf, isf와 같다). 결과는 float64 정수이고, double 범위를 넘으면 inf다. 정확도는 후방 오차다. 목표를 상대 1e-14 안에서 움직였을 때의 정확한 분위수 중 하나를 낸다(`icdf`·`iccdf`는 p와 1 − p 중 작은 쪽, `ilogcdf`·`ilogccdf`는 입력 t에 대해). 목표가 계단 바로 옆이면 이웃 정수도 맞고, 분위수가 아주 커서 계단이 cdf의 분해능보다 촘촘하면(Zipf a ≈ 1) 그 폭만큼 허용된다. `logpmf`는 상대 1e-14, `entropy`는 연속과 같은 혼합 기준이다. `median()`은 `icdf(1/2)`, `mode()`는 최빈값이 둘 이상이면 NaN이다(T21). 매개변수 하나가 점질량을 만들면(이항 p = 0·1, 포아송 μ = 0 등) 분산은 0, 왜도·첨도는 NaN이다.
- 정수여야 하는 매개변수(이항·베타이항의 n, 초기하의 셋, 이산 균등의 두 끝)는 정수가 아니면 NaN이다. 음이항의 n은 양의 실수다(special의 `bdtr`, `nbdtr`과 같다).
- `sample`의 `rng`는 `sd.random.default_rng(...)`의 Generator이고 꼭 줘야 한다(T27; §2.5).

### §2.1 분포 목록 (T4–T6의 31개는 확정, 나머지는 초안)
| 분포 | 클래스 | scipy |
|---|---|---|
| 정규 | `Normal(*, mu=0, sigma=1)` | `norm`, `Normal` |
| t | `StudentT(*, df)` | `t` |
| 카이제곱 | `ChiSquared(*, df)` | `chi2` |
| F | `F(*, dfn, dfd)` | `f` |
| 감마 | `Gamma(*, a, scale=1)` | `gamma` |
| 베타 | `Beta(*, a, b)` | `beta` |
| 지수 | `Exponential(*, scale=1)` | `expon` |
| 로그정규 | `LogNormal(*, mu=0, sigma=1)` | `lognorm` |
| 균등 | `Uniform(a, b)` | `uniform`, `Uniform` |
| 코시 | `Cauchy(*, mu=0, sigma=1)` | `cauchy` |
| 라플라스 | `Laplace(*, mu=0, sigma=1)` | `laplace` |
| 베이불 | `Weibull(*, c, scale=1)` | `weibull_min` |
| 로지스틱 | `Logistic(*, mu=0, sigma=1)` | `logistic` |
| Gumbel | `Gumbel(*, mu=0, sigma=1)` (최댓값형) | `gumbel_r` |
| 일반화 극값 | `GEV(*, c, mu=0, sigma=1)` (c = −ξ, scipy와 같다) | `genextreme` |
| 파레토 | `Pareto(*, b, scale=1)` (x ≥ scale) | `pareto` |
| 레일리 | `Rayleigh(*, scale=1)` | `rayleigh` |
| 역감마 | `InverseGamma(*, a, scale=1)` | `invgamma` |
| 비중심 카이제곱 | `NoncentralChiSquared(*, df, nc)` (nc ≥ 0) | `ncx2` |
| 비중심 t | `NoncentralT(*, df, nc)` | `nct` |
| 비중심 F | `NoncentralF(*, dfn, dfd, nc)` (nc ≥ 0) | `ncf` |
| 이항 | `Binomial(*, n, p)` | `binom` |
| 포아송 | `Poisson(*, mu)` | `poisson` |
| 기하 | `Geometric(*, p)`: 시행 횟수, k ≥ 1 | `geom` |
| 음이항 | `NegativeBinomial(*, n, p)`: 실패 횟수, k ≥ 0 | `nbinom` |
| 초기하 | `Hypergeometric(*, total, successes, draws)` | `hypergeom(M, n, N)` |
| 베르누이 | `Bernoulli(*, p)` | `bernoulli` |
| 이산 균등 | `DiscreteUniform(*, low, high)`: [low, high) | `randint` |
| 베타이항 | `BetaBinomial(*, n, a, b)` | `betabinom` |
| 로그 급수 | `LogSeries(*, p)` | `logser` |
| Zipf | `Zipf(*, a)` | `zipf` |

### §2.2 T4의 보장 영역과 잰 정확도
- **보장 영역** (사용자 결정 2026-11-15, T24): 닫힌 꼴 7개(`Normal Exponential Uniform Laplace Cauchy LogNormal Weibull`)는 매개변수와 입력의 정의역 전체다. special 기반 5개는 special이 보장하는 범위다: `Gamma` a ≤ 1e6, `ChiSquared` df ≤ 2e6, `Beta` a, b ≤ 1e5, `StudentT` df ≤ 1e6, `F` dfn, dfd ≤ 1e5. 그 밖도 값을 계산하지만 보장하지 않는다.
- **느슨한 영역** (잰 최대): `StudentT`의 `icdf`·`iccdf`에서 df ≥ 1e5이고 확률이 2^-1000(약 9.3e-302) 아래인 구석은 1.4e-14다(`ilog…`는 9.8e-15). 나머지 범위 밖 점은 모두 1e-14 안이었다. `Gamma` a > 1e6의 역함수는 전에 2.5e-14였는데, special의 `gammaincinv`를 고친 뒤 1.7e-16이 됐다(2026-11-16; 보장 영역은 그대로다).
- **잰 정확도** (bench/stats/t4/accuracy.md, Arb 참값 대비, 계약 단위): 보장 영역 안 최대 9.2e-15(87만 점). 같은 점에서 scipy(고전형)는 꼬리에서 상대오차 1에 이르거나 inf를 낸다(예: `Beta.pdf`, `StudentT.cdf`의 깊은 꼬리, `LogNormal.pdf`).
- **속도** (bench/stats/t4/speed.md, 원소 10⁵개): 1스레드는 scipy의 0.3–4.6배(중앙값 0.94), 10스레드는 1.7–39배(중앙값 5.4). Python float 하나는 scipy보다 38–81배 빠르다(호출당 0.16–0.36 µs).
- scipy와 다르게 동작하는 곳은 docs/design/stats.md §2.4에 있다.

### §2.3 T5의 보장 영역과 잰 정확도
- **보장 영역** (사용자 결정 2026-11-16): 잰 매개변수 범위다. 이항 n ≤ 1e6, 포아송 μ ≤ 1e6, 기하 p ≥ 1e-9, 음이항 n ∈ [0.1, 1e4]·p ≥ 1e-4, 초기하 total ≤ 3000, 베타이항 n ≤ 2000·a, b ∈ [0.1, 100], 로그 급수 p ≥ 1e-8, Zipf a ≤ 60. `Bernoulli`와 `DiscreteUniform`은 닫힌 꼴이라 정의역 전체다. 그 밖도 값을 계산하지만 보장하지 않는다.
- **잰 정확도** (bench/stats/t5/accuracy.md, Arb 참값 대비, 계약 단위): 61만 점 전부 계약 안이다. 값 메서드의 최대는 6.5e-15이고, 분위수는 모든 점이 허용 구간 안이다. 같은 점에서 scipy는 꼬리에서 상대오차 1이나 inf를 낸다(이항·음이항·초기하의 cdf, 초기하·로그 급수의 logpmf). 분위수에서는 scipy의 답이 허용 구간 안에 드는 비율이 64–100%다(기하 `isf` 64%, 포아송 `isf` 64%, 이항 `isf` 88%).
- **속도** (bench/stats/t5/speed.md, 원소 10⁵개): 1스레드는 scipy의 0.22–232배(중앙값 1.5), 10스레드는 2.5–1432배(중앙값 7.7)다. Python float 하나는 scipy보다 11–114배 빠르다(호출당 0.17–2.1 µs). 1스레드에서 scipy보다 느린 것은 이항(0.2–0.4배)과 음이항·베타이항의 pmf(0.3배)다. 깊은 꼬리까지 상대 정확도를 지키는 double-double 접두의 비용이다.

### §2.4 T6의 보장 영역과 잰 정확도
- **보장 영역** (사용자 결정 2026-11-16, T26):
  - 닫힌 꼴 5개(`Logistic Gumbel GEV Pareto Rayleigh`)는 매개변수와 입력의 정의역 전체다. `InverseGamma`는 감마처럼 a ≤ 1e6이다.
  - 비중심 셋은 special의 비중심 함수가 잰 범위다(docs/api/special.md §14): `NoncentralChiSquared` df ∈ [1e-2, 1e4]·nc ≤ 1e4, `NoncentralT` df ∈ [0.1, 1e5]·|nc| ≤ 100, `NoncentralF` dfn, dfd ∈ [1e-2, 1e4]·nc ≤ 1e4. 확률은 1e-300까지 보장한다(작은 쪽 꼬리가 그 아래인 x와, 그 아래 확률의 역함수는 보장 밖). 모멘트와 `mode`도 1e-14다.
  - 비중심 셋의 `entropy`는 적응 구적이고 보장하지 않는다. 참값은 아직 만들지 않았다(10코어로 약 12시간; 나중에 잰다).
- **잰 정확도** (bench/stats/t6/accuracy.md, Arb 참값 대비, 계약 단위): 60만 점에서 보장 영역 안 최대 5.5e-15이고, 밖도 1.2e-15 안이다. 같은 점에서 scipy는 꼬리에서 틀린다. 예: `NoncentralF.cdf` 상대오차 2.3e95, `NoncentralT.cdf`·`GEV.ccdf`·`Pareto.logcdf` inf, `InverseGamma.icdf` 2.4e-5.
- **속도** (bench/stats/t6/speed.md, 원소 10⁵개): 1스레드는 scipy의 0.09–2.8배(중앙값 0.56), 10스레드는 0.8–23배(중앙값 4.2)다. Python float 하나는 scipy보다 13–98배 빠르다(호출당 0.17–0.98 µs). 1스레드에서 느린 것은 다음과 같다.
  - 비중심 t의 `pdf`(0.09배): 양의 적분이라, 급수가 상쇄하는 쪽 꼬리까지 상대 정확도를 지킨다.
  - 비중심 F의 `pdf`(0.14배): double-double 포아송 혼합이다.
  - `Pareto`의 `pdf`·`cdf`(0.22–0.25배)와 `InverseGamma.pdf`(0.27배): double-double 지수다.

### §2.5 표본 (T7)
```
X.sample(shape=(), /, *, rng)
```
- `rng`: `sd.random.default_rng(seed)`의 Generator이고, 꼭 줘야 한다(T27). 다른 것은 `TypeError`다. 표본은 생성기의 스트림에서 차례로 받으므로, 호출하면 `rng.state["position"]`이 원소 수만큼 나아간다.
- **모양**: `shape`(int 또는 int의 tuple) + 매개변수의 브로드캐스트 모양이다(scipy 신형과 같다). 매개변수가 (3,)이고 `shape=(1000,)`이면 결과는 (1000, 3)이라 분포마다 1000개씩이다. 둘 다 비면 Python float이다.
- **값**: float64다. 이산 분포도 정수인 float64다(icdf와 같다; Zipf가 double을 넘으면 inf). 매개변수가 정의역 밖이거나 NaN인 원소는 NaN이다.
- **재현**: 같은 시드와 같은 호출 순서면 같은 값이다. 스레드 수와 무관하고, 한 호출을 둘로 나누어 불러도 같다. numpy·scipy와 값은 다르다(생성기와 방법이 다르다).
- **방법**: 분포마다 정확한 방법(기각이나 조합, 근사 없음)이다. 정규는 ziggurat, 감마는 Marsaglia–Tsang, 이항은 BTRS, 포아송은 PTRS 등이고 표는 docs/design/stats.md §4에 있다.
- **분포 검사** (bench/stats/t7/quality.md): 62개 경우에서 표본 10⁶개의 KS·카이제곱 p값이 모두 0.03 이상이다(가장 작은 값 0.034). 아주 작은 모양도 포함한다(감마 a = 1e-3은 질량의 47.5%가 5e-324 아래라 그 표본은 0이다).
- **속도** (bench/stats/t7/speed.md, 표본 10⁶개, numpy Generator(PCG64) 대비): 1스레드 0.42–2.7배(중앙값 0.84), 10스레드 3.6–23배(중앙값 7.4). 1스레드에서 느린 것은 다음과 같다.
  - 큰 n의 이항(0.42배)
  - 비중심 F(0.48배)
  - 모양이 둘 다 1보다 작은 베타(0.51배)
  - 정규(0.55배): Philox는 단어당 1.1 ns이고, PCG64가 더 싸다.

  Python float 하나는 호출당 0.24–0.5 µs다(numpy의 0.4–2배).

## §3. 다변량 분포 (초안)
```
MultivariateNormal(mu, cov)        MultivariateT(mu, shape, df)
Multinomial(n, p)                  Dirichlet(alpha)
Wishart(df, scale)                 InverseWishart(df, scale)
MatrixNormal(mean, rowcov, colcov)
```
- 메서드: `pdf logpdf`(다항은 `pmf logpmf`), `sample`, `mean`, `covariance`, `entropy`.
- `MultivariateNormal`과 `MultivariateT`의 `cdf`는 1e-14 계약 밖이다. Genz 방법을 고정 시드로 결정적으로 계산하고, 값과 오차 추정을 함께 돌려준다(docs/design/stats.md §6.2).

## §4. 기술통계 (초안)
```
describe(x, /, *, axis, correction, bias)
skew(x, /, *, axis, bias)
kurtosis(x, /, *, axis, fisher, bias)
moment(x, /, *, order, axis)
zscore(x, /, *, axis, correction)
rankdata(x, /, *, method, axis)
sem(x, /, *, axis, correction)
iqr(x, /, *, axis)
mode(x, /, *, axis)
entropy(pk, qk=None, /, *, base, axis)
differential_entropy(x, /, *, axis, method)
```
- 축의 기본값, NaN 처리, 편향 보정의 기본값은 T8 전에 정한다. 자유도 인자는 배열의 `var`처럼 `correction`이다(scipy의 `ddof`).

## §5. 배열과 special에 더하는 것
- 배열(T0, 완료): `cov`, `corrcoef`, `quantile`, `median`, NaN 리덕션. docs/api/array.md §7.9–§7.11.
- special(T1, T2): `stdtr stdtrit chdtr chdtrc chdtri fdtr fdtrc fdtri pdtr pdtrc bdtr bdtrc nbdtr nbdtrc`, `chndtr chndtrix nctdtr nctdtrit ncfdtr ncfdtri ndtri_exp`. docs/api/special.md에 더한다.

## §6. `sd.stats.tests` (나중)
- 검정, 상관(Pearson·Spearman·Kendall의 계수와 p값). 목록과 결과 객체는 분포·기술통계를 구현한 뒤에 정한다.
- scipy 이름 → 위치의 대응표를 그때 이 절에 둔다(예: `scipy.stats.ttest_ind` → `sd.stats.tests.ttest_ind`, `scipy.stats.pearsonr` → `sd.stats.tests.pearsonr`).
