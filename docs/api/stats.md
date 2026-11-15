# sd.stats API

> 상태: **T4 구현** (2026-11-15): 연속 분포 12개(§2.1의 앞 12줄). 나머지는 초안이다. 결정과 내부 구조는 [docs/design/stats.md](../design/stats.md) (T1–T23).

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
- `sample`의 `rng`는 `sd.random.default_rng(...)`의 Generator다. 생략할 수 있는지는 T7 전에 정한다.

### §2.1 분포 목록 (T4의 12개는 확정, 나머지는 초안)
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
| 로지스틱 | `Logistic(mu, sigma)` | `logistic` |
| Gumbel | `Gumbel(mu, sigma)` | `gumbel_r` |
| 일반화 극값 | `GEV(c, mu, sigma)` | `genextreme` |
| 파레토 | `Pareto(b, scale)` | `pareto` |
| 레일리 | `Rayleigh(sigma)` | `rayleigh` |
| 역감마 | `InverseGamma(a, scale)` | `invgamma` |
| 비중심 카이제곱 | `NoncentralChiSquared(df, nc)` | `ncx2` |
| 비중심 t | `NoncentralT(df, nc)` | `nct` |
| 비중심 F | `NoncentralF(dfn, dfd, nc)` | `ncf` |
| 이항 | `Binomial(n, p)` | `binom` |
| 포아송 | `Poisson(mu)` | `poisson` |
| 기하 | `Geometric(p)` | `geom` |
| 음이항 | `NegativeBinomial(n, p)` | `nbinom` |
| 초기하 | `Hypergeometric(M, n, N)` | `hypergeom` |
| 베르누이 | `Bernoulli(p)` | `bernoulli` |
| 이산 균등 | `DiscreteUniform(low, high)` | `randint` |
| 베타이항 | `BetaBinomial(n, a, b)` | `betabinom` |
| 로그 급수 | `LogSeries(p)` | `logser` |
| Zipf | `Zipf(a)` | `zipf` |

### §2.2 T4의 보장 영역과 잰 정확도
- **보장 영역** (사용자 결정 2026-11-15, T24): 닫힌 꼴 7개(`Normal Exponential Uniform Laplace Cauchy LogNormal Weibull`)는 매개변수와 입력의 정의역 전체다. special 기반 5개는 special이 보장하는 범위다: `Gamma` a ≤ 1e6, `ChiSquared` df ≤ 2e6, `Beta` a, b ≤ 1e5, `StudentT` df ≤ 1e6, `F` dfn, dfd ≤ 1e5. 그 밖도 값을 계산하지만 보장하지 않는다.
- **느슨한 영역** (잰 최대): `Gamma` a > 1e6의 `icdf` 1.1e-14, `ilogcdf`·`ilogccdf` 2.5e-14. `StudentT`의 `icdf`·`iccdf`(`ilog…`도)에서 df ≥ 1e5이고 확률이 2^-1000(약 9.3e-302) 아래인 구석은 1.4e-14다. 나머지 범위 밖 점은 모두 1e-14 안이었다.
- **잰 정확도** (bench/stats/t4/accuracy.md, mpmath 대비, 계약 단위): 보장 영역 안 최대 9.2e-15. 같은 점에서 scipy(고전형)는 꼬리에서 상대오차 1에 이르거나 inf를 낸다(예: `Beta.pdf`, `StudentT.cdf`의 깊은 꼬리, `LogNormal.pdf`).
- **속도** (bench/stats/t4/speed.md, 원소 10⁵개): 1스레드는 scipy의 0.3–4.6배(중앙값 0.94), 10스레드는 1.7–39배(중앙값 5.4). Python float 하나는 scipy보다 38–81배 빠르다(호출당 0.16–0.36 µs).
- scipy와 다르게 동작하는 곳은 docs/design/stats.md §2.4에 있다.

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
