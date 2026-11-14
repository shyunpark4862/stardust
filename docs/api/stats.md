# sd.stats API (초안)

> 상태: **초안** (2026-11-15). 결정과 내부 구조는 [docs/design/stats.md](../design/stats.md). 분포의 매개변수 이름은 T4 전에 확정한다. 아직 구현되지 않았다.

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
- 매개변수는 분포를 만들 때 키워드로 준다: `sd.stats.Normal(mu=0.0, sigma=2.0)`. 매개변수와 메서드의 입력은 서로 브로드캐스트한다.
- `ccdf`는 꼬리에서도 상대 정확도다(1 − cdf가 아니다).
- `sample`의 `rng`는 `sd.random.default_rng(...)`의 Generator다. 생략할 수 있는지는 T7 전에 정한다.

### §2.1 분포 목록 (매개변수 이름은 초안)
| 분포 | 클래스 (안) | scipy |
|---|---|---|
| 정규 | `Normal(mu, sigma)` | `norm`, `Normal` |
| t | `StudentT(df)` | `t` |
| 카이제곱 | `ChiSquared(df)` | `chi2` |
| F | `F(dfn, dfd)` | `f` |
| 감마 | `Gamma(a, scale)` | `gamma` |
| 베타 | `Beta(a, b)` | `beta` |
| 지수 | `Exponential(scale)` | `expon` |
| 로그정규 | `LogNormal(mu, sigma)` | `lognorm` |
| 균등 | `Uniform(a, b)` | `uniform`, `Uniform` |
| 코시 | `Cauchy(mu, sigma)` | `cauchy` |
| 라플라스 | `Laplace(mu, sigma)` | `laplace` |
| 베이불 | `Weibull(c, scale)` | `weibull_min` |
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
