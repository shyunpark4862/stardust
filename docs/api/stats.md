# sd.stats API

> 상태: **T10 구현** (2026-11-16): 연속 분포 12개(T4), 이산 분포 10개(T5), 극값·수명 6개와 비중심 3개(T6), 모든 분포의 `sample`(T7), 기술통계(T8), 다변량 4개(T9), 행렬 분포 3개(T10). §6(검정)은 이 영역에서 뺐다(2026-11-18). 결정과 내부 구조는 [docs/design/stats.md](../design/stats.md) (T1–T50).

## §1. 공통 규칙
- 입력은 위치로만(`/` 앞), 옵션은 키워드로만(`*` 뒤). 별칭은 없다.
- 피연산자는 `sd.ndarray`와 Python 스칼라. float64로 계산하고(bool·int64는 바꿔서), complex는 `TypeError`다. 모두 Python 스칼라면 결과도 Python `float`다.
  - int64 범위를 벗어난 Python int 입력은 special처럼 `OverflowError`다: `cdf(): Python int too large to convert to int64`(사용자 결정 2026-11-19, T36). 매개변수는 따로 정한 규칙을 따른다(§2).
- 분포의 매개변수가 정의역 밖이면 그 원소의 결과는 NaN이다. 에러는 없다.
- 메서드의 에러는 메서드 이름으로 시작한다: `cdf(): 'x' must be an sd.ndarray or a real number, not str`, `pdf(): 'x' must be real, not a complex128 array`(Python complex는 `not complex`). 다변량·행렬 분포도 같다(`cdf(): 'x' must have 2 components on its last axis, not 3`, `cdf(): 'max_points' must be an int or None, not float`) (사용자 결정 2026-11-18). Python이 직접 내는 호출 모양의 에러는 Python의 꼴 그대로이고 메서드 이름만 쓴다: `sample() missing 1 required keyword-only argument: 'rng'`, `cdf() got an unexpected keyword argument 'foo'`. 생성자와 기술통계 함수의 `Normal(): ...` 꼴로 바꾸지 않는 것은 일부러 둔 예외다(사용자 결정 2026-11-20, D-29 (b)): 감싸는 층이 호출마다 약 50 ns이고, 메서드 호출 하나가 수백 ns다.
- 생성자의 에러는 클래스 이름으로 시작한다(데이터프레임의 `Series(): `와 같다; 사용자 결정 2026-11-19, T36): `Normal(): 'mu' must be a real number or an sd.ndarray, not str`, `Normal(): shapes could not be broadcast together: (2,) (3,)`, `MultivariateNormal(): 'cov' must be (2, 2) on its last two axes ...`. 틀린 호출 모양도 같은 꼴이다(2026-11-19; 전에는 Python의 `Normal.__init__() got an unexpected keyword argument 'cool'`): `Normal(): unexpected keyword argument 'cool'`, `Gamma(): missing 1 required keyword-only argument: 'a'`, `Normal(): takes 0 positional arguments but 1 was given`. 다변량·행렬 분포도 같다(`Wishart(): unexpected keyword argument 'q'`). 기반 클래스 `ContinuousDistribution`, `DiscreteDistribution`은 `isinstance` 검사용이고 만들 수 없다(`TypeError`).
- 같은 기계, 같은 스레드 수에서는 실행마다 같은 비트다(API.md §3). 스레드 수와 무관한 비트는 약속이 아니지만, 이 영역은 지금 모든 결과가 스레드 수와 무관하다(단변량 메서드는 원소별이고 다변량·기술통계·표본은 분할이 크기에만 달린다). 표본도 같은 `rng` 상태에서는 같다(docs/api/random.md). scipy와 비트가 같다는 약속은 없다: 반올림 수준의 값은 정확도 계약 안이면 되고, 반올림이 아닌 규칙(NaN·±0, 지지집합, 기본값)은 scipy와 같게 둔다(사용자 결정 2026-11-20, D-33).
- 결과 배열: 입력을 받는 메서드(`pdf`부터 `ilogccdf`, 다변량의 `cdf`)의 결과는 새 배열이고 쓸 수 있다(`out=`과 같다). 결과 객체(`CdfResult`, 기술통계의 `DescribeResult`·`ModeResult`)의 배열 필드는 쓰기 금지다(D-4). 매개변수만의 메서드(모멘트, `median`, `mode`, `entropy`, `support`, 다변량의 `mean`·`covariance`, 행렬 분포의 `mean`·`mode`·`variance`)가 내는 배열은 쓰기 금지다(사용자 결정 2026-11-20, D-4; 고치려면 `.copy()`). 쓰면 `ValueError: assignment destination is read-only`다.
- 정확도: 보장 영역에서 참값 대비 상대 1e-14(special과 같은 계약). 분포마다 보장 영역과 잰 값은 해당 절에 적는다.
- 검정과 상관은 이 영역에 없다. 새 최상위 모듈 `sd.htest`에 둔다(§6, docs/design/htest.md).

## §2. 단변량 분포
```
X.pdf(x, /, *, out=None)          X.logpdf(x, /, *, out=None)        # 이산 분포는 pmf, logpmf
X.cdf(x, /, *, out=None)          X.logcdf(x, /, *, out=None)
X.ccdf(x, /, *, out=None)         X.logccdf(x, /, *, out=None)
X.icdf(p, /, *, out=None)         X.iccdf(p, /, *, out=None)
X.ilogcdf(logp, /, *, out=None)   X.ilogccdf(logp, /, *, out=None)
X.support()
X.mean()  X.variance()  X.standard_deviation()  X.skewness()  X.kurtosis()
X.median()  X.mode()  X.entropy()
X.sample(shape=None, /, *, rng)
```
- **매개변수**는 분포를 만들 때 키워드로만 준다(T19): `sd.stats.Normal(mu=0.0, sigma=2.0)`. 위치 인자는 `TypeError`다. 기본값은 위치 0·척도 1에만 있고(`mu=0`, `sigma=1`, `scale=1`), 모양 매개변수와 `Uniform(*, a, b)`는 꼭 줘야 한다.
- 매개변수는 실수 스칼라나 complex가 아닌 `sd.ndarray`다. 실수 스칼라는 Python의 bool·int·float와 numpy의 실수 스칼라다(메서드 입력·special과 같다; API.md §2: numpy 정수도 int). Python 수는 준 값 그대로 두고, numpy의 정수·실수 스칼라는 Python float로, numpy bool은 bool로 바꿔 둔다(`Normal(mu=np.int64(1))`의 repr은 `Normal(mu=1.0, sigma=1.0)`). 그 밖의 형은 `TypeError`, 서로 브로드캐스트할 수 없는 모양은 만들 때 `ValueError`다. 매개변수는 읽기 전용 속성이다. `sd.ndarray` 매개변수는 만들 때 한 번 복사해 쓰기 금지로 들고, 속성도 그 사본을 돌려준다(사용자 결정 2026-11-20, D-5): 만든 뒤 준 배열을 고쳐도 분포는 바뀌지 않고, `X.mu[0] = 1`은 `ValueError`다. 비용은 매개변수 크기의 복사 한 번이다. `repr`은 `Normal(mu=0.0, sigma=2.0)` 꼴이다.
- 이름 규칙(T18): 위치가 있는 분포는 `mu`·`sigma`(코시·라플라스의 `sigma`는 표준편차가 아니라 척도), 로그정규는 밑 정규의 `mu`·`sigma`. [0, ∞)의 분포는 모양 + `scale`(rate가 아니다). 모양 매개변수는 scipy 이름(`a b c df dfn dfd`).
- **입력을 받는 메서드**(`pdf`부터 `ilogccdf`까지)는 원소별이고, 매개변수와 입력이 서로 브로드캐스트한다. special처럼 `out=`을 받는다(T22). **매개변수만의 메서드**(모멘트, `median`, `mode`, `entropy`, `support`)는 매개변수의 브로드캐스트 모양을 돌려주고 `out=`이 없다. 모든 피연산자가 Python 스칼라면 결과는 Python `float`다. `support()`는 `(lo, hi)` 튜플이다.
- 매개변수가 정의역 밖이거나 NaN이면 그 원소의 모든 메서드가 NaN이다(T8). 입력이 NaN이면 NaN이다.
- **무한한 자유도는 극한 분포다**(사용자 결정 2026-11-19, T31). 모든 메서드(pdf부터 entropy, 모멘트, `support`, `sample`)가 그 분포의 값이고, special의 `stdtr(inf, t)`, `fdtr(dfn, inf, x)`, `nctdtr(inf, nc, t)`, `ncfdtr` 같은 극한, `MultivariateT(df=inf)`와 맞는다.
  - `StudentT(df=inf)`는 `Normal()`, `NoncentralT(df=inf, nc)`는 `Normal(mu=nc)`다.
  - `F(dfn, dfd=inf)`는 χ²(dfn)/dfn, `NoncentralF(dfn, dfd=inf, nc)`는 비중심 χ²(dfn, nc)/dfn이다.
  - `F(dfn=inf, dfd)`와 `NoncentralF(dfn=inf, dfd, nc)`는 dfd/χ²(dfd), 곧 `InverseGamma(a=dfd/2, scale=dfd/2)`다.
  - 진짜 분포가 아닌 극한은 NaN 그대로다: `ChiSquared(df=inf)`, `NoncentralChiSquared(df=inf)`, 두 자유도가 모두 inf인 F와 비중심 F(1에 몰린 점질량), 무한한 nc.
- **지지집합**(`support()`)은 매개변수가 퇴화해도 분포 족의 지지집합이다(사용자 결정 2026-11-19, T32; scipy와 같다). `Binomial(n=10, p=0)`은 (0, 10), `Bernoulli(p=0)`은 (0, 1), `Poisson(mu=0)`은 (0, inf), `Geometric(p=1)`은 (1, inf), `NegativeBinomial(n, p=1)`은 (0, inf)다. 연속 분포도 매개변수와 상관없이 족의 지지집합이다(감마는 언제나 (0, inf)).
- 지지집합 밖: `pdf`는 0(`logpdf`는 −inf), `cdf`는 0 또는 1. `icdf`·`iccdf`는 [0, 1] 밖이면 NaN이고 0과 1에서 지지집합의 끝이다(`Poisson(mu=0).icdf(1)`은 inf, scipy의 `ppf(1)`과 같다). `ilogcdf`·`ilogccdf`는 양수면 NaN이고 0과 −inf에서 끝이다.
- `ccdf`, `logccdf`는 위꼬리에서도 상대 정확도다(1 − cdf가 아니다). `logcdf`, `logccdf`는 꼬리가 언더플로하는 곳에서도 값이 있다(예: `Normal().logcdf(-1e5)`).
- `kurtosis()`는 **초과 첨도**다(정규 = 0, T23).
- `mode()`는 밀도가 가장 큰 점이다. 그 점이 지지집합의 끝이어도, 거기서 밀도가 발산해도 그 점이다(감마 a < 1 → 0). 하나로 정해지지 않으면 NaN이다(균등, 베타 a, b < 1 또는 a = b = 1; T21).
- 부호 있는 0: 분산·표준편차와, 지지집합이 [0, ∞) 안인 분포의 평균·`median`·`mode`는 −0.0을 내지 않는다(`Poisson(mu=-0.0).variance()` = 0.0). 대칭인 t의 `iccdf(1/2)`·`ilogccdf(log 1/2)`도 0.0이다(정규와 df = inf의 극한과 같다).
- 모멘트가 정의되지 않으면 NaN, 무한이면 inf다. 평균이 +∞로 발산하면(X ≥ 0인 F의 dfd ≤ 2) inf다. 분산 이상은 분산이 유한하고 그 모멘트가 발산하면 inf, 분산이 유한하지 않으면 NaN이다(t의 df ≤ 1 평균, 코시의 모든 모멘트는 NaN). 예: F의 첨도는 4 < dfd ≤ 8에서 inf, dfd ≤ 4에서 NaN이다. 대칭인 t의 3차 모멘트는 2 < df ≤ 3에서 ∞ − ∞라 정의되지 않으므로 왜도가 NaN이다.
- `entropy()`의 단위는 nat다. 연속 분포는 미분 엔트로피, 이산 분포는 섀넌 엔트로피다.
- 정확도(T20): 보장 영역에서 상대 1e-14. 단 `logpdf`와 `entropy`는 |오차| ≤ 1e-14·max(1, |값|)이다(값이 0을 지난다). 역함수 넷은 special의 역함수처럼 |Δx/x| ≤ 1e-14·max(1, κ)이고, κ는 그 메서드의 입력에 대한 상대 조건수다(`icdf(p)`는 |p / (x f(x))|, `ilogcdf(logp)`는 |logp·p / (x f(x))|).
- **이산 분포**(T25): `pdf`·`logpdf` 대신 `pmf`·`logpmf`이고 나머지 메서드는 같다. 정수가 아닌 k에서 `pmf`는 0, `cdf`·`ccdf`와 그 log는 floor(k)에서의 값이다. `icdf(p)`는 cdf(k) ≥ p인 가장 작은 k, `iccdf(p)`는 ccdf(k) ≤ p인 가장 작은 k이다(scipy의 ppf, isf와 같다). 결과는 float64 정수이고, double 범위를 넘으면 inf다. 정확도는 후방 오차다. 목표를 상대 1e-14 안에서 움직였을 때의 정확한 분위수 중 하나를 낸다(`icdf`·`iccdf`는 p와 1 − p 중 작은 쪽, `ilogcdf`·`ilogccdf`는 입력 t에 대해). 목표가 계단 바로 옆이면 이웃 정수도 맞고, 분위수가 아주 커서 계단이 cdf의 분해능보다 촘촘하면(Zipf a ≈ 1) 그 폭만큼 허용된다. 찾은 k에서 같은 분포의 메서드가 내는 double이 이미 목표에 닿으면(`icdf`는 `cdf(k − 1) ≥ p`, `iccdf`는 `ccdf(k − 1) ≤ p`, log 판도 같다) k − 1을 낸다(사용자 결정 2026-11-20, D-13). 그래서 반올림된 cdf를 다시 넣으면 같은 k가 나온다(`icdf(cdf(k)) == k`). 단 k − 1이 위의 후방 오차 계약 안일 때만이다: p > 1/2인 `icdf`는 작은 쪽 1 − p에 대해 상대 1e-15 안이어야 한다. cdf의 double은 참값과 2⁻⁵³ 남짓(절대) 다를 수 있으므로, 그것이 1e-15·(1 − p)를 넘을 수 있는 곳, 곧 1 − p ≲ 2⁻⁵³/1e-15 ≈ 0.11이면 왕복이 k + 1로 남을 수 있다. 드문 꼬리만의 일이 아니다: `Binomial(n=1000, p=0.3)`에서 cdf(k) < 1인 k 가운데 k = 326(cdf 0.9655, 1 − p = 0.0345)부터 422까지의 50곳이 k + 1이고(cdf가 1로 반올림된 k ≥ 424는 `icdf(1)` = 1000), `Poisson(mu=3).icdf(Poisson(mu=3).cdf(13))`은 14다. `Poisson(mu=0.0025).icdf(1 − 7.8e-16)`은 계약대로 5이고, cdf(4)는 그 p로 반올림된다. `iccdf`의 q > 1/2(ccdf가 1 가까운 낮은 k)도 같다: 같은 이항에서 ccdf(k) < 1인 k = 185–274 가운데 42곳이 k + 1이다(1 − q ≤ 0.038). 사용자 결정 2026-11-20: 계약이 먼저다(2026-11-22에 범위를 잰 값으로 고침). 작은 쪽(log 판은 |t|)이 2⁻¹⁰⁰⁰ 아래이거나 t가 끝(0, 1, −inf)이면 이 단계가 없다. `logpmf`는 상대 1e-14, `entropy`는 연속과 같은 혼합 기준이다. `median()`은 `icdf(1/2)`, `mode()`는 최빈값이 둘 이상이면 NaN이다(T21). 매개변수 하나가 점질량을 만들면(이항 p = 0·1이나 n = 0, 포아송 μ = 0, 기하 p = 1, 음이항 p = 1이나 n = 0) 분산은 0, 왜도·첨도는 NaN이다.
- 정수여야 하는 매개변수(이항·베타이항의 n, 초기하의 셋, 이산 균등의 두 끝)는 정수가 아니면 NaN이다. 음이항의 n은 0 이상의 실수이고 n = 0은 0의 점질량이다(이항의 n = 0, 포아송의 μ = 0과 같고, special의 `nbdtr(k, 0, p)` = 1과도 같다; 사용자 결정 2026-11-19, T33).
  - special과 다른 곳: special은 함수의 극한값을 내지만(`nbdtr(k, n, 0)` = 0, `pdtr(k, inf)` = 0), 분포로는 질량이 무한대로 달아나 진짜 분포가 아니므로 `NegativeBinomial(p=0)`과 `Poisson(mu=inf)`는 NaN이다.
- `sample`의 `rng`는 `sd.random.default_rng(...)`의 Generator이고 꼭 줘야 한다(T27; §2.5). `shape=None`이 기본이고 매개변수 집합마다 표본 하나다.

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
| 균등 | `Uniform(*, a, b)` | `uniform`, `Uniform` |
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
- **느슨한 영역** (잰 최대): `StudentT`의 `icdf`·`iccdf`에서 df ≥ 1e5이고 확률이 2^-1000(약 9.3e-302) 아래인 구석은 3.2e-16이다(`ilog…`는 2.8e-16; 사용자 결정 2026-11-19, T42). 전에는 1.4e-14였다: 깊은 역함수가 베타의 인자 w = df/(df + t²) ≈ 1을 log w로 구하고 t를 1 − w에서 냈는데, log w를 2⁻⁵⁴까지 맞춰도 1 − w에는 그 1/|log w|배로 옮았다. 지금은 w > 1/2이면 log(1 − w)에서 Newton을 한 번 더 해 1 − w를 직접 구한다. 나머지 범위 밖 점도 모두 1e-14 안이다. `Gamma` a > 1e6의 역함수는 전에 2.5e-14였는데, special의 `gammaincinv`를 고친 뒤 1.7e-16이 됐다(2026-11-16; 보장 영역은 그대로다).
- **잰 정확도**: Arb 참값 대비 오차는 [bench/stats/accuracy.md](../../bench/stats/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 메서드마다 있다. 계약 단위의 값은 계약 대비 열이다. 같은 점에서 잰 scipy·R·Boost의 오차는 그 표의 비교 패키지 열에 있다.
- **속도**: scipy와 견준 잰 값은 [bench/stats/speed.md](../../bench/stats/speed.md)(기준선에서 생성한다, bench/README.md §1)에 있다. 원소별 커널과 스칼라 호출의 짧은 길은 docs/design/stats.md §3.4에 있다.
- **보장 영역 밖과 NaN** (사용자 결정 2026-11-17): 모든 호출은 정해진 시간 안에 끝나고, 구할 수 없는 값은 NaN이다. 닫힌 꼴 분포는 정의역 전체에서 값을 낸다(z나 x/scale이 double을 넘으면 극한값). special 기반 분포가 NaN이 되는 곳:
  - `Gamma`·`InverseGamma`의 a > 2¹⁰⁰, `ChiSquared`의 df > 2¹⁰¹: cdf 계열과 역함수.
  - `Gamma`·`InverseGamma`의 아주 작은 모양(2026-11-18): 서브노멀 a(< 2⁻¹⁰²²)에서도 cdf 계열과 역함수가 값을 낸다(기준 모양 2⁻⁶⁰⁰에서 계산해 옮긴다. `Gamma(a=5e-324).logccdf(1.0) = −745.957`). a ≤ 2⁻⁶⁰⁰에서 위꼬리 목표 Q ≥ 1600·a인 근은 0이다(`InverseGamma`는 inf). `InverseGamma`의 역함수 넷과 median은 근이 double 밖이면 끝으로 정한다: 가장 큰 double에서도 아래꼬리가 목표에 못 미치면 inf, 가장 작은 양수에서 이미 넘으면 0이다(사용자 결정 2026-11-19, T51; 전에는 scale < 2⁻⁹⁰⁰이거나 서브노멀 a에서 NaN이었다: `InverseGamma(a=5e-311, scale=5e-311).median()` = inf, `.icdf(1e-310)` = 6.07e-310). 엔트로피는 a < 2⁻¹⁰⁰⁰에서 1/a + log(s/a) + 1 + γ(1/a가 넘치면 inf)다. `F(dfn=inf, dfd)`(역감마의 극한)도 같다. 작은 a의 위꼬리 근(Q = c·a이면 y ≈ e^(−c−γ))은 double로 나타낼 수 있으면 언제나 찾는다(사용자 결정 2026-11-19, T52): 전에는 a < 1e-6에서 c가 약 665–745(근이 2⁻⁹⁶⁰ 아래의 정규수나 서브노멀)이면 NaN이었다(`Gamma(a=1e-8).iccdf(700e-8)` = 5.52e-305). 근이 가장 작은 서브노멀의 절반 아래면 0이고, 척도가 크면 그 곱이 값이다(`Gamma(a=1e-8, scale=1e300).iccdf(800e-8)` = 2.05e-48, 전에는 NaN). `InverseGamma`도 같다(`InverseGamma(a=1e-100, scale=1e-300).icdf(800e-100)` = 4.86e47, 전에는 NaN). 2⁻⁶⁰⁰ 아래의 정규수 a도 같은 방법이다: 위꼬리는 기준 모양에서, 아래꼬리는 log y = (log p + log Γ(1 + a))/a의 닫힌 꼴로 푼다(`Gamma(a=1e-307).iccdf(5e-307)` = 0.0037974640015846902, `Gamma(a=1e-307).icdf(1e-20)` = 0; 둘 다 NaN이던 곳). 남은 한계는 서브노멀 근의 반올림(절대 2⁻¹⁰⁷⁴ 단위)뿐이다. `ChiSquared`는 df < 2⁻¹⁰²¹에서 NaN 그대로다.
  - 감마 계열의 깊은 꼬리(확률 < 2⁻¹⁰⁰⁰) 역함수는 반복이 100번 안에 수렴하지 않으면 NaN이다(a ≥ 1e12에서 나온다. 전에는 수렴하지 않은 마지막 값을 냈다: `Gamma(a=1e-300).ilogccdf(-1e50)`이 1.27e80, 참값 1e50).
  - 베타·t·F의 깊은 꼬리 역함수도 같다(2026-11-19): log x에서 Newton 걸음이 2⁻⁵⁴·max(1, |log x|) 아래가 되면 멈추고, 100번째에도 마지막 걸음이 2⁻⁴⁸·max(1, |log x|)를 넘으면 NaN이다. 전에는 기준이 log x의 간격보다 촘촘해 보장 영역의 Newton 점 셋에 하나꼴로 100번을 다 돌았다(값은 맞았지만 호출 하나가 0.3–0.5 ms였다: `Beta(a=1.16e5, b=0.063).ilogcdf(-1627.7)`).
  - `Beta`의 a, b가 모두 2¹⁰⁰ 초과, `F`의 dfn, dfd가 모두 2¹⁰¹ 초과: cdf 계열과 역함수.
  - `ChiSquared`·`StudentT`·`F`의 df < 2⁻¹⁰²¹(서브노멀 전부와 가장 작은 정규수의 구간 [2⁻¹⁰²², 2⁻¹⁰²¹)): pdf, cdf 계열, 역함수, median, entropy. df의 절반을 double로 정확히 나타낼 수 없다(F는 dfn, dfd 중 하나라도). 비중심 t·F도 같고(§2.4), 표본도 같은 경계다(§2.5). 일부러 둔 예외는 nc > 0인 `NoncentralChiSquared`다: df < 2⁻¹⁰²¹에서도 값을 낸다(0의 점질량을 품은 잘 정의된 포아송 혼합이다; 사용자 결정 2026-11-19, T44, 정의는 §2.4). `ChiSquared`와 nc = 0인 비중심 카이제곱은 표본도 NaN이다(T48). `StudentT`의 entropy가 [2⁻¹⁰²², 2⁻¹⁰²¹)에서 4.5e307을 내던 것을 NaN으로 고쳤다(2026-11-19).
  - **`Beta`·`BetaBinomial`의 작은 모양** (사용자 결정 2026-11-19, T50): a나 b가 2⁻¹⁰²¹ 아래이면 모양이 0으로 가는 극한 분포다: 두 점 0과 1(`BetaBinomial`은 0과 n)이고 P(위) = a/(a + b)다(서브노멀 모양도 반올림 하나). 다른 모양이 작지 않으면 한쪽 질량이 아주 작거나 0으로 반올림되어 사실상 한 점의 점질량이다(`Beta(a=5e-324, b=2)`는 0의 점질량, `Beta(a=2**-1022, b=2)`의 P(1)은 2⁻¹⁰²³). 2⁻¹⁰²¹을 지날 때 연속이다: `Beta(a=2**-1022, b=2**-1021)`의 P(1)은 1/3이고, 그 바로 위의 베타 자신과 같은 값이다. 모든 메서드가 그 분포의 값이다: 밀도는 질량이 있는 점에서 inf, 그 밖에서 0(`pmf`는 질량), cdf·ccdf와 log 판은 계단, 역함수는 그 점(`icdf(1)`·`iccdf(0)`은 T32대로 위 끝), 평균 q·(1 또는 n), 분산 q(1 − q)(n²배), 한 점이면 분산 0이고 왜도·첨도 NaN, `Beta`의 `mode`는 베타의 규칙(b ≥ 1이면 0, a ≥ 1이면 1, 그 밖은 NaN), `BetaBinomial`은 큰 질량의 점(같으면 NaN), 엔트로피는 `Beta`가 −inf(T47처럼), `BetaBinomial`은 두 질량의 섀넌 엔트로피(한 점이면 정확히 0; 전에는 8.8e-19), 지지집합은 족의 것(T32). 표본도 같다(§2.5). 다른 모양이 클 때 크기 a인 양은 두 점의 값이다(평균은 베타 자신과 같고, 분산과 ccdf는 1 + b배쯤 다르다: `Beta(a=5e-324, b=0.5).standard_deviation()` = 3.14e-162, 베타 자신은 2.57e-162). 두 점의 값은 Arb의 불완전 베타와 1e-15 안에서 같다(`Beta(a=1e-310, b=3e-310).cdf(0.5)` = 0.75). 같은 점에서 scipy는 틀린다: `beta.cdf(0.5)` = 1.0, `var()` = NaN, `ppf(0.9)`는 `OverflowError`, `betabinom.pmf(7)`·`entropy()`는 NaN. 전에는 b가 작으면 `icdf`·`median`이 NaN, 두 모양이 서브노멀이면 `pdf`가 NaN, `BetaBinomial`의 평균·분산이 NaN이었다.
  - 역함수: 아래 범위 밖에서는 구한 근에서 정방향 log 꼬리가 목표와 상대 1e-12 안에서 맞을 때만 값을 내고, 아니면 NaN이다. 감마 계열 a ∉ [1e-6, 1e6], `Beta` a, b ∉ [1e-3, 1e5], `StudentT` df ∉ [1e-3, 1e6], `F` dfn, dfd ∉ [1e-3, 1e5].
  - `F.entropy`: dfn이나 dfd가 1e-2 미만이면 NaN.
  - `F.logpdf`: 두 자유도가 모두 1e15 초과이고 베타 인자가 double 아래로 내려갈 때.
  - `Beta.logcdf`: b < 1e-3이고 x > 0.9이며 cdf가 정규수 아래(< 2⁻¹⁰²²)일 때(`logccdf`는 a < 1e-3이고 x < 0.1일 때). cdf가 2⁻⁹⁶⁰ 아래의 정규수이면 그 값의 log다(cdf는 이 구석에서 Arb와 4e-16 안이다: `Beta(a=1, b=1e-300).logcdf(0.95)` = log(2.996e-300) = −689.678).
- scipy와 다르게 동작하는 곳은 docs/design/stats.md §2.4에 있다.

### §2.3 T5의 보장 영역과 잰 정확도
- **보장 영역** (사용자 결정 2026-11-16): 잰 매개변수 범위다. 이항 n ≤ 1e6, 포아송 μ ≤ 1e6, 기하 p ≥ 1e-9, 음이항 n ∈ [0.1, 1e4]·p ≥ 1e-4, 초기하 total ≤ 3000, 베타이항 n ≤ 2000·a, b ∈ [0.1, 100], 로그 급수 p ≥ 1e-8, Zipf a ≤ 60. `Bernoulli`와 `DiscreteUniform`은 닫힌 꼴이라 정의역 전체다. 그 밖도 값을 계산하지만 보장하지 않는다.
- **잰 정확도**: Arb 참값 대비 오차는 [bench/stats/accuracy.md](../../bench/stats/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 메서드마다 있다. 계약 단위의 값은 계약 대비 열이고, 분위수는 허용 구간 안이면 0이다. 같은 점에서 잰 scipy·R·Boost의 오차는 그 표의 비교 패키지 열에 있다(분위수는 허용 구간 밖으로 벗어난 정도가 계약 대비 열이다).
- **속도**: 잰 값은 [bench/stats/speed.md](../../bench/stats/speed.md)(기준선에서 생성한다, bench/README.md §1)에 있다. 지지집합이 작은 경우(이항 n = 40 등)는 정수점 값을 기억하는 표가 거의 모든 원소에 답해서 원소당 비용이 아주 작고, 넓은 경우(이항 n = 1e9, 포아송 μ = 3e8 등, 거의 모든 k가 다름)는 원소마다 한 번 계산하는 비용이다.

- **보장 영역 밖** (사용자 결정 2026-11-17): 모든 호출은 정해진 시간 안에 끝나고, 구할 수 없는 값은 NaN이다. NaN이 되는 곳:
  - 이항: n > 2⁵²의 cdf 계열·분위수·mode·entropy, n·p·q > 2⁴²의 entropy. n > 2⁵³이면 평균에서 먼 k의 pmf·logpmf도 NaN이다(`Binomial(n=2**60, p=0.5).pmf(1)`; 평균 근처는 값이다).
  - 음이항: n < 2⁻¹²⁸, entropy는 n·q/p² > 2⁴² 또는 p < 2⁻²⁰.
  - 초기하: total > 2⁵², kurtosis는 total > 2²⁴, 합이 2²⁰항을 넘는 cdf 계열(total 1e12부터 일부).
  - 베타이항: n + a + b > 2³⁵(pmf 포함), 합이 2²⁰항을 넘는 cdf 계열(n = 1e9부터 일부), 모멘트는 a, b가 [2⁻¹⁵⁰, 2¹⁵⁰] 밖(2⁻¹⁰²¹ 아래의 모양은 극한 분포의 모멘트다, §2.2의 T50).
  - 포아송·음이항: k ≥ 2⁵³에서 pmf(k)가 무시할 수 없는 cdf 계열.
  - `DiscreteUniform`: high − low가 넘칠 때.

### §2.4 T6의 보장 영역과 잰 정확도
- **보장 영역** (사용자 결정 2026-11-16, T26):
  - 닫힌 꼴 5개(`Logistic Gumbel GEV Pareto Rayleigh`)는 매개변수와 입력의 정의역 전체다. `InverseGamma`는 감마처럼 a ≤ 1e6이다.
  - 비중심 셋은 special의 비중심 함수가 잰 범위다(docs/api/special.md §14): `NoncentralChiSquared` df ∈ [1e-2, 1e4]·nc ≤ 1e4, `NoncentralT` df ∈ [0.1, 1e5]·|nc| ≤ 100, `NoncentralF` dfn, dfd ∈ [1e-2, 1e4]·nc ≤ 1e4. 확률은 1e-300까지 보장한다(작은 쪽 꼬리가 그 아래인 x와, 그 아래 확률의 역함수는 보장 밖). 모멘트와 `mode`도 1e-14다.
  - 비중심 셋의 `entropy`는 적응 구적이고 보장하지 않는다. 참값은 아직 만들지 않았다(10코어로 약 12시간; 나중에 잰다).
  - nc = 0이면 중심 분포(`ChiSquared`, `StudentT`, `F`)와 모든 메서드가 비트까지 같다: 밀도부터 역함수, 모멘트, median, mode, entropy, `sample`(같은 `rng` 상태에서). 2026-11-19에 비중심 F의 평균·분산·표준편차·왜도·첨도·median과 비중심 카이제곱·F의 표본을 맞췄다(전에는 `NoncentralF(dfn=3, dfd=11, nc=0).variance()`가 1.7072310405643738, `F(3, 11)`은 1.707231040564374였고, 표본은 비중심의 방법으로 뽑았다).
- **잰 정확도**: Arb 참값 대비 오차는 [bench/stats/accuracy.md](../../bench/stats/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 메서드마다 있다. 계약 단위의 값은 계약 대비 열이다. 같은 점에서 잰 scipy·R·Boost의 오차는 그 표의 비교 패키지 열에 있다.
- **속도**: 잰 값은 [bench/stats/speed.md](../../bench/stats/speed.md)(기준선에서 생성한다, bench/README.md §1)에 있다. 비중심 분포는 항 여러 개의 합에 중심 분포의 불완전 함수가 더 들어 같은 꼴의 중심 분포보다 무겁다.

- **보장 영역 밖과 NaN** (사용자 결정 2026-11-17): 비중심 셋은 급수가 정해진 항 수 안에 끝나지 않으면 NaN이다.
  - cdf 계열·역함수·median: 카이제곱은 nc가 약 2.33e9를 넘으면(평균 근처부터, df = 1–1e4에서 같다), F는 nc가 약 1.47e9를 넘으면, t는 |nc|가 약 3.8e4를 넘으면 NaN이다. 카이제곱은 df가 1.1e24를 넘으면 평균 근처에서 NaN이다.
    - 경계가 다른 이유: 1에 가까운 쪽의 합은 가중치의 종 전체(약 19·√(nc/2)항)를 지나야 해서 nc ≈ 1.47e9에서 걸음 수를 넘는다. 그 합이 끝나지 않으면 작은 쪽 꼬리 Q ≤ 1/2를 구해 1 − Q로 낸다(2026-11-19; 전에는 카이제곱도 nc ≈ 1.47e9부터 1에 가까운 쪽과, 그 쪽을 거치는 역함수가 NaN이었다: `NoncentralChiSquared(df=3, nc=1.5e9).icdf(0.5)`). 카이제곱은 작은 쪽의 항이 종을 다 지나기 전에 사라져 2.33e9까지 간다. F는 베타 인자가 가중치의 종 안에서 거의 변하지 않아 두 쪽 다 종 전체를 지나야 하므로 1.47e9에서 평균 근처 전체가 NaN이 된다.
  - pdf·logpdf: F는 nc가 약 1.3e8을 넘을 때, t는 |nc|나 |t|가 약 1e15를 넘을 때 NaN이다. 카이제곱은 큰 df와 큰 nc가 함께일 때만 NaN이다(df가 작으면 nc = 1e300까지 값이 나온다).
  - mode: t는 |nc| > 128·√(1 + df)이면 NaN이다. F는 nc > 1.3e8이면 NaN이다.
  - `ilogcdf`·`ilogccdf`: logp ≤ −2⁶²는 NaN이다.
  - `NoncentralT`의 df < 2⁻¹⁰²¹, `NoncentralF`의 dfn이나 dfd < 2⁻¹⁰²¹: t·F처럼(§2.2) pdf, cdf 계열, 역함수, median, entropy가 NaN이다(2026-11-19; 전에는 `NoncentralT(df=1e-310, nc=1).median()`이 inf, pdf가 값이었다). df = 1e-300처럼 경계 위는 값이다(`NoncentralT(df=1e-300, nc=1).median()` = inf가 맞는 값이다: T의 부호가 Z + nc의 부호이고 크기는 무한대로 간다). `NoncentralChiSquared`는 nc > 0이면 df < 2⁻¹⁰²¹에서도 값을 낸다(일부러 둔 예외: 0의 점질량을 품은 포아송 혼합이라 잘 정의된다; 사용자 결정 2026-11-19, T44). nc = 0이면 `ChiSquared`처럼 NaN이다.
    - **점질량** (사용자 결정 2026-11-19, T46·T47): 이 영역은 df = 0의 극한 분포다. 0에 질량 e^(−nc/2)가 있고(j = 0 항, 모양이 정규수 아래인 카이제곱이라 질량이 0에 붙어 있다), 나머지는 χ²(2j), j ≥ 1의 포아송 혼합이다. `cdf(0)` = e^(−nc/2)(전에는 0이었고 `cdf(1e-300)`은 0.6065였다), `ccdf(0)` = 1 − e^(−nc/2), `logcdf(0)` = −nc/2다. x > 0의 밀도와 꼬리는 극한 분포의 값이다(Arb 대비 1e-14; 전에는 0 근처의 `ccdf`·`logcdf`·`logccdf`가 NaN이었다). `pdf(0)`은 inf(점질량; df < 2의 밀도처럼), `mode()`는 0이다. 역함수는 목표가 질량 안이면 0이다: `icdf(p)`는 p ≤ e^(−nc/2), `iccdf(q)`는 q ≥ 1 − e^(−nc/2), `ilogcdf(t)`는 t ≤ −nc/2에서 0이고, median은 e^(−nc/2) ≥ 1/2이면 0이다. `entropy()`는 −inf다(점질량이라 유한한 미분 엔트로피가 없다). 모멘트는 혼합의 값(평균 df + nc 등)이고, 표본은 질량의 비율만큼 0이다. 예: `NoncentralChiSquared(df=2**-1022, nc=1).cdf(0)` = 0.60653, `.cdf(1)` = 0.73288.
  - `NoncentralF`의 dfn이 2⁻¹⁰²¹ 위이지만 아주 작을 때(약 2⁻⁹⁹⁷ 아래)는 cdf 계열과 역함수가 NaN이었다(special `ncfdtr`가 다음 항을 무시해도 되는지 보는 검사가 비를 dfn/2로 나눴다). 이제 j = 0 항의 값이다: `NoncentralF(dfn=2**-1021, dfd=5, nc=2).cdf(1)` = e^(−1) = 0.36788(2026-11-19).
  - `entropy`(보장하지 않는 계약)는 영역 안에서도 NaN일 수 있다: 질량이 double 범위 밖에 남는 아주 작은 자유도(F의 dfd = 0.01, 카이제곱 df = 0.01·nc ≤ 30 등), 그리고 반분 12번에 수렴하지 않는 큰 nc(카이제곱 df = 3.5에서 약 7e7부터).

### §2.5 표본 (T7)
```
X.sample(shape=None, /, *, rng)
```
- `rng`: `sd.random.default_rng(seed)`의 Generator이고, 꼭 줘야 한다(T27). 다른 것은 `TypeError`다. 표본은 생성기의 스트림에서 차례로 받으므로, 호출하면 `rng.state["position"]`이 원소 수만큼 나아간다.
- **모양** (사용자 결정 2026-11-19, T38): `shape=None`(기본값)이면 매개변수 집합마다 표본 하나다. 매개변수가 스칼라면 Python float, 배열이면 그 브로드캐스트 모양이다. int나 int의 tuple을 주면 결과는 `shape` + 매개변수의 브로드캐스트 모양이다(scipy 신형과 같다). 매개변수가 (3,)이고 `shape=(1000,)`이면 결과는 (1000, 3)이라 분포마다 1000개씩이다. `shape=()`는 `sd.random`의 메서드처럼 `ValueError`다(0차원 배열이 없다): `sample(): 0-dimensional arrays are not supported (stardust has no 0-d arrays)`. 음수 축은 `sample(): negative dimensions are not allowed`다.
- 물리 메모리보다 큰 요청은 스트림이 움직이기 전에 `MemoryError`다(`sample(): cannot allocate ... bytes: more than this machine's physical memory`; docs/api/random.md).
- **값**: float64다. 이산 분포도 정수인 float64다(icdf와 같다; Zipf가 double을 넘으면 inf). 매개변수가 정의역 밖이거나 NaN인 원소는 NaN이다.
- **뽑을 수 없는 매개변수** (사용자 결정 2026-11-17): 표본기의 반복은 모두 횟수가 정해져 있고, 그 안에 뽑을 수 없으면 NaN이다. `Hypergeometric`은 표준편차가 65536을 넘거나 total ≥ 2⁵³일 때, `DiscreteUniform`은 폭이 2⁶⁴ 이상일 때, `StudentT`·`NoncentralT`는 df < 2⁻¹⁰²¹일 때(값 메서드와 같은 경계, §2.2), `NoncentralF`는 두 자유도 중 하나라도 2⁻¹⁰²¹ 아래이면 NaN이다(값 메서드와 같다; 사용자 결정 2026-11-19, T45. 전에는 둘 다 작을 때만 NaN이고 하나만 작으면 0이나 inf가 나왔다. nc = 0도 NaN이라, 극한값을 내는 `F`의 표본과는 이 경계 아래에서 다르다. 다른 자유도가 inf이면 극한 분포의 표본이다)(2026-11-19에 맞췄다: 처음에는 1e-310에서 ±inf가 나왔고, 그 뒤 경계가 서브노멀(< 2⁻¹⁰²²)이라 `StudentT(df=2**-1022)`의 표본이 inf, cdf·median은 NaN이었다). `ChiSquared`와 nc = 0인 `NoncentralChiSquared`는 df < 2⁻¹⁰²¹이면 NaN이다(값 메서드와 같다; 사용자 결정 2026-11-19, T48. 전에는 0이었다). nc > 0인 비중심 카이제곱은 점질량의 분포라(§2.4) 표본이 값이다. `F`는 dfn이나 dfd가 2⁻¹⁰²¹ 아래이면 NaN이다(값 메서드와 같다; 사용자 결정 2026-11-19, T49. 전에는 0이나 inf의 극한값이었다. dfd = inf이면 `ChiSquared(dfn)`처럼 NaN이고, dfn = inf이면 값이 있는 역감마의 극한이라 표본도 값이다). `Beta`·`BetaBinomial`은 모양이 2⁻¹⁰²¹ 아래이면 값 메서드와 같은 극한 분포의 표본이다(§2.2의 T50: P(위) = a/(a + b)인 두 점, 한쪽 질량이 0이면 한 점). `Zipf`는 a > 1025이면 1이다. `Gamma`·`InverseGamma`는 서브노멀 모양에서도 값 메서드가 값이라(§2.2) 표본도 값(0, inf)이다. 극한값을 내면서 값 메서드가 NaN인 표본기는 이제 없다.
- **재현**: 같은 시드와 같은 호출 순서면 같은 값이다. 스레드 수와 무관하고, 한 호출을 둘로 나누어 불러도 같다. numpy·scipy와 값은 다르다(생성기와 방법이 다르다). 같은 시드의 값은 판 사이에 반올림만큼 바뀔 수 있다(계산 순서를 바꾼 `BetaBinomial`, 다변량 정규·t의 표본: docs/design/stats.md T73, T74).
- **방법**: 분포마다 정확한 방법(기각이나 조합, 근사 없음)이다. 정규는 ziggurat, 감마는 Marsaglia–Tsang, 이항은 BTRS, 포아송은 PTRS 등이고 표는 docs/design/stats.md §4에 있다.
- **분포 검사** (커밋 4151d2a의 bench/stats/t7/quality.md): 62개 경우에서 표본 10⁶개의 KS·카이제곱 p값은 가장 작은 것이 `Beta(a=0.5, b=3)`의 KS 0.0159이고, 나머지는 모두 0.03 이상이다(62개 검정에서 기대할 만한 크기다). 아주 작은 모양도 포함한다(감마 a = 1e-3은 질량의 47.5%가 5e-324 아래라 그 표본은 0이다).
- **속도**: numpy Generator(PCG64)와 견준 잰 값은 [bench/stats/speed.md](../../bench/stats/speed.md)(기준선에서 생성한다, bench/README.md §1)에 있다. 단어를 둘 이상 쓰는 방법은 원소마다 Philox 블록 하나를 더 계산하는데, PCG64는 이어지는 두 단어를 싸게 낸다. 스칼라 매개변수의 `sample()`(`shape=None`)은 생성기의 상태에서 원소 하나를 바로 예약하는 짧은 길이다(docs/design/stats.md T75).

## §3. 다변량 분포 (T9)
```
MultivariateNormal(*, mu, cov)        MultivariateT(*, mu, shape, df)
Multinomial(*, n, p)                  Dirichlet(*, alpha)

X.pdf(x, /)  X.logpdf(x, /)           # Multinomial: pmf, logpmf
X.cdf(x, /, *, atol=1e-6, return_error=False, max_points=None)    # MultivariateNormal, MultivariateT; -> CdfResult
X.mean()  X.covariance()  X.entropy()
X.sample(shape=None, /, *, rng)
```
- **매개변수** (키워드만, T19): `mu` (..., d), `cov`·`shape` (..., d, d), `df` (...), `n` (...), `p` (..., K), `alpha` (..., K). 앞 축끼리 브로드캐스트한다(배치, T29). 공분산은 아래 삼각만 읽는다(sd.linalg.cholesky와 같다). 요약값(`covariance()`, MultivariateT의 `covariance()`)도 아래 삼각으로 만든 대칭 행렬에서 나온다(2026-11-19; 전에는 위 삼각을 그대로 옮겼다). 매개변수는 읽기 전용 속성이고 `repr`은 단변량과 같은 꼴이다. 배열 매개변수는 단변량처럼 만들 때 한 번 복사해 쓰기 금지로 든다(D-5). 스칼라 매개변수(`df`, `n`)의 numpy 실수 스칼라는 float로 받는다. 배치끼리 브로드캐스트할 수 없으면 `MultivariateNormal(): shapes could not be broadcast together: (2,) (3,)`이다(배열·단변량과 같은 문장; 스칼라 매개변수는 나오지 않는다). x와 배치도 같다: `logpdf(): shapes could not be broadcast together: (3,) (4,)`.
  - 매개변수는 `sd.ndarray`이고, 스칼라 매개변수(`df`, `n`)는 Python 실수도 된다. 리스트·tuple·numpy 배열은 단변량, special, linalg처럼 `TypeError`다(사용자 결정 2026-11-19, T34): `MultivariateNormal(): 'mu' must be an sd.ndarray, not list`. 입력 x도 `sd.ndarray`만 받는다: `logpdf(): 'x' must be an sd.ndarray, not list`(단변량 메서드와 같다; 사용자 결정 2026-11-19, T39).
- **입력** x는 (..., d)이고, 앞 축이 배치와 브로드캐스트한다. 결과는 브로드캐스트한 모양이고, 축이 없으면 Python float이다. `mean()`은 배치 + (d,), `covariance()`는 배치 + (d, d), `entropy()`는 배치 모양이고, 배열이면 쓰기 금지다(D-4). Multinomial의 공분산 대각은 n·p_i·Σ_{j≠i} p_j, Dirichlet은 (m_i·Σ_{j≠i} a_j/a0)/(a0 + 1)(m = a/a0)로 계산해 한 성분이 지배해도 상쇄하지 않고(`Dirichlet(alpha=[1e8, 1e-8]).covariance()[0, 0]` = 1.0e-24), a0가 커도 넘치지 않는다. K = 1인 Dirichlet의 엔트로피는 정확히 0이다(한 점).
- **정의역** (T29, T8): 양의 정부호가 아닌 공분산(allow_singular는 없다), NaN이나 inf인 mu, df ≤ 0, 정수가 아닌 n, 음수가 있거나 합이 1에서 sqrt(eps) 밖인 p, 0 이하인 alpha는 모두 NaN이다. 그 배치 원소의 모든 메서드가 NaN이다. p는 합으로 나눠 쓴다(sd.random.choice의 p와 같다).
- **지지집합**: Multinomial의 x가 합이 n인 음이 아닌 정수가 아니면 pmf는 0이다. Dirichlet의 x는 K개 성분이다. 음수가 있거나 합이 1에서 sqrt(eps)·K 밖이면 pdf는 0이다. 0인 성분에서는 극한값이다(alpha < 1이면 inf, = 1이면 그 항이 1, > 1이면 0). 정규·t에서 inf인 성분이 있으면 pdf는 0이다.
- **MultivariateT**: df = inf는 MultivariateNormal(mu, shape)이다. df < 2⁻¹⁰²¹이면 모든 메서드가 NaN이다(StudentT와 같은 경계; 사용자 결정 2026-11-19, T43. 전에는 2⁻¹⁰²²). mean은 df > 1, covariance는 df > 2에서 shape·df/(df − 2)이다. 1 < df ≤ 2이면 발산하고(shape의 부호로 ±inf, 0인 성분은 0), df ≤ 1이면 NaN이다(StudentT의 분산과 같다).
- **정확도** (T29): 보장 영역 κ(cov) ≤ 1e9, d ≤ 100에서 logpdf·entropy는 |오차| ≤ 1e-14·max(1, |값|), pdf는 상대 1e-14다. 촐레스키 분해, 삼각 풀이, 이차 형식이 모두 double-double이다. 잰 값(Arb 참값 대비, 계약 단위는 계약 대비 열)은 [bench/stats/accuracy_multivariate.md](../../bench/stats/accuracy_multivariate.md)의 표(`python -m bench.accuracy`가 만든다)에 있다. 같은 점에서 잰 scipy와 R(mvtnorm)의 오차는 그 표의 비교 패키지 열에 있다. Multinomial의 logpmf·entropy와 Dirichlet의 logpdf·entropy도 double-double이다. n = 1e5인 다항 엔트로피(log n!과 기댓값이 1e6에서 12.6으로 상쇄)와 alpha = 1e6인 디리클레 엔트로피가 Arb 참값과 1e-14 안이다(scipy의 오차는 그 표의 `Multinomial.entropy`·`Dirichlet.entropy` 행).
- **보장 영역 밖** (사용자 결정 2026-11-17): Multinomial의 logpmf는 n이 약 1e13을 넘으면 항들의 상쇄가 큰 곳(최빈값 근처)에서 NaN일 수 있다. entropy는 범주별 표준편차의 합이 2¹⁷을 넘거나 n > 2⁵³이면 NaN이다(점질량은 0).
- **cdf** (MultivariateNormal, MultivariateT): P(X ≤ x)다. 정규의 d ≤ 3과 t의 d ≤ 3은 정확한 공식(1-D 적분)이고, 오차 출력은 0이다. d ≥ 4는 Genz의 준몬테카를로(고정 시드, 결정적)로 계산하고 `atol`(기본 1e-6)까지 점을 늘린다. `max_points`는 평가 수의 상한이다(기본 10⁶ × 변수 수). 1 이상인 int이고, int64 밖이면 `cdf(): 'max_points' is out of range`(`OverflowError`), 1 미만이면 `cdf(): 'max_points' must be >= 1`(`ValueError`)이다. `return_error=True`이면 `CdfResult(value, error)`(namedtuple, 튜플처럼 풀 수 있다)이고, 오차 추정은 표준오차의 3배다(통계적이라 일부 점에서 넘을 수 있다: 잰 값으로 최대 1.9배). +inf인 좌표는 빼고, −inf가 있으면 0, NaN이면 NaN이다. `atol`은 유한한 양수(int, float, numpy 정수)이고, 아니면 `cdf(): 'atol' must be a finite number > 0`(`ValueError`)이다. double을 넘는 int는 `cdf(): 'atol' is out of range`(`OverflowError`)다(2026-11-19; 전에는 inf에도 `'atol' must be a positive number`였다).
  - **보장 영역** (상대 1e-14): 정규 d = 2는 P ≥ 1e-15이면 모든 ρ이고, d = 3은 음의 상관이 하나 이하이고 P ≥ 1e-15이거나 P ≥ 1e-10이고 문제의 조건수 ≤ 100일 때다. t의 d = 2, 3은 P ≥ 7e-13이다(잰 최대 1.9e-15).
  - **그 밖** (잰 값은 [bench/stats/accuracy_multivariate.md](../../bench/stats/accuracy_multivariate.md)의 `cdf.*` 행, P의 크기별로는 커밋 4151d2a의 bench/stats/t9/cdf_accuracy.md): d = 2는 약 6·eps·|ln P|이고(1e-300 근처에서 2.7e-13), d = 3은 조건수를 잴 수 있는 점에서 3.1u × 조건수 안이고, 전체 최대는 1.5e-13이다(음의 상관이 둘이고 P가 1e-254 근처인 칸; 음의 상관 셋·특이에 가까운 행렬의 P ≈ 1e-14 칸은 1.2e-13). 특이에 가까운 행렬에서 조건부 적분이 틀리던 것(κ = 7.7e7에서 1.7e-7)을 고치면서 깊은 꼬리 두 칸이 6.8e-14, 5.5e-14에서 이 값으로 늘었다(사용자 결정 2026-11-18: 영역 안의 값을 우선한다).
  - **NaN** (사용자 결정 2026-11-17): 값이 NaN이면 오차 출력도 NaN이다. t의 d = 3은 |x/sd| > 1e100이고 조건부 적분으로 넘어가야 할 때 NaN이다(df ≤ 1에서 나온다). QMC는 96단계와 `max_points`에서 멈춘다.
  - **속도**: 잰 값은 [bench/stats/speed.md](../../bench/stats/speed.md)(기준선에서 생성한다, bench/README.md §1)에 있다. 기본 atol = 1e-6은 scipy의 기본값(1e-5, t는 실제 오차 약 5e-5)보다 엄격해서, 기본값끼리 견주면 sd가 더 많은 점을 쓴다.
- **표본**: 벡터 하나가 생성기 스트림의 원소 하나다. 정규·t는 mu + L z(·sqrt(df/V)), 다항은 조건부 이항, 디리클레는 감마를 정규화한다(아주 작은 alpha는 log로). 스레드 수와 무관하게 같은 값이다.
- **속도**: 잰 값은 [bench/stats/speed.md](../../bench/stats/speed.md)(기준선에서 생성한다, bench/README.md §1)에 있다. 정규·t의 logpdf는 순서가 정해진 NEON 내적과 점마다의 오차 상한을 쓴다(d ≤ 16은 이웃한 두 점을 함께 계산한다). scipy는 점을 모아 BLAS로 푸는데, BLAS는 합의 순서를 몰라 상한을 셀 수 없다(docs/design/stats.md §3.4).

### §3.1 행렬 분포 (T10)
```
Wishart(*, df, scale)        InverseWishart(*, df, scale)        MatrixNormal(*, mu, rowcov, colcov)

X.pdf(x, /)  X.logpdf(x, /)
X.mean()  X.mode()  X.variance()  X.entropy()        # MatrixNormal: mean, entropy
X.sample(shape=None, /, *, rng)
```
- **매개변수** (T30, scipy의 이름): `df` (...)는 실수이고 df > p − 1이어야 한다. `scale`·`rowcov`·`colcov` (..., p, p)는 대칭 양의 정부호이고 아래 삼각만 읽는다(`mean`·`mode`·`variance`도 아래 삼각으로 만든 대칭 행렬에서 나온다). 매개변수와 입력의 형은 다변량과 같다(리스트는 `TypeError`, T34, T39). 행렬 정규의 평균은 `mu` (..., n, p)다(scipy의 `mean`은 `mean()` 메서드와 이름이 겹친다; T18의 위치 이름). 배치는 다변량과 같다(T29).
- **입력** x는 (..., p, p)(행렬 정규는 (..., n, p))이고, 아래 삼각만 읽는다. 양의 정부호가 아니면 지지집합 밖이다(pdf 0, logpdf −inf).
- **메서드**: `mean`, `mode`, `variance`(원소별 분산; scipy의 `var`, 이름은 단변량의 `variance()`와 같다: 사용자 결정 2026-11-20, D-25)다. 결과 배열은 쓰기 금지다(D-4). 행렬 정규의 엔트로피는 세 항(np/2·(1 + log 2π), p/2·log|U|, n/2·log|V|)을 double-double로 더한다(크고 부호가 반대인 항이 상쇄해도 1e-14 안). (p, p, p, p) 공분산 텐서는 없다. 행렬 정규는 `mean`과 `entropy`만 있다.
  - Wishart의 mode는 df ≥ p + 1일 때 (df − p − 1)·scale이고, 그 밖은 NaN이다.
  - 역Wishart의 mean은 df > p + 1이면 scale/(df − p − 1)이고, 그 밖은 발산한다(성분의 부호로 ±inf, 0인 성분은 0). variance는 df > p + 3이면 유한하고, p + 1 < df ≤ p + 3이면 inf, df ≤ p + 1이면 NaN이다. p = 1이면 InverseGamma(df/2, scale/2)와 같다.
- **정의역**: df ≤ p − 1(특이 Wishart), 양의 정부호가 아닌 scale, NaN인 mu는 NaN이다(T8).
- **큰 df** (사용자 결정 2026-11-19, T35): Wishart·역Wishart의 logpdf는 df ≥ 1e9에서 크기 n·p·log n인 항들을 식으로 지운 꼴로 계산한다(감마의 logpdf처럼 log Γ마다 Stirling 꼴). W = V⁻¹X/n(역Wishart는 ΨX⁻¹/n)의 고유값이 최빈값에서 1 근처라, W 대신 고유값이 같은 대칭 행렬 I + E를 쓴다. E = L_A⁻¹(B − n·A)L_A⁻ᵀ/n이고 (A, B)는 (V, X)(역Wishart는 (X, Ψ))다. B − n·A는 입력에서 정확히 만들므로 E의 반올림은 E 자신에 비례한다. W를 인수의 곱(L_X⁻¹L_Ψ)으로 만들면 W − I에 1에 대한 κ·2⁻¹⁰⁴이 남고, 그것이 n·|W − I|배로 커진다(κ = 1e9, df = 1e30에서 2.7e-13). log|W| − (tr W − p)는 I + E의 촐레스키 피벗 1 + u_j(u_j = E_jj − s_j, s_j = Σ_k L_jk²)로 Σ_j[(log1p(u_j) − u_j) − s_j]이다. 각 부분이 |E|² 크기라 앞의 n/2에 |E| 크기의 반올림이 곱해지지 않는다. p = 1은 Gamma(df/2, 2·scale)·InverseGamma(df/2, scale/2)와 그 정확도 안에서 같다.
  - **보장 영역**: κ(scale) ≤ 1e9, p ≤ 100, df ≤ 1e30에서 1e-14 혼합 기준이다. 잰 값(Arb 참값 대비, p ≤ 12, κ ≤ 1e9, 평균·최빈값과 그 분포에서 뽑은 표본)은 df = 1e9–1e30에서 1.1e-16 이하다. 전에는 상쇄 검사가 항의 정확도를 2⁻¹⁰⁰으로 잡았지만 실제는 2⁻⁹⁰ 안팎이라, p = 1, df = 1e15에서 6.6e-13, p = 2, df = 1e17에서 1.1e-12가 NaN 대신 나왔다.
  - 그 밖: 남는 반올림의 상한 2⁻¹⁰⁰·p·(n·δ²·(64 + p + κ̂) + p·(64 + log n + δ))이 1e-14·max(1, |값|)를 넘으면 NaN이다(n = df, δ = max|W − I|, κ̂는 L_A의 대각 비의 제곱으로 어림한 κ(A)). n이 붙은 항은 n·δ²에 비례하고, 값에는 n/2·(log|W| − tr W + p) ≈ −n·‖W − I‖²_F/4가 들어 있어 상한이 값과 함께 자란다.
    - 2026-11-20 측정(p = 1–100, κ(scale) = 1–1e9, 평균·최빈값·뽑은 점, df 격자 10^(1/8)로 1e47까지, 그 위는 10²씩 1e306까지): 역Wishart는 어느 점도 NaN이 아니다.
    - Wishart는 n·max V가 2¹⁰⁰⁰(약 1.1e301)을 넘으면 n·V를 만들 수 없어 원래 식으로 돌아가고, 그 식의 항이 상쇄해 NaN이다(κ = 1에서 df = 1e302, κ = 1e9에서 1e294부터).
    - 보장 영역 밖의 Arb 대비(p ≤ 12, κ ≤ 1e9, Wishart df = 1e35–1e290, 역Wishart 1e35–1e300)도 1.1e-16 이하다.
- **아주 큰 인자** (2026-11-18): 행렬 정규·Wishart·역Wishart의 logpdf는 원소가 아무리 커도(double 최댓값, inf 포함) 값이나 −inf이고 NaN이 아니다(pdf는 0). 예: `MatrixNormal`(1×1 표준)의 x = 1.5e154에서 −1.125e308, 1.9e154부터 −inf.
- **정확도**: logpdf·entropy는 촐레스키 분해, V⁻¹, 삼각 풀이를 double-double로 계산한다. 엔트로피의 크기 n log n인 항은 다시 묶어서 지운다. 잰 값(Arb 참값 대비)은 [bench/stats/accuracy_multivariate.md](../../bench/stats/accuracy_multivariate.md)의 표에 있다. 같은 점에서 잰 scipy의 오차는 그 표의 비교 패키지 열에 있다. scipy 1.18의 `invwishart.entropy()`는 공식이 틀렸다(그 표의 `InverseWishart.entropy` 행; 몬테카를로로도 확인했다, tests/stats/test_matrix.py). 보장 영역(사용자 2026-11-16)은 T9와 같이 κ(scale), κ(rowcov), κ(colcov) ≤ 1e9, p ≤ 100에서 logpdf·entropy 1e-14 혼합 기준이다. 표의 점은 κ = 1e9, p = 30(행렬 정규 20×15), df = 1e30까지다.
- **표본**: Wishart는 Bartlett 분해로 뽑는다. 대각은 √χ²(df − j)이고 실수 df도 된다. 역Wishart는 (L A⁻ᵀ)(L A⁻ᵀ)ᵀ, 행렬 정규는 μ + L_U Z L_Vᵀ다. 행렬 하나가 스트림 원소 하나다.
- **속도**: 잰 값은 [bench/stats/speed.md](../../bench/stats/speed.md)(기준선에서 생성한다, bench/README.md §1)에 있다.

## §4. 기술통계 (T8)
```
describe(x, /, *, axis=None, correction=1, bias=True, nan_policy='propagate')
skew(x, /, *, axis=None, bias=True, nan_policy='propagate', keepdims=False)
kurtosis(x, /, *, axis=None, fisher=True, bias=True, nan_policy='propagate', keepdims=False)
moment(x, /, *, order, center=None, axis=None, nan_policy='propagate', keepdims=False)
zscore(x, /, *, axis=None, correction=0, nan_policy='propagate')
rankdata(x, /, *, method='average', axis=None, nan_policy='propagate')
sem(x, /, *, axis=None, correction=1, nan_policy='propagate', keepdims=False)
iqr(x, /, *, axis=None, q=(0.25, 0.75), scale=1.0, method='linear', nan_policy='propagate', keepdims=False)   # scale: 수 또는 'normal'
mode(x, /, *, axis=None, nan_policy='propagate', keepdims=False)
entropy(pk, qk=None, /, *, base=None, axis=None, nan_policy='propagate', keepdims=False)
differential_entropy(x, /, *, window_length=None, base=None, axis=None, method='auto',
                     nan_policy='propagate', keepdims=False)
```
- **공통**: 입력은 위치로, 옵션은 키워드로만 준다. `axis`는 None(전체, sd.mean·sd.var와 같다; scipy는 0), int, int의 tuple이나 리스트다(배열·linalg의 축 인자와 같다; 사용자 결정 2026-11-19, T37). 중복된 축은 배열과 같은 `skew(): repeated axis`(`ValueError`)다. `rankdata`의 axis는 int나 None이고, 메시지도 그것을 말한다: `rankdata(): 'axis' must be an int or None, not float`(2026-11-19; 전에는 tuple이 아닌 틀린 값에 다른 함수의 `'axis' must be an int or a tuple or list of ints`가 나왔다). 수 옵션(`correction`, `base`, `iqr`의 `scale`, `moment`의 `center`와 `order`)은 int와 float이고 numpy 정수도 int다(API.md §2). double을 넘는 int는 `OverflowError`다: `sem(): 'correction' is out of range`, `moment(): 'order' is out of range`(2026-11-19; 전에는 numpy 정수가 `TypeError`, 큰 int가 Python의 `int too large to convert to float`였고 `entropy(base=10**400)`은 값을 냈다). 입력은 `sd.ndarray`, 실수의 시퀀스, Python 실수이고 float64로 계산한다(complex는 `TypeError`). 배열 함수(`sd.sum`, `sd.var` 등)와 분포의 메서드는 리스트를 받지 않지만, 기술통계는 일부러 받는다(사용자 결정 2026-11-19, T40): 손에 든 데이터를 바로 요약하는 함수라 scipy.stats처럼 리스트를 그대로 넘기는 쓰임이 흔하기 때문이다. 결과는 float64이고, 축이 남지 않으면 Python float이다. 결정은 docs/design/stats.md T28.
- **입력**(사용자 결정 2026-11-20, D-30): `sd.ndarray`, 실수의 시퀀스, 실수 스칼라(Python과 numpy의 스칼라, 0차원 numpy 배열), frame의 `Series`·`DataFrame`이다(sd.regression과 같다; frame의 결측값은 NaN이 되어 `nan_policy`를 따른다. 문자열 열은 `skew(): 'x' must have numeric columns`). 그 밖의 형은 `skew(): 'x' must be an sd.ndarray, a sequence of numbers or a frame Series or DataFrame, not dict`(`TypeError`)이고, 들쭉날쭉한 리스트는 `sd.asarray`의 `ValueError`를 함수 이름으로 낸다(`skew(): inhomogeneous nested sequence ...`). 입력 모양을 지키는 `zscore`·`rankdata`는 스칼라 입력이면 Python float다. `axis`가 int64 밖이면 `skew(): 'axis' is out of range`(`OverflowError`)다.
- **무한대**: 보상 합도 합이 유한하지 않으면 보정을 버린다. +inf만 있는 데이터의 평균은 inf(`describe([1, inf, 2]).mean` = inf, sd.mean과 같다), 거듭제곱이 넘치는 `moment`도 inf다(`moment([1e160, 2e160, -1e160], order=2, center=0)` = inf). +inf와 −inf가 함께면 NaN이다.
- **`nan_policy`**(scipy와 같다): 'propagate'면 NaN이 있는 줄의 결과가 NaN이다(mode는 NaN을 하나의 값으로 센다). 'omit'이면 줄마다 NaN을 빼고 계산하고, 'raise'면 NaN이 있을 때 `ValueError`다.
- **scipy와의 차이** (이름과 기본값만이고 값은 계약 안에서 같다): 축의 기본값(None), 자유도 인자 이름(`ddof` → `correction`), iqr의 `rng=(25, 75)`·`interpolation` → `q=(0.25, 0.75)`·`method`(sd.quantile과 같다), moment의 `order`는 꼭 준다. 빈 줄의 `describe`는 `nobs=0`과 NaN이다(다른 기술통계와 같다; scipy는 `ValueError`; 사용자 결정 2026-11-20, D-32).
- **`iqr`의 `q`**: 수 둘의 시퀀스이고 각각 [0, 1] 안이다. scipy처럼 정렬해 쓰므로 범위는 음수가 아니다(`q=(0.75, 0.25)`도 7.0). 원소는 다른 수 옵션처럼 int·float·numpy 수이고 bool·str는 `iqr(): 'q' must be a pair of numbers in [0, 1], not bool`(`TypeError`), 개수나 범위가 틀리면 `ValueError`다. 두 분위수는 한 번의 선택으로 구한다. `scale='normal'`이면 표준정규의 IQR 1.349…로 나눈다(σ의 일치 추정).
- **`differential_entropy`의 `window_length`**: 1 이상인 int이고, int64 밖이면 `differential_entropy(): 'window_length' is out of range`(`OverflowError`), 1 미만이면 `'window_length' must be >= 1`(`ValueError`)이다.
- **`differential_entropy`의 동점**: 같은 값이 있어 간격이 0이면 −inf다(scipy와 같다. 상수 데이터 포함). 무한한 값이 있어 간격이 inf이면 inf다(scipy와 같다. 둘이 함께면 NaN). `method='correa'`는 상수인 창에서 0/0이라 NaN이다(scipy와 같다).
- **나눗셈의 끝**: `iqr(scale=0)`, `entropy(base=1)`, `differential_entropy(base=1)`은 IEEE 나눗셈이다: inf, 분자가 0이면 NaN(scipy와 같다).
- **문자열 옵션**(`method`, `nan_policy`)은 정확한 이름만 받는다(대소문자를 바꾸지 않는다; scipy의 `differential_entropy`는 대문자도 받는다). 메시지는 모든 영역의 형식이다(사용자 결정 2026-11-19, T41): `skew(): 'nan_policy' must be 'propagate', 'omit' or 'raise', not 'x'`(`ValueError`), 문자열이 아니면 `..., not NoneType`(`TypeError`). `iqr`의 `method` 검사는 sd.quantile의 것이지만 메시지는 `iqr(): 'method' must be ...`다. `moment(order=[])`는 `ValueError`(`'order' must not be empty`), `order`를 빠뜨리면 `moment(): missing 1 required keyword-only argument: 'order'`, `order=None`은 다른 틀린 값과 같은 `'order' must be an int or a sequence of ints, not NoneType`이다(2026-11-19; 기술통계 함수의 호출 모양 에러는 모두 `skew(): unexpected keyword argument 'ddof'`처럼 함수 이름 형식이다, 생성자의 T36과 같다), `entropy(base='e')`는 `TypeError`다.
- **결과 객체**: `describe`는 `DescribeResult(nobs, minmax, mean, variance, skewness, kurtosis)`, `mode`는 `ModeResult(mode, count)`다(scipy와 같은 namedtuple; 정의는 모든 영역의 결과 형이 있는 `stardust._results`이고, `sd.stats`가 내보낸다. 결과 형을 내보내는 영역은 stats와 regression이다: 사용자 결정 2026-11-20, D-25). nobs는 propagate면 int이고, omit이면 줄마다의 개수다. 배열인 필드는 쓰기 금지다(사용자 결정 2026-11-20, D-4: 모든 영역의 결과 객체와 같다). 다변량 `cdf`의 `CdfResult`는 내보내지 않는다.
- **정확도**: 적률은 평균을 두 부분(hi + lo)으로 들고, 결과가 그 오차를 키우는 합만 보상 합(TwoSum)으로 계산한다: 평균을 내는 `describe`·`zscore`는 편차 d의 합, 3차 적률을 내는 `skew`·`moment`·`describe`는 d와 d³의 합, 초과 첨도를 내는 `kurtosis`·`describe`는 d²와 d⁴의 합이다. 나머지(`sem`, 분산, 짝수 적률)는 보상 없는 고정 순서의 합이다(docs/design/stats.md T69). 그래서 평균이 퍼짐보다 아주 큰 데이터(평균 1e8·퍼짐 1)에서도 평균의 반올림으로 잃지 않는다. scipy는 같은 데이터에서 skew·kurtosis가 상대 1e-8쯤 틀린다(평균을 double로 반올림한 채 빼기 때문이다). 그 밖의 값은 scipy와 1e-13 안에서 같다(tests/stats/test_descriptive.py).
- **속도**: 잰 값은 [bench/stats/speed.md](../../bench/stats/speed.md)(기준선에서 생성한다, bench/README.md §1)에 있다.

## §5. 배열과 special에 더하는 것
- 배열(T0, 완료): `cov`, `corrcoef`, `quantile`, `median`, NaN 리덕션. docs/api/array.md §7.9–§7.11.
- special(T1, T2): `stdtr stdtrit chdtr chdtrc chdtri fdtr fdtrc fdtri pdtr pdtrc bdtr bdtrc nbdtr nbdtrc`, `chndtr chndtrix nctdtr nctdtrit ncfdtr ncfdtri ndtri_exp`. docs/api/special.md에 더한다.

## §6. 검정과 상관 (`sd.htest`로 옮김)
- 검정은 이 영역에서 뺐다(사용자 결정 2026-11-18). 2026-11-20에 위치를 새 최상위 모듈 `sd.htest`로 정했다(docs/design/htest.md H1). 검정과 상관(Pearson·Spearman·Kendall의 계수와 p값)은 그곳에 있다.
- scipy 이름 → 위치: `scipy.stats.ttest_ind` → `sd.htest.ttest_ind`, `scipy.stats.pearsonr` → `sd.htest.pearsonr`처럼 같은 이름이다. 전체 대응표는 htest의 API 문서에 둔다.
