# sd.stats API

> 상태: **T10 구현** (2026-11-16): 연속 분포 12개(T4), 이산 분포 10개(T5), 극값·수명 6개와 비중심 3개(T6), 모든 분포의 `sample`(T7), 기술통계(T8), 다변량 4개(T9), 행렬 분포 3개(T10). §6(검정)은 이 영역에서 뺐다(2026-11-18). 결정과 내부 구조는 [docs/design/stats.md](../design/stats.md) (T1–T30).

## §1. 공통 규칙
- 입력은 위치로만(`/` 앞), 옵션은 키워드로만(`*` 뒤). 별칭은 없다.
- 피연산자는 `sd.ndarray`와 Python 스칼라. float64로 계산하고(bool·int64는 바꿔서), complex는 `TypeError`다. 모두 Python 스칼라면 결과도 Python `float`다.
- 분포의 매개변수가 정의역 밖이면 그 원소의 결과는 NaN이다. 에러는 없다.
- 메서드의 에러는 메서드 이름으로 시작한다: `cdf(): 'x' must be an sd.ndarray or a real number, not str`, `pdf(): 'x' must be real, not a complex128 array`. 다변량·행렬 분포도 같다(`cdf(): 'x' must have 2 components on its last axis, not 3`) (사용자 결정 2026-11-18).
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 표본도 같은 `rng` 상태에서는 같다(docs/api/random.md).
- 정확도: 보장 영역에서 참값 대비 상대 1e-14(special과 같은 계약). 분포마다 보장 영역과 잰 값은 해당 절에 적는다.
- 검정과 상관은 아직 없다. 나중에 `sd.stats.tests`에 둔다(§6).

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
- **속도** (bench/stats/t4/speed.md, 원소 10⁵개; 2026-11-17 최적화 뒤, docs/design/stats.md §3.4): 1스레드는 scipy의 0.68–6.6배(중앙값 1.84), 10스레드는 2.2–48배(중앙값 10.6). Python float 하나는 scipy보다 43–66배 빠르다(호출당 0.19–0.29 µs). 1스레드에서 scipy보다 느린 것은 96행 중 12행이다: 카이제곱의 cdf·ccdf(0.68–0.71배; 반정수가 아닌 모양의 불완전 감마)와 icdf(0.76–0.78배), t의 icdf(0.83배), t·F의 cdf(0.9배대).
- **보장 영역 밖과 NaN** (사용자 결정 2026-11-17): 모든 호출은 정해진 시간 안에 끝나고, 구할 수 없는 값은 NaN이다. 닫힌 꼴 분포는 정의역 전체에서 값을 낸다(z나 x/scale이 double을 넘으면 극한값). special 기반 분포가 NaN이 되는 곳:
  - `Gamma`·`InverseGamma`의 a > 2¹⁰⁰, `ChiSquared`의 df > 2¹⁰¹: cdf 계열과 역함수.
  - `Gamma`·`InverseGamma`의 아주 작은 모양(2026-11-18): 서브노멀 a(< 2⁻¹⁰²²)에서도 cdf 계열과 역함수가 값을 낸다(기준 모양 2⁻⁶⁰⁰에서 계산해 옮긴다. `Gamma(a=5e-324).logccdf(1.0) = −745.957`). a ≤ 2⁻⁶⁰⁰에서 위꼬리 목표 Q ≥ 1600·a인 근은 0이다(`InverseGamma`는 inf, scale < 2⁻⁹⁰⁰이면 NaN). 서브노멀 a에서 Q가 약 665·a와 1600·a 사이이면 NaN이다(근이 2⁻⁹⁶⁰ 아래). `ChiSquared`는 df < 2⁻¹⁰²¹에서 NaN 그대로다.
  - 감마 계열의 깊은 꼬리(확률 < 2⁻¹⁰⁰⁰) 역함수는 반복이 100번 안에 수렴하지 않으면 NaN이다(a ≥ 1e12에서 나온다. 전에는 수렴하지 않은 마지막 값을 냈다: `Gamma(a=1e-300).ilogccdf(-1e50)`이 1.27e80, 참값 1e50).
  - `Beta`의 a, b가 모두 2¹⁰⁰ 초과, `F`의 dfn, dfd가 모두 2¹⁰¹ 초과: cdf 계열과 역함수.
  - `ChiSquared`·`StudentT`·`F`의 df < 2⁻¹⁰²¹: pdf, cdf 계열, 역함수, median, entropy. df의 절반을 double로 나타낼 수 없다.
  - 역함수: 아래 범위 밖에서는 구한 근에서 정방향 log 꼬리가 목표와 상대 1e-12 안에서 맞을 때만 값을 내고, 아니면 NaN이다. 감마 계열 a ∉ [1e-6, 1e6], `Beta` a, b ∉ [1e-3, 1e5], `StudentT` df ∉ [1e-3, 1e6], `F` dfn, dfd ∉ [1e-3, 1e5].
  - `F.entropy`: dfn이나 dfd가 1e-2 미만이면 NaN.
  - `F.logpdf`: 두 자유도가 모두 1e15 초과이고 베타 인자가 double 아래로 내려갈 때.
  - `Beta.logcdf`: b < 1e-3이고 x > 0.9일 때.
- scipy와 다르게 동작하는 곳은 docs/design/stats.md §2.4에 있다.

### §2.3 T5의 보장 영역과 잰 정확도
- **보장 영역** (사용자 결정 2026-11-16): 잰 매개변수 범위다. 이항 n ≤ 1e6, 포아송 μ ≤ 1e6, 기하 p ≥ 1e-9, 음이항 n ∈ [0.1, 1e4]·p ≥ 1e-4, 초기하 total ≤ 3000, 베타이항 n ≤ 2000·a, b ∈ [0.1, 100], 로그 급수 p ≥ 1e-8, Zipf a ≤ 60. `Bernoulli`와 `DiscreteUniform`은 닫힌 꼴이라 정의역 전체다. 그 밖도 값을 계산하지만 보장하지 않는다.
- **잰 정확도** (bench/stats/t5/accuracy.md, Arb 참값 대비, 계약 단위): 61만 점 전부 계약 안이다. 값 메서드의 최대는 6.5e-15이고, 분위수는 모든 점이 허용 구간 안이다. 같은 점에서 scipy는 꼬리에서 상대오차 1이나 inf를 낸다(이항·음이항·초기하의 cdf, 초기하·로그 급수의 logpmf). 분위수에서는 scipy의 답이 허용 구간 안에 드는 비율이 64–100%다(기하 `isf` 64%, 포아송 `isf` 64%, 이항 `isf` 88%).
- **속도** (bench/stats/t5/speed.md, 2026-11-17 최적화 뒤): 두 묶음을 잰다. 지지집합이 작은 묶음(이항 n = 40 등, 원소 10⁵개)은 정수점 값을 기억하는 표가 거의 모든 원소에 답해서 1스레드가 scipy의 1.15배 이상(중앙값 37배)이다. 이 묶음은 원소당 1–2 ns라 10스레드가 1스레드보다 빠르지 않은 행이 25개 있다. 넓은 묶음(이항 n = 1e9, 포아송 μ = 3e8 등, 거의 모든 k가 다름)은 한 번 계산하는 비용을 보인다: 1스레드 0.66–6055배(중앙값 5.3), 10스레드 중앙값 20배. 넓은 묶음에서 scipy보다 느린 것은 베타이항의 pmf·logpmf(0.66배; double-double log Γ 넷)뿐이고, 포아송의 cdf 계열은 scipy와 같은 수준(1.0배)이다. Python float 하나는 scipy보다 32–107배 빠르다(호출당 0.17–0.65 µs).

- **보장 영역 밖** (사용자 결정 2026-11-17): 모든 호출은 정해진 시간 안에 끝나고, 구할 수 없는 값은 NaN이다. NaN이 되는 곳:
  - 이항: n > 2⁵²의 cdf 계열·분위수·mode·entropy, n·p·q > 2⁴²의 entropy.
  - 음이항: n < 2⁻¹²⁸, entropy는 n·q/p² > 2⁴² 또는 p < 2⁻²⁰.
  - 초기하: total > 2⁵², kurtosis는 total > 2²⁴, 합이 2²⁰항을 넘는 cdf 계열(total 1e12부터 일부).
  - 베타이항: n + a + b > 2³⁵(pmf 포함), 합이 2²⁰항을 넘는 cdf 계열(n = 1e9부터 일부), 모멘트는 a, b가 [2⁻¹⁵⁰, 2¹⁵⁰] 밖.
  - 포아송·음이항: k ≥ 2⁵³에서 pmf(k)가 무시할 수 없는 cdf 계열.
  - `DiscreteUniform`: high − low가 넘칠 때.

### §2.4 T6의 보장 영역과 잰 정확도
- **보장 영역** (사용자 결정 2026-11-16, T26):
  - 닫힌 꼴 5개(`Logistic Gumbel GEV Pareto Rayleigh`)는 매개변수와 입력의 정의역 전체다. `InverseGamma`는 감마처럼 a ≤ 1e6이다.
  - 비중심 셋은 special의 비중심 함수가 잰 범위다(docs/api/special.md §14): `NoncentralChiSquared` df ∈ [1e-2, 1e4]·nc ≤ 1e4, `NoncentralT` df ∈ [0.1, 1e5]·|nc| ≤ 100, `NoncentralF` dfn, dfd ∈ [1e-2, 1e4]·nc ≤ 1e4. 확률은 1e-300까지 보장한다(작은 쪽 꼬리가 그 아래인 x와, 그 아래 확률의 역함수는 보장 밖). 모멘트와 `mode`도 1e-14다.
  - 비중심 셋의 `entropy`는 적응 구적이고 보장하지 않는다. 참값은 아직 만들지 않았다(10코어로 약 12시간; 나중에 잰다).
- **잰 정확도** (bench/stats/t6/accuracy.md, Arb 참값 대비, 계약 단위): 60만 점에서 보장 영역 안 최대 5.5e-15이고, 밖도 1.2e-15 안이다. 같은 점에서 scipy는 꼬리에서 틀린다. 예: `NoncentralF.cdf` 상대오차 2.3e95, `NoncentralT.cdf`·`GEV.ccdf`·`Pareto.logcdf` inf, `InverseGamma.icdf` 2.4e-5.
- **속도** (bench/stats/t6/speed.md, 원소 10⁵개; 2026-11-17 최적화 뒤): 1스레드는 scipy의 0.54–5.2배(중앙값 1.62), 10스레드는 2.0–44배(중앙값 9.9)다. Python float 하나는 scipy보다 36–84배 빠르다. 1스레드에서 scipy보다 느린 것은 72행 중 11행이고, 모두 비중심 분포와 역감마 밀도다.
  - 비중심 F의 `logpdf`(0.54배)·`pdf`(0.69배)·`cdf`(0.70배)·`ccdf`(0.79배), 비중심 t의 `cdf`(0.67배), 비중심 카이제곱의 `ccdf`(0.72배): 항 20–50개의 합에 중심 분포의 불완전 함수 한 번이 더 든다.
  - 비중심 t의 `pdf`·`ccdf`, 역감마의 `pdf`·`logpdf`: 0.9배 안팎.

- **보장 영역 밖과 NaN** (사용자 결정 2026-11-17): 비중심 셋은 급수가 정해진 항 수 안에 끝나지 않으면 NaN이다.
  - cdf 계열·역함수·median: 카이제곱·F는 nc가 약 2.3e9를 넘으면, t는 |nc|가 약 3.8e4를 넘으면 NaN이다. 카이제곱은 df가 1.1e24를 넘으면 평균 근처에서 NaN이다.
  - pdf·logpdf: F는 nc가 약 1.3e8을 넘을 때, t는 |nc|나 |t|가 약 1e15를 넘을 때 NaN이다. 카이제곱은 큰 df와 큰 nc가 함께일 때만 NaN이다(df가 작으면 nc = 1e300까지 값이 나온다).
  - mode: t는 |nc| > 128·√(1 + df)이면 NaN이다. F는 nc > 1.3e8이면 NaN이다.
  - `ilogcdf`·`ilogccdf`: logp ≤ −2⁶²는 NaN이다.
  - `entropy`(보장하지 않는 계약)는 영역 안에서도 NaN일 수 있다: 질량이 double 범위 밖에 남는 아주 작은 자유도(F의 dfd = 0.01, 카이제곱 df = 0.01·nc ≤ 30 등), 그리고 반분 12번에 수렴하지 않는 큰 nc(카이제곱 df = 3.5에서 약 7e7부터).

### §2.5 표본 (T7)
```
X.sample(shape=(), /, *, rng)
```
- `rng`: `sd.random.default_rng(seed)`의 Generator이고, 꼭 줘야 한다(T27). 다른 것은 `TypeError`다. 표본은 생성기의 스트림에서 차례로 받으므로, 호출하면 `rng.state["position"]`이 원소 수만큼 나아간다.
- **모양**: `shape`(int 또는 int의 tuple) + 매개변수의 브로드캐스트 모양이다(scipy 신형과 같다). 매개변수가 (3,)이고 `shape=(1000,)`이면 결과는 (1000, 3)이라 분포마다 1000개씩이다. 둘 다 비면 Python float이다.
- **값**: float64다. 이산 분포도 정수인 float64다(icdf와 같다; Zipf가 double을 넘으면 inf). 매개변수가 정의역 밖이거나 NaN인 원소는 NaN이다.
- **뽑을 수 없는 매개변수** (사용자 결정 2026-11-17): 표본기의 반복은 모두 횟수가 정해져 있고, 그 안에 뽑을 수 없으면 NaN이다. `Hypergeometric`은 표준편차가 65536을 넘거나 total ≥ 2⁵³일 때, `DiscreteUniform`은 폭이 2⁶⁴ 이상일 때, `StudentT`·`NoncentralT`는 df가 서브노멀일 때, `NoncentralF`는 두 자유도가 모두 서브노멀일 때 NaN이다. `Zipf`는 a > 1025이면 1이고, `Beta`·`BetaBinomial`·`F`는 서브노멀 모양에서 극한값이다.
- **재현**: 같은 시드와 같은 호출 순서면 같은 값이다. 스레드 수와 무관하고, 한 호출을 둘로 나누어 불러도 같다. numpy·scipy와 값은 다르다(생성기와 방법이 다르다).
- **방법**: 분포마다 정확한 방법(기각이나 조합, 근사 없음)이다. 정규는 ziggurat, 감마는 Marsaglia–Tsang, 이항은 BTRS, 포아송은 PTRS 등이고 표는 docs/design/stats.md §4에 있다.
- **분포 검사** (bench/stats/t7/quality.md): 62개 경우에서 표본 10⁶개의 KS·카이제곱 p값이 모두 0.03 이상이다(가장 작은 값 0.034). 아주 작은 모양도 포함한다(감마 a = 1e-3은 질량의 47.5%가 5e-324 아래라 그 표본은 0이다).
- **속도** (bench/stats/t7/speed.md, 표본 10⁶개, numpy Generator(PCG64) 대비; 2026-11-17 최적화 뒤): 1스레드 0.87–7.9배(중앙값 1.09), 10스레드 7.3–63배(중앙값 9.8). 1스레드에서 numpy보다 느린 것은 31행 중 10행이고 모두 0.87–0.97배다(비중심 카이제곱, 감마, 코시, 베타, F, 큰 n의 이항 등). 단어를 둘 이상 쓰는 방법은 원소마다 Philox 블록 하나를 더 계산하는데, PCG64는 이어지는 두 단어를 싸게 낸다.

  Python float 하나는 호출당 0.24–0.5 µs다(numpy의 0.4–2배).

## §3. 다변량 분포 (T9)
```
MultivariateNormal(*, mu, cov)        MultivariateT(*, mu, shape, df)
Multinomial(*, n, p)                  Dirichlet(*, alpha)

X.pdf(x, /)  X.logpdf(x, /)           # Multinomial: pmf, logpmf
X.cdf(x, /, *, atol=1e-6, return_error=False, max_points=None)    # MultivariateNormal, MultivariateT
X.mean()  X.covariance()  X.entropy()
X.sample(shape=(), /, *, rng)
```
- **매개변수** (키워드만, T19): `mu` (..., d), `cov`·`shape` (..., d, d), `df` (...), `n` (...), `p` (..., K), `alpha` (..., K). 앞 축끼리 브로드캐스트한다(배치, T29). 공분산은 아래 삼각만 읽는다(sd.linalg.cholesky와 같다). 매개변수는 읽기 전용 속성이고 `repr`은 단변량과 같은 꼴이다.
- **입력** x는 (..., d)이고, 앞 축이 배치와 브로드캐스트한다. 결과는 브로드캐스트한 모양이고, 축이 없으면 Python float이다. `mean()`은 배치 + (d,), `covariance()`는 배치 + (d, d), `entropy()`는 배치 모양이다.
- **정의역** (T29, T8): 양의 정부호가 아닌 공분산(allow_singular는 없다), NaN이나 inf인 mu, df ≤ 0, 정수가 아닌 n, 음수가 있거나 합이 1에서 sqrt(eps) 밖인 p, 0 이하인 alpha는 모두 NaN이다. 그 배치 원소의 모든 메서드가 NaN이다. p는 합으로 나눠 쓴다(sd.random.choice의 p와 같다).
- **지지집합**: Multinomial의 x가 합이 n인 음이 아닌 정수가 아니면 pmf는 0이다. Dirichlet의 x는 K개 성분이다. 음수가 있거나 합이 1에서 sqrt(eps)·K 밖이면 pdf는 0이다. 0인 성분에서는 극한값이다(alpha < 1이면 inf, = 1이면 그 항이 1, > 1이면 0). 정규·t에서 inf인 성분이 있으면 pdf는 0이다.
- **MultivariateT**: df = inf는 MultivariateNormal(mu, shape)이다. df < 2⁻¹⁰²²이면 모든 메서드가 NaN이다. mean은 df > 1, covariance는 df > 2에서 shape·df/(df − 2)이다. 1 < df ≤ 2이면 발산하고(shape의 부호로 ±inf, 0인 성분은 0), df ≤ 1이면 NaN이다(StudentT의 분산과 같다).
- **정확도** (T29): 보장 영역 κ(cov) ≤ 1e9, d ≤ 100에서 logpdf·entropy는 |오차| ≤ 1e-14·max(1, |값|), pdf는 상대 1e-14다. 촐레스키 분해, 삼각 풀이, 이차 형식이 모두 double-double이다. 잰 값(bench/stats/t9/accuracy.md, 50자리 참값 대비)은 κ = 1e9, d = 100, df = 1e9까지 반올림 수준(2.7e-16 이하)이다. 같은 점에서 scipy는 κ에 비례해 1e-13–1e-9이고, df = 1e9인 t의 logpdf는 1e-7까지 틀린다. Multinomial의 logpmf·entropy와 Dirichlet의 logpdf·entropy도 double-double이다. n = 1e5인 다항 엔트로피(log n!과 기댓값이 1e6에서 12.6으로 상쇄)와 alpha = 1e6인 디리클레 엔트로피가 50자리 참값과 1e-14 안이다(scipy는 각각 1e-9, 3e-9가 틀린다).
- **보장 영역 밖** (사용자 결정 2026-11-17): Multinomial의 logpmf는 n이 약 1e13을 넘으면 항들의 상쇄가 큰 곳(최빈값 근처)에서 NaN일 수 있다. entropy는 범주별 표준편차의 합이 2¹⁷을 넘거나 n > 2⁵³이면 NaN이다(점질량은 0).
- **cdf** (MultivariateNormal, MultivariateT): P(X ≤ x)다. 정규의 d ≤ 3과 t의 d ≤ 3은 정확한 공식(1-D 적분)이고, 오차 출력은 0이다. d ≥ 4는 Genz의 준몬테카를로(고정 시드, 결정적)로 계산하고 `atol`(기본 1e-6)까지 점을 늘린다. `max_points`는 평가 수의 상한이다(기본 10⁶ × 변수 수). `return_error=True`이면 (값, 오차 추정)이고, 오차 추정은 표준오차의 3배다(통계적이라 일부 점에서 넘을 수 있다: 잰 값으로 최대 1.9배). +inf인 좌표는 빼고, −inf가 있으면 0, NaN이면 NaN이다.
  - **보장 영역** (상대 1e-14): 정규 d = 2는 P ≥ 1e-15이면 모든 ρ이고, d = 3은 음의 상관이 하나 이하이고 P ≥ 1e-15이거나 P ≥ 1e-10이고 문제의 조건수 ≤ 100일 때다. t의 d = 2, 3은 P ≥ 7e-13이다(잰 최대 1.9e-15).
  - **그 밖** (bench/stats/t9/cdf_accuracy.md): d = 2는 약 6·eps·|ln P|이고(1e-300 근처에서 2.7e-13), d = 3은 조건수를 잴 수 있는 점에서 3.1u × 조건수 안이고, 전체 최대는 1.5e-13이다(음의 상관이 둘이고 P가 1e-254 근처인 칸; 음의 상관 셋·특이에 가까운 행렬의 P ≈ 1e-14 칸은 1.2e-13). 특이에 가까운 행렬에서 조건부 적분이 틀리던 것(κ = 7.7e7에서 1.7e-7)을 고치면서 깊은 꼬리 두 칸이 6.8e-14, 5.5e-14에서 이 값으로 늘었다(사용자 결정 2026-11-18: 영역 안의 값을 우선한다).
  - **NaN** (사용자 결정 2026-11-17): 값이 NaN이면 오차 출력도 NaN이다. t의 d = 3은 |x/sd| > 1e100이고 조건부 적분으로 넘어가야 할 때 NaN이다(df ≤ 1에서 나온다). QMC는 96단계와 `max_points`에서 멈춘다.
  - **속도** (bench/stats/t9/cdf_speed.md, 1스레드, 점 하나): 정규 d = 2는 0.4 µs(scipy의 18배), d = 3은 1.0 µs(1700배), t d = 2, 3은 162–516 µs(1.6–3.5배)다. QMC는 같은 atol = 1e-5에서 정규 d = 5가 0.73 ms(2.3배), d = 10이 25.6 ms(1.3배)다. 기본 atol = 1e-6은 scipy의 기본값(1e-5, t는 실제 오차 약 5e-5)보다 엄격해서, 기본값끼리 비교하면 0.2–0.3배다. d ≥ 4인 t는 atol = 1e-4에서 1.9–2.1배다.
- **표본**: 벡터 하나가 생성기 스트림의 원소 하나다. 정규·t는 mu + L z(·sqrt(df/V)), 다항은 조건부 이항, 디리클레는 감마를 정규화한다(아주 작은 alpha는 log로). 스레드 수와 무관하게 같은 값이다.
- **속도** (bench/stats/t9/speed.md, 10⁵개, scipy 대비, 1스레드 / 10스레드):
  - 표본: 1.05–2.2배 / 7.6–17배.
  - 다항 logpmf: 3.2배 / 26배.
  - 디리클레 logpdf: 1.5배 / 11배.
  - 정규·t의 logpdf: d = 3에서 0.70–0.79배 / 2.9–4.1배, d = 10에서 0.56배 / 3.3–4.4배, d = 50에서 0.40배 / 3.0–3.3배다(2026-11-17 최적화 전에는 d = 50이 0.05배). 순서가 정해진 NEON 내적과 점마다의 오차 상한의 비용이다. scipy는 점을 모아 BLAS로 푸는데, BLAS는 합의 순서를 몰라 상한을 셀 수 없다(docs/design/stats.md §3.4).

### §3.1 행렬 분포 (T10)
```
Wishart(*, df, scale)        InverseWishart(*, df, scale)        MatrixNormal(*, mu, rowcov, colcov)

X.pdf(x, /)  X.logpdf(x, /)
X.mean()  X.mode()  X.var()  X.entropy()        # MatrixNormal: mean, entropy
X.sample(shape=(), /, *, rng)
```
- **매개변수** (T30, scipy의 이름): `df` (...)는 실수이고 df > p − 1이어야 한다. `scale`·`rowcov`·`colcov` (..., p, p)는 대칭 양의 정부호이고 아래 삼각만 읽는다. 행렬 정규의 평균은 `mu` (..., n, p)다(scipy의 `mean`은 `mean()` 메서드와 이름이 겹친다; T18의 위치 이름). 배치는 다변량과 같다(T29).
- **입력** x는 (..., p, p)(행렬 정규는 (..., n, p))이고, 아래 삼각만 읽는다. 양의 정부호가 아니면 지지집합 밖이다(pdf 0, logpdf −inf).
- **메서드**: scipy처럼 `mean`, `mode`, `var`(원소별 분산)다. (p, p, p, p) 공분산 텐서는 없다. 행렬 정규는 `mean`과 `entropy`만 있다.
  - Wishart의 mode는 df ≥ p + 1일 때 (df − p − 1)·scale이고, 그 밖은 NaN이다.
  - 역Wishart의 mean은 df > p + 1이면 scale/(df − p − 1)이고, 그 밖은 발산한다(성분의 부호로 ±inf, 0인 성분은 0). var는 df > p + 3이면 유한하고, p + 1 < df ≤ p + 3이면 inf, df ≤ p + 1이면 NaN이다. p = 1이면 InverseGamma(df/2, scale/2)와 같다.
- **정의역**: df ≤ p − 1(특이 Wishart), 양의 정부호가 아닌 scale, NaN인 mu는 NaN이다(T8).
- **보장 영역 밖** (사용자 결정 2026-11-17): Wishart·역Wishart의 logpdf는 df가 약 1e17을 넘고 항들이 서로 상쇄하는 곳(밀도가 큰 곳 근처)에서 NaN이다. 1e15까지는 1e-13 안이다.
- **아주 큰 인자** (2026-11-18): 행렬 정규·Wishart·역Wishart의 logpdf는 원소가 아무리 커도(double 최댓값, inf 포함) 값이나 −inf이고 NaN이 아니다(pdf는 0). 예: `MatrixNormal`(1×1 표준)의 x = 1.5e154에서 −1.125e308, 1.9e154부터 −inf.
- **정확도** (bench/stats/t10/accuracy.md, Arb 참값 대비): logpdf·entropy는 촐레스키 분해, V⁻¹, 삼각 풀이를 double-double로 계산한다. 엔트로피의 크기 n log n인 항은 다시 묶어서 지운다. 잰 값은 κ(scale) = 1e8, p = 12, df = 1e8까지 반올림 수준이다(entropy 5.3e-16 이하, logpdf 0). 같은 점에서 scipy의 logpdf는 최대 6.5e-8, Wishart 엔트로피는 df = 1e8에서 6.5e-9가 틀린다. scipy 1.18의 `invwishart.entropy()`는 공식이 틀려서 상대오차가 2.9까지 간다(몬테카를로로도 확인; tests/stats/test_matrix.py). 보장 영역(사용자 2026-11-16)은 T9와 같이 κ(scale), κ(rowcov), κ(colcov) ≤ 1e9, p ≤ 100에서 logpdf·entropy 1e-14 혼합 기준이다. 측정은 κ = 1e9, p = 30(행렬 정규 20×15), df = 1e8까지이고, 모두 2e-16 이하였다.
- **표본**: Wishart는 Bartlett 분해로 뽑는다. 대각은 √χ²(df − j)이고 실수 df도 된다. 역Wishart는 (L A⁻ᵀ)(L A⁻ᵀ)ᵀ, 행렬 정규는 μ + L_U Z L_Vᵀ다. 행렬 하나가 스트림 원소 하나다.
- **속도** (bench/stats/t10/speed.md, 10⁴개, scipy 대비, 1스레드 / 10스레드): Wishart·역Wishart의 logpdf는 1.8–87배 / 15–650배이고, 표본은 2.4–18배 / 20–75배다. 행렬 정규는 logpdf가 0.87–1.25배 / 5.6–10배, 표본이 1.9–2.6배 / 16배다.

## §4. 기술통계 (T8)
```
describe(x, /, *, axis=None, correction=1, bias=True, nan_policy='propagate')
skew(x, /, *, axis=None, bias=True, nan_policy='propagate', keepdims=False)
kurtosis(x, /, *, axis=None, fisher=True, bias=True, nan_policy='propagate', keepdims=False)
moment(x, /, *, order, center=None, axis=None, nan_policy='propagate', keepdims=False)
zscore(x, /, *, axis=None, correction=0, nan_policy='propagate')
rankdata(x, /, *, method='average', axis=None, nan_policy='propagate')
sem(x, /, *, axis=None, correction=1, nan_policy='propagate', keepdims=False)
iqr(x, /, *, axis=None, q=(0.25, 0.75), scale=1.0, method='linear', nan_policy='propagate', keepdims=False)
mode(x, /, *, axis=None, nan_policy='propagate', keepdims=False)
entropy(pk, qk=None, /, *, base=None, axis=None, nan_policy='propagate', keepdims=False)
differential_entropy(x, /, *, window_length=None, base=None, axis=None, method='auto',
                     nan_policy='propagate', keepdims=False)
```
- **공통**: 입력은 위치로, 옵션은 키워드로만 준다. `axis`는 None(전체, sd.mean·sd.var와 같다; scipy는 0), int, int의 tuple이다. 입력은 `sd.ndarray`, 실수의 시퀀스, Python 실수이고 float64로 계산한다(complex는 `TypeError`). 결과는 float64이고, 축이 남지 않으면 Python float이다. 결정은 docs/design/stats.md T28.
- **`nan_policy`**(scipy와 같다): 'propagate'면 NaN이 있는 줄의 결과가 NaN이다(mode는 NaN을 하나의 값으로 센다). 'omit'이면 줄마다 NaN을 빼고 계산하고, 'raise'면 NaN이 있을 때 `ValueError`다.
- **scipy와의 차이** (이름과 기본값만이고 값은 같다): 축의 기본값(None), 자유도 인자 이름(`ddof` → `correction`), iqr의 `rng=(25, 75)`·`interpolation` → `q=(0.25, 0.75)`·`method`(sd.quantile과 같다), moment의 `order`는 꼭 준다.
- **`differential_entropy`의 동점**: 같은 값이 있어 간격이 0이면 −inf다(scipy와 같다. 상수 데이터 포함). `method='correa'`는 상수인 창에서 0/0이라 NaN이다(scipy와 같다).
- **결과 객체**: `describe`는 `DescribeResult(nobs, minmax, mean, variance, skewness, kurtosis)`, `mode`는 `ModeResult(mode, count)`다(scipy와 같은 namedtuple). nobs는 propagate면 int이고, omit이면 줄마다의 개수다.
- **정확도**: 적률은 평균을 두 부분(hi + lo)으로 들고 보상 합으로 계산한다. 평균 1e8·퍼짐 1의 데이터(2000개)에서 정확한 값(유리수 계산)과 비교한 상대오차는 skew 3.2e-16이고 kurtosis·m2는 0이다. scipy는 같은 데이터에서 skew 1.2e-8, kurtosis 1.1e-8이다(평균을 double로 반올림한 채 빼기 때문이다; m2는 0). 그 밖의 값은 scipy와 1e-13 안에서 같다(tests/stats/test_descriptive.py).
- **속도** (bench/stats/t8/speed.md, 10⁶개, scipy 대비; 2026-11-17 최적화 뒤): 1스레드에서 skew·kurtosis·moment 6.0–7.5배, describe 4.9–6.0배, zscore 1.3배, rankdata 1.25–1.7배, mode 1.0–2.7배, iqr 1.3–2.1배, differential_entropy 2.0–3.1배, omit이 있으면 9–40배다. 10스레드에서는 대부분 10–45배다. 1스레드에서 scipy보다 느린 것은 sem(0.63–0.87배)뿐이다. 보상 합 셋을 쓰는 비용이다(scipy의 std는 보상하지 않는다). 10스레드에서는 3.8–4.0배로 빠르다.

## §5. 배열과 special에 더하는 것
- 배열(T0, 완료): `cov`, `corrcoef`, `quantile`, `median`, NaN 리덕션. docs/api/array.md §7.9–§7.11.
- special(T1, T2): `stdtr stdtrit chdtr chdtrc chdtri fdtr fdtrc fdtri pdtr pdtrc bdtr bdtrc nbdtr nbdtrc`, `chndtr chndtrix nctdtr nctdtrit ncfdtr ncfdtri ndtri_exp`. docs/api/special.md에 더한다.

## §6. `sd.stats.tests` (나중, 이 영역에서 뺌)
- 검정은 이 영역에서 뺐다(사용자 결정 2026-11-18: 나중에 따로 한다). 아래는 그때를 위해 정해 둔 위치다.
- 검정, 상관(Pearson·Spearman·Kendall의 계수와 p값). 목록과 결과 객체는 분포·기술통계를 구현한 뒤에 정한다.
- scipy 이름 → 위치의 대응표를 그때 이 절에 둔다(예: `scipy.stats.ttest_ind` → `sd.stats.tests.ttest_ind`, `scipy.stats.pearsonr` → `sd.stats.tests.pearsonr`).
