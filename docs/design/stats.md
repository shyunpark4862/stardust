# 통계 영역 설계 (sd.stats)

> 상태: **T10과 최적화 구현** (2026-11-18). 검정은 이 영역에서 뺐다(사용자 결정 2026-11-18: 나중에 따로 한다)(§7). 큰 결정은 사용자가 정했다(§1.1). 마일스톤마다 시작 전에 물을 것은 §11에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

`sd.stats`(scipy.stats의 교육용 재구현, 부분집합)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md)에 있다. 공개 함수와 시그니처는 [docs/api/stats.md](../api/stats.md)에 있다.
- 바닥은 `sd.special`(분포의 cdf·icdf 대부분)과 배열·`sd.linalg`(다변량)다. 표본의 난수는 `sd.random`([random.md](random.md))에서 받는다.
- 비교 대상은 scipy 1.18.1이다. 수치는 Apple M4 Pro 한 대에서 잰다.

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-15)
| 번호 | 결정 |
|---|---|
| T1 범위 | `sd.stats` = 분포 + 기술통계 (§1.2) |
| T2 검정 | 하위 모듈 `sd.stats.tests`. 상관(Pearson·Spearman·Kendall)은 계수까지 전부 tests에 둔다. 상세 설계(검정 목록, 결과 객체, 정확 검정)는 분포·기술통계를 구현한 뒤에 정한다 (§7) |
| T3 경계 | p값을 내거나 귀무분포를 쓰면 tests, 아니면 stats. `rankdata`·`zscore`·`skew`·`kurtosis`·`describe` 같은 보조·요약 함수는 stats의 공개 함수이고, tests가 stats를 부르는 방향만 허용한다. 단순 공분산·Pearson 상관행렬은 배열 영역의 `sd.cov`, `sd.corrcoef`다 |
| T4 단변량 분포 | 기본 17 + 극값·수명 6 + 비중심 3 + 이산 추가 5 = 31개 (§1.2) |
| T5 다변량 분포 | 다변량 정규, 다변량 t, 다항, 디리클레, Wishart, 역Wishart, 행렬 정규 (§6) |
| T6 분포 API | scipy의 **신형**: `Normal(mu=0, sigma=1).cdf(x)`. 다변량도 같은 꼴(이름은 우리가 정한다) |
| T7 매개변수화 | 분포마다 하나(교과서 표준). 이름은 scipy 신형을 따른다 |
| T8 잘못된 매개변수 | 정의역 밖이면 원소별 NaN. 에러 없음 (special의 "값 때문에 실패하지 않는다"와 같다) |
| T9 메서드 | 핵심 + 모멘트: `pdf logpdf`(이산은 `pmf logpmf`), `cdf logcdf`, `ccdf logccdf`, `icdf iccdf`, `ilogcdf ilogccdf`, `support`, `mean variance standard_deviation skewness kurtosis`, `median`, `mode`, `entropy`, `sample`. 변환(산술, truncate), `Mixture`, `order_statistic`은 없다 |
| T10 정확도 | special과 같은 계약: 보장 영역에서 상대 1e-14, 밖은 잰 값을 문서화. 참값은 Arb(엄밀한 구간 산술). `ccdf`는 1 − cdf가 아니라 직접 계산해 꼬리도 상대 정확도 (§2.3) |
| T11 구현 위치 | C++ `_core`의 원소별 Op(special과 같은 n항 디스패처·스레드풀). 분포 객체, 검정·기술통계의 조립은 Python (§3) |
| T12 표본 | 분포별 표본은 `sd.random`이 아니라 각 분포의 `X.sample(shape, rng=rng)`. 분포마다 직접 알고리즘 또는 역함수 법 (§4) |
| T13 기술통계 | `describe skew kurtosis moment zscore rankdata sem iqr mode entropy differential_entropy` (§5) |
| T14 배열 추가 | `cov corrcoef quantile median`과 NaN을 건너뛰는 리덕션 (§8.1) |
| T15 special 추가 | 감싸기(`stdtr stdtrit chdtr chdtrc chdtri fdtr fdtrc fdtri pdtr pdtrc bdtr bdtrc nbdtr nbdtrc`)와 새 계산법(`chndtr chndtrix nctdtr nctdtrit ncfdtr ncfdtri ndtri_exp`)을 scipy 이름으로 `sd.special`에 공개, 같은 계약 (§8.2) |
| T16 다변량 cdf | 다변량 정규·t의 cdf는 느슨한 계약으로 넣는다: Genz 방법을 고정 시드로 결정적으로, 오차 추정을 함께 돌려주고 잰 정확도를 문서화한다. 2·3차원은 정확한 공식을 쓸 수 있다 (§6.2) |
| T17 난수 | 별도 모듈 `sd.random` (random.md) |
| T18 매개변수 이름 (T4, 2026-11-15) | 위치가 있는 분포는 정규처럼 `mu`·`sigma`(코시·라플라스의 `sigma`는 표준편차가 아니라 척도), 로그정규는 밑 정규의 `mu`·`sigma`. [0, ∞) 분포는 모양 + `scale`(감마·지수·베이불; rate가 아니다). 모양 매개변수는 scipy 이름(`a b c df dfn dfd`) |
| T19 생성자 | 매개변수는 키워드로만(`Gamma(a=2, scale=3)`; 위치 인자는 TypeError). 기본값은 위치 0·척도 1(`mu=0, sigma=1, scale=1`)에만 있고, 모양 매개변수와 `Uniform(a, b)`는 필수다. 매개변수는 읽기 전용 속성(`X.mu`, 준 값 그대로) |
| T20 log 계약 | `logcdf`·`logccdf`는 상대 1e-14(`log_ndtr`처럼; 값이 0을 지나지 않는다). `logpdf`는 \|오차\| ≤ 1e-14·max(1, \|값\|)(pdf = 1에서 0을 지난다) |
| T21 mode | 최댓값이 한 점이면 그 점(경계여도, 밀도가 발산해도: 감마 a < 1 → 0). 하나로 정해지지 않으면 NaN(균등, 베타 a ≤ 1·b ≤ 1) |
| T22 out= | 입력을 받는 원소별 메서드(`pdf`…`ilogccdf`)는 special처럼 `out=`을 받는다. 매개변수만의 메서드(모멘트, `support` 등)에는 없다 |
| T23 첨도 | `kurtosis()`는 초과 첨도(정규 = 0), 옵션 없음. T8의 기술통계 `kurtosis`의 기본값도 이에 맞춘다 |
| T24 T4 보장 영역 (2026-11-15) | 닫힌 꼴 7개(정규·지수·균등·라플라스·코시·로그정규·베이불)는 정의역 전체. special 기반은 special의 보장 범위에 맞춘다: 감마 a ≤ 1e6(카이제곱 df ≤ 2e6), 베타 a, b ≤ 1e5, t df ≤ 1e6, F 자유도 ≤ 1e5. 그 밖은 잰 값을 문서화한다. t의 역함수에서 df ≥ 1e5이고 확률이 2^-1000 아래인 구석은 느슨한 영역이었다(1.4e-14. 원인은 'BGRAT에 log 꼴이 없다'가 아니라 1 − w를 log w에서 낸 것이었고, T42로 고쳐 3.2e-16이다). `entropy`는 `logpdf`처럼 \|오차\| ≤ 1e-14·max(1, \|값\|), 베이불의 왜도·첨도(0을 지난다)도 같다. 모멘트의 특수값은 수학적 규칙(§2.3)을 따르고 scipy와 다른 곳은 §2.4에 둔다 |
| T25 이산 분포 (T5, 2026-11-15) | 메서드는 `pmf`·`logpmf`(T9), 나머지는 연속과 같다. 정수가 아닌 x에서 `pmf`는 0, `cdf`는 floor(x)에서의 값. `icdf(p)`는 cdf(k) ≥ p인 가장 작은 k, `iccdf(p)`는 ccdf(k) ≤ p인 가장 작은 k(scipy의 ppf·isf), `median`은 `icdf(1/2)`. `mode`는 T21을 따른다: 최빈값이 둘 이상이면 NaN(이항의 (n+1)p, 포아송의 μ가 정수일 때, 이산 균등). `Geometric`은 첫 성공까지의 시행 횟수(k ≥ 1, scipy), `NegativeBinomial`은 n번째 성공 전의 실패 횟수(k ≥ 0). `DiscreteUniform(low, high)`는 [low, high)(scipy randint, `sd.random.integers`와 같다). `Hypergeometric(total, successes, draws)`(scipy의 M, n, N). 분위수의 정확도는 후방 오차(2026-11-16): 목표를 상대 1e-14 안에서 움직였을 때의 정확한 분위수 중 하나(`icdf`·`iccdf`는 p와 1 − p 중 작은 쪽, `ilog…`는 t에 대해; §3.2) |
| T26 극값·수명·비중심 (T6, 2026-11-16) | `GEV`의 모양은 scipy의 c(= −ξ, c > 0이면 위가 막힌 Weibull형). `Rayleigh(*, scale=1)`(T18의 양수형 규칙). `Gumbel`은 최댓값형(gumbel_r)만. 비중심의 nc ≥ 0(nc = 0이면 중심 분포). 비중심 셋의 `mode`는 pdf′ = 0의 근으로 1e-14 계약, `entropy`는 적응 구적이고 잰 정확도를 느슨한 한계로 문서화. 비중심의 보장 영역은 special(§14)의 잰 범위(확률 1e-300까지; 더 깊은 log 꼬리는 보장하지 않는다), 비중심 t의 모멘트는 큰 df에서 1/df 급수로 1e-14(2026-11-16). GEV의 왜도·첨도는 베이불과 같은 혼합 기준(T24; W = Y^c가 베이불이다) |
| T27 표본 (T7, 2026-11-16) | `X.sample(shape=None, /, *, rng)`의 `rng`는 꼭 준다(sd.random의 R3: 전역 상태가 없다). 결과 모양은 `shape` + 매개변수의 브로드캐스트 모양이다(scipy 신형과 같다). `shape`의 기본값과 `()`는 T38이 바꿨다(처음에는 `shape=()`가 기본값이었다). 이산 분포의 표본도 float64다(icdf와 같다: T25; int64를 넘는 Zipf·로그 급수의 값과 inf). 방법은 분포마다 역함수 법과 전용 방법을 재서 골랐다(§4) |
| T28 기술통계 (T8, 2026-11-16) | `axis=None`(전체)이 기본이다(sd.mean·sd.var와 같다; scipy는 0). NaN은 scipy의 `nan_policy`('propagate', 'omit', 'raise')로 받는다. 편향 보정은 scipy의 기본값을 따른다(skew·kurtosis는 `bias=True`, kurtosis는 `fisher=True`, describe·sem은 `correction=1`, zscore는 `correction=0`; 이름은 sd.var의 `correction`). describe·mode는 scipy와 같은 namedtuple(`DescribeResult`, `ModeResult`)이다. iqr은 `q=(0.25, 0.75)`(비율)와 `method`다(sd.quantile과 같다. scipy의 `rng`는 sd.random의 생성기와 이름이 겹친다). moment의 `order`는 꼭 주고, `center`도 둔다. rankdata는 다섯 방법 모두이고 scipy 1.18처럼 모두 float64다(질문의 설명에 적힌 'average만 float64'는 옛 scipy의 동작이었다. scipy와 같게로 정했다). differential_entropy는 scipy의 다섯 방법 모두(vasicek, van es, ebrahimi, correa, auto)다 |
| T29 다변량 (T9, 2026-11-16) | 매개변수는 배치로 받는다(앞 축끼리 브로드캐스트; mu (..., d), cov (..., d, d); x (..., d)). 양의 정부호가 아닌 공분산은 NaN이다(allow_singular 없음, T8). cdf는 값을 돌려주고 `return_error=True`면 (값, 오차 추정)이다. 목표 정확도는 `atol=1e-6`이고 고정 시드라 결정적이다. 다변량 정규의 d ≤ 3은 정확한 공식(이변량·삼변량, 1e-14급)이고, d ≥ 4와 다변량 t는 Genz의 준몬테카를로다. pdf·logpdf·entropy는 κ(cov) ≤ 1e6, d ≤ 100에서 1e-14다(logpdf·entropy는 혼합 기준). 범위는 재고 나서 다시 묻는다. Multinomial의 p는 합이 1에서 sqrt(eps) 안이면 합으로 나누고(sd.random.choice의 R10과 같다), 그 밖이거나 음수·NaN이면 NaN이다. Dirichlet의 x는 K개 성분이고, 음수가 있거나 합이 1에서 sqrt(eps)·K 밖이면 지지집합 밖이다(pdf 0). MultivariateT의 df = inf는 정규의 극한이다. 추가 결정(2026-11-16, 구현 중 측정 뒤): 다변량 t의 d = 2, 3은 QMC 대신 1-D 적분 P = ∫f_χ(r)Φ_d(r·h; R)dr이다(안은 정확한 이변량·삼변량 정규, 바깥은 적응 Gauss–Kronrod). QMC는 카이 변수의 끝점 특이성 때문에 오차가 N⁻¹로만 줄어 점 하나에 280 ms였다. d ≥ 4인 t의 QMC는 역 카이를 점마다 Chebyshev 보간으로 계산한다. 사후에 1e-14로 확인하고, 1e-13 아래의 치우침이 남는다(잰 값 8.3e-17). 보장 영역(2026-11-16, 측정 뒤): pdf·logpdf·entropy는 κ(cov) ≤ 1e9, d ≤ 100(1e6에서 넓혔다: 50자리 참값 대비 κ = 1e9, d = 100, df = 1e9까지 2.7e-16 이하). 정규 cdf의 d = 2는 P ≥ 1e-15이면 모든 ρ, d = 3은 음의 상관이 하나 이하이고 P ≥ 1e-15이거나, P ≥ 1e-10이고 문제 조건수 ≤ 100이다. t cdf의 d = 2, 3은 P ≥ 7e-13(참값이 있는 범위)이다. 기본 atol은 1e-6 그대로다(scipy는 1e-5) |
| T30 행렬 분포 (T10, 2026-11-16) | `Wishart(*, df, scale)`, `InverseWishart(*, df, scale)`, `MatrixNormal(*, mu, rowcov, colcov)`(scipy의 이름. 행렬 정규의 평균만 `mu`다: 매개변수가 같은 이름의 속성이라(T19) `mean()` 메서드와 겹치고, 위치는 mu라는 T18과도 맞는다; 사용자 2026-11-16). 메서드는 scipy처럼 pdf, logpdf, sample, mean, mode, var(원소별 분산), entropy다(행렬 정규는 mode·var 없이 mean과 entropy). (…, p, p, p, p) 공분산 텐서는 두지 않는다. 입력 행렬 x는 아래 삼각만 읽고, 양의 정부호가 아니면 지지집합 밖이다(pdf 0). 배치와 정확도는 T9와 같다(앞 축 브로드캐스트, logpdf·entropy는 double-double로 1e-14 혼합 기준, 범위는 재고 묻는다). df는 실수이고 df > p − 1이어야 한다(특이 Wishart는 NaN). 보장 영역(2026-11-16, 측정 뒤): T9와 같이 κ ≤ 1e9, p ≤ 100(잰 범위는 κ = 1e9, p = 30, df = 1e8에서 2e-16 이하) |
| T31 무한한 자유도 (사용자 결정 2026-11-19) | 자유도가 inf이면 극한 분포다. 모든 메서드(밀도부터 엔트로피, 모멘트, `support`, `sample`)가 그 분포의 값이다. `StudentT(inf)` = `Normal()`, `NoncentralT(inf, nc)` = `Normal(nc, 1)`, `F(dfn, inf)` = χ²(dfn)/dfn, `NoncentralF(dfn, inf, nc)` = 비중심 χ²(dfn, nc)/dfn, `F(inf, dfd)`와 `NoncentralF(inf, dfd, nc)` = `InverseGamma(dfd/2, dfd/2)`. special의 `stdtr`, `fdtr`, `nctdtr`, `ncfdtr`의 inf 극한, `MultivariateT(df=inf)`와 맞춘다. 진짜 분포가 아닌 것(`ChiSquared(inf)`, 비중심 카이제곱의 inf, 두 자유도가 모두 inf인 F, 무한한 nc)은 NaN 그대로다. 구현은 엔진의 극한(§3.1)이다: 분포가 `at_limit`과 메서드별 `limit_in`·`limit_par`를 주면 엔진이 valid보다 먼저 묻고, 값은 다른 분포의 Op(청크 커널을 한 원소 청크로 불러 Python float과 같은 비트)나 그 척도 변환에서 온다 |
| T32 지지집합 (사용자 결정 2026-11-19) | `support()`는 매개변수가 퇴화해도 분포 족의 지지집합이고, `icdf(1)`·`iccdf(0)`은 그 위 끝이다. `Poisson(0)` = (0, inf), `Geometric(1)` = (1, inf), `NegativeBinomial(n, 1)` = (0, inf), `Binomial(10, 0)` = (0, 10), `Bernoulli(0)` = (0, 1). 연속 분포도 같은 규칙이고(감마는 언제나 (0, inf)), scipy의 `support()`·`ppf(1)`과 같다. 전에는 포아송·기하·음이항만 점질량에서 지지집합을 줄였다 |
| T33 음이항 n = 0 (사용자 결정 2026-11-19) | `NegativeBinomial(n=0)`은 0의 점질량이다(이항의 n = 0, 포아송의 μ = 0, special의 `nbdtr(k, 0, p)` = 1과 같다). p = 0과 `Poisson(mu=inf)`는 질량이 무한대로 달아나 진짜 분포가 아니므로 NaN이다(special은 함수의 극한값 `nbdtr(k, n, 0)` = 0, `pdtr(k, inf)` = 0을 낸다). 구현은 T31의 극한 장치로 p = 1의 점질량에 보낸다 |
| T34 다변량·행렬 매개변수 (사용자 결정 2026-11-19) | 다변량·행렬 분포의 매개변수는 `sd.ndarray`(스칼라 매개변수 `df`·`n`은 Python 실수도)이고, 리스트·tuple·numpy 배열은 단변량, special, linalg처럼 `TypeError`다. 입력 x는 실수의 시퀀스도 받는 지금 규칙 그대로다 |
| T35 큰 df의 Wishart (사용자 결정 2026-11-19) | Wishart·역Wishart의 logpdf는 df ≥ 1e9에서 크기 n·p·log n인 항들을 식으로 지운다(감마의 logpdf처럼, log Γ마다 Stirling 꼴; §6.3). p = 1은 감마·역감마와 그 정확도 안에서 같다. 보장 영역은 κ ≤ 1e9, p ≤ 100, df ≤ 1e30, 1e-14 혼합 기준(잰 값 9.4e-16 이하). 그 밖은 남는 반올림의 상한이 1e-14·max(1, \|값\|)를 넘으면 NaN이다. 그 df는 점에 달렸다(2026-11-19 측정): 정확히 나타낸 최빈값·평균(δ = max\|W − I\| ≤ (p+1)/n)은 df = 1e46까지 값, 원소가 반올림된 점은 1e31–3e35부터, 분포에서 뽑은 점은 2.4e32–1e35부터 NaN이다(docs/api/stats.md §3.1) |
| T36 메시지와 큰 int (사용자 결정 2026-11-19) | 분포 생성자의 에러는 모두 클래스 이름으로 시작한다(`Normal(): ...`, `MultivariateNormal(): ...`, 브로드캐스트 에러 `Normal(): shapes could not be broadcast together: ...`; 데이터프레임의 `Series(): `와 같다). Python이 내던 호출 모양의 에러도 같은 꼴로 바꾼다(`Normal(): unexpected keyword argument 'cool'`, `Gamma(): missing 1 required keyword-only argument: 'a'`, `Normal(): takes no positional arguments but 1 was given`; 2026-11-19, `_distribution._keyword_constructor`). 메서드 옵션은 메서드 이름이다(`cdf(): 'max_points' ...`, 클래스 이름 없이). 메서드 입력의 int64 밖 Python int는 special처럼 `OverflowError`(`cdf(): Python int too large to convert to int64`)이고, 매개변수는 지금 규칙(double로 읽고, double 밖이면 `'mu' is out of range`)이다 |
| T37 기술통계의 axis (사용자 결정 2026-11-19) | `axis`는 int, int의 tuple이나 리스트(배열·linalg의 축 인자와 같다)이고, 중복된 축은 배열과 같은 `repeated axis` 메시지다. `rankdata`만 int나 None이다 |
| T38 표본의 모양 (사용자 결정 2026-11-19) | `sample(shape=None, /, *, rng)`: None(기본값)은 매개변수 집합마다 표본 하나(스칼라 매개변수면 Python float, 배열이면 그 모양)다. `shape=()`는 sd.random의 메서드처럼 `ValueError`다(0차원 배열이 없다; 메시지 `sample(): 'shape' must not be () (stardust has no 0-d arrays; None gives one sample)`). `rng.random()`의 None·()과 같은 규칙이다 |
| T39 다변량·행렬의 입력 (사용자 결정 2026-11-19) | 다변량·행렬 분포 메서드(pdf, logpdf, cdf, pmf …)의 입력 x도 `sd.ndarray`만 받는다. 리스트·tuple·numpy 배열은 단변량 메서드처럼 `TypeError`다(`logpdf(): 'x' must be an sd.ndarray, not list`). T34와 함께, 분포는 매개변수와 입력 모두 special·linalg와 같은 형 규칙이다 |
| T40 기술통계의 데이터 (사용자 결정 2026-11-19) | 기술통계 함수는 데이터로 실수의 시퀀스를 계속 받는다(scipy와 같다). 배열 함수(`sd.sum` 등)와 분포의 메서드가 리스트를 받지 않는 것과 일부러 다르다: 손에 든 데이터를 바로 요약하는 함수라 리스트를 그대로 넘기는 쓰임이 흔하다 |
| T41 문자열 옵션의 메시지 (사용자 결정 2026-11-19) | stats의 문자열 옵션(`nan_policy`, `rankdata`·`differential_entropy`의 `method`)은 API.md의 형식이다: `'nan_policy' must be 'propagate', 'omit' or 'raise', not 'x'`(`ValueError`), 문자열이 아니면 `..., not NoneType`(`TypeError`). sd.regression의 `nan_policy`도 같은 날 맞췄다(special에는 그런 옵션이 없다) |
| T43 MultivariateT의 작은 df (사용자 결정 2026-11-19) | df < 2⁻¹⁰²¹이면 정의역 밖이다(모든 메서드 NaN). StudentT의 경계(df의 절반이 정확하지 않다)와 같다. d = 1이 StudentT와 같아진다. 전에는 2⁻¹⁰²²(서브노멀)이었다 (§6.1) |
| T44 비중심 카이제곱의 작은 df (사용자 결정 2026-11-19) | t·F·비중심 t·F는 df < 2⁻¹⁰²¹에서 밀도, cdf 계열, 역함수, median, entropy, 표본이 NaN이다(절반이 정확하지 않다). nc > 0인 `NoncentralChiSquared`는 일부러 둔 예외로 값을 낸다: 0의 점질량을 품은 잘 정의된 포아송 혼합이다. nc = 0은 `ChiSquared`처럼 NaN이다 |
| T46 비중심 카이제곱의 점질량 (사용자 결정 2026-11-19) | T44의 분포(df < 2⁻¹⁰²¹, nc > 0)는 df = 0의 극한이다: 0에 질량 e^(−nc/2), 나머지는 χ²(2j), j ≥ 1의 포아송 혼합. `cdf(0)` = e^(−nc/2), `ccdf(0)`·log 판도 그에 맞추고, x > 0의 밀도·꼬리·역함수는 special의 df = 0 함수(chndtr의 점질량)에서 낸다. 질량 안의 목표는 역함수가 0, `pdf(0)` = inf, `mode()` = 0. special의 `chndtr(df, nc, x)`는 df > 0이라 x = 0에서 0 그대로이고(점질량은 분포의 결정이다), x > 0에서 df = 0의 값에 j = 0 항의 꼬리 e^(−nc/2)(df/2)E₁(x/2)를 더해 쓴다(df < 2⁻⁹⁶⁰; docs/design/special.md §5.16). 전에는 `cdf(0)` = 0, `cdf(1e-300)` = 0.6065, 0 근처의 ccdf·log 판이 NaN이었다 (§3.3) |
| T47 그 분포의 엔트로피 (사용자 결정 2026-11-19) | −inf다. 점질량이 있어 유한한 미분 엔트로피가 없다 |
| T48 카이제곱의 작은 df 표본 (사용자 결정 2026-11-19) | `ChiSquared`와 nc = 0인 `NoncentralChiSquared`의 표본은 df < 2⁻¹⁰²¹이면 NaN이다(T45와 같은 원리: 값 메서드가 NaN). 전에는 0이었다. 극한값을 내는 다른 표본기(`F`, `Beta`, `BetaBinomial`의 서브노멀 모양)는 바꾸지 않았다(값 메서드가 섞여 있다: docs/api/stats.md §2.5) |
| T49 F의 작은 df 표본 (사용자 결정 2026-11-19) | dfn이나 dfd가 2⁻¹⁰²¹ 아래이면 `F`의 표본은 NaN이다(T45·T48과 같은 원리: 값 메서드가 NaN). 전에는 0이나 inf의 극한값이었다. dfd = inf에서 작은 dfn은 `ChiSquared(dfn)`처럼 NaN이고, dfn = inf에서 작은 dfd는 역감마의 극한(값이 있다)이라 표본도 값이다 |
| T50 베타·베타이항의 작은 모양 (사용자 결정 2026-11-19) | a나 b가 2⁻¹⁰²¹ 아래이면 그 모양의 0 극한이다(T44·T46의 원리): 두 점 0과 1(베타이항은 0과 n)이고 P(위) = a/(a + b), P(아래) = b/(a + b)다(둘 다 1 아래이면 2¹⁰⁰⁰배 사본으로 나눠 반올림 하나). 다른 모양이 작지 않으면 한쪽 질량이 아주 작거나(`Beta(a=2⁻¹⁰²², b=2)`의 P(1) = 2⁻¹⁰²³) 0으로 반올림되어(5e-324/2) 한 점의 점질량이 되고, 그때는 한 점의 값이 정확히 나온다(베타이항 엔트로피 0, 분산 0). 2⁻¹⁰²¹을 지날 때 연속이다(a = 2⁻¹⁰²², b = 2⁻¹⁰²¹이면 P(1) = 1/3; 처음 구현은 a만 작으면 0의 점질량이라 여기서 값이 뛰었다). 모든 메서드와 표본이 그 분포의 값이다: 원자의 밀도 inf, 베타의 엔트로피 −inf, 베타이항의 섀넌 엔트로피, 점질량의 왜도·첨도 NaN, 베타의 mode는 베타 자신의 규칙(b ≥ 1이면 0, a ≥ 1이면 1, 그 밖은 NaN; T21), 베타이항의 mode는 큰 질량의 점(같으면 NaN; T25), 지지집합은 족의 것. 다른 모양이 클 때 크기 a인 양(평균 a/(a + b)는 베타 자신과 같다; 분산·ccdf는 두 점의 값이라 베타 자신과 1 + b배쯤 다르다)은 극한의 값이다. 베타이항의 [2⁻¹⁵⁰, 2¹⁵⁰] 모멘트 제한은 이 극한에서만 풀린다. 구현은 엔진의 극한(`stats/dist/two_point.hpp`) |
| T51 역감마의 작은 모양 (사용자 결정 2026-11-19) | `InverseGamma`(와 그 극한인 `F(dfn=inf, dfd)`)의 역함수 넷과 median은 감마의 근이 모든 배율 밖이라 x를 놓지 못하면(서브노멀 a: Y는 e^(−1/a) 아래; scale < 2⁻⁹⁰⁰) 끝에서 정한다: 가장 큰 double에서 아래꼬리가 목표에 못 미치면 inf, 가장 작은 양수에서 이미 넘으면 0. 전에는 NaN이었다(`F(dfn=inf, dfd=1e-310)`의 표본은 inf, cdf(1)은 3.57e-308인데 median·icdf가 NaN). 엔트로피는 a < 2⁻¹⁰⁰⁰에서 닫힌 꼴 1/a + log(s/a) + 1 + γ + O(a)(log Γ(a) = −log a − γa, ψ(a) = −1/a − γ + O(a))이고 Arb와 맞는다(a = 1e-308에서 1e308, 1/a가 넘치면 inf) |
| T45 비중심 F의 표본 (사용자 결정 2026-11-19) | dfn이나 dfd 중 하나라도 2⁻¹⁰²¹ 아래이면 표본이 NaN이다. 값 메서드와 같은 규칙이다(전에는 둘 다 작을 때만 NaN). nc = 0도 같아서, 이 경계 아래에서는 극한값을 내는 `F`의 표본과 다르다. 다른 자유도가 inf이면 극한 분포(T31)의 표본이다 |
| T42 깊은 역함수의 1 − x (사용자 결정 2026-11-19) | 베타 계열의 깊은 역함수(`deep_log_inverse_both`)는 근 x가 1/2를 넘으면 log x를 구한 뒤 log(1 − x)에서 Newton을 다시 해(같은 정지 규칙: 2⁻⁵⁴·max(1, \|log y\|), 100번째에 2⁻⁴⁸ 넘으면 NaN), t와 F의 분위수가 쓰는 1 − x를 직접 낸다. t의 느슨한 구석(df ≥ 1e5, 확률 < 2⁻¹⁰⁰⁰)이 1.4e-14에서 3.2e-16이 되어 범위 밖도 1e-14 안이다(bench/stats/t4/accuracy.md). F와 베타의 행은 비트가 같았다 |

### 1.2 범위
- **단변량 연속 (21)**
  - 기본 12: 정규, t, 카이제곱, F, 감마, 베타, 지수, 로그정규, 균등, 코시, 라플라스, 베이불
  - 극값·수명 6: 로지스틱, Gumbel, 일반화 극값(GEV), 파레토, 레일리, 역감마
  - 비중심 3: 비중심 카이제곱, 비중심 t, 비중심 F
- **단변량 이산 (10)**
  - 기본 5: 이항, 포아송, 기하, 음이항, 초기하
  - 추가 5: 베르누이, 이산 균등, 베타이항, 로그 급수, Zipf
- **다변량 (7)**: 다변량 정규, 다변량 t, 다항, 디리클레, Wishart, 역Wishart, 행렬 정규
- **기술통계**: §5의 11개
- **없는 것**
  - 그 밖의 scipy 분포(약 100개), 무작위 행렬·방향 분포(`ortho_group`, `random_correlation`, von Mises–Fisher 등), 다변량 초기하, 디리클레-다항
  - 분포 맞추기(`fit`), KDE, 분포 변환·혼합·순서통계량
  - 고전형 API(`norm.cdf(x, loc, scale)`)와 그 `loc`·`scale` 규칙
  - 평균의 변형(`gmean` 등), 절삭 통계(`trim_mean` 등)
  - 검정은 `sd.stats.tests`에 따로 (§7)

### 1.3 공통 원칙
- special의 원칙을 그대로 따른다: 입력은 위치로만(`/` 앞), 옵션은 키워드로만(`*` 뒤), float64 실수(bool·int64는 float64로), 같은 입력이면 스레드 수와 무관하게 같은 비트, `f(x)[i] == f(x[i])`, 경고 없음.
- 별칭이 없다. scipy 이름 → 우리 위치는 docs/api/stats.md에 대응표로 둔다(검정이 tests에 있는 것 등).
- **모든 호출은 정해진 시간 안에 끝난다** (사용자 결정 2026-11-17). 합·탐색·반복에는 근거가 있는 상한을 상수로 두고, 상한에 닿으면 부분합이나 마지막 추정이 아니라 NaN을 돌려준다. 보장 영역 안에서는 상한에 닿지 않고 값이 맞아야 한다. 영역 밖에서 구할 수 없는 값은 NaN이다(조용히 틀린 값을 내지 않는다). 루프 카운터를 double로 두지 않는다(`j - i`가 `j`와 같아지는 크기에서 끝나지 않는다). 경위와 상한 목록은 §3.5.
- **참값과 검증은 Arb(python-flint)만 쓴다** (사용자 결정 2026-11-17). mpmath는 쓰지 않는다(느리고, 구적 캐시가 끝없이 자란다). 테스트·bench·tools 어디에도 import가 없고, 테스트 의존성에서도 뺐다.
- 영역 간 일관성(DESIGN.md)이 먼저다. 예: 기술통계의 자유도 인자는 배열의 `var`처럼 `correction`이다(scipy의 `ddof`가 아니다).

### 1.4 층과 파일 (안)
| 위치 | 내용 |
|---|---|
| `csrc/stats/` | 분포마다 C++ 구조체(매개변수 검사, pdf·cdf·…의 원소 함수), 메서드마다 Op(§3) |
| `csrc/bind/stats_bind.cpp` | `stardust._core.stats` 하위 모듈 |
| `src/stardust/stats/` | 분포 클래스(매개변수를 들고 `_core`를 부른다), 기술통계, `tests/` 하위 패키지 |
| `tests/stats/`, `tools/stats_ref.py`, `bench/stats/` | 테스트, Arb 참값, 측정(special과 같은 구조) |

## 2. 분포의 의미론

### 2.1 매개변수와 브로드캐스트
- 매개변수는 Python 스칼라나 `sd.ndarray`다. 분포를 만들 때 그대로 들고 있다가, 메서드를 부를 때 입력과 함께 브로드캐스트한다.
- 결과는 float64다. 모든 피연산자가 Python 스칼라면 Python `float`다.
- 매개변수의 이름과 매개변수화는 분포마다 하나다(T7). 목록은 docs/api/stats.md(초안, 마일스톤 전에 확정).

### 2.2 정의역과 지지집합
- 매개변수가 정의역 밖이면 그 원소의 모든 메서드가 NaN이다(T8).
- 입력이 지지집합 밖이면 `pdf`는 0(`logpdf`는 −inf), `cdf`는 0 또는 1이다. `icdf`는 [0, 1] 밖이면 NaN, 0과 1에서 지지집합의 끝이다.
- scipy와 다른 특수값은 special처럼 마일스톤마다 묻는다(§11). 원칙은 참값·극한이다.

### 2.3 정확도 계약
- 보장 영역에서 참값 대비 상대 1e-14다. 서브노멀 결과는 절대 45·2⁻¹⁰⁷⁴다(special과 같다).
- `ccdf`, `logccdf`는 꼬리에서도 상대 정확도다(1 − cdf로 만들지 않는다). `logcdf`는 작은 cdf에서 log를 직접 계산한다.
- `icdf`·`iccdf`는 special의 역함수와 같은 기준: |Δx/x| ≤ 1e-14·max(1, κ). 이산 분포의 역함수는 후방 오차다(T25, §3.2).
- log 메서드의 계약은 T20이다. 분포마다의 보장 영역은 재고 나서 묻는다(§11).
- 모멘트·엔트로피도 같은 계약이다. 정의되지 않으면(코시의 평균 등) NaN, 무한이면 inf.

### 2.4 scipy와 다르게 동작하는 곳
- 모멘트(T24, 규칙은 docs/api/stats.md §2): t의 df ≤ 1 평균은 NaN(scipy inf), F의 dfd ≤ 2 분산은 NaN(scipy inf), F의 4 < dfd ≤ 6 왜도와 6 < dfd ≤ 8 첨도는 inf(scipy NaN).
- `mode`(T21): 균등과 a, b ≤ 1인 베타(a = 1, b < 1 같은 한쪽 끝 하나는 빼고)는 NaN. scipy는 수치 최적화의 임의의 점(균등 0.5, 베타(1, 1) 0.62)을 낸다.
- `kurtosis`는 초과 첨도(T23). scipy 신형의 기본은 그냥 첨도다.
- 꼬리에서 scipy가 0, 1, inf를 내는 곳에서 참값을 낸다(예: `F(dfn=3, dfd=7).iccdf(1e-30)` = 1.1e9, scipy inf; `Beta(a=2.5, b=0.7).logcdf(1e-200)` = −1151.9, scipy −inf). 같은 점의 비교는 bench/stats/t4/accuracy.md.

## 3. 엔진
- 원소 함수는 분포 구조체의 정적 함수다: `Normal::cdf(x, mu, sigma)`. Op는 (분포, 메서드)마다 템플릿 `DistOp<D, M>`이 만든다(`csrc/stats/engine.hpp`). 이름은 `<분포>_<메서드>`, 입력은 (매개변수..., x) 또는 (매개변수...)이고, `stardust._core.stats`에 special과 같은 n항 디스패처로 묶는다(`csrc/bind/nary_bind.hpp`를 special과 함께 쓴다). 분포 12개 × 메서드 20개(지지집합의 두 끝 포함)를 손으로 쓰지 않는다.
- `DistOp`이 먼저 매개변수의 정의역(`D::valid`)과 NaN 입력을 걸러 NaN을 돌려준다(T8). 그래서 분포의 함수는 유효한 매개변수와 NaN이 아닌 입력만 받는다.
- **극한** (T31, T33): 분포가 valid 밖의 어떤 매개변수에서 다른 분포가 되면(t의 df = inf는 정규, 음이항의 n = 0은 0의 점질량) `at_limit(p...)`, `limit_in<M>(x, p...)`, `limit_par<M>(p...)`를 둔다. `DistOp::f`와 청크 루프가 valid보다 먼저 `at_limit`을 묻고, 그 원소의 값을 `limit_*`에서 받는다. 값은 다른 분포의 메서드(`method_of<T, M>`: T에 청크 커널이 있으면 원소 하나짜리 청크로 불러 T 자신이 Python float에 주는 비트와 같다)이거나 그 척도 변환(`scaled_of<T, M>`: X = Y/c의 밀도·cdf·역함수·모멘트·엔트로피)이다. 표본기는 `at_limit`인 매개변수도 받고, 분포의 표본기가 그 경우를 직접 뽑는다. `standard_deviation`은 분산의 제곱근, `median`은 `icdf(1/2)`가 기본이고, 닫힌 꼴이 있는 분포는 직접 둔다.
- 비용 가중치(병렬 분할), 스레드풀, `f(x)[i] == f(x[i])`, 스레드 수와 무관한 비트는 special과 같다(docs/design/special.md의 3장).
- Python 쪽(`src/stardust/stats/_distribution.py`)은 매개변수를 들고 네이티브 함수를 부르는 얇은 층이다. 매개변수의 형과 브로드캐스트 가능 여부만 만들 때 검사한다.

### 3.1 수치 기법 (`csrc/stats/numeric.hpp`, `csrc/stats/dist/`)
분포를 special 함수의 조합으로 만들면 조합하는 자리에서 자릿수를 잃는다. 반복해서 쓴 해결:
- **표준화 변수를 double-double로**: z = (x − μ)/σ를 float64로 반올림하면 exp(−z²/2)가 그 오차를 z²배로 키운다(z = 30에서 900 ULP). z는 정확한 입력에서 double-double로 만들고, 지수 z²/2 + log(σ√2π)를 double-double로 한 번에 지수화한다. cdf는 반올림한 z.hi에서 계산하고 f(z.hi)·z.lo로 1차 보정한다(2차 항은 z.lo² 크기). 감마의 y = x/scale, 베이불의 c·log(x/scale), 로그정규의 log x도 같다.
- **역함수의 뉴턴 한 단계**: ndtri의 z는 몇 ULP 오차가 있고, μ + σz가 상쇄되면 |σz/x|배, exp(μ + σz)(로그정규)는 |σz|배로 커진다. 꼬리에서 p에 대한 조건수는 1/|z| 정도뿐이다. 그래서 작은 쪽 확률 q의 로그에서 뉴턴 한 단계를 더한다. dz = (log q − log ndtr(z))/(φ/Φ)(z)이고, log ndtr(z)는 −z²/2를 정확하게 둔 double-double이다. 결과 오차는 ULP/|z| 정도다.
- **작은 쪽 꼬리를 직접**: `ccdf`는 1 − cdf가 아니다. 두 쪽 중 작은 쪽을 계산하고 큰 쪽은 1 − 작은 쪽, log는 `log1p(−작은 쪽)`이다.
- **깊은 log 꼬리**: 꼬리가 float64의 정규수 아래로 내려가면 log를 값에서 얻을 수 없다. 정규분포는 `log_ndtr`, 감마는 불완전 감마의 접두(지수가 double-double)와 급수·연분수(`igam_detail::log_tail`), 베타·t·F는 접두의 로그와 BPSER·BFRAC가 곱하는 인수의 로그다(TOMS 708의 함수에 단위 접두를 주어 비율만 얻는다). t의 x = df/(df + t²)마저 1e-270 아래면 주항 x^a/(aB)를 쓴다.
- **깊은 역함수**: 확률이 2^-1000 아래거나 근이 작으면(주항으로 어림한 근이 1e-17 아래) 확률의 로그를 double-double로 두고 log 쪽에서 푼다. 근이 1e-9 아래면 급수 I = x^a/(aB)(1 + cx + …)에서 log x를 닫힌 꼴로 얻는다(1차 보정까지, 감마도 같다). 그보다 크면 log x에서 뉴턴을 돌린다. 척도는 지수에 넣는다(근이 서브노멀이어도 척도를 곱한 결과는 정규수일 수 있다). t와 F는 1 − x를 −expm1(log x)로 만든다(x ≈ 1에서 1 − x는 자릿수를 잃는다).
- **엔트로피와 모멘트의 상쇄**: 감마·베타·t·F의 엔트로피는 크기 z log z인 log Γ와 (z − k)ψ(z) 항들의 합이고 값은 O(log z)다. 그래서 큰 항을 손으로 지운 묶음 G(z), Gc(z)(Stirling 급수)와 로그들의 double-double 합으로 바꿨다. 베이불의 중심 모멘트는 c가 크면 크기 1인 항들이 O(1/c²)로 상쇄된다. 그래서 D_k = log Γ(1 + k/c) − k log Γ(1 + 1/c)를 ζ 급수(선형 항이 정확히 지워진다)나 Stirling 꼴(double-double)로 만들고, 중심 모멘트 비를 E_k = e^{D_k} − 1의 double-double 조합으로 얻는다. 로그정규의 첨도는 t = e^{σ²} − 1의 양항 다항식이다. 베타의 첨도 분자는 double-double이다.
- **정확하지 않은 a − 1**: a < 1/2이면 a − 1.0이 반올림된다. (a − 1)·log y(log y는 745까지)는 a·log y − log y로 쓴다(39 ULP였다).
- special의 `dd::log`는 지수에 맞춘 함수라 정확도가 약 2^-64(측정 최악 1e-23)다. 그래서 큰 인자의 log Γ를 double-double로 직접 빼는 데(t의 log B(a, 1/2), df = 1e14에서 1e-12)에는 쓰지 않고, 큰 항을 손으로 지운 꼴(L(a) = log Γ(a + 1/2) − log Γ(a) − log(a)/2)을 쓴다.

### 3.2 이산 분포의 수치 기법 (`csrc/stats/discrete.hpp`, `csrc/stats/dist/discrete_*.hpp`)
- **분위수**: cdf(k) ≥ p인 가장 작은 정수를 추정값에서 질주(1, 2, 4, …칸)한 뒤 이분한다. p > 1/2이면 정확한 1 − p와 ccdf를 비교하고, 2^-1000 아래는 log 쪽에서 비교한다. 2^53을 넘으면 두 double 사이에 정수가 없으므로 이분을 멈추고, 2^1023을 넘으면 inf다. Zipf처럼 깊은 꼬리의 분위수가 큰 분포는 log ccdf와 log q를 double-double로 비교한다. log q ≈ −729의 ULP는 1.1e-13이고, 분위수는 그 1/(a − 1)배만큼 움직인다(q = 1e-317에서 2.7e-14였다). 계약은 후방 오차다(T25 보충, 사용자 결정 2026-11-16). 목표를 상대 1e-14 안에서 움직였을 때의 정확한 분위수 중 하나면 된다. `icdf`·`iccdf`는 p와 1 − p 중 작은 쪽, `ilog…`는 t 자체에 대해서다. 계단 경계에 붙은 목표에서는 이웃 정수도 맞고, Zipf a ≈ 1처럼 분위수가 1e40을 넘어 계단이 cdf의 분해능보다 촘촘하면 그 폭만큼 허용된다. 참값 파일은 점마다 허용 구간 [kmin, kmax]를 갖는다.
- **닫힌 꼴이 없는 cdf**(초기하, 베타이항): 1/2 아래인 쪽을 직접 더한다. 그쪽의 가장 큰 항(그쪽에 최빈값이 있으면 최빈값)에서 바깥으로 비율 점화식 pmf(j ± 1)/pmf(j)로 더한다. 비율의 네 인수는 정확한 double-double이고 항은 double-double로 이어 가므로(곱과 몫이 각각 약 2⁻¹⁰⁴) 수천 항에도 다시 맞출 필요가 없고, 합은 보상 합이다(`disc::tail_sum_dd`). 전에는 비율을 double로 반올림해서 8항마다 double-double log pmf에서 다시 맞췄다(32항이면 반올림이 1e-14까지 쌓였다). 그 꼴은 모든 항이 필요한 합에서 항이 1e-200 아래로 언더플로할 때의 뒷길로만 남았다. 합의 log는 log pmf와 따로 double-double로 두어 깊은 꼬리(1e-300)에서도 지수화 오차가 없다. 어느 쪽이 작은지는 평균으로 어림하고, 틀리면 반대쪽을 다시 더한다. 최빈값만 보고 쪽을 고르면 퍼진 분포(베타이항 n = 1831, a < 1)에서 1 − 0.992의 상쇄로 4e-13이 났다.
- **초기하의 정확한 정수**: 전체가 120 이하면 pmf와 cdf가 정확한 128비트 개수의 비다. 그래서 cdf가 정확히 1/2인 동점(중앙값)이 정확하다. double-double 경로는 이 값을 0.49999999999999994로 내어 중앙값이 한 칸 어긋났다. 왜도는 닫힌 꼴이라 N = 2K나 N = 2n에서 정확히 0이고, 첨도의 괄호는 N ≤ 2^24에서 정확한 정수다.
- **베타이항의 첨도**: m4/m2² − 3은 0 근처에서 상쇄한다(0.013에서 4e-13). 초과 첨도를 [(b − a)²A/4 − (a + b)²B/2]/(ab·n(s + 2)(s + 3)(s + n))로 고쳐 쓰고(s = a + b, A·B는 양항 다항식; s² − d² = 4ab를 교과서 꼴에 넣어 −3을 지웠다), 괄호를 정확한 s, d에서 double-double로 계산한다. 왜도는 닫힌 꼴이라 a = b에서 정확히 0이다.
- **로그 급수의 꼬리** T(k) = Σ_{j>k} p^j/j = p^{k+1}Φ(p, 1, k + 1): p < 0.9면 급수를 직접(42/(−log p)항) 더한다. 그 위는 Euler–Maclaurin이다. 앞 항을 V = k + 1 + M ≥ 20까지 직접 더하고, 적분은 e^x E₁(x)(x = V·(−log p), E₁의 연분수가 이 비율을 바로 준다), 보정은 B_2j/(2j)!·f^{(2j−1)}(0) 10항이다(h = max(−log p, 1/V) ≤ 0.1이라 2^-60에 닿는다). 처음 쓴 2F1의 로그 꼴(A&S 15.3.10)은 항들이 e^{vw} 크기로 상쇄해 vw ≈ 14에서 음수, 곧 NaN이 났다. 점근 급수는 vw = 30에서 1e-12밖에 안 됐다. log pmf는 log(L/p) = log1p(Σ p^j/(j + 1))로 두어 k = 1에서 log p끼리 지운다(p = 1e-8의 log pmf −p/2가 살아난다).
- **Zipf**: log ζ(a)는 ζ(a) − 1 = ζ(a, 2) < 1/2이면 log1p다(a = 53의 log pmf(1) = −8e-17). k > 1000의 cdf는 1000항의 합에 Hurwitz 차 ζ(a, 1001) − ζ(a, k + 1)를 더한다. 차는 Euler–Maclaurin 꼴의 주항 M^{1−a}(1 − (K/M)^{1−a})/(a − 1)을 expm1로 계산해 a ≈ 1의 상쇄가 없다. 1 − cdf가 1/2을 넘는 logccdf는 log1p(−cdf)다. k^-a는 exp(−a log k)가 아니라 pow다(a = 9에서 a log k의 반올림이 6e-15였다).
- **엔트로피**: 최빈값에서 바깥으로 −pmf·log pmf를 Neumaier 보정 합으로 더한다. 음이항 p = 1.6e-4는 항이 10⁶개라 보정이 없으면 2e-13이었다.
- **참값 쪽 버그**(tools/stats_ref.py): 베르누이의 1 − p는 mpmath의 작업 정밀도에서 p = 1e-123을 잃어 log pmf 참값이 0이었다. 정확한 뺄셈(`fsub(..., exact=True)`)으로 고쳤다. 음이항 엔트로피의 합 범위 40σ는 n < 1에서 (1 − p)^k 꼬리를 잘랐다. 60/p를 더하고 pmf를 점화식으로 계산한다.
- **Arb 참값이 드러낸 것** (2026-11-16, 옛 mpmath 도구가 시간 초과로 버리던 점들):
  - **엔트로피와 중심 적률의 합이 일찍 끝났다.** 전에는 "항이 합의 2⁻⁶⁰ 아래"에서 멈췄다. 그런데 표준편차가 큰 분포에서는 남은 꼬리가 그 항의 수만 배다. 음이항(n = 544, p = 1e-4, σ = 2.3e5)은 8.6σ에서 멈춰 엔트로피가 5e-13 틀렸다. 이제는 이번 항과 앞 항의 비 r로 남은 꼬리를 t·r/(1 − r)로 어림하고, 그것이 합의 2⁻⁶⁰ 아래일 때 멈춘다(`entropy_sum`, `disc_sum::central`). LogSeries 엔트로피(p = 0.9999988, 6.9e-13)도 같은 원인이었다.
  - **음이항 logccdf의 깊은 꼬리**: k = 3.4e7, p = 1.2e-4에서 7e-10이 틀렸다. log I_x(a, b)를 BPSER로 구했는데, x = 1 − p에서 그 급수는 약 40/p항이 필요해 상한 10만 항에서 잘렸다. 이제 a > 15, b ≤ 1이고 1 − x < 1e-3이면 TOMS 708처럼 BGRAT를 쓴다. 값이 언더플로하는 곳이라, 급수 항을 Q로 나눈 배율에서 더하고 log Q를 따로 더하는 로그판이다(`ibeta_detail::log_bgrat`). 1 − x ≥ 1e-3에서는 BPSER를 그대로 둔다. t의 깊은 꼬리(1 − x ~ 2e-3)에서 BPSER가 2e-16, BGRAT가 4e-16이었다.
  - **Zipf 분위수**: 위로 탐색할 때 2¹⁰²³를 넘으면 inf를 돌려줬다. 그래서 참 분위수가 4.9e307이나 1.64e308처럼 그 위의 유한한 double일 때 틀렸다. 이제는 가장 큰 double에서 한 번 시험하고, 거기서도 조건이 거짓일 때만 inf다.

### 3.3 극값·수명·비중심의 수치 기법 (`csrc/stats/dist/extreme.hpp`, `lifetime.hpp`, `noncentral.hpp`)
- **Gumbel과 GEV**: cdf = e^(−t), t = e^E이다. Gumbel은 E = −z이고, GEV(scipy의 c)는 E = log(1 − cz)/c이다. 이 E를 −z − μ(u)/c(u = −cz, μ(u) = u − log(1 + u))로 쓰면 O(cz²)라서 c = 0을 지나며 연속이다. 단 |u| ≥ 1/2에서는 log(1 + u)를 직접 계산한다(u − μ(u)는 u가 크면 상쇄한다. |z| = 1e89에서 틀린 값이 났다). t는 베이불처럼 exp의 반올림을 접어 넣는다. 역함수는 lt = log t에서 z = −expm1(c·lt)/c로 간다. c·lt > 700이면 σz가 유한해도 expm1이 넘치므로 log로 계산한다. expm1(c·lt)는 libm의 expm1에 double-double log(1 + m)로 Newton 한 걸음을 밟아 얻는다(2⁻⁷⁷; 2⁻²⁰ 아래는 6차 테일러). 처음의 double-double 테일러 급수는 원소당 240 ns라 scipy의 0.06배였다. GEV의 왜도·첨도는 W = Y^c(Y ~ Exp(1))가 모양 1/c인 베이불이라서 베이불의 D_k 계산을 x = c로 쓴다(weibull_detail::skewness_x, kurtosis_x). 첨도의 분자는 c² 항이 정확히 지워져 O(c⁴)만 남는다. 그래서 |c| < 1e-4에서는 c = 0 둘레의 테일러 급수(mpmath의 120자리 taylor에서 얻은 계수, 수렴 반지름 1/4)를 쓴다.
- **로지스틱**: 작은 쪽 e^w/(1 + e^w), w = −|z|이다. 역함수 z = log(p/(1 − p))는 p ≤ 1/4에서 log p − log1p(−p), 그 위에서 −log1p((1 − 2p)/p)이다(1 − 2p는 정확하다).
- **파레토·레일리**: 지수 L = b·log(x/s)와 t = y²/2를 double-double로 둔다. log(x/s)는 x ≤ 2s에서 log1p((x − s)/s)이다(x − s는 정확하다). 역함수 x = s·exp(L/b)에서 L은 double-double이다(exp가 L의 오차를 L/b배로 키운다). 레일리의 아주 작은 log p는 x = s·exp((log p + log 2)/2)이다.
- **역감마**: X = s/Y, Y ~ 감마(a)이다. cdf는 Q(a, s/x)이고, 밀도·역함수·엔트로피는 감마의 조각을 쓴다(엔트로피 = 감마의 것 + log s − 2ψ(a)). 감마의 역함수가 내는 y가 double 범위 밖이면 1/s를 지수에 접어 x = 1/((1/s)·y)이다.
- **비중심 cdf와 역함수**: special의 chndtr·nctdtr·ncfdtr와 그 위꼬리 함수(두 꼬리를 직접 더한다)를 쓴다. 역함수는 special의 브래킷·Brent(nc_detail::solve)로 주어진 쪽의 확률을 그대로 푼다. iccdf에서 1 − q를 만들지 않는다. nc = 0이면 중심 분포의 함수를 그대로 부른다(비트까지 같다). 모멘트, median, 표본도 그렇다(2026-11-19: 비중심 F의 모멘트·median은 식이 달라 마지막 비트가, 비중심 카이제곱·F의 표본은 방법이 달라 값이 달랐다). 자유도의 경계도 중심 분포와 같다: t·F처럼 df < 2⁻¹⁰²¹(절반이 정확하지 않다)이면 밀도, cdf 계열, 역함수, median, entropy가 NaN이다. 예외는 nc > 0인 비중심 카이제곱이고, df = 0의 극한(0의 점질량 e^(−nc/2)과 χ²(2j), j ≥ 1의 혼합)으로 계산한다(T44, T46, T47; `NoncentralChiSquared::atom`).
  - **넓은 지수의 꼬리** (2026-11-16, Arb 참값이 드러냄): log 메서드는 e^lp를 만들어서 lp < −745에서 0이나 inf를 냈다. 서브노멀 확률은 절대 정확도만 있어서, 역함수의 log(F/목표)가 거칠었다(1e-7–7e-5). 그래서 special의 혼합 합이 값을 m·2^e(64비트 지수)로도 돌려준다(`nc_detail::chndtr_tail`·`ncfdtr_tail`·`nctdtr_tail`). 여기에 필요한 것은 셋이다. 1e5에서 자르지 않는 exp, 앞 인자의 넓은 판, y ≈ 1e-307에서 T의 비 (s+1)/y가 넘치지 않게 프레임을 옮기는 것. stats의 log 메서드와 역함수는 이 값으로 log(꼬리/목표)를 푼다. special의 double 함수는 같은 합을 감싸서 결과가 그대로다(`nctdtrit`만 21.5 → 9.8 ULP로 좋아졌다).
  - **`tiny_upper`** (작은 인자가 2⁻¹⁰⁰⁰ 아래인 꼬리): 전에는 double lgamma 둘의 차(크기 ~1e4)를 썼고, 결과의 1e-12가 틀렸다(t ≈ −1e264에서 5.1e-12). 이제 가장 큰 항에서 log Γ 비를 double-double로 한 번 구하고, 나머지 항은 double-double 비로 걷는다.
- **비중심 카이제곱·F의 밀도**: 감마·베타 밀도의 포아송 혼합이라 모든 항이 양수다. 가장 큰 항(점화식 비의 이차식 근에서 시작해 비로 맞춘다)에서 바깥으로 더하고, 8항마다 double-double log 항에서 다시 맞춘다. 같은 합에서 E[j]도 얻는다. 최빈값은 각 항의 log 도함수가 j에 대해 선형이라는 데서 나온다. 카이제곱은 y = A(y) − 1, F는 x = (A − 1)·dfd/(dfn(1 + dfd/2))이다(A = E[shape]). 이 고정점을 special의 브래킷으로 푼다. df/2 − 1을 먼저 더해야 df = 2에서 E[j] ≈ 1e-300이 1에 흡수되지 않는다.
- **비중심 t의 밀도**: f(t) = E[S·φ(tS − δ)], S = √(V/df)의 적분이다. 피적분 함수가 어디서나 양수다(급수는 δ와 반대쪽 꼬리에서 두 계열이 상쇄한다). special의 nct_opposite처럼 정류점에서 F*(double-double)와 상쇄 없는 나머지로 나누고 sinh 사상 위의 사다리꼴로 적분한다. 같은 함수가 M_k = ∫s^k φ f_S를 주고, 최빈값은 M_2 무게 아래 E[t·S − δ] = 0의 근을 같은 격자에서 푼다(t·M_3/M_2 = δ로 풀면 δ·eps를 잃었고, 최빈값에서 기울기가 1e-3이라 5e-13이 남았다). |t| > 1e154에서는 t²이 넘치므로 정류점을 w = |t|·s로 푼다.
- **비중심 모멘트**: F = (X/dfn)(dfd/V)와 T = (Z + δ)R, R = √(df/V)로 독립인 두 인자의 곱으로 쓴다. 중심 모멘트를 전개하면 양의 항만 남는다(F의 3·4차는 다항식 전개로 계산한다). t는 Var = δ²y₂ + df/(df − 2), μ₃ = δ(δ²y₃ + 3y₃ + 6r·y₂), μ₄ − 3Var² = δ⁴κ₄(R) + 6δ²(2r·y₃ + y₄ − y₂²) + 3·Var(R²)이다. R의 모멘트비 D_k = log E[R^k] − k·log E[R]는 log Γ 비의 베르누이 다항식 급수(1/x, x = df/2 ≥ 32. 처음의 x ≥ 8에서는 자르기 오차가 2e-13이었다)로 계산해 선형 항을 정확히 지운다(계수는 tools/gen_nct_moments.py가 유리수로 만든다). 그 아래는 double-double log Γ이다.
- **비중심 밀도의 서브노멀 x**: x/2와 dfn·x가 서브노멀에서 반올림돼 1e-5까지 틀렸다. x < 2⁻¹⁰²⁰에서는 항을 log에서 바로 만든다. 모양 a + j도 double로 반올림하면 df ≈ 7e3의 카이제곱 밀도에서 2.4e-13이 틀려서, double-double과 1차 보정으로 둔다. F의 첨도는 m4/m2² − 3이 상쇄하므로, 초과분을 17개 항이 모두 양수인 큐뮬런트 전개로 더한다.
- **비중심 엔트로피**: −∫f·log f를 사다리꼴로 계산하고, 두 추정이 2^-50까지 맞을 때까지 간격을 반으로 줄인다(카이제곱·F는 x = m·e^u, t는 t = mode + s·sinh u). 보장하지 않는 느슨한 계약이고 잰 값을 문서화한다(T26).

### 3.4 빠른 경로: chunk 커널 (2026-11-16–17 최적화; `csrc/stats/engine.hpp`, `chunk.hpp`, `chunk_dd.hpp`)
T4–T10을 마치고 속도를 재 보니 1스레드에서 scipy보다 느린 행이 많았다(T4 96행 중 53행, T6 72행 중 59행). scipy의 원소별 루프는 SIMD이고, 우리 것은 원소마다 스칼라 libm과 double-double을 불렀기 때문이다. 사용자 결정(2026-11-16): 정확도 계약(1e-14) 안에서 빠르게 한다.

- **구조**: 분포가 `static void <메서드>_chunk(const Chunk<NP>& c, double* out)`을 두면 엔진이 배열을 1024원소씩 모아 그 커널에 넘긴다(`chunk_loop_impl`: 매개변수와 x를 연속 버퍼로 모으고, 정의역 밖과 NaN을 채운다). `Chunk`는 원소 수, x, 매개변수 배열, 매개변수가 chunk 안에서 같은지(`same`)를 준다. 커널이 없는 메서드는 전처럼 원소별 스칼라 정의로 간다.
- **빠른 경로와 정밀 경로**: 커널은 plain double(vForce의 exp·log·pow 등)로 값을 내고, 원소마다 그 값의 오차 상한을 2⁻⁵³ 단위로 센다. 상한이 `kFastBudget` = 16(1.8e-15) 안이면 그 값을 쓰고, 넘으면 그 원소만 기존 정밀 정의(스칼라, double-double)로 보낸다(`ck::fallback`, `ckd::retry`). 그래서 계약을 지키는 근거는 정밀 정의와 상한 계산 두 가지뿐이다.
- **오차를 세는 규칙**: 반올림하는 연산 하나가 1단위, libm 호출이 2단위다. vForce는 직접 잰 값을 올림해 쓴다: exp 2.5, log 4.0(`bench/stats/opt/a/vforce_error.cpp`), pow 2.5, tanh 6.5, atanh 4.5, atan 4. 처음에는 연산당 1/2단위로 세었는데 2⁻⁵³ 단위에서는 반올림 하나가 1단위라서 틀린 셈이었고, 전부 다시 세었다.
- **쓸 수 있는 vForce 함수**: 원소의 값이 배열 안 위치와 길이에 무관한 것만 쓴다(exp, log, pow, tanh, atanh, atan을 확인했다). tan은 위치에 따라 비트가 달라져서 쓰지 않는다. 그래야 f(x)[i] == f(x[i])와 스레드 수 무관이 유지된다.
- **스칼라와 배열 매개변수**: 매개변수에서만 나오는 상수(log σ, log Γ 등)는 스칼라일 때와 배열일 때 같은 함수로 계산한다. 다른 함수(예: vForce log와 libm log)를 쓰면 같은 매개변수에서 비트가 달라진다.
- **배열 double-double** (`chunk_dd.hpp`, `ckd`): 표준화 z = (x − μ)/σ와 비 x/s를 스칼라 `num::standardize`·`num::ratio`와 비트까지 같은 분기 없는 루프로, log를 표 기반의 가벼운 double-double(`log_dd`, 2⁻⁵⁹ 보장, 3.8 ns; `dd::log`는 7.3 ns)로 준다. 1 − e^(−t)는 tanh 꼴, log1p는 atanh 꼴이다(libm expm1은 섞인 인자에서 5 ns, tanh 꼴은 1.5 ns).
- **분포별 요점**:
  - 닫힌꼴 12개(정규·로그정규·지수·균등·라플라스·코시·베이불·로지스틱·Gumbel·GEV·파레토·레일리): 87개 메서드에 커널. 정규의 cdf는 special의 `Ndtr::chunk`에 φ·z.lo 보정을 더한다. 베이불·GEV·파레토는 `pow_dd`(double-double 밑의 vForce pow)가 첫 단계이고, 거절된 원소는 double-double 지수 단계로 간다.
  - 감마 계열 여섯(감마·카이제곱·역감마·베타·t·F)의 밀도: log 밀도를 k₁·log x₁ + k₂·log x₂ − lin + c 꼴로 쓰고 log만 빼고 전부 정확하게(two_prod, two_sum, fma) 둔다. log는 vForce(상한이 맞을 때)나 표 기반 log다. 스칼라 pdf·logpdf도 이 커널의 한 원소 호출이다.
  - 감마 계열의 cdf: 2a가 정수이고 a ≤ 30, a ≤ y ≤ 690이면 위꼬리를 유한 합(double-double Horner, exp·erfc)으로 낸다(4–8단위).
  - 베타·F의 cdf와 역함수: 모양 ≤ 6에서 special의 Boost 커널을 쓴다. 이 부분은 엄밀한 상한이 아니라 Arb로 잰 정확도에 기댄다.
  - 이산 분포: 평균 근처의 pmf·logpmf를 Loader의 안장점 꼴로 float64에서 낸다(`saddle::`; pmf 0.5E + 9.9단위, logpmf 8.1단위). 초기하·베타이항의 꼬리 합은 double-double 비 점화식으로 바꿨다(전의 베타이항 cdf는 n ≈ 2.6e5에서 6.8e-14 틀렸다). 분위수 탐색은 넓은 분포에서 Cornish–Fisher 추정에서 시작한다(답은 여전히 탐색이 정한다).
  - **정수점 메모** (`KMemo`, 사용자 결정 2026-11-17: 둔다): (분포, 메서드)마다 스레드별 1024칸 직접 사상 표가 매개변수의 비트와 k로 값을 기억한다. 정밀 정의가 준 값을 그대로 돌려주므로 비트는 같다. 지지집합이 작은 자료(n = 40의 이항)는 원소당 1–2 ns가 된다. 표는 쓰인 (분포, 메서드)마다 스레드당 약 16 KB다. 128원소 뒤 빗나감이 75%를 넘는 chunk는 표를 건너뛴다. 속도표는 이 효과와 한 번 계산하는 비용을 따로 보이려고 좁은·넓은 매개변수 두 묶음을 잰다.
  - 비중심 셋: cdf는 j = 0에서 시작하는 양의 항 합(`nc_detail::lite`; 아래꼬리는 불완전 함수 없이, 위꼬리는 커널 한 번), 밀도는 중심 log 밀도에 초기하 합(₀F₁, ₁F₁)의 log를 더한다. 역함수는 이 합 위의 Newton(log x에서)이고, 예측 오차가 맞지 않으면 전의 브래킷 풀이로 간다. 위꼬리 합은 안에서 부르는 불완전 함수의 잰 오차를 물려받는다(엄밀한 16단위가 아니다).
  - 다변량 정규·t와 행렬 정규의 logpdf: 인자 L의 역 W = L⁻¹를 double-double로 한 번 구해 double로 반올림해 두고, z = W·r을 NEON 내적으로 낸다. 점마다 Higham의 γ 상한을 세고, 넘으면 전의 double-double 전진 대입으로 간다. 이 메서드들만 상한을 40단위(4.4e-15, `ddla::kQuadBudget`)로 둔다(사용자 결정 2026-11-16). BLAS(dgemm)는 합의 순서를 몰라 상한도, 위치 무관도 보장할 수 없어 쓰지 않는다.
  - 표본: 첫 단어를 16개씩 받고 ziggurat·감마의 흔한 경로를 인라인했다(비트 같음). 모양 < 1의 감마 계열은 U^(1/a) = exp(−E/a), 비중심 카이제곱(df ≥ 1)은 χ²(df − 1) + (Z + √nc)²로 바꿨다. 이 둘과 Dirichlet, d ≥ 9의 다변량 정규·t는 같은 시드의 값이 바뀐다(사용자 결정 2026-11-17: 허용).
  - 기술통계: 보정 합을 NEON으로 쓰고(비트 같음), 긴 lane 하나의 정렬은 결정적 samplesort로 풀에서 돈다.
- **역함수의 계약 단위** (사용자 결정 2026-11-17: 허용): 정규·라플라스·로지스틱·Gumbel·GEV의 빠른 역함수는 결과가 0 근처로 상쇄하는 점에서 max(1, κ) 단위의 계약만 지킨다(예: Normal(1.5, 2).icdf(0.2267) = 4.824e-4의 상대 오차 3.3e-13이고, κ = 3120이라 계약 단위로 1.1e-16이다).
- **병렬 분할의 비용** (`csrc/stats/cost_table.hpp`, `tools/gen_stats_cost.py`): 커널이 빨라진 뒤에도 비용 추정이 20 ns로 남아 있어, 작업이 39원소씩 잘려 10스레드가 1스레드보다 느렸다. 공식 속도표의 1스레드 시간에서 (분포, 메서드)별 비용표를 만들어 쓴다.
- **작은 모양의 불완전 베타** (사용자 결정 2026-11-17: 계약을 넘는 것은 지금 고친다): Beta·F의 cdf 계열이 모양 1e-4–1e-2에서 5e-15–1.3e-13이었다. 원인은 a, b ≤ 1일 때 BRATIO(와 그것을 따른 Boost)가 I가 0.9에 이르러도 멱급수로 I를 구해 1 − I를 빼기로 만드는 것이다(급수 오차의 9배). 이제 급수가 I > 1/2를 주면 1 − I를 BRATIO의 다른 두 방법으로 직접 구한다. x < 2⁻¹⁰⁰⁰은 log x를 받는 `ibeta_tiny`로 간다(양의 항만; 전에는 서브노멀 x에서 Boost가 25%까지 틀리고 우리 경로는 NaN이었다). Arb로 잰 21만 점에서 여덟 메서드 모두 4.2e-15 이하다(`bench/stats/opt/f/tiny_shapes.py`). special 쪽은 docs/design/special.md §5.5.
- **이 과정에서 찾은 기존 오류**: Exponential pdf·logpdf가 x/scale이 넘칠 때 NaN, StudentT pdf(0)이 df ≥ 4에서 NaN, Gamma logpdf가 아주 작은 x와 큰 모양에서 NaN, 베타이항 cdf의 6.8e-14, Dirichlet pdf의 3.8e-14(α − 1의 반올림이 log x에 곱해졌다. Wishart의 n − p − 1도 같은 꼴이라 둘 다 two_sum으로 정확히 둔다).

### 3.5 끝나지 않던 호출과 반복의 상한 (2026-11-17)
최적화를 검증하다가 `NoncentralChiSquared(3.5, 2).pdf(1e100)`이 돌아오지 않는 것을 찾았다. 극단 인자 3164개 호출 중 297개가 15초 안에 끝나지 않았다(T5·T6부터 있던 코드). 원인은 셋이다.
- **double 카운터**: 가장 큰 항 j에서 양쪽으로 더하는 급수(`ncd::log_series`)의 아래쪽 루프 조건이 `j - i >= jmin`이었다. x = 1e100이면 j ≈ 1e50이라 `j - i == j`다.
- **상한 없는 탐색**: 역함수의 구간을 넓히는 루프에 횟수 제한이 없었다.
- **수렴만 보는 합**: "항이 합의 2⁻⁶⁰ 아래면 멈춤"은 항 수를 보장하지 않는다. 일부는 1e8 항에서 말없이 멈추고 중간 합을 돌려줬다(`NoncentralChiSquared(3.5, 1e15).cdf`가 어떤 x에서도 2.28e-9).

고친 것(영역 안은 Arb 대조, 영역 밖은 NaN이거나 Arb·극한꼴로 확인한 값):
- **비중심** (`special/ops/noncentral.hpp`, `stats/dist/noncentral.hpp`): 급수 2¹⁸항(`kMaxTerms`), 혼합 합 2¹⁸–2¹⁹항, 밀도 급수 한쪽 2¹⁶항(`kSeriesTerms`), 구간 넓히기 256회, zeroin 200회, 엔트로피는 일 2²⁵ 단위. 영역 안의 최대는 각각 2421, 4645, 10345, 32, 16이다(급수와 혼합 합은 2026-11-19에 영역 안의 참값 점(tests/special/ref, tests/stats/ref)에서 다시 셌다. 혼합 합은 확률이 나가는 길 2300·돌아오는 길 2701걸음, log 꼴과 역함수의 넓은 합이 3805·4645걸음이다. 봉우리가 j* = 1e8에 있는 넓은 꼬리라면 9·√(j*/2) = 6.4e4걸음 나가고 그 두 배를 돌아오며, 이것도 상한 안이다). 카이제곱 밀도는 √(nc·x) ≥ 2¹⁸에서 닫힌 꼴 f = ½e^(−(√y−√μ)²)(y/μ)^((a−1)/2)·ive(a−1, 2√(μy))의 double-double Hankel 전개를 쓴다(Arb 대비 4.6e-16). `std::min(1.0, NaN)`이 1.0을 내던 자리를 모두 바꿨다. 영역 안에서 함께 고친 것: 먼 x의 카이제곱 logccdf가 −inf, cdf(1e100) = 0, df = 0.01의 깊은 log 꼬리가 e^146배, t의 반대쪽 꼬리가 |t| ≈ 1e308에서 NaN. 영역 안 13만 점은 이 수정점 말고 이전과 비트가 같다.
- **이산** (`stats/discrete.hpp`): 분위수 탐색 배증·이분 1100회(`kMaxSearchSteps`), 꼬리 합 2²⁰항, 엔트로피 합 2²⁵항(분산이 2⁴² 넘으면 시작하지 않는다), 중심 모멘트 4096항. 영역 안에서 가장 느린 호출은 `NegativeBinomial(1e4, 1e-4).entropy()` 0.23초다. 함께 고친 것: `Binomial(9, 0.3).mode()`가 3(맞는 값 2; 반올림된 곱의 floor였다), 포아송·기하·음이항·로그 급수가 k = 1e300 이상에서 NaN.
- **NaN이 되는 범위**는 docs/api/stats.md §2.3, §2.4에 있다.
- **테스트**: 끝나지 않던 호출과 영역 밖 호출 몇 개를 시간 제한과 함께 고정했다(`tests/stats/test_discrete.py`, `test_distributions.py`, `tests/special/test_noncentral.py`).
- **나머지 영역도 같은 방식으로 훑었다** (2026-11-17–18; 호출마다 시간 제한을 두고 Arb 값과 대조):
  - **중심 연속 18개** (약 80만 호출): 멈춤은 없었다. 닫힌 꼴 12개는 z나 x/scale이 넘칠 때 NaN이나 틀린 값을 내던 것(5,586점, NaN 49,866점)을 극한값으로 고쳤다. special 기반 6개는 아주 큰·작은 모양에서 값 또는 NaN으로 정리했고, 측정 범위 밖의 역함수는 구한 근을 정방향으로 검증한 뒤에만 낸다. `F(1e100, 13).logpdf(1e300)`은 −5174.3123이 됐다.
  - **다변량·행렬·표본·기술통계** (약 13만 호출): 표본기 77개(Zipf, 초기하, Beta)와 정규·t의 cdf 13개가 멈췄다. `adapt`의 구간 수, QMC 96단계, 이항 역변환 k ≤ 1024, 초기하 최빈값에서 2²⁰ 안, 다항 엔트로피 2²³항으로 묶었다. 특이에 가까운 3차원 t는 한 좌표에 조건을 거는 적분(`tvt_condition`)으로 넘긴다(45–215초 → 0.03–0.12초). 3차원 정규의 조건부 경로가 영역 안(κ = 7.7e7)에서 1.7e-7 틀리던 것도 고쳤다. 기술통계의 모멘트는 거듭제곱이 넘치는 줄을 2의 거듭제곱으로 다시 스케일해 계산한다.
  - **special 전체** (약 126만 호출): 멈춤 1,594개와 프로세스 종료 53개가 0이 됐다. 상한 목록은 docs/design/special.md §1.3.
- **NaN이 되는 범위**는 docs/api/stats.md §2.2, §2.5, §3, §3.1에 있다.
- **적률 (2026-11-18)**: `Beta(5e-324, 5e-324).kurtosis()`가 −4이던 것을 −2로, `F`의 왜도·첨도가 아주 작거나 큰 dfn·dfd에서 틀리던 것을 고쳤다(`F(0.5, 1.7e308).kurtosis()` inf → 24, `F(1e300, 13).skewness()` 0 → 2.4244).
- **감마 계열 역함수의 영역 밖 (2026-11-18)**: 넓힌 격자(1.95만 건, 서브노멀 모양 포함)에서 Arb 기준으로 틀린 값 209 → 0, 정방향 꼬리 746 → 0. 원인 넷을 고쳤다.
  - 근 검증의 허용폭을 검사 대상인 정방향 꼬리에서 구해서, 꼬리가 −inf이면 허용폭이 무한대였다. 이제 꼬리와 목표 가운데 큰 쪽으로 기울기를 구하고 해석적 상한(아래꼬리 a, 위꼬리 y + 1)으로 자른다. Beta·F·StudentT가 쓰는 `num::accept_root`도 같다(그쪽 9,352건은 값 변화 없음).
  - `deep_inverse`가 수렴하지 않아도 마지막 반복값을 냈다. 이제 100번 안에 수렴하지 않으면 NaN이다.
  - special의 `igam_detail::log_tail`이 log(a / tail)의 몫이 서브노멀이 되는 곳(a/y < 2⁻¹⁰²²)에서 틀렸다(`Gamma(a=1e-200).logccdf(1e150)`이 −inf, 참값 −1e150). 그때만 log a − log tail로 나눈다.
  - 서브노멀 모양은 불완전 감마의 조각이 자릿수를 잃어서 기준 모양 a0 = 2⁻⁶⁰⁰에서 계산해 옮긴다(Q(a, y) = (a/a0)·Q(a0, y), 상대 2⁻⁵⁷⁰).
  - a ∈ [1e-6, 1e6]의 역함수 6,706건은 비트까지 같고 T4·T6 정확도표도 같다. 속도는 실제 모듈 21행이 0.993–1.023배다. 확인하지 못한 것: a ≥ 1e20 등 1,497점은 Arb가 수렴하지 않아 정방향 검증만 거쳤다. special의 `gammaincinv`를 서브노멀 a로 직접 부르는 경우는 보지 않았다.
- **남은 것** (영역 밖): QMC는 기본 `max_points`에서 d = 50이 8.9초다(scipy 13.8초; 그대로 둔다, 사용자 2026-11-18).
- **고친 것 (2026-11-18)**: 행렬 정규의 logpdf가 |X − M| ≳ 1e154에서 NaN이었다(제곱이 넘쳐 inf − inf). Wishart는 x ≈ 1.7e308에서 NaN이었다(Cholesky의 곱이 넘침). 이제 2⁴⁰⁰(Wishart는 2⁵⁰⁰) 위의 인자는 2⁻⁶⁰⁰을 곱해 계산하고 로그 행렬식과 이차형식을 되돌린다. 이차형식의 절반이 double을 넘으면 −inf(pdf는 0)이고, 무한한 원소도 −inf다. 상수·동점 데이터의 `differential_entropy`는 NaN이었다(간격 0의 log가 보정 합에서 inf − inf가 됨). 이제 scipy처럼 −inf이고, Correa는 0/0이라 NaN이다. 메서드의 에러는 커널 이름(`normal_cdf():`) 대신 메서드 이름(`cdf():`)을 쓴다(사용자 결정 2026-11-18). 다변량·행렬 분포도 `MultivariateNormal.pdf():`에서 `pdf():`로 맞췄다.

## 4. 표본 (`sample`)
- `X.sample(shape=None, /, *, rng)`(T27, T38): `rng`는 `sd.random`의 Generator다. `shape=None`이면 매개변수 집합마다 하나(스칼라 매개변수면 Python float)이고, 주면 `shape` + 매개변수의 브로드캐스트 모양이다. `shape=()`는 `ValueError`다. 매개변수가 정의역 밖이면 그 원소는 NaN이다.
- **결정성** (`csrc/stats/sample.hpp`): 호출의 원소 i(C 순서)는 생성기 스트림의 원소 e = pos + i다. 첫 단어는 sd.random과 같은 블록 (e/4, 0, 0, 0)에서 받고, 추가 단어는 (e, j, 0, 0), j ≥ 1 블록에서 차례로 받는다(docs/design/random.md §3). 그래서 표본 하나는 (키, e, 매개변수)만으로 정해진다. 스레드 수, 작업 분할, 호출을 나누어 부르는 것과 무관하다. 매개변수가 (3,)이고 shape이 (4,)이면 원소 (i, j)는 스트림의 원소 3i + j다. 기각은 그 원소의 추가 단어에서 다시 뽑는다.
- **역함수 법**이 기본이다. u = (m + ½)·2⁻⁵³(m은 단어의 위 53비트, 0과 1이 나오지 않는다)에서 u ≤ ½이면 icdf(u), 위면 iccdf(1 − u)다. 두 꼬리가 모두 2⁻⁵⁴까지 닿는다. 처음에는 모든 분포를 이것으로 뽑았는데, 감마·베타·비중심·이항·포아송·초기하가 numpy보다 수십–수백 배 느렸다(감마 icdf 200 ns, 초기하 7.6 µs). 그래서 31개 모두 전용 방법을 재서 골랐다(`csrc/stats/sample_dist.hpp`). 모두 분포가 정확한 방법이다(기각이나 조합; 근사는 없다).

| 분포 | 방법 |
|---|---|
| 정규, 로그정규 | Marsaglia–Tsang의 ziggurat(256층, numpy와 같은 구성). 부호는 비트로 붙인다(무작위 비트의 분기는 절반이 예측 실패라 3 ns) |
| 지수, Gumbel, GEV, 베이불, 파레토, 레일리 | 지수의 ziggurat(256층)와 닫힌 꼴 변환. GEV는 z = −expm1(c·log E)/c |
| 균등, 라플라스, 코시, 로지스틱 | 닫힌 꼴 역함수 |
| 감마, 카이제곱 | Marsaglia–Tsang(a ≥ 1). a < 1은 G(a+1)·U^(1/a)를 log로(아주 작은 a의 표본은 0으로 반올림되는 것이 맞다: a = 1e-3이면 질량의 47.5%가 5e-324 아래다) |
| 베타, t, F, 역감마, 비중심 셋 | 감마·정규의 조합. 모양이 1 이상이면 직접, 1보다 작으면 log 감마로. 베타의 두 모양이 1보다 작으면 Jöhnk의 방법(log). 비중심 카이제곱은 df > 1이면 χ²(df − 1) + (Z + √nc)², df = 1이면 (Z + √nc)², df < 1이면 2·G(df/2 + N), N ~ Poisson(nc/2) |
| 포아송 | μ < 10은 0에서부터의 역함수(단어 하나), 그 위는 PTRS(Hörmann 1993) |
| 이항 | np < 10은 역함수, 그 위는 BTRS(Hörmann 1993). p > ½은 n − X |
| 음이항, 베타이항 | 포아송(감마), 이항(베타)의 혼합 |
| 초기하 | 최빈값에서 바깥으로 가는 역함수(pmf가 큰 이웃부터, 점화식; 표준편차 정도의 걸음) |
| 로그 급수 | Kemp의 LK(1981) |
| Zipf | Devroye의 기각(1986). X가 double을 넘으면 inf다(T25). 그때도 판정은 X → ∞ 극한으로 한다(먼저 받아들이기만 하면 a = 1.01에서 inf가 0.125%였다; 참값 0.082%) |
| 베르누이, 이산 균등, 기하 | u < p, Lemire의 정수, ceil(log U / log(1 − p)) |

- **기각 판정의 정확도**: 포아송·이항의 기각은 log pmf 비교를 double-double log pmf로 한다. lgamma 차를 double로 계산하면 μ ~ 1e12에서 그 차의 크기가 1e13이라 판정이 틀린다.
- **속도 기법**: 추가 단어를 쓰는 방법은 이웃한 원소 넷의 첫 추가 블록을 함께 계산한다(블록 하나는 지연 시간 12 ns, 넷을 섞으면 원소당 4.5 ns). 매개변수가 스칼라인 호출은 검사와 준비(감마의 d, c, PTRS의 상수)를 한 번만 한다. Python float 하나는 배열 없이 바로 돌려준다(1.9 → 0.28 µs).
- **잰 값**: bench/stats/t7/(speed.md, quality.md). 요약은 docs/api/stats.md §2.5.

## 5. 기술통계 (`src/stardust/stats/_descriptive.py`, `csrc/bind/stats_desc.cpp`, T8)
| 함수 | 내용 |
|---|---|
| `describe` | 개수, (최소, 최대), 평균, 분산(`correction`), 왜도, 첨도(초과) — `DescribeResult` |
| `skew`, `kurtosis` | m3/m2^(3/2), m4/m2² (− 3); `bias=False`면 수정 계수(n > 2, n > 3) |
| `moment` | (x − center)^order의 평균, center = None이면 평균(중심 적률) |
| `zscore` | (x − 평균)/표준편차 |
| `rankdata` | 순위(average, min, max, dense, ordinal) |
| `sem` | std(correction)/√n |
| `iqr` | 분위수 q[1] − q[0](sd.quantile), `scale`로 나눔('normal'이면 1.349…) |
| `mode` | 최빈값(동점은 가장 작은 값)과 개수 — `ModeResult` |
| `entropy` | 정규화한 pk의 −Σp log p, qk가 있으면 Σp log(p/q) (`sd.special.entr`, `rel_entr`) |
| `differential_entropy` | 표본 간격에서 미분 엔트로피 추정(scipy의 네 방법) |
- **결정**: T28. scipy의 값과 경계를 따른다. 상수 표본의 skew·kurtosis·zscore는 NaN이고, 원소가 하나인 sem은 NaN이다. propagate에서 mode는 NaN을 하나의 값으로 센다. 빈 표본의 mode는 (NaN, 0)이고, 빈 describe는 `ValueError`다. 2m ≥ n인 differential_entropy는 NaN이다.
- **줄(lane)**: 줄여야 할 축을 끝으로 옮겨 (줄 수, n) float64 행렬로 만들고(Python), 커널이 줄마다 계산한다(`desc_moments`, `desc_moment`, `desc_rank`, `desc_mode`, `desc_diffent`). omit은 줄마다 자기 개수로 계산한다. scipy의 omit은 마스크 배열이라 느리다(10⁶개 axis 0에서 50 ms, 우리 2.3 ms).
- **적률의 정확도**: 평균을 hi + lo(보상 합 / n과 잔차의 평균)로 든다. 평균을 double로 반올림하면 평균 1e8에서 최대 7e-9가 틀리고, 그것이 m3에 3·오차·m2로 들어간다(퍼짐 1인 데이터에서 왜도가 1e-8 틀렸다). d = x − hi의 S_k = mean(d^k)를 한 번 훑어 구하고, lo = S1로 m2 = S2 − lo², m3 = S3 − 3 lo S2 + 2 lo³, m4 = S4 − 4 lo S3 + 6 lo² S2 − 3 lo⁴로 바꾼다(세 번째로 훑지 않는다). 합은 모두 Knuth의 TwoSum 보상 합이다(분기가 없어 벡터화된다). zscore도 (x − hi) − lo를 쓴다. 평균 1e8·퍼짐 1의 데이터 2000개에서 유리수로 계산한 정확한 값과 비교하면 skew 3.2e-16, kurtosis·m2 0이다. scipy는 skew 1.2e-8, kurtosis 1.1e-8이다. 테스트는 1e-14로 확인한다.
- **병렬**: 적률은 줄마다 8192개씩 고정된 블록으로 나눠 (줄, 블록)을 풀에서 계산하고, 블록 순서대로 합친다. 블록이 스레드 수와 무관하므로 비트도 무관하고, axis=None의 줄 하나도 병렬이다. 보상 합 여덟 개를 나란히 둬서 의존 사슬을 끊는다. 정렬하는 함수(rankdata, mode, differential_entropy)는 줄 단위로만 병렬이다.
- **잰 값**: bench/stats/t8/speed.md. 요약은 docs/api/stats.md §4.

## 6. 다변량 분포

### 6.1 공통 (`src/stardust/stats/_multivariate.py`, `csrc/bind/stats_mv.cpp`, T9)
- `MultivariateNormal(mu=..., cov=...)`처럼 단변량과 같은 꼴이다(T6). 메서드는 `pdf logpdf`(다항은 `pmf logpmf`), `sample`, `mean`, `covariance`, `entropy`와 정규·t의 `cdf`다. 결정은 T29.
- **정의역**: T29의 규칙(양의 정부호가 아닌 공분산, NaN·inf인 mu 등)에 더해, MultivariateT의 df < 2⁻¹⁰²¹은 정의역 밖이다(T43; StudentT와 같은 경계).
- **배치**: 매개변수의 앞 축을 브로드캐스트해 B개의 배치로 펼치고, 커널은 (B, ...) 매개변수와 (M, d) 점에 점마다의 배치 번호를 받는다. 표본의 결과 모양은 shape + 배치 + (d,)이고, 행 i의 배치는 i mod B다.
- **분해**: 공분산은 double-double 촐레스키(`mv_chol`)로 한 번 분해해 들고 있는다. 거기서 L의 hi·lo, log det = 2Σlog L_jj, 1/L_jj를 얻는다. 피벗이 양수가 아니면 그 배치 원소가 NaN이다(T8). double로 분해하면 L이 A + E(|E| ~ d·eps·|A|)의 인수라서, 이차 형식이 κ·eps만큼 움직인다(scipy의 오차가 κ에 비례하는 이유다).
- **이차 형식**: 앞으로 대입해 z = L⁻¹(x − μ)와 Q = |z|²를 double-double로 구한다. 내적은 Ogita–Rump–Oishi의 Dot2로 한다. fma로 곱을 정확히 구하고 합의 오차를 double 하나에 모으는 방식이라, 이 합에서 double-double 산술만큼 정확하면서 비용은 그 1/3이다. 피벗은 역수를 곱한다. x − μ는 two_sum으로 정확히 뺀다.
- **log 밀도**: 정규는 −Q/2 + c, t는 c − (ν+d)/2·log1p(Q/ν)다. 상수 c(log det, log Γ 차)는 배치마다 한 번 구한다. 처음에는 점마다 log Γ를 두 번 계산해서 d = 3인 t가 7배 느렸다. log Γ((ν+d)/2) − log Γ(ν/2)는 ν/2 ≥ 12에서 Stirling 차로 구한다: (a − ½)log1p(h/a) + h log(a+h) − h + S(a+h) − S(a). log1p(Q/ν)는 Q/ν < 2⁻¹⁰이면 급수로 계산한다(1 + Q/ν가 Q/ν의 아래 비트를 잃는다). pdf는 double-double log에서 exp를 한 번만 한다.
- **엔트로피**: 정규는 (d/2)(1 + log 2π) + log det/2다. t는 ν/2 ≥ 12에서 정규의 값 + [(a+h)Δψ − (a+h−½)log1p(h/a) − ΔS]다(크기 a log a인 항이 없어진다). Δψ는 `digamma_diff_asym`이다. 다항은 −log n! − nΣp log p + Σ E[log X_i!]다. 기댓값은 최빈값에서 바깥으로 double-double로 더하고, 각 pmf는 double-double log Γ에서 exp_full로 구한다. 이항의 log pmf는 앞 인자의 log를 double로 반올림해서, n = 1e5에서 4e-11이 틀렸다. 디리클레는 베타 엔트로피의 G·Gc 묶음을 K개로 넓혔다: ΣG(a_i) + Gc(a0) + Σlog(a_i)/2 + (K−1)log(2π)/2 + (2−K)ψ(a0) − 3 log(a0)/2. Σa_i와 a0이 정확히 지워진다.
- **다항 logpmf**: log n! − Σlog x_i! + Σx_i log p_i − n log(Σp)를 double-double로 계산한다. p를 합으로 나누지 않고 log의 차로 쓴다(나누면 log p_i마다 1 ULP, 전체로 n·eps가 틀린다). log k!은 4096까지 double-double 표(log의 누적합)에서 찾고, 그 위는 log_gamma다.
- **표본**: 벡터 하나가 스트림 원소 하나다(표본 커널의 Words, §4). 정규는 ziggurat d개, t는 거기에 감마(log) 하나를 더한다. 다항은 조건부 이항(BTRS·역함수, §4)이고, 남은 질량은 끝에서부터 더한다. 디리클레는 log 감마를 정규화한다.
- **잰 값**: bench/stats/t9/(accuracy.md, speed.md).
- 행렬 분포는 §6.3(T10).

### 6.2 다변량 정규·t의 cdf (T16, T29; `csrc/stats/mvcdf.hpp`, `csrc/bind/stats_mvcdf.cpp`)
- **꼴**: P(X ≤ x)(모든 좌표의 아래 끝은 −inf)다. 커널 `mv_cdf(cov, x − mu, batch, df, atol, max_points) → (값, 오차)`다. 판정 순서는 다음과 같다.
  1. 행렬이 잘못됐거나 df가 0 이하·NaN이면 NaN이다.
  2. x에 NaN이 있으면 NaN이다(−inf보다 먼저다).
  3. −inf가 있으면 0이다.
  4. +inf인 좌표는 뺀다(주변 분포; t는 같은 df).
  5. 남는 것이 없으면 1이다.
  행렬은 아래 삼각만 읽는다. 오차 출력은 정확한 공식(정규와 t의 남은 좌표 ≤ 3)이면 0이고, 그 밖은 표준오차의 3배다.
- **d = 1**: `special::Ndtr`, t는 `special::Stdtr`.
- **정규 d = 2**: Plackett의 항등식을 각도로 적분한다(Drezner–Wesolowsky; Genz 2004).
  - TVPACK의 고정 규칙은 절대 1e-15까지만 가서, 대신 적응 Gauss–Kronrod 10/21을 쓴다.
  - 적분이 시작값에 더해지는 꼴을 고른다(r ≥ 0이면 0에서, r < 0이면 −1에서). 지수는 음이 아닌 항의 합이다.
  - 극점의 계단은 첫 구간을 기하적으로 잘게 나눠 잡는다. 그렇지 않으면 h ≈ −k에서 2e-8을 잃었다.
- **정규 d = 3**: Plackett/Genz의 경로 적분이고, 상쇄하지 않는 쌍을 고른다. 음의 상관이 둘 이상이면 변수 하나를 뒤집어 본다. 그래도 상쇄하면 조건부 ∫φ(z)Φ₂(·)dz로 바꾸는데, 이 꼴은 상쇄가 전혀 없다. 어디서나 3.4u × 조건수 안이다(u = 2⁻⁵³; 조건수는 입력에 대한 Σ|d log P / d log x|).
- **t d = 2, 3**: 척도 혼합 E[Φ_d(R h)](v = log(x/a)에서 적응 구적, 무게는 Stirling 급수와 상쇄 없는 e^v − 1 − v라 큰 df에서 잃지 않는다. 그대로 쓰면 df = 1e6에서 7e-10이 틀렸다)을 바탕으로 한다. 왼쪽 꼬리는 직교 확률로 닫는다.
  - d = 2(`bvt`): Plackett의 항등식을 R에 대해 섞은 꼴이다. dP/dθ = (1 + q/a)^−a/(2π)가 닫힌 꼴이라, r = 0의 시작값(Φ 둘의 곱의 혼합 하나)에 경로 변수의 적분 하나를 더한다. r < 0에서는 시작값에서 빼는데, r ≤ −1/2이거나 그 뺄셈이 두 비트 넘게 잃으면 r = −1의 값(stdtr로 구한 P(−k ≤ X ≤ h))에서 시작한다. 지수 a log(1 + q/a)는 정확한 t, h, k에서 double-double로 만든다.
  - d = 3(`tvt`): tvn 위의 척도 혼합이다. tvn의 경로가 상쇄하는 곳에서는 노드마다 조건부 tvn이 bvn을 수백 번 부르는데, 모두 합쳐 bvn 20000번(`kTvtBudget`)을 넘으면 값 전체를 X_k에 대한 조건부 적분(`tvt_condition`, 안쪽은 `bvt`)으로 구한다.
  - 처음 정한 QMC는 카이 변수의 끝점 특이성 때문에 오차가 N⁻¹로만 줄어 점 하나에 280 ms였다. 지금은 d = 2가 34 µs, d = 3이 474 µs다(bench/stats/t9/cdf_speed.md).
- **d ≥ 4 (정규, t)**: Genz의 준몬테카를로다.
  - 점마다 Genz–Bretz 재배열을 하고, 변수 분리 적분에 텐트 변환을 쓴다.
  - 점 집합은 fast-CBC rank-1 격자다(Nuyens & Cools 2006; 곱 무게 0.8^j, scipy와 같은 생성 벡터). 처음의 Richtmyer 규칙보다 같은 오차에서 평가가 3–3.5배 적다.
  - 이동은 Philox 키 {0x5eed, n}에서 12개를 받는다. 소수 크기의 단계를 약 1.5배씩 늘리고, 단계들은 분산의 역수로 합친다(Genz의 DKBVRC).
  - 생성 벡터 캐시는 잠금 아래 있고, 512개까지만 둔다.
  - atol에 닿거나 평가 수가 max_points(기본 10⁶ × 적분 변수 수)에 닿으면 멈춘다.
  - 오차 추정은 통계적이다. 잰 값으로 0–7%의 점이 추정을 넘었고, 넘은 폭은 최대 1.9배였다.
- **t의 역 카이** (d ≥ 4의 QMC): df마다 조각별 Chebyshev 표를 쓴다(w ≤ Φ(−2)는 log w, 그 위는 z = Φ⁻¹(w)에서). 조각마다 33점을 1e-14로 확인했고, 촘촘한 표본 512k개에서 최악 1.4e-14다. 호출당 260 ns가 15 ns가 됐다. 같은 격자에서 정확한 역함수와의 치우침은 8.3e-17 이하다. df < 1이면 정확한 gammaincinv를 쓴다.
- **결정성**: 스레드 1·3·10에서, 그리고 점을 혼자 계산하든 묶어 계산하든 같은 비트다.
- **잰 값**: bench/stats/t9/(cdf_accuracy.md, cdf_speed.md). 참값은 tests/stats/ref_mvcdf/(tools/mvcdf_ref.py)이다. 정규(bvn, tvn, 한 인자 d = 4–10)와 d = 2, 3의 t는 Arb의 엄밀한 값이고, 한 인자 t(d = 2–8)는 scipy의 2단 적응 구적(약 1e-11)이다.

### 6.3 행렬 분포 (T10; `src/stardust/stats/_matrix.py`, `csrc/bind/stats_matrix.cpp`, `csrc/stats/dd_linalg.hpp`)
- **결정**: T30. 배치와 매개변수의 촐레스키 분해(`mv_chol`)는 T9와 같다. double-double 선형대수(제곱근, log1p, 촐레스키, 앞으로 대입, L에서 역행렬, log Γ_p)는 `stats/dd_linalg.hpp`에 모아 T9 커널과 함께 쓴다.
- **Wishart** W_p(n, V): log W(X) = (n−p−1)/2 log|X| − tr(V⁻¹X)/2 − (np/2) log 2 − (n/2) log|V| − log Γ_p(n/2)다. 점마다 X의 아래 삼각을 double-double로 분해한다. 이 분해가 양의 정부호 검사이고, 실패하면 지지집합 밖이다. V⁻¹은 배치마다 한 번 L⁻ᵀL⁻¹로 구한다. tr(V⁻¹X)는 아래 삼각의 내적이다.
- **역Wishart** IW_p(n, Ψ): log IW(X) = (n/2) log|Ψ| − (np/2) log 2 − log Γ_p(n/2) − (n+p+1)/2 log|X| − tr(ΨX⁻¹)/2다. tr(ΨX⁻¹) = ‖L_X⁻¹L_Ψ‖²_F를 열마다 앞으로 대입해서 구한다. 반 대각합이 double을 넘으면(서브노멀 피벗이 있는 X 포함) 밀도는 0이다(2026-11-19; 전에는 double-double 합이 inf − inf로 NaN을 냈다. p = 1에서 InverseGamma와 같아졌다).
- **큰 df** (T35, df ≥ 1e9): 위 식의 항들은 크기 n·p·log n이고 최빈값 근처에서 O(p² log n)로 상쇄한다. 그 항의 log는 2⁻⁷⁷–2⁻⁹⁰ 정확도라(special의 double-double log, dd.hpp) df = 1e15에서 6.6e-13이 틀렸다. 그래서 W = L_V⁻¹XL_V⁻ᵀ/n(역Wishart는 L_ΨᵀX⁻¹L_Ψ/n)과 a_j = (n + 1 − j)/2의 Stirling 꼴로 큰 항을 식에서 지운다:
  - W: log W(X) = (n−p−1)/2 log|W| − (n/2)(tr W − p) − p(p+1)/4 log(2n) + R − (p+1)/2 log|V|
  - IW: log IW(X) = (n+p+1)/2 log|W| − (n/2)(tr W − p) + 3p(p+1)/4 log n − p(p+1)/4 log 2 + R − (p+1)/2 log|Ψ|
  - R = −p(p−1)/4 (1 + log π) − (p/2) log 2π − Σ_j[(a_j − ½) log1p((1 − j)/n) + S(a_j)], S(a) = 1/(12a) − 1/(360a³)
  - log|W|는 E = W − I의 촐레스키로 구한다. 피벗 1 + u_j의 u_j = E_jj − Σ L_jk²를 그대로 두고 log1p(u_j)를 더한다. W의 원소의 반올림은 log|W|와 tr W에 같이 들어가 1차에서 지워진다(밀도가 최빈값에서 W에 대해 평평하다). 피벗을 1 근처의 double-double로 반올림하면 2⁻¹⁰⁶이 u가 아니라 1에 대한 것이라, 그것이 n/2배로 남았다(df = 1e30, p = 2에서 1.6e-5, 첫 시도). 남는 반올림의 상한은 2⁻¹⁰⁰·p·(n·δ(1 + pδ) + p(64 + log n)), δ = max|W − I|이고, 이것이 1e-14·max(1, |값|)를 넘으면 NaN이다.
  - 잰 값(Arb 400비트 참값): p = 1–12, κ ≤ 1e9, 최빈값 근처와 분포에서 뽑은 X에서 df = 1e9–1e30까지 9.4e-16 이하(κ ≤ 1e4이면 2.4e-16). p = 1은 감마의 logpdf와 2e-15 안에서 같다(tests/stats/test_review_1119.py).
  - NaN이 시작하는 곳(2026-11-19 측정, p = 1–100, κ = 1–1e9, df 격자 10^(1/8)): 상한의 n·δ 항이 정한다. 최빈값에서 W − I의 크기는 (p + 1)/n이라, 행렬이 정확하면(scale = I인 Wishart의 최빈값·평균) n·δ가 p + 1에 머물러 df = 1e46까지 값이 나온다. 반올림된 원소는 δ ≈ 1e-16을 남기고, 상한이 1e-14·|값|(|값|은 p² log n 정도)을 넘는 n ≈ 1.3e16·|값|/(p·δ)에서 NaN이다: 한 ULP 옮긴 평균 1e34–5.6e34, 일반 scale의 역Wishart 최빈값·평균 1e31(p = 10, κ = 1e4)–3e35. 뽑은 X는 δ ≈ √(2/n)이라 √n이 상한을 정하고 2.4e32–1e35다. 점이 분포의 폭 1/√n보다 멀면 |값| ≈ n·δ²/4가 상한보다 빨리 자라 값이 다시 나온다. 전에 적은 '최빈값 근처에서 1e35부터'는 scale = I, 뽑은 점 하나의 값이었다.
- **엔트로피**: a_j = n/2 + (1 − j)/2, G(a) = log Γ(a) − (a − 1)ψ(a) + a − log(a)/2 − log(2π)/2 = O(1)로 다시 묶는다(디리클레와 같은 방식). 그러면 −Σa_j + np/2 = p(p−1)/4가 정확히 남는다.
  - W: (p+1)/2 log|V| + p(p+1)/2 log 2 + p(p−1)/4 (1 + log π) + Σ_j[G(a_j) + log(a_j)/2 + log(2π)/2 + (p−j)/2 ψ(a_j)]
  - IW: (p+1)/2 log|Ψ| − p(p+1)/2 log 2 + p(p−1)/4 (1 + log π) + Σ_j[G(a_j) + log(a_j)/2 + log(2π)/2 − (1 + (p+j)/2) ψ(a_j)]

  IW의 식은 X = Y⁻¹, Y ~ W(n, Ψ⁻¹)로 변수를 바꿔 얻는다(야코비안 |Y|^−(p+1), E log|Y| = ψ_p(n/2) + p log 2 − log|Ψ|). scipy 1.18의 invwishart.entropy는 이 값과 다르고, 몬테카를로(−E log pdf)는 우리 값과 맞는다.
- **행렬 정규** MN(M, U, V): A = L_U⁻¹(X − M)L_V⁻ᵀ다. X − M은 two_sum으로 정확히 빼고, 열마다 L_U, 행마다 L_V로 앞으로 대입한다. log pdf = −(np/2) log 2π − (p/2) log|U| − (n/2) log|V| − ‖A‖²/2다.
- **표본**: Wishart는 Bartlett 분해다. A는 아래 삼각이고, A_jj = √(2 G((n − j)/2))(j = 0..p−1, 실수 n), 그 아래는 N(0, 1)이다. 결과는 (LA)(LA)ᵀ다. 역Wishart는 X⁻¹ ~ W(n, Ψ⁻¹)이고 W(n, I)가 직교 불변이라 Ψ⁻¹의 아무 제곱근을 써도 되므로, X = (L_Ψ A⁻ᵀ)(L_Ψ A⁻ᵀ)ᵀ다. 행렬 정규는 M + L_U Z L_Vᵀ다. 결과는 정확히 대칭으로 쓴다.
- **잰 값**: bench/stats/t10/(accuracy.md, speed.md). 참값에서 x는 아래 삼각에서 대칭으로 만든다. scipy의 표본은 반올림만큼 비대칭이라, 특이에 가까운 X에서는 그 차이가 X⁻¹를 움직였다(처음 측정에서 3e-8로 보였다).

## 7. `sd.stats.tests` (나중, 이 영역에서 뺌)
- 검정은 이 영역에서 뺐다(사용자 결정 2026-11-18: 나중에 따로 한다).
- 위치와 경계만 정했다(T2, T3). 검정 목록, 결과 객체(`statistic`, `pvalue`, 자유도, 신뢰구간), 대립가설, 정확 검정과 점근 근사, 순열·부트스트랩은 분포·기술통계를 구현한 뒤에 묻는다.
- 그때 필요하면 배열에 `searchsorted`, `bincount` 등을, special에 `kolmogorov`·`smirnov` 등을 더한다.

## 8. 배열과 special에 더하는 것

### 8.1 배열 영역 (T14)
- `cov`, `corrcoef`: numpy처럼 배열 영역에 둔다(자유도 인자는 `correction`).
- `quantile`, `median`
- NaN을 건너뛰는 리덕션: `nanmean nansum nanvar nanstd nanmin nanmax nanquantile nanmedian` 등
- 설계와 API는 docs/design/array.md와 docs/api/array.md에 그 마일스톤에서 더한다.

### 8.2 special (T15)
- 감싸기: 기존 함수의 조합이다. 예: `stdtr(df, t)`는 `betainc`, `chdtr`은 `gammainc`, `pdtr`은 `gammaincc`.
- 새 계산법: 비중심 카이제곱(`chndtr`: Marcum Q, `chndtrix`), 비중심 t(`nctdtr`, `nctdtrit`), 비중심 F(`ncfdtr`, `ncfdtri`), `ndtri_exp`(log 확률에서 정규의 역함수). 보장 영역은 재고 나서 묻는다.
- 설계와 API는 docs/design/special.md와 docs/api/special.md에 더한다.

## 9. 정답지와 측정
- special과 같다: `tools/stats_ref.py`가 Arb(python-flint, `tools/arbref.py`)로 참값을 만들어 `tests/stats/ref/`에 둔다. 공의 반지름이 2⁻¹¹⁰ 아래로 정착한 값만 쓰고, 우리가 자른 급수·적분은 꼬리 상한을 공에 더한다(docs/design/special.md §6). 점마다 시간 제한, 결과는 바로 출력한다. 병렬은 10프로세스, 워커는 500점마다 교체한다. 입력점은 고정한다(사용자 2026-11-16). 파일이 이미 있으면 그 x 배열을 다시 쓰고 참값만 계산한다. 새로 뽑으려면 `--resample`을 준다. 표본을 우리 라이브러리의 icdf(zipf, 로그 급수, 비중심)나 scipy(t)로 뽑는 파일도 있어서, 고정하지 않으면 라이브러리를 고칠 때마다 입력이 바뀌었다.
- 정확도표(보장·느슨한 영역의 최대와 99.9%)와 scipy의 같은 값, 속도(1·10스레드, 스칼라)를 scipy의 고전형과 신형 둘과 비교한다.

## 10. 마일스톤
ROADMAP.md의 "통계 영역"에 있다(T0–T10; T11 검정은 뺐다). 진행 원칙은 special과 같고, 해설과 안내서는 쓰지 않는다(2026-11-15 사용자 결정).

## 11. 마일스톤을 시작하기 전에 물을 것
- **매 마일스톤**: scipy와 다르게 할 특수값(참값·극한), 분포·함수마다의 보장 영역
- **T0 배열 추가**: `quantile`의 방법(numpy의 13가지 중 무엇을 둘지), NaN 리덕션의 목록과 이름, `cov`·`corrcoef`의 인자(`rowvar` 등) — 정했다(2026-11-15): ROADMAP T0, docs/api/array.md §7.9–§7.11
- **T3 sd.random**: random.md §6
- **T4 분포 엔진**: 분포마다의 매개변수 이름, log 메서드의 계약, `mode`, `out=`, 생성자, 첨도 — 정했다(2026-11-15): §1.1 T18–T23. 보장 영역과 특수값은 T24. 이산 분포는 T25
- **T7 표본**: 분포마다 직접 알고리즘인지 역함수 법인지(재서), `sample`에서 `rng`를 생략할 수 있는지
- **T8 기술통계**: 축 인자의 기본값, NaN 처리, 편향 보정의 기본값, 결과 객체
- **T9–T10 다변량**: 특이 공분산, Genz의 기본 정확도, 행렬 분포의 매개변수 규칙
