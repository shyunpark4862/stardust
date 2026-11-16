# 통계 영역 설계 (sd.stats, sd.stats.tests)

> 상태: **T4 구현** (2026-11-15). 큰 결정은 사용자가 정했다(§1.1). 마일스톤마다 시작 전에 물을 것은 §11에 있다.
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
| T10 정확도 | special과 같은 계약: 보장 영역에서 상대 1e-14, 밖은 잰 값을 문서화. 참값은 Arb(엄밀한 구간 산술, 2026-11-16 mpmath에서 바꿈). `ccdf`는 1 − cdf가 아니라 직접 계산해 꼬리도 상대 정확도 (§2.3) |
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
| T24 T4 보장 영역 (2026-11-15) | 닫힌 꼴 7개(정규·지수·균등·라플라스·코시·로그정규·베이불)는 정의역 전체. special 기반은 special의 보장 범위에 맞춘다: 감마 a ≤ 1e6(카이제곱 df ≤ 2e6), 베타 a, b ≤ 1e5, t df ≤ 1e6, F 자유도 ≤ 1e5. 그 밖은 잰 값을 문서화한다. t의 역함수에서 df ≥ 1e5이고 확률이 2^-1000 아래인 구석은 느슨한 영역(1.4e-14, 베타 쪽 BGRAT에 log 꼴이 없다). `entropy`는 `logpdf`처럼 \|오차\| ≤ 1e-14·max(1, \|값\|), 베이불의 왜도·첨도(0을 지난다)도 같다. 모멘트의 특수값은 수학적 규칙(§2.3)을 따르고 scipy와 다른 곳은 §2.4에 둔다 |
| T25 이산 분포 (T5, 2026-11-15) | 메서드는 `pmf`·`logpmf`(T9), 나머지는 연속과 같다. 정수가 아닌 x에서 `pmf`는 0, `cdf`는 floor(x)에서의 값. `icdf(p)`는 cdf(k) ≥ p인 가장 작은 k, `iccdf(p)`는 ccdf(k) ≤ p인 가장 작은 k(scipy의 ppf·isf), `median`은 `icdf(1/2)`. `mode`는 T21을 따른다: 최빈값이 둘 이상이면 NaN(이항의 (n+1)p, 포아송의 μ가 정수일 때, 이산 균등). `Geometric`은 첫 성공까지의 시행 횟수(k ≥ 1, scipy), `NegativeBinomial`은 n번째 성공 전의 실패 횟수(k ≥ 0). `DiscreteUniform(low, high)`는 [low, high)(scipy randint, `sd.random.integers`와 같다). `Hypergeometric(total, successes, draws)`(scipy의 M, n, N). 분위수의 정확도는 후방 오차(2026-11-16): 목표를 상대 1e-14 안에서 움직였을 때의 정확한 분위수 중 하나(`icdf`·`iccdf`는 p와 1 − p 중 작은 쪽, `ilog…`는 t에 대해; §3.2) |
| T26 극값·수명·비중심 (T6, 2026-11-16) | `GEV`의 모양은 scipy의 c(= −ξ, c > 0이면 위가 막힌 Weibull형). `Rayleigh(*, scale=1)`(T18의 양수형 규칙). `Gumbel`은 최댓값형(gumbel_r)만. 비중심의 nc ≥ 0(nc = 0이면 중심 분포). 비중심 셋의 `mode`는 pdf′ = 0의 근으로 1e-14 계약, `entropy`는 적응 구적이고 잰 정확도를 느슨한 한계로 문서화. 비중심의 보장 영역은 special(§14)의 잰 범위(확률 1e-300까지; 더 깊은 log 꼬리는 보장하지 않는다), 비중심 t의 모멘트는 큰 df에서 1/df 급수로 1e-14(2026-11-16). GEV의 왜도·첨도는 베이불과 같은 혼합 기준(T24; W = Y^c가 베이불이다) |

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
- `DistOp`이 먼저 매개변수의 정의역(`D::valid`)과 NaN 입력을 걸러 NaN을 돌려준다(T8). 그래서 분포의 함수는 유효한 매개변수와 NaN이 아닌 입력만 받는다. `standard_deviation`은 분산의 제곱근, `median`은 `icdf(1/2)`가 기본이고, 닫힌 꼴이 있는 분포는 직접 둔다.
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
- **닫힌 꼴이 없는 cdf**(초기하, 베타이항): 1/2 아래인 쪽을 직접 더한다. 그쪽의 가장 큰 항(그쪽에 최빈값이 있으면 최빈값)에서 바깥으로 비율 점화식 pmf(j ± 1)/pmf(j)로 더하고, 8항마다 double-double log pmf에서 다시 맞춘다(32항이면 비율의 반올림이 1e-14까지 쌓였다). 합의 log는 log pmf와 따로 double-double로 두어 깊은 꼬리(1e-300)에서도 지수화 오차가 없다. 어느 쪽이 작은지는 평균으로 어림하고, 틀리면 반대쪽을 다시 더한다. 최빈값만 보고 쪽을 고르면 퍼진 분포(베타이항 n = 1831, a < 1)에서 1 − 0.992의 상쇄로 4e-13이 났다.
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
- **비중심 cdf와 역함수**: special의 chndtr·nctdtr·ncfdtr와 그 위꼬리 함수(두 꼬리를 직접 더한다)를 쓴다. 역함수는 special의 브래킷·Brent(nc_detail::solve)로 주어진 쪽의 확률을 그대로 푼다. iccdf에서 1 − q를 만들지 않는다. nc = 0이면 중심 분포의 함수를 그대로 부른다(비트까지 같다).
  - **넓은 지수의 꼬리** (2026-11-16, Arb 참값이 드러냄): log 메서드는 e^lp를 만들어서 lp < −745에서 0이나 inf를 냈다. 서브노멀 확률은 절대 정확도만 있어서, 역함수의 log(F/목표)가 거칠었다(1e-7–7e-5). 그래서 special의 혼합 합이 값을 m·2^e(64비트 지수)로도 돌려준다(`nc_detail::chndtr_tail`·`ncfdtr_tail`·`nctdtr_tail`). 여기에 필요한 것은 셋이다. 1e5에서 자르지 않는 exp, 앞 인자의 넓은 판, y ≈ 1e-307에서 T의 비 (s+1)/y가 넘치지 않게 프레임을 옮기는 것. stats의 log 메서드와 역함수는 이 값으로 log(꼬리/목표)를 푼다. special의 double 함수는 같은 합을 감싸서 결과가 그대로다(`nctdtrit`만 21.5 → 9.8 ULP로 좋아졌다).
  - **`tiny_upper`** (작은 인자가 2⁻¹⁰⁰⁰ 아래인 꼬리): 전에는 double lgamma 둘의 차(크기 ~1e4)를 썼고, 결과의 1e-12가 틀렸다(t ≈ −1e264에서 5.1e-12). 이제 가장 큰 항에서 log Γ 비를 double-double로 한 번 구하고, 나머지 항은 double-double 비로 걷는다.
- **비중심 카이제곱·F의 밀도**: 감마·베타 밀도의 포아송 혼합이라 모든 항이 양수다. 가장 큰 항(점화식 비의 이차식 근에서 시작해 비로 맞춘다)에서 바깥으로 더하고, 8항마다 double-double log 항에서 다시 맞춘다. 같은 합에서 E[j]도 얻는다. 최빈값은 각 항의 log 도함수가 j에 대해 선형이라는 데서 나온다. 카이제곱은 y = A(y) − 1, F는 x = (A − 1)·dfd/(dfn(1 + dfd/2))이다(A = E[shape]). 이 고정점을 special의 브래킷으로 푼다. df/2 − 1을 먼저 더해야 df = 2에서 E[j] ≈ 1e-300이 1에 흡수되지 않는다.
- **비중심 t의 밀도**: f(t) = E[S·φ(tS − δ)], S = √(V/df)의 적분이다. 피적분 함수가 어디서나 양수다(급수는 δ와 반대쪽 꼬리에서 두 계열이 상쇄한다). special의 nct_opposite처럼 정류점에서 F*(double-double)와 상쇄 없는 나머지로 나누고 sinh 사상 위의 사다리꼴로 적분한다. 같은 함수가 M_k = ∫s^k φ f_S를 주고, 최빈값은 M_2 무게 아래 E[t·S − δ] = 0의 근을 같은 격자에서 푼다(t·M_3/M_2 = δ로 풀면 δ·eps를 잃었고, 최빈값에서 기울기가 1e-3이라 5e-13이 남았다). |t| > 1e154에서는 t²이 넘치므로 정류점을 w = |t|·s로 푼다.
- **비중심 모멘트**: F = (X/dfn)(dfd/V)와 T = (Z + δ)R, R = √(df/V)로 독립인 두 인자의 곱으로 쓴다. 중심 모멘트를 전개하면 양의 항만 남는다(F의 3·4차는 다항식 전개로 계산한다). t는 Var = δ²y₂ + df/(df − 2), μ₃ = δ(δ²y₃ + 3y₃ + 6r·y₂), μ₄ − 3Var² = δ⁴κ₄(R) + 6δ²(2r·y₃ + y₄ − y₂²) + 3·Var(R²)이다. R의 모멘트비 D_k = log E[R^k] − k·log E[R]는 log Γ 비의 베르누이 다항식 급수(1/x, x = df/2 ≥ 32. 처음의 x ≥ 8에서는 자르기 오차가 2e-13이었다)로 계산해 선형 항을 정확히 지운다(계수는 tools/gen_nct_moments.py가 유리수로 만든다). 그 아래는 double-double log Γ이다.
- **비중심 밀도의 서브노멀 x**: x/2와 dfn·x가 서브노멀에서 반올림돼 1e-5까지 틀렸다. x < 2⁻¹⁰²⁰에서는 항을 log에서 바로 만든다. 모양 a + j도 double로 반올림하면 df ≈ 7e3의 카이제곱 밀도에서 2.4e-13이 틀려서, double-double과 1차 보정으로 둔다. F의 첨도는 m4/m2² − 3이 상쇄하므로, 초과분을 17개 항이 모두 양수인 큐뮬런트 전개로 더한다.
- **비중심 엔트로피**: −∫f·log f를 사다리꼴로 계산하고, 두 추정이 2^-50까지 맞을 때까지 간격을 반으로 줄인다(카이제곱·F는 x = m·e^u, t는 t = mode + s·sinh u). 보장하지 않는 느슨한 계약이고 잰 값을 문서화한다(T26).

## 4. 표본 (`sample`)
- `X.sample(shape, rng=rng)`: `rng`는 `sd.random`의 Generator다. 원소마다 생성기의 독립 부분 수열을 받아(random.md §3) 스레드 수와 무관하게 같은 값이 나온다.
- 분포마다 방법을 고른다. 후보(마일스톤에서 재서 정한다):
  - 역함수 법: `icdf(u)`. 어느 분포에나 되고 정확하지만 느릴 수 있다.
  - 정규: ziggurat(Marsaglia–Tsang). 감마: Marsaglia–Tsang. 이항: BTPE(Kachitvichyanukul–Schmeiser). 포아송: PTRS(Hörmann). 베타·t·카이제곱·F·디리클레·Wishart: 감마·정규의 조합.
- 기각 표본은 원소 안에서 다시 뽑는다(원소의 부분 수열에서). 그래서 원소마다 쓰는 난수의 개수가 달라도 결정성이 깨지지 않는다.

## 5. 기술통계
| 함수 | 내용 |
|---|---|
| `describe` | 개수, 최소·최대, 평균, 분산, 왜도, 첨도 (결과 namedtuple) |
| `skew`, `kurtosis` | 표본 왜도·첨도 (편향 보정 옵션) |
| `moment` | 중심 적률 |
| `zscore` | 표준화 |
| `rankdata` | 순위 (동점: 평균·최소·최대·조밀·순서) |
| `sem` | 평균의 표준오차 |
| `iqr` | 사분위 범위 (배열의 `quantile`을 쓴다) |
| `mode` | 최빈값과 그 개수 |
| `entropy` | 확률 벡터의 엔트로피, 두 벡터면 KL 발산 (`sd.special.entr`, `rel_entr`) |
| `differential_entropy` | 표본에서 미분 엔트로피 추정 |
- 축 인자의 기본값(scipy는 `axis=0`, 배열 영역은 `axis=None`), NaN 처리(scipy의 `nan_policy`), 결과 객체는 마일스톤 전에 묻는다(§11).

## 6. 다변량 분포

### 6.1 공통
- `MultivariateNormal(mu=..., cov=...)`처럼 단변량과 같은 꼴이다(T6). 메서드는 `pdf logpdf`(다항은 `pmf logpmf`), `sample`, `mean`, `covariance`, `entropy`와 정규·t의 `cdf`.
- 공분산은 `sd.linalg.cholesky`로 분해해 들고 있는다. 양의 정부호가 아니면 NaN(T8). 반정부호(특이) 공분산을 받을지는 마일스톤 전에 묻는다.
- 행렬 분포(Wishart, 역Wishart, 행렬 정규)의 매개변수 규칙은 그 마일스톤에서 정한다.

### 6.2 다변량 정규·t의 cdf (T16)
- Genz의 방법(변수 바꿈과 격자 준몬테카를로)을 고정 시드로 돌린다. 같은 입력이면 같은 값이고, 오차 추정을 함께 돌려준다.
- 1e-14 계약 밖이다. 기본 정확도와 잰 오차를 문서화한다. 2차원(Owen의 T, Drezner–Wesolowsky)·3차원은 정확한 공식을 쓸 수 있다(그 마일스톤에서 정한다).

## 7. `sd.stats.tests` (나중)
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
ROADMAP.md의 "통계 영역"에 있다(T0–T11). 진행 원칙은 special과 같고, 해설과 안내서는 쓰지 않는다(2026-11-15 사용자 결정).

## 11. 마일스톤을 시작하기 전에 물을 것
- **매 마일스톤**: scipy와 다르게 할 특수값(참값·극한), 분포·함수마다의 보장 영역
- **T0 배열 추가**: `quantile`의 방법(numpy의 13가지 중 무엇을 둘지), NaN 리덕션의 목록과 이름, `cov`·`corrcoef`의 인자(`rowvar` 등) — 정했다(2026-11-15): ROADMAP T0, docs/api/array.md §7.9–§7.11
- **T3 sd.random**: random.md §6
- **T4 분포 엔진**: 분포마다의 매개변수 이름, log 메서드의 계약, `mode`, `out=`, 생성자, 첨도 — 정했다(2026-11-15): §1.1 T18–T23. 보장 영역과 특수값은 T24. 이산 분포는 T25
- **T7 표본**: 분포마다 직접 알고리즘인지 역함수 법인지(재서), `sample`에서 `rng`를 생략할 수 있는지
- **T8 기술통계**: 축 인자의 기본값, NaN 처리, 편향 보정의 기본값, 결과 객체
- **T9–T10 다변량**: 특이 공분산, Genz의 기본 정확도, 행렬 분포의 매개변수 규칙
