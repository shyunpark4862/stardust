# 통계 영역 설계 (sd.stats, sd.stats.tests)

> 상태: **초안** (2026-11-15). 큰 결정은 사용자가 정했다(§1.1). 마일스톤마다 시작 전에 물을 것은 §11에 있다.
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
| T10 정확도 | special과 같은 계약: 보장 영역에서 상대 1e-14, 밖은 잰 값을 문서화. 참값은 mpmath. `ccdf`는 1 − cdf가 아니라 직접 계산해 꼬리도 상대 정확도 (§2.3) |
| T11 구현 위치 | C++ `_core`의 원소별 Op(special과 같은 n항 디스패처·스레드풀). 분포 객체, 검정·기술통계의 조립은 Python (§3) |
| T12 표본 | 분포별 표본은 `sd.random`이 아니라 각 분포의 `X.sample(shape, rng=rng)`. 분포마다 직접 알고리즘 또는 역함수 법 (§4) |
| T13 기술통계 | `describe skew kurtosis moment zscore rankdata sem iqr mode entropy differential_entropy` (§5) |
| T14 배열 추가 | `cov corrcoef quantile median`과 NaN을 건너뛰는 리덕션 (§8.1) |
| T15 special 추가 | 감싸기(`stdtr stdtrit chdtr chdtrc chdtri fdtr fdtrc fdtri pdtr pdtrc bdtr bdtrc nbdtr nbdtrc`)와 새 계산법(`chndtr chndtrix nctdtr nctdtrit ncfdtr ncfdtri ndtri_exp`)을 scipy 이름으로 `sd.special`에 공개, 같은 계약 (§8.2) |
| T16 다변량 cdf | 다변량 정규·t의 cdf는 느슨한 계약으로 넣는다: Genz 방법을 고정 시드로 결정적으로, 오차 추정을 함께 돌려주고 잰 정확도를 문서화한다. 2·3차원은 정확한 공식을 쓸 수 있다 (§6.2) |
| T17 난수 | 별도 모듈 `sd.random` (random.md) |

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
| `tests/stats/`, `tools/stats_ref.py`, `bench/stats/` | 테스트, mpmath 참값, 측정(special과 같은 구조) |

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
- `icdf`·`iccdf`는 special의 역함수와 같은 기준: |Δx/x| ≤ 1e-14·max(1, κ).
- log 메서드의 계약(상대인지 |오차| ≤ 1e-14·max(1, |값|)인지), 분포마다의 보장 영역은 마일스톤 전에 묻는다(§11).
- 모멘트·엔트로피도 같은 계약이다. 정의되지 않으면(코시의 평균 등) NaN, 무한이면 inf.

### 2.4 scipy와 다르게 동작하는 곳
- (마일스톤마다 채운다)

## 3. 엔진
- 원소 함수는 분포 구조체의 정적 함수다: `Normal::cdf(x, mu, sigma)`. Op는 (분포, 메서드)마다 템플릿으로 만든다: `DistOp<Normal, Cdf>`(nin = 1 + 매개변수 수, nout = 1). 분포 31개 × 메서드 약 13개라 손으로 쓰지 않는다.
- 디스패처, 비용 가중치, 스레드풀은 special과 같다(docs/design/special.md의 3장). 피연산자는 입력과 출력을 합쳐 최대 5개(`kMaxOperands`)라 매개변수는 셋까지 된다. 가장 많은 것은 셋(비중심 F, 초기하, 베타이항, GEV)이다.
- 대부분의 cdf·icdf는 special 함수의 조합이다: t → `stdtr`, 감마 → `gammainc`, 이항 → `bdtr` 등. 조합이 계약을 지키는지는 참값으로 잰다.
- 모멘트·엔트로피·median·mode·support는 매개변수만의 함수다(Op의 nin = 매개변수 수).

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
- special과 같다: `tools/stats_ref.py`가 mpmath로 참값을 만들어 `tests/stats/ref/`에 둔다. 점마다 시간 제한, 결과는 바로 출력한다.
- 정확도표(보장·느슨한 영역의 최대와 99.9%)와 scipy의 같은 값, 속도(1·10스레드, 스칼라)를 scipy의 고전형과 신형 둘과 비교한다.

## 10. 마일스톤
ROADMAP.md의 "통계 영역"에 있다(T0–T11). 진행 원칙은 special과 같고, 해설과 안내서는 쓰지 않는다(2026-11-15 사용자 결정).

## 11. 마일스톤을 시작하기 전에 물을 것
- **매 마일스톤**: scipy와 다르게 할 특수값(참값·극한), 분포·함수마다의 보장 영역
- **T0 배열 추가**: `quantile`의 방법(numpy의 13가지 중 무엇을 둘지), NaN 리덕션의 목록과 이름, `cov`·`corrcoef`의 인자(`rowvar` 등) — 정했다(2026-11-15): ROADMAP T0, docs/api/array.md §7.9–§7.11
- **T3 sd.random**: random.md §6
- **T4 분포 엔진**: 분포마다의 매개변수 이름(docs/api/stats.md의 초안), log 메서드의 계약, 이산 분포의 `median`·`mode` 정의, 메서드 결과의 모양
- **T7 표본**: 분포마다 직접 알고리즘인지 역함수 법인지(재서), `sample`에서 `rng`를 생략할 수 있는지
- **T8 기술통계**: 축 인자의 기본값, NaN 처리, 편향 보정의 기본값, 결과 객체
- **T9–T10 다변량**: 특이 공분산, Genz의 기본 정확도, 행렬 분포의 매개변수 규칙
