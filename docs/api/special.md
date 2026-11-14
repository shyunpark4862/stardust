# 특수함수 API (sd.special)

> 상태: **확정** (2026-11-12). S3까지 구현했다(§2의 8개, §3의 8개, §5의 7개, §7의 `zeta`). 설계와 결정 근거는 [docs/design/special.md](../design/special.md)에 있다. 보장 영역과 특수값 표는 각 마일스톤에서 측정한 뒤 채운다.

`import stardust as sd`의 하위 모듈 `sd.special`이 특수함수 API다 (scipy.special의 부분집합).

- 라이브러리 전체의 공통 규칙(dtype, 에러 종류, 스레드)은 루트 [API.md](../../API.md)에 있다. 원소별 함수의 피연산자, `out=`, 0차원 결과 규칙은 [docs/api/array.md](array.md) §1.2, §6.2를 그대로 따른다.
- 절 번호(`§N`, `§N.M`)는 코드 주석이 인용한다. 번호를 바꾸지 않는다.
- 언어 표시가 없는 코드 블록은 `tests/special/test_api_surface.py`가 읽는다. 공개 이름과 시그니처만 쓰고, 한 줄에 함수 하나다.

---

## §1. 특수함수 공통 규칙

### §1.1 시그니처와 이름
- 함수 이름은 scipy.special의 이름이다. Array API에는 특수함수가 없다.
- **입력은 모두 positional-only(`/` 앞)다.** 옵션(`out`, `axis`, `keepdims`, `k`)은 keyword-only(`*` 뒤)다.
- sd에 이미 있는 함수(`expm1`, `log1p`, `exp2`, `cbrt`, `round` 등)는 여기에 다시 두지 않는다. 별칭(`psi`, `jn`)도 없다.

### §1.2 피연산자와 결과
- 피연산자는 `sd.ndarray`와 Python 스칼라다 (docs/api/array.md §6.2). 서로 브로드캐스트한다.
- bool과 int64는 float64로 계산한다. 결과는 float64다.
- complex128은 `TypeError`다. 예외는 §12의 함수다.
- 피연산자가 모두 Python 스칼라이면 결과는 Python `float`다 (docs/api/array.md §1.2).
- `out=`의 규칙은 원소별 함수와 같다 (docs/api/array.md §6.2). 출력이 여럿인 `airy`에는 `out=`이 없다.

### §1.3 값 때문에 실패하지 않는다
- 정의역 밖은 NaN, 극점은 ±inf다. 에러도 경고도 없다 (API.md §2).
- NaN 입력은 NaN을 낸다. 예외는 함수별 절에 적는다 (예: `xlogy(0, y)`).
- 결과가 float64 범위를 넘으면 ±inf, 아래로 넘으면 0이다.

### §1.4 정확도
- 함수마다 **보장 영역**을 둔다. 그 안에서 고정밀 값 대비 상대오차가 1e-14 이하다.
- 보장 영역 밖(정의역 안)도 값을 계산한다. 잰 최대 오차를 함수별 절에 적는다.
- 진동 함수(`j0 j1 y0 y1 jv yv jve yve`, x < 0의 `airy`)는 상대오차 대신 `|오차| ≤ 1e-14 × 포락선`을 보장한다. 영점 근처에서는 상대오차를 정의할 수 없기 때문이다. 포락선의 식은 함수별 절에 적는다.
- 같은 입력이면 위치, stride, 스레드 수와 상관없이 같은 비트가 나온다 (`f(x)[i] == f(x[i])`, API.md §3).
- 일부 함수는 구간에 따라 Boost.Math를 부른다(S10, docs/design/special.md §5.14). 그 구간도 같은 계약을 지키고, 특수값은 이 문서의 규칙을 따른다.

### §1.5 scipy와 다른 곳
docs/design/special.md §2.4에 모았다.

---

## §2. 오차함수
```
erf(x, /, *, out=None)
erfc(x, /, *, out=None)
erfcx(x, /, *, out=None)
erfinv(y, /, *, out=None)
erfcinv(z, /, *, out=None)
ndtr(x, /, *, out=None)
log_ndtr(x, /, *, out=None)
ndtri(p, /, *, out=None)
```
- `erfcx(x) = exp(x²)·erfc(x)`
- `ndtr`는 표준정규분포의 cdf, `ndtri`는 그 역함수다. `ndtri(0)`은 `-inf`, `ndtri(1)`은 `inf`, [0, 1] 밖은 NaN
- `erfinv`는 [−1, 1], `erfcinv`는 [0, 2]에서 정의된다.
- **보장 영역**: 여덟 함수 모두 정의역 전체다 (사용자 결정 2026-11-13). `erf erfc erfcx ndtr log_ndtr`는 유한한 모든 x, `erfinv`는 −1 < y < 1, `erfcinv`는 0 < z < 2, `ndtri`는 0 < p < 1(서브노멀 포함)이다. 결과가 서브노멀이면 docs/design/special.md §2.3의 절대오차 규칙을 따른다.
- **잰 정확도** (mpmath 대비 최대 ULP, bench/special/s2/accuracy.md): `erf` 2.3, `erfc` 7.8, `erfcx` 6.1, `ndtr` 6.7, `log_ndtr` 7.0, `ndtri` 5.4, `erfinv` 4.6, `erfcinv` 4.5
- **특수값**: ±0, ±inf, NaN, 경계값에서 scipy와 비트까지 같다. 다른 곳은 아래 표의 굵은 줄이다 (docs/design/special.md §2.4).

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `erf(±0)`, `erf(±inf)` | ±0, ±1 | |
  | `erfc(+inf)`, `erfc(-inf)` | 0, 2 | |
  | `erfc(x)`, 26.55 < x < 27.23 | 서브노멀 값 | **scipy는 x > 26.7에서 0** |
  | `erfcx(+inf)`, `erfcx(-inf)` | 0, inf | `erfcx(x)`는 x < −26.63에서 inf |
  | `ndtr(+inf)`, `ndtr(-inf)` | 1, 0 | |
  | `ndtr(x)`, −38.47 < x < −37.5 | 서브노멀 값 | **scipy는 x < −37.7에서 0** |
  | `log_ndtr(+inf)`, `log_ndtr(-inf)` | −0.0, −inf | |
  | `log_ndtr(x)`, 37.5 < x < 38.47 | 서브노멀 값 | **scipy는 x > 37.7에서 −0.0** |
  | `ndtri(±0)`, `ndtri(1)`, `ndtri(0.5)` | −inf, inf, 0 | [0, 1] 밖과 ±inf는 NaN |
  | `erfinv(±1)`, `erfinv(-0.0)` | ±inf, **−0.0** | **scipy는 `erfinv(-0.0) = 0.0`**. [−1, 1] 밖은 NaN |
  | `erfcinv(0)`, `erfcinv(2)`, `erfcinv(1)` | inf, −inf, −0.0 | [0, 2] 밖은 NaN |
  | `erfcinv(5e-324)` | 27.21 | **scipy는 inf** (z/2가 0으로 언더플로) |

## §3. 감마와 베타
```
gamma(x, /, *, out=None)
gammaln(x, /, *, out=None)
rgamma(x, /, *, out=None)
digamma(x, /, *, out=None)
polygamma(n, x, /, *, out=None)
beta(a, b, /, *, out=None)
betaln(a, b, /, *, out=None)
binom(n, k, /, *, out=None)
```
- `gammaln(x) = log|Γ(x)|`. 부호는 주지 않는다 (scipy와 같다).
- `rgamma(x) = 1/Γ(x)`. 극점에서 0이다.
- 극점: `gamma(0.0)`은 `inf`, `gamma(-0.0)`은 `-inf`, 음의 정수에서는 NaN이다. `gammaln`은 0과 음의 정수에서 `inf`다 (scipy와 같다).
- `polygamma(n, x)`의 `n`은 int64나 float64다. 0 이상의 정수 값만 정의되고, 정수가 아니거나 음수이면 NaN이다. `n = 0`은 `digamma`와 같다.
- `binom(n, k)`는 실수 n, k의 일반화 이항계수다.
- **보장 영역** (사용자 결정 2026-11-14): 상대오차 1e-14는 `gamma`, `rgamma`의 정의역 전체, `gammaln`, `digamma`, `polygamma`의 x > 0, `beta`의 a, b > 0, `binom`의 n ≥ 0이다. 그 밖(음수 쪽)과 `betaln` 전체는 |오차| ≤ 1e-14·max(1, |값|)이다. log|Γ|, ψ, log B에는 음수 쪽이나 B = 1인 곡선 위에 0이 무한히 많아 상대오차를 보장할 수 없기 때문이다.
- **잰 정확도** (bench/special/s3/accuracy.md, 보장 영역 최대 ULP): `gamma` 6.2, `rgamma` 5.1, `gammaln` 5.5, `digamma` 2.1, `polygamma` 17, `beta` 9.0, `binom` 9.3. 혼합 한계 영역(오차/max(1,|값|), 2⁻⁵² 단위): `gammaln` 2.1, `digamma` 15, `polygamma` 3.1, `beta` 6.8, `betaln` 7.6, `binom` 8.0
- **특수값**: 극점과 ±inf의 규약은 scipy와 비트까지 같다. 다른 곳(docs/design/special.md §2.4)은 아래 표의 굵은 줄이다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `gamma(±0)`, `gamma(-n)`, `gamma(-inf)` | ±inf, NaN, NaN | |
  | `gamma(x)`, −184 < x < −171 (정수 아님) | 서브노멀 값, 부호 포함 | **scipy는 0**. 더 아래는 부호 있는 0 (`gamma(-1e10-0.5) = -0.0`, **scipy +0.0**) |
  | `rgamma(±0)`, `rgamma(-n)`, `rgamma(±inf)` | ±0, 0, 0 | |
  | `rgamma(x)`, 171.6 < x < 177.8 | 서브노멀 값 | **scipy는 0** |
  | `rgamma(-5e-324)` | −5e-324 | **scipy는 −0.0** |
  | `rgamma(-1e10-0.5)` | −inf | **scipy는 inf** |
  | `gammaln(±0)`, `gammaln(-n)`, `gammaln(-inf)` | inf, inf, −inf | |
  | `gammaln(±5e-324)` | 744.44 | **scipy는 inf** |
  | `digamma(+0)`, `digamma(-0)`, `digamma(-n)` | −inf, inf, NaN | |
  | `polygamma(n, x)` | n이 음수·정수 아님·±inf이면 NaN | |
  | `beta`, `betaln`의 0·음의 정수 인자 | scipy(Cephes `beta.c`)의 규약 | NaN이 하나라도 있으면 NaN (**scipy는 inf**) |
  | `beta(inf, 1e300)` | 0 | **scipy는 NaN** (내부의 1e6·b 오버플로) |
  | `binom(n, k)`, k ≥ 0 정수 | n(n−1)…(n−k+1)/k! | **음의 정수 n도** (`binom(-5, 2) = 15`, scipy NaN) |
  | `binom(n, k)`, 거대한 k | 반사식과 Γ 비율 | `binom(-0.5, 1e300) = 5.6e-151` (**scipy −5.1e-151**), `binom(0, 1e300) = 0` (**scipy −2.9e-301**) |
  | `zeta(1)`, `zeta(inf)`, `zeta(-2n)` | inf, 1, 0 | |
  | `zeta(inf, q)` | q > 1이면 0, q = 1이면 1, 0 < q < 1이면 inf | **scipy는 q > 1에서 NaN** |

## §4. 불완전 감마와 베타
```
gammainc(a, x, /, *, out=None)
gammaincc(a, x, /, *, out=None)
gammaincinv(a, p, /, *, out=None)
gammainccinv(a, q, /, *, out=None)
betainc(a, b, x, /, *, out=None)
betaincc(a, b, x, /, *, out=None)
betaincinv(a, b, p, /, *, out=None)
betainccinv(a, b, q, /, *, out=None)
```
- 모두 정규화된 함수다. `gammainc`는 P(a, x), `gammaincc`는 Q(a, x) = 1 − P, `betainc`는 I_x(a, b), `betaincc`는 1 − I_x(a, b)다.
- 보수(`*cc`)는 1에서 빼서 계산하지 않는다. 꼬리에서도 상대 정확도가 유지된다.
- 역함수는 `gammainc(a, x) = p`, `gammaincc(a, x) = q`, `betainc(a, b, x) = p`, `betaincc(a, b, x) = q`의 x를 돌려준다.
- **보장 영역** (사용자 결정 2026-11-14): 네 순방향 함수는 정의역 전체(a, b > 0, x의 정의역 안)에서 상대오차 1e-14다. 결과가 서브노멀이면 docs/design/special.md §2.3의 절대오차 규칙을 따른다. 네 역함수는 `|Δx/x| ≤ 1e-14·max(1, κ)`다. κ = |F/(x·F′)|는 p(또는 q)에 대한 x의 조건수다. 입력의 마지막 비트가 허용하는 만큼 정확하다는 뜻이다. 예를 들어 a가 작으면 x ≈ (p·Γ(a+1))^(1/a)라서 p의 1 ULP가 x를 1/a ULP 움직인다. 테스트는 감마 a ≤ 1e6, 베타 a, b ≤ 1e5를 덮는다.
- **잰 정확도** (bench/special/s4/accuracy.md, 최대 ULP): `gammainc` 8.9, `gammaincc` 13, `betainc` 33, `betaincc` 19. 역함수(ULP / max(1, κ)): `gammaincinv` 6.7, `gammainccinv` 8.8, `betaincinv` 59 ULP(상대오차로 기준 안), `betainccinv` 20
- **특수값**: 끝점과 ±inf, NaN의 규약은 scipy와 비트까지 같다. 다른 곳(docs/design/special.md §2.4)은 아래 표의 굵은 줄이다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `gammainc(0, x)`, x > 0 | 1 | `gammainc(0, 0)`은 NaN |
  | `gammainc(a, 0)`, `gammainc(a, inf)` | 0, 1 | `gammainc(inf, inf)`은 NaN, `gammainc(inf, x)`는 0 |
  | 음수 a, x | NaN | |
  | `gammainc(1e-300, 5e-324)` | 1.0 | **scipy는 1.0000000000000238 (> 1)** |
  | `gammainc(5e-324, 0.5)`, `gammainc(5e-324, 1e-300)` | 1.0 | **scipy는 0** (참값 1 − 2.8e-324) |
  | `gammainc(1, 5e-324)` | 5e-324 | **scipy는 0** |
  | `betainc(a, b, 0)`, `betainc(a, b, 1)` | 0, 1 | x의 끝점이 a = 0, b = 0보다 먼저다 (`betainc(0, 1, 0) = 0`) |
  | `betainc(0, b, x)`, `betainc(a, 0, x)`, 0 < x < 1 | 1, 0 | `betainc(0, 0, x)`, `betainc(inf, inf, x)`는 NaN |
  | `betainc(inf, b, x)`, `betainc(a, inf, x)` | 0, 1 | x가 [0, 1] 밖이면 NaN |
  | `betainc(2, 1e300, 5e-324)` | 1.2205e-47 | **scipy는 NaN** |
  | `gammaincinv(a, 0)`, `gammaincinv(a, 1)` | 0, inf | `gammainccinv`은 반대. a = 0, inf이면 (0, 1) 안에서 NaN |
  | `gammaincinv(5e-324, 0.5)` | 0 | **scipy는 NaN** (근 0.5^(1/a)가 언더플로) |
  | `betaincinv(a, b, 0)`, `betaincinv(a, b, 1)` | 0, 1 | `betainccinv`은 반대. a, b가 0이면 끝점에서도 NaN, inf이면 (0, 1) 안에서 NaN |
  | `betaincinv(2, 3, 5e-324)`, `betainccinv(2, 3, 5e-324)` | 7.2e-163, 1.0 | **scipy는 NaN** |
  | `betaincinv(0.0102, 36.2, 3.13e-169)` | 0 | **scipy는 2.2e-308** (DBL_MIN에 고정. 근은 1e-16522) |

## §5. 로지스틱과 정보량
```
expit(x, /, *, out=None)
logit(p, /, *, out=None)
log_expit(x, /, *, out=None)
xlogy(x, y, /, *, out=None)
xlog1py(x, y, /, *, out=None)
entr(x, /, *, out=None)
rel_entr(x, y, /, *, out=None)
```
- `xlogy(x, y) = x·log(y)`. `x = 0`이고 y가 NaN이 아니면 0이다. `xlog1py`도 같다.
- `entr(x) = -x·log(x)`. `entr(0)`은 0, 음수는 `-inf`다.
- `rel_entr(x, y) = x·log(x/y)`. 규약은 scipy와 같다: `x = 0, y ≥ 0`이면 0, 그 밖의 음수는 `inf`다.
- **보장 영역**: 일곱 함수 모두 정의역 전체다 (사용자 결정 2026-11-13). `expit`, `log_expit`는 유한한 모든 x, `logit`은 0 < p < 1, `xlogy`는 y > 0, `xlog1py`는 y > −1, `entr`는 x ≥ 0, `rel_entr`는 x, y > 0이다. 결과가 넘치거나 서브노멀이면 §1.3과 docs/design/special.md §2.3을 따른다.
- **잰 정확도** (mpmath 대비 최대 ULP, bench/special/s1/accuracy.md): `expit` 2.1, `logit` 2.8, `log_expit` 1.2, `xlogy` 1.2, `xlog1py` 1.2, `entr` 1.3, `rel_entr` 2.1
- **특수값**: scipy와 비트까지 같다. 예외는 `expit`의 서브노멀 꼬리 하나다(아래 표). 시험 입력은 ±0, ±inf, NaN, 서브노멀, 경계값을 서로 조합한 것이다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `expit(±0)` | 0.5 | |
  | `expit(+inf)`, `expit(-inf)` | 1.0, 0.0 | |
  | `expit(x)`, −745.13 < x < −709.78 | eˣ (서브노멀) | **scipy는 0** (docs/design/special.md §2.4) |
  | `logit(±0)`, `logit(1)` | −inf, inf | |
  | `logit(p)`, p < 0 또는 p > 1, ±inf | NaN | |
  | `log_expit(+inf)`, `log_expit(-inf)` | −0.0, −inf | |
  | `xlogy(0, y)`, `xlog1py(0, y)` | 0 | y가 NaN이 아니면 (y ≤ 0, ±inf 포함). x = −0도 +0 |
  | `xlogy(x, 0)` | x > 0: −inf, x < 0: inf | |
  | `xlogy(inf, 1)` | NaN | inf · 0 |
  | `xlog1py(x, -1)` | x > 0: −inf | |
  | `entr(0)`, `entr(1)`, `entr(+inf)` | 0, −0.0, −inf | |
  | `entr(x)`, x < 0 | −inf | |
  | `rel_entr(0, y)`, y ≥ 0 (inf 포함) | 0 | |
  | `rel_entr(x, y)`, 그 밖의 음수 입력 | inf | |
  | `rel_entr(inf, y)`, `rel_entr(x, inf)` | inf, −inf | x, y는 양의 유한수. `rel_entr(inf, inf)`는 NaN |

## §6. 리덕션
```
logsumexp(x, /, *, axis=None, keepdims=False)
softmax(x, /, *, axis=None)
log_softmax(x, /, *, axis=None)
```
- `x`는 `sd.ndarray`여야 한다. `axis` 규칙은 배열 리덕션과 같다 (docs/api/array.md §1.3, §7.2): int, tuple, 리스트를 받는다. `None`은 모든 축이다.
- `logsumexp`의 결과가 0차원이면 Python `float`다. `keepdims=True`이면 줄인 축을 길이 1로 남긴다.
- `softmax`, `log_softmax`의 결과는 `x`와 shape이 같다. 그래서 `keepdims`가 없다.
- 빈 배열: `logsumexp`는 `-inf`다 (빈 합 0의 log).
- 모든 원소가 `-inf`이면 `logsumexp`는 `-inf`, `+inf`가 하나라도 있으면 `inf`, NaN은 전파한다.
- 무한대의 규약은 scipy와 같다. `logsumexp`·`log_softmax`는 최댓값이 무한이면 0으로 바꿔 빼고(`log_softmax([inf, 1]) = [nan, -inf]`), `softmax`는 바꾸지 않는다(`softmax([inf, 1]) = [nan, nan]`, 모두 `-inf`이면 NaN).
- **보장** (사용자 결정 2026-11-14): `logsumexp`·`log_softmax`는 |오차| ≤ 1e-14·max(1, |값|), `softmax`는 원소마다 상대오차 1e-14(서브노멀은 절대오차 규칙)다. 값이 0을 지날 수 있는 둘은 상대오차를 정의할 수 없어서 혼합 한계를 쓴다.
- **잰 정확도** (tests/special/ref/reduce_cases.npz, 122개 배열, 2⁻⁵² 단위): `logsumexp` 0.69, `softmax` 2.1(scipy 257: xᵢ − max의 반올림), `log_softmax` 1.3

## §7. 제타와 지수적분
```
zeta(x, q=None, /, *, out=None)
expi(x, /, *, out=None)
expn(n, x, /, *, out=None)
```
- `zeta(x)`는 Riemann ζ, `zeta(x, q)`는 Hurwitz ζ(x, q)다. Riemann은 `x < 1`도 정의된다(함수방정식). Hurwitz의 `x < 1`은 NaN이다 (scipy와 같다).
- `zeta`는 S3에서 구현했다(`expi`, `expn`은 S5). 보장 영역: Riemann은 정의역 전체, Hurwitz는 q > 0(그 밖은 |오차| ≤ 1e-14·max(1, |값|)). 잰 최대: Riemann 24 ULP(0 < x < 1, Euler–Maclaurin의 상쇄), Hurwitz 5.5 ULP. 특수값은 §3의 표에 있다.
- `expi`는 지수적분 Ei, `expn(n, x)`는 일반화 지수적분 Eₙ(x)다. `n`은 int64나 float64이고, 0 이상의 정수 값만 정의된다. 정수가 아니거나 음수이면 NaN이다. **scipy는 `expn(1.5, 1.0)`에서 n을 1로 자른다. 우리는 NaN이다.**
- **보장 영역** (사용자 결정 2026-11-14): `expi`, `expn`은 정의역 전체에서 상대오차 1e-14다. `expi`의 양의 0점 x₀ = 0.3725… 근처도 상대 정확도다. `expn`의 테스트는 n ≤ 1e5를 덮는다. 서브노멀 결과는 절대오차 규칙이다.
- **잰 정확도** (bench/special/s5/accuracy.md, 최대 ULP): `expi` 13.6, `expn` 6.3
- **특수값**: 아래 표의 굵은 줄 말고는 scipy와 비트까지 같다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `expi(±0)`, `expi(inf)`, `expi(-inf)` | −inf, inf, −0.0 | |
  | `expi(x)`, 709.78 < x < 716.4 | 유한한 값 | **scipy는 inf** (`expi(710) = 3.15e305`) |
  | `expn(0, 0)`, `expn(1, 0)`, `expn(n, 0)` | inf, inf, 1/(n−1) | x < 0은 NaN |
  | `expn(n, x)`, x > 709.78 | 서브노멀 값 | **scipy는 0** (`expn(1, 712) = 8.5e-313`) |
  | `expn(2**31, 1)` | 1.71e-10 | **scipy는 NaN** (n이 int32를 넘음) |
  | `expn(1.5, x)`, `expn(inf, x)`, `expn(-1, x)` | NaN | **scipy는 n을 자른다** (`expn(inf, 1) = 1.7e-10`) |

## §8. Bessel 함수

### §8.1 0차와 1차
```
i0(x, /, *, out=None)
i0e(x, /, *, out=None)
i1(x, /, *, out=None)
i1e(x, /, *, out=None)
j0(x, /, *, out=None)
j1(x, /, *, out=None)
y0(x, /, *, out=None)
y1(x, /, *, out=None)
k0(x, /, *, out=None)
k0e(x, /, *, out=None)
k1(x, /, *, out=None)
k1e(x, /, *, out=None)
```
- e-스케일 판: `i0e(x) = exp(-|x|)·i0(x)`, `k0e(x) = exp(x)·k0(x)` (1차도 같다)
- **보장** (사용자 결정 2026-11-14): `i0 i0e i1 i1e k0 k0e k1 k1e`는 정의역 전체에서 상대오차 1e-14다. `j0 j1 y0 y1`은 진동 함수 기준이다. 함수의 첫 양의 영점 z₁에 대해 x < 0.99·z₁이면 상대 1e-14, 그 뒤는 |오차| ≤ 1e-14·M(x)다. 포락선 M(x) = √(J_n(x)² + Y_n(x)²)는 큰 x에서 √(2/(πx))다. 영점 바로 앞에서는 상대오차를 정의할 수 없어서 경계를 0.99·z₁에 둔다.
- **특수값**: 아래 표의 굵은 줄 말고는 scipy와 비트까지 같다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `i0(±inf)`, `i1(±inf)` | inf, ±inf | **scipy는 NaN** |
  | `j0 j1 y0 y1(inf)`, `j0 j1(-inf)` | 0 | **scipy는 NaN**. 음의 x에서 `y0 y1`은 NaN |
  | `i0(x)`, `i1(x)`, 713 ≤ x < 713.98 | 유한한 값 | **scipy는 inf** |
  | `k0(5e-324)`, `k0e(5e-324)` | 744.56 | **scipy는 inf** (x/2가 0으로 언더플로) |
  | `k1(5e-324)`, `k1e(5e-324)` | inf | **scipy는 NaN** |
  | `k0(0)`, `y0(0)` | inf, −inf | 음의 x에서 `k0 k1 y0 y1`은 NaN |

### §8.2 실수 차수
```
jv(v, x, /, *, out=None)
yv(v, x, /, *, out=None)
iv(v, x, /, *, out=None)
kv(v, x, /, *, out=None)
jve(v, x, /, *, out=None)
yve(v, x, /, *, out=None)
ive(v, x, /, *, out=None)
kve(v, x, /, *, out=None)
```
- 차수 `v`와 인자 `x`는 실수다. 음의 `x`는 정수 차수에서만 정의되고, 그 밖은 NaN이다 (scipy와 같다).
- 실수 인자에서 `jve`, `yve`는 `jv`, `yv`와 같은 값이다 (scipy와 같다: 스케일 인자 `exp(-|Im x|)`가 1이다).
- 큰 차수(|v| ≫ x)는 맞는 값이지만 계산 시간이 |v|에 비례한다.
- **보장** (사용자 결정 2026-11-14): `iv`(v ≥ 0 또는 정수), `kv`, `ive`, `kve`는 정의역 전체에서 상대오차 1e-14다. `jv yv jve yve`는 §8.1의 진동 함수 기준이다. M = √(J_v² + Y_v²)이고, 음의 비정수 v는 모든 x에서 포락선 기준이다. 음의 비정수 v의 `iv`는 I₋ₐ = Iₐ + (2/π)sin(aπ)Kₐ의 두 항이 상쇄할 수 있어서 |오차| ≤ 1e-14·(|Iₐ| + |(2/π)sin(aπ)Kₐ|)다. 테스트는 |v| ≤ 1000, x ≤ 1e5를 덮는다.
- **특수값**: 아래 표의 굵은 줄 말고는 scipy와 비트까지 같다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `jv yv(v, inf)`, `iv(v, inf)`, `kv(v, inf)` | 0, 0, inf, 0 | **scipy는 jv·yv·iv에서 NaN** |
  | `ive(v, inf)`, `kve(v, inf)` | 0 | **scipy는 NaN** |
  | `yv(inf, x)`, `kv(1e10, x)`, `kve(1e10, x)` | −inf, inf, inf | **scipy는 NaN** |
  | `ive(1e10, x)`, `yve(1e10, x)` | 0, −inf | **scipy는 NaN, +inf** |
  | `iv(-1.5, 0)` | −inf | **scipy는 inf**. 부호는 sin(aπ)의 부호다 |
  | `yv(-0.5, 0)`, `yv(-1.5, 0)` | 0 | **scipy는 ∓inf**. 반정수에서 cos(aπ) = 0 |
  | `jve yve ive(v, 0)` | `jv yv iv`와 같다 | **scipy는 NaN** |
  | `jv(v, x)`, `iv(v, x)`, x < 0 | 정수 v에서 (−1)ⁿ·f(v, −x), 그 밖은 NaN | `yv kv`는 음의 x에서 NaN |

## §9. Airy 함수
```
airy(x, /)
```
- `AiryResult(ai, aip, bi, bip)`를 돌려준다. Ai, Ai′, Bi, Bi′다.
- 스칼라 입력이면 각 필드가 Python `float`다.
- **보장** (사용자 결정 2026-11-14): x ≥ 0은 네 값 모두 상대오차 1e-14, −1e12 ≤ x < 0은 |오차| ≤ 1e-14·M(x)(Ai, Bi; M = √(Ai² + Bi²))와 1e-14·N(x)(Ai′, Bi′; N = √(Ai′² + Bi′²))다.
- **느슨한 영역**: x < −1e12. 위상 ζ = (2/3)|x|^(3/2)를 double-double로 만들어서 그 절대 오차 ζ·2⁻¹⁰⁶이 위상을 흐린다. 잰 값(포락선 대비 2⁻⁵² 단위): −1e12에서 22, −1e14에서 6e3, −1e20에서 8e12. 입력이 1 ULP만 달라도 위상이 |x|^1.5·2⁻⁵² 움직이는 나쁜 조건의 영역이다 (사용자 결정 2026-11-14).
- **특수값**: `airy(0)`은 scipy와 같다. **`airy(inf)` = (0, −0, inf, inf)** (scipy NaN), `airy(-inf)` = (0, NaN, 0, NaN)(도함수는 진폭이 커져 극한이 없다), **`airy(105)`의 Ai는 서브노멀 2.7e-313, Bi는 inf** (scipy 0, NaN)

## §10. 타원적분
```
ellipk(m, /, *, out=None)
ellipe(m, /, *, out=None)
```
- 매개변수는 scipy와 같은 `m`(= k²)이다. `ellipk(1)`은 `inf`, `m > 1`은 NaN이다.
- **보장 영역** (사용자 결정 2026-11-14): 둘 다 m ≤ 1 전체에서 상대오차 1e-14다.
- **잰 정확도** (bench/special/s5/accuracy.md, 최대 ULP): `ellipk` 2.3, `ellipe` 2.0
- **특수값**: scipy와 비트까지 같다. `ellipk(0) = ellipe(0) = π/2`, `ellipe(1) = 1`, `ellipk(-inf) = 0`, `ellipe(-inf) = inf`

## §11. 초기하 함수와 Lambert W
```
hyp1f1(a, b, x, /, *, out=None)
hyp2f1(a, b, c, z, /, *, out=None)
lambertw(x, /, *, k=0, out=None)
```
- `hyp1f1`은 Kummer 함수 ₁F₁(a; b; x)다. `b`가 0 이하의 정수이면 극점이다.
- `hyp2f1`은 Gauss 함수 ₂F₁(a, b; c; z)다. z < 1에서 정의된다.
  - z ≥ 1: 다항식(a나 b가 0 이하의 정수)이면 그 값, z = 1이고 `c-a-b > 0`이면 Gauss 공식의 값이다. 그 밖은 NaN이다. **scipy는 일부 경우에 `inf`를 낸다** (docs/design/special.md §2.2).
- `lambertw(x, *, k=0)`: `k`는 0 또는 −1인 int다. 다른 값은 `ValueError`다(long을 넘는 int도).
  - k = 0은 x ≥ −1/e, k = −1은 −1/e ≤ x < 0에서 정의된다. 그 밖은 NaN이다.
  - 결과는 float64다. scipy는 complex를 돌려주고 `tol=`을 받는다. 우리는 둘 다 다르다.
  - **−1/e에 가장 가까운 double(−0.36787944117144233)은 분기점으로 본다**: 두 가지 모두 −1이다 (사용자 결정 2026-11-14, Boost와 같다). 그 double은 참 −1/e보다 8e-18 바깥이라 scipy는 NaN이다. 그보다 작은 x는 NaN이다.
- **보장** (사용자 결정 2026-11-14):
  - `lambertw`: 두 가지 모두 정의역 전체에서 상대오차 1e-14다. 분기점 근처는 e·x + 1을 double-double로 만들어 상대 정확도를 낸다(scipy는 그곳에서 2e7 ULP를 잃는다).
  - `hyp1f1`: 상쇄 없는 표현이 있는 a ≥ 0, b > 0이고 (x ≥ 0 또는 b ≥ a)인 곳만 상대 1e-14다. 그 밖은 느슨한 영역이다(아래 측정).
  - `hyp2f1`: a, b ≥ 0, c > 0, c ≥ min(a, b), −1 ≤ z ≤ 0.9에서 상대 1e-14다. 그 밖은 느슨한 영역이다.
  - 느슨한 영역의 잰 최대(참값 대비 상대오차, `bench/special/s10/accuracy.md`의 표본): `hyp1f1` 2.8e-13(S10부터 Boost.Math. S7의 우리 식은 a < 0, x > 0에서 3.9e-3이었다. scipy 2.9e-13), `hyp2f1` 5.8e-13(scipy 3.2e-4).
- **특수값**: 아래 표의 굵은 줄 말고는 scipy와 같다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `hyp1f1(a, b, 0)`, `hyp1f1(0, b, x)` | 1 | **`hyp1f1(0, 0, x) = 1`** (scipy inf) |
  | `hyp1f1(a, b, x)`, b가 0 이하의 정수 | inf | a가 0 이하의 정수이고 a ≥ b이면 다항식의 값 |
  | `hyp1f1(a, b, ±inf)` | 극한 | **`hyp1f1(-1, 2, inf) = -inf`** (scipy inf). 부호는 Γ(b)/Γ(a)(다항식은 최고차항)의 부호 |
  | `hyp2f1(a, b, c, z)`, c가 0 이하의 정수 | inf | 다항식이 먼저 끝나면 그 값 |
  | `hyp2f1(a, b, c, -inf)` | 극한 | **a, b > 0이면 0** (scipy NaN) |
  | `hyp2f1(a, b, c, z)`, z > 1 또는 z = 1이고 c − a − b ≤ 0 | NaN | 다항식이면 값. scipy는 일부 경우 inf |
  | `lambertw(-0.0)`, `lambertw(0, k=-1)`, `lambertw(inf)` | −0.0, −inf, inf | |

## §12. 복소수 (S8)
```
wofz(z, /, *, out=None)
loggamma(z, /, *, out=None)
```
- S8부터 `erf`, `erfc`, `erfcx`, `lambertw`가 complex를 받는다. 입력이 complex(complex128 배열이나 Python complex)이면 복소 루프로 가고 결과도 complex128이다. 실수 입력은 그대로 실수 루프다(결과 float64).
- `wofz`는 Faddeeva 함수 `w(z) = exp(-z²)·erfc(-iz)`다. 실수 입력도 결과는 complex128이다(scipy와 같다).
- `loggamma`는 log Γ의 주가지다. 음의 실수축을 자른 평면에서 연속인 가지이고(scipy, mpmath와 같다), 자름 위의 x ± 0i는 허수부 0의 부호로 위·아래를 고른다. 실수 입력은 x > 0에서 `gammaln`과 같고, `loggamma(±0.0)`은 `inf`, x < 0은 NaN이다(scipy와 같다).
- `lambertw(z, *, k=0)`: `k`는 int다. complex 입력은 모든 정수 가지를 받는다. 실수 입력은 그대로 0과 −1만 받는다(다른 값은 `ValueError`). `k`가 int가 아니면(bool 포함) `TypeError`다.
  - 가지 자름(k = 0은 (−∞, −1/e], k ≠ 0은 (−∞, 0)) 위의 x ± 0i는 C99처럼 허수부 0의 부호로 위(+0)·아래(−0)의 극한을 고른다(사용자 결정 2026-11-14). **(−1/e, 0) 위의 x − 0i에서 k = −1은 아래쪽 극한(= k = 1 위쪽 값의 켤레)이다. scipy는 이곳에서 실수 값(위쪽)을 낸다.**
- **오차의 정의** (사용자 결정 2026-11-14): 복소수의 상대오차는 크기 기준 |오차| / |값|이다. 실수부와 허수부를 따로 재지 않는다.
- **보장** (사용자 결정 2026-11-14): 정의역 전체에서 아래와 같다.
  - `wofz`: Im z ≥ 0은 상대 1e-14다. Im z < 0에서는 w에 영점이 있어서 포락선 기준 |오차| ≤ 1e-14·(2|exp(−z²)| + |w(−z)|)이다(계산식 w(z) = 2exp(−z²) − w(−z)의 두 항 크기).
  - `erfc`: Re z ≥ 0은 상대, Re z < 0은 |오차| ≤ 1e-14·(2 + |erfc(−z)|)다.
  - `erf`: |z| < 2는 상대, |z| ≥ 2는 |오차| ≤ 1e-14·(1 + |erfc(±z)|)다(±는 Re를 0 이상으로 만드는 쪽).
  - `erfcx`: Re z ≥ 0은 상대, Re z < 0은 |오차| ≤ 1e-14·(2|exp(z²)| + |erfcx(−z)|)다.
  - `loggamma`: Re z > 0은 상대 1e-14다. Re z ≤ 0은 |오차| ≤ 1e-14·max(1, |loggamma(z)|)다(S3의 `gammaln`과 같은 꼴).
  - `lambertw`(complex): 모든 가지, 정의역 전체에서 상대 1e-14다.
  - exp(−z²)의 위상 −2xy는 double-double로 정확히 만든다. |xy|가 double을 넘으면(|z| > 1e154이고 |Re z| ≈ |Im z|) 위상을 표현할 수 없어서, 결과의 크기가 0이나 1로 정해지지 않는 곳은 NaN이다(scipy도 NaN).
- **특수값**: 아래 표의 굵은 줄 말고는 scipy와 같다. 실수축·허수축 위의 0인 부분의 부호는 도함수의 방향을 따른다(아래). scipy의 부호는 일관되지 않아서 비트까지는 맞추지 않는다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `erf(x ± 0i)`, `erfc(x ± 0i)`, `erfcx(x ± 0i)` | 실수 함수의 값, 허수부 ±0 | 허수부 부호: `erf`는 Im z의 부호, `erfc`·`erfcx`는 그 반대 |
  | `erf(iy)`, `erfc(iy)`, `wofz(iy)` | i·erfi(y), 1 − i·erfi(y), erfcx(y) | 0인 부분은 정확히 0(`wofz`의 허수부는 Re z의 부호) |
  | **`erf(1e300+1e300j)`** | **1** | scipy NaN. |erfc| ≤ 1/(√π|z|)라 위상과 상관없이 1 |
  | 결과가 오버플로하는 곳(예: `erfc(1+30j)`) | 참 위상 쪽의 ±inf | scipy는 반올림한 위상의 부호나 NaN |
  | `loggamma(-n ± 0i)` (n = 0, 1, …) | **inf + nan·i** | 극점: 실수부는 inf, 허수부는 다가가는 방향마다 다르다. scipy nan + nan·i |
  | **`loggamma(inf ± 0i)`, `loggamma(inf + iy)`** | **inf ± 0i, inf ± inf·i** | scipy NaN |
  | **`loggamma(x ± inf·i)`, `loggamma(-inf + iy)`(y ≠ 0)** | **−inf ± inf·i, −inf ∓ inf·i** | scipy NaN. `loggamma(-inf ± 0i)`은 NaN(극점이 쌓인다) |
  | **`lambertw(5e-324, k ≠ 0)`** | **참값(−751.06 ± 3.146i)** | scipy inf + nan·i |
  | **`lambertw(z, k)`, z가 무한대** | **inf + i(arg z + 2πk)** | scipy는 `infj` 등에서 NaN |
  | **`lambertw(-0.36787944117144233 ± 0j, k)`** | **참값 −1 ± 8.2e-9i** | scipy NaN. 실수 입력은 S7대로 −1 |
  | `lambertw(0j)`, `lambertw(0j, k ≠ 0)` | 0(부호 유지), −inf + 0i | scipy와 같다 |

## §13. 분포 함수 (T1)
```
stdtr(df, t, /, *, out=None)
stdtrit(df, p, /, *, out=None)
chdtr(v, x, /, *, out=None)
chdtrc(v, x, /, *, out=None)
chdtri(v, p, /, *, out=None)
fdtr(dfn, dfd, x, /, *, out=None)
fdtrc(dfn, dfd, x, /, *, out=None)
fdtri(dfn, dfd, p, /, *, out=None)
pdtr(k, m, /, *, out=None)
pdtrc(k, m, /, *, out=None)
bdtr(k, n, p, /, *, out=None)
bdtrc(k, n, p, /, *, out=None)
nbdtr(k, n, p, /, *, out=None)
nbdtrc(k, n, p, /, *, out=None)
ndtri_exp(y, /, *, out=None)
```
- 누적분포 P(X ≤ x)와 그 보수(`*c`), 역함수다. 이름과 뜻은 scipy.special과 같다.
  - `stdtr`: Student t. `stdtrit`은 `stdtr(df, t) = p`의 t다.
  - `chdtr`: 카이제곱, `gammainc(v/2, x/2)`. `chdtri`는 **`chdtrc`의** 역함수다(scipy와 같다): `chdtrc(v, x) = p`의 x.
  - `fdtr`: F 분포. `fdtri`는 `fdtr(dfn, dfd, x) = p`의 x다.
  - `pdtr`: 포아송, `bdtr`: 이항, `nbdtr`: 음이항(n번째 성공 전의 실패 수, 성공 확률 p).
  - `ndtri_exp(y)`: `ndtri(exp(y))`. `exp(y)`가 언더플로하는 y(−1e300까지)도 계산한다.
- **계산**: betainc·gammainc 등의 조합이다(docs/design/special.md §5.15). 꼬리는 보수 함수로 직접 계산해서 1에서 빼지 않는다. 그래서 꼬리에서도 상대 정확도가 유지된다.
- **정수 인자** (T1 결정): `pdtr bdtr nbdtr`의 k는 내림한다(P(X ≤ k)의 뜻). `bdtr`의 n은 음이 아닌 정수여야 한다(아니면 NaN). `nbdtr`의 n은 0 이상의 실수다. scipy는 n도 내림한다.
- **특수값** (참값·극한, T1 결정). scipy와 다른 곳은 굵게 적었다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `stdtr(inf, t)` | `ndtr(t)` | 정규분포 |
  | `stdtr(df, ±inf)`, `stdtr(df, 0)` | 1 / 0, 0.5 | df ≤ 0, NaN이면 NaN |
  | `stdtrit(df, 0)`, `stdtrit(df, 1)` | −inf, inf | |
  | `stdtrit(3, 5e-324)` | −6.07e107 | **scipy는 inf** |
  | `chdtr(0, x)`, x > 0 | 1 | `chdtr(0, 0)`은 NaN(scipy와 같다) |
  | **`fdtr(dfn, inf, x)`** | `gammainc(dfn/2, dfn·x/2)` | 극한(dfn·F → 카이제곱). **scipy는 NaN** |
  | **`fdtr(inf, dfd, x)`** | `gammaincc(dfd/2, dfd/(2x))` | 극한(dfd/F → 카이제곱). **scipy는 NaN** |
  | **`fdtr(inf, inf, x)`** | x < 1이면 0, x ≥ 1이면 1 | 1에 몰린 분포. **scipy는 NaN** |
  | `fdtri`의 무한 자유도 | 위 극한의 역함수 | **scipy는 NaN** |
  | **`pdtr(k < 0, m)`**, **`pdtrc(k < 0, m)`** | 0, 1 | **scipy는 NaN** |
  | **`bdtr(k < 0, n, p)`** | 0 | **scipy는 NaN**(`bdtrc`는 scipy도 1) |
  | **`bdtr(k ≥ n, n, p)`** | 1 (`bdtrc`는 0) | **scipy는 k > n에서 NaN** |
  | **`bdtr(k, 5.5, p)`** | NaN | n이 정수가 아니다. **scipy는 n을 내린다** |
  | **`nbdtr(k, 5.5, p)`** | `betainc(5.5, k+1, p)` | 실수 n. **scipy는 n을 내린다** |
  | **`nbdtr(k, 0, p)`**, 0 < p ≤ 1 | 1 | 실패가 없다. **scipy는 NaN** |
  | `nbdtr(k, inf, p)` | 0 | |
  | `ndtri_exp(0)`, `ndtri_exp(-inf)`, y > 0 | inf, −inf, NaN | |
- **보장 영역** (사용자 결정 2026-11-15): 잰 범위 전체다. 순방향 11개와 `ndtri_exp`는 상대오차 1e-14(약 45 ULP), 역함수 `stdtrit chdtri fdtri`는 §4처럼 `|Δx/x| ≤ 1e-14·max(1, κ)`다. 결과가 서브노멀이면 docs/design/special.md §2.3의 절대오차 규칙을 따른다. 범위는 아래 '잰 범위'이고, 그 밖(더 큰 자유도·k·n)도 값을 계산하지만 보장하지 않는다. 경계값 `fdtr(inf, inf, x)`, `nbdtr(k, 0, p)`, `chdtri(inf, p)` = NaN도 같은 날 확정했다.
  - **잰 값** (bench/special/t1/accuracy.md, 최대 ULP; 역함수는 ULP/max(1, κ)): `stdtr` 9.0, `stdtrit` 6.9, `chdtr` 6.1, `chdtrc` 9.7, `chdtri` 7.2, `fdtr` 16, `fdtrc` 18, `fdtri` 23, `pdtr` 4.5, `pdtrc` 13, `bdtr` 10, `bdtrc` 9.5, `nbdtr` 14, `nbdtrc` 19, `ndtri_exp` 6.5
  - 같은 점에서 scipy: 최대 25 ULP(`chdtri`)부터 4.9e16 ULP(`fdtri`), inf(`stdtrit` 등)까지.
  - 잰 범위: 자유도 1e-2–1e6(t), 1e-2–1e4(F), k·n 1e4까지, `ndtri_exp`의 y는 −1e300–0. 근이 double 범위 밖이면(`fdtri` → 0, `stdtrit` → −inf) 순방향 함수로 확인한다(tests/special/test_distributions.py).
