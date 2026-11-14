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

## §7. 제타와 지수적분
```
zeta(x, q=None, /, *, out=None)
expi(x, /, *, out=None)
expn(n, x, /, *, out=None)
```
- `zeta(x)`는 Riemann ζ, `zeta(x, q)`는 Hurwitz ζ(x, q)다. Riemann은 `x < 1`도 정의된다(함수방정식). Hurwitz의 `x < 1`은 NaN이다 (scipy와 같다).
- `zeta`는 S3에서 구현했다(`expi`, `expn`은 S5). 보장 영역: Riemann은 정의역 전체, Hurwitz는 q > 0(그 밖은 |오차| ≤ 1e-14·max(1, |값|)). 잰 최대: Riemann 24 ULP(0 < x < 1, Euler–Maclaurin의 상쇄), Hurwitz 5.5 ULP. 특수값은 §3의 표에 있다.
- `expi`는 지수적분 Ei, `expn(n, x)`는 일반화 지수적분 Eₙ(x)다. `n`은 int64나 float64이고, 0 이상의 정수 값만 정의된다. 정수가 아니거나 음수이면 NaN이다. **scipy는 `expn(1.5, 1.0)`에서 n을 1로 자른다. 우리는 NaN이다.**

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

## §9. Airy 함수
```
airy(x, /)
```
- `AiryResult(ai, aip, bi, bip)`를 돌려준다. Ai, Ai′, Bi, Bi′다.
- 스칼라 입력이면 각 필드가 Python `float`다.

## §10. 타원적분
```
ellipk(m, /, *, out=None)
ellipe(m, /, *, out=None)
```
- 매개변수는 scipy와 같은 `m`(= k²)이다. `ellipk(1)`은 `inf`, `m > 1`은 NaN이다.

## §11. 초기하 함수와 Lambert W
```
hyp1f1(a, b, x, /, *, out=None)
hyp2f1(a, b, c, z, /, *, out=None)
lambertw(x, /, *, k=0, out=None)
```
- `hyp1f1`은 Kummer 함수 ₁F₁(a; b; x)다. `b`가 0 이하의 정수이면 극점이다.
- `hyp2f1`은 Gauss 함수 ₂F₁(a, b; c; z)다. z < 1에서 정의된다.
  - z ≥ 1: 다항식(a나 b가 0 이하의 정수)이면 그 값, z = 1이고 `c-a-b > 0`이면 Gauss 공식의 값이다. 그 밖은 NaN이다. **scipy는 일부 경우에 `inf`를 낸다** (docs/design/special.md §2.2).
- `lambertw(x, *, k=0)`: `k`는 0 또는 −1인 int다. 다른 값은 `ValueError`다.
  - k = 0은 x ≥ −1/e, k = −1은 −1/e ≤ x < 0에서 정의된다. 그 밖은 NaN이다.
  - 결과는 float64다. scipy는 complex를 돌려주고 `tol=`을 받는다. 우리는 둘 다 다르다.

## §12. 복소수 (S8)
```
wofz(z, /, *, out=None)
loggamma(z, /, *, out=None)
```
- S8부터 `erf`, `erfc`, `erfcx`, `lambertw`가 complex를 받는다. complex 입력이면 결과도 complex128이다.
- `wofz`는 Faddeeva 함수 `w(z) = exp(-z²)·erfc(-iz)`다.
- `loggamma`는 log Γ의 주가지다. 실수 입력은 x > 0에서 `gammaln`과 같고, `loggamma(0.0)`은 `inf`, x < 0은 NaN이다 (scipy와 같다).
- `lambertw`의 `k`: S8부터 complex 입력은 모든 정수 가지를 받는다. 실수 입력은 그대로 0과 −1만 받는다.
- 이 절의 이름은 S8 전까지 `sd.special.__all__`에 없다.
