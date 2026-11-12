# 특수함수 API (sd.special)

> 상태: **확정** (2026-11-12). S0(기반)까지 구현했고 공개 함수는 아직 없다. 설계와 결정 근거는 [docs/design/special.md](../design/special.md)에 있다. 보장 영역과 특수값 표는 각 마일스톤에서 측정한 뒤 채운다.

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
- 보장 영역, 특수값: S2에서 확정

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
- 보장 영역, 특수값: S3에서 확정

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
- 보장 영역, 특수값: S4에서 확정

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
- 특수값: S1에서 확정

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
