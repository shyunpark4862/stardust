# 특수함수 API (sd.special)

> 상태: **확정** (2026-11-12). §2–§14의 함수가 모두 구현되어 있다. 설계와 결정 근거는 [docs/design/special.md](../design/special.md)에 있다.

`import stardust as sd`의 하위 모듈 `sd.special`이 특수함수 API다 (scipy.special의 부분집합).

- 라이브러리 전체의 공통 규칙(dtype, 에러 종류, 스레드)은 루트 [API.md](../../API.md)에 있다. 원소별 함수의 피연산자, `out=`, 0차원 결과 규칙은 [docs/api/array.md](array.md) §1.2, §6.2를 그대로 따른다.
- 절 번호(`§N`, `§N.M`)는 코드 주석이 인용한다. 번호를 바꾸지 않는다.
- 언어 표시가 없는 코드 블록은 `tests/special/test_api_surface.py`가 읽는다. 공개 이름과 시그니처만 쓰고, 한 줄에 함수 하나다.

**인자 순서** (사용자 결정 2026-11-20). 입력은 모두 위치 전용이라 순서를 바꿔 넣으면 에러 없이 다른 값이 나온다. 인자가 둘 이상인 함수의 순서는 아래와 같다. 굵게 적은 줄만 scipy.special과 순서가 다르다.

| 함수 | 순서 |
|---|---|
| `polygamma`, `expn` | (차수, x) |
| `beta`, `betaln`, `binom` | (a, b), (a, b), (n, k) |
| `gammainc`, `gammaincc`, `gammaincinv`, `gammainccinv` | (a, x), (a, x), (a, p), (a, q) |
| `betainc`, `betaincc`, `betaincinv`, `betainccinv` | (a, b, x), (a, b, x), (a, b, p), (a, b, q) |
| `xlogy`, `xlog1py`, `rel_entr` | (x, y) |
| `zeta`, `jv yv iv kv jve yve ive kve` | (x, q), (v, x) |
| `hyp1f1`, `hyp2f1` | (a, b, x), (a, b, c, z) |
| `stdtr`, `stdtrit` | (df, t), (df, p) |
| `chdtr`, `chdtrc`, `chdtri` | (df, x), (df, x), (df, q) |
| `fdtr`, `fdtrc`, `fdtri` | (dfn, dfd, x), (dfn, dfd, x), (dfn, dfd, p) |
| `pdtr`, `pdtrc`, `bdtr`, `bdtrc`, `nbdtr`, `nbdtrc` | (k, m), (k, n, p) |
| **`chndtr`, `chndtrc`, `chndtrix`** | **(df, nc, x), (df, nc, p)**. scipy는 `chndtr(x, df, nc)`, `chndtrix(p, df, nc)` |
| `nctdtr`, `nctdtrc`, `nctdtrit` | (df, nc, t), (df, nc, p) |
| `ncfdtr`, `ncfdtrc`, `ncfdtri` | (dfn, dfd, nc, x), (dfn, dfd, nc, p) |

---

## §1. 특수함수 공통 규칙

### §1.1 시그니처와 이름
- 함수 이름은 scipy.special의 이름이다. Array API에는 특수함수가 없다.
- **입력은 모두 positional-only(`/` 앞)다.** 옵션(`out`, `axis`, `keepdims`, `k`)은 keyword-only(`*` 뒤)다. 입력에도 이름이 있다(`help()`와 에러 메시지에 아래 시그니처의 이름이 나온다). 이름은 보여 주기만 하고 키워드로는 받지 않는다.
- sd에 이미 있는 함수(`expm1`, `log1p`, `round` 등)는 여기에 다시 두지 않는다. 별칭(`psi`, `jn`)도 없다.

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
- 보장 영역 밖(정의역 안)도 값을 계산한다. 잰 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표에 영역(보장·느슨)별로 있고, 함수별 절이 그 표를 가리킨다.
- 모든 호출은 정해진 시간 안에 끝난다 (사용자 결정 2026-11-17). 급수·연분수·반복은 횟수의 상한이 있고, 상한 안에 수렴하지 않으면 부분합이나 마지막 반복값 대신 NaN을 낸다. 보장 영역 안에서는 상한에 닿지 않는다. NaN이 되는 범위는 함수별 절에 적는다.
- 진동 함수(`j0 j1 y0 y1 jv yv jve yve`, x < 0의 `airy`)는 상대오차 대신 `|오차| ≤ 1e-14 × 포락선`을 보장한다. 영점 근처에서는 상대오차를 정의할 수 없기 때문이다. 포락선의 식은 함수별 절에 적는다.
- 같은 입력이면 위치, stride, 스레드 수와 상관없이 같은 비트가 나온다 (`f(x)[i] == f(x[i])`, API.md §3).
- scipy와 같은 비트는 약속이 아니다 (사용자 결정 2026-11-20). 특수값, NaN, ±0, 극점의 규약은 아래 표대로 scipy와 같고(다른 곳은 굵게 적는다), 그 밖의 값은 위의 정확도 계약 안이면 마지막 자리가 다를 수 있다.
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
- **잰 정확도**: Arb 참값 대비 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 함수마다 있다.
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
- `polygamma`는 n > 1e6이고 값이 0도 ±inf도 아닌 좁은 구간에서 NaN이다(n ≤ 1e6까지 double-double log로 계산한다).
- `binom(n, k)`는 실수 n, k의 일반화 이항계수다.
- **보장 영역** (사용자 결정 2026-11-14): 상대오차 1e-14는 `gamma`, `rgamma`의 정의역 전체, `gammaln`, `digamma`, `polygamma`의 x > 0, `beta`의 a, b > 0, `binom`의 n ≥ 0이다. 그 밖(음수 쪽)과 `betaln` 전체는 |오차| ≤ 1e-14·max(1, |값|)이다. log|Γ|, ψ, log B에는 음수 쪽이나 B = 1인 곡선 위에 0이 무한히 많아 상대오차를 보장할 수 없기 때문이다.
- **잰 정확도**: Arb 참값 대비 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 함수마다 있다. 보장 영역은 guaranteed 행의 ULP이고, 혼합 한계 영역(오차/max(1,|값|), 2⁻⁵² 단위)은 mixed 행의 mixed 열이다.
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
  | `polygamma(n, x)` | n이 음수·정수 아님·±inf이면 NaN | **`polygamma(inf, 1)`: scipy는 inf**. ψ⁽ⁿ⁾(1) = (−1)ⁿ⁺¹ n! ζ(n+1)은 크기가 발산하고 부호가 n마다 바뀌어 극한이 없다 |
  | `beta`, `betaln`의 0·음의 정수 인자 | scipy(Cephes `beta.c`)의 규약 | NaN이 하나라도 있으면 NaN (**scipy는 inf**) |
  | `beta(inf, 1e300)` | 0 | **scipy는 NaN** (내부의 1e6·b 오버플로) |
  | `beta(a, b)`, a, b 모두 서브노멀 | 1/a + 1/b | B = (a + b)/(ab)·(1 + O(a + b)). 대부분 inf다(`beta(5e-324, 5e-324)`) |
  | **`beta(-inf, 1)`**, **`betaln(-inf, 1)`** | −0.0, −inf | 극한(B(a, 1) = 1/a). **scipy는 inf, inf** |
  | `binom(n, k)`, k ≥ 0 정수 | n(n−1)…(n−k+1)/k! | **음의 정수 n도** (`binom(-5, 2) = 15`, scipy NaN) |
  | **`binom(-inf, 2)`** | inf | 극한(n(n−1)/2). **scipy는 NaN** |
  | `binom(n, -inf)`, 유한한 n > −1 | 0 | 극한(크기가 \|k\|^−(n+1)로 줄고 부호만 진동한다). 그 밖의 n과 k = +inf는 NaN이다. **`binom(inf, -inf)`: scipy는 0** |
  | `binom(n, k)`, 거대한 k | 반사식과 Γ 비율 | `binom(-0.5, 1e300) = 5.6e-151` (**scipy −5.1e-151**), `binom(0, 1e300) = 0` (**scipy −2.9e-301**) |
  | `zeta(1)`, `zeta(inf)`, `zeta(-2n)` | inf, 1, 0 | |
  | `zeta(inf, q)` | q > 1이면 0, q = 1이면 1, 0 < q < 1이면 inf | **scipy는 q > 1에서 NaN** |
  | `zeta(s, q)`, `polygamma(n, x)`, 음수 q·x와 큰 차수 | 값 또는 ±inf | 0 옆의 두 항((q + N − 1)⁻ˢ, (q + N)⁻ˢ)이 double을 넘을 때는 가까운 쪽의 첫 항에 대한 상대값으로 더한다(m·2ᵉ). 반정수 q이고 s가 홀수이면 두 합이 정확히 지워져 ζ(s, 1 − q)가 남는다: `zeta(1025, -0.5)` = `zeta(1025, 1.5)`, `polygamma(7000, -0.5)` = −inf, `polygamma(3000, -2.25)` = inf |

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
- **잰 정확도**: Arb 참값 대비 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 함수마다 있다. 역함수는 ULP/max(1, κ) 열이다.
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
  | `betainc(a, b, x)`, a, b < 2⁻⁷⁰⁰, 0 < x < 1 | b/(a + b) | 두 끝에 질량이 몰린 극한. `betainc(5e-324, 5e-324, 0.3) = 0.5` (2026-11-18, 전에는 1.0) |
  | `gammaincinv(a, 0)`, `gammaincinv(a, 1)` | 0, inf | `gammainccinv`은 반대. a = 0, inf이면 (0, 1) 안에서 NaN |
  | `gammaincinv(5e-324, 0.5)` | 0 | **scipy는 NaN** (근 0.5^(1/a)가 언더플로) |
  | `betaincinv(a, b, 0)`, `betaincinv(a, b, 1)` | 0, 1 | `betainccinv`은 반대. a, b가 0이면 끝점에서도 NaN |
  | **`betaincinv(inf, b, p)`**, **`betaincinv(a, inf, p)`**, 0 < p < 1 | 1, 0 | 극한(`betainc(inf, b, x) = 0`처럼 질량이 1, 0에 몰린다; 사용자 결정 2026-11-20). `betainccinv`도 같은 근이다. a, b가 모두 inf이면 NaN. **scipy는 NaN** |
  | `betaincinv(a, b, p)`, 아주 작은 모양 | 극한 꼴의 근 | 작은 모양이 곱해지지 않은 첫 항 말고는 모든 항에 그 모양이 곱해진다. 하나가 서브노멀이고 둘 다 2⁻⁷⁰⁰ 아래이면 질량 b/(a + b)와 p를 비교해 0이나 1이다. b가 min(2⁻²⁰⁰, 2⁻⁸⁰a) 아래이면(서브노멀이든 정규수든) 근은 p/b에만 달려서(`betaincinv(1, 5e-324, 5e-324)` = 1 − 1/e) p와 b를 같은 2의 거듭제곱으로 그 크기까지 키운 문제의 근이다. 서브노멀 a만 작으면 0이다 |
  | `betaincinv(2, 3, 5e-324)`, `betainccinv(2, 3, 5e-324)` | 9.07e-163, 1.0 | **scipy는 NaN** |
  | `betaincinv(0.0102, 36.2, 3.13e-169)` | 0 | **scipy는 2.2e-308** (DBL_MIN에 고정. 근은 1e-16522) |
  | `betaincinv(a, b, p)`, a, b ≥ 2⁶⁰, 서브노멀 p | 정규 극한의 근 | 꼬리의 로그로 풀 수 없는 곳에서 평균 m = a/(a + b)와 표준편차 σ에 왜도 항을 더한 m + σ(z + g(z² − 1)/6)를 낸다(Cornish–Fisher). 다음 항은 근의 2⁻⁷⁴ 아래다. `betaincinv(1e100, 1e100, 5e-324)` = 0.5 |
- **NaN이 되는 곳**: 한 모양이 약 2.3e6에서 2⁸⁰(1.2e24) 사이이고 다른 모양이 1보다 작을 때, 서브노멀 p의 `betaincinv(큰 모양, 작은 모양, p)`와 `betainccinv(작은 모양, 큰 모양, p)`(근이 x = 1 쪽에 붙는 꼬리)는 NaN이다. 꼬리의 로그를 구할 급수도 연분수도 거기에 닿지 않는다. 2⁸⁰ 위는 감마 극한으로 값을 낸다. `fdtri(dfn, dfd, p)`도 dfn/2가 그 범위이고 dfd < 2이면 같다.
- **아주 작은 a의 역함수** (2026-11-19, T52): a < 2⁻⁹⁰⁰(서브노멀 a 포함)이면 double이 닿는 모든 y에서 Q(a, y) = a·E₁(y)가 반올림 안에서 맞으므로, `gammainccinv(a, q)`는 E₁(y) = c의 근이다. c = q/a는 나눗셈 한 번으로 구한다(log q − log a는 −710 근처의 두 로그의 차라 c에 1e-13의 오차를 넣고, 근의 상대오차는 c배가 된다). c > 40이면 y = e^(−c−γ), 그 밖은 이분이다. Arb 대비 Q의 상대오차는 c = 5–700에서 4.5e-16 이하다(`gammainccinv(1e-310, 5e-310)` = 0.0037974640015846885). 근 y 자체의 반올림이 Q를 y·|Q′/Q|배로 옮기므로, c가 아주 작아 y가 크면 Q의 상대오차는 근의 반올림의 y배까지 커진다(`gammainccinv(2⁻¹⁰²², 1e-320)`에서 1.6e-15). 근의 정확도는 §4의 역함수 기준 안이다. 2⁻¹⁰²² 근처의 정규수 a에서 q < 1e-300이면(Boost를 쓰지 않는 곳) 다른 경로가 NaN이나 7e-13의 오차를 냈는데, 이 식은 거기도 같다. `gammaincinv(a, p)`는 서브노멀 a에서 p < 1이면 0이다(근이 e^(log p / a), 모든 double 아래).

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
- **잰 정확도**: Arb 참값 대비 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 함수마다 있다.
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
- `zeta`는 S3에서 구현했다(`expi`, `expn`은 S5). 보장 영역: Riemann은 정의역 전체, Hurwitz는 q > 0(그 밖은 |오차| ≤ 1e-14·max(1, |값|)). 잰 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)의 `zeta`, `zeta_hurwitz` 행이다. Boost에 맡기는 구간과 그렇지 않은 구간은 docs/design/special.md §5.14에 있다. 짝수 정수 2–52는 표의 값이라 올바르게 반올림된다. 특수값은 §3의 표에 있다.
- `expi`는 지수적분 Ei, `expn(n, x)`는 일반화 지수적분 Eₙ(x)다. `n`은 int64나 float64이고, 0 이상의 정수 값만 정의된다. 정수가 아니거나 음수이면 NaN이다. **scipy는 `expn(1.5, 1.0)`에서 n을 1로 자른다. 우리는 NaN이다.**
- **보장 영역** (사용자 결정 2026-11-14): `expi`, `expn`은 정의역 전체에서 상대오차 1e-14다. `expi`의 양의 0점 x₀ = 0.3725… 근처도 상대 정확도다. `expn`의 테스트는 n ≤ 1e5를 덮는다. 서브노멀 결과는 절대오차 규칙이다.
- **잰 정확도**: Arb 참값 대비 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 함수마다 있다.
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
- 큰 차수(|v| ≫ x)는 맞는 값이지만 계산 시간이 |v|에 비례한다. 점화와 연분수는 1e7번에서 멈추고, 그보다 많이 필요한 `jv yv jve yve`는 NaN이다. 그런 곳은 셋이다. (가) x > |v| > 1e7이고 x가 (|v| + 1)²/4 아래일 때다(위로 가는 점화가 1e7걸음을 넘고, 큰 x의 Hankel 전개는 첫 항 (4v² − 1)/(8x)가 2를 넘어 받지 않는다. v = 1.2e7–2e7에서 잰 경계가 v²/x = 4.0이다). (나) |v| − x > 1e7이고 값이 0이나 무한대로 증명되지 않을 때다(연분수가 1e7항을 넘는다). (다) |v| ≳ 1e12이고 x가 전이 구간 |v| − x ≲ 1000·|v|^(1/3) 안에 있을 때다(x = v에서 연분수의 항 수가 약 6|v|^(1/3)로 늘고, 그 뒤의 Y 연분수와 점화가 끝나지 않는다. 1e12–2e12는 점마다 다르고, 2e12부터는 이 구간 전체가 NaN이다). x = v 근처는 |v| < 1e12에서 값이다: `jv(2e7, 2e7)` = 0.0016479, `jv(1e12, 1e12)` = 4.473e-05. x ≥ 2⁵³이고 |v| − x < 1e-12·x이면 연분수의 걸음 2/x가 반올림에 묻혀 끝나지 않는 것이 미리 보이므로 걷지 않고 바로 NaN이다(`jv(1e300, 1e300)`). 1e7걸음의 점화 한 번은 약 0.25초다(`yv(2e7, 1e7)` = −inf에 0.23초. 같은 점의 `jv`는 아래의 0 판정으로 바로 0이다).
  - **J의 0** (사용자 결정 2026-11-19): 0 < x < v에서 엄밀한 상한이 가장 작은 서브노멀의 절반(2⁻¹⁰⁷⁵ = e^−745.13) 아래임을 보이면 `jv`, `jve`는 0이다. 상한은 둘이다. DLMF 10.14.5의 |J_v(v sech α)| ≤ e^(−v(α − tanh α))(Debye의 지수)와 DLMF 10.14.4의 (x/2)^v/Γ(v + 1)(x ≪ v에서 더 좁다). 반올림을 덮으려고 e^−746.2 아래일 때만 쓴다. 예: `jv(3e7, 1.9e7)` = 0(참값 약 e^(−7.6e6); 전에는 NaN, scipy 0). 음의 정수 차수는 (−1)ⁿ·0, 음의 비정수 차수는 Y_a가 함께 들어가므로 이 판정을 쓰지 않는다.
  - **Y**: |Y_v|의 엄밀한 하한이 없어서(DLMF 10.14는 J만 다룬다) `yv`의 −inf는 넓히지 않았다. J와 Y를 함께 정하는 옛 판정(x < v/2이고 (x/2)^v/Γ(v + 1) < e^(−760)/(2v), `jv(3e7, 1.49e7)` = 0과 `yv` = −inf)만 −inf이고, 그 밖에서 |v| − x > 1e7이면 `yv`는 NaN이다(`yv(3e7, 1.9e7)`; scipy −inf).
- **보장** (사용자 결정 2026-11-14): `iv`·`ive`(v ≥ 0 또는 정수), `kv`, `kve`는 정의역 전체에서 상대오차 1e-14다. `jv yv jve yve`는 §8.1의 진동 함수 기준이다. M = √(J_v² + Y_v²)이고, 음의 비정수 v는 모든 x에서 포락선 기준이다. 음의 비정수 v의 `iv`·`ive`는 I₋ₐ = Iₐ + (2/π)sin(aπ)Kₐ의 두 항이 상쇄할 수 있어서 |오차| ≤ 1e-14·(|Iₐ| + |(2/π)sin(aπ)Kₐ|)다(`ive`는 두 항에 e⁻ˣ를 곱한 크기). 테스트는 |v| ≤ 1000, x ≤ 1e5를 덮는다(`ive`의 참값 파일은 v ≥ 0만).
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
- x < −1e20은 네 값 모두 NaN이다 (사용자 결정 2026-11-18). 위상의 오차가 1e-3 rad를 넘는다.
- **특수값**: `airy(0)`은 scipy와 같다. **`airy(inf)` = (0, −0, inf, inf)** (scipy NaN), `airy(-inf)` = (0, NaN, 0, NaN)(도함수는 진폭이 커져 극한이 없다), **`airy(105)`의 Ai는 서브노멀 2.7e-313, Bi는 inf** (scipy 0, NaN)

## §10. 타원적분
```
ellipk(m, /, *, out=None)
ellipe(m, /, *, out=None)
```
- 매개변수는 scipy와 같은 `m`(= k²)이다. `ellipk(1)`은 `inf`, `m > 1`은 NaN이다.
- **보장 영역** (사용자 결정 2026-11-14): 둘 다 m ≤ 1 전체에서 상대오차 1e-14다.
- **잰 정확도**: Arb 참값 대비 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 함수마다 있다.
- **특수값**: scipy와 비트까지 같다. `ellipk(0) = ellipe(0) = π/2`, `ellipe(1) = 1`, `ellipk(-inf) = 0`, `ellipe(-inf) = inf`

## §11. 초기하 함수와 Lambert W
```
hyp1f1(a, b, x, /, *, out=None)
hyp2f1(a, b, c, z, /, *, out=None)
lambertw(x, /, *, k=0, out=None)
```
- `hyp1f1`은 Kummer 함수 ₁F₁(a; b; x)다. `b`가 0 이하의 정수이면 극점이다.
- `hyp2f1`은 Gauss 함수 ₂F₁(a, b; c; z)다. z < 1에서 정의된다.
  - 다항식(a나 b가 0 이하의 정수 −n)은 급수를 그대로 더한다. 0 < z < 1에서 그 합이 1e16배 넘게 상쇄하면(교대하는 (−n)ₖ 때문에 차수 60–200부터) Pfaff 변환 ₂F₁(−n, b; c; z) = (1 − z)ⁿ ₂F₁(−n, c − b; c; z/(z − 1))의 다항식으로 더한다. c > 0이고 c − b > 0이면 모든 항이 양수다(Jacobi·Gegenbauer 다항식이 이 꼴이다): `hyp2f1(-200, 1, 2, 0.3)` = 0.016583747927031506. 그 밖은 NaN이다.
  - z ≥ 1: 다항식(a나 b가 0 이하의 정수)이면 그 값, z = 1이고 `c-a-b > 0`이면 Gauss 공식의 값이다. 그 밖은 NaN이다. **scipy는 일부 경우에 `inf`를 낸다** (docs/design/special.md §2.2).
- `lambertw(x, *, k=0)`: `k`는 0 또는 −1인 int다(numpy 정수도 받는다. bool과 float는 `TypeError`). int64 밖은 다른 int 인자처럼 `OverflowError: lambertw(): 'k' is out of range`이고, 그 안의 다른 값은 `ValueError`다.
  - k = 0은 x ≥ −1/e, k = −1은 −1/e ≤ x < 0에서 정의된다. 그 밖은 NaN이다.
  - 결과는 float64다. scipy는 complex를 돌려주고 `tol=`을 받는다. 우리는 둘 다 다르다.
  - **−1/e에 가장 가까운 double(−0.36787944117144233)은 분기점으로 본다**: 두 가지 모두 −1이다 (사용자 결정 2026-11-14, Boost와 같다). 그 double은 참 −1/e보다 8e-18 바깥이라 scipy는 NaN이다. 그보다 작은 x는 NaN이다.
- **보장** (사용자 결정 2026-11-14):
  - `lambertw`: 두 가지 모두 정의역 전체에서 상대오차 1e-14다. 분기점 근처는 e·x + 1을 double-double로 만들어 상대 정확도를 낸다(scipy는 그곳에서 2e7 ULP를 잃는다).
  - `hyp1f1`: 상쇄 없는 표현이 있는 a ≥ 0, b > 0이고 (x ≥ 0 또는 b ≥ a)인 곳만 상대 1e-14다. 그 밖은 느슨한 영역이다(아래 측정). 잰 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)의 `hyp1f1` guaranteed 행이다. Boost의 값은 0 ≤ x < 16, b < 11, a ≤ 100에서만 쓴다(그 밖의 x에서는 297 ULP까지 틀렸다).
  - `hyp2f1`: a, b ≥ 0, c > 0, c ≥ min(a, b), −1 ≤ z ≤ 0.9에서 상대 1e-14다. 그 밖은 느슨한 영역이다.
  - 느슨한 영역은 값을 내되 보장하지 않는다(사용자 결정 2026-11-18: 값 유지, 잰 최대를 적는다). 잰 최대(Arb 참값 대비 상대오차, 무작위 1.1만 점씩, 인자 크기 1e6까지): `hyp1f1` 9e-12(Tricomi 전개; Boost를 쓰는 |a|, |b| ≤ 100은 1.2e-12), `hyp2f1` 1.7e-12(z < −9이고 |c| > 1e4). 1e-12를 넘는 점은 각각 3점이었다. 정확도표에서는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 `hyp1f1`, `hyp2f1` loose 행이다. 같은 점에서 scipy는 `hyp1f1` 2.33e3 ULP, `hyp2f1` 1.83e12 ULP다(커밋 4151d2a의 bench/special/arb/accuracy.md).
  - NaN이 되는 곳(보장 영역 밖, 2026-11-18 갱신): 손실을 아는 식이 없으면 값을 내지 않는다.
    - `hyp1f1`: 인자 하나가 크기 1e6을 넘을 때(짧은 다항식과 큰 x의 점근식은 값). 그 안에서는 max(|a|, |b|) > 100이고 다음 가운데 어느 것도 쓸 수 없을 때다: 상쇄 손실이 1e13배 이하인 급수(1e6항까지), b > 0에서 a에 대한 점화식(53비트를 지킬 때), 큰 |x|의 점근식, Tricomi 전개(a < −1e3, b ≤ 170). 대부분 b < 0이고 |a|나 |b|가 큰 곳이지만, b > 0에도 a < 0이고 x ≈ |a|·b 근처인 띠가 있다: `hyp1f1(-150.5, 1, 150)`은 NaN이다(scipy는 1e-15 안으로 맞는다). |a|, |b| ≤ 100은 Boost의 값을 쓴다.
    - `hyp2f1`: 급수가 1e6항(극한 꼴의 합은 1e5항) 안에 끝나지 않을 때, 급수의 상쇄가 1e16배를 넘을 때, 1 − z(z < −9에서는 1/(1 − z))가 1e-3 아래인데 c − a − b가 정수에서 2⁻⁴⁸–1e-3만큼 떨어져 있을 때, 극한 꼴의 합이 64배 넘게 상쇄하거나 Γ가 범위를 벗어날 때, 큰 항들이 정확히 지워진 합(전에는 0을 냈다)이다.
    - 이 정리 전에는 느슨한 영역에서 조용히 틀린 값이 났다: `hyp1f1(-1000, -100.5, 0.999)`가 4.29e152(참값 34799.142), `hyp2f1(20, 1, 1, -10)`이 0(참값 1.486e-21)이었다. 무작위 1.1만 점에서 1e-12 넘게 틀린 점이 `hyp1f1` 256 → 3, `hyp2f1` 103 → 3이 됐다(남은 최대 9e-12, 1.7e-12).
- **특수값**: 아래 표의 굵은 줄 말고는 scipy와 같다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `hyp1f1(a, b, 0)`, `hyp1f1(0, b, x)` | 1 | **`hyp1f1(0, 0, x) = 1`** (scipy inf) |
  | `hyp1f1(a, b, x)`, b가 0 이하의 정수 | inf | a가 0 이하의 정수이고 a ≥ b이면 다항식의 값 |
  | `hyp1f1(a, b, ±inf)` | 극한 | **`hyp1f1(-1, 2, inf) = -inf`** (scipy inf), **`hyp1f1(0.5, 2, -inf) = 0`** (scipy NaN; a > 0이면 M ~ Γ(b)/Γ(b−a)·(−x)⁻ᵃ). 부호는 Γ(b)/Γ(a)(다항식은 최고차항)의 부호 |
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
- `lambertw(z, *, k=0)`: `k`는 int다. complex 입력은 |k| ≤ 2⁵³인 모든 정수 가지를 받는다(k는 double로 루프에 들어가므로 그 위는 `ValueError`, int64 밖은 `OverflowError`). 실수 입력은 그대로 0과 −1만 받는다(다른 값은 `ValueError`). `k`가 int가 아니면(bool 포함) `TypeError: lambertw(): 'k' must be an int, not float` 꼴이다.
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
chdtr(df, x, /, *, out=None)
chdtrc(df, x, /, *, out=None)
chdtri(df, q, /, *, out=None)
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
  - `chdtr`: 카이제곱, `gammainc(df/2, x/2)`. `chdtri`는 **`chdtrc`의** 역함수다(scipy와 같다): `chdtrc(df, x) = q`의 x.
  - `fdtr`: F 분포. `fdtri`는 `fdtr(dfn, dfd, x) = p`의 x다.
  - `pdtr`: 포아송, `bdtr`: 이항, `nbdtr`: 음이항(n번째 성공 전의 실패 수, 성공 확률 p).
  - `ndtri_exp(y)`: `ndtri(exp(y))`. `exp(y)`가 언더플로하는 y도 double의 끝(−1.8e308)까지 계산한다(2026-11-19; 전에는 −2y가 넘치는 y ≤ −8.99e307에서 NaN이었다).
- **계산**: betainc·gammainc 등의 조합이다(docs/design/special.md §5.15). 꼬리는 보수 함수로 직접 계산해서 1에서 빼지 않는다. 그래서 꼬리에서도 상대 정확도가 유지된다.
- **정수 인자** (T1 결정): `pdtr bdtr nbdtr`의 k는 내림한다(P(X ≤ k)의 뜻). `bdtr`의 n은 음이 아닌 정수여야 한다(아니면 NaN). `nbdtr`의 n은 0 이상의 실수다. scipy는 n도 내림한다.
- **특수값** (참값·극한, T1 결정). scipy와 다른 곳은 굵게 적었다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | `stdtr(inf, t)` | `ndtr(t)` | 정규분포 |
  | `stdtr(df, ±inf)`, `stdtr(df, 0)` | 1 / 0, 0.5 | df ≤ 0, NaN이면 NaN |
  | **`stdtrit(df, 0)`**, `stdtrit(df, 1)` | −inf, inf | p = 0의 분위수. **scipy는 `stdtrit(df, 0)`에서 inf** |
  | `stdtrit(3, 5e-324)` | −6.07e107 | **scipy는 inf** |
  | `stdtrit(df, p)`, df < 2⁻¹⁰⁰⁰ | p < 0.5이면 −inf, p > 0.5이면 inf | \|t\| = √df·(q·a·B)^(−1/df)가 double을 넘는다 |
  | `stdtr(df, t)`, 서브노멀 df, t ≠ 0 유한 | 0.5 | df → 0에서 질량이 ±∞로 반씩 간다. df = 5e-324의 df/2는 double이 아니라 반을 a·B(a, 1/2) → 1의 극한으로 들고 간다 |
  | `chdtr(df, 0)`, `chdtrc(df, 0)`, df > 0 | 0, 1 | 서브노멀 df도(df > 0에는 0의 점질량이 없다) |
  | `chdtri(df, q)`, 마지막 비트가 홀수인 서브노멀 df | Q(df, y) = 2q의 근의 2배 | df/2가 double이 아니다. Q(a, y)는 a → 0에서 a에 선형이라 Q(df/2, y) = Q(df, y)/2. q > 1/2이면 0 |
  | `fdtri(dfn, dfd, p)`, dfn이나 dfd가 5e-324 | 극한 | dfn만이면 0(분포가 0에 몰린다), dfd만이면 inf(p ≤ 1/2이면 dfd에 대해 선형인 `fdtr`로 2p의 근), 둘 다 2⁻⁶⁹⁹ 아래이면 질량 dfd/(dfn + dfd)와 p를 비교해 0이나 inf |
  | `chdtr(0, x)`, x > 0 | 1 | `chdtr(0, 0)`은 NaN(scipy와 같다) |
  | **`fdtr(dfn, inf, x)`** | `gammainc(dfn/2, dfn·x/2)` | 극한(dfn·F → 카이제곱). **scipy는 NaN** |
  | **`fdtr(inf, dfd, x)`** | `gammaincc(dfd/2, dfd/(2x))` | 극한(dfd/F → 카이제곱). **scipy는 NaN** |
  | **`fdtr(inf, inf, x)`** | x < 1이면 0, x ≥ 1이면 1 | 1에 몰린 분포. **scipy는 NaN** |
  | `fdtri`의 무한 자유도 | 위 극한의 역함수 | **scipy는 NaN** |
  | **`pdtr(k < 0, m)`**, **`pdtrc(k < 0, m)`** | 0, 1 | **scipy는 NaN** |
  | **`bdtr(k < 0, n, p)`** | 0 | **scipy는 NaN**(`bdtrc`는 scipy도 1) |
  | **`bdtr(k ≥ n, n, p)`** | 1 (`bdtrc`는 0) | k = inf도. **scipy는 k > n에서 NaN** |
  | **`bdtr(k, 5.5, p)`** | NaN | n이 정수가 아니다. **scipy는 n을 내린다** |
  | **`nbdtr(k, 5.5, p)`** | `betainc(5.5, k+1, p)` | 실수 n. **scipy는 n을 내린다** |
  | **`nbdtr(k, 0, p)`**, 0 < p ≤ 1 | 1 | 실패가 없다. **scipy는 NaN** |
  | `nbdtr(k, inf, p)` | 0 | |
  | **`nbdtr(k < 0, n, p)`** | 0 | P(X ≤ k < 0). **scipy는 NaN** |
  | **`nbdtr(inf, n, p)`** | 1 | cdf의 k → ∞. **scipy는 NaN** |
  | `ndtri_exp(0)`, `ndtri_exp(-inf)`, y > 0 | inf, −inf, NaN | |
- **보장 영역** (사용자 결정 2026-11-15): 잰 범위 전체다. 순방향 11개와 `ndtri_exp`는 상대오차 1e-14(2⁻⁵² 단위로 약 45), 역함수 `stdtrit chdtri fdtri`는 §4처럼 `|Δx/x| ≤ 1e-14·max(1, κ)`다. 결과가 서브노멀이면 docs/design/special.md §2.3의 절대오차 규칙을 따른다. 범위는 아래 '잰 범위'이고, 그 밖(더 큰 자유도·k·n)도 값을 계산하지만 보장하지 않는다. 경계값 `fdtr(inf, inf, x)`, `nbdtr(k, 0, p)`, `chdtri(inf, p)` = NaN도 같은 날 확정했다.
  - **잰 값**: Arb 참값 대비 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 함수마다 있다. 역함수는 ULP/max(1, κ) 열이다.
  - 같은 점에서 잰 scipy·R·Boost의 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 비교 패키지 열에 있다.
  - 잰 범위: 자유도 1e-2–1e6(t), 1e-2–1e5(카이제곱), 1e-2–1e4(F), k·n 1e4까지, 카이제곱의 x는 v/100–100v와 1e-6–50, `chdtri`의 p는 1e-300까지, `ndtri_exp`의 y는 −1e300–0. 근이 double 범위 밖이면(`fdtri` → 0, `stdtrit` → −inf) 순방향 함수로 확인한다(tests/special/test_distributions.py).

## §14. 비중심 분포 함수 (T2)
```
chndtr(df, nc, x, /, *, out=None)
chndtrc(df, nc, x, /, *, out=None)
chndtrix(df, nc, p, /, *, out=None)
nctdtr(df, nc, t, /, *, out=None)
nctdtrc(df, nc, t, /, *, out=None)
nctdtrit(df, nc, p, /, *, out=None)
ncfdtr(dfn, dfd, nc, x, /, *, out=None)
ncfdtrc(dfn, dfd, nc, x, /, *, out=None)
ncfdtri(dfn, dfd, nc, p, /, *, out=None)
```
- 비중심 카이제곱·t·F의 누적분포 P(X ≤ x), 위꼬리 P(X > x)(`*c`), 아래꼬리의 역함수다. nc는 비중심 모수다(카이제곱·F는 λ ≥ 0, t는 δ, 부호가 있다).
- **인자 순서** (T2 결정): 모든 함수에서 x(또는 p)가 맨 뒤다. **scipy의 `chndtr(x, df, nc)`, `chndtrix(p, df, nc)`와 순서가 다르다.** §13의 `chdtr(df, x)`와 다른 비중심 함수에 맞췄다(머리말의 순서 표).
- **위꼬리** (T2 결정): `chndtrc nctdtrc ncfdtrc`는 scipy.special에 없는 이름이다. 1에서 빼지 않고 직접 더하므로 꼬리에서도 상대 정확도가 유지된다. 매개변수에 대한 역함수(scipy의 `chndtridf`, `nctdtrinc` 등)는 두지 않는다.
- **계산**: 카이제곱·F는 포아송 혼합 Σ_j e^(−λ/2)(λ/2)^j/j! · G(j)이다. G는 불완전 감마·베타이고, 두 꼬리 모두 양의 항만 더한다. t는 같은 꼴의 두 급수다. 다만 δ와 반대쪽 꼬리는 그 두 급수가 상쇄하므로 T = (Z + δ)/√(V/df)의 적분으로 계산한다. 역함수는 작은 쪽 꼬리를 브래킷과 Brent법으로 푼다. 자세한 것은 docs/design/special.md §5.16에 있다.
- **특수값** (참값·극한, T1·T2 결정). scipy와 다른 곳은 굵게 적었다.

  | 입력 | 값 | 비고 |
  |---|---|---|
  | nc = 0 | 중심 함수(`chdtr`, `stdtr`, `fdtr`와 역함수)와 비트까지 같다 | |
  | **`chndtr(0, nc, x)`**, x > 0 | 0에 e^(−nc/2)가 몰린 분포의 값 | **scipy는 NaN** |
  | `chndtr(0, nc, 0)` | NaN | `chdtr(0, 0)`과 같다 |
  | `chndtrix(0, nc, p)`, p ≤ e^(−nc/2) | 0 | 점질량 안 |
  | `chndtr(df, nc, x)`, 0 < df < 2⁻⁹⁶⁰, x > 0 | df = 0의 값(`chndtrc`는 거기에 j = 0 항의 꼬리 e^(−nc/2)·(df/2)·E₁(x/2)를 더한다) | 모양 j와 df/2 + j의 차이는 상대 2⁻⁹⁵⁰ 아래다. x = 0은 0(df > 0에는 점질량이 없다). 2026-11-19, 전에는 df ≲ 1e-305의 작은 x와 df < 2⁻¹⁰²¹·nc < 2⁻⁸⁹⁹에서 `chndtrc`가 NaN(`chndtrc(1e-307, 1, 1e-300)`) |
  | `ncfdtr(5e-324, dfd, nc, f)` | `ncfdtr(1e-323, dfd, nc, f)`의 값 | dfn/2가 0으로 반올림된다. 차이는 절대 2⁻¹⁰⁶³ 아래. 2026-11-19, 전에는 f ≈ 1e23–1e173에서 NaN |
  | `ncfdtri`, `chndtrix`, 근이 double 밖 | inf(꼬리가 가장 큰 double에서도 목표에 못 미칠 때), 0(가장 작은 양수에서 이미 넘을 때) | 2026-11-19, 전에는 NaN: `ncfdtri(5e-324, 5, 2, 0.5)` = inf |
  | `ncfdtr(dfn, dfd, nc, f)`, dfn이 아주 작다(w < 2⁻¹⁰⁰⁰) | j = 0 항의 값(e^(−nc/2)·`fdtr`) | 2026-11-19, 전에는 dfn ≲ 2⁻⁹⁹⁷에서 NaN: `ncfdtr(2**-1021, 5, 2, 1)` = 0.36788 |
  | **`chndtr(inf, nc, x)`**, **`chndtr(df, inf, x)`** | 0 (`chndtrc`는 1) | 분포가 무한대로 간다. **scipy는 NaN**. x도 inf이면 NaN |
  | `nctdtr(inf, nc, t)` | `ndtr(t − nc)` | 정규분포. scipy와 같다 |
  | **`nctdtr(df, ±inf, t)`** | 0 / 1 | **scipy는 NaN**. t도 같은 부호의 inf이면 NaN |
  | `nctdtr(df, nc, 0)` | `ndtr(−nc)` | |
  | **`ncfdtr(dfn, inf, nc, f)`** | `chndtr(dfn, nc, dfn·f)` | 극한. **scipy는 NaN** |
  | **`ncfdtr(inf, dfd, nc, f)`** | `fdtr(inf, dfd, f)` | 분자 → 1. **scipy는 NaN** |
  | **`ncfdtr(dfn, dfd, inf, f)`** | 0 (`ncfdtrc`는 1) | **scipy는 NaN**. f도 inf이면 NaN |
  | 역함수의 p = 0, 1 | 정의역의 두 끝(0 또는 −inf, inf) | |
  | `chndtr(df, nc, x)`, Chernoff 한계가 e⁻¹⁰⁰⁰ 아래인 꼬리 | 0 또는 1 | 적률생성함수에서 P(X ≤ x) ≤ e^(x/2)·2^(−df/2)·e^(−nc/4), P(X ≥ x) ≤ e^(−x/4)·2^(df/2)·e^(nc/2). 합을 걷기 전에 거른다: `chndtr(1, 1e16, 1)` = 0 |
- **보장 영역** (사용자 결정 2026-11-15): 아래 '잰 범위' 전체다. 순방향 6개는 상대오차 1e-14(2⁻⁵² 단위로 약 45), 역함수 `chndtrix nctdtrit ncfdtri`는 §4처럼 `|Δx/x| ≤ 1e-14·max(1, κ)`다. 결과가 서브노멀이면 docs/design/special.md §2.3의 절대오차 규칙을 따른다. `chndtr(0, nc, 0)` = NaN(`chdtr(0, 0)`과 일관)도 같은 날 확정했다.
  - **잰 값**: Arb 참값 대비 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표(`python -m bench.accuracy`가 만든다)에 함수마다 있다. 역함수는 ULP/max(1, κ) 열이다.
  - 같은 점에서 잰 scipy(Boost를 감싼 것)·R·Boost 단독의 오차는 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 비교 패키지 열에 있다.
  - 잰 범위: 자유도 1e-2–1e4(카이제곱·F), 1e-1–1e5(t); nc 1e-3–1e4(카이제곱·F), |δ| ≤ 100(t); 확률은 1e-300까지. 근이 double 범위 밖이면 0이나 ±inf이고 순방향 함수로 확인한다(tests/special/test_noncentral.py).
  - NaN이 되는 곳(잰 범위 밖): 포아송 혼합을 걷는 길이가 2¹⁸걸음에서 멈추므로(정해진 시간), 비중심 모수가 크면 분포의 중심에서 NaN이다. 카이제곱·F는 λ ≳ 4e9(λ/2 ≳ 2e9)부터 중심 근처에서, λ ≥ 9e15(무게의 최빈값이 2⁵²를 넘는다)부터는 위의 Chernoff 한계로 걸러지지 않는 모든 x에서 NaN이다(`chndtr(1, 4e9, 4e9)`, `chndtrix(1, 1e10, 0.5)`). t는 |δ| ≳ 3.9e4(δ²/2 ≈ 7.6e8)부터 모든 t에서 NaN이다(`nctdtr(10, 4e4, 4e4)`, `nctdtrit(3.64e23, 51019.1, 0.0564)`). 모두 몇 ms 안에 끝난다. `ncfdtr`·`ncfdtrc`는 먼 꼬리에서 구한 쪽의 보수가 1/64 아래일 때(dfn이나 dfd ≤ 1e-10에서만 봤다. 전에는 `ncfdtr(1e7, 1e-300, 2, 0.1)`이 참값 3.4e-298 대신 2.2e-16). `chndtrix nctdtrit ncfdtri`는 nc ≠ 0이고 p < 2⁻¹⁰²²(서브노멀)일 때다(전에는 9–16% 틀렸다). t 함수는 |nc| < 2⁻⁵⁰⁰이면 중심 분포의 값이다.
  - 잰 범위 밖의 값은 보장하지 않는다(사용자 결정 2026-11-18: 값 유지): `nctdtr`·`nctdtrc`는 df ≤ 1e-3에서 상대 2.4e-13까지 틀린 점이 있었다(극단 인자 점검의 23점).
  - 그 밖(거대한 t·f에서 작은 인자가 2⁻¹⁰⁰⁰ 아래인 꼬리)은 첫 항 y^b로 계산한다. mpmath와 1e-16 수준으로 맞지만 보장하지 않는다.
