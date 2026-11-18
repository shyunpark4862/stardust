# 특수함수 영역 설계 (sd.special)

> 상태: **확정** (2026-11-12). 1–3라운드(Q1–Q19)의 결정을 반영했다. 마일스톤을 시작하기 전에 물을 것은 §10에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.** 시작은 통합 로드맵의 M7이 끝난 뒤다 (Q14). M7은 완료되었다 (a9fccbc).

`sd.special`(scipy.special의 교육용 재구현)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙과 통합 결정은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md)에 있다. 공개 함수와 시그니처는 [docs/api/special.md](../api/special.md)에 있다.
- 배열 영역의 내부(원소별 엔진, 버퍼 변환, 풀, NEON 규칙)는 [array.md](array.md)를 인용한다.
- 이 영역은 나중의 `sd.stats`(scipy.stats)의 바닥이다. 분포의 cdf·ppf는 대부분 여기의 함수로 계산된다.
- 수치는 Apple M4 Pro 한 대에서 잰 값이다. 비교 대상은 scipy 1.18.1이다.

---

## 1. 기반

### 1.1 결정 기록
| 번호 | 결정 (2026-11-12) |
|---|---|
| Q1 언어 | C++20 + nanobind, 같은 `_core`의 `csrc/special/`. Python 층은 거의 없다 |
| Q2 범위 | **대** (§1.2) |
| Q3 정확도 | 보장 영역에서 상대오차 1e-14 이내. 극단 영역은 잰 한계를 문서에 적는다 (§2.3) |
| Q4 근사 계수 | 공개된 계수를 출처를 밝혀 가져온다. 계수를 직접 만들지 않는다 (§4.1) |
| Q5 NEON | 분기가 적은 핫 함수만 손으로 짠다: erf/erfc/ndtr/ndtri, expit/logit, gammaln (§3.4) |
| Q6 dtype | float64 실수만. bool/int64는 float64로 바꾸고, complex는 `TypeError` (§2.1). 예외는 Q7 |
| Q7 복소수 | 실수를 모두 끝낸 뒤 마지막 마일스톤에서 erf/wofz, loggamma, lambertw만 복소수를 받는다. 복소수 Bessel(AMOS급)은 하지 않는다 |
| Q8 구간 한도 | 근사 방식을 바꾸는 갈래는 함수마다 2–3개. 정의역을 옮기는 변환(반사, 점화식, 선형 변환, 대칭)은 세지 않는다. 퇴화 경우는 일반 극한 공식 하나로 처리한다 (§4.2) |
| Q9 보장 영역 밖 | 값은 계산하고, 정확도가 느슨하다는 것을 문서에 적는다. NaN은 수학적 정의역 밖에서만 나온다 |
| Q10 정답지 | 엄밀한 구간 산술 Arb(python-flint)를 테스트 의존성으로 더한다(처음에는 mpmath, 2026-11-16 Arb로 바꿨다, §6). 정확도는 고정밀 값 대비 ULP로 잰다. scipy는 차이와 성능을 비교하는 데 쓴다 (§6) |
| Q11 n항 디스패처 | 배열 영역에 일반 n항 원소별 디스패처를 더하고 `kMaxOperands`를 5로 올린다. 별도 마일스톤(S0)이고, 기존 함수의 결과는 비트까지 같다 (§3.1) |
| Q12 이름 | scipy 이름을 쓴다. 입력은 모두 `/` 앞, `out=`은 `*` 뒤. sd에 이미 있는 함수는 다시 두지 않는다 |
| Q13 리덕션 옵션 | `logsumexp`, `softmax`, `log_softmax`의 옵션은 `axis`(기본 `None`, int·tuple·리스트도 받는다)와 `logsumexp`의 `keepdims`(기본 `False`)뿐이다. scipy의 `b`, `return_sign`은 없다 |
| Q14 시작 | 통합 로드맵 M7이 끝난 뒤 |
| Q15 무거운 함수 | `jv yv iv kv`와 `jve yve ive kve`(실수 차수, 실수 인자), `hyp1f1`, `hyp2f1`(실수), `lambertw`(가지 0, −1), `airy`(`AiryResult`) |
| Q16 커널 안의 `exp`/`log` | Q5의 NEON 함수는 커널 안에서 1024원소 버퍼로 vForce를 한꺼번에 부르고, 스칼라 입력도 n = 1로 같은 경로를 탄다. 나머지 함수는 libm (§3.4) |
| Q17 정수 차수 인자 | int64와 float64를 받는다. 정수가 아닌 값과 음수는 NaN. 자르지 않는다 (§2.1) |
| Q18 진동 함수의 영점 근처 | 상대오차 대신 `|오차| ≤ 1e-14 × 포락선`을 보장한다. 영점 인수분해는 하지 않는다 (§2.3) |
| Q19 복소수의 대상 | `wofz`, `loggamma`(새 함수)와 `erf erfc erfcx lambertw`의 complex 입력 (§5.13) |

### 1.2 범위
- **있는 것** (함수 목록과 시그니처는 docs/api/special.md)
  - 오차함수: `erf erfc erfcx erfinv erfcinv ndtr log_ndtr ndtri`
  - 감마: `gamma gammaln rgamma digamma polygamma beta betaln binom`
  - 불완전 함수: `gammainc gammaincc gammaincinv gammainccinv betainc betaincc betaincinv betainccinv`
  - 로지스틱·정보: `expit logit log_expit xlogy xlog1py entr rel_entr`
  - 리덕션: `logsumexp softmax log_softmax`
  - 제타와 지수적분: `zeta expi expn`
  - Bessel: 0·1차 `i0 i0e i1 i1e j0 j1 y0 y1 k0 k0e k1 k1e`, 임의 실수 차수 `jv yv iv kv jve yve ive kve`
  - Airy: `airy`
  - 타원적분: `ellipk ellipe`
  - 초기하와 Lambert W: `hyp1f1 hyp2f1 lambertw`
  - 복소수(마지막 마일스톤): `wofz`(새 함수), `loggamma`(새 함수), `erf erfc erfcx lambertw`의 complex 입력 (Q19)
- **없는 것**
  - sd에 이미 있는 함수: `expm1 log1p round` 등 (Q12)
  - 별칭: `jn`(= `jv`), `psi`(= `digamma`), `yn`(`yv`가 대신한다). 함수 하나에 이름 하나다 (docs/api/array.md §1.1)
  - Mathieu, 회전타원체, Struve, Kelvin, 직교다항식 `eval_*`, 구면조화, 구면 Bessel, `ellipkinc`/`ellipeinc`, `dawsn`, `erfi`, `fresnel`, `owens_t` 등
  - scipy의 오류 보고 체계(`errstate`, `seterr`, `SpecialFunctionWarning`): 경고를 내지 않는다 (API.md §2)

### 1.3 공통 원칙
- **모든 공개 함수는 원소별 함수다** (리덕션 셋과 `airy` 제외). 브로드캐스트, `out=`, Python 스칼라 결과, 버퍼 변환은 배열의 원소별 함수와 같다 (docs/api/array.md §6.2).
- **경로는 원소의 값으로만 고른다.** 급수/연분수/점근전개 중 무엇을 쓸지는 그 원소의 인자 값이 정한다. 이것은 함수의 정의의 일부다 (libm과 같다).
  - 배열의 크기, 위치, 정렬, 스레드 수로 경로를 고르지 않는다. 그래서 `f(x)[i] == f(x[i])`이고, 결과는 스레드 수와 무관하게 비트까지 같다 (API.md §3).
  - 배열 영역의 "데이터를 보고 알고리즘을 고르는 휴리스틱을 넣지 않는다"(array.md §1.4)는 배열 전체를 보고 고르는 것을 말한다. 원소 하나의 값으로 근사 구간을 고르는 것은 여기에 해당하지 않는다.
- **NEON 경로는 스칼라 정의와 비트가 같다** (array.md §5.2와 같은 규칙). 테스트가 세 레이아웃(연속, 스칼라 브로드캐스트, strided)을 비트로 비교한다.
- **원소의 값 때문에 실패하지 않는다** (docs/api/array.md §1.6). 정의역 밖은 NaN, 극점은 ±inf다.
- **모든 호출은 정해진 시간 안에 끝난다** (사용자 결정 2026-11-17; docs/design/stats.md §1.3과 같다). 반복에는 상한을 상수로 두고, 상한에 닿으면 부분합이나 마지막 반복값이 아니라 NaN을 낸다. 보장 영역 안에서는 상한에 닿지 않는다. 마지막 비트에서 두 값을 오가는 반복은 그 값을 낸다.
  - 상한: Halley 100회, 불완전 감마의 급수·연분수 1e5항, BPSER·BFRAC 1e5항, `expn`·`expi` 급수, Carlson RD 200회, AGM 64회, `lambertw` 64회, Bessel 멱급수 500항, 실수 차수 J·Y의 점화와 CF1 1e7(CF2 1e5; 1e7걸음에 약 0.25초), Hurwitz 제타의 직접 합 82항, `hyp1f1`·`hyp2f1` 급수 1e6항(극한 꼴 1e5항). 길이가 정해진 전개(BGRAT 30, BASYM 20, Hankel·Debye)는 그대로다.
  - Boost가 평가 오류(급수 1e6항, 근 찾기 200회)를 내면 `bst::call`이 NaN으로 바꾼다.
  - `pow_scaled`·`exp_scaled`는 2ᵏ를 제곱으로 만든다(1014번 이하).
- **알고리즘 선택 규칙**: 함수마다 후보(§5)를 아래 순서로 견주어 고른다. 고른 것과 이유는 그 마일스톤의 해설에 측정과 함께 적는다.
  1. 보장 영역에서 정확도 목표(§2.3)를 지키는가
  2. 구간 한도(Q8)를 지키는가
  3. 빠른가 (1스레드, 원소 10⁶개)
  - 이 순서로 정할 수 없거나, 보장 영역을 줄여야 할 때는 사용자에게 묻는다.

### 1.4 층과 파일
```text
sd.special (Python)          src/stardust/special/__init__.py  (_core.special 재노출)
      │
bind/special_bind.cpp        함수 등록, 에러 메시지 이름 (bind/errors.hpp)
      │
csrc/special/                ─────▶ array/ (NdIter, n항 디스패처, 버퍼 변환, 리덕션 드라이버)
                             ─────▶ parallel/ (풀), accelerate/ (vForce, §3.4)
```
- `special/`은 `array/`를 쓴다. `array/`는 `special/`을 모른다 (ARCHITECTURE §1).
- 파일 (제안)

  | 파일 | 내용 |
  |---|---|
  | `special/common.hpp` | 상수(π, √π, 오일러 상수, ε), 다항식·유리식 평가(Horner, Estrin), 계수 표 형식 |
  | `special/cf.hpp` | 연분수(수정 Lentz), 급수 합의 공용 반복기와 수렴 판정 |
  | `special/prefix.hpp` | `x^a e^{-x} / Γ(a)`, `x^a (1-x)^b / B(a,b)` 같은 앞 인자를 Lanczos로 계산 (§5.4) |
  | `special/ops/{erf,gamma,incgamma,incbeta,logistic,zeta,expint,bessel01,bessel,airy,ellip,hyper,lambertw,wofz,loggamma,distr,noncentral,testing}.hpp` | 계열마다 Op 구조체 (스칼라 정의) |
  | `special/kernels/simd.hpp`, `special/kernels/chunked.hpp` | Q5의 NEON 커널: 레인 둘과 스칼라가 함께 쓰는 연산(`simd`), 1024원소 청크 루프(`chunked`) |
  | `special/gamma_core.hpp`, `special/dd.hpp`, `special/halley.hpp`, `special/cplx.hpp`, `special/boost.hpp` | 감마 계열의 바닥(정확한 인자, `sinpi`, 가수·지수 분리), double-double, 역함수의 Halley, 복소수 보조 함수, Boost.Math 호출(§5.14) |
  | `special/reduce.cpp` | `logsumexp`, `softmax`, `log_softmax` |
  | `special/coeffs/*.hpp` | 가져온 계수 표. 파일 머리에 출처와 라이선스 |
  | `special/cost.hpp` | 함수마다 원소당 비용 (§3.3) |
- 테스트 `tests/special/`, 벤치마크 `bench/special/`, 기준값 생성 `tools/special_ref.py` (§6)

---

## 2. 의미론 규칙

### 2.1 입력과 dtype
- 피연산자는 `sd.ndarray`와 Python 스칼라다 (docs/api/array.md §6.2).
- bool, int64는 float64로 계산한다 (버퍼 변환, array.md §4.3). 결과는 float64다.
- complex128은 `TypeError`다. 마지막 마일스톤(S8) 뒤에는 §1.2의 복소수 함수만 complex를 받는다.
- 모두 Python 스칼라이면 결과는 Python `float`다.
- **정수 차수 인자**(`expn`의 `n`, `polygamma`의 `n`, Q17): int64와 float64를 받는다. 값이 정수가 아니거나 음수이면 그 원소는 NaN이다. scipy처럼 잘라 쓰지 않는다 (§2.2).

### 2.2 특수값
- 특수값(±0, ±inf, NaN, 극점, 정의역 경계)은 **scipy를 따른다.** 예: `gamma(0.0)`은 `inf`, `gamma(-0.0)`은 `-inf`, `gamma(-1.0)`은 `nan`, `rgamma(-1.0)`은 `0.0`, `logit(0.0)`은 `-inf`
- 다음 경우는 scipy를 따르지 않고 수학적 값을 쓴다. 해당 마일스톤을 시작하기 전에 목록을 사용자와 확정한다.
  - scipy가 실수로 표현할 수 없는 값에 `inf`를 낸다. 예: `hyp2f1(0.5, 0.5, 1.5, 2.0)`의 참값은 복소수인데 scipy는 `inf`다. 우리는 NaN이다.
  - scipy가 경고와 함께 인자를 바꾼다. 예: `expn(1.5, 1.0)`은 n을 1로 잘라 `E₁(1)`을 낸다. 우리는 NaN이다 (Q17).
- 함수별 특수값 표는 docs/api/special.md의 각 절에 둔다.

### 2.3 정확도 계약
- **보장 영역**: 함수마다 docs/api/special.md에 적는다. 그 안에서 참값(Arb, §6) 대비 상대오차 ≤ 1e-14다.
  - 1e-14는 float64로 약 45 ULP다. 실제 목표는 그보다 작고, 잰 최대 ULP를 표로 남긴다.
  - **진동 함수**(`j0 j1 y0 y1 jv yv jve yve`, x < 0의 `airy`, Q18): 영점 근처에서는 상대오차를 정의할 수 없다. 그래서 `|오차| ≤ 1e-14 × A(x)`를 보장한다. `A(x)`는 포락선(진폭)이다. 예: Hankel 점근의 진폭 `√(2/(πx))`, Airy는 `π^{-1/2}|x|^{-1/4}`. 포락선의 정확한 식은 함수마다 docs/api/special.md에 적는다.
- **느슨한 영역**: 보장 영역 밖이지만 정의역 안이다. 값을 계산하고, 잰 최대 오차를 문서에 적는다 (Q9).
- **오버플로·언더플로**: 참값이 float64 범위를 넘으면 ±inf, 서브노멀 아래면 0이다. 결과가 서브노멀이면 상대오차 대신 절대오차 2⁻¹⁰⁷⁴·45를 본다.
- 테스트의 허용치는 이 계약에서 나온다. scipy와의 차이는 허용치가 아니라 보고 대상이다 (§6).

### 2.4 scipy와 다르게 동작하는 곳
| scipy | sd.special | 이유 |
|---|---|---|
| 리스트, numpy 배열, numpy 스칼라를 받는다 | `sd.ndarray`와 Python 스칼라만 (numpy 스칼라도 받는다) | docs/api/array.md §6.2 |
| `out`을 위치 인자로도 받는다 | keyword-only | docs/api/array.md §1.1 |
| complex 입력을 대부분 받는다 | 실수만 (S8 뒤 일부) | Q6, Q7 |
| `logsumexp(a, axis, b, keepdims, return_sign)` | `logsumexp(x, /, *, axis=None, keepdims=False)` | Q13 |
| `erfc`, `ndtr`, `log_ndtr`의 서브노멀 꼬리가 0(−0.0) | 참값(서브노멀) | 서브노멀 규칙(§2.3), 사용자 결정 (S2, 2026-11-13) |
| `erfcinv(5e-324)`가 inf | 27.21 | z/2의 언더플로, 사용자 결정 (S2) |
| `erfinv(-0.0)`이 0.0 | −0.0 | 홀함수의 부호, 사용자 결정 (S2) |
| gamma·rgamma의 서브노멀 꼬리와 깊은 음수의 부호, `gammaln(±5e-324)`·`betaln(5e-324, 1)` = inf, `rgamma(-5e-324)` = −0.0, beta·betaln의 NaN 입력 → inf, `zeta(inf, q>1)` = NaN | 참값 | 사용자 결정 (S3, 2026-11-14). 같은 원칙으로 `beta(inf, 1e300)`, 거대한 k의 `binom` |
| `binom(n, k)`가 음의 정수 n에서 NaN | 정수 k ≥ 0이면 n(n−1)…(n−k+1)/k! | 사용자 결정 (S3) |
| `expi(x)`가 709.78 < x < 716.4에서 inf, `expn(n, x)`의 서브노멀 꼬리가 0, `expn`의 n이 int32를 넘으면 NaN | 참값 | 사용자 결정 (S5, 2026-11-14) |
| `expn(inf, x)`가 1.7e-10 | NaN (Q17: 정수가 아니다) | 사용자 결정 (S5) |
| `hyp1f1(-1, 2, inf)` = inf, `hyp1f1(0, 0, x)` = inf, `hyp2f1(a, b, c, -inf)` = NaN | 참값·극한 | 사용자 결정 (S7, 2026-11-14) |
| `lambertw`(−1/e에 가장 가까운 double) = NaN | −1 (분기점으로 본다) | 사용자 결정 (S7, Boost와 같다) |
| `erf(1e300+1e300j)` = NaN, `loggamma`의 무한대 인자 NaN, `lambertw(5e-324, k≠0)` = inf + nan·i, `lambertw(infj)` 등 NaN, `lambertw`(−1/e에 가장 가까운 double + 0j) = NaN | 참값·극한 | 사용자 결정 (S8, 2026-11-14) |
| `loggamma`의 극점이 nan + nan·i | inf + nan·i (허수부는 방향마다 다르다) | 사용자 결정 (S8) |
| `lambertw(x − 0j, k=-1)`이 −1/e < x < 0에서 실수(위쪽 값) | 아래쪽 극한(가지 자름의 ±0j는 C99식) | 사용자 결정 (S8) |
| 결과가 오버플로하는 곳의 ±inf 부호(반올림한 위상), `erf(5e-324+30j)` = inf | 정확한 위상의 부호, 참값 | 같은 원칙 (S8) |
| 실수축·허수축 위 0인 부분의 부호가 일관되지 않다(`erfc(0+0j)`은 −0, `erfc(0.5+0j)`은 +0) | 도함수의 방향 | S8 (docs/api/special.md §12) |
| Bessel·Airy의 ±inf 인자에서 NaN, `i0`·`i1`의 713–713.98 inf, `k0`·`k1`의 5e-324, 차수 극한(`yv(inf, x)` 등)의 NaN과 `yve`의 부호, 음의 차수 x = 0의 `iv(-1.5, 0)`·`yv(-½, 0)`·`yv(-1.5, 0)`, e-스케일 판의 x = 0 NaN, `airy(105)` | 참값 | 사용자 결정 (S6, 2026-11-14). `k0(5e-324)`는 같은 원칙 |
| `gammainc(1e-300, 5e-324)` > 1, `gammainc(5e-324, x)` = 0, `gammainc(1, 5e-324)` = 0, `betainc(2, 1e300, 5e-324)` = NaN, `gammaincinv(5e-324, 0.5)`·`betaincinv(2, 3, 5e-324)`·`betainccinv(2, 3, 5e-324)` = NaN, 근이 언더플로하는 `betaincinv`가 DBL_MIN | 참값 | 사용자 결정 (S4, 2026-11-14) |
| `expit(x)`가 x < −709.78에서 0 | 참값 eˣ(서브노멀), x < −745.13에서 0 | 서브노멀 규칙(§2.3), 사용자 결정 (S1, 2026-11-13) |
| `lambertw`는 언제나 complex를 돌려주고 `tol=`이 있다 | 실수 입력이면 float64. `tol=` 없음 | Q6, 결정성 (§5.11) |
| `airy`는 네 배열의 tuple | `AiryResult(ai, aip, bi, bip)` namedtuple | DESIGN §9.7 |
| 경고(`SpecialFunctionWarning`) | 내지 않는다 | API.md §2 |
| 별칭(`psi`, `jn`) | 없다 | Q12 |
| `bdtr(k<0)` NaN, `bdtr(k>n)` NaN, `pdtr(k<0)` NaN, `nbdtr(k, 0, p)` NaN | 참값 0, 1, 0, 1 | 참값·극한, 사용자 결정 (T1, 2026-11-15) |
| `bdtr`, `nbdtr`의 n을 내린다 | `bdtr`은 정수가 아니면 NaN, `nbdtr`은 실수 n | 사용자 결정 (T1) |
| `fdtr`, `fdtri`의 무한 자유도가 NaN | 극한 분포(카이제곱, 1의 점질량) | 참값·극한 (T1) |
| `stdtrit(3, 5e-324)`가 inf | −6.07e107(참값) | 참값 (T1) |
| `chndtr(x, df, nc)`, `chndtrix(p, df, nc)`: x가 맨 앞 | `chndtr(df, nc, x)`, `chndtrix(df, nc, p)`: x가 맨 뒤 | 다른 분포 함수와 일관성, 사용자 결정 (T2, 2026-11-15) |
| 비중심 분포의 위꼬리 함수가 없다 | `chndtrc`, `nctdtrc`, `ncfdtrc` | 꼬리의 상대 정확도, 사용자 결정 (T2) |
| `chndtr(0, nc, x)`, 무한 자유도·무한 nc의 `chndtr`·`nctdtr`·`ncfdtr`가 NaN | 참값·극한(0의 점질량, 극한 분포) | 참값·극한 (T2) |
| 비중심 분포의 꼬리가 0이나 틀린 값(최대 8.8e15 ULP), δ와 반대쪽 t 꼬리가 무의미한 값 | 참값 | 참값 (T2) |

---

## 3. 엔진

### 3.1 n항 디스패처 (배열 영역에 더한다, Q11)
- 지금 배열에는 `unary<Op>`, `binary<Op>`만 있다 (`array/ops/ufunc.hpp`). `where`는 자기 경로를 갖는다.
- 더하는 것: `nary<Op, N>(operands..., out)`. 입력 N개(1–4)의 디스패치 네 단계(피연산자, 루프, 변환, 실행)를 그대로 한다 (array.md §4.4).
  - `unary`, `binary`는 손대지 않는다. 기존 함수의 결과와 경로가 비트까지 같다 (골든 출력으로 확인).
  - `kMaxOperands` 4 → 5. `hyp2f1(a, b, c, z) → out`이 5개다. `NdIter`, `BufferedLoop`의 고정 배열이 하나씩 커진다. 이 비용(스택 몇십 바이트, 루프 하나)을 S0에서 잰다.
- 출력이 여럿인 원소별 함수(`airy`의 넷)는 출력 4 + 입력 1 = 5개다. 같은 디스패처가 출력 개수를 받는다.
- **S0에서 정한 모양**
  - Op 구조체가 `nin`, `nout`, 인자 이름 `args`를 갖는다. `f`는 `nout == 1`이면 값 하나, 아니면 `std::array`를 돌려준다.
  - 출력이 여럿이면 결과는 tuple이다(스칼라 입력이면 float의 tuple). `out=`은 없다.
  - 루프가 없는 dtype의 메시지는 입력 dtype을 모두 적는다. 셋 이상이면 `float64, complex128 and float64`처럼 쉼표로 잇는다 (둘일 때는 전과 같다).
  - `kMaxOperands` 5의 비용: 짧은 원소별 호출에서 행마다 ±5% 안팎으로 흔들리고 일관된 회귀는 보이지 않았다 (bench/special/s0/operands.md).

### 3.2 Op 구조체
- 배열의 `scalar_ops.hpp`와 같은 형태다: 루프가 있는 저장 타입(`ok<T>`, 여기서는 `double`만)과 원소 하나의 함수(`f`).
- 원소 함수는 `static double f(double...)`이다. 순수 함수다(전역 상태 없음, `errno` 안 봄).
- 이 형태라서 `elementwise.hpp`의 연속 fast path(array.md §5.1)가 그대로 붙는다.

### 3.3 병렬화: 원소당 비용
- 함수마다 원소당 비용이 100배 넘게 다르다 (xlogy 2.5ns, betaincinv 744ns, scipy 1스레드 기준).
- 그래서 배열의 기준값(`kParallelThreshold`, 원소 2¹⁵개)을 그대로 쓰지 않는다. 함수마다 원소당 비용 `c`(ns)를 `special/cost.hpp`에 두고, 일의 양을 `n·c`로 센다. linalg의 비용 모델(linalg.md §7)과 같은 생각이다.
  - 배열의 이터레이터는 이미 작업량 가중치를 받는다 (레인 처리, array.md §4.5). 같은 장치를 쓴다.
  - S4에서 더함: 가중치의 상한 `kMaxWeight` = 256. 이터레이터는 태스크를 4096/가중치 원소로 자르는데, 가중치 800–8500인 불완전 함수는 태스크 하나가 원소 하나가 되어 10스레드가 4.8–7.8배에 그쳤다. 상한 256(태스크 16원소)에서 8.1–8.9배다 (bench/special/s4/threads.md).
  - `c`는 S0의 측정 도구로 함수마다 재고, 입력 분포는 보장 영역에서 고르게 뽑는다.
  - `cost.hpp`의 값은 각 마일스톤에서 잰 값이고 §5의 속도 개선 뒤에 다시 재지 않았다. 48.6 ns(가중치 256) 이상인 값은 상한에 걸려 분할이 같으므로, 무거운 함수의 값이 달라져도 영향이 없다.
  - 가중치의 단위는 연속 float64 `add`의 원소당 비용이다 (`kAddNs`, 1스레드에서 0.18–0.19ns). 가중치 `w`인 함수는 원소 `2¹⁵/w`개부터 병렬이 되고, 태스크 하나는 원소 약 `2¹²/w`개다 (`bench/special/cost.py`).
- 원소마다 반복 횟수가 다르다(연분수, Halley 반복). 정적 분할이면 스레드마다 일이 달라진다. 풀의 청크 분할(array.md §10)이 이것을 얼마나 흡수하는지 S4에서 잰다.

### 3.4 NEON (Q5)
- 대상(청크 커널, `Op::chunk`): `erf erfc ndtr log_ndtr ndtri erfinv erfcinv`(`ops/erf.hpp`)와 `expit logit`(`ops/logistic.hpp`). `erfcx`와 `gammaln`은 스칼라 libm 정의다.
- 기법
  - 다항식 평가: Horner(곱셈-덧셈이 차례로 이어짐)와 Estrin(트리, 병렬도 높음)을 비교한다. `-ffp-contract=off`이므로 FMA는 코드에 적은 곳에서만 생긴다 (array.md §1.3). 스칼라 정의도 같은 식을 써야 비트가 같다.
  - 분기 없애기: 두 구간을 모두 계산한 뒤 `vbslq`로 고른다. 구간이 셋이면 셋 다 계산하는 비용과 분기 비용을 잰다.
  - S3: `gammaln`은 NEON으로 짜지 않았다. 구간이 넷(작은 x, Taylor, 점화, Stirling)이고 S2의 측정(분기 방식이 빠르거나 같음)으로 보아 레인 둘의 이득이 없다. 스칼라 libm 정의다.
  - **S2의 측정 결과** (bench/special/s2/variants.md): 차수 5–8의 유리식이 셋인 `erfc`와 `ndtri`에서는 원소마다 자기 구간만 계산하는 스칼라 분기가 NEON 고르기보다 빠르거나 같았다(구간이 무작위로 섞인 입력만 예외, 4.7 대 4.2 ns). 고르기용 상수와 남의 구간 계산이 레인 두 개의 이득보다 크다. 그래서 S2 함수는 1단계(exp·log의 인자)만 NEON, 2단계는 vForce, 3단계는 스칼라 분기다. 결과의 비트는 두 방식이 같다. Estrin은 Horner와 속도가 같고 오차가 조금 컸다(9.1 대 7.8 ULP).
- **안의 `exp`/`log`** (Q16): 커널이 입력을 1024원소(8KB) 스택 버퍼 단위로 처리한다.
  1. NEON으로 `exp`/`log`의 인자(예: `-x²`)를 버퍼에 만든다.
  2. vForce(`vvexp`, `vvlog`)를 버퍼 전체에 한 번 부른다.
  3. NEON으로 다항식을 계산하고 결과를 합친다.
  - 스칼라 입력과 비연속 레이아웃도 같은 버퍼 경로를 탄다 (n = 1 포함). 그래서 `f(x)[i] == f(x[i])`가 성립한다. 배열의 vForce 규칙과 같다 (array.md §5.4).
  - vForce의 오차(libm 대비 ≤ 6 ULP)는 정확도 계약(약 45 ULP) 안이다. 특수값(±0, ±inf, NaN)이 vForce에서 부호까지 맞는지는 S1에서 그 함수들에 대해 다시 확인한다.
  - 분기가 많은 나머지 함수는 원소마다 libm `exp`/`log`를 부른다.
- 그 밖의 함수는 손으로 짜지 않는다. 연속 fast path의 자동 벡터화와 스레드만 쓴다.

### 3.5 리덕션 (`logsumexp`, `softmax`, `log_softmax`)
- 정의: `logsumexp(x) = m + log(Σ exp(x - m))`, `m = max(x)`
- 두 패스로 한다: 1) `max` 2) `exp(x - m)`의 pairwise 합 (base/pairwise.hpp). 배열의 리덕션 드라이버와 블록 분할(array.md §6.5)을 쓴다. 결과는 스레드 수와 무관하다.
  - `axis`는 배열 리덕션의 규칙 그대로다(`sd.max`, `sd.sum`에 넘긴다).
  - `logsumexp`는 `max`를 두 번 부른다: `x`와 브로드캐스트할 `keepdims=True`의 m, 결과에 더할 요청한 `keepdims`의 m.
  - `x - m`은 TwoSum으로 hi + lo를 만들고 `exp(hi)·(1 + lo)`로 계산한다. 빼기를 반올림하면 exp에서 |x − m|·2⁻⁵³(−700에서 7.8e-14)을 잃는다.
- 한 패스 방식(온라인 softmax: 최댓값이 바뀔 때마다 누적값을 다시 척도 조정)은 메모리 왕복이 한 번 줄지만 반올림이 늘어난다. 두 방식을 해설에서 비교하고, 공개 함수는 두 패스를 쓴다.
- `m`이 ±inf일 때: 모두 `-inf`이면 결과는 `-inf`, `+inf`가 있으면 `+inf`, NaN은 전파한다 (scipy와 같다).
- `softmax`는 `exp(x - logsumexp(x))`가 아니라 `exp(x - m) / Σ`로 계산한다. 나눗셈 하나가 log·exp 왕복보다 정확하다.

---

## 4. 수치 기법의 선

### 4.1 근사 계수 (Q4)
- 유리 근사와 Chebyshev 계수는 공개된 것을 가져온다. 출처: Cephes(scipy가 BSD로 포함), Boost.Math(Boost Software License), 논문(AS241 등)
- `special/coeffs/*.hpp`의 머리에 출처, 판, 라이선스, 원 논문을 적는다. 계수는 원본의 십진 표기를 그대로 옮긴다(비트 확인 테스트).
- 해설은 계수가 어떻게 만들어졌는지(minimax, Remez 교대 정리)를 설명하되 직접 만들지는 않는다.

### 4.2 구간 한도 (Q8)
- **세는 것**: 근사 방식이 바뀌는 갈래. 예: 거듭제곱급수 ↔ 연분수 ↔ 점근전개, 다른 계수 표로 넘어가는 경계
- **세지 않는 것**: 정의역을 옮기는 변환. 예: 반사(`Γ(x)Γ(1-x) = π/sin πx`), 점화식(`Γ(x+1) = xΓ(x)`), 대칭(`I_x(a,b) = 1 - I_{1-x}(b,a)`), Kummer·Pfaff 변환, hyp2f1의 선형 변환
- **퇴화 경우**: 일반 공식이 0/0이 되는 매개변수(예: hyp2f1의 `c-a-b` 정수)는 극한 공식 하나로 처리한다. 정수 거리마다 따로 짜지 않는다.
- **가져오지 않는 것**: Boost·Cephes가 극단 영역마다 둔 특수 분기(예: DiDonato–Morris BRATIO의 십여 갈래), 상수 미세 튜닝, 매개변수 범위마다의 fast path

### 4.3 공통 기법 (해설의 주제)
- 보수함수를 따로 계산한다: `erfc`, `gammaincc`, `betaincc`는 `1 - f`가 아니다. 꼬리에서 상대 정확도가 유지된다.
- log 공간: `gammaln`, `betaln`, 앞 인자의 log 형태로 오버플로와 소거를 피한다.
- Lentz 연분수: 분모가 0에 가까울 때를 작은 수로 바꿔 계속한다. `b0 = 0`이면 그 작은 수(2⁻¹⁰⁰⁰)가 결과의 척도가 되므로, 꼬리를 따로 계산해 나눈다 (S0에서 `_cf_tan`의 5e-324가 9e-302로 나와 찾았다).
- 역함수: 좋은 초기값 + Halley(또는 Newton). 도함수가 앞 인자(§5.4)라서 거의 공짜다.
- 정확한 빼기: `2p - 1`(p ∈ [0.25, 1]에서 Sterbenz 보조정리로 정확), `x²`를 두 부분으로 나눠 `exp(-x²)`의 오차 줄이기

---

## 5. 함수별 계획 (후보와 구간)

구간 수는 Q8로 센 수다. 후보가 둘이면 §1.3의 규칙으로 고른다.

### 5.1 오차함수
- `erf`, `erfc`: Cephes `ndtr.c`의 세 구간. `|x| < 1`은 erf 유리식, `1 ≤ x < 8`과 `x ≥ 8`은 `exp(-x²)·P/Q`, `exp(-x²)·R/S`. 음수는 대칭. `exp(-x²)`는 `x²`를 나눠 계산한다 (Cephes `expx2`). **3구간**
- `erfcx(x) = exp(x²) erfc(x)`: 같은 유리식에서 `exp(-x²)`만 빼면 된다(`x ≥ 1`). `x < 1`은 `exp(x²)·erfc(x)` 그대로, 큰 x는 점근 `1/(x√π)`. 음수는 `2exp(x²) - erfcx(-x)` (변환). **3구간**. Faddeeva 패키지의 100구간 Chebyshev는 쓰지 않는다.
- `ndtri`: Wichura AS241 (PPND16). 중심 `|q| ≤ 0.425`의 유리식, 꼬리 `r = √(-log p)`의 유리식 둘. 정확도 약 1e-16. **3구간**
- `erfinv(y)`, `erfcinv(z)`: ndtri의 핵심을 `q`로 직접 부른다. `erfinv(y)`는 `q = y/2`(정확)로 중심식을 써서 작은 y에서 `(1+y)/2`의 소거가 없다. `erfcinv(z) = -ndtri(z/2)/√2`
- `ndtr(x) = erfc(-x/√2)/2`. `log_ndtr`: `x > 6`은 `log1p(-ndtr(-x))`, 중간은 `log(ndtr(x))`, `x < -20`은 점근급수. **3구간**
- **S2에서 바꾼 것**
  - `exp(-x²)`는 Cephes `expx2`의 지수 두 번 대신 `hi = x·x`, `lo = fma(x, x, −hi)`로 `exp(−hi)·(1 − lo)`다. `ndtr`은 `t = x/√2`의 반올림을 피해 `exp(−x²/2)`를 x에서 직접 만든다.
  - `erfcx`는 `x ≥ 8`을 Cephes R/S 대신 점근급수(17항)로 한다. R/S는 x = 26.64까지 맞춘 식이라 x ≈ 10⁴에서 21 ULP가 났다. 구간은 셋(x < 1, P/Q, 점근)이다.
  - `log_ndtr`는 두 구간이다: `x ≥ 0`은 `log1p(−ndtr(−x))`, `x < 0`은 `−x²/2 + log(erfcx(−x/√2)/2)`. 두 항의 부호가 같고 erfcx는 언더플로하지 않아서 점근급수가 필요 없다.
  - `erfinv`의 중심 구간은 `y·(비율/(2√2))`다. `q = y/2`를 먼저 만들면 서브노멀 y의 마지막 비트가 사라진다. `erfcinv`의 꼬리는 `log(min(z, 2−z)) − log 2`다.

### 5.2 감마
- `gamma`: 후보 A Lanczos(Boost `lanczos13m53`, g ≈ 6.0247, 13항), 후보 B Cephes(구간 [2, 3]의 유리식 + 점화식, x > 33은 Stirling). `x < 0.5`는 반사. 오버플로는 x > 171.62
- `gammaln`: 1과 2 근처에서 값이 0이 되어 상대 정확도가 어렵다. Cephes `lgam`(x < 13은 [2, 3]으로 옮겨 `(x-2)·P/Q`, 그 위는 Stirling급수)이 이 문제를 식의 모양으로 푼다. **2구간**. 음수는 반사(`log|Γ|`)
- `rgamma = 1/Γ`: 극점에서 0. 반사와 Lanczos
- `digamma`: `x < 0` 반사(`π cot πx`), 작은 x는 점화식으로 올린 뒤 Bernoulli 점근급수, 양의 근 x₀ ≈ 1.4616 근처는 근을 중심으로 한 급수(상대 정확도). **2구간**
- `polygamma(n, x)`: `n = 0`은 digamma, `n ≥ 1`은 `(-1)^{n+1} n! ζ(n+1, x)` (§5.6의 Hurwitz zeta)
- `beta`, `betaln`: Lanczos 비(Boost `beta`)로 세 Γ를 한 식에 모아 오버플로와 소거를 피한다. 큰 인자는 log 형태
- `binom(n, k)`: 정수 k가 작으면 곱(정확), 그 밖은 `1/((n+1)·B(n-k+1, k+1))`. 음의 n은 반사
- **S3에서 정하고 바꾼 것**
  - `gamma`: 후보 B(Cephes)를 골랐다. 최대 4.4 대 6.5 ULP(x > 0), 3.4배 빠르다(§1.3의 규칙). A(Lanczos)는 오버플로를 넘는 scaled 경로(서브노멀 꼬리, beta, 음수 zeta, polygamma)에 쓴다. 공개 `gamma`는 x > −170(극점 제외)에서 Boost를 먼저 부르고(§5.14), Cephes 식은 Boost의 결과가 정상 범위가 아닐 때와 `rgamma`·`beta` 안에서 쓴다.
  - 정확한 인자: 반사식은 `Γ(x)Γ(−x) = −π/(x·sinpi(x))`, `Γ(x+1) = x·Γ(x)`. 꼭 만들어야 하는 합(a+b, n−k+1)은 TwoSum으로 hi + lo를 들고 `Γ(hi+lo) ≈ Γ(hi)(1 + ψ(hi)·lo)`. `sinpi`·`cospi`는 인자를 정확히 줄인다.
  - `gammaln`(x > 0): 2 둘레의 Taylor 전개(계수는 ζ(k)−1, 28항)와 Stirling 두 근사. 1 근처는 `lgamma(2+e) − log1p(e)`. Cephes의 [2, 3] 유리식은 쓰지 않는다. **2구간**
  - `beta`: a+b ≤ 170은 Cephes Γ 셋, 그 위는 Γ를 가수·지수로 나눈 scaled 곱, 한 인자가 7000을 넘으면 Γ 비율 `L(a)/L(c)·e^E·cgh^{−b}`(E가 작게 남도록 정리)
  - `zeta`: Riemann x ≥ 0은 Euler–Maclaurin(q = 1), x < 0은 함수방정식(scaled). Hurwitz는 Cephes `zeta.c`. q^{−s}가 언더플로하면 scaled로 더한다(polygamma). 공개 Riemann `zeta`는 x ≥ −1과 음의 홀수 정수 −3, …, −59(Boost가 Bernoulli 수의 표에서 읽는다)에서 Boost를 먼저 부르고, 2–52의 짝수는 올바르게 반올림한 표, 음의 짝수는 0이다(§5.14). 정수가 아닌 −60 < x < −1은 우리 함수방정식을 double로 계산한다(`zeta_detail::reflect`: 1 − x를 반올림하지 않고 넣고, 두 지수 함수를 하나로 묶어 그 반올림을 1차 보정으로 붙인다). Boost는 그 구간에서 1 − x를 반올림한 뒤 Γ에 넣고 반올림한 2π를 거듭제곱해서 x ∈ (−32, −31)에서 65 ULP였다. x ≤ −60은 인수마다 scaled로 계산한 함수방정식이다. x ≥ 0의 Euler–Maclaurin은 Boost의 결과가 정상 범위가 아닐 때만 쓴다.

### 5.3 불완전 감마 (`gammainc`, `gammaincc`)
- `x < a + 1`: 거듭제곱급수로 P, `x ≥ a + 1`: Lentz 연분수로 Q. 나머지는 보수
- 큰 a, `x ≈ a`(a > 20, |x − a| < 0.4a)는 급수와 연분수가 모두 느리고 소거가 생긴다. Temme 균등 점근전개를 쓴다 (DESIGN 1라운드 Q3의 예). **3구간**
- 앞 인자 `x^a e^{-x}/Γ(a)`는 §5.4

### 5.4 앞 인자 (`special/prefix.hpp`)
- `x^a e^{-x}/Γ(a)`를 따로 계산하면 a가 크면 넘친다. Lanczos 형태로 `(x/(a+g-½))^a · e^{a-x}·…`처럼 한 식으로 모은다 (Boost `regularised_gamma_prefix`). 베타도 같다 (`ibeta_power_terms`)
- 이 값은 불완전 함수의 도함수이기도 하다. 역함수의 Halley 반복이 이것을 다시 쓴다.

- **속도 (2026-11-15, 최적화 1단계)**: 앞 인자가 `gammainc` 시간의 3/4였다(174 ns 중 132 ns). 정밀도는 그대로 두고 계산만 바꿨다.
  - **표 기반 로그**: λ = 2^k·c·(1 + r), c = i/128(97점, `coeffs/dd_log_table.hpp`, `tools/gen_dd_log_table.py`), |r| ≤ 2⁻⁷·⁵⁸. log(1 + r)는 r², r³까지 hi + lo, 나머지는 float64다. 31 ns → 7 ns. Arb 대비 상대오차(6만 점)는 최대 log 2⁻⁷⁷·⁶(급수판 2⁻⁶⁴·⁶), mu 2⁻⁶⁸·⁹(급수판 2⁻⁶²·⁴)로 오히려 좋다. 급수판은 비정상 입력(서브노멀 등)의 뒷길로 남겼다.
  - **나눗셈**: double-double 몫을 나눗셈 두 번과 fma 하나로 줄였다(세 번째 보정항은 지수에 쓸모가 없었다).
  - **a < 13의 앞 인자**: Stirling 꼴을 거치지 않고 e^(a log x − x)/Γ(a+1)로 바로 구한다(지수는 double-double).
  - **지수 분리 생략**: 결과가 정상 범위(|E| < 700)이면 `Scaled`의 frexp·ldexp를 건너뛴다.
  - 결과: 앞 인자 132 → 38 ns, `gammainc` 핵심 174 → 74 ns. 순방향 값은 최대 4 ULP 달라졌고 정확도 표는 같거나 좋아졌다. 역함수의 출력은 비트까지 같다. 측정은 bench/special/opt1/.

### 5.5 불완전 베타 (`betainc`, `betaincc`)
- `x > (a+1)/(a+b+2)`이면 대칭으로 옮긴다 (변환). 기본은 Lentz 연분수 (NR `betacf`). b·x가 작거나 b < 1이면 거듭제곱급수
- 큰 a, b(예: 둘 다 > 10⁴)에서 연분수의 반복이 `O(√max(a,b))`로 는다. 세 번째 구간(Temme 균등 점근전개)은 S4에서 정확도와 속도를 잰 뒤 넣을지 사용자에게 묻는다. **2–3구간**
- 역함수(`gammaincinv`, `gammainccinv`, `betaincinv`, `betainccinv`): 초기값 2–3가지(감마: Wilson–Hilferty, 작은 p의 거듭제곱 근사; 베타: 정규 근사, 작은 x의 거듭제곱 근사) + Halley 반복. 반복은 상대 변화가 ε 아래가 되거나 상한(예: 20회)에서 멈춘다. DiDonato–Morris의 초기값 열 가지 남짓은 가져오지 않는다.
- **S4에서 정하고 바꾼 것**
  - 앞 인자: Lanczos 대신 Stirling 정규화 `Γ(z) = √(2π/z)(z/e)^z Γ*(z)`로 큰 부분을 지수 하나 E = a·μ(x/a − 1)(μ(t) = t − log(1+t))에 모았다. E는 정확한 입력에서 double-double로 만든다(`special/dd.hpp`). scipy(Cephes)는 a log x − x − lgamma(a)를 float64로 만들어 꼬리에서 수천 ULP를 잃는다.
  - 불완전 감마: 네 갈래다. x < 1.1의 작은 a는 Boost의 `tgamma_small_upper_part`(Γ(1+a) − 1과 x^a − 1을 expm1로), 급수 P, 연분수 Q, a > 20이고 |x − a| < 0.4a이면 Temme. Temme의 c_k(η) 계수는 DLMF 8.12의 정의식을 정확한 유리수로 전개해 만들었다(`tools/gen_temme_igamma.py`, 맞춘 값이 아니다).
  - 연분수는 Lentz로 항 수만 찾고 값은 꼬리부터 거꾸로 계산한다(`cf.hpp` `lentz_backward`). Lentz 그대로는 60항에서 17 ULP를 잃었다.
  - 불완전 베타: NR `betacf` 대신 DiDonato–Morris의 BRATIO(TOMS 708) 구조를 따른다. BPSER(급수), BUP(유한합으로 a·b 옮기기), BGRAT(큰 a·작은 b, x ≈ 1), BFRAC(평균 아래 연분수), BASYM(a, b > 100의 평균 근처, Temme형 균등 점근전개)이다. BGRAT와 BASYM은 측정 뒤 사용자 결정으로 넣었다(2026-11-14). 근사 방식이 다섯이라 §1.1 Q8의 "2–3구간"을 넘는다.
  - 역함수: Halley는 F와 목표가 두 배 넘게 차이 나는 동안 log F에 대한 Newton 걸음을 쓴다(꼬리에서 선형 걸음은 한 번에 F/F′ ≈ 1씩만 움직인다). 감마의 초기값은 하한 둘(작은 x 근사, 큰 x 점근식)과 Wilson–Hilferty, 베타는 AS 109다.

- **속도 (2026-11-15, 최적화 2단계)**:
  - **1 < a, b이고 b0 < 40인 가지를 연분수(bfrac)로 바꿨다.** TOMS 708은 여기서 bup으로 b를 (0, 1]까지 내린 뒤 bpser나 bgrat을 쓴다. 이 길은 앞 인자를 세 번 만든다. 연분수는 한 번이면 되고 더 정확했다: a, b가 6–40에서 330 → 157 ns, `bdtr` 10.1 → 4.9 ULP, `nbdtrc` 19.4 → 8.8, `fdtrc` 18.1 → 10.7, `ncfdtrc` 22.6 → 15.2.
  - **앞 인자는 쓰는 가지만 만든다.** 전에는 도함수(역함수만 쓴다) 때문에 언제나 만들었다. bgrat과 basym은 쓰지 않는다.
  - **연분수의 뒤로 가는 계산**(cf.hpp)은 앞으로 갈 때 만든 항 64개를 다시 쓴다.
  - 남은 것: a, b ≥ 100이고 평균 근처인 basym(계수 점화식이 O(n³), scipy의 0.5배)과 작은 a·큰 b의 bgrat(0.7배). 측정은 bench/special/opt2/.

- **작은 모양과 아주 작은 x (2026-11-17, stats 최적화 중 Arb로 찾음)**: a, b ≤ 1에서 BRATIO는 x^a ≤ 0.9이면 멱급수로 I를 구하고 1 − I를 빼기로 만든다. I가 0.9에 이르면 급수 오차가 9배가 된다(우리 9.8e-15, 같은 규칙을 쓰는 Boost 6.0e-15). 그래서 I > 1/2이면 1 − I를 BRATIO의 다른 두 방법으로 직접 구한다. 급수의 첫 인자만으로 I > 1/2 + 2⁻⁴⁰임을 알면(b ≤ 1에서는 모든 항이 양수라 하한이다) 급수를 돌리지 않는다. x < 2⁻¹⁰⁰⁰은 log x를 받는 `ibeta_tiny`가 2⁻¹⁰⁰⁰에서의 값에서 양의 항만으로 내려간다(`fdtr`·`fdtrc`의 아주 작은 인자도 이것을 쓴다; 전에는 1 − i로 1.3e-13까지 틀렸다). x가 0으로 언더플로한 인자는 (0, 1)이다. 한 모양이 2⁹⁰⁰ 이상이면 감마 극한 I_x(s, c) = P(s, c·x)를 쓴다(`gamma_limit`; 전에는 `fdtr(2, 1e308, 1)`이 1.0, 참값 0.632). Boost의 값은 a, b ≤ 6, x ≥ 2⁻¹⁰²², 결과 ≥ 2⁻⁹⁶⁰일 때 쓴다: 서브노멀 x에서 Boost는 25%까지 틀리고, 결과가 2⁻¹⁰¹⁶ 아래이면 2.2e-13까지 틀린다. 그 안에서도 Boost가 빼기로 만드는 쪽은 부르기 전에 가려 우리 것으로 보낸다(`boost_subtracts`: a, b ≤ 1, 보수 쪽, a0 < b0, x0 < 0.02, 0.6 < x0^a0 ≤ 0.9. 우리가 쓰는 Boost는 x0 ≥ 0.02에서 연분수라 빼지 않는다. 0.6은 I ≤ x0^a0/y0에서 나온다). 6만 점에서 Boost가 남는 곳은 1.7e-15, 넘긴 곳은 Boost 6.8e-15 대 우리 1.3e-15다. 큰 지수의 e^(−E)(1 − E.lo)는 E ≥ 2⁵²에서 +0이다(전에는 −0.0, E ≈ 1e300부터 NaN: `betainc(1e300, 2, 0.5)`).
- **극단 인자 정리 2차 (2026-11-18, 사용자 결정 "전부 값 또는 NaN으로")**: 극단 인자 점검에서 남았던 틀린 값 255점을 다시 재현해 83점으로 줄였다(Airy와 복소 erf·wofz를 빼면 218 → 38). 남은 것은 아래 '남은 결함'과 `hyp1f1`·`hyp2f1` 절에 적었다.
  - `binom`: n + 1을 반올림한 뒤 k의 거듭제곱에 넣어 204 ULP를 잃던 것과, n ≥ 0에서 큰 음의 비정수 k가 0 대신 ±1·±inf를 내던 것을 고쳤다. n ≥ 0의 거대 인자 가지는 인자를 정확히 넣는 식으로 다시 썼다. 무작위 8000점에서 최대 8 ULP, 새 NaN은 없다.
  - 서브노멀 모양: a, b < 2⁻⁷⁰⁰이면 `betainc`는 닫힌 꼴 b/(a + b)다(Boost는 1.0). `fdtr`·`fdtrc`·`chdtrc`의 df = 5e-324는 0이나 NaN 대신 값이다. `stdtr(0.9, -1.8e308)`은 130 → 0.8 ULP, `betaln(1e300, 1.8e308)`은 −inf → −2.0007e301이다.
  - 비중심: t 함수는 |nc| < 2⁻⁵⁰⁰이면 중심 함수로 간다(`nctdtr(0.5, -5e-324, -1e300)`이 10% 틀렸다). F의 먼 꼬리에서 구한 쪽의 보수가 1/64 아래이면 NaN이다. 역함수는 nc ≠ 0이고 p가 서브노멀이면 NaN이다(Arb도 참값을 정하지 못했고, 정방향 검증으로 틀린 것만 알았다).
  - 보장 영역에서 함께 찾은 것: `hyp1f1(0.5, 1000, -1e5)`가 4200 ULP(log Γ의 차) → 1.2 ULP, a = 1e5·b < 11의 Boost 값 83 ULP → 우리 식, `hyp2f1`의 큰 a에서 Pfaff 변환이 NaN이나 1e-12 오차를 내던 것, 넘치는 양의 합이 inf 대신 NaN이던 것.
  - `hyp1f1` 급수의 뿌리 원인: 상쇄 추정을 보정하지 않은 합으로 해서 1e17 근처에서 포화했고, 1e13 한계가 걸리지 않았다. 느슨한 영역은 이제 우리 급수(1e13을 지킬 때), b > 0의 a 점화식(double-double), 큰 |x| 전개, Boost(|a|, |b| ≤ 100만), Tricomi 순으로 시도하고 모두 안 되면 NaN이다. 느슨한 영역의 호출은 느려졌다(표본 점에서 505 → 825 ns; 속도표의 행은 아니다).
  - `hyp1f1` 보장 영역의 Boost 구간을 0 ≤ x < 16으로 줄였다(사용자 결정 2026-11-18: 보장을 넘는 곳은 우리 급수로). b < 11, a ≤ 100에서 쓰던 Boost 값은 x ≥ 18부터 4%의 점이 45 ULP를 넘고 x ≈ 500에서 297 ULP였다(무작위 4만 점, a 1e-2–100, b 1e-3–11, x 1–690). 우리 급수는 같은 점에서 최대 48 ULP(x ≈ 650, 상대 1e-14 안)이고 그 밖은 20 ULP 아래다. 정확도표의 보장 행은 78.6 → 15.9 ULP가 됐다. 속도표 입력에서는 1.2%의 점이 이 구간이라 1스레드가 2.8% 느려졌다(A/B 다섯 번 교대, 236–239 → 243–248 ms).
  - `hyp2f1`: 급수가 지수를 따로 들고, c − b와 z/(z − 1)을 정확히 넣으며, 1 − z 변환은 스케일한 Γ와 상쇄 검사(1024배 초과면 NaN)를 쓴다.
  - 속도(C++ A/B, 정확도 입력, 1스레드): 모든 행이 ±1% 안이고 `hyp2f1` +2.4%, `chndtrix` +2.1%다. 정확도표는 106행 가운데 104행이 같고 느슨한 두 행이 좋아졌다(`hyp1f1` 2.2e3 → 1.17e3, `hyp2f1` 3.07e3 → 2.73e3).
- **남은 결함** (고치지 않음): b ≳ 1e200에서 `ibeta(14.38, 1e200, 1.5e-199)`가 NaN이고, `betaincinv(14.38, 1e200, 0.617)`이 1 − p의 근을 낸다. 2026-11-18 뒤에 남은 것: `hyp1f1` 느슨한 영역의 Boost 최대 1.2e-12·Tricomi 9e-12, `hyp2f1` 느슨한 영역 1.7e-12(z < −9, |c| > 1e4), `nctdtr`·`nctdtrc`의 df ≤ 1e-3에서 2.4e-13(23점), `fdtri(14.38, 1e-300, 5e-324)` 59 단위. 점검 점 가운데 8370점은 Arb 참값이 정해지지 않았고 6798점은 시간 초과라 검증하지 못했다.
### 5.6 제타와 지수적분
- `zeta(x, q)`: Hurwitz, Euler–Maclaurin (Cephes `zeta.c`: 앞 N항 직접 합 + Bernoulli 꼬리). Riemann(`q` 생략)의 `x < 1`은 함수방정식(변환). Hurwitz의 `x < 1`은 NaN (scipy와 같다)
- `expn(n, x)`: `x ≤ 1` 거듭제곱급수, `x > 1` Lentz 연분수 (NR `expint`), 큰 n은 균등 점근. **3구간**
- `expi(x)`: `x < 0`은 `-E₁(-x)` (변환), `0 < x < 40`은 거듭제곱급수(양의 항만이라 소거 없음), 큰 x는 점근급수. **2구간**. Cephes `ei.c`의 여섯 구간 유리식은 쓰지 않는다.
- **S5에서 정하고 바꾼 것**
  - `expn`: 큰 n의 균등 점근은 넣지 않았다. 연분수의 b = x + n이 크면 곧바로 수렴하고, 급수는 x^k/k!로 줄어서 n = 1e5까지 6.3 ULP다. **2구간**
  - `expi`: 양의 0점 x₀ = 0.3725… 둘레 [0.3, 0.45]에 Taylor 급수를 더했다(정확한 d = x − x₀, 계수는 `tools/gen_ei_taylor.py`). 거듭제곱급수는 거기서 γ + log x와 상쇄해 scipy가 5e15 ULP를 잃는다. **3구간**
  - 공개 `expi`는 x ≥ −700에서 Boost를 먼저 부른다(§5.14). 위의 세 구간(x > 0)은 Boost의 결과가 정상 범위가 아닐 때만, −E₁(−x)는 그때와 x < −700(서브노멀 꼬리)에서 쓴다.

### 5.7 Bessel 0·1차 (`i0 i1 k0 k1 j0 j1 y0 y1`와 e-스케일 판)
- `i0 i1 k0 k1`: Cephes Chebyshev 전개, 함수마다 **2구간** (x ≤ 8 / x > 8 또는 x ≤ 2 / x > 2). e-스케일 판은 같은 전개에서 `exp`를 빼면 된다.
- `j0 j1 y0 y1`: Cephes. `x ≤ 5`는 유리식, `x > 5`는 Hankel 점근(진폭과 위상의 유리식). **2구간**. Cephes는 첫 영점들을 인수로 빼서(`(x² - r₁²)…`) 영점 근처의 상대 정확도를 얻는다. 우리는 이것을 쓰지 않는다. 정확도는 포락선 기준이다 (Q18, §2.3)

### 5.8 Bessel 임의 실수 차수 (`jv yv iv kv`와 e-스케일 판)
- Temme 방법 (NR `bessjy`, `bessik`):
  - 차수를 `ν = n + μ`, `|μ| ≤ ½`로 나눈다.
  - `x < 2`: Temme 급수로 `Y_μ`, `Y_{μ+1}` (`K_μ`, `K_{μ+1}`)
  - `x ≥ 2`: Steed 방법(CF2)으로 같은 값
  - CF1(Lentz)로 `J_{ν+1}/J_ν` 비, Y·K는 앞으로 점화, Wronskian으로 J·I를 맞춘다.
  - 큰 x(x > max(25, ν²)): Hankel 점근 (CF1의 반복이 x에 비례해 늘기 때문)
  - **3구간**. 큰 ν의 Debye 균등 점근은 넣지 않는다. 큰 ν는 맞지만 O(ν)로 느리다. 느슨한 영역에 시간도 적는다.
  - S10에서 바꿈(코드리뷰), S11에서 넓힘: `iv kv ive kve`는 ν ≥ 30에서 Debye 균등 점근전개(DLMF 10.41.3–4, U₀–U₁₀, `coeffs/debye_u.hpp`, `tools/gen_debye.py`; 다음 항은 ν = 30에서 2e-16)를 쓴다(S10에서는 ν ≥ 1e4, U₀–U₄). ν = 100–1e4에서 원소당 14–121 µs이던 것이 약 0.15 µs가 됐다. O(ν) 점화식은 `iv(1e9, 1e9)`에 26초가 걸리고 CF1의 항 수 상한(5e7)에 걸려 0을 냈다(참값은 inf). 지수 νη(스케일 판은 νη ∓ x)는 ν와 s = √(ν² + x²)가 크게 상쇄하므로 double-double로 만들고 한 번만 지수를 취한다. ν = 1e4–4e4에서 점화식과 7e-16 안으로 일치한다. `jv yv`의 큰 차수는 그대로 O(ν)다.
- 음의 ν는 반사 공식(변환). 음의 x는 정수 ν에서만 정의(scipy와 같다), 그 밖은 NaN

### 5.9 Airy (`airy`)
- 작은 |x|(≤ 2 근처): Maclaurin 급수. 큰 양의 x: 지수 감소 점근, 큰 음의 x: 진동 점근. 중간: 1/3차 Bessel 관계식(§5.8의 함수) **3구간**
- 출력 넷(Ai, Ai′, Bi, Bi′)을 한 번에 계산한다. 같은 중간값을 함께 쓴다.

- **S6에서 정하고 바꾼 것**
  - 0·1차: Cephes 그대로이되 Hankel 위상을 반올림하지 않는다(libm의 sin x, cos x로 덧셈정리), e^±x는 m·2^e, log(x/2)는 log x − log 2.
  - 실수 차수: Γ₁·Γ₂는 1/Γ(1+z)의 Taylor 계수(생성), x ≥ 30이고 ν < x이면 작은 차수의 Hankel에서 위로 재귀(진동 영역의 CF1은 0.3·x ULP를 잃는다), 다섯 재귀 모두 double-double, x² < ν + 1이면 거듭제곱급수, Hankel은 항이 앞항의 2배를 넘지 않을 때만. **4구간**(급수, Temme/Steed, 위로 재귀, Hankel)
  - Airy: 위상 ζ를 double-double로. |x| ≲ 1e12까지 포락선 기준 안이다.

### 5.10 타원적분 (`ellipk(m)`, `ellipe(m)`; scipy와 같은 매개변수 m)
- `ellipk`: AGM. `K = π / (2·AGM(1, √(1-m)))`, 이차 수렴(약 5회). `m < 0`은 변환. m → 1의 log 특이점은 `1-m`이 정확하므로(m ≥ 0.5에서 Sterbenz) AGM이 그대로 처리한다.
- `ellipe`: AGM의 `E = K·(1 - Σ 2^{n-1} c_n²)`는 m → 1에서 `∞ × 0`의 소거가 생긴다. 그래서 Carlson 대칭형 `E = R_F(0, 1-m, 1) - (m/3)·R_D(0, 1-m, 1)`(복제 정리)를 쓴다. 두 방법의 대비가 해설의 주제다.
- **S5에서 바꾼 것** (사용자 결정 2026-11-14): 공개 함수는 Cephes `ellpk`·`ellpe`의 다항식 `P(y) − log(y)·Q(y)`, `P(y) − log(y)·y·Q(y)`(y = 1 − m)다. 위의 AGM·Carlson과 정확도가 같고(최대 2.3/2.0 대 2.9/4.2 ULP) 4–7배 빠르다. m < 0은 y = 1/(1−m)를 직접 만든다.
  - AGM과 Carlson은 비교용 비공개 함수 `_ellipk_agm`, `_ellipe_carlson`으로 남겼다. 위의 Carlson 식은 m → 1에서 K − (K − E)의 상쇄가 있어서(y = 1e-16에서 약 20배), 두 항이 모두 양수인 `E = (y/3)(R_D(0, y, 1) + R_D(0, 1, y))`(DLMF 19.25.1)로 바꿨다.

### 5.11 초기하와 Lambert W
- `hyp1f1(a, b, x)`: 거듭제곱급수, `x < 0`은 Kummer 변환 `e^x M(b-a, b, -x)`, 큰 |x|는 점근전개. **3구간**. `b`가 0 이하의 정수이면 극점. 큰 |a|에서의 소거는 느슨한 영역으로 적는다.
- `hyp2f1(a, b, c, z)`, z < 1의 실수:
  - `|z| ≤ ½` 근처는 급수. 나머지는 선형 변환(`z → z/(z-1)`, `1-z`, `1/z`, `1/(1-z)`)으로 급수가 빨리 수렴하는 곳으로 옮긴다 (변환, 구간 수에 세지 않음)
  - `c-a-b`나 `a-b`가 정수이면 변환 공식이 0/0이 된다. digamma가 들어간 극한 공식 하나로 처리한다 (AS 15.3.10–12 형태, Q8)
  - a나 b가 0 이하의 정수이면 다항식(유한합)
  - z ≥ 1: 다항식이면 값, 아니면 참값이 복소수이거나 발산하므로 NaN(z = 1은 `c-a-b > 0`이면 Gauss 공식)
- `lambertw(x, *, k=0)`, 가지 k ∈ {0, −1}:
  - 초기값: 분기점(−1/e) 근처는 `p = √(2(ex+1))`의 급수, 그 밖은 log 점근(`L₁ - L₂ + L₂/L₁`), k = 0의 작은 x는 Padé
  - Halley 반복을 수렴할 때까지(상한 있음). scipy의 `tol=`은 두지 않는다: 결과의 정확도는 함수가 정한다.
  - k = 0은 x ≥ −1/e, k = −1은 −1/e ≤ x < 0에서 정의. 그 밖은 실수 값이 없으므로 NaN (S8 뒤 complex 입력은 complex 결과)

- **S7에서 정하고 바꾼 것**
  - `hyp1f1`: 급수 항을 double-double로(연이은 곱이 k ULP를 쌓아 x ≈ 500에서 120 ULP였다), Kummer의 b − a도 double-double, x ≤ −700 근처는 Kummer 형태의 점근식(Γ(b − a)를 1차 보정). b < 0에서 두 형태가 모두 상쇄하면 양의 b′에서 b에 대한 재귀(DLMF 13.3.2). 큰 |a|(a < 0)의 상쇄는 느슨한 영역으로 남았다(Bessel형 전개를 넣지 않음).
  - `hyp2f1`: 급수 항을 double-double로. c − a − b가 정수에 1e-3 안으로 가깝지만 정수가 아니면 1 − z ≥ 1e-3에서 급수, 1e-8 안이면 정수 m의 극한식.
  - `lambertw`: 분기점 둘레 급수의 계수를 정의에서 생성(`tools/gen_lambertw_branch.py`), e·x + 1은 e를 double-double로.

- **속도 (2026-11-15, 최적화 3단계)**: 두 급수는 항의 곱이 k ULP씩 오차를 쌓아서 항마다 double-double 곱·나눗셈을 했다(항당 약 70연산). 둘로 나눴다.
  - **float64 첫 패스**: 그냥 더하면서 Σ(k+1)|t_k|를 함께 잰다. 이 값이 4|s| 이하이면(한 걸음에 반올림이 여섯 번이라 항 k는 많아야 약 3(k+1) ULP, 합은 많아야 약 12 ULP; 잰 최대 4.1) 그 합을 쓴다. 합은 보상 합이다. 항이 처음부터 줄어드는 입력 대부분이 여기서 끝난다: |z| < 0.5에서 840 → 93 ns.
  - **긴 급수는 보정 루프**: 곱과 몫의 반올림 오차를 fma로 구해 1차까지 들고 간다(항과 합이 hi + lo, 항당 나눗셈 1번). 남는 오차는 2차라 double-double 루프와 정확도가 같았다(z ∈ [0.5, 0.9]에서 둘 다 최대 2.7 ULP). 시간은 절반이다.
  - **해 봤지만 버린 것**: 1 − z 변환의 경계를 0.9에서 0.6으로 내리면 빠르지만, a, b가 크면 두 항이 상쇄해 값이 완전히 틀렸다(7e10 ULP).
  - 결과: `hyp2f1` 0.05 → 0.31배, `hyp1f1` 0.16 → 0.81배(1스레드, scipy 대비; 1단계의 몫 포함). 남은 것은 z가 0.5–0.9인 긴 급수(0.3–0.5배)와 x가 수백인 `hyp1f1`(0.17배)이다. scipy는 여기서 float64 급수를 쓰고 100–200 ULP를 잃는다. 측정은 bench/special/opt3/.

### 5.12 로지스틱·정보
- `expit`: `x ≥ 0`이면 `1/(1+e^{-x})`, 아니면 `e^x/(1+e^x)`. NEON은 둘 다 계산해 고른다.
- `logit(p)`: `log(p/(1-p))`는 p ≈ ½에서 상대 정확도를 잃는다. `p ∈ [0.25, 0.75]`는 `2·atanh(2p-1)`(빼기가 정확), 그 밖은 `log(p/(1-p))`. **2구간**
  - S1에서 바꿈: 처음 계획한 바깥 식 `log(p) - log1p(-p)`는 vForce에 맞는 `log1p`가 없어서(−0의 부호를 잃음, array.md §5.4) 쓰지 않았다. 바깥에서는 `|log(p/(1-p))| ≥ log 3`이라 비의 반올림이 몇 ULP에 그친다. 잰 최대 2.8 ULP다.
- `log_expit`: `x ≥ 0`은 `-log1p(e^{-x})`, 아니면 `x - log1p(e^x)`
- `xlogy`, `xlog1py`, `entr`: 정의 그대로, 0의 규약은 scipy를 따른다 (`xlogy(0, y)`는 y가 NaN이 아니면 0)
- `rel_entr(x, y)`: `x·log(x/y)` 그대로면 x ≈ y에서 비의 반올림 오차가 0에 가까운 log에 그대로 남는다(잰 상대오차 7e-5). 그래서 `x/y ∈ [½, 2]`이면 `x·log1p((x-y)/y)`(x − y는 Sterbenz로 정확), 비가 0·서브노멀·inf가 되면 `x·(log x − log y)`. scipy 1.18도 x ≈ y에서 정확하다.

### 5.13 복소수 (S8)
- 대상은 Q19로 정했다: `wofz`, `loggamma`, 그리고 `erf erfc erfcx lambertw`의 complex 입력. complex 입력이면 결과도 complex128이다. 실수와 복소수를 둘 다 받는 함수는 바인딩이 입력의 dtype으로 루프를 고른다(실수 Op와 복소 Op가 따로 있다).
- `wofz`(Faddeeva): 위쪽 반평면 Re z ≥ 0만 계산하고 나머지는 대칭 w(−z̄) = conj w(z)와 w(z) = 2exp(−z²) − w(−z)로 옮긴다.
  - 0 ≤ Re z ≤ 7, 0 ≤ Im z ≤ 4: 0.1 격자의 가장 가까운 점 z₀ 둘레 Taylor 급수. 표(`coeffs/wofz_grid.hpp`, `tools/gen_wofz_grid.py`)에는 w(z₀)만 double-double로 두고, 계수는 w′ = −2zw + 2i/√π의 재귀 a_{n+1} = −(2z₀aₙ + 2a_{n−1})/(n+1)로 만든다. |z − z₀| ≤ 0.0707이라 20항 안팎이다.
  - 그 밖: Laplace 연분수. 항 수는 Poppe–Wijers 식을 Johnson이 맞춘 것에 4를 더했다(더하지 않으면 76 ULP, 4이면 2.8 ULP). |z|가 3e8을 넘으면 첫 항 i/(√π z)만 쓴다.
  - exp(−z²): −z² = (y−x)(y+x) − 2ixy를 double-double로 정확히 만들고, 위상의 두 조각을 libm의 sin·cos(정확한 인자 축소)에 넣는다.
  - 설계의 후보(Weideman 유리 근사, Poppe–Wijers 한 식)는 쓰지 않았다. scipy(Johnson의 Faddeeva 패키지, 연분수와 Algorithm 916)를 재 보니 위쪽 반평면에서도 178 ULP라, 정확도 기준(§1.3)으로 격자 Taylor를 골랐다.
- `erfcx(z) = w(iz)`. `erfc(z)`는 Re z ≥ 0에서 exp(−z²)·w(iz)(m·2ᵉ 꼴로 곱해 오버플로를 피한다), Re z < 0에서 2 − erfc(−z). `erf(z)`는 |z| < ½에서 Taylor 급수, 그 밖 Re z ≥ 0에서 1 − erfc(z), Re z < 0에서 −erf(−z). 실수축·허수축은 따로 다뤄 0인 부분을 정확히 0으로 낸다.
- `loggamma`: Re z > 7이나 |Im z| > 7은 Stirling 8항, Re z < 0.1은 반사식(Hare의 2πik), 0.1 ≤ Re z ≤ 4이고 |Im z| ≤ 3이면 아래의 상쇄 없는 꼴, 나머지는 위로 올리는 점화식(곱이 음의 축을 건너는 횟수를 센다).
  - 1과 2의 영점 둘레에서 점화식은 Stirling 값(크기 13 안팎)과 곱의 로그가 크게 상쇄한다(잰 최대 293 ULP, scipy 439). 그곳은 Re z ≤ 1.5까지 log1p(z−2) + loggamma(z−1)로 내린 뒤, w = z − 1로 loggamma(1+w) = −γw + Σ_{j≤6} μ(w/j) + Σ d_k w^k (μ(t) = t − log(1+t), d_k = (−1)^k ζ(k, 7)/k, `coeffs/loggamma_taylor.hpp`)로 계산한다. 각 부분이 서로 상쇄하지 않는다.
- `lambertw`의 complex: 분기점 근처(|z + 1/e| < 0.3, k = 0과 위쪽의 k = −1, 아래쪽의 k = 1)는 S7의 분기점 급수(e·z + 1을 double-double로), 그 밖은 log1p(z)나 점근식 L₁ − L₂ + L₂/L₁에서 시작하는 Halley 반복. |z|가 1e150을 넘거나 (k ≠ 0에서) 1e-150보다 작으면 w + log w = log z + 2πik에 Halley를 쓴다(w·eʷ가 넘치지 않게). 실수축 위의 실수 가지(k = 0, 위쪽의 k = −1, 아래쪽의 k = 1)는 실수 함수의 값을 그대로 낸다.
- **S8에서 정하고 바꾼 것**
  - `wofz`: 설계 후보 대신 격자 Taylor + 연분수(위). 연분수 항 수에 여유 4.
  - `loggamma`: scipy와 같은 구조에 1–2 둘레의 상쇄 없는 꼴을 더했다(Taylor 원판 둘 대신).
  - 복소수의 오차는 크기 기준, 영점 쪽은 포락선(사용자 결정). 오버플로하는 결과의 부호는 정확한 위상을 따른다.

### 5.14 Boost.Math (S10)
- **사용자 결정 (2026-11-14, S9 뒤)**: "Boost로 갈아끼는 게 이득인 부분은 다 갈고, 손해인 부분만 우리가 작성한다." 계수만 가져오지 않고 Boost.Math를 직접 부른다. 헤더는 쓰는 것만 `third_party/boost_math/`에 둔다(125개, 2.7 MB, BSL-1.0, `tools/vendor_boost_math.py`).
- **손해의 기준** (사용자 결정): 구간마다 Boost가 계약(1e-14, 특수값은 우리 규칙)을 참값 파일의 모든 점에서 지키고, 1스레드에서 느리지 않을 때(10% 안은 같은 것으로 본다)만 바꾼다. 더 정확해도 느리면 우리 것이다(`erf` 계열, `gammaln`, `j0 j1 y0 y1`). 일부 점만 틀리면 구간을 잘게 나눠 지키는 구간만 바꾼다.
- **바꾼 구간** (`csrc/special/boost.hpp`, 측정 `bench/special/boost/routing.md`). 특수값(NaN, inf, 0, 극점)은 Boost를 부르기 전에 우리가 정하고, Boost의 결과가 정상 범위가 아니면(서브노멀, 0, inf, NaN) 우리 코드로 다시 계산한다.

  | 함수 | Boost 구간 | 우리 구간 (이유) |
  |---|---|---|
  | `zeta` | x ≥ −1, 음의 홀수 정수 −3 … −59 (자명한 영점과 2–52의 짝수 제외) | 정수가 아닌 −60 < x < −1 (Boost 65 ULP), x ≤ −60 (Boost 1.7e3 ULP) |
  | `i0 i1 k0 k1` | 전부 | 결과가 정상 범위 밖(i0·i1의 713 근처 참값, k1(5e-324)) |
  | `iv`, `kv` | \|ν\| < 30, \|ν\| < 400 (x > 0) | 그 밖 (Boost가 결과가 작은 곳에서 30배까지). ν ≥ 30은 Debye. `kv`의 30 ≤ ν < 400은 둘 다 계약 안이고 Boost가 5% 빨라 Boost |
  | `ive`, `kve` (S11) | 같은 구간에서 Boost의 값 × e^{∓x}, 0 < x < 700 | 그 밖 |
  | `i0e i1e k0e k1e` (S11) | Boost의 값 × e^{∓x}, 0 < x < 700 | 그 밖 |
  | `betaln` (S11) | log(Boost의 beta), a, b > 0, a + b < 40 | 그 밖. `rgamma` = 1/Boost gamma는 16% 느려져서 되돌렸다 |
  | `expi` | x ≥ −700 | 서브노멀 꼬리 |
  | `lambertw` (실수, 두 가지) | \|x + 1/e\| ≥ 0.05 | 분기점 근처 (Boost 2e7 ULP) |
  | `digamma` | x > 0 | x < 0 (Boost가 극점 근처에서 틀림) |
  | `gamma` | x > −170, 극점 제외 | 서브노멀 꼬리 |
  | `beta` | a, b > 0, a + b < 40 | 큰 인자 (Boost 1.8e3 ULP) |
  | `gammaincinv`, `gammainccinv` | p ≥ 1e-300이고 근이 1e-300 이상 / 전부 | 서브노멀 근 |
  | `betainc`, `betaincc` | a, b ≤ 6 | 큰 a, b (Boost 18배까지) |
  | `betaincinv`, `betainccinv` | a, b ≥ 10 / a, b ≥ 1 | 작은 a, b (Boost가 언더플로를 틀림) |
  | `hyp1f1` | 느슨한 영역 전부, 보장 영역의 x ≥ 0이고 b < 11 | 보장 영역의 나머지 (Boost 208 ULP) |
- 그대로 둔 것: Airy(우리는 네 값을 한 번에 계산해 277 ns, Boost는 넷을 따로 불러 약 720 ns이고 \|x\| > 15에서 틀린다), `expit`, `logit`, `erf` 계열, `expn`, `polygamma`, 타원적분, `jv`·`yv`, 불완전 감마(Boost가 느리거나 틀린다). Boost에 없는 함수(복소수 전부, `erfcx`, 정규분포, e-스케일 Bessel, `hyp2f1`, Hurwitz 제타, 리덕션)도 우리 것이다.
- `hyp1f1`의 느슨한 영역에서 Boost도 값을 내지 못하면(큰 음의 a), a < −1e3이면 Tricomi의 Bessel 함수 전개(A&S 13.3.7)를 쓴다. |a| ≥ 1e3의 mpmath 참값 46점에서 Boost와 같은 수준(상대 1e-12 안팎)이었다. `hyp1f1(-1e12, 2, 0.5)`가 inf에서 −3.47e-10이 됐다(코드리뷰).
- 내부 호출: `hyp2f1`과 `hyp1f1`의 우리 식이 부르는 `Gamma::f`, `Digamma::f`도 바뀐 구간에서는 Boost다. `betainc`와 `betaincc`의 합은 4.5 ULP까지 1에서 벗어난다(모양 [1e-5, 6]의 34만 점; Boost가 둘을 따로 계산한다. 최대인 점은 둘 다 Boost 값이다).

---

### 5.15 분포 함수 (`special/ops/distr.hpp`, T1)
- 15개 모두 기존 함수의 조합이다. 조합에서 정확도를 잃지 않는 것이 일의 거의 전부였다.
- **인자를 반올림하지 않는다.** `stdtr`의 x = df/(df + t²), y = t²/(df + t²)와 `fdtr`의 w, 1 − w를 입력에서 double-double로 만들고, 반올림하지 않은 채 불완전 베타의 핵심(`ibeta_detail::ibeta(a, b, x, y)`, §5.4)에 넘긴다. 매개변수가 크면 I_x는 x에 민감하다(조건수 ~ a·x). 그래서 x를 double로 반올림하면 꼬리에서 100–300 ULP를 잃었다(처음 측정: fdtr 290, fdtrc 363). dd로 넘긴 뒤 16–18 ULP다. 핵심은 I와 1 − I를 한 번에 주므로 보수도 같은 호출이다.
- **작은 모양은 반올림한 인자로 `betainc`를 부른다.** `stdtr`은 df/2 ≤ 6, `fdtr`·`fdtrc`는 dfn/2, dfd/2 ≤ 6이면 x, y 중 작은 쪽을 double로 반올림해 `Betainc::f`·`Betaincc::f`(§5.14의 Boost 구간)에 넘긴다. I_x의 x에 대한 조건수가 a·x ≤ 6이라 반올림이 3 ULP 안이다(`t_tail`, `f_value`).
- **작은 인자 급수**: 인자가 2⁻¹⁰⁰⁰ 아래(언더플로 근처)이면 I_x(a, b) ≈ x^a/(a B(a, b))를 직접 쓴다. 예: `stdtr(1, −1e200)` = 3.2e-201인데 x = 1e-400은 언더플로한다.
- **역함수는 작은 쪽을 푼다.** `stdtrit`, `fdtri`는 x와 1 − x 중 작은 쪽을 betaincinv/betainccinv로 푼다. 어느 쪽이 작은지는 p를 I₁/₂(a, b)와 비교해서 정한다. 처음에는 p ≤ 1/2로 골라서, x가 아주 작은데 1 − x를 풀었다. 그 결과 `stdtrit` 4e-9, `fdtri`는 0을 냈다.
- **역함수의 먼 꼬리**: 근이 2⁻⁶⁰ 아래이면 급수 p ≈ x^a/(aB)를 거꾸로 풀어 거듭제곱으로 얻는다. 지수 −log(paB)/a를 double로 계산하면, 반올림한 지수에 |log|(~450)이 곱해져 191 ULP를 잃었다. 그래서 dd로 계산하고 exp를 한 번만 반올림한다(`exp_dd`, `log_product`; 서브노멀 p는 2⁶⁴를 곱해 둔다). 그 뒤 `stdtrit` 6.9 ULP다.
- **`ndtri_exp`**: y > log ½이면 −ndtri(−expm1(y)), y > −700이면 ndtri(exp(y))다. 그 아래는 점근식 x² = −2y − log 2π − 2 log(−x) + 2 log(1 − 1/x² + 3/x⁴)의 고정점이다. −1e8 위에서는 그 값에서 log_ndtr로 Newton을 한다. Newton의 기울기 φ/Φ는 크기 x²/2인 두 수의 차라서, |x| > 1e6부터 자릿수를 잃었다(y = −1e17에서 2e13 ULP). 그래서 −1e8 아래는 고정점만 쓴다(다음 항 15/x⁶이 마지막 비트보다 훨씬 작다). y < −2¹⁰⁰⁰에서는 −2y가 넘치므로(y ≤ −8.99e307) 같은 고정점을 x²/2에 대해 풀고 x = −√2·√(−y − log(2π)/2 − log(−x) + log S)로 쓴다(2026-11-19). `sd.stats.Normal`의 `ilogcdf`·`ilogccdf`는 log q < −2¹⁰⁰⁰에서 이 값을 그대로 쓴다(Newton의 z²/2가 넘친다).
- **`stdtrit`의 속도 (2026-11-15, 최적화 4단계)**: betaincinv로 풀면 꼬리 평가가 네댓 번이다(1100 ns, scipy의 0.21배). df ≥ 1이고 양쪽 꼬리 확률이 1e-12보다 크면 Hill의 근사(CACM 396, 6자리)에서 출발해 꼬리 함수에 Halley 걸음을 한다. 걸음이 4e-6|t| 아래이면 남는 오차가 1e-17이라, 대부분 평가 한 번으로 끝난다(약 310 ns, scipy의 약 0.5배). 참값 최대 7.4 ULP, 무작위 4천 점(df 1–1.6e5) 최대 4 ULP. 더 깊은 꼬리와 df < 1은 전처럼 betaincinv다.
- **S4의 결함을 고쳤다.** T1을 재다가 `betaincinv`, `betainccinv`가 a, b < 1의 약 8%에서 0이나 1을 돌려주는 것을 찾았다. AS109의 처음 추정이 U자 밀도에서 엉뚱한 끝을 골라 (0, 1) 밖의 값(예: 1 − x = 5400)을 냈고, 코드는 그것을 "근이 언더플로한다"로 읽었다. 이제는 범위 밖이면 다른 끝의 근사를, 그것도 안 되면 0.5에서 출발한다(`as109`). 40만 개 표본에서 scipy와 모두 일치한다. 그 영역의 참값 파일도 다시 만들었다. 옛 참값은 출발점으로 쓴 우리 함수의 0을 그대로 받아 0이 들어 있었다.
- **참값 생성의 함정** (tools/special_ref.py, mpmath 시절. Arb로 바꾼 뒤에는 공이 자릿수 손실을 그대로 드러낸다, §6):
  - mpmath의 betainc는 인자가 1에 가까우면 자릿수를 잃는다. 그래서 x와 1 − x 중 작은 쪽에서 계산한다(`_ibeta_pair`).
  - 값이 2^-1074보다 훨씬 작으면 수렴을 포기한다. 이때는 선두 항이 e^-1000 아래인지 보고 0으로 둔다(`_safe_betainc`).
  - `ndtri_exp`는 x < −1e6에서 log Φ의 점근 급수와 해석적 기울기 −x/S(x)를 쓴다. mpmath의 ncdf, npdf가 넘치기 때문이다.
  - 역함수 `fdtri`는 w가 아니라 x에 대해 푼다. w는 x가 크면 1로 반올림되기 때문이다.

### 5.16 비중심 분포 (`special/ops/noncentral.hpp`, T2)
- **Boost는 쓰지 않는다.** scipy 1.18이 쓰는 Boost.Math의 비중심 분포를 먼저 쟀다. 중앙값은 1 ULP 남짓이었지만 깊은 꼬리에서 0을 내거나 수백–수천 ULP가 틀렸다(최대 8.8e15). Boost 정책(§5.14)에 따라 직접 계산하고, 헤더도 옮기지 않았다.
- **포아송 혼합**: 카이제곱 S = Σ_j D(j, λ/2) P(df/2 + j, x/2), F는 I_w(dfn/2 + j, dfd/2)로 같다. D(a, μ) = μ^a e^(−μ)/Γ(a+1)이다. t는 x = t²/(df + t²)에서 Φ(−δ) + ½Σ[D(j, δ²/2) I_x(j+½, df/2) + sign(δ) D(j+½, δ²/2) I_x(j+1, df/2)]다. 위꼬리는 G를 Q, 1 − I로 바꾼 같은 꼴이라 양의 항만 더한다.
- **한 번의 걷기**: G(s) = G(s+1) + T(s)(아래 족) 또는 G(s+1) = G(s) + T(s)(위 족)다. 이 점화식은 T를 더하는 방향으로만 안정하다. 그래서 반대쪽의 먼 끝 J에서 출발한다. J는 무게 비율과 G 비율의 엄밀한 상한(Γ(s, y) ≤ y^(s−1)e^(−y)·y/(y−s+1) 등)의 곱이 2⁻⁵⁶ 아래로 떨어지는 곳이다. 거기서 G·T·무게를 한 번씩 직접 구하고, 정점을 지나 항이 사라질 때까지 걷는다. 처음에는 정점과 양 끝 세 곳에서 직접 계산했는데, 호출 한 번에 직접 계산이 6–9번 들어가 scipy의 1/14 속도였다.
- **드리프트 없는 점화식**: 무게와 T는 비율 μ/(j+1+h), y/(s+1), x(s+b)/(s+1)로 걷는다. double로 걸으면 32걸음에 수십 ULP가 쌓였다(8걸음마다 직접 다시 맞추면 정확하지만 느리다). 그래서 double-double로 걷는다. 나눗셈은 역수(`recip`, 나눗셈 1번 + Newton 보정)의 곱으로 바꿨다. G와 합은 보상 합이다.
- **모양 s = s0 + j는 double-double이다.** double로 반올림하면 G가 s·eps·|∂log G/∂s|만큼 움직인다. 꼬리에서는 s·eps·|log(y/s)|이고 s = 800에서 250 ULP였다. G와 T는 s.hi에서 구한 뒤 s.lo만큼 1차 보정한다. T는 log-도함수(digamma)로, G는 전방 차분으로 보정한다. t의 δ²/2도 같은 이유로 double-double이다(δ = 17에서 89 ULP).
- **프레임**: 항이 double 범위 밖이어도 비트를 잃지 않게 다음처럼 다룬다. 무게와 G의 직접값은 지수가 분리된 꼴(`Scaled`)이다. 걷는 동안 무게는 2^±500 안에 두고 남는 지수는 G로 옮긴다. G가 2^±500을 넘으면 합과 함께 프레임을 옮긴다. 아주 작은 G(< 2⁻¹⁰⁰⁰)는 앞 인자 × 급수나 연분수로 직접 구한다. 그래서 서브노멀 결과도 절대오차 규칙을 지킨다.
- **t의 반대쪽 꼬리**: t > 0이고 δ < 0인 P(T > t)에서는 두 급수가 상쇄한다(scipy는 1.4e-76에 1.6e-33, 그 아래는 0). 그래서 Q = ∫ ½erfcx(z/√2) e^(E(v)) dv로 계산한다. z = a + t e^v이고, E는 −z²/2와 S = √(V/df)의 로그 밀도의 합이다.
  - E를 정류점 v*(닫힌 꼴)에서 E*(double-double)와 상쇄 없는 나머지로 나눈다.
  - 적분은 v − v* = c·A·sinh(u/A) 위의 사다리꼴 규칙이다. c = min(σ, 0.4), A = 8, 간격 0.25다.
  - E 안의 e^(2v) 때문에 해석적 띠의 폭이 σ와 상관없이 π/4로 묶인다. 그래서 정점 근처를 선형으로 촘촘히 두어야 했다. A = 4이면 2e-13, A = 8이면 1e-17이었다.
- **역함수**: 작은 쪽 꼬리의 log(F/목표)를 근으로 푼다.
  - log F − log 목표로 하면 F ≈ 1e-238에서 각 로그가 546 eps의 절대오차를 가져 근이 190 ULP 움직였다.
  - 출발점은 Patnaik 근사(카이제곱·F)와 정규 근사(t)다. 로그 또는 asinh 눈금에서 브래킷을 잡아 2배 폭까지 이분하고, x 자체에서 Brent법으로 마무리한다.
  - 근이 double 범위 밖이면 0이나 inf다. 서브노멀 간격보다 작은 걸음은 진전이 없으니 걸음을 키운다.
- **측정 범위 밖의 견고함** (2만 점 무작위 입력: 자유도·nc 1e5, x 1e±300):
  - **무시할 합**: j*의 무게와 G 상한의 대략적인 로그(lgamma)가 −2000 아래이면 0을 돌려준다. 이것이 없을 때 x ~ 1e100의 꼬리는 j* ~ 2⁵²에서 J까지 걸으며 멈추지 않았다. 지수 −1.4e5 아래에서는 Scaled의 exp가 포화해 NaN도 냈다.
  - **작은 인자의 꼬리**: t나 f가 거대해서 작은 인자(df/(df+t²), dfd/(dfn·f+dfd))가 2⁻¹⁰⁰⁰ 아래이면 I_y(b, s) ≈ y^b Γ(b+s)/(Γ(b+1)Γ(s))로 혼합 합을 바로 더한다. 이것이 없을 때 F의 위꼬리는 0이었다(dfd = 0.01, f = 1e300의 참값은 0.031). b·log y는 double-double이다(−1e3에 달해 double이면 4e-14).
  - **기타**: 확률은 [0, 1]로 자른다(무게의 합이 1을 1 ULP 넘었다). j*를 찾는 이분법의 상한이 정수가 아니어서(4μ) 끝나지 않던 결함도 고쳤다.
- **속도 (2026-11-15, 최적화 4단계)**: 처음 1스레드 0.14–0.25배(scipy 대비)에서 0.32–0.57배가 됐다. 앞 인자(§5.4)와 연분수(§5.5)의 몫이 대부분이다. 여기서 한 것은 셋이다.
  - j*는 무게의 최빈값에서 비율을 한 번 보고, 꼬리 쪽일 때만 이분한다.
  - 무시할 합의 검사(lgamma 둘)는 j*가 최빈값이 아니거나 인자가 모양에서 2²⁰배 떨어진 때만 한다.
  - 역함수의 Brent법은 걸음이 3e-9|x| 아래가 되면 마지막 두 점의 할선으로 끝낸다(브래킷을 닫는 평가 둘을 아낀다. 오차는 두 걸음의 곱이라 1e-17 아래다).
  - 남은 비용은 구조에서 온다: 먼 끝의 G(모양 보정의 차분까지 불완전 함수 2번), 앞 인자 2번, 걸음마다 double-double 연산이다.
- **T1·S4의 결함을 고쳤다** (T2의 넓은 입력이 찾았다):
  - **`fdtr`**: dfd가 아주 작고 dfn·x가 거대하면(`fdtr(3218.14, 0.0129, 1e20)`) 1.5e-9가 틀렸다. w = 1 − 4e-26을 나눗셈으로 만든 double-double은 1에서의 거리를 절대 1e-32까지만 갖는다. 그런데 `bgrat`이 log w를 그 w에서 계산했다. 이제 x > ½이면 log x = log(1 − y)를 작은 쪽 y에서 계산한다. S4·T1 참값의 정확도는 그대로다.
  - **`chdtr`**: x가 서브노멀이면 x/2가 반올림되거나 0이 되었다(`chdtr(0.18, 5e-324)` = 0, 참값 7.9e-30). 이제 x < 2⁻¹⁰²¹에서는 (x/2)^a/Γ(a+1)을 double-double 지수로 계산한다.
- **참값** (tools/special_ref.py): 두 꼬리 모두 양의 항의 합으로 계산한다. 아래 족은 먼 끝에서 거꾸로, 위 족은 0에서 앞으로 더하고, 끝의 항과 꼬리가 합의 1e-45 아래가 될 때까지 늘린다. t의 반대쪽 꼬리는 Arb의 엄밀한 적분(`acb.integral`)으로 정의 그대로 적분하고, 적분 구간 밖의 상한을 공에 더한다. 역함수의 밀도는 카이제곱이 Bessel 식, t가 두 CDF의 차, F가 ₁F₁ 식이다.

## 6. 정답지와 측정
- **기준값**: `tools/special_ref.py`가 Arb(python-flint)로 기준값을 만들어 `tests/special/ref/*.npz`에 커밋한다. 테스트는 Arb를 매번 부르지 않는다. python-flint는 이 도구와 일부 테스트의 의존성이다 (`pyproject.toml`의 `test`).
  - **엄밀한 공**: 값은 참값을 반드시 품는 공 [중심 ± 반지름]이다. 반지름이 크기의 2⁻¹¹⁰ 아래가 될 때까지 정밀도를 두 배씩 올린다(`tools/arbref.py` `rigorous`, 최대 8192비트). 그러면 hi + lo는 참값을 반올림한 것이다. 정착하지 않으면 그 점을 뺀다. 2⁻¹⁰²² 아래 값은 절대 2⁻¹¹³²까지만 정착시킨다(서브노멀의 반올림을 정하기에 충분하다). 복소수는 작은 쪽 부분을 2⁻¹²⁰|f|까지만 정착시킨다(계약이 |오차|/|f|이므로).
  - **우리가 정하는 자르기**: Arb가 스스로 보증하지 못하는 것은 우리가 정한 급수의 자르기다. 그 꼬리의 상한(기하급수 상한, 부분적분의 나머지 등)을 공의 반지름에 더한다(`add_error`, `geometric_tail`). Arb의 `gamma_lower`, `beta_lower`, `hypgeom_1f1`이 큰 매개변수(a ≳ 1000, a, b ~ 1e4)에서 반지름을 줄이지 못하는 곳은 꼬리 상한을 붙인 급수로 직접 썼다.
  - **역함수**: 근 x는 x(1 ± 2⁻¹¹⁰)에서 부호가 바뀌는 것으로 인증한다(`certify_root`). Newton은 공의 중심에서 하고, 인증이 안 되면 정밀도를 올린다.
  - **mpmath에서 바꾼 이유** (2026-11-16): mpmath의 규칙("두 정밀도가 30자리까지 일치")은 두 정밀도가 함께 틀린 값을 받아들였다. 바꾸면서 찾은 옛 참값의 오류: `betaincinv` 41점·`betainccinv` 119점은 mpmath가 조용히 실패해 우리 함수의 출발값이 참값으로 들어가 있었다(I_x/p − 1 = 1.7e-12). `zeta_hurwitz` 14점(7.3e-28, mpmath는 300자리에서야 맞음), `lambertw` 4점(허수부 lo), 리덕션 17점(1 + e⁻²⁰⁵를 잃음), 오래된 `wofz` 입력 64점. 계수 표 `zeta_minus_one`의 j = 94–110 lo(1e-28 상대, ζ(j) − 1의 상쇄)도 고쳤다. 다른 계수 표 7개는 바이트까지 같았다. 또 mpmath가 버렸던 점 약 900개를 이제 얻는다(`betainccinv` +248, `nctdtrit` +150 등). 메모리도 일정하다: mpmath의 quad는 구간 끝점마다 노드를 캐시해 140GB까지 쌓였다.
  - **새 참값이 드러낸 라이브러리 오차**: `gammainccinv`(a = 9.5e5, q = 3.3e-33)가 1.21e-14였다. Boost의 `gamma_q_inv`는 Halley 반복을 자릿수의 절반에서 멈추므로 큰 a에서 100 ULP까지 남는다. 이제 Boost의 근에서 우리 P, Q로 Halley 한 걸음을 더 한다(`igam_polish`, 오차 e가 e³이 된다). 참값 최대가 `gammaincinv` 9.3e-15 → 7.3e-16, `gammainccinv` 1.2e-14 → 2.4e-15가 됐다. 1백만 원소 비용은 10 ms 안팎 그대로다.
  - 입력 집합: 함수마다 보장 영역의 격자 + 로그 균등 무작위 + 경계(구간 경계의 앞뒤 ULP, 극점, 0, 정의역 끝)
- **정확도 보고**: `bench/special/accuracy.py`가 함수마다 보장 영역·느슨한 영역의 최대·99.9% ULP를 표로 낸다. scipy의 같은 값도 함께 낸다(scipy보다 나쁜 줄은 원인을 적는다).
- **성능**: scipy와 같은 자리에서 번갈아 잰다 (통합 때의 `sameseat` 방식). 1스레드와 10스레드, 원소 10⁶개와 스칼라 1개.
- **비트 테스트**: NEON 경로 = 스칼라 정의, 스레드 1·3·10이 같은 비트, `f(x)[i] == f(x[i])`

## 7. 기준선 (scipy 1.18.1, 원소 10⁶개, 1스레드)
| 함수 | ms | 함수 | ms |
|---|---|---|---|
| erf | 9.8 | gammainc | 63 |
| erfc | 7.4 | gammaincc | 63 |
| erfcx | 16.3 | gammaincinv | 245 |
| erfinv | 13.3 | betainc | 201 |
| ndtr | 10.4 | betaincinv | 744 |
| log_ndtr | 16.9 | expit | 5.7 |
| ndtri | 7.0 | logit | 6.7 |
| gamma | 15.2 | xlogy | 2.6 |
| gammaln | 14.2 | logsumexp (1000×1000, axis=1) | 8.0 |
| digamma | 11.8 | i0 | 12.1 |
| beta | 47.6 | j0 | 3.2 |
| betaln | 48.5 | zeta | 84 |

- scipy.special의 원소별 함수는 1스레드다. 계산이 무거운 함수는 풀(P코어 10)만으로도 수 배 차이를 기대한다. 메모리에 묶이는 가벼운 함수(xlogy, expit)는 그렇지 않다.

---

## 8. 마일스톤
- 마일스톤 S0–S9와 진행 원칙은 [ROADMAP.md](../../ROADMAP.md)의 "특수함수 영역 (sd.special)"에 있다. 진행 상태도 그곳에 적는다.
- 나중의 `sd.stats`는 S4가 끝나면 설계를 시작할 수 있다 (분포의 cdf·ppf가 S2–S4에 달려 있다).

---

## 9. 해설의 주제 (교육 목표)
- 유리 근사와 minimax: 계수는 어디서 오고, 왜 구간을 나누는가 (S2)
- 보수함수와 log 공간: 1 − f가 왜 틀리는가 (S1, S2)
- Lanczos 근사와 앞 인자: 넘치지 않게 식을 모으기 (S3, S4)
- 급수와 연분수의 수렴 영역, Lentz 방법 (S4)
- 역함수: 초기값의 질과 Halley 반복 횟수 (S2, S4)
- Euler–Maclaurin, Bernoulli 수 (S3)
- AGM과 Carlson 대칭형: 같은 적분의 두 길 (S5)
- 두 패스와 한 패스 softmax (S5)
- Chebyshev 전개, Hankel 점근, Temme 방법과 Steed 방법 (S6)
- 선형 변환과 퇴화 극한 (S7)
- 성능: 원소당 비용 모델로 정한 병렬 기준, 분기 없는 NEON, Horner와 Estrin (S0–S3)

---

## 10. 마일스톤을 시작하기 전에 물을 것
결정 Q16–Q19는 §1.1에 있다. 아래는 측정이나 목록이 있어야 정할 수 있어서, 해당 마일스톤을 시작하기 전(또는 측정 뒤)에 사용자에게 묻는다.
- **매 마일스톤**: 특수값을 scipy와 다르게 할 곳의 목록 (§2.2), 함수마다의 보장 영역 (§2.3)
- **S0**: `kMaxOperands` 5의 비용 측정 결과 (§3.1)
- **S4**: 불완전 베타에 세 번째 구간(큰 a, b의 Temme 균등 점근전개)을 넣을지. 정확도와 속도를 잰 뒤 묻는다 (§5.5)
- **알고리즘 후보**: §1.3의 규칙으로 정할 수 없거나 보장 영역을 줄여야 할 때 (예: §5.2의 `gamma` 후보 A/B가 정확도에서 갈리지 않고 속도만 다를 때는 규칙으로 정한다)
