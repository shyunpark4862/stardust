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
| Q10 정답지 | mpmath를 테스트 의존성으로 더한다. 정확도는 고정밀 값 대비 ULP로 잰다. scipy는 차이와 성능을 비교하는 데 쓴다 (§6) |
| Q11 n항 디스패처 | 배열 영역에 일반 n항 원소별 디스패처를 더하고 `kMaxOperands`를 5로 올린다. 별도 마일스톤(S0)이고, 기존 함수의 결과는 비트까지 같다 (§3.1) |
| Q12 이름 | scipy 이름을 쓴다. 입력은 모두 `/` 앞, `out=`은 `*` 뒤. sd에 이미 있는 함수는 다시 두지 않는다 |
| Q13 리덕션 옵션 | `logsumexp`, `softmax`, `log_softmax`는 `axis=None`(과 `logsumexp`의 `keepdims=False`)만 둔다 |
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
  - sd에 이미 있는 함수: `expm1 log1p exp2 exp10 cbrt round sinc` 등 (Q12)
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
  | `special/ops/{erf,gamma,incomplete,logistic,zeta,expint,bessel01,bessel,airy,ellip,hyper,lambertw}.hpp` | 계열마다 Op 구조체 (스칼라 정의) |
  | `special/kernels/neon.hpp` | Q5의 NEON 커널 |
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
- **보장 영역**: 함수마다 docs/api/special.md에 적는다. 그 안에서 mpmath(50자리) 값 대비 상대오차 ≤ 1e-14다.
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
| `lambertw`는 언제나 complex를 돌려주고 `tol=`이 있다 | 실수 입력이면 float64. `tol=` 없음 | Q6, 결정성 (§5.11) |
| `airy`는 네 배열의 tuple | `AiryResult(ai, aip, bi, bip)` namedtuple | DESIGN §9.7 |
| 경고(`SpecialFunctionWarning`) | 내지 않는다 | API.md §2 |
| 별칭(`psi`, `jn`) | 없다 | Q12 |

---

## 3. 엔진

### 3.1 n항 디스패처 (배열 영역에 더한다, Q11)
- 지금 배열에는 `unary<Op>`, `binary<Op>`만 있다 (`array/ops/ufunc.hpp`). `where`는 자기 경로를 갖는다.
- 더하는 것: `nary<Op, N>(operands..., out)`. 입력 N개(1–4)의 디스패치 네 단계(피연산자, 루프, 변환, 실행)를 그대로 한다 (array.md §4.4).
  - `unary`, `binary`는 손대지 않는다. 기존 함수의 결과와 경로가 비트까지 같다 (골든 출력으로 확인).
  - `kMaxOperands` 4 → 5. `hyp2f1(a, b, c, z) → out`이 5개다. `NdIter`, `BufferedLoop`의 고정 배열이 하나씩 커진다. 이 비용(스택 몇십 바이트, 루프 하나)을 S0에서 잰다.
- 출력이 여럿인 원소별 함수(`airy`의 넷)는 출력 4 + 입력 1 = 5개다. 같은 디스패처가 출력 개수를 받는다.
- **S0에서 정한 모양** (docs/learn/S0-foundation.md)
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
  - `c`는 S0의 측정 도구로 함수마다 재고, 입력 분포는 보장 영역에서 고르게 뽑는다.
  - 가중치의 단위는 연속 float64 `add`의 원소당 비용이다 (`kAddNs`, 1스레드에서 0.18–0.19ns). 가중치 `w`인 함수는 원소 `2¹⁵/w`개부터 병렬이 되고, 태스크 하나는 원소 약 `2¹²/w`개다 (`bench/special/cost.py`).
- 원소마다 반복 횟수가 다르다(연분수, Halley 반복). 정적 분할이면 스레드마다 일이 달라진다. 풀의 청크 분할(array.md §10)이 이것을 얼마나 흡수하는지 S4에서 잰다.

### 3.4 NEON (Q5)
- 대상: `erf erfc ndtr ndtri expit logit gammaln`. 분기가 적어서 레인 두 개가 같은 길을 가는 함수들이다.
- 기법
  - 다항식 평가: Horner(곱셈-덧셈이 차례로 이어짐)와 Estrin(트리, 병렬도 높음)을 비교한다. `-ffp-contract=off`이므로 FMA는 코드에 적은 곳에서만 생긴다 (array.md §1.3). 스칼라 정의도 같은 식을 써야 비트가 같다.
  - 분기 없애기: 두 구간을 모두 계산한 뒤 `vbslq`로 고른다. 구간이 셋이면 셋 다 계산하는 비용과 분기 비용을 잰다.
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

### 5.2 감마
- `gamma`: 후보 A Lanczos(Boost `lanczos13m53`, g ≈ 6.0247, 13항), 후보 B Cephes(구간 [2, 3]의 유리식 + 점화식, x > 33은 Stirling). `x < 0.5`는 반사. 오버플로는 x > 171.62
- `gammaln`: 1과 2 근처에서 값이 0이 되어 상대 정확도가 어렵다. Cephes `lgam`(x < 13은 [2, 3]으로 옮겨 `(x-2)·P/Q`, 그 위는 Stirling급수)이 이 문제를 식의 모양으로 푼다. **2구간**. 음수는 반사(`log|Γ|`)
- `rgamma = 1/Γ`: 극점에서 0. 반사와 Lanczos
- `digamma`: `x < 0` 반사(`π cot πx`), 작은 x는 점화식으로 올린 뒤 Bernoulli 점근급수, 양의 근 x₀ ≈ 1.4616 근처는 근을 중심으로 한 급수(상대 정확도). **2구간**
- `polygamma(n, x)`: `n = 0`은 digamma, `n ≥ 1`은 `(-1)^{n+1} n! ζ(n+1, x)` (§5.6의 Hurwitz zeta)
- `beta`, `betaln`: Lanczos 비(Boost `beta`)로 세 Γ를 한 식에 모아 오버플로와 소거를 피한다. 큰 인자는 log 형태
- `binom(n, k)`: 정수 k가 작으면 곱(정확), 그 밖은 `1/((n+1)·B(n-k+1, k+1))`. 음의 n은 반사

### 5.3 불완전 감마 (`gammainc`, `gammaincc`)
- `x < a + 1`: 거듭제곱급수로 P, `x ≥ a + 1`: Lentz 연분수로 Q. 나머지는 보수
- 큰 a, `x ≈ a`(예: a > 20, |x − a| < 0.3a)는 급수와 연분수가 모두 느리고 소거가 생긴다. Temme 균등 점근전개를 쓴다 (DESIGN 1라운드 Q3의 예). **3구간**
- 앞 인자 `x^a e^{-x}/Γ(a)`는 §5.4

### 5.4 앞 인자 (`special/prefix.hpp`)
- `x^a e^{-x}/Γ(a)`를 따로 계산하면 a가 크면 넘친다. Lanczos 형태로 `(x/(a+g-½))^a · e^{a-x}·…`처럼 한 식으로 모은다 (Boost `regularised_gamma_prefix`). 베타도 같다 (`ibeta_power_terms`)
- 이 값은 불완전 함수의 도함수이기도 하다. 역함수의 Halley 반복이 이것을 다시 쓴다.

### 5.5 불완전 베타 (`betainc`, `betaincc`)
- `x > (a+1)/(a+b+2)`이면 대칭으로 옮긴다 (변환). 기본은 Lentz 연분수 (NR `betacf`). b·x가 작거나 b < 1이면 거듭제곱급수
- 큰 a, b(예: 둘 다 > 10⁴)에서 연분수의 반복이 `O(√max(a,b))`로 는다. 세 번째 구간(Temme 균등 점근전개)은 S4에서 정확도와 속도를 잰 뒤 넣을지 사용자에게 묻는다. **2–3구간**
- 역함수(`gammaincinv`, `gammainccinv`, `betaincinv`, `betainccinv`): 초기값 2–3가지(감마: Wilson–Hilferty, 작은 p의 거듭제곱 근사; 베타: 정규 근사, 작은 x의 거듭제곱 근사) + Halley 반복. 반복은 상대 변화가 ε 아래가 되거나 상한(예: 20회)에서 멈춘다. DiDonato–Morris의 초기값 열 가지 남짓은 가져오지 않는다.

### 5.6 제타와 지수적분
- `zeta(x, q)`: Hurwitz, Euler–Maclaurin (Cephes `zeta.c`: 앞 N항 직접 합 + Bernoulli 꼬리). Riemann(`q` 생략)의 `x < 1`은 함수방정식(변환). Hurwitz의 `x < 1`은 NaN (scipy와 같다)
- `expn(n, x)`: `x ≤ 1` 거듭제곱급수, `x > 1` Lentz 연분수 (NR `expint`), 큰 n은 균등 점근. **3구간**
- `expi(x)`: `x < 0`은 `-E₁(-x)` (변환), `0 < x < 40`은 거듭제곱급수(양의 항만이라 소거 없음), 큰 x는 점근급수. **2구간**. Cephes `ei.c`의 여섯 구간 유리식은 쓰지 않는다.

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
- 음의 ν는 반사 공식(변환). 음의 x는 정수 ν에서만 정의(scipy와 같다), 그 밖은 NaN

### 5.9 Airy (`airy`)
- 작은 |x|(≤ 2 근처): Maclaurin 급수. 큰 양의 x: 지수 감소 점근, 큰 음의 x: 진동 점근. 중간: 1/3차 Bessel 관계식(§5.8의 함수) **3구간**
- 출력 넷(Ai, Ai′, Bi, Bi′)을 한 번에 계산한다. 같은 중간값을 함께 쓴다.

### 5.10 타원적분 (`ellipk(m)`, `ellipe(m)`; scipy와 같은 매개변수 m)
- `ellipk`: AGM. `K = π / (2·AGM(1, √(1-m)))`, 이차 수렴(약 5회). `m < 0`은 변환. m → 1의 log 특이점은 `1-m`이 정확하므로(m ≥ 0.5에서 Sterbenz) AGM이 그대로 처리한다.
- `ellipe`: AGM의 `E = K·(1 - Σ 2^{n-1} c_n²)`는 m → 1에서 `∞ × 0`의 소거가 생긴다. 그래서 Carlson 대칭형 `E = R_F(0, 1-m, 1) - (m/3)·R_D(0, 1-m, 1)`(복제 정리)를 쓴다. 두 방법의 대비가 해설의 주제다.

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

### 5.12 로지스틱·정보
- `expit`: `x ≥ 0`이면 `1/(1+e^{-x})`, 아니면 `e^x/(1+e^x)`. NEON은 둘 다 계산해 고른다.
- `logit(p)`: `log(p/(1-p))`는 p ≈ ½에서 상대 정확도를 잃는다. `p ∈ [0.25, 0.75]`는 `2·atanh(2p-1)`(빼기가 정확), 그 밖은 `log(p) - log1p(-p)`. **2구간**
- `log_expit`: `x ≥ 0`은 `-log1p(e^{-x})`, 아니면 `x - log1p(e^x)`
- `xlogy`, `xlog1py`, `entr`, `rel_entr`: 정의 그대로, 0의 규약은 scipy를 따른다 (`xlogy(0, y)`는 y가 NaN이 아니면 0)

### 5.13 복소수 (S8)
- `wofz`(Faddeeva): 후보 Weideman 유리 근사(N = 32, 한 식), Poppe–Wijers. `erf`, `erfc`, `erfcx`의 complex는 `wofz`로 계산한다.
- `loggamma`: 복소 Stirling + 점화식 + 반사, 가지를 연속으로 맞춘다 (scipy와 같은 가지).
- `lambertw`의 complex: 임의 가지 k, Halley
- 대상은 Q19로 정했다: `wofz`, `loggamma`, 그리고 `erf erfc erfcx lambertw`의 complex 입력. complex 입력이면 결과도 complex128이다.

---

## 6. 정답지와 측정
- **기준값**: `tools/special_ref.py`가 mpmath(50자리)로 기준값을 만들어 `tests/special/ref/*.npz`에 커밋한다. 테스트는 mpmath를 매번 부르지 않는다 (느리다). mpmath는 이 도구와 일부 테스트의 의존성이다 (`pyproject.toml`의 `test`).
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
