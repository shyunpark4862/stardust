# 수치 미적분 영역 설계 (sd.integrate, sd.differentiate)

> 상태: **설계 초안** (2026-11-21). 큰 결정(§1.1 C1–C7)과 C0 전의 결정(C8–C17)은 사용자가 정했다. 나머지는 안이다. C0은 완료·승인(위임, 2026-11-21)이다. C1–C8은 시작 전이다. 시작 전에 정할 것은 §9에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

`sd.integrate`(적분: 적응 구적, 배치 tanh-sinh, 급수, 다차원 구적, 표본의 구적 규칙)와 `sd.differentiate`(배치 수치 미분, 야코비안, 헤시안)의 확정 사항과 내부 구조의 안을 적는다. 두 모듈을 이 문서 하나와 마일스톤 하나(C0, C1, …)로 다룬다(C1).

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처의 안은 [docs/api/integrate.md](../api/integrate.md)와 [docs/api/differentiate.md](../api/differentiate.md)다.
- 비교 대상은 scipy 1.18.1의 `scipy.integrate`와 `scipy.differentiate`, 그 바닥인 `scipy._lib._elementwise_iterative_method`, QUADPACK(Piessens, de Doncker-Kapenga, Überhuber, Kahaner 1983)이다.
- 바닥은 배열(`sd.sum`·`sd.cumulative_sum`의 합 규칙), 스레드풀, optimize의 배치 고리(C4)와 차분 간격(C3)이다.
- 이 문서의 잰 값은 Apple M4 Pro, macOS 26.6, scipy 1.18.1·numpy 2.5.4(BLAS는 Accelerate), python-flint 0.9.0에서 본 것이다(2026-11-21, §7.4). 탐침 스크립트는 저장소에 두지 않았다.
- ε은 float64의 기계 엡실론 2⁻⁵² ≈ 2.22×10⁻¹⁶이다.

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-21)
| 번호 | 결정 |
|---|---|
| C1 이름과 자리 | 최상위 모듈 둘: `sd.integrate`와 `sd.differentiate`다(scipy 1.15부터의 구조, 이름은 scipy와 1:1). 설계 문서 하나(이 문서)와 마일스톤 하나(C0, C1, …)가 둘을 함께 다룬다 (사용자 결정 2026-11-21) |
| C2 첫 판의 범위 | 적분과 미분을 먼저 한다. integrate: `quad quad_vec dblquad tplquad nquad tanhsinh nsum cubature fixed_quad newton_cotes trapezoid simpson romb cumulative_trapezoid cumulative_simpson`. differentiate: `derivative jacobian hessian`. 상미분방정식(`solve_ivp odeint solve_bvp`와 풀이 클래스 RK45 등)은 나중의 마일스톤 묶음이다(§8에 적어만 둔다). 어느 쪽이든 넣지 않는 것: 옛 `ode`·`complex_ode`, `lebedev_rule`, `qmc_quad` (사용자 결정 2026-11-21) |
| C3 차분 커널 옮기기 | 지금 `csrc/optimize/diff.hpp`에 있는 고정 간격 차분의 커널(`approx_fprime`, `check_grad`, `jac='2-point'`·`'3-point'`가 쓰는 scipy `approx_derivative`의 간격, O18–O31)을 differentiate 영역의 C++로 옮기고 optimize가 그것을 부른다(H17처럼: 함께 쓰는 커널은 그 본질에 가장 가까운 영역에 둔다). `approx_fprime`·`check_grad`는 scipy의 이름대로 `sd.optimize`에 공개된 채로 둔다. 값은 비트까지 같고 optimize의 골든은 바뀌지 않는다. 지금은 계획만 하고 코드는 옮기지 않는다 (사용자 결정 2026-11-21) |
| C4 원소별 반복 무리 | `derivative`, `tanhsinh`, `nsum`은 optimize의 `find_root find_minimum bracket_root bracket_minimum`(O2·O5의 배치)과 같은 scipy의 "원소별 반복" 무리다. optimize의 규칙을 쓴다: 결과 이름은 `converged`, `n_iter`, `nfev`(O4; scipy의 `success`·`nit`가 아니다), `args`는 이 무리에서만 받는다(O39), status 번호는 scipy의 것(O17), C++ 반복 고리 하나를 함께 쓴다(`csrc/optimize/batch.cpp`에서 끌어낼 계획) (사용자 결정 2026-11-21) |
| C5 정확도 | optimize의 O9처럼: scipy와 같은 걸음을 목표로 하고, 결정적인 알고리즘(QUADPACK, tanh-sinh, `derivative`의 간격 줄이기)은 호출점까지 비트가 같게 하며 예외는 문서에 적는다. 따로, Arb의 엄밀한 적분(python-flint `acb.integral`)으로 만든 참값으로 결과와 오차 추정의 정직함을 확인한다 (사용자 결정 2026-11-21) |
| C6 시험 문제 | QUADPACK 책의 시험 적분, scipy의 시험, Arb 참값이 있는 어려운 적분(끝점 특이점, 진동, 무한 구간)이다. netlib의 QUADPACK Fortran 원본을 받아 견줄지는 나중에 묻는다(C17로 정했다) (사용자 결정 2026-11-21) |
| C7 다른 영역의 전용 구적은 그대로 | 첫 판에서 다른 영역의 전용 구적·급수는 바꾸지 않는다(optimize O7처럼: stats·special은 자기 것을 둔다). 나중의 쓰임(stats의 `expect` 등)은 적어만 둔다(§1.4) (사용자 결정 2026-11-21) |
| C8 측정 영역 (C0 전) | `bench/areas.py`에 영역 둘 `integrate`·`differentiate`를 더한다(stats·random처럼 모듈마다 영역 하나; 테스트·참값·골든·측정도 영역마다). 구현할 때 더한다 (사용자 결정 2026-11-21) |
| C9 결과 클래스 (C0 전) | 모듈마다 하나: `sd.integrate.IntegrateResult`, `sd.differentiate.DifferentiateResult`. `OptimizeResult`의 규칙(없는 필드는 `AttributeError`, 읽기 전용, O103)이고 바탕을 함께 쓴다 (사용자 결정 2026-11-21) |
| C10 `quad` 계열의 반환 (C0 전) | `quad dblquad tplquad nquad quad_vec`는 결과 객체를 돌려주고 그것이 `(integral, error)` 둘로 풀린다(htest H7). `full_output`은 없다 (사용자 결정 2026-11-21) |
| C11 허용오차의 이름 (C0 전) | 어디서나 `atol`·`rtol`이다. `quad` 계열의 scipy `epsabs`·`epsrel`도 그렇다(`[scipy와 다름]`) (사용자 결정 2026-11-21) |
| C12 상한·수준의 이름 (C0 전) | O14·O57의 꼴: `max_iter`(`derivative jacobian hessian`), `max_level`·`min_level`(`tanhsinh`, 결과 필드 `max_level`), `max_terms`(`nsum`), `max_p1`(`quad`). `limit`·`limlst`(와 `max_subdivisions`·`order`·`n`)은 그대로다 (사용자 결정 2026-11-21) |
| C13 값 필드의 이름 (C0 전) | 어디서나 `integral`(급수는 `sum`), `error`, `nfev`다: `cubature`의 `estimate`, `quad`의 `neval`도 이 이름이다. 미분은 scipy의 `df`·`ddf` (사용자 결정 2026-11-21) |
| C14 C4에서 따라 나오는 것 (C0 전) | 확인: `callback`(O37; status −4는 `nsum`의 뜻만), `kwargs`, `preserve_shape`는 없다(`args`만, O39). 아직 끝나지 않은 원소만 부르고(O38), 첫 호출도 1차원이며, 입력이 모두 실수면 결과 필드는 Python 수다(O48) (사용자 결정 2026-11-21) |
| C15 C3의 범위 (C0 전) | `diff.hpp`와 함께 `src/stardust/optimize/_diff.py`의 `Differences`(nls의 정확한 5점 차분, `approx_fprime`의 기본)도 differentiate의 내부로 옮긴다. 공개 이름(`approx_fprime`, `check_grad`)은 optimize에 그대로이고, 차분의 고리(호출 수 세기, 다시 해 보기, 기억)는 부르는 쪽에 둔다. 비트와 골든은 그대로다 (사용자 결정 2026-11-21) |
| C16 반복 고리의 자리 (C0 전) | 함께 쓰는 반복 고리는 csrc 아래 새 바닥 폴더 `elementwise/`에 둔다(scipy `_lib/_elementwise_iterative_method.py`에 해당; optimize·integrate·differentiate가 아래로 의존한다) (사용자 결정 2026-11-21) |
| C17 QUADPACK 원본 (C0 전) | netlib의 Fortran 원본을 받아 gfortran으로 돌린 결과와 견준다(optimize O13처럼). 받기 전에 정확한 파일 목록과 크기를 적어 사용자에게 다시 묻는다(§7.6). 원본은 저장소 밖에 두고 저장소에는 결과만 넣는다 (사용자 결정 2026-11-21) |
| C18 `dgtsl` (C0 뒤) | QUADPACK의 `dqc25f`가 부르는 LINPACK `dgtsl`은 netlib의 원본 `linpack/dgtsl.f`(파일 하나, 3,240바이트, 바깥 호출 없음)를 받아 QUADPACK과 함께 빌드한다. 그 전에 문서의 알고리즘으로 도구 안에 쓴 대체는 빌드에서 빼고 견주기에만 쓴다: 116개 실행 모두 값·점이 원본과 비트로 같다(§7.6). QAWO·QAWF의 대조 다섯 실행은 이제 잠정이 아니다 (추천안 사용자 위임 2026-11-21; 사용자 승인 2026-11-21, 원본 받음) |
| C19 double 끝의 참값 (C0 뒤) | 참값은 double로 낸 문제 그대로의 정확한 적분이다(상한 `math.pi / 2`는 π/2보다 6.1×10⁻¹⁷ 작다). 문제의 뜻인 정확한 π/2·π의 적분은 따로 두지 않고, 그 차이(f9 2×10⁻¹⁵, f10 1.6×10⁻⁸, cos(2^α sin x) 2.5×10⁻¹⁶)는 문서에 적는다(§7.2, §7.7) (추천안, 사용자 위임 2026-11-21) |
| C20 `summary()` (C0 뒤) | `IntegrateResult`·`DifferentiateResult`도 `OptimizeResult`처럼 `summary()`가 있다(API.md §2.3의 상자: 값·오차·수렴·`nfev`의 머리, 배치는 줄마다). 결과를 내는 첫 함수가 공개될 때(C2·C4) 만든다 (추천안, 사용자 위임 2026-11-21) |
| C21 풀기의 범위 (C0 뒤) | `(integral, error)`로 풀리는 것은 `quad` 계열(`quad dblquad tplquad nquad quad_vec`)의 결과뿐이다. `tanhsinh`, `nsum`, `cubature`와 공개 생성자로 만든 결과는 이름으로 읽는다(지금 구현) (추천안, 사용자 위임 2026-11-21) |

- C5와 D-33(API.md §4, docs/design/optimize.md §3.7): 호출점의 비트 동일은 지금 맞추는 목표이고 약속이 아니다. 지키는 것은 다른 영역과 같다(스레드 수와 상관없는 비트, 자기 골든, Arb 계약). 반올림이 아닌 동작(멈춤 판정, status의 뜻, 기본값, 구간을 나누는 순서)은 scipy 그대로다. 호출점이 같다는 것은 사용자 함수를 부르는 점의 순서와 비트, 호출 수가 같다는 뜻이다.

### 1.2 범위 (안)
scipy 1.18.1의 `dir(scipy.integrate)`와 `dir(scipy.differentiate)`에서 뽑았다. 마일스톤은 §8이다.

**sd.integrate**
| scipy | sd | 마일스톤 | 방법과 비고 |
|---|---|---|---|
| `quad` | `quad` | C4·C5 | QUADPACK: 유한 구간 QAGS, 무한 QAGI, `points` QAGP, `weight` QAWO·QAWF·QAWS·QAWC(§3.1). 결과 객체(둘로 풀린다, C10) |
| `dblquad`, `tplquad`, `nquad` | 같다 | C4 | `quad`를 겹쳐 부른다(scipy와 같은 순서) |
| `quad_vec` | `quad_vec` | C7 | 벡터 값 함수의 적응 Gauss–Kronrod(scipy의 Python 판: gk21, gk15, trapezoid) |
| `cubature` | `cubature` | C7 | 다차원 적응 구적(gk21·gk15의 곱 규칙, Genz–Malik) |
| `tanhsinh` | `tanhsinh` | C6 | 원소별 반복 무리(C4). 배치 tanh-sinh |
| `nsum` | `nsum` | C6 | 원소별 반복 무리(C4). 직접 합 + 적분 경계(안에서 `tanhsinh`) |
| `fixed_quad` | `fixed_quad` | C3 | n점 Gauss–Legendre |
| `newton_cotes` | `newton_cotes` | C3 | 닫힌 Newton–Cotes 가중치와 오차 계수 |
| `trapezoid`, `simpson`, `romb` | 같다 | C3 | 표본(`sd.ndarray`)의 구적 규칙 |
| `cumulative_trapezoid`, `cumulative_simpson` | 같다 | C3 | 표본의 누적 구적 |

**sd.differentiate**
| scipy | sd | 마일스톤 | 방법과 비고 |
|---|---|---|---|
| `derivative` | `derivative` | C2 | 원소별 반복 무리(C4). 중첩 스텐실의 고차 차분을 간격을 줄이며 되풀이 |
| `jacobian` | `jacobian` | C2 | `derivative`를 변수마다 함께 |
| `hessian` | `hessian` | C2 | `jacobian`의 `jacobian`(안쪽 rtol을 100배 조인다) |

**넣지 않는 것과 이유**
- 상미분방정식 `solve_ivp`, `odeint`, `solve_bvp`, 풀이 클래스 `RK23 RK45 DOP853 Radau BDF LSODA`, `OdeSolver`·`DenseOutput`·`OdeSolution`: 나중의 묶음(C2, §8).
- 옛 `ode`·`complex_ode`(`vode`, `zvode`, `lsoda`, `dopri5`, `dop853`의 상태 객체): scipy 스스로 `solve_ivp`로 대신한다. 나중의 ODE 묶음에도 넣지 않는다(C2).
- `lebedev_rule`(구면 구적의 표): 쓰임이 좁다(C2).
- `qmc_quad`: `scipy.stats.qmc`(준난수열)가 sd에 없다(C2).
- `IntegrationWarning`, `ODEintWarning`: 경고를 내지 않으므로(§1.3) 없다.
- scipy의 옛 이름공간 `quadpack`, `odepack`, `vode`, `dop`, `lsoda`, 이미 없어진 `quadrature`·`romberg`·`simps`·`trapz`·`cumtrapz`: 없다.
- `LowLevelCallable`(C 함수 포인터를 `quad`에 주기): 받지 않는다. 사용자 함수는 Python 호출 가능한 객체다(§2.3).
- QNG(QUADPACK의 비적응 21–87점 규칙)와 QAG(규칙을 고르는 적응 구적): scipy의 `quad`가 부르는 길이 없다. 그래서 공개하지 않는다(§3.1).

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 입력은 위치로(`func`, `a`, `b`, `y`, `x`), 옵션은 키워드로만. 별칭이 없다. float64 실수로 계산한다(bool·int64는 바꿔서). 복소수는 §9 C4의 질문이다.
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 사용자 함수가 결정론적이면 결과도 결정론적이다.
- **모든 호출은 정해진 시간 안에 끝난다.** 적응 방법은 구간 수(`limit`), 수준(`max_level`), 항 수(`max_terms`), 반복 수(`max_iter`)에 상한이 있다. 상한에 닿으면 마지막 값과 `converged=False`를 돌려준다.
- 경고를 내지 않는다. scipy가 `IntegrationWarning`으로 알리는 것(구간 수 상한, 반올림, 나쁜 피적분함수, 발산)은 결과의 `converged`, `status`, `message`로 알린다. 화면에 쓰지 않는다(`romb(show=True)` 없음).
- 참값은 Arb만 쓴다. mpmath는 쓰지 않는다.
- 영역 간 일관성이 먼저다: 결과 필드의 이름(O4), 메시지의 꼴(`quad(): …`), 원소별 반복 무리의 호출 규약(O38), `args`(O39), 배열 결과의 읽기 전용 규칙(O103, API.md §2.2)은 optimize를 따른다.

### 1.4 다른 영역과의 관계
- **optimize** (C3, C4): 차분 간격의 커널이 이 영역으로 오고(§6.1), 배치 고리가 세 영역이 함께 쓰는 자리로 간다(§6.2). optimize의 공개 함수와 결과는 비트까지 그대로다(골든 불변).
  - 옮긴 뒤에도 `diff.hpp`의 간격 규칙을 쓰는 곳은 optimize뿐이다: `sd.differentiate.derivative`는 scipy처럼 다른 방법(중첩 스텐실의 고차 차분, §3.7)이라 `approx_derivative`의 간격을 쓰지 않는다. 옮기는 까닭은 H17의 원칙(함께 쓰는 커널은 그 본질의 영역에 둔다)이다.
- **stats·special** (C7): 그대로 둔다. 지금 있는 전용 구적: `csrc/stats/mvcdf.hpp`(다변량 정규·t cdf의 적응 Gauss–Kronrod, QUADPACK의 QK21을 자기 사본으로), `csrc/stats/dist/noncentral.hpp`와 `csrc/special/ops/noncentral.hpp`(sinh 사상 위의 사다리꼴 규칙). 나중에 쓸 곳: 분포의 `expect`(scipy `rv_continuous.expect`는 `quad`), 사용자 정의 분포의 cdf·적률, 회귀의 적분 우도. 그때 QK21 표를 함께 쓸지 정한다.
- **배열**: 표본 규칙(`trapezoid` 등)은 sd의 합 규칙(docs/api/array.md §7.8: 누산기 16개·512개 단위의 pairwise)을 쓴다(§3.6).

---

## 2. 입력과 결과

### 2.1 결과 객체 (C9, C10, C13)
- 모듈마다 결과 클래스 하나(C9): `sd.integrate.IntegrateResult`, `sd.differentiate.DifferentiateResult`(optimize의 `OptimizeResult`와 같은 이름 짓기: 모듈 이름 + Result). 규칙은 `OptimizeResult`와 같다: 함수가 주는 필드만 있고 없는 필드는 `AttributeError`, 필드는 읽기 전용, 배열 필드는 읽기 전용이고 읽을 때 복사하지 않는다(O103), `repr`는 한 줄, dict처럼 쓰지 않는다. 바탕은 `src/stardust/optimize/_result.py`의 것을 끌어내 셋이 함께 쓴다(`OptimizeResult`의 동작은 그대로).
- 이름(C4, O4): `converged`(scipy `success`), `n_iter`(scipy `nit`), `nfev`(scipy `quad`의 `neval`도), `status`, `message`(배치 빼고). 값의 필드(C13): 적분은 어디서나 `integral`, 급수는 `sum`, 오차 추정은 `error`, 미분은 `df`·`ddf`(scipy의 것).
- C0에서 만든 것: 바탕은 `src/stardust/_results.py`의 `_FieldResult`다(필드 속성, 읽기 전용, `_Lazy`, 생성자 `_make`, 배치의 한 줄 `repr`). `OptimizeResult`는 그 위에 자기 `__init__`·`repr`·`summary()`를 그대로 두었고(Python의 호출 모양 에러가 클래스 이름을 대므로 `__init__`는 클래스마다 둔다), optimize의 테스트와 골든 3,876케이스가 그대로다. `IntegrateResult`·`DifferentiateResult`의 `repr`는 API.md §2.3의 꼴(`IntegrateResult(integral=0.746824, error=8.29e-15, converged=True, nfev=21)`)이다.
- 둘로 풀리는 것(C10)은 결과를 만드는 함수가 정한다: `quad` 계열은 `IntegrateResult._make(…, unpack=True)`로 만들고, 그 결과만 `a, b = r`, `r[0]`, `len(r) == 2`가 된다. 다른 결과(`tanhsinh`, `nsum`, `cubature`, 공개 생성자로 만든 것)에서는 `TypeError`이고, `bool(r)`은 늘 참이다(C21). `summary()`는 C2·C4에서 만든다(C20).
- `quad` 계열(`quad dblquad tplquad nquad quad_vec`)은 scipy가 튜플 `(y, abserr)`를 돌려준다. 여기서는 결과 객체이고(C10) `integral, error = sd.integrate.quad(f, 0, 1)`처럼 둘로 풀린다(htest H7의 규칙). `full_output`은 없고 scipy `infodict`의 값(`neval`, `last`, `alist`…)은 필드다(docs/api/integrate.md §1.3, §2).
- 원소별 반복 무리의 결과는 필드마다 배치 모양이고, 입력이 모두 실수면 Python 수다(O48).

### 2.2 원소별 반복 무리 (C4)
`derivative`, `tanhsinh`, `nsum`은 optimize의 배치 넷과 같은 틀이다: 배치의 원소마다 따로 풀고, 반복마다 아직 끝나지 않은 원소의 점을 모아 사용자 함수를 한 번 부른다.
- **status** (O17, scipy의 번호): 0 수렴, −1(`derivative`: 오차 추정이 늘어 멈춤; `nsum`: `a`·`b`·`step`이 틀림; `tanhsinh`: 쓰지 않음), −2 반복·수준 상한, −3 유한하지 않은 값, −4(`nsum`만: 부분합의 마지막 항이 허용오차보다 커서 오차 추정이 허용오차를 넘음). scipy 원소별 무리의 −4는 `callback`이 멈춘 것인데 `nsum`은 같은 번호를 다른 뜻으로 쓴다. `callback`이 없으므로(O37) 그 −4는 없고, `nsum`의 −4는 scipy의 뜻 그대로다.
- **호출 규약** (O38 + 여러 점): optimize의 배치는 원소마다 점 하나라 사용자 함수가 (k,)를 받는다. 이 무리는 원소마다 여러 점을 한 번에 부른다(scipy: `derivative`는 첫 반복 order개, 다음부터 2개; `tanhsinh`은 처음 66개, 다음 수준부터 64·128·…). 넘기는 꼴은 §9 C2의 질문이다(추천: (k, p) 2차원, 원소의 순서는 입력의 C 순서, `args`는 (k, 1)).
- scipy의 첫 호출은 원소마다 점 하나(입력 모양 그대로: `tanhsinh`은 구간의 가운데, `derivative`는 x)다. O48처럼 첫 호출도 1차원 (k,)이다.
- `args`(O39): 값마다 배치 모양으로 브로드캐스트하고, 호출마다 남은 원소로 줄여 넘긴다. `kwargs`(scipy 1.18의 이 셋에는 있다), `callback`, `preserve_shape`는 없다(C14).
- 허용오차는 scipy의 `tolerances` dict 대신 키워드다(O36): `derivative(atol=, rtol=)`, `nsum(atol=, rtol=)`. `tanhsinh`은 scipy도 키워드다.
- 배치 결과의 `n_iter`·`nfev`는 그 원소가 멈출 때까지의 수다. `tanhsinh`은 scipy처럼 `n_iter` 대신 마지막 수준 `max_level`을 준다(scipy `maxlevel`; C12).
- 진행 중인 원소가 2¹⁵개 이상이면 원소를 지나가는 패스를 스레드에 나누고 GIL을 놓는다(optimize §3.3.1과 같은 기준과 조각).

### 2.3 사용자 함수와 호출
- 사용자 함수는 Python 호출 가능한 객체다. 반복은 C++가 돌고 함숫값은 콜백으로 받는다(optimize와 같다). 콜백을 부르는 동안 GIL을 쥔다.
- **`quad` 계열**: scipy처럼 점 하나마다 Python `float`로 부르고 실수 하나를 받는다(numpy·Python의 수, 0차원 배열). 순서는 QUADPACK의 순서다: QK21이면 가운데, 다음에 Gauss 점의 쌍(c − h·x₂, c + h·x₂, c − h·x₄, …), 그 뒤 Kronrod만의 점의 쌍(§7.4: scipy와 21점 모두 비트까지 같다). 한 규칙의 점들을 배열 하나로 부르는 선택은 scipy에 없어 두지 않는다(§9 C4).
- **`dblquad`·`tplquad`·`nquad`**: 안쪽 적분을 바깥 점마다 `quad`로 계산한다(scipy와 같은 겹침, 같은 순서). `gfun`·`hfun`(`qfun`·`rfun`)은 수 또는 바깥 변수의 함수다.
- **`quad_vec`**: 점 하나마다 부르고 같은 모양의 배열(`sd.ndarray`, 시퀀스, numpy 배열)을 받는다.
- **`cubature`**: scipy처럼 점들을 한 번에 (n_points, ndim) 배열로 부르고 (n_points, …)를 받는다.
- **원소별 무리**: §2.2. 사용자 함수는 원소별이어야 한다(`f(x)[i]`가 `f(x[i])`와 같다; scipy의 계약). 구간의 끝 바로 옆을 계산하므로 특이점에서 예외를 내지 말고 inf·NaN을 돌려줘야 한다: scipy `tanhsinh`은 반올림으로 끝점 자체(예: x = 0)를 부르고 그 값을 버린다(§7.4). 이것을 문서의 계약으로 적는다.
- `jacobian`·`hessian`의 f는 (m, …)을 받아 (k, …)을 돌려주는 벡터화된 함수다(scipy의 계약: 첫 축이 변수, 뒤 축이 점). 첫 호출부터 (m, m)·(m, m, p)를 받는다(§7.4). sd에서 넘기는 꼴은 §9 C2.
- 사용자 함수가 낸 에러는 그대로 나간다. 틀린 모양·형의 값은 optimize와 같은 꼴의 에러다(`quad(): 'func' must return a real number, not shape (3,)`).

---

## 3. 수치 방법 (안)

### 3.1 QUADPACK과 `quad`의 옵션
scipy 1.18.1의 `quad`는 옵션으로 QUADPACK의 일곱 루틴 가운데 하나를 고른다(`_quadpack_py.py`의 `_quad`·`_quad_weight`):

| `quad`의 입력 | 루틴 | 규칙과 가속 |
|---|---|---|
| 유한 a, b, `weight=None`, `points=None` | QAGS(`dqagse`) | QK21, 전역 적응 이분 + ε-알고리즘(`dqelg`) 외삽 |
| 무한 a 또는 b(`points` 없음) | QAGI(`dqagie`) | (0, 1]로 바꾼 뒤 QK15(`dqk15i`) + ε-알고리즘 |
| `points=[...]` (유한 구간만) | QAGP(`dqagpe`) | 중단점으로 나눈 처음 구간들에서 QAGS의 방법 |
| `weight='cos'`·`'sin'`, 유한 구간 | QAWO(`dqawoe`) | 작은 구간은 QK15W, 진동이 많은 구간은 25점 Clenshaw–Curtis와 체비쇼프 모멘트(`maxp1`개까지) |
| `weight='cos'`·`'sin'`, [a, ∞) 또는 (−∞, b] | QAWF(`dqawfe`) | 반주기 묶음마다 QAWO, 그 합의 ε-알고리즘(`limlst`개 주기까지); (−∞, b]는 x → −x로 바꾼다. 양쪽 무한은 `ValueError` |
| `weight='alg'`, `'alg-loga'`, `'alg-logb'`, `'alg-log'` | QAWS(`dqawse`) | (x − a)^α (b − x)^β·로그 가중치의 수정 Clenshaw–Curtis |
| `weight='cauchy'` | QAWC(`dqawce`) | 코시 주값 1/(x − c)의 수정 Clenshaw–Curtis |

- scipy는 a > b를 받으면 끝을 바꿔 계산하고 결과의 부호를 바꾼다(`flip`). a == b는 0이다. 가중치를 주면 `points`를 무시하고 경고한다: sd에서는 함께 주면 `ValueError`다(O29·O62의 규칙: 쓰이지 않는 인자).
- `points`: scipy는 `np.unique`로 정렬하고 (a, b) 안의 것만 남긴다. 무한 구간과 함께는 `ValueError`(scipy와 같다).
- **status와 message** (O17): QUADPACK의 `ier` 0–7과 scipy의 글(구간 수 상한 1, 반올림 2, 나쁜 피적분함수 3, 외삽 표의 수렴 실패 4, 발산 5, 비정상 끝 7; QAWF는 1·4·7의 글이 다르다). `ier = 6`(입력이 틀림)은 scipy처럼 `ValueError`이고 글도 scipy의 갈래(`epsabs <= 0`과 `epsrel`의 하한, 중단점의 자리 등)를 따른다. `converged`는 `ier == 0`이다.
- **호출점의 비트** (C5): 규칙의 상수(xgk, wgk, wg)는 QUADPACK의 10진 리터럴 그대로다. scipy 1.18.1의 QUADPACK은 scipy가 Fortran에서 옮긴 C 판이고 clang의 기본 FP 축약으로 빌드된다고 본다(O2의 brentq에서 본 것과 같다). 점의 계산(c ± h·xgk, 이분점 0.5(a + b), QAGI의 a + (1 − t)/t)에는 축약할 곱셈-덧셈이 없거나 곱이 ±1이라 영향이 없다. 결정의 자리(구간 오차, ε-알고리즘의 표, 반올림 검출 `iroff`)의 축약은 C4에서 scipy의 호출 순서와 견주어 찾고 그 자리에 `std::fma`를 적는다(brentq와 같은 방법; optimize §3.2).
- 오차 추정은 QUADPACK의 것(`abserr`: |K21 − G10|을 (200·|K − G|/resasc)^1.5로 조이고 반올림 하한 50ε·resabs)이다. 정직함의 측정은 §7.4다: 대개 실제 오차보다 몇 자리 크지만, 좁은 봉우리를 처음 21점이 놓치면 틀린 값을 수렴으로 알린다(4⁻ᵅ/((x − π/4)² + 16⁻ᵅ), α = 19·20에서 값 ≈ 0, 참값 π, `ier = 0`). 이것은 scipy와 같게 두고 문서에 `points`로 피하는 법을 적는다.

### 3.2 `dblquad`, `tplquad`, `nquad`
- scipy의 `nquad`(`_NQuad`)처럼 가장 안쪽 변수부터 `quad`를 겹친다. 층마다 `opts`(`atol`·`rtol`·`limit`·`points`·`weight`·`wvar`; C11)를 받는다. `dblquad(func, a, b, gfun, hfun)`의 `func`는 scipy처럼 `func(y, x)`(안쪽 변수가 먼저)다.
- 돌려주는 오차는 scipy처럼 바깥 `quad`의 추정뿐이다(안쪽 적분의 오차는 더하지 않는다). 이 점을 문서에 적는다.
- `nfev`는 모든 층의 호출 합이다(scipy `nquad(full_output=True)`의 `neval`). 탐침: 단위 정사각형의 exp(−(x² + y²))는 441번(21 × 21), 0.09 ms다.

### 3.3 `quad_vec`
- scipy의 Python 판(`_quad_vec.py`)을 옮긴다: 전역 적응, 오차가 큰 구간을 힙에서 꺼내 반으로 나눔, 규칙 `'gk21'`(유한 기본)·`'gk15'`(무한 기본)·`'trapezoid'`, 무한 구간은 scipy의 변수 치환, 수렴은 `norm`(`'max'`·`'2'`)으로 잰 오차 ≤ max(`epsabs`, `epsrel`·‖I‖).
- scipy의 `cache_size`(구간 값의 기억), `workers`(다중 프로세스)는 없다(사용자 함수가 Python이고 sd는 다중 프로세스를 띄우지 않는다). `full_output`의 `intervals`, `integrals`, `errors`는 필드다.

### 3.4 `cubature`
- scipy의 Python 판(`_cubature.py`)을 옮긴다: 영역의 힙, 오차가 가장 큰 영역을 축마다 반으로 나눔(2ⁿ개), 규칙 `'gk21'`·`'gk15'`(1차원 Gauss–Kronrod의 곱: 점 21ⁿ·15ⁿ개), `'genz-malik'`(차수 7, 점 2ⁿ + 2n² + 2n + 1개; n ≥ 2). 무한 축은 scipy의 치환(x = (1 − |t|)/t, x = a + (1 − t)/t). `points`는 그 점을 지나게 영역을 처음에 나눈다.
- 이름 `'gauss-kronrod'`는 `'gk21'`의 별칭이라 받지 않는다(별칭 없음). 사용자 규칙 객체(scipy `_rules.Rule`), `workers`는 첫 판에 없다(§9 C7).
- 탐침(§7.4): 2차원 가우스를 gk21로는 첫 영역에서 끝내고(441점), genz-malik은 106번 2,491점이다.

### 3.5 `tanhsinh`
scipy의 `_tanhsinh.py`(Bailey–Jeyabalan–Li 2005, Vanherck–Sorée–Magnus 2020)를 원소마다 옮긴다.
- **변환**: 끝을 정리한다(a = b는 0, b < a는 바꿔 계산하고 부호를 바꿈, 양쪽 무한은 x = t/(1 − t²), 한쪽 무한은 x = 1/t − 1 + a, 왼쪽 무한은 x → −x). 가중치는 그 야코비안을 곱한다.
- **점과 가중치**: 기본 간격 h₀ = asinh(log(2/(4·tiny) − 1)/π)/8(xjc가 아래넘침하는 tmax의 1/8), 수준 k의 간격 h₀/2ᵏ, 수준 0은 j = 0…8, 다음 수준부터 홀수 j만. u₁ = (π/2)cosh(jh), u₂ = (π/2)sinh(jh), w = u₁/cosh(u₂)², 1 − x = 1/(exp(u₂)cosh(u₂)). 끝에 닿은 점(반올림)은 가중치 0이다.
- **호출점의 비트** (C5): scipy의 표는 numpy의 `cosh`·`sinh`·`exp`로 계산하는데 이 점들에서 macOS libm과 같다. 그래서 C++에서 `std::cosh`·`std::sinh`·`std::exp`(libm)와 곱 c·c(numpy의 `**2`)로 계산하면 수준 0–10의 16,386값이 모두 비트까지 같다(§7.4; Python의 `x**2`를 쓰면 11값이 다르다). sd의 원소별 `exp`·`cosh`·`sinh`(vForce)는 이 점의 약 3분의 1에서 libm과 다르므로 쓰지 않는다. 표는 처음 쓸 때 한 번 만들어 둔다.
  - 이 표 자체는 정확하지 않다: 같은 식의 참값과 견주면 최대 980 ULP, 평균 47 ULP다(u₂의 반올림이 exp(u₂)에서 u₂배로 커진다). 적분값에는 거의 영향이 없지만(꼬리의 가중치가 작다), Arb로 만든 정확한 표로 바꿀지는 §9 C6의 질문이다.
- **합과 오차 추정**: 수준마다 Euler–Maclaurin 합 Sₙ = h·Σ f_j w_j(이전 수준은 Sₙ₋₁/2를 더함)과 scipy §5의 추정: d1 = |Sₙ − Sₙ₋₁|, d2 = |Sₙ − Sₙ₋₂|, d3 = ε·max|f_j w_j|, d4 = 가장 바깥 항의 크기, e = clip(max(d1^(log d1/log d2), d1², d3, d4), ε|Sₙ|, d1). 멈춤은 e < atol 또는 e < rtol·|Sₙ|(기본 rtol = ε^0.75, atol = 0), 수준 상한 10(`maxlevel`), 시작 수준 2(`minlevel`: 처음 한 번의 호출에 수준 0–2의 점을 모두 넣는다).
  - scipy의 Σ는 numpy의 pairwise 합(한 줄을 따라: 8개 누산기, 128개 단위)이다. 이 합이 멈춤 판정을 정하므로 호출점의 비트를 위해 이 자리에서는 numpy의 합 순서를 그대로 쓴다(줄의 길이가 수준 10에서 16,384 이하라 비용이 작다). 다른 리덕션(sd의 16누산기)을 쓰면 판정의 경계에서 수준이 하나 달라질 수 있다.
  - 유한하지 않은 f_j는 가장 가까운 유한한 끝의 값으로 바꾼다(scipy, van Engelen의 제안). Sₙ이 유한하지 않으면 status −3이다.
- `log=True`: f가 log(피적분함수)를 돌려주고 합은 logsumexp다(실수 피적분함수가 음수면 log가 복소수다: 복소수의 질문과 함께 §9 C6).

### 3.6 `nsum`
scipy의 `nsum`을 옮긴다: 항 수가 `maxterms` 이하면 직접 합, 아니면 적분 판정의 경계. ∫ₐ^∞ f를 `tanhsinh`으로 구해 하한으로 삼고, f(c) < atol + rtol·S_l인 c를 2의 거듭제곱 걸음으로 찾고(없으면 c = a + maxterms·step), 합 = Σ_{k<c} f(k) + ∫_c^∞ f + f(c)/2, 오차 = f(c)/2 + 적분의 오차. 왼쪽 무한은 x → −x, 양쪽 무한은 f(x) + f(−x)(x = 0의 가중치 1/2)의 오른쪽 합이다. 기본 rtol = ε^0.5. f는 매끄럽고 양이고 단봉이어야 한다(scipy의 계약). scipy 1.18.1의 코드는 단조 감소를 가정한 경계이고, 단봉 함수의 봉우리를 따로 찾지 않는다(문서의 "봉우리 둘레를 직접 더함"은 직접 합이 덮을 때만이다).
- 직접 합은 scipy처럼 점들을 한 번에 부르고(`(1, 8192)` 같은 큰 호출) 합의 순서는 numpy의 것이다(§3.5와 같은 까닭).
- 탐침: Σ1/k²는 10번 호출, `nfev` 8,561, 실제 오차 3×10⁻¹³, 추정 7.4×10⁻⁹였다.

### 3.7 표본의 규칙과 `fixed_quad`, `newton_cotes`
- **`trapezoid`, `simpson`, `romb`, `cumulative_trapezoid`, `cumulative_simpson`**: 표본 배열에 대한 원소별 연산과 리덕션이다. 식은 scipy의 것(`simpson`은 scipy 1.11부터의 꼴: 짝수 개 점의 마지막 구간을 Cartwright의 보정으로; `cumulative_simpson`은 scipy의 구간별 이차식)이고, 합은 sd의 합 규칙(docs/api/array.md §7.8)이다. 그래서 scipy·numpy와 끝자리가 다를 수 있고(D-33), 스레드 수와 상관없이 같은 비트다. `trapezoid`는 numpy의 `np.trapezoid`와 같은 함수다(scipy의 것이 numpy의 것과 비트까지 같다, §7.4): sd 배열 쪽에는 `trapezoid`가 없으므로 `sd.integrate.trapezoid` 하나만 둔다.
  - dtype: 배열의 규칙(int·bool은 float64로; complex128은 받는다: 실수부·허수부의 합). `axis`는 scipy·numpy처럼 기본 −1(§9 C3).
- **`fixed_quad(func, a, b, n=5)`**: n점 Gauss–Legendre. scipy는 `roots_legendre(n)`을 쓰고 f를 n점 배열로 한 번 부른다(벡터화된 f). 점과 가중치를 어디서 얻는지가 §9 C3의 질문이다: scipy의 `roots_legendre`는 n = 5에서 Arb와 같지만 n = 100에서 점 8 ULP, 가중치 83,682 ULP, n = 500에서 가중치 1.5×10⁷ ULP까지 다르다(§7.4). 추천은 올바로 반올림한 점과 가중치(호출점이 scipy와 n ≥ 20에서 다를 수 있다: C5의 예외)다.
- **`newton_cotes(rn, equal=0)`**: 닫힌 Newton–Cotes 가중치와 오차 계수. 정수 점(`rn`이 int)이면 정확한 유리수로 계산해 반올림한다(scipy는 같은 간격이면 표·유리수, 아니면 Vandermonde 풀이).

### 3.8 `derivative`, `jacobian`, `hessian`
scipy의 `_differentiate.py`를 원소마다 옮긴다.
- **방법**: Richardson 외삽이 아니다(scipy의 주석이 그것을 "나중 일"로 적는다). order차(기본 8, 홀수는 올림) 차분 공식을 간격을 step_factor(c, 기본 2)씩 줄이며 되풀이한다. 중앙 스텐실은 x ± h/cᵏ(k = 0…n − 1, n = order/2)이고 다음 반복은 x ± h/cⁿ 두 점만 새로 부른다. 한쪽 스텐실(`step_direction` ≠ 0)은 d = √c로 x + h/dᵏ(k = 0…2n − 1), 다음 반복에 두 점이다. 가중치는 Vandermonde 계를 풀어 얻는다(scipy는 `np.linalg.solve`; 중앙 가중치는 반대칭을 강제한다).
- **멈춤**: 오차 = |dfₙ − dfₙ₋₁|, error < atol + rtol·|df|이면 0(기본 atol = 가장 작은 정규수, rtol = √ε), 유한하지 않으면 −3, 오차가 직전의 10배를 넘으면 −1(반올림이 이기기 시작함), `maxiter`(기본 10)면 −2.
- **호출점의 비트** (C5): 점은 원소별 식(x + h/cᵏ)이라 scipy와 같게 계산할 수 있다. 다만 df는 `fc @ wc`(numpy의 행렬·벡터 곱 = BLAS dgemv, 지금 Accelerate)라 합의 순서가 scipy 쪽의 BLAS에 달려 있고, 멈춤 판정의 경계에서 반복 수가 갈릴 수 있다(호출점의 예외로 적고 C2에서 갈리는 비율을 잰다).
- **가중치의 정확도**: scipy의 order 8 중앙 가중치는 정확한 유리수 값과 9개 가운데 8개가 다르고 최대 상대 2.7×10⁻¹⁴다(order 12면 1.7×10⁻¹¹; §7.4). c는 double이라 유리수이므로 중앙 가중치는 정확한 유리수 풀이로, 한쪽 가중치(√c)는 Arb로 올바로 반올림할 수 있다. 가중치는 점에 영향을 주지 않으므로 호출점은 그대로다. 어느 쪽을 쓸지는 §9 C2의 질문이다.
- **`jacobian(f, x)`**: x (m, …)에서 변수 i만 바꾼 복사본 m개를 쌓아 f를 한 번에 부른다(f는 (m, m, …, p)를 받는다). 결과 `df` (k, m, …), 원소마다의 `status`·`n_iter`·`nfev`. **`hessian`**: `jacobian`의 `jacobian`, 안쪽 rtol = rtol/100. scipy는 rtol < 100ε에 경고하고 100ε로 올린다: 경고가 없으므로 §9 C2. `nfev`는 원소마다 안쪽 호출을 합한 (m, m) 배열이다(scipy는 33번 호출에 121–297, Rosenbrock n = 3; §7.4).
- scipy의 기본 `initial_step = 0.5`는 절대 간격이라 x의 척도를 보지 않는다: log의 x = 10⁻³에서 x − 0.5 < 0이 NaN이라 status −3이다(§7.4). scipy 그대로 두고(기본값은 scipy의 것) 문서에 `initial_step`·`step_direction`을 적는다.

---

## 4. 정확도 계약 (안)
- **적응 방법**(`quad` 계열, `quad_vec`, `cubature`, `tanhsinh`, `nsum`): 결과는 "멈춤 판정을 통과한 값"이고, 오차 추정은 엄밀한 상계가 아니다(QUADPACK, tanh-sinh 모두 경험적이다). 그래서 수치 계약은 (1) scipy와 같은 걸음(§7.3의 기록과 호출 순서·`nfev`·`status`가 같다)과 (2) 시험 모음에서 잰 정확도와 정직함의 표다. 표는 `bench/<영역>/accuracy.md`로 만들고 기준선과 견준다(optimize의 "결과로 확인되는 판정 + 기준선 비교"와 같다).
  - **정직함**: ρ = 오차 추정 / |참 오차|(Arb). converged이고 ρ < 1인 경우를 세어 표에 적는다. 첫 탐침(§7.4)에서 `quad`는 QUADPACK 책의 세 무리 62경우 가운데 수렴이라고 알린 것 가운데 둘(봉우리, α = 19·20)이 ρ < 1이고, `tanhsinh`은 수렴한 46경우 가운데 여섯이 ρ < 1(최소 0.015)이었다.
- **표본 규칙**: 같은 double 입력에 그 규칙(식)을 정확히 계산한 값(Arb)과 견준다. 계약은 리덕션의 오차 한계(n·ε·Σ|항|의 크기; 배열의 합 규칙)다. 구적 자체의 이산화 오차는 계약이 아니다.
- **`fixed_quad`, `newton_cotes`**: 점·가중치는 올바로 반올림(§9 C3의 추천)이면 그것이 계약이다(0 ULP). 값은 n점 합의 오차 한계다.
- **`derivative` 무리**: 수렴한 결과는 error < atol + rtol·|df|이고, 해석적 도함수의 Arb 값과 견준 상대오차를 표에 적는다(탐침: exp·sin의 1에서 1.2×10⁻¹⁴·9×10⁻¹⁵).
- scipy·Arb와 다른 동작은 API 문서에 `[scipy와 다름]`으로 적는다.

---

## 5. 성능 (안)
- 시간의 대부분은 사용자의 Python 함수다. 기준은 (1) 같은 문제의 `nfev`(scipy와 같다), (2) 콜백을 뺀 오버헤드, (3) 원소별 무리의 처리량(원소 10–10⁶)이다.
- **`quad`**: scipy는 C 고리에서 점마다 Python을 부른다. 잰 값: exp(−x²)의 [0, 1]은 21번에 2.4 µs(함숫값 하나에 113 ns), (−∞, ∞)는 270번에 16.8 µs(62 ns); 같은 함수의 맨 Python 호출은 52 ns다. sd도 점마다 부르므로 목표는 scipy와 같거나 빠름(사이 계산은 수 ns)이다. GIL은 쥔 채다(호출 사이의 일이 10 µs보다 훨씬 작다).
- **원소별 무리**: scipy는 반복마다 numpy 연산 수십 개를 하므로 작은 배치에서 느리다. 잰 값: `tanhsinh`(exp, 수준 3)은 원소 10개 0.44 ms(함숫값 하나 512 ns), 10³ 1.27 ms(15.4 ns), 10⁵ 103 ms(12.5 ns); `derivative`(exp)는 10개 0.41 ms, 10³ 0.51 ms(46 ns), 10⁵ 22.3 ms(20 ns). optimize의 배치 고리는 호출 사이의 일이 함숫값 하나에 4–10 ns라(optimize §3.3.1) 작은 배치에서 수십 배, 큰 배치에서 사용자 함수가 numpy·sd 원소별이면 1.5–3배를 목표로 한다.
- **`cubature`·`quad_vec`**: scipy의 Python 고리와 힙이 오버헤드다(2차원 gk21 첫 호출 7.4 ms는 대부분 준비; genz-malik 106번 1.6 ms). C++ 힙으로 옮긴다.
- **표본 규칙**: 배열의 원소별 연산과 리덕션의 속도(배열 영역의 기준)다.
- 측정은 1·10스레드. 사례는 `bench/speed/cases/` 아래 영역 파일에 마일스톤마다 더한다(§7.5).
- **C0의 scipy 시간** (2026-11-21): 기계가 바빠(다른 작업 둘이 돌았다, 부하 2.0) 기준선을 재지 않았다. `bench/speed/cases/integrate.py`·`differentiate.py`의 계획한 사례 26개(아직 공개 함수가 없어 `cases()`는 비었고 `run integrate differentiate --smoke`는 고를 사례가 없다고 알린다)에서 scipy 쪽만 데운 뒤 한 번씩 불렀다(점검 실행과 같은 한 번의 호출이라 잡음이 크다). 기준선(`bench/speed/results/`)에 넣지 않았고, 사례가 공개되는 마일스톤에서 조용한 기계로 다시 잰다.

| 사례 | scipy | 사례 | scipy |
|---|---|---|---|
| `trapezoid` 10³ / 10⁶ | 14.7 µs / 694 µs | `quad` exp(−x²) [0, 1] (21번) | 5.4 µs |
| `simpson` 10³ / 10⁶ | 61 µs / 4.45 ms | `quad` exp(−x²) (−∞, ∞) (270번) | 16.2 µs |
| `cumulative_trapezoid` 10³ / 10⁶ | 15.3 µs / 2.65 ms | `quad` 봉우리 α = 5 (525번) | 39.8 µs |
| `cumulative_simpson` 10³ / 10⁶ | 44 µs / 9.28 ms | `dblquad` 단위 정사각형 (441번) | 68 µs |
| `fixed_quad` n = 5 | 14.0 µs | `quad` `weight='cos'` [0, ∞) | 13.9 µs |
| `tanhsinh` exp 10 / 10³ / 10⁵ | 0.46 / 1.11 / 79 ms | `nsum` 1/k² | 0.86 ms |
| `cubature` gk21 / genz-malik 2차원 | 0.32 / 1.52 ms | `quad_vec` 함수 셋 | 0.17 ms |
| `derivative` exp 10 / 10³ / 10⁵ | 0.29 / 0.45 / 17.4 ms | `jacobian` / `hessian` Rosenbrock n = 3 | 0.28 / 4.08 ms |

---

## 6. 층과 파일 (안)

### 6.1 차분 간격 커널 옮기기 (C3)
- 지금: `csrc/optimize/diff.hpp`(`rel_step_2point`, `rel_step_3point`, `absolute_step`, `adjust_one_sided`, `adjust_two_sided`)를 `csrc/optimize/lsq.cpp`(least_squares·nls의 야코비안), `csrc/optimize/objective.hpp`(minimize의 차분 기울기), `csrc/optimize/hessian.hpp`(차분 헤세 곱), `csrc/optimize/scalar.cpp`(차분 newton)가 쓴다.
- 안: 이 헤더를 csrc 아래 새 `differentiate/` 폴더의 `steps.hpp`(이름공간 `sd::differentiate`)로 옮기고, 네 파일은 include와 이름만 바꾼다. 차분의 고리(호출 수 세기, 한쪽이 유한하지 않을 때 반대쪽 다시 해 보기, 기억)는 부르는 쪽마다 다르므로 그 자리에 둔다. 식과 계산 순서가 같아 비트가 그대로이고, optimize·regression(nls)의 골든이 바뀌지 않음을 확인한다(C1).
- Python 쪽의 `src/stardust/optimize/_diff.py`의 `Differences`(nls의 정확한 5점 차분, `approx_fprime`의 기본)도 differentiate의 Python 내부(`_differences.py`)로 옮긴다(C15). `regression.nls`와 `approx_fprime`·`check_grad`가 그것을 부르고, `approx_fprime`·`check_grad`는 `sd.optimize`의 공개 이름 그대로다. nls·optimize의 결과는 바이트까지 같아야 한다.
- ARCHITECTURE.md §1의 그림에 optimize/ ──▶ differentiate/가 생긴다(differentiate/는 array/·parallel/만 쓴다).

### 6.2 함께 쓰는 반복 고리 (C4)
- 지금 `csrc/optimize/batch.cpp` 안에 있는 고리의 부품: `for_elements`(2¹⁵개부터 스레드에 나눔, GIL 놓기), `gather`(남은 원소의 `args` 모으기), `Call`(남은 원소의 점을 새 float64 `sd.ndarray`에 써서 사용자 함수를 한 번 부르고 값을 검사), `Progress`(진행 중인 원소의 목록과 멈춘 원소의 `status`·`n_iter`·`nfev`), `advance`(갱신·판정 뒤 남은 원소를 순서대로 줄이기). 원소의 상태는 알고리즘마다의 구조체다.
- 안: 이 부품을 헤더 하나(scipy `_lib/_elementwise_iterative_method.py`에 해당)로 끌어내 optimize의 네 함수, `derivative`, `tanhsinh`, `nsum`이 함께 쓴다. 자리는 csrc 아래 새 바닥 폴더 `elementwise/`다(C16; 세 영역이 아래로만 의존한다. ARCHITECTURE.md §1의 그림에 optimize/·integrate/·differentiate/ ──▶ elementwise/ ──▶ array/·parallel/이 생긴다).
- 넓힐 것: 원소마다 p개 점의 호출(`Call`이 (k, p)를 만들고 값 (k, p)를 검사), 수준마다 p가 바뀌는 호출(`tanhsinh`), 원소마다 다른 점 수(`nsum`의 직접 합은 원소마다 항 수가 다르다: scipy는 가장 큰 수로 맞추고 남는 자리를 NaN으로 채운다), 결과 필드의 모음. optimize 쪽은 p = 1이라 그대로의 비트다(골든 불변).
- 끌어내기는 optimize의 코드를 바꾸는 일이라 C1에서 하고, optimize의 배치 테스트·골든·속도(`bench/optimize/speed.md`의 배치 사례)가 그대로인지 본다.

### 6.3 이 영역의 파일
- Python: `src/stardust/{integrate,differentiate}/`. integrate는 `_quadpack.py`(`quad`, `dblquad`, `tplquad`, `nquad`), `_quad_vec.py`, `_cubature.py`, `_tanhsinh.py`(`tanhsinh`, `nsum`), `_rules.py`(`fixed_quad`, `newton_cotes`, 표본 규칙), `_result.py`, `_common.py`. differentiate는 `_derivative.py`(`derivative`, `jacobian`, `hessian`), `_result.py`.
- C++: `csrc/{integrate,differentiate}/`. integrate는 `quadpack.cpp`(dqagse·dqagie·dqagpe·dqawoe·dqawfe·dqawse·dqawce와 규칙 dqk15·dqk21·dqk15i·dqk15w·dqc25f·dqc25s·dqc25c, ε-알고리즘 dqelg; 상수는 QUADPACK의 리터럴), `adaptive.cpp`(`quad_vec`·`cubature`의 힙), `tanhsinh.cpp`(표, 합, 오차 추정; `nsum`), `samples.cpp`(표본 규칙; 합은 배열의 커널), `gauss.hpp`(Gauss–Legendre). differentiate는 `steps.hpp`(C3), `derivative.cpp`(스텐실, 가중치, `jacobian`·`hessian`). 바인딩 `csrc/bind/{integrate,differentiate}_bind.cpp`(`stardust._core.integrate`, `stardust._core.differentiate`).
- 테스트·측정·골든은 영역마다(§7.5): `tests/{integrate,differentiate}/`, `tests/golden/cases_{integrate,differentiate}.py`, `bench/speed/cases/{integrate,differentiate}.py`, `bench/accuracy/{integrate,differentiate}.py`.

---

## 7. 참값과 검증

### 7.1 시험 문제 (C6)
- **QUADPACK 책**(Piessens 외 1983)의 시험 적분 무리: ∫₀¹ x^α log(1/x)(끝점 특이점, α = −0.9…2), ∫₀¹ 4^−α/((x − π/4)² + 16^−α)(좁은 봉우리, α = 0…20), ∫₀^π cos(2^α sin x) = πJ₀(2^α)(진동, α = 0…10), 내부 특이점(|x − λ|^α 꼴), 가중치 루틴의 예(QAWO의 ∫₀¹ log(x) sin(10πx) 같은 유한 구간 진동, QAWF의 반무한 푸리에 적분, QAWS의 x^α(1 − x)^β·로그, QAWC의 주값). 무리의 정확한 목록은 C0에서 책으로 확인한다. 책의 표(필요한 수와 오차)는 문헌의 값이라 문제의 식과 닫힌 꼴만 모음에 쓴다.
- **scipy의 시험**: `scipy/integrate/tests/`의 `test_quadpack.py`(가중치·중단점·무한 구간·복소수의 경우), `test_tanhsinh.py`(Bailey–Jeyabalan–Li의 f1–f15와 적분 변환·로그 경우), `test_cubature.py`, `test__quad_vec.py`, `test_quadrature.py`(표본 규칙), `scipy/differentiate/tests/test_differentiate.py`. 식과 참값을 옮기고 출처를 적는다(optimize의 `tests/optimize/functions.py`와 같은 방식).
- **어려운 적분**(Arb 참값): 끝점 특이점(x^−0.9, log(x)², x^−1/2 log x), 내부의 꺾임과 특이점(|x − 1/3|, 중단점과 함께), 진동(cos(100x), sin(1/x)의 구간), 무한 구간(e^−x²의 실수 전체, e^−x x^−1/2의 [0, ∞): 끝점 특이점과 무한 구간이 함께), 급수(ζ(p), 교대급수, 양쪽 무한의 e^−k²).
- **미분**: 초등 함수 11개(optimize O1의 것)의 9점, 끝 근처(log의 10⁻³, √의 10⁻⁶, `step_direction`), 도함수가 0인 점(x³의 0: 기본 atol로는 status −2, §7.4), 큰 값(exp의 30), 야코비안·헤시안은 MGH 문제(`tests/optimize/mgh.py`; 해석적 야코비안과 `ad.Dual2`의 헤시안).
- **C0에서 만든 것** (`tests/integrate/problems.py`, `tests/differentiate/problems.py`): 피적분함수는 한 번만 쓰고 이름공간 셋(Python float와 `math`: `quad`의 호출, numpy 배열: `tanhsinh`·`nsum`의 호출, Arb `acb`와 `analytic` 깃발: 참값)에서 돈다. 문제의 상수와 끝(π/4, 상한 π, 중단점, 가중치의 매개변수)은 코드의 double이고, 참값은 double로 낸 문제 그대로의 정확한 적분이다.
  - 1차원 적분 116개: QUADPACK 책의 세 무리 62(x^α log(1/x) 30, 봉우리 21, cos(2^α sin x) 11), QUADPACK 루틴마다의 고전 예제 7(QAGS log(x)/√x, QAGP x³ log|(x² − 1)(x² − 2)|와 중단점 1·√2, QAGI log(x)/(1 + 100x²), QAWO log(x) sin(10πx), QAWF x^−1/2 cos(πx/2), QAWS log(x)/(1 + log²x)², QAWC 1/(5x³ + 6)의 주값), scipy `test_quadpack.py`의 1차원 경우 14, `test_tanhsinh.py`의 f1–f15와 정규 밀도의 변환 다섯, 어려운 적분 13(x^−1/2, x^−0.9, x^−1/2 log x, cos(100x), [0.01, 1]의 sin(1/x), e^−x², 1/(x² + 10⁻⁶), |x − 1/3|^α(α = −0.5, 0.5, 1)를 중단점과 함께·없이).
  - 책의 무리의 목록은 책을 갖고 있지 않아 책으로 확인하지 못했다. 설계 초안의 세 무리(Gonnet 2012 등이 책의 무리로 인용하는 꼴)와 루틴 예제를 쓰고, 모든 닫힌 꼴을 Arb의 수치 적분이나 scipy와 견주었다(아래 §7.2, `test_problems.py`). 책의 다른 무리는 책을 볼 수 있을 때 더한다.
  - 무리 셋의 배치(`tanhsinh`에 매개변수를 `args` 배열로; `BATCHES`), 겹친 적분 7(scipy `test_quadpack.py`의 `dblquad`·`tplquad`와 단위 정사각형·평면·정육면체의 가우스), 벡터 값 2, 급수 9(ζ(2), ζ(3), ζ(1.5), H(1000, 2), 걸음 0.5, 1/(k² + 1), e^−k/10, 양쪽 무한의 e^−k², 왼쪽 무한), 표본 12벌(같은 간격 2–4,097점, 제곱 간격, 2차원).
  - 미분: 초등 함수 11개(exp log sin cos tan sqrt atan tanh x³ 1/x e^−x²) × 9점과 끝 근처 14경우(`EDGES`), MGH 47경우(헤시안은 n ≤ 10의 42경우).

### 7.2 Arb 참값 (`tools/<영역>_ref.py` → `tests/<영역>/ref/`)
- `acb.integral`은 피적분함수가 구간에서 해석적이고 유계일 때 빠르고 엄밀하다(cos(16 sin x)의 [0, π] 1 ms, cos(100x)의 [0, 1] 1 ms 미만). 끝점에서 유계가 아니면 NaN(무한 공)을 돌려준다: x^−1/2, x^−0.9, log x/√x의 [0, 1]이 모두 그랬다(eval_limit를 늘려도). 그래서 참값은 세 방법이다.
  1. 닫힌 꼴(Arb의 함수로): 무리 대부분(1/(1 + α)², atan의 합, πJ₀(2^α), √π, ζ(p) 등).
  2. `acb.integral`: 해석적이고 유계인 피적분함수(가지 자름이 있는 함수는 `analytic` 깃발을 넘긴다).
  3. 치환으로 끝점 특이점을 없앤 뒤 `acb.integral`(x = t^k, 무한 구간은 x = t/(1 − t)나 꼬리의 엄밀한 상계와 자르기).
- 형식은 `tests/reffile.py`와 `_provenance`(다른 영역과 같다). 각 참값에 방법(닫힌 꼴, 적분, 치환)을 적는다. 미분의 참값은 해석적 도함수를 Arb로 계산한 값이다.
- **C0에서 만든 것**: `tools/integrate_ref.py`(→ `tests/integrate/ref/integrals.json` 116개, `series.json` 9, `multiple.json` 7, `vector.json` 2)와 `tools/differentiate_ref.py`(→ `tests/differentiate/ref/derivatives.json` 104점의 f′·f″, `mgh.npz`: 야코비안 47·헤시안 42). 값은 2⁻¹¹⁰까지 정한 공(double-double)이다. 문제가 허락하는 방법을 모두 계산하고 공이 겹쳐야 한다(겹치지 않으면 도구가 멈춘다): 116개 가운데 닫힌 꼴 103·`acb.integral` 13이 저장되고, 18개는 `acb.integral`, 5개는 치환이 함께 확인했다. 항목마다 문제 정의의 해시가 있어 문제를 바꾸면 테스트가 낡은 참값을 알린다.
  - `acb.integral`은 끝에서 유계인 대수 특이점(√(1 − t²), √t)은 받는다(NaN은 유계가 아닐 때뿐이다). 치환은 x = t^(1/(1+α))(x^−0.9: 상수), x = u²(e^−x x^−1/2: 2e^−u²), x = e^−u(QAWS 예제)와 꼬리의 엄밀한 상계(e^−144/12, e^−90/90³)다. 확인 하나가 실제로 잘못을 잡았다: x^−0.9의 치환 상수를 double로 쓴 첫 판이 닫힌 꼴과 겹치지 않았다.
  - 헤시안의 참값은 Arb 위의 2차 전진 미분(`tests/differentiate/jets.py`, optimize `ad.Dual2`의 Arb 판)이고, 야코비안은 optimize의 `derivatives.npz`와 같은 값이다.
  - double 상한의 차이: f9 log(cos t)와 f10 √tan t의 상한 `math.pi / 2`는 π/2보다 6.1×10⁻¹⁷ 작아 참값이 닫힌 꼴과 2×10⁻¹⁵·1.6×10⁻⁸ 다르다(남는 조각을 엄밀한 상계와 함께 뺐다). cos(2^α sin x)의 상한 π도 같아 그 무리는 `acb.integral`이 참값이고 πJ₀(2^α)와 2.5×10⁻¹⁶ 안이다.

### 7.3 scipy 1.18.1의 기록 (`tests/<영역>/data/`)
- 함수마다 시험 모음 전부의 결과(값, 오차, `nfev`, `status`)와, 호출점 비트 동일을 볼 함수(`quad` 계열, `tanhsinh`, `nsum`, `derivative`)의 호출 순서 전부(점의 hex)를 기록한다. 테스트는 sd의 호출 순서와 견준다(optimize O9의 방식: scipy 판이 1.18.1이 아니면 실패).
- netlib Fortran 원본과의 대조(C17): 받은 뒤 optimize O13처럼 gfortran(`-O2 -ffp-contract=off -std=legacy`)으로 돌린 결과를 고정 데이터로 둔다. 받을 파일은 §7.6이다.
- **C0에서 만든 것**: `tools/gen_integrate_scipy.py` → `tests/integrate/data/`의 `scipy_quad.json`(116개: 값, 오차, `nfev`, scipy 자신의 분기에서 읽은 `ier`, 글, `infodict`), `scipy_calls.npz`(`quad`의 점 전부를 부른 순서대로), `scipy_tanhsinh.json`(101개와 배치 셋), `scipy_nsum.json`, `scipy_multiple.json`(`dblquad`·`tplquad`·`nquad`, 상자 영역의 `cubature` 규칙 셋), `scipy_vector.json`, `scipy_samples.json`(표본 규칙, `fixed_quad`, `newton_cotes`); `tools/gen_differentiate_scipy.py` → `tests/differentiate/data/`의 `scipy_derivative.json`(점마다·배치·`EDGES`), `scipy_jacobian.json`, `scipy_hessian.json`. 배열을 넘기는 함수의 호출은 모양의 차례(연속 길이로 줄여)와 점 수, 모든 점의 바이트의 SHA-256이다(`tanhsinh`은 적분 하나에 16,387점까지 불러 점을 다 두지 않는다; 테스트는 자기 호출의 해시와 견준다). 테스트는 scipy 판이 1.18.1이 아니면 실패하고, 기록의 표본을 다시 돌려 비트까지 같은지 본다.

### 7.4 설계 초안의 탐침 (scipy 1.18.1, 2026-11-21)
**한 문제씩** (`quad` 기본 허용오차 1.49×10⁻⁸, `tanhsinh` 기본; 참값은 닫힌 꼴과 `acb.integral`; ρ = 오차 추정/참 오차)

| 적분 | quad 상대오차 | quad ρ | quad nfev | tanhsinh 상대오차 | tanhsinh ρ | tanhsinh nfev, status |
|---|---|---|---|---|---|---|
| x^−1/2 log(1/x), [0, 1] | 4.2e-15 | 16 | 315 | 1.1e-16 | 2.0 | 67, 0 |
| x^−1/2, [0, 1] | 3.3e-16 | 5.7 | 231 | 1.1e-16 | 537 | 67, 0 |
| x^−0.9, [0, 1] | 1.8e-16 | 366 | 231 | 3.6e-16 | 34 | 67, 0 |
| 4⁻⁵/((x − π/4)² + 16⁻⁵), [0, 1] | 2.8e-15 | 2.8e5 | 525 | 1.2e-6 | 1.3e3 | 16,387, −2 |
| cos(16 sin x), [0, π] | 2.0e-16 | 6.7e7 | 147 | 1.4e-15 | 19 | 259, 0 |
| \|x − 1/3\|, [0, 1] | 0 | 25 | 189 | 8.3e-8 | 0.021 | 16,387, −2 |
| cos(100x), [0, 1] | 1.3e-13 | 9.0e5 | 315 | 1.1e-14 | 72 | 515, 0 |
| e^−x², (−∞, ∞) | 1.3e-16 | 9.8e7 | 270 | 1.3e-16 | 2.8 | 515, 0 |
| 1/(1 + x²), [0, ∞) | 0 | 4.2e6 | 45 | 4.5e-15 | 6.1 | 131, 0 |
| e^−x x^−1/2, [0, ∞) | 2.3e-13 | 236 | 345 | 1.2e-8 | 0.012 | 16,387, −2 |
| log(x)², [0, 1] | 1.1e-15 | 4.4 | 315 | 1.1e-16 | 63 | 67, 0 |
| 1/(x² + 10⁻⁶), [−1, 1] | 2.9e-16 | 5.8e4 | 819 | 9.4e-3 | 15 | 16,387, −2 |

- 가중치 루틴: QAWF sin(x)/x의 [0, ∞)는 상대 1.1×10⁻¹¹(추정 3.8×10⁻⁹, 430번), QAWS x^−1/2 log x는 0(40번), QAWC 주값 ∫₋₁¹ 1/(x − 0.3)은 0(25번).
- `tanhsinh`은 내부의 꺾임·좁은 봉우리·끝점 특이점과 무한 구간이 함께인 경우(e^−x x^−1/2)에 수준 10까지 가서 실패하고(status −2), 그때 오차 추정은 참 오차보다 50–80배 작다.

**QUADPACK 책의 세 무리**
| 무리 | 경우 | quad | tanhsinh |
|---|---|---|---|
| x^α log(1/x), α = −0.9…2.0 | 30 | 모두 경고 없음, 최대 상대 8.1e-11, ρ ≥ 1.4 | 모두 수렴, 최대 상대 3.9e-16, ρ ≥ 1.3 |
| 봉우리 α = 0…20 | 21 | α ≤ 9 정확(경고 없음); α = 10–18은 값 ≈ 0에 경고(ier ≠ 0); **α = 19·20은 값 ≈ 0(참값 π)인데 경고 없음** | α ≤ 4만 수렴(그 가운데 셋 ρ = 0.015–0.054); α ≥ 5 실패 |
| cos(2^α sin x), α = 0…10 | 11 | α ≤ 8 경고 없음; α = 9·10 경고(α = 10 상대 2.0; `limit=500`이면 6.6e-14) | 모두 수렴, 최대 상대 5.7e-13, 그 가운데 셋 ρ = 0.22–0.89 |

**호출의 모양**
- `quad`: f는 Python `float` 하나를 받는다. [0, 1]의 exp는 21번이고 순서가 dqk21(가운데, Gauss 쌍, Kronrod 쌍)이며 c ∓ h·xgk(QUADPACK의 리터럴)와 21점 모두 비트까지 같다. exp(−x)의 [0, ∞)는 135번(QK15 9구간).
- `tanhsinh`: 스칼라 입력이면 `()`(가운데) 뒤 `(1, 66)`(수준 0–2). 원소 셋이면 `(3,)`, `(3, 66)`, `(2, 64)`, `(1, 128)`: 수준마다 남은 원소가 한 번에. 끝점을 부를 수 있어(e^−x x^−1/2의 x = 0) Python `math`로 쓴 f는 `ZeroDivisionError`를 낸다.
- `nsum`(Σ1/k²): 10번, 모양 `()`, `(1,)`, `(1, 66)`, `(1, 21)`, …, `(1, 8192)`.
- `derivative`(exp, 1): `()`, `(1, 8)`, `(1, 2)`, `nfev` 11, 반복 2, 상대 1.2e-14. log의 10⁻³과 √의 10⁻⁶는 첫 반복의 NaN으로 status −3, x³의 0은 반복 10번 뒤 −2(`nfev` 27).
- `jacobian`(Rosenbrock, n = 3): 3번, `(3, 3)`, `(3, 3, 8)`, `(3, 3, 2)`. `hessian`: 33번, 원소의 `nfev` 121–297, 상대 1.3e-12.

**표와 가중치**
- tanh-sinh 표: libm(`math.cosh`·`sinh`·`exp`)과 c·c로 scipy의 수준 0–10 표 16,386값이 모두 같다. `x**2`(pow)로 하면 11값이 다르다. sd의 `exp`·`cosh`·`sinh`는 수준 10의 8,193점에서 libm과 2,167·2,903·2,797점이 다르다. 표는 같은 식의 참값(Arb)과 최대 980 ULP, 평균 47 ULP 다르다.
- `derivative`의 중앙 가중치(c = 2): 올바로 반올림한 유리수 값과 order 4에서 4/5개(최대 상대 1.7e-16), 6에서 6/7(1.2e-15), 8에서 8/9(2.7e-14), 10에서 10/11(8.0e-13), 12에서 12/13(1.7e-11)이 다르다.
- `scipy.special.roots_legendre`와 Arb(`arb.legendre_p_root`): n = 5는 점·가중치 모두 같다(가중치 17 ULP 이내), n = 20 점 1 ULP·가중치 261 ULP, n = 100 점 8·가중치 83,682 ULP, n = 500 점 51·가중치 1.5×10⁷ ULP.
- `np.trapezoid`와 `scipy.integrate.trapezoid`는 비트까지 같다(exp, 4,097점). 같은 점의 `simpson`은 상대 1.3e-16, `romb`는 0이다.

**속도**: §5.

### 7.5 정확도·속도 체계와 골든
- `bench/areas.py`에 영역 둘 `integrate`·`differentiate`를 더한다(C8). 골든은 모든 공개 이름의 사례(또는 `EXCLUDED`의 까닭), 1·3·10스레드 같음. 정확도 스윕은 §4의 표(정확도와 ρ), 표본 규칙의 리덕션 계약. 속도 사례는 §5의 크기.
- 테스트는 주제 이름(`test_quadpack.py`, `test_tanhsinh.py`, `test_samples.py`, `test_derivative.py` 등)이고, `test_api_surface.py`가 API 문서의 시그니처 블록을 읽는다.
- **C0에서 한 것**: `bench/areas.py`에 `integrate`·`differentiate`, 골든 `tests/golden/cases_integrate.py`(13케이스)·`cases_differentiate.py`(9케이스: 결과 클래스의 필드, 읽기 전용, 풀기, `repr`, 에러), 정확도 `bench/accuracy/integrate.py`·`differentiate.py`(스윕은 함수가 공개되면 생긴다), 속도 `bench/speed/cases/integrate.py`·`differentiate.py`(마일스톤마다의 사례 계획 26개; `DONE`과 공개 이름으로 거른다). 테스트는 `tests/integrate/`(`test_result.py`, `test_api_surface.py`, `test_problems.py`, `test_reference_data.py`)와 `tests/differentiate/`(같은 넷)다.

### 7.6 QUADPACK Fortran 원본 (C17)
사용자의 승인(2026-11-21)을 받아 netlib의 21개 파일을 받았고, 뒤에 승인을 받아 LINPACK의 `dgtsl.f`도 받았다(C18).

- 받은 곳과 둔 곳: `https://www.netlib.org/quadpack/<이름>.f`에서 `dqagse dqk21 dqelg dqpsrt dqagie dqk15i dqagpe dqawoe dqc25f dqk15w dqwgtf dqcheb dqawfe dqawse dqc25s dqmomo dqwgts dqawce dqc25c dqwgtc dqng`(.f, 모두 252 KB)와 `https://www.netlib.org/linpack/dgtsl.f`(3,240바이트)를 저장소 밖 `build/comparators/quadpack/`(git이 보지 않는다)에 `SHA256SUMS`와 함께 두었다. 원본은 저장소에 넣지 않는다. 도구가 해시를 확인하고 결과 파일에 적는다.
- 도구: `tools/gen_integrate_fortran.py SRC_DIR BUILD_DIR`(optimize의 `tools/gen_optimize_fortran.py`와 같은 꼴). 22개 파일과 대체 루틴 둘을 `gfortran -O2 -ffp-contract=off -std=legacy -shared -fPIC`(GNU Fortran, Homebrew GCC 16.2.0)로 라이브러리 하나로 빌드하고 ctypes로 부른다. 그래서 피적분함수는 scipy의 기록이 부른 것과 같은 Python 함수(`problems.MATH`)이고, 두 실행은 같은 값을 보며 점마다 기록된다. 인자는 scipy `quad`의 것이다(허용오차 1.49×10⁻⁸, `limit` 50, `maxp1` 50, `limlst` 50, 끝을 바꾸면 부호, (−∞, b]의 푸리에 가중치는 x → −x; QAGP의 중단점은 scipy처럼 정렬하고 두 칸을 더한다). 결과는 `tests/integrate/data/fortran_quadpack.json`(루틴마다의 값·오차·`nfev`·`ier`·`last`·호출 수와 `dqng` 30개)과 `fortran_vs_scipy.json`(scipy의 기록과 비트 비교, 처음 다른 점, ULP 거리)이다.
- 원본이 부르는 바깥 루틴 가운데 둘은 도구가 대신 쓴다: `d1mach`(Fortran의 `TINY`·`HUGE`·`EPSILON`; QUADPACK은 1·2·4번을 쓴다), `xerror`(`dqng.f`만 부른다: 아무것도 하지 않음). **`dgtsl`**(LINPACK의 삼중대각 풀이; `dqc25f.f`가 QAWO·QAWF에서 |ω(b − a)/2| ≤ 24인 구간의 체비쇼프 모멘트를 풀 때 두 번 부른다)은 LINPACK의 원본이다(C18).
  - 원본을 받기 전에는 LINPACK의 문서에 적힌 알고리즘(부분 피벗의 가우스 소거: 아래 대각 성분이 큰 줄을 피벗으로, 같으면 바꾼다)과 그 연산 순서로 도구 안에 쓴 대체를 썼다. 지금 그 대체(`DGTSL_STANDIN`)는 빌드에서 빠졌고, 도구가 실행마다 대체를 넣은 두 번째 라이브러리(`BUILD_DIR/standin`, 부른 수를 센다)로 모음을 다시 돌려 원본과 견준다: 116개 실행 모두 값·오차·`nfev`·`ier`·점이 비트로 같다(`dgtsl`을 부르는 다섯 실행, 24번). 결과 파일의 `dgtsl_standin`에 그 비교와 대체의 해시를 적고, 실행마다의 `dgtsl_calls`는 그 셈이다. 원본으로 바꿔도 `fortran_quadpack.json`·`fortran_vs_scipy.json`의 값은 그대로다.
  - FMA 축약을 켠 빌드(`--probe-contract`, 아래, 원본 `dgtsl`로)에서 `dgtsl`을 부르는 다섯 가운데 둘(QAWO 예제 6번, (−∞, 0]의 cos 가중치 4번)이 값·오차·점까지 scipy와 비트로 같다. scipy의 `dgtsl` 옮김도 같은 산술로 보인다.

**scipy 1.18.1과 견준 것** (`fortran_vs_scipy.json`, 116개)
- 점과 순서: 116개 가운데 112개가 같다. 다른 넷은 QAWF의 첫 주기(QK15W의 점 c + h·x가 끝자리에서 다르다; 셋)와 x·log(1/x)(α = 1) QAGS 하나(멈춤 판정이 반대로 갈려 Fortran 231번, scipy 189번; 둘 다 수렴)다. `nfev`·`ier`는 그 하나 말고 모두 같다.
- 값: 적분 62개, 오차 추정 32개가 비트로 같고, 모두 같은 것은 26개다. 다른 적분은 대개 1–26 ULP, ε-알고리즘이 외삽한 경우 114–162 ULP다.
- 까닭의 탐침(`--probe-contract`: 같은 원본을 `-ffp-contract=fast`로 빌드해 견주기만 하고 아무것도 쓰지 않는다): 모두 같은 것이 26 → 72개, 적분 62 → 100개, 오차 81개가 되지만 점이 같은 것은 112 → 90개로 준다. scipy의 C 판은 누적(합)은 FMA로 축약하고 점의 계산(c ± h·x)은 축약하지 않은 것처럼 보인다(clang의 축약은 한 식 안에서만 일어나고, scipy가 점을 따로 계산한 줄은 축약할 곱셈-덧셈이 없다). C4에서 이 자리를 scipy의 호출 순서와 결과로 찾아 `std::fma`를 적는다(§3.1, §8의 C4). 오차 추정이 축약으로도 남는 차이는 식의 꼴(`**1.5`와 `pow`의 차이 등)로 보고 C4에서 찾는다.
- QAWS의 로그 가중치 예제(`alg-loga`)는 점이 같고도 적분이 4×10⁻⁹ 다르다(Fortran −0.189275181, scipy −0.189275185, 참값 −0.189275188; 둘 다 허용오차 안). 축약으로도 남는다. 모멘트(`dqmomo`)나 수정 Clenshaw–Curtis(`dqc25s`)의 옮김에서 갈린 것으로 보고 C5에서 찾는다(§8의 C5).
- Fortran과 scipy가 수렴이라고 알린 값은 참값과 오차 추정 안이다(scipy와 같은 예외 셋 빼고: §7.7).

### 7.7 C0의 기록에서 찾은 것 (scipy 1.18.1, 2026-11-21)
- **오차 추정의 정직함**(수렴이라고 알렸는데 |값 − 참값| > 오차 추정): `quad`는 116개 가운데 수렴 104개에서 셋이다. 봉우리 α = 19·20(값 ≈ 0, 참값 ≈ π; 설계 초안과 같다)과 f10 √tan t다. f10은 외삽이 double 상한이 아니라 정확한 π/2의 적분(π√2/2, 8×10⁻¹³ 안)으로 가서 double 상한의 참값과 1.6×10⁻⁸ 다르다(참값은 double 문제 그대로다: C19). `tanhsinh`은 수렴 76개 가운데 여덟이다: 봉우리 α = 0·2·4, cos(2^α sin x)의 α = 6·7·9·10, f9(double 상한의 조각 2×10⁻¹⁵). 배치로 부른 무리 셋은 원소마다 혼자 부른 것과 값·오차·`nfev`가 비트로 같다.
- `tanhsinh`이 수렴하지 못하는 것(status −2): 봉우리 α ≥ 5, 내부 특이점(QAGP 예제, |x − 1/3|^α 셋), 끝점 특이점과 무한 구간(f12), 좁은 봉우리 1/(x² + 10⁻⁶), f7, f10, f15. `nsum`은 급수 아홉이 모두 수렴하고 오차 추정 안이다.
- `quad`가 수렴하지 못하는 것: 봉우리 α = 10–18(ier 5), cos(2^α sin x)의 α = 9(ier 1)·10(ier 5), f15 sin(t)/t의 [0, ∞)(ier 5; QAGI는 진동을 위한 것이 아니다).
- `derivative`: 같은 점을 혼자 부를 때와 배치(9점)로 부를 때 `status`·`nfev`·`n_iter`는 같지만 df가 99점 가운데 29점에서 끝자리가 다르다(최대 상대 2.8×10⁻¹⁰, 언제나 오차 추정 안). df = `fc @ wc`가 BLAS(dgemv)라 합의 순서가 배치의 크기에 달려 있고, 스텐실의 상쇄가 그 차이를 키운다. sd는 원소마다 같은 비트를 지키므로(§1.3) C2의 호출점 비교는 혼자 부른 기록과 견주고 df의 끝자리는 C5의 예외로 적는다(§3.8). 끝 근처의 기록: log의 10⁻³·√의 10⁻⁶은 기본값으로 status −3(9번), `step_direction=1`의 log 10⁻³은 status −2(오차가 줄지 않는다), x³의 0은 −2(`nfev` 27), tan의 1.5는 x + 0.5가 π/2를 넘는데도 7번 반복 뒤 수렴(값은 맞다).

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 |
|---|---|
| C0 준비 (완료·승인(위임) 2026-11-21) | API 문서의 확정, 결과 객체의 바탕(`OptimizeResult`에서 끌어내기, 동작 불변), 모듈 `sd.integrate`·`sd.differentiate`(공개 이름은 결과 클래스뿐), 시험 모음(§7.1), Arb 참값 도구(§7.2), scipy 1.18.1 기록과 호출 순서(§7.3), QUADPACK 원본 대조(C17, §7.6), `bench/areas.py`와 측정·골든의 등록, scipy 시간(바쁜 기계라 점검 수준, §5) |
| C1 함께 쓰는 부품 | C3·C15: `diff.hpp`와 `Differences`를 differentiate로 옮기기. C4·C16: `batch.cpp`에서 반복 고리를 csrc 아래 `elementwise/`로 끌어내 넓히기. 공개 함수 없음. optimize·regression의 테스트·골든·배치 속도가 그대로 |
| C2 미분 | `derivative`, `jacobian`, `hessian` |
| C3 표본과 고정 규칙 | `trapezoid simpson romb cumulative_trapezoid cumulative_simpson fixed_quad newton_cotes` |
| C4 QUADPACK 1 | `quad`(QAGS, QAGI, QAGP), `dblquad`, `tplquad`, `nquad`. scipy의 C 판이 FMA로 축약한 누적의 자리를 호출 순서와 결과로 찾아 `std::fma`를 적는다(§7.6의 축약 탐침). `summary()`(C20) |
| C5 QUADPACK 2 | `quad`의 `weight`(QAWO, QAWF, QAWS, QAWC). QAWS 로그 가중치 예제의 scipy·Fortran 4×10⁻⁹ 차이(§7.6)의 까닭을 찾는다. QAWO·QAWF의 Fortran 대조는 LINPACK 원본 `dgtsl`로 돈 것이다(C18) |
| C6 배치 적분 | `tanhsinh`, `nsum` |
| C7 벡터 값과 다차원 | `quad_vec`, `cubature` |
| C8 최적화와 일관성 점검 | 모든 공개 함수를 크기마다 scipy와 견주어 느린 곳을 고치고, 메시지·시그니처의 영역 간 일관성을 보고, 코드 리뷰를 한다 |

- 순서의 까닭: C1의 고리를 처음 쓰는 C2가 고리의 넓힘(원소마다 여러 점)을 바로 시험한다. C3은 다른 것에 기대지 않는다. C6의 `nsum`은 C6의 `tanhsinh`을, C7의 무한 축 치환은 C4·C6과 같은 식을 쓴다.
- **나중의 ODE 묶음**(C2, 이름은 그때): `solve_ivp`(RK23, RK45, DOP853, Radau, BDF, LSODA; 조밀 출력, 사건), `odeint`(LSODA), `solve_bvp`. 원소별 무리가 아니라 상태가 있는 풀이라 이 묶음의 고리를 쓰지 않는다. 큰 결정은 그때 묻는다.
- 진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.

---

## 9. 마일스톤을 시작하기 전에 물을 것
추천을 첫째에 둔다.

- **C0**: 모두 정했다(C8–C17; C0을 마치며 생긴 넷은 추천안으로 C18–C21, 사용자 위임 2026-11-21). `dgtsl`의 질문도 닫혔다: 원본을 받았다(C18, 사용자 승인 2026-11-21).
- **C1**: 끌어낸 고리에서 optimize 배치의 속도가 1% 넘게 느려지면 되돌릴지(추천: 되돌리고 그 자리만 템플릿으로); `steps.hpp`의 이름과 이름공간.
- **C2**
  1. 원소마다 여러 점을 넘기는 꼴: (a) (k, p) 2차원(scipy의 꼴, `args`는 (k, 1)) / (b) 1차원 (k·p,)으로 펴고 `args`도 p번 되풀이(O38의 1차원 계약을 그대로).
  2. `jacobian`·`hessian`의 f: (a) (m, N)을 받아 (k, N)을 돌려준다(scipy 계약의 2차원 판, x는 1차원 (m,)만) / (b) scipy처럼 (m, …)의 모든 꼴.
  3. 가중치: (a) 올바로 반올림(중앙은 정확한 유리수, 한쪽은 Arb; 점은 scipy와 같다, df는 끝자리가 다르다) / (b) scipy처럼 Vandermonde를 LAPACK으로 풀기.
  4. `hessian`의 rtol < 100ε: (a) `ValueError` / (b) scipy처럼 말없이 100ε로.
- **C3**
  1. 입력: (a) 데이터 함수의 규칙(API.md §5: `sd.ndarray`, 시퀀스, numpy 배열, `Series`) / (b) 배열 함수처럼 `sd.ndarray`만.
  2. `x`의 꼴: (a) `trapezoid(y, /, x=None, *, dx=1.0, axis=-1)`처럼 위치와 키워드 모두(scipy의 `simpson(y, x=x)`가 된다; `cumulative_simpson`도 같게: `[scipy와 다름]`, scipy는 키워드만) / (b) 함수마다 scipy 그대로.
  3. `axis`의 기본: (a) scipy·numpy의 −1 / (b) sd `cumulative_sum`처럼 2차원부터는 꼭 준다.
  4. Gauss–Legendre: (a) 올바로 반올림한 점·가중치(n ≤ 100은 Arb로 만든 표, 그 위는 정확한 알고리즘; n ≥ 20에서 호출점이 scipy와 다를 수 있다: C5의 예외), 공개 이름은 없음 / (b) 같은 것을 `sd.special.roots_legendre`로도 공개 / (c) scipy처럼 Golub–Welsch.
  5. `fixed_quad`의 반환: (a) 값 하나(scipy의 `(val, None)` 튜플이 아니다) / (b) scipy 그대로. `newton_cotes`는 scipy처럼 `(an, B)` 튜플.
- **C4**
  1. `quad` 결과의 구간 정보(scipy `infodict`의 `alist blist rlist elist iord last`): (a) 필드 `intervals`(last, 2), `interval_integrals`, `interval_errors`(QUADPACK의 `iord` 순서), `n_intervals` / (b) 두지 않는다.
  2. 복소수 피적분함수: (a) `quad(complex_func=True)`만(scipy처럼 실수부·허수부의 두 `quad`), 나머지는 실수만 / (b) 모두 실수만 / (c) 모두 복소수도.
  3. 점 하나씩이 아니라 규칙의 점 21개를 배열 하나로 부르는 선택(scipy에 없음): (a) 두지 않는다 / (b) `vectorized=True`를 더한다.
  4. scipy 글의 줄바꿈(`"\n  "`): (a) 그대로(O86) / (b) 한 줄로.
- **C5**: scipy의 `wopts`(미리 계산한 체비쇼프 모멘트): (a) 두지 않는다 / (b) 둔다.
- **C6**
  1. tanh-sinh 표: (a) scipy와 같은 libm 식(호출점 비트 동일) / (b) Arb로 만든 정확한 표(최대 980 ULP를 고침; 호출점이 끝자리에서 다르다).
  2. `log=True`와 음의 피적분함수(복소 log): C4의 복소수 답을 따른다.
  3. 끝점의 호출: (a) scipy처럼 부르고 버린다(계약: f는 예외 대신 inf·NaN) / (b) 끝점에 닿은 점은 부르지 않는다(호출점이 scipy와 달라진다).
- **C7**: `cubature`의 `status`(scipy는 문자열 `'converged'`·`'not_converged'`): (a) `converged`(bool)와 정수 `status`(0, 1) / (b) scipy의 문자열; `regions`(영역 객체의 목록)를 둘지; 사용자 규칙 객체와 `workers`는 없음.
- **C8**: 정할 것은 그때.
