# 수치 미적분 영역 설계 (sd.integrate, sd.differentiate)

> 상태: **설계 초안** (2026-11-21). 큰 결정(§1.1 C1–C7)과 C0 전의 결정(C8–C17)은 사용자가 정했다. 나머지는 안이다. C0은 완료·승인(위임, 2026-11-21)이다. C1은 완료·승인(위임, 2026-11-21)이다. C2는 완료·승인(위임, 2026-11-21)이다(결정 C26–C38은 추천안, 사용자 위임). C3은 완료·승인(위임, 2026-11-22)이다(결정 C39–C54는 추천안, 사용자 위임). C4는 완료·승인(위임, 2026-11-22)이다(결정 C55–C70은 추천안, 사용자 위임). C5는 완료·승인(위임, 2026-11-22)이다(결정 C71–C83은 추천안, 사용자 위임). C6은 완료·승인(위임, 2026-11-22)이다(결정 C84–C93은 추천안, 사용자 위임). C7은 완료·승인(위임, 2026-11-22)이다(결정 C94–C107은 추천안, 사용자 위임). C8은 완료·승인(위임, 2026-11-22)이다(결정 C108–C116은 추천안, 사용자 위임). 첫 판의 마일스톤은 모두 끝났다. 코드 리뷰의 고침은 C117–C123이다(C123은 사용자 결정, 나머지는 추천안·사용자 위임, 2026-11-22).
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
| C3 차분 커널 옮기기 | 커밋 1bc8c2b의 `csrc/optimize/diff.hpp`에 있던 고정 간격 차분의 커널(`approx_fprime`, `check_grad`, `jac='2-point'`·`'3-point'`가 쓰는 scipy `approx_derivative`의 간격, O18–O31)을 differentiate 영역의 C++로 옮기고 optimize가 그것을 부른다(H17처럼: 함께 쓰는 커널은 그 본질에 가장 가까운 영역에 둔다). `approx_fprime`·`check_grad`는 scipy의 이름대로 `sd.optimize`에 공개된 채로 둔다. 값은 비트까지 같고 optimize의 골든은 바뀌지 않는다. 지금은 계획만 하고 코드는 옮기지 않는다 (사용자 결정 2026-11-21) |
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
| C22 고리의 속도 (C1 전) | 끌어낸 고리에서 optimize 배치의 속도가 1% 넘게 느려지면 그 부분을 되돌리고 그 자리만 템플릿으로 둔다. 잰 결과 느려지지 않아(§6.2) 되돌린 곳이 없다 (추천안, 사용자 위임 2026-11-21) |
| C23 이름과 자리 (C1 전) | 차분 간격은 `csrc/differentiate/steps.hpp`(이름공간 `sd::differentiate`, 함수 이름은 그대로), 반복 고리는 `csrc/elementwise/loop.hpp`(이름공간 `sd::elementwise`; scipy `eim._loop`의 자리), Python의 5점 차분은 `src/stardust/differentiate/_differences.py`(`Differences`)다. 모두 헤더·내부 모듈이라 CMake와 공개 이름은 그대로다 (추천안, 사용자 위임 2026-11-21) |
| C24 여러 점의 호출 꼴 (C1) | 고리의 `Call`은 원소마다 점 하나((k,), optimize)와 p개((k, p) C 순서, `args`는 (k, 1), 값 검사는 `check(value, (k, p))`)를 둘 다 받고, p는 호출마다 바뀔 수 있다. 점이 모자란 원소는 부르는 쪽이 NaN으로 채운다(scipy와 같다). 1차원으로 편 꼴(§9 C2.1 (b))도 같은 `Call`로 된다(점 k·p개, 원소의 번호를 p번 되풀이한 `idx`). 어느 꼴을 공개 계약으로 할지는 §9 C2.1에 남긴다 (추천안, 사용자 위임 2026-11-21) |
| C25 R175·R176 (C1) | R175: `_differences.py`의 `_ROUNDING_TOL = 2⁻²⁶`과 `_forward`의 간격을 √u라 부른 주석을 √ε(= 2⁻²⁶, ε = 2u)로 고쳤다(값은 그대로). R176: `approx_fprime([1e-12], exp)`가 f를 61번 부르는 것은 간격 키우기(O104)가 4배씩 오르기 때문이다. 고치면 고른 간격과 값이 바뀌어 `approx_fprime`·nls의 골든이 바뀌므로 고치지 않고 잰 호출 수를 API 문서(docs/api/optimize.md §2)와 테스트에 적었다(§6.1) (추천안, 사용자 위임 2026-11-21) |
| C26 여러 점의 꼴 (C2) | `derivative`의 둘째 호출부터 f는 원소마다 p개 점의 (k, p) 2차원 float64 `sd.ndarray`를 받는다(scipy의 꼴; 원소의 순서는 입력의 C 순서, 아직 끝나지 않은 원소만, O38). `args`의 값은 (k, 1)이다. 첫 호출은 (k,)다(O48). 까닭: scipy의 사용자 함수가 그대로 돌고(`f(x)[i]`가 원소 i의 점들), 1차원으로 펴면 `args`를 p번 되풀이해야 한다 (추천안, 사용자 위임 2026-11-21) |
| C27 `jacobian`·`hessian`의 f (C2) | f는 (m, N)을 받아 (k, N)(스칼라 함수는 (N,))을 돌려준다. scipy 계약의 2차원 판이고 x는 1차원 (m,)만이다. 넘기는 열은 scipy의 `xph`를 C 순서로 편 것이라 바이트가 scipy와 같다(`jacobian`: 열 j·p + q는 변수 j의 q번째 점; `hessian`의 안쪽: 열 (j·B + b)·p + q) (추천안, 사용자 위임 2026-11-21) |
| C28 가중치 (C2) | 가중치는 정확한 값을 한 번 올바로 반올림한 것이다(중앙은 정확한 유리수, 한쪽은 Q(√c)의 정확한 값; C34). 점은 scipy의 식 그대로라 비트로 같고, df는 scipy의 LAPACK 가중치(order 8에서 정확한 값과 2.7×10⁻¹⁴까지 다르다)와 끝자리가 다를 수 있다. 그래서 멈춤 판정의 경계에서 반복 수가 갈리는 경우가 생긴다(§7.8에 잰 비율) (추천안, 사용자 위임 2026-11-21) |
| C29 `hessian`의 rtol (C2) | 0 < rtol < 100ε이면 `ValueError`다(scipy는 경고하고 100ε로 올린다; 안쪽 `jacobian`의 rtol/100이 ε 아래가 된다). scipy의 조건 그대로 rtol = 0은 받는다(atol만의 판정) (추천안, 사용자 위임 2026-11-21) |
| C30 `initial_step`의 꼴 (C2) | `jacobian`·`hessian`의 `initial_step`(과 `jacobian`의 `step_direction`)은 수 하나 또는 변수마다 m개다. `hessian`에서는 바깥과 안쪽 미분 모두 변수 j에 j번째 값을 쓴다. scipy의 `hessian`은 배열을 주면 안쪽 호출의 브로드캐스트가 틀어져 에러다(`[scipy와 다름]`). 까닭: `jacobian`과 같은 꼴, 같은 뜻 (추천안, 사용자 위임 2026-11-21) |
| C31 `step_factor`의 범위 (C2) | 유한하고 0보다 크고 1이 아닌 수다. 아니면 `ValueError`다(scipy: 1과 inf는 Vandermonde가 특이해 `LinAlgError`, 0은 모든 원소가 NaN·status −3). 1보다 작은 값은 받는다(간격이 커진다; scipy와 같다) (추천안, 사용자 위임 2026-11-21) |
| C32 NaN `step_direction` (C2) | 그 원소의 점은 NaN이고 첫 반복 뒤 status −3, df NaN이다. scipy는 그 원소를 어느 쪽에도 넣지 않아 우연히 x = 0을 부르고 같은 결과(NaN, −3)를 낸다. 결과는 같고 호출점만 다르다. `initial_step` ≤ 0(scipy: NaN 간격)과 같은 꼴이다 (추천안, 사용자 위임 2026-11-21) |
| C33 `jacobian`의 호출 범위 (C2) | 모든 항목이 멈출 때까지 반복마다 모든 열(변수)을 부른다(scipy의 `preserve_shape=True`와 같다). O38의 "남은 원소만"은 원소별 f(`derivative`)의 규칙이고, `jacobian`의 f는 원소별이 아니다(한 열이 k개 출력을 함께 낸다). 까닭: 호출점이 scipy와 같고(C5), f의 계약이 늘 (m, N)이다. 멈춘 항목의 결과는 멈춘 때의 것이다 (추천안, 사용자 위임 2026-11-21) |
| C34 한쪽 가중치의 계산 (C2) | python-flint(Arb)는 라이브러리의 의존이 아니라서(테스트 의존) 한쪽 가중치는 실행 중에 Q(√c)에서 정확히(a + b√c, a·b 유리수) 풀고, √c를 정수로 감싼 구간을 정밀도를 올리며 좁혀 양 끝이 같게 반올림될 때 그 double을 쓴다. c가 유리수의 제곱이면 유리수로 계산한다. 테스트가 모든 가중치를 Arb의 풀이와 견준다(`tests/differentiate/test_weights.py`). (n, c)마다 기억해 둔다 (추천안, 사용자 위임 2026-11-21) |
| C35 틀린 값의 글 (C2) | `derivative`는 optimize 배치의 꼴이다: `derivative(): 'f' must return shape (2, 8) like its argument, not (2, 1)`(첫 호출은 `(k,)`, 스칼라는 `not a scalar`). `jacobian`·`hessian`은 받은 모양을 함께 적는다: `jacobian(): 'f' must return shape (2,) or (k, 2) for its argument of shape (2, 2), not (3, 3)`, 다음 호출부터 첫 호출이 정한 꼴(`(k, N)` 또는 `(N,)`; `hessian`은 처음부터 `(N,)`). 까닭: 영역 간 일관성 (추천안, 사용자 위임 2026-11-21) |
| C36 결과의 모양과 `summary()` (C2) | `jacobian`·`hessian`의 결과는 항목마다 따로 멈춘 배치다: 배치 모양은 df·ddf의 모양((k, m), 스칼라 f는 (m,); (m, m))이고 `repr`는 `DifferentiateResult(batch shape (3, 3), 7 of 9 converged)`, `summary()`는 항목마다 한 줄(`[i, j]`)이다. `derivative`의 배치는 입력의 브로드캐스트 모양이고 스칼라면 Python 수다(O48). `summary()`의 제목은 `Differentiation Results`, 머리는 하나면 `Converged`·`Status`·`Point`와 `Derivative`(#.6g)·`Error estimate`(#.4g)·`No. Iterations`·`Function Evals`, 배치면 `Batch shape`·`Converged: j of n`과 열 `x df|ddf error converged status n_iter nfev`(있는 것만)다(C20) (추천안, 사용자 위임 2026-11-21) |
| C37 `jacobian`·`hessian`의 x (C2) | `derivative`의 x와 같은 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열)이고 1차원·비지 않음만 검사한다. 유한하지 않은 값은 받는다(그 항목이 status −3; scipy와 같다). optimize의 `x0`(유한해야 함, `Series`도 받음)의 규칙은 쓰지 않는다: 이 무리의 x는 계산할 점이지 시작점이 아니다 (추천안, 사용자 위임 2026-11-21) |
| C38 중앙 스텐실의 합 (C2) | 중앙 스텐실의 df는 마디 −t, t의 짝마다 w·(f(x − ht) − f(x + ht))를 바깥 짝부터 더한다(가중치가 정확히 반대칭이라 scipy의 `fc @ wc`와 정확한 산술에서 같은 값). 가까운 두 값의 차는 정확하므로(Sterbenz) f(x)가 f′(x)h보다 훨씬 클 때의 상쇄가 없다: 모음의 최대 상대오차가 차례대로의 합 3.2×10⁻¹⁰에서 6.2×10⁻¹¹로 줄고, scipy와 같은 반복에 멈추는 경우가 늘었다(`jacobian` 호출이 같은 것 39 → 43/47). D-33의 셋째 규칙(더 정확한 계산 순서)이다. 한쪽 스텐실은 차례대로 더한다 (추천안, 사용자 위임 2026-11-21) |
| C39 표본의 입력 (C3) | 데이터 함수의 규칙(API.md §5)이다: `sd.ndarray`, 수의 시퀀스, numpy 배열, frame의 `Series`(결측값은 NaN). `DataFrame`은 받지 않는다. 까닭: 표본 규칙은 밖에서 가져온 자료를 처음 넣는 자리다 (추천안, 사용자 위임 2026-11-21) |
| C40 `x`의 꼴 (C3) | `trapezoid(y, /, x=None, *, dx=1.0, axis=-1)`처럼 `x`는 위치와 키워드 모두다(`simpson`, `cumulative_trapezoid`, `cumulative_simpson`도). scipy의 `cumulative_simpson`·`simpson`은 `x`가 키워드뿐이다(`[scipy와 다름]`; scipy 코드의 `simpson(y, x=x)`는 그대로 된다) (추천안, 사용자 위임 2026-11-21) |
| C41 `axis`의 기본 (C3) | scipy·numpy처럼 −1이다. sd `cumulative_sum`의 "2차원부터는 꼭 준다"는 쓰지 않는다: 적분은 언제나 한 축을 따라가고 scipy 코드가 그대로 돌아야 한다 (추천안, 사용자 위임 2026-11-21) |
| C42 Gauss–Legendre (C3) | 점과 가중치는 올바로 반올림한 double이다. n ≤ 100은 Arb로 만든 표(`tools/gen_gauss_legendre.py` → `csrc/integrate/gauss_table.hpp`), 그 위는 실행 중의 정확한 알고리즘(C54)이다. 공개 이름은 없다. n ≥ 20에서 호출점이 scipy와 다를 수 있다(C5의 예외, §7.9) (추천안, 사용자 위임 2026-11-21) |
| C43 `fixed_quad`·`newton_cotes`의 반환 (C3) | `fixed_quad`는 값 하나다(scipy의 `(val, None)` 튜플이 아니다, `[scipy와 다름]`). `newton_cotes`는 scipy처럼 `(an, B)` 튜플이다 (추천안, 사용자 위임 2026-11-21) |
| C44 `romb` (C3) | `dx`는 다른 표본 규칙과 같은 꼴(C45)이고 `show`는 없다(§1.3: 화면에 쓰지 않는다). 점은 `axis`를 따라 2ᵏ + 1개(k ≥ 0)여야 하고 아니면 `ValueError`다(점 하나도: scipy와 같다) (추천안, 사용자 위임 2026-11-21) |
| C45 `x`·`dx`의 꼴 (C3) | `x`는 실수이고(int·bool은 바꿔서, complex는 `TypeError`) `axis`를 따라 y의 길이인 1차원이거나, y와 차원 수가 같고 y의 모양으로 브로드캐스트되며 `axis`의 길이가 y와 같은 배열이다. scipy는 함수마다 다르다(`trapezoid`는 더 낮은 차원의 브로드캐스트도, `cumulative_simpson`은 y와 같은 모양만): 다섯 함수에 한 규칙이다. `dx`는 수(유한하지 않아도 받는다) 또는 y의 모양에서 `axis`만 1인 배열(줄마다의 간격; scipy `cumulative_simpson`의 꼴)이고 다섯 함수가 같다. `x`와 함께 1.0이 아닌 `dx`를 주면 `ValueError`다(O29·O62: 쓰이지 않는 인자; scipy는 무시한다). 까닭: 영역 안의 일관성 (추천안, 사용자 위임 2026-11-21) |
| C46 누적의 `initial` (C3) | scipy 그대로: `cumulative_trapezoid`는 None 또는 0(맨 앞에 넣기만 한다; 다른 수는 `ValueError`), `cumulative_simpson`은 수 또는 줄마다의 배열(C45의 `dx`와 같은 모양)이고 모든 누적값에 더한 뒤 맨 앞에 넣는다. 실수 y에 복소수 `initial`은 `TypeError`다(scipy도 에러) (추천안, 사용자 위임 2026-11-21) |
| C47 짝수 개 점의 `simpson` (C3) | scipy 1.11부터의 꼴이다: 앞 구간들은 Simpson, 마지막 구간은 Cartwright의 보정(α·β·η), 점 둘이면 사다리꼴. `cumulative_simpson`은 scipy의 구간마다의 Cartwright 식(왼쪽·오른쪽)이고 점이 셋보다 적으면 사다리꼴이다 (추천안, 사용자 위임 2026-11-21) |
| C48 dtype (C3) | y는 float64 또는 complex128로 계산한다(bool·int64는 float64로, string은 `TypeError`). 결과는 y의 dtype이고 0차원이면 Python 수다. 복소수 y는 실수 계수를 실수부·허수부에 따로 곱한다(numpy는 계수를 복소수로 올려 곱한다: 한쪽이 inf·NaN일 때만 다르다) (추천안, 사용자 위임 2026-11-21) |
| C49 빈 입력과 짧은 입력 (C3) | `trapezoid`·`simpson`은 점이 하나 이하면 0이다(scipy의 `simpson`은 점 0개에 `IndexError`, `[scipy와 다름]`: 빈 합은 0이다, 배열의 규칙). 누적은 점 0개가 `ValueError`(scipy와 같다), 점 하나면 빈 결과(`initial`이 있으면 그 값 하나)다. 다른 축이 비면 빈 배열이다. 스칼라 y는 `ValueError`다 (추천안, 사용자 위임 2026-11-21) |
| C50 NaN과 inf (C3) | 값은 검사하지 않고 퍼진다(배열의 원소별 규칙). `cumulative_simpson`의 "x는 순증가"는 scipy처럼 x가 있고 점이 셋 이상일 때 `diff(x) <= 0`이 하나라도 있으면 `ValueError`다(NaN은 걸리지 않는다). `simpson`의 0 간격은 scipy의 `apply_where`처럼 그 몫을 0으로 둔다 (추천안, 사용자 위임 2026-11-21) |
| C51 표본 규칙의 합 (C3) | 줄(축을 따라간 한 줄)마다 항을 sd의 pairwise 합(누산기 16개, 512개 단위; `sd.sum`이 연속 축에서 하는 것)으로 첨자 순서대로 더하고, 누적은 `cumulative_sum`의 스캔이다. 그래서 줄의 값은 그 줄의 수에만 달려 있다: 축, 메모리 배치, 스레드 수와 상관없다(`sd.sum`의 바깥 축 리덕션은 차례대로 더한다) (추천안, 사용자 위임 2026-11-21) |
| C52 `fixed_quad`의 함수 (C3) | f는 (n,) float64 `sd.ndarray`를 한 번 받고 (n,) 또는 (…, n)을 돌려준다(scipy의 벡터 값; 결과는 (…) 배열). 값은 실수다(복소수·bool은 `TypeError`, §1.2). a, b는 유한한 실수다(NaN도 `ValueError`; scipy는 inf만 막고 NaN은 NaN을 낸다). 점은 scipy의 식 (b − a)·(x + 1)/2.0 + a, 합은 C51이다 (추천안, 사용자 위임 2026-11-21) |
| C53 `newton_cotes` (C3) | 모든 입력에서 가중치와 오차 계수를 정확한 유리수로 계산해 한 번 반올림한다(라그랑주 기저의 정확한 적분; scipy는 같은 간격 N ≤ 14만 정확한 표이고 그 밖은 Vandermonde 역행렬). N ≤ 14의 같은 간격은 scipy와 비트로 같고 그 밖은 다를 수 있다(C5의 예외, §7.9). `rn`은 1 이상의 int 또는 2개 이상의 유한하고 서로 다른 위치(0에서 시작해 N에서 끝남, 언제나 검사; scipy는 표 밖에서만 본다), `equal=True`면 길이만 본다(scipy와 같다). 같은 입력은 기억한다 (추천안, 사용자 위임 2026-11-21) |
| C54 표 밖의 Gauss–Legendre (C3) | n > 100: C++가 점마다 double Newton 뒤 double-double Newton 두 번과 가중치를 double-double(약 106비트)로 계산하고(점마다 따로, 스레드와 상관없음), Python이 반올림한다. double-double의 오차 한계(점 n²·2⁻⁵⁷ ulp, 가중치 n²·2⁻¹⁰⁰ 상대; n = 101–3000에서 Arb와 견준 실제 오차의 2¹⁰배 이상) 안에 반올림 경계가 있으면 정수의 정확한 산술로 정한다(점: 두 반올림 경계에서 P_n의 부호, 가중치: 근을 감싼 구간과 립시츠 한계). 비용은 O(n²)이다(10⁴점 0.7초; C8부터 n ≤ 10⁴, C109); 최근 8개의 n을 기억한다 (추천안, 사용자 위임 2026-11-21) |
| C55 구간 정보 (C4) | `quad`(가중치 없는 길)의 결과에 필드 넷을 둔다: `intervals`((last, 2), 구간의 끝), `interval_integrals`, `interval_errors`(구간마다의 값과 오차 추정), `n_intervals`(scipy `last`). 순서는 QUADPACK의 `iord`다: 오차 추정이 큰 것부터 QUADPACK이 순위를 매긴 kmax개(`last` ≤ `limit`/2 + 2면 모두, 아니면 `limit` + 1 − `last`개; scipy 문서의 kmax), 그 뒤 나머지(더 나누지 않을 작은 것들)를 오차 추정의 내림차순(같으면 만든 순서)으로 둔다. QAGI의 구간은 scipy처럼 바꾼 변수 t ∈ (0, 1]의 것이다. 세 배열은 C++가 만든 읽기 전용 배열이다. 겹친 적분(`dblquad tplquad nquad`)은 scipy `nquad`처럼 두지 않는다. 까닭: scipy `infodict`의 값을 필드로(C10), 순서는 QUADPACK의 것 (추천안, 사용자 위임 2026-11-21) |
| C56 복소수 피적분함수 (C4) | `quad(complex_func=True)`만 받는다: scipy처럼 실수부의 `quad`를 마친 뒤 허수부의 `quad`를 한다(f를 점마다 두 번, 같은 점들을 두 번 부른다). 다른 함수와 `complex_func=False`의 값은 실수만이다(복소수는 `TypeError`). 실수 값도 받는다(허수부 0). 까닭: scipy의 범위 그대로이고 실수 QUADPACK의 비트를 그대로 쓴다 (추천안, 사용자 위임 2026-11-21) |
| C57 복소수 결과의 필드 (C4) | `integral`과 `error`는 complex(scipy의 re + 1j·im, 오차도), `converged`는 둘 다, `status`·`message`는 실수부의 것이 0이 아니면 그것, 아니면 허수부의 것, `nfev`는 합이다. 두 실수 `quad`의 결과는 새 필드 `real`·`imag`(`IntegrateResult`, 구간 정보와 함께; scipy `full_output`의 `{"real": …, "imag": …}`)이고 맨 위에는 구간 필드가 없다. `repr`·`summary()`는 복소수를 (re+imj)로 적는다. 까닭: 풀기(C10)가 scipy의 `(integral, error)`와 같고, 부분의 정보는 잃지 않는다 (추천안, 사용자 위임 2026-11-21) |
| C58 규칙 한 번의 배열 호출 (C4) | `vectorized=True`(QK21의 21점을 배열 하나로 부르기)는 두지 않는다. 까닭: scipy에 없고, 점마다 부르는 것이 scipy의 호출점과 같다 (추천안, 사용자 위임 2026-11-21) |
| C59 글의 줄바꿈 (C4) | scipy 글의 `"\n  "`은 그대로 둔다(O86). `summary()`는 공백을 하나로 줄여 한 줄의 글로 보인다 (추천안, 사용자 위임 2026-11-21) |
| C60 수렴의 글 (C4) | status 0의 `message`는 QUADPACK 문서의 `ier = 0` 설명 `Normal and reliable termination of the routine. It is assumed that the requested accuracy has been achieved.`다(scipy는 0에 글이 없다). 까닭: 다른 영역처럼 `message`가 늘 있고, 다른 번호의 글(scipy가 QUADPACK 문서에서 옮긴 것)과 같은 출처다 (추천안, 사용자 위임 2026-11-21) |
| C61 C4의 `weight` (C4) | 시그니처는 API 문서 그대로(`weight wvar max_p1 limlst`)이고, C5 전에는 알맞은 이름의 `weight`가 `NotImplementedError`(`quad(): 'weight' (QAWO, QAWF, QAWS, QAWC) comes with milestone C5`), 모르는 이름은 `ValueError`, `weight` 없는 `wvar`는 `ValueError`, 기본값이 아닌 `max_p1`·`limlst`도 쓰이지 않는 인자라 `ValueError`다(O29·O62; 형과 하한의 검사가 먼저). `nquad`의 `opts`도 같다. C5에서 이 갈래를 고친다 (추천안, 사용자 위임 2026-11-21) |
| C62 FMA의 자리 (C4) | scipy의 C 판이 축약한 곳은 Kronrod 규칙(QK21, QK15I)의 누적 넷 resg, resk, resabs, resasc의 s = s + w·v뿐이고 `std::fma(w, v, s)`로 적었다(clang의 `-ffp-contract=on`은 한 식 안의 곱과 합만 합친다; 점 c ∓ h·x는 원본에서 두 문장이라 합치지 않는다). 그래서 가중치 없는 106개 모두 scipy와 점·값·오차·`nfev`·status·글·구간이 비트로 같다(§7.10) (추천안, 사용자 위임 2026-11-21) |
| C63 겹친 적분의 오차·`nfev`·status (C4) | `error`는 scipy `_NQuad`의 `self.abserr`처럼 모든 층의 모든 `quad`가 낸 오차 추정의 최댓값이다(설계 초안의 "바깥 적분의 추정뿐"은 틀렸다; Python `max`처럼 NaN은 건너뛴다). `nfev`는 가장 안쪽 `quad`들의 `nfev` 합(scipy `neval`, f를 부른 수)이다. `converged`는 모든 `quad`가 수렴했을 때이고, `status`·`message`는 끝나는 순서로(안쪽이 먼저) 처음 수렴하지 못한 `quad`의 것이다: scipy가 처음 내는 경고이고 안쪽의 실패가 바깥 실패의 까닭인 경우가 많다 (추천안, 사용자 위임 2026-11-21) |
| C64 겹침의 자리 (C4) | 모든 층을 C++가 겹쳐 돌린다(`nquad_solve`): 층마다 범위와 옵션을 바깥 점마다 계산하고(사용자 함수면 불러서 Python이 검사), 가장 안쪽만 f(x₀, x₁, …)를 부른다. 범위 함수는 scipy처럼 바깥 변수를 가까운 것부터 받고(`ranges[i](x_{i+1}, …)`), 옵션 함수도 같다. `dblquad`·`tplquad`의 범위는 scipy의 `temp_ranges`·`ranges0`·`ranges1`과 같은 순서로 `gfun`·`hfun`(`qfun(x, y)`·`rfun(x, y)`)을 부르고, 모두 수면 함수를 부르지 않는다. 호출점은 scipy와 같다(§7.10) (추천안, 사용자 위임 2026-11-21) |
| C65 `quad`의 인자 (C4) | `a`·`b`는 실수(±inf 받음, NaN은 `ValueError`). `points`는 None 또는 실수의 시퀀스(`sd.ndarray`·numpy 배열도; 수 하나는 `TypeError`, scipy는 받는다; NaN은 `ValueError`)이고 scipy처럼 정렬·중복 제거 뒤 (a, b) 안의 것만 쓴다(빈 목록도 QAGP). 무한 구간과 `points`는 `ValueError`, 남은 점의 수 ≥ `limit`도 `ValueError`다. `limit`은 1 ≤ `limit` ≤ 2³¹ − 1의 int(QUADPACK의 C int; 구간 배열은 `last`에 따라 자라므로 큰 `limit`이 메모리를 먼저 잡지 않는다). `atol`·`rtol`은 유한한 0 이상의 수(scipy는 음수 `epsabs`를 받는다)이고 `atol` = 0이면 `rtol` ≥ max(50ε, 5×10⁻²⁹)이어야 한다(QUADPACK의 `ier = 6`; 글은 이 이름으로). 옵션은 a == b여도 검사한다(scipy는 그 전에 0을 돌려준다). f의 값은 optimize의 스칼라 규칙(Python·numpy의 실수, 0차원 배열; bool·복소수는 `TypeError`, 모양이 있으면 `ValueError`) (추천안, 사용자 위임 2026-11-21) |
| C66 `nquad`·`dblquad`·`tplquad`의 인자 (C4) | `ranges`는 비지 않은 시퀀스이고 항목은 (lo, hi) 쌍 또는 쌍을 돌려주는 함수다. `opts`는 None, dict 하나(모든 층), 또는 `ranges`와 길이가 같은 dict·함수의 시퀀스다. 키는 `quad`의 이름 `atol rtol limit points weight wvar`이고 모르는 키는 `ValueError`(scipy의 `epsabs`·`epsrel`에는 새 이름을 알린다; optimize O16). 범위의 끝과 `gfun`…`rfun`의 값은 실수(NaN은 `ValueError`, 글에 그 이름) (추천안, 사용자 위임 2026-11-21) |
| C67 QNG (C4) | scipy의 `quad`는 `dqng`를 부르지 않으므로(§1.2) 옮기지 않았다. `fortran_quadpack.json`의 `dqng` 30개는 견줄 짝이 없어 그대로 둔다 (추천안, 사용자 위임 2026-11-21) |
| C68 파일과 층 (C4) | QUADPACK의 루틴은 `csrc/integrate/quadpack.hpp`(템플릿 헤더: 피적분함수는 아무 호출 가능한 것이라 겹친 적분의 안쪽 `quad`와 Python 콜백이 같은 커널을 쓴다; 배열은 원본처럼 1부터), scipy의 갈래·구간 배열·겹침은 `quadpack.cpp`, 인자 검사·결과는 `src/stardust/integrate/_quadpack.py`다. 결과는 빠른 생성자(`IntegrateResult._quad`)로 만든다(필드 검사를 건너뛴다; 값은 함수가 만든 것뿐) (추천안, 사용자 위임 2026-11-21) |
| C69 `summary()` (C4) | 제목 `Integration Results`, 머리 왼쪽 `Converged`·`Status`, 오른쪽 `Integral`(#.6g)·`Error estimate`(#.4g)·`Function Evals`·`Intervals`(있으면), 넓은 줄 `Message`, 표는 구간(열 `a b integral error`, 줄 `[k]`, 20개를 넘으면 앞뒤 5개)이다. 복소수 결과는 `real`·`imag` 두 줄의 표다. 배치(C6의 `tanhsinh`·`nsum`)는 `DifferentiateResult`처럼 줄마다 한 원소다(C20) (추천안, 사용자 위임 2026-11-21) |
| C70 C4의 속도 (C4) | 함숫값 21개의 가장 작은 경우에서 호출 사이의 일이 scipy와 비슷하도록 인자 검사의 빠른 길(기본값, float 끝)과 결과의 빠른 생성자를 두었다. 남은 차이는 §7.10의 표다 (추천안, 사용자 위임 2026-11-21) |
| C71 `wopts` (C5) | scipy의 `wopts`(앞 호출의 체비쇼프 모멘트 `momcom`·`chebcom`을 넘겨 같은 ω·같은 길이의 다음 호출에서 다시 쓰기)는 두지 않는다(§9 C5의 (a)). 까닭: 모멘트는 한 호출 안에서 이미 수준마다 기억하고(QAWF는 주기 사이에도), 넘기는 배열이 QUADPACK의 내부 배열이라 공개 계약으로 둘 만하지 않다 (추천안, 사용자 위임 2026-11-21) |
| C72 가중치 결과의 필드 (C5) | QAWO·QAWS·QAWC는 가중치 없는 길의 구간 필드 넷이다(C55, 같은 순서). QAWF는 구간 대신 주기의 필드 넷 `cycle_integrals`·`cycle_errors`(float64)·`cycle_status`(int64, 주기마다의 QAWO `ier`)·`n_cycles`다(scipy `infodict`의 `rslst`·`erlst`·`ierlst`·`lst`, 만든 순서; 읽기 전용 배열). scipy QAWO의 `nnlog`·`momcom`·`chebmo`는 두지 않는다(QAGP의 `level`·`ndin`을 두지 않은 것과 같고 `wopts`가 없다, C71). QAWF의 ier 1·4·7은 scipy처럼 주기의 글이다. `summary()`는 `Cycles:`와 주기 표(열 `integral error status`)이고 복소수 결과의 두 줄 표는 `intervals` 또는 `cycles` 열이다. 겹친 적분의 층도 가중치를 받는다. 까닭: scipy `infodict`의 값을 필드로(C10, C55) (추천안, 사용자 위임 2026-11-21) |
| C73 가중치의 인자 (C5) | `wvar`는 `'cos'`·`'sin'`·`'cauchy'`에서 유한한 실수(ω, c), `'alg…'`에서 유한한 실수 둘의 쌍 (α, β)이고 둘 다 > −1이다(QUADPACK의 `ier = 6` 조건; scipy의 글 "≥ −1"은 원본과 다르다). 꼴이 틀리면 `TypeError`, 값이 틀리면 `ValueError`다. `weight`와 `points`를 함께 주면 `ValueError`(scipy는 경고하고 `points`를 버린다), QAWS의 `limit`은 2 이상(원본의 조건; scipy는 "The input is invalid."), `max_p1`은 1 이상 2³¹ − 1 이하의 int, `limlst`는 3 이상 2³¹ − 1 이하의 int다. 쓰이지 않는 `max_p1`(cos·sin이 아닐 때)·`limlst`(QAWF가 아닐 때, 유한 구간의 cos·sin도)는 `ValueError`다(C61의 규칙 그대로). 범위에 달린 검사(무한 구간의 `'alg…'`·`'cauchy'`, 양쪽 무한의 cos·sin, c가 a 또는 b, QAWF의 허용오차와 `limlst`)는 C++가 a == b의 지름길 다음에 한다(무한 구간의 `points`처럼, C65): a == b면 그 검사 없이 0이고 겹친 적분의 층마다 같다. 까닭: 다른 옵션과 같은 규칙 (추천안, 사용자 위임 2026-11-21) |
| C74 QAWF의 허용오차 (C5) | QAWF(`dqawfe`)는 `epsabs`만 쓴다(scipy도 `epsrel`을 넘기지 않고 조용히 버린다). 그래서 `atol`은 0보다 커야 하고(scipy: "… must have 'epsabs'>0."), 기본값이 아닌 `rtol`은 쓰이지 않는 인자라 `ValueError`다(`quad(): 'rtol' is not used with weight='cos' on an infinite range (QAWF uses 'atol' only)`, `[scipy와 다름]`). 까닭: O29·O62, 상대 허용오차를 준 사용자가 조용히 무시되지 않게 (추천안, 사용자 위임 2026-11-21) |
| C75 `dqc25s`의 오타 (C5) | netlib 원본 `dqc25s.f`의 a 끝 로그 모멘트 고리(라벨 50)는 `res24 = res12+cheb24(i)*rg(i)`라 24차 합을 12차 합에서 다시 시작한다(오타). scipy의 C 판은 `res24 = res24 + …`이고 sd도 그렇다. §7.6의 QAWS 로그 가중치 예제의 4×10⁻⁹ 차이가 이것이다: 그 줄만 고쳐 빌드한 Fortran(`tools/gen_integrate_fortran.py --probe-qc25s`)의 적분이 scipy와 비트로 같다(§7.11). 참값과의 거리는 원본 6.6×10⁻⁹, scipy·sd 2.5×10⁻⁹이다. 까닭: scipy와 같고 옳은 식이다 (추천안, 사용자 위임 2026-11-21) |
| C76 `dgtsl`의 꼴 (C5) | `dqc25f`가 \|ω(b − a)/2\| ≤ 24의 모멘트를 푸는 삼중대각 풀이는 scipy C 판의 꼴이다: LINPACK `dgtsl`처럼 부분 피벗이지만 소거를 n − 2번에서 멈추고(마지막 줄의 아래 대각 성분을 0으로 보고 푼다) 크기가 같으면 줄을 바꾸지 않는다. scipy의 기계 코드(`_quadpack` 확장의 `dqc25f`, dgtsl이 안에 펼쳐져 있다)에서 읽었고, \|ω h\| ∈ (2, 24]의 102개에서 모멘트 25개가 scipy의 `chebmo`와 모두 비트로 같다(LINPACK 꼴로는 84개; 다른 것은 \|ω h\| ≥ 19.4). 정확도는 같다: Arb로 계산한 모멘트와의 오차가 두 꼴 모두 최대 1,962 ULP(모멘트 방법 자체의 오차), 평균 10.8·11.0 ULP다. Fortran 대조(§7.6, C18)는 LINPACK 원본 그대로다. 까닭: 호출점의 비트(C5)이고 정확도를 잃지 않는다 (추천안, 사용자 위임 2026-11-21) |
| C77 `dqc25f`의 모멘트 행 (C5) | 모멘트를 `max_p1` − 1 수준까지 기억한 뒤(`momcom` = `max_p1` − 1)에는 새 수준의 첫 반쪽이 마지막 행을 다시 계산하고 둘째 반쪽(`ksave = 1`)은 앞 호출의 행 m을 쓴다. 원본은 m을 호출 사이에 남는 지역 변수에 기대고, sd는 그것을 `Work`에 둔다. scipy의 C 판은 m을 호출마다 1행(수준 0)에서 시작해 둘째 반쪽이 수준 0의 모멘트를 쓴다: 틀린 값을 수렴으로 알린다(log x·cos 300x의 [0, 1]: 참값 −0.00524에 `max_p1` = 2·3·5에서 0.0094·0.0081·0.0066, `ier = 0`). sd는 원본의 뜻대로다(`[scipy와 다름]`). 기본 `max_p1` = 50에서는 수준 49 이상의 구간(\|ω\|(b − a) > 2⁵¹)에서만 생긴다. 까닭: 틀린 결과를 따르지 않는다 (추천안, 사용자 위임 2026-11-21) |
| C78 QAWF의 ω = 0 (C5) | 원본 `dqawfe`는 ω = 0의 cos에서 `dqagie(f, 0, 1, …)`라 a와 상관없이 [0, ∞)를 적분한다(scipy도: ∫₁^∞ e⁻ˣ에 1). sd는 [a, ∞)의 QAGI다(원본처럼 `epsrel` = 0; 주기 하나). sin이면 0, `nfev` 0이다(scipy와 같다). a = 0이면 scipy와 비트로 같다(`[scipy와 다름]`은 a ≠ 0뿐). 까닭: 틀린 결과를 따르지 않는다 (추천안, 사용자 위임 2026-11-21) |
| C79 `dqawoe`의 작은 구간 (C5) | 구간이 너무 작아 더 나눌 수 없다는 판정(ier 4, status 3) max(\|a₁\|, \|b₂\|) ≤ (1 + 100ε)(\|a₂\| + 1000·uflow)에서 scipy C 판의 `dqawoe`만 uflow 자리에 ε가 있다(1000ε = 2.2×10⁻¹³; 다른 루틴은 1000·uflow, 기계 코드의 상수로 확인). sd는 scipy의 판정 그대로다: 0.37의 계단을 향해 나눌 때 폭 2×10⁻¹³에서 status 3으로 멈춘다(원본은 더 나눈다). 까닭: 멈춤 판정은 scipy 그대로(아래의 원칙)이고 값이 틀리지 않는다(수렴하지 않았다고 알린다) (추천안, 사용자 위임 2026-11-21) |
| C80 QAWO의 구간 하나 (C5) | `dqawoe`는 첫 규칙에서 끝나면(수렴 또는 `limit` = 1) `last`를 0으로 둔다(scipy `infodict`의 `last` 0). sd의 결과는 그 구간 하나다(`n_intervals` 1, [a, b]와 그 값·오차). 까닭: 구간 필드는 값을 낸 구간이다(C55) (추천안, 사용자 위임 2026-11-21) |
| C81 가중치 규칙의 FMA (C5) | scipy의 C 판(clang `-ffp-contract=on`)이 합친 곳은 원본의 한 식 안에서 곱과 합이 만나는 모든 곳이고 `std::fma`로 적었다: `dqk15w`의 누적 넷, `dqcheb`의 조합, `dqc25f`의 점 centr ± hlgth·x(원본에서 한 식이다; `dqk21`·`dqk15w`의 점은 두 문장이라 합치지 않는다)·모멘트의 식과 점화식·급수의 합·결과, `dgtsl`의 소거와 뒤 대입, `dqmomo`의 점화식, `dqc25s`·`dqc25c`의 급수의 합과 c의 사상, `dqawfe`의 abserr + 10·correc. 곱이 둘이면 왼쪽 곱을 합친다(clang의 규칙). 자리는 곱·합으로 적은 같은 소스를 clang `-ffp-contract=on -emit-llvm`으로 빌드해 `llvm.fmuladd`가 생긴 127곳에서 찾았고, 그 빌드와 이 구현이 1,500개 실행에서 점·값이 비트로 같다. 자리마다 풀어 본 표는 §7.11이다 (추천안, 사용자 위임 2026-11-21) |
| C82 큰 ω (C5) | QAWF의 주기 (2⌊\|ω\|⌋ + 1)π/\|ω\|의 ⌊\|ω\|⌋를 64비트 정수로 계산한다. scipy는 32비트라 \|ω\| ≥ 2³⁰에서 2⌊\|ω\|⌋ + 1이 넘쳐 주기가 음수가 된다(`[scipy와 다름]`; \|ω\| < 2³⁰에서는 같다). 까닭: 넘침 없는 정의된 동작 (추천안, 사용자 위임 2026-11-21) |
| C83 가중치 길의 Arb 계약 (C5) | 수렴한 결과의 참 오차는 오차 추정 + 8ε\|I\| 안이다(`bench/accuracy/integrate.py`의 `quad_weights`, `tests/integrate/test_quadpack_weights.py`). Clenshaw–Curtis 규칙의 오차 추정에는 Kronrod 규칙의 반올림 하한(50ε·resabs)이 없어 다항식의 QAWC에서 0이고, `dqc25f`의 모멘트는 정확한 값과 2×10³ ULP까지 다르므로 추정만으로는 값의 끝자리를 덮지 못한다. 109개(모음 10, 스윕 99) 모두 수렴하고 계약 안이며(최대 0.46), 추정만으로 넘는 것은 16개(QAWO 7, QAWC 9, 모두 5ε\|I\| 이하)다. scipy도 같은 비트다 (추천안, 사용자 위임 2026-11-21). 상쇄가 큰 적분(\|I\| ≪ ∫\|f\|)에서는 이 계약이 QAWO·QAWF에서 깨졌다(e⁻ˣcos ωx의 [0, ∞)에서 ω = 2²⁰ 근처부터 참 오차가 추정 + 8ε\|I\|의 13–4×10⁸배, scipy와 같은 비트). 그래서 `dqc25f`의 추정에 바닥 ε·resabs를 두어(C123) 모든 ω에서 지킨다: 계약 대비 최대 QAWO 0.075, QAWF 0.0042, 추정만으로 넘는 것은 QAWC 9개뿐이다 |
| C84 tanh-sinh 표 (C6) | scipy의 `_compute_pair` 식을 libm의 `cosh`·`sinh`·`exp`와 numpy `**2`의 곱 c·c로 계산한다(§9 C6의 (a)). 수준 0–10의 16,386값이 scipy와 비트로 같고 호출점도 같다. 표는 정확하지 않다: 같은 식을 같은 double jh(j·h의 반올림)에서 정확히 계산한 값과 최대 980 ULP(평균 47), double h의 정확한 j·h와는 최대 3,449 ULP(평균 161)다(§7.12). 수준은 처음 쓸 때 수준마다 한 번 만들어 두고 옮기지 않는다. 까닭: 호출점의 비트(C5) (추천안, 사용자 위임 2026-11-21) |
| C85 `log=True`와 음의 피적분함수 (C6) | 값은 실수뿐이다(C56의 답: 복소수는 `quad(complex_func=True)`만). f는 실수의 log를 돌려주고(복소수는 `TypeError`) 피적분함수는 음이 아니어야 한다. `tanhsinh(log=True)`에 b < a인 원소가 있으면 `ValueError`다(scipy는 log I + πi). `nsum` 안의 적분이 거꾸로인 경우(log이고 a = b = −∞인 원소; scipy는 0에서 −∞로 적분해 복소수를 낸다)는 그 적분이 NaN이다(§9 C6의 2). 오차 추정의 d1·d2(차의 log)는 scipy처럼 복소 logsumexp로 계산하고 실수부만 쓴다 (추천안, 사용자 위임 2026-11-21) |
| C86 끝점의 호출 (C6) | scipy처럼 반올림으로 끝에 닿은 점도 부르고 그 값을 버린다(가중치 0; §9 C6의 (a)). 계약: f는 특이점에서 예외 대신 inf·NaN을 돌려준다(API 문서 §1.2, §6). 까닭: 호출점이 scipy와 같다 (추천안, 사용자 위임 2026-11-21) |
| C87 numpy의 산술 (C6) | scipy의 줄마다의 `np.sum`은 numpy의 pairwise 합(누산기 8개, 128개 단위, 리덕션의 항등원 0에서 시작)을 그대로 쓴다(sd의 16누산기가 아니다; §3.5). log의 합은 scipy의 `logsumexp`를 단계마다 옮겼다(최댓값과 같은 항을 모두 빼고 log1p(Σ/m) + log m + max, 유한하지 않으면 log Σ exp). 복소 logsumexp는 numpy의 복소 연산 그대로다(시스템의 `cexp`·`clog`, numpy의 log1p = log(hypot(re + 1, im)), 복소 나눗셈과 pairwise 합). 까닭: 합이 멈춤 판정을 정하므로 호출점의 비트가 여기에 달려 있다 (추천안, 사용자 위임 2026-11-21) |
| C88 수준 (C6) | `max_level`은 0 이상 16 이하의 int, `min_level`은 0 이상의 int이고 `max_level`보다 크면 `max_level`이다(scipy와 같다). scipy는 상한이 없고 `10.0` 같은 정수값 실수도 받는다(`[scipy와 다름]`). 까닭: 수준 k의 호출은 원소마다 8·2ᵏ점이고 표도 그만큼 자라므로 상한이 메모리의 상한이다(수준 16: 원소마다 524,288점(4 MiB), 표 524,289쌍(8 MiB)). 수준 10의 간격 h₀/2¹⁰ = 7.5×10⁻⁴에서 이미 매끄러운 피적분함수는 double의 정확도에 닿는다. int만 받는 것은 다른 영역의 `max_iter`와 같다 (추천안, 사용자 위임 2026-11-21) |
| C89 `max_terms` (C6) | 0 이상 2²⁴ 이하의 int다(scipy는 상한 없이 정수값 실수도 받는다, `[scipy와 다름]`). 까닭: 직접 합은 원소마다 최대 `max_terms` + 1점을 한 번에 부르므로(scipy의 꼴) 상한이 메모리의 상한이다(2²⁴: 원소마다 128 MiB, 기본값의 16배) (추천안, 사용자 위임 2026-11-21) |
| C90 허용오차 (C6) | log=False면 유한한 0 이상의 수(다른 영역의 규칙), log=True면 inf보다 작은 실수다(−inf는 허용오차 0의 log; NaN은 `ValueError`, scipy `tanhsinh`은 NaN을 받는다). 기본값은 scipy의 것: `tanhsinh` atol 0·rtol ε^0.75, `nsum` atol 0·rtol √ε, log면 그 log (추천안, 사용자 위임 2026-11-21) |
| C91 첫 호출과 갈래 (C6) | 첫 호출은 scipy의 `eim._initialize`처럼 모든 원소에서 한 번, (k,)다(O48): `tanhsinh`은 구간의 가운데(무한한 끝은 b − 1·a + 1, 양쪽 무한은 0), `nsum`은 a(틀린 원소에서도). `tanhsinh`에서 그 값이 NaN이면 원소가 −3으로 멈춘다(scipy와 같다). 빈 배치는 f를 부르지 않는다(optimize의 배치와 같다; scipy는 빈 배열로 부른다). `nsum`은 scipy처럼 갈래마다 따로 부르고(그 갈래의 원소만, `args`도) 순서는 직접 합, 오른쪽, 왼쪽(f(−x)), 양쪽(f(x)·f(−x) 두 번)이다 (추천안, 사용자 위임 2026-11-21) |
| C92 `nsum`의 두 예외 (C6) | (1) log의 오차는 네 항의 복소 logsumexp인데 numpy는 그 합(축 0)을 원소 하나면 pairwise((e₀ + e₁) + (e₂ + e₃)), 여럿이면 차례대로 더한다. 원소의 비트가 배치와 상관없도록(§1.3) 원소 하나의 꼴을 쓴다: 원소 여럿으로 부른 scipy와는 이 오차의 끝자리가 다를 수 있다. (2) 항의 위치 a + n·step가 넘쳐 직접 합의 항 수가 유한하지 않으면 scipy는 에러(`int(inf)`)를 내지만 sd는 그 원소의 직접 합을 비운다(다른 원소는 그대로). 까닭: 원소의 비트는 배치와 상관없고 원소의 문제는 그 원소의 결과다 (추천안, 사용자 위임 2026-11-21) |
| C93 결과와 검사 (C6) | `tanhsinh`의 필드는 `integral error converged status nfev max_level`, `nsum`은 `sum error converged status nfev`이고 풀리지 않는다(C21). `max_level`은 scipy처럼 `min_level` + 반복 수 − 1(첫 호출 뒤 멈추면 −1)이다. `summary()`·`repr`는 C69의 배치 꼴이다. f의 값 검사와 글은 `derivative`의 것을 함께 쓴다(C35의 꼴: `tanhsinh(): 'f' must return shape (1, 66) like its argument, not (66,)`). 원소 패스는 2¹⁵개부터 스레드에 나눈다(§2.2; C8부터 원소 × 점 2¹⁵개, C111) (추천안, 사용자 위임 2026-11-21) |
| C94 `cubature`의 status (C7) | scipy의 문자열 `'converged'`·`'not_converged'` 대신 다른 결과처럼 `converged`(bool)와 정수 `status`다: 0 수렴, 1 `max_subdivisions`에 닿음(`[scipy와 다름]`). 유한하지 않은 값의 3은 C103이다. `message`는 `quad_vec`의 같은 번호의 글(`Target precision reached.`, `Target precision not reached.`, `Non-finite values encountered.`)이다. 까닭: 영역 간 일관성(O4, O17), 같은 알고리즘 무리인 `quad_vec`과 같은 번호·같은 글 (추천안, 사용자 위임 2026-11-21) |
| C95 영역·구간의 필드 (C7) | `cubature`의 영역은 C55의 꼴인 읽기 전용 배열 넷이다: `regions`((n, 2, ndim): `[k, 0]`이 a, `[k, 1]`이 b), `region_integrals`·`region_errors`((n,) 또는 (n, …), 값의 모양), `n_regions`. `quad_vec`도 C55의 이름 그대로 `intervals`((n, 2)), `interval_integrals`((n,) 또는 (n, …)), `interval_errors`((n,)), `n_intervals`다(scipy `full_output`의 `intervals integrals errors`). 순서는 scipy의 힙이 꺼내는 차례(오차 추정·최대 오차가 큰 것부터, 같으면 힙의 비교 그대로)이고 무한 구간·무한 축은 바꾼 변수의 것이다. 값은 `integral`에 더해지는 몫이다(치환의 부호, 뒤집은 축의 부호를 곱한다; scipy의 목록은 곱하기 전의 값). scipy `CubatureResult`의 `atol`·`rtol`은 두지 않는다(입력을 되돌려 주지 않는다). 까닭: scipy의 목록·`infodict`를 필드로(C10, C55), 순서는 C55의 "큰 것부터" (추천안, 사용자 위임 2026-11-21) |
| C96 넣지 않는 것 (C7) | `cubature`의 규칙 객체(scipy `rule=`에 `_rules.Rule`)와 `workers`, `quad_vec`의 `cache_size`·`workers`·`full_output`, 둘의 `args`는 없다. 규칙은 이름 셋(`'gk21' 'gk15' 'genz-malik'`)뿐이고 별칭 `'gauss-kronrod'`는 `ValueError`(글에 `'gk21'`을 알린다)다 (추천안, 사용자 위임 2026-11-21) |
| C97 numpy·Python의 산술 (C7) | 두 함수는 scipy의 Python 코드라 산술이 numpy·Python의 것이다(C 판 QUADPACK의 축약이 없다: FMA를 한 곳도 쓰지 않는다). 원소별 단계는 scipy의 순서 그대로, 합은 numpy의 것(C100), `norm='2'`는 numpy `linalg.norm`(원소 하나면 x·x, 아니면 0.0 + Accelerate `cblas_ddot`의 √), 힙은 Python `heapq`(C 모듈의 siftdown·siftup)를 scipy의 비교(`quad_vec`: 튜플 (−오차, a, b), `cubature`: max|오차|의 `>`)로 옮겼다(같은 값·NaN에서도 같은 차례), `min`·`max`·`**1.5`는 Python의 것이다. `cblas_ddot`은 897원소부터 벡터의 주소(1024 이상의 나머지)에 따라 끝자리가 달라지고(16바이트 정렬이면 그 아래는 같고, 페이지 정렬끼리는 같다; 2026-11-22 잼) numpy의 임시 배열은 256바이트 정렬이라 scipy 자신도 실행마다 다를 수 있다: sd는 페이지 정렬 버퍼에 옮겨 부른다(결정적이고, numpy의 배열이 페이지 정렬이면 같다). Python `**`이 넘치면 scipy는 `OverflowError`를 내지만(200·err/dabs > 10²⁰⁵) sd는 inf로 이어 간다(`min(1.0, inf)`). 까닭: 호출점의 비트(C5)와 스레드·주소와 상관없는 비트(§1.3) (추천안, 사용자 위임 2026-11-21) |
| C98 scipy의 구간 기억 (C7) | `quad_vec`은 scipy의 `LRUDict`를 그대로 둔다: 구간 값을 100×10⁶바이트 ÷ `sys.getsizeof`(첫 구간의 값: Python float 24, numpy 수 32, 배열 96 + 16·ndim + 8·원소 수)개까지 기억하고 넘치면 오래 쓰지 않은 것부터 버린다. 버린 구간을 나눌 때는 다시 계산하고(규칙 한 번 더) 결과의 그 칸은 NaN이다. 그래서 값이 큰 배열(8×50,000바이트면 249개)에서도 호출과 `nfev`가 scipy와 같고 기억이 100 MB 안이다. `cache_size`는 인자가 아니다(C96) (추천안, 사용자 위임 2026-11-21) |
| C99 `cubature`의 낮은 Gauss 규칙 (C7) | `'gk21'`·`'gk15'`의 오차 추정에 쓰는 낮은 규칙은 scipy의 `roots_legendre(10)`·`(7)` 값을 비트 그대로 둔 표다(QUADPACK의 Kronrod 점 사이의 Gauss 점이 아니다; scipy와 같다). 올바로 반올림한 값(C42)과 점은 최대 1 ULP, 가중치는 19 ULP 다르다. 높은 규칙은 scipy의 10진 리터럴(QUADPACK의 값)이다. 까닭: 이 점들이 f를 부르는 점이다(C5) (추천안, 사용자 위임 2026-11-21) |
| C100 인자의 배치와 합의 순서 (C7) | scipy의 `genz-malik` 점은 전치한 배열이라 f가 받는 (n, ndim)이 Fortran 순서다(곱 규칙은 C 순서). sd도 같은 배치의 `sd.ndarray`를 넘긴다. 점을 따라 더하는 numpy의 `sum(w·f, axis=0)`은 f의 값의 배치에 달려 있다(numpy가 곱의 배치를 입력에서 고른다): 점의 축이 메모리에서 가장 안쪽이면(점마다 값 하나, Fortran 순서, 원소별 계산이 그 배치를 이어받은 값) 값마다 pairwise, 아니면 점의 차례대로 0부터 더한다. sd는 f가 돌려준 배열의 걸음으로 같은 것을 고른다(걸음 0의 브로드캐스트 축은 보지 않는다). 그래서 C·Fortran·뒤집은 열·건너뛴 줄·브로드캐스트·3차원의 값이 scipy와 같다(§7.13). 까닭: 이 합이 멈춤 판정을 정한다 (추천안, 사용자 위임 2026-11-21) |
| C101 (−∞, b]의 치환 (C7) | `cubature`는 (−∞, b]의 축에서 f를 x = b − (1 − t)/t에서 부른다(scipy 문서의 치환). scipy 1.18.1의 코드는 −b + (1 − t)/t에서 불러 [−b, ∞)를 적분한다(∫₋∞⁰ eˣ에 inf를 수렴으로, e^(−x²)·[x > 0]의 (−∞, 1]에 0.886(참값 0.747)). 점의 역변환도 같은 뜻으로 t = 1/((b − x) + 1)이다. (−∞, b]가 아닌 경우는 scipy와 비트로 같다(`quad_vec`의 (−∞, b]는 scipy가 옳고 같다). `[scipy와 다름]`. 까닭: 틀린 결과를 따르지 않는다(C77, C78) (추천안, 사용자 위임 2026-11-21) |
| C102 무한 축과 `points` (C7) | 무한 축이 있으면 `points`의 각 좌표를 그 축의 치환으로 옮긴다(유한 축은 그대로, 실수 전체는 x = 0이 inf, 아니면 1/(x + sign x), 반무한은 1/(x − a + 1)). scipy의 `inv`는 이중 무한 축이 하나뿐이 아닌 모든 경우(이중 무한 둘 이상, 반무한만)에 `IndexError`·`ValueError`를 낸다: sd는 scipy가 뜻한 대로 옮긴다(scipy가 도는 경우는 같다). 무한한 좌표가 남은 점은 쓰지 않는다(scipy와 같다) (추천안, 사용자 위임 2026-11-21) |
| C103 유한하지 않은 값 (C7) | `cubature`가 멈췄을 때 값이나 오차의 어느 원소가 유한하지 않으면 `converged=False`, status 3(`Non-finite values encountered.`; `quad_vec`의 번호와 글)이다. scipy는 NaN에서 판정 err > atol + rtol·|est|가 거짓이라 멈추고 `'converged'`라고 알린다. 걸음은 그대로다(멈추는 곳은 scipy와 같다). `[scipy와 다름]`. 까닭: 틀린 수렴을 알리지 않는다, `quad_vec`과 같은 번호 (추천안, 사용자 위임 2026-11-21) |
| C104 `quad_vec`의 인자와 값 (C7) | `a`, `b`는 실수(±inf, NaN은 `ValueError`; a > b, a == b, 같은 무한은 scipy 그대로 계산한다), `atol`·`rtol`은 유한한 0 이상의 수, `norm`은 `'2'`·`'max'`(scipy의 호출 가능한 노름과 None은 받지 않는다), `limit`은 1 이상의 int(scipy는 실수도), `points`는 `quad`의 꼴(C65; NaN `ValueError`, 무한 구간에서도 받아 scipy처럼 옮긴다), `quadrature`는 None과 이름 셋이다. f의 값은 실수 하나 또는 실수 배열(첫 값이 꼴을 정하고 다른 꼴은 `ValueError`, 빈 배열·복소수·bool·글은 에러)이고 float64로 계산한다(scipy는 float32 배열이면 float32로 계산한다, `[scipy와 다름]`). `nfev`는 scipy의 `neval`이다(0으로 둔 |t| < √DBL_MIN의 점도 세고, 사다리꼴은 scipy의 `lru_cache`처럼 나누기 하나에서 같은 점을 한 번). 까닭: 다른 함수의 규칙, scipy의 걸음 (추천안, 사용자 위임 2026-11-21) |
| C105 `cubature`의 인자 (C7) | `a`, `b`는 길이가 같은 비지 않은 1차원 실수 시퀀스다(scipy는 수 하나를 브로드캐스트한다, `[scipy와 다름]`). NaN은 `ValueError`, 같은 무한(a_i = b_i = ±inf)도 `ValueError`(scipy는 그 축의 f를 inf에서 불러 0이나 NaN을 낸다). `rtol`·`atol`은 유한한 0 이상의 수, `max_subdivisions`는 1 이상의 int(scipy의 None(상한 없음)은 §1.3의 "정해진 시간"에 어긋나고, 0은 그래도 한 번 나누어 뜻이 틀린다), `genz-malik`은 ndim ≥ 2(scipy와 같다), `points`는 점마다 ndim개의 실수(NaN `ValueError`)다. f의 값은 (n,) 또는 (n, …)이고 뒤 모양이 호출마다 같아야 한다(빈 값은 `ValueError`). `nfev`는 f에 넘긴 점의 수다. 점이 모든 영역을 폭 0의 조각으로 가르면 f를 부르지 않고 0(Python 수)이다(scipy와 같다) (추천안, 사용자 위임 2026-11-21) |
| C106 `summary()`와 `repr` (C7) | C69의 상자에 더한다: 배열 값은 머리에 `Integral: shape (3,)`(`cubature`는 `Error estimate`도)을 적고 원소마다 한 줄의 표(열 `integral`, `cubature`는 `error`도, 줄 이름은 원소의 첨자)를 둔다. `quad_vec`의 구간 표는 배열 값이면 `integral` 열을 뺀다. `cubature`의 영역 표는 열 `a b integral error`(모서리는 (x, y, …), 배열 값이면 `a b max error`)이고 머리에 `Subdivisions:`·`Regions:`가 있다. `repr`는 바탕의 한 줄 꼴이다(짧은 배열은 `[v, v]`) (추천안, 사용자 위임 2026-11-21) |
| C107 Arb 계약 (C7) | 수렴한 결과의 참 오차는 오차 추정 + 8ε\|I\| 안이다(`quad_vec`은 그 노름으로, `cubature`는 원소마다; `cubature`의 \|K − G\| 추정에는 반올림 하한이 없다, C83과 같은 꼴). 넘는 것은 scipy도 같은 비트인 짧은 목록이다(`tests/integrate/adaptive.py`의 `QUAD_VEC_SHORT` 5개, `CUBATURE_SHORT` 규칙·무리마다 3–8개: 끝점 특이점 x^−0.9…−0.7, 좁은 봉우리, f10·f12). 계약 대비 최대는 0.95·0.96이다(§7.13) (추천안, 사용자 위임 2026-11-21) |
| C108 `newton_cotes`의 int 기억 (C8) | int N의 두 번째 호출부터는 기억한 `(an, B)`(64개까지)의 사본을 돌려준다. 전에는 부를 때마다 위치 N + 1개를 `Fraction`으로 만들고 그 튜플로 `lru_cache`를 찾아 N = 8에 3.8 µs(scipy 1.3 µs)였고, 지금 0.25 µs다. 값은 같은 반올림 값의 복사이고 `an`은 늘 새 배열이라 고쳐도 다음 호출은 그대로다. 까닭: 같은 입력의 빠른 길(C70과 같은 꼴) (추천안, 사용자 위임 2026-11-22) |
| C109 Gauss–Legendre의 상한 (C8) | `fixed_quad`의 `n`은 10⁴ 이하의 int다(`fixed_quad(): 'n' must be at most 10000`, `[scipy와 다름]`). 표 밖(n > 100)의 처음 계산은 점마다 세 항 점화식의 double-double Newton(C54)이라 n²에 비례한다: 1스레드·10스레드로 1,000점 0.07·0.008초, 3,000점 0.59·0.077초, 10⁴점 6.4·0.88초, 2×10⁴점은 10스레드 3.3초였다(10⁶점은 사실상 끝나지 않는다, §1.3의 "정해진 시간"). P_n을 점근 전개로 O(1)에 계산하는 알고리즘(Hale–Townsend, Bogaert)은 double-double 정확도의 전개와 오차 한계를 새로 세워야 해서 두지 않았다. 상수배의 개선(점화식의 나눗셈을 역수 표의 곱으로, 수렴한 double Newton 뒤의 double-double Newton을 한 번으로)은 C54의 오차 한계를 Arb로 다시 재야 하므로 나중의 최적화 세션에 남긴다(§7.14). scipy `roots_legendre`도 모든 n에서 Golub–Welsch(O(n²))다. 까닭: 고정 규칙에서 수천 점을 넘는 점은 쓸 일이 거의 없고(적응 구적이 맞다) 상한이 시간의 상한이다 (추천안, 사용자 위임 2026-11-22) |
| C110 `newton_cotes`의 상한 (C8) | N은 256 이하다(int N, 위치는 257개까지; `newton_cotes(): 'rn' must be at most 256 (the number of intervals)`, `[scipy와 다름]`). 정확한 유리수의 첫 계산이 N³ 꼴이다: N = 50 6 ms, 100 25 ms, 200 0.13초, 400 0.91초. scipy는 N ≥ 15에서 Vandermonde 역행렬이라 N = 30의 `B`가 10⁶배 틀린다(§7.9). 까닭: C109와 같다 (추천안, 사용자 위임 2026-11-22) |
| C111 원소별 무리의 병렬 기준 (C8) | `derivative`, `tanhsinh`, `nsum`의 원소 패스(점 쓰기, 갱신과 판정, `nsum`의 직접 합)는 일(진행 중인 원소 수 × 원소마다의 점 수)이 2¹⁵점 이상이면 스레드에 나눈다(`csrc/elementwise/loop.hpp`의 `for_weighted`·`advance_weighted`: 조각은 2¹³점쯤이고 한 원소 이상, 조각이 17개 이상이라 자는 일꾼도 깨어난다; htest H141). 전에는 원소 2¹⁵개가 기준이라 원소 10³–10⁴개 × 66점이나 10개 × 2¹⁷항의 직접 합이 한 스레드였다. 또 이 무리의 호출(`CallPoints`)은 점의 사본을 두지 않는다(다시 읽는 것은 optimize의 배치뿐이다; 10⁵ × 66점의 첫 호출에서 53 MB를 0으로 채우는 일이 없어진다). optimize의 `for_elements`·`advance`·`Call`은 그대로다(C22). 원소마다의 산술과 남은 원소의 순서가 같아 비트가 그대로다(A/B의 모든 결과 해시, 골든, 1·3·10스레드). 잰 것은 §7.14 (추천안, 사용자 위임 2026-11-22) |
| C112 `approx_fprime`의 간격 키우기 (C8) | 정확한 차분(`Differences`, C15)은 처음 간격 2⁻⁸\|x_j\|의 열의 반올림 한계가 열 크기의 2⁻³⁶(1.5×10⁻¹¹)을 넘으면 간격을 키워 다시 찾는다(O104의 키우기 고리 그대로; 전에는 √ε = 2⁻²⁶을 넘을 때만). 그 사이의 시작은 아래로만 찾아 처음 간격의 반올림이 그대로 오차였다: exp의 x = 3×10⁻⁶·10⁻⁵·10⁻⁴에서 상대 1.5×10⁻⁸·5.1×10⁻⁹·2.6×10⁻¹⁰(문서의 "약 10⁻¹¹–10⁻¹³" 밖, §6.1). 지금 7.4×10⁻¹⁴·1.5×10⁻¹³·9.3×10⁻¹⁴다(`sd.exp`로 잰 값; `math.exp`로는 10⁻⁵에서 4.6×10⁻¹³). 초등 함수 10개 × 32점(10⁻¹⁴–1)의 탐침에서 바뀐 47경우는 모두 절대오차가 줄었고(최대 10⁵배) 늘어난 것은 없다. 그래도 f가 f′보다 훨씬 큰 몇 점(cos·exp(−x²)의 10⁻¹¹–3×10⁻⁷, log(1 + x)의 10⁻¹⁴)은 틀린 채였고 C122가 고쳤다. 도함수가 1 근처인 함수의 최대 상대오차는 1.6×10⁻⁸ → 2.5×10⁻¹¹(x = ±3×10⁻³)이다. 호출 수는 그 x에서 13 → 25–57번이다(MGH 47경우의 x0와 초등 함수 11개 × 9점의 수는 그대로). 키운 간격이 반올림 수준을 벗어나지 못하면 전과 같은 찾기로 돌아간다. 골든은 regression의 둘이 바뀌어 그 둘만 다시 기록했다: `nls ENSO/start2/bounded/diff`(bse가 해석적 야코비안으로 푼 결과와 상대 3.0×10⁻¹² → 3.7×10⁻¹³)와 `nls MGH17/start1/bounded/diff`(조건수 2.6×10¹³의 실패한 적합: b5 열의 간격이 2⁻¹² → 2⁻¹³, 그 열의 오차 2.0×10⁻¹⁰ → 2.9×10⁻¹⁰, bse가 상대 5×10⁻⁶ 바뀐다). optimize와 다른 영역의 골든은 그대로이고, exp의 3×10⁻⁶·10⁻⁵·10⁻⁴ 셋을 optimize 골든에 더했다. 까닭: 문서가 약속한 정확도를 지킨다(API.md §4의 넷째 규칙: 고침이 필요할 때 비트를 깬다) (추천안, 사용자 위임 2026-11-22) |
| C113 C8의 측정 (C8) | H139처럼 모든 공개 함수를 `bench/speed`의 사례로 크기마다 scipy와 견준다. 없던 사례를 더했다: `romb` 2¹⁰ + 1·2²⁰ + 1점, `fixed_quad` n = 200, `newton_cotes` 8·20, `quad`의 `points`·`complex_func`, `tplquad`·`nquad`, `nsum`의 배치 10·10³, `derivative`의 한쪽 스텐실, `jacobian` n = 10·30과 벡터 값, `hessian` n = 10(integrate 40, differentiate 10사례). 기계가 바빠(부하 2.7–3.9) 1라운드로 재고 기준선에 올리지 않았다: `bench/integrate/speed.md`·`bench/differentiate/speed.md`는 고친 뒤의 실행 파일에서 만든 표(`bench.speed table --from`)이고, 고친 것은 고치기 전 빌드(796c7f6의 C++)와 새 빌드를 번갈아 띄운 프로세스에서 1·10스레드로 재고 결과의 해시를 견주었다(A/B). 조용한 기계의 전체 재측정과 깊은 최적화는 나중의 세션이 한다 (추천안, 사용자 위임 2026-11-22) |
| C114 남긴 것 (C8) | 비트나 계약을 바꿔야 하거나 이득이 작아 고치지 않은 것(§7.14): 함숫값 21개의 `quad`(scipy의 0.95배; 결과의 구간 배열 셋과 인자 검사, C70), QAWS의 가장 작은 경우(0.96–0.98배), `nsum` 배치의 1스레드(0.93배: 시간의 대부분이 사용자 함수의 sd `**`이고 그 1스레드가 numpy보다 약 10% 느리다, 배열 영역), `tanhsinh` 10⁵원소의 첫 호출(새 (k, 66) 배열의 페이지 폴트), Gauss–Legendre 처음 계산의 상수배(C109), `newton_cotes`의 첫 계산(C110) (추천안, 사용자 위임 2026-11-22) |
| C115 일관성 점검 (C8) | 영역의 모든 공개 함수와 optimize의 배치 넷·`approx_fprime`을 같은 틀로 불러 견주었다(없는 키워드, 틀린 `atol`·`rtol`·`max_iter`·`limit`·`args`·`axis`·`log`·`order`·`step_factor`·문자열 옵션, 부를 수 없는 f, NaN·문자열·bool 끝, 브로드캐스트되지 않는 `args`, `Series` 입력, 결과의 `repr`·`summary()`와 그 인자, 필드와 배열에 쓰기). 모두 `이름(): ` 꼴이고 같은 실수는 같은 문장이며, 상한은 `'max_level' must be at most 16`의 꼴이다(C109·C110의 글도). 영역 안에서 고칠 것은 없었다. 다른 영역과 다른 것과 다른 영역에서 본 것은 §7.14에 적는다 (추천안, 사용자 위임 2026-11-22) |
| C116 코드 리뷰 (C8) | C8에서는 전체 코드 리뷰를 하지 않는다. 다음의 리뷰 세션이 영역 전체를 본다(H149와 같다) (추천안, 사용자 위임 2026-11-22) |
| C117 `cubature`의 크기 상한 (리뷰) | 영역 하나의 점(오차 추정의 호출이 받는 두 규칙의 점: gk21 21^d + 10^d, gk15 15^d + 7^d, genz-malik 2^d + 4d² + 4d + 2)의 값(점 수 × d)이 2²⁵개를 넘으면 f를 부르기 전에 `ValueError`다(`cubature(): rule='gk21' in 8 dimensions needs 21**8 + 10**8 points per region of 8 values each, more than 2**25 values; use rule='genz-malik'`). C++의 곱 규칙도 곱마다 같은 상한으로 넘침을 검사한다(15차원부터 `size_t`의 21^d가 넘쳐 할당보다 넓게 쓰던 정의되지 않은 동작). `points`가 만드는 처음 영역은 나눌 때마다 세어, 영역 수 × 영역의 점이 2²⁴을 넘으면 `ValueError`다. gk21·gk15는 5차원, genz-malik은 20차원까지다: 점의 배열 하나(규칙의 표, f에 넘기는 배열)가 256 MiB를 넘지 않는다. 처음 안은 점 2²⁴개였는데, 그 안에서도 gk15 6차원(1,150만 점)·genz-malik 22차원(420만 점)이 RSS 2.4 GB를 넘어 값의 상한으로 조였다 (추천안, 사용자 위임 2026-11-22). 잰 최대 RSS는 gk21 5차원(418만 점, 2.1×10⁷값) 1.2 GB, genz-malik 20차원(105만 점, 2.1×10⁷값) 0.88 GB(값마다 대략 7 × 8바이트: 규칙의 표 둘, f에 넘기는 배열, f의 중간 배열)다. 까닭: 정해진 시간과 메모리(§1.3, C109·C110의 꼴); 8차원 gk21은 점의 표만 2 TB라 프로세스가 죽었다. 골든: 새 케이스 `err cubature gk21 8-D`(값의 상한으로 조이며 글이 바뀌어 다시 기록) (추천안, 사용자 위임 2026-11-22) |
| C118 `order`의 상한 (리뷰) | `derivative`·`jacobian`·`hessian`의 `order`는 18 이하다(`derivative(): 'order' must be at most 18`, `[scipy와 다름]`: scipy는 상한이 없고 30부터 틀린 값을 수렴으로 낸다). 상한은 Arb로 쟀다: 초등 함수 11개 × 9점의 중앙 스텐실에서 2–18의 모든 짝수 차수가 계약(atol + rtol·\|df\|) 안이고(최대 0.21), 20은 log의 10⁴에서 1.02배, 28은 sqrt의 10⁴에서 1.45배, 32–44는 64배, 46부터 수렴하지 않는 점이 늘고, 60부터 틀린 값이 수렴으로 나간다(exp′(1): order 160에서 0, status 0; order 1000은 150초에 끝나지 않고 10⁸은 RSS 1 GB를 넘는다). 첫 반복의 안쪽 마디 h/c^(n−1)이 반올림 쪽으로 내려가고 정확한 가중치(C28)가 O(n²)개의 큰 유리수 연산이기 때문이다. 넓힌 격자(sin(ωx) ω ≤ 300, exp·log·sqrt·atan의 10⁻³–10³)에서는 상한 안에서도 넘는 점이 있다(order 6의 sin(100x): 간격 0.5가 주기보다 커서, order 16의 atan(10³)): 차수의 문제가 아니라 기본 `initial_step`의 문제라 상한에 넣지 않았다. 골든: `derivative order 20`이 상한의 `ValueError`로 바뀌었고(그 까닭), `derivative order 18`을 더했다 (추천안, 사용자 위임 2026-11-22) |
| C119 `quad` 계열의 유한하지 않은 결과와 허용오차의 빠른 길 (리뷰) | `quad`·`dblquad`·`tplquad`·`nquad`에서 QUADPACK이 0으로 끝났는데 적분이나 오차가 유한하지 않으면 `converged=False`, status −3, `Non-finite values encountered.`다(`tanhsinh`·`nsum`의 번호, `quad_vec`의 글; `[scipy와 다름]`: scipy는 ier 0). 0이 아닌 ier는 그대로 둔다(scipy의 status 1–5 기록이 비트로 같다). 까닭: C103과 같이 틀린 수렴을 알리지 않는다; 영역의 다른 함수와 optimize의 `converged`(마지막 값이 유한해야 참)와 같다. 또 `quad`·`dblquad`의 기본값 빠른 길은 형(`int`·`float`)을 먼저 보고 값을 견준다: 배열 `atol`은 `quad(): 'atol' must be an int or a float, not ndarray`이고(전에는 접두 없는 진릿값 에러), numpy 0차원 배열은 `quad`도 `dblquad`처럼 받지 않는다(영역의 다른 함수와 같다). 골든: `quad inf`(converged·status가 바뀜), 새 케이스 `dblquad inf` (추천안, 사용자 위임 2026-11-22) |
| C120 문서로만 둔 것 (리뷰) | 동작은 scipy 그대로 두고 문서에 적는다: (1) 원소별 무리의 호출 하나의 메모리(API 문서 §6: 원소 k개 × 가장 긴 원소의 점 수 m × 8바이트, `nsum` n = 40의 (40, 131,072)에서 최대 RSS 154 MB; differentiate §3: `hessian`의 안쪽 호출 (m, (m·p)²), 8m³p² 바이트). 원소를 길이별로 묶어 부르는 것은 호출점이 scipy와 같다는 C5와 부딪친다. (2) `max_iter`의 하한: `derivative` 무리는 1, optimize의 원소별 넷은 0이고 둘 다 scipy의 것이다. (3) 간격이 x의 ulp보다 작으면 점이 x로 반올림되어 df 0·error 0으로 수렴한다(`derivative(sd.sin, [1e300, 1e17, 1e16])`; scipy와 같은 점·판정): \|x\|에 맞춘 `initial_step`을 준다 (추천안, 사용자 위임 2026-11-22) |
| C121 결과의 공개 생성자 (리뷰) | `IntegrateResult`·`DifferentiateResult`·`OptimizeResult`(그리고 같은 바탕 `stardust._results._FieldResult`의 sparse 결과)의 공개 생성자는 넘긴 `sd.ndarray`·numpy 배열·수의 리스트를 결과 자신의 읽기 전용 `sd.ndarray`로 한 번 복사하고(numpy 0차원은 그 수, 튜플은 원소마다), 공통 필드의 종류를 검사한다: `message`·`method`는 str, `converged`는 bool이나 bool 배열, 개수와 번호(`status nfev n_iter njev nhev`, integrate의 `max_level n_intervals cycle_status n_cycles subdivisions n_regions`)는 int나 정수 배열, 나머지는 str·bool·dict·set이 아닌 값(수, 배열, 튜플, 결과, None)이다. 틀리면 `IntegrateResult(): 'integral' must be a number or an array of numbers, not str`(`TypeError`). 위치 인자는 `_calls.keyword_init`으로 클래스 이름의 글이다. 함수가 만드는 결과(`_make`)는 복사도 검사도 하지 않는다. 까닭: 문서의 "넘긴 배열은 한 번 복사한다"와 API.md §2.2의 읽기 전용을 지킨다(전에는 numpy 배열을 그대로 두어 호출한 쪽의 변경을 따라가고 repr에서 빠졌다). 골든: 세 클래스의 `err … positional`(글이 바뀜; 문장은 Python의 `takes 0 positional arguments but 1 was given`이다, stats T53·API.md §2.1. 1·3·10스레드 같은 비트) (추천안, 사용자 위임 2026-11-22) |
| C122 `approx_fprime`의 작은 0 아닌 x (리뷰, C112의 뒤) | 정확한 차분의 간격 찾기(`Differences.choose_step`)에 둘을 더했다. (1) 키우기(O104, C112)가 상한 2⁻⁷(x_j = 0의 처음 간격의 두 배)에 닿았는데 열의 반올림이 √ε을 넘으면 전에는 처음 간격부터 아래로만 찾았다(반올림 수준의 간격이라 cos의 3.41×10⁻⁸에서 −4.77×10⁻⁷, 부호까지 틀림). 이제 그 열이 잡음이 아니면(반올림의 16배 밖) 그 간격부터 찾는다. (2) 처음 간격에서 찾은 추정들이 끝내 맞지 않으면(마지막 두 추정의 상대 차이 > 2⁻²⁰) x_j = 0의 처음 간격 2⁻⁸에서 다시 찾고, 그 추정들이 맞으면 그것을 쓴다: f 자신의 반올림이 u\|f\|보다 큰 경우다(math.log(1 + x)의 10⁻¹⁴에서 1 + x가 2⁻⁵²의 배수로 반올림되어 처음 간격 2⁻⁵⁴의 추정이 −1/3, 다음이 0). 둘 다 \|x_j\| < 1(처음 간격이 2⁻⁷보다 작은 경우)에서만 갈린다. 잰 것(Arb): 정확도 스윕 `approx_fprime`의 `small \|x\|`(10함수 × 32점, ±10⁻¹⁴–1)의 최대 절대오차 1.33 → 5.3×10⁻¹²(scipy 7.3×10⁻⁸), 상대오차가 큰 것은 도함수가 값보다 훨씬 작은 점뿐이다(cos의 10⁻¹⁴: 절대 10⁻¹⁴, 상대 1). MGH 47경우(기울기·야코비안)와 초등 함수 11개 × 9점은 값과 호출 수가 비트까지 그대로이고, 무작위 660점(10⁻¹⁵–10^2.5)에서 바뀐 것은 cos의 작은 x 20점뿐으로 모두 절대오차가 줄었다(최대 1.7×10⁸배). 호출 수는 cos의 10⁻¹¹–10⁻⁷에서 17 → 21번, log(1 + x)의 ±10⁻¹⁴에서 9 → 29–33번이다. R258: C112의 "1.6×10⁻¹³"(exp, x = 10⁻⁵)은 `sd.exp`로 잰 1.5×10⁻¹³이고 `math.exp`로는 4.6×10⁻¹³이다(3×10⁻⁶ 7.4×10⁻¹⁴, 10⁻⁴ 9.3×10⁻¹⁴는 둘이 같다). scipy의 전진 차분 간격으로 물러나는 길(ASK-4의 (b))은 쓰지 않았다. 골든: 바뀐 것 없음(regression의 `nls */diff`도 그대로); 새 케이스 `approx_fprime cos x0=3.41e-08`, `approx_fprime log(1+x) x0=1e-14` (추천안, 사용자 위임 2026-11-22) |
| C123 QAWO·QAWF 추정의 반올림 바닥 (리뷰, C83의 뒤) | `dqc25f`의 Clenshaw–Curtis 추정(\|12차 − 24차\|)을 ε·resabs 아래로 내리지 않는다(resabs = 24차 체비쇼프 계수의 절댓값 합 × 반구간, ∫\|f\|의 어림; resabs > uflow/ε일 때). Kronrod 규칙의 바닥(50ε·resabs)과 같은 꼴이고 계수는 ε다. 까닭: 계수와 모멘트의 반올림이 값에 ε·∫\|f\| 크기로 남는데 두 급수의 차는 그것을 보지 못해, 상쇄가 큰 적분에서 C83이 깨졌다(e⁻ˣcos ωx의 [0, ∞): ρ = 참 오차/(추정 + 8ε\|I\|)가 ω = 1.37·2²⁰에서 13, 2²⁴에서 3.4×10³, 2⁴⁸에서 4.3×10⁸; ω < 2³⁰은 scipy와 같은 비트, 그 위는 C82의 sd 길). 바닥을 두면 ω = 1.37·2ᵏ(k = 0–62)의 cos·sin과 [0, 1]의 QAWO에서 ρ ≤ 0.11이고 모두 status 0이다. 결과: QAWO·QAWF의 `error`는 scipy보다 작지 않고(`[scipy와 다름]`), 바닥이 판정·이분·외삽을 바꾸는 입력은 점·status·구간 수·값도 다르다(가중치 스윕 99개와 모음은 점·값·`nfev`·status가 그대로이고 error만 커진 것 9개, 만든 400개 가운데 하나 `99-cos3-cos`가 rtol 10⁻¹⁴에서 status 1). 정확도 스윕 `quad_weights`: QAWO 계약 대비 0.46 → 0.075(추정만으로 넘는 것 7 → 0), QAWF 0.0042 그대로. scipy와 비트로 견주던 테스트는 cos·sin에서 error를 "scipy 이상"으로 견준다(DESIGN §1.6의 넷째 규칙). 골든(error·주기 오차·요약이 바뀜, `qawf limlst 3`은 외삽이 바뀌어 값도 2 ulp): `quad quadpack/qawf`, `quad weights/qawo/exp-{cos,sin}/{-2.0/10.0, -1.0/40.0, 0.3/45.0, 1.0/1000.0}/…`(8개), `quad weights/qawo/exp-sin/0.5/-7.0/0.0/3.0`, `quad weights/qawf/{lorentz-cos/1.0, lorentz-cos/5.0, rsqrt-cos/3.0, rsqrt-sin/3.0}`, `quad cos small interval`, `quad qawf limlst 3`, `quad qawf limit 3`, `quad qawf reversed`, `quad qawf summary`, `quad weight complex summary`; 새 케이스 `quad qawf omega 2^40` (사용자 결정 2026-11-22) |

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
- 입력은 위치로(`func`, `a`, `b`, `y`, `x`), 옵션은 키워드로만. 별칭이 없다. float64 실수로 계산한다(bool·int64는 바꿔서). 복소수 피적분함수는 `quad(complex_func=True)`만 받는다(C56).
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 사용자 함수가 결정론적이면 결과도 결정론적이다.
- **모든 호출은 정해진 시간 안에 끝난다.** 적응 방법은 구간 수(`limit`), 수준(`max_level`), 항 수(`max_terms`), 반복 수(`max_iter`)에 상한이 있다. 상한에 닿으면 마지막 값과 `converged=False`를 돌려준다.
- 경고를 내지 않는다. scipy가 `IntegrationWarning`으로 알리는 것(구간 수 상한, 반올림, 나쁜 피적분함수, 발산)은 결과의 `converged`, `status`, `message`로 알린다. 화면에 쓰지 않는다(`romb(show=True)` 없음).
- 참값은 Arb만 쓴다. mpmath는 쓰지 않는다.
- 영역 간 일관성이 먼저다: 결과 필드의 이름(O4), 메시지의 꼴(`quad(): …`), 원소별 반복 무리의 호출 규약(O38), `args`(O39), 배열 결과의 읽기 전용 규칙(O103, API.md §2.2)은 optimize를 따른다.

### 1.4 다른 영역과의 관계
- **optimize** (C3, C4): 차분 간격의 커널이 이 영역으로 왔고(§6.1), 배치 고리가 세 영역이 함께 쓰는 자리로 갔다(§6.2; 둘 다 C1). optimize의 공개 함수와 결과는 비트까지 그대로다(골든 불변).
  - 옮긴 뒤에도 `diff.hpp`의 간격 규칙을 쓰는 곳은 optimize뿐이다: `sd.differentiate.derivative`는 scipy처럼 다른 방법(중첩 스텐실의 고차 차분, §3.7)이라 `approx_derivative`의 간격을 쓰지 않는다. 옮기는 까닭은 H17의 원칙(함께 쓰는 커널은 그 본질의 영역에 둔다)이다.
- **stats·special** (C7): 그대로 둔다. 지금 있는 전용 구적: `csrc/stats/mvcdf.hpp`(다변량 정규·t cdf의 적응 Gauss–Kronrod, QUADPACK의 QK21을 자기 사본으로), `csrc/stats/dist/noncentral.hpp`와 `csrc/special/ops/noncentral.hpp`(sinh 사상 위의 사다리꼴 규칙). 나중에 쓸 곳: 분포의 `expect`(scipy `rv_continuous.expect`는 `quad`), 사용자 정의 분포의 cdf·적률, 회귀의 적분 우도. 그때 QK21 표를 함께 쓸지 정한다.
- **배열**: 표본 규칙(`trapezoid` 등)은 sd의 합 규칙(docs/api/array.md §7.8: 누산기 16개·512개 단위의 pairwise)을 쓴다(§3.6).

---

## 2. 입력과 결과

### 2.1 결과 객체 (C9, C10, C13)
- 모듈마다 결과 클래스 하나(C9): `sd.integrate.IntegrateResult`, `sd.differentiate.DifferentiateResult`(optimize의 `OptimizeResult`와 같은 이름 짓기: 모듈 이름 + Result). 규칙은 `OptimizeResult`와 같다: 함수가 주는 필드만 있고 없는 필드는 `AttributeError`, 필드는 읽기 전용, 배열 필드는 읽기 전용이고 읽을 때 복사하지 않는다(O103), `repr`는 한 줄, dict처럼 쓰지 않는다. 바탕은 `src/stardust/optimize/_result.py`의 것을 끌어내 셋이 함께 쓴다(`OptimizeResult`의 동작은 그대로).
- 이름(C4, O4): `converged`(scipy `success`), `n_iter`(scipy `nit`), `nfev`(scipy `quad`의 `neval`도), `status`, `message`(배치 빼고). 값의 필드(C13): 적분은 어디서나 `integral`, 급수는 `sum`, 오차 추정은 `error`, 미분은 `df`·`ddf`(scipy의 것).
- C0에서 만든 것: 바탕은 `src/stardust/_results.py`의 `_FieldResult`다(필드 속성, 읽기 전용, `_Lazy`, 생성자 `_make`, 배치의 한 줄 `repr`). `OptimizeResult`는 그 위에 자기 `__init__`·`repr`·`summary()`를 그대로 두었고(Python의 호출 모양 에러가 클래스 이름을 대므로 `__init__`는 클래스마다 둔다), optimize의 테스트와 골든 3,876케이스가 그대로다. 그 뒤 O120(2026-11-21)으로 `OptimizeResult`의 `repr`도 바탕의 한 줄 꼴(`one_line`)이 되었다. `IntegrateResult`·`DifferentiateResult`의 `repr`는 API.md §2.3의 꼴(`IntegrateResult(integral=0.746824, error=8.29e-15, converged=True, nfev=21)`)이다.
- 둘로 풀리는 것(C10)은 결과를 만드는 함수가 정한다: `quad` 계열은 `IntegrateResult._make(…, unpack=True)`로 만들고, 그 결과만 `a, b = r`, `r[0]`, `len(r) == 2`가 된다. 다른 결과(`tanhsinh`, `nsum`, `cubature`, 공개 생성자로 만든 것)에서는 `TypeError`이고, `bool(r)`은 늘 참이다(C21). `summary()`는 C2(`DifferentiateResult`)와 C4(`IntegrateResult`, C69)에서 만들었다(C20).
- `quad` 계열(`quad dblquad tplquad nquad quad_vec`)은 scipy가 튜플 `(y, abserr)`를 돌려준다. 여기서는 결과 객체이고(C10) `integral, error = sd.integrate.quad(f, 0, 1)`처럼 둘로 풀린다(htest H7의 규칙). `full_output`은 없고 scipy `infodict`의 값(`neval`, `last`, `alist`…)은 필드다(docs/api/integrate.md §1.3, §2).
- 원소별 반복 무리의 결과는 필드마다 배치 모양이고, 입력이 모두 실수면 Python 수다(O48).

### 2.2 원소별 반복 무리 (C4)
`derivative`, `tanhsinh`, `nsum`은 optimize의 배치 넷과 같은 틀이다: 배치의 원소마다 따로 풀고, 반복마다 아직 끝나지 않은 원소의 점을 모아 사용자 함수를 한 번 부른다.
- **status** (O17, scipy의 번호): 0 수렴, −1(`derivative`: 오차 추정이 늘어 멈춤; `nsum`: `a`·`b`·`step`이 틀림; `tanhsinh`: 쓰지 않음), −2 반복·수준 상한, −3 유한하지 않은 값, −4(`nsum`만: 부분합의 마지막 항이 허용오차보다 커서 오차 추정이 허용오차를 넘음). scipy 원소별 무리의 −4는 `callback`이 멈춘 것인데 `nsum`은 같은 번호를 다른 뜻으로 쓴다. `callback`이 없으므로(O37) 그 −4는 없고, `nsum`의 −4는 scipy의 뜻 그대로다.
- **호출 규약** (O38 + 여러 점): optimize의 배치는 원소마다 점 하나라 사용자 함수가 (k,)를 받는다. 이 무리는 원소마다 여러 점을 한 번에 부른다(scipy: `derivative`는 첫 반복 order개, 다음부터 2개; `tanhsinh`은 처음 66개, 다음 수준부터 64·128·…). 넘기는 꼴은 (k, p) 2차원이다(C26: 원소의 순서는 입력의 C 순서, `args`는 (k, 1)).
- scipy의 첫 호출은 원소마다 점 하나(입력 모양 그대로: `tanhsinh`은 구간의 가운데, `derivative`는 x)다. O48처럼 첫 호출도 1차원 (k,)이다.
- `args`(O39): 값마다 배치 모양으로 브로드캐스트하고, 호출마다 남은 원소로 줄여 넘긴다. `kwargs`(scipy 1.18의 이 셋에는 있다), `callback`, `preserve_shape`는 없다(C14).
- 허용오차는 scipy의 `tolerances` dict 대신 키워드다(O36): `derivative(atol=, rtol=)`, `nsum(atol=, rtol=)`. `tanhsinh`은 scipy도 키워드다.
- 배치 결과의 `n_iter`·`nfev`는 그 원소가 멈출 때까지의 수다. `tanhsinh`은 scipy처럼 `n_iter` 대신 마지막 수준 `max_level`을 준다(scipy `maxlevel`; C12).
- 원소를 지나가는 패스의 일(진행 중인 원소 수 × 원소마다의 점 수)이 2¹⁵점 이상이면 그 패스를 스레드에 나누고 GIL을 놓는다(C111; optimize §3.3.1의 기준은 원소마다 점 하나인 경우다).

### 2.3 사용자 함수와 호출
- 사용자 함수는 Python 호출 가능한 객체다. 반복은 C++가 돌고 함숫값은 콜백으로 받는다(optimize와 같다). 콜백을 부르는 동안 GIL을 쥔다.
- **`quad` 계열**: scipy처럼 점 하나마다 Python `float`로 부르고 실수 하나를 받는다(numpy·Python의 수, 0차원 배열). 순서는 QUADPACK의 순서다: QK21이면 가운데, 다음에 Gauss 점의 쌍(c − h·x₂, c + h·x₂, c − h·x₄, …), 그 뒤 Kronrod만의 점의 쌍(§7.4: scipy와 21점 모두 비트까지 같다). 한 규칙의 점들을 배열 하나로 부르는 선택은 scipy에 없어 두지 않는다(C58). `complex_func=True`면 실수부의 `quad`를 마친 뒤 허수부의 `quad`가 같은 점들을 다시 부른다(C56).
- **`dblquad`·`tplquad`·`nquad`**: 안쪽 적분을 바깥 점마다 `quad`로 계산한다(scipy와 같은 겹침, 같은 순서; 겹침은 C++에서, C64). `gfun`·`hfun`(`qfun`·`rfun`)은 수 또는 바깥 변수의 함수이고, 가장 안쪽 f는 모든 변수를 안쪽부터 Python `float`로 받는다.
- **`quad_vec`**: 점 하나마다 Python `float`로 부르고 수 하나 또는 같은 모양의 배열(`sd.ndarray`, 시퀀스, numpy 배열)을 받는다(첫 값이 꼴을 정한다, C104). 순서는 scipy의 것(규칙의 점을 큰 것부터, 구간은 힙에서 꺼낸 차례)이고 점의 비트까지 같다.
- **`cubature`**: scipy처럼 영역마다 두 번, 점들을 한 번에 (n_points, ndim) float64 배열로 부르고(`genz-malik`은 Fortran 순서, C100) (n_points, …)를 받는다.
- **원소별 무리**: §2.2. 사용자 함수는 원소별이어야 한다(`f(x)[i]`가 `f(x[i])`와 같다; scipy의 계약). 구간의 끝 바로 옆을 계산하므로 특이점에서 예외를 내지 말고 inf·NaN을 돌려줘야 한다: scipy `tanhsinh`은 반올림으로 끝점 자체(예: x = 0)를 부르고 그 값을 버린다(§7.4). 이것을 문서의 계약으로 적는다.
- `jacobian`·`hessian`의 f는 (m, …)을 받아 (k, …)을 돌려주는 벡터화된 함수다(scipy의 계약: 첫 축이 변수, 뒤 축이 점). 첫 호출부터 (m, m)·(m, m, p)를 받는다(§7.4). sd에서는 (m, N)을 받아 (k, N)이나 (N,)을 돌려준다(C27; 바이트는 scipy와 같다).
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
- **호출점의 비트** (C5, C62): 규칙의 상수(xgk, wgk, wg)는 QUADPACK의 10진 리터럴 그대로다. scipy 1.18.1의 QUADPACK은 scipy가 Fortran에서 옮긴 C 판이고 clang의 기본 FP 축약(`-ffp-contract=on`: 한 식 안의 곱과 합만 합친다)으로 빌드된다. 원본을 문장마다 옮기면 합쳐지는 곳은 Kronrod 규칙의 누적 넷(resg, resk, resabs, resasc의 s = s + w·v)뿐이고, 그 자리에 `std::fma(w, v, s)`를 적었다. 점(c ∓ h·x)은 원본에서 `absc = hlgth*xgk(j)`와 `f(centr-absc)`의 두 문장이라 합쳐지지 않고, 이분점 0.5(a + b), QAGI의 boun + dinf·(1 − t)/t(나눗셈이 마지막), 구동 루틴과 ε-알고리즘(`dqelg`)에는 한 식 안에서 곱과 합이 만나는 곳이 없거나(`|a2| + 1000·uflow`의 곱은 정확하다) 결과가 같다. 그래서 가중치 없는 106개 모두 scipy와 점·값·오차·`nfev`·status·글·구간이 비트로 같다(§7.10). brentq(optimize §3.2)와 같은 방법이다.
- **가중치 루틴** (C5, C71–C83): QAWO·QAWF·QAWS·QAWC와 규칙 `dqk15w`(16자리 리터럴, `dqk15i`의 것과 다르다)·`dqc25f`·`dqc25s`·`dqc25c`, `dqcheb`, `dqmomo`, `dgtsl`을 같은 방법으로 옮겼다. 축약의 자리는 C81이다. scipy의 C 판이 원본과 다른 곳은 scipy를 따르고(`dqc25s`의 오타를 고친 식 C75, `dgtsl`의 꼴 C76, `dqawoe`의 작은 구간 판정 C79), scipy가 틀린 값을 내는 곳은 원본의 뜻을 따른다(모멘트 행 C77, QAWF의 ω = 0 C78, 큰 ω C82). (−∞, b]의 cos·sin은 scipy처럼 x → −x로 [−b, ∞)의 f(−x)·−f(−x)를 적분하고 f는 −x에서 부른다. 그래서 모음의 가중치 10개와 스윕 99개, 만든 400개 실행이 scipy와 점·값·오차·`nfev`·status·글·구간(QAWF는 주기)이 비트로 같았다(§7.11). `dqc25f`의 추정에 반올림 바닥을 둔 뒤(C123)로는 QAWO·QAWF의 오차가 scipy 이상이고, 바닥이 경로를 바꾸는 하나(만든 400개의 `99-cos3-cos`)는 점과 status도 다르다. 나머지(QAWS·QAWC, 그리고 바닥이 판정을 바꾸지 않는 QAWO·QAWF의 점·값·`nfev`·status·글)는 그대로 비트로 같다.
- 오차 추정은 QUADPACK의 것(`abserr`: |K21 − G10|을 (200·|K − G|/resasc)^1.5로 조이고 반올림 하한 50ε·resabs)이다. 정직함의 측정은 §7.4다: 대개 실제 오차보다 몇 자리 크지만, 좁은 봉우리를 처음 21점이 놓치면 틀린 값을 수렴으로 알린다(4⁻ᵅ/((x − π/4)² + 16⁻ᵅ), α = 19·20에서 값 ≈ 0, 참값 π, `ier = 0`). 이것은 scipy와 같게 두고 문서에 `points`로 피하는 법을 적는다. 봉우리가 아주 좁으면(같은 봉우리의 α = 19: 너비 3.6×10⁻¹², 구간의 2⁻³⁸) `points=[π/4]`, 봉우리 둘레의 작은 구간, `limit=200`도 status 5로 끝나고 `tanhsinh`은 봉우리에서 나눠 불러도 status −2다. 변수를 바꿔 봉우리를 넓히면(x = π/4 + 4⁻ᵅ sinh t: 피적분함수 1/cosh t) `quad`가 참값과 1.3×10⁻¹⁵로 수렴한다. API 문서 §1.4에 그 방법을 적는다.

### 3.2 `dblquad`, `tplquad`, `nquad`
- scipy의 `nquad`(`_NQuad`)처럼 가장 안쪽 변수부터 `quad`를 겹친다. 층마다 `opts`(`atol`·`rtol`·`limit`·`points`·`weight`·`wvar`; C11)를 받는다. `dblquad(func, a, b, gfun, hfun)`의 `func`는 scipy처럼 `func(y, x)`(안쪽 변수가 먼저)다.
- 돌려주는 오차는 scipy `_NQuad`의 `self.abserr`처럼 모든 층의 모든 `quad`가 낸 추정의 최댓값이다(안쪽 적분의 오차를 바깥에 더하지 않으므로 엄밀한 상계가 아니다; C63). 설계 초안의 "바깥 `quad`의 추정뿐"은 scipy 코드를 잘못 읽은 것이었다. 이 점을 API 문서에 적는다.
- `nfev`는 가장 안쪽 `quad`들의 `nfev` 합, 곧 f를 부른 수다(scipy `nquad(full_output=True)`의 `neval`). `converged`·`status`·`message`는 C63이다. 탐침: 단위 정사각형의 exp(−(x² + y²))는 441번(21 × 21), 0.09 ms다.

### 3.3 `quad_vec`
- scipy의 Python 판(`_quad_vec.py`)을 옮겼다(C7, `csrc/integrate/adaptive.cpp`): 전역 적응, 한 바퀴에 오차가 큰 구간을 128개까지(그 오차의 합이 전체 오차 − tol/8을 넘을 때까지) 힙에서 꺼내 반으로 나눔, 규칙 `'gk21'`(유한 기본)·`'gk15'`(무한 기본)·`'trapezoid'`, 무한 구간은 scipy의 변수 치환((0, 1] 또는 [−1, 1]과 t = 0의 나눔), 멈춤은 구간이 둘 이상일 때 `norm`(`'max'`·`'2'`)으로 잰 오차 < max(`atol`, `rtol`·‖I‖)/8(0), 오차 < 반올림 오차(2), 유한하지 않은 오차(3), 구간 수 ≥ `limit`(1).
  - 전체 값은 구간 값의 합을 다시 내지 않고 나눌 때마다 (s₁ + s₂) − s_old를 더한다(scipy와 같다). 오차 추정은 QUADPACK의 꼴을 노름으로 옮긴 것(err = dabs·min(1, (200 err/dabs)^1.5), 반올림 하한 50ε·h·∫|f|)이다.
  - 산술(C97), 구간 기억(C98), 필드(C95), 인자(C104)는 결정 기록에 있다. scipy의 `cache_size`(인자), `workers`(다중 프로세스), `full_output`은 없다(C96; 그 값은 필드다).

### 3.4 `cubature`
- scipy의 Python 판(`_cubature.py`)을 옮긴다: 영역의 힙, 오차가 가장 큰 영역을 축마다 반으로 나눔(2ⁿ개), 규칙 `'gk21'`·`'gk15'`(1차원 Gauss–Kronrod의 곱: 점 21ⁿ·15ⁿ개), `'genz-malik'`(차수 7, 점 2ⁿ + 2n² + 2n + 1개; n ≥ 2). 무한 축은 scipy의 치환(x = (1 − |t|)/t, x = a + (1 − t)/t). `points`는 그 점을 지나게 영역을 처음에 나눈다.
- 이름 `'gauss-kronrod'`는 `'gk21'`의 별칭이라 받지 않는다(별칭 없음). 사용자 규칙 객체(scipy `_rules.Rule`), `workers`는 없다(C96).
- 옮긴 것(C7, `csrc/integrate/adaptive.cpp`): 영역마다 규칙의 점으로 한 번(값), 두 규칙의 점으로 한 번(오차 = |Σ w·f − Σ w_low·f|, 원소마다) f를 부른다. 점은 (x + 1)·(길이·0.5) + a, 가중치는 w·(Π 길이/2ⁿ)(곱은 차례대로)이고 Genz–Malik의 점·가중치는 scipy의 순서(`_distinct_permutations`, `itertools.product`)와 Python의 산술(정수의 나눗셈은 올바로 반올림)이다. 처음 영역들은 힙으로 만들지 않은 목록이다(scipy가 그대로 꺼낸다). 뒤집은 축의 부호, 무한 축의 치환과 야코비안 1/Π t², 치환이 만든 원점의 나눔은 scipy와 같고 다른 셋은 C101–C103이다. 낮은 규칙(C99), 배치와 합(C100), 인자(C105)는 결정 기록에 있다.
- 탐침(§7.4): 2차원 가우스를 gk21로는 첫 영역에서 끝내고(441점), genz-malik은 106번 2,491점이다.

### 3.5 `tanhsinh`
scipy의 `_tanhsinh.py`(Bailey–Jeyabalan–Li 2005, Vanherck–Sorée–Magnus 2020)를 원소마다 옮긴다.
- **변환**: 끝을 정리한다(a = b는 0, b < a는 바꿔 계산하고 부호를 바꿈, 양쪽 무한은 x = t/(1 − t²), 한쪽 무한은 x = 1/t − 1 + a, 왼쪽 무한은 x → −x). 가중치는 그 야코비안을 곱한다.
- **점과 가중치**: 기본 간격 h₀ = asinh(log(2/(4·tiny) − 1)/π)/8(xjc가 아래넘침하는 tmax의 1/8), 수준 k의 간격 h₀/2ᵏ, 수준 0은 j = 0…8, 다음 수준부터 홀수 j만. u₁ = (π/2)cosh(jh), u₂ = (π/2)sinh(jh), w = u₁/cosh(u₂)², 1 − x = 1/(exp(u₂)cosh(u₂)). 끝에 닿은 점(반올림)은 가중치 0이다.
- **호출점의 비트** (C5): scipy의 표는 numpy의 `cosh`·`sinh`·`exp`로 계산하는데 이 점들에서 macOS libm과 같다. 그래서 C++에서 `std::cosh`·`std::sinh`·`std::exp`(libm)와 곱 c·c(numpy의 `**2`)로 계산하면 수준 0–10의 16,386값이 모두 비트까지 같다(§7.4, §7.12; Python의 `x**2`를 쓰면 11값이 다르다; C84). sd의 원소별 `exp`·`cosh`·`sinh`(vForce)는 이 점의 약 3분의 1에서 libm과 다르므로 쓰지 않는다. 표는 처음 쓸 때 한 번 만들어 둔다.
  - 이 표 자체는 정확하지 않다: 같은 식의 참값과 견주면 최대 980 ULP, 평균 47 ULP다(u₂의 반올림이 exp(u₂)에서 u₂배로 커진다). 적분값에는 거의 영향이 없지만(꼬리의 가중치가 작다) scipy의 표를 그대로 쓴다(C84). 정확한 j·h에서 재면 최대 3,449 ULP다(§7.12).
- **합과 오차 추정**: 수준마다 Euler–Maclaurin 합 Sₙ = h·Σ f_j w_j(이전 수준은 Sₙ₋₁/2를 더함)과 scipy §5의 추정: d1 = |Sₙ − Sₙ₋₁|, d2 = |Sₙ − Sₙ₋₂|, d3 = ε·max|f_j w_j|, d4 = 가장 바깥 항의 크기, e = clip(max(d1^(log d1/log d2), d1², d3, d4), ε|Sₙ|, d1). 멈춤은 e < atol 또는 e < rtol·|Sₙ|(기본 rtol = ε^0.75, atol = 0), 수준 상한 10(`maxlevel`), 시작 수준 2(`minlevel`: 처음 한 번의 호출에 수준 0–2의 점을 모두 넣는다).
  - scipy의 Σ는 numpy의 pairwise 합(한 줄을 따라: 8개 누산기, 128개 단위)이다. 이 합이 멈춤 판정을 정하므로 호출점의 비트를 위해 이 자리에서는 numpy의 합 순서를 그대로 쓴다(줄의 길이가 수준 10에서 16,384 이하라 비용이 작다). 다른 리덕션(sd의 16누산기)을 쓰면 판정의 경계에서 수준이 하나 달라질 수 있다. C6에서 그렇게 했다(C87; 끝점의 계산과 오차 추정의 clip·max도 numpy의 NaN 규칙 그대로).
  - 유한하지 않은 f_j는 가장 가까운 유한한 끝의 값으로 바꾼다(scipy, van Engelen의 제안). Sₙ이 유한하지 않으면 status −3이다.
- `log=True`: f가 log(피적분함수)를 돌려주고 합은 scipy의 logsumexp다(C87). 값은 실수뿐이라 피적분함수는 음이 아니어야 하고 b < a는 `ValueError`다(C85).

### 3.6 `nsum`
scipy의 `nsum`을 옮긴다: 항 수가 `maxterms` 이하면 직접 합, 아니면 적분 판정의 경계. ∫ₐ^∞ f를 `tanhsinh`으로 구해 하한으로 삼고, f(c) < atol + rtol·S_l인 c를 2의 거듭제곱 걸음으로 찾고(없으면 c = a + maxterms·step), 합 = Σ_{k<c} f(k) + ∫_c^∞ f + f(c)/2, 오차 = f(c)/2 + 적분의 오차. 왼쪽 무한은 x → −x, 양쪽 무한은 f(x) + f(−x)(x = 0의 가중치 1/2)의 오른쪽 합이다. 기본 rtol = ε^0.5. f는 매끄럽고 양이고 단봉이어야 한다(scipy의 계약). scipy 1.18.1의 코드는 단조 감소를 가정한 경계이고, 단봉 함수의 봉우리를 따로 찾지 않는다(문서의 "봉우리 둘레를 직접 더함"은 직접 합이 덮을 때만이다).
- 직접 합은 scipy처럼 점들을 한 번에 부르고(`(1, 8192)` 같은 큰 호출) 합의 순서는 numpy의 것이다(§3.5와 같은 까닭, C87). 갈래와 호출의 순서는 C91, scipy와 다른 두 곳은 C92다.
- 탐침: Σ1/k²는 10번 호출, `nfev` 8,561, 실제 오차 3×10⁻¹³, 추정 7.4×10⁻⁹였다.

### 3.7 표본의 규칙과 `fixed_quad`, `newton_cotes` (C3에서 했다)
- **`trapezoid`, `simpson`, `romb`, `cumulative_trapezoid`, `cumulative_simpson`**: 식과 연산의 순서는 scipy 1.18.1의 것(`integrate/_quadrature.py`)이다. `trapezoid`의 항은 d·(y[i+1] + y[i])/2.0, `simpson`은 scipy 1.11부터의 꼴(점이 홀수 개면 `_basic_simpson`, 짝수 개면 앞 구간들의 Simpson과 마지막 구간의 Cartwright 보정 α·y[−1] + β·y[−2] − η·y[−3], 점 둘이면 사다리꼴; C47), `romb`은 scipy의 Richardson 표(R(i, 0) = 0.5·(R(i − 1, 0) + h·Σ), R(i, j) = R(i, j − 1) + (R(i, j − 1) − R(i − 1, j − 1))/(4ʲ − 1)), `cumulative_simpson`은 구간마다 Cartwright의 식(짝수 번째 구간은 그 구간과 다음 점의 왼쪽 식, 홀수 번째와 마지막은 앞 점의 오른쪽 식; scipy의 `h1`·`h2` 부분적분)이다. `simpson`의 0 간격은 scipy의 `apply_where`처럼 그 몫을 0으로 둔다(C50).
  - **합** (C51): 줄마다 항을 sd의 pairwise 합(`base/pairwise.hpp`: 누산기 16개, 512개 단위; `sd.sum`이 연속 축에서 하는 것)으로 첨자 순서대로 더하고, 누적은 `cumulative_sum`의 스캔(`scan_lane`: 8,192개부터 4,096개 블록의 두 단계)이다. 그래서 scipy·numpy와 끝자리가 다를 수 있고(D-33), 줄의 값은 축·메모리 배치·스레드 수와 상관없다. 줄이 2개 이상이고 일이 2¹⁵ 이상이면 줄을 스레드에 나누고, 줄 하나가 길면 같은 나무의 부분합을 나눈다(`pairwise_sum_mt`). 일이 2¹⁵ 이상이면 GIL을 놓는다.
  - dtype과 입력 (C39, C45, C48): y는 float64 또는 complex128(복소수는 실수 계수를 부분마다 곱한다), x는 실수, `dx`·`initial`은 수 또는 줄마다의 배열이다. `trapezoid`는 numpy의 `np.trapezoid`와 같은 함수다(scipy의 것이 numpy의 것과 비트까지 같다, §7.4): sd 배열 쪽에는 `trapezoid`가 없으므로 `sd.integrate.trapezoid` 하나만 둔다.
  - 파일: Python `src/stardust/integrate/_samples.py`(인자 검사와 변환), C++ `csrc/integrate/samples.cpp`(줄마다의 규칙; `stardust._core.integrate.samples`).
- **`fixed_quad(func, a, b, /, *, n=5)`** (C42, C43, C52): n점 Gauss–Legendre. 점은 scipy의 식 (b − a)·(x + 1)/2.0 + a이고 f를 (n,) 배열로 한 번 부른다. 값은 (b − a)/2.0·Σ wᵢfᵢ이고 합은 표본 규칙의 커널(같은 pairwise 합)이다. f가 (…, n)을 돌려주면 결과는 (…) 배열이다(scipy의 벡터 값).
  - **점과 가중치** (C42, C54; `src/stardust/integrate/_gauss.py`, `csrc/integrate/gauss.cpp`): 올바로 반올림한 double이다. n ≤ 100은 Arb의 `arb.legendre_p_root`를 2⁻¹¹⁰까지 정해 반올림한 표(`tools/gen_gauss_legendre.py` → `csrc/integrate/gauss_table.hpp`; 반올림 경계를 품은 공이 있으면 도구가 멈춘다)이고 음이 아닌 점 ⌈n/2⌉개만 둔다(나머지는 부호를 바꾼 것). n > 100은 점마다 점근 근사 cos(π(i − 1/4)/(n + 1/2))에서 double Newton, 이어 double-double Newton 두 번(P_n·P_{n−1}은 세 항 점화식, P_n′ = n(xP_n − P_{n−1})/(x² − 1))과 가중치 2/((1 − x²)P_n′²)를 double-double로 계산하고, Python이 반올림한다. double-double이 반올림 경계에서 오차 한계(점 n²·2⁻⁵⁷ ulp, 가중치 n²·2⁻¹⁰⁰ 상대) 안이면 정확한 정수 산술로 정한다: 점은 두 반올림 경계(dyadic 유리수)에서 P_n의 부호(점화식을 p_k = k!·2^{Ek}·P_k로 바꾼 정수 점화식 p_{k+1} = (2k + 1)M p_k − k²4^E p_{k−1}), 가중치는 근을 감싼 구간 [a, b]에서 g = 1/w = (1 − x²)P_n′²/2의 둘레 g(a) ± L(b − a)(L = (3/4)(n(n + 1))², 르장드르 다항식의 한계 |P_n′| ≤ P_n′(1) = n(n + 1)/2에서. 마르코프 부등식이면 n²이다)이 한 double로 반올림될 때까지 구간을 반으로 줄인다. 이 길은 테스트가 모든 값을 강제로 보내 확인한다(§7.9).
  - scipy의 `roots_legendre`(모든 n에서 Golub–Welsch: 삼중대각 행렬의 고윳값과 Newton 보정, 가중치의 정규화)는 n = 5에서 점이 같고 가중치가 17 ULP까지, n = 100에서 점 8·가중치 83,682 ULP까지 다르다(§7.4). 그래서 n ≥ 20의 호출점이 scipy와 다르다(C5의 예외).
- **`newton_cotes(rn, /, *, equal=False)`** (C43, C53): 닫힌 Newton–Cotes 가중치 aᵢ = ∫₀ᴺ Lᵢ(s) ds(Lᵢ는 위치 rᵢ의 라그랑주 기저; W(s) = Π(s − rⱼ)를 (s − rᵢ)로 나눈 몫의 정확한 적분)와 오차 계수 B = (N^{p+1}/(p + 1) − Σ aᵢrᵢᵖ)/p!(p는 scipy의 차수: 같은 간격의 짝수 N이면 N + 2, 아니면 N + 1)를 `fractions.Fraction`으로 정확히 계산해 한 번씩 반올림한다(`src/stardust/integrate/_rules.py`). scipy의 같은 간격 N ≤ 14 표도 정확한 유리수를 한 번 반올림한 것이라 비트로 같다. 같은 입력은 기억한다(N = 8의 첫 계산 0.2 ms).

### 3.8 `derivative`, `jacobian`, `hessian`
scipy의 `_differentiate.py`를 원소마다 옮긴다.
- **방법**: Richardson 외삽이 아니다(scipy의 주석이 그것을 "나중 일"로 적는다). order차(기본 8, 홀수는 올림) 차분 공식을 간격을 step_factor(c, 기본 2)씩 줄이며 되풀이한다. 중앙 스텐실은 x ± h/cᵏ(k = 0…n − 1, n = order/2)이고 다음 반복은 x ± h/cⁿ 두 점만 새로 부른다. 한쪽 스텐실(`step_direction` ≠ 0)은 d = √c로 x + h/dᵏ(k = 0…2n − 1), 다음 반복에 두 점이다. 가중치는 Vandermonde 계를 풀어 얻는다(scipy는 `np.linalg.solve`; 중앙 가중치는 반대칭을 강제한다).
- **멈춤**: 오차 = |dfₙ − dfₙ₋₁|, error < atol + rtol·|df|이면 0(기본 atol = 가장 작은 정규수, rtol = √ε), 유한하지 않으면 −3, 오차가 직전의 10배를 넘으면 −1(반올림이 이기기 시작함), `maxiter`(기본 10)면 −2.
- **호출점의 비트** (C5): 점은 원소별 식(x + h/cᵏ)이라 scipy와 같게 계산한다. scipy의 식을 그대로 옮겼다: 첫 반복의 cᵏ·dᵏ은 numpy의 `c ** arange`(libm `pow`), 다음 반복의 c^(n−1)과 d = c**0.5는 Python의 `pow`, 간격은 반복마다 h /= c, 점은 x + (−h/cᵏ)·x ± h/dᵏ. 다만 df는 scipy에서 `fc @ wc`(numpy의 행렬·벡터 곱 = BLAS dgemv, 지금 Accelerate)라 합의 순서가 scipy 쪽의 BLAS에 달려 있고, 멈춤 판정의 경계에서 반복 수가 갈릴 수 있다. 갈리는 비율은 §7.8이다.
- **df의 합** (C2, C38): 중앙 스텐실은 마디 −t, t의 짝마다 w·(f(x − ht) − f(x + ht))를 바깥 짝부터 더하고, 한쪽 스텐실은 스텐실의 순서대로 곱하고 더한다(w₀f₀에서 시작해 한 항씩). 축약은 없다(D-33). 원소마다 따로 계산하므로 원소의 비트는 배치의 크기·스레드 수와 상관없다(scipy는 배치의 크기에 따라 끝자리가 다르다, §7.7). 중앙 스텐실의 가운데 가중치는 0이고, 첫 반복에서만 f(x)가 0·f(x)로 들어간다(f(x)가 유한하지 않으면 NaN, scipy와 같다). scipy는 다음 반복부터 그 칸에 앞서 쓴 왼쪽 점의 값을 두는데 그 값은 유한하거나(아니면 원소가 이미 −3으로 멈췄다) 가중치 0이라 셈에서 뺐다.
- **가중치의 정확도** (C28, C34): scipy의 order 8 중앙 가중치는 정확한 유리수 값과 9개 가운데 8개가 다르고 최대 상대 2.7×10⁻¹⁴다(order 12면 1.7×10⁻¹¹; §7.4). 한쪽 가중치는 9개 모두 다르고 최대 상대 1.1×10⁻¹⁰이다(order 12면 1.5×10⁻⁴). sd의 가중치는 마디 tᵢ의 라그랑주 기저의 도함수 Lᵢ′(0)(scipy의 Vandermonde 계의 정확한 해)를 한 번 반올림한 것이다: 중앙은 `fractions.Fraction`의 정확한 유리수(가운데는 정확히 0, 나머지는 정확히 반대칭), 한쪽은 Q(√c)의 정확한 값을 올바로 반올림한다(`src/stardust/differentiate/_weights.py`). order 8은 0.5 ms 안에 계산되고 (n, c)마다 기억한다. 중앙과 한쪽 가중치는 따로 기억하고, 한쪽 가중치(Q(√c)의 계산이라 처음 비용의 대부분: n = 50, c = 2에서 162 ms 가운데 143 ms)는 어느 원소가 한쪽 스텐실을 쓸 때만 계산한다(커널에는 빈 배열을 넘긴다). 가중치는 점에 영향을 주지 않으므로 호출점은 그대로다. `order`는 18 이하다(C118).
  - 정확한 가중치와 짝의 합의 효과 하나: x³의 0에서 중앙 스텐실의 합이 매번 같은 작은 값이라 오차 추정이 0이 되어 둘째 반복에 수렴한다(status 0, `nfev` 11). scipy는 가중치와 합의 반올림 흔들림(10⁻¹⁴) 때문에 기본 atol로 끝내 멈추지 못한다(−2, `nfev` 27; `[scipy와 다름]`). 짝을 안쪽부터 더하면 sd도 −2라, 이것은 반올림이 가르는 경계의 경우다.
- **`jacobian(f, x)`**: x (m,)에서 변수 j만 바꾼 열을 쌓아 f를 한 번에 부른다. 첫 호출은 열 m개가 모두 x인 (m, m)이고(scipy의 `eim._initialize`), 반복마다 (m, m·p)다(C27; 열 j·p + q는 변수 j의 q번째 점, 다른 변수는 x). scipy처럼 모든 항목이 멈출 때까지 모든 열을 부르고(C33), 항목(출력 i, 변수 j)마다 따로 판정한다. 결과 `df` (k, m)(스칼라 f는 (m,)), 항목마다의 `status`·`n_iter`·`nfev`. `nfev`는 그 항목이 멈출 때까지 부른 점의 수(열 하나의 몫: 1 + 2n + 2·(반복 − 1))다.
- **`hessian`**: `jacobian`의 `jacobian`이다. 바깥 `jacobian`의 함수는 그 호출의 열 N개마다의 기울기이고, 안쪽 `jacobian`(rtol/100, 같은 atol·`max_iter`·`order`·`initial_step`·`step_factor`, 중앙)이 그 N개 점을 한 번에 계산한다(안쪽 f는 (m, m·N·p)을 받는다). rtol은 0이거나 100ε 이상이다(C29). `nfev`는 scipy의 셈이다: 항목 (i, j)는 바깥 호출마다 안쪽 항목 i의 변수 j 열들(점 p개)의 `nfev` 합을 그 항목이 멈출 때까지 더한다(Rosenbrock n = 3에서 121–297; §7.4). `n_iter`와 `x`는 없다(scipy와 같다).
- scipy의 기본 `initial_step = 0.5`는 절대 간격이라 x의 척도를 보지 않는다: log의 x = 10⁻³에서 x − 0.5 < 0이 NaN이라 status −3이다(§7.4). scipy 그대로 두고(기본값은 scipy의 것) 문서에 `initial_step`·`step_direction`을 적는다.

---

## 4. 정확도 계약 (안)
- **적응 방법**(`quad` 계열, `quad_vec`, `cubature`, `tanhsinh`, `nsum`): 결과는 "멈춤 판정을 통과한 값"이고, 오차 추정은 엄밀한 상계가 아니다(QUADPACK, tanh-sinh 모두 경험적이다). 그래서 수치 계약은 (1) scipy와 같은 걸음(§7.3의 기록과 호출 순서·`nfev`·`status`가 같다)과 (2) 시험 모음에서 잰 정확도와 정직함의 표다. 표는 `bench/<영역>/accuracy.md`로 만들고 기준선과 견준다(optimize의 "결과로 확인되는 판정 + 기준선 비교"와 같다).
  - **정직함**: ρ = 오차 추정 / |참 오차|(Arb). converged이고 ρ < 1인 경우를 세어 표에 적는다. 첫 탐침(§7.4)에서 `quad`는 QUADPACK 책의 세 무리 62경우 가운데 수렴이라고 알린 것 가운데 둘(봉우리, α = 19·20)이 ρ < 1이고, `tanhsinh`은 수렴한 46경우 가운데 여섯이 ρ < 1(최소 0.015)이었다.
  - C4에서 잰 것(§7.10): `quad`는 scipy와 같은 걸음(가중치 없는 106개 모두 비트까지)이고, 수렴한 94개 가운데 참 오차가 오차 추정 밖인 것은 scipy와 같은 셋이다. 이 셋을 뺀 "수렴이면 참 오차 ≤ 오차 추정"이 정확도 스윕의 계약이다(판정의 뜻이지 증명된 상계가 아니다). 겹친 적분도 같다(오차는 모든 `quad`의 최댓값, C63).
- C6에서 잰 것(§7.12): `tanhsinh`·`nsum`은 scipy와 같은 걸음(기록과 기록 밖의 모든 경우가 비트까지)이고, `tanhsinh`이 수렴한 76개 가운데 참 오차가 오차 추정 밖인 것은 scipy와 같은 여덟이다(`TANHSINH_SHORT`). 이 여덟을 뺀 "수렴이면 참 오차 ≤ 오차 추정"이 정확도 스윕의 계약이고(한계의 0.78까지), `nsum`은 급수 아홉 모두 그 안이다(0.89까지). 모음 밖에서는 오차 추정이 더 자주 짧다(§7.12의 e^(−cx²)).
- C7에서 잰 것(§7.13): `quad_vec`·`cubature`는 scipy와 같은 걸음(기록과 기록 밖, 모음의 모든 경우가 비트까지; `cubature`의 다른 셋 C101–C103 빼고)이다. 계약은 C83의 꼴인 "수렴이면 참 오차 ≤ 오차 추정 + 8ε|I|"이고(`cubature`의 |K − G| 추정에는 반올림 하한이 없다), 넘는 것은 scipy도 같은 짧은 목록(C107)이다. 계약 대비 최대는 `quad_vec` 0.95, `cubature` 0.96이다.
- **표본 규칙**: 같은 double 입력에 그 규칙(식)을 정확히 계산한 값과 견준다(식이 유리 연산뿐이라 참값은 `fractions.Fraction`의 정확한 유리수다, `tests/integrate/ref/samples.json`). 구적 자체의 이산화 오차는 계약이 아니다.
  - C3에서 정한 계약(`bench/accuracy/integrate.py`, `tests/integrate/test_samples.py`): |값 − 참값| ≤ c·ε·S. S는 항의 크기 Σ|dᵢ|(|yᵢ| + |yᵢ₊₁|)(누적은 k번째까지), c는 합이면 ⌈log₂ n⌉ + 10(pairwise 나무), 누적의 k번째 값이면 k + 10(스캔)이다. 모음의 12벌 모두에서 지키고 가장 큰 것은 한계의 0.050이다(§7.9).
- **`fixed_quad`, `newton_cotes`** (C3): 점·가중치는 올바로 반올림한 값이고 그것이 계약이다(0 ULP; n ≤ 100은 모두 Arb와, n = 101–1000의 일곱 n은 `tests/integrate/ref/gauss.json`과 비트로 같다). `fixed_quad`의 값은 f가 돌려준 값으로 낸 (b − a)/2·Σwᵢfᵢ의 정확한 유리수와 (⌈log₂ n⌉ + 4)·ε·|(b − a)/2|·Σ|wᵢfᵢ| 안이다(테스트). `newton_cotes`는 정확한 값을 한 번 반올림한 것이다(다른 알고리즘, 유리수의 모멘트 방정식 소거와 테스트가 견준다).
- **`derivative` 무리**: 수렴한 결과는 error < atol + rtol·|df|이고, 해석적 도함수의 Arb 값과 견준 상대오차를 표에 적는다(탐침: exp·sin의 1에서 1.2×10⁻¹⁴·9×10⁻¹⁵).
  - C2에서 정한 계약(`bench/accuracy/differentiate.py`, 테스트 `test_derivative.py`·`test_jacobian.py`): `derivative`는 수렴한 결과의 참 오차가 atol + rtol·|df| 이하다(모음 전부에서 지킨다; 판정의 뜻이지 증명된 상계는 아니다). `jacobian`·`hessian`은 노름으로: 수렴한 항목의 오차가 rtol·max|참 행렬| 이하다. 항목마다의 상대 판정은 0에 가까운 항목에서 뜻이 없어 계약이 아니다(§7.8).
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

### 6.1 차분 간격 커널 옮기기 (C3, C15; C1에서 했다)
- 전: 커밋 1bc8c2b의 `csrc/optimize/diff.hpp`(`rel_step_2point`, `rel_step_3point`, `absolute_step`, `adjust_one_sided`, `adjust_two_sided`)를 `csrc/optimize/lsq.cpp`(least_squares·nls의 야코비안), `csrc/optimize/objective.hpp`(minimize의 차분 기울기), `csrc/optimize/hessian.hpp`(차분 헤세 곱), `csrc/optimize/scalar.cpp`(차분 newton)가 썼다.
- C1: 이 헤더를 `csrc/differentiate/steps.hpp`(이름공간 `sd::differentiate`, C23)로 옮겼고, 네 파일은 include와 이름(`differentiate::absolute_step` 등)만 바뀌었다. 차분의 고리(호출 수 세기, 한쪽이 유한하지 않을 때 반대쪽 다시 해 보기, 기억)는 부르는 쪽마다 다르므로 그 자리에 두었다. 헤더뿐이라 CMake는 그대로다.
- Python 쪽은 `src/stardust/optimize/_diff.py`의 `Differences`(nls의 정확한 5점 차분, `approx_fprime`의 기본)와 그 상수를 `src/stardust/differentiate/_differences.py`로 옮겼다(C15). `approx_fprime`·`check_grad`(`sd.optimize`의 공개 이름 그대로)와 값 검사·에러의 고리는 `_diff.py`에 남았고, `regression.nls`와 `_diff.py`가 `Differences`를 부른다. 공개 이름은 아니다(`sd.differentiate.__all__`은 그대로).
- 확인: 식과 계산 순서가 같아 골든이 모든 영역에서 1·3·10스레드로 그대로다(optimize 3,876, regression 902케이스). optimize·regression·integrate·differentiate의 테스트(scipy와 비트까지 같은 걸음, O9)가 모두 지난다. `tests/differentiate/test_differences.py`가 커널이 하나인지와 R175·R176의 것(아래)을 본다.
- R175(C25): `_ROUNDING_TOL = 2⁻²⁶`과 `_forward`의 간격을 √u라 부른 주석은 틀렸다. 같은 파일의 u = 2⁻⁵³이라 √u = 2^−26.5이고, 2⁻²⁶은 √ε(ε = 2⁻⁵² = 2u)다. 주석만 고쳤다(값은 그대로; docs/design/optimize.md의 O104 설명은 처음부터 √ε이다).
- R176(C25): 잰 호출 수(f(x₀) 한 번을 넣어; exp): x = 0은 17, 10⁻³⁰⁰·10⁻¹⁰⁰·10⁻³⁰은 25, 10⁻¹²·10⁻⁸·10⁻⁶은 61, 10⁻⁵–10⁻²는 13, 1은 17번이다. 10⁻¹²에서 처음 간격 2⁻⁴⁷의 열은 반올림 한계가 열 크기의 0.023(√ε의 2²¹배)이라 간격을 2⁻²⁶으로 한 번에 키우고, 거기서 절단 오차가 반올림을 넘는 2⁻⁸ 근처까지 4배씩 아홉 번(호출 36번) 더 오른다. 0 근처에서 값이 작은 함수(sin, atan, tanh, log1p)는 열이 반올림 수준이 아니라 9–25번이다. 2의 거듭제곱을 더 크게 뛰거나 이분으로 찾으면 고른 간격이 달라져 값과 nls의 골든이 바뀌므로 고치지 않았다. 함수값이 기울기보다 훨씬 큰 x ≈ 0이 드문 경우이고 상한이 있다(간격 키우기는 2⁻⁷에서 멈춘다).
  - 같이 본 것: 처음 간격이 반올림 한계 √ε 바로 아래인 x(exp의 10⁻⁵·10⁻⁴는 13번)는 간격을 키우지 않고 아래로만 찾아 상대오차가 5.1×10⁻⁹·2.6×10⁻¹⁰이다(10⁻¹²·10⁻⁶은 키워서 2×10⁻¹³, 10⁻²는 5.5×10⁻¹²; sin(x + 1)의 10⁻⁵는 4.3×10⁻⁹). cos의 10⁻⁹(기울기 −10⁻⁹)는 절대오차 10⁻⁹다. 문서(docs/api/optimize.md §2)의 "10⁻¹²·10⁻³⁰⁰에서도 0과 같은 정확도"는 그 점에서는 맞지만 그 사이의 10⁻⁵ 근처는 덮지 않는다. 바꾸면 골든이 바뀌므로 C1에서는 재기만 했다(따로 알린다). C8에서 고쳤다(C112: 반올림 한계가 2⁻³⁶을 넘는 시작도 간격을 키운다).
- ARCHITECTURE.md §1의 그림에 optimize/ ──▶ differentiate/를 더했다(differentiate/는 아무것도 include하지 않는 헤더 하나다).

### 6.2 함께 쓰는 반복 고리 (C4, C16; C1에서 했다)
- 전: `csrc/optimize/batch.cpp` 안에 고리의 부품이 있었다: `for_elements`(2¹⁵개부터 스레드에 나눔, GIL 놓기), `gather`(남은 원소의 `args` 모으기), `Call`(남은 원소의 점을 새 float64 `sd.ndarray`에 써서 사용자 함수를 한 번 부르고 값을 검사), `Progress`(진행 중인 원소의 목록과 멈춘 원소의 `status`·`n_iter`·`nfev`), `advance`(갱신·판정 뒤 남은 원소를 순서대로 줄이기). 원소의 상태는 알고리즘마다의 구조체다.
- C1: 이 부품을 `csrc/elementwise/loop.hpp`(이름공간 `sd::elementwise`, C23; scipy `_lib/_elementwise_iterative_method.py`의 `_loop`에 해당)로 끌어냈다. 함께 옮긴 것: scipy의 status 번호(`sConverged` 0, `sSignErr` −1, `sConvErr` −2, `sValueErr` −3, `sInputErr` −5, `sInProgress` 1; −4는 `callback`이 없어 두지 않았다), numpy의 `sign`·`minimum`·`maximum`(`npsign`·`npmin`·`npmax`), Python 층의 1차원 배열과 벡터를 오가는 `read`·`out`. `batch.cpp`에는 네 알고리즘과 그 자리의 뜻(−1의 `bracket_*` 한계, 2의 `bracket_root` 반대쪽)만 남았다. 헤더뿐이라 CMake는 그대로다. ARCHITECTURE.md §1의 그림에 optimize/ ──▶ elementwise/ ──▶ array/·parallel/을 더했다(integrate/·differentiate/의 화살표는 그 함수가 생길 때 더한다).
- 넓힌 것(C24): `Call::start(k, p)`가 원소마다 p개 점의 호출((k, p) C 순서, `args`는 (k, 1))을 만들고 `put(r, j, x)`·`x(r, j)`·`value(r, j)`로 쓰고 읽는다. 돌아온 값이 float64 (k, p)가 아니면 `check(value, (k, p))`를 부른다((k,)는 지금처럼 `check(value, k)`). p는 호출마다 바뀔 수 있고(`tanhsinh`의 수준), 점이 모자란 원소의 칸은 부르는 쪽이 NaN으로 채운다(`nsum`; scipy와 같다). 1차원으로 편 꼴(§9 C2.1 (b))은 `start(k·p)`와 원소 번호를 p번 되풀이한 `idx`로 된다. optimize 쪽은 (k,)라 그대로다. 넓힌 길은 C2의 `derivative`가 처음 썼다(`tests/differentiate/test_derivative.py`의 호출 꼴·스레드 시험; `loop.hpp`는 C2에서 바뀌지 않아 optimize 배치의 비트와 속도는 그대로다).
- 확인(C22): 기계어를 견주면 원소 패스(`advance`·`for_elements`의 람다)는 명령 수가 모두 같고, 네 함수의 바깥 몸통은 레지스터 배정이 달라 19–73개 줄었고, `Call::run`은 `gather`를 따로 둔 만큼 짧아졌다. 속도는 같은 소스(다른 작업의 C++ 변경도 함께인 같은 시점의 빌드 둘: 옮기기 전과 뒤)의 `bench/speed`의 optimize 배치 사례(사용자 함수가 sd 배열 연산인 x³ − 2x − c와 (x − c)² + 0.01·c·x⁴)를 한 프로세스에서 사례마다 0.6초씩 되풀이해 재고, 두 빌드를 번갈아 14번(1스레드)·6번(10스레드) 돌린 가운데값의 비(뒤/전)다. 기계가 바빴다(부하 1.9–6.6, 다른 작업 둘).

| 사례 | 1스레드 전 (ms) | 뒤/전 | 10스레드 전 (ms) | 뒤/전 |
|---|---|---|---|---|
| `find_root` 10³ / 10⁴ | 0.135 / 1.169 | 0.997 / 1.003 | | |
| `find_root` 10⁵ / 10⁶ | 12.26 / 123.0 | 1.001 / 1.004 | 3.92 / 31.6 | 0.999 / 0.983 |
| `bracket_root` 10⁴ / 10⁶ | 0.645 / 63.6 | 1.004 / 1.002 | — / 29.3 | — / 0.989 |
| `find_minimum` 10³ / 10⁴ | 0.248 / 2.000 | 0.994 / 0.997 | | |
| `find_minimum` 10⁵ / 10⁶ | 19.25 / 191.7 | 0.995 / 0.992 | 6.46 / 55.3 | 0.998 / 1.004 |
| `bracket_minimum` 10⁴ / 10⁶ | 0.562 / 53.8 | 1.003 / 0.999 | — / 23.6 | — / 1.006 |

  - 모두 ±0.5% 안(10스레드 `find_root` 10⁶만 −1.7%)이라 느려진 곳이 없고 되돌린 곳도 없다. 사례마다의 가장 빠른 값의 비는 0.97–1.013으로 잡음이 더 크다. `bench.speed run optimize --match elementwise --smoke`도 12사례 모두 지났다(검사 0 실패). 기준선(`bench/speed/results/`)은 바꾸지 않았다.

### 6.3 이 영역의 파일
- Python: `src/stardust/{integrate,differentiate}/`. integrate는 `_quadpack.py`(`quad`, `dblquad`, `tplquad`, `nquad`; C4), `_quad_vec.py`, `_cubature.py`, `_tanhsinh.py`(`tanhsinh`, `nsum`), `_samples.py`(표본 규칙의 인자 검사, C3), `_rules.py`(`fixed_quad`, `newton_cotes`, C3), `_gauss.py`(Gauss–Legendre의 반올림과 정확한 산술, C3), `_result.py`. 인자 검사 도우미는 `stardust.optimize._common`(`_callable`, `_check_int`, `_real` 등)에서, 원소별 무리의 입력(`_inputs`, `_args`, `_shaped`)은 `stardust.optimize._elementwise`에서 가져온다(integrate·differentiate의 Python 층이 optimize에 기댄다; C++의 고리는 C16대로 `csrc/elementwise/`에 있다). differentiate는 `_derivative.py`(`derivative`, `jacobian`, `hessian`: 입력 검사와 결과), `_weights.py`(올바로 반올림한 가중치, C28·C34), `_result.py`(`DifferentiateResult`와 `summary()`), `_differences.py`(C15).
- C++: `csrc/{integrate,differentiate}/`. integrate는 `quadpack.hpp`(C4: dqagse·dqagie·dqagpe, 규칙 dqk21·dqk15i, 목록 dqpsrt, ε-알고리즘 dqelg의 템플릿; 상수는 QUADPACK의 리터럴; C5: dqawoe·dqawfe·dqawse·dqawce, 규칙 dqk15w·dqc25f·dqc25s·dqc25c, dqcheb, dqmomo, dgtsl)와 `quadpack.cpp`(scipy의 갈래, 구간 배열, `nquad`의 겹침; C68), `adaptive.cpp`(`quad_vec`·`cubature`의 힙), `tanhsinh.cpp`(표, 합, 오차 추정; `nsum`), `numpy_ops.hpp`(numpy의 pairwise 합과 `linalg.norm`의 페이지 정렬 `cblas_ddot`; `tanhsinh.cpp`와 함께 쓴다, C97), `samples.cpp`(표본 규칙과 `fixed_quad`의 합; 합은 배열의 커널, C3), `gauss.cpp`(Gauss–Legendre: n ≤ 100의 표 `gauss_table.hpp`(생성: `tools/gen_gauss_legendre.py`)와 그 위의 double-double Newton, C3), `integrate.hpp`(바인딩 선언). differentiate는 `steps.hpp`(C3), `derivative.hpp`·`derivative.cpp`(스텐실의 점과 갱신, `derivative`는 `elementwise/loop.hpp`의 고리, `jacobian`·`hessian`의 열 호출과 안쪽 `jacobian`; C2). 바인딩 `csrc/bind/{integrate,differentiate}_bind.cpp`(`stardust._core.integrate`, `stardust._core.differentiate`).
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
- **C3에서 더한 것**: `tools/integrate_ref.py samples`(→ `tests/integrate/ref/samples.json`: 표본 12벌마다 다섯 규칙의 식을 같은 double에서 `fractions.Fraction`으로 정확히 계산한 값; 식이 유리 연산뿐이라 Arb가 필요 없다. 누적은 줄마다의 값 전부, 항목마다 표본의 해시)와 `tools/integrate_ref.py gauss`(→ `tests/integrate/ref/gauss.json`: n = 101, 128, 150, 200, 256, 500, 1000의 음이 아닌 점·가중치를 Arb `arb.legendre_p_root`로 2⁻¹¹⁰까지 정해 반올림한 값, `tools/gen_gauss_legendre.py`의 함수). n ≤ 100의 표는 `tools/gen_gauss_legendre.py`가 같은 방법으로 `csrc/integrate/gauss_table.hpp`에 쓰고, 테스트가 Arb로 다시 계산해 견준다(`tests/integrate/test_fixed.py`).
- **C0에서 만든 것**: `tools/integrate_ref.py`(→ `tests/integrate/ref/integrals.json` 116개, `series.json` 9, `multiple.json` 7, `vector.json` 2)와 `tools/differentiate_ref.py`(→ `tests/differentiate/ref/derivatives.json` 104점의 f′·f″, `mgh.npz`: 야코비안 47·헤시안 42). 값은 2⁻¹¹⁰까지 정한 공(double-double)이다. 문제가 허락하는 방법을 모두 계산하고 공이 겹쳐야 한다(겹치지 않으면 도구가 멈춘다): 116개 가운데 닫힌 꼴 103·`acb.integral` 13이 저장되고, 18개는 `acb.integral`, 5개는 치환이 함께 확인했다. 항목마다 문제 정의의 해시가 있어 문제를 바꾸면 테스트가 낡은 참값을 알린다.
- **C5에서 더한 것**: `tools/integrate_ref.py weights`(→ `tests/integrate/ref/weights.json`): `tests/integrate/weights.py`의 가중치 스윕 99개(QAWO 18, QAWF 17, QAWS 40, QAWC 24)의 닫힌 꼴(e^(cx)의 [e^((c+iω)x)/(c+iω)], Si·Ci, 베타 함수와 그 도함수(digamma·trigamma), Ei, 부분분수의 주값)이고, QAWO의 e^(cx) 14개는 `acb.integral`이 함께 확인했다.
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
- 까닭의 탐침(`--probe-contract`: 같은 원본을 `-ffp-contract=fast`로 빌드해 견주기만 하고 아무것도 쓰지 않는다): 모두 같은 것이 26 → 72개, 적분 62 → 100개, 오차 81개가 되지만 점이 같은 것은 112 → 90개로 준다. scipy의 C 판은 누적(합)은 FMA로 축약하고 점의 계산(c ± h·x)은 축약하지 않은 것처럼 보인다(clang의 축약은 한 식 안에서만 일어나고, scipy가 점을 따로 계산한 줄은 축약할 곱셈-덧셈이 없다). C4에서 이 자리를 scipy의 호출 순서와 결과로 찾아 `std::fma`를 적는다(§3.1, §8의 C4). 오차 추정이 축약으로도 남는 차이는 식의 꼴(`**1.5`와 `pow`의 차이 등)로 보고 C4에서 찾는다. C4에서 찾았다(§7.10, C62): 합쳐진 곳은 Kronrod 규칙의 누적 넷뿐이고, `pow(x, 1.5)`는 scipy와 같다(sd가 `pow`로 106개 모두 같다). 축약을 켠 gfortran 빌드에 남은 차이는 GCC가 문장을 넘어 합친 자리(점의 c − h·x 등) 때문으로 보인다.
- QAWS의 로그 가중치 예제(`alg-loga`)는 점이 같고도 적분이 4×10⁻⁹ 다르다(Fortran −0.189275181, scipy −0.189275185, 참값 −0.189275188; 둘 다 허용오차 안). 축약으로도 남는다. C5에서 찾았다: 원본 `dqc25s.f`의 오타(`res24 = res12 + …`)이고 scipy의 C 판은 고친 식이다(C75, §7.11).
- Fortran과 scipy가 수렴이라고 알린 값은 참값과 오차 추정 안이다(scipy와 같은 예외 셋 빼고: §7.7).

### 7.7 C0의 기록에서 찾은 것 (scipy 1.18.1, 2026-11-21)
- **오차 추정의 정직함**(수렴이라고 알렸는데 |값 − 참값| > 오차 추정): `quad`는 116개 가운데 수렴 104개에서 셋이다. 봉우리 α = 19·20(값 ≈ 0, 참값 ≈ π; 설계 초안과 같다)과 f10 √tan t다. f10은 외삽이 double 상한이 아니라 정확한 π/2의 적분(π√2/2, 8×10⁻¹³ 안)으로 가서 double 상한의 참값과 1.6×10⁻⁸ 다르다(참값은 double 문제 그대로다: C19). `tanhsinh`은 수렴 76개 가운데 여덟이다: 봉우리 α = 0·2·4, cos(2^α sin x)의 α = 6·7·9·10, f9(double 상한의 조각 2×10⁻¹⁵). 배치로 부른 무리 셋은 원소마다 혼자 부른 것과 값·오차·`nfev`가 비트로 같다.
- `tanhsinh`이 수렴하지 못하는 것(status −2): 봉우리 α ≥ 5, 내부 특이점(QAGP 예제, |x − 1/3|^α 셋), 끝점 특이점과 무한 구간(f12), 좁은 봉우리 1/(x² + 10⁻⁶), f7, f10, f15. `nsum`은 급수 아홉이 모두 수렴하고 오차 추정 안이다.
- `quad`가 수렴하지 못하는 것: 봉우리 α = 10–18(ier 5), cos(2^α sin x)의 α = 9(ier 1)·10(ier 5), f15 sin(t)/t의 [0, ∞)(ier 5; QAGI는 진동을 위한 것이 아니다).
- `derivative`: 같은 점을 혼자 부를 때와 배치(9점)로 부를 때 `status`·`nfev`·`n_iter`는 같지만 df가 99점 가운데 29점에서 끝자리가 다르다(최대 상대 2.8×10⁻¹⁰, 언제나 오차 추정 안). df = `fc @ wc`가 BLAS(dgemv)라 합의 순서가 배치의 크기에 달려 있고, 스텐실의 상쇄가 그 차이를 키운다. sd는 원소마다 같은 비트를 지키므로(§1.3) C2의 호출점 비교는 혼자 부른 기록과 견주고 df의 끝자리는 C5의 예외로 적는다(§3.8). 끝 근처의 기록: log의 10⁻³·√의 10⁻⁶은 기본값으로 status −3(9번), `step_direction=1`의 log 10⁻³은 status −2(오차가 줄지 않는다), x³의 0은 −2(`nfev` 27), tan의 1.5는 x + 0.5가 π/2를 넘는데도 7번 반복 뒤 수렴(값은 맞다).

### 7.8 C2에서 잰 것 (scipy 1.18.1, 2026-11-21)
`tests/differentiate/data/`의 scipy 기록(§7.3)과 견준 것. 호출점은 sd가 부른 점의 바이트를 scipy 기록과 같은 방법(호출 순서대로의 SHA-256, 점의 수, 호출 수)으로 잰 것이다. 테스트가 아래 수를 그대로 확인한다(`tests/differentiate/test_derivative.py`, `test_jacobian.py`, `test_weights.py`).

| 함수 | 기록 | 호출점이 같다 | status·`n_iter`·`nfev`가 같다 | df가 비트로 같다 |
|---|---|---|---|---|
| `derivative` | 124(점마다 99, 배치 11, 끝 14) | 123 | 123 | 5 |
| `jacobian` | 47(MGH) | 43 | 항목 7,289 / 7,456 (status·`nfev`) | 항목 2,937 / 7,456 |
| `hessian` | 42(MGH, n ≤ 10) | 33 | 항목 1,517 / 1,748 (status·`nfev`) | 항목 66 / 1,748 |

- `derivative`에서 다른 하나는 x³의 0(기본값)이다: sd는 둘째 반복에 수렴하고 scipy는 −2다(§3.8, C28, C38). 다른 123개는 점·모양·수·status·`n_iter`·`nfev`가 모두 같다. df가 비트로 같은 것이 적은 것은 가중치(C28)와 합(C38)이 둘 다 다르기 때문이다.
- 까닭을 가르는 탐침: sd의 커널에 scipy의 LAPACK 가중치를 넣으면 `derivative`는 124개 모두 호출점·status가 같다(x³의 0도 −2). `jacobian`은 scipy 가중치로도 같은 네 경우의 호출이 다르다: 그 경우들은 0에 가까운 항목이나 판정의 경계에 있는 항목이 다른 반복에 멈춘 것이고, 가르는 것은 합의 순서다. scipy의 합(BLAS dgemv, Accelerate)의 순서는 행의 수에 따라 바뀌어 하나로 흉내 낼 수 없다(차례대로, FMA 사슬, 2·4·8갈래 누산을 1–1,000행에서 견주었다: 어느 것도 늘 같지 않다). 차례대로의 합에 scipy 가중치를 넣은 탐침에서는 df가 124개 가운데 70개까지 비트로 같았다.
  - 호출이 다른 경우: `jacobian`은 mgh20(n9), mgh27(n30, n40), mgh28(n10); `hessian`은 mgh13, mgh10, mgh35(n9), mgh04, mgh14, mgh18, mgh24(n10), mgh30, mgh31. 호출이 같아도 항목 하나가 다른 반복에 멈출 수 있다(다른 항목이 남아 호출이 이어지므로).
  - 차례대로의 합(C38 전의 구현)과 견주면: `jacobian`의 같은 호출 39 → 43, 같은 항목 7,156 → 7,289; `hessian`의 같은 호출 35 → 33, 같은 항목 1,488 → 1,517; `derivative`는 그대로 123.
- 둘 다 수렴한 값의 차이: `derivative`는 atol + rtol·|df| 안, `jacobian`은 최대 9.1×10⁻¹⁴·max|J|, `hessian`은 4.7×10⁻¹²·max|H|(rtol의 3×10⁻⁴)다.
- **Arb 계약**: `derivative`는 모음 113점(배치 빼고) 가운데 109점이 수렴하고 모두 참 오차 ≤ atol + rtol·|df|, 최대 상대오차 6.2×10⁻¹¹(tanh의 −0.5; scipy도 6.2×10⁻¹¹)이다. 수렴하지 못한 넷은 scipy와 같다(log 10⁻³ 기본값 −3, log 10⁻³ 오른쪽 −2, √ 10⁻⁶ 기본값 −3, exp의 order 2는 −2). 오차 추정이 참 오차보다 작은 것은 다섯이다(log 10⁴ 0.15배, √의 10·100 0.95·0.64배, atan의 10 0.48배, √ 10⁻⁶ 오른쪽 0.26배). `jacobian`은 수렴한 항목 7,389개의 노름 오차가 최대 rtol의 0.0037배(mgh11), `hessian`은 1,612개가 0.11배(mgh04, Brown의 나쁜 척도)다. 항목마다 atol + rtol·|df|를 넘는 것은 0에 가까운 항목에서 `jacobian` 210, `hessian` 4다.
- **가중치** (`test_weights.py`): (n, c) = (1, 2), (2, 2), (4, 2), (6, 2), (8, 2), (4, 1.5), (4, 4), (4, 0.5), (3, 3), (5, 1.1), (4, 2.25)의 모든 가중치가 Arb(400비트의 Vandermonde 풀이) 공에 가장 가까운 double이다.
- **속도** (한 번의 탐침, 기계가 바빴다: 부하 1.5–1.9; 기준선은 `bench/speed`가 조용한 기계에서): `derivative`(exp) 원소 10·10³·10⁵에서 sd 0.0096·0.045·1.44 ms, scipy 0.198·0.401·16.0 ms; `jacobian`(Rosenbrock n = 3) 0.014·0.237 ms; `hessian` 0.073·3.61 ms. 같은 입력이면 1·3·10스레드에서 비트가 같다(원소 4만 개로 확인).

### 7.9 C3에서 잰 것 (scipy 1.18.1, 2026-11-22)
테스트가 아래 수를 그대로 확인한다(`tests/integrate/test_samples.py`, `test_fixed.py`).

**표본 규칙**
- 정확한 식과의 계약(§4): 12벌 모두 한계 안이고 가장 큰 것은 한계의 0.050(`simpson`, 같은 간격 17점)이다. 상대오차의 최대: `trapezoid` 1.4×10⁻¹⁶, `simpson` 3.3×10⁻¹⁶, `romb` 1.7×10⁻¹⁶, `cumulative_trapezoid` 1.7×10⁻¹⁵, `cumulative_simpson` 4.0×10⁻¹⁵(4,097점의 누적). scipy는 같은 모음에서 9.8×10⁻¹⁷, 1.3×10⁻¹⁶, 1.3×10⁻¹⁶이고 누적은 같은 값이다(정확도 기준선 `bench/accuracy/results/integrate.json`).
- scipy의 기록(`scipy_samples.json`, 값 65개: 다섯 규칙과 `initial=0`)과 비트로 같은 것은 59개다. 다른 여섯(17점의 `trapezoid`·`simpson`, 33점의 `simpson`·`romb`, 4,097점의 `romb`, 2차원 `trapezoid`)은 합의 순서(16누산기와 numpy의 8누산기)로 끝자리가 다르고 상대 2.6×10⁻¹⁶ 이하다. 누적은 둘 다 차례대로 더해(sd의 스캔은 8,192점부터 블록) 모두 같다.
- 줄의 값은 축·배치·스레드와 상관없다: 5줄의 각 줄을 혼자, 2차원으로, 전치해 `axis=0`으로, 띄엄띄엄한 뷰로 부른 값이 비트로 같고, 2¹⁷ + 1점 한 줄과 40 × 4,097의 다섯 규칙이 1·3·10스레드에서 같다.

**`fixed_quad`의 점과 가중치**
- n = 1–100의 표 2,550개 점·가중치가 모두 Arb(256비트)에 가장 가까운 double이고, n = 101, 128, 150, 200, 256, 500, 1000의 실행 중 계산이 `gauss.json`과 비트로 같다. 모든 값을 정확한 산술로 보내도(테스트의 강제; n = 101, 128) 같다.
- double-double의 실제 오차(Arb와 견줌): 점은 n = 101에서 6.1×10⁻¹⁷ ulp, 1000에서 1.5×10⁻¹⁶ ulp, 3000에서 4.7×10⁻¹⁶ ulp; 가중치는 상대 2⁻⁹⁸(101), 2⁻⁹⁰·⁵(1000), 2⁻⁹⁴(3000). 반올림을 정확한 산술로 넘기는 한계(C54)는 이것의 2¹⁰배 이상이다.
- scipy `roots_legendre`와: n = 5는 점이 모두 같고 가중치 5개가 17 ULP까지, n = 20은 점 4개가 1 ULP·가중치 20개가 261 ULP까지, n = 100은 점 38개가 8 ULP·가중치 100개가 83,682 ULP까지 다르다(§7.4와 같은 수). 그래서 scipy의 기록(유한한 매끄러운 적분 30개 × n = 5, 20, 100)과 호출점이 같은 것은 n = 5에서 30개, n = 20·100에서 0개이고, 값이 비트로 같은 것은 4·4·0개다.
- 합의 계약: 처음 40개 문제의 n = 5, 20, 100, 150에서 값이 f의 값으로 낸 정확한 합과 (⌈log₂ n⌉ + 4)·ε·Σ|wᵢfᵢ| 안이고 가장 큰 것은 한계의 0.12다.

**`newton_cotes`**
- N = 1–14(같은 간격)는 scipy와 `an`·`B`가 비트로 같다(scipy의 표도 정확한 유리수의 반올림이다). 그 위의 scipy는 Vandermonde 역행렬이라 정확한 값과 N = 15에서 가중치 상대 1.2×10⁻¹², `B` 7.2×10⁻⁶, N = 20에서 가중치 1.9×10⁻¹¹, `B` 0.09, N = 30에서 `B`가 10⁶배 다르다(`B`의 식이 크게 상쇄한다). 위치를 준 경우([0, 0.5, 2])도 scipy와 끝자리가 다르다(5.6×10⁻¹⁷, `B` −0.1111111111111111과 scipy −0.11111111111111116).
- 다른 알고리즘(모멘트 방정식 Σaᵢrᵢᵏ = Nᵏ⁺¹/(k + 1)의 유리수 소거)과 다섯 위치 집합에서 비트로 같다.

**속도** (한 번의 탐침, 기계가 바빴다: 부하 2.9; 기준선은 `bench/speed`가 조용한 기계에서): 스레드는 기본(10), 시간은 따로 적지 않으면 µs.

| 사례 | sd | scipy | 사례 | sd | scipy |
|---|---|---|---|---|---|
| `trapezoid` 10³ / 10⁶ (x 있음) | 1.5 / 73 | 3.6 / 646 | `romb` 2¹⁰ + 1 / 2²⁰ + 1 | 1.4 / 75 | 21 / 334 |
| `simpson` 10³ / 10⁶ | 2.1 / 150 | 36 / 3,770 | `fixed_quad` n = 5 / 200 | 2.4 / 2.8 | 6.4 / 6.9 |
| `cumulative_trapezoid` 10³ / 10⁶ | 3.0 / 259 | 7.1 / 2,558 | `newton_cotes` 8 / 20 (기억한 뒤) | 약 2 | 1.3 / 24 |
| `cumulative_simpson` 10³ / 10⁶ | 4.5 / 366 | 29 / 8,079 | Gauss–Legendre 점 계산 n = 1,000 / 3,000 / 10⁴ (처음 한 번) | 9 ms / 68 ms / 0.74 s | |

- `newton_cotes`의 첫 계산은 정확한 유리수라 느리다(N = 8 0.19 ms, N = 20 1.0 ms; scipy 1.3 µs·24 µs). 같은 입력은 기억한다(C53).

### 7.10 C4에서 잰 것 (scipy 1.18.1, 2026-11-22)
테스트가 아래 수를 그대로 확인한다(`tests/integrate/test_quadpack.py`).

**scipy의 기록과** (가중치 없는 106개: `scipy_quad.json`, `scipy_calls.npz`)
- 106개 모두 점의 순서와 비트, 적분, 오차 추정, `nfev`, status, 글(ier ≠ 0), 구간(scipy `infodict`의 `last`·`alist`·`blist`·`rlist`·`elist`와 `iord`의 순위)이 같다. 가중치 있는 10개는 C5다(지금은 `NotImplementedError`, C61).
- FMA의 자리(C62)를 가르는 탐침(같은 커널을 축약 없이, 또는 한 자리만 풀어 빌드해 견주었다; 저장소에 두지 않았다):

| 빌드 | 모두 같다 | 점 | 적분 | 오차 추정 | 구간 |
|---|---|---|---|---|---|
| `std::fma` 넷(이 구현) | 106 | 106 | 106 | 106 | 106 |
| 축약 없음 | 9 | 105 | 60 | 29 | 10 |
| resg만 풀기 | 55 | 106 | 106 | 79 | 59 |
| resk만 풀기 | 12 | 105 | 60 | 35 | 17 |
| resabs만 풀기 | 46 | 106 | 106 | 103 | 46 |
| resasc만 풀기 | 55 | 106 | 106 | 84 | 61 |
| 같은 소스, `std::fma` 대신 곱·합을 쓰고 clang `-ffp-contract=on` | 106 | 106 | 106 | 106 | 106 |

  - 마지막 줄이 scipy의 빌드 조건이다: 원본을 문장마다 옮긴 C를 clang의 기본 축약으로 빌드하면 scipy와 같다. 넷 모두 결과에 닿는다: resk는 적분 자체, resg는 |K − G|, resabs는 반올림 하한 50ε·resabs, resasc는 (200|K − G|/resasc)^1.5의 크기 조절이다. 점이 다른 하나(축약 없음, resk)는 x·log(1/x)(α = 1)이고 Fortran 원본의 차이(§7.6)와 같은 곳이다(멈춤 판정이 끝자리로 갈린다).
  - Fortran 원본(`-ffp-contract=off`)이 scipy와 모두 같은 것이 26개, gfortran의 `-ffp-contract=fast`가 72개였던 것(§7.6)은 이것으로 설명된다: 앞은 축약이 없고, 뒤는 GCC가 문장을 넘어 점의 c − h·x까지 합쳐 점이 달라진 것이다.
  - `pow(x, 1.5)`는 원본의 `**1.5`이고 macOS libm의 `pow`다(scipy와 같은 값; x·√x로 바꾸지 않는다).
- 기록 밖의 경우(지금의 scipy와 그 자리에서 견줌): `limit` 1·3·400, `atol` = 0과 `rtol` 10⁻¹³, `rtol` = 0, NaN·inf를 돌려주는 f, (−∞, b], 끝을 바꾼 `points`(겹친 점과 구간 밖의 점), 계단 함수, 중단점 7개와 `limit` 8의 12경우와 `complex_func=True`가 점·값·오차·`nfev`·글까지 같다.

**겹친 적분** (`scipy_multiple.json`)
- `dblquad`·`tplquad` 7개 모두 f를 부른 점(호출 순서대로의 SHA-256), 적분, 오차, `nfev`가 같고, 상자 영역 4개의 `nquad`도 값·오차·`nfev`가 같다. scipy 문서의 4차원 예(옵션 함수가 바깥 변수로 중단점을 준다)와 범위 함수·옵션 함수의 3차원 경우도 점의 순서와 값이 같다.
- 오차가 모든 `quad`의 최댓값인 것(C63): √|x − y|의 단위 정사각형에서 sd·scipy 모두 바깥 `quad`의 추정보다 크다.

**Arb 참값과** (§4의 계약; `bench/accuracy/integrate.py`의 `quad`·`multiple`)
- 가중치 없는 106개 가운데 94개가 수렴하고, 그 가운데 91개는 참 오차가 오차 추정 안이다(ρ = 추정/참 오차: 참 오차 0이 둘, 나머지 89개는 최소 1.43(x·log(1/x), α = 1), 가운데값 1,464, 최대 1.2×10⁹). 밖의 셋은 scipy와 같다(§7.7): 봉우리 α = 19·20(ρ = 1.7×10⁻⁹·4.1×10⁻¹⁰: 처음 21점이 봉우리를 놓친다)과 f10(ρ = 0.027: 정확한 π/2의 적분으로 외삽). 정확도 기준선 `bench/accuracy/results/integrate.json`의 계약 최대는 0.70(앞의 α = 1)이다.
- 겹친 적분 7개: 상대오차 최대 5.1×10⁻¹⁵, 참 오차/오차 추정 최대 0.0067.

**속도** (한 번의 탐침, 기계가 바빴다: 부하 3.6–4.1; 기준선은 `bench/speed`가 조용한 기계에서): 시간은 µs, 같은 Python 함수.

| 사례 | sd | scipy | 사례 | sd | scipy |
|---|---|---|---|---|---|
| `quad` exp(−x²) [0, 1] (21번) | 2.13 | 1.98 | `quad` `points=[0.3]` (42번) | 4.61 | 6.53 |
| `quad` exp(−x²) (−∞, ∞) (270번) | 14.1 | 17.2 | `quad` `complex_func=True` (42번) | 7.06 | 8.18 |
| `quad` 봉우리 α = 5 (525번) | 38.1 | 44.5 | `dblquad` 단위 정사각형 (441번) | 34.3 | 64.3 |
| | | | `tplquad` 단위 정육면체 (9,261번) | 468 | 1,060 |

- 함숫값 21개(맨 Python 호출만 0.91 µs)의 가장 작은 경우만 7% 느리다: 인자 검사와 결과(필드 열·구간 배열 셋)를 만드는 데 1.2 µs, scipy는 1.1 µs다. 빠른 길(C70)을 두기 전에는 7.4 µs였다. 겹친 적분은 안쪽 `quad`가 C++ 안에서 돌아(C64) 두 배 빠르다.

### 7.11 C5에서 잰 것 (scipy 1.18.1, 2026-11-22)
테스트가 아래 수를 그대로 확인한다(`tests/integrate/test_quadpack.py`, `test_quadpack_weights.py`).

**scipy와** (가중치 있는 10개의 기록, 스윕 99개와 만든 400개는 지금의 scipy와 그 자리에서)
- 기록 10개: 점의 순서와 비트, 적분, 오차 추정, `nfev`, status, 구간(scipy `infodict`의 `alist`…`elist`와 `iord`의 순위; QAWF는 `lst`·`rslst`·`erlst`·`ierlst`)이 모두 같다. 그래서 모음 116개 모두 같다.
- 스윕 99개: 98개가 같고, 다른 하나는 C78(QAWF의 ω = 0, a = 1)이다. 만든 400개(가중치 일곱, 첫 규칙 셋(\|ωh\| ≤ 2, ≤ 24, > 24), 허용오차·`limit`·`max_p1` 50·100·`limlst`, status 0·1·2·7)는 모두 같다. 복소수 피적분함수(QAWO·QAWS·QAWC·QAWF)와 가중치를 층마다 준 `nquad`(옵션 함수 포함)도 점과 값이 같다.
- 원본과 다른 scipy의 세 곳(C75, C76, C79)과 sd가 scipy와 다른 세 곳(C77, C78, C82)을 가르는 탐침(같은 커널을 바꿔 빌드해 1,476개 실행(ω = 0의 QAWF 빼고)과 견주었다; 저장소에 두지 않았다): LINPACK 꼴의 `dgtsl`이면 9개, 원본의 1000·uflow면 1개가 다르다. C77을 scipy처럼 두면 1,476개 모두 같고, 원본의 뜻대로 두면(이 구현) `max_p1` = 1·3·5로 모멘트를 다 쓴 29개가 다르다.
- FMA의 자리(C81): 같은 탐침에서 한 무리씩 풀어 본 것(`max_p1`의 차이는 scipy처럼 둔 빌드에서):

| 빌드 | 기록 10 | 1,476개 모두 | 점 | 적분 | 오차 추정 |
|---|---|---|---|---|---|
| `std::fma` 모두(이 구현) | 10 | 1,476 | 1,476 | 1,476 | 1,476 |
| `dqk15w`의 누적만 풀기 | 6 | 1,021 | 1,473 | 1,288 | 1,058 |
| `dqcheb`만 | 6 | 1,004 | 1,474 | 1,287 | 1,023 |
| `dgtsl`만 | 9 | 1,428 | 1,476 | 1,466 | 1,434 |
| `dqc25f`의 모멘트만 | 6 | 1,315 | 1,475 | 1,397 | 1,343 |
| `dqc25f`의 점만 | 7 | 1,225 | 1,225 | 1,383 | 1,278 |
| `dqc25f`의 합·결과만 | 6 | 1,184 | 1,474 | 1,328 | 1,207 |
| `dqawfe`만 | 10 | 1,467 | 1,476 | 1,476 | 1,467 |
| `dqmomo`만 | 10 | 1,381 | 1,476 | 1,441 | 1,391 |
| `dqc25s`만 | 9 | 1,189 | 1,473 | 1,344 | 1,210 |
| `dqc25c`만 | 8 | 1,414 | 1,476 | 1,439 | 1,430 |
| 축약 없음 | 2 | 432 | 1,220 | 896 | 533 |
| 같은 소스를 곱·합으로, clang `-ffp-contract=on` | 10 | 1,476 | 1,476 | 1,476 | 1,476 |

  - 모든 무리가 결과에 닿는다. 점이 다른 것은 `dqc25f`의 점 centr ± hlgth·x가 한 식이기 때문이다(§7.6의 "QAWF 첫 주기의 점이 끝자리에서 다르다"가 이것이다).

**Fortran 원본과** (`tools/gen_integrate_fortran.py`, §7.6)
- `--probe-qc25s`(`dqc25s.f`의 오타 한 줄만 고친 빌드, 축약 없음): QAWS 로그 가중치 예제의 적분이 scipy와 비트로 같다(오차 추정은 축약 때문에 다르다). 가중치 10개 가운데 적분이 같은 것은 3개(QAWS 셋), 모두 같은 것은 2개다. 나머지의 차이는 축약(C81)이다.

**Arb 참값과** (C83; `bench/accuracy/integrate.py`의 `quad_weights`)
- 모음 10개와 스윕 99개가 모두 수렴하고 참 오차 ≤ 오차 추정 + 8ε\|I\|다(최대 0.46: e^(−x)·cos 40x의 [0, 1]). 상대오차의 최대는 QAWO 4.4×10⁻¹⁵, QAWF 1.2×10⁻⁹(허용오차 1.49×10⁻⁸ 안), QAWS 1.3×10⁻⁸(QAWS 예제), QAWC 6.5×10⁻¹⁴이다. 추정만으로 넘는 16개는 QAWO 7(\|ωh\|가 5–500인 `dgtsl`·점화식의 모멘트, 오차 추정 0 둘)·QAWC 9(다항식·유리식의 주값, 오차 추정 0)이고 모두 5ε\|I\| 이하다.
- scipy는 같은 비트라 같은 수이고, QAWF 영역만 ω = 0의 a = 1에서 상대오차 1.72(C78)다.
- C123(`dqc25f` 추정의 바닥 ε·resabs) 뒤: 계약 대비 최대 QAWO 0.075(e^(−x)·sin 7x의 [0, 3]), QAWF 0.0042, 추정만으로 넘는 것은 QAWC 9개뿐이다. scipy와는 위의 점·값·`nfev`·status가 그대로이고(만든 400개 가운데 `99-cos3-cos` 하나만 경로가 바뀌어 status 1), QAWO·QAWF의 오차 추정은 scipy 이상이다(스윕에서 9개가 커졌다).

**속도** (한 번의 탐침, 부하 2.6; 기준선은 `bench/speed`가 조용한 기계에서): 시간은 µs, 같은 Python 함수.

| 사례 | sd | scipy | 사례 | sd | scipy |
|---|---|---|---|---|---|
| QAWF e^(−x) cos 3x, [0, ∞) (175번) | 11.8 | 12.2 | QAWS e^(−x²) alg-loga (50번) | 4.5 | 4.4 |
| QAWO log x sin 10πx (305번) | 20.2 | 22.3 | QAWC e^(−x²)/(x − 0.3) (185번) | 10.9 | 12.3 |

- 가중치의 인자 검사에 빠른 길(기본 허용오차, float의 `wvar`, float 둘의 튜플)을 두어 QAWS의 가장 작은 경우가 6.8 → 4.5 µs가 되었다.

### 7.12 C6에서 잰 것 (scipy 1.18.1, 2026-11-22)
테스트가 아래 수를 그대로 확인한다(`tests/integrate/test_tanhsinh.py`).

**scipy의 기록과** (`scipy_tanhsinh.json`, `scipy_nsum.json`)
- `tanhsinh` 101개, 배치 셋(30·21·11원소, 매개변수는 `args`), `nsum`의 급수 아홉이 모두 부른 점(호출 순서대로의 SHA-256, 점의 수, 호출의 모양; 첫 호출은 scipy의 () 대신 (1,), O48), 적분(합), 오차 추정, `nfev`, status, `max_level`까지 비트로 같다. 처음 만든 구현이 그대로 같았다: 바꾼 자리가 없다.
- 기록 밖의 경우(지금의 scipy와 그 자리에서 견줌): `tanhsinh` 42경우(log=True 일곱, `min_level` 0·1·3·5와 log의 0·1·3, `max_level` 0·1·2·3·12, `min_level` > `max_level`, 거꾸로·같은·NaN 끝과 무한 끝의 조합, 가운데의 NaN, 모두 inf·NaN·0인 f, `atol`·`rtol`, 빠른 진동, 배치 다섯: `args`, 끝의 배치, 2차원 브로드캐스트, log, 2,000원소)와 `nsum` 29경우(log 여섯, 양쪽·왼쪽, 틀린 `step`, b < a, NaN, `max_terms` 0·1·1,000, −4, 발산, inf 항, 네 갈래가 섞인 배치, 허용오차, 단봉, a = b의 셋)가 점·값·오차·`nfev`·status까지 같다. 다른 것은 log=True이고 a = b = −∞인 `nsum` 하나다(C85: sd는 NaN과 status −2, scipy는 복소수).
- 알고리즘의 세부로 확인한 것: scipy의 `_euler_maclaurin_sum`이 유한하지 않은 값을 가장 가까운 유한한 점의 값으로 바꾸는 것은 전치 배열의 reshape이 뷰라서 실제로 일어난다. numpy의 줄마다 `np.sum`은 줄의 수와 상관없이 0 + pairwise(줄)이다(1–20,000점, 1–5,000줄에서 확인). `np.exp`·`log`·`log1p`·`cosh`·`sinh`·`power`·`hypot`, 복소 `exp`·`log`는 macOS libm(`cexp`·`clog`)과 비트로 같다(Python의 `math.hypot`은 다르다).

**표** (C84)
- 수준 0–10의 8,193쌍(16,386값)이 scipy의 `_compute_pair`와 비트로 같고 h₀ = asinh(log(2/(4·tiny) − 1)/π)/8도 같다. 수준 16까지 524,289쌍을 만들 수 있다(C88).
- 정확한 값과의 거리(Arb, 200비트): 같은 식을 같은 double jh(j·h의 반올림)에서 계산한 값과 최대 980 ULP(979.6, 평균 46.8; 설계 초안의 수), double h의 정확한 j·h에서는 최대 3,449 ULP(평균 161), 정확한 h₀(tmax/8)의 점과는 5,438 ULP(평균 258)다. 가장 큰 것은 꼬리(u₂가 큰 점)의 가중치다.

**Arb 참값과** (§4의 계약; `bench/accuracy/integrate.py`의 `tanhsinh`·`nsum`, 기준선 `bench/accuracy/results/integrate.json`)
- `tanhsinh`: 101개 가운데 76개가 수렴하고(25개는 −2), 참 오차가 오차 추정 밖인 것은 scipy와 같은 여덟이다(§7.7): 봉우리 α = 0·2·4(ρ = 0.025·0.054·0.015), cos(2^α sin x)의 α = 6·7·9·10(0.94·0.85·0.30·0.22), f9(오차 추정 0, 참 오차 8×10⁻¹⁶). 나머지 68개는 참 오차 0이 둘, ρ 최소 1.28, 가운데값 34, 최대 5,124다. 계약 대비 최대는 0.78이다.
- `nsum`: 아홉 모두 수렴하고 참 오차가 오차 추정 안이다(ρ = 1.12(H(1000, 2): 직접 합의 추정 ε|S|)–5.2×10⁵, 상대오차 최대 3.0×10⁻¹²(ζ(3))). 계약 대비 최대 0.89.
- 모음 밖에서 오차 추정이 짧은 예: e^(−cx²)의 [0, 2]를 c ∈ [0.5, 3]의 40,001개로 부르면(scipy와 비트로 같다) 모두 수렴하고 1,093개가 참 오차보다 작은 추정을 낸다(수준 3에서 멈춘 1,022개, 2의 43개 모두, 4의 28개; 추정이 0인 것 28개). 가장 나쁜 c ≈ 1.956은 수준 2에서 추정 1.1×10⁻¹², 상대오차 1.1×10⁻⁸이다. 오차 추정은 scipy의 것이고 엄밀한 상계가 아니다(API 문서 §6).

**속도** (한 번의 탐침, 기계가 바빴다: 부하 1.5–2.2; 기준선은 `bench/speed`가 조용한 기계에서): 시간은 ms, f는 양쪽 같은 식(sd는 `sd.exp`, scipy는 `np.exp`), 스레드는 기본(10).

| 사례 | sd | scipy | 사례 | sd | scipy |
|---|---|---|---|---|---|
| `tanhsinh` exp(cx) 원소 1 | 0.010 | 0.227 | `tanhsinh` 원소 10³ | 0.40 | 1.01 |
| `tanhsinh` 원소 10 | 0.015 | 0.38 | `tanhsinh` 원소 10⁵ (함숫값 7.8×10⁶) | 17.7 (1스레드 41.5) | 75.3 |
| `nsum` 1/k² (10번, 8,561점) | 0.035 | 0.79 | | | |

- 원소 1·10과 `nsum`에서 23–25배, 원소 10³에서 2.5배, 10⁵에서 4.3배(1스레드 1.8배) 빠르다. 1스레드 10⁵원소는 함숫값 하나에 5.3 ns(사용자 함수의 `sd.exp` 포함)다. `csrc/elementwise/loop.hpp`는 바꾸지 않아 optimize 배치와 `derivative`의 비트와 속도는 그대로다.

### 7.13 C7에서 잰 것 (scipy 1.18.1, 2026-11-22)
테스트가 아래 수를 그대로 확인한다(`tests/integrate/test_quad_vec.py`, `test_cubature.py`; 경우의 목록은 `tests/integrate/adaptive.py`).

**scipy와** (기록은 `scipy_vector.json`·`scipy_multiple.json`, 나머지는 지금의 scipy와 그 자리에서)
- `quad_vec`: 기록 2개, 기록 밖의 30경우(무한 구간 넷과 거꾸로·같은 끝, 규칙 셋, 노름 둘, `points`(무한 구간에서도), `limit` 1·40, 허용오차 0, `atol`, NaN, 2차원 값, 1,000원소, 진동, 반올림 멈춤, 정수·numpy 수의 값), 모음 106개(가중치 없는 것, 수 값, `quad`의 중단점), 무리 셋을 벡터 값 하나로(노름 둘)와 scipy의 기억이 넘치는 경우(50,000원소의 √|x − c|: 기억 249개, `limit` 400에서 구간 512개 가운데 263칸이 NaN이고 다시 계산한 호출(`nfev` 24,318)까지 같다)가 부른 점(호출 순서대로의 SHA-256, 호출 수), 적분, 오차, `nfev`, status, 글, 구간(끝·값·오차)까지 비트로 같다. 처음 만든 구현이 그대로 같았다.
- `cubature`: 기록 12개(상자 넷 × 규칙 셋, 호출의 모양까지), 기록 밖의 39경우(1–4차원, 벡터·3차원 값, C·Fortran·뒤집은 열·건너뛴 줄·브로드캐스트·전치한 3차원 배치, 정수 값, f가 인자를 그대로 돌려주는 경우, 평면·반평면·반직선·직선, 뒤집은 끝, `points`(모서리 위의 점, 폭 0의 조각), `max_subdivisions`, `atol`), 1차원 모음 105개 × 규칙 둘과 무리 셋 × 규칙 둘이 같다(점, 값, 오차, `subdivisions`, 영역).
  - 처음 구현은 `genz-malik`의 정수 값과 인자를 그대로 돌려주는 경우에서 오차의 끝자리가 달랐다: scipy가 f에 Fortran 순서의 배열을 넘기고(점의 표가 전치) 원소별 계산의 값이 그 배치를 이어받아 numpy의 합이 pairwise가 된다(C100). 같은 배치를 넘기고 같은 규칙으로 더하자 모두 같아졌다.
  - 다른 것은 셋이고 모두 C101–C103이다: (−∞, b](모음의 (−∞, 0]의 정규 밀도: sd는 Arb 값과 오차 추정 안), 무한 축과 `points`(scipy는 에러), NaN(scipy는 `'converged'`, sd는 status 3).
- numpy에서 확인한 것(같은 탐침): 1차원 `np.sum`은 0 + pairwise(5–194,481점), C 순서 (n, m)의 `sum(axis=0)`은 0부터 차례대로, Fortran 순서는 줄마다 0 + pairwise, `np.prod`는 차례대로다. numpy의 새 배열은 256바이트 정렬이다. `cblas_ddot`은 8바이트씩 옮긴 주소에서 2,250개 가운데 967개, 16바이트 정렬끼리 56개(모두 897원소 이상), 128바이트 정렬끼리도 700개 가운데 255개가 달랐고, 페이지 정렬끼리는 152개 모두 같았다(C97).

**Arb 참값과** (§4의 계약, C107; `bench/accuracy/integrate.py`의 `quad_vec`·`cubature`, 기준선 `bench/accuracy/results/integrate.json`; scipy는 같은 비트라 같은 수다)

| 영역 | 경우 | 수렴 | 짧음 | 계약 대비 최대 |
|---|---|---|---|---|
| `quad_vec` 모음 | 106 | 99 | 5(봉우리 α = 15–18, f10) | 0.95 |
| `quad_vec` 무리, 노름 `'2'` / `'max'` | 3 / 3 | 2 / 2 | 0 / 0 | 0.84 / 0.84 |
| `quad_vec` 벡터 값 | 2 | 2 | 0 | 0.0047 |
| `cubature` 1차원 `'gk21'` / `'gk15'` | 106 / 106 | 105 / 105 | 8 / 6 | 0.96 / 0.96 |
| `cubature` 무리의 원소 `'gk21'` / `'gk15'` | 62 / 62 | 62 / 62 | 3 / 5 | 0.64 / 0.62 |
| `cubature` 상자 `'gk21'` / `'gk15'` / `'genz-malik'` | 4 / 4 / 4 | 4 / 4 / 4 | 0 | 0.093 / 1.2×10⁻⁵ / 0.082 |

- 짧은 것(`tests/integrate/adaptive.py`): `quad_vec`은 `quad`가 수렴하지 못하는 봉우리 α = 15–18을 수렴으로 알리고(참 오차가 추정의 2.2–33배), f10은 `quad`와 같다. `cubature`는 끝점 특이점 x^−0.9·−0.8·−0.7과 x^−0.9(추정의 1.4–6.0배), 좁은 봉우리, f12, |x − 1/3|^−0.5다. 무리를 벡터 값으로 부르면 원소마다의 판정이라 x^−0.9 하나와 봉우리 몇이 남는다.
- 계약의 8ε|I|가 덮는 것: `cubature`의 \|K − G\|에는 반올림 하한이 없어 끝자리까지 맞은 원소(cos(2^α sin x)의 α = 0–3, 매끄러운 x^α log(1/x))의 추정이 참 오차보다 작다(무리의 `'gk21'`에서 21개가 추정만으로는 넘고 모두 8ε|I| 안).

**속도** (한 번의 탐침, 부하 2.2–3.0; 기준선은 `bench/speed`가 조용한 기계에서): 시간은 ms, f는 양쪽 같은 식(`cubature`의 sd는 `sd.exp`·`sd.sum`, scipy는 numpy).

| 사례 | sd | scipy | 사례 | sd | scipy |
|---|---|---|---|---|---|
| `cubature` gk21 2차원 (호출 2번) | 0.021 | 0.116 | `quad_vec` 함수 셋 (63번) | 0.020 | 0.149 |
| `cubature` genz-malik 2차원 (106번) | 0.13 | 1.34 | `quad_vec` 봉우리 둘 (819번) | 0.23 | 1.92 |
| `cubature` gk15 평면 (448번) | 1.63 | 13.6 | `quad_vec` 수 하나, 실수 전체 (330번) | 0.023 | 0.104 |
| `cubature` genz-malik 4차원 (rtol 10⁻⁶) | 0.73 | 5.12 | `quad_vec` cos(kx), 1,000원소 (315번) | 1.17 | 2.11 |

- 호출 사이의 일(힙, 규칙의 합, 구간 기억)이 C++라 함숫값이 작은 경우 4.5–10배 빠르다. 1,000원소의 `quad_vec`은 사용자 함수(`sd.cos`)와 값의 읽기가 시간의 대부분이라 1.8배다.

### 7.14 C8에서 잰 것 (scipy 1.18.1, 2026-11-22)
테스트가 아래의 고침을 확인한다(`tests/integrate/test_tanhsinh.py`·`tests/differentiate/test_derivative.py`의 `test_threads_by_points`, `tests/integrate/test_fixed.py`, `tests/differentiate/test_differences.py`, `tests/optimize/test_diff.py`).

**측정** (C113; 기계가 바빴다: 부하 2.7–3.9, 다른 세션들; 1라운드, 기준선에 올리지 않음): 고치기 전 `build/speed/20261122-045253-c8-before.jsonl`, 뒤 `build/speed/20261122-051631-c8-after.jsonl`(표는 `bench/integrate/speed.md`, `bench/differentiate/speed.md`). scipy ÷ sd의 묶음별 중앙값(1스레드 / 10스레드):

| 묶음 | 전 | 뒤 | sd가 느린 줄(뒤) |
|---|---|---|---|
| 표본 규칙과 고정 규칙(14) | 2.64 / 5.67 | 2.82 / 7.98 | 0 / 0 (전에는 `newton_cotes` 8의 0.35배) |
| QUADPACK(12) | 1.18 / 1.16 | 1.18 / 1.18 | 2 / 2 (함숫값 21개의 `quad` 0.95–0.96배, QAWS 0.96–0.98배) |
| 배치 tanh-sinh와 급수(6) | 2.20 / 3.83 | 2.28 / 10.7 | 1 / 0 (`nsum` 배치 10³의 1스레드 0.93배) |
| 벡터 값과 다차원(8) | 7.13 / 7.22 | 7.24 / 7.45 | 0 / 0 |
| 배치 도함수(4) | 7.69 / 9.55 | 8.50 / 12.1 | 0 / 0 |
| 야코비안과 헤시안(6) | 15.9 / 15.8 | 16.1 / 16.0 | 0 / 0 |

**A/B** (C111; 고치기 전 빌드의 사본과 새 빌드를 번갈아 띄운 프로세스 셋씩, 가운데값, ms; 모든 결과(값, 오차, `nfev`, status, 수준·반복)의 해시가 두 빌드와 1·10스레드에서 같다):

| 사례 | 1스레드 전 → 뒤 | 10스레드 전 → 뒤 | scipy |
|---|---|---|---|
| `tanhsinh` exp 원소 10³ | 0.385 → 0.362 | 0.371 → 0.134 | 1.05 |
| `tanhsinh` 원소 10⁴ | 3.96 → 3.77 | 3.39 → 0.580 | |
| `tanhsinh` 원소 10⁵ | 40.8 → 39.6 | 15.8 → 5.38 | 76.4 |
| `tanhsinh` log, 원소 10⁴ | 18.8 → 18.4 | 18.7 → 2.33 | |
| `tanhsinh` 실수 전체, 원소 10⁴ | 27.8 → 26.8 | 22.9 → 3.79 | |
| `nsum` 1/k^p 배치 10 / 10³ | 4.84 → 4.76 / 518 → 500 | 1.72 → 0.84 / 200 → 64.8 | 5.68 / 460 |
| `nsum` 직접 합 원소 10³ | 3.40 → 2.91 | 3.28 → 0.546 | |
| `derivative` exp 원소 10⁴ / 10⁵ | 0.375 → 0.351 / 3.97 → 3.75 | 0.355 → 0.265 / 1.51 → 1.07 | — / 17.1 |
| `derivative` 한쪽, 원소 10⁵ | 3.47 → 3.03 | 1.47 → 1.00 | |

- 1스레드가 3–15% 빨라진 것은 점의 사본이 없어서이고(C111), 10스레드는 원소 10³–10⁴개에서 3–8배다. 원소 10개의 `tanhsinh`(12–13 µs)는 같은 빌드끼리의 잡음(±9%) 안이다. `jacobian`·`hessian`은 원소별 고리를 쓰지 않아 그대로다. `newton_cotes` 8·20(기억한 뒤)은 3.8·7.9 µs → 0.25 µs(scipy 1.36·21.8 µs; C108).
- tanhsinh 10⁵원소의 호출 사이(10스레드): 첫 호출 전 6.5 → 1.0 ms(점의 사본과 0 채우기가 없어지고 점 쓰기가 병렬), 수준 사이 4.4 → 2.4 ms, 마지막 갱신(남은 19,585원소 × 64점; 전에는 원소 2¹⁵개 아래라 한 스레드) 4.0 → 0.8 ms.

**남긴 것** (C114; 나중의 최적화 세션에 넘긴다)
- 함숫값 21개의 `quad`(2.07 µs, scipy 1.99 µs): 네이티브 호출이 1.47 µs(그 가운데 Python 콜백 21번이 약 0.87 µs), 결과 0.29 µs, 인자 검사 약 0.27 µs다. 남은 차이는 구간 배열 셋(C55)을 만드는 일이라 결과의 필드를 늦게 만드는 길(`_Lazy`)이 후보다. QAWS의 가장 작은 경우(4.5 µs, scipy 4.4 µs)도 같다.
- `nsum` 배치 10³의 1스레드(493 ms, scipy 460 ms): 사용자 함수 1/k^p의 (1000, 131072) 배열이 0.34초(numpy 0.27초)이고, 나머지는 sd 0.13초·scipy 0.17초다. sd의 `**`가 1스레드에서 numpy보다 약 10% 느린 것(배열 영역)이다.
- `tanhsinh` 10⁵원소의 1스레드 첫 호출 5.2 ms: 새 (10⁵, 66) 배열의 페이지 폴트와 원소 상태의 준비다.
- Gauss–Legendre 처음 계산의 상수배(C109)와 `newton_cotes`의 첫 계산(C110).

**일관성** (C115)
- 영역 안: 모든 공개 함수의 에러가 `이름(): ` 꼴이다. 같은 실수는 같은 문장이다: 없는 키워드, `'atol' must be a finite number >= 0`(NaN·음수), `'atol' must be an int or a float, not str`, `'max_iter' must be >= 1`·`must be an int, not float`, `'args' must be a tuple, not int`, `'log' must be True or False, not int`, 부를 수 없는 f(`'f' must be callable, not float`; `quad` 계열과 `fixed_quad`는 scipy의 이름 `'func'`), `axis`(`axis 2 is out of bounds for an array of dimension 1`, 표본 규칙의 `'axis' must be an int`는 정수 하나만 받는 뜻이다: sd.stats의 튜플 축과 다르다), 상한(`'max_level' must be at most 16`, `'max_terms' must be at most 16777216 (2**24)`, C109·C110). 결과는 모두 필드와 배열 필드가 쓰기 금지이고, `summary()`는 인자가 없으며(`summary(): unexpected keyword argument 'alpha'`, `summary(): takes 0 positional arguments but 1 was given`), `repr`는 한 줄이다(API.md §2.3; 자유도가 없어 G195는 해당이 없다). 원소별 무리의 입력(`tanhsinh`·`nsum`의 끝, `derivative`의 x)과 optimize 배치의 `init`은 같은 함수(`_inputs`)라 같은 글이다.
- 다른 영역과 뜻이 달라 그대로 둔 것: `derivative`·`jacobian`·`hessian`의 `max_iter`는 1 이상이고 optimize 배치 넷은 0을 받는다(둘 다 scipy의 범위: 차분은 첫 반복이 처음 추정이다). `quad` 계열의 `atol`·`rtol`은 수 기본값이라 None을 받지 않고 원소별 무리는 None이 기본값이다(scipy의 시그니처). 원소별 무리(optimize의 배치도)는 frame의 `Series`를 받지 않고 `approx_fprime`의 `xk`와 optimize의 `x0`·표본 규칙의 `y`·`cubature`의 끝은 받는다(API.md §5의 데이터 입력과 원소별 함수의 구분).
- 다른 영역에서 본 것(고치지 않고 적는다): (a) optimize의 원소별 입력 검사(`_inputs`)는 같은 인자의 틀린 형에 두 문장을 쓴다: 문자열은 `'init' must be an sd.ndarray or a sequence of numbers, not str`, bool은 `'init' must be a real number or an array, not bool`. (b) `OptimizeResult`의 `repr`는 필드를 이름 순으로 적고(`converged=True, fun=0, n_iter=1, nfev=3, x=0.5`; O120) `IntegrateResult`·`DifferentiateResult`는 값부터 적는다(API.md §2.3의 예). (c) sd의 `**`(배열 영역)는 큰 배열의 1스레드에서 numpy보다 약 10% 느리다(위의 `nsum`).
- `approx_fprime`(optimize의 공개 이름, 이 영역의 커널): C112로 고쳤다.

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 |
|---|---|
| C0 준비 (완료·승인(위임) 2026-11-21) | API 문서의 확정, 결과 객체의 바탕(`OptimizeResult`에서 끌어내기, 동작 불변), 모듈 `sd.integrate`·`sd.differentiate`(공개 이름은 결과 클래스뿐), 시험 모음(§7.1), Arb 참값 도구(§7.2), scipy 1.18.1 기록과 호출 순서(§7.3), QUADPACK 원본 대조(C17, §7.6), `bench/areas.py`와 측정·골든의 등록, scipy 시간(바쁜 기계라 점검 수준, §5) |
| C1 함께 쓰는 부품 (완료·승인(위임) 2026-11-21) | C3·C15: `diff.hpp`와 `Differences`를 differentiate로 옮기기. C4·C16: `batch.cpp`에서 반복 고리를 csrc 아래 `elementwise/`로 끌어내 넓히기. 공개 함수 없음. optimize·regression의 테스트·골든·배치 속도가 그대로(§6.1, §6.2; C22–C25) |
| C2 미분 (완료·승인(위임) 2026-11-21) | `derivative`, `jacobian`, `hessian`(C26–C38; §3.8, §7.8) |
| C3 표본과 고정 규칙 (완료·승인(위임) 2026-11-22) | `trapezoid simpson romb cumulative_trapezoid cumulative_simpson fixed_quad newton_cotes`(C39–C54; §3.7, §7.9) |
| C4 QUADPACK 1 (완료·승인(위임) 2026-11-22) | `quad`(QAGS, QAGI, QAGP), `dblquad`, `tplquad`, `nquad`. scipy의 C 판이 FMA로 축약한 누적의 자리를 호출 순서와 결과로 찾아 `std::fma`를 적는다(§7.6의 축약 탐침). `summary()`(C20) (C55–C70; §3.1, §3.2, §7.10) |
| C5 QUADPACK 2 (완료·승인(위임) 2026-11-22) | `quad`의 `weight`(QAWO, QAWF, QAWS, QAWC). QAWS 로그 가중치 예제의 scipy·Fortran 4×10⁻⁹ 차이(§7.6)의 까닭을 찾는다(C75). QAWO·QAWF의 Fortran 대조는 LINPACK 원본 `dgtsl`로 돈 것이다(C18) (C71–C83; §3.1, §7.11) |
| C6 배치 적분 (완료·승인(위임) 2026-11-22) | `tanhsinh`, `nsum`(C84–C93; §3.5, §3.6, §7.12) |
| C7 벡터 값과 다차원 (완료·승인(위임) 2026-11-22) | `quad_vec`, `cubature`(C94–C107; §3.3, §3.4, §7.13) |
| C8 최적화와 일관성 점검 (완료·승인(위임) 2026-11-22) | 모든 공개 함수를 크기마다 scipy와 견주어 느린 곳을 고치고(C111, C108), 남은 질문(Gauss–Legendre의 상한 C109, `approx_fprime`의 간격 C112)을 닫고, 메시지·시그니처의 영역 간 일관성을 본다(C115). 코드 리뷰는 리뷰 세션이 한다(C116). 결정 C108–C116, 결과 §7.14 |

- 순서의 까닭: C1의 고리를 처음 쓰는 C2가 고리의 넓힘(원소마다 여러 점)을 바로 시험한다. C3은 다른 것에 기대지 않는다. C6의 `nsum`은 C6의 `tanhsinh`을, C7의 무한 축 치환은 C4·C6과 같은 식을 쓴다.
- **나중의 ODE 묶음**(C2, 이름은 그때): `solve_ivp`(RK23, RK45, DOP853, Radau, BDF, LSODA; 조밀 출력, 사건), `odeint`(LSODA), `solve_bvp`. 원소별 무리가 아니라 상태가 있는 풀이라 이 묶음의 고리를 쓰지 않는다. 큰 결정은 그때 묻는다.
- 진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.

---

## 9. 마일스톤을 시작하기 전에 물을 것
추천을 첫째에 둔다.

- **C0**: 모두 정했다(C8–C17; C0을 마치며 생긴 넷은 추천안으로 C18–C21, 사용자 위임 2026-11-21). `dgtsl`의 질문도 닫혔다: 원본을 받았다(C18, 사용자 승인 2026-11-21).
- **C1**: 정했다(C22–C25, 추천안, 사용자 위임 2026-11-21).
- **C2**: 정했다(C26–C38, 추천안, 사용자 위임 2026-11-21). 물었던 넷은 (a)다: (k, p) 2차원(C26), (m, N)의 f(C27), 올바로 반올림한 가중치(C28; 한쪽은 Arb가 아니라 Q(√c)의 정확한 계산, C34), `hessian`의 작은 rtol은 `ValueError`(C29). 구현하며 생긴 아홉(C30–C38)도 추천안으로 정했다.
- **C3**: 정했다(C39–C54, 추천안, 사용자 위임 2026-11-21). 물었던 다섯은 (a)다: 데이터 함수의 입력(C39), `x`는 위치와 키워드 모두(C40), `axis`의 기본 −1(C41), 올바로 반올림한 Gauss–Legendre(공개 이름 없음, C42), `fixed_quad`는 값 하나·`newton_cotes`는 `(an, B)`(C43). 구현하며 생긴 열하나(C44–C54: `romb`, `x`·`dx`의 꼴, 누적의 `initial`, 짝수 개 점의 `simpson`, dtype, 빈 입력, NaN, 합의 순서, `fixed_quad`의 함수, `newton_cotes`의 정확한 계산, 표 밖의 Gauss–Legendre)도 추천안으로 정했다.
- **C4**: 정했다(C55–C70, 추천안, 사용자 위임 2026-11-21). 물었던 넷은 (a)다: 구간 정보의 필드 넷(C55), 복소수는 `quad(complex_func=True)`만(C56), `vectorized=True` 없음(C58), scipy 글의 줄바꿈 그대로(C59). 구현하며 생긴 열둘(C57, C60–C70: 복소수 결과의 필드, 수렴의 글, C5 전의 `weight`, FMA의 자리, 겹친 적분의 오차·`nfev`·status, 겹침의 자리, `quad`와 겹친 적분의 인자, QNG, 파일, `summary()`, 속도)도 추천안으로 정했다.
- **C5**: 정했다(C71–C83, 추천안, 사용자 위임 2026-11-21). 물었던 하나는 (a)다: `wopts`를 두지 않는다(C71). 구현하며 생긴 열둘(C72–C83: 결과의 필드, 인자, QAWF의 허용오차, `dqc25s`의 오타, `dgtsl`의 꼴, 모멘트 행, QAWF의 ω = 0, `dqawoe`의 작은 구간, QAWO의 구간 하나, FMA의 자리, 큰 ω, Arb 계약)도 추천안으로 정했다.
- **C6**: 정했다(C84–C93, 추천안, 사용자 위임 2026-11-21). 물었던 셋: tanh-sinh 표는 (a) scipy의 libm 식(C84), `log=True`는 C4의 답대로 실수만(C85), 끝점은 (a) 부르고 버린다(C86). 구현하며 생긴 일곱(C87–C93: numpy의 합과 logsumexp, 수준과 `max_terms`의 상한, log의 허용오차, 첫 호출과 갈래, `nsum`의 두 예외, 결과와 검사)도 추천안으로 정했다.
- **C7**: 정했다(C94–C107, 추천안, 사용자 위임 2026-11-21). 물었던 셋: `cubature`는 (a) `converged`와 정수 `status`(C94; 유한하지 않은 값의 3은 구현하며 더했다, C103), `regions`는 C55의 꼴인 배열 필드(C95), 규칙 객체·`workers`는 없음(C96). 구현하며 생긴 열하나(C97–C107: numpy·Python의 산술, scipy의 구간 기억, 낮은 Gauss 규칙, 배치와 합, (−∞, b]의 치환, 무한 축의 `points`, 유한하지 않은 값, 두 함수의 인자, `summary()`, Arb 계약)도 추천안으로 정했다.
- **C8**: 정했다(C108–C116, 추천안, 사용자 위임 2026-11-21). 맡긴 범위(속도, C0–C7에서 남긴 것, 일관성, 코드 리뷰는 남김)를 추천안대로 했다. 구현하며 정한 것: `newton_cotes`의 int 기억(C108), 두 상한(C109, C110), 점 수로 나누는 병렬(C111), `approx_fprime`의 간격(C112), 측정·남긴 것·일관성·리뷰(C113–C116).
