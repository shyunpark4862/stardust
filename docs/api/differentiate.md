# sd.differentiate API

> 상태: **확정** (C0, 2026-11-21). 세 함수는 마일스톤 C2에서 공개되었다(완료 2026-11-21, 승인 대기). 아래 시그니처는 scipy 1.18.1 `scipy.differentiate`의 이름과 인자 이름(C1, C2)에 stardust의 규칙과 C4·C8–C17의 결정(결과 클래스 `DifferentiateResult`, `atol`·`rtol`, `max_iter`, `nfev`, C4에서 따라 나오는 것)을 입힌 것이고, C2의 결정은 docs/design/integrate.md §1.1 C26–C38이다.

- 설계와 결정의 근거는 [docs/design/integrate.md](../design/integrate.md)(§1.1 C1–C38, 방법은 §3.8, scipy와 견준 것은 §7.8)다. 이 모듈은 설계 문서를 `sd.integrate`와 함께 쓴다(C1). 적분은 [docs/api/integrate.md](integrate.md), 공통 규칙은 [API.md](../../API.md)다.
- `sd.optimize`의 `approx_fprime`·`check_grad`(고정 간격 차분)는 scipy처럼 `sd.optimize`에 있다. 그 간격의 커널(C++ `csrc/differentiate/steps.hpp`)과 nls의 5점 차분 `Differences`(Python `stardust.differentiate._differences`)는 이 영역의 내부다(C3, C15; C1에서 옮겼고 값은 그대로다). 공개 이름은 아니다.
- 언어 표시가 없는 코드 블록에는 공개 이름과 시그니처만 둔다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- 함수 이름과 인자 이름은 scipy의 것이고, 다른 영역의 꼴이 먼저다(C4): 결과의 `converged`(scipy `success`), `n_iter`(scipy `nit`), `nfev`; 반복 상한 `max_iter`(scipy `maxiter`, O14); 허용오차는 scipy `tolerances` dict 대신 키워드 `atol`·`rtol`(O36).
- **함수와 점은 위치로만, 옵션은 키워드로만** 받는다. 별칭은 없다.
- `args`는 `derivative`만 받는다(원소별 반복 무리, C4·O39; scipy의 `jacobian`·`hessian`에도 없다). `kwargs`, `callback`, `preserve_shape`는 없다(C4, C14, O37; `[scipy와 다름]`).
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1): `derivative(): unexpected keyword argument 'tolerances'`.

### §1.2 사용자 함수
- Python 호출 가능한 객체다. 사용자 함수가 낸 에러는 그대로 나간다. 받은 배열을 고쳐도 계산은 그대로다(점은 따로 둔다).
- **`derivative`**: 원소별 함수다(`f(x)[i]`가 `f(x[i])`; scipy의 계약). 첫 호출은 모든 원소의 x를 담은 (k,)다(O48). 다음부터 아직 끝나지 않은 원소만(O38) 원소마다 여러 점을 담은 (k, p) 2차원 float64 `sd.ndarray`다(C26; 원소의 순서는 입력의 C 순서, 첫 반복 p = `order`(짝수로 올림), 다음 반복 p = 2). `args`의 값은 같은 원소로 줄인 (k,)·(k, 1)이다. 돌려받는 값은 같은 모양의 실수(`sd.ndarray`, 시퀀스, numpy 배열)다. 함수가 정의되지 않는 점에서는 예외 대신 NaN을 돌려줘야 한다: 그 원소가 status −3으로 멈춘다.
- **`jacobian`, `hessian`**: f는 벡터화된 함수다: 첫 축이 변수인 (m, N) 배열을 받아 첫 축이 출력인 (k, N)(스칼라 함수는 (N,))을 돌려준다(C27; scipy 계약의 2차원 판). x는 1차원 (m,)이다. 첫 호출은 열 m개가 모두 x인 (m, m)이고, 반복마다 변수마다 그 변수만 바꾼 p개의 열을 쌓은 (m, m·p)다(열 j·p + q가 변수 j의 q번째 점). scipy처럼 모든 항목이 멈출 때까지 반복마다 모든 열을 부른다(C33). 첫 호출이 (k, N)과 (N,) 가운데 어느 꼴인지 정하고 다음 호출도 그 꼴이어야 한다. `hessian`의 f는 스칼라 함수((N,))다.
- 결과 모양이 맞지 않으면 `ValueError`다(C35): `derivative(): 'f' must return shape (2, 8) like its argument, not (2, 1)`, `jacobian(): 'f' must return shape (2,) or (k, 2) for its argument of shape (2, 2), not (3, 3)`, `hessian(): 'f' must return shape (4,) for its argument of shape (2, 4), not (2, 4)`. 수 하나는 `not a scalar`이고, bool 배열·복소수·문자열은 `TypeError`다(optimize와 같다).

### §1.3 결과 `DifferentiateResult` (C9, C13)
```
DifferentiateResult
```
- 세 함수가 이 클래스를 돌려준다. 규칙은 `sd.optimize.OptimizeResult`와 같다(없는 필드는 `AttributeError`, 읽기 전용, 배열은 읽기 전용이고 복사하지 않음, O103).

| 필드 | 뜻 | 주는 함수 |
|---|---|---|
| `df` | 도함수(`derivative`, 배치 모양) 또는 야코비안 (k, m)(`jacobian`; 스칼라 f는 기울기 (m,)) | `derivative`, `jacobian` |
| `ddf` | 헤시안 (m, m) | `hessian` |
| `error` | 오차 추정: 이번 반복과 직전 반복의 추정의 차의 크기 | 모두 |
| `converged` | error < atol + rtol·\|df\|로 멈췄는지(bool, 배치면 bool 배열) | 모두 |
| `status` | scipy의 번호(O17): 0 수렴, −1 오차 추정이 직전의 10배를 넘어 멈춤, −2 `max_iter`, −3 유한하지 않은 x나 값 | 모두 |
| `n_iter` | 반복 수(원소마다) | `derivative`, `jacobian` |
| `nfev` | 사용자 함수의 값을 계산한 점의 수(원소마다; `jacobian`은 열 하나의 몫, `hessian`은 안쪽 호출을 scipy처럼 합한 (m, m)) | 모두 |
| `x` | 미분한 점(`args`·`initial_step`·`step_direction`과 브로드캐스트한 뒤) | `derivative` |

- 원소별 결과는 배치 모양이다: `derivative`는 입력의 브로드캐스트 모양이고 입력이 모두 수면 Python 수다(O48). `jacobian`·`hessian`의 결과는 항목마다 따로 멈추므로 df·ddf의 모양을 배치 모양으로 가진다(C36). scipy의 −4(`callback`)는 없다.
- 결과는 풀리지 않는다(이름으로 읽는다). 공개 생성자 `DifferentiateResult(*, batch=(), **fields)`, 없는 필드의 `AttributeError`(`(the function does not give it)`)는 `IntegrateResult`와 같다(docs/api/integrate.md §1.3).
- `repr`(API.md §2.3): `DifferentiateResult(df=2.71828, error=7.13385e-12, converged=True, n_iter=2, nfev=11)`, 배치는 `DifferentiateResult(batch shape (3, 3), 7 of 9 converged)`.
- `summary()`(C20, C36): 제목 `Differentiation Results`의 상자다. 원소 하나면 머리에 `Converged`·`Status`·`Point`와 `Derivative`·`Error estimate`·`No. Iterations`·`Function Evals`, 배치면 머리에 `Batch shape`·`Converged: j of n`, 표는 원소(항목)마다 한 줄(`[0]`, `[1, 2]`)에 있는 것만의 열 `x df ddf error converged status n_iter nfev`다.

### §1.4 멈춤, 상한, 경고
- 모든 호출은 `max_iter`번 안에 끝난다. 상한에 닿은 원소는 마지막 추정과 status −2다.
- 원소마다의 판정은 scipy의 순서다: error < atol + rtol·|df|이면 0, 아니고 x나 df가 유한하지 않으면 −3(df는 NaN), 아니고 오차 추정이 직전 반복의 10배를 넘으면 −1(반올림이 이기기 시작했다; 그때의 df와 error를 돌려준다). −1은 셋째 반복부터다(오차 추정은 둘째 반복에 처음 생기고, 비교할 직전 오차는 셋째에 생긴다).
- 경고를 내지 않는다. 같은 입력이면 스레드 수·배치의 크기와 상관없이 원소마다 같은 비트다(원소를 배치에 넣어도 혼자 부른 것과 같다; scipy는 BLAS의 합 때문에 끝자리가 다르다). 진행 중인 원소가 2¹⁵개 이상이면 원소를 지나가는 패스를 스레드에 나누고 GIL을 놓는다(optimize의 배치와 같다).
- 도함수가 0인 점에서는 기본 `atol`(가장 작은 정규수)이 사실상 절대 판정을 하지 않으므로 오차가 반올림 수준이어도 status −2가 될 수 있다(scipy와 같다: sin(x₀)의 x₁에 대한 편미분처럼 함수가 그 변수에 기대지 않는 항목). 그런 점에는 `atol`을 준다. x³의 0은 sd에서 둘째 반복에 수렴한다(정확한 가중치와 짝의 합이라 추정이 매번 같다; scipy는 −2: `[scipy와 다름]`, docs/design/integrate.md §3.8).

## §2. 도함수 `derivative` (C2)
```
derivative(f, x, /, *, args=(), atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0)
```
- 원소별 도함수 f′(x). 방법(docs/design/integrate.md §3.8): `order`차(짝수로 올림) 차분 공식을 간격을 `step_factor`씩 줄이며 되풀이하고, 이웃한 두 추정의 차가 error다. 중앙 스텐실은 처음에 `order`점, 다음 반복부터 두 점만 새로 부른다. scipy와 같은 점을 같은 순서로 부른다(C5; 모음의 124개 기록 가운데 123개가 점·수·status까지 같다, docs/design/integrate.md §7.8).
- 가중치는 정확한 값을 올바로 반올림한 것이다(C28). scipy는 Vandermonde 계를 LAPACK으로 풀어 order 8에서 정확한 값과 상대 2.7×10⁻¹⁴(한쪽은 1.1×10⁻¹⁰)까지 다르다. 중앙 스텐실의 합은 마디의 짝마다의 차를 더한다(C38; 상쇄가 적어 더 정확하다). 그래서 df는 scipy와 끝자리가 다를 수 있고(허용오차 안), 판정의 경계에서는 반복 수가 갈릴 수 있다(`[scipy와 다름]`).
- `x`, `initial_step`, `step_direction`, `args`의 값은 서로 브로드캐스트한다. `atol`(기본: 가장 작은 정규수)·`rtol`(기본 √ε): error < atol + rtol·|df|이면 멈춘다. 둘 다 유한한 0 이상의 수다. `initial_step`(기본 0.5): 처음의 절대 간격이다. 0 이하·NaN인 원소는 점이 NaN이고 첫 반복 뒤 status −3이다(scipy와 같다). x의 척도를 보지 않으므로 함수가 x ± 0.5에서 정의되지 않으면 더 작은 값이나 `step_direction`을 준다(scipy와 같다: log의 x = 10⁻³은 기본값으로 status −3). `step_factor`(기본 2.0): 유한하고 0보다 크고 1이 아니어야 한다(C31; `ValueError`, scipy는 1·inf에 `LinAlgError`). 1보다 작으면 간격이 커진다. `step_direction`: 0 중앙, 음수면 x 왼쪽만, 양수면 오른쪽만(한쪽 스텐실, 마디 1/√c의 거듭제곱). NaN인 원소는 status −3이다(C32).
- `order`는 1 이상의 int(홀수는 다음 짝수), `max_iter`는 1 이상의 int다.
- 결과: `df error converged status n_iter nfev x`.

## §3. 야코비안과 헤시안 (C2)
```
jacobian(f, x, /, *, atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0)
hessian(f, x, /, *, atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0)
```
- x는 1차원의 비지 않은 수의 배열(`sd.ndarray`, 시퀀스, numpy 배열)이다(C37). 유한하지 않은 값은 받고 그 변수가 든 항목이 status −3이 된다(scipy와 같다). `initial_step`·`step_direction`은 수 하나 또는 변수마다 m개다(C30).
- `jacobian`: f: Rᵐ → Rᵏ의 야코비안 (k, m)(스칼라 f면 기울기 (m,)). 항목마다(출력 i, 변수 j) `derivative`의 방법으로 따로 멈춘다.
  - 결과: `df error converged status n_iter nfev`(항목마다 (k, m)). scipy처럼 `x`는 없다.
- `hessian`: 스칼라 f의 헤시안 (m, m). 기울기를 `jacobian`으로 계산하는 함수의 `jacobian`이다(scipy와 같다). 안쪽 `jacobian`은 rtol/100이고 나머지 옵션(atol, `max_iter`, `order`, `initial_step`, `step_factor`)은 같으며 늘 중앙 스텐실이다. `initial_step`의 j번째 값은 바깥과 안쪽 모두 변수 j의 간격이다(C30; scipy는 배열을 주면 에러다, `[scipy와 다름]`). rtol은 0이거나 100ε(2.22×10⁻¹⁴) 이상이어야 한다(C29: `ValueError`; scipy는 경고하고 100ε로 올린다, `[scipy와 다름]`).
  - 결과: `ddf error converged status nfev`. `nfev`는 항목 (i, j)마다 바깥 호출에서 안쪽 항목 i의 변수 j 열들이 부른 점의 수를 그 항목이 멈출 때까지 더한 (m, m) int64 배열이다(scipy의 셈). scipy처럼 `n_iter`는 없다(바깥의 반복 수뿐이라 뜻이 없다).
- 두 함수 모두 `args`가 없다(scipy와 같다): 상수는 클로저로 넘긴다.

## §4. scipy 이름 → sd 이름
| scipy 1.18.1 | sd | 다른 점 |
|---|---|---|
| `derivative(f, x, *, args, kwargs, tolerances, maxiter, order, initial_step, step_factor, step_direction, preserve_shape, callback)` | `derivative(f, x, /, *, args, atol, rtol, max_iter, order, initial_step, step_factor, step_direction)` | `[scipy와 다름]` `kwargs`·`preserve_shape`·`callback` 없음, `tolerances`→`atol`·`rtol`, `maxiter`→`max_iter`, `success`→`converged`, `nit`→`n_iter`; 첫 호출은 (k,); 가중치는 올바로 반올림; `step_factor`가 1·0·inf면 `ValueError` |
| `jacobian(f, x, *, tolerances, maxiter, order, initial_step, step_factor, step_direction)` | `jacobian(f, x, /, *, atol, rtol, max_iter, ...)` | `[scipy와 다름]` 위와 같은 이름들; f의 입력은 (m, N) 2차원, x는 1차원만 |
| `hessian(f, x, *, tolerances, maxiter, order, initial_step, step_factor)` | `hessian(f, x, /, *, atol, rtol, max_iter, ...)` | `[scipy와 다름]` 위와 같은 이름들; 0 < rtol < 100ε는 `ValueError`; `initial_step`은 변수마다 줄 수 있다 |
| `scipy.optimize.approx_fprime`, `check_grad` | `sd.optimize.approx_fprime`, `check_grad` | 그대로 `sd.optimize`에 있다(docs/api/optimize.md §2) |

## §5. 정확도
- 수렴한 `derivative`의 결과는 시험 모음(초등 함수 11개 × 9점과 끝 근처 14경우) 모두에서 참값(Arb)과 atol + rtol·|df| 안이다(최대 상대오차 6.2×10⁻¹¹). `jacobian`·`hessian`(MGH 47·42경우)은 수렴한 항목의 오차가 rtol·max|참 행렬|의 0.004배·0.11배 이하다. 오차 추정은 엄밀한 상계가 아니다: 모음에서 참 오차보다 작은 것이 `derivative` 109개 가운데 다섯이다(docs/design/integrate.md §7.8).
