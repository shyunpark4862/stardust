# sd.differentiate API

> 상태: **확정** (C0, 2026-11-21; 승인 대기). 공개된 이름은 결과 클래스 `DifferentiateResult`뿐이다. 아래 시그니처는 scipy 1.18.1 `scipy.differentiate`의 이름과 인자 이름(C1, C2)에 stardust의 규칙과 C4·C8–C17의 결정(결과 클래스 `DifferentiateResult`, `atol`·`rtol`, `max_iter`, `nfev`, C4에서 따라 나오는 것)을 입히고, 남은 질문(docs/design/integrate.md §9 C2)은 그 추천안을 넣은 안이다. 세 함수는 마일스톤 C2에서 공개된다(docs/design/integrate.md §8).

- 설계와 결정의 근거는 [docs/design/integrate.md](../design/integrate.md)(§1.1 C1–C17, 방법은 §3.8)다. 이 모듈은 설계 문서를 `sd.integrate`와 함께 쓴다(C1). 적분은 [docs/api/integrate.md](integrate.md), 공통 규칙은 [API.md](../../API.md)다.
- `sd.optimize`의 `approx_fprime`·`check_grad`(고정 간격 차분)는 scipy처럼 `sd.optimize`에 있다. 그 간격의 커널(C++ `csrc/differentiate/steps.hpp`)과 nls의 5점 차분 `Differences`(Python `stardust.differentiate._differences`)는 이 영역의 내부다(C3, C15; C1에서 옮겼고 값은 그대로다). 공개 이름은 아니다.
- 언어 표시가 없는 코드 블록에는 공개 이름과 시그니처만 둔다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- 함수 이름과 인자 이름은 scipy의 것이고, 다른 영역의 꼴이 먼저다(C4): 결과의 `converged`(scipy `success`), `n_iter`(scipy `nit`), `nfev`; 반복 상한 `max_iter`(scipy `maxiter`, O14); 허용오차는 scipy `tolerances` dict 대신 키워드 `atol`·`rtol`(O36).
- **함수와 점은 위치로만, 옵션은 키워드로만** 받는다. 별칭은 없다.
- `args`는 `derivative`만 받는다(원소별 반복 무리, C4·O39; scipy의 `jacobian`·`hessian`에도 없다). `kwargs`, `callback`, `preserve_shape`는 없다(C4, C14, O37; `[scipy와 다름]`).
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1): `derivative(): unexpected keyword argument 'tolerances'`.

### §1.2 사용자 함수
- Python 호출 가능한 객체다. 사용자 함수가 낸 에러는 그대로 나간다.
- **`derivative`**: 원소별 함수다(`f(x)[i]`가 `f(x[i])`; scipy의 계약). 아직 끝나지 않은 원소의 점을 모아 부른다(O38). 첫 호출은 (k,)(원소마다 x 하나, O48), 다음부터 원소마다 여러 점이다(첫 반복 `order`개, 다음 반복 2개). 넘기는 꼴은 docs/design/integrate.md §9 C2의 질문이다(추천: (k, p) 2차원 float64 `sd.ndarray`, 원소의 순서는 입력의 C 순서; `args`의 값은 (k, 1)). 돌려받는 값은 같은 모양의 실수(`sd.ndarray`, 시퀀스, numpy 배열)다. 함수가 정의되지 않는 점에서는 예외 대신 NaN을 돌려줘야 한다: 그 원소가 status −3으로 멈춘다.
- **`jacobian`, `hessian`**: f는 벡터화된 함수다: 첫 축이 변수인 (m, N) 배열을 받아 첫 축이 출력인 (k, N)(스칼라 함수는 (N,))을 돌려준다(scipy 계약의 2차원 판; docs/design/integrate.md §9 C2의 추천). x는 1차원 (m,)이다. scipy처럼 첫 호출부터 변수마다 한 변수만 바꾼 m개의 열을 받는다(N = m·p).
- 결과 모양이 맞지 않으면 `ValueError`다: `derivative(): 'f' must return 3 values (one per point), not 2`, `jacobian(): 'f' must return an array whose last axis has 24 points, not shape (2, 3)`. bool 배열은 `TypeError`다(optimize와 같다).

### §1.3 결과 `DifferentiateResult` (C9, C13)
```
DifferentiateResult
```
- 세 함수가 이 클래스를 돌려준다. 규칙은 `sd.optimize.OptimizeResult`와 같다(없는 필드는 `AttributeError`, 읽기 전용, 배열은 읽기 전용이고 복사하지 않음, O103).

| 필드 | 뜻 | 주는 함수 |
|---|---|---|
| `df` | 도함수(`derivative`, 배치 모양) 또는 야코비안 (k, m)(`jacobian`) | `derivative`, `jacobian` |
| `ddf` | 헤시안 (m, m) | `hessian` |
| `error` | 오차 추정: 이번 반복과 직전 반복의 추정의 차의 크기 | 모두 |
| `converged` | error < atol + rtol·\|df\|로 멈췄는지(bool, 배치면 bool 배열) | 모두 |
| `status` | scipy의 번호(O17): 0 수렴, −1 오차 추정이 늘어 멈춤, −2 `max_iter`, −3 유한하지 않은 값 | 모두 |
| `n_iter` | 반복 수(원소마다) | `derivative`, `jacobian` |
| `nfev` | 사용자 함수의 값을 계산한 점의 수(원소마다; `hessian`은 안쪽 호출을 합한 (m, m)) | 모두 |
| `x` | 미분한 점(`args`·`step_direction`과 브로드캐스트한 뒤) | `derivative` |

- 원소별 결과는 입력의 배치 모양이고 입력이 모두 실수면 Python 수다(O48). scipy의 −4(`callback`)는 없다.
- 결과는 풀리지 않는다(이름으로 읽는다). 공개 생성자 `DifferentiateResult(*, batch=(), **fields)`, 없는 필드의 `AttributeError`(`(the function does not give it)`), `repr`(`DifferentiateResult(df=2.71828, error=4.5e-14, converged=True, n_iter=2, nfev=11)`, 배치는 `DifferentiateResult(batch shape (3,), 2 of 3 converged)`)는 `IntegrateResult`와 같다(docs/api/integrate.md §1.3). `summary()`는 `OptimizeResult`처럼 두고 C2에서 만든다(docs/design/integrate.md §1.1 C20).

### §1.4 멈춤, 상한, 경고
- 모든 호출은 `max_iter`번 안에 끝난다. 상한에 닿은 원소는 마지막 추정과 status −2다.
- 경고를 내지 않는다. 같은 입력이면 스레드 수와 상관없이 같은 비트다. 진행 중인 원소가 2¹⁵개 이상이면 원소를 지나가는 패스를 스레드에 나누고 GIL을 놓는다(optimize의 배치와 같다).
- 도함수가 0인 점에서는 기본 `atol`(가장 작은 정규수)이 사실상 절대 판정을 하지 않으므로 오차가 반올림 수준이어도 status −2가 될 수 있다(scipy와 같다: x³의 0에서 반복 10번). 그런 점에는 `atol`을 준다.

## §2. 도함수 `derivative` (C2)
```
derivative(f, x, /, *, args=(), atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0)
```
- 원소별 도함수 f′(x). 방법(docs/design/integrate.md §3.8): `order`차(짝수로 올림) 차분 공식을 간격을 `step_factor`씩 줄이며 되풀이하고, 이웃한 두 추정의 차가 error다. 중앙 스텐실은 처음에 `order`점, 다음 반복부터 두 점만 새로 부른다. scipy와 같은 점을 같은 순서로 부른다(C5; 판정의 경계에서 반복 수가 갈릴 수 있는 것은 docs/design/integrate.md §3.8).
- `x`, `initial_step`, `step_direction`, `args`의 값은 서로 브로드캐스트한다. `atol`(기본: 가장 작은 정규수)·`rtol`(기본 √ε): error < atol + rtol·|df|이면 멈춘다. `initial_step`(기본 0.5, 0보다 커야 한다; 0 이하인 원소는 NaN과 status −3): 처음의 절대 간격이다. x의 척도를 보지 않으므로 함수가 x ± 0.5에서 정의되지 않으면 더 작은 값이나 `step_direction`을 준다(scipy와 같다: log의 x = 10⁻³은 기본값으로 status −3). `step_factor`(기본 2.0, 1보다 작으면 간격이 커진다). `step_direction`: 0 중앙, 음수면 x 왼쪽만, 양수면 오른쪽만(한쪽 스텐실).
- `order`는 1 이상의 int(홀수는 다음 짝수), `max_iter`는 1 이상의 int다.
- 결과: `df error converged status n_iter nfev x`.

## §3. 야코비안과 헤시안 (C2)
```
jacobian(f, x, /, *, atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0)
hessian(f, x, /, *, atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0)
```
- `jacobian`: f: Rᵐ → Rᵏ의 야코비안 (k, m)(스칼라 f면 기울기 (m,)). 원소마다(출력 i, 변수 j) `derivative`의 방법으로 따로 멈춘다. `initial_step`·`step_direction`은 수 하나 또는 m개다.
  - 결과: `df error converged status n_iter nfev`(원소마다 (k, m)). scipy처럼 `x`는 없다.
- `hessian`: 스칼라 f의 헤시안 (m, m). 기울기를 `jacobian`으로 계산하는 함수의 `jacobian`이다(scipy와 같다). 안쪽 `jacobian`의 rtol은 rtol/100이다. rtol은 100ε 이상이어야 한다(docs/design/integrate.md §9 C2의 추천: `ValueError`; scipy는 경고하고 100ε로 올린다, `[scipy와 다름]`).
  - 결과: `ddf error converged status nfev`. `nfev`는 원소마다 안쪽 호출의 합인 (m, m) int64 배열이다. scipy처럼 `n_iter`는 없다(바깥의 반복 수뿐이라 뜻이 없다).
- 두 함수 모두 `args`가 없다(scipy와 같다): 상수는 클로저로 넘긴다.

## §4. scipy 이름 → sd 이름
| scipy 1.18.1 | sd | 다른 점 |
|---|---|---|
| `derivative(f, x, *, args, kwargs, tolerances, maxiter, order, initial_step, step_factor, step_direction, preserve_shape, callback)` | `derivative(f, x, /, *, args, atol, rtol, max_iter, order, initial_step, step_factor, step_direction)` | `[scipy와 다름]` `kwargs`·`preserve_shape`·`callback` 없음, `tolerances`→`atol`·`rtol`, `maxiter`→`max_iter`, `success`→`converged`, `nit`→`n_iter` |
| `jacobian(f, x, *, tolerances, maxiter, order, initial_step, step_factor, step_direction)` | `jacobian(f, x, /, *, atol, rtol, max_iter, ...)` | `[scipy와 다름]` 위와 같은 이름들; f의 입력은 (m, N) 2차원 |
| `hessian(f, x, *, tolerances, maxiter, order, initial_step, step_factor)` | `hessian(f, x, /, *, atol, rtol, max_iter, ...)` | `[scipy와 다름]` 위와 같은 이름들; rtol < 100ε는 `ValueError` |
| `scipy.optimize.approx_fprime`, `check_grad` | `sd.optimize.approx_fprime`, `check_grad` | 그대로 `sd.optimize`에 있다(docs/api/optimize.md §2) |

## §5. 마일스톤별 남은 질문
C0 전의 것은 정했다(docs/design/integrate.md §1.1 C8–C17). 남은 것은 docs/design/integrate.md §9 C2이고, 이 문서의 꼴은 그 추천안이다: 1(여러 점의 꼴), 2(`jacobian`의 f), 3(가중치), 4(`hessian`의 rtol).
