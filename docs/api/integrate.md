# sd.integrate API

> 상태: **확정** (C0, 2026-11-21; 승인(위임)). 공개된 이름은 결과 클래스 `IntegrateResult`와 C3의 일곱 함수(`trapezoid simpson romb cumulative_trapezoid cumulative_simpson fixed_quad newton_cotes`, 2026-11-22; §7, §8), C4의 넷(`quad dblquad tplquad nquad`, 2026-11-22; §2, §3; `quad`의 `weight`는 C5, 2026-11-22), C6의 둘(`tanhsinh nsum`, 2026-11-22; §6), C7의 둘(`quad_vec cubature`, 2026-11-22; §4, §5)이다. 첫 판의 15개가 모두 공개되었다. 아래 시그니처는 scipy 1.18.1 `scipy.integrate`의 이름과 인자 이름(C1, C2)에 stardust의 규칙과 C4·C8–C17의 결정(결과 클래스, 둘로 풀림, `atol`·`rtol`, `max_…`, `integral`·`error`·`nfev`)을 입히고, 남은 질문(docs/design/integrate.md §9)은 그 추천안을 넣은 안이다. 함수는 마일스톤 C3–C7에서 공개된다(docs/design/integrate.md §8).

- 설계와 결정의 근거는 [docs/design/integrate.md](../design/integrate.md)(§1.1 C1–C123), 미분은 [docs/api/differentiate.md](differentiate.md), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록에는 공개 이름과 시그니처만 둔다(나중의 `test_api_surface.py`가 읽는다).

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- 함수 이름과 인자 이름은 scipy의 것이다. 다른 영역에 이미 있는 꼴이 먼저다: 결과 이름은 `converged`·`n_iter`·`nfev`(C4, O4), 반복 상한은 `max_…`(O14, C12: `max_iter`, `max_level`·`min_level`, `max_terms`, `max_p1`; `limit`·`limlst`는 그대로), 허용오차는 어디서나 `atol`·`rtol`(C11; scipy `quad` 계열의 `epsabs`·`epsrel`, `[scipy와 다름]`), 값 필드는 `integral`·`error`·`nfev`(C13).
- **피적분함수와 데이터는 위치로만, 옵션은 키워드로만** 받는다. 적분의 끝 `a`·`b`와 안쪽 끝의 함수 `gfun`·`hfun`·`qfun`·`rfun`, 표본 `y`도 위치다. 표본 규칙의 `x`는 위치와 키워드 모두다(§8; O112의 `bracket_root(xr0)`과 같은 꼴).
- **별칭이 없다**: `cubature(rule='gauss-kronrod')`(scipy에서 `'gk21'`과 같다), scipy의 옛 이름(`simps`, `trapz`, `cumtrapz`), 대소문자만 다른 이름을 받지 않는다.
- **`args`는 원소별 반복 무리(`tanhsinh`, `nsum`)에서만** 받는다(C4, O39). `quad`·`dblquad`·`tplquad`·`nquad`·`quad_vec`·`cubature`·`fixed_quad`의 `args`는 없다(O12, `[scipy와 다름]`): 상수는 클로저나 `lambda`로 넘긴다. `kwargs`는 어디에도 없다.
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1, O114): `quad(): unexpected keyword argument 'epsabs'`.

### §1.2 사용자 함수
- Python 호출 가능한 객체다. 반복은 C++가 돌고 함숫값은 콜백으로 받는다. 사용자 함수가 낸 에러는 그대로 나간다.
- **`quad`·`dblquad`·`tplquad`·`nquad`**: 점 하나마다 Python `float`로 부르고 실수 하나(Python·numpy의 수, 0차원 배열)를 받는다. 부르는 순서는 QUADPACK의 순서이고 scipy 1.18.1과 같다(점의 비트까지; C5, docs/design/integrate.md §2.3, §7.10). 다른 값은 `quad(): 'func' must return a real number, not shape (3,)`(`ValueError`), bool과 복소수는 `TypeError`다(optimize와 같다). 겹친 적분의 f는 모든 변수를 안쪽부터 받는다(`dblquad`는 `func(y, x)`, `nquad`는 `func(x0, x1, …)`).
  - `quad(vectorized=True)`(C127): 규칙 한 번의 점들(QK21의 21개, 반무한 QAGI의 15개, 실수 전체는 양쪽 30개, 가중치 규칙의 15·25개)을 (n,) float64 `sd.ndarray` 하나로 부르고 같은 모양의 값을 받는다. 값의 읽기와 에러는 `tanhsinh`·`nsum`과 같다: `sd.ndarray`, 실수의 시퀀스, numpy 배열을 받고 float64로 읽으며, 다른 모양은 `quad(): 'func' must return shape (21,) like its argument, not (20,)`(수 하나는 `not a scalar`, `ValueError`), 복소수·bool·문자열 배열은 `TypeError`(`'func' must return real values, not a complex128 array`)다. `complex_func=True`와 함께는 complex128 배열(실수 배열은 허수부 0)을 받는다.
- **`quad_vec`**: 점 하나마다 `float`로 부르고 언제나 같은 모양의 배열(`sd.ndarray`, 실수의 시퀀스, numpy 배열)을 받는다. 모양이 바뀌면 `ValueError`다.
- **`cubature`**: 점들을 (n_points, ndim) float64 `sd.ndarray`로 한 번에 부르고 (n_points, …)를 받는다(scipy와 같다).
- **`fixed_quad`**: n점을 (n,) float64 `sd.ndarray`로 한 번 부르고 (n,) 또는 (…, n)의 실수 값(`sd.ndarray`, 시퀀스, numpy 배열)을 받는다(scipy와 같다). 다른 모양은 `fixed_quad(): 'func' must return shape (5,) or (..., 5) for its argument of shape (5,), not (4,)`(`ValueError`), 복소수·bool 값은 `TypeError`다.
- **`tanhsinh`, `nsum`**: 아직 끝나지 않은 원소의 점을 모아 부른다(O38). 넘기는 꼴은 (k, p) 2차원 float64 `sd.ndarray`이고 `args`는 (k, 1)이다(C26). 첫 호출은 (k,)이다(O48). 사용자 함수는 원소별이어야 하고(`f(x)[i]`가 `f(x[i])`), 끝점 바로 옆이나 끝점 자체(반올림)를 받을 수 있으므로 특이점에서 예외를 내지 말고 inf·NaN을 돌려줘야 한다(그 값은 버린다; scipy와 같다).
- 피적분함수의 값은 실수다. 복소수는 `quad(complex_func=True)`만 받는다(docs/design/integrate.md §1.1 C56).

### §1.3 결과 `IntegrateResult` (C9, C10, C13)
```
IntegrateResult
```
- 모든 함수가 이 클래스를 돌려준다(표본 규칙, `fixed_quad`, `newton_cotes` 빼고: 값이나 배열이다). 규칙은 `sd.optimize.OptimizeResult`와 같다: 함수가 주는 필드만 있고 없는 필드는 `AttributeError`, 필드는 읽기 전용, 배열 필드는 읽기 전용이며 읽을 때 복사하지 않는다(O103, API.md §2.2), `repr`는 한 줄, dict처럼 쓰지 않는다.
- `quad`·`dblquad`·`tplquad`·`nquad`·`quad_vec`의 결과는 scipy의 `(y, abserr)`처럼 둘로 풀린다: `integral, error = sd.integrate.quad(f, 0, 1)`(htest H7의 규칙). `r[0]`, `r[1]`, `len(r) == 2`도 된다. 다른 필드는 이름으로 읽는다. scipy의 `full_output`은 없다(필드가 늘 있다).
  - 다른 함수의 결과(`tanhsinh`, `nsum`, `cubature`)와 `IntegrateResult(...)`로 직접 만든 결과는 풀리지 않는다: `TypeError: this IntegrateResult does not unpack (only the quad family's results do): read its fields by name`. `bool(r)`은 언제나 참이다.
- 공개 생성자 `IntegrateResult(*, batch=(), **fields)`는 `OptimizeResult`·`DifferentiateResult`와 같다(C121): 필드는 키워드로만, 모르는 이름은 `TypeError`(`IntegrateResult(): unexpected keyword argument 'estimate'`), 위치 인자도 클래스 이름으로 시작하는 `TypeError`다(`IntegrateResult(): takes 0 positional arguments but 1 was given`, API.md §2.1). 넘긴 `sd.ndarray`·numpy 배열·수의 리스트는 결과 자신의 읽기 전용 `sd.ndarray`로 한 번 복사하고(numpy 0차원 배열은 그 수), 튜플은 원소마다 그렇게 한다. 공통 필드의 종류를 검사한다: `message`는 str, `converged`는 bool이나 bool 배열, 개수와 번호(`status nfev max_level n_intervals cycle_status n_cycles subdivisions n_regions`)는 int나 정수 배열, 나머지는 수나 수의 배열(str·bool이 아님)이다(`IntegrateResult(): 'integral' must be a number or an array of numbers, not str`, `TypeError`). 없는 필드를 읽으면 `AttributeError: 'IntegrateResult' object has no attribute 'sum' (the function does not give it)`다. 풀리는 것이 `quad` 계열뿐인 것은 C21이다.
- `summary()`(C20, C69)는 API.md §2.3의 상자다: 제목 `Integration Results`, 머리에 `Converged`·`Status`와 `Integral`·`Error estimate`·`Function Evals`·`Intervals`, 넓은 줄 `Message`(줄바꿈을 한 칸으로), `quad`의 구간 표(열 `a b integral error`, 오차 추정이 큰 것부터; 20개를 넘으면 앞뒤 5개), 복소수 결과는 `real`·`imag` 두 줄의 표다. 배열 값(`quad_vec`, `cubature`)은 머리에 `shape (3,)`을 적고 원소마다 한 줄의 표(열 `integral`, `cubature`는 `error`도)를 더한다. `quad_vec`의 배열 값은 구간 표에 `integral` 열이 없고, `cubature`의 영역 표는 열 `a b integral error`(배열 값이면 `a b max error`)다(C106). 배치는 줄마다 한 원소다. `repr`는 `IntegrateResult(integral=1.71828, error=1.90768e-14, converged=True, nfev=21)`(복소수는 `(0.5+1j)`)다.

| 필드 | 뜻 | 주는 함수 |
|---|---|---|
| `integral` | 적분값 | 적분 함수 모두(`nsum` 빼고) |
| `sum` | 급수의 값 | `nsum` |
| `error` | 절대 오차의 추정(엄밀한 상계가 아니다) | 모두 |
| `converged` | 멈춤 판정을 통과했는지 | 모두 |
| `status` | 멈춘 이유의 번호(scipy의 번호, O17) | 모두 |
| `message` | 멈춘 이유의 글(scipy의 글; `quad` 계열의 0은 QUADPACK 문서의 글, C60; `cubature`는 `quad_vec`의 글, C94) | `quad` 계열, `quad_vec`, `cubature` |
| `nfev` | 사용자 함수의 값을 계산한 점의 수 | 모두 |
| `max_level` | 마지막 수준(scipy `maxlevel`) | `tanhsinh` |
| `intervals`, `interval_integrals`, `interval_errors`, `n_intervals` | 쓴 구간 (n, 2)과 구간마다의 값·오차 (n,)(`quad_vec`의 배열 값은 (n, …)), 구간 수 n(scipy `infodict`의 `alist blist rlist elist last`, `quad_vec`의 `full_output`; 오차 추정이 큰 것부터, C55, C95) | `quad`(QAWF 빼고, 실수), `quad_vec` |
| `cycle_integrals`, `cycle_errors`, `cycle_status`, `n_cycles` | QAWF의 주기마다의 값·오차·status (n,)와 주기 수 n(scipy `infodict`의 `rslst erlst ierlst lst`, C72) | `quad`(`weight='cos'`·`'sin'`의 무한 구간) |
| `real`, `imag` | 실수부와 허수부의 `quad` 결과(`IntegrateResult`; scipy `full_output`의 `real`·`imag`, C57) | `quad(complex_func=True)` |
| `subdivisions` | 나눈 수 | `cubature` |
| `regions`, `region_integrals`, `region_errors`, `n_regions` | 영역의 모서리 (n, 2, ndim)과 영역마다의 값·오차 (n,) 또는 (n, …), 영역 수 n(scipy `CubatureResult.regions`; 최대 오차가 큰 것부터, C95) | `cubature` |

- 원소별 반복 무리(`tanhsinh`, `nsum`)의 필드는 입력의 배치 모양이고(`integral` float64, `converged` bool, `status`·`nfev`·`max_level` int64), 입력이 모두 실수면 Python 수다(O48). `repr`는 `IntegrateResult(batch shape (1000,), 998 of 1000 converged)`다.

### §1.4 멈춤, 상한, 경고
- **모든 호출은 정해진 시간 안에 끝난다.** 상한에 닿으면 마지막 값과 `converged=False`다. 예외가 아니다.
- **경고를 내지 않는다.** scipy가 `IntegrationWarning`으로 알리는 것은 `converged=False`와 `status`·`message`다. scipy의 `show`(`romb`)는 없다(화면에 쓰지 않는다, `[scipy와 다름]`).
- 오차 추정은 경험적이다: 수렴이라고 알려도 실제 오차가 추정보다 클 수 있다. 좁은 봉우리를 처음 규칙이 놓치면 `quad`가 틀린 값을 수렴으로 알린다(docs/design/integrate.md §7.4). 어려운 곳의 자리를 알면 `points`로 준다. 봉우리의 너비가 구간의 2⁻³⁸ 근처로 아주 좁으면(QUADPACK 책의 4⁻ᵅ/((x − π/4)² + 16⁻ᵅ), α = 19: 너비 3.6×10⁻¹²) `points=[π/4]`나 봉우리 둘레의 작은 구간도 status 5로 끝나고 값이 틀린다. 그때는 변수를 바꿔 봉우리를 넓힌다: x = π/4 + 4⁻ᵅ sinh t로 바꾸면 피적분함수가 1/cosh t가 되어 `quad`가 수렴한다(참값과 1.3×10⁻¹⁵). `tanhsinh`은 봉우리에서 나눠 불러도 status −2다(docs/design/integrate.md §3.1).
- 적분이나 오차 추정이 유한하지 않으면 수렴이 아니다: `quad` 계열은 status −3 `Non-finite values encountered.`(C119), `quad_vec`·`cubature`는 3, `tanhsinh`·`nsum`은 −3이다.
- 같은 기계, 같은 스레드 수에서는 실행마다 같은 비트다(API.md §3). 스레드 수와 무관한 비트는 약속이 아니지만, 이 영역은 지금 결과가 스레드 수와 무관하다.

### §1.5 에러
- 공개 함수의 에러는 `quad(): …`로 시작한다(API.md §2.1). 끝: `a`, `b`는 실수(±inf 받음; NaN은 `ValueError`, 원소별 무리는 그 원소의 status −3)다. 옵션의 검사와 글은 다른 영역과 같은 꼴이다(`'limit' must be an int, not float`, `'rtol' must be a finite number >= 0`).
- 쓰이지 않는 옵션을 함께 주면 `ValueError`다(O29·O62의 규칙): `quad(): 'points' is not used with 'weight'`(scipy는 경고하고 무시한다, `[scipy와 다름]`), `quad(): 'wvar' is only used with 'weight'`.

### §1.6 스레드와 GIL
- 사용자 함수가 Python이라 GIL을 쥔 채 부른다. `quad` 계열은 호출 사이의 일이 작아 GIL을 놓지 않는다.
- 원소별 무리는 원소를 지나가는 패스의 일(진행 중인 원소 수 × 원소마다의 점 수)이 2¹⁵점 이상이면 그 패스를 스레드에 나누고 GIL을 놓는다(C111: optimize의 배치는 원소마다 점 하나라 같은 기준이다). 표본 규칙은 배열 리덕션의 기준이다(API.md §3).

## §2. 적응 구적 `quad` (C4, C5)
```
quad(func, a, b, /, *, atol=1.49e-08, rtol=1.49e-08, limit=50, points=None, weight=None, wvar=None, max_p1=50, limlst=50, complex_func=False, vectorized=False)
```
- ∫ₐᵇ func(x) dx를 QUADPACK으로 계산한다. 루틴은 scipy처럼 입력이 고른다: 유한 구간 QAGS, 무한 구간 QAGI, `points` QAGP(C4); `weight` QAWO·QAWF·QAWS·QAWC(C5). 표는 docs/design/integrate.md §3.1이다. 점과 결과(값, 오차, `nfev`, status, 구간)는 scipy 1.18.1과 비트까지 같다(시험 모음 116개, 가중치 스윕 99개 가운데 98개와 만든 400개; docs/design/integrate.md §7.10, §7.11). 그 둘이 다른 곳: QAWO·QAWF(`weight='cos'`·`'sin'`)의 `error`는 Clenshaw–Curtis 추정의 반올림 바닥(아래의 정직함, C123) 때문에 scipy보다 작지 않고, 바닥이 경로를 바꾸면(만든 400개 가운데 하나) 점과 status도 다르다. 적분이나 오차가 유한하지 않은 정상 끝은 status −3이다(아래, C119). 가중치 길에서 scipy와 다른 것은 그 밖에 아래의 셋이다.
- `a`, `b`: 실수, ±inf를 받는다(NaN은 `ValueError`). a > b면 scipy처럼 끝을 바꿔 계산하고 부호를 바꾼다(`error`는 그대로). a == b면 0, `nfev` 0이고 f를 부르지 않는다(무한한 같은 끝도). 옵션의 검사는 그래도 한다(scipy는 그 전에 0을 돌려준다).
- `atol`·`rtol`(scipy `epsabs`·`epsrel`, 기본 1.49×10⁻⁸): |I − 값| ≤ max(atol, rtol·|I|)를 목표로 한다. 둘 다 유한한 0 이상의 수다(scipy는 음수 `epsabs`도 받는다). `atol`이 0이면 `rtol`은 50ε(1.11×10⁻¹⁴) 이상이어야 한다(QUADPACK `ier = 6`의 조건; `quad(): if 'atol' is 0, 'rtol' must be at least 50 * machine epsilon (1.11e-14)`).
- `limit`(기본 50): 구간 수의 상한, 1 이상 2³¹ − 1 이하의 int. 닿으면 status 1. 큰 `limit`도 쓴 구간만큼만 메모리를 쓴다.
- `points`: None 또는 구간 안의 중단점들(실수의 시퀀스, `sd.ndarray`·numpy 배열; 수 하나는 `TypeError`, NaN은 `ValueError`). scipy처럼 정렬하고 겹친 점을 빼고 (a, b) 안의 것만 쓴다(빈 목록이나 모두 밖이어도 QAGP다). 무한 구간과 함께는 `ValueError`(`quad(): 'points' cannot be used with an infinite range`), 남은 점의 수가 `limit` 이상이어도 `ValueError`다.
- `weight`와 `wvar`(C5): `'cos'`·`'sin'`(wvar = ω: cos(ωx)·sin(ωx); 유한 구간은 QAWO, 반무한은 QAWF, 양쪽 무한은 `ValueError`), `'alg'`·`'alg-loga'`·`'alg-logb'`·`'alg-log'`(wvar = (α, β), α, β > −1: (x − a)^α (b − x)^β와 그 로그), `'cauchy'`(wvar = c: 1/(x − c)의 주값; c는 a, b가 아니어야 한다). 가중치는 유한 구간만 받는 것이 scipy와 같다(QAWF 빼고). `max_p1`(scipy `maxp1`, 기본 50, 1 이상 2³¹ − 1 이하): QAWO·QAWF의 체비쇼프 모멘트를 기억하는 수준 수의 상한. `limlst`(기본 50, 3 이상 2³¹ − 1 이하): QAWF의 주기 수의 상한. scipy의 `wopts`(미리 계산한 모멘트)는 없다(C71, `[scipy와 다름]`).
  - 검사(C73): `wvar`는 cos·sin·cauchy에서 유한한 실수(`quad(): 'wvar' must be an int or a float, not NoneType`, `'wvar' must be finite for weight='sin'`), alg에서 유한한 실수의 쌍(`'wvar' must be a pair (alpha, beta) of real numbers`, `'wvar' (alpha, beta) must both be > -1 for weight='alg'`)이다. `points`와 함께는 `ValueError`(`quad(): 'points' is not used with 'weight'`), alg의 `limit`은 2 이상이다. 쓰이지 않는 `max_p1`(cos·sin이 아닐 때)·`limlst`(QAWF가 아닐 때)는 `ValueError`다(C61). 범위에 달린 것(`weight='cos' cannot be used on (-inf, inf)`, `weight='alg' cannot be used with an infinite range`, `'wvar' must not equal 'a' or 'b' with weight='cauchy'`)은 a == b의 0 다음에 검사한다.
  - QAWF는 `atol`만 쓴다(C74): `atol` > 0이어야 하고 기본값이 아닌 `rtol`은 `ValueError`다(`quad(): 'rtol' is not used with weight='cos' on an infinite range (QAWF uses 'atol' only)`; scipy는 조용히 버린다, `[scipy와 다름]`). (−∞, b]는 scipy처럼 x → −x로 바꾸고 f를 −x에서 부른다.
  - scipy와 다른 셋(`[scipy와 다름]`; scipy가 틀린 값을 내는 곳): `max_p1`의 모멘트를 다 쓴 뒤의 구간(scipy는 수준 0의 모멘트를 써 틀린 값을 수렴으로 알린다; 기본 50에서는 |ω|(b − a) > 2⁵¹일 때만, C77), QAWF의 ω = 0(cos는 [a, ∞)의 QAGI, scipy는 a와 상관없이 [0, ∞), C78), |ω| ≥ 2³⁰의 QAWF(scipy의 주기가 32비트로 넘친다, C82).
  - 정직함(C83, C123): 수렴한 결과의 참 오차는 오차 추정 + 8ε|I| 안이다(시험 모음, 가중치 스윕 99개, cos·sin의 ω = 1.37·2ᵏ, k = 12–56). QAWO·QAWF의 Clenshaw–Curtis 추정은 반올림 바닥 ε·resabs(체비쇼프 계수의 절댓값 합, ∫|f|의 어림) 아래로 내려가지 않는다(`[scipy와 다름]`: scipy에는 바닥이 없다). 상쇄가 큰 적분(|I| ≪ ∫|f|, 큰 |ω|)에서는 값에 ε·∫|f| 크기의 반올림이 남는데, 바닥이 없으면 ω = 2²⁰ 근처부터 추정이 참 오차의 1/13–1/(4×10⁸)이었다. 그래서 QAWO·QAWF의 `error`는 scipy보다 작지 않고(가중치 스윕의 cos·sin 35개 가운데 QAWO 9개가 더 크다), 바닥이 판정을 바꾸는 입력은 status와 구간 수가 scipy와 다르다(만든 400개 가운데 하나: rtol 10⁻¹⁴에서 status 1). QAWC의 추정에는 바닥이 없어 다항식의 주값은 추정 0이고, 추정만으로 참 오차를 덮지 못하는 것(9개, 모두 5ε|I| 이하)은 QAWC뿐이다.
- `complex_func=True`(C56, C57): func가 복소수를 돌려주면 scipy처럼 실수부의 `quad`를 마친 뒤 허수부의 `quad`를 한다(같은 점들을 두 번 부른다; 실수 값은 허수부 0). `integral`·`error`는 complex(scipy의 re + 1j·im), `converged`는 둘 다, `status`·`message`는 실수부가 0이 아니면 그것, 아니면 허수부의 것, `nfev`는 합이고, 두 결과는 `real`·`imag` 필드다(구간 정보는 거기에 있다).
- `vectorized=True`(C127, `[scipy와 다름]`: scipy에 없는 인자; 이름은 htest의 것): func를 규칙 한 번에 한 번, 그 규칙의 점들을 1차원 float64 `sd.ndarray`로 부른다(§1.2). 점, 점의 순서, 더하는 순서가 `vectorized=False`와 같아, func가 점마다 같은 값을 내면 결과(값, 오차, `nfev`, status, 구간·주기, `complex_func`의 두 결과)가 비트까지 같다. `nfev`는 그대로 점의 수다(호출 수가 아니다). True·False만 받는다(`quad(): 'vectorized' must be True or False, not int`). func는 원소별이어야 한다(`f(x)[i]`가 `f(x[i])`). 호출 사이의 Python 비용이 규칙의 점 수만큼 줄어, 배열 연산으로 쓴 피적분함수는 점마다 부르는 것보다 몇 배 빠르다(`bench/integrate/speed.md`). `dblquad`·`tplquad`·`nquad`에는 없다(docs/design/integrate.md C127).
- **status와 message** (O17): QUADPACK의 `ier`와 scipy의 글이다(scipy의 `"\n  "` 줄바꿈 그대로, C59). 0 수렴(`Normal and reliable termination of the routine. …`, QUADPACK 문서의 글, C60), 1 `The maximum number of subdivisions (50) has been achieved. …`, 2 반올림, 3 나쁜 피적분함수, 4 외삽 표가 수렴하지 않음, 5 발산 또는 느린 수렴, 7 비정상 끝(QAWF는 1·4·7이 주기의 글이다: `The maximum number of cycles allowed has been achieved. …` 등, scipy의 것, C72). 6(입력이 틀림)은 결과가 아니라 `ValueError`다(위의 글). −3 `Non-finite values encountered.`(C119, `[scipy와 다름]`): QUADPACK이 0으로 끝났는데 적분이나 오차가 유한하지 않다(`quad(lambda x: math.inf, 0, 1)`, 넘치는 구간 `quad(lambda x: 1.0, -1.7e308, 1.7e308)`; scipy는 ier 0이다). 번호는 `tanhsinh`·`nsum`의 것, 글은 `quad_vec`의 것이다. 0이 아닌 QUADPACK의 번호는 값이 유한하지 않아도 그대로 둔다. `dblquad`·`tplquad`·`nquad`도 같다. `converged`는 `status == 0`이다.
- 결과: `integral error converged status message nfev`와 구간 정보 `intervals`((n, 2)), `interval_integrals`, `interval_errors`((n,)), `n_intervals`(C55; QAWO·QAWS·QAWC도 같고, QAWO가 첫 규칙에서 끝나면 그 구간 하나다(scipy `last` 0, C80)). QAWF는 구간 대신 주기마다의 `cycle_integrals`, `cycle_errors`(float64), `cycle_status`(int64, 주기의 QAWO status), `n_cycles`다(scipy의 `rslst erlst ierlst lst`, 만든 순서; C72). 구간은 오차 추정이 큰 것부터다: QUADPACK의 `iord`가 순위를 매긴 것(n ≤ `limit`/2 + 2면 모두, 아니면 `limit` + 1 − n개) 다음에 나머지를 오차의 내림차순으로. 무한 구간(QAGI)의 구간은 scipy처럼 바꾼 변수 t ∈ (0, 1]의 것이다(x = a + (1 − t)/t, x = b − (1 − t)/t, 실수 전체는 ±(1 − t)/t). 오차 추정 아래로 수렴한 경우 구간 값의 합이 `integral`이고, 외삽(ε-알고리즘)으로 끝났으면 `integral`은 외삽 값이다. 둘로 풀린다.

## §3. 겹친 적분 (C4)
```
dblquad(func, a, b, gfun, hfun, /, *, atol=1.49e-08, rtol=1.49e-08)
tplquad(func, a, b, gfun, hfun, qfun, rfun, /, *, atol=1.49e-08, rtol=1.49e-08)
nquad(func, ranges, /, *, opts=None)
```
- `dblquad`: ∫ₐᵇ ∫_{g(x)}^{h(x)} func(y, x) dy dx(scipy처럼 안쪽 변수가 먼저). `gfun`·`hfun`은 수 또는 x의 함수다(바깥 점마다 `gfun` 다음 `hfun`을 부른다; scipy와 같다). `tplquad`: ∫ₐᵇ ∫_{g(x)}^{h(x)} ∫_{q(x, y)}^{r(x, y)} func(z, y, x) dz dy dx(`qfun(x, y)`·`rfun(x, y)`은 scipy의 인자 순서). 둘 다 모든 층에 같은 `atol`·`rtol`, `limit` 50이다. 함수가 돌려준 끝은 실수여야 한다(NaN은 `dblquad(): 'hfun' returned NaN`).
- `nquad`: `ranges`는 변수마다(안쪽부터) `(lo, hi)` 쌍 또는 바깥 변수들을 가까운 것부터 받아(`ranges[i](x_{i+1}, x_{i+2}, …)`; scipy와 같다) 쌍을 돌려주는 함수다. `opts`는 None, dict 하나(모든 층), 또는 층마다의 dict·함수(범위 함수와 같은 인자)의 시퀀스(`ranges`와 길이가 같아야 한다)이고, 키는 `quad`의 옵션 이름(`atol rtol limit points weight wvar`; scipy의 `epsabs`·`epsrel`이 아니다)이다. 모르는 키는 `ValueError`(`nquad(): 'opts' has no option 'epsabs' (scipy's 'epsabs' is 'atol'); …`, optimize O16의 규칙; scipy는 `TypeError`)다. 함수가 돌려준 범위와 옵션도 같은 규칙으로 검사한다. `weight`·`wvar`도 층마다 받는다(C5; 범위 함수가 돌려준 범위에 대한 검사도 층마다).
- 안쪽 적분은 바깥 점마다 `quad`로 계산한다(scipy와 같은 순서, f를 부르는 점의 비트까지; 모든 층이 C++ 안에서 돈다, C64). `error`는 scipy처럼 모든 층의 모든 `quad`가 낸 오차 추정의 최댓값이다(안쪽 적분의 오차를 바깥에 더하지 않으므로 엄밀한 상계가 아니다; C63). `nfev`는 f를 부른 수(가장 안쪽 `quad`들의 합, scipy의 `neval`)다. `converged`는 모든 `quad`가 수렴했을 때이고, `status`·`message`는 끝나는 순서로(안쪽이 먼저) 처음 수렴하지 못한 `quad`의 것이다(scipy는 그 `quad`마다 경고한다). 결과에 구간 정보는 없다(scipy `nquad`와 같다).
- scipy `nquad`의 `args`와 `full_output`은 없다.

## §4. 벡터 값 함수 `quad_vec` (C7)
```
quad_vec(f, a, b, /, *, atol=1e-200, rtol=1e-08, norm='2', limit=10000, points=None, quadrature=None)
```
- 수 또는 배열 값 f의 ∫ₐᵇ f(x) dx를 scipy의 전역 적응 구적으로 계산한다(scipy `_quad_vec.py`의 알고리즘: 오차가 큰 구간을 한 번에 128개까지 반으로 나눈다). 점, 값, 오차, `nfev`, status, 구간은 scipy 1.18.1과 비트까지 같다(기록 2개, 기록 밖의 경우 30개, 모음 106개와 무리 셋; docs/design/integrate.md §7.13). 둘로 풀린다(`integral, error = sd.integrate.quad_vec(f, 0, 1)`).
- f는 점 하나마다 Python `float`로 불리고 실수 하나(Python·numpy의 수, 0차원 배열) 또는 실수 배열(`sd.ndarray`, numpy 배열, 시퀀스)을 돌려준다. 첫 값이 꼴을 정한다: 수면 `integral`은 Python 수, 배열이면 그 모양의 읽기 전용 `sd.ndarray`이고 다른 꼴이 오면 `ValueError`(`quad_vec(): 'f' must return the same shape at every point, not (3,) and then (2,)`), 빈 배열도 `ValueError`다. 값은 float64로 계산한다(scipy는 float32 배열을 float32로 계산한다, `[scipy와 다름]`, C104).
- `a`, `b`: 실수, ±inf를 받는다(NaN은 `ValueError`). 무한 구간은 scipy의 치환이다: [a, ∞)는 x = a + (1 − t)/t, (−∞, b]는 x = b − (1 − t)/t(t ∈ (0, 1]), 실수 전체는 x = (1 − |t|)/t(t ∈ [−1, 1], t = 0에서 나눈다). |t| < √(DBL_MIN)인 점은 f를 부르지 않고 0으로 둔다(scipy와 같다). a > b면 그대로 계산한다(유한한 끝은 반대 방향의 규칙, 무한한 끝은 부호). a == b면 scipy처럼 f를 그 점(무한한 같은 끝은 x = 0)에서 부르고 0이다.
- `atol`·`rtol`(scipy `epsabs`·`epsrel`, 기본 10⁻²⁰⁰·10⁻⁸): 유한한 0 이상의 수다. 오차 ≤ max(atol, rtol·‖I‖)/8에서 멈춘다(scipy의 판정). `norm`(기본 `'2'`): 오차를 재는 노름 `'2'`(numpy `linalg.norm`) 또는 `'max'`; 값이 수면 둘 다 |·|다. `limit`(기본 10⁴): 구간 수의 상한, 1 이상의 int(한 번에 128개까지 나누므로 끝의 구간 수는 `limit`을 조금 넘을 수 있다). `points`: 나눌 점(실수의 시퀀스; NaN은 `ValueError`). 정렬해 (a, b) 안의 것만 쓰고, 무한 구간이면 scipy처럼 바꾼 변수 t로 옮긴다. `quadrature`: None(유한 구간 `'gk21'`, 무한 구간 `'gk15'`), `'gk21'`, `'gk15'`, `'trapezoid'`.
- 결과: `integral error converged status message nfev`와 구간 정보 `intervals`((n, 2)), `interval_integrals`((n,), 배열 값이면 (n, …)), `interval_errors`((n,)), `n_intervals`(C55의 이름, C95). 구간은 오차 추정이 큰 것부터(scipy의 힙이 꺼내는 순서)이고 무한 구간이면 바꾼 변수 t의 것이다. `interval_integrals`는 `integral`에 더해지는 몫(치환의 부호를 곱한 값)이다. scipy처럼 구간 값을 100×10⁶바이트(첫 값의 `sys.getsizeof`로 센 개수)까지만 기억하고 넘치면 오래된 것부터 버린다: 버린 구간을 나눌 때는 다시 계산하고(f를 21·15번 더 부른다; scipy와 같다), 결과의 그 칸은 NaN이다(C98).
- `nfev`는 scipy의 `neval`이다: 규칙의 점마다 하나(t가 너무 작아 0으로 둔 점도 센다), 사다리꼴은 나누기 하나 안에서 같은 점을 한 번만 센다(scipy의 `lru_cache`).
- status와 message(scipy의 번호와 글): 0 `Target precision reached.`, 1 `Target precision not reached.`(`limit`), 2 `Target precision could not be reached due to rounding error.`, 3 `Non-finite values encountered.`. `converged`는 `status == 0`이다.
- scipy의 `cache_size`, `workers`, `full_output`, `args`는 없다(`[scipy와 다름]`, C96).

## §5. 다차원 `cubature` (C7)
```
cubature(f, a, b, /, *, rule='gk21', rtol=1e-08, atol=0.0, max_subdivisions=10000, points=None)
```
- 상자 [a, b] 위의 배열 값 f의 적분을 scipy의 적응 구적으로 계산한다(scipy `_cubature.py`: 최대 오차가 가장 큰 영역을 축마다 반으로 나눠 2^ndim개로). 점, 값, 오차, `subdivisions`, 영역은 scipy 1.18.1과 비트까지 같다(기록 12개, 기록 밖의 경우 39개, 1차원 모음의 105개를 규칙 둘로 210개, 무리 셋을 규칙 둘로 여섯; 아래의 다른 셋 빼고; docs/design/integrate.md §7.13). 풀리지 않는다(C21).
- f는 영역마다 두 번 불린다: 규칙의 점(n, ndim)과, 오차 추정을 위한 두 규칙의 점(높은 규칙의 점 다음 낮은 규칙의 점) 모두다(scipy와 같다). 넘기는 것은 float64 `sd.ndarray`이고 그 배치는 scipy의 것이다: `'gk21'`·`'gk15'`는 C 순서, `'genz-malik'`는 Fortran 순서(scipy가 점의 표를 전치한다). f는 (n,) 또는 (n, …)을 돌려준다: (n,)이면 `integral`·`error`가 Python 수, 아니면 뒤 모양의 읽기 전용 배열이고, 뒤 모양이 호출마다 같아야 한다(`ValueError`). 점들에 따라 더하는 합은 numpy의 것이다: 점의 축이 메모리에서 가장 안쪽인 값(점마다 값 하나, Fortran 순서의 값)은 pairwise, 아니면 점의 차례대로(C100).
- `a`, `b`: 길이 ndim ≥ 1의 실수 시퀀스(`sd.ndarray`, numpy 배열, 리스트)이고 길이가 같아야 한다(scipy는 수 하나를 다른 쪽에 맞춰 늘린다, `[scipy와 다름]`). ±inf를 받고 NaN과 같은 무한(a_i = b_i = ±inf)은 `ValueError`다(scipy는 그 축에서 f를 inf에서 부른다). a_i > b_i면 그 축을 바꾸고 부호를 바꾼다. 무한 축은 scipy의 치환이다: 실수 전체 x = (1 − |t|)/t(t ∈ (−1, 1)), [a, ∞) x = a + (1 − t)/t, (−∞, b] x = b − (1 − t)/t(t ∈ (0, 1)); 무한 축이 있으면 t = 0(원점)에서 처음 영역을 나눈다.
  - (−∞, b]는 scipy 문서의 치환이다: scipy의 코드는 f를 −b + (1 − t)/t에서 불러 [−b, ∞)를 적분한다(∫₋∞⁰ eˣ에 inf, `[scipy와 다름]`, C101).
- `rule`: `'gk21'`(기본), `'gk15'`(1차원 Gauss–Kronrod 규칙의 곱: 점 21^ndim·15^ndim개, 낮은 규칙은 scipy의 `roots_legendre(10)`·`(7)` 값 그대로, C99), `'genz-malik'`(차수 7과 5, 점 2^ndim + 2ndim² + 2ndim + 1개; ndim ≥ 2).
  - 영역 하나의 점(오차 추정의 호출이 받는 두 규칙의 점: gk21 21^d + 10^d, gk15 15^d + 7^d, genz-malik 2^d + 4d² + 4d + 2)이 가진 값(점 수 × d)이 2²⁵개를 넘으면 f를 부르기 전에 `ValueError`다(C117): `cubature(): rule='gk21' in 8 dimensions needs 21**8 + 10**8 points per region of 8 values each, more than 2**25 values; use rule='genz-malik'`. 그래서 점의 배열 하나가 256 MiB를 넘지 않고, gk21·gk15는 5차원, genz-malik은 20차원까지다. scipy는 상한이 없다(gk21의 8차원은 점의 표만 2 TB다). 영역의 점은 점마다 d개의 double을 몇 벌 두고(규칙의 표, 두 규칙을 합친 표, f에 넘기는 배열과 f의 중간 배열), 잰 최대 RSS는 gk21 5차원(418만 점, 2.1×10⁷값)에서 1.2 GB, 값마다 대략 7 × 8바이트다. `points`가 만드는 처음 영역의 수 × 영역의 점이 2²⁴을 넘어도 `ValueError`다(`cubature(): 'points' split the box into more than 11444 regions of 1466 points each (more than 2**24 points)`; 점 하나가 그 점을 품은 영역을 2^d개까지 나눈다).
- `rtol`·`atol`(기본 10⁻⁸·0): 유한한 0 이상의 수다. 모든 값에서 error ≤ atol + rtol·|integral|이면 멈춘다(scipy의 판정). `max_subdivisions`(기본 10⁴): 나누기 수의 상한, 1 이상의 int(scipy는 None(상한 없음)과 0(그래도 한 번 나눈다)도 받는다, `[scipy와 다름]`). `points`: 피할 점의 목록(점마다 ndim개의 실수; NaN은 `ValueError`, 무한한 좌표의 점은 쓰지 않는다). 그 점이 영역의 모서리에 오게 처음 영역을 나누고, 폭이 0인 조각은 버린다(scipy와 같다). 무한 축이 있으면 각 좌표를 그 축의 치환으로 옮긴다(scipy는 이중 무한 축 하나만인 경우 말고 에러를 낸다, C102).
- 결과: `integral error converged status message nfev subdivisions`와 영역 정보 `regions`((n, 2, ndim): `regions[k, 0]`이 a, `regions[k, 1]`이 b), `region_integrals`·`region_errors`((n,) 또는 (n, …)), `n_regions`(C95; scipy `CubatureResult`의 `regions` 목록을 배열로). 영역은 최대 오차가 큰 것부터(scipy의 힙이 꺼내는 순서)이고 무한 축이면 바꾼 변수의 것이다. `region_integrals`는 `integral`에 더해지는 몫(뒤집은 축의 부호를 곱한 값)이다. scipy의 `estimate`는 `integral`이고(C13), `atol`·`rtol`은 결과에 두지 않는다. `nfev`는 f를 부른 점의 수다.
- status(C94; scipy의 문자열 `'converged'`·`'not_converged'` 대신, `[scipy와 다름]`): 0 `Target precision reached.`, 1 `Target precision not reached.`(`max_subdivisions`), 3 `Non-finite values encountered.`(멈춘 값이나 오차가 유한하지 않다: scipy는 NaN에서 판정이 거짓이라 `'converged'`라고 알린다, C103). 번호와 글은 `quad_vec`의 것이다. 점이 모두 폭 0인 조각으로 갈리면 f를 부르지 않고 0(수)이다.
- scipy의 `args`와 `workers`는 없다(`[scipy와 다름]`, C96).

## §6. 배치 tanh-sinh와 급수 (C6)
```
tanhsinh(f, a, b, /, *, args=(), log=False, max_level=None, min_level=2, atol=None, rtol=None)
nsum(f, a, b, /, *, step=1, args=(), log=False, max_terms=1048576, atol=None, rtol=None)
```
- 원소별 반복 무리다(C4): `a`, `b`(, `step`)와 `args`의 값을 브로드캐스트한 배치의 원소마다 따로 계산한다. 입력은 수, 수의 시퀀스, `sd.ndarray`, numpy 배열이고(bool은 `TypeError`), 결과 필드는 그 브로드캐스트 모양이며 입력이 모두 수면 Python 수다(O48). 호출 규약은 §1.2다: 첫 호출은 원소마다 점 하나의 (k,)(`tanhsinh`은 구간의 가운데, `nsum`은 `a`), 그다음은 아직 끝나지 않은 원소의 (k, p)이고 `args`는 (k, 1)이다(C26). 빈 배치는 f를 부르지 않는다(optimize의 배치와 같다; scipy는 빈 배열로 한 번 부른다).
- 점, 값, 오차 추정, `nfev`, status, `max_level`은 scipy 1.18.1과 비트까지 같다: 시험 모음의 101개, 배치 셋, 급수 아홉과 기록 밖의 경우(log, 수준, 끝의 변환, 유한하지 않은 값, `nsum`의 갈래) 모두(docs/design/integrate.md §7.12; 예외는 `nsum`의 두 경우, 아래).
- 끝점을 부를 수 있다: 반올림으로 점이 끝에 닿으면(예: [0, ∞)의 변환에서 x = 0) f는 그 점을 받고 그 값은 버려진다(scipy와 같다, C86). 그래서 f는 특이점에서 예외를 내지 말고 inf·NaN을 돌려줘야 한다. f가 낸 예외는 그대로 나간다.
- 오차 추정은 엄밀한 상계가 아니다(scipy와 같은 추정이다): 수렴이라고 알려도 참 오차가 더 클 수 있다. 시험 모음에서 수렴한 76개 가운데 여덟(넓은 봉우리, 빠른 진동)이 그렇고, e^(−cx²)의 [0, 2]를 c ∈ [0.5, 3]의 40,001개로 부르면 1,093개가 그렇다(상대 1.1×10⁻⁸까지; docs/design/integrate.md §7.12).
- **`tanhsinh`**: ∫ₐᵇ f. 끝은 무한이어도 되고(양쪽 무한은 x = t/(1 − t²), 한쪽 무한은 x = 1/t − 1 + a, 왼쪽은 x → −x로 바꾼다) 끝점 특이점을 받는다. 구간 안의 특이점·꺾임은 범위 밖이다(구간을 나눠 부른다). a == b면 0(`max_level` −1, `nfev` 1), b < a면 끝을 바꿔 계산하고 부호를 바꾼다. 끝이 NaN이거나 가운데의 값이 NaN이면 첫 호출 뒤 status −3이다(scipy와 같다).
  - 수준 k의 간격은 h₀/2ᵏ이고 점은 scipy의 표(libm의 cosh·sinh·exp로 만든 것; 같은 식의 정확한 값과 최대 980 ULP 다르다, C84)다. 수준 0은 18점, 수준 k ≥ 1은 8·2ᵏ점이다. `max_level`(scipy `maxlevel`, 기본 10): 수준의 상한, 0 이상 16 이하의 int(C88; 수준 16의 호출은 원소마다 524,288점). `min_level`(scipy `minlevel`, 기본 2): 처음 한 번의 호출에 넣는 수준(수준 0–`min_level`의 점), 0 이상의 int이고 `max_level`보다 크면 `max_level`이다(scipy와 같다). 첫 호출에 그 수준까지의 점을 모두 계산하므로, 그보다 일찍 수렴할 함수에서는 `nfev`와 `max_level`도 커진다(scipy와 같다: [0, 1]의 eˣ는 기본값에서 `nfev` 67·`max_level` 2, `min_level=5`에서 515·5).
  - `atol`(기본 0)·`rtol`(기본 ε^0.75 ≈ 1.82×10⁻¹²): error < atol 또는 error < rtol·|integral|에서 멈춘다. 둘 다 유한한 0 이상의 수다(C90).
  - `log=True`(C85): f는 log(피적분함수)를 돌려주고 `integral`·`error`·`atol`·`rtol`도 log다(기본 atol −inf, rtol 0.75·log ε). 합은 scipy의 logsumexp다. 값은 실수뿐이라(§1.2, C56) 피적분함수는 음이 아니어야 하고(f가 복소수를 돌려주면 `TypeError`), b < a인 원소가 있으면 `ValueError`다(`tanhsinh(): with log=True, 'b' must not be less than 'a' (the log of the negative integral is complex)`; scipy는 log I + πi의 복소수를 돌려준다, `[scipy와 다름]`). `atol`·`rtol`은 inf보다 작은 실수다(−inf는 허용오차 0; NaN은 `ValueError`, scipy는 받는다).
  - 결과: `integral error converged status nfev max_level`. status: 0 수렴, −2 `max_level`, −3 유한하지 않은 적분값(scipy의 번호; −1은 쓰지 않는다). `error`는 수준 2를 마치기 전에 멈추면 NaN이다. 풀리지 않는다(C21).
- **`nsum`**: Σ_{k} f(a + k·step), a ≤ a + k·step ≤ b. f는 매끄럽고 양이며 단봉인 함수여야 한다(실수 전체에서 정의; scipy의 계약). 항 수 ⌊(b − a)/step⌋ + 1이 `max_terms` 이하면 한 번의 호출로 직접 더하고, 아니면 적분 판정의 경계로 나머지를 `tanhsinh`(기본 수준)으로 근사한다: a가 유한하면 오른쪽, a = −∞이고 b가 유한하면 f(−x)의 오른쪽, 양쪽 무한이면 f(x) + f(−x)(x = 0은 반)의 오른쪽 합이다(마지막은 f를 호출마다 두 번 부른다). 갈래마다 따로 부르고 순서는 scipy의 것이다(직접 합, 오른쪽, 왼쪽, 양쪽).
  - `step`(기본 1): 유한한 양수(아니면 그 원소의 status −1). `max_terms`(scipy `maxterms`, 기본 2²⁰): 직접 더하는 항 수의 상한, 0 이상 2²⁴ 이하의 int(C89; scipy는 `1e3` 같은 실수도 받는다). `atol`(기본 0)·`rtol`(기본 √ε)은 scipy `tolerances`의 두 키다(O36). `log=True`는 `tanhsinh`과 같다(기본 atol −inf, rtol 0.5·log ε).
  - 결과: `sum error converged status nfev`. status: 0, −1(`a`·`b`·`step`이 틀림: step ≤ 0, NaN 등), −2(적분의 수준 상한; 발산할 수 있다), −3(유한하지 않은 값), −4(부분합의 마지막 항이 허용오차보다 크다: `max_terms`를 늘리거나 나눠 더한다). b < a면 0이다(status 0). `error`는 모든 항이 음이 아니고 f가 정확하다고 가정한 추정이다(scipy의 것).
  - scipy와 다른 것(docs/design/integrate.md §7.12, C92): `log=True`이고 a = b = −∞인 원소(scipy는 0에서 −∞로의 적분을 거쳐 복소수를 낸다)는 NaN이고 status −2다. `log=True`의 오차(네 항의 복소 logsumexp)는 원소 하나로 부른 scipy와 같고, 원소 여럿의 배치로 부른 scipy와는 합의 순서가 달라 끝자리가 다를 수 있다. 항의 위치 a + n·step가 넘쳐 무한이 되는 원소는 그 직접 합을 비운다(scipy는 에러).
- **메모리**(C120): 원소마다의 상한(`max_level`, `max_terms`)은 원소 하나의 상한이다. 호출 하나는 아직 끝나지 않은 원소 k개의 점을 한 배열 (k, m)으로 받고, m은 그 가운데 가장 긴 원소의 점 수다(짧은 원소의 나머지 칸은 NaN이고 f는 그 칸도 계산한다; scipy와 같은 꼴, C5). 인자는 k·m·8바이트이고 f의 중간 배열은 그 몇 배다. `tanhsinh`의 m은 수준 k에서 8·2ᵏ(수준 16에서 원소마다 4 MiB), `nsum`의 직접 합에서 m은 항 수(2²⁰까지, 원소마다 8 MiB)다. 잰 것: `nsum(lambda k, p: k**-p, 1, inf, args=(sd.linspace(1.5, 4, n),))`의 가장 큰 호출은 n = 1에서 (1, 131,072), n = 40에서 (40, 131,072)(인자 42 MB)이고 최대 RSS는 54 MB·154 MB다. 원소가 많은 배치는 점이 비슷한 원소끼리 나눠 부른다.
- scipy의 `kwargs`, `callback`, `preserve_shape`는 없다(C4, C14, O37; `[scipy와 다름]`).

## §7. 고정 규칙 (C3)
```
fixed_quad(func, a, b, /, *, n=5)
newton_cotes(rn, /, *, equal=False)
```
- **`fixed_quad`**: n점 Gauss–Legendre로 ∫ₐᵇ func. func는 점 (b − a)·(x + 1)/2.0 + a(scipy의 식)의 (n,) 배열을 한 번 받아 (n,) 또는 (…, n)을 돌려준다(§1.2). 돌려주는 것은 값 하나(Python float)이고, f가 (…, n)을 돌려주면 (…) 배열이다(scipy의 `(val, None)` 튜플이 아니다, `[scipy와 다름]`; docs/design/integrate.md §1.1 C43, C52).
  - a, b는 유한한 실수다. inf와 NaN은 `ValueError`다(`fixed_quad(): 'b' must be finite (Gaussian quadrature needs a finite interval)`; scipy는 NaN에 NaN을 돌려준다, `[scipy와 다름]`). a > b면 값의 부호가 바뀌고 a == b면 0이다(f는 그래도 한 번 부른다, scipy와 같다). `n`은 1 이상 10⁴ 이하의 int다(`fixed_quad(): 'n' must be at most 10000`; scipy는 상한이 없다, `[scipy와 다름]`, C109: 표 밖의 점은 처음 한 번 n²에 비례하는 비용이라 상한이 시간의 상한이다). scipy의 `args`는 없다(§1.1).
  - 점과 가중치는 올바로 반올림한 double이다(C42): n ≤ 100은 Arb로 만든 표, 그 위는 실행 중의 정확한 계산(C54; 처음 한 번의 비용은 n²에 비례한다: 1,000점 9 ms, 10⁴점 0.9초(1스레드 6.4초). 최근 8개의 n을 기억한다). 기본 n = 5의 점은 scipy와 같고 가중치는 끝자리가 다르다. n = 20부터는 scipy `roots_legendre`의 점과도 끝자리가 다르다(docs/design/integrate.md §7.9).
  - 합 Σwᵢfᵢ는 sd의 합 규칙이다(§8과 같다).
- **`newton_cotes`**: `rn`이 int N(1 이상 256 이하)이면 같은 간격 N + 1점, 길이 N + 1(2 이상 257 이하)의 실수 시퀀스(데이터 함수의 입력)면 그 상대 위치(0에서 시작해 N에서 끝나고 서로 다른 유한한 값)의 닫힌 Newton–Cotes 가중치 `an`(float64 `sd.ndarray`)과 오차 계수 `B`(float)의 튜플 `(an, B)`다(scipy와 같다). ∫_{x₀}^{x_N} f = Δx·Σaᵢfᵢ + B·Δx^{p+1}·f⁽ᵖ⁾(ξ)이고 p는 같은 간격의 짝수 N이면 N + 2, 아니면 N + 1이다(scipy의 차수).
  - `an`과 `B`는 정확한 유리수를 한 번 반올림한 값이다(C53). 같은 간격 N ≤ 14는 scipy의 표와 비트로 같고, 그 밖(N ≥ 15나 위치를 준 경우)은 scipy의 Vandermonde 풀이보다 정확하다(scipy의 `B`는 N = 20에서 9%, N = 30에서 10⁶배 틀린다; docs/design/integrate.md §7.9). 같은 입력은 기억한다(첫 계산은 N = 8 0.2 ms, N = 256 0.3초; int N의 두 번째부터는 기억한 가중치의 사본이라 0.2 µs, C108). N의 상한 256은 정확한 유리수의 비용(N³에 비례)이 시간의 상한이 되게 한다(C110; scipy는 상한이 없고 N ≥ 15에서는 Vandermonde 풀이라 N = 30의 `B`가 10⁶배 틀린다, `[scipy와 다름]`).
  - `equal=True`면 위치를 보지 않고 길이만 쓴다(scipy와 같다). `equal`은 bool만 받는다(scipy는 int 0·1). 위치의 검사(0에서 시작, N에서 끝남)는 언제나 한다(scipy는 같은 간격 N ≤ 14면 하지 않는다, `[scipy와 다름]`).

## §8. 표본의 규칙 (C3)
```
trapezoid(y, /, x=None, *, dx=1.0, axis=-1)
simpson(y, /, x=None, *, dx=1.0, axis=-1)
romb(y, /, *, dx=1.0, axis=-1)
cumulative_trapezoid(y, /, x=None, *, dx=1.0, axis=-1, initial=None)
cumulative_simpson(y, /, x=None, *, dx=1.0, axis=-1, initial=None)
```
- **입력** (C39, C45, C48): `y`는 데이터 함수의 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, frame의 `Series`; 결측값은 NaN, API.md §5)이고 float64(bool·int64는 바꿔서) 또는 complex128로 계산한다. 1차원 이상이어야 한다(스칼라는 `ValueError`).
  - `x`(위치와 키워드 모두, C40)는 실수이고 `axis`를 따라 y의 길이인 1차원이거나, y와 차원 수가 같고 y의 모양으로 브로드캐스트되며 `axis`의 길이가 y와 같은 배열이다(scipy는 함수마다 받는 꼴이 조금 다르다). `x`가 없으면 간격 `dx`다.
  - `dx`는 수 또는 y의 모양에서 `axis`만 1인 배열(줄마다의 간격; scipy `cumulative_simpson`의 꼴을 다섯 함수에)이다. `x`와 함께 1.0이 아닌 `dx`를 주면 `ValueError`다(`trapezoid(): 'dx' is not used with 'x' (the spacing comes from 'x')`; scipy는 무시한다, `[scipy와 다름]`).
  - `axis`는 int이고 기본은 scipy·numpy의 −1이다(C41). 범위 밖은 `ValueError`(`axis 2 is out of bounds for an array of dimension 2`)다.
- **결과**: `trapezoid`·`simpson`·`romb`은 `axis`를 뺀 모양, 누적은 `axis`의 길이가 하나 준 모양(`initial`이 있으면 그대로)이다. dtype은 y의 것이고 0차원이면 Python 수(float, complex)다. 복소수 y는 실수 계수를 실수부와 허수부에 따로 곱한다(C48; numpy는 계수를 복소수로 올린다: 한쪽이 inf·NaN일 때만 다르다).
- **규칙** (식과 연산 순서는 scipy 1.18.1의 것):
  - `trapezoid`: Σ dᵢ(yᵢ₊₁ + yᵢ)/2. 점이 하나 이하면 0이다.
  - `simpson`: scipy 1.11부터의 꼴(C47). 점이 홀수 개면 합성 Simpson(간격이 다르면 scipy의 식), 짝수 개면 앞 구간들의 Simpson과 마지막 구간의 Cartwright 보정, 점 둘이면 사다리꼴, 하나 이하면 0이다(scipy는 점 0개에 `IndexError`, `[scipy와 다름]`; C49). 0인 간격의 몫은 scipy처럼 0으로 둔다.
  - `romb`: 2ᵏ + 1개(k ≥ 0) 같은 간격 점의 Romberg 적분(scipy의 Richardson 표). 점 수가 아니면 `ValueError`(`romb(): 'y' must have 2^k + 1 samples along 'axis' (k >= 0), not 4`)다. scipy의 `show`(표를 화면에 쓰기)는 없다(§1.4, C44, `[scipy와 다름]`).
  - `cumulative_trapezoid`: 사다리꼴의 누적. `initial`은 None 또는 0이다(맨 앞에 넣는다; 다른 수는 `ValueError`, bool은 `TypeError`; scipy와 같다, C46).
  - `cumulative_simpson`: 구간마다 Cartwright의 식(세 점의 이차식, scipy와 같다)의 누적. 점이 셋보다 적으면 사다리꼴이다. `x`가 있고 점이 셋 이상이면 `x`는 `axis`를 따라 순증가해야 한다(`ValueError`; NaN은 걸리지 않는다, scipy와 같다, C50). `initial`은 수 또는 `dx`와 같은 줄마다의 배열이고 모든 누적값에 더한 뒤 맨 앞에 넣는다(scipy와 같다). 실수 y에 복소수 `initial`은 `TypeError`다.
  - 누적은 점이 하나도 없으면 `ValueError`(`'y' must have at least one point along 'axis'`), 하나면 빈 결과(또는 `initial` 하나)다. 다른 축이 비면 빈 배열이다(C49). NaN·inf는 값으로 퍼진다(C50).
- **합** (C51): 줄(축을 따라간 한 줄)마다 항을 sd의 pairwise 합(docs/api/array.md §7.8: 누산기 16개, 512개 단위)으로 첨자 순서대로 더하고, 누적은 `cumulative_sum`의 스캔이다. 그래서 scipy·numpy와 끝자리가 다를 수 있고(docs/design/integrate.md §7.9: scipy 기록 65개 가운데 59개가 비트로 같고 나머지도 상대 2.6×10⁻¹⁶ 안), 줄의 값은 축·메모리 배치·스레드 수와 상관없다. 일이 2¹⁵ 이상이면 줄이나 한 줄의 부분합을 스레드에 나누고 GIL을 놓는다.
- sd 배열 쪽에는 `trapezoid`가 없다(numpy의 `np.trapezoid`는 이 함수다).

## §9. scipy 이름 → sd 이름
| scipy 1.18.1 | sd | 다른 점 |
|---|---|---|
| `quad(func, a, b, args, full_output, epsabs, epsrel, limit, points, weight, wvar, wopts, maxp1, limlst, complex_func)` | `quad(func, a, b, /, *, atol, rtol, limit, points, weight, wvar, max_p1, limlst, complex_func, vectorized)` | `[scipy와 다름]` `args`·`full_output`·`wopts` 없음, `epsabs`→`atol`, `epsrel`→`rtol`, `maxp1`→`max_p1`, `vectorized` 더함(C127), 결과 객체(둘로 풀림), 경고 대신 status |
| `dblquad`, `tplquad` | 같다 | `[scipy와 다름]` `args` 없음, `atol`·`rtol` |
| `nquad(func, ranges, args, opts, full_output)` | `nquad(func, ranges, /, *, opts)` | `[scipy와 다름]` `args`·`full_output` 없음, `opts`의 키 이름 |
| `quad_vec(..., epsabs, epsrel, norm, cache_size, limit, workers, points, quadrature, full_output, args)` | `quad_vec(f, a, b, /, *, atol, rtol, norm, limit, points, quadrature)` | `[scipy와 다름]` `cache_size`·`workers`·`full_output`·`args` 없음(`infodict`의 값은 필드), 노름은 `'2'`·`'max'`만, float32 값도 float64로 |
| `cubature(..., rule, rtol, atol, max_subdivisions, args, workers, points)` | `cubature(f, a, b, /, *, rule, rtol, atol, max_subdivisions, points)` | `[scipy와 다름]` `args`·`workers`·규칙 객체·`'gauss-kronrod'` 없음, `estimate`→`integral`, 정수 status(0, 1, 3), `regions`는 배열 필드, (−∞, b]의 치환, `max_subdivisions`는 1 이상의 int |
| `tanhsinh(f, a, b, *, args, kwargs, log, maxlevel, minlevel, atol, rtol, preserve_shape, callback)` | `tanhsinh(f, a, b, /, *, args, log, max_level, min_level, atol, rtol)` | `[scipy와 다름]` `kwargs`·`preserve_shape`·`callback` 없음, `success`→`converged`, `maxlevel`→`max_level` |
| `nsum(f, a, b, *, step, args, kwargs, log, maxterms, tolerances)` | `nsum(f, a, b, /, *, step, args, log, max_terms, atol, rtol)` | `[scipy와 다름]` `kwargs` 없음, `tolerances`→`atol`·`rtol`, `maxterms`→`max_terms` |
| `fixed_quad(func, a, b, args, n)` | `fixed_quad(func, a, b, /, *, n)` | `[scipy와 다름]` `args` 없음, 값 하나(또는 배열)를 돌려준다, NaN 끝은 `ValueError`, `n` ≤ 10⁴ |
| `newton_cotes(rn, equal=0)` | `newton_cotes(rn, /, *, equal=False)` | `[scipy와 다름]` `equal`은 bool, 정확한 값의 반올림(N ≥ 15·위치를 준 경우 scipy와 다름), 위치의 검사는 언제나, N ≤ 256 |
| `trapezoid(y, x, dx, axis)`, `cumulative_trapezoid(y, x, dx, axis, initial)` | `trapezoid(y, /, x=None, *, dx, axis)` 등 | `[scipy와 다름]` `dx`가 키워드뿐, `x`와 함께 준 `dx`는 `ValueError` |
| `simpson(y, *, x, dx, axis)`, `cumulative_simpson(y, *, x, dx, axis, initial)` | `simpson(y, /, x=None, *, dx, axis)` 등 | `[scipy와 다름]` `x`를 위치로도(C40), `simpson`의 점 0개는 0 |
| `romb(y, dx, axis, show)` | `romb(y, /, *, dx, axis)` | `[scipy와 다름]` `show` 없음, `dx`가 키워드뿐 |
| `solve_ivp`, `odeint`, `solve_bvp`, `RK45` 등 | 없음(나중의 ODE 묶음) | |
| `ode`, `complex_ode`, `lebedev_rule`, `qmc_quad`, `IntegrationWarning`, `ODEintWarning` | 없음 | docs/design/integrate.md §1.2 |

## §10. 마일스톤별 남은 질문
C0 전의 것은 정했다(docs/design/integrate.md §1.1 C8–C17). C2(C26–C38), C3(C39–C54), C4(C55–C70), C5(C71–C83), C6(C84–C93), C7(C94–C107), C8(C108–C116)도 정했고, 코드 리뷰의 고침은 C117–C123이다. 첫 판의 마일스톤에 남은 질문은 없다(docs/design/integrate.md §9).
