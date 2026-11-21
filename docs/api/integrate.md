# sd.integrate API

> 상태: **초안** (설계 초안, 2026-11-21). 공개된 이름은 아직 없다. 아래 시그니처는 scipy 1.18.1 `scipy.integrate`의 이름과 인자 이름(C1, C2)에 stardust의 규칙과 C4·C8–C17의 결정(결과 클래스, 둘로 풀림, `atol`·`rtol`, `max_…`, `integral`·`error`·`nfev`)을 입히고, 남은 질문(docs/design/integrate.md §9 C2–C7)은 그 추천안을 넣은 안이다. 함수는 마일스톤 C3–C7에서 공개된다(docs/design/integrate.md §8).

- 설계와 결정의 근거는 [docs/design/integrate.md](../design/integrate.md)(§1.1 C1–C17), 미분은 [docs/api/differentiate.md](differentiate.md), 공통 규칙은 [API.md](../../API.md)다.
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
- **`quad`·`dblquad`·`tplquad`·`nquad`**: 점 하나마다 Python `float`로 부르고 실수 하나(Python·numpy의 수, 0차원 배열)를 받는다. 부르는 순서는 QUADPACK의 순서이고 scipy 1.18.1과 같다(C5, docs/design/integrate.md §2.3). 다른 값은 `quad(): 'func' must return a real number, not shape (3,)`(`ValueError`), bool은 `TypeError`다(optimize와 같다).
- **`quad_vec`**: 점 하나마다 `float`로 부르고 언제나 같은 모양의 배열(`sd.ndarray`, 실수의 시퀀스, numpy 배열)을 받는다. 모양이 바뀌면 `ValueError`다.
- **`cubature`**: 점들을 (n_points, ndim) float64 `sd.ndarray`로 한 번에 부르고 (n_points, …)를 받는다(scipy와 같다).
- **`fixed_quad`**: n점을 (n,) float64 `sd.ndarray`로 한 번 부르고 (n,)을 받는다(scipy와 같다).
- **`tanhsinh`, `nsum`**: 아직 끝나지 않은 원소의 점을 모아 부른다(O38). 넘기는 꼴은 docs/design/integrate.md §9 C2의 질문이다(추천: (k, p) 2차원, `args`는 (k, 1)). 첫 호출은 (k,)이다(O48). 사용자 함수는 원소별이어야 하고(`f(x)[i]`가 `f(x[i])`), 끝점 바로 옆이나 끝점 자체(반올림)를 받을 수 있으므로 특이점에서 예외를 내지 말고 inf·NaN을 돌려줘야 한다(그 값은 버린다; scipy와 같다).
- 피적분함수의 값은 실수다. 복소수는 `quad(complex_func=True)`만 받는다(docs/design/integrate.md §9 C4의 추천).

### §1.3 결과 `IntegrateResult` (C9, C10, C13)
```
IntegrateResult
```
- 모든 함수가 이 클래스를 돌려준다(표본 규칙, `fixed_quad`, `newton_cotes` 빼고: 값이나 배열이다). 규칙은 `sd.optimize.OptimizeResult`와 같다: 함수가 주는 필드만 있고 없는 필드는 `AttributeError`, 필드는 읽기 전용, 배열 필드는 읽기 전용이며 읽을 때 복사하지 않는다(O103, API.md §2.2), `repr`는 한 줄, dict처럼 쓰지 않는다.
- `quad`·`dblquad`·`tplquad`·`nquad`·`quad_vec`의 결과는 scipy의 `(y, abserr)`처럼 둘로 풀린다: `integral, error = sd.integrate.quad(f, 0, 1)`(htest H7의 규칙). 다른 필드는 이름으로 읽는다. scipy의 `full_output`은 없다(필드가 늘 있다).

| 필드 | 뜻 | 주는 함수 |
|---|---|---|
| `integral` | 적분값 | 적분 함수 모두(`nsum` 빼고) |
| `sum` | 급수의 값 | `nsum` |
| `error` | 절대 오차의 추정(엄밀한 상계가 아니다) | 모두 |
| `converged` | 멈춤 판정을 통과했는지 | 모두 |
| `status` | 멈춘 이유의 번호(scipy의 번호, O17) | 모두 |
| `message` | 멈춘 이유의 글(scipy의 글) | `quad` 계열, `quad_vec`, `cubature` |
| `nfev` | 사용자 함수의 값을 계산한 점의 수 | 모두 |
| `max_level` | 마지막 수준(scipy `maxlevel`) | `tanhsinh` |
| `intervals`, `interval_integrals`, `interval_errors`, `n_intervals` | 쓴 구간과 구간마다의 값·오차(scipy `infodict`의 `alist blist rlist elist last`; docs/design/integrate.md §9 C4) | `quad`(가중치 없는 것), `quad_vec` |
| `subdivisions` | 나눈 수 | `cubature` |

- 원소별 반복 무리(`tanhsinh`, `nsum`)의 필드는 입력의 배치 모양이고(`integral` float64, `converged` bool, `status`·`nfev`·`max_level` int64), 입력이 모두 실수면 Python 수다(O48). `repr`는 `IntegrateResult(batch shape (1000,), 998 of 1000 converged)`다.

### §1.4 멈춤, 상한, 경고
- **모든 호출은 정해진 시간 안에 끝난다.** 상한에 닿으면 마지막 값과 `converged=False`다. 예외가 아니다.
- **경고를 내지 않는다.** scipy가 `IntegrationWarning`으로 알리는 것은 `converged=False`와 `status`·`message`다. scipy의 `show`(`romb`)는 없다(화면에 쓰지 않는다, `[scipy와 다름]`).
- 오차 추정은 경험적이다: 수렴이라고 알려도 실제 오차가 추정보다 클 수 있다. 좁은 봉우리를 처음 규칙이 놓치면 `quad`가 틀린 값을 수렴으로 알린다(docs/design/integrate.md §7.4). 어려운 곳의 자리를 알면 `points`로 준다.
- 같은 입력이면 스레드 수와 상관없이 같은 비트다.

### §1.5 에러
- 공개 함수의 에러는 `quad(): …`로 시작한다(API.md §2.1). 끝: `a`, `b`는 실수(±inf 받음; NaN은 `ValueError`, 원소별 무리는 그 원소의 status −3)다. 옵션의 검사와 글은 다른 영역과 같은 꼴이다(`'limit' must be an int, not float`, `'rtol' must be a finite number >= 0`).
- 쓰이지 않는 옵션을 함께 주면 `ValueError`다(O29·O62의 규칙): `quad(): 'points' is not used with 'weight'`(scipy는 경고하고 무시한다, `[scipy와 다름]`), `quad(): 'wvar' is only used with 'weight'`.

### §1.6 스레드와 GIL
- 사용자 함수가 Python이라 GIL을 쥔 채 부른다. `quad` 계열은 호출 사이의 일이 작아 GIL을 놓지 않는다.
- 원소별 무리는 진행 중인 원소가 2¹⁵개 이상이면 원소를 지나가는 패스를 스레드에 나누고 GIL을 놓는다(optimize의 배치와 같다). 표본 규칙은 배열 리덕션의 기준이다(API.md §3).

## §2. 적응 구적 `quad` (C4, C5)
```
quad(func, a, b, /, *, atol=1.49e-08, rtol=1.49e-08, limit=50, points=None, weight=None, wvar=None, max_p1=50, limlst=50, complex_func=False)
```
- ∫ₐᵇ func(x) dx를 QUADPACK으로 계산한다. 루틴은 scipy처럼 입력이 고른다: 유한 구간 QAGS, 무한 구간 QAGI, `points` QAGP(C4); `weight` QAWO·QAWF·QAWS·QAWC(C5). 표는 docs/design/integrate.md §3.1이다.
- `atol`·`rtol`(scipy `epsabs`·`epsrel`, 기본 1.49×10⁻⁸): |I − 값| ≤ max(atol, rtol·|I|)를 목표로 한다. `atol <= 0`이면 `rtol`은 max(50ε, 5×10⁻²⁹) 이상이어야 한다(scipy의 `ier = 6`의 갈래 그대로, `ValueError`).
- `limit`(기본 50): 구간 수의 상한(1 이상). 닿으면 status 1. `points`: 구간 안의 중단점(유한 구간만, 무한이면 `ValueError`); scipy처럼 정렬하고 끝과 겹친 것은 뺀다. 개수는 `limit`보다 작아야 한다.
- `weight`와 `wvar`: `'cos'`·`'sin'`(wvar = ω: cos(ωx)·sin(ωx); 유한 구간은 QAWO, 반무한은 QAWF, 양쪽 무한은 `ValueError`), `'alg'`·`'alg-loga'`·`'alg-logb'`·`'alg-log'`(wvar = (α, β), α, β > −1: (x − a)^α (b − x)^β와 그 로그), `'cauchy'`(wvar = c: 1/(x − c)의 주값; c는 a, b가 아니어야 한다). 가중치는 유한 구간만 받는 것이 scipy와 같다(QAWF 빼고). `max_p1`(scipy `maxp1`, 기본 50): QAWO의 체비쇼프 모멘트 수의 상한. `limlst`(기본 50, 3 이상): QAWF의 주기 수의 상한. scipy의 `wopts`(미리 계산한 모멘트)는 없다(docs/design/integrate.md §9 C5의 추천, `[scipy와 다름]`).
- `complex_func=True`: func가 복소수를 돌려주면 실수부와 허수부를 따로 `quad`로 계산한다(scipy와 같다). `integral`·`error`는 complex이고 다른 필드는 `real`·`imag` 두 결과의 것이다(C4에서 정한다).
- a > b면 scipy처럼 끝을 바꿔 계산하고 부호를 바꾼다. a == b면 0, `nfev` 0이다.
- **status와 message** (O17): QUADPACK의 `ier`와 scipy의 글이다. 0 수렴, 1 `The maximum number of subdivisions (50) has been achieved. …`, 2 반올림, 3 나쁜 피적분함수, 4 외삽 표가 수렴하지 않음, 5 발산 또는 느린 수렴, 7 비정상 끝(QAWF는 1·4·7의 글이 scipy의 것으로 다르다). 6(입력이 틀림)은 결과가 아니라 scipy의 글을 쓴 `ValueError`다. `converged`는 `status == 0`이다.
- 결과: `integral error converged status message nfev`와 구간 정보(§1.3; 가중치 루틴은 C5에서 정한다). 둘로 풀린다.

## §3. 겹친 적분 (C4)
```
dblquad(func, a, b, gfun, hfun, /, *, atol=1.49e-08, rtol=1.49e-08)
tplquad(func, a, b, gfun, hfun, qfun, rfun, /, *, atol=1.49e-08, rtol=1.49e-08)
nquad(func, ranges, /, *, opts=None)
```
- `dblquad`: ∫ₐᵇ ∫_{g(x)}^{h(x)} func(y, x) dy dx(scipy처럼 안쪽 변수가 먼저). `gfun`·`hfun`은 수 또는 x의 함수다. `tplquad`: ∫ₐᵇ ∫_{g(x)}^{h(x)} ∫_{q(x, y)}^{r(x, y)} func(z, y, x) dz dy dx.
- `nquad`: `ranges`는 변수마다(안쪽부터) `(lo, hi)` 쌍 또는 바깥 변수를 받아 쌍을 돌려주는 함수다. `opts`는 dict 하나(모든 층) 또는 층마다의 dict·함수의 시퀀스이고, 키는 `quad`의 옵션 이름(`atol rtol limit points weight wvar`; scipy의 `epsabs`·`epsrel`이 아니다)이다. 모르는 키는 `ValueError`(optimize O16의 규칙; scipy는 `TypeError`)다.
- 안쪽 적분은 바깥 점마다 `quad`로 계산한다(scipy와 같은 순서). `error`는 scipy처럼 바깥 적분의 추정뿐이다(안쪽의 오차를 더하지 않는다). `nfev`는 모든 층의 호출 합이다. `converged`는 모든 층의 모든 `quad`가 수렴했을 때이고, `status`·`message`는 바깥의 것 또는 처음 수렴하지 못한 안쪽의 것이다(C4에서 정한다; scipy는 안쪽의 경고를 그대로 낸다).
- scipy `nquad`의 `args`와 `full_output`은 없다.

## §4. 벡터 값 함수 `quad_vec` (C7)
```
quad_vec(f, a, b, /, *, atol=1e-200, rtol=1e-08, norm='2', limit=10000, points=None, quadrature=None)
```
- 배열 값 f의 ∫ₐᵇ f(x) dx를 원소마다 한 번에 적응 Gauss–Kronrod로 계산한다(scipy의 알고리즘). `norm`은 `'2'`(기본) 또는 `'max'`: 오차를 재는 노름. `quadrature`: None(유한 구간 `'gk21'`, 무한 `'gk15'`), `'gk21'`, `'gk15'`, `'trapezoid'`.
- 결과의 `integral`·`error`는 f의 모양(`error`는 노름의 값 하나, scipy와 같다)이다. `status`: 0 수렴, 1 `limit`, 2 반올림(scipy의 번호와 글). scipy의 `cache_size`, `workers`, `full_output`은 없다(`[scipy와 다름]`).

## §5. 다차원 `cubature` (C7)
```
cubature(f, a, b, /, *, rule='gk21', rtol=1e-08, atol=0.0, max_subdivisions=10000, points=None)
```
- `a`·`b`는 길이 ndim의 끝(±inf 받음). `rule`: `'gk21'`, `'gk15'`(1차원 Gauss–Kronrod의 곱), `'genz-malik'`(ndim ≥ 2). scipy의 `'gauss-kronrod'`(별칭)와 규칙 객체는 받지 않는다(`[scipy와 다름]`). `points`: 피해야 할 점의 목록(scipy와 같다).
- 결과: `integral error converged status message subdivisions nfev`. scipy `CubatureResult`의 `estimate`는 `integral`, 문자열 `status`는 `converged`와 정수 `status`다(docs/design/integrate.md §9 C7의 추천, `[scipy와 다름]`). `regions`는 C7에서 정한다. `workers`는 없다.

## §6. 배치 tanh-sinh와 급수 (C6)
```
tanhsinh(f, a, b, /, *, args=(), log=False, max_level=None, min_level=2, atol=None, rtol=None)
nsum(f, a, b, /, *, step=1, args=(), log=False, max_terms=1048576, atol=None, rtol=None)
```
- 원소별 반복 무리다(C4): `a`, `b`(, `step`)와 `args`의 값을 브로드캐스트한 배치의 원소마다 따로 계산한다. 호출 규약은 §1.2.
- **`tanhsinh`**: ∫ₐᵇ f. 끝은 무한이어도 되고 끝점 특이점을 받는다. 구간 안의 특이점·꺾임은 범위 밖이다(구간을 나눠 부른다). `max_level`(scipy `maxlevel`, 기본 10): 수준의 상한, `min_level`(scipy `minlevel`, 기본 2): 처음 한 번의 호출에 넣는 수준(수준 0–min_level의 점; 호출 수만 바꾼다). `atol`(기본 0)·`rtol`(기본 ε^0.75): error < atol 또는 error < rtol·|integral|에서 멈춘다. `log=True`면 f는 log(피적분함수)를 돌려주고 `integral`·`error`·`atol`·`rtol`도 log다.
  - 결과: `integral error converged status nfev max_level`. status: 0 수렴, −2 `max_level`, −3 유한하지 않은 값(scipy의 번호; −1은 쓰지 않는다). a == b면 0(`max_level` −1). b < a면 부호를 바꾼다.
- **`nsum`**: Σ_{k} f(a + k·step), a ≤ a + k·step ≤ b. f는 매끄럽고 양이며 단봉인 함수여야 한다(실수 전체에서 정의; scipy의 계약). 항 수 ⌊(b − a)/step⌋ + 1이 `max_terms`(scipy `maxterms`, 기본 2²⁰) 이하면 직접 더하고, 아니면 적분 판정의 경계로 나머지를 `tanhsinh`으로 근사한다. `atol`(기본 0)·`rtol`(기본 √ε)은 scipy `tolerances`의 두 키다(O36).
  - 결과: `sum error converged status nfev`. status: 0, −1(`a`·`b`·`step`이 틀림: step ≤ 0 등), −2(적분의 수준 상한; 발산할 수 있다), −3(유한하지 않은 값), −4(부분합의 마지막 항이 허용오차보다 크다: `max_terms`를 늘리거나 나눠 더한다). b < a면 0이다.
- scipy의 `kwargs`, `callback`, `preserve_shape`는 없다(C4, C14, O37; `[scipy와 다름]`).

## §7. 고정 규칙 (C3)
```
fixed_quad(func, a, b, /, *, n=5)
newton_cotes(rn, /, *, equal=False)
```
- `fixed_quad`: n점 Gauss–Legendre로 ∫ₐᵇ func. func는 (n,) 배열을 받아 (n,)을 돌려준다. a, b는 유한한 실수다. 돌려주는 것은 값 하나다(scipy의 `(val, None)`이 아니다; docs/design/integrate.md §9 C3의 추천, `[scipy와 다름]`). 점과 가중치는 올바로 반올림한 값이다(기본 n = 5에서는 scipy와 같은 점이다; n = 20에서 이미 scipy `roots_legendre`와 끝자리가 다른 점이 있다, docs/design/integrate.md §7.4).
- `newton_cotes`: `rn`이 int N이면 같은 간격 N + 1점, 길이 N + 1의 정수 배열이면 그 상대 위치(0부터 N까지)의 닫힌 Newton–Cotes 가중치 `an`과 오차 계수 `B`의 튜플 `(an, B)`다(scipy와 같다). `equal=True`면 같은 간격으로 본다(scipy는 int 0·1; bool만 받는다).

## §8. 표본의 규칙 (C3)
```
trapezoid(y, /, x=None, *, dx=1.0, axis=-1)
simpson(y, /, x=None, *, dx=1.0, axis=-1)
romb(y, /, *, dx=1.0, axis=-1)
cumulative_trapezoid(y, /, x=None, *, dx=1.0, axis=-1, initial=None)
cumulative_simpson(y, /, x=None, *, dx=1.0, axis=-1, initial=None)
```
- `y`는 표본(`sd.ndarray`, 시퀀스, numpy 배열, `Series`; docs/design/integrate.md §9 C3의 추천)이고 float64(int·bool은 바꿔서) 또는 complex128로 계산한다. `x`는 `axis`를 따라 같은 길이의 1차원 또는 `y`와 같은 모양이다. `x`가 없으면 간격 `dx`다.
- `axis`의 기본은 scipy·numpy의 −1이다. 결과는 `axis`를 뺀 모양(누적은 `axis`의 길이가 하나 준다, `initial`이 있으면 그대로)이고, 0차원이면 Python 수다.
- `simpson`은 scipy 1.11부터의 꼴(점이 짝수 개면 마지막 구간을 Cartwright의 보정으로)이다. `romb`은 점이 2ᵏ + 1개여야 하고 아니면 `ValueError`다. `cumulative_trapezoid`의 `initial`은 None 또는 0(scipy와 같다), `cumulative_simpson`의 `initial`은 수 또는 `axis`의 길이가 1인 배열이다.
- 합은 sd의 합 규칙이다(docs/api/array.md §7.8): scipy·numpy와 끝자리가 다를 수 있고 스레드 수와 상관없이 같은 비트다. sd 배열 쪽에는 `trapezoid`가 없다(numpy의 `np.trapezoid`는 이 함수다).

## §9. scipy 이름 → sd 이름
| scipy 1.18.1 | sd | 다른 점 |
|---|---|---|
| `quad(func, a, b, args, full_output, epsabs, epsrel, limit, points, weight, wvar, wopts, maxp1, limlst, complex_func)` | `quad(func, a, b, /, *, atol, rtol, limit, points, weight, wvar, max_p1, limlst, complex_func)` | `[scipy와 다름]` `args`·`full_output`·`wopts` 없음, `epsabs`→`atol`, `epsrel`→`rtol`, `maxp1`→`max_p1`, 결과 객체(둘로 풀림), 경고 대신 status |
| `dblquad`, `tplquad` | 같다 | `[scipy와 다름]` `args` 없음, `atol`·`rtol` |
| `nquad(func, ranges, args, opts, full_output)` | `nquad(func, ranges, /, *, opts)` | `[scipy와 다름]` `args`·`full_output` 없음, `opts`의 키 이름 |
| `quad_vec(..., epsabs, epsrel, norm, cache_size, limit, workers, points, quadrature, full_output, args)` | `quad_vec(f, a, b, /, *, atol, rtol, norm, limit, points, quadrature)` | `[scipy와 다름]` `cache_size`·`workers`·`full_output`·`args` 없음 |
| `cubature(..., rule, rtol, atol, max_subdivisions, args, workers, points)` | `cubature(f, a, b, /, *, rule, rtol, atol, max_subdivisions, points)` | `[scipy와 다름]` `args`·`workers`·규칙 객체·`'gauss-kronrod'` 없음, `estimate`→`integral`, status |
| `tanhsinh(f, a, b, *, args, kwargs, log, maxlevel, minlevel, atol, rtol, preserve_shape, callback)` | `tanhsinh(f, a, b, /, *, args, log, max_level, min_level, atol, rtol)` | `[scipy와 다름]` `kwargs`·`preserve_shape`·`callback` 없음, `success`→`converged`, `maxlevel`→`max_level` |
| `nsum(f, a, b, *, step, args, kwargs, log, maxterms, tolerances)` | `nsum(f, a, b, /, *, step, args, log, max_terms, atol, rtol)` | `[scipy와 다름]` `kwargs` 없음, `tolerances`→`atol`·`rtol`, `maxterms`→`max_terms` |
| `fixed_quad(func, a, b, args, n)` | `fixed_quad(func, a, b, /, *, n)` | `[scipy와 다름]` `args` 없음, 값 하나를 돌려준다 |
| `newton_cotes(rn, equal=0)` | `newton_cotes(rn, /, *, equal=False)` | `[scipy와 다름]` `equal`은 bool |
| `trapezoid`, `simpson`, `cumulative_trapezoid` | 같다 | `x`는 위치·키워드 모두(scipy와 같다) |
| `cumulative_simpson(y, *, x, dx, axis, initial)` | `cumulative_simpson(y, /, x=None, *, ...)` | `[scipy와 다름]` `x`를 위치로도 |
| `romb(y, dx, axis, show)` | `romb(y, /, *, dx, axis)` | `[scipy와 다름]` `show` 없음 |
| `solve_ivp`, `odeint`, `solve_bvp`, `RK45` 등 | 없음(나중의 ODE 묶음) | |
| `ode`, `complex_ode`, `lebedev_rule`, `qmc_quad`, `IntegrationWarning`, `ODEintWarning` | 없음 | docs/design/integrate.md §1.2 |

## §10. 마일스톤별 남은 질문
C0 전의 것은 정했다(docs/design/integrate.md §1.1 C8–C17). 남은 것은 docs/design/integrate.md §9에 모았고, 이 문서의 꼴은 그 추천안이다: C2의 1(여러 점의 꼴), C3의 1–5, C4의 1–2, C5, C6, C7.
