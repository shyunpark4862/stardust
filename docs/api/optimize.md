# sd.optimize API

> 상태: **초안** (O0, 2026-11-20; O1 2026-11-20). 지금 공개된 것은 결과 객체 `OptimizeResult`(§1.3), 수치 미분 `approx_fprime`·`check_grad`(§2), 비선형 최소제곱 `least_squares`(§3)다(O1). 나머지 풀이 함수는 O2–O5에서 구현한다(docs/design/optimize.md §8). 아래 시그니처는 scipy 1.18.1의 이름과 인자 이름(O3, O10–O12)에 stardust의 규칙을 입힌 안이다. 마일스톤을 시작할 때 정할 것은 본문에 "(O1에서 정한다)"처럼 적고 §9에 모았다.

## §1. 공통 규칙

### §1.1 입력과 옵션
- 입력은 위치로만(`fun`, `x0`, `A`, `b`), 옵션은 키워드로만 받는다. 별칭은 없다: scipy의 `success`·`nit` 같은 다른 이름도, 대소문자만 다른 방법 이름도 받지 않는다.
- **방법 이름은 소문자만** 받는다(O15): `'bfgs'`, `'l-bfgs-b'`, `'nelder-mead'`, `'newton-cg'`, `'trust-ncg'`, `'trust-exact'`, `'brentq'`, `'hybr'` 등. scipy가 받는 다른 대소문자는 소문자 꼴을 짚는 `ValueError`다: `minimize(): 'method' must be lowercase: 'bfgs', not 'BFGS'`. 모르는 이름은 다른 문자열 옵션과 같은 꼴이다(`minimize(): 'method' must be 'nelder-mead', 'bfgs', 'l-bfgs-b', 'newton-cg', 'trust-ncg' or 'trust-exact', not 'cg'`).
- **반복 상한의 이름은 `max_iter`와 `max_nfev`**다(O14, regression의 `glm`·`lasso`의 `max_iter`, `nls`의 `max_nfev`와 같다). scipy의 `maxiter`(반복 수)는 `max_iter`, `maxfev`·`maxfun`(함수 계산 수)은 `max_nfev`다. 함수의 키워드와 `options`의 키 모두 그렇다. scipy의 이름은 받지 않는다(`minimize(): method='bfgs' has no option 'maxiter' (the iteration cap is 'max_iter')`).
- 사용자 함수의 추가 인자 `args=()`·`kwargs`는 없다(O12). 상수는 클로저나 `lambda`로 넘긴다: `minimize(lambda x: f(x, a), x0)`.
- scipy의 개별 함수(`brentq`, `toms748`, `bisect`, `newton`, `fminbound`, `brent`, `golden`, `fmin`, `fmin_bfgs`, `fmin_l_bfgs_b`, `fmin_ncg`, `fsolve`, `leastsq`)는 없다(O10). 방법은 통합 함수의 `method=`로 고른다. `curve_fit`도 없다(O11): 곡선 적합은 `sd.regression.nls`가 추론까지 한다.
- `x0`(그리고 `A`, `b`, 구간의 끝)은 `sd.ndarray`, 실수의 시퀀스, numpy 배열을 받고 float64로 계산한다(bool·int64는 바꿔서). complex는 `TypeError`(`minimize(): 'x0' must be real, not a complex128 array`)다. `x0`는 1차원이고 유한해야 한다(`ValueError`). 스칼라 함수의 `x0`·`bracket`의 끝은 실수다.
- 옵션의 검사와 메시지는 다른 영역과 같은 꼴이다(API.md §2.1): bool은 `'adaptive' must be True or False, not int`, int는 `'max_iter' must be an int, not float`, 수는 `'tol' must be a finite number >= 0`, 문자열은 `'method' must be 'trf' or 'lm', not 'x'`.
- 방법마다 다른 옵션은 scipy처럼 `options=`의 dict로 받는다(O16; 키는 scipy의 이름(반복 상한만 O14의 이름), 기본값도 scipy의 것). 방법이 모르는 키는 경고가 아니라 `ValueError`다(scipy는 `OptimizeWarning`만 낸다): `minimize(): method='bfgs' has no option 'xatol'`. dict가 아닌 값은 `TypeError`(`minimize(): 'options' must be a dict or None, not list`)다.
- **`status`와 `message`는 scipy의 것**이다(O17): 방법마다 scipy 1.18.1의 번호와 글을 그대로 쓴다(MINPACK의 info, L-BFGS-B의 task 글 등). 이 라이브러리에만 있는 멈춤(예: 유한하지 않은 마지막 값)의 번호와 글은 그 마일스톤에서 정한다.
- **scipy와 같은 걸음** (O9): 각 방법은 scipy 1.18.1과 같은 순서로 계산한다. 같은 문제에서 반복점, `nfev`·`njev`, 멈춘 이유가 같다. 합과 노름처럼 반올림의 순서가 다른 곳은 끝자리가 다를 수 있고, 비트까지 같게 하지는 않는다. 대조 자료는 `tests/optimize/data/`다(docs/design/optimize.md §7).

### §1.2 사용자 함수
- 사용자 함수는 Python 호출 가능한 객체다. 반복은 C++ 커널이 돌고 함수값은 콜백으로 받는다(`sd.regression.nls`와 같다).
- 다변수 함수에 넘기는 x는 float64 `sd.ndarray`(1차원, 길이 n)이고, 함수가 고쳐도 반복에는 영향이 없다(새 배열이다). 스칼라 함수(`root_scalar`, `minimize_scalar`)에는 Python `float`를 넘긴다.
- 돌려받는 값: 스칼라 함수와 `minimize`의 `fun`은 실수 하나, `least_squares`·`root`의 `fun`은 길이 m의 1차원 값, 야코비안은 (m, n)이다. `sd.ndarray`, 실수의 시퀀스, numpy 배열을 받는다(nls의 규칙).
- 사용자 함수가 낸 에러는 그대로 나간다.
- 유한하지 않은 함수값의 처리(걸음을 거절하고 신뢰 영역을 줄일지, 멈출지)는 방법마다 정한다(§9). `least_squares`는 nls의 규칙(G74)을 옮겼다: 시험점에서 유한하지 않으면 걸음을 거절하고 신뢰 영역을 줄이며, `x0`에서 유한하지 않으면 `ValueError`(`least_squares(): 'fun' is not finite at 'x0'`)다.
- 배치 스칼라 함수(§6)의 콜백은 모양이 (k,)인 float64 `sd.ndarray`를 받아 같은 모양을 돌려준다. 이미 끝난 원소는 다음 호출에서 뺀다(scipy `elementwise`와 같다; O2에서 정한다).

### §1.3 결과 `OptimizeResult`
```
OptimizeResult
```
- 모든 함수가 이 클래스 하나를 돌려준다(`nnls`는 O5에서 정한다, §9). 방법이 주는 필드만 있고, 없는 필드를 읽으면 `AttributeError`(`'OptimizeResult' object has no attribute 'hess_inv' (its method does not give it)`)다. 그래서 `hasattr(r, 'hess_inv')`로 알 수 있다.
- 필드는 읽기 전용이다(regression G98; 대입하면 `property 'x' of 'OptimizeResult' object has no setter`). 배열 필드는 읽을 때마다 새 복사본이다(G113): 돌려받은 배열을 고쳐도 결과는 그대로다. 결과가 만들어질 때 받은 배열도 복사해 둔다.
- 이름은 다른 영역의 결과와 같은 규칙이다(O4): scipy의 `success`는 `converged`, `nit`는 `n_iter`다. 별칭은 없다.
- `repr`는 한 줄 요약이다: `OptimizeResult(method='bfgs', converged=True, fun=1.2e-17, n_iter=34, nfev=41, x=[1.0 1.0])`. 최소제곱은 `fun` 대신 `cost`를 보인다. 배치는 `OptimizeResult(batch shape (1000,), 998 of 1000 converged)`다. `summary()`는 없다(추론이 없다).
- scipy의 `OptimizeResult`처럼 dict로 쓰는 것(`r['x']`, `r.keys()`)은 없다.

| 필드 | 뜻 | 주는 함수 |
|---|---|---|
| `x` | 해. 다변수는 (n,) `sd.ndarray`, 스칼라는 `float`, 배치는 배치 모양의 배열 | 모두 |
| `fun` | 해에서의 함수값: 스칼라 목적함수는 `float`, `least_squares`·`root`·`lsq_linear`는 잔차 (m,) | 모두 |
| `jac` | 해에서의 기울기(최소화) 또는 야코비안 (m, n)(최소제곱) | `minimize`의 기울기를 쓰는 방법, `least_squares` |
| `hess` | 해에서의 헤시안 | `minimize`의 trust-ncg·trust-exact(`hess`를 줄 때) |
| `hess_inv` | 헤시안 역행렬의 근사 | `minimize`의 BFGS, L-BFGS-B (꼴은 O3에서 정한다) |
| `grad`, `cost`, `optimality`, `active_mask` | scipy `least_squares`와 같은 뜻: Jᵀf, ½‖f‖², 1차 최적성의 척도, 경계에 걸린 변수(−1 아래, 0 안, 1 위) | `least_squares`, `lsq_linear` |
| `bracket`, `f_bracket` | 마지막 구간과 그 끝의 함수값(배열의 튜플) | 배치 스칼라(§6) |
| `final_simplex` | 마지막 단체와 그 꼭짓점의 함수값 | `minimize`의 Nelder–Mead (O3에서 정한다) |
| `converged` | 방법의 멈춤 판정을 통과했는지(bool, 배치면 bool 배열) | 모두 |
| `status` | 멈춘 이유의 번호(scipy의 번호, O17) | 모두 |
| `message` | 멈춘 이유의 글(scipy의 글, O17) | 모두(배치 빼고) |
| `n_iter` | 반복 수(int, 배치면 int64 배열) | 모두 |
| `nfev`, `njev`, `nhev` | 함수, 야코비안(기울기), 헤시안을 계산한 수 | 방법마다 |
| `method` | 쓴 방법의 이름 | 모두(배치 빼고) |

- 스칼라 문제의 숫자 필드는 Python 수다(`converged`는 `bool`, `status`·`n_iter`·`nfev`는 `int`, `fun`은 `float`). 배치 스칼라는 필드마다 배치 모양의 `sd.ndarray`다(`x` float64, `converged` bool, `status`·`nfev`·`n_iter` int64).
- `nfev`는 사용자 함수를 부른 수 전부다. 차분 야코비안의 호출도 센다(O18, nls와 같다). scipy 1.18.1의 `least_squares`는 두 방법 모두, `minimize`도 차분의 호출을 세지 않으므로 차분을 쓰면 scipy의 `nfev`와 다르다(scipy의 수는 `nfev − (차분 한 번의 호출 수)·njev`다). `njev`는 야코비안(기울기)을 계산한 수다: 사용자 함수의 호출 수, 또는 차분 야코비안의 수.
- `root_scalar`의 결과도 이 클래스다(scipy의 `RootResults`의 `root`·`iterations`·`function_calls`·`flag`가 아니라 `x`·`n_iter`·`nfev`·`message`; O2에서 정한다).

### §1.4 멈춤, 상한, 경고
- **모든 호출은 정해진 시간 안에 끝난다.** 반복 방법은 반복 수와 함수 호출 수에 상한이 있다(기본값은 scipy의 것). 상한에 닿으면 마지막 점과 `converged=False`를 돌려준다(regression G32와 같다). 예외는 내지 않는다(scipy의 `nnls`는 `RuntimeError`를 낸다; O5에서 정한다).
- 경고를 내지 않는다. 수렴, 상한, 유한하지 않은 값은 `converged`, `status`, `message`로 알린다. scipy의 `disp`, `verbose`, `iprint`, `return_all`은 없다(화면에 쓰지 않는다).
- `converged`는 scipy의 `success`와 같은 판정이다. 다만 판정을 통과해도 마지막 값(`least_squares`의 `cost`)이 유한하지 않으면 False다(O20, nls의 G89). 그때 `status`와 `message`는 scipy의 것 그대로다.
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 사용자 함수가 결정론적이면 결과도 결정론적이다.

### §1.5 에러
- 공개 함수의 에러는 함수 이름과 `(): `로 시작한다(API.md §2.1): `least_squares(): 'x0' must be 1-dimensional, not shape (2, 2)`, `minimize(): 'fun' must be callable, not int`.
- 사용자 함수가 틀린 모양을 돌려주면 `ValueError`다: `minimize(): 'fun' must return a real number, not shape (3,)`, `least_squares(): 'fun' must return a 1-dimensional array, not shape (3, 2)`, `least_squares(): 'jac' must return shape (40, 3), not (40, 2)`, `root(): 'fun' must return 3 values (one per variable), not 2`.
- 방법과 맞지 않는 인자는 `ValueError`다: `least_squares(): method='lm' takes no bounds (method='trf' does)`(nls와 같다), `root_scalar(): 'x0' is required when method='newton'`, `root_scalar(): 'bracket' is required when method='brentq'`, `minimize(): method='nelder-mead' takes no 'hess'`.
- 구간: `root_scalar(): 'bracket' must be a pair (a, b), not 3 values`, `minimize_scalar(): 'bounds' must be a pair (lower, upper), not 3 values`, `minimize_scalar(): the lower bound must be less than the upper bound`, `least_squares(): 'x0' must be within 'bounds'`.
- 근을 끼지 않는 구간(`f(a)`와 `f(b)`의 부호가 같다)을 `ValueError`로 할지 `converged=False`로 할지는 O2에서 정한다.

### §1.6 스레드와 GIL
- 사용자 함수가 Python이라 반복은 GIL을 쥔 채 돈다(API.md §3). 큰 선형대수(QR, SVD, Cholesky)만 linalg의 10µs 규칙으로 GIL을 놓는다. 사용자 함수 안의 큰 sd 연산은 그 영역의 규칙대로 GIL을 놓는다.
- 배치 스칼라 함수(§6)는 원소마다의 상태 갱신을 C++에서 배열 전체로 하므로, 원소가 많으면 그 부분이 스레드로 나뉜다(기준값은 O2에서 잰다).

## §2. 수치 미분 (O1)
```
approx_fprime(xk, f, /, *, epsilon=None)
check_grad(func, grad, x0, /, *, epsilon=None, direction='all', rng=None)
```
- `approx_fprime`: f(xk)의 기울기(f가 실수 하나를 돌려주면 (n,)) 또는 야코비안(f가 1차원 값 m개를 돌려주면 (m, n), m = 1이어도; O30)을 float64 `sd.ndarray`로 낸다. 인자의 순서(`xk`, `f`)는 scipy의 것이다. f에는 새 float64 `sd.ndarray` x를 넘긴다. f가 처음 점에서와 다른 개수를 돌려주면 `ValueError`다.
  - `epsilon=None`(기본)이면 nls의 정확한 차분이다(O6, regression G70): 변수마다 간격을 처음 한 번 고르는 5점 중앙 차분(2⁻⁸|x_j|에서 4배씩 줄이며 이웃한 두 추정의 차이가 8배 넘게 줄어드는 동안; 간격은 2의 거듭제곱이라 x_j ± kh가 정확하다), 한쪽에서 f가 유한하지 않으면 한쪽 5점 차분. MGH 47경우에서 해석적 야코비안과의 상대 차이는 최대 1.2×10⁻¹¹(기울기 6.4×10⁻¹¹)이다. 변수 하나에 f를 보통 8–12번 부른다. 반올림에 묻힌 열을 0으로 두지 않는다(nls의 G88과 다르다; O30). f가 xk에서 유한하지 않으면 `ValueError`(`approx_fprime(): 'f' is not finite at the point`), 어느 쪽 차분도 안 되면 `ValueError`(`approx_fprime(): 'f' is not finite near the point (the differences of variable 0)`)다.
  - `epsilon`에 양수 하나 또는 n개를 주면 scipy와 같은 앞쪽 차분 (f(x + ε_j e_j) − f(x))/((x_j + ε_j) − x_j)다(x_j + ε_j가 x_j로 반올림되면 scipy처럼 √u·sign(x_j)·max(1, |x_j|)). scipy의 기본값 √u = 2⁻²⁶을 주면 scipy와 비트까지 같다(MGH 47경우). xk에서 f가 유한하지 않아도 scipy처럼 계산한다(O30). 기본값의 결과는 scipy와 다르다(더 정확하다).
- `check_grad`: ‖grad(x0) − approx_fprime(x0, func, epsilon=epsilon)‖₂(scipy와 같다; `epsilon=2**-26`이면 scipy와 10⁻¹² 안). `func`는 실수 하나, `grad`는 n개를 돌려줘야 한다. `direction='random'`이면 무작위 방향 v(표준정규 n개, `sd.stats.Normal().sample((n,), rng=rng)`) 하나로 |grad(x0)·v − d/dw func(x0 + wv)|(w = 0)다. 그때는 `rng`(`sd.random.default_rng(seed)`의 Generator)를 꼭 준다(O31, rlm·`lasso_cv`와 같은 규칙: 없으면 `check_grad(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType`, `direction='all'`에 주면 `check_grad(): 'rng' is only used when direction='random'`). 같은 시드면 같은 값이다(scipy와는 난수가 달라 값이 다르다).

## §3. 최소제곱 (O1, O5)
```
least_squares(fun, x0, /, *, jac='2-point', bounds=None, method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, diff_step=None, max_nfev=None)
lsq_linear(A, b, /, *, bounds=None, method='trf', tol=1e-10, max_iter=None)
nnls(A, b, /, *, max_iter=None)
```
- **`least_squares`** (O1): ½Σ f_i(x)²를 `x0`에서 시작해 최소화한다. 커널은 nls의 것을 옮겼다(O6: nls는 이것을 쓰고 추론만 맡으며 결과는 비트까지 같다).
  - `method='trf'`(scipy의 기본): 신뢰 영역 반사법(scipy 1.18.1 `trf`의 'exact' 부분문제, x_scale = 1, 손실 함수 없음), 경계를 받는다. 경계가 없으면 scipy의 `trf_no_bounds`와 같은 걸음이다. 잔차가 변수보다 적어도(m < n) 된다. `'lm'`: MINPACK `lmder`의 Levenberg–Marquardt(factor 100, 야코비안 열 노름의 척도 = scipy 1.16부터의 `x_scale='jac'`), 경계를 받지 않고 m ≥ n이어야 한다(`least_squares(): method='lm' needs at least as many residuals as variables, not 2 < 3`). scipy의 `'dogbox'`는 없다.
  - `fun(x)`는 길이 m(≥ 1; 없으면 `ValueError`, 빈 `x0`도 `ValueError`, O28)의 1차원 값을 돌려준다(`sd.ndarray`, 실수의 시퀀스, numpy 배열). x는 새 float64 `sd.ndarray`다.
  - `jac`: 호출 가능한 객체((m, n)을 돌려준다; n = 1이면 (m,)도), `'2-point'`(기본), `'3-point'`(O21). 차분은 scipy 1.18.1 `approx_derivative`와 같다: 2점은 h = √u·sign(x)·max(1, |x|), 3점은 h = u^(1/3)·max(1, |x|)의 중앙 차분, 경계에 걸리면 scipy처럼 한쪽(2점은 반대쪽, 3점은 (x, x + h, x + 2h))으로 바꾸고 간격을 줄인다. 2점에서 한쪽의 값이 유한하지 않으면 반대쪽으로 한 번 더 해 보고(nls와 같다), 3점은 바로 `ValueError`다(O27). `'cs'`(복소 걸음)는 없다(사용자 함수에 complex를 넘기지 않는다).
  - `diff_step`: 차분의 상대 간격(scipy와 같다: h = diff_step·sign(x)·|x|, x + h가 x로 반올림되면 기본 간격), 양수 하나 또는 n개. `jac`가 호출 가능한 객체면 `ValueError`(`least_squares(): 'diff_step' is only used when 'jac' is '2-point' or '3-point'`)다(O29).
  - `bounds`: `None`(기본, 경계 없음; O19) 또는 `(lower, upper)`, 각각 수 하나 또는 n개, ±inf는 경계 없음. nls·`minimize`와 같은 꼴이다. `x0`가 경계 밖이면 `ValueError`, 경계에서 10⁻¹⁰·max(1, |경계|) 안이면 scipy처럼 그만큼 안으로 옮겨 시작한다(nls와 같다). `method='lm'`은 무한한 경계만 받는다.
  - 허용오차의 뜻과 기본값은 scipy의 것이다(nls §7과 같다). 세 값이 모두 u보다 작으면 `ValueError`(`least_squares(): at least one of 'ftol', 'xtol' and 'gtol' must be >= 2.2e-16 (the machine epsilon)`)다. scipy는 'lm'에서 셋 모두 u 이상을 요구한다.
  - `max_nfev`: 차분의 호출도 센 함수 계산 수의 상한이다(O18). None이면 `jac`이 호출 가능한 객체일 때 100n, 차분이면 100n(kn + 1)이다(O23, O25; k는 차분 야코비안 하나가 변수마다 부르는 수: `'2-point'` 1, `'3-point'` 2). scipy 1.18.1의 기본은 차분의 호출을 세지 않은 100n이므로, 반복 하나가 시험점 하나면 상한에 닿는 반복 수가 같다. nls의 기본은 그대로다(차분이면 100p(p + 1)). 상한에 닿으면 `status=0`, `converged=False`다. 마지막 반복점의 야코비안이 상한을 넘겨 계산될 수 있다(차분이면 kn번까지).
  - 결과: `x cost fun jac grad optimality active_mask converged status message n_iter nfev njev method`. `fun`·`jac`·`grad`는 `x`에서의 잔차, 야코비안, Jᵀf이고, `cost`는 ½‖f‖², `optimality`는 scipy와 같다('trf'는 Coleman–Li 척도를 곱한 ‖v·g‖∞, 'lm'은 ‖g‖∞). `active_mask`는 scipy `find_active_constraints`(rtol = xtol)의 int64 (n,)다. `status`·`message`는 scipy의 번호와 글이다(0 `The maximum number of function evaluations is exceeded.`, 1 `` `gtol` termination condition is satisfied.``, 2 ftol, 3 xtol, 4 `` Both `ftol` and `xtol` termination conditions are satisfied.``). `converged`는 `status > 0`이고 `cost`가 유한할 때다(O20).
  - 셈(O18): `nfev`는 `fun`을 부른 수 전부(차분의 호출 포함), `njev`는 야코비안을 계산한 수, `n_iter`는 반복이 계산한 야코비안 수(nls와 같다; scipy의 결과에는 반복 수가 없다). 'lm'은 scipy처럼 마지막 반복점의 야코비안을 결과로 주므로, 마지막 야코비안이 다른 점에서 계산됐으면 한 번 더 계산하고 센다(O24): 그때 `njev = n_iter + 1`이다. scipy와 견주면 'trf'의 `njev`와 'lm'의 `n_iter`가 scipy의 `njev`이고, scipy의 `nfev`는 `nfev − (2점 n, 3점 2n)·njev`다.
  - 사용자 함수가 낸 에러는 그대로 나간다. 반복 중 야코비안이 유한하지 않으면 `ValueError`(O26, `least_squares(): 'jac' is not finite at x = [...]`, 차분이면 `the differences of 'fun' are not finite`)다(nls와 같다).
  - 넣지 않은 것(O22): scipy의 `loss`(soft_l1·huber·cauchy·arctan)·`f_scale`, `x_scale`, `tr_solver='lsmr'`·`tr_options`, `jac_sparsity`, `callback`, `workers`, nls의 다듬기(G73). `verbose`는 없다(§1.4).
- **`lsq_linear`** (O5): ½‖Ax − b‖²를 `bounds` 안에서 최소화한다. `method='trf'`(scipy의 신뢰 영역 반사법)와 `'bvls'`(Stark–Parker). `max_iter`의 기본은 scipy처럼 trf 100, bvls n이다. `A`는 (m, n) `sd.ndarray`(희소 행렬과 `LinearOperator`는 받지 않는다), `b`는 (m,)이다. 결과는 `x cost fun optimality active_mask converged status message n_iter method`다(scipy의 `unbounded_sol`은 O5에서 정한다). scipy의 `lsq_solver='lsmr'`·`lsmr_tol`·`lsmr_maxiter`를 넣을지는 O5에서 정한다.
- **`nnls`** (O5): x ≥ 0에서 ‖Ax − b‖₂를 최소화한다(Lawson–Hanson, scipy 1.18.1의 구현과 같은 걸음). `max_iter`의 기본은 3n(scipy의 `maxiter`)이다. scipy는 `(x, rnorm)` 튜플을 돌려주고 상한에서 `RuntimeError`를 낸다. 이 라이브러리에서 결과를 `OptimizeResult`(`x cost fun converged status message n_iter`, rnorm = √(2·cost))로 할지 이름 붙은 튜플 `NNLSResult(x, rnorm)`로 할지는 O5에서 정한다(§9).

## §4. 스칼라 근 (O2)
```
root_scalar(f, /, *, method=None, bracket=None, fprime=None, fprime2=None, x0=None, x1=None, xtol=None, rtol=None, max_iter=None, options=None)
```
- `method`: `'brentq'`(Brent 1973, scipy의 C 구현과 같은 걸음), `'toms748'`(Alefeld–Potra–Shi 1995), `'bisect'`, `'newton'`, `'halley'`, `'secant'`. scipy의 `'brenth'`, `'ridder'`는 없다.
- `method=None`이면 scipy의 규칙으로 고른다: `bracket`이 있으면 `'brentq'`, 없고 `x0`와 `fprime`·`fprime2`가 있으면 `'halley'`, `fprime`만 있으면 `'newton'`, `x1`이 있으면 `'secant'`, `x0`만 있으면 `'newton'`(도함수는 2점 차분). 아무것도 없으면 `ValueError`(`root_scalar(): give 'bracket' or 'x0'`)다.
- 기본 허용오차는 scipy의 것이다: 구간법(brentq, toms748, bisect)은 `xtol` 2×10⁻¹², `rtol` 4u ≈ 8.88×10⁻¹⁶, `max_iter` 100; newton·halley·secant는 `xtol` 1.48×10⁻⁸, `rtol` 0, `max_iter` 50. 구간법의 기본을 "근을 끼는 구간이 1 ULP가 될 때까지"로 할지는 O2에서 정한다(구간법은 그 정확도에 이를 수 있다; 참값은 `tools/optimize_ref.py`).
- `fprime=True`(f가 (f, f′)를 돌려준다)처럼 scipy가 bool을 받는 꼴을 둘지는 O2에서 정한다. `options`에 남는 것은 toms748의 `k`(기본 1)다.
- 결과: `x fun converged status message n_iter nfev method`. 근을 끼지 않는 구간은 O2에서 정한다(§1.5).

## §5. 스칼라 최솟값 (O2)
```
minimize_scalar(fun, /, *, bracket=None, bounds=None, method=None, tol=None, options=None)
```
- `method`: `'brent'`(scipy의 `Brent`: 포물선 보간과 황금분할), `'golden'`, `'bounded'`(scipy `fminbound`, Brent의 fmin). None이면 `bounds`가 있을 때 `'bounded'`, 없으면 `'brent'`다(scipy).
- `bracket`: (a, b) 또는 (a, b, c). 두 점이면 scipy의 `bracket`처럼 내리막으로 넓혀 세 점을 찾는다. `bounds=(lower, upper)`는 `'bounded'`만 받는다(`minimize_scalar(): method='brent' takes no bounds`).
- 옵션(scipy의 기본): brent `xtol=1.48e-08`, `max_iter=500`; golden `xtol=1.4901161193847656e-08`, `max_iter=5000`; bounded `xatol=1e-05`, `max_iter=500`. `tol`은 brent·golden의 `xtol`, bounded의 `xatol`이다.
- 최솟값의 위치는 근처에서 f가 평평해 상대 정확도가 √u 정도에서 막힌다. 기본 허용오차가 그 크기다.
- 결과: `x fun converged status message n_iter nfev method`.

## §6. 배치 스칼라 (O2, O5)
```
find_root(f, init, /, *, tolerances=None, max_iter=None, callback=None)
bracket_root(f, xl0, xr0=None, /, *, xmin=None, xmax=None, factor=None, max_iter=1000)
find_minimum(f, init, /, *, tolerances=None, max_iter=100, callback=None)
bracket_minimum(f, xm0, /, *, xl0=None, xr0=None, xmin=None, xmax=None, factor=None, max_iter=1000)
```
- scipy `optimize.elementwise`의 이름과 뜻이다(O5). 원소마다 따로 풀고 사용자 함수는 배열 전체로 한 번에 부른다. 입력(`init`의 끝, `xl0` 등)은 서로 브로드캐스트하고 그 모양이 결과의 배치 모양이다.
- `find_root`: Chandrupatla(1997)의 구간법. `init=(xl, xr)`이고 f(xl)과 f(xr)의 부호가 달라야 한다(아니면 그 원소의 `status`가 −1). 멈춤은 scipy와 같다: |xr − xl| < xatol + |x_min|·xrtol 또는 |f(x_min)| ≤ fatol + |f_min0|·frtol, 기본 xatol = 4·tiny, xrtol = 4u, fatol = tiny, frtol = 0(tiny는 가장 작은 정규수).
- `find_minimum`: Chandrupatla(1998)의 최솟값 판. `init=(x1, x2, x3)`이고 x1 < x2 < x3, f(x1) ≥ f(x2) ≤ f(x3)(하나는 엄격). 멈춤: |x3 − x2|/2 ≤ |x2|·xrtol + xatol 또는 (f1 − 2f2 + f3)/2 ≤ |f2|·frtol + fatol, 기본 xrtol = √u, 나머지는 tiny.
- `bracket_root`, `bracket_minimum`: scipy와 같은 넓히기(`factor` 기본 2, `xmin`·`xmax`가 있으면 그쪽으로 1/factor씩 다가간다).
- `status`(scipy의 번호): 0 수렴, −1 구간이 맞지 않음(넓히기는 한계에 닿음), −2 `max_iter`, −3 유한하지 않은 값, −4 `callback`이 멈춤, −5 처음 구간이 `xmin ≤ xl0 < xr0 ≤ xmax`가 아님.
- 결과: `x fun bracket f_bracket converged status n_iter nfev`(배치 모양의 배열). `nfev`는 그 원소의 함수값 수다(호출 수가 아니다). scipy의 이름 `f_x`를 `fun`으로 할지, `tolerances`의 dict를 `xatol`·`xrtol`·`fatol`·`frtol` 키워드로 풀지, `callback`을 둘지는 O2에서 정한다.

## §7. 다변수 최소화 (O3, O4)
```
minimize(fun, x0, /, *, method=None, jac=None, hess=None, hessp=None, bounds=None, tol=None, callback=None, options=None)
```
- `method`: `'nelder-mead'`, `'bfgs'`, `'l-bfgs-b'`(O3), `'newton-cg'`, `'trust-ncg'`, `'trust-exact'`(O4). None이면 scipy처럼 `bounds`가 없으면 `'bfgs'`, 있으면 `'l-bfgs-b'`다. scipy의 제약 최적화(`constraints`, SLSQP·trust-constr·COBYLA)와 Powell·CG·TNC·dogleg·trust-krylov는 없다(O2).
- `jac`: 기울기 함수, `True`(fun이 (f, g)를 돌려준다), `'2-point'`, `'3-point'`, None(2점 차분). `hess`: 헤시안 함수 또는 `'2-point'`·`'3-point'`(기울기의 차분), `hessp`: 헤시안·벡터 곱 함수. scipy의 `HessianUpdateStrategy`(BFGS·SR1 객체)는 받지 않는다.
- `bounds=(lower, upper)`(nls·`least_squares`와 같은 꼴)는 `'l-bfgs-b'`·`'nelder-mead'`만 받는다. scipy의 변수마다 (min, max) 쌍의 시퀀스와 `Bounds` 객체를 받을지는 O3에서 정한다.
- 방법마다의 `options`(scipy 1.18.1의 이름과 기본값, 반복 상한은 O14의 이름; n은 변수 수):

  | 방법 | 옵션 |
  |---|---|
  | nelder-mead | `max_iter`·`max_nfev`(둘 다 None이면 200n; scipy의 `maxiter`·`maxfev`), `xatol=0.0001`, `fatol=0.0001`, `adaptive=False`, `initial_simplex=None` |
  | bfgs | `gtol=1e-05`, `norm=inf`, `eps=1.4901161193847656e-08`, `max_iter=None`(200n), `finite_diff_rel_step=None`, `xrtol=0`, `c1=0.0001`, `c2=0.9`, `hess_inv0=None` |
  | l-bfgs-b | `maxcor=10`, `ftol=2.220446049250313e-09`, `gtol=1e-05`, `eps=1e-08`, `max_nfev=15000`(scipy의 `maxfun`), `max_iter=15000`, `maxls=20`, `finite_diff_rel_step=None` |
  | newton-cg | `xtol=1e-05`, `eps=1.4901161193847656e-08`, `max_iter=None`(200n), `c1=0.0001`, `c2=0.9` |
  | trust-ncg, trust-exact | `initial_trust_radius=1.0`, `max_trust_radius=1000.0`, `eta=0.15`, `gtol=0.0001`, `max_iter=None`(200n) |

  `tol`은 scipy처럼 방법의 주된 허용오차(nelder-mead `xatol`·`fatol`, bfgs·l-bfgs-b·trust `gtol`, newton-cg `xtol`, l-bfgs-b는 `ftol`도)를 정한다. scipy의 `disp`, `return_all`, `iprint`, `workers`는 없다.
- 결과: `x fun jac converged status message n_iter nfev njev method`, 방법에 따라 `hess_inv`(BFGS는 (n, n); L-BFGS-B는 scipy가 `LinearOperator`를 준다: 행렬로 줄지 O3에서 정한다), `hess`·`nhev`(newton-cg, trust), `final_simplex`(Nelder–Mead).
- `callback`(scipy의 `callback(intermediate_result)`, `StopIteration`으로 멈춤)을 둘지, `x0`가 L-BFGS-B의 경계 밖일 때 scipy처럼 경계 안으로 옮길지 `ValueError`로 할지는 O3에서 정한다.

## §8. 방정식계의 근 (O5)
```
root(fun, x0, /, *, method='hybr', jac=None, tol=None, options=None)
```
- `method='hybr'`: MINPACK `hybrj`(`jac`를 주면)·`hybrd`(차분), Powell의 dogleg. `'lm'`: §3의 `lmder` 커널(m = n). scipy의 `broyden1`·`krylov` 같은 다른 방법을 넣을지는 O5에서 정한다.
- `jac`: 야코비안 함수, `True`(fun이 (f, J)를 돌려준다), None(hybrd의 전진 차분, `'lm'`은 `lmdif`의 차분).
- 옵션(scipy의 기본): hybr `xtol=1.49012e-08`, `max_nfev=None`(scipy의 `maxfev=0`; `jac`이 있으면 100(n + 1), 없으면 200(n + 1)), `band=None`, `eps=None`, `factor=100`, `diag=None`; lm `xtol=1.49012e-08`, `ftol=1.49012e-08`, `gtol=0.0`, `max_nfev=None`(scipy의 `maxiter=0`, 실제로는 함수 계산 수의 상한; `jac`이 있으면 100(n + 1), 없으면 200(n + 1)), `eps=0.0`, `factor=100`, `diag=None`. scipy의 `col_deriv`는 없다(야코비안은 언제나 (m, n)이다). `tol`은 `xtol`이다. scipy의 `root`도 이 두 방법에는 `callback`을 쓰지 않는다.
- 결과: `x fun converged status message nfev njev method`(`status`는 MINPACK의 info). scipy가 주는 MINPACK의 내부 값(`fjac`, `r`, `qtf`)을 줄지는 O5에서 정한다.

## §9. 마일스톤을 시작할 때 정할 것
O0에서 본문에 남긴 질문을 마일스톤별로 모았다. docs/design/optimize.md §9의 질문에 더해진 것이다.

- 모든 마일스톤에 걸친 질문(반복 상한의 이름, 방법 이름의 대소문자, `options` dict, `status`·`message`)은 정했다(O14–O17, §1.1).
- **O1**: 정했다(2026-11-20, docs/design/optimize.md §1.1 O18–O31).
- **O2**: 근을 끼지 않는 구간(`ValueError` 또는 `converged=False`), 구간법의 기본 허용오차, `root_scalar`의 결과 이름(§4), `fprime=True`, 배치의 `f_x`/`fun`, `tolerances` dict/키워드, `callback`, 배치의 콜백 규약.
- **O3–O4**: 기본 방법, `callback`, `hess_inv`의 꼴(L-BFGS-B), `bounds`의 꼴(쌍의 시퀀스, `Bounds`), 경계 밖의 `x0`, 유한하지 않은 함수값, `final_simplex`, L-BFGS-B의 `maxls`(선 탐색 상한)도 O14처럼 `max_ls` 꼴로 바꿀지.
- **O5**: hybr의 기본 허용오차, `root`의 다른 방법, `fjac`·`r`·`qtf`, `nnls`의 결과(§3), `lsq_linear`의 `unbounded_sol`과 `lsmr`.

## §10. 다른 라이브러리와의 대응 (안)
| 하는 일 | scipy 1.18.1 | R | sd.optimize |
|---|---|---|---|
| 구간 안의 근 | `brentq`, `toms748`, `bisect`, `root_scalar(method=)` | `uniroot` | `root_scalar(f, bracket=(a, b))` |
| 뉴턴·할리·할선 | `newton`, `root_scalar(method='newton')` | | `root_scalar(f, x0=, fprime=)` |
| 배치 근·최솟값 | `optimize.elementwise.find_root` 등 | | `find_root`, `bracket_root`, `find_minimum`, `bracket_minimum` |
| 스칼라 최솟값 | `brent`, `golden`, `fminbound`, `minimize_scalar` | `optimize` | `minimize_scalar` |
| 비선형 최소제곱 | `least_squares`, `leastsq` | `nls`, minpack.lm `nls.lm` | `least_squares` (곡선 적합과 추론은 `sd.regression.nls`) |
| 경계 있는 선형 최소제곱 | `lsq_linear`, `nnls` | `nnls::nnls` | `lsq_linear`, `nnls` |
| 다변수 최소화 | `minimize`, `fmin`, `fmin_bfgs`, `fmin_l_bfgs_b` | `optim`(Nelder-Mead, BFGS, L-BFGS-B) | `minimize(method=)` |
| 방정식계 | `root`, `fsolve` | | `root` |
| 수치 미분 | `approx_fprime`, `check_grad` | `numDeriv::grad` | `approx_fprime`, `check_grad` |
