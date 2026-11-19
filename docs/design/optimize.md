# 최적화 영역 설계 (sd.optimize)

> 상태: **설계 초안** (2026-11-20). 큰 결정(§1.1 O1–O31)은 사용자가 정했다. O0·O1을 구현했다(§8). 나머지는 안이고, 시작 전에 정할 것은 §9에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

`sd.optimize`(근 찾기, 최소화, 최소제곱, 수치 미분)의 확정 사항과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md)에 있다. 공개 함수와 시그니처의 안은 O0에서 [docs/api/optimize.md](../api/optimize.md)로 쓴다.
- 비교 대상은 scipy 1.18.1의 `scipy.optimize`와 `scipy.optimize.elementwise`, MINPACK, L-BFGS-B 3.0, R의 `optim`·`uniroot`·`optimize`다.
- 바닥은 `sd.linalg`(QR, Cholesky, SVD, 삼각 풀이, 고유값)와 배열 연산이다.

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-20)
| 번호 | 결정 |
|---|---|
| O1 범위 | 다섯 덩어리: 스칼라 근과 최솟값, 최소제곱, 다변수 최소화, 방정식계의 근, 수치 미분 (§1.2) |
| O2 빼는 것 | 확률적 최적화(SGD, Adam 등), `linprog`·`milp`, 제약 최적화(SLSQP, trust-constr, COBYLA), 전역 최적화(differential_evolution, basinhopping 등), `linear_sum_assignment`. 나중에 따로 정한다 |
| O3 API | scipy의 함수 이름과 인자 이름을 따른다. 입력과 옵션의 꼴(위치 인자와 키워드, 별칭 없음, 메시지)은 stardust의 규칙을 따른다 |
| O4 결과 이름 | regression과 같은 규칙: `converged`, `n_iter`, `nfev`, `njev` (scipy의 `success`, `nit`가 아니다) |
| O5 배치 | 스칼라 근과 최솟값에 배치 버전을 둔다(scipy `optimize.elementwise`의 이름): 원소마다 따로 풀고 콜백은 배열 전체로 한 번에 부른다 |
| O6 nls 옮기기 | `regression.nls`의 LM·TRF 커널과 정확한 야코비안을 `sd.optimize`로 옮긴다. `nls`는 그것을 쓰고 추론만 맡는다. 옮기기이므로 `nls`의 결과는 비트까지 같다 |
| O7 전용 풀이는 그대로 | special·stats 역함수의 근 찾기(`csrc/special/halley.hpp` 등), glm·rlm의 IRLS, 정규화의 좌표 하강, quantreg의 내부점법·단체법, MM의 S 단계 뉴턴은 바꾸지 않는다. 문제의 구조를 쓰는 풀이라 범용 최적화로 바꾸면 느려지고 정확도를 잃는다 |
| O8 확률적 최적화 | 이번 영역에 넣지 않는다. 자동 미분이 없어 쓸 곳이 적고, 결정론적 부분과 공유하는 코드도 거의 없다. 쓸 곳(회귀의 SGD 모형, 머신러닝·자동 미분 영역)이 생기면 그때 정한다 |
| O9 scipy와 같은 걸음 (O0) | 각 방법은 scipy 1.18.1과 같은 걸음이다: 같은 문제에서 반복점, `nfev`·`njev`, 멈춘 이유가 같다. 반올림의 순서가 다른 곳(합, 노름)은 끝자리가 다를 수 있다. 비트까지 같게 하지는 않는다 (사용자 결정 2026-11-20) |
| O10 통합 함수만 (O0) | `root_scalar`, `minimize_scalar`, `minimize`, `root`, `least_squares`만 두고 scipy의 개별 함수(`brentq`, `toms748`, `bisect`, `newton`, `fminbound`, `fmin`, `fmin_bfgs`, `fsolve`, `leastsq` 등)는 두지 않는다. 방법은 `method=`로 고른다 (사용자 결정 2026-11-20) |
| O11 curve_fit (O0) | 두지 않는다. 곡선 적합은 `regression.nls`가 한다 (사용자 결정 2026-11-20) |
| O12 args (O0) | 사용자 함수의 추가 인자 `args=()`·`kwargs`는 두지 않는다(nls와 같다). 클로저나 lambda로 넘긴다 (사용자 결정 2026-11-20) |
| O13 시험 문제와 대조 (O0) | Moré–Garbow–Hillstrom 35개와 scipy의 시험 함수로 문제 모음을 만들고 scipy 1.18.1과 비교한다. MINPACK(netlib)과 L-BFGS-B 3.0의 Fortran 원본을 gfortran으로 돌린 결과도 고정 데이터로 둔다(scipy가 옮긴 판과 원본을 둘 다 확인한다) (사용자 결정 2026-11-20) |
| O14 반복 상한의 이름 (O0) | 반복 상한은 어디서나 `max_iter`(반복 수)와 `max_nfev`(함수 계산 수)다. regression(`glm`·`lasso`의 `max_iter`, `nls`의 `max_nfev`)과 같다. scipy의 `maxiter`·`maxfev`·`maxfun`은 함수의 키워드에서도 `options`의 키에서도 쓰지 않는다 (사용자 결정 2026-11-20) |
| O15 방법 이름 (O0) | `method`는 소문자 이름만 받는다(`'bfgs'`, `'l-bfgs-b'`, `'nelder-mead'` …). 다른 대소문자는 소문자 꼴을 짚는 `ValueError`다 (사용자 결정 2026-11-20) |
| O16 방법마다의 옵션 (O0) | scipy처럼 `options=`의 dict로 받는다. 방법이 모르는 키는 `ValueError`다(scipy는 경고만 낸다) (사용자 결정 2026-11-20) |
| O17 `status`와 `message` (O0) | 방법마다 scipy의 번호와 글을 그대로 쓴다 (사용자 결정 2026-11-20) |
| O18 `nfev`가 세는 것 (O1) | `least_squares`의 `nfev`는 차분 야코비안의 함수 호출도 센다(두 방법 모두, nls와 같다). `max_nfev`도 그 호출을 넣은 수다. scipy의 'trf'(와 1.18.1의 'lm')는 차분의 호출을 세지 않으므로 차분을 쓰면 `nfev`가 scipy와 다르다 (사용자 결정 2026-11-20) |
| O19 `bounds`의 기본값 (O1) | `least_squares(bounds=None)`이 기본이고 경계가 없다는 뜻이다. `(lower, upper)`로 경계를 준다(nls와 같은 꼴) (사용자 결정 2026-11-20) |
| O20 유한하지 않은 마지막 값 (O1) | 멈춤 판정을 통과해도 마지막 `cost`가 유한하지 않으면 `converged=False`다(nls G89). `status`와 `message`는 scipy의 것 그대로다 (사용자 결정 2026-11-20) |
| O21 반복 안의 차분 (O1) | `jac=`는 `'2-point'`(기본), `'3-point'`, 호출 가능한 객체를 받고 `diff_step`을 둔다. `'cs'`는 없다(사용자 함수에 complex를 넘기지 않는다) (사용자 결정 2026-11-20) |
| O22 뒤로 미룬 것 (O1) | `loss`·`f_scale`, `x_scale`, `tr_solver='lsmr'`, `jac_sparsity`, `callback`, `workers`, 다듬기 옵션은 O1에 넣지 않는다 (사용자 결정 2026-11-20) |
| O23 `max_nfev`의 기본값 (O1) | 두 방법 모두 nls의 규칙: `jac`이 있으면 100n, 차분이면 100n(n + 1)(차분의 호출을 센 수). nls는 그대로다. 3점 차분은 O25가 고쳤다 (사용자 결정 2026-11-20) |
| O24 'lm'의 결과 야코비안 (O1) | 'lm'은 결과의 `jac`을 위해 마지막 반복점에서 야코비안을 계산한다(마지막 야코비안이 다른 점의 것일 때, scipy의 `jac(x)`와 같다). 그 호출을 `nfev`·`njev`에 센다(그때 `njev = n_iter + 1`) (사용자 결정 2026-11-20) |
| O25 차분의 `max_nfev` 기본 (O1) | O23을 바꾼다: 기본은 100n(kn + 1)이고 k는 차분 야코비안 하나가 변수마다 부르는 수(`'2-point'` 1, `'3-point'` 2)다. 호출 가능한 `jac`은 100n, 2점은 그대로 100n(n + 1)이다. 그래서 두 차분 모두 상한에 닿는 반복 수가 scipy(100n, 차분 호출을 세지 않음)와 같다 (사용자 결정 2026-11-20) |
| O26 유한하지 않은 야코비안 (O1) | 반복 중 야코비안(사용자 `jac` 또는 차분)이 유한하지 않으면 `ValueError`다(nls와 같다; scipy는 NaN으로 계속 간다) (사용자 결정 2026-11-20) |
| O27 차분의 한쪽 값 (O1) | `'2-point'`에서 한쪽 값이 유한하지 않으면 반대쪽으로 한 번 더 해 본다(nls의 fdjac2와 같다). `'3-point'`는 바로 `ValueError`다 (사용자 결정 2026-11-20) |
| O28 빈 입력 (O1) | 빈 `x0`(`'x0' is empty (no variables)`)와 값을 하나도 돌려주지 않는 `fun`(`'fun' must return at least 1 value`)은 `ValueError`다 (사용자 결정 2026-11-20) |
| O29 `diff_step`과 호출 가능한 `jac` (O1) | 함께 주면 `ValueError`다(`rng`의 규칙과 같다; scipy는 무시한다) (사용자 결정 2026-11-20) |
| O30 `approx_fprime`의 세부 (O1) | 길이 1인 배열을 돌려주는 f는 (1, n), 실수 하나를 돌려주는 f는 (n,)이다. 반올림에 묻힌 열을 0으로 두는 것(G88)은 하지 않는다. xk에서 f가 유한하지 않으면 기본(5점)은 `ValueError`, `epsilon`을 주면 scipy처럼 계산한다 (사용자 결정 2026-11-20) |
| O31 `check_grad`의 `rng` (O1) | `direction='random'`에 `rng`가 없으면 rlm·`lasso_cv`와 같은 꼴의 `TypeError`(`'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType`)다 (사용자 결정 2026-11-20) |

### 1.2 범위 (안)
| 덩어리 | 공개 함수 | 방법 |
|---|---|---|
| 스칼라 근 | `root_scalar` | brentq, toms748, bisect, newton, halley, secant |
| 스칼라 최솟값 | `minimize_scalar` | brent, golden, bounded |
| 배치 스칼라 (O5) | `find_root`, `bracket_root`, `find_minimum`, `bracket_minimum` | scipy `elementwise`의 방법(Chandrupatla 꼴) |
| 최소제곱 | `least_squares`, `lsq_linear`, `nnls` | lm, trf (O6); lsq_linear는 trf·bvls; nnls는 Lawson–Hanson |
| 다변수 최소화 | `minimize` | Nelder–Mead, BFGS, L-BFGS-B, Newton-CG, trust-ncg, trust-exact |
| 방정식계의 근 | `root` | hybr(MINPACK hybrj·hybrd), lm |
| 수치 미분 | `approx_fprime`, `check_grad` | 모수마다 간격을 고르는 5점 차분(nls의 것, O6) |

- `curve_fit`은 두지 않는다: 같은 일을 `regression.nls`가 추론까지 해서 한다(O11).
- `brentq` 같은 scipy의 개별 함수 이름은 두지 않는다(O10).

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 입력은 위치로(`fun`, `x0`), 옵션은 키워드로만. 별칭이 없다. float64 실수로 계산한다(bool·int64는 바꿔서). complex는 `TypeError`다.
- 같은 입력이면 스레드 수와 상관없이 같은 비트다. 사용자 함수가 결정론적이면 결과도 결정론적이다.
- **모든 호출은 정해진 시간 안에 끝난다.** 반복 방법은 함수 호출 수와 반복 수에 근거 있는 상한을 둔다. 상한에 닿으면 마지막 점과 `converged=False`를 돌려준다(regression G32와 같다).
- 경고를 내지 않는다. 수렴, 상한, 유한하지 않은 값은 결과 객체의 필드(`converged`, `status`, `message`)로 알린다.
- 참값은 Arb만 쓴다. mpmath는 쓰지 않는다.
- 영역 간 일관성이 먼저다. 결과 필드의 이름(O4), 메시지의 꼴(`least_squares(): …`), `rng` 인자, 배치의 축 규칙은 기존 영역을 따른다.

---

## 2. 입력과 결과

### 2.1 사용자 함수
- 사용자 함수는 Python이다. 반복은 C++ 커널이 돌고 함수값은 Python 콜백으로 받는다(nls와 같다, regression §3.7). 콜백을 부르는 동안 GIL을 쥔다. 무거운 선형대수(QR·SVD)만 linalg의 10 µs 규칙으로 GIL을 놓는다.
- 콜백에 넘기는 x는 `sd.ndarray`(float64, 1차원; 스칼라 함수는 Python float)다. 돌려받는 값은 `sd.ndarray`나 숫자 시퀀스를 받는다(nls의 `_values`와 같은 규칙). numpy 배열을 돌려줘도 받는다.
- 추가 인자를 넘기는 `args=()`는 두지 않는다(O12, nls와 같다).
- 배치 스칼라 함수(O5)의 콜백은 모양이 (...)인 `sd.ndarray`를 받아 같은 모양을 돌려준다. 이미 수렴한 원소는 다음 호출에서 뺀다(scipy `elementwise`와 같다).

### 2.2 결과 객체 `OptimizeResult`
- 필드: `x`, `fun`, `jac`(있을 때), `hess_inv`(BFGS·L-BFGS-B), `converged`, `status`, `message`, `n_iter`, `nfev`, `njev`, `nhev`(있을 때). 최소제곱은 `cost`, `fun`(잔차), `jac`, `grad`, `optimality`, `active_mask`(scipy `least_squares`와 같은 뜻).
- 이름은 regression의 결과와 같은 규칙이다(O4). scipy의 `success`·`nit`를 별칭으로 두지 않는다(별칭 없음).
- 배치 스칼라 결과는 필드마다 입력의 배치 모양이다(`x` (...), `converged` (...) bool, `nfev` (...) int64).
- 결과는 읽기 전용이고 배열 필드는 복사본을 돌려준다(regression G98, G113과 같다).
- `repr`는 짧은 요약이다. `summary()`는 두지 않는다(추론이 없다).

---

## 3. 수치 방법 (안)

### 3.1 수치 미분 (O1에서 구현)
- nls의 정확한 야코비안(regression §3.7, G70)을 `src/stardust/optimize/_diff.py`의 `Differences`로 옮겼다: 5점 중앙 차분, 2의 거듭제곱 간격, 변수마다 처음 한 번 간격을 고르기(2⁻⁸|x_j|에서 4배씩 줄이며 이웃 추정의 차이가 직전의 1/8보다 작은 동안), 경계나 유한하지 않은 쪽에서는 한쪽 5점 차분, 열의 반올림 수준(`noise`)과 이계 차분(`second`, nls의 뉴턴 다듬기). 반올림 수준에 묻힌 열을 0으로 두는 것(G88)은 nls만 켠다(`zero_noise`).
- `approx_fprime(xk, f, *, epsilon=None)`: epsilon이 None이면 위의 5점 차분(경계 없음), 수를 주면 scipy와 같은 앞쪽 차분(scipy의 √u를 주면 비트까지 같다). `check_grad`는 그 차이의 2-노름이고, `direction='random'`이면 `rng`의 표준정규 방향 하나로 잰다.
- 반복 안의 야코비안(`least_squares`, O21)은 scipy 1.18.1 `approx_derivative`의 '2-point'(h = √u·sign(x)·max(1, |x|), 경계에서 `_adjust_scheme_to_bounds`의 '1-sided')와 '3-point'(u^(1/3), '2-sided': 경계에서 (x, x + h, x + 2h))를 C++로 옮겼다(`csrc/optimize/diff.hpp`). 2점에서 한쪽 값이 유한하지 않으면 반대쪽으로 한 번 더 해 본다(nls의 fdjac2와 같다). nls의 반복은 지금처럼 MINPACK의 fdjac2(h = √u·|x|)다.

### 3.2 스칼라 근
- brentq: scipy의 C 구현(`brentq.c`, Brent 1973)과 같은 걸음과 판정. toms748: Alefeld–Potra–Shi(1995) 알고리즘 748(scipy의 Python 구현과 같은 걸음). bisect, secant, newton, halley: scipy `newton`의 규칙.
- 기본 허용오차는 scipy의 것(xtol 2×10⁻¹², rtol 4u)이다. 구간법(brentq, toms748, bisect)은 근을 끼는 구간이 1 ULP가 될 때까지 갈 수 있으므로, 기본을 "반올림까지"로 할지는 §9 O2 질문이다.
- 배치(`find_root`, `bracket_root`): scipy `elementwise`의 Chandrupatla(1997) 꼴. 원소마다 구간과 상태를 C++에서 들고, 남은 원소만 모아 콜백을 부른다.

### 3.3 스칼라 최솟값
- brent: scipy의 `Brent` 클래스(포물선 보간과 황금분할)와 같은 걸음. golden: 황금분할. bounded: scipy `fminbound`(Brent의 fmin).
- 배치(`find_minimum`, `bracket_minimum`): scipy `elementwise`의 Chandrupatla 꼴의 최솟값 판.
- 최솟값의 위치는 근처에서 f가 평평해 √u 상대 정확도가 한계다. 문서에 그 한계를 적는다.

### 3.4 최소제곱
- `least_squares` (O1에서 구현): nls의 커널을 `csrc/optimize/lsq.cpp`로 옮겼다(O6; `stardust._core.optimize.lsq_solve`). 'lm'은 MINPACK lmder(factor 100, mode 1), 'trf'는 scipy 1.18.1 `trf`('exact' 부분문제, x_scale = 1, 손실 함수 없음)를 같은 순서로 옮긴 것이다(regression §3.7, G69). 경계가 없으면 'trf'는 v = 1, dv = 0이 되어 scipy의 `trf_no_bounds`와 같은 걸음이다.
- 야코비안마다 m 크기의 일은 LAPACK의 Householder QR 하나(J = Q₀R₀, Q₀ᵀf)이고 나머지는 n 크기다(lmder의 열 피벗 QR은 R₀의 것, trf의 SVD는 [R₀d; D]의 것). m < n이면('trf'만) J에 0인 행을 n까지 붙인다(같은 문제다).
- 커널이 nls와 다른 것은 꼴뿐이다(nls를 부를 때는 지금까지와 같은 계산): 사용자 함수에 새 float64 `sd.ndarray`를 넘기기(nls는 Python float의 리스트), 돌려받은 값이 float64 `sd.ndarray`이고 모양이 맞으면 그대로 읽고 아니면 Python의 검사(`_conform`)를 거치기, x0에서의 값을 받기(모양을 알려고 Python에서 먼저 부른 것; nfev에 센다), scipy의 2점·3점 차분, 마지막 반복점의 잔차와 야코비안을 돌려주기('lm'은 마지막 야코비안이 다른 점의 것이면 x에서 한 번 더 계산한다: scipy의 `jac(x)`).
- 결과(`grad` = Jᵀf, `optimality`, `active_mask`)는 scipy의 정의다(`optimality`는 'trf'가 Coleman–Li 척도의 ‖v·g‖∞, 'lm'은 ‖g‖∞; `active_mask`는 `find_active_constraints`(rtol = xtol)).
- nls에만 있는 것(다듬기 G73, 정확한 야코비안의 공분산, 겹친 모수 G76)은 `nls` 쪽에 남겼다. scipy의 `loss`, `x_scale`, `f_scale`, `tr_solver='lsmr'`, `jac_sparsity`, `callback`, `workers`, 다듬기 옵션은 넣지 않았다(O22).
- `lsq_linear`: trf(scipy의 경계 있는 선형 최소제곱)와 bvls(Stark–Parker). `nnls`: Lawson–Hanson(scipy 1.18.1의 구현과 같은 걸음). 바닥은 linalg의 QR이다.

### 3.5 다변수 최소화
- Nelder–Mead: scipy의 `_minimize_neldermead`(적응 계수 옵션 `adaptive` 포함)와 같은 걸음.
- BFGS: scipy의 BFGS(Wolfe 선 탐색은 MINPACK-2 `dcsrch`, scipy의 `line_search_wolfe1`).
- L-BFGS-B: L-BFGS-B 3.0(Morales–Nocedal 2011, BSD). scipy 1.15부터 C로 옮긴 구현과 같은 걸음이 목표다.
- Newton-CG, trust-ncg(Steihaug–Toint CG), trust-exact(Moré–Sorensen): scipy의 구현과 같은 걸음. 헤세 곱은 사용자 `hess`·`hessp`나 기울기의 차분.
- 기울기를 주지 않으면 §3.1의 차분이다(scipy는 2점 차분).

### 3.6 방정식계의 근
- hybr: MINPACK hybrj(야코비안을 주면)·hybrd(차분). Powell의 dogleg. lm: §3.4의 lmder 커널을 그대로 쓴다.

### 3.7 scipy와 같은 걸음
- 각 방법은 scipy와 같은 순서의 계산을 목표로 한다. 그래서 같은 문제에서 반복점, `nfev`, 멈춘 이유가 scipy와 같다(반올림의 순서가 다른 곳은 끝자리가 다를 수 있다). 비트까지 같게 하지는 않는다(O9).

---

## 4. 정확도와 계약 (안)
- 반복 방법의 결과는 "멈춤 판정을 만족한 점"이다. 계약은 판정의 뜻 그대로다: 예를 들어 BFGS는 ‖∇f‖∞ ≤ gtol, brentq는 |x − x*| ≤ xtol + rtol·|x*|(근을 끼는 구간의 폭).
- 확인은 최적성 조건의 잔차(기울기, KKT, 근에서의 f)로 하고, 같은 문제의 scipy·MINPACK 결과와 비교한다.
- 구간법의 근은 Arb 참값과 비교한다(f를 Arb로 계산할 수 있는 시험 함수에서).
- nls는 옮긴 뒤에도 지금의 계약(다듬기 뒤 반올림 수준)을 그대로 지킨다(O6).

---

## 5. 성능 (안)
- 시간의 대부분은 사용자의 Python 함수다. 그래서 비교 기준은 (1) 같은 문제의 `nfev`·`njev`(scipy와 같거나 적게), (2) 콜백을 뺀 반복 하나의 오버헤드(scipy의 Python 반복 대비), (3) 배치 스칼라의 처리량(원소 수 10³–10⁶, scipy `elementwise`·원소마다 `brentq` 대비)이다.
- 크기: 스칼라, 변수 10·100·1000개(L-BFGS-B는 10⁴까지), 최소제곱 (m, n) = (10³, 10), (10⁵, 10), (10⁴, 100).
- 측정은 1·10스레드. 사용자 함수가 Python이므로 병렬은 배치의 커널 부분과 큰 선형대수에만 있다.

---

## 6. 층과 파일 (안)
- `src/stardust/optimize/`: 공개 함수(`_scalar.py`, `_minimize.py`, `_lsq.py`, `_root.py`, `_diff.py`), 결과 객체(`_result.py`), 입력 정리와 메시지
- `csrc/optimize/`: `scalar.hpp`(근·최솟값, 배치 상태), `diff.hpp`, `lsq.cpp`(nls에서 옮긴 lmder·trf), `nelder_mead.cpp`, `bfgs.cpp`(+ `dcsrch`), `lbfgsb.cpp`, `trust.cpp`, `hybrd.cpp`, `nnls.cpp`; 바인딩 `csrc/bind/optimize_bind.cpp`
- `regression.nls`는 `csrc/optimize/lsq.cpp`와 `_diff.py`를 쓴다(O6). `csrc/regression/nls.cpp`는 없앴다(O1).
- `tests/optimize/`, `bench/optimize/`

---

## 7. 참값과 검증
O0에서 만든 것(2026-11-20). 문제 모음은 `tests/optimize/`, 고정 데이터는 `tests/optimize/data/`(1.8 MB), 만드는 도구는 `tools/`에 있다.

- **시험 문제** (O13)
  - `mgh.py`: Moré–Garbow–Hillstrom(1981)의 35개. 잔차 F: Rⁿ → Rᵐ와 해석적 야코비안을 한 번 쓰고 최소제곱(½ΣF²), 최소화(f = ΣF², 기울기 2JᵀF, 헤시안은 `ad.Dual2`), m = n인 방정식계에 함께 쓴다. 경우는 110개다: MINPACK 최소제곱 드라이버(file17·file19, 자료 file22)의 53개와, 드라이버에 없는 17개 문제를 논문의 크기에서 x0·10x0·100x0으로 57개. 시작점의 배수는 MINPACK `INITPT`의 규칙(0인 시작점은 배수의 벡터)이고, 논문이 열어 둔 것은 MINPACK을 따랐다(나선 골짜기의 x1 = 0에서 θ = sign(x2)/4). Gulf의 m은 99로 정했다(논문은 3 ≤ m ≤ 100). 논문의 최적값(`KNOWN`, 6자리)과 정확한 최소점(`MINIMIZERS`)이 함께 있다.
  - `systems.py`: HYBRD·HYBRJ 드라이버(file15·file16, 자료 file21)의 14개 방정식계, 55개 경우. 넷(로젠브록의 식 순서, Wood·Watson·가변 차원은 최소제곱 문제의 기울기)이 MGH의 잔차와 다르다.
  - MINPACK 드라이버에 있는 문제는 `SSQFCN`·`SSQJAC`·`VECFCN`·`VECJAC`의 연산 순서 그대로 썼다. 그래서 gfortran(`-ffp-contract=off`)으로 계산한 값과 비트까지 같다(드라이버의 시작점과 둘째 점, 216점; `data/minpack_functions.json`).
  - `functions.py`: scipy의 시험 함수. Rosenbrock(`rosen`과 도함수·헤시안·곱), scipy `CheckOptimize`의 최대 엔트로피 문제, L-BFGS-B 3.0 driver1의 경계 있는 문제(n = 25); 스칼라 근 204개(scipy `_tstutils`의 original 5, Alefeld–Potra–Shi 154, Chandrupatla 45); 스칼라 최솟값 57개(scipy 2, Chandrupatla 최소화 논문 55와 그 구간 찾기); 선형 최소제곱 12개(scipy `test_lsq_linear`·`test_nnls`의 것과 결정적인 행렬 셋).
  - scipy `_tstutils`의 이계도함수 셋이 틀렸다: `aps02_fpp`(부호), `aps09_fpp`(−12n(1 − nx)², 맞는 것은 −12n²(1 − nx)²), `aps10_fpp`(지수 인자의 −n 자리에 −nx). 모음에는 고친 식을 쓴다.
  - `ad.py`: 같은 식을 세 가지 수로 계산한다. `FLOAT`(libm 함수, IEEE 특수값: 넘침은 inf, 0으로 나누기는 inf), `ARB`(python-flint), `Dual2`(2차 전진 자동 미분: 정확한 기울기와 헤시안).
  - 검사(`tests/optimize/test_problems.py`): 모든 야코비안·기울기·헤시안·도함수를 Arb 320비트 중앙 차분(간격 2⁻⁸⁰)과 `Dual2`로 비교한다(상대 10⁻¹² 안; 실제 최대 2.4×10⁻¹⁵). 논문의 최소점에서 최적값, Fortran 값과 비트, scipy의 정의와 같음(위의 세 이계도함수 빼고)을 확인한다. Chandrupatla의 두 논문이 적은 횟수(근 45개의 함수 계산 수, 최소화 55개의 반복 수)를 scipy `elementwise`가 논문의 허용오차로 그대로 낸다: 옮겨 쓴 문제와 구간이 맞다는 증거다.
- **근의 참값** (`tools/optimize_ref.py` → `data/scalar_roots.json`): 근 문제 204개마다, 코드에 적힌 함수(상수는 코드의 double)의 근을 1 ULP로 찾는다. double x에서 f의 부호를 Arb로 정밀도를 올리며 확정하고, double의 비트 순서로 이분한다(64번 안). 154개는 이웃한 두 double, 50개는 f가 정확히 0인 double이다(x⁴ − 1의 1, 코드가 0을 돌려주는 가지 등). scipy의 brentq·toms748·bisect가 수렴한 근은 모두 이 구간에서 xtol + rtol·|x| 안이거나, double로 계산한 f가 정확히 0인 점이다(Chandrupatla 7번 x·exp(−x⁻²)는 |x| < 0.037에서 double로 0이 된다).
- **scipy 1.18.1 대조** (`tools/gen_optimize_scipy.py` → `data/scipy_*.json`, 3,781번): §1.2의 모든 방법을 scipy의 기본값으로 돌린 결과(x, fun 또는 cost, nfev·njev·nhev·nit, status, success, message, 또는 낸 예외). `least_squares` 'trf'·'lm' × 해석적·'2-point' 야코비안 × 110경우와 x0 둘레의 상자 경계, `minimize` 6방법(BFGS·L-BFGS-B는 차분도) × 110경우와 Rosenbrock·최대 엔트로피·driver1, `root` hybr·lm × 야코비안 유무 × 55계, `root_scalar` 6방법, `minimize_scalar` 3방법, `elementwise` 넷(scipy의 허용오차와 논문의 것), `lsq_linear`·`nnls`, `approx_fprime`·`check_grad`. 테스트는 표본을 다시 돌려 설치된 scipy가 기록과 같은지도 본다.
- **Fortran 원본** (`tools/gen_optimize_fortran.py`): netlib의 MINPACK(file05)과 드라이버 넷, L-BFGS-B 3.0을 gfortran 16.2(`-O2 -ffp-contract=off`)로 빌드했다. 원본은 저장소에 넣지 않고 결과만 넣었다: 드라이버를 그대로 돌린 요약표(`minpack_drivers.txt`, `lbfgsb_drivers.txt`), 같은 루틴(LMDER1, LMDIF1, HYBRJ1, HYBRD1)을 같은 경우에서 돌리며 모든 계산점을 기록한 결과(`fortran_minpack.json`), scipy의 드라이버 고리(`_minimize_lbfgsb`)를 흉내 낸 setulb를 MINPACK 18문제(f = ΣF², 경계 없음과 x0 둘레의 상자)와 driver1에 돌린 결과(`fortran_lbfgsb.json`). driver1(m = 5)은 원본 드라이버와 같은 23반복, 28계산, f = 1.0834900834300615×10⁻⁹이다.
- **scipy의 옮긴 판과 원본** (`data/fortran_vs_scipy.json`; scipy는 드라이버와 같은 옵션: `leastsq`(ftol = xtol = √u, gtol = 0, maxfev), `fsolve`(diag = 1, 즉 mode 2), `minimize(method='L-BFGS-B')`(같은 m, factr, pgtol)):

  | 루틴 | 경우 | 계산점이 비트까지 같은 순서 | 같은 호출 수 | 같은 info | 마지막 x 상대 차이 ≤ 10⁻¹² | ≤ 10⁻⁶ |
  |---|---:|---:|---:|---:|---:|---:|
  | LMDER1 / `leastsq(Dfun=)` | 53 | 2 | 47 | 44 | 37 | 45 |
  | LMDIF1 / `leastsq` | 53 | 1 | 34 | 52 | 13 | 31 |
  | HYBRJ1 / `fsolve(fprime=)` | 55 | 4 | 42 | 50 | 37 | 44 |
  | HYBRD1 / `fsolve` | 55 | 4 | 41 | 48 | 33 | 44 |
  | setulb / L-BFGS-B | 83 | 7 | 74 (nfev와 nit) | | 62 | 79 |

  - scipy 1.15부터의 C 판은 원본과 같은 알고리즘이지만 계산의 반올림이 다르다. 같은 x0에서 시작해 첫 시험점이 이미 끝자리 몇 ULP 다르다(LMDER1에서 53경우 가운데 47경우가 첫 시험점에서 갈린다). 함수값은 두 쪽이 같은 비트이므로(위의 216점) 차이는 루틴 안에서 생긴다. Fortran을 FMA로 빌드해도(`-ffp-contract=fast`) 같은 정도로 다르다.
  - 그 차이가 잘 조건화된 문제에서는 커지지 않는다: Bard, Osborne 1·2 등은 info가 같고 마지막 x가 10⁻¹³ 안이다(Meyer는 멈춘 판정이 2와 3으로 갈리지만 x는 1.4×10⁻¹⁰ 안이다). 해가 특이하거나 유일하지 않은 문제(Powell 특이 함수의 x → 0, 랭크 1 선형, Chebyquad의 여러 근, Brown–Dennis)는 경로가 갈려 다른 점에서 멈추고 멈춘 이유도 다를 수 있다. 눈에 띄는 것: HYBRJ1·HYBRD1에서 Brown almost-linear n = 30·40은 Fortran이 info 1(수렴)인데 scipy는 info 5(진전 없음)이고, LMDER1의 Powell 특이 함수는 Fortran info 4, scipy info 8이다.
  - scipy의 MINPACK 감싸개는 MINPACK이 시작하기 전에 x0에서 fun을 한 번 더 부른다(모양 확인 호출과 별개로; MINPACK의 nfev에는 들어가지 않는다).
  - L-BFGS-B의 멈춤 글이 다르다: 원본 `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`, scipy `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`(사이 띄어쓰기). 83경우 가운데 1경우는 멈춘 판정도 다르다.
  - 그래서 O9("scipy와 같은 걸음")의 대상은 scipy 1.18.1이다. 이 라이브러리의 커널은 scipy의 C 판을 옮기므로, O1–O5의 대조는 `scipy_*.json`이 기준이고 Fortran의 결과는 원본에서 얼마나 떨어지는지를 보는 데 쓴다.
- nls를 옮긴 뒤 G6의 테스트 전부와 골든 출력(NIST 27개 × 시작점 2개, 두 방법)이 비트까지 같아야 한다(O1). NIST StRD 비선형 27개는 이미 `tests/regression/data/nls/`에 있다. CUTEst는 쓰지 않는다.

O1에서 확인한 것(2026-11-20):

- **nls의 비트 동일**: 옮기기 전에 골든 출력을 만들었다: NIST 27개 × 시작점 2개 × 'lm'·'trf' × 차분·사용자 야코비안(216적합), 같은 216경우의 시작점 둘레 상자 경계(108적합), 경계·가중치·0인 가중치·`max_nfev`·느슨한 허용오차·유한하지 않은 시험점·x0에서 유한하지 않음·겹친 모수의 14경우, 모두 338개. 결과의 모든 필드(params, bse, tvalues, pvalues, resid, fittedvalues, ssr, scale, llf, aic, bic, nobs, df_model, df_resid, rank, converged, n_iter, nfev, njev, cov_params, conf_int, summary 글, repr, 경계에 걸린 모수, 멈춘 번호, 다듬기 수, predict, predict_interval, 에러의 글)를 float의 16진 표기로 적었다. 커널을 옮긴 뒤와 정확한 차분을 옮긴 뒤에 다시 만든 출력이 바이트까지 같다(`cmp`). `tests/regression/test_nls.py`도 모두 통과한다.
- **scipy 1.18.1과 같은 걸음** (O9; `tests/optimize/test_least_squares.py`). 기록된 실행(`data/scipy_least_squares.json`)과 견준 수(110경우, 상자는 47경우; nfev는 차분의 호출을 뺀 scipy의 셈, njev는 'trf'의 njev와 'lm'의 `n_iter`, x는 max|Δx|/max(‖x‖∞, ‖x0‖∞) ≤ 10⁻⁶):

  | 방법 | 같은 status | 같은 nfev | 같은 njev | 같은 x | converged = success |
  |---|---:|---:|---:|---:|---:|
  | trf 해석적 | 110 | 107 | 107 | 108 | 110 |
  | trf 해석적, 상자 | 47 | 45 | 46 | 45 | 47 |
  | trf 2점 | 106 | 97 | 94 | 93 | 110 |
  | trf 2점, 상자 | 47 | 42 | 43 | 41 | 47 |
  | lm 해석적 | 108 | 101 | 97 | 102 | 110 |
  | lm 2점 | 109 | 84 | 110(기록 없음) | 87 | 110 |

  - 사용자 함수를 부른 점을 차례대로 scipy와 견주면(실행하며 기록), 해석적 야코비안의 'trf'는 110경우 가운데 103경우가 호출 수가 같고 모든 점이 10⁻¹⁰ 안이다(106경우가 10⁻⁶ 안; 상자 43/47, 'lm' 81/110). x0와 첫 야코비안의 점들은 비트까지 같다. 첫 시험점부터 끝자리가 다른데, 같은 J에서 이 커널은 QR을 먼저 하고 n 크기에서 SVD·피벗 QR을 하고 scipy는 m × n에서 한다(반올림의 순서가 다르다).
  - 갈리는 곳은 반올림이 결정하는 곳이다: 거의 특이하거나 정확히 계수가 모자란 야코비안(선형 계수 1 함수 33·34, Watson n = 9·12, x0에서 두 열이 같은 Biggs EXP6, 100x0의 Gaussian(야코비안이 사실상 0), cond(J(x0)) ≈ 10¹⁶인 100x0의 Brown almost-linear: 첫 신뢰 영역 걸음이 3×10⁻³ 다르다), MINPACK의 `nsing`처럼 정확한 0을 보는 판정, ftol·xtol 판정이 반올림 수준의 감소로 갈리는 곳(선형 함수 32). 차분은 반복점의 끝자리 차이를 다음 야코비안의 10⁻⁸ 수준 차이로 키우므로 2점 차분의 경로가 더 일찍 갈린다.
  - 3점 차분과 `diff_step`은 기록이 없어 실행하며 견줬다: 잘 조건화된 7경우에서 status, nfev(차분 2n·njev를 뺀 수), x가 같다.
  - 3점 차분의 `max_nfev` 기본이 O23의 100n(n + 1)일 때는 상한에 닿아 scipy는 수렴하고 이쪽은 `status=0`인 경우가 MGH에 4개(trf 2, lm 1, 상자 trf 1) 있었다: 3점 야코비안은 2n번을 부르므로 상한에 닿는 반복 수가 scipy의 절반쯤이었다. O25(100n(2n + 1))로 바꾼 뒤 그 넷은 scipy와 같은 곳에서 멈춘다(차분을 뺀 nfev가 같다). 다시 견준 3점(실행하며 견줌, 110경우): 같은 status trf 107, lm 105, 상자 47/47; converged = success는 trf 110, lm 109(mgh15 100x0은 scipy가 상한 100n에 닿고 이쪽은 수렴한다), 상자 47.
- **NIST StRD** (27개 × 시작점 2개 = 54적합, scipy를 실행하며 견줌): 같은 status는 trf 해석적 52, trf 2점 49, lm 해석적 50, lm 2점 52이고, x가 10⁻⁶ 안인 것은 54, 50, 51, 52다. 갈리는 것은 MGH09, MGH10, MGH17, BoxBOD, Bennett5(대부분 시작점 1)다. 인증값을 6자리 이상 맞히는 적합의 수는 scipy와 같거나 많다(trf 해석적 35/35, 2점 30/29, lm 해석적 35/34, 2점 31/31).
- **Fortran LMDER1** (드라이버의 53경우, tol = √u, gtol = 0, maxfev = 100(n + 1)): 'lm' 해석적이 같은 info 48, 같은 호출 수(f와 야코비안) 47, x가 10⁻¹² 안 35, 10⁻⁶ 안 44다(scipy `leastsq`는 44, 47, 37, 45). 커널을 nls의 fdjac2 차분으로 돌리면 LMDIF1과 같은 info 50, 같은 호출 수 30, x 13·31이다(scipy `leastsq`는 52, 34, 13, 31).
- **approx_fprime**: `epsilon=2**-26`(scipy의 기본)이면 MGH 47경우의 기울기와 야코비안이 scipy와 비트까지 같고 `check_grad`는 10⁻¹² 안이다. 기본(정확한 차분)은 해석적 도함수와의 상대 차이가 최대 1.2×10⁻¹¹(야코비안), 6.4×10⁻¹¹(기울기)이다.
- 속도는 `bench/optimize/o1/speed.md`: 작은 문제의 반복당 자기 비용이 scipy의 1/4–1/20, (10⁴, 100)의 'lm'은 반복당 6.5 ms 대 42 ms, m = n = 40에서만 비슷하다.

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 |
|---|---|
| O0 준비 | API 문서 초안, 결과 객체의 틀, 시험 문제 모음(MGH 35, scipy 시험 함수), 대조 도구, 벤치마크 기준선(scipy). 완료(2026-11-20): §7, docs/api/optimize.md, `bench/optimize/o0/` |
| O1 수치 미분과 최소제곱 | `approx_fprime`, `check_grad`, nls 커널을 `least_squares`로 옮기기(nls 비트 동일). 완료(2026-11-20): §3.1, §3.4, §7, `bench/optimize/o1/` |
| O2 스칼라 | `root_scalar`, `minimize_scalar`, 배치 `find_root bracket_root find_minimum bracket_minimum` |
| O3 최소화 1 | `minimize`: Nelder–Mead, BFGS, L-BFGS-B |
| O4 최소화 2 | `minimize`: Newton-CG, trust-ncg, trust-exact |
| O5 방정식계와 선형 최소제곱 | `root`(hybr, lm), `lsq_linear`, `nnls` |
| O6 최적화와 일관성 점검 | 커널 속도, 메시지·시그니처의 영역 간 일관성, 코드 리뷰 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.

---

## 9. 마일스톤을 시작하기 전에 물을 것
- **O0**: 정했다(2026-11-20, O9–O13). O0에서 API 초안을 쓰며 생긴 질문 가운데 모든 마일스톤에 걸친 넷(반복 상한의 이름, 방법 이름의 대소문자, `options` dict, `status`·`message`)도 정했다(O14–O17). 마일스톤별로 남은 질문은 docs/api/optimize.md §9에 있다(`nfev`가 차분을 세는지, `least_squares`의 `bounds` 기본값, `root_scalar`의 결과 이름, `nnls`의 결과 등).
- **O1**: 정했다(2026-11-20, O18–O23, 구현하며 생긴 질문은 O24–O31). 옮기는 순서는 설계대로 먼저 옮기고 비트를 확인한 뒤 공개 함수를 만들었다.
- **O2**: 구간법의 기본 허용오차(scipy 값 또는 반올림까지), 근을 끼지 않는 구간의 결과(ValueError 또는 NaN과 `converged=False`), 배치의 콜백 규약(남은 원소만 넘기기).
- **O3–O4**: 기본 방법(scipy: 경계가 없으면 BFGS, 있으면 L-BFGS-B), `callback`을 둘지, `hess_inv`를 결과에 둘지, 유한하지 않은 함수값의 처리(걸음 거절 또는 멈춤).
- **O5**: hybr의 기본 허용오차(scipy 1.49×10⁻⁸), `root`의 다른 방법(broyden 등)을 넣을지.
