# 최적화 영역 설계 (sd.optimize)

> 상태: **설계 초안** (2026-11-20). 큰 결정(§1.1 O1–O76)은 사용자가 정했다. O0–O4를 구현했다(§8). 나머지는 안이고, 시작 전에 정할 것은 §9에 있다.
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
| O12 args (O0) | 사용자 함수의 추가 인자 `args=()`·`kwargs`는 두지 않는다(nls와 같다). 클로저나 lambda로 넘긴다. 배치 스칼라 함수만 예외다(O39) (사용자 결정 2026-11-20) |
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
| O32 근을 끼지 않는 구간 (O2) | `root_scalar`에서 f(a)와 f(b)의 부호가 같으면 `ValueError`다(scipy와 같다). 배치 `find_root`는 그 원소의 `status`가 −1이다(scipy와 같다) (사용자 결정 2026-11-20) |
| O33 구간법의 기본 허용오차 (O2) | scipy의 것이다: brentq·toms748·bisect는 xtol 2×10⁻¹², rtol 4u, max_iter 100; newton·halley·secant는 xtol 1.48×10⁻⁸, rtol 0, max_iter 50. 기본은 "1 ULP까지"가 아니다. `xtol=0`이면 더 간다는 것을 문서에 적는다 (사용자 결정 2026-11-20) |
| O34 `fprime=True` (O2) | 두지 않는다. `fprime`·`fprime2`는 호출 가능한 객체나 None이다 (사용자 결정 2026-11-20) |
| O35 배치 결과의 이름 (O2) | scipy의 `f_x` 대신 `fun`이다. `f_bracket`은 그대로 둔다 (사용자 결정 2026-11-20) |
| O36 배치의 허용오차 (O2) | scipy의 `tolerances` dict 대신 키워드 `xatol`, `xrtol`, `fatol`, `frtol`이다 (사용자 결정 2026-11-20) |
| O37 `callback` (O2) | `root_scalar`, `minimize_scalar`, 배치 함수 모두 `callback`이 없다. 그래서 scipy의 status −4(`callback`이 멈춤)도 없다 (사용자 결정 2026-11-20) |
| O38 배치의 호출 규약 (O2) | 배치 함수는 사용자 함수를 아직 끝나지 않은 원소만 모은 1차원 배열로 부른다(scipy의 규약). 문서에 모양의 계약을 적는다 (사용자 결정 2026-11-20) |
| O39 배치의 `args` (O2) | 배치 함수(`find_root`, `bracket_root`, `find_minimum`, `bracket_minimum`)는 scipy처럼 `args=()`를 받는다. 값마다 배치 모양과 브로드캐스트하고, 호출마다 x와 같은 남은 원소로 줄여 넘긴다. 다른 함수는 O12 그대로다 (사용자 결정 2026-11-20) |
| O40 `root_scalar`의 `fun` (O2) | 없다(scipy와 같다) (사용자 결정 2026-11-20) |
| O41 `root_scalar`의 셈 (O2) | f, f′, f″의 호출을 `nfev`, `njev`, `nhev`로 따로 센다 (사용자 결정 2026-11-20) |
| O42 `f_bracket` (O2) | scipy의 이름 그대로다 (사용자 결정 2026-11-20) |
| O43 newton·secant의 멈춤 글 (O2) | 도함수가 0이거나 secant의 두 함수값이 같아 멈추면 scipy가 경고로 내는 글을 `message`에 두고 `converged=False`다 (사용자 결정 2026-11-20) |
| O44 brent·golden의 `status` (O2) | bounded의 0–2에 3(구간을 못 찾음)과 4(구간 찾기의 상한; scipy는 `RuntimeError`)를 더한다 (사용자 결정 2026-11-20) |
| O45 toms748의 유한하지 않은 값 (O2) | toms748에서 f가 유한하지 않으면(구간의 끝에서도) `ValueError`가 아니라 brentq·bisect의 NaN처럼 status −3의 결과다. scipy와 다르다 (사용자 결정 2026-11-20) |
| O46 toms748의 `k` (O2) | 100을 넘으면 `ValueError`다 (사용자 결정 2026-11-20) |
| O47 구간의 끝 (O2) | 끝은 유한해야 한다. brentq·bisect는 (b, a)를 받고 toms748은 a < b여야 한다. a == b는 `ValueError`다(근을 끼지 않는다). `minimize_scalar`의 `bounds`는 lower == upper를 받는다 (사용자 결정 2026-11-20) |
| O48 배치의 실수 입력 (O2) | 입력이 모두 실수이면 결과의 필드는 Python 수다. 첫 호출도 1차원 배열이다 (사용자 결정 2026-11-20) |
| O49 배치 입력의 NaN·inf (O2) | 그 원소만 status −3 또는 −5이고 나머지는 풀린다 (사용자 결정 2026-11-20) |
| O50 기본 방법 (O3) | scipy를 따른다: `bounds`가 없으면 `'bfgs'`, 있으면 `'l-bfgs-b'` (사용자 결정 2026-11-20) |
| O51 `bounds`의 꼴 (O3) | `(lower, upper)` 하나뿐이다(각각 수 하나 또는 n개, ±inf는 경계 없음; `least_squares`·nls와 같다). 변수마다의 쌍의 시퀀스와 `Bounds` 객체는 받지 않는다. `'l-bfgs-b'`와 `'nelder-mead'`만 받는다 (사용자 결정 2026-11-20) |
| O52 경계 밖의 `x0` (O3) | `ValueError`다(scipy는 말없이 경계 안으로 옮긴다) (사용자 결정 2026-11-20) |
| O53 유한하지 않은 값 (O3) | 반복 중의 함수값·기울기가 유한하지 않으면 scipy의 처리를 따른다(O9: 선 탐색이 걸음을 줄이거나 실패하고, scipy의 `status`·`message`, `converged=False`). `x0`에서 유한하지 않으면 `ValueError`다 (사용자 결정 2026-11-20) |
| O54 `callback` (O3) | 두지 않는다 (사용자 결정 2026-11-20) |
| O55 `hess_inv` (O3) | L-BFGS-B는 저장된 쌍으로 만든 (n, n) 행렬이고 처음 읽을 때 계산한다. BFGS는 그 (n, n) 행렬이다 (사용자 결정 2026-11-20) |
| O56 `final_simplex` (O3) | Nelder–Mead는 scipy처럼 마지막 꼭짓점들과 그 함수값을 준다 (사용자 결정 2026-11-20) |
| O57 `max_ls` (O3) | L-BFGS-B의 선 탐색 상한은 `max_ls`다(scipy `maxls`; O14와 같은 꼴). `maxcor`는 상한이 아니므로 이름을 그대로 둔다 (사용자 결정 2026-11-20) |
| O58 고정 변수 (O3) | `minimize`는 lower = upper를 scipy처럼 받는다: 모두 고정이면 `fun`을 한 번 부르고 끝내고, 일부가 고정이고 기울기가 차분이면 그 변수를 빼고 푼다. 뺀 변수가 있으면 결과에 `hess_inv` 필드가 없다(None이 아니다; 다른 없는 필드처럼 `AttributeError`). `least_squares`는 그대로 lower < upper다 (사용자 결정 2026-11-20) |
| O59 L-BFGS-B ABNORMAL의 `fun` (O3) | 돌려주는 `x`에서의 값이다(scipy는 마지막 시험점의 값). 걸음은 scipy와 같다 (사용자 결정 2026-11-20) |
| O60 L-BFGS-B `hess_inv`의 순서 (O3) | scipy처럼 저장된 쌍을 저장 순서로 쓴다(scipy의 `todense()`와 같은 행렬) (사용자 결정 2026-11-20) |
| O61 BFGS의 H 갱신 (O3) | 지금은 scipy의 n³ 갱신(행렬곱 두 번)을 둔다. n²의 갱신은 O6에서 다시 본다 (사용자 결정 2026-11-20) |
| O62 옵션의 검사 (O3) | 그대로 둔다: `eps`·`finite_diff_rel_step`을 쓰이지 않는 `jac`과 주면 `ValueError`(O29를 넓힘), `c1`·`c2`는 반복 전에 검사, `tol`과 같은 옵션 키를 함께 주면 `ValueError` (사용자 결정 2026-11-20) |
| O63 `hess`·`hessp`의 꼴 (O4) | `hess(x)`는 (n, n) 배열(`sd.ndarray`, numpy 배열, 중첩 시퀀스), `hessp(x, p)`는 (n,) 배열을 돌려준다. `LinearOperator`, 희소 행렬, `HessianUpdateStrategy`는 받지 않는다(늘 쓰는 꼴의 `TypeError`) (사용자 결정 2026-11-20) |
| O64 차분 헤시안 (O4) | `hess='2-point'`·`'3-point'`(`jac`의 차분)을 O21의 `jac`처럼 받는다. `'cs'`는 없다. `hess`·`hessp`가 없는 newton-cg는 scipy처럼 기울기의 차분으로 헤세 곱을 만든다 (사용자 결정 2026-11-20) |
| O65 `jac`이 꼭 있다 (O4) | newton-cg, trust-ncg, trust-exact는 `jac`이 있어야 한다(scipy의 'Jacobian is required'). 없으면 stardust 꼴의 `ValueError`다 (사용자 결정 2026-11-20) |
| O66 `bounds` (O4) | 세 방법에 `bounds`를 주면 `ValueError`다(scipy는 경고하고 무시한다). bfgs와 같은 규칙이다 (사용자 결정 2026-11-20) |
| O67 trust-ncg의 CG 상한 (O4) | 부분문제의 CG는 20n번까지이고 넘으면 status 3이다. scipy에는 상한이 없어 비대칭 헤시안에서 끝나지 않을 수 있다 (사용자 결정 2026-11-20) |
| O68 `nhev`가 세는 것 (O4) | 사용자 `hess`·`hessp`(와 차분 연산자의 곱)를 실제로 부른 수만 센다. `hessp`만 준 trust-ncg에서 scipy가 자리 채움 `hess` 몫으로 더하는 1은 없다 (사용자 결정 2026-11-20) |
| O69 trust-ncg의 결과 `hess` (O4) | 결과를 위해 끝에 x에서 부르는 `hess`도 `nhev`에 센다(scipy는 세지 않는다) (사용자 결정 2026-11-20) |
| O70 newton-cg의 결과 `jac` (O4) | 돌려주는 x에서의 기울기다(scipy는 마지막 반복을 시작한 점의 것). 기억에 있으면 부르지 않고, 없으면 `jac`을 한 번 더 부르고 `njev`에 센다 (사용자 결정 2026-11-20) |
| O71 반복 중의 유한하지 않은 기울기·헤시안 (O4) | 받아들인 점의 기울기나 trust-exact가 새 점에서 만든 헤시안이 유한하지 않으면 그 점을 적은 `ValueError`다(O26과 같은 꼴; scipy는 `scipy.linalg`의 검사로 `ValueError`) (사용자 결정 2026-11-20) |
| O72 trust-exact 부분문제의 상한 (O4) | 걸음을 하나도 만들지 못하고 상한에 닿으면 status 3이다(scipy는 `UnboundLocalError`) (사용자 결정 2026-11-20) |
| O73 `subproblem_max_iter` (O4) | trust-exact의 옵션 이름은 `subproblem_max_iter`(scipy의 `subproblem_maxiter`, O14의 꼴), 기본 25다 (사용자 결정 2026-11-20) |
| O74 newton-cg의 `hess`와 `hessp` (O4) | 함께 주면 `ValueError`다(O62의 규칙: `hessp`가 쓰이지 않는다; scipy는 `hessp`를 버린다) (사용자 결정 2026-11-20) |
| O75 신뢰 영역의 `max_iter=0` (O4) | trust-ncg·trust-exact는 반복 없이 x0에서 멈춘다(newton-cg·BFGS와 같다; scipy는 반복 하나를 한다) (사용자 결정 2026-11-20) |
| O76 trust-exact의 차분 `hess` (O4) | `hess='2-point'`·`'3-point'`는 scipy처럼 `ValueError`다(행렬을 분해해야 한다) (사용자 결정 2026-11-20) |

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
- 배치 스칼라 함수(O5)의 사용자 함수는 아직 끝나지 않은 원소만 모은 float64 `sd.ndarray` (k,)를 받아 k개 값을 돌려준다(O38, scipy `elementwise`와 같다). 시작점에서의 계산은 모든 원소를 받는다.

### 2.2 결과 객체 `OptimizeResult`
- 필드: `x`, `fun`, `jac`(있을 때), `hess`(trust-ncg·trust-exact, `hess`를 줄 때), `hess_inv`(BFGS·L-BFGS-B), `converged`, `status`, `message`, `n_iter`, `nfev`, `njev`, `nhev`(있을 때). 최소제곱은 `cost`, `fun`(잔차), `jac`, `grad`, `optimality`, `active_mask`(scipy `least_squares`와 같은 뜻).
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

### 3.2 스칼라 근 (O2에서 구현)
- 커널은 `csrc/optimize/scalar.cpp`(`stardust._core.optimize.root_solve`)다. 사용자 함수는 Python float를 받는 콜백이고, 반복은 그 사이의 계산을 C++에서 한다(GIL을 쥔 채).
- brentq: scipy의 C 구현(`Zeros/brentq.c`, Brent 1973)을 그대로 옮겼다. scipy는 clang의 기본(`-ffp-contract=on`)으로 빌드되어 a·b + c를 한 번의 반올림(FMA)으로 계산한다. 이 라이브러리는 축약을 끄므로(`-ffp-contract=off`) 그 자리를 `std::fma`로 적었다: 외삽의 분자 f_blk·d_blk − f_pre·d_pre는 fma(f_blk, d_blk, −f_pre·d_pre)라야 204문제 모두 f를 부르는 점이 scipy와 비트까지 같다(축약 없이는 192문제, 다른 쪽을 축약하면 185문제). 멈춤 판정의 xtol + rtol·|x|와 3|s_bis| − δ는 이 문제들에서 결과를 바꾸지 않지만 clang이 같은 꼴을 축약하므로 같이 적었다. bisect(`Zeros/bisect.c`)의 판정도 같다.
- toms748: scipy의 `TOMS748Solver`(Alefeld–Potra–Shi 1995, k = 1이면 알고리즘 4.1, 2면 4.2)를 계산 순서까지 옮겼다: 나눗셈 차분, 역 3차 보간의 Neville 알고리즘(numpy 합의 순서), 이차 뉴턴 걸음, 두 끝의 f 지수가 2⁵⁰ 넘게 다를 때의 31:1 점, 넓이가 반으로 줄지 않을 때의 이분. scipy의 `k ≥ 2`에서 둘째 걸음이 보간을 받지 못하면 앞의 점을 다시 계산하는 동작도 그대로다.
- newton, halley, secant: scipy `newton()`의 스칼라 판. 멈춤은 numpy `isclose`(|p − p0| ≤ xtol + rtol·|p0|)다. `fprime`이 없으면 scipy `root_scalar`처럼 `approx_derivative`의 2점 차분(h = 2⁻²⁶·sign(x)·max(1, |x|))이고, scipy가 차분 안에서 다시 부르는 f(x)는 반복의 값을 쓴다(같은 점, 같은 값).
- 허용오차는 scipy의 기본값이다(O33). 근을 끼지 않는 구간은 `ValueError`(O32)다.
- NaN: scipy는 구간법의 f를 `_wrap_nan_raise`로 감싸 NaN에서 멈추고 `root_scalar`가 그 점과 글을 결과로 돌려준다. 같게 했다(status −3). toms748은 ±inf에서도 그렇게 멈춘다(O45; scipy는 `ValueError`).
- 배치(`find_root`, `bracket_root`): §3.3의 배치와 함께 적는다.

### 3.3 스칼라 최솟값 (O2에서 구현)
- 커널은 `csrc/optimize/scalar.cpp`(`min_solve`)다. brent는 scipy의 `Brent` 클래스와 `bracket()`(두 점이나 없는 구간에서 세 점 찾기, 황금비 1.618034, 넓힘 상한 110, 1,000번), golden은 `_minimize_scalar_golden`(황금비 0.61803399), bounded는 `_minimize_scalar_bounded`(fminbound: √(2.2×10⁻¹⁶)·|x| + xatol/3의 허용오차)를 같은 순서로 옮겼다. scipy의 이 셋은 Python이라 FMA가 없다.
- 구간을 찾지 못하면 scipy의 `_recover_from_bracket_error`처럼 세 점 가운데 가장 낮은 점과 `converged=False`다. scipy의 brent·golden 결과에는 `status`가 없어 bounded의 번호(0 수렴, 1 상한, 2 NaN)에 3(구간을 못 찾음)과 4(구간 찾기의 상한: scipy는 `RuntimeError`)를 더했다.
- 최솟값의 위치는 근처에서 f가 평평해 √u 상대 정확도가 한계다. 문서에 그 한계를 적었다.

### 3.3.1 배치 스칼라 (O2에서 구현)
- 커널은 `csrc/optimize/batch.cpp`(`find_root_solve`, `find_minimum_solve`, `bracket_root_solve`, `bracket_minimum_solve`)다. scipy `elementwise`의 넷(`_chandrupatla`, `_chandrupatla_minimize`, `_bracket_root`, `_bracket_minimum`)과 그 고리(`eim._loop`)를 원소마다 옮겼다: 반복마다 남은 원소의 점을 새 float64 `sd.ndarray`에 모아 사용자 함수를 한 번 부르고(O38), 원소마다 갱신·멈춤 판정·다음 걸음을 계산한다. 원소의 `n_iter`와 `nfev`는 그 원소가 멈출 때까지의 반복 수와 함수값 수다(scipy의 고리와 같다).
- scipy는 단계마다(새 점, 갱신, 판정 하나하나, 다음 걸음) 남은 원소 전체에 numpy 연산을 하고 배열을 줄인다. 모두 원소별 연산이라 여기서는 원소 하나를 한 번 지나가며 갱신, 판정, 줄이기, 다음 걸음을 하고, 한 번 더 지나가며 다음 호출의 점을 쓴다. 원소 하나의 상태(find_root는 double 10개)는 한곳에 둔다: 반복마다 원소를 차례로 읽는다. 상태를 필드마다 다른 배열에 두면 10⁶ 원소에서 열 개 넘는 흐름을 메모리에서 읽게 되어 같은 계산이 4–5배 느리다.
- `bracket_root`는 scipy처럼 원소 하나를 두 쪽(왼쪽 n개, 오른쪽 n개)으로 나눠 따로 넓히고, 한쪽이 부호 바뀜을 찾으면 다른 쪽도 멈춘다(내부 status 2). 결과는 scipy의 `customize_result`처럼 수렴한 쪽(둘 다면 좁은 쪽)의 구간이다.
- 커널은 1스레드다. 호출 사이의 계산은 함수값 하나에 6–12 ns로 사용자 함수의 계산보다 작다(`bench/optimize/o2/speed.md`).
- 원소마다 다른 상수는 `args`(O39)로 받는다. 값마다 Python에서 배치 모양으로 브로드캐스트한 1차원 배열(자기 dtype)이 되고, 커널이 호출마다 남은 원소(`bracket_root`는 쪽 i의 원소 i mod n)를 새 배열에 모아 `f(x, *args)`로 넘긴다.

### 3.4 최소제곱
- `least_squares` (O1에서 구현): nls의 커널을 `csrc/optimize/lsq.cpp`로 옮겼다(O6; `stardust._core.optimize.lsq_solve`). 'lm'은 MINPACK lmder(factor 100, mode 1), 'trf'는 scipy 1.18.1 `trf`('exact' 부분문제, x_scale = 1, 손실 함수 없음)를 같은 순서로 옮긴 것이다(regression §3.7, G69). 경계가 없으면 'trf'는 v = 1, dv = 0이 되어 scipy의 `trf_no_bounds`와 같은 걸음이다.
- 야코비안마다 m 크기의 일은 LAPACK의 Householder QR 하나(J = Q₀R₀, Q₀ᵀf)이고 나머지는 n 크기다(lmder의 열 피벗 QR은 R₀의 것, trf의 SVD는 [R₀d; D]의 것). m < n이면('trf'만) J에 0인 행을 n까지 붙인다(같은 문제다).
- 커널이 nls와 다른 것은 꼴뿐이다(nls를 부를 때는 지금까지와 같은 계산): 사용자 함수에 새 float64 `sd.ndarray`를 넘기기(nls는 Python float의 리스트), 돌려받은 값이 float64 `sd.ndarray`이고 모양이 맞으면 그대로 읽고 아니면 Python의 검사(`_conform`)를 거치기, x0에서의 값을 받기(모양을 알려고 Python에서 먼저 부른 것; nfev에 센다), scipy의 2점·3점 차분, 마지막 반복점의 잔차와 야코비안을 돌려주기('lm'은 마지막 야코비안이 다른 점의 것이면 x에서 한 번 더 계산한다: scipy의 `jac(x)`).
- 결과(`grad` = Jᵀf, `optimality`, `active_mask`)는 scipy의 정의다(`optimality`는 'trf'가 Coleman–Li 척도의 ‖v·g‖∞, 'lm'은 ‖g‖∞; `active_mask`는 `find_active_constraints`(rtol = xtol)).
- nls에만 있는 것(다듬기 G73, 정확한 야코비안의 공분산, 겹친 모수 G76)은 `nls` 쪽에 남겼다. scipy의 `loss`, `x_scale`, `f_scale`, `tr_solver='lsmr'`, `jac_sparsity`, `callback`, `workers`, 다듬기 옵션은 넣지 않았다(O22).
- `lsq_linear`: trf(scipy의 경계 있는 선형 최소제곱)와 bvls(Stark–Parker). `nnls`: Lawson–Hanson(scipy 1.18.1의 구현과 같은 걸음). 바닥은 linalg의 QR이다.

### 3.5 다변수 최소화 (O3에서 Nelder–Mead, BFGS, L-BFGS-B, O4에서 Newton-CG, trust-ncg, trust-exact를 구현)
- 커널은 `csrc/optimize/nelder_mead.cpp`, `bfgs.cpp`, `lbfgsb.cpp`, `newton_cg.cpp`, `trust.cpp`(`stardust._core.optimize.nelder_mead_solve`, `bfgs_solve`, `lbfgsb_solve`, `newton_cg_solve`, `trust_solve`)이고 Python 층은 `src/stardust/optimize/_minimize.py`다. 사용자 함수는 새 float64 `sd.ndarray` x를 받고, 커널은 콜백을 부르는 동안 GIL을 쥔다. 큰 선형대수(BFGS의 행렬곱, L-BFGS-B의 방향 계산과 메모리 갱신, Nelder–Mead의 무게중심)만 10 µs 규칙으로 GIL을 놓는다.
- 기울기는 scipy 1.18.1의 `ScalarFunction`과 같다(`csrc/optimize/objective.hpp`): 마지막 점과 그 점의 f·g를 기억하고, 다른 점(`np.array_equal`: NaN은 같지 않고 −0.0과 0.0은 같다)이거나 아직 계산하지 않았을 때만 계산한다. 사용자 `jac`, `jac=True`(fun이 (f, g)를 돌려준다; scipy의 `MemoizeJac`처럼 x로 기억한다), 차분(`jac=None`은 `approx_derivative`의 '2-point'에 절대 간격 `eps`, `'2-point'`·`'3-point'`는 상대 간격 `finite_diff_rel_step` 또는 기본; 경계에 맞춘 조정은 §3.1의 diff.hpp)이다. `nfev`는 차분의 호출을 센다: scipy의 `ScalarFunction.nfev`도 세므로 O18의 셈이 scipy의 셈과 같다.
- Nelder–Mead: scipy의 `_minimize_neldermead`. 처음 단체(x0와 x0[k]를 1.05배, 0이면 0.00025), `initial_simplex`, `adaptive`(Gao–Han의 계수), 경계(위 경계 너머의 꼭짓점을 반사하고 자르기, 새 점마다 자르기), `xatol`·`fatol`, `max_nfev`를 넘는 호출은 하지 않고 그 반복을 멈추기, 정렬은 numpy `argsort`의 introsort(17개 미만은 삽입 정렬이라 같은 값의 순서가 그대로이고, 그 위는 아니다)를 그대로 옮겼다. 무게중심은 numpy처럼 +0.0에서 행을 차례로 더한다. 정렬은 행을 옮기지 않고 행 번호를 옮긴다.
- BFGS: scipy의 `_minimize_bfgs`와 `_line_search_wolfe12`(`csrc/optimize/wolfe.hpp`, Newton-CG와 같이 쓴다): `line_search_wolfe1`(scipy의 Python `DCSRCH`·`dcstep`, c1·c2, xtol 10⁻¹⁴, 걸음 [10⁻¹⁰⁰, 10¹⁰⁰], 100번)이 걸음을 못 찾으면 `line_search_wolfe2`(두 배로 넓히기와 3차·2차 보간의 zoom, 10번씩), 둘 다 못 찾으면 status 2. H는 scipy처럼 (I − ρsyᵀ)H(I − ρysᵀ) + ρssᵀ를 두 번의 행렬곱(BLAS dgemm)으로 만든다. scipy의 Python 코드는 FMA가 없으므로 이 파일은 축약 없이 빌드한다. zoom의 보간은 scipy가 `np.errstate(divide, over, invalid='raise')` 안에서 하고 그 에러에서 보간을 버린다: 여기서는 연산마다 numpy의 규칙(유한한 0 아닌 수를 0으로 나누기, 유한한 값에서의 넘침, NaN이 아닌 값에서 NaN)으로 같은 판정을 한다(2 × 2 곱은 numpy처럼 dgemv). 사용자 함수가 Python float을 돌려줄 때 scipy는 Python의 곱셈 넘침을 에러로 보지 않으므로 그런 경계의 경우만 다를 수 있다.
- L-BFGS-B: scipy 1.15부터의 C 판(`scipy/optimize/src/lbfgsb.c`, L-BFGS-B 3.0의 setulb·mainlb와 하위 루틴: cauchy, hpsolb, bmv, freev, formk, cmprlb, subsm, lnsrlb, dcsrch, dcstep, matupd, formt, projgr)과 scipy의 드라이버 고리(`_minimize_lbfgsb`: NEW_X마다 반복을 세고 `max_iter`·`max_nfev`(> 비교)로 멈춤)를 계산 순서까지 옮겼다. scipy가 원본과 다른 곳(2-노름을 dnrm2로, 0부터 세는 번호에서 결코 참이 되지 않는 Cauchy 점의 마지막 구간 판정 `nleft == -1 && nbreak == n`)도 그대로다.
  - scipy는 이 C 파일을 clang의 기본(`-ffp-contract=on`)으로 빌드하여 한 식 안의 a·b + c를 FMA 하나로 계산한다. 그래서 이 파일은 `#pragma clang fp contract(on)`으로 같은 축약을 하고 식을 scipy의 꼴 그대로 적었다(축약을 끄면 경계 없는 114경우 가운데 13경우만 호출점이 비트까지 같다).
  - Accelerate의 ddot은 둘째 인자가 16바이트 정렬인지에 따라 합의 순서가 다르다(길이 5 이상에서 끝자리가 달라진다). scipy의 작업 배열 `wa`는 numpy 배열 하나를 setulb가 나눠 쓰므로 부분마다 정렬이 다르다: 여기서도 작업 공간 하나를 scipy와 같은 위치로 나눈다(그래야 경계 있는 문제의 호출점이 비트까지 같다).
  - scipy의 setulb는 f를 값으로 받으므로 ABNORMAL(선 탐색 실패)에서 x와 g는 앞의 반복점으로 되돌려도 결과의 `fun`은 마지막 시험점의 값이다. 여기서는 되돌린 `x`의 값(앞의 반복점의 f)을 준다(O59; 걸음은 같다).
  - `hess_inv`(O55)는 scipy의 `LbfgsInvHessProduct(s[:k], y[:k]).todense()`와 같은 행렬이다: 저장된 쌍을 저장 순서로(메모리가 한 바퀴 돈 뒤에는 시간 순서가 아니다; scipy와 같다, O60), H₀ = I로 두 고리 재귀를 단위 행렬에 적용한다. 처음 읽을 때 계산하고 결과에 둔다(n²의 메모리: n = 10⁴이면 800 MB).
  - 모든 변수가 고정된 경계(lower = upper)면 scipy처럼 fun을 한 번 부르고 끝낸다(`'All independent variables were fixed by bounds.'`). 일부가 고정되고 기울기가 차분이면 scipy처럼 그 변수를 빼고 푼다(결과의 `jac`은 그 자리가 NaN, `hess_inv` 필드는 없다; O58).
- 헤시안(`csrc/optimize/hessian.hpp`): scipy의 `ScalarFunction`처럼 마지막 점의 헤시안을 기억한다. 호출 가능한 `hess`는 처음에 x0에서(f와 기울기 다음) 부르고, 다른 점에서만 다시 부른다(fun, 기울기, 헤시안 가운데 어느 요청으로든 점이 바뀌면 기억을 버린다). `'2-point'`·`'3-point'`는 scipy `approx_derivative`의 선형 연산자다: 만들 때는 부르지 않고 그 점의 기울기를 쓰며, p ≠ 0과의 곱은 (g(x + dx·p) − g)/dx(dx = √u/‖p‖) 또는 (g(x + dx/2·p) − g(x − dx/2·p))/dx(dx = 2u^(1/3)/‖p‖)이고, 그 기울기는 기억을 거치지 않고 부른다(`njev`에 센다). p = 0이면 부르지 않고 0이다. 사용자가 돌려준 행렬은 C 순서로 복사해 두고, 그 행렬을 쥔 부분문제와 같이 쓴다.
- Newton-CG: scipy의 `_minimize_newtoncg`. 반복마다 Hp = −g를 0에서 공액 기울기로 푼다(20n번까지; Σ|r_i| ≤ min(1/2, √‖g‖₁)·‖g‖₁이거나 곡률 pᵀHp가 [0, 3u]에 들거나 음수이면 멈추고, 첫 걸음의 음의 곡률은 가장 가파른 내리막 (rᵀr/(−pᵀHp))·(−g)다). 그 방향으로 BFGS와 같은 `_line_search_wolfe12`를 scipy의 Newton-CG 인자로 한다: wolfe1의 걸음은 [10⁻⁸, 50], wolfe2는 amax가 없고, 첫 반복은 앞의 값이 없어 첫 걸음이 1이다. 걸음의 ‖·‖₁이 n·xtol 이하이면 멈추고, 선 탐색이 걸음을 못 찾으면 status 2, CG가 20n번 안에 멈추지 않으면 status 3이다. 헤세 곱은 `hess`의 행렬(반복마다 한 번 계산), `hessp`, 차분 연산자, 둘 다 없으면 scipy의 `approx_fhess_p`(기울기의 앞쪽 차분, 간격 `eps`; `ScalarFunction`의 기억을 거치므로 두 번째 곱부터 g(x)를 다시 부른다)다.
- trust-ncg, trust-exact: scipy의 `_minimize_trust_region`(ρ < 1/4이면 반지름을 1/4로, ρ > 3/4이고 걸음이 경계에 있으면 두 배로(`max_trust_radius`까지), ρ > η이면 걸음을 받아들이고, 예측 감소가 양수가 아니면 status 2)과 부분문제 둘이다. `max_iter=0`이면 반복 없이 x0에서 멈춘다(O75; scipy는 상한을 첫 반복 뒤에 보아 반복 하나를 한다). trust-ncg는 `CGSteihaugSubproblem`(Nocedal–Wright 알고리즘 7.2: 허용오차 min(1/2, √‖g‖)·‖g‖, 양이 아닌 곡률에서는 모형 값이 작은 경계점, 영역을 벗어나는 걸음은 경계점; CG는 20n번까지이고 넘으면 status 3, O67: scipy에는 상한이 없어 비대칭 헤시안에서 끝나지 않을 수 있고, 대칭이면 n번 남짓이다), trust-exact는 `IterativeSubproblem`(Moré–Sorensen: H + λI의 Cholesky, λ의 뉴턴 걸음, 어려운 경우의 Cline–Moler–Stewart–Wilkinson 최소 특이값 추정, Conn–Gould–Toint의 경계(Gershgorin, Frobenius·∞-노름), k_easy 0.1, k_hard 0.2, `subproblem_max_iter`번까지)이다. trust-exact는 새 점의 부분문제를 만들 때 그 점의 헤시안을 f보다 먼저 계산한다(scipy와 같은 호출 순서).
- 계산은 numpy·scipy.linalg가 부르는 루틴을 같은 배치로 부른다. `np.dot`(벡터 둘)은 원소 하나면 곱이고 아니면 0.0 + cblas_ddot이다: BLAS가 돌려주는 −0.0이 +0.0이 된다. trust-ncg의 경계 교점은 b = 2zᵀd의 부호를 `copysign`으로 보므로 z = 0에서 이 차이가 걸음을 바꾼다. 행렬·벡터 곱은 row-major dgemv, `scipy.linalg.norm`은 dnrm2, H의 ∞-노름은 lange('1', Hᵀ), Frobenius 노름은 √(ddot)이다. Cholesky는 scipy가 potrf('U')에 넘기는 열 우선 복사에 한다(potrf가 위 삼각만 읽으므로 그것만 복사하고, 아래 삼각은 scipy의 래퍼처럼 0으로 둔다). 풀이는 potrs·trtrs이고, scipy가 연속이 아닌 조각(실패한 분해의 앞 블록)을 `solve_triangular`에 넘길 때 전치를 복사해 아래 삼각·전치로 푸는 것도 따른다. 합은 numpy의 쌍별 합, 최대·최소는 Python의 `max`·`min`이다.
- scipy가 예외를 내는 곳은 `scipy.linalg`의 `check_finite`(받아들인 점의 ‖g‖, trust-exact가 새 점에서 만든 헤시안, 부분문제 안의 벡터)와 `math.sqrt`의 음수다. 여기서는 그 점에서 `ValueError`(`minimize(): 'jac' is not finite at x = [...]`, `'hess' is not finite at x = [...]`, `the trust-region subproblem is not finite at x = [...]`)다(O53, O71). trust-exact의 부분문제가 걸음을 하나도 만들지 못한 채 `subproblem_max_iter`에 닿으면 scipy는 `UnboundLocalError`이고 여기서는 status 3이다(O72).
- 셈: `nfev`·`njev`는 `ScalarFunction`의 것(scipy와 같다)이다. `nhev`는 사용자 `hess`·`hessp`와 차분 연산자의 곱을 실제로 부른 수다(O68): 호출 가능한 `hess`는 x0의 것을 포함한 호출 수, `hessp`와 차분은 곱의 수이고, trust-ncg가 결과의 `hess`를 위해 끝에 x에서 부른 것(마지막 부분문제가 계산하지 않았을 때, scipy의 `m.hess`)도 센다(O69). scipy와 다른 곳: `hessp`만 준 trust-ncg에서 scipy는 부르지 않은 자리 채움 `hess`의 1을 더하고, trust-ncg의 끝 호출과 newton-cg가 x0에서 처음 부른 `hess`(`max_iter=0`일 때)를 세지 않는다.
- newton-cg의 결과 `jac`은 돌려주는 x에서의 기울기다(O70). `ScalarFunction`의 기억에 있으면 부르지 않고(마지막 선 탐색이 x에서 기울기를 계산했을 때), 없으면(기울기 차분의 곱이나 wolfe2가 기억을 옮겼을 때) 한 번 더 부르고 `njev`에 센다. scipy의 것은 마지막 반복을 시작한 점의 기울기(`gfk`)다.
- newton-cg에 `hess`와 `hessp`를 함께 주면 `ValueError`다(O74; scipy는 `hessp`를 버린다). trust-ncg는 함께 받고 scipy처럼 곱에 `hessp`, x0와 결과에 `hess`를 쓴다. 차분 `hess`와 `hessp`는 함께 줄 수 없다.
- 그래서 여섯 방법 모두 scipy 1.18.1과 호출점이 비트까지 같다(§7). O9는 끝자리 차이를 허용하지만 이 방법들은 차이가 없다. 다른 것은 위에 적은 결정의 자리뿐이다: newton-cg 끝의 기울기 호출 하나(O70), `nhev`(O68, O69), 신뢰 영역의 `max_iter=0`(O75), scipy가 끝나지 않거나 예외를 내는 곳(O67, O71, O72).

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
- `src/stardust/optimize/`: 공개 함수(`_scalar.py`, `_elementwise.py`, `_minimize.py`, `_lsq.py`, `_root.py`, `_diff.py`), 결과 객체(`_result.py`), 입력 정리와 메시지(`_common.py`)
- `csrc/optimize/`: `scalar.hpp`·`scalar.cpp`(스칼라 근·최솟값), `batch.cpp`(배치 스칼라), `diff.hpp`, `lsq.cpp`(nls에서 옮긴 lmder·trf), `minimize.hpp`, `objective.hpp`(scipy `ScalarFunction`의 기억과 차분), `wolfe.hpp`(scipy Python 판의 `DCSRCH`와 wolfe2: BFGS와 Newton-CG의 선 탐색), `hessian.hpp`(헤시안의 기억과 차분 연산자), `nelder_mead.cpp`, `bfgs.cpp`, `lbfgsb.cpp`(C 판의 `dcsrch`), `newton_cg.cpp`, `trust.cpp`(trust-ncg·trust-exact), `hybrd.cpp`, `nnls.cpp`; 바인딩 `csrc/bind/optimize_bind.cpp`
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

O2에서 확인한 것(2026-11-20):

- **scipy 1.18.1과 같은 걸음** (O9; `tests/optimize/test_scalar.py`, `test_elementwise.py`). 기록된 실행(`data/scipy_root_scalar.json` 1,244번, `scipy_minimize_scalar.json` 171번, `scipy_elementwise.json` 621번; 두 점 구간의 `find_minimum` 하나는 scipy도 에러다)과 근, 반복 수, 호출 수, 멈춘 이유가 모두 같다. scipy를 실행하며 사용자 함수를 부른 점을 차례대로 견주면 모두 비트까지 같다: `root_scalar` 6방법 1,244번(newton의 차분은 scipy가 f(x)를 한 번 더 부르는 것을 빼고), `minimize_scalar` 3방법과 두 점·없는 구간의 brent·golden 399번, 배치 넷의 원소 하나 실행 621번, 옵션을 바꾼 `root_scalar` 7가지(204문제씩), 2,000원소 배치 7가지(NaN, 무한한 끝, 근을 끼지 않는 구간, 한계, `max_iter` 섞임; 필드 전부와 호출마다의 점).
  - brentq·bisect는 scipy의 C 코드가 FMA로 빌드된 것을 따라 `std::fma`를 적은 뒤에야 비트까지 같다(§3.2). FMA 없이 계산하면 204문제 가운데 12문제의 호출점이 끝자리에서 갈리고 7문제의 근이 다르다(가장 큰 차이는 근이 10⁻¹⁶ 근처인 aps.03.01·02의 상대 10⁻¹⁰).
  - 기록과 다른 것은 `message` 하나다: scipy가 `'convergence error'`로 적고 경고로 이유를 내는 newton(도함수 0)과 secant(같은 두 값) 62번은 그 경고의 글을 `message`로 둔다(O43).
  - `args`(O39): 원소마다 다른 c의 x³ − c와 (x − c/10)² + kx⁴(k는 (5, 1)로 브로드캐스트)에서 배치 넷이 scipy의 `args`와 필드 전부, 호출마다의 x와 args가 같다.
- **Arb 참값** (`data/scalar_roots.json`): 기본 허용오차로 수렴한 구간법의 근(brentq 185, toms748 204, bisect 204)은 모두 Arb 근에서 xtol + rtol·|x| 안이거나 f가 double로 정확히 0인 점이다. `xtol=0`이면: brentq(`max_iter=200`) 197문제 수렴, 근을 끼는 두 double에서 3 ULP 안(171문제가 1 ULP 안), toms748(`rtol` = u) 199문제, 1 ULP 안, bisect(`max_iter=1100`) 204문제, 4 ULP 안(나머지는 모두 f가 double로 0인 점). `find_root`(기본)는 204문제 가운데 182문제가 4 ULP 안이고, 13문제는 f가 0인 점, 9문제는 x⁹·x¹⁹꼴의 근 0에서 |f| ≤ fatol(tiny)로 멈춘 10⁻¹⁷–10⁻³⁵의 점이다(scipy와 같은 점).
- **배치와 원소 하나**: 2,000원소의 `find_root`·`bracket_root` 배치는 원소마다 그 원소 하나로 부른 결과와 필드가 모두 같다. 근은 `root_scalar`(toms748)와 10⁻⁹ 안이다.
- 속도는 `bench/optimize/o2/speed.md`: 스칼라 방법 자체의 비용이 호출 하나에 0.2–1.5 µs(scipy의 C 구현의 1/1.5–1/2.8, Python 구현의 1/2–1/38), 배치(O0의 문제, c는 `args`)는 함수값 하나에 6–12 ns(scipy 7–105 ns): 10⁶ 원소 1스레드에서 find_root 143 ms 대 330 ms, find_minimum 250 ms 대 649 ms, bracket_minimum 64 ms 대 66 ms(10스레드는 사용자 함수의 sd 연산이 나뉘어 99, 129, 39 ms).

O3에서 확인한 것(2026-11-20):

- **scipy 1.18.1과 비트까지 같은 걸음** (O9; `tests/optimize/test_minimize.py`). 기록된 실행(`data/scipy_minimize.json`: MGH 110경우, Rosenbrock n = 2·5, 최대 엔트로피, driver1)의 Nelder–Mead 114, BFGS 113, BFGS 차분 113, L-BFGS-B 114, L-BFGS-B 차분 114번에서 결과의 모든 필드(x의 비트, fun, nfev, njev, nit, status, message, success)가 같다. scipy를 실행하며 `fun`·`jac`을 부른 점을 차례대로 견주면 여덟 묶음(위의 다섯과 x0 둘레의 상자 경계에서 Nelder–Mead 51, L-BFGS-B 해석적·차분 51씩)이 모두 비트까지 같다. 옵션과 가장자리(`jac=True`·`'2-point'`·`'3-point'`, `eps`, `finite_diff_rel_step`, `norm`, `xrtol`, `c1`·`c2`, `hess_inv0`, `maxcor`, `max_ls` 1·2(재시작과 ABNORMAL), `max_iter`, `max_nfev`, `tol`, 고정 변수, 반쪽 경계, `adaptive`, `initial_simplex`, 시험점에서 inf·NaN)를 MGH 7문제와 Rosenbrock에서 41가지씩(328번) 견주어도 모두 같다(테스트는 4문제, 164번).
  - L-BFGS-B를 비트까지 맞추는 데 둘이 필요했다: scipy의 C 판이 clang 기본의 FMA 축약으로 빌드된 것(같은 축약 없이는 경계 없는 114경우 가운데 13경우만 같다)과, Accelerate의 ddot이 둘째 인자의 16바이트 정렬에 따라 합의 순서를 바꾸는 것(scipy의 작업 배열과 같은 위치로 나누기 전에는 상자 경계 51경우 가운데 43경우만 같았다). scipy의 lbfgsb.c를 같은 clang으로 빌드한 것이 scipy의 바이너리와 51경우 모두 같다는 것도 확인했다.
  - BFGS의 `hess_inv`는 scipy와 비트까지 같고(MGH 110경우), L-BFGS-B의 것은 scipy의 `LbfgsInvHessProduct.todense()`와 상대 2.8×10⁻¹⁴ 안이다(`maxcor` 10·3, 220번, 메모리가 한 바퀴 돈 185번 포함).
- **L-BFGS-B 3.0의 Fortran 원본** (`data/fortran_lbfgsb.json`, 83번): 모든 실행이 scipy의 기록(`fortran_vs_scipy.json`)과 x의 비트, nfev, nit, message까지 같다. 그래서 원본과의 거리도 scipy와 같다: 호출 수와 반복 수가 같은 것 74, x가 10⁻¹² 안 62, 10⁻⁶ 안 79, 같은 멈춤 판정 82번. driver1(m = 5)은 원본 드라이버처럼 23반복, 28계산이다.
- O18의 셈과 scipy의 셈이 같다: scipy의 `minimize`는 `ScalarFunction.nfev`로 차분의 호출을 센다(`least_squares`와 다르다).
- scipy에서 찾은 것: L-BFGS-B의 setulb는 f를 값으로 받아 ABNORMAL에서 결과의 `fun`이 되돌린 `x`의 값이 아니다(여기서는 `x`의 값을 준다, O59: 기록과 다른 것은 이 `fun`뿐이다). `hess_inv`는 저장 순서의 쌍을 쓴다. Cauchy 점의 마지막 구간 판정(`nbreak == n`)은 0부터 세는 번호에서 참이 될 수 없다. BFGS의 zoom 보간은 사용자 함수가 Python float과 numpy float 가운데 무엇을 돌려주는지에 따라 넘침의 처리가 다르다.
- 속도는 `bench/optimize/o3/speed.md`: 작은 n의 반복당 자기 비용이 scipy의 1/2–1/20(BFGS n = 2: 1.0 µs 대 22 µs), 차분의 기울기는 1/3–1/12(L-BFGS-B n = 1000: 0.35 ms 대 4.3 ms), Nelder–Mead n = 1000은 0.14 ms 대 0.76 ms, BFGS n = 1000은 같은 dgemm이라 9.4 ms 대 11.5 ms, L-BFGS-B n = 10⁴는 같은 C 알고리즘이라 0.35 ms 대 0.37 ms다.

O4에서 확인한 것(2026-11-20):

- **scipy 1.18.1과 비트까지 같은 걸음** (O9; `tests/optimize/test_minimize_newton.py`). 기록된 실행(`data/scipy_minimize.json`: MGH 110경우, Rosenbrock n = 2·5, 최대 엔트로피; 헤시안은 `ad.Dual2`의 정확한 값)의 Newton-CG, trust-ncg, trust-exact 113번씩, 339번 모두 결과의 필드(x의 비트, fun, nfev, nit, status, message, success)가 같다. `njev`는 trust의 두 방법이 같고 newton-cg가 같거나 하나 많다(O70의 끝 호출), `nhev`는 실제 호출 수(O68, O69)다. scipy를 실행하며 `fun`·`jac`·`hess`·`hessp`를 부른 점(곱의 p까지)을 차례대로 견주면 열 묶음(Newton-CG의 `hess`·`hessp`·`'2-point'`·`'3-point'`·없음, trust-ncg의 `hess`·`hessp`·`'2-point'`·`'3-point'`, trust-exact) 1,130번이 모두 비트까지 같다(newton-cg는 끝에 x에서 기울기를 한 번 더 부를 수 있다, O70). 결과의 `x`, `fun`, `hess`, `nfev`, `status`, `message`도 같고, `jac`은 trust가 scipy와, newton-cg가 x의 기울기와 같으며, `njev`·`nhev`는 결정대로 센 수다.
  - 옵션과 가장자리(`jac=True`, trust-ncg의 `hess`와 `hessp`를 함께, `max_iter` 1·3(newton-cg는 0도), `tol`, `xtol`, `eps`, `c1`·`c2`, `gtol`, 반지름, `eta`, `subproblem_max_iter`, 시험점의 inf·NaN(함수값·기울기·헤시안))를 MGH 7문제에서 525번 견주어도 모두 같다. scipy가 `ValueError`를 내는 42번은 같은 점까지 부르고 `ValueError`다. `subproblem_max_iter=1`에서 scipy가 `UnboundLocalError`를 내는 3번은 status 3이다.
  - 무작위 비볼록 사차 함수 Σa_i x_i⁴ + xᵀBx/2 + cᵀx(n = 2–12, 넷에 하나는 비대칭 헤시안) 300문제 × 7가지(2,100번)와 Rosenbrock n = 64–200 28번도 모두 같다. trust-exact의 실패한 분해(앞 블록 k = 2–12)와 어려운 경우를 지나고, ‖g‖ ≤ n·u·‖H‖∞의 가지는 `gtol=0`의 무작위 60문제 120번에서 지나며 모두 같다.
  - 비트까지 맞추는 데 둘이 필요했다. numpy의 `np.dot`(벡터 둘)은 0.0 + cblas_ddot이라 BLAS의 −0.0이 +0.0이 된다: cblas_ddot을 그대로 쓰면 z = 0에서 b = 2zᵀd가 −0.0이 되고 `copysign`이 다른 근의 식을 골라 MGH 8경우의 trust-ncg 걸음이 끝자리에서 갈렸다. trust-exact는 실패한 분해의 앞 블록을 scipy처럼 전치해 아래 삼각으로 푼다.
- **scipy에서 찾은 것**: trust-ncg의 Steihaug CG에는 상한이 없어 비대칭 헤시안에서 끝나지 않는 경우가 있다(무작위 문제 하나가 26번째 부분문제에서 돈다). 대칭 헤시안에서는 CG 걸음이 n번 남짓이므로(MGH에서 많아야 1.2n, 무작위 대칭 400문제에서 n, Rosenbrock n = 1000에서 8번) 여기서는 20n번(Newton-CG의 상한)에서 status 3으로 멈춘다(O67). 그 밖에: newton-cg의 결과 `jac`은 x가 아니라 마지막 반복을 시작한 점의 기울기다. trust-ncg는 `hessp`만 주면 `nhev`에 부르지 않은 자리 채움 `hess`의 1을 더하고, 결과 `hess`를 셈 뒤에 계산해 그 호출을 세지 않는다. trust의 `max_iter=0`은 반복 하나를 한다. trust-exact의 `subproblem_maxiter`가 작으면 걸음 없이 끝나 `UnboundLocalError`가 난다. 받아들인 점의 기울기나 trust-exact가 새 점에서 만든 헤시안이 유한하지 않으면 `scipy.linalg`의 `check_finite`가 `ValueError`를 낸다.
- 속도는 `bench/optimize/o4/speed.md`: n = 2–100의 반복당 자기 비용이 scipy의 1/3–1/60(trust-exact n = 10: 2.2 µs 대 59 µs), 차분의 헤세 곱은 n = 1000에서도 1/9(Newton-CG '2-point': 0.034 ms 대 0.31 ms), 행렬 `hess`를 쓰는 n = 1000의 Newton-CG·trust-ncg는 같은 dgemv가 대부분이라 비슷하거나 조금 느리고(0.23 대 0.22 ms, 0.17 대 0.13 ms; 사용자의 행렬을 복사해 두는 몫), trust-exact n = 1000은 5.2 ms 대 7.2 ms다.

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 |
|---|---|
| O0 준비 | API 문서 초안, 결과 객체의 틀, 시험 문제 모음(MGH 35, scipy 시험 함수), 대조 도구, 벤치마크 기준선(scipy). 완료(2026-11-20): §7, docs/api/optimize.md, `bench/optimize/o0/` |
| O1 수치 미분과 최소제곱 | `approx_fprime`, `check_grad`, nls 커널을 `least_squares`로 옮기기(nls 비트 동일). 완료(2026-11-20): §3.1, §3.4, §7, `bench/optimize/o1/` |
| O2 스칼라 | `root_scalar`, `minimize_scalar`, 배치 `find_root bracket_root find_minimum bracket_minimum`. 완료(2026-11-20, 승인 대기): §3.2, §3.3, §7, `bench/optimize/o2/` |
| O3 최소화 1 | `minimize`: Nelder–Mead, BFGS, L-BFGS-B. 완료(2026-11-20, 승인 대기): §3.5, §7, `bench/optimize/o3/` |
| O4 최소화 2 | `minimize`: Newton-CG, trust-ncg, trust-exact. 완료(2026-11-20, 승인 대기): §3.5, §7, `bench/optimize/o4/` |
| O5 방정식계와 선형 최소제곱 | `root`(hybr, lm), `lsq_linear`, `nnls` |
| O6 최적화와 일관성 점검 | 커널 속도(BFGS의 H를 n²으로 갱신하기, O61), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.

---

## 9. 마일스톤을 시작하기 전에 물을 것
- **O0**: 정했다(2026-11-20, O9–O13). O0에서 API 초안을 쓰며 생긴 질문 가운데 모든 마일스톤에 걸친 넷(반복 상한의 이름, 방법 이름의 대소문자, `options` dict, `status`·`message`)도 정했다(O14–O17). 마일스톤별로 남은 질문은 docs/api/optimize.md §9에 있다(`nfev`가 차분을 세는지, `least_squares`의 `bounds` 기본값, `root_scalar`의 결과 이름, `nnls`의 결과 등).
- **O1**: 정했다(2026-11-20, O18–O23, 구현하며 생긴 질문은 O24–O31). 옮기는 순서는 설계대로 먼저 옮기고 비트를 확인한 뒤 공개 함수를 만들었다.
- **O2**: 정했다(2026-11-20, O32–O38). 구현하며 생긴 질문도 정했다(O39–O49).
- **O3**: 정했다(2026-11-20, O50–O57, 구현하며 생긴 질문은 O58–O62).
- **O4**: 정했다(2026-11-20, O63–O66, 구현하며 생긴 질문은 O67–O76).
- **O5**: hybr의 기본 허용오차(scipy 1.49×10⁻⁸), `root`의 다른 방법(broyden 등)을 넣을지.
