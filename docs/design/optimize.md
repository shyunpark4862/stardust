# 최적화 영역 설계 (sd.optimize)

> 상태: **설계 초안** (2026-11-20). 큰 결정(§1.1 O1–O8)은 사용자가 정했다. 나머지는 안이고, 시작 전에 정할 것은 §9에 있다.
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

- `curve_fit`은 두지 않는다: 같은 일을 `regression.nls`가 추론까지 해서 한다(§9 O0 질문 3).
- `brentq` 같은 개별 함수 이름(scipy에 있는 `brentq`, `toms748`, `newton`, `fmin`, `fsolve` 등)을 따로 둘지는 §9 O0 질문 2다.

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
- 추가 인자를 넘기는 `args=()`를 둘지는 §9 O0 질문 4다(nls에는 없다).
- 배치 스칼라 함수(O5)의 콜백은 모양이 (...)인 `sd.ndarray`를 받아 같은 모양을 돌려준다. 이미 수렴한 원소는 다음 호출에서 뺀다(scipy `elementwise`와 같다).

### 2.2 결과 객체 `OptimizeResult`
- 필드: `x`, `fun`, `jac`(있을 때), `hess_inv`(BFGS·L-BFGS-B), `converged`, `status`, `message`, `n_iter`, `nfev`, `njev`, `nhev`(있을 때). 최소제곱은 `cost`, `fun`(잔차), `jac`, `grad`, `optimality`, `active_mask`(scipy `least_squares`와 같은 뜻).
- 이름은 regression의 결과와 같은 규칙이다(O4). scipy의 `success`·`nit`를 별칭으로 두지 않는다(별칭 없음).
- 배치 스칼라 결과는 필드마다 입력의 배치 모양이다(`x` (...), `converged` (...) bool, `nfev` (...) int64).
- 결과는 읽기 전용이고 배열 필드는 복사본을 돌려준다(regression G98, G113과 같다).
- `repr`는 짧은 요약이다. `summary()`는 두지 않는다(추론이 없다).

---

## 3. 수치 방법 (안)

### 3.1 수치 미분
- nls의 정확한 야코비안(regression §3.7, G70)을 공용으로 옮긴다: 5점 중앙 차분, 2의 거듭제곱 간격, 모수마다 처음 한 번 간격을 고르기(2⁻⁸|θ_j|에서 4배씩 줄이며 이웃 추정의 차이가 직전의 1/8보다 작은 동안), 경계나 유한하지 않은 쪽에서는 한쪽 5점 차분.
- 반복 안의 야코비안은 scipy처럼 2점 차분(`'2-point'`, h = √u·max(1, |x|))을 기본으로 하고 `'3-point'`·`'5-point'`를 고를 수 있게 할지는 §9 O1 질문이다.
- `approx_fprime(xk, f, *, epsilon=None)`: epsilon이 None이면 간격을 고르는 5점 차분, 수를 주면 scipy와 같은 앞쪽 차분. `check_grad`는 그 차이의 2-노름이다.

### 3.2 스칼라 근
- brentq: scipy의 C 구현(`brentq.c`, Brent 1973)과 같은 걸음과 판정. toms748: Alefeld–Potra–Shi(1995) 알고리즘 748(scipy의 Python 구현과 같은 걸음). bisect, secant, newton, halley: scipy `newton`의 규칙.
- 기본 허용오차는 scipy의 것(xtol 2×10⁻¹², rtol 4u)이다. 구간법(brentq, toms748, bisect)은 근을 끼는 구간이 1 ULP가 될 때까지 갈 수 있으므로, 기본을 "반올림까지"로 할지는 §9 O2 질문이다.
- 배치(`find_root`, `bracket_root`): scipy `elementwise`의 Chandrupatla(1997) 꼴. 원소마다 구간과 상태를 C++에서 들고, 남은 원소만 모아 콜백을 부른다.

### 3.3 스칼라 최솟값
- brent: scipy의 `Brent` 클래스(포물선 보간과 황금분할)와 같은 걸음. golden: 황금분할. bounded: scipy `fminbound`(Brent의 fmin).
- 배치(`find_minimum`, `bracket_minimum`): scipy `elementwise`의 Chandrupatla 꼴의 최솟값 판.
- 최솟값의 위치는 근처에서 f가 평평해 √u 상대 정확도가 한계다. 문서에 그 한계를 적는다.

### 3.4 최소제곱
- `least_squares`: nls의 커널을 옮긴다(O6). 'lm'은 MINPACK lmder, 'trf'는 scipy 1.18.1 `trf`('exact' 부분문제, x_scale = 1, 손실 함수 없음)를 같은 순서로 옮긴 것이다(regression §3.7, G69).
- nls에만 있는 것(다듬기 G73, 정확한 야코비안의 공분산, 겹친 모수 G76)은 `nls` 쪽에 남긴다. `least_squares`가 다듬기를 옵션으로 가질지는 §9 O1 질문이다.
- scipy의 `loss`(soft_l1, huber, cauchy, arctan), `x_scale`, `f_scale`, `tr_solver='lsmr'`, `jac_sparsity`를 넣을지는 §9 O1 질문이다.
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
- 각 방법은 scipy와 같은 순서의 계산을 목표로 한다. 그래서 같은 문제에서 반복점, `nfev`, 멈춘 이유가 scipy와 같다(반올림의 순서가 다른 곳은 끝자리가 다를 수 있다). 비트까지 같게 할지는 §9 O0 질문 1이다.

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
- `regression.nls`는 `csrc/optimize/lsq.cpp`와 `_diff.py`를 쓴다(O6). `csrc/regression/nls.cpp`는 없어진다.
- `tests/optimize/`, `bench/optimize/`

---

## 7. 참값과 검증 (안)
- **시험 문제**: Moré–Garbow–Hillstrom(1981)의 35개(최소제곱·최소화·방정식계; MINPACK의 시험 드라이버와 같은 시작점과 배수), NIST StRD 비선형 27개(이미 `tests/regression/data/nls/`), scipy의 시험 모음(`scipy/optimize/tests`의 Rosenbrock 등). CUTEst는 쓰지 않는다(§9 O0 질문 5).
- **대조**: scipy 1.18.1(이미 설치됨)과 같은 문제에서 반복점·`nfev`·결과를 비교한다. MINPACK·L-BFGS-B는 원래의 Fortran 결과를 한 번 만들어 고정 데이터로 둘지 §9 O0 질문 5다.
- **근의 참값**: 구간법의 근은 Arb로 f의 부호를 확정해 1 ULP 구간을 구한다(`tools/`에 도구).
- nls를 옮긴 뒤 G6의 테스트 전부와 골든 출력(NIST 27개 × 시작점 2개, 두 방법)이 비트까지 같아야 한다.

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 |
|---|---|
| O0 준비 | API 문서 초안, 결과 객체의 틀, 시험 문제 모음(MGH 35, scipy 시험 함수), 대조 도구, 벤치마크 기준선(scipy) |
| O1 수치 미분과 최소제곱 | `approx_fprime`, `check_grad`, nls 커널을 `least_squares`로 옮기기(nls 비트 동일) |
| O2 스칼라 | `root_scalar`, `minimize_scalar`, 배치 `find_root bracket_root find_minimum bracket_minimum` |
| O3 최소화 1 | `minimize`: Nelder–Mead, BFGS, L-BFGS-B |
| O4 최소화 2 | `minimize`: Newton-CG, trust-ncg, trust-exact |
| O5 방정식계와 선형 최소제곱 | `root`(hybr, lm), `lsq_linear`, `nnls` |
| O6 최적화와 일관성 점검 | 커널 속도, 메시지·시그니처의 영역 간 일관성, 코드 리뷰 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.

---

## 9. 마일스톤을 시작하기 전에 물을 것
- **O0**
  1. scipy와 같은 걸음의 정도: 반복점과 `nfev`가 같은 것(끝자리는 다를 수 있음)으로 할지, 비트까지 같게 할지.
  2. 개별 함수 이름(`brentq`, `toms748`, `bisect`, `newton`, `fmin`, `fminbound`, `fsolve`, `leastsq` 등)을 `root_scalar`·`minimize`와 따로 둘지.
  3. `curve_fit`을 둘지(지금 안: 두지 않고 `regression.nls`를 쓴다).
  4. 사용자 함수의 추가 인자 `args=()`를 둘지(nls에는 없다; 클로저로 대신할 수 있다).
  5. 시험 문제와 대조: MGH 35개와 scipy 시험 함수로 충분한지, MINPACK·L-BFGS-B Fortran 원본의 결과를 고정 데이터로 만들지.
- **O1**: 반복 안의 차분 꼴(2점 기본, 3·5점 선택), `least_squares`의 다듬기 옵션, `loss`·`x_scale`·`lsmr`·`jac_sparsity`를 넣을지, nls를 옮기는 순서(먼저 옮기고 비트 확인 뒤 공개 함수).
- **O2**: 구간법의 기본 허용오차(scipy 값 또는 반올림까지), 근을 끼지 않는 구간의 결과(ValueError 또는 NaN과 `converged=False`), 배치의 콜백 규약(남은 원소만 넘기기).
- **O3–O4**: 기본 방법(scipy: 경계가 없으면 BFGS, 있으면 L-BFGS-B), `callback`을 둘지, `hess_inv`를 결과에 둘지, 유한하지 않은 함수값의 처리(걸음 거절 또는 멈춤).
- **O5**: hybr의 기본 허용오차(scipy 1.49×10⁻⁸), `root`의 다른 방법(broyden 등)을 넣을지.
