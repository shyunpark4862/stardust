# sd.optimize 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/optimize.json에서 만든다. 손으로 고치지 않는다.

반복 방법의 결과는 멈춤 판정을 만족한 점이라, 계약은 결과만으로 확인되는 판정이다(docs/design/optimize.md §4). 구간법(brentq·toms748·bisect)의 기본 허용오차에서는 근이 Arb 근(tests/optimize/ref/scalar_roots.json, tools/optimize_ref.py; 근 문제 204개)에서 xtol + rtol·|x*| 안이거나 그 점의 f가 double로 정확히 0이다(계약 대비는 거리/(xtol + rtol·|x|)). 기울기로 멈추는 minimize(bfgs, trust-ncg, trust-exact, 그리고 l-bfgs-b는 투영 기울기로 멈추는 문제 목록)는 수렴했다고 하면 정확한 기울기의 노름이 gtol 이하다(bfgs·l-bfgs-b는 ∞-노름 1e-5, trust는 2-노름 1e-4; MGH 문제 tests/optimize/mgh.py, 헤시안이 필요한 방법은 n ≤ 10). 그 밖은 기록만 한다(기준선과의 비교): xtol = 0과 find_root의 근의 ULP(Arb 근은 이웃한 두 double의 괄호라 0은 괄호 안, 곧 그 둘 중 하나다; 근이 0이면 절대 거리), minimize_scalar(brent·golden·bounded)와 find_minimum이 멈춘 점에서 가장 가까운 Arb 극소점까지의 거리(절대·상대·허용오차 대비; 문제 57개, tests/optimize/ref/scalar_minima.json), 수렴한 minimize의 ‖∇f‖∞와 MGH의 알려진 최솟값 f* 대비 초과, least_squares가 수렴한 적합의 NIST StRD 인증값(27문제 × 시작점 2) 대비 모수 상대오차(인증값에 6자리까지 닿는 실행의 목록과 다른 점에서 멈춘 적합), root(MINPACK 방정식계 55경우)가 수렴한 점의 ‖F‖∞, lsq_linear·nnls의 KKT 잔차, approx_fprime(기본 차분)과 해석적 도함수의 노름 상대 차이(MGH 47경우의 x0; 도함수는 tests/optimize/ref/derivatives.npz, tools/optimize_ref.py로 Arb에서 계산했다). 비교 패키지는 scipy(같은 이름의 방법)와 R이다. R은 reticulate로 같은 Python 시험 함수를 불러(함수 값이 같은 비트) uniroot(brentq와 같은 xtol), optimize(minimize_scalar 'bounded'와 같은 Brent fmin, tol = xatol = 1e-5), optim(BFGS·L-BFGS-B·Nelder-Mead), minpack.lm의 nls.lm('lm')을 잰다. 반복 상한과 허용오차는 R에 같은 설정이 있으면 Python 호출의 기본값에 맞췄다: optim은 maxit = 200n(Nelder-Mead는 f 호출 수, BFGS는 반복 수), L-BFGS-B는 maxit = 15000·lmm = 10·pgtol = 1e-5, nls.lm은 ftol = ptol = gtol = 1e-8·maxfev = 100n·maxiter = 1024. 멈춤 판정 자체는 R의 것이다(optim의 BFGS·Nelder-Mead는 f의 상대 감소 √eps로 멈춘다). 속도 측정(bench/speed)도 같은 호출을 잰다.

- ULP: sd의 값과 참값을 double로 반올림한 값 사이에 있는 double의 수(ulp_distance; 0이면 올바르게 반올림한 값, 1이면 그 이웃). 99.9%는 그 분위수다.
- 상대오차: |sd − 참값|/|참값|.
- 노름 상대오차(∞): max|sd − 참값|/max|참값|, 벡터·행렬의 원소 가운데 가장 큰 오차를 가장 큰 참값으로 나눈 값.
- 최대 절대오차: |sd − 참값|의 최대.
- 최대 ‖∇f‖: 결과에서의 정확한 기울기의 노름(gradient stop 영역은 그 방법의 판정 노름).
- 최대 (f − f*)/max(1, f*): MGH의 알려진 최솟값보다 높은 정도. 다른 극소점에서 멈추면 크다.
- 거리/허용오차: 가장 가까운 Arb 극소점까지의 거리를 그 점에서 방법의 멈춤 허용오차(brent 1.48e-8·|x*| + 1e-11, golden 1.49e-8·|x*|, bounded √eps·|x*| + xatol/3, find_minimum √eps·|x*|)로 나눈 값. 극소점에서 f가 평평할수록(f″ = 0) 크다: f를 double로 구분할 수 있는 거리가 허용오차보다 넓다.
- 최대 ‖F‖∞: 수렴한 점에서의 방정식계의 잔차.
- KKT 잔차/eps: 자유 변수의 기울기와 경계에서 잘못된 부호의 기울기 가운데 가장 큰 것을 eps·‖A‖₁(‖A‖∞‖x‖∞ + ‖b‖∞)로 나눈 값.
- 수렴 못 함: 수렴하지 못했거나 에러를 낸 실행의 수(그 실행은 다른 열에 들지 않는다).
- 판정 밖: 영역의 문제 목록(l-bfgs-b가 투영 기울기로 멈추는 문제, least_squares가 인증값에 6자리까지 닿는 실행)에 있는데 그 판정에 닿지 못한 실행의 수. 목록은 sd의 결과와 상관없이 고정이라 늘면 compare가 짚는다.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.
- 비교 패키지(scipy 1.18.1, R 4.6.2): 같은 참값, 같은 점에서 같은 척도로 잰 그 라이브러리의 오차다. 같은 양을 계산하는 공개 함수가 없으면 —, R이 없는 기계에서는 not run, 함수가 입력 전체를 거부하면 에러, 시간(600초)·메모리(2GB) 한도를 넘으면 멈춤, R의 glm이 기본 설정으로 시작값을 찾지 못하면 시작값 필요다. 제외: 참값은 유한한데 그 라이브러리가 NaN·inf를 낸 점의 수(정의가 다르거나 실패한 점; 다른 열에서 빠진다). compare는 이 열들의 변화를 알리기만 하고 종료 코드에 넣지 않는다.

측정: 20261122-060114-d0366c1+ (2026-11-22T06:01:14, d0366c1 dirty, 참값 28d5e52d860b), 20261122-071515-f0a7f92+ (2026-11-22T07:15:15, f0a7f92 dirty, 참값 28d5e52d860b).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | 최대 상대오차 | 노름 상대오차(∞) | 최대 절대오차 | 최대 ‖∇f‖ | 최대 (f − f*)/max(1, f*) | 거리/허용오차 | 최대 ‖F‖∞ | KKT 잔차/eps | 수렴 못 함 | 판정 밖 | 계약 대비 | scipy 최대 ULP | scipy 99.9% ULP | scipy 최대 상대오차 | scipy 최대 ‖∇f‖ | scipy 최대 ‖F‖∞ | scipy 거리/허용오차 | scipy KKT 잔차/eps | scipy 수렴 못 함 | scipy 계약 대비 | R 최대 ULP | R 99.9% ULP | R 최대 상대오차 | R 최대 ‖∇f‖ | R 거리/허용오차 | R 수렴 못 함 | R 계약 대비 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `approx_fprime` | gradient | 47 |  |  |  | 6.44e-11 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | — | — | — | — | — | — | — |
| `approx_fprime` | jacobian | 47 |  |  |  | 1.21e-11 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | — | — | — | — | — | — | — |
| `approx_fprime` | small |x| | 317 |  |  | 1 |  | 5.29e-12 |  |  |  |  |  |  |  |  |  |  | 7.45e+5 |  |  |  |  |  |  | — | — | — | — | — | — | — |
| `find_minimum` | default | 56 |  |  | 0.00135 |  | 0.00269 |  |  | 9.04e+04 |  |  | 0 |  |  |  |  | 0.00135 |  |  | 9.04e+04 |  | 0 |  | — | — | — | — | — | — | — |
| `find_root` | default | 191 | 3 | 3 |  |  |  |  |  |  |  |  | 0 |  |  | 3 | 3 |  |  |  |  |  | 0 |  | — | — | — | — | — | — | — |
| `find_root` | root 0 | 13 |  |  |  |  | 5.55e-17 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | — | — | — | — | — | — | — |
| `least_squares.lm` | certified minimum | 35 |  |  | 6.61e-7 |  |  |  |  |  |  |  | 2 | 0 |  |  |  | 6.61e-7 |  |  |  |  | 3 |  |  |  | 6.61e-7 |  |  | 3 |  |
| `least_squares.lm` | elsewhere | 17 |  |  | 6.40e-4 |  |  |  |  |  |  |  |  |  |  |  |  | 160 |  |  |  |  |  |  |  |  | 160 |  |  |  |  |
| `least_squares.trf` | certified minimum | 35 |  |  | 7.72e-7 |  |  |  |  |  |  |  | 2 | 0 |  |  |  | 7.72e-7 |  |  |  |  | 2 |  | — | — | — | — | — | — | — |
| `least_squares.trf` | elsewhere | 17 |  |  | 0.0012 |  |  |  |  |  |  |  |  |  |  |  |  | 0.0012 |  |  |  |  |  |  | — | — | — | — | — | — | — |
| `lsq_linear` | cases | 12 |  |  |  |  |  |  |  |  |  | 1.27e+12 |  |  |  |  |  |  |  |  |  | 1.27e+12 |  |  | — | — | — | — | — | — | — |
| `minimize.bfgs` | mgh | 102 |  |  |  |  |  | 9.78e-6 | 49 |  |  |  | 8 |  |  |  |  |  | 9.98e-6 |  |  |  | 7 |  |  |  |  | 106 |  | 14 |  |
| `minimize.bfgs` | gradient stop | 102 |  |  |  |  |  | 9.78e-6 |  |  |  |  |  |  | 0.98 |  |  |  | 9.98e-6 |  |  |  |  | 1 | — | — | — | — | — | — | — |
| `minimize.l-bfgs-b` | mgh | 109 |  |  |  |  |  | 3.11e+6 | 1.09e+7 |  |  |  | 1 |  |  |  |  |  | 3.11e+6 |  |  |  | 0 |  |  |  |  | 3.11e+6 |  | 4 |  |
| `minimize.l-bfgs-b` | gradient stop | 48 |  |  |  |  |  | 9.56e-6 |  |  |  |  |  | 0 | 0.96 |  |  |  | 9.56e-6 |  |  |  |  | 0.96 | — | — | — | — | — | — | — |
| `minimize.nelder-mead` | mgh | 62 |  |  |  |  |  | 1.82e+12 | 1.73e+6 |  |  |  | 48 |  |  |  |  |  | 1.82e+12 |  |  |  | 48 |  |  |  |  | 5.81e+27 |  | 29 |  |
| `minimize.newton-cg` | mgh | 81 |  |  |  |  |  | 0.179 | 49 |  |  |  | 20 |  |  |  |  |  | 0.179 |  |  |  | 20 |  | — | — | — | — | — | — | — |
| `minimize.trust-exact` | mgh | 94 |  |  |  |  |  | 8.34e-5 | 49 |  |  |  | 7 |  |  |  |  |  | 8.34e-5 |  |  |  | 7 |  | — | — | — | — | — | — | — |
| `minimize.trust-exact` | gradient stop | 94 |  |  |  |  |  | 9.54e-5 |  |  |  |  |  |  | 0.95 |  |  |  | 9.54e-5 |  |  |  |  | 0.95 | — | — | — | — | — | — | — |
| `minimize.trust-ncg` | mgh | 96 |  |  |  |  |  | 9.77e-5 | 49 |  |  |  | 5 |  |  |  |  |  | 9.77e-5 |  |  |  | 5 |  | — | — | — | — | — | — | — |
| `minimize.trust-ncg` | gradient stop | 96 |  |  |  |  |  | 9.77e-5 |  |  |  |  |  |  | 0.98 |  |  |  | 9.77e-5 |  |  |  |  | 0.98 | — | — | — | — | — | — | — |
| `minimize_scalar.bounded` | default | 57 |  |  | 0.0013 |  | 0.0026 |  |  | 772 |  |  | 0 |  |  |  |  | 0.0013 |  |  | 772 |  | 0 |  |  |  | 0.0013 |  | 772 | 0 |  |
| `minimize_scalar.brent` | default | 57 |  |  | 0.0013 |  | 0.0026 |  |  | 8.79e+04 |  |  | 0 |  |  |  |  | 0.0013 |  |  | 8.79e+04 |  | 0 |  | — | — | — | — | — | — | — |
| `minimize_scalar.golden` | default | 57 |  |  | 0.00138 |  | 0.00276 |  |  | 9.27e+04 |  |  | 0 |  |  |  |  | 0.00138 |  |  | 9.27e+04 |  | 0 |  | — | — | — | — | — | — | — |
| `nnls` | cases | 12 |  |  |  |  |  |  |  |  |  | 0.355 |  |  |  |  |  |  |  |  |  | 0.355 |  |  | — | — | — | — | — | — | — |
| `root.hybr` | converged | 41 |  |  |  |  |  |  |  |  | 9.06e-9 |  | 14 |  |  |  |  |  |  | 9.06e-9 |  |  | 14 |  | — | — | — | — | — | — | — |
| `root.lm` | converged | 46 |  |  |  |  |  |  |  |  | 0.0483 |  | 9 |  |  |  |  |  |  | 0.0483 |  |  | 9 |  | — | — | — | — | — | — | — |
| `root_scalar.bisect` | default | 204 |  |  |  |  |  |  |  |  |  |  | 0 |  | 0.89 |  |  |  |  |  |  |  | 0 | 0.89 | — | — | — | — | — | — | — |
| `root_scalar.bisect` | xtol=0 | 191 | 4 | 4 |  |  |  |  |  |  |  |  | 0 |  |  | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `root_scalar.bisect` | xtol=0, root 0 | 13 |  |  |  |  | 0 |  |  |  |  |  |  |  |  | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `root_scalar.brentq` | default | 185 |  |  |  |  |  |  |  |  |  |  | 19 |  | 0.43 |  |  |  |  |  |  |  | 19 | 0.43 |  |  |  |  |  | 8 | 0.47 |
| `root_scalar.brentq` | xtol=0 | 189 | 3 | 3 |  |  |  |  |  |  |  |  | 7 |  |  | — | — | — | — | — | — | — | — | — | 3 | 3 |  |  |  | 5 |  |
| `root_scalar.brentq` | xtol=0, root 0 | 8 |  |  |  |  | 0 |  |  |  |  |  |  |  |  | — | — | — | — | — | — | — | — | — |  |  |  |  |  |  |  |
| `root_scalar.toms748` | default | 204 |  |  |  |  |  |  |  |  |  |  | 0 |  | 0.5 |  |  |  |  |  |  |  | 0 | 0.5 | — | — | — | — | — | — | — |
| `root_scalar.toms748` | xtol=0 | 191 | 1 | 1 |  |  |  |  |  |  |  |  | 0 |  |  | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `root_scalar.toms748` | xtol=0, root 0 | 13 |  |  |  |  | 0 |  |  |  |  |  |  |  |  | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
