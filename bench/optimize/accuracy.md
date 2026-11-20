# sd.optimize 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/optimize.json에서 만든다. 손으로 고치지 않는다.

반복 방법의 결과는 멈춤 판정을 만족한 점이라, 계약은 결과만으로 확인되는 판정이다(docs/design/optimize.md §4). 구간법(brentq·toms748·bisect)의 기본 허용오차에서는 근이 Arb 근(tests/optimize/data/scalar_roots.json, tools/optimize_ref.py; 근 문제 204개)에서 xtol + rtol·|x*| 안이거나 그 점의 f가 double로 정확히 0이다(계약 대비는 거리/(xtol + rtol·|x|)). 기울기로 멈추는 minimize(bfgs, trust-ncg, trust-exact, 투영 기울기로 멈춘 l-bfgs-b)는 수렴했다고 하면 정확한 기울기의 노름이 gtol 이하다(bfgs·l-bfgs-b는 ∞-노름 1e-5, trust는 2-노름 1e-4; MGH 문제 tests/optimize/mgh.py, 헤시안이 필요한 방법은 n ≤ 10). 그 밖은 기록만 한다(기준선과의 비교): xtol = 0과 find_root의 근의 ULP(근이 0이면 절대 거리), 수렴한 minimize의 ‖∇f‖∞와 MGH의 알려진 최솟값 f* 대비 초과, least_squares가 NIST StRD 인증값(27문제 × 시작점 2)에 6자리까지 닿은 적합의 모수 상대오차와 다른 점에서 멈춘 적합, root(MINPACK 방정식계 55경우)가 수렴한 점의 ‖F‖∞, lsq_linear·nnls의 KKT 잔차, approx_fprime(기본 차분)과 해석적 도함수의 노름 상대 차이(MGH 47경우의 x0; 도함수는 tests/optimize/ref/derivatives.npz, tools/optimize_accuracy_ref.py로 Arb에서 계산했다).

- ULP: |sd − 참값|을 참값에서의 float64 간격으로 나눈 값. 99.9%는 그 분위수다.
- 상대오차: |sd − 참값|/|참값|.
- 노름 상대오차: 벡터 필드의 ‖sd − 참값‖₂/‖참값‖₂.
- 최대 절대오차: |sd − 참값|의 최대.
- 최대 ‖∇f‖: 결과에서의 정확한 기울기의 노름(gradient stop 영역은 그 방법의 판정 노름).
- 최대 (f − f*)/max(1, f*): MGH의 알려진 최솟값보다 높은 정도. 다른 극소점에서 멈추면 크다.
- 최대 ‖F‖∞: 수렴한 점에서의 방정식계의 잔차.
- KKT 잔차/eps: 자유 변수의 기울기와 경계에서 잘못된 부호의 기울기 가운데 가장 큰 것을 eps·‖A‖₁(‖A‖∞‖x‖∞ + ‖b‖∞)로 나눈 값.
- 수렴 못 함: 수렴하지 못했거나 에러를 낸 실행의 수(그 실행은 다른 열에 들지 않는다).
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.
- 비교 패키지(scipy 1.18.1): 같은 참값, 같은 점에서 같은 척도로 잰 그 라이브러리의 오차다. 같은 양을 계산하는 공개 함수가 없으면 —, R이 없는 기계에서는 not run, 함수가 입력 전체를 거부하면 에러, 시간(30초)·메모리(2GB) 한도를 넘으면 멈춤이다. 제외: 참값은 유한한데 그 라이브러리가 NaN·inf를 낸 점의 수(정의가 다르거나 실패한 점; 다른 열에서 빠진다). compare는 이 열들의 변화를 알리기만 하고 종료 코드에 넣지 않는다.

측정: 20261120-173620-0a89dde (2026-11-20T17:36:20, 0a89dde).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | 최대 상대오차 | 노름 상대오차 | 최대 절대오차 | 최대 ‖∇f‖ | 최대 (f − f*)/max(1, f*) | 최대 ‖F‖∞ | KKT 잔차/eps | 수렴 못 함 | 계약 대비 | scipy 최대 ULP | scipy 99.9% ULP | scipy 최대 상대오차 | scipy 노름 상대오차 | scipy 최대 ‖∇f‖ | scipy 최대 ‖F‖∞ | scipy KKT 잔차/eps | scipy 수렴 못 함 | scipy 계약 대비 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `approx_fprime` | gradient | 47 |  |  |  | 6.44e-11 |  |  |  |  |  |  |  |  |  |  | 2.15e+294 |  |  |  |  |  |
| `approx_fprime` | jacobian | 47 |  |  |  | 1.21e-11 |  |  |  |  |  |  |  |  |  |  | 1.10e-6 |  |  |  |  |  |
| `find_root` | default | 191 | 2 | 2 |  |  |  |  |  |  |  | 0 |  | 2 | 2 |  |  |  |  |  | 0 |  |
| `find_root` | root 0 | 13 |  |  |  |  | 5.55e-17 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `least_squares.lm` | certified minimum | 35 |  |  | 6.61e-7 |  |  |  |  |  |  | 2 |  |  |  | 6.61e-7 |  |  |  |  | 3 |  |
| `least_squares.lm` | elsewhere | 19 |  |  | 604 |  |  |  |  |  |  |  |  |  |  | 604 |  |  |  |  |  |  |
| `least_squares.trf` | certified minimum | 35 |  |  | 7.72e-7 |  |  |  |  |  |  | 2 |  |  |  | 7.72e-7 |  |  |  |  | 2 |  |
| `least_squares.trf` | elsewhere | 19 |  |  | 30.2 |  |  |  |  |  |  |  |  |  |  | 30.2 |  |  |  |  |  |  |
| `lsq_linear` | cases | 12 |  |  |  |  |  |  |  |  | 1.27e+12 |  |  |  |  |  |  |  |  | 1.27e+12 |  |  |
| `minimize.bfgs` | mgh | 102 |  |  |  |  |  | 9.78e-6 | 49 |  |  | 8 |  |  |  |  |  | 9.98e-6 |  |  | 7 |  |
| `minimize.bfgs` | gradient stop | 102 |  |  |  |  |  | 9.78e-6 |  |  |  |  | 0.98 |  |  |  |  | 9.98e-6 |  |  |  | 1 |
| `minimize.l-bfgs-b` | mgh | 109 |  |  |  |  |  | 3.11e+6 | 1.09e+7 |  |  | 1 |  |  |  |  |  | 3.11e+6 |  |  | 0 |  |
| `minimize.l-bfgs-b` | gradient stop | 48 |  |  |  |  |  | 9.56e-6 |  |  |  |  | 0.96 |  |  |  |  | 9.56e-6 |  |  |  | 0.96 |
| `minimize.nelder-mead` | mgh | 62 |  |  |  |  |  | 1.82e+12 | 1.73e+6 |  |  | 48 |  |  |  |  |  | 1.82e+12 |  |  | 48 |  |
| `minimize.newton-cg` | mgh | 81 |  |  |  |  |  | 0.179 | 49 |  |  | 20 |  |  |  |  |  | 0.179 |  |  | 20 |  |
| `minimize.trust-exact` | mgh | 94 |  |  |  |  |  | 8.34e-5 | 49 |  |  | 7 |  |  |  |  |  | 8.34e-5 |  |  | 7 |  |
| `minimize.trust-exact` | gradient stop | 94 |  |  |  |  |  | 9.54e-5 |  |  |  |  | 0.95 |  |  |  |  | 9.54e-5 |  |  |  | 0.95 |
| `minimize.trust-ncg` | mgh | 96 |  |  |  |  |  | 9.77e-5 | 49 |  |  | 5 |  |  |  |  |  | 9.77e-5 |  |  | 5 |  |
| `minimize.trust-ncg` | gradient stop | 96 |  |  |  |  |  | 9.77e-5 |  |  |  |  | 0.98 |  |  |  |  | 9.77e-5 |  |  |  | 0.98 |
| `nnls` | cases | 12 |  |  |  |  |  |  |  |  | 0.355 |  |  |  |  |  |  |  |  | 0.355 |  |  |
| `root.hybr` | converged | 41 |  |  |  |  |  |  |  | 9.06e-9 |  | 14 |  |  |  |  |  |  | 9.06e-9 |  | 14 |  |
| `root.lm` | converged | 46 |  |  |  |  |  |  |  | 0.0483 |  | 9 |  |  |  |  |  |  | 0.0483 |  | 9 |  |
| `root_scalar.bisect` | default | 204 |  |  |  |  |  |  |  |  |  | 0 | 0.89 |  |  |  |  |  |  |  | 0 | 0.89 |
| `root_scalar.bisect` | xtol=0 | 191 | 4 | 4 |  |  |  |  |  |  |  | 0 |  | — | — | — | — | — | — | — | — | — |
| `root_scalar.bisect` | xtol=0, root 0 | 13 |  |  |  |  | 0 |  |  |  |  |  |  | — | — | — | — | — | — | — | — | — |
| `root_scalar.brentq` | default | 185 |  |  |  |  |  |  |  |  |  | 19 | 0.43 |  |  |  |  |  |  |  | 19 | 0.43 |
| `root_scalar.brentq` | xtol=0 | 189 | 3 | 3 |  |  |  |  |  |  |  | 7 |  | — | — | — | — | — | — | — | — | — |
| `root_scalar.brentq` | xtol=0, root 0 | 8 |  |  |  |  | 0 |  |  |  |  |  |  | — | — | — | — | — | — | — | — | — |
| `root_scalar.toms748` | default | 204 |  |  |  |  |  |  |  |  |  | 0 | 0.5 |  |  |  |  |  |  |  | 0 | 0.5 |
| `root_scalar.toms748` | xtol=0 | 191 | 1 | 1 |  |  |  |  |  |  |  | 0 |  | — | — | — | — | — | — | — | — | — |
| `root_scalar.toms748` | xtol=0, root 0 | 13 |  |  |  |  | 0 |  |  |  |  |  |  | — | — | — | — | — | — | — | — | — |
