# sd.stats 다변량·행렬 분포 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/multivariate.json에서 만든다. 손으로 고치지 않는다.

참값은 Arb로 계산한 값이다: 밀도·엔트로피는 tests/stats/ref_mv(tools/mv_ref.py; 행렬식과 풀이는 주어진 행렬 그대로 Arb의 LU), cdf는 tests/stats/ref_mvcdf(tools/mvcdf_ref.py; factor_t는 2단 적응 구적의 약 1e-11 참값)다. 밀도·엔트로피의 점은 모두 보장 영역(κ ≤ 1e9, d·p ≤ 100, Wishart df ≤ 1e30) 안이고 계약은 |오차| ≤ 1e-14·max(1, |값|), pdf는 그 오차가 exp를 지난 상대 1e-14·max(1, |logpdf|)다(|logpdf| < 700인 점). cdf의 영역과 한계는 tests/stats/test_mvcdf.py의 것이다: guaranteed는 상대 1e-14, loose는 그 밖의 정규수 범위(bvn 10·eps·max(1, −ln P), tvn 8u·조건수), tiny는 P < 1e-300(절대 1e-300), bvt·tvt는 상대 5e-15·1e-14, QMC(atol=…)는 오차 추정 ≤ atol, 실제 오차 ≤ 1.5·atol, 실제 오차 ≤ 2·추정 + 여유이고, factor_t의 d ≤ 3은 정확한 혼합이라 상대 1e-13이다.

- ULP: |sd − 참값|을 참값에서의 float64 간격으로 나눈 값. 99.9%는 그 분위수다.
- ULP/max(1, κ): 역함수의 계약 단위(κ는 입력에 대한 상대 조건수). 역함수만 있다.
- mixed: |sd − 참값|/max(1, |참값|)를 2⁻⁵² 단위로 잰 값(45가 1e-14다). 0을 지나는 값(log 밀도, 엔트로피, gammaln 등)의 계약 척도이고, 그 함수에만 있다. 최대와 99.9%다.
- 상대오차: |sd − 참값|/|참값|. 노름 상대오차: 벡터 필드의 ‖sd − 참값‖₂/‖참값‖₂.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.
- 영역: guaranteed는 보장 영역, loose와 mixed는 그 밖(mixed는 그 척도로 잰다).
- 최대 절대오차: |sd − 참값|의 최대.
- 상대오차/(u·조건수): 입력의 반올림(u = 2⁻⁵³)이 조건수만큼 커진 크기에 대한 상대오차. 조건수는 입력마다 |d log P/d log x|의 합이다(sd 자신의 값의 중앙 차분).
- 오차/추정: 실제 오차를 sd가 돌려준 오차 추정(표준오차의 3배)으로 나눈 값의 최대. 추정은 통계적이라 1을 넘는 점이 있다.

측정: 20261120-152811-deeafe2 (2026-11-20T15:28:11, deeafe2 dirty).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | mixed (최대 / 99.9%) | 최대 상대오차 | 최대 절대오차 | 상대오차/(u·조건수) | 오차/추정 | 계약 대비 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Dirichlet.entropy` | guaranteed | 5 | 2.7 | 2.7 | 1.52 / 1.52 | 3.38e-16 |  |  |  | 0.034 |
| `Dirichlet.logpdf` | guaranteed | 20 | 5.35 | 5.35 | 0.878 / 0.878 | 7.77e-16 |  |  |  | 0.02 |
| `InverseWishart.entropy` | guaranteed | 52 | 5.86 | 5.86 | 4.41 / 4.41 | 9.80e-16 |  |  |  | 0.098 |
| `InverseWishart.logpdf` | guaranteed | 247 | 1.23e+03 | 1.23e+03 | 1.22e+03 / 1.22e+03 | 2.70e-13 |  |  |  | **27 초과** |
| `MatrixNormal.entropy` | guaranteed | 13 | 0.443 | 0.443 | 0.376 / 0.376 | 8.35e-17 |  |  |  | 0.0084 |
| `MatrixNormal.logpdf` | guaranteed | 52 | 1.02 | 1.02 | 0.738 / 0.738 | 1.64e-16 |  |  |  | 0.016 |
| `Multinomial.entropy` | guaranteed | 5 | 3.4 | 3.4 | 2.22 / 2.22 | 4.92e-16 |  |  |  | 0.049 |
| `Multinomial.logpmf` | guaranteed | 25 | 0.499 | 0.499 | 0.404 / 0.404 | 8.98e-17 |  |  |  | 0.009 |
| `MultivariateNormal.entropy` | guaranteed | 17 | 0.483 | 0.483 | 0.317 / 0.317 | 7.04e-17 |  |  |  | 0.007 |
| `MultivariateNormal.logpdf` | guaranteed | 150 | 4.43 | 4.43 | 3.67 / 3.67 | 8.14e-16 |  |  |  | 0.081 |
| `MultivariateNormal.pdf` | guaranteed | 132 | 1.76 | 1.76 |  | 2.28e-16 |  |  |  | 0.01 |
| `MultivariateT.entropy` | guaranteed | 68 | 2.32 | 2.32 | 1.41 / 1.41 | 3.12e-16 |  |  |  | 0.031 |
| `MultivariateT.logpdf` | guaranteed | 600 | 4.11 | 4.11 | 2.28 / 2.28 | 5.05e-16 |  |  |  | 0.051 |
| `MultivariateT.pdf` | guaranteed | 564 | 133 | 133 |  | 1.70e-14 |  |  |  | 0.016 |
| `Wishart.entropy` | guaranteed | 52 | 1.96 | 1.96 | 1.62 / 1.62 | 3.60e-16 |  |  |  | 0.036 |
| `Wishart.logpdf` | guaranteed | 247 | 10 | 10 | 6.57 / 6.57 | 1.46e-15 |  |  |  | 0.15 |
| `cdf.bvn` | guaranteed | 4778 | 42.1 | 24.6 |  | 5.20e-15 |  |  |  | 0.52 |
| `cdf.bvn` | loose | 2110 | 347 | 300 |  | 5.20e-14 |  |  |  | 0.1 |
| `cdf.bvn` | tiny | 732 |  |  |  |  | 9.45e-315 |  |  | 9.4e-15 |
| `cdf.bvt` | guaranteed | 119 | 3.93 | 3.93 |  | 4.71e-16 |  |  |  | 0.094 |
| `cdf.factor_normal` | atol=1e-06 | 84 |  |  |  |  | 2.40e-7 |  | 0.873 | 0.87 |
| `cdf.factor_normal` | atol=0.0001 | 84 |  |  |  |  | 1.65e-6 |  | 0.873 | 0.43 |
| `cdf.factor_t` | exact d<=3 | 24 |  |  |  | 3.98e-15 |  |  |  | 0.04 |
| `cdf.factor_t` | atol=1e-05 | 60 |  |  |  |  | 4.00e-6 |  | 1.16 | 0.81 |
| `cdf.tvn` | guaranteed | 1607 | 64.7 | 39.4 |  | 9.48e-15 |  |  |  | 0.95 |
| `cdf.tvn` | loose | 1205 | 1.05e+03 | 915 |  | 1.48e-13 |  | 2.17 |  | 0.27 |
| `cdf.tvn` | tiny | 167 |  |  |  |  | 2.46e-317 |  |  | 2.5e-17 |
| `cdf.tvt` | guaranteed | 59 | 20.6 | 20.6 |  | 3.45e-15 |  |  |  | 0.34 |
