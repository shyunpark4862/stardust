# G3 정확도: 정규화

`bench/regression/g3/accuracy.py`로 쟀다(2026-11-18, Apple M4 Pro, scikit-learn 1.9.1; 원래 출력은 `accuracy.log`). 참값은 `tools/regref.py`의 `penalized`다: sd 해의 활성 집합(0 아닌 계수)과 부호로 정상 조건을 Arb에서 정확히 풀고(선형은 연립방정식, GLM은 뉴턴법과 Krawczyk 검사), 부호와 0인 계수마다의 엄격한 KKT 부등식 |xⱼᵀW r|/sw < alpha·l1_ratio·sⱼ로 그 해가 최소점임을 증명한다. 오차는 `params` 전체(절편 포함)의 상대오차(노름)다.

## 1. 참값 문제 (n = 60, `tests/regression/ref/penalized.json`)

scikit-learn은 같은 목적함수의 `Lasso`·`ElasticNet`·`Ridge`(svd)·`LogisticRegression`(saga, L2는 newton-cholesky)·`PoissonRegressor`(newton-cholesky)다. `standardize`와 L1이 있는 포아송은 대응하는 것이 없어 비웠다.

| 문제 | sd | sd `dual_gap` | scikit-learn `tol=1e-4` | `tol=1e-12` |
|---|---:|---:|---:|---:|
| lasso | 1.3e-15 | 1.6e-15 | 7.3e-6 | 5.2e-14 |
| lasso_small_alpha | 1.3e-15 | 1.2e-15 | 2.0e-5 | 1.7e-13 |
| lasso_weights_std | 1.3e-15 | 3.8e-15 | | |
| enet | 1.1e-15 | 3.6e-15 | 3.6e-6 | 5.9e-14 |
| enet_weights | 1.5e-15 | 4.8e-15 | 6.5e-5 | 5.9e-13 |
| enet_noint | 6.9e-16 | 8.0e-15 | 7.8e-5 | 1.4e-12 |
| enet_std | 9.9e-16 | 2.2e-15 | | |
| ridge | 8.3e-16 | | 1.2e-15 | 1.3e-15 |
| ridge_weights_std | 1.2e-15 | | | |
| ridge_noint | 5.1e-16 | | 7.1e-16 | 5.1e-16 |
| lasso_constant_column | 9.1e-16 | 1.7e-15 | 2.2e-5 | 1.5e-13 |
| lasso_kappa1e6 | 7.6e-17 | 1.7e-17 | 7.2e-5 | 7.4e-13 |
| enet_kappa1e6_refined | 5.3e-17 | 2.8e-17 | 6.0e-2 | 7.8e-10 |
| ridge_kappa1e6 | 1.4e-17 | | 2.5e-9 | 2.6e-12 |
| binomial, l1_ratio 1 | 1.4e-16 | 1.7e-17 | 1.2e-4 | 5.8e-13 |
| binomial, 0.5 | 1.2e-16 | 5.5e-17 | 2.2e-4 | 1.4e-12 |
| binomial, 0 | 1.0e-16 | 0 | 7.7e-8 | 3.2e-15 |
| poisson, 1 | 1.4e-16 | 6.5e-17 | | |
| poisson, 0.5 | 2.8e-16 | 7.3e-17 | | |
| poisson, 0 | 1.0e-16 | 0 | 5.1e-7 | 1.2e-13 |
| binomial_weights_std | 1.5e-16 | 9.8e-17 | | |

- `dual_gap`은 목적함수 단위다(scikit-learn의 `dual_gap_`은 sw배). 연마한 해에서는 반올림 수준이다.
- `kappa1e6`은 중심화한 설계의 특이값이 1에서 10⁻⁶까지인 문제다. lasso(alpha = 1e-4)와 엘라스틱 넷(alpha = 3e-5)은 활성 집합의 Gram 블록 조건수가 커서(엘라스틱 넷은 약 10⁹) 연마가 데이터의 잔차로 개선했고, ridge(alpha = 1e-8)는 해의 조건수 (s₁² + alpha)/(s_min² + alpha)가 커서 개선했다. 개선의 잔차가 double-double이라(G51) 셋 다 u 수준이다. saga는 무작위 순서라 그 열의 값은 실행마다 조금 다르다.

## 2. 조건수 (n = 200, k = 8, 씨앗 2개의 최대; sd / scikit-learn `tol=1e-4`)

X = U diag(s) Vᵀ + 3(s는 중심화 뒤 1에서 1/κ까지 등비), y = Xβ + 잡음.

| κ | lasso alpha_max/20 | lasso 1e-3·alpha_max | 엘라스틱 넷 1e-3 | ridge 1e-3 | ridge 1e-8 |
|---:|---|---|---|---|---|
| 1 | 1.7e-15 / 6.7e-7 | 9.4e-16 / 5.9e-7 | 5.8e-16 / 5.9e-7 | 1.1e-15 / 3.0e-15 | 1.3e-15 / 3.8e-15 |
| 1e2 | 8.6e-15 / 3.4e-3 | 1.9e-16 / 4.6e-2 | 2.3e-16 / 1.2e-2 | 9.7e-17 / 3.0e-14 | 2.0e-16 / 2.5e-13 |
| 1e4 | 2.1e-15 / 1.8e-4 | 1.0e-16 / 1.9e-2 | 9.7e-17 / 9.1e-3 | 1.8e-16 / 1.3e-13 | 1.8e-16 / 1.5e-9 |
| 1e6 | 7.0e-17 / 5.1e-3 | 2.8e-16 / 3.4e-1 | 5.8e-17 / 7.9e-3 | 1.5e-16 / 8.0e-14 | 8.7e-17 / 4.1e-9 |

- scikit-learn의 기본 `tol`은 쌍대 간격만 보므로 계수가 √(간격) 남짓까지만 맞는다(간격 ≥ (μ/2)‖β − β*‖², μ는 활성 집합의 이차 항의 가장 작은 고윳값). 조건수가 크면 μ가 작아 1e-3–0.3이 된다.
- 개선이 도는 칸(ridge는 해의 조건수 > 64k, lasso·엘라스틱 넷은 활성 Gram 블록의 조건수 어림 > 64a)은 double-double 잔차(G51)로 3e-16 안이다. 개선이 돌지 않는 칸은 Gram 블록의 반올림이 정하고, 가장 큰 것은 κ = 10², lasso alpha_max/20의 8.6e-15다(활성 집합이 작아 조건수 어림이 문턱 아래).

## 3. 행 수 (k = 5)

| n | lasso | 엘라스틱 넷 | ridge | 로지스틱 L1 | 포아송 L2 |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 5.0e-16 | 4.6e-16 | 3.0e-16 | 5.6e-17 | 1.9e-16 |
| 20,000 | 1.7e-16 | 3.6e-16 | 5.6e-16 | 3.7e-17 | 1.2e-16 |

## 4. ridge, 최소제곱 가까이 (n = 200, k = 8, 씨앗 2개의 최대)

2의 문제에서 κ = 10⁸, 10¹⁰, alpha = 1e-12, 1e-18. 개선의 잔차와 기울기의 합은 double-double이다(G51; G1의 OLS와 같은 방식).

| κ | alpha 1e-12 | alpha 1e-18 |
|---:|---:|---:|
| 1e8 | 1.6e-16 | 2.3e-16 |
| 1e10 | 1.4e-16 | 1.7e-15 |
