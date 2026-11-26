# 권 구성 초안 (논의 중, 2026-11-25)

확정된 것이 아니다. 사용자와 함께 고쳐 가는 초안이고, 정해진 것은 GUIDE.md로 옮긴다.

## 권 목록 (예비 네 권 + 주제 열두 권; 예비 네 권의 묶음과 5권은 2026-11-25 확정)
| 권 | 제목(가제) | 방식 | stardust 영역 |
|---|---|---|---|
| 1 | Calculus, Linear Algebra, and Differential Equations | 리뷰 | — |
| 2 | Analysis | 리뷰 | — |
| 3 | Probability and Statistics | 리뷰 | — |
| 4 | Computing | 리뷰 | — |
| 5 | Computer Arithmetic | 완전 증명 | 모든 수치 영역의 바탕 (별도 권, 2026-11-25 확정) |
| 6 | Arrays | 완전 증명 | `sd` |
| 7 | Data Frames | 〃 | `sd.frame` |
| 8 | Dense Linear Algebra | 〃 | `sd.linalg` |
| 9 | Sparse Matrices and Graphs | 〃 | `sd.sparse`, `sd.graph` |
| 10 | Elementary and Special Functions | 〃 | 배열의 수학 함수, `sd.special` |
| 11 | Approximation and Numerical Calculus | 〃 | `sd.interpolate`, `sd.fft`, `sd.integrate`, `sd.differentiate`, `sd.smooth` |
| 12 | Optimization | 〃 | `sd.optimize` |
| 13 | Random Numbers and Distributions | 〃 | `sd.random`, `sd.stats` |
| 14 | Statistical Inference | 〃 | `sd.htest` |
| 15 | Statistical Models | 〃 | `sd.regression` |
| 16 | Geometry and Graphics | 〃 | `sd.spatial`, `sd.plot` |

## 예비 권의 과목 구성 — 과목 하나가 부(Part) 하나
범위는 학부 표준 과목의 첫 강의까지. 그 너머는 note 상자.

### 1권 Calculus, Linear Algebra, and Differential Equations (1·2학년 수학)
- Part I Foundations — 논리와 증명, 집합·함수·관계, 수 체계, 셈과 이산 구조(조합, 점화식, 그래프 용어)
- Part II Calculus — 일변수 미적분, 급수와 Taylor 전개, 다변수 미적분, 벡터 미적분
- Part III Linear Algebra — 벡터와 행렬, 연립방정식, 벡터 공간과 선형 사상, 내적과 직교, 행렬식, 고유값과 대각화, 스펙트럼 정리, 특이값 분해, 이차 형식, Jordan 꼴
- Part IV Differential Equations (2026-11-25 확정) — 1계 방정식, 상수 계수 선형 방정식, 선형계와 행렬 지수, 거듭제곱 급수 해, 존재와 유일성(진술), 차분방정식과 점화식. 편미분방정식은 다루지 않는다. 존재와 유일성의 증명은 2권 실해석 부에서 축약 사상 정리의 응용으로

### 2권 Analysis (해석학 계열)
- Part I Real Analysis — 실수의 완비성, 거리 공간, 연속과 컴팩트, 함수열과 고른 수렴, Riemann 적분, R^n의 미분과 역함수·음함수 정리
- Part II Complex Analysis — 해석 함수, Cauchy 정리와 적분 공식, 거듭제곱·Laurent 급수, 유수 정리, 해석적 연속
- Part III Measure Theory — σ-대수와 측도, Lebesgue 측도와 적분, 수렴 정리, 곱측도와 Fubini, L^p 공간, Radon–Nikodym

### 3권 Probability and Statistics (확률·통계 계열)
- Part I Probability — 확률 공간, 확률변수와 분포, 기댓값, 독립, 조건부 기댓값, 수렴의 양식, 큰 수의 법칙, 특성함수와 중심극한정리, 기본 분포, 다변량 정규, Markov 연쇄
- Part II Mathematical Statistics — 통계 모형, 충분성, 점추정(적률법·최대우도), Fisher 정보와 Cramér–Rao, 최대우도추정량의 점근 이론, 신뢰구간, 가설검정(Neyman–Pearson, 우도비), 순서통계량과 비모수 기초, Bayes 추론 기초

### 4권 Computing (컴퓨터 과학 계열)
- Part I Data Structures — 배열과 연결 리스트, 스택·큐, 해시 테이블, 힙, 탐색 트리, union–find
- Part II Algorithms — 복잡도와 점화식, 분할 정복, 정렬과 선택, 그래프 알고리즘, 동적 계획법, 탐욕법, 무작위 알고리즘
- Part III Computer Architecture — 수의 표현, 명령어와 프로세서, 파이프라인과 분기 예측, 메모리 계층, SIMD와 다중 코어, 운영 체제와 언어 실행 환경의 기본

## 주제 권 장 목차 (초안, 5–16권)
완전 증명 방식. API 묶음은 한글판 목차(textbook/OUTLINE.md)의 API 목록을 참고해 다시 나눴다. 집필 전에 공용 빌드에서 공개 API 목록을 뽑아 빠진 것이 없는지 대조한다. 각 권의 Part I은 그 권의 알고리즘이 직접 기대는 이론이다.

### 5권 Computer Arithmetic (8장)
1. Floating-Point Numbers — 집합과 간격, 비트 배치, 반올림, 특수값, 정수 변환
2. Rounding Error Analysis — 표준 모형, 전방·후방 오차, 누적 오차의 한계
3. Conditioning and Stability — 조건수, 후방 안정성, 소거
4. Error-Free Transformations — TwoSum, TwoProduct, FMA, double-double 산술
5. Summation — 재귀·쌍별·보상 합, 정확한 누산, 내적
6. Polynomial and Series Evaluation — Horner, 보상 Horner, 급수 절단
7. Decimal Conversion — 십진수 읽기(Eisel–Lemire), 최단 왕복 출력(Ryū; stardust는 libc++의 `std::to_chars`에 맡긴다)
8. Compilers and Reproducibility — 결합법칙, 축약, 스레드와 비트 재현성

### 6권 Arrays (14장) — `sd`
1. The Array Model — shape·strides·offset, 뷰와 소유, 메모리 순서
2. Data Types and Promotion — `bool int64 float64 complex128`, `result_type can_cast finfo iinfo astype`
3. Array Creation — `asarray zeros ones full arange linspace eye meshgrid from_dlpack` 등
4. Shape Manipulation and Views — `reshape ravel permute_dims concat stack flip roll repeat tile` 등
5. Indexing — 기본·고급 인덱스, 대입, `take take_along_axis nonzero where`
6. Broadcasting and the Elementwise Engine — 브로드캐스트, 이터레이터, 안쪽 루프, SIMD와 스레드 분할
7. The Thread Pool and Determinism — 스레드풀, 분할 규칙, `set_num_threads`
8. Elementwise Arithmetic — 산술·비교·논리·비트, 정수 의미론, 복소 연산, `clip copysign nextafter`
9. Reductions — `sum prod mean var std min max any all`, `nan*`, `cumulative_*`, 계산 순서
10. Order Statistics and Counting — `quantile median`, `bincount histogram cov corrcoef`
11. Sorting, Searching, and Sets — `sort argsort searchsorted unique_*`, 기수·비교 정렬, NaN 순서
12. Array Products — `matmul tensordot vecdot`, BLAS
13. Printing — `repr`·`str`
14. Interchange — 버퍼 프로토콜, DLPack, numpy와의 경계

### 7권 Data Frames (13장) — `sd.frame`
1. The Columnar Model — 열 버퍼, 유효 비트맵, `Series DataFrame`, Copy-on-Write
2. Missing Values — NaN과 `NA`, `isna fillna dropna`
3. Construction and Type Inference
4. Indexes and Hashing — `Index RangeIndex`, 라벨 해시 테이블, `reindex`
5. Selection, Alignment, and Assignment — `[] .loc .iloc`
6. Elementwise Operations — 산술·비교, `astype`, 문자열 열
7. Reductions and Descriptive Methods
8. Uniqueness and Counting — `unique_values duplicated drop_duplicates value_counts`
9. Group-By — 키 인코딩, 해시 집계, `agg`
10. Sorting Tables — `sort_values sort_index`
11. Joins and Concatenation — `merge`, `concat`
12. Reading CSV — 토크나이저, 타입 추론, 청크 병렬
13. Table Display

### 8권 Dense Linear Algebra (13장) — `sd.linalg`
Part I Theory
1. Matrix Norms and Perturbation Theory — 노름, 조건수, 선형계와 고유값의 섭동
2. Orthogonal Transformations — Householder 반사, Givens 회전, 그 안정성
Part II Linear Systems
3. Basic Operations and BLAS — `vecdot outer cross kron trace vector_norm matrix_norm`
4. Gaussian Elimination and LU — `lu lu_solve solve det slogdet inv`
5. Triangular, Banded, and Tridiagonal Systems — `solve_triangular solve_banded solve_tridiagonal`
6. Symmetric Factorizations — `cholesky cholesky_solve ldl`
7. Condition Estimation — `cond`
Part III Least Squares and Eigenvalues
8. QR Factorization — `qr qr_pivoted`
9. Least Squares and Rank — `lstsq pinv matrix_rank`
10. The Symmetric Eigenvalue Problem — `eigh eigvalsh`
11. The Nonsymmetric Eigenvalue Problem — `eig eigvals schur hessenberg`
12. The Singular Value Decomposition — `svd svdvals`
13. Matrix Functions and Matrix Equations — `expm matrix_power solve_sylvester`

### 9권 Sparse Matrices and Graphs (14장) — `sd.sparse`, `sd.graph`
Part I Sparse Matrices
1. Storage Formats — `coo_array csr_array csc_array dok_array`
2. Construction and Indexing — `eye_array diags_array random_array kron hstack vstack`
3. Sparse Arithmetic — 행렬–벡터 곱, 희소–희소 곱, 리덕션
4. Sparse Direct Methods — `spsolve splu factorized`, 채움과 순서 매기기
5. Krylov Subspaces — `LinearOperator`, Arnoldi, Lanczos
6. Iterative Methods for Linear Systems — `cg minres gmres bicgstab`, 전처리
7. Iterative Least Squares — `lsqr lsmr`
8. Large Eigenvalue Problems — `eigsh eigs svds`
9. Sparse Matrix Functions — `expm expm_multiply onenormest norm`
Part II Graphs
10. Graph Representation and Traversal — `connected_components breadth_first_* depth_first_* topological_sort`
11. Shortest Paths — `shortest_path dijkstra bellman_ford johnson floyd_warshall yen`
12. Trees, Flows, and Matchings — `minimum_spanning_tree maximum_flow maximum_bipartite_matching linear_sum_assignment`
13. Connectivity and Centrality — `articulation_points bridges core_number pagerank betweenness_centrality`
14. Spectral and Ordering Methods — `laplacian reverse_cuthill_mckee structural_rank`

### 10권 Elementary and Special Functions (17장) — 배열의 수학 함수, `sd.special`
Part I Theory
1. Asymptotic Expansions — 점근 급수, Watson 보조정리, Laplace 방법, 최급강하법
2. Differential Equations of Special Functions — Frobenius 방법, 연결 공식, Liouville–Green 근사
3. Recurrences and Continued Fractions — 3항 점화식의 안정성, Miller 알고리즘, 연분수의 계산
4. Polynomial and Rational Approximation — Chebyshev 급수, minimax 근사, 표 방식
Part II Elementary Functions
5. Exponentials and Logarithms — `exp expm1 log log1p log2 log10 pow`
6. Trigonometric and Hyperbolic Functions — `sin cos tan asin … atanh hypot`, 인자 축소
7. Vectorized Kernels — NEON 커널, 정확도 계약, 특수값
Part III Special Functions
8. Principles of Special Function Computation — 영역 나누기, 방법의 선택, 검증
9. Error Functions — `erf erfc erfcx erfinv ndtr log_ndtr ndtri wofz`
10. The Gamma Function and Its Relatives — `gamma gammaln digamma polygamma beta binom`
11. Incomplete Gamma and Beta Functions — `gammainc gammaincc betainc`와 역함수
12. Logistic and Entropy Functions — `expit logit xlogy entr logsumexp softmax`
13. Zeta Function and Exponential Integrals — `zeta expi expn`
14. Bessel Functions of Integer Order — `j0 j1 y0 y1 i0 i1 k0 k1`
15. Bessel Functions of Real Order and Airy Functions — `jv yv iv kv airy`
16. Elliptic Integrals, Hypergeometric Functions, and Lambert W — `ellipk ellipe hyp1f1 hyp2f1 lambertw`
17. Distribution Functions — `stdtr chdtr fdtr pdtr bdtr`와 역함수, 비중심 꼴

### 11권 Approximation and Numerical Calculus (17장) — `sd.interpolate`, `sd.fft`, `sd.integrate`, `sd.differentiate`, `sd.smooth`
Part I Interpolation
1. Polynomial Interpolation — `BarycentricInterpolator KroghInterpolator`, 오차 공식, Runge 현상
2. Rational Approximation — `FloaterHormannInterpolator AAA`
3. Piecewise Polynomials — `PPoly CubicSpline PchipInterpolator Akima1DInterpolator`, `sd.interp`
4. B-Splines — `BSpline make_interp_spline`
5. Interpolation in Several Variables — `RegularGridInterpolator RBFInterpolator LinearNDInterpolator griddata`
Part II Differentiation and Integration
6. Numerical Differentiation — `derivative jacobian hessian`
7. Quadrature Rules — `fixed_quad newton_cotes trapezoid simpson romb`, Gauss·Kronrod 규칙
8. Adaptive Quadrature — `quad quad_vec dblquad tplquad nquad`
9. Double Exponential Quadrature and Series — `tanhsinh nsum`
10. Multidimensional Integration — `cubature`
Part III Fourier Transforms
11. The Discrete Fourier Transform — 정의와 성질, 표본화와 에일리어싱
12. The Fast Fourier Transform — `fft ifft fft2 fftn`, 혼합 기수, Bluestein
13. Real and Trigonometric Transforms — `rfft irfft hfft dct dst`
14. The Fast Hankel Transform — `fht ifht`
Part IV Smoothing
15. Scatterplot Smoothing — `lowess loess savgol_*`
16. Smoothing Splines — `make_smoothing_spline smoothing_spline pspline`
17. Kernel Methods — `kernel_regression kde`, 대역폭 선택

### 12권 Optimization (11장) — `sd.optimize`
Part I Theory
1. Convexity — 볼록 집합과 함수, 분리 정리, 부분미분
2. Optimality Conditions and Duality — KKT 조건, 쌍대성
Part II Methods
3. Scalar Root Finding — `root_scalar find_root bracket_root`
4. Minimization in One Variable — `minimize_scalar find_minimum bracket_minimum`
5. Line Search and Quasi-Newton Methods — Wolfe 조건, BFGS
6. Limited-Memory and Bound-Constrained Methods — L-BFGS-B, Newton-CG
7. Trust-Region Methods
8. Derivative-Free Optimization — Nelder–Mead
9. Nonlinear Least Squares — `least_squares`
10. Linear Least Squares with Bounds — `lsq_linear nnls`
11. Systems of Nonlinear Equations — `root`

### 13권 Random Numbers and Distributions (9장) — `sd.random`, `sd.stats`
1. Computing with Probabilities — 꼬리 확률, 로그 확률, 분위수의 조건수
2. Random Number Generation — `default_rng Generator`, Philox, 스트림과 재현성
3. Sampling from Distributions — 역변환, 기각, ziggurat
4. The Distribution Framework — `pdf cdf ccdf icdf` 등의 메서드와 정확도
5. Continuous Distributions
6. Discrete Distributions
7. Noncentral Distributions
8. Multivariate Distributions — `MultivariateNormal Dirichlet Wishart` 등
9. Descriptive Statistics — `describe skew kurtosis moment rankdata`

### 14권 Statistical Inference (10장) — `sd.htest`
1. Exact and Asymptotic Null Distributions — 순열 분포, 정확 분포의 동적 계획, 점근 근사
2. Tests for Means and Variances — `ttest_* f_oneway bartlett levene fligner`
3. Goodness of Fit — `chisquare shapiro anderson ks_1samp cramervonmises`
4. Rank Tests — `mannwhitneyu wilcoxon kruskal friedmanchisquare`
5. Comparing Samples — `ks_2samp anderson_ksamp epps_singleton_2samp`
6. Correlation and Association — `pearsonr spearmanr kendalltau somersd`
7. Contingency Tables — `chi2_contingency fisher_exact barnard_exact mcnemar`
8. Proportions and Counts — `binomtest proportions_* proportion_confint`
9. Multiple Comparisons — `tukey_hsd dunnett multipletests combine_pvalues`
10. Resampling and Monte Carlo — `permutation_test monte_carlo_test bootstrap`

### 15권 Statistical Models (6장) — `sd.regression`
1. Linear Models — `ols wls gls`, 추론, 샌드위치 공분산
2. Model Diagnostics and Comparison — `anova`, 영향력
3. Generalized Linear Models — `glm`, IRLS
4. Regularized Regression — `ridge lasso elastic_net *_cv`, 좌표 하강
5. Robust and Quantile Regression — `rlm theilslopes quantreg`
6. Nonlinear Regression and Variable Selection — `nls stepwise best_subset`

### 16권 Geometry and Graphics (11장) — `sd.spatial`, `sd.plot`
Part I Geometry
1. Distances — `pdist cdist squareform`
2. Nearest-Neighbor Search — `KDTree kneighbors radius_neighbors`
3. Geometric Predicates and Convex Hulls — `ConvexHull`, 정확한 산술
4. Delaunay Triangulations and Voronoi Diagrams — `Delaunay Voronoi`
Part II Graphics
5. Rasterization and Compositing
6. Figures, Axes, and Scales
7. Text and Mathematical Typesetting
8. Curves and Adaptive Sampling — `line function parametric implicit contour`
9. Statistical Graphics — `histogram density boxplot qq ecdf`
10. Three-Dimensional Graphics — `surface scatter3d curve3d`
11. File Formats — PNG, PDF, SVG

## API 대조 결과 (2026-11-25, 기준 커밋 66fe52e, 전체는 API-INVENTORY.md)
- 완료 영역 17개(공개 이름 738개)는 모두 권이 있다. `sd.cluster`·embed·classify는 구현 중·예정이라 뺐다.
- 장 줄의 API 이름은 대표만 적은 것이다. 장의 얼개를 짤 때 API-INVENTORY.md의 "목차 대조 (a)" 목록으로 빠짐없이 채운다.
- 내용 자체가 목차에 없어 넣을 자리를 정해 둔 것:
  - 배열 `diff`, `logaddexp` → 6권 8·11장. 배열과 프레임의 경계(`to_array`, `to_numpy`) → 7권 3장
  - 복소 인자의 특수 함수 → 10권 해당 장마다 절로. `nbdtr` 계열 → 10권 17장
  - FFT 도우미(`fftfreq fftshift next_fast_len`) → 11권 12장. FITPACK 계열 스플라인(`make_lsq_spline make_splrep make_splprep generate_knots`) → 11권 4·16장
  - `approx_fprime`, `check_grad` → 11권 6장
  - 다변량 cdf의 Genz 준몬테카를로 → 13권 8장
  - 영역 공통의 결과 객체와 `summary()` → 15권 1장에서 소개, 각 권에서 사용
  - plot의 기본 표시(`scatter bar errorbar area step text annotate`), 벡터장(`phase vectorfield streamlines`), 모형 그림 → 16권 6·8·9장에 절 추가
  - htest 32개, regression의 명세 검정·`penalized_glm`·`siegelslopes`, graph 6개 → 해당 장에 이름 보충

## 주제 권 앞머리로 보낸 이론 (완전 증명)
- 반올림 오차 해석, 조건수와 안정성, 오차 없는 변환 → 5권
- 행렬 섭동 이론 → 8권
- 특수 함수가 직접 쓰는 미분방정식 이론(Frobenius 방법, 3항 점화식의 안정성, 해의 점근 전개) → 10권
- 볼록성과 최적성 조건 → 12권
- 분포 계산과 표본 생성에 직접 쓰이는 분포 이론 → 13권
- 검정 통계량의 정확·점근 분포 → 14권, 선형 모형·GLM 이론 → 15권

## 권 사이의 의존
1권 → 2권(실해석은 미적분 위에), 1·2권 → 3권(확률은 측도론과 선형대수 위에, 통계는 확률 위에). 4권은 1권의 기초 부만 전제한다.

## 1권 장 목차 (초안, 27장)
리뷰 방식: 정리는 모두 state하고 중요한 것만 증명한다. 괄호 안은 증명할 핵심 정리의 후보.

### Part I Foundations
1. Logic and Proof — 명제와 한정 기호, 직접 증명·대우·귀류, 수학적 귀납법, 반례
2. Sets, Functions, and Relations — 집합 연산, 단사·전사·역함수, 동치 관계와 분할, 부분 순서와 전순서, 가산과 비가산 (Cantor의 대각선 논법)
3. Number Systems — 자연수와 정수(나눗셈, Euclid 호제법, 합동), 유리수, 실수(완비 순서체, 상한), 복소수(산술, 극형식), 기본 부등식
4. Counting and Discrete Structures — 순열·조합, 이항 정리, 포함–배제, 비둘기집 원리, 생성함수의 기초, 그래프와 트리의 용어

### Part II Calculus
5. Limits and Continuity — 수열과 함수의 극한, 연속, 중간값 정리, 최대·최소 정리
6. Differentiation — 도함수와 미분 법칙, 평균값 정리, l'Hôpital, 단조성과 볼록성, Newton 방법 맛보기 (평균값 정리)
7. Integration — Riemann 적분, 미적분의 기본 정리, 적분 기법, 이상 적분 (미적분의 기본 정리)
8. Sequences, Series, and Approximation — 수렴 판정, 거듭제곱 급수와 수렴 반지름, Taylor 정리와 나머지, 점근 표기(O, o, ~), Stirling 공식 (Taylor 정리)
9. Elementary Transcendental Functions — 지수·로그, 삼각·역삼각, 쌍곡 함수의 정의와 항등식, 급수 표현
10. Differential Calculus in Several Variables — 편미분, 그래디언트와 방향 도함수, 미분 가능성, 연쇄 법칙, Jacobian·Hessian, 2차 Taylor 전개, 극값과 Lagrange 승수
11. Integral Calculus in Several Variables — 중적분, 반복 적분, 변수 변환과 Jacobian, 극·구면 좌표, Gauss 적분, 감마·베타 함수 맛보기
12. Vector Calculus — 곡선과 선적분, 면적분, Green·Stokes·발산 정리

### Part III Linear Algebra
13. Vectors and Matrices — R^n, 행렬 연산, 전치, 특수한 행렬, 블록 행렬
14. Linear Systems — 소거법과 행 사다리꼴, LU의 생각, 역행렬, 계수
15. Vector Spaces — 부분공간, 일차 독립, 기저와 차원, 네 기본 부분공간
16. Linear Maps — 사상의 행렬, 핵과 상, 기저 변환과 닮음 (계수–퇴화차수 정리)
17. Determinants — 성질, 여인수 전개, 부피, 곱의 행렬식
18. Inner Products and Orthogonality — 노름, 직교 사영, Gram–Schmidt와 QR, 직교·유니터리 행렬, 최소제곱과 정규방정식 (Cauchy–Schwarz, 사영 정리)
19. Eigenvalues and Eigenvectors — 특성다항식, 대각화, Cayley–Hamilton, 복소 고유값, 행렬의 거듭제곱
20. Symmetric Matrices and Quadratic Forms — 양정치, Rayleigh 몫, 최소–최대 원리 (스펙트럼 정리)
21. The Singular Value Decomposition — 기하적 의미, 낮은 계수 근사, 유사역행렬, 행렬 노름과 조건수 (SVD의 존재, Eckart–Young)
22. Canonical Forms and Matrix Functions — Schur 분해, Jordan 꼴, 행렬 지수와 행렬 함수

### Part IV Differential Equations
23. First-Order Equations — 변수 분리, 선형, 완전 방정식, 방향장, 존재와 유일성(진술), Euler 방법 맛보기
24. Linear Equations of Higher Order — 상수 계수와 특성방정식, 미정계수법, 매개변수 변환, Wronskian, 진동 (해 공간의 차원)
25. Linear Systems — 고유값 방법, 행렬 지수, 위상 평면, 안정성
26. Series Solutions — 정상점과 정칙 특이점, Frobenius 방법의 개요, Bessel·Legendre·Airy 방정식
27. Difference Equations and Recurrences — 선형 점화식과 특성근, 생성함수와의 관계, 안정성, 3항 점화식

넣지 않은 것: Laplace 변환, 편미분방정식 (필요하면 note 상자).

## 2권 Analysis 장 목차 (초안, 21장)

### Part I Real Analysis
1. The Real Numbers — 완비성 공리, 상한과 하한, 축소 구간, Cauchy 판정법 (Bolzano–Weierstrass)
2. Metric Spaces — 거리, 열린·닫힌 집합, 수렴과 완비성, R^n의 노름과 그 동치
3. Continuity and Compactness — 연속, 컴팩트, 고른 연속, 연결성 (Heine–Borel, 최대·최소 정리)
4. Differentiation — 도함수의 엄밀한 정의, 평균값 정리, Taylor 정리, 볼록 함수와 Jensen 부등식
5. The Riemann Integral — Darboux 합, 적분 가능성, 미적분의 기본 정리, 이상 적분
6. Sequences and Series of Functions — 점별·고른 수렴, Weierstrass M-판정, 극한의 교환, 거듭제곱 급수, Weierstrass 근사 정리 (고른 극한 정리)
7. Differentiation in R^n — 전미분, 평균값 부등식, 역함수 정리와 음함수 정리 (역함수 정리)
8. Fixed Points and Existence Theorems — 축약 사상 정리, 상미분방정식의 존재와 유일성(Picard–Lindelöf), 반복법의 수렴 (축약 사상 정리)
9. Fourier Series — 직교 함수계, 수렴, Parseval 등식, Gibbs 현상

### Part II Complex Analysis
10. Holomorphic Functions — 복소 미분, Cauchy–Riemann 방정식, 초등 함수와 가지
11. Complex Integration — 경로 적분, Cauchy 정리, Cauchy 적분 공식, Liouville 정리, 최대 절댓값 원리 (Cauchy 적분 공식)
12. Series Representations — Taylor·Laurent 급수, 영점과 특이점, 항등 정리, 해석적 연속
13. The Residue Calculus — 유수 정리, 실적분의 계산, 편각 원리와 Rouché 정리 (유수 정리)
14. Conformal Maps and Harmonic Functions — 등각 사상, Möbius 변환, 조화 함수
15. Entire and Meromorphic Functions — 무한곱, 감마 함수와 그 해석적 연속, 반사 공식

### Part III Measure Theory
16. Measures — σ-대수, 측도, 외측도와 Carathéodory 구성, Lebesgue 측도
17. Measurable Functions and the Lebesgue Integral — 단순 함수, 적분의 정의, Riemann 적분과의 관계
18. Convergence Theorems — 단조 수렴, Fatou 보조정리, 지배 수렴, 수렴의 여러 양식 (단조 수렴, 지배 수렴)
19. Product Measures — Fubini–Tonelli 정리, 합성곱
20. L^p Spaces — Hölder·Minkowski 부등식, 완비성, L^2와 직교 사영
21. Signed Measures and Differentiation — Radon–Nikodym 정리, Lebesgue 분해, 절대 연속, 변수 변환

넣지 않은 것: 적분의 점근 계산(Laplace 방법, 최급강하법)은 10권 앞머리에서 완전 증명으로. 함수해석, 초함수는 다루지 않는다.

## 3권 Probability and Statistics 장 목차 (초안, 20장)

### Part I Probability
1. Probability Spaces — 공리, 셈 모형, 조건부 확률과 Bayes 정리, 사건의 독립
2. Random Variables and Distributions — 분포함수, 이산·연속 분포, 분위수 함수, 확률변수의 변환
3. Expectation — 측도에 대한 적분으로서의 기댓값, 모멘트, 적률생성함수 (Markov·Chebyshev·Jensen 부등식)
4. Random Vectors — 결합분포, 독립, 공분산, 변수 변환, 합성곱, 순서통계량
5. Conditional Expectation — 이산·밀도의 경우, 일반 정의, 성질, L^2 사영
6. Families of Distributions — 이산·연속 분포족, 분포 사이의 관계, 지수족
7. The Multivariate Normal Distribution — 선형 변환, 조건부 분포, 이차 형식과 카이제곱·t·F 분포 (Cochran 정리)
8. Modes of Convergence — 거의 확실·확률·L^p·분포 수렴, Borel–Cantelli, Slutsky, 연속 사상 정리
9. Laws of Large Numbers — 약법칙과 강법칙 (강법칙)
10. Characteristic Functions and the Central Limit Theorem — 반전과 연속성 정리, Lindeberg 조건, Berry–Esseen(진술), 델타 방법 (중심극한정리)
11. Stochastic Processes: A First Look — Markov 연쇄와 정상 분포, 확률 보행, Poisson 과정

### Part II Mathematical Statistics
12. Statistical Models and Sampling — 표본과 통계량, 표집 분포, 경험 분포 (Glivenko–Cantelli)
13. Sufficiency — 분해 정리, 완비성, Rao–Blackwell, Lehmann–Scheffé
14. Point Estimation — 적률법, 최대우도, 편향·분산·평균제곱오차, Fisher 정보 (Cramér–Rao 부등식)
15. Asymptotic Theory — 일치성, 최대우도추정량의 점근 정규성, 효율 (점근 정규성)
16. Confidence Sets — 추축량, 점근 신뢰구간, 붓스트랩 입문
17. Hypothesis Testing — 오류와 검정력, p값, 균일 최강력 검정, 우도비·Wald·점수 검정, Wilks 정리 (Neyman–Pearson 보조정리)
18. Linear Models — 최소제곱, 정규 이론, t·F 검정, 분산분석 (Gauss–Markov 정리)
19. Nonparametric Methods — 순위와 순열 검정, Kolmogorov–Smirnov, 붓스트랩
20. Bayesian Inference — 사전·사후 분포, 켤레족, 신용 구간, 결정 이론의 기초

넣지 않은 것: 마팅게일, 연속 시간 과정은 note 상자. 다중 비교와 검정별 분포는 14권, GLM은 15권.

## 4권 Computing 장 목차 (초안, 24장)

### Part I Data Structures
1. Abstract Data Types and Cost — 추상 자료형, 비용 모형, 분할상환 분석
2. Arrays, Lists, Stacks, and Queues — 동적 배열(두 배 늘리기), 연결 리스트, 덱
3. Hash Tables — 해시 함수, 체이닝과 열린 주소법, 적재율, 보편 해싱
4. Search Trees — 이진 탐색 트리, 균형 트리, B-트리
5. Heaps and Priority Queues — 이진 힙, 힙 만들기, d진 힙
6. Disjoint Sets — union–find, 경로 압축과 랭크

### Part II Algorithms
7. Analysis of Algorithms — 점근 분석, 점화식과 마스터 정리
8. Divide and Conquer — 병합 정렬, Karatsuba, Strassen
9. Sorting and Selection — 퀵 정렬, 비교 정렬의 하한, 계수·기수 정렬, 선택 (하한 정리)
10. Graph Traversal — 그래프의 표현, 너비·깊이 우선 탐색, 위상 정렬, 강연결 요소
11. Shortest Paths and Spanning Trees — Dijkstra, Bellman–Ford, Floyd–Warshall, Kruskal, Prim, 최대 흐름과 매칭의 기초
12. Dynamic Programming
13. Greedy Algorithms
14. Randomized Algorithms — 기댓값 분석, Las Vegas와 Monte Carlo, 표집
15. Computational Complexity — P와 NP, 환산, NP-완전성

### Part III Computer Architecture
16. Data Representation — 비트, 정수(2의 보수), 부동소수점 형식, 문자(ASCII, UTF-8)
17. Digital Logic and Arithmetic Circuits — 게이트, 가산기, 곱셈기
18. Instructions and Machine Language — 명령어 집합, 레지스터, 주소 지정, 호출과 스택
19. The Processor — 데이터 경로, 파이프라인과 해저드, 분기 예측, 비순차 실행
20. The Memory Hierarchy — 캐시와 지역성, 가상 메모리와 TLB
21. Data-Level Parallelism — SIMD와 벡터 레지스터, 벡터화
22. Multicore and Threads — 공유 메모리, 동기화, 경쟁 상태, 거짓 공유, Amdahl의 법칙
23. From Source to Execution — 컴파일과 링크, 인터프리터, 메모리 관리, 운영 체제의 기초
24. Language Interoperability — 호출 규약, 외부 함수 인터페이스, 객체 모델과 참조 계수

넣지 않은 것: k-d 트리 같은 공간 자료구조는 16권, 부동소수점의 오차 해석은 5권.


## 별권 Excursions (가제) — 특별 장 모음 (2026-11-26 사용자와 논의해 정함)
책의 본래 내용과 무관하게, 사용자가 평소 관심 있게 생각하는 주제들을 모은 권. 예비 네 권 뒤(4권 이후)에 둔다. 권 번호를 줄지(5권으로 하고 주제 권을 6–17권으로 미룰지), 번호 없는 별권으로 둘지는 아직 묻는 중.

방식: 리뷰 권과 같되 수학 교양서처럼 쓴다. 결과의 함의와 의미를 충분히 서술하고, 도입·일화·역사·곁이야기를 note 상자가 아니라 본문에서 풀어낸다(이 권에만 허용하는 예외; 서술체 허용). 증명을 대충 한다는 뜻이 아니다: 장마다 중심 정리 하나는 완전히 증명하고 나머지는 진술 + 줄거리 + 참고 문헌. 일화는 출처가 불확실한 것이 많으므로 인용 검증을 본문 권보다 더 엄격히 한다. 장마다 stardust(공용 빌드)로 돌린 작은 모의실험 절을 둔다. 장 단위는 주제 하나에 장 하나를 기본으로 하고, 같은 원리인 것만 합친다. 분량 제한 없음.

묶음 E(유명한 문제들)의 흐름: 그 문제를 독자가 이해할 수 있게 설명하고 → 그간의 성과를 소개하고 → 어떤 의미와 함의를 갖는지 충분히 설명한다. 완전히 증명할 수 있는 조각이 있으면 하나를 중심에 둔다(예: FLT의 n = 4, Euler 곱).

(추가) = 메인이 추천해 사용자가 받아들인 주제.

### Part A — Love and Choice
1. The Romeo and Juliet Equations — 2×2 선형계와 위상 평면(1권 25장)
2. When to Stop — 비서 문제와 1/e 규칙, 변형(되돌아가기, 전체 정보), Bruss의 odds 정리, (추가) prophet 부등식
3. Stable Marriage — Gale–Shapley, 제안하는 쪽 최적성, 전략적 조작, 병원–전공의 매칭, 룸메이트 문제의 반례
4. Becker's Marriage Market — 배정 문제와 쌍대성, 초모듈성 ⇒ 끼리끼리 짝, 최적 수송과의 연결
5. Dividing Fairly — Sperner 보조정리로 집세 나누기, (추가) 케이크 자르기(시샘 없는 분할)

### Part B — Luck and Bets
6. The Martingale System — 도박꾼의 파산, 선택 정지 정리, 이기는 시스템은 없다
7. The Kelly Criterion — 로그 최적 성장, St. Petersburg 역설, Shannon과의 관계, Samuelson의 반론
8. Multi-Armed Bandits — 탐색과 활용, Gittins 지수, Lai–Robbins 하한, UCB, Thompson 표집
9. (추가) Surprises of the Random Walk — 아크사인 법칙, Pólya의 재귀 정리
10. (추가) Parrondo's Paradox — 지는 게임 둘을 섞으면 이긴다

### Part C — Sampling Paradoxes and Networks
11. The Bus Waiting Paradox — 검사 역설과 갱신 이론
12. The Friendship Paradox and the Majority Illusion — 크기에 비례한 표집이 그래프에서 나타난 것
13. Small Worlds — Milgram, Watts–Strogatz, (추가) 무작위 그래프의 문턱과 선호적 연결
14. Rumours and Epidemics — Daley–Kendall, Maki–Thompson, (추가) SIR 모형과 문턱 정리
15. Structural Balance — Cartwright–Harary 정리(균형 ⇔ 두 진영), 균형으로 가는 동역학
16. (추가) Segregation and Thresholds — Schelling의 분리 모형, Granovetter의 문턱 모형

### Part D — Games and Crowds
17. (추가) Games and Equilibrium — 최소최대 정리, Nash 균형(이 부의 공통 토대)
18. Braess's Paradox — Wardrop 균형, 무정부의 대가 4/3
19. Keynes's Beauty Contest — 반복 소거, level-k, 실험 결과
20. The Evolution of Cooperation — 죄수의 딜레마, Axelrod 토너먼트, tit-for-tat, (추가) zero-determinant 전략
21. (추가) Paradoxes of Voting — Condorcet 순환, Arrow의 불가능성 정리

### Part E — Famous Problems
22. The Collatz Conjecture
23. The Riemann Hypothesis
24. Fermat's Last Theorem
25. The Poincaré Conjecture
26. P versus NP
27. (추가) Proofs by Computer — 4색 정리와 Kepler 추측
28. (추가) The Limits of Proof and Computation — Gödel 불완전성과 정지 문제

## 아직 정하지 않은 것
- 1–16권 장 목차의 세부(초안 확인 대기). 주제 권은 공용 빌드의 공개 API 목록과 대조가 남았다.
- 집필 순서
