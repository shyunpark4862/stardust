# 목차 (2026-11-25 사용자 확정)

**Foundations of Scientific Computing — Theory, Algorithms, and Implementation**

GUIDE.md §5의 기준으로 다시 짠 목차다. 옛 목차(PLAN.md §4, 8부 78장)는 이론 중심이어서 stardust의 공개 API 약 750개 가운데 많은 것이 한두 문단이나 instardust 상자로만 다뤄졌다. 새 목차는 라이브러리의 API 목록에서 출발한다.

## 원칙

- 세 축은 대등하다. 장마다 **Theory**(정의, 정리와 증명, 오차·수렴 해석) → **Algorithms**(유사코드, 단계 설명, 복잡도, 안정성) → **Implementation**(자료 배치, 경계와 특수값, 블록화·벡터화·병렬화, 검증, stardust의 선택)의 차례로 간다.
- 간단하지 않은 공개 API는 모두 어느 장의 Algorithms·Implementation 절에서 유사코드로 설명한다. 장 끝에 그 장이 다룬 API의 목록을 싣는다.
- 장마다 그림을 둔다. 개요는 5줄 이내.
- 권 하나는 300–500쪽을 예상한다. 장 번호는 전체에서 이어진다.
- 아래에서 괄호 안은 그 장이 구현까지 설명할 API이고, "←"는 재료가 되는 옛 장 번호이다.

## 권 구성 (14권, 149장)

| 권 | 제목 | 다루는 영역 (공개 이름 수) |
|---|---|---|
| I | Mathematical Preliminaries | 책 전체가 쓰는 수학 |
| II | Computer Arithmetic and Systems | 부동소수점, 기계, 병렬, 기본 알고리즘 |
| III | Arrays | 최상위 배열 API (171) |
| IV | Data Frames | sd.frame (Series·DataFrame의 메서드 포함) |
| V | Dense Linear Algebra | sd.linalg (41) |
| VI | Sparse and Iterative Methods | sd.sparse (41) |
| VII | Approximation, Calculus, and Transforms | sd.interpolate (21), differentiate (4), integrate (16), fft (35), 스칼라 근 |
| VIII | Special Functions | sd.special (89) |
| IX | Optimization | sd.optimize (14) |
| X | Probability and Distributions | sd.stats (53), sd.random |
| XI | Statistical Inference | sd.htest (71) |
| XII | Statistical Models | sd.regression (50), sd.smooth (29), sd.cluster |
| XIII | Geometry and Graphs | sd.spatial (23), sd.graph (31) |
| XIV | Graphics | sd.plot (39) |

## Volume I — Mathematical Preliminaries

1. Notation and Order ← 1
2. Integers and Modular Arithmetic ← 2
3. Asymptotic Analysis ← 3
4. Matrix Analysis ← 4
5. Complex Analysis ← 5
6. Differential Equations and Recurrences ← 6
7. Combinatorics and Generating Functions ← 7
8. Convexity and Optimality (볼록 집합·함수, 최적성 조건; 옛 42장의 이론 부분을 앞으로) ← 42
9. Measure-Theoretic Probability (확률 공간, 조건부 기댓값, 수렴; 옛 49·51장의 이론 부분) ← 49, 51

## Volume II — Computer Arithmetic and Systems

10. Floating-Point Numbers ← 8
11. Rounding Error Analysis ← 9
12. Exact Sums and Products (pairwise, Kahan, 오차 없는 변환, double-double, 정확한 누산; 유사코드 전부) ← 10
13. Implementing Elementary Functions I: Exponentials and Logarithms (`exp expm1 log log1p log2 log10 pow`의 인자 축소·표·다항식) ← 11
14. Implementing Elementary Functions II: Trigonometric and Hyperbolic Functions (`sin cos tan asin … atanh hypot`, Payne–Hanek) ← 11
15. Decimal Conversion and Text (Eisel–Lemire, Ryū, 최단 왕복 출력, UTF-8) ← 12
16. Computer Architecture (파이프라인, 캐시, 분기 예측, SIMD/NEON, 가상 메모리) ← 13
17. Parallel Computation and Determinism (스레드풀, 결정적 리덕션, `set_num_threads`) ← 14
18. Sorting and Selection (introsort, 기수 정렬, 정렬 네트워크, quickselect, 이분 탐색) ← 15
19. Hash Tables, Heaps, and Union-Find ← 16
20. The Language Boundary (C++, CPython 객체 모델, 버퍼 프로토콜, DLPack, GIL) ← 18

## Volume III — Arrays

21. The Array Model (shape·strides·offset, 뷰와 소유, 메모리 순서, 0차원 규칙, `ndarray`의 속성과 프로토콜) ← 17
22. Data Types and Promotion (`bool int64 float64 complex128 string`, `result_type can_cast isdtype finfo iinfo astype`, 변환 규칙)
23. Array Creation (`asarray zeros ones empty full *_like arange linspace eye diag meshgrid tril triu from_dlpack`, list 추론, 버퍼 가져오기)
24. Shape Manipulation and Views (`reshape ravel permute_dims moveaxis squeeze expand_dims concat stack block flip roll repeat tile unstack`, 복사가 필요한 때의 판정)
25. Indexing (기본·고급 인덱스, 대입, 메모리 겹침, `take take_along_axis nonzero where`)
26. Broadcasting and the Elementwise Engine (브로드캐스트 규칙, nd 이터레이터, 안쪽 루프의 선택, `out=`, SIMD와 스레드 분할)
27. Elementwise Arithmetic (`add … pow`, 정수 나눗셈·나머지·시프트의 의미론, 복소 연산, 비교·논리·비트, `clip copysign nextafter` 등)
28. Elementwise Mathematical Functions (13–14장의 스칼라 커널을 배열에 싣기: 벡터화, 정확도 계약, 특수값)
29. Reductions (`sum prod mean var std min max argmin argmax any all`, 축 루프, pairwise 합, 계산 순서와 결정성, `nan*`, `cumulative_*`)
30. Order Statistics and Counting (`quantile median nanquantile nanmedian`, 선택 알고리즘, 아홉 가지 분위수 정의, `bincount histogram cov corrcoef`)
31. Sorting, Searching, and Sets (`sort argsort searchsorted unique_* count_nonzero interp diff`, 안정 정렬, NaN 순서)
32. Array Products (`matmul tensordot vecdot`, BLAS로 넘기기, 축 재배열, 작은 행렬 커널)
33. Printing and Interchange (`repr`·`str`의 요약과 정렬, 버퍼 프로토콜 내보내기, 연산자와 Python 프로토콜)

## Volume IV — Data Frames

34. The Columnar Model (열 버퍼, 유효 비트맵, dtype, `Series DataFrame`의 구조, Copy-on-Write)
35. Missing Values (NaN과 `NA`의 의미론, 승격, `isna notna fillna dropna`, 정수 연산)
36. Construction and Type Inference (생성자, list·배열에서 만들기, `to_array to_numpy tolist`)
37. Indexes and Hashing (`Index RangeIndex`, 라벨 해시 테이블, 중복 라벨, `reindex`)
38. Selection, Alignment, and Assignment (`[] .loc .iloc` 마스크, 라벨 정렬, 대입)
39. Elementwise Operations (산술·비교·논리, `astype`, 문자열 열)
40. Reductions and Descriptive Methods (dtype별 결과, NA 처리, 계산 순서)
41. Uniqueness and Counting (`unique duplicated drop_duplicates value_counts`, 해시와 정렬 경로)
42. Group-By (그룹 키 인코딩, 해시 집계, `agg`, 그룹 순서, 병렬 집계)
43. Sorting Tables (`sort_values sort_index`, 다중 키, 기수·비교 정렬의 선택, NA 위치)
44. Joins and Concatenation (`merge`의 해시 조인과 정렬 병합 조인, 다대다, `concat`)
45. Reading CSV (토크나이저 상태 기계, 따옴표, 타입 추론, 빠른 수 해석, 청크 병렬) ← 12
46. Table Display (`repr`, 열 폭, 잘라 보이기)

## Volume V — Dense Linear Algebra

47. Basic Operations, Norms, and BLAS (`vecdot outer cross kron kronsum trace diagonal vector_norm matrix_norm`, BLAS 층, 블록화)
48. Gaussian Elimination and LU (`lu lu_solve solve det slogdet inv`, 부분 피벗, 성장 인자, 블록·재귀 LU, 반복 개선) ← 19
49. Triangular, Banded, and Tridiagonal Systems (`solve_triangular solve_banded solve_tridiagonal bandwidth block_diag`)
50. Symmetric Factorizations (`cholesky cholesky_solve ldl`, Bunch–Kaufman) ← 20
51. Orthogonal Transformations and QR (`qr qr_pivoted`, Householder, compact WY, 열 피벗) ← 21
52. Least Squares and Rank (`lstsq pinv matrix_rank`, COD, 섭동 이론) ← 22
53. Condition Estimation (`cond`, 1-노름 추정, 후진 오차의 계산)
54. The Symmetric Eigenvalue Problem (`eigh eigvalsh`, 삼중대각화, 암시적 QR, 분할 정복, MRRR) ← 23
55. The Nonsymmetric Eigenvalue Problem (`eig eigvals schur hessenberg`, 균형, 다중 이동 QR, 고유벡터) ← 23
56. The Singular Value Decomposition (`svd svdvals`, 이중대각화, dqds, 분할 정복) ← 24
57. Matrix Functions and Matrix Equations (`expm matrix_power solve_sylvester`) ← 25

## Volume VI — Sparse and Iterative Methods

58. Sparse Storage Formats (`coo_array csr_array csc_array dok_array`, 표준 꼴, 변환) ← 26
59. Sparse Construction and Indexing (`eye_array diags_array random_array kron hstack vstack block_array tril triu find`, 인덱싱과 대입)
60. Sparse Arithmetic (행렬–벡터 곱, 희소–희소 곱, 원소별 연산, 리덕션, 합의 순서)
61. Sparse Direct Methods (`spsolve splu factorized inv spsolve_triangular`, 채움, 순서 매기기, 기호 분해, supernode) ← 26
62. Linear Operators and Krylov Subspaces (`LinearOperator aslinearoperator`, Arnoldi·Lanczos) ← 27
63. Methods for Symmetric Systems (`cg minres`, 전처리) ← 27
64. Methods for Nonsymmetric Systems (`gmres bicgstab`) ← 27
65. Iterative Least Squares (`lsqr lsmr`) ← 27
66. Large Eigenvalue and Singular Value Problems (`eigsh eigs svds`, 암시적 재시작, 이동–역) ← 28
67. Sparse Matrix Functions and Norms (`expm expm_multiply onenormest norm`)

## Volume VII — Approximation, Calculus, and Transforms

68. Polynomial Interpolation (`BarycentricInterpolator KroghInterpolator`) ← 29
69. Rational Approximation (`FloaterHormannInterpolator AAA`) ← 30
70. Piecewise Polynomials (`PPoly CubicHermiteSpline CubicSpline PchipInterpolator Akima1DInterpolator`, `sd.interp`) ← 31
71. B-Splines (`BSpline make_interp_spline`, de Boor, 띠 행렬 풀이) ← 31
72. Interpolation on Grids (`RegularGridInterpolator interpn`) ← 32
73. Scattered Data Interpolation (`RBFInterpolator NearestNDInterpolator LinearNDInterpolator CloughTocher2DInterpolator griddata`) ← 32
74. Numerical Differentiation (`derivative jacobian hessian approx_fprime check_grad`) ← 33
75. Scalar Root Finding (`root_scalar find_root bracket_root`, Brent, TOMS 748, 배치) ← 34
76. Quadrature Rules (`fixed_quad newton_cotes trapezoid simpson romb cumulative_*`, Gauss·Kronrod 규칙의 계산) ← 35
77. Adaptive Quadrature (`quad quad_vec dblquad tplquad nquad`, QUADPACK의 알고리즘 하나하나) ← 36
78. Double Exponential Quadrature and Series (`tanhsinh nsum`) ← 36
79. Multidimensional Integration (`cubature`) ← 36
80. Fourier Analysis ← 37
81. The Fast Fourier Transform (`fft ifft fft2 fftn`과 도우미, Stockham, 혼합 기수, Bluestein, 계획 캐시) ← 38
82. Real, Hermitian, and Trigonometric Transforms (`rfft irfft hfft dct dst`와 n차원 꼴) ← 38
83. The Fast Hankel Transform (`fht ifht fhtoffset`) ← 38

## Volume VIII — Special Functions

84. Principles of Special Function Computation (영역 나누기, 급수·연분수·점근 전개의 선택, 정확도 계약과 Arb 검증)
85. Error Functions and the Normal Distribution (`erf erfc erfcx erfinv erfcinv ndtr log_ndtr ndtri ndtri_exp wofz`) ← 39
86. The Gamma Function and Its Relatives (`gamma gammaln loggamma rgamma digamma polygamma beta betaln binom`) ← 39
87. Incomplete Gamma Functions (`gammainc gammaincc`와 역함수, Temme) ← 40
88. Incomplete Beta Functions (`betainc betaincc`와 역함수) ← 40
89. Logistic, Entropy, and Log-Sum-Exp Functions (`expit logit log_expit xlogy xlog1py entr rel_entr logsumexp softmax log_softmax`)
90. Zeta Function and Exponential Integrals (`zeta expi expn`) ← 41
91. Bessel Functions of Integer Order (`j0 j1 y0 y1 i0 i1 k0 k1`와 지수 척도 꼴) ← 41
92. Bessel Functions of Real Order and Airy Functions (`jv yv iv kv`와 `*e`, `airy`) ← 41
93. Elliptic Integrals, Hypergeometric Functions, and Lambert W (`ellipk ellipe hyp1f1 hyp2f1 lambertw`) ← 41
94. Distribution Functions (`stdtr chdtr fdtr pdtr bdtr nbdtr`와 역함수, 비중심 `chndtr nctdtr ncfdtr`)

## Volume IX — Optimization

95. Minimization in One Variable (`minimize_scalar find_minimum bracket_minimum`) ← 43
96. Line Search Methods (Wolfe 조건, Moré–Thuente, BFGS) ← 44
97. Limited-Memory and Bound-Constrained Methods (L-BFGS-B, Newton-CG, `minimize`의 방법들) ← 44
98. Trust-Region Methods ← 45
99. Derivative-Free Optimization (Nelder–Mead) ← 46
100. Nonlinear Least Squares (`least_squares`: LM, TRF, dogbox) ← 47
101. Linear Least Squares with Bounds (`lsq_linear nnls`, BVLS) ← 48
102. Systems of Nonlinear Equations (`root`: Powell 혼성법, Broyden, LM) ← 34
103. Linear Programming and Duality (KKT, 쌍대성, 심플렉스, 내부점법) ← 48

## Volume X — Probability and Distributions

104. Random Number Generation (`default_rng Generator`, Philox, 균등 정수와 실수, 섞기, 스트림과 재현성) ← 53
105. Sampling from Distributions (역변환, 기각, ziggurat, 분포별 표본 알고리즘) ← 53
106. The Distribution Framework (`ContinuousDistribution DiscreteDistribution`의 메서드: `pdf cdf ccdf icdf` 등, 꼬리의 정확도, 배열 호출)
107. Continuous Distributions (스물 남짓한 연속 분포 각각의 계산식과 극단 인자) ← 50
108. Discrete Distributions (이산 분포 각각) ← 50
109. Noncentral Distributions ← 50
110. Multivariate Distributions (`MultivariateNormal MultivariateT Multinomial Dirichlet Wishart InverseWishart MatrixNormal`, 다변량 확률의 계산) ← 52
111. Descriptive Statistics (`describe skew kurtosis moment zscore rankdata sem iqr mode entropy differential_entropy`)

## Volume XI — Statistical Inference

112. Estimation ← 54
113. Tests and Confidence Intervals ← 55
114. Tests for Means and Variances (`ttest_* f_oneway alexandergovern bartlett levene fligner`) ← 56
115. Goodness of Fit for a Specified Distribution (`chisquare power_divergence skewtest kurtosistest normaltest jarque_bera shapiro anderson ks_1samp cramervonmises goodness_of_fit`) ← 56, 57
116. Rank Tests (`mannwhitneyu ranksums wilcoxon kruskal friedmanchisquare brunnermunzel ansari mood median_test page_trend_test`, 정확 분포의 동적 계획) ← 57
117. Comparing Samples (`ks_2samp cramervonmises_2samp anderson_ksamp epps_singleton_2samp bws_test`) ← 57
118. Correlation and Association (`pearsonr spearmanr kendalltau weightedtau somersd chatterjeexi pointbiserialr multiscale_graphcorr`)
119. Contingency Tables (`chi2_contingency fisher_exact barnard_exact boschloo_exact mcnemar cochrans_q odds_ratio relative_risk`) ← 58
120. Proportions and Counts (`binomtest quantile_test poisson_means_test proportions_* proportion_confint *_2indep`) ← 58
121. Multiple Comparisons (`tukey_hsd dunnett multipletests false_discovery_control combine_pvalues`)
122. Resampling and Monte Carlo (`permutation_test monte_carlo_test bootstrap`과 방법 객체, BCa, 준몬테카를로)

## Volume XII — Statistical Models

123. Linear Models (`ols wls gls`, 추론, 샌드위치 공분산)
124. Model Diagnostics and Comparison (`anova`, 설정 검정, 영향력)
125. Generalized Linear Models (`glm`, IRLS, 분리)
126. Regularized Regression (`ridge lasso elastic_net *_cv penalized_glm`, 좌표 하강)
127. Robust Regression (`rlm theilslopes siegelslopes`)
128. Quantile Regression (`quantreg`)
129. Nonlinear Regression and Variable Selection (`nls stepwise best_subset`)
130. Scatterplot Smoothing (`lowess loess savgol_*`)
131. Smoothing Splines (`make_lsq_spline make_smoothing_spline make_splrep smoothing_spline pspline`)
132. Kernel Methods (`kernel_regression kde`와 대역폭 선택 `bw_*`)
133. Hierarchical and Partitioning Clustering
134. Density-Based Clustering and Mixture Models

## Volume XIII — Geometry and Graphs

135. Distances (`pdist cdist squareform`과 거리 함수들)
136. Nearest-Neighbor Search (`KDTree kneighbors radius_neighbors`)
137. Geometric Predicates and Convex Hulls (`ConvexHull`, 정확한 산술, 기호 섭동)
138. Delaunay Triangulations and Voronoi Diagrams (`Delaunay Voronoi`)
139. Graph Representation and Traversal (`connected_components breadth_first_* depth_first_* topological_sort articulation_points bridges biconnected_components core_number`)
140. Shortest Paths (`shortest_path dijkstra bellman_ford johnson floyd_warshall yen`)
141. Trees, Flows, and Matchings (`minimum_spanning_tree maximum_flow maximum_bipartite_matching min_weight_full_bipartite_matching linear_sum_assignment`)
142. Spectral and Ordering Methods (`laplacian pagerank betweenness_centrality reverse_cuthill_mckee structural_rank`)

## Volume XIV — Graphics

143. Rasterization and Compositing (안티에일리어싱, 반투명 합성)
144. Figures, Axes, and Scales (`Figure`, 눈금 고르기, 배치)
145. Text and Mathematical Typesetting
146. Curves and Adaptive Sampling (`line function parametric implicit contour`)
147. Statistical Graphics (`histogram density boxplot qq ecdf …`)
148. Three-Dimensional Graphics (`surface scatter3d curve3d`, 투영, 조명, 깊이)
149. File Formats (PNG·PDF·SVG, 압축)

## 진행 순서 (제안)

1. 이 목차를 확정한다(권 수, 장 나눔, 빠진 API).
2. 시범 장 하나를 새 기준으로 쓴다. 후보: 48장 Gaussian Elimination and LU(옛 19장) 또는 29장 Reductions(새 장, 구현 축이 가장 잘 드러남).
3. 시범 장으로 분량·깊이·그림·유사코드의 기준을 맞춘 뒤 권 단위로 진행한다. 옛 원고는 새 장의 재료로 쓰고, 새 장이 끝나면 옛 장 파일을 대체한다.
