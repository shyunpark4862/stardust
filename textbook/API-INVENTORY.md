# stardust 공개 API 목록 (공용 빌드 기준)

- 기준: 커밋 `66fe52e` (2026-11-25), 공용 빌드 `.sdbuild/stardust` (BUILD.md). 작업 저장소의 커밋하지 않은 변경은 들어 있지 않다.
- 뽑은 방법: 공용 빌드의 인터프리터로 `tools/api_inventory_md.py`를 돌려 각 이름공간의 `__all__`(모든 영역에 정의되어 있다)과 클래스의 공개 멤버를 읽었다. 클래스 줄의 `이름*`은 속성·필드, 나머지는 메서드다. 함수가 돌려주지만 `__all__`에 없는 결과 클래스(namedtuple과 결과 객체)는 영역마다 따로 적었다. `sd.plot`의 내부 클래스(`_scene`, `_layers` 등)와 밑줄로 시작하는 이름은 뺐다.
- 결과 객체 일부(`sd.smooth`의 결과, `KDEResult`)는 필드를 실행 중에 찾으므로 클래스에서 보이지 않는다. 그런 클래스는 docstring의 필드 목록을 함께 적었다.
- 다시 뽑기: `.sdbuild/stardust/.venv/bin/python -I tools/api_inventory_md.py` (JSON은 `tools/api_inventory.py`).

## 영역의 상태

판정 근거: ROADMAP.md의 마일스톤 표(가장 최신), docs/api/*.md 머리의 상태 줄, README.md의 영역 표, 실제 import. README의 영역 표와 일부 API 문서의 상태 줄은 이 커밋에서 낡아 있다(아래 "비고"). 이 목록은 ROADMAP과 실제로 공개된 이름을 따랐다.

| 영역 | 이름공간 | 상태 | `__all__` | 그 밖의 결과 클래스 | 비고 |
|---|---|---|---:|---:|---|
| 배열 | `sd` | 완료 | 171 | 6 | M0–M9 완료 |
| 선형대수 | `sd.linalg` | 완료 | 41 | 12 | M2·M3·M9 |
| 데이터프레임 | `sd.frame` | 완료 | 8 (클래스 4) | 2 | `GroupBy`, `NAType` |
| 특수함수 | `sd.special` | 완료 | 89 | 1 | S0–S11, API 문서 "확정" |
| 통계(분포·기술통계) | `sd.stats` | 완료 | 53 (클래스 42) | 1 | T0–T10 완료·승인. API 문서는 "T10 구현" 표기. 검정은 `sd.htest`로 옮겨 감 |
| 난수 | `sd.random` | 완료 | 2 | 0 | T3 완료·승인. 분포별 표본은 `sd.stats`의 `.sample` |
| 회귀 | `sd.regression` | 완료 | 50 (클래스 31) | 0 | G0–G7b 완료·승인. `linregress`는 뺐다(G188) |
| 최적화 | `sd.optimize` | 완료 | 14 | 0 | O0–O6 완료·승인. API 문서 머리는 아직 "초안" 표기지만 "계획한 풀이 함수는 모두 공개" |
| 가설검정 | `sd.htest` | 완료 | 71 (클래스 3) | 64 | H0–H10 완료·승인. API 문서 머리는 "초안"(H9까지) 표기로 낡음 |
| 희소 | `sd.sparse` | 완료 | 41 (클래스 6) | 11 | SP0–SP8 완료·승인. README 영역 표의 "만드는 중"은 낡음, API 문서도 "모든 공개 이름이 나왔다" |
| 적분 | `sd.integrate` | 완료 | 16 | 0 | C0–C8 완료·승인. README의 "만드는 중: 지금은 결과 클래스만"은 낡음 |
| 미분 | `sd.differentiate` | 완료 | 4 | 0 | C2 완료·승인(README의 "승인 대기"는 낡음) |
| 공간 | `sd.spatial` | 완료 | 23 (클래스 4) | 2 | SPT0–SPT14 완료(SPT5 BallTree는 건너뜀, 첫 판에 없음) |
| 그래프 | `sd.graph` | 완료 | 31 | 9 | GR0–GR7 |
| 보간 | `sd.interpolate` | 완료 | 21 (클래스 15) | 0 | IP0–IP9 (흩어진 점 보간 IP9 포함) |
| 푸리에 변환 | `sd.fft` | 완료 | 35 | 0 | FT0–FT7 |
| 평활 | `sd.smooth` | 완료 | 29 (클래스 7) | 1 | SM0–SM8 |
| 그림 | `sd.plot` | 완료 | 39 (클래스 1: `Figure`, 메서드 60) | — | PL0–PL9 완료·승인 |
| 군집 | `sd.cluster` | **구현 전 — 책에서 제외** | 모듈 없음 | — | ROADMAP "설계 확정, 구현 시작 전"(CL0–CL8 모두 시작 전). 이 커밋에는 docs/api/cluster.md와 docs/design/cluster.md만 있고 `import stardust.cluster`는 실패한다. 작업 저장소에는 커밋하지 않은 `src/stardust/cluster/`가 있으나 공용 빌드에 들어 있지 않다 |
| 임베딩 | `sd.embed` | **계획만 — 제외** | 모듈 없음 | — | docs/design/cluster.md CL1–CL3에서 이름과 순서(cluster → embed → classify)만 정함 |
| 분류기 | `sd.classify` | **계획만 — 제외** | 모듈 없음 | — | 위와 같다 |

- 완료 17개 이름공간의 `__all__` 합계는 738개, `__all__` 밖의 결과 클래스는 109개다. 클래스의 메서드·속성은 따로 세지 않았다(아래 각 줄의 `[n]`).
- docs/api/*.md의 언어 표시 없는 코드 블록(API 표면 테스트가 읽는 블록)에 있는 모듈 수준 이름 가운데 `__all__`에 없는 것은 없었다. 간단한 줄 파싱으로 본 것이고, `htest.md`는 블록 꼴이 달라 이 방법으로 확인하지 못했다. 표면 테스트 자체는 돌리지 않았다.
- 최적화의 방법 이름(`method=`): `minimize` nelder-mead · bfgs · l-bfgs-b · newton-cg · trust-ncg · trust-exact, `root_scalar` brentq · toms748 · bisect · newton · halley · secant, `minimize_scalar` brent · golden · bounded, `least_squares` trf · lm, `lsq_linear` trf · bvls, `root` hybr · lm.

## 이름공간별 목록

### `sd` (최상위, 배열) — 171개

- dtypes and constants (10): `bool` `int64` `float64` `complex128` `string` `e` `pi` `inf` `nan` `newaxis`
- the array type (1): `ndarray`
- creation (17): `asarray` `zeros` `ones` `empty` `full` `zeros_like` `ones_like` `empty_like` `full_like` `arange` `linspace` `eye` `diag` `meshgrid` `tril` `triu` `from_dlpack`
- shape manipulation (17): `reshape` `ravel` `permute_dims` `matrix_transpose` `moveaxis` `squeeze` `expand_dims` `broadcast_to` `broadcast_arrays` `concat` `stack` `block` `flip` `roll` `repeat` `tile` `unstack`
- conversion and data types (6): `astype` `can_cast` `result_type` `isdtype` `finfo` `iinfo`
- elementwise (70): `add` `subtract` `multiply` `divide` `floor_divide` `remainder` `pow` `negative` `positive` `abs` `square` `sqrt` `sign` `exp` `expm1` `log` `log1p` `log2` `log10` `sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `floor` `ceil` `trunc` `round` `equal` `not_equal` `less` `less_equal` `greater` `greater_equal` `logical_and` `logical_or` `logical_xor` `logical_not` `bitwise_and` `bitwise_or` `bitwise_xor` `bitwise_invert` `bitwise_left_shift` `bitwise_right_shift` `maximum` `minimum` `isnan` `isinf` `isfinite` `real` `imag` `conj` `clip` `where` `copysign` `hypot` `logaddexp` `nextafter` `reciprocal` `signbit` `deg2rad` `rad2deg`
- reductions (24): `sum` `prod` `min` `max` `mean` `var` `std` `argmin` `argmax` `any` `all` `cumulative_sum` `cumulative_prod` `nansum` `nanprod` `nanmin` `nanmax` `nanmean` `nanvar` `nanstd` `nanargmin` `nanargmax` `nancumulative_sum` `nancumulative_prod`
- statistics (8): `quantile` `nanquantile` `median` `nanmedian` `cov` `corrcoef` `bincount` `histogram`
- indexing and searching (7): `take` `take_along_axis` `nonzero` `count_nonzero` `searchsorted` `interp` `diff`
- sorting, sets, linear algebra (9): `sort` `argsort` `unique_values` `unique_counts` `unique_inverse` `unique_all` `matmul` `tensordot` `vecdot`
- execution (2): `set_num_threads` `get_num_threads`

클래스:

- `ndarray` [25]: T* all any argmax argmin astype base* copy dtype* itemsize* mT* max mean min nbytes* ndim* prod reshape shape* size* std strides* sum tolist var

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `FinfoResult` (tuple) [6]: bits* eps* max* min* smallest_normal* dtype*
- `HistogramResult` (tuple) [2]: hist* bin_edges*
- `IinfoResult` (tuple) [4]: bits* max* min* dtype*
- `UniqueAllResult` (tuple) [4]: values* indices* inverse_indices* counts*
- `UniqueCountsResult` (tuple) [2]: values* counts*
- `UniqueInverseResult` (tuple) [2]: values* inverse_indices*

### `sd.linalg` — 41개

- products and norms (12): `vecdot` `outer` `cross` `kron` `kronsum` `trace` `diagonal` `vector_norm` `matrix_norm` `matrix_power` `block_diag` `bandwidth`
- LU and determinants (6): `lu` `lu_solve` `solve` `det` `slogdet` `inv`
- structured systems (3): `solve_triangular` `solve_banded` `solve_tridiagonal`
- symmetric factorizations (3): `cholesky` `cholesky_solve` `ldl`
- QR, least squares, rank (6): `qr` `qr_pivoted` `lstsq` `pinv` `matrix_rank` `cond`
- eigenvalues and SVD (8): `eigh` `eigvalsh` `eig` `eigvals` `schur` `hessenberg` `svd` `svdvals`
- matrix functions and equations (2): `expm` `solve_sylvester`
- exceptions (1): `LinAlgError`

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `BandwidthResult` (tuple) [2]: lower* upper*
- `EigResult` (tuple) [2]: eigenvalues* eigenvectors*
- `EighResult` (tuple) [2]: eigenvalues* eigenvectors*
- `HessenbergResult` (tuple) [2]: H* Q*
- `LDLResult` (tuple) [3]: L* D* perm*
- `LUResult` (tuple) [3]: L* U* perm*
- `LstsqResult` (tuple) [3]: solution* rank* singular_values*
- `QRPivotedResult` (tuple) [3]: Q* R* perm*
- `QRResult` (tuple) [2]: Q* R*
- `SVDResult` (tuple) [3]: U* S* Vh*
- `SchurResult` (tuple) [2]: T* Z*
- `SlogdetResult` (tuple) [2]: sign* logabsdet*

### `sd.frame` — 8개

- 값 (1): `NA`
- 함수 · merge (2): `concat` `merge`
- 함수 · io (1): `read_csv`

클래스 4개 (`이름*` = 속성·필드, 나머지는 메서드):

- `DataFrame` (Reductions) [32]: astype columns* count drop_duplicates dropna dtypes* duplicated fillna groupby head iloc* index* isna isnan loc* max mean merge min notna nunique reindex shape* sort_index sort_values std sum tail to_array to_numpy value_counts var
- `Index` [5]: dtype* equals is_unique* name* tolist
- `RangeIndex` [8]: dtype* equals is_unique* name* start* step* stop* tolist
- `Series` (Reductions) [32]: astype count drop_duplicates dropna dtype* duplicated fillna head iloc* index* isna isnan loc* max mean min name* notna nunique reindex shape* sort_index sort_values std sum tail to_array to_numpy tolist unique_values value_counts var

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `GroupBy` (Reductions) [11]: agg count first last max mean min size std sum var
- `NAType`: (공개 멤버 없음)

### `sd.special` — 89개

- logistic and information (10): `expit` `logit` `log_expit` `xlogy` `xlog1py` `entr` `rel_entr` `logsumexp` `softmax` `log_softmax`
- error functions (10): `erf` `erfc` `erfcx` `erfinv` `erfcinv` `ndtr` `log_ndtr` `ndtri` `ndtri_exp` `wofz`
- gamma, beta, zeta (10): `gamma` `gammaln` `rgamma` `loggamma` `digamma` `polygamma` `beta` `betaln` `binom` `zeta`
- incomplete gamma and beta (8): `gammainc` `gammaincc` `gammaincinv` `gammainccinv` `betainc` `betaincc` `betaincinv` `betainccinv`
- exponential integrals, elliptic (4): `expi` `expn` `ellipk` `ellipe`
- Bessel (integer order) (12): `j0` `j1` `y0` `y1` `i0` `i0e` `i1` `i1e` `k0` `k0e` `k1` `k1e`
- Bessel (real order) and Airy (9): `jv` `yv` `iv` `kv` `jve` `yve` `ive` `kve` `airy`
- hypergeometric, Lambert W (3): `hyp1f1` `hyp2f1` `lambertw`
- distribution functions (14): `stdtr` `stdtrit` `chdtr` `chdtrc` `chdtri` `fdtr` `fdtrc` `fdtri` `pdtr` `pdtrc` `bdtr` `bdtrc` `nbdtr` `nbdtrc`
- noncentral distribution functions (9): `chndtr` `chndtrc` `chndtrix` `nctdtr` `nctdtrc` `nctdtrit` `ncfdtr` `ncfdtrc` `ncfdtri`

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `AiryResult` (tuple) [4]: ai* aip* bi* bip*

### `sd.stats` — 53개

- 함수 · descriptive (11): `describe` `skew` `kurtosis` `moment` `zscore` `rankdata` `sem` `iqr` `mode` `entropy` `differential_entropy`

클래스 42개 (`이름*` = 속성·필드, 나머지는 메서드):

- `ContinuousDistribution` [20]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample skewness standard_deviation support variance
- `DiscreteDistribution` [20]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode pmf sample skewness standard_deviation support variance
- `Normal` (ContinuousDistribution) [22]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode mu* pdf sample sigma* skewness standard_deviation support variance
- `StudentT` (ContinuousDistribution) [21]: ccdf cdf df* entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample skewness standard_deviation support variance
- `ChiSquared` (ContinuousDistribution) [21]: ccdf cdf df* entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample skewness standard_deviation support variance
- `F` (ContinuousDistribution) [22]: ccdf cdf dfd* dfn* entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample skewness standard_deviation support variance
- `Gamma` (ContinuousDistribution) [22]: a* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample scale* skewness standard_deviation support variance
- `Beta` (ContinuousDistribution) [22]: a* b* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample skewness standard_deviation support variance
- `Exponential` (ContinuousDistribution) [21]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample scale* skewness standard_deviation support variance
- `LogNormal` (ContinuousDistribution) [22]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode mu* pdf sample sigma* skewness standard_deviation support variance
- `Uniform` (ContinuousDistribution) [22]: a* b* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample skewness standard_deviation support variance
- `Cauchy` (ContinuousDistribution) [22]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode mu* pdf sample sigma* skewness standard_deviation support variance
- `Laplace` (ContinuousDistribution) [22]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode mu* pdf sample sigma* skewness standard_deviation support variance
- `Weibull` (ContinuousDistribution) [22]: c* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample scale* skewness standard_deviation support variance
- `Logistic` (ContinuousDistribution) [22]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode mu* pdf sample sigma* skewness standard_deviation support variance
- `Gumbel` (ContinuousDistribution) [22]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode mu* pdf sample sigma* skewness standard_deviation support variance
- `GEV` (ContinuousDistribution) [23]: c* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode mu* pdf sample sigma* skewness standard_deviation support variance
- `Pareto` (ContinuousDistribution) [22]: b* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample scale* skewness standard_deviation support variance
- `Rayleigh` (ContinuousDistribution) [21]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample scale* skewness standard_deviation support variance
- `InverseGamma` (ContinuousDistribution) [22]: a* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode pdf sample scale* skewness standard_deviation support variance
- `NoncentralChiSquared` (ContinuousDistribution) [22]: ccdf cdf df* entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode nc* pdf sample skewness standard_deviation support variance
- `NoncentralT` (ContinuousDistribution) [22]: ccdf cdf df* entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode nc* pdf sample skewness standard_deviation support variance
- `NoncentralF` (ContinuousDistribution) [23]: ccdf cdf dfd* dfn* entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpdf mean median mode nc* pdf sample skewness standard_deviation support variance
- `Binomial` (DiscreteDistribution) [22]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode n* p* pmf sample skewness standard_deviation support variance
- `Poisson` (DiscreteDistribution) [21]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode mu* pmf sample skewness standard_deviation support variance
- `Geometric` (DiscreteDistribution) [21]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode p* pmf sample skewness standard_deviation support variance
- `NegativeBinomial` (DiscreteDistribution) [22]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode n* p* pmf sample skewness standard_deviation support variance
- `Hypergeometric` (DiscreteDistribution) [23]: ccdf cdf draws* entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode pmf sample skewness standard_deviation successes* support total* variance
- `Bernoulli` (DiscreteDistribution) [21]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode p* pmf sample skewness standard_deviation support variance
- `DiscreteUniform` (DiscreteDistribution) [22]: ccdf cdf entropy high* iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf low* mean median mode pmf sample skewness standard_deviation support variance
- `BetaBinomial` (DiscreteDistribution) [23]: a* b* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode n* pmf sample skewness standard_deviation support variance
- `LogSeries` (DiscreteDistribution) [21]: ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode p* pmf sample skewness standard_deviation support variance
- `Zipf` (DiscreteDistribution) [21]: a* ccdf cdf entropy iccdf icdf ilogccdf ilogcdf kurtosis logccdf logcdf logpmf mean median mode pmf sample skewness standard_deviation support variance
- `MultivariateNormal` [9]: cdf cov* covariance entropy logpdf mean mu* pdf sample
- `MultivariateT` [10]: cdf covariance df* entropy logpdf mean mu* pdf sample shape*
- `Multinomial` [8]: covariance entropy logpmf mean n* p* pmf sample
- `Dirichlet` [7]: alpha* covariance entropy logpdf mean pdf sample
- `Wishart` [9]: df* entropy logpdf mean mode pdf sample scale* variance
- `InverseWishart` [9]: df* entropy logpdf mean mode pdf sample scale* variance
- `MatrixNormal` [8]: colcov* entropy logpdf mean mu* pdf rowcov* sample
- `DescribeResult` (tuple) [6]: nobs* minmax* mean* variance* skewness* kurtosis*
- `ModeResult` (tuple) [2]: mode* count*

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `CdfResult` (tuple) [2]: value* error*

### `sd.random` — 2개

- 함수 · random (1): `default_rng`

클래스 1개 (`이름*` = 속성·필드, 나머지는 메서드):

- `Generator` [6]: choice integers permutation random spawn state*

### `sd.regression` — 50개

- 함수 · linear (3): `ols` `wls` `gls`
- 함수 · glm (1): `glm`
- 함수 · penalized (7): `ridge` `lasso` `elastic_net` `ridge_cv` `lasso_cv` `elastic_net_cv` `penalized_glm`
- 함수 · robust (3): `rlm` `theilslopes` `siegelslopes`
- 함수 · quantile (1): `quantreg`
- 함수 · nls (1): `nls`
- 함수 · regression (1): `anova`
- 함수 · select (2): `stepwise` `best_subset`

클래스 31개 (`이름*` = 속성·필드, 나머지는 메서드):

- `OLSResult` [49]: aic* bic* breusch_pagan bse* centered_tss* compare_f_test compare_lr_test condition_number* conf_int cooks_distance* cov_params cov_type* df_model* df_resid* df_resid_inference* dfbetas* dffits* durbin_watson* ess* f_pvalue* fittedvalues* fvalue* goldfeld_quandt jarque_bera leverage* llf* nobs* param_names* params* predict predict_interval pvalues* rank* reset_test resid* resid_studentized_external* resid_studentized_internal* rsquared* rsquared_adj* scale* ssr* summary t_test tvalues* uncentered_tss* vif* wald_test white_test wresid*
- `WLSResult` [49]: aic* bic* breusch_pagan bse* centered_tss* compare_f_test compare_lr_test condition_number* conf_int cooks_distance* cov_params cov_type* df_model* df_resid* df_resid_inference* dfbetas* dffits* durbin_watson* ess* f_pvalue* fittedvalues* fvalue* goldfeld_quandt jarque_bera leverage* llf* nobs* param_names* params* predict predict_interval pvalues* rank* reset_test resid* resid_studentized_external* resid_studentized_internal* rsquared* rsquared_adj* scale* ssr* summary t_test tvalues* uncentered_tss* vif* wald_test white_test wresid*
- `GLSResult` [49]: aic* bic* breusch_pagan bse* centered_tss* compare_f_test compare_lr_test condition_number* conf_int cooks_distance* cov_params cov_type* df_model* df_resid* df_resid_inference* dfbetas* dffits* durbin_watson* ess* f_pvalue* fittedvalues* fvalue* goldfeld_quandt jarque_bera leverage* llf* nobs* param_names* params* predict predict_interval pvalues* rank* reset_test resid* resid_studentized_external* resid_studentized_internal* rsquared* rsquared_adj* scale* ssr* summary t_test tvalues* uncentered_tss* vif* wald_test white_test wresid*
- `GLMResult` [47]: aic* bic* bse* compare_f_test compare_lr_test condition_number* conf_int converged* cooks_distance* cov_params cov_type* deviance* df_model* df_resid* df_resid_inference* dfbetas* dffits* family* fittedvalues* leverage* linear_predictor* link* llf* llnull* llr* llr_pvalue* n_iter* nobs* null_deviance* param_names* params* pearson_chi2* predict predict_interval pseudo_rsquared pvalues* rank* resid_deviance* resid_pearson* resid_response* resid_working* scale* separation* summary t_test tvalues* wald_test
- `JarqueBeraResult` [3]: pvalue* statistic* summary
- `BreuschPaganResult` (BreuschPaganResult, tuple) [5]: lm* lm_pvalue* fvalue* f_pvalue* summary
- `RidgeResult` [13]: alpha* df_model* fittedvalues* l1_ratio* nobs* param_names* params* predict resid* rsquared* ssr* standardize* summary
- `LassoResult` [16]: alpha* converged* df_model* dual_gap* fittedvalues* l1_ratio* n_iter* nobs* param_names* params* predict resid* rsquared* ssr* standardize* summary
- `ElasticNetResult` [16]: alpha* converged* df_model* dual_gap* fittedvalues* l1_ratio* n_iter* nobs* param_names* params* predict resid* rsquared* ssr* standardize* summary
- `RidgeCVResult` (RidgeResult) [18]: alpha* alpha_1se* alpha_min* alphas* cv_error* cv_se* df_model* fittedvalues* l1_ratio* nobs* param_names* params* predict resid* rsquared* ssr* standardize* summary
- `LassoCVResult` (LassoResult) [21]: alpha* alpha_1se* alpha_min* alphas* converged* cv_error* cv_se* df_model* dual_gap* fittedvalues* l1_ratio* n_iter* nobs* param_names* params* predict resid* rsquared* ssr* standardize* summary
- `ElasticNetCVResult` (ElasticNetResult) [21]: alpha* alpha_1se* alpha_min* alphas* converged* cv_error* cv_se* df_model* dual_gap* fittedvalues* l1_ratio* n_iter* nobs* param_names* params* predict resid* rsquared* ssr* standardize* summary
- `PenalizedGLMResult` [20]: alpha* converged* deviance* deviance_ratio* df_model* dual_gap* family* fittedvalues* l1_ratio* linear_predictor* n_iter* nobs* null_deviance* param_names* params* predict resid_response* separation* standardize* summary
- `RLMResult` [27]: bse* condition_number* conf_int converged* cov_params cov_type* df_model* df_resid* df_resid_inference* fittedvalues* method* n_iter* nobs* norm* param_names* params* predict predict_interval pvalues* rank* resid* scale* summary t_test tvalues* wald_test weights*
- `TheilslopesResult` (TheilslopesResult, tuple) [5]: slope* intercept* low_slope* high_slope* summary
- `SiegelslopesResult` (SiegelslopesResult, tuple) [3]: slope* intercept* summary
- `QuantRegResult` [29]: bandwidth* bse* condition_number* conf_int converged* cov_params cov_type* df_model* df_resid* df_resid_inference* fittedvalues* n_iter* n_pivots* nobs* nonunique* objective* param_names* params* predict predict_interval prsquared* pvalues* q* rank* resid* summary t_test tvalues* wald_test
- `NLSResult` [31]: aic* bic* bse* condition_number* conf_int converged* cov_params cov_type* df_model* df_resid* df_resid_inference* fittedvalues* llf* method* n_iter* nfev* njev* nobs* param_names* params* predict predict_interval pvalues* rank* resid* scale* ssr* summary t_test tvalues* wald_test
- `TTestResult` [7]: conf_int df* effect* pvalue* sd* summary tvalue*
- `WaldTestResult` [7]: df* df_denom* df_num* distribution* pvalue* statistic* summary
- `CompareFTestResult` [5]: df_denom* df_num* fvalue* pvalue* summary
- `CompareLRTestResult` [4]: df* pvalue* statistic* summary
- `AnovaResult` [13]: df* df_diff* df_resid* mean_sq* pvalue* ss_diff* ssr* statistic* sum_sq* summary terms* test* typ*
- `WhiteTestResult` [8]: df* df_denom* df_num* f_pvalue* fvalue* lm* lm_pvalue* summary
- `GoldfeldQuandtResult` [6]: alternative* df_denom* df_num* fvalue* pvalue* summary
- `ResetTestResult` [5]: df_denom* df_num* fvalue* pvalue* summary
- `StepwiseResult` [8]: criterion* direction* fit* selected* selected_names* skipped* steps* summary
- `BestSubsetResult` [8]: criterion* fit* selected* selected_names* skipped* subsets* summary values*
- `SelectionStep` (tuple) [3]: action* term* criterion*
- `SkippedFit` (tuple) [4]: step* action* term* reason*
- `SkippedSubset` (tuple) [2]: columns* reason*

### `sd.optimize` — 14개

- 함수 · diff (2): `approx_fprime` `check_grad`
- 함수 · elementwise (4): `bracket_minimum` `bracket_root` `find_minimum` `find_root`
- 함수 · lsq (1): `least_squares`
- 함수 · linear (2): `lsq_linear` `nnls`
- 함수 · minimize (1): `minimize`
- 함수 · scalar (2): `minimize_scalar` `root_scalar`
- 함수 · root (1): `root`

클래스 1개 (`이름*` = 속성·필드, 나머지는 메서드):

- `OptimizeResult` [21]: active_mask* bracket* converged* cost* f_bracket* final_simplex* fun* grad* hess* hess_inv* jac* message* method* n_iter* nfev* nhev* njev* optimality* status* summary x*

### `sd.htest` — 71개

- 함수 · parametric (9): `ttest_1samp` `ttest_rel` `ttest_ind` `ttest_ind_from_stats` `f_oneway` `alexandergovern` `bartlett` `levene` `fligner`
- 함수 · gof (14): `chisquare` `power_divergence` `skewtest` `kurtosistest` `normaltest` `jarque_bera` `shapiro` `anderson` `ks_1samp` `cramervonmises` `ks_2samp` `cramervonmises_2samp` `anderson_ksamp` `epps_singleton_2samp`
- 함수 · rank (10): `mannwhitneyu` `ranksums` `wilcoxon` `kruskal` `friedmanchisquare` `brunnermunzel` `ansari` `mood` `median_test` `page_trend_test`
- 함수 · correlation (8): `pearsonr` `spearmanrho` `spearmanr` `pointbiserialr` `kendalltau` `weightedtau` `somersd` `chatterjeexi`
- 함수 · contingency (8): `chi2_contingency` `fisher_exact` `barnard_exact` `boschloo_exact` `mcnemar` `cochrans_q` `odds_ratio` `relative_risk`
- 함수 · proportion (8): `binomtest` `quantile_test` `poisson_means_test` `proportions_ztest` `proportions_chisquare` `proportion_confint` `test_proportions_2indep` `confint_proportions_2indep`
- 함수 · multicomp (5): `tukey_hsd` `dunnett` `multipletests` `false_discovery_control` `combine_pvalues`
- 함수 · resampling (6): `permutation_test` `monte_carlo_test` `bootstrap` `bws_test` `multiscale_graphcorr` `goodness_of_fit`

클래스 3개 (`이름*` = 속성·필드, 나머지는 메서드):

- `PermutationMethod` [3]: batch* n_resamples* rng*
- `MonteCarloMethod` [4]: batch* n_resamples* rng* rvs*
- `BootstrapMethod` [4]: batch* method* n_resamples* rng*

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `AlexandergovernResult` [4]: df* pvalue* statistic* summary
- `AndersonKsampResult` [3]: pvalue* statistic* summary
- `AndersonResult` [3]: pvalue* statistic* summary
- `AnsariResult` [4]: method* pvalue* statistic* summary
- `BarnardExactResult` [3]: pvalue* statistic* summary
- `BartlettResult` [4]: df* pvalue* statistic* summary
- `BinomtestResult` [7]: alternative* conf_int k* n* pvalue* statistic* summary
- `BootstrapResult` [4]: bootstrap_distribution* conf_int standard_error* summary
- `BoschlooExactResult` (BarnardExactResult) [3]: pvalue* statistic* summary
- `BrunnermunzelResult` (KruskalResult) [4]: df* pvalue* statistic* summary
- `BwsTestResult` (PermutationTestResult) [4]: null_distribution* pvalue* statistic* summary
- `ChatterjeexiResult` (SpearmanrhoResult) [3]: pvalue* statistic* summary
- `Chi2ContingencyResult` [5]: df* expected_freq* pvalue* statistic* summary
- `ChisquareResult` [4]: df* pvalue* statistic* summary
- `CochransQResult` [4]: df* pvalue* statistic* summary
- `CombinePvaluesResult` [3]: pvalue* statistic* summary
- `Cramervonmises2sampResult` [4]: method* pvalue* statistic* summary
- `CramervonmisesResult` [3]: pvalue* statistic* summary
- `DunnettResult` [5]: conf_int df* pvalue* statistic* summary
- `EppsSingleton2sampResult` [4]: df* pvalue* statistic* summary
- `FOnewayResult` [5]: df_denom* df_num* pvalue* statistic* summary
- `FisherExactResult` [3]: pvalue* statistic* summary
- `FlignerResult` [4]: df* pvalue* statistic* summary
- `FriedmanchisquareResult` (KruskalResult) [4]: df* pvalue* statistic* summary
- `GoodnessOfFitResult` [5]: fit_result* null_distribution* pvalue* statistic* summary
- `KendalltauResult` [4]: method* pvalue* statistic* summary
- `KruskalResult` [4]: df* pvalue* statistic* summary
- `Ks1sampResult` [6]: method* pvalue* statistic* statistic_location* statistic_sign* summary
- `Ks2sampResult` (Ks1sampResult) [6]: method* pvalue* statistic* statistic_location* statistic_sign* summary
- `KurtosistestResult` [3]: pvalue* statistic* summary
- `LeveneResult` [5]: df_denom* df_num* pvalue* statistic* summary
- `MannwhitneyuResult` [5]: method* pvalue* statistic* summary zstatistic*
- `McnemarResult` (BarnardExactResult) [3]: pvalue* statistic* summary
- `MedianTestResult` [6]: df* median* pvalue* statistic* summary table*
- `MonteCarloTestResult` (PermutationTestResult) [4]: null_distribution* pvalue* statistic* summary
- `MoodResult` (RanksumsResult) [3]: pvalue* statistic* summary
- `MultipletestsResult` [5]: alpha_bonferroni* alpha_sidak* pvalues_corrected* reject* summary
- `MultiscaleGraphcorrResult` [4]: mgc_dict* pvalue* statistic* summary
- `NormaltestResult` [3]: pvalue* statistic* summary
- `OddsRatioResult` [3]: conf_int statistic* summary
- `PageTrendTestResult` [4]: method* pvalue* statistic* summary
- `PearsonrResult` [4]: conf_int pvalue* statistic* summary
- `PermutationTestResult` [4]: null_distribution* pvalue* statistic* summary
- `PointbiserialrResult` (SpearmanrhoResult) [3]: pvalue* statistic* summary
- `PoissonMeansTestResult` [3]: pvalue* statistic* summary
- `PowerDivergenceResult` (ChisquareResult) [4]: df* pvalue* statistic* summary
- `ProportionsChisquareResult` [6]: df* expected_freq* pvalue* statistic* summary table*
- `ProportionsZtestResult` (PoissonMeansTestResult) [3]: pvalue* statistic* summary
- `QuantileTestResult` [5]: conf_int pvalue* statistic* statistic_type* summary
- `RanksumsResult` [3]: pvalue* statistic* summary
- `RelativeRiskResult` [7]: conf_int control_cases* control_total* exposed_cases* exposed_total* relative_risk* summary
- `ShapiroResult` [3]: pvalue* statistic* summary
- `SkewtestResult` [3]: pvalue* statistic* summary
- `SomersdResult` [4]: pvalue* statistic* summary table*
- `SpearmanrResult` (SpearmanrhoResult) [3]: pvalue* statistic* summary
- `SpearmanrhoResult` [3]: pvalue* statistic* summary
- `TestProportions2indepResult` [13]: alternative* compare* diff* method* odds_ratio* prop1_null* prop2_null* pvalue* ratio* statistic* summary value* variance*
- `Ttest1sampResult` [5]: conf_int df* pvalue* statistic* summary
- `TtestIndFromStatsResult` [5]: conf_int df* pvalue* statistic* summary
- `TtestIndResult` [5]: conf_int df* pvalue* statistic* summary
- `TtestRelResult` [5]: conf_int df* pvalue* statistic* summary
- `TukeyHsdResult` [5]: conf_int df* pvalue* statistic* summary
- `WeightedtauResult` (SpearmanrhoResult) [3]: pvalue* statistic* summary
- `WilcoxonResult` (MannwhitneyuResult) [5]: method* pvalue* statistic* summary zstatistic*

### `sd.sparse` — 41개

- 함수 · sparse (2): `issparse` `matrix_transpose`
- 함수 · construct (12): `eye_array` `diags_array` `random_array` `kron` `kronsum` `hstack` `vstack` `block_array` `block_diag` `tril` `triu` `find`
- 함수 · direct (7): `spsolve` `splu` `factorized` `inv` `spsolve_triangular` `is_sptriangular` `spbandwidth`
- 함수 · interface (1): `aslinearoperator`
- 함수 · iterative (6): `cg` `gmres` `minres` `bicgstab` `lsqr` `lsmr`
- 함수 · eigen (3): `eigsh` `eigs` `svds`
- 함수 · matfuncs (4): `expm` `expm_multiply` `onenormest` `norm`

클래스 6개 (`이름*` = 속성·필드, 나머지는 메서드):

- `sparray` [56]: T* argmax argmin asformat asin asinh astype atan atanh ceil conj copy count_nonzero deg2rad diagonal expm1 floor format* imag* log1p mT* max maximum mean min minimum multiply nanmax nanmin ndim* nonzero power rad2deg real* reshape resize round setdiag shape* sign sin sinh size* sqrt sum tan tanh to_scipy toarray tocoo tocsc tocsr todok trace transpose trunc
- `coo_array` (sparray) [63]: T* argmax argmin asformat asin asinh astype atan atanh ceil conj coords* copy count_nonzero data* deg2rad diagonal dtype* eliminate_zeros expm1 floor format* has_canonical_format* imag* log1p mT* max maximum mean min minimum multiply nanmax nanmin ndim* nnz* nonzero power rad2deg real* reshape resize round setdiag shape* sign sin sinh size* sqrt sum sum_duplicates tan tanh to_scipy toarray tocoo tocsc tocsr todok trace transpose trunc
- `csr_array` (sparray) [62]: T* argmax argmin asformat asin asinh astype atan atanh ceil conj copy count_nonzero data* deg2rad diagonal dtype* eliminate_zeros expm1 floor format* imag* indices* indptr* log1p mT* max maximum mean min minimum multiply nanmax nanmin ndim* nnz* nonzero power rad2deg real* reshape resize round setdiag shape* sign sin sinh size* sqrt sum tan tanh to_scipy toarray tocoo tocsc tocsr todok trace transpose trunc
- `csc_array` (sparray) [62]: T* argmax argmin asformat asin asinh astype atan atanh ceil conj copy count_nonzero data* deg2rad diagonal dtype* eliminate_zeros expm1 floor format* imag* indices* indptr* log1p mT* max maximum mean min minimum multiply nanmax nanmin ndim* nnz* nonzero power rad2deg real* reshape resize round setdiag shape* sign sin sinh size* sqrt sum tan tanh to_scipy toarray tocoo tocsc tocsr todok trace transpose trunc
- `dok_array` (sparray) [58]: T* argmax argmin asformat asin asinh astype atan atanh ceil conj copy count_nonzero deg2rad diagonal dtype* expm1 floor format* imag* log1p mT* max maximum mean min minimum multiply nanmax nanmin ndim* nnz* nonzero power rad2deg real* reshape resize round setdiag shape* sign sin sinh size* sqrt sum tan tanh to_scipy toarray tocoo tocsc tocsr todok trace transpose trunc
- `LinearOperator` [9]: H* T* adjoint dtype* matmat matvec rmatmat rmatvec shape*

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `BicgstabResult` [7]: converged* message* n_iter* normr* status* summary x*
- `CgResult` [7]: converged* message* n_iter* normr* status* summary x*
- `EigsResult` [9]: converged* eigenvalues* eigenvectors* message* n_converged* n_iter* n_op* status* summary
- `EigshResult` [9]: converged* eigenvalues* eigenvectors* message* n_converged* n_iter* n_op* status* summary
- `GmresResult` [8]: converged* message* n_inner* n_iter* normr* status* summary x*
- `LsmrResult` [11]: conda* converged* message* n_iter* norma* normar* normr* normx* status* summary x*
- `LsqrResult` [13]: conda* converged* message* n_iter* norma* normar* normr* normr_damped* normx* status* summary var* x*
- `MinresResult` [11]: conda* converged* message* n_iter* norma* normar* normr* normx* status* summary x*
- `OnenormestResult` [6]: est* n_iter* n_op* summary v* w*
- `SparseLU` [2]: shape* solve
- `SvdsResult` [10]: S* U* Vh* converged* message* n_converged* n_iter* n_op* status* summary

### `sd.integrate` — 16개

- 함수 · cubature (1): `cubature`
- 함수 · samples (5): `cumulative_simpson` `cumulative_trapezoid` `romb` `simpson` `trapezoid`
- 함수 · quadpack (4): `dblquad` `nquad` `quad` `tplquad`
- 함수 · rules (2): `fixed_quad` `newton_cotes`
- 함수 · tanhsinh (2): `nsum` `tanhsinh`
- 함수 · quad_vec (1): `quad_vec`

클래스 1개 (`이름*` = 속성·필드, 나머지는 메서드):

- `IntegrateResult` [24]: converged* cycle_errors* cycle_integrals* cycle_status* error* imag* integral* interval_errors* interval_integrals* intervals* max_level* message* n_cycles* n_intervals* n_regions* nfev* real* region_errors* region_integrals* regions* status* subdivisions* sum* summary

### `sd.differentiate` — 4개

- 함수 · derivative (3): `derivative` `hessian` `jacobian`

클래스 1개 (`이름*` = 속성·필드, 나머지는 메서드):

- `DifferentiateResult` [9]: converged* ddf* df* error* n_iter* nfev* status* summary x*

### `sd.spatial` — 23개

- 함수 · distance (17): `pdist` `cdist` `squareform` `braycurtis` `canberra` `chebyshev` `cityblock` `correlation` `cosine` `euclidean` `hamming` `haversine` `jaccard` `mahalanobis` `minkowski` `seuclidean` `sqeuclidean`
- 함수 · neighbors (2): `kneighbors` `radius_neighbors`

클래스 4개 (`이름*` = 속성·필드, 나머지는 메서드):

- `KDTree` [13]: count_neighbors data* leafsize* m* maxes* mins* n* query query_ball_point query_ball_tree query_pairs size* sparse_distance_matrix
- `Delaunay` [14]: convex_hull* coplanar* find_simplex max_bound* min_bound* ndim* neighbors* npoints* nsimplex* points* simplices* transform* vertex_neighbor_vertices* vertex_to_simplex*
- `ConvexHull` [12]: area* equations* max_bound* min_bound* ndim* neighbors* npoints* nsimplex* points* simplices* vertices* volume*
- `Voronoi` [10]: max_bound* min_bound* ndim* npoints* point_region* points* regions* ridge_points* ridge_vertices* vertices*

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `QueryResult` (tuple) [2]: distances* indices*
- `RadiusResult` [5]: distances* indices* indptr* shape* tocsr

### `sd.graph` — 31개

- 함수 · traversal (6): `connected_components` `breadth_first_order` `depth_first_order` `breadth_first_tree` `depth_first_tree` `topological_sort`
- 함수 · convert (2): `csgraph_from_dense` `csgraph_to_dense`
- 함수 · paths (8): `shortest_path` `dijkstra` `bellman_ford` `johnson` `floyd_warshall` `yen` `construct_dist_matrix` `reconstruct_path`
- 함수 · tree (1): `minimum_spanning_tree`
- 함수 · flow (1): `maximum_flow`
- 함수 · matching (3): `maximum_bipartite_matching` `min_weight_full_bipartite_matching` `linear_sum_assignment`
- 함수 · ordering (3): `reverse_cuthill_mckee` `structural_rank` `laplacian`
- 함수 · structure (4): `articulation_points` `bridges` `biconnected_components` `core_number`
- 함수 · centrality (2): `pagerank` `betweenness_centrality`

클래스 1개 (`이름*` = 속성·필드, 나머지는 메서드):

- `NegativeCycleError` (ValueError, Exception, BaseException): (공개 멤버 없음; A cycle of negative weight that a shortest-path search can reach from the nodes it starts at)

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `AssignmentResult` (tuple) [2]: row_ind* col_ind*
- `BiconnectedComponentsResult` (tuple) [3]: n_components* edges* labels*
- `ConnectedComponentsResult` (tuple) [2]: n_components* labels*
- `LaplacianResult` (tuple) [2]: laplacian* diag*
- `MaximumFlowResult` [2]: flow* flow_value*
- `MinOnlyResult` (tuple) [3]: dist_matrix* predecessors* sources*
- `ShortestPathResult` (tuple) [2]: dist_matrix* predecessors*
- `TraversalResult` (tuple) [2]: node_array* predecessors*
- `YenResult` (tuple) [2]: dist_array* predecessors*

### `sd.interpolate` — 21개

- 함수 · ppoly (1): `pchip_interpolate`
- 함수 · bspline (1): `make_interp_spline`
- 함수 · polyint (2): `barycentric_interpolate` `krogh_interpolate`
- 함수 · grid (1): `interpn`
- 함수 · scattered (1): `griddata`

클래스 15개 (`이름*` = 속성·필드, 나머지는 메서드):

- `PPoly` [11]: antiderivative axis* c* derivative extend extrapolate* from_spline* integrate roots solve x*
- `CubicHermiteSpline` (PPoly) [11]: antiderivative axis* c* derivative extend extrapolate* from_spline* integrate roots solve x*
- `CubicSpline` (PPoly) [11]: antiderivative axis* c* derivative extend extrapolate* from_spline* integrate roots solve x*
- `PchipInterpolator` (PPoly) [11]: antiderivative axis* c* derivative extend extrapolate* from_spline* integrate roots solve x*
- `Akima1DInterpolator` (PPoly) [11]: antiderivative axis* c* derivative extend extrapolate* from_spline* integrate roots solve x*
- `BSpline` [13]: antiderivative axis* basis_element* c* derivative design_matrix* extrapolate* from_power_basis* insert_knot integrate k* t* tck*
- `BarycentricInterpolator` [7]: add_xi derivative derivatives set_yi wi* xi* yi*
- `KroghInterpolator` [2]: derivative derivatives
- `FloaterHormannInterpolator` [4]: poles residues roots weights*
- `AAA` [9]: clean_up converged* errors* poles residues roots support_points* support_values* weights*
- `RegularGridInterpolator` [5]: bounds_error* fill_value* grid* method* values*
- `RBFInterpolator`: (공개 멤버 없음; The radial basis function interpolant of values d at the scattered points y.)
- `NearestNDInterpolator`: (공개 멤버 없음; The value of the nearest data point (ties to the smaller index).)
- `LinearNDInterpolator` [4]: fill_value* points* tri* values*
- `CloughTocher2DInterpolator` [7]: converged* fill_value* grad* niter* points* tri* values*

### `sd.fft` — 35개

- 함수 · basic (18): `fft` `ifft` `rfft` `irfft` `hfft` `ihfft` `fft2` `ifft2` `fftn` `ifftn` `rfft2` `irfft2` `rfftn` `irfftn` `hfft2` `ihfft2` `hfftn` `ihfftn`
- 함수 · realtransforms (8): `dct` `idct` `dst` `idst` `dctn` `idctn` `dstn` `idstn`
- 함수 · fftlog (3): `fht` `ifht` `fhtoffset`
- 함수 · helper (6): `fftfreq` `rfftfreq` `fftshift` `ifftshift` `next_fast_len` `prev_fast_len`

### `sd.smooth` — 29개

- 함수 · local (1): `lowess`
- 함수 · loess (1): `loess`
- 함수 · savgol (2): `savgol_coeffs` `savgol_filter`
- 함수 · spline (4): `make_lsq_spline` `make_smoothing_spline` `smoothing_spline` `pspline`
- 함수 · fitpack (3): `make_splrep` `make_splprep` `generate_knots`
- 함수 · kreg (2): `kernel_regression` `bw_dpill`
- 함수 · kde (1): `kde`
- 함수 · bandwidth (8): `bw_scott` `bw_silverman` `bw_nrd0` `bw_nrd` `bw_sj` `bw_isj` `bw_cv_ls` `bw_cv_ml`

클래스 7개 (`이름*` = 속성·필드, 나머지는 메서드):

- `LowessResult` (SmootherResult) [3]: predict predict_interval summary — 동적 필드(docstring): The result of lowess(): fittedvalues, resid, nobs, df, leverage, scale, frac, k, it, delta, robust_weights; predict(x_new) and predict_interval(x_new, alpha=0.05) evaluate the same local line at new points with the last robust weights. Not a public constructor (docs/api/smooth.md §7).
- `LoessResult` (SmootherResult) [3]: predict predict_interval summary — 동적 필드(docstring): The result of loess(): fittedvalues, resid, nobs, df, leverage, scale, span, q, degree, family, enp, one_delta, two_delta, robust_weights, n_rank_deficient; predict(X_new) and predict_interval(X_new, alpha=0.05) evaluate the same local polynomial at new points with the last robust weights. Not a pub
- `SmoothingSplineResult` (SmootherResult) [4]: predict predict_interval spline* summary — 동적 필드(docstring): The result of smoothing_spline(): fittedvalues, resid, nobs, df, leverage, scale, spline, lam, gcv, cv, criterion, status, converged; predict(x_new) is spline(x_new). Not a public constructor (docs/api/smooth.md §7).
- `PSplineResult` (SmootherResult) [4]: predict predict_interval spline* summary — 동적 필드(docstring): The result of pspline(): the fields of SmoothingSplineResult and n_segments, k, diff_order. Not a public constructor (docs/api/smooth.md §7).
- `KernelRegressionResult` (SmootherResult) [4]: gradient predict predict_interval summary — 동적 필드(docstring): The result of kernel_regression(): fittedvalues, resid, nobs, df, leverage, scale, bw, degree, kernel, criterion, criterion_value, status, converged; predict(X_new), predict_interval(X_new, alpha=0.05) and gradient(X_new) evaluate the same estimator at new points. Not a public constructor (docs/api/
- `KDEResult` [6]: cdf density logpdf marginal pdf sample — 동적 필드(docstring): The result of kde(): the fields bw, covariance, factor, neff, nobs, ndim, kernel, bounds, weights and the methods pdf, logpdf, cdf, sample, density and marginal. Not a public constructor (docs/api/smooth.md §7).
- `DensityResult` (tuple) [2]: x* density*

`__all__` 밖의 결과 클래스 (함수가 돌려준다):

- `SmootherResult` [1]: summary — 동적 필드(docstring): Base class: SmootherResult(**fields). Not a public constructor (docs/api/smooth.md §7).

### `sd.plot` — 39개

- 함수 · basic (35): `annotate` `area` `bar` `boxplot` `coefficients` `contour` `curve3d` `density` `distribution` `ecdf` `errorbar` `function` `grid` `heatmap` `hexbin` `hist2d` `histogram` `hline` `implicit` `line` `parametric` `parametric_surface` `phase` `polar` `qq` `region` `scatter` `scatter3d` `step` `streamlines` `strip` `surface` `text` `vectorfield` `vline`
- 함수 · stat (2): `diagnostics` `pairs`
- 함수 · facet (1): `facet`

클래스 1개 (`이름*` = 속성·필드, 나머지는 메서드):

- `Figure` [60]: annotate area axes* bar boxplot coefficients compare contour curve3d density distribution ecdf errorbar fit function heatmap height* hexbin hist2d histogram hline implicit info* inset legend line parametric parametric_surface phase polar qq region save scatter scatter3d scene shape* size step streamlines strip surface text title to_pdf to_png to_svg twin vectorfield vline width* xlabel xlim xscale xticks ylabel ylim yscale yticks zlabel

## 목차 대조

OUTLINE.md "주제 권 장 목차"(6–17권)의 장 줄에 백틱으로 적힌 이름(약어 `ttest_*` 등은 와일드카드로 풀었다)과 위 목록을 기계적으로 견준 뒤, 장 제목·설명이 분명히 다루는 것(예: 7.8의 "산술·비교·논리·비트", 11.11·11.17의 "와 역함수", "비중심 꼴", 14.5의 "Continuous Distributions")은 손으로 걸렀다. 완료 영역만 대상이다.

### (a) 어느 장도 분명히 다루지 않는 공개 API

표기: **[주제 없음]** = 그 내용 자체가 목차에 없다(절을 새로 넣거나 새 장이 필요). 표시 없는 줄 = 장의 주제에는 들어 있으나 장 줄의 API 목록에 이름이 없다(목록에 더하면 된다).

**7권 Arrays (`sd`)**
- 7.2: `string`(다섯째 dtype; 장 줄은 넷만 적음), `isdtype`, 상수 `e pi inf nan`, 결과 `FinfoResult IinfoResult`
- 7.3: `empty zeros_like ones_like empty_like full_like diag tril triu block`
- 7.4: `matrix_transpose moveaxis squeeze expand_dims unstack`
- 7.5: `newaxis`; 7.10: `count_nonzero`
- 7.6: `broadcast_to broadcast_arrays`
- 7.7: `get_num_threads`
- 7.8: 산술·비교·논리·비트 함수 전부(`add … bitwise_right_shift`, `maximum minimum isnan isinf isfinite real imag conj reciprocal signbit sign abs square`)와 반올림 `floor ceil trunc round`, `sqrt`(정확히 반올림되는 연산이므로 6권 또는 7.8), `deg2rad rad2deg`
- 7.9: `argmin argmax`, `nanargmin nanargmax`(`nan*`에 들어 있음)
- **[주제 없음]** `diff`(이웃 차분, n차): 7.9나 7.4에 절 추천
- 7.11: 결과 `UniqueAllResult UniqueCountsResult UniqueInverseResult`; 7.10: `HistogramResult`
- `ndarray`의 메서드·속성(`T mT base itemsize nbytes strides tolist copy` 등): 7.1
- **[주제 없음]** 11.5 또는 11.12: `logaddexp`(배열 영역의 로그 합; `logsumexp`와 짝)

**8권 Data Frames (`sd.frame`)**
- 8.12: `read_csv`(장 제목이 다룸, 이름만 없음)
- 8.7: `count nunique head tail`, `isnan notna`(8.2)
- 8.9: `GroupBy`의 `first last size count`
- 8.4: `Index.equals`, `is_unique`
- **[주제 없음]** `Series.to_array DataFrame.to_array to_numpy`(배열과의 경계, 복사 없는 읽기 전용 뷰와 F-order 행렬): 8.1 또는 7.14에 절 추천

**9권 Dense Linear Algebra (`sd.linalg`)**
- 9.3: `diagonal kronsum block_diag`
- 9.5: `bandwidth`(`BandwidthResult`)
- 9.4: `LinAlgError`(특이 행렬의 처리)

**10권 Sparse Matrices and Graphs**
- 10.1: `sparray issparse`, 형식 메서드(`tocsr tocsc tocoo todok asformat sum_duplicates eliminate_zeros has_canonical_format`)
- 10.2: `block_array block_diag kronsum tril triu find matrix_transpose`, 메서드 `setdiag diagonal trace resize reshape`
- 10.3: 원소별 메서드(`sin sqrt expm1 … power multiply maximum minimum`)와 리덕션 메서드(`sum mean max min argmax nanmax count_nonzero`)
- 10.4: `inv spsolve_triangular is_sptriangular spbandwidth`, 결과 `SparseLU`
- 10.5: `aslinearoperator`
- 10.6–10.8: 결과 `CgResult GmresResult MinresResult BicgstabResult LsqrResult LsmrResult EigshResult EigsResult SvdsResult`; 10.9: `OnenormestResult`
- 10.10: `csgraph_from_dense csgraph_to_dense`(변환 도구)
- 10.11: `construct_dist_matrix reconstruct_path NegativeCycleError`
- 10.12: `min_weight_full_bipartite_matching`
- 10.13: `biconnected_components`

**11권 Elementary and Special Functions (`sd.special`)**
- 11.9: `erfcinv ndtri_exp`
- 11.10: `rgamma betaln loggamma`
- 11.11: `gammaincinv gammainccinv betaincc betaincinv betainccinv`(장 줄의 "역함수"가 다룸)
- 11.12: `log_expit xlog1py rel_entr log_softmax`
- 11.14: 지수 배율 판 `i0e i1e k0e k1e`; 11.15: `jve yve ive kve`
- 11.17: `stdtrit chdtrc chdtri fdtrc fdtri pdtrc bdtrc`, **`nbdtr nbdtrc`(음이항; 장 줄의 다섯 계열에 없음)**, 비중심 `chndtr chndtrc chndtrix nctdtr nctdtrc nctdtrit ncfdtr ncfdtrc ncfdtri`
- **[주제 없음]** 복소수 인자: `erf erfc erfcx lambertw wofz loggamma`의 복소 판(README가 따로 적는 기능). 11.8이나 각 장에 복소 계산 절 추천
- `airy`의 결과 `AiryResult`: 11.15

**12권 Approximation and Numerical Calculus**
- 12.1: `barycentric_interpolate krogh_interpolate`
- 12.3: `CubicHermiteSpline pchip_interpolate`
- 12.4: `PPoly.from_spline`, `BSpline`의 `design_matrix insert_knot from_power_basis basis_element`
- 12.5: `interpn NearestNDInterpolator CloughTocher2DInterpolator`(뒤의 둘과 `LinearNDInterpolator griddata`는 17권의 `Delaunay`에 기대므로 순서 주의)
- 12.7: `cumulative_trapezoid cumulative_simpson`
- 12.12–12.13: 다차원·역변환 `ifft2 ifftn rfft2 irfft2 rfftn irfftn hfft2 ihfft2 hfftn ihfftn ihfft idct idst dctn idctn dstn idstn`
- 12.14: `fhtoffset`
- **[주제 없음]** 12.11 또는 12.12: 도우미 `fftfreq rfftfreq fftshift ifftshift next_fast_len prev_fast_len`(특히 `next_fast_len`은 혼합 기수 길이 고르기와 이어짐)
- **[주제 없음]** 최소제곱 스플라인과 매듭 고르기 `make_lsq_spline make_splrep make_splprep generate_knots`(FITPACK 계열): 12.4 또는 12.16에 절 추천
- 12.17: 대역폭 규칙 `bw_scott bw_silverman bw_nrd0 bw_nrd bw_sj bw_isj bw_cv_ls bw_cv_ml bw_dpill`("대역폭 선택"이 다룸), 결과 `KDEResult`(`marginal cdf sample`) `DensityResult`
- 12.15–12.17: 결과 `LowessResult LoessResult SmoothingSplineResult PSplineResult KernelRegressionResult`의 `predict_interval`

**13권 Optimization (`sd.optimize`)**
- **[주제 없음]** `approx_fprime check_grad`(고정 간격 유한 차분 기울기와 검사): 12.6과 묶거나 13.5 앞에 절 추천
- 13.5–13.8: `minimize`(장들이 방법을 다루지만 함수 이름이 어느 장 줄에도 없음; 13.5에 적기 추천)
- 공통: `OptimizeResult`

**14권 Random Numbers and Distributions**
- 14.2: `Generator`의 `spawn state choice permutation integers`
- 14.4: `ContinuousDistribution DiscreteDistribution`, 메서드 `logpdf logcdf logccdf ilogcdf ilogccdf iccdf support mode median entropy skewness kurtosis`
- 14.5·14.6·14.7: 분포 클래스 31개(장 제목이 다룸)
- 14.8: `MultivariateT Multinomial InverseWishart MatrixNormal`, `CdfResult`(`cdf(..., return_error=True)`; d ≤ 3 정규는 정확한 공식, 그 밖은 Genz의 준몬테카를로. **[주제 없음]** 다변량 cdf 계산의 절 추천)
- 14.9: `zscore sem iqr mode entropy differential_entropy`, 결과 `DescribeResult ModeResult`

**15권 Statistical Inference (`sd.htest`)**
- 15.2: `alexandergovern`
- 15.3: `power_divergence skewtest kurtosistest normaltest jarque_bera`
- 15.4: `ranksums brunnermunzel ansari mood median_test page_trend_test`
- 15.5: `cramervonmises_2samp bws_test`
- 15.6: `spearmanrho pointbiserialr weightedtau chatterjeexi multiscale_graphcorr`
- 15.7: `boschloo_exact cochrans_q odds_ratio relative_risk`
- 15.8: `quantile_test poisson_means_test test_proportions_2indep confint_proportions_2indep`
- 15.9: `false_discovery_control`
- 15.10: `PermutationMethod MonteCarloMethod BootstrapMethod goodness_of_fit`
- 결과 클래스 64개(`__all__` 밖)의 공통 꼴과 `summary`: 15.1이나 권 머리

**16권 Statistical Models (`sd.regression`)**
- 16.1: 결과 `OLSResult WLSResult GLSResult`, `predict_interval`, `cov_type`, `TTestResult`
- 16.2: 진단·명세 검정 `breusch_pagan white_test goldfeld_quandt reset_test jarque_bera durbin_watson vif`, 영향력 `leverage cooks_distance dfbetas dffits resid_studentized_*`, 모형 비교 `t_test wald_test compare_f_test compare_lr_test`, 결과 `AnovaResult WaldTestResult CompareFTestResult CompareLRTestResult WhiteTestResult GoldfeldQuandtResult ResetTestResult BreuschPaganResult JarqueBeraResult`
- 16.3: `GLMResult`(`separation pseudo_rsquared` 등)
- 16.4: `penalized_glm`(정규화 GLM; 장 줄에 없음), 결과 `RidgeResult … ElasticNetCVResult PenalizedGLMResult`
- 16.5: `siegelslopes`, 결과 `RLMResult TheilslopesResult SiegelslopesResult QuantRegResult`
- 16.6: 결과 `NLSResult StepwiseResult BestSubsetResult SelectionStep SkippedFit SkippedSubset`
- **[주제 없음]** 영역을 넘는 결과 객체와 `summary()`(회귀·검정·최적화·평활이 함께 쓰는 상자 꼴 렌더러, 읽기 전용 배열): 16.1이나 7.13에 절 추천

**17권 Geometry and Graphics**
- 17.1: 거리 함수 14개 `braycurtis canberra chebyshev cityblock correlation cosine euclidean hamming haversine jaccard mahalanobis minkowski seuclidean sqeuclidean`(장 제목이 다룸)
- 17.2: `KDTree`의 `query query_ball_point query_ball_tree query_pairs count_neighbors sparse_distance_matrix`, 결과 `QueryResult RadiusResult`
- 17.4: `Delaunay.find_simplex vertex_neighbor_vertices transform`
- 17.6: `Figure`의 꾸밈과 배치 `title xlabel ylabel zlabel xlim ylim xscale yscale xticks yticks legend inset twin grid facet`, `save to_pdf to_png to_svg`(17.11)
- 17.8: `polar region`
- 17.9: `heatmap hexbin hist2d strip distribution pairs`
- 17.10: `parametric_surface`
- **[주제 없음]** 기본 표시 `scatter bar errorbar area step hline vline text annotate`: 어느 장 줄에도 없다. 17.6을 늘리거나 "Basic Marks" 새 장 추천
- **[주제 없음]** 벡터장과 위상 그림 `phase vectorfield streamlines`: 17.8에 절 또는 새 장 추천
- **[주제 없음]** 모형·검정 결과의 그림 `coefficients diagnostics`, `Figure.fit Figure.compare`(회귀선, 검정 괄호): 17.9에 절 추천(16·15권과 이어짐)

### (b) 목차에 있으나 이 빌드에 없는 이름

| 장 | 목차의 이름 | 실제 | 현재 이름 |
|---|---|---|---|
| 8.8 | `unique` | `Series`·`DataFrame`에 `unique` 없음 | `Series.unique_values` (배열 영역의 Array API 이름과 맞춘 것) |
| 7.13 | `repr`·`str` | 공개 API 이름이 아니라 Python 프로토콜 | `ndarray.__repr__`·`__str__` (그대로 두어도 됨) |

- 그 밖의 백틱 이름은 모두 있다. `agg`(8.9)는 `__all__` 밖의 `GroupBy.agg`, `pdf cdf ccdf icdf`(14.4)는 분포 메서드, `sd.interp`(12.3)는 최상위에 있다.
- 장 설명의 방법 이름도 맞다: Nelder–Mead·BFGS·L-BFGS-B·Newton-CG·trust 방법(`minimize`), Philox(`Generator`), ziggurat(`csrc/stats/sample_dist.hpp`의 정규·지수 표본), Eisel–Lemire(`csrc/frame/ops/parse.hpp`).
- 참고: 6.7의 "최단 왕복 출력(Ryū)"은 stardust가 직접 구현하지 않고 `std::to_chars`(libc++)에 맡긴다(`csrc/frame/ops/cast.cpp`). 소스에 "Ryu"라는 말은 없다. 그 장은 libc++의 알고리즘을 설명해야 한다.

### (c) 빼야 할 장·권, 권이 없는 영역

- **16.7 Hierarchical Clustering, 16.8 Partitioning and Density-Based Clustering**: `sd.cluster`는 이 커밋에서 설계만 있고 구현 전(CL0 시작 전)이다. 빼야 한다.
- **16.9 Embedding and Classification**: `sd.embed`, `sd.classify`는 계획만 있다. 빼야 한다.
- 그러면 16권은 `sd.regression`만의 6장이 된다. 권 표의 제목 "Statistical Models and Learning"과 영역 칸의 `sd.cluster`, `(embed, classify)`, 장 수 "(9장+)"도 고쳐야 한다(예: "Statistical Models", 6장).
- 완료된 영역 가운데 권이 없는 것은 없다. 완료 17개 이름공간(`sd linalg frame special stats random regression optimize htest sparse integrate differentiate spatial graph interpolate fft smooth plot`)이 모두 6–17권 어딘가에 배정되어 있다.
- 다만 영역 안의 기능 가운데 장이 없는 것은 (a)의 **[주제 없음]** 줄이다: `optimize`의 유한 차분 기울기, 영역을 넘는 결과 객체와 `summary()`, 배열–프레임 경계(`to_array`), 다변량 cdf의 적분, FITPACK 계열 스플라인, FFT 도우미, 복소 특수함수, `plot`의 기본 표시·벡터장·모형 그림, `diff`, `logaddexp`.
