# stardust API

`import stardust as sd`. 이 문서는 라이브러리 전체의 공통 규칙과 영역별 API 문서의 색인이다.

- 영역 사이의 일관성이 가장 앞선 원칙이다 (DESIGN.md §1.4). 같은 개념은 어느 영역에서든 같은 이름, 같은 시그니처, 같은 동작이다. 기준은 Python Array API 관례다 (DESIGN.md §8.1).
- 절 번호(`§N`)는 코드 주석이 인용한다. 번호를 바꾸지 않는다.
- 언어 표시가 없는 코드 블록은 영역마다의 API 표면 테스트(`tests/<영역>/test_api_surface.py`)가 읽는다. 이 문서의 블록은 배열 영역의 테스트가 읽는다.

| 영역 | 이름공간 | 문서 |
|---|---|---|
| 배열 | `sd` | [docs/api/array.md](docs/api/array.md) |
| 선형대수 | `sd.linalg` | [docs/api/linalg.md](docs/api/linalg.md) |
| 데이터프레임 | `sd.frame` | [docs/api/frame.md](docs/api/frame.md) |
| 특수함수 | `sd.special` | [docs/api/special.md](docs/api/special.md) (로지스틱·정보, 오차함수, 감마·베타·제타, 불완전 감마·베타, 리덕션·지수적분·타원, Bessel·Airy, 초기하·Lambert W, 복소수 판) |
| 통계 | `sd.stats` | [docs/api/stats.md](docs/api/stats.md) (단변량·다변량·행렬 분포, 기술통계) |
| 회귀 | `sd.regression` | [docs/api/regression.md](docs/api/regression.md) (`ols wls gls glm ridge lasso elastic_net ridge_cv lasso_cv elastic_net_cv penalized_glm rlm theilslopes siegelslopes quantreg nls anova stepwise best_subset`, 결과 객체 `<모형>Result`, 선택의 결과 `StepwiseResult BestSubsetResult`와 그 원소의 형 `SelectionStep SkippedFit SkippedSubset`) |
| 난수 | `sd.random` | [docs/api/random.md](docs/api/random.md) (Philox4×64-10, `default_rng`, `random`·`integers`·`choice`·`permutation`·`spawn`·`state`) |
| 최적화 | `sd.optimize` | [docs/api/optimize.md](docs/api/optimize.md) (`approx_fprime check_grad least_squares lsq_linear nnls root_scalar minimize_scalar find_root bracket_root find_minimum bracket_minimum minimize root`, 결과 객체 `OptimizeResult`) |
| 가설검정 | `sd.htest` | [docs/api/htest.md](docs/api/htest.md) (초안: H1의 t 검정·분산분석·분산의 동질성, H2의 적합도 검정, H3의 순위 검정, H4의 표본끼리의 적합도 검정, H5의 상관 검정, H6의 분할표 검정, H7의 비율·이항 검정, H8의 다중 비교와 보정, H9의 재표본과 몬테카를로가 공개. scipy.stats의 검정 전부와 statsmodels의 다중 비교 보정·비율 검정, 검정마다의 결과 객체) |
| 희소 | `sd.sparse` | [docs/api/sparse.md](docs/api/sparse.md) (초안: 공개된 것은 형식과 변환, 연산·곱·리덕션, 인덱싱과 만들기, 직접 풀이와 `LinearOperator`, 반복법, 고유값과 특이값, 행렬 함수와 노름이다(모든 공개 이름). 희소 배열 `coo_array csr_array csc_array dok_array`와 희소 선형대수 `spsolve splu cg gmres eigsh svds expm norm LinearOperator` 등; 설계 [docs/design/sparse.md](docs/design/sparse.md)) |
| 적분 | `sd.integrate` | [docs/api/integrate.md](docs/api/integrate.md) (확정: 결과 클래스 `IntegrateResult`와 scipy.integrate의 적분 15개 `quad dblquad tplquad nquad quad_vec cubature tanhsinh nsum fixed_quad newton_cotes trapezoid simpson romb cumulative_trapezoid cumulative_simpson`가 공개되었다; 설계 [docs/design/integrate.md](docs/design/integrate.md)) |
| 미분 | `sd.differentiate` | [docs/api/differentiate.md](docs/api/differentiate.md) (확정: `derivative jacobian hessian`과 결과 클래스 `DifferentiateResult`가 공개되었다; 설계는 적분과 함께 [docs/design/integrate.md](docs/design/integrate.md)) |
| 공간 | `sd.spatial` | [docs/api/spatial.md](docs/api/spatial.md) (확정: 공개 이름 23개가 공개되었다. 거리 `pdist cdist squareform`과 거리 함수 열넷, 한 번에 끝나는 이웃 찾기 `kneighbors radius_neighbors`, `KDTree`, 계산 기하 `Delaunay ConvexHull Voronoi`(2·3차원, 정확한 술어); 설계 [docs/design/spatial.md](docs/design/spatial.md), 계산 기하 [docs/design/spatial_geometry.md](docs/design/spatial_geometry.md)) |
| 그래프 | `sd.graph` | [docs/api/graph.md](docs/api/graph.md) (확정: 공개 이름 31개가 공개되었다. scipy.sparse.csgraph 전부(순회와 연결 요소, 최단 경로 `shortest_path dijkstra bellman_ford johnson floyd_warshall yen`, `minimum_spanning_tree`, `maximum_flow`, 매칭, `reverse_cuthill_mckee structural_rank laplacian`, 변환 도구)와 `linear_sum_assignment`, scipy에 없는 `topological_sort articulation_points bridges biconnected_components core_number pagerank betweenness_centrality`; 설계 [docs/design/graph.md](docs/design/graph.md)) |
| 보간 | `sd.interpolate` | [docs/api/interpolate.md](docs/api/interpolate.md) (확정. scipy.interpolate의 보간: 조각 다항식 `PPoly CubicHermiteSpline CubicSpline PchipInterpolator Akima1DInterpolator pchip_interpolate`, B-스플라인 `BSpline make_interp_spline`, 다항식·유리식 `BarycentricInterpolator barycentric_interpolate KroghInterpolator krogh_interpolate FloaterHormannInterpolator AAA`, 격자 `RegularGridInterpolator interpn`, 흩어진 점 `RBFInterpolator NearestNDInterpolator`; numpy의 `interp`는 배열 영역의 `sd.interp`; 설계 [docs/design/interpolate.md](docs/design/interpolate.md)) |
| 푸리에 변환 | `sd.fft` | [docs/api/fft.md](docs/api/fft.md) (확정: 공개 이름 35개가 공개되었다. scipy.fft 전부(numpy.fft의 이름을 모두 담는다): 복소 `fft ifft fft2 ifft2 fftn ifftn`, 실수·에르미트 `rfft irfft hfft ihfft rfft2 irfft2 rfftn irfftn hfft2 ihfft2 hfftn ihfftn`, `dct idct dst idst dctn idctn dstn idstn`(형 1–4), FFTLog `fht ifht fhtoffset`, 도우미 `fftfreq rfftfreq fftshift ifftshift next_fast_len prev_fast_len`; 설계 [docs/design/fft.md](docs/design/fft.md)) |
| 평활 | `sd.smooth` | [docs/api/smooth.md](docs/api/smooth.md) (확정. 산점도 평활 `lowess loess`, 평활 스플라인 `make_lsq_spline make_smoothing_spline make_splrep make_splprep generate_knots smoothing_spline pspline`, 핵 회귀 `kernel_regression bw_dpill`, 핵밀도추정 `kde`와 대역폭 `bw_scott bw_silverman bw_nrd0 bw_nrd bw_sj bw_isj bw_cv_ls bw_cv_ml`, Savitzky–Golay `savgol_coeffs savgol_filter`, 결과 객체 `<이름>Result`; 설계 [docs/design/smooth.md](docs/design/smooth.md)) |
| 군집 | `sd.cluster` | [docs/api/cluster.md](docs/api/cluster.md) (확정. 계층 `linkage agglomerative divisive`와 scipy의 도구 `fcluster cut_tree cophenet leaves_list optimal_leaf_ordering dendrogram` 등, 분할 `kmeans minibatch_kmeans bisecting_kmeans kmedoids fuzzy_cmeans`, 밀도 `dbscan hdbscan optics`, 혼합 `gaussian_mixture`, 지표 `silhouette_samples adjusted_rand_score adjusted_mutual_info_score` 등과 `gap_statistic gower_distance`, 그 밖 `affinity_propagation`; 설계 [docs/design/cluster.md](docs/design/cluster.md)) |
| 그림 | `sd.plot` | [docs/api/plot.md](docs/api/plot.md) (`Figure`와 `grid`, 기본 그림 `line scatter bar area step errorbar hline vline text annotate`, 함수 그림 `function parametric polar implicit region contour density phase`, 3D `surface parametric_surface curve3d scatter3d`, 통계 그림 `histogram ecdf boxplot strip qq heatmap pairs`, 결과의 그림 `diagnostics coefficients distribution`과 메서드 `compare fit`; 설계 [docs/design/plot.md](docs/design/plot.md)) |

- 설계의 근거와 측정은 `docs/design/<영역>.md`, 통합의 결정은 [DESIGN.md](DESIGN.md), 진행 상황은 [ROADMAP.md](ROADMAP.md)에 있다.

---

## §1. dtype
```
bool  int64  float64  complex128  string
```
- dtype은 위의 객체 다섯이다. 배열과 데이터프레임 모두 `.dtype`이 이 객체를 돌려준다 (`x.dtype is sd.float64`).
  - 배열은 앞의 넷을, 데이터프레임은 `complex128`을 뺀 넷을 쓴다. 영역이 쓰지 않는 dtype을 주면 `TypeError`다 (예: `sd.zeros(3, dtype=sd.string)`).
- dtype을 받는 인자(`dtype=`, `astype`)는 **이 객체나 그 이름 문자열**을 받는다: `dtype=sd.float64`와 `dtype="float64"`는 같다.
  - Python 타입(`float` 등), numpy dtype, 다른 이름(`"float32"`)은 `TypeError`다.
- dtype 객체는 `==`로 비교하고 해싱할 수 있다. `str(sd.int64)`는 `'int64'`다. 이름 문자열과 `==`로 같지는 않다 (`sd.int64 == "int64"`는 False).
- `sd.bool`은 Python의 `bool`과 다른 객체다.

## §2. 에러 종류
- **`TypeError`**: 인자의 타입이 맞지 않는다.
  - 배열이 필요한 자리에 다른 객체를 넣었다.
  - 함수가 그 dtype을 받지 않는다 (예: complex의 순서 비교).
  - 암묵적 하향 변환이다 (docs/api/array.md §1.5). 데이터프레임의 대입과 `fillna`도 같다 (docs/api/frame.md §3.6, §5.7).
  - bool 옵션에 `True`/`False`가 아닌 값을 넣었다 (docs/api/array.md §1.1). `0`, `1`, `None`, `numpy.bool_`도 그렇다. `numpy.bool_`은 메시지에서 `not numpy.bool`로 보인다(§2.1).
  - int 인자에 int가 아닌 값을 넣었다. **int 인자는 `operator.index`의 규칙으로 읽는다**: Python int와 numpy 정수(`np.int64(3)` 등 `__index__`가 있는 객체)는 int이고, bool과 `numpy.bool_`은 아니다. 모든 영역의 int 인자(`shape`, `axis`, `k`, `n`, `num`, `head(n)`, `RangeIndex`, `set_num_threads` …)와 인덱싱이 같다. int나 float를 받는 인자(`correction`, `ord`, `rtol`)도 numpy 정수를 int로 받는다 (DESIGN.md §10.1 13번).
  - 원소 하나에 스칼라가 아닌 값을 대입했다: `x[0] = 'a'`, `x[0] = [7]` (docs/api/array.md §5.3)
- **`ValueError`**: 타입은 맞지만 값이 맞지 않는다.
  - shape이 맞지 않는다. 축 번호가 범위 밖이거나 중복되었다.
  - 인자 값이 허용 범위 밖이다 (음수 길이, `step=0`, 스레드 수).
  - 0차원 배열을 요청했다 (docs/api/array.md §1.2).
  - `copy=False`인데 복사가 필요하다 (docs/api/array.md §1.8).
  - 쓰기 금지 배열에 썼다.
  - 원소가 없는 `min`/`max`/`argmin`/`argmax` (docs/api/array.md §7.4)
- **`IndexError`**: 인덱스가 범위 밖이거나 허용하지 않는 형태다 (docs/api/array.md §5). 데이터프레임의 위치 인덱스(`.iloc`)도 같다: `s.iloc[1.0]`, `s.iloc[True]`, `s.iloc[[1.0]]` (docs/api/frame.md §5.4).
- **`KeyError`**: 없는 라벨을 찾았다 (데이터프레임의 `.loc`, docs/api/frame.md §5.4).
- **`BufferError`**: 버퍼나 DLPack으로 데이터를 주고받을 수 없다. Array API와 DLPack이 정한 예외다. CPU가 아닌 장치의 배열을 `from_dlpack`으로 가져올 때, 읽기 전용 배열을 옛 DLPack 형식으로 내보낼 때가 그렇다 (docs/api/array.md §3.5, §10.5). 버퍼 프로토콜의 내보내기 실패도 Python의 규칙대로 `BufferError`다.
- **`OverflowError`**: int64 범위를 벗어난 Python int를 int64로 바꿔야 한다 (docs/api/array.md §1.5, docs/api/frame.md §2.2). float로 계산할 수 있으면 float로 바꾼다.
  - int 인자(`shape`, `axis`, `k`, `n`, `num` 등)도 int64 범위 밖이면 `OverflowError`다. 배열과 linalg가 같다 (docs/api/array.md §1.3, docs/api/linalg.md §1.1). 예외는 `set_num_threads`의 `n`이다: 범위 밖은 크기와 상관없이 `ValueError`다 (§3).
  - double 범위를 넘는 int(예: `10**400`)를 실수로 바꿔야 할 때도 `OverflowError`다(사용자 결정 2026-11-18: 예외는 그대로, 메시지에 함수 이름을 붙인다). 실수 인자는 `var(): 'correction' is out of range`(데이터프레임의 `s.var()`도 같다), `matrix_rank(): 'rtol' is out of range`처럼 인자 이름을 쓰고, 데이터는 `asarray(): Python int too large to convert to float64`다(`Series():`, `full():` 등 같은 꼴). 연산자(`x * 10**400`)는 함수 이름이 없다. `sd.stats` 분포의 매개변수는 생성자의 이름을 붙여 `Normal(): 'mu' is out of range`다(사용자 결정 2026-11-19: 분포의 생성자도 `Series(): `처럼 클래스 이름으로 시작한다). 분포 메서드의 입력은 special과 같은 규칙이라, int64 밖의 int가 `cdf(): Python int too large to convert to int64`다.
- **`MemoryError`**: 배열 하나의 요청이 이 컴퓨터의 물리 메모리보다 크면 할당하기 전에 낸다(사용자 결정 2026-11-18). 메시지는 `zeros(): cannot allocate 8796093022208 bytes: more than this machine's physical memory`다. `sd.random`의 메서드와 분포의 `sample`도 부른 메서드의 이름으로 시작하고(`permutation(): cannot allocate ...`), 스트림을 움직이기 전에 낸다(생성기의 `position`이 그대로다). 전에는 운영체제가 주소 공간만 내주고, 쓰는 도중에 프로세스를 강제 종료했다(`rng.permutation(2**40)`). 물리 메모리보다 작지만 남은 메모리보다 큰 요청은 운영체제에 맡긴다.
- **영역 고유의 예외**는 위 내장 예외의 하위 클래스다. `sd.linalg.LinAlgError(ValueError)`: 특이 행렬, 양정치가 아닌 행렬, 수렴 실패 (docs/api/linalg.md). `sd.graph.NegativeCycleError(ValueError)`: 최단 경로의 원점에서 닿는 음수 순환 (docs/api/graph.md §3).
- 배열의 원소별 연산은 원소의 값 때문에 실패하지 않는다 (docs/api/array.md §1.6).
- 어느 영역도 경고를 내지 않는다. 문제가 있으면 에러를 낸다.

### §2.1 에러 메시지
- **공개 함수와 메서드가 내는 에러는 그 이름과 `(): `로 시작한다.** 예: `vector_norm(): 'ord' must be ...`, `astype(): 'dtype' must be ...`
  - 메서드는 클래스 이름 없이 메서드 이름만 쓴다. `x.astype(...)`과 `sd.astype(x, ...)`은 둘 다 `astype(): `, `s.sort_values()`와 `df.sort_values()`는 둘 다 `sort_values(): `다. 같은 이름은 어느 클래스, 어느 영역에서든 같은 일을 하기 때문이다 (DESIGN.md §1.4).
  - 생성자는 클래스 이름이다: `Series(): `, `DataFrame(): `, `Index(): `, `RangeIndex(): `
  - 프로퍼티는 점을 붙인 이름이다: `.T: `, `.mT: `
  - 사용자가 부른 함수의 이름이다. 여러 함수가 함께 쓰는 검사와 커널은 이름을 넘겨받는다. 예: `drop_duplicates(keep=...)`의 검사는 `duplicated(): `가 아니라 `drop_duplicates(): `, `GroupBy.sum()`은 `sum(): `, `agg("sum")`은 `agg(): `
  - `LinAlgError`도 같다: `inv(): singular matrix`
- **인자는 작은따옴표 안에 이름을 쓴다**: `'axis'`, `'dtype'`. 인자와 값을 함께 보일 때는 호출 모양 그대로 쓴다: `copy=False`, `header=False`
  - 인자가 아닌 낱말에는 따옴표가 없다. 예: `axis 3 is out of bounds for an array of dimension 2`의 axis는 배열의 축이다.
  - 받은 값은 타입 이름으로 보인다(`not list`). 문자열은 그 자체로 보인다(`not 'float32'`). 타입은 모듈을 붙인 이름으로 보인다(`not numpy.int64`; 내장 타입은 `not float`). `sd.int64`와 헷갈리지 않게 하려는 것이다(사용자 결정 2026-11-18, 전에는 `<class 'int64'>`).
  - **타입 이름은 모듈 경로 없는 짧은 이름이다**: `not ndarray`, `not list`, `not DType`. `stardust._core.ndarray`처럼 쓰지 않는다.
    - 예외는 numpy의 타입이다. 공개 모듈을 붙인다(비공개 부분 `_generator` 등은 뺀다): `not numpy.bool`, `not numpy.int64`, `not numpy.float64`, `not numpy.ndarray`, `not numpy.dtypes.Int64DType`, `not numpy.random.Generator`. Python의 `bool`, sd의 `ndarray`·`int64`와 헷갈리지 않게 하려는 것이다(사용자 결정 2026-11-19; 전에는 `sum(): 'keepdims' must be True or False, not bool`처럼 numpy의 bool이 Python의 bool로 읽혔다). 모든 영역이 한 함수(C++ `py_type_name`, Python에서는 `_core._type_name`)로 이름을 만든다.
    - Python이 직접 내는 메시지도 같다. C++로 묶은 타입도 Python으로 쓴 클래스처럼 짧은 이름을 갖는다: `unsupported operand type(s) for +: 'ndarray' and 'list'`, `unhashable type: 'ndarray'`, `cannot pickle 'ndarray' object`, `'ndarray' object has no attribute 'foo'`
- **같은 실수에는 어느 영역에서든 같은 메시지다** (DESIGN.md §10.3). 옵션의 종류마다 검사가 하나라서 형식이 하나다.
  - bool 옵션: `sort(): 'stable' must be True or False, not int`, `sum(): 'skipna' must be True or False, not NoneType`
  - int 인자: `head(): 'n' must be an int, not float`, int64 범위 밖은 `head(): 'n' is out of range`(`OverflowError`)
  - 문자열 옵션: `merge(): 'how' must be 'inner', 'left', 'right' or 'outer', not NoneType`(타입, `TypeError`), `..., not 'x'`(값, `ValueError`). linalg의 `mode`, 배열의 `meshgrid(indexing=)`, `searchsorted(side=)`, `quantile(method=)`도 같다.
  - 수 인자: `std(): 'correction' must be an int or a float, not NoneType`
  - **`None`도 같은 검사를 거친다.** `None`이 뜻이 있는 옵션(`axis=None`, `dtype=None`, `copy=None`, `rtol=None`)이 아니면 다른 틀린 값과 같다: `svd(): 'mode' must be 'reduced' or 'complete', not NoneType`, `astype(): 'dtype' must be ... not NoneType`
- **공개 함수의 메시지에 내부 이름이 나오지 않는다.** C++ 클래스(`Column`), 바인딩의 시그니처(nanobind의 `incompatible function arguments` 목록), Python 내부의 메시지(`object of type 'int' has no len()`, `'>=' not supported between ...`, `AttributeError`)가 그렇다. 인자를 먼저 검사해서 위의 형식으로 낸다.
- **Python 프로토콜은 Python의 형식이고 이름을 붙이지 않는다**: 연산자, 인덱싱과 그 대입(`x[i]`, `.loc`, `.iloc`, `df["c"]`), `len`, `in`, `bool`, `iter`, `repr`, 속성 대입
  - 연산자: `unsupported operand dtypes for <: complex128 and complex128`, 단항 연산자는 `unsupported operand dtype for unary -: string`, `abs()`는 `unsupported operand dtype for abs(): string`. 받지 않는 타입의 피연산자는 Python 자신의 `unsupported operand type(s) for +: ...`, `'==' not supported between instances of ...`다.
  - 같은 연산도 부르는 길에 따라 형식이 다르다. `sd.add(x, y)`의 shape 오류는 `add(): shapes could not be broadcast together: ...`이고 `x + y`는 `shapes could not be broadcast together: ...`다.
  - 인덱싱이 없는 라벨이나 열을 찾으면 dict처럼 `KeyError(label)`다. 함수가 없는 열을 받으면 `sort_values(): no column 'x'`다.
  - 속성 대입: `property 'shape' of 'ndarray' object has no setter`
- **틀린 호출 모양도 함수 이름으로 시작한다**: 없는 키워드, 빠진 인자, 인자 개수, 위치 전용 인자를 키워드로 준 호출이다. Python으로 쓴 무거운 입구가 그렇다: 모듈 수준의 함수(`sd.stats`의 기술통계, `sd.regression`과 `sd.optimize`의 모든 함수, `sd.random.default_rng`, `sd.frame`의 `read_csv`·`merge`·`concat`), 생성자(frame의 `Series`·`DataFrame`·`Index`·`RangeIndex`, 분포), 그리고 frame의 `groupby`.
  - 문장은 Python의 것이고, 앞의 `f() got an`이 `f(): `가 된다: `ols(): unexpected keyword argument 'foo'`, `moment(): missing 1 required keyword-only argument: 'order'`, `Normal(): takes 0 positional arguments but 1 was given`, `merge(): got some positional-only arguments passed as keyword arguments: 'right'`
  - 이름은 위의 규칙과 같다. 생성자는 클래스 이름이다.
  - 이유: 값이 틀린 에러와 호출이 틀린 에러의 꼴이 같아야 메시지만 보고 어느 함수인지 안다.
- **예외: 작은 메서드는 Python의 기본 꼴이다**: frame의 메서드, `Generator`의 메서드, 분포의 메서드(`Series.head() got some positional-only arguments passed as keyword arguments: 'n'`, `Generator.random() got an unexpected keyword argument 'foo'`, `sample() missing 1 required keyword-only argument: 'rng'`처럼 Python이 만드는 그대로다. 콜론이 없고, 클래스 이름이 붙기도 한다).
  - 이유: 꼴을 바꾸는 감싸기가 호출마다 약 60 ns를 더한다. 이 메서드들은 한 번에 수백 ns로 끝나는 일이 많아서 그 비용이 크고, 호출 모양은 코드를 쓸 때 한 번 고치는 실수다.
- **예외: C++ 바인딩(nanobind)으로 바로 묶은 함수**(배열의 함수와 메서드, `sd.linalg`, `sd.special`)의 틀린 호출 모양은 nanobind의 메시지다: `sum(): incompatible function arguments. The following argument types are supported: ...`. 받을 수 있는 시그니처를 인자 이름과 함께 보인다(위치 전용 인자도 이름이 있고, `help()`의 시그니처와 같다).
  - 이유: 이 메시지를 바꾸려면 바인딩마다 모든 인자를 `*args, **kwargs`로 받아 직접 풀어야 하고, 그 비용이 호출마다 든다. 호출 모양은 코드를 쓸 때 한 번 고치는 실수다.
  - 인자의 **값**이 틀린 것은 이 함수들도 언제나 위의 형식이다.
- 파일을 열 때의 `FileNotFoundError`는 Python의 것 그대로다.
- **메시지에 문서의 절 번호를 넣지 않는다.** 문서의 번호는 바뀌어도 메시지는 사용자 코드가 붙잡고 있을 수 있다.

### §2.2 결과의 배열과 실패의 표시
- **결과 객체의 배열은 모두 쓰기 금지다.** 모든 영역의 결과 객체와 namedtuple의 배열 필드가 그렇다: 회귀와 최적화의 결과 객체, 배열의 `unique_all`·`unique_counts`·`unique_inverse`, 선형대수의 `QRResult`·`LstsqResult` 등, 통계의 `DescribeResult`·`ModeResult` 등. 분포의 매개변수 속성과 요약값(`mean()`, `covariance()` 등)이 돌려주는 배열도 같다.
  - 쓰면 쓰기 금지 배열에 쓴 것과 같은 `ValueError`다(`assignment destination is read-only`). 고쳐 쓰려면 `.copy()`로 복사한다.
  - 읽을 때 복사하지 않는다. 결과가 가진 메모리를 읽기 전용으로 보여 줄 뿐이라, 큰 필드도 읽는 비용이 없다.
  - 이유: 결과의 필드는 서로에게서 나온다(회귀의 `bse`는 `cov_params()`의 대각에서 나온다). 쓸 수 있으면 한 필드를 고친 뒤에 처음 읽는 다른 필드가 바뀌어서, 같은 결과의 값이 읽는 순서에 따라 달라진다. 읽을 때마다 복사해 주면 큰 배열(최적화의 `hess_inv`)은 원소 하나를 읽는 데도 전체를 복사한다.
- **분포는 배열 매개변수를 생성할 때 한 번 복사한다.** 생성 뒤에 넘긴 배열을 고쳐도 분포는 그대로다. 매개변수 속성은 그 복사본을 읽기 전용으로 돌려준다. 같은 이유다: 속성과 메서드가 서로 다른 값을 보면 안 된다.
- **반복 계산이 끝까지 가지 못하면**(반복 상한, 수렴 실패, 유한하지 않은 값):
  - 상태 필드(`converged`, `status`, `message`)가 있는 결과는 마지막 값과 그 상태를 돌려준다. 에러가 아니다.
  - 상태 필드가 없는 원소별 함수(`sd.special`, 분포의 메서드)는 그 원소를 NaN으로 둔다.
  - 배치(여러 문제나 원소를 한 번에 계산하는 호출)에서는 실패한 문제나 원소만 그렇게 되고 나머지는 정상 값이다. 풀 것이 없는 문제(행이 없는 회귀)도 그 문제만 NaN 결과다(docs/api/regression.md).
  - 예외는 선형대수의 분해다: 수렴하지 못하면 `LinAlgError`다(§2).
  - 이유: 상태 필드가 있으면 사용자가 실패를 확인할 수 있고, 마지막 값은 무엇이 잘못됐는지 보는 데 쓴다. 원소별 함수는 실패를 알릴 자리가 값밖에 없고, 경고를 내지 않으므로(§2) NaN이 그 표시다. 원소 하나 때문에 배치 전체를 잃지 않는다.
- **정할 수 없는 값은 NaN이다.** 예: 잔차가 정확히 0인 회귀의 t값과 p값(분모인 표준오차가 0이다). ±inf와 0을 내면 반올림 크기의 계수까지 "매우 유의"로 보인다.


### §2.3 결과의 `summary()`와 `repr`
- **회귀·가설검정·최적화의 결과 객체는 모두 `summary()`가 있고 글(`str`)을 돌려준다.** 모양은 하나다: 회귀 모형의 statsmodels 꼴 상자(78칸 가운데의 제목, `=` 줄, 38칸씩 두 열의 "이름: 값" 머리, `-` 줄 아래의 표, 맺는 `=` 줄, 필요하면 `Notes:`). 그리는 곳도 하나(`stardust/_summary.py`)다. 희소 배열, 분포, frame은 대상이 아니다.
  - 머리의 수는 statsmodels의 꼴이다: 통계량 `#.4g`, p값 `#.3g`, 자유도는 정수면 정수. 표의 수는 statsmodels의 forg(소수 3자리, 계수는 4자리; [1e-4, 1e4) 밖은 유효숫자)이고 표의 p값은 소수 3자리다. 표가 78칸보다 넓으면 줄만 넓어지고 제목과 머리는 78칸 그대로다.
  - 구간이 있는 결과(`conf_int(alpha=)`가 있는 결과)만 `summary(*, alpha=0.05)`이고 `alpha`의 검사와 메시지는 `conf_int`와 같다(regression G99, htest H16). 다른 결과의 `summary()`는 인자가 없다.
  - 배치(여러 문제를 한 번에 푼 결과)는 줄마다 한 문제(레인)인 표다. 줄의 이름은 그 레인의 첨자(`[0]`, `[1, 2]`)다. 20줄을 넘는 표는 앞뒤 5줄씩만 두고 가운데에 `...` 줄을 둔다. 긴 x(최적화)도 같다.
  - 배치에서 자유도(`df`, `df_num`, `df_denom`, `df_model`)는 모든 레인이 같은 값이면 머리에 한 번, 레인마다 다르면 표의 열에 둔다(regression G195). 회귀의 검정과 비교(`wald_test`, `compare_f_test`, `t_test` 등), htest의 검정, 모형 배치의 `Df Model`이 모두 이 규칙이다. 이유: 같은 자료의 모양(같은 X의 배치)이면 어느 결과든 자유도가 같은 자리에 있어야 하고, 레인마다 다른 값(열이 겹친 레인의 계수 수)을 첫 레인의 값 하나로 보이면 틀린다.
  - 이유: 영역마다 다른 꼴이면 사용자가 꼴마다 읽는 법을 배워야 한다. 회귀 상자는 statsmodels 사용자에게 익숙하고, 머리와 표의 두 부분으로 어떤 결과든 담긴다.
- **`repr`는 한 줄 `Name(field=value, …)`이고 주된 필드를 보인다.** 배치는 `Name(batch shape (4,))`다(회귀 모형은 뒤에 계수 수를 붙인다: `OLSResult(batch shape (3,), 6 parameters)`). 회귀와 htest의 모든 결과(모형, 정규화, 검정, `BreuschPaganResult`·`TheilslopesResult`·`SiegelslopesResult`)는 같은 꼴로 수를 유효숫자 6자리로, 짧은 1차원 배열(6개까지)을 `[v, v]`로, 다른 배열을 `<array of shape (m, k)>`로 적는다(regression G194): `OLSResult(nobs=40, rank=4, rsquared=0.804348, params=[0.421749, 1.19944, -1.922, 0.302937])`. 쓰는 곳은 `stardust/_summary.py`의 한 함수다. 여러 줄의 표는 `summary()`의 몫이다.

## §3. 스레드와 결정성
```
set_num_threads(n, /)
get_num_threads()
```
- `n`은 1 이상 1024 이하의 int다.
  - 범위 밖이면 `ValueError`, int가 아니면(`bool` 포함) `TypeError`다.
- 기본값은 P코어 수(10)다.
- **같은 기계, 같은 스레드 수, 같은 입력이면 실행마다 비트까지 같다** (사용자 결정 2026-11-22). 리덕션과 정렬도 결정적이다.
  - 스레드 수가 달라도 같은 비트인 것은 약속이 아니다. 성능이 먼저이고 정확도가 그다음이라, 스레드 수에 따라 마지막 자리가 달라지는 더 빠른 계산도 쓸 수 있다. 지금은 모든 영역의 분할이 데이터의 크기로만 정해져 결과가 스레드 수와 무관하다(영역 문서마다 적는다). `sd.random`은 이것을 약속한다(docs/api/random.md).
  - 메모리 레이아웃(C 순서, F 순서, strided)이 달라도 같은 비트인 것도 약속이 아니다. 몇몇 경로는 레이아웃에 따라 마지막 자리가 다르다(docs/api/linalg.md §1.11, docs/api/regression.md).
  - BLAS·LAPACK(Accelerate)을 부르는 계산(행렬 곱, 분해, 회귀의 QR 등)은 Accelerate의 스레드 모드(단일 스레드인지 다중 스레드인지, 환경 변수 `VECLIB_MAXIMUM_THREADS`가 정한다)가 같을 때 비트까지 같다. 두 모드 사이에서는 마지막 자리 몇 개가 다를 수 있다(ols의 `bse`가 상대 2e-16, `lstsq`가 3e-14).
  - 이유: Accelerate가 안에서 일을 나누는 방식은 sd가 정하지 않는다. sd가 모드를 고정하면 사용자의 설정을 덮고, 큰 행렬 곱이 느려질 수 있다. 그래서 sd는 모드를 바꾸지 않는다.
- `n`이 1이면 스레드풀을 쓰지 않는다. 환경 변수는 두지 않는다.
- **GIL**: 작업량이 영역의 병렬 기준값 이상이면 스레드 수와 상관없이 GIL을 놓고 계산한다. 그래서 `set_num_threads(1)`이어도 큰 계산이 도는 동안 다른 Python 스레드가 돈다. 기준값보다 작은 작업은 GIL을 쥔 채 계산한다.
  - 기준값은 스레드풀로 나누기 시작하는 작업량이고, 모든 영역에 같은 규칙을 쓴다.
    - 배열, special, stats의 분포 함수, random: float64 덧셈 32,768개 분량이다. 원소 하나의 계산이 무거운 함수(`exp`, `special.erf`, 분포의 `cdf` 등)는 그만큼 적은 원소부터, 원소가 1바이트인 `bool` 배열의 원소별 연산과 리덕션은 262,144 원소부터다.
    - linalg: 호출 하나의 추정 계산 시간이 10µs 이상일 때다. 행렬 하나만 넘겨도 같다.
    - 데이터프레임: 131,072 원소다 (docs/api/frame.md §12).
  - 기준값과 상관없이 GIL을 놓는 계산: `cov`, `corrcoef`의 행렬 곱, stats의 여러 기술통계 함수와 다변량 분포, 분포의 `sample`, random의 `permutation`.
  - 기준값 이상이어도 GIL을 쥔 채 도는 계산:
    - Python 객체를 읽거나 만드는 변환: 리스트를 받는 `asarray`, `tolist`.
    - 인덱스 배열 하나로 하는 대입 `x[idx] = v`. 같은 인덱스가 여러 번 나오면 마지막 값이 남아야 해서 순서대로 쓴다.
    - 새 메모리를 받기만 하는 `zeros`, `eye`.
- 다른 Python 스레드가 이미 스레드풀을 쓰고 있으면, 새 호출은 1스레드로 계산한다. 지금은 결과가 스레드 수와 무관하므로 결과도 같다.

## §4. numpy·scipy와 같아야 하는 것
numpy, scipy, pandas는 정답지다. 무엇이 정답지와 같아야 하고 무엇은 달라도 되는지는 아래 네 규칙이 정한다 (DESIGN.md §1.6).

1. **sd가 지키는 비트**: 같은 기계, 같은 스레드 수에서 실행마다 같은 비트다(§3). 스레드 수와 무관한 비트와 numpy·scipy와의 비트 동일은 약속이 아니다 (사용자 결정 2026-11-22). 공개 함수의 결과는 저장해 둔 sd 자신의 출력(골든)과 견주어, 바뀌면 안 되는 변경에서 비트까지 그대로인지 확인한다. 정확도는 Arb로 구한 참값과 견준 계약으로 정한다(영역 문서의 정확도 절).
   - 이유: 재현성은 사용자가 같은 프로그램에서 같은 답을 얻는 것이고, 그것은 sd가 정할 수 있다. 옳음은 참값으로 재야 한다. 다른 라이브러리의 반올림과 같다고 옳은 것이 아니다.
2. **반올림이 아닌 동작은 numpy·scipy와 똑같다**: dtype 규칙, 정수 연산, 정렬 순서, NaN과 ±0의 규칙, 상태 값의 뜻, 기본값, 멈춤 규칙. 다르게 정한 곳은 영역 문서에 `[numpy와 다름]`, `[scipy와 다름]`으로 이유와 함께 적는다.
   - 이유: 사용자 코드는 이것들에 기대어 갈라진다. 결과의 형, 고르는 원소, 분기, 반복 수가 달라지면 반올림 크기의 차이가 아니라 다른 프로그램이 된다.
3. **반올림 크기의 결과는 정확도 계약 안이면 된다**: numpy·scipy와 마지막 자리가 다를 수 있다. 더 빠르거나 더 정확한 계산 순서를 쓴다. 성능이 먼저이고 정확도가 그다음이다 (사용자 결정 2026-11-22). 최적화는 같은 알고리즘과 같은 멈춤 규칙을 쓰고, 마지막 자리의 차이 때문에 반복의 경로와 횟수가 조금 달라질 수 있다.
   - 이유: 마지막 자리까지 같으려면 numpy·scipy의 안쪽 계산 순서를 그대로 따라야 한다. 그 순서는 그 라이브러리의 약속이 아니라서 판과 기기마다 바뀌고, 따라 하면 병렬화, SIMD, 더 정확한 합을 쓸 수 없다.
   - 예: int64의 `mean`은 정수로 정확히 더한 뒤 한 번 반올림한다(numpy는 원소마다 float64로 바꿔 더한다). FMA는 코드가 정한 곳에서만 쓰고 컴파일러의 자동 축약은 끈다(DESIGN.md §1.6).
4. **지금 비트까지 같은 것은 바꾸려고 바꾸지 않는다**: 고침이나 최적화가 필요로 할 때만 깬다. numpy·scipy와 비트까지 견주는 테스트는 통과하는 동안 그대로 두고, 의도한 고침이나 최적화 때문에만 깨지면 허용오차와 반복 수의 비교로 바꾼다.
   - 이유: 비트가 같으면 그것은 공짜로 얻는 가장 강한 검사다. 이유 없이 버리지 않지만, 그것 때문에 고침과 개선을 막지도 않는다.

## §5. 받는 입력
- **데이터를 받는 상위 함수는 `sd.ndarray`, 수의 시퀀스(리스트, 튜플), numpy 배열, `sd.frame`의 열(`Series`, 회귀는 `DataFrame`도)을 받는다**: `sd.stats`의 기술통계, `sd.regression`의 모형 함수, `sd.optimize`의 시작점 `x0`. 받은 것은 그 함수가 계산하는 dtype의 배열로 바꿔 쓰고, frame의 결측값은 NaN이 된다.
  - 이유: 이 함수들은 사용자가 밖에서 가져온 자료를 처음 넣는 자리다. 바꾸는 비용은 계산에 비해 작고, 매번 `sd.asarray`를 쓰게 하면 얻는 것 없이 번거롭다.
- **배열·선형대수·특수함수의 함수와 분포는 `sd.ndarray`만 받는다**(원소별 함수와 분포의 매개변수·메서드는 Python 스칼라도 받는다). 리스트와 numpy 배열은 `TypeError`다(`sum(): 'x' must be an sd.ndarray, not list`). 바꾸려면 `sd.asarray`를 쓴다.
  - 이유: 이 함수들은 다른 계산을 짜는 부품이라 자주, 작은 입력으로 불린다. 암묵적 변환은 그때마다 복사와 dtype 추론을 숨긴다. sd의 dtype은 numpy보다 적어서(§1) 변환이 어떤 dtype을 낳는지가 눈에 보여야 한다.

## §6. 같은 이름, 다른 뜻
같은 개념은 같은 이름이다(DESIGN.md §1.4). 아래는 같은 이름이 다른 개념을 가리키거나, 같은 개념의 기본값이 영역마다 다른 곳이다. 모두 다른 라이브러리의 관례에서 왔고, 바꾸면 그쪽 사용자의 기대가 깨진다.
- **`rtol`**
  - `sd.linalg`(`matrix_rank`, `pinv`, `lstsq`)와 `sd.regression`(`ols` 등): 랭크를 정하는 상대 기준이다. linalg는 가장 큰 특이값 대비, 회귀는 열마다 그 열의 노름 대비다(`|R_jj| ≤ rtol·‖x_j‖`). 기본값의 식(행과 열 수 가운데 큰 것 × float64의 머신 엡실론)은 같다.
  - `sd.optimize.root_scalar`: 근의 상대 허용오차다(scipy와 같은 이름과 뜻). 다른 풀이 함수는 같은 뜻에 `xrtol`을 쓴다(`find_root`, `find_minimum`).
- **`histogram`**: `sd.histogram`은 값(`HistogramResult`)을, `sd.plot.histogram`은 그림(`Figure`)을 돌려준다. `bins`, `range`, `density`는 같은 뜻이고 그림의 막대 높이가 `sd.histogram`의 결과다(docs/design/plot.md PL20).
- **`correction`의 기본값**: 배열의 `sd.var`·`sd.std`와 frame의 `var`·`std`는 `correction=0`이다(Array API, numpy; DESIGN.md §8.2). `sd.stats`의 `describe`와 `sem`은 `correction=1`이다(scipy.stats와 같은 기본값). 같은 이름이고 뜻도 같지만(자유도에서 빼는 수) 기본값이 다르다.
  - 이유: 배열은 numpy의 기본값을, 통계는 scipy.stats의 기본값을 따른다. 한쪽으로 맞추면 그 라이브러리에서 옮겨 온 코드가 조용히 다른 값을 낸다. 표본 분산은 어디서든 `correction=1`로 적어 주면 같다.

## §7. sd.sparse와 밀집 배열
- **sd.sparse의 기능은 밀집의 부분집합이거나 희소에만 필요한 것이다**(사용자 결정 2026-11-22). 희소에 있는 기능이 희소의 것이 아니면(희소 저장과 상관없는 만들기, 판정, 옵션) 밀집 배열에도 있다. 희소에만 필요한 것은 형식과 변환(`tocsr`, `asformat`, `to_scipy` …), 저장의 관리(`eliminate_zeros`, `sum_duplicates`, `nnz`, `explicit=`), 희소의 밀도와 자리(`random_array`), 희소 우변의 풀이 같은 것이다.
  - 이유: 희소는 같은 수학을 다른 저장으로 하는 것이다. 희소에서만 되는 일이 있으면 사용자는 밀집 문제를 풀려고 희소로 바꾸게 되고, 두 쪽의 결과를 견줄 기준도 없다.
- **복사이거나 같은 반올림 한 번인 짝만 비트까지 같다**: 둘 다 있는 입력에서 그런 짝은 밀집 함수의 결과가 희소 함수의 `toarray()`와 같다(만들기, 인덱싱, 원소별 연산, `diagonal`, float의 `trace`, `kron`·`kronsum`, 대각 행렬과 2 × 2 삼각 행렬의 `expm`). 다를 수 있는 것은 0의 부호다(희소는 0을 저장하지 않는다).
  - 계산 순서가 다른 짝은 두 쪽 모두 정확도 계약 안에서 끝자리가 다를 수 있다: 풀이(`spsolve`, `inv`, `spsolve_triangular`, `splu`), 일반 행렬의 `expm`, 리덕션(희소는 저장된 값만 더하므로 pairwise 트리가 다르다), `norm`이다. 짝마다의 규칙과 예는 각 영역 문서가 적는다(sparse.md §3, §5, §8).
  - 결과 dtype이 다른 짝은 아래 표가 적는다. bool·int 입력에서 희소 쪽은 배열 영역의 규칙(`sd.sum`, `sd.multiply`)이라 int64이고, 밀집 linalg 쪽은 float64다(DESIGN §9.6). float64·complex128 입력에서는 두 쪽이 비트까지 같다.
- **희소 함수 → 밀집 짝**

| sd.sparse | 밀집 | 문서 |
|---|---|---|
| `eye_array`, `tril`, `triu` | `sd.eye`, `sd.tril`, `sd.triu` | array.md §3 |
| `diags_array` | `sd.diag`(대각 하나, 여럿은 합) | array.md §3.7 |
| `setdiag` 메서드 | 고급 인덱스 대입 `x[i + max(-k, 0), i + max(k, 0)] = v` | array.md §3.7 |
| `hstack`, `vstack` | `sd.concat`, `sd.stack`(이름은 각자 기준 라이브러리의 것) | array.md §4.2 |
| `block_array` | `sd.block`(None 블록 대신 `zeros`) | array.md §4.5 |
| `find`, `nonzero` | `sd.nonzero`와 인덱싱 | array.md §5.4 |
| `resize` 메서드 | `sd.zeros`와 슬라이스 대입(밀집 배열의 모양은 바뀌지 않는다) | array.md §5.3 |
| 원소별 메서드(`sin` … `sign`), `power`, 리덕션 | 같은 이름의 배열 함수, `sd.pow` | array.md §6, §7 |
| `deg2rad`, `rad2deg` 메서드 | `sd.deg2rad`, `sd.rad2deg` | array.md §6.4 |
| `kron` | `sd.linalg.kron`(1차원 입력 포함); bool·int의 결과 dtype이 다르다(희소 int64, 밀집 float64) | linalg.md §2, sparse.md §4 |
| `kronsum` | `sd.linalg.kronsum`; bool·int의 결과 dtype이 다르다(희소 int64, 밀집 float64) | linalg.md §2, sparse.md §4 |
| `block_diag` | `sd.linalg.block_diag` | linalg.md §2 |
| `trace`, `diagonal` 메서드, `norm` | `sd.linalg.trace`, `diagonal`, `matrix_norm`·`vector_norm`; bool·int의 `trace` 결과 dtype이 다르다(희소 `A.trace()`는 int64, 밀집 `sd.linalg.trace`는 float64) | linalg.md §2, §3, sparse.md §4 |
| `spsolve`, `inv`, `spsolve_triangular`, `expm` | `sd.linalg.solve`, `inv`, `solve_triangular`, `expm` | linalg.md §4, §7 |
| `splu`, `factorized`, `SparseLU.solve(trans=)` | `sd.linalg.lu`, `lu_solve(trans=)` | linalg.md §4 |
| `spbandwidth`, `is_sptriangular` | `sd.linalg.bandwidth`(`upper == 0`이면 아래 삼각) | linalg.md §6 |
| `eigsh`, `eigs`, `svds`, `onenormest`, `expm_multiply`, `LinearOperator`, 반복법 | 그 함수가 밀집 배열도 받는다 | sparse.md §5–§8 |
