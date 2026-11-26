# array (`sd`) — 얼개

기준: `API-INVENTORY.md`의 `sd` 171개 이름 + `ndarray` 멤버. 개요는 `src/content/areas/array.mdx`, 페이지는 `src/content/api/array/<slug>.mdx`.
페이지 이름(slug)은 아래가 기본이고, 작성 중 더 나은 묶음이 보이면 바꿔도 된다(개요의 groups와 맞출 것).

## 개요 (A1에서 씀)
- title: Arrays, ns: `sd`, lead: N차원 배열과 그 위의 원소별 함수·집계·정렬·행렬곱; 선형대수 분해는 `sd.linalg`, 표 데이터는 `sd.frame`.
- glance: asarray, filled(zeros…), arange/linspace, reshape, sum, mean, sort, matmul
- Concepts: 배열(모양·축·dtype), 원소별 함수와 브로드캐스팅, 집계와 `axis`/`keepdims`, 뷰와 복사 — 각 개념 페이지로 링크.
- For NumPy, SciPy and R users: 대응표(stardust | NumPy | R) + 차이(이름: concat/permute_dims/pow 등 Array API 이름, dtype 5종만, 정수 나눗셈·NaN 규칙, `out=`, 기본 correction=0 등 — 실제로 돌려 확인).

## 그룹과 페이지

### A1 Array model and creation
- **Array model**: `ndarray`(class: 속성 T, mT, base, dtype, itemsize, nbytes, ndim, shape, size, strides 표; 메서드 표 — 함수와 같은 메서드는 그 함수 페이지로 링크(`slug`), copy·tolist는 이 페이지 `### `), `dtypes`(concept: bool·int64·float64·complex128·string, 승격 규칙), `constants`(e, pi, inf, nan, newaxis), `indexing`(concept: 정수·슬라이스·불리언·정수 배열 인덱싱, 대입), `broadcasting`(concept), `views`(concept: 뷰와 복사, base, copy=)
- **Creating arrays**: `asarray`, `filled`(zeros, ones, empty, full), `filled-like`(zeros_like, ones_like, empty_like, full_like), `arange`, `linspace`, `eye`, `diag`, `tril-triu`, `meshgrid`, `from_dlpack`
- **Shape and axes**: `reshape`(+ndarray.reshape), `ravel`, `permute_dims`(+matrix_transpose), `moveaxis`, `squeeze-expand_dims`, `broadcast_to`(+broadcast_arrays)
- **Joining and repeating**: `concat`, `stack-unstack`, `block`, `flip`, `roll`, `repeat`, `tile`
- **Data types**: `astype`(+ndarray.astype), `can_cast-result_type`, `isdtype`, `finfo-iinfo`

### A2 Element-wise functions
- **Arithmetic**: `arithmetic`(add, subtract, multiply, divide, negative, positive), `floor_divide-remainder`, `pow`, `abs`, `square-sqrt`, `sign`, `reciprocal`, `maximum-minimum`, `clip`, `copysign`, `hypot`, `nextafter`
- **Exponents and logarithms**: `exp-log`(exp, expm1, log, log1p, log2, log10), `logaddexp`
- **Trigonometric and hyperbolic**: `trig`(sin…atan), `atan2`, `hyperbolic`, `deg2rad-rad2deg`
- **Rounding**: `rounding`(floor, ceil, trunc, round)
- **Comparison, logic and bits**: `comparison`, `logical`, `bitwise`
- **Classification and selection**: `isnan-isinf-isfinite`, `signbit`, `where`
- **Complex numbers**: `complex`(real, imag, conj)
- 연산자(`+`, `@`, `==` …)와 원소별 함수의 대응은 `arithmetic`·`comparison` 등 해당 페이지에, `out=` 규칙은 Conventions로 링크.

### A3 Reductions and statistics
- **Reductions**: `sum`(+ndarray.sum), `prod`, `min-max`, `mean`, `var-std`, `argmin-argmax`, `any-all`, `count_nonzero`
- **Cumulative**: `cumulative`(cumulative_sum, cumulative_prod)
- **Skipping NaN**: `nan-reductions`(nansum … nanargmax, nancumulative_*) — 한 페이지 또는 둘
- **Statistics**: `quantile`(+nanquantile), `median`(+nanmedian), `cov-corrcoef`, `bincount`, `histogram`

### A4 Indexing, sorting and products
- **Indexing and searching**: `take`, `take_along_axis`, `nonzero`, `searchsorted`, `interp`, `diff`
- **Sorting and sets**: `sort-argsort`, `unique`(unique_values, unique_counts, unique_inverse, unique_all)
- **Matrix products**: `matmul`, `tensordot`, `vecdot`
- **Threads**: `threads`(set_num_threads, get_num_threads)

## 확인할 점
- `sd.vecdot`과 `sd.linalg.vecdot`이 같은 것인지(같으면 한 쪽 페이지 + 별칭 표기).
- dtype 이름 값(`sd.bool` 등)은 `dtypes` 페이지의 names에.
