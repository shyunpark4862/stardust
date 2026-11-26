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

## 공통 결정 (모든 작성·검증 브리프에 들어감)
- 본보기: `src/content/api/linalg/solve.mdx`, `lstsq.mdx`, `src/content/areas/linalg.mdx`, 클래스는 `api/regression/TheilslopesResult.mdx`. 문체·frontmatter 꼴·예제 꼴을 이들과 맞춘다.
- 그룹 이름(정확히 이대로): Array model / Creating arrays / Shape and axes / Joining and repeating / Data types / Arithmetic / Exponents and logarithms / Trigonometric and hyperbolic / Rounding / Comparison, logic and bits / Classification and selection / Complex numbers / Reductions / Cumulative / Skipping NaN / Statistics / Indexing and searching / Sorting and sets / Matrix products / Threads
- 타입 표기: 입력은 `array` 또는 `array of shape (...)`, `array_like`는 쓰지 않음(실제로 파이썬 리스트·스칼라를 받는지는 돌려 보고 받으면 `array or scalar` 식으로 적음). 반환은 `ndarray` 또는 `ndarray of shape (...)`. dtype 이름은 백틱(`float64`). `int`, `bool`, `float`, `tuple of int`, `int or None`.
- 용어: array(“tensor”, “matrix”는 2-D 문맥에서만), element-wise, axis/axes, shape, dtype, broadcast, view / copy, reduction, NaN, "0-D array". 차원은 "1-D", "2-D".
- 여러 페이지가 공유하는 규칙은 한 곳에만 설명하고 링크: 브로드캐스팅 → `/api/array/broadcasting/`, dtype 승격 → `/api/array/dtypes/`, 뷰·복사 → `/api/array/views/`, 인덱싱 → `/api/array/indexing/`, `axis`·`keepdims`·`out=`·NaN 일반 규칙 → `/start/conventions/`. 함수 페이지에는 그 함수 고유의 내용만.
- 묶음(family) 페이지: title은 짧은 영어 제목(예: "zeros, ones, full, empty"처럼 이름 나열), signatures에 전부, params는 공유 인자 한 번(fmcheck가 요구하는 꼴을 따름).
- 비교 절 제목은 정확히 `## For NumPy, SciPy and R users`. array 영역은 NumPy와 R base만 해당. 차이는 고정 인터프리터의 numpy로 실제 확인한 것만.
- 예제 배열 출력은 고정 빌드의 repr 그대로(`array([...], dtype=float64)`).
- 아직 없는 페이지로의 링크·seeAlso는 이 문서의 slug 목록에 있는 것만 허용.
