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
- (A1 일치 점검 뒤 추가, A2–A4 작성·검증이 따름)
  - 용어: 산문은 "dtype". "data type"은 그 페이지에서 처음 소개할 때(요약, 첫 문장, `dtype` 인자 설명 첫머리 "Data type of the result.")와 "Data types" 페이지 제목·링크 글자에만 쓴다. "(C) order"는 쓰지 않고 "row-major order".
  - 공통 규칙은 다시 설명하지 말고 한 문장 + 아래 고정 앵커로 링크(제목을 바꾸면 앵커가 깨지므로 이 제목들은 바꾸지 않음):
    `/api/array/dtypes/#type-promotion`, `#python-scalars`, `#arithmetic-on-booleans`, `#functions-with-floating-point-results`, `#implicit-conversion`, `#explicit-conversion`, `#integer-overflow`, `#data-type-of-a-new-array`;
    `/start/conventions/#array-shapes`, `#axis-and-keepdims`, `#the-out-argument`, `#missing-values`, `#determinism-and-threads`, `#errors-and-warnings`;
    `/api/array/views/#the-copy-argument`, `#read-only-arrays`, `#overlapping-memory`; `/api/array/indexing/#assignment`; `/api/array/broadcasting/#the-rule`.
  - `out=`를 받는 함수 페이지: 인자 설명은 "Array to write the result into. See [the `out=` argument](/start/conventions/#the-out-argument)." 꼴로 짧게, 변환·브로드캐스팅·별칭 규칙은 되풀이하지 않음.
  - 메모리 배치: 규칙은 `ndarray` Notes("Memory layout")에 한 번. 함수 페이지는 필요할 때만 "The result is laid out in row-major order." 또는 "The result has the memory layout of `x`." 중 하나를 그대로 씀.
  - 뷰·복사: 함수 페이지는 "The result is a view of `x` or a copy (see [Views and copies](/api/array/views/))." 정도로만, `base` 설명은 Views에만. `from_dlpack` 결과의 `base`는 어디에도 적지 않음(이상 기록됨).
  - 메서드: `ndarray`의 메서드와 같은 함수 페이지(A3의 `sum`, `prod`, `min-max`, `mean`, `var-std`, `argmin-argmax`, `any-all`)는 `names`에 `ndarray.<method>`를 넣고(예: `names: [sum, ndarray.sum]`), `signatures`에 `sd.ndarray.<method>(...)` 줄을 더하고, `x` 인자 설명에 "In the method form, the array the method is called on."을 붙임(`reshape`·`astype` 페이지와 같은 꼴).
  - 그 함수 페이지의 `summary`는 `ndarray.mdx`의 `methods[].summary`와 글자까지 같게 한다. 지금 `ndarray.mdx`에 적힌 요약(예: "Sum the elements over the given axes.")을 그대로 쓰거나, 바꾸면 검증 단계에서 `ndarray.mdx` 쪽도 같이 고친다.
  - 아직 없는 페이지를 가리키는 메서드 `slug`는 링크 없이 보이고 `--links`에서 pending으로 나온다(빌드는 실패하지 않음).
- (A2 일치 점검 뒤 추가, A3–A4 작성·검증이 따름. 원문은 `notes/consistency/array-A2.md`의 W1–W7)
  - 입력 인자: 단항 `x`는 "<역할>. An `sd.ndarray` or a Python `bool`, `int`, `float` or `complex`."(받지 않는 꼴은 빼고, 거부는 raises에). 이항은 `x1`·`x2`를 따로 적고, `x2`는 "<역할>, in the same forms as `x1`. It broadcasts against `x1` (see [Broadcasting](/api/array/broadcasting/))." 받는 것은 "`sd.ndarray` 또는 파이썬 스칼라"이고 리스트는 받지 않음(실제로 돌려 확인). "`sd.ndarray` object(s)"라 쓰지 않음.
  - 낱말: 원소별 함수의 입력은 "operand", 셋 이상이거나 역할이 다르면 "argument"(`where`). "input array"는 쓰지 않음.
  - 반환: "<무엇>, with the shape of `x`." / "with the broadcast shape of the operands." + dtype 규칙 + "A Python scalar when `x` is a Python scalar."(이항: "when both operands are Python scalars"). dtype이 고정이면 "Always `float64`."
  - raises 문구: TypeError "If `x` is not an `sd.ndarray` or a Python scalar, or if `out` cannot hold the result dtype."(복소 거부는 앞에 "If `x` is complex, "; 결과가 `bool`이면 `out` 구절은 빼며, `bool`은 모든 dtype으로 바뀌어 TypeError가 나지 않음), "(a list, for example)" 같은 괄호는 붙이지 않음. ValueError "If `x` does not broadcast to the shape of `out`, or if `out` is read-only." OverflowError "If `x` is a Python `int` outside the `int64` range."(이항: "... is an operand and the promoted dtype is `bool` or `int64`.").
  - 실수 결과 규칙: "`bool` and `int64` input is converted to `float64` first (see [functions with floating-point results](/api/array/dtypes/#functions-with-floating-point-results))." 문장 중간 링크 글자는 소문자.
  - 정확도: "For `float64` input, the results are within a few units in the last place of the exact value." "correctly rounded"는 IEEE 정확 연산(`sqrt`, `1 / x`, 정확 지수의 `pow`)에만. ULP 숫자는 쓰지 않음.
  - NumPy 절: 결과가 NumPy와 마지막 자리에서 다르면(직접 재어 확인) "Results for `float64` input can differ from NumPy's in the last digit." 경고는 "<경우> give <값> without a warning. NumPy issues a `RuntimeWarning`"; R은 실제로 경고할 때만(`Rscript`로 확인) 'R warns "NaNs produced"'.
  - A2 slug(링크 가능): arithmetic, floor_divide-remainder, pow, abs, square-sqrt, sign, reciprocal, maximum-minimum, clip, copysign, hypot, nextafter, exp-log, logaddexp, trig, atan2, hyperbolic, deg2rad-rad2deg, rounding, comparison, logical, bitwise, isnan-isinf-isfinite, signbit, where, complex. NaN 판정은 `/api/array/isnan-isinf-isfinite/`, `bool` 결과 규칙은 `/api/array/dtypes/#arithmetic-on-booleans`(시프트는 `bool` 거부).
