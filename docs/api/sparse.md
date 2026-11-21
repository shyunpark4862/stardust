# sd.sparse API

> 상태: **초안** (SP2, 2026-11-21). 공개된 것은 §2의 형식과 변환(SP1: `sparray coo_array csr_array csc_array dok_array issparse matrix_transpose`와 그 메서드·속성)과 §3의 연산(SP2: 연산자, 원소별 메서드, 곱, 리덕션)이다. 나머지 시그니처는 scipy 1.18.1의 `scipy.sparse`(희소 **배열** 꼴)와 `scipy.sparse.linalg`의 이름과 인자 이름에 stardust의 규칙과 SP1–SP34의 결정을 입힌 안이고, 마일스톤 SP2–SP7에서 하나씩 공개된다(docs/design/sparse.md §8). 마일스톤에서 정할 것은 그 자리에 "(SP<n>에서 정한다)"로 적고 §10에 추천과 함께 모았다.

- 설계와 결정의 근거는 [docs/design/sparse.md](../design/sparse.md)(§1.1 SP1–SP47), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/sparse/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 클래스는 `name`), 클래스의 메서드(`Class.method(...)`)와 속성(`Class.attr`)이다. 모듈 수준의 이름이 계획한 공개 이름 전부이고, 공개된 것은 시그니처가 블록과 같아야 한다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- **scipy의 희소 배열 꼴만 둔다**(SP2): `coo_array`, `csr_array`, `csc_array`, `dok_array`. `*`는 원소별 곱, `@`는 행렬 곱이다(`sd.ndarray`와 같다). 옛 `spmatrix` 클래스(`csr_matrix` 등)와 그것을 돌려주는 옛 함수(`eye`, `identity`, `diags`, `spdiags`, `random`, `rand`, `bmat`)는 없다.
- **이름공간은 평평하다**(SP3): scipy.sparse.linalg의 함수도 하위 모듈 없이 `sd.sparse`에 있다: `sd.sparse.spsolve`, `sd.sparse.cg`, `sd.sparse.eigsh`, `sd.sparse.norm`, `sd.sparse.LinearOperator`. `sd.sparse.linalg`는 없다(`AttributeError`; 별칭을 두지 않는다). 두 scipy 모듈의 공개 이름은 서로 겹치지 않는다. 다른 영역과 같은 개념의 이름(`inv`, `expm`, `kron`, `tril`, `triu`, `matrix_transpose`)은 그 이름 그대로 둔다(DESIGN.md §1.4).
- **데이터는 위치로만, 옵션은 키워드로만** 받는다(API.md, 다른 영역과 같다). 데이터 인자의 이름은 scipy의 것이다(SP28: `A`, `b`, `B`, `x`, `blocks`). 메시지는 그 이름을 쓴다: `spsolve(): 'A' must be a sparse array, an sd.ndarray or a LinearOperator, not list`.
- **별칭이 없다**(SP24): `toarray`만(`todense` 없음), `conj`만(`conjugate` 없음), `@`만(`dot` 없음), `coords`만(COO의 `row`·`col` 없음). scipy의 `random_state`는 없고 무작위는 `rng`뿐이다.
- **반복 상한은 `max_iter`다**(optimize O14): 반복법의 scipy `maxiter`, `lsqr`의 `iter_lim`, `eigsh`·`eigs`·`svds`의 `maxiter`, `onenormest`의 `itmax`가 모두 `max_iter`다(`[scipy와 다름]`).
- `callback`, `show`, `check`, `calc_var`, `options`, `use_umfpack`, `permc_spec`, `diag_pivot_thresh`, `overwrite_*`, `solver`는 두지 않는다(SP5·SP4·SP6에서 정한다; 추천은 두지 않기, §10).
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1): `spsolve(): unexpected keyword argument 'permc_spec'`.

### §1.2 형식과 dtype (SP6–SP9)
- **형식**: COO, CSR, CSC, DOK. BSR·DIA는 첫 판에 없다(SP6; 나중 후보). 그래서 scipy에서 DIA를 돌려주는 `eye_array`·`diags_array`와 BSR을 돌려주는 `kron`은 `format=None`이면 CSR을 돌려준다(`[scipy와 다름]`; 값은 같고 `.format`만 다르다). `format=`에 `'dia'`·`'bsr'`·`'lil'`을 주면 `ValueError`다.
- **만들기용 형식은 DOK 하나다**(SP7). LIL(`lil_array`)은 없다. 행 단위로 만들 때는 `coo_array((data, (row, col)))`나 `vstack`을 쓴다.
- **값의 dtype**은 배열의 넷(`bool`, `int64`, `float64`, `complex128`)이고 승격 규칙도 배열과 같다(SP9, API.md §1). 받는 값은 `sd.asarray`의 규칙으로 바꾼다(docs/api/array.md §3.3): numpy의 정보 손실 없는 dtype(int8–int32, uint8–uint32, float16·float32, complex64)은 넓히고(`[scipy와 다름]`: scipy는 float32 등을 그대로 둔다), uint64·문자열·object는 `TypeError`다. 희소 선형대수는 float64와 complex128로 계산한다: bool·int64는 float64로 바꾼다(sd.linalg와 같다). 직접 풀이·반복법·고유값 모두 complex를 받는다.
- **색인의 dtype은 안팎 모두 int64다**(SP8). `indices`, `indptr`, `coords`는 int64 `sd.ndarray`다. scipy의 `get_index_dtype`처럼 크기에 따라 int32를 고르지 않는다. 받는 색인 배열은 정수 dtype이어야 하고(int32 등은 넓힌다; bool·float은 `TypeError`), scipy 객체의 int32 색인도 int64로 넓힌다.

### §1.3 표준 꼴과 0 (SP10–SP12)
- **CSR·CSC는 언제나 표준 꼴이다**(SP10): 줄마다 색인이 오름차순이고 중복이 없다. 모든 생성자와 연산의 결과가 그렇다. 그래서 CSR·CSC에는 `has_canonical_format`, `has_sorted_indices`, `sort_indices`, `sorted_indices`, `sum_duplicates`, `check_format`, `prune`이 없다(`[scipy와 다름]`). `(data, indices, indptr)`로 만들 때 범위를 검사하고(틀리면 `ValueError`), 정렬되지 않았거나 중복이 있으면 정렬하고 더한다(`copy=False`면 `ValueError`).
- **COO는 비표준 꼴을 가질 수 있다**(scipy와 같다). `has_canonical_format`과 `sum_duplicates()`는 COO에만 있다. `has_canonical_format`은 저장된 원소가 실제로 행 우선 정렬이고 중복이 없는지다(SP32; 처음 물을 때 알아내 둔다; `[scipy와 다름]`: scipy는 `(data, (row, col))`·전치·`astype` 뒤에 확인 없이 False로 둔다).
- **중복은 입력 순서대로 왼쪽부터 차례로 더한다**: `[1, 1e16, 1, -1e16]`(같은 자리)은 0이다. scipy의 `tocsr()`는 줄의 원소가 15개 이하일 때 이 규칙과 같고, 더 긴 줄과 COO의 `sum_duplicates()`(첫 원소 + 나머지의 `np.add.reduce`)는 순서가 다르다(반올림 크기의 차이, docs/design/sparse.md §2.3).
- **연산이 만든 0을 지우는지는 연산마다 scipy의 규칙이다**(SP11): 희소끼리의 `+`, `-`, `*`, 비교와 희소 × 희소 곱은 결과의 정확한 0을 지우고, 스칼라 곱(`A * 0`), `-A`, 원소별 함수는 패턴을 그대로 둔다. 표는 §3.1이다. 희소 × 희소는 두 쪽이 다 저장한 자리만 곱한다(아래의 강한 0).
- **암시적 0은 강한 0이다**(SP12): 저장되지 않은 원소는 산술에 들지 않는다. `A * nan`, `A * inf`는 저장된 원소만 NaN·inf가 되고, `A @ x`에서 x의 NaN은 그 열에 저장된 원소가 있는 행에만 퍼진다.
  - `A / c`는 저장된 원소만 나눈다. **c = 0이면 예외 없이 저장된 원소가 ±inf·NaN이 되고 암시적 0은 0이다**(`[scipy와 다름]`: scipy는 `ZeroDivisionError`; 배열의 원소별 연산은 값 때문에 실패하지 않는다, API.md §2).
  - 결과가 밀집인 연산(희소 ÷ 희소, 희소 ± 밀집)은 IEEE대로다(0/0은 NaN).

### §1.4 차원 (SP13)
- COO, CSR, DOK는 1차원과 2차원, CSC는 2차원이다(scipy 1.18.1과 같다). 3차원 이상과 배치는 없다. 희소 선형대수의 함수는 행렬 하나를 받고, 우변은 (n,)이나 (n, k)다.

### §1.5 결정성과 합의 순서 (SP15)
- **같은 입력이면 sd의 스레드 수, 저장 형식(CSR·CSC·COO)과 상관없이 같은 비트다**(API.md §3). 한 출력 원소의 항은 줄의 색인 오름차순이고, 4096개(L)까지는 0에서 시작하는 순차 합(곱의 합은 `fma`), 그보다 길면 L개씩의 조각을 순차로 더해 조각들을 pairwise 트리로 더한다. 전체 합(`sum()`)은 논리적 C 순서의 pairwise 합이다. 그래서 줄이 L 이하인 SpMV(`A @ x`)와 축 합은 scipy와 비트까지 같다.
- 직접 풀이는 Accelerate LU(AMD 순서)다(SP18). Accelerate의 스레드 모드(`VECLIB_MAXIMUM_THREADS`)가 같으면 같은 비트다(BLAS·LAPACK과 같은 규칙). 여러 Python 스레드에서 동시에 불러도 같다.
- 무작위를 쓰는 것(`random_array`, `eigsh`·`eigs`·`svds`의 시작 벡터, `onenormest`)은 정해진 키나 `rng`로 결정적이다(§4, §7, §8).

### §1.6 에러
- API.md §2의 종류와 꼴이다. 공개 함수의 에러는 `spsolve(): `로 시작하고, 메서드는 메서드 이름(`sum(): `), 생성자는 클래스 이름(`csr_array(): `)이다. 경고는 내지 않는다.
- scipy가 경고를 내는 곳(`SparseEfficiencyWarning`, `MatrixRankWarning`, 반복법의 수렴 실패)은 에러, 상태 필드, 또는 문서다. scipy의 `NotImplementedError`, `RuntimeError`, `ZeroDivisionError`, `ArpackError`, `ArpackNoConvergence`는 API.md §2의 목록의 예외로 바꾼다: 특이 행렬은 `sd.linalg.LinAlgError("spsolve(): singular matrix")`, 밀집이 될 연산(`A + c`, `A ** 0`)은 `ValueError`, 수렴 실패는 결과의 상태다.
- 희소 선형대수의 입력에 NaN·inf가 저장되어 있으면 `ValueError`다(sd.linalg와 같다, 저장된 원소만 본다).
- Accelerate는 행·열 수와 행 색인을 32비트로 받는다. 행이나 열이 2³¹ − 1을 넘는 행렬을 직접 풀이에 주면 `ValueError`다.

### §1.7 scipy와 다른 점 (모든 함수에 걸친 것)
- 배열 꼴만(SP2), 평평한 이름공간(SP3), BSR·DIA·LIL 없음과 `format=None`의 CSR(SP6, SP7), int64 색인(SP8), dtype 넷(SP9), 언제나 표준 꼴인 CSR·CSC(SP10), 예외 없는 `A / 0`(SP12), 강한 0의 희소 × 희소, IEEE대로의 비교, numpy 배열은 피연산자가 아님(SP35), 원소별 함수는 메서드만(SP36), 축 리덕션은 밀집(SP37), `A * 밀집`은 A의 형식(SP38), `A + c`·`A ** 0`은 `ValueError`(SP39), 리덕션의 `keepdims`(SP40), 경고 없음, `max_iter`, 데이터는 위치·옵션은 키워드, 별칭 없음, `rng` 필수(`random_array`)와 정해진 시작 벡터(`eigsh` 등).
- 직접 풀이의 바탕이 SuperLU가 아니라 Accelerate LU(문턱 부분 피벗 ≤ 0.5, AMD 순서, 척도 조정)다. 해의 끝자리와 역오차의 크기가 scipy와 다를 수 있고, `splu`의 결과에서 `L`·`U`·`perm_r`·`perm_c`를 꺼낼 수 없다(SP4에서 정한다).

## §2. 희소 배열 (SP1)
```
sparray
coo_array(arg, /, *, shape=None, dtype=None, copy=None)
csr_array(arg, /, *, shape=None, dtype=None, copy=None)
csc_array(arg, /, *, shape=None, dtype=None, copy=None)
dok_array(arg, /, *, shape=None, dtype=None, copy=None)
issparse(x, /)
matrix_transpose(A, /)
```
- 희소 배열은 `sd.ndarray` 필드를 가진 Python 클래스다(SP21). `sparray`는 네 형식의 바탕 클래스다(`isinstance(A, sd.sparse.sparray)`). 직접 만들지 않는다(`sparray(): the base class is not constructed directly; ...`, `TypeError`). `issparse(x)`는 `isinstance(x, sparray)`와 같다(scipy의 희소 객체는 False다).
- **생성자의 `arg`**(scipy와 같은 꼴):
  - 다른 희소 배열(형식 변환; `copy`를 따른다), scipy의 희소 배열과 옛 행렬(모든 형식; **언제나 복사한다**(SP23), int32 색인은 int64로; `copy=False`면 `ValueError`),
  - 밀집 `sd.ndarray`·numpy 배열·수의 시퀀스(1·2차원; 0이 아닌 원소만 저장한다. NaN은 저장되고 -0.0은 저장되지 않는다),
  - `(M, N)`이나 `(N,)`의 모양 tuple(빈 배열; `dtype=None`이면 float64),
  - COO·CSR·DOK·CSC 모두 `(data, (row, col))`(1차원은 `(data, (i,))`; scipy의 DOK는 받지 않는다)과 CSR·CSC의 `(data, indices, indptr)`. 각 배열은 `sd.ndarray`, numpy 배열이나 시퀀스이고, 색인은 정수여야 한다(아니면 `TypeError`). tuple을 가르는 규칙은 scipy와 같다: 정수만이면 모양, 둘째 원소가 배열의 tuple이면 좌표, CSR·CSC에서 원소 셋이 배열이면 압축된 부분이다.
  - `(data, indices, indptr)`는 검사한다(`ValueError`): `indptr`은 줄 수 + 1개이고 0에서 시작해 줄어들지 않고 저장 수에서 끝난다, 색인은 범위 안이다. 줄 안이 정렬되지 않았거나 중복이 있으면 정렬하고 입력 순서대로 더한다(§1.3).
- **색인 배열은 언제나 복사하고 검사한다**(SP30; `copy`와 상관없다): 받은 배열을 나중에 고쳐도 이 배열의 표준 꼴이 깨지지 않는다. 범위 밖은 `csr_array(): index 5 is out of bounds for axis 1 with size 4`(`ValueError`)다. `[scipy와 다름]`: scipy는 받은 색인 배열을 검사하지 않고 그대로 쓴다.
- `shape`: 없으면 색인의 최댓값 + 1이다(scipy와 같다; 빈 색인이면 `ValueError`). int 하나(1차원)나 int의 tuple이고, 입력의 모양과 다르면 `ValueError`다.
- `dtype`: 배열의 `dtype=` 규칙이다(dtype 객체나 그 이름; API.md §1). 주면 명시적 변환이다(float → int는 포화, NaN → 0). 밀집 입력은 0이 아닌 자리를 원래 값으로 고른 뒤에 바꾼다(scipy와 같다: 0이 된 값도 저장된 원소로 남는다).
- `copy`(SP27): 배열의 `copy=` 규칙으로 값 배열 `data`에 대한 것이다(None은 필요할 때만 복사, True는 언제나, False는 복사가 필요하면 `ValueError`; docs/api/array.md §1.8). 값 배열은 dtype이 맞고 연속이고, CSR·CSC로는 이미 표준 꼴이면 받은 것을 그대로 쓴다. 밀집 입력, 시퀀스, scipy 객체, DOK로의 변환, 정렬·중복 더하기는 복사다. 같은 형식의 희소 배열을 받으면 배열을 나누는 새 객체다(DOK는 해시 맵을 나눈다, scipy와 같다). scipy의 기본 `copy=False`("복사하지 않을 수 있으면")는 이쪽의 None과 같다.
- `matrix_transpose(A)`: 2차원의 전치(`A.T`와 같다). 1차원은 `ValueError`다(배열과 같다).

```
sparray.shape
sparray.ndim
sparray.size
sparray.nnz
sparray.dtype
sparray.format
sparray.T
sparray.mT
sparray.real
sparray.imag
coo_array.data
coo_array.coords
coo_array.has_canonical_format
csr_array.data
csr_array.indices
csr_array.indptr
csc_array.data
csc_array.indices
csc_array.indptr
sparray.toarray()
sparray.to_scipy(*, copy=None)
sparray.tocoo(*, copy=None)
sparray.tocsr(*, copy=None)
sparray.tocsc(*, copy=None)
sparray.todok(*, copy=None)
sparray.asformat(format, /, *, copy=None)
sparray.astype(dtype, /, *, copy=True)
sparray.copy()
sparray.transpose()
sparray.conj()
sparray.reshape(shape, /, *, copy=None)
sparray.resize(shape, /)
sparray.nonzero()
sparray.count_nonzero(*, axis=None)
coo_array.eliminate_zeros()
csr_array.eliminate_zeros()
csc_array.eliminate_zeros()
coo_array.sum_duplicates()
```
- `nnz`는 저장된 원소의 수(저장된 0 포함), `count_nonzero()`는 0이 아닌 값의 수다. `size`는 원소의 수(Python int, 2⁶³을 넘을 수 있다). `format`은 `'coo'`, `'csr'`, `'csc'`, `'dok'`이다.
- `data`는 쓸 수 있는 `sd.ndarray`이고(SP22; 값만 바꾸는 흔한 쓰임: `A.data *= 2`), 같은 모양의 `sd.ndarray`를 대입할 수 있다(dtype이 바뀌어도 된다, scipy와 같다; 다르면 `.data: the values must have shape (3,), got (2,)`). `indices`, `indptr`, `coords`는 **쓰기 금지 배열**이고 대입할 수 없다(바꾸면 표준 꼴이 깨진다). 읽을 때 복사하지 않는다.
- `.T`, `.mT`, `transpose()`: 2차원이면 같은 일이다. CSR의 `.T`는 같은 세 배열의 CSC 뷰다(scipy와 같다). 1차원의 `.T`·`.mT`는 `sd.ndarray`처럼 `ValueError`이고(SP31)(`.T: the array must have 2 dimensions, got 1`; `[scipy와 다름]`: scipy는 자신), `transpose()`는 자신이다.
- `real`, `imag`, `conj()`: 결과의 형식은 scipy의 것이다(COO·CSR·CSC는 같은 형식, DOK의 `real`·`imag`와 complex의 `conj()`는 CSR). 실수 dtype의 `real`은 값 배열을 나누고, `conj()`는 언제나 새 배열이다.
- `toarray()`: C 순서의 새 `sd.ndarray`이고 0에 저장된 값을 더한 것이다(scipy와 같다): 저장된 -0.0은 +0.0이 되고, COO의 중복은 저장 순서로 더해진다. `np.asarray(A)`와 `sd.asarray(A)`는 `toarray()`를 짚는 `TypeError`다(밀집으로 몰래 바뀌지 않는다; SP26). `len(A)`, `bool(A)`, `iter(A)`도 `TypeError`다(`[scipy와 다름]`: scipy의 DOK는 `len`이 저장 수, `bool`은 `ValueError`이고 (1, 1)이면 값).
- `to_scipy(copy=None)`(SP23): 같은 형식의 scipy 배열(`scipy.sparse.csr_array` 등)이다. 값·색인 배열은 복사하지 않는 numpy 뷰이고 색인 뷰는 읽기 전용이다(그래서 scipy의 제자리 메서드 `eliminate_zeros` 등은 그 객체에서 `ValueError`다). 값 뷰는 쓸 수 있고 이 배열과 나눈다. `copy=True`면 복사한다. DOK는 언제나 복사한다(`copy=False`면 `ValueError`).
- `tocoo`, `tocsr`, `tocsc`, `todok`, `asformat`(SP27, SP29): 같은 형식이면 자신(`copy=True`면 복사)이고, 아니면 새 객체다. 나눌 수 있는 배열은 나눈다(CSR → COO는 값과 열 색인을, 표준 꼴 COO → CSR은 값과 열 색인을). 복사가 필요한 변환(CSR ↔ CSC, DOK와의 변환, 정렬·중복 더하기)에 `copy=False`면 `ValueError`다(`tocsc(): copy=False, but converting csr to csc requires a copy`). `asformat(None)`은 자신이고, 그 밖의 형식 이름은 `ValueError`(`'format' must be 'coo', 'csr', 'csc', 'dok' or None, not 'lil'`)다. 1차원은 CSC가 될 수 없다(`ValueError`).
  - DOK에서 바꾸면 키를 정렬하므로 넣은 순서와 무관한 행 우선 순서다(`[scipy와 다름]`: scipy는 넣은 순서). CSC → COO는 열 우선 순서다(scipy와 같다).
- `astype`: 배열의 명시적 변환 규칙(float → int는 포화, NaN → 0; complex → 실수는 `TypeError`)이고 변환 뒤 0이 된 값도 저장된 원소로 남는다(scipy와 같다). 형식은 그대로이고, COO는 바꾼 뒤 중복을 더한다(scipy와 같다). `copy`는 배열의 `astype`과 같다(같은 dtype이면 `copy=True`만 복사, 다른 dtype에 `copy=False`면 `ValueError`). scipy의 `casting=`은 없다.
- `copy()`: 값이 이 배열과 메모리를 나누지 않는 같은 형식의 배열이다(쓰기 금지인 색인 배열은 나눈다).
- `reshape(shape, copy=None)`: 모양은 인자 하나이고(`A.reshape((3, 2))`, `A.reshape(6)`; -1 하나를 받는다; 배열과 같다) 1·2차원이다. C 순서다(`order=`는 없다). 결과는 모든 형식에서 COO다(scipy와 같다). 모양이 같으면 자신이다(`copy=True`면 복사). 값 배열은 나눈다(DOK는 복사라 `copy=False`면 `ValueError`).
- `resize(shape)`: 제자리에서 모양을 바꾸고 None을 돌려준다(scipy와 같다). 새 모양 밖의 원소는 버리고 새 행·열은 비어 있다. 차원 수는 바뀌지 않는다(`[scipy와 다름]`: scipy의 COO는 바꾼다).
- `nonzero()`: 0이 아닌 값의 자리, 축마다 int64 `sd.ndarray` 하나(쓸 수 있는 새 배열)다. 순서는 scipy와 같다: CSR·CSC·DOK는 행 우선, COO는 저장된 순서(중복도 그대로).
- `count_nonzero(axis=None)`: COO의 중복을 먼저 더해(배열은 바꾸지 않는다; `[scipy와 다름]`: scipy는 제자리에서 `sum_duplicates()`를 한다) 0이 아닌 값을 센다. `axis=None`이면 Python int, 0이나 1(음수 포함)이면 다른 축의 색인마다의 int64 `sd.ndarray`다(1차원은 int).
- `eliminate_zeros()`: 저장된 0(-0.0 포함)을 제자리에서 지운다(COO·CSR·CSC; scipy처럼 DOK에는 없다). `sum_duplicates()`: COO를 제자리에서 표준 꼴(행 우선, 중복은 입력 순서대로 더함)로 만든다.
- DOK에는 Python `dict`의 메서드(`keys`, `items`, `get`, `pop`, `update` …)가 없다(SP25). 원소는 `A[i, j]`, `A[i, j] = v`(1차원은 `A[i]`)로 읽고 쓴다: 축마다 int 하나(numpy의 정수 포함, bool 제외; 음수는 끝에서), 범위 밖은 `IndexError`, 읽으면 Python 스칼라(없는 자리는 그 dtype의 0), 쓰기는 배열의 원소 대입 규칙(암묵적 하향 변환은 `TypeError`)이고 0(-0.0 포함)을 넣으면 그 키를 지운다(scipy와 같다). 그 밖의 인덱싱은 §4다. 다른 형식에서 바꿀 때 저장된 0은 남는다(scipy와 같다).
- `repr`(와 `str`)는 한 줄이다: `<csr_array of shape (3, 4), float64, 5 stored elements>`.

## §3. 연산 (SP2)

### §3.1 연산자
| 연산 | 결과 | 0을 지우나 (scipy와 같게) |
|---|---|---|
| 희소 `+`·`-` 희소, 희소 `*` 희소(원소별), 비교(`==` `!=` `<` `<=` `>` `>=`), `maximum`·`minimum` | 희소 (CSR; 왼쪽이 CSC면 CSC) | 지운다 |
| 희소 `*`·`/` 스칼라, `-A`, `abs(A)`, 원소별 함수(§3.2), `A ** p`(p > 0) | 희소, 같은 패턴 | 지우지 않는다 |
| 희소 `*`·`/` 밀집, `A.multiply(dense)` | 희소, A의 형식(SP38; scipy는 COO) | 지우지 않는다 |
| 희소 비교·`maximum`·`minimum` 스칼라 | 희소 (CSR; A가 CSC면 CSC) | 지운다 |
| 희소 `+`·`-` 밀집, 희소 비교·`maximum`·`minimum` 밀집, 희소 `/` 희소 | 밀집 `sd.ndarray` | — |
| 희소 `@` 밀집, 밀집 `@` 희소 | 밀집 `sd.ndarray` (1차원 @ 1차원은 Python 스칼라) | — |
| 희소 `@` 희소 | 희소 (CSR; 왼쪽이 CSC면 CSC; 1차원 결과는 COO, scipy와 같다; 1차원 @ 1차원은 Python 스칼라) | 지운다 |
| `A + c`(c ≠ 0), `A - c`, `c - A`, `A ** 0` | `ValueError` (SP39; `adding a nonzero scalar to a sparse array is not supported; use A.toarray() + c`) | — |

- **값**: 저장된 값마다 배열의 같은 연산이라, 그 자리의 밀집 연산과 비트까지 같다. 결과 dtype은 배열의 승격 규칙이다(bool + bool과 bool @ bool은 int64, `-A`의 bool은 int64; scipy와 다르다). 정수 산술은 wrap한다(scipy·numpy와 같다).
- **희소끼리**: 모양이 같아야 한다(`shapes could not be broadcast together: (2, 3) (3, 2)`, `ValueError`; 브로드캐스트는 없다). 한쪽에만 저장된 자리는 다른 쪽의 암시적 0과의 연산이다. 단 `*`는 **두 쪽이 다 저장한 자리만**이다(강한 0, SP12: `nan * 암시적 0`은 0; `[scipy와 다름]`: scipy는 한쪽의 NaN·inf를 `a * 0` = NaN으로 저장한다). 결과의 형식은 scipy의 규칙이다: 왼쪽이 CSC면 CSC, 아니면 CSR(SP42; `[scipy와 다름]`: scipy의 `dok + 희소`는 DOK).
- **비교**는 IEEE대로다(NaN과의 비교는 `!=`만 참; `[scipy와 다름]`: scipy의 `<=`·`>=`는 `>`·`<`의 부정이라 NaN에서 참). 두 암시적 0 사이가 참인 비교(`A == B`, `A <= 1`, `A == 0`)는 나머지 모든 자리에 True를 저장한다(scipy와 같다, 경고 없음; 결과가 밀집에 가까우므로 큰 배열에서는 `!=`·`<`를 쓴다). 결과는 bool이고 True만 저장한다.
- **스칼라**는 Python의 bool·int·float·complex와 numpy의 스칼라다. `A * c`, `c * A`, `A / c`는 저장된 값에만 하고 패턴과 형식(DOK 포함; COO의 중복도)을 지킨다: `A * 0`은 패턴이 남고, `A / 0`은 저장된 값만 ±inf·NaN이다(SP12; `[scipy와 다름]`: scipy는 `ZeroDivisionError`이고 `A / c`를 `A * (1/c)`로 한다). `A + 0`, `A - 0`, `0 - A`는 같은 패턴의 새 배열이다. 스칼라와의 비교·`maximum`·`minimum`은 op(0, c)가 0이면 만든 0을 지우고, 아니면 모든 자리가 결과에 든다(`A.maximum(2)`는 `A.toarray()`의 maximum을 A의 형식으로 담는다; SP47, scipy와 같다, 경고 없음).
- **밀집 피연산자**는 `sd.ndarray`다(SP35). A의 모양으로 브로드캐스트되어야 한다(SP41)(행 `(n,)`·`(1, n)`, 열 `(m, 1)`, `(1, 1)`; 결과가 A보다 커지는 꼴은 `ValueError`). `A * D`, `D * A`, `A / D`는 저장된 자리의 D만 읽는다(암시적 0은 D가 NaN이어도 0, scipy와 같다). `A ± D`, 비교, `maximum`·`minimum`은 `A.toarray()`와의 배열 연산이다. `D / A`와 `c / A`는 `TypeError`다(scipy도 받지 않는다).
- **numpy 배열·리스트는 피연산자가 아니다**(SP35, API.md §5): 연산자는 `TypeError`(Python의 `unsupported operand type(s)`)이고 `==`·`!=`도 Python의 조용한 False가 아니라 `TypeError`다. `A @ 2`도 `TypeError`다(배열과 같다; scipy는 `ValueError`).
- **`@`**: 피연산자의 dtype은 배열의 `matmul` 규칙으로 승격한다. 모양이 맞지 않으면 배열의 메시지다(`the last axis of 'x1' (3) must match the second-to-last axis of 'x2' (2)`). 합의 순서는 §1.5다: SpMV·SpMM·`x @ A`(줄이 4096개 이하면)와 SpGEMM은 scipy와 비트까지 같다. SpGEMM의 결과는 언제나 표준 꼴이다(scipy의 결과는 줄이 정렬되지 않았다).
- 제자리 연산자(`A *= 2` 등)는 두지 않는다(SP43): Python이 `A = A * 2`로 새 배열을 만든다(`[scipy와 다름]`: scipy는 값 배열을 제자리에서 바꾼다; 제자리는 `A.data *= 2`).

### §3.2 원소별 함수와 이항 메서드
```
sparray.multiply(other, /)
sparray.maximum(other, /)
sparray.minimum(other, /)
sparray.power(n, /)
sparray.sin()
sparray.tan()
sparray.asin()
sparray.atan()
sparray.sinh()
sparray.tanh()
sparray.asinh()
sparray.atanh()
sparray.sqrt()
sparray.expm1()
sparray.log1p()
sparray.deg2rad()
sparray.rad2deg()
sparray.floor()
sparray.ceil()
sparray.trunc()
sparray.round()
sparray.sign()
```
- **메서드로만 둔다**(SP36). f(0) = 0인 것만이다(scipy의 메서드 목록의 함수). 이름은 배열의 것이다(SP46): `asin`·`atan`·`asinh`·`atanh`·`round`(`[scipy와 다름]`: scipy는 `arcsin`·`arctan`·`arcsinh`·`arctanh`·`rint`; 별칭은 없다, SP24). 값은 같은 이름의 배열 함수를 저장된 값에 쓴 것과 비트까지 같다(`A.sin().data`와 `sd.sin(A.data)`). 배열에 없는 `deg2rad`·`rad2deg`는 numpy의 식 x·(π/180), x·(180/π)다(complex는 `TypeError`). dtype과 정의역 밖의 값(sqrt(−1) = NaN)은 배열의 규칙이다.
- `sd.sin(A)` 등 배열의 원소별 함수에 희소 배열을 주면 메서드나 연산자를 짚는 `TypeError`다: `sin(): 'x' must be an sd.ndarray or a bool, int, float or complex, not csr_array; for a sparse array use A.sin()`, `add(): ...; for a sparse array use A + B`, 메서드가 없는 함수는 `cos(): ...; a sparse array has no cos: use cos(A.toarray())`.
- 결과의 형식은 scipy의 것이다: CSR·CSC·COO는 같은 형식이고 COO는 중복을 먼저 더한 표준 꼴(scipy는 A를 제자리에서 정리하고, 이쪽은 A를 바꾸지 않는다), DOK는 CSR이다(scipy의 `abs`·`power`와 같다). `abs(A)`, `power`, `**`도 같다.
- `power(n)`: n은 실수이고 n > 0이다. n = 0은 `ValueError`(SP39), n < 0과 NaN도 `ValueError`(SP44; 0 ** n이 0이 아니다), complex는 `TypeError`다. scipy의 `dtype=`은 없다(`astype`으로).
- `multiply`, `maximum`, `minimum`: 희소 배열, 밀집 `sd.ndarray`, 수를 받는다(§3.1의 규칙). 그 밖은 `multiply(): 'other' must be a sparse array, an sd.ndarray or a number, not list`(`TypeError`)다.

### §3.3 리덕션
```
sparray.sum(*, axis=None, dtype=None, keepdims=False)
sparray.mean(*, axis=None, keepdims=False)
sparray.max(*, axis=None, keepdims=False, explicit=False)
sparray.min(*, axis=None, keepdims=False, explicit=False)
sparray.nanmax(*, axis=None, keepdims=False, explicit=False)
sparray.nanmin(*, axis=None, keepdims=False, explicit=False)
sparray.argmax(*, axis=None, keepdims=False, explicit=False)
sparray.argmin(*, axis=None, keepdims=False, explicit=False)
```
- `axis`가 None이면 Python 스칼라, 있으면 **언제나 밀집 `sd.ndarray`**다(SP37; `[scipy와 다름]`: scipy의 `max(axis=)`·`min(axis=)`은 COO 희소). 1차원의 `axis=0`은 전체 리덕션이라 스칼라다(배열과 같다). `axis`는 int(음수 포함)나 None이다(SP45). `keepdims`는 배열과 같은 뜻이다(SP40; scipy에는 없다). scipy의 `out=`은 없다.
- 합의 순서는 §1.5다. 축 합은 scipy의 `ones @ A`·`A @ ones`와 비트까지 같다(scipy가 자기 줄을 따라 더하는 CSR의 `axis=1`, CSC의 `axis=0`은 `np.add.reduceat`이라 줄이 8개 이상이면 끝자리가 다를 수 있다). 전체 합은 C 순서의 pairwise 합이다(CSR이면 `sd.sum(A.data)`). COO의 중복은 먼저 더하므로 형식과 상관없이 같은 비트다.
- bool·int64의 합은 int64(wrap), `mean`은 정확한 합을 한 번 반올림해 한 번 나눈 float64다(배열과 같다). float·complex의 `mean`은 합 ÷ 원소 수다(`[scipy와 다름]`: scipy는 값마다 1/원소 수를 곱해 더한다). `sum(dtype=)`은 배열의 `sum`의 규칙이다. 빈 배열의 `sum`은 0, `mean`은 NaN(배열과 같다; scipy는 `ZeroDivisionError`), `max`·`argmax` 등은 `ValueError`(`max(): undefined for an empty array (or along an empty axis)`)다.
- `max`·`min`은 NaN을 전파하고(배열과 같다), 암시적 0이 하나라도 있으면 0도 후보다. `explicit=True`면 저장된 값만 본다(SP40, scipy와 같다; 빈 줄은 0). `nanmax`·`nanmin`은 NaN을 건너뛴다. complex는 배열처럼 `TypeError`다.
- `argmax`·`argmin`: 암시적 0도 자리로 센다. 동점이면 C 순서의 첫 자리, NaN이면 첫 NaN의 자리다(배열과 같다; `[scipy와 다름]`: scipy는 NaN이 있어도 첫 암시적 0의 자리). 2차원에서 `axis=None`이면 펼친 C 순서의 자리다. `explicit=True`면 저장된 값 중에서이고(빈 줄은 0), 아무것도 저장되지 않은 배열의 `axis=None`은 `ValueError`다(scipy와 같다).
- `trace`, `diagonal`은 SP3이다(§4.1).

## §4. 인덱싱과 만들기 (SP3)

### §4.1 인덱싱과 대입
- `A[i, j]`는 Python 스칼라, `A[i]`는 1차원 희소(scipy와 같다), 슬라이스·정수 배열·bool 마스크와 그 조합은 희소다. 범위 밖은 `IndexError`다(API.md §2).
- 대입 `A[idx] = v`: DOK는 O(1), CSR·CSC는 이미 저장된 자리면 값만 바꾸고 새 자리면 패턴을 다시 만든다(O(nnz); 경고 없음; SP3에서 정한다). 같은 자리가 여러 번 나오면 마지막 값이다(배열의 scatter와 같다).

```
sparray.trace(*, k=0)
sparray.diagonal(*, k=0)
sparray.setdiag(values, /, *, k=0)
eye_array(n_rows, n_cols=None, /, *, k=0, dtype=None, format=None)
diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None)
random_array(shape, /, *, density=0.01, format='coo', dtype=None, rng=None, data_sampler=None)
kron(A, B, /, *, format=None)
kronsum(A, B, /, *, format=None)
hstack(blocks, /, *, format=None, dtype=None)
vstack(blocks, /, *, format=None, dtype=None)
block_array(blocks, /, *, format=None, dtype=None)
block_diag(mats, /, *, format=None, dtype=None)
tril(A, /, *, k=0, format=None)
triu(A, /, *, k=0, format=None)
find(A, /)
```
- `eye_array`: 인자 이름과 기본값은 `sd.eye`의 것이다(`n_rows`, `n_cols`, `dtype=None`은 float64). scipy의 기본 형식 DIA 대신 CSR이다(SP6).
- `diags_array`: `format=None`이면 CSR(scipy는 DIA)이다. 정수 대각의 dtype은 int64다(scipy 1.18.1은 float64와 `FutureWarning`; SP3에서 정한다).
- `random_array`: `rng`에 `sd.random.Generator`를 꼭 받는다. 없으면 `random_array(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType`(`TypeError`; optimize O31과 같다). 자리는 비복원 추출로 정확히 round(density·m·n)개, 값은 `data_sampler(size)` 또는 [0, 1) 균등이다. 같은 `rng` 상태면 스레드 수와 상관없이 같은 비트다. scipy와 같은 값은 아니다(난수 생성기가 다르다).
- `kron`, `kronsum`, `hstack`, `vstack`, `block_array`, `block_diag`, `tril`, `triu`: `format=None`의 결과 형식은 scipy의 규칙이되 DIA·BSR이 나올 자리는 CSR이다(scipy는 `kron`이 BSR, `block_array`·`block_diag`·`tril`·`triu`가 COO; 어느 형식이든 값은 같다; SP3에서 정한다).
- `trace(k=)`: scipy의 `offset=`이 아니라 sd.linalg `trace`의 이름 `k`다(DESIGN.md §1.4; `[scipy와 다름]`). `diagonal(k=)`는 밀집 1차원 `sd.ndarray`다.
- `find(A)`: `(row, col, data)`의 세 밀집 `sd.ndarray`(표준 꼴의 행 우선 순서, 저장된 0은 뺀다; scipy와 같다).

## §5. 직접 풀이와 `LinearOperator` (SP4)
```
spsolve(A, b, /)
splu(A, /)
factorized(A, /)
inv(A, /)
spsolve_triangular(A, b, /, *, lower=True, unit_diagonal=False)
is_sptriangular(A, /)
spbandwidth(A, /)
LinearOperator(shape, matvec, /, *, rmatvec=None, matmat=None, rmatmat=None, dtype=None)
aslinearoperator(A, /)
SparseLU.solve(b, /, *, trans='N')
SparseLU.shape
LinearOperator.shape
LinearOperator.dtype
LinearOperator.T
LinearOperator.H
LinearOperator.matvec(x, /)
LinearOperator.rmatvec(x, /)
LinearOperator.matmat(X, /)
LinearOperator.rmatmat(X, /)
LinearOperator.adjoint()
```
- **바탕**은 Accelerate의 희소 LU(`SparseFactorizationLU`, 문턱 부분 피벗)이고 순서는 AMD로 고정한다(SP18). 대칭 행렬도 LU로 푼다: Accelerate의 Cholesky·LDLᵀ는 기본 스레드 모드에서 실행마다 비트가 달라서 첫 판에 쓰지 않는다.
- `spsolve(A, b)`: A는 정방 희소 배열(CSR·CSC는 그대로, 다른 형식은 CSC로 바꾼다). b가 (n,)이면 (n,), (n, k)이면 (n, k) 밀집, 희소면 희소다(scipy와 같다). 0 × 0이면 빈 결과다(Accelerate는 받지 않으므로 앞에서 답한다).
- **특이**: 구조적으로 특이하면 `LinAlgError("spsolve(): singular matrix")`다. 값으로 특이한 행렬은 Accelerate가 성공(status 0)을 돌려주고 ±inf나 크기 1/ε의 유한한 해를 낸다(docs/design/sparse.md §7). 그것을 무엇으로 판정할지는 SP4에서 정한다(§10). `[scipy와 다름]`: scipy `spsolve`는 `MatrixRankWarning`과 NaN, `splu`는 `RuntimeError("Factor is exactly singular")`다.
- `splu(A)`: 내보내지 않는 분해 클래스(`SparseLU`, 이름은 SP4에서 정한다)를 돌려준다. `solve(b, trans='N' | 'T' | 'H')`, `shape`. scipy `SuperLU`의 `L`, `U`, `perm_r`, `perm_c`, `nnz`는 Accelerate가 인수를 내보이지 않아 줄 수 없다. `factorized(A)`는 `splu(A).solve`와 같은 호출 가능한 객체다.
- 우변이 여럿이면 Accelerate가 열을 함께 푼다. 한 열씩 푼 것과 반올림 크기만큼 다르다(SP4에서 정한다).
- `inv(A)`: 단위 행렬을 우변으로 푼 희소 결과(0은 지운다; scipy와 같다). 대개 밀집에 가깝다.
- `spsolve_triangular`: 이 라이브러리의 대입 커널(합은 §1.5의 줄 합). 대각의 0은 `LinAlgError("spsolve_triangular(): singular matrix")`다. scipy의 `overwrite_A`, `overwrite_b`는 없다.
- `is_sptriangular(A)`: `(lower, upper)`의 bool 둘, `spbandwidth(A)`: `(below, above)`의 int 둘(scipy와 같다). 패턴만 본다.
- `LinearOperator`: Python 클래스. `shape`과 `matvec`은 위치, 나머지는 키워드다(scipy는 `dtype`·`rmatvec`도 위치로 받는다; SP4에서 정한다). `@`, `+`, `-`, 스칼라 `*`, `**`(정수)를 받는다. scipy처럼 하위 클래스에서 `_matvec`을 정의하는 꼴도 받는다. `aslinearoperator(A)`는 희소 배열, 밀집 `sd.ndarray`, `LinearOperator`를 받고, 희소·밀집에서 만든 연산자는 C++ 커널을 바로 쓴다.
- `sd.linalg`의 함수에 희소 배열을 주면 쓸 함수를 짚는 `TypeError`다: `solve(): 'a' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.spsolve`(대응은 docs/design/sparse.md §3.10).

## §6. 반복법 (SP5)
```
cg(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, max_iter=None, M=None)
gmres(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, restart=None, max_iter=None, M=None)
minres(A, b, /, *, x0=None, rtol=1e-05, shift=0.0, max_iter=None, M=None)
bicgstab(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, max_iter=None, M=None)
lsqr(A, b, /, *, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, max_iter=None, x0=None)
lsmr(A, b, /, *, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, max_iter=None, x0=None)
```
- **이 라이브러리가 scipy 1.18.1의 알고리즘과 멈춤 판정으로 짠다**(SP16): `cg`·`bicgstab`은 ‖r‖ ≤ max(rtol·‖b‖, atol), `gmres`는 재시작 GMRES(`restart` 기본 20, `max_iter`는 scipy의 `maxiter`처럼 바깥 반복), `minres`는 Paige–Saunders(`shift`), `lsqr`은 Paige–Saunders 1982, `lsmr`은 Fong–Saunders 2011(`damp`, `atol`·`btol`·`conlim`). 반복 수와 상태는 scipy와 같은 것을 목표로 한다. Accelerate의 반복법은 쓰지 않는다(멈춤 판정이 처음 잔차에 대한 상대이고 반복 수를 돌려주지 않는다).
- 입력 `A`, `M`(전처리기)은 희소 배열, 밀집 `sd.ndarray`, `LinearOperator`다. 희소·밀집이면 반복 전체가 C++이고, Python 함수의 `LinearOperator`는 반복마다 Python 콜백이다.
- `max_iter=None`의 상한은 scipy의 것이다: `cg`·`bicgstab`은 10n, `gmres`는 바깥 반복 10n(`restart`는 min(20, n)), `minres`는 5n, `lsqr`은 2n(n은 열 수), `lsmr`은 min(m, n). 상한에 닿으면 마지막 값과 상태를 돌려준다.
- **결과**: scipy의 튜플 대신 읽기 전용 결과 객체이고 scipy가 돌려주는 개수와 순서로 풀린다(`x, info = cg(A, b)`; `lsqr`은 열, `lsmr`은 여덟). 필드는 `x`, `converged`, `status`(scipy의 `info`·`istop`), `n_iter`와 `lsqr`·`lsmr`의 `normr`, `normar`, `norma`, `conda`, `normx`다(이름과 풀기는 SP5에서 정한다). 배열 필드는 쓰기 금지다(API.md §2.2).
- scipy의 `callback`, `callback_type`, `show`, `check`, `calc_var`은 없다(SP5에서 정한다).

## §7. 고유값과 특이값 (SP6)
```
eigsh(A, /, *, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, max_iter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal', rng=None)
eigs(A, /, *, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, max_iter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None, rng=None)
svds(A, /, *, k=6, ncv=None, tol=0, which='LM', v0=None, max_iter=None, return_singular_vectors=True, rng=None)
```
- Accelerate에 ARPACK이 없어 이 라이브러리가 짠다: `eigsh`는 implicitly restarted Lanczos, `eigs`는 implicitly restarted Arnoldi(알고리즘을 ARPACK의 계산 순서까지 옮길지는 SP6에서 정한다), `svds`는 Golub–Kahan–Lanczos 이중대각화(scipy의 `solver=` 없음; SP6에서 정한다). `sigma`(shift-invert)는 §5의 LU다.
- `v0=None`이면 정해진 시작 벡터(sd.random의 고정 키)를 쓰고 `rng=`로 바꿀 수 있다(scipy는 시드 없는 무작위; SP6에서 정한다).
- 결과는 scipy처럼 `(w, v)`(`return_eigenvectors=False`면 `w`), `svds`는 `(u, s, vh)`다. 수렴하지 못하면 찾은 쌍만 돌려주고 상태를 알린다(scipy의 `ArpackNoConvergence` 대신; 꼴은 SP6에서 정한다).
- 실수 입력의 `eigs` 결과는 scipy처럼 complex다.

## §8. 행렬 함수와 노름 (SP7)
```
expm(A, /)
expm_multiply(A, B, /, *, start=None, stop=None, num=None, endpoint=None, traceA=None)
onenormest(A, /, *, t=2, max_iter=5, compute_v=False, compute_w=False)
norm(x, /, *, ord=None, axis=None)
```
- `expm`: Al-Mohy–Higham(2009)의 크기 조정과 제곱(scipy와 같은 알고리즘; 밀집 `sd.linalg.expm`과 같은 꼴의 계약). 결과는 희소다(형식은 SP7에서 정한다; scipy는 CSC).
- `expm_multiply`: Al-Mohy–Higham(2011). 매개변수와 멈춤(2⁻⁵³)은 scipy와 같다.
- `onenormest`: Higham–Tisseur(2000). 무작위 ±1 열은 고정 키에서 와서 결정적이다(SP7에서 정한다). 반복 상한은 scipy의 `itmax`가 아니라 `max_iter`다(§1.1).
- `norm`(SP14): scipy의 뜻(행렬: `'fro'`, 1, −1, inf, −inf, 2; 벡터 축: 정수 p, inf, −inf, 0). `ord`의 규칙과 메시지는 `sd.linalg.matrix_norm`·`vector_norm`의 것이다: 행렬이면 `norm(): 'ord' must be 'fro', 'nuc', 1, -1, 2, -2, inf or -inf, not 3`(`ValueError`), 벡터 축이면 `norm(): 'ord' must be an int or a float, not str`(`TypeError`). scipy에 없는 행렬의 `'nuc'`·`-2`는 SP7에서 정한다(§10). `ord=2`는 `svds`로 구한다. 합의 순서는 §1.5다. scipy처럼 `ord`가 위치 인자인 꼴은 받지 않는다(키워드만).

## §9. scipy 이름 → sd 이름

### §9.1 scipy.sparse
| scipy.sparse | sd.sparse | 마일스톤·결정 |
|---|---|---|
| `coo_array`, `csr_array`, `csc_array`, `dok_array`, `sparray`, `issparse` | 같은 이름 | SP1 |
| `lil_array` | 없음(DOK, COO 꼴의 만들기, `vstack`) | SP7 |
| `bsr_array`, `dia_array` | 없음, 나중 후보 | SP6 |
| `coo_matrix` 등 `*_matrix`, `spmatrix`, `isspmatrix*` | 없음 | SP2 |
| `eye_array`, `diags_array`, `random_array`, `block_array` | 같은 이름(기본 형식 CSR) | SP3 |
| `eye`, `identity`, `diags`, `spdiags`, `random`, `rand`, `bmat` | 없음(옛 `spmatrix`를 돌려주는 함수) | SP2 |
| `kron`, `kronsum`, `hstack`, `vstack`, `block_diag`, `tril`, `triu`, `find` | 같은 이름 | SP3 |
| `matrix_transpose` | 같은 이름 | SP1 |
| `expand_dims`, `swapaxes`, `permute_dims` | 없음(2차원까지라 쓸 일이 없다) | SP13 |
| `save_npz`, `load_npz` | 나중 후보 | — |
| `get_index_dtype`, `safely_cast_index_arrays` | 없음(색인은 언제나 int64) | SP8 |
| `SparseWarning`, `SparseEfficiencyWarning` | 없음(경고가 없다) | — |
| `csgraph` | 없음: 나중의 따로 된 그래프 영역 | SP1 |
| `sparsetools`, `sputils`, 형식 모듈들(`csr`, `coo` …) | 없음(비공개) | — |
| 메서드 `todense`, `conjugate`, `dot` | 없음(`toarray`, `conj`, `@`) | 별칭 없음 |
| 메서드 `sort_indices`, `sorted_indices`, `has_sorted_indices`, `check_format`, `prune`(CSR·CSC) | 없음(언제나 표준 꼴) | SP10 |
| 메서드 `tobsr`, `todia`, `tolil` | 없음 | SP6, SP7 |
| `trace(offset=)` | `trace(k=)` | DESIGN.md §1.4 |
| `sum(axis=, dtype=, out=)` 등의 `out=` | 없음; `keepdims=`를 더한다 | §3.3, SP40 |
| `max(axis=)`·`min(axis=)`·`nanmax`·`nanmin`(COO 희소) | 밀집 `sd.ndarray` | SP37 |
| `A.multiply(dense)`(COO) | A의 형식 | SP38 |
| `np.sin(A)` 꼴의 ufunc | 없음; `A.sin()` 등 메서드 | SP36 |
| 메서드 `arcsin`, `arctan`, `arcsinh`, `arctanh`, `rint` | `asin`, `atan`, `asinh`, `atanh`, `round`(배열의 이름; 별칭 없음) | SP46 |
| `power(n, dtype=)` | `power(n)` | §3.2 |
| `__round__`(`round(A)`), 제자리 `*=`·`/=` | 없음(`A.round()`, `A.data *= c`) | §3.1, SP43 |

### §9.2 scipy.sparse.linalg
| scipy.sparse.linalg | sd.sparse | 마일스톤·결정 |
|---|---|---|
| `spsolve`, `splu`, `factorized`, `inv`, `spsolve_triangular` | 같은 이름(Accelerate LU/AMD, 삼각은 이 라이브러리) | SP4, SP18 |
| `SuperLU` (`splu`의 결과 형) | 다른 이름의 내보내지 않는 클래스 | SP4 |
| `is_sptriangular`, `spbandwidth` | 같은 이름 | SP4 |
| `LinearOperator`, `aslinearoperator` | 같은 이름 | SP4 |
| `cg`, `gmres`, `minres`, `bicgstab`, `lsqr`, `lsmr` | 같은 이름(`maxiter`·`iter_lim` → `max_iter`) | SP5, SP16 |
| `eigsh`, `eigs`, `svds` | 같은 이름(`maxiter` → `max_iter`) | SP6 |
| `expm`, `expm_multiply`, `norm`, `onenormest` | 같은 이름(`onenormest`의 `itmax` → `max_iter`) | SP7, SP14 |
| `ArpackError`, `ArpackNoConvergence` | 없음(결과의 상태) | SP6 |
| `MatrixRankWarning` | 없음(`LinAlgError`) | SP4 |
| `use_solver` | 없음 | SP4 |
| `spilu`, `lobpcg`, `bicg`, `cgs`, `qmr`, `tfqmr`, `lgmres`, `gcrotmk`, `matrix_power`, `funm_multiply_krylov`, `LaplacianNd` | 나중 후보 | — |
| 하위 모듈 `dsolve`, `isolve`, `eigen`, `interface`, `matfuncs` | 없음 | SP3 |
| `scipy.sparse.linalg.X` 전체 | `sd.sparse.X` (`sd.sparse.linalg`는 없다) | SP3 |

## §10. 마일스톤별 남은 질문 (추천을 첫째에)
docs/design/sparse.md §9와 같은 목록이다. 결정이 나면 이 문서와 그 문서를 함께 고친다.

- **SP0**: 정했다(SP6–SP20, docs/design/sparse.md §1.1).
- **SP1**: 정했다(SP21–SP34, docs/design/sparse.md §1.1).
- **SP2**: 정했다(SP35–SP47, docs/design/sparse.md §1.1).
- **SP3**
  1. 만들기 함수의 이름: `eye_array`·`diags_array`·`random_array`·`block_array`(추천) / 짧은 이름.
  2. `random_array`의 `rng`: 꼭 받는다(추천) / 없으면 고정 시드.
  3. CSR·CSC의 새 자리 대입: 허용하고 O(nnz)(추천) / `ValueError`로 막고 DOK를 짚는다.
  4. `format=None`의 결과 형식: scipy의 규칙에서 DIA·BSR만 CSR로(추천; `kron`은 CSR, `block_array`·`tril`은 COO) / 모든 만들기 함수가 CSR.
  5. 정수 대각의 `diags_array`: int64(추천, 배열의 dtype 규칙) / scipy처럼 float64.
- **SP4**
  1. LU의 순서: AMD 고정(추천; Metis는 동시 호출에서 비트가 다르고 MTMetis는 동시 호출에서 프로세스가 죽는다) / `ordering=` 옵션.
  2. `splu`의 결과: 내보내지 않는 `SparseLU`(`solve`, `shape`; `L`·`U` 없음)(추천) / 이 라이브러리의 LU(나중 후보)를 기다린다.
  3. scipy의 옵션(`permc_spec`, `diag_pivot_thresh`, `use_umfpack` 등): 두지 않는다(추천) / 받되 `ValueError`.
  4. 특이의 판정: 구조적 특이 + 유한하지 않은 해 + 분해의 조건수 추정(`onenormest`로 ‖A⁻¹‖₁을 어림해 κ₁·ε ≥ 1이면 `LinAlgError`)(추천; 정확히 특이한 행렬이 Accelerate에서 크기 1/ε의 유한한 해를 내기 때문, docs/design/sparse.md §7) / 구조적 특이 + 유한하지 않은 해만(scipy와 비슷하게 큰 유한한 해를 돌려준다).
  5. 반복 다듬기: 잰 역오차가 계약을 넘는 행렬이 있으면 한 번(추천) / 넣지 않는다.
  6. 우변이 여럿: Accelerate가 함께 푼다(추천; 결과는 k와 값으로 정해지고 결정적) / 열마다 따로 풀어 한 열의 비트가 k와 무관하게.
  7. `LinearOperator`의 위치 인자: `shape`, `matvec`만(추천) / scipy처럼 `rmatvec`, `matmat`, `dtype`도.
- **SP5**
  1. 결과: scipy의 튜플 개수로 풀리는 읽기 전용 결과 객체(추천) / 튜플.
  2. 이름: `max_iter`(추천, optimize O14) / scipy의 `maxiter`·`iter_lim`.
  3. `callback`: 두지 않는다(추천, optimize O37·O54) / 둔다.
  4. `gmres`의 `restart`·`maxiter`의 뜻: scipy 1.18.1 그대로(추천).
- **SP6**
  1. 알고리즘: ARPACK을 계산 순서까지 옮긴다(추천) / Krylov–Schur.
  2. `v0=None`: 정해진 시작 벡터, `rng=`로 바꿀 수 있다(추천) / `v0`나 `rng`를 꼭 받는다.
  3. `svds`: Golub–Kahan–Lanczos 하나, `solver=` 없음(추천) / scipy처럼 `'arpack'`(AᵀA).
  4. `eigsh`의 `mode='buckling'`·`'cayley'`: 넣는다(추천) / `'normal'`만.
  5. 수렴 실패: 결과 객체의 상태(추천) / `LinAlgError`.
  6. R의 `RSpectra`·`irlba`: 이 마일스톤에서 묻고 설치한다(추천).
- **SP7**
  1. `onenormest`의 무작위 열: 고정 키로 결정적(추천) / `rng`를 받는다.
  2. `expm`의 결과 형식: 입력의 형식(추천) / scipy처럼 CSC.
  3. `onenormest`의 `itmax` → `max_iter`(추천, §1.1) / scipy의 이름.
  4. `norm`의 행렬 `ord='nuc'`·`-2`(scipy에 없고 모든 특이값이 필요하다): 받지 않고 그 값의 `ValueError`(추천; 메시지의 목록은 linalg의 것에서 둘을 뺀다) / `svds`로 받는다.
- **SP8**: 범위는 모든 공개 함수를 크기마다 scipy·R과 견주는 것(추천). 코드 리뷰를 사용자가 따로 하는지 그때 묻는다.
