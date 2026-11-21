# sd.sparse API

> 상태: **초안** (SP5, 2026-11-21). 공개된 것은 §2의 형식과 변환(SP1: `sparray coo_array csr_array csc_array dok_array issparse matrix_transpose`와 그 메서드·속성), §3의 연산(SP2: 연산자, 원소별 메서드, 곱, 리덕션), §4의 인덱싱과 만들기(SP3: 인덱싱과 대입, `trace diagonal setdiag`, `eye_array diags_array random_array kron kronsum hstack vstack block_array block_diag tril triu find`), §5의 직접 풀이와 `LinearOperator`(SP4: `spsolve splu factorized inv spsolve_triangular is_sptriangular spbandwidth LinearOperator aslinearoperator`, `splu`의 결과 `SparseLU`), §6의 반복법(SP5: `cg gmres minres bicgstab lsqr lsmr`와 그 결과 객체)이다. 나머지 시그니처는 scipy 1.18.1의 `scipy.sparse`(희소 **배열** 꼴)와 `scipy.sparse.linalg`의 이름과 인자 이름에 stardust의 규칙과 SP1–SP34의 결정을 입힌 안이고, 마일스톤 SP2–SP7에서 하나씩 공개된다(docs/design/sparse.md §8). 마일스톤에서 정할 것은 그 자리에 "(SP<n>에서 정한다)"로 적고 §10에 추천과 함께 모았다.

- 설계와 결정의 근거는 [docs/design/sparse.md](../design/sparse.md)(§1.1 SP1–SP82), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/sparse/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 클래스는 `name`), 클래스의 메서드(`Class.method(...)`)와 속성(`Class.attr`)이다. 모듈 수준의 이름이 계획한 공개 이름 전부이고, 공개된 것은 시그니처가 블록과 같아야 한다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- **scipy의 희소 배열 꼴만 둔다**(SP2): `coo_array`, `csr_array`, `csc_array`, `dok_array`. `*`는 원소별 곱, `@`는 행렬 곱이다(`sd.ndarray`와 같다). 옛 `spmatrix` 클래스(`csr_matrix` 등)와 그것을 돌려주는 옛 함수(`eye`, `identity`, `diags`, `spdiags`, `random`, `rand`, `bmat`)는 없다.
- **이름공간은 평평하다**(SP3): scipy.sparse.linalg의 함수도 하위 모듈 없이 `sd.sparse`에 있다: `sd.sparse.spsolve`, `sd.sparse.cg`, `sd.sparse.eigsh`, `sd.sparse.norm`, `sd.sparse.LinearOperator`. `sd.sparse.linalg`는 없다(`AttributeError`; 별칭을 두지 않는다). 두 scipy 모듈의 공개 이름은 서로 겹치지 않는다. 다른 영역과 같은 개념의 이름(`inv`, `expm`, `kron`, `tril`, `triu`, `matrix_transpose`)은 그 이름 그대로 둔다(DESIGN.md §1.4).
- **데이터는 위치로만, 옵션은 키워드로만** 받는다(API.md, 다른 영역과 같다). 데이터 인자의 이름은 scipy의 것이다(SP28: `A`, `b`, `B`, `x`, `blocks`). 메시지는 그 이름을 쓴다: `spsolve(): 'A' must be a sparse array, an sd.ndarray or a LinearOperator, not list`.
- **별칭이 없다**(SP24): `toarray`만(`todense` 없음), `conj`만(`conjugate` 없음), `@`만(`dot` 없음), `coords`만(COO의 `row`·`col` 없음). scipy의 `random_state`는 없고 무작위는 `rng`뿐이다.
- **반복 상한은 `max_iter`다**(optimize O14): 반복법의 scipy `maxiter`, `lsqr`의 `iter_lim`, `eigsh`·`eigs`·`svds`의 `maxiter`, `onenormest`의 `itmax`가 모두 `max_iter`다(`[scipy와 다름]`).
- `callback`, `show`, `check`, `calc_var`, `options`, `use_umfpack`, `permc_spec`, `diag_pivot_thresh`, `overwrite_*`, `solver`는 두지 않는다(직접 풀이의 것은 SP61, 반복법의 것은 SP74로 정했다; 고유값의 것은 SP6에서 정한다, 추천은 두지 않기, §10).
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
- 직접 풀이의 바탕이 SuperLU가 아니라 Accelerate LU(문턱 부분 피벗, AMD 순서, 척도 조정)와 반복 다듬기 한 번이다. 해의 끝자리와 역오차의 크기가 scipy와 다르고, `splu`의 결과(`SparseLU`)에서 `L`·`U`·`perm_r`·`perm_c`·`nnz`를 꺼낼 수 없다(SP60). 특이의 판정에 조건수 추정이 있어(SP62) scipy가 큰 해를 돌려주는 행렬(κ₁ ≥ 1/ε)이 `LinAlgError`다.
- 반복법은 이 라이브러리의 커널이 scipy의 알고리즘을 그대로 따른다(SP16). 결과가 scipy의 튜플로 풀리는 결과 객체이고(SP72) `callback`이 없으며(SP74), 내적과 노름의 반올림 순서가 달라 반올림에 민감한 문제에서 반복 수가 scipy와 다를 수 있다(§6).

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
- **키는 배열의 규칙이다**(docs/api/array.md §5): int(numpy 정수 포함, bool 제외; 음수는 끝에서), slice(음수 step 포함), `...`, None, 정수 배열과 bool 마스크(`sd.ndarray`, 중첩 리스트·튜플, numpy 배열 같은 버퍼; 반복과 아무 순서), 그 조합. 범위 밖(`index 5 is out of bounds for axis 1 with size 4`), 틀린 객체, 축보다 많은 인덱스, 브로드캐스트되지 않는 배열, 맞지 않는 마스크, 떨어진 고급 인덱스(`A[[0, 1], None, [0, 1]]`)는 배열과 같은 종류와 메시지의 예외다. 희소 배열은 인덱스가 아니다(`IndexError`). 결과가 3차원 이상인 키(`A[None, :]`, 2차원 정수 배열 하나)는 `IndexError`다(`this index gives a result of 3 dimensions; a sparse array has 1 or 2`).
- **결과**: 모든 축이 int면 Python 스칼라다(저장된 값, 없으면 그 dtype의 0; COO의 중복은 저장 순서로 더한 값). 아니면 희소 배열이고 `A[key].toarray()`는 `A.toarray()[key]`와 비트까지 같다. 결과의 차원 수가 A와 같으면 A의 형식이고(CSR의 `A[rows]`, `A[:, cols]`, `A[1:3, ::2]`는 CSR, DOK는 DOK), 다르면 COO다(2차원의 `A[i]`·`A[:, j]`, 원소 고르기, 1차원의 None; scipy와 같다). `[scipy와 다름]`(SP54): scipy의 CSR·CSC는 원소 고르기 `A[rows, cols]`와 2차원 마스크에 밀집 numpy 배열(2차원 결과는 밀집 값에서 다시 만든 배열)을 돌려준다.
- 두 축에 배열이 있으면 원소 고르기(정수와 배열도 함께 브로드캐스트; 배열의 규칙), 아니면 축마다 고른 것의 곱이다. 고른 자리의 저장된 원소가 결과에 저장된다(저장된 0도). COO는 중복과 저장 순서를 그대로 가져간다(scipy와 같다). `[scipy와 다름]`: scipy의 DOK는 대부분의 경로에서 저장된 0을 뺀다.
- **대입** `A[key] = value`: 고른 영역의 모양에 배열의 대입 규칙으로 맞춘다(docs/api/array.md §5.3). 원소 하나의 자리에는 Python 스칼라만(`only a bool, int, float or complex can be assigned to a single element, not list`; `[scipy와 다름]`: scipy는 `[3]`도 받는다), 그 밖은 영역의 모양으로 엄격하게 브로드캐스트되는 값(`sd.ndarray`, 시퀀스, 버퍼; `cannot broadcast a value of shape (1, 4) to the indexed shape (4,)`)이나 희소 배열이다. 값은 A의 dtype으로 배열의 규칙대로 바뀐다(암묵적 하향 변환은 `TypeError`; `[scipy와 다름]`: scipy는 2.5를 int64에 2로 넣는다). 같은 자리가 여러 번이면 마지막 값이다(배열과 같다; `[scipy와 다름]`: scipy의 COO는 첫 값). 대입의 결과는 `A.toarray()`에 같은 대입을 한 것과 같다.
- **저장되는 것은 형식마다 scipy의 규칙이다**(SP53):
  - CSR·CSC: 이미 저장된 자리는 값 배열 `data`를 제자리에서 바꾼다(그 값 배열을 나누는 다른 배열·`.T`·`to_scipy()` 뷰에도 보인다; scipy와 같다). 새 자리가 있으면 배열을 다시 만든다(O(nnz), 경고 없음; SP50). 대입한 0도 저장된다(scipy와 같다). 희소 값이면 영역 안에 저장된 자리를 0으로 바꾸고 값의 저장된 원소를 넣는다.
  - DOK: 0(-0.0 포함)을 넣으면 그 키를 지운다(희소 값은 그 밀집 값으로).
  - COO: 영역의 원소(중복 포함)를 지우고 0이 아닌 새 값을 행 우선으로 뒤에 붙인다(원소 하나의 대입과 희소 값의 저장된 원소는 0이어도 붙인다).

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
- `diagonal(k=)`: k번째 대각(원소 `A[i, i + k]`)의 밀집 1차원 `sd.ndarray`이고 `sd.linalg.diagonal(A.toarray(), k=k)`와 비트까지 같다(대각이 밖이면 길이 0). `trace(k=)`는 그 대각의 `sd.sum`인 Python 스칼라다(bool·int64는 int64, SP55; float64는 `sd.linalg.trace(A.toarray(), k=k)`와 비트까지 같다). 인자 이름은 scipy의 `offset=`이 아니라 sd.linalg의 `k`다(DESIGN.md §1.4; `[scipy와 다름]`). 2차원만이다(1차원은 `ValueError`).
- `setdiag(values, k=)`: 제자리에서 대각을 바꾸고 None을 돌려준다. values가 수면 대각 전체, 1차원이면 앞의 `len(values)`개다(길면 자른다; scipy와 같다). 저장 규칙은 대입과 같다(CSR·CSC·COO는 0도 저장, DOK는 지움). k가 모양 밖이면 `ValueError`(`setdiag(): k 3 exceeds the array's dimensions (3, 3)`)다.
- **데이터 인자**(`A`, `B`, 블록)는 희소 배열이나 밀집 `sd.ndarray`다(밀집은 0이 아닌 값만 저장한다). 리스트와 numpy 배열은 `TypeError`다(배열의 함수와 같다, API.md §5; `kron(): 'B' must be a sparse array or an sd.ndarray, not list`).
- **`format=None`의 결과 형식**(SP51): scipy의 규칙이되 DIA·BSR이 나올 자리는 CSR이다. `eye_array`·`diags_array`·`kron`은 CSR(scipy는 DIA·BSR), `kronsum`은 CSR, `hstack`·`vstack`·`block_array`는 블록이 모두 CSR이면 CSR, 모두 CSC면 CSC, 아니면 COO, `block_diag`·`tril`·`triu`·`random_array`는 COO다(어느 형식이든 값은 같다). `format=`은 `'coo'`, `'csr'`, `'csc'`, `'dok'`이다.
- **dtype**: `dtype=None`은 입력들의 배열 승격이고 `dtype=`은 명시적 변환이다. `kron`의 값은 배열의 곱(`sd.multiply`)이라 bool × bool은 int64다(SP56; `[scipy와 다름]`: scipy는 bool). 모든 값이 입력의 값이거나 곱 하나라 `toarray()`가 밀집 결과(`sd.eye`, `sd.tril`, `sd.triu`)와 비트까지 같다(`sd.linalg.kron`과는 0의 부호만 다를 수 있다: 저장하지 않은 자리는 +0.0).
- `eye_array`: 인자 이름과 기본값은 `sd.eye`의 것이다(`n_rows`, `n_cols`, `dtype=None`은 float64). 대각이 밖이면(`eye_array(2, 3, k=5)`) 빈 배열이다(`sd.eye`와 같다; `[scipy와 다름]`: scipy는 `ValueError`). scipy의 기본 형식 DIA 대신 CSR이다(SP6, SP48).
- `diags_array`: offsets가 int면 `diagonals`는 대각 하나(`sd.ndarray`나 수), offsets가 리스트면 같은 길이의 리스트(각각 `sd.ndarray`나 수; SP57)다. 길이 1인 대각은 그 대각 전체를 채우고, 더 긴 것은 자르고(scipy와 같다), 짧으면 `ValueError`다. 같은 offset이 두 번이거나 모양 밖이면 `ValueError`다. `shape`가 없으면 `len(diagonals[0]) + |offsets[0]|`의 정사각이다. 0은 저장하지 않는다(scipy와 같다). 정수 대각의 dtype은 int64다(SP52; `[scipy와 다름]`: scipy 1.18.1의 기본은 float64와 `FutureWarning`).
- `random_array`: `rng`에 `sd.random.Generator`를 꼭 받는다(SP49). 없으면 `random_array(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType`(`TypeError`; optimize O31과 같다). 자리는 비복원 추출로 정확히 round(density·m·n)개(C 순서의 위치를 뽑은 순서대로), 값은 `data_sampler(n)`(위치 인자 하나) 또는 dtype마다 [0, 1) 균등(float64), 전체 범위의 정수(int64), 실수부·허수부가 [0, 1) 균등(complex128), 균등의 0이 아닌 것(bool)이다. `density`는 [0, 1] 안이어야 한다(`ValueError`). 같은 `rng` 상태면 스레드 수와 상관없이 같은 비트다. scipy와 같은 값은 아니다(난수 생성기가 다르다). `[scipy와 다름]`(SP58): scipy는 `data_sampler(size=n)`으로 부르고 `dtype=None`이면 그 결과를 float64로 바꾼다.
- `kron`: 블록 a_ij·B의 배열이다. 1차원은 한 행이고 둘 다 1차원이면 1차원 결과다(scipy와 같다). 저장된 0과 곱해서 0이 된 값도 저장된다(scipy의 `format='csr'`과 같다).
- `kronsum(A, B)`: `kron(I_n, A) + kron(B, I_m)`이다(정사각만; 희소끼리의 `+`라 만든 0은 지운다; scipy와 같다).
- `hstack`, `vstack`, `block_array`, `block_diag`: 블록은 희소 배열, 밀집 `sd.ndarray`, None(빈 블록; `block_diag`에는 없다)이고, 1차원 블록은 한 행이다(scipy와 같다). 행·열의 크기가 맞지 않으면 `ValueError`(`vstack(): blocks[:,0] has incompatible column dimensions: blocks[1,0].shape[1] is 3, expected 4`)다. `block_array`의 `blocks`는 블록 행의 리스트다. `block_diag`의 밀집 블록도 0이 아닌 값만이다(`[scipy와 다름]`: scipy는 밀집 블록의 모든 자리를 저장한다).
- `tril`, `triu`: col ≤ row + k(col ≥ row + k)인 원소를 A의 COO 순서 그대로(중복도) 남긴다(scipy와 같다). 인자 이름은 `sd.tril`·`sd.triu`와 같다. 2차원만이다.
- `find(A)`: `(row, col, data)`의 쓸 수 있는 새 밀집 `sd.ndarray` 셋이다(중복을 더한 표준 꼴의 행 우선 순서, 저장된 0은 뺀다; scipy와 같다). 2차원만이다.

## §5. 직접 풀이와 `LinearOperator` (SP4)
```
spsolve(A, b, /)
splu(A, /)
factorized(A, /)
inv(A, /)
spsolve_triangular(A, b, /, *, upper, unit_diagonal=False)
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
- **바탕**은 Accelerate의 희소 LU(`SparseFactorizationLU`, 문턱 부분 피벗)이고 순서는 AMD로 고정한다(SP18, SP59; `ordering=` 없음). 대칭 행렬도 LU로 푼다: Accelerate의 Cholesky·LDLᵀ는 기본 스레드 모드에서 실행마다 비트가 달라서 첫 판에 쓰지 않는다. scipy SuperLU의 옵션(`permc_spec`, `diag_pivot_thresh`, `relax`, `panel_size`, `options`, `use_umfpack`)은 없다(SP61: 주면 `unexpected keyword argument`의 `TypeError`).
- **A**는 정방 희소 배열이다. 모든 형식을 CSC로 바꿔 분해하므로 결과의 비트는 형식과 무관하다. 밀집 `sd.ndarray`는 `spsolve(): 'A' must be a sparse array, not ndarray; for a dense array use sd.linalg.solve`(`TypeError`)다. 값은 float64(bool·int64에서)나 complex128로 계산하고, 저장된 NaN·inf는 `spsolve(): 'A' contains NaN or infinity`(`ValueError`)다. 정방이 아니면 `'A' must be square, got 3x4`, 1차원이면 `'A' must have 2 dimensions, got 1`이다.
- **b**는 밀집 `sd.ndarray`((n,)이나 (n, k); 리스트와 numpy 배열은 `TypeError`, API.md §5)이고 `spsolve`는 희소 배열도 받는다. 행 수가 다르면 `'A' is 16x16 but 'b' has 3 rows`다.
- **특이**(SP62): 구조적 특이, 유한하지 않은 해, 분해의 조건수 추정 κ₁ = ‖A‖₁·est(‖A⁻¹‖₁)(LAPACK `lacn2`의 1-노름 추정, 결정적)이 κ₁·ε ≥ 1인 것이 모두 `LinAlgError("spsolve(): singular matrix")`다(함수 이름은 부른 것: `splu(): `, `inv(): `, `factorized(): `). 정확히 특이한 행렬은 Accelerate가 성공을 돌려주고 크기 1/ε의 유한한 해를 내기 때문이다. `[scipy와 다름]`: scipy `spsolve`는 구조적 특이에서 `MatrixRankWarning`과 NaN, `splu`는 `RuntimeError("Factor is exactly singular")`이고, 값으로 특이하거나 κ₁ ≥ 1/ε인 행렬(SuiteSparse의 sherman3, cvxqp3 등)에는 경고 없이 해를 돌려준다(SP66: 그 해는 앞오차 κε ≥ 1이라 맞는 자리가 없다).
- **반복 다듬기**(SP63): 모든 풀이가 x ← x + solve(b − Ax)를 한 번 한다. 정규화 역오차는 시험 행렬 모두에서 0.50ε 이하다(다듬기 없이 404ε인 행렬이 있었다; docs/design/sparse.md §3.6).
- `spsolve(A, b)`: b가 (n,)이면 (n,), (n, k)이면 (n, k) 밀집이다(SP69; `[scipy와 다름]`: scipy는 (n, 1)을 (n,)으로 편다; `sd.linalg.solve`와 같다). b가 희소면 결과도 희소이고 A의 형식이다(CSR이면 CSR, 나머지는 CSC; 정확한 0은 뺀다; scipy와 같다; SP70, `[scipy와 다름]`: 열이 하나인 희소 b와 1차원 희소 b도 희소 결과(1차원은 CSR)이고 scipy는 밀집). dtype은 A와 b의 승격이다. 0 × 0이면 빈 결과다(Accelerate는 받지 않으므로 앞에서 답한다).
- `splu(A)`: 내보내지 않는 `SparseLU`를 돌려준다(SP60). `solve(b, trans='N' | 'T' | 'H')`는 A, Aᵀ, Aᴴ의 풀이이고 결과는 b의 모양이다. `shape`은 A의 모양, repr은 `SparseLU(shape=(n, n), dtype=float64)`다. scipy `SuperLU`의 `L`, `U`, `perm_r`, `perm_c`, `nnz`는 Accelerate가 인수를 내보이지 않아 없다. 실수 인수에 복소 b를 주면 실수부와 허수부를 함께 푼다(`[scipy와 다름]`: scipy는 `TypeError`). `SparseLU()`를 직접 만들면 `TypeError`다. `factorized(A)`는 `splu(A).solve`(묶인 메서드)다.
- **우변이 여럿**(SP64): 보통 풀이는 Accelerate가 열을 함께 푼다. 결과는 열 수 k와 값으로 정해지고 결정적이며, 열 하나의 비트는 그 열을 혼자 푼 것과 반올림 크기만큼 다를 수 있다. `trans='T'`·`'H'`는 열마다 따로 푼다(Accelerate의 전치 풀이가 여러 열에서 틀리기 때문이다; docs/design/sparse.md §1.4).
- **결정성**: 같은 입력이면 sd의 스레드 수, A의 형식, 동시에 부르는 Python 스레드와 상관없이 같은 비트다. Accelerate의 스레드 모드(`VECLIB_MAXIMUM_THREADS`)가 바뀌면 끝자리가 다를 수 있다(BLAS·LAPACK과 같은 규칙).
- `inv(A)`: 단위 행렬을 우변으로 푼 희소 결과(A의 형식, CSR이 아니면 CSC; 정확한 0은 뺀다; scipy와 같다). 대개 밀집에 가깝다.
- `spsolve_triangular(A, b, upper=..., unit_diagonal=...)`: 이 라이브러리의 대입 커널이다. `upper`는 꼭 주는 키워드다(SP68; `sd.linalg.solve_triangular`와 같다; `[scipy와 다름]`: scipy는 `lower=True`가 기본이다). 고른 삼각(`upper=True`면 위, `False`면 아래)과 대각만 읽고 나머지 원소는 무시한다(`sd.linalg.solve_triangular`와 같다; `[scipy와 다름]`: scipy 1.18.1은 그런 원소가 있으면 경고 없이 틀린 답을 낸다). 행마다의 합은 §1.5의 줄 합이고, 우변이 여럿이면 열마다 따로 계산해 열 하나의 비트가 k와 무관하다. 대각이 없거나 0이면 `LinAlgError("spsolve_triangular(): singular matrix")`, `unit_diagonal=True`면 대각을 읽지 않는다. 결과는 b의 모양과 승격한 dtype이다. scipy의 `overwrite_A`, `overwrite_b`는 없다.
- `is_sptriangular(A)`: `(lower, upper)`의 bool 둘, `spbandwidth(A)`: `(below, above)`의 int 둘(scipy와 같다). 패턴만 보고 저장된 0도 센다. 아무것도 저장하지 않은 행렬은 `(True, True)`, `(0, 0)`이다(`[scipy와 다름]`: scipy 1.18.1은 빈 CSR에서 예외).
- `LinearOperator(shape, matvec, /, *, rmatvec=None, matmat=None, rmatmat=None, dtype=None)`: Python 클래스. `shape`과 `matvec`만 위치이고 나머지는 키워드다(SP65; `[scipy와 다름]`: scipy는 `rmatvec`, `matmat`, `dtype`도 위치로 받는다). `matvec(x)`는 (N,)이나 (N, 1)의 `sd.ndarray`를 받아 (M,)이나 (M, 1)을 돌려주고, 함수의 결과는 `sd.asarray`의 규칙으로 바꿔 그 모양으로 맞춘다(크기가 다르면 `ValueError`). `matmat`·`rmatmat`이 없으면 열마다 `matvec`·`rmatvec`이다. `dtype=None`이면 0의 벡터로 `matvec`을 한 번 불러 정한다(scipy와 같다). `rmatvec`이 없는 연산자의 `rmatvec`은 `rmatvec(): the operator was made without 'rmatvec'`(`ValueError`, SP71; scipy는 `NotImplementedError`)다. 하위 클래스는 `_matvec`이나 `_matmat`(그리고 `_rmatvec`, `_rmatmat`)을 정의하고 `LinearOperator.__init__(self, shape, None, dtype=...)`을 부른다. repr은 `LinearOperator(shape=(2, 3), dtype=float64)`(하위 클래스는 그 이름)다.
  - 연산: `L @ x`(1차원이나 열 하나는 `matvec`, 2차원은 `matmat`), `x @ L`, `L @ M`(곱), `L + M`, `L - M`, `-L`, `c * L`, `L * c`(스칼라만), `L ** p`(정수 p ≥ 0, 정방), `.T`, `.H`, `adjoint()`. 결과의 모양과 dtype은 scipy와 같다. `[scipy와 다름]`: `L * x`(벡터)와 `L * M`(연산자)은 `TypeError`다(`@`만, 별칭 없음 SP24). `dot`, `__call__`은 없다.
- `aslinearoperator(A)`: 희소 배열, 2차원 `sd.ndarray`, `LinearOperator`(그 자신)를 받는다. 희소·밀집에서 만든 연산자의 곱은 `A @ x`와 `Aᴴ @ x`이고 비트가 같다(반복법(SP5)은 그 행렬을 바로 쓴다).
- `sd.linalg`의 함수에 희소 배열을 주면 쓸 함수를 짚는 `TypeError`다: `solve(): 'a' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.spsolve`(대응은 docs/design/sparse.md §3.10). 짚을 함수가 아직 없으면(`eigh` → `eigsh`는 SP6) `; a sparse array has no eigh: use eigh(A.toarray())`이고, 그 함수가 생기면 저절로 그것을 짚는다.
- 배열의 함수도 같다(SP67): `sum(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use A.sum()`, `matmul(): ...; for a sparse array use A @ B`. 희소의 짝이 없으면 `; a sparse array has no sort: use sort(A.toarray())`다.

## §6. 반복법 (SP5)
```
cg(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, max_iter=None, M=None)
gmres(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, restart=None, max_iter=None, M=None)
minres(A, b, /, *, x0=None, rtol=1e-05, shift=0.0, max_iter=None, M=None)
bicgstab(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, max_iter=None, M=None)
lsqr(A, b, /, *, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, max_iter=None, x0=None)
lsmr(A, b, /, *, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, max_iter=None, x0=None)
CgResult.x
CgResult.converged
CgResult.status
CgResult.message
CgResult.n_iter
CgResult.normr
CgResult.summary()
BicgstabResult.x
BicgstabResult.converged
BicgstabResult.status
BicgstabResult.message
BicgstabResult.n_iter
BicgstabResult.normr
BicgstabResult.summary()
GmresResult.x
GmresResult.converged
GmresResult.status
GmresResult.message
GmresResult.n_iter
GmresResult.n_inner
GmresResult.normr
GmresResult.summary()
MinresResult.x
MinresResult.converged
MinresResult.status
MinresResult.message
MinresResult.n_iter
MinresResult.normr
MinresResult.norma
MinresResult.conda
MinresResult.normar
MinresResult.normx
MinresResult.summary()
LsqrResult.x
LsqrResult.converged
LsqrResult.status
LsqrResult.message
LsqrResult.n_iter
LsqrResult.normr
LsqrResult.normr_damped
LsqrResult.norma
LsqrResult.conda
LsqrResult.normar
LsqrResult.normx
LsqrResult.var
LsqrResult.summary()
LsmrResult.x
LsmrResult.converged
LsmrResult.status
LsmrResult.message
LsmrResult.n_iter
LsmrResult.normr
LsmrResult.normar
LsmrResult.norma
LsmrResult.conda
LsmrResult.normx
LsmrResult.summary()
```
- **이 라이브러리가 scipy 1.18.1의 알고리즘과 멈춤 판정으로 짠다**(SP16): scipy의 Python 판(`_isolve`의 `iterative.py`, `minres.py`, `lsqr.py`, `lsmr.py`)을 계산 순서까지 옮겼다. `cg`·`bicgstab`은 재귀 잔차로 ‖r‖ < max(rtol·‖b‖, atol), `gmres`는 재시작 GMRES(scipy의 왼쪽 전처리 꼴, 바깥 반복마다 참 잔차 ‖r‖ ≤ max(rtol·‖b‖, atol), 안쪽 판정은 scipy의 `ptol`; Givens 회전은 LAPACK 3.10의 `lartg`), `minres`는 Paige–Saunders(`shift`), `lsqr`은 Paige–Saunders 1982, `lsmr`은 Fong–Saunders 2011(`damp`, `atol`·`btol`·`conlim`)이다. Accelerate의 반복법은 쓰지 않는다(멈춤 판정이 처음 잔차에 대한 상대이고 반복 수를 돌려주지 않는다).
- **같은 반복 수와 상태**: 시험 문제(scipy 기록 40개)에서 상태가 모두 scipy와 같고 반복 수가 36개에서 같으며, 단계마다의 잔차 노름이 scipy와 상대 10⁻⁶ 안에서 같다. 예외는 scipy 자신이 반올림에 민감한 문제다: 무작위 비대칭 행렬의 BiCGSTAB, 부정 행렬의 MINRES 뒷부분, 반복이 수백인 LSQR·LSMR은 b를 상대 10⁻¹⁶ 바꾼 scipy끼리도 반복 수가 20%까지 다르다. 그런 문제에서는 상태가 같고 반복 수가 그 퍼짐 안이다(docs/design/sparse.md §3.7; 나머지 하나는 `bicgstab`이 반 단계에서 멈춘 반복을 `n_iter`가 세는 것이다).
- **입력**(SP77): `A`, `M`(전처리기, A⁻¹을 어림한 것)은 희소 배열(모든 형식), 2차원 `sd.ndarray`, `LinearOperator`다. 행렬(`aslinearoperator`로 감싼 것도)이면 반복 전체가 C++이고, 밀집 행렬은 모든 원소를 저장한 CSR로 곱한다. 다른 `LinearOperator`는 곱마다 Python으로 그 `matvec`(`lsqr`·`lsmr`은 `rmatvec`도)을 부른다. `b`와 `x0`은 밀집 `sd.ndarray`((n,)이나 (n, 1); 리스트와 numpy 배열은 `TypeError`, `spsolve`와 같다)이고 결과의 `x`는 언제나 (n,)이다(scipy와 같다). scipy의 `x0='Mb'`는 없다. 저장된 NaN·inf는 `ValueError`(`cg(): 'A' contains NaN or infinity`; §1.6)다.
- **dtype**(SP78): 계산은 float64나 complex128이다. A, b, x0, M 가운데 하나라도 복소면 복소다(`[scipy와 다름]`: scipy는 A와 b로만 정해 복소 x0의 허수부를 경고와 함께 버리고, 실수 계에 복소 M을 주면 실패한다). 실수로 선언한 `LinearOperator`가 복소 값을 돌려주면 `TypeError`다.
- **인자**(SP73, SP76): 반복 상한은 `max_iter`다(`[scipy와 다름]`: scipy의 `maxiter`, `lsqr`의 `iter_lim`). 기본값은 scipy의 것이다: `cg`·`bicgstab`은 10n, `gmres`는 바깥 반복 10n, `minres`는 5n, `lsqr`은 2n(n은 열 수), `lsmr`은 min(m, n). `gmres`의 `restart`는 한 바깥 반복의 안쪽 반복 수(기본 20, n을 넘으면 n), `max_iter`는 바깥 반복 수다(SP75, scipy와 같다). `max_iter`·`restart`는 None이나 1 이상의 int다(`[scipy와 다름]`: scipy는 0을 받고 `info = 0`을 돌려준다). `rtol`·`atol`·`btol`·`damp`는 유한한 0 이상의 수, `conlim`은 0 이상의 수(0과 inf는 한계 없음, scipy와 같다), `shift`는 유한한 수다(`[scipy와 다름]`: scipy는 `atol`만 검사한다). 메시지는 sd.optimize와 같다: `cg(): 'rtol' must be a finite number >= 0`, `gmres(): 'restart' must be >= 1`.
- `callback`, `callback_type`, `show`, `check`, `calc_var`은 없다(SP74; 주면 `unexpected keyword argument`의 `TypeError`). `lsqr`의 `var`(A^H A + damp² I의 역의 대각 어림)는 언제나 계산한다(scipy의 `calc_var=True`와 같은 값; 반복마다 n번의 곱셈이다). 복소 행렬에서는 |d_k|²를 더한다(`[scipy와 다름]`: scipy는 d_k²를 더해 복소 `var`를 낸다).
- **결과**(SP72, SP79): 함수마다 읽기 전용 결과 객체(`CgResult` … `LsmrResult`, 내보내지 않는다)이고 scipy의 튜플로 풀린다: `x, info = cg(A, b)`(`cg`·`bicgstab`·`gmres`·`minres`), `x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(A, b)`, `x, istop, itn, normr, normar, norma, conda, normx = lsmr(A, b)`. `len(r)`과 `r[i]`도 그 튜플이다. 필드:
  - `x`(쓰기 금지 배열), `status`(scipy의 `info`·`istop`), `converged`, `message`(멈춘 까닭), `n_iter`.
  - `status`: `cg`·`gmres`는 0(수렴)이나 `max_iter`, `bicgstab`은 그 밖에 −10(ρ 붕괴), −11(ω 붕괴), `minres`는 0이나 `max_iter`(scipy의 `info`; 멈춘 판정은 `message`), `lsqr`·`lsmr`은 scipy의 `istop` 0–7이다. `converged`는 `cg`·`bicgstab`·`gmres`·`minres`에서 `status == 0`, `lsqr`·`lsmr`에서 `istop`이 0, 1, 2, 4, 5인 것이다(3과 6은 조건수 한계, 7은 반복 상한).
  - `n_iter`: 반복 수(`cg`·`bicgstab`·`minres`, `lsqr`·`lsmr`의 `itn`). `bicgstab`이 반 단계의 ‖s‖ 판정에서 멈추면 그 반복도 센다(scipy의 `callback`은 세지 않는다). `gmres`의 `n_iter`는 바깥 반복(재시작) 수(`max_iter`와 같은 단위)이고 `n_inner`가 안쪽(Arnoldi) 반복의 합이다(scipy의 `callback_type='pr_norm'` 호출 수).
  - `normr`: 방법이 쓴 잔차 노름(`cg`·`bicgstab`은 재귀 잔차, `gmres`는 마지막 참 잔차, `minres`·`lsqr`·`lsmr`은 그 어림). `minres`·`lsqr`·`lsmr`의 `norma`, `conda`, `normar`(‖Aᴴr‖, minres는 ‖Ar‖), `normx`는 scipy의 어림이다(`lsqr`의 `anorm`, `acond`, `arnorm`, `xnorm`은 `lsmr`의 이름으로 부른다). `lsqr`의 `normr_damped`는 scipy의 `r2norm` = √(‖r‖² + damp²‖x − x0‖²)다.
  - `summary()`는 API.md §2.3의 상자(멈춘 까닭, 반복 수와 노름, 그리고 x; 긴 x는 앞뒤 다섯, 복소 x는 실수부·허수부)이고 repr은 `CgResult(converged=True, status=0, n_iter=69, normr=0.00017519, x=<array of shape (900,)>)`이다.
- **`minres`의 실패**(SP81): 양정치가 아닌 전처리기와 βₖ < 0(대칭이 아닌 행렬)은 `LinAlgError("minres(): the preconditioner is not positive definite")`, `LinAlgError("minres(): the matrix is not symmetric")`다(`[scipy와 다름]`: scipy는 `ValueError`; `LinAlgError`는 `ValueError`의 하위 클래스다).
- **결정성**(SP80): 곱은 §1.5의 SpMV, 내적은 `sd.linalg.vecdot`의 합(복소는 첫 벡터의 켤레), 노름은 `sd.linalg.vector_norm`의 것이다. 모두 크기만으로 나누므로 같은 입력이면 sd의 스레드 수, A의 형식(CSR·CSC·COO·DOK), 행렬과 그 `aslinearoperator`, `A @ v`를 돌려주는 Python `LinearOperator`와 상관없이 같은 비트다. 행렬만이면 일이 기준값 이상일 때 GIL을 놓고 돈다. Python `LinearOperator`가 있으면 그 호출 때문에 GIL을 쥐고 돈다(다른 커널은 스스로 놓는다).
- 정확도(docs/design/sparse.md §4): 결과는 "멈춤 판정을 만족한 점"이다. 수렴한 결과의 다시 계산한 참 잔차가 판정을 만족하고, Arb의 해와 견준 앞오차가 그 잔차가 허락하는 ‖A⁻¹‖·‖b − Ax̂‖(최소제곱은 ‖(AᴴA)⁻¹‖·‖Aᴴr̂‖) 안이다(테스트).

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
| `eye_array`, `diags_array`, `random_array`, `block_array` | 같은 이름(기본 형식 CSR; `random_array`는 `rng` 필수) | SP48, SP49, SP51 |
| `eye`, `identity`, `diags`, `spdiags`, `random`, `rand`, `bmat` | 없음(옛 `spmatrix`를 돌려주는 함수) | SP2 |
| `kron`, `kronsum`, `hstack`, `vstack`, `block_diag`, `tril`, `triu`, `find` | 같은 이름(`kron`의 기본 형식 CSR) | SP3, SP51 |
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
| `SuperLU` (`splu`의 결과 형) | 내보내지 않는 `SparseLU`(`solve`, `shape`) | SP60 |
| `is_sptriangular`, `spbandwidth` | 같은 이름 | SP4 |
| `LinearOperator`, `aslinearoperator` | 같은 이름 | SP4 |
| `cg`, `gmres`, `minres`, `bicgstab`, `lsqr`, `lsmr` | 같은 이름(`maxiter`·`iter_lim` → `max_iter`; 튜플 대신 그 튜플로 풀리는 결과 객체; `callback`·`show`·`check`·`calc_var` 없음) | SP5, SP16, SP72–SP74 |
| `eigsh`, `eigs`, `svds` | 같은 이름(`maxiter` → `max_iter`) | SP6 |
| `expm`, `expm_multiply`, `norm`, `onenormest` | 같은 이름(`onenormest`의 `itmax` → `max_iter`) | SP7, SP14 |
| `ArpackError`, `ArpackNoConvergence` | 없음(결과의 상태) | SP6 |
| `MatrixRankWarning` | 없음(`LinAlgError`) | SP4, SP62 |
| `use_solver` | 없음 | SP4 |
| `spilu`, `lobpcg`, `bicg`, `cgs`, `qmr`, `tfqmr`, `lgmres`, `gcrotmk`, `matrix_power`, `funm_multiply_krylov`, `LaplacianNd` | 나중 후보 | — |
| 하위 모듈 `dsolve`, `isolve`, `eigen`, `interface`, `matfuncs` | 없음 | SP3 |
| `scipy.sparse.linalg.X` 전체 | `sd.sparse.X` (`sd.sparse.linalg`는 없다) | SP3 |

## §10. 마일스톤별 남은 질문 (추천을 첫째에)
docs/design/sparse.md §9와 같은 목록이다. 결정이 나면 이 문서와 그 문서를 함께 고친다.

- **SP0**: 정했다(SP6–SP20, docs/design/sparse.md §1.1).
- **SP1**: 정했다(SP21–SP34, docs/design/sparse.md §1.1).
- **SP2**: 정했다(SP35–SP47, docs/design/sparse.md §1.1).
- **SP3**: 정했다(SP48–SP52, docs/design/sparse.md §1.1). SP3 뒤에 정했다(SP53–SP58).
- **SP4**: 정했다(SP59–SP65, docs/design/sparse.md §1.1). SP4 뒤에 정했다(SP66–SP71; 추천안, 사용자 위임).
- **SP5**: 정했다(SP72–SP75, docs/design/sparse.md §1.1; 추천안, 사용자 위임). SP5에서 정했다(SP76–SP82; 추천안, 사용자 위임).
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
