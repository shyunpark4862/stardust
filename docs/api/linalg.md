# 선형대수 API (sd.linalg)

`import stardust as sd`의 하위 모듈 `sd.linalg`가 선형대수 API다. 이 문서는 선형대수의 공개 API와 동작 규칙을 모두 적는다.

- 라이브러리 전체의 공통 규칙(dtype, 에러 종류, 스레드)은 루트 [API.md](../../API.md)에 있다. 이 문서는 선형대수 영역의 규칙이다.
- 배열 자체의 규칙(0차원 결과, 변환, 수치 의미론)은 [docs/api/array.md](array.md)에 있다. 이 문서는 필요한 곳에서 그 절을 인용한다.
- 절 번호(`§N`, `§N.M`)는 코드 주석이 인용한다. 번호를 바꾸지 않는다.
- 언어 표시가 없는 코드 블록은 `tests/linalg/test_api_surface.py`가 읽는다.
  - 이 블록에는 공개 이름과 시그니처만 쓴다. 한 줄에 함수 하나이고, 줄 끝에 `#` 주석을 붙일 수 있다.
  - 블록에 나온 이름과 `sd.linalg.__all__`은 정확히 같아야 한다. `*` 뒤의 이름은 그 함수의 keyword-only 인자와 정확히 같아야 한다.
  - 예시는 본문의 인라인 코드로 쓴다.

---

## §1. 선형대수 공통 규칙

### §1.1 시그니처와 이름
- **positional-only (`/` 앞)**
  - 배열 인자
  - 함수의 본 인자. 예: `matrix_power`의 `n`, `eigh`의 `b`
- **keyword-only (`*` 뒤)**: 이름이 붙은 옵션은 모두 keyword-only다.
  - 필수 옵션이 셋 있다: `solve_triangular`의 `upper`, `solve_banded`의 `lower_bandwidth`, `upper_bandwidth`. 빠뜨리면 `TypeError`다.
- `help()`는 이 문서의 시그니처를 그대로 보인다. 예: `solve_banded(ab, b, /, *, lower_bandwidth, upper_bandwidth)`
  - 틀린 호출 모양(없는 키워드, 인자 개수, positional-only 인자를 키워드로)은 nanobind의 `incompatible function arguments` 메시지다. C++로 묶은 함수의 예외다 (API.md §2.1).
- **이름**
  - Array API(`linalg` 확장)에 있는 함수는 그 이름을 쓴다. 예: `vecdot`, `matrix_norm`, `svdvals`
  - Array API에 없는 함수는 scipy.linalg의 이름을 따르거나 풀어 쓴다. 예: `ldl`, `schur`, `expm`, `cholesky_solve`, `solve_banded`
  - 함수 하나에 이름 하나다. 별칭은 없다.
- **행렬의 구조는 옵션이 아니라 함수 이름으로 고른다.**
  - 예: 삼각은 `solve_triangular`, 양정치는 `cholesky_solve`, 띠는 `solve_banded`
  - `solve`는 언제나 일반 LU로 푼다. 입력을 보고 구조를 판별하지 않는다. scipy의 `assume_a=` 같은 옵션은 없다.
- **두지 않는 옵션**: `out=`, `overwrite_a=`, `check_finite=`, `driver=`, `lwork=`
  - 유한성 검사는 끌 수 없다 (§1.9). 함수마다 LAPACK 루틴은 하나로 고정이다.

**옵션 이름**

| 이름 | 뜻 | 받는 값 | 쓰는 함수 |
|---|---|---|---|
| `upper` | 삼각 행렬이 위 삼각인가 | `True`/`False` | `cholesky`, `cholesky_solve`, `solve_triangular` |
| `unit_diagonal` | 대각을 1로 보고 읽지 않는다 | `True`/`False` | `solve_triangular` |
| `mode` | `'reduced'` 또는 `'complete'` | `str` | `qr`, `qr_pivoted`, `svd` |
| `rtol` | 가장 큰 특이값 대비 상대 허용오차 | `None` 또는 실수 (§1.1 아래) | `matrix_rank`, `pinv`, `lstsq` |
| `ord` | 노름의 종류 | 함수마다 다르다 (§3) | `vector_norm`, `matrix_norm`, `cond` |
| `k` | 대각선 오프셋. 양수는 위쪽, 음수는 아래쪽 | `int` | `trace`, `diagonal` |
| `axis` | 축 | `vecdot`, `cross`는 `int`, `vector_norm`은 `int`, `tuple`, `list`, `None` | `vecdot`, `cross`, `vector_norm` |
| `keepdims` | 줄인 축을 길이 1로 남긴다 | `True`/`False` | `vector_norm`, `matrix_norm` |
| `dtype` | 누적 dtype | dtype 객체, 그 이름, 또는 `None` | `trace` |
| `lower_bandwidth`, `upper_bandwidth` | 띠의 아래·위 폭 | 0 이상의 `int` | `solve_banded` |

- **옵션의 검사와 메시지는 배열·데이터프레임과 같다** (API.md §2.1). `None`을 주어도 그 옵션의 검사가 받는다. `None`이 뜻이 있는 옵션(`axis=None`, `rtol=None`, `dtype=None`)이 아니면 다른 틀린 값과 같은 `TypeError`다. 예: `svd(x, mode=None)`은 `svd(): 'mode' must be 'reduced' or 'complete', not NoneType`
- **bool 옵션**(`upper`, `unit_diagonal`, `keepdims`)은 `True`/`False`만 받는다. `0`, `1`, `None`, `numpy.bool_`은 `TypeError`다: `cholesky(): 'upper' must be True or False, not int`
- **int 인자**(`k`, `n`, `axis`, 대역폭)는 `operator.index`의 규칙으로 읽는다: `int`와 numpy 정수(`np.int64(2)`)를 받는다. `bool`, `numpy.bool_`, `float`은 `TypeError`, int64 범위 밖은 `OverflowError`다 (API.md §2, 배열 영역과 같다).
  - 메시지: `matrix_power(): 'n' must be an int, not float`. `vector_norm`의 `axis`는 tuple과 list도 받으므로 `'axis' must be an int or a tuple or list of ints, not float`다 (배열과 같다).
- **`mode`**: 다른 문자열은 `ValueError`, 문자열이 아니면 `TypeError`다. 메시지는 데이터프레임의 문자열 옵션과 같다: `'mode' must be 'reduced' or 'complete', not 'full'`, `..., not NoneType`
- **`rtol`**
  - `None`이면 `max(M, N) · eps`다. `eps`는 float64의 머신 엡실론이다.
  - 유한한 0 이상의 `int` 또는 `float`이다(numpy 정수 포함). 음수, NaN, inf는 `ValueError`다.
  - `bool`, 배열, 문자열은 `TypeError`다. 배치마다 다른 허용오차(배열)는 받지 않는다.
  - float로 바꿀 수 없을 만큼 큰 int는 `OverflowError`다.
- **`dtype`**: dtype 객체나 그 이름을 받는다 (API.md §1). 다른 것은 `TypeError`다.

### §1.2 입력 배열
- **배열 인자는 `sd.ndarray`만 받는다.** 리스트, numpy 배열, Python 스칼라는 `TypeError`다.
  - 필요하면 먼저 `sd.asarray`로 변환한다. numpy 배열은 복사 없이 바뀐다 (array.md §3.3).
- 메모리 레이아웃은 무엇이든 받는다: C 순서, F 순서, 임의의 strides, 쓰기 금지 배열, `broadcast_to`의 뷰
  - 결과는 입력의 레이아웃과 무관하게 비트까지 같다 (§1.11).
- 입력 배열은 바뀌지 않는다.
- **행렬과 배치**
  - 배열의 마지막 두 축이 행렬이고, 그 앞의 축은 배치다.
  - 배열 인자가 여럿이면 배치 축끼리 브로드캐스트한다. 브로드캐스트할 수 없으면 `ValueError`다.
  - 예: `solve`의 `a`가 `(2, 1, 3, 3)`, `b`가 `(4, 3, 2)`이면 결과는 `(2, 4, 3, 2)`다.
- **벡터 인자**는 마지막 한 축이 벡터다. 예: `outer`, `solve_tridiagonal`의 세 대각, `lu_solve`의 `perm`
- **차원과 모양**
  - 행렬 인자가 2차원 미만이면 `ValueError`다.
  - 정사각 행렬이 필요한 함수에 정사각이 아닌 행렬을 넣으면 `ValueError`다.
  - 인자끼리 크기가 맞지 않으면 `ValueError`다. 예: `solve`의 `a`가 3×3인데 `b`의 행이 2개

### §1.3 우변 `b`
solve류 함수(`solve`, `solve_triangular`, `cholesky_solve`, `lu_solve`, `lstsq`, `solve_tridiagonal`, `solve_banded`)는 같은 규칙을 따른다.
- `b.ndim == 1`이면 벡터 하나 `(M,)`다. 결과는 `(..., N)`이다.
  - 이 벡터는 행렬의 배치 전체에 브로드캐스트한다. 예: `a`가 `(2, 3, 3)`, `b`가 `(3,)`이면 결과는 `(2, 3)`이다.
- `b.ndim >= 2`이면 행렬의 묶음 `(..., M, K)`다. 결과는 `(..., N, K)`다.
  - 예: `a`가 `(3, 3)`, `b`가 `(4, 3, 1)`이면 결과는 `(4, 3, 1)`이다.
- Array API의 `solve` 규칙과 같다.
- `K = 0`(우변이 없음)도 받는다.
- `solve_sylvester`의 `c`는 이 규칙을 따르지 않는다. 언제나 행렬이다 (§6).

### §1.4 대칭, 에르미트, 삼각 입력
- **대칭/에르미트 입력은 아래 삼각만 읽는다.** 위 삼각은 보지 않는다. 대칭인지 검사하지 않는다.
  - 대상: `cholesky`, `eigh`, `eigvalsh`(`a`와 `b` 모두), `ldl`
  - 읽을 삼각을 고르는 옵션은 없다.
- complex 입력은 언제나 에르미트로 본다. 복소 대칭 행렬은 지원하지 않는다.
  - 대각의 허수부는 읽지 않는다. 예: `eigvalsh(asarray([[1+5j]]))`는 `[1.0]`이다.
- **삼각 인수를 받는 함수는 그 삼각만 읽는다.**
  - `solve_triangular`: `upper`가 고른 삼각. `unit_diagonal=True`이면 대각도 읽지 않는다.
  - `cholesky_solve`: `upper`가 고른 삼각
  - `lu_solve`: `l`의 대각 아래, `u`의 대각과 그 위
- 읽지 않는 원소는 결과에 영향이 없다. NaN이 있어도 된다 (§1.9).

### §1.5 결과 dtype
- **결과 dtype은 입력의 dtype만으로 정해진다.** 값에 따라 바뀌지 않는다.
- **계산 dtype**: 배열 인자 중 하나라도 complex128이면 complex128, 아니면 float64다.
  - bool과 int64 입력은 float64로 바꿔서 계산한다. 계산만 하는 함수(`vecdot`, `outer`, `kron`, `kronsum`, `cross`, `matrix_power`, `trace`)도 같다.
    - **그래서 int 입력의 결과도 float64다.** 배열 영역과 다른 점이다: `sd.matmul`, `sd.sum`은 int64를 유지한다. 예: `trace(int64 행렬)`은 `float`, `sd.sum(diagonal(int64 행렬), axis=-1)`은 `int`다.
    - 이유: LAPACK과 BLAS는 float에서만 돈다. int 결과가 필요하면 배열 영역의 함수(`sd.matmul`, `sd.sum`)를 쓰거나 `trace(dtype=sd.int64)`를 준다 (DESIGN §9.6).
  - 예: `det(int64 행렬)`은 `float`, 실수 `a`와 complex `b`의 `solve`는 complex128이다.
- **함수별 결과 dtype**

| 결과 | 실수 계산 | complex 계산 |
|---|---|---|
| 아래에 없는 결과 (해, 역행렬, 분해의 행렬, `det`, `vecdot`, `trace` 등) | float64 | complex128 |
| `eig`, `eigvals`의 결과 전부 | complex128 | complex128 |
| `eigh`, `eigvalsh`의 고유값 | float64 | float64 |
| 특이값 (`svd`의 `S`, `svdvals`, `lstsq`의 `singular_values`) | float64 | float64 |
| `vector_norm`, `matrix_norm`, `cond` | float64 | float64 |
| `slogdet`의 `sign` / `logabsdet` | float64 / float64 | complex128 / float64 |
| `matrix_rank`, `lstsq`의 `rank` | int64 | int64 |
| `perm` (`lu`, `qr_pivoted`, `ldl`) | int64 | int64 |
| `bandwidth`의 `lower`, `upper` | int64 | int64 |

- **예외 셋**
  - `diagonal`과 `block_diag`는 원소를 옮기기만 한다. 입력 dtype(`block_diag`는 그 승격)을 그대로 유지한다. bool은 bool, int64는 int64다.
  - `trace(dtype=)`을 주면 그 dtype이다 (§2).
  - `bandwidth`는 입력 dtype과 상관없이 정수다 (§6).

### §1.6 결과의 형태
- **결과가 0차원이면 Python 스칼라다** (array.md §1.2).
  - 예: 행렬 하나의 `det`은 `float`(complex 계산이면 `complex`), `matrix_rank`는 `int`, 1차원끼리의 `vecdot`은 `float`이다.
  - 배치가 있으면 배열이다. 배치 크기가 1이어도 배열이다. 예: `det(ones((1, 2, 2)))`의 shape은 `(1,)`이다.
  - namedtuple의 필드도 같다. 예: 행렬 하나의 `slogdet`은 Python 스칼라 둘, `lstsq`의 `rank`는 `int`다.
- **결과의 형태는 옵션에 따라 바뀌지 않는다.** 결과의 개수가 다른 변형은 함수를 따로 둔다.
  - 예: `eigh` / `eigvalsh`, `eig` / `eigvals`, `svd` / `svdvals`, `qr` / `qr_pivoted`
  - 그래서 `schur`는 언제나 `Z`를, `hessenberg`는 언제나 `Q`를 함께 계산한다.
- **결과가 여럿이면 namedtuple이다.** 타입은 세 영역이 함께 쓰는 `stardust._results`에 있고, 이름은 함수 이름의 `<함수>Result`다.

| 타입 | 필드 | 함수 |
|---|---|---|
| `SlogdetResult` | `sign`, `logabsdet` | `slogdet` |
| `LstsqResult` | `solution`, `rank`, `singular_values` | `lstsq` |
| `LUResult` | `L`, `U`, `perm` | `lu` |
| `QRResult` | `Q`, `R` | `qr` |
| `QRPivotedResult` | `Q`, `R`, `perm` | `qr_pivoted` |
| `LDLResult` | `L`, `D`, `perm` | `ldl` |
| `EighResult` | `eigenvalues`, `eigenvectors` | `eigh` |
| `EigResult` | `eigenvalues`, `eigenvectors` | `eig` |
| `SchurResult` | `T`, `Z` | `schur` |
| `HessenbergResult` | `H`, `Q` | `hessenberg` |
| `SVDResult` | `U`, `S`, `Vh` | `svd` |
| `BandwidthResult` | `lower`, `upper` | `bandwidth` |

  - 필드 이름: Array API에 있는 함수는 Array API의 이름이다 (`slogdet`, `eigh`, `svd`, `qr`, `eig`). 나머지는 행렬을 대문자 한 글자로 쓴다(`bandwidth`는 정수 둘이라 scipy의 이름 `lower`, `upper`다).
  - **순열**은 순열 행렬이 아니라 int64 인덱스 벡터 `perm`이다. 언제나 마지막 필드다.
  - 결과 타입은 `sd.linalg`의 공개 이름이 아니다. 필드 이름이나 위치로 꺼낸다.

### §1.7 결과의 메모리 레이아웃
- **행렬 결과는 행렬마다 F 순서(열 우선)다.** 배치 축은 그 바깥에 C 순서로 놓인다.
  - 예: `inv`의 `(2, 3, 3)` float64 결과의 strides는 `(72, 8, 24)`다.
  - 대상: 마지막 두 축이 행렬인 모든 결과. 해 `(..., N, K)`, 역행렬, 분해의 행렬, `outer`, `kron`, `kronsum`, `block_diag`, `matrix_power`, `expm`, `solve_sylvester`
  - 결과 순서는 영역마다 정한다 (array.md §1.9). 배열 영역이 새로 만드는 배열은 C 순서이고, linalg의 행렬 결과는 F 순서다. 값은 같고 strides만 다르다.
- **벡터 결과는 연속이다.** 배치 축은 그 바깥에 C 순서로 놓인다.
  - 대상: 해 `(..., N)`, 고유값, 특이값, `perm`, `diagonal`, `det` 등의 배치 결과
- `cross`, `vecdot`, `vector_norm`, `matrix_norm`, `trace`의 배열 결과는 C 순서다.
- 결과는 언제나 새 배열이다. 입력과 메모리를 공유하지 않는다.
  - 배열 하나를 돌려주는 함수(`inv`, `solve` 등)의 결과는 쓸 수 있다.
  - **namedtuple의 배열 필드는 쓰기 금지다** (API.md §2, 모든 영역의 결과 객체와 같다). 쓰면 `ValueError: assignment destination is read-only`다. 그 필드에서 만든 뷰도 쓰기 금지다. 고쳐 쓰려면 `.copy()`를 쓴다.
  - 결과는 자기 메모리를 소유한다. `base`는 `None`이다 (새로 만든 배열과 같다, array.md §10.1).

### §1.8 실패할 때
**에러 종류** (API.md §2)
- **`TypeError`**
  - 배열 인자가 `sd.ndarray`가 아니다.
  - 옵션의 타입이 틀렸다 (§1.1). 필수 옵션을 빠뜨렸다. positional-only 인자를 키워드로 넘겼다.
  - `ord`의 타입이 틀렸다. `vector_norm`, `matrix_norm`, `cond` 모두 같다 (§3).
  - `lu_solve`의 `perm`이 int64 배열이 아니다.
  - `trace(dtype=)`의 변환이 허용되지 않는다 (§2).
- **`ValueError`**
  - 차원, 정사각, 크기, 배치 브로드캐스트가 맞지 않는다 (§1.2).
  - 옵션 값이 허용 범위 밖이다. 예: `mode='r'`, `ord=3`, `rtol=-1`, 음수 대역폭, 범위 밖 `axis`
  - LAPACK을 부르는 함수의 입력에 NaN이나 inf가 있다 (§1.9).
  - `lu_solve`의 `perm`이 순열이 아니다.
  - 값이 정의되지 않는다: 빈 행렬의 `cond` (§1.10)
  - 결과가 너무 크다: 배열을 만들 때와 같은 검사, 같은 메시지다(`array is too big`, `arrays may have at most 32 dimensions`). 브로드캐스트 뷰의 배치 shape 곱이 int64를 넘을 때나, `outer`에 32차원 입력을 줄 때(결과가 33차원) 나온다.
- **`LinAlgError`** (`ValueError`의 하위 클래스, §8): 수치적인 실패

  | 메시지 (`함수(): ` 다음) | 언제 |
  |---|---|
  | `singular matrix` | 특이 행렬을 풀거나 역행렬을 구한다 |
  | `matrix is not positive definite` | `cholesky`의 입력, `eigh`/`eigvalsh`의 `b`가 양정치가 아니다 |
  | `eigenvalue algorithm did not converge` | 고유값 계산이 수렴하지 않았다 |
  | `SVD did not converge` | SVD가 수렴하지 않았다 |
  | `a and -b have a common or close eigenvalue` | `solve_sylvester`의 해가 하나로 정해지지 않는다 |
- **`OverflowError`**
  - int 인자(`k`, `n`, `axis`, 대역폭)가 int64 범위 밖이다 (§1.1).
  - `ord`, `rtol`에 float로 바꿀 수 없을 만큼 큰 int를 넣었다.

**특이의 판정**
- LAPACK이 보고하는 **정확한 0 피벗**만 특이로 본다. 삼각 인수는 대각의 정확한 0이다.
- 조건수를 보고 "ill-conditioned"를 판정하지 않는다. 경고도 없다.
- 예외: `solve_sylvester`는 LAPACK의 근접 판정을 그대로 쓴다 (§6).
- **특이 행렬에서 에러를 내지 않는 함수**: 결과가 정의상 정상이기 때문이다.
  - `det`은 `0`, `slogdet`은 `(0, -inf)`, `cond`는 `inf`(`ord=-2`는 `0`)다.
  - `lu`, `ldl`은 분해를 그대로 돌려준다. `U`의 대각이나 `D`에 0이 나타난다.
  - `matrix_rank`, `pinv`, `lstsq`는 허용오차로 처리한다.

**배치의 실패**
- 배치 중 하나라도 실패하면 호출 전체가 실패한다. 일부만 계산된 결과는 돌려주지 않는다.
- 메시지 끝에 실패한 배치 인덱스를 붙인다: `(at batch index (i, j, ...))`
  - 인덱스는 브로드캐스트한 배치 shape 안의 위치다.
  - 배치가 없으면 붙이지 않는다.
  - 예: `singular matrix (at batch index (1,))`
- **어느 에러를 보고하는가**
  - 배치 원소 하나 안에서는 인자 순서대로 본다. 각 행렬의 특이 판정은 그 행렬의 NaN/inf 검사 바로 뒤에 온다.
    - 예: `solve`는 `a`의 NaN/inf → `a`의 특이 → `b`의 NaN/inf 순서다.
    - 예: `lu_solve`는 `l`의 NaN/inf → `u`의 NaN/inf → `u`의 특이 → `perm` → `b`의 NaN/inf 순서다.
  - 배치 전체에서는 **인덱스가 가장 작은**(C 순서) 원소의 에러 하나만 보고한다. 에러의 종류는 따지지 않는다.
    - 예: 원소 0이 특이하고 원소 1에 NaN이 있으면 원소 0의 `LinAlgError`다.
  - 보고하는 에러는 스레드 수와 무관하다 (§1.11).
- 타입, 차원, 모양, 옵션의 에러는 계산 전에 낸다. 배치 인덱스가 없다.

**메시지 형식**
- 라이브러리 공통 형식이다 (API.md §2.1). 모든 메시지가 `함수(): `로 시작하고, 인자 이름을 따옴표로 쓴다.
  - 예: `solve(): 'a' must be square in its last two dimensions, got 2x3`
- `LinAlgError`의 메시지는 `함수(): ` 다음에 위 표의 문장이다. 예: `inv(): singular matrix (at batch index (1,))`

### §1.9 유한성 검사
- **LAPACK을 부르는 함수는 계산 전에 언제나 유한성을 검사한다.** NaN이나 inf가 있으면 `ValueError`다.
  - 메시지: `inv(): 'x' contains NaN or infinity`
  - 끄는 옵션은 없다.
  - **읽는 원소만 검사한다.** 읽지 않는 삼각(§1.4)과 띠 저장 형식의 모서리(§6)는 검사하지 않는다.
  - 검사는 호출 단위다. 같은 함수라도 LAPACK을 부르는 경로만 검사한다.
- **검사하는 것**: `det`, `slogdet`, `matrix_rank`, `cond`, `matrix_norm`(`ord`가 `2`, `-2`, `'nuc'`), `matrix_power`(`n < 0`), 선형 시스템(§4), 분해(§5), 특수 구조(§6), `expm`
- **검사하지 않고 NaN/inf를 그대로 전파하는 것**: `vecdot`, `outer`, `trace`, `diagonal`, `vector_norm`, `matrix_norm`(`'fro'`, `1`, `-1`, `inf`, `-inf`), `matrix_power`(`n >= 0`), `kron`, `kronsum`, `block_diag`, `bandwidth`(NaN은 0이 아닌 원소다), `cross`
- 원소별 결과가 넘치면 inf다. 경고나 에러는 없다. 예: `expm(asarray([[1000.0]]))`는 `[[inf]]`다.

### §1.10 빈 행렬
- **수학적 관례를 따른다.**
  - `det`은 `1`, `slogdet`은 `(1, 0)`이다.
  - `inv`, `cholesky`, `expm`, 분해, 해는 그 모양의 빈 배열이다.
  - `matrix_rank`는 `0`이다. `lstsq`의 `rank`는 `0`이고 `solution`은 0이다.
  - 노름은 모든 `ord`에서 `0`이다. `vecdot`, `trace`도 `0`이다.
- **값이 정의되지 않으면 `ValueError`다**: 빈 행렬(행이나 열이 0개)의 `cond`
- **배치 축의 크기가 0이면** 결과도 그 모양의 빈 배열이다. 에러가 없다.
  - 예: `det(zeros((0, 3, 3)))`의 shape은 `(0,)`이다.
  - `cond`도 배치가 비어 있으면 빈 배열이다. 행렬 자체가 비어 있을 때만 에러다.

### §1.11 결정성과 스레드
- 스레드 수와 결정성은 API.md §3을 따른다. 결과 비트와 에러 메시지는 스레드 수와 무관하다.
- **결과는 입력의 메모리 레이아웃과 무관하다.** C 순서, F 순서, strided 입력이 같은 비트를 낸다.
- **같은 행렬은 배치 안의 어느 자리에 있어도 같은 비트를 받는다.** 배치 결과는 행렬 하나씩 부른 결과와 같다.
- 배치와 출력 하나의 긴 합은 `sd.get_num_threads()`개의 스레드로 나눠 계산한다.
- **부동소수점 합은 pairwise 합이다.** 트리는 배열의 합과 같고, 항의 순서는 논리적 C 순서다. 오차는 항 수의 log에 비례한다.
  - 대상: `vecdot`, `trace`, `vector_norm`, `matrix_norm`(`'fro'`, `±1`, `±inf`, `'nuc'`), `cond`(같은 `ord`)
  - 그냥 합(`trace`, `ord=1`과 p-노름, `matrix_norm`의 `±1`, `±inf`)은 같은 항이면 `sd.sum`과 비트까지 같다. 예: `trace(x) == sum(diagonal(x), axis=-1)`
  - 곱의 합(`vecdot`, 2-노름, `'fro'`)은 곱을 합에 융합한다(곱-덧셈을 한 번 반올림). 그래서 `vecdot(x, y)`는 `sum(conj(x) * y, axis=-1)`과 끝 비트가 다를 수 있다. 정확도는 같거나 낫다.
- **numpy·scipy와의 비트 일치는 약속하지 않는다** (API.md의 비트 규칙).
  - 약속하는 것: 위의 비트 불변성(스레드 수, 레이아웃, 배치 위치), 이 문서가 적은 우리 함수끼리의 비트 관계(예: `inv(x)`와 `solve(x, I)`), 정확도다.
  - 반올림이 아닌 동작(dtype 규칙, NaN과 ±0, 특이의 판정, 에러)은 문서대로 numpy·scipy와 같다.
  - 이 문서에서 "`np.kron`과 비트까지 같다"처럼 numpy·scipy와 비트가 같다고 적은 곳은 지금의 사실이다. 고침이나 최적화로 마지막 자리가 달라질 수 있다.

---

## §2. 기본 연산
```
vecdot(x1, x2, /, *, axis=-1)          # x1의 켤레를 쓴다
outer(x1, x2, /)                       # (..., M), (..., N) -> (..., M, N)
trace(x, /, *, k=0, dtype=None)
diagonal(x, /, *, k=0)                 # 복사본. 입력 dtype을 유지한다
matrix_power(x, n, /)
kron(x1, x2, /)                        # (..., M, N), (..., P, Q) -> (..., M*P, N*Q)
kronsum(x1, x2, /)                     # (..., M, M), (..., N, N) -> (..., M*N, M*N)
block_diag(arrays, /)                  # 블록 대각 행렬. 입력 dtype의 승격을 유지한다
cross(x1, x2, /, *, axis=-1)           # 축 길이 3
```
- **`vecdot`**
  - `sum(conj(x1) * x2, axis=axis)`이다. 결과의 shape은 브로드캐스트한 shape에서 `axis`를 뺀 것이다.
  - `axis`는 `int`만 받는다. 브로드캐스트한 shape 기준이다.
  - 축약하는 축 자체는 브로드캐스트하지 않는다. 두 입력의 그 축 길이가 같아야 한다. 다르면 `ValueError`다.
- **`outer`**
  - 마지막 축끼리의 외적이다. 배치 축은 브로드캐스트한다. Array API의 `outer`는 1차원만 받는다.
  - 켤레를 취하지 않는다.
- **`trace`**
  - 오프셋 `k`의 대각선 합이다. 대각선이 행렬 밖이면 `0`이다.
  - `dtype=None`이면 §1.5의 계산 dtype이다. bool, int64 입력도 float64다.
  - `dtype=`은 `sd.sum`의 `dtype=`과 같다 (array.md §7.5). 입력을 그 dtype으로 명시적 변환한 뒤 누적한다.
    - float → int64는 0 방향으로 자르고 포화시킨다 (array.md §1.5). int64 누적은 wrap한다 (array.md §1.6).
    - complex 입력에 실수 dtype은 `TypeError`다. 배열의 캐스팅 규칙(array.md §1.5)대로다: complex → float64/int64는 명시적 변환에서도 허수부를 잃으므로 받지 않는다. 실수부의 대각합은 `trace(sd.real(x), dtype=...)`로 구한다.
    - `dtype=sd.bool`도 `TypeError`다 (`sd.sum`과 같다).
- **`diagonal`**
  - 오프셋 `k`의 대각선을 `(..., L)`로 꺼낸 **복사본**이다. 뷰가 아니다.
  - 정사각이 아니어도 된다. 대각선이 행렬 밖이면 길이 0이다.
- **`matrix_power`**
  - `n`은 `int`다. `n = 0`이면 단위행렬, `n < 0`이면 역행렬의 `-n`제곱이다.
  - `n >= 0`이면 행렬곱만 쓴다. NaN/inf를 전파한다. `matrix_power(x, 0)`은 `x`에 NaN이 있어도 단위행렬이다.
  - `n < 0`이면 역행렬을 `inv`와 같은 경로로 구한다. 유한성을 검사하고, 특이 행렬이면 `LinAlgError`다.
    - `matrix_power(x, -1)`은 `inv(x)`와 비트까지 같다.
- **`kron`**
  - 마지막 두 축(행렬)끼리의 크로네커 곱이다. 배치 축은 다른 행렬 함수처럼 브로드캐스트한다.
  - 1차원 입력 `(N,)`은 한 행 `(1, N)`이고 배치 축이 없다. 둘 다 1차원이면 결과도 1차원 `(N·Q,)`다(np.kron, `sd.sparse.kron`과 같다). 하나만 1차원이면 결과는 행렬이다.
  - 1·2차원 입력에서는 `np.kron`과 비트까지 같다. 3차원 이상에서는 다르다: `np.kron`은 배치 축까지 곱한다.
- **`kronsum`**
  - 크로네커 합 `kron(I_N, x1) + kron(x2, I_M)`이다(`scipy.sparse.kronsum`의 뜻). `x1`은 `(..., M, M)`, `x2`는 `(..., N, N)`이고 정사각이어야 한다(`ValueError`). 배치 축은 브로드캐스트한다.
  - 결과는 M×M 블록의 N×N 격자다. 대각 블록 (i, i)는 `x1`, 블록 (i, j)의 대각은 `x2[i, j]`다. 그래서 원소마다 `x1`의 값, `x2`의 값, 또는 주대각의 합 `x1[p, p] + x2[i, i]` 하나다(덧셈 한 번, 올바르게 반올림한다). 나머지는 +0이다.
  - 단위 행렬과 곱하지 않는다. 그래서 NaN과 inf는 제자리에만 있다(밀집 `kron(I, x1)`은 inf × 0으로 NaN을 낸다).
  - 계산 dtype은 §1.5다(bool과 int64는 float64).
  - float64·complex128 입력에서 `sd.sparse.kronsum(x1, x2).toarray()`와 비트까지 같다(bool·int64는 희소가 int64를 유지한다). 다를 수 있는 것은 0의 부호뿐이다(희소는 0을 저장하지 않는다; 희소도 단위 행렬과 곱하지 않고 값을 놓는다, sparse SP111).
- **`block_diag`**
  - `arrays`는 `sd.ndarray`의 `list` 또는 `tuple`이다(`concat`, `sd.sparse.block_diag`와 같은 꼴). 빈 목록은 `ValueError`다. **[scipy와 다름]** `scipy.linalg.block_diag(*arrs)`는 가변 인자다. 같은 일을 하는 `sd.sparse.block_diag(mats)`, 배열을 잇는 `concat`, `stack`, `block`이 모두 목록 하나를 받으므로 이쪽도 그렇다.
  - 블록 `(..., Mᵢ, Nᵢ)`를 대각에 차례로 놓은 `(..., ΣMᵢ, ΣNᵢ)`다. 나머지는 0(`False`)이다. 1차원 블록 `(N,)`은 한 행 `(1, N)`이다(scipy와 같다). 빈 블록(`(0, 3)` 등)도 자리를 차지한다.
  - 배치 축은 모든 블록 사이에서 브로드캐스트한다(scipy 1.18과 같다).
  - 값을 옮기기만 한다. 결과 dtype은 입력들의 승격(array.md §1.4)이고, bool은 bool, int64는 int64다(`diagonal`과 같은 예외, §1.5). 값은 그대로 복사한다.
  - `sd.sparse.block_diag(arrays).toarray()`와 비트까지 같다(0의 부호만 다를 수 있다).
- **`cross`**
  - 길이 3 벡터의 외적이다. 켤레를 취하지 않는다.
  - `axis`는 `int`만 받는다. 브로드캐스트한 shape 기준이고, 그 축 자체는 브로드캐스트하지 않는다.
  - 두 입력 모두 그 축의 길이가 3이어야 한다. 2는 받지 않는다 (Array API).
  - 결과는 브로드캐스트한 shape, C 순서다. `np.cross`와 비트까지 같다.

---

## §3. 노름과 지표
```
vector_norm(x, /, *, axis=None, keepdims=False, ord=2)
matrix_norm(x, /, *, keepdims=False, ord='fro')
cond(x, /, *, ord=2)
det(x, /)
slogdet(x, /)                          # -> SlogdetResult(sign, logabsdet)
matrix_rank(x, /, *, rtol=None)
```
- **`vector_norm`**
  - `axis`는 `int`, `tuple`, `None`이다. `None`이면 모든 축이다. `()`이면 아무 축도 줄이지 않는다(원소별 크기).
    - 리스트도 tuple처럼 받는다. 예: `axis=[0, 1]`. 배열 영역의 `axis`와 같다 (array.md §1.3).
    - 중복이나 범위 밖은 `ValueError`다 (array.md §1.3).
  - `ord`는 `int` 또는 `float`이다. numpy 정수는 그 int다(`ord=np.int64(1)`은 `ord=1`). `bool`, `numpy.bool_`, 문자열, `None`은 `TypeError`, NaN은 `ValueError`다.

    | `ord` | 값 |
    |---|---|
    | `inf` | `max(abs(x))` |
    | `-inf` | `min(abs(x))` |
    | `0` | 0이 아닌 원소의 개수. NaN도 센다 |
    | 그 밖의 `p` (음수 포함) | `sum(abs(x)**p) ** (1/p)` |
  - NaN이 있으면 결과는 NaN이다(`ord=0` 제외). NaN 없이 inf가 있으면 `p > 0`에서 inf다.
  - 중간값이 넘치거나 비정규 수 근처로 작아지면 최댓값으로 나눠 다시 계산한다. 예: `vector_norm(asarray([1e200, 1e200]))`은 `1.414e200`이다. numpy는 `inf`다.
- **`matrix_norm`**
  - 마지막 두 축의 노름이다. `keepdims=True`이면 그 두 축을 길이 1로 남긴다.
  - `ord`

    | `ord` | 값 | LAPACK |
    |---|---|---|
    | `'fro'` | 프로베니우스 노름 | 아니다 |
    | `'nuc'` | 특이값의 합 | SVD |
    | `1` / `-1` | 열 절댓값 합의 최댓값 / 최솟값 | 아니다 |
    | `2` / `-2` | 가장 큰 / 가장 작은 특이값 | SVD |
    | `inf` / `-inf` | 행 절댓값 합의 최댓값 / 최솟값 | 아니다 |
    - 같은 값의 `float`와 numpy 정수도 받는다. 예: `ord=1.0`, `ord=np.int64(1)`
    - `str`, `int`, `float`가 아닌 타입은 `TypeError`다. `bool`(`True`)도 `TypeError`다 (`vector_norm`과 같다).
    - 타입은 맞지만 위에 없는 값은 `ValueError`다. 예: `ord=3`, 대소문자가 다른 문자열(`'FRO'`)
  - SVD를 쓰는 `ord`는 유한성을 검사하고, `svdvals`와 같은 특이값을 쓴다 (§1.9).
- **`cond`**
  - `ord`의 집합과 에러 규칙(타입은 `TypeError`, 값은 `ValueError`)은 `matrix_norm`과 같다.
  - `ord`가 `2`, `-2`이면 특이값의 비다. 정사각이 아니어도 된다.
  - 그 밖의 `ord`는 `matrix_norm(x) * matrix_norm(inv(x))`다. 정사각이어야 한다(`ValueError`).
  - 모든 `ord`에서 유한성을 검사한다.
  - **특이 행렬이면 `inf`다.** `ord=-2`는 `0`이다.
    - "특이"는 정확한 0이다 (§1.8). `ord`가 `±2`이면 가장 작은 특이값이 정확히 0일 때, 그 밖에는 LU의 정확한 0 피벗일 때다.
    - 그래서 반올림 때문에 가장 작은 특이값이 0이 아닌 행렬은 큰 유한값을 낸다. 예: `[[1, 2], [2, 4]]`는 `ord=2`에서 약 `2.76e16`, `ord=1`에서 `inf`다.
  - 특이가 아니어도 역행렬이 double 범위를 넘으면(`inv(x)`에 `inf`나 `nan`이 나오면) `inf`다. `ord=±2`가 아닌 모든 `ord`에서 같다. 예: `[[1, 0], [0, 5e-324]]`
  - 빈 행렬은 `ValueError`다 (§1.10).
- **`det`, `slogdet`**
  - LU 분해로 계산한다. 특이 행렬에서 에러를 내지 않는다 (§1.8).
  - `slogdet`의 `sign`은 실수 계산이면 `1.0`, `-1.0`, `0.0`이고, complex 계산이면 크기 1인 complex 또는 `0`이다.
  - `logabsdet`는 `log(abs(det))`이다. 특이 행렬이면 `-inf`다.
- **`matrix_rank`**
  - `rtol × (가장 큰 특이값)`보다 큰 특이값의 개수다. 경계값은 0으로 친다(`≤`).
  - `rtol >= 1`이면 언제나 `0`이다.
  - 특이값은 `svdvals`와 같은 값을 쓴다. 정사각이 아니어도 된다.

---

## §4. 선형 시스템
```
solve(a, b, /)
solve_triangular(a, b, /, *, upper, unit_diagonal=False)
cholesky_solve(l, b, /, *, upper=False)    # l은 cholesky의 결과
lu_solve(l, u, perm, b, /, *, trans='N')   # l, u, perm은 lu의 결과
lstsq(a, b, /, *, rtol=None)               # -> LstsqResult(solution, rank, singular_values)
inv(x, /)
pinv(x, /, *, rtol=None)
```
- `b`는 §1.3을 따른다. 특이 행렬은 `LinAlgError`다.
- **`solve`**: 일반 LU로 푼다 (§1.1). `a`는 정사각이다.
- **`solve_triangular`**
  - `upper`는 필수다. 위·아래를 잘못 지정하면 조용히 틀린 답이 나오기 때문이다.
  - `upper`가 고른 삼각만 읽는다 (§1.4).
  - 대각에 정확한 0이 있으면 `LinAlgError`다. `unit_diagonal=True`이면 대각을 읽지 않으므로 이 검사도 없다.
- **`cholesky_solve`**
  - `upper=False`이면 `l`은 아래 삼각 `L`이고 `a = L Lᴴ`를 푼다. `upper=True`이면 위 삼각 `U`이고 `a = Uᴴ U`를 푼다.
  - 고른 삼각만 읽는다. 대각에 정확한 0이 있으면 `LinAlgError`다.
- **`lu_solve`**
  - `a[..., perm, :] = l @ u`인 `a`에 대해 `a x = b`를 푼다.
  - `trans`는 `'N'`(`a x = b`), `'T'`(`aᵀ x = b`), `'H'`(`aᴴ x = b`; 실수면 `'T'`와 같다)다(`sd.sparse.splu(A).solve(b, trans=)`와 같은 이름과 값). 다른 문자열은 `ValueError`, 문자열이 아니면 `TypeError`다: `lu_solve(): 'trans' must be 'N', 'T' or 'H', not 'X'`. `'T'`·`'H'`는 `uᵀ y = b`, `lᵀ z = y`를 풀고 `x[perm] = z`로 놓는다(LAPACK `getrs`의 전치 풀이와 같은 순서). **[scipy와 다름]** `scipy.linalg.lu_solve`의 `trans`는 정수 0·1·2다. 이쪽은 희소의 `SparseLU.solve`와 같은 문자다.
  - `l`의 대각은 읽지 않고 1로 본다. `u`의 대각에 정확한 0이 있으면 `LinAlgError`다.
  - `perm`은 int64 배열 `(..., N)`이어야 한다(`TypeError`). 배치 원소마다 `0..N-1`의 순열이어야 한다(`ValueError`).
  - 네 인자의 배치 축을 브로드캐스트한다.
  - 계산 dtype이 같으면 `lu_solve(*lu(a), b)`는 `solve(a, b)`와 비트까지 같다.
- **`lstsq`**
  - `a`는 `(..., M, N)`이다. 정사각이 아니어도 된다.
  - 최소 노름 최소제곱해를 준다. `rtol × (가장 큰 특이값)` 이하인 특이값은 0으로 친다.
  - 결과
    - `solution`: `(..., N)` 또는 `(..., N, K)` (§1.3)
    - `rank`: 행렬 하나면 `int`, 배치면 int64 배열
    - `singular_values`: `(..., min(M, N))`, 내림차순
  - `rtol >= 1`이면 모든 특이값이 0이므로 `rank`는 `0`, `solution`은 0이다. LAPACK과 numpy는 이 경우 `rtol`을 eps로 바꿔 버린다.
  - `rtol`이 eps보다 작으면(`rtol=0` 포함) `matrix_rank`와 같은 특이값(`svdvals`의 값)으로 랭크를 정하고 SVD로 푼다. 그래서 `rank`는 `matrix_rank(a, rtol=rtol)`과 언제나 같다(사용자 결정 2026-11-19). LAPACK의 `gelsd`는 eps 아래의 특이값을 eps로 올려 정확한 0도 세었다(열 하나가 0인 30×30에서 랭크 30).
  - 남긴 특이값이 2⁻¹⁰²² 아래라 1/σ가 double을 넘어도, 해의 정확한 0은 0이고 넘치는 성분은 ±inf다(2026-11-19). 예: `lstsq(diag(1, 5e-324), [1, 1], rtol=0)`의 해는 [1, inf]다(참값 [1, 2.02e323]; 전에는 [nan, inf]). `pinv`도 같다.
  - 잔차(`residuals`)는 돌려주지 않는다.
- **`inv`**
  - `inv(x)`는 `solve(x, I)`와 비트까지 같다.
- **`pinv`**
  - `(..., M, N)` → `(..., N, M)`이다. SVD로 계산한다.
  - 허용오차는 `matrix_rank`, `lstsq`와 같은 정의다. `rtol >= 1`이면 영행렬이다.
  - 남긴 특이값이 2⁻¹⁰²² 아래여도 정확한 0인 성분은 0이고, double을 넘는 성분은 ±inf다: `pinv(diag(1, 5e-324), rtol=0)`은 [[1, 0], [0, inf]]다(2026-11-19; 전에는 [[nan, nan], [nan, inf]]).

---

## §5. 분해
```
lu(x, /)                               # -> LUResult(L, U, perm)
cholesky(x, /, *, upper=False)
qr(x, /, *, mode='reduced')            # -> QRResult(Q, R)
qr_pivoted(x, /, *, mode='reduced')    # -> QRPivotedResult(Q, R, perm)
ldl(x, /)                              # -> LDLResult(L, D, perm)
eigh(a, b=None, /)                     # -> EighResult(eigenvalues, eigenvectors)
eigvalsh(a, b=None, /)
eig(x, /)                              # -> EigResult(eigenvalues, eigenvectors)
eigvals(x, /)
schur(x, /)                            # -> SchurResult(T, Z)
hessenberg(x, /)                       # -> HessenbergResult(H, Q)
svd(x, /, *, mode='reduced')           # -> SVDResult(U, S, Vh)
svdvals(x, /)
```
`x`가 `(..., M, N)`일 때 `K = min(M, N)`이다.

| 함수 | 관계 | 결과의 모양 |
|---|---|---|
| `lu` | `x[..., perm, :] = L @ U` | `L` `(..., M, K)`, `U` `(..., K, N)`, `perm` `(..., M)` |
| `cholesky` | `x = L Lᴴ` (`upper=True`이면 `x = Uᴴ U`) | `(..., N, N)` |
| `qr` (`'reduced'`) | `x = Q @ R` | `Q` `(..., M, K)`, `R` `(..., K, N)` |
| `qr` (`'complete'`) | `x = Q @ R` | `Q` `(..., M, M)`, `R` `(..., M, N)` |
| `qr_pivoted` | `x[..., :, perm] = Q @ R` | `qr`과 같고, `perm` `(..., N)` |
| `ldl` | `x[..., perm, :][..., :, perm] = L @ D @ Lᴴ` | `L`, `D` `(..., N, N)`, `perm` `(..., N)` |
| `eigh` | `a v = λ v` (`b`가 있으면 `a v = λ b v`) | 고유값 `(..., N)`, 고유벡터 `(..., N, N)` |
| `eig` | `x v = λ v` | 고유값 `(..., N)`, 고유벡터 `(..., N, N)` |
| `schur` | `x = Z T Zᴴ` | `T`, `Z` `(..., N, N)` |
| `hessenberg` | `x = Q H Qᴴ` | `H`, `Q` `(..., N, N)` |
| `svd` (`'reduced'`) | `x = U diag(S) Vh` | `U` `(..., M, K)`, `S` `(..., K)`, `Vh` `(..., K, N)` |
| `svd` (`'complete'`) | `x = U diag(S) Vh` | `U` `(..., M, M)`, `S` `(..., K)`, `Vh` `(..., N, N)` |

- 고유벡터는 열(`v[..., :, i]`)이다. 고유벡터와 특이벡터의 부호와 위상은 정하지 않는다(LAPACK이 내는 그대로).
- `lu`, `qr`, `qr_pivoted`, `svd`, `svdvals`는 정사각이 아니어도 된다. 나머지는 정사각이어야 한다.
- **정렬 순서**
  - `eigh`, `eigvalsh`: 고유값 오름차순
  - `svd`, `svdvals`: 특이값 내림차순
  - `eig`, `eigvals`: (실수부, 허수부)의 사전순. 배열이 complex를 정렬하는 순서와 같다 (array.md §8.3). 고유벡터도 같은 순서로 재배열한다.
    - 같은 값이면 LAPACK이 낸 순서를 유지한다. 실수 입력의 켤레쌍은 허수부가 음수인 쪽이 먼저다.
    - Array API와 LAPACK은 이 순서를 정하지 않는다.
- **값만 구하는 함수**(`eigvalsh`, `eigvals`, `svdvals`)는 LAPACK의 값 전용 모드로 계산한다.
  - 그래서 `eigh`, `eig`, `svd`의 값과 반올림 수준에서 다를 수 있다.
  - `matrix_rank`, `cond`(`ord` `±2`), `matrix_norm`(`2`, `-2`, `'nuc'`)은 `svdvals`와 같은 값을 쓴다.
- **`lu`**
  - `L`은 대각이 1인 아래 삼각, `U`는 위 삼각이다. 반대쪽 삼각은 정확히 0이다.
  - 특이 행렬도 에러 없이 분해한다. `U`의 대각에 0이 나타난다.
- **`cholesky`**
  - 아래 삼각만 읽는다 (§1.4). 결과의 반대쪽 삼각은 정확히 0이다.
  - `cholesky(x, upper=True)`는 `cholesky(x)`의 켤레 전치와 비트까지 같다.
  - 양정치가 아니면 `LinAlgError`다.
- **`qr`, `qr_pivoted`**
  - `qr_pivoted`의 `R`은 대각 원소의 절댓값이 감소한다.
- **`ldl`**
  - 아래 삼각만 읽는다 (§1.4). `L`은 대각이 1인 아래 삼각이다.
  - `D`는 1×1과 2×2 블록으로 된 블록 대각 행렬이다. 꽉 찬 `(N, N)` 행렬로 돌려준다.
  - `perm`, `L`, `D`는 scipy.linalg.ldl과 같다. 우리 `L`은 scipy의 `lu[perm]`이다.
  - 특이 행렬도 에러 없이 분해한다. `D`에 0 블록이 생긴다.
- **`eigh`, `eigvalsh`**
  - 아래 삼각만 읽는다 (§1.4). 고유값은 float64다 (§1.5).
  - `b`가 있으면 일반화 고유값 문제 `a v = λ b v`를 푼다.
    - `b`는 에르미트 양정치여야 한다. 아래 삼각만 읽는다. 양정치가 아니면 `LinAlgError`다.
    - 고유벡터는 `vᴴ b v = I`로 정규화된다.
    - `a`와 `b`의 배치 축을 브로드캐스트한다. 둘 다 같은 크기의 정사각 행렬이어야 한다.
    - `a b v = λ v`, `b a v = λ v` 형태는 없다.
- **`eig`, `eigvals`**
  - 결과는 입력이 실수여도 언제나 complex128이다.
  - 고유벡터는 2-노름이 1이고, 가장 큰 성분이 실수다(LAPACK 그대로).
- **`schur`, `hessenberg`**
  - 실수 입력이면 실수 결과다. `schur`의 `T`는 2×2 블록이 있는 준삼각이다.
  - 복소 Schur 형식이 필요하면 complex128로 바꿔서 넣는다.
  - `schur`의 대각(고유값)은 정렬하지 않는다.
  - `schur`는 언제나 `Z`를, `hessenberg`는 언제나 `Q`를 함께 계산한다 (§1.6).
- **`svd`, `svdvals`**
  - `mode`의 기본값은 `'reduced'`다. `qr`과 이름과 기본값이 같다. Array API의 `full_matrices=True`와 다르다.
  - 특이값은 float64다 (§1.5).
- **수렴 실패**: 고유값 계산은 `eigenvalue algorithm did not converge`, SVD는 `SVD did not converge`의 `LinAlgError`다.

---

## §6. 특수 구조
```
solve_tridiagonal(dl, d, du, b, /)     # dl, du: (..., N-1), d: (..., N)
solve_banded(ab, b, /, *, lower_bandwidth, upper_bandwidth)
solve_sylvester(a, b, c, /)            # a x + x b = c
bandwidth(x, /)                        # -> BandwidthResult(lower, upper)
```
- **`solve_tridiagonal`**
  - `dl`은 부대각, `d`는 대각, `du`는 초대각이다. `dl`, `du`의 길이는 `d`보다 하나 짧아야 한다(`ValueError`).
  - 네 인자의 배치 축을 브로드캐스트한다.
  - `b`는 §1.3을 따른다. 특이 행렬(정확한 0 피벗)은 `LinAlgError`다.
- **`solve_banded`**
  - `ab`는 띠 저장 형식 `(..., lower_bandwidth + upper_bandwidth + 1, N)`이다: `ab[..., upper_bandwidth + i - j, j] = a[..., i, j]`
    - `ab.shape[-2]`가 맞지 않으면 `ValueError`다.
  - 두 대역폭은 필수이고, 0 이상의 `int`다. 음수는 `ValueError`다.
  - 행렬 안에 해당하는 원소만 읽고 검사한다. 모서리(행렬 밖)는 NaN이어도 된다. 대역폭이 행렬보다 넓어도 된다.
  - `lower_bandwidth = upper_bandwidth = 1`이면 `solve_tridiagonal`과 같은 루틴으로 푼다. 결과가 비트까지 같다.
  - 그 밖에는 띠 행렬의 LU다. 행이 8개 이하이면 띠 밖을 0으로 둔 밀집 행렬의 LU로 푼다(호출 고정비 때문, 결과는 반올림 수준에서 다르다).
  - `b`는 §1.3을 따른다. 특이 행렬(정확한 0 피벗)은 `LinAlgError`다.
- **`solve_sylvester`**
  - `a`는 `(..., M, M)`, `b`는 `(..., N, N)`, `c`는 `(..., M, N)`이다. 결과는 `(..., M, N)`이다.
  - `c`는 언제나 행렬이다. 1차원 `c`는 `ValueError`다 (§1.3의 규칙을 따르지 않는다).
  - 세 인자의 배치 축을 브로드캐스트한다. 결과 dtype은 세 인자로 정한다 (§1.5).
  - `a`와 `-b`가 공통(또는 eps 수준으로 가까운) 고유값을 가지면 해가 하나로 정해지지 않으므로 `LinAlgError`다.
    - 정확한 0이 아니라 LAPACK의 근접 판정이다. 문턱은 `eps · max(‖a‖, ‖b‖)` 수준이다.
  - 에러를 고르는 순서는 `a` → `a`의 Schur 분해 → `b` → `b`의 Schur 분해 → `c` → 공통 고유값이다.
- **`bandwidth`**
  - 행렬의 아래·위 대역폭이다(`scipy.linalg.bandwidth`). `lower`는 0이 아닌 원소 `x[i, j]`(i > j)의 i − j 가운데 가장 큰 것, `upper`는 j > i인 원소의 j − i 가운데 가장 큰 것이고, 없으면 0이다. 정사각이 아니어도 된다.
  - 0이 아닌 것은 `!= 0`이다. NaN은 0이 아니고 -0.0은 0이다. complex는 실수부나 허수부가 0이 아니면 0이 아니다.
  - 결과는 `BandwidthResult(lower, upper)`이고 tuple로 풀 수 있다. 2차원이면 두 필드가 Python `int`, 배치 `(..., M, N)`이면 배치 shape의 int64 배열(쓰기 금지, API.md §2.2)이다(scipy와 같다: 슬라이스마다 하나).
  - 삼각 판정은 이것으로 한다: `upper == 0`이면 아래 삼각, `lower == 0`이면 위 삼각이다(`sd.sparse.is_sptriangular`의 짝). `solve_banded`의 `lower_bandwidth`, `upper_bandwidth`에 그대로 쓸 수 있다.
  - 띠를 넓힐 수 있는 원소만 읽는다. 행렬이 띠 모양이면 원소 대부분을 읽지 않는다.
  - `sd.sparse.spbandwidth(A)`와 같은 값이다(A가 이 행렬에서 만든 희소 배열일 때: 희소는 0을 저장하지 않는다).

---

## §7. 행렬 함수
```
expm(x, /)                             # 행렬 지수 함수
```
- Al-Mohy–Higham 2009의 scaling & squaring과 Padé 근사다. scipy.linalg.expm과 같은 알고리즘이다.
- 정사각만 받는다. 유한성을 검사한다 (§1.9). 대각 행렬도 같다.
- 대각 행렬은 대각의 `exp`를 그대로 쓴다. 삼각 행렬은 결과도 삼각이고, 반대쪽 삼각은 정확히 0이다.
- 결과가 넘치면(예: 고유값의 실수부가 710보다 크면) `inf`가 나온다. 경고나 에러는 없다.

---

## §8. 예외
```
LinAlgError(ValueError)
```
- 특이 행렬, 양정치가 아닌 행렬, 수렴 실패, 해가 하나로 정해지지 않는 `solve_sylvester`에서 낸다. 메시지는 §1.8에 있다.
- `ValueError`의 하위 클래스다. `except ValueError`로도 잡힌다.
- 이 영역의 고유 예외는 이것 하나다 (API.md §2).
