# stardust-linalg 공개 API

> 3라운드에서 확정했다 (추천안 수락, 2026-11-11). 결정 이유는 DESIGN.md §2.8에 있다.

`import stardust_linalg as sdla`

## 0. 공통 규칙

### 0.1 stardust-numpy에서 이어받는 규칙 (stardust-numpy API.md §0)
- **positional-only (`/` 앞)**: 배열 인자, 그 함수의 "본 인자" (예: `matrix_power`의 `n`)
- **keyword-only (`*` 뒤)**: 이름이 붙은 옵션은 항상 keyword-only다.
- **이름**: Array API에 있는 함수는 Array API 이름을 쓴다.
- **dtype 지정**: dtype 객체만 받는다 (`sdnp.float64` 등).
- **0차원 결과**: 결과가 0차원이면 Python 스칼라를 반환한다 (Z1). 예: 행렬 하나의 `det`은 `float`이다.

### 0.2 입력
- **배열 인자는 `sdnp.ndarray`만 받는다.** 다른 객체는 `TypeError`다.
  - 리스트나 numpy 배열은 먼저 `sdnp.asarray`로 바꾼다 (numpy 배열은 복사 없이 바뀐다).
  - stardust-numpy의 원소별 함수도 `sdnp.ndarray`와 Python 스칼라만 받는다 (stardust-numpy API.md §5).
- **배치**: 배열의 마지막 두 축이 행렬이고, 그 앞의 축은 배치다. 배열이 여러 개면 배치 축끼리 브로드캐스팅한다.
- **우변 `b`의 해석** (모든 solve류 함수에 같은 규칙)
  - `b.ndim == 1`이면 벡터 하나 `(M,)`로 본다.
  - 그 외에는 행렬의 묶음 `(..., M, K)`로 본다.
  - Array API의 `solve` 규칙과 같다.
- **대칭/에르미트 입력**: 아래 삼각만 읽는다. 위 삼각은 보지 않는다. 대칭인지 검사하지 않는다.
  - 대상: `cholesky`, `eigh`, `eigvalsh`, `ldl`
  - complex 입력은 항상 에르미트로 본다 (복소 대칭 행렬은 지원하지 않는다).
- 정사각 행렬이 필요한 함수에 정사각이 아닌 행렬을 넣으면 `ValueError`다.

### 0.3 결과
- **결과 dtype은 입력 dtype만으로 정해진다.** 값에 따라 dtype이 바뀌지 않는다.
  - bool, int64 입력은 float64로 계산한다.
  - `eig`, `eigvals`는 입력이 실수여도 항상 complex128이다.
  - 고유값(`eigh`), 특이값, 노름은 complex 입력에서도 float64다.
  - `matrix_rank`와 `lstsq`의 `rank`는 int64다.
- **결과 형태는 옵션에 따라 바뀌지 않는다.** 결과의 개수가 달라지는 변형은 함수를 따로 둔다 (stardust-numpy C8과 같은 원칙).
  - 예: `eigh` / `eigvalsh`, `svd` / `svdvals`, `qr` / `qr_pivoted`
- 결과가 여러 개면 namedtuple을 반환한다.
  - 행렬 필드는 대문자 한 글자(`L`, `U`, `Q`, `R`), 벡터 필드는 소문자 단어(`perm`, `eigenvalues`)다.
- **순열**은 순열 행렬이 아니라 int64 인덱스 벡터 `perm`으로 반환한다. 결과 tuple의 마지막에 둔다.
- **레이아웃**: 행렬 결과는 F-order로 나온다 (DESIGN.md §2.3).

### 0.4 옵션 이름
| 이름 | 뜻 | 쓰는 함수 |
|---|---|---|
| `upper` | 삼각 행렬이 위 삼각인가 (bool) | `cholesky`, `cholesky_solve`, `solve_triangular` |
| `mode` | `'reduced'` 또는 `'complete'` (기본 `'reduced'`) | `qr`, `qr_pivoted`, `svd` |
| `rtol` | 가장 큰 특이값 대비 상대 허용오차. `None`이면 `max(M, N) · eps` | `matrix_rank`, `pinv`, `lstsq` |
| `ord` | 노름의 종류 | `vector_norm`, `matrix_norm`, `cond` |
| `k` | 대각선 오프셋 | `trace`, `diagonal` |
| `axis`, `keepdims`, `dtype` | stardust-numpy와 같다 | |

- 두지 않는 옵션: `out=`, `overwrite_a=`, `check_finite=` (DESIGN §2.4), `driver=` (DESIGN §2.7), `lwork=`
- 행렬의 구조(삼각, 양정치, 띠 등)는 옵션이 아니라 **함수 이름으로** 고른다. scipy의 `assume_a=` 같은 옵션은 없다.
  - `solve`는 항상 일반 LU로 푼다. 입력을 보고 구조를 자동 판별하지 않는다 (DESIGN §2.7의 휴리스틱 배제와 같은 원칙).

### 0.5 빈 행렬
- 수학적 관례를 따른다.
  - `det`은 `1`, `slogdet`은 `(1, 0)`이다 (numpy 2.5.4도 같다, 확인함).
  - `inv`, `cholesky`, 분해 결과는 빈 행렬이다.
  - `matrix_rank`는 `0`, 노름은 `0`이다.
- 값이 정의되지 않으면 `ValueError`다: `cond`
- 배치 축의 크기가 0이면 결과도 그 모양의 빈 배열이다.

---

## 1. 기본 연산
```
vecdot(x1, x2, /, *, axis=-1)          # x1의 켤레를 쓴다 (Array API)
outer(x1, x2, /)                       # (..., M), (..., N) -> (..., M, N)
trace(x, /, *, k=0, dtype=None)
diagonal(x, /, *, k=0)                 # 복사본을 반환한다
matrix_power(x, n, /)                  # n은 int. 음수면 역행렬의 거듭제곱
kron(x1, x2, /)                        # 확장 단계
cross(x1, x2, /, *, axis=-1)           # 확장 단계, 축 길이 3
```

## 2. 노름과 지표
```
vector_norm(x, /, *, axis=None, keepdims=False, ord=2)
matrix_norm(x, /, *, keepdims=False, ord='fro')     # ord: 'fro', 'nuc', 1, -1, 2, -2, inf, -inf
cond(x, /, *, ord=2)                                # ord는 matrix_norm과 같은 집합
det(x, /)
slogdet(x, /)            # -> SlogdetResult(sign, logabsdet)
matrix_rank(x, /, *, rtol=None)
```

## 3. 선형 시스템
```
solve(a, b, /)
solve_triangular(a, b, /, *, upper, unit_diagonal=False)   # upper는 필수
cholesky_solve(l, b, /, *, upper=False)    # l은 cholesky의 결과
lu_solve(l, u, perm, b, /)                 # l, u, perm은 lu의 결과
lstsq(a, b, /, *, rtol=None)    # -> LstsqResult(solution, rank, singular_values)
inv(x, /)
pinv(x, /, *, rtol=None)
```

## 4. 분해
```
lu(x, /)                        # -> LUResult(L, U, perm)            x[..., perm, :] = L @ U
cholesky(x, /, *, upper=False)  # upper=False이면 L (x = L Lᴴ), True이면 U (x = Uᴴ U)
qr(x, /, *, mode='reduced')           # -> QRResult(Q, R)               x = Q @ R
qr_pivoted(x, /, *, mode='reduced')   # -> QRPivotedResult(Q, R, perm)  x[..., :, perm] = Q @ R
ldl(x, /)                       # -> LDLResult(L, D, perm)           x[perm][:, perm] = L D Lᴴ
eigh(a, b=None, /)              # -> EighResult(eigenvalues, eigenvectors)   b가 있으면 a v = λ b v
eigvalsh(a, b=None, /)
eig(x, /)                       # -> EigResult(eigenvalues, eigenvectors)
eigvals(x, /)
schur(x, /)                     # -> SchurResult(T, Z)               x = Z T Zᴴ
hessenberg(x, /)                # -> HessenbergResult(H, Q)          x = Q H Qᴴ
svd(x, /, *, mode='reduced')    # -> SVDResult(U, S, Vh)             x = U diag(S) Vh
svdvals(x, /)
```
- **정렬 순서**
  - `eigh`, `eigvalsh`: 고유값 오름차순 (LAPACK과 같다)
  - `svd`, `svdvals`: 특이값 내림차순 (LAPACK과 같다)
  - `eig`, `eigvals`: (실수부, 허수부)의 사전순. stardust-numpy가 complex를 정렬하는 순서와 같다. 고유벡터도 같은 순서로 재배열한다.
    - Array API 2025.12는 `eig`, `eigh`의 순서를 정하지 않는다. LAPACK의 `geev`는 순서를 보장하지 않는다.
- `lu`는 정사각이 아닌 `(M, N)`도 받는다. `L`은 `(M, K)`, `U`는 `(K, N)`이다 (`K = min(M, N)`).
- `ldl`의 `D`는 1×1과 2×2 블록으로 된 블록 대각 행렬이다. 꽉 찬 `(N, N)` 행렬로 반환한다.
- `schur`, `hessenberg`: 실수 입력이면 실수 결과다 (`schur`는 2×2 블록이 있는 준삼각). 복소 Schur 형식이 필요하면 complex128로 바꿔서 넣는다.
- 고유벡터와 특이벡터의 부호와 위상은 정하지 않는다 (LAPACK이 내는 그대로).
- `eigh(a, b)`: `b`는 에르미트 양정치여야 한다. 고유벡터는 `vᴴ b v = I`로 정규화된다.
  - `a v = λ b v` 형태만 지원한다. scipy의 `type=2, 3`(`a b v = λ v`, `b a v = λ v`)은 없다.
- `schur`는 항상 `Z`를, `hessenberg`는 항상 `Q`를 함께 계산한다. 결과 형태를 고정하는 규칙(§0.3) 때문이다. 대신 그만큼 계산이 더 든다.

## 5. 특수 구조
```
solve_tridiagonal(dl, d, du, b, /)    # dl, du: (..., N-1), d: (..., N)
solve_banded(ab, b, /, *, lower_bandwidth, upper_bandwidth)   # 둘 다 필수
solve_sylvester(a, b, c, /)           # a x + x b = c를 푼다
```
- `solve_banded`의 `ab`는 띠 저장 형식이다: `(..., lower_bandwidth + upper_bandwidth + 1, N)`, `ab[..., upper_bandwidth + i - j, j] = a[..., i, j]`

## 6. 확장 단계
```
expm(x, /)
```
- randomized SVD의 시그니처는 난수 생성을 정한 뒤에 정한다.

## 7. 예외
```
LinAlgError(ValueError)
```
- 언제 내는지는 DESIGN.md §2.4에 있다.
