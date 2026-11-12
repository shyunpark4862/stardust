# 8장. `sd.linalg` 쓰기

> 37개 함수가 함께 따르는 규칙은 무엇이고, numpy와 scipy의 어떤 불일치를 정리했나?

## 1. 무엇이 있나
| 묶음 | 함수 |
|---|---|
| 기본 연산 | `vecdot`, `outer`, `trace`, `diagonal`, `matrix_power`, `kron`, `cross` |
| 노름과 지표 | `vector_norm`, `matrix_norm`, `det`, `slogdet`, `matrix_rank`, `cond` |
| 선형 시스템 | `solve`, `solve_triangular`, `cholesky_solve`, `lu_solve`, `lstsq`, `inv`, `pinv` |
| 분해 | `lu`, `cholesky`, `qr`, `qr_pivoted`, `ldl`, `eigh`, `eigvalsh`, `eig`, `eigvals`, `schur`, `hessenberg`, `svd`, `svdvals` |
| 특수 구조 | `solve_tridiagonal`, `solve_banded`, `solve_sylvester` |
| 행렬 함수 | `expm` |

- 행렬곱(`matmul`, `@`)과 전치(`.mT`)는 배열 영역에 있다. linalg가 다시 만들지 않는다.
- 분해 알고리즘은 직접 짜지 않는다. Accelerate의 LAPACK을 부른다. 직접 짠 것은 LAPACK에 없는 `expm`과, LAPACK이 필요 없는 원소 루프(`vecdot`, `trace`, 노름 등)뿐이다.
- 기준 라이브러리끼리 서로 어긋나는 곳이 많다(scipy의 `lower=` 기본값이 함수마다 다르다, `eigh(eigvals_only=)`처럼 옵션으로 결과 형태가 바뀐다). 하나의 규칙으로 정리했다. 이 장은 그 규칙이다.

## 2. 배치: 모든 함수가 행렬 묶음을 받는다
- 행렬 인자는 `(..., M, N)`이다. 마지막 두 축이 행렬 하나, 앞의 축이 배치다. 여러 인자의 배치 축은 브로드캐스트한다.
- solve류의 우변 `b`는 1차원이면 벡터 하나 `(M,)`, 그 밖에는 행렬 묶음 `(..., M, K)`다.
- 배치 크기 0은 에러가 아니라 빈 결과다.

```python
import stardust as sd

a = sd.asarray([[4.0, 2.0], [2.0, 3.0]])
batch = sd.stack([a, a * 2.0])                       # (2, 2, 2)
print(sd.linalg.det(batch).tolist())  # [8.0, 32.0]
print(sd.linalg.solve(batch, sd.asarray([1.0, 2.0])).shape)  # (2, 2)
print(sd.linalg.inv(sd.zeros((0, 3, 3))).shape)  # (0, 3, 3)
```

## 3. 입력과 dtype
- **배열 인자는 `sd.ndarray`만 받는다.** 리스트와 numpy 배열은 `sd.asarray`로 먼저 바꾼다(numpy 배열은 복사 없음, 2장 §7). 어떤 strides든 받는다: 비연속, 음수, 0(브로드캐스트).
- **계산 dtype은 float64와 complex128이다.** bool과 int64는 float64로 바꿔 계산한다. 인자 하나라도 complex면 complex로 계산한다.
- **계산하지 않는 함수는 dtype을 유지한다.** `diagonal`은 원소를 꺼내기만 한다.
- `trace(dtype=)`는 `sd.sum(dtype=)`과 같은 명시적 변환이다 (3장 §5). float → int는 0 쪽으로 자르고 포화한다.

```python
import stardust as sd

i = sd.asarray([[1, 2], [3, 4]])
print(sd.linalg.inv(i).dtype, sd.linalg.diagonal(i).dtype)  # float64 int64
print(sd.linalg.trace(i), sd.sum(sd.linalg.diagonal(i)))  # 5.0 5
try:
    sd.linalg.solve([[1.0]], [1.0])
except TypeError as e:
    print(e)  # solve(): 'a' must be an sd.ndarray, not list
```

- `trace(int 행렬)`은 float이고 `sd.sum(diagonal(int 행렬))`은 int다. 배열의 `sum`, `matmul`은 int를 int로 두지만 linalg는 BLAS/LAPACK이 float에서만 돈다. M6에서 다시 보고 그대로 두기로 했다 (DESIGN.md §9.6).

## 4. 실패: `LinAlgError`
- **수치적 실패는 `sd.linalg.LinAlgError`다.** `ValueError`의 하위 클래스다.
  - 특이 행렬, 양정치가 아닌 행렬, 수렴 실패, Sylvester의 공통 고유값
- **특이는 LAPACK이 보고하는 정확한 0 피벗뿐이다.** 조건수를 보고 "ill-conditioned"를 판정하지 않는다. 삼각 풀기(`solve_triangular` 등)는 LAPACK이 검사 없이 0으로 나누므로 대각을 직접 본다.
- **특이 행렬에서도 결과가 정의되면 에러가 아니다**: `det` = 0, `slogdet` = (0, −inf), `cond` = inf, `lu`와 `ldl`의 분해
- **배치 중 하나라도 실패하면 호출 전체가 실패한다.** 메시지에 가장 작은 실패 인덱스를 넣는다(스레드 수와 무관, 6장 §5).

```python
import stardust as sd

print(sd.linalg.slogdet(sd.zeros((2, 2))))  # SlogdetResult(sign=0.0, logabsdet=-inf)
print(sd.linalg.cond(sd.zeros((2, 2))))  # inf
try:
    sd.linalg.inv(sd.stack([sd.eye(2), sd.zeros((2, 2))]))
except sd.linalg.LinAlgError as e:
    print(e, isinstance(e, ValueError))  # inv(): singular matrix (at batch index (1,)) True
```

## 5. NaN과 inf: LAPACK을 부르면 검사한다
- **LAPACK을 부르는 호출은 항상 유한성을 검사한다.** NaN이나 inf면 `ValueError`다. LAPACK은 유한하지 않은 입력에서 결과를 보장하지 않는다. 검사하지 않으면 같은 NaN 입력에 함수마다 결과가 달라진다(쓰레기 값, 수렴 실패).
  - 비용은 O(n²)이고, 패킹과 한 패스로 한다 (9장). 끄는 옵션(`check_finite=`)은 없다.
- **LAPACK을 쓰지 않는 호출은 NaN을 전파한다**: `vecdot`, `trace`, `kron`, `matrix_norm`(fro, ±1, ±inf) 등
- 규칙은 함수가 아니라 **호출** 단위다. `matrix_norm`도 `ord=2`는 SVD를 부르므로 검사한다.
- **읽는 원소만 검사한다.** 대칭 입력(`cholesky`, `eigh`)은 아래 삼각만 읽으므로, 위 삼각에 무엇이 있든(NaN 포함) 결과에 영향이 없다.

```python
import stardust as sd

x = sd.asarray([[float("nan"), 1.0], [1.0, 1.0]])
print(sd.linalg.matrix_norm(x, ord=1))  # nan
try:
    sd.linalg.matrix_norm(x, ord=2)
except ValueError as e:
    print(e)  # matrix_norm(): 'x' contains NaN or infinity
s = sd.asarray([[4.0, float("nan")], [2.0, 3.0]])   # 위 삼각의 NaN은 읽지 않는다
print(sd.linalg.cholesky(s).tolist()[1])  # [1.0, 1.4142135623730951]
```

## 6. 이름과 인자
| 규칙 | 예 | 이유 |
|---|---|---|
| 대각 오프셋은 `k` | `diagonal(x, k=1)` | `sd.eye`의 `k`와 같다 (Array API는 `offset`) |
| `svd(mode="reduced" \| "complete")`, 기본 `"reduced"` | `qr`과 같은 이름, 같은 기본값 | 기본값에서 큰 U(M×M)를 만들지 않는다 (Array API는 `full_matrices=True`) |
| 대칭 입력은 아래 삼각만 읽는다. 결과의 삼각은 `upper=`로 고른다 | `cholesky(a, upper=True)` | scipy는 함수마다 기본값이 다르다 |
| `solve_triangular`의 `upper`는 필수다 | | 잘못 지정하면 조용히 틀린 답이 나온다 |
| 결과 형태는 옵션으로 바뀌지 않는다 | `eigh`/`eigvalsh`, `svd`/`svdvals`, `qr`/`qr_pivoted`가 따로 | 반환값의 모양이 인자에 따라 바뀌지 않는다 |
| `eig`의 고유값은 (실수부, 허수부) 사전순 | | 배열의 complex 순서와 같다. LAPACK은 순서를 정하지 않는다 |
| 분해 결과를 그대로 넘긴다 | `lu_solve(l, u, perm, b)` | scipy의 `(lu, piv)` 같은 압축 형식이 없다 |
| 순열은 int64 벡터 `perm`, 결과의 마지막 | `lu(a).perm` | 순열 행렬을 만들지 않는다 |
| 허용오차는 `rtol` 하나 | `matrix_rank(x, rtol=...)` | 가장 큰 특이값의 `rtol`배 이하를 0으로 본다 |

```python
import stardust as sd

a = sd.asarray([[4.0, 2.0], [2.0, 3.0]])
print(sd.linalg.cholesky(a, upper=True).tolist())  # [[2.0, 1.0], [0.0, 1.4142135623730951]]
r = sd.linalg.svd(sd.ones((3, 2)))
print(type(r).__name__, r._fields, r.U.shape)  # SVDResult ('U', 'S', 'Vh') (3, 2)
print(sd.linalg.eigvals(sd.asarray([[0.0, 1.0], [-2.0, 0.0]])).tolist())  # [-1.4142135623730951j, 1.4142135623730951j]
print(sd.linalg.lstsq(sd.eye(2), sd.asarray([1.0, 2.0]))._fields)  # ('solution', 'rank', 'singular_values')
```

- `lstsq`에는 `residuals`가 없다. 랭크가 모자라거나 M ≤ N이면 비거나 NaN이 되는 값이고, scipy 1.18에서 동작이 바뀌기도 했다.
- 결과 namedtuple의 필드 이름은 Array API에 그 함수가 있으면 그것을, 없으면 행렬을 대문자 한 글자로 쓴다(`L`, `U`, `Q`, `R`).
- 함수마다 LAPACK 드라이버는 하나다. `driver=` 옵션이 없고, 크기나 값을 보고 드라이버를 바꾸지 않는다. 경로가 바뀌면 같은 함수의 결과 비트가 크기에 따라 다른 계산에서 나온다 (docs/design/linalg.md §1.3).

## 7. 결과는 F-order 배열
- **행렬 결과는 열 우선(F-order)이다.** LAPACK이 만든 레이아웃을 C-order로 되돌리는 복사를 하지 않는다. 배치 축은 바깥에 C-order로 놓인다.
- 결과는 보통의 `sd.ndarray`다. 쓰기 가능하고, 자기 메모리를 가진다(`.base`는 None).
- 이후의 배열 연산은 느려지지 않는다. 이터레이터가 stride 순서로 축을 재정렬하고(3장), `matmul`은 F-order 입력을 전치 플래그로 복사 없이 BLAS에 넘긴다.
  - (64, 256, 256) F-order 대 C-order: `add` 249 대 253µs, `matmul` 2401 대 2375µs.
- 결과가 0차원이면 Python 스칼라다: 배치 없는 `det`, `trace`, `vector_norm(axis=None)`.

```python
import stardust as sd

a = sd.asarray([[4.0, 2.0], [2.0, 3.0]])
q = sd.linalg.inv(a)
print(q._flags()["f_contiguous"], q.strides, q.base)  # True (8, 16) None
print(sd.linalg.inv(sd.stack([a, a])).strides)  # (32, 8, 16)
print(type(sd.linalg.det(a)).__name__)  # float
```

## 8. 경고는 없다
- 넘치면 inf, 정의되지 않으면 에러다.
- `vector_norm`은 넘치지 않게 재조정한다. `[1e200, 1e200]`의 2-노름은 inf가 아니라 1.41e200이다 (numpy는 inf와 경고).

```python
import stardust as sd

print(sd.linalg.vector_norm(sd.asarray([1e200, 1e200])))  # 1.414213562373095e+200
```

## 더 읽을 곳
- [docs/api/linalg.md §1.1](../api/linalg.md) 시그니처와 이름, [docs/api/linalg.md §1.2](../api/linalg.md) 입력 배열, [docs/api/linalg.md §1.5](../api/linalg.md) 결과 dtype, [docs/api/linalg.md §1.8](../api/linalg.md) 실패할 때, [docs/api/linalg.md §1.9](../api/linalg.md) 유한성 검사
- [docs/design/linalg.md §2.1](../design/linalg.md) 배치와 dtype, [docs/design/linalg.md §2.2](../design/linalg.md) `LinAlgError`, [docs/design/linalg.md §2.3](../design/linalg.md) 유한성 검사, [docs/design/linalg.md §2.6](../design/linalg.md) API 규칙과 이유
- [docs/design/linalg.md §2.7](../design/linalg.md) numpy/scipy와 다르게 동작하는 곳, [docs/design/linalg.md §4](../design/linalg.md) 결과 레이아웃
