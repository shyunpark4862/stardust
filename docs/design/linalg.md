# 선형대수 영역 설계 (sd.linalg)

`sd.linalg`(선형대수 영역)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙과 통합 결정은 [DESIGN.md](../../DESIGN.md), 층 구조와 디렉터리는 [ARCHITECTURE.md](../../ARCHITECTURE.md)에 있다. 공개 함수 목록과 시그니처는 `docs/api/linalg.md`, 영역 공통 규칙(dtype, 에러 종류, 스레드)은 루트 `API.md`에 있다.
- 이 문서는 "지금 코드가 어떻게 되어 있고 왜 그런가"만 적는다. 코드 주석은 이 문서의 절 번호(`linalg.md §N.M`)를 가리킨다.
- 배열 영역의 내부(풀, pairwise 합, 버퍼 규칙)는 [array.md](array.md)를 인용한다.
- 수치는 Apple M4 Pro 한 대에서 잰 값이다. 따로 적지 않으면 float64, 기본 10스레드다. 비교 대상은 numpy 2.5.4, scipy 1.18.1이다 (§12.1).
- 통합 중 바뀔 곳은 한 줄로만 표시한다 (`통합 중 Mn`).

---

## 1. 기반

### 1.1 범위
- **있는 것**
  - 기본 연산: `vecdot`, `outer`, `trace`, `diagonal`, `matrix_power`, `kron`, `cross`
  - 노름과 지표: `vector_norm`, `matrix_norm`(fro, ±1, ±inf, ±2, nuc), `det`, `slogdet`, `matrix_rank`, `cond`
  - 선형 시스템: `solve`, `solve_triangular`, `cholesky_solve`, `lu_solve`, `lstsq`, `inv`, `pinv`
  - 분해: `lu`, `cholesky`, `qr`, `qr_pivoted`, `ldl`, `eigh`/`eigvalsh`(일반화 문제 `eigh(a, b)` 포함), `eig`/`eigvals`, `schur`, `hessenberg`, `svd`/`svdvals`
  - 특수 구조: `solve_tridiagonal`, `solve_banded`, `solve_sylvester`
  - 행렬 함수: `expm` (§11)
- **없는 것**
  - `matmul`, `matrix_transpose`: 배열 영역에 있다 (array.md §9). 다시 만들지 않는다.
  - `tensordot`, `einsum`
  - `sqrtm`, `logm`, `funm`, Toeplitz, Lyapunov 방정식
  - 희소 행렬과 반복법(CG, GMRES)
  - randomized SVD: 난수 생성이 먼저 필요하다. 시그니처가 정해지지 않았다.

### 1.2 Accelerate LAPACK을 감싼다 (`linalg/lapack.hpp`)
- **분해 알고리즘을 직접 짜지 않는다.** LAPACK을 부른다.
  - 직접 짜는 것은 LAPACK에 없는 것뿐이다: `expm`(§11), 그리고 LAPACK이 필요 없는 원소 루프(`vecdot`, `outer`, `trace`, `diagonal`, `kron`, `cross`, 노름, §10.8)
- **래퍼**: 루틴 하나에 템플릿 함수 하나다. `double`이면 `d*`, `std::complex<double>`이면 `z*`를 부른다 (예: `getrf<T>`).
  - LAPACK의 `info`를 돌려준다. 음수 `info`는 우리 인자의 버그이므로 `std::logic_error`다.
  - 감싼 루틴: `getrf getf2 getrs potrf potrs trtrs geqrf geqp3 orgqr gelsd syevd sygv gesdd gesvd sytrf geev gees gehrd orghr gtsv gbsv trsyl3 lacn2`, 그리고 BLAS `gemm`, `gemv`.
  - 회귀 영역(regression.md §3.1)이 쓰는 실수 전용 루틴: `ormqr potri larfg larf trcon`. `trcon`(1-노름 조건수의 역수 추정)은 작업 공간을 Scratch에서 받는다(§6.3). complex판(`ungqr`, `heevd`, `hegvd` 등)은 같은 래퍼가 부른다. `sygv`는 인자 하나로 `sygvd`와 `sygv`를 고른다.
  - 행렬곱(`matrix_power`, `expm`, `solve_sylvester`)은 `sd.matmul`을 거치지 않고 `cblas_dgemm`/`cblas_zgemm`을 직접 부른다. 작업 버퍼가 이미 C++ 안에 F-order로 있기 때문이다.
- **인터페이스**: 새 LAPACK, ILP64 (`ACCELERATE_NEW_LAPACK`, `ACCELERATE_LAPACK_ILP64`, array.md §1.3). 정수 인자는 `lint`(64비트)이고 `static_assert`로 확인한다.
- `accelerate/`에는 LAPACK(`lapack.hpp`, linalg), BLAS(`blas.hpp`, 배열의 `matmul`), vForce(`vforce.hpp`, 배열의 초월함수) 래퍼가 함께 있다 (ARCHITECTURE §4.4).

**Accelerate에 대해 측정한 사실** (설계의 전제)
- **내부 스레드를 쓰는 것은 `getrf`의 n ≥ 768뿐이다.**
  - `getrf`: CPU 시간/경과 시간이 n = 768에서 2.19, 1024에서 2.98이다. 1스레드 고정 대비 1.12배, 1.84배 빠르다.
  - `gesv`, `potrf`, `geqrf`, `syevd`, `gesdd`, `geev`: n = 1024까지 사실상 1스레드다 (CPU/경과 ≤ 1.11).
  - 그래서 배치가 아닌 큰 행렬 하나는 대부분 코어 하나로 계산된다. numpy와 scipy도 같은 Accelerate라서 똑같다.
- **여러 스레드에서 동시에 불러도 결과가 같다.** 7개 루틴(`getrf`, `gesv`, `potrf`, `geqrf`, `syevd`, `gesdd`, `geev`)을 n = 8, 64, 256에서 10스레드로 동시에 불렀다. 모두 호출 스레드의 결과와 비트가 같았다.
- **Accelerate의 스레드 수는 결과를 바꾸지 않는다.** n = 64, 512, 1024에서 `BLASSetThreading`으로 1스레드에 고정해도 기본 설정과 비트가 같았다(내부 스레드를 쓰는 `getrf` n = 1024 포함).
  - `BLASSetThreading`은 쓰지 않는다. 프로세스 전체 설정이라 배열 영역의 `matmul`까지 바뀐다.
- **배열의 16바이트 정렬에 따라 반올림이 달라진다** (§6.4).
- **작은 호출 일부는 코어들이 함께 쓰는 AMX에서 직렬로 돈다** (§7.5).

### 1.3 공통 원칙
- **모든 함수가 일관된 시그니처와 동작을 가진다.** 그 안에서 가급적 효율적으로 만든다. 라이브러리 전체의 일관성 원칙(DESIGN §1.4)의 linalg판이다.
- **함수마다 LAPACK 드라이버는 하나다.** `driver=` 옵션이 없고, 입력 값을 보고 드라이버를 고르지 않는다. 어느 드라이버로 할지는 속도와 정확도를 재서 정했다 (§10.1).
  - 예외는 넷이다. 모두 "입력 값으로 고르기"가 아니다.
    - `solve_banded`의 대역폭 (1, 1)은 `gtsv`로 푼다. 대역폭은 값이 아니라 옵션이다 (§10.7).
    - `solve_banded`의 n ≤ 8은 밀집 행렬의 LU로 푼다 (§7.4, 아래 크기 규칙).
    - LU 분해의 n = 32–46은 `getf2`다 (§7.5, 아래 크기 규칙).
    - `expm`의 대각·삼각 입력 처리는 알고리즘의 일부다. 결과의 의미(exp(x))는 그대로다 (§11.3).
- **구조는 함수 이름으로 고른다.** `solve`는 항상 LU다. scipy의 `assume_a=` 같은 자동 판별은 없다.
- **크기에 따라 경로를 바꾸는 것은 이득이 2배를 넘을 때만이다** (사용자 결정 2026-11-20, BP-9). 경로가 바뀌면 같은 함수의 결과 비트가 크기에 따라 다른 계산에서 나온다. 그래서 작은 이득으로는 바꾸지 않는다. `inv`의 `getri` 전환(n ≥ 256에서 최대 1.6배)은 이 이유로 택하지 않았다 (§10.2).
  - 고르는 것은 모양(크기, 대역폭)뿐이다. 값을 보고 고르지 않는다. 그래서 같은 모양이면 언제나 같은 경로이고, 레이아웃·배치 위치·스레드 수와 무관한 비트도 그대로다.
  - 경계 양쪽의 크기를 테스트한다(§13).
  - 결과를 바꾸지 않는 경로 선택(병렬 분할, 그 자리에서 읽기, 큰 행렬의 병렬 패킹)은 크기로 정해도 된다.

### 1.4 파일

| 파일 | 하는 일 |
|---|---|
| `bind/linalg_bind.cpp` | 하위 모듈 `_core.linalg` 정의 (DESIGN §3.2), 실험용 비공개 함수 (§7.7) |
| `linalg/args.*` | 배열이 아닌 인자: int(bool 거부), `dtype=`, `axis=`, `rtol=` |
| `linalg/batch.*` | 배치 브로드캐스팅(`Batch`), 작업 메모리(`Scratch`), 배치 드라이버(`run_batch`), 실패 메시지 (§6) |
| `linalg/cost.hpp` | 비용 모델, `kBandSplitMin`, `lu_task_cap` (§7) |
| `linalg/errors.*` | C++ `LinAlgError`와 Python 예외로의 변환 (§2.2) |
| `linalg/pack.hpp` | 패킹, 유한성 검사 (§5) |
| `linalg/pairwise.hpp` | pairwise 합과 긴 합의 병렬화 (§8.2, §8.3) |
| `linalg/pyrefs.*` | Python 객체 캐시 (§3.4) |
| `linalg/types.hpp` | `cplx`, `lint`, dtype 보조, 원소 곱 `mul` |
| `linalg/input.*`, `linalg/outblock.*` | 경계 코드: 빌린 입력 `Input`과 출력 블록 `Out` (§3.6) |
| `linalg/ops/common.hpp` | 실패 코드와 `raise`, 우변 규칙(`Rhs`), `LapackSlot`, `triangle_operand`, `lu_factor`, `lu_inverse` |
| `linalg/ops/lu.cpp` | `solve`, `inv`, `det`, `slogdet` |
| `linalg/ops/basic.cpp` | `vecdot`, `outer`, `trace`, `diagonal`, `matrix_power`, `kron`, `cross` |
| `linalg/ops/norms.cpp`, `vnorm.hpp` | `vector_norm`, `matrix_norm`, 노름 공통 루프 |
| `linalg/ops/factor.cpp` | `solve_triangular`, `cholesky`, `cholesky_solve`, `lu`, `lu_solve` |
| `linalg/ops/qr.cpp` | `qr`, `qr_pivoted`, `lstsq` |
| `linalg/ops/spectral.cpp`, `svals.hpp` | `eigh`, `eigvalsh`, `svd`, `svdvals`, `matrix_rank`, `pinv`, `cond`, 값 전용 특이값 |
| `linalg/ops/nonsym.cpp` | `eig`, `eigvals`, `schur`, `hessenberg`, `ldl` |
| `linalg/ops/special.cpp` | `solve_tridiagonal`, `solve_banded`, `solve_sylvester` |
| `linalg/ops/expm.cpp` | `expm` (§11) |
| `linalg/lapack.hpp` | LAPACK/BLAS 래퍼 (§1.2) |
| `src/stardust/linalg/` | `__init__.py`(재노출), `_errors.py`(`LinAlgError`). 결과 namedtuple은 세 영역 공통의 `src/stardust/_results.py` |

- 층은 아래로만 의존한다: `linalg/` → `array/`, `parallel/`, `accelerate/`, `base/` (ARCHITECTURE §1).

---

## 2. 의미론 규칙

### 2.1 배치와 dtype
- **모든 함수가 `(..., M, N)` 형태의 행렬 묶음을 받는다.** 마지막 두 축이 행렬 하나, 앞의 축이 배치다. 벡터 인자는 마지막 축 하나가 벡터다.
  - 여러 인자의 배치 축은 브로드캐스트한다 (§6.1).
  - solve류의 우변 `b`: 1차원이면 벡터 하나 `(M,)`, 그 밖에는 행렬 묶음 `(..., M, K)`다 (`rhs_of`).
- **계산 dtype은 float64와 complex128이다.** bool과 int64는 float64로 바꿔 계산한다. 변환은 패킹 안에서 한다 (§5.1).
  - 결과 dtype은 모든 인자의 dtype으로 정한다. 하나라도 complex면 complex로 계산한다.
  - **계산하지 않는 함수는 dtype을 유지한다.** `diagonal`은 원소를 꺼내기만 하므로 입력 dtype 그대로다.
- `trace(dtype=)`는 `sd.sum(dtype=)`과 같은 명시적 변환이다 (array.md §2.4). 입력을 그 dtype으로 바꾼 뒤 누적한다.
  - float → int는 0 방향으로 자르고 포화시킨다. int 누적은 wrap한다.
  - complex → 실수는 TypeError(array.md §2.4), `dtype=bool`도 TypeError다.
  - `trace(complex, dtype=float64)`의 TypeError는 의도한 동작이다. 배열의 캐스팅 규칙 그대로이고, 실수부의 대각합은 `trace(sd.real(x), dtype=...)`로 구한다 (DESIGN §9.6, docs/api/linalg.md §2).
- **경고는 내지 않는다.** 넘치면 inf, 정의되지 않으면 에러다.

### 2.2 실패: `LinAlgError`
모든 함수에 같은 규칙을 적용한다.
- **수치적 실패는 `LinAlgError`다.** `ValueError`의 하위 클래스다 (`src/stardust/linalg/_errors.py`).
  - 특이 행렬: `"singular matrix"`
  - 양정치가 아님: `"matrix is not positive definite"` (`cholesky`, `eigh(a, b)`의 `b`)
  - 수렴 실패: `"eigenvalue algorithm did not converge"`, `"SVD did not converge"`
  - Sylvester의 공통 고유값: `"a and -b have a common or close eigenvalue"` (§10.7)
- **특이는 LAPACK이 보고하는 정확한 0 피벗(`info > 0`)만이다.** 조건수를 보고 "ill-conditioned"를 판정하지 않는다.
  - 삼각 인수의 대각에 정확한 0이 있어도 특이다. `potrs`, `getrs`, `trtrs`는 검사 없이 0으로 나누므로 대각을 직접 본다 (O(n), `zero_on_diagonal`). 대상: `solve_triangular`, `cholesky_solve`, `lu_solve`.
  - 예외: `solve_sylvester`는 LAPACK(`trsyl3`)의 판정을 그대로 쓴다. Schur 분해의 반올림 때문에 정확한 특이성을 구분할 수 없다 (§10.7).
- **특이 행렬에서도 정상적인 결과가 정의되는 함수는 에러를 내지 않는다.**
  - `det` = 0, `slogdet` = (0, −inf)
  - `lu`, `ldl`: 분해는 있다. 특이성은 U의 대각, D의 블록에서 보인다.
  - `cond` = inf(`ord=-2`는 0) (§10.5)
- **배치 중 하나라도 실패하면 호출 전체가 실패한다.** 일부만 계산된 결과는 돌려주지 않는다.
  - 메시지에 실패한 배치 인덱스를 넣는다: `"inv(): singular matrix (at batch index (1, 2))"`. 배치 축이 없으면 붙이지 않는다.
  - 보고하는 것은 **가장 작은 실패 인덱스**다. 스레드 수와 무관하게 같은 메시지가 나온다 (§6.5).
- C++ 쪽은 `sd::linalg::LinAlgError`(`std::runtime_error`)를 던지고, 등록한 변환기가 Python `LinAlgError`로 바꾼다 (`errors.cpp`).
- 메시지는 함수 이름으로 시작한다 (API.md §2.1). 공개 함수는 배열처럼 `bind/errors.hpp`의 `named("inv", &inv)`로 감싸 바인딩한다(`ops/*.cpp`의 `m.def`). 함수 이름을 이미 아는 곳(배치 실패의 `raise`, `ops/common.hpp`; 인자 검사, `args.cpp`)은 직접 `func(): `를 붙이고, `named`는 이미 그 이름으로 시작하는 메시지를 그대로 둔다 (array.md §12.2).
- 시그니처: 배열 인자는 이름 없는 `nb::arg()`로 둔다. nanobind는 이름 없는 인자만 positional-only로 받기 때문이다. 대신 `m.def`마다 `nb::sig("def solve(a, b, /)")`로 api 문서의 시그니처를 그대로 준다. `help()`와 nanobind의 호출 모양 에러가 `arg0: object | None` 대신 문서의 이름을 보인다. tests/linalg/test_api_surface.py가 문서와 대조한다.

### 2.3 유한성 검사
- **LAPACK을 부르는 호출은 항상 유한성을 검사한다.** NaN이나 inf가 있으면 `ValueError`(`"det(): 'x' contains NaN or infinity"`)다.
  - 이유: LAPACK은 유한하지 않은 입력에서 결과를 보장하지 않는다. 반복 알고리즘(`eig`, `svd`)은 쓰레기 값을 내거나 수렴 실패를 보고할 수 있다. 검사하지 않으면 같은 NaN 입력에도 함수마다 결과가 달라진다.
  - 비용은 O(n²)이고 O(n³) 계산에 비해 작다. 패킹과 한 패스로 한다 (§5.1).
  - 끄는 옵션(`check_finite=`)은 없다.
- **LAPACK을 쓰지 않는 함수는 NaN을 그대로 전파한다**: `vecdot`, `outer`, `trace`, `diagonal`, `kron`, `cross`, `vector_norm`, `matrix_norm`(fro, ±1, ±inf)
- **규칙은 함수 단위가 아니라 호출 단위다.** 같은 함수라도 LAPACK을 부르는 경로만 검사한다.
  - `matrix_power`: n < 0만 검사한다(역행렬을 `getrf`로 구함). n ≥ 0은 BLAS 곱만 쓰므로 전파한다. `matrix_power(x, 0)`은 x에 NaN이 있어도 단위행렬이다.
  - `matrix_norm`: 2, −2, nuc는 SVD를 부르므로 검사한다. 나머지 ord는 전파한다.
- **읽는 원소만 검사한다.** 삼각만 읽는 함수는 그 삼각만, 띠 저장은 행렬 안의 원소만 검사한다 (§5.4, §10.7). 읽지 않는 원소는 무엇이 들어 있든(NaN 포함) 결과에 영향이 없다.
  - 대칭/에르미트 입력(`cholesky`, `eigh`, `eigvalsh`, `ldl`)은 아래 삼각만 읽는다.
  - `solve_triangular`: 지정한 삼각, `unit_diagonal`이면 대각 제외
  - `cholesky_solve`: 인수의 삼각. `lu_solve`: `l`의 대각 아래, `u`의 대각과 그 위

### 2.4 에러를 고르는 순서
- **배치 원소 하나 안에서는 인자 순서대로 본다.** 각 행렬의 특이 판정은 그 행렬의 유한성 검사 바로 뒤에 온다.
  - `solve`: `a` NaN → `a` 특이 → `b` NaN
  - `lu_solve`: `l` NaN → `u` NaN → `u` 특이 → `perm` 검증 → `b` NaN
  - `solve_tridiagonal`, `solve_banded`: `gtsv`/`gbsv`는 분해와 풀기를 한 번에 하지만, 피벗이 `b`와 무관하므로 행렬의 특이성을 `b`의 NaN보다 먼저 보고한다. 특이한 행렬과 NaN이 든 `b`는 `LinAlgError`다.
  - `solve_sylvester`: `a` → `a`의 Schur → `b` → `b`의 Schur → `c` → 공통 고유값
- **배치 전체에서는 가장 작은 인덱스의 에러 하나만 보고한다.** 종류(ValueError인지 LinAlgError인지)는 따지지 않는다.
- 구현: 배치 몸체가 실패 코드(`kNonFinite`, `kSingular`, `kNonFiniteB`, …, `ops/common.hpp`)를 돌려주고, 끝난 뒤 `raise`가 메시지를 만든다.
- 배열이 아닌 인자와 shape의 에러(TypeError, ValueError)는 배치를 돌기 전에 낸다.

### 2.5 빈 행렬과 배치 크기 0
- **빈 행렬은 수학적 관례를 따른다**: `det` = 1, `slogdet` = (1, 0), 빈 집합의 노름은 모든 `ord`에서 0이다(`-inf`, 음수 p 포함).
- **배치 크기 0은 빈 결과다.** 에러가 아니다.
- 값이 정의되지 않는 `cond`만 빈 행렬에서 `ValueError`다.

### 2.6 API 규칙과 이유
기준은 Array API다 (DESIGN §8.1). 시그니처는 `docs/api/linalg.md`에 있다. 기준 라이브러리끼리 서로 어긋나는 곳이 많아서, 하나의 규칙으로 정리했다.
- **대각 오프셋은 `k`다.** `sd.eye`와 같다. Array API의 `trace`/`diagonal`은 `offset`이라 다르다.
- **`svd`는 `mode='reduced'|'complete'`, 기본은 `'reduced'`다.** `qr`과 이름, 기본값을 맞췄다. 기본값에서 큰 U(M×M)를 만들지 않는다. Array API의 `full_matrices=True`와 다르다.
- **삼각 규칙**
  - 대칭/에르미트 입력은 아래 삼각만 읽는다. 읽을 삼각을 고르는 옵션은 없다. (scipy는 `cholesky`, `solve_triangular`가 `lower=False`, `eigh`, `ldl`이 `lower=True`다.)
  - 결과로 낼 삼각은 `upper=False`로 고른다.
  - `solve_triangular`의 `upper`는 필수다. 위·아래를 잘못 지정하면 조용히 틀린 답이 나오기 때문이다.
- **`eig`, `eigvals`의 고유값은 (실수부, 허수부) 사전순으로 정렬한다.** `sd.sort`는 complex를 받지 않으므로 이 순서는 여기서만 쓴다. Array API와 LAPACK은 순서를 정하지 않는다.
- **분해 결과를 그대로 넘긴다**: `cholesky_solve(l, b, *, upper)`, `lu_solve(l, u, perm, b)`. scipy의 `(c, lower)`, `(lu, piv)` 같은 tuple이나 압축 형식을 쓰지 않는다.
- **`lstsq`는 `(solution, rank, singular_values)`다.** `residuals`는 없다. 랭크가 부족하거나 M ≤ N이면 비거나 NaN이 되는 값이고, scipy 1.18에서 동작이 바뀌기도 했다. rank와 특이값은 `gelsd`가 추가 비용 없이 준다.
- **순열은 int64 인덱스 벡터 `perm`이고 결과 tuple의 마지막에 둔다.** 순열 행렬은 돌려주지 않는다.
- **결과 형태는 옵션으로 바뀌지 않는다.** `qr_pivoted`는 별도 함수이고, `eigh`/`eigvalsh`, `svd`/`svdvals`도 따로 있다. `schur`는 항상 `Z`를, `hessenberg`는 항상 `Q`를 함께 계산한다. 일관성을 위해 그만큼의 계산 비용을 받아들였다. (scipy는 `hessenberg(calc_q=)`, `eigh(eigvals_only=)`, `svd(compute_uv=)`, `qr(mode='r')`로 형태가 바뀐다.)
- **일반화 고유값 문제는 `eigh(a, b=None, /)`의 두 번째 인자다.** `a v = λ b v`만 지원한다.
- **배열 인자는 `sd.ndarray`만 받는다.** 리스트와 numpy 배열은 `sd.asarray`로 먼저 바꾼다(numpy 배열은 복사 없음, array.md §12.3).
- `outer`는 배치를 허용한다(Array API는 1차원만). `diagonal`은 복사본을 돌려준다.
- **허용오차는 `rtol` 하나다.** 정의: 가장 큰 특이값의 `rtol`배 이하인 특이값은 0이다. 기본값은 `max(m, n)·eps`다. 유한한 0 이상의 실수(int 또는 float) 스칼라만 받는다. 음수, NaN, inf는 ValueError, bool과 문자열은 TypeError다. Array API는 배치마다 다른 rtol(배열)도 허용하지만 받지 않는다.
- Array API와 다른 곳은 `k`, `svd(mode=)`, `outer`의 배치다. 모두 일관성을 우선했다. M6에서도 그대로 두기로 했다 (DESIGN §9.6).
- 에러 메시지는 `func(): 'arg' ...` 형식이다. M6에서 라이브러리 전체가 이 형식으로 맞췄다 (API.md §2.1). `LinAlgError`도 `inv(): singular matrix`처럼 함수 이름으로 시작한다.
- 결과 namedtuple(`SlogdetResult` 등)은 M6에서 `stardust._results` 한 모듈로 모였다 (DESIGN §9.7).

### 2.7 numpy/scipy와 다르게 동작하는 곳
설계로 정한 차이다. 비트 수준의 차이는 §12.2에 있다.

| 동작 | sd.linalg | numpy/scipy |
|---|---|---|
| 입력 | `sd.ndarray`만 | 리스트 등 |
| 경고 | 없다. `vector_norm([1e200, 1e200])`은 재조정해서 1.41e200 | inf와 경고 |
| NaN/inf | LAPACK을 부르면 항상 검사 | 함수마다 다름 (`check_finite=`) |
| `rtol ≥ 1` | 랭크 0, 해 0 (§10.4) | LAPACK이 `rcond ≥ 1`을 eps로 바꿔 랭크 최대 |
| `cond` 특이 | inf (`ord=-2`는 0) | 함수마다 다름 |
| `eig` 순서 | (실수부, 허수부) 순 | LAPACK 순서 |
| `kron` | 행렬 배치: `(..., M, N) ⊗ (..., P, Q) → (..., MP, NQ)` | `np.kron`은 모든 축을 곱한다 (2차원에서는 같음) |
| `cross` | 길이 3만 | 2, 3 |
| 정수 입력 | float64로 계산 (`diagonal`만 유지) | `kron`/`cross`는 정수 유지 |
| 구조 판별 | 함수 이름으로 | scipy `solve(assume_a=None)`은 배치마다 자동 판별 |

---

## 3. 배열과의 경계
배열과 linalg는 같은 확장 모듈 `_core` 안에 있다 (DESIGN §3.2). linalg는 `NDArray`를 C++에서 직접 주고받는다 (DESIGN §3.1, ARCHITECTURE §3).

### 3.1 얇은 Python, 두꺼운 C++
- **C++가 하는 것**: 인자 검사(배열인지, dtype, shape), 배치 브로드캐스팅, 패킹, LAPACK 호출, `info` 해석, 결과 배열 만들기, 결과 namedtuple 채우기
- **Python이 하는 것**
  - `__init__.py`: `_core.linalg`의 함수를 재노출하고, import할 때 한 번 `_core._setup(LinAlgError, stardust._results.LINALG)`로 Python 쪽 타입을 C++에 넘긴다.
  - `stardust/_results.py`(세 영역 공통): linalg의 결과 namedtuple 11개. 필드 이름은 Array API에 그 함수가 있으면 그것을 따르고, 나머지는 행렬을 대문자 한 글자로 쓴다 (docs/api/linalg.md §1.6).
  - `_errors.py`: `LinAlgError(ValueError)`
- **이유**: 함수마다 Python 래퍼를 두면 호출마다 약 1–2µs가 더 든다. 작은 행렬(3×3 `solve` 약 0.5µs)에서는 LAPACK 계산보다 크다.
- 기각한 대안: 각 함수를 Python으로 감싸고 C++은 "패킹 + LAPACK"만 맡기. 읽기 쉽지만 작은 행렬에서 호출 비용이 몇 배가 된다.

### 3.2 입력: `const NDArray&`
- 배열 인자는 C++에서 `NDArray`로 직접 받는다. nanobind 타입 테이블이 하나라서 `is_ndarray`/`as_ndarray`(`bind/pyutil.hpp`)로 바로 꺼낸다. `Input`(`linalg/input.*`)은 그 `NDArray`를 빌려 두고 에러 메시지용 이름(함수, 인자)을 함께 든다. 호출 동안 Python 인자가 배열을 살려 둔다 (ARCHITECTURE §3).
  - `sd.ndarray`가 아니면 TypeError다: `"solve(): 'a' must be an sd.ndarray, not list"`.
- 읽는 것: `data` 포인터, shape, 바이트 strides, dtype (`base/dtype.hpp`, array.md §3.1, §3.3)
- **어떤 strides든 받는다.** 연속이 아니어도, 음수나 0 stride(브로드캐스트)여도 된다. 패킹이 모든 레이아웃을 같은 F-order 버퍼로 만든다 (§5).
- 행렬 인자는 2차원 이상(`require_matrix`), 정사각이 필요하면 마지막 두 축이 같아야 한다(`require_square`). 아니면 ValueError다.
- bool/int64 → float64, float → int64(`trace(dtype=)`)의 원소 변환은 배열의 규칙을 그대로 쓴다: float → int64는 모든 영역이 쓰는 `saturate_to_int64`(`base/saturate.hpp`, array.md §2.4)다. dtype은 `base/dtype.hpp`의 `DType` 하나다.

### 3.3 결과: F-order `NDArray`
- **결과는 `NDArray`로 직접 만든다.** strides는 linalg가 정하고(`linalg/outblock.cpp`의 `alloc_matrices`, `alloc_vectors`, `alloc_c`), 메모리와 플래그는 배열의 `new_array_strided`(`array/ndarray.hpp`)가 맡는다.
  - 행렬은 열 우선이고, 배치 축은 바깥에 C-order다 (§4).
  - 벡터 결과(`(..., M)`, 고유값, 특이값, `perm`)와 행렬이 아닌 결과(노름, `det`, 랭크)는 C-order다.
  - 길이 0인 축이 있어도 strides 공식은 그대로다(0을 곱한다). 배열의 `new_array`는 0인 축을 1로 쳐서 strides가 다르다. linalg 결과의 strides는 옛 패키지와 같아야 하므로(골든 출력) linalg의 공식을 쓴다.
- **크기 검사는 배열의 `new_array`와 같다.** 축은 32개까지(`arrays may have at most 32 dimensions`), 바이트 수는 int64 안(`array is too big`)이다. 메시지 앞에 함수 이름이 붙는다.
  - 0 stride 브로드캐스트 뷰는 메모리가 작아도 shape이 커서, 배치 shape의 곱이나 결과 바이트 수가 int64를 넘을 수 있다. `Batch`가 배치 원소 수를, `alloc_*`가 결과의 축 수와 바이트 수를 `__builtin_mul_overflow`로 검사한다. `kron`은 M·P, N·Q도 검사한다.
  - 길이 0인 축이 있으면 빈 배열이지만, 나머지 축의 곱은 그래도 검사한다(numpy와 같다).
  - `outer`만 결과가 입력보다 한 차원 많다. 32차원 입력은 계산 전에 막는다.
- 메모리는 배열의 `Buffer`다. 128B로 정렬된다 (array.md §3.2). 결과는 자기 메모리를 소유한다(`base`는 None). 배열 하나인 결과는 쓰기 가능하다.
- **결과가 0차원이면 Python 스칼라다** (array.md §2.2). 예: 배치 없는 `det`, `trace`, `vector_norm(axis=None)`, `lstsq`의 `rank`. 이때는 배열을 만들지 않는다.
- 결과 namedtuple은 C++가 필드 배열을 모두 만든 뒤 캐시해 둔 타입을 vectorcall로 불러 채운다 (`make_result`). 필드 2개짜리 namedtuple 하나에 77ns다.
  - `make_result`가 배열 필드의 `writeable`을 끈다(쓰기 금지, API.md §2). 복사하지 않는다. 결과 객체의 필드는 한 계산에서 나온 값들이라 따로 고치지 않는다.

### 3.4 Python 객체 캐시 (`linalg/pyrefs.*`)
- C++가 쓰는 Python 객체는 **결과 namedtuple 타입 11개와 `LinAlgError`뿐이다.** `_setup`이 import할 때 한 번 받는다.
  - 순서는 C++ `enum class Result`와 `_results.LINALG`가 같아야 한다. 개수가 다르면 `_setup`이 ValueError를 낸다.
- 참조는 일부러 놓지 않는다. 프로세스가 끝날 때까지 살고, 인터프리터 종료 중에 놓는 것은 안전하지 않다.
- `LinAlgError`를 받기 전(이론상 import 도중)에 실패가 나면 `ValueError`로 바꾼다.

### 3.5 스레드 수
- linalg는 배열 영역과 같은 풀 하나를 쓴다 (`parallel/pool.*`, array.md §10). 풀의 크기는 `parallel::num_threads()`를 직접 읽는다.
- `sd.set_num_threads(n)` 하나가 배열과 linalg를 함께 바꾼다. 다음 호출부터 바로 반영된다.
- 나눌지 정하는 곳은 `detail::begin_parallel()` 하나다: `parallel::available()`. 태스크 안이면(중첩) false, 1스레드면 false다 (§8.1).

### 3.6 경계를 없앤 효과
- 별도 패키지였을 때 linalg는 배열을 buffer protocol로 읽고(`PyObject_GetBuffer`, format 문자열로 dtype 판정), 결과를 자기 블록에 만든 뒤 캐시해 둔 `sd.asarray`로 감쌌다. 스레드 수는 나눌 때마다 Python으로 `get_num_threads()`를 불러 자기 풀에 맞췄다.
- 그때 잰 비용: `sd.asarray`로 감싸기 113ns, 블록 할당과 해제 82ns, 입력의 `GetBuffer`/`Release` 5ns, `get_num_threads` 7ns. 결과 하나에 약 200ns로 예상했다.
- **잰 값** (커밋 4151d2a의 `bench/merge/m3_boundary.py`, 같은 자리에서 앞뒤로 측정, 단위 ns):

  | 호출 | 전 | 후 | 차이 |
  |---|---:|---:|---:|
  | `inv` 2×2 | 402 | 283 | −119 |
  | `solve` 3×3 | 414 | 300 | −115 |
  | `cholesky` 3×3 | 312 | 188 | −123 |
  | `qr` 3×3 (결과 2개) | 1178 | 935 | −243 |
  | `eigh` 3×3 (결과 2개) | 1377 | 1185 | −193 |
  | `svd` 3×3 (결과 3개) | 3246 | 3008 | −238 |
  | `det` 2×2 (스칼라) | 136 | 137 | +1 |
  | `trace` 3×3 (스칼라) | 51 | 48 | −3 |

- **결과 배열 하나에 약 120ns가 줄었다.** 예상(200ns)보다 작은 이유: `NDArray`와 그 Python 객체를 만드는 데 여전히 약 75ns가 든다. 줄어든 것은 `asarray` 호출, 블록 객체, buffer protocol 왕복이다.
- 스칼라를 돌려주는 함수(`det`, `trace`, `vector_norm`)는 결과 배열을 만들지 않으므로 그대로다. 입력 쪽 절약(약 5ns)은 측정 잡음 안이다.
- 배치 (1000, 3, 3)의 `solve`, `inv`는 결과가 배열 하나라 약 2–4%다. 계산이 대부분이다.

---

## 4. 결과 레이아웃
- **LAPACK이 만든 F-order 결과를 그대로 돌려준다.** C-order로 되돌리는 복사를 하지 않는다.
  - 배치 결과의 각 행렬은 열 우선이고, 배치 축은 바깥쪽에 C-order로 놓인다. 원소 `(b..., i, j)`의 자리는 `batch_offset + (j·m + i)·itemsize`다.
  - 값은 numpy와 같고 메모리 순서(`_flags()`)만 다르다.
- **이후 배열 연산이 느려지지 않는다.**
  - 이터레이터가 stride 순서로 축을 재정렬하고 이어진 축을 합친다 (array.md §4.2). 원소별 연산의 새 출력도 F-order 입력을 따른다.
  - `matmul`은 F-order 입력을 전치 플래그로 복사 없이 BLAS에 넘긴다 (array.md §9.2).
  - 측정 (`(64, 256, 256)`, F-order 결과 대 C-order): `add` 249µs 대 253µs, `matmul` 2401µs 대 2375µs. `sum`은 연속인 축이 바뀔 뿐이다: axis=−1은 156µs 대 127µs, axis=−2는 127µs 대 157µs.
- **입력은 F-order 작업 버퍼로 복사한다** (§5). LAPACK은 대부분 입력 행렬을 덮어쓰므로, 사용자 입력을 보존하려면 작업용 복사가 어차피 필요하다. 이 복사를 F-order로 하는 비용은 보통 복사와 비슷하다.
- 기각한 대안
  - 결과를 C-order로 되돌리기: 복사가 한 번 더 생긴다.
  - 함수마다 "C-order A = F-order Aᵀ" 트릭: `solve`, `cholesky`, `svd`에는 통하지만 `qr`, `eig`에는 안 통한다. 함수마다 방식이 달라진다. 그리고 같은 행렬이 레이아웃에 따라 다른 비트를 받는다 (§5.7).

---

## 5. 패킹 (`linalg/pack.hpp`)

### 5.1 한 패스에 세 가지
`pack<T>(src_dtype, p, m, n, rs, cs, dst, ld, check)`는 strided 행렬 하나를 F-order 버퍼(leading dimension `ld`)로 옮긴다. 한 패스로 세 가지 일을 한다.
1. 임의의 행·열 stride(바이트, 음수와 0 허용)로 읽어 열 우선으로 쓴다.
2. bool/int64를 float64로 바꾼다 (`convert`). complex 원본이 실수 버퍼로 가는 일은 없다: 계산 타입은 모든 인자를 승격해서 정한다.
3. `check`이면 유한성을 검사한다. 유한하지 않아도 행렬 전체를 복사하고 false를 돌려준다. float 원본만 검사한다(정수는 늘 유한하다).

### 5.2 출력 자리에 바로 패킹한다
- LAPACK이 결과를 입력 자리에 덮어쓰는 함수는 작업 버퍼를 따로 두지 않는다. 많은 함수가 "입력 복사 1번 + LAPACK"으로 끝난다.
  - `cholesky`: 입력의 아래 삼각을 출력 자리에 패킹하고 `potrf`를 그 자리에서 부른 뒤, 위 삼각을 0으로 채운다.
  - `solve`: `a`는 scratch에, `b`는 출력 자리에 패킹한다. `getrf` 다음 `getrs`가 출력 자리를 해로 덮어쓴다.
  - `qr`: 반사 벡터를 Q의 출력 자리에 복사하고 `orgqr`/`ungqr`가 그 자리에서 Q를 만든다.
- 출력 자리가 16바이트 정렬이 아니면 정렬된 scratch에서 계산하고 복사한다 (§6.4).

### 5.3 빠른 경로
경로는 세 가지다. 어느 경로든 원소 `(i, j)`는 `dst[j·ld + i]`로 가므로 결과는 같다.
- **열이 연속(F-order)**: 열마다 복사와 유한성 검사를 한 패스로 한다(`copy_run`).
  - 검사는 비트로 한다. 지수 필드가 모두 1이면 NaN 또는 inf다. 정수 연산이라 벡터화된다. 부동소수점 비교는 `-ffp-contract=off`에서 순서가 정해진 리덕션이 되어 벡터화되지 않는다.
- **행이 연속(C-order)**: 32×32 타일로 전치한다. 읽기와 쓰기가 모두 캐시 라인 몇 개 안에 머문다. 유한성 검사는 써 넣은 열에 비트 검사(`all_finite`)를 한 번 더 돈다.
  - 타일 8, 16, 64도 재 봤다. 차이가 잡음 수준이었다.
  - 열 순서로 그냥 읽으면 1000×1000에서 원소마다 8KB씩 건너뛴다. 큰 `solve_triangular`, `lu_solve`가 이 때문에 scipy보다 2–3배 느렸다.
- **그 밖의 strided**: 원소마다 읽는다.

### 5.4 삼각만 읽기 (`pack_triangle`)
- 삼각만 읽는 함수는 그 삼각(대각 포함 여부 선택)만 복사하고 검사한다 (§2.3). 패킹이 절반이다.
- 버퍼의 나머지는 손대지 않는다. 뒤따르는 LAPACK 루틴이 읽지 않는다.
- C-order 경로는 열마다 타일 안의 범위를 계산한다. 원소마다 분기하면 오히려 느렸다.

### 5.5 읽기만 하는 삼각은 그 자리에서 (`triangle_operand`)
- 대상: `solve_triangular`의 `a`, `cholesky_solve`의 `l`, `lu_solve`의 `l`과 `u`. LAPACK이 덮어쓰지 않는 입력이다.
- **조건**: 원소 타입이 계산 타입과 같고, 열 우선이고(행 stride = 원소 크기), 시작이 16바이트 정렬이고, 열 간격도 16바이트의 배수이고, 열 간격 ≥ n일 때
  - 이때는 복사하지 않고 읽는 삼각만 검사한다(`triangle_finite`). 아니면 `pack_triangle`로 scratch에 복사한다.
- 계산은 어느 쪽이든 같다. 결과 비트도 같다.
- C-order 입력은 여전히 전치하며 복사한다. 그 자리에서 `trans='T'`로 읽으면 레이아웃에 따라 비트가 달라진다 (§5.7).

### 5.6 큰 C-order 행렬의 전치를 풀에서
- **원소 2¹⁷개(`kPackParallelMin` = 131072) 이상인 C-order 행렬 하나**의 패킹을 열 32개 단위의 태스크로 나눈다 (`on_pool_by_columns`). 태스크마다 자기 열만 쓰므로 결과는 직렬과 같다.
  - 362² = 131044는 바로 아래다. 정사각은 363²부터 넘는다.
- 측정 (`solve_triangular`, C-order): 256²에서 1.25배, 362² 1.4배, 1000² 1.5배, 2000² 2배. `inv`는 181²–300²에서 7–11% 느려졌고 362²에서 같았다. 그래서 2¹⁷이다.
- **F-order와 strided 입력은 나누지 않는다.** strided는 나누면 0.77–0.96배였다. 이득은 전치에서만 났다.
- **작은 행렬의 경로를 지킨다.** 큰 행렬의 경로는 dtype 분기 **앞**의 크기 검사 하나와 out-of-line 함수(`pack_large`, `pack_triangle_large`, `[[gnu::noinline]]`)다.
  - 나누는 코드를 패킹 함수 안에 두었을 때는 작은 호출마다 3–10ns가 늘었다(`solve_tridiagonal` (1000,3,3) 1스레드 36 → 49µs, `cholesky` 30 → 36µs). 분기만 out-of-line으로 빼면 49 → 44µs였고, 문턱값을 constexpr로 해도 같았다.
  - 지금 남은 차이는 +1–8%, 호출당 3ns 이하다.
- 결과 (C-order 1000²): `solve_triangular` 1.44배, `cholesky_solve` 1.39배, 큰 `solve`, `det`, `slogdet`, `lu` 1.10–1.17배

### 5.7 레이아웃 불변성
- **결과는 입력 레이아웃(C, F, strided, 음수 stride, 브로드캐스트)과 무관하게 비트까지 같다.** 패킹이 모든 레이아웃을 같은 F-order 버퍼로 만들기 때문이다. 테스트로 고정한다 (§13).
- 그래서 레이아웃에 따라 LAPACK 호출을 바꾸는 최적화(C-order 입력을 `trans='T'`로 그 자리에서 읽기)는 하지 않는다. 큰 C-order `solve_triangular`가 scipy보다 느린 이유다 (§12.4).

---

## 6. 배치 드라이버 (`linalg/batch.hpp`)
모든 행렬 함수는 같은 순서를 거친다.
1. 인자 검사, 배치 축 브로드캐스팅으로 결과 shape 정하기 (`Batch`)
2. 결과 배열 할당 (§3.3)
3. `run_batch`: 배치 원소마다 패킹 → LAPACK → `info` 기록 → 후처리(삼각 0 채우기, `eig` 정렬 등). 몸체는 0 또는 실패 코드를 돌려준다.
4. 실패가 있으면 `raise` (§2.4)

### 6.1 배치 브로드캐스팅 (`Batch`)
- 피연산자 최대 4개(`kMaxOperands`)의 배치 shape을 오른쪽 정렬로 브로드캐스트한다. 크기 1인 축이 늘어난다. 맞지 않으면 ValueError(`"batch shapes do not broadcast: (2,) (3,)"`)다.
- 피연산자마다 바이트 strides를 두고, 브로드캐스트되는 축은 0이다. `offsets(idx, out)`이 C 순서의 배치 인덱스 하나를 피연산자별 바이트 오프셋으로 바꾼다.
- 배치 원소 하나는 항상 LAPACK 호출 한 번(또는 정해진 몇 번)이다. 원소끼리 독립이므로 어떻게 나눠도 결과가 같다 (§8.5).

### 6.2 `run_batch`
- `run_batch(batch, element_ns, body, max_tasks = 0)`
  - `element_ns`: 원소 하나의 추정 직렬 비용 (§7.1). 0이면 나누지 않는다.
  - `max_tasks`: 태스크 수 상한. 0이면 없다 (§7.5).
- 직렬이면 `Scratch` 하나로 원소를 차례로 돈다. 나누면 태스크마다 자기 `Scratch`를 만든다.
- 나누는 판단은 §7.2에 있다.

### 6.3 작업 메모리 (`Scratch`)
- **슬롯** 9개의 바이트 버퍼와 정수 버퍼 2개다. 슬롯은 독립된 영역이고, 필요한 만큼 커진다.
  - 슬롯 0–3(정수 0): 함수 몸체
  - 슬롯 4–5(정수 1): LAPACK 래퍼의 workspace (`work`, `rwork`, `iwork`)
  - 슬롯 6–8: 정렬되지 않은 출력 자리의 대역 (`LapackSlot`, §6.4)
- **0으로 채우지 않는다.** `std::vector`의 `resize`는 새 메모리를 0으로 채운다. 1000²에서 이 비용이 패킹만큼 들었다. 모든 사용처가 읽기 전에 쓴다.
- **작은 버퍼는 캐시 줄을 따로 갖는다.** 4KB보다 작은 버퍼는 128바이트(M 시리즈의 캐시 줄) 경계에서 시작하고 크기도 128의 배수로 올린다. 4KB부터는 보통 `new[]`다.
  - 이유: 나눈 배치에서는 태스크마다 `Scratch`가 있는데, malloc이 여러 스레드의 아주 작은 블록(`lu_solve`의 3바이트 `seen`, 24바이트 `perm`)을 같은 줄에서 내줄 때가 있다. 원소마다 쓰는 줄이라 코어 사이를 오갔다.
  - 프로세스마다 malloc 상태가 달라서 같은 빌드에서도 프로세스 단위로 갈렸다: `lu_solve` (1000, 3, 3), 10스레드에서 대부분 16.5–18µs, 줄이 겹치지 않은 프로세스만 13–14µs. 빌드가 바뀌면 그 비율이 바뀌어 커밋 사이 회귀처럼 보였다. 맞춘 뒤로는 모든 프로세스가 12.8–13.9µs다.
  - 큰 버퍼까지 정렬 할당으로 받으면 안 된다. 정렬된 `operator new`는 malloc이 놓아 둔 블록을 다시 쓰지 않아서, 큰 버퍼가 호출마다 새 페이지로 왔다. `solve_triangular` (300, 300), 1스레드에서 37 → 49µs, (1000, 1000)에서 362 → 474µs였다(M8의 같은 자리 비교에서 찾음). 작은 버퍼만 정렬하니 둘 다 원래대로다.
- **workspace 크기는 `Scratch`마다 한 번만 묻는다** (`Scratch::sizes`).
  - 키는 (루틴, 크기 인자, 원소 크기)다. 배치 안의 행렬은 모두 같은 크기라 첫 행렬만 `lwork = -1`로 묻는다. 직렬이면 배치당 한 번, 나누면 태스크당 한 번이다.
  - 행렬마다 물을 때와 **같은 `lwork`**를 쓴다. `gesdd`는 `lwork`를 보고 내부 경로를 고르므로, 크기가 바뀌면 결과 비트가 바뀐다 (§10.5).
  - 행렬마다 물을 때는 작은 행렬 배치의 분해가 scipy보다 느렸다(`qr` (1000, 3, 3) 0.75배: `geqrf`, `orgqr`에서 두 번 물었다). 이것과 §9의 고정비 정리로 작은 행렬 배치의 분해가 1.2–1.4배 빨라져 numpy와 같거나 빨라졌다.

### 6.4 16바이트 정렬 (`LapackSlot`, `ops/common.hpp`)
- **LAPACK과 BLAS에 넘기는 배열은 모두 16바이트 정렬한다.** 예외는 아래 구현의 scratch 하위 배열 몇 개와 `lstsq`의 `gelsd` 경로의 b다. b는 numpy와 비트를 맞추려고 numpy가 두는 자리(a 뒤 8mn바이트, 실수에서 m·n이 홀수이면 16바이트 경계에서 8바이트 어긋남)에 일부러 둔다(§10.4, 2026-11-19).
- 이유: Accelerate는 배열이 16바이트 정렬인지에 따라 다르게 반올림한다.
  - float64 배치에서 행렬 크기가 홀수이면(3×3이면 72바이트 간격) 홀수 번째 원소의 자리가 8바이트 어긋난다. 같은 행렬이 배치 안의 위치에 따라 다른 비트를 받았다.
  - n = 3, 5, 7, 9의 무작위 120개 중 실수 `schur` 18–30개, `svd` 3–17개, `eigh` 27–45개가 달랐다. `eigh(a, b)`도 같다. complex(원소 16바이트라 늘 정렬)는 0개였다.
  - 16의 배수지만 64의 배수가 아닌 자리(n = 2, 6, 10, 14, 18, 30)는 모두 같았다. 그래서 16바이트면 충분하다.
- 구현
  - `LapackSlot<T>(slot, count, scratch, scratch_slot)`: 출력 자리가 정렬이면 그대로, 아니면 scratch 대역을 주고 소멸할 때 자리로 복사한다. 그 자리에 쓰는 모든 것은 `get()`을 거친다.
  - scratch 안의 하위 배열(`gees`의 `wr`/`wi`, `solve_sylvester`의 R/S/U/W, 삼중대각의 세 대각)도 16바이트 경계에서 시작한다 (`pad2`).
  - 그 자리에서 읽는 삼각(§5.5)도 정렬 조건을 둔다.
  - 예외: scratch 안에서 위치가 원소마다 같은 하위 배열은 16바이트 경계가 아닐 수 있다. `lstsq` SVD 풀이(`rtol < eps`)의 `vh`(`u` 뒤 m·min(m, n)개)와 그 행 `rn`부터의 부분(`vh + rn`), `pinv`의 `u`의 열 `rn`부터의 부분(`u + rn·m`)이다. 어긋남이 배치 위치와 무관하므로 같은 행렬은 어디서나 같은 비트를 받는다. 뒤의 둘은 비정규 특이값이 있을 때만 지난다.
- 결과: 모든 함수에서 배치 결과가 행렬 하나씩 부른 결과와 비트까지 같다. 테스트로 고정한다(n = 3, 5, 7, 40, §13).

### 6.5 실패 보고
- 몸체는 첫 실패에서 코드를 돌려주고, 드라이버가 **가장 작은 실패 인덱스**를 고른다.
- 직렬: 첫 실패에서 멈춘다. 앞에서부터 보므로 그것이 가장 작은 인덱스다.
- 나눌 때: 태스크마다 첫 실패에서 멈추고 기록한다. 지금까지 알려진 가장 작은 실패 인덱스(atomic `first`)보다 뒤의 원소는 건너뛴다. 끝나면 태스크들의 기록에서 가장 작은 것을 고른다.
  - 그래서 모든 원소를 다 돌지 않아도 보고되는 인덱스가 직렬과 같다. 메시지가 스레드 수와 무관하다.
- 기각한 대안: 처음 발견한 실패에서 바로 멈추기. 빨리 끝날 수 있지만, 병렬이면 어느 인덱스가 보고될지 스레드 타이밍에 달린다.

---

## 7. 비용 모델과 배치 분할 (`linalg/cost.hpp`)

### 7.1 원소 하나의 비용: `a + c·w` ns
- 배치 원소 하나의 직렬 비용을 `a + c·w` ns로 추정한다.
  - w는 행렬의 m·n(정사각이면 n²)이다. 스트리밍 함수는 원소 하나가 건드리는 원소 수다.
- n = 2–256에서 함수마다 잰 직렬 시간에 맞췄다. 이 범위에서는 n²로 맞추는 것이 n³보다 잘 맞았다(고정비와 n² 작업이 크다). 오차는 대부분 1.2–2배, 최대 3배다. 분할 판단에는 이 정도로 충분하다.
  - 큰 n에서는 과소 추정한다(일은 n³로 자란다). 하지만 그때는 배치 2개만 돼도 이미 나눈다.

| 계열 | a | c | 함수 | w |
|---|---:|---:|---|---|
| `kLU` | 70 | 3 | `solve`, `inv`, `det`, `slogdet`, `lu`, `cholesky`, `ldl`, `matrix_power`, `cond`(±2 외) | m·n |
| `kTriangular` | 30 | 0.5 | `solve_triangular`, `cholesky_solve`, `lu_solve` | n² |
| `kQR` | 180 | 10 | `qr`, `qr_pivoted`, `hessenberg` | m·n |
| `kEigh` | 250 | 45 | `eigh`, `eigvalsh` (일반화 포함) | n² |
| `kSVD` | 1000 | 80 | `svd`, `svdvals`, `matrix_rank`, `pinv`, `cond`(±2), `lstsq`, `matrix_norm` 2/−2/nuc | m·n |
| `kEig` | 700 | 200 | `eig`, `eigvals`, `schur` | n² |
| `kSylvester` | 500 | 370 | `solve_sylvester` | (m² + n²)/2 |
| `kBanded` | 20 | 3 | `solve_tridiagonal`, `solve_banded` | n·(대역폭 + 1) |
| `kStream` | 12 | 0.3 | `vecdot`, `trace`, 노름 | 출력 하나에 더하는 원소 수 |
| `kCopy` | 2 | 0.5 | `diagonal`, `outer`, `kron`, `cross` | 복사하는 원소 수 |
| `kExpm` | 600 | 15 | `expm` | n² (n = 2–256의 직렬 시간에 0.65–1.5배로 맞음) |

- **우변 항**: 풀이 계열(`solve`, `lstsq`, `solve_triangular`, `cholesky_solve`, `lu_solve`, `solve_tridiagonal`, `solve_banded`)은 위 값에 `kRhsNs·n·k`(`kRhsNs` = 1ns, n×k는 우변, `lstsq`는 m×k)를 더한다.
  - 행렬 항만으로는 작은 행렬에 우변이 많은 호출(3×3 행렬과 우변 3·10⁶열)을 100ns 남짓으로 보고 GIL을 쥔 채 돈다. 실제로는 수 ms다.
  - 1ns는 n = 3–4에서 잰 우변 원소 하나의 시간(0.9–1.6ns, 패킹과 대입)이다. 큰 n에서는 과소 추정하지만 그때는 행렬 항이 크다.
  - 배치 하나는 나누지 않으므로 결과는 그대로다. 배치를 나누는 크기만 달라진다(결과는 원소가 어디서 돌든 같다, §6.4).
- `cond`의 `ord`가 `±2`가 아니면 LU와 역행렬로 계산하므로 `kLU`를 쓴다. `±2`는 `kSVD`다.
- 기각한 대안: 첫 원소의 시간을 재서 정하기. 상수가 필요 없지만, 실행마다 분할 여부가 달라질 수 있다.

### 7.2 두 상수
| 상수 | 값 | 근거 |
|---|---|---|
| `kParallelMinNs` | 10µs | 배치 전체의 추정 직렬 시간(`element_ns × n`)이 이보다 작으면 직렬이다. 직렬로 약 8–10µs를 넘으면 나누는 쪽이 이기기 시작했다(3×3 `svd` 배치 8은 17µs → 0.51배 시간, 3×3 `solve` 배치 64는 8µs → 0.70배). 그 아래에서 나누면 3–10% 손해다 |
| `kTaskNs` | 5µs | 태스크 하나의 추정 직렬 시간. 태스크 하나는 `max(1, kTaskNs / element_ns)`개의 원소다. 2.5µs는 작은 배치에 조금 낫고 큰 배치에 조금 나빴고, 10µs는 그 반대였다. 80µs는 태스크 수가 모자라 느렸다 |

- 나누는 조건: 배치 원소 2개 이상, `max_tasks ≠ 1`(§7.4), `element_ns × n ≥ kParallelMinNs`, 그리고 `begin_parallel()`이 참(태스크 밖, 2스레드 이상)
- 배열의 기준값 `kParallelThreshold`(원소 2¹⁵개, array.md §10.4)는 linalg의 배치 분할에 쓰지 않는다. 일의 크기를 원소 수가 아니라 시간으로 재기 때문이다 (ARCHITECTURE §4.1).

### 7.3 크기 기준은 없다
- **큰 행렬의 배치도 나눈다.** 배치 8에서 n = 128–1024 모든 크기에서 나누는 쪽이 빨랐다. 내부 스레드를 쓰는 `getrf`(n ≥ 768)의 `solve`, `inv`도 1.7–2배 빨랐다.
- 이유: Accelerate는 거의 모든 루틴을 1스레드로 돈다 (§1.2). 그래서 "큰 행렬은 Accelerate 내부 스레드에 맡긴다"는 전제가 성립하지 않는다.
- 배열 영역의 `matmul`은 다르다: 큰 행렬은 Accelerate가 스레드를 쓰므로 차례로 돈다 (array.md §9.4).

### 7.4 예외: 작은 띠의 `gbsv`는 나누지 않는다
- `gbsv`에는 원소마다 약 1µs의 고정비가 있다. 이 부분은 스레드를 늘려도 빨라지지 않았다. 작은 띠에서 나누면 1.1–2.7배 느려졌다.
- 교차점: kl = ku = 2에서 n = 64–96, kl = ku = 5에서 n = 32–48
- 그래서 `n·(kl + ku + 1) ≥ 400`(`kBandSplitMin`)일 때만 나눈다. 그 이상에서는 2–4배 빨라진다. 그 아래는 `max_tasks = 1`(나누지 않음)을 넘긴다.
  - 비용은 그대로 넘긴다. 나누지 않아도 배치 전체의 추정 시간이 기준(§7.2) 이상이면 GIL을 놓는다 (§8.1). 나눌지와 GIL을 놓을지는 따로 정한다.
- 대역폭 (1, 1)은 `gtsv`라서(§10.7) 이 문제가 없다. 항상 비용 모델대로 한다.
- **n ≤ 8은 밀집 행렬로 푼다** (`kBandDenseMax`, §1.3의 크기 규칙). 띠 밖을 0으로 채운 n × n 행렬을 LU(`lu_in_place`)와 `getrs`로 푼다. `gbsv`의 호출당 고정비가 여기서는 계산보다 크다.
  - 측정 (`gbsv` 대 밀집, 대역폭 (0, 0)–(5, 5), (0, 4)): n = 8에서 행렬 하나 2.2–2.5배, 1스레드 배치 3.4–3.6배, n = 4에서 2.7–3.2배와 6.2–6.4배다. n = 12는 행렬 하나 1.7–2.1배라 2배를 넘지 않는다. n = 24부터는 `gbsv`가 빠르다.
  - 밀집 경로는 `gbsv`의 고정비가 없으므로 비용 모델대로 나눈다(`kLU`). 10스레드 배치는 n = 8에서 21–23배다(`gbsv` 쪽은 나누지 않는 경로).
  - 읽고 검사하는 원소는 `gbsv`와 같다(행렬 안의 띠 원소만). 특이는 똑같이 정확한 0 피벗이다. 결과는 `gbsv`와 반올림 수준에서 다르다.

### 7.5 태스크 수 상한: AMX에서 직렬로 도는 경로
- **Accelerate는 두 경로를 코어들이 함께 쓰는 AMX에서 직렬로 돌린다.** 스레드를 늘려도 빨라지지 않고, 몇 개를 넘으면 오히려 느려진다.
  - `getrf`의 n = 32–40 (n ≤ 31과 n ≥ 48은 잘 나뉜다)
  - 우변 **8개 이상**의 삼각 풀기(`trsm`, `getrs`/`trtrs`/`potrs` 안)의 n = 24–40. 우변 7개 이하와 n ≤ 23은 잘 나뉜다. 경계는 정확히 k = 7과 8 사이다.
    - k ≥ 8이면 `trsm`이 블록 경로로 바뀐다. 이 경로는 호출마다 약 1.3–1.5µs의 고정비가 있고, 10스레드에서도 원소당 시간이 1스레드와 같다.
- **n = 32–46의 LU 분해는 `getf2`로 한다** (`lapack::lu_in_place`, `kGetf2Min`/`kGetf2Max`, §1.3의 크기 규칙). 블록이 없는 `getf2`는 AMX 경로를 타지 않는다. 행과 열이 모두 이 범위일 때다(`lu`의 직사각 입력도 같다). LU를 쓰는 모든 곳이 같은 규칙이다: `solve`, `inv`, `det`, `slogdet`, `lu`, `cond`(±2 외), `matrix_power`(n < 0), `expm`의 Padé 풀기.
  - 측정 (`getrf`+상한 대 `getf2`, `det` 배치): 10스레드에서 n = 32–46이 2.1–4.4배, 47은 1.96배, 48은 1.7배다. 1스레드에서도 1.2–1.6배, 행렬 하나는 1.2–1.7배다. n ≤ 31은 같고, n ≥ 64는 `getf2`가 느리다(n = 128에서 0.4배).
  - 결과는 `getrf`와 반올림 수준에서 다를 수 있다. 피벗 규칙(부분 피벗)과 특이의 판정(정확한 0 피벗)은 같다.
- **규칙** (`lu_task_cap(n, k)`, 우변을 푸는 경로만)

  | 경로 | 크기 | 상한 |
  |---|---|---:|
  | 우변 k ≥ 8 | n = 24–34 | 3 |
  | 우변 k ≥ 8 | n = 35–40 | 5 |

- 대상: `solve`, `inv`, `det`, `slogdet`, `lu`, `lu_solve`, `solve_triangular`, `cholesky_solve`, `matrix_power`(n < 0), `cond`(±2 외의 ord)
- 측정 (상한 끔/켬): 해당 크기에서 1.1–1.5배(`inv` (100, 32, 32) 1.39배), 그 밖에서는 같다.
- 어느 원소가 어디서 도는지는 결과를 바꾸지 않는다 (§8.5).
- 우변 쪽 경로를 피하는 방법(우변을 4열씩 나눠 풀기)은 n, k, 스레드 수에 따른 규칙이 필요해서 하지 않았다 (§12.5).

### 7.6 관찰: 일부 크기에서 확장성이 낮다
- 배치 512, 10스레드의 속도 향상은 대부분 5–9배다. `det` n = 32(2.1배), `cholesky` n = 48(3.2배), `matrix_power` n ≥ 48(3.6–4.3배)이 낮다.
- Accelerate의 블록 알고리즘이 행렬곱을 AMX에서 돌리고, 블록 경로가 시작되는 크기에서 떨어지는 것과 맞는다. 측정 모양에서 추론한 가설이다. 우리 코드로 바꿀 수 없는 하드웨어 쪽 한계로 보고 기록만 한다. (§7.5의 두 경로는 직접 확인해서 상한을 둔 경우다.)
- 배열의 작은 행렬 배치 `matmul`도 32×32부터 약 2배에서 멈춘다 (array.md §9.4).

### 7.7 실험용 비공개 함수
측정 실험만 쓴다. 기본값은 위의 값이고, 다른 코드는 바꾸지 않는다.

| 함수 | 바꾸는 것 |
|---|---|
| `_core.linalg._set_tuning(min_ns, task_ns)` | `kParallelMinNs`, `kTaskNs` |
| `_set_pack_parallel_min(n)` | `kPackParallelMin` (§5.6) |
| `_set_lu_task_cap(on)` | `lu_task_cap` 켜고 끄기 (§7.5) |
| `_set_getf2_range(lo, hi)` | `getf2`로 분해하는 크기 `kGetf2Min`, `kGetf2Max` (§7.5) |
| `_set_band_dense_max(n)` | `solve_banded`를 밀집으로 푸는 크기 `kBandDenseMax` (§7.4) |
| `_set_schur_pair_min(n)` | `kSchurPairMin` (§8.4) |
| `_set_expm_estimate_min(n)` | `kExpmEstimateMin` (§11.2) |

- 테스트(§13)가 이것들로 최적화를 끈 경로와 켠 경로의 비트를 비교한다.

---

## 8. 병렬화와 결정성

### 8.1 풀
- 배열 영역의 풀 하나를 쓴다 (array.md §10). 상주 워커, spin 후 대기, atomic 카운터로 태스크 분배, 호출 스레드 참여다.
- **GIL**: 배치의 추정 시간(원소당 추정 시간 × 원소 수)이 `kParallelMinNs`(10µs) 이상이면 스레드 수와 상관없이 GIL을 놓고 돈다 (API.md §3). 풀로 나누는 기준과 같은 값이다.
  - 나눠서 돌 때는 `parallel_for`가 놓는다. 나누지 않을 때(행렬이 하나, 1스레드, 풀이 바쁨)는 `run_batch`의 직렬 루프가 `parallel::Unlocked`로 놓는다. 본문은 태스크에서 도는 것과 같은 함수라 Python 객체를 만지지 않는다.
  - 나누지 않기로 한 경로(`max_tasks = 1`, 작은 띠의 `gbsv`, §7.4)도 같은 기준으로 GIL을 놓는다.
  - 긴 합(`kSumParallelMin` = 2¹⁵ 항 이상의 `pairwise_sum_mt`, `vnorm`, 행렬 노름의 줄 합)도 나누지 않을 때 `parallel::Unlocked`로 놓는다.
- **중첩하지 않는다.** 태스크 안에서는 `begin_parallel()`이 false다(`parallel::available()`). 그래서 배치를 나눈 태스크 안의 긴 합, 큰 패킹, Schur 쌍은 그 스레드에서 직렬로 돈다.
- 다른 Python 스레드가 풀을 쓰는 중이면 그 호출은 호출 스레드 혼자 돈다 (array.md §10.1).
- 태스크 안에서는 Python 객체를 만지지 않는다. 결과 배열과 namedtuple은 태스크 밖에서 만든다.
- 다른 Python 스레드의 일로 풀이 바쁘면 `available()`이 false라 그 호출은 나누지 않는다 (array.md §10.2).

### 8.2 pairwise 합 (`base/pairwise.hpp`, `linalg/pairwise.hpp`)
- **부동소수점 합은 모두 pairwise 합이다.** 대상: `vecdot`, `trace`, `vector_norm`(1, 2, p), `matrix_norm`(fro, ±1, ±inf, nuc), `cond`(fro, ±1, ±inf, nuc)
  - 트리는 배열의 합과 같다 (`base/pairwise.hpp`, array.md §6.2): 잎 하나 분량 이하는 바로 더하고, 그보다 길면 8의 배수에서 둘로 나눠 재귀한다. 트리는 길이로만 정해진다.
  - 합은 `0.0 + 트리`다.
- **두 가지 잎**
  - 그냥 합(`trace`, 1-노름과 p-노름, `matrix_norm`의 ±1, ±inf): 배열의 잎(`pairwise::leaf`, `leaf_complex`)이다. 같은 항이면 `sd.sum`과 비트까지 같다.
  - 곱의 합(`vecdot`, 2-노름과 'fro'의 제곱): 곱을 잎 안에서 합에 융합한다(`fma_dot_leaf`, `fma_cdot_leaf`, `linalg/pairwise.hpp`). NEON 레지스터 8개(레인 16개)에 `vfmaq`로 곱-덧셈을 한 번 반올림하고, 레인은 pairwise로 더한다. complex `vecdot`은 complex 4개를 레지스터 쌍으로 둔다: `x·y`의 두 레인 합이 실수부, `x·swap(y)`의 두 레인 차가 허수부다.
    - 곱마다의 반올림 하나가 없어서 오차는 같거나 작다. 여전히 log n에 비례한다.
    - 그래서 `vecdot(x, y)`는 `sum(conj(x) * y)`와, 2-노름은 `sqrt(sum(x * x))`와 끝 비트가 다를 수 있다(사용자 결정 2026-11-20, BP-4). numpy와의 비트도 약속이 아니다 (§12.2).
- **항의 순서는 논리적 C 순서다**(마지막 축이 가장 빠름). 메모리 레이아웃과 무관하므로 결과도 레이아웃과 무관하다.
- 그냥 합은 같은 항이면 배열의 합과 비트까지 같다. 테스트로 고정한다.
  - `matrix_norm`의 ±1, ±inf는 열(±1)이나 행(±inf) 하나하나를 논리 순서로 pairwise 합한다. 그 줄을 연속 벡터로 놓고 `sd.sum`으로 더한 값과 비트가 같다(테스트로 고정). **numpy의 `matrix_norm` 자체와는 끝 비트가 다를 수 있다**(사용자 결정 2026-11-18: 코드는 두고 문서를 실제대로): numpy와 배열의 `sum(axis=)`은 더하는 축이 메모리에서 연속이 아니면 바깥 축을 따라 줄을 차례로 더한다(pairwise가 아니다). 그래서 C 순서 입력의 ±1과 F 순서 입력의 ±inf는 500×500 무작위 20개 가운데 0–2개만 numpy와 비트가 같았다. C 순서의 ±inf와 F 순서의 ±1은 20개 모두 같다. 'fro'는 numpy가 다른 식을 써서 15–16개가 같다. 우리 값은 레이아웃과 무관하다.
  - `trace(x) == sum(diagonal(x))`
- **입력을 읽는 길**: 그냥 합은 잎 하나 분량을 버퍼에 채운 뒤 더한다(어떤 레이아웃이든 같은 합산 루프). 곱의 합은 계산 dtype의 연속 입력이면 **버퍼 없이** 입력을 바로 읽는다. 그 밖(strided, 다른 dtype)은 두 버퍼에 값을 바꿔 채운 뒤 같은 잎을 부른다. 잎은 항의 값만의 함수라서 두 길의 비트가 같다(레이아웃 불변, §5.7).
  - 2-노름의 연속 float64 입력은 제곱합과 최댓값(|x|), NaN을 한 번 읽으며 구한다(`sq_leaf_in_place`). `x·x`와 `|x|·|x|`는 정확히 같으므로 버퍼 길(크기를 채운 뒤 `fma_dot_leaf`)과 비트가 같다. 연속의 판정은 논리적 C 순서로 double이 이어지는지다(complex는 실수부, 허수부를 따로 센다).
  - 재조정 패스(넘쳤거나 모두 아주 작을 때, §10.8)는 그대로다: 크기를 최댓값으로 나눈 제곱을 그냥 합으로 더한다.
- complex의 2-노름은 실수부와 허수부를 따로 항으로 넣는다(`re0², im0², re1², …`). complex의 크기(`std::abs`)는 `np.abs`와 마지막 비트가 다를 수 있다. 합과는 무관한 차이다.
- **그냥 합은 잎 버퍼를 거친다.** `fill(begin, len, buf)`이 항을 잎에 채우고, 같은 루프가 더한다. 원본 레이아웃과 무관하게 더하는 루프가 하나라서 벡터화된다.
- 오차는 n이 아니라 log n에 비례해 자란다.
- 기각한 대안: 단순 누산기 8개(오차가 여전히 n에 비례하고, 어떤 기준과도 비트가 맞지 않음)
- **정의는 한 벌이다** (`base/pairwise.hpp`): 트리 모양(`split`), 그냥 합의 잎(`leaf`, `leaf_complex`), 병렬 계획과 합치기(`plan`, `combine`)를 배열과 함께 쓴다. linalg만의 것은 잎을 채우는 방식(위의 `fill`)과 곱의 합의 잎이다. 배열은 같은 정의를 접근자로 부른다 (array.md §6.2).
- **잎 하나짜리 합은 트리를 부르지 않는다**: n ≤ 128(complex는 64)이면 채우고 바로 더한다 (`pairwise_sum`). 3×3 행렬의 열 합처럼 짧은 합이 배치마다 반복될 때 재귀 함수 호출이 드러난다: `matrix_norm(ord=1)`의 (1000, 3, 3)이 이 길 없이 17µs, 있으면 13µs다(1스레드).

### 8.3 출력 하나의 긴 합
- **출력 하나의 합이 2¹⁵항(`kSumParallelMin`, 배열의 `kParallelThreshold`) 이상이면 나눈다.** 배열과 같은 방법이다 (array.md §6.2).
  - pairwise 트리는 길이로만 정해진다. 2¹²항(`kSumLeaf`, 배열의 `kParallelTask`) 이하인 부분 트리(잎)를 병렬로 더하고, 같은 트리 순서로 합친다 (`pairwise_plan`, `pairwise_run`, `pairwise_combine`).
  - 덧셈 하나하나가 직렬과 같은 순서이므로 결과는 직렬과 비트까지 같다. 곱의 합도 같은 계획을 쓴다 (`pairwise_tree_mt`).
- 대상: `vecdot`, `trace`, `vector_norm`(모든 ord), `matrix_norm` fro
  - 최댓값, 최솟값, 개수(`ord=0`)는 순서와 무관하므로 잎마다 구해 합친다.
  - 2-노름의 최댓값과 NaN 추적은 잎마다 따로 둔다. 공유 변수에 쓰면 경쟁이 생긴다.
- `matrix_norm` 1/inf는 큰 행렬 하나의 열(행) 합들을 태스크로 나눈다. 줄 하나의 합은 직렬과 같고, max/min은 순서와 무관하다.
- **짧은 합에는 아무것도 할당하지 않는다.** 2¹⁵항 미만이면 계획을 만들지 않고 바로 직렬 함수로 간다. 처음에는 출력마다 병렬 계획용 `std::vector`를 2–3개 만들었는데, 출력이 많은 짧은 노름이 5배 느려졌다(태스크 안에서는 할당끼리 경쟁까지 했다).
- 결과: 출력 하나의 긴 합 5–6배, 행 노름 4–8배(10스레드 대 1스레드)

### 8.4 함수 하나 안의 병렬
- **`solve_sylvester`의 Schur 분해 두 개를 동시에 돈다** (`kSchurPairMin` = 8).
  - `gees`는 Accelerate 안에서 코어 하나만 쓴다. `a`와 `b`의 분해는 독립이므로 풀의 태스크 두 개로 돌리고, `b`의 분해는 두 번째 `Scratch`를 쓴다. 같은 배열과 workspace 크기라서 비트가 같다(m, n = 3–200, 실수/복소수).
  - 측정: min(m, n) < 6에서 0.76–0.96배, 6에서 1.05배, 8–24에서 1.1–1.3배, 32 이상에서 1.4–1.7배. 그래서 8이다.
  - 결과: `solve_sylvester` 300², 1000² 1.6배, 32² 1.35배
  - `b`가 유한하지 않으면 동시에 돌리지 않는다. 에러 순서(§2.4)가 그대로다.
  - 배치를 나눈 태스크 안에서는 중첩 금지로 차례로 돈다.
- 큰 C-order 행렬의 패킹 (§5.6)

### 8.5 결정성
**결과는 스레드 수와 무관하게 비트까지 같다** (DESIGN §1.5, array.md §10.5).

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 배치 | 원소 묶음 (§7) | 원소 하나는 어디서 돌든 같은 계산이다: 자기 `Scratch`, 같은 `lwork`(§6.3), 16바이트 정렬(§6.4), LAPACK 결과는 스레드와 무관(§1.2) |
| 실패 보고 | 태스크마다 첫 실패 | 가장 작은 인덱스를 고른다 (§6.5) |
| 출력 하나의 긴 합 | pairwise 트리의 잎 | 같은 트리 순서로 합친다 (§8.3) |
| `matrix_norm` 1/inf의 줄 | 줄 묶음 | 줄 하나의 합은 직렬과 같고, max/min은 순서와 무관 |
| 큰 C-order 패킹 | 열 32개 블록 | 복사뿐이다 (§5.6) |
| Schur 쌍 | 두 분해 | 같은 배열과 workspace (§8.4) |
| 태스크 수 상한 | — | 어느 원소가 어디서 도는지만 바뀐다 (§7.5) |

---

## 9. 호출 비용
작은 행렬에서는 LAPACK 계산이 아니라 호출 경로의 고정비가 시간을 정한다. 결과를 바꾸지 않고 줄인 것들이다.
- **경계**: 배열을 직접 받고 직접 만든다 (§3). buffer protocol과 `sd.asarray`를 거치던 때보다 결과 배열 하나에 약 120ns 빠르다 (§3.6).
- **Python 래퍼 없음**: 함수는 C++에서 바로 바인딩한다 (§3.1).
- **workspace 크기 기억, 0 채우기 없음** (§6.3)
- **짧은 합에는 할당하지 않는다** (§8.3)
- **작은 행렬 경로에는 크기 검사 하나만** (§5.6)
- **함수별 작은 고정비**
  - `diagonal`: 원소마다 `memcpy`를 부르던 것을 타입이 정해진 복사로 바꿨다.
  - `matrix_power`: r의 복사 두 번을 없애고, 마지막 곱을 출력 자리에 바로 쓴다. 곱은 scratch 두 개를 번갈아 쓴다.
  - `eig`: 정렬용 인덱스를 행렬마다 `std::vector`로 만들던 것을 scratch로 바꿨다.
- **남은 고정비** (10스레드, 짧은 호출): `trace` 3×3 0.05µs, `det` 0.18µs, `diagonal` 0.27µs(그중 결과 감싸기 0.16µs). 이미 numpy보다 2–8배 빠르다.
  - (1000, 3, 3) 배치의 `matrix_norm` fro는 원소당 28ns(numpy 9ns)지만 절대량이 10µs 안팎이다.
- 결과: 3×3 `solve` 0.48µs 대 numpy 1.99µs. 작은 행렬 하나의 기하평균 2.8배 (§12.4)

---

## 10. 함수별 메모

### 10.1 드라이버
모두 속도와 정확도를 재서 골랐다. numpy는 `gesv`, `gesdd`, `syevd`, `gelsd`를 쓴다.

| 함수 | LAPACK | 근거 |
|---|---|---|
| `solve` | `getrf` + `getrs` | `gesv`보다 거의 모든 크기에서 빠르다 (§10.2) |
| `inv` | `getrf` + 단위행렬 우변 `getrs` | n ≤ 128에서 `getri`보다 빠르고, `inv(a) == solve(a, I)` (§10.2) |
| `det`, `slogdet`, `lu` | `getrf` | — |
| `lu_solve` | `trtrs` 두 번 (L 단위 대각, U) | `getrs`와 비트가 같고 L, U가 따로 있어도 된다 (§10.2) |
| `solve_triangular` | `trtrs` | — |
| `cholesky`, `cholesky_solve` | `potrf('L')`, `potrs` | — |
| `qr`, `qr_pivoted` | `geqrf`/`geqp3` + `orgqr`/`ungqr` | numpy와 같은 호출 |
| `lstsq` | `gelsd` | §10.4 |
| `eigh` | `syevd`/`heevd` | n = 1000: 49ms. `syevr` 91ms, `syev` 474ms. 모든 크기에서 가장 빠르거나 같다 |
| `eigh(a, b)` | `sygvd`/`hegvd` | 57ms. `sygv` 474ms |
| `eigvalsh` | `syevd` 'N' | 값만 구할 때는 세 드라이버가 같다(약 30ms). `eigh`와 같은 계열 |
| `eigvalsh(a, b)` | `sygv` 'N' | 36ms. `sygvd` 'N' 74ms |
| `svd`, `pinv` | `gesdd` | 64ms. `gesvd` 681ms |
| `svdvals`, `matrix_rank`, `cond` ±2, `matrix_norm` 2/−2/nuc | `gesvd` 'N' | `gesdd` 'N'보다 작은 크기에서 25%, 큰 크기에서 5% 빠르다 |
| `eig`, `eigvals` | `geev` ('V' / 'N') | — |
| `schur` | `gees` (정렬 없음) | — |
| `hessenberg` | `gehrd` + `orghr`/`unghr` | — |
| `ldl` | `sytrf`/`hetrf` (Bunch–Kaufman) | §10.6 |
| `solve_tridiagonal` | `gtsv` | §10.7 |
| `solve_banded` | `gbsv`, 대역폭 (1, 1)이면 `gtsv` | §10.7 |
| `solve_sylvester` | `gees` 두 번 + `trsyl3` + `gemm` 네 번 | §10.7 |
| `expm` | `gemm`, `gemv`, `getrf` + `getrs`, `lacn2` | §11 |

- 같은 함수의 '값만'과 '벡터 포함'은 서로 다른 함수다. 그래서 드라이버가 달라도 "함수마다 하나"(§1.3)에 맞는다.

### 10.2 LU 계열 (`solve`, `inv`, `det`, `slogdet`, `lu`, `lu_solve`)
- **`solve`는 `getrf` + `getrs`다.** n = 2–2048, 우변 1, 16, n개로 쟀다.
  - 작은 행렬에서 30–40% 빠르다. n = 1024에서 1.6배, 2048에서 2.2배 빠르다. Accelerate의 `gesv`는 스레드를 쓰는 `getrf` 경로를 타지 않는다(n = 1024에서 `gesv` 9.0ms, `getrf` 4.8ms). n = 512만 2% 느리다(측정 오차 수준).
  - n ≥ 4에서 `gesv`와 결과 비트가 다르다. Accelerate의 `gesv`는 내부적으로 다른 계산 순서를 쓴다. 그래서 `solve`는 numpy와 비트까지 같지는 않다.
- **`inv`는 `getrf` 다음 단위행렬을 우변으로 둔 `getrs`다** (`lu_inverse`).
  - 측정 (µs, `getrs(I)` 대 `getri`): n = 3은 0.10 대 0.28, 64는 19.8 대 34.5, 256은 348 대 332, 1024는 13390 대 9740, 2048은 95573 대 61410
  - n ≤ 128에서 빠르고, `inv(a)`가 `solve(a, I)`와 비트까지 같아진다. 대신 n ≥ 256에서는 `getri`보다 최대 1.6배 느리다. 크기에 따라 바꾸는 안은 §1.3에 따라 택하지 않았다.
- `det`, `slogdet`는 U의 대각 곱과 피벗 교환 횟수의 부호로 계산한다. 특이(`info > 0`)도 에러가 아니다 (§2.2).
- `lu`의 `perm`은 `ipiv`(교환 목록)를 순열로 바꾼 것이다. `x[perm] = L U`다. 특이 행렬에서도 에러를 내지 않는다.
- **complex LU의 피벗 경계**: `zgetrf`는 `|re| + |im|`이 가장 큰 원소를 피벗으로 고른다(`izamax`). 그래서 보장되는 것은 `|L| ≤ √2`다(실수는 1). 테스트도 이 경계로 확인한다.
- **`lu_solve(l, u, perm, b)`**
  - `l`의 대각은 읽지 않고 1로 본다. `u`의 대각에 정확한 0이 있으면 `LinAlgError`다.
  - `perm`은 int64 배열이어야 하고(TypeError), 배치 원소마다 0..n−1의 순열이어야 한다(ValueError).
  - 계산: `x = b[perm]`(행을 정확히 모으므로 반올림이 없다) → `trtrs`를 L(단위 대각)과 U에 한 번씩. n = 3–1000에서 `getrs`(교환 없음)와 비트가 같음을 확인했다. L, U를 `getrf`의 압축 형식으로 다시 붙일 필요가 없고, 그 자리에서 읽을 수 있다 (§5.5).
  - 그래서 계산 타입이 같으면 `lu_solve(*lu(a), b) == solve(a, b)`다. 실수 `a`와 complex `b`이면 반올림 수준에서만 같다: `solve`는 `a`의 complex 사본을 `zgetrf`로, `lu`는 실수 `dgetrf`로 분해한다.
- 비트 관계: `inv(a) == solve(a, I)`, `matrix_power(x, -1) == inv(x)`, `lu_solve(*lu(a), b) == solve(a, b)`. 테스트로 고정한다.

### 10.3 삼각과 Cholesky (`solve_triangular`, `cholesky`, `cholesky_solve`)
- `solve_triangular`: `trtrs`. 대각의 정확한 0을 LAPACK 호출 **전에** 확인한다. 그래야 에러 순서(§2.4)를 지킨다.
- **`cholesky(x, upper=True)`는 `cholesky(x)`의 켤레 전치다.** `potrf('L')`로 L을 구한 뒤 `U = Lᴴ`로 옮겨 쓴다. 두 결과가 비트까지 같다. `potrf('U')`를 따로 부르지 않는다.
  - 양정치가 아니면(`info > 0`) `LinAlgError("matrix is not positive definite")`다.
- `cholesky_solve`: `potrs`. 인수의 대각에 정확한 0이 있으면 `LinAlgError("singular matrix")`다.

### 10.4 QR과 최소제곱 (`qr`, `qr_pivoted`, `lstsq`)
- `qr`: `geqrf`로 분해한 뒤 반사 벡터를 Q의 출력 자리에 복사하고 `orgqr`/`ungqr`가 그 자리에서 Q를 만든다. numpy와 같은 호출이라 비트까지 같다.
- `qr_pivoted`: `geqp3`. `perm`은 scipy와 같다. `x[:, perm] = Q @ R`이고 |R의 대각|은 감소한다.
- **`lstsq`는 `gelsd`다.** 1×1부터 2000×200, 1000×1000, 200×1000까지, 정칙인 경우와 랭크가 절반인 경우를 쟀다.
  - `gelss`보다 거의 모든 경우에 빠르다. n = 1000 정사각에서 6.6–8.3배. 8×8 이하에서만 `gelss`가 2–9% 빠르다.
  - `gelsy`는 정칙인 정사각에서 2–4배 빠르다. 하지만 랭크가 부족하면 최대 2.4배, 가로로 긴 행렬에서 1.7배 느리다. 무엇보다 특이값을 주지 않는다(§2.6의 결과에 필요).
  - 기본 `rtol`에서 세 드라이버의 해는 1e-15 수준으로 같다. `rcond = eps`로 재면 랭크가 부족한 경우에 드라이버마다 랭크를 다르게 판정해 해가 크게 달랐다. 허용오차가 너무 작으면 랭크 판정이 불안정하다.
  - numpy, scipy와 같은 드라이버라서 결과도 numpy와 비트까지 같다. 단 b의 메모리 위치까지 같아야 한다(2026-11-19): Accelerate의 `dgelsd`는 열이 더 많은 경로에서 b가 16바이트 경계에 있는지에 따라 마지막 비트가 달랐다(같은 루틴을 ctypes로 b의 자리만 8바이트 옮겨 부르면 무작위 연립 40개 중 9×11은 35개, 5×7은 34개의 비트가 바뀌고, 3×8과 11×9는 0개다). numpy는 a, b, s를 malloc 하나에 이어 두어 b가 a 뒤 8mn바이트(실수)에 있고, 우리 Scratch는 캐시 줄 경계였다. ctypes로 같은 루틴을 b의 위치만 바꿔 부르면 numpy와 우리 결과가 각각 재현된다. 이제 b를 numpy와 같은 위치(경계에서 8mn mod 16바이트)에 둔다. 모양 1–15×1–15와 큰 모양, 랭크 부족, 우변 여럿, 배치, complex에서 numpy와 모두 비트까지 같다(tests/linalg/test_lstsq_numpy_bits.py). lwork, liwork, rcond(eps·max(m, n)), ldb, NRHS는 처음부터 같았다.
- **`rtol ≥ 1`이면 모든 특이값이 0이다.** 랭크 0, 해 0이다. 정의(§2.6)를 그대로 따른다. 경계값도 0으로 친다(≤).
  - LAPACK의 `xLALSD`는 `rcond ≥ 1`을 아무 말 없이 eps로 바꾼다. 그래서 numpy는 `rcond=1.0`에서 랭크가 최대다. 따르지 않는다.
  - 구현: `gelsd`를 `rcond = 0.5`로 불러 특이값을 얻고, 해는 버리고 랭크와 해를 0으로 둔다.
- **`rtol < eps`이면 SVD로 푼다** (사용자 결정 2026-11-19). 랭크가 `matrix_rank(rtol)`과 비트까지 같아야 하기 때문이다.
  - `xLALSD`는 min(M, N) > 25이면 분할 정복 전에 가장 큰 값의 eps 아래인 이중대각 원소를 ±eps로 바꾼다. 그래서 정확히 0인 특이값도 0이 아니게 되어, `rtol = 0`에서 열 하나가 0인 30×30의 랭크가 30이었다(`matrix_rank`는 29). 대각 (1, 5e-324)는 반대로 1이었다(2). `rcond ≤ 0`을 eps로 바꾸는 것을 피하려고 가장 작은 양수를 넣던 이전 방식은 n ≤ 25에서만 맞았다.
  - 구현: 특이값은 `matrix_rank`, `svdvals`와 같은 `gesvd`의 값 전용 모드(`svals.hpp`)로 구하고, `rtol × 가장 큰 값`보다 큰 것의 수가 랭크다. 해는 `gesdd`의 U, Vᴴ로 V S_r⁻¹ Uᴴ b이고, 큰 쪽부터 랭크 개만 역수를 취한다. 돌려주는 `singular_values`도 `svdvals`의 값이다.
  - `rtol ≥ eps`는 그대로 `gelsd`다. `xLALSD`가 eps로 올리는 값은 원래 eps·σ_max 아래라 `rtol` 기준으로도 0이므로 판정이 바뀌지 않는다. 기본 `rtol`은 max(M, N)·eps라 늘 이 경로다.
- **남긴 특이값이 2⁻¹⁰²² 아래일 때** (2026-11-19): 1/σ가 inf로 넘치고, 특이벡터의 정확한 0과 곱해져 NaN이 됐다. `lstsq(diag(1, 5e-324), [1, 1], rtol=0)`의 해가 [nan, inf](참값 [1, 2.02e323 → inf]), `pinv`가 [[nan, nan], [nan, inf]]였다. 이제 그 항들만 σ·2⁶⁰⁰(2의 거듭제곱이라 정확하다. 역수는 2⁴⁷⁴ 이하)으로 따로 더하고 끝에 2⁶⁰⁰을 곱한다(`svals.hpp`의 `normal_rank`, `add_shifted`). 정확한 0은 0으로 남고, double을 넘는 성분은 참값을 반올림한 ±inf이며, 다른 항의 비트는 그대로다(그런 특이값이 없으면 계산이 전과 같다). `pinv`와 `rtol < eps`의 SVD 풀이가 이 길이다. 복소수와 배치도 같다. `gelsd` 경로(`rtol ≥ eps`)는 손대지 않는다: 거기서 남긴 특이값이 2⁻¹⁰²² 아래이려면 σ_max < 2⁻⁹⁷⁰이어야 하고, `gelsd`는 가장 큰 원소가 smlnum = sfmin/eps = 2⁻⁹⁷⁰(약 1e-292, dgelsd.f; 제곱근이 아니다) 아래인 행렬을 범위 안으로 비례 조정해 풀고 해를 되돌린다. σ_max < 2⁻⁹⁷⁰이면 가장 큰 원소도 그 아래라 언제나 이 조정을 거친다. 그래서 정확한 0은 0, 넘침은 inf이고 numpy와 비트가 같다(작은 원소의 무작위 연립 9000개에서 NaN 없음). 처음에는 `diag(5e-324, 5e-324)`의 `rtol=1e-300`이 `gelsd`에서 NaN을 낸다고 적고 SVD로 다시 풀었는데, `rtol=1e-300`은 eps 아래라 SVD 경로였다(2026-11-19 확인 뒤 다시 풀기를 뺐다).
- **우변 0개(k = 0)**: 특이값과 랭크는 여전히 계산한다. `xLALSD`는 nrhs = 0을 받지 않고 받으면 stderr에 경고를 쓴다. 그래서 0으로 채운 우변 한 열을 넣고 결과는 버린다.

### 10.5 대칭 고유값과 SVD (`eigh`, `eigvalsh`, `svd`, `svdvals`, `matrix_rank`, `pinv`, `cond`)
- **값만 구하는 함수는 LAPACK의 값 전용 모드를 쓴다**: `eigvalsh`, `svdvals`, 그리고 이것을 쓰는 `matrix_rank`, `cond`(±2), `matrix_norm`(2, −2, nuc)
  - 1.6–2배 빠르다. 대신 값이 `eigh`/`svd`가 주는 값과 1e-15 수준에서 다르다. 측정한 모든 드라이버에서 jobz 'N'과 'V'의 비트가 달랐다. numpy도 같다.
  - 특이값만 쓰는 함수는 모두 같은 헬퍼(`svals.hpp`의 `singular_values`)로 `svdvals`와 똑같은 값을 쓴다.
- **`eigh(a, b)`**: `b`가 양정치가 아니면(`sygvd`의 `info > n`) `LinAlgError("matrix is not positive definite")`다. 고유벡터는 `vᴴ b v = I`로 정규화된다. `a`, `b` 모두 아래 삼각만 읽고 검사한다.
- `eigh`는 numpy와 비트까지 같다(같은 `syevd`/`heevd`, 아래 삼각).
- **`svd`는 numpy와 비트가 다를 수 있다.** 특이값은 같지만 U, Vh가 다른 경우가 있다(무작위 300개 중 실수 8개, complex 22개).
  - `gesdd`는 workspace 크기(`lwork`)를 보고 내부 경로를 고른다. 우리는 LAPACK이 알려 주는 최적 크기를 쓴다. numpy와 scipy는 각자 다른 방법으로 크기를 정한다.
  - complex에서는 scipy와 300개 모두 같고 numpy만 다르다.
  - 그래서 workspace를 재사용해도 같은 `lwork`를 쓴다 (§6.3).
- **`matrix_rank`, `pinv`의 허용오차**: `lstsq`와 같은 `rtol` 정의다. `rtol ≥ 1`이면 랭크 0, `pinv`는 영행렬이다.
  - `rtol < eps`이면 `lstsq`의 랭크도 `matrix_rank`의 `gesvd` 값에서 나와 언제나 같다(§10.4). 그 밖에서는 각자의 SVD(`gesvd`와 `gelsd` 내부)에서 나와, 경계에 아주 가까운 특이값이 있으면 이론상 다를 수 있다. 무작위 입력의 테스트에서는 늘 같았다.
- **`cond`는 특이 행렬에서 inf다.** 모든 ord에서 그렇고, `ord=-2`는 그 역수인 0이다.
  - "특이"는 각 경로의 정확한 0이다. ±2는 가장 작은 특이값이 정확히 0일 때만, 그 밖의 ord는 `getrf`의 정확한 0 피벗일 때다. 그래서 반올림 때문에 σ_min이 0이 아닌 행렬은 큰 유한값을 낸다. 예: `[[1, 2], [2, 4]]`는 ord 2에서 2.76e16, ord 1에서 inf
  - ±2는 σ로 계산하므로 정사각이 아니어도 된다. 나머지 ord는 `norm(x)·norm(inv(x))`라서 정사각이 필요하다(ValueError). 역행렬은 `inv`와 같은 경로(`lu_factor` + `lu_inverse`)다.
- **너무 큰 정수 `ord`**(`10**400`)는 `vector_norm`처럼 OverflowError다 (`cond`, `matrix_norm`).

### 10.6 일반 고유값, Schur, Hessenberg, LDL
- **`eig`의 결과**
  - 항상 complex128이다.
  - (실수부, 허수부) 사전순으로 안정 정렬하고, 고유벡터 열도 같은 순서로 옮긴다 (§2.6). 실수 입력의 켤레쌍은 실수부가 정확히 같으므로 허수부가 음수인 쪽이 먼저 온다.
  - 고유벡터는 LAPACK 그대로다: 2-노름 1, 가장 큰 성분이 실수다.
  - 실수 `geev`는 켤레쌍의 벡터를 실수 열 두 개(`vr[:,j]`, `vr[:,j+1]`)로 준다. 이것을 complex 열 두 개 `vr[:,j] ± i·vr[:,j+1]`로 조립한다.
- `schur`: `gees`, 정렬 없음. `hessenberg`: `gehrd` + `orghr`/`unghr`
  - **scipy와 다른 곳**: n ≤ 2인 complex 행렬의 `hessenberg`. scipy는 n ≤ 2면 LAPACK을 부르지 않고 H = x, Q = I를 돌려준다. `zgehrd`는 길이 1짜리 반사로 부대각 원소를 실수로 만든다. 둘 다 `Q H Qᴴ = x`를 만족하는 올바른 Hessenberg 형식이다.
- **`ldl`은 `sytrf`/`hetrf`(Bunch–Kaufman)다.** `sytrf_rook`과 비교했다.
  - 속도: `sytrf`가 10–66% 빠르다.
  - 역오차: 둘 다 `0.003–0.19·n·eps`로 차이가 없다. rook 피벗은 L의 원소 크기를 제한하지만, 무작위 행렬에서는 정확도 차이가 드러나지 않았다. scipy도 `sytrf`를 쓴다.
  - 특이 행렬에서도 에러를 내지 않는다. D에 0 블록이 생길 뿐이다. 아래 삼각만 읽고 검사한다. complex는 에르미트로 본다.
  - **`perm`, `L`, `D`는 scipy와 같다.** `ipiv`(2×2 블록 포함)를 scipy.linalg.ldl(`_ldl_sanitize_ipiv`, `_ldl_construct_tri_factor`)과 같은 방식으로 읽는다. 우리 `L`은 scipy의 `lu[perm]`이다. 그래서 `x[perm][:, perm] = L D Lᴴ`이고 L은 진짜 아래 삼각이다. 테스트에서 scipy와 비트까지 같음을 확인한다.

### 10.7 특수 구조 (`solve_tridiagonal`, `solve_banded`, `solve_sylvester`)
- **`solve_tridiagonal`은 `gtsv`다.** `gttrf` + `gttrs`보다 1.2–3배 빠르고 결과 비트가 같다. 역오차는 둘 다 `0.15·n·eps` 이하다.
- **`solve_banded`는 `gbsv`다.** `gbtrf` + `gbtrs`와 속도도 비트도 같다. 호출이 하나라 단순한 쪽을 골랐다.
  - **n ≤ 8은 밀집 행렬의 LU다** (§7.4). 호출당 고정비 때문이다.
  - **대역폭 (1, 1)은 `gtsv`로 푼다.** 삼중대각에서 `gtsv`가 `gbsv`보다 4–5배 빠르고(n = 1000: 8µs, 38µs), 두 결과는 반올림 수준에서 다르다. 그래서 `solve_banded(ab, b, lower_bandwidth=1, upper_bandwidth=1)`은 `solve_tridiagonal`과 비트까지 같고, scipy와도 같다(scipy도 이 경우 `gtsv`를 부른다).
  - 기각한 대안: 항상 `gbsv`. 규칙은 단순하지만 (1, 1)에서 4–5배 느리다.
- **`solve_sylvester`는 Bartels–Stewart다.** `a = U R Uᴴ`, `b = W S Wᴴ`(`gees`), `F = Uᴴ c W`(gemm 2번), `trsyl3`로 `R Y + Y S = F`, `x = U Y Wᴴ`(gemm 2번)
  - `trsyl3`(LAPACK 3.11의 블록 버전)는 `trsyl`보다 n = 300에서 1.5배, 1000에서 4.8배 빠르다. n ≤ 32에서는 결과 비트까지 같고, 잔차는 둘 다 eps 수준이다.
  - scipy는 `b`의 켤레 전치를 Schur 분해하고 `trsyl(tranb='C')`를 부른다. 우리는 `b`를 바로 분해하고 'N', 'N'으로 부른다. 수학적으로 같은 방정식이다.
  - **`a`와 `−b`가 공통(또는 eps 수준으로 가까운) 고유값을 가지면 `LinAlgError`다.** `trsyl3`는 이때 값을 살짝 바꾼 방정식을 풀고 `info = 1`을 알린다. scipy는 이것을 무시하고 그 해를 돌려준다. 요청한 방정식의 해가 아니고 경고를 쓰지 않으므로, 알릴 방법은 에러뿐이다. 문턱이 `eps·max(‖R‖, ‖S‖)` 수준이라 사실상 수치적으로 특이한 경우에만 걸린다.
  - **`scale`**: `trsyl3`는 `R Y + Y S = scale·F`를 푼다. 해는 `Y / scale`이다. (scipy 1.18.1은 `scale`을 곱한다. `scale < 1`은 넘치기 직전에만 나오므로 보통은 차이가 없다.)
  - **실수 `a`와 complex `b`이면 둘 다 complex Schur로 계산한다.** scipy 1.18.1은 이 경우 `a`를 실수 Schur(준삼각)로 분해한 채 complex `trsyl`에 넘긴다. complex `trsyl`은 위 삼각만 보므로 2×2 블록의 부대각을 무시한다. `a`가 켤레쌍 고유값을 가지면 틀린 답이 나온다(3×3 예에서 잔차 1.26). 테스트에 이 경우를 넣었다.
  - 실수 `a`, `b`이면 실수 준삼각 Schur 형식을 `trsyl3`가 그대로 다룬다.
- **읽는 원소만 검사한다** (§2.3): `solve_banded`는 띠 저장 형식 `ab`에서 행렬 안에 해당하는 원소만 읽고 검사한다. 모서리(행렬 밖)는 NaN이어도 된다. (1, 1)에서도 같다. 대역폭이 행렬보다 넓어도 되고, 남는 행은 읽지 않는다.
- **모양**
  - `solve_tridiagonal`: `dl`, `du`의 길이는 `d`보다 하나 짧다. n = 0이면 길이 0이다. 네 인자의 배치 축을 모두 브로드캐스트한다.
  - `solve_banded`: `ab.shape[-2]`가 `lower_bandwidth + upper_bandwidth + 1`이 아니면 ValueError. 두 대역폭은 keyword-only 필수 int이고 음수면 ValueError다.
  - `b`가 맞지 않을 때의 메시지는 `"the matrix is NxN but 'b' has R rows"`다. 구조화된 인자 자체는 N×N 모양이 아니기 때문이다.
  - `solve_sylvester`의 `c`는 항상 행렬 `(..., M, N)`이다. 1차원 `b` 규칙(§2.1)은 solve류의 우변에만 적용된다.

### 10.8 LAPACK을 쓰지 않는 함수
`vecdot`, `outer`, `trace`, `diagonal`, `matrix_power`(n ≥ 0), `kron`, `cross`, `vector_norm`, `matrix_norm`(fro, ±1, ±inf). 직접 루프로 짠다. NaN을 전파한다 (§2.3).
- **원소 곱은 배열(그리고 numpy)과 같은 식이다** (`types.hpp`의 `mul`).
  - complex: `(ar·br − ai·bi, ar·bi + ai·br)`를 `fma(ar, br, −(ai·bi))`, `fma(ar, bi, ai·br)`로 계산한다. 배열의 complex 곱과 같다 (array.md §5.3). numpy는 FMA 축약을 켜고 빌드되어 있어서 `std::complex`의 곱셈과 비트가 다르다.
  - 그래서 `outer(x, y)`는 `x[..., :, None] * y[..., None, :]`와, `kron`은 `np.kron`(2차원)과, `cross`는 `np.cross`와 비트까지 같다.
- **`vecdot`, `cross`의 축**: `axis`는 브로드캐스트된 shape 기준이다. 축약하는 축 자체는 브로드캐스트하지 않는다. 두 입력의 그 축 크기가 같아야 하며, 다르면 ValueError다 (Array API). `cross`는 길이 3만 받는다.
  - `cross`는 각 곱을 반올림한 뒤 빼는 numpy의 순서 그대로다.
- **`kron`은 행렬 배치다**: `(..., M, N) ⊗ (..., P, Q) → (..., M·P, N·Q)`. 배치 축은 다른 행렬 함수처럼 브로드캐스트한다. 입력은 2차원 이상이어야 한다. 원소 곱은 x1이 왼쪽이다.
- **`diagonal`**은 입력 dtype을 유지하는 복사본이다. 계산하지 않기 때문이다.
- **`matrix_power`**: 이진 거듭제곱(최상위 비트부터)이다. `log2(e)`번 제곱과 `popcount(e) − 1`번 곱, 모두 `gemm`이다. n < 0은 `inv`와 같은 경로로 역행렬을 구한 뒤 거듭제곱한다(§2.3).
  - `|n|`은 `INT64_MIN`에서도 넘치지 않게 부호 없는 정수로 구한다.
- **p-노름의 넘침/언더플로 회피**: 기본은 스케일 없는 한 번의 패스다. 아래 경우에만 최댓값으로 나눈 두 번째 패스를 돈다.
  - 합이 inf로 넘쳤다.
  - 최댓값의 p제곱이 2⁻⁹⁶⁹(= DBL_MIN·2⁵³) 미만이다. 이때는 모든 항이 subnormal 근처라 비트를 잃는다.
  - numpy는 이 처리를 하지 않는다: `vector_norm([1e200, 1e200])`이 inf다(경고 포함).
- **노름의 특수값**
  - 빈 집합의 노름은 모든 `ord`에서 0이다 (§2.5).
  - `ord=0`은 0이 아닌 원소의 개수이고, NaN도 센다.
  - NaN이 있으면 나머지 `ord`의 결과는 NaN이다. NaN 없이 inf가 있으면 p > 0에서 inf다.
- 노름은 원소를 논리적 C 순서로 걷는다(`vnorm.hpp`의 walk, 힙 할당 없음). 최대 33축(배열의 32축 + complex 분할)이다.

### 10.9 남겨 둔 것
결과에는 영향이 없고, 이유가 있어 그대로 둔 것들이다.
- **`ord` 파서 두 벌**(`cond`의 `parse_cond_ord`, `matrix_norm`의 `parse_matrix_ord`): 받는 값의 집합이 같지만 따로 있다. 에러 종류는 M6에서 맞췄다: 타입이 틀리면(예: `ord=True`) 세 노름 모두 TypeError, 값이 틀리면 ValueError다 (DESIGN §9.6). 합치는 일은 결과를 바꾸지 않으므로 남겨 두었다.
- **`check_rhs_rows`**(`solve_tridiagonal`, `solve_banded`): `rhs_of`도 행 수를 보지만, 구조화된 인자는 그 자체가 n×n이 아니라서 "the matrix is n x n"이라고 따로 말해야 한다. 먼저 돌아서 `rhs_of`의 검사에는 닿지 않는다. 중복이 아니라 다른 메시지다.
- **`cond`(±2 외의 ord)의 두 번째 패킹**: `lu_factor`가 패킹한 사본을 덮어쓰므로 x의 노름을 위해 한 번 더 패킹한다. 없애려면 이미 패킹한 행렬을 받는 `lu_factor` 변형이 필요하다. 드문 경로 하나라서 두었다.
- 함께 고친 것은 §10.10에 있다.

### 10.10 고친 것 (결과는 같다)
- `trace`/`diagonal`에서 행렬 밖을 가리키는 오프셋(`k >= n` 또는 `k <= -m`)은 빈 대각을 (0, 0)에서 시작한다. 전에는 k = −2⁶³에서 `-k`가, 아주 큰 |k|에서 `r0 * stride`가 부호 있는 오버플로(정의되지 않은 동작)였다. 테스트: `test_offset_past_the_matrix_is_an_empty_diagonal`.
- `ldl_unpack`의 `perm` 임시 버퍼는 함수 몫의 scratch 슬롯 3을 쓴다. 전에는 LAPACK workspace 몫인 슬롯 4를 빌렸다 (§6.3).
- `conj_if_complex`(네 벌)와 `parse_mode`(두 벌)는 `linalg/ops/common.hpp` 한 곳에 있다.

---

## 11. expm (`linalg/ops/expm.cpp`)
LAPACK에 없으므로 직접 구현한다. 알고리즘은 Al-Mohy & Higham 2009 (SIAM J. Matrix Anal. Appl. 31(3), Algorithm 5.1)이고, scipy.linalg.expm과 같다.

### 11.1 알고리즘
- `exp(A) = (exp(2⁻ˢA))^(2ˢ)`, `exp(B) ≈ r_m(B) = (V − U)⁻¹(V + U)`. r_m은 [m/m] Padé 근사, U는 B의 홀수 다항식, V는 짝수 다항식이다.
- **차수 m ∈ {3, 5, 7, 9, 13}과 제곱 횟수 s**를 역방향 오차가 단위 반올림 아래인 가장 싼 쌍으로 고른다.
  - 판단에 ‖A‖가 아니라 ‖A^k‖^(1/k)를 쓴다. 비정규 행렬에서는 이것이 ‖A‖보다 훨씬 작을 수 있다. 스케일을 덜 하고, 제곱을 덜 하고, 오차가 줄어든다.
  - 보정 ℓ(A, m): ‖|A|^(2m+1)‖₁로 제곱을 더한다. |A|는 음이 아니므로 이 노름은 eᵀ|A|^p의 최댓값이고, `gemv` p번으로 정확히 구한다. 넘치지 않게 |A|를 2⁻ᵉ로 나눠서 log2로 계산한다.
  - θ₁₃ = 4.25 (scipy와 같은 값). 계수와 θ는 논문 Table 3.1이다.
- **결과는 `I + 2(V − U)⁻¹U`로 만든다.** `(V−U)⁻¹(V+U)`와 수학적으로 같지만, I를 작은 보정에 정확히 더하므로 노름이 작을 때 대각의 마지막 비트를 잃지 않는다. 처음 형태로는 ‖A‖ ~ 1e-3에서 1ulp(2.2e-16)를 잃었다. scipy도 이 형태다.
  - 풀기는 `getrf` + `getrs`다. `V − U`가 특이한 경우는 이 차수들에서 이론상 생기지 않지만 검사는 둔다(`LinAlgError`).
- 행렬곱은 `gemm`이고, 거듭제곱 A², A⁴, A⁶(차수 13이면 A⁸, A¹⁰의 노름까지)을 만든다.

### 11.2 노름: 정확히, 상한 먼저, 큰 n은 추정
- **노름은 정확한 값을 쓴다.** 추정기로 시작하지 않았다.
- **거듭제곱은 필요할 때만 만든다.** 이미 아는 노름의 곱으로 만든 상한(‖A^(i+j)‖ ≤ ‖A^i‖·‖A^j‖)으로 시험이 통과하면 정확한 값도 통과한다. 그래서 고르는 차수가 같다(임계값 바로 위의 반올림만 예외). 예: 차수 3이면 A²만 만든다.
  - 상한은 올바른 근이어야 한다. ‖A⁴‖로 ‖A⁸‖을 묶으면 `‖A⁸‖^(1/8) ≤ ‖A⁴‖^(1/4)`다 (`d8`의 `root(n4, 4)`). 무작위 2–4차 행렬 20만 개로 상한과 정확한 값의 판정을 대조했다.
- **n ≥ 100(`kExpmEstimateMin`)이면 ‖A⁸‖, ‖A¹⁰‖은 LAPACK의 1-노름 추정기 `lacn2`(Hager–Higham, 무작위 없음)로 추정한다.** A⁴, A⁶에 `gemv`를 두 번씩 해서 곱을 만들지 않는다.
  - 이유: 차수 13이 필요한 큰 행렬(n ≥ 500) 하나에서 A⁸, A¹⁰을 만드는 행렬곱 2번이 호출의 12–14%였다. 그 줄만 scipy보다 6–9% 느렸다.
  - 기준 n = 100: n ≥ 100에서 1.03–1.15배 빠르고, n ≤ 64에서는 같거나 느리다. 잰 60개 행렬 모두에서 추정이 결과를 바꾸지 않았다(같은 s).
  - scipy는 n ≥ 400부터 무작위 시작 벡터의 추정기를 쓴다.
- **차수 선택용 노름은 싸게 계산한다.** 몇 ulp 차이는 고르는 차수에 상관없다.
  - complex 원소의 절댓값은 `hypot` 대신 `max·sqrt(1 + (min/max)²)`다(분기 없음, 벡터화).
  - 열 합은 누산기 8개로 나눠 더한다. `ldexp` 대신 2의 거듭제곱을 곱한다.
  - 이 세 가지로 n = 32–100에서 0.8배이던 속도가 scipy 이상이 되었다. 노름 계산(`hypot`, 직렬 덧셈)이 행렬곱보다 비쌌다.

### 11.3 대각·삼각 입력
논문과 scipy대로다. 판별은 알고리즘의 일부이고 결과의 의미는 그대로다 (§1.3).
- **대각**이면 `exp(대각)`, 나머지는 정확히 0이다. 1×1도 여기로 간다.
- **삼각**이면 Padé 결과와 매 제곱 뒤에 대각을 `exp(2⁻ᵏ a_ii)`로, 첫 부대각을 2×2 블록의 정확한 식으로 다시 쓴다(논문 Code Fragment 2.1). 반대쪽 삼각은 정확히 0으로 둔다. s = 0에서도 한다(scipy는 s = 0이면 하지 않는다).
- **2×2 블록의 식**
  - λ가 가까우면(|λ₂ − λ₁| < 1) 논문의 식 (2.2) `t·exp(평균)·sinch(차/2)`
  - 멀면 차분몫 `t·(e^λ₂ − e^λ₁)/(λ₂ − λ₁)`. 멀 때 (2.2)만 쓰면 λ₁ = −2000, λ₂ = 0에서 `exp(−1000)·sinh(1000)`이 0·inf = NaN이 된다. 멀면 차분몫에 상쇄가 없다.
  - scipy는 항상 차분몫이라 가까운 λ에서 상쇄가 생긴다. `[[10, 1], [0, 10+1e-12]]`의 (0, 1) 원소의 상대오차가 scipy 2e-5, sd 2e-15다.

### 11.4 극단값 가드
- **NaN/inf 입력은 ValueError**다(대각 행렬도). 마지막 풀기가 LAPACK이기 때문이다 (§2.3). 결과가 넘치면 inf이고 경고는 없다.
- **‖A‖₁ > 2¹⁰⁰이면 먼저 2^(−s₀)로 줄인다**(s₀번 제곱이 더해진다). 그래야 A²–A¹⁰과 노름이 넘치지 않는다. 고유값이 −1e40 근처인 2×2에서 정답은 거의 0인데, scipy는 단위행렬을 낸다(sd는 0).
- **노름이 비정규 수**(‖A‖₁ < 2⁻¹⁰²⁴, 예: `[[1e-310]*2]*2`): ℓ용 `2⁻ᵉ`가 inf로 넘치지 않게, e < −1022이면 2¹⁰⁰⁰과 2^(−e−1000)을 차례로 곱한다. 둘 다 정확한 곱이고 결과는 1 이하다. 가드가 없으면 ℓ이 `static_cast<int>(inf)`(약 2³¹)를 돌려줘 제곱이 끝나지 않는다. 결과는 I + A이고 scipy와 비정규 수 ulp 하나 차이다.
- **원소는 유한한데 노름이 inf**(예: `[[1e308]*2]*2`): `frexp(inf)`가 지수를 주지 않아 2¹⁰⁰ 가드를 건너뛴다. 먼저 2⁻ᵏ(2ᵏ ≥ 4n)를 곱해 열 합을 유한하게 만든다(복소수 크기는 √2·DBL_MAX 이하). 그다음 2¹⁰⁰ 가드가 동작한다.
  - 값은 inf/NaN이다. 이런 행렬은 scaling and squaring의 한계로 훨씬 앞에서 정확도를 잃는다.
    - `[[-c, c], [c, -c]]`: c = 1e12에서 상대오차 7.6e-6(scipy 2.3e-5), 1e18에서 X[0,0] = 4.4e6(정답 0.5), 1e20 이상은 inf, 0, 0.5 등 반올림 운에 달린 값
    - 멱영 `[[c, ic], [ic, -c]]`: c = 1e8에서 이미 수십 자릿수 틀리고, 1e12부터 NaN
    - scipy는 이 범위(멱영은 1e12, `[[-c, c], [c, -c]]`는 1e40 근처)부터 끝나지 않는다.

### 11.5 정확도와 속도
- **정확도**: 80자리 `decimal`로 계산한 기준값과 37개 경우를 비교했다. 대부분 scipy와 같거나 더 정확하다. 무작위 행렬(노름 10–100)에서 scipy 1.6e-15–2.3e-12, sd 4.8e-16–1.3e-14다.
  - scipy보다 2배 넘게 나쁜 2개는 반올림 운이다. Moler–Van Loan 행렬은 같은 알고리즘이 s에 따라 2e-15–3e-14로 흔들리고, 삼각 n = 6은 전치하면 결과가 뒤집힌다.
- **비트**: scipy와 같은 알고리즘이지만 BLAS 호출 순서와 노름 계산이 달라서 비트는 다르다.
- **속도**: 행렬 하나 1.0–1.4배, 배치 3.6–8배(scipy는 배치를 하나씩 돈다). 32², 1000²의 행렬 하나는 1.01–1.07배로 같다.
- 배치는 `kExpm`(600, 15)으로 나눈다 (§7.1).

---

## 12. numpy/scipy와 비교하는 방법

### 12.1 비교 기준
- **numpy 2.5.4, scipy 1.18.1.** 둘 다 BLAS와 LAPACK이 Accelerate다(`show_config`로 확인). 같은 LAPACK을 쓰므로, 차이는 감싸는 비용(복사, 레이아웃, 배치, Python 오버헤드)에서 나온다.
  - 기준 시간은 numpy, 그 함수가 numpy에 없으면 scipy다.
- **정확도는 잔차로 본다.** ‖Ax − b‖, ‖QR − A‖, ‖QᴴQ − I‖, ‖AV − VΛ‖ 등
  - 이유: 고유벡터와 특이벡터는 드라이버마다 부호와 위상이 달라도 둘 다 맞는 답이다. numpy와 값을 직접 비교할 수 없다.
- **답이 하나로 정해지는 값**(고유값, 특이값, `det`, `solve`의 해)은 numpy, scipy와 직접 비교한다.

### 12.2 비트 동일성
- **numpy·scipy와의 비트 일치는 약속이 아니다** (사용자 결정 2026-11-20, API.md의 비트 규칙). 아래 표는 지금의 사실이다.
  - 약속은 셋이다: 스레드 수·레이아웃·배치 위치와 무관한 비트, 우리 함수끼리의 비트 관계(아래), 우리 골든 출력과 정확도 계약.
  - 지금 비트가 같은 코드를 그것만을 위해 바꾸지 않는다. 고침이나 최적화에 필요하면 깨도 된다. 그때 numpy·scipy와 비트로 비교하던 테스트는 허용오차 비교로 바꾼다.
- 테스트가 이 구분을 지금 상태대로 고정한다 (§13).

| 범주 | 관계 |
|---|---|
| `eigh`, `qr`, `lstsq` | numpy와 비트까지 같다 (같은 드라이버, 같은 호출) |
| `outer`, `kron`, `cross` | numpy와 같다 (같은 곱셈 식, §10.8) |
| 그냥 합 (`trace`, 1-노름, p-노름) | 배열의 합과 같다 (같은 트리와 잎, §8.2). numpy와는 배열의 잎이 numpy의 것일 때만 같다 |
| 곱의 합 (`vecdot`, 2-노름, 'fro') | 곱을 잎에 융합한다(FMA). `sum(conj(x)*y)`, numpy와 끝 비트가 다를 수 있다 (§8.2) |
| `matrix_norm`의 ±1, ±inf, 'fro' | 줄마다의 pairwise 합이다. numpy와는 더하는 축이 연속일 때만 같다 (§8.2) |
| `cholesky`, `cholesky_solve` | numpy·scipy와 비트까지 같다 (같은 `potrf`/`potrs`) |
| `qr_pivoted` | perm, Q, R 모두 scipy와 비트까지 같다 (같은 `geqp3`) |
| `ldl`, `solve_tridiagonal`, `solve_banded`(n > 8) | scipy와 같다 (같은 피벗 해석, 같은 `gtsv`/`gbsv`) |
| `solve_banded`(n ≤ 8, 대역폭 (1, 1) 외) | scipy와 반올림 수준에서 다르다 (밀집 LU, §7.4) |
| LU 계열의 n = 32–46 | `getf2`라서 numpy·scipy(`getrf`)와 반올림 수준에서 다를 수 있다 (§7.5) |
| `solve` | numpy와 반올림 수준에서 다르다 (`gesv` 대 `getrf + getrs`, §10.2) |
| `svd` | 다를 수 있다 (`gesdd`의 `lwork`, §10.5) |
| `eigvalsh`, `svdvals` | `eigh`/`svd`의 값과 1e-15 수준에서 다르다 (값 전용 모드, numpy도 같음) |
| `expm` | scipy와 다르다. 정확도는 같거나 낫다 (§11.5) |

- **우리 함수끼리 같은 것**: `lu_solve(*lu(a), b) == solve(a, b)`, `inv(a) == solve(a, I)`, `matrix_power(x, -1) == inv(x)`, `cholesky(x, upper=True) == cholesky(x)ᴴ`, `solve_banded(ab, b, 1, 1) == solve_tridiagonal(...)`
- **불변성**: 결과는 입력 레이아웃(§5.7), 배치 안의 위치(§6.4), 스레드 수(§8.5)와 무관하게 비트까지 같다.

### 12.3 측정 방법
- **기준선은 고정된 줄 묶음이다.** 함수마다 정해진 shape 세트(행렬 하나 2×2–2048², 배치, 레이아웃, 긴 벡터)를 잰다. 줄 단위로 numpy/scipy와 비교해 빠름/같음(±10%)/느림을 센다. 커밋 4151d2a의 `bench/merge/`에서는 linalg 258줄이었다. 지금은 `bench/speed/cases/linalg.py`의 사례이고, 표는 `bench/linalg/speed.md`다(`.venv/bin/python -m bench.speed run --area linalg`).
- **두 번 이상 돌려서 확인한다.** `eig` 배치가 한 번 0.35배로 나왔고 재현되지 않았다(다시 재면 3.5ms 대 그 실행의 21.9ms). 그 실행 전체가 느렸다.
- **버전끼리는 같은 자리에서 번갈아 잰다.** 각 시점의 커밋을 다시 빌드하고, 같은 줄을 빌드마다 번갈아 두 바퀴 재어 가장 빠른 값을 쓴다.
  - 이유: 같은 빌드가 날에 따라 최대 2배 다르게 나왔다. 같은 옛 빌드가 날에 따라 "빠름 109줄"과 "122줄"로 갈렸다.
  - 의심되는 줄은 이전 빌드와 A/B로 확인한다. 두 빌드가 함께 튀면 기계 상태다.
- **벤치마크 중에는 다른 작업을 돌리지 않는다.**
- **같은 방식으로 설치해서 잰다.** editable 설치(scikit-build-core의 import 훅)로 부른 빌드에서 `solve_tridiagonal` (1000,3,3) 10스레드가 프로세스마다 약 8µs 또는 18µs로 갈렸다(16번 중 10번 느림). `pip install --target` + `python -S`로는 12번 중 0번이었다. 원인은 확인하지 못했다(메모리 페이지나 import 훅 쪽의 상태로 보인다).
- µs 단위의 병렬 호출은 실행 사이에 ±10–30% 흔들린다. 큰 시간은 안정적이다.
- 통합 중에는 `bench/merge/baseline.md`(커밋 4151d2a)의 기준선과 비교한다. 10% 넘게 느려진 줄은 원인을 밝힌다 (ROADMAP 진행 원칙).

### 12.4 위치
기준 시간 / sd 시간. 258줄 중 빠름 195, 같음 43, 느림 10이다 (10스레드).
- **10스레드(기본)**
  - 작은 행렬 하나: 기하평균 2.8배. 3×3 `solve` 0.48µs 대 numpy 1.99µs
  - 배치: 7.3배. numpy와 scipy는 배치를 코어 하나로 돈다. scipy가 Python 수준에서 배치를 하나씩 도는 함수(`schur`, `hessenberg`, `ldl` 등)는 3.9–96배, `expm` 배치는 3.6–8배다.
  - 출력 하나의 긴 합: 1.9배 (`vecdot` 1e6 29.6µs 대 37.2µs, `vector_norm` 1e6 62.8µs 대 251µs)
  - 큰 행렬 하나의 분해(`eig`, `svd`, `schur`, `lstsq`, …): 0.94–1.28배, 대부분 1.0–1.2배다. 같은 Accelerate다. 예외: `solve_sylvester` 1000²는 `trsyl3` 덕분에 1.8배, `ldl` 1.4–1.9배, 큰 `cholesky` 1.4배
- **1스레드**
  - 작은 행렬은 10스레드와 같다(2.79). 원래 스레드를 쓰지 않던 곳이다.
  - 배치는 1.45배다. numpy와 같은 LAPACK을 부르면서 호출 비용만큼 빠르다.
  - **긴 합과 노름은 numpy보다 느리다** (0.2–0.7배). 잎 버퍼를 거치는 두 단계 때문이다 (§8.2).

    - 아래 표는 곱의 합을 잎 버퍼로 더하던 때의 측정이다. 융합 잎(§8.2) 뒤의 값은 다시 재야 한다.

    | 1스레드, 1e6 | sd.linalg | 같은 트리 (배열, NEON) | numpy |
    |---|---:|---:|---:|
    | `vecdot` | 198µs | `sd.sum` 112µs (곱은 미리) | `np.dot` 74µs (BLAS) |
    | `vector_norm` | 382µs | `sd.sum(x*x)` 230µs | 233µs |
    | 행 노름 1000² | 407µs | `sd.sum(M*M, axis=-1)` 233µs | 238µs |

- **느린 10줄** (10스레드)

  | 줄 | 비율 | 이유 |
  |---|---:|---|
  | `solve_triangular` 300², 1000² (C-order) | 0.87, 0.60 | O(n²) 계산에 O(n²) 전치 패킹이 그대로 보인다 (§5.7) |
  | `lu_solve` 300², 1000² | 0.82, 0.72 | 인수는 F-order라 그 자리에서 읽고 전치도 없다. 원인은 분석하지 않았다 |
  | `matrix_power(x, 3)` 300², 1000² | 0.76, 0.89 | 입력을 패킹한다. numpy는 입력을 그대로 gemm에 넘긴다 |
  | `vecdot` 행 (3,3), (32,32) | 0.81, 0.69 | 0.1µs 차이의 고정비 (인자 해석, 잎 버퍼) |
  | `diagonal` (1000,3,3) | 0.76 | numpy는 뷰에 `.copy()` 한 번, 우리는 배치 드라이버를 거친다 |
  | `matrix_norm` fro (1000,3,3) | 0.77 | 짧은 합 1000개의 고정비 |

### 12.5 하지 않은 최적화
- **결과 비트를 바꾸므로 하지 않은 것** (설계 결정이 필요하다). numpy·scipy와의 비트는 이유가 되지 않는다(§12.2). 남은 이유는 크기에 따른 경로(§1.3), 레이아웃 불변성(§5.7), 규칙의 복잡도다.

  | 후보 | 이득 | 바뀌는 것 |
  |---|---|---|
  | 우변을 4열씩 나눠 풀기 (§7.5) | n = 24–48, k = 8에서 1스레드 1.2–2배, 10스레드 4.6–17배 | 비트. k가 n만큼 많으면 1스레드에서 0.4–0.8배로 느리고, n, k ≥ 64이면 10스레드에서도 블록 경로가 3–7배 낫다. n, k, 스레드 수에 따른 규칙이 필요하다 |
  | C-order 입력의 삼각 풀기를 `trans='T'`로 그 자리에서 | 전치 패킹 제거 | 레이아웃 불변성 (§5.7) |
  | 작은 띠 행렬을 직접 구현 | `gbsv`의 호출당 1µs 고정비 제거 | "LAPACK을 감싼다"(§1.2)의 예외가 된다 |

- **결과를 바꾸지 않지만 아직 하지 않은 것**
  - `matrix_power`의 F-order 입력을 패킹 없이 `gemm`에: 300² 185 → 약 170µs, 1000² 4409 → 약 4110µs(numpy 4108µs)
  - `lu`의 L/U 풀기: n = 300에서 `getrf` 302µs 뒤의 풀기가 51µs다(scipy 28µs). 약 8%다.
- **할 것이 없는 곳**: 큰 행렬 하나의 LAPACK 분해(같은 Accelerate, 0.94–1.28배). 바꾸려면 분해를 직접 짜야 한다(§1.2의 원칙 밖).

---

## 13. 테스트 (`tests/linalg/`)
- **pytest + hypothesis.** C++ 단위 테스트는 없다. 무작위 shape, 배치 shape, stride, dtype, 레이아웃(C, F, 비연속, 음수 stride, 브로드캐스트 배치)을 만든다 (`conftest.py`의 `with_layout`, `batch_shapes`, `random_values`).
  - 잘 조건화된 행렬은 대각 우세로 만든다(`well_conditioned`). int64(원소 [−3, 3])에서도 엄격한 대각 우세다: 대각 ≥ 3n − 2 > 3(n − 1) ≥ 비대각 행 합
- **잔차 검사**: `‖A x − b‖∞ ≤ c·n·eps·(‖A‖∞‖x‖∞ + ‖b‖∞)`, **c = 10** (`RESIDUAL_C`)
  - 일반 무작위 행렬(n = 1–256)에서 잰 실제 역오차는 최대 `0.39·n·eps`다. 약 25배의 여유가 있다.
  - 분해는 재구성 잔차(`‖x[perm] − L U‖` 등), 직교성(`‖QᴴQ − I‖`), 고유값/특이값의 numpy 비교로 본다.
- **규칙 테스트**: 빈 행렬, 배치 크기 0, NaN 입력의 ValueError, 특이 행렬의 `LinAlgError`와 그 배치 인덱스, 에러 순서, 결과 dtype, F-order 결과, 배열이 아닌 인자의 TypeError
- **비트 동일성 테스트** (§12.2)
  - 지금 numpy/scipy와 비트가 같은 함수(약속이 아니다, §12.2), 우리 함수끼리의 비트 관계
  - 레이아웃: 같은 행렬을 C, F, strided로 넣어 같은 바이트
  - 배치 안의 위치: 홀수 n(3, 5, 7, 40)의 배치 결과가 하나씩 부른 결과와 같은 바이트 (§6.4)
  - 스레드 수: 36개 호출 × 2 dtype × 배치 3가지를 스레드 1, 3, 10에서 비교, 실패 메시지 비교, Python 스레드 4개 동시 호출, 반복 20회
  - 최적화를 끄고 켜기: §7.7의 비공개 함수로 끈 경로와 켠 경로, 스레드 1/3/10의 비트를 비교한다(328개).
- **크기로 고르는 경로의 경계** (`test_review_fixes_2.py`): LU 계열의 n = 31, 32, 46, 47과 직사각 `lu`, `solve_banded`의 n = 7, 8, 9, 12에서 답(잔차, numpy와의 허용오차), 레이아웃·배치 위치·스레드 수와 무관한 비트, 모서리를 읽지 않음, 특이의 판정을 본다 (§1.3).
- **끝나지 않던 입력** (`test_review_fixes.py`): `expm`의 극단값(§11.4)은 자식 프로세스에서 시간 제한을 두고 돈다. 가드가 없으면 약 2³¹번 제곱해서 테스트 전체가 멈춘다.
- **API 표면** (`test_api_surface.py`): `docs/api/linalg.md`의 코드 블록에 있는 이름과 모듈이 내보내는 이름이 같은지, keyword-only 옵션이 문서와 같은지 확인한다. nanobind 함수는 `inspect.signature`가 `(*args, **kwargs)`라서 `__nb_signature__`를 읽는다.
- **골든 출력**: 통합 중에는 옛 패키지로 만든 linalg 1,844개 결과(에러 메시지 포함)를 스레드 수 1, 3, 10에서 바이트로 비교한다 (커밋 4151d2a의 `bench/merge/golden/`, ROADMAP 진행 원칙).
- **남은 위험 하나**: 전체 테스트 한 번에서 1건이 실패했는데 출력을 남기지 않아 어느 테스트인지 모른다. 이후 전체 테스트 160회와 hypothesis 2000예제 실행에서 다시 나오지 않았다. hypothesis는 실행마다 다른 예제를 만드므로 드문 입력 하나였을 수 있다.
- **고치기 전 코드에서 실패하는지 확인한다.** 버그를 막는 테스트는 고치기 전 빌드에서 실패해야 한다 (array.md §13.1과 같다).
