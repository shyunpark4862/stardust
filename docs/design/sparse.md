# 희소 영역 설계 (sd.sparse)

> 상태: **설계 초안** (2026-11-20). 큰 결정(§1.1 SP1–SP5)은 사용자가 정했다. 나머지는 안이고, 마일스톤(SP0–SP8)은 모두 시작 전이다. 시작 전에 정할 것은 §9에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

`sd.sparse`(희소 배열과 희소 선형대수)의 확정 사항과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처의 안은 SP0에서 `docs/api/sparse.md`로 쓴다.
- 비교 대상은 scipy 1.18.1의 `scipy.sparse`와 `scipy.sparse.linalg`, R 4.6.2의 `Matrix` 1.7.6(설치되어 있다)이다. 바닥은 Accelerate의 Sparse Solvers(`Solve.h`), 배열(`csrc/array/`), 스레드풀(`csrc/parallel/`), linalg의 LAPACK 래퍼와 비용 규칙(`csrc/linalg/`)이다.
- 이 문서의 잰 값은 이 초안을 쓰며 Apple M4 Pro, macOS 26.6.2, SDK 26.5에서 본 것이다(scratchpad의 탐침 프로그램). SP0의 기준선에서 다시 잰다.
- ε은 float64의 기계 엡실론 2⁻⁵² ≈ 2.22×10⁻¹⁶이다.

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-20)
| 번호 | 결정 |
|---|---|
| SP1 범위 | 첫 판은 저장 형식과 연산, 그리고 희소 선형대수다. 형식: COO, CSR, CSC와 만들기용 형식(DOK·LIL이나 그에 맞먹는 것). BSR·DIA는 넣을 만한지 조사해 묻는다(§2.1, §9). 연산: 산술, 원소별 연산, 희소·밀집과의 행렬 곱, 슬라이싱과 인덱싱, 리덕션, 전치, 형식 변환, 만들기 함수(eye, diags, random, kron, hstack·vstack, block), `sd.ndarray`·numpy·scipy.sparse와의 변환. 희소 선형대수: `spsolve splu spsolve_triangular factorized`, 반복법 `cg gmres minres bicgstab lsqr lsmr`, `eigsh eigs svds`, `inv expm expm_multiply norm onenormest`, `LinearOperator`. 그래프 알고리즘(scipy의 csgraph)은 이 영역에 넣지 않는다(§1.2 끝) (사용자 결정 2026-11-20) |
| SP2 API | scipy의 새 희소 **배열** 꼴만 둔다(`csr_array`, `coo_array` …): `*`는 원소별 곱, `@`는 행렬 곱이다. `sd.ndarray`와 같다. 옛 `spmatrix` 클래스(`csr_matrix` 등)는 두지 않는다 (사용자 결정 2026-11-20) |
| SP3 위치 | 희소 선형대수는 하위 모듈 없이 `sd.sparse`에 바로 둔다: `sd.sparse.spsolve`, `sd.sparse.splu`, `sd.sparse.cg`, `sd.sparse.eigsh`, `sd.sparse.norm`, `sd.sparse.LinearOperator` 등. 이름은 scipy.sparse.linalg의 것이다(대응표는 §1.2). `sd.sparse.linalg`는 없다. 다른 영역(`sd.optimize`, `sd.htest`)처럼 평평하다. `sd.linalg`는 밀집만 받는다: 희소 배열을 넘기면 쓸 `sd.sparse`의 함수를 짚는 `TypeError`다(§3.10) (사용자 결정 2026-11-20) |
| SP4 바탕 | 직접 분해(Cholesky, LDLᵀ, QR)와 알맞은 반복법(CG, GMRES, LSMR)은 Accelerate의 Sparse Solvers를 쓴다. 형식, 변환, 산술, SpMV·SpMM은 이 라이브러리의 C++ 커널이고, 기존 스레드풀과 `csrc/parallel`·`csrc/linalg`·API.md의 규칙을 따른다. Accelerate가 무엇을 어느 OS부터 주는지 SDK 헤더로 확인하고, scipy에 있는데 Accelerate에 없는 것(비대칭 LU, ARPACK 꼴의 Lanczos, `expm_multiply` 등)과 만들 것을 적는다(§1.4) (사용자 결정 2026-11-20) |
| SP5 나중의 쓰임 | 약속이 아니라 메모다: optimize가 미룬 `lsq_linear`의 lsmr(O81)과 `least_squares`의 `jac_sparsity`·`tr_solver='lsmr'`(O22), 큰 희소 설계 행렬의 회귀, htest는 없음(§1.5) (사용자 결정 2026-11-20) |

### 1.2 범위 (안)
마일스톤은 §8이다. "SPn"은 그 마일스톤에서 만든다는 뜻이고 "나중"은 첫 판에 넣지 않는 후보다.

**형식과 만들기 (scipy.sparse → sd.sparse)**
| scipy.sparse | sd.sparse | 마일스톤 |
|---|---|---|
| `coo_array`, `csr_array`, `csc_array` | 같은 이름 | SP1 |
| `dok_array` | 같은 이름 (만들기용, §2.1) | SP1 |
| `lil_array` | 없음 (안, §9 SP0의 2번) | — |
| `bsr_array`, `dia_array` | 없음, 나중 (안, §2.1, §9 SP0의 1번) | 나중 |
| `csr_matrix` 등 `*_matrix`, `spmatrix`, `isspmatrix*` | 없음 (SP2) | — |
| `sparray`, `issparse` | 같은 이름 (바탕 클래스, 판별) | SP1 |
| `eye_array`, `diags_array`, `random_array`, `block_array` | 같은 이름 (안, §9 SP3의 1번) | SP3 |
| `eye`, `identity`, `diags`, `spdiags`, `random`, `rand`, `bmat` | 없음: 옛 `spmatrix`를 돌려주는 함수다 | — |
| `kron`, `kronsum`, `hstack`, `vstack`, `block_diag`, `tril`, `triu`, `find` | 같은 이름 | SP3 |
| `expand_dims`, `swapaxes`, `permute_dims`, `matrix_transpose` | `matrix_transpose`만 (2차원뿐이라, §2.4) | SP1 |
| `save_npz`, `load_npz` | 나중 | 나중 |
| `get_index_dtype`, `safely_cast_index_arrays`, `SparseWarning`, `SparseEfficiencyWarning` | 없음: 색인은 언제나 int64(§2.2), 경고가 없다 | — |
| `csgraph` | 없음: 나중의 다른 영역(아래) | — |

**희소 선형대수 (scipy.sparse.linalg.X → sd.sparse.X)**
| scipy.sparse.linalg | sd.sparse | 바탕 | 마일스톤 |
|---|---|---|---|
| `spsolve`, `splu`, `factorized`, `inv` | 같은 이름 | Accelerate LU (§3.6) | SP4 |
| `spsolve_triangular` | 같은 이름 | 이 라이브러리의 커널 | SP4 |
| `LinearOperator`, `aslinearoperator` | 같은 이름 | Python | SP4 |
| `cg`, `gmres`, `minres`, `bicgstab`, `lsqr`, `lsmr` | 같은 이름 | 이 라이브러리의 커널 (안, §3.7) | SP5 |
| `eigsh`, `eigs`, `svds` | 같은 이름 | 이 라이브러리의 Lanczos·Arnoldi (§3.8) | SP6 |
| `expm`, `expm_multiply`, `norm`, `onenormest` | 같은 이름 | 이 라이브러리의 커널 (§3.9) | SP7 |
| `SuperLU` (`splu`의 결과 형) | 다른 이름 (안, §9 SP4의 2번): 바탕이 SuperLU가 아니다 | | SP4 |
| `is_sptriangular`, `spbandwidth` | 같은 이름 (싼 도구, 안) | | SP4 |
| `ArpackError`, `ArpackNoConvergence` | 없음: 수렴하지 못하면 결과의 상태 필드다(§3.8) | | — |
| `spilu`, `lobpcg`, `bicg`, `cgs`, `qmr`, `tfqmr`, `lgmres`, `gcrotmk`, `matrix_power`, `funm_multiply_krylov`, `LaplacianNd` | 나중 | | 나중 |
| `use_solver`, `MatrixRankWarning`, 하위 모듈(`dsolve`, `isolve`, `eigen`, `interface`, `matfuncs`) | 없음 | | — |

- **평평한 이름공간의 충돌**(SP3): scipy.sparse와 scipy.sparse.linalg의 공개 이름은 서로 겹치지 않는다(scipy 1.18.1에서 `dir()`로 확인). 겹치는 것은 이름공간 밖의 뜻이다. 모두 §9 SP0의 9번에서 묻는다.
  - `sd.linalg`와 같은 이름: `inv`, `expm`, `kron`(그리고 나중의 `matrix_power`). 같은 개념이라 같은 이름이 맞다(DESIGN §1.4).
  - `sd`의 최상위와 같은 이름: `tril`, `triu`, `matrix_transpose`(그리고 두지 않는 `eye`, `permute_dims`, `expand_dims`). 같은 개념이다.
  - `norm`: `sd.linalg`에는 `norm`이 없고 Array API의 `matrix_norm`·`vector_norm`이 있다. 같은 개념의 다른 이름이 된다.
  - `random`: scipy.sparse의 옛 함수 이름이고 `sd.random`(모듈)과 헷갈린다. 배열 꼴 이름 `random_array`를 쓰면 생기지 않는다.
  - scipy 사용자가 쓰던 `sd.sparse.linalg.spsolve`는 `AttributeError`다. 별칭을 두지 않는다(별칭 없음 규칙).
- **그래프 알고리즘은 이 영역 밖이다**(SP1). scipy의 `csgraph`(연결 성분, 최단 경로, 너비·깊이 우선 순회, 최소 신장 트리, 최대 흐름, 이분 매칭, `reverse_cuthill_mckee`, `structural_rank`, `laplacian` 등)는 나중의 따로 된 최상위 영역이다(이름은 나중에 정한다, 예: `sd.graph`). 그 영역은 희소 배열(과 밀집 배열)을 입력으로 받는다. 이 영역은 그것을 미리 막지 않도록 희소 배열의 C++ 표현(§2.2)을 다른 영역이 읽을 수 있게 둔다. 분해에 필요한 순서 매기기(AMD, nested dissection)는 Accelerate가 안에서 한다(§3.6).

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다. 별칭이 없다(scipy의 `todense`·`toarray`, `conj`·`conjugate`, `dot`·`@` 같은 두 이름은 하나만 둔다; §9 SP1의 4번).
- **같은 입력이면 sd의 스레드 수와 상관없이 같은 비트다**(API.md §3). 희소 합의 순서는 §2.6이 정한다. Accelerate를 부르는 계산은 BLAS·LAPACK처럼 Accelerate의 스레드 모드가 같을 때 같은 비트이고, 같은 모드 안에서 실행마다 비트가 달라지는 Accelerate의 경로는 쓰지 않는다(§1.4의 잰 값, §3.6).
- 반올림이 아닌 동작(결과의 형식과 형, dtype 규칙, 저장되는 원소의 규칙, 상태 값의 뜻, 기본값, 멈춤 규칙)은 scipy 1.18.1과 같다(API.md §4의 규칙 2). 다르게 정한 곳은 `[scipy와 다름]`으로 이유와 함께 적는다. 반올림 크기의 결과는 정확도 계약(§4) 안이면 된다.
- **모든 호출은 정해진 시간 안에 끝난다.** 반복법과 고유값 풀이는 근거 있는 반복 상한을 두고, 상한에 닿으면 마지막 값과 상태를 돌려준다(API.md §2.2). 반복 상한의 이름은 `max_iter`다(optimize O14).
- 경고를 내지 않는다. scipy가 경고를 내는 곳(`SparseEfficiencyWarning`, `MatrixRankWarning`, 반복법의 수렴 실패)은 에러, 상태 필드, 또는 문서다.
- 참값은 Arb(python-flint 0.9.0)와 정확한 유리수만 쓴다. mpmath는 쓰지 않는다.
- 에러 종류와 메시지는 API.md §2·§2.1이다. scipy의 `NotImplementedError`, `RuntimeError`, `ZeroDivisionError`는 그 목록의 예외로 바꾼다(§2.5, §3.6).

### 1.4 Accelerate가 주는 것과 없는 것 (SDK 헤더와 잰 값)
SDK의 `Accelerate.framework/Frameworks/vecLib.framework/Headers/Sparse/`(`Types.h`, `BLAS.h`, `Solve.h` 8,339줄)를 읽었다. 이 기기의 macOS는 26.6.2라 아래가 모두 있다.

| 묶음 | 내용 | OS |
|---|---|---|
| Sparse BLAS (`BLAS.h`, `sparse_matrix_*`) | 불투명한 행렬 객체에 원소를 넣고(`sparse_insert_*`, `sparse_commit`) 곱한다: SpMV(`sparse_matrix_vector_product_dense_*`), SpMM(`sparse_matrix_product_dense_*`), 희소 × 희소(`sparse_matrix_product_sparse_*`), 삼각 풀이, 노름, 대각합, 행·열 치환, 블록 행렬. float·double | macOS 10.11 |
| 같은 것의 complex | `*_float_complex`, `*_double_complex` | macOS 15.5 |
| Sparse Solvers (`Solve.h`), 직접 | `SparseFactor`·`SparseSolve`·`SparseRefactor`·`SparseCreateSubfactor`: Cholesky, LDLᵀ(기본 TPP, Unpivoted, SBK, TPP), QR, CholeskyAtA; 순서 매기기 AMD·Metis·COLAMD·사용자; 대각 척도 조정; `SparseGetInertia`(LDLᵀ의 관성) | macOS 10.13 |
| 같은 것의 LU | `SparseFactorizationLU`(기본 = TPP), `LUUnpivoted`, `LUSPP`(초마디 안에서만 피벗), `LUTPP`(문턱 부분 피벗). 행·열 척도 `S_r A S_c = P Ā Q`, Dulmage–Mendelsohn 블록 삼각화 뒤 강연결 성분마다 LU. complex, `SparseHermitian`, `SparseUpdateFactor`(부분 재분해) | macOS 15.5 |
| 순서·척도 추가 | `SparseOrderMTMetis`(다중 스레드 Metis; LU의 기본 순서), `SparseScalingHungarian*` | macOS 26.0 |
| Sparse Solvers, 반복 | `SparseSolve(SparseConjugateGradient() / SparseGMRES() / SparseLSMR(), …)`; GMRES는 DQGMRES(기본)·재시작 GMRES·FGMRES; 전처리기는 대각(Jacobi)·열 노름 척도·사용자; 한 걸음씩 도는 `SparseIterate`(연산자는 Objective-C 블록 `^`) | macOS 10.13 (complex 15.5) |

**쓸 때의 제약** (헤더에서)
- 행렬은 열 압축(CSC; `blockSize`가 1이 아니면 블록 CSC)이다. 행 수·열 수와 **행 색인은 `int`(32비트)**, 열 시작은 `long`이다. 그래서 행·열 수는 2³¹ − 1 이하이고, int64 색인(§2.2)을 넘길 때 행 색인을 int32로 한 번 복사한다(O(nnz), 분해에 비해 작다). CSR은 CSC에 `transpose` 속성을 켜서 넘긴다(복사 없음).
- `reportError`가 NULL이면 매개변수 오류에서 `__builtin_trap()`으로 **프로세스를 멈춘다**. 언제나 콜백을 넘기고 메시지를 받아 `ValueError`·`LinAlgError`로 바꾼다.
- 분해의 인수는 불투명하다. LU는 P, Q, S_r, S_c를 곱하고 푸는 객체로만 꺼낼 수 있고 L과 U를 희소 행렬로 꺼내는 API가 없다. scipy `SuperLU`의 `L`, `U`는 줄 수 없다(§3.6).
- `pivotTolerance`는 [0, 0.5]로 잘린다. scipy SuperLU의 기본 `diag_pivot_thresh=1.0`(엄격한 부분 피벗)은 만들 수 없다.
- LU에 COLAMD 순서를 주면 `reportError`로 거절한다("not supported for this factorization type"). LU의 순서는 AMD, Metis, MTMetis, 사용자다.
- 특이 판정: 구조적으로 특이하면 `SparseMatrixIsSingular`다. **수치적으로 특이하면 성공(status 0)을 돌려준다**: 2 × 2 [[1, 2], [2, 4]]의 LU는 status 0이고, `zeroTolerance = 0`이면 풀이가 ±inf, `zeroTolerance = 10⁻¹⁴`이면 틀린 유한한 값(x = (1, −0); Ax ≠ b)이다. QR도 status 0에 x ≈ 2.6×10¹⁵다. 그래서 특이의 판정은 이쪽이 한다(§3.6).
- 반복법의 멈춤 판정은 ‖b − Ax‖₂ < rtol·‖b − Ax₀‖₂ + atol이고(처음 잔차에 대한 상대, scipy는 ‖b‖에 대한 상대), rtol의 기본은 √ε다. `SparseSolve`는 반복 수를 돌려주지 않는다(상태 넷과 `reportStatus` 글뿐).

**결정성** (이 초안에서 잼: 5점 격자 행렬 n = 3,600, 9×10⁴, 4.9×10⁵; 같은 프로세스에서 같은 입력을 6번 분해하고 풀어 결과의 바이트를 견줌)

| 분해 / 순서 | 기본 스레드 모드 | `VECLIB_MAXIMUM_THREADS=1` |
|---|---|---|
| Cholesky, LDLᵀ(넷 모두) / AMD, Metis, MTMetis | **실행마다 다르다**(6번이 6가지, 노름 상대 1–2×10⁻¹⁵) | 같다 |
| LU, LUSPP, LUTPP, LUUnpivoted / AMD, Metis | 같다 (세 크기 모두) | 같다 |
| LU / MTMetis(= LU의 기본 순서) | **실행마다 다르다**(n = 9×10⁴에서 6가지, 노름 상대 3×10⁻¹⁶; 순서 매기기가 결정적이지 않다) | 같다 |
| QR, CholeskyAtA / COLAMD | 같다 | 같다 |
| 같은 인수로 다시 풀기(`SparseSolve`) | 같다 (모든 분해) | 같다 |

- `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)`는 희소 Cholesky를 결정적으로 만들지 않는다(n = 9×10⁴, 10⁶ 모두 5번이 5가지). 희소 풀이의 스레드는 환경 변수 `VECLIB_MAXIMUM_THREADS`만 따른다. 이 라이브러리는 그 값을 정하지도 바꾸지도 않는다(API.md §3).
- 두 스레드 모드 사이에서는 LU도 비트가 다르다. BLAS·LAPACK과 같은 규칙이다(API.md §3).
- 속도(같은 꼴의 행렬, 기본 모드): n = 9×10⁴에서 Cholesky 19 ms, LDLᵀ 21 ms, LU/AMD 73 ms, LU/Metis 258 ms, LU/MTMetis 131 ms, QR 186 ms. 1스레드 모드의 Cholesky는 32 ms(n = 9×10⁴), 494 ms(n = 10⁶; 기본 모드 280 ms). scipy `splu`(SuperLU, COLAMD)는 2차원 라플라시안 n = 9×10⁴에서 225 ms다.

**Accelerate에 없어서 만들 것** (scipy에 있는 것)
- 비대칭 LU: **있다**(macOS 15.5부터). `splu`·`spsolve`는 Accelerate LU로 한다(§3.6). SP4의 결정이 적은 Cholesky·LDLᵀ·QR 가운데, scipy.sparse.linalg의 공개 함수가 쓰는 것은 LU뿐이다(scipy에는 희소 Cholesky가 없다).
- ARPACK(implicitly restarted Lanczos·Arnoldi): 없다. `eigsh`·`eigs`·`svds`는 이 라이브러리가 짠다(§3.8). 작은 밀집 부분 문제는 linalg의 LAPACK 래퍼를 쓴다.
- MINRES, BiCGSTAB, LSQR: 없다. CG·GMRES·LSMR은 있지만 멈춤 판정·반복 수·전처리의 뜻이 scipy와 다르다(§3.7).
- 삼각 풀이: Solve.h의 투명한 행렬에는 없다(Sparse BLAS의 불투명 객체에만 있고, 그러려면 원소를 복사해 넣어야 한다). 이 라이브러리가 짠다(§3.6).
- `expm`, `expm_multiply`, `onenormest`, `norm`(ord=2 포함), `LinearOperator`, 불완전 LU(`spilu`): 없다.
- 형식 변환, 원소별 연산, 인덱싱, 만들기: Sparse BLAS는 불투명 객체라 데이터를 그대로 보여 줄 수 없다. SP4대로 이 라이브러리가 짠다.

### 1.5 다른 영역과의 관계 (SP5의 메모)
- **sd.optimize**: `lsq_linear(lsq_solver='lsmr')`(O81)는 `sd.sparse.lsmr`와 `LinearOperator`가 생기면 열 수 있다. `least_squares`의 `tr_solver='lsmr'`도 같다. `jac_sparsity`(O22)는 희소 패턴에서 열을 묶는 일(scipy의 `group_columns`, 탐욕 색칠)이 필요하다: 그래프 영역을 기다리지 않고 optimize 안에 둘 수 있는 작은 알고리즘이다. 둘 다 약속이 아니다.
- **sd.regression**: 큰 희소 설계 행렬(더미가 많은 회귀)은 정규방정식 대신 희소 QR(Accelerate QR/COLAMD는 결정적이다, §1.4)이나 LSQR·LSMR로 풀 수 있다. Cholesky 경로(XᵀX)는 Accelerate의 희소 Cholesky가 결정적이지 않아서 지금은 쓸 수 없다(§3.6). 약속이 아니다.
- **sd.htest**: 없다.
- **sd.linalg**: 밀집만 받는다(SP3). 희소 배열을 받으면 짚는 함수가 있는 `TypeError`다(§3.10).
- **sd.random**: `random_array`는 `rng=`(sd.random의 `Generator`)를 꼭 받는다(§3.4).
- **나중의 그래프 영역**: 희소 배열을 입력으로 받는다(§1.2 끝).

---

## 2. 데이터 구조와 형식

### 2.1 형식 (안)
- **CSR, CSC**: 계산의 형식이다. `data`(값), `indices`(줄 안의 색인), `indptr`(줄의 시작, 길이 줄 수 + 1). CSR은 행, CSC는 열이 줄이다.
- **COO**: 만들기와 주고받기의 형식이다. `data`, `coords`(차원마다 색인 배열의 tuple; scipy의 `row`·`col`은 그 둘의 이름이다). 중복과 정렬되지 않은 원소를 가질 수 있다(§2.3).
- **DOK**: 하나씩 넣는 만들기용 형식이다. 키 (i, j) → 값의 해시 맵을 C++에 둔다(scipy는 Python `dict`의 하위 클래스다). `A[i, j] = v`가 O(1)이고, 계산할 때는 CSR·CSC로 바꾼다. Python `dict`의 메서드(`keys`, `items`, `get`, `pop`, `update` …)는 두지 않는 안이다(§9 SP1의 5번).
- **LIL은 두지 않는 안**이다. scipy의 LIL은 행마다의 Python 리스트이고 행 단위 대입이 쓰임새다. DOK와 COO 꼴의 만들기(`coo_array((data, (row, col)))`), `vstack`이 같은 일을 하고 더 빠르다(§9 SP0의 2번).
- **BSR, DIA는 첫 판에 넣지 않는 안**이다(조사, §9 SP0의 1번).
  - BSR(블록 CSR): 자유도가 여럿인 유한요소 행렬처럼 작은 밀집 블록이 모인 행렬에서 색인이 적고 SpMV가 레지스터 블록이 된다. Accelerate의 `SparseMatrix`가 `blockSize`를 그대로 받으므로 분해에도 이득이 있을 수 있다. 하지만 scipy의 BSR은 산술·인덱싱 대부분을 CSR로 바꿔서 하고, 블록 크기가 맞는 행렬에서만 이득이다. 나중에 잰 이득과 함께 묻는다.
  - DIA(대각 저장): 격자 스텐실처럼 몇 개의 대각만 있는 행렬에서 SpMV가 연속 메모리의 SIMD 루프가 된다. scipy에서는 `eye_array`·`diags_array`의 기본 형식이 DIA이고 `kron`의 기본은 BSR이다. 넣지 않으면 그 기본 형식이 CSR이 된다(`[scipy와 다름]`; 값은 같고 `.format`만 다르다).
  - 둘 다 나중 후보로 적는다. 넣으면 형식 하나마다 변환·연산·인덱싱이 따라 붙으므로, 넣는 기준은 같은 행렬에서 CSR 대비 2배(linalg §1.3의 크기 규칙과 같은 문턱)다.
- **차원**: CSR, COO, DOK는 1차원과 2차원, CSC는 2차원이다(scipy 1.18.1과 같다). scipy의 n차원 COO는 나중 후보다. 1차원은 `A[i]`(행 하나)와 희소 벡터를 위해 둔다(§9 SP0의 8번).
- **배치는 없다**(안, §9 SP0의 8번). 희소 선형대수의 함수는 행렬 하나를 받는다. 우변은 (n,)이나 (n, k)이고 k개의 열을 함께 푼다. 밀집 linalg의 `(..., M, N)` 배치와 다르다: 희소 행렬의 묶음은 행렬마다 패턴이 달라 한 배열에 담을 꼴이 없다(블록 대각 `block_diag`로 하나로 만들 수 있다).

### 2.2 색인과 값의 dtype (안)
- **값의 dtype**: 배열의 네 dtype(`bool`, `int64`, `float64`, `complex128`)이다(API.md §1). 승격 규칙도 배열과 같다(`bool < int64 < float64 < complex128`). int64·bool은 구조와 개수(인접 행렬, 패턴)와 정수 산술을 위해 둔다. scipy의 다른 dtype(int32, float32, uint8 …)은 받을 때 `TypeError`다(배열의 `asarray`와 같은 규칙; 변환은 명시적으로).
- **선형대수의 dtype**: float64와 complex128로 계산한다. bool·int64는 float64로 바꾼다(sd.linalg와 같다, DESIGN §9.6). complex는 Accelerate LU가 macOS 15.5부터 받으므로 직접 풀이도 complex를 한다(안, §9 SP0의 4번).
- **색인의 dtype은 언제나 int64다**(안, §9 SP0의 3번). `indices`, `indptr`, `coords`는 int64 `sd.ndarray`다.
  - 이유: 이 라이브러리의 정수 dtype은 int64 하나다(API.md §1). scipy처럼 값에 따라 int32와 int64를 고르면(`get_index_dtype`) 같은 연산의 결과 dtype이 데이터 크기에 따라 달라지고, 사용자에게 보이는 색인 배열의 dtype을 sd가 나타낼 수 없다.
  - 비용: SpMV가 원소마다 읽는 바이트가 12(값 8 + int32 색인 4)에서 16으로 늘어난다. 메모리 대역에 묶인 SpMV는 최대 약 1.33배 느려질 수 있다. SP0에서 같은 커널을 int32·int64로 재고, 이 비용이 1.3배를 넘으면 안쪽에만 int32를 두는 안(밖에 보일 때 int64로 복사)을 다시 묻는다.
  - Accelerate에 넘길 때는 행 색인을 int32로 복사한다(§1.4). 행·열 수가 2³¹ − 1을 넘으면 Accelerate를 부르는 함수는 `ValueError`다.
- **C++ 표현**(`csrc/sparse/format.hpp`): `Compressed{ kind(CSR|CSC), shape, NDArray data, NDArray indices, NDArray indptr }`, `Coo{ shape, NDArray data, NDArray row, NDArray col, canonical }`. 버퍼는 배열의 `Buffer`이고, Python의 `.data` 등은 같은 버퍼의 뷰다(복사 없음). 나중의 그래프 영역도 이것을 읽는다.

### 2.3 표준 꼴 (안, §9 SP0의 5번)
- **CSR·CSC는 언제나 표준 꼴이다**: 줄마다 색인이 오름차순이고 중복이 없다. 모든 생성자와 연산의 결과가 그렇다. `has_canonical_format`, `has_sorted_indices`는 언제나 True라 두지 않고, `sort_indices`, `sorted_indices`, `sum_duplicates`, `check_format`도 CSR·CSC에는 두지 않는다. `[scipy와 다름]`: scipy는 `(data, indices, indptr)`에서 만든 행렬을 검사하지 않고 비표준 꼴로 두었다가 연산마다 필요하면 정리한다.
  - 이유: 합의 순서(§2.6)가 줄의 색인 순서로 정해지므로 표준 꼴이어야 결과의 비트가 "행렬의 값"만으로 정해진다. 이분 탐색 인덱싱, Accelerate 넘기기, 두 행렬의 병합 연산이 모두 정렬을 전제한다. 생성자의 검사는 O(nnz)이고, 정렬되어 있으면 복사도 하지 않는다.
  - `(data, indices, indptr)`로 만들 때: 범위(0 ≤ 색인 < 줄 길이, `indptr` 단조)를 검사하고(틀리면 `ValueError`), 정렬되지 않았거나 중복이 있으면 정렬하고 더한다(`copy=False`면 `ValueError`).
- **COO는 비표준 꼴을 가질 수 있다**: 만들기의 형식이기 때문이다(scipy와 같다). `sum_duplicates()`는 행 우선(C 순서)으로 정렬하고 중복을 더한다. `has_canonical_format`은 COO에만 있다.
- **중복은 입력 순서대로 왼쪽부터 차례로 더한다**(COO → CSR·CSC, `sum_duplicates`, 중복이 있는 `(data, indices, indptr)`). 같은 자리의 원소를 입력에 나온 순서대로 놓는 안정 정렬 뒤에 순차 합이다. scipy와 같은 결과다(`[1, 10¹⁶, 1, −10¹⁶]` → 0을 이 초안에서 확인; 3개 이상의 중복에서 scipy의 순서는 SP0에서 더 확인한다). 정렬은 줄 번호로 안정 계수 정렬(고정 조각마다 개수를 세고 조각 순서대로 누적 → 스레드 수와 무관하게 안정), 줄 안은 색인으로 안정 정렬이다.

### 2.4 저장된 0과 NaN (안, §9 SP0의 6번·7번)
- **저장된 0(explicit zero)**: 사용자가 넣은 0은 저장된 원소로 남는다(scipy와 같다). `nnz`는 저장된 원소 수이고 `count_nonzero()`는 0이 아닌 값의 수다. `eliminate_zeros()`가 저장된 0을 지운다. 패턴을 지키는 것은 같은 패턴으로 다시 분해(`SparseRefactor`)하거나 사용자가 정한 구조를 쓰는 데 필요하다.
- **연산이 만든 0을 지우는지**는 연산마다 scipy의 규칙을 따르는 안이다(§3.2의 표): 희소끼리의 `+`, `-`, `*`, 비교와 희소 × 희소 곱은 결과의 정확한 0을 지우고(scipy의 `csr_binop_csr`, `csr_matmat`), 스칼라 곱(`A * 0`)과 원소별 함수는 패턴을 그대로 둔다. scipy 안에서도 둘이 다르지만(`A - A`는 nnz 0, `A * 0`은 nnz 그대로) `nnz`는 보이는 동작이라 규칙 2(API.md §4)대로 따른다. 대안은 "결과의 패턴은 입력의 패턴만으로 정한다"(값을 보지 않는다)이다.
- **암시적 0은 강한 0이다**: 저장되지 않은 원소는 산술에 들어가지 않는다. 그래서
  - `A * nan`, `A * inf`(스칼라)는 저장된 원소만 NaN·inf가 되고 암시적 0은 0이다(IEEE의 0·inf = NaN이 아니다). scipy와 같다.
  - `A @ x`에서 x_j가 NaN이어도 열 j에 저장된 원소가 없는 행은 NaN이 되지 않는다(BLAS의 희소 관례, scipy와 같다).
  - `A.multiply(B)`(밀집 B)도 암시적 0 자리는 0이다(B가 NaN이나 inf여도). scipy와 같다.
  - `A / c`: 저장된 원소만 나눈다. **c = 0이면 저장된 원소가 ±inf·NaN이 되고 암시적 0은 0이다**(`A * (1/c)`와 같은 규칙). `[scipy와 다름]`: scipy는 Python의 `ZeroDivisionError`다. 배열의 원소별 연산은 값 때문에 실패하지 않는다(API.md §2, docs/api/array.md §1.6)는 규칙을 따른다.
  - 예외는 결과가 밀집인 연산이다: 희소 ÷ 희소는 scipy처럼 밀집 배열을 돌려주고 0/0은 NaN이다(scipy와 같다). 희소 + 밀집은 밀집이고 IEEE대로다.
- **리덕션의 NaN**: `sum`, `mean`은 저장된 NaN을 전파한다. `max`, `min`은 NaN을 전파하고(배열과 같다), 암시적 0이 하나라도 있으면 0도 후보다(scipy와 같다). `nanmax`, `nanmin`은 NaN을 건너뛴다.
- **비교**: IEEE대로다(저장된 NaN끼리 `==`는 False, 암시적 0끼리는 True). `A == 0`, `A <= 1`처럼 암시적 0에서 참인 비교는 scipy처럼 True가 많은 희소 bool 배열이다(경고 없음).

### 2.5 Python 클래스와 변환 (안)
- 희소 배열은 Python 클래스다(frame의 `Series`처럼 "얇은 Python"의 예외, ARCHITECTURE §3). 필드는 `sd.ndarray`이고 계산은 C++ 커널 한 번 호출이다. 바탕 클래스 `sparray`가 공통 메서드를 갖는다.
- **`indices`, `indptr`, `coords`는 쓰기 금지 배열이다**(안, §9 SP1의 2번). 바꾸면 표준 꼴(§2.3)이 깨지기 때문이다. `data`는 쓸 수 있다(값만 바꾸는 흔한 쓰임; `A.data *= 2`). scipy는 모두 쓸 수 있다(`[scipy와 다름]`).
- **sd.ndarray ↔ 희소**
  - `csr_array(x)`(밀집 `sd.ndarray`, 1·2차원): 0이 아닌 원소만 저장한다. NaN은 0이 아니므로 저장된다.
  - `A.toarray()`: C 순서의 새 `sd.ndarray`다. `todense()`는 두지 않는 안이다(scipy의 배열 꼴에서 둘은 같은 일을 한다; 별칭 없음).
  - 밀집으로 몰래 바뀌는 것을 막는다: `np.asarray(A)`는 `TypeError`(`__array__`가 `toarray()`를 짚는다)인 안이다(scipy는 0차원 object 배열을 만든다).
- **numpy**: 생성자는 데이터를 받는 자리라 numpy 배열(밀집 2차원, 또는 `(data, indices, indptr)`·`(data, (row, col))`의 각 배열)을 받는다(API.md §5의 데이터 입력). 받은 것은 복사한다. 연산자의 밀집 피연산자는 `sd.ndarray`만이다: `A @ np_array`는 `TypeError`(배열과 같은 규칙, API.md §5). 
- **scipy.sparse**
  - 들여오기: 생성자가 scipy의 희소 배열과 옛 행렬(모든 형식)을 받는다. **언제나 복사한다**(안): scipy 객체는 나중에 제자리에서 바뀔 수 있고(scipy의 `sum_duplicates`, `sort_indices`는 제자리 정렬이다) int32 색인은 어차피 넓혀야 한다. dtype은 §2.2의 규칙이다.
  - 내보내기: `A.to_scipy()`(이름은 frame의 `to_numpy`와 같은 꼴; §9 SP1의 3번)가 같은 형식의 scipy 배열(`csr_array` 등)을 만든다. 값·색인 배열은 복사하지 않는 numpy 뷰다(배열의 버퍼 프로토콜; 이 초안에서 scipy 1.18.1이 int64 색인을 `(data, indices, indptr)`로 받을 때 복사하지 않는 것을 확인했다). 색인 뷰는 읽기 전용이다. `copy=True`면 복사한다.
- **Python 스칼라와 연산자**: `A + c`(c ≠ 0)와 `A ** 0`은 결과가 밀집이 된다. scipy는 `NotImplementedError`다. 이쪽은 `ValueError`(`'adding a nonzero scalar to a sparse array is not supported; use A.toarray() + c'`)인 안이다(`[scipy와 다름]`: 예외 종류만, API.md §2의 목록에 `NotImplementedError`가 없다).

### 2.6 합의 순서와 결정성
**희소 합의 결과는 행렬의 값과 모양으로만 정해진다**: sd의 스레드 수, 저장 형식(CSR·CSC·COO), 태스크 분할과 무관하다(안, §9 SP0의 10번).

- **줄 합의 규칙**: 한 출력 원소에 들어가는 항이 n개이면, 항을 그 줄의 색인 오름차순으로 놓고 0에서 시작해 차례로 더한다. 곱의 합은 항마다 `fma(a, x, acc)`다. n이 L(안: 4096)을 넘으면 L개씩의 고정 조각을 각각 차례로 더하고, 조각들의 합을 `base/pairwise.hpp`의 트리로 더한다. 트리는 n으로만 정해진다.
  - **SpMV**(`A @ x`): yᵢ = Σⱼ aᵢⱼxⱼ, j 오름차순. scipy의 `csr_matvec`는 `sum += Ax[jj]*Xx[Aj[jj]]`를 clang이 FMA로 축약한 순차 합이다: 이 초안에서 2000 × 2000(밀도 0.01) 무작위 행렬로 scipy의 결과가 "FMA 순차 합"과 비트까지 같고 "축약 없는 순차 합"과는 다른 것을 확인했다. 그래서 줄이 L 이하이면(실제 행렬의 거의 모든 줄) **scipy와 비트까지 같다**. scipy의 CSC `@`(열 순서의 흩뿌리기)도 CSR과 비트가 같았다: 열 순서로 흩뿌리면 각 행이 j 오름차순의 순차 합이 되기 때문이다. 같은 규칙이 형식과 무관한 비트를 준다.
  - **SpMM**(`A @ X`, X는 (n, k)): 출력 원소마다 위와 같은 합이다. 그래서 결과의 열 j는 `A @ X[:, j]`와 비트가 같다(scipy도 같다, 확인함). 구현은 X의 행을 NEON으로 k 방향으로 묶지만 원소 하나의 순서는 바꾸지 않는다.
  - **밀집 @ 희소**(`x @ A`)와 **Aᵀ @ x**: 같은 규칙을 Aᵀ에 쓴다(항은 행 번호 오름차순).
  - **희소 @ 희소**(SpGEMM): Cᵢⱼ = Σₖ aᵢₖbₖⱼ, k 오름차순의 FMA 순차 합. scipy의 `csr_matmat`(A의 행 i의 원소 순서대로 B의 행을 흩뿌려 더함)과 같은 순서다. L 조각 규칙은 여기에 쓰지 않는다(흩뿌리기의 누산기 하나라 조각을 나눌 자리가 없다; 결과의 비트는 여전히 값과 모양만으로 정해진다).
  - **축 리덕션**(`sum(axis=0|1)`, `mean`): 줄마다 위의 규칙(곱 없이 덧셈). scipy는 `A @ ones`로 구해 같은 순차 합이다(줄이 L 이하면 비트가 같다). 대안은 줄마다 pairwise(`base/pairwise.hpp`의 트리)다: 긴 줄에서 더 정확하지만 L 이하에서도 scipy와 다르다.
  - **전체 리덕션**(`sum()`): 저장된 값을 논리적 C 순서(행 우선, 줄 안은 색인 오름차순)로 놓고 `base/pairwise.hpp`의 트리로 더한다. CSR이면 `sd.sum(A.data)`와 비트가 같다. CSC·COO도 같은 C 순서로 더한다(CSC는 그 순서를 만드는 한 번의 훑기가 든다). linalg의 "항의 순서는 논리적 C 순서"(linalg.md §8.2)와 같은 원칙이다. scipy는 `data.sum()`(numpy의 pairwise, 저장 순서)이라 끝자리가 다를 수 있다(array.md §6.2).
  - **중복 더하기**: §2.3(입력 순서의 순차 합).
- **병렬 분할** (모두 크기만으로 정해 비트를 바꾸지 않는다)
  - CSR의 행 방향 일(SpMV, SpMM, 행 리덕션, 원소별 연산): 출력 행을 nnz가 약 2¹⁴개씩 되는 고정 구간으로 나눈다(`indptr`만으로 정함). 행 하나는 한 태스크가 계산하므로 순서가 직렬과 같다. L보다 긴 행의 조각은 다른 태스크가 계산해도 같은 트리로 합친다.
  - CSR의 열 방향 일(Aᵀx, 열 합, CSC의 SpMV): 출력 구간으로 나누고 태스크마다 모든 줄에서 자기 구간을 이분 탐색으로 찾는다(색인이 정렬되어 있어서 된다). 항의 순서가 직렬의 흩뿌리기와 같다. 줄 수가 많고 줄이 짧아 이분 탐색이 비싸면 한 번 전치(병렬 계수 정렬)해서 행 방향 일로 한다. 어느 쪽인지는 모양으로만 고르고(SP2에서 잰다), 결과는 같다.
  - 변환과 중복 더하기: 고정 조각의 계수 정렬(§2.3).
  - 기준값은 배열의 규칙이다(array.md §10.4): 작업량(바이트로 센 nnz와 출력 크기)이 2¹⁵ float64 분량 이상이면 나누고, 그 이상이면 스레드 수와 상관없이 GIL을 놓는다(API.md §3).
- **Accelerate**: 같은 입력, 같은 Accelerate 스레드 모드에서 비트가 같은 경로만 쓴다(§1.4의 표: LU는 AMD·Metis 순서, QR은 COLAMD). 분해 호출은 Accelerate가 안에서 스레드를 정하고, sd는 GIL을 놓고 차례로 부른다(linalg의 큰 호출과 같다, array.md §9.4).
- 테스트는 1, 2, 3, 7, 10스레드에서 같은 바이트, 같은 행렬의 CSR·CSC·COO에서 같은 바이트를 확인한다.

---

## 3. 수치 방법 (안)

### 3.1 형식 변환과 만들기 (SP1)
- COO → CSR·CSC: 줄 번호의 안정 계수 정렬(고정 조각, §2.3) → 줄 안의 안정 정렬(짧은 줄은 삽입 정렬, 긴 줄은 기수 정렬) → 중복의 순차 합. 이미 표준 꼴인 COO(`has_canonical_format`)는 정렬 없이 `indptr`만 만든다.
- CSR ↔ CSC: 전치와 같은 계수 정렬이고 정렬된 입력에서 결과도 바로 정렬된다(안정이라서). `.T`는 복사 없는 뷰다: CSR의 `.T`는 같은 버퍼의 CSC다(scipy와 같다, 이 초안에서 확인).
- 밀집 → 희소: 0이 아닌 원소를 세는 패스 하나와 채우는 패스 하나, 행 구간으로 병렬.
- `astype`: 배열의 명시적 변환 규칙(array.md §2.4: float → int는 포화, NaN → 0). 변환 뒤 0이 된 값도 저장된 원소로 남는다(scipy와 같다).
- DOK: C++ 해시 맵(열림 주소법). CSR로 바꿀 때 키를 정렬한다(결과는 넣은 순서와 무관하다).

### 3.2 산술과 원소별 연산 (SP2)
| 연산 | 결과 | 0을 지우나 (scipy와 같게) |
|---|---|---|
| 희소 ± 희소, 희소 `*` 희소, 희소 비교 희소 | 희소 (CSR, 피연산자가 둘 다 CSC면 CSC) | 지운다 |
| 희소 `*`·`/` 스칼라, `-A`, `abs(A)`, f(0) = 0인 원소별 함수 | 희소, 같은 패턴 | 지우지 않는다 |
| 희소 `*` 밀집, `A.multiply(dense)` | 희소 (scipy는 COO; 이쪽은 A의 형식인 안) | 지우지 않는다 |
| 희소 ± 밀집, 희소 ÷ 희소 | 밀집 `sd.ndarray` | — |
| 희소 `@` 밀집, 밀집 `@` 희소 | 밀집 `sd.ndarray` (1차원 @ 1차원은 Python 스칼라) | — |
| 희소 `@` 희소 | 희소 | 지운다 |
| `A + c`(c ≠ 0), `A ** 0` | `ValueError` (§2.5) | — |

- 희소끼리의 이항 연산은 두 정렬된 줄의 병합이다(줄마다 독립, 행 구간으로 병렬). 결과 dtype은 배열의 승격 규칙이다.
- 원소별 함수는 f(0) = 0인 것만이다(scipy의 메서드 목록: `sin tan arcsin arctan sinh tanh arcsinh arctanh sqrt expm1 log1p deg2rad rad2deg floor ceil trunc rint sign`, `power(p)`(p > 0), `conj`, `real`, `imag`). 값은 배열의 같은 함수와 비트까지 같다(`A.sin().data`와 `sd.sin(A.data)`). 이것들을 메서드로 둘지 `sd.sin(A)`가 받게 할지는 §9 SP2의 2번이다.
- 정수 산술은 배열처럼 wrap한다(scipy·numpy와 같다: 이 초안에서 `[[2⁶², 2⁶²]].sum()`이 −2⁶³).
- SpGEMM: Gustavson의 행 단위 알고리즘. 행마다 밀집 누산기(열 수가 크면 해시 누산기)로 k 오름차순에 더하고, 결과 행의 색인을 정렬한다. 두 패스(기호: 결과 nnz 세기, 수치: 채우기), 행 구간으로 병렬.
- SpMV·SpMM: §2.6의 순서. 행마다 FMA 사슬 하나이고, 이웃한 행의 사슬은 독립이라 비순차 실행이 겹친다. SpMM은 X의 행 k개를 NEON 레인으로 묶는다(k ≥ 2).

### 3.3 인덱싱과 리덕션 (SP2, SP3)
- 리덕션: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `nanmax`, `nanmin`, `count_nonzero`, `trace`, `diagonal`. 순서는 §2.6. `axis`가 있으면 **언제나 밀집 `sd.ndarray`를 돌려주는 안**이다(`[scipy와 다름]`: scipy는 `sum(axis=)`·`mean(axis=)`이 밀집, `max(axis=)`·`min(axis=)`이 COO 희소라 같은 꼴의 리덕션이 다른 형을 낸다; §9 SP2의 3번). 빈 행렬의 `max`는 `ValueError`(배열·scipy와 같다), `sum`은 0이다. bool의 `sum`은 int64다(배열과 같다).
- `argmax`·`argmin`: 암시적 0도 자리로 센다. 동점이면 C 순서의 첫 자리, NaN이면 첫 NaN의 자리(배열과 같다).
- 인덱싱(SP3): 정수, 슬라이스, 정수 배열, bool 마스크, 그 조합. `A[i, j]`는 Python 스칼라, `A[i]`는 1차원 희소(scipy와 같다), 슬라이스·배열 인덱싱은 희소다. CSR에서 행 고르기는 줄 복사, 열 고르기는 줄마다 이분 탐색(정렬된 열 배열이면 병합)이다. CSC는 거울이다.
- 대입(SP3): DOK는 O(1). CSR·CSC의 대입은 이미 저장된 자리면 값만 바꾸고, 새 자리면 패턴을 다시 만든다(O(nnz)). scipy처럼 경고(`SparseEfficiencyWarning`)는 없다. 같은 자리가 여러 번 나오면 마지막 값이 남는다(배열의 scatter와 같다, API.md §3).

### 3.4 만들기 함수 (SP3)
- `eye_array(m, n=None, *, k=0, dtype=float64, format=None)`, `diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None)`, `random_array(shape, *, density=0.01, format='coo', dtype=None, rng, data_sampler=None)`, `kron`, `kronsum`, `hstack`, `vstack`, `block_array`, `block_diag`, `tril`, `triu`, `find`.
- `format=None`의 기본 형식: scipy는 `eye_array`·`diags_array`가 DIA, `kron`이 BSR이다. DIA·BSR이 없으면 CSR로 하는 안이다(§2.1, `[scipy와 다름]`).
- `random_array`: `rng`는 sd.random의 `Generator`이고 꼭 받는다(optimize O31·htest와 같은 꼴의 `TypeError`; scipy의 `rng=None`은 시드 없는 무작위다). `random_state`는 없다. 자리 고르기는 비복원 추출(정확히 round(density·m·n)개, scipy와 같은 뜻), 값은 `data_sampler(size)` 또는 [0, 1) 균등이다. 같은 `rng` 상태면 스레드 수와 상관없이 같은 비트다. scipy와 같은 값은 아니다(난수 생성기가 다르다).
- `kron(A, B)`: 결과의 원소 aᵢⱼ·bₖₗ은 곱 하나라 순서 문제가 없다. 결과의 저장 순서는 표준 꼴(§2.3)이다.

### 3.5 전치, 모양, 그 밖 (SP1)
- `.T`, `.mT`, `transpose()`, `matrix_transpose(A)`: 2차원이면 같은 일이다(배열과 같은 이름들; `.T`는 뷰). `conj()`, `real`, `imag`, `copy()`, `astype`, `reshape`(COO를 거친다), `resize`, `nonzero()`, `eliminate_zeros()`, `setdiag`, `tocoo`·`tocsr`·`tocsc`·`todok`, `asformat`, `count_nonzero`, `nnz`, `shape`, `ndim`, `size`, `dtype`, `format`.
- 두지 않는 것: `dot`(`@`와 같다), `conjugate`(`conj`와 같다), `todense`(`toarray`와 같다), `power`의 정수 지수 0, `prune`(표준 꼴이라 할 일이 없다), `getrow`·`getcol`(옛 꼴).

### 3.6 직접 풀이 (SP4)
- **LU는 Accelerate의 `SparseFactorizationLU`(문턱 부분 피벗)를 AMD 순서로 부른다**(안, §9 SP4의 1번). LU의 기본 순서 MTMetis와 `SparseOrderDefault`는 쓰지 않는다: 실행마다 결과가 다르다(§1.4). Metis는 결정적이지만 이 꼴의 행렬에서 AMD보다 3.5배 느렸다. 순서를 고르는 옵션을 둘지도 같은 질문이다.
- **Cholesky·LDLᵀ는 첫 판에 쓰지 않는다**(안, §9 SP0의 13번). Accelerate의 대칭 분해는 기본 스레드 모드에서 실행마다 비트가 다르고, 프로세스 안에서 그것을 막을 길이 없다(`BLASSetThreading`도 듣지 않는다, §1.4). scipy.sparse.linalg의 공개 함수에는 대칭 분해가 없으므로(모두 SuperLU의 LU다) 범위는 줄지 않는다. 대칭 행렬도 LU로 푼다(대칭을 쓰지 못해 n = 9×10⁴에서 Cholesky 19 ms 대 LU 73 ms; scipy의 SuperLU 225 ms보다는 빠르다).
  - 대안 1: 이 라이브러리가 결정적인 희소 Cholesky를 짠다(초마디 Cholesky: 기호 분석과 제거 트리, 초마디마다 밀집 `potrf`·`trsm`·`syrk`, 트리 위의 병렬은 형제의 결과를 고정 순서로 더한다). CHOLMOD 꼴이라 큰 일이고 나중 후보다(regression의 정규방정식, eigsh의 shift-invert에 쓰인다).
  - 대안 2: 실행마다 다른 비트를 문서화하고 Accelerate를 그대로 쓴다. 프로젝트 규칙(API.md §3)의 예외가 된다.
- **특이의 판정**: Accelerate는 구조적 특이만 알린다(§1.4). `zeroTolerance = 0`으로 분해하면 정확한 0 피벗이 풀이에 ±inf·NaN을 낸다. 그래서 (1) 구조적 특이(`SparseMatrixIsSingular`)는 `LinAlgError("singular matrix")`, (2) 유한한 입력의 풀이가 유한하지 않으면 `LinAlgError("singular matrix")`다(안, §9 SP4의 4번). linalg의 "특이는 정확한 0 피벗만"(linalg.md §2.2)에 가장 가깝지만, 피벗을 볼 수 없으므로 넘침으로 생긴 inf도 특이로 보고한다. `splu(A)`는 분해할 때 1의 벡터로 한 번 풀어 확인한다(O(nnz(L + U)), 분해보다 작다).
  - `[scipy와 다름]`: scipy `spsolve`는 특이 행렬에서 `MatrixRankWarning`과 NaN 배열을, `splu`는 `RuntimeError("Factor is exactly singular")`를 낸다. 여기서는 둘 다 `LinAlgError`다(linalg와 같다).
- **유한성 검사**: `A`와 `b`에 NaN·inf가 있으면 `ValueError`다(linalg.md §2.3과 같다; 저장된 원소만 본다).
- `spsolve(A, b, /)`: A는 정방 CSR·CSC(다른 형식은 CSC로 바꾼다). b가 (n,)이면 (n,), (n, k)이면 (n, k) 밀집, b가 희소면 결과도 희소(scipy와 같다). `permc_spec`, `use_umfpack`은 없다.
- `splu(A, /)`: 분해 객체. 메서드 `solve(b, /, *, trans='N')`(`'N'`, `'T'`, `'H'`; Accelerate의 전치·켤레 전치 풀이), 속성 `shape`, `nnz`(인수의 크기를 줄 수 있으면). scipy `SuperLU`의 `L`, `U`, `perm_r`, `perm_c`는 줄 수 없다(§1.4; §9 SP4의 2번). `permc_spec`, `diag_pivot_thresh`, `relax`, `panel_size`, `options`는 SuperLU의 옵션이라 두지 않는다.
- `factorized(A, /)`: `splu(A).solve`와 같은 호출 가능한 객체.
- `inv(A, /)`: 단위 행렬을 우변으로 푼 희소 결과(scipy와 같다; 결과의 0은 지운다). 대개 밀집이 되므로 문서에 적는다.
- `spsolve_triangular(A, b, /, *, lower=True, unit_diagonal=False)`: 이 라이브러리의 커널이다. CSR의 앞(뒤)으로 대입: xᵢ = (bᵢ − Σⱼ aᵢⱼxⱼ)/aᵢᵢ, 합은 §2.6의 줄 합(j 오름차순 FMA). 우변이 여럿이면 열마다 병렬(열끼리 독립). 한 우변 안의 병렬은 단계 스케줄링(같은 단계의 행은 서로 기대지 않는다)으로 하고 행 하나의 순서는 바뀌지 않는다. 대각의 0은 `LinAlgError("singular matrix")`(linalg.md §2.2의 `zero_on_diagonal`과 같다). scipy의 `overwrite_A`, `overwrite_b`는 없다.
- `is_sptriangular`, `spbandwidth`: 패턴만 보는 O(nnz) 도구.
- 정확도는 SuperLU와 다른 피벗(문턱 ≤ 0.5)과 척도 조정이라 끝자리뿐 아니라 역오차의 크기도 다를 수 있다. 계약은 역오차로 정한다(§4). 반복 다듬기(iterative refinement)를 넣을지는 SP4에서 잰 역오차를 보고 묻는다.

### 3.7 반복법 (SP5)
- **이 라이브러리가 짜는 안이다**(§9 SP0의 11번; SP4의 "알맞은 반복법은 Accelerate"를 바꾼다). 이유:
  - scipy와 같은 멈춤 판정과 상태(`info`)를 지킬 수 없다: Accelerate의 판정은 처음 잔차에 대한 상대이고(scipy는 ‖b‖에 대한 상대, rtol 기본 10⁻⁵ 대 √ε), 반복 수를 돌려주지 않으며, GMRES의 기본은 DQGMRES다(scipy는 재시작 GMRES, `restart=20`). 한 걸음씩 부르는 `SparseIterate`로 판정을 직접 하면 판정은 맞출 수 있지만, 그 안의 직교화·Givens 회전·내적의 순서는 열 수 없다.
  - MINRES, BiCGSTAB, LSQR은 어차피 짜야 한다.
  - 반복법의 비용은 SpMV(이 라이브러리의 커널, §2.6)와 내적·axpy다. 알고리즘 자체는 짧다(CG 수십 줄). 같은 커널과 결정적인 내적(linalg의 `fma_dot_leaf` pairwise, linalg.md §8.2)을 쓰면 비트가 스레드 수와 무관하다.
  - Accelerate의 CG·GMRES·LSMR은 SP0에서 속도의 비교 대상으로 잰다.
- **scipy 1.18.1과 같은 알고리즘과 멈춤 판정**(optimize O9와 같은 원칙): `cg`(전처리 CG, ‖r‖ ≤ max(rtol·‖b‖, atol)), `gmres`(재시작 GMRES, 오른쪽이 아닌 scipy의 전처리 꼴, `restart`, scipy의 `maxiter`가 세는 바깥 반복), `minres`(Paige–Saunders, `shift`), `bicgstab`, `lsqr`(Paige–Saunders 1982, `damp`, `atol`·`btol`·`conlim`), `lsmr`(Fong–Saunders 2011, `damp`, `atol`·`btol`·`conlim`). scipy의 Python·Fortran 판을 계산 순서까지 옮기고, 반올림의 순서가 다른 곳(내적, 노름)만 끝자리가 다를 수 있다. 반복 수와 상태는 scipy와 같은 것을 목표로 하고 시험 문제(§7)에서 견준다.
- 입력: `A`는 희소 배열, 밀집 `sd.ndarray`, `LinearOperator`. `M`(전처리기)도 같다. `x0`.
- 이름(안, §9 SP5): 반복 상한은 `max_iter`(scipy `maxiter`, lsqr의 `iter_lim`; optimize O14). `callback`, `show`, `check`, `calc_var`(lsqr)는 두지 않는다(optimize O37·O54와 같다).
- 결과(안, §9 SP5의 1번): scipy의 튜플 대신 읽기 전용 결과 객체이고, scipy가 돌려주는 개수와 순서로 풀린다(htest H7과 같은 규칙): `x, info = cg(A, b)`. 필드는 `x`, `converged`, `status`(scipy의 `info`·`istop`), `n_iter`, 그리고 lsqr·lsmr의 `normr`, `normar`, `norma`, `conda`, `normx` 등(scipy의 이름을 regression·optimize의 이름 규칙으로 다듬는다). 배열 필드는 쓰기 금지다(API.md §2.2).
- `LinearOperator`가 Python 함수면 반복마다 Python 콜백이다(GIL을 쥔다, optimize §2.1과 같다). 희소·밀집 배열이면 반복 전체가 C++이다.

### 3.8 고유값과 특이값 (SP6)
- Accelerate에 ARPACK이 없다(§1.4). 이 라이브러리가 짠다.
- **`eigsh`**: implicitly restarted Lanczos(ARPACK `dsaupd`의 IRLM; Sorensen 1992, Lehoucq–Sorensen–Yang 1998). 재직교화는 전체 재직교화(ARPACK은 DGKS 다시 직교화), 작은 삼중대각 문제는 LAPACK `dstev`·`dsteqr`(linalg 래퍼에 더한다), 이동은 정확한 이동(exact shifts). `which='LM' 'SM' 'LA' 'SA' 'BE'`, `k`, `ncv`, `tol`(0이면 ε), `M`(일반화 문제), `sigma`(shift-invert: (A − σM)의 LU, §3.6), `mode='normal'`(그리고 `'buckling'`, `'cayley'`는 §9 SP6의 4번).
- **`eigs`**: implicitly restarted Arnoldi(ARPACK `dnaupd`의 IRAM). 작은 헤센베르크 문제는 `dhseqr`·`dtrevc`(래퍼에 더한다). `which='LM' 'SM' 'LR' 'SR' 'LI' 'SI'`. 실수 입력의 결과는 scipy처럼 complex다.
- ARPACK을 계산 순서까지 옮길지(같은 시작 벡터에서 scipy와 반복점이 같다), 같은 수학의 Krylov–Schur(Stewart 2001; 재시작이 단순하고 튼튼하다)로 할지는 §9 SP6의 1번이다. 추천은 ARPACK 옮기기다: 멈춤 판정·`ncv`의 뜻·상태가 scipy와 같아야 한다(규칙 2).
- **시작 벡터**: scipy는 `v0=None`이면 `rng=None`의 시드 없는 무작위라 실행마다 결과가 다르다. 여기서는 `v0=None`이면 정해진 벡터(sd.random의 Philox, 고정된 키에서 [−1, 1) 균등; 문서에 적는다)를 쓰고 `rng=`로 바꿀 수 있는 안이다(§9 SP6의 2번).
- **`svds`**: 추천은 Golub–Kahan–Lanczos 이중대각화의 두꺼운 재시작(Baglama–Reichel 2005, IRLBA; PROPACK 꼴)이다. scipy의 기본 `solver='arpack'`은 AᵀA(또는 AAᵀ)의 `eigsh`라 조건수를 제곱해 작은 특이값의 정확도를 잃는다. `solver=`는 두지 않는 안이다(§9 SP6의 3번). `which='LM' 'SM'`, `k`, `ncv`, `tol`, `v0`, `max_iter`, `return_singular_vectors`.
- 수렴하지 못하면: scipy는 `ArpackNoConvergence`(예외)다. 여기서는 linalg의 분해처럼 `LinAlgError`로 할지, 결과 객체의 `converged`·찾은 개수로 할지 §9 SP6의 5번이다. 추천은 결과 객체의 상태(API.md §2.2의 "상태 필드가 있으면 마지막 값")이고, scipy처럼 `(w, v)`로 풀린다.
- 결정성: 시작 벡터가 정해져 있고 SpMV·내적·직교화가 결정적이며, 작은 부분 문제는 LAPACK(같은 스레드 모드에서 같은 비트)이다.

### 3.9 행렬 함수와 노름 (SP7)
- **`expm(A, /)`**: scipy와 같은 Al-Mohy–Higham(2009) 크기 조정과 제곱, Padé 차수(3, 5, 7, 9, 13)를 희소 곱과 `spsolve`(LU, §3.6)로 한다. 차수 선택의 노름은 `onenormest`(정확히 구하는 작은 경우는 정확히, linalg.md §11.2와 같은 원칙). 결과는 희소다(scipy와 같다; 대개 밀집에 가깝다). 밀집 `sd.linalg.expm`과 같은 알고리즘이고 계약도 같은 꼴이다(§4).
- **`expm_multiply(A, B, /, *, start=None, stop=None, num=None, endpoint=None, traceA=None)`**: Al-Mohy–Higham(2011)의 잘린 Taylor와 (s, m) 선택. 노름 추정은 `onenormest`로 한다. scipy와 같은 매개변수와 멈춤(허용오차 2⁻⁵³)이다.
- **`onenormest(A, /, *, t=2, itmax=5, compute_v=False, compute_w=False)`**: Higham–Tisseur(2000) 블록 1-노름 추정. scipy는 첫 열 말고는 numpy의 시드 없는 무작위 ±1 열이라 실행마다 다를 수 있다. 여기서는 고정된 열(sd.random의 고정 키)이라 결정적이다(안, §9 SP7의 1번).
- **`norm(x, /, ord=None, axis=None)`**: scipy의 뜻(행렬: `'fro'`, 1, −1, inf, −inf, 2(가장 큰 특이값, §3.8의 `svds`); 벡터 축: 정수 p). 합은 §2.6. `sd.linalg`의 `matrix_norm`·`vector_norm`과 이름이 다른 문제는 §9 SP0의 9번이다.
- **`LinearOperator`**: Python 클래스. `shape`, `dtype`, `matvec`, `rmatvec`, `matmat`, `rmatmat`, `@`, `+`, 스칼라 `*`, `.T`, `.H`, `adjoint()`. `aslinearoperator(A)`는 희소·밀집·`LinearOperator`를 받는다. 희소·밀집에서 만든 연산자는 C++ 커널을 바로 쓴다(Python을 거치지 않는다).

### 3.10 sd.linalg의 TypeError (SP4)
- `sd.linalg`의 함수가 희소 배열을 받으면 `TypeError`이고 메시지가 쓸 함수를 짚는다(SP3): `solve(): 'a' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.spsolve`.

  | sd.linalg | 짚는 것 |
  |---|---|
  | `solve` | `sd.sparse.spsolve` |
  | `solve_triangular` | `sd.sparse.spsolve_triangular` |
  | `lu`, `lu_solve` | `sd.sparse.splu` |
  | `inv` | `sd.sparse.inv` |
  | `expm` | `sd.sparse.expm` |
  | `eigh`, `eigvalsh` | `sd.sparse.eigsh` |
  | `eig`, `eigvals` | `sd.sparse.eigs` |
  | `svd`, `svdvals` | `sd.sparse.svds` |
  | `matrix_norm`, `vector_norm` | `sd.sparse.norm` |
  | `lstsq` | `sd.sparse.lsqr` 또는 `sd.sparse.lsmr` |
  | `kron` | `sd.sparse.kron` |
  | 그 밖 | `A.toarray()`로 바꾸라는 글 |

- 배열의 함수(`sd.matmul`, `sd.sum` 등)도 희소 배열에 같은 꼴의 `TypeError`다(`A @ x`, `A.sum()`을 짚는다). 이것은 `sd.linalg`·배열의 인자 검사에 판별 하나를 더하는 일이고, 다른 입력의 결과는 바뀌지 않는다(골든 그대로).

---

## 4. 정확도 계약 (안)
- **정확한 연산**: 형식 변환, 전치, 인덱싱, 만들기, 패턴 연산, 원소 하나의 산술(희소끼리의 ±, `*`, 스칼라 곱, 원소별 함수)은 같은 원소의 배열 연산과 비트까지 같다. 원소별 함수의 정확도는 배열의 것(array.md §2.7의 표)이다.
- **합과 곱의 합**(SpMV, SpMM, SpGEMM, 리덕션): §2.6의 순서로 계산한 값이 참값과 |ŷ − y| ≤ γ_m·Σ|aᵢⱼxⱼ|(m = min(n, L) + ⌈log₂(n/L)⌉, γ_m = mε/(1 − mε))다. Arb의 정확한 내적과 견준다(상쇄가 큰 줄, 긴 줄, 아주 큰·작은 값). 순서 자체가 정의이므로 같은 순서의 다른 구현(scipy의 SpMV)과는 비트까지 같다.
- **직접 풀이**: 정규화한 역오차 ‖b − Ax̂‖∞ / (‖A‖∞‖x̂‖∞ + ‖b‖∞)가 c·ε 이하다. c는 SP4에서 시험 행렬로 재서 정한다(문턱 피벗이라 c는 행렬에 따라 자랄 수 있다; 넘는 행렬은 문서화한다). 작은 행렬(n ≤ 2000)은 Arb의 정확한 풀이와 견주어 앞오차가 κ∞(A)·c·ε 안인지 본다.
- **삼각 풀이**: 대입의 표준 역오차 |ΔA| ≤ γ_n|A|(Higham 2002, 8장)를 원소별로 확인한다.
- **반복법**: 결과는 "멈춤 판정을 만족한 점"이다(optimize §4와 같은 원리). `converged=True`면 반환 시점에 다시 계산한 참 잔차가 판정을 만족하는지 시험에서 확인한다(재귀 잔차와 참 잔차가 갈라지는 경우는 기록한다). 같은 문제에서 반복 수·상태를 scipy와 견준다.
- **고유값·특이값**: 각 쌍의 잔차 ‖Av − λv‖₂ ≤ tol·‖A‖₂ 꼴(ARPACK의 판정; tol = 0이면 ε), 고유벡터의 직교성 ‖VᴴV − I‖ ≤ c·k·ε. 대칭 문제의 고유값은 작은 행렬(n ≤ 500)에서 Arb의 고유값과 |λ̂ − λ| ≤ c·ε‖A‖₂다.
- **`expm`**: 밀집 `sd.linalg.expm`과 같은 꼴의 계약(linalg.md §11.5)이고 작은 행렬에서 Arb의 `arb_mat_exp`와 견준다. `expm_multiply`는 알고리즘의 허용오차(2⁻⁵³ 상대)에 조건수를 곱한 꼴이다.
- **`onenormest`**: 추정값은 참값 이하이고(하한), 시험 행렬에서 참값과 같은 비율을 기록한다(계약이 아니다). `norm`은 §2.6의 합의 계약, `ord=2`는 `svds`의 계약이다.
- scipy·R과의 차이는 계약이 아니라 정확도 표의 비교 패키지 열로 보인다.

---

## 5. 성능 (안)
이 초안에서 잰 scipy 1.18.1(1스레드)의 시간이다. 무작위 n × n, 행마다 평균 10개, float64.

| nnz | SpMV | SpMM (k = 16) | COO → CSR | CSR → CSC | `A.T @ x` | `A + A` | `A @ A` | `sum(axis=0)` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10⁵ | 0.06 ms | 0.46 | 1.30 | 0.27 | 0.08 | 0.29 | 7.0 | 0.10 |
| 10⁶ | 1.02 | 6.3 | 14.4 | 4.1 | 1.32 | 3.2 | 135 | 1.19 |
| 10⁷ | 11.7 | 344 | 215 | 104 | 14.8 | 31.1 | — | 13.2 |

풀이(2차원 라플라시안): n = 10⁴에서 `splu` 47 ms, `spsolve` 11 ms, `cg`(rtol 10⁻¹⁰) 18 ms, `eigsh(k=6, 'LM')` 226 ms; n = 9×10⁴에서 225, 193, 286 ms, 5.7 s.

**목표** (같은 기기, 1·10스레드, scipy 대비 배율)
- SpMV, `A.T @ x`, 축 리덕션: 1스레드에서 scipy와 같거나 빠르게(int64 색인의 대역 비용이 있어 0.8배까지를 받아들이고 넘으면 §2.2의 int32 안을 묻는다), 10스레드에서 nnz ≥ 10⁶이면 4배 이상(메모리 대역의 한계, array.md §10.6의 `f + g` 1.7배보다 SpMV는 무작위 읽기라 더 나뉜다).
- SpMM: scipy가 k 방향으로 벡터화하지 않아 느리다(10⁷에서 SpMV의 29배). 1스레드에서 3배 이상.
- 변환(COO → CSR, CSR → CSC, 중복 더하기): 1스레드에서 2배, 10스레드에서 5배 이상.
- SpGEMM, 이항 연산: 1스레드에서 같거나 빠르게, 10스레드에서 4배 이상.
- 직접 풀이: Accelerate LU/AMD는 이 꼴에서 SuperLU보다 3배 빠르다(73 대 225 ms). `spsolve`·`splu`는 2배 이상을 목표로 하고 행렬 꼴(2·3차원 격자, 회로, 무작위)마다 잰다.
- 반복법: 반복 하나의 비용이 SpMV + 내적이다. scipy의 Python 반복 대비 같은 반복 수에서 1스레드 1.5배, 10스레드 4배 이상.
- `eigsh`·`svds`: ARPACK(Fortran)과 같은 SpMV 수에서 같거나 빠르게, 10스레드에서 SpMV 몫만큼 빠르게.
- 작은 행렬(nnz ≤ 10³): 호출 하나의 고정 비용이 scipy(Python 층, 수 µs–수십 µs)보다 작게.
- 크기: nnz 10⁴, 10⁵, 10⁶, 10⁷(SpGEMM은 10⁶까지), 행 길이 3·10·100과 긴 행이 섞인 행렬, 풀이는 2·3차원 격자 n = 10⁴–10⁶. 사례는 벤치 세션이 `bench/speed/cases/sparse.py`에 넣는다(optimize O101과 같다; 이쪽은 목록을 넘긴다).

---

## 6. 층과 파일 (안)
- **층**: `sparse/`는 `array/`, `linalg/`(LAPACK 래퍼, 비용 규칙, pairwise 잎), `parallel/`, `accelerate/`, `base/` 위에 있다(ARCHITECTURE §1의 "아래로만"). 다른 영역은 `sparse/`를 모른다. 예외는 §3.10의 판별 하나(`bind/pyutil`의 `is_sparse` 꼴)와 나중의 그래프 영역이다.
- `src/stardust/sparse/` (하위 패키지 없음, SP3): `__init__.py`(공개 이름), `_base.py`(`sparray`: 공통 메서드, 연산자, 리덕션), `_coo.py`, `_compressed.py`(`csr_array`, `csc_array`), `_dok.py`, `_construct.py`(만들기 함수), `_index.py`(인덱싱과 대입), `_interop.py`(sd.ndarray·numpy·scipy), `_interface.py`(`LinearOperator`, `aslinearoperator`), `_direct.py`(`spsolve splu factorized inv spsolve_triangular is_sptriangular spbandwidth`), `_iterative.py`(`cg gmres minres bicgstab lsqr lsmr`), `_eigen.py`(`eigsh eigs svds`), `_matfuncs.py`(`expm expm_multiply onenormest norm`), `_result.py`(결과 클래스), `_common.py`(입력 정리와 메시지).
- `csrc/sparse/`: `format.hpp`(§2.2의 표현, 검사), `canonical.cpp`(정렬, 중복 더하기, 계수 정렬), `convert.cpp`(형식 사이, 밀집 사이), `spmv.cpp`(SpMV·SpMM·Aᵀx, §2.6의 순서), `spgemm.cpp`, `binop.cpp`(병합 이항 연산, 비교), `elementwise.cpp`(배열의 원소별 커널을 `data`에), `reduce.cpp`, `index.cpp`, `construct.cpp`, `dok.hpp`·`dok.cpp`(해시 맵), `triangular.cpp`, `direct.cpp`(Accelerate LU, 특이의 판정), `iterative.cpp`, `lanczos.cpp`(IRLM, 이중대각화), `arnoldi.cpp`(IRAM), `matfuncs.cpp`(expm, expm_multiply, onenormest), `parallel.hpp`(nnz 구간 분할, 기준값).
- `csrc/accelerate/sparse.hpp`: Sparse Solvers 래퍼(ARCHITECTURE §4.4: Accelerate 래퍼는 `accelerate/` 한곳). int64 → int32 행 색인 복사, `reportError` 콜백과 예외 변환, 순서·피벗 옵션 고정, `SparseCleanup`의 RAII.
- `csrc/linalg/lapack.hpp`에 더할 래퍼: `stev`·`steqr`(삼중대각 고유값), `hseqr`·`trevc`(헤센베르크), 필요하면 `bdsqr`(이중대각 SVD).
- `csrc/bind/sparse_bind.cpp`(`stardust._core.sparse`), `csrc/bind/module.cpp`에 하위 모듈 하나.
- `CMakeLists.txt`: 새 `.cpp`만 더한다. Accelerate는 이미 링크되어 있다. 블록(`^`)을 쓰는 `SparseIterate`는 쓰지 않으므로 `-fblocks`가 필요 없다(이 초안의 탐침에서 clang++이 블록을 기본으로 받는 것도 확인했다).
- `tests/sparse/`(마일스톤마다 `test_<묶음>.py`, `test_api_surface.py`가 `docs/api/sparse.md`를 읽는다), 시험 행렬 `tests/sparse/matrices.py`, 참값 `tests/sparse/ref/`, 대조 데이터 `tests/sparse/data/`, 도구 `tools/sparse_ref.py`(Arb), `tools/gen_sparse_scipy.py`, `tools/gen_sparse_r.py`, 기준선 `bench/sparse/sp0/`.

---

## 7. 참값과 검증 (계획)
SP0에서 만든다.

- **시험 행렬** (`tests/sparse/matrices.py`, 만드는 식과 고정 시드; 내려받지 않는다)
  - 격자: 1·2·3차원 라플라시안(Dirichlet·Neumann; scipy `LaplacianNd`와 같은 정의), 5점·7점 대류-확산(대각 우세로 조건수를 묶는다: 이 초안에서 대각 우세가 아닌 비대칭 계수(1.3, 0.7, 1.2, 0.9; 합 4.1 > 대각 4)는 n = 9×10⁴에서 조건수가 10²⁰을 넘어 Accelerate LU·QR 모두 잔차가 10¹¹–10²²였다. 시험 행렬은 조건수를 알고 만든다).
  - 무작위(밀도, 행 길이 분포, 긴 행 섞기), 띠, 블록 대각, 안장점(대칭 부정), 랭크 부족, 구조적 특이, 수치적 특이, 복소 에르미트, 빈 행·열, 0 × n, 저장된 0과 중복이 많은 COO.
  - SuiteSparse Matrix Collection의 실제 행렬은 내려받아야 한다. SP0에서 이름·크기·출처를 적어 묻는다(§9 SP0의 12번).
- **참값** (`tools/sparse_ref.py`, python-flint 0.9.0): 정확한 내적(Arb, 충분한 정밀도)으로 SpMV·SpGEMM·리덕션의 참값, 작은 행렬(n ≤ 2000)의 `arb_mat` 풀이, 작은 대칭 행렬의 고유값(`arb_mat`·`acb_mat`의 고유값), `arb_mat_exp`(expm). 큰 행렬은 잔차·역오차 계약(§4)으로 본다.
- **scipy 대조** (`tools/gen_sparse_scipy.py`): 반올림이 아닌 동작의 표(결과 형식·형·dtype, `nnz`와 0 지우기, NaN 규칙, 예외)를 함수마다 기록해 규칙 2를 확인한다. SpMV·SpMM은 지금 비트까지 같으므로(§2.6) 비트로 견준다(API.md §4의 규칙 4: 통과하는 동안 남긴다). 반복법·고유값은 반복 수·상태·결과를 기록한다. scipy 쪽 무작위(`eigsh`의 `rng`, `onenormest`)는 고정한다.
- **R 대조** (`tools/gen_sparse_r.py`, R 4.6.2): `Matrix` 1.7.6이 설치되어 있다: `sparseMatrix`, `%*%`, `crossprod`, `kronecker`, `rowSums`·`colSums`, `solve`(CHOLMOD·CSparse LU), `lu`, `qr`, `expm`, `norm`, `rcond`. `SparseM` 1.84.2도 있다. 고유값·특이값의 `RSpectra`·`irlba`는 없다. 필요하면 SP6에서 이름·CRAN 출처·크기를 적어 묻고 설치한다(htest H15와 같다). 지금은 설치하지 않는다.
- **Accelerate 탐침** (`bench/sparse/sp0/accelerate_probe.cpp`): §1.4의 표를 다시 재고 넓힌다: 행렬 꼴 열 가지 이상, 반복 20번, 여러 Python 스레드에서 동시에, 두 스레드 모드, 수치적 특이 행렬, int32 한계 근처, complex. LU/AMD의 결정성이 한 번이라도 깨지면 §3.6을 다시 묻는다.
- **정확도 체계** (`bench/accuracy`, 계획): `bench/accuracy/sparse.py`에 함수마다의 경우(Arb 참값 대 sd·scipy·R의 오차, 반복법·고유값은 잔차)를 넣는다. 비교 패키지 열은 scipy, R Matrix다.
- **속도 체계** (`bench/speed`, 계획): `bench/speed/cases/sparse.py`의 사례 목록(§5의 크기)을 벤치 세션에 넘긴다. 비교 대상은 scipy와 R Matrix, 직접 풀이와 반복법에는 Accelerate를 직접 부른 값(이쪽의 감싸기 비용을 보기 위해)이다.
- **골든**: 공개 함수의 결과를 `tools/golden.py`의 골든에 더한다(1·3·10스레드).

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 |
|---|---|
| SP0 준비 | API 문서 초안(`docs/api/sparse.md`: 모든 공개 이름의 시그니처, scipy → sd 이름의 대응표, 마일스톤별 질문), 시험 행렬, 참값 도구(Arb), scipy·R 대조 도구와 데이터, Accelerate 탐침(§7), int32·int64 색인의 SpMV 측정(§2.2), 속도 기준선, 정확도·속도 체계에 넣는 계획 |
| SP1 형식과 변환 | `coo_array csr_array csc_array dok_array sparray issparse`, 생성자, 표준 꼴(§2.3), 형식 변환, `sd.ndarray`·numpy·scipy와 주고받기(§2.5), `astype copy .T transpose matrix_transpose conj real imag reshape resize nonzero eliminate_zeros sum_duplicates`, repr |
| SP2 연산 | 산술과 비교(§3.2), 원소별 함수, `multiply maximum minimum power`, SpMV·SpMM·밀집 @ 희소·SpGEMM, 리덕션(§3.3) |
| SP3 인덱싱과 만들기 | 인덱싱과 대입, `diagonal setdiag trace`, `eye_array diags_array random_array kron kronsum hstack vstack block_array block_diag tril triu find` |
| SP4 직접 풀이 | `spsolve splu factorized inv spsolve_triangular is_sptriangular spbandwidth`, `LinearOperator aslinearoperator`, `sd.linalg`·배열의 `TypeError`(§3.10) |
| SP5 반복법 | `cg gmres minres bicgstab lsqr lsmr` |
| SP6 고유값과 특이값 | `eigsh eigs svds`, linalg 래퍼에 `stev steqr hseqr trevc` |
| SP7 행렬 함수와 노름 | `expm expm_multiply onenormest norm` |
| SP8 최적화와 일관성 점검 | 모든 공개 함수를 크기마다 scipy·R과 견주어 고치기, 메시지·시그니처의 영역 간 일관성, 코드 리뷰 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다. SP1–SP3은 차례로 간다(앞의 것에 기댄다). SP4가 `LinearOperator`를 만들므로 SP5–SP7은 그 뒤이고, 셋끼리는 순서를 바꿀 수 있다(SP7의 `norm(ord=2)`는 SP6의 `svds`를 쓴다). 기존 영역의 결과는 비트까지 같아야 한다(§3.10의 판별은 희소 입력에서만 동작한다).

**나중 후보** (첫 판 밖): BSR·DIA(§2.1), n차원 COO, `save_npz`·`load_npz`, `spilu`, 결정적인 희소 Cholesky와 LDLᵀ(§3.6의 대안 1; 그러면 `cholesky` 같은 scipy에 없는 함수를 둘지도 묻는다), 희소 QR로 하는 직접 최소제곱, `lobpcg`, `bicg cgs qmr tfqmr lgmres gcrotmk`, `matrix_power`, `funm_multiply_krylov`, `LaplacianNd`. 그래프 알고리즘은 따로 된 영역이다(§1.2).

---

## 9. 마일스톤을 시작하기 전에 물을 것
추천을 첫째에 둔다.

- **SP0** (설계 전체에 걸친 것)
  1. BSR·DIA: 첫 판에 넣지 않고 나중 후보로(추천; `eye_array`·`diags_array`·`kron`의 `format=None`은 CSR, `[scipy와 다름]`) / DIA만 넣는다(격자 행렬의 SpMV) / 둘 다 넣는다.
  2. 만들기용 형식: DOK만(C++ 해시 맵; 추천) / DOK와 LIL / 둘 다 없이 COO만.
  3. 색인 dtype: 안팎 모두 int64(추천; SpMV 비용을 SP0에서 재고 1.3배를 넘으면 다시 묻는다) / 안은 int32(넘치면 int64), 밖에는 int64 복사.
  4. 값 dtype: 배열의 넷 모두 저장과 연산에, 선형대수는 float64·complex128(complex를 직접 풀이·반복법·고유값 모두에; 추천) / complex는 저장과 연산만, 선형대수는 float64만 / float64만.
  5. 표준 꼴: CSR·CSC는 언제나 표준(추천, `[scipy와 다름]`) / scipy처럼 비표준을 허용하고 필요할 때 정리.
  6. 연산이 만든 0: 연산마다 scipy의 규칙(추천, §2.4) / 패턴만으로 정하는 규칙(값을 보지 않는다; `A - A`의 nnz가 A의 nnz).
  7. 강한 0과 `A / 0`: 암시적 0은 산술에 들지 않고(scipy와 같다), `A / 0`은 예외 없이 저장된 원소만 ±inf·NaN(추천, `[scipy와 다름]`) / scipy처럼 `A / 0`은 예외(종류는 `ValueError`).
  8. 차원과 배치: 1·2차원(CSC는 2차원), 배치 없음(추천) / 2차원만(`A[i]`는 (1, n)) / n차원 COO까지.
  9. 이름: 평평한 이름공간 안의 충돌은 없다(§1.2). 뜻이 겹치는 곳의 추천: `norm`은 사용자가 정한 scipy의 이름 그대로 두고 `ord`의 값과 메시지를 `sd.linalg.matrix_norm`·`vector_norm`과 맞춘다(추천) / `sd.sparse`에도 `matrix_norm`·`vector_norm`을 두고 `norm`은 두지 않는다(DESIGN §1.4의 "같은 개념은 같은 이름"을 linalg 쪽으로) / 셋 다. `inv`·`expm`·`kron`·`tril`·`triu`·`matrix_transpose`는 기존 영역과 같은 이름 그대로(추천). `sd.sparse.linalg` 별칭은 두지 않는다(추천).
  10. 합의 순서(§2.6): 줄은 L = 4096까지 순차(FMA), 넘으면 L 조각의 pairwise; 전체 합은 C 순서의 pairwise; 형식과 무관한 비트(추천: SpMV·축 합이 scipy와 비트까지 같다) / 줄도 처음부터 pairwise(긴 줄에 더 정확하고 scipy와 다르다) / 형식마다 저장 순서(CSC의 전체 합이 CSR과 끝자리가 다를 수 있다).
  11. 반복법: 이 라이브러리가 scipy의 알고리즘과 멈춤 판정으로 짠다(추천, §3.7; SP4의 "알맞은 반복법은 Accelerate"를 바꾼다) / CG·GMRES·LSMR은 Accelerate의 `SparseIterate`에 이쪽의 판정을 붙인다(반복 수가 scipy와 다를 수 있다).
  12. 시험 행렬: 만든 행렬만으로 시작하고, SuiteSparse의 실제 행렬 몇 개(회로, 구조 역학, 최적화의 KKT; 이름·크기·출처를 적어)는 SP0에서 내려받아도 되는지 묻는다(추천) / 만든 행렬만.
  13. Accelerate의 대칭 분해(§3.6): 첫 판에 쓰지 않고 LU로(추천) / 결정적인 Cholesky를 이 라이브러리가 짜서 첫 판에 넣는다(마일스톤 하나가 는다) / 실행마다 다른 비트를 문서화하고 Accelerate를 쓴다(API.md §3의 예외).
- **SP1**
  1. 클래스: Python 클래스에 `sd.ndarray` 필드(추천, frame의 `Series`와 같은 예외) / nanobind C++ 클래스.
  2. `indices`·`indptr`·`coords`는 쓰기 금지, `data`는 쓸 수 있다(추천) / 모두 쓸 수 있다(바꾸면 사용자 책임, scipy와 같다).
  3. scipy와 주고받기: 들여올 때 언제나 복사, 내보내기는 `to_scipy(*, copy=None)`가 뷰(추천) / `sd.sparse.asarray(scipy_obj, *, copy=None)` 꼴의 함수 하나로.
  4. 별칭: `toarray`만(`todense` 없음), `conj`만(`conjugate` 없음), `@`만(`dot` 없음)(추천) / scipy처럼 모두.
  5. DOK의 `dict` 메서드(`keys`, `items`, `get`, `pop`, `update` …): 두지 않는다(추천; 인덱싱과 `tocoo()`로) / scipy처럼 둔다.
  6. `np.asarray(A)`: `TypeError`(추천) / scipy처럼 0차원 object 배열.
- **SP2**
  1. 밀집 피연산자로 numpy 배열: `TypeError`(추천, API.md §5) / 받는다(scipy 사용자의 코드가 그대로 돈다).
  2. 원소별 함수: scipy처럼 메서드(`A.sin()`), `sd.sin(A)`는 메서드를 짚는 `TypeError`(추천) / `sd.sin(A)`도 받는다(f(0) = 0인 함수만).
  3. 축 리덕션의 결과: 언제나 밀집 `sd.ndarray`(추천, `[scipy와 다름]`: scipy의 `max(axis=)`는 COO) / scipy 그대로.
  4. `A * 밀집`의 형식: A의 형식(추천) / scipy처럼 COO.
  5. `A + c`(c ≠ 0)·`A ** 0`: `ValueError`(추천) / 밀집 결과를 돌려준다.
- **SP3**
  1. 만들기 함수의 이름: scipy 배열 꼴의 `eye_array`·`diags_array`·`random_array`·`block_array`(추천: 옛 이름 `eye`·`diags`·`random`은 scipy에서 `spmatrix`를 돌려주고, `random`은 `sd.random`과 헷갈린다) / 짧은 이름 `eye`·`diags`·`random`·`block`(sd의 `eye`와 같은 꼴).
  2. `random_array`의 `rng`: 꼭 받는다(추천, optimize O31) / 없으면 고정 시드.
  3. CSR·CSC의 새 자리 대입: 허용하고 O(nnz)(추천, scipy와 같고 경고만 없다) / `ValueError`로 막고 DOK를 짚는다.
- **SP4**
  1. LU의 순서: AMD 고정(추천) / `ordering='amd' | 'metis'` 옵션(둘 다 결정적).
  2. `splu`의 결과: 내보내지 않는 분해 클래스(이름 `SparseLU`, `solve()`·`shape`·`nnz`; `L`·`U`·`perm_r`·`perm_c` 없음)(추천) / `L`·`U`가 꼭 필요하면 이 라이브러리의 LU(나중 후보)를 기다린다.
  3. scipy의 옵션(`permc_spec`, `diag_pivot_thresh`, `use_umfpack` 등): 두지 않는다(추천) / 받되 `ValueError`로 다른 바탕임을 알린다.
  4. 특이의 판정: 구조적 특이 + 유한하지 않은 풀이를 `LinAlgError`로(추천, §3.6) / 구조적 특이만.
  5. 반복 다듬기: 잰 역오차가 계약(§4)을 넘는 행렬이 있으면 한 번의 다듬기를 넣는다(추천) / 넣지 않는다.
- **SP5**
  1. 결과: scipy의 튜플 개수로 풀리는 읽기 전용 결과 객체(`x`, `converged`, `status`, `n_iter`, …; 추천) / scipy처럼 튜플.
  2. 이름: `max_iter`(추천, optimize O14) / scipy의 `maxiter`·`iter_lim`.
  3. `callback`: 두지 않는다(추천, optimize O37·O54) / 둔다.
  4. `gmres`의 기본 `restart`와 `maxiter`의 뜻: scipy 1.18.1 그대로(추천).
- **SP6**
  1. 알고리즘: ARPACK을 계산 순서까지 옮긴다(추천) / Krylov–Schur.
  2. `v0=None`: 정해진 시작 벡터(고정 키의 Philox), `rng=`로 바꿀 수 있다(추천) / `v0`나 `rng`를 꼭 받는다.
  3. `svds`: Golub–Kahan–Lanczos 이중대각화 하나, `solver=` 없음(추천, `[scipy와 다름]`) / scipy처럼 `'arpack'`(AᵀA) 기본.
  4. `eigsh`의 `mode='buckling'`·`'cayley'`: 넣는다(추천, scipy에 있다) / `'normal'`만.
  5. 수렴 실패: 결과 객체의 상태(찾은 쌍만 돌려주고 `converged=False`)(추천) / `LinAlgError`.
  6. R의 `RSpectra`·`irlba`: 이 마일스톤에서 묻고 설치한다(추천).
- **SP7**
  1. `onenormest`의 무작위 열: 고정 키로 결정적(추천) / `rng`를 받는다.
  2. `expm`의 결과 형식: 입력의 형식(추천) / scipy처럼 CSC.
- **SP8**: 범위는 optimize O97처럼 모든 공개 함수를 크기마다 견주는 것(추천). 코드 리뷰를 사용자가 따로 하는지 그때 묻는다.
