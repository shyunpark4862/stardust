# 희소 영역 설계 (sd.sparse)

> 상태: **설계 초안** (2026-11-21). 큰 결정(§1.1 SP1–SP5)과 SP0·SP1의 결정(SP6–SP34)은 사용자가 정했다. 나머지는 안이다. SP0(준비)은 완료·승인, SP1(형식과 변환)은 완료·승인, SP2–SP8은 시작 전이다. 시작 전에 정할 것은 §9에 있다.

`sd.sparse`(희소 배열과 희소 선형대수)의 확정 사항과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처의 안은 SP0에서 `docs/api/sparse.md`로 쓴다.
- 비교 대상은 scipy 1.18.1의 `scipy.sparse`와 `scipy.sparse.linalg`, R 4.6.2의 `Matrix` 1.7.6(설치되어 있다)이다. 바닥은 Accelerate의 Sparse Solvers(`Solve.h`), 배열(`csrc/array/`), 스레드풀(`csrc/parallel/`), linalg의 LAPACK 래퍼와 비용 규칙(`csrc/linalg/`)이다.
- 이 문서의 잰 값은 Apple M4 Pro, macOS 26.6.2, SDK 26.5에서 본 것이다. §1.4·§2.2·§5의 수는 SP0에서 다시 쟀다(`tools/sparse_probe.py`, 결과는 §7).
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
| SP6 BSR·DIA | 첫 판에 넣지 않는다(나중 후보). `eye_array`, `diags_array`, `kron`의 `format=None`은 CSR이다(`[scipy와 다름]`: scipy는 DIA·BSR) (사용자 결정 2026-11-20) |
| SP7 만들기용 형식 | DOK 하나만 둔다. LIL은 없다 (사용자 결정 2026-11-20) |
| SP8 색인 dtype | 안팎 모두 int64다. SP0에서 SpMV를 int32·int64로 재서 int64가 1.3배를 넘게 느리면 멈추고 다시 묻는다(§2.2, §7.5; 문턱의 결과는 SP19) (사용자 결정 2026-11-20) |
| SP9 값 dtype | `bool`, `int64`, `float64`, `complex128`을 저장하고 연산한다. 선형대수는 float64와 complex128이다(직접 풀이·반복법·고유값 모두 complex를 받는다) (사용자 결정 2026-11-20) |
| SP10 표준 꼴 | CSR·CSC는 언제나 표준 꼴(정렬, 중복 없음)이다(`[scipy와 다름]`, §2.3) (사용자 결정 2026-11-20) |
| SP11 연산이 만든 0 | 연산이 만든 0을 지우는지는 연산마다 scipy의 규칙을 따른다(§2.4, §3.2) (사용자 결정 2026-11-20) |
| SP12 강한 0 | 암시적 0은 강한 0이다. `A / 0`은 예외 없이 저장된 원소만 ±inf·NaN이 된다(`[scipy와 다름]`: scipy는 `ZeroDivisionError`) (사용자 결정 2026-11-20) |
| SP13 차원 | 1차원과 2차원만(CSC는 2차원), 배치는 없다 (사용자 결정 2026-11-20) |
| SP14 이름과 `norm` | 평평한 이름공간 안의 충돌은 없다. `inv`·`expm`·`kron`·`tril`·`triu`·`matrix_transpose`는 기존 영역과 같은 이름이고, `sd.sparse.linalg` 별칭은 없다(§1.2). `norm`은 scipy의 이름 그대로 두고 `ord`의 규칙과 메시지를 `sd.linalg`의 `matrix_norm`·`vector_norm`과 맞춘다(§3.9) (사용자 결정 2026-11-20) |
| SP15 합의 순서 | 줄은 L = 4096까지 FMA 순차 합, 더 긴 줄은 L 조각의 순차 합을 pairwise로 합친다. 전체 합은 C 순서의 pairwise다. 비트는 형식(CSR·CSC·COO)과 무관하다(§2.6) (사용자 결정 2026-11-20) |
| SP16 반복법 | `cg gmres minres bicgstab lsqr lsmr`은 이 라이브러리가 scipy의 알고리즘과 멈춤 판정으로 짠다. Accelerate의 반복법은 쓰지 않는다(SP4의 "알맞은 반복법은 Accelerate"를 바꾼다, §3.7) (사용자 결정 2026-11-20) |
| SP17 실제 행렬 | SuiteSparse 컬렉션의 실제 행렬 몇 개를 시험에 써도 된다. 내려받기 전에 목록(이름, 주소, 크기)을 보이고 승인을 받는다(§7.2의 목록을 SP20으로 승인) (사용자 결정 2026-11-20) |
| SP18 직접 풀이 | 첫 판의 직접 풀이는 모두 Accelerate의 LU(AMD 순서, 결정적)다. Cholesky·LDLᵀ(기본 스레드 모드에서 실행마다 비트가 다르다)는 쓰지 않는다(SP4의 "직접 분해(Cholesky, LDLᵀ, QR)"를 바꾼다). 결정적인 대칭 분해를 이 라이브러리가 짜는 것은 나중 후보다(§3.6) (사용자 결정 2026-11-20) |
| SP19 색인 dtype의 문턱 | int64 색인을 그대로 둔다. 1스레드 SpMV는 1.3배 아래(1.02–1.16배)이고, 10스레드의 2차원 라플라시안 n = 10⁶만 int32보다 1.33–1.39배 느리지만(0.33 대 0.44 ms) 그래도 scipy 1스레드(1.7 ms)보다 4배 빠르다(§7.5) (사용자 결정 2026-11-21) |
| SP20 SuiteSparse 행렬 | §7.2의 여덟 개(약 14 MB)만 내려받는다: `HB/bcsstk14`, `Boeing/msc10848`, `HB/sherman3`, `Goodwin/goodwin`, `Hamm/memplus`, `GHS_indef/cvxqp3`, `HB/well1850`, `HB/young1c`. 그 밖은 받지 않는다. 파일은 버전 관리 밖(`build/sparse/suitesparse/`)에 두고 저장소에는 받는 도구(`tools/sparse_suitesparse.py`: 주소, 받은 압축 파일의 sha256, `--jobs`)만 둔다(optimize O13의 Fortran 원본과 같다). 그 행렬을 쓰는 테스트는 external 등급이고 파일이 없으면 까닭을 적고 건너뛴다. 거기서 나온 참값·기록은 작으면 `tests/sparse/ref`·`data`에 `_provenance`와 함께 둔다 (사용자 결정 2026-11-21) |
| SP21 클래스 | 희소 배열은 `sd.ndarray` 필드를 가진 Python 클래스다(frame의 `Series`와 같은 예외, ARCHITECTURE §3). 계산은 네이티브 커널 한 번 호출이다. DOK만 필드가 네이티브 해시 맵이다(§2.1) (사용자 결정 2026-11-21) |
| SP22 쓰기 금지 색인 | `indices`, `indptr`, `coords`는 쓰기 금지 배열이다(바꾸면 표준 꼴이 깨진다). `data`는 쓸 수 있다(`A.data *= 2`) (사용자 결정 2026-11-21) |
| SP23 scipy와 주고받기 | scipy의 희소 배열·행렬을 받을 때는 언제나 복사한다. `A.to_scipy()`는 복사 없는 뷰다(색인 뷰는 읽기 전용, `copy=True`면 복사; DOK는 언제나 복사) (사용자 결정 2026-11-21) |
| SP24 별칭 없음 | `toarray`, `conj`, `@`만 둔다(`todense`, `conjugate`, `dot` 없음). COO에는 `coords`만 있다(`row`, `col` 없음) (사용자 결정 2026-11-21) |
| SP25 DOK의 메서드 | Python `dict`의 메서드(`keys`, `items`, `get`, `pop`, `update` …)를 두지 않는다. 원소는 `A[i, j]`로, 전체는 `tocoo()`로 읽는다 (사용자 결정 2026-11-21) |
| SP26 밀집으로 몰래 바뀌지 않음 | `np.asarray(A)`는 `A.toarray()`를 짚는 `TypeError`다(scipy는 0차원 object 배열) (사용자 결정 2026-11-21) |
| SP27 `copy` | `copy=`는 `sd.ndarray`의 규칙이다: None은 필요할 때만, True는 언제나, False는 복사가 필요하면 `ValueError`. 생성자, `tocoo`·`tocsr`·`tocsc`·`todok`·`asformat`·`reshape`·`to_scipy`의 기본은 None이다(`[scipy와 다름]`: scipy의 `copy=False`는 "복사하지 않을 수 있으면"의 뜻이다) (사용자 결정 2026-11-21) |
| SP28 데이터 인자의 이름 | scipy의 이름이다(`A`, `b`, `x`; 메시지는 `'A'`) (사용자 결정 2026-11-21) |
| SP29 변환의 `copy` | SP27을 변환 메서드까지 넓힌다: `tocoo`·`tocsr`·`tocsc`·`todok`·`asformat`·`reshape`·`to_scipy`의 기본은 `copy=None`이다(`[scipy와 다름]`: scipy는 `copy=False`) (사용자 결정 2026-11-21) |
| SP30 색인의 복사 | 받은 색인 배열(`indices`, `indptr`, `coords`)은 `copy`와 상관없이 언제나 복사하고 검사한다(받은 배열을 나중에 고쳐도 표준 꼴이 깨지지 않는다; §2.5) (사용자 결정 2026-11-21) |
| SP31 1차원 전치 | 1차원의 `.T`·`.mT`·`matrix_transpose`는 `sd.ndarray`처럼 `ValueError`다. `transpose()`는 자신이다(`[scipy와 다름]`: scipy의 1차원 `.T`는 자신; §3.5) (사용자 결정 2026-11-21) |
| SP32 `has_canonical_format` | 저장된 원소의 실제 상태(행 우선 정렬, 중복 없음)다. 처음 물을 때 알아내 둔다(`[scipy와 다름]`: scipy는 확인 없는 표시; §2.3) (사용자 결정 2026-11-21) |
| SP33 1스레드 변환 속도 | nnz 10⁵–10⁶의 1스레드 COO → CSR·CSR → CSC(scipy의 1.1–1.5배, 목표 2배)는 SP8의 고칠 목록에 넣는다(§5, §8) (사용자 결정 2026-11-21) |
| SP34 다시 재기 | 다른 프로세스가 바빴던 속도 사례(`bench/sparse/speed.md`의 † 줄)는 기계가 한가할 때 `--rerun-busy`로 다시 잰다. 나중 일이다(§7.6) (사용자 결정 2026-11-21) |

### 1.2 범위 (안)
마일스톤은 §8이다. "SPn"은 그 마일스톤에서 만든다는 뜻이고 "나중"은 첫 판에 넣지 않는 후보다.

**형식과 만들기 (scipy.sparse → sd.sparse)**
| scipy.sparse | sd.sparse | 마일스톤 |
|---|---|---|
| `coo_array`, `csr_array`, `csc_array` | 같은 이름 | SP1 |
| `dok_array` | 같은 이름 (만들기용, §2.1) | SP1 |
| `lil_array` | 없음 (SP7) | — |
| `bsr_array`, `dia_array` | 없음, 나중 (SP6, §2.1) | 나중 |
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
| `cg`, `gmres`, `minres`, `bicgstab`, `lsqr`, `lsmr` | 같은 이름 | 이 라이브러리의 커널 (SP16, §3.7) | SP5 |
| `eigsh`, `eigs`, `svds` | 같은 이름 | 이 라이브러리의 Lanczos·Arnoldi (§3.8) | SP6 |
| `expm`, `expm_multiply`, `norm`, `onenormest` | 같은 이름 | 이 라이브러리의 커널 (§3.9) | SP7 |
| `SuperLU` (`splu`의 결과 형) | 다른 이름 (안, §9 SP4의 2번): 바탕이 SuperLU가 아니다 | | SP4 |
| `is_sptriangular`, `spbandwidth` | 같은 이름 (싼 도구, 안) | | SP4 |
| `ArpackError`, `ArpackNoConvergence` | 없음: 수렴하지 못하면 결과의 상태 필드다(§3.8) | | — |
| `spilu`, `lobpcg`, `bicg`, `cgs`, `qmr`, `tfqmr`, `lgmres`, `gcrotmk`, `matrix_power`, `funm_multiply_krylov`, `LaplacianNd` | 나중 | | 나중 |
| `use_solver`, `MatrixRankWarning`, 하위 모듈(`dsolve`, `isolve`, `eigen`, `interface`, `matfuncs`) | 없음 | | — |

- **평평한 이름공간의 충돌**(SP3): scipy.sparse와 scipy.sparse.linalg의 공개 이름은 서로 겹치지 않는다(scipy 1.18.1에서 `dir()`로 확인). 겹치는 것은 이름공간 밖의 뜻이고 SP14로 정했다.
  - `sd.linalg`와 같은 이름: `inv`, `expm`, `kron`(그리고 나중의 `matrix_power`). 같은 개념이라 같은 이름이 맞다(DESIGN §1.4).
  - `sd`의 최상위와 같은 이름: `tril`, `triu`, `matrix_transpose`(그리고 두지 않는 `eye`, `permute_dims`, `expand_dims`). 같은 개념이다.
  - `norm`: `sd.linalg`에는 `norm`이 없고 Array API의 `matrix_norm`·`vector_norm`이 있다. 같은 개념의 다른 이름이 된다.
  - `random`: scipy.sparse의 옛 함수 이름이고 `sd.random`(모듈)과 헷갈린다. 배열 꼴 이름 `random_array`를 쓰면 생기지 않는다.
  - scipy 사용자가 쓰던 `sd.sparse.linalg.spsolve`는 `AttributeError`다. 별칭을 두지 않는다(별칭 없음 규칙).
- **그래프 알고리즘은 이 영역 밖이다**(SP1). scipy의 `csgraph`(연결 성분, 최단 경로, 너비·깊이 우선 순회, 최소 신장 트리, 최대 흐름, 이분 매칭, `reverse_cuthill_mckee`, `structural_rank`, `laplacian` 등)는 나중의 따로 된 최상위 영역이다(이름은 나중에 정한다, 예: `sd.graph`). 그 영역은 희소 배열(과 밀집 배열)을 입력으로 받는다. 이 영역은 그것을 미리 막지 않도록 희소 배열의 C++ 표현(§2.2)을 다른 영역이 읽을 수 있게 둔다. 분해에 필요한 순서 매기기(AMD, nested dissection)는 Accelerate가 안에서 한다(§3.6).

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다. 별칭이 없다(scipy의 `todense`·`toarray`, `conj`·`conjugate`, `dot`·`@` 같은 두 이름은 하나만 둔다; SP24).
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
- LU에 COLAMD 순서를 주면 `reportError`에 "not supported for this factorization type"을 보내지만 **status는 0이고 다른 순서로 분해한다**. 그래서 래퍼는 콜백이 불렸는지를 status와 따로 보고 메시지가 있으면 실패로 다룬다(`csrc/accelerate/sparse.hpp`). LU의 순서는 AMD, Metis, MTMetis, 사용자다.
- 0 × 0 행렬은 받지 않는다(`rowCount must be > 0`, 매개변수 오류). 직사각 행렬의 LU도 매개변수 오류다(status −4).
- 특이 판정: 구조적으로 특이하면 `SparseMatrixIsSingular`다(LU는 메시지도 보내고, QR은 보내지 않는다). **수치적으로 특이하면 성공(status 0)을 돌려준다**: 2 × 2 [[1, 2], [2, 4]]의 LU는 status 0이고, `zeroTolerance = 0`이면 풀이가 ±inf, `zeroTolerance = 10⁻¹⁴`이면 틀린 유한한 값(x = (1, −0); Ax ≠ b)이다. QR도 status 0에 x ≈ 2.6×10¹⁵다. 반올림 때문에 0 피벗이 정확히 0이 되지 않는 정확히 특이한 행렬(Neumann 라플라시안, 같은 열 둘)은 `zeroTolerance = 0`에서도 status 0과 크기 1/ε의 유한한 해다(SP0, §7). 그래서 특이의 판정은 이쪽이 한다(§3.6).
- 반복법의 멈춤 판정은 ‖b − Ax‖₂ < rtol·‖b − Ax₀‖₂ + atol이고(처음 잔차에 대한 상대, scipy는 ‖b‖에 대한 상대), rtol의 기본은 √ε다. `SparseSolve`는 반복 수를 돌려주지 않는다(상태 넷과 `reportStatus` 글뿐).

**결정성** (SP0에서 잼, `tools/sparse_probe.py det·conc`: 행렬 꼴 열 가지(2·3차원 라플라시안, 대류-확산, 대각 우세 무작위, 띠, 밀집 블록 대각, 안장점, Neumann + 10⁻⁸I, 직사각, 복소), 크기 n ≈ 3.6×10³, 9×10⁴, 4.9×10⁵; 같은 프로세스에서 같은 입력을 20번 분해하고 풀어 결과의 바이트를 견줌. 칸은 "다른 결과가 나온 꼴 / 잰 꼴")

| 분해 / 순서 | 기본 스레드 모드 | `VECLIB_MAXIMUM_THREADS=1` | 4개의 스레드에서 동시에 (n = 9×10⁴) |
|---|---|---|---|
| **LU, LUTPP, LUSPP, LUUnpivoted / AMD** | **같다** (0/9, 0/9, 0/3; complex 포함) | 같다 | **같다** (20번 모두 직렬과 같다, 두 모드) |
| LU 셋 / Metis | 작은 꼴 셋에서 20번 중 2가지, 나머지는 같다 | 같다 | **20번이 20가지**(1스레드 모드도; 잔차는 정상) |
| LU 넷 / MTMetis(= LU의 기본 순서 `SparseOrderDefault`) | **다르다** (모든 꼴, 20번 중 4–20가지; 노름 상대 10⁻¹⁶–2×10⁻¹³) | 같다 | **프로세스가 죽는다**(SIGSEGV·SIGBUS; 1스레드 모드는 "Unkown iptype of 0"을 찍고 죽는다) |
| Cholesky, LDLᵀ 넷 / AMD·Metis·MTMetis | **다르다** (양정치 꼴 모두 20가지; 노름 상대 10⁻¹⁵–10⁻¹³) | 같다 | 다르다 (1스레드 모드에서는 같다) |
| QR, CholeskyAtA / COLAMD·AMD·Metis | 같다 | 같다 | QR/COLAMD 같다 |
| 같은 인수로 다시 풀기(`SparseSolve`) | 같다 (모든 분해) | 같다 | — |

- **LU/AMD는 한 번도 깨지지 않았다**: 아홉 꼴 × 두 크기 × 네 LU 변형, 큰 크기 셋, complex, 두 모드, 동시 호출. sd는 분해를 GIL을 놓고 부르므로 사용자의 여러 Python 스레드가 동시에 풀 수 있다. AMD만이 이 조건에서 결정적이고 안전하다(SP18).
- `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)`는 희소 Cholesky를 결정적으로 만들지 않는다(n = 9×10⁴, 10⁶ 모두 5번이 5가지). 희소 풀이의 스레드는 환경 변수 `VECLIB_MAXIMUM_THREADS`만 따른다. 이 라이브러리는 그 값을 정하지도 바꾸지도 않는다(API.md §3).
- 두 스레드 모드 사이에서는 LU도 비트가 다르다. BLAS·LAPACK과 같은 규칙이다(API.md §3).
- 우변이 여럿이면 Accelerate가 열을 함께 풀어, 열 하나의 결과가 혼자 푼 것과 반올림 크기만큼 다르다(`tests/sparse/test_accelerate.py`; 결정적이다).
- 안장점에서 LDLᵀ(기본 = TPP)·LUSPP·LUUnpivoted는 status 0에 NaN 해를, LDLᵀSBK는 잔차 1.9×10⁶의 해를 냈다. LU/LUTPP만 정상이다(잔차 10⁻¹¹). status 0이 성공을 뜻하지 않는다.
- 기본 모드의 MTMetis는 떨어진 성분이 많은 행렬(블록 대각 n = 9×10⁴)에서 분해 하나가 98–106초다(AMD 0.09초).
- 속도: §5의 직접 풀이 표(SP0에서 잼). 2차원 라플라시안 n = 9×10⁴에서 LU/AMD 92 ms, Cholesky/AMD 21 ms, scipy `splu` 139 ms다.

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

### 2.1 형식
- **CSR, CSC**: 계산의 형식이다. `data`(값), `indices`(줄 안의 색인), `indptr`(줄의 시작, 길이 줄 수 + 1). CSR은 행, CSC는 열이 줄이다.
- **COO**: 만들기와 주고받기의 형식이다. `data`, `coords`(차원마다 색인 배열의 tuple; scipy의 `row`·`col`은 그 둘의 이름이다). 중복과 정렬되지 않은 원소를 가질 수 있다(§2.3).
- **DOK**: 하나씩 넣는 만들기용 형식이다. 키 (i, j) → 값의 해시 맵을 C++에 둔다(`csrc/sparse/dok.cpp`: 열린 주소법, 선형 탐사; scipy는 Python `dict`의 하위 클래스다). `A[i, j] = v`와 `A[i, j]`가 O(1)이고, 계산할 때는 CSR·CSC로 바꾼다. Python `dict`의 메서드는 두지 않는다(SP25). 0을 넣으면 그 키를 지운다(scipy와 같다). 다른 형식에서 바꿀 때는 저장된 0도 남는다(scipy와 같다).
- **LIL은 두지 않는다**(SP7). scipy의 LIL은 행마다의 Python 리스트이고 행 단위 대입이 쓰임새다. DOK와 COO 꼴의 만들기(`coo_array((data, (row, col)))`), `vstack`이 같은 일을 하고 더 빠르다(§9 SP0의 2번).
- **BSR, DIA는 첫 판에 넣지 않는다**(SP6; 아래는 그 조사).
  - BSR(블록 CSR): 자유도가 여럿인 유한요소 행렬처럼 작은 밀집 블록이 모인 행렬에서 색인이 적고 SpMV가 레지스터 블록이 된다. Accelerate의 `SparseMatrix`가 `blockSize`를 그대로 받으므로 분해에도 이득이 있을 수 있다. 하지만 scipy의 BSR은 산술·인덱싱 대부분을 CSR로 바꿔서 하고, 블록 크기가 맞는 행렬에서만 이득이다. 나중에 잰 이득과 함께 묻는다.
  - DIA(대각 저장): 격자 스텐실처럼 몇 개의 대각만 있는 행렬에서 SpMV가 연속 메모리의 SIMD 루프가 된다. scipy에서는 `eye_array`·`diags_array`의 기본 형식이 DIA이고 `kron`의 기본은 BSR이다. 넣지 않으면 그 기본 형식이 CSR이 된다(`[scipy와 다름]`; 값은 같고 `.format`만 다르다).
  - 둘 다 나중 후보로 적는다. 넣으면 형식 하나마다 변환·연산·인덱싱이 따라 붙으므로, 넣는 기준은 같은 행렬에서 CSR 대비 2배(linalg §1.3의 크기 규칙과 같은 문턱)다.
- **차원**: CSR, COO, DOK는 1차원과 2차원, CSC는 2차원이다(scipy 1.18.1과 같다). scipy의 n차원 COO는 나중 후보다. 1차원은 `A[i]`(행 하나)와 희소 벡터를 위해 둔다(SP13).
- **배치는 없다**(SP13). 희소 선형대수의 함수는 행렬 하나를 받는다. 우변은 (n,)이나 (n, k)이고 k개의 열을 함께 푼다. 밀집 linalg의 `(..., M, N)` 배치와 다르다: 희소 행렬의 묶음은 행렬마다 패턴이 달라 한 배열에 담을 꼴이 없다(블록 대각 `block_diag`로 하나로 만들 수 있다).

### 2.2 색인과 값의 dtype
- **값의 dtype**: 배열의 네 dtype(`bool`, `int64`, `float64`, `complex128`)이다(API.md §1). 승격 규칙도 배열과 같다(`bool < int64 < float64 < complex128`). int64·bool은 구조와 개수(인접 행렬, 패턴)와 정수 산술을 위해 둔다. 받는 값은 `sd.asarray`의 규칙으로 바꾼다(docs/api/array.md §3.3): numpy의 정보 손실 없는 dtype(int8–int32, uint8–uint32, float16·float32, complex64)은 넓히고, uint64·문자열·object는 `TypeError`다. `[scipy와 다름]`: scipy는 float32 등을 그대로 둔다.
- **선형대수의 dtype**: float64와 complex128로 계산한다. bool·int64는 float64로 바꾼다(sd.linalg와 같다, DESIGN §9.6). complex는 Accelerate LU가 macOS 15.5부터 받으므로 직접 풀이도 complex를 한다(SP9).
- **색인의 dtype은 언제나 int64다**(SP8). `indices`, `indptr`, `coords`는 int64 `sd.ndarray`다.
  - 이유: 이 라이브러리의 정수 dtype은 int64 하나다(API.md §1). scipy처럼 값에 따라 int32와 int64를 고르면(`get_index_dtype`) 같은 연산의 결과 dtype이 데이터 크기에 따라 달라지고, 사용자에게 보이는 색인 배열의 dtype을 sd가 나타낼 수 없다.
  - 비용: SpMV가 원소마다 읽는 바이트가 12(값 8 + int32 색인 4)에서 16으로 늘어난다. 메모리 대역에 묶인 SpMV는 최대 약 1.33배 느려질 수 있다. SP0에서 같은 커널을 int32·int64로 쟀다: 1스레드 1.02–1.16배, 10스레드의 연속 행렬 하나가 1.33–1.39배이고 int64를 그대로 둔다(SP19, §7.5).
  - Accelerate에 넘길 때는 행 색인을 int32로 복사한다(§1.4). 행·열 수가 2³¹ − 1을 넘으면 Accelerate를 부르는 함수는 `ValueError`다.
- **표현**: 배열은 Python 클래스이고 필드가 `sd.ndarray`다(SP21): COO는 `data`와 축마다의 색인 배열(`coords`), CSR·CSC는 `data`, `indices`, `indptr`. 커널(`csrc/sparse/`)은 이 배열들을 연속 메모리의 포인터로 받는다. `format.hpp`가 커널이 함께 쓰는 것(줄(line), major·minor의 말, 중복의 합, 0의 판정, 표준 꼴의 `Compressed{indptr, indices, data}`)을 둔다. 나중의 그래프 영역도 같은 세 배열을 읽는다.

### 2.3 표준 꼴 (SP10)
- **CSR·CSC는 언제나 표준 꼴이다**: 줄마다 색인이 오름차순이고 중복이 없다. 모든 생성자와 연산의 결과가 그렇다. `has_canonical_format`, `has_sorted_indices`는 언제나 True라 두지 않고, `sort_indices`, `sorted_indices`, `sum_duplicates`, `check_format`도 CSR·CSC에는 두지 않는다. `[scipy와 다름]`: scipy는 `(data, indices, indptr)`에서 만든 행렬을 검사하지 않고 비표준 꼴로 두었다가 연산마다 필요하면 정리한다.
  - 이유: 합의 순서(§2.6)가 줄의 색인 순서로 정해지므로 표준 꼴이어야 결과의 비트가 "행렬의 값"만으로 정해진다. 이분 탐색 인덱싱, Accelerate 넘기기, 두 행렬의 병합 연산이 모두 정렬을 전제한다. 생성자의 검사는 O(nnz)이고, 정렬되어 있으면 복사도 하지 않는다.
  - `(data, indices, indptr)`로 만들 때: 범위(0 ≤ 색인 < 줄 길이, `indptr` 단조)를 검사하고(틀리면 `ValueError`), 정렬되지 않았거나 중복이 있으면 정렬하고 더한다(`copy=False`면 `ValueError`).
- **COO는 비표준 꼴을 가질 수 있다**: 만들기의 형식이기 때문이다(scipy와 같다). `sum_duplicates()`는 행 우선(C 순서)으로 정렬하고 중복을 더한다. `has_canonical_format`은 COO에만 있고, 저장된 원소가 실제로 표준 꼴인지를(SP32) 처음 물을 때 한 번 훑어 알아내 둔다(색인 배열은 바뀌지 않는다). `[scipy와 다름]`: scipy의 같은 이름은 확인하지 않고 두는 표시라, `(data, (row, col))`·전치·`copy`·`astype` 뒤에는 정렬되어 있어도 False다. 값만 바꾸는 연산은 결과가 같으므로 차이는 이 속성에만 보인다.
- **중복은 입력 순서대로 왼쪽부터 차례로 더한다**(COO → CSR·CSC, `sum_duplicates`, 중복이 있는 `(data, indices, indptr)`). 같은 자리의 원소를 입력에 나온 순서대로 놓는 안정 정렬 뒤에 순차 합이다(`[1, 10¹⁶, 1, −10¹⁶]` → 0). scipy의 순서는 하나가 아니다(SP0, `tools/gen_sparse_scipy.py`): `tocsr()`는 줄의 원소가 15개 이하일 때 이 규칙과 같지만(207/207) 더 긴 줄에서는 `csr_sort_indices`의 불안정 정렬 때문에 순서가 바뀌고(17–59개에서 193줄 중 146줄만 같다), COO의 `sum_duplicates()`는 첫 원소에 나머지의 `np.add.reduce`를 더한다(2–15개에서 207줄 중 36줄이 다르다). 그래서 "입력 순서"를 정의로 두고, scipy와는 반올림 크기에서 다를 수 있다(규칙 3). 정렬은 줄 번호로 안정 계수 정렬(고정 조각마다 개수를 세고 조각 순서대로 누적 → 스레드 수와 무관하게 안정), 줄 안은 색인으로 안정 정렬이다.

### 2.4 저장된 0과 NaN (SP11, SP12)
- **저장된 0(explicit zero)**: 사용자가 넣은 0은 저장된 원소로 남는다(scipy와 같다). `nnz`는 저장된 원소 수이고 `count_nonzero()`는 0이 아닌 값의 수다. `eliminate_zeros()`가 저장된 0을 지운다. 패턴을 지키는 것은 같은 패턴으로 다시 분해(`SparseRefactor`)하거나 사용자가 정한 구조를 쓰는 데 필요하다.
- **연산이 만든 0을 지우는지**는 연산마다 scipy의 규칙을 따른다(SP11, §3.2의 표): 희소끼리의 `+`, `-`, `*`, 비교와 희소 × 희소 곱은 결과의 정확한 0을 지우고(scipy의 `csr_binop_csr`, `csr_matmat`), 스칼라 곱(`A * 0`)과 원소별 함수는 패턴을 그대로 둔다. scipy 안에서도 둘이 다르지만(`A - A`는 nnz 0, `A * 0`은 nnz 그대로) `nnz`는 보이는 동작이라 규칙 2(API.md §4)대로 따른다. 대안은 "결과의 패턴은 입력의 패턴만으로 정한다"(값을 보지 않는다)이다.
- **암시적 0은 강한 0이다**: 저장되지 않은 원소는 산술에 들어가지 않는다. 그래서
  - `A * nan`, `A * inf`(스칼라)는 저장된 원소만 NaN·inf가 되고 암시적 0은 0이다(IEEE의 0·inf = NaN이 아니다). scipy와 같다.
  - `A @ x`에서 x_j가 NaN이어도 열 j에 저장된 원소가 없는 행은 NaN이 되지 않는다(BLAS의 희소 관례, scipy와 같다).
  - `A.multiply(B)`(밀집 B)도 암시적 0 자리는 0이다(B가 NaN이나 inf여도). scipy와 같다.
  - `A / c`: 저장된 원소만 나눈다. **c = 0이면 저장된 원소가 ±inf·NaN이 되고 암시적 0은 0이다**(`A * (1/c)`와 같은 규칙). `[scipy와 다름]`: scipy는 Python의 `ZeroDivisionError`다. 배열의 원소별 연산은 값 때문에 실패하지 않는다(API.md §2, docs/api/array.md §1.6)는 규칙을 따른다.
  - 예외는 결과가 밀집인 연산이다: 희소 ÷ 희소는 scipy처럼 밀집 배열을 돌려주고 0/0은 NaN이다(scipy와 같다). 희소 + 밀집은 밀집이고 IEEE대로다.
- **리덕션의 NaN**: `sum`, `mean`은 저장된 NaN을 전파한다. `max`, `min`은 NaN을 전파하고(배열과 같다), 암시적 0이 하나라도 있으면 0도 후보다(scipy와 같다). `nanmax`, `nanmin`은 NaN을 건너뛴다.
- **비교**: IEEE대로다(저장된 NaN끼리 `==`는 False, 암시적 0끼리는 True). `A == 0`, `A <= 1`처럼 암시적 0에서 참인 비교는 scipy처럼 True가 많은 희소 bool 배열이다(경고 없음).

### 2.5 Python 클래스와 변환 (SP21–SP30)
- 희소 배열은 Python 클래스다(SP21; frame의 `Series`처럼 "얇은 Python"의 예외, ARCHITECTURE §3). 필드는 `sd.ndarray`이고 계산은 C++ 커널 한 번 호출이다. 바탕 클래스 `sparray`가 공통 메서드를 갖고 직접 만들지 않는다(`TypeError`). 파일은 §6이다.
- **`indices`, `indptr`, `coords`는 쓰기 금지 배열이다**(SP22). 바꾸면 표준 꼴(§2.3)이 깨지기 때문이다. `data`는 쓸 수 있다(값만 바꾸는 흔한 쓰임). `A.data *= 2`는 제자리에서 바꾼 배열을 다시 대입하므로, `data`에는 같은 모양의 `sd.ndarray`를 대입할 수 있다(dtype이 바뀌어도 된다, scipy와 같다). `indices` 등에는 대입할 수 없다. scipy는 모두 쓸 수 있다(`[scipy와 다름]`).
- **생성자의 입력**(`_interop.py`): 다른 희소 배열(형식 변환), scipy의 희소 배열·옛 행렬(모든 형식), 모양 tuple(빈 배열, `dtype=None`은 float64), `(data, (row, col))`·`(data, (i,))`(네 형식 모두; scipy의 DOK는 받지 않는다), CSR·CSC의 `(data, indices, indptr)`, 밀집 배열(`sd.ndarray`, numpy 배열, 중첩 시퀀스). tuple을 가르는 규칙은 scipy와 같다: 정수만이면 모양, 둘째가 배열의 tuple이면 좌표, CSR·CSC에서 원소 셋이 배열이면 압축된 부분이다.
  - 값은 `sd.asarray`의 규칙이다(§2.2). `dtype=`은 명시적 변환이고, 밀집 입력에서는 0이 아닌 자리를 원래 값으로 고른 뒤에 바꾼다(scipy와 같다: 0.4가 int64의 0이 되어도 저장된 원소로 남는다).
  - **색인 배열은 언제나 복사하고 검사한다**(SP30; `copy`와 상관없다): 범위(밖이면 `ValueError`, 배열의 메시지 꼴 `index 5 is out of bounds for axis 1 with size 4`), `indptr`의 시작 0·줄어들지 않음·끝 = 저장 수. 받은 배열을 나중에 고쳐도 이 배열의 표준 꼴이 깨지지 않게 하려는 것이다(커널은 색인을 믿고 읽는다). 검사가 어차피 한 번 훑으므로 복사는 그 훑기에 붙어 있다. scipy는 검사도 복사도 하지 않는다(`[scipy와 다름]`; 비용은 §5의 표).
  - `copy`(SP27)는 값 배열 `data`에 대한 것이다: None이면 그대로 쓸 수 있을 때(dtype이 맞고 연속이고, CSR·CSC로는 이미 표준 꼴일 때) 받은 배열을 그대로 쓴다. 밀집 입력, 시퀀스, scipy 객체, DOK로의 변환, 정렬·중복 더하기가 필요한 입력은 복사가 필요하므로 `copy=False`면 `ValueError`다(`csr_array(): copy=False, but a dense input requires a copy`).
  - 같은 형식의 희소 배열을 받으면 새 객체이고 배열을 나눈다(`copy=True`면 값을 복사한다). DOK는 해시 맵을 나눈다(scipy의 `dict`와 같다).
- **형식 변환** `tocoo tocsr tocsc todok asformat`(SP27, SP29): 같은 형식이면 자신, 아니면 새 객체다. 나눌 수 있는 것은 나눈다: CSR → COO는 `data`와 `indices`(열 색인)를 그대로 쓰고 행 색인만 펼친다, 표준 꼴 COO → CSR은 `data`와 열 색인을 그대로 쓴다, `.T`는 같은 세 배열의 다른 형식이다. CSR ↔ CSC와 DOK는 언제나 새 배열이다. 1차원은 CSC가 될 수 없다(`ValueError`).
  - 결과의 형식은 scipy의 것이다: `reshape`은 모든 형식에서 COO(같은 모양이면 자신), DOK의 `real`·`imag`와 complex의 `conj()`는 CSR, CSR의 전치는 CSC, COO의 `astype`은 dtype을 바꾼 뒤 중복을 더한 표준 꼴이다.
  - DOK → 다른 형식은 키를 정렬하므로 결과가 넣은 순서와 무관하다(행 우선, `has_canonical_format`은 True). `[scipy와 다름]`: scipy의 DOK는 `dict`라 넣은 순서다.
  - `nonzero()`의 순서는 scipy와 같다: CSR·CSC·DOK는 행 우선, COO는 저장된 순서(중복도 그대로). `count_nonzero()`는 COO의 중복을 먼저 더하되 배열을 바꾸지 않는다(`[scipy와 다름]`: scipy는 제자리에서 `sum_duplicates()`를 한다).
- **sd.ndarray ↔ 희소**
  - `csr_array(x)`(밀집 `sd.ndarray`, 1·2차원): 0이 아닌 원소만 저장한다. NaN은 0이 아니므로 저장되고 -0.0은 0이므로 저장되지 않는다(scipy와 같다).
  - `A.toarray()`: C 순서의 새 `sd.ndarray`이고 0에 저장된 값을 더한 것이다. 그래서 저장된 -0.0은 밀집에서 +0.0이고, COO의 중복은 저장 순서로 더해진다(scipy와 같다). `todense()`는 없다(SP24).
  - 밀집으로 몰래 바뀌는 것을 막는다(SP26): `np.asarray(A)`는 `A.toarray()`를 짚는 `TypeError`, `sd.asarray(A)`도 `TypeError`다. `len(A)`, `bool(A)`, `iter(A)`도 `TypeError`다(`[scipy와 다름]`: scipy의 `len`은 `TypeError`이되 DOK는 저장 수, `bool`은 `ValueError`이되 (1, 1)이면 값). 배열과 같은 규칙이다(docs/api/array.md: 배열은 크기와 상관없이 `bool`이 `TypeError`).
- **numpy**: 생성자는 데이터를 받는 자리라 numpy 배열(밀집 2차원, 또는 `(data, indices, indptr)`·`(data, (row, col))`의 각 배열)을 받는다(API.md §5의 데이터 입력). 연산자의 밀집 피연산자는 `sd.ndarray`만이다: `A @ np_array`는 `TypeError`(배열과 같은 규칙, API.md §5; SP2).
- **scipy.sparse**(SP23)
  - 들여오기: 생성자가 scipy의 희소 배열과 옛 행렬(모든 형식; BSR·DIA·LIL은 scipy의 `tocoo()`로)을 받는다. **언제나 복사한다**: scipy 객체는 나중에 제자리에서 바뀔 수 있고(scipy의 `sum_duplicates`, `sort_indices`는 제자리 정렬이다) int32 색인은 어차피 넓혀야 한다. `copy=False`면 `ValueError`다. dtype은 §2.2의 규칙이다.
  - 내보내기: `A.to_scipy(*, copy=None)`가 같은 형식의 scipy 배열(`csr_array` 등)을 만든다. 값·색인 배열은 복사하지 않는 numpy 뷰다(배열의 버퍼 프로토콜로; 빈 scipy 배열을 만들고 세 배열을 대입해 scipy가 색인을 int32로 줄이지 않게 한다). 색인 뷰는 읽기 전용이라 scipy의 제자리 메서드(`eliminate_zeros` 등)는 `ValueError`다. 값 뷰는 쓸 수 있고 이 배열과 나눈다. `copy=True`면 복사한다. DOK는 언제나 복사한다(`copy=False`면 `ValueError`).
- **Python 스칼라와 연산자**: `A + c`(c ≠ 0)와 `A ** 0`은 결과가 밀집이 된다. scipy는 `NotImplementedError`다. 이쪽은 `ValueError`(`'adding a nonzero scalar to a sparse array is not supported; use A.toarray() + c'`)인 안이다(`[scipy와 다름]`: 예외 종류만, API.md §2의 목록에 `NotImplementedError`가 없다).

### 2.6 합의 순서와 결정성
**희소 합의 결과는 행렬의 값과 모양으로만 정해진다**: sd의 스레드 수, 저장 형식(CSR·CSC·COO), 태스크 분할과 무관하다(SP15).

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
`csrc/sparse/canonical.cpp`가 표준 꼴을 만든다. 입력은 원소 (major, minor, 값)의 목록이고(COO → CSR이면 (행, 열, 값), → CSC면 (열, 행, 값)), 출력은 (major, minor) 순서의 `indptr`·`indices`·`data`, 같은 자리는 입력 순서로 더한 하나다. 길은 둘이고 같은 함수를 계산한다(모든 정렬이 안정적이고 중복의 합이 입력 순서라서, 결과는 입력만으로 정해진다). 그래서 어느 길을 탈지는 속도로만 크기에 따라 고르고, 비트는 길과 스레드 수와 무관하다(테스트는 두 길과 1·2·3·7·10스레드의 바이트를 견준다).
- **곧은 길**: `indptr`을 계수기로 쓰는 major의 계수 정렬(scipy의 `coo_tocsr`과 같은 꼴), 그 뒤 줄마다 minor로 안정 정렬(32개까지 삽입 정렬, 넘으면 `std::stable_sort`)과 중복의 차례 합. 줄에 놓을 때 입력 위치 대신 값을 함께 옮겨, 나중에 값을 흩어 읽는 패스가 없다.
- **양동이 길**: major를 2^s개씩 묶은 양동이(64–1024개, nnz와 줄 수로만 정함)로 원소를 안정적으로 흩고(입력의 고정 조각 2^14개마다 양동이별 개수를 세어 조각 순서로 누적), 양동이마다 줄의 계수 정렬과 위의 줄 처리를 한다. 양동이가 캐시 크기라 한 스레드에서도 큰 입력에서 빠르고, 양동이끼리는 스레드풀에서 독립이다. 줄이 아주 많고 원소가 적은 양동이(줄 수 > 4·원소 수 + 1024)는 양동이째 안정 정렬한다.
- **고르는 크기**(M4 Pro, 줄마다 10개인 무작위 행렬에서 잼): 스레드풀을 쓸 수 있으면 2^16개부터 양동이 길, 한 스레드에서는 정렬·합이 필요한 변환이 2^17개부터, 전치(CSR ↔ CSC: 줄 안이 이미 정렬되어 정렬·합이 없다)가 2^20개부터다. 양동이를 64개보다 적게 두면 10스레드에서 몇 스레드가 늦게 끝나 2^17개 근처가 1스레드와 같았다.
- 이미 표준 꼴인 COO(`has_canonical_format`)는 정렬 없이 `indptr`만 만들고 값과 minor 색인을 그대로 쓴다.
- **CSR ↔ CSC**: 위의 전치(scipy의 `csr_tocsc`와 같은 계수 정렬; 안정이라 결과도 정렬되어 있다). `.T`는 복사 없는 뷰다: CSR의 `.T`는 같은 세 배열의 CSC다(scipy와 같다).
- **사용자의 `(data, indices, indptr)`**: `indptr`과 `indices`를 복사하며 검사하고(줄 구간으로 병렬; 틀린 곳은 스레드 수와 상관없이 첫 자리를 알린다), 줄마다 정렬되어 있고 중복이 없으면 그대로, 아니면 위의 표준 꼴 만들기로 간다.
- 밀집 → 희소(`csrc/sparse/convert.cpp`): 줄마다 0이 아닌 수를 세는 패스와 채우는 패스, 줄 구간으로 병렬(임의의 strides). 희소 → 밀집: 0으로 채운 C 순서 배열에 줄마다 쓴다. 중복이 있을 수 있는 COO만 저장 순서로 더하므로 한 스레드다.
- `astype`: 배열의 명시적 변환 규칙(array.md §2.4: float → int는 포화, NaN → 0)으로 `data`를 바꾼다(`sd.astype`과 비트까지 같다). 변환 뒤 0이 된 값도 저장된 원소로 남는다(scipy와 같다).
- DOK(`csrc/sparse/dok.cpp`): 키 (i, j)와 값의 평평한 배열, 열린 주소법과 선형 탐사, 부하 1/2 이하(지운 칸은 표시하고 다시 지을 때 버린다). 다른 형식으로 바꿀 때 키를 정렬한다.
- 그 밖(`convert.cpp`): `indptr` 펼치기, 범위·0으로 거르기(`resize`, `eliminate_zeros`, `nonzero`), 0이 아닌 수 세기(줄마다, 색인마다), `reshape`의 좌표(C 순서의 위치를 128비트 정수로 계산해 2^63을 넘는 크기도 된다).

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
- `format=None`의 기본 형식: scipy는 `eye_array`·`diags_array`가 DIA, `kron`이 BSR이다. 여기서는 CSR이다(SP6, `[scipy와 다름]`).
- `random_array`: `rng`는 sd.random의 `Generator`이고 꼭 받는다(optimize O31·htest와 같은 꼴의 `TypeError`; scipy의 `rng=None`은 시드 없는 무작위다). `random_state`는 없다. 자리 고르기는 비복원 추출(정확히 round(density·m·n)개, scipy와 같은 뜻), 값은 `data_sampler(size)` 또는 [0, 1) 균등이다. 같은 `rng` 상태면 스레드 수와 상관없이 같은 비트다. scipy와 같은 값은 아니다(난수 생성기가 다르다).
- `kron(A, B)`: 결과의 원소 aᵢⱼ·bₖₗ은 곱 하나라 순서 문제가 없다. 결과의 저장 순서는 표준 꼴(§2.3)이다.

### 3.5 전치, 모양, 그 밖 (SP1)
- `.T`, `.mT`, `transpose()`, `matrix_transpose(A)`: 2차원이면 같은 일이다(배열과 같은 이름들; `.T`는 뷰). 1차원에서 `.T`, `.mT`, `matrix_transpose`는 `sd.ndarray`처럼 `ValueError`이고(SP31)(`.T: the array must have 2 dimensions, got 1`), `transpose()`는 자신이다. `[scipy와 다름]`: scipy의 1차원 `.T`는 자신이다(numpy와 같다). 영역 사이에서 같은 이름이 같은 일을 하도록 sd의 배열을 따른다.
- `conj()`, `real`, `imag`, `copy()`, `astype`, `reshape`(COO를 거친다, 결과는 COO), `resize`, `nonzero()`, `eliminate_zeros()`(COO·CSR·CSC; scipy처럼 DOK에는 없다), `tocoo`·`tocsr`·`tocsc`·`todok`, `asformat`, `count_nonzero`, `nnz`, `shape`, `ndim`, `size`, `dtype`, `format`. `setdiag`은 SP3이다.
- 실수 dtype의 `real`은 값 배열을 나누고(scipy와 같다) `conj()`는 복사다. `copy()`는 값만 복사하고 쓰기 금지인 색인 배열은 나눈다(바뀌지 않으므로 보이는 차이가 없다).
- `resize(shape)`: 제자리에서 바꾸고 None을 돌려준다. 차원 수는 그대로다(`[scipy와 다름]`: scipy의 COO는 1·2차원 사이도 바꾸고(펼친 순서로), DOK는 `NotImplementedError`다; 한 규칙으로 둔다).
- 두지 않는 것: `dot`(`@`와 같다), `conjugate`(`conj`와 같다), `todense`(`toarray`와 같다), `power`의 정수 지수 0, `prune`(표준 꼴이라 할 일이 없다), `getrow`·`getcol`(옛 꼴), scipy의 `reshape(order=)`·`transpose(axes=, copy=)`·`astype(casting=)`.

### 3.6 직접 풀이 (SP4)
- **LU는 Accelerate의 `SparseFactorizationLU`(문턱 부분 피벗)를 AMD 순서로 부른다**(SP18). LU의 기본 순서 MTMetis와 `SparseOrderDefault`는 쓰지 않는다: 실행마다 결과가 다르고 동시 호출에서 프로세스가 죽는다(§1.4). Metis는 동시 호출에서 실행마다 다르다. 3차원 격자에서는 AMD의 채움이 더 많아 Metis보다 1.5–2배 느리다(§5). 순서를 고르는 옵션을 둘지는 §9 SP4의 1번이다.
- **Cholesky·LDLᵀ는 첫 판에 쓰지 않는다**(SP18). Accelerate의 대칭 분해는 기본 스레드 모드에서 실행마다 비트가 다르고, 프로세스 안에서 그것을 막을 길이 없다(`BLASSetThreading`도 듣지 않는다, §1.4). scipy.sparse.linalg의 공개 함수에는 대칭 분해가 없으므로(모두 SuperLU의 LU다) 범위는 줄지 않는다. 대칭 행렬도 LU로 푼다(대칭을 쓰지 못해 2차원 라플라시안 n = 9×10⁴에서 Cholesky 21 ms 대 LU 92 ms; scipy의 SuperLU 139 ms보다는 빠르다, §5).
  - 나중 후보: 이 라이브러리가 결정적인 희소 Cholesky를 짠다(초마디 Cholesky: 기호 분석과 제거 트리, 초마디마다 밀집 `potrf`·`trsm`·`syrk`, 트리 위의 병렬은 형제의 결과를 고정 순서로 더한다). CHOLMOD 꼴이라 큰 일이고 나중 후보다(regression의 정규방정식, eigsh의 shift-invert에 쓰인다).
- **특이의 판정**: Accelerate는 구조적 특이만 알린다(§1.4). `zeroTolerance = 0`으로 분해하면 정확한 0 피벗이 풀이에 ±inf·NaN을 낸다. 그래서 (1) 구조적 특이(`SparseMatrixIsSingular`)는 `LinAlgError("singular matrix")`, (2) 유한한 입력의 풀이가 유한하지 않으면 `LinAlgError("singular matrix")`다. linalg의 "특이는 정확한 0 피벗만"(linalg.md §2.2)에 가장 가깝지만, 피벗을 볼 수 없으므로 넘침으로 생긴 inf도 특이로 보고한다. `splu(A)`는 분해할 때 1의 벡터로 한 번 풀어 확인한다(O(nnz(L + U)), 분해보다 작다).
  - **SP0에서 찾은 빈틈**: 반올림 때문에 0 피벗이 정확히 0이 되지 않는 정확히 특이한 행렬(Neumann 라플라시안, 같은 열 둘)은 status 0과 크기 1/ε의 유한한 해다(n = 100에서 2.6×10¹⁶). 그 해의 정규화 역오차는 ε 크기라(1.6×10⁻¹⁶) 잔차 검사로는 보이지 않는다. 판정을 더할지(분해의 조건수 추정: `onenormest`로 ‖A⁻¹‖₁을 어림해 κ₁·ε ≥ 1이면 `LinAlgError`)는 §9 SP4의 4번이다. scipy의 SuperLU도 이런 행렬(dupcol/40, lap2d/10n)에서 경고 없이 크기 3×10¹⁹·5×10¹⁴의 해를 낸다(SP0의 기록).
- **빈 행렬**: Accelerate는 0 × 0을 받지 않는다(매개변수 오류). `spsolve`는 부르기 전에 빈 결과로 답한다.
  - `[scipy와 다름]`: scipy `spsolve`는 특이 행렬에서 `MatrixRankWarning`과 NaN 배열을, `splu`는 `RuntimeError("Factor is exactly singular")`를 낸다. 여기서는 둘 다 `LinAlgError`다(linalg와 같다).
- **유한성 검사**: `A`와 `b`에 NaN·inf가 있으면 `ValueError`다(linalg.md §2.3과 같다; 저장된 원소만 본다).
- `spsolve(A, b, /)`: A는 정방 CSR·CSC(다른 형식은 CSC로 바꾼다). b가 (n,)이면 (n,), (n, k)이면 (n, k) 밀집, b가 희소면 결과도 희소(scipy와 같다). `permc_spec`, `use_umfpack`은 없다.
- `splu(A, /)`: 분해 객체. 메서드 `solve(b, /, *, trans='N')`(`'N'`, `'T'`, `'H'`; Accelerate의 전치·켤레 전치 풀이), 속성 `shape`, `nnz`(인수의 크기를 줄 수 있으면). scipy `SuperLU`의 `L`, `U`, `perm_r`, `perm_c`는 줄 수 없다(§1.4; §9 SP4의 2번). `permc_spec`, `diag_pivot_thresh`, `relax`, `panel_size`, `options`는 SuperLU의 옵션이라 두지 않는다.
- `factorized(A, /)`: `splu(A).solve`와 같은 호출 가능한 객체.
- `inv(A, /)`: 단위 행렬을 우변으로 푼 희소 결과(scipy와 같다; 결과의 0은 지운다). 대개 밀집이 되므로 문서에 적는다.
- `spsolve_triangular(A, b, /, *, lower=True, unit_diagonal=False)`: 이 라이브러리의 커널이다. CSR의 앞(뒤)으로 대입: xᵢ = (bᵢ − Σⱼ aᵢⱼxⱼ)/aᵢᵢ, 합은 §2.6의 줄 합(j 오름차순 FMA). 우변이 여럿이면 열마다 병렬(열끼리 독립). 한 우변 안의 병렬은 단계 스케줄링(같은 단계의 행은 서로 기대지 않는다)으로 하고 행 하나의 순서는 바뀌지 않는다. 대각의 0은 `LinAlgError("singular matrix")`(linalg.md §2.2의 `zero_on_diagonal`과 같다). scipy의 `overwrite_A`, `overwrite_b`는 없다.
- `is_sptriangular`, `spbandwidth`: 패턴만 보는 O(nnz) 도구.
- **우변이 여럿**: Accelerate가 열을 함께 풀어 열 하나의 비트가 혼자 푼 것과 다르다(반올림 크기, 결정적). 열마다 따로 풀지는 §9 SP4의 6번이다.
- 정확도는 SuperLU와 다른 피벗(문턱 ≤ 0.5)과 척도 조정이라 끝자리뿐 아니라 역오차의 크기도 다를 수 있다. 계약은 역오차로 정한다(§4). 반복 다듬기(iterative refinement)를 넣을지는 SP4에서 잰 역오차를 보고 묻는다.

### 3.7 반복법 (SP5)
- **이 라이브러리가 짠다**(SP16; SP4의 "알맞은 반복법은 Accelerate"를 바꾼다). 이유:
  - scipy와 같은 멈춤 판정과 상태(`info`)를 지킬 수 없다: Accelerate의 판정은 처음 잔차에 대한 상대이고(scipy는 ‖b‖에 대한 상대, rtol 기본 10⁻⁵ 대 √ε), 반복 수를 돌려주지 않으며, GMRES의 기본은 DQGMRES다(scipy는 재시작 GMRES, `restart=20`). 한 걸음씩 부르는 `SparseIterate`로 판정을 직접 하면 판정은 맞출 수 있지만, 그 안의 직교화·Givens 회전·내적의 순서는 열 수 없다.
  - MINRES, BiCGSTAB, LSQR은 어차피 짜야 한다.
  - 반복법의 비용은 SpMV(이 라이브러리의 커널, §2.6)와 내적·axpy다. 알고리즘 자체는 짧다(CG 수십 줄). 같은 커널과 결정적인 내적(linalg의 `fma_dot_leaf` pairwise, linalg.md §8.2)을 쓰면 비트가 스레드 수와 무관하다.
  - Accelerate의 CG·GMRES·LSMR은 SP5에서 속도의 비교 대상으로 잰다.
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
- **`onenormest(A, /, *, t=2, max_iter=5, compute_v=False, compute_w=False)`**: Higham–Tisseur(2000) 블록 1-노름 추정. scipy는 첫 열 말고는 numpy의 시드 없는 무작위 ±1 열이라 실행마다 다를 수 있다. 여기서는 고정된 열(sd.random의 고정 키)이라 결정적이다(안, §9 SP7의 1번).
- **`norm(x, /, *, ord=None, axis=None)`**: scipy의 뜻(행렬: `'fro'`, 1, −1, inf, −inf, 2(가장 큰 특이값, §3.8의 `svds`); 벡터 축: 정수 p). 합은 §2.6. 이름은 scipy의 것이고 `ord`의 규칙과 메시지는 `sd.linalg`의 `matrix_norm`·`vector_norm`과 같다(SP14; docs/api/sparse.md §8).
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
SP0에서 잰 scipy 1.18.1의 시간이다(ms, 호출 하나의 가장 좋은 값, `VECLIB_MAXIMUM_THREADS=1`; scipy.sparse의 커널은 1스레드다). 무작위 n × n, 행마다 10개(n = nnz/10), float64, scipy가 만드는 int32 색인. 기기를 다른 작업과 나눠 써서 한 칸은 흔들릴 수 있다. 같은 호출이 `bench/speed/cases/sparse.py`의 비교 대상이다.

| 연산 | nnz = 10⁴ | 10⁵ | 10⁶ | 10⁷ |
|---|---:|---:|---:|---:|
| COO → CSR (중복 있음) | 0.077 | 1.19 | 12.7 | 193 |
| CSR → CSC | 0.023 | 0.19 | 2.64 | 81 |
| `A @ x` | 0.006 | 0.035 | 0.645 | 7.38 |
| `A @ X` (k = 16) | 0.032 | 0.39 | 4.82 | 298 |
| `A.T @ x` | 0.013 | 0.055 | 0.91 | 10.0 |
| `A + B` | 0.050 | 0.67 | 7.2 | 74 |
| `A * B` (원소별) | 0.052 | 0.67 | 7.1 | 71 |
| `A @ B` | 0.36 | 3.6 | 56 | — |
| `sum(axis=0)` | 0.018 | 0.067 | 0.94 | 10.3 |
| `sum()` | 0.003 | 0.014 | 0.12 | 1.21 |
| `max(axis=1)` | 0.038 | 0.18 | 1.56 | 15.4 |
| `A[:, :n/10]` | 0.023 | 0.13 | 1.09 | 11.6 |
| `kron(eye_array(10), A)` | 0.24 | 1.6 | 15.3 | — |
| `tril(A)` | 0.062 | 0.40 | 3.94 | 39 |
| `norm(A, 1)` | 0.033 | 0.095 | 1.09 | 12.6 |
| `onenormest(A)` | 0.22 | 1.6 | 16.9 | 197 |
| `svds(A, k=6)` | 4.6 | 26 | 565 | — |

풀이(Dirichlet 라플라시안; 직접 분해는 Accelerate를 기본 스레드 모드로, 괄호는 1스레드 모드):

| 연산 | 2차원 n = 10⁴ | 9×10⁴ | 4.9×10⁵ | 10⁶ | 3차원 n = 1.1×10⁴ | 9.1×10⁴ |
|---|---:|---:|---:|---:|---:|---:|
| scipy `splu` (SuperLU, COLAMD) | 10.2 | 139 | 1,231 | 3,553 | 146 | 17,717 |
| Accelerate LU/AMD | 8.7 (9.5) | 92 (91) | 521 (656) | 1,054 (1,442) | 36 (44) | 975 (1,696) |
| Accelerate LU/Metis | 26 | 304 | 1,627 | 3,470 | 63 | 929 |
| Accelerate Cholesky/AMD (쓰지 않는다) | 2.4 | 21 | 114 | 247 | 7.8 | 203 |
| R Matrix `lu` (CSparse) | 28 | 1,250 | 10,662 | — | — | — |
| R Matrix `solve` (대칭: CHOLMOD) | 5.1 | 120 | 1,403 | — | — | — |
| scipy `splu(...).solve` | 0.33 | 4.2 | 30 | 64 | 1.9 | 57 |
| scipy `cg` (rtol 10⁻¹⁰) | 15 | 268 | 2,938 | 8,790 | 5.1 | 59 |
| scipy `minres` (rtol 10⁻¹⁰) | 14 | 183 | 2,021 | 5,506 | 5.3 | 49 |
| scipy `gmres` (rtol 10⁻⁸, restart 20) | 217 | 2,567 | 13,951 | — | 18 | 302 |
| scipy `eigsh(k=6, 'LM')` | 93 | 4,816 | 189,266 | — | 51 | 894 |
| scipy `expm_multiply(−0.1 A, b)` | 1.2 | 8.5 | 43 | 91 | 1.4 | 10 |

- Accelerate LU/AMD는 scipy `splu`보다 2차원에서 1.2–3.4배(1스레드 모드 1.1–2.5배), 3차원에서 4–18배 빠르다. 설계 초안의 "3배(73 대 225 ms)"는 SP0에서 1.5배(92 대 139 ms)였다(scipy 쪽이 빨랐다). 3차원에서는 AMD의 채움이 Metis보다 많아(n = 2.2×10⁵에서 4.5 대 3.5 GB) 1.5–2배 느리다. 결정성(SP18)의 값이다. Accelerate LU의 풀이 하나는 n = 10⁶에서 34 ms(scipy 64 ms)다.
- R Matrix 1.7.6(1스레드): `sparseMatrix` 만들기 nnz 10⁶ 13.8 ms, `A %*% x` 1.0 ms, `t(A)` 3.4 ms, `A + A` 31 ms, `colSums` 0.75 ms, `A %*% A` 220 ms.

**SP1에서 잰 것**(표는 `bench/sparse/speed.md`): 변환(COO → CSR, CSR → CSC, CSR → COO, `sum_duplicates`)은 nnz 10⁷에서 1스레드 1.9–80배, 10스레드 10–500배 scipy보다 빠르다. 1스레드의 COO → CSR·CSR → CSC는 10⁵–10⁶에서 1.1–1.4배로 목표(2배)에 못 미친다(같은 계수 정렬이고 int64 색인이라 옮기는 바이트가 많다). SP8에서 고친다(SP33). `(data, indices, indptr)`·`(data, (row, col))`로 만들기는 색인을 검사하고 복사하므로 scipy(검사도 복사도 없음)보다 느리다(10⁷에서 1스레드 8–11 ms, 10스레드 1–1.5 ms; scipy 4 µs·4 ms). `toarray`는 scipy와 같다(0으로 채우는 비용이 대부분이다). `to_scipy`는 뷰라 크기와 상관없이 수십 µs다.

**목표** (같은 기기, 1·10스레드, scipy 대비 배율)
- SpMV, `A.T @ x`, 축 리덕션: 1스레드에서 scipy와 같거나 빠르게(int64 색인의 대역 비용이 있어 0.8배까지를 받아들이고 넘으면 §2.2의 int32 안을 묻는다), 10스레드에서 nnz ≥ 10⁶이면 4배 이상(메모리 대역의 한계, array.md §10.6의 `f + g` 1.7배보다 SpMV는 무작위 읽기라 더 나뉜다).
- SpMM: scipy가 k 방향으로 벡터화하지 않아 느리다(10⁷에서 SpMV의 29배). 1스레드에서 3배 이상.
- 변환(COO → CSR, CSR → CSC, 중복 더하기): 1스레드에서 2배, 10스레드에서 5배 이상.
- SpGEMM, 이항 연산: 1스레드에서 같거나 빠르게, 10스레드에서 4배 이상.
- 직접 풀이: Accelerate LU/AMD는 2차원 격자에서 SuperLU보다 1.2–3.4배, 3차원에서 4–18배 빠르다(위 표). `spsolve`·`splu`는 n ≥ 10⁵에서 2배 이상을 목표로 하고 행렬 꼴(2·3차원 격자, 회로, 무작위)마다 잰다.
- 반복법: 반복 하나의 비용이 SpMV + 내적이다. scipy의 Python 반복 대비 같은 반복 수에서 1스레드 1.5배, 10스레드 4배 이상.
- `eigsh`·`svds`: ARPACK(Fortran)과 같은 SpMV 수에서 같거나 빠르게, 10스레드에서 SpMV 몫만큼 빠르게.
- 작은 행렬(nnz ≤ 10³): 호출 하나의 고정 비용이 scipy(Python 층, 수 µs–수십 µs)보다 작게.
- 크기: nnz 10⁴, 10⁵, 10⁶, 10⁷(SpGEMM은 10⁶까지), 행 길이 3·10·100과 긴 행이 섞인 행렬, 풀이는 2·3차원 격자 n = 10⁴–10⁶. 사례는 `bench/speed/cases/sparse.py`에 마일스톤마다의 계획으로 있고, 그 마일스톤의 이름이 공개되면 사례가 된다(§7).

---

## 6. 층과 파일
- **층**: `sparse/`는 `array/`, `linalg/`(LAPACK 래퍼, 비용 규칙, pairwise 잎), `parallel/`, `accelerate/`, `base/` 위에 있다(ARCHITECTURE §1의 "아래로만"). 다른 영역은 `sparse/`를 모른다. 예외는 §3.10의 판별 하나(`bind/pyutil`의 `is_sparse` 꼴)와 나중의 그래프 영역이다.
- `src/stardust/sparse/` (하위 패키지 없음, SP3): `__init__.py`(공개 이름, `issparse`, `matrix_transpose`), `_base.py`(`sparray`: 공통 메서드, 나중에 연산자와 리덕션), `_coo.py`, `_compressed.py`(`csr_array`, `csc_array`), `_dok.py`, `_construct.py`(만들기 함수), `_index.py`(인덱싱과 대입), `_interop.py`(생성자의 입력과 `to_scipy`), `_interface.py`(`LinearOperator`, `aslinearoperator`), `_direct.py`(`spsolve splu factorized inv spsolve_triangular is_sptriangular spbandwidth`), `_iterative.py`(`cg gmres minres bicgstab lsqr lsmr`), `_eigen.py`(`eigsh eigs svds`), `_matfuncs.py`(`expm expm_multiply onenormest norm`), `_result.py`(결과 클래스), `_common.py`(입력 정리와 메시지).
- `csrc/sparse/`(SP0: `direct.hpp`·`direct.cpp`, 비공개 `_lu_solve`로 래퍼를 시험한다; SP1: `format.hpp`, `parallel.hpp`, `canonical.cpp`, `convert.cpp`, `dok.cpp`; 나머지는 마일스톤마다): `format.hpp`(§2.2의 표현, 커널이 함께 쓰는 것), `canonical.cpp`(표준 꼴: 정렬, 중복 더하기, 전치, 입력의 검사; §3.1), `convert.cpp`(밀집 사이, 펼치기, 거르기, 세기, `reshape`의 좌표), `spmv.cpp`(SpMV·SpMM·Aᵀx, §2.6의 순서), `spgemm.cpp`, `binop.cpp`(병합 이항 연산, 비교), `elementwise.cpp`(배열의 원소별 커널을 `data`에), `reduce.cpp`, `index.cpp`, `construct.cpp`, `dok.cpp`(해시 맵), `triangular.cpp`, `direct.cpp`(Accelerate LU, 특이의 판정), `iterative.cpp`, `lanczos.cpp`(IRLM, 이중대각화), `arnoldi.cpp`(IRAM), `matfuncs.cpp`(expm, expm_multiply, onenormest), `parallel.hpp`(크기로만 정하는 분할, 기준값 2^15).
- `csrc/accelerate/sparse.hpp`(SP0): Sparse Solvers 래퍼(ARCHITECTURE §4.4: Accelerate 래퍼는 `accelerate/` 한곳). int64 → int32 행 색인 복사(`Csc::narrow`), 스레드마다 모으는 `reportError` 콜백(`ErrorScope`; status가 0이어도 메시지가 있으면 실패), status → `Error{Singular, Parameter, Failed, Internal}`, LU/AMD·`zeroTolerance = 0` 고정, `SparseCleanup`의 RAII(`Lu<T>`, float64·complex128).
- `csrc/linalg/lapack.hpp`에 더할 래퍼: `stev`·`steqr`(삼중대각 고유값), `hseqr`·`trevc`(헤센베르크), 필요하면 `bdsqr`(이중대각 SVD).
- `csrc/bind/sparse_bind.cpp`(`stardust._core.sparse`, SP0), `csrc/bind/module.cpp`에 하위 모듈 하나.
- `CMakeLists.txt`: 새 `.cpp`만 더한다. Accelerate는 이미 링크되어 있다. 블록(`^`)을 쓰는 `SparseIterate`는 쓰지 않으므로 `-fblocks`가 필요 없다(clang++이 블록을 기본으로 받는 것도 탐침에서 확인했다).
- `tests/sparse/`(주제마다 `test_<주제>.py`, `test_api_surface.py`가 `docs/api/sparse.md`를 읽는다), 시험 행렬 `tests/sparse/matrices.py`, 참값 `tests/sparse/ref/`와 그 읽기·계약 `tests/sparse/refdata.py`, 대조 데이터 `tests/sparse/data/`, 도구 `tools/sparse_ref.py`(Arb), `tools/gen_sparse_scipy.py`, `tools/gen_sparse_r.py`, 탐침 `tools/sparse_probe.py`(`tools/sparse_probe_accelerate.cpp`, `tools/sparse_probe_spmv.cpp`), 측정 `bench/speed/cases/sparse.py`·`bench/accuracy/sparse.py`, 골든 `tests/golden/cases_sparse.py`.

---

## 7. 참값과 검증 (SP0에서 만듦)

### 7.1 시험 행렬 (`tests/sparse/matrices.py`)
- 만드는 식과 고정 시드의 행렬 64개다(내려받지 않는다). 좌표(`row`, `col`, `data`)로 들고 `csr()`·`csc()`·`dense()`·`scipy()`로 바꾼다. 아이디로 고르고(`get("lap2d/30")`) 태그로 찾는다(`ids(tag="spd", small=True)`). 큰 여섯(`HEAVY`, n ≥ 9×10⁴)은 시간 재기에만 쓰고 테스트는 만들지 않는다. `digest()`가 참값·기록 파일이 어느 행렬에서 왔는지 적는다.
  - 격자: 1·2·3차원 라플라시안(Dirichlet = 양정치, Neumann = 정확히 특이), 대각 우세 대류-확산(대각 우세가 아닌 계수(1.3, 0.7, 1.2, 0.9; 합 4.1 > 대각 4)는 n = 9×10⁴에서 조건수가 10²⁰을 넘어 Accelerate LU·QR의 잔차가 10¹¹–10²²였다. 시험 행렬은 조건수를 알고 만든다).
  - 무작위: 대각 우세 비대칭, 양정치, 대칭 부정, 복소 에르미트, 복소 비대칭; 띠, 밀집 블록 대각(블록 안에서 피벗), 안장점 [[A, Bᵀ], [B, 0]].
  - 어려운 것: 등급 매긴 삼중대각(조건수 10⁸–10¹²), Neumann + 10⁻⁸I, 구조적 특이(빈 열), 값으로 특이(같은 열 둘), 직사각(최소제곱; 랭크 부족 하나), L = 4096보다 긴 행, 합이 상쇄하는 행, 빈 행·열과 0 × n.
  - 비표준 COO 입력: 정렬되지 않은 것, 중복(`[1, 10¹⁶, 1, −10¹⁶]` 포함)과 저장된 0, int64, bool, complex, NaN·±inf·−0.
- `tests/sparse/test_matrices.py`가 태그가 맞는지(대칭, 양정치, 부정, 특이, 조건수), 표준 꼴의 규칙(§2.3)을 확인한다.

### 7.2 SuiteSparse의 실제 행렬 (SP17, SP20)
SP20으로 승인된 여덟 개다. 주소는 SuiteSparse Matrix Collection의 Matrix Market 파일(`https://suitesparse-collection-website.herokuapp.com/MM/<group>/<name>.tar.gz`, 설명은 `https://sparse.tamu.edu/<group>/<name>`)이다. 서버가 내용 없는 크기 조회(HEAD)를 거절해(S3 403) 파일 크기는 저장 원소 수에서 어림한 값이다(대칭 행렬은 아래 삼각만 저장; Matrix Market 글 한 줄 약 30바이트, gzip 약 40%). 받는 도구는 `tools/sparse_suitesparse.py`이고 파일은 `build/sparse/suitesparse/<group>/<name>.mtx`(git이 무시한다)에 둔다. 읽는 테스트는 `tests/sparse/test_suitesparse.py`(external 등급, 파일이 없으면 받는 명령을 적고 건너뛴다)다.
- **2026-11-21의 내려받기는 되지 않았다**: 여덟 주소 모두 컬렉션의 파일 서버(`sparse-files.engr.tamu.edu`, S3)로 넘어가 `403 Forbidden`(AccessDenied)을 돌려줬다(`https://sparse.tamu.edu/MM/...`도 같은 곳으로 간다). 그래서 받은 파일, 실제 크기, sha256이 아직 없다(`SHA256`은 비어 있고, 처음 받을 때 도구가 값을 찍는다). 서버가 다시 열리면 같은 도구로 받고 sha256을 적는다. 여기서 나온 참값·기록도 그때 만든다.

| 행렬 | 꼴 | n | nnz | 조건수(사이트) | 쓰임 | 어림 크기 |
|---|---|---:|---:|---:|---|---:|
| `HB/bcsstk14` | 구조, 대칭 양정치 | 1,806 | 63,454 | — | 직접 풀이, `cg`, `eigsh` | 0.4 MB |
| `Boeing/msc10848` | 구조, 대칭 양정치 | 10,848 | 1,229,776 | 1.0×10¹⁰ | 큰 양정치, 긴 행 | 7 MB |
| `HB/sherman3` | 유체, 비대칭 | 5,005 | 20,033 | 5.0×10¹⁷ | 거의 특이한 LU | 0.3 MB |
| `Goodwin/goodwin` | 유체, 비대칭 | 7,320 | 324,772 | 2.0×10¹⁰ | LU, `gmres` | 4 MB |
| `Hamm/memplus` | 회로, 비대칭 | 17,758 | 99,147 | 1.3×10⁵ | 회로 꼴의 LU·`gmres` | 1.2 MB |
| `GHS_indef/cvxqp3` | 최적화 KKT, 대칭 부정 | 17,500 | 114,962 | 7.1×10¹⁵ | 안장점, `minres` | 0.8 MB |
| `HB/well1850` | 최소제곱, 직사각 | 1,850 × 712 | 8,755 | 1.1×10² | `lsqr`, `lsmr` | 0.1 MB |
| `HB/young1c` | 음향, 복소 비대칭 | 841 | 4,089 | 4.2×10² | 복소 LU | 0.1 MB |

모두 합쳐 약 14 MB다.

### 7.3 참값 (`tools/sparse_ref.py` → `tests/sparse/ref/`, `tests/reffile.py`의 npz)
python-flint 0.9.0(Arb)과 정확한 유리수다. mpmath는 쓰지 않는다. 점마다 행렬 아이디와 `digest`를 적는다. 모두 2⁻¹¹⁰까지 정해졌고 빠진 점은 없다. 실행은 6프로세스에서 3초, 작업 프로세스의 최대 RSS 0.41 GB다(`maxtasksperchild`).

| 묶음 | 점 | 방법 |
|---|---:|---|
| `spmv` | 112 | y = Ax, z = Aᵀw의 정확한 이진 유리수 합(실수·복소) |
| `sums` | 55 | 행 합, 열 합, 전체 합(정확) |
| `norms` | 53 | 1, −1, inf, −inf, Frobenius(정확한 합과 Arb의 제곱근), 2-노름(min(m, n) ≤ 220, Rayleigh 몫과 역반복, Kato–Temple 포함) |
| `solve` | 30 | Rump 꼴의 검증된 풀이(‖A⁻¹‖ ≤ ‖R‖/(1 − α), 정확한 잔차로 다듬어 2⁻¹¹⁰‖x‖까지; graded/400x12의 κ∞ = 7.5×10¹⁴, near_singular/20, 복소까지) |
| `lstsq` | 2 | 정확한 정규방정식에 같은 검증 |
| `eig` | 20 | 대칭·에르미트 행렬의 가장 작은·큰 고유값 6개씩(n ≤ 220은 Arb `eig`, Dirichlet 격자는 닫힌 꼴), 직사각 행렬의 특이값 |
| `expm` | 38 | `arb_mat.exp`·`acb_mat.exp`(n ≤ 200), 1-노름이 1과 30 근처가 되는 2의 거듭제곱 척도 둘 |

- 빠진 것: n > 220의 2-노름과 고유값, n > 200의 expm(밀집 Arb의 비용), 랭크 부족·넓은 직사각의 최소제곱, NaN이 있는 `coo/nan_inf`의 합.
- `tests/sparse/refdata.py`가 파일을 읽고 계약(§4)의 척도를 둔다. 테스트와 정확도 체계가 같이 쓴다.

### 7.4 scipy 1.18.1과 R 4.6.2의 기록
- **scipy** (`tools/gen_sparse_scipy.py` → `tests/sparse/data/scipy.json`, 기록 661개): 반올림이 아닌 동작(결과 형식·형·dtype, `nnz`와 0 지우기, NaN 규칙, 예외), SpMV·SpMM의 비트, 반복법의 반복 수·상태·해, `eigsh`·`eigs`·`svds`(고정 `v0`·`rng`), 직접 풀이와 특이 행렬, `expm`, `norm`. 찾은 것:
  - SpMV: CSR·CSC·COO의 `@ x`가 모두 FMA 순차 합과 비트까지 같다(cancel/50, long_rows/300 포함). 더 넓은 무작위 행렬(nnz 10⁵–10⁶, 행 길이 3–100)과 2차원 라플라시안에서 `csr @ x`(int32·int64 색인), `csc @ x`, `csr @ X`(k = 16)도 같다(`tools/sparse_probe.py spmv`).
  - 중복 더하기의 순서는 하나가 아니다(§2.3).
  - 결과 형식: `eye_array`·`diags_array`는 DIA, `kron`·`block_array`·`block_diag`·`tril`·`triu`·`random_array`·`A.multiply(dense)`·`max(axis=)`는 COO, CSR의 전치는 CSC, `A / B`와 `sum(axis=)`는 밀집이다. 정수 대각의 `diags_array`는 float64와 `FutureWarning`이다.
  - 예외: `A / 0`은 `ZeroDivisionError`, `A + 1`은 `NotImplementedError`다. `A * 0`은 패턴을 두고 `A - A`는 nnz 0이다. `A.multiply(NaN이 있는 밀집)`은 저장된 자리에만 NaN이다.
  - 특이: 구조적 특이에서 `spsolve`는 `MatrixRankWarning`과 NaN, `splu`는 `RuntimeError: Factor is exactly singular`다. 값으로 특이한 행렬(dupcol/40, lap2d/10n)은 경고 없이 크기 3×10¹⁹, 5×10¹⁴의 유한한 해다(Accelerate와 같은 빈틈, §3.6).
- **R** (`tools/gen_sparse_r.py` → `tests/sparse/data/r.json`, Matrix 1.7.6, 행렬 23개): `%*%`, `crossprod`, `kronecker`, `rowSums`·`colSums`, `solve`(CSparse LU, 대칭 양정치는 CHOLMOD), `lu`, `qr`(최소제곱), `expm`, `norm`, `rcond`. R의 `solve`는 특이 행렬(dupcol/40, lap2d/10n, 피벗 비가 4.9×10⁻²³인 cancel/50)에서 "computationally singular"로, struct_singular/40에서 "LU factorization of .gCMatrix failed"로 멈춘다. Matrix에는 복소 희소 클래스가 없어 복소 행렬은 빠졌다.
- 참값 대비: scipy와 R의 모든 값이 상대 10⁻¹³ 안이다(expm은 10⁻¹⁴, 풀이는 κε 크기). 큰 것은 near_singular/20의 3.9×10⁻⁹뿐이고 κ = 8×10⁸과 맞는다.
- `tests/sparse/test_reference_data.py`가 파일과 행렬의 `digest`, 출처(`_provenance`), double-double의 꼴, 작은 경우의 정확한 잔차, scipy·R과 참값의 일치를 확인한다.

### 7.5 Accelerate 탐침과 색인 폭 (`tools/sparse_probe.py`)
`tools/sparse_probe_accelerate.cpp`(절 `det`, `conc`, `sing`, `speed`, 두 스레드 모드)와 `tools/sparse_probe_spmv.cpp`(절 `spmv`)를 sd의 플래그로 빌드해 돌린다. 결정성·동시 호출·특이 판정은 §1.4·§3.6, 분해 속도는 §5에 옮겼다. 메모리: det 절(기본 모드)의 최대 RSS는 9.8 GB였고, 가장 큰 것은 안장점 n = 1.1×10⁵의 LU/AMD(5.3 GB, 분해 15초; 0 대각 블록 때문에 피벗이 채움을 키운다)다. 무작위 패턴(열이 행 전체에 흩어진 것)은 n = 3,600에서도 인수가 80 MB(거의 밀집)라 큰 무작위 행렬은 대각 ±100 안의 열로 만든다.

**int32 대 int64 색인의 SpMV** (SP8; 같은 커널(행마다 j 오름차순의 `std::fma` 순차 합)을 색인 폭만 바꿔, int32·int64를 번갈아 부른 회차의 중앙값, ms; 비율은 int64 / int32, 세 번 잰 것의 범위. 10스레드는 nnz 약 2¹⁴개씩의 고정 행 구간)

| 행렬 | nnz | 1스레드 int32 | int64 | **1스레드 비율** | 10스레드 비율 | SpMM k = 16 1스레드 비율 |
|---|---:|---:|---:|---:|---:|---:|
| 무작위 행마다 3 | 10⁵ | 0.065 | 0.071 | 0.97–1.12 | 1.05–1.18 | 1.05 |
| 무작위 행마다 10 | 10⁵ | 0.033 | 0.036 | 1.08–1.10 | 1.01–1.09 | 1.06–1.08 |
| 무작위 행마다 100 | 10⁵ | 0.056 | 0.057 | 1.01–1.02 | 0.93–1.06 | 1.05 |
| 무작위 행마다 3 | 10⁶ | 0.78 | 0.83 | 1.06 | 1.08–1.12 | 0.96–0.98 |
| 무작위 행마다 10 | 10⁶ | 0.71 | 0.74 | 1.04–1.05 | 1.06–1.11 | 1.02–1.03 |
| 무작위 행마다 100 | 10⁶ | 0.58 | 0.60 | 1.04 | 1.06–1.07 | 1.05 |
| 무작위 행마다 3 | 10⁷ | 17.5 | 20.3 | 1.13–1.18 | 1.07–1.20 | — |
| 무작위 행마다 10 | 10⁷ | 7.68 | 8.07 | 0.99–1.05 | 1.13–1.16 | 1.01 |
| 무작위 행마다 100 | 10⁷ | 7.04 | 7.16 | 1.02–1.09 | 1.13–1.30 | 1.02 |
| 2차원 라플라시안 | 4.5×10⁵ | 0.147 | 0.158 | 1.07 | 1.05–1.11 | 1.07 |
| 2차원 라플라시안 | 5.0×10⁶ | 1.74 | 1.87 | 1.07–1.08 | **1.33–1.39** | 1.06–1.07 |

- 1스레드의 int64 비용은 모든 경우에 1.3배 아래다(0.97–1.18배; 대표 크기 nnz 10⁶–10⁷은 1.02–1.16배). 값 8 + 색인 4 대 8 + 8바이트의 상한(1.33배)까지 가지 않는다: x를 모으는 무작위 읽기가 비용의 큰 몫이다.
- **10스레드의 메모리 대역이 다 쓰이는 연속 행렬(2차원 라플라시안 n = 10⁶)은 세 번 모두 1.3배를 넘었다(0.33 대 0.44 ms).** 사용자 결정(SP8)의 문턱에 걸려 물었고 int64를 그대로 둔다(SP19). 무작위 행렬의 10스레드는 1.30배 이하다.
- 비트: int32·int64, 1·10스레드의 결과가 모든 경우에 같고 scipy와도 같다(§7.4). 이 커널의 int64판(nnz 10⁶ 0.74 ms, 10⁷ 8.07 ms)은 scipy(0.645, 7.38 ms)의 0.87–0.91배 빠르기다(목표 0.8배 이상, §5).

### 7.6 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/sparse.py`, 등록함): 지금은 스윕이 없다. SP1의 변환은 정확한 연산이라(§4) 테스트가 비트까지 본다. 묶음은 그 함수가 공개되는 마일스톤에 들어간다(`spmv`·`sums`는 SP2, `solve`는 SP4, `lstsq`는 SP5, `eig`는 SP6, `norms`·`expm`은 SP7). 판정은 `tests/sparse/refdata.py`의 계약이고 비교 패키지 열은 scipy·R의 기록이다.
- **속도** (`bench/speed/cases/sparse.py`, 등록함): 사례마다 마일스톤과 필요한 sd.sparse 이름을 적어 두고, 그 마일스톤이 `DONE`에 들고 이름이 공개되면 사례가 된다(SP1의 형식 변환과 만들기 12가지, 크기마다 36개; 기준선 `bench/speed/results/sparse.json`, 표 `bench/sparse/speed.md`). 희소 결과는 값·(행 + 1)·(열 + 1)로 가중한 세 합으로 견준다(형식·순서·중복과 무관). 비교 대상은 scipy와 R Matrix(같은 R 세션), 크기는 §5다. 직접 풀이에 Accelerate를 직접 부른 값(감싸기 비용)을 더할지는 SP4에서 정한다.
- **골든** (`tests/golden/cases_sparse.py`, 기록 `tests/golden/data/sparse.json`): SP1의 812케이스(모든 입력 꼴, 네 형식 사이의 변환, 밀집, 전치, `reshape`·`resize`, `nonzero`·`count_nonzero`, `eliminate_zeros`·`sum_duplicates`, 값의 메서드, DOK 원소, 틀린 호출). 2×10⁵개의 무작위 COO는 3·10스레드에서 양동이 길을 탄다. 1·3·10스레드의 결과가 모두 같다. 공개 이름마다 그 마일스톤에서 더한다.
- **나중 일**(SP34): SP1의 속도는 다른 에이전트의 일과 함께 재서 † 줄(다른 프로세스가 1코어 넘게 쓴 사례)이 있다. 기계가 한가할 때 `.venv/bin/python -m bench.speed run --rerun-busy build/speed/<SP1 실행>.jsonl --promote`로 다시 잰다.

---

## 8. 마일스톤
| 마일스톤 | 내용 |
|---|---|
| SP0 준비 (완료·승인) | API 문서 초안(`docs/api/sparse.md`: 모든 공개 이름의 시그니처, scipy → sd 이름의 대응표, 마일스톤별 질문), 시험 행렬, 참값 도구(Arb), scipy·R 대조 도구와 데이터, Accelerate 탐침(§7), int32·int64 색인의 SpMV 측정(§2.2), 속도 기준선, 정확도·속도 체계에 넣는 계획 |
| SP1 형식과 변환 (완료·승인 2026-11-21) | `coo_array csr_array csc_array dok_array sparray issparse`, 생성자, 표준 꼴(§2.3), 형식 변환, `sd.ndarray`·numpy·scipy와 주고받기(§2.5), `astype copy .T transpose matrix_transpose conj real imag reshape resize nonzero eliminate_zeros sum_duplicates`, repr |
| SP2 연산 | 산술과 비교(§3.2), 원소별 함수, `multiply maximum minimum power`, SpMV·SpMM·밀집 @ 희소·SpGEMM, 리덕션(§3.3) |
| SP3 인덱싱과 만들기 | 인덱싱과 대입, `diagonal setdiag trace`, `eye_array diags_array random_array kron kronsum hstack vstack block_array block_diag tril triu find` |
| SP4 직접 풀이 | `spsolve splu factorized inv spsolve_triangular is_sptriangular spbandwidth`, `LinearOperator aslinearoperator`, `sd.linalg`·배열의 `TypeError`(§3.10) |
| SP5 반복법 | `cg gmres minres bicgstab lsqr lsmr` |
| SP6 고유값과 특이값 | `eigsh eigs svds`, linalg 래퍼에 `stev steqr hseqr trevc` |
| SP7 행렬 함수와 노름 | `expm expm_multiply onenormest norm` |
| SP8 최적화와 일관성 점검 | 모든 공개 함수를 크기마다 scipy·R과 견주어 고치기(SP1에서 넘어온 것: nnz 10⁵–10⁶의 1스레드 COO → CSR·CSR → CSC, SP33), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다. SP1–SP3은 차례로 간다(앞의 것에 기댄다). SP4가 `LinearOperator`를 만들므로 SP5–SP7은 그 뒤이고, 셋끼리는 순서를 바꿀 수 있다(SP7의 `norm(ord=2)`는 SP6의 `svds`를 쓴다). 기존 영역의 결과는 비트까지 같아야 한다(§3.10의 판별은 희소 입력에서만 동작한다).

**나중 후보** (첫 판 밖): BSR·DIA(§2.1), n차원 COO, `save_npz`·`load_npz`, `spilu`, 결정적인 희소 Cholesky와 LDLᵀ(§3.6의 나중 후보; 그러면 `cholesky` 같은 scipy에 없는 함수를 둘지도 묻는다), 희소 QR로 하는 직접 최소제곱, `lobpcg`, `bicg cgs qmr tfqmr lgmres gcrotmk`, `matrix_power`, `funm_multiply_krylov`, `LaplacianNd`. 그래프 알고리즘은 따로 된 영역이다(§1.2).

---

## 9. 마일스톤을 시작하기 전에 물을 것
추천을 첫째에 둔다.

- **SP0**: 정했다(SP6–SP20, §1.1).
- **SP1**: 정했다(SP21–SP34, §1.1).
- **SP2**
  1. 밀집 피연산자로 numpy 배열: `TypeError`(추천, API.md §5) / 받는다.
  2. 원소별 함수: 메서드만, `sd.sin(A)`는 메서드를 짚는 `TypeError`(추천) / `sd.sin(A)`도 받는다(f(0) = 0인 함수만).
  3. 축 리덕션의 결과: 언제나 밀집 `sd.ndarray`(추천) / scipy 그대로(`max(axis=)`는 COO).
  4. `A * 밀집`의 형식: A의 형식(추천) / scipy처럼 COO.
  5. `A + c`(c ≠ 0)·`A ** 0`: `ValueError`(추천) / 밀집 결과.
  6. 리덕션의 `keepdims`: 둔다(추천, 배열과 같다) / scipy처럼 없다. `explicit=`: scipy처럼 둔다(추천).
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
  4. 특이의 판정: 구조적 특이 + 유한하지 않은 해 + 분해의 조건수 추정(`onenormest`로 ‖A⁻¹‖₁을 어림해 κ₁·ε ≥ 1이면 `LinAlgError`)(추천; 정확히 특이한 행렬이 Accelerate에서 크기 1/ε의 유한한 해를 내기 때문, §7.5) / 구조적 특이 + 유한하지 않은 해만(scipy와 비슷하게 큰 유한한 해를 돌려준다).
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
