# 희소 영역 설계 (sd.sparse)

> 상태: **설계 초안** (2026-11-21). 큰 결정(§1.1 SP1–SP5)과 SP0–SP8의 결정(SP6–SP120; SP66–SP96·SP98–SP120은 추천안을 사용자가 위임, SP97은 사용자 승인)은 사용자가 정했다. 나머지는 안이다. SP0(준비)은 완료·승인, SP1(형식과 변환)은 완료·승인, SP2(연산)는 완료·승인, SP3(인덱싱과 만들기)은 완료·승인(2026-11-21), SP4(직접 풀이)는 완료·승인(위임, 2026-11-21), SP5(반복법)는 완료·승인(위임, 2026-11-21), SP6(고유값과 특이값)은 완료·승인(위임, 2026-11-21), SP7(행렬 함수와 노름)은 완료·승인(위임, 2026-11-22)이다. SP8(최적화와 일관성 점검)은 완료·승인(위임, 2026-11-22)이다(SP111–SP120은 추천안, 사용자 위임). 2026-11-22의 리뷰 고침이 SP121–SP129를 더했다(SP121–SP128은 추천안을 사용자가 위임, SP129는 사용자 결정). 2026-11-22의 성능 점검이 SP130–SP136을 더했다(추천안, 사용자 위임). 시작 전에 정할 것은 §9에 있다.

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
| SP28 데이터 인자의 이름 | scipy의 이름이다(`A`, `b`, `x`; 메시지는 `'A'`) (사용자 결정 2026-11-21). 예외: `block_diag`는 `arrays`(SP119) |
| SP29 변환의 `copy` | SP27을 변환 메서드까지 넓힌다: `tocoo`·`tocsr`·`tocsc`·`todok`·`asformat`·`reshape`·`to_scipy`의 기본은 `copy=None`이다(`[scipy와 다름]`: scipy는 `copy=False`) (사용자 결정 2026-11-21) |
| SP30 색인의 복사 | 받은 색인 배열(`indices`, `indptr`, `coords`)은 `copy`와 상관없이 언제나 복사하고 검사한다(받은 배열을 나중에 고쳐도 표준 꼴이 깨지지 않는다; §2.5). 2026-11-22에 다시 보고 지켰다(SP135) (사용자 결정 2026-11-21) |
| SP31 1차원 전치 | 1차원의 `.T`·`.mT`·`matrix_transpose`는 `sd.ndarray`처럼 `ValueError`다. `transpose()`는 자신이다(`[scipy와 다름]`: scipy의 1차원 `.T`는 자신; §3.5) (사용자 결정 2026-11-21) |
| SP32 `has_canonical_format` | 저장된 원소의 실제 상태(행 우선 정렬, 중복 없음)다. 처음 물을 때 알아내 둔다(`[scipy와 다름]`: scipy는 확인 없는 표시; §2.3) (사용자 결정 2026-11-21) |
| SP33 1스레드 변환 속도 | nnz 10⁵–10⁶의 1스레드 COO → CSR·CSR → CSC(scipy의 1.1–1.5배, 목표 2배)는 SP8의 고칠 목록에 넣는다(§5, §8) (사용자 결정 2026-11-21) |
| SP34 다시 재기 | 다른 프로세스가 바빴던 속도 사례(`bench/sparse/speed.md`의 † 줄)는 기계가 한가할 때 `--rerun-busy`로 다시 잰다. 나중 일이다(§7.6) (사용자 결정 2026-11-21) |
| SP35 numpy 피연산자 | 연산자의 밀집 피연산자는 `sd.ndarray`와 Python 스칼라(numpy의 스칼라 포함)다. numpy 배열은 `TypeError`다(API.md §5의 연산자 규칙; 연산자는 `NotImplemented`를 돌려주고 `==`·`!=`도 Python의 조용한 False가 아니라 `TypeError`다) (사용자 결정 2026-11-21) |
| SP36 원소별 함수 | 메서드로만 둔다(`A.sin()`). `sd.sin(A)` 같은 배열의 함수는 메서드(또는 연산자)를 짚는 `TypeError`다. f(0) = 0인 함수만이다(scipy의 메서드 목록, §3.2) (사용자 결정 2026-11-21) |
| SP37 축 리덕션의 결과 | `axis`가 있으면 언제나 밀집 `sd.ndarray`다(`[scipy와 다름]`: scipy의 `max(axis=)`·`min(axis=)`은 COO 희소) (사용자 결정 2026-11-21) |
| SP38 `A * 밀집`의 형식 | 희소 `*`(`multiply`)·`/` 밀집은 A의 형식을 지킨다(`[scipy와 다름]`: scipy는 COO) (사용자 결정 2026-11-21) |
| SP39 밀집이 될 연산 | `A + c`(c ≠ 0)와 `A ** 0`은 결과가 희소가 아니므로 `ValueError`다(`[scipy와 다름]`: scipy는 `NotImplementedError`) (사용자 결정 2026-11-21) |
| SP40 리덕션의 인자 | `keepdims`를 둔다(배열과 같은 뜻; scipy에는 없다). `explicit=`은 scipy처럼 둔다 (사용자 결정 2026-11-21) |
| SP41 브로드캐스트 | 결과가 A의 모양인 것만 받는다: 밀집 피연산자는 A의 모양으로 브로드캐스트되어야 하고, 희소끼리는 모양이 같아야 한다(`[scipy와 다름]`: scipy의 `multiply`는 결과가 커지는 꼴과 희소끼리의 브로드캐스트도 받는다; §3.2) (사용자 결정 2026-11-21) |
| SP42 희소끼리 연산의 결과 형식 | 왼쪽이 CSC면 CSC, 아니면 CSR이다(scipy의 규칙; `[scipy와 다름]`: scipy의 `dok + 희소`는 DOK) (사용자 결정 2026-11-21) |
| SP43 제자리 연산자 | `A *= c`, `A /= c` 등을 두지 않는다: Python이 `A = A * c`로 새 객체를 만든다. 제자리는 `A.data *= c`다(`[scipy와 다름]`: scipy는 값 배열을 제자리에서 바꾼다) (사용자 결정 2026-11-21) |
| SP44 `power`의 지수 | 음수와 NaN 지수는 `ValueError`다(0 ** n이 0이 아니다; `[scipy와 다름]`: scipy는 받고 암시적 0을 0으로 둔다) (사용자 결정 2026-11-21) |
| SP45 리덕션의 `axis` | int(음수 포함)나 None만 받는다(`count_nonzero`와 같다; 배열의 tuple·list 축은 받지 않는다) (사용자 결정 2026-11-21) |
| SP46 원소별 메서드의 이름 | 배열의 이름이다: `asin atan asinh atanh round`(`[scipy와 다름]`: scipy는 `arcsin arctan arcsinh arctanh rint`). 별칭은 없다(SP24). 같은 개념은 같은 이름(DESIGN.md §1.4, `trace(k=)`와 같은 원칙) (사용자 결정 2026-11-21) |
| SP47 양수와의 `maximum` | `maximum(A, c > 0)`·`minimum(A, c < 0)`(NaN 포함)은 scipy처럼 모든 자리를 저장한 A의 형식이다(`A <= 1`과 같은 꼴; 경고 없음) (사용자 결정 2026-11-21) |
| SP48 만들기 함수의 이름 | scipy의 배열 꼴 이름이다: `eye_array`, `diags_array`, `random_array`, `block_array`. 옛 함수의 짧은 이름(`eye`, `diags`, `random`, `bmat`)은 두지 않는다 (사용자 결정 2026-11-21) |
| SP49 `random_array`의 `rng` | 꼭 받는다. 시드 없는 기본값이 없고, 주지 않으면 `TypeError`다(라이브러리의 다른 무작위 함수와 같다: optimize O31, htest) (사용자 결정 2026-11-21) |
| SP50 CSR·CSC의 새 자리 대입 | 받는다. 새 자리가 있으면 배열을 다시 만들어 O(nnz)이고 경고는 없다(이미 저장된 자리는 값만 바꾼다) (사용자 결정 2026-11-21) |
| SP51 `format=None`의 결과 형식 | scipy의 규칙이되 DIA·BSR이 나올 자리는 CSR이다: `eye_array`·`diags_array`·`kron`은 CSR, `kronsum`은 CSR, `block_array`·`block_diag`·`tril`·`triu`·`random_array`는 COO(쌓기는 블록이 모두 CSR이면 CSR, 모두 CSC면 CSC) (사용자 결정 2026-11-21) |
| SP52 정수 대각의 `diags_array` | int64다(배열의 dtype 규칙: 대각들의 승격; `[scipy와 다름]`: scipy 1.18.1의 기본은 float64와 `FutureWarning`) (사용자 결정 2026-11-21) |
| SP53 대입한 0의 저장 | 형식마다 scipy의 규칙이다(SP11의 원칙): CSR·CSC는 대입한 0도 저장하고, DOK는 그 키를 지우고, COO는 배열 대입의 0을 빼되 원소 하나의 대입과 희소 값의 저장된 원소는 0이어도 붙인다(§3.3) (사용자 결정 2026-11-21) |
| SP54 CSR·CSC의 원소 고르기 | `A[rows, cols]`와 2차원 마스크의 결과는 COO 희소 배열이다(scipy의 COO·DOK와 같다; `[scipy와 다름]`: scipy의 CSR·CSC는 밀집 배열) (사용자 결정 2026-11-21) |
| SP55 `trace()`의 dtype | bool·int64는 int64다(`sd.sum`, scipy와 같다; `dtype=`은 두지 않는다) (사용자 결정 2026-11-21) |
| SP56 bool의 `kron` | bool × bool은 int64다(배열의 `multiply`, 희소 `A * B`와 같다; `[scipy와 다름]`: scipy는 bool) (사용자 결정 2026-11-21) |
| SP57 `diags_array`의 대각 | 대각 하나하나는 `sd.ndarray`나 수다(API.md §5의 배열 함수 입력 규칙; 수의 리스트는 `TypeError`) (사용자 결정 2026-11-21) |
| SP58 `random_array`의 `data_sampler` | `data_sampler(n)`으로 위치 인자 하나로 부르고, `dtype=None`이면 그 결과의 dtype을 둔다(`[scipy와 다름]`: scipy는 `data_sampler(size=n)`과 float64) (사용자 결정 2026-11-21) |
| SP59 LU의 순서 | AMD로 고정한다. `ordering=` 옵션은 없다(Metis는 동시 호출에서 비트가 다르고 MTMetis는 동시 호출에서 프로세스가 죽는다, §1.4) (사용자 결정 2026-11-21) |
| SP60 `splu`의 결과 | 내보내지 않는 `SparseLU`다. `solve`와 `shape`만 있고 `L`·`U`·`perm_r`·`perm_c`·`nnz`는 없다(Accelerate가 인수를 내보이지 않는다) (사용자 결정 2026-11-21) |
| SP61 scipy의 옵션 | `permc_spec`, `diag_pivot_thresh`, `use_umfpack`, `relax`, `panel_size`, `options`, `overwrite_A`, `overwrite_b`를 받지 않는다(그런 키워드가 없어 `unexpected keyword argument`의 `TypeError`다) (사용자 결정 2026-11-21) |
| SP62 특이의 판정 | 구조적 특이(Accelerate의 `SparseMatrixIsSingular`), 유한하지 않은 해, 분해의 조건수 추정(‖A‖₁ × `onenormest` 꼴로 어림한 ‖A⁻¹‖₁ = κ₁, κ₁·ε ≥ 1이면 특이) 셋이 모두 `LinAlgError("<함수>(): singular matrix")`다. 정확히 특이한 행렬이 Accelerate에서 크기 1/ε의 유한한 해를 내기 때문이다(§1.4, §3.6) (사용자 결정 2026-11-21) |
| SP63 반복 다듬기 | 잰 역오차가 계약을 넘는 행렬이 있으면 한 번 넣는다. SP4에서 쟀고 saddle/600x150이 404ε이라 모든 풀이에 한 번 넣었다(§3.6의 측정). SP133에서 첫 풀이의 성분별 역오차가 8ε를 넘을 때만으로 바꿨다 (사용자 결정 2026-11-21) |
| SP64 우변이 여럿 | Accelerate가 함께 푼다. 결과는 열 수 k와 값으로 정해지고 결정적이다. 열 하나의 비트는 혼자 푼 것과 다를 수 있다. SP134부터 8열씩의 고정 묶음으로 푼다 (사용자 결정 2026-11-21) |
| SP65 `LinearOperator`의 위치 인자 | `shape`, `matvec`만 위치다. `rmatvec`, `matmat`, `rmatmat`, `dtype`은 키워드다(API 초안대로) (사용자 결정 2026-11-21) |
| SP66 κ₁·ε ≥ 1의 판정 | 그대로 둔다: SuiteSparse의 sherman3(κ₁ 6.9×10¹⁶)·cvxqp3(2.2×10¹⁶)도 `LinAlgError`다. scipy는 이 행렬에서 해를 돌려주지만 앞오차가 κε ≥ 1이라 맞는 자리가 없는 해다(§3.6, §7.2) (추천안, 사용자 위임 2026-11-21) |
| SP67 배열의 함수 | 배열의 함수(`sd.matmul`, `sd.sum` 등)도 희소 배열을 받으면 쓸 희소 메서드·연산자·함수를 짚는 `TypeError`다. 배열 영역의 한 곳(`csrc/bind/pyutil.cpp`의 `require_ndarray`와 표 `sparse_hint`, 이름을 붙이는 `csrc/bind/errors.cpp`의 `rethrow_named`)에서 실패한 호출에만 한다(§3.10) (추천안, 사용자 위임 2026-11-21) |
| SP68 `spsolve_triangular`의 `upper` | `sd.linalg.solve_triangular`처럼 꼭 주는 키워드 `upper`를 받는다. `lower`는 없다(`[scipy와 다름]`: scipy는 `lower=True`가 기본) (추천안, 사용자 위임 2026-11-21) |
| SP69 (n, 1)의 밀집 b | 결과도 (n, 1)이다(`sd.linalg.solve`와 같다; `[scipy와 다름]`: scipy는 (n,)) (추천안, 사용자 위임 2026-11-21) |
| SP70 열 하나·1차원의 희소 b | 결과도 희소다(1차원은 CSR; `[scipy와 다름]`: scipy는 밀집) (추천안, 사용자 위임 2026-11-21) |
| SP71 `rmatvec` 없는 연산자 | `rmatvec`은 `ValueError`(`rmatvec(): the operator was made without 'rmatvec'`)다(`[scipy와 다름]`: scipy는 `NotImplementedError`) (추천안, 사용자 위임 2026-11-21) |
| SP72 반복법의 결과 | 함수마다 읽기 전용 결과 객체(`CgResult` … `LsmrResult`, 내보내지 않는다)이고 scipy의 튜플로 풀린다: `x, info = cg(A, b)`, `lsqr`은 열, `lsmr`은 여덟 값. `summary()`와 한 줄 repr은 API.md §2.3의 꼴이다 (추천안, 사용자 위임 2026-11-21) |
| SP73 반복 상한의 이름 | `max_iter`다(optimize O14). scipy의 `maxiter`, `lsqr`의 `iter_lim`은 없다 (추천안, 사용자 위임 2026-11-21) |
| SP74 `callback` | 두지 않는다(optimize O37·O54). `callback_type`, `show`, `check`, `calc_var`도 없다 (추천안, 사용자 위임 2026-11-21) |
| SP75 `gmres`의 `restart`·`max_iter` | scipy 1.18.1 그대로다: `restart`는 한 바깥 반복의 안쪽 반복 수(기본 20, n에서 자른다), `max_iter`는 바깥 반복 수(기본 10n)이고 상한에 닿으면 `status = max_iter`다 (추천안, 사용자 위임 2026-11-21) |
| SP76 허용오차와 인자의 검사 | 이름과 기본값은 scipy의 것이다(`rtol=1e-5`, `atol=0.0`; `minres`는 `rtol`만; `lsqr`·`lsmr`은 `atol=btol=1e-6`, `conlim=1e8`, `damp=0`). `rtol`·`atol`·`btol`·`damp`는 유한한 0 이상의 수, `conlim`은 0 이상(inf 포함, 0과 inf는 한계 없음), `shift`는 유한한 수, `max_iter`·`restart`는 None이나 1 이상의 int다. 메시지는 sd.optimize의 것이다. 이유: 영역 간 같은 검사(optimize `_check_tol`)이고, scipy가 받는 음수 rtol과 `maxiter=0`(scipy는 `info = 0`, 곧 "수렴"을 돌려준다)은 뜻 없는 호출이다(`[scipy와 다름]`) (추천안, 사용자 위임 2026-11-21) |
| SP77 입력 | `A`·`M`은 희소 배열(모든 형식), 2차원 `sd.ndarray`, `LinearOperator`다. `b`·`x0`은 밀집 `sd.ndarray`((n,)·(n, 1); 리스트·numpy 배열은 `TypeError`, `spsolve`와 같다)이고 `x`는 언제나 (n,)이다(scipy와 같다; `spsolve`의 SP69와 달리 scipy의 꼴을 지킨다). scipy의 `x0='Mb'`는 없다. 저장된 NaN·inf는 `ValueError`다(§1.6의 영역 규칙). 이유: 희소 영역의 데이터 입력 규칙(SP4)과 같고, 결과의 꼴은 scipy 코드가 기대는 비반올림 동작이다 (추천안, 사용자 위임 2026-11-21) |
| SP78 계의 dtype | A, b, x0, M 가운데 하나라도 복소면 complex128, 아니면 float64다(bool·int64는 float64). `[scipy와 다름]`: scipy는 A와 b로만 정해 복소 x0의 허수부를 경고와 함께 버리고, 실수 계의 복소 M에서 실패한다. 이유: 값을 몰래 버리지 않는다(경고 없음 원칙) (추천안, 사용자 위임 2026-11-21) |
| SP79 결과의 필드 | 필드 이름은 regression·optimize의 것이다: `x`, `converged`, `status`(scipy의 `info`·`istop`), `message`, `n_iter`, `normr`; `gmres`는 `n_iter`(바깥 반복, `max_iter`의 단위)와 `n_inner`(안쪽 반복); `minres`·`lsqr`·`lsmr`은 `norma`, `conda`, `normar`, `normx`(lsmr의 이름; lsqr의 `anorm`·`acond`·`arnorm`·`xnorm`도 이 이름), `lsqr`은 `normr_damped`(scipy `r2norm`)와 `var`. `converged`는 `cg`·`bicgstab`·`gmres`·`minres`에서 `status == 0`, `lsqr`·`lsmr`에서 istop ∈ {0, 1, 2, 4, 5}다. `bicgstab`의 반 단계 멈춤은 `n_iter`가 센다 (추천안, 사용자 위임 2026-11-21) |
| SP80 연산자와 결정성 | 행렬(희소, 밀집, `aslinearoperator`로 감싼 것)은 반복 전체를 C++에서 돌린다: 곱은 §2.6의 SpMV(밀집 행렬은 모든 원소를 저장한 CSR), 내적은 `sd.linalg.vecdot`의 합, 노름은 `sd.linalg.vector_norm`이고, 원소별 갱신은 실수 fma·복소 `cmul`이다. 비트는 스레드 수, 형식, 연산자의 꼴과 무관하다. 다른 `LinearOperator`는 곱마다 Python 호출(GIL을 쥔다)이다. Accelerate의 반복법은 쓰지 않는다(SP16; 잰 값은 §3.7) (추천안, 사용자 위임 2026-11-21) |
| SP81 `minres`의 실패 | 양정치가 아닌 전처리기와 βₖ < 0은 `LinAlgError`(`minres(): the preconditioner is not positive definite`; βₖ < 0의 메시지는 SP126)다(`[scipy와 다름]`: scipy는 `ValueError`; `LinAlgError`는 그 하위 클래스다). 이유: 행렬의 성질이 맞지 않는 실패는 linalg와 같은 예외다(`cholesky`의 양정치) (추천안, 사용자 위임 2026-11-21) |
| SP82 `lsqr`의 `var` | `calc_var`이 없으므로(SP74) `var`를 언제나 계산한다(scipy의 `calc_var=True`와 같은 값; 반복마다 n번의 곱셈과 덧셈). 복소 행렬은 \|dₖ\|²를 더한다(`[scipy와 다름]`: scipy는 dₖ²를 더해 복소 `var`를 낸다). 이유: 결과가 scipy의 열 값으로 풀려야 하고(SP72), 늘 0인 필드는 뜻이 없다 (추천안, 사용자 위임 2026-11-21) |
| SP83 고유값의 알고리즘 | ARPACK(implicitly restarted Lanczos·Arnoldi)을 scipy 1.18.1이 돌리는 계산 순서까지 옮긴다(`dsaupd`·`dseupd`, `dnaupd`·`dneupd`, `znaupd`·`zneupd`). 바탕은 이 라이브러리의 SpMV·`LinearOperator`와 결정적인 내적·노름이고 스레드 수와 무관하게 같은 비트다(§3.8). Krylov–Schur는 쓰지 않는다. OP 수가 scipy와 같다는 약속은 SP130에서 scipy 자신의 퍼짐 안으로 바꿨다 (추천안, 사용자 위임 2026-11-21) |
| SP84 시작 벡터 | `v0=None`이면 정해진 벡터다: u = 2·g.random(n) − 1, g = `sd.random.default_rng(0)`(복소 문제는 연속한 두 값이 실수부·허수부; scipy의 [−1, 1) 균등과 같은 꼴). `rng=`를 주면 그 생성기에서 뽑는다. ARPACK이 불변 부분공간에서 새 벡터를 청하면 같은 생성기에서 뽑는다. 시드 없는 무작위는 없다(`[scipy와 다름]`) (추천안, 사용자 위임 2026-11-21) |
| SP85 `svds` | Golub–Kahan–Lanczos 이중대각화의 두꺼운 재시작 하나(Baglama–Reichel 2005, R `irlba`의 방법)이고 `solver=`·`options`는 없다(`[scipy와 다름]`: scipy의 기본 `'arpack'`은 AᴴA의 `eigsh`라 작은 특이값의 정확도를 잃는다; §3.8의 rect/80x30d) (추천안, 사용자 위임 2026-11-21) |
| SP86 `eigsh`의 `mode` | `'normal'`, `'buckling'`, `'cayley'`를 모두 둔다(ARPACK의 모드 3, 4, 5; scipy와 같다) (추천안, 사용자 위임 2026-11-21) |
| SP87 수렴 실패 | `LinAlgError`가 아니라 결과 객체의 상태다: `converged`, `status`(ARPACK의 info: 0, 1 반복 상한, 3 이동을 쓸 수 없음, −9999 분해를 늘릴 수 없음), `message`, 그리고 수렴한 쌍(scipy의 `ArpackNoConvergence` 대신). `summary()`와 한 줄 repr이 있다(API.md §2.3) (추천안, 사용자 위임 2026-11-21) |
| SP88 인자의 이름 | scipy의 이름이다(SP28): `k`, `M`, `sigma`, `which`, `v0`, `ncv`, `tol`, `return_eigenvectors`, `Minv`, `OPinv`, `mode`, `OPpart`, `return_singular_vectors`, `rng`. 반복 상한만 `max_iter`다(SP73; scipy의 `maxiter`). 이유: 데이터·옵션 이름은 scipy를 따르고 상한의 이름은 모든 영역이 같다 (추천안, 사용자 위임 2026-11-21) |
| SP89 `tol`과 `max_iter` | `tol`은 유한한 0 이상의 수이고 0이면 ε = 2⁻⁵²다(scipy의 ARPACK이 쓰는 값: 2⁻⁵³(`dlamch('E')`)로는 OP 곱의 수가 scipy와 갈라졌다, §3.8). `max_iter`는 None이나 1 이상의 int, 기본은 10n(`svds`는 10·min(m, n))이다. 이유: SP76의 검사 규칙, scipy의 기본값 (추천안, 사용자 위임 2026-11-21) |
| SP90 결과 객체 | 함수마다 읽기 전용 결과 객체(`EigshResult`, `EigsResult`, `SvdsResult`, 내보내지 않는다)이고 scipy의 값으로 풀린다: (w, v), (u, s, vh). 필드는 `sd.linalg`의 이름(`eigenvalues`, `eigenvectors`; `U`, `S`, `Vh`)과 `converged`, `status`, `message`, `n_iter`, `n_converged`, `n_op`이다. `return_eigenvectors=False`·`return_singular_vectors`가 계산하지 않은 쪽은 None이고 결과의 꼴은 그대로다(`[scipy와 다름]`: scipy는 배열 하나). 이유: 결과의 꼴은 옵션으로 바뀌지 않는다(docs/api/linalg.md §1.6), 풀기는 SP72와 같다 (추천안, 사용자 위임 2026-11-21) |
| SP91 고유벡터의 정규화와 순서 | 부호와 위상은 정하지 않는다(ARPACK이 내는 그대로; `sd.linalg`의 `eigh`·`eig`·`svd`와 같다). 값의 순서는 scipy의 것이다: `eigsh` 오름차순(정규 모드의 `return_eigenvectors=False`는 ARPACK의 `which` 순서), `eigs` ARPACK의 순서, `svds` 오름차순. 이유: 영역 간 같은 규칙, 순서는 반올림이 아닌 동작(규칙 2) (추천안, 사용자 위임 2026-11-21) |
| SP92 `k`와 `ncv` | scipy의 범위다(`eigsh` 1 ≤ k < n, k < ncv ≤ n; `eigs` 1 ≤ k < n − 1, k + 1 < ncv ≤ n; `svds` 1 ≤ k < min(m, n), k < ncv ≤ min(m, n)). k가 넘치면 `ValueError`이고 메시지가 `sd.linalg.eigh(A.toarray())`(`eig`)를 짚는다(`[scipy와 다름]`: scipy는 경고와 함께 밀집 풀이로 넘어간다). 이유: 경고를 내지 않고(§1.3) 밀집으로 몰래 바뀌지 않는다(SP26) (추천안, 사용자 위임 2026-11-21) |
| SP93 `sigma` 없는 `mode` | `sigma` 없이 `'normal'`이 아닌 `mode`는 `ValueError`다(`[scipy와 다름]`: scipy는 조용히 무시하고 정규 모드로 푼다). `Minv`·`OPinv`·`OPpart`의 짝 규칙은 scipy와 같다. 이유: 뜻 없는 인자를 몰래 버리지 않는다 (추천안, 사용자 위임 2026-11-21) |
| SP94 shift-invert의 역 | (A − σM)⁻¹, M⁻¹는 SP4의 Accelerate LU/AMD(밀집 행렬은 0이 아닌 원소의 희소 배열로)와 반복 다듬기이고, σ가 고유값이면 `LinAlgError("eigsh(): singular matrix")`(SP62)다. A·M이 일반 `LinearOperator`면 scipy처럼 `gmres`(rtol max(tol, 1000√n·ε))로 푼다. 이유: 직접 풀이의 한 바탕(SP18)과 같은 특이 판정. LU의 풀이는 SP132부터 다듬지 않는다 (추천안, 사용자 위임 2026-11-21) |
| SP95 고유값의 계약 | `eigsh`·`svds`의 값은 Arb와 C_EIG·ε·‖A‖₂ 안이다(C_EIG = 1000: 잰 가장 큰 것은 ARPACK이 762번 재시작한 lap1d/1000 'LA'의 497, scipy 217). `eigs`는 그것에 고유값의 조건수 κ(λ)를 곱한다. 잔차와 직교성도 같은 꼴이다(§4) (추천안, 사용자 위임 2026-11-21) |
| SP96 `svds`의 판정과 재시작 | 특이 삼중쌍은 ‖Aᴴu − σv‖ ≤ tol·max(σ, ε^(2/3)σ₁)이면 수렴한다(ARPACK의 판정을 특이값에; σ₁은 가장 큰 Ritz 특이값). 재시작은 원하는 쪽의 k + min(수렴 수, (ncv − k)/2)개를 남기고(ARPACK `dsaup2`의 nev 조정, 1이면 ncv/2), 새 벡터의 재직교화는 DGKS 판정(0.717)이다. `v0`는 min(m, n)개(scipy의 `'arpack'`과 같다) (추천안, 사용자 위임 2026-11-21) |
| SP97 R의 비교 대상 | R의 `RSpectra` 0.16.2(Spectra의 재시작 Lanczos·Arnoldi)와 `irlba` 2.4.1(Baglama–Reichel)을 비교 대상으로 둔다: 기록 `tests/sparse/data/r_eigen.json`(`tools/gen_sparse_r.py --eigen`)과 정확도 스윕의 R 열 (사용자 승인 2026-11-21, 설치함) |
| SP98 `onenormest`의 무작위 열 | 호출마다 `sd.random.default_rng(0)`(고정 키 0)에서 뽑아 결정적이다. `rng` 인자는 없다(`[scipy와 다름]`: scipy는 numpy의 시드 없는 전역 난수라 실행마다 추정값이 다를 수 있다). 이유: 같은 호출은 같은 결과(SP84의 `v0`와 같은 규칙) (추천안, 사용자 위임 2026-11-21) |
| SP99 `expm`의 결과 형식 | 입력의 형식이다(CSR → CSR, CSC → CSC, COO → COO, DOK → DOK). `[scipy와 다름]`: scipy의 형식은 경로마다 다르다(CSR → CSR, CSC → CSC, COO → COO나 CSR, DOK → CSR). 이유: 입력과 같은 꼴(SP38의 `A * 밀집`과 같은 원칙) (추천안, 사용자 위임 2026-11-21) |
| SP100 `onenormest`의 반복 상한 | scipy의 `itmax`가 아니라 `max_iter`다(SP73, optimize O14). 2 이상의 int이고 아니면 `ValueError`(`'max_iter' must be >= 2`; scipy의 "at least two iterations are required") (추천안, 사용자 위임 2026-11-21) |
| SP101 `norm`의 `'nuc'`·`-2` | 행렬의 `ord='nuc'`·`-2`는 받지 않는다: `norm(): 'ord' must be 'fro', 1, -1, 2, inf or -inf, not 'nuc'`(`ValueError`; linalg의 목록에서 둘을 뺀 것). 이유: 모든 특이값이 필요해 희소의 뜻이 없고 scipy에도 없다(`-2`는 scipy의 `NotImplementedError`) (추천안, 사용자 위임 2026-11-21) |
| SP102 `onenormest`의 결과 | 옵션과 상관없이 언제나 읽기 전용 결과 객체 하나(`OnenormestResult`, 내보내지 않는다; SP72·SP90의 결과와 같은 바탕)이고 `(est, v, w)`로 풀린다. 필드는 `est`, `v`, `w`(`compute_v`·`compute_w`가 아니면 None), `n_iter`(반복 수; t ≥ n이면 0), `n_op`(A·X와 Aᴴ·S의 곱 수), `summary()`와 한 줄 repr은 API.md §2.3의 꼴이다. `[scipy와 다름]`: scipy는 기본으로 float 하나를 돌려주고 `(est, v)`·`(est, w)`·`(est, v, w)`로 꼴이 바뀐다. `t`는 1 이상의 int(t ≥ n이면 명시한 행렬의 정확한 노름, scipy와 같다), A는 희소·2차원 `sd.ndarray`·`LinearOperator`(SP77과 같은 꼴, 저장된 NaN·inf는 `ValueError`), 0 × 0은 0.0. 이유: 결과의 꼴은 옵션에 따라 바뀌지 않는다(docs/api/linalg.md §1.6, SP90의 `return_eigenvectors=False`와 같다); 처음의 안(벡터를 묻지 않으면 float)은 그 규칙을 어겨 고쳤다 (추천안, 사용자 위임 2026-11-21; 2026-11-22 고침) |
| SP103 `expm`의 알고리즘 | 밀집 `sd.linalg.expm`의 선택을 따른다(linalg.md §11): 노름은 정확한 값을 쓰되 아는 노름의 곱으로 만든 상한을 먼저 보고, ‖A⁸‖·‖A¹⁰‖은 n ≥ 100(linalg의 `kExpmEstimateMin`)부터 `onenormest`(t = 2)로 어림한다. ℓ은 정확(\|A\|ᵀ의 SpMV 2m + 1번, 2⁻ᵉ 척도). 결과는 I + 2(V − U)⁻¹U, 대각 행렬은 exp(대각) 그대로, 위·아래 삼각은 모든 차수와 제곱 뒤에 대각과 첫 부대각을 정확한 값으로 다시 쓴다, ‖A‖₁ > 2¹⁰⁰이면 먼저 줄인다. 곱은 SpGEMM, 풀이는 SP4의 LU(다듬기 한 번, 특이 판정 SP62). `[scipy와 다름]`: scipy는 n ≥ 200에서 dₖ의 대부분을 무작위 추정으로 하고, 삼각의 고침은 위 삼각·차수 13에만 한다. 이유: 같은 개념은 같은 알고리즘·계약(SP14·DESIGN §1.4); 잰 오차가 scipy의 절반(§4) (추천안, 사용자 위임 2026-11-21) |
| SP104 `expm`의 입력 | 희소 배열만(밀집은 `sd.linalg.expm`을 짚는 `TypeError`, `inv`와 같다), bool·int64는 float64, 저장된 NaN·inf는 `ValueError`, 저장된 0은 무시하고 결과의 정확한 0은 뺀다(`inv`와 같다). 모든 형식을 먼저 CSR로 바꾸므로 형식과 상관없이 같은 비트다 (추천안, 사용자 위임 2026-11-21) |
| SP105 `expm_multiply`의 시간 구간 | scipy의 `start`·`stop`·`num`·`endpoint` 그대로다: 넷 중 하나라도 주면 구간이고 `start`·`stop`은 둘 다 있어야 한다(유한한 실수; 없으면 `ValueError`), `num`은 2 이상의 int(기본 50), `endpoint`는 bool(기본 True); 점은 numpy `linspace`의 것, 결과는 (num,) + B.shape다. 허용오차는 scipy처럼 2⁻⁵³ 고정(인자 없음)이고 (m*, s) 선택과 세 구간 경로(q ≤ s, q가 s의 배수, 나머지)도 scipy의 것이다. 이유: scipy의 이름·뜻을 따른다(SP28) (추천안, 사용자 위임 2026-11-21) |
| SP106 `expm_multiply`의 B | 밀집 `sd.ndarray` (n,)·(n, k)만 받는다(희소 B는 `B.toarray()`를 짚는 `TypeError`; `[scipy와 다름]`: scipy는 희소 B도 받는다). bool·int64는 float64, NaN·inf는 `ValueError`, A·B·`traceA` 가운데 복소가 있으면 complex128이다. 이유: exp(A)B는 대개 밀집이고, 반복법의 b와 같은 규칙(SP77) (추천안, 사용자 위임 2026-11-21) |
| SP107 `expm_multiply`의 `traceA` | 수(int·float·complex, bool은 `TypeError`)나 None. None이면 행렬은 정확한 대각합, 일반 `LinearOperator`는 scipy의 Hutch++ 추정(`traceest`, 단순 호출 m3 = 1, 구간 5)을 `sd.random.default_rng(0)`의 고정된 ±1 열로 한다. `[scipy와 다름]`: scipy는 경고를 내고 시드 없는 무작위로 추정한다. 이유: 대각합은 A를 μI만큼 옮기는 데만 쓰인다(먼 값의 정확도는 SP129); 경고 없음(§1.3)과 SP98의 고정 열 (추천안, 사용자 위임 2026-11-21) |
| SP108 `norm`의 세부 | `ord`는 키워드만. 2차원에 축 둘(None 포함)이면 행렬 노름, 1차원이나 축 하나면 벡터 노름(`ord` None = 2, inf, −inf, 0, 그 밖의 모든 p(음수 포함); `sd.linalg.vector_norm`의 뜻: 암시적 0도 원소이고 `ord=0`은 0이 아닌 원소의 수(float), 넘침·비정규 수에서 줄의 최댓값으로 다시 계산). `axis`는 int, 축 1–2개의 tuple·list(vector_norm의 규칙과 메시지; `()`은 `ValueError`). 결과: 행렬·1차원은 float, 축 하나면 밀집 float64. `'fro'`는 표준 꼴 값의 `vector_norm`, `ord=2`는 `svds(k=1)`의 값(행·열이 하나면 그 벡터의 2-노름, 0인 행렬은 0, 저장된 NaN·inf는 `ValueError`, 수렴하지 못하면 `LinAlgError`). `[scipy와 다름]`: scipy의 `'f'` 별칭, 위치 인자 `ord`, `ord=0`의 int, 음수 p의 틀린 값(저장된 원소만 거듭제곱), 중복 있는 COO의 `'fro'`(저장된 값을 그대로)는 따르지 않는다. 이유: SP14(규칙과 메시지는 linalg의 것) (추천안, 사용자 위임 2026-11-21) |
| SP109 SP7의 계약 | `expm`·`expm_multiply`는 max\|y − ref\| ≤ C_EXPM·ε·max(1, ‖sA‖₁)·척도(`expm`: max\|exp(sA)\|, `expm_multiply`: maxᵢ(\|exp(tsA)\|\|b\|)ᵢ)이고 C_EXPM = 10(잰 가장 큰 것 2.4; scipy 4.8, 밀집 `sd.linalg.expm` 3.5). 합으로 된 노름은 \|v − ref\| ≤ (γ_m + 2ε)·ref(m은 가장 긴 줄의 항 수, Frobenius는 저장된 원소 수), `ord=2`는 `svds`의 C_EIG·ε·‖A‖₂(§4) (추천안, 사용자 위임 2026-11-21) |
| SP110 SP7의 층 | 네 함수는 Python(`src/stardust/sparse/_matfuncs.py`)이 이 라이브러리의 커널(SpMV·SpMM, SpGEMM, SP4의 LU, §2.6의 리덕션)을 부른다. 설계 초안이 C++에 두려던 matfuncs 파일은 만들지 않았다. 이유: 일이 모두 기존 커널에 있고 호출 하나의 Python 비용(0.3–2 µs)이 곱 하나보다 작다; smoke에서 scipy와 같거나 빠르다(§5) (추천안, 사용자 위임 2026-11-21) |
| SP111 `kronsum`의 값 | 단위 행렬과 곱하지 않고 값을 놓는다: 대각 블록마다 A의 원소, 블록 (i, j)의 대각에 bᵢⱼ, 겹치는 자리(aₚₚ와 bᵢᵢ)는 그 합 하나(A 쪽이 먼저)이고 만든 정확한 0은 지운다. 그래서 `sd.linalg.kronsum`과 0의 부호만 빼고 비트까지 같다. 실수·정수 입력의 비트는 그대로이고, complex에서 0 + inf·j가 NaN + inf·j로, 한쪽만 있는 자리의 허수부 −0.0이 +0.0으로 바뀌던 것이 고쳐진다(골든은 새 케이스 `kronsum complex special` 하나). 이유: 밀집 짝과 같은 값(API.md §7의 부분집합 규칙; 같은 개념은 같은 결과) (추천안, 사용자 위임 2026-11-21) |
| SP112 `random_array`의 `rng` | 기본값이 없는 키워드다(`rng=None`이 아니다): 없으면 `random_array(): missing 1 required keyword-only argument: 'rng'`, None을 주면 지금의 `'rng' must be an sd.random.Generator … not NoneType`(둘 다 `TypeError`; SP49의 뜻 그대로). 이유: 언제나 필요한 `rng`는 htest의 `monte_carlo_test`·`bootstrap`, 분포의 `sample`처럼 기본값이 없다; None 기본값은 쓰지 않을 때도 있는 함수(`check_grad`, `eigsh`)의 꼴이다. 골든의 `error random_array rng` 메시지가 바뀐다 (추천안, 사용자 위임 2026-11-21) |
| SP113 결과의 repr | 희소의 모든 결과 객체가 같은 한 줄 꼴이다(API.md §2.3, `_summary.compact`): 6개까지의 실수 1차원 배열은 값을 적는다(`EigshResult(…, eigenvalues=[58.0469, 60.4509])`, `SvdsResult`의 `S`, `OnenormestResult`의 `v`·`w`). 복소 배열은 모양이다(compact가 실수만 적는다). 전에는 고유값·특이값 결과와 `onenormest`만 짧은 배열도 모양으로 적었다. 골든에는 repr가 없어 그대로다 (추천안, 사용자 위임 2026-11-21) |
| SP114 `count_nonzero`의 `keepdims` | `count_nonzero(*, axis=None, keepdims=False)`: `sd.count_nonzero`와 희소의 다른 리덕션(SP40)과 같은 뜻이다(`axis=None`이면 모든 축이 1인 밀집 int64 배열). scipy에는 없다(희소가 밀집의 부분집합이 되게 하는 더하기) (추천안, 사용자 위임 2026-11-21) |
| SP115 `spbandwidth`의 결과와 밀집 입력 | `spbandwidth`는 `sd.linalg.bandwidth`와 같은 `BandwidthResult(lower, upper)`(tuple이라 scipy의 `(below, above)`로 풀린다)를 돌려준다. `spbandwidth`·`is_sptriangular`에 밀집 배열을 주면 `spsolve` → `sd.linalg.solve`처럼 `sd.linalg.bandwidth`를 짚는다. 골든의 `spbandwidth` 넷은 결과의 형(tuple → BandwidthResult)만 바뀐다 (추천안, 사용자 위임 2026-11-21) |
| SP116 SP8의 속도 고침 | 비트를 바꾸지 않는 길만 고쳤다(골든 7,295케이스의 값 그대로): 1스레드 COO → CSR의 LSD 길(최소 색인으로 안정 계수 정렬 뒤 주 색인으로 안정 계수 정렬, 중복은 도착할 때 더함, 2²¹개까지)과 짧은 줄(16개 이하)의 순위 정렬(NEON), 1스레드 전치의 열 블록 길(2¹⁵개부터), 흩뿌리기(`x @ A`, CSC `@`, 열 합, `norm(ord=1)`)의 출력 블록 병렬(줄 구간마다 블록으로 나눈 항을 블록마다 줄 순서로 더함), 전치된 결과의 흩뿌리기(`X @ A`)는 연속한 작업 배열에, SpMM의 x 행 미리 읽기, SpGEMM의 행 정렬 대신 요약 비트가 있는 비트 집합(요약이 짧을 때), 정렬되지 않은·정렬된 색인 배열 고르기의 바이트 표, Lanczos·Arnoldi·`svds`의 여러 열 내적(잎마다 모든 열, 실수는 두 열씩)과 갱신의 작은 행 구간, `gmres`의 MGS에서 갱신과 다음 내적을 한 번에 훑기, `.T`와 값만 바꾼 배열이 긴 자리 판정(`_minor_over`)을 함께 둠. 잰 값은 §5의 SP8 표. 여러 열 내적, 작은 행 구간의 갱신, `gmres`의 한 번 훑기는 SP131에서 Accelerate `dgemv`와 linalg의 `fastsum`으로 바꿨다 (추천안, 사용자 위임 2026-11-21) |
| SP117 SP8에서 고치지 않은 것 | (1) `SparseLU.solve`의 작은 행렬: 시간의 97%가 Accelerate 풀이 두 번(다듬기, SP63)이라 다듬기를 빼야 줄고, 그러면 비트와 역오차가 바뀐다(saddle/600x150의 404ε) — 그대로 둔다. (2) n = 10⁴의 `eigsh`·`eigs`: 기저의 내적·갱신이 메모리 대역에 묶여 있고 scipy의 Accelerate `dgemv`(AMX)가 7배 빠르다; 결정적인 이쪽 합을 BLAS로 바꾸면 비트가 바뀐다. (3) `norm(ord=2)`의 1스레드(`svds`의 GKL, tol = 0). (4) `vstack`: 1스레드는 int64 색인의 바이트 몫(scipy의 4/3)이고 10스레드는 scipy보다 빠르다(SP3의 0.57배는 smoke 한 번의 첫 호출이었다). (5) Frobenius는 `sd.linalg.vector_norm`(두 번 읽는다; linalg의 것). (6) `random_array`는 `sd.random`의 비복원 `choice`(numpy의 4–5배; random 영역의 BN-13). (7) 코드 리뷰는 사용자가 따로 한다(optimize O96의 둘째 부분과 같다). (1)은 SP133, (2)와 (3)은 SP131에서 고쳤고, (5)는 linalg의 빠른 `vector_norm`으로 나아졌다 (추천안, 사용자 위임 2026-11-21) |
| SP118 메서드의 틀린 호출 | 희소 배열(`sparray`, `_compressed`와 네 형식), `LinearOperator`, `SparseLU`의 모든 공개 메서드의 틀린 호출 모양(없는 키워드, 빠진 인자, 인자 개수)이 다른 함수처럼 메서드 이름으로 시작한다(API.md §2.1의 꼴: `sum(): unexpected keyword argument 'x'`, `tocsr(): takes 0 positional arguments but 1 was given`; self는 세지 않는다). 한 곳(`sd.sparse/__init__.py`의 `_calls.methods(cls)`)에서 클래스마다 한 번 감싼다; 메서드 안에서 난 `TypeError`는 그대로다. `issparse`·`matrix_transpose`도 `_calls.function`으로 감쌌다. 비용은 호출마다 약 60 ns(API.md가 frame·Generator·분포의 작은 메서드를 예외로 둔 까닭)이고 희소 메서드의 일보다 작다. 골든에는 이 꼴의 케이스가 없어 그대로다 (추천안, 사용자 위임 2026-11-22) |
| SP119 `block_diag`의 인자 이름 | 희소와 밀집 모두 `arrays`다(`block_diag(arrays, /, *, format=None, dtype=None)`; scipy의 `mats`가 아니다). 메시지도 `'arrays'`, `arrays[1]`이다(`block_diag(): 'arrays' must hold at least one array`). 이유: 같은 개념은 영역을 가로질러 한 이름(DESIGN §1.4); SP28(scipy의 이름)의 명시한 예외다. 위치 전용 인자라 호출은 그대로다 (추천안, 사용자 위임 2026-11-22) |
| SP120 속도 표 | `bench/sparse/speed.md`는 생성 파일 그대로 둔다: SP8의 실행(`build/speed/sp8-sparse.jsonl`)에서 만든 것이고, 사용자가 다시 재면(기준선 `promote`) 다시 만들어진다. 마일스톤의 A/B와 찾은 것은 §5의 SP8 절에 있다 (추천안, 사용자 위임 2026-11-22) |
| SP121 나눈 DOK의 `resize` | `dok_array(D)`는 D의 해시 맵을 나눈다(scipy와 같다: 원소 쓰기가 서로 보인다). `resize`는 맵을 제자리에서 바꾸지 않고 새 모양의 키만 담은 새 맵을 만든다(copy-on-resize; `_DokMap.resized`). 그래서 나눈 다른 쪽은 자기 모양과 키를 그대로 갖고, 그 뒤로 둘은 나누지 않는다. 덧붙여 `_parts`·`_tocsr`는 맵의 모양이 객체의 모양과 다르면 내부 오류(`RuntimeError`)로 막는다(불변식). 이유: 나눈 맵의 모양을 제자리에서 바꾸면 다른 쪽이 범위 밖 키를 받고, 그 `tocsr()`이 옛 줄 수로 압축해 흩뿌리기 커널이 힙 밖에 쓴다(프로세스가 죽는다). resize는 어차피 모든 칸을 훑으므로 복사의 값이 같은 차수다 (추천안, 사용자 위임 2026-11-22) |
| SP122 `expm`의 고침과 밀집 짝의 비트 | 대각 행렬과 삼각 고침의 대각·첫 비대각은 밀집 `sd.linalg.expm`과 같은 식을 C++(`_expm_fix`, `csrc/sparse/convert.cpp`)에서 같은 libm `exp`·`sinh`와 `ldexp`로 계산한다(`sd.exp`는 libm과 다른 구현이라, 예를 들어 `sd.exp(10)`이 0.62 ulp여서 그것을 쓰면 2 × 2 삼각 행렬의 모든 원소가 밀집과 1 ulp 다르다). 대각 행렬과 2 × 2 삼각 행렬은 밀집과 비트까지 같다. 일반 행렬은 곱(SpGEMM)과 마지막 풀이(LU)의 순서가 밀집과 달라 끝자리가 다를 수 있음을 docs/api/sparse.md §8에 적었고, 풀이 짝(spsolve·inv·spsolve_triangular·splu)도 §5에 적었다(API.md §7의 범위는 그 문서가 정한다). 이유: 같은 개념(삼각 고침의 정확한 대각)은 영역 사이에 같은 계산이다 (추천안, 사용자 위임 2026-11-22) |
| SP123 `A * numpy 배열`의 메시지 | 희소 배열과 `LinearOperator`의 연산자가 numpy 배열을 보면 Python의 문장 그대로 `TypeError`를 직접 낸다: `unsupported operand type(s) for *: 'csr_array' and 'numpy.ndarray'`(numpy가 어느 쪽에 있어도). 비교(`<` 등)는 numpy가 양보해 Python의 `'<' not supported between instances of ...`가 이미 나온다. 이유: 문서(§3.1)의 약속에 코드를 맞춘다; 그대로 두면 numpy의 반사 연산이 `operand 'csr_array' does not support ufuncs (__array_ufunc__=None)`를 낸다 (추천안, 사용자 위임 2026-11-22) |
| SP124 문서로만 적은 것 | 코드는 그대로 두고 docs/api/sparse.md에 적었다: (1) 특이 행렬의 `gmres`(§6): 회전한 H(k, k)가 scipy는 정확히 0, sd는 반올림 크기라 x가 아주 다르다(둘 다 `converged=False`; scipy처럼 정확한 0 비교만 하고 상대 문턱은 두지 않는다). (2) 반복 상한의 `n_iter`(§7): `eigsh`·`eigs`는 ARPACK의 iparam(3)이라 `max_iter` + 1, `svds`는 `max_iter`다(scipy와 같다). (3) γ_m(§4): 코드(`tests/sparse/refdata.py`의 `gamma`)대로 γ_m = mu/(1 − mu), u = 2⁻⁵³(mε/(1 − mε)보다 2배 빠듯한 경계다). (4) dtype이 다른 짝: 희소 `trace()`·`kron`·`kronsum`의 bool·int는 int64(배열 영역의 `sd.sum`·`sd.multiply`, SP55·SP56), 밀집 짝 `sd.linalg.trace`·`kron`·`kronsum`은 float64다(DESIGN.md §9.6). 이유: scipy와 같은 동작이거나 이미 정한 결정이고, 문서가 그 동작을 말하지 않았다 (추천안, 사용자 위임 2026-11-22) |
| SP125 `spsolve_triangular`의 유한성 검사 | 읽는 원소(고른 삼각과, `unit_diagonal=False`면 대각)만 유한해야 한다. 읽지 않는 쪽의 NaN·inf는 결과에 영향이 없다(`[1.0, nan], [1.0, 1.0]`의 아래 삼각 풀이는 `[1.0, 0.0]`). §1.6(docs/api/sparse.md)의 "저장된 NaN·inf는 `ValueError`"에 이 함수를 예외로 적었다. 이유: 밀집 짝 `sd.linalg.solve_triangular`의 약속(읽지 않는 원소는 NaN이어도 된다)을 따른다; 검사는 저장된 값에 유한하지 않은 것이 있을 때만 줄과 열을 비교한다 (추천안, 사용자 위임 2026-11-22) |
| SP126 `minres`의 βₖ < 0 | 반복 중의 βₖ = r₂ᴴ M r₂ < 0도 `minres(): the preconditioner is not positive definite`다. βₖ < 0은 M이 양정치면 생기지 않으므로 원인은 M이다. 대칭이 아닌 A는 검사하지 않는다(scipy와 같다; M이 없거나 양정치면 반복이 끝까지 돌고 수렴으로 끝날 수도 있다). 이유: 메시지가 원인을 짚는다(scipy의 "non-symmetric matrix"는 그 원인을 잘못 짚는다); 검사를 더하면 반복마다 곱이 하나 는다 (추천안, 사용자 위임 2026-11-22) |
| SP127 복소 에르미트 `eigsh`의 순서와 열이 0개인 B | (1) 복소 에르미트 `eigsh`(`eigs`의 복소 Arnoldi로 푼다)의 고유값을 오름차순으로 정렬하고 고유벡터의 열도 같이 옮긴다(안정 정렬). `[scipy와 다름]`: scipy는 ARPACK이 남긴 순서다. 실수 경로와 같은 순서이고 문서(§7)의 약속이다. (2) `expm_multiply`에 열이 0개인 B (n, 0)를 주면 빈 결과다(구간이면 (num, n, 0); n = 0과 같은 꼴). 단계 선택의 식 `n0 * m_max`가 0이 되는 경우다(scipy는 `ZeroDivisionError`) (추천안, 사용자 위임 2026-11-22) |
| SP128 희소 정확도 기준선 | `bench/accuracy/sparse.py`의 일곱 스윕(spmv, sums, solve, iterative, eig, norms, matfuncs; 26영역)을 비교 대상(scipy, R Matrix, R RSpectra·irlba)과 함께 재고 `promote`해 `bench/accuracy/results/sparse.json`과 `bench/sparse/accuracy.md`를 만들었다(2026-11-22, 작업 트리). 계약을 넘은 영역은 0이고 가장 큰 것은 `rmatvec` 0.83이다. 복소 값의 ULP 열이 10¹⁸대인 것(eig, expm, total)은 복소 ULP 척도의 문제라 따로 고친다(공용 `bench/accuracy/common.py`) (추천안, 사용자 위임 2026-11-22) |
| SP129 `expm_multiply`의 `traceA` | 정확한 대각합(또는 None)일 때 결과가 계약 C_EXPM 안이다. 참 대각합에서 먼 값은 A − μI의 노름을 키워 Taylor 합의 상쇄로 정확도를 잃는다(scipy와 같다; `expm_multiply(eye_array(2), ones(2), traceA=100.0)`은 상대 9.4×10⁻⁹, 계약은 약 4×10⁻¹⁵). 코드는 그대로이고 SP107의 "어떤 값이어도 결과는 exp(A)B"를 docs/api/sparse.md §8에서 이 문장으로 고쳤다 (사용자 결정 2026-11-22) |
| SP130 ARPACK의 OP 수 | 계산 순서는 SP83대로 scipy가 돌리는 ARPACK의 것이고, OP 수와 값은 scipy 자신의 퍼짐으로 견준다: v0의 한 원소를 한 ulp 바꾼 scipy 여섯 실행의 OP 수 범위를 10% 넓힌 안이고, 반올림에 둔한 문제는 값이 순서까지 같으며(상대 10⁻⁹), 민감한 문제는 scipy의 실행 하나와 같은 값이다(무작위 비대칭 'LI' k = 4는 scipy 여섯 실행 가운데 둘이 찾는 고유값 쌍을 찾는다). 바꾼 것: SP83이 약속한 OP 수의 일치(스무 문제 중 열다섯이 같았다). 이유: 기저의 곱을 Accelerate `dgemv`로 바꾸면(SP131) 합의 끝자리가 scipy의 BLAS와 달라 OP 수가 scipy의 ulp 퍼짐만큼 움직인다(lap1d/400 'LM' k = 3: 4,483, scipy 4,466–4,483). 스무 문제에서 상태는 모두 같고 OP 수는 열셋이 같다. 1·3·10스레드의 OP 수와 비트는 같다 (추천안, 사용자 위임 2026-11-22) |
| SP131 Krylov의 벡터 연산 | Lanczos·Arnoldi·GKL의 Vᴴx와 V·c는 행 2¹⁴개씩의 고정 조각마다 Accelerate `dgemv`·`zgemv`를 한 번 부른다(Vᴴx는 조각의 부분합을 조각 순서로 더한다). 내적과 2-노름은 linalg의 `fastsum`(`fast::dot`·`cdot`: n으로만 정한 조각마다 Accelerate `ddot`·`zdotc`나 16레인 잎, 조각끼리 균형 트리; 노름은 제곱합을 믿을 수 있으면 그 제곱근, 아니면 `vector_norm`의 척도 조정 길)이다. `gmres`의 갱신과 다음 내적은 `axpy` 뒤 `dot`이다. 바꾼 것: SP116의 여러 열 내적(잎마다 모든 열, 열마다 `dot`과 비트가 같음), 열 순서 fma의 행 구간 갱신, MGS의 한 번 훑기, `sd.linalg.vecdot`의 pairwise 내적. 이유: n = 10⁴에서 `dgemv`(AMX)가 NEON 루프의 약 10배라 `eigsh`·`eigs`가 약 2배, `svds`가 1.2–1.6배 빨라졌다(scipy와 비슷하거나 빠르다). 분할이 n과 열 수로만 정해져 스레드 수와 무관하고, 같은 입력을 300번, 부하 아래에서, 네 스레드가 동시에 불러도 비트가 같았다(실행 사이의 결정성). ARPACK 스무 문제와 반복법(`cg`·`bicgstab`·`gmres`·`minres`·`lsqr`, 격자와 무작위 비대칭 n = 10⁵까지)의 반복 수와 비트가 1·3·10스레드에서 같다. 정확도(계약 단위, 전 → 후): eigsh 0.496 → 0.70(lap1d/1000 'LA'), eigs 0.011 → 0.021(93 → 99 ulp), svds 0.0089 → 0.0100, 2-노름 0.0126 → 0.014, 반복법은 그대로; 모두 계약 안이다 (추천안, 사용자 위임 2026-11-22) |
| SP132 shift-invert의 풀이 | `eigsh`·`eigs`의 (A − σM)⁻¹, M⁻¹ 연산자는 LU 풀이 하나이고 다듬지 않는다(`_Factor.solve(refine=False)`). 바꾼 것: SP94의 반복 다듬기(OP 하나에 풀이 둘과 SpMV 하나). 이유: scipy(SuperLU)와 RSpectra도 걸음마다 풀이 하나이고, 성분별 역오차가 몇 ε(lap2d/100 2ε, lap3d/22 7ε)인 OP를 ARPACK이 같은 판정으로 수렴시킨다. 걸음이 약 절반 시간이라 σ = 0의 `eigsh`·`eigs`가 약 1.5배 빨라졌다. 바뀐 것은 값의 끝자리이고 정확도 스윕의 eig는 계약 안이다(SP131의 값). SuiteSparse goodwin의 σ = 0 `eigs`는 scipy와 상대 10⁻⁶ 안이다(전에는 10⁻⁹; §7.2) (추천안, 사용자 위임 2026-11-22) |
| SP133 반복 다듬기의 판정 | `spsolve`, `SparseLU.solve`, `factorized`, `inv`는 첫 풀이의 성분별 역오차 maxᵢ\|rᵢ\| / (\|op(A)\|\|x̂\| + \|b\|)ᵢ(LAPACK `dgerfs`의 판정; 복소는 \|re\| + \|im\|)가 8ε를 넘을 때만 한 번 다듬는다. 우변 묶음은 한 열이라도 넘으면 묶음 전체를 다듬고, NaN이면 다듬는다. 잔차와 판정은 C++ `_residual`(CSC를 `'N'`은 열 순서로 흩고 `'T'`·`'H'`는 열을 따라 모은다)이라 순서가 정해져 있고 스레드 수와 무관하다. 바꾼 것: SP63의 "모든 풀이에 한 번". 이유: 잘 조건된 문제는 첫 풀이가 1–8ε라 풀이 하나로 끝나고(`splu().solve`가 lap2d/100·300, lap3d/22에서 약 1.6–1.9배), 문턱 피벗이 키운 오차(saddle/600x150 약 4,000ε, blockdiag/200x8 300–800ε, long_rows/9000 17ε, random_spd/1500 9ε)는 그대로 다듬어 정규화 역오차 1ε 아래가 된다. 정규화 역오차(계약 C_BACKWARD = 4ε)의 가장 큰 것은 다듬지 않은 lap3d/22 `trans='T'`의 3.54ε다(늘 다듬을 때는 시험 행렬 모두 0.60ε 이하). 앞오차는 계약 C_SOLVE·κ∞ε 단위로 0.097 → 0.253(random_unsym/1000x8)이다. 큰 격자(lap2d/1000, lap3d/45)의 풀이 시간은 그대로다 (추천안, 사용자 위임 2026-11-22) |
| SP134 우변의 묶음 | Accelerate의 풀이에 우변을 8열씩의 고정 묶음으로 넘긴다(실수 인수의 복소 우변은 실수부·허수부의 2k열에서). 바꾼 것: SP64의 k열 한 번. 이유: Accelerate의 묶은 풀이는 열 하나의 비용이 8열에서 가장 작다(lap2d/30·100에서 혼자 푸는 것의 약 0.45배; 64열에서는 약 1.2배로 오히려 크다). `inv`(희소 우변을 64열까지 묶어 푼다)가 약 1.7–2.3배 빨라졌다(SP133과 함께 약 3–4배). 열 하나의 비트는 k와 그 열의 묶음 안 자리로 정해진다(결정적, 스레드 수와 무관) (추천안, 사용자 위임 2026-11-22) |
| SP135 색인 복사의 재검토 | SP30(받은 색인 배열은 `copy`와 상관없이 언제나 복사하고 검사한다)을 지킨다. 복사와 검사는 한 번의 NEON 훑기다(§3.1). 이유: 받은 배열을 나누면 scipy처럼 거의 O(1)이지만(nnz 10⁷에서 메모리 복사 몫인 수백 배의 차이가 남는다), 사용자가 그 배열을 고치면 표준 꼴이 깨지고 커널이 범위 밖을 읽는다. 보이는 동작(별칭)을 바꾸는 결정이라 §9에 사용자에게 물을 것으로 둔다 (추천안, 사용자 위임 2026-11-22) |
| SP136 비트를 바꾸지 않은 커널 | 결과의 비트를 바꾸지 않고 고쳤다(해당 골든 케이스는 그대로): (1) `(data, indices, indptr)`·`(data, (row, col))` 생성자의 한 번 훑기(SP135)와 `_index_copy`·`_index_max`의 NEON, Python 층의 인자 지름길(작은 입력의 고정 비용이 약 1/4), (2) `diags_array`는 COO와 표준 꼴 만들기를 거치지 않고 CSR을 줄마다 바로 채운다(0은 그 뒤에 거른다; 대각 수가 (줄마다 평균 저장 수 + 1)의 네 배를 넘으면 COO 길), (3) `tril`·`triu`는 CSR·CSC의 줄마다 정렬된 앞부분이나 뒷부분을 바로 옮긴다(`tocoo()`의 순서; COO·DOK 입력은 COO 길), (4) `kronsum`은 줄마다 B의 왼쪽 블록, A의 줄과 bᵢᵢ(겹치면 a + b), B의 오른쪽 블록을 바로 채운다(SP111의 COO 길과 비트가 같다; 빈 A·B는 COO 길), (5) `is_sptriangular`는 가운데·처음·마지막 행을 먼저 보고 둘 다 거짓이 되면 멈춘다, `spbandwidth`는 행 구간으로 병렬, (6) CSR·CSC와 밀집의 `+`·`-`는 D ± 0의 밀집 연산 뒤 저장된 값을 제자리에 더한다(`_dense_update`; ±0의 부호까지 `A.toarray()` ± D와 같다), (7) `X @ A`의 흩뿌리기는 Xᵀ의 행을 연속으로 옮겨 놓고 쓰며, 출력이 8 MB를 넘으면 16항 앞의 출력 행을 미리 읽는다(캐시에 드는 출력에서는 미리 읽기가 손해였다), (8) CSR의 줄 합(SpMV, 반복법·고유값 풀이의 곱)은 네 줄의 FMA 사슬을 함께 돌린다(줄마다의 순서는 그대로), (9) SpGEMM은 스레드풀을 쓸 수 없을 때(1스레드, 작은 일) 세는 패스 없이 줄 구간마다 자기 버퍼에 쓰고 제자리로 옮긴다(풀에서는 두 패스가 빠르다), (10) 정렬되지 않은 열 고르기는 고른 것이 없는 줄을 건너뛰고 하나인 줄은 정렬하지 않는다, (11) `spsolve_triangular`의 우변 하나는 탐색·특이 판정·대입을 한 번 훑는다(특이면 훑은 뒤 `LinAlgError`). 해 보고 버린 것: 삼각 풀이의 유한성 검사를 커널의 훑기에 넣기(배열의 `isfinite`가 빨랐다), SpGEMM 안쪽 고리의 분기 없애기(같은 시간), Accelerate 풀이에 작업 공간 미리 주기(같은 시간) (추천안, 사용자 위임 2026-11-22) |

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
| `eye_array`, `diags_array`, `random_array`, `block_array` | 같은 이름 (SP48) | SP3 |
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
| `eigsh`, `eigs`, `svds` | 같은 이름 | 이 라이브러리가 옮긴 ARPACK, Golub–Kahan–Lanczos (SP83, SP85, §3.8) | SP6 |
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
- **같은 입력이면 sd의 스레드 수와 상관없이 같은 비트다**(API.md §3). 희소 합의 순서는 §2.6이 정한다. 2026-11-22의 우선순위(빠르기, 다음 정확도)로 스레드 수와 무관한 비트는 더 이상 요구가 아니고 같은 스레드 수에서 실행마다 같은 비트만 요구지만, 이 영역은 모든 경로가 분할을 크기로만 정해 스레드 수와 무관한 비트를 그대로 지킨다(Accelerate `dgemv`·`ddot`을 부르는 Krylov의 벡터 연산도 n으로만 정한 조각마다 부른다, SP131). numpy·scipy와 비트가 같은 것도 목표가 아니다: SpMV처럼 순서가 같아 같은 곳은 그대로이고, ARPACK의 OP 수는 scipy의 퍼짐 안이다(SP130). Accelerate를 부르는 계산은 BLAS·LAPACK처럼 Accelerate의 스레드 모드가 같을 때 같은 비트이고, 같은 모드 안에서 실행마다 비트가 달라지는 Accelerate의 경로는 쓰지 않는다(§1.4의 잰 값, §3.6).
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
- **SP4에서 찾은 것**(SDK 26.5, 모든 시험 행렬에서 재현):
  - 전치로 도는 풀이(CSC 인수의 `SparseGetTranspose`, 또는 CSR을 전치 속성으로 넘긴 인수의 보통 풀이)는 **우변이 둘 이상이면 틀린 답**을 낸다(k = 2, 5에서 잔차가 1 크기; 실수·복소, 대칭 행렬도). 우변이 하나면 맞다. 그래서 래퍼는 전치 풀이를 열마다 따로 부른다(`csrc/accelerate/sparse.hpp`).
  - 복소 LU의 `SparseGetConjugateTranspose`로 풀면 몇백 행(random_complex/800)에서 **프로세스가 멈춘다**(SIGTRAP). Aᴴx = b는 conj(Aᵀ conj(x)) = b로 푼다(켤레 둘과 전치 풀이).
  - CSR을 전치 속성으로 넘겨 분해한 인수와 같은 행렬의 CSC로 분해한 인수는 풀이의 끝자리가 다르다(상대 10⁻¹⁶–10⁻¹⁵). 형식이 비트를 바꾸지 않도록 모든 입력을 CSC로 바꿔 분해한다(§3.6).
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
- **DOK**: 하나씩 넣는 만들기용 형식이다. 키 (i, j) → 값의 해시 맵을 C++에 둔다(`csrc/sparse/dok.cpp`: 열린 주소법, 선형 탐사; scipy는 Python `dict`의 하위 클래스다). `A[i, j] = v`와 `A[i, j]`가 O(1)이고, 계산할 때는 CSR·CSC로 바꾼다. Python `dict`의 메서드는 두지 않는다(SP25). 0을 넣으면 그 키를 지운다(scipy와 같다). 다른 형식에서 바꿀 때는 저장된 0도 남는다(scipy와 같다). 맵은 제자리에서 resize하지 않는다(SP121: 나눈 맵을 지킨다). 다른 형식으로 바꿀 때는 키를 칸 순서로 모아(스레드풀에서는 고정 크기의 칸 묶음마다) COO → CSR의 계수 정렬로 정렬한다: 키가 서로 달라 아무것도 더하지 않으므로 값은 그대로 옮겨지고, 비교 정렬을 하지 않는다. 행이 원소에 비해 아주 많으면(2⁶²행에 원소 몇 개) indptr을 만들지 않고 키를 비교 정렬한다. 맵을 만들기(`from_coo`)와 읽어 내기(`to_coo`, `to_csr`)는 Python 객체를 만지지 않아 병렬 기준값부터 GIL을 놓는다(잰 것, nnz 3×10⁶, 200000², 1스레드: `todok` 동안 다른 Python 스레드가 초당 64M번 돈다(GIL을 놓는 다른 연산과 같다), DOK → CSR 62 ms, COO → CSR 32 ms; `std::sort`로 정렬하면 DOK → CSR이 392 ms다).
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
  - **색인 배열은 언제나 복사하고 검사한다**(SP30; `copy`와 상관없다): 범위(밖이면 `ValueError`, 배열의 메시지 꼴 `index 5 is out of bounds for axis 1 with size 4`), `indptr`의 시작 0·줄어들지 않음·끝 = 저장 수. 받은 배열을 나중에 고쳐도 이 배열의 표준 꼴이 깨지지 않게 하려는 것이다(커널은 색인을 믿고 읽는다). 검사와 복사는 한 번의 NEON 훑기다(§3.1). scipy는 검사도 복사도 하지 않는다(`[scipy와 다름]`): 큰 입력에서 이쪽은 메모리 복사만큼의 시간이 들고 scipy는 거의 들지 않는다(SP135, §9).
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
- **Python 스칼라와 연산자**(SP39): `A + c`(c ≠ 0, NaN 포함)와 `A ** 0`은 결과가 밀집이 된다. scipy는 `NotImplementedError`다. 이쪽은 `ValueError`(`adding a nonzero scalar to a sparse array is not supported; use A.toarray() + c`)다(`[scipy와 다름]`: 예외 종류만, API.md §2의 목록에 `NotImplementedError`가 없다). `A + 0`, `A - 0`, `0 - A`는 같은 패턴의 새 배열이고 값은 배열의 덧셈·뺄셈이다(scipy는 복사; 저장된 -0.0 + 0이 +0.0이 되는 것만 다르다).

### 2.6 합의 순서와 결정성
**희소 합의 결과는 행렬의 값과 모양으로만 정해진다**: sd의 스레드 수, 저장 형식(CSR·CSC·COO), 태스크 분할과 무관하다(SP15).

- **줄 합의 규칙**: 한 출력 원소에 들어가는 항이 n개이면, 항을 그 줄의 색인 오름차순으로 놓고 0에서 시작해 차례로 더한다. 곱의 합은 항마다 `fma(a, x, acc)`다. n이 L(안: 4096)을 넘으면 L개씩의 고정 조각을 각각 차례로 더하고, 조각들의 합을 `base/pairwise.hpp`의 트리로 더한다. 트리는 n으로만 정해진다.
  - **SpMV**(`A @ x`): yᵢ = Σⱼ aᵢⱼxⱼ, j 오름차순. scipy의 `csr_matvec`는 `sum += Ax[jj]*Xx[Aj[jj]]`를 clang이 FMA로 축약한 순차 합이다: 이 초안에서 2000 × 2000(밀도 0.01) 무작위 행렬로 scipy의 결과가 "FMA 순차 합"과 비트까지 같고 "축약 없는 순차 합"과는 다른 것을 확인했다. 그래서 줄이 L 이하이면(실제 행렬의 거의 모든 줄) **scipy와 비트까지 같다**. scipy의 CSC `@`(열 순서의 흩뿌리기)도 CSR과 비트가 같았다: 열 순서로 흩뿌리면 각 행이 j 오름차순의 순차 합이 되기 때문이다. 같은 규칙이 형식과 무관한 비트를 준다.
  - **항 하나**(SP2에서 잼): 실수는 `fma(a, x, acc)`, 정수는 wrap하는 곱과 합, 복소는 `acc + a·x`(numpy의 복소 곱 `cmul`: 실수부 `fma(ar, xr, -(ai·xi))`, 허수부 `fma(ar, xi, ai·xr)`, 그 뒤 성분별 덧셈)다. scipy의 복소 `csr_matvec`가 이것과 비트까지 같다(무작위 300 × 300 복소 행렬에서 일곱 후보 중 이것만 같았다). 희소 쪽 값이 언제나 왼쪽 인수다(`x @ A`와 CSC × CSC도): 복소 곱은 인수를 바꾸면 끝자리가 다르기 때문이다. 피연산자의 dtype이 다르면 배열의 `matmul` 규칙으로 승격한 dtype에서 계산한다(bool @ bool은 int64).
  - **SpMM**(`A @ X`, X는 (n, k)): 출력 원소마다 위와 같은 합이다. 그래서 결과의 열 j는 `A @ X[:, j]`와 비트가 같다(scipy도 같다, 확인함). 구현은 X의 행을 NEON으로 k 방향으로 묶지만 원소 하나의 순서는 바꾸지 않는다.
  - **밀집 @ 희소**(`x @ A`)와 **Aᵀ @ x**: 같은 규칙을 Aᵀ에 쓴다(항은 행 번호 오름차순). `X @ A`(X는 (k, n))는 Xᵀ의 행(X의 열)을 연속한 작업 배열로 옮겨 두고 흩뿌린다(SP136: 항마다 k개의 캐시 줄을 건드리던 것을 하나로).
  - **희소 @ 희소**(SpGEMM): Cᵢⱼ = Σₖ aᵢₖbₖⱼ, k 오름차순의 FMA 순차 합. scipy의 `csr_matmat`(A의 행 i의 원소 순서대로 B의 행을 흩뿌려 더함)과 같은 순서다. L 조각 규칙은 여기에 쓰지 않는다(흩뿌리기의 누산기 하나라 조각을 나눌 자리가 없다; 결과의 비트는 여전히 값과 모양만으로 정해진다).
  - **축 리덕션**(`sum(axis=0|1)`, `mean`): 줄마다 위의 규칙(곱 없이 덧셈). SP2에서 확인한 scipy: 자기 줄을 가로지르는 합(CSR의 `axis=0`, CSC의 `axis=1`)은 `ones @ A`·`A @ ones`라 같은 순차 합이고 비트가 같다. 자기 줄을 따르는 합(CSR의 `axis=1`, CSC의 `axis=0`)은 `np.add.reduceat`(numpy의 pairwise)라 줄이 8개 이상이면 끝자리가 다르다(설계 초안의 "scipy는 `A @ ones`"는 반만 맞았다). 대안은 줄마다 pairwise(`base/pairwise.hpp`의 트리)다: 긴 줄에서 더 정확하지만 L 이하에서도 scipy와 다르다. 정수(bool, int64)의 합은 128비트로 정확히 더해 int64로 wrap한다(순서와 무관하다). 평균은 그 합을 축의 길이로 한 번 나눈다: 정수는 정확한 합을 한 번 반올림한 뒤 나누고(배열의 `mean`과 같은 규칙; 배열은 바깥 축에서는 float 합이지만 희소는 언제나 정확하다), float·complex는 순서대로 더한 합 ÷ 길이다(scipy는 값마다 1/길이를 곱한 뒤 더한다).
  - **전체 리덕션**(`sum()`): 저장된 값을 논리적 C 순서(행 우선, 줄 안은 색인 오름차순)로 놓고 `base/pairwise.hpp`의 트리로 더한다. CSR이면 `sd.sum(A.data)`와 비트가 같다. CSC·COO도 같은 C 순서로 더한다(CSC는 그 순서를 만드는 한 번의 훑기가 든다). linalg의 "항의 순서는 논리적 C 순서"(linalg.md §8.2)와 같은 원칙이다. scipy는 `data.sum()`(numpy의 pairwise, 저장 순서)이라 끝자리가 다를 수 있다(array.md §6.2).
  - **중복 더하기**: §2.3(입력 순서의 순차 합).
- **병렬 분할** (모두 크기만으로 정해 비트를 바꾸지 않는다)
  - CSR의 행 방향 일(SpMV, SpMM, 행 리덕션, 원소별 연산): 출력 행을 nnz가 약 2¹⁴개씩 되는 고정 구간으로 나눈다(`indptr`만으로 정함). 행 하나는 한 태스크가 계산하므로 순서가 직렬과 같다. L보다 긴 행의 조각은 다른 태스크가 계산해도 같은 트리로 합친다.
  - CSR의 열 방향 일(Aᵀx, `x @ A`, 열 합, CSC의 SpMV): SP2에서는 **한 스레드의 흩뿌리기**다(줄 순서대로 흩뿌리면 각 출력이 색인 오름차순의 순차 합이다). 한 번 전치(병렬 계수 정렬)하고 행 방향으로 모으는 길을 쟀는데 nnz 10⁵–10⁷의 10스레드에서 흩뿌리기보다 느리거나 같았다(10⁷에서 10.6 대 9.5 ms). 출력 구간마다 줄을 이분 탐색하는 길도 줄이 짧으면 탐색이 일의 몇 배다. SP8에서 병렬 흩뿌리기를 더했다(SP116): 출력을 2^s개씩의 블록(값 4096개 이하)으로, 줄을 nnz 2¹⁴개씩의 고정 구간으로 나눠 구간마다 항을 블록별로 안정 계수 정렬하고(항마다 출력 색인과 필요한 값: k = 1이면 두 인수, 리덕션이면 항), 블록마다 구간 순서 — 곧 줄 순서 — 로 더한다. 출력 하나의 항 순서가 한 스레드의 흩뿌리기와 같으므로 비트가 같다. 한 스레드의 흩뿌리기는 출력이 8 MB를 넘을 때만 16항 앞의 출력 행을 미리 읽는다(SP136; 캐시에 드는 출력에서는 손해였다). 풀이 2¹⁶ 단위 일부터이고 블록이 넷 이상일 때다(결과는 어느 길이든 같다). 출력 하나가 L개 넘는 항을 받으면 출력마다 16비트 개수로 조각을 끊는다. 그런 출력이 없다는 것(어느 색인도 L번 넘게 나오지 않음)은 배열마다 처음 한 번 세어 두고(색인은 바뀌지 않는다), 없으면 개수를 세지 않는다(세면 1스레드 흩뿌리기가 두 배 느렸다).
  - 변환과 중복 더하기: 고정 조각의 계수 정렬(§2.3).
  - 기준값은 배열의 규칙이다(array.md §10.4): 작업량(바이트로 센 nnz와 출력 크기)이 2¹⁵ float64 분량 이상이면 나누고, 그 이상이면 스레드 수와 상관없이 GIL을 놓는다(API.md §3).
- **Accelerate**: 같은 입력, 같은 Accelerate 스레드 모드에서 비트가 같은 경로만 쓴다(§1.4의 표: LU는 AMD(Metis는 동시 호출에서 실행마다 달라 쓰지 않는다, §1.4·§3.6), QR은 COLAMD). 분해 호출은 Accelerate가 안에서 스레드를 정하고, sd는 GIL을 놓고 차례로 부른다(linalg의 큰 호출과 같다, array.md §9.4).
- 테스트는 1, 2, 3, 7, 10스레드에서 같은 바이트, 같은 행렬의 CSR·CSC·COO에서 같은 바이트를 확인한다.

---

## 3. 수치 방법 (안)

### 3.1 형식 변환과 만들기 (SP1)
`csrc/sparse/canonical.cpp`가 표준 꼴을 만든다. 입력은 원소 (major, minor, 값)의 목록이고(COO → CSR이면 (행, 열, 값), → CSC면 (열, 행, 값)), 출력은 (major, minor) 순서의 `indptr`·`indices`·`data`, 같은 자리는 입력 순서로 더한 하나다. 길은 둘이고 같은 함수를 계산한다(모든 정렬이 안정적이고 중복의 합이 입력 순서라서, 결과는 입력만으로 정해진다). 그래서 어느 길을 탈지는 속도로만 크기에 따라 고르고, 비트는 길과 스레드 수와 무관하다(테스트는 두 길과 1·2·3·7·10스레드의 바이트를 견준다).
- **곧은 길**: `indptr`을 계수기로 쓰는 major의 계수 정렬(scipy의 `coo_tocsr`과 같은 꼴), 그 뒤 줄마다 minor로 안정 정렬(16개까지는 키 minor·16 + 위치의 순위 정렬(NEON, 분기 없음; SP8), 32개까지 삽입 정렬, 넘으면 `std::stable_sort`)과 중복의 차례 합. 줄에 놓을 때 입력 위치 대신 값을 함께 옮겨, 나중에 값을 흩어 읽는 패스가 없다.
- **LSD 길**(SP8, SP116; 한 스레드, 정렬·합이 필요한 변환, 2²¹개 아래, 가장 큰 minor가 2·nnz + 2¹⁶ 이하): minor로 안정 계수 정렬한 뒤 major로 안정 계수 정렬해 줄에 바로 놓는다. 두 정렬이 안정이라 줄은 minor 순서이고 같은 자리는 입력 순서로 만나며, 줄의 마지막과 같은 자리는 도착할 때 더한다(왼쪽부터 차례로). 중복이 있으면 줄을 당겨 붙인다. 비교 정렬이 없어(무작위 열 개의 줄을 정렬하는 데 원소마다 약 5 ns) 곧은 길의 1.4–1.6배다. 같은 함수를 계산하므로 비트가 같다.
- **양동이 길**: major를 2^s개씩 묶은 양동이(64–1024개, nnz와 줄 수로만 정함)로 원소를 안정적으로 흩고(입력의 고정 조각 2^14개마다 양동이별 개수를 세어 조각 순서로 누적), 양동이마다 줄의 계수 정렬과 위의 줄 처리를 한다. 양동이가 캐시 크기라 한 스레드에서도 큰 입력에서 빠르고, 양동이끼리는 스레드풀에서 독립이다. 줄이 아주 많고 원소가 적은 양동이(줄 수 > 4·원소 수 + 1024)는 양동이째 안정 정렬한다.
- **고르는 크기**(M4 Pro, 줄마다 10개인 무작위 행렬에서 잼): 스레드풀을 쓸 수 있으면 2^16개부터 양동이 길, 한 스레드에서는 정렬·합이 필요한 변환이 2^21개부터(SP8; 그 아래는 LSD 길, 전에는 2^17개부터였다), 전치(CSR ↔ CSC: 줄 안이 이미 정렬되어 정렬·합이 없다)는 한 스레드에서 2^15개부터 아래의 열 블록 길이다(SP8; 전에는 2^20개부터 양동이 길). 양동이를 64개보다 적게 두면 10스레드에서 몇 스레드가 늦게 끝나 2^17개 근처가 1스레드와 같았다.
- 이미 표준 꼴인 COO(`has_canonical_format`)는 정렬 없이 `indptr`만 만들고 값과 minor 색인을 그대로 쓴다.
- **CSR ↔ CSC**: 위의 전치(scipy의 `csr_tocsc`와 같은 계수 정렬; 안정이라 결과도 정렬되어 있다). 한 스레드의 2^15개부터는 열 블록 길(SP8, SP116): 열 2^s개(블록마다 원소 약 2048개)의 블록으로 원소를 행 순서대로 흩고(키 = 행·2^s + 블록 안의 열, 값), 블록마다 열을 세어 자리에 놓는다. 무작위 n열로 흩어 쓰는 계수 정렬 대신 차례로 두 번 쓴다(nnz 10⁶ 1.4배, 10⁷ 1.5배; 키가 64비트에 들지 않으면 계수 정렬). 결과는 같다. `.T`는 복사 없는 뷰다: CSR의 `.T`는 같은 세 배열의 CSC다(scipy와 같다).
- **사용자의 `(data, indices, indptr)`**(SP135, SP136): `indptr`을 복사하며 줄어들지 않는지 보고, `indices`는 원소 2¹⁵개씩의 고정 조각마다 한 번의 NEON 훑기로 복사하며 범위(부호 없는 비교 하나: 음수도 걸린다)와 내림(ixs[k] ≤ ixs[k − 1]의 수)을 센다. 조각마다 그 안에서 시작하는 비어 있지 않은 줄 가운데 첫 원소에서 내리는 줄을 세고(조각의 첫 줄은 균등 분포의 어림에서 지수 탐색으로 찾는다), 두 수가 같으면 모든 내림이 줄의 시작이라 줄마다 정렬되어 있고 중복이 없다. 그러면 값을 그대로, 아니면 위의 표준 꼴 만들기로 간다. 틀린 곳은 표시된 첫 조각을 다시 훑어 스레드 수와 상관없이 첫 자리를 알린다. `(data, (row, col))`의 `_index_copy`·`_index_max`(모양을 주지 않을 때)도 같은 꼴의 NEON 훑기다(최댓값은 비교와 선택의 사슬이 깊어 레인 여덟 개로 나눴다).
- 밀집 → 희소(`csrc/sparse/convert.cpp`): 줄마다 0이 아닌 수를 세는 패스와 채우는 패스, 줄 구간으로 병렬(임의의 strides). 희소 → 밀집: 0으로 채운 C 순서 배열에 줄마다 쓴다. 중복이 있을 수 있는 COO만 저장 순서로 더하므로 한 스레드다.
- `astype`: 배열의 명시적 변환 규칙(array.md §2.4: float → int는 포화, NaN → 0)으로 `data`를 바꾼다(`sd.astype`과 비트까지 같다). 변환 뒤 0이 된 값도 저장된 원소로 남는다(scipy와 같다).
- DOK(`csrc/sparse/dok.cpp`): 키 (i, j)와 값의 평평한 배열, 열린 주소법과 선형 탐사, 부하 1/2 이하(지운 칸은 표시하고 다시 지을 때 버린다). 다른 형식으로 바꿀 때 키를 정렬한다.
- 그 밖(`convert.cpp`): `indptr` 펼치기, 범위·0으로 거르기(`resize`, `eliminate_zeros`, `nonzero`), 0이 아닌 수 세기(줄마다, 색인마다), `reshape`의 좌표(C 순서의 위치를 128비트 정수로 계산해 2^63을 넘는 크기도 된다).

### 3.2 산술과 원소별 연산 (SP2)
| 연산 | 결과 | 0을 지우나 (scipy와 같게) |
|---|---|---|
| 희소 ± 희소, 희소 `*` 희소, `maximum`·`minimum`, 희소 비교 희소 | 희소 (CSR, 왼쪽이 CSC면 CSC: scipy의 규칙) | 지운다 |
| 희소 `*`·`/` 스칼라, `-A`, `abs(A)`, f(0) = 0인 원소별 함수, `power` | 희소, 같은 패턴 | 지우지 않는다 |
| 희소 `*`·`/` 밀집, `A.multiply(dense)` | 희소, A의 형식(SP38; scipy는 COO) | 지우지 않는다 |
| 희소 ± 밀집, 희소 비교 밀집, `maximum`·`minimum`(밀집), 희소 ÷ 희소 | 밀집 `sd.ndarray` | — |
| 희소 비교·`maximum`·`minimum` 스칼라 | 희소 (CSR, A가 CSC면 CSC) | 지운다 |
| 희소 `@` 밀집, 밀집 `@` 희소 | 밀집 `sd.ndarray` (1차원 @ 1차원은 Python 스칼라) | — |
| 희소 `@` 희소 | 희소 (CSR, 왼쪽이 CSC면 CSC; 1차원 결과는 COO, scipy와 같다) | 지운다 |
| `A + c`(c ≠ 0), `A ** 0` | `ValueError` (SP39, §2.5) | — |

- **값은 배열의 연산이다**: 저장된 값마다 배열의 같은 연산(`array/kernels/scalar_ops.hpp`의 `ops::Add` 등을 커널이 그대로 부르거나, Python 층이 `sd.multiply` 등을 `data`에 부른다)이라, 그 자리의 밀집 연산과 비트까지 같다. 결과 dtype은 배열의 승격 규칙이다(bool + bool은 int64, bool의 `maximum`은 bool, `-A`의 bool은 int64: 모두 배열과 같고 scipy와는 다르다). 정수 산술은 배열처럼 wrap한다.
- **희소끼리**: 모양이 같아야 한다(`shapes could not be broadcast together: (2, 3) (3, 2)`, `ValueError`; 희소끼리의 브로드캐스트는 없다). 두 정렬된 줄의 병합이다: 한쪽에만 있는 자리는 다른 쪽의 암시적 0과의 연산(`op(a, 0)`, `op(0, b)`)이다. 줄마다 독립이고 행 구간으로 병렬이다. 한 스레드에서는 scipy처럼 가장 큰 크기의 배열에 한 번에 쓰고 자른다(두 번 병합하는 병렬 길과 같은 결과).
  - **`*`는 두 쪽이 다 저장한 자리만이다**(SP12의 강한 0): `nan * 암시적 0`은 0이다. `[scipy와 다름]`: scipy의 `csr_elmul_csr`는 합집합에서 `a * 0`을 계산해 한쪽의 NaN·inf가 NaN으로 저장된다(SP2에서 확인).
  - **비교는 IEEE대로다**: 두 암시적 0 사이가 참인 비교(`==`, `<=`, `>=`)는 나머지 모든 자리에 True를 저장한다(scipy처럼 True가 많은 희소 bool; 경고 없음). `[scipy와 다름]`: scipy의 `<=`·`>=`는 `>`·`<`의 부정이라 NaN과의 비교가 True가 된다. scipy의 모양이 다른 `==`·`!=`(Python bool을 돌려줌)와 달리 이쪽은 `ValueError`다.
  - 결과 형식(SP42): 왼쪽이 CSC면 CSC, 아니면 CSR이다(다른 쪽은 그 형식으로 바꾼다). `[scipy와 다름]`: scipy의 `dok + 희소`는 DOK(만든 0도 남는다)인데 `dok - 희소`는 CSR이다. 이쪽은 하나의 규칙(CSR)이다.
- **스칼라**: `A * c`, `c * A`, `A / c`는 저장된 값에만 하고 패턴과 형식(DOK 포함)을 지킨다(scipy와 같다; `A * 0`은 패턴이 남는다). COO의 중복은 그대로 둔다(선형이라). `A / c`는 배열의 나눗셈이다(`[scipy와 다름]`: scipy는 `A * (1/c)`라 끝자리가 다를 수 있고 c = 0이면 `ZeroDivisionError`).
  - 비교, `maximum`, `minimum`: op(0, c)가 0(False)이면 저장된 값에만 하고 만든 0을 지운다(CSR, A가 CSC면 CSC; scipy와 같다). op(0, c)가 0이 아니면 모든 자리가 결과에 든다: 비교는 저장되지 않은 자리가 모두 True, `maximum(A, c > 0)`·`minimum(A, c < 0)`(NaN 포함)은 scipy처럼 `A.toarray()`에 한 결과를 A의 형식으로 담는다(SP47; 경고 없음). 제자리 연산자(`A *= c`)는 두지 않는다(SP43).
- **밀집**(`sd.ndarray`, SP35, SP41): 밀집 피연산자는 A의 모양으로 브로드캐스트되어야 한다(결과의 모양이 A의 모양; 아니면 `ValueError`). scipy도 `+`·`-`는 그렇다. `[scipy와 다름]`: scipy의 `multiply`는 희소 행 벡터 × 밀집 열 벡터처럼 결과가 A보다 커지는 브로드캐스트와 희소끼리의 브로드캐스트도 받는다(SP41).
  - `A * D`, `D * A`, `A.multiply(D)`, `A / D`: D를 저장된 자리에서 읽어(`csrc/sparse/binop.cpp`의 `_gather_*`) 배열의 곱·나눗셈을 한다. 암시적 0은 D가 NaN·inf여도 0이다(SP12, scipy와 같다). A의 형식이다(SP38).
  - 비교, `maximum`·`minimum`: `A.toarray()`와 D의 배열 연산이다(밀집 결과; 그래서 그 비트는 정의상 배열과 같다). `A ± D`, `D ± A`(A가 CSR·CSC이고 결과가 bool이 아닐 때)는 D + 0, 0 − D, D − 0을 배열 연산으로 만들고 저장된 값 v' = 0 + v를 제자리에 더하거나 뺀다(`_dense_update`, SP136): v' + d = (d + 0) + v', v' − d = (0 − d) + v', d − v' = (d − 0) − v'이라 ±0의 부호까지 `A.toarray()` ± D와 같다(정확한 0 항은 −0.0을 +0.0으로만 바꾸고 다른 쪽도 그렇다). 밀집 패스가 둘에서 하나가 된다. COO·DOK는 `A.toarray()`의 길이다(중복의 합 순서 때문). `D / A`와 `c / A`는 scipy처럼 받지 않는다(`TypeError`).
- **numpy 배열·리스트는 피연산자가 아니다**(SP35): 연산자는 `NotImplemented`를 돌려 Python이 `TypeError`를 낸다. `==`·`!=`는 다른 쪽의 비교를 물어 보고 그것도 받지 않으면 `TypeError`다(배열의 규칙: 조용한 False가 되지 않는다). 메서드(`multiply` 등)는 `multiply(): 'other' must be a sparse array, an sd.ndarray or a number, not numpy.ndarray`다. numpy의 스칼라(`numpy.float64`, `numpy.bool` …)는 스칼라다(배열의 연산자와 같다). `A @ 2`는 배열처럼 `TypeError`다(`[scipy와 다름]`: scipy는 `ValueError`).
- **원소별 함수**(SP36): 메서드로만 둔다. scipy의 f(0) = 0인 메서드 목록(`_ufuncs_with_fixed_point_at_zero`)의 함수를 배열의 이름으로 둔다(SP46): `sin tan asin atan sinh tanh asinh atanh sqrt expm1 log1p deg2rad rad2deg floor ceil trunc round sign`. 값은 같은 이름의 배열 함수를 저장된 값에 쓴 것과 비트까지 같다. `[scipy와 다름]`: scipy의 이름은 `arcsin arctan arcsinh arctanh rint`이고 별칭은 없다(SP24). `deg2rad`·`rad2deg`도 배열의 `sd.deg2rad`·`sd.rad2deg`를 저장된 값에 쓴다(SP8; 전에는 배열에 없어 같은 식 x·(π/180), x·(180/π)를 따로 썼다. 값은 numpy와 비트까지 같고 complex는 `TypeError`). 결과의 형식은 scipy의 것이다: COO는 중복을 먼저 더한 표준 꼴(scipy는 A를 제자리에서 바꾸고, 이쪽은 A를 바꾸지 않는다), DOK는 CSR(scipy의 DOK에는 이 메서드들이 없다; `abs`·`power`의 scipy 결과가 CSR이라 그것을 따랐다), CSR·CSC는 자신의 형식. `abs(A)`, `power`, `**`도 같은 규칙이다. `-A`는 선형이라 COO의 중복과 DOK를 그대로 둔다(scipy와 같다).
  - `sd.sin(A)` 등 배열의 원소별 함수는 희소 배열을 받으면 그 메서드나 연산자를 짚는 `TypeError`다(`sin(): 'x' must be an sd.ndarray or a bool, int, float or complex, not csr_array; for a sparse array use A.sin()`, 메서드가 없는 함수는 `use cos(A.toarray())`). 배열 영역의 `csrc/array/ops/ufunc.cpp`에 실패한 호출에서만 보는 판별 하나를 더했다(다른 입력의 결과와 골든은 그대로다).
  - `power(n)`, `A ** n`: n은 실수(bool·int·float, numpy 포함)이고 n > 0이다. n = 0은 `ValueError`(SP39), n < 0과 NaN도 `ValueError`(0 ** n이 0이 아니다; scipy는 음수 지수를 받아 암시적 0을 0으로 둔다; SP44), complex n은 `TypeError`다. scipy의 `dtype=`은 없다.
- **SpGEMM**: Gustavson의 행 단위 알고리즘. 행마다 밀집 누산기(스레드마다 한 줄 길이, 표시(stamp)로 비우지 않고 다시 쓴다)로 k 오름차순에 더하고, 결과 행의 색인을 오름차순으로 꺼낸다: SP8부터는 만난 자리의 비트 집합(64자리마다 한 단어, 단어마다 요약 비트 하나)에서 요약 단어 n/4096개를 훑고 켜진 비트만 찾는다(count trailing zeros; 행의 평균 일의 4배 + 16보다 요약이 짧을 때; SP116). 그 밖은 전처럼 정렬한다(배열 영역의 NEON 정렬, `array/kernels/simd_sort.hpp`; 값이 서로 달라 불안정 정렬이 보이지 않는다). 순서가 정확히 같아 비트가 같다. 일(방문하는 B의 원소 수)로 나눈 행 구간으로 나눈다. 스레드풀에서는 두 패스(기호: 행마다 다른 자리 수, 수치: 채우기)이고, 풀을 쓸 수 없으면(1스레드, 작은 일) 구간마다 자기 버퍼에 한 패스로 쓰고 제자리로 옮긴다(SP136: 기호 패스가 B를 두 번 읽었다; 풀에서는 구간 버퍼의 할당이 손해라 두 패스가 빠르다). 결과는 어느 길이든 같다. 정확한 0은 지운다(scipy). SP2에서는 1스레드의 시간 대부분이 행 정렬이었다(비트 집합으로 1.3–1.7배 줄었다, §5): scipy의 `csr_matmat`은 행을 정렬하지 않으므로 표준 꼴까지 견주면(`sort_indices()`를 더하면) 이쪽이 빠르다.
- SpMV·SpMM: §2.6의 순서. 행마다 FMA 사슬 하나이고, SpMV(반복법·고유값 풀이의 CSR 곱도)는 네 행의 사슬을 한 고리에서 함께 돌린다(`arith.hpp`의 `line_dots`, SP136: 행마다 항 순서는 그대로; 짧은 행은 사슬 하나의 지연이 시간이었다). SpMM은 X의 행이 연속이면 열 16개를 한 묶음으로 NEON 레인에 두고(원소 하나의 항 순서는 그대로다), 8원소 뒤의 X 행을 미리 읽는다(SP8; 무작위 행 읽기가 시간의 대부분이었다).

### 3.3 인덱싱과 리덕션 (SP2, SP3)
- 리덕션(SP2): `sum`, `mean`, `max`, `min`, `nanmax`, `nanmin`, `argmax`, `argmin`(`count_nonzero`는 SP1). `trace`, `diagonal`은 §8대로 SP3이다. 순서는 §2.6. `axis`가 None이면 Python 스칼라, 있으면 **언제나 밀집 `sd.ndarray`**다(SP37; `[scipy와 다름]`: scipy는 `sum(axis=)`·`mean(axis=)`이 밀집, `max(axis=)`·`min(axis=)`이 COO 희소). 1차원의 `axis=0`은 전체 리덕션이라 스칼라다(배열과 같다). `keepdims`는 배열과 같은 뜻이다(SP40; `axis=None`이면 모양 (1, 1)의 밀집 배열). `axis`는 int나 None만 받는다(`count_nonzero`와 같다; 배열의 tuple 축은 받지 않는다; SP45). 빈 배열의 `max`·`argmax`는 배열의 메시지로 `ValueError`(`max(): undefined for an empty array (or along an empty axis)`), `sum`은 0, `mean`은 NaN이다(배열과 같다; scipy의 빈 `mean`은 `ZeroDivisionError`). bool의 `sum`은 int64, `mean`은 float64다. `sum(dtype=)`은 배열의 `sum`의 `dtype=` 규칙과 메시지다. dtype이 받아지지 않는 것(complex의 `max`)은 배열의 메시지로 `TypeError`다.
- `max`·`min`은 줄마다 저장된 값을 배열의 `maximum`·`minimum`으로 접고(NaN 전파), 줄이 꽉 차 있지 않으면 암시적 0과 한 번 더 접는다(scipy와 같다). `nanmax`·`nanmin`은 NaN을 건너뛴다(모두 NaN이고 암시적 0이 있으면 0). `explicit=True`면 저장된 값만 본다(SP40; 빈 줄은 0, scipy와 같다). 그래서 `explicit=False`의 값은 `sd.max(A.toarray(), axis=)`와 같다(±0의 부호만 다를 수 있다: `toarray`가 저장된 -0.0을 +0.0으로 만든다).
- `argmax`·`argmin`: 암시적 0도 자리로 센다. 동점이면 C 순서의 첫 자리, NaN이면 첫 NaN의 자리다(배열과 같다; `[scipy와 다름]`: scipy는 NaN이 있어도 첫 암시적 0의 자리를 돌려준다). 2차원에서 `axis=None`이면 펼친 C 순서의 자리다. `explicit=True`면 저장된 값 중에서고 빈 줄은 0, 아무것도 저장되지 않은 배열의 `axis=None`은 `ValueError`다(scipy와 같다).
- **인덱싱**(SP3, `src/stardust/sparse/_index.py`, `csrc/sparse/index.cpp`): 키는 배열의 규칙으로 읽는다(docs/api/array.md §5): int(음수는 끝에서), slice(음수 step 포함), `...`, None, 정수 배열(반복과 아무 순서), bool 마스크(1차원은 축 하나, 2차원은 두 축), 그 조합. 범위 밖·틀린 꼴·떨어진 고급 인덱스의 예외와 메시지는 배열의 것과 같다(테스트가 `A.toarray()[key]`의 에러와 글자까지 견준다). 결과가 3차원 이상이 되는 키(`A[None, :]`, 2차원 정수 배열 하나)는 `IndexError`다(SP13).
  - **결과**: 모든 축이 int면 Python 스칼라(저장된 값, 없으면 그 dtype의 0; COO의 중복은 저장 순서로 더한 값, 곧 표준 꼴의 값). 아니면 희소 배열이고 `A[key].toarray()`는 `A.toarray()[key]`와 비트까지 같다. 결과의 차원 수가 A와 같으면 A의 형식(CSR은 CSR, DOK는 DOK), 다르면(2차원의 `A[i]`·`A[:, j]`, 원소 고르기 `A[rows, cols]`, 1차원의 None) COO다(scipy와 같다). `[scipy와 다름]`: scipy의 CSR·CSC는 1차원 원소 고르기와 2차원 마스크에 밀집 numpy 배열을, 2차원 원소 고르기에 밀집 값에서 다시 만든 배열(저장된 0이 빠진다)을 돌려준다. 이쪽은 네 형식이 같은 규칙이다(SP54; 1차원 CSR·DOK의 2차원 정수 배열 결과도 COO다; scipy는 그 형식).
  - **고르는 꼴**: 두 축에 배열이 있으면(정수와 배열이 함께여도 배열의 규칙으로 브로드캐스트) 원소 고르기, 아니면 축마다의 고르기의 곱(바깥 고르기)이다. 원소 고르기가 바깥 곱의 꼴(`rows[:, None]`와 `cols`)이면 바깥 고르기로 한다.
  - **저장되는 것**: 고른 자리에 저장된 원소가 결과에 저장된다(저장된 0도; 원소 고르기에서 같은 자리가 여러 번이면 여러 번). COO는 중복과 저장 순서를 그대로 가져간다(scipy와 같다): 원소마다 그것을 고르는 결과 자리 쌍마다 하나씩, 원소 순서, 행 자리, 열 자리 순서다. `[scipy와 다름]`: scipy의 DOK는 경로에 따라 저장된 0을 빼거나(대부분) 두거나(슬라이스 × 슬라이스의 O(nnz) 길) 한다. 이쪽 DOK는 언제나 둔다(DOK에 저장된 0은 다른 형식에서 바꿀 때만 생긴다).
  - **커널**: 축 하나의 고르기는 "전부 / (start, step, count) / 정수 배열"이고, 결과 자리를 원래 색인으로부터 찾는 표(원래 축 길이가 16 × 고른 수 + 65,536 이하이면 원래 색인마다 결과 자리의 시작; 넘으면 원래 색인으로 정렬한 자리의 이분 탐색; step ≠ ±1인 slice도 같은 표)를 만든다. CSR·CSC의 바깥 고르기는 결과 줄마다 원래 줄을 훑어 두 번(세기, 채우기) 쓴다: step 1인 slice는 줄 안의 구간(짧은 줄은 앞에서 훑고 긴 줄은 이분 탐색)을 통째로 옮기고, 정렬된 배열·양의 step은 줄 순서대로, 음의 step은 줄을 거꾸로, 정렬되지 않은 배열은 줄의 결과를 자리로 정렬한다(한 줄의 결과 자리는 서로 다르다; 첫 패스가 센 수로 고른 것이 없는 줄은 건너뛰고 하나뿐인 줄은 정렬하지 않는다, SP136). 결과는 다시 표준 꼴이다. COO와 DOK(키를 정렬한 COO를 거친다)는 원소마다 같은 표를 쓴다. 원소 고르기는 CSR·CSC에서 쌍마다 줄 안의 탐색, COO에서 쌍을 (행, 열)로 안정 정렬한 뒤 원소마다의 탐색이다. 일은 결과 줄(이나 원소)의 고정 구간으로 나누므로 결과는 스레드 수와 무관하다.
- **대입** `A[key] = v`(SP3, SP50): 같은 키 규칙이고 고른 영역의 모양(None의 1 포함)에 `v`를 배열의 규칙으로 맞춘다(docs/api/array.md §5.3): 원소 하나의 자리에는 Python 스칼라만(`[scipy와 다름]`: scipy는 `[3]`도 받는다), 그 밖은 영역의 모양으로 엄격하게 브로드캐스트되는 값(`sd.ndarray`, 시퀀스, 버퍼)이나 희소 배열. 값은 배열의 대입 규칙으로 A의 dtype이 된다(암묵적 하향 변환은 `TypeError`; `[scipy와 다름]`: scipy는 2.5를 int64에 2로 넣는다). 같은 자리가 여러 번 나오면 마지막 값이다(배열의 scatter와 같다; `[scipy와 다름]`: scipy의 COO는 첫 값). 저장되는 것은 형식마다 scipy의 것이다(SP53):
  - CSR·CSC: 이미 저장된 자리는 `data`를 제자리에서 바꾸고(값 배열을 나누는 다른 배열·`.T`·`to_scipy()` 뷰에도 보인다; scipy와 같다; 값 배열이 쓰기 금지면 새 값 배열), 새 자리는 줄마다 병합해 세 배열을 새로 만든다(O(nnz), 경고 없음; SP50). 대입한 0도 저장된다(scipy와 같다). 희소 값은 영역 안에 저장된 자리를 0으로 바꾼 뒤(패턴은 그대로) 값의 저장된 원소를 넣는다(scipy와 같다).
  - DOK: 0(-0.0 포함)을 넣으면 그 키를 지운다(SP1, scipy와 같다). 희소 값은 그 밀집 값으로 영역 전체를 넣는다(scipy와 같다).
  - COO: 영역의 자리에 저장된 원소(중복 포함)를 지우고, 0이 아닌 새 값을 행 우선 순서로 뒤에 붙인다(원소 하나의 대입과 희소 값의 저장된 원소는 0이어도 붙인다; scipy와 같다). `has_canonical_format`은 다시 알아낸다.
- **대각**(SP3): `diagonal(k=)`는 `A.toarray()`의 k번째 대각과 비트까지 같은 밀집 1차원 `sd.ndarray`다(`sd.linalg.diagonal`과 같다: 저장된 -0.0은 +0.0, COO의 중복은 저장 순서의 합; 대각이 밖이면 길이 0). `trace(k=)`는 그 대각의 `sd.sum`(Python 스칼라; bool·int64는 int64, scipy와 같다. float64는 `sd.linalg.trace(A.toarray(), k=k)`와 비트까지 같다; SP55). `setdiag(values, k=)`는 scipy의 뜻이다: 수면 대각 전체, 1차원이면 앞의 `len(values)`개(길면 자른다). 자리마다 위의 대입 규칙(CSR·CSC·COO는 0도 저장, DOK는 지움)이고 COO는 scipy처럼 옛 대각 원소를 지우고 새 것을 붙인다. 1차원 배열은 `ValueError`, k가 모양 밖이면 `ValueError`다(scipy와 같다). 인자 이름은 `k`다(sd.linalg의 `trace`·`diagonal`; scipy `trace`의 `offset`은 없다).

### 3.4 만들기 함수 (SP3)
- 시그니처는 docs/api/sparse.md §4다. 파일은 `src/stardust/sparse/_construct.py`와 `csrc/sparse/construct.cpp`(`_kron`, `_kronsum`, `_diags`, `_stack_minor`, `_stack_major`, `_band_coo`, `_band_compressed`)다.
- **데이터 인자**는 희소 배열이나 밀집 `sd.ndarray`다(밀집은 `coo_array(x)`처럼 0이 아닌 값만). 리스트와 numpy 배열은 배열의 함수처럼 `TypeError`다(API.md §5; `[scipy와 다름]`: scipy는 받고, 모두 밀집이면 `DeprecationWarning`). `diags_array`의 대각은 `sd.ndarray`나 수다(SP57).
- **`format=None`의 결과**(SP51): scipy의 규칙이되 DIA·BSR 대신 CSR이다. `eye_array`·`diags_array`(scipy DIA), `kron`(scipy BSR; 이쪽의 값은 scipy의 `format='csr'`과 같다: scipy의 BSR은 B의 0까지 블록에 저장한다), `kronsum`(scipy와 같다), `hstack`·`vstack`·`block_array`(블록이 모두 CSR이면 CSR, 모두 CSC면 CSC, 아니면 COO; scipy와 같다), `block_diag`·`tril`·`triu`·`random_array`(COO). `format=`에 `'dia'`·`'bsr'`·`'lil'`은 `ValueError`다.
- **dtype**: `dtype=None`은 입력들의 배열 승격이고(SP52: 정수 대각은 int64), `dtype=`은 명시적 변환(배열의 `astype`)이다. `kron`의 값은 배열의 `multiply`라 bool × bool은 int64다(SP56; `[scipy와 다름]`: scipy는 bool). scipy 1.18.1의 `kron`은 한쪽이 아무것도 저장하지 않으면 dtype을 잃고 float64를 돌려준다(이쪽은 승격한 dtype).
- **값**: 모든 값이 입력의 값이거나 두 값의 배열의 곱(`kron`)이라 `toarray()`는 밀집 결과(`sd.eye`, `sd.tril`, `sd.triu`, 밀집 블록)와 비트까지 같다. `sd.linalg.kron(A.toarray(), B.toarray())`와는 0의 부호만 다를 수 있다(밀집의 0 × (−x)는 −0.0, 희소는 저장하지 않는다: 강한 0).
- `eye_array(n_rows, n_cols=None, *, k=0, dtype=None)`: `sd.eye`의 인자와 기본값(float64)이다. 대각이 밖이면(`k ≥ n_cols`) 빈 배열이다(`sd.eye`와 같다; `[scipy와 다름]`: scipy는 `ValueError`).
- `diags_array(diagonals, *, offsets=0, shape=None)`: offsets가 int면 대각 하나(`sd.ndarray`나 수), 리스트면 같은 길이의 대각 리스트다. 대각의 길이가 1이면 그 대각 전체, 더 길면 자른다(scipy와 같다), 짧으면 `ValueError`. 같은 offset이 두 번이면 `ValueError`(scipy와 같다), 모양 밖의 offset은 `ValueError`(모서리 바로 바깥은 빈 대각; scipy와 같다). shape가 없으면 정사각 `len(diagonals[0]) + |offsets[0]|`이다. 0은 저장하지 않는다(scipy의 DIA → CSR·COO와 같다). 결과는 `_diags`가 바로 채운 CSR이다(SP136): offset을 정렬해 두고 행 i가 걸치는 대각(−i ≤ offset ≤ n − 1 − i; 행이 늘면 두 끝이 함께 내려간다)을 offset 순서로 적으면 열이 오름차순이다. 행 구간의 시작 자리는 대각마다 그 앞 행에 놓인 수의 합이라 한 패스이고, 0이 있으면 그 뒤에 `_filter_compressed`로 거른다. 대각 수가 (행마다 평균 저장 수 + 1)의 네 배를 넘으면(대각 대부분이 행 대부분을 비켜 가면) 행마다 대각을 훑는 일이 커서 COO를 거쳐 표준 꼴로 만든다(같은 결과).
- `random_array(shape, *, density=0.01, dtype=None, rng, data_sampler=None)`: `rng`는 `sd.random.Generator`이고 꼭 받는다(SP49). 자리는 `rng.choice(m·n, round(density·m·n), replace=False)`로 뽑은 C 순서의 위치이고(m·n이 2⁶³ − 1을 넘으면 축마다 `integers`로 뽑아 처음 나온 자리만 둔다; scipy와 같은 꼴), 그 순서 그대로의 COO다. 값은 `data_sampler(n)`(위치 인자 하나; 결과는 `sd.asarray`의 규칙, `dtype=`이면 명시적 변환) 또는 dtype마다: float64(None)는 `rng.random(n)`([0, 1)), int64는 `rng.integers(−2⁶³, 2⁶³ − 1)`, complex128은 실수부 `random(n)` 다음 허수부 `random(n)`, bool은 `random(n)`의 0이 아닌 것(scipy와 같은 뜻). 같은 `rng` 상태면 스레드 수와 상관없이 같은 비트다. 난수 생성기가 달라 scipy와 같은 값은 아니다. `[scipy와 다름]`: scipy는 `data_sampler(size=n)`으로 부르고 `dtype=None`이면 결과를 float64로 바꾼다(이쪽은 그 dtype; SP58). density가 [0, 1] 밖이면 `ValueError`다.
- **넘침과 큰 할당**: `kron`·`kronsum`·쌓기·`block_diag`는 결과의 행·열 수를 Python int로 먼저 계산해 int64를 넘으면 `OverflowError("kron(): the result's shape is out of range")`다(검사하지 않으면 넘친 모양과 음수 색인의 배열이 생기고, `kron`은 indptr을 2⁶⁴번 써서 프로세스가 죽는다). C++ `_kron`에도 행 수·열 색인·저장 수의 넘침 검사(`__builtin_mul_overflow`)가 있다. 모든 indptr은 `new_indptr`(줄 수 + 1을 계산하기 전에 2⁶³ − 1 줄을 거절, `array is too big`)로 만든다. 커널의 할당 실패(`cannot allocate`, `array is too big`)는 Python 쪽에서 사용자가 부른 함수 이름으로 바꾼다(`C.sized`, `_convert`, 생성자의 `build`). `k`·`offsets`·`n_rows`·`n_cols`는 int64 범위를 검사한다(`C.int64_arg`).
- `kron(A, B)`: 결과의 행 i·p + r은 A의 행 i의 저장된 원소마다(열 오름차순) B의 행 r의 원소를 잇는다(열 j·q + l): 그래서 정렬 없이 표준 꼴의 CSR이다. 저장된 0과 곱해서 0이 된 값도 저장된다(scipy의 COO 길과 같다). 1차원은 한 행으로 들고, 둘 다 1차원이면 1차원 결과다(scipy와 같다). 결과 행의 고정 구간으로 병렬이다.
- `kronsum(A, B)`: `kron(I_n, A) + kron(B, I_m)`(정사각만; scipy와 같다). SP8부터 단위 행렬과 곱하지 않고 값을 놓는다(SP111): A의 원소를 대각 블록마다, bᵢⱼ를 블록 (i, j)의 대각에 COO로 잇고 CSR로 바꾸면 겹치는 자리(aₚₚ와 bᵢᵢ)만 a + b(입력 순서)로 더해지고, 정확한 0은 지운다(희소끼리의 `+`와 같은 결과의 패턴). 값의 dtype은 `kron`의 규칙(bool은 int64)이다. A·B가 비어 있지 않으면 `_kronsum`이 행 i·m + p를 바로 채운다(SP136): B의 행 i에서 블록 i의 왼쪽 원소(열 j·m + p), A의 행 p를 블록 i에(bᵢᵢ의 자리 i·m + p에서 겹치면 add_value(a, b)), 오른쪽 원소의 차례이고, 0(−0.0 포함)은 적지 않는다. COO 길과 비트가 같다. 전에는 1 × a의 복소 곱이 0 + inf·j를 NaN + inf·j로, 한쪽만 있는 자리의 a + 0이 허수부 −0.0을 +0.0으로 바꿨다. 실수·정수 입력의 비트는 그대로다.
- `hstack`·`vstack`·`block_array`·`block_diag`: 블록은 희소 배열, 밀집 `sd.ndarray`, None(`block_array`·`hstack`·`vstack`의 빈 블록: 그 행·열의 다른 블록이 크기를 정한다)이고 1차원 블록은 한 행이다(scipy와 같다). 행·열의 크기가 맞지 않으면 scipy의 글(`blocks[0,:] has incompatible row dimensions: ...`)의 `ValueError`다. CSR끼리(CSC끼리)는 줄을 잇는 커널(`_stack_minor`: 같은 줄을 블록 순서로 이어 붙임, `_stack_major`: 줄들을 차례로)이라 결과가 바로 표준 꼴이고, 그 밖은 블록의 COO 원소를 블록 순서(행 우선)로 잇는다(scipy와 같은 순서). `block_diag`의 밀집 블록은 0이 아닌 값만이다(`[scipy와 다름]`: scipy는 밀집 블록의 모든 자리를 0까지 저장한다).
- `tril(A, k=)`, `triu(A, k=)`: col ≤ row + k(col ≥ row + k)인 원소를 A의 COO 순서 그대로(중복도) 남긴다(scipy와 같다). CSR·CSC는 줄이 정렬되어 있어 줄마다 남는 것이 앞부분이나 뒷부분이다: `_band_compressed`가 경계를 세어(긴 줄은 이분 탐색) 그 구간을 바로 옮긴다(`tocoo()`를 거친 것과 같은 COO; SP136). COO·DOK는 원소마다 거른다(`_band_coo`). `sd.tril`·`sd.triu`와 인자 이름이 같다. 2차원만이다.
- `find(A)`: 중복을 더한 표준 꼴에서 0이 아닌 값의 `(row, col, data)`, 쓸 수 있는 새 밀집 배열 셋이다(scipy와 같다). 2차원만이다(`[scipy와 다름]`: scipy의 1차원은 행 0의 배열을 돌려준다).

### 3.5 전치, 모양, 그 밖 (SP1)
- `.T`, `.mT`, `transpose()`, `matrix_transpose(A)`: 2차원이면 같은 일이다(배열과 같은 이름들; `.T`는 뷰). 1차원에서 `.T`, `.mT`, `matrix_transpose`는 `sd.ndarray`처럼 `ValueError`이고(SP31)(`.T: the array must have 2 dimensions, got 1`), `transpose()`는 자신이다. `[scipy와 다름]`: scipy의 1차원 `.T`는 자신이다(numpy와 같다). 영역 사이에서 같은 이름이 같은 일을 하도록 sd의 배열을 따른다.
- `conj()`, `real`, `imag`, `copy()`, `astype`, `reshape`(COO를 거친다, 결과는 COO), `resize`, `nonzero()`, `eliminate_zeros()`(COO·CSR·CSC; scipy처럼 DOK에는 없다), `tocoo`·`tocsr`·`tocsc`·`todok`, `asformat`, `count_nonzero`, `nnz`, `shape`, `ndim`, `size`, `dtype`, `format`. `setdiag`은 SP3이다.
- 실수 dtype의 `real`은 값 배열을 나누고(scipy와 같다) `conj()`는 복사다. `copy()`는 값만 복사하고 쓰기 금지인 색인 배열은 나눈다(바뀌지 않으므로 보이는 차이가 없다).
- `resize(shape)`: 제자리에서 바꾸고 None을 돌려준다. 차원 수는 그대로다(`[scipy와 다름]`: scipy의 COO는 1·2차원 사이도 바꾸고(펼친 순서로), DOK는 `NotImplementedError`다; 한 규칙으로 둔다).
- 두지 않는 것: `dot`(`@`와 같다), `conjugate`(`conj`와 같다), `todense`(`toarray`와 같다), `power`의 정수 지수 0, `prune`(표준 꼴이라 할 일이 없다), `getrow`·`getcol`(옛 꼴), scipy의 `reshape(order=)`·`transpose(axes=, copy=)`·`astype(casting=)`.

### 3.6 직접 풀이 (SP4)
- **LU는 Accelerate의 `SparseFactorizationLU`(문턱 부분 피벗, `pivotTolerance` 0.01)를 AMD 순서로 부른다**(SP18, SP59). LU의 기본 순서 MTMetis와 `SparseOrderDefault`는 쓰지 않는다: 실행마다 결과가 다르고 동시 호출에서 프로세스가 죽는다(§1.4). Metis는 동시 호출에서 실행마다 다르다. 3차원 격자에서는 AMD의 채움이 더 많아 Metis보다 1.5–2배 느리다(§5). 순서를 고르는 옵션은 없다(SP59).
- **Cholesky·LDLᵀ는 첫 판에 쓰지 않는다**(SP18). Accelerate의 대칭 분해는 기본 스레드 모드에서 실행마다 비트가 다르고, 프로세스 안에서 그것을 막을 길이 없다(`BLASSetThreading`도 듣지 않는다, §1.4). scipy.sparse.linalg의 공개 함수에는 대칭 분해가 없으므로(모두 SuperLU의 LU다) 범위는 줄지 않는다. 대칭 행렬도 LU로 푼다.
  - 나중 후보: 이 라이브러리가 결정적인 희소 Cholesky를 짠다(초마디 Cholesky: 기호 분석과 제거 트리, 초마디마다 밀집 `potrf`·`trsm`·`syrk`, 트리 위의 병렬은 형제의 결과를 고정 순서로 더한다). CHOLMOD 꼴이라 큰 일이고 나중 후보다(regression의 정규방정식, eigsh의 shift-invert에 쓰인다).
- **입력**: A는 희소 배열이다(밀집 `sd.ndarray`는 `sd.linalg.solve` 등을 짚는 `TypeError`). 모든 형식을 CSC로 바꿔 분해한다: Accelerate에 CSR을 전치 속성으로 넘기면 같은 행렬의 CSC와 끝자리가 다르고(형식이 비트를 바꾼다), 그 인수의 여러 우변 풀이가 틀리기 때문이다(§1.4). CSR → CSC는 O(nnz)라 분해에 비해 작다. 값은 float64(bool·int64에서)나 complex128이고, 저장된 NaN·inf는 `ValueError`(`'A' contains NaN or infinity`, linalg와 같은 문장)다. 우변 b는 밀집 `sd.ndarray`((n,)이나 (n, k); numpy 배열과 리스트는 `TypeError`, API.md §5)이고 `spsolve`는 희소 b도 받는다.
- **특이의 판정**(SP62, `csrc/sparse/direct.cpp`): 분해 뒤에 한 번 한다(`splu`는 만들 때, `spsolve`·`inv`·`factorized`는 풀기 전). 셋 중 하나면 `LinAlgError("<함수>(): singular matrix")`다.
  1. 구조적 특이: Accelerate의 `SparseMatrixIsSingular`.
  2. 조건수 추정 안의 풀이가 유한하지 않다(정확한 0 피벗; `zeroTolerance = 0`이라 inf가 나온다; 넘침도 여기에 든다).
  3. κ₁ = ‖A‖₁ · est(‖A⁻¹‖₁)이 κ₁·ε ≥ 1이다. est는 LAPACK의 1-노름 추정기 `lacn2`(Hager 1984, Higham 1988; `gecon`이 쓰는 것, linalg의 래퍼)를 A와 Aᴴ의 인수 풀이로 돌린 것이다. 시작 벡터가 정해져 있어 결정적이고, 참값의 하한이다. 풀이는 보통 4–5번이다. SP7의 `onenormest`(Higham–Tisseur의 블록 t = 2)와 같은 꼴이지만 t = 1이고 무작위 열이 없다.
  - 잰 것(SP4, 시험 행렬 64개 중 정방 전부): 특이로 판정한 것은 정확히 특이한 행렬 모두(Neumann 라플라시안 lap1d/10n·lap2d/10n·lap2d/60n, 같은 열 dupcol/40·dupcol/1000, 상쇄 행렬 cancel/50·cancel/400(dense κ₁ 3×10²⁴·3×10²⁸, 랭크 47/50·382/400), 구조적 특이 셋)이고, 특이가 아닌 행렬은 모두 풀었다(가장 나쁜 graded/400x12의 κ₁ 추정 7.5×10¹⁴, near_singular/20 8×10⁸). 추정값은 작은 행렬에서 dense κ₁의 1/10 이상, 그 이하다(테스트). scipy의 SuperLU는 dupcol/40·lap2d/10n에서 경고 없이 크기 3×10¹⁹·5×10¹⁴의 해를 낸다.
  - SuiteSparse(§7.2)의 sherman3(κ₁ 6.9×10¹⁶)과 cvxqp3(κ₁ 2.2×10¹⁶)은 κ₁·ε ≥ 1이라 `LinAlgError`다(SP66). scipy는 경고 없이 해를 돌려주지만 앞오차 κε가 1을 넘어 맞는 자리가 없다.
  - 비용: 분해 시간의 약 10–15%(2차원 라플라시안 n = 9×10⁴에서 `_lu_solve`의 분해 + 풀이 대 `spsolve`).
  - `[scipy와 다름]`: scipy `spsolve`는 구조적 특이에서 `MatrixRankWarning`과 NaN 배열을, `splu`는 `RuntimeError("Factor is exactly singular")`를 내고, 값으로 특이한 행렬은 경고 없이 큰 유한한 해를 낸다. 여기서는 모두 `LinAlgError`다(linalg와 같은 예외).
- **반복 다듬기**(SP63, SP133): `spsolve`, `SparseLU.solve`, `factorized`, `inv`는 첫 풀이 x̂의 잔차 r = b − op(A)x̂와 성분별 역오차 maxᵢ\|rᵢ\| / (\|op(A)\|\|x̂\| + \|b\|)ᵢ(LAPACK `dgerfs`의 판정; 복소는 \|re\| + \|im\|, 0/0은 0, r/0은 inf)를 C++ `_residual`로 한 번에 계산하고(CSC의 `'N'`은 열 순서의 흩뿌리기, `'T'`·`'H'`는 열을 따라 모으기; 정해진 순서), 그것이 8ε를 넘거나 NaN이면 x ← x̂ + solve(r)로 한 번 다듬는다. 우변 묶음은 가장 나쁜 열이 정한다. 잰 것(시험 행렬, 첫 풀이의 성분별 역오차): 라플라시안·대류확산·띠 행렬은 0.4–8ε라 다듬지 않고, saddle/600x150 4,362·3,117ε(`'N'`·`'T'`; 문턱 피벗 0.01의 성장, 정규화로 404ε), blockdiag/200x8 302·800ε, long_rows/9000 17ε, coo/unsorted 43·153ε, random_spd·random_indef/1500 8.5–9.9ε는 다듬는다. 정규화 역오차는 다듬은 것이 모두 0.6ε 이하, 다듬지 않은 것의 가장 큰 것이 3.54ε(lap3d/22 `'T'`; 계약 4ε)이고, 앞오차는 참값 대비 C_SOLVE·κ∞ε의 0.253배 이하다(늘 다듬을 때 0.097; scipy의 SuperLU는 정규화 0.10–4.4ε). 다듬지 않는 풀이는 시간이 약 절반이다.
- **빈 행렬**: Accelerate는 0 × 0을 받지 않는다(매개변수 오류). `spsolve`·`splu`·`inv`는 Accelerate를 부르지 않고 빈 결과로 답한다(scipy와 같다).
- `spsolve(A, b, /)`: b가 (n,)이면 (n,), (n, k)이면 (n, k) 밀집이다. `[scipy와 다름]`(SP69): scipy는 (n, 1)을 (n,)으로 펴지만 여기서는 (n, 1)이다(`sd.linalg.solve`와 같다). b가 희소면 결과도 희소이고 A의 형식이다(CSR은 CSR, 나머지는 CSC; scipy와 같다). 희소 b는 크기로만 정해지는 열 묶음(약 2²²개 값, 1–64열)마다 밀집으로 풀어 정확한 0을 뺀다(NaN은 남는다; scipy는 열마다 풀어 `flatnonzero`). `[scipy와 다름]`(SP70): scipy는 열이 하나인 희소 b와 1차원 희소 b에 밀집 결과를 주지만 여기서는 희소다(1차원은 CSR). dtype은 A와 b의 승격이다: 실수 A와 복소 b는 실수 인수로 실수부·허수부를 함께 푼다(scipy는 A를 복소로 바꿔 분해한다; 반올림만 다르다).
- `splu(A, /)`: 내보내지 않는 `SparseLU`(SP60). `solve(b, /, *, trans='N')`(`'N'`, `'T'`, `'H'`), `shape`, repr `SparseLU(shape=(n, n), dtype=float64)`. `L`, `U`, `perm_r`, `perm_c`, `nnz`는 없다(Accelerate가 인수를 내보이지 않는다, §1.4). 실수 인수에 복소 b를 주면 실수부·허수부를 함께 푼다(`[scipy와 다름]`: scipy의 `SuperLU.solve`는 `TypeError`; `sd.linalg.lu_solve`와 같다). 한 인수를 여러 Python 스레드가 동시에 풀어도 같은 비트다(GIL을 놓는다). scipy의 SuperLU 옵션(`permc_spec`, `diag_pivot_thresh`, `relax`, `panel_size`, `options`)은 없다(SP61).
- `factorized(A, /)`: `splu(A).solve`(묶인 메서드)다. scipy(umfpack이 없을 때)와 같다.
- `inv(A, /)`: 단위 행렬을 희소 우변으로 푼 결과(scipy와 같다; A의 형식, 정확한 0은 뺀다). 대개 밀집에 가까워 크기 n²의 메모리가 든다.
- **우변이 여럿**(SP64, SP134): 보통 풀이(`trans='N'`)는 Accelerate가 8열씩의 고정 묶음을 함께 푼다(열 하나의 비용이 8열에서 가장 작다). 열 하나의 비트는 혼자 푼 것과 반올림 크기만큼 다를 수 있고 k와 묶음 안의 자리로 정해진다(결정적이다). 전치 풀이(`'T'`, `'H'`)는 Accelerate의 결함 때문에 열마다 따로 푼다(§1.4).
- `spsolve_triangular(A, b, /, *, upper, unit_diagonal=False)`: 이 라이브러리의 대입 커널이다(`csrc/sparse/triangular.cpp`). `upper`는 꼭 주는 키워드다(SP68; `sd.linalg.solve_triangular`와 같다; `[scipy와 다름]`: scipy는 `lower=True`가 기본이다). A를 CSR로 바꾸고(형식과 무관한 비트) 고른 삼각만 읽는다: xᵢ = (bᵢ − sᵢ)/aᵢᵢ, sᵢ는 행의 삼각 원소 Σⱼ aᵢⱼxⱼ를 §2.6의 줄 합(0에서 j 오름차순 FMA, L 조각)으로 더한 것이다. 복소의 나눗셈은 배열의 `divide`와 같은 꼴이다. 열은 서로 기대지 않으므로 우변이 여럿이면 16열씩의 고정 묶음이 스레드풀에서 돌고, 열 하나의 비트는 다른 열이나 k와 무관하다(scipy도 열마다다). 우변 하나는 한 스레드다(단계 스케줄링은 하지 않는다: 나중 후보); 그때는 행마다의 대각 탐색, 특이 판정, 대입을 한 번 훑는다(SP136). 대각이 없거나 0이면 `LinAlgError("spsolve_triangular(): singular matrix")`(linalg.md §2.2의 `zero_on_diagonal`과 같다; 우변이 여럿이면 풀기 전에, 하나면 훑은 뒤에 내고 결과는 버린다). `unit_diagonal=True`면 대각을 읽지 않는다. 결과는 b의 모양이다(scipy와 같다).
  - `[scipy와 다름]`: 고른 삼각 밖의 원소는 읽지 않는다(`sd.linalg.solve_triangular`와 같다). 유한성도 읽는 원소만 본다(SP125). scipy 1.18.1은 그 원소가 있으면 경고 없이 틀린 답을 낸다(SuperLU `gstrs`에 그대로 넘긴다; 문서는 `LinAlgError`라 한다). scipy의 `overwrite_A`, `overwrite_b`는 없다. scipy는 A·diag(1/d)로 단위 삼각을 푼 뒤 1/d를 곱하는 순서라 반올림만 다르다.
  - 정확도: 대입의 표준 역오차 |b − Tx̂| ≤ γₙ|T||x̂|를 정확한 유리수 잔차로 원소마다 확인했다(테스트). 정해진 순서와 비트까지 같다(Python의 `math.fma`로 같은 순서를 짠 것과 견준다).
- `is_sptriangular(A, /)`, `spbandwidth(A, /)`: 패턴만 보는 O(nnz) 도구다(저장된 0도 센다; scipy와 같다). 행마다 첫 열과 마지막 열만 본다(정렬된 표준 꼴). `is_sptriangular`는 가운데·처음·마지막 행을 먼저 보고(scipy와 같은 차례), 아래·위 삼각이 둘 다 거짓이 되면 남은 행을 보지 않는다(SP136; 답은 그때 (False, False)로 정해져 있어 멈춘 자리와 무관하다). `spbandwidth`는 행 구간마다의 최댓값을 구간 순서로 합친다. 아무것도 저장하지 않은 행렬은 `(0, 0)`과 `(True, True)`다(`[scipy와 다름]`: scipy 1.18.1의 `spbandwidth`는 빈 CSR에서 `ValueError`, `is_sptriangular`는 0 × 0 CSR에서 `IndexError`).

### 3.7 반복법 (SP5)
- **이 라이브러리가 짠다**(SP16; SP4의 "알맞은 반복법은 Accelerate"를 바꾼다). 이유:
  - scipy와 같은 멈춤 판정과 상태(`info`)를 지킬 수 없다: Accelerate의 판정은 처음 잔차에 대한 상대이고(scipy는 ‖b‖에 대한 상대, rtol 기본 10⁻⁵ 대 √ε), 반복 수를 돌려주지 않으며, GMRES의 기본은 DQGMRES다(scipy는 재시작 GMRES, `restart=20`). 한 걸음씩 부르는 `SparseIterate`로 판정을 직접 하면 판정은 맞출 수 있지만, 그 안의 직교화·Givens 회전·내적의 순서는 열 수 없다.
  - MINRES, BiCGSTAB, LSQR은 어차피 짜야 한다.
  - 반복법의 비용은 SpMV(이 라이브러리의 커널, §2.6)와 내적·axpy다. 같은 커널과 n으로만 나누는 내적을 쓰면 비트가 스레드 수와 무관하다.
- **구현**(`csrc/sparse/iterative.cpp`, `_cg _bicgstab _gmres _minres _lsqr _lsmr`; Python 층 `src/stardust/sparse/_iterative.py`, 결과 `_result.py`): scipy 1.18.1의 Python 판(`scipy/sparse/linalg/_isolve`의 `iterative.py`, `minres.py`, `lsqr.py`, `lsmr.py`)을 계산 순서까지 옮겼다. 스칼라는 numpy의 산술이다: 복소인 양은 복소로 두고(`ops::cmul`, `ops::cdiv`; 실수로 나누는 것도 numpy처럼 복소 나눗셈), 복소의 비교는 numpy의 순서(실수부, 다음 허수부)다. `minres`의 `gamma`는 numpy처럼 실수 노름을 그 dtype으로 바꾼 값이다. `gmres`의 Givens 회전은 scipy가 부르는 LAPACK `lartg`(3.10부터의 Anderson 2017 알고리즘, 실수·복소; Accelerate의 LAPACK 3.12도 같은 알고리즘이다)를 `iterative.cpp`에 옮겼다(linalg의 LAPACK 래퍼에 영역 밖의 함수를 더하지 않고, 회전 하나의 호출 비용을 피한다).
  - 연산자(SP80): 행렬은 CSR 배열 셋으로 C++에 넘어가고, 곱은 §2.6의 SpMV(줄 구간마다 태스크; `A @ x`와 비트가 같다)다. 밀집 행렬은 모든 원소를 저장한 CSR이고(밀집 `@`의 BLAS와 끝자리가 다를 수 있다), `lsqr`·`lsmr`의 Aᴴ는 Aᵀ의 CSR(복소는 켤레)이다. `aslinearoperator`의 연산자는 감싼 행렬을 바로 쓴다. 다른 `LinearOperator`는 곱마다 Python의 `matvec`·`rmatvec`을 새 배열로 부른다(결과는 계의 dtype, 연속으로 맞춘다).
  - 벡터 연산(SP131): 내적은 linalg의 `fastsum`(`linalg::fast::dot`·`cdot`, 첫 벡터의 켤레; numpy의 `dot`·`vdot` 자리: n으로만 정한 2¹⁴개의 조각마다 Accelerate `ddot`·`zdotc`, 짧으면 16레인 fma 잎, 조각끼리 균형 트리), 노름은 그 제곱합의 제곱근이고 제곱합이 넘치거나 작아 믿을 수 없으면 `sd.linalg.vector_norm`의 척도 조정 길이다(복소 벡터는 실수부·허수부를 한 실수 벡터로; numpy의 `norm` 자리). 갱신(`x += αp` 등)은 원소마다 실수 `fma`, 복소 `cmul` 뒤 덧셈이다. 모두 크기만으로 나누므로(2¹⁵부터 스레드풀) 같은 입력이면 스레드 수와 상관없이 같은 비트다(격자와 무작위 비대칭 n = 10⁵까지의 `cg`·`bicgstab`·`gmres`·`minres`·`lsqr`에서 1·3·10스레드의 반복 수와 x의 비트가 같다). numpy와 내적·노름의 반올림 순서가 달라 끝자리가 다르다.
  - GIL: 연산자가 모두 행렬이고 한 반복의 일(저장 수 + n)이 기준값 이상이면 풀이 전체가 GIL 없이 돈다. Python 연산자가 있으면 GIL을 쥔 채 돌고(곱마다 Python을 부른다) 커널은 스스로 GIL을 놓는다.
  - `history`: C++ 결과에 방법이 단계마다 쓴 잔차 노름의 열이 있고(cg·bicgstab: 반복 첫머리의 ‖r‖, gmres: 안쪽 반복마다의 `presid`, minres: φ̄, lsqr: r1norm, lsmr: normr), 공개 결과에는 넣지 않는다(테스트가 비공개 `_iterative._run`으로 scipy와 견준다).
- **scipy와 견준 것**(`tests/sparse/test_iterative.py`, `test_suitesparse.py`; scipy 1.18.1을 그 자리에서):
  - `tests/sparse/data/scipy.json`의 반복법 기록 40개(여섯 함수 × 행렬 셋–넷 × 기본·조인 허용오차): 상태는 40개 모두, 반복 수는 36개가 같다(`gmres`는 `callback_type='pr_norm'`의 호출 수 = `n_inner`). 나머지 넷은 `bicgstab`이다: convdiff2d/40 조인 허용오차는 반 단계의 ‖s‖ 판정에서 멈춰 `n_iter`가 그 반복을 세고 scipy의 `callback`은 세지 않는다(41 대 40). random_unsym/1000x8·random_complex/800의 셋(36 대 37, 64 대 61, 59 대 57)은 아래의 반올림에 민감한 문제다.
  - 단계마다의 잔차 노름(gmres는 scipy의 `pr_norm` × ‖b‖, cg·bicgstab·minres는 scipy 반복점의 참 잔차, lsqr·lsmr은 scipy를 k번째 반복에서 멈춘 r1norm·normr): 상대 10⁻⁶ 안에서 같다(대부분 10⁻¹⁵ 크기).
  - **scipy 자신이 반올림에 민감한 문제**: 무작위 비대칭 행렬(random_unsym/1000x8, random_complex/800)의 BiCGSTAB, 부정 행렬(random_indef/1500, saddle/600x150)의 MINRES 뒷부분, 반복이 456–506인 lap2d/30의 LSQR·LSMR. b를 상대 10⁻¹⁶ 바꾼 scipy끼리도 BiCGSTAB의 잔차가 열 걸음 안에 10⁻¹ 상대만큼 갈라지고 반복 수가 54–71(기록 57), lsqr의 마지막 normr가 4%까지 다르다. 그래서 거기서는 상태가 같고 반복 수와 normr가 scipy의 그 퍼짐 안(10% 넓혀)인지, 잔차 열은 scipy가 자기 자신과 10⁻⁹ 안에서 같은 앞부분에서만 견준다(sd와 scipy의 차이도 같은 꼴이다: random_unsym/1000x8 조인 허용오차에서 64 대 61).
  - 전처리(Jacobi), `x0`, `gmres`의 `restart`·`max_iter` 넷, `minres`의 `shift` 둘, `lsqr`·`lsmr`의 `damp`·`x0`(직사각·넓은·랭크 부족·복소): 상태와 반복 수가 같고 추정값(노름, 조건수)이 상대 10⁻⁶ 안이며 `lsqr`의 `var`가 scipy의 `calc_var=True`와 같다(10⁻⁶; 랭크 부족 행렬은 var가 잘 정해지지 않아 10⁻⁸에서 어긋났다). 붕괴(`bicgstab`의 −11), 정확한 해에서의 `gmres` 붕괴, 상한, b = 0도 scipy와 같다.
  - SuiteSparse(§7.2, external): memplus의 GMRES(20)는 안쪽 5,425번으로 scipy와 같고, cvxqp3의 MINRES는 같은 79번째 반복에서 멈추고, bcsstk14·msc10848의 Jacobi 전처리 CG는 370·5,272번(scipy 370·5,276; 5,000번 넘는 걸음에서 반복 수가 내적의 끝자리를 따라 몇 번 움직인다, §7.2)이다. young1c의 GMRES 2,809번(scipy 2,822), well1850의 LSQR 323번(324)·LSMR 338번(338). BiCGSTAB은 이 행렬들에서도 반올림에 민감하다: 상태는 같고 반복 수는 memplus 901(scipy 1,166), young1c 342(394), sherman3은 둘 다 ρ 붕괴(−10, 266번째 대 728번째).
- **참 잔차와 재귀 잔차**(§4): `cg`·`bicgstab`·`gmres`의 수렴한 결과는 다시 계산한 참 잔차가 판정의 1% 안이다(시험 문제, rtol 10⁻⁵·10⁻¹⁰). `minres`의 판정은 φ̄(잔차의 어림)에 대한 것이라 조건수가 큰 행렬에서 참 잔차와 크게 갈라진다: graded/100x8(κ 10¹⁰)은 rtol 10⁻¹⁰의 "a solution was found"로 멈추지만 참 상대 잔차가 7×10⁻²이고(graded/400x12 0.55, near_singular/20 1.4×10⁻²) scipy도 같다. MINRES의 잔차 간격이 ε·κ(A)·‖A‖‖x‖까지 자라는 알려진 성질이다(Sleijpen, van der Vorst, Modersitzki 2000). 정확도 스윕은 이것을 계약에 넣는다(§7.6).
- **속도**(표는 `bench/sparse/speed.md`): 1스레드는 격자에서 scipy와 같은 반복 수에 1.0–1.6배이고(n ≥ 9×10⁴은 메모리 대역에 묶여 1.0–1.15배), 10스레드는 큰 격자에서 `cg`·`lsmr` 약 4–5배, `gmres` 약 2–3배다(안쪽 반복마다 MGS의 내적과 갱신을 기저 벡터마다 훑는다). Accelerate의 반복법(`tools/sparse_probe.py iter`)은 두 스레드 모드에서 같은 시간이라 한 스레드로 돈다: CG는 sd 1스레드보다 약 1.2–1.3배 빠르고 sd 10스레드보다 2.4–3.7배 느리며, 재시작 GMRES는 sd 1스레드의 약 절반 시간이다(DQGMRES는 다른 알고리즘이라 반복이 적다). 멈춤 판정과 반복 수가 scipy와 달라 쓰지 않는 결정(SP16)은 그대로다.
- 결정(SP72–SP82, §1.1): 결과 객체와 필드, `max_iter`, `callback` 없음, `gmres`의 단위, 검사, 입력, dtype, 연산자, `minres`의 예외, `lsqr`의 `var`.

### 3.8 고유값과 특이값 (SP6)
- Accelerate에 ARPACK이 없다(§1.4). 이 라이브러리가 짠다(SP83, SP85).
- **ARPACK 옮기기**(`csrc/sparse/arpack.cpp`, `_eigsh`·`_eigs`; Python 층 `src/stardust/sparse/_eigen.py`, 결과 `_result.py`): scipy 1.18.1이 돌리는 ARPACK(arpack-ng를 옮긴 scipy의 C 판, `scipy/sparse/linalg/_eigen/arpack/arpack.py`의 `iterate`가 요청을 받는 순서)을 서브루틴 하나에 C++ 함수 하나로 옮겼다: `getv0`, `saitr`·`naitr`(DGKS 재직교화, 0.717 판정, 한 번까지), `seigt`·`neigh`, `sgets`·`ngets`(정렬은 ARPACK의 shell sort `dsortr`·`dsortc`·`zsortc`·`dsesrt`를 그대로), `sconv`·`nconv`, `sapps`·`napps`(정확한 이동의 암시적 QR: 대칭은 Givens, 실수 비대칭은 켤레 이동쌍의 Householder 이중 이동, 복소는 복소 Givens), `saup2`·`naup2`(재시작과 nev 조정; 실수 비대칭에는 scipy가 덧댄 `nev ≤ ncv − 2`), `seupd`·`neupd`(Ritz 벡터: 수렴한 Schur 벡터의 QR과 V에의 적용, 정화 단계). 역통신(`ido`)은 연산자의 직접 호출로 바꿨고, 요청마다 무엇을 곱하는지는 scipy의 `iterate`와 같다(`op1`: OP·x와 그것에 필요한 B·x, `opfull`: 시작 벡터의 OP·x, `bx`: B·x; 대칭 모드 2는 x 자리에 A·x를 남긴다).
  - 작은 밀집 문제는 ARPACK이 부르는 LAPACK(Accelerate)이다: `dsteqr`(ARPACK의 `dstqrb`는 `dsteqr`의 고유벡터 마지막 행만 갱신한 판이라, 전체 Z의 마지막 행과 같다), `dlahqr`, `dtrevc`, `dtrsen`, `dgeqr2`와 복소형, `dlarfg`·`dlarf`, `dlanhs`(`csrc/linalg/lapack.hpp`에 래퍼를 더했다). Accelerate의 `dlahqr`는 Z의 행 하나(`iloz = ihiz = 1`, `ldz = 1`)를 주면 그 행을 갱신하지 않아, 단위 행렬 전체를 주고 마지막 행을 읽는다(행마다 따로 갱신되므로 같은 값이다).
  - 큰 벡터의 일은 `csrc/sparse/krylov.hpp`의 것이다(SP5의 `iterative.cpp`와 함께 쓴다; SP131): Vᴴx(`krylov::dots`)와 x − Vh··V·Q(`krylov::gaxpy`)는 행 2¹⁴개씩의 고정 조각마다 Accelerate `dgemv`·`zgemv`(ARPACK의 `dgemv` 'T'·'N', `zgemv` 'C'·'N' 자리)를 한 번 부르고, Vᴴx는 조각의 부분합을 조각 순서로 더한다. 조각은 스레드풀의 태스크다. 내적과 노름은 §3.7의 `fastsum`, Ritz 벡터의 Householder 적용(`dorm2r`)과 `dtrmm`은 행마다다. 모두 n과 열 수로만 나누므로 스레드 수와 무관하게 같은 비트다. `dgemv`의 합은 scipy가 부르는 BLAS와 끝자리가 달라 OP 수가 scipy와 반올림 크기만큼 갈라질 수 있다(SP130).
  - **옮기며 확인한 scipy의 동작**(실험으로; 원본 소스는 내려받지 않았다): `getv0`는 `itry = 1`이면 B = I여도 시작 벡터에 OP를 한 번 곱한다(arpack-ng의 판; 첫 `matvec`의 입력이 v0 자체이고 둘째가 ‖Av0‖로 나눈 Av0이다). `tol = 0`은 2⁻⁵²다(`dlamch('E')` = 2⁻⁵³로 두면 lap1d 꼴에서 OP 수가 1,441 대 1,421로 갈라지고, 2⁻⁵²이면 같다). ε^(2/3)과 `dsapps`의 분리 판정은 2⁻⁵³의 것이다(2⁻⁵²로 바꾸면 lap1d/400 'LM' k = 3이 4,483 대 4,466으로 갈라진다). `dnaitr`는 걸음마다 betaj = rnorm으로 H(j, j−1)을 채운다. `dnapps`의 마지막 분리 판정은 |H(i+1, i)|로 한다(이중 이동의 Householder가 부부대각을 음수로 만들기 때문이다).
- **scipy와 견준 것**(`tests/sparse/test_eigen.py`, scipy를 그 자리에서, 같은 `v0`; OP 수는 `LinearOperator`의 호출 수):
  - 스무 문제(1·2차원 라플라시안, 무작위 대칭, 무작위 비대칭, 무작위 복소; 모든 `which`)에서 상태가 모두 같고 열셋에서 OP 수가 같다(SP130). 테스트는 OP 수가 scipy 자신의 퍼짐(v0의 한 원소를 한 ulp 바꾼 여섯 실행) 범위를 10% 넓힌 안인지 본다. 퍼짐 안에서 다른 것: 1차원 라플라시안 400의 'LM' k = 3 4,483(scipy 4,466–4,483), 2차원 라플라시안 20 × 20의 'SA' k = 3 349(333–365), 겹친 고유값 'LM' k = 3 313(313–331)·'BE' k = 6 339(349–351), 무작위 비대칭 'LI' k = 4 1,995(2,009–2,777), 무작위 복소 'LM'·'SR' 621(557–621)·1,311(1,311–1,411). 둔한 문제는 값이 순서까지 같고(상대 10⁻⁹), 민감한 문제는 scipy 실행 하나의 값과 순서와 상관없이 같다('LI' k = 4는 scipy의 여섯 실행 가운데 둘과 같은 고유값 쌍; 1·3·10스레드에서 같다).
  - 모드: M(모드 2), shift-invert(3), buckling(4), Cayley(5), 실수 비대칭의 복소 σ와 `OPpart` 'r'·'i', 복소 에르미트(`znaupd`)에서 값이 scipy와 같다(순서도 같고, 복소 에르미트 `eigsh`만 오름차순으로 정렬한다: SP127). `return_eigenvectors=False`의 실수 `eigs`에서 k가 켤레쌍을 가르면 고른 쪽이 scipy와 반대일 수 있다(같은 |λ|의 동점; 두 값 모두 맞다). 실수 A와 복소 σ의 `eigs(return_eigenvectors=False)`는 scipy에서 고유값이 모두 0이다(`OPpart` 'r'·'i'; 값이 Ritz 벡터의 Rayleigh 몫인데 벡터를 구하지 않는다). sd는 그때 안에서 벡터를 구해 값을 낸다(`[scipy와 다름]`).
  - SuiteSparse(§7.2, external): bcsstk14 'LM' k = 6의 OP 106번, msc10848 'LM' k = 4의 37번, memplus `eigs` 'LM' k = 6의 125번, young1c(복소) 372번이 scipy와 같다. bcsstk14·goodwin의 shift-invert(σ = 0)는 값이 같다(goodwin은 상대 10⁻⁶: shift-invert가 다듬지 않은 풀이라서, SP132·§7.2). well1850의 `svds`는 밀집 SVD와 상대 10⁻¹⁰ 안이다.
  - 기록(`tests/sparse/data/scipy.json`, v0 = ones): `eigsh` 여섯과 `eigs` 셋의 값이 상대 10⁻¹⁰·10⁻⁹ 안이다.
- **`svds`**(`csrc/sparse/svds.cpp`, `_svds`): X P = Q B, Xᴴ Q = P Bᵀ + f e_wᵀ의 Golub–Kahan–Lanczos 이중대각화에 두꺼운 재시작(Baglama–Reichel 2005)이다. m < n이면 Aᴴ를 이중대각화한다(v0는 min(m, n)개). B = U_B S V_Bᵀ(LAPACK `dgesdd`, B는 복소 A에서도 실수다)의 Ritz 삼중쌍이 ‖Xᴴu − σv‖ = ‖f‖·|U_B(w, i)| ≤ tol·max(σ, ε^(2/3)σ₁)이면 수렴한다(SP96). 재시작은 원하는 쪽 kk = k + min(수렴 수, (w − k)/2)개(1이면 w/2)를 남기고 B를 [diag(s) | ‖f‖ U_B(w, :)ᵀ]의 화살 꼴로 둔다. 새 벡터는 모든 벡터에 대해 고전 Gram–Schmidt, 첫 번에 ‖x‖가 0.717 아래로 줄면 두 번이다. 붕괴(불변 부분공간, 랭크 부족)에는 생성기의 무작위 방향을 직교화해 잇는다.
  - scipy의 `'arpack'`(AᴴA의 `eigsh`, tol²)과 견준 것: 정확도 스윕에서 rect/80x30d(랭크 부족)의 'SM'은 scipy가 계약의 2×10¹²배(작은 특이값이 AᴴA의 반올림에 묻힌다), sd가 0.009다. 'SM'은 Ritz 재시작이라 느리다: 1차원 라플라시안 200의 'SM'은 sd도 scipy도 기본 `ncv`로는 상한까지 수렴하지 못한다(harmonic Ritz 재시작은 나중 후보).
- **시작 벡터와 결정성**(SP84): `v0=None`이면 `sd.random.default_rng(0)`의 2u − 1이고 `rng=`로 바꾼다. 같은 입력이면 sd의 스레드 수(1·3·10), A의 형식(CSR·CSC·COO·DOK·밀집), `aslinearoperator`, `A @ v`의 Python `LinearOperator`와 상관없이 같은 비트다(테스트와 골든). shift-invert의 LU는 Accelerate의 스레드 모드가 같을 때 같은 비트다(§1.4).
- **결과**(SP87, SP90): §1.1의 SP87–SP92와 docs/api/sparse.md §7.
- **속도**(표는 `bench/sparse/speed.md`): 같은 OP 수에서 `eigsh`·`eigs`는 n = 10⁴에서 scipy와 비슷하거나 빠르다(Vᴴx·V·h가 시간의 대부분이었고 SP131로 약 2배). shift-invert(σ = 0)는 큰 행렬에서 Accelerate LU 덕에 scipy보다 빠르고, 작은 행렬에서는 다듬지 않는 풀이(SP132)로 scipy와 비슷해졌다(조건수 추정의 몫이 남는다). `svds`는 nnz 10⁵에서 scipy보다 빠르고 nnz 10⁶의 1스레드는 비슷하다(곱이 시간의 대부분).
- 결정(SP83–SP97, §1.1).

### 3.9 행렬 함수와 노름 (SP7)
- **구현**(SP110, `src/stardust/sparse/_matfuncs.py`): 네 함수 모두 Python이 이 라이브러리의 커널을 부른다(SpMV·SpMM(§2.6), SpGEMM, SP4의 LU와 다듬기·특이 판정, 축 리덕션). 그 커널이 모두 결정적이라 결과는 스레드 수와 상관없이 같은 비트다(`expm`은 입력을 먼저 CSR로 바꿔 형식과도 상관없다).
- **`onenormest(A, /, *, t=2, max_iter=5, compute_v=False, compute_w=False)`**: Higham–Tisseur(2000) Algorithm 2.4를 scipy 1.18.1의 `_onenormest_core` 순서 그대로 옮겼다. t ≥ n이면 명시한 행렬 A·I의 열 절댓값 합의 최댓값이다(scipy와 같다).
  - 무작위 ±1 열(처음 X의 둘째 열부터, 그리고 평행한 열을 다시 뽑을 때)은 호출마다 `sd.random.default_rng(0)`의 `integers(0, 2)·2 − 1`이다(SP98). scipy는 numpy의 시드 없는 전역 난수다.
  - h의 순서는 값의 내림차순, 같은 값은 색인 오름차순이다(`sd.argsort(descending=True)`의 안정 순서; 필요한 앞의 r ≤ 16개는 argmax를 r번 해서 같은 색인을 얻는다). scipy의 `argsort(h)[::-1]`은 안정 정렬이 아니라 같은 값끼리의 순서가 정해져 있지 않다.
  - 열의 평행은 원소마다 vᵢwᵢ = 1이다(실수면 같은 ±1 열; scipy의 `dot(v, w) == n`과 같은 판정). 복소 A의 부호 열은 Y/|Y|(0은 1)다.
  - 결과는 `OnenormestResult`(SP102, `src/stardust/sparse/_result.py`)이고 `(est, v, w)`로 풀린다. `n_iter`는 scipy의 k(Y = AX를 구한 횟수), `n_op`는 scipy의 nmults다.
  - 곱: 희소는 CSR의 SpMM과 켤레 전치의 CSR, 밀집은 matmul, `LinearOperator`는 그 `@`와 `.H @`(`rmatvec`이 없으면 그 `ValueError`, SP71).
  - 잰 것: 작은 정방 시험 행렬 42개에서 추정값이 참값과 같은 것이 26개, 가장 작은 비율 0.76(struct_singular/40)이다(scipy, `np.random.seed(0)`: 25개, 0.76). t = 1(scipy도 무작위가 없다)은 다섯 행렬에서 scipy와 상대 10⁻¹⁴ 안이다. scipy 기록(`matfuncs/<행렬>`, seed 0)의 넷 가운데 셋이 같은 값이다.
- **`expm(A, /)`**: Al-Mohy–Higham(2009) Algorithm 5.1을 밀집 `sd.linalg.expm`의 선택(SP103, linalg.md §11)으로 한다.
  - 차수 m ∈ {3, 5, 7, 9, 13}과 제곱 수 s의 판정은 linalg와 같은 식과 순서다: ‖A²‖은 언제나 정확, ‖A⁴‖·‖A⁶‖은 아는 노름의 곱으로 만든 상한이 시험을 지나지 못할 때만 거듭제곱을 만들어 정확히, ‖A⁸‖·‖A¹⁰‖은 n < 100이면 거듭제곱을 만들어 정확히, n ≥ 100이면 A⁴·A⁴, A⁴·A⁶의 곱 연산자에 `onenormest`(t = 2, 고정 열)로 어림한다(linalg는 같은 문턱에서 LAPACK `lacn2`). ℓ(A, m)은 |A|·2⁻ᵉ의 전치에 SpMV 2m + 1번으로 정확히 구한다.
  - 곱은 SpGEMM, 다항식의 합은 희소 덧셈(linalg의 `combine`과 같은 항의 순서), 결과는 I + 2(V − U)⁻¹U이고 풀이는 V − U의 SP4 LU(우변 U를 열 묶음으로, 다듬기 한 번, 특이면 `LinAlgError("expm(): singular matrix")`; 이 차수들에서는 일어나지 않는다)다.
  - 대각 행렬은 exp(대각) 그대로다. 위·아래 삼각은 Padé 뒤와 제곱마다 대각을 exp(2⁻ᵏaᵢᵢ), 첫 부대각을 2 × 2 블록의 정확한 식(고윳값이 가까우면 exp(평균)·sinch, 멀면 차분몫; linalg.md §11.3)으로 다시 쓰고 반대쪽 삼각을 지운다. 그 값은 C++의 `_expm_fix`가 linalg와 같은 식, libm `exp`·`sinh`, `ldexp`로 낸다(SP122; 대각 행렬과 2 × 2 삼각 행렬은 밀집과 비트까지 같다). 예: [[10, 1], [0, 10 + 10⁻¹²]]의 (0, 1)은 상대 9×10⁻¹⁶(scipy 2×10⁻⁵).
  - ‖A‖₁이 inf이면 2⁻ᵏ(2ᵏ ≥ 4n), 2¹⁰⁰을 넘으면 2^(100−e)로 먼저 줄이고 그만큼 더 제곱한다(linalg.md §11.4). 고윳값이 −10⁴⁰ 근처인 행렬은 0이다(scipy는 I).
  - 결과는 입력의 형식(SP99)이고 정확한 0은 뺀다(SP104). 대개 밀집에 가까워 n²의 메모리가 든다.
  - 잰 것(§4): Arb 대비 ε·max(1, ‖sA‖₁)·max|exp(sA)| 단위로 가장 큰 것 2.4(lap2d/10n, s = 1/8). 같은 점에서 scipy 4.8, R Matrix 4.2, 밀집 `sd.linalg.expm` 3.5.
- **`expm_multiply(A, B, /, *, start=None, stop=None, num=None, endpoint=None, traceA=None)`**: Al-Mohy–Higham(2011)의 잘린 Taylor를 scipy 1.18.1의 `_expm_multiply_simple`·`_expm_multiply_interval`의 순서 그대로 옮겼다: μ = tr(A)/n만큼 옮긴 A − μI의 정확한 1-노름(연산자는 `onenormest`), (m*, s)의 선택(Code Fragment 3.1, θ 표, m_max = 55, ℓ = 2; dₚ = `onenormest`((tA)ᵖ)^(1/p)), 단계마다 두 항이 허용오차 2⁻⁵³·‖F‖∞ 아래면 멈춤, 구간은 scipy의 세 경로(q ≤ s면 다시 고른 (m*, s)로 점마다, 아니면 d = ⌊q/s⌋개 점씩 항 K_p를 두고; scipy의 core_1과 core_2는 같은 값을 계산해 하나로 짰다).
  - 입력(SP105–SP107): A는 희소·2차원 밀집·`LinearOperator`, B는 밀집 (n,)·(n, k), `traceA`가 None이면 행렬은 정확한 대각합(희소 `trace()`, 밀집 `sd.linalg.trace`), 연산자는 scipy의 Hutch++ 추정을 고정된 ±1 열로 한다. 희소 A − μI는 행렬로 만들어 곱하고(scipy와 같다; CSR, 1스레드에서는 CSC 입력을 그대로 — §5), 연산자는 Ax + (−μ)x다.
  - scipy와의 비트: 곱(SpMV·SpMM이 scipy와 같은 순서), 스칼라 곱과 덧셈, 멈춤 판정의 ∞-노름이 같아 (m*, s)가 같으면 비트까지 같다. scipy의 dₚ는 무작위 추정이라 고르는 (m*, s)가 다를 수 있다: scipy 기록 넷 가운데 둘이 비트까지 같고(나머지 둘은 대각합의 합 순서로 μ의 끝자리가 다르다; 상대 3×10⁻¹⁶ 안), 실시간 대조 전체가 상대 10⁻¹³ 안이다. ‖A‖₁이 작아 (m*, s)를 ‖A‖₁만으로 고르는 경우(scipy의 조건 3.13)는 비트까지 같다(테스트).
  - 잰 것(§4): Arb(`expmv`, §7.3) 대비 ε·max(1, ‖tsA‖₁)·maxᵢ(|exp(tsA)||b|)ᵢ 단위로 t = 1이 2.0, 구간 2.5이고 scipy와 같다.
- **`norm(x, /, *, ord=None, axis=None)`**(SP14, SP101, SP108): 행렬 노름 `'fro'`(표준 꼴 값의 `sd.linalg.vector_norm`), 1·−1(열 절댓값 합 `sum(axis=0)`의 최댓값·최솟값), inf·−inf(행), 2(`svds(k=1)`의 값; 행이나 열이 하나면 그 벡터의 2-노름). 벡터 노름은 줄마다 저장된 값의 |x|ᵖ를 §2.6의 축 합으로 더하고, 암시적 0은 inf·−inf·0·음수 p에서 원소로 센다(음수 p는 0이 있는 줄이 0). p > 0의 합이 넘치거나 모든 항이 2⁻⁹⁶⁹ 아래면 그 줄만 최댓값으로 나눠 다시 더한다(vector_norm의 둘째 패스와 같은 판정). NaN은 결과가 NaN이다(`ord=0` 빼고; vector_norm과 같다). bool·int64는 float64로 바꾼다.
  - 잰 것(§4): 합의 노름은 계약의 0.17배(가장 긴 줄의 γ_m 단위), Frobenius 0.007배, 2-노름 0.013배(18 ulp; ARPACK을 AᵀA에 쓰는 scipy는 2 ulp). scipy의 2-노름은 empty/rows(0인 행이 있다)에서 `ArpackError`(시작 벡터 0)다.
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

- **SP4의 구현**: linalg의 모든 입력이 지나는 한 곳(`csrc/linalg/input.cpp`의 `Input`)이 실패한 호출에서만 받은 객체가 `stardust.sparse`의 클래스인지 보고 메시지 끝에 짚는 글을 붙인다. 짚는 함수가 아직 `sd.sparse`에 없으면(SP5–SP7의 `eigsh`, `eigs`, `svds`, `norm`, `lsqr`·`lsmr`, `expm`) 그 마일스톤까지는 "그 밖"의 글(`; a sparse array has no eigh: use eigh(A.toarray())`, 배열의 원소별 함수와 같은 꼴)이고, 함수가 생기면 저절로 그 함수를 짚는다. SP7부터 표의 함수가 모두 있어 `matrix_norm`·`vector_norm`은 `sd.sparse.norm`을, `expm`은 `sd.sparse.expm`을 짚는다(다른 영역의 골든은 그대로다: 그 거절을 기록한 케이스가 없다). 다른 입력의 메시지와 결과는 그대로다(linalg 골든 같음). 메시지 예: `inv(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.inv`.
- **배열의 함수**(SP67): `sd.matmul`, `sd.sum` 등도 희소 배열에 같은 꼴의 `TypeError`다: `sum(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use A.sum()`, `matmul(): ...; for a sparse array use A @ B`, 짝이 없으면 `; a sparse array has no sort: use sort(A.toarray())`. 배열의 입력 검사 `require_ndarray`(`csrc/bind/pyutil.cpp`)가 희소 배열에서만 `SparseOperand`(TypeError)를 내고, 함수 이름을 붙이는 `rethrow_named`(`csrc/bind/errors.cpp`)가 표 `sparse_hint`(같은 파일; SP36의 원소별 함수 표를 `csrc/array/ops/ufunc.cpp`에서 옮겨 넓혔다)의 글을 붙인다. 실패한 호출에서만 동작하고 다른 입력의 메시지와 결과는 그대로다(모든 영역의 골든 같음). `require_ndarray`를 쓰는 다른 영역(special, stats, htest)의 함수도 같은 글을 받는다(짝이 없으면 `A.toarray()`).

---

## 4. 정확도 계약 (안)
- **정확한 연산**: 형식 변환, 전치, 인덱싱, 만들기, 패턴 연산, 원소 하나의 산술(희소끼리의 ±, `*`, 스칼라 곱, 원소별 함수)은 같은 원소의 배열 연산과 비트까지 같다. 원소별 함수의 정확도는 배열의 것(array.md §2.7의 표)이다.
- **합과 곱의 합**(SpMV, SpMM, SpGEMM, 리덕션): §2.6의 순서로 계산한 값이 참값과 |ŷ − y| ≤ γ_m·Σ|aᵢⱼxⱼ|(m = min(n, L) + ⌈log₂(n/L)⌉, γ_m = mu/(1 − mu), u = 2⁻⁵³(단위 반올림 ε/2; `tests/sparse/refdata.py`의 `gamma`))다. Arb의 정확한 내적과 견준다(상쇄가 큰 줄, 긴 줄, 아주 큰·작은 값). 순서 자체가 정의이므로 같은 순서의 다른 구현(scipy의 SpMV)과는 비트까지 같다.
- **직접 풀이**: 정규화한 역오차 ‖b − Ax̂‖∞ / (‖A‖∞‖x̂‖∞ + ‖b‖∞)가 c·ε 이하이고 c = 4다(`tests/sparse/refdata.py`의 `C_BACKWARD`). 잰 것(SP133의 조건부 다듬기): 정방이고 특이가 아닌 시험 행렬에서 다듬은 풀이는 0.6ε 이하, 다듬지 않은 풀이(첫 풀이의 성분별 역오차가 8ε 이하)는 가장 큰 것이 3.54ε(lap3d/22, `trans='T'`)이다. 우변 여럿과 `trans='T'`·`'H'`는 네 시험 행렬과 SuiteSparse 다섯(§7.2)에서 테스트가 계약 안을 확인한다. 작은 행렬(n ≤ 2000, 참값 30점)은 Arb의 검증된 풀이와 견주어 앞오차가 C_SOLVE·κ∞(A)·ε(C_SOLVE = 4) 안이다(정확도 스윕 `solve`: 계약 단위 0.253, random_unsym/1000x8; 늘 다듬을 때는 0.097).
- **삼각 풀이**: 대입의 표준 역오차 |b − Tx̂| ≤ γ_n|T||x̂|(Higham 2002, 8장)를 원소마다 정확한 유리수 잔차로 확인한다(SP4, 실수·복소·긴 행).
- **반복법**: 결과는 "멈춤 판정을 만족한 점"이다(optimize §4와 같은 원리). `converged=True`면 반환 시점에 다시 계산한 참 잔차가 판정을 만족하는지 시험에서 확인한다(`cg`·`bicgstab`·`gmres`: 판정의 1% 안). 재귀 잔차와 참 잔차가 갈라지는 경우는 `minres`의 조건수가 큰 행렬이다(§3.7: 잔차 간격 ε·κ‖A‖‖x‖). Arb의 해와 견준 앞오차는 그 결과의 잔차가 허락하는 ‖A⁻¹‖∞·‖b − Ax̂‖∞(최소제곱: ‖(AᵀA)⁻¹‖∞·‖Aᵀr̂‖∞) 안이다(테스트, 잔차의 반올림 한계를 더한 위쪽 경계). 정확도 스윕의 계약은 판정이 약속하는 앞오차다(§7.6). 같은 문제에서 반복 수·상태를 scipy와 견준다(§3.7).
- **고유값·특이값**(SP95): 대칭·에르미트 문제의 고유값과 특이값은 Arb의 참값과 |λ̂ − λ| ≤ C_EIG·ε·‖A‖₂(C_EIG = 1000, `tests/sparse/refdata.py`), 비대칭 문제는 그것에 고유값의 조건수 κ(λ) = ‖y‖‖x‖/|yᴴx|를 곱한 것이다. 잔차 ‖Av − λMv‖, ‖Aᴴu − σv‖도 C_EIG·ε·‖A‖₂, 직교성 ‖VᴴMV − I‖는 C_EIG·k·ε 안이다(테스트). 잰 것: 대칭 점 17개 × 양 끝과 직사각 셋, 비대칭 셋(`tests/sparse/ref/eig.npz`)에서 가장 큰 것이 lap1d/1000 'LA'의 0.70(ARPACK이 수백 번 재시작하며 V·Q를 거듭 곱한 반올림이다; SP131의 `dgemv` 전에는 0.50, 같은 문제의 scipy는 0.22), `svds` 0.010, `eigs` 0.021(99 ulp)이다. 수렴 판정 자체는 ARPACK의 것이다: Ritz 값 θ의 잔차 어림이 tol·max(ε^(2/3), |θ|) 이하(tol = 0이면 ε).
- **`expm`·`expm_multiply`**(SP109): max|ŷ − y| ≤ C_EXPM·ε·max(1, ‖sA‖₁)·척도이고 C_EXPM = 10(`tests/sparse/refdata.py`)이다. 척도는 `expm`이 max|exp(sA)|, `expm_multiply`가 maxᵢ(|exp(tsA)||b|)ᵢ다(t의 구간은 t마다 ‖tsA‖₁). Arb(`arb_mat.exp`·`acb_mat.exp`, n ≤ 200, §7.3의 `expm`·`expmv`)와 견준다. SP7에서 잰 것: `expm` 2.4(scipy 4.8, R Matrix 4.2, 밀집 `sd.linalg.expm` 3.5), `expm_multiply` 2.0, 구간 2.5(scipy와 같다). 척도 조정과 제곱·Taylor의 반올림이 ‖A‖₁에 비례해 쌓이는 꼴이다(scipy와 linalg의 잰 값이 같은 꼴에 든다).
- **`onenormest`**: 추정값은 참값 이하이고(하한), 시험 행렬에서 참값과 같은 비율을 기록한다(계약이 아니다; SP7: 42개 중 26개가 같고 가장 작은 비율 0.76). **`norm`**(SP109): 합으로 된 노름(1, −1, inf, −inf, Frobenius)은 |v̂ − v| ≤ (γ_m + 2ε)·v(γ_m은 위의 mu/(1 − mu), m은 가장 긴 줄의 항 수, Frobenius는 저장된 원소 수; 2ε는 복소 원소의 크기와 마지막 반올림), `ord=2`는 `svds`의 계약 C_EIG·ε·‖A‖₂다. SP7에서 잰 것: 합 0.17, Frobenius 0.007, 2-노름 0.013(계약 단위; SP131 뒤 0.014).
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

**SP2에서 잰 것**(2026-11-21; 다른 에이전트가 같은 기기를 쓰는 중이라 기준선으로 올리지 않았다(SP34와 같다): `--smoke` 한 번 부른 값과, 같은 행렬에서 7번 중 가장 좋은 값을 따로 잰 것; nnz 10⁶, 행마다 10개, scipy는 1스레드; 배율은 scipy ÷ sd)

| 연산 | sd 1스레드 | sd 10스레드 | scipy | 1스레드 배율 | 10스레드 배율 |
|---|---:|---:|---:|---:|---:|
| `A @ x` | 0.71 | 0.15 | 0.63 | 0.89 | 4.4 |
| `A @ X` (k = 16) | 2.85 | 0.58 | 4.53 | 1.6 | 7.8 |
| `x @ A`, `A.T @ x` | 0.89 | 0.88 | 0.87 | 0.98 | 0.99 |
| `sum(axis=0)` | 0.80 | 0.81 | 0.89 | 1.1 | 1.1 |
| `A + B` | 5.7 | 1.5 | 6.5 | 1.1 | 4.4 |
| `A * B` | 5.3 | 1.3 | 6.5 | 1.2 | 4.9 |
| `A @ B` | 93 | 14 | 62 | 0.67 | 4.3 |

- `A @ B`의 1스레드는 행 정렬이 시간의 약 60%다. scipy의 결과는 행이 정렬되지 않은 꼴이고 `sort_indices()`까지 더하면 nnz 10⁵에서 20.6 ms로 이쪽(8.7 ms)이 2.4배 빠르다.
- 흩뿌리기(`x @ A`, CSC의 `@`, CSR의 열 합)는 1스레드 scipy와 같고 스레드로 빨라지지 않는다(§2.6의 병렬 분할). SP8의 고칠 목록에 넣는다. SpMM의 목표(1스레드 3배)도 아직이다(1.6배).
- `sum(axis=1)`, `max(axis=1)`, `A.sin()`, `A * 2.0`, `argmax(axis=0)`는 smoke 한 번에서 1스레드 2–40배다(scipy의 `argmax(axis=)`는 줄마다 Python 반복이다).

**SP3에서 잰 것**(2026-11-21; `--smoke` 한 번 부른 값이라 기준선으로 올리지 않았다(SP34와 같다); 다른 에이전트가 같은 기기를 쓰는 중이었다; nnz 10⁶, n = 10⁵, 행마다 10개, scipy는 1스레드; 배율은 scipy ÷ sd, ms)

| 연산 | sd 1스레드 | sd 10스레드 | scipy | 1스레드 배율 | 10스레드 배율 |
|---|---:|---:|---:|---:|---:|
| `A[:n/10]` | 0.08 | 0.08 | 0.25 | 3.3 | 3.3 |
| `A[무작위 행 n/10]` | 0.35 | 0.34 | 0.57 | 1.6 | 1.7 |
| `A[:, :n/10]` | 1.40 | 0.28 | 1.70 | 1.2 | 6.1 |
| `A[:, 무작위 열 n/10]` | 3.70 | 0.76 | 2.38 | 0.64 | 3.1 |
| `A[::-1, ::2]` | 6.49 | 1.19 | 6.23 | 0.96 | 5.2 |
| `diagonal()` | 0.28 | 0.37 | 0.32 | 1.1 | 0.9 |
| `vstack([A, B])` | 1.72 | 1.87 | 0.98 | 0.57 | 0.52 |
| `hstack([A, B])` | 2.74 | 1.87 | 4.09 | 1.5 | 2.2 |
| `tril(A)` | 2.93 | 1.03 | 4.37 | 1.5 | 4.3 |
| `kron(eye_array(10), A)` | 9.13 | 2.22 | 59.2 | 6.5 | 27 |
| `random_array` (밀도 10/n) | 80.6 | 67.1 | 17.0 | 0.21 | 0.25 |

- `vstack`은 값과 색인을 옮기는 것뿐이라 메모리 대역에 묶인다: int64 색인(SP19)이라 옮기는 바이트가 scipy(int32)의 4/3배이고 새 배열의 첫 쓰기가 대부분이다. 10스레드로도 빨라지지 않았다.
- 정렬되지 않은 열 고르기는 줄마다 결과를 자리로 정렬하는 몫이 있다(1스레드 0.64배). `random_array`의 시간 대부분은 `sd.random`의 `choice(10¹⁰, 10⁶, replace=False)`(약 70 ms; numpy는 약 15 ms)다. 둘 다 SP8의 고칠 목록에 넣는다.
- 원소 하나 읽기 `A[i, j]`는 호출 하나에 약 1.4 µs(scipy 4.2 µs)다(키 읽기가 대부분).

**SP4에서 잰 것**(2026-11-21; `--smoke` 한 번 부른 값이라 기준선으로 올리지 않았다(SP34와 같다); 다른 에이전트가 같은 기기를 쓰는 중이었다; Dirichlet 라플라시안, sd는 Accelerate 기본 스레드 모드, scipy는 1스레드; 배율은 scipy ÷ sd, ms)

| 연산 | 2차원 n = 10⁴ | 9×10⁴ | 10⁶ | 3차원 n = 1.1×10⁴ | 9.1×10⁴ |
|---|---:|---:|---:|---:|---:|
| `spsolve` sd (1·10스레드) | 11.3 · 10.8 | 89.6 · 83.8 | 1,149 · 1,146 | 39.9 · 38.1 | 1,055 · 1,068 |
| `spsolve` scipy | 11.2 | 153 | 3,772 | 151 | 18,191 |
| 배율 | 0.99 | 1.7–1.8 | 3.3 | 3.8–4.0 | 17 |
| `splu` sd | 10.2 | 83.6 | 1,173 | 37.9 | 1,039 |
| `splu` scipy | 11.3 | 154 | 3,752 | 148 | 17,913 |
| `SparseLU.solve` sd (다듬기 포함) | 1.09 | 4.33 | 30.6 | 1.78 | 29.0 |
| `SuperLU.solve` scipy | 0.41 | 5.51 | 79.4 | 2.04 | 68.3 |
| `spsolve_triangular` sd (`tril`) | 0.10 | 0.78 | 9.0 | 0.10 | 0.76 |
| `spsolve_triangular` scipy | 0.72 | 4.09 | 47.0 | 0.77 | 5.12 |

- `spsolve`·`splu`는 n ≥ 9×10⁴에서 1.7–17배다(목표 2배: 2차원 n = 9×10⁴의 1.7–1.8배만 못 미친다). n = 10⁴에서는 scipy와 같다(조건수 추정과 다듬기의 몫이 분해의 10–15%). 스레드 수는 거의 상관없다(Accelerate의 LU는 sd의 스레드풀을 쓰지 않는다).
- `SparseLU.solve`는 다듬기(풀이 둘과 SpMV 하나) 때문에 작은 행렬에서 scipy보다 느리다(n = 10⁴에서 0.38배; 큰 행렬은 Accelerate의 풀이가 빨라 1.3–2.6배). 다듬기를 빼는 옵션은 두지 않았다(SP63). SP133부터 잘 조건된 행렬은 다듬지 않아 이 차이가 거의 없어졌다(아래 2026-11-22의 점검).
- `spsolve_triangular`는 5–7배다(scipy는 A·diag(1/d)를 만들고 SuperLU `gstrs`를 거친다). 우변 하나는 한 스레드라 10스레드도 같다(단계 스케줄링은 하지 않는다: 나중 후보).
- R의 Matrix는 행렬 객체에 분해를 저장해 두므로(`A@factors`) 같은 행렬을 다시 부르면 찾기만 한다(0.01–0.4 ms). 그래서 직접 풀이 사례에는 R 비교가 없다(처음 분해는 lap2d/1000·lap3d/45에서 약 11분, 6.5 GB였다). R의 값은 위 SP0 표다.

**SP7에서 잰 것**(2026-11-22; `--smoke` 한 번 부른 값이라 기준선으로 올리지 않았다(SP34와 같다); 다른 에이전트가 같은 기기를 쓰는 중이었다; ms, 배율은 scipy ÷ sd, scipy는 1스레드)

| 연산 | 크기 | sd 1스레드 | sd 10스레드 | scipy | R Matrix | 1스레드 배율 | 10스레드 배율 |
|---|---|---:|---:|---:|---:|---:|---:|
| `norm(A, ord=1)` | nnz 10⁶ | 1.45 | 1.65 | 1.29 | 1.34 | 0.89 | 0.78 |
| `norm(A, ord=1)` | nnz 10⁷ | 18.0 | 17.3 | 12.5 | 12.6 | 0.70 | 0.72 |
| `norm(A, ord=inf)` | nnz 10⁷ | 10.9 | 9.15 | 11.0 | 13.4 | 1.00 | 1.20 |
| `norm(A)` (Frobenius) | nnz 10⁷ | 1.12 | 0.64 | 0.70 | 770 | 0.63 | 1.10 |
| `norm(A, axis=1)` | nnz 10⁷ | 22.2 | 11.1 | 18.7 | — | 0.84 | 1.69 |
| `norm(A, ord=2)` | nnz 10⁶ | 309 | 94 | 203 | — | 0.66 | 2.17 |
| `onenormest(A)` | nnz 10⁶ | 11.9 | 3.79 | 16.3 | — | 1.37 | 4.31 |
| `onenormest(A)` | nnz 10⁷ | 156 | 51.9 | 191 | — | 1.22 | 3.67 |
| `expm_multiply(−0.1 A, b)` | 2차원 n = 9×10⁴ | 5.62 | 2.97 | 8.47 | — | 1.51 | 2.85 |
| `expm_multiply(−0.1 A, b)` | 2차원 n = 10⁶ | 69.6 | 34.5 | 92.9 | — | 1.34 | 2.70 |
| `expm_multiply(−0.1 A, b)` | 3차원 n = 9.1×10⁴ | 7.85 | 3.12 | 10.4 | — | 1.32 | 3.32 |
| `expm_multiply`, 0..1의 11점 | 2차원 n = 9×10⁴ | 16.4 | 7.14 | 46.2 | — | 2.81 | 6.47 |
| `expm(0.1 A)` | 2차원 n = 900 | 100 | 98 | 187 | — | 1.88 | 1.91 |
| `expm(0.1 A)` | 3차원 n = 1000 | 245 | 211 | 367 | — | 1.50 | 1.74 |

- 작은 크기(nnz 10⁴)는 모든 노름이 scipy의 2–4배다(Python 층이 얇다). nnz 10⁵의 10스레드가 1스레드보다 느린 것은 smoke 한 번의 첫 호출(스레드풀을 깨우는 몫)로 보인다. `norm(A, ord=1)`과 Frobenius의 큰 크기는 0.6–0.9배다: CSR의 열 합은 흩뿌리기(SP2의 `x @ A`와 같은 1스레드 커널)이고, Frobenius는 전체 값의 `vector_norm`(두 번 읽는다)이다. SP8의 고칠 목록에 넣는다. `norm(A, ord=2)`는 `svds`(GKL, tol = 0)라 scipy(AᵀA의 ARPACK, k = 1)보다 1스레드에서 느리다.
- `expm_multiply`는 CSC 입력을 1스레드에서는 그대로 곱하고(흩뿌리기가 CSR의 SpMV와 같은 빠르기다) 여러 스레드에서는 CSR로 바꿔 곱한다. 어느 형식의 곱이든 같은 비트라(§2.6) 결과는 스레드 수와 상관없이 같다. 처음에 언제나 CSR로 바꿨을 때는 n = 10⁶의 1스레드가 122 ms(0.75배)였다. 구간(11점)은 scipy가 점마다 Python 반복으로 항을 더해 2.8–3.4배, 10스레드 6.5–7.6배다. `expm`은 SpGEMM과 LU의 우변 열 묶음이라 1.5–1.9배다.
- R Matrix의 `norm(A, "F")`은 nnz 10⁷에서 770 ms다(밀집으로 바꾸는 것으로 보인다).

**SP8에서 잰 것**(2026-11-22; 다른 에이전트가 같은 기기를 쓰는 중이었다. 전체 표는 `bench/sparse/speed.md`(238사례를 한 라운드, `build/speed/sp8-sparse.jsonl`에서 만든 표; 기준선으로 올리지 않았다(SP34와 같다); † 줄은 다른 프로세스가 1코어 넘게 쓴 사례). 고친 커널은 구판(cda05f3)과 새판을 프로세스로 번갈아 부른 A/B(라운드 3–5의 중앙값)로 쟀다. 결정은 SP116·SP117)

| 연산 | 크기 | 구판 ÷ 새판 1스레드 | 10스레드 | 새판의 scipy 대비 1스레드 |
|---|---|---:|---:|---:|
| COO → CSR | nnz 10⁵ / 10⁶ / 10⁷ | 1.59 / 1.45 / 1.37 | 1.11 / 1.23 / 1.21 | 2.5 / 2.0 / 2.6 |
| CSR → CSC | nnz 10⁵ / 10⁶ / 10⁷ | 1.17 / 1.41 / 1.47 | 1.0 | 1.27 / 1.49 / 3.3 |
| `x @ A`, CSC `@`, 열 합, `norm(A, 1)`, `A.T @ x` | nnz 10⁶–10⁷ | 1.0(`A.T @ x` 1.5) | 1.7–2.5 | 0.94–1.2 |
| `X.mT @ A`(k = 16) | nnz 10⁵ / 10⁶ | 2.95 / 2.94 | 6.0 / 7.1 | 0.84 / 0.89 |
| `A @ X`(k = 16) | nnz 10⁵ / 10⁶ / 10⁷ | 1.28 / 1.30 / 1.95 | 1.14 / 1.11 / 1.56 | 1.9 / 2.0 / 3.2 |
| `A @ B` | nnz 10⁴ / 10⁵ / 10⁶ | 1.69 / 1.32 / 1.27 | 1.53 / 0.98 / 1.12 | 0.90 / 0.68 / 0.75 |
| `A[:, 무작위 열 n/10]` | nnz 10⁵ / 10⁶ | 1.41 / 1.39 | 1.3 / 1.13 | 0.57 / 0.80 |
| `eigsh(k=6)` | lap2d/100, lap3d/22 | 1.09 / 1.08 | 1.67 / 1.92 | 0.53 / 0.53 |
| `eigs(k=6)` | lap2d/100 | 1.09 | 1.14 | 0.49 |
| `svds(k=6)` | nnz 10⁵ / 10⁶ | 1.07 / 1.0 | 1.93 / 1.11 | 0.79 / 0.76 |
| `gmres` rtol 10⁻⁸ | lap3d/45 | 1.04 | 1.28 | 1.08 |
| `norm(A, 2)` | nnz 10⁶ | 1.0 | 1.44 | 0.67 |
| `SparseLU.solve` | lap2d/100, lap3d/22 | 1.0 | 1.0 | 0.40 / 0.81 |

- 표의 묶음별 중앙값(scipy ÷ sd, 1스레드 / 10스레드): 형식 변환 2.98 / 5.73, 만들기 1.00 / 1.84, 곱 0.99 / 1.81, 산술 1.31 / 3.51, 리덕션 3.04 / 4.19, 원소별 함수 3.89 / 23.5, 인덱싱 1.36 / 2.35, 고유값 0.65 / 1.21, 노름 1.49 / 2.37, 직접 풀이 2.13 / 3.21, 반복법 1.23 / 4.09, 행렬 함수 1.57 / 3.19(238사례 가운데 63사례가 바쁠 때 잰 † 줄).
- SP33의 목표(1스레드 2배): COO → CSR은 닿았고(10⁵–10⁶에서 2.0–2.5배), CSR → CSC는 1.3–1.5배에 머문다. 무작위 열로 흩어 쓰는 일이 두 번의 차례 쓰기가 되었지만 int64 색인이라 옮기는 바이트가 scipy의 4/3배다.
- 흩뿌리기의 10스레드는 1.7–2.5배다(목표 4배): 블록으로 나눈 항(출력과 두 인수, 24바이트)을 한 번 쓰고 읽는 몫이 메모리 대역에 묶인다. 항이 많은 `X.mT @ A`는 6–7배다.
- SpMM은 1스레드 2.0배(10⁶; 목표 3배)다. x 행을 무작위로 모으는 지연이 남는다.
- SpGEMM은 정렬하지 않는 scipy의 0.68–0.9배이고, scipy에 `sort_indices()`까지 더하면 이쪽이 빠르다(SP2의 2.4배보다 더).
- n = 10⁴의 Lanczos·Arnoldi는 여전히 scipy의 0.5배 근처다: 기저의 내적·갱신이 메모리 대역(약 70–100 GB/s)에 닿았고 scipy가 부르는 Accelerate `dgemv`(AMX)는 같은 일을 7배 빨리 한다(V 10⁴ × 20의 Vᵀx 3.2 µs 대 22.7 µs). 결정적인 합을 BLAS로 바꾸면 비트가 바뀌어 하지 않았다(SP117). 10스레드에서는 작은 n도 나뉘어 1.7–1.9배 빨라졌다. 2026-11-22에 n으로 정한 조각마다의 `dgemv`로 바꿨다(SP131).
- `SparseLU.solve`는 시간의 97%가 Accelerate 풀이 두 번(다듬기)이다(lap2d/100: 풀이 하나 0.39 ms, 다듬기의 SpMV와 뺄셈 0.03 ms). SP63을 지키므로 그대로였다. 2026-11-22에 다듬기를 조건부로 바꿨다(SP133).
- 표의 굵은 줄 가운데 SP8에서 고치지 않은 것(만들기와 고유값은 2026-11-22에 고쳤다): `(data, indices, indptr)`·`(data, (row, col))`의 만들기(색인을 복사하고 검사한다, SP30), `random_array`(sd.random의 비복원 `choice`, BN-13), `A @ x`의 1스레드 0.82–0.97배(int64 색인, 목표의 0.8배 안), `vstack`의 1스레드 0.81배(바이트 몫; SP3 표의 0.57배는 smoke 한 번의 첫 호출이었다), 노름의 작은 크기(Python 층)와 Frobenius(`vector_norm`), 고유값(위), `lsqr`·`lsmr`의 큰 격자(메모리 대역, 0.98–0.99배).

**2026-11-22의 성능 점검**(SP130–SP136; 표는 `bench/sparse/speed.md`, 숫자는 옮기지 않는다): 같은 기기를 다른 일과 나눠 쓴 실행을 그 전의 전체 실행과 견줬다(1스레드, 대략의 배율).
- 만들기: `csr_array((data, indices, indptr))`·`coo_array((data, (row, col)))`가 약 3배(작은 입력의 고정 비용은 약 1/4), `diags_array`가 3–12배, `tril`·`triu`가 2.6–4배, `kronsum`이 3–4배, `is_sptriangular`(삼각이 아닌 행렬)는 상수 시간이다. 남은 차이: 생성자는 색인 복사(SP135, §9), `diags_array`는 scipy가 DIA를 돌려주는 몫이다(§9).
- 곱과 연산: `X @ A`(k = 16)가 nnz 10⁷에서 약 4배·10⁴–10⁶에서 1.2–1.4배, CSR의 SpMV가 1.2–1.8배(네 줄의 사슬), `A + D`가 1.35–1.45배, 정렬되지 않은 열 고르기가 1.4배, 원소 쌍 고르기가 2–3배, SpGEMM의 1스레드가 약 1.2배다. 흩뿌리기(`x @ A`, `A.T @ x`, 열 합)는 그대로다(미리 읽기는 큰 출력에서만).
- 풀이: `splu().solve`가 lap2d/100·300, lap3d/22에서 1.6–1.9배(SP133), `inv`가 3–4배(SP133, SP134), 우변 16개의 `spsolve`가 1.2배, `spsolve_triangular`가 1.2–1.3배다. 큰 격자(lap2d/1000, lap3d/45)의 풀이와 분해는 그대로다.
- 고유값: `eigsh`·`eigs`가 1.8–2.2배(SP131), σ = 0이 1.5–1.6배(SP132), `svds`가 1.2–1.6배, `norm(A, 2)`가 1.2–1.8배다.
- 버린 시도는 SP136의 끝에 있다.

**목표** (같은 기기, 1·10스레드, scipy 대비 배율)
- SpMV, `A.T @ x`, 축 리덕션: 1스레드에서 scipy와 같거나 빠르게(int64 색인의 대역 비용이 있어 0.8배까지를 받아들이고 넘으면 §2.2의 int32 안을 묻는다), 10스레드에서 nnz ≥ 10⁶이면 4배 이상(메모리 대역의 한계, array.md §10.6의 `f + g` 1.7배보다 SpMV는 무작위 읽기라 더 나뉜다).
- SpMM: scipy가 k 방향으로 벡터화하지 않아 느리다(10⁷에서 SpMV의 29배). 1스레드에서 3배 이상.
- 변환(COO → CSR, CSR → CSC, 중복 더하기): 1스레드에서 2배, 10스레드에서 5배 이상.
- SpGEMM, 이항 연산: 1스레드에서 같거나 빠르게, 10스레드에서 4배 이상.
- 직접 풀이: Accelerate LU/AMD는 2차원 격자에서 SuperLU보다 1.2–3.4배, 3차원에서 4–18배 빠르다(위 표). `spsolve`·`splu`는 n ≥ 10⁵에서 2배 이상을 목표로 하고 행렬 꼴(2·3차원 격자, 회로, 무작위)마다 잰다(SP4의 smoke: 위 표).
- 반복법: 반복 하나의 비용이 SpMV + 내적이다. scipy의 Python 반복 대비 같은 반복 수에서 1스레드 1.5배, 10스레드 4배 이상. SP5에서 잰 것은 §3.7이다(1스레드 1.03–1.6배, 10스레드 cg·lsmr 3.7–5.3배·gmres 2.3–2.8배).
- `eigsh`·`svds`: ARPACK(Fortran)과 같은 SpMV 수에서 같거나 빠르게, 10스레드에서 SpMV 몫만큼 빠르게. 지금의 값은 §3.8이다(n = 10⁴의 Lanczos·Arnoldi는 SP131로 scipy와 비슷하거나 빠르다).
- `expm`·`expm_multiply`·`onenormest`·`norm`: 같은 곱의 수에서 scipy와 같거나 빠르게, 10스레드에서 곱의 몫만큼 빠르게. SP7에서 잰 것은 위 표다(1스레드 0.63–4배, 10스레드 0.72–7.6배; 못 미친 곳은 SP8).
- 작은 행렬(nnz ≤ 10³): 호출 하나의 고정 비용이 scipy(Python 층, 수 µs–수십 µs)보다 작게.
- 크기: nnz 10⁴, 10⁵, 10⁶, 10⁷(SpGEMM은 10⁶까지), 행 길이 3·10·100과 긴 행이 섞인 행렬, 풀이는 2·3차원 격자 n = 10⁴–10⁶. 사례는 `bench/speed/cases/sparse.py`에 마일스톤마다의 계획으로 있고, 그 마일스톤의 이름이 공개되면 사례가 된다(§7).

---

## 6. 층과 파일
- **층**: `sparse/`는 `array/`, `linalg/`(LAPACK 래퍼, 비용 규칙, pairwise 잎, `fastsum.hpp`의 내적·노름), `parallel/`, `accelerate/`(Sparse Solvers, `blas.hpp`의 CBLAS), `base/` 위에 있다(ARCHITECTURE §1의 "아래로만"). 다른 영역은 `sparse/`를 모른다. 예외는 §3.10의 판별 하나(`bind/pyutil`의 `is_sparse` 꼴)와 나중의 그래프 영역이다.
- `src/stardust/sparse/` (하위 패키지 없음, SP3): `__init__.py`(공개 이름, `issparse`, `matrix_transpose`), `_base.py`(`sparray`: 공통 메서드, SP2의 연산자·원소별 메서드·리덕션의 입구), `_ops.py`(SP2: 연산자, 원소별 함수, 곱의 규칙), `_reduce.py`(SP2: 리덕션), `_coo.py`, `_compressed.py`(`csr_array`, `csc_array`), `_dok.py`, `_construct.py`(만들기 함수), `_index.py`(인덱싱과 대입), `_interop.py`(생성자의 입력과 `to_scipy`), `_interface.py`(SP4: `LinearOperator`, `aslinearoperator`), `_direct.py`(SP4: `spsolve splu factorized inv spsolve_triangular is_sptriangular spbandwidth`, `SparseLU`), `_iterative.py`(SP5: `cg gmres minres bicgstab lsqr lsmr`, 입력의 검사와 연산자), `_eigen.py`(SP6: `eigsh eigs svds`, 시작 벡터, 스펙트럼 변환의 역), `_matfuncs.py`(SP7: `expm expm_multiply onenormest norm`; 이 라이브러리의 커널을 부르는 Python, SP110), `_result.py`(SP5: 반복법의 결과 클래스 `CgResult` … `LsmrResult`; SP6: `EigshResult`, `EigsResult`, `SvdsResult`), `_common.py`(입력 정리와 메시지).
- `csrc/sparse/`(SP0: `direct.hpp`·`direct.cpp`, 비공개 `_lu_solve`로 래퍼를 시험한다; SP1: `format.hpp`, `parallel.hpp`, `canonical.cpp`, `convert.cpp`, `dok.cpp`; 나머지는 마일스톤마다): `format.hpp`(§2.2의 표현, 커널이 함께 쓰는 것), `canonical.cpp`(표준 꼴: 정렬, 중복 더하기, 전치, 입력의 검사; §3.1), `convert.cpp`(밀집 사이, 밀집에 제자리로 더하기 `_dense_update`, 펼치기, 거르기, 세기, `reshape`의 좌표), `arith.hpp`(SP2: 항 하나와 긴 줄의 조각 합, 네 줄을 함께 도는 `line_dots`, §2.6), `spmv.cpp`(SP2: SpMV·SpMM·Aᵀx·밀집 @ 희소, 축 합과 평균, §2.6의 순서), `spgemm.cpp`(SP2), `binop.cpp`(SP2: 병합 이항 연산, 비교, 밀집 값 읽기), `reduce.cpp`(SP2: max·min·argmax·argmin; 원소별 함수는 커널이 따로 없다: Python 층이 배열의 함수를 `data`에 한 번 부른다), `index.cpp`, `construct.cpp`(만들기: `_kron`, `_kronsum`, `_diags`, 쌓기, `_band_coo`·`_band_compressed`), `dok.cpp`(해시 맵), `triangular.cpp`(SP4: 대입, `_bandwidth`, `_triangular`), `direct.cpp`(SP4: `_lu_factor`·`_LuFactor`, Accelerate LU, 특이의 판정과 조건수 추정, 다듬기의 판정 `_residual`), `iterative.cpp`(SP5: 여섯 풀이), `krylov.hpp`(SP6에 `iterative.cpp`에서 꺼냄: numpy의 스칼라 산술, 연산자(CSR·Python·항등), `fastsum`의 내적·노름, 갱신, 조각마다의 `dgemv`인 기저의 곱(SP131), `lartg`), `arpack.cpp`(SP6: ARPACK의 `dsaupd`·`dseupd`·`dnaupd`·`dneupd`·`znaupd`·`zneupd`), `svds.cpp`(SP6: 두꺼운 재시작의 Golub–Kahan–Lanczos), `parallel.hpp`(크기로만 정하는 분할, 기준값 2^15). SP7은 C++ 파일이 없다(SP110).
- `csrc/accelerate/sparse.hpp`(SP0): Sparse Solvers 래퍼(ARCHITECTURE §4.4: Accelerate 래퍼는 `accelerate/` 한곳). int64 → int32 행 색인 복사(`Csc::narrow`), 스레드마다 모으는 `reportError` 콜백(`ErrorScope`; status가 0이어도 메시지가 있으면 실패), status → `Error{Singular, Parameter, Failed, Internal}`, LU/AMD·`zeroTolerance = 0` 고정, `SparseCleanup`의 RAII(`Lu<T>`, float64·complex128).
- `csrc/linalg/lapack.hpp`에 SP6이 더한 래퍼: `steqr`, `lahqr`, `trevc`, `trsen`, `geqr2`, `lanhs1`, `larf`(ARPACK이 부르는 것; 이중대각 B의 SVD는 기존의 `gesdd`).
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
- **받은 곳**: 컬렉션의 공식 파일 서버(`sparse-files.engr.tamu.edu`, S3; `https://suitesparse-collection-website.herokuapp.com/MM/...`와 `https://sparse.tamu.edu/MM/...`가 모두 그리로 간다)는 2026-10부터 모든 파일에 `403 Forbidden`(AccessDenied)을 돌려준다. 그래서 2026-11-21에 사용자의 승인을 받아 컬렉션의 원래 자리인 University of Florida의 사본(`https://www.cise.ufl.edu/research/sparse/MM/<group>/<name>.tar.gz`, 여덟 개 합쳐 10.4 MB)에서 받았다. `tools/sparse_suitesparse.py`의 `URLS`는 공식 주소를 먼저, 그 사본을 다음에 두고, `SHA256`에 받은 압축 파일 여덟의 sha256이 있다. 저장 원소 수는 컬렉션의 목록과 맞는다(파일에는 저장된 0이 남아 있다: memplus는 저장 126,150개 대 0이 아닌 것 99,147개). SP20의 다른 조건(버전 관리 밖, external 등급, 없으면 건너뜀)은 그대로다.
- **SP4에서 쓴 것**(`tests/sparse/test_suitesparse.py`, external; `--tier all`에서 22개 통과, 최대 RSS 1.97 GB): 정방 일곱 가운데 다섯(bcsstk14, msc10848, goodwin, memplus, young1c(복소))은 `splu`의 `trans='N'·'T'·'H'` 두 우변 풀이가 역오차 계약(§4) 안이고 κ₁ 추정이 1/ε 아래다(1.3×10¹⁰, 3.3×10¹⁰, 3.3×10¹⁰, 2.7×10⁵, 5.2×10²; 역오차 9×10⁻⁸–0.28ε, scipy SuperLU 1.3×10⁻⁷–0.74ε). sherman3(κ₁ 6.9×10¹⁶)과 cvxqp3(2.2×10¹⁶; 둘 다 scipy의 LU로 잰 `onenormest`도 같다)는 κ₁·ε ≥ 1이라 SP62대로 `LinAlgError`다. 분해 시간(`splu`, 판정 포함, 한 번 잰 값; Accelerate 기본 스레드 모드, scipy 1스레드): msc10848 70 ms(scipy 302), goodwin 63 ms(274), memplus 23 ms(307), cvxqp3 2.0초(13.5초; 판정에서 멈춘다).
- **SP5에서 쓴 것**(같은 파일, external, scipy를 그 자리에서): 열 개의 반복법 문제(memplus의 GMRES·BiCGSTAB, young1c의 GMRES·BiCGSTAB, sherman3의 BiCGSTAB, cvxqp3의 MINRES, bcsstk14·msc10848의 Jacobi 전처리 CG, well1850의 LSQR·LSMR; b = 1). 결과는 §3.7이다. 열 개가 7.6초에 돈다. msc10848의 Jacobi CG는 반복 수를 scipy와 8번 안으로 본다(5,272 대 5,276; 5,000번 넘는 걸음에서 반복 수가 내적의 끝자리를 따라 움직인다, SP131; 전에는 1번 안).
- **SP6에서 쓴 것**: 고유값 문제(§3.8). goodwin의 shift-invert(σ = 0) `eigs`는 scipy와 상대 10⁻⁶ 안으로 본다(shift-invert의 풀이를 다듬지 않아서, SP132; 전에는 10⁻⁹).

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
| `eig` | 23 | 대칭·에르미트 행렬의 가장 작은·큰 고유값 6개씩(n ≤ 220은 Arb `eig`, Dirichlet 격자는 닫힌 꼴), 직사각 행렬의 특이값, 비대칭 실수·복소 행렬 셋(convdiff2d/10, random_unsym/50x5, random_complex/40)의 모든 고유값(`eigvals`, 복소; SP6) |
| `expm` | 38 | `arb_mat.exp`·`acb_mat.exp`(n ≤ 200), 1-노름이 1과 30 근처가 되는 2의 거듭제곱 척도 둘 |
| `expmv` | 38 | `expm`의 점마다 b = `matrices.rhs(a)`의 exp(sA)b(`expmv`)와 t = 0, 1/4, 1/2, 3/4, 1의 exp(tsA)b(`interval`); Arb의 `exp`와 행렬·벡터 곱, maxᵢ(\|exp(tsA)\|\|b\|)ᵢ의 2⁻¹¹⁰까지(SP7; 3프로세스 5초, 작업 프로세스의 최대 RSS 0.13 GB) |

- 빠진 것: n > 220의 2-노름과 고유값, n > 200의 expm·expm_multiply(밀집 Arb의 비용), 축 하나의 벡터 노름(테스트가 `sd.linalg.vector_norm`과 scipy와 견준다), 랭크 부족·넓은 직사각의 최소제곱, NaN이 있는 `coo/nan_inf`의 합.
- `tests/sparse/refdata.py`가 파일을 읽고 계약(§4)의 척도를 둔다. 테스트와 정확도 체계가 같이 쓴다.

### 7.4 scipy 1.18.1과 R 4.6.2의 기록
- **scipy** (`tools/gen_sparse_scipy.py` → `tests/sparse/data/scipy.json`, 기록 661개): 반올림이 아닌 동작(결과 형식·형·dtype, `nnz`와 0 지우기, NaN 규칙, 예외), SpMV·SpMM의 비트, 반복법의 반복 수·상태·해, `eigsh`·`eigs`·`svds`(고정 `v0`·`rng`), 직접 풀이와 특이 행렬, `expm`, `norm`. 찾은 것:
  - SpMV: CSR·CSC·COO의 `@ x`가 모두 FMA 순차 합과 비트까지 같다(cancel/50, long_rows/300 포함). 더 넓은 무작위 행렬(nnz 10⁵–10⁶, 행 길이 3–100)과 2차원 라플라시안에서 `csr @ x`(int32·int64 색인), `csc @ x`, `csr @ X`(k = 16)도 같다(`tools/sparse_probe.py spmv`).
  - 중복 더하기의 순서는 하나가 아니다(§2.3).
  - 결과 형식: `eye_array`·`diags_array`는 DIA, `kron`·`block_array`·`block_diag`·`tril`·`triu`·`random_array`·`A.multiply(dense)`·`max(axis=)`는 COO, CSR의 전치는 CSC, `A / B`와 `sum(axis=)`는 밀집이다. 정수 대각의 `diags_array`는 float64와 `FutureWarning`이다.
  - 예외: `A / 0`은 `ZeroDivisionError`, `A + 1`은 `NotImplementedError`다. `A * 0`은 패턴을 두고 `A - A`는 nnz 0이다. `A.multiply(NaN이 있는 밀집)`은 저장된 자리에만 NaN이다.
  - 특이: 구조적 특이에서 `spsolve`는 `MatrixRankWarning`과 NaN, `splu`는 `RuntimeError: Factor is exactly singular`다. 값으로 특이한 행렬(dupcol/40, lap2d/10n)은 경고 없이 크기 3×10¹⁹, 5×10¹⁴의 유한한 해다(Accelerate와 같은 빈틈, §3.6).
- **R의 고유값**(SP6, SP97: `tools/gen_sparse_r.py --eigen` → `tests/sparse/data/r_eigen.json`, RSpectra 0.16.2·irlba 2.4.1, 기록 18개): `eigs_sym`(LM, SA, LA, BE와 σ = 0), `eigs`(LM, LR), `svds`, `irlba`. RSpectra의 Lanczos는 겹친 고유값의 중복도를 놓친다(lap2d/30 'LM': 7.898·7.949가 두 번씩인데 한 번씩만 찾는다); 테스트는 R의 값이 모두 참값이고, R이 원하는 집합을 찾은 곳에서 sd와 같은지 본다.
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
- **정확도** (`bench/accuracy/sparse.py`, 등록함): SP2에서 `spmv`(영역 `matvec`, `rmatvec`)와 `sums`(`row_sums`, `col_sums`, `total`) 두 스윕이 생겼다. 점은 출력 원소 하나이고, 판정은 `tests/sparse/refdata.py`의 계약(γ_m Σ|aᵢⱼxⱼ| 단위, 경계 1)이다. 비교 패키지 열은 scipy(그 자리에서 부름)와 R Matrix(`%*%`, `crossprod`, `rowSums`, `colSums`, `sum`)다. 2026-11-21의 실행(작업 중인 트리)에서 다섯 영역 모두 계약 안(가장 큰 것 0.83)이고 scipy·R도 sd의 경계 안이다. 기준선은 커밋 뒤에 깨끗한 트리에서 `promote`한다. SP4에서 `solve`(영역 `solve`, 참값 30점, 원소마다 |x̂ᵢ − xᵢ| / (C_SOLVE·κ∞·ε·‖x‖∞), 비교 패키지 scipy `spsolve`·R `solve`)를 더했다. SP5에서 `iterative`(영역 `cg`, `minres`, `gmres`, `bicgstab`, `lsqr`, `lsmr`)를 더했다: `solve`의 참값 30점(cg는 양정치, minres는 대칭·에르미트, gmres·bicgstab은 모두)과 `lstsq`의 2점(lsqr·lsmr)을 rtol(atol = btol) 10⁻¹⁰과 scipy의 기본 상한(gmres만 n 주기)으로 풀고 수렴한 결과만 잰다(못 한 것은 `excluded`; 스윕 전체가 6초). 계약은 판정이 약속하는 앞오차다: 정방은 |x̂ᵢ − xᵢ| ≤ ‖A⁻¹‖∞·rtol·‖b‖₂(‖A⁻¹‖∞ = κ∞/‖A‖∞), minres는 그 판정이 잔차의 어림에 대한 것이라 ‖A⁻¹‖∞·(rtol + ε·κ∞)·‖A‖_F‖x̂‖₂(§3.7의 잔차 간격), 최소제곱은 ‖(AᵀA)⁻¹‖∞·atol·‖A‖_F‖r̂‖₂다. 2026-11-21의 실행에서 여섯 영역 모두 계약 안이다(가장 큰 것: bicgstab 0.39, gmres 0.31, cg 0.10, minres 0.081, lsqr·lsmr 0.051; 수렴하지 못해 뺀 것 cg 2, minres 2, gmres 6, bicgstab 4: graded·near_singular·blockdiag 꼴). 비교 패키지는 scipy(같은 허용오차; bicgstab 0.32, 나머지는 sd와 같은 크기)이고 R Matrix에는 반복법이 없다. SP6에서 `eig`(영역 `eigsh`, `svds`, `eigs`)를 더했다: eig.npz의 점마다 값 하나, |v − ref| / (C_EIG·ε·‖A‖₂)(`eigs`는 κ(λ)를 곱한다; 가장 가까운 참값). 2026-11-21의 실행에서 셋 모두 계약 안이다(eigsh 0.50, svds 0.009, eigs 0.011; 4초). 비교 패키지는 scipy(eigsh 0.15, eigs 0.020, svds는 rect/80x30d 'SM'에서 2×10¹²: AᴴA의 반올림)와 R RSpectra(eigsh 2.4: lap1d/1000 'SA'에서 tol 10⁻¹²에 멈춘 값, eigs 0.27, svds 0.011; 'SM'은 RSpectra에 없어 뺐다)다. SP7에서 `norms`(영역 `1`, `-1`, `inf`, `-inf`, `fro`, `2`; norms.npz의 점마다 값 하나, SP109의 계약)와 `matfuncs`(영역 `expm`: expm.npz의 모든 원소, `expm_multiply`: expmv.npz의 exp(sA)b, `expm_multiply_interval`: t = 0..1의 다섯 점; C_EXPM 단위)를 더했다. 2026-11-22의 실행에서 아홉 영역 모두 계약 안이다(가장 큰 것: expm_multiply_interval 0.25, expm 0.24, expm_multiply 0.20, 합의 노름 0.17, 2-노름 0.013, fro 0.007; 2초). 비교 패키지는 scipy(expm 0.48, expm_multiply는 sd와 같은 값, 노름은 sd와 같은 크기; 2-노름은 empty/rows에서 `ArpackError`라 그 점을 뺐다)와 R Matrix(`norm(A, "1")`·`"I"`·`"F"`·`"2"`, `expm`; 실수 행렬; expm 0.42)다. `onenormest`는 계약이 없어(하한) 스윕이 없다. SP3의 함수(인덱싱, 대입, 대각, 만들기)는 반올림이 없는 연산이라 스윕이 없다(값은 입력의 값이거나 곱 하나이고, 테스트가 밀집 결과와 비트까지 견준다). 기준선은 SP128로 만들었다(`bench/accuracy/results/sparse.json`, 표 `bench/sparse/accuracy.md`; 일곱 스윕 26영역 모두 계약 안, 가장 큰 것 0.83). SP122·SP127과 노름 다시 계산의 나눗셈 뒤에 잰 것이고, 영역마다의 값은 위에 적은 마일스톤의 실행과 같다(합의 노름 0.17, Frobenius 0.0069, 2-노름 0.013, expm 0.24, eigsh 0.50). 2026-11-22의 성능 점검(SP130–SP136) 뒤의 실행도 26영역 모두 계약 안이고, 기준선보다 나빠진 것은 `solve`·`splu`·`factorized` 0.097 → 0.253(SP133), eigsh 0.496 → 0.70, eigs 0.011 → 0.021, svds 0.0089 → 0.0100, 2-노름 0.0126 → 0.014(SP131)다.
- **속도** (`bench/speed/cases/sparse.py`, 등록함): 사례마다 마일스톤과 필요한 sd.sparse 이름을 적어 두고, 그 마일스톤이 `DONE`에 들고 이름이 공개되면 사례가 된다(SP1의 형식 변환과 만들기 12가지, SP2의 15가지, SP3의 인덱싱 5가지·만들기 6가지(`random_array`는 난수가 달라 모양과 저장 수만 견준다), 크기마다; 기준선 `bench/speed/results/sparse.json`, 표 `bench/sparse/speed.md`). 희소 결과는 값·(행 + 1)·(열 + 1)로 가중한 세 합으로 견준다(형식·순서·중복과 무관). 비교 대상은 scipy와 R Matrix(같은 R 세션), 크기는 §5다. SP4에서 `spsolve`·`splu`·`splu solve`·`spsolve_triangular`(격자 다섯)를 더했다. SP5에서 반복법 여섯(`cg`·`minres` rtol 10⁻¹⁰과 `bicgstab` rtol 10⁻⁸은 격자 다섯, `gmres` rtol 10⁻⁸과 `lsqr`·`lsmr` atol = btol = 10⁻⁸은 lap2d/1000을 뺀 넷; 27사례, 결과는 x를 상대 10⁻²로 견준다)을 더했다. R Matrix에는 반복법이 없고, Accelerate의 반복법은 사례가 아니라 탐침(`tools/sparse_probe.py iter`)으로 잰 값을 §3.7에 적었다. Accelerate를 직접 부른 값(감싸기 비용)은 사례로 두지 않고 §3.6에 잰 값으로 적었다(분해의 10–15%). SP6에서 `eigsh(k=6)`·`eigs(k=6)`(lap2d/100, lap3d/22; 같은 v0 = ones), `eigsh(k=6, sigma=0)`(lap2d/100·300, lap3d/22), `svds(k=6)`(무작위 nnz 10⁴–10⁶)을 더했다(10사례, 값은 정렬한 고유값·특이값을 상대 10⁻⁸로 견준다). 큰 격자의 'LM'은 scipy가 한 번에 수 분이라(§5 표: n = 4.9×10⁵에서 189초) 사례로 두지 않았다. R의 RSpectra는 속도 사례에 넣지 않았다(정확도 스윕과 기록의 비교 대상). SP7에서 `norm`(ord 1·inf·Frobenius·`axis=1`·2; R Matrix는 1·inf·Frobenius), `onenormest`(무작위 nnz 10⁴–10⁷; scipy의 열이 무작위라 값은 상대 0.7로 견준다), `expm_multiply`(격자 다섯, 0..1의 11점은 lap2d/1000을 뺀 넷), `expm`(lap2d/30, lap3d/10; 결과가 거의 밀집이라 작은 격자만)을 더했다(34사례).
- **골든** (`tests/golden/cases_sparse.py`, 기록 `tests/golden/data/sparse.json`): SP1의 812케이스(모든 입력 꼴, 네 형식 사이의 변환, 밀집, 전치, `reshape`·`resize`, `nonzero`·`count_nonzero`, `eliminate_zeros`·`sum_duplicates`, 값의 메서드, DOK 원소, 틀린 호출). 2×10⁵개의 무작위 COO는 3·10스레드에서 양동이 길을 탄다. SP2가 4,627케이스를 더했다(모두 5,439)(같은 아홉 행렬 × 네 형식의 모든 연산자, 스칼라·밀집과의 연산과 행·열 브로드캐스트, 원소별 함수, 곱, `axis`·`keepdims`·`dtype`·`explicit`의 리덕션, 틀린 호출; 큰 두 행렬은 병렬 줄 구간과 1스레드 길을 탄다). 1·3·10스레드의 결과가 모두 같다. SP3이 1,069케이스를 더했다(모두 6,508)(같은 아홉 행렬 × 네 형식에서 모든 꼴의 키로 읽기, 스칼라·밀집·희소 값의 대입(복사본에), `diagonal`·`trace`·`setdiag`, 만들기 함수 열두 개와 그 형식·dtype, 틀린 호출; 큰 두 행렬은 병렬 줄 구간의 고르기를 탄다). 1·3·10스레드의 결과가 모두 같다. SP4가 158케이스를 더했다(모두 6,666)(2차원 라플라시안 144 × 144, 무작위 비대칭 300 × 300, 복소 60 × 60을 네 형식으로: 밀집 1·2차원·희소 우변의 `spsolve`, `trans='N'·'T'·'H'`의 `splu().solve`, `factorized`, `inv`, 아래·위·단위 대각의 `spsolve_triangular`(64열은 3·10스레드에서 열 묶음이 스레드풀에서 돈다), `spbandwidth`·`is_sptriangular`, `LinearOperator`의 곱과 조합, 특이와 틀린 호출, `sd.linalg`의 거절 메시지). 1·3·10스레드의 결과가 모두 같다(Accelerate의 LU는 sd의 스레드 수와 무관하다). SP5가 230케이스를 더했다(모두 6,896)(2차원 라플라시안, 무작위 비대칭, 복소 에르미트, 복소 비대칭, 직사각 행렬에 여섯 풀이를 네 형식·밀집·`aslinearoperator`·Python `LinearOperator`로, `x0`·Jacobi 전처리·상한·`atol`·`restart`·`shift`·`damp`·`conlim`, 붕괴와 b = 0, 결과 객체 전체(필드와 `summary()`), 틀린 호출; n = 4×10⁴의 띠 행렬 일곱 문제는 3·10스레드에서 곱·내적·노름·갱신이 스레드풀에서 돈다). 1·3·10스레드의 결과가 모두 같다. SP6이 129케이스를 더했다(모두 7,025)(2차원 라플라시안, 무작위 대칭·비대칭, 복소 에르미트·비대칭, 직사각 행렬에 `eigsh`·`eigs`·`svds`를 네 형식·밀집·연산자로, 모든 `which`, M·σ·buckling·Cayley·`OPpart`·`OPinv`, `v0`·`rng`·`ncv`·`tol`, 반복 상한, 결과 객체 전체, 틀린 호출; n = 4×10⁴의 띠 행렬 넷은 3·10스레드에서 곱·내적·기저 갱신이 스레드풀에서 돈다). 1·3·10스레드의 결과가 모두 같다. `error sd.linalg.eigh sparse`는 SP6에서 그 글이 `sd.sparse.eigsh`를 짚게 바뀌었다(§3.10의 계획대로). SP7이 270케이스를 더했다(모두 7,295)(2차원 라플라시안, 무작위 비대칭, 복소, 위·아래 삼각, 대각 행렬의 `expm`을 네 형식으로, `expm_multiply`의 벡터·행렬 B와 구간의 모든 경로·밀집·연산자·`traceA`, `onenormest`의 `t`·`max_iter`·벡터·밀집·연산자와 결과 객체(`summary()`, repr, 풀기; SP102를 고치며 다시 기록했고 바뀐 것은 `onenormest` 케이스뿐이다), `norm`의 모든 `ord`와 `axis`(2차원·1차원, 실수·복소·정수·bool, 넘침의 다시 계산, NaN), 틀린 호출, `sd.linalg.expm`·`matrix_norm`의 거절 글; n = 4×10⁴의 띠 행렬 다섯 문제는 3·10스레드에서 곱과 벡터 연산이 스레드풀에서 돈다). 1·3·10스레드의 결과가 모두 같고 기존 7,025케이스는 그대로다. SP8은 커널을 바꾸고도 값이 모두 그대로였고, 고친 동작의 다섯(`error random_array rng`의 메시지(SP112), `spbandwidth` 넷의 결과 형(SP115))을 다시 기록하고 `kronsum complex special`(SP111) 하나를 더했다(모두 7,296; 모든 영역 34,109케이스가 1·3·10스레드에서 같다). 공개 이름마다 그 마일스톤에서 더한다. 2026-11-22의 리뷰 고침에서 바뀐 24케이스만 까닭과 함께 다시 기록했다(1·3·10스레드에서 같다; 다른 영역의 골든은 그대로): `eigsh herm`의 아홉(`LA`, `LM`, `SM`, `coo`, `csc`, `dok`, `dense`, `python operator`, `sigma`)과 `eigsh long complex`는 복소 에르미트 결과를 오름차순으로 정렬했다(SP127; 값과 열의 자리만 바뀐다), `expm diag`·`expm lower`·`expm upper`의 넷씩(형식 넷)은 대각·삼각 고침이 libm `exp`라 1 ulp 바뀌었다(SP122; 밀집 `sd.linalg.expm`과 같은 비트), `error lsqr no rmatvec`는 메시지가 `lsqr(): the operator was made without 'rmatvec'`이 되었고(풀이 함수의 이름, 반복 전 검사), `error numpy operand`는 Python의 `unsupported operand type(s)` 문장이 되었다(SP123). 2026-11-22의 성능 점검은 값의 끝자리를 바꾼 결정의 케이스만 바꾼다(1·3·10스레드에서 같다): `eigsh`·`eigs`·`svds`의 모든 케이스와 그것을 쓰는 `norm`의 `ord=2`·`summary`·`unpack`(SP131, SP132), 띠 행렬의 `cg`·`minres`·`gmres`·`bicgstab`·`lsqr`·`lsmr`(`fastsum`의 내적, SP131), `spsolve`·`splu().solve`·`factorized`·`inv`(SP133, SP134), LU의 풀이를 쓰는 `expm`(SP133, SP134). 만들기·곱·인덱싱의 케이스는 그대로다(SP136).
  - **워커의 메모리**(사용자 승인 2026-11-21): SP1의 큰 무작위 행렬은 20,000 × 15,000(3×10⁸자리)과 30 × 10⁶이라 `toarray`·밀집 케이스 하나가 워커 하나를 약 10 GB로 키웠다(`--jobs 2`에서 11 GB). 같은 2×10⁵개의 중복 있는 원소(양동이 길과 병렬 줄 구간을 타는 크기)를 2,000 × 1,500(자리의 약 7%)과 30 × 10⁵(줄마다 약 6,700개: L = 4096을 넘는 줄의 조각 합)에 두어, 밀집 결과가 많아야 3×10⁶개가 되게 줄였다. 다시 기록한 뒤 `/usr/bin/time -l tools/golden.py check --jobs 10`(모든 영역, 다른 무거운 일 없음)에서 sparse 워커의 최대 RSS는 0.58 GB였고(모든 워커 중 가장 큰 것은 array의 1.56 GB, `time -l`의 최대 1.63 GB), 전체 확인은 29초, sparse 기록은 1분 2초에서 12초가 되었다.
- **나중 일**(SP34): SP1의 속도는 다른 에이전트의 일과 함께 재서 † 줄(다른 프로세스가 1코어 넘게 쓴 사례)이 있다. 기계가 한가할 때 `.venv/bin/python -m bench.speed run --rerun-busy build/speed/<SP1 실행>.jsonl --promote`로 다시 잰다.

---

## 8. 마일스톤
| 마일스톤 | 내용 |
|---|---|
| SP0 준비 (완료·승인) | API 문서 초안(`docs/api/sparse.md`: 모든 공개 이름의 시그니처, scipy → sd 이름의 대응표, 마일스톤별 질문), 시험 행렬, 참값 도구(Arb), scipy·R 대조 도구와 데이터, Accelerate 탐침(§7), int32·int64 색인의 SpMV 측정(§2.2), 속도 기준선, 정확도·속도 체계에 넣는 계획 |
| SP1 형식과 변환 (완료·승인 2026-11-21) | `coo_array csr_array csc_array dok_array sparray issparse`, 생성자, 표준 꼴(§2.3), 형식 변환, `sd.ndarray`·numpy·scipy와 주고받기(§2.5), `astype copy .T transpose matrix_transpose conj real imag reshape resize nonzero eliminate_zeros sum_duplicates`, repr |
| SP2 연산 (완료·승인 2026-11-21) | 산술과 비교(§3.2), 원소별 함수, `multiply maximum minimum power`, SpMV·SpMM·밀집 @ 희소·SpGEMM, 리덕션(§3.3; `trace`·`diagonal`은 SP3) |
| SP3 인덱싱과 만들기 (완료·승인 2026-11-21) | 인덱싱과 대입, `diagonal setdiag trace`, `eye_array diags_array random_array kron kronsum hstack vstack block_array block_diag tril triu find` |
| SP4 직접 풀이 (완료·승인(위임) 2026-11-21) | `spsolve splu factorized inv spsolve_triangular is_sptriangular spbandwidth`, `LinearOperator aslinearoperator`, `sd.linalg`·배열의 `TypeError`(§3.10) |
| SP5 반복법 (완료·승인(위임) 2026-11-21) | `cg gmres minres bicgstab lsqr lsmr`와 결과 객체 |
| SP6 고유값과 특이값 (완료·승인(위임) 2026-11-21) | `eigsh eigs svds`와 결과 객체(ARPACK 옮기기, 두꺼운 재시작 GKL), linalg 래퍼에 `steqr lahqr trevc trsen geqr2 lanhs1 larf` |
| SP7 행렬 함수와 노름 (완료·승인(위임) 2026-11-22) | `expm expm_multiply onenormest norm` |
| SP8 최적화와 일관성 점검 (완료·승인(위임) 2026-11-22) | 모든 공개 함수를 크기마다 scipy·R과 견주어 고치기(SP1에서 넘어온 것: nnz 10⁵–10⁶의 1스레드 COO → CSR·CSR → CSC, SP33), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다. SP1–SP3은 차례로 간다(앞의 것에 기댄다). SP4가 `LinearOperator`를 만들므로 SP5–SP7은 그 뒤이고, 셋끼리는 순서를 바꿀 수 있다(SP7의 `norm(ord=2)`는 SP6의 `svds`를 쓴다). 기존 영역의 결과는 비트까지 같아야 한다(§3.10의 판별은 희소 입력에서만 동작한다).

**나중 후보** (첫 판 밖): BSR·DIA(§2.1), n차원 COO, `save_npz`·`load_npz`, `spilu`, 결정적인 희소 Cholesky와 LDLᵀ(§3.6의 나중 후보; 그러면 `cholesky` 같은 scipy에 없는 함수를 둘지도 묻는다), 희소 QR로 하는 직접 최소제곱, `lobpcg`, `bicg cgs qmr tfqmr lgmres gcrotmk`, `matrix_power`, `funm_multiply_krylov`, `LaplacianNd`. 그래프 알고리즘은 따로 된 영역이다(§1.2).

---

## 9. 마일스톤을 시작하기 전에 물을 것
추천을 첫째에 둔다.

- **SP0**: 정했다(SP6–SP20, §1.1).
- **SP1**: 정했다(SP21–SP34, §1.1).
- **SP2**: 정했다(SP35–SP47, §1.1).
- **SP3**: 정했다(SP48–SP52, §1.1). SP3 뒤에 정했다(SP53–SP58, §1.1).
- **SP4**: 정했다(SP59–SP65, §1.1). SP4 뒤에 정했다(SP66–SP71, §1.1; 추천안, 사용자 위임).
- **SP5**: 정했다(SP72–SP75, §1.1; 추천안, 사용자 위임 2026-11-21). SP5에서 정했다(SP76–SP82, §1.1; 추천안, 사용자 위임).
- **SP6**: 정했다(SP83–SP87, §1.1; 추천안, 사용자 위임 2026-11-21). SP6에서 정했다(SP88–SP96, §1.1; 추천안, 사용자 위임). R의 `RSpectra`·`irlba`는 사용자 승인으로 설치했다(SP97, 2026-11-21).
- **SP7**: 정했다(SP98–SP101, §1.1; 추천안, 사용자 위임 2026-11-21). SP7에서 정했다(SP102–SP110, §1.1; 추천안, 사용자 위임).
- **SP8**: 정했다(SP111–SP117, §1.1; 추천안, 사용자 위임 2026-11-21). SP8 뒤에 정했다(SP118–SP120; 추천안, 사용자 위임 2026-11-22): 메서드의 틀린 호출도 이름으로 시작, `block_diag(arrays)`, 속도 표는 생성 파일 그대로. 코드 리뷰는 사용자가 따로 한다.
- **2026-11-22의 성능 점검**: 정했다(SP130–SP136; 추천안, 사용자 위임). 사용자에게 물을 것(1·2는 보이는 동작을 바꾼다; 추천은 모두 지금대로):
  1. 생성자의 색인 나누기(SP30, SP135): 받은 `indices`·`indptr`·`coords`가 int64이고 연속이면 복사하지 않고 나눠 쓴다(검사는 남긴다). 이득: nnz 10⁷의 `csr_array((data, indices, indptr))`가 메모리 복사 몫만큼 빨라진다(지금은 scipy의 수백 배 시간이고 생성자 시간의 대부분이 복사다; 10⁴에서는 Python 층이라 scipy와 비슷하다). 손해: 사용자가 받은 배열을 나중에 고치면 표준 꼴이 깨지고 커널이 범위 밖을 읽을 수 있다(색인을 쓰기 금지로 바꿀 수는 없다: 사용자의 배열이다).
  2. `diags_array`의 형식(SP6, SP51): `format=None`이 CSR이라 scipy(DIA를 돌려준다)의 0.13–0.45배다(nnz 10⁷–10⁵; 10⁴에서는 1.8배). DIA를 넣으면(SP6의 나중 후보) 형식 하나의 변환·연산·인덱싱이 따라 붙는다.
  3. 직접 풀이의 다듬기 문턱(SP133)은 8ε로 정했다: 다듬지 않은 풀이의 정규화 역오차가 가장 큰 것 3.54ε(lap3d/22 `trans='T'`)로 계약 4ε에 가깝다. 4ε로 내리면 계약을 성분별로도 지키지만 lap3d/22처럼 4–8ε인 풀이가 다시 두 번 푼다.
