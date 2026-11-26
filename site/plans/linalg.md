# linalg (`sd.linalg`) — 얼개

기준: `API-INVENTORY.md`의 `sd.linalg` 41개 이름. 개요 `src/content/areas/linalg.mdx`(이미 있음, 그룹 8개), 본보기로 쓴 `solve.mdx`, `lstsq.mdx`가 이미 있다(L1에서 새 공통 결정에 맞춰 다시 점검).

## 그룹과 페이지 (그룹 이름은 개요 frontmatter 그대로)

### L1 Systems, factorizations and eigenvalues
- **Linear systems**: `solve`(있음), `lstsq`(있음), `inv`, `pinv`, `lu_solve`, `cholesky_solve`, `solve_triangular`
- **Factorizations**: `lu`, `cholesky`, `qr`(qr, qr_pivoted), `ldl`, `schur`, `hessenberg`, `svd`(svd, svdvals)
- **Eigenvalues**: `eigh`(eigh, eigvalsh), `eig`(eig, eigvals)

### L2 Norms, products, structured matrices and matrix functions
- **Norms and scalar measures**: `vector_norm`, `matrix_norm`, `cond`, `det`(det, slogdet), `matrix_rank`
- **Products and powers**: `vecdot`(sd.vecdot과 다른 함수: bool·int를 float64로 계산 — `/api/array/vecdot/`와 서로 링크하고 차이를 한 줄로), `outer`, `cross`, `kron`(kron, kronsum), `matrix_power`, `trace`, `diagonal`, `block_diag`
- **Structured matrices**: `solve_tridiagonal`, `solve_banded`, `solve_sylvester`, `bandwidth`
- **Matrix functions**: `expm`
- **Errors**: `LinAlgError`(class)
- L2 끝에 개요 최종 점검(glance, 그룹 설명, Concepts, 비교 표).

## 공통 결정
- 배열 영역의 공통 결정(`plans/array.md` "공통 결정")과 `WRITING.md`를 그대로 따른다(타입 표기, operand/argument, raises 문구, 정확도 문구, NumPy 끝자리 문장, 예제 출력 정확 일치 등).
- 본보기: `api/linalg/solve.mdx`, `lstsq.mdx`, `areas/linalg.mdx`, 그리고 배열 영역의 `matmul.mdx`.
- 모양 표기: 행렬은 `(..., m, n)`, 정사각은 `(..., n, n)`, 벡터 `(n,)`. 스택·브로드캐스트·우변 `b` 읽는 법은 개요 Concepts에만 설명하고 페이지는 아래 앵커로 해당 절을 가리킨다(`#concepts` 하나로 묶어 걸지 않는다).
- 결과가 여러 개인 함수(lu, qr, svd, eigh, eig, lstsq, slogdet …): 반환 객체의 필드 이름과 언패킹 가능 여부를 돌려 보고 returns에 필드별로 적는다. 결과 클래스가 공개 이름이 아니면 클래스 페이지를 만들지 않는다.
- 페이지마다: 입력 dtype별 결과 dtype(bool·int → float64, complex), 정사각/비정사각·빈 행렬·스택, NaN·inf 입력의 처리(ValueError인지), 특이·비양정치 등에서 `LinAlgError`가 나는 조건, 분해의 부호·순서 규약(고유값 정렬 순서, 특이값 내림차순, 피벗 표현, Q·R의 부호), 재구성 예제(`q @ r` 등)를 넣는다.
- 알고리즘은 사용자가 선택·해석에 필요할 때만 Notes에(예: partial pivoting, 대칭 문제 전용 여부). 외부 라이브러리 이름은 쓰지 않는다.
- 비교 절: 제목 `## Comparison with other libraries`, NumPy(`numpy.linalg`)와 SciPy(`scipy.linalg`)만(R은 쓰지 않음, 2026-11-27 사용자 결정; `WRITING.md` §10). 두 라이브러리를 다 다루면 `**NumPy.**`, `**SciPy.**` 굵은 머리말로 나누고(이 순서), 하나뿐이면 그 이름을 대는 문장 뒤에 bullet. 돌려서 확인한 것만.
- 스레드: `set_num_threads`가 제한하지 않는 연산이 있다는 사실은 `/api/array/threads/`에만 있고, 페이지에서 되풀이하지 않는다.
- (L1 점검 뒤 추가) 개요 Concepts 앵커: `#matrices-and-stacks`, `#right-hand-sides`, `#result-types`, `#named-tuples`, `#finite-values`, `#singular-matrices`, `#errors-in-a-stack`, `#symmetric-and-hermitian-input`. 페이지 앵커: `/api/linalg/lu/#factor-format`, `/api/linalg/cholesky/#factor-format`, `/api/linalg/cholesky/#positive-definiteness`(제목 바꾸지 않음).
- 입력 문구: "<역할>, of any dtype, or a stack of them (see [matrices and stacks](/api/linalg/#matrices-and-stacks))." 우변: "Right-hand side: one vector, or `k` vectors as columns (see [right-hand sides](/api/linalg/#right-hand-sides))."
- 반환 dtype: 첫 결과에 "`float64`, or `complex128` when `x` is complex (see [result types](/api/linalg/#result-types))", 나머지는 "Same dtype as `L`.", 정수 결과는 "Always `int64`." "Integer and boolean inputs are converted to `float64` first."는 페이지에 쓰지 않는다(개요 Result types에만). 규칙이 다른 함수(`diagonal`, `block_diag`는 입력 dtype 유지, 확인함)만 페이지에 적는다.
- 이름 있는 튜플: "The result is a named tuple `LUResult(L, U, perm)`. Its fields are read by name or by position, and its arrays are read-only (see [named tuples](/api/linalg/#named-tuples))." 클래스·필드 이름은 돌려서 확인. 배열 하나만 돌려주는 함수는 쓰기 가능.
- 따옴표: `default`·`choices`·raises와 산문 속 단독 값은 작은따옴표(`'reduced'`, Python 오류 메시지와 같게), 산문 속 호출식은 예제와 같게(`mode="complete"`), 예제 코드는 그대로. choices 뜻 칸은 qr·svd의 `mode` 꼴.
- 결정성(부호·순서·기저 선택이 있을 때만): "The function makes the same choice on every run with the same number of threads (see [determinism and threads](/start/conventions/#determinism-and-threads))."
- raises: 모양은 "has fewer than two dimensions or is not square". NaN은 "if `x` contains NaN or infinity"(한 삼각형만 읽으면 "if its lower triangle contains NaN or infinity"). `LinAlgError` 항목마다 끝에 "For a stack, the message gives the index of the first such matrix."(특이·0 대각처럼 조건이 있을 때) 또는 "... of the first matrix that failed."(수렴 실패). 모든 실패 코드에 색인이 붙는다(확인함). 검사 순서·스택에서 첫 실패가 예외를 정하는 규칙은 개요 Errors in a stack에만.
- 비트 동일 주장은 실수·복소 모두(스택 포함) 직접 재 본 것만 쓰고, 아니면 "agree only up to rounding". 잰 사실: `inv(a) == solve(a, eye)`, `lu_solve(*lu(a), b) == solve(a, b)`(단 `a` 실수·`b` 복소는 다름), `cholesky(x, upper=True)`는 `cholesky(x)`의 켤레 전치와 같음, `cond`(`ord` 2, -2)·`matrix_norm`(`ord` 2, -2, `'nuc'` = `sd.sum(svdvals(x))`)·`matrix_rank`(기본 `rtol`)는 `svdvals` 값으로 계산한 것과 같은 비트(실수·복소, 계수 부족 포함 각 140건).
