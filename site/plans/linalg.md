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
- 모양 표기: 행렬은 `(..., m, n)`, 정사각은 `(..., n, n)`, 벡터 `(n,)`. 스택·브로드캐스트·우변 `b` 읽는 법은 개요 Concepts에만 설명하고 페이지는 `/api/linalg/#concepts`의 해당 문단을 가리킨다(실제 앵커 이름은 빌드된 개요에서 확인).
- 결과가 여러 개인 함수(lu, qr, svd, eigh, eig, lstsq, slogdet …): 반환 객체의 필드 이름과 언패킹 가능 여부를 돌려 보고 returns에 필드별로 적는다. 결과 클래스가 공개 이름이 아니면 클래스 페이지를 만들지 않는다.
- 페이지마다: 입력 dtype별 결과 dtype(bool·int → float64, complex), 정사각/비정사각·빈 행렬·스택, NaN·inf 입력의 처리(ValueError인지), 특이·비양정치 등에서 `LinAlgError`가 나는 조건, 분해의 부호·순서 규약(고유값 정렬 순서, 특이값 내림차순, 피벗 표현, Q·R의 부호), 재구성 예제(`q @ r` 등)를 넣는다.
- 알고리즘은 사용자가 선택·해석에 필요할 때만 Notes에(예: partial pivoting, 대칭 문제 전용 여부). 외부 라이브러리 이름은 쓰지 않는다.
- 비교 절: NumPy(`numpy.linalg`), SciPy(`scipy.linalg`), R(base: `solve`, `qr`, `svd`, `eigen`, `chol`, `det`, `kronecker` 등). 돌려서 확인한 것만.
- 스레드: `set_num_threads`가 제한하지 않는 연산이 있다는 사실은 `/api/array/threads/`에만 있고, 페이지에서 되풀이하지 않는다.
