# 선형대수 문서 참조 대응표 (임시)

> 코드 주석과 테스트의 옛 참조("API.md §0.2", "S11" 등)를 새 문서로 바꾸는 데 쓴다. 참조를 모두 바꾸면 이 파일을 지운다.

- 옛 문서: `stardust-linalg/API.md`, `DESIGN.md`(절 번호, 규칙 ID `S1`–`S11`), `ARCHITECTURE.md`(`A1`, `A2`)
- 새 문서: `docs/api/linalg.md` (아래 표의 `§`는 따로 적지 않으면 이 문서의 절), 루트 `API.md`, `docs/api/array.md`
- "인용 수"는 이 저장소의 `csrc/linalg/`, `csrc/bind/linalg_bind.cpp`, `src/stardust/linalg/`, `tests/linalg/`에서 센 값이다.
  - 코드는 옛 API.md를 `API.md §N`과 `API §N` 두 형태로 인용한다. 둘 다 셌다.
  - `API.md §0.1, §0.4`처럼 여러 절을 한 번에 인용하면 절마다 하나로 셌다.

## 1. 옛 API.md의 절

| 옛 참조 | 내용 | 새 절 | 인용 수 |
|---|---|---|---|
| 머리말 (`import stardust_linalg as sdla`) | 확정 이력, import | `sd.linalg` (머리말). 이력은 옮기지 않는다 | 0 |
| §0 | 공통 규칙 | §1 | 0 |
| §0.1 | stardust-numpy에서 이어받는 규칙 | 시그니처 §1.1, dtype 객체 API.md §1, 0차원 결과 §1.6 (array.md §1.2) | 4 |
| §0.2 | 입력: `sd.ndarray`만, 배치 브로드캐스트, 우변 `b`, 대칭/에르미트 입력, 정사각 | 입력과 배치 §1.2, 우변 `b` §1.3, 대칭·에르미트·삼각 §1.4 | 8 |
| §0.3 | 결과: dtype, 형태, namedtuple, `perm`, 레이아웃, 결정성, 스레드, pairwise 합 | dtype §1.5, 형태·namedtuple·`perm` §1.6, 레이아웃 §1.7, 결정성·스레드·합 §1.11 (API.md §3) | 6 |
| §0.4 | 옵션 이름, 두지 않는 옵션, 구조는 함수 이름으로 | §1.1 (옵션 표와 그 아래), `rtol`의 정의 §3 `matrix_rank`, §4 `lstsq` | 6 |
| §0.5 | 빈 행렬 | §1.10 | 0 |
| §1 | 기본 연산 | §2 | 3 |
| §2 | 노름과 지표 | §3 | 3 |
| §3 | 선형 시스템 | §4 | 5 |
| §4 | 분해 | §5 | 7 |
| §5 | 특수 구조 | §6 | 1 |
| §6 | 확장 단계 (`expm`, randomized SVD) | §7 (`expm`). randomized SVD는 공개 API가 아니라서 새 문서에 없다 | 2 |
| §7 | 예외 | §8, 메시지와 규칙은 §1.8 | 1 |

## 2. 규칙 ID

| ID | 출처 | 내용 | 새 절 | 인용 수 |
|---|---|---|---|---|
| Z1 | stardust-numpy DESIGN (옛 API.md §0.1이 인용) | 0차원 결과는 Python 스칼라 | §1.6, array.md §1.2 | 1 |
| C8 | stardust-numpy DESIGN (옛 API.md §0.3이 인용) | 결과 개수가 다른 변형은 함수를 나눈다 | §1.6 ("결과의 형태는 옵션에 따라 바뀌지 않는다"). 배열 쪽은 array.md §8.3 | 0 |
| S1 | linalg DESIGN §2.8 | 대각선 오프셋 이름은 `k` | §1.1 (옵션 표) | 0 |
| S2 | linalg DESIGN §2.8 | `svd(mode='reduced')` | §5 (`svd`), §1.1 | 0 |
| S3 | linalg DESIGN §2.8 | 삼각 규칙: 아래 삼각만 읽기, `upper=False`, `solve_triangular`의 `upper`는 필수 | §1.4, §4 (`solve_triangular`), §1.1 (필수 옵션) | 1 (`tests/linalg/test_m3_factor.py`) |
| S4 | linalg DESIGN §2.8 | `eig`, `eigvals`를 (실수부, 허수부) 사전순으로 정렬 | §5 ("정렬 순서") | 2 (`ops/nonsym.cpp`, `tests/linalg/test_m6_nonsym.py`) |
| S5 | linalg DESIGN §2.8 | `cholesky_solve`, `lu_solve`는 분해 결과를 그대로 받는다 | §4 | 0 |
| S6 | linalg DESIGN §2.8 | `lstsq`는 `(solution, rank, singular_values)`, `residuals` 없음 | §4 (`lstsq`), §1.6 | 1 |
| S7 | linalg DESIGN §2.8 | 순열은 int64 `perm`, 마지막 필드 | §1.6 | 0 |
| S8 | linalg DESIGN §2.8 | 결과 형태는 옵션으로 바뀌지 않는다 | §1.6 | 0 |
| S9 | linalg DESIGN §2.8 | 일반화 고유값은 `eigh(a, b=None, /)`, `a v = λ b v`만 | §5 (`eigh`) | 0 |
| S10 | linalg DESIGN §2.8 | `sd.ndarray`만, `outer`는 배치 허용, `diagonal`은 복사본 | §1.2, §2 (`outer`, `diagonal`) | 2 |
| S11 | linalg DESIGN §2.8 | 빈 행렬의 관례, 배치 크기 0, `cond`만 `ValueError` | §1.10 | 4 |
| A1 | linalg ARCHITECTURE §1 | Python과 C++의 경계 | API가 아니다. `docs/design/linalg.md` | 0 (`ARCHITECTURE.md §1`로 1) |
| A2 | linalg ARCHITECTURE §3.4 | 배치 실패는 가장 작은 인덱스를 보고 | 관찰되는 동작은 §1.8 ("배치의 실패"). 구현은 `docs/design/linalg.md` | 1 |

- **주의: `S3`, `S4`가 두 가지로 쓰인다.**
  - `csrc/linalg/types.hpp`의 `S3`(60행), `S4`(70행)는 stardust-numpy DESIGN의 ID다: float → int 포화, complex → 실수 변환 에러. 새 절은 array.md §1.5 ("명시적 변환")다. array-refmap.md의 S3, S4와 같다.
  - 나머지 `S3`, `S4`는 위 표의 linalg ID다.

## 3. 옛 DESIGN.md의 절 (코드가 인용하는 것)
API에 해당하는 부분만 새 절로 옮겼다. 구현과 측정 근거는 아직 없는 `docs/design/linalg.md`로 간다 (DESIGN.md §5.4).

| 옛 참조 | 내용 | 새 절 | 인용 수 |
|---|---|---|---|
| DESIGN §1.4 | 공통 규칙: 배치, bool/int64 → float64, 경고 없음 | §1.2, §1.5 | 1 |
| DESIGN §1.5 | 빌드 환경 (ILP64, 새 LAPACK) | `docs/design/linalg.md` | 1 |
| DESIGN §2.3 | 메모리 레이아웃 | 결과 레이아웃 §1.7. 입력 패킹은 `docs/design/linalg.md` | 1 |
| DESIGN §2.4 | 실패할 때의 동작 | §1.8, §1.9 | 7 |
| DESIGN §2.5 | 배치 병렬화, 스레드 수 | §1.11, API.md §3. 분할 방식은 `docs/design/linalg.md` | 2 |
| DESIGN §2.6 | 비교 기준 (numpy, scipy, 잔차 검사) | `docs/design/linalg.md` | 1 |
| DESIGN §2.9 | 설계 검증 측정 (경계 비용 등) | `docs/design/linalg.md` | 1 |
| DESIGN §2.10 | `solve`/`inv`의 LAPACK 경로, 배치 실패, 에러 순서 | `inv == solve(x, I)` §4, 에러 순서 §1.8. 경로와 측정은 `docs/design/linalg.md` | 3 |
| DESIGN §2.11 | `diagonal` dtype, `matrix_power` 검사, `trace(dtype=)`, p-노름, 노름 특수값, `vecdot` 축 | §1.5, §2, §3 (`vector_norm`). complex 곱의 FMA 식은 `docs/design/linalg.md` | 7 |
| DESIGN §2.12 | 삼각만 읽기, `cholesky(upper=True)`, `lu`의 특이, 대각 0, 에러 순서, `lu_solve`, 레이아웃 무관 | §1.4, §1.8, §1.9, §1.11, §4, §5 | 9 |
| DESIGN §2.13 | `lstsq`의 드라이버, `rtol ≥ 1`, `rtol`의 타입, `qr` | §1.1 (`rtol`), §4 (`lstsq`), §5. 드라이버는 `docs/design/linalg.md` | 3 |
| DESIGN §2.14 | 고유값·SVD 드라이버, 값 전용 모드, `cond`, `matrix_norm` 검사, 수렴 실패 메시지 | §3 (`cond`, `matrix_norm`), §5 ("값만 구하는 함수"), §1.8 (메시지). 드라이버는 `docs/design/linalg.md` | 11 |
| DESIGN §2.15 | `eig` 정렬과 벡터, `ldl`, `schur`, `hessenberg` | §5 | 3 |
| DESIGN §2.16 | 특수 구조: 드라이버, (1, 1) 띠, Sylvester의 공통 고유값, 검사 순서, 모양 | §6 | 4 |
| DESIGN §2.17 | pairwise 합, 16바이트 정렬, workspace, 패킹 | 합과 배치 비트 §1.11. 나머지는 `docs/design/linalg.md` | 4 |
| DESIGN §2.18 | 스레드풀, 분할 기준, 결정성 | §1.11, API.md §3. 분할 기준은 `docs/design/linalg.md` | 2 |
| DESIGN §2.19 | `kron`, `cross`, `expm` | §2, §7. `expm`의 알고리즘 세부는 `docs/design/linalg.md` | 2 |
| DESIGN §2.21 | 결과를 바꾸지 않는 최적화 | `docs/design/linalg.md` | 1 |
| DESIGN §2.22 | 코드리뷰 버그 수정 (`expm`, 큰 `ord`) | 큰 `ord`의 `OverflowError` §1.8. `expm` 수정은 `docs/design/linalg.md` | 2 |

## 4. 대응하지 못한 것

| 참조 | 이유 | 제안 |
|---|---|---|
| `DESIGN.md §3.2` (`src/stardust/linalg/__init__.py`, `csrc/bind/linalg_bind.cpp`, 2회) | 옛 문서가 아니라 새 루트 DESIGN.md §3.2(확장 모듈 하나)다 | 그대로 둔다 |
| `ARCHITECTURE.md §3.1`–`§3.4`, `§6` (7회) | 내부 구조와 테스트 방침이다. API가 아니다 | `docs/design/linalg.md` (`§3.4`의 실패 보고는 §1.8도 함께) |
| `ARCHITECTURE.md §6.5` (`input.cpp`, `outblock.cpp`, 2회) | stardust-numpy ARCHITECTURE의 버퍼 내보내기 절이다 | array.md §10.5 |
| `stardust-numpy API §6` (`ops/basic.cpp`, 1회) | 옛 배열 문서의 리덕션 절이다 | array.md §7.5 |
| 옛 API.md §6의 randomized SVD | 구현되지 않은 계획이다 | 새 문서에 없다. 구현할 때 새 절을 만든다 |
| 옛 API.md 머리말의 "3라운드 확정" | 결정 이력이다 | 옮기지 않는다 (DESIGN.md §5.4) |
| M0–M11 등 마일스톤 번호, `ROADMAP.md M<n>` | 문서 참조가 아니라 이력이다 | DESIGN.md §5.4에 따라 지운다 |
