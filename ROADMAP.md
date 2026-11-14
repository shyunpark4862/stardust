# stardust 로드맵

> 상태: **확정** (2026-11-12). 앞부분은 세 프로젝트의 통합(M0–M7, 완료), 뒷부분은 새 영역 [특수함수 (sd.special)](#특수함수-영역-sdspecial)(S0–S9, 구현 전)이다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

## 역할 분담 (세 프로젝트와 같다)
- 구현은 Claude가 전부 한다.
- 마일스톤마다 해설을 쓴다: 무엇을, 어떻게, 왜 바꿨는지, 측정 결과와 함께.
  - 위치: `docs/learn/M<n>-<주제>.md` (2026-11-15에 지웠다. 안내서와 해설은 나중에 한 번에 다시 쓴다)
- **영역을 옮기는 마일스톤은 그 영역의 문서도 새로 쓴다** (DESIGN §5.4): `docs/design/<영역>.md`, `docs/api/<영역>.md`, 코드 주석의 옛 문서 참조 교체
- 마일스톤마다 멈춘다: 커밋하고, 해설을 전달하고, 사용자가 승인하면 다음으로 간다.

## 진행 원칙
- **옮기기와 고치기를 같은 마일스톤에 섞지 않는다.**
  - 옮기는 마일스톤(M1, M2, M5)은 결과가 비트까지 같아야 한다.
  - 고치는 마일스톤(M3, M4, M6)은 바뀌는 것을 미리 DESIGN.md에 적고, 그 밖의 결과는 비트까지 같아야 한다.
- **안전망 세 가지** (M0에서 만든다)
  1. 세 프로젝트의 테스트 전부 (약 13,000줄). 마일스톤마다 전부 통과한다.
  2. **골든 출력**: 옛 패키지 세 개로 고정된 입력 집합의 결과를 바이트로 저장해 둔다. 통합판의 결과와 비교한다 (스레드 수 1, 3, 10).
  3. **벤치마크 기준선**: 세 프로젝트의 최종 보고서 표를 같은 자리에서 다시 잰다. 통합판이 10% 넘게 느려진 줄은 원인을 밝힌다.
- 옛 저장소 세 개는 고치지 않는다. 골든 출력과 벤치마크 기준선을 만드는 데 쓰고, 통합이 끝나면 동결한다.

## 진행 상태
| 마일스톤 | 단계 | 상태 |
|---|---|---|
| M0 저장소, 이력, 기준선 | 준비 | 완료 (2026-11-12) |
| M1 배열 코어 옮기기 | 1단계 | 완료 (2026-11-12) |
| M2 linalg 옮기기 | 1단계 | 완료 (2026-11-12) |
| M3 linalg 경계 없애기 | 1단계 | 완료 (2026-11-12) (1단계 보고서 포함) |
| M4 공통 바닥 정리 | 2단계 준비 | 완료 (2026-11-12) |
| M5 frame 옮기기 | 2단계 | 완료 (2026-11-12) |
| M6 frame과 배열 잇기, 일관성 맞추기 | 2단계 | 완료 (2026-11-12) |
| M7 마무리 | 마무리 | 완료 (2026-11-12) (최종 보고서) |
| M8 일관성 2차 | 다듬기 | 완료 (2026-11-12), 변경 목록 DESIGN §10 |

## 마일스톤

### M0. 저장소, 이력, 기준선
- 새 저장소 `~/Desktop/stardust/stardust` (DESIGN §6 Q1)
- **세 저장소의 이력을 보존해서 들여온다.**
  - 이 Mac의 git에는 `git subtree`와 `git filter-repo`가 없다. 표준 명령만으로 한다.
  - 저장소마다: `git fetch ../stardust-numpy main` → `git merge -s ours --no-commit --allow-unrelated-histories FETCH_HEAD` → `git read-tree --prefix=legacy/numpy/ -u FETCH_HEAD` → 커밋
  - 결과: `legacy/numpy/`, `legacy/linalg/`, `legacy/pandas/` 아래에 세 프로젝트가 이력과 함께 들어온다. 이후 마일스톤은 `git mv`로 여기서 꺼내 간다.
  - 옛 문서와 experiments는 들여오지 않는다 (DESIGN §5.4). git에 없는 bench 스크립트만 `legacy/<영역>/bench/`로 복사한다.
- 골든 출력 스크립트와 결과 (`bench/merge/golden/`)
- 벤치마크 기준선 (`bench/merge/baseline.md`)
- 뼈대: `pyproject.toml`(패키지 이름 `stardust`), `CMakeLists.txt`, 빈 `_core`, `import stardust as sd` 확인
- `.gitignore`: 세 프로젝트의 정책 + `docs/api/` 공개 (DESIGN §5.2)

### M1. 배열 코어 옮기기
- stardust-numpy의 C++과 Python을 `csrc/array/`, `csrc/base/`, `csrc/parallel/`, `src/stardust/`로 옮긴다.
- `stardust_numpy` → `stardust`. 테스트의 import만 바꾼다.
- 동작은 하나도 바꾸지 않는다. 골든 출력이 비트까지 같다.
- 문서: `docs/design/array.md`, `docs/api/array.md`, `API.md`(색인과 공통 규칙), README 뼈대. `test_api_surface`가 `docs/api/array.md`를 읽게 바꾼다. 배열 코드의 주석 참조를 바꾼다.

### M2. linalg 옮기기
- linalg의 C++을 같은 `_core`에 넣고 `_core.linalg` 하위 모듈로 묶는다. `sd.linalg`로 노출한다.
- 이 단계에서는 경계 코드(`Input`, `outblock`, `pyrefs`)를 **그대로 둔다.** buffer protocol로 같은 모듈 안의 배열을 읽는 어색한 상태지만, 옮기기만 하고 결과가 같은지 먼저 확인한다.
- 이름공간 `sdla` → `sd::linalg`. linalg의 풀 복사본은 지운다(배열의 풀을 쓴다).
- 컴파일 플래그 통일(`-fno-builtin-sin/-cos`)의 영향을 여기서 확인한다.
- 문서: `docs/design/linalg.md`(경계 부분은 M3의 모습으로 미리 쓴다), `docs/api/linalg.md`. linalg 코드의 주석 참조를 바꾼다.
  - linalg 주석은 stardust-numpy 문서도 인용한다 (예: stardust-numpy ARCHITECTURE의 6.5절). `docs/design/array-refmap.md`, `docs/api/array-refmap.md`로 새 배열 문서에 잇는다.
  - 공통 규칙(dtype 객체, 에러 종류, 스레드)을 `docs/api/array.md`의 0절에서 루트 `API.md`로 옮기고, 그 절을 인용하던 주석을 고친다.

### M3. linalg 경계 없애기
- `Input` → `const NDArray&`, `alloc_*` + `sdnp.asarray` → `new_array_strided`, `pyrefs`의 sdnp 참조 제거 (ARCHITECTURE §3)
- linalg의 `DType`, `saturate_to_int`, `pairwise.hpp`를 `base/`의 것으로 바꾼다.
- linalg의 알려진 정리 항목(`docs/design/linalg.md` §10.9)을 함께 고친다. 결과는 바뀌지 않는다.
  - `trace`/`diagonal`에서 k = −2⁶³일 때 `-k`의 부호 있는 오버플로(정의되지 않은 동작)
  - `ldl`이 LAPACK workspace 몫의 scratch 슬롯을 임시 버퍼로 쓰는 것, `ord` 파서와 도우미의 중복, `cond`의 이중 패킹
- **측정**: 작은 행렬(`det` 2×2, `inv` 2×2, `solve` 3×3, `qr` 3×3)의 호출 시간이 얼마나 줄었는가. 예상은 결과 하나에 약 200ns.
- 1단계 끝. 여기서 1단계 보고서를 쓴다.

### M4. 공통 바닥 정리 (frame을 받을 준비)
- 스레드풀을 pandas판(상위 집합)으로 바꾼다. 병렬화 기준값을 영역별 상수로 옮긴다 (ARCHITECTURE §4.1).
- `base/neon.hpp`, `accelerate/vforce.hpp`가 `array/kernels/scalar_ops.hpp`를 include하는 역방향 의존을 없앤다. (M4에서 한 방법: 둘은 배열의 연산 정의에 묶인 배열 코드라서 `array/kernels/`로 돌려보냈다. vForce는 연산과 무관한 래퍼만 `accelerate/`에 남겼다. 스칼라 변환 규칙을 `base/`로 내릴 필요는 없었다.)
- 풀을 바꾼 뒤 같은 자리 비교(`bench/merge/sameseat_table.py`)를 돌린다. 짧은 병렬 호출(예: `solve_tridiagonal` (1000, 3, 3), 10스레드)은 프로세스마다 약 7µs와 10.5µs로 갈리므로 프로세스를 여러 번 띄워 분포로 본다.
- `Buffer`에 `size()`를 더한다 (ARCHITECTURE §4.2).
- `base/dtype.hpp`에 `String`을 더한다 (DESIGN §4.2). `Datetime64`와 Column용 방문자는 frame 코드가 들어오는 M5 이후에 더한다. 사용자에게 보이는 dtype 표기는 M6에서 바꾼다.
- `accelerate/` 정리: BLAS, LAPACK, vForce 래퍼를 한곳에 모은다.
- 배열과 linalg의 결과는 비트까지 같다.

### M5. frame 옮기기
- stardust-pandas의 C++과 Python을 `csrc/frame/`, `src/stardust/frame/`으로 옮긴다. `sd.frame`으로 노출한다.
- 이름공간 `sd` → `sd::frame` (Buffer와 풀은 `base/`, `parallel/`의 것을 쓴다).
- `sdpd.set_num_threads` → `sd.set_num_threads`. 테스트의 import와 스레드 수 호출만 바꾼다.
- dtype 표기 통일(DESIGN §4.2, Q5)은 M6에서 한다. 여기서는 frame의 문자열 dtype을 그대로 둔다.
- frame의 pairwise 합(세 번째 사본)을 `base/pairwise.hpp`의 정의로 바꾼다. `for_chunks`, `Unlocked`, `kChunkRows`는 `frame/`으로, frame의 기준값은 `sd::frame::kParallelThreshold`(2¹⁷)로.
- frame의 결과는 비트까지 같다.
- 문서: `docs/design/frame.md`, `docs/api/frame.md`. frame 코드의 주석 참조를 바꾼다.

### M6. frame과 배열 잇기, 일관성 맞추기
- `Series.to_array()`, `DataFrame.to_array()`, 배열에서 Series/DataFrame 만들기 (DESIGN §4.3)
- DESIGN §4.4와 2라운드(§8)에서 정한 동작, 시그니처, 이름을 세 영역에 걸쳐 맞춘다. 이 마일스톤에서만 결과와 공개 API가 바뀔 수 있다.
  - 시작 전에 바뀌는 것의 목록(DESIGN §8.4)을 함수 단위로 확정하고, 골든 비교의 "예상된 차이"로 적는다.
  - 3번(int/float 비교)과 11번(int 평균)은 배열 쪽 비용을 먼저 쟀고(DESIGN §9.8), 사용자가 두 항목 모두 배열 규칙으로 통일하기로 정했다 (DESIGN §8).
- dtype 표기 통일: `.dtype`은 객체, 두 영역 모두 객체와 이름 문자열을 받는다 (DESIGN §4.2).
- 문서: 바뀐 규칙을 `docs/api/`와 `docs/design/`에 반영한다.
- `tests/interop/`: 경계를 오가는 테스트, Copy-on-Write와 읽기 전용 뷰의 테스트

### M7. 마무리
- 안내서를 새로 쓴다: 영역을 넘나드는 한 권 (DESIGN §5.4; 2026-11-15에 지웠다)
- README 완성, `API.md` 색인 정리
- 최종 보고서: 기준선 대비 통합판의 성능, 줄어든 코드(중복 제거), 경계 비용
- 남은 옛 문서 참조가 없는지 확인한다 (코드, 테스트, 새 문서). `legacy/`가 비었는지 확인하고 지운다. 옛 참조 대응표(`docs/*/<영역>-refmap.md`)도 지운다.
- 옛 저장소 세 개를 동결한다 (README에 "stardust로 옮겨 감" 표시, DESIGN §5.2)

---

### M8. 일관성 2차
- M7 뒤의 감사에서 찾은 비일관과 DESIGN §8.4의 "남음" 항목을 고친다. 함수 단위 목록은 DESIGN §10 (사용자 확정 2026-11-12).
- M6처럼 골든 비교의 "예상된 차이"(`bench/merge/golden/expected.py`)로 묶고, 그 밖의 차이는 없어야 한다.
- 단계
  1. frame의 동작과 이름 (§10.1의 frame 항목, §10.2)
  2. 배열·linalg 항목, 에러 메시지 정리, 문서 (§10.1의 나머지, §10.3, §10.4)
  3. 골든·같은 자리 성능 확인

# 특수함수 영역 (sd.special)

> 상태: **확정** (2026-11-12). 설계는 [docs/design/special.md](docs/design/special.md), API는 [docs/api/special.md](docs/api/special.md).
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

## 진행 원칙
- 통합 로드맵의 원칙을 따른다: 마일스톤마다 커밋하고, 결과를 전달하고, 사용자가 승인하면 다음으로 간다. 해설과 안내서는 쓰지 않는다(2026-11-15 사용자 결정: 나중에 한 번에 쓴다. 그때까지 쓴 `docs/learn/`, `docs/guide/`는 지웠다).
- **배열 영역을 고치는 일은 S0에만 있다.** 기존 함수의 결과는 비트까지 같아야 한다 (골든 출력, 스레드 수 1, 3, 10).
- 마일스톤마다 확인하는 것
  - 정확도: mpmath 기준값 대비 최대·99.9% ULP를 보장 영역과 느슨한 영역으로 나눠 표로 낸다 (docs/design/special.md §6).
  - 비트: NEON 경로 = 스칼라 정의, 스레드 1·3·10, `f(x)[i] == f(x[i])`
  - 성능: scipy와 같은 자리에서 번갈아 잰다. 1스레드와 10스레드, 원소 10⁶개와 스칼라 1개.
- 마일스톤을 시작하기 전에 물을 것은 docs/design/special.md §10에 있다.

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| S0 기반 | 완료 (2026-11-13) |
| S1 로지스틱·정보 | 완료 (2026-11-13) |
| S2 오차함수 | 완료 (2026-11-14) |
| S3 감마와 제타 | 완료 (2026-11-14) |
| S4 불완전 함수 | 완료 (2026-11-14) |
| S5 리덕션, 지수적분, 타원 | 완료 (2026-11-14) |
| S6 Bessel과 Airy | 완료 (2026-11-14) |
| S7 초기하와 Lambert W | 완료 (2026-11-14) |
| S8 복소수 | 완료 (2026-11-14) |
| S9 마무리 | 완료 (2026-11-14) |
| S10 Boost.Math 도입 (S9 뒤 후속, 사용자 결정) | 완료 (2026-11-14) |
| S11 최적화 (Debye ν ≥ 30, Boost 조합, 스칼라 지름길) | 완료·승인 (2026-11-15), bench/special/s11/ |

## 마일스톤

### S0. 기반
- **배열 영역** (docs/design/special.md §3.1, §3.3)
  - 일반 n항 원소별 디스패처 `nary<Op, N>`. `unary`, `binary`는 그대로 둔다.
  - `kMaxOperands` 4 → 5. 그 비용을 잰다 (짧은 원소별 호출, 1스레드).
  - 이터레이터에 원소당 비용(작업량 가중치)을 넘기는 길
  - 골든 출력으로 배열·linalg·frame의 결과가 비트까지 같은지 확인한다.
- **special 뼈대**: `csrc/special/`(`common.hpp`, `cf.hpp`, `cost.hpp`), `bind/special_bind.cpp`, `src/stardust/special/`, `tests/special/`(`test_api_surface.py`가 docs/api/special.md를 읽는다), `bench/special/`
- **도구**: mpmath 기준값 생성 `tools/special_ref.py`와 `tests/special/ref/`, 정확도 표 `bench/special/accuracy.py`, scipy와의 같은 자리 비교, 원소당 비용 측정. `pyproject.toml`의 `test`에 mpmath를 더한다.

### S1. 로지스틱과 정보량
- `expit logit log_expit xlogy xlog1py entr rel_entr`
- NEON `expit`, `logit`. 커널 안의 vForce 버퍼 경로(Q16)를 처음 만든다. vForce의 특수값을 이 함수들에 대해 다시 확인한다.

### S2. 오차함수
- `erf erfc erfcx erfinv erfcinv ndtr log_ndtr ndtri` (Cephes 3구간, AS241)
- NEON `erf erfc ndtr ndtri`. Horner와 Estrin, 분기 없애기를 잰다.

### S3. 감마와 제타
- `gamma gammaln rgamma digamma beta betaln binom zeta polygamma`
- `gamma`의 후보 A(Lanczos)와 B(Cephes)를 견주어 고른다 (docs/design/special.md §1.3). NEON `gammaln`

### S4. 불완전 함수
- `gammainc gammaincc betainc betaincc`, 역함수 `gammaincinv gammainccinv betaincinv betainccinv`, 앞 인자(`special/prefix.hpp`)
- 원소마다 반복 횟수가 다를 때의 스레드 분할을 잰다.
- 불완전 베타의 세 번째 구간을 넣을지 측정 뒤 묻는다.
- 이 마일스톤이 끝나면 `sd.stats` 설계를 시작할 수 있다.

### S5. 리덕션, 지수적분, 타원적분
- `logsumexp softmax log_softmax` (두 패스, 한 패스 방식은 해설에서 비교), `expi expn`, `ellipk`(AGM), `ellipe`(Carlson)

### S6. Bessel과 Airy
- 0·1차 12개 (Cephes Chebyshev, Hankel 점근)
- `jv yv iv kv`, `jve yve ive kve` (Temme, Steed, Hankel)
- `airy` (`AiryResult`, 출력 넷)

### S7. 초기하 함수와 Lambert W
- `hyp1f1 hyp2f1` (실수, 선형 변환, 퇴화 극한 공식 하나), `lambertw` (가지 0, −1)

### S8. 복소수
- `wofz`, `loggamma` (새 함수), `erf erfc erfcx lambertw`의 complex 입력

### S9. 마무리
- 정확도 보고서, 성능 보고서 (scipy 대비), 안내서의 special 장(2026-11-15에 지웠다), README와 `API.md` 색인

---

# 통계 영역 (sd.stats, sd.random)

> 상태: **설계 초안** (2026-11-15). 설계는 [docs/design/stats.md](docs/design/stats.md), [docs/design/random.md](docs/design/random.md), API는 [docs/api/stats.md](docs/api/stats.md), [docs/api/random.md](docs/api/random.md).
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

## 진행 원칙
- special과 같다: 마일스톤마다 시작 전에 scipy와 다를 특수값과 보장 영역을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.
- 배열·linalg·frame의 기존 결과는 비트까지 같아야 한다(골든 출력, 스레드 1·3·10).
- 마일스톤을 시작하기 전에 물을 것은 docs/design/stats.md의 11장과 docs/design/random.md의 6장에 있다.

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| T0 배열 추가 | 승인 (2026-11-15) |
| T0b 배열: Array API 채우기 | 승인 (2026-11-15) |
| T1 special 감싸기와 `ndtri_exp` | 완료·승인 (2026-11-15) |
| T2 special 비중심 분포 함수 | 완료, 보장 영역 결정·승인 대기 (2026-11-15) |
| T3 `sd.random` | 대기 |
| T4 분포 엔진과 연속 기본 12 | 대기 |
| T5 이산 10 | 대기 |
| T6 극값·수명 6과 비중심 3 | 대기 |
| T7 표본 (`sample`) | 대기 |
| T8 기술통계 | 대기 |
| T9 다변량: 정규, t, 다항, 디리클레 | 대기 |
| T10 행렬 분포: Wishart, 역Wishart, 행렬 정규 | 대기 |
| T11 `sd.stats.tests` (설계 질문부터) | 대기 |

## 마일스톤

### T0. 배열 추가
- `cov`, `corrcoef`, `quantile`, `median`, NaN을 건너뛰는 리덕션 (배열 영역, docs/design/array.md와 docs/api/array.md에 더한다)
- 결정(2026-11-15 사용자): `quantile`은 numpy의 13가지 메서드 전부, 값은 numpy와 비트까지 같게, 결과는 언제나 float64. NaN 리덕션은 기본 세트와 `nancumulative_*`. `cov`는 `rowvar`와 `correction=0`만. 모두 NaN이거나 빈 레인은 numpy와 같은 값이고 경고가 없다(`nanargmin`·`nanargmax`는 `ValueError`).
- 결과: docs/api/array.md §7.9–§7.11, docs/design/array.md §6.10–§6.12, 측정 [bench/array/t0_statistics.md](bench/array/t0_statistics.md), 테스트 `tests/array/test_statistics.py`.

### T0b. 배열: Array API 채우기 (2026-11-15 사용자 추가)
- Array API 2024에서 빠진 28개: `meshgrid tril triu take take_along_axis flip repeat roll tile unstack count_nonzero nonzero searchsorted diff copysign hypot logaddexp nextafter reciprocal signbit tensordot vecdot can_cast finfo iinfo isdtype result_type from_dlpack`. 여기에 numpy 시그니처의 `bincount`, `histogram`을 더한다.
- 결정(사용자): `nonzero`를 넣는다(전에는 뺐다). `take`의 indices는 numpy처럼 여러 차원과 int도 받는다. DLPack은 가져오기(`from_dlpack`)와 내보내기(`__dlpack__`, `__dlpack_device__`) 둘 다. `flip`, `unstack`은 numpy처럼 뷰다. `diff`의 bool은 subtract 규칙(int64)이다(numpy는 not_equal). `histogram`의 `bins`에 문자열 규칙은 없다.
- 결과: docs/api/array.md §2.1, §3.5–§3.6, §4.4, §5.4, §6(원소별 6개), §7.12, §8.4, §9, §10.5; docs/design/array.md §1.2, §5.5, §6.13, §8.8, §12.5; 측정 [bench/array/t0b_array_api.md](bench/array/t0b_array_api.md); 테스트 `tests/array/test_array_api_additions.py`.

### T1. special 감싸기와 `ndtri_exp`
- `stdtr stdtrit chdtr chdtrc chdtri fdtr fdtrc fdtri pdtr pdtrc bdtr bdtrc nbdtr nbdtrc`(기존 함수의 조합), `ndtri_exp`
- 결정(2026-11-15 사용자): 계획한 15개만(매개변수 역함수 11개는 빼고). `pdtr bdtr nbdtr`의 k는 내림(누적분포의 뜻), n은 참값(`bdtr`의 정수 아닌 n은 NaN, `nbdtr`은 실수 n을 받는다; scipy는 내림). scipy가 일관되지 않거나 참값과 다른 특수값은 참값·극한(예: `bdtr(k<0)=0`, `bdtr(k≥n)=1`, 무한 자유도는 극한 분포, `stdtrit`은 꼬리에서도 참값). 정확도는 1e-14가 목표이고, 조합에서 깎이는 곳은 직접 계산하며, 보장 영역은 재고 나서 묻는다.
- 보장 영역(2026-11-15 사용자): 잰 범위 전체(t 자유도 ≤ 1e6, F 자유도 ≤ 1e4, k·n ≤ 1e4, `ndtri_exp`는 y 전체). 경계값 `fdtr(inf, inf, x)`는 1에 몰린 점질량, `nbdtr(k, 0, p) = 1`, `chdtri(inf, p) = NaN`으로 확정. 승인 2026-11-15.
- 결과: docs/api/special.md §13, docs/design/special.md §5.15·§2.4; 측정 bench/special/t1/(accuracy.md, speed.md); 테스트 tests/special/test_distributions.py와 참값 tests/special/ref/(15개). S4의 `betaincinv`·`betainccinv` 결함(a, b < 1에서 0·1)을 함께 고쳤다.

### T2. special 비중심 분포 함수
- `chndtr chndtrix nctdtr nctdtrit ncfdtr ncfdtri`. 보장 영역은 재고 나서 묻는다.
- 결정(2026-11-15 사용자): x(또는 p)를 맨 뒤로(`chndtr(df, nc, x)`; scipy와 다름). 위꼬리 `chndtrc nctdtrc ncfdtrc`도 공개(scipy에 없는 이름). 매개변수 역함수는 뺀다. 특수값은 T1처럼 참값·극한. Boost를 먼저 재고 깎이는 곳만 직접(재 보니 꼬리 전체가 깎여 Boost는 쓰지 않음).
- 결과: docs/api/special.md §14, docs/design/special.md §5.16·§2.4; 측정 bench/special/t2/(accuracy.md, speed.md); 테스트 tests/special/test_noncentral.py와 참값 tests/special/ref/(9개). T1·S4 결함 둘을 함께 고쳤다(`fdtr`의 bgrat log 1 근처, `chdtr`의 서브노멀 x).

### T3. `sd.random`
- 계수기형 생성기, `default_rng`, `random integers choice permutation shuffle`

### T4. 분포 엔진과 연속 기본 12
- 분포 구조체와 (분포, 메서드) Op, 신형 API의 Python 클래스, 참값 도구(`tools/stats_ref.py`)
- 정규, t, 카이제곱, F, 감마, 베타, 지수, 로그정규, 균등, 코시, 라플라스, 베이불

### T5. 이산 10
- 이항, 포아송, 기하, 음이항, 초기하, 베르누이, 이산 균등, 베타이항, 로그 급수, Zipf

### T6. 극값·수명 6과 비중심 3
- 로지스틱, Gumbel, GEV, 파레토, 레일리, 역감마, 비중심 카이제곱·t·F

### T7. 표본
- 모든 분포의 `sample` (직접 알고리즘 또는 역함수 법)

### T8. 기술통계
- `describe skew kurtosis moment zscore rankdata sem iqr mode entropy differential_entropy`

### T9. 다변량: 정규, t, 다항, 디리클레
- `pdf logpdf sample mean covariance entropy`, 다변량 정규·t의 `cdf`(Genz, 느슨한 계약)

### T10. 행렬 분포
- Wishart, 역Wishart, 행렬 정규

### T11. `sd.stats.tests`
- 검정과 상관. 설계 질문부터 시작한다.

