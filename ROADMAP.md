# stardust 로드맵

> 상태: **확정** (2026-11-12). 앞부분은 세 프로젝트의 통합(M0–M7, 완료), 뒷부분은 새 영역 [특수함수 (sd.special)](#특수함수-영역-sdspecial)(S0–S11, 완료), 그 뒤 [통계 (sd.stats, sd.random)](#통계-영역-sdstats-sdrandom)(T0–T10, 완료·승인; 검정은 뺐다), 그 뒤 [회귀 (sd.regression)](#회귀-영역-sdregression)(G0–G6 완료·승인, G7a 완료·승인, G7b 시작 전), 그 뒤 [최적화 (sd.optimize)](#최적화-영역-sdoptimize)(O0 완료·승인, O1 완료·승인, O2 완료·승인, O3 완료·승인, O4 완료·승인, O5 완료·승인, O6 1부 완료·승인), 그 뒤 [가설검정 (sd.htest)](#가설검정-영역-sdhtest)(H0 완료·승인, H1 완료·승인, H2–H10 시작 전), 그 뒤 [희소 (sd.sparse)](#희소-영역-sdsparse)(SP0–SP8 시작 전, 설계 초안)이다.
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
- 골든 출력 스크립트와 결과 (커밋 4151d2a의 `bench/merge/golden/`)
- 벤치마크 기준선 (커밋 4151d2a의 `bench/merge/baseline.md`)
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
- 풀을 바꾼 뒤 같은 자리 비교(커밋 4151d2a의 `bench/merge/sameseat_table.py`)를 돌린다. 짧은 병렬 호출(예: `solve_tridiagonal` (1000, 3, 3), 10스레드)은 프로세스마다 약 7µs와 10.5µs로 갈리므로 프로세스를 여러 번 띄워 분포로 본다.
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
- M6처럼 골든 비교의 "예상된 차이"(커밋 4151d2a의 `bench/merge/golden/expected.py`)로 묶고, 그 밖의 차이는 없어야 한다.
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
  - 정확도: 기준값(처음 mpmath, 2026-11-16부터 Arb) 대비 최대·99.9% ULP를 보장 영역과 느슨한 영역으로 나눠 표로 낸다 (docs/design/special.md §6).
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
| S11 최적화 (Debye ν ≥ 30, Boost 조합, 스칼라 지름길) | 완료·승인 (2026-11-15), 커밋 4151d2a의 bench/special/s11/ |

## 마일스톤

### S0. 기반
- **배열 영역** (docs/design/special.md §3.1, §3.3)
  - 일반 n항 원소별 디스패처 `nary<Op, N>`. `unary`, `binary`는 그대로 둔다.
  - `kMaxOperands` 4 → 5. 그 비용을 잰다 (짧은 원소별 호출, 1스레드).
  - 이터레이터에 원소당 비용(작업량 가중치)을 넘기는 길
  - 골든 출력으로 배열·linalg·frame의 결과가 비트까지 같은지 확인한다.
- **special 뼈대**: `csrc/special/`(`common.hpp`, `cf.hpp`, `cost.hpp`), `bind/special_bind.cpp`, `src/stardust/special/`, `tests/special/`(`test_api_surface.py`가 docs/api/special.md를 읽는다), `bench/special/`
- **도구**: 기준값 생성 `tools/special_ref.py`(2026-11-16부터 Arb, `tools/arbref.py`)와 `tests/special/ref/`, 정확도 표 `bench/special/accuracy.py`, scipy와의 같은 자리 비교, 원소당 비용 측정. `pyproject.toml`의 `test`에 mpmath를 더한다(이제 python-flint).

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

> 상태: **T0–T10 구현·승인 완료** (2026-11-18, T4–T10과 최적화 승인). 검정은 이 영역에서 뺐다(사용자 결정 2026-11-18: 나중에 따로 한다). 설계는 [docs/design/stats.md](docs/design/stats.md), [docs/design/random.md](docs/design/random.md), API는 [docs/api/stats.md](docs/api/stats.md), [docs/api/random.md](docs/api/random.md).
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
| T2 special 비중심 분포 함수 | 완료·승인 (2026-11-15) |
| T3 `sd.random` | 완료·승인 (2026-11-15) |
| T4 분포 엔진과 연속 기본 12 | 완료·승인 (2026-11-15, 승인 2026-11-18) |
| T5 이산 10 | 완료·승인 (2026-11-16, 승인 2026-11-18) |
| T6 극값·수명 6과 비중심 3 | 완료·승인 (2026-11-16, 승인 2026-11-18) |
| T7 표본 (`sample`) | 완료·승인 (2026-11-16, 승인 2026-11-18) |
| T8 기술통계 | 완료·승인 (2026-11-16, 승인 2026-11-18) |
| T9 다변량: 정규, t, 다항, 디리클레 | 완료·승인 (2026-11-16, 승인 2026-11-18) |
| T10 행렬 분포: Wishart, 역Wishart, 행렬 정규 | 완료·승인 (2026-11-16, 승인 2026-11-18) |
| stats 최적화: chunk 커널, 이산 메모, 비중심 합, 다변량 빠른 경로 (docs/design/stats.md §3.4) | 완료·승인 (2026-11-17, 승인 2026-11-18) |
| ~~T11 `sd.stats.tests`~~ | 이 영역에서 뺌 (2026-11-18, 나중에 따로). 2026-11-20 새 영역 `sd.htest`로 옮김: [가설검정 영역](#가설검정-영역-sdhtest) |

## 마일스톤

### T0. 배열 추가
- `cov`, `corrcoef`, `quantile`, `median`, NaN을 건너뛰는 리덕션 (배열 영역, docs/design/array.md와 docs/api/array.md에 더한다)
- 결정(2026-11-15 사용자): `quantile`은 numpy의 13가지 메서드 전부, 값은 numpy와 비트까지 같게, 결과는 언제나 float64. NaN 리덕션은 기본 세트와 `nancumulative_*`. `cov`는 `rowvar`와 `correction=0`만. 모두 NaN이거나 빈 레인은 numpy와 같은 값이고 경고가 없다(`nanargmin`·`nanargmax`는 `ValueError`).
- 결과: docs/api/array.md §7.9–§7.11, docs/design/array.md §6.10–§6.12, 측정(커밋 4151d2a) `bench/array/t0_statistics.md`, 테스트 `tests/array/test_statistics.py`.

### T0b. 배열: Array API 채우기 (2026-11-15 사용자 추가)
- Array API 2024에서 빠진 28개: `meshgrid tril triu take take_along_axis flip repeat roll tile unstack count_nonzero nonzero searchsorted diff copysign hypot logaddexp nextafter reciprocal signbit tensordot vecdot can_cast finfo iinfo isdtype result_type from_dlpack`. 여기에 numpy 시그니처의 `bincount`, `histogram`을 더한다.
- 결정(사용자): `nonzero`를 넣는다(전에는 뺐다). `take`의 indices는 numpy처럼 여러 차원과 int도 받는다. DLPack은 가져오기(`from_dlpack`)와 내보내기(`__dlpack__`, `__dlpack_device__`) 둘 다. `flip`, `unstack`은 numpy처럼 뷰다. `diff`의 bool은 subtract 규칙(int64)이다(numpy는 not_equal). `histogram`의 `bins`에 문자열 규칙은 없다.
- 결과: docs/api/array.md §2.1, §3.5–§3.6, §4.4, §5.4, §6(원소별 6개), §7.12, §8.4, §9, §10.5; docs/design/array.md §1.2, §5.5, §6.13, §8.8, §12.5; 측정(커밋 4151d2a) `bench/array/t0b_array_api.md`; 테스트 `tests/array/test_array_api_additions.py`.

### T1. special 감싸기와 `ndtri_exp`
- `stdtr stdtrit chdtr chdtrc chdtri fdtr fdtrc fdtri pdtr pdtrc bdtr bdtrc nbdtr nbdtrc`(기존 함수의 조합), `ndtri_exp`
- 결정(2026-11-15 사용자): 계획한 15개만(매개변수 역함수 11개는 빼고). `pdtr bdtr nbdtr`의 k는 내림(누적분포의 뜻), n은 참값(`bdtr`의 정수 아닌 n은 NaN, `nbdtr`은 실수 n을 받는다; scipy는 내림). scipy가 일관되지 않거나 참값과 다른 특수값은 참값·극한(예: `bdtr(k<0)=0`, `bdtr(k≥n)=1`, 무한 자유도는 극한 분포, `stdtrit`은 꼬리에서도 참값). 정확도는 1e-14가 목표이고, 조합에서 깎이는 곳은 직접 계산하며, 보장 영역은 재고 나서 묻는다.
- 보장 영역(2026-11-15 사용자): 잰 범위 전체(t 자유도 ≤ 1e6, F 자유도 ≤ 1e4, k·n ≤ 1e4, `ndtri_exp`는 y 전체). 경계값 `fdtr(inf, inf, x)`는 1에 몰린 점질량, `nbdtr(k, 0, p) = 1`, `chdtri(inf, p) = NaN`으로 확정. 승인 2026-11-15.
- 결과: docs/api/special.md §13, docs/design/special.md §5.15·§2.4; 측정(커밋 4151d2a) bench/special/t1/(accuracy.md, speed.md); 테스트 tests/special/test_distributions.py와 참값 tests/special/ref/(15개). S4의 `betaincinv`·`betainccinv` 결함(a, b < 1에서 0·1)을 함께 고쳤다.

### T2. special 비중심 분포 함수
- `chndtr chndtrix nctdtr nctdtrit ncfdtr ncfdtri`. 보장 영역은 재고 나서 묻는다.
- 결정(2026-11-15 사용자): x(또는 p)를 맨 뒤로(`chndtr(df, nc, x)`; scipy와 다름). 위꼬리 `chndtrc nctdtrc ncfdtrc`도 공개(scipy에 없는 이름). 매개변수 역함수는 뺀다. 특수값은 T1처럼 참값·극한. Boost를 먼저 재고 깎이는 곳만 직접(재 보니 꼬리 전체가 깎여 Boost는 쓰지 않음).
- 결과: docs/api/special.md §14, docs/design/special.md §5.16·§2.4; 측정(커밋 4151d2a) bench/special/t2/(accuracy.md, speed.md); 테스트 tests/special/test_noncentral.py와 참값 tests/special/ref/(9개). T1·S4 결함 둘을 함께 고쳤다(`fdtr`의 bgrat log 1 근처, `chdtr`의 서브노멀 x).
- 보장 영역(2026-11-15 사용자): 잰 범위 전체(카이제곱·F 자유도 1e-2–1e4, nc 1e-3–1e4; t 자유도 1e-1–1e5, |δ| ≤ 100; 확률 1e-300까지). `chndtr(0, nc, 0)` = NaN(`chdtr(0, 0)`과 일관)으로 확정. 승인 2026-11-15.

### T3. `sd.random`
- 계수기형 생성기, `default_rng`, `random integers choice permutation` (`shuffle`은 R9로 뺐다)
- 결정(2026-11-15 사용자): docs/design/random.md R5–R10 (Philox4×64-10, numpy SeedSequence 해시, spawn·state·피클, 공유 시 구간 원자적 예약, shuffle 없이 permutation, integers 브로드캐스트, choice p 허용치). 승인 때 R11: `integers`의 `shape=None`은 브로드캐스트 모양, `default_rng(Generator)`는 같은 객체. 승인 2026-11-15.
- 결과: docs/api/random.md, docs/design/random.md §2–§6; 측정(커밋 4151d2a) bench/random/t3_speed.md; 테스트 tests/random/test_random.py.

### T4. 분포 엔진과 연속 기본 12
- 분포 구조체와 (분포, 메서드) Op, 신형 API의 Python 클래스, 참값 도구(`tools/stats_ref.py`)
- 정규, t, 카이제곱, F, 감마, 베타, 지수, 로그정규, 균등, 코시, 라플라스, 베이불
- 결정(2026-11-15 사용자): docs/design/stats.md T18–T24 (매개변수 이름, 키워드 생성자와 기본값, 읽기 전용 속성, log 계약, mode, out=, 초과 첨도, 보장 영역과 특수값).
- 결과: csrc/stats/(engine.hpp, numeric.hpp, dist/), csrc/bind/stats_bind.cpp, src/stardust/stats/; docs/api/stats.md §2, docs/design/stats.md §2–§3; 측정(커밋 4151d2a) bench/stats/t4/(accuracy.md, speed.md); 테스트 tests/stats/(test_distributions.py, test_reference.py)와 참값 tests/stats/ref/(216개).

### T5. 이산 10
- 이항, 포아송, 기하, 음이항, 초기하, 베르누이, 이산 균등, 베타이항, 로그 급수, Zipf
- 결정(사용자): docs/design/stats.md T25(2026-11-15: mode 동점 NaN, 기하는 시행 횟수 k ≥ 1, 이산 균등 [low, high), 초기하 `Hypergeometric(total, successes, draws)`, median = icdf(1/2)). 2026-11-16: 분위수 계약은 후방 오차(목표를 상대 1e-14 안에서 움직인 정확한 분위수 중 하나, `ilog…`는 t에 대해), 보장 영역은 잰 매개변수 범위.
- 결과: csrc/stats/discrete.hpp, csrc/stats/dist/(discrete_basic.hpp, discrete_more.hpp), src/stardust/stats/_discrete.py; docs/api/stats.md §2·§2.3, docs/design/stats.md §3.2; 측정(커밋 4151d2a) bench/stats/t5/(accuracy.md, speed.md); 테스트 tests/stats/test_discrete.py와 참값 tests/stats/ref/(이산 분위수 파일에 kmin·kmax).

### T6. 극값·수명 6과 비중심 3
- 로지스틱, Gumbel, GEV, 파레토, 레일리, 역감마, 비중심 카이제곱·t·F
- 결정(사용자, 2026-11-16): docs/design/stats.md T26 (GEV는 scipy의 c, `Rayleigh(*, scale=1)`, Gumbel은 최댓값형만, 비중심 nc ≥ 0, mode는 1e-14, entropy는 느슨한 한계, 비중심의 보장 영역은 special의 잰 범위, 비중심 t 모멘트는 1/df 급수로 1e-14).
- **참값을 Arb로 바꿨다** (사용자, 2026-11-16): special과 stats의 참값 도구 전부를 python-flint(Arb)로 바꾸고 참값을 다시 만들었다(`tools/arbref.py`, docs/design/special.md §6). 입력점은 고정하고(`--resample`로만 새로 뽑는다), 병렬은 10코어다. 옛 mpmath 참값의 오류와 새 참값이 드러낸 라이브러리 오차(`gammaincinv`, 이산 엔트로피의 자르기, 음이항 깊은 꼬리, Zipf 분위수, 비중심의 넓은 꼬리)를 고쳤다. 비중심 엔트로피 셋의 참값(10코어로 약 12시간)은 나중에 만든다.
- 결과: csrc/stats/dist/(extreme.hpp, lifetime.hpp, noncentral.hpp), csrc/stats/coeffs/nct_moments.hpp(tools/gen_nct_moments.py), csrc/special/ops/noncentral.hpp의 넓은 지수 꼬리; docs/api/stats.md §2.4, docs/design/stats.md §3.3; 측정(커밋 4151d2a) bench/stats/t6/(accuracy.md, speed.md), 그때의 special 정확도표 bench/special/arb/accuracy.md; 참값 tests/stats/ref/(555개).

### T7. 표본
- 모든 분포의 `sample` (직접 알고리즘 또는 역함수 법)
- 결정(사용자, 2026-11-16): docs/design/stats.md T27 (`rng`는 꼭 준다, 결과 모양은 shape + 매개변수 모양, 이산도 float64, 방법은 분포마다 재서 고른다).
- 결과: csrc/stats/(sample.hpp, sample_dist.hpp), csrc/bind/stats_sample.cpp, src/stardust/stats/_distribution.py의 `sample`; docs/api/stats.md §2.5, docs/design/stats.md §4; 측정(커밋 4151d2a) bench/stats/t7/(speed.md, quality.md); 테스트 tests/stats/test_sample.py.

### T8. 기술통계
- `describe skew kurtosis moment zscore rankdata sem iqr mode entropy differential_entropy`
- 결정(사용자, 2026-11-16): docs/design/stats.md T28 (axis=None 기본, nan_policy, scipy의 보정 기본값과 namedtuple, iqr의 q·method, moment의 order는 꼭, rankdata·differential_entropy는 scipy의 방법 전부).
- 결과: src/stardust/stats/_descriptive.py, csrc/bind/stats_desc.cpp; docs/api/stats.md §4, docs/design/stats.md §5; 측정(커밋 4151d2a) bench/stats/t8/speed.md; 테스트 tests/stats/test_descriptive.py.

### T9. 다변량: 정규, t, 다항, 디리클레
- `pdf logpdf sample mean covariance entropy`, 다변량 정규·t의 `cdf`(Genz, 느슨한 계약)
- 결정(사용자, 2026-11-16): docs/design/stats.md T29 (배치, 특이 공분산은 NaN, cdf는 값과 return_error, 정규 d ≤ 3과 t d ≤ 3은 정확한 공식, 보장 영역 κ ≤ 1e9·d ≤ 100, 다항 p는 허용치 안에서 나눔, 디리클레의 단체 밖은 0, t의 df = inf는 정규).
- 결과: src/stardust/stats/_multivariate.py, csrc/bind/(stats_mv.cpp, stats_mvcdf.cpp), csrc/stats/mvcdf.hpp; docs/api/stats.md §3, docs/design/stats.md §6; 측정(커밋 4151d2a) bench/stats/t9/(accuracy.md, speed.md, cdf_accuracy.md, cdf_speed.md); 테스트 tests/stats/(test_multivariate.py, test_mvcdf.py)와 참값 tests/stats/ref_mvcdf/(tools/mvcdf_ref.py).
- 미룬 것(사용자): 정규·t logpdf의 d ≥ 10 속도(SIMD로 여러 점을 함께 풀기).

### T10. 행렬 분포
- Wishart, 역Wishart, 행렬 정규
- 결정(사용자, 2026-11-16): docs/design/stats.md T30 (scipy의 이름과 메서드: mean·mode·var, 행렬 정규의 평균은 mu, x는 아래 삼각을 읽고 양의 정부호가 아니면 0, 배치와 정확도는 T9와 같다, df > p − 1 실수).
- 결과: src/stardust/stats/_matrix.py, csrc/bind/stats_matrix.cpp, csrc/stats/dd_linalg.hpp; docs/api/stats.md §3.1, docs/design/stats.md §6.3; 측정(커밋 4151d2a) bench/stats/t10/(accuracy.md, speed.md); 테스트 tests/stats/test_matrix.py.

### T11. `sd.stats.tests` (뺌)
- 검정은 이 영역에서 뺐다(사용자 결정 2026-11-18: 나중에 따로 한다). 위치(`sd.stats.tests`, 상관도 그곳)만 정해 둔 상태다(docs/design/stats.md §7).
- 위치를 바꿨다(사용자 결정 2026-11-20): 검정은 `sd.stats.tests`가 아니라 새 최상위 모듈 `sd.htest`다(상관도 그곳). 계획은 이 문서 끝의 [가설검정 영역](#가설검정-영역-sdhtest), 설계는 [docs/design/htest.md](docs/design/htest.md)(H1).

---

# 회귀 영역 (sd.regression)

> 상태: **G0–G6 구현·승인 완료** (2026-11-19, 최적화 승인 포함), **G7a 구현·승인** (2026-11-20). 그 뒤의 코드 리뷰와 사이트 작업에서 찾은 문제는 결정 G83–G116으로 고쳤다(2026-11-20까지). 설계는 [docs/design/regression.md](docs/design/regression.md), API는 [docs/api/regression.md](docs/api/regression.md).
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

## 진행 원칙
- 다른 영역과 같다: 마일스톤마다 시작 전에 물을 것(docs/design/regression.md §8)을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.
- 큰 결정(사용자, 2026-11-18): 추론과 예측을 동등하게, 선형·GLM·정규화·강건·분위·비선형 전부, 함수 + 결과 객체(배열 입력), 모듈 `sd.regression`, 절편 기본, 회귀에 딸린 t·p·F·구간은 결과에, 참값은 Arb와 statsmodels·scikit-learn 대조 (docs/design/regression.md §1.1 G1–G7).
- 기존 영역의 결과는 비트까지 같아야 한다.
- G0·G1의 질문은 정했다(사용자, 2026-11-18): docs/design/regression.md §1.1 G8–G18.
- G7의 큰 결정(사용자, 2026-11-20): 범위는 변수 선택·모형 비교와 일반 검정·진단 추가이고 다른 모형족은 나중 후보, 변수 선택은 결과 객체를 돌려주는 독립 함수(`stepwise`, `best_subset`), 대상은 `ols wls gls glm`, 교차검증 순차 선택은 정규화 교차검증과 같은 겹과 `rng`, 비교·검정·진단은 결과의 메서드이고 분산분석표는 함수 `anova` (docs/design/regression.md §1.1 G126–G130, 설계 초안 §3.8, 시작 전에 물을 것 §8; htest와 겹치는 곳도 §3.8·§8).
- htest와의 공통 규칙(사용자, 2026-11-20; docs/design/regression.md §1.1 G131–G136): 구간은 `conf_int(alpha=)`, 두 영역이 같은 값을 계산하는 커널은 htest에(`jarque_bera()`는 htest의 H2에서 htest 커널을 부른다; 같은 비트, p값은 `ChiSquared(df=2).ccdf`), 같은 검정은 같은 형(`JarqueBeraResult`는 htest의 클래스), F는 `df_num`·`df_denom`이고 χ²·t는 `df`, 분산 0은 sd.stats `skew`의 규칙(이것만 `jarque_bera()`의 골든을 바꾼다), 결과 배열은 복사 없는 쓰기 금지.

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| G0 준비: 대조 패키지, Arb 참값 도구, NIST StRD | 완료·승인 (2026-11-18) |
| G1 선형: `ols wls gls linregress`, 추론·예측, 강건 표준오차, 진단, summary | 완료·승인 (2026-11-18) |
| G2 GLM: 정규·이항·포아송·감마, IRLS + 뉴턴, 분리 증명서, 강건 공분산 | 완료·승인 (2026-11-18) |
| G3 정규화: `ridge lasso elastic_net`, 경로, 교차검증, `penalized_glm` | 완료·승인 (2026-11-18) |
| G4 강건: `rlm`(M·MM), `theilslopes`, `siegelslopes` | 완료·승인 (2026-11-18) |
| G5 분위: `quantreg` | 완료·승인 (2026-11-19) |
| G6 비선형 최소제곱: `nls` | 완료·승인 (2026-11-19) |
| 회귀 최적화: Dot2 행 훑기, 64행 타일, 정준 링크 연마, expit 공유, MM의 S 뉴턴 | 완료·승인 (2026-11-19) |
| 코드 리뷰와 사이트 작업에서 찾은 문제 (P1–P4, R1–R112) | 고침 (2026-11-19–20, 결정 G83–G116) |
| G7a 비교·검정·진단: `t_test wald_test compare_f_test compare_lr_test anova vif dffits dfbetas wresid white_test goldfeld_quandt reset_test`, GLM의 영향력 | 완료·승인 (2026-11-20, 결정 G137–G164) |
| G7b 변수 선택: `stepwise best_subset`(AIC·BIC·p값·교차검증; `ols wls gls glm`) | 시작 전 (설계 초안 2026-11-20) |

## 마일스톤

### G0. 준비
- 결과: statsmodels 0.15.0·scikit-learn 1.9.1을 test 의존성에 넣고 설치했다(Python 3.15에서 설치됨). `tools/regref.py`(Arb 참값), `tests/regression/data/`(NIST StRD 11개, itl.nist.gov), `tests/regression/nist.py`; 테스트 tests/regression/test_reference_tool.py. 기존 패키지의 NIST 성적은 docs/design/regression.md §6.
- 결과 객체의 공통 틀은 G1로 옮겼다(OLS의 필드와 함께 만든다).


### G1. 선형
- `ols wls gls linregress`, 추론(표준오차·t·p·구간·F·우도), 예측과 예측 구간, 강건 표준오차(HC0–HC3, HAC, 군집), 랭크 부족, 진단(지렛대, 스튜던트화 잔차, Cook 거리, Breusch–Pagan, Jarque–Bera), `summary`, 배치
- G1a(선형 모형과 추론)와 G1b(HAC·군집, 진단, summary)로 나눠 진행했다.
- 결정(사용자, 2026-11-18): docs/design/regression.md §1.1 G9–G30 (R식 랭크 부족, statsmodels의 필드 이름, 앞 축 배치, 필요할 때까지 최대 3번의 반복 개선, 완전 적합의 극한값, 강건 공분산도 t·F, inf는 ValueError, HAC의 지연 수, 일원 군집과 Stata의 보정).
- 결과: csrc/regression/(linear.cpp, lsq.hpp), csrc/bind/regression_bind.cpp, src/stardust/regression/_linear.py, src/stardust/_results.py; docs/api/regression.md §1–§2, docs/design/regression.md §2·§3.1–§3.3; 측정(커밋 4151d2a) bench/regression/g1/(speed.md, refine.log), bench/regression/g1b/speed.md; 테스트 tests/regression/test_linear.py, test_reference_tool.py.

### G2. GLM
- 분포족(정규·이항·포아송·감마)과 링크, IRLS와 뉴턴, 분리 증명서, 우도비와 편차·Pearson 잔차, 강건 공분산
- 결정(사용자, 2026-11-18): docs/design/regression.md §1.1 G31–G44 (미수렴은 마지막 값과 `converged=False`, 완전 분리의 판정과 결과, 척도, 이항 반응의 꼴, 가중치, 검정 분포, 비정준 링크의 마지막 걸음, 느린 수렴).
- 결과: csrc/regression/glm.cpp, src/stardust/regression/_glm.py; docs/api/regression.md §3, docs/design/regression.md §3.4; 측정(커밋 4151d2a) bench/regression/g2/(speed.md, noncanonical.md, refine.md, accuracy.log); 테스트 tests/regression/test_glm.py, test_against_r.py(tools/gen_regression_r.py).

### G3. 정규화
- `ridge lasso elastic_net`, 정규화 경로, 교차검증(`ridge_cv lasso_cv elastic_net_cv`), `penalized_glm`
- 결정(사용자, 2026-11-18): docs/design/regression.md §1.1 G45–G52 (scikit-learn의 목적함수, 경로와 교차검증, 정규화 GLM의 범위와 이름, 예측 중심의 결과(표준오차 없음), alpha = 0의 정규화 GLM은 glm의 규칙, ridge의 반복 개선).
- 결과: csrc/regression/penalized.cpp, src/stardust/regression/_penalized.py; docs/api/regression.md §4, docs/design/regression.md §3.5; 측정(커밋 4151d2a) bench/regression/g3/(accuracy.md, speed.md); 테스트 tests/regression/test_penalized.py.

### G4. 강건
- `rlm`(M·MM 추정, Huber·bisquare·Hampel), `theilslopes`, `siegelslopes`
- 결정(사용자, 2026-11-18): docs/design/regression.md §1.1 G53–G59 (statsmodels의 M과 robustbase의 MM, ψ 함수와 상수, 반복마다 MAD 척도와 Huber의 제안 2, scipy의 Theil–Sen·Siegel, rlm의 추론, MM의 큰 n).
- 결과: csrc/regression/(robust.cpp, slopes.cpp), src/stardust/regression/_robust.py; docs/api/regression.md §5, docs/design/regression.md §3.6; 측정(커밋 4151d2a) bench/regression/g4/(accuracy.md, speed.md); 테스트 tests/regression/test_robust.py, test_robust_diagnostics.py, test_against_r.py(tools/gen_robust_r.py).

### G5. 분위
- `quantreg`: Frisch–Newton 내부점법과 기저 교차, Barrodale–Roberts 단체법, 여러 분위, 표준오차(nid·iid·ker)
- 결정(사용자, 2026-11-18–19): docs/design/regression.md §1.1 G60–G67 (내부점법 뒤 정확한 꼭짓점, R quantreg의 표준오차, 여러 분위를 한 번에, 유일하지 않은 해는 사전식 최소 꼭짓점, 수렴, 정의되지 않는 표준오차, 10진 q의 동률, summary.rq의 eps).
- 결과: csrc/regression/quantile.cpp, src/stardust/regression/_quantile.py; docs/api/regression.md §6, docs/design/regression.md §3.6; 측정(커밋 4151d2a) bench/regression/g5/(accuracy.md, speed.md); 테스트 tests/regression/test_quantreg.py(tools/gen_quantreg_r.py).

### G6. 비선형 최소제곱
- `nls`: Levenberg–Marquardt(MINPACK lmder)와 TRF(scipy `least_squares`), 모수마다 간격을 고르는 5점 차분, 가우스–뉴턴·뉴턴 다듬기, 추론과 예측 구간
- 결정(사용자, 2026-11-19): docs/design/regression.md §1.1 G68–G79 (`curve_fit` 꼴의 이름과 입력, 두 방법, 정확한 야코비안, 사용자 함수의 입력, 가중치와 척도, 수렴과 정확도, 유한하지 않은 함수값, 겹친 모수, rtol 기본값, llf·aic·bic, 차분 비용).
- 결과: csrc/regression/nls.cpp, src/stardust/regression/_nls.py; docs/api/regression.md §7, docs/design/regression.md §3.7; 측정(커밋 4151d2a) bench/regression/g6/(accuracy.md, speed.md); 테스트 tests/regression/test_nls.py, nist_nls.py(NIST StRD 비선형 27개, tests/regression/data/nls/), tools/gen_nls_r.py.

### 회귀 최적화
- 행을 도는 계산의 NEON Dot2 훑기(csrc/regression/dot2.hpp), 64행 Packer 타일, 정준 링크 GLM의 정규방정식 연마(G80), glm과 정규화 로지스틱이 같이 쓰는 expit 커널(G81), MM의 S 단계 뉴턴(G82)
- 결과: 커밋 8772322; docs/design/regression.md §3.4–§3.6의 새 수치; 테스트 tests/regression/test_kernel_paths.py. g1–g4의 speed.md는 최적화 전 측정이라 다시 재야 한다(사용자가 나중에 직접 잰다).

### 코드 리뷰와 사이트 작업에서 찾은 문제
- 사이트 문서를 쓰며 찾은 P1–P4·R1–R112와 회귀 전체 코드 리뷰(2026-11-19)를 고쳤다. 사용자 결정은 docs/design/regression.md §1.1 G83–G116이다(0의 부호, 입력 복사, 상수 y의 정확한 해, 수렴과 다듬기, df_resid = 0, 빈 입력, 읽기 전용 설정 필드, `conf_int(*, alpha)`, pseudo R², 강건·분위·비선형의 예측 구간, alpha = 0의 정규화, 경계의 상수 y, 좌표 하강의 정확한 0 등).
- 결과: 커밋 6ec75a2, 5db5a13, b56b891, d0f8553, 1c56929, 2de64d1, 9f8736e, b215b6e, 2aecdc1, 4bcd109, a2c1bbd(통계·특수함수 쪽 수정 포함); 테스트 tests/regression/test_review_decisions.py.

### G7a 비교·검정·진단
- 결정(사용자, 2026-11-20): docs/design/regression.md §1.1 G137–G164(`vif dffits dfbetas`는 속성, 일반 검정은 `t_test`·`wald_test` 둘이고 분포는 모형의 규칙, R과 q만, `rlm quantreg nls`에도, 비교의 같은 자료·내포 검사, 강건 공분산의 비교는 에러, GLM의 비교, `AnovaResult`, Type I·III, 중심 VIF, GLM의 영향력, White·Goldfeld–Quandt·RESET의 꼴, R car·leaps 설치, 공개 결과 클래스, statsmodels의 필드 이름, `wresid`). 구현하며 정한 세부 G160과 G161–G164(GLS의 L·X, GLM의 X·y 사본, htest 결과 바탕의 상속, GLM 우도비 Type III는 나중 후보)도 정했다.
- 결과: csrc/regression/(linear.cpp, lsq.hpp: 적합의 얇은 Q를 내는 `thin_q` 방식), src/stardust/regression/(_tests.py, _diagnostics.py, _linear.py, _glm.py, _nls.py, __init__.py); docs/api/regression.md §9.1–§9.2, docs/design/regression.md §3.8; Arb 참값 tools/regref_g7a.py(tests/regression/refdata_g7a.py → ref/g7a.json), R 대조 tools/gen_regression_g7a_r.py(ref/g7a_r.json); 측정 bench/regression/g7a/(speed.py, results.md); 테스트 tests/regression/test_g7a.py, test_g7a_against_r.py.

---

# 최적화 영역 (sd.optimize)

> 상태: **O0 완료·승인**, **O1 완료·승인**, **O2 완료·승인**, **O3 완료·승인**, **O4 완료·승인**, **O5 완료·승인**, **O6 1부(성능) 완료·승인** (2026-11-20). 설계는 [docs/design/optimize.md](docs/design/optimize.md), API의 초안은 [docs/api/optimize.md](docs/api/optimize.md)다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

## 진행 원칙
- 다른 영역과 같다: 마일스톤마다 시작 전에 물을 것(docs/design/optimize.md §9)을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.
- 큰 결정(사용자, 2026-11-20): 다섯 덩어리(스칼라 근·최솟값, 최소제곱, 다변수 최소화, 방정식계의 근, 수치 미분), scipy의 이름과 stardust의 규칙, 결과 이름은 `converged`·`n_iter`, 스칼라의 배치 버전, nls의 커널을 옮겨 같이 쓰기, 전용 풀이는 그대로, 확률적 최적화·LP·제약·전역 최적화는 뺀다 (docs/design/optimize.md §1.1 O1–O8).
- O0의 결정(사용자, 2026-11-20): scipy 1.18.1과 같은 걸음(비트까지는 아님), 통합 함수만(개별 함수 없음), `curve_fit` 없음, `args` 없음, MGH 35와 scipy 시험 함수, Fortran 원본 대조 (docs/design/optimize.md §1.1 O9–O13). API 초안 뒤의 결정(2026-11-20): 반복 상한은 `max_iter`·`max_nfev`, 방법 이름은 소문자만, 방법마다의 옵션은 `options` dict(모르는 키는 `ValueError`), `status`·`message`는 scipy의 것 (O14–O17).
- O1의 결정(사용자, 2026-11-20): `nfev`는 차분의 호출도 센다, `bounds=None`이 기본, 유한하지 않은 마지막 cost는 `converged=False`, 반복 안의 차분은 `'2-point'`·`'3-point'`·`diff_step`(`'cs'` 없음), `loss`·`x_scale`·`lsmr`·`jac_sparsity`·`callback`·`workers`·다듬기는 미룬다, `max_nfev`의 기본은 nls의 규칙 (O18–O23). 구현 뒤의 결정(2026-11-20): 'lm'의 마지막 야코비안을 센다, 3점 차분의 `max_nfev` 기본은 100n(2n + 1), 유한하지 않은 야코비안과 3점의 한쪽 값은 `ValueError`, 빈 입력과 `diff_step`+호출 가능한 `jac`은 `ValueError`, `approx_fprime`의 모양·잡음 열·유한하지 않은 값, `check_grad`의 `rng` 에러 꼴 (O24–O31).
- O2의 결정(사용자, 2026-11-20): 근을 끼지 않는 구간은 `ValueError`(배치는 status −1), 허용오차의 기본은 scipy의 것(`xtol=0`이면 더 간다), `fprime=True` 없음, 배치 결과는 `fun`, 배치의 허용오차는 키워드, `callback` 없음, 배치는 남은 원소만 1차원으로 부른다 (O32–O38). 구현 뒤의 결정(2026-11-20): 배치 함수만 scipy의 `args`(남은 원소로 함께 줄인다), `root_scalar`는 `fun` 없이 f·f′·f″를 따로 센다, scipy 경고의 글을 `message`로, brent·golden의 status 3·4, toms748의 유한하지 않은 값은 status −3(scipy와 다름), 구간의 끝과 배치 입력의 규칙 (O39–O49).
- O3의 결정(사용자, 2026-11-20): 기본 방법은 scipy의 것, `bounds`는 `(lower, upper)` 하나, 경계 밖의 `x0`는 `ValueError`, 반복 중의 유한하지 않은 값은 scipy의 처리(`x0`에서는 `ValueError`), `callback` 없음, L-BFGS-B의 `hess_inv`는 처음 읽을 때 만드는 (n, n) 행렬, `final_simplex`는 scipy처럼, 선 탐색 상한은 `max_ls` (O50–O57). 구현 뒤의 결정(2026-11-20): 고정 변수는 scipy처럼(뺀 변수가 있으면 `hess_inv` 필드 없음), ABNORMAL의 `fun`은 `x`의 값(scipy와 다름), `hess_inv`는 scipy의 순서, BFGS의 n³ 갱신은 O6에서 다시, 옵션 검사는 그대로 (O58–O62).
- O4의 결정(사용자, 2026-11-20): `hess`는 (n, n), `hessp`는 (n,)을 돌려준다(`LinearOperator`·희소 행렬·`HessianUpdateStrategy`는 받지 않는다), `hess='2-point'`·`'3-point'`를 받는다(`'cs'` 없음), 세 방법은 `jac`이 꼭 있다, `bounds`는 `ValueError` (O63–O66). 구현 뒤의 결정(2026-11-20): trust-ncg의 CG는 20n번까지(status 3), `nhev`는 실제 호출만(자리 채움 1 없음, trust-ncg 결과의 `hess` 호출 포함), newton-cg의 `jac`은 x의 기울기, 반복 중의 유한하지 않은 기울기·헤시안은 점을 적은 `ValueError`, trust-exact 부분문제의 상한은 status 3, `subproblem_max_iter`, newton-cg의 `hess`+`hessp`는 `ValueError`, 신뢰 영역의 `max_iter=0`은 반복 없음, trust-exact의 차분 `hess`는 `ValueError` (O67–O76).
- O5의 결정(사용자, 2026-11-20): `nnls`는 `OptimizeResult`(상한에서 예외 없이 `converged=False`), `root`는 'hybr'·'lm'만, `fjac`·`r`·`qtf`는 없음, `root`의 기본 허용오차는 scipy의 것, `lsq_linear`에 lsmr과 `unbounded_sol`은 없음 (O77–O81). 구현 뒤의 결정(2026-11-20): `root`에 `n_iter`, 차분에도 `njev`, 정방 계만, hybr 시험점의 유한하지 않은 F는 실패한 걸음(랭크 1 갱신 건너뜀), scipy의 글 그대로, lm의 비트 동일은 O6(`root('lm')`과 `least_squares('lm')`은 끝자리가 다를 수 있다), 옵션 검사는 `ValueError`, `nnls`의 status 1·3, `nnls`에 `method`·`rnorm` 없음, 유한하지 않거나 빈 입력은 `ValueError`, trf의 backtracking은 scipy대로, `active_mask`는 int64 (O82–O93).
- O6의 결정(사용자, 2026-11-20): 비트와 속도가 부딪히면 비트를 지키고 비트를 바꾸는 큰 개선은 재서 하나씩 묻는다, `least_squares('lm')`·`root('lm')`은 scipy의 빌드(FMA)로 비트 동일하게 하고 nls는 지금의 빌드(한 소스의 두 템플릿 인스턴스), 성능 뒤에 영역 전체의 코드 리뷰, 최적화는 알려진 것과 모든 함수의 scipy 대조 (O94–O97). 1부 뒤의 결정(2026-11-20): 골든의 'lm' 경우를 다시 기록, BFGS는 O(n²) 갱신(O9의 예외: 걸음이 반올림만큼 다름; n = 1은 scipy의 곱의 꼴), 'trf'의 SVD는 그대로, bench/speed 사례는 벤치 세션 (O98–O102).
- 기존 영역의 결과는 비트까지 같아야 한다(nls를 옮겨도).

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| O0 준비: API 문서, 결과 객체, 시험 문제(MGH 35), 대조 도구, 기준선 | 완료·승인 (2026-11-20) |
| O1 수치 미분과 최소제곱: `approx_fprime check_grad least_squares`, nls 옮기기 | 완료·승인 (2026-11-20) |
| O2 스칼라: `root_scalar minimize_scalar`, 배치 `find_root bracket_root find_minimum bracket_minimum` | 완료·승인 (2026-11-20) |
| O3 최소화 1: Nelder–Mead, BFGS, L-BFGS-B | 완료·승인 (2026-11-20) |
| O4 최소화 2: Newton-CG, trust-ncg, trust-exact | 완료·승인 (2026-11-20) |
| O5 방정식계와 선형 최소제곱: `root`(hybr, lm), `lsq_linear`, `nnls` | 완료·승인 (2026-11-20) |
| O6 최적화와 일관성 점검(BFGS의 n² 갱신, O61; `root('lm')`·`least_squares('lm')`의 scipy 비트 동일, O87) | 1부(성능) 완료·승인 (2026-11-20; BFGS O(n²) 갱신 포함); 2부 코드 리뷰(O96) 남음 |

## 마일스톤

### O0. 준비
- API 초안, 결과 객체, 시험 문제 모음, 참값·대조 도구, Fortran 원본 대조, scipy 기준선.
- 결정(사용자, 2026-11-20): docs/design/optimize.md §1.1 O9–O13, API 초안 뒤에 O14–O17.
- 결과: docs/api/optimize.md(공개 함수 13개의 시그니처 초안과 마일스톤별 질문 §9), src/stardust/optimize/(`__init__.py`, `_result.py`의 `OptimizeResult`), `sd.optimize`; 시험 문제 tests/optimize/(`mgh.py` MGH 35개·110경우, `systems.py` MINPACK 방정식계 14개·55경우, `functions.py` scipy 시험 함수·스칼라 근 204·최솟값 57·선형 최소제곱 12, `ad.py` IEEE double·Arb·2차 자동 미분); 도구 tools/gen_optimize_scipy.py(scipy 1.18.1 3,781번), tools/gen_optimize_fortran.py(MINPACK·L-BFGS-B 3.0 원본 빌드·실행·scipy와 비교), tools/optimize_ref.py(Arb 1 ULP 근); 고정 데이터 tests/optimize/data/(1.8 MB); 기준선(커밋 4151d2a) bench/optimize/o0/(speed.py, speed.md: scipy의 시간과 nfev, 10스레드가 빨라지는 곳은 BFGS n = 1000뿐); 테스트 tests/optimize/(test_result.py, test_api_surface.py, test_problems.py, test_reference_data.py). 결과는 docs/design/optimize.md §7.
- 찾은 것: MINPACK 드라이버의 문제는 Fortran과 비트까지 같다(216점). scipy의 C 판 MINPACK·L-BFGS-B는 첫 시험점부터 끝자리가 원본과 달라(계산점의 순서까지 같은 경우는 LMDER1 2/53, HYBRJ1 4/55, L-BFGS-B 7/83) 특이·비유일 해의 문제에서 다른 점에서 멈춘다. scipy `_tstutils`의 이계도함수 셋(aps02·aps09·aps10)이 틀렸다.

### O1. 수치 미분과 최소제곱
- `approx_fprime`, `check_grad`, `least_squares`('trf'·'lm', 해석적·'2-point'·'3-point' 야코비안, `diff_step`, 경계); nls의 LM·TRF 커널과 정확한 야코비안을 sd.optimize로 옮기고 nls가 그것을 쓴다.
- 결정(사용자, 2026-11-20): docs/design/optimize.md §1.1 O18–O23, 구현 뒤에 O24–O31.
- 결과: csrc/optimize/(lsq.cpp, lsq.hpp, diff.hpp; csrc/regression/nls.cpp에서 옮김), csrc/bind/optimize_bind.cpp(`stardust._core.optimize`), src/stardust/optimize/(`_lsq.py`, `_diff.py`, `_common.py`), src/stardust/regression/_nls.py(커널과 `Differences`를 씀); docs/api/optimize.md §2–§3, docs/design/optimize.md §3.1·§3.4·§7; 측정(커밋 4151d2a) bench/optimize/o1/(speed.py, speed.md); 테스트 tests/optimize/(test_least_squares.py, test_diff.py, test_api_surface.py).
- 결과 수치: nls는 옮기기 전후 골든 출력 338적합(NIST 27 × 시작점 2 × 두 방법 × 야코비안 둘, 상자 경계, 가중치 등)의 모든 필드가 바이트까지 같다. MGH 110경우에서 scipy 1.18.1과 같은 status는 trf 해석적 110, lm 해석적 108, trf 2점 106, lm 2점 109(상자 47/47), 해석적 'trf'는 103경우에서 모든 호출점이 10⁻¹⁰ 안. NIST 54적합의 x가 scipy와 10⁻⁶ 안인 것 50–54. `approx_fprime(epsilon=2**-26)`은 scipy와 비트까지 같고 기본(5점)은 해석적 도함수와 1.2×10⁻¹¹ 안. 반복당 자기 비용은 작은 문제에서 scipy의 1/4–1/20, (10⁴, 100) 'lm' 6.5 ms 대 42 ms.
- 찾은 것: 갈리는 경우는 모두 반올림이 결정하는 곳(거의 특이한 야코비안, 정확한 0을 보는 판정, 반올림 수준의 감소에서의 ftol·xtol)이다. 3점 차분은 O23의 기본 `max_nfev`에서 scipy보다 반복을 절반쯤만 할 수 있어 MGH 4경우가 상한에 닿았다: O25(100n(2n + 1))로 고친 뒤 넷 모두 scipy와 같은 곳에서 멈춘다. m = n = 40에서는 반복당 비용이 scipy와 비슷하다(QR을 먼저 하는 것이 덜어 주는 것이 없다; O6의 후보).

### O2. 스칼라
- `root_scalar`(brentq, toms748, bisect, newton, halley, secant), `minimize_scalar`(brent, golden, bounded), 배치 `find_root`, `bracket_root`, `find_minimum`, `bracket_minimum`(scipy `elementwise`, Chandrupatla).
- 결정(사용자, 2026-11-20): docs/design/optimize.md §1.1 O32–O38, 구현 뒤에 O39–O49.
- 결과: csrc/optimize/(scalar.hpp, scalar.cpp, batch.cpp), csrc/bind/optimize_bind.cpp, src/stardust/optimize/(`_scalar.py`, `_elementwise.py`, `_common.py`); docs/api/optimize.md §4–§6·§9, docs/design/optimize.md §3.2·§3.3·§7; 측정(커밋 4151d2a) bench/optimize/o2/(speed.py, speed.md); 테스트 tests/optimize/(test_scalar.py, test_elementwise.py, test_api_surface.py).
- 결과 수치: 시험 문제(근 204, 최솟값 57)의 모든 실행에서 scipy 1.18.1과 f를 부른 점이 차례대로 비트까지 같다(`root_scalar` 1,244번, `minimize_scalar` 399번, 배치의 원소 하나 621번, 2,000원소 배치 7가지; 기록과 다른 것은 newton·secant의 `message` 62번). 기본 허용오차의 구간법 근은 모두 Arb 근에서 xtol + rtol·|x| 안, `xtol=0`이면 brentq 3, toms748 1, bisect 4 ULP 안. 스칼라 방법의 자기 비용은 호출 하나에 0.2–1.5 µs(scipy C 구현의 1/1.5–1/2.8, Python 구현의 1/2–1/38), 배치는 함수값 하나에 6–12 ns(10⁶ 원소 1스레드에서 find_root 143 ms 대 scipy 330 ms, find_minimum 250 ms 대 649 ms).
- 찾은 것: scipy의 C brentq는 FMA로 빌드되어 있어(clang 기본) 외삽의 분자를 `std::fma`로 적어야 같은 걸음이다(아니면 204문제 가운데 12문제의 호출점이 갈린다). 남은 원소만 받는 배치 함수는 `args` 없이는 원소마다 다른 상수를 쓸 수 없어 배치 함수에만 `args`를 두었다(O39).

### O3. 최소화 1
- `minimize`: Nelder–Mead, BFGS, L-BFGS-B(해석적·`jac=True`·'2-point'·'3-point'·절대 간격 차분, 경계, 고정 변수). Newton-CG·trust-ncg·trust-exact는 O4.
- 결정(사용자, 2026-11-20): docs/design/optimize.md §1.1 O50–O57. 구현 뒤에 O58–O62.
- 결과: csrc/optimize/(minimize.hpp, objective.hpp, nelder_mead.cpp, bfgs.cpp, lbfgsb.cpp), csrc/bind/optimize_bind.cpp, CMakeLists.txt, src/stardust/optimize/(`_minimize.py`, `_result.py`의 처음 읽을 때 계산하는 필드, `_common.py`, `__init__.py`); docs/api/optimize.md §1·§7·§9, docs/design/optimize.md §1.1·§3.5·§6·§7·§8·§9; 측정(커밋 4151d2a) bench/optimize/o3/(speed.py, speed.md); 테스트 tests/optimize/(test_minimize.py, test_api_surface.py).
- 결과 수치: 세 방법 모두 scipy 1.18.1과 호출점이 비트까지 같다: 기록된 실행 568번의 모든 필드, 실행하며 견준 MGH·시험 함수 721번(상자 경계 153번 포함)과 옵션·가장자리 328번의 모든 호출점. BFGS의 `hess_inv`는 비트까지, L-BFGS-B의 것은 scipy `todense()`와 2.8×10⁻¹⁴ 안. L-BFGS-B Fortran 원본 83번은 scipy의 기록과 같아 원본과의 거리도 scipy와 같다(호출 수 74, x 10⁻¹² 안 62, 10⁻⁶ 안 79). 반복당 자기 비용은 작은 n에서 scipy의 1/2–1/20, 차분 기울기는 1/3–1/12, Nelder–Mead n = 1000은 1/5, BFGS n = 1000(같은 dgemm)과 L-BFGS-B n = 10⁴(같은 C 알고리즘)는 비슷하다.
- 찾은 것: scipy의 C 판 L-BFGS-B는 FMA 축약으로 빌드되어 있고, Accelerate의 ddot은 둘째 인자의 16바이트 정렬에 따라 끝자리가 다르다: 둘 다 맞춰야 비트까지 같다. scipy의 L-BFGS-B는 ABNORMAL에서 `fun`이 되돌린 `x`의 값이 아니고(여기서는 `x`의 값, O59), `hess_inv`는 저장 순서의 쌍을 쓴다(그대로 따름, O60).

### O4. 최소화 2
- `minimize`: Newton-CG(`hess`, `hessp`, `hess='2-point'`·`'3-point'`, 둘 다 없으면 기울기의 차분), trust-ncg(`hess`, `hessp`, 차분 `hess`), trust-exact(`hess`); `jac`은 호출 가능한 객체나 `True`.
- 결정(사용자, 2026-11-20): docs/design/optimize.md §1.1 O63–O66. 구현 뒤에 O67–O76(다섯은 scipy와 다르게: O68–O70, O74, O75).
- 결과: csrc/optimize/(newton_cg.cpp, trust.cpp, hessian.hpp, wolfe.hpp(bfgs.cpp의 선 탐색을 옮겨 같이 씀), objective.hpp의 점 셈과 기억을 거치지 않는 기울기, minimize.hpp), csrc/bind/optimize_bind.cpp, CMakeLists.txt, src/stardust/optimize/(`_minimize.py`, `_common.py`); docs/api/optimize.md §7·§9, docs/design/optimize.md §1.1·§2.2·§3.5·§6·§7·§8·§9; 측정(커밋 4151d2a) bench/optimize/o4/(speed.py, speed.md); 테스트 tests/optimize/(test_minimize_newton.py, test_minimize.py, test_api_surface.py).
- 결과 수치: 세 방법 모두 scipy 1.18.1과 호출점이 비트까지 같다(결정으로 다른 곳만 빼고: newton-cg 끝의 기울기 호출, `nhev`, 신뢰 영역의 `max_iter=0`): 기록된 실행 339번의 필드, 실행하며 견준 MGH·시험 함수 1,130번(`hess`·`hessp`·차분·기울기 차분의 열 묶음; 결과의 `hess`·`jac`까지)과 옵션·가장자리 525번의 모든 호출점, 무작위 비볼록 사차 함수 2,100번. 반복당 자기 비용은 n = 2–100에서 scipy의 1/3–1/60, 차분 헤세 곱은 n = 1000에서도 1/9, 행렬 `hess`의 n = 1000은 같은 dgemv라 비슷하다(Newton-CG 0.23 대 0.22 ms, trust-ncg 0.17 대 0.13 ms), trust-exact n = 1000은 5.2 대 7.2 ms.
- 찾은 것: numpy의 `np.dot`은 0.0 + cblas_ddot이라 BLAS의 −0.0을 +0.0으로 바꾼다(trust-ncg의 경계 교점이 그 부호를 쓴다: 맞추기 전에는 MGH 8경우가 끝자리에서 갈렸다). scipy의 trust-ncg CG는 상한이 없어 비대칭 헤시안에서 끝나지 않는다(여기서는 20n번에서 status 3). newton-cg의 결과 `jac`은 마지막 반복을 시작한 점의 것, `hessp`만 준 trust-ncg의 `nhev`는 부르지 않은 자리 채움 1을 더한 수, trust의 `max_iter=0`은 반복 하나, 작은 `subproblem_maxiter`의 trust-exact는 `UnboundLocalError`다.

### O5. 방정식계와 선형 최소제곱
- `root`('hybr': MINPACK hybrj·hybrd, 'lm': lmder·lmdif; 해석적·`jac=True`·차분 야코비안, `band`·`eps`·`factor`·`diag`), `lsq_linear`('trf', 'bvls'), `nnls`.
- 결정(사용자, 2026-11-20): docs/design/optimize.md §1.1 O77–O81. 구현 뒤에 O82–O93.
- 결과: csrc/optimize/(hybrd.cpp, root.hpp, linear.cpp, linear.hpp, lsq.cpp의 lmder 옵션(factor, diag, epsfcn, lmder의 info, m = n의 바로 QR)과 `lm_solve`), csrc/bind/optimize_bind.cpp, CMakeLists.txt, src/stardust/optimize/(`_root.py`, `_linear.py`, `__init__.py`); docs/api/optimize.md §1·§3·§8·§9, docs/design/optimize.md §1.1·§3.4·§3.6·§6·§7·§8·§9; 측정(커밋 4151d2a) bench/optimize/o5/(speed.py, speed.md); 테스트 tests/optimize/(test_root.py, test_lsq_linear.py, test_api_surface.py).
- 결과 수치: 'hybr'은 MINPACK 계 55경우 × 야코비안 유무 110번 모두 scipy 1.18.1과 호출점이 비트까지 같고 기록된 필드(x의 비트, F, status, message, njev, nfev − 2)가 같다(옵션 42번, `jac=True`도). Fortran HYBRJ1·HYBRD1 110번은 scipy `fsolve`의 기록과 x의 비트까지 같아 원본과의 거리도 scipy와 같다(HYBRJ1 같은 info 50/55, x 10⁻⁶ 안 44; HYBRD1 48, 44). 'lm'은 같은 status 54/55(해석적·차분), x 10⁻⁶ 안 49·45. `lsq_linear`('trf'·'bvls')는 기록된 12문제와 무작위 824번씩, 작은 모양 480번, (10⁴, 100)까지 모든 필드가 비트까지 같고, `nnls`는 무작위 801번(반복 상한 포함)에서 x가 비트까지 같다. nls 골든 출력 338적합은 바이트까지 같다. 반복당 자기 비용은 `root`가 scipy와 비슷하고(scipy도 반복이 C; lm 작은 n은 1/2), `lsq_linear` (100, 10) 'trf'는 1/9, 큰 문제는 같은 LAPACK이라 같다.
- 찾은 것: scipy의 C 판 MINPACK은 FMA 축약으로 빌드되어 있고, 제곱(`t**2`)은 따로 반올림하며, qrfac의 내적만 clang이 벡터화해 8개씩의 곱을 따로 반올림한다: 셋 다 맞춰야 비트까지 같다(이쪽 clang은 다른 합 고리도 같은 식으로 벡터화해, 합을 `std::fma`로 적었다). Accelerate의 gelsd는 numpy의 버퍼 배치(정렬)에 따라 끝자리가 달라 그 배치를 따랐다. scipy `nnls`의 C 판은 기계어에서 읽어 옮겼다(dlarfgp·dlarf·dlartgp; 떨어진 열의 꼬리를 바꾼 채 두고, P가 비면 음수 개수로 계속 가는 버그가 있다). `trf_linear`의 `backtracking`은 x를 바꾸지 않는다. scipy의 `root`는 x0에서 `fun`을 두 번 더 부르고 `nfev`에 센다.

### O6. 최적화와 일관성 점검 (1부: 성능)
- 결정(사용자, 2026-11-20): docs/design/optimize.md §1.1 O94–O97, 1부 뒤에 O98–O102.
- 결과: csrc/optimize/(minpack.hpp: scipy MINPACK의 enorm·dot8·열 여덟 개의 반사를 hybrd.cpp와 같이 씀; lsq.cpp: lmder를 템플릿 `Fu`의 두 빌드로, scipy 빌드는 m × n의 피벗 QR·열 묶음·스레드, 야코비안 읽기; hybrd.cpp: qrfac·qform의 열 묶음; batch.cpp: `args` 모으기와 원소 패스의 스레드; hessian.hpp·trust.cpp: C 순서의 사용자 헤시안을 그대로 쥠; bfgs.cpp: H의 O(n²) 갱신(O99; n = 1은 곱의 꼴, O102); linear.cpp: nnls의 남는 계산, 유한성 검사를 커널에서), src/stardust/optimize/(`_lsq.py`: 'lm'은 scipy 빌드, `_linear.py`); docs/api/optimize.md §3·§8, docs/design/optimize.md §1.1·§3.3.1·§3.4·§3.5·§3.6·§7·§8·§9; 측정(커밋 4151d2a) bench/optimize/o6/(speed.py: O1–O5의 사례를 그대로 돌리는 전체 대조, speed.md, speed_before.json, speed_after.json); 테스트 tests/optimize/(test_root.py, test_least_squares.py, test_elementwise.py, test_minimize.py: BFGS는 허용오차로 견줌); 골든 tests/golden/data/optimize.json(O98: 'lm' 175경우를 다시 기록).
- 결과 수치: `root('lm')`은 55계 × 야코비안 유무 110번 모두 scipy와 호출점과 결과가 비트까지 같다(옵션 36번도). `least_squares('lm')`은 MGH 110경우 가운데 해석적 108, 2점 104경우가 비트까지 같다(scipy가 기억으로 건너뛰는 같은 x의 호출과 O24의 끝 호출을 빼고; 나머지는 G74의 유한하지 않은 시험점과 O25의 `max_nfev`). NIST 54적합 52·52. nls 골든 338적합은 바이트까지 같다. 고정 골든은 optimize의 'lm' 175경우만 바뀌어 다시 기록했다(O98). BFGS는 O(n²) 갱신으로 n = 1000의 반복당 8.8 → 0.28 ms, scipy와 status가 같은 것 113/114(기록된 113번에서 해석적 112, 차분 107). 속도는 커밋 4151d2a의 bench/optimize/o6/speed.md: hybr n = 300 자기 비용이 scipy의 0.55배, `root('lm')` n = 100 0.17 → 0.11 ms, m = n = 40의 'lm' 0.063 → 0.036 ms, 배치 10⁶ 10스레드 find_root 89 → 31 ms, newton-cg·trust-ncg `hess` n = 1000 0.23 → 0.15, 0.16 → 0.10 ms. O95로 m ≫ n의 'lm'은 느려졌다((10⁴, 100) 6.6 → 17 ms; scipy 37 ms).
- 비트를 바꾸는 후보(O94로 재서 물음): BFGS O(n²) 갱신은 넣었고(O99), 경계 없는 'trf'의 n × n SVD(5%)는 넣지 않는다(O100).

---

# 가설검정 영역 (sd.htest)

> 상태: **H0 완료·승인, H1 완료·승인** (2026-11-20). H2–H10은 시작 전이다. 설계는 [docs/design/htest.md](docs/design/htest.md), API의 초안은 [docs/api/htest.md](docs/api/htest.md)다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

## 진행 원칙
- 다른 영역과 같다: 마일스톤마다 시작 전에 물을 것(docs/design/htest.md §9)을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.
- 큰 결정(사용자, 2026-11-20): 새 최상위 모듈 `sd.htest`(`sd.stats.tests`가 아니다, 상관도 여기), 첫 판에 scipy.stats의 가설검정 전부(모수, 순위, 적합도, 분할표, 상관, 재표본과 그 밖), scipy의 이름과 stardust의 규칙(위치 입력과 키워드 옵션, 별칭 없음, `nan_policy`, sd.stats의 `axis`·배치, `rng=`, 읽기 전용 결과 객체), 통계량과 p값은 Arb·정확한 유리수 참값에 대한 상대 1e-14 계약(scipy·R과 비트 동일은 목표가 아니다), statsmodels의 다중 비교 보정(`multipletests` 등)과 비율·이항 검정(`proportions_ztest`, `proportion_confint`, 두 독립 비율, `mcnemar`, `cochrans_q`); TOST와 검정력 분석은 뺀다 (docs/design/htest.md §1.1 H1–H5).
- H0의 결정(사용자, 2026-11-20): 검정마다 읽기 전용 결과 클래스 `<함수>Result`(공통 바탕, 내보내지 않음), scipy가 푸는 곳만 같은 개수로 풀기, 신뢰구간은 ~~scipy의 `confidence_interval(confidence_level=0.95)`~~ regression의 `conf_int(alpha=0.05)`(H16), `axis=None` 기본과 표의 앞 축 배치, 자유도는 어디서나 `df`(`chisquare`의 `ddof`는 그대로), 퇴화한 입력은 경고 없는 NaN, `'auto'`가 쓴 방법을 `method` 필드에, `'exact'`의 상한을 넘으면 `ValueError`, 새 귀무분포는 htest 안에만, R 꾸러미는 마일스톤마다 묻는다 ; 영역 사이(H16–H21): 신뢰구간은 regression의 `conf_int(*, alpha=0.05)`, 두 영역이 함께 계산하는 값의 커널은 htest(Jarque–Bera는 H2에서 옮기고 p값은 `ChiSquared(df=2).ccdf`), 같은 검정은 같은 형(`sd.regression.JarqueBeraResult`가 htest의 클래스), F 꼴의 자유도는 `df_num`·`df_denom`, 분산 0은 sd.stats `skew`의 규칙 하나, 배열 필드는 복사 없는 쓰기 금지 (docs/design/htest.md §1.1 H6–H21).
- 비교 대상은 scipy 1.18.1, statsmodels 0.15.0, R 4.6.2이고, 정확도 체계(`bench/accuracy`)와 속도 체계(`bench/speed`)에서 견준다.
- 기존 영역의 결과는 비트까지 같아야 한다(regression의 `jarque_bera()` 진단 등 같은 커널을 쓰게 되는 곳도).

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| H0 준비: API 문서, 결과 객체, 공통 입력 층, 참값(Arb·정확한 유리수)·R 대조 도구, 기준선 | 완료·승인 (2026-11-20) |
| H1 평균과 분산: `ttest_1samp ttest_rel ttest_ind ttest_ind_from_stats f_oneway alexandergovern bartlett levene fligner` | 완료·승인 (2026-11-20) |
| H2 적합도 1: `chisquare power_divergence skewtest kurtosistest normaltest jarque_bera shapiro anderson ks_1samp kstest cramervonmises` | 시작 전 |
| H3 순위 검정: `mannwhitneyu ranksums wilcoxon kruskal friedmanchisquare brunnermunzel ansari mood median_test page_trend_test` | 시작 전 |
| H4 적합도 2: `ks_2samp cramervonmises_2samp anderson_ksamp epps_singleton_2samp` | 시작 전 |
| H5 상관: `pearsonr spearmanrho spearmanr pointbiserialr kendalltau weightedtau somersd chatterjeexi` | 시작 전 |
| H6 분할표: `chi2_contingency fisher_exact barnard_exact boschloo_exact mcnemar cochrans_q` | 시작 전 |
| H7 비율과 이항: `binomtest quantile_test poisson_means_test proportions_ztest proportions_chisquare proportion_confint test_proportions_2indep confint_proportions_2indep` | 시작 전 |
| H8 다중 비교와 보정: `tukey_hsd dunnett multipletests false_discovery_control combine_pvalues` | 시작 전 |
| H9 재표본: `permutation_test monte_carlo_test bootstrap`, 재표본 `method=` 객체, `bws_test` | 시작 전 |
| H10 최적화와 일관성 점검 | 시작 전 |

## 마일스톤

### H0. 준비
- API 초안, 결과 객체의 틀, 공통 입력 층, 참값 도구, scipy·statsmodels·R 대조 데이터, 속도 기준선, 정확도·속도 체계에 넣는 계획.
- 결정(사용자, 2026-11-20): docs/design/htest.md §1.1 H6–H15, 영역 사이의 결정 H16–H21.
- 결과: docs/api/htest.md(계획한 공개 이름 72개의 시그니처, 결과 필드와 풀기, 에러, scipy·statsmodels → sd 이름의 대응표, 마일스톤별 질문 §12), src/stardust/htest/(`__init__.py` — `sd.htest`, 공개 이름은 아직 없다; `_result.py`의 `_TestResult`와 `conf_int`; `_common.py`의 `Samples`·`Tables`와 옵션 검사), tests/htest/(`problems.py` 문제 281개, test_api_surface.py, test_result.py, test_common.py, test_reference_data.py; 63개), tools/htest_ref.py(Arb·정확한 유리수, 17묶음 138문제 → tests/htest/ref/), tools/gen_htest_scipy.py(→ tests/htest/data/scipy_h1–h8.json, statsmodels.json), tools/gen_htest_r.py(base R 145문제 → tests/htest/data/r.json), bench/htest/h0/(speed.py, speed.md, speed.json: scipy·statsmodels 41검정 × n = 10, 10³, 10⁶, 배치 10⁴ × 30), docs/design/htest.md §2.2·§7(정확도·속도 체계에 넣는 파일 목록).
- 찾은 것: scipy·R이 참값에서 크게 벗어나는 곳은 상쇄다: Bartlett의 거의 같은 분산(scipy 53배, R은 음수의 통계량), G 검정의 O ≈ E(scipy 1e-7), |r| ≈ 1의 Pearson p값(scipy 5e-7–0.11, R 2e-7–0.09), 큰 오프셋의 t(scipy·R 1e-8–3e-7; R은 짝 t 1e-6, F 2e-8). R은 깊은 꼬리에서 p = 0을 낸다(한 표본 KS 정확 n = 80의 참 1.7e-21, wilcox.test 정규 근사의 참 6.1e-51). 그 밖의 scipy 값은 참값과 1e-12 안이고, 정수 통계량의 정확 분포는 3e-16 안이다. scipy의 2 × 2 `fisher_exact`는 양쪽 규칙의 허용오차가 1e-14다(설계 초안의 1e-7은 `binomtest`와 R의 것). scipy `wilcoxon`의 `zstatistic`은 `method='asymptotic'`을 준 때만 있다. 작은 표본에서 scipy 호출 하나의 고정 비용은 0.06–0.25 ms, 배치에서 줄마다 Python 고리인 `kendalltau`는 10⁴ × 30에 950 ms다.
- sd.regression과 겹치는 곳(`jarque_bera`와 `r.jarque_bera()`, `f_oneway`와 분산분석표·더미 OLS, `pearsonr`과 `linregress`, `multipletests`와 단계적 선택): H16–H21로 정했고 docs/design/htest.md §1.4, docs/api/htest.md §13에 적었다(Jarque–Bera의 커널과 클래스는 H2에서 옮긴다; `anova`와 `f_oneway`는 계약 안에서 같고 비트는 아니다).

### H1. 평균과 분산
- t 검정 넷(Student, Welch, Yuen의 절삭), 일원 분산분석(Welch 포함), Alexander–Govern, 분산의 동질성(Bartlett, Levene, Fligner–Killeen). 신뢰구간은 `conf_int(alpha=0.05)`.
- 결정(사용자, 2026-11-20): `df`는 언제나 float, `ttest_ind_from_stats`에 `df`와 `conf_int()`, 표준오차 0의 t·p는 NaN, 실수 스칼라는 원소 하나의 표본, Welch 분산분석은 `f_oneway(equal_var=False)`, `f_oneway`에 `df_num`·`df_denom` (docs/design/htest.md §1.1 H22–H27). 결과를 보고(사용자, 2026-11-20): 분산 0인 집단의 NaN을 다표본 검정에도, `levene`에 `df_num`·`df_denom`과 `bartlett`·`fligner`·`alexandergovern`에 `df`(풀기는 scipy 그대로), 표본이 둘 미만이면 다섯 모두 `TypeError`, Welch의 작은 표본은 `df` NaN, 속도 숙제는 H10으로, Levene의 R 대조는 `car::leveneTest` (H28–H33).
- 결과: src/stardust/htest/_parametric.py(아홉 함수와 결과 클래스), csrc/htest/(moments.hpp: 줄의 double-double 적률과 고정 덩어리의 스레드풀 분할, 2의 거듭제곱 단위, 분산 0의 판정, 키 정렬; parametric.cpp: 검정, p값은 sd.stats 분포의 메서드), csrc/bind/htest_bind.cpp(`stardust._core.htest`), tests/htest/test_parametric.py(참값 67문제의 계약, 배치, 스레드, `nan_policy`, 퇴화, 단위, regression의 OLS와의 일치, 메시지), 참값 6묶음 67문제(tools/htest_ref.py; Yuen, `alexandergovern`, `levene`, `fligner`, 신뢰구간과 `kappa`를 더했다), 문제 37개 더(313개), bench/htest/h1/(speed.py, speed.md, speed.json).
- 찾은 것: 모든 참값이 계약 안이다(통계량 3.8e-16, p값은 max(1, κ_p)로 나눠 3.8e-16, 자유도 2.1e-16). scipy·R은 큰 오프셋(t·F·Welch·Alexander–Govern·Bartlett·Levene 2.6e-9–1e-7, 짝의 차 1.1e-9)과, 짝수 n의 중앙값을 반올림해 동점을 가르는 Fligner(p 6.4e-9–2.9e-7)에서 틀린다. statsmodels의 t·분산분석도 오프셋에서 6.5e-9–3.2e-7이다. 속도는 10스레드에서 모든 경우에 scipy보다 빠르다(n = 10에서 10–40배, 10⁶에서 1.0–6.4배, 배치 2.4–5.7배); 1스레드의 10⁶과 배치, `levene`의 긴 줄(중앙값 선택이 한 스레드)은 H10으로 넘겼다(H32). R `car::leveneTest`는 오프셋 10⁸에서 Levene의 W가 8.4e-9 틀린다 (docs/design/htest.md §7).

### H2. 적합도 1: 분포를 정한 검정
- 카이제곱과 Cressie–Read, D'Agostino의 셋, Jarque–Bera, Shapiro–Wilk(AS R94), Anderson–Darling(1.17부터의 꼴), Kolmogorov–Smirnov 한 표본(정확·근사·점근), Cramér–von Mises 한 표본.

### H3. 순위 검정
- Mann–Whitney(정확·정규), 순위 합, Wilcoxon 부호 순위(정확·정규·나열), Kruskal–Wallis, Friedman, Brunner–Munzel, Ansari–Bradley(정확·정규), Mood, Mood의 중앙값 검정, Page의 경향 검정.

### H4. 적합도 2: 표본끼리
- Kolmogorov–Smirnov 두 표본(격자 경로의 정확 분포), Cramér–von Mises 두 표본(정확·점근), Anderson–Darling k 표본, Epps–Singleton.

### H5. 상관
- Pearson(1 − r²을 상쇄 없이), Spearman(두 꼴), 점 이연 상관, Kendall τ-b·τ-c(정확·정규), 가중 τ, Somers' D, Chatterjee ξ.

### H6. 분할표
- 카이제곱 독립성 검정(Yates, Cressie–Read), Fisher 정확 검정(2 × 2와 r × c), Barnard·Boschloo(장애 모수의 참 상한), McNemar, Cochran Q.

### H7. 비율과 이항
- 이항 검정과 비율의 구간, 분위수 검정, 포아송 평균의 E-test, statsmodels의 비율 z·카이제곱 검정, 비율의 신뢰구간(모든 방법), 두 독립 비율의 차·비·오즈비 검정과 구간. 비교 대상에 R `prop.test`·`binom.test`.

### H8. 다중 비교와 보정
- Tukey HSD(스튜던트화 범위 분포), Dunnett(다변량 t), `multipletests`(열 방법), `false_discovery_control`, `combine_pvalues`. 비교 대상에 R `p.adjust`·`ptukey`.

### H9. 재표본
- `permutation_test`, `monte_carlo_test`, `bootstrap`, `PermutationMethod`·`MonteCarloMethod`·`BootstrapMethod`와 앞 검정들의 `method=`, Baumgartner–Weiss–Schindler(그리고 §9의 질문에 따라 `multiscale_graphcorr`, `goodness_of_fit`).

### H10. 최적화와 일관성 점검
- 모든 공개 함수를 크기마다 scipy·statsmodels·R과 견주어 느린 곳을 고치고, 메시지·시그니처의 영역 간 일관성을 점검하고, 코드 리뷰를 한다.
- H1에서 넘어온 것(H32): 1스레드의 double-double 적률 훑기를 NEON으로(10⁶에서 scipy 1.6 ms 대 5.0 ms), `levene`의 집단마다 중앙값 선택을 병렬로(10⁶ 23 ms, scipy 25 ms), `fligner` 정렬의 병합 단계를 병렬로.

# 희소 영역 (sd.sparse)

> 상태: **설계 초안** (2026-11-20). SP0–SP8은 모두 시작 전이다. 설계는 [docs/design/sparse.md](docs/design/sparse.md)다. API의 초안(`docs/api/sparse.md`)은 SP0에서 쓴다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

## 진행 원칙
- 다른 영역과 같다: 마일스톤마다 시작 전에 물을 것(docs/design/sparse.md §9)을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.
- 큰 결정(사용자, 2026-11-20): 첫 판은 저장 형식(COO, CSR, CSC와 만들기용 형식)과 연산, 그리고 희소 선형대수(직접 풀이, 반복법, 고유값·특이값, 행렬 함수, `LinearOperator`); scipy의 희소 **배열** 꼴만(`*`는 원소별, `@`는 행렬 곱, `spmatrix` 없음); 희소 선형대수도 하위 모듈 없이 `sd.sparse`에 scipy.sparse.linalg의 이름으로(`sd.linalg`는 밀집만, 희소를 받으면 짚는 `TypeError`); 직접 분해와 알맞은 반복법은 Accelerate의 Sparse Solvers, 형식·변환·산술·SpMV·SpMM은 이 라이브러리의 C++ 커널; 그래프 알고리즘은 나중의 따로 된 영역; optimize·regression의 나중 쓰임은 메모 (docs/design/sparse.md §1.1 SP1–SP5).
- 비교 대상은 scipy 1.18.1의 `scipy.sparse`·`scipy.sparse.linalg`와 R 4.6.2의 `Matrix` 1.7.6이고, 정확도 체계(`bench/accuracy`)와 속도 체계(`bench/speed`)에서 견준다. 참값은 Arb다.
- 설계 초안에서 찾은 것: Accelerate의 대칭 분해(Cholesky, LDLᵀ)와 LU의 기본 순서(MTMetis)는 기본 스레드 모드에서 실행마다 비트가 다르다. LU(AMD·Metis 순서)와 QR은 같다 (docs/design/sparse.md §1.4). 그래서 첫 판의 직접 풀이는 LU/AMD로 하는 안이고, 이것과 반복법을 이 라이브러리가 짜는 안은 SP0 전에 묻는다.
- 기존 영역의 결과는 비트까지 같아야 한다(`sd.linalg`·배열이 희소 입력에 내는 `TypeError`는 희소 입력에서만 동작한다).

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| SP0 준비: API 문서, 시험 행렬, 참값(Arb)·scipy·R 대조 도구, Accelerate 탐침, 색인 dtype의 측정, 기준선 | 시작 전 |
| SP1 형식과 변환: `coo_array csr_array csc_array dok_array`, 표준 꼴, `sd.ndarray`·numpy·scipy와 주고받기 | 시작 전 |
| SP2 연산: 산술·비교·원소별 함수, SpMV·SpMM·SpGEMM, 리덕션 | 시작 전 |
| SP3 인덱싱과 만들기: 인덱싱과 대입, `eye_array diags_array random_array kron kronsum hstack vstack block_array block_diag tril triu find` | 시작 전 |
| SP4 직접 풀이: `spsolve splu factorized inv spsolve_triangular`, `LinearOperator`, `sd.linalg`의 `TypeError` | 시작 전 |
| SP5 반복법: `cg gmres minres bicgstab lsqr lsmr` | 시작 전 |
| SP6 고유값과 특이값: `eigsh eigs svds` | 시작 전 |
| SP7 행렬 함수와 노름: `expm expm_multiply onenormest norm` | 시작 전 |
| SP8 최적화와 일관성 점검 | 시작 전 |

## 마일스톤

### SP0. 준비
- API 초안(scipy.sparse·scipy.sparse.linalg → sd.sparse 이름의 대응표 포함), 시험 행렬(격자, 대류-확산, 무작위, 특이, 복소 에르미트 등; 실제 행렬은 내려받아도 되는지 묻는다), Arb 참값 도구, scipy·R `Matrix` 대조 데이터, Accelerate 탐침(결정성, 특이 판정, int32 한계), int32·int64 색인의 SpMV 측정, 속도 기준선, 정확도·속도 체계에 넣는 계획.

### SP1. 형식과 변환
- COO·CSR·CSC·DOK, 언제나 표준 꼴인 CSR·CSC(정렬, 중복은 입력 순서대로 더함), 형식 변환, `sd.ndarray`·numpy·scipy.sparse와 주고받기(scipy로는 복사 없는 뷰), 전치·`astype`·`reshape` 등.

### SP2. 연산
- 산술과 비교(연산마다 scipy의 0 지우기 규칙, 암시적 0은 강한 0), 원소별 함수, SpMV·SpMM·밀집 @ 희소·SpGEMM(줄 합은 색인 오름차순의 FMA 순차 합: SpMV가 scipy와 비트까지 같다), 리덕션.

### SP3. 인덱싱과 만들기
- 정수·슬라이스·배열·마스크 인덱싱과 대입, 대각, 만들기 함수(`random_array`는 `rng`를 꼭 받는다).

### SP4. 직접 풀이
- Accelerate LU(AMD 순서)로 `spsolve`·`splu`·`factorized`·`inv`, 이 라이브러리의 삼각 풀이, 특이의 판정(`LinAlgError`), `LinearOperator`, `sd.linalg`가 희소 입력에 쓸 함수를 짚는 `TypeError`.

### SP5. 반복법
- CG, GMRES, MINRES, BiCGSTAB, LSQR, LSMR(scipy 1.18.1과 같은 알고리즘과 멈춤 판정, 결정적인 SpMV와 내적).

### SP6. 고유값과 특이값
- implicitly restarted Lanczos·Arnoldi(`eigsh`, `eigs`; shift-invert는 LU), Golub–Kahan–Lanczos 이중대각화(`svds`). 비교 대상에 R `RSpectra`·`irlba`가 필요하면 이때 묻는다.

### SP7. 행렬 함수와 노름
- `expm`(Al-Mohy–Higham 2009), `expm_multiply`(Al-Mohy–Higham 2011), `onenormest`(Higham–Tisseur, 결정적인 열), `norm`.

### SP8. 최적화와 일관성 점검
- 모든 공개 함수를 크기마다 scipy·R과 견주어 느린 곳을 고치고, 메시지·시그니처의 영역 간 일관성을 점검하고, 코드 리뷰를 한다.
