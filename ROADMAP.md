# stardust 로드맵

> 상태: **확정** (2026-11-12). 앞부분은 세 프로젝트의 통합(M0–M7, 완료), 뒷부분은 새 영역 [특수함수 (sd.special)](#특수함수-영역-sdspecial)(S0–S11, 완료), 그 뒤 [통계 (sd.stats, sd.random)](#통계-영역-sdstats-sdrandom)(T0–T10, 완료·승인; 검정은 뺐다), 그 뒤 [회귀 (sd.regression)](#회귀-영역-sdregression)(G0–G6 완료·승인, G7a 완료·승인, G7b 완료·승인), 그 뒤 [최적화 (sd.optimize)](#최적화-영역-sdoptimize)(O0 완료·승인, O1 완료·승인, O2 완료·승인, O3 완료·승인, O4 완료·승인, O5 완료·승인, O6 완료·승인), 그 뒤 [가설검정 (sd.htest)](#가설검정-영역-sdhtest)(H0 완료·승인, H1 완료·승인, H2 완료·승인, H3 완료·승인, H4 완료·승인, H5 완료·승인, H6 완료·승인(위임), H7 완료·승인(위임), H8 완료·승인(위임), H9 완료·승인(위임), H10 완료·승인(위임)), 그 뒤 [희소 (sd.sparse)](#희소-영역-sdsparse)(SP0 완료·승인, SP1 완료·승인, SP2 완료·승인, SP3 완료·승인, SP4 완료·승인(위임), SP5 완료·승인(위임), SP6 완료·승인(위임), SP7 완료·승인(위임), SP8 완료·승인(위임)), 그 뒤 [수치 미적분 (sd.integrate, sd.differentiate)](#수치-미적분-영역-sdintegrate-sddifferentiate)(C0 완료·승인(위임), C1 완료·승인(위임), C2 완료·승인(위임), C3 완료·승인(위임), C4 완료·승인(위임), C5 완료·승인(위임), C6 완료·승인(위임), C7 완료·승인(위임), C8 완료·승인(위임)), 그 뒤 [공간 (sd.spatial)](#공간-영역-sdspatial)(SPT0 완료·승인(위임), SPT1 완료·승인(위임), SPT2 완료·승인(위임), SPT3 완료·승인(위임), SPT4 완료·승인(위임), SPT6 완료·승인(위임), SPT7 완료·승인(위임)), 그리고 [그림 (sd.plot)](#그림-영역-sdplot)(PL0–PL9 완료·승인)이다.
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
| M9 희소의 밀집 짝 | 다듬기 | 완료·승인(위임) (2026-11-22) |

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

### M9. 희소의 밀집 짝 (2026-11-22 사용자 규칙)
- 규칙: sd.sparse의 기능은 밀집의 부분집합이거나 희소에만 필요한 것이다(API.md §7). 희소에 있고 밀집에 없던 것을 배열·선형대수에 더한다(추천안, 사용자 위임 2026-11-22).
- 배열: `diag`, `block`, `deg2rad`, `rad2deg`. `setdiag`의 짝은 함수가 아니라 대입(docs/design/array.md §1.6).
- 선형대수: `kronsum`, `block_diag`, `bandwidth`(`BandwidthResult`), `kron`의 1차원 입력, `lu_solve`의 `trans=`(docs/design/linalg.md §1.5).
- 확인: 희소 짝의 `toarray()`와 비트 비교, 골든(새 이름만), 속도 사례 `--smoke`.

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

> 상태: **G0–G6 구현·승인 완료** (2026-11-19, 최적화 승인 포함), **G7a 구현·승인** (2026-11-20), **G7b 구현·승인** (2026-11-21). 그 뒤의 코드 리뷰와 사이트 작업에서 찾은 문제는 결정 G83–G116으로 고쳤다(2026-11-20까지). 설계는 [docs/design/regression.md](docs/design/regression.md), API는 [docs/api/regression.md](docs/api/regression.md).
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
| G1 선형: `ols wls gls linregress`(`linregress`는 2026-11-21에 뺐다, G188), 추론·예측, 강건 표준오차, 진단, summary | 완료·승인 (2026-11-18) |
| G2 GLM: 정규·이항·포아송·감마, IRLS + 뉴턴, 분리 증명서, 강건 공분산 | 완료·승인 (2026-11-18) |
| G3 정규화: `ridge lasso elastic_net`, 경로, 교차검증, `penalized_glm` | 완료·승인 (2026-11-18) |
| G4 강건: `rlm`(M·MM), `theilslopes`, `siegelslopes` | 완료·승인 (2026-11-18) |
| G5 분위: `quantreg` | 완료·승인 (2026-11-19) |
| G6 비선형 최소제곱: `nls` | 완료·승인 (2026-11-19) |
| 회귀 최적화: Dot2 행 훑기, 64행 타일, 정준 링크 연마, expit 공유, MM의 S 뉴턴 | 완료·승인 (2026-11-19) |
| 코드 리뷰와 사이트 작업에서 찾은 문제 (P1–P4, R1–R112) | 고침 (2026-11-19–20, 결정 G83–G116) |
| G7a 비교·검정·진단: `t_test wald_test compare_f_test compare_lr_test anova vif dffits dfbetas wresid white_test goldfeld_quandt reset_test`, GLM의 영향력 | 완료·승인 (2026-11-20, 결정 G137–G164) |
| G7b 변수 선택: `stepwise best_subset`(AIC·BIC·p값·교차검증; `ols wls gls glm`) | 완료·승인 (2026-11-21, 결정 G166–G187) |

## 마일스톤

### G0. 준비
- 결과: statsmodels 0.15.0·scikit-learn 1.9.1을 test 의존성에 넣고 설치했다(Python 3.15에서 설치됨). `tools/regref.py`(Arb 참값), `tests/regression/data/`(NIST StRD 11개, itl.nist.gov), `tests/regression/nist.py`; 테스트 tests/regression/test_reference_tool.py. 기존 패키지의 NIST 성적은 docs/design/regression.md §6.
- 결과 객체의 공통 틀은 G1로 옮겼다(OLS의 필드와 함께 만든다).


### G1. 선형
- `ols wls gls linregress`(`linregress`와 `LinregressResult`는 2026-11-21에 뺐다: `ols(y, x)`와 `htest.pearsonr`과 같은 값이고 자기만의 옵션이 없다, docs/design/regression.md G188), 추론(표준오차·t·p·구간·F·우도), 예측과 예측 구간, 강건 표준오차(HC0–HC3, HAC, 군집), 랭크 부족, 진단(지렛대, 스튜던트화 잔차, Cook 거리, Breusch–Pagan, Jarque–Bera), `summary`, 배치
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
- 결과: csrc/regression/(linear.cpp, lsq.hpp: 적합의 얇은 Q를 내는 `thin_q` 방식), src/stardust/regression/(_tests.py, _diagnostics.py, _linear.py, _glm.py, _nls.py, __init__.py); docs/api/regression.md §9.1–§9.2, docs/design/regression.md §3.8; Arb 참값 tools/regression_ref.py model_tests(식은 tools/regref.py; tests/regression/refdata_model_tests.py → ref/model_tests.json), R 대조 tools/gen_model_tests_r.py(data/model_tests_r.json); 측정 bench/speed/cases/regression.py(묶음 model-tests, model-tests-batched)와 bench/accuracy(스윕 regression/model_tests/…); 테스트 tests/regression/test_model_tests.py, test_model_tests_against_r.py.

### G7b 변수 선택
- 결정(사용자, 2026-11-21): docs/design/regression.md §1.1 G166–G180(모형 함수의 키워드 그대로, 'both'는 전체 모형에서와 `start=`, AIC·BIC는 1e-7보다 크게 줄 때만 옮기고 동률은 앞 열, 기준 목록과 기본 'aic', p값은 부분 F·우도비와 0.05/0.10, BIC의 n은 `nobs`, 교차검증은 MSE·평균 편차와 겹마다 다시 고르기, GLM의 분리·미수렴은 기록만, 감마는 결과의 `aic`, NaN은 NaN 결과, 배치는 에러, 부분집합의 상한 50·15, `StepwiseResult`·`BestSubsetResult`, 보정 옵션 없음, `include`만). 구현하며 정한 세부 G181–G187(기준은 `cov_type`을 쓰지 않음, GLS의 교차검증은 에러, 모수가 없는 모형, 겹친 열의 부분집합은 후보 아님, 원래 열 이름, GLM 후보는 따뜻한 시작 없음, 작업량 상한 없음)도 정했다.
- 결과: csrc/regression/select.cpp(삼각 인자, 후보의 갱신, 교차검증, leaps and bounds), src/stardust/regression/(_select.py, __init__.py); docs/api/regression.md §9.3, docs/design/regression.md §3.8; Arb 참값 tools/regression_ref.py selection(식은 tools/regref.py; tests/regression/refdata_selection.py → ref/selection.json), R 대조 tools/gen_selection_r.py(data/selection_r.json); 측정 bench/speed/cases/regression.py(묶음 selection)와 bench/accuracy(스윕 regression/selection/…); 테스트 tests/regression/test_selection.py, test_selection_against_r.py.

---

# 최적화 영역 (sd.optimize)

> 상태: **O0 완료·승인**, **O1 완료·승인**, **O2 완료·승인**, **O3 완료·승인**, **O4 완료·승인**, **O5 완료·승인**, **O6 완료·승인** (2026-11-20; 2부 코드 리뷰는 사용자가 직접 함). 설계는 [docs/design/optimize.md](docs/design/optimize.md), API의 초안은 [docs/api/optimize.md](docs/api/optimize.md)다.
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
| O6 최적화와 일관성 점검(BFGS의 n² 갱신, O61; `root('lm')`·`least_squares('lm')`의 scipy 비트 동일, O87) | 완료·승인 (2026-11-20). 1부 성능(BFGS O(n²) 갱신 포함), 2부 코드 리뷰(O96)는 사용자가 직접 했다 |

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

> 상태: **H0 완료·승인, H1 완료·승인, H2 완료·승인, H3 완료·승인, H4 완료·승인, H5 완료·승인, H6 완료·승인(위임), H7 완료·승인(위임), H8 완료·승인(위임) (2026-11-21), H9 완료·승인(위임) (2026-11-22), H10 완료·승인(위임) (2026-11-22)**. 설계는 [docs/design/htest.md](docs/design/htest.md), API의 초안은 [docs/api/htest.md](docs/api/htest.md)다.
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
| H2 적합도 1: `chisquare power_divergence skewtest kurtosistest normaltest jarque_bera shapiro anderson ks_1samp cramervonmises` | 완료·승인 (2026-11-20) |
| H3 순위 검정: `mannwhitneyu ranksums wilcoxon kruskal friedmanchisquare brunnermunzel ansari mood median_test page_trend_test` | 완료·승인 (2026-11-21) |
| H4 적합도 2: `ks_2samp cramervonmises_2samp anderson_ksamp epps_singleton_2samp` | 완료·승인 (2026-11-21) |
| H5 상관: `pearsonr spearmanrho spearmanr pointbiserialr kendalltau weightedtau somersd chatterjeexi` | 완료·승인 (2026-11-21) |
| H6 분할표: `chi2_contingency fisher_exact barnard_exact boschloo_exact mcnemar cochrans_q odds_ratio relative_risk` | 완료·승인(위임) (2026-11-21) |
| H7 비율과 이항: `binomtest quantile_test poisson_means_test proportions_ztest proportions_chisquare proportion_confint test_proportions_2indep confint_proportions_2indep` | 완료·승인(위임) (2026-11-21) |
| H8 다중 비교와 보정: `tukey_hsd dunnett multipletests false_discovery_control combine_pvalues` | 완료·승인(위임) (2026-11-21) |
| H9 재표본: `permutation_test monte_carlo_test bootstrap`, 재표본 `method=` 객체, `bws_test multiscale_graphcorr goodness_of_fit` | 완료·승인(위임) (2026-11-22) |
| H10 최적화와 일관성 점검 | 완료·승인(위임) (2026-11-22) |

## 마일스톤

### H0. 준비
- API 초안, 결과 객체의 틀, 공통 입력 층, 참값 도구, scipy·statsmodels·R 대조 데이터, 속도 기준선, 정확도·속도 체계에 넣는 계획.
- 결정(사용자, 2026-11-20): docs/design/htest.md §1.1 H6–H15, 영역 사이의 결정 H16–H21.
- 결과: docs/api/htest.md(계획한 공개 이름 72개의 시그니처, 결과 필드와 풀기, 에러, scipy·statsmodels → sd 이름의 대응표, 마일스톤별 질문 §12), src/stardust/htest/(`__init__.py` — `sd.htest`, 공개 이름은 아직 없다; `_result.py`의 `_TestResult`와 `conf_int`; `_common.py`의 `Samples`·`Tables`와 옵션 검사), tests/htest/(`problems.py` 문제 281개, test_api_surface.py, test_result.py, test_common.py, test_reference_data.py; 63개), tools/htest_ref.py(Arb·정확한 유리수, 17묶음 138문제 → tests/htest/ref/), tools/gen_htest_scipy.py(→ tests/htest/data/scipy_h1–h8.json, statsmodels.json), tools/gen_htest_r.py(base R 145문제 → tests/htest/data/r.json), bench/htest/h0/(speed.py, speed.md, speed.json: scipy·statsmodels 41검정 × n = 10, 10³, 10⁶, 배치 10⁴ × 30), docs/design/htest.md §2.2·§7(정확도·속도 체계에 넣는 파일 목록).
- 찾은 것: scipy·R이 참값에서 크게 벗어나는 곳은 상쇄다: Bartlett의 거의 같은 분산(scipy 53배, R은 음수의 통계량), G 검정의 O ≈ E(scipy 1e-7), |r| ≈ 1의 Pearson p값(scipy 5e-7–0.11, R 2e-7–0.09), 큰 오프셋의 t(scipy·R 1e-8–3e-7; R은 짝 t 1e-6, F 2e-8). R은 깊은 꼬리에서 p = 0을 낸다(한 표본 KS 정확 n = 80의 참 1.7e-21, wilcox.test 정규 근사의 참 6.1e-51). 그 밖의 scipy 값은 참값과 1e-12 안이고, 정수 통계량의 정확 분포는 3e-16 안이다. scipy의 2 × 2 `fisher_exact`는 양쪽 규칙의 허용오차가 1e-14다(설계 초안의 1e-7은 `binomtest`와 R의 것). scipy `wilcoxon`의 `zstatistic`은 `method='asymptotic'`을 준 때만 있다. 작은 표본에서 scipy 호출 하나의 고정 비용은 0.06–0.25 ms, 배치에서 줄마다 Python 고리인 `kendalltau`는 10⁴ × 30에 950 ms다.
- sd.regression과 겹치는 곳(`jarque_bera`와 `r.jarque_bera()`, `f_oneway`와 분산분석표·더미 OLS, `pearsonr`과 `linregress`(뒤에 `linregress`를 뺐다, G188), `multipletests`와 단계적 선택): H16–H21로 정했고 docs/design/htest.md §1.4, docs/api/htest.md §13에 적었다(Jarque–Bera의 커널과 클래스는 H2에서 옮긴다; `anova`와 `f_oneway`는 계약 안에서 같고 비트는 아니다).

### H1. 평균과 분산
- t 검정 넷(Student, Welch, Yuen의 절삭), 일원 분산분석(Welch 포함), Alexander–Govern, 분산의 동질성(Bartlett, Levene, Fligner–Killeen). 신뢰구간은 `conf_int(alpha=0.05)`.
- 결정(사용자, 2026-11-20): `df`는 언제나 float, `ttest_ind_from_stats`에 `df`와 `conf_int()`, 표준오차 0의 t·p는 NaN, 실수 스칼라는 원소 하나의 표본, Welch 분산분석은 `f_oneway(equal_var=False)`, `f_oneway`에 `df_num`·`df_denom` (docs/design/htest.md §1.1 H22–H27). 결과를 보고(사용자, 2026-11-20): 분산 0인 집단의 NaN을 다표본 검정에도, `levene`에 `df_num`·`df_denom`과 `bartlett`·`fligner`·`alexandergovern`에 `df`(풀기는 scipy 그대로), 표본이 둘 미만이면 다섯 모두 `TypeError`, Welch의 작은 표본은 `df` NaN, 속도 숙제는 H10으로, Levene의 R 대조는 `car::leveneTest` (H28–H33).
- 결과: src/stardust/htest/_parametric.py(아홉 함수와 결과 클래스), csrc/htest/(moments.hpp: 줄의 double-double 적률과 고정 덩어리의 스레드풀 분할, 2의 거듭제곱 단위, 분산 0의 판정, 키 정렬; parametric.cpp: 검정, p값은 sd.stats 분포의 메서드), csrc/bind/htest_bind.cpp(`stardust._core.htest`), tests/htest/test_parametric.py(참값 67문제의 계약, 배치, 스레드, `nan_policy`, 퇴화, 단위, regression의 OLS와의 일치, 메시지), 참값 6묶음 67문제(tools/htest_ref.py; Yuen, `alexandergovern`, `levene`, `fligner`, 신뢰구간과 `kappa`를 더했다), 문제 37개 더(313개), bench/htest/h1/(speed.py, speed.md, speed.json).
- 찾은 것: 모든 참값이 계약 안이다(통계량 3.8e-16, p값은 max(1, κ_p)로 나눠 3.8e-16, 자유도 2.1e-16). scipy·R은 큰 오프셋(t·F·Welch·Alexander–Govern·Bartlett·Levene 2.6e-9–1e-7, 짝의 차 1.1e-9)과, 짝수 n의 중앙값을 반올림해 동점을 가르는 Fligner(p 6.4e-9–2.9e-7)에서 틀린다. statsmodels의 t·분산분석도 오프셋에서 6.5e-9–3.2e-7이다. 속도는 10스레드에서 모든 경우에 scipy보다 빠르다(n = 10에서 10–40배, 10⁶에서 1.0–6.4배, 배치 2.4–5.7배); 1스레드의 10⁶과 배치, `levene`의 긴 줄(중앙값 선택이 한 스레드)은 H10으로 넘겼다(H32). R `car::leveneTest`는 오프셋 10⁸에서 Levene의 W가 8.4e-9 틀린다 (docs/design/htest.md §7).

### H2. 적합도 1: 분포를 정한 검정
- 카이제곱과 Cressie–Read, D'Agostino의 셋, Jarque–Bera, Shapiro–Wilk(AS R94), Anderson–Darling(1.17부터의 꼴), Kolmogorov–Smirnov 한 표본(정확·근사·점근), Cramér–von Mises 한 표본.
- 결정(사용자, 2026-11-20): `kstest` 감싸개는 없다, 분포 인자는 sd.stats 분포 객체나 호출 가능한 `cdf`(문자열·`args` 없음), `anderson`의 `dist`는 sd.stats 분포 클래스이고 기본 `method='interpolate'`, Shapiro–Wilk는 AS R94, Jarque–Bera의 커널을 지금 htest로 옮기고 H20의 분산 0 규칙을 쓴다 (docs/design/htest.md §1.1 H34–H38). 결과를 보고(사용자, 2026-11-20): `sd.regression.JarqueBeraResult`를 htest의 클래스로(namedtuple의 튜플 동작이 없어진다; regression G165), `anderson`의 Weibull은 H9에서, Cramér–von Mises 꼬리의 절대 1e-16은 그대로, 큰 n의 `ks_1samp(method='exact')`는 scipy처럼 근사로, `chisquare`·`power_divergence`의 `df`는 둔다, R `nortest`·`goftest`를 대조용으로 설치 (H39–H44).
- 결과: src/stardust/htest/_gof.py(열 함수와 결과 클래스), csrc/htest/(gof.cpp: 검정; normality.hpp: 중심 적률 합과 Jarque–Bera, regression의 `resid_stats`·`jarque_bera()`가 부른다; ks.hpp: Smirnov(double-double 로그의 Birnbaum–Tingey 합), Kolmogorov, 한 표본 양쪽 분포(double-double Durbin 행렬); cvm.hpp: Csörgő–Faraway; lanes.hpp), tests/htest/(test_goodness_of_fit.py, refdata.py: 테스트와 정확도 측정이 같이 쓰는 계약), 참값 7묶음 111문제, 측정 bench/speed/cases/htest.py·bench/accuracy/htest.py(영역 htest를 측정 체계에 넣었다; H0·H1의 bench/htest/h0·h1은 지웠다), 골든 tests/golden/cases_htest.py.
- 찾은 것: 모든 참값이 계약 안이다. regression의 골든 804케이스가 옮긴 커널에서 비트까지 같다(분산 0 규칙은 상수 잔차의 적합만 바꾼다: 옛 JB 245.66, 지금 NaN). scipy는 O ≈ E의 Cressie–Read(λ = 2/3에서 66배), 관측 0의 Freeman–Tukey(NaN), 왜도 0의 `skewtest`(y = 0을 1로 바꿔 Z ≠ 0), 오프셋 데이터의 D'Agostino(1e-8), 로지스틱 Anderson의 맞추기(fsolve xtol 1e-5: 통계량 7.8e-9), n > 140의 한 표본 KS 양쪽 p값(Pelz–Good, 4.4e-6), Cramér–von Mises의 1e-7에서 멈춘 급수(꼬리 0.016)에서 틀린다. R의 `ks.test`는 깊은 꼬리에서 1 − cdf = 0(참 1e-113, 1e-251)을 낸다.
- R의 `nortest` 1.0-4·`goftest` 1.2-3(H44; 더 설치된 꾸러미 없음)과 견주면 같은 통계량은 1e-12 안이고, `goftest::cvm.test`의 p값은 꼬리(참 5.2e-11)에서 0이다.

### H3. 순위 검정
- Mann–Whitney(정확·정규), 순위 합, Wilcoxon 부호 순위(정확·정규·나열), Kruskal–Wallis, Friedman, Brunner–Munzel, Ansari–Bradley(정확·정규), Mood, Mood의 중앙값 검정, Page의 경향 검정.
- 결정(사용자, 2026-11-21): 동점이 있으면 `'exact'`는 중간 순위의 조건부 정확 분포, `'auto'`의 문턱은 scipy 그대로, `zstatistic`은 정규 근사를 쓴 때마다, 배치의 `method`는 줄마다의 문자열 배열(중첩 리스트), `median_test`에 `axis`, `ranksums`는 둔다 (docs/design/htest.md §1.1 H45–H50).
- 결과: src/stardust/htest/_rank.py(열 함수와 결과 클래스), csrc/htest/(rank.cpp: 검정, 정확한 중간 순위; rank_exact.hpp: 128비트 정수의 가우스 이항계수·부분합, double-double의 조건부 분포, Page의 행 분포; divergence.hpp: gof.cpp에서 옮긴 Cressie–Read 항), tests/htest/test_rank.py, 참값 10묶음 87문제(정수로 센 정확 분포, Arb의 꼬리, Friedman의 작은 문제의 정확 p값), 문제 51개 더(429개), 골든과 측정 사례(H2의 `ks_1samp` 배치 `method`도 줄마다의 리스트가 되어 그 골든 둘만 다시 기록했다).
- 찾은 것: 모든 참값이 계약 안이다(최대 0.028). scipy의 `ansari` 정확 p값은 N = 45–108에서 6.5e-9–2.6e-7 틀리고(`gscale`), `wilcoxon`의 정확한 꼬리는 1 − cdf라 n = 60에서 0(참 1.7e-18)이며, scipy와 R의 Kruskal–Wallis H는 N = 1100에서 3.2e-13 상쇄한다. 동점이 있는 `'exact'`의 조건부 분포는 R `wilcox.test`와 1.3e-15 안이다 (docs/design/htest.md §7).

### H4. 적합도 2: 표본끼리
- Kolmogorov–Smirnov 두 표본(격자 경로의 정확 분포), Cramér–von Mises 두 표본(정확·점근), Anderson–Darling k 표본, Epps–Singleton.
- 결정(사용자, 2026-11-21): `ks_2samp`의 동점이 있는 `'exact'`는 관측한 동점의 조건부 정확 분포(R `psmirnov`와 같다), `cramervonmises_2samp`의 `'exact'`는 큰 n까지의 동적 계획(`'auto'` 문턱 20은 그대로), `anderson_ksamp`의 기본 `variant='midrank'`과 둘로 풀리는 결과 (docs/design/htest.md §1.1 H55–H57). 결과를 보고(사용자, 2026-11-21): `cramervonmises_2samp`의 동점도 조건부, `epps_singleton_2samp`에 `df`, 점근 Cramér–von Mises 꼬리는 H41 그대로, `anderson_ksamp`의 `samples`는 리스트·튜플만 (H58–H61).
- 결과: src/stardust/htest/_gof.py(네 함수와 결과 클래스), csrc/htest/(gof2.cpp: 검정; ks2.hpp: 두 표본 KS의 블록 동적 계획(확률의 양수 합, 동점의 조건부)과 Hodges의 닫힌 꼴; cvm2.hpp: 두 표본 Cramér–von Mises의 창 동적 계획), tests/htest/test_goodness_of_fit_samples.py, 참값 4묶음 58문제(라벨 배치의 정수 셈, Arb의 점근·표·특성함수), 문제 48개 더(477개), R `ks.test(exact = TRUE)` 대조, 골든과 측정 사례.
- 찾은 것: 모든 참값이 계약 안이다(최대 0.078; 점근 Cramér–von Mises 꼬리는 H41의 절대 규칙). 동점이 있는 KS의 조건부 분포는 R과 1.1e-15 안이고 scipy와는 0.22–8.5 다르다. scipy의 점근 KS는 n > 140의 Pelz–Good으로 3e-7, 점근 Cramér–von Mises 꼬리는 1e-7에서 멈춘 급수, 오프셋의 Epps–Singleton은 1e-9 틀린다. Cramér–von Mises의 정확 분포는 표본마다 약 100개까지(scipy 20) (docs/design/htest.md §7).

### H5. 상관
- Pearson(1 − r²을 상쇄 없이), Spearman(두 꼴), 점 이연 상관, Kendall τ-b·τ-c(정확·정규), 가중 τ, Somers' D, Chatterjee ξ.
- 결정(사용자, 2026-11-21): `spearmanr`와 `spearmanrho`를 둘 다, Spearman의 정확 분포는 넣지 않는다, `weightedtau`는 넣고 `pvalue`는 NaN, 별칭 `correlation`은 없다, `pearsonr`의 r은 htest의 정확한 커널이고 `sd.corrcoef`와 비트까지 맞추지 않는다 (docs/design/htest.md §1.1 H62–H66).
- 결과: src/stardust/htest/_correlation.py(여덟 함수와 결과 클래스), csrc/htest/(correlation.cpp: 검정과 Pearson 커널; kendall.hpp: 역순 쌍의 정확 분포, 128비트 정수와 양수의 블록 합), tests/htest/test_correlation.py, 참값 6묶음 68문제(새 묶음 `weightedtau`·`somersd`·`chatterjeexi`), 문제 49개 더(526개), R `cor.test` 대조, 골든과 측정 사례. 결과를 보고(사용자, 2026-11-21): `linregress`가 htest의 Pearson 커널을 부르게 했다가 같은 날 `linregress` 자체를 뺐다(regression G188; H67을 바꿨다), 동점이 있는 Kendall `'exact'`는 `ValueError`, 표준오차 0의 `somersd` p값은 NaN, `weightedtau`의 NaN은 NaN, `chatterjeexi`의 x 동점은 위치로, `somersd`의 표는 10⁷칸까지, `spearmanr`의 기본 `axis=None` (H67–H73).
- 찾은 것: 모든 참값이 계약 안이다(최대 0.038). scipy의 `pearsonr` p값은 r ≈ 1에서 0.11(eps 10⁻⁷)–1(eps 10⁻⁹: p = 0, 참 1.2e-74), R `cor.test`는 5.6e10배 틀린다. R의 Kendall 정확 p값은 n > 33에서 1 − cdf라 깊은 꼬리가 6.8e-14(참 2.4e-82)·음수다. scipy의 `weightedtau`는 동점에서 1e-14 (docs/design/htest.md §7).

### H6. 분할표
- 카이제곱 독립성 검정(Yates, Cressie–Read, 가중 표), Fisher 정확 검정(2 × 2와 r × c), Barnard·Boschloo(장애 모수의 참 상한), McNemar, Cochran Q, 오즈비(조건부 MLE·표본)와 상대위험의 추정과 구간.
- 결정(사용자, 2026-11-21): r × c `fisher_exact`의 기본은 정확한 망 알고리즘과 작업량 상한, `barnard_exact`·`boschloo_exact`의 `n`은 처음 격자 크기로만(결과는 참 상한), `odds_ratio`·`relative_risk`를 넣는다, `chi2_contingency`는 가중 표를 받는다, `mcnemar`의 기본 `exact=True`, `chi2_contingency`의 `df`는 float (docs/design/htest.md §1.1 H77–H82).
- 결과: src/stardust/htest/_contingency.py(여덟 함수와 결과 클래스), csrc/htest/(contingency.cpp: 검정; fexact.hpp: r × c 망 알고리즘(정확한 경로 병합과 가운데에서 만나기); nuisance.hpp: Bernstein 꼴의 장애 모수 함수와 테일러 상한의 분기 한정; nchg.hpp: Fisher의 비중심 초기하; divergence.hpp: 0인 칸의 극한을 커널에), tests/htest/test_contingency.py, 참값 8묶음 88문제(r × c는 열의 동적 계획, Barnard·Boschloo·오즈비는 정수 다항식의 근을 flint로 고립), 문제 70개 더(596개), scipy·statsmodels·R(`chisq.test`, `fisher.test`, `mcnemar.test`) 대조, 골든과 측정 사례.
- 찾은 것: 모든 참값이 계약 안이다(최대 0.25: 오즈비의 한쪽 구간 끝). scipy의 `barnard_exact`는 반올림한 Wald 통계량으로 |T| = |T_obs|인 거울 표를 빠뜨려 [[7, 12], [8, 3]]에서 p값이 9.5e-4 낮고, `shgo`가 상한을 1.2e-7 놓치는 표가 있다; Cressie–Read의 0인 칸은 NaN, O ≈ E의 G는 1.6e-11이다. R `fisher.test`의 r × c는 참값과 1e-12 안이고, 기본 workspace가 거부하는 4 × 5(N = 100)는 workspace = 2e8에서 7초에 sd와 1.7e-10 다르다(sd 1.1초). R의 조건부 오즈비와 구간은 `uniroot`의 허용오차로 6e-4까지 다르다. 결과를 보고하며 물은 것도 정했다: `median_test`도 0인 칸의 극한 규칙, R `Exact`는 설치하지 않는다(H83–H84) (docs/design/htest.md §7).

### H7. 비율과 이항
- 이항 검정과 비율의 구간, 분위수 검정, 포아송 평균의 E-test, statsmodels의 비율 z·카이제곱 검정, 비율의 신뢰구간(모든 방법), 두 독립 비율의 차·비·오즈비 검정과 구간. 비교 대상에 R `prop.test`·`binom.test`.
- 결정(추천안, 사용자 위임 2026-11-21): `proportion_confint`의 방법은 scipy에 있으면 scipy의 이름(`'exact'`, `'wilson'`, `'wilsoncc'`), 그 밖은 statsmodels의 이름, 기본 `'normal'`; `alternative`는 `'less'`·`'greater'`; `prop_var`는 None 또는 실수; `test_proportions_2indep`는 `return_results` 없이 statsmodels의 필드, 둘로 풀린다; `compare`는 statsmodels의 값; 기본 방법은 statsmodels 0.15.0의 것으로 고정; `binom_test` 없음; `proportions_chisquare`에 Yates 보정 없음 (docs/design/htest.md §1.1 H85–H92). 구현하며 정한 것(추천안, 위임): `proportions_chisquare`는 셋으로 풀리고 `df`를 더한다, 구간 함수는 (…, 2) 배열, 한쪽 구간은 scipy의 방향(statsmodels의 `'larger'`는 [0, 위 끝]), `proportion_estimate` 없음, 인자 범위와 개수의 배치, 차의 점수 검정은 참 제한 최우추정(statsmodels의 삼차식 버그), E-test는 모든 결과의 합, `quantile_test`의 스칼라 `q`·`p`와 `nan_policy`, `'binom_test'` 구간은 채택 영역의 끝 (H93–H101).
- 결과: src/stardust/htest/_proportion.py(여덟 함수와 결과 클래스), csrc/htest/proportion.cpp(이항의 양쪽 규칙과 구간, 뒤집은 이항 검정의 조각 탐색, 분위수, E-test의 꼬리 합, z·카이제곱, 두 비율의 Wald 꼴·점수 검정·Koopman·점수 구간의 뒤집기), tests/htest/test_proportion.py, 참값 8묶음 188문제(새 묶음 일곱; E-test는 정확한 T 비교와 Arb의 꼬리 합, `'binom_test'` 구간은 정확한 유리수의 조각 이분), 문제 140개 더(736개), scipy·statsmodels·R(`binom.test`, `prop.test`, 다른 검정인 `poisson.test`) 대조, 골든과 측정 사례.
- 찾은 것: 모든 참값이 계약 안이다(최대 0.13: logit-smoothed 구간의 0인 개수). statsmodels 0.15.0의 차의 점수 검정은 Miettinen–Nurminen 삼차식의 L1에 n0 v 대신 c0 v를 써서 귀무값이 0이 아니면 다른 추정값을 낸다(p값 0.28%, 점수 구간의 끝 1.5%; 끝에 있는 추정값에서는 NaN). scipy의 E-test는 두 합을 잘라 깊은 꼬리에서 4.9e-9–9000배 낮고, 격자와 관측의 T를 다른 식으로 계산해 관측만큼 극단인 결과를 빠뜨린다(무작위의 약 8%, 6.6e-4까지). statsmodels의 `'binom_test'` 구간은 단조가 아닌 p값에서 채택 영역 밖의 교차점에 닿기도 하고(48/48), count2 = nobs2의 Koopman 구간은 이중근을 numpy가 1e-7쯤 틀린다. R `binom.test`·`prop.test`는 참값과 1e-12 안이다 (docs/design/htest.md §7).

### H8. 다중 비교와 보정
- Tukey HSD(스튜던트화 범위 분포), Dunnett(다변량 t), `multipletests`(열 방법), `false_discovery_control`, `combine_pvalues`. 비교 대상에 R `p.adjust`·`ptukey`.
- 결정(추천안, 사용자 위임 2026-11-21): `false_discovery_control`(scipy)과 `multipletests`(statsmodels, 긴 이름의 방법 전부)를 한 커널로, `fdrcorrection` 없음; `multipletests`의 기본 `'holm-sidak'`; 결과 필드 `reject pvalues_corrected alpha_sidak alpha_bonferroni`(넷으로 풀린다); `multipletests`에 `axis`; `fdr_gbs` 없음; 스튜던트화 범위는 htest 안에; `dunnett`에 `rng` 없음 (docs/design/htest.md §1.1 H102–H108). 구현하며 정한 것(같은 위임): `lfdr` 없음, `max_iter`는 0 이상의 int, 기각은 정확한 보정값과 alpha의 비교, p값·alpha의 범위 검사, 사후 비교의 `df`, 사후 비교의 입력 규칙, Dunnett은 한 인자 꼴의 결정적 적분(`MultivariateT.cdf` 대신), 사후 비교의 `summary()` (H109–H116).
- 결과: src/stardust/htest/_multicomp.py(다섯 함수와 결과 클래스), csrc/htest/(multicomp.cpp: 바인딩·`combine_pvalues`·집단의 double-double 적률; multitest.hpp: 열 방법의 한 커널, Hommel의 볼록 껍질 O(m log m); studentized_range.hpp: 스튜던트화 범위의 꼬리(sd.stats `t_mixture` 위의 상쇄 없는 범위 적분)와 꼬리를 뒤집는 근 찾기; dunnett.hpp: 한 인자 다변량 t의 꼬리), tests/htest/test_multicomp.py, 참값 4묶음(`tukey_hsd`·`dunnett`은 새 묶음: 안쪽은 Arb의 엄밀한 적분, 바깥쪽은 2⁻⁹⁵까지의 Gauss–Legendre 패널; 보정은 정확한 유리수(두 단계와 기각 포함), `combine_pvalues`는 Mudholkar–George와 `kappa`를 더했다), 문제 43개 더, R `multcomp`의 Dunnett 대조, 골든과 측정 사례.
- 찾은 것: 모든 참값이 계약 안이다(최대 0.18: Dunnett 구간의 끝, 상대 1.8e-15). scipy의 `studentized_range.sf`는 꼬리를 잃어 q = 17.8, ν = 87에서 0(참 7.8e-21)이고 R `ptukey`는 3.1e-10이다(1 − cdf); 오프셋 10⁸의 평균의 차는 scipy 2.5e-8, R 3.3e-7 틀린다. scipy의 Dunnett은 무작위 준몬테카를로라 절대 9e-5까지, R `multcomp`는 7e-7까지 다르다. 상수 집단에서 R은 p 5.4e-10, t 1.2e15를 낸다. statsmodels의 두 단계 FDR은 단계마다 반올림한 비교를 쓰고 `alphacSidak`은 1.3e-14 상쇄한다. Hommel은 O(m log m)이라 m = 10⁵에서 15 ms다(statsmodels 10³에서 3.6 ms). Dunnett은 결정적이지만 scipy보다 4–6배 느리다(H10) (docs/design/htest.md §7).

### H9. 재표본
- `permutation_test`, `monte_carlo_test`, `bootstrap`, `PermutationMethod`·`MonteCarloMethod`·`BootstrapMethod`와 앞 검정들의 `method=`(`ttest_ind mannwhitneyu wilcoxon ansari anderson anderson_ksamp pearsonr spearmanrho chatterjeexi chi2_contingency fisher_exact`, `pearsonr`의 `conf_int(method=)`), Baumgartner–Weiss–Schindler, `multiscale_graphcorr`, `goodness_of_fit`, `anderson`의 세 모수 Weibull(H40).
- 결정(추천안, 사용자 위임 2026-11-21): scipy의 이름인 설정 객체와 `rng` 필수, `rvs(size, *, rng)`, 내장 통계량의 빠른 길은 H10, `multiscale_graphcorr`과 `goodness_of_fit`(`anderson`의 족과 닫힌 꼴의 셋, 최우추정)을 넣는다, `bws_test`는 나열 아니면 `rng`, `axis=None`, `bootstrap_result` 없음, R은 설치된 `coin`·`MASS`만 (docs/design/htest.md §1.1 H117–H125). 구현하며 정한 것(추천안, 위임): `spearmanrho`의 몬테카를로는 순위 상관, 재표본 j는 (키, 위치, j)의 함수라 스레드·`batch`와 상관없다, MGC의 `compute_distance='euclidean'`·`workers` 없음, `goodness_of_fit`의 `guessed_params` 없음·`fit_result`는 sd.stats 객체, Weibull의 프로파일 최우추정과 NaN, `anderson`의 몬테카를로는 `goodness_of_fit`, NaN 통계량의 p는 NaN, 분할표는 초기하 추출, 순위 검정의 `method='permutation'`, `n_resamples=inf`의 상한 10⁷, 결과 꼴 (H126–H138).
- 결과: src/stardust/htest/_resampling.py(여섯 함수, 세 설정 객체, 결과 클래스, 앞 검정들이 부르는 공통 길), csrc/htest/(resampling.cpp: 재표본과 나열, p값, 분위수, BWS, 분할표의 추출, 바인딩; mgc.hpp; mle.hpp: 여덟 족의 최우추정과 통계량, `anderson`이 같이 쓴다), 앞 검정 다섯 파일의 `method=`, tests/htest/test_resampling.py(참값 57문제의 계약, scipy·coin·MASS, 재현성과 스레드·배치, 통계적 타당성, 앞 검정들의 정확한 나열 = 그 정확 분포·scipy), 참값 4묶음 57문제(tools/htest_ref.py), data/scipy_h9.json·r_h9.json, 골든 109케이스(모두 1,342), bench/speed/cases/htest.py의 "재표본과 몬테카를로", bench/accuracy/htest.py의 H9 묶음.
- 찾은 것: 모든 참값이 계약 안이다(계약 대비 최대: MGC의 통계량 0.75(7.5e-15), `goodness_of_fit`의 CvM 0.53, 정확한 나열의 p값 0.007). 정확한 나열의 귀무분포는 scipy와 같은 순서이고 p값도 같다. scipy의 `goodness_of_fit` Weibull은 범용 최적화라 통계량이 5e-6–5.4e-5, `MASS::fitdistr`는 로지스틱 3e-6·Weibull 8e-5 다르다. 속도(1스레드, n = 100, 9999 재표본): `permutation_test` 6.1 ms(scipy 19.8), `bootstrap` 1.7 ms(2.7), MGC 33 ms(1.45 s), `goodness_of_fit` 정규 99 ms(38; 원소마다의 분포 호출, H10).

### H10. 최적화와 일관성 점검
- 모든 공개 함수를 크기마다 scipy·statsmodels·R과 견주어 느린 곳을 고치고, 메시지·시그니처의 영역 간 일관성을 점검하고, 코드 리뷰를 한다.
- H1에서 넘어온 것(H32): 1스레드의 double-double 적률 훑기를 NEON으로(10⁶에서 scipy 1.6 ms 대 5.0 ms), `levene`의 집단마다 중앙값 선택을 병렬로(10⁶ 23 ms, scipy 25 ms), `fligner` 정렬의 병합 단계를 병렬로.
- 결정(추천안, 사용자 위임 2026-11-21): 범위는 O97처럼 모든 공개 함수, 바쁜 기계의 1라운드 측정과 고치기 전후 빌드의 A/B(기준선은 올리지 않는다), 비트를 지키는 개선만, 일관성은 htest 안을 고치고 다른 영역은 적기만, 코드 리뷰는 사용자에게, R `BWStest`·`mgc`·`energy`는 설치하지 않는다 (docs/design/htest.md §1.1 H139–H149).
- 결과: 1스레드 NEON double-double(적률, `pearsonr`, 중심 적률; csrc/htest/vdd.hpp), 몇 개의 무거운 조각도 자는 일꾼을 깨우는 `for_pieces_woken`(csrc/htest/lanes.hpp), 긴 줄의 병렬(`pearsonr`, 항의 병렬과 차례 합: `power_divergence`·`anderson`, 키 정렬의 병합 경로, 값 정렬, `levene`의 중앙값), 분포 메서드의 배열 호출(csrc/htest/dist_array.hpp: `goodness_of_fit`·`anderson`·z 검정), 재표본의 통계량을 커널에서 바로(H119: `ttest_ind`·`pearsonr`·`spearmanrho`·`anderson_ksamp`), 배치의 정확 분포 공유(`mannwhitneyu wilcoxon ansari`), 보정의 정렬과 Bonferroni·Šidák, 구간의 분위수 하나. 모든 영역의 골든이 1·3·10스레드에서 같다. 측정 사례 셋을 더했다(`ttest_ind_from_stats`, `relative_risk`와 그 구간). 표는 bench/htest/speed.md(바쁜 기계의 1라운드, 기준선이 아니다), 수는 docs/design/htest.md §7.
- 일관성: 결과의 `conf_int()`를 틀린 꼴로 부른 에러를 regression의 꼴(`conf_int(): takes 0 positional arguments ...`)로, `mannwhitneyu`의 `method` 메시지 순서. 다른 영역은 적기만 했다(docs/design/htest.md §7).
- 코드 리뷰는 사용자가 직접 한다(H149).

# 희소 영역 (sd.sparse)

> 상태: **설계 초안** (2026-11-21). SP0은 완료·승인, SP1은 완료·승인, SP2는 완료·승인, SP3은 완료·승인(2026-11-21), SP4는 완료·승인(위임, 2026-11-21), SP5는 완료·승인(위임, 2026-11-21), SP6은 완료·승인(위임, 2026-11-21), SP7은 완료·승인(위임, 2026-11-22), SP8은 완료·승인(위임, 2026-11-22)이다. 설계는 [docs/design/sparse.md](docs/design/sparse.md), API의 초안은 [docs/api/sparse.md](docs/api/sparse.md)다.
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
| SP0 준비: API 문서, 시험 행렬, 참값(Arb)·scipy·R 대조 도구, Accelerate 탐침, 색인 dtype의 측정, 기준선 | 완료·승인 (2026-11-20) |
| SP1 형식과 변환: `coo_array csr_array csc_array dok_array`, 표준 꼴, `sd.ndarray`·numpy·scipy와 주고받기 | 완료·승인 (2026-11-21) |
| SP2 연산: 산술·비교·원소별 함수, SpMV·SpMM·SpGEMM, 리덕션 | 완료·승인 (2026-11-21) |
| SP3 인덱싱과 만들기: 인덱싱과 대입, `eye_array diags_array random_array kron kronsum hstack vstack block_array block_diag tril triu find` | 완료·승인 (2026-11-21) |
| SP4 직접 풀이: `spsolve splu factorized inv spsolve_triangular`, `LinearOperator`, `sd.linalg`의 `TypeError` | 완료·승인(위임) (2026-11-21) |
| SP5 반복법: `cg gmres minres bicgstab lsqr lsmr` | 완료·승인(위임) (2026-11-21) |
| SP6 고유값과 특이값: `eigsh eigs svds` | 완료·승인(위임) (2026-11-21) |
| SP7 행렬 함수와 노름: `expm expm_multiply onenormest norm` | 완료·승인(위임) (2026-11-22) |
| SP8 최적화와 일관성 점검 | 완료·승인(위임) (2026-11-22) |

## 마일스톤

### SP0. 준비
- API 초안(scipy.sparse·scipy.sparse.linalg → sd.sparse 이름의 대응표 포함), 시험 행렬(격자, 대류-확산, 무작위, 특이, 복소 에르미트 등; 실제 행렬은 내려받아도 되는지 묻는다), Arb 참값 도구, scipy·R `Matrix` 대조 데이터, Accelerate 탐침(결정성, 특이 판정, int32 한계), int32·int64 색인의 SpMV 측정, 속도 기준선, 정확도·속도 체계에 넣는 계획.
- 결정(사용자, 2026-11-20·21): docs/design/sparse.md §1.1 SP6–SP20(BSR·DIA 없음과 `format=None`의 CSR, DOK만, int64 색인, dtype 넷, 언제나 표준 꼴인 CSR·CSC, 0 지우기는 scipy의 규칙, 강한 0과 예외 없는 `A / 0`, 1·2차원, 이름과 `norm`, 합의 순서, 반복법은 이 라이브러리, 실제 행렬은 목록을 보이고 승인 뒤, 직접 풀이는 모두 LU/AMD; SP0 뒤에 int64 색인 유지와 SuiteSparse 여덟 개).
- 결과: docs/api/sparse.md(계획한 공개 이름 41개와 클래스의 메서드·속성 80개의 시그니처, scipy.sparse·scipy.sparse.linalg → sd.sparse 대응표, 마일스톤별 질문 §10), src/stardust/sparse/(`sd.sparse`, 공개 이름은 아직 없다), csrc/accelerate/sparse.hpp(Accelerate LU 래퍼: 언제나 넘기는 `reportError`, status와 메시지의 예외 변환, LU/AMD 고정, RAII)·csrc/sparse/direct.cpp(비공개 `_lu_solve`)·csrc/bind/sparse_bind.cpp, 시험 행렬 tests/sparse/matrices.py(64개), 참값 tools/sparse_ref.py(Arb·정확한 유리수 7묶음 310점 → tests/sparse/ref/), 기록 tools/gen_sparse_scipy.py(scipy 1.18.1 661개)·tools/gen_sparse_r.py(R Matrix 1.7.6, 행렬 23개) → tests/sparse/data/, 탐침 tools/sparse_probe.py(Accelerate 결정성·동시 호출·특이, int32·int64 SpMV), 측정 등록 bench/speed/cases/sparse.py(마일스톤마다의 사례 계획)·bench/accuracy/sparse.py·tests/golden/cases_sparse.py(아직 비었다), 테스트 tests/sparse/(test_api_surface.py, test_accelerate.py, test_matrices.py, test_reference_data.py, test_suitesparse.py), 받는 도구 tools/sparse_suitesparse.py. 결과는 docs/design/sparse.md §5·§7.
- 찾은 것: LU/AMD는 아홉 꼴·세 크기·두 스레드 모드·동시 호출에서 한 번도 비트가 바뀌지 않았다. Metis는 동시 호출에서 매번 다르고, MTMetis(LU의 기본 순서)는 동시 호출에서 프로세스가 죽는다. 정확히 특이하지만 반올림 피벗이 0이 아닌 행렬은 Accelerate(와 scipy의 SuperLU)가 크기 1/ε의 유한한 해를 내고 정규화 역오차가 ε 크기라, SP4의 특이 판정에 조건수 추정이 필요하다. Accelerate는 0 × 0을 받지 않고, LU에 COLAMD를 주면 메시지를 보내고도 status 0이다. 우변이 여럿이면 열을 함께 풀어 열 하나의 비트가 혼자 푼 것과 다르다. int64 색인의 SpMV는 1스레드 1.02–1.16배(문턱 1.3 안)이고 10스레드의 2차원 라플라시안 n = 10⁶은 1.33–1.39배다(int64 유지, SP19). SpMV(CSR·CSC·COO, int32·int64, SpMM k = 16)는 scipy와 비트까지 같다. scipy의 중복 더하기는 경로마다 순서가 다르다(`tocsr`은 줄이 15개 이하일 때만 입력 순서). Accelerate LU/AMD는 scipy `splu`보다 2차원 격자에서 1.2–3.4배, 3차원에서 4–18배 빠르다. SuiteSparse 행렬 여덟 개(약 14 MB)는 받기로 했지만(SP20) 2026-11-21에 파일 서버가 403을 돌려줘 아직 받지 못했다(docs/design/sparse.md §7.2).


### SP1. 형식과 변환
- COO·CSR·CSC·DOK, 언제나 표준 꼴인 CSR·CSC(정렬, 중복은 입력 순서대로 더함), 형식 변환, `sd.ndarray`·numpy·scipy.sparse와 주고받기(scipy로는 복사 없는 뷰), 전치·`astype`·`reshape` 등.
- 결정(사용자, 2026-11-21): docs/design/sparse.md §1.1 SP21–SP34(`sd.ndarray` 필드의 Python 클래스, 쓰기 금지인 `indices`·`indptr`·`coords`, scipy에서는 언제나 복사·`to_scipy()`는 뷰, 별칭 없음과 COO의 `coords`만, DOK에 `dict` 메서드 없음, `np.asarray(A)`는 `TypeError`, `copy=`는 배열의 규칙, 데이터 인자의 이름은 scipy의 것; SP1 뒤에 변환 메서드의 `copy=None`, 색인은 언제나 복사, 1차원 `.T`는 `ValueError`, `has_canonical_format`은 실제 상태, 1스레드 변환 속도는 SP8로, 바빴던 속도 사례는 나중에 다시 잼).
- 결과: 공개 이름 `sparray coo_array csr_array csc_array dok_array issparse matrix_transpose`와 메서드·속성(docs/api/sparse.md §2), src/stardust/sparse/(`_base.py`, `_coo.py`, `_compressed.py`, `_dok.py`, `_interop.py`, `_common.py`), csrc/sparse/(`format.hpp`, `parallel.hpp`, `canonical.cpp`: 표준 꼴의 두 길(곧은 길과 양동이 길), `convert.cpp`, `dok.cpp`: 해시 맵), 테스트 tests/sparse/(test_formats.py, test_canonical.py, test_conversions.py, test_dok.py, conftest.py; test_api_surface.py가 클래스의 멤버까지 본다), 골든 812케이스(1·3·10스레드 같음), 속도 사례 12가지 × 크기(bench/sparse/speed.md).
- 찾은 것: 변환의 결과는 scipy와 구조·값이 비트까지 같다(시험 행렬 64개 전부; 결과 형식도 scipy의 것: `reshape`은 COO, DOK의 `real`·`imag`는 CSR, COO의 `astype`은 중복을 더한다). 다른 곳은 중복 합의 순서(scipy의 `tocsr`은 긴 줄에서, COO의 `sum_duplicates`·`todok`은 `np.add.reduceat`이라 `[1, 1e16, 1, -1e16]`이 sd 0, scipy 2), DOK의 순서(sd 행 우선, scipy 넣은 순서), 1차원 `.T`(sd `ValueError`), `has_canonical_format`(sd는 실제 상태)이다. 표준 꼴은 1·2·3·7·10스레드와 두 길에서 비트가 같다. 속도는 nnz 10⁷에서 scipy 대비 COO → CSR 1스레드 1.9배·10스레드 11.6배, CSR → CSC 2.3배·9.8배, CSR → COO 2.9배·21배, `sum_duplicates` 82배이고, `(data, indices, indptr)`·`(data, (row, col))`로 만들기는 색인을 검사하고 복사하므로 scipy(검사도 복사도 없음)보다 느리다(10⁷에서 1스레드 7.9·10.6 ms).
- 나중 일(SP34): SP1의 속도 표(bench/sparse/speed.md)의 † 줄은 다른 에이전트의 일과 함께 잰 것이라, 기계가 한가할 때 `bench.speed run --rerun-busy ... --promote`로 다시 잰다. 1스레드 nnz 10⁵–10⁶의 COO → CSR·CSR → CSC(목표 2배에 못 미침)는 SP8에서 고친다(SP33).

### SP2. 연산
- 산술과 비교(연산마다 scipy의 0 지우기 규칙, 암시적 0은 강한 0), 원소별 함수, SpMV·SpMM·밀집 @ 희소·SpGEMM(줄 합은 색인 오름차순의 FMA 순차 합: SpMV가 scipy와 비트까지 같다), 리덕션.
- 결정(사용자, 2026-11-21): docs/design/sparse.md §1.1 SP35–SP47(numpy 배열은 피연산자가 아님, 원소별 함수는 메서드만이고 `sd.sin(A)`는 메서드를 짚는 `TypeError`, 축 리덕션은 언제나 밀집, `A * 밀집`은 A의 형식, `A + c`·`A ** 0`은 `ValueError`, 리덕션의 `keepdims`와 `explicit=`; SP2 뒤에: 결과가 A의 모양인 브로드캐스트만, 왼쪽이 CSC면 CSC, 제자리 연산자 없음, 음수·NaN 지수는 `ValueError`, `axis`는 int나 None, 원소별 메서드는 배열의 이름(`asin atan asinh atanh round`), 양수와의 `maximum`은 scipy처럼).
- 결과: 연산자(`+ - * / ** @`, 비교, `-A`, `abs`), 메서드 `multiply maximum minimum power`와 f(0) = 0인 원소별 함수 18개, 리덕션 `sum mean max min nanmax nanmin argmax argmin`(docs/api/sparse.md §3; `trace`·`diagonal`은 §8대로 SP3), src/stardust/sparse/(`_ops.py`, `_reduce.py`, `_base.py`의 입구), csrc/sparse/(`arith.hpp`: 항 하나와 긴 줄의 조각 합, `spmv.cpp`: SpMV·SpMM·흩뿌리기·축 합과 평균, `binop.cpp`: 병합 이항 연산·비교·밀집 값 읽기, `spgemm.cpp`, `reduce.cpp`), 배열 영역의 `csrc/array/ops/ufunc.cpp`에 희소 배열을 짚는 메시지(실패한 호출에서만), 테스트 tests/sparse/(test_arithmetic.py, test_elementwise.py, test_products.py, test_reductions.py), 정확도 스윕 `spmv`·`sums`(계약 안, 가장 큰 것 0.83), 골든 4,627케이스를 더함(1·3·10스레드 같음), 속도 사례 SP2의 15가지 × 크기(smoke만; 기계가 바빠 기준선으로 올리지 않았다).
- 찾은 것: scipy의 희소 × 희소는 한쪽의 NaN·inf를 `a * 0` = NaN으로 저장하고(이쪽은 강한 0), `<=`·`>=`는 NaN에서 참이며, `argmax`는 NaN이 있어도 첫 암시적 0을 돌려준다. scipy의 축 합은 자기 줄을 따르는 쪽(CSR의 `axis=1`)만 `np.add.reduceat`(pairwise)라 이쪽과 끝자리가 다르다. scipy의 복소 SpMV는 numpy의 복소 곱 뒤 성분별 덧셈과 비트까지 같다. 1스레드 SpMV 0.89배, SpMM 1.6배, 이항 연산 1.1–1.2배, 흩뿌리기 1.0배, SpGEMM 0.67배(행 정렬 때문; 정렬까지 한 scipy보다 2.4배 빠르다), 10스레드는 흩뿌리기 말고 4–8배다(docs/design/sparse.md §5). 흩뿌리기의 병렬과 SpMM의 목표(3배)는 SP8로 넘긴다.

### SP3. 인덱싱과 만들기
- 정수·슬라이스·배열·마스크 인덱싱과 대입(키는 배열의 규칙과 메시지), 대각, 만들기 함수.
- 결정(사용자, 2026-11-21): docs/design/sparse.md §1.1 SP48–SP52(만들기 함수는 scipy의 배열 꼴 이름 `eye_array diags_array random_array block_array`, `random_array`는 `rng`를 꼭 받는다, CSR·CSC의 새 자리 대입은 받고 O(nnz), `format=None`은 scipy의 규칙이되 DIA·BSR 자리는 CSR, 정수 대각의 `diags_array`는 int64).
- 결과: 인덱싱(`A[key]`: 결과의 차원 수가 A와 같으면 A의 형식, 다르면 COO; `toarray()`가 밀집 인덱싱과 비트까지 같다)과 대입(형식마다 scipy의 저장 규칙, 같은 자리는 마지막 값), `trace diagonal setdiag`, 공개 이름 `eye_array diags_array random_array kron kronsum hstack vstack block_array block_diag tril triu find`(docs/api/sparse.md §4), src/stardust/sparse/(`_index.py`, `_construct.py`), csrc/sparse/(`index.cpp`: 축의 고르기 표, 바깥 고르기·원소 고르기·대입·대각, `construct.cpp`: kron·쌓기·삼각; `dok.cpp`에 `set_many`), 테스트 tests/sparse/(test_indexing.py, test_construct.py; scipy 1.18.1과 실시간 대조), 골든 1,069케이스를 더함(모두 6,508, 1·3·10스레드 같음), 속도 사례 SP3의 11가지 × 크기(smoke만; 기준선으로 올리지 않았다). 정확도 스윕은 없다(반올림이 없는 연산).
- 찾은 것: scipy 1.18.1의 CSR·CSC는 원소 고르기와 2차원 마스크에 밀집 배열을 돌려주고(COO·DOK는 COO), DOK는 경로마다 저장된 0을 빼거나 두며, COO의 같은 자리 대입은 첫 값을 남긴다. scipy의 버그 둘: `kron`은 한쪽이 아무것도 저장하지 않으면 dtype을 잃고 float64를 돌려주고, CSR의 `A[bool 행 마스크, ::2]`는 `ValueError`(reshape)다. 속도(nnz 10⁶, smoke): 행 고르기 1.6–3.3배, 열 slice 1.2배(10스레드 6배), `kron` 6.5배(10스레드 27배), `tril`·`hstack` 1.5배이고, `vstack`(0.57배, int64 색인의 대역), 정렬되지 않은 열 고르기(0.64배), `random_array`(0.21배, `sd.random`의 비복원 `choice`)는 SP8의 고칠 목록이다(docs/design/sparse.md §5).
- SP3 뒤의 결정(사용자, 2026-11-21): docs/design/sparse.md §1.1 SP53–SP58(대입한 0의 저장은 형식마다 scipy의 규칙, CSR·CSC의 원소 고르기는 COO, bool·int64의 `trace`는 int64, bool `kron`은 int64, `diags_array`의 대각은 `sd.ndarray`나 수, `random_array`는 `data_sampler(n)`과 그 dtype; 모두 구현한 그대로).

### SP4. 직접 풀이
- Accelerate LU(AMD 순서)로 `spsolve`·`splu`·`factorized`·`inv`, 이 라이브러리의 삼각 풀이, 특이의 판정(`LinAlgError`), `LinearOperator`, `sd.linalg`가 희소 입력에 쓸 함수를 짚는 `TypeError`.
- 결정(사용자, 2026-11-21): docs/design/sparse.md §1.1 SP59–SP65(LU 순서는 AMD 고정, `splu`는 내보내지 않는 `SparseLU`(`solve`, `shape`), scipy의 SuperLU 옵션 없음, 특이의 판정에 조건수 추정(κ₁·ε ≥ 1), 잰 역오차가 계약을 넘으면 반복 다듬기 한 번, 우변 여럿은 함께, `LinearOperator`는 `shape`·`matvec`만 위치).
- SP4 뒤의 결정(추천안, 사용자 위임 2026-11-21): docs/design/sparse.md §1.1 SP66–SP71(κ₁·ε ≥ 1의 판정 그대로, 배열의 함수도 희소 배열에 쓸 것을 짚는 `TypeError`(`csrc/bind/pyutil.cpp`·`errors.cpp`), `spsolve_triangular`는 꼭 주는 `upper`, (n, 1)의 b는 (n, 1), 열 하나·1차원의 희소 b는 희소 결과, `rmatvec` 없는 연산자는 `ValueError`; 모두 구현함).
- 결과: 공개 이름 `spsolve splu factorized inv spsolve_triangular is_sptriangular spbandwidth LinearOperator aslinearoperator`(docs/api/sparse.md §5), src/stardust/sparse/(`_direct.py`, `_interface.py`), csrc/sparse/(`direct.cpp`: `_lu_factor`·`_LuFactor`와 `lacn2`의 조건수 추정, `triangular.cpp`: 대입과 띠 폭; `arith.hpp`로 줄 합 `ordered_sum`을 옮김), csrc/accelerate/sparse.hpp(전치·켤레 전치 풀이), linalg의 입력 검사 한 곳(`csrc/linalg/input.cpp`)에 희소 배열을 짚는 메시지(실패한 호출에서만), 테스트 tests/sparse/(test_direct.py, test_interface.py; test_accelerate.py·test_suitesparse.py에 더함), 정확도 스윕 `solve`, 골든 158케이스를 더함(모두 6,666, 1·3·10스레드 같음), 속도 사례 SP4의 4가지 × 격자 다섯(smoke만; 기준선으로 올리지 않았다).
- 잰 것: 반복 다듬기 없이 saddle/600x150의 정규화 역오차가 404ε(나머지는 4ε 이하)이라 한 번 넣었고(SP63), 그 뒤 모든 시험 행렬과 SuiteSparse 다섯에서 0.50ε 이하, 참값 대비 앞오차는 0.39κ∞ε 이하다(계약 C_BACKWARD = C_SOLVE = 4). 특이 판정은 정확히 특이한 행렬 모두(Neumann 라플라시안, 같은 열, 상쇄, 구조적)를 잡고 특이가 아닌 것(κ 7.5×10¹⁴까지)은 모두 푼다. 비용은 분해의 10–15%다. 속도(smoke, scipy 대비): `spsolve`·`splu`는 2차원 n = 10⁴에서 같고 9×10⁴에서 1.7–1.8배, 10⁶에서 3.3배, 3차원 9.1×10⁴에서 17배, `SparseLU.solve`는 다듬기 때문에 n = 10⁴에서 0.38배(큰 행렬 1.3–2.6배), `spsolve_triangular`는 5–7배다.
- 찾은 것(Accelerate, SDK 26.5): 전치로 도는 풀이는 우변이 둘 이상이면 틀린 답을 내고(그래서 열마다), 복소 LU의 켤레 전치 풀이는 몇백 행에서 프로세스를 멈추고(SIGTRAP; 켤레와 전치로 돌림), CSR을 전치 속성으로 넘긴 분해는 CSC와 끝자리가 다르다(그래서 모두 CSC로). scipy 1.18.1의 `spsolve_triangular`는 고른 삼각 밖의 원소가 있으면 경고 없이 틀린 답을 내고, `spbandwidth`·`is_sptriangular`는 빈 CSR에서 예외다. SuiteSparse의 sherman3·cvxqp3(κ₁ 6.9×10¹⁶·2.2×10¹⁶)은 판정대로 `LinAlgError`다(scipy는 해를 돌려준다).

### SP5. 반복법
- CG, GMRES, MINRES, BiCGSTAB, LSQR, LSMR(scipy 1.18.1과 같은 알고리즘과 멈춤 판정, 결정적인 SpMV와 내적).
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/sparse.md §1.1 SP72–SP75(scipy의 튜플로 풀리는 읽기 전용 결과 객체와 `summary()`, `max_iter`, `callback` 없음, `gmres`의 `restart`·`max_iter`는 scipy 그대로). 구현하며 정한 것(추천안, 사용자 위임): SP76–SP82(허용오차·상한의 검사는 optimize의 규칙, `b`·`x0`은 `sd.ndarray`이고 `x`는 (n,), A·b·x0·M 가운데 복소가 있으면 복소, 결과의 필드 이름(`status`, `n_iter`, gmres의 `n_inner`, lsqr의 `normr_damped`·`var` 등), 행렬은 C++의 SpMV·결정적인 내적과 노름·Python 연산자는 곱마다 호출, `minres`의 실패는 `LinAlgError`, `lsqr`의 `var`는 언제나 계산).
- 결과: 공개 이름 `cg gmres minres bicgstab lsqr lsmr`와 결과 클래스 여섯(docs/api/sparse.md §6), src/stardust/sparse/(`_iterative.py`, `_result.py`), csrc/sparse/iterative.cpp(여섯 풀이, CSR·Python·항등 연산자, `sd.linalg.vecdot`·`vector_norm`의 합으로 내적·노름, LAPACK 3.10의 `lartg`), 테스트 tests/sparse/test_iterative.py(scipy 기록·실시간 대조, 잔차 열, Arb 참값 대비 앞오차, 스레드·형식·연산자의 비트)와 test_suitesparse.py의 열 문제, 정확도 스윕 `iterative`(여섯 영역 계약 안, 가장 큰 것 0.39), 골든 230케이스를 더함(모두 6,896, 1·3·10스레드 같음), 속도 사례 SP5의 27가지(smoke만; 기준선으로 올리지 않았다), Accelerate 반복법 탐침(`tools/sparse_probe.py iter`).
- 잰 것: scipy 기록 40개에서 상태는 모두, 반복 수는 36개가 같고 단계마다의 잔차가 상대 10⁻⁶ 안이다. 나머지는 scipy 자신이 반올림에 민감한 문제(무작위 비대칭 행렬의 BiCGSTAB: b를 10⁻¹⁶ 바꾼 scipy끼리도 반복 수가 54–71)와 BiCGSTAB의 반 단계 멈춤이다. SuiteSparse에서 memplus의 GMRES(20)는 안쪽 5,425번으로 같다. 속도(scipy 대비): 1스레드 1.03–1.6배, 10스레드 cg·lsmr 3.7–5.3배, gmres 2.3–2.8배(목표 1.5배·4배에 못 미친 곳은 SP8). Accelerate의 반복법은 한 스레드로 돌고 CG가 sd 1스레드보다 1.2–1.3배 빠르다(sd 10스레드보다 2.4–3.7배 느리다).
- 찾은 것: MINRES의 멈춤 판정은 잔차의 어림(φ̄)에 대한 것이라 조건수가 큰 행렬에서 참 잔차와 크게 갈라진다(graded/100x8, κ 10¹⁰: "a solution was found"로 멈추지만 참 상대 잔차 7×10⁻²; scipy도 같다). scipy의 `cg`·`bicgstab`·`gmres`·`minres`는 `maxiter=0`에서 `info = 0`(수렴)을 돌려주고, 복소 x0의 허수부를 경고와 함께 버리며, `lsqr(calc_var=True)`는 복소 행렬에서 dₖ²를 더해 복소 `var`를 낸다.

### SP6. 고유값과 특이값
- implicitly restarted Lanczos·Arnoldi(`eigsh`, `eigs`; shift-invert는 LU), Golub–Kahan–Lanczos 이중대각화(`svds`).
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/sparse.md §1.1 SP83–SP87(ARPACK을 scipy의 계산 순서까지 옮김, `v0=None`은 정해진 벡터이고 `rng`로 바꿈, `svds`는 GKL 하나이고 `solver=` 없음, `mode='buckling'`·`'cayley'` 둠, 수렴 실패는 결과의 상태). 구현하며 정한 것(추천안, 사용자 위임): SP88–SP96(scipy의 인자 이름과 `max_iter`, `tol = 0`은 2⁻⁵², 옵션과 상관없이 같은 꼴의 결과 객체(`eigenvalues`·`eigenvectors`, `U`·`S`·`Vh`, 상태 필드), 고유벡터의 부호는 정하지 않음, `k` 범위 밖은 `ValueError`, `sigma` 없는 `mode`는 `ValueError`, shift-invert의 역은 Accelerate LU, 계약 C_EIG = 1000, `svds`의 판정과 재시작). R의 `RSpectra` 0.16.2·`irlba` 2.4.1은 사용자 승인으로 설치했다(SP97).
- 결과: 공개 이름 `eigsh eigs svds`와 결과 클래스 셋(docs/api/sparse.md §7), src/stardust/sparse/`_eigen.py`, csrc/sparse/`arpack.cpp`(ARPACK의 여섯 드라이버), `svds.cpp`, `krylov.hpp`(SP5의 연산자·벡터 연산을 함께 씀), linalg 래퍼(`steqr lahqr trevc trsen geqr2 lanhs1 larf`), 테스트 tests/sparse/test_eigen.py(scipy 대조: 상태·OP 수·값, 모든 모드, Arb 계약, R 기록, 비트)와 test_suitesparse.py의 일곱 문제, 참값 `eig`에 비대칭 셋을 더함(23점), R 기록 `tests/sparse/data/r_eigen.json`(18개), 정확도 스윕 `eig`(세 영역 계약 안, 가장 큰 것 0.50), 골든 129케이스를 더함(모두 7,025, 1·3·10스레드 같음), 속도 사례 10가지(smoke만; 기준선으로 올리지 않았다).
- 잰 것: 같은 `v0`이면 scipy와 스무 문제 중 열다섯에서 OP 곱의 수까지 같고 나머지 다섯은 scipy 자신이 한 ulp에 흔들리는 폭 안이다. SuiteSparse 넷(bcsstk14, msc10848, memplus, young1c)의 OP 수가 scipy와 같다. Arb 대비 가장 큰 오차는 lap1d/1000 'LA'의 497ε‖A‖₂(762번 재시작; scipy 217). scipy `svds`('arpack')는 랭크 부족 rect/80x30d의 'SM'에서 계약의 2×10¹²배이고 sd는 0.009다. 속도(smoke): 같은 OP 수에서 n = 10⁴은 scipy의 0.5–0.6배(Vᴴx·V·h가 BLAS보다 느리다: SP8), shift-invert는 큰 행렬에서 1.2–2배, `svds` nnz 10⁶의 10스레드 2.3배.
- 찾은 것: scipy의 ARPACK은 `getv0`에서 B = I여도 v0에 OP를 곱하고 `tol = 0`을 2⁻⁵²로 쓴다(`dlamch('E')`가 아니다). scipy는 `v0=None`이면 실행마다 결과가 다르고, `sigma` 없는 `mode`를 조용히 무시하고, 실수 A와 복소 σ의 `eigs(return_eigenvectors=False)`는 고유값이 모두 0이다(`OPpart` 'r'·'i' 모두; Ritz 벡터 없이 Rayleigh 몫을 구한다. sd는 이때 안에서 벡터를 구한다). Accelerate의 `dlahqr`는 Z의 행 하나만 주면 갱신하지 않는다. RSpectra는 겹친 고유값의 중복도를 놓친다.

### SP7. 행렬 함수와 노름
- `expm`(Al-Mohy–Higham 2009), `expm_multiply`(Al-Mohy–Higham 2011), `onenormest`(Higham–Tisseur, 결정적인 열), `norm`.
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/sparse.md §1.1 SP98–SP101(`onenormest`의 열은 고정 키로 결정적, `expm`은 입력의 형식, `itmax` → `max_iter`, `norm`의 `'nuc'`·`-2`는 `ValueError`). 구현하며 정한 것(추천안, 사용자 위임): SP102–SP110(`onenormest`는 언제나 `(est, v, w)`로 풀리는 결과 객체(묻지 않은 쪽 None; 2026-11-22 고침), `expm`은 밀집 `sd.linalg.expm`의 선택(정확한 노름과 상한 먼저, n ≥ 100부터 ‖A⁸‖·‖A¹⁰‖ 어림, 삼각의 고침, 2¹⁰⁰ 가드), `expm`은 희소 입력만·정확한 0은 뺌, `expm_multiply`의 구간은 scipy 그대로, B는 밀집만, 연산자의 `traceA`는 고정 열의 Hutch++ 어림, `norm`의 세부(키워드 `ord`, vector_norm의 뜻, `'f'` 없음), 계약 C_EXPM = 10과 노름의 합 계약, C++ 파일 없이 Python이 기존 커널을 부름).
- 결과: 공개 이름 `expm expm_multiply onenormest norm`(docs/api/sparse.md §8), src/stardust/sparse/`_matfuncs.py`, 테스트 tests/sparse/test_matfuncs.py(Arb 계약, scipy 실시간·기록 대조, 밀집 `sd.linalg`와 대조, 결정, 비트·스레드·형식, 틀린 호출), 참값 `expmv`(38점: exp(sA)b와 t = 0..1의 다섯 점; tools/sparse_ref.py), 정확도 스윕 `norms`·`matfuncs`(아홉 영역 계약 안, 가장 큰 것 0.25), 골든 270케이스를 더함(모두 7,295, 1·3·10스레드 같음, 다른 영역 그대로), 속도 사례 34가지(smoke만; 기준선으로 올리지 않았다). `sd.linalg`의 `expm`·`matrix_norm`·`vector_norm`이 희소 배열에서 이 함수들을 짚는다.
- 잰 것: Arb 대비 `expm` 2.4(ε·max(1, ‖sA‖₁)·max|exp(sA)| 단위; scipy 4.8, R Matrix 4.2, 밀집 `sd.linalg.expm` 3.5), `expm_multiply` 2.0(구간 2.5, scipy와 같다), 합의 노름은 계약의 0.17배, 2-노름 18 ulp. `expm_multiply`는 단계가 같으면 scipy와 비트까지 같다(scipy 기록 넷 가운데 둘; 나머지는 대각합의 끝자리). `onenormest`는 작은 시험 행렬 42개 중 26개가 참값과 같고 가장 작은 비율 0.76(scipy seed 0: 25개, 0.76). 속도(smoke, scipy 대비): 노름은 작은 크기 2–4배·nnz 10⁷의 열 합 0.70배·Frobenius 0.63배, `onenormest` 1스레드 1.2–1.8배·10스레드 3.7–4.3배, `expm_multiply` 1스레드 1.3–1.6배·10스레드 1.6–3.3배, 구간 2.8–3.4배·6.5–7.6배, `expm` 1.5–1.9배(못 미친 곳은 SP8).
- 찾은 것: scipy의 `onenormest`는 시드 없는 전역 난수에 `argsort`가 안정 정렬이 아니라 같은 값의 순서도 정해져 있지 않다. scipy `expm`의 삼각 고침은 차수 13에만 있어 가까운 고윳값의 [[10, 1], [0, 10 + 10⁻¹²]]에서 (0, 1)의 상대오차가 2×10⁻⁵다(sd 9×10⁻¹⁶). scipy의 `norm(A, 2)`는 0인 행이 있는 empty/rows에서 `ArpackError`(시작 벡터 0)다. scipy의 희소 `expm`의 결과 형식은 경로마다 다르다(COO 입력이 COO나 CSR). R Matrix의 `norm(A, "F")`는 nnz 10⁷에서 765 ms다.

### SP8. 최적화와 일관성 점검
- 모든 공개 함수를 크기마다 scipy·R과 견주어 느린 곳을 고치고, 메시지·시그니처의 영역 간 일관성을 점검하고, 코드 리뷰를 한다.
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/sparse.md §1.1 SP111–SP117(`kronsum`은 단위 행렬과 곱하지 않고 값을 놓아 `sd.linalg.kronsum`과 같은 값, `random_array`의 `rng`는 기본값 없는 키워드, 결과의 repr은 짧은 실수 배열을 적는다, `count_nonzero(keepdims=)`, `spbandwidth`는 `BandwidthResult`이고 밀집 입력은 `sd.linalg.bandwidth`를 짚는다, 비트를 바꾸지 않는 속도 고침, 고치지 않은 것). 코드 리뷰는 사용자가 따로 한다(optimize O96의 둘째 부분과 같다).
- 결과: 1스레드 COO → CSR의 LSD 길과 짧은 줄의 순위 정렬, 1스레드 전치의 열 블록 길, 흩뿌리기(`x @ A`, CSC `@`, 열 합, `norm(ord=1)`)의 출력 블록 병렬, `X @ A`의 연속 작업 배열, SpMM의 미리 읽기, SpGEMM의 비트 집합, 색인 배열 고르기의 바이트 표, Krylov의 여러 열 내적·작은 행 구간·`gmres` MGS의 갱신과 내적 한 번에(csrc/sparse/`canonical.cpp spmv.cpp spgemm.cpp index.cpp krylov.hpp arpack.cpp svds.cpp iterative.cpp`). 골든 7,295케이스의 값은 모두 그대로이고, 바뀐 것은 고친 동작의 다섯(`error random_array rng`의 메시지, `spbandwidth` 넷의 결과 형)과 새 케이스 `kronsum complex special`이다. `deg2rad`·`rad2deg`는 이제 `sd.deg2rad`·`sd.rad2deg`를 부른다(같은 비트). 속도 표는 bench/sparse/speed.md(바쁜 기계의 한 라운드; 기준선으로 올리지 않았다).
- 잰 것(구판과 새판을 번갈아 부른 A/B, 1스레드 / 10스레드, 구판 ÷ 새판): COO → CSR nnz 10⁵ 1.59 / 1.11, 10⁶ 1.45 / 1.23(scipy 대비 1스레드 2.5배·2.0배: SP33의 목표에 닿음), CSR → CSC 10⁶ 1.41 / 1.0, 10⁷ 1.47(scipy 대비 1스레드 10⁵ 1.27배·10⁶ 1.49배: 목표 2배에 못 미침), `x @ A`·열 합·`norm(A, 1)` 10스레드 1.7–2.5(메모리 대역에 묶여 4배에 못 미침), `X @ A`(k = 16) 1스레드 2.9·10스레드 5.7–7.1, SpMM 1스레드 1.3–1.95(scipy 대비 10⁶ 2.0배: 목표 3배에 못 미침), SpGEMM 1스레드 1.27–1.69(정렬하지 않는 scipy의 0.68–0.9배), 정렬되지 않은 열 고르기 1스레드 1.4, `eigsh` n = 10⁴ 10스레드 1.67–1.92, `svds` nnz 10⁵ 10스레드 1.9, `gmres` lap3d/45 10스레드 1.28, `norm(A, 2)` 10스레드 1.44.
- SP8 뒤의 결정(추천안, 사용자 위임 2026-11-22): SP118–SP120(희소 배열·`LinearOperator`·`SparseLU`의 공개 메서드도 틀린 호출 모양이 메서드 이름으로 시작(`_calls.methods`), `block_diag`의 인자는 희소·밀집 모두 `arrays`(SP28의 예외), bench/sparse/speed.md는 SP8 실행에서 만든 생성 파일 그대로이고 A/B는 설계 §5).

---

# 수치 미적분 영역 (sd.integrate, sd.differentiate)

> 상태: **설계 초안** (2026-11-21). C0은 완료·승인(위임, 2026-11-21), C1은 완료·승인(위임, 2026-11-21), C2는 완료·승인(위임, 2026-11-21), C3은 완료·승인(위임, 2026-11-22), C4는 완료·승인(위임, 2026-11-22), C5는 완료·승인(위임, 2026-11-22), C6은 완료·승인(위임, 2026-11-22), C7은 완료·승인(위임, 2026-11-22), C8은 완료·승인(위임, 2026-11-22)이다. 설계는 [docs/design/integrate.md](docs/design/integrate.md)(두 모듈을 함께), API의 초안은 [docs/api/integrate.md](docs/api/integrate.md)와 [docs/api/differentiate.md](docs/api/differentiate.md)다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

## 진행 원칙
- 다른 영역과 같다: 마일스톤마다 시작 전에 물을 것(docs/design/integrate.md §9)을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.
- 큰 결정(사용자, 2026-11-21): 최상위 모듈 둘 `sd.integrate`·`sd.differentiate`(scipy의 이름 그대로, 설계 문서와 마일스톤은 하나); 첫 판은 적분 15개와 미분 3개, 상미분방정식은 나중의 묶음(옛 `ode`·`complex_ode`, `lebedev_rule`, `qmc_quad`는 넣지 않음); optimize의 고정 간격 차분 커널(`csrc/optimize/diff.hpp`)을 differentiate로 옮기고 optimize가 부른다(`approx_fprime`·`check_grad`는 optimize에 그대로, 비트·골든 불변); `derivative`·`tanhsinh`·`nsum`은 optimize 배치의 규칙(`converged`·`n_iter`·`nfev`, 이 무리만 `args`, scipy의 status, 함께 쓰는 C++ 반복 고리); scipy와 같은 걸음과 호출점의 비트 동일을 목표로 하고 Arb `acb.integral`로 결과와 오차 추정의 정직함을 확인; 시험 문제는 QUADPACK 책·scipy의 시험·Arb 참값의 어려운 적분(QUADPACK 원본을 받을지는 C0 전에 묻는다); 다른 영역의 전용 구적은 그대로 (docs/design/integrate.md §1.1 C1–C7).
- C0 전의 결정(사용자, 2026-11-21): 측정 영역 둘 `integrate`·`differentiate`; 결과 클래스 `IntegrateResult`·`DifferentiateResult`(`OptimizeResult`의 규칙); `quad` 계열은 결과 객체이고 `(integral, error)`로 풀린다(`full_output` 없음); 허용오차는 어디서나 `atol`·`rtol`; 상한은 `max_iter`·`max_level`·`min_level`·`max_terms`·`max_p1`(`limit`·`limlst`는 그대로); 값 필드는 어디서나 `integral`·`error`·`nfev`; `callback`·`kwargs`·`preserve_shape` 없음과 O38·O48의 확인; 차분 커널과 함께 nls의 `Differences`도 옮긴다; 반복 고리는 csrc 아래 새 바닥 `elementwise/`; QUADPACK Fortran 원본은 받아 gfortran으로 견주되 받기 전에 파일 목록과 크기를 묻는다(docs/design/integrate.md §1.1 C8–C17, 목록은 §7.6).
- 비교 대상은 scipy 1.18.1의 `scipy.integrate`·`scipy.differentiate`와 QUADPACK이고, 정확도 체계(`bench/accuracy`)와 속도 체계(`bench/speed`)에서 견준다. 참값은 Arb다.
- 설계 초안에서 찾은 것(docs/design/integrate.md §7.4): `quad`는 좁은 봉우리를 처음 규칙이 놓치면 틀린 값을 경고 없이 수렴으로 알린다(QUADPACK 책의 봉우리 α = 19·20). `tanhsinh`은 수렴한 46경우 가운데 여섯에서 오차 추정이 실제 오차보다 작다(최소 0.015배). Arb의 `acb.integral`은 끝점에서 유계가 아닌 피적분함수에 NaN을 돌려줘 참값은 닫힌 꼴·적분·치환의 세 방법이다. tanh-sinh 표는 libm과 c·c로 scipy와 16,386값이 비트까지 같다. scipy `derivative`의 order 8 가중치는 정확한 값과 상대 2.7×10⁻¹⁴까지 다르고, `roots_legendre`의 가중치는 n = 100에서 83,682 ULP까지 다르다.
- 기존 영역의 결과는 비트까지 같아야 한다(차분 커널과 반복 고리를 옮겨도 optimize·regression의 골든이 그대로).

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| C0 준비: API 문서, 결과 객체의 바탕, 시험 모음, Arb 참값 도구, scipy 기록과 호출 순서, QUADPACK 원본 대조, 측정·골든 등록, scipy 시간 | 완료·승인(위임) (2026-11-21) |
| C1 함께 쓰는 부품: 차분 간격 커널과 `Differences` 옮기기(C3, C15), 반복 고리를 `elementwise/`로 끌어내기(C4, C16); 공개 함수 없음 | 완료·승인(위임) (2026-11-21) |
| C2 미분: `derivative jacobian hessian` | 완료·승인(위임) (2026-11-21) |
| C3 표본과 고정 규칙: `trapezoid simpson romb cumulative_trapezoid cumulative_simpson fixed_quad newton_cotes` | 완료·승인(위임) (2026-11-22) |
| C4 QUADPACK 1: `quad`(QAGS, QAGI, QAGP), `dblquad tplquad nquad` | 완료·승인(위임) (2026-11-22) |
| C5 QUADPACK 2: `quad`의 `weight`(QAWO, QAWF, QAWS, QAWC) | 완료·승인(위임) (2026-11-22) |
| C6 배치 적분: `tanhsinh nsum` | 완료·승인(위임) (2026-11-22) |
| C7 벡터 값과 다차원: `quad_vec cubature` | 완료·승인(위임) (2026-11-22) |
| C8 최적화와 일관성 점검: 모든 공개 함수의 속도, 남은 질문(Gauss–Legendre의 상한, `approx_fprime`의 간격), 영역 간 일관성 | 완료·승인(위임) (2026-11-22) |

## 마일스톤

### C0. 준비
- 결정(사용자, 2026-11-21): docs/design/integrate.md §1.1 C8–C17(C0 전에 정할 것을 모두 정했다).
- API 초안의 확정(C11–C13의 이름), 결과 객체 `IntegrateResult`·`DifferentiateResult`의 바탕(`OptimizeResult`에서 끌어내되 동작 불변), 시험 모음(QUADPACK 책의 무리, scipy 시험의 f1–f15 등, 어려운 적분, 미분의 초등 함수·MGH), Arb 참값 도구(닫힌 꼴, `acb.integral`, 치환), scipy 1.18.1의 결과와 호출 순서의 기록, QUADPACK 원본 21개 파일(docs/design/integrate.md §7.6)을 크기와 함께 물은 뒤 `build/comparators/quadpack/`에 받아 gfortran으로 돌린 결과(원본은 저장소에 넣지 않는다), `bench/areas.py`·골든·측정의 등록, scipy 기준선.
- 결과: docs/api/integrate.md·differentiate.md(확정; 풀기와 생성자의 규칙), 결과의 바탕 `src/stardust/_results.py`의 `_FieldResult`(`OptimizeResult`가 그 위로 옮겨 가고 동작·골든 불변), src/stardust/integrate/(`IntegrateResult`)·src/stardust/differentiate/(`DifferentiateResult`), 시험 모음 tests/integrate/problems.py(1차원 116, 배치 3, 겹친 7, 벡터 2, 급수 9, 표본 12)·tests/differentiate/problems.py(초등 함수 11 × 9점, 끝 근처 14, MGH 47), Arb 참값 tools/integrate_ref.py(닫힌 꼴·`acb.integral`·치환을 모두 계산해 겹쳐야 저장; 134개 → tests/integrate/ref/)·tools/differentiate_ref.py(f′·f″ 104점, 야코비안 47·헤시안 42 → tests/differentiate/ref/; 헤시안은 Arb 위의 2차 전진 미분 tests/differentiate/jets.py), scipy 기록 tools/gen_integrate_scipy.py·tools/gen_differentiate_scipy.py(값과 호출 순서: `quad`는 점 전부, 배열 호출은 모양과 SHA-256), QUADPACK 원본 tools/gen_integrate_fortran.py(netlib 21개 파일과 LINPACK `dgtsl.f`를 build/comparators/quadpack/에, gfortran 라이브러리를 ctypes로; `d1mach`·`xerror`는 도구가 쓴 대체, 먼저 쓴 `dgtsl` 대체는 원본과 116개 실행 모두 비트로 같다 → tests/integrate/data/fortran_quadpack.json, fortran_vs_scipy.json), 측정·골든 등록(bench/areas.py, bench/accuracy/integrate.py·differentiate.py, bench/speed/cases/integrate.py·differentiate.py의 사례 계획 26개, tests/golden/cases_integrate.py·cases_differentiate.py). scipy 시간은 기계가 바빠 점검 수준으로 한 번씩만 쟀고 기준선에 넣지 않았다(docs/design/integrate.md §5).
- 찾은 것(docs/design/integrate.md §7.6, §7.7): Fortran 원본과 scipy의 C 판은 116개 가운데 112개에서 같은 점을 같은 순서로 부르지만 적분이 비트로 같은 것은 62개다. FMA 축약을 켜고 빌드하면 적분 100개가 같아지고 점은 90개만 같아, scipy는 누적만 축약한 것으로 보인다(C4에서 `std::fma`의 자리를 찾는다). QAWS의 로그 가중치 예제는 같은 점에서 4×10⁻⁹ 다르다(C5에서 본다). 수렴이라고 알린 값이 오차 추정 밖인 것: `quad`는 봉우리 α = 19·20과 f10(double 상한 `math.pi / 2` 대신 π/2의 적분으로 외삽), `tanhsinh`은 76개 가운데 여덟. scipy `derivative`는 같은 점을 혼자와 배치로 부르면 df가 99점 가운데 29점에서 끝자리가 다르다(BLAS dgemv의 합 순서). 책의 무리 목록은 책 없이 설계 초안의 세 무리와 루틴 예제를 쓰고 닫힌 꼴을 Arb로 확인했다. C0을 마치며 생긴 넷(`dgtsl`, double 끝의 참값, `summary()`, 풀기의 범위)은 추천안으로 정했다(사용자 위임 2026-11-21, docs/design/integrate.md §1.1 C18–C21); `dgtsl`은 사용자 승인으로 LINPACK 원본을 받아 빌드에 넣었다.

### C1. 함께 쓰는 부품
- `csrc/optimize/diff.hpp`를 differentiate의 C++로, `src/stardust/optimize/_diff.py`의 `Differences`를 differentiate의 Python 내부로 옮기고 optimize·regression이 그것을 부른다. 반복 고리는 csrc 아래 새 바닥 `elementwise/`로 간다. `csrc/optimize/batch.cpp`의 반복 고리(원소 패스의 병렬, `args` 모으기, 호출, 진행 목록)를 끌어내 원소마다 여러 점의 호출로 넓힌다. optimize·regression의 테스트·골든·배치 속도가 그대로인지 확인한다.
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/integrate.md §1.1 C22–C25(1% 넘게 느려지면 그 자리만 템플릿으로 되돌리기: 되돌린 곳 없음; 이름 `csrc/differentiate/steps.hpp`·`sd::differentiate`, `csrc/elementwise/loop.hpp`·`sd::elementwise`, `differentiate/_differences.py`; `Call`의 여러 점 호출 꼴; R175는 주석만 고치고 R176은 재서 문서에 적기).
- 결과: csrc/differentiate/steps.hpp(optimize의 `diff.hpp`에서 옮김; `lsq.cpp`·`objective.hpp`·`hessian.hpp`·`scalar.cpp`는 include와 이름만), csrc/elementwise/loop.hpp(`batch.cpp`에서 `for_elements`·`gather`·`Call`·`Progress`·`advance`·scipy의 status 번호·numpy의 sign·minimum·maximum을 끌어내고 `Call`을 원소마다 p개 점의 (k, p) 호출로 넓힘), csrc/optimize/batch.cpp(네 알고리즘만), src/stardust/differentiate/_differences.py(`Differences`; optimize/_diff.py와 regression/_nls.py가 부른다), tests/differentiate/test_differences.py; docs/design/integrate.md §1.1·§6.1·§6.2·§8·§9, docs/design/optimize.md §3.1·§3.3.1·§3.5·§6, docs/api/optimize.md §2(R176의 호출 수), docs/api/differentiate.md, ARCHITECTURE.md §1·§2. 공개 함수와 CMake는 그대로다.
- 확인: 골든은 모든 영역이 1·3·10스레드에서 그대로(32,220케이스, optimize 3,876·regression 902), optimize·regression·integrate·differentiate 테스트 3,524개와 tests/bench가 지난다. 배치 속도는 옮기기 전과 뒤의 빌드를 번갈아 잰 가운데값의 비가 1스레드 0.992–1.004, 10스레드 0.983–1.006이다(기계가 바빴다; docs/design/integrate.md §6.2의 표).
- 찾은 것: R175의 2⁻²⁶은 √ε(√u가 아니다). R176: exp의 `approx_fprime`은 x = 10⁻¹²·10⁻⁸·10⁻⁶에서 61번 부른다(간격 키우기가 4배씩 오른다); 고치면 골든이 바뀌어 두었다. 같이 본 것: 처음 간격의 반올림이 √ε 바로 아래인 x(exp의 10⁻⁵·10⁻⁴)는 간격을 키우지 않아 상대오차가 5.1×10⁻⁹·2.6×10⁻¹⁰이다(문서의 정확도보다 나쁘다; docs/design/integrate.md §6.1).

### C2. 미분
- scipy `derivative`의 중첩 스텐실과 간격 줄이기, `jacobian`(변수마다 함께), `hessian`(`jacobian`의 `jacobian`). 호출점은 scipy와 같게, 가중치의 정확도는 C2 전에 묻는다.
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/integrate.md §1.1 C26–C38(여러 점은 (k, p) 2차원·`args`는 (k, 1); `jacobian`·`hessian`의 f는 (m, N) → (k, N)·(N,); 가중치는 올바로 반올림(중앙은 정확한 유리수, 한쪽은 Q(√c)의 정확한 계산); `hessian`의 0 < rtol < 100ε는 `ValueError`; 구현하며 생긴 아홉: 변수마다의 `initial_step`, `step_factor`의 범위, NaN `step_direction`, `jacobian`은 scipy처럼 모든 열을 부름, 한쪽 가중치의 계산, 틀린 값의 글, 결과의 배치 모양과 `summary()`, x의 규칙, 중앙 스텐실은 마디의 짝마다의 차를 더함).
- 결과: csrc/differentiate/derivative.{hpp,cpp}(스텐실의 점과 갱신; `derivative`는 csrc/elementwise/loop.hpp의 (k, p) 호출을 처음 쓴다, `jacobian`의 열 호출과 `hessian`의 안쪽 `jacobian`), csrc/bind/differentiate_bind.{hpp,cpp}(`stardust._core.differentiate`), src/stardust/differentiate/(`_derivative.py`, `_weights.py`, `_result.py`의 `summary()`), tests/differentiate/(`test_derivative.py`, `test_jacobian.py`, `test_weights.py`), 골든 265케이스, 정확도 스윕 셋(bench/accuracy/differentiate.py), 속도 사례 다섯(bench/speed/cases/differentiate.py); docs/design/integrate.md §1.1·§2·§3.8·§4·§6.3·§7.8·§8·§9, docs/api/differentiate.md, ARCHITECTURE.md §1·§2.
- 찾은 것(docs/design/integrate.md §7.8): scipy 기록과 견주면 `derivative`는 124개 가운데 123개가 점·모양·수·status·`n_iter`·`nfev`까지 같다(다른 하나 x³의 0은 sd가 둘째 반복에 수렴; scipy는 −2). scipy의 가중치를 넣으면 124개 모두 같다. `jacobian`은 47개 가운데 43개, `hessian`은 42개 가운데 33개의 호출이 같다(다른 것은 0에 가까운 항목이 합의 끝자리로 다른 반복에 멈춘 것; scipy BLAS의 합 순서는 흉내 낼 수 없다). df가 비트로 같은 것은 적다(가중치와 합이 다르다). Arb: 수렴한 도함수 109개 모두 atol + rtol·|df| 안(최대 상대 6.2×10⁻¹¹), 야코비안·헤시안의 노름 오차는 rtol의 0.004배·0.11배 이하. 속도는 scipy보다 10–50배 빠르다(바쁜 기계의 탐침).

### C3. 표본과 고정 규칙
- 표본 규칙(sd의 합 규칙), `fixed_quad`(Gauss–Legendre의 점과 가중치), `newton_cotes`.
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/integrate.md §1.1 C39–C54(데이터 함수의 입력; `x`는 위치와 키워드 모두; `axis`의 기본 −1; 올바로 반올림한 Gauss–Legendre, 공개 이름 없음; `fixed_quad`는 값 하나·`newton_cotes`는 `(an, B)`; 구현하며 생긴 열하나: `romb`의 `dx`와 `show` 없음, `x`·`dx`의 꼴(다섯 함수에 한 규칙, 줄마다의 `dx`, `x`와 함께 준 `dx`는 `ValueError`), 누적의 `initial`(scipy 그대로), scipy 1.11의 짝수 개 점 `simpson`, dtype(float64·complex128), 빈 입력(점 하나 이하는 0), NaN은 퍼짐, 줄마다의 pairwise 합과 `cumulative_sum`의 스캔, `fixed_quad`의 (…, n) 값과 유한한 끝, `newton_cotes`의 정확한 유리수, 표 밖의 Gauss–Legendre는 double-double과 정확한 정수 산술).
- 결과: src/stardust/integrate/(`_samples.py`, `_rules.py`, `_gauss.py`), csrc/integrate/(`samples.cpp`: 줄마다의 규칙과 합, `gauss.cpp`: 표와 double-double Newton, `gauss_table.hpp`: 생성한 표), csrc/bind/integrate_bind.{hpp,cpp}(`stardust._core.integrate`), tools/gen_gauss_legendre.py(n ≤ 100의 표, Arb), tools/integrate_ref.py의 `samples`·`gauss`(→ tests/integrate/ref/samples.json: 규칙의 식의 정확한 유리수, gauss.json: n = 101–1000의 일곱 n), tests/integrate/(`test_samples.py`, `test_fixed.py`), 골든 142케이스(모두 155), 정확도 스윕 둘(bench/accuracy/integrate.py, 기준선 bench/accuracy/results/integrate.json), 속도 사례 아홉(bench/speed/cases/integrate.py); docs/design/integrate.md §1.1·§3.7·§4·§6.3·§7.2·§7.9·§8·§9, docs/api/integrate.md, ARCHITECTURE.md §1·§2.
- 찾은 것(docs/design/integrate.md §7.9): 표본 규칙은 정확한 식과의 계약을 12벌 모두에서 지키고(한계의 0.050까지), scipy의 기록 65개 가운데 59개와 비트로 같다(나머지는 합의 순서로 끝자리). `fixed_quad`의 점은 n = 5에서 scipy와 같고 n = 20·100에서는 모두 다르다(scipy `roots_legendre`의 끝자리). scipy의 `newton_cotes`는 N ≥ 15에서 Vandermonde 풀이라 `B`가 N = 20에서 9%, N = 30에서 10⁶배 틀린다(sd는 정확한 값의 반올림). 표 밖의 Gauss–Legendre는 처음 한 번 O(n²)라 10⁴점이 0.7초다(n에 상한이 없다; C8에서 다시 본다). 속도는 scipy보다 2–25배 빠르다(`newton_cotes`의 첫 계산만 느리고 기억한다).

### C4. QUADPACK 1
- QAGS·QAGI·QAGP(QK21·QK15I, ε-알고리즘), scipy의 status와 글, 겹친 적분. 결정의 자리의 FMA는 scipy의 호출 순서와 견주어 찾는다.
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/integrate.md §1.1 C55–C70(물었던 넷: 구간 정보의 필드 넷 `intervals interval_integrals interval_errors n_intervals`, 복소수는 `quad(complex_func=True)`만, `vectorized=True` 없음, scipy 글의 줄바꿈 그대로; 구현하며 생긴 열둘: 복소수 결과의 `real`·`imag`, 수렴의 글, C5 전의 `weight`는 `NotImplementedError`, FMA는 Kronrod 누적 넷, 겹친 적분의 오차는 모든 `quad`의 최댓값·status는 처음 실패한 것, 겹침은 C++에서, `quad`·`nquad`의 인자, QNG는 옮기지 않음, 파일, `summary()`, 속도의 빠른 길).
- 결과: csrc/integrate/quadpack.hpp(dqagse·dqagie·dqagpe·dqk21·dqk15i·dqpsrt·dqelg를 문장마다 옮긴 템플릿, `std::fma` 넷), csrc/integrate/quadpack.cpp(scipy의 갈래, 구간 배열, `nquad_solve`의 겹침), src/stardust/integrate/_quadpack.py(`quad dblquad tplquad nquad`), `_result.py`의 `summary()`와 `real`·`imag`, tests/integrate/test_quadpack.py, 골든 171케이스 더함(모두 326), 정확도 스윕 둘(`quad`, `multiple`; 기준선 갱신), 속도 사례 넷 공개(bench/speed/cases/integrate.py); docs/design/integrate.md §1.1·§1.3·§2·§3.1·§3.2·§4·§6.3·§7.10·§8·§9, docs/api/integrate.md.
- 찾은 것(docs/design/integrate.md §7.10): scipy의 C 판이 FMA로 합친 곳은 Kronrod 규칙의 누적 넷(resg, resk, resabs, resasc)뿐이고, 그 자리에 `std::fma`를 적으면 가중치 없는 106개 모두 점·값·오차·`nfev`·status·글·구간이 scipy와 비트로 같다(축약 없이는 9개, 한 자리만 풀면 12–55개). 겹친 적분 7개와 `nquad`도 점과 값이 같다. scipy `nquad`의 오차는 바깥 `quad`의 것이 아니라 모든 `quad`의 최댓값이다(설계 초안을 고쳤다). Arb: 수렴한 94개 가운데 참 오차가 오차 추정 밖인 것은 scipy와 같은 셋(봉우리 α = 19·20, f10). 속도는 함숫값 21개의 가장 작은 경우만 7% 느리고 나머지는 1.2–2.3배 빠르다(바쁜 기계의 탐침).

### C5. QUADPACK 2
- QAWO·QAWF(체비쇼프 모멘트, Clenshaw–Curtis), QAWS·QAWC(수정 Clenshaw–Curtis).
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/integrate.md §1.1 C71–C83(물었던 하나: `wopts` 없음; 구현하며 생긴 열둘: QAWF의 주기 필드 넷 `cycle_integrals cycle_errors cycle_status n_cycles`, `wvar`·`limit`·`max_p1`·`limlst`의 검사와 범위에 달린 검사의 자리, QAWF는 `atol`만(기본값이 아닌 `rtol`은 `ValueError`), scipy 판을 따르는 셋(`dqc25s`의 오타 수정, `dgtsl`의 꼴, `dqawoe`의 작은 구간 판정), scipy가 틀린 값을 내 원본을 따르는 셋(모멘트 행, QAWF의 ω = 0, 큰 ω), QAWO의 구간 하나, 가중치 규칙의 FMA 자리, Arb 계약).
- 결과: csrc/integrate/quadpack.hpp(dqawoe·dqawfe·dqawse·dqawce·dqk15w·dqc25f·dqc25s·dqc25c·dqcheb·dqmomo·dgtsl을 문장마다 옮김, `std::fma` 127곳), csrc/integrate/quadpack.cpp(가중치의 갈래와 범위 검사, QAWF의 주기 배열, `nquad`의 가중치 층), src/stardust/integrate/_quadpack.py(`weight wvar max_p1 limlst`의 검사, QAWF의 글), `_result.py`의 주기 필드와 `summary()`, tests/integrate/weights.py(가중치 스윕 99개)와 tools/integrate_ref.py의 `weights`(→ tests/integrate/ref/weights.json), tests/integrate/test_quadpack.py(모음 116개 모두)·test_quadpack_weights.py, tools/gen_integrate_fortran.py의 `--probe-qc25s`, 골든 141케이스 더함(모두 508), 정확도 스윕 `quad_weights`(기준선 갱신), 속도 사례 넷 공개; docs/design/integrate.md §1.1·§3.1·§6.3·§7.2·§7.6·§7.11·§8·§9, docs/api/integrate.md §1.3·§2·§3.
- 찾은 것(docs/design/integrate.md §7.11): 가중치 루틴에서 scipy의 C 판은 원본의 한 식 안의 곱·합을 모두 축약하고(`dqc25f`의 점 centr ± hlgth·x도), 그 127곳에 `std::fma`를 적으면 모음 116개 모두와 스윕 99개 가운데 98개, 만든 400개가 점·값·오차·`nfev`·status·글·구간까지 scipy와 비트로 같다(축약 없이는 1,476개 가운데 432개). §7.6의 QAWS 4×10⁻⁹ 차이는 원본 `dqc25s.f`의 오타(`res24 = res12 + …`)이고 scipy는 고쳤다. scipy의 기계 코드에서 원본과 다른 곳 셋을 더 찾았다: `dgtsl`은 마지막 줄을 소거하지 않고(정확도는 같다), `dqawoe`의 작은 구간 판정에 uflow 대신 ε, `dqc25f`의 모멘트 행이 호출마다 1행이라 `max_p1`을 다 쓰면 틀린 값을 수렴으로 알린다(sd는 원본대로). QAWF의 ω = 0은 원본·scipy 모두 [0, ∞)를 적분한다(sd는 [a, ∞)). Arb: 109개 모두 수렴하고 참 오차 ≤ 오차 추정 + 8ε|I|(Clenshaw–Curtis 추정에는 반올림 하한이 없어 추정만으로 넘는 것이 16개, scipy와 같은 비트). 속도는 scipy와 같거나 1.1배 빠르다(바쁜 기계의 탐침).

### C6. 배치 적분
- `tanhsinh`(scipy의 표·합·오차 추정, 무한 구간의 치환), `nsum`(직접 합과 적분 경계).
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/integrate.md §1.1 C84–C93(물었던 셋: tanh-sinh 표는 scipy의 libm 식, `log=True`는 C4의 답대로 실수만(b < a는 `ValueError`), 끝점은 scipy처럼 부르고 버린다(f는 예외 대신 inf·NaN); 구현하며 생긴 일곱: 합은 numpy의 pairwise와 scipy의 logsumexp 그대로(복소 단계는 시스템의 `cexp`·`clog`), `max_level` ≤ 16과 `max_terms` ≤ 2²⁴(메모리의 상한), log의 허용오차, 첫 호출과 `nsum`의 갈래 순서, `nsum`의 두 예외(log 오차의 합 순서, 넘친 직접 합), 결과와 값 검사).
- 결과: csrc/integrate/tanhsinh.cpp(표, Euler–Maclaurin 합과 오차 추정, numpy의 pairwise 합·logsumexp, `nsum`의 갈래와 f(−x)·f(x) + f(−x)의 평가; 고리는 csrc/elementwise/loop.hpp를 바꾸지 않고 씀), src/stardust/integrate/_tanhsinh.py, tests/integrate/test_tanhsinh.py, 골든 42케이스, 정확도 스윕 둘(`tanhsinh`, `nsum`; 기준선 갱신), 속도 사례 넷 공개(bench/speed/cases/integrate.py); docs/design/integrate.md §1.1·§3.5·§3.6·§4·§7.12·§8·§9, docs/api/integrate.md §1.2·§6·§10.
- 찾은 것(docs/design/integrate.md §7.12): 시험 모음의 `tanhsinh` 101개·배치 셋·`nsum` 아홉과 기록 밖의 71경우(log, 수준, 끝의 변환, `nsum`의 네 갈래)가 모두 scipy와 점·값·오차·`nfev`·status까지 비트로 같다(다른 것은 scipy가 복소수를 내는 log의 a = b = −∞ 하나). 표는 scipy와 비트로 같고 정확한 값과는 최대 980 ULP(정확한 j·h로 재면 3,449). Arb: 수렴한 76개 가운데 오차 추정 밖은 scipy와 같은 여덟, `nsum`은 아홉 모두 안. 모음 밖의 e^(−cx²)(c 40,001개)에서는 1,093개가 오차 추정보다 큰 오차를 낸다(상대 1.1×10⁻⁸까지; scipy와 같다). 속도는 작은 배치에서 23–25배, 10⁵원소에서 4배 빠르다(바쁜 기계의 탐침).

### C7. 벡터 값과 다차원
- `quad_vec`(적응 Gauss–Kronrod, 노름), `cubature`(gk21·gk15의 곱, Genz–Malik, 무한 축의 치환; C++의 힙).
- 결정(추천안, 사용자 위임 2026-11-21): docs/design/integrate.md §1.1 C94–C107(물었던 셋: `cubature`는 `converged`와 정수 `status`, `regions`는 C55의 꼴인 배열 필드 `regions region_integrals region_errors n_regions`, 규칙 객체·`workers` 없음; 구현하며 생긴 열하나: numpy·Python의 산술(FMA 없음, numpy의 합, 페이지 정렬 `cblas_ddot`의 노름, Python `heapq`), scipy의 구간 기억(100 MB의 LRU) 그대로, 낮은 Gauss 규칙은 scipy의 `roots_legendre` 값, `genz-malik`의 Fortran 순서 인자와 배치를 따르는 합, scipy와 다른 셋((−∞, b]의 치환, 무한 축의 `points`, 유한하지 않은 값의 status 3), 두 함수의 인자, `summary()`, Arb 계약).
- 결과: csrc/integrate/adaptive.cpp(두 알고리즘, 힙, 규칙, 치환), csrc/integrate/numpy_ops.hpp(numpy의 합·노름; tanhsinh.cpp와 함께), src/stardust/integrate/_quad_vec.py·_cubature.py, tests/integrate/test_quad_vec.py·test_cubature.py·adaptive.py, 골든 62케이스(결과 클래스의 필드 넷이 늘어 기존 케이스는 `absent` 목록만 바뀜), 정확도 스윕 둘(`quad_vec`, `cubature`; 기준선 갱신), 속도 사례 여덟 공개; docs/design/integrate.md §1.1·§2.3·§3.3·§3.4·§4·§6.3·§7.13·§8·§9, docs/api/integrate.md §1.3·§4·§5·§9·§10.
- 찾은 것(docs/design/integrate.md §7.13): 기록 14개와 기록 밖의 69경우, 모음과 무리가 scipy와 점·값·오차·`nfev`·status·구간·영역까지 비트로 같다(scipy의 구간 기억이 넘쳐 다시 계산하는 경우도). scipy `cubature`의 결함 셋: (−∞, b]에서 [−b, ∞)를 적분, 무한 축과 `points`에서 에러, NaN을 'converged'로. `genz-malik`은 f에 Fortran 순서의 배열을 넘겨 합의 순서가 바뀐다. Accelerate의 `cblas_ddot`은 897원소부터 주소에 따라 끝자리가 달라 scipy의 `quad_vec` 자신도 결정적이지 않다. Arb: 수렴하면 참 오차 ≤ 오차 추정 + 8ε|I|이고 넘는 것은 scipy와 같은 짧은 목록(끝점 특이점 x^−0.9, 좁은 봉우리). 속도는 함숫값이 작은 경우 4.5–10배 빠르다(바쁜 기계의 탐침).

### C8. 최적화와 일관성 점검
- 모든 공개 함수를 크기마다 scipy와 견주어 느린 곳을 고치고, 메시지·시그니처의 영역 간 일관성을 점검하고, 코드 리뷰를 한다.
- 결정(추천안, 사용자 위임 2026-11-22): docs/design/integrate.md §1.1 C108–C116(`newton_cotes`의 int 기억; `fixed_quad`의 n ≤ 10⁴와 `newton_cotes`의 N ≤ 256(처음 계산이 n²·N³에 비례: 시간의 상한); 원소별 무리의 병렬 기준을 원소 × 점 2¹⁵개로, 점의 사본 없는 호출; `approx_fprime`의 정확한 차분은 처음 열의 반올림 한계가 2⁻³⁶을 넘으면 간격을 키운다; 측정은 1라운드·기준선에 올리지 않음; 남긴 것; 일관성 점검; 코드 리뷰는 리뷰 세션).
- 결과: csrc/elementwise/loop.hpp(`for_weighted`·`advance_weighted`, `CallT<Keep>`과 `CallPoints`; optimize의 `for_elements`·`advance`·`Call`은 그대로), csrc/integrate/tanhsinh.cpp·csrc/differentiate/derivative.cpp(그것을 쓴다), src/stardust/integrate/_rules.py(`newton_cotes`의 int 기억, 두 상한), src/stardust/differentiate/_differences.py(`_GROW_TOL`), 속도 사례 34 → 50(bench/speed/cases/integrate.py·differentiate.py), 표 bench/integrate/speed.md·bench/differentiate/speed.md(실행 파일에서; 기준선 없음), 테스트(점 수로 나누는 병렬의 1·3·10스레드와 원소 혼자, 상한, 기억한 규칙의 사본, exp의 3×10⁻⁶–10⁻⁴), 골든 5케이스 더함, regression 골든 둘 다시 기록(`nls ENSO/start2/bounded/diff`, `nls MGH17/start1/bounded/diff`; C112).
- 확인: 골든은 모든 영역이 1·3·10스레드에서 그대로다(바꾼 둘과 새 다섯 빼고). A/B의 모든 결과 해시가 두 빌드와 1·10스레드에서 같다. `tools/test.py integrate differentiate optimize regression`과 tests/bench가 지난다.
- 찾은 것(docs/design/integrate.md §7.14): 원소별 무리의 10스레드가 원소 10³–10⁴개에서 3–8배 빨라졌다(`tanhsinh` 10³원소 0.37 → 0.13 ms, 10⁵ 15.8 → 5.4 ms; `nsum` 배치 10³ 200 → 65 ms). `approx_fprime`은 x = 3×10⁻⁶–10⁻³에서 상대 1.5×10⁻⁸까지 틀렸다(지금 2.5×10⁻¹¹ 이하). 남은 sd가 느린 줄은 함숫값 21개의 `quad`(0.95배)·QAWS의 가장 작은 경우(0.96배)·`nsum` 배치의 1스레드(0.93배: 사용자 함수의 sd `**`)다. 일관성에서 영역 안에 고칠 것은 없었고, 다른 영역에서 본 셋(optimize의 원소별 입력 글 둘, `OptimizeResult`의 `repr` 순서, sd `**`의 1스레드)을 적었다. 코드 리뷰는 리뷰 세션에 남겼다(C116).
- 그 뒤의 상미분방정식 묶음(`solve_ivp`, `odeint`, `solve_bvp`와 풀이 클래스)은 큰 결정을 그때 묻는다.

# 공간 영역 (sd.spatial)

> 상태: **완료** (SPT0–SPT4, SPT6, SPT7, 2026-11-23; 승인(위임)). SPT5(BallTree)는 건너뛰었고, `haversine`의 이웃은 단위 구 위의 KD 트리가 맡는다(SPT83, 사용자 결정 2026-11-23). 설계는 [docs/design/spatial.md](docs/design/spatial.md)(확정), API는 [docs/api/spatial.md](docs/api/spatial.md)(확정)다.

## 진행 원칙
- 다른 영역과 같다: 마일스톤마다 시작 전에 물을 것(docs/design/spatial.md §9)을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. 해설과 안내서는 쓰지 않는다.
- 큰 결정(사용자, 2026-11-22): 첫 판은 거리(`pdist`, `cdist`, `squareform`과 거리 함수), 전체 행렬을 만들지 않는 축약, 정확한 이웃 찾기(`KDTree`와 한 번에 끝나는 kNN·반경 질의); 유클리드는 GEMM을 후보를 고르는 데만 쓰고 돌려주는 거리와 경계의 판정은 직접 계산; 반경 결과는 CSR 꼴의 결과 객체와 `.tocsr()`; scipy.spatial의 이름과 (거리, 색인)의 동점 순서; 최상위 평평한 `sd.spatial` (docs/design/spatial.md §1.1 SPT1–SPT5). §9의 질문 스물은 추천안으로 정했다(SPT6–SPT25, 사용자 위임 2026-11-22).
- 비교 대상은 scipy 1.18.1의 `scipy.spatial`, scikit-learn 1.9.1의 `sklearn.neighbors`·`sklearn.metrics.pairwise`, R 4.6.2의 `stats::dist`이고, 정확도 체계(`bench/accuracy`)와 속도 체계(`bench/speed`)에서 견준다. 참값은 Arb다.
- 기존 영역의 결과는 비트까지 같아야 한다(이 영역은 다른 영역을 고치지 않는다).

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| SPT0 준비: API 문서, 영역 등록, 시험 점, 참값(Arb)·scipy·sklearn·R 기록, §3.2의 상수와 증명, 짝 커널 두 순서와 GEMM 동시 호출의 측정, §5의 다시 잰 시간 | 완료·승인(위임, 2026-11-22) |
| SPT1 거리: `pdist cdist squareform`, 거리 함수 열넷(`haversine` 포함), `w`·`V`·`VI`, 배치, bool의 비트 묶음 | 완료·승인(위임, 2026-11-22) |
| SPT2 축약과 전수 이웃: `kneighbors radius_neighbors`, `QueryResult`, `RadiusResult`와 `tocsr()`, GEMM 고르기 | 완료·승인(위임, 2026-11-22) |
| SPT3 KD 트리와 kNN: `KDTree`, `query` | 완료·승인(위임, 2026-11-22) |
| SPT4 반경·짝·개수: `query_ball_point query_ball_tree query_pairs count_neighbors sparse_distance_matrix` | 완료·승인(위임, 2026-11-22) |
| SPT5 BallTree | 건너뜀(SPT8: 첫 판에 넣지 않는다, 나중 후보) |
| SPT6 성능과 일관성 점검 | 완료·승인(위임, 2026-11-23) |
| SPT7 문서(와 `haversine` 이웃의 트리 길) | 완료·승인(위임, 2026-11-23) |

## 마일스톤

### SPT0. 준비
- 결정(추천안, 사용자 위임 2026-11-22): docs/design/spatial.md §1.1 SPT26–SPT28(타일 순서의 꼴: 32항 덩어리, BLAS 없음; 신중한 경로는 2의 거듭제곱으로 비율을 맞춤; GEMM 고르기는 n ≤ 2²¹ − 2에서 정확한 상수로). SPT22는 측정에 따라 타일 순서로 바뀌었다.
- 결과: docs/api/spatial.md(계획한 공개 이름 20개와 `KDTree`의 메서드·속성, 결과 객체의 필드, scipy·sklearn → sd.spatial 대응표), src/stardust/spatial/(`sd.spatial`, 공개 이름은 아직 없다), csrc/spatial/pair_kernel.hpp(짝 커널 두 순서의 시제품), 시험 점 tests/spatial/points.py(점 집합 49개, 가장자리 짝 27, 가중치·분산·`VI`·haversine 짝의 생성기), 참값 tools/spatial_ref.py(Arb·정확한 유리수 4묶음 1,065점 → tests/spatial/ref/), 탐침 tools/spatial_probe.py·tools/spatial_probe_kernels.cpp(짝 커널, `cblas_ddot`의 주소, GEMM 동시 호출, `pow`의 오차, scipy·sklearn·R의 기록 → tests/spatial/data/, §5의 시간), 측정 등록 bench/areas.py·bench/speed/cases/spatial.py(마일스톤마다의 사례 계획 27개)·bench/accuracy/spatial.py·tests/golden/cases_spatial.py(아직 비었다), 테스트 tests/spatial/(test_api_surface.py, test_reference_data.py).
- 찾은 것(docs/design/spatial.md §3.1, §3.2, §3.5, §5): `vector_norm`과 같은 순서의 묶음 커널은 그 비트를 모든 길이에서 지켰지만 타일 순서보다 n ≥ 16에서 1.34–2.68배, BLAS 구간에서 3.85배 느려 타일 순서로 바꿨다(n < 16에서는 두 순서가 같다). sklearn의 전개식은 1e8 + [0, 1)³에서 거리를 모두 0으로 내고 가장 가까운 점을 하나도 맞히지 못했다. scipy의 `KDTree`와 `cdist`는 같은 짝의 거리가 n = 3에서 12%, n = 3000에서 96% 다르고, 제곱의 자리에서 하는 반경 판정은 경계의 23%에서 √의 자리와 갈렸다. scipy의 `k=[0]`은 신호 11로 프로세스를 죽인다. GEMM을 여러 스레드에서 동시에 부르면 합한 속도가 4–6스레드에서 1스레드의 약 2.2배로 찬다. libm의 `pow`는 정상 범위에서 0.504 ULP 이하였다. scipy와 sklearn의 시간은 처음 초안과 3–10% 안에서 같았다.

### SPT1. 거리
- `pdist cdist squareform`과 거리 함수 열넷, 타일 순서의 짝 커널(`metric.hpp`), `haversine`의 식 고르기(Arb), `cosine`의 두 안 재기(docs/design/spatial.md §3.1).
- 결정(추천안, 사용자 위임 2026-11-22): docs/design/spatial.md §1.1 SPT29–SPT36(`cosine`·`correlation`은 정규화한 행의 ‖x̂ − ŷ‖²/2; `minkowski`의 근에 뉴턴 한 걸음과 최댓값으로 나누는 신중한 경로; `haversine`은 2·atan2(√h, √g)에 g를 상쇄 없는 꼴로; 거리마다의 정확도 계약과 2⁻¹⁰⁷⁴의 바닥; 신중한 경로의 문턱 2⁻⁹⁶⁸과 `cosine`·`correlation` 행의 2ᵏ 비율; NaN 자료의 기본 `VI`는 NaN; `hamming`·`jaccard`의 두 dtype 맞추기; SPT0 시제품을 tools/로 옮김).
- 결과: csrc/spatial/metric.hpp(연산마다 NEON과 스칼라의 같은 연산, `tile<Op, R, W>`와 `pair_scalar`, 신중한 경로, `minkowski`·`mahalanobis`·`haversine`, 행의 준비와 비트 묶음), csrc/spatial/distance.cpp(`_pdist _cdist _paired _squareform_*`, 크기로만 나누는 태스크, 두 짝씩의 마무리), csrc/bind/spatial_bind.cpp, src/stardust/spatial/(`_common.py`, `_distance.py`; 공개 이름 17), tests/spatial/(test_distance·test_pair_kernel·test_squareform·test_reference·test_errors·test_threads), 골든 tests/golden/cases_spatial.py(235케이스, 1·3·10스레드 같음), 정확도 bench/accuracy/spatial.py(distances·counting·haversine 28영역, 모두 계약 안; 기준선 bench/accuracy/results/spatial.json, 표 bench/spatial/accuracy.md), 속도 사례 bench/speed/cases/spatial.py(SPT1 20개, `--smoke`로 확인; 전체 측정은 하지 않았다).
- 찾은 것(docs/design/spatial.md §4, §5): 참값의 모든 짝에서 계약의 0.68 이하(`chebyshev`·가중치 없는 셈은 올바른 반올림), 최대 ULP는 `euclidean` 2, `braycurtis` 7(가중), `minkowski` 6(p = 0.5), `haversine` 2(대척점 포함). scipy는 1e160·1e-160·비정상 수의 짝에서 inf·0·NaN을 내고, `cosine`의 1 − c는 거의 평행한 짝에서 모든 자릿수를 잃는다(d ≈ 5×10⁻¹⁷에서 상대 6.2; sd 4×10⁻⁹). 1스레드에서 scipy보다 `cdist` euclidean n = 64가 2.1배, n = 512가 2.6배, bool `hamming`이 55배 빠르고, `minkowski`(9% 느림)·`haversine`(sklearn보다 약 3배 느림)·`mahalanobis`(같음)는 SPT6에서 다시 본다.

### SPT2. 축약과 전수 이웃
- `kneighbors radius_neighbors`, 결과 객체, §3.2의 GEMM 고르기(중심 빼기, 정리 3의 상수), 문턱 n₀.
- 결정(추천안, 사용자 위임 2026-11-22): docs/design/spatial.md §1.1 SPT37–SPT49(`kneighbors`는 d̂ < 상한인 짝만이라 inf·NaN 거리는 나오지 않음; 전수 길은 `cdist`의 구동기를 블록마다(`CdistBlocks`); L̃는 §3.2의 꼴을 NEON 두 레인으로; GEMM 고르기는 12 ≤ n ≤ 2¹⁶, mB ≥ 64(n₀ = 12); 첫 후보는 L̃가 작은 것을 작은 힙 한 번 훑기로; `XA` 조각과 `XB` 구간으로 나누고 (d̂, j)로 합침; 중심 빼기는 언제나; `RadiusResult`는 `_radius.py`, 검사·복사 없는 생성자; 큰 k는 min(k, mB)만 찾고 채움; `r`은 실수나 (mA,)로 브로드캐스트되는 배열; `finite()`·`radius()`를 `_common.py`에; 반경 결과는 태스크마다 모아 복사; `XA`는 2차원만).
- 결과: csrc/spatial/neighbors.hpp·neighbors.cpp(`_kneighbors`, `_radius_neighbors`, 힙, 나누기, GEMM 고르기), csrc/spatial/distance.cpp에 `CdistBlocks`(블록마다의 `cdist`), src/stardust/spatial/_neighbors.py·_radius.py(`RadiusResult`)와 `_common.py`의 검사 둘, tests/spatial/(test_neighbors·test_neighbors_reference·test_neighbors_errors·test_neighbors_threads, 93개), 골든 tests/golden/cases_spatial.py에 50케이스(1·3·10스레드 같음; 기록은 SPT3과 함께 다시 한다), 정확도 bench/accuracy/spatial.py의 `neighbors`(36영역, sd는 모두 계약 안: 격자는 Arb의 순서와 같고 일반 집합은 계약의 0.27 이하), 속도 사례 bench/speed/cases/spatial.py(SPT2 10개, `--smoke`로 확인).
- 찾은 것(docs/design/spatial.md §3.2, §5): 정리 3의 L̃는 모든 시험(적대적 집합 열셋, hypothesis 군집 1,000개, 격자, 1 ULP 껍질)에서 답을 바꾸지 않았다. 중심 빼기로 1e8 + [0, 1)¹⁶의 먼 군집도 원점 근처와 같은 시간이다. 처음 판은 첫 후보를 `nth_element`로 골라 k = 1이 4배 느렸다(SPT41로 고침). 10스레드에서 모든 사례가 sklearn보다 빠르고(4000² n = 64 k = 1 4.9 대 8.6 ms, n = 512 23.0 대 40.0 ms), 1스레드의 n = 512 k = 10은 sklearn의 1.8배 느리다(SPT6에서 다시 본다). scipy·sklearn은 격자의 동점 순서가 Arb의 (RN(d), 색인)과 다르고, sklearn의 전개식은 far/1e8·far/1e15·shell/3에서 k번째 거리가 참값과 10¹⁸ ULP 넘게 다르고 nn/mixture8에서 계약의 114배다.

### SPT3. KD 트리와 kNN
- `KDTree`(만들기, 속성, 피클)와 `query`(k, k 시퀀스, 상한, p), 트리 대 전수의 길.
- 결정(추천안, 사용자 위임 2026-11-22): docs/design/spatial.md §1.1 SPT50–SPT62(폭이 0인 노드도 나눠 노드 수가 n과 `leafsize`로만 정해지고 잎은 색인 순서; 가지치기의 한계는 상자의 가장 가까운 점까지의 같은 커널 거리와 여유 c = 1 + 4(m + 12)u·η = 2⁻¹⁰⁷⁰, p = inf는 그대로; p = 2는 제곱합으로 먼저 거름; 하위 트리 태스크와 행 옮기기의 병렬 만들기; 질의 64점 태스크; `k` 시퀀스는 순서·중복 자유; 피클은 데이터 바이트로 다시 만듦; `data`는 Python이 쥐는 입력 순서 사본; `k`·`p`·상한의 검사를 `_common.py`에(`kneighbors`와 공용); 전수 길의 자리 `_search`; `repr`; 빈 트리; 질의점의 검사).
- 결과: csrc/spatial/kdtree.hpp(트리의 꼴, 거리 정책, 가지치기 문턱과 그 증명, `KdSearch`)·kdtree.cpp(만들기, `_KDTree` 바인딩), src/stardust/spatial/_kdtree.py(`KDTree`; 공개 이름에 더함)와 `_common.py`의 검사 넷, tests/spatial/(test_kdtree·test_kdtree_query, 46개), 골든 tests/golden/cases_spatial.py에 46케이스(1·3·10스레드 같음; 기록은 SPT2와 함께 다시 한다), 정확도 bench/accuracy/spatial.py의 `kdtree`(36영역, sd는 모두 계약 안: 격자는 Arb의 (RN(d), 색인) 순서와 같고 일반 집합은 계약의 0.27 이하), 속도 사례 bench/speed/cases/spatial.py(SPT3 6개, `--smoke`로 확인).
- 찾은 것(docs/design/spatial.md §3.3, §3.4, §5): 트리의 결과는 모든 시험(무작위, 중복, 동점, 격자, 군집, 1e8의 먼 군집, 1e±200, 비정상 수, DBL_MAX 둘레, m = 64, hypothesis)에서 `cdist`로 만든 전수와 색인·거리의 비트까지 같았다. 10스레드에서 만들기(10⁶점 79 ms 대 scipy 260 ms)와 질의(m = 3 k = 10 3.25 ms 대 scipy `workers=-1` 5.36 ms, m = 16 k = 10 546 대 1166 ms) 모두 scipy보다 빠르고, 1스레드는 만들기가 scipy와 비슷하고(225–235 대 230–260 ms) 질의는 1.1–2.4배 빠르다. 10스레드 만들기는 위 단계의 `nth_element`가 직렬이라 3배에 머문다(SPT6의 후보). scipy·sklearn은 격자의 동점 순서가 Arb의 (RN(d), 색인)과 다르다. `sd.ndarray`와 희소 배열이 피클되지 않는다(다른 영역에서 찾은 것).

### SPT4. 반경·짝·개수
- `query_ball_point query_ball_tree query_pairs count_neighbors sparse_distance_matrix`, 가중 셈의 고정된 순서(SPT25).
- 결정(추천안, 사용자 위임 2026-11-22): docs/design/spatial.md §1.1 SPT63–SPT70(두 상자 사이의 가까운 한계와 먼 한계를 상자 모서리의 짝 커널 거리로, 먼 한계는 SPT51의 여유 그대로 증명; 이중 트리는 깊이 4의 조각 짝이 태스크, `query_pairs`는 a ≤ b; `count_neighbors`는 정렬한 반경의 첫 칸 셈(delta)과 Gray–Moore의 한꺼번에 더하기, 반경이 많으면 얕게; `count_neighbors`의 r과 무게의 꼴(NaN·음수 r, NaN 무게는 에러, `(None, None)`은 정수); `query_ball_point`의 r은 배치 모양을 바꾸지 않는 브로드캐스트; 큰 목록은 찾는 중에 세어 `MemoryError`; 잎 쌍의 점마다 상자 검사는 재서 느려 넣지 않음; 파일 `query.cpp`·`dual.cpp`와 `kdtree.hpp`의 `KdPair`).
- 결과: csrc/spatial/kdtree.hpp(`KdPair`와 먼 한계의 증명, `kd_distance`·`kd_beyond`, 거리 정책의 `below`), csrc/spatial/query.cpp(`_ball_point`), csrc/spatial/dual.cpp(`_ball_tree`, `_pairs`, `_count`), src/stardust/spatial/_kdtree.py(메서드 다섯)와 `_common.py`의 검사 셋, tests/spatial/(test_kdtree_radius·test_kdtree_pairs, 59개; test_api_surface의 계획 목록을 지움), 골든 tests/golden/cases_spatial.py에 96케이스(1·3·10스레드 같음; 기록은 메인 세션이 다시 한다), 정확도 bench/accuracy/spatial.py의 `ball`(36영역, sd는 모두 계약 안), 속도 사례 bench/speed/cases/spatial.py(SPT4 8개, `--smoke`로 확인).
- 찾은 것(docs/design/spatial.md §3.4, §5, §7.5): 모든 결과가 모든 시험(무작위, 중복, 동점, 격자, 군집, 먼 군집, 1e±200, 비정상 수, DBL_MAX 둘레, m = 64, 경계 반경, hypothesis, 400개의 적대적 무작위 시험)에서 `cdist`·`pdist`·`radius_neighbors`로 만든 전수와 바이트까지(셈은 정확히) 같았다. 10스레드에서 모든 사례가 scipy보다 빠르고(`query_ball_point` 3.1 대 `workers=-1` 7.1 ms, `count_neighbors` r 20개 35 대 349 ms), 1스레드는 `query_pairs`가 scipy의 `ndarray` 꼴보다 1.17배 느린 것을 빼고 빠르다. scipy의 `query_pairs(output_type='ndarray')`는 정렬되어 있지 않다(설계 문서의 "정렬된 배열"을 고쳤다). scipy와 sklearn은 `euclidean` 격자에서 거리가 꼭 r인 점을 질의 110개 중 28개에서 빠뜨린다(r²과 견줌). scipy의 `count_neighbors`는 r = NaN에서 모든 짝을 센다.

### SPT5. BallTree (건너뜀)
- SPT8로 첫 판에 넣지 않는다. SPT6에서 KD 트리가 칠 수 없는 거리의 전수 시간을 재 보고 다시 본다.

### SPT6. 성능과 일관성 점검 — 완료·승인(위임, 2026-11-23)
- 모든 공개 함수를 크기마다 scipy·sklearn·R과 견주고, 길의 문턱(n₀, 트리 대 전수, `leafsize`, 블록 크기)을 정하고, 메시지·시그니처의 영역 간 일관성과 코드 리뷰를 한다.
- 결정(추천안, 사용자 위임 2026-11-23): docs/design/spatial.md §1.1 SPT71–SPT82(GEMM 고르기의 블록 공통 한계와 옮긴 `XB`의 공유; 첫 후보는 Ê로, 후보가 많으면 L̃가 작은 k개 먼저; `kneighbors`·`radius_neighbors`의 KD 트리 길과 문턱; GEMM 문턱 1 ≤ n, mA ≥ 64, mB ≥ 2048(n ≥ 8이면 1024); `query`의 전수 길(p = 2, 5m/6 + log₂(k)/4 + 7 − log₂ n ≥ 0.5); `KDTree.data`는 처음 읽을 때 만듦; `query_pairs`의 병렬 모으기; `leafsize` 10 그대로; 트리 위 단계의 수준별 병렬과 표본 띠 선택; `minkowski` p = 3은 (a·a)·a(c_pow 2u → γ₂); `haversine`은 asin 두 갈래; 비트가 같은 빠른 커널(긴 짝의 덩어리 나란히, `mahalanobis` 8짝 레인)).
- 결과: csrc/spatial/metric.hpp·distance.cpp·neighbors.cpp·kdtree.cpp·dual.cpp, src/stardust/spatial/_neighbors.py·_kdtree.py·_common.py, tests/spatial/(트리 길·먼 점부터 정렬·전수 `query`·표본 띠의 테스트, 경로 `path=3`), tests/spatial/refdata.py의 c_pow.
- 찾은 것(docs/design/spatial.md §5): 1스레드에서 `kneighbors` 4000² n = 3 k = 1이 9.2배, `query` m = 16 k = 10이 10배, `mahalanobis`가 3.8–4배, `minkowski` p = 3이 8.6배, 10⁶점 만들기의 10스레드가 1.9배 빨라졌고 결과는 같다(골든은 p = 3 `minkowski` 다섯과 `haversine` 셋이 1 ULP 바뀜). `haversine`의 kNN은 sklearn `BallTree`보다 30배 넘게 느려 SPT5(BallTree)를 열지 사용자에게 묻는다.

### SPT7. 문서 — 완료·승인(위임, 2026-11-23)
- docs/api/spatial.md의 확정과 설계 문서를 구현대로 고치기. 그 앞에 SPT6이 물은 `haversine`의 이웃 찾기를 사용자의 결정대로 했다.
- 결정: docs/design/spatial.md §1.1 SPT83(사용자 결정 2026-11-23: (위도, 경도)를 단위 구 위의 3차원 점으로 옮겨 기존 KD 트리가 현 거리로 후보를 고르고, 돌려주는 거리와 모든 판정은 `haversine` 커널), SPT84–SPT89(추천안, 사용자 위임 2026-11-23: 증명한 오차의 4배 넘는 여유 cut(T); 여유의 정의역 |위도| ≤ fl(π/2), |경도| ≤ 8, 밖은 전수; 잎의 점은 3차원 제곱합으로 먼저 거르고 cut은 T가 바뀔 때만; 문턱 mB ≥ 128, mA ≥ 16, 8k ≤ mB와 4–64점의 질의 조각; `haversine` 커널의 경도 차를 TwoSum으로 정확히, h의 언더플로는 2⁶⁰⁰배로; 파일 `sphere.cpp`와 참값 `sphere.npz`).
- 결과: csrc/spatial/sphere.cpp(`_sphere_kneighbors`, `_sphere_radius`, 정리 4의 주석), csrc/spatial/metric.hpp(`haversine_pair`의 경도 차와 신중한 경로), src/stardust/spatial/_neighbors.py(길 고르기), 참값 tests/spatial/points.py의 `sphere` 집합 여섯과 tools/spatial_ref.py의 `sphere` 묶음(tests/spatial/ref/sphere.npz, 120점), 테스트 tests/spatial/test_neighbors_haversine.py(16개)와 test_reference_data.py, 정확도 bench/accuracy/spatial.py의 `sphere`(6영역, 모두 계약 안, 최대 0.096), 속도 사례 bench/speed/cases/spatial.py에 둘(`--smoke`로 확인; sklearn `BallTree`와 결과가 같다), 골든 다섯을 더함(바뀐 셋과 함께 기록은 메인 세션이 다시 한다), docs/api/spatial.md 확정(상태, `haversine`의 식과 트리 길, `leafsize`, 대응표, 정확도), docs/design/spatial.md 확정(상태, §3.4의 증명, §4–§8의 잰 값·파일·시험).
- 찾은 것(docs/design/spatial.md §3.4, §4, §5): 10⁴ × 10⁵ k = 5가 10스레드 3.61초 → 6.9 ms, 1스레드 26.1초 → 27.9 ms가 되어 sklearn `BallTree`(98 ms)보다 10스레드 14배, 1스레드 3.5배 빠르다(반경 r = 0.01은 6.7 대 47.4 ms). 여유를 상대 2⁻⁵⁰만 모자라게 하면 적대적 시험 여섯이 실패한다(여유가 꼭 필요하다). 기존 `haversine` 커널은 날짜 변경선을 사이에 둔 가까운 짝에서 상대 uπ/d를 잃었고(d = 10⁻¹¹에서 계약의 2×10⁹배; sklearn은 d = 10⁻¹³에서 상대 4×10⁻³) 10⁻³⁰⁰ 둘레 좌표의 짝에서는 0을 냈다(SPT88로 고침; 고친 뒤 적대적 집합에서 계약의 0.20, 참값 파일에서 1 ULP). sklearn `BallTree`는 극의 집합에서 계약의 3.8×10¹⁴배다.

---

# 그림 영역 (sd.plot)

> 상태: **완료** (PL0–PL9, 2026-11-23; 사용자 승인). 기본 그림, 함수 그림, 통계 그림, 3D에 더해 디자인 손질(기본 색 jco·cool, 범례 자리, 2D와 3D의 배치, MuPAD 꼴의 조명), 반투명 곡면, 적응형 표본과 그 조절 인자, 카메라의 자동 선택까지 들어갔다(docs/design/plot.md PL38, PL41–PL47). 설계는 [docs/design/plot.md](docs/design/plot.md), API는 [docs/api/plot.md](docs/api/plot.md)다. 커밋은 c6eb9ab, 74d6118, d408a9b, f80ddd4다.

## 진행 원칙
- 큰 결정(사용자, 2026-11-22): 수학 함수의 그림과 논문 그림의 두 갈래를 한 바닥 위에; 자체 렌더러; 함수가 `Figure`를 돌려주는 꼴; 모든 글자는 LaTeX; 글꼴 두 벌; 3D는 깊이 버퍼의 그림에 벡터 축; 출력 PDF·SVG·PNG; 상호작용 없음; 성능은 노트북에서 그림 한 장의 반응 시간으로 잰다 (docs/design/plot.md §1.1 PL1–PL17).
- 그림에는 참값이 없다. 검증은 장면과 픽셀의 골든, 성질 테스트, 다른 렌더러와의 견줌이고(docs/design/plot.md §8), 예쁜지는 사용자가 본보기 그림으로 본다.
- 사용자가 남은 결정을 위임하고 끝까지 진행하라고 했다(2026-11-22). 기본 색(범주 팔레트 jco, 연속 색 cool)은 후보 그림을 보고 사용자가 골랐고, 3D의 빛과 색은 사용자가 든 본보기(MuPAD의 곡면)에 맞췄다.

## 진행 상태
| 마일스톤 | 상태 |
|---|---|
| PL0 준비: 글자 길의 탐침, API 문서 초안 | 완료 |
| PL1 장면과 글자: 장면, SVG, LaTeX 두 길, 글리프 캐시, 숫자 늘어놓기, 영역 등록 | 완료 |
| PL2 래스터와 PDF: C++ 2D 래스터라이저, PNG, PDF | 완료 |
| PL3 축과 배치: 눈금, 배치, 범례, 테마, 색 | 완료 |
| PL4 기본 그림: `line scatter bar area step errorbar hline vline text annotate`, `grid` | 완료 |
| PL5 함수 그림: `function parametric polar implicit region contour density phase` | 완료 |
| PL6 통계 그림과 결과 객체: `histogram ecdf boxplot strip qq heatmap pairs`, `compare fit diagnostics coefficients distribution` | 완료 |
| PL7 3D: `surface parametric_surface curve3d scatter3d` | 완료 |
| PL8 성능과 일관성 점검: 큰 자료, 병렬 점 찍기, 속도 사례 | 완료·승인 |
| PL9 문서: API 문서의 확정 | 완료 (2026-11-23) |

