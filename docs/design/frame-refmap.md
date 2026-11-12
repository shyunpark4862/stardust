# 옛 문서 참조 → frame.md (임시)

> 코드와 테스트의 옛 문서 참조를 `docs/design/frame.md`로 바꾸는 데 쓰는 표다. 바꾸기가 끝나면 지운다 (ROADMAP M7).
>
> - 옛 문서: `stardust-pandas/DESIGN.md`, `stardust-pandas/ARCHITECTURE.md`, `stardust-pandas/docs/learn/M*.md` (동결된 옛 저장소)
> - "§N"은 `frame.md §N`이다. 여러 개면 첫 번째가 주된 자리다. 다른 문서는 "array.md §N", "루트 DESIGN §N"처럼 적는다.
> - 옛 `API.md §N` 참조와 API 규칙 ID(`S1`–`S8`, `C1`–`C7`, `A6`)는 이 표의 대상이 아니다. `docs/api/frame-refmap.md`가 다룬다.
> - **인용 수**: HEAD(M4 커밋)의 `legacy/pandas/csrc`, `legacy/pandas/src`, `legacy/pandas/tests`, `legacy/pandas/CMakeLists.txt`에서 grep으로 셌다. 이 `csrc`는 옛 저장소의 `csrc`와 파일마다 같다(확인함). M5에서 옮기는 중인 `csrc/frame/`에서는 일부가 이미 바뀌었다.
>   - 한 주석이 여러 절을 인용하면 절마다 하나로 셌다 (`DESIGN.md §2.5, R3`는 §2.5에 하나, R3에 하나).
>   - M5에서 옮기지 않고 지운 파일의 몫은 따로 적는다: `core/buffer.{hpp,cpp}`(공용 `base/buffer.*`로 대체), `parallel/pool.cpp`(공용 풀로 대체), `tests/test_build.py`(루트 `tests/test_build.py`로 합침). `parallel/pool.hpp`는 `frame/parallel.hpp`로 옮겼다. `legacy/pandas/CMakeLists.txt`는 루트 `CMakeLists.txt`가 대신한다.

## 1. 옛 ARCHITECTURE.md

| 옛 참조 | 내용 | 새 절 | 인용 수 |
|---|---|---|---|
| ARCHITECTURE §0 | 하드웨어 사실 (128B, NEON, 64비트 정수 곱 없음) | array.md §1.1 | 1 (지운 `core/buffer.hpp`) |
| ARCHITECTURE §1 (A1) | Python API 층 + C++ Column과 커널 | §3.1 | 1 (`series.py`) |
| ARCHITECTURE §1.1 | M0 검증, Column은 값 타입 (190 대 89ns, nanobind keep-alive) | §3.2 (A1의 측정은 §3.1) | 2 (`core/column.hpp`, `bind/module.cpp` → `bind/frame_bind.cpp`) |
| ARCHITECTURE §2 (A2, A3) | Column 메모리 구조 | §4 | 0 |
| ARCHITECTURE §2.1 | Buffer: 참조 카운트, 128B | §4.1 (Buffer 자체는 array.md §3.2) | 1 (지운 `core/buffer.hpp`) |
| ARCHITECTURE §2.2 | Column 구성, zero-copy 슬라이스, null 개수 | §4.3 | 1 (`core/column.hpp`) |
| ARCHITECTURE §2.3 | validity 비트맵: LSB 우선, 1은 값, 결측 없으면 없음 | §4.4 | 1 (`core/bitmap.hpp`) |
| ARCHITECTURE §2.4 | NA 자리의 값은 0, 문자열은 길이 0 | §4.5 | 11: `ARCHITECTURE.md §2.4` 10 (`core/convert.cpp` 2, `core/column.cpp` 1, `core/column.hpp` 1, `ops/elementwise.cpp` 2, `ops/selection.cpp` 3, `tests/test_reduce.py` 1) + `ops/reduce.cpp`의 맨 "§2.4" 1 |
| ARCHITECTURE §2.5 | dtype별 값 저장, bool은 1바이트(A3), offsets는 int64 | §4.6 | 1 (`core/column.hpp`) |
| ARCHITECTURE §3 (A4) | DataFrame과 Copy-on-Write | §5 | 0 |
| ARCHITECTURE §3.1 | DataFrame: 이름 → Column 매핑 + Index, 2D 블록 없음 | §5.1 | 1 (`frame.py`) |
| ARCHITECTURE §3.2 | Copy-on-Write: 쓰기 직전에 참조 카운트 | §5.2 | 1 (`ops/selection.hpp`) |
| ARCHITECTURE §3.3 | Copy-on-Write의 구현: `share()`, `use_count()`, 문자열은 새 Buffer | §5.2 (스냅샷은 §5.3) | 0 |
| ARCHITECTURE §4 (A5) | RangeIndex, Index, 조회 엔진(`LabelTable`), CSR | §6.1, §6.2 | 2: `ops/hash.hpp`(엔진) → §6.2, `index.py`(RangeIndex) → §6.1 |
| ARCHITECTURE §5 (A6) | 정렬(alignment): 같으면 위치대로, 다르면 합집합 + `take` | §6.3 | 1 (`_select.py`) |
| A6 (ID만) | 합집합 라벨의 순서 | `docs/api/frame-refmap.md`에서 센다. 구현은 §6.3 | (2, API 표의 몫) |
| ARCHITECTURE §6 (A7) | 커널 체계, 병렬화, 테스트 | §7–§13 | 0 |
| ARCHITECTURE §6.1 | 공통 기본 연산 (take, filter, factorize, 리덕션) | §7.1 | 1 (`ops/selection.hpp`) |
| ARCHITECTURE §6.1.1 | 원소별 연산: `binary_op`, 1행 스칼라, validity | §7.2 | 0 |
| ARCHITECTURE §6.1.2 | factorize: 슬롯 16B, 정규화, NA, 다중 키 | §8.1, §8.2, §8.3 (슬롯), §8.6 | 0 |
| ARCHITECTURE §6.1.3 | 리덕션: 1행 Column, pairwise, int128, FMIN/FMAX | §7.4 | 0 |
| ARCHITECTURE §6.1.4 | groupby: factorize → counting sort → 그룹마다 리덕션, 그룹 정렬 | §9.1, §9.2 | 0 |
| ARCHITECTURE §6.1.5 | 정렬: 정렬 키 변환, LSD, NA 떼어 두기 | §9.3 | 0 |
| ARCHITECTURE §6.1.6 | merge | §9.7 (이 절의 "함께 factorize"는 지금 코드와 다르다, 4절) | 0 |
| ARCHITECTURE §6.1.7 | read_csv | §10 (토크나이저 §10.2, 추론 §10.5) | 0 |
| ARCHITECTURE §6.1.8 | 단일 코어 최적화 | Swiss table §8.3, 문자열 해시 §8.4, 접두사 정렬 §9.4, build/probe §9.7, 바이트 펼치기 §4.4, 필터 §7.3, CSV §10.2–§10.4 | 0 |
| ARCHITECTURE §6.2 | 병렬화: `for_chunks`, `Unlocked`, `FirstRow`, 커널별 분할 | §11 (factorize §8.7, counting sort §9.6, 기수 정렬 §9.5, CSV §10.6, 앨리어싱 §7.6) | 0 |
| ARCHITECTURE §6.3 (첫 번째, M12 이후) | 새 입력 측정, 미리 잡기, 직렬 기수 정렬, 조회 | §12 (§13.2, §8.5, §9.5, §8.8) | 0 |
| ARCHITECTURE §6.3 (두 번째, 테스트) | pytest + hypothesis, pandas 정답 | §13.1 | 0 |
| ARCHITECTURE §6.4 | 벤치마크, 비교 대상 | §13.2 | 0 |

## 2. 옛 DESIGN.md

| 옛 참조 | 내용 | 새 절 | 인용 수 |
|---|---|---|---|
| DESIGN §1.1 (Q1) | C++20, nanobind, Python API | 대응 없음: 루트 DESIGN §1. 경계는 §3.1 | 0 |
| DESIGN §1.2 (Q2) | 독립 프로젝트, 부품(풀)을 복사해 온다 | 대응 없음: 폐기 (루트 DESIGN §2). 풀은 하나다 (§11.1, 루트 ARCHITECTURE §4.1). 주석은 지운다 | 1 (`parallel/pool.hpp` → `frame/parallel.hpp`) |
| DESIGN §1.3 (R1) | 이름, 빌드, Python 3.15, 기기, 비교 기준 | 비교 기준 §13.2. 기본 스레드 수(P코어 10)는 array.md §10.3. 이름과 빌드는 루트 DESIGN §1.1, ARCHITECTURE §5 | 1 (지운 `tests/test_build.py`) |
| DESIGN §1.4 (R2) | dtype Tier 1, Tier 2 | §2.1 (C++는 §4.2). Tier 2는 §1.1 "없는 것" | 2 (`core/column.hpp`, `_dtypes.py`) |
| DESIGN §2.1 (Q3) | 결측 모델, NaN은 값, 결측 없는 fast path | NA와 NaN §2.2, 비트맵이 없는 컬럼 §4.4 | 3: `core/column.hpp`, `ops/hash.cpp` → §4.4. `CMakeLists.txt`(`-ffast-math` 없음) → §1.4, array.md §1.3 |
| DESIGN §2.2 (Q4) | RangeIndex, 단일 Index, MultiIndex 없음 | §6.1, §1.1 | 0 |
| DESIGN §2.3 (Q4) | Copy-on-Write, `inplace=` 없음 | §5.2, §1.1 | 0 |
| DESIGN §2.4 (R2) | 범위 (Tier 1, Tier 2, 제외) | §1.1 | 0 |
| DESIGN §2.5 (R3–R7) | 최적화 포함/제외 | 아래 행들 | 8 |
| (§2.5 factorize 중심) | groupby, merge, unique가 그 위에 | §8.1 | 1 (`ops/hash.hpp` 머리말) |
| (§2.5 해시 테이블, R3) | linear probing 다음 Swiss table, 비교 측정 | §8.3 | 1 (`ops/hash.hpp`의 `LinearTable`) |
| (§2.5 임계값 상수 하나) | | §11.2 (영역마다 하나: §11.1) | 1 (`parallel/pool.hpp` → `frame/parallel.hpp`) |
| (§2.5 Index 조회 엔진 캐시, R6) | 처음 조회할 때 만들고 캐시 | §6.2 | 3 (`index.py`) |
| (§2.5 제외: 데이터를 보고 경로를 고르는 휴리스틱) | 정렬된 라벨의 이진 탐색 없음 | §1.3, §6.4 | 2 (`_select.py`, `tests/test_indexing.py`) |
| (§2.5 groupby의 그룹별 pairwise 합) | Series.sum()과 비트까지 같다 | §9.1 | 0 |
| (§2.5 join, 정렬, 문자열 R4, CSV, 필터) | | §9.7, §9.3, §4.6, §10, §7.3 | 0 |
| (§2.5 멀티스레딩: GIL, 스냅샷, 256 파티션) | | §11.4, §5.3, §8.7 | 0 |
| (§2.5 제외 목록: BlockManager, lazy, JIT, 인터닝 …) | | §1.3 | 0 |
| DESIGN §2.6 (R7) | 결정성, 예외(factorize, counting sort, CSV는 스레드 수를 따른다) | §11.5 | 8: `parallel/pool.hpp` 3 → `frame/parallel.hpp`, `ops/radix.hpp` 2, `ops/hash.hpp` 1, `bind/module.cpp` 1(`set_num_threads`, 지금은 `sd.set_num_threads`라 그 바인딩이 없다), `tests/test_threads.py` 1 |
| R3 (ID만) | linear probing으로 먼저 만들고 Swiss table과 잰다 | §8.3 | 1 (`ops/hash.hpp`) |
| DESIGN §3 | 미확정 | 대응 없음 (이력) | 0 |

## 3. 마일스톤 표시
DESIGN.md §5.4에 따라 주석의 마일스톤 번호는 지운다. 설계 사실이 있는 곳은 그 절을 가리키고, 이력뿐인 곳은 "지운다"다.

- 테스트 파일 첫 줄의 `"""Mn: …` 접두어는 모두 이력이다. `Mn:`을 지우고 API 참조만 남긴다.
- "X (Mn)", "Mn: …"처럼 만든 때만 적은 괄호는 지운다. "Mn used …", "Mn's …"처럼 **기각한 대안이나 바꾸기 전의 모습**을 말하는 곳은 마일스톤 대신 그 대안을 적은 절을 가리킨다.

| 표시 | 인용 수 | 있는 곳 → 새 절 |
|---|---|---|
| M0 | 2 | `bind/module.cpp`(복사한 풀의 점검 함수, 이미 옮기지 않았다), 지운 `tests/test_build.py` → 지운다 |
| M1 | 2 | `tests/test_construction.py`, `tests/test_repr.py` 첫 줄 → 지운다 |
| M1-M3 | 1 | `tests/test_indexing.py` ("the M1-M3 provisional types") → 지운다. "한때 임시였던 반환 타입(list, dict)"처럼 내용으로 적는다 |
| M2 | 2 | `ops/parse.hpp` 머리말("astype (M2)"), `tests/test_elementwise.py` 첫 줄 → 지운다 (문법 공유는 §7.5) |
| M3 | 4 | `ops/hash.hpp` 머리말 → 지운다. "M3 used FNV-1a" → §8.4. "linear probing mispredicts (M3)" → §8.3. `tests/test_unique.py` 첫 줄 → 지운다 |
| M4 | 3 | 지운 `core/buffer.hpp`("Copy-on-Write (M4)") → §5.2. `ops/selection.hpp`("M4's alignment uses it") → §6.3. `tests/test_indexing.py` 첫 줄 → 지운다 |
| M5 | 6 | `ops/groupby.cpp` 2("like M5", "as in M5") → §7.4. `ops/pairwise.hpp`("reductions of M5") → §7.4. `ops/groupby.hpp`("as M5 reduces a whole column") → §9.1. `tests/test_groupby.py`("the oracle is M5 itself") → §9.1, §13.1: "그 그룹의 행만 고른 Series의 리덕션"으로 적는다. `tests/test_reduce.py` 첫 줄 → 지운다 |
| M6 | 3 | `ops/pairwise.hpp`("grouped ones of M6") → §9.1. `ops/groupby.hpp` 머리말, `tests/test_groupby.py` 첫 줄 → 지운다 |
| M7 | 3 | `ops/radix.hpp`, `ops/sort.hpp` 머리말 → 지운다 (§9.3). `tests/test_sort.py` 첫 줄 → 지운다 |
| M8 | 3 | `ops/join.hpp` 머리말 → 지운다. `ops/join.cpp`("M8 factorized both sides together") → §9.7 (기각한 대안). `tests/test_merge.py` 첫 줄 → 지운다 |
| M9 | 5 | `io/csv.cpp`("M9 counted them for every record") → §10.3. `io/csv.hpp`, `ops/parse.hpp` 머리말 → 지운다. `tests/test_threads.py`("the M9 oracle") → §13.1: "test_csv의 직렬 정답 구현"으로. `tests/test_csv.py` 첫 줄 → 지운다 |
| M10 | 31 | 아래 표 |
| "M10 explainer §6" | 1 | `ops/selection.cpp`(문자열 압축 두 번 실패) → §7.3 |
| M11 | 16 | 아래 표 |
| M12 | 4 | `ops/radix.hpp`(직렬 기수 정렬) → §9.5. `ops/hash.hpp`(미리 잡기) → §8.5. `ops/hash.cpp`(`find_rows`) → §8.8. `tests/test_api_surface.py` 첫 줄 → 지운다 |
| "M12+ review" | 10 | `ops/hash.hpp`(첫 `reserve`/`insert`에서만 할당) → §8.3. `index.py`, `tests/test_indexing.py` 1(RangeIndex의 `in`은 산술) → §6.1. `_select.py`, `tests/test_indexing.py` 1(중복 라벨 마스크) → §6.3. `groupby.py`, `tests/test_groupby.py`(부른 순간의 컬럼) → §5.3. `tests/test_csv.py`(`\r`) → §10.1. 지운 `parallel/pool.cpp` 2(스레드 수 atomic, 바쁜 풀) → array.md §10.3, §10.1 |

### 3.1 M10 (31)
| 있는 곳 | 내용 | 새 절 |
|---|---|---|
| `core/bitmap.hpp` (`kExpand`), `core/column.hpp` (`valid_bytes`), `core/column.cpp` 1, `ops/reduce.cpp` 1 ("a byte per row"), `ops/selection.cpp` 1 (`expand`) | 비트맵을 바이트로 펼친다 | §4.4 (`finish_fixed`는 §4.5) |
| `core/column.cpp` 1, `ops/elementwise.cpp` 1, `ops/selection.cpp` 1 ("the M10 aliasing problem") | `uint8_t` 앨리어싱, 경계를 지역 변수로 | §7.6 |
| `io/csv.cpp` ("structural index") | 64바이트 비트마스크, PMULL | §10.2 |
| `io/csv.cpp` 3 (16바이트 `Field`, `line_at`, sink) | 한 번에 읽기 | §10.3 |
| `io/csv.hpp` 머리말 ("M10 adds NEON scanning") | 이력. 머리말의 "Single-threaded"도 지금과 다르다 | 지운다. 머리말을 §10으로 다시 쓴다 |
| `ops/radix.hpp` 2 (값의 순서, 접두사 정렬), `ops/groupby.cpp` 2 (고유값의 순서) | 문자열 접두사 기수 정렬 | §9.4 (그룹 순서는 §9.2) |
| `ops/radix.hpp` 1 ("M10's sort runs instead") | 한 번 읽기 히스토그램의 직렬 정렬 | §9.5 |
| `ops/hash.hpp` 2 (Swiss table, "after measuring it against LinearTable") | | §8.3 |
| `ops/hash.hpp` 1 (문자열 해시) | mum + fmix64 | §8.4 |
| `ops/hash.hpp` 1 ("The build side of a hash join"), `ops/join.cpp` 1 | build/probe | §9.7 |
| `ops/selection.hpp`, `_select.py`, `frame.py` | 분기 없는 마스크 압축 | §7.3 |
| `ops/reduce.cpp` 1 ("With NA (M10): blocks …") | NA가 있는 min/max | §7.4 |
| `ops/parse.hpp` 3, `tests/test_elementwise.py` 1 | Eisel-Lemire, 한 패스, SWAR | §10.4 |

### 3.2 M11 (16)
| 있는 곳 | 내용 | 새 절 |
|---|---|---|
| `parallel/pool.hpp` → `frame/parallel.hpp` ("M11 measured") | 기준값 2¹⁷ | §11.2 |
| `io/csv.cpp` 2, `tests/test_threads.py` 1 ("read_csv in chunks (M11)") | 파일을 청크로 | §10.6 (테스트는 §13.1) |
| `io/csv.hpp` 머리말 ("M11 parallel chunks") | 이력 | 지운다 (§10.6) |
| `ops/radix.hpp` 1 ("In parallel (M11)") | 자리마다 청크별 히스토그램 | §9.5 |
| `ops/radix.hpp` 1 ("groups (M11)") | (청크 × 키) 카운터를 피하는 두 단계 | §9.6 |
| `ops/hash.hpp` 4 (파티션, 해시를 한 번, 흩뿌리기, 파티션의 `reserve`) | 해시로 나눈 factorize | §8.7 (`reserve`는 §8.5도) |
| `ops/elementwise.cpp` 1 ("Lambdas, not function pointers") | 태스크 안의 간접 호출 | §7.6 |
| `ops/pairwise.hpp` 1 ("The pairwise tree in parallel") | 잎 2¹⁴, 같은 트리로 합치기 | §7.4 |
| `ops/hash.cpp` 2 (행마다 묻던 조회, 앨리어싱) | `find_rows` | §8.8 (앨리어싱은 §7.6) |
| `tests/test_threads.py` 첫 줄 | | 지운다 (규칙은 §11.5) |

## 4. 코드에만 있는 표시

| 코드의 표시 | 뜻 | 새 절 | 인용 수 |
|---|---|---|---|
| "user decision 2026-11-11" (`io/csv.cpp`, `ops/radix.hpp`, `ops/hash.hpp` 2) | 결과가 나누는 방식과 무관하면 나누는 방식이 스레드 수를 따른다 | §11.5 (각각 §10.6, §9.6, §8.7) | 4 |
| `experiments/m12_small/radix.cpp` (`ops/radix.hpp`) | 한 번 읽기 히스토그램이 1천~1만 키에서 25–29% 빠르다 | §9.5 | 1 |
| `experiments/m12_small/reserve2.cpp`, `reserve_str.cpp` (`ops/hash.hpp`) | 미리 잡기의 int, string 측정 | §8.5 | 2 |
| "copied pool", "copied from stardust-numpy" (`parallel/pool.hpp`, `bind/module.cpp`) | 풀을 복사해 왔다 | 대응 없음: 풀은 하나 (§11.1). 지운다 | (5절, 3절 M0에서 셈) |

## 5. stardust-numpy 참조 → array.md
`docs/design/array-refmap.md`를 따른다.

| 옛 참조 (인용한 곳) | 내용 | 새 자리 | 인용 수 |
|---|---|---|---|
| "stardust-numpy (its M8)" (`parallel/pool.hpp` 첫 줄) | 풀을 복사해 왔다 | 대응 없음: 풀은 하나 (array.md §10, §11.1). 지운다 | 1 |
| "stardust-numpy used 1 << 15" (`parallel/pool.hpp` → `frame/parallel.hpp`) | 배열 영역의 기준값 | array.md §10.4 (frame의 측정은 §11.2) | 1 |
| stardust-numpy M8 (지운 `parallel/pool.cpp`) | spin을 시계로, 깨우기 약 25µs, mutex 경합 약 18µs, 새 워커의 세대 번호 | array.md §10.1 | 4 (모두 지운 파일) |
| stardust-numpy ARCHITECTURE.md §5 (지운 `parallel/pool.cpp`) | E코어를 기다리지 않게 P코어 수, 바쁜 풀이면 직렬 | array.md §10.3, §10.1 | 2 (모두 지운 파일) |
| stardust-numpy M7.1 (`CMakeLists.txt`) | `NOMINSIZE`: `-Os`에서는 벡터화가 안 된다 | array.md §1.3 | 1 |
| "as in stardust-numpy", "stardust-numpy's pairwise_parallel" (`ops/pairwise.hpp`) | numpy의 pairwise 합과 그 병렬화 | array.md §6.2, `base/pairwise.hpp` (frame 쪽은 §7.4) | 2 |

## 6. 대응이 없는 옛 내용, 코드와 어긋난 옛 문서

| 옛 항목 | 이유 |
|---|---|
| DESIGN §1.1–§1.3의 이름(`stardust-pandas`, `sdpd`), 독립 프로젝트, 부품 복사 | 폐기되었다. 루트 DESIGN §1.1, §2가 대신한다 |
| DESIGN §1.2, ARCHITECTURE §6.2의 "스레드풀은 stardust-numpy의 것을 복사해 온다" | 풀은 하나다 (루트 ARCHITECTURE §4.1). frame은 `parallel/pool.hpp`를 쓰고 정책만 `frame/parallel.hpp`에 둔다 (§11.1) |
| DESIGN §2.5의 "groupby: 코드 기준 scatter 집계, 스레드별 부분 집계를 고정된 순서로 합친다" | 지금 코드와 다르다. 행을 그룹별로 모으고(counting sort) 그룹 묶음을 태스크로 나눈다. 스레드별 부분 집계는 없다 (§9.1) |
| DESIGN §2.5의 "숫자 파싱은 `from_chars`로 시작, Eisel-Lemire는 측정 후에 묻는다" | 결정되었다. Eisel-Lemire를 직접 구현했고 `from_chars`는 마지막 대체다 (§10.4) |
| DESIGN §2.6의 "1스레드에서도 같은 구조로 계산한다"(예외 없이 읽히는 문장) | 예외가 있다 (§11.5) |
| ARCHITECTURE §4의 "M2: Index는 int64 Column만", "M4: 엔진은 `KeyTable`(linear probing)" | 이력이다. 지금 Index는 int64 또는 string이고(§6.1), 엔진은 Swiss table 위에 있다(§6.2, §8.3) |
| ARCHITECTURE §6.1.6의 "왼쪽 키와 오른쪽 키를 이어 붙여 함께 factorize" | 지금 코드와 다르다. 오른쪽만 build, 왼쪽은 probe다 (§9.7). 옛 §6.1.8에는 맞게 적혀 있다 |
| ARCHITECTURE §6.1.7의 "컬럼마다 두 번 훑는다: 추론, 만들기" | 지금 코드와 다르다. 추론하면서 만든다 (§10.3) |
| ARCHITECTURE §6.1.8과 M10 해설의 "빈 칸 = 0x80, 찬 칸 = 태그 0x00..0x7F, 그룹 번호 = 해시의 위쪽 비트" | 지금 코드와 다르다. 빈 칸은 0(calloc한 메모리가 빈 테이블), 찬 칸은 `0x80 \| 태그`, 그룹은 `(h >> 7) & mask`다. 위쪽 비트는 파티션이 쓴다 (§8.3, §8.4) |
| M10 해설의 "`kExpand`: 비트 → 0xFF 또는 0x00" | 지금 코드와 다르다. 0 또는 1 바이트다. 마스크가 필요한 곳은 `0 - b`로 만든다 (§4.4, §4.5) |
| M12 보고서의 "정렬과 CSV는 원소당 비용(cost)을 곱해서 임계값을 넘으므로 10만 행에서도 병렬로 돈다" | CSV는 기준값을 쓰지 않는다. 풀을 쓸 수 있으면 512KB 바이트 청크로 나눈다 (§10.6). 정렬은 흩뿌리기에 2를 곱한다 (§9.5) |
| `io/csv.hpp` 머리말의 "Single-threaded; M10 adds …, M11 …" | 낡은 주석이다. 지금은 청크 병렬이다 (§10.6) |
| ARCHITECTURE §2.5의 `datetime64`, `categorical`(Tier 2)과 DESIGN §2.4의 Tier 2 목록 | 코드가 없다. §1.1의 "없는 것" |
| ARCHITECTURE 머리말의 "A1–A7 확정", `✱` 표시, 각 절의 라운드 번호(Q1–Q4, R1–R7), "사용자 결정 2026-11-1x" | 확정 절차의 이력이다 (루트 DESIGN §5.4) |
| 마일스톤 해설의 처음 측정과 고친 단계(예: M9의 소문자 복사본 109 → 74ms, M2의 "이번에 바로 고친 것") | 이력이다. 지금 구조의 근거가 되는 전후 수치만 해당 절에 남겼다 |
| 옛 `bench/results/*`, `experiments/*`(코드가 인용하는 `experiments/m12_small/*` 포함) | 옛 저장소에 남는다. 수치와 근거만 옮겼다 (4절의 표) |
| M12 보고서의 단계별 비교(M9 → M10 → M11의 기하평균)와 "기록만 한다"는 회귀 표 | 이력이다. 지금 위치는 §13.3, 남은 것은 §12.3 |
