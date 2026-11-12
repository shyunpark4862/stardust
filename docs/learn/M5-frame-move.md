# M5. frame 옮기기

> 옛 stardust-pandas를 `sd.frame`으로 옮겼다. C++은 `csrc/frame/`(이름공간 `sd::frame`)로, Python은 `src/stardust/frame/`으로 갔고, Buffer, DType, 스레드풀, pairwise 합은 공용 부품을 쓴다. **frame의 결과는 비트까지 같다**: 골든 2,834개 중 다른 결과 0개. 바뀐 것은 스레드 수 함수가 `sd`로 옮겨 간 것 하나다.

## 1. 결과
| 항목 | 결과 |
|---|---|
| 테스트 | 2,813개 통과 (M4의 2,430개 + frame 377개 + 풀 테스트 6개) |
| 골든 출력 | frame 2,834개: 다른 결과 0, 메시지만 다른 것 6 (§6). 배열 7,746개, linalg 1,844개 모두 같음 |
| 성능 | 옛 패키지와 같은 자리 비교: frame 기본 스레드 1.014, 1스레드 0.981 (기하평균). 표에 걸린 줄 하나는 회귀가 아니었다 (§8) |
| 컴파일 경고 | 없음 (`-Wall -Wextra -Wpedantic`) |

## 2. 무엇이 어디로 갔나
| 옛 stardust-pandas | 통합판 |
|---|---|
| `csrc/core/{column,bitmap,convert}` | `csrc/frame/` |
| `csrc/ops/*`, `csrc/io/csv.*` | `csrc/frame/ops/*`, `csrc/frame/io/csv.*` |
| `csrc/bind/module.cpp` | `csrc/bind/frame_bind.cpp` (`_core.frame` 하위 모듈) |
| `csrc/core/buffer.*` | 지움: `base/buffer.*`를 쓴다 |
| `csrc/parallel/pool.cpp` | 지움: `parallel/pool.cpp`를 쓴다 |
| `csrc/parallel/pool.hpp`의 frame 정책 | `csrc/frame/parallel.hpp` |
| `src/stardust_pandas/` | `src/stardust/frame/` |
| `tests/test_*.py` | `tests/frame/` (`test_build.py`의 풀 테스트는 `tests/test_build.py`로) |
| `tools/gen_pow5_table.py` | `tools/gen_pow5_table.py` (출력 경로와 이름공간만 바꿈, 다시 만든 표가 같음) |

- 모두 `git mv`로 옮겼다. `git log --follow`로 옛 이력이 이어진다.
  - 예외: 내용이 절반 넘게 바뀐 넷은 git이 새 파일로 본다: `frame/parallel.hpp`(옛 `pool.hpp`에서 frame 정책만 남김), `frame/ops/pairwise.hpp`(합을 공용 정의로), `bind/frame_bind.cpp`(옛 `module.cpp`), `frame/__init__.py`. 옛 이력은 옛 경로(`git log -- legacy/pandas/csrc/parallel/pool.hpp` 등)로 본다.
- 이름공간: `sd` → `sd::frame`, `sd::bitmap` → `sd::frame::bitmap`, `sd::text` → `sd::frame::text`.
- 옛 API.md, README, CMakeLists, pyproject는 지웠다(이력에 남는다). `legacy/pandas/bench/`만 남는다. 같은 자리 비교가 쓰고, M7에서 지운다.

## 3. 공용 부품으로 바꾼 것
### 3.1 Buffer
- frame의 Buffer는 배열의 것과 정렬(128B), `allocate`, `size()`가 같았다. 통합판의 Buffer는 여기에 두 가지가 더 있다: 256B 이하의 데이터를 Buffer 객체 안에 두는 것, 외부 메모리를 `adopt`하는 것.
- 그래서 frame의 작은 Column(원소 32개 이하의 int64/float64, 2,048행 이하의 비트맵)도 할당 한 번으로 만들어진다. 결과는 같고(골든), 작은 연산의 속도는 §8의 표에서 같거나 조금 빠르다.
- Copy-on-Write는 `shared_ptr`의 참조 카운트를 본다. 데이터가 Buffer 안에 있든 밖에 있든 같다.

### 3.2 DType
- frame의 `enum DType { Bool, Int64, Float64, String }`, `dtype_name`, `item_size`를 지우고 `base/dtype.hpp`의 것(`itemsize`)을 쓴다.
- enum 값의 숫자가 바뀌었다(`String`이 3에서 4로). frame 코드에 숫자를 쓰는 곳(표의 첨자, 직렬화)이 없음을 확인했다.
- `Complex128`이 생기자 컴파일러가 `switch` 12곳을 짚었다(`-Wswitch`). 모두 `case DType::Complex128: break;  // not a frame dtype`을 더했다. 그 뒤의 코드는 원래 "도달하지 않는 dtype"의 처리(에러나 기본값)라서 동작은 같다.
- `parse_dtype`은 frame에 남았다. frame은 아직 dtype을 문자열로 받는다. `"complex128"`은 전처럼 "unknown dtype"이다. 표기 통일은 M6이다.
- **계획과 다르게 한 것**: ROADMAP은 Column용 방문자를 M5에 더하기로 했다. 더하지 않았다. frame 커널은 dtype마다 `switch`로 고르고 있고, 방문자로 바꾸는 것은 옮기기가 아니라 고치기다. M6에서 Column ↔ 배열을 이을 때 다시 본다.
- 주의: 이름 찾기. `sd::frame` 안의 함수가 `dtype_name(dt)`를 부르면, `DType`이 `sd`에 있으므로 인자 의존 찾기(ADL)가 `sd::dtype_name`을 찾는다. frame에 같은 시그니처의 `dtype_name`을 남겼다면 모호해졌을 것이다. 그래서 중복 정의는 남기지 않고 지웠다.

### 3.3 스레드풀과 frame의 정책
- 풀 자체는 M4에서 이미 pandas판의 의미가 되었다. 옛 pandas 헤더에서 frame의 정책만 떼어 `frame/parallel.hpp`에 두었다.
  - `kParallelThreshold` = 2¹⁷ (옛 `parallel::kThreshold`), `kChunkRows` = 2¹⁴, `num_chunks`, `for_chunks`, `Unlocked`
  - 배열의 `sd::kParallelThreshold`(2¹⁵)와 이름이 같고 값이 다르다. 영역이 각자 잰 값이다 (M4 해설 §3).
- frame 코드의 `parallel::for_chunks`, `parallel::kChunkRows`, `parallel::Unlocked`는 앞의 `parallel::`만 지웠다(같은 이름공간이 되었다). `parallel::parallel_for`, `parallel::available`, `parallel::FirstRow`는 공용 풀의 것이라 그대로다.
- 옛 pandas의 `kTaskWork`는 frame 코드가 쓰지 않아서 버렸다.

### 3.4 pairwise 합
- frame의 `pairwise_sum`(세 번째 사본)을 `base/pairwise.hpp`의 `tree`와 `leaf`로, 병렬 계획을 `pairwise::plan`/`pairwise::combine`으로 바꿨다. frame이 정하는 것은 잎 태스크의 크기(`kPairwiseLeaf` = 2¹⁴)와 언제 나누는가(`kParallelThreshold`)뿐이다.
- 트리 모양, 잎의 합, 나누는 자리가 셋 다 같은 정의라서 비트가 같다(골든). M3처럼 느려지지 않았다: `float sum · 1000`은 1스레드에서 321ns → 289ns다 (§8).
- `ieee_minmax`는 frame에 남았다(min/max의 NaN과 −0.0 규칙).

## 4. Python 쪽
- `sd.frame`의 `__init__`이 맨 앞에서 `_core = stardust._core.frame`을 둔다. 하위 모듈의 `from . import _core`는 이 속성을 찾는다.
- `from ._core import Column` 같은 줄은 그대로 둘 수 없었다. nanobind의 하위 모듈은 `sys.modules`에 없어서 `stardust.frame._core`라는 모듈을 import할 수 없다. 그래서 `Column = _core.Column`으로 바꿨다(7개 파일). `sys.modules`에 등록하는 방법도 있지만, 속성 하나로 충분하고 linalg(`_core = _sd_core.linalg`)와 같은 방식이다.
- `sd.frame.__all__`에서 `set_num_threads`, `get_num_threads`가 빠졌다. 라이브러리 공통의 `sd.set_num_threads`, `sd.get_num_threads`를 쓴다 (API.md §3).
- `import stardust`가 `sd.frame`도 불러온다(`sd.linalg`와 같다).
- 공개되어 버린 하위 모듈(`sd.frame.frame`, `sd.frame.index`, `sd.frame.io`, `sd.frame.series`)은 그대로다. 숨기는 것은 공개 API가 바뀌는 일이라 M6에서 한다 (DESIGN §8.4).

## 5. 테스트
- `import stardust_pandas as sdpd` → `import stardust as sd`, `sdpd.X` → `sd.frame.X`, 스레드 수 호출은 `sd.set_num_threads`로 바꿨다. 이 밖의 테스트 내용은 바꾸지 않았다.
- `tests/frame/`이 패키지가 되면서 `from test_csv import …`가 깨졌다. `from .test_csv import …`로 바꿨다(배열, linalg 테스트와 같은 방식).
- `test_build.py`의 풀 테스트(기본 10스레드, 태스크마다 한 번씩, 범위 밖 스레드 수)는 풀이 공용이 되었으므로 `tests/test_build.py`로 옮겼다. 그 테스트가 부르는 `_pool_smoke_test`도 `_core`로 옮겼다. frame 쪽의 `test_import`, `_build_info`는 루트의 것과 같아서 지웠다.
- `test_api_surface.py`는 옛 API.md의 §21을 잘라 읽었다. 이제 `docs/api/frame.md` 전체의 시그니처 블록을 공용 파서(`tests/_api_doc.py`)로 읽는다. 블록의 형식은 모듈 이름은 접두어 없이(`merge(...)`, `NA`), 멤버는 `Owner.name`이다.
- 테스트 수: 옛 pandas 388개 = 표면 3 + 빌드 8 + 나머지 377. 통합판은 나머지 377 + 표면 3, 빌드 테스트는 루트로 합쳐졌다.

## 6. 골든 출력
- 처음 비교에서 18개가 달랐다. 모두 스레드 함수를 `sd.frame`에서 찾는 케이스였다(`AttributeError`). 케이스가 스레드 함수를 라이브러리의 스레드 모듈(`L._threads`: 옛 쪽은 `stardust_pandas`, 새 쪽은 `sd`)에서 찾게 고쳤다.
- 남은 6개는 **메시지만 다르다**. 예외 종류는 같다.

  | 케이스 | 옛 pandas | 통합판 |
  |---|---|---|
  | `set_num_threads(2.0)` | TypeError: nanobind의 "incompatible function arguments … `arg: int`" | TypeError: "n must be an int" |
  | `set_num_threads(n=2)` | TypeError: 같은 메시지, `arg: int` | TypeError: 같은 메시지, `arg: object` |

- 통합판의 메시지는 배열 영역의 것이다(배열 골든과 같다). 함수가 하나가 되었으니 메시지도 하나다.
- 골든에 없는 차이 하나: 옛 pandas는 `set_num_threads(True)`를 1로 받았다. 통합판은 `TypeError`다 (API.md §3, "`bool` 포함"). 배열 영역은 원래 그랬다.

## 7. 문서
- 새로 썼다: `docs/api/frame.md`(공개 API와 동작 규칙, §1–§13), `docs/design/frame.md`(내부 구조와 근거, 1–13절). 코드 주석의 옛 참조는 임시 대응표(`docs/api/frame-refmap.md`, `docs/design/frame-refmap.md`)로 바꿨다.
- 루트 문서
  - API.md: 색인에 frame 문서, §1에 "frame은 아직 문자열 dtype"이라는 예외, §2에 `KeyError`, §3에 "1이면 풀을 쓰지 않는다", "frame은 큰 작업이면 1스레드여도 GIL을 놓는다"
  - DESIGN §8.4: API 문서를 쓰다 찾은 배열 ↔ frame의 차이 일곱 가지를 더했다(§4.4 표에 없던 것). 범위 밖 Python int, numpy에서 만들 때 복사, `==`, int 비트 연산, `copy` 인자, dtype 인자, `KeyError`. M6 전에 정리할 목록이다.
  - ARCHITECTURE: frame의 자리, Buffer와 풀의 M5 상태, 컴파일 플래그 확인
- **옛 문서와 코드가 어긋났던 곳**(설계 문서를 쓰다 찾음, 모두 코드를 따랐다)
  - Swiss table: 옛 문서는 빈 칸이 0x80이라 했지만, 코드는 빈 칸이 0이다(calloc한 메모리가 그대로 빈 테이블). 찬 칸은 `0x80 | 태그`다.
  - merge: 옛 문서는 "양쪽 키를 이어 붙여 함께 factorize"라 했지만, 코드는 오른쪽을 만들고 왼쪽으로 찾는다.
  - read_csv: "컬럼마다 두 번 훑는다"가 아니라 추론하면서 만든다. CSV는 기준값을 쓰지 않고, 풀을 쓸 수 있으면 512KB 바이트 청크로 나눈다. `csv.hpp` 머리말의 "Single-threaded"는 낡은 주석이었다(고쳤다).
  - groupby의 병렬화: "스레드별 부분 집계"가 아니라 행을 그룹별로 모아 그룹 묶음을 태스크로 나눈다.
  - Index: "int64만, linear probing"이 아니라 int64 또는 string, Swiss table이다.

## 8. 성능
옛 패키지와 통합판을 같은 자리에서 번갈아 2라운드 (`bench/merge/sameseat.sh`, `bench/merge/m5/final/table.md`):

| 스레드 | 줄 | 통합판/옛 패키지 기하평균 | 10% 넘게 느려짐 | 10% 넘게 빨라짐 |
|---|---:|---:|---:|---:|
| 10 | 49 | 1.014 | 0 | 0 |
| 1 | 49 | 0.981 | 1 | 4 |

pandas 대비(10스레드, 49줄): 통합판이 빠름 44, 비슷 4, 느림 1. 기하평균 4.6배.

- 빨라진 넷은 작은 호출이다: `s.sum()` 10행 190 → 163ns, `df[df["a"] > 0]` 10행 1.8 → 1.6µs, `float sum` 1,000행 321 → 289ns, `sort int64` 1,000행 15.8 → 12.9µs. 작은 Column이 Buffer 안에 놓여 할당이 준 것과 맞는다. 원인을 하나씩 확인하지는 않았다.
- **걸린 한 줄은 회귀가 아니었다.** `float + float` 100만 행 1스레드가 표에서 158 → 199µs였다. 프로세스를 네 번씩 새로 띄워 번갈아 쟀다 (`bench/merge/m5/procs.py`, 단위 µs, 15번 중 최솟값/중앙값):

  | 옛 패키지 | 통합판 |
  |---|---|
  | 99/105, 100/108, 136/136, 133/134 | 99/103, 101/109, 100/104, 127/135 |

  둘 다 프로세스마다 약 100µs 아니면 약 135µs로 갈린다. M4의 배열 `f + g`와 같은 현상이다.
- 같은 자리 비교 스크립트를 이번에 `bench/merge/sameseat.sh`로 저장소에 넣었다(세 영역, 옛 패키지는 `.venv-legacy`, 통합판은 `.venv` + `bench/merge/shim`). frame용 shim(`shim/stardust_pandas.py`)은 `sd.frame`과 스레드 함수를 내보낸다.
- M5는 배열과 linalg 코드를 바꾸지 않아서(모듈 정의에 한 줄) 두 영역은 다시 재지 않았다. 골든은 다시 돌렸다.

## 9. 리뷰 포인트
1. **결과는 같고, 바뀐 공개 API는 스레드 함수 하나다.** `sd.frame.set_num_threads`가 없어지고 `sd.set_num_threads`를 쓴다. 메시지 두 개와 `True`의 처리가 배열 쪽으로 맞춰졌다.
2. **공용 부품 넷**(Buffer, DType, 풀, pairwise 합)을 frame이 쓰게 되었다. 세 벌이던 pairwise 합이 이제 정말 한 벌이다.
3. **frame의 정책은 `frame/parallel.hpp`에 있다.** 기준값의 이름은 영역마다 같고 값은 다르다.
4. **계획과 다르게 한 것**: Column용 방문자를 더하지 않았다(§3.2). `_pool_smoke_test`와 풀 테스트를 루트로 옮겼다(§5).
5. **새로 찾은 불일치 일곱 가지**를 DESIGN §8.4에 더했다. M6 전에 함수 단위 목록을 정할 때 함께 본다.
6. 다음은 M6(frame과 배열 잇기, 일관성 맞추기)이다. 시작 전에 DESIGN §8.3의 남은 R3 항목(1, 4, 5, 8, 10)과 R4의 함수별 이름·시그니처 목록을 확정해야 한다.
