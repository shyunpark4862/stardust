# M1. 배열 코어 옮기기

> stardust-numpy를 `stardust`의 최상위 `sd`로 옮겼다. **동작은 하나도 바꾸지 않았다.** 바뀐 것은 파일의 위치, 패키지 이름, 그리고 새로 쓴 문서다.

## 1. 결과
| 항목 | 결과 |
|---|---|
| 옮긴 파일 | C++ 41개, Python 3개, 테스트 18개 (`git mv` 62건 + 덮어쓴 3건) |
| 테스트 | 2,085개 통과 (옛 stardust-numpy와 같은 수) |
| 골든 출력 | 배열 7,746개 중 다른 것 0, 메시지만 다른 것 0 |
| 이력 | `git log --follow csrc/array/ops/matmul.cpp`가 옛 M6 커밋까지 이어진다 |
| 새 문서 | `docs/api/array.md` (41KB), `docs/design/array.md` (88KB) |
| 주석 참조 | 58개 파일, 약 240곳을 새 문서로 바꿈. 코드 토큰은 하나도 바뀌지 않았음을 확인 |

## 2. 파일이 간 곳
ARCHITECTURE.md §2.1의 표대로 옮겼다.

| 옛 위치 | 새 위치 | 이유 |
|---|---|---|
| `core/buffer`, `core/dtype` | `base/` | frame과 linalg도 쓸 바닥이다 (M4에서 pandas판과 합친다) |
| `kernels/neon.hpp` | `base/neon.hpp` | NEON 보조 함수는 frame도 쓴다 |
| `kernels/vforce.hpp` | `accelerate/vforce.hpp` | BLAS, LAPACK 래퍼와 한곳에 모은다 (M4) |
| `core/{ndarray,iter,convert}`, 나머지 `kernels/`, `ops/` | `array/`, `array/kernels/`, `array/ops/` | 배열 영역 |
| `parallel/pool` | `parallel/` | 그대로. M4에서 pandas판으로 바뀐다 |
| `bind/ndarray_bind` | `bind/array_bind` | 영역 이름에 맞췄다 |

- include 경로 26종을 스크립트로 바꿨다. 바꾼 뒤 모든 include가 실제 파일을 가리키는지 확인했다.
- **아직 남은 역방향 의존**: `base/neon.hpp`와 `accelerate/vforce.hpp`가 `array/kernels/scalar_ops.hpp`를 include한다. 바닥 층이 위 층을 아는 셈이다. 옮기기만 하는 마일스톤이라 그대로 두었고, M4의 바닥 정리에서 고친다 (`docs/design/array.md` §1.5).
- `kernels/reduce.hpp`의 pairwise 합은 아직 `base/pairwise.hpp`로 떼어내지 않았다. linalg와 frame의 것과 합치는 M3, M4에서 한 번에 한다.

## 3. 이름 바꾸기
- 패키지: `stardust_numpy` → `stardust`. C++ 안에서 Python 모듈을 부르는 두 곳(`stardust._results`, `stardust._format`)도 바꿨다.
- **사용자에게 보이는 메시지**: `sdnp.ndarray` → `sd.ndarray`, "stardust-numpy has no 0-d arrays" → "stardust has no 0-d arrays".
  - 동작은 아니지만 사용자에게 보이는 차이라서, 골든 비교가 이 이름 바꾸기를 알게 했다. `golden.py`의 `RENAMES`에 `sdnp.` → `sd.`, `stardust-numpy` → `stardust`를 더하고 세 영역을 다시 기록했다 (옛 패키지는 `.venv-legacy`에 따로 설치되어 있어서 그대로 다시 기록할 수 있다).
- 테스트: `import stardust_numpy as sdnp` → `import stardust as sd`. 지역 변수 `sd`를 쓰던 두 곳(`test_m5_reduction.py`의 `sd, ref = ...`, `test_m2_iter.py`의 `sd()` 도우미)은 먼저 이름을 바꿨다. 안 그러면 모듈 이름을 가린다.

## 4. 테스트를 패키지로
- 옛 테스트는 `from conftest import assert_same`처럼 conftest를 모듈로 import했다. pytest의 기본 import 방식에서는 conftest가 있는 디렉터리가 `sys.path`에 들어가서 이게 동작한다.
- M2에서 `tests/linalg/conftest.py`가 생기면 `conftest`라는 같은 이름의 모듈이 둘이 되어 충돌한다.
- 그래서 `tests/__init__.py`, `tests/array/__init__.py`를 두고 `from .conftest import ...`로 바꿨다. 영역마다 패키지가 따로라 이름이 겹치지 않는다.
- `test_build.py`는 뼈대에서 만든 것과 옛 것이 내용이 같아서, 옛 것을 `tests/test_build.py` 자리로 옮겨 이력을 남겼다.

## 5. 문서를 새로 썼다
DESIGN §5.4대로 옛 문서를 옮기지 않고 새로 썼다. 옛 문서의 결정 이력(라운드, 날짜, 마일스톤별 변화)은 빼고, 지금 유효한 규칙과 구조를 측정 근거와 함께 적었다.

### 5.1 `docs/api/array.md`
- §0 공통 규칙(dtype, 에러 종류, 스레드), §1 배열 공통 규칙(시그니처, 0차원 없음, 승격, 변환, 수치 의미론, 메모리 겹침, `copy`, 메모리 순서), §2–§10 함수 계열별
- 쓰면서 **코드를 기준으로 맞춘 곳**: 옛 API.md가 코드와 달랐던 곳은 새 문서에 코드대로 적었다.
  - `arange(start, stop, step, /)`: 셋 다 위치 전용이다 (옛 문서는 `stop=`, `step=`을 키워드로 쓸 수 있게 적었다).
  - `zeros` 등의 `dtype` 기본값은 `None`(= float64)이다.
  - 옛 문서에 없던 동작: `concat(axis=None)`은 펼친 뒤 잇는다. `real`/`imag`는 Python 스칼라를 거부한다(`conj`는 받는다). 대입의 오른쪽에 리스트, 튜플, numpy 배열을 받는다. `set_num_threads`의 상한은 1024다. `copy`/`pickle`은 TypeError다.
  - 이 중 일관성 문제인 것은 DESIGN §8.4에 옮겨 적었다 (M6에서 정한다).
- **API 표면 테스트를 튼튼하게 고쳤다.** 옛 테스트는 정규식 `` ```\n(.*?)``` ``로 코드 블록을 찾았다. 문서에 ```` ```python ```` 블록이 하나라도 있으면, 그 블록을 닫는 펜스를 다음 블록의 시작으로 읽어서 엉뚱한 본문을 시그니처로 읽는다. 이제 펜스를 한 줄씩 열고 닫으며 언어 표시가 없는 블록만 읽는다 (`signature_blocks`).

### 5.2 `docs/design/array.md`
- §1 기반(하드웨어, 컴파일 옵션과 이유, 넣지 않은 최적화, 층 의존) … §13 테스트와 측정. 13장, 811줄.
- 옛 문서와 코드를 대조하다 **옛 문서가 더 이상 맞지 않던 곳**을 고쳐 적었다.
  - 버퍼 단위 캐스팅은 입력만 한다. 더 넓은 `out=`은 임시 배열에 계산한 뒤 대입한다.
  - 내림차순 정렬은 결과를 뒤집지 않는다고 했지만, NEON 값 정렬 경로는 같은 값끼리 구별이 안 될 때만 `std::reverse`를 쓴다.
  - `stable=False`는 introsort라고 했지만, NEON 정렬과 기수 정렬이 먼저고 introsort는 나머지만 맡는다.
  - "병렬화 기준값은 상수 하나"는 이터레이터에만 맞는다. 마스크, 정렬 조각, int64 matmul, 작은 BLAS 배치는 각자 기준이 있다.
  - buffer protocol로 복사 없이 가져오려면 자연 정렬도 맞아야 하고, big-endian 버퍼는 거부한다.

### 5.3 옛 참조 대응표
- `docs/design/array-refmap.md`, `docs/api/array-refmap.md`: 옛 문서의 절과 규칙 ID(S6, C13, Z1 …)가 새 문서의 어느 절로 갔는지.
- 이번 마일스톤의 주석 교체에 썼고, M2에서 linalg 주석(stardust-numpy 문서를 인용한다)을 고칠 때도 쓴다. M7에서 지운다.
- 정의가 남아 있지 않은 ID 두 개(Q1, Q4)는 코드 문맥으로 추정했다: Q4는 "배열은 스칼라가 아니다", Q1은 "`dtype=`은 명시적 변환".

## 6. 코드 주석의 옛 참조
- C++ 37개, Python 21개 파일에서 약 240곳을 바꿨다.
  - `DESIGN.md §2.4`, `ARCHITECTURE.md §5`, `API.md §9` 같은 옛 절 번호와 S6, C13, Z1 같은 규칙 ID → `docs/design/array.md §N`, `docs/api/array.md §N`
  - 마일스톤 번호("M7.6", "M10+", "user decision 2026-11-10")는 지웠다. 주석에 적힌 이유와 측정 수치는 남기고, 근거가 새 설계 문서에 있으면 그 절을 가리킨다.
  - 예: `// changed (user decision 2026-11-10; DESIGN.md §2.4).` → `// changed (docs/design/array.md §6.5, §6.9).`
- 인용 규칙: C++과 `src/`의 주석은 설계 문서를, 테스트에서 공개 계약을 확인하는 주석은 API 문서를 가리킨다.
- **주석만 바뀌었다는 증명**
  - C++: clang의 원시 토큰(`-Xclang -dump-raw-tokens`)에서 주석과 공백을 뺀 토큰열을 HEAD와 비교했다. 37개 파일 모두 같다.
  - Python: `tokenize`에서 주석과 docstring을 뺀 토큰열을 비교했다. 21개 파일 모두 같다.
  - 다시 빌드한 뒤 테스트 2,085개 통과, 골든 7,746개 모두 같음.
- 남은 대문자+숫자 토큰은 참조가 아니다: dtype 코드(`I64`, `F64`, `C128` …), `C99`, `L1` 캐시, 칩 이름 `M4 Pro`.

## 7. 리뷰 포인트
1. **문서를 읽어 봐 달라.** `docs/api/array.md`와 `docs/design/array.md`는 옛 문서 다섯 개(API, DESIGN, ARCHITECTURE, 해설, 안내서)를 줄여 새로 쓴 것이다. 빠진 결정이나 틀린 설명이 있으면 이후 영역 문서도 같은 방식으로 쓰게 되니 지금 바로잡는 것이 좋다.
2. **공통 규칙의 자리**: dtype, 에러 종류, 스레드 규칙은 지금 `docs/api/array.md §0`에 있다. M2에서 linalg가 들어올 때 루트 `API.md`로 올린다.
3. **역방향 의존 하나** (`base/` → `array/kernels/scalar_ops.hpp`)가 남았다. M4에서 고친다.
4. 다음은 M2(linalg 옮기기)다. 경계 코드(`Input`, `outblock`, `pyrefs`)는 그대로 두고 같은 `_core`에 넣어, 골든 1,844개가 같은지 먼저 본다.
