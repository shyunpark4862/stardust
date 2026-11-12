# stardust 내부 구조

> 상태: **확정** (2026-11-12). 통합이 끝났을 때의 목표 구조다. 결정 근거는 [DESIGN.md](DESIGN.md)에 있다.
>
> 지금의 세 저장소(`stardust-numpy`, `stardust-linalg`, `stardust-pandas`)를 하나로 합친 뒤의 모습을 적는다. 각 영역 안의 세부 구조(이터레이터, 배치 드라이버, Copy-on-Write 등)는 영역 문서 `docs/design/<영역>.md`에 새로 쓴다 (§6).

## 1. 층 구조

```
             Python: stardust (sd)   sd.linalg   sd.frame   (이후 영역 …)
                         │              │           │
bind/  ──────────────────┴──────────────┴───────────┴──────────  _core 확장 모듈 하나
                         │              │           │
                      array/  ◀──── linalg/       frame/
                         │                          │
                         ├──────────┬───────────────┤
                    accelerate/  parallel/        base/
```

- **아래로만 의존한다.** 위 층은 아래 층을 쓰고, 아래 층은 위 층을 모른다.
  - `array/`는 `linalg/`와 `frame/`을 include하지 않는다.
  - `linalg/`는 `array/`의 `NDArray`를 직접 받는다. buffer protocol을 거치지 않는다 (DESIGN §3.1).
  - `frame/`은 `array/`를 모른다. 둘을 잇는 코드는 `frame/interop.{hpp,cpp}` 한 곳에만 둔다 (DESIGN §4.3, M6). `array/` 헤더를 include하는 frame 파일은 `interop.cpp` 하나다. 다른 frame 파일은 `interop.hpp`(Column과 nanobind만 쓰는 선언)를 통해 부른다 (docs/design/frame.md §5.4).
- **새 영역은 디렉터리 하나와 Python 하위 모듈 하나로 붙는다.** 아래 층만 쓰고, 기존 영역을 고치지 않는다. 예: 나중의 sklearn류 `ml/`은 `linalg/`와 `array/` 위에 올라간다.
  - 새 영역은 `docs/design/<영역>.md`, `docs/api/<영역>.md`, `tests/<영역>/`, `bench/<영역>/`을 함께 갖는다.

## 2. 디렉터리 구조

```
stardust/                         (저장소: ~/Desktop/stardust/stardust)
  pyproject.toml  CMakeLists.txt
  README.md  API.md               (API.md는 영역별 API 문서의 색인과 공통 규칙)
  DESIGN.md  ARCHITECTURE.md  ROADMAP.md
  legacy/                         M0에서 이력과 함께 들여온 세 저장소. 마일스톤마다 git mv로 꺼내 가고, M7에서 빈다

  src/stardust/
    __init__.py                   배열 API + 스레드 수 + dtype 객체 다섯 (sd.string 포함)
    _format.py                    배열 repr
    _results.py                   세 영역의 결과 namedtuple (<함수>Result 하나의 모듈, DESIGN §9.7)
    linalg/
      __init__.py                 _core.linalg 재노출, LinAlgError
      _errors.py                  LinAlgError(ValueError)
    frame/
      __init__.py                 DataFrame, Series, Index, RangeIndex, NA, concat, merge, read_csv
      _series.py  _frame.py  _index.py  _groupby.py  _merge.py  _io.py  _*.py   (모든 모듈이 밑줄 이름: 공개 이름은 __init__.py만)
      _column.py                  데이터 → Column, 명시적/암묵적/정확한 값 변환, to_numpy/to_array
      _errors.py                  에러 메시지의 함수 이름 (API.md §2.1)

  csrc/
    base/                         모든 영역이 쓰는 바닥
      buffer.hpp/.cpp             참조 카운트 메모리 블록 (numpy판 + pandas판 통합, §4.2)
      dtype.hpp                   DType enum과 이름, 저장 타입 (§4.3)
      pairwise.hpp                numpy식 pairwise 합의 정의 (트리, 잎, 병렬 계획; 세 벌 → 한 벌)
      saturate.hpp                float64 → int64 명시적 변환 (포화, NaN → 0; 배열, linalg, frame이 함께 쓴다, M6)
    parallel/
      pool.hpp/.cpp               스레드풀 하나 (pandas판의 의미, §4.1), FirstRow
    accelerate/
      blas.hpp                    BLAS 래퍼 (dot, gemv, gemm; 배열의 matmul이 쓴다)
      lapack.hpp                  LAPACK 래퍼 (linalg가 쓴다)
      vforce.hpp                  vForce 래퍼 (연산과 무관한 부분)
    array/                        (stardust-numpy csrc/의 core, kernels, ops)
      ndarray  iter  convert  parallel_limits
      kernels/  elementwise buffered cast scalar_ops reduce sort simd_sort neon vforce
      ops/      asarray copy creation indexing matmul reduction shape sort ufunc lanes
    linalg/                       (지금 stardust-linalg csrc/)
      args  batch  pack  cost  errors  common
      ops/      lu basic norms vnorm factor qr spectral svals nonsym special expm
    frame/                        (M5, 이름공간 sd::frame)
      column  bitmap  convert  parallel (for_chunks, Unlocked, kParallelThreshold)
      interop.hpp/.cpp             Column ↔ NDArray, 배열의 float64 합 (DESIGN §4.3, M6). frame에서 array/를 아는 유일한 파일
      ops/      cast elementwise groupby hash join reduce selection sort radix parse pow5_table
      io/       csv
    bind/
      module.cpp                  _core와 하위 모듈 _core.linalg, _core.frame 정의
      array_bind.cpp  pyutil.cpp  linalg_bind.cpp  frame_bind.cpp
      errors.hpp/.cpp              공개 함수의 에러 메시지에 이름 붙이기 (named, call_named; API.md §2.1, M6)

  tests/
    array/  linalg/  frame/  interop/       interop: frame ↔ 배열의 공유, 복사, 읽기 전용 뷰, Copy-on-Write (M6)
    test_build.py  test_error_messages.py   빌드와 공용 풀, 세 영역의 에러 메시지 형식 (API.md §2.1)
    _api_doc.py                   API 문서의 시그니처 블록 파서 (영역마다의 test_api_surface가 쓴다). conftest.py는 array/, linalg/에 따로 있다
  bench/      array/  linalg/  frame/  merge/      스크립트는 옮기고, 결과는 새로 잰다
  experiments/                                     새 실험만
  docs/                                            모두 새로 쓴다 (DESIGN §5.4)
    design/   array.md  linalg.md  frame.md        영역의 확정 사항과 내부 구조
    api/      array.md  linalg.md  frame.md        영역별 API 문서 (공개)
    guide/                                         안내서 한 권
    learn/                                         통합 마일스톤 해설
  tools/
    gen_pow5_table.py             frame/ops/pow5_table.hpp를 만든다
```

### 2.1 지금 파일이 가는 곳

| 지금 | 통합 후 |
|---|---|
| numpy `csrc/core/{ndarray,iter,convert}` | `csrc/array/` |
| numpy `csrc/core/{buffer,dtype}` | `csrc/base/` (pandas판과 합침) |
| numpy `csrc/kernels/neon.hpp` | `csrc/array/kernels/neon.hpp` (M1에서 `base/`에 두었다가 M4에서 옮김: 배열의 연산 정의에 묶여 있다) |
| numpy `csrc/kernels/vforce.hpp` | 둘로 나눔 (M4): 연산과 무관한 래퍼는 `csrc/accelerate/vforce.hpp`, 연산 → vForce 함수 표는 `csrc/array/kernels/vforce.hpp` |
| numpy `csrc/kernels/reduce.hpp`의 pairwise 부분 | `csrc/base/pairwise.hpp` |
| numpy `csrc/parallel/pool.*` | 버림 (pandas판으로 대체) |
| linalg `csrc/core/{input,outblock,pyrefs}` | `csrc/linalg/`에 남되 경계 코드는 없어짐 (M3, §3): `Input`은 `NDArray`를 빌리고, `outblock`은 `NDArray`를 만들고, `pyrefs`는 결과 타입과 `LinAlgError`만 든다 |
| linalg `csrc/core/{types,pairwise}.hpp` | M3: `DType`은 `base/dtype.hpp`, pairwise의 정의는 `base/pairwise.hpp`로 (`linalg/pairwise.hpp`에는 잎을 채우는 방식만 남음), float → int64는 배열의 `saturate_to_int64`. M6에서 그 함수가 `base/saturate.hpp`로 내려가 frame도 쓴다 (M4에서는 내릴 필요가 없었다, ROADMAP M4) |
| linalg `csrc/core/{args,batch,pack,cost,errors}` | `csrc/linalg/` |
| linalg `csrc/lapack/lapack.hpp` | `csrc/accelerate/lapack.hpp` |
| linalg `csrc/parallel/pool.*` | 버림 |
| pandas `csrc/core/buffer.*` | `csrc/base/buffer.*` (numpy판과 합침) |
| pandas `csrc/core/{column,bitmap,convert}` | `csrc/frame/` |
| pandas `csrc/ops/pairwise.hpp` | `csrc/frame/ops/pairwise.hpp`가 `csrc/base/pairwise.hpp`의 정의를 부른다 (min/max는 그대로) |
| pandas `csrc/parallel/pool.*` | `csrc/parallel/pool.*` (통합판의 기준). `for_chunks`, `Unlocked`, `kChunkRows`, 기준값은 `csrc/frame/parallel.hpp` |

## 3. Python과 C++의 경계

- **"얇은 Python, 두꺼운 C++"** 원칙은 세 프로젝트가 같았고 그대로 둔다.
  - 예외도 그대로다: `Series`, `DataFrame`은 Python 클래스다 (pandas DESIGN §1.1).
- **확장 모듈은 `_core` 하나다** (DESIGN §3.2).
  - `nb::module_::def_submodule`로 `_core.linalg`, `_core.frame`을 만든다.
  - nanobind 타입 테이블이 하나라서, C++ 어디서든 `is_ndarray`/`as_ndarray`(`bind/pyutil.hpp`)로 배열을 직접 받는다.
- **linalg의 경계 코드를 없앴다** (M3).

  | 별도 패키지 시절 (linalg) | 지금 |
  |---|---|
  | `Input`: `PyObject_TypeCheck` + `PyObject_GetBuffer` + format 문자열 해석 | `const NDArray&` 인자 하나 |
  | `alloc_matrices` → 블록 객체 → `sdnp.asarray` 호출 | linalg가 정한 strides로 `new_array_strided`가 `NDArray`를 직접 만든다 |
  | `pyrefs`: `sdnp.ndarray`, `asarray`, `get_num_threads`, dtype 객체 캐시 | 결과 namedtuple과 `LinAlgError`만 남는다 |
  | 호출마다 `sdnp.get_num_threads()` Python 호출 | `parallel::num_threads()` |

  - 예상은 결과 하나에 약 200ns(asarray 113ns, 블록 할당·해제 82ns, 나머지 12ns)였고, 잰 값은 결과 배열 하나에 약 120ns다. `NDArray`와 Python 객체를 만드는 약 75ns는 남는다 (docs/design/linalg.md §3.6).
- `array/`에는 F-order 전용이 아니라 **strides를 받는 생성 함수**를 더했다: `new_array_strided(dtype, shape, strides, nbytes)`. 레이아웃(행렬마다 열 우선, 배치 축은 바깥에 C-order, 길이 0인 축도 그대로 곱함)은 linalg의 정책이라 linalg에 두고, 메모리와 플래그만 배열이 맡는다.

## 4. 공통 부품

### 4.1 스레드풀
- 하나만 둔다. **pandas판이 기준이다.** numpy판의 상위 집합이기 때문이다.
  - 핵심(상주 워커, spin 후 대기, atomic 카운터, 호출 스레드 참여, 중첩 금지)은 세 벌이 같다.
  - pandas판에만 있는 것: `busy()`, `available()`의 "풀이 바쁘면 직렬 경로", 항상 GIL 해제, `for_chunks`, `FirstRow`, `Unlocked`.
  - M4: 앞의 셋과 `FirstRow`(영역과 무관)는 공용 풀에 들어갔다. `for_chunks`, `Unlocked`, `kChunkRows`는 frame의 기준값(2¹⁷)과 비트맵에 맞춘 청크 크기에 묶여 있어서 M5에서 `frame/parallel.hpp`로 갔다.
  - linalg판에만 있는 것: `available_here()`. M2에서 배열의 풀에 더했다가, M3에서 linalg가 `parallel::available()`만 쓰게 되어 다시 뺐다.
- **병렬화 기준값은 영역마다 따로 둔다.** 측정해서 정한 값이 다르기 때문이다.
  - array: `1 << 15` (numpy M8), frame: `1 << 17` (pandas M11), linalg: 비용 모델 (`linalg/cost.hpp`, linalg M9)
  - 풀 헤더의 `kThreshold`는 없앴다(M4). 영역마다 자기 이름공간에 같은 이름의 상수를 둔다: 배열은 `sd::kParallelThreshold`, `sd::kParallelTask`(`array/parallel_limits.hpp`), frame은 `sd::frame::kParallelThreshold`(`frame/parallel.hpp`).
- 스레드 수 API는 `sd.set_num_threads`, `sd.get_num_threads` 하나다. `sd.frame`에는 두지 않는다.

### 4.2 Buffer
- numpy판과 pandas판은 이름(`sd::Buffer`)과 정렬(128B)이 같고 내용이 조금 다르다. 한 모듈에 그대로 넣으면 ODR 위반이다.
- 통합판 = numpy판 + pandas판의 `size()`. M5부터 frame도 이것을 쓴다 (256B 이하의 Column 버퍼도 Buffer 안에 놓인다).
  - numpy판: 작은 배열은 Buffer 안에 데이터를 둔다(256B, numpy M7.7), 외부 메모리를 `adopt`한다.
  - pandas판: `size()`를 갖는다. Copy-on-Write가 `shared_ptr`의 참조 카운트를 본다 (pandas ARCHITECTURE §3.3).
- M6부터 배열 뷰와 Column이 같은 Buffer를 가리킬 수 있다 (`Series.to_array`). Copy-on-Write 규칙은 그대로다: 읽기 전용 배열 뷰가 `shared_ptr<Buffer>`를 하나 쥐고 있으면 `use_count()`가 늘어 있으므로 frame 쪽 쓰기가 먼저 복사한다 (DESIGN §4.3, docs/design/frame.md §5.4).

### 4.3 dtype
- C++ enum은 `base/dtype.hpp`에 하나 둔다 (DESIGN §4.2의 추천안).
  - `Bool, Int64, Float64, Complex128, String` (M4). `Datetime64`는 frame이 그것을 구현할 때 더한다(지금은 어느 영역에도 코드가 없다).
  - 컨테이너마다 받는 부분집합을 명시한다: 배열은 앞의 넷(`kNumArrayDTypes`, `is_array_dtype`), Column은 `Complex128`을 뺀 넷.
- `visit_dtype`은 배열 dtype만 받는다. 모든 커널이 모든 dtype을 인스턴스화하지 않게 하기 위해서다.
  - Column용 방문자는 더하지 않았다(M5). frame 코드는 dtype마다 `switch`로 고르고, 그 `switch`들이 `Complex128`을 "frame dtype이 아님"으로 따로 적는다. M6에서 Column ↔ 배열을 이을 때도 더하지 않았다. `interop.cpp`는 고정폭 dtype의 바이트를 복사할 뿐이라 `itemsize`로 충분했다.

### 4.4 Accelerate 래퍼
- BLAS(numpy `matmul`), LAPACK(linalg), vForce(numpy 초월함수)를 `accelerate/` 한곳에 둔다.
- 컴파일 정의 `ACCELERATE_NEW_LAPACK`, `ACCELERATE_LAPACK_ILP64`는 모듈 전체에 건다 (지금 numpy와 linalg가 같다).

## 5. 빌드
- `nanobind_add_module(_core NB_STATIC NOMINSIZE ...)` 하나.
- 컴파일 플래그는 numpy판으로 통일한다: `-mcpu=native -ffp-contract=off -fno-builtin-sin -fno-builtin-cos -Wall -Wextra -Wpedantic`
  - linalg와 pandas에는 `-fno-builtin-sin/-cos`가 없었다. 넣어도 결과가 같다: linalg는 M2, frame은 M5의 골든 출력으로 확인했다.
- editable 설치는 바뀐 파일만 다시 컴파일한다. 전체 빌드는 세 프로젝트를 합친 만큼 길어진다.

## 6. 영역별 세부 구조
영역 안의 구조는 그 영역을 옮기는 마일스톤에서 `docs/design/<영역>.md`에 새로 쓴다. 옛 문서에서 지금도 맞는 내용만 가져오고, 이력은 쓰지 않는다.

| 영역 문서 | 다룰 것 | 옛 출처 (참고만) | 쓰는 때 |
|---|---|---|---|
| `array.md` | NDArray와 뷰, 이터레이터, 캐스팅, 인덱싱, ufunc, 리덕션, 정렬, matmul, 병렬화 기준 | stardust-numpy DESIGN·ARCHITECTURE | M1 |
| `linalg.md` | 공통 규칙(실패, 유한성 검사, 결과 레이아웃), 패킹, 배치 드라이버, 비용 모델, 함수별 메모 | stardust-linalg DESIGN·ARCHITECTURE (경계와 "두 개의 풀"은 이 문서 §3, §4.1이 대신함) | M2에서 쓰고 M3에서 경계 부분을 고침 |
| `frame.md` | Column, 비트맵, NA 규칙, Copy-on-Write, Index와 정렬(alignment), 커널 체계, CSV | stardust-pandas DESIGN·ARCHITECTURE | M5에서 쓰고 M6에서 규칙 변경을 반영 |
