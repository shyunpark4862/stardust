# stardust 내부 구조

라이브러리의 층, 디렉터리, 공통 부품, 그리고 측정·시험 체계의 자리를 적는다. 결정의 근거는 [DESIGN.md](DESIGN.md)에, 영역 안의 구조(이터레이터, 배치 드라이버, Copy-on-Write, 분포 엔진 등)는 영역 문서 `docs/design/<영역>.md`에 있다.

## 1. 층 구조

영역은 여덟이다: array, linalg, frame, special, stats, regression, random, optimize(`bench/areas.py`의 순서). Python 하위 모듈 하나와 `csrc/`의 디렉터리 하나가 영역 하나다.

```
 Python:  sd   sd.linalg   sd.frame   sd.special   sd.stats   sd.regression   sd.random   sd.optimize
            │       │          │           │            │            │               │            │
 bind/ ─────┴───────┴──────────┴───────────┴────────────┴────────────┴───────────────┴────────────┴──  _core 확장 모듈 하나
                                                        │
                     regression/ ──▶ linalg/, special/, random/
                     optimize/   ──▶ linalg/, differentiate/(차분의 간격), elementwise/(원소별 반복 고리)
                     elementwise/ ──▶ array/, parallel/  (differentiate/의 steps는 표준 라이브러리만 쓴다)
                     stats/      ──▶ special/, random/
                     special/, linalg/, random/, frame/(interop만) ──▶ array/
                                                        │
                                  array/ ──▶ base/, parallel/, accelerate/
```

- **아래로만 의존한다.** 위 층은 아래 층을 쓰고, 아래 층은 위 층을 모른다.
  - `array/`는 다른 영역을 include하지 않는다. 예외는 의존이 없는 `special/sincos.hpp`(같은 x의 sin과 cos를 한 번에, docs/design/array.md §1.3) 하나다.
  - `linalg/`는 `array/`의 `NDArray`를 직접 받는다. buffer protocol을 거치지 않는다 (DESIGN §3.1). LAPACK 래퍼(`linalg/lapack.hpp`, `linalg/ops/common.hpp`)는 regression과 optimize도 쓴다.
  - `frame/`은 `array/`를 모른다. 둘을 잇는 코드는 `frame/interop.{hpp,cpp}` 한 곳에만 둔다 (DESIGN §4.3). `array/` 헤더를 include하는 frame 파일은 `interop.cpp` 하나다. 다른 frame 파일은 `interop.hpp`(Column과 nanobind만 쓰는 선언)를 통해 부른다 (docs/design/frame.md §5.4).
  - `stats/`는 special의 커널(불완전 감마·베타, 오차함수 등)과 random의 생성기를 쓴다. `regression/`은 linalg의 LAPACK 래퍼, special의 double-double·오차함수·감마·로지스틱 함수, random의 생성기를 쓴다. Python 쪽에서 `regression.nls`는 optimize의 최소제곱(`csrc/optimize/lsq.cpp`)을 쓴다.
  - Python 경계의 공용 도우미 `bind/pyutil.hpp`(배열 받기와 만들기)와 `bind/errors.hpp`(에러 메시지에 공개 함수 이름 붙이기)는 모든 영역의 `ops/`가 쓴다. 영역마다의 바인딩 파일(`bind/<영역>_bind.cpp`)은 맨 위 층이다.
- **새 영역은 디렉터리 하나와 Python 하위 모듈 하나로 붙는다.** 아래 층만 쓰고, 기존 영역을 고치지 않는다.
  - 새 영역은 `docs/design/<영역>.md`, `docs/api/<영역>.md`, `tests/<영역>/`, 골든 케이스 `tests/golden/cases_<영역>.py`, 속도 사례 `bench/speed/cases/<영역>.py`를 함께 갖고, `bench/areas.py`의 목록에 들어간다. 참값이 있으면 `tests/<영역>/ref/`, `tests/<영역>/refdata.py`, `tools/<영역>_ref.py`와 정확도 스윕을 더한다.

## 2. 디렉터리 구조

```
stardust/
  pyproject.toml  CMakeLists.txt
  README.md  API.md  DESIGN.md  ARCHITECTURE.md  ROADMAP.md

  src/stardust/
    __init__.py                 배열 API, 스레드 수, dtype 객체 다섯 (sd.string 포함)
    _format.py                  배열 repr
    _results.py                 모든 영역의 결과 namedtuple (<함수>Result, DESIGN §9.7)
    _calls.py                   호출 꼴 에러(인자 수, 모르는 키워드)를 라이브러리의 꼴로 (API.md §2.1)
    linalg/                     __init__.py(_core.linalg 재노출), _errors.py(LinAlgError)
    frame/                      __init__.py와 밑줄 이름의 모듈들(_series, _frame, _index, _column, _groupby, _merge, _io, …)
    special/  random/           __init__.py: _core의 하위 모듈을 공개 이름으로
    stats/                      _distribution(공통 기반), _continuous, _discrete, _multivariate, _matrix, _descriptive
    regression/                 _linear, _glm, _penalized, _robust, _quantile, _nls
    optimize/                   _scalar, _elementwise, _minimize, _lsq, _linear, _root, _diff, _result, _common

  csrc/
    base/         buffer(참조 카운트 메모리 블록), dtype(DType enum), pairwise(pairwise 합의 정의),
                  saturate(float64 → int64 명시적 변환), int_sum(int64의 정확한 합)
    parallel/     pool(스레드풀 하나, GIL 해제, FirstRow, Unlocked)
    accelerate/   blas(dot, gemv, gemm), vforce(연산과 무관한 vForce 래퍼)
    array/        ndarray  iter  convert  parallel_limits
                  kernels/  elementwise buffered cast scalar_ops reduce sort simd_sort neon vforce
                  ops/      asarray copy creation indexing manipulation shape matmul reduction
                            covariance quantile histogram searching sort ufunc lanes pieces dlpack dtype_info
    linalg/       args batch pack cost errors input outblock pyrefs types pairwise lapack
                  ops/      lu basic norms vnorm factor qr spectral svals nonsym special expm common
    frame/        column bitmap convert parallel interop
                  ops/      cast elementwise groupby hash join reduce selection sort radix parse pairwise pow5_table
                  io/       csv
    special/      common dd cf halley prefix reduce sincos gamma_core cplx boost cost cost_table
                  ops/      logistic erf gamma loggamma zeta incgamma incbeta distr noncentral expint ellip
                            bessel bessel01 airy hyper lambertw wofz testing
                  kernels/  simd chunked
                  coeffs/   생성한 계수 표 (tools/gen_*.py)
    stats/        engine chunk chunk_dd numeric discrete dd_linalg mvcdf sample sample_dist cost_table
                  dist/     분포 계열마다 한 파일 (normal, gamma, beta, t_f, noncentral, discrete_basic, …)
                  coeffs/   생성한 계수 표
    regression/   linear lsq slopes glm penalized robust quantile dot2
    random/       philox random
    optimize/     scalar batch lsq minpack minimize objective wolfe hessian nelder_mead bfgs lbfgsb
                  newton_cg trust root hybrd linear
    differentiate/ steps(고정 간격 차분의 간격: optimize가 씀)
    elementwise/  loop(원소별 반복 고리: optimize의 배치 넷, 미분·적분의 원소별 무리; docs/design/integrate.md §6.2)
    bind/         module.cpp(_core와 하위 모듈), <영역>_bind.cpp, stats_{desc,matrix,mv,mvcdf,sample}.cpp,
                  nary_bind.hpp(원소별 함수의 공통 바인딩), pyutil, errors

  third_party/boost_math/       special이 쓰는 Boost.Math 헤더 (README.md에 커밋)

  tests/
    conftest.py  _tier.py       등급(quick, deep, all), 마커, hypothesis 프로필, --match, restore_threads
    _bits.py                    값이 같은지 보는 도우미 (bits_equal, bits_equal_any_nan, value_equal, max_rel_error)
    reffile.py                  npz 참값 파일의 한 형식
    _api_doc.py                 API 문서의 시그니처 블록 파서 (영역마다의 test_api_surface가 쓴다)
    test_*.py                   영역을 넘는 것: 빌드와 공용 풀, 에러 메시지 형식, GIL 해제, 범위 밖 입력, 도구의 명령행
    <영역>/                     영역의 테스트(test_<주제>.py), conftest.py, refdata.py, ref/(참값), data/(바깥 입력, 다른 라이브러리의 기록)
    interop/                    frame ↔ 배열의 공유, 복사, 읽기 전용 뷰, Copy-on-Write
    golden/                     골든 케이스 cases_<영역>.py, 기록 data/<영역>.json, test_golden.py
    bench/                      측정 체계 자신의 테스트, 정확도 기준선 비교(test_accuracy_baseline.py)

  bench/
    areas.py  cli.py            영역 목록 하나, 모든 측정·시험 도구의 명령행 관례
    console.py                  콘솔 출력의 공통 틀
    speed/                      속도 측정 (cases/<영역>.py, results/<영역>.json 기준선)
    accuracy/                   정확도 측정 (results/<영역>.json 기준선)
    <영역>/                     기준선에서 생성한 표: speed.md, accuracy.md

  tools/
    test.py  golden.py  coverage.py    테스트, 골든, 커버리지 (tests/README.md)
    <영역>_ref.py                      Arb 참값 생성기 (special, stats, array, linalg, regression, optimize)
    arbref.py  regref.py  mvref.py  mvcdfref.py   생성기가 함께 쓰는 Arb 도우미와 식
    gen_*.py                           C++ 계수 표(csrc/*/coeffs, frame의 pow5_table)와 다른 라이브러리의 기록(R, scipy, Fortran)
    check.py  format.py  check_refs.py  procrun.py  vendor_boost_math.py

  docs/
    design/<영역>.md            영역의 결정과 내부 구조, 측정 근거
    api/<영역>.md               영역별 공개 API (test_api_surface가 읽는다)
```

## 3. Python과 C++의 경계

- **"얇은 Python, 두꺼운 C++"**: 계산은 C++에 있고, Python은 인자 정리, 결과 객체, 메시지를 맡는다.
  - 예외: `Series`, `DataFrame`, 분포 객체(`sd.stats`), 회귀·최적화의 결과 객체는 Python 클래스다.
- **확장 모듈은 `_core` 하나다** (DESIGN §3.2).
  - `nb::module_::def_submodule`로 영역마다 `_core.linalg`, `_core.frame`, `_core.special`, `_core.stats`, `_core.regression`, `_core.random`, `_core.optimize`를 만든다(`bind/module.cpp`가 차례로 부른다).
  - nanobind 타입 테이블이 하나라서, C++ 어디서든 `is_ndarray`/`as_ndarray`(`bind/pyutil.hpp`)로 배열을 직접 받는다.
- **linalg는 배열을 C++에서 직접 주고받는다.**
  - 입력은 `const NDArray&`이고, 결과는 linalg가 정한 strides로 `new_array_strided(dtype, shape, strides, nbytes)`가 `NDArray`를 직접 만든다. 레이아웃(행렬마다 열 우선, 배치 축은 바깥에 C-order, 길이 0인 축도 그대로 곱함)은 linalg의 정책이라 linalg에 두고, 메모리와 플래그만 배열이 맡는다.
  - `pyrefs`에는 결과 namedtuple과 `LinAlgError`만 있다. 스레드 수는 `parallel::num_threads()`로 읽는다.
  - 결과 배열 하나를 만드는 비용은 약 120ns이고, 그 가운데 `NDArray`와 Python 객체를 만드는 약 75ns는 피할 수 없다 (docs/design/linalg.md §3.6).

## 4. 공통 부품

### 4.1 스레드풀
- 하나만 둔다(`parallel/pool.hpp`): 상주 워커, spin 후 대기, atomic 카운터, 호출 스레드 참여, 중첩 금지. 풀이 다른 Python 스레드의 일로 바쁘면 직렬 경로로 간다(`available()`). 병렬로 도는 동안 GIL을 놓는다. `FirstRow`(병렬 검사에서 실패한 가장 앞의 위치: 에러가 스레드 수와 상관없이 같은 위치를 말한다)와 `Unlocked`(풀을 쓰지 않는 긴 직렬 구간에서 GIL 놓기)가 함께 있다.
- **병렬화 기준값은 영역마다 따로 둔다.** 측정해서 정한 값이 다르기 때문이다. 이름은 같고 이름공간이 다르다.
  - array: `sd::kParallelThreshold` = 2¹⁵, `sd::kParallelTask`(`array/parallel_limits.hpp`, docs/design/array.md §10.4)
  - frame: `sd::frame::kParallelThreshold` = 2¹⁷, `for_chunks`, `Unlocked`(`frame/parallel.hpp`)
  - linalg: 비용 모델(`linalg/cost.hpp`)
  - special·stats: 원소마다 잰 비용표(`csrc/special/cost_table.hpp`, `csrc/stats/cost_table.hpp`; bench/README.md의 C++ 비용표)
- 스레드 수 API는 `sd.set_num_threads`, `sd.get_num_threads` 하나다. 하위 모듈에는 두지 않는다.

### 4.2 Buffer
- `base/buffer.hpp`의 `sd::Buffer` 하나를 배열과 Column이 함께 쓴다. 정렬은 128B이고, 256B 이하의 데이터는 Buffer 객체 안에 둔다. 외부 메모리를 `adopt`하고, `size()`로 바이트 수를 준다.
- 배열 뷰와 Column이 같은 Buffer를 가리킬 수 있다(`Series.to_array`). Copy-on-Write 규칙은 그대로다: 읽기 전용 배열 뷰가 `shared_ptr<Buffer>`를 하나 쥐고 있으면 `use_count()`가 늘어 있으므로 frame 쪽 쓰기가 먼저 복사한다 (DESIGN §4.3, docs/design/frame.md §5.4).

### 4.3 dtype
- C++ enum은 `base/dtype.hpp`에 하나 둔다: `Bool, Int64, Float64, Complex128, String`.
  - 컨테이너마다 받는 부분집합을 명시한다: 배열은 앞의 넷(`kNumArrayDTypes`, `is_array_dtype`), Column은 `Complex128`을 뺀 넷.
- `visit_dtype`은 배열 dtype만 받는다. 모든 커널이 모든 dtype을 인스턴스화하지 않게 하기 위해서다. frame 코드는 dtype마다 `switch`로 고르고, 그 `switch`들이 `Complex128`을 "frame dtype이 아님"으로 따로 적는다. `interop.cpp`는 고정폭 dtype의 바이트를 복사할 뿐이라 `itemsize`로 충분하다.

### 4.4 Accelerate 래퍼
- 영역을 넘어 쓰는 BLAS(`accelerate/blas.hpp`: 배열의 `matmul`·공분산)와 vForce(`accelerate/vforce.hpp`: 배열·special·stats의 원소별 초월함수)는 `accelerate/`에 둔다. LAPACK 래퍼는 `linalg/lapack.hpp`에 있다(§1).
- 컴파일 정의 `ACCELERATE_NEW_LAPACK`, `ACCELERATE_LAPACK_ILP64`는 모듈 전체에 건다.

## 5. 빌드
- `nanobind_add_module(_core NB_STATIC NOMINSIZE ...)` 하나.
- 컴파일 플래그는 모듈 전체에 하나다: `-mcpu=native -ffp-contract=off -fno-builtin-sin -fno-builtin-cos -Wall -Wextra -Wpedantic`
  - `-ffp-contract=off`: 컴파일러가 곱셈과 덧셈을 FMA로 스스로 합치지 않는다. FMA는 재서 이득이 확인된 곳에만 명시적으로(`std::fma`, NEON `vfmaq`) 쓴다 (DESIGN.md §1.6).
  - `-fno-builtin-sin`, `-fno-builtin-cos`: 같은 x의 `sin`과 `cos`를 컴파일러가 `sincos` 하나로 바꾸지 않는다. 함께 구하는 곳은 코드가 `special/sincos.hpp`로 명시한다. 같은 이유다.
- editable 설치(`pip install --no-build-isolation -e ".[test]"`)는 `import stardust`에서 바뀐 파일만 다시 컴파일한다. 빌드 폴더는 `build/<휠 태그>`다.

## 6. 측정·시험 체계

네 체계가 서로 다른 질문에 답한다. 모두 `bench/areas.py`의 영역 목록과 `bench/cli.py`의 명령행 관례를 같이 쓴다.

| 체계 | 묻는 것 | 자리 | 문서 |
|---|---|---|---|
| 테스트와 등급 | 동작이 맞는가 (quick, deep, all) | `tests/`, `tools/test.py` | tests/README.md §1 |
| 골든 | 결과가 비트까지 그대로인가 (스레드 수 1, 3, 10) | `tests/golden/`, `tools/golden.py` | tests/README.md §2 |
| 정확도 | Arb 참값에서 얼마나 떨어졌고, 계약 안이며, 기준선보다 나빠지지 않았는가 | `bench/accuracy/`, 참값 `tests/<영역>/ref/`, 생성기 `tools/<영역>_ref.py` | bench/README.md §2 |
| 속도 | 비교 대상보다 얼마나 빠르고, 기준선보다 느려지지 않았는가 | `bench/speed/` | bench/README.md §1 |

- 커버리지(`tools/coverage.py`, tests/README.md §3)는 테스트와 골든을 계측 빌드로 돌린다.
- 정확도와 테스트는 참값 파일을 읽고 계약 대비 오차를 내는 함수(`tests/<영역>/refdata.py`)를 같이 쓴다. 속도와 정확도는 비교 대상 라이브러리, R 세션 규칙, Boost.Math 빌드를 같이 쓴다.
- 기준선(`bench/speed/results/`, `bench/accuracy/results/`)과 골든 기록(`tests/golden/data/`)은 커밋한다. 실행 파일과 로그(`build/`)는 커밋하지 않는다.
