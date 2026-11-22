# 테스트 실행 안내

모든 명령은 저장소 루트에서 `.venv`의 Python으로 돌린다. 이 문서는 테스트(§1), 골든(§2), 커버리지(§3)를 다룬다. 속도와 정확도 측정은 [bench/README.md](../bench/README.md)에 있다.

## 도구 공통 관례

`tools/test.py`, `tools/golden.py`, `tools/coverage.py`, `python -m bench.speed`, `python -m bench.accuracy`는 명령행을 `bench/cli.py`로 만든다. 같은 이름의 옵션은 어느 도구에서나 같은 뜻, 같은 기본값이다.

| 관례 | 뜻 |
|---|---|
| `AREA …` | 위치 인자. 영역 이름 `array linalg frame special stats regression random optimize htest sparse integrate differentiate spatial`(`bench/areas.py`, README.md의 표 순서) 가운데 그 도구가 다루는 것. 주지 않으면 모두다. 결과와 표도 이 순서다. |
| `--match PATTERN` | id가 맞는 것만 남긴다. `*`, `?`가 있으면 id 전체와 맞춰 보는 glob이고, 없으면 부분 문자열이다. `[`, `]`는 보통 글자라 표에서 복사한 id(`erf(x) [10^6]`)가 그대로 맞는다. 여러 번 주면 하나만 맞아도 된다. |
| `--jobs N` | 작업 프로세스 수. 기본 8, 최대 10. |
| `--out FILE` | 결과를 쓸 파일(폴더가 아니다). |
| 종료 코드 | 0 모두 됐다. 1 실패, 차이, 계약을 넘은 값, 기준선보다 나빠진 측정(`WORSE`·`slower`), 또는 견줄 것이 없다. 2 명령행 오류나 도구 자신의 오류. |
| 로그 | `build/logs/<도구>/<시각>.log`, 도구마다 최근 20개를 둔다. 하위 도구(pytest, cmake, llvm-cov)의 출력과 traceback은 터미널이 아니라 여기에 남는다. |

- 터미널 출력은 모두 `bench/console.py`의 같은 틀이다: 머리 블록, 단계마다 진행 막대와 결과 한 줄, 끝에 요약 블록·표·실패 목록·번호 붙은 Notes.
- 옛 꼴의 옵션(`--area`, `--case`, `-n`, `--group`, 정규식 고르기)은 없다.
- 참값 도구 `tools/<영역>_ref.py`는 `--jobs`만 같이 쓴다. 위치 인자는 영역이 아니라 만들 함수·분포·묶음의 이름이다(bench/README.md의 정확도 절).

## 1. 테스트

```bash
.venv/bin/python tools/test.py                            # quick, 병렬 (워커 8개, 그 뒤 serial 테스트 따로)
.venv/bin/python tools/test.py frame --match merge        # tests/frame에서 id에 merge가 든 테스트
.venv/bin/python tools/test.py --tier deep                # deep (SD_TIER=deep과 같다)
.venv/bin/python tools/test.py -- tests/bench -x          # -- 뒤는 pytest의 인자 그대로 (-k, -m, -n은 받지 않는다)
.venv/bin/python -m pytest -q -p no:cacheprovider         # quick, 한 프로세스
```

- **등급**은 `--tier`나 환경 변수 `SD_TIER` 하나로 고른다. 정의는 `tests/conftest.py`와 `tests/_tier.py`에 있다.
  - `quick`(기본): `deep`·`external` 마커가 없는 모든 테스트. hypothesis는 `derandomize=True`라 매번 같은 예제다(병렬로 돌려도 결정적이다).
  - `deep`: quick과 `@pytest.mark.deep` 테스트(Arb 참값을 그 자리에서 계산하는 것, 넓은 격자, 골든 비교 `tests/golden/test_golden.py`, 정확도 비교 `tests/bench/test_accuracy_baseline.py`). hypothesis는 무작위이고 예제 수가 10배다.
  - `all`: deep과 `@pytest.mark.external`(R이나 Fortran 컴파일러를 실제로 부르는 테스트와, 저장소 밖에서 받은 파일을 읽는 테스트: `tests/sparse/test_suitesparse.py`는 `tools/sparse_suitesparse.py`로 받은 행렬이 없으면 건너뛴다; 희소의 R 비교(RSpectra·irlba)는 기록 `tests/sparse/data/r_eigen.json`을 읽고 기록이 없으면 건너뛴다).
  - 고르지 않은 등급의 테스트는 deselect된다(요약에 `N deselected`).
- **예제 수**: hypothesis 테스트는 대부분 프로필 기본값(quick 100, deep 1000)을 쓴다. 수를 따로 정하는 테스트는 `examples(n)`(`tests/_tier.py`)으로 써서 quick에서 n, deep에서 10n이다.
- **고르기**: `AREA`는 `tests/<영역>`이고, 주지 않으면 영역 밖(`tests/interop`, `tests/bench`, `tests/test_*.py`)까지 모두다. `--match`는 테스트 id(`tests/<영역>/test_<주제>.py::test_<이름>[…]`)에 맞춘다. `tests/conftest.py`가 적용하므로 `pytest`에 바로 줘도 된다.
- **병렬**: `pytest`만 치면 한 프로세스다. `tools/test.py`는 `serial`이 아닌 테스트를 pytest-xdist 워커 `--jobs`개로 돌리고, 그 뒤에 `serial` 테스트를 혼자 돌린다. `serial`은 다른 테스트 프로세스와 기계를 나누면 안 되는 테스트다(GIL 해제를 벽시계로 재는 것, 시간 제한이 있는 하위 프로세스, 물리 메모리 근처의 할당, 스스로 작업 프로세스를 띄우는 것). 모듈 전체는 `tests/conftest.py`의 `SERIAL`에, 테스트 하나는 `@pytest.mark.serial`로 적는다. 모두 보려면 `.venv/bin/python -m pytest -m serial --co -q`.
  - 워커 수 기본 8은 이 Mac에서 quick 전체를 잰 값으로 정했다(2026-11-20, 다른 실행이 없을 때, serial 부분 44초 포함): 워커 6개 76.6초·워커 합 RSS 7.6GB, 8개 69.6초·9.7GB, 10개 75.7초·12.9GB. 워커마다 sd 스레드 수는 기본값 그대로가 가장 빨랐다(2로 줄이면 느렸다).
- **종료 코드**: 0 고른 테스트가 모두 통과, 1 실패가 있거나 두 단계 모두 고른 테스트가 없음, 2 명령행 오류나 pytest 자신의 오류(중단, 내부 오류). 두 단계 중 나쁜 쪽이 정한다. pytest의 출력과 실패한 테스트의 traceback은 `build/logs/test/<시각>.log`에 있다.
- **테스트 수**(2026-11-22에 센 값): quick 16,977개(그 가운데 serial 29개), deep 17,700개(serial 31개, deep 마커 723개). 수는 커밋마다 바뀌므로 모으기만 해서 센다: `.venv/bin/python -m pytest --collect-only -q -p no:cacheprovider | tail -1`(deep은 `--tier deep`, serial은 `-m serial`을 더한다).
- `--strict-markers`라서 등록하지 않은 마커는 에러다. 마커는 `deep`, `external`, `serial`이다.
- **파일 이름과 도우미**
  - 테스트 파일은 주제로 이름 짓는다(`test_<주제>.py`). 리뷰 번호나 날짜는 파일 이름에 쓰지 않고 docstring이나 주석에만 적는다.
  - 값이 같은지는 `tests/_bits.py`로만 본다: `bits_equal`(바이트까지, NaN의 payload와 0의 부호 포함), `bits_equal_any_nan`(NaN끼리는 어떤 NaN이든 같고 ±0은 다르다), `value_equal`(원소마다 `==`, NaN끼리 같다). 원소별 최대 상대오차는 `max_rel_error`다.
  - 영역 안에서 함께 쓰는 도우미는 `tests/<영역>/conftest.py`(array, linalg, frame, optimize)와 `tests/regression/smalldata.py`에 있다. `sd.set_num_threads`를 부르는 테스트는 `tests/conftest.py`의 `restore_threads` 픽스처로 스레드 수를 되돌린다.
  - 참값 파일을 읽고 계약 대비 오차를 내는 함수는 영역마다 `tests/<영역>/refdata.py` 하나이고, 테스트와 정확도 측정(`bench/accuracy`)이 같이 쓴다.
- **optimize의 scipy 비교**: scipy를 함께 돌려 단계마다 견주는 테스트는 scipy 1.18.1(`tests/optimize/conftest.py`의 `SCIPY_VERSION`, `tests/optimize/data/scipy_*.json`을 기록한 판)이 아니면 건너뛰지 않고 실패한다. 문제 모음을 도는 테스트는 quick에서 문제 목록의 여덟 개에 하나와 MGH 모음 밖의 문제만, scipy 스윕은 스윕의 첫 문제만 돌고, 나머지는 deep이다.
- **numpy·scipy와 견주는 테스트**: 반올림이 아닌 동작(dtype, 정수 연산, 정렬 순서, NaN·±0, 상태 값, 기본값, 멈춤 규칙)은 정확히 견준다. 반올림 크기의 값은 정확도 계약으로 견주고, 비트까지 같아야 하는 기준은 sd 자신의 골든이다(§2; API.md §4). 지금 numpy·scipy와 비트까지 견주는 테스트는 통과하는 동안 그대로 둔다. 의도한 고침이나 최적화 때문에만 깨지면 허용오차와 반복 수(최적화)의 비교로 바꾼다.

## 2. 골든 (비트 동일성)

공개 함수의 결과를 정해진 입력에서 비트까지 저장해 두고, 결과가 바뀌면 안 되는 변경(옮기기, 리팩터링, 포매터)의 전후를 비교한다.

```bash
.venv/bin/python tools/golden.py check                    # 모든 영역을 돌려 저장된 결과와 비교
.venv/bin/python tools/golden.py check special            # 영역만
.venv/bin/python tools/golden.py check stats --match Normal
.venv/bin/python tools/golden.py record                   # 다시 기록 (결과를 일부러 바꾼 뒤)
.venv/bin/python tools/golden.py list                     # 영역별 케이스 수
```

- 케이스는 `tests/golden/cases_<영역>.py`, 기록은 `tests/golden/data/<영역>.json`이다. 영역은 `bench/areas.py`의 열셋이다(spatial은 공개 함수가 생기는 SPT1부터 케이스가 있다).
- 영역마다 스레드 수 1, 3, 10에서 따로 프로세스를 띄워 돌리고(동시에 `--jobs`개), 세 결과가 같아야 한다. 스레드 수에 따라 결과가 다른 케이스는 저장하지 않는다: `record`가 그 케이스 이름을 적고 종료 코드 1로 끝난다. 예외는 케이스 파일의 `THREAD_DEPENDENT = {케이스 이름: 이유}`에 적은 케이스뿐이고, 그 결과는 스레드 수마다 저장한다.
- 전체 비교(2026-11-22, 34,391케이스)는 영역마다 `--jobs 2`로 차례로 돌린 합이 약 2분 15초다. 작업 프로세스 하나의 최대 RSS는 약 1.8GB(array)라서 `--jobs` 기본 8은 메모리가 넉넉할 때만 쓴다.
- **저장 형식**(형식 2): 결과를 정확한 JSON으로 바꿔 정규화한 글자열끼리 비교한다. 그래서 형도 결과의 일부다(`True`는 `1`이 아니고, numpy 스칼라는 Python 스칼라가 아니다).
  - bool, int, str, None은 그대로, float는 `float.hex`(기본 NaN이 아닌 NaN은 비트까지), complex는 그 둘, numpy 스칼라는 dtype과 값, 내장 형의 하위 클래스는 클래스와 값이다.
  - 에러도 결과다. 에러와 케이스가 낸 경고는 클래스(`module.qualname`)와 메시지로 저장한다.
  - 원소가 1024개 이하인 배열은 원소마다 저장한다. 그보다 큰 배열은 dtype·shape·strides, 바이트의 SHA-256, 원소 조각마다의 해시(조각은 256원소 이상, 배열 하나에 256조각 이하), 정해진 표본 원소다.
  - stardust 클래스의 객체는 공개 속성 전부와 인자 없이 부를 수 있는 메서드 전부를 저장한다.
- **`check`의 출력**: 바뀐 케이스마다 어느 스레드 수에서 어디가 바뀌었는지 적는다. 1024원소 이하의 배열은 바뀐 원소 수, 첫 위치, 옛 값과 새 값의 hex·10진·ULP 거리(두 값 사이에 있는 double의 수, `bench/accuracy/common.py`의 `ulp_distance`)이고, 큰 배열은 해시가 다른 조각의 수와 첫 조각의 원소 범위다. 새 케이스는 `NEW`, 없어진 케이스는 `GONE`이다.
- **종료 코드**: 0 다른 것이 없다. 1 차이, 골든 파일이 없거나 옛 형식, `record`의 스레드 수에 따른 결과. 2 명령행 오류나 워커의 실패(케이스 이름과 오류 끝부분을 적는다; 시간 초과 포함). 로그는 `build/logs/golden/<시각>.log`다.
- **옵션**: `--match PATTERN`(케이스 이름; `check`는 맞는 케이스만 견주고, `record`는 맞는 케이스만 고쳐 쓰고 더는 없는 맞는 이름을 지운다), `--jobs N`, `--out FILE`(영역 하나의 골든 파일을 `tests/golden/data/<영역>.json` 대신 FILE에), `--threads 1,3,10`, `-v`(케이스의 차이를 모두 적는다), `--slow N`(가장 느린 케이스 N개), `--any-nan`(`check`에서 부호나 payload만 다른 NaN을 같다고 본다).
- 멈춘 케이스: 케이스 하나가 `GOLDEN_TIMEOUT`초(기본 120)를 넘기면 그 작업 프로세스를 끝내고 케이스 이름을 알린다. `GOLDEN_TRACE=1`이면 작업 프로세스가 케이스 이름을 시작할 때마다 찍는다.
- 결과를 일부러 바꾼 변경이면 `check`로 바뀐 케이스가 의도한 것뿐인지 본 뒤 `record`로 다시 기록하고 함께 커밋한다.
- 새 공개 함수를 더하면 그 영역의 `cases_<영역>.py`에 케이스를 더한다. 케이스는 `(이름, 인자 없는 함수)`를 내는 생성기이고, 입력은 고정 리터럴과 `L.rng(seed)`로만 만든다. 영역의 공개 이름은 모두 케이스에 나오거나 `EXCLUDED = {이름: 이유}`에 있어야 한다(quick 테스트 `tests/golden/test_golden.py`가 본다).

## 3. 커버리지

C++ 코어는 clang의 소스 기반 커버리지(`-fprofile-instr-generate -fcoverage-mapping`), Python은 pytest-cov(coverage.py)로 잰다.

```bash
.venv/bin/python tools/coverage.py all                    # 계측 빌드, 테스트, 보고서
.venv/bin/python tools/coverage.py all --golden           # 골든 비교도 계측 코어로 돌린다
.venv/bin/python tools/coverage.py run special --match bessel   # 일부 테스트만 (pytest 인자는 -- 뒤에)
.venv/bin/python tools/coverage.py report                 # 보고서만 다시
```

- 고르기는 `tools/test.py`와 같다(`AREA`, `--match`, `--` 뒤의 pytest 인자; `-k`, `-m`, `-n`은 받지 않는다). `--jobs`는 계측 빌드의 컴파일 작업 수다.
- 계측 빌드는 `build/coverage/cmake`에 따로 만든다. 평소 빌드(`build/<휠 태그>`)와 `CMakeLists.txt`는 건드리지 않는다. 같은 CMake 프로젝트에 계측 플래그만 더한 Release(-O3) 빌드이고, 처음에는 몇 분, 그 뒤로는 바뀐 파일만 다시 컴파일한다.
- `--golden`은 고른 영역의 `tools/golden.py check --any-nan`을 계측 코어로 돌린다. 계측한 코드는 기계어가 달라서 몇 케이스에서 NaN 결과의 부호가 다르다(2026-11-20에 10케이스). 그 밖의 비트는 모두 같아야 한다.
- 계측 코어에서만 죽는 테스트는 `tools/coverage.py`의 `CRASHES`에 그 버그를 적어 빼고 돌린다. 버그가 고쳐지면 목록에서 지운다.
- 테스트는 `src/`의 stardust에 계측한 `_core`를 끼워 돌린다(`build/coverage/site/sitecustomize.py`가 `SD_COVERAGE_CORE`가 있는 모든 프로세스에서 한다. 하위 프로세스도 센다).
- 종료 코드는 0 모든 단계 통과, 1 테스트나 골든 비교의 실패, 2 명령행 오류나 단계의 실패다. 하위 도구의 출력은 `build/logs/coverage/<시각>.log`에 있다.
- 보고서는 모두 `build/coverage/`에 생기고 커밋하지 않는다.
  - `summary.md`: 영역별 C++ 줄·함수·분기, Python 줄·분기의 비율
  - `<영역>.md`: 파일별 비율과 실행되지 않은 분기의 목록(`파일:줄:열`, 안 간 쪽, 그 줄의 소스)
  - `html/cpp/`, `html/py/`: 줄마다 실행 수와 분기 수를 단 소스
- C++ 분기는 소스 위치마다 한 번 센다(템플릿 인스턴스의 수를 합친다). `true never`·`false never`는 한 번도 가지 않은 쪽, `never reached`는 조건을 한 번도 평가하지 않은 분기다.
