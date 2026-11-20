# 테스트 실행 안내

모든 명령은 저장소 루트에서 `.venv`의 Python으로 돌린다.

## 1. 테스트

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider        # quick, 한 프로세스
.venv/bin/python tools/test.py                           # quick, 병렬 (워커 6개 + serial 따로)
SD_TIER=deep .venv/bin/python tools/test.py              # deep
.venv/bin/python tools/test.py tests/frame -k merge      # pytest 인자를 그대로 넘긴다 (-m은 빼고)
```

- **등급**은 환경 변수 `SD_TIER` 하나(또는 `--tier`)로 고른다. 정의는 `tests/conftest.py`에 있다.
  - `quick`(기본): `deep`·`external` 마커가 없는 모든 테스트. hypothesis는 `derandomize=True`라 매번 같은 예제다(병렬로 돌려도 결정적이다).
  - `deep`: quick과 `@pytest.mark.deep` 테스트(Arb 참값을 그 자리에서 계산하는 것, 넓은 격자), hypothesis는 무작위이고 예제 수가 10배(프로필 기본 1000)다. 골든 비교와 정확도 비교(`tests/golden/test_golden.py`)도 여기 든다.
  - `all`: deep과 `@pytest.mark.external`(R이나 Fortran 컴파일러를 실제로 부르는 테스트; 지금은 없다).
  - 고르지 않은 등급의 테스트는 deselect된다(요약에 `N deselected`). 파일 하나의 deep 테스트를 돌리려면 `SD_TIER=deep`을 함께 준다.
- **예제 수**는 테스트마다 `examples(n)`(`tests/_tier.py`)으로 쓴다. quick에서 n, deep에서 10n이다. 옛 환경 변수 `SDLA_EXAMPLES`, `SDPD_EXAMPLES`는 없앴다.
- **병렬**: `pytest`만 치면 한 프로세스다. `tools/test.py`는 `serial`이 아닌 테스트를 pytest-xdist 워커 `-n`개(기본 6, 최대 8)로 돌리고, 그 뒤에 `serial` 테스트를 혼자 돌린다. `serial`은 다른 테스트 프로세스와 기계를 나누면 안 되는 모듈이다(GIL 해제를 벽시계로 재는 것, 시간 제한이 있는 하위 프로세스, 물리 메모리 근처의 할당; 목록은 `tests/conftest.py`의 `SERIAL`).
- **테스트 수**(2026-11-20에 센 값): quick 10,484개(그 가운데 serial 39개), deep 10,626개(deep 마커 142개). 지금 수는 모으기만 해서 센다: `.venv/bin/python -m pytest --collect-only -q -p no:cacheprovider | tail -1`(deep은 앞에 `SD_TIER=deep`).
- 잰 시간(2026-11-20, 이 Mac, 그때 quick 9,348개): 한 프로세스 약 150초·최대 RSS 3.7GB, `tools/test.py` 70초(병렬 34초 + serial 36초)·워커 합 약 9.4GB. 워커를 8개로 늘려도 빨라지지 않고(48초) GIL 테스트가 부하로 실패했다. 워커마다 sd 스레드 수는 기본값 그대로가 가장 빨랐다(2로 줄이면 51초).
- `--strict-markers`라서 등록하지 않은 마커는 에러다. 마커는 `deep`, `external`, `serial`이다.
- **numpy·scipy와 견주는 테스트**: 반올림이 아닌 동작(dtype, 정수 연산, 정렬 순서, NaN·±0, 상태 값, 기본값, 멈춤 규칙)은 정확히 견준다. 반올림 크기의 값은 정확도 계약으로 견주고, 비트까지 같아야 하는 기준은 sd 자신의 골든이다(§2; API.md §4). 지금 numpy·scipy와 비트까지 견주는 테스트는 통과하는 동안 그대로 둔다. 의도한 고침이나 최적화 때문에만 깨지면 허용오차와 반복 수(최적화)의 비교로 바꾼다.

## 2. 골든 (비트 동일성)

공개 함수의 결과를 정해진 입력에서 비트까지 저장해 두고, 결과가 바뀌면 안 되는 변경(옮기기, 리팩터링, 포매터)의 전후를 비교한다.

```bash
.venv/bin/python tools/golden.py check            # 모든 영역을 돌려 저장된 결과와 비교
.venv/bin/python tools/golden.py check special    # 영역만
.venv/bin/python tools/golden.py record           # 다시 기록 (결과를 일부러 바꾼 뒤)
.venv/bin/python tools/golden.py list             # 영역별 케이스 수
```

- 케이스는 `tests/golden/cases_<영역>.py`, 기록은 `tests/golden/data/<영역>.json`이다. 영역은 array, linalg, random, special, stats, frame, regression, optimize다.
- 영역마다 스레드 수 1, 3, 10에서 따로 프로세스를 띄워 돌리고(동시에 최대 8개), 세 결과가 같아야 한다. 스레드 수에 따라 결과가 다른 케이스는 기록할 때 `THREADS`로 알린다.
- 전체 비교는 약 20초이고 작업 프로세스의 메모리 합은 최대 약 16GB다.
- 저장 형식: float는 `float.hex`(NaN은 기본 NaN이 아니면 비트까지), 원소가 1024개 이하인 배열은 원소마다, 그보다 큰 배열은 dtype·shape·strides·바이트의 SHA-256과 정해진 표본 원소다. 에러도 결과다(형과 메시지). stardust 클래스의 객체는 공개 속성 전부와 인자 없이 부를 수 있는 메서드 전부를 저장한다. 케이스가 낸 경고도 저장한다.
- `check`의 출력: 바뀐 케이스마다 어느 스레드 수에서 어디가 바뀌었는지(배열은 바뀐 원소 수, 첫 위치, 옛 값과 새 값의 hex·10진·ULP 거리), 새 케이스(`NEW`), 없어진 케이스(`GONE`)를 적는다. 하나라도 있으면 종료 코드가 1이다.
- 옵션: `--case 정규식`(이름이 맞는 케이스만; `record`면 그 케이스만 고쳐 쓴다), `--threads 1,3,10`, `--jobs N`(최대 8), `-v`(차이를 모두 적는다), `--slow N`(가장 느린 케이스 N개).
- 멈춘 케이스: 케이스 하나가 `GOLDEN_TIMEOUT`초(기본 120)를 넘기면 그 작업 프로세스를 끝내고 케이스 이름을 알린다. `GOLDEN_TRACE=1`이면 작업 프로세스가 케이스 이름을 시작할 때마다 찍는다.
- 결과를 일부러 바꾼 변경이면 `check`로 바뀐 케이스가 의도한 것뿐인지 본 뒤 `record`로 다시 기록하고 함께 커밋한다.
- 새 공개 함수를 더하면 그 영역의 `cases_<영역>.py`에 케이스를 더한다. 케이스는 `(이름, 인자 없는 함수)`를 내는 생성기이고, 입력은 고정 리터럴과 `L.rng(seed)`로만 만든다.

## 3. 커버리지

C++ 코어는 clang의 소스 기반 커버리지(`-fprofile-instr-generate -fcoverage-mapping`), Python은 pytest-cov(coverage.py)로 잰다.

```bash
.venv/bin/python tools/coverage.py all              # 계측 빌드, 테스트, 보고서
.venv/bin/python tools/coverage.py all --golden     # 골든 비교도 계측 코어로 돌린다
.venv/bin/python tools/coverage.py run tests/special   # 일부 테스트만 (pytest 인자를 그대로 넘긴다)
.venv/bin/python tools/coverage.py report           # 보고서만 다시
```

- 계측 빌드는 `build/coverage/cmake`에 따로 만든다. 평소 빌드(`build/<휠 태그>`)와 `CMakeLists.txt`는 건드리지 않는다. 같은 CMake 프로젝트에 계측 플래그만 더한 Release(-O3) 빌드이고, 처음에는 몇 분, 그 뒤로는 바뀐 파일만 다시 컴파일한다.
- `--golden`의 비교는 `--any-nan`이다: 계측한 코드는 기계어가 달라서 몇 케이스에서 NaN 결과의 부호가 다르다(2026-11-20에 10케이스). 그 밖의 비트는 모두 같아야 한다.
- 계측 코어에서만 죽는 테스트는 `tools/coverage.py`의 `CRASHES`에 적어 빼고 돌린다. 지금은 하나다: `sd.corrcoef`가 변수 0개에서 0으로 나누는 버그(`csrc/array/ops/covariance.cpp`의 `kParallelTask / p`). 평소 빌드에서는 arm64의 정수 나눗셈이 0을 내서 우연히 지나간다. 고쳐지면 목록에서 지운다.
- 테스트는 `src/`의 stardust에 계측한 `_core`를 끼워 돌린다(`build/coverage/site/sitecustomize.py`가 `SD_COVERAGE_CORE`가 있는 모든 프로세스에서 한다. 하위 프로세스도 센다).
- 보고서는 모두 `build/coverage/`에 생기고 커밋하지 않는다.
  - `summary.md`: 영역별 C++ 줄·함수·분기, Python 줄·분기의 비율
  - `<영역>.md`: 파일별 비율과 실행되지 않은 분기의 목록(`파일:줄:열`, 안 간 쪽, 그 줄의 소스)
  - `html/cpp/`, `html/py/`: 줄마다 실행 수와 분기 수를 단 소스
- C++ 분기는 소스 위치마다 한 번 센다(템플릿 인스턴스의 수를 합친다). `true never`·`false never`는 한 번도 가지 않은 쪽, `never reached`는 조건을 한 번도 평가하지 않은 분기다.

## 4. 정확도 측정

Arb 참값 대비 오차를 함수마다 재서 표로 만들고, 이전 측정보다 나빠진 것을 찾는다. 속도 체계(`bench/speed`)와 같은 꼴이다.

```bash
.venv/bin/python -m bench.accuracy run                         # 모든 영역을 재서 build/accuracy/<실행>.json
.venv/bin/python -m bench.accuracy run --area special --match erf
.venv/bin/python -m bench.accuracy compare build/accuracy/<실행>.json   # 기준선보다 나빠진 것
.venv/bin/python -m bench.accuracy promote build/accuracy/<실행>.json   # 기준선에 넣고 accuracy.md 다시 만들기
.venv/bin/python -m bench.accuracy table                       # 기준선에서 표만 다시
```

- 스윕은 테스트가 쓰는 참값 파일 그대로다: `tests/special/ref`(파일 98개 가운데 함수의 참값 97개; `reduce_cases`는 테스트만 읽는다, `tools/special_ref.py`), `tests/stats/ref`(555개, `tools/stats_ref.py`). 파일마다 입력, 참값(hi + lo), 보장 영역, 계약의 배율(`scale`), 출처(`arb`, `target_bits`; `refdata.provenance`가 읽는다)가 있어서 실행은 stardust 쪽만 계산한다(두 영역 합 약 1초). 형식은 `tests/special/refdata.py`에 있다.
- 새 참값은 같은 폴더에 커밋하고 영역당 60MB까지 둔다(2026-11-20에 `tests/stats/ref` 57MB, `tests/special/ref` 11MB).
- 지표는 (함수, 영역)마다: 최대·99.9% ULP, 배율이 1보다 큰 점이 있는 함수의 ULP/배율(역함수의 max(1, κ), Bessel·Airy·복소 오차함수·wofz의 영점 근처 포락선 배율), 0을 지나는 값의 mixed(|오차|/max(1, |값|), 2⁻⁵² 단위), 최대 상대오차, 계약 대비(1 이하가 계약 안). 회귀는 참값이 유한한데 sd가 NaN·inf를 낸 점의 수도 센다. 계약 판정은 `tests/<영역>/test_reference.py`의 것을 그대로 부른다.
- `run`은 계약을 넘는 것이 있으면 종료 코드 1이다. `compare`는 계약을 넘거나, 계약 안이라도 어느 지표든 기준선보다 커진 것을 모두 `WORSE`로 적고 종료 코드 1이다.
- 기준선은 `bench/accuracy/results/<영역>.json`(커밋한다), 표는 `bench/<영역>/accuracy.md`(기준선에서 만든다. 손으로 고치지 않는다), 실행 파일은 `build/accuracy/`(커밋하지 않는다).
- 영역은 special, stats(단변량), multivariate, regression이다.
  - multivariate: 다변량·행렬 분포 7개의 log 밀도·밀도·엔트로피(참값 `tests/stats/ref_mv`, `tools/mv_ref.py`로 만든다; 2.4MB, 8프로세스로 약 3초)와 다변량 정규·t의 cdf(`tests/stats/ref_mvcdf`). 계약과 영역은 `tests/stats/test_multivariate.py`·`test_matrix.py`·`test_mvcdf.py`의 것이고, 표는 `bench/stats/accuracy_multivariate.md`다.
  - regression: 문제별 고정 참값(linear 17, glm 17, penalized 21, robust 19). 필드마다 상대오차와 ULP이고, 수치 계약이 없어 기준선과의 비교만 한다.
  - linalg, array, optimize는 아직 없다.
