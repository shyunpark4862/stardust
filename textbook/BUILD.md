# 공용 빌드 (책의 모든 실행과 측정에 쓴다)

## 기준 커밋
- `66fe52ed7a72f006f1ca7f59689c4b86de5911c1` (2026-11-25 18:38:11 +0900, "textbook: chapter 10 (Floating-Point Numbers) under the new standard")
- 그때의 HEAD가 그대로 빌드되고 `import stardust`가 되었다. 커밋을 거슬러 올라갈 필요가 없었다.
- 작업 중인 저장소의 커밋하지 않은 변경(`sd.cluster` 등)은 들어 있지 않다.

- **기준 커밋을 올리지 않는다** (2026-11-27 사용자 지시): 사용자가 따로 말하기 전까지 공용 빌드는 66fe52e에 고정한다. 작업 저장소의 라이브러리가 바뀌어도 복제본을 pull·재빌드하지 않고, 책의 실행·측정·API 대조는 계속 이 빌드로 한다.

## 위치
- 복제본: `/Users/sanghyunpark/Desktop/stardust/textbook/.sdbuild/stardust` (git 무시)
- **인터프리터: `/Users/sanghyunpark/Desktop/stardust/textbook/.sdbuild/stardust/.venv/bin/python`**
- 확장 모듈: `.sdbuild/stardust/build/cp315-cp315-macosx_26_0_arm64/install/platlib/stardust/_core.cpython-315-darwin.so`
- 작업 중인 저장소(`/Users/sanghyunpark/Desktop/stardust`) 안에서 그 저장소의 `.venv`나 `src/`로 실행하지 않는다. 스크립트는 복제본 밖(예: scratchpad)이나 `textbook/` 안에서 위 인터프리터로 돌린다. `sd.__file__`이 `.sdbuild/stardust/src/stardust/__init__.py`인지 보면 확인된다.

## 폴더 이름 변경 (2026-11-27)
- 책 폴더가 `textbook-eng`에서 `textbook`으로 바뀌었다(옛 한글판 `textbook/`은 삭제). 가상환경의 경로를 새 이름으로 고치고, 복제본의 `build/`를 지운 뒤 같은 커밋(66fe52e)에서 `pip install --no-build-isolation --no-deps --no-index -e .`로 확장 모듈을 다시 빌드했다(내려받은 것 없음). 아래 빌드 명령의 경로도 새 이름으로 읽는다.

## 기계
- Apple M4 Pro (14코어: 성능 10 + 효율 4), 메모리 48 GB
- macOS 26.6.2 (25G83), SDK 26.5
- Apple clang 21.0.0 (clang-2100.1.1.101), CMake 4.4.3, Ninja 1.13.2, 빌드 형식 Release
- Python 3.15.0 (`/opt/homebrew/bin/python3.15`), pip 26.2.1

## 빌드 명령 (README.md의 방법 그대로, 복제본 안에서)
```bash
git clone /Users/sanghyunpark/Desktop/stardust /Users/sanghyunpark/Desktop/stardust/textbook/.sdbuild/stardust
cd /Users/sanghyunpark/Desktop/stardust/textbook/.sdbuild/stardust
python3.15 -m venv .venv && .venv/bin/pip install nanobind scikit-build-core
CMAKE_BUILD_PARALLEL_LEVEL=8 .venv/bin/pip install --no-build-isolation -e ".[test]"
```
- 빌드는 약 76초(8병렬)였다. 크기는 `.venv` 560 MB, `build/` 78 MB.
- `[test]` 묶음(numpy, scipy, pandas, polars, python-flint, statsmodels, scikit-learn, pytest 등)은 책에서 정답지와 견줄 때 쓰려고 함께 넣었다. 테스트와 벤치는 돌리지 않았다. `[bench]` 묶음(tqdm, pyarrow, threadpoolctl)은 넣지 않았다(threadpoolctl은 scikit-learn의 의존으로 들어왔다).
- mpmath는 설치되어 있지 않다. GUIDE.md §5에 따라 오차 측정에 쓰려면 따로 넣어야 한다(아직 넣지 않았다).

## 내려받은 것 (PyPI)
- 1단계: nanobind 3.1.0, scikit-build-core 1.1.2, packaging 26.4, pathspec 1.1.1 (캐시에서 왔는지는 기록하지 않았다)
- 2단계, 새로 내려받음: cloudpickle 3.1.3, coverage 7.17.5, execnet 2.1.2, formulaic 1.2.2, hypothesis 6.170.17, interface_meta 2.0.1, narwhals 2.29.0, numpy 2.5.5, patsy 1.0.3, polars 2.2.0, polars_runtime_32 2.2.0 (45.6 MB), pytest-cov 7.1.0, pytest-xdist 3.8.0, python-flint 0.9.0, scikit-learn 1.9.1, statsmodels 0.15.0, wrapt 2.6.0.dev1 (formulaic의 의존 `wrapt>=1.17.0rc1`이 사전 공개판을 허용해서 dev 판이 골라졌다)
- 2단계, pip 캐시에서: iniconfig 2.3.1, joblib 1.6.0, pandas 3.1.1, pluggy 1.7.0, pygments 2.21.0, pytest 9.2.0, python-dateutil 4.0.0, scipy 1.18.1, sortedcontainers 2.4.0, threadpoolctl 3.8.0, typing-extensions 4.16.0, tzdata 2026.6
- 작업 저장소의 `.venv`와 버전이 조금 다르다(numpy 2.5.5 대 2.5.4, polars 2.2.0 대 2.1.0). 정답지 수치를 인용할 때는 이 빌드의 버전을 적는다.

## 확인
- `import stardust as sd` 약 1초. 영역마다 한 번씩 부른 것(`sd.sum`, `linalg.solve`, `frame.DataFrame`, `special.gamma`, `stats.Normal().cdf`, `random.default_rng().random`, `regression.ols`, `optimize.minimize_scalar`, `htest.ttest_1samp`, `sparse.csr_array`, `integrate.quad`, `differentiate.derivative`, `spatial.pdist`, `graph.shortest_path`, `interpolate.CubicSpline`, `fft.fft`, `smooth.savgol_coeffs`, `plot.line`)이 모두 동작했다.
- `stardust.cluster`, `stardust.embed`, `stardust.classify`는 이 커밋에 없다.

## 다시 빌드하기
- 같은 커밋 그대로: 편집 가능 설치(`editable.rebuild = true`)라 소스가 바뀌면 import할 때 C++ 코어가 저절로 다시 빌드된다. 복제본의 소스는 건드리지 않는다.
- 기준 커밋을 바꿀 때: 복제본에서 `git fetch origin && git checkout <커밋>` 뒤 위의 마지막 `pip install` 줄을 다시 돌린다(또는 그냥 import하면 다시 빌드된다). 바꾼 커밋과 날짜를 이 문서에 적고, API-INVENTORY.md를 `tools/api_inventory.py`로 다시 뽑는다.
- 처음부터: `.sdbuild/stardust`를 지우고 위 명령을 처음부터 돌린다.
