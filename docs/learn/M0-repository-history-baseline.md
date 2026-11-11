# M0. 저장소, 이력, 기준선

> 통합 1단계를 시작하기 전의 준비 마일스톤이다. 코드는 하나도 옮기지 않았다. 옛 세 패키지를 그대로 빌드해서 "지금 무엇이 어떻게 동작하는가"를 기록해 두는 것이 목적이다. 이후 마일스톤은 모두 이 기록과 비교한다.

## 1. 무엇을 했나
| 항목 | 결과 |
|---|---|
| 새 저장소 | `~/Desktop/stardust/stardust`, 브랜치 `main` |
| 옛 이력 | stardust-numpy 47커밋, stardust-linalg 16커밋, stardust-pandas 17커밋이 `legacy/numpy`, `legacy/linalg`, `legacy/pandas` 아래로 이력과 함께 들어왔다 |
| 개발 환경 | `.venv`(새 패키지), `.venv-legacy`(옛 세 패키지). 둘 다 Python 3.15.0, 의존성 버전은 옛 프로젝트와 같다 |
| 뼈대 | `import stardust as sd`, 빈 `_core`(툴체인 확인만), 테스트 3개 통과 |
| 옛 테스트 | `.venv-legacy`에서 numpy 2085개, linalg 339개, pandas 388개 모두 통과 |
| 골든 출력 | 케이스 12,424개 (배열 7,746, linalg 1,844, frame 2,834). 스레드 수 1, 3, 10에서 모두 같은 결과, 두 번 기록해도 바이트까지 같다 |
| 벤치마크 기준선 | 세 프로젝트의 최종 보고서 줄 324개 (배열 17, linalg 258, frame 49)를 한자리에서 2라운드 잼 (22분) |

## 2. 이력을 어떻게 들여왔나

### 2.1 목표
나중 마일스톤에서 `git mv legacy/numpy/csrc/ops/matmul.cpp csrc/array/ops/matmul.cpp`를 했을 때, `git log --follow csrc/array/ops/matmul.cpp`가 stardust-numpy 시절의 커밋까지 거슬러 올라가야 한다. 파일 하나의 역사를 읽을 수 있어야 "왜 이렇게 짰는가"를 커밋 메시지에서 찾을 수 있다.

### 2.2 첫 시도: 표준 명령만으로 (실패)
처음에는 이 Mac의 git에 `git subtree`도 `git filter-repo`도 없어서, 표준 명령으로 하는 "subtree merge"를 썼다.
```bash
git fetch ../stardust-numpy main
git merge -s ours --no-commit --allow-unrelated-histories FETCH_HEAD
git read-tree --prefix=legacy/numpy/ -u FETCH_HEAD
git commit
```
커밋은 모두 들어왔지만, 옛 커밋 안에서 파일 경로는 여전히 `csrc/ops/matmul.cpp`다. 그래서 `git log -- legacy/numpy/csrc/ops/matmul.cpp`는 가져오기 커밋 하나에서 멈췄다. 경로가 중간에 갑자기 바뀌는데, git은 그 바뀜을 "이름 바꾸기"로 보지 못한다.

### 2.3 두 번째 시도: 경로를 다시 쓴 이력 (채택)
`git filter-repo`를 임시 venv에 설치하고(저장소 밖, scratch 디렉터리), 옛 저장소의 **복제본**에서 모든 커밋의 경로 앞에 `legacy/numpy/`를 붙였다.
```bash
git clone --no-local ../stardust-numpy /tmp/clone-numpy
cd /tmp/clone-numpy && git filter-repo --to-subdirectory-filter legacy/numpy
cd stardust && git fetch /tmp/clone-numpy main && git merge --allow-unrelated-histories FETCH_HEAD
```
- 옛 커밋 안에서도 경로가 `legacy/numpy/...`라서, `git log --follow`가 끝까지 이어진다. 확인: `git log --follow -- legacy/numpy/csrc/ops/matmul.cpp`가 M8 커밋까지 나온다.
- 대가: 커밋 해시가 바뀐다(경로가 다르니 다른 커밋이다). 옛 해시는 가져오기 커밋 메시지에 적었다.
- 옛 저장소는 건드리지 않았다. 복제본에서만 다시 썼다.

## 3. 환경을 둘로 나눈 이유
- **`.venv-legacy`**: 옛 세 패키지를 `legacy/`의 마지막 커밋에서 그대로 빌드해 설치했다(편집 모드 아님).
  - 골든 출력과 벤치마크 기준선을 만드는 데만 쓴다.
  - 옛 프로젝트 폴더의 `.venv`를 쓰지 않은 이유: stardust-linalg의 `.venv`에 설치된 stardust-numpy는 언젠가의 복사본이라 지금 커밋과 같다는 보장이 없다. 기준은 커밋 하나로 정확히 정해야 한다.
- **`.venv`**: 새 패키지 `stardust`의 편집 모드 설치. 정답지(numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, Polars 2.1.0)도 여기에 있다.

## 4. 골든 출력

### 4.1 왜 테스트만으로는 부족한가
옛 테스트(약 2,800개)는 대부분 numpy, scipy, pandas를 정답지로 삼는다. 그래서 "numpy와 같은가"는 보지만, "옮기기 전과 비트까지 같은가"는 보지 못한다.
- 예: numpy와 일부러 다르게 만든 곳(축 0 리덕션의 합 순서, `prod`의 오버플로)은 테스트가 허용 오차로 비교한다. 이 부분이 옮기다가 바뀌어도 테스트는 통과한다.
- F-order로 돌려주는 linalg 결과의 **strides**처럼 값이 아닌 것은 테스트가 거의 보지 않는다.
- 에러 메시지는 테스트가 일부만 본다.

### 4.2 어떻게 만들었나
- `bench/merge/golden/golden.py`가 틀이다. 영역마다 `cases_<영역>.py`가 `(이름, 함수)` 쌍을 낸다.
- **케이스 파일 하나를 옛 패키지와 새 패키지 양쪽에서 돌린다.** 케이스는 `L.A.sum(...)`, `L.LA.solve(...)`, `L.F.Series(...)`처럼 모듈을 `L`을 통해서만 부른다. `L`은 `record`일 때 옛 모듈(`stardust_numpy` …), `check`일 때 새 모듈(`stardust` …)을 가리킨다.
- 결과는 이렇게 저장한다.
  - 배열: dtype, shape, strides, 그리고 바이트의 SHA-256. 값과 레이아웃이 모두 비교된다.
  - Python 스칼라: float는 `float.hex()`(NaN은 페이로드 비트까지).
  - 에러: 종류와 메시지. 메시지는 옛 패키지 이름을 새 이름으로 바꾼 뒤 비교하고, 메시지만 다른 경우는 따로 보고한다(nanobind 시그니처 문자열 같은 것은 옮기면서 바뀔 수 있다).
  - frame: Series는 dtype, 이름, NA 마스크, 값 비트, index를 재귀적으로. `cases_frame.py`의 `encode`가 공개 API만으로 한다.
- 케이스마다 스레드 수 1, 3, 10에서 돌린다. 옛 패키지에서 셋이 다르면 "unstable"로 따로 표시하게 했는데, **하나도 없었다.** 세 프로젝트의 "결과는 스레드 수와 무관하다"는 약속이 12,424개 케이스에서 지켜졌다.

### 4.3 무엇을 덮나
| 영역 | 케이스 | 의도한 에러 | 기록 시간 | 덮은 것 |
|---|---:|---:|---:|---|
| 배열 | 7,746 | 1,826 | 9.5초 | `__all__` 전부, ndarray의 메서드·속성·연산자, dtype 4개 × 레이아웃(C, F, strided, 음수 stride, broadcast, 0크기, 5차원) × 크기(스레드 기준 2¹⁵의 양쪽, 1M 원소) |
| linalg | 1,844 | 247 | 7.4초 | 37개 함수 전부, 실수/복소수 경로, 큰 C-order 입력의 풀 전치 경계(362/363), 배치 (1000,3,3)과 (100,32,32), 긴 합(32767/32768항), 특이 행렬의 배치 인덱스 |
| frame | 2,834 | 366 | 20초 | API §21의 94개 이름 전부, 기준 2¹⁷의 양쪽(300k, 1M행), groupby, merge, 정렬, CSV 7MB(청크 경계를 넘는 따옴표 필드) |

- 덮었는지는 글자 검색이 아니라 **실제로 불린 이름을 추적해서** 확인했다(모듈을 대리 객체로 감싸서 접근을 기록).
- 케이스가 공유 입력을 바꾸지 않았는지 마지막 케이스가 확인한다.

### 4.4 쓰는 법
```bash
.venv-legacy/bin/python bench/merge/golden/golden.py record array   # 옛 패키지 → array.json (다시 만들 때만)
.venv/bin/python bench/merge/golden/golden.py check array           # M1부터: 통합판을 비교
```

### 4.5 기록하다가 찾은 것
옛 패키지의 동작 그대로 기록했다. 일관성 문제는 DESIGN §8.4에 옮겨 적었다.
- 배열: `full_like(int 배열, 1.5)`가 조용히 자른다(암묵적 하향 변환은 에러라는 규칙과 어긋남). `2**64`를 `divide`, `atan2`만 받는다. 이름이 빈 에러 메시지("property ''").
- frame: `df.iloc[0, 9]`의 메시지가 열인데도 "rows"라고 한다. `groupby(Series)`의 엉뚱한 메시지.
- linalg: `cond(ones((3,3)))`가 inf가 아니라 1.5e48이다. 특이값이 정확히 0이 아니라 아주 작아서다. 버그는 아니다.

## 5. 벤치마크 기준선
- `bench/merge/baseline.sh`: 세 프로젝트의 최종 보고서 스크립트를 `.venv-legacy`에서 그대로 돌렸다. 기본 스레드(10)와 1스레드, 그리고 비교 대상(numpy, numpy/scipy, pandas, Polars)을 같은 자리에서 2라운드 재고, 행마다 빠른 값을 남긴다.
- 표: `bench/merge/baseline.md` (`baseline_table.py`가 만든다). M7에서 같은 스크립트를 통합판에 돌려 `--compare`로 비교한다.

| 영역 | 줄 | 비교 대상 / 옛 패키지 (기하평균) | 옛 패키지가 빠름 / 비슷 / 느림 |
|---|---:|---|---|
| 배열 | 17 | numpy 5.37배 | 16 / 0 / 1 |
| linalg | 248 (비교 가능한 줄) | numpy·scipy 3.06배 | 189 / 49 / 10 |
| frame | 49 | pandas 4.64배, Polars(14스레드) 1.78배 | pandas 대비 44 / 4 / 1 |

- **측정 환경이 믿을 만한가**: linalg는 옛 최종 보고서의 195 / 43 / 10과 거의 같다. 배열의 `f + g` 39.8µs, `sum(f)` 23.9µs도 옛 보고서(39µs, 24µs)와 같다. 같은 코드를 같은 기계에서 재면 같은 값이 나오는 상태다.
- 배열에서 numpy보다 느린 한 줄은 `tiny a + b`(180ns 대 158ns)다. 옛 보고서에도 있던 것이다.

## 6. 리뷰 포인트
1. **이력 보존 방식**: 커밋 해시가 바뀌는 대신 `git log --follow`가 옛 커밋까지 이어진다. 옛 해시는 가져오기 커밋 메시지에 남겼다.
2. **모두 git에 넣는다** (사용자 결정, DESIGN §5.2): 처음에는 세 프로젝트의 정책대로 `bench/`와 문서를 로컬 전용으로 두었다. 그러면 골든 케이스 파일(약 4,100줄)이 이 Mac에만 남는다는 점을 물었고, 사용자가 "보통 git에 넣지 않는 것만 빼고 모두 넣는다"로 정했다. 설계 문서, 이 해설, bench 스크립트, 측정 결과(약 3.6MB), 골든 출력이 모두 커밋에 들어갔다.
3. **일관성 원칙(DESIGN §1.4)과 2라운드(§8)**: R1(Array API 관례가 기준), R2(`correction=0`), R3의 6번(`astype`의 NaN → 0, 포화)이 정해졌다. 남은 것(R3의 나머지 행, R4 함수별 목록)은 M6 전에 정한다. M1–M5는 옮기기만 하므로 이 결정을 기다리지 않는다.
4. 다음은 M1(배열 코어 옮기기)이다. 끝나면 `golden.py check array`가 7,746개 모두 같아야 하고, `docs/design/array.md`와 `docs/api/array.md`를 새로 쓴다.
