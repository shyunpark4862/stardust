# stardust 설계 문서

> 상태: **확정, 일부 2라운드 진행 중** (2026-11-12). 1라운드 질문(§6)은 Q9만 사용자가 다른 안을 골랐고, 나머지는 추천안을 수락했다. 일관성 원칙(§1.4)이 더해져 §4.4와 시그니처·이름 규칙을 2라운드에서 정한다. M1–M5(옮기기)는 2라운드와 상관없이 진행할 수 있다.

stardust-numpy, stardust-linalg, stardust-pandas를 하나의 라이브러리 `stardust`로 합친다. 앞으로 영역을 하나씩 더해 가며 키운다.

- 이 문서는 통합에 관한 확정 사항의 단일 기준이다.
- 세 프로젝트에서 이미 확정한 결정은 **이 문서가 바꾸지 않는 한 그대로 유효하다.** 바뀌는 것은 §2에 모았다.
  - 옛 문서는 옮기지 않고 새로 쓴다 (§5.4). 새 영역 문서가 생기면 그 영역의 옛 결정은 새 문서가 대신한다.
- 내부 구조는 [ARCHITECTURE.md](ARCHITECTURE.md), 진행 순서는 [ROADMAP.md](ROADMAP.md)에 있다.

---

## 1. 기반 (사용자 결정 2026-11-12)

### 1.1 이름과 확장 방식
- 패키지 이름은 `stardust`, import는 `import stardust as sd`다.
- **하나의 라이브러리를 점점 키운다.** 새 영역은 하위 모듈로 붙인다.
  - 지금 있는 영역: 배열(최상위 `sd`), 선형대수(`sd.linalg`), 데이터프레임(`sd.frame`)
  - 나중의 영역도 같은 방식이다 (예: linalg DESIGN 머리말의 "sklearn류").

### 1.2 두 단계로 합친다
- **1단계: 배열 + linalg를 깊게 합친다.** 둘은 같은 배열을 다룬다. linalg가 배열의 C++ 내부를 직접 쓴다 (§3).
- **2단계: pandas를 `sd.frame`으로 옮긴다.** Column 데이터 모델은 그대로 두고, 아래층(Buffer, 스레드풀, pairwise 합, 빌드)만 공유한다 (§4).

### 1.3 구조와 구현을 고쳐도 된다
- 단순히 합치는 데서 그치지 않고, 구조와 구현을 많이 고쳐도 된다 (사용자 2026-11-12).
- 다만 **옮기기와 고치기를 같은 마일스톤에 섞지 않는다** (ROADMAP 진행 원칙). 옮기는 단계는 결과가 비트까지 같아야 한다.

### 1.4 일관성을 가장 앞에 둔다 (사용자 결정 2026-11-12)
- **영역 사이의 일관성을 중시한다.** 같은 개념은 어느 영역에서든 같게 동작하고, 같은 이름과 같은 시그니처를 쓴다.
  - 동작: 같은 연산이 같은 입력에 같은 결과(값, dtype, 에러 종류)를 낸다.
  - 시그니처: 같은 개념의 인자는 이름, 기본값, 위치/키워드 규칙이 같다.
  - 이름: 같은 연산은 같은 이름을 쓴다.
- 차이를 남기려면 이유가 데이터 모델에 있어야 한다(예: frame에만 NA가 있다). 그 차이도 문서 한곳에 모은다.
- 이 원칙에 따라 §4.4의 판단 기준과 추천안을 2라운드에서 다시 정한다. 시그니처와 이름의 차이도 2라운드에서 함께 정한다.
- 세 프로젝트가 각자 갖고 있던 일관성 규칙(linalg의 "모든 함수가 일관된 시그니처와 동작", pandas API §11 시그니처 일관성)은 라이브러리 전체의 규칙 하나로 합친다.

### 1.5 이어받는 것
세 프로젝트에서 같았던 것은 그대로 둔다.
- 대상 기기: Apple M4 Pro 한 대. 언어: C++20 + nanobind. 빌드: scikit-build-core + CMake. Python: 3.15만.
- "얇은 Python, 두꺼운 C++". 경고를 내지 않는다. 결과는 스레드 수와 무관하게 비트까지 같다.
- 코드, 식별자, 코드 주석, 커밋 메시지는 영어, 문서는 한글 (세 저장소가 실제로 이렇게 했다).
- 정답지와 비교 기준: numpy, scipy, pandas, Polars (각 프로젝트에서 쓰던 버전 그대로)

---

## 2. 옛 결정 중 바뀌는 것

| 옛 결정 | 통합 후 |
|---|---|
| numpy: 패키지 `stardust-numpy`, `import stardust_numpy as sdnp` | `stardust`, `import stardust as sd` |
| linalg §1.1: 별도 패키지, `sdnp.linalg`로 넣지 않는다 | **폐기.** `sd.linalg` |
| linalg §1.3: 패키지 `stardust-linalg`, `sdla` | `sd.linalg` |
| linalg §2.2: 양방향 buffer protocol, stardust-numpy를 고치지 않는다 | **폐기.** `NDArray`를 직접 주고받는다 (§3.1). 1단계 M3에서 바꾼다 |
| linalg §2.5: 스레드풀 복사, 스레드 수는 `sdnp.get_num_threads()`를 따른다 | 풀 하나, 스레드 수 API 하나 (§5.1) |
| pandas §1.2: 독립 프로젝트, 부품은 복사, 공통 라이브러리로 분리하지 않는다 | 아래층은 공유한다. **"stardust-numpy를 백엔드로 쓰지 않고 Column을 직접 만든다"는 유지한다** (§4.1) |
| pandas §1.2: stardust-numpy 배열과 buffer protocol로 주고받는다 | 코드에는 구현되지 않았다(지금은 `sdnp.ndarray`를 넣으면 TypeError). 직접 변환으로 새로 만든다 (§4.3) |
| pandas §1.3: 패키지 `stardust-pandas`, `sdpd` | `sd.frame` |

---

## 3. 1단계: 배열 + linalg

### 3.1 경계를 없앤다
- linalg는 `sdnp.ndarray`를 buffer protocol로 읽고, 결과를 자기 블록에 만든 뒤 `sdnp.asarray`로 감쌌다. 별도 패키지라서 생긴 구조다.
- 통합 후에는 C++에서 `const NDArray&`를 직접 받고, 결과도 `NDArray`로 직접 만든다 (ARCHITECTURE §3).
- 없어지는 것: `Input`, `outblock`, `pyrefs`의 sdnp 참조, 중복된 `DType`·변환 규칙·pairwise 합, 스레드풀 복사본
- **기대 효과**: linalg M0에서 잰 경계 비용은 결과 하나에 약 200ns였다. 작은 행렬 호출(약 600–700ns)의 30% 가까이다. M3에서 실제로 줄어든 양을 잰다.

### 3.2 확장 모듈은 하나
- `_core` 하나에 모든 영역을 넣고, `def_submodule`로 `_core.linalg`, `_core.frame`을 만든다.
- 검토하고 채택하지 않은 안
  - **영역마다 확장 모듈**(`_core`, `_linalg`, `_frame`): 모듈마다 nanobind 타입 테이블과 정적 링크한 부품(스레드풀 등)이 따로 생긴다. 지금의 문제(풀 세 벌, 타입을 직접 못 넘김)가 그대로 남는다.
  - **공유 라이브러리(.dylib) + 영역별 모듈**: 위 문제는 풀리지만, 빌드와 설치(rpath, nanobind 공유 빌드)가 복잡해진다. 영역이 아주 많아져서 빌드 시간이 문제가 되면 그때 다시 본다.

### 3.3 linalg 결과의 메모리 레이아웃은 그대로
- LAPACK이 만든 F-order 결과를 그대로 돌려준다 (linalg §2.3).
- `array/`에 F-order 생성 함수를 하나 더한다 (ARCHITECTURE §3).

---

## 4. 2단계: frame

### 4.1 Column 데이터 모델은 그대로 둔다
- Column(1차원, validity 비트맵, 문자열 offsets)과 NDArray(N차원, 임의의 strides, 결측 없음)는 다른 물건이다.
- Column을 NDArray 위에 다시 올리면 pandas M1–M12의 설계를 뒤집게 되고, 얻는 것이 적다.
- 공유하는 것: Buffer, 스레드풀, pairwise 합, NEON 보조 함수, 빌드 플래그, dtype 정의(§4.2)

### 4.2 dtype
- **C++**: `base/dtype.hpp`의 enum 하나 (`Bool, Int64, Float64, Complex128, String, Datetime64`). 배열은 앞의 넷, Column은 `Complex128`을 뺀 다섯을 받는다 (ARCHITECTURE §4.3).
  - 지금은 numpy와 pandas가 같은 이름 `sd::DType`에 다른 값(`Complex128` / `String`)을 쓴다. 한 모듈에서는 어차피 둘 중 하나로 정리해야 한다.
- **사용자가 dtype을 쓰는 방식** (§6 Q5): 지금 둘이 다르다.
  - 배열: dtype 객체(`sdnp.float64`)만 받고, `.dtype`도 객체다. 문자열과 Python 타입은 거부한다.
  - frame: 문자열(`"float64"`)이나 Python 타입(`float`)을 받고, `.dtype`은 문자열이다.
  - **추천안**: 객체 `sd.bool, sd.int64, sd.float64, sd.complex128, sd.string`(+ 나중의 `sd.datetime64`)을 기준으로 한다. 두 영역 모두 객체와 그 이름 문자열을 받고, `.dtype`은 객체를 돌려준다. Python 타입(`float` 등)은 둘 다 받지 않는다(`sd.bool`이 Python `bool`과 헷갈리는 문제도 같이 없앤다).

### 4.3 Column과 배열 사이 (§6 Q7)
**추천안**
- **frame → 배열**: `Series.to_array()`, `DataFrame.to_array()`
  - 결측이 없는 숫자 컬럼: Buffer를 공유하는 **읽기 전용** 뷰 (복사 없음). 지금의 `to_numpy`와 같은 규칙이다.
  - 결측이 있으면 `na_value=`가 있어야 한다. 문자열 컬럼은 TypeError (배열에 문자열 dtype이 없다).
  - `DataFrame.to_array()`는 2차원 배열을 새로 만든다. 컬럼마다 따로 연속이므로 **F-order로 만들어 컬럼 하나를 memcpy 한 번에 복사한다.** linalg가 F-order를 원하므로 `sd.linalg.lstsq(df.to_array(), ...)`에서 패킹 비용도 줄어든다.
- **배열 → frame**: `sd.frame.Series(arr)`, `sd.frame.DataFrame(arr2d)`
  - **항상 복사한다.** 배열은 쓰기 가능한 뷰를 통해 바뀔 수 있어서, frame의 Copy-on-Write가 추적할 수 없다.
  - complex128 배열은 TypeError (frame에 complex가 없다).
- **Copy-on-Write**: 읽기 전용 배열 뷰가 Buffer 참조를 하나 쥐고 있으면, frame 쪽 쓰기는 참조 카운트를 보고 복사한다 (pandas ARCHITECTURE §3.3의 규칙이 그대로 동작). M6에서 테스트로 확인한다.
- 진짜 numpy와는 지금처럼 주고받는다: 배열의 buffer protocol, frame의 `to_numpy`와 `__array__`

### 4.4 동작 규칙의 차이 (§6 Q6)
두 프로젝트의 API.md와 코드를 대조했다 (2026-11-12). 같은 것: 정수 오버플로는 wrap, `/`는 float64, IEEE NaN 비교, NaN은 min/max/sum에 전파, 경고 없음, pairwise 합, 두 번 읽는 `var`, 스레드 수 API의 모양.

> **2라운드에서 다시 정한다** (§1.4, 2026-11-12). 아래 표의 "추천" 열은 1라운드의 판단 기준으로 쓴 것이다. 1라운드에서 수락했지만, 일관성 원칙(§1.4)에 따라 "유지" 항목을 다시 본다.

**1라운드의 판단 기준**: 데이터 모델에서 나오는 차이(frame에만 NA가 있다)와, 각 영역이 일부러 고른 엄격함의 차이(무엇을 에러로 막는가)는 남긴다. **같은 연산이 이유 없이 다른 값을 내는 것**은 통일한다.

| # | 항목 | 배열 (옛 numpy) | frame (옛 pandas) | 추천 |
|---|---|---|---|---|
| 1 | `int // 0`, `int % 0`, `int ** 음수` | 0, `a`, 정수 규칙 (실패하지 않음) | NA | **유지.** 배열에는 NA가 없어서 생긴 차이다 |
| 2 | bool 산술 (`True + True`) | int64로 승격 | TypeError (리덕션 `sum`은 허용) | **통일 → 배열 규칙.** 에러가 값이 될 뿐이라 기존 결과는 바뀌지 않는다 |
| 3 | int와 float 비교 | float64로 바꿔 비교 (`2**53+1 == 2.0**53`이 True) | 수학적으로 정확한 비교 (False) | **통일 → 정확한 비교.** 배열의 결과가 \|int\| > 2⁵³에서만 바뀐다. 비용을 M6에서 잰다 |
| 4 | 암묵적 하향 변환 (대입, `out=`) | dtype만 보고 TypeError | 값을 보고 판단 (2.0은 int 컬럼에 들어감, 1.5는 ValueError) | **유지.** 에러냐 값이냐의 차이다. 문서에 명시한다 |
| 5 | 생성자 `dtype=`의 손실 변환 | 명시적 변환으로 허용 (`[1.5]` → `[1]`) | 손실 없는 변환만 | **유지** (4와 같은 이유) |
| 6 | `astype` float → int의 NaN, 범위 밖 값 | NaN → 0, 포화 | ValueError | **유지** (4와 같은 이유) |
| 7 | `std`/`var`의 자유도 | `correction=0` (Array API) | `ddof=1` (pandas) | **유지.** numpy와 pandas 사용자의 기대가 각각 다르다. 문서 첫 화면에 적는다 |
| 8 | 빈 입력, 자유도 부족 리덕션 | NaN 또는 ValueError | NA | **유지** (1과 같은 이유) |
| 9 | 내림차순 정렬에서 NaN 위치 | 양방향 모두 맨 뒤 | NaN을 +inf보다 큰 값으로: 내림차순이면 맨 앞 | **통일, 방향은 사용자 결정.** 추천은 frame 규칙(NaN = 가장 큰 값). 내림차순이 오름차순의 거울이 되고, max가 NaN을 내는 것과 맞는다 |
| 10 | `unique`/키에서 NaN끼리 | 서로 다른 값 (Array API) | 같은 키, `-0.0`과 `0.0`도 같음 | **유지.** groupby에서 NaN마다 그룹이 따로 생기면 쓸모가 없다. 배열은 표준을 따른다 |
| 11 | `mean(int64)` | float64로 바꿔 블록 합 | int128로 정확히 합한 뒤 나눔 | **통일 → int128.** 더 정확하다. 배열의 축 리덕션에서 드는 비용을 M6에서 잰다 |
| 12 | 에러 종류 | 값 때문에 실패하는 일이 없다 | 값 때문이면 ValueError, 타입 때문이면 TypeError | 충돌 없음. frame 규칙을 라이브러리 공통 규칙으로 적는다 |

---

## 5. 공통

### 5.1 스레드
- `sd.set_num_threads(n, /)`, `sd.get_num_threads()` 하나만 둔다. 기본값은 P코어 10개 (두 프로젝트가 같았다).
- 풀은 하나, 병렬화 기준값은 영역마다 측정한 값을 그대로 쓴다 (ARCHITECTURE §4.1).
- 결정성 규칙은 두 프로젝트의 것을 그대로 합친다.
  - 기본: 나누는 방식은 데이터 크기로만 정한다.
  - frame의 예외(나누는 방식과 무관하게 결과가 정확히 정해지는 factorize, 안정 counting sort, CSV 읽기는 스레드 수로 나눈다)는 유지한다.
- 다른 점 하나를 M4에서 정리한다: GIL을 언제 놓는가. 배열은 실제로 병렬로 돌 때만 놓고, frame은 기준값 이상이면 1스레드에서도 놓는다.

### 5.2 git과 공개 범위
- **보통 git에 넣지 않는 것(가상환경, 빌드 산출물, 캐시, `.so`, `.DS_Store`)만 빼고 모두 git에 넣는다** (사용자 결정 2026-11-12, M0).
  - 설계 문서, 해설, bench 스크립트와 측정 결과, 골든 출력, experiments 모두 들어간다.
  - 세 프로젝트의 "문서와 bench, experiments는 로컬 전용" 정책은 이 저장소에서는 쓰지 않는다. `legacy/*/.gitignore`에서도 그 규칙을 뺐다.
- 세 저장소의 git 이력을 보존해서 들여온다 (ROADMAP M0).
- 옛 저장소 세 개는 통합이 끝나면 동결한다. 지우지 않고 README에 "stardust로 옮겨 감"을 적는다.

### 5.3 테스트와 벤치마크
- 테스트: `tests/array`, `tests/linalg`, `tests/frame`, `tests/interop`
- 통합 전후 비교: 옛 패키지 세 개로 만든 골든 출력, 같은 자리에서 다시 잰 벤치마크 기준선 (ROADMAP 진행 원칙)
- 세 프로젝트의 `test_api_surface.py`는 옛 API.md를 읽어 공개 이름과 시그니처를 맞춘다. 영역을 옮기는 마일스톤에서 새 API 문서를 함께 쓰고, 테스트가 그 문서를 읽게 바꾼다 (§5.4).

### 5.4 문서: 옮기지 않고 새로 쓴다 (1라운드 Q9, 사용자 결정 2026-11-12)
- 옛 문서(DESIGN, ARCHITECTURE, API, 안내서, 마일스톤 해설, bench 결과, experiments)는 **새 저장소로 옮기지 않는다.** 동결한 옛 저장소에 그대로 남는다.
- **새로 쓰는 문서**

  | 문서 | 내용 | 쓰는 때 |
  |---|---|---|
  | `DESIGN.md` (이 문서) | 공통 규칙과 통합 결정 | M0 전 (지금) |
  | `docs/design/<영역>.md` | 영역의 확정 사항: 옛 DESIGN과 ARCHITECTURE에서 **지금도 유효한 결정과 구조**만, 측정 근거와 함께. 결정의 이력(라운드, 날짜, 마일스톤별 변화)은 쓰지 않는다 | 그 영역을 옮기는 마일스톤 |
  | `docs/api/<영역>.md` | 영역의 공개 API. `test_api_surface`가 읽는 형식을 지킨다 | 그 영역을 옮기는 마일스톤, 규칙이 바뀌는 M6에서 고침 |
  | `API.md` | 영역별 API 문서의 색인, 공통 규칙(dtype, 에러 종류, 스레드) | M1에서 만들고 영역이 늘 때마다 더함 |
  | `docs/guide/` | 처음 보는 사람을 위한 안내서 하나 (옛 13장 + 13장을 영역을 넘나드는 한 권으로) | M7 |
  | `docs/learn/M<n>-*.md` | 통합 마일스톤의 해설 | 마일스톤마다 |
  | `README.md` | 라이브러리 소개 | M1에서 뼈대, M7에서 완성 |
- **코드 주석의 문서 참조**: 지금 코드와 테스트에는 옛 문서와 옛 마일스톤을 가리키는 참조가 약 750개 있다 (numpy 159, linalg 229, pandas 357).
  - 영역을 옮기는 마일스톤에서 그 영역의 참조를 새 문서로 바꾼다.
  - 옛 마일스톤 번호("M7.1에서 쟀다")는 지운다. 측정 근거가 필요한 주석은 근거를 `docs/design/<영역>.md`에 옮겨 적고 그쪽을 가리킨다.
  - 주석에 적힌 이유와 수치 자체는 지우지 않는다.
- bench 스크립트는 코드이므로 옮긴다(`bench/<영역>/`). 옛 측정 결과는 옮기지 않고, M0에서 기준선을 다시 잰다.

---

## 6. 1라운드 결과 (2026-11-12)
- **Q1. 폴더와 저장소** (추천안 수락): `~/Desktop/stardust/stardust`, 새 git 저장소, 세 저장소의 이력을 `legacy/` 경유로 보존해서 들여온다. 원격 저장소(GitHub)는 1단계가 끝난 뒤에 정한다.
- **Q2. 옛 저장소** (추천안 수락): 통합이 끝나면 동결한다 (§5.2).
- **Q3. 최상위 이름공간** (추천안 수락): 최상위 `sd`에는 배열만 둔다. 영역은 하위 모듈에만 둔다 (`sd.frame.DataFrame`).
- **Q4. 확장 모듈** (추천안 수락): `_core` 하나 (§3.2).
- **Q5. dtype 표기** (추천안 수락): 객체 기준. 두 영역 모두 객체와 이름 문자열을 받고, `.dtype`은 객체다. Python 타입은 받지 않는다 (§4.2).
- **Q6. 동작 규칙의 차이** (추천안 수락): §4.4 표의 추천안대로. 9번은 frame 규칙(NaN = 가장 큰 값, 내림차순이면 맨 앞).
- **Q7. Column과 배열 사이** (추천안 수락): §4.3대로.
- **Q8. 개발 도구** (추천안 수락): `.venv` + pip를 유지한다.
- **Q9. 문서** (사용자 결정): 옛 문서는 옮기지 않고 새로 쓴다 (§5.4).

## 7. 미확정
- 원격 저장소(GitHub)를 만들지, 만든다면 공개 범위 (1단계 뒤)
- §4.4의 3번(정확한 비교)과 11번(int128 평균)은 M6에서 비용을 잰 뒤 최종 확인한다. 비용이 크면 다시 묻는다.
- 2라운드 (§8)

---

## 8. 2라운드: 일관성 (진행 중, 2026-11-12)

**결정된 것 (사용자 2026-11-12, M0 끝)**
- **R1**: Array API 관례를 라이브러리 전체의 기준으로 한다 (추천안 수락).
- **R2**: `std`/`var`의 기본값은 `correction=0` (추천안 수락).
- **R3의 6번**: `astype` float → int는 배열 규칙(포화, NaN → 0)으로 통일한다 (추천안 수락).
- **남은 것**: R3의 나머지 행(1, 4, 5, 8, 10)과 R4의 함수별 목록. M6을 시작하기 전에 목록으로 확정한다.
세 공개 API를 대조했다: 동작은 §4.4, 시그니처와 이름은 아래. 대조는 `.venv-legacy`에서 `inspect`로 실제 시그니처를 확인했다.

### 8.1 R1. 기준을 하나로
지금 배열과 linalg는 **Python Array API 관례**를, frame은 **pandas 관례**를 따른다. 차이는 대부분 이 출신의 차이에서 나온다.

| 개념 | 배열·linalg (Array API) | frame (pandas) |
|---|---|---|
| 정렬 방향 | `descending=False` | `ascending=True` (`value_counts`만 `ascending=False`) |
| 정렬 안정성 | `stable=True` | 인자 없음 (항상 안정) |
| 자유도 | `correction=0` | `ddof=1` |
| 위치/키워드 | 배열 인자와 주 인자는 `/` 앞, 옵션은 `*` 뒤 | 주 인자도 키워드로 쓸 수 있다 (`s.astype(dtype=...)`, `merge(left=..., right=...)`) |
| dtype | 객체 (`sd.float64`) | 문자열, Python 타입 |
| `.dtype` | 객체 | 문자열 |
| 리스트 변환 | `tolist()` | `to_list()` |
| 고유값 | `unique_values` (정렬), `unique_counts` 등 namedtuple | `Series.unique()` (처음 나온 순서), `value_counts()` → DataFrame |
| 에러 메시지 | linalg는 `func(): ...`, 배열은 섞여 있음 | `(API.md §N)`을 붙임 |

- **추천: Array API 관례를 라이브러리 전체의 기준으로 한다.**
  - 배열과 linalg는 이미 이 관례다. 바뀌는 쪽은 주로 frame이다.
  - 배열에 없는 개념(NA, 라벨, groupby, merge, CSV)에만 frame 고유의 이름(`skipna`, `na_position`, `by`, `how`, `on`)을 쓴다. 그 이름도 위치/키워드 규칙과 기본값 규칙은 공통 규칙을 따른다.
- 대안: frame에 pandas 관례를 남긴다. §1.4와 맞지 않는다.

### 8.2 R2. `std`/`var`의 기본 자유도
- 인자 이름은 R1에 따라 `correction` 하나다. 기본값은 하나로 정해야 한다.
- **추천: `correction=0`** (Array API, numpy). 표본 분산은 `correction=1`로 명시한다.
- 대안: `correction=1` (pandas, 통계에서 흔히 쓰는 값). 배열의 결과가 바뀐다.

### 8.3 R3. 값 규칙 다시 정하기 (§4.4의 "유지" 항목)
| # | 항목 | 1라운드 | 2라운드 추천 |
|---|---|---|---|
| 1 | `int // 0`, `% 0`, `** 음수` | 유지 | **유지.** frame에만 NA가 있다(데이터 모델). 문서 한곳에 모은다 |
| 4 | 암묵적 하향 변환 | 유지 | **통일 → dtype만 보고 판단(배열 규칙).** frame에서 `2.0`을 int 컬럼에 넣으면 TypeError가 된다 |
| 5 | 생성자 `dtype=`의 손실 변환 | 유지 | **통일 → 명시적 변환으로 허용(배열 규칙).** `dtype=`를 준 것은 `astype`과 같은 명시적 요청이다 |
| 6 | `astype` float→int의 NaN, 범위 밖 | 유지 | **통일, 방향은 사용자 결정.** (a) 배열 규칙: 포화, NaN → 0 (값 때문에 실패하지 않는다는 배열 원칙과 맞음). (b) 둘 다 ValueError. 추천은 (a) |
| 7 | 자유도 기본값 | 유지 | R2 |
| 8 | 빈 입력 리덕션 | 유지 | **유지.** frame은 NA, 배열은 NaN/ValueError. NA가 있느냐의 차이다 |
| 10 | `unique`에서 NaN끼리 | 유지 | **통일 → NaN끼리 같은 값.** groupby에는 필수이고, numpy 2의 `np.unique` 기본값도 그렇다. Array API(서로 다른 값)에서 벗어나는 곳이라 문서에 적는다 |
| 2, 3, 9, 11 | | 통일 | 그대로 |

### 8.4 R4. 이름과 시그니처 목록
R1이 추천안대로 정해지면, 아래 후보를 함수 하나하나 확정한 목록으로 만들어 M6 전에 보여 준다.
- frame: `sort_values(by, /, *, descending=False, stable=True, na_position=...)`, `value_counts`의 방향 인자, `to_list` → `tolist`, `unique()` → `unique_values()`(정렬 순서까지 배열과 같게), `astype(dtype, /, *, copy=...)`, `concat(objs, /, *, ...)`, `merge(left, right, /, *, ...)`, `read_csv(path, /, *, ...)`, `std/var(*, skipna=True, correction=0)`
- frame에 NaN 판별 `isnan()`을 더한다 (NA 판별 `isna()`와 별개의 개념).
- 이름은 같은데 개념이 다른 것: 배열 `.size`(원소 수) 대 `GroupBy.size()`(그룹별 행 수)
- frame의 공개되어 버린 하위 모듈(`frame`, `index`, `io`, `series`)은 `_`로 숨긴다. 안 그러면 `sd.frame.index`가 Index가 아니라 모듈이 된다.
- 에러 메시지 형식은 `func(): 'arg' ...`(linalg 방식)로 통일하고, 메시지에 문서 절 번호를 넣지 않는다.
- 결과 namedtuple은 한 모듈(`stardust._results`)에 모으고, 이름은 `<함수>Result`로 통일한다.
- 배열 문서와 구현이 다른 곳 두 군데(`zeros` 등의 `dtype` 기본값 표기, `arange`의 위치 인자)를 맞춘다.
- 배열 ↔ linalg: 대각 오프셋 `k`(Array API는 `offset`), `svd(mode=)`(Array API는 `full_matrices`). 라이브러리 안에서는 이미 일관되므로 유지를 추천한다.
- 골든 케이스를 쓰다가 찾은 배열 안의 불일치 (M0, 옛 패키지 그대로 기록함)
  - `full_like(int 배열, 1.5)`는 조용히 1로 자르는데, 암묵적 하향 변환은 에러라는 규칙(S6)과 맞지 않는다. `full_like(int 배열, 1j)`는 에러다.
  - Python int `2**64`를 `divide`, `atan2`만 받고 다른 이항 함수는 모두 OverflowError를 낸다.
  - `arange(nan)`이 에러 없이 빈 배열을 돌려준다.
  - `.T`는 2차원만 받는다. `permute_dims`에 리스트를 주면 "axis must be an int"라는 엉뚱한 메시지가 나온다.
  - `shape`나 `dtype`에 대입하면 "property '' of 'ndarray' object has no setter"처럼 이름이 빈 메시지가 나온다.
  - linalg: `trace(complex, dtype=float64)`는 배열의 캐스팅 규칙 때문에 TypeError다. linalg에서 의도한 동작인지 확인한다.
  - 배열 (M1에서 API 문서를 새로 쓰다 찾음): `real`/`imag`는 Python 스칼라를 거부하는데 `conj`는 받는다. 연산자 피연산자로 `numpy.float64`는 받고 `numpy.int64`는 거부한다. `arange`의 int 인자는 `dtype=float64`여도 int64 범위여야 한다. 옛 API.md는 `arange`의 `stop`, `step`을 키워드로 쓸 수 있게 적었지만 코드는 셋 다 위치 전용이다(새 문서는 코드대로 적었다).
  - 배열 ↔ linalg (M2에서 linalg API 문서를 새로 쓰다 찾음)
    - **결과 dtype**: linalg는 bool/int64 입력을 float64로 계산한다(`vecdot`, `trace`, `outer`, `kron`, `cross`, `matrix_power`). 배열은 int64를 유지한다(`sum`, `matmul`). 그래서 `sd.linalg.trace(int 행렬)`은 float이고 `sd.sum(sd.linalg.diagonal(int 행렬))`은 int다.
    - **결과 레이아웃**: linalg의 행렬 결과는 F-order다(DESIGN §3.3에서 유지하기로 함). 배열 API의 "새 배열은 C-order" 규칙(docs/api/array.md §1.9)과 다르다. 규칙 쪽을 "영역마다 정한 순서"로 넓힐지 정한다.
    - **`.base`**: linalg 결과의 `.base`는 내부 블록 객체다. 배열 규칙(새로 만든 배열의 `.base`는 `None`)과 다르다. M3에서 결과를 `NDArray`로 직접 만들면 자연히 맞는다.
    - **에러 메시지 형식**: `LinAlgError`와 `OverflowError` 메시지에는 `func(): ` 접두사가 없다. `trace(dtype=sd.bool)`은 콜론 없이 "trace() cannot…"이다.
    - **에러 종류**: 잘못된 `ord`의 타입이 `vector_norm`에서는 TypeError, `matrix_norm`/`cond`에서는 ValueError다. int64를 넘는 Python int가 `k`, `n`, `axis`에서는 ValueError인데, 공통 규칙(API.md §2)은 OverflowError다.
    - `axis=[0]`의 메시지가 "must be an int, not list"인데 튜플은 받는다.
    - 결과 namedtuple이 두 모듈(`stardust._results`, `stardust.linalg._results`)에 나뉘어 있다.
  - frame: `df.iloc[0, 9]`의 메시지가 열 위치인데도 "out of range for 4 rows"라고 한다.
  - frame: `df.groupby(Series)`가 Series의 값을 키 이름으로 읽어서 "key columns must be unique"를 낸다.
  - frame: `DataFrame({"a": [1, 2]}, index=[1, 2, 3])`가 index 길이 문제인데 "all columns must have the same length"를 낸다.
  - frame: CSV에서 같은 따옴표 실수가 위치에 따라 다른 메시지를 낸다 ("never closed" 대 "inside an unquoted field").

### 8.5 언제 바꾸나
- 옮기기(M1, M2, M5)는 2라운드와 상관없이 지금 규칙 그대로 한다.
- 2라운드에서 정한 변경은 **M6에서 한 번에** 한다. M6의 이름을 "frame과 배열 잇기, 일관성 맞추기"로 넓힌다. 배열과 linalg 쪽 변경(unique의 NaN, 에러 메시지 형식, 결과 namedtuple)도 M6에 들어간다.
- 골든 비교에서 M6의 변경은 "예상된 차이" 목록으로 먼저 적고, 그 밖의 차이는 없어야 한다.
