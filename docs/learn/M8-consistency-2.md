# M8. 일관성 2차

> M7 뒤의 감사에서 찾은 비일관과 DESIGN §8.4에 "남음"으로 적어 둔 항목을 고쳤다. 목록은 DESIGN §10이다. 사용자가 제안대로 확정했다(2026-11-12). M6처럼 바뀌는 것을 먼저 함수 단위로 적고, 골든 비교의 "예상된 차이"로 묶었다. 그 밖의 결과는 바뀌지 않았다.

## 1. 결과
| 항목 | 결과 |
|---|---|
| 테스트 | 3,226개 통과 (M7의 2,968개 + 258개) |
| 골든 출력 | 세 영역 모두 예상된 차이 외 0개. 예상된 차이: 배열 1,770, linalg 92, frame 540 (M6의 것 포함, 메시지만 다른 것 포함) |
| 문서 참조 | 1,483개, 깨진 참조 0 (`tools/check_refs.py`) |
| 성능 | 같은 자리 비교(기하평균, 기본/1스레드): 배열 0.997/1.053, linalg 0.952/0.951, frame 0.998/0.974. 걸린 줄의 회귀 둘을 찾아 고쳤다(§6) |
| 컴파일 경고 | 없음 |

커밋: `3848cf2`(목록), `d1628e6`(1단계: frame), `ecd0c2e`(2단계: 배열·linalg, 메시지, 문서), 그다음 커밋(성능 회귀 둘, 해설).

## 2. 어떻게 찾았나
- M7을 마친 뒤 서브에이전트 하나가 세 영역의 같은 개념을 직접 돌려 비교했다(스크립트 27개). 이미 알던 차이는 목록으로 넘겨 다시 보고하지 않게 했다.
- 찾은 것의 대부분은 두 갈래였다.
  - **M6에서 배열만 고치고 frame을 놓친 곳**: `==`에 받을 수 없는 피연산자(§9.1 13번), 옵션 검사
  - **bool을 숫자로 보는 규칙(§9.1 2번)이 산술에만 들어간 것**: list 추론, `concat`, 배열로 내보내기, bool의 `std`/`var`, bool과 int의 비교
- 교훈: 규칙을 "연산 하나"에 넣으면 같은 규칙이 필요한 옆 경로가 남는다. 이번에는 규칙을 한곳에 두고 경로마다 그것을 부르게 했다(§3).

## 3. 1단계: frame의 동작과 이름 (`d1628e6`)
- **승격 규칙 하나**: C++ `promote_columns`(`frame/column.hpp`)가 배열의 `sd::promote`를 쓴다(bool < int64 < float64, string은 string끼리만). list 추론, `concat`, `to_numpy`/`to_array`의 공통 dtype, 비교 커널이 이것을 부른다. `read_csv`의 추론도 float 옆의 큰 int를 list처럼 float64로 읽는다.
  - int만 있는 CSV 컬럼에 int64 밖의 값이 있으면 float64다. list는 이때 OverflowError지만, CSV 추론은 에러 대신 다음 dtype으로 넘어간다.
- **옵션 검사 한곳**: `frame/_errors.py`의 `check_bool`, `check_int`, `check_choice`. bool 기본값을 가진 인자 46개를 모두 검사하고, `tests/frame/test_options.py`가 공개 시그니처를 훑어 빠진 것이 없는지 본다.
- **TypeError로 바뀐 것**: `s == [1, 2]`, `Index == …`(내용 비교는 `.equals()`), `bool(Series)`, `bool(DataFrame)`, `bool(Index)`
- **스칼라 `NA`**: 한 행짜리 Series와 같은 커널로 계산한다. `NA + 1`은 NA, `NA | True`는 True다. `NA == NA`도 NA다. 그래서 `NA in [1, 2]`는 TypeError다(pandas의 `pd.NA`와 같다).
- **배열과 주고받기**: DataFrame과 Index에도 `__array__`가 생겼다. `.iloc`/`.loc`가 `sd.ndarray` 키를 받는다.
- **그 밖의 변경**
  - `RangeIndex`는 int64 범위를 검사하고 첫 인자가 필수다.
  - Series, DataFrame, Index를 pickle할 수 있다. Column이 자기 행만 비트 그대로 저장한다.
  - Index는 해싱할 수 없다.
  - `in`은 Series 규칙을 따르고, bool도 숫자로 본다.
  - `concat`과 컬럼 이름 인자는 list와 tuple만 받는다. 그래서 `groupby(Series)`는 TypeError다.
- **이름**: `value_counts`의 컬럼은 `values`, `counts`다(`normalize=True`의 `proportion`은 그대로). `agg("size")`를 받는다. Index의 repr은 `dtype=int64`다.
- **덤으로 고친 버그**: `agg("std")`가 M6 뒤에도 `correction=1`이었다. `std()`와 같게 0으로 고쳤다.

## 4. 2단계: 배열·linalg, 메시지, 문서 (`ecd0c2e`)
- **numpy에서 가져오기**
  - big-endian은 두 영역 모두 받는다. 배열은 복사하며 바이트를 뒤집는다.
  - uint64는 두 영역 모두 dtype만 보고 TypeError다. 전에는 frame이 값을 보고 판단했다.
  - object 배열은 frame만 받는다. 문자열 dtype이 frame에만 있어서다(docs/api/frame.md §2.4).
- **`arange`**: 결과가 float나 complex이고 int 인자가 int64 밖이면, 인자를 float로 읽는다. int64 안의 int는 지금처럼 정확히 계산한 뒤 바꾼다.
- **`sd.asarray(range)`**: 받는다. generator는 두 영역 모두 받지 않는다.
- **numpy 정수 스칼라 (새로 정한 것, DESIGN §10.1 13번)**: int 인자를 받는 모든 곳이 `np.int64(3)` 같은 numpy 정수도 받는다(`operator.index`, bool은 제외). 전에는 인덱싱, `arange`, `.iloc`만 받고 `zeros`, `eye`, `axis`, `head` 등은 거부했다.
- **메시지**
  - bool 옵션에 bool이 아닌 값, `object`로 받는 옵션에 `None`을 주면 이제 세 영역 모두 `func(): 'x' must be True or False, not int` 꼴이다. 전에는 nanobind의 `incompatible function arguments` 덤프가 나왔다. 바인딩은 이런 인자를 `nb::handle`로 받아 먼저 검사한다.
  - 받은 타입은 짧은 이름이다(`not ndarray`, `not int64`). 모듈 경로를 떼었다.
  - frame의 위치 키 형태가 틀리면 IndexError다(배열과 같게).
  - 배열 원소 하나에 스칼라가 아닌 값을 대입하면 TypeError다(`x[0] = [7]`, `x[0] = sd.asarray([7])`).
  - DESIGN §10.3의 틀린 메시지를 고쳤다. 규칙은 API.md §2, §2.1에 적었다.
  - 남은 것: 없는 키워드나 인자 수가 틀린 호출은 여전히 nanobind의 메시지다. 값이 아니라 호출 모양의 문제라 그대로 두었다(API.md §2.1).
- **문서**: API 문서, 설계 문서, 안내서(02, 03, 04, 11, 12, 13장)의 해당 절을 고쳤다. DESIGN §8.4에 M8에서 해결한 항목을 표시했다.
  - frame의 엄격한 파싱(docs/api/frame.md §3.5)에는 코드가 받는 철자를 그대로 적었다. `nan`, `inf`, `infinity`는 대소문자를 가리지 않고 부호를 붙일 수 있다.

## 5. 골든 출력
- 케이스 파일은 고치지 않았다. 그래서 옛 패키지로 다시 기록할 필요가 없었다.
- 바뀐 결과는 `bench/merge/golden/expected.py`에 이유별로 적었다. 1단계는 frame에 패턴 10개(value_counts 컬럼 55건, bool `std`/`var` 34건 등), 2단계는 배열 1개, frame 2개다.
- 메시지만 바뀐 것은 M6의 `EXPECTED_MESSAGES`가 받는다. 그 이유 문구에 §10.3을 덧붙였다.

## 6. 성능
옛 패키지와 같은 자리에서 번갈아 2라운드 (`bench/merge/m8/final/table.md`):

| 영역 | 기본 스레드 | 1스레드 | 비교 대상 대비 |
|---|---:|---:|---|
| 배열 | 0.997 | 1.053 | |
| linalg | 0.952 | 0.951 | numpy/scipy의 3.21배 (빠름 192, 비슷 47, 느림 9) |
| frame | 0.998 | 0.974 | pandas의 4.64배 |

**걸린 줄을 프로세스를 여러 번 띄워 다시 쟀다.** 둘은 진짜 회귀였고 고쳤다.
- **`solve_triangular`** (300, 300)·(1000, 1000)·(100, 32, 32): 1스레드에서 1.2–1.6배 느렸다. M8의 변경이 아니라 M6 끝의 `lu_solve` 수정(`eac7ac3`)이 원인이었다. 그때 linalg의 임시 버퍼를 모두 128B 정렬 할당으로 바꿨는데, 정렬된 `operator new`는 malloc이 놓아 둔 블록을 다시 쓰지 않아서 큰 버퍼가 호출마다 새 페이지로 왔다. 그 측정은 M6의 같은 자리 비교 뒤에 들어가서 표에 잡히지 않았다. 4KB보다 작은 버퍼만 정렬하게 고쳤다(`linalg/batch.hpp`, docs/design/linalg.md §6.3).
  - 고친 뒤 n=300: 옛 36.6–36.8µs, 통합판 36.6–39.1µs. n=1000: 364–366µs 대 361–364µs. `lu_solve` (1000, 3, 3)는 13.2–14.3µs로 빠른 쪽 그대로다.
- **frame의 아주 작은 리덕션** (`s.sum()` 10행 181 → 227ns, `float sum` 1,000행 314 → 354ns): 1단계에서 들어간 옵션 검사가 Python 함수를 두 번 부르고, C++ 커널에 함수 이름을 키워드 인자와 `std::string` 변환으로 넘긴 비용이었다. 정상 값(`True`/`False`)이면 함수를 부르지 않고 지나가게 하고, 이름은 위치 인자와 Python 객체 그대로 넘기게 고쳤다. 6쌍 평균 옛 173.7ns, 통합판 172ns.
- **배열 1스레드의 원소별 연산** (`f + g`, `ft + 1.0`, `maximum`, 캐스팅): 옛 패키지와 통합판 모두 프로세스마다 두 갈래로 갈린다(예: `ft + 1.0`이 양쪽 다 87–89µs 아니면 115–126µs). M4부터 본 현상이고 회귀가 아니다.

- 1단계: list 추론(`Series` 100만 개), `concat`, `read_csv`의 시간이 바꾸기 전과 같다(약 3% 안).
- 2단계: 옵션을 바인딩에서 먼저 검사하게 되어 작은 호출이 느려지지 않았는지 쟀다. 바꾸기 전과 후의 wheel 두 개를 번갈아 여섯 번 재니 모두 1.00–1.04배였다(`sum(x3)` 92.4 → 92.5ns, `inv` 2×2 286 → 287ns 등). 같은 빌드를 시간 차이를 두고 따로 재면 모든 줄이 4–5% 느려 보였다. 시점에 따른 흔들림이었다. 이런 비교는 번갈아 재야 한다.

## 7. 그대로 둔 것 (DESIGN §10.5)
- NA에서 나온 차이: `int // 0` 등의 NA, 값이 모자란 리덕션의 NA
- frame은 numpy와 배열에서 만들 때 언제나 복사한다(Copy-on-Write).
- linalg는 int 입력을 float64로 계산한다. `GroupBy.size()`, linalg의 `k`, `svd(mode=)`도 그대로다.
- 그룹별 int 평균은 4,096행까지만 `Series.mean`과 비트가 같다.
- object 배열 입력은 frame에만 있다.
- Series에 `ndim`, `size`, `prod`, `any`, `all`이 없는 것은 기능의 범위 문제다.

## 8. 리뷰 포인트
1. **규칙은 한곳에**: 승격 규칙(`promote_columns`)과 옵션 검사(`_errors.py`, `pyutil.cpp`의 `parse_bool`, `index_value`)를 한곳에 두고 경로마다 부른다. M6에서 규칙이 산술에만 들어갔던 일을 되풀이하지 않으려는 것이다.
2. **새로 정한 것 하나**: numpy 정수 스칼라를 int 인자로 받는다(§10.1 13번).
3. **구현하며 정한 것**: `NA == NA`는 NA, `True in Series([1])`은 True, CSV의 int 컬럼에 int64 밖 값이 있으면 float64, 원소 대입에 배열도 TypeError
4. **측정 방법**: 작은 차이를 볼 때는 전과 후를 번갈아 잰다. 그리고 고친 뒤에도 같은 자리 비교를 다시 돌린다: M6 끝의 수정이 M6 표에 잡히지 않았다가 M8에서 드러났다.
