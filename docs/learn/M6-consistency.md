# M6. frame과 배열 잇기, 일관성 맞추기

> DESIGN §9의 변경 목록을 여섯 단계로 구현했다. frame의 값 규칙, 시그니처, 이름을 배열 영역(Python Array API 관례)에 맞추고, `Series`/`DataFrame`과 `sd.ndarray`를 잇고(`frame/interop.{hpp,cpp}`), 배열과 linalg의 남은 불일치를 고치고, 세 영역의 에러 메시지를 한 형식으로 모았다. 이 마일스톤은 결과를 일부러 바꾼다. **바뀐 것은 모두 골든 비교의 "예상된 차이"로 미리 적었고, 그 밖의 차이는 세 영역 모두 0개다.**

## 1. 결과
| 항목 | 결과 |
|---|---|
| 테스트 | 2,968개 통과 (M5의 2,813개에서 늘었다. 단계별로 3단계 2,850, 4단계 2,864, 5단계 2,935, 6단계 2,968) |
| 골든 출력: 배열 | 7,746개 중 그 밖의 차이 0. 예상된 차이 1,697개(메시지만 다른 것 포함) |
| 골든 출력: linalg | 1,844개 중 그 밖의 차이 0. 예상된 차이 78개(메시지만 다른 것 포함) |
| 골든 출력: frame | 2,834개 중 그 밖의 차이 0. 예상된 차이 449개(메시지만 다른 것 포함) |
| 성능 | (측정 중) |

- 예상된 차이의 수는 케이스 이름의 수다 (`golden.py`의 마지막 줄). 메시지 형식을 바꾼 6단계 전까지는 배열 112, linalg 1, frame 231개였다. 늘어난 몫은 거의 다 메시지다 (§3).
- 단계별 골든: 2단계 190, 3단계 226, 4단계 231개(모두 frame), 5단계 배열 112, linalg 1, frame 231개. 매 단계 그 밖의 차이는 0이었다.

## 2. 단계별로 바뀐 것
커밋 여섯 개다 (`git log --oneline 4e2bef4..a6476c6`). 일곱째 단계는 문서와 이 해설이다.

### 2.1 dtype 표기 하나로 (7c8a128, DESIGN §9.2)
- `sd.string`이 dtype 객체에 들어왔다. 객체는 다섯이다: `sd.bool`, `sd.int64`, `sd.float64`, `sd.complex128`, `sd.string` (API.md §1).
- **dtype 인자는 어느 영역에서든 객체나 그 이름을 받는다.** `sd.zeros(3, dtype="int64")`가 된다. Python 타입(`int`, `float`, `bool`, `str`)은 어디서나 TypeError다. frame은 전에 Python 타입을 받았다.
  - 영역이 쓰지 않는 dtype은 TypeError다: 배열의 `sd.string`, frame의 `sd.complex128`.
- frame의 `.dtype`은 객체, `DataFrame.dtypes`는 `{컬럼: dtype 객체}` dict다. frame에는 객체를 담는 dtype이 없어서 Series로 돌려줄 수 없다.
- frame 코드는 안에서 이름 문자열을 그대로 쓴다(`_dtype`, Column의 `dtype`). 객체로 바꾸는 것은 공개 속성뿐이다 (`_dtypes.py`).
- 이 단계에서 골든 비교에 "예상된 차이" 장치를 넣었다 (§3).
- 이유: 1라운드 Q5에서 정한 대로다(DESIGN §4.2). `sd.bool`이 Python `bool`과 헷갈리는 문제도 Python 타입을 받지 않으면서 없어진다.

### 2.2 frame의 값 규칙을 배열 규칙으로 (b3116e4, DESIGN §9.1)
| # | 바뀐 것 | 전 (옛 pandas) | 지금 (배열과 같음) |
|---|---|---|---|
| 2 | bool 산술 | TypeError | int64 산술: `True + True == 2`, `-s`, `abs(s)` |
| 3 | int와 float 비교 | 반올림 없는 정확한 비교 | int를 float64로 바꿔 비교: `2**53 + 1 == 2.0**53`이 True |
| 4 | 대입, `fillna` | 값을 보고 판단(int 컬럼에 `2.0`은 되고 `1.5`는 ValueError) | dtype만 보고 위로만 `bool → int64 → float64`. int 컬럼에 float는 값과 상관없이 TypeError |
| 5 | 생성자 `dtype=`, `to_numpy(dtype=)` | 손실 없는 변환만 | 명시적 변환: 추론한 컬럼을 `astype`. int와 float가 섞인 list는 float64로 반올림 |
| 6 | `astype` float → int | NaN, inf, 범위 밖은 ValueError | 0 쪽으로 자르고 포화, NaN → 0 |
| 7 | `std`, `var` | `ddof=1`(0 이상의 int) | `correction=0`(int나 float) |
| 12 | 범위 밖 Python int | 언제나 에러 | float 옆에서는 float, int64가 되어야 하면 배열과 같은 메시지의 OverflowError |

- `saturate_to_int64`(ARM `FCVTZS` 한 명령)가 `array/convert.hpp`에서 `base/saturate.hpp`로 내려갔다. 이제 배열, linalg(`trace(dtype=)`), frame이 같은 함수 하나를 쓴다.
- frame의 정확한 int–float 비교(`compare_int_double`)는 지웠다. 섞인 dtype의 비교는 같은 dtype끼리의 비교와 같은 커널이다.
- **조회 `x in s`만 손실 없는 변환을 남겼다** (`_column.exact`). `x in s`는 "컬럼 dtype에 `x`와 같은 값이 있는가"를 묻는다. 그런 값이 없으면 답은 False다. 그래서 `2**53 + 1 in Series([2.0**53])`은 False이고 `==`는 True다.
- 변환이 세 갈래로 나뉘었다 (`_column.py`): 명시적(`Column.astype`), 암묵적(`implicit`, `implicit_scalar`), 정확한 값(`exact`). 배열 영역의 나눔(array.md §2.4, §2.5)과 같다.
- 골든 케이스는 옛 인자 이름(`ddof`)으로 쓰여 있다. 어댑터가 새 이름으로 바꿔 부른다 (§3.2).
- 이유: DESIGN §8.3의 R3(2라운드 추천안 수락)과 3번, 11번의 사용자 결정(배열 규칙으로). 배열 규칙은 numpy, pandas와 같고 배열 쪽 비용이 없다 (DESIGN §9.8).

### 2.3 frame의 시그니처와 이름 (031dc61, DESIGN §9.3, §9.4)
- 주 인자는 positional-only, 옵션은 keyword-only다: `s.head(n=3)`, `df.sort_values(by="a")`는 TypeError다.
- `ascending` → `descending=False`, 정렬은 `stable=True`도 받는다(frame의 정렬은 언제나 안정이라 `stable=False`여도 결과가 같다). `value_counts`는 `descending=True`가 기본이다.
- `to_list` → `tolist`. `Series.unique()` → `unique_values()`: 처음 나온 순서가 아니라 정렬한 순서다. NaN은 하나, NA는 맨 뒤다. 배열의 `sd.unique_values`와 같다.
- 새것: `isnan()`(NA 행은 NA), int64 비트 연산 `& | ^ ~ << >>`(배열의 규칙), `to_numpy(copy=None|True|False)`(배열 `asarray`의 뜻)
- 공개되어 있던 하위 모듈 여섯(`frame`, `index`, `io`, `series`, `groupby`, `merge`)을 `_frame` 등으로 숨겼다. 이제 `sd.frame.index`라는 속성이 없다.
- 이유: R1(Array API 관례를 라이브러리 전체의 기준으로). 배열에 없는 개념(`skipna`, `na_position`, `by`, `how`, `on`, `keep`)만 frame의 이름이다.

### 2.4 Series/DataFrame ↔ `sd.ndarray` (cf1d6f1, DESIGN §4.3, §9.1 11번)
- `csrc/frame/interop.{hpp,cpp}`가 frame이 배열 영역을 만나는 유일한 곳이다. 하는 일은 넷이다: 컬럼의 읽기 전용 뷰, 배열을 컬럼으로 복사, 컬럼들로 F-order 2차원 배열 만들기, 배열의 float64 합 (docs/design/frame.md §5.4).
- **`Series.to_array`, `DataFrame.to_array`**는 `to_numpy`의 규칙을 따른다. NA와 dtype 변환이 없으면 컬럼의 Buffer를 함께 쓰는 읽기 전용 뷰다. 뷰가 살아 있는 동안 frame 쪽에 쓰면 frame이 먼저 복사한다(Copy-on-Write).
  - Copy-on-Write 코드는 고치지 않았다. 뷰의 `NDArray`가 같은 `shared_ptr<Buffer>`를 쥐므로 `use_count()`가 늘어 있고, `set_rows`가 원래 규칙대로 복사한다.
  - `DataFrame.to_array`는 F-order다. 컬럼마다 `memcpy` 한 번이고, `sd.linalg`가 받는 레이아웃이다.
- **`Series(배열)`, `DataFrame(2차원 배열)`은 언제나 복사한다.** 배열은 다른 쓰기 가능한 뷰로 바뀔 수 있어서 frame의 Copy-on-Write가 볼 수 없다. complex128은 TypeError다.
- **`Series.mean`(int64)은 `sd.sum(x, dtype=sd.float64) / 개수`다.** 컬럼 위의 뷰로 배열의 합을 그대로 부르므로 두 영역이 비트까지 같다 (`test_int_mean_has_the_array_bits`). 전에는 int128로 정확히 더했다.
  - 그룹 평균은 배열을 부르지 않고 그룹마다 float64 pairwise로 더한다. Series의 평균과 비트가 같은 것은 그룹이 4,096행 이하일 때다(배열의 형변환 버퍼가 4,096원소다).
- `tests/interop/`가 새로 생겼다: 공유와 복사, 읽기 전용 뷰, 경계를 넘는 Copy-on-Write, F-order, `lstsq(df.to_array(), ...)`, 두 영역이 같아야 하는 규칙(int 평균, 비교, float → int).

### 2.5 배열과 linalg의 규칙 (139146a, DESIGN §9.1 9번, 10번, 13번, §9.5, §9.6)
- **9번**: 내림차순 `sort`/`argsort`에서 NaN이 맨 앞이다. NaN은 가장 큰 값이다(frame과 같다). NaN 분리(`load_split_nans`)가 내림차순이면 NaN을 앞에 둘 뿐이라 비용이 없다.
- **10번**: `unique_*`가 모든 NaN(NaN 부분이 있는 complex 포함)을 하나로, `-0.0`과 `0.0`을 하나로 본다. 보여 주는 값은 처음 나온 것이다(frame과 같다).
- **13번**: `==`, `!=`도 다른 연산자가 받지 않는 피연산자(리스트, numpy 배열, `None`)에 TypeError다. 전에는 Python이 객체 동일성으로 대신해서 조용히 False였다.
- `dtype=` 없는 `full_like`는 대입처럼 채움 값을 검사한다: `full_like(int 배열, 1.5)`는 TypeError다.
- 범위 밖 Python int는 피연산자가 bool이나 int64로 승격하면 언제나 OverflowError다. `divide`, `atan2`도 그렇다.
- **두 영역의 int 인자(`shape`, `axis`, `k`, `n`)는 int64 범위 밖이면 OverflowError다** (API.md §2). 전에는 ValueError였다.
- `arange`는 nan과 inf를 ValueError로 막는다. `axis`와 속성 대입의 메시지가 바른 이름을 말한다. `real`/`imag`가 Python 스칼라를 받는다. numpy의 bool, 정수, 실수, 복소수 스칼라를 `asarray`가 받는 모든 자리에서 피연산자로 받는다.
- linalg: 잘못된 타입의 `ord`는 세 노름 모두 TypeError다. `vector_norm`은 `axis`를 리스트로도 받는다.
- 정렬과 unique의 시간은 그대로다(커밋 메시지).

### 2.6 에러 메시지 한 형식, 결과 모듈 하나 (a6476c6, DESIGN §9.7)
- **공개 함수와 메서드의 에러는 그 이름으로 시작한다**: `zeros(): ...`, `inv(): singular matrix`, `sort_values(): 'descending' must be ...`. 인자는 작은따옴표 안에 쓴다. 연산자, 인덱싱 등 Python 프로토콜은 Python의 형식이다. 메시지에 문서의 절 번호를 넣지 않는다. 규칙은 API.md §2.1이다.
- 이름을 붙이는 곳은 바인딩이다.
  - 배열과 linalg: 공개 함수를 `bind/errors.hpp`의 `named("zeros", &zeros)`로 감싸 바인딩한다. 예외를 잡아 같은 종류로 메시지만 바꿔 다시 던진다. 그래서 여러 함수가 함께 쓰는 헬퍼는 이름을 몰라도 된다. 이미 그 이름으로 시작하는 메시지는 그대로 둔다.
  - frame: Python이 이름을 안다(`frame/_errors.py`의 `named`). 사용자가 닿는 커널은 `func=` 인자를 받아 바인딩의 `call_named`가 이름을 붙인다.
- linalg의 결과 namedtuple이 `stardust._results`로 옮겨 왔다. 세 영역의 결과 namedtuple이 한 모듈에 있다.
- 골든: 영역마다 메시지만 다른 차이를 받는 `EXPECTED_MESSAGES`를 더했다 (§3.1).
- `tests/test_error_messages.py`가 세 영역의 메시지 형식을 본다.
- 예외 종류와 결과는 이 단계에서 바뀌지 않았다.

## 3. 골든 출력의 장치
### 3.1 `expected.py`: 예상된 차이를 미리 적는다
- `bench/merge/golden/expected.py`의 두 표가 영역마다 `(케이스 이름의 정규식, 이유)`를 든다. 이유에는 DESIGN의 절 번호를 쓴다.
  - **`EXPECTED`**: 맞는 케이스는 어떤 차이(값, 예외 종류, 메시지)든 받는다. 예: `(r"^cmp \w+ (exact|exact scalar|...)$", "§9.1 #3: int vs float compares as float64")`
  - **`EXPECTED_MESSAGES`**: 메시지만 다른 차이만 받는다. 맞는 케이스라도 값이나 예외 종류가 다르면 여전히 MISMATCH다. 그래서 패턴이 넓어도(영역마다 `r"."` 하나) 메시지 말고는 아무것도 숨기지 않는다.
- `golden.py check`는 `EXPECTED`를 먼저 적용하고, 남은 메시지만의 차이에 `EXPECTED_MESSAGES`를 적용한다. 결과는 이유별 케이스 수로 보인다(`-v`면 케이스마다).
- **아무것도 맞추지 못한 패턴은 STALE로 보인다.** 목록이 실제 차이보다 넓어지지 않게 한다. 실패(종료 코드 1)는 MISMATCH나 MISSING이 있을 때뿐이다.
- M5에서 남았던 메시지만의 차이(`set_num_threads(2.0)`, `set_num_threads(n=2)`, M5 해설 §6)도 이유를 붙여 `EXPECTED`에 들어갔다.

### 3.2 케이스 파일의 어댑터
- 골든 케이스는 옛 패키지와 통합판에서 같은 코드로 돈다. M6이 이름을 바꾼 곳은 케이스가 옛 이름으로 쓰고, `cases_frame.py`의 도우미가 통합판에서만 새 이름으로 바꾼다(`_NEW`는 모듈 이름으로 정한다).
  - `_kw`: `ddof` → `correction`, `ascending` → `descending`(값을 뒤집는다, 키마다의 리스트도)
  - `_tl`: `to_list()` → `tolist()`, `_uv`: `unique()` → `unique_values()`
  - 주 인자는 위치로 넘긴다. 옛 패키지는 위치와 키워드를 다 받으므로 두 쪽에서 같은 호출이다.
  - Python 타입으로 dtype을 쓰던 케이스는 이름(`"float64"`)으로 바꿨다. 두 패키지가 모두 받는다.
- dtype 객체(M6)는 옛 패키지의 문자열과 같은 글자로 기록한다(`cases_frame.py`의 인코더). 그래서 `.dtype`이 문자열에서 객체로 바뀐 것만으로는 차이가 나오지 않는다. `DataFrame.dtypes`는 모양(Series → dict)이 바뀌어 예상된 차이로 적었다.

### 3.3 어댑터가 아무것도 바꾸지 않았다는 확인
- 어댑터는 케이스가 계산하는 것을 바꾸면 안 된다. 그래서 케이스 파일을 고친 뒤 **옛 패키지로 다시 기록해서**(`golden.py record`) 저장된 골든(`frame.json`)과 같은지 봤다. 1단계와 3단계에서 확인했고 둘 다 같았다(커밋 메시지). 3단계의 기록은 2단계에서 더한 어댑터(`ddof` → `correction`)까지 포함한다.
- 저장된 골든 파일 세 개(`array.json`, `linalg.json`, `frame.json`)는 M6에서 바뀌지 않았다. 마지막으로 바뀐 것이 M1이다.

## 4. 진행 중에 정한 것
DESIGN §9의 목록에 없거나 목록과 다르게 한 것이다. DESIGN §9 머리말에도 적었다.
1. **배열의 int 인자도 OverflowError** (DESIGN §9 (a)): §9.6은 linalg의 `k`, `n`, `axis`만 적었다. 배열의 `shape`, `axis`, `k`, `num`도 같은 규칙으로 맞췄다. 예외는 `set_num_threads`다: 범위 밖은 크기와 상관없이 ValueError (API.md §3).
2. **그룹 평균은 그룹마다 pairwise** (DESIGN §9 (b)): Series의 int 평균은 배열의 합을 부르고, 그룹 평균은 그룹마다 float64 pairwise로 더한다. 4,096행 이하의 그룹에서 비트가 같다.
3. **`to_numpy(dtype=)`, `to_array(dtype=)`도 명시적 변환** (DESIGN §9 (c)): 생성자의 `dtype=`과 같은 규칙이다.
4. **int와 float가 섞인 list는 반올림** (DESIGN §9 (d)): 2⁵³을 넘는 int도 에러가 아니다. 배열의 `asarray`, pandas와 같다. int만 있는 list를 `dtype="float64"`로 요청하면 처음부터 float64로 읽어서 int64 범위 밖의 int도 받는다.
5. **조회 `x in s`는 손실 없는 변환** (§2.2): 비교 연산자와 다르다.
6. **`correction`은 int나 float 아무것이나**: 옛 frame의 "0 이상의 int"를 배열 규칙으로 넓혔다. `개수 - correction <= 0`이면 NA다.
7. **`stable=False`도 받는다**: frame의 정렬은 언제나 안정이라 결과가 같다. 배열에서도 `stable=False`는 "안정이 아니어도 된다"는 허락일 뿐이다.
8. **배열에서 방금 만든 컬럼은 쓸 수 있는 배열로**: `to_array`가 dtype 변환이나 NA 채우기로 새로 만든 컬럼은 아무도 들고 있지 않으므로 읽기 전용으로 막지 않는다.
9. **메시지의 이름**: 메서드는 클래스 이름 없이 메서드 이름만(`s.sort_values()`와 `df.sort_values()`가 같은 일을 하므로), 생성자는 클래스 이름, 프로퍼티는 `.T: `. 사용자가 부른 함수의 이름이다: `drop_duplicates(keep=...)`의 검사는 `drop_duplicates(): `다 (API.md §2.1).
10. **Column용 방문자는 이번에도 더하지 않았다**: `interop.cpp`는 고정폭 dtype의 바이트를 복사할 뿐이라 `itemsize`로 충분했다 (ARCHITECTURE §4.3).

## 5. 문서
- `docs/api/frame.md`: 머리말의 "dtype은 문자열"을 지우고, 값 규칙이 바뀐 곳을 모두 고쳤다(§1.4, §2.2, §2.3, §2.7, §3.1–§3.6, §5.7, §5.8, §6, §7.5, §10.4, §12). 배열과 다른 곳은 **[배열과 다름]**으로 새로 표시했다.
- `docs/design/frame.md`: `interop.{hpp,cpp}`와 경계를 넘는 Copy-on-Write(§5.4), 변환의 세 갈래(§2.4), `base/saturate.hpp`, float64로 바꾸는 비교(§7.2), int 평균(§7.4, §9.1), 숨긴 하위 모듈과 `_errors.py`(§1.4). "통합 중 M6"으로 표시했던 곳을 실제 설계로 바꿨다.
- `docs/design/array.md`, `docs/design/linalg.md`: "M6에서 정확한 비교로 바뀐다", "M6에서 int128 평균으로 바뀐다"처럼 실제와 반대가 된 줄을 고쳤다. `ord` 파서, `trace(complex, dtype=float64)`, `saturate_to_int64`의 자리, linalg 메시지의 이름 붙이기를 지금 코드대로 적었다.
- API.md §2: OverflowError에 int 인자와 frame의 list 추론을 더했다. ARCHITECTURE: `frame/interop`, `base/saturate.hpp`, `bind/errors.*`, `frame/_errors.py`, `tests/interop/`, `test_error_messages.py`. ROADMAP: M6 완료. README: frame 행.
- DESIGN §8.4의 목록에 결과를 적었다(해결, 그대로, 남음). §7에 `axis` 리스트를 더했다.

## 6. 남은 것
- **`axis`에 리스트 (사용자 결정 대기)**: linalg의 `vector_norm`은 `axis=[0]`을 받고 배열 영역은 TypeError다. 추천은 "배열도 리스트를 받는다"다 (DESIGN §7).
- **코드에 남은 배열 ↔ frame 차이** (DESIGN §8.4 끝, 문서를 고치다 찾음): bool과 int의 비교(배열은 되고 frame은 TypeError), bool의 단항 `+`(배열은 int64, frame은 TypeError), CSV 추론의 정확성 규칙(list는 반올림, CSV는 string으로 남김). 고치면 결과가 바뀌므로 문서에만 적었다.
- **M6에서 다루지 않은 메시지** (DESIGN §8.4): `df.iloc[0, 9]`, `df.groupby(Series)`, `DataFrame({...}, index=...)`의 길이 메시지, CSV 따옴표 메시지 둘. `arange`의 int 인자는 `dtype=float64`여도 int64 범위여야 한다.
- numpy의 int 평균 덧셈 순서를 흉내 낼지 (DESIGN §7, 그대로)
- **새 규칙으로 다시 재지 않은 frame 커널**: int–float 비교, int 평균, float → int `astype` (docs/design/frame.md 머리말). 성능 표(§1)가 채워지면 함께 본다.
- 테스트 파일의 낡은 주석 하나: `tests/frame/test_reduce.py`의 머리말이 "int64 means sum exactly"라고 한다. 테스트 내용은 새 규칙이다(이번 단계는 문서만 고쳐서 두었다).
- **M7**: 옛 참조 대응표(`docs/*/<영역>-refmap.md`)와 `legacy/`를 지우고, 안내서와 README를 마무리한다.

## 7. 리뷰 포인트
1. **바뀐 결과는 모두 예상된 차이로 미리 적은 것이다.** 세 영역 모두 그 밖의 차이는 0개다. 메시지만의 차이는 넓은 패턴으로 받았지만, 그 패턴은 값이나 예외 종류의 차이를 숨기지 못한다 (§3.1).
2. **어댑터는 계산을 바꾸지 않았다.** 옛 패키지로 다시 기록해 저장된 골든과 같음을 확인했다 (§3.3).
3. **frame이 배열을 아는 곳은 `interop.cpp` 하나다.** Copy-on-Write는 고치지 않고 그대로 동작한다. 나갈 때 읽기 전용 공유, 들어올 때 복사다.
4. **int 평균이 두 영역에서 비트까지 같다.** 그룹 평균은 4,096행까지만 같다. 이것은 목록과 다르게 한 곳이다 (§4).
5. **`x in s`와 `==`는 일부러 다르다**: 조회는 손실 없는 변환, 비교는 float64로 바꿔서.
6. **사용자 결정 하나가 남았다**: `axis` 리스트 (DESIGN §7).
7. 성능은 측정 중이다. 결과가 나오면 §1의 표에 넣는다.
