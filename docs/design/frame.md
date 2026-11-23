# 데이터프레임 영역 설계 (sd.frame)

`sd.frame`(데이터프레임 영역)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙과 통합 결정은 [DESIGN.md](../../DESIGN.md), 층 구조와 디렉터리는 [ARCHITECTURE.md](../../ARCHITECTURE.md)에 있다. 공개 API와 동작 규칙은 `docs/api/frame.md`, 영역 공통 규칙(에러 종류, 스레드)은 루트 `API.md`에 있다.
- 이 문서는 "지금 코드가 어떻게 되어 있고 왜 그런가"만 적는다. 코드 주석은 이 문서의 절 번호(`frame.md §N.M`)를 가리킨다.
- 배열 영역과 함께 쓰는 부품(Buffer, 풀, pairwise 합의 정의)은 [array.md](array.md)를 인용한다.
- 수치는 Apple M4 Pro 한 대에서 잰 값이다 (하드웨어는 array.md §1.1). 따로 적지 않으면 1M 행, 1스레드, 호출당 최솟값이다. 비교 대상은 pandas 3.1.1, Polars 2.1.0이다. 옮기기 전의 같은 코드(stardust-pandas)로 잰 값이고, M5는 결과와 속도를 바꾸지 않는다.
  - M6에서 값 규칙이 바뀐 커널(int–float 비교, int64 평균, float → int64 `astype`)은 새 규칙으로 다시 재지 않았다. 옛 규칙으로 잰 수치는 그렇다고 적었고, 새 경로의 시간은 적지 않았다.
- 통합 중 바뀔 곳은 한 줄로만 표시한다 (`통합 중 Mn`). 진행 순서는 [ROADMAP.md](../../ROADMAP.md)에 있다.

---

## 1. 기반

### 1.1 범위
- **있는 것**
  - `Series`, `DataFrame`, `Index`, `RangeIndex`, `NA`
  - 생성: dict, list, numpy 배열, `sd.ndarray`. 꺼내기: `to_numpy`, `to_array`, `tolist`
  - 선택: `[]`, `.loc`, `.iloc` (스칼라, 슬라이스, 리스트, bool 마스크), 라벨 정렬(alignment), `reindex`, 대입
  - 산술, 비교, Kleene 논리, int64 비트 연산, `astype`, `isna`/`notna`, `isnan`, `fillna`, `dropna`
  - 리덕션: `sum`, `mean`, `min`, `max`, `count`, `std`, `var`
  - `unique_values`, `nunique`, `value_counts`, `duplicated`, `drop_duplicates`
  - `sort_values`(다중 키, 키별 방향과 NA 위치), `sort_index`
  - `groupby` + 집계(`sum`, `mean`, `count`, `size`, `min`, `max`, `first`, `last`, `std`, `var`), `agg`
  - `merge`(inner, left, right, outer, 다중 키), 행 방향 `concat`
  - `read_csv` (옵션 8개), `repr`, `head`, `tail`
- **없는 것**
  - MultiIndex. groupby의 키는 결과의 일반 컬럼이 된다.
  - `inplace=`, chained assignment
  - object dtype, `datetime64`, `categorical`, 시간대, Period, Interval, Sparse
  - CSV 밖의 IO, plotting, `resample`, `eval`/`query`, Styler, ExtensionArray
  - `.str`, `.dt`, `rolling`, `pivot_table`, `transform`, `apply` 등 옛 계획의 Tier 2. 코드가 없다.

### 1.2 원칙
- **결과의 dtype과 모양은 데이터가 아니라 입력의 dtype과 모양으로 정한다** (§2.4).
- **같은 결과는 같은 코드에서 나온다.** "무엇이 같은 키인가"는 factorize 한 곳(§8.2), "값의 순서"는 정렬 키 변환 한 곳(§9.3), float 합은 pairwise 정의 하나(§7.4)에 있다. unique, groupby, merge, sort, Index 조회가 모두 그 위에 올라간다.
- **같은 입력은 같은 스레드 수에서 실행마다 비트까지 같다** (요구). frame은 지금 모든 결과가 스레드 수와도 무관하다 (§11.5).
- **커널은 행 루프 안에서 분기하지 않는다.** dtype, 스칼라 여부, NA 유무, 연산 종류는 루프 밖에서(템플릿 인자로) 고른다. 행마다 고르는 구현은 매번 2~10배 느렸다 (§7.2, §7.4, §8.8).

### 1.3 넣지 않은 것
- **결과를 바꾸는 데이터 휴리스틱.** 데이터를 보고 고르는 길은 결과(값의 비트, 행 순서, dtype, 결과 타입, 에러)가 모든 길에서 같을 때만 둔다 (FR18). 성능만 데이터에 따라 달라진다.
  - 둔 것: 이미 정렬된 키(§9.3), 기수 정렬에서 모든 키가 같은 자리 건너뛰기(§9.3), 작은 범위 정수 키의 직접 주소 표(§8.9), 오른쪽 키가 겹치지 않는 merge(§9.7), 연속된 그룹(§9.1), 남는 행 비율에 따른 필터 방식(§7.3).
  - 두지 않은 것: 결과 타입이 데이터에 따라 바뀌는 길. 라벨 슬라이스는 정렬된 Index에서도 이진 탐색으로 없는 라벨을 받아 주지 않는다 (§6.4). 이미 정렬된 RangeIndex Series를 정렬해도 결과 라벨은 take와 같은 Index다 (§2.4).
  - 컬럼의 속성(결측이 있는가, dtype이 무엇인가)과 길이로 고르는 것은 처음부터 이 금지에 들지 않았다: "결측이 없으면 비트맵 검사를 건너뛴다"(§4.4), "8바이트 키와 문자열 키의 테이블 크기"(§8.5).
- pandas의 BlockManager (같은 dtype 컬럼을 2D 블록으로 합치기). 컬럼마다 Column 하나를 둔다 (§5.1).
- lazy 평가와 쿼리 최적화기, JIT, 표현식 융합. pandas처럼 eager로 실행한다.
- 문자열 인터닝, 작은 할당 캐시, 손으로 짠 어셈블리, 런타임 CPU 디스패치

### 1.4 파일과 층
- C++은 `csrc/frame/`, 이름공간 `sd::frame`이다. `frame/`은 `base/`와 `parallel/`만 쓰고 `array/`를 모른다 (루트 ARCHITECTURE §1). 예외는 `frame/interop.cpp` 한 파일이다 (§5.4).

  | 파일 | 내용 |
  |---|---|
  | `frame/column.{hpp,cpp}` | Column, 슬라이스, 손실 없는 int64 ↔ float64 변환(조회 `x in s`용, §2.4), 결과 마무리(`finish_fixed`) |
  | `frame/bitmap.hpp` | validity 비트맵 (§4.4) |
  | `frame/convert.{hpp,cpp}` | Python list, numpy ↔ Column (§3.4) |
  | `frame/interop.{hpp,cpp}` | Column ↔ `sd.ndarray`, 배열의 float64 합. `frame/`에서 `array/`를 아는 유일한 파일 (§5.4) |
  | `frame/parallel.hpp` | frame의 병렬화 정책: 기준값, 청크, `for_chunks`, `Unlocked` (§11) |
  | `frame/ops/elementwise`, `cast`, `selection`, `reduce` | 원소별 연산(int64 비트 연산 포함), astype, take/filter/대입/`isnan`, 리덕션 (§7) |
  | `frame/ops/hash` | 해시 테이블, factorize, Index 조회 엔진 (§8) |
  | `frame/ops/groupby`, `sort`, `radix`, `join` | groupby, 정렬, 기수 정렬과 counting sort, merge (§9) |
  | `frame/ops/pairwise.hpp` | float 합(`base/pairwise.hpp`를 부른다), IEEE min/max (§7.4) |
  | `frame/ops/parse.hpp`, `pow5_table.hpp` | 엄격한 숫자 문법, Eisel-Lemire (§7.5, §10.4) |
  | `frame/io/csv.{hpp,cpp}` | `read_csv` (§10) |
  | `bind/frame_bind.cpp` | 하위 모듈 `stardust._core.frame` (확장 모듈은 `_core` 하나, 루트 DESIGN §3.2) |

- 아래 층에서 쓰는 것: `base/buffer.hpp`(Buffer, §4.1), `base/dtype.hpp`(DType, §4.2), `base/pairwise.hpp`(pairwise 합, §7.4), `base/saturate.hpp`(float → int64, §7.5), `parallel/pool.hpp`(풀, §11.1)
- 바인딩에서 쓰는 것: `bind/errors.hpp`(에러 메시지의 함수 이름, §3.1), `bind/pyutil.hpp`(dtype 객체와 배열 꺼내기, `interop.cpp`가 쓴다)
- Python은 `src/stardust/frame/`: `_series.py`, `_frame.py`, `_index.py`, `_groupby.py`, `_merge.py`, `_io.py`, `_column.py`, `_dtypes.py`, `_errors.py`, `_na.py`, `_ops.py`, `_reduce.py`, `_select.py`, `_sort.py`, `_unique.py`, `_format.py`. 테스트는 `tests/frame/`, 배열과의 경계는 `tests/interop/`.
  - **모든 모듈이 밑줄 이름이다.** 공개 이름은 `__init__.py`의 `__all__`이 전부다. 구현 모듈을 `index`, `series`처럼 공개 이름으로 두면 `sd.frame.index`가 Index가 아니라 모듈이 된다 (루트 DESIGN §9.4).
  - `_column.py`: 사용자 데이터 → Column(`column_from`), 변환의 세 갈래(§2.4), `to_numpy`, `to_array`의 규칙
  - `_dtypes.py`: dtype 인자 읽기(`resolve`)와 dtype 객체(`obj`). Column은 이름 문자열(`"int64"`)을 들고, 공개 `.dtype`만 객체다
  - `_errors.py`: 메시지의 함수 이름(`named`)과 여러 함수가 함께 쓰는 인자 검사 (§3.1)
- 컴파일 플래그는 모듈 전체의 것이다 (array.md §1.3). frame에는 특히 `-ffp-contract=off`(곱셈과 덧셈을 FMA로 합치는 곳은 컴파일러가 아니라 코드가 정한다)와 `-ffast-math` 없음(NaN은 평범한 값이다, §2.2)이 중요하다. frame은 Accelerate를 쓰지 않는다.

---

## 2. 의미론 규칙
규칙의 전체 목록과 pandas와 다른 곳은 `docs/api/frame.md`에 있다. 여기에는 내부 구조를 정한 규칙만 적는다.

### 2.1 dtype
- 넷이다: `bool`, `int64`, `float64`, `string`(UTF-8). C++에서는 `sd::DType`의 `Bool`, `Int64`, `Float64`, `String`이다 (§4.2).
- Python 쪽 dtype은 라이브러리의 dtype 객체(`sd.bool`, `sd.int64`, `sd.float64`, `sd.string`)다. 인자는 객체나 그 이름을 받고, Python 타입(`int` 등)과 `sd.complex128`은 TypeError다 (`_dtypes.resolve`, API.md §1). `.dtype`은 객체, `DataFrame.dtypes`는 `{컬럼: 객체}` dict다.
- 안에서는 이름 문자열을 쓴다. Column의 `dtype`과 Python 쪽의 `_dtype`은 `"int64"` 같은 이름이고, 공개 속성에서만 객체로 바꾼다(`_dtypes.obj`). `_core.frame`의 바인딩도 dtype을 이름으로 주고받는다.

### 2.2 NA와 NaN
- **결측은 `NA` 하나다.** 모든 dtype이 validity 비트맵으로 결측을 갖는다 (§4.4).
- **int 컬럼은 결측이 생겨도 int로 남는다.** pandas 고전 방식(결측이 생기면 float로 바뀜)을 따르지 않는다.
- **NaN은 결측이 아니라 평범한 float 값이다.** `isna`는 NA만 센다. NaN은 `skipna`와 상관없이 리덕션에 전파된다.
- 그래서 NA를 몰래 NaN으로 바꾸는 경로가 없다. `to_numpy`도 NA가 있으면 `na_value=`를 요구한다 (§3.4).

### 2.3 같은 키와 순서
- **같은 키** (unique, groupby, merge, Index 조회): NaN끼리 같다(페이로드, 부호 무관). `-0.0`과 `0.0`이 같다. groupby와 unique에서는 NA끼리 같다. merge에서는 NA 키가 아무것과도 맞지 않는다.
  - 같은 키로 묶인 값은 **먼저 나온 값**을 대표로 보여 준다.
- **순서** (sort, groupby의 그룹 순서): NaN은 +inf보다 큰 값이다. `-0.0`과 `0.0`은 같은 값이다. 문자열은 코드포인트 순서(= UTF-8 바이트 순서), `False < True`.
  - 내림차순은 오름차순의 정확한 거울이다. NaN은 내림차순에서 맨 앞에 온다.
  - NA는 값의 순서에 들어가지 않는다. `na_position`으로 앞이나 뒤에 붙는다. 그룹 순서에서는 맨 뒤다.
  - 정렬은 항상 안정 정렬이다. 같은 값은 원래 순서를 지킨다(내림차순에서도).
- "같은 키"와 "순서"는 맞물린다: factorize에서 같은 키는 정렬에서도 같은 값이다. 둘 다 같은 정규화(`float_key`, §8.2)에서 나온다.

### 2.4 결과의 모양은 입력의 모양으로 정한다
- 결과 dtype은 입력 dtype으로만 정한다. 예: NA 스칼라와의 연산은 반대쪽 dtype의 연산이 된다 (§7.2). 모두 NA인 list와 CSV 열은 float64다.
- 결과 타입(RangeIndex인지 Index인지, 복사인지 공유인지)은 **키의 모양**으로 정한다 (§6.4).
  - RangeIndex를 take한 결과는 남은 라벨이 우연히 등차수열이어도 Index다. 이미 정렬된 Series의 `sort_values`처럼 take를 건너뛰는 길(§9.3)도 RangeIndex를 같은 Index로 바꿔 돌려준다.
  - 마스크가 우연히 연속된 행을 골라도 복사다. take를 건너뛰는 길(정렬된 키의 정렬, 왼쪽 행이 모두 차례로 남는 merge)은 컬럼을 Copy-on-Write로 공유한다. 쓰기는 먼저 복사하므로(§5.2) 복사와 구별되지 않는다.
- 같은 코드가 데이터에 따라 성공하기도 하고 실패하기도 하지 않게 한다. 예: `na_value`가 결과 dtype에 맞는지는 NA가 없어도 검사한다.
- **변환은 세 갈래다** (`_column.py`). 어느 것도 값을 보고 에러를 고르지 않는다. 배열 영역과 같은 나눔이다 (array.md §2.4, §2.5).

  | 갈래 | 쓰는 곳 | 규칙 | 함수 |
  |---|---|---|---|
  | 명시적 | 생성자의 `dtype=`, `astype`, `to_numpy(dtype=)`, `to_array(dtype=)` | 추론한 컬럼을 `astype`으로. 손실이 있어도 바꾼다 (float → int64는 포화, NaN → 0) | `Column.astype` |
  | 암묵적 | 행 대입, `fillna` | dtype만 보고 위로만: `bool → int64 → float64`. 아래로는 값과 상관없이 TypeError (int 컬럼에 `2.0`도) | `implicit`, `implicit_scalar` |
  | 정확한 값 | 조회 `x in s` | 컬럼 dtype에 정확히 같은 값이 있을 때만 (손실 없는 int64 ↔ float64, `Column::convert`). 아니면 그 행이 없는 것이다(False) | `exact` |

  - 조회만 손실 없는 변환을 쓴다. `x in s`는 컬럼 dtype에 `x`와 같은 값이 있는지 묻는 것이라, 그 dtype에 같은 값이 없으면 답은 False다. 그래서 `2**53 + 1 in Series([2.0**53])`은 False이고, `==`는 배열처럼 반올림해 비교하므로 True다 (§7.2).
  - 범위 밖 Python int는 float64가 될 자리에서만 받는다(`implicit_scalar`, 연산자의 `scalar_column`, `from_list`). int64가 되어야 하면 OverflowError다 (배열과 같다, array.md §2.2).

### 2.5 numpy, pandas와 비트가 같은 곳
- numpy, pandas와 비트까지 같은 것은 약속이 아니다 (루트 DESIGN.md). 약속하는 것은 같은 스레드 수에서 실행마다 같은 비트(§11.5), 이 영역 안의 같은 결과(그룹 합 = 그 행만 고른 Series의 합 등)와 정확도다. 스레드 수와 무관한 비트는 더 이상 요구가 아니지만 frame은 지금 모든 결과에서 그렇다. 반올림이 아닌 동작(dtype, 정수 연산, 정렬 순서, NaN과 ±0의 규칙)은 numpy, pandas와 정확히 같게 둔다.
- 아래는 지금 비트까지 같은 곳과 그 방법이다. 고치거나 빠르게 하는 데 필요하면 마지막 자리가 달라질 수 있다. 이유 없이 바꾸지는 않는다.

| 연산 | 맞추는 방법 |
|---|---|
| float 합, mean, var, std | numpy와 같은 pairwise 알고리즘에 잎이 더 넓다(누산기 16개, 블록 512개), `0.0 +` 시작, var의 제곱은 fma (§7.4). 지금은 numpy와 마지막 자리가 다를 수 있다. 배열 영역과는 비트까지 같다 |
| float `//`, `%` | numpy의 `npy_divmod`를 그대로 (§7.2) |
| float min/max | IEEE 754-2019 minimum/maximum = AArch64 `FMIN`/`FMAX` (§7.4) |
| int64 산술 | wrap, 바닥 나눗셈 (numpy, Python과 같다) |
| float → string | Python `str(float)` (§7.5) |
| string → float, CSV의 float | Python `float(text)`와 비트까지 같다 (§10.4) |
| int64 평균 | 정확한 정수 합(`base/int_sum.hpp`)을 한 번 반올림 / 개수. 배열의 `sd.mean`, groupby의 그룹 평균과 비트가 같다. numpy보다 정확해서 numpy와는 마지막 자리가 다를 수 있다 (§7.4) |

- pandas의 groupby 합은 Kahan 보정 합이라서 우리와 비트가 다를 수 있다. 우리 그룹 합은 그 그룹의 행만 고른 `Series.sum()`과 같다 (§9.1).

### 2.6 배열 영역과 다른 곳
- 값 규칙은 배열 영역과 같다: bool 산술은 int64 산술, int와 float는 float64로 바꿔 비교, 변환의 세 갈래(§2.4), float → int64는 포화와 NaN → 0, `std`/`var`는 `correction=0`, 내림차순의 NaN은 맨 앞, `unique`에서 NaN은 하나 (루트 DESIGN §9.1).
- **dtype 승격은 한 곳이다** (루트 DESIGN §10.1 2번, M8): `promote_columns`(`column.hpp`)가 배열의 `sd::promote`(bool < int64 < float64)를 쓰고, string은 string하고만 만난다. list 추론(`convert.cpp`의 `inferred_dtype`)이 C++에서 부르고, Python의 `concat`, `DataFrame.to_numpy`/`to_array`의 공통 dtype은 `_dtypes.common` → `_core.promote_types`로 같은 함수를 부른다. 비교 커널은 bool이 숫자를 만나면 bool을 int64로 바꾼다(산술과 같은 길). bool의 `std`/`var`는 0과 1의 값으로 float 경로를 그대로 탄다.
- **옵션 검사도 한 곳이다** (`_errors.py`): `check_bool`, `check_int`, `check_real`, `check_choice`, `name_list`. 메시지는 API.md §2.1의 형식이다. `check_int`는 배열의 `index_value`와 같은 `operator.index` 규칙이다(numpy 정수는 int, bool과 `numpy.bool_`은 아니다, 루트 DESIGN §10.1 13번).
- **남은 차이는 NA에서 나온다** (루트 DESIGN §9.1 1번, 8번): `int // 0`, `int % 0`, `int ** 음수`는 NA(배열은 0, `a`, 정수 결과), 값이 모자란 리덕션은 NA(배열은 NaN이나 ValueError).
- 그 밖에 코드에 남은 차이(bool과 int의 비교, bool의 단항 `+`, CSV 추론의 정확성 규칙)는 루트 DESIGN §8.4에 적었다.

---

## 3. Python과 C++의 경계

### 3.1 얇은 Python, C++ Column과 커널
- **C++**: Buffer, 비트맵, Column, 해시 테이블과 Index 조회 엔진, 모든 커널(원소별, take, filter, factorize, 리덕션, 정렬, groupby, join, CSV)
- **Python**: `Series`, `DataFrame`, `Index`, `GroupBy` 클래스, 인자 검증, 정렬(alignment) 절차 조립, `repr`
- **에러 메시지의 함수 이름** (API.md §2.1): 공개 메서드의 Python 코드가 이름을 안다. 여러 메서드가 함께 쓰는 헬퍼는 `func`를 받아 `_errors.named(func, ...)`로 메시지를 만들고, 사용자가 닿는 커널(`Column.from_list`, `astype`, `reduce`, `Groups.reduce`, `join_indices` 등)은 `func=` 인자를 받아 바인딩에서 이름을 붙인다(`bind/errors.hpp`의 `call_named`). 연산자와 인덱싱은 `func=None`으로 불러 Python의 형식을 남긴다.
- 배열 영역은 반대로 ndarray까지 C++로 만들었다 (array.md §12.1). frame은 그렇게 하지 않는다.
  - pandas 연산 하나의 고정비는 수 µs다. Python 층이 수백 ns를 더해도 pandas보다 훨씬 빠르다.
  - 데이터가 크면 커널 시간이 전부다.
  - `merge`처럼 옵션이 많은 API를 C++로 파싱하면 코드가 길고 읽기 어렵다.
- 검증: 같은 C++ Column과 커널 위에 세 모양을 만들어 쟀다 (원소 10개).

  | 연산 | 커널만 | C++ Series | **Python Series** | pandas | Polars |
  |---|---:|---:|---:|---:|---:|
  | `s + s` | 152 ns | 92 ns | **256 ns** | 9.16 µs | 716 ns |
  | `s.sum()` | 52 ns | 30 ns | **66 ns** | 3.28 µs | 80 ns |
  | `df["a"]` | 22 ns | 64 ns | **91 ns** | 3.09 µs | 285 ns |
  | `df[df["a"] > 0]` | 695 ns | 400 ns | **1.10 µs** | 39.5 µs | 30.5 µs |

  - Python 층이 더하는 비용은 연산 하나에 30~200ns, 컬럼 3개 필터는 0.7µs다. 원소 100만 개에서는 세 모양의 차이가 측정 오차 안이다.

### 3.2 Column은 값 타입으로 넘긴다
- `_core.frame.Column`은 값 타입으로 바인딩한다. 안에 Buffer의 `shared_ptr` 셋(값, 비트맵, 문자열 offsets)을 든다 (§4.3).
- `std::shared_ptr<Column>` holder로 돌려주면 원소 10개 `add`가 190ns, 값 타입이면 89ns다.
  - 원인(nanobind `stl/shared_ptr.h`): shared_ptr를 돌려받으면 nanobind는 Python 객체마다 `shared_ptr<void>` 복사본을 힙에 새로 만들고 전역 keep-alive 표에 등록한다. 객체가 사라질 때 둘 다 되돌린다.
  - 값 타입은 Python 객체 안에 미리 잡힌 저장 공간으로 move만 한다.
- `Column.share()`는 Buffer를 공유하는 새 Column 객체다. shared_ptr 셋의 복사뿐이다. Copy-on-Write가 이것을 쓴다 (§5.2).

### 3.3 작은 연산의 호출 비용
원소 10개.

| 연산 | sd.frame | pandas | Polars |
|---|---:|---:|---:|
| `s + s` | 454 ns | 9.93 µs | 737 ns |
| `s.sum()` | 173 ns | 3.50 µs | 83 ns |
| `df["a"]` | 175 ns | 3.22 µs | 300 ns |
| `df[df["a"] > 0]` | 1.73 µs | 43.0 µs | 31.2 µs |

- §3.1의 측정보다 늘어난 몫은 NA 전파, 이름 규칙, dtype 검사다.
- 10행 연산은 pandas보다 기하평균 21배, Polars보다 2.1배 빠르다. 1천 행에서도 호출 비용이 시간을 정한다 (§13.3).
- `s.sum()`만 Polars가 빠르다. 어느 단계가 차이를 만드는지는 재지 않았다.

### 3.4 numpy와 주고받기
- **들어올 때는 항상 복사한다** (`from_numpy`).
  - numpy 배열은 밖에서 언제든 바뀔 수 있다. 메모리를 공유하면 Copy-on-Write가 "아무도 쓰지 않는다"고 판단할 근거가 없다.
  - 복사하면서 int32, float16 등을 int64, float64로 넓힌다. stride가 있는 배열도 읽는다. 문자열과 object 배열은 Python 층이 `tolist()`로 바꿔 `from_list`로 보낸다.
  - big-endian 배열은 Python 층이 native 순서로 바꾼 뒤 넘긴다. uint64는 값을 보지 않고 TypeError다. 둘 다 배열의 `asarray`와 같은 규칙이다 (루트 DESIGN §10.1 10번).
  - 1M int64에 109µs다 (pandas 101µs, Polars는 공유해서 1.75µs). 대부분 새 메모리의 페이지 폴트다.
  - 그래서 frame은 `Buffer::adopt`(버퍼 프로토콜로 가져온 메모리)를 쓰지 않는다.
- **나갈 때는 공유한다** (`to_numpy_view`).
  - NA가 없는 bool/int64/float64 컬럼은 Buffer를 그대로 가리키는 **읽기 전용** numpy 배열이 된다 (448ns).
  - 배열은 `shared_ptr<Buffer>` 복사본을 든 capsule을 owner로 갖는다. Series가 먼저 사라져도 배열은 유효하고, 그동안 그 Buffer의 참조 카운트가 하나 늘어 있으므로 frame 쪽 쓰기는 복사한다 (§5.2).
  - 읽기 전용이어야 하는 이유: 쓸 수 있으면 사용자가 배열을 고쳐 Series를 바꿀 수 있다 (pandas 3도 같다).
- **NA가 있으면 `na_value=`가 필요하다** (§2.2).
- **list에서 만들기** (`from_list`)는 C++에서 list를 두 번 훑는다.
  1. 원소마다 종류(NA, bool, int, float, str)를 정하고, 숫자는 바로 읽어 둔다. 문자열은 UTF-8 길이만 더한다(`PyUnicode_AsUTF8AndSize`는 UTF-8 표현을 str 객체에 캐시하므로 2단계에서 다시 불러도 싸다).
  2. 본 종류의 조합으로 dtype을 정하고 채운다. 문자열 바이트 버퍼는 한 번에 할당한다.
  - int64 범위 밖의 int는 1단계에서 double로 읽어 두고 표시만 한다. 2단계에서 float64 컬럼이면 그 double을 쓰고, int64 컬럼이면 OverflowError다 (`int_value`, 배열과 같은 규칙).
  - `dtype=`은 추론한 컬럼에 `astype`을 한 번 더 한다(명시적 변환, §2.4). int만 있는 list를 float64로 요청할 때만 처음부터 float64로 채운다. 범위 밖의 int도 받기 위해서다.
  - bool을 int보다 먼저 확인한다(`True`는 int의 하위 타입이다). `numpy.int32`, `numpy.bool` 같은 numpy 스칼라는 하위 타입이 아니라서 따로 읽는다.
  - 1M `list[int]` 4.77ms (pandas 50.7ms, Polars 4.76ms). pandas는 시간의 대부분을 입력 정리와 dtype 추론(`sanitize_array`)에 쓴다.
- `sd.ndarray`와 주고받는 `to_array`, `Series(arr)`도 같은 이유로 같은 규칙이다: 나갈 때 읽기 전용 공유, 들어올 때 복사 (§5.4, 루트 DESIGN §4.3).

---

## 4. 데이터 모델

### 4.1 Buffer (`base/buffer.hpp`)
모든 영역이 함께 쓰는 참조 카운트 메모리 블록이다 (array.md §3.2).
- **정렬**: 할당 시작을 128B(캐시 라인)로 맞춘다.
- **작은 데이터는 Buffer 안에**: 256B(`kInline`) 이하의 데이터는 Buffer 객체 안에 두고, Buffer는 `make_shared`로 참조 카운트와 함께 할당된다. 작은 컬럼의 값, 비트맵, offsets는 각각 할당 한 번이다.
  - 옛 frame의 Buffer에는 이것이 없었다. 원소 32개 이하의 int64/float64 컬럼, 2048행 이하의 비트맵이 여기에 든다.
- `size()`: 데이터의 바이트 수. Column이 범위를 확인할 때 쓴다.
- `adopt`(외부 메모리)는 frame이 쓰지 않는다 (§3.4).
- Copy-on-Write는 Buffer를 가리키는 `shared_ptr`의 `use_count()`를 본다 (§5.2). 데이터가 Buffer 안에 있든 밖에 있든 같다.

### 4.2 DType (`base/dtype.hpp`)
- 모든 영역의 enum 하나다: `enum class DType : uint8_t { Bool, Int64, Float64, Complex128, String }`.
- frame은 `Bool`, `Int64`, `Float64`, `String`을 쓴다. `Complex128`은 배열 영역에만 있다 (루트 ARCHITECTURE §4.3).
- `itemsize(String)`은 0이다. 문자열의 바이트는 offsets와 UTF-8 버퍼에 있다 (§4.6).
- `visit_dtype`은 배열 dtype만 받는다. frame 커널은 dtype별로 자기 루프를 고른다.

### 4.3 Column (`frame/column.hpp`)
```
Column (값 타입: 정수 넷 + shared_ptr 셋)
├─ dtype, length, offset, null_count
├─ values   ──► Buffer [v0 v1 v2 ...]           bool 1B, int64/float64 8B, string은 UTF-8 바이트
├─ validity ──► Buffer [1 0 1 1 ...] (비트)      null_count == 0이면 없다
└─ offsets  ──► Buffer [0 3 3 7 ...] (int64)     string만. 행 i는 values[o[i], o[i+1])
```
- 1차원이다. NDArray(N차원, 임의의 strides, 결측 없음)와는 다른 물건이라서 그 위에 다시 올리지 않는다 (루트 DESIGN §4.1).
- **슬라이스는 복사하지 않는다** (`slice`). `offset`과 `length`만 바꾸고 Buffer는 공유한다. `offset`은 values의 원소, validity의 비트, offsets의 칸을 똑같이 센다.
  - 그래서 문자열 offsets는 **절대 위치**다. 슬라이스가 offsets를 다시 계산하지 않는다.
  - `head`, `tail`, repr의 행 생략, `.iloc[a:b]`(step 1)가 슬라이스다. 1M 행 frame의 `head()`가 656ns다.
- **null 개수를 함께 든다.** 비트맵은 `null_count > 0`일 때만, 그리고 그때는 반드시 있다.
  - 슬라이스는 자기 범위의 비트를 popcount로 다시 센다(바이트 단위, 양 끝의 덜 찬 바이트만 마스크). 0개면 비트맵을 버린다. 그래서 "결측이 없으면 비트맵이 없다"가 슬라이스에서도 성립한다.
  - 비용은 64행에 popcount 8번 정도다. 1M 행을 슬라이스해도 수 µs다.

### 4.4 validity 비트맵 (`frame/bitmap.hpp`)
- 행마다 1비트, **LSB 우선**(행 i는 바이트 i / 8의 비트 i % 8, Arrow와 같다). 1은 값이 있음, 0은 NA다.
- **결측이 없으면 비트맵이 없다**(null 포인터). 커널은 이것을 보고 NA 검사가 없는 루프를 고른다. 데이터를 보는 휴리스틱이 아니라 컬럼의 속성이다 (§1.3).
- **비트를 0부터 다시 맞추기** (`copy_bits`, `rebased_validity`): 슬라이스는 비트 오프셋을 가지므로, 두 입력의 비트맵을 합칠 때는 비트 0에서 시작하도록 옮긴 뒤 바이트 단위로 AND한다. 출력 바이트 하나는 입력 바이트 둘을 시프트해서 붙인 것이다.
- **바이트로 펼치기** (`kExpand`, `expand`, `valid_bytes`): 비트맵 1바이트를 256칸 표로 읽어 행마다 0/1 바이트 8개를 한 번에 쓴다. 커널은 그 바이트로 분기 없이 고른다(`finish_fixed`: 값에 전부 1이거나 전부 0인 마스크를 AND한다. NEON `and`로 벡터화된다).
  - 비트맵은 메모리를 1/8만 쓰지만, 행마다 비트를 꺼내고 분기하면 느리다. 바이트로 펼치고 루프 경계를 지역 변수로 꺼내자(§7.6) `isna`가 약 8배, `int + int`(NA 10%)와 float `std`가 약 1.6~2배 빨라졌다.
- **8바이트 결과의 원소별 연산은 펼치지 않는다** (`map2`, §7.2): 비트맵 1바이트(8행)마다 NEON 비트 검사(`vtstq_u64`)로 행마다 64비트 마스크를 만들어 결과 둘씩에 AND한다. 바이트 버퍼를 쓰고 다시 읽는 일이 없어 NA가 있는 `f + g`가 NA가 없는 것과 거의 같은 시간이다 (FR13).
- **`count_set`**은 가운데 바이트를 8바이트씩 popcount한다.
- **비트 0에서 시작하는 비트맵 복사**(`rebased_validity`)는 오프셋이 8의 배수면 바이트 복사(`memcpy`)다. 아니면 위의 `copy_bits`다.

### 4.5 NA 자리의 값은 0
- NA 행의 값 자리는 **항상 0이다.** 문자열은 길이 0이다.
- 이유
  - 해시와 비교에 쓰레기 값이 섞이지 않는다. 예: `NA + 5`의 값 자리에 5가 남으면 나중에 키가 된다.
  - 결과가 비트까지 결정적이다.
  - 분기 없는 커널을 짤 수 있다: float 합은 NA 행(0.0)을 그대로 더한다. NA가 10%인 float `sum`이 86µs로 pandas(1.33ms)보다 15배 빠르다. pandas는 마스크로 값을 골라낸다.
  - 마스크 필터는 bool 마스크의 NA 행이 0이라서 False로 읽는다.
- 결과를 만드는 커널은 NA 행에 0을 쓰는 일을 값을 쓰는 루프에서 함께 하거나, 끝에 `finish_fixed`를 부른다. `finish_fixed`는 NA 개수를 세고, 0이면 비트맵을 버리고, NA 행의 값을 분기 없이 지운다(`v &= 0 - valid`, 바이트로 펼친 비트맵으로). 값 위를 한 번 더 지나므로 많이 쓰는 커널은 같은 루프에서 한다: 원소별 연산(§7.2), `take`(§7.1), `fill_na`, CSV(§10.3).
  - int64와 float64는 같은 루프다. float64의 0.0은 모든 비트가 0이다.

### 4.6 dtype별 값 저장
| dtype | 값 Buffer |
|---|---|
| `bool` | 값 하나에 1바이트 (0 또는 1) |
| `int64` | 8바이트 |
| `float64` | 8바이트 |
| `string` | offsets(int64, 길이+1개) + 연속 UTF-8 바이트 (Arrow 레이아웃) |

- **bool을 바이트로**: NEON 비교의 결과(바이트 마스크)를 그대로 저장하고, 마스크 압축 커널이 단순해진다. 대가로 `bool.sum()`은 1M 바이트를 더한다(44µs, 비트로 저장하는 Polars는 0.09µs).
- **offsets는 int64**: int32면 한 컬럼이 2GB로 제한된다 (Arrow `large_string`과 같다).
- **문자열 레이아웃의 득실**
  - 해시, 비교, 필터, 정렬이 Python 객체를 거치지 않고 바이트를 바로 읽는다. 1M 문자열의 `nunique`(키 10만 종)가 pandas 21.5ms 대 15.2ms, 이어 붙이기가 37.4ms 대 4.46ms였다.
  - `tolist()`는 원소마다 UTF-8을 디코딩해 str 객체를 새로 만든다. 1M에 15.7ms, pandas 3의 `str` dtype은 객체를 이미 들고 있어서 5.3ms다. 받아들인 대가다.
  - 짧은 문자열을 인라인으로 두는 레이아웃(Polars의 German string)은 쓰지 않는다.

---

## 5. DataFrame과 Copy-on-Write

### 5.1 DataFrame
- 컬럼 이름 → Column의 순서 있는 매핑 + Index 하나다. 모든 컬럼의 길이가 같다.
- 2D 블록은 없다 (§1.3). 컬럼마다 dtype을 따로 지키고, 컬럼 하나를 꺼내는 것(`df["a"]`)이 Column 객체 하나를 넘기는 일이다.
- 행 하나를 고르면 1행 DataFrame이다 (object dtype이 없으므로 컬럼별 dtype을 지킨다). `df.sum()`도 1행 DataFrame이다 (§7.4).

### 5.2 Copy-on-Write
```python
s = df["a"]
s.iloc[0] = 99      # s만 바뀌고 df는 그대로 (pandas 3과 같다)
```
쓰기는 두 단계다.
1. **쓰는 쪽은 먼저 자기만의 Column 객체를 갖는다**: `self._col = self._col.share()`
   - Python 객체 둘이 같은 Column 객체를 들 수 있다(`df["a"]`). 그 객체를 고치면 둘 다 바뀐다.
   - 이전 객체를 아무도 들고 있지 않으면 이 대입으로 해제되고, Buffer의 참조 카운트가 함께 줄어든다.
2. **`set_rows`가 values와 validity Buffer의 `use_count()`를 본다** (`detach`, `ops/selection.cpp`).
   - 1이면 그 자리에서 쓴다.
   - 다른 주인(다른 Series, DataFrame, 슬라이스, `to_numpy` 뷰의 capsule)이 있으면 자기 행 범위만 새 Buffer로 복사한 뒤 쓴다.
```
df._cols["a"] ──► Column A ──┐
                              ├──► Buffer (use_count 2)
s._col ─────────► Column B ──┘          s.iloc[0] = 99 → 복사
```
- 복사는 필요할 때만 한다. 테스트는 아무도 Buffer를 들고 있지 않을 때 `to_numpy()`의 주소가 쓰기 전후에 같은지 본다.
- 10행 Series의 `s.iloc[3] = 1`이 0.94µs다 (pandas 5.94µs).
- **문자열 컬럼은 항상 새 Buffer를 만든다.** 값의 길이가 바뀔 수 있다.
- `df["c"] = ...`는 컬럼을 통째로 바꾸는 것이라 dtype이 바뀌어도 된다. 기존 컬럼의 행에 쓸 때는 암묵적 변환(dtype만 보고 위로만, §2.4)으로 값을 컬럼 dtype에 맞춘 뒤 `set_rows`가 쓴다 (`_series.assignment_values`).

### 5.3 스냅샷 규칙
- 연산은 **호출 시점의 값**을 본다. 다른 Python 스레드가 연산 중에 대입해도 결과가 섞이지 않는다.
- 커널이 GIL을 놓으므로(§11.4) 대입과 연산이 겹칠 수 있다. 그래도 규칙이 지켜지는 이유:
  - 진행 중인 연산은 nanobind 인자로 옛 Column 객체를 잡고 있다. 대입은 `share()`로 새 Column 객체를 만든 뒤 쓴다. 그래서 Buffer의 참조 카운트가 2 이상이고, 대입은 복사한다 (§5.2).
  - 남은 틈은 하나다: `set_rows`가 Buffer를 그 자리에서 고치는 동안 다른 스레드의 연산이 시작되는 경우. 그래서 **`set_rows`는 크기와 상관없이 GIL을 쥔 채 돈다.**
- **groupby는 부른 순간의 컬럼 매핑을 복사해 둔다** (`_groupby.py`). 키 코드는 처음 쓸 때 계산해서 캐시하는데, 값 컬럼을 매번 df에서 읽으면 그 사이 `df["k"] = ...`가 옛 키와 새 값을 섞는다. 데이터는 복사하지 않는다(행 대입은 CoW로 복사되므로 보이지 않는다).
- 테스트: 한 스레드가 `s.iloc[k] = 1`을 k = 0, 1, 2, …로 쓰는 동안 다른 스레드가 `s + 0`을 반복한다. 모든 결과가 "앞의 k개만 1"인 꼴이어야 한다.

### 5.4 배열과의 경계
- **frame이 배열 영역을 만나는 곳은 `frame/interop.{hpp,cpp}` 하나다** (루트 DESIGN §4.3, ARCHITECTURE §1). `array/`의 헤더(`ndarray.hpp`)와 `bind/pyutil.hpp`를 include하는 frame 파일은 이것뿐이다. 하는 일은 셋이다. 두 영역이 같은 비트를 내야 하는 합(pairwise 트리, 정확한 정수 합)은 `base/`에 있어서 두 영역이 각자 부른다.

  | 함수 | 하는 일 | 쓰는 곳 |
  |---|---|---|
  | `to_array_view(col, writeable)` | NA 없는 bool/int64/float64 컬럼의 값 위에 1차원 `NDArray`를 만든다. `buffer`는 Column의 values Buffer를 그대로 가리키고, `data`는 `offset`만큼 민 자리다. 복사하지 않는다 | `Series.to_array`, `Column.to_array_view` |
  | `from_array(arr)` | 1차원 배열을 새 Column으로 복사한다. 연속이면 `memcpy` 한 번, 아니면 원소마다. complex128은 TypeError | `Series(arr)`, `DataFrame(arr2d)`(열마다 `arr[:, j]`) |
  | `columns_to_array(cols, n)` | 같은 dtype, NA 없는 k개 컬럼으로 (n, k) F-order 배열을 만든다. strides는 `(itemsize, n·itemsize)`, 컬럼마다 `memcpy` 한 번 | `DataFrame.to_array` |

- **나갈 때는 공유, 읽기 전용이다.** 뷰의 `NDArray::buffer`는 Column과 같은 `shared_ptr<Buffer>`이므로, 뷰가 살아 있는 동안 그 Buffer의 `use_count()`가 하나 늘어 있다. frame 쪽 대입은 §5.2의 규칙 그대로 이것을 보고 먼저 복사한다. Copy-on-Write를 고칠 필요가 없었다. 뷰의 값은 바뀌지 않는다.
  - 읽기 전용이어야 하는 이유는 numpy 뷰와 같다(§3.4): 쓸 수 있으면 배열을 통해 Series가 바뀐다.
  - 예외: 이 호출이 방금 만든 Column(dtype 변환이나 NA 채우기의 결과)은 아무도 들고 있지 않으므로 쓸 수 있는 배열로 내준다 (`writeable`, `_column.array_column`의 `fresh`). `copy=True`는 배열 쪽 `asarray(copy=True)`로 복사한다.
- **들어올 때는 언제나 복사한다.** 배열은 다른 쓰기 가능한 뷰를 통해 바뀔 수 있어서, frame의 Copy-on-Write가 그 쓰기를 볼 수 없다. numpy에서 들어올 때(§3.4)와 같은 이유다.
- **F-order**: Column은 컬럼마다 따로 연속이다. (n, k) 배열을 F-order로 만들면 컬럼 하나가 연속된 한 덩어리라서 복사가 컬럼마다 `memcpy` 한 번이다. `sd.linalg`의 패킹은 F-order 입력을 가장 빠른 경로(열마다 복사와 유한성 검사를 한 패스로)로 읽는다 (linalg.md §5.3).
- 테스트: `tests/interop/test_frame_array.py`가 공유와 복사, 읽기 전용 뷰, 경계를 넘는 Copy-on-Write, F-order strides, `sd.linalg.lstsq(df.to_array(), ...)`, int 평균의 비트, 비교와 float → int 변환이 두 영역에서 같은지를 본다.

---

## 6. Index와 정렬(alignment)

### 6.1 RangeIndex와 Index (`_index.py`)
- **`RangeIndex`**: `(start, stop, step)`만 든다. 버퍼가 없다.
  - 스칼라 조회와 `in`은 산술이다. 라벨을 한꺼번에 볼 때(정렬, 여러 라벨 조회)만 라벨 배열을 만든다. `5 in RangeIndex(3e7)`이 라벨 3천만 개와 테이블을 만들면 158ms가 든다.
- **`Index`**: 불변 Column 하나(int64 또는 string, NA 없음)와 이름을 든다. float와 bool은 Index가 될 수 없다 (`docs/api/frame.md`).
- 기본 RangeIndex(0부터 1씩)에서 take한 라벨은 위치 배열 그 자체다. 복사하지 않고 위치 배열을 라벨로 쓴다. 결과는 Index다 (§2.4).

### 6.2 조회 엔진 (`LabelTable`, `ops/hash.hpp`)
```
labels  ["a", "b", "a", "c"]
codes   [0,   1,   0,   2  ]          라벨 → 코드 (factorize, §8)
first   [0, 1, 3]                     코드 → 처음 위치   (스칼라 조회, reindex, 정렬)
CSR     start [0, 2, 3, 4]            코드 → 모든 위치   (중복 라벨의 리스트 조회)
        rows  [0, 2, 1, 3]
```
- **처음 조회할 때 만들어서 Index에 캐시한다.** Index는 불변이라 캐시를 무효화할 일이 없다. `is_unique`도 엔진을 만들며 얻는다.
  - 라벨이 기준값(2¹⁷, §11.2) 이상이면 `Partitioned::build`가 GIL을 놓는다(§11.4). 그래서 두 Python 스레드가 같은 Index를 처음 조회하면 둘 다 엔진을 만들 수 있다. 결과가 같은 엔진이고, 나중에 끝난 쪽이 캐시에 남는다. 만든 뒤의 조회는 읽기만 한다.
- CSR은 모든 위치를 묻는 조회(`all_positions`)가 처음 올 때 만든다. 라벨 리스트뿐 아니라 라벨 하나(`.loc["a"]`)와 라벨 슬라이스의 양 끝도 이 조회를 쓴다(라벨이 고유한지 봐야 한다). `LabelTable` 안의 직렬 counting sort이고(§9.6의 `counting_sort`를 부르지 않는다), GIL을 쥔 채 만든다.
- 1M string 라벨에서 처음 만들기 37ms(pandas 33ms), 그 뒤 스칼라 조회 1.56µs(pandas 2.06µs), 라벨 1,000개 조회 123µs(pandas 132µs)다. 만들기 시간은 같은 라벨의 factorize 시간과 같다.
- 라벨의 타입은 Index와 맞아야 한다. int Index에 `10.0`이나 `True`는 TypeError다. 같은 키의 규칙이 dtype마다 다르기 때문이다 (§2.3).

### 6.3 정렬(alignment) (`_select.py`의 `align`)
```python
if a is b or a.equals(b): 위치대로              # 먼저 객체 동일성, 아니면 원소 비교
else:                                           # 양쪽 라벨이 고유해야 한다
    union = a의 라벨 + (b에만 있는 라벨, b의 순서)
    a_pos = [0, 1, ..., len(a)-1, -1, -1, ...]
    b_pos = b.get_indexer(union)               # 없는 라벨은 -1
```
- 각 컬럼은 `take(positions)`로 다시 배열한다. **위치 -1은 NA다** (§7.1).
- "같은 Index면 위치대로"는 의미 규칙이다. 휴리스틱이 아니다.
- `Series ⊕ Series`, `DataFrame ⊕ DataFrame`, Series dict로 DataFrame 만들기, `df["c"] = Series`, `reindex`, 마스크가 이 함수 하나를 쓴다.
- 마스크의 Index가 다르면 Index가 고유해야 한다. 길이가 같고 중복이 있으면 라벨 집합이 다를 수 있기 때문이다.
- **합집합을 정렬하지 않는다.** pandas는 두 Index가 다르면 합집합을 정렬한다. 1M string 라벨을 섞은 순서로 더하면 pandas 691ms, 우리는 해시 조회 두 번으로 99ms다.

### 6.4 선택 (`_select.py`)
`.loc`/`.iloc`/`[]`의 키는 세 모양으로 바뀐다. 결과 타입은 키의 모양으로 정한다 (§2.4).

| 모양 | 예 | 결과 |
|---|---|---|
| 하나 | `s.loc["a"]`, `s.iloc[-1]` | Series는 스칼라, DataFrame은 1행 DataFrame |
| 슬라이스 | `s.loc["b":"d"]`, `s.iloc[::2]` | step이 1이면 Buffer 공유(§4.3). RangeIndex는 RangeIndex로 남는다 |
| take | 라벨 리스트, 마스크 | 복사. 결과 라벨은 Index |

- 라벨 슬라이스는 양 끝 라벨의 위치를 찾아 위치 슬라이스로 바꾼다. 양 끝 라벨은 있어야 하고 고유해야 한다. 정렬된 Index에서도 이진 탐색으로 없는 라벨을 받아 주지 않는다 (§1.3).
- 마스크로 고르는 행은 컬럼마다 바로 압축한다 (§7.3).
- **위치 리스트**(`.iloc[list]`, 정수 numpy 배열, int64 `sd.ndarray`)는 한 번에 int64 Column으로 바꾸고, 범위 검사와 음수 보정을 커널 하나(`wrap_positions`)가 한다. 범위 밖 위치는 `parallel::FirstRow`로 첫 것을 찾아 메시지에 쓴다. 원소가 모두 int가 아닌 키(bool, float, 섞인 list)만 원소마다 검사해 메시지를 만든다. 1M 위치에서 원소마다 Python으로 검사하면 100~170ms, 커널로 몇 ms다.
- RangeIndex의 라벨 하나는 `operator.index`로 Python int로 바꿔 range의 산술로 찾는다. `range`는 정확히 `int`일 때만 산술로 답하고, numpy 정수면 처음부터 훑는다(3천만 행에 1.5초).
- 스칼라로 컬럼을 채울 때(`df["c"] = 0`, dict의 스칼라 값, 모두 NA인 컬럼)는 1행 Column으로 dtype을 정하고 커널(`full`)이 n행으로 넓힌다. 길이 n의 Python list를 만들지 않는다.
- `.iloc[i]` 435ns(pandas 1.42µs), `.iloc[a:b]` 837ns(pandas 4.76µs)

---

## 7. 원소별 커널과 리덕션

### 7.1 공통 기본 연산
- **`take(positions)`**: gather. 위치 -1은 NA 행이다. reindex, 정렬(alignment), merge, sort, groupby의 `min`/`max`/`first`/`last`가 이 위에 올라간다. 그래서 그 연산들은 dtype마다 따로 코드를 쓰지 않는다.
  - 고정폭 dtype은 한 번 지나며 끝낸다 (`take_fixed`): 행마다 범위 검사를 누적하고(`(p + 1) > length`를 OR), 값을 읽고(-1이면 0행), "위치가 있고 원래 행에 값이 있다"를 비트로 64행씩 모아 비트맵 워드로 쓰고, NA 행에는 0을 쓴다. 분기가 없다. 범위 밖 위치가 있으면 출력을 버리고 IndexError다. 범위 검사, gather, `finish_fixed`의 세 패스였을 때보다 약 2배 빠르다 (FR13).
  - 문자열은 길이를 세는 패스와 복사하는 패스 둘이다.
- **`fill_na(value)`**: 청크의 비트맵을 바이트로 펼쳐 `valid ? x : value`를 고른다. 행마다 `is_valid`와 `memcpy`를 부르던 때보다 약 4배 빠르다.
- **`filter(mask)`**: 마스크로 압축 (§7.3). **`nonzero(mask)`**: True인 위치(NA는 False).
- **factorize**: 값 → 처음 나타난 순서의 정수 코드 (§8)
- **리덕션**: 결과는 1행 Column (§7.4)
- **`set_rows`**: Copy-on-Write 대입 (§5.2). **`concat`**: 같은 dtype 컬럼 이어 붙이기 (§9.8)

### 7.2 원소별 연산 (`ops/elementwise.cpp`)
```
s + 1  →  Series.__add__  →  1행 Column(1)
                          →  _core.frame.binary_op("add", a, b, scalar_side=2)
                                 dtype 조합으로 분기 → int / float / 비교 / 논리 / 이어 붙이기
```
- **진입점은 `binary_op` 하나다. 스칼라는 1행 Column이다.**
  - 행 i에서 스칼라를 읽을지 컬럼을 읽을지는 템플릿 인자(`In<T, Scalar>`)로 정한다. 행 루프에 분기가 없고, 컴파일러가 세 경우(컬럼⊕컬럼, 컬럼⊕스칼라, 스칼라⊕컬럼)를 따로 벡터화한다.
  - 반대쪽 연산(`1 - s`)은 피연산자 순서만 바꿔 같은 커널을 부른다.
  - **NA 스칼라는 반대쪽 dtype의 1행 NA Column이다.** `s_int + NA`는 int64, `s_int / NA`는 float64다. NA를 위한 dtype 규칙이 따로 없다.
  - `__array_ufunc__ = None`: `np.array + s`에서 numpy가 Series를 배열로 바꿔 버리지 않게 한다.
  - 그 대가로 numpy가 받지 않는 스칼라(complex, `datetime64` …)를 오른쪽에 두면 Python이 numpy 스칼라의 반대쪽 연산을 부르고, numpy는 ufunc의 문장(`operand 'Series' does not support ufuncs`)을 낸다. 그래서 정방향 연산자가 그런 numpy 스칼라를 만나면 NotImplemented 대신 Python의 문장(`unsupported operand type(s) for +: 'Series' and 'numpy.complex128'`, 순서 비교는 `'<' not supported between instances of ...`)을 직접 낸다(`_ops.python_error`; 2026-11-22). 반대쪽(`np.complex128(1) + s`)은 Python이 그 문장을 낸다. `1j < s`처럼 왼쪽이 numpy 스칼라인 순서 비교는 Python이 `s > 1j`로 돌려 부르므로 기호가 `>`로 보인다.
- **연산 종류도 템플릿 인자다.** 행마다 `switch`로 고르던 int–float 비교가 1.90ms, 템플릿 인자로 올리자 0.71ms였다(옛 규칙의 정확한 비교에서 잰 값). Kleene `&`는 5.09ms → 1.07ms였다(분기 없는 식, 연산 종류를 템플릿 인자로).
- **결과 validity**
  1. 한쪽 컬럼에만 NA가 있고 그 비트맵이 행 0에서 시작하면, 결과는 그 Buffer를 공유하고 NA 개수도 그대로 쓴다. 결과의 NA 행은 정확히 그 입력의 NA 행이다. 공유한 비트맵은 아무도 그 자리에서 고치지 않는다(쓰기 전에 복사, §5.2).
  2. 그 밖에는 두 입력의 비트맵을 비트 0으로 맞춘 뒤 AND한다 (`and_validity`). NA 스칼라면 모든 행이 NA다.
  3. NA 행의 값은 값을 계산하는 루프에서 0이 된다: 8바이트 결과는 비트맵 바이트마다 NEON 마스크로(§4.4), 1바이트 결과는 펼친 바이트로 고른다. `finish_fixed`의 두 번째 패스가 없다.
  4. 커널이 직접 NA를 만드는 연산(`int // 0`, `int % 0`, `int ** 음수`)은 비트맵을 자기 것으로 만들어 지운 뒤 `finish_fixed`를 부른다.
- **정수**
  - 오버플로는 wrap이다. 부호 있는 오버플로는 C++에서 정의되지 않은 동작이라 `uint64`로 계산한 뒤 되돌린다. 하드웨어 명령은 같고, 컴파일러가 "오버플로는 없다"고 가정해 코드를 바꾸는 것만 막는다.
  - `INT64_MIN // -1`: x86 `idiv`는 프로세스를 죽이고, ARM `sdiv`는 wrap하고, C++ 표준은 정의하지 않는다. 그래서 나누는 수가 -1이면 따로 계산한다(결과 `INT64_MIN`).
  - 바닥 나눗셈: C++의 `/`는 0 쪽으로 자른다. 나머지가 있고 부호가 다르면 몫에서 1을 뺀다. `%`는 나누는 수의 부호를 따른다.
  - 거듭제곱은 제곱을 반복하고 wrap한다 (numpy와 같다).
- **float `//`, `%`는 numpy의 `npy_divmod`를 그대로 옮겼다.** 정의대로 `floor(a / b)`를 계산하면 `a / b`가 먼저 반올림되어 다르다(`1.0 // 0.1`이 numpy는 9, 정의대로면 10). `fmod`로 정확한 나머지를 먼저 구하고 `(a - mod) / b`를 정수로 맞춘다. 나머지가 0이면 `copysign(0, b)`다.
- **int와 float는 int를 float64로 바꿔 비교한다** (`compare_mixed` → `compare_plain<int64_t, double>`). 배열 영역, numpy, pandas와 같은 규칙이다 (루트 DESIGN §9.1 3번). C++의 산술 변환이 int64를 가장 가까운 double로 바꾸고, 같은 dtype끼리의 비교와 같은 커널이다. NaN과의 비교는 IEEE 754대로 False(`!=`만 True)다.
  - 반올림 없는 정확한 비교는 하지 않는다. 두 규칙의 비용은 루트 DESIGN §9.8에 있다. 이 커널의 시간은 다시 재지 않았다.
- **bool의 산술은 int64로 바꾼 뒤 int 커널로 간다** (`binary_op`, `unary_op`의 `neg`, `abs`). 배열 영역과 같다. bool 컬럼은 `astype(Int64)`로 한 번 복사된다. 같은 길로 `// 0`의 NA 같은 int 규칙이 그대로 적용된다.
- **int64 비트 연산** (`int_bitwise`, 단항 `~`는 `unary_op`의 `invert`): `&`, `|`, `^`는 int64끼리(bool은 int64로 올림), `<<`, `>>`는 int64끼리만이다. 시프트 수는 부호 없는 수로 읽어 64 이상(음수 포함)이면 모든 비트가 밀려 나간다(C++에서는 정의되지 않은 동작이라 따로 검사한다). 배열의 `bitwise_*`와 같은 규칙이다. bool끼리의 `&`, `|`, `^`는 아래의 Kleene 논리다.
- **Kleene 논리** (`&`, `|`): 한쪽이 확실한 False(`&`)나 True(`|`)면 결과가 정해진다. validity를 바이트로 펼치고, 식을 그대로 쓰고, 출력 비트를 8개씩 모아 쓴다.

  | 연산 | 값 | NA가 아닌 조건 |
  |---|---|---|
  | `&` | `a & b` | `(va & vb) \| (va & !a) \| (vb & !b)` |
  | `\|` | `a \| b` | `(va & vb) \| (va & a) \| (vb & b)` |
  | `^` | `a ^ b` | `va & vb` |

- **문자열**
  - 비교는 `string_view::compare`다. `char_traits<char>`가 바이트를 unsigned로 비교하므로 UTF-8 바이트 순서 = 코드포인트 순서다. 디코딩하지 않아도 Python str과 같은 순서다.
  - 이어 붙이기는 두 번 훑는다(청크마다 바이트 수 → 청크마다 자기 시작 위치부터 쓰기). 1M에 4.46ms, pandas 37.4ms다.

### 7.3 마스크 필터 (`ops/selection.cpp`)
- **고정폭 dtype(bool, int64, float64)을 거르는 방법은 압축과 모으기 둘이다.** 컬럼 하나의 `filter`(`Series[mask]`, `dropna`, NA가 있는 컬럼)는 언제나 위치 배열 없이 분기 없이 압축한다. `df[mask]`(`filter_columns`)의 NA 없는 고정폭 컬럼만 둘 중 하나를 고른다: 남는 행이 마스크의 2/3 이하이면 위치(`nonzero`)로 모으고(gather), 그보다 많으면 압축한다. 둘 다 같은 행을 같은 순서로 내므로 결과는 같다 (FR14).
  ```
  압축:  k = 0
         for i: out[k] = in[i]; k += keep[i]     ← 항상 쓰고, 남길 때만 k를 늘린다
  모으기: for j: out[j] = in[pos[j]]             ← 남는 행만 읽고 쓴다
  ```
  - 압축은 모든 행을 지나고, 모으기는 남는 행만 지난다. 절반이 남는 100열 프레임에서 모으기가 1스레드 약 2배 빠르다.
  - 압축이 끝까지 돌면 마지막으로 남길 행 다음 칸에도 쓴다. 그 칸은 다음 청크의 첫 칸이거나(병렬에서 경쟁) 출력의 끝 너머다. 그래서 청크는 자기의 마지막으로 남길 행까지만 돈다.
  - 병렬: 압축은 청크마다 남길 행 수를 세고 청크 순서의 누적합이 출력 시작 위치다. 모으기는 출력 위치를 청크로 나눈다.
- **문자열과 NA가 있는 컬럼은 따로 한다.** 문자열은 `take(nonzero(mask))`다. 문자열을 같은 방식으로 압축하는 시도 둘이 `take`보다 느렸다(길이가 제각각이라 "항상 쓰기"가 비싸다). NA가 있는 고정폭 컬럼은 값과 validity 바이트를 압축한 뒤 비트로 모은다.
- **`df[mask]`는 컬럼을 한꺼번에 거른다** (`filter_columns`). 마스크의 청크별 개수와 `nonzero`는 한 번만 만들고(위치는 Index의 라벨과 문자열 컬럼이 함께 쓴다), NA 없는 고정폭 컬럼은 (컬럼, 청크) 쌍이 태스크다. 컬럼 하나가 기준값 아래인 넓은 프레임(10만 행 × 100열)도 나뉜다. 문자열과 NA가 있는 컬럼은 하나씩 부르고, 각 커널이 스스로 나눈다(한 태스크로 묶으면 그 컬럼이 한 스레드에서 돈다).
- `Series[mask]`, `dropna`는 컬럼 하나의 `filter`다.

### 7.4 리덕션 (`ops/reduce.cpp`, `ops/pairwise.hpp`)
- **결과는 1행 Column이다** (NA면 1행 NA). Series는 스칼라로 꺼내고, DataFrame은 컬럼마다 하나씩 모아 1행 DataFrame을 만든다. 행 라벨은 연산 이름이다.
- **DataFrame의 리덕션은 한 번에 부른다** (`reduce_columns`). 컬럼이 모두 기준값 아래이고 합쳐서는 기준값 이상이면 컬럼이 태스크다(10만 행 × 100열의 `df.sum()`). 결과는 컬럼마다 `reduce`를 부른 것과 같다. 에러는 태스크 전에 컬럼 순서로 검사해 첫 컬럼의 것이 나온다 (FR14).
- 값이 모자라면(`n - correction ≤ 0`, 모두 NA인 `mean`) NA다. NaN은 실제 계산에서 나올 때만 쓴다. `correction`은 배열처럼 int나 float이고 double로 받는다 (`_reduce._check_correction`).
- **float 합은 pairwise 합이다.** 정의는 모든 영역이 함께 쓰는 `base/pairwise.hpp` 한 벌이다(트리 모양, 잎의 합, 병렬 계획과 합치기; array.md §6.2). frame은 접근자로 부른다. 그래서 frame의 합은 배열의 합과 비트까지 같다.
  ```
  n < 16     : 0.0에서 시작해 차례로
  n ≤ 512    : 누산기 16개에 16개씩 번갈아 → 이웃끼리, 그 다음 짝끼리 접기, 나머지를 차례로
  그 밖      : 절반(16의 배수로 맞춤)씩 재귀
  sum = 0.0 + pairwise(전체)          ← reduce는 항등원 0.0에서 시작한다
  ```
  - numpy와 같은 알고리즘에 잎이 넓다(numpy는 누산기 8개, 블록 128개). 그래서 numpy와 마지막 자리가 다를 수 있다. 사슬이 여럿 동시에 진행되어 빠르고, 반올림 오차가 log n에 비례한다. 1M 합의 시간 86~98µs(Polars 85µs, pandas 202µs)는 누산기 8개의 잎으로 잰 값이다.
  - **`0.0 +`**: `np.sum([-0.0] * 9)`는 `+0.0`이다. 이 시작 때문이다.
  - NA 행은 0.0이라 분기 없이 함께 더한다 (§4.5). 0.0을 더해도 합은 변하지 않는다.
  - float의 `mean`, `var`, `std`는 pandas가 계산하는 식(평균 → `(x - 평균)²` → pairwise 합 → `/(n - correction)`)을 따른다. 제곱은 배열의 `var`처럼 잎에서 fma 하나로 더한다(`pairwise::leaf_squares`, `squares_sum`). 그래서 NA가 없으면 배열의 `var`, `std`와 비트까지 같다. pandas, numpy와는 반올림 오차 안에서 같다.
  - **병렬**: 2¹⁴개(`kPairwiseLeaf`) 이하의 부분 트리가 태스크이고(`pairwise::plan`), 위쪽 노드는 같은 트리 모양으로 더한다(`pairwise::combine`). 모든 노드가 직렬 재귀의 노드라서 스레드 수와 무관하게 비트가 같다. 잎 크기는 배열(2¹²)과 다르지만 같은 트리 위의 서로 다른 자르기라서 비트는 같다. 나누는 기준은 frame의 기준값이다 (§11.2).
- **int64 합**은 wrap한다. 결합법칙이 성립하므로 청크 결과를 청크 순서로 접는다.
- **int64 평균은 정확한 정수 합이다** (`mean_of` → `int_sum`, `base/int_sum.hpp`의 `IntSum`). 값을 위 32비트와 아래 32비트로 나눠 int64 두 개로 더하고(벡터화되는 정수 덧셈 둘), 청크의 합을 `__int128`로 합친다. 정수 덧셈이라 청크를 어느 순서로 합쳐도 비트가 같다. 합을 double로 한 번 반올림하고 NA가 아닌 개수로 한 번 나눈다(`sd::mean_of`). NA 행은 0이라 그대로 더한다.
  - 배열의 `sd.mean(int)`이 같은 `IntSum`을 쓴다. 그래서 `Series.mean()`과 `sd.mean(s.to_array())`가 비트까지 같다 (루트 DESIGN §9.1 11번, 테스트 `test_int_mean_has_the_array_bits`). groupby의 그룹 평균도 같은 합이라서 그룹 크기와 상관없이 Series와 같다 (§9.1).
  - 2^53을 넘는 값도 반올림 없이 더한다. numpy, pandas(값마다 float64로 바꿔 더한다)보다 정확하고, 마지막 자리가 다를 수 있다.
  - 이 경로의 시간은 frame에서 다시 재지 않았다.
  - `var`, `std`의 평균도 이 값이다. 편차 제곱의 합은 위의 fma pairwise다.
  - bool 평균은 True의 개수(정수 합)를 개수로 나눈다.
- **float min/max는 IEEE 754-2019 minimum/maximum이다** (`ieee_minmax`). AArch64의 `FMIN`/`FMAX`가 정확히 이것이다: NaN을 전파하고 `-0.0 < 0.0`이다.
  - NA가 없으면 NEON 누산기 4개(각 2레인)로 접는다. 스칼라 경우도 같은 명령의 한 레인으로 계산해서 두 경로의 비트가 같다.
  - NA가 있으면 1,024행 블록마다 NA 행을 항등원(min은 +inf, max는 -inf)으로 바꿔 복사한 뒤 같은 명령으로 접는다. 복사는 `valid ? x : 항등원`이고, 컴파일된 코드는 행마다 분기한다(선택 명령으로 바뀌지 않았다).
  - 결합법칙과 교환법칙이 성립하므로 청크 결과를 어떻게 묶어도 같다.
  - 처음 구현(이긴 행의 위치를 들고, 행마다 dtype `switch`와 NA 비트 확인)은 848µs였다. 값 자체를 들고 다니고 NEON으로 접어 73µs다 (pandas 325µs, Polars 125µs).
  - int64 min은 NEON에 64비트 정수 min이 없어 비교 + `bsl`로 벡터화된다. 119µs, numpy 79µs다.
- **문자열 min/max**는 이긴 행(같으면 앞의 것)을 골라 1행 슬라이스로 돌려준다.
- groupby는 같은 함수(`float_sum`, `ieee_minmax`)를 그룹마다 부른다. int64의 그룹 평균은 배열을 부르지 않고 그룹의 행을 float64로 바꿔 `float_sum`으로 더한다 (§9.1).

### 7.5 astype과 문자열 문법 (`ops/cast.cpp`, `ops/parse.hpp`)
- **float → string은 Python `str(float)`이다.** 다시 읽어 같은 값이 되는 가장 짧은 숫자열(`std::to_chars`)을 구하고, 10진 지수가 (-4, 16]이면 고정소수점, 아니면 지수 표기(`1e+16`, 지수는 최소 두 자리)로 쓴다. 정수 값에는 `.0`을 붙인다. NaN, ±inf, 비정규수를 포함한 임의의 float에서 `str(v)`와 같다.
  - 청크마다 자기 문자열에 청크 기준 offsets로 쓴 뒤, 청크 순서로 이어 붙인다.
- **string → 숫자는 엄격한 ASCII 문법이다.** 부호, 숫자, 소수점, 지수, 그리고 float는 `nan`, `inf`, `infinity`(대소문자 무관)만 받는다. 공백, `_`, 유니코드 숫자는 받지 않는다(Python `int()`는 받는다).
  - **`astype`과 `read_csv`가 같은 문법 코드(`parse.hpp`)를 쓴다.** 그래서 `read_csv(..., dtype="string")` 뒤에 `astype`해도 같은 결과다.
  - 파서는 §10.4다. 범위를 넘는 float(`1e400`)는 Python처럼 inf다. `astype`은 텍스트 앞의 읽을 수 있는 메모리를 넘기지 않아, 짧은 텍스트는 넓은 읽기 대신 자리마다 읽는 길로 간다(결과는 같다).
- **float → int64**: 배열 영역과 같은 `saturate_to_int64`(`base/saturate.hpp`)다. 0 쪽으로 자르고, 범위 밖과 ±inf는 포화, NaN은 0이다. ARM `FCVTZS` 명령 하나라서 실패하는 행이 없고 분기도 없다 (array.md §2.4). frame에서 이 경로의 시간은 다시 재지 않았다.
- 실패할 수 있는 변환(string → 숫자, string → bool)은 행 순서로 가장 앞의 실패를 보고한다 (§11.6).
- string → bool은 `"True"`와 `"False"`만 받는다.

### 7.6 벡터화를 막는 앨리어싱
- **`uint8_t` 포인터로 쓰는 루프는 경계와 포인터를 지역 변수로 꺼낸다.**
  - `for (i = 0; i < col.length; ++i) out[i] = …`에서 `out`이 `uint8_t*`면, C++ 규칙상 `out[i]`가 어떤 객체든(`col.length`도) 바꿀 수 있다. 컴파일러가 매번 `col.length`를 다시 읽고 벡터화를 포기한다.
  - `const int64_t n = col.length;`로 꺼내자 `isna`가 288 → 48µs였다.
- **람다가 값으로 캡처한 변수도 클로저 객체의 멤버다.** 태스크 안의 `[o, flip](lo, hi) { … o[i] ^= 1; }`에서 컴파일러는 매 반복 `this->o`를 다시 읽었다. 병렬로 바꾸자 1스레드 `isna`가 44 → 276µs가 되었고, 태스크 안에서 지역 변수로 옮기자 돌아왔다.
- **연산은 함수 포인터가 아니라 람다로 넘긴다.** 직렬 루프에서는 함수 포인터가 상수로 전파되어 인라인되지만, `std::function`을 거치는 태스크 안에서는 행마다 간접 호출이 된다(`int + int` 293 → 854µs).
- **Column을 참조로 잡은 람다가 행마다 그 포인터를 다시 읽는 것**도 같은 문제다. int64 저장이 그 포인터를 바꿀 수 있다고 본다 (§8.8).

---

## 8. 해시와 factorize (`ops/hash.hpp`, `ops/hash.cpp`)

### 8.1 factorize 하나 위에
```
값      [3.0, NaN, 1.0, 3.0, -0.0, 0.0, NA, NaN]
codes   [0,   1,   2,   0,   3,    3,   4,  1  ]   처음 나타난 순서의 정수 코드
first   [0, 1, 2, 4, 6]                           코드마다 처음 나온 행
```
| 기능 | 계산 |
|---|---|
| `unique_values()` | 작은 범위의 int64·bool은 비트맵 하나로 바로 (`unique_sorted`, §8.9). 그 밖에는 `col.take(first)`를 groupby의 그룹 순서(§9.2)로 정렬 (`sort_groups`) |
| `drop_duplicates()` | 처음 나온 행만 남긴다 (`duplicated(keep="first")`의 반대) |
| `nunique()` | 코드의 개수 (`dropna=True`이고 NA가 있으면 1을 뺀다) |
| `duplicated(keep="first")` | `위치 != first.take(codes)` |
| `duplicated(keep="last")` | 위와 같고 코드마다 마지막 행을 쓴다 |
| `duplicated(keep="none")` | `counts.take(codes) > 1` |
| `value_counts()` | 개수로 안정 정렬한 코드 순서대로 `take(first)`와 개수. 정렬은 개수(내림차순이면 `~개수`) 키의 기수 정렬이다(§9.3) |
| groupby | 코드가 그룹 번호 (§9.1) |
| merge | 오른쪽 키의 코드로 왼쪽 키를 조회 (§9.7) |
| Index 엔진 | 라벨의 코드 (§6.2) |

- 그래서 "같은 키"의 규칙은 C++ factorize 한 곳에만 있다.
- 결과에는 원래 값이 나온다. `first`가 원래 행 번호이므로 `take(first)`가 먼저 나온 원래 값(`-0.0`이나 원래 NaN 비트)을 돌려준다.
- 코드는 **처음 나타난 순서**다. 결과의 순서(`drop_duplicates`, `value_counts(sort=False)`, `sort=False`의 groupby)가 이것으로 정해지고, 스레드 수와 무관하다 (§8.7, §8.9).
- **어느 테이블인가**: 값이 작은 범위에 있는 int64와 bool은 직접 주소 표(§8.9), float64와 범위가 넓은 int64는 Swiss table(§8.3), 문자열은 해시를 든 Swiss table이다. 셋 다 같은 코드를 낸다.
- factorize는 공개하지 않는다.

### 8.2 키 정규화 (`float_key`)
```cpp
uint64_t float_key(double d) {
    if (d != d) return 0x7ff8000000000000;   // NaN은 하나의 NaN으로
    if (d == 0.0) return 0;                  // -0.0은 +0.0의 비트로
    return bits(d);
}
```
- 해시 테이블은 "비트가 같으면 같은 키"일 때 가장 단순하다. 그래서 같은 키 규칙(§2.3)을 해시 전에 적용한다.
- 정렬 키(§9.3)도 같은 함수에서 시작한다.
- **NA 행은 테이블에 넣지 않는다.** 처음 나온 NA 행에 코드 하나를 주고, 이후 NA 행은 모두 그 코드를 쓴다. 코드는 그 첫 행의 순서대로 매긴다.

### 8.3 Swiss table (`SwissTable`)
```
제어 바이트 (칸마다 1바이트): 0 = 빈 칸, 0x80 | 해시의 아래 7비트 = 찬 칸 (태그)
그룹 (칸 16개) = (해시 >> 7) & (그룹 수 - 1)

조회:
  ctrl  = 그룹의 제어 바이트 16개            vld1q_u8
  match = (ctrl == 태그)                     vceqq_u8, 비교 한 번에 16칸
  match의 칸마다 키를 실제로 비교            태그 7비트가 같을 확률은 1/128
  빈 칸이 있으면 여기서 끝 (없음)            vceqzq_u8
  아니면 다음 그룹 (선형)
```
- **칸** `{word, code}` 16B: `word`는 8바이트 키(int64, float64, bool)면 키 자체, 문자열이면 해시다. 8바이트 키는 칸만 읽고 비교가 끝난다. 문자열은 해시가 같을 때만 코드로 키를 꺼내 비교한다.
- NEON에는 x86 `movemask`가 없다. `vshrn_n_u16(…, 4)`로 16바이트 비교 결과를 64비트 정수 하나(칸마다 4비트)로 줄이고 `ctz`로 다음 후보를 찾는다.
- 빈 칸이 0이라서 `calloc`한 메모리가 그대로 빈 테이블이다. 칸 배열은 `malloc`(초기화 안 함)이고 제어 바이트가 "찼다"고 할 때만 읽는다. 그래서 크게 잡아도 실제로 쓴 페이지만 메모리를 차지한다 (§8.5).
- 7/8이 차면 두 배로 늘린다. 태그로 걸러지므로 꽉 차도 키 비교가 적다.
- 테이블은 처음 `reserve`나 `insert`에서만 메모리를 잡는다.
- **쓰는 곳**: 문자열, float64, 범위가 넓은 int64 키. 작은 범위의 정수는 해시 없이 직접 주소 표다 (§8.9).
- **왜 Swiss table인가: 분기 예측 실패.** linear probing(칸마다 "비었나, 키가 같나"를 분기로 묻는다)은 1M 행, 키 1,000종에서 테이블(2,048칸 × 16B = 32KB)이 L1에 다 들어가는데도 행마다 약 19사이클이었다.
  - probe 횟수는 원인이 아니었다(평균 1.48회, 적재율 0.49의 이론값).
  - 프로파일 샘플의 2/3가 "첫 칸의 키가 다르다 → 다음 칸" 분기 바로 뒤에 몰렸다. 첫 칸에서 찾을지(약 2/3) 한 칸 더 갈지를 예측할 수 없다. 틀릴 때마다 키 읽기 → fmix64 → 칸 읽기 → 비교의 지연 사슬이 처음부터 다시 시작된다.
  - Swiss table에서 남는 분기는 "후보가 있나"와 "빈 칸이 있나"이고, 둘 다 거의 항상 같은 쪽으로 간다.

  - Swiss table이 int `nunique`(키 1,000종)에서 약 2배, 키 63만 종에서 약 1.4배 빨랐다.

- `LinearTable`은 비교용으로 남아 있고 쓰지 않는다 (`KeyTable = SwissTable`).

### 8.4 해시 함수
- **8바이트 키**: murmur3의 마무리 함수 fmix64(곱셈 2번, 시프트 3번). 입력의 모든 비트가 출력의 모든 비트에 영향을 준다. 키가 0, 1, 2, …로 붙어 있어도 그룹을 고르는 비트와 태그가 고르게 섞인다.
- **문자열** (`HashBytes`): 8바이트씩 읽어 128비트 곱셈의 두 반쪽을 XOR해 섞는다(wyhash의 "mum"). 길이를 섞어 `"a"`와 `"a\0"`가 다르고, 끝에 fmix64를 한 번 더 한다.
  - 마지막 1~8바이트는 문자열을 벗어나지 않는 겹친 읽기다(wyhash와 같다): 8바이트 이상이면 끝에서 8바이트를 읽어 이미 섞은 바이트를 시프트로 버리고, 4~7바이트는 4바이트 읽기 둘, 1~3바이트는 첫·가운데·끝 바이트다. 가변 길이 `memcpy`(라이브러리 호출)로 꼬리를 읽던 때보다 짧은 키의 `nunique`가 5~15% 빠르다. 해시 값만 바뀌고 코드는 같다 (FR12).
  - 바이트마다 곱셈 하나(FNV-1a + fmix64)에서 곱셈 섞기로 바꾼 이득은 약 10%였다. 시간 대부분이 해시가 아니라 키 비교(`memcmp`)와 메모리 접근에 있다.
- 파티션(§8.7)은 해시의 **위쪽** 8비트, 태그는 **아래** 7비트, 그룹은 그 위의 비트를 쓴다. 서로 겹치지 않는다.

### 8.5 테이블을 미리 잡는다
- **8바이트 키는 행 수만큼, 문자열 키는 최대 2¹⁶행만큼 처음부터 잡는다** (`reserve`). 행 수는 키 수의 상한이라서 8바이트 키 테이블은 자라지 않는다.
  - 길이와 dtype으로만 정하므로 데이터를 보는 휴리스틱이 아니다.
- 32칸에서 시작해 두 배씩 키우면 키 1,000개에 6번 자란다. 자랄 때마다 새 메모리를 받고 첫 쓰기마다 페이지 폴트가 난다. 1만 행 groupby의 25%, merge의 14%가 `grow()`였다.
- 키워 가기 대비 시간 비율 (int 키):

  | 행 / 키 | 2¹⁶까지만 | 2¹⁸까지만 | 행 수만큼 |
  |---|---:|---:|---:|
  | 1천 / 1천 | 0.41 | 0.42 | 0.42 |
  | 10만 / 10만 | 0.48 | 0.48 | 0.48 |
  | 1M / 1천 | 0.85 | 0.92 | 0.97 |
  | 1M / 10만 | 0.84 | 0.69 | 0.82 |
  | 1M / 1M | 0.95 | 0.78 | **0.54** |

- string 키 (키워 가기 대비 시간 비율):

  | 행 / 키 | 2¹⁶까지만 | 행 수만큼 |
  |---|---:|---:|
  | 1M / 10만 | **0.95** | 1.26 |
  | 1M / 1M | 1.00 | 0.95 |

  - int는 행 수만큼 잡아도 키워 가기보다 느린 경우가 없다.
  - string은 행 수만큼 잡으면 1M 행, 키 10만 개에서 26% 느리다. 32MB 칸 배열에 키 10만 개가 흩어지고, 일치를 확인할 때 코드로 키를 한 번 더 읽는다(`keys_[code]`). 2¹⁶까지만 잡으면 어디서도 느리지 않다.
- 메모리: 1M 행이면 제어 바이트 2MB, 칸 32MB를 잡지만 §8.3대로 키가 들어간 페이지만 차지한다. 큰 테이블의 페이지 비용은 키가 적을 때 약 6%였다. (작은 범위의 int는 지금 §8.9의 표를 쓴다.)
- 파티션(§8.7)은 파티션의 행 수만큼 잡는다.

### 8.6 다중 키
1. 컬럼마다 factorize한다.
2. 코드를 혼합 기수로 합친다: `acc = acc × n_j + code_j`. 키 튜플 하나가 int64 하나가 된다.
3. 곱이 2⁶²를 넘을 것 같으면, 그때까지 합친 `acc`를 먼저 factorize해 실제로 나온 조합의 수로 좁힌다.
4. 마지막에 `acc`를 한 번 더 factorize한다. 코드가 빈틈없고 처음 나타난 순서를 따르게 된다.
- 테스트는 좁히는 단계가 돌도록 만든다: 각 2,999종(소수)인 컬럼 7개, 2999⁷ > 2⁶².

### 8.7 해시로 나눈 병렬 factorize (`Partitioned`)
행 수 × 비용이 기준값(§11.2) 이상이고 풀을 쓸 수 있으면 **해시의 위쪽 8비트로 P = 256개 파티션**에 나눈다. 아니면 P = 1, 곧 직렬 패스다. 비용은 8바이트 키 1, 문자열 4다(해시와 바이트 비교가 행마다 몇 배 든다). 그래서 문자열은 10만 행에서도 나뉜다 (FR12).
```
1. 행마다 해시, (청크, 파티션)별 개수                      청크 병렬
2. (파티션, 청크)별 시작 위치: 파티션 순서, 그 안에서 청크 순서
3. (key 또는 해시, 행) 쌍을 파티션으로 흩뿌림              청크 병렬. 파티션 안에서는 행 순서
4. 파티션마다 Swiss table로 지역 코드와 첫 등장 행         파티션 병렬. 테이블은 파티션 행 수만큼 미리 잡는다
5. 전역 코드 = 그 키의 첫 등장 행보다 앞선 첫 등장 행의 수  행 청크 병렬
6. 행 순서로 코드 쓰기                                    청크 병렬
```
- 같은 키는 같은 해시라서 항상 한 파티션에 간다. 파티션 안에서는 행이 원래 순서라서 지역 코드도 처음 나타난 순서다. 5번이 이것을 전역의 처음 나타난 순서로 바꾼다. **코드는 직렬 factorize와 정확히 같다.** 그래서 P가 스레드 수를 따라도 된다 (§11.5).
- **왜 해시 분할인가**: "청크마다 테이블을 만들고 합치기"는 키가 많으면 합치기가 직렬이 된다. 해시로 나누면 키가 많아도 고르게 나뉘고, 파티션마다 테이블이 작다.
  - 계산: 1M 행이면 파티션 하나가 약 3,900행이고, 칸 8,192개 × 16B = 128KB짜리 테이블이다. 63만 종이 한 테이블이면 2M 칸 × 16B = 32MB로 probe마다 캐시 미스가 난다.
  - 256은 기수 정렬의 바이트 자리(256칸), counting sort의 버킷 수(256)와 같은 값이다. (청크 × 파티션) 카운터가 1M 행에서 62 × 256개다.
- **처음 구현은 1스레드에서 2~4배 느렸다.** 단계별로 재서 다섯 가지를 고쳤다 (1M 행, 키 63만 종).

  | 문제 | 원인 | 고친 것 |
  |---|---|---|
  | 4단계가 느림 | 파티션이 행 번호로 키와 해시를 읽어 매번 캐시 미스 | (키, 행) 쌍 자체를 흩뿌린다(3단계). 파티션은 앞에서부터 차례로 읽는다 |
  | 전역 번호가 느림 | 첫 등장 행마다 이진 탐색(63만 × 20단계) | 첫 등장 행에만 순위를 써 두는 배열 |
  | 10스레드에서 3배뿐 | 모든 파티션이 공유 배열의 아무 곳에나 씀(거짓 공유) | 전역 번호와 코드 쓰기를 행 청크 단위로. 태스크는 자기 청크 범위에만 쓴다 |
  | 테이블이 자람 | 32칸에서 두 배씩, 재해시와 할당 | 파티션 행 수만큼 미리 잡는다. 빈 칸 = 0이라 calloc의 0 페이지를 그대로 쓴다 |
  | 청크마다 이진 탐색 | 256 파티션 × 62 청크 × 3번, 분기 예측 실패(63만 종에서 7ms) | 파티션마다 청크 경계를 한 번 훑어 계산. 순위 매기기도 분기 없이 |

  - 그래도 키 1,000종에서 분할이 1스레드로 직렬 패스의 약 2.2배였다. 단계가 여럿이라 직렬 한 번보다 메모리를 여러 번 지난다. 분할의 본질적 비용이라서 1스레드는 직렬 패스를 쓴다 (§11.5).
- 10스레드에서 키가 많은 경우 약 4~5배, 키 1,000종은 약 2배다(직렬도 테이블이 L1에 들어간다). 작은 범위의 int는 이 분할을 쓰지 않는다 (§8.9).

### 8.8 조회 (`find_rows`)
merge의 build/probe(§9.7)와 Index 엔진의 여러 라벨 조회(§6.2)가 쓴다.
- **dtype과 NA 유무를 호출마다 한 번 고른다.** 행 루프는 찾기와 저장뿐이다.
  - 행마다 "NA인가, 문자열인가, 어느 dtype인가"를 묻던 때 1스레드 1M 행 merge의 48%가 조회였다.
  - 읽는 포인터(값, 비트맵, 문자열 offsets)는 태스크의 지역 변수다 (§7.6).
- NA 코드 보정은 `c - (c > skip)`이다. NA가 없으면 `skip`이 int64 최댓값이라 아무것도 바뀌지 않는다.
- build 쪽(`ValueCodes`)이 작은 범위의 int64·bool이면 해시 테이블 대신 직접 주소 표(§8.9)에 코드를 두고, 조회는 행마다 한 번 읽기다: 범위 밖 키와 NA 행은 분기 없이 빈 칸을 읽어 -1이 된다. 코드는 해시 경로와 같다.

### 8.9 직접 주소 표 (작은 범위의 정수 키)
int64와 bool 컬럼의 값이 `[lo, lo + span)`에 있고 `span ≤ min(max(2n, 2¹⁶), 2²⁸)`이면(`direct_range`, NA 행의 0도 범위에 든다) 해시를 쓰지 않는다. 키 `lo + k`의 칸은 `k`이고, 칸에는 1 + 코드(0은 아직 없음)를 4바이트로 둔다. 표는 많아야 행마다 8바이트로 코드 배열보다 크지 않다. 해시, 태그, probe가 없고 행마다 메모리 접근이 하나다 (FR4).
- **직렬 패스** (`factorize_direct`): 행 순서로 칸을 읽고, 0이면 다음 코드를 준다. 키가 많으면(`span ≥ n / 4`) "새 키인가"를 분기 없이 처리한다: 모든 행이 칸과 `first`의 다음 자리를 쓰고, 새 키일 때만 `first`의 끝이 앞으로 간다. 1M 행 위에 키 63만 종이면 새 키 여부는 거의 무작위라서 분기가 예측되지 않았고, 분기 없는 판이 약 2.6배 빠르다. 키가 적으면 새 키가 드물어 분기를 그대로 둔다(분기 없는 판의 store→load 사슬이 더 느렸다).
- **병렬** (키가 적을 때만, `span < n / 4`): 분기 없는 네 패스다. 코드는 직렬 패스와 같다.
  1. `head[k]` = 키 k의 첫 행: 행 청크 병렬의 원자 최솟값(`__atomic_fetch_min`, AArch64 `LDSMIN`). 이미 더 작은 행이 있으면 읽기만 하므로 캐시 라인이 공유 상태로 남는다. 최솟값은 쓰는 순서와 상관없이 같다.
  2. 첫 행의 비트맵(`head[key(i)] == i`)과 64행 워드마다의 개수
  3. 워드 개수의 누적합: 첫 행의 코드는 그보다 앞선 첫 행의 수(비트맵 순위)다.
  4. 행마다 `code = rank(head[key(i)])`, 첫 행은 `first[code]`도 쓴다. NA 행은 첫 NA 행의 코드다.
  - 키가 많으면 병렬 판을 쓰지 않는다. 거의 모든 행이 원자 연산을 해서 10스레드가 직렬 한 번보다 느렸다(1M 행, 키 1M 종에서 약 4배). 그래서 풀이 있어도 직렬이다. 코드가 같으므로 경로가 스레드 수와 키 수를 따라도 된다 (§11.5).
- **쓰는 곳**: factorize(그 위의 `nunique`, `duplicated`, `value_counts`, groupby, 다중 키의 첫 단계), merge의 build와 조회(§8.8, §9.7), `unique_values`.
- **`unique_values`** (`unique_sorted`): 범위만큼의 비트맵에 나온 키를 표시하고, 비트맵을 차례로 읽어 값을 낸다. 값이 곧 키라서 대표 행을 고를 일도, 정렬도 없다. NA가 있으면 끝에 NA 하나다. factorize + take + 정렬보다 1M 행, 키 63만 종에서 약 5배 빠르다 (FR5).
- 정수가 넓은 범위에 흩어져 있으면 표가 커서 쓰지 않는다(Swiss table, §8.3). 범위는 길이와 값의 최솟값·최댓값으로만 정하고, 어느 표든 결과가 같다 (§1.3).

---

## 9. groupby, 정렬, merge

### 9.1 groupby (`ops/groupby.cpp`)
```
키      [b,  a,  b,  NA]       factorize (§8)
codes   [0,  1,  0,  2 ]
                                counting sort: 그룹별로 행 모으기 (CSR, §9.6)
start   [0, 2, 3, 4]            그룹 g의 행 = rows[start[g] : start[g+1]]
rows    [0, 2, 1, 3]            그룹 안에서는 원래 행 순서
```
1. 키 컬럼을 factorize해서 행마다 그룹 코드를 얻는다.
2. `Groups`가 행을 그룹별로 모은다. 그룹 안에서 원래 행 순서를 지킨다.
   - 코드가 0에서 시작해 내려가지 않고 1씩만 오르면(0, 0, 1, 1, 1, 2, … : 키로 이미 묶인 컬럼, 키로 정렬된 컬럼) 그룹마다 행이 한 구간이다. 구간의 경계만 찾고 행을 옮기지 않는다(`contiguous_`, 위치 배열 없음, 행 j는 j). 그런 코드의 counting sort는 개수와 자리를 올릴 때마다 앞의 저장을 기다려 가장 느린 경우였다. 키로 정렬된 1M 행의 `sum`이 약 4배 빠르다 (FR7).
   - 그 밖에는 안정 counting sort(§9.6)다.
3. 그룹마다 §7.4와 같은 함수로 리덕션한다. dtype과 NA 유무와 행의 위치(구간이면 j, 아니면 `rows[j]`)는 그룹 루프 밖에서 템플릿으로 고른다.
   - **sum, mean, var, std**: 그룹의 행을 원래 순서대로 pairwise로 더한다. 그래서 그룹 결과는 `df[df["k"] == 키]["v"].sum()`과 **비트까지 같다.** 테스트의 정답이 이것이다. 그룹마다 차례로 더하기(일반 누적)는 이 일치를 버리지만 CSR을 쓰는 지금보다 뚜렷이 빠르지 않아 하지 않았다 (FR7).
   - **int64의 mean**(과 var, std 안의 평균): 그룹의 행을 `IntSum`으로 정확히 더해 한 번 반올림한다. Series의 int64 `mean`(§7.4)과 같은 계산이라 그룹 크기와 상관없이 비트가 같다.
   - **min, max**: dtype별 루프로 그룹의 이긴 행을 고른다(같으면 앞의 행, float는 IEEE 754-2019 minimum/maximum: NaN이 이기고 첫 NaN이 남는다, `-0.0 < 0.0`). **first, last**: 처음이나 마지막 값. 그 뒤 `take`한다 (§7.1). 값이 없는 그룹은 -1이라 NA가 된다.
   - NA가 없는 컬럼은 그룹의 NA 개수를 세지 않는다(크기가 곧 개수).
4. 출력 순서: 코드 순서(처음 나타난 순서)다. `sort=True`면 §9.2의 순서, `dropna=True`면 NA 키 그룹을 뺀다.
- 행을 그룹별로 모으는 대가로 행 위치 배열(8B × n)을 한 번 더 쓴다(구간이면 없다). 대신 float 합이 Series와 같아지고, 그룹을 태스크로 나누기 쉽다.
- **병렬**: 그룹을 약 16K행(`kChunkRows`)씩 통째로 묶어 태스크로 나눈다. 그룹마다 독립이라 경계가 어디든 결과는 같다. 그룹별 NA는 바이트로 적은 뒤(태스크가 비트맵 바이트 하나를 나눠 가질 수 있다) 끝에 비트맵으로 접는다.
- **값 컬럼이 여럿이면 한 번에 부른다** (`Groups.reduce_columns`): 행이 기준값 아래이고 행 × 컬럼은 기준값 이상이면 컬럼이 태스크다. 10만 행 × 100열의 `groupby("k").sum()`이 10스레드에서 약 5배다. 결과는 컬럼마다 부른 것과 같고, 에러는 컬럼 순서로 먼저 검사한다 (FR14).
- 키는 결과의 일반 컬럼이고 결과 Index는 RangeIndex다. float와 bool은 Index가 될 수 없어서, 키 dtype이나 개수와 상관없이 결과 모양을 하나로 했다 (`docs/api/frame.md`).

### 9.2 그룹 순서
- **키가 숫자나 bool 하나면** 그룹마다 한 행씩인 대표 값을 안정 기수 정렬(`argsort`, §9.3)한 순서가 곧 그룹 순서다. NA는 맨 뒤, 같은 값(`-0.0`과 `0.0`, NaN끼리)은 그룹 순서대로 남는다. 아래의 factorize, 순위, counting sort를 거치지 않는다. 키 63만 종에서 그룹 순서 정하기가 약 3.5배 빠르다 (FR8).
- **그 밖(문자열 키, 여러 키)에는 그룹이 아니라 키 값을 정렬한다.** 마지막 키부터:
  1. 그룹들의 그 키 값의 고유값을 정렬한다 (기수 정렬, §9.3, §9.4). 그룹마다 그 값의 순위 0..u-1을 준다. NA는 맨 뒤다.
  2. 순위로 그룹을 안정 counting sort한다.
  - 안정 정렬이라 뒤쪽 키의 순서가 같은 순위 안에서 유지된다(LSD). 앞쪽 키가 우선이 된다.
- 그룹 번호를 `std::stable_sort`로 정렬하며 비교마다 키 컬럼을 차례로 비교하면, 그룹 약 95만 개(int 1,000종 × string 1만 종, 1M 행에서 나타난 쌍은 951,459개)에서 비교가 약 1,900만 번(그룹 수 × log₂ 그룹 수)이고 비교마다 문자열 비교다. 키 값의 순위 + counting sort가 약 4배 빠르다.

### 9.3 정렬 키와 LSD 기수 정렬 (`ops/radix.hpp`, `ops/sort.cpp`)
- `argsort(keys, ascending, na_last)`가 행의 순서를 돌려주고, `sort_values`와 `sort_index`는 그 위치로 `take`한다.
- **이미 정렬된 키 하나**: 키가 하나이고 NA가 없으면 먼저 컬럼을 한 번 읽어 그 방향으로 이미 정렬되어 있는지 본다(같은 값이 이어져도 된다. 안정 정렬이 그 순서를 지킨다). 그렇으면 `argsort`가 None을 돌려주고, 결과는 take 없이 컬럼을 Copy-on-Write로 공유한다. Index도 같다. 단, RangeIndex는 take가 돌려줄 Index(라벨 0..n-1)로 바꾼다(§2.4). 순서가 끝까지 정해진 뒤에도 위치가 0..n-1이면(여러 키, NA가 있는 경우) None이다. 정렬된 1M int64가 약 70배 빠르다 (FR2).
- **값을 대소 관계가 보존되는 uint64로 바꾼다** (`radix_key`).

  | dtype | 변환 | 이유 |
  |---|---|---|
  | int64 | `x ^ (1 << 63)` | 부호 비트를 뒤집으면 음수가 양수보다 작은 부호 없는 수가 된다 |
  | bool | 0 / 1 | |
  | float64 | `float_key`(§8.2) 다음 양수는 `bits \| (1 << 63)`, 음수는 `~bits` | 양수는 비트 순서가 값 순서다. 음수는 비트가 클수록 작으므로 모두 뒤집는다 |
  | string | 8바이트 접두사 (§9.4) | |

  - **NaN > +inf**: NaN의 비트 `0x7ff8…`가 +inf의 `0x7ff0…`보다 크다. 변환 뒤에도 가장 크다.
  - **`-0.0 == 0.0`**: 정규화로 같은 키가 된다. 안정 정렬이라 원래 순서를 지킨다.
  - **내림차순은 키의 모든 비트를 뒤집는다**(`~key`). 순서가 정확히 뒤집히고 같은 키는 계속 같아서 **안정성이 유지된다.** "오름차순으로 정렬한 뒤 뒤집기"는 같은 값의 순서까지 뒤집는다.
  - **NA는 키에 넣지 않는다.** 정렬 전에 떼어 두었다가 `na_position`에 따라 앞이나 뒤에 붙인다. 방향과 상관없이 위치가 정해진다. NA가 없으면 위치 배열을 그 자리에서 정렬한다(떼고 붙이는 복사가 없다).
- **LSD 기수 정렬** (`radix_sort`): 낮은 자리부터 안정적으로 나눠 담는다. 비교가 없다. 행마다 키 8바이트와 위치 8바이트를 패스마다 옮긴다. 먼저 키를 한 번 훑어(`scan_keys`) 필요한 패스만 정한다 (FR1).
  - **이미 오름차순**(같은 값 허용)이면 아무것도 옮기지 않는다.
  - **내림차순**(같은 값 허용)이면 같은 키의 묶음을 통째로 뒤에서부터 늘어놓는다(`reverse_runs`). 묶음 안의 순서는 그대로라 안정 정렬과 같다.
  - **그 밖에는 키가 다른 비트만 정렬한다**: (a) `키 − 최솟값`의 비트, 또는 (b) 모든 키에서 같지는 않은 바이트 자리 가운데 패스가 적은 쪽이다. 모든 키에서 같은 자리를 지나는 패스는 행을 한 칸도 옮기지 않으므로 빼도 순서가 같다. 0..10⁶의 int64는 8패스가 아니라 2~3패스다.
  - **자리의 폭**: 나누지 않는 정렬(§9.5)은 (a)의 비트를 11비트 이하의 고른 자리로 나눈다(17~22비트 키는 2패스). 패스 하나의 비용이 8~11비트에서 거의 같고, 2¹⁵개 아래에서는 2,048칸 히스토그램을 비우고 더하는 비용이 줄인 패스보다 커서 바이트 자리다(`kWideRadixKeys`). 나누는 정렬은 바이트 자리다: 11비트 자리(2,048개 출력 흐름)는 10스레드에서 캐시를 넘쳐 더 느렸다.
  - 호출하는 쪽이 정하는 자리의 상한(bool 1바이트, 문자열 순위는 `u − 1`을 담는 바이트)은 그대로 있다. 위의 훑기가 같은 일을 하므로 이제는 상한일 뿐이다.
  - 어느 길이든 결과는 하나의 안정 정렬이다. 같은 값의 순서(guardrail: 정렬의 동률)가 스레드 수나 길에 따라 바뀌지 않는다.
- **다중 키**: 마지막 키부터 한 키씩 안정 정렬한다. 키마다 자기 방향과 NA 위치만 적용하면 된다.
- pandas의 `kind="stable"`은 비교 정렬이라 무작위 int64 1M에서 우리가 15배 넘게 빠르다. 생성되는 속도 표는 `bench/frame/speed.md`다.

### 9.4 문자열의 순서
1. 문자열마다 앞 8바이트를 big-endian uint64로 읽는다(모자라면 0으로 채움). 정수의 대소가 바이트 사전식 순서와 같다.
2. 이 접두사로 LSD 기수 정렬한다.
3. 접두사가 같은 구간은 다음 8바이트로 같은 일을 되풀이한다(MSD로 내려가기). 구간이 128행(`kStringRadixRun`) 아래이거나, 구간의 문자열에 남은 바이트가 없으면 비교 정렬한다. 구간은 시작하는 청크가 맡는다(구간끼리 독립). 기준값(131,072행) 이상인 구간은 호출 스레드가 하나씩 맡아 다음 8바이트를 병렬 기수 정렬한다.
- `"a"`와 `"a\0"`는 0으로 채운 키가 끝까지 같아서 같은 구간에 남고, 비교 정렬에서 정확히 정렬된다.
- 무작위 키에서는 접두사가 거의 다 달라 3단계가 거의 없다.
- 흔한 키(`customer_…`, `https://…`)는 컬럼 전체가 접두사 하나라서, 비교 정렬로만 하면 직렬 비교 정렬 하나가 된다. 1M 고유 문자열 `customer_` 접두사의 `sort_values`가 MSD로 내려가면 비교 정렬로만 할 때보다 1스레드 약 2배, 10스레드 약 8배 빠르다. 어느 경로든 결과는 하나의 안정 정렬이라서 같다.
- 고유 문자열 10만 개를 비교 정렬하면 비교가 약 170만 번이고, string 키 groupby 시간의 대부분이 그 안의 `memcmp`였다. 접두사 기수 정렬로 groupby string(10만 종) 38.4 → 22.6ms.
- `sort_values`의 string 정렬과 groupby의 그룹 순서가 같은 코드(`order_rows`)를 쓴다. `sort_values`에서 문자열 키는 키가 하나여도 고유값의 순위가 키를 대신한다(방향 반전 = 순위 반전): factorize → 고유값을 `order_rows`로 정렬 → 행마다 순위를 기수 정렬한다.
- string 정렬 1M 행(고유값 10만)은 pandas(비교 정렬)보다 약 6배 빠르다(`bench/frame/speed.md`).

### 9.5 기수 정렬: 나눌 때와 나누지 않을 때
- **나누는 조건**: 풀을 쓸 수 있고 키가 기준값(2¹⁷) 이상일 때다. 그 아래에서는 나누지 않는 정렬의 넓은 자리(§9.3)가 바이트 자리로 나누는 정렬보다 빨랐다(10만 키에서 약 1.3배) (FR3).
- **나눌 때**: 자리마다 청크별 히스토그램 → (자리 값, 청크) 순서의 시작 위치 → 청크별로 흩뿌리기. 원소가 놓이는 자리가 직렬 패스와 정확히 같다. 키 훑기(§9.3)도 청크별로 하고 청크 순서로 합친다.
  - 자리마다 청크의 원소가 달라지므로 히스토그램을 매번 다시 센다.
  - **히스토그램을 4개로 나눠 센다.** 같은 칸을 연달아 올리는 의존 사슬이 끊긴다. 하나로 세면 1스레드가 약 1.3배 느렸다.
  - 청크의 자리 위치는 지역 배열로 복사해 쓴다. int64 행 저장이 힙의 배열과 앨리어싱될 수 있어서다.
  - 자리는 바이트다. 11비트 자리는 10스레드에서 약 1.8배 느렸다(2,048개 출력 흐름이 캐시를 넘친다).
- **나누지 않을 때**(풀을 쓸 수 없을 때, 또는 키가 기준값 아래): 키를 한 번 읽어 모든 자리의 히스토그램을 세는 직렬 정렬을 쓴다(자리마다 세는 것보다 1천~1만 키에서 약 1.3배 빠르다). 같은 안정 정렬이라 결과가 같다.
  - GIL 규칙: 일(`n × 2`)이 기준값 이상이면 `Unlocked`로 놓는다.

### 9.6 안정 counting sort (`counting_sort`)
위치 0..n-1을 키(0..K-1)로 안정 정렬해 CSR(`start`, `perm`)을 만든다. groupby의 행 모으기(§9.1), 그룹 정렬(§9.2), merge의 build 쪽(§9.7)이 쓴다. Index 엔진의 CSR(§6.2)은 `LabelTable` 안에서 같은 일을 직렬로 한다.
- 1스레드면 직렬 counting sort다: 키마다 개수 → 누적합 → 한 번 더 훑으며 제자리에.
- 나눌 때 교과서 방법은 (청크 × 키) 카운터가 필요하다. 1M 행, 그룹 63만 개면 62 × 63만 = 3,900만 개라 쓸 수 없다. 그래서 두 단계로 한다.
  1. 키의 위쪽 비트로 256개 버킷에 안정적으로 흩뿌린다 (청크 × 256 카운터).
  2. 버킷마다 자기 키 범위만 지역 counting sort한다 (버킷 병렬).
- 버킷 너비는 2의 거듭제곱이라 버킷은 시프트 하나다. 처음에 `key / width`로 나눴을 때 행마다 나눗셈 하나가 직렬 counting sort 전체보다 비쌌다.
- 버킷 경계의 `start[k0]` 칸은 앞 버킷의 끝이다. 버킷은 `(k0, k1]` 칸만 쓴다. 그러지 않으면 두 버킷이 같은 칸을 쓴다.
- 결과는 직렬 counting sort와 같다. 그래서 나누는 방식이 스레드 수를 따른다 (§11.5).

### 9.7 merge (`ops/join.cpp`)
```
1. 오른쪽 키만 테이블에 넣는다 (build, ValueCodes)        → 오른쪽 행의 코드
2. 왼쪽 키는 조회만 한다 (probe, §8.8)                    → 왼쪽 행의 코드, 없으면 -1
3. 오른쪽 행을 코드별로 모은다 (counting sort, CSR)       → 코드마다 오른쪽 행 목록, 오른쪽 순서
4. 왼쪽 행을 차례로: 맞는 오른쪽 행들 → (l, r) 쌍. 없으면 (left/outer) (l, -1)
5. outer: 맞은 적 없는 오른쪽 행 → (-1, r), 오른쪽 순서로
```
- **행 순서가 반복문의 순서다**: 왼쪽 순서, 한 왼쪽 행 안에서는 오른쪽 순서, outer는 남은 오른쪽 행을 뒤에. right 조인은 양쪽을 바꾼 left 조인이다(왼쪽 코드를 처음 나타난 순서로 다시 번호 매겨 build 쪽으로 쓴다).
- 결과는 (왼쪽 위치, 오른쪽 위치) 두 배열이다. 컬럼은 모두 `take`로 만든다(-1은 NA). `on`으로 합친 키 컬럼은 왼쪽 키를 `take`한 뒤 왼쪽 행이 없는 줄만 오른쪽 키로 채운다(그런 줄은 right와 outer에만 있어 다른 조인은 찾지 않는다).
  - 왼쪽 위치가 정확히 0..n-1이면(모든 왼쪽 행이 한 번씩 차례로: 오른쪽 키가 겹치지 않는 left 조인, 모두 맞는 inner) 바인딩이 None을 돌려주고 왼쪽 컬럼은 take 없이 Copy-on-Write로 공유한다 (FR6).
- 결과 행 수를 먼저 청크마다 세고, 청크마다 자기 자리를 채운다. 코드별 오른쪽 행은 `{시작, 개수}` 한 칸에서 읽고(코드 -1은 개수 0인 칸), 맞는 행이 0이나 1개인 왼쪽 행은 반복문 없이 쓴다. 버리는 행(inner에서 맞지 않는 행)은 다음 청크의 첫 자리가 아니라 태스크의 지역 칸에 쓴다(분기 없이, 경쟁 없이).
- **빠른 길: 정수 키 하나, 오른쪽 키가 겹치지 않음** (`unique_direct_join`, inner/left/outer). 오른쪽 키가 작은 범위의 int64·bool이면(§8.9) 키마다 오른쪽 행을 직접 주소 표에 둔다. 겹치는 키를 만나면 이 길을 버리고 위의 일반 길로 간다. 왼쪽 키를 한 번 훑으며 행마다 맞는 오른쪽 행을 읽는다(범위 밖과 NA는 분기 없이 빈 칸). 코드, CSR, 개수 세기가 없다. 결과는 일반 길과 같다. 1M × 10만 int inner가 1스레드 약 3배, left 약 4배 빠르다 (FR6).
- **여러 정수 키**: 모든 오른쪽 키가 작은 범위이고 범위의 곱이 직접 주소 표의 상한 안이면, 키 튜플을 int64 하나(`(x0 − lo0) × span1 × … + (x1 − lo1) × …`)로 바꾼 뒤 키 하나로 조인한다(`composite_keys`). 오른쪽 범위 밖의 값이 하나라도 있는 왼쪽 행은 -1, NA가 하나라도 있으면 NA다. 튜플이 같을 때만 합친 키가 같으므로 맞는 쌍이 같다. 2키 merge가 약 3배 빠르다 (FR6).
- **같은 키**는 factorize가 정한다(NaN끼리, `0.0`과 `-0.0`). **NA 키는 아무것과도 맞지 않는다**: NA 행의 코드는 -1이다. groupby(`dropna=False`)에서는 NA끼리 같은 그룹이라는 점만 다르다.
- **일반 다중 키**: 컬럼마다 오른쪽에 build한 테이블로 코드를 얻고, §8.6처럼 혼합 기수로 합친다. 2⁶²를 넘을 것 같으면 지금까지의 튜플 코드를 오른쪽에 build한 테이블로 좁히고, 왼쪽은 같은 테이블로 조회한다. 어디서든 -1이면 그 행은 맞지 않는다.
- 키 dtype이 다르면 TypeError다. 키의 해시 규칙이 dtype마다 하나로 남는다.
- **build/probe와 "양쪽을 이어 붙여 함께 factorize"**: 함께 factorize하면 다중 키, NaN, `-0.0`을 factorize가 그대로 처리하지만 양쪽 키를 복사해 이어 붙이고 큰 쪽까지 모두 테이블에 넣는다. Swiss table에서는 넣기와 찾기의 비용이 거의 같아 build/probe로 바꿔도 빨라지지 않았지만(측정 오차 안), 메모리를 덜 쓰고 코드가 단순해서 build/probe를 쓴다.
- `sort=True`는 결과를 키로 안정 정렬한다(§9.3). outer+sort는 pandas와 비슷하다(`bench/frame/speed.md`).

### 9.8 concat
- 같은 dtype 컬럼을 그대로 이어 붙인다. 문자열은 offsets를 다시 계산하고, 비트맵도 이어 붙인다. 고정폭 컬럼의 복사는 부분마다 행 청크로 나눠 풀에서 돈다.
- 같은 이름의 컬럼(Series끼리도)의 dtype이 다르면 먼저 승격한다: bool < int64 < float64 (`_merge.py`의 `_promoted`, `docs/api/frame.md` §9.2). 배열의 `concat`과 같은 규칙이고, dtype만 보고 정하므로 결과 dtype이 데이터에 따라 바뀌지 않는다. int는 `astype`처럼 float64로 반올림한다. string은 string끼리만 잇고, 다른 dtype과 만나면 TypeError다.
- 행 방향만 있다. 컬럼 집합(`join="outer"`/`"inner"`)과 라벨 규칙은 `docs/api/frame.md`에 있다.
- 결과 라벨: 모든 부분이 RangeIndex면 각 range를 결과 라벨 배열에 바로 쓴다(`concat_ranges`). 부분마다 라벨 배열을 만들어 다시 잇던 복사 한 번이 없다. 결과는 여전히 Index다 (FR15).


---

## 10. read_csv (`io/csv.cpp`)

### 10.1 규칙
옵션과 추론 규칙은 `docs/api/frame.md`에 있다. 구조를 정한 원칙은 **"파일에 적힌 것과 다른 값이 나오지 않는다"**이다. 예외 하나는 숫자의 반올림이다: float 컬럼의 int는 list에서 만들 때처럼 float64로 반올림한다 (§10.5, 루트 DESIGN §10.1 3번).
- NA는 **따옴표 없는 빈 필드**와 `na_values`의 텍스트뿐이다. 텍스트 `NaN`은 float의 NaN 값이고, `""`는 빈 문자열이다.
- 공백이 섞인 숫자, `true` 같은 모호한 값은 추측하지 않고 string으로 둔다.
- 형식이 틀린 파일(필드 개수, 따옴표 규칙, 올바르지 않은 UTF-8)은 줄 번호와 함께 ValueError다.
- 줄 끝은 `\n`과 `\r\n`뿐이다. 그 밖의 `\r`은 값의 바이트이고, 닫는 따옴표 뒤의 `\r`은 에러다.
- 숫자 문법은 `astype`과 같은 코드다 (§7.5).

### 10.2 토크나이저: structural index
simdjson/simdcsv의 방식이다. 청크(§10.6)마다 레코드의 필드 끝을 색인한다 (`index_records`).
```
블록 64바이트마다:
  quote = '"'인 바이트의 64비트 마스크       NEON 비교 4번 + 쌍별 덧셈(vpaddq_u8)
  sep, nl = ',', '\n'인 바이트의 마스크
  inside = prefix_xor(quote)                따옴표 안쪽인 바이트 (블록에 따옴표가 없고 바깥이면 건너뜀)
  필드 끝 = (sep | nl) & ~inside            ctz로 하나씩 꺼내 색인에 쓴다: 바이트마다 분기가 없다
```
- **prefix XOR**는 캐리 없는 곱셈 한 번이다: 마스크에 모든 비트가 1인 수를 `vmull_p64`(PMULL)로 곱한다. 이전 블록이 따옴표 안에서 끝났으면 결과를 뒤집는다.
- `""`(값 안의 따옴표)는 따옴표 둘이라 XOR이 상쇄되어 안쪽으로 남는다. 따로 처리하지 않는다.
- **색인**은 필드마다 그 필드를 끝내는 바이트(구분자, 줄바꿈, 입력의 끝)의 위치 하나다. 레코드마다 필드가 정확히 열 수만큼이라 필드 (행 r, 열 i)는 `r × 열 수 + i`번째다. 레코드의 필드 수는 줄바꿈에서 바로 검사한다.
- **따옴표 플래그**: 청크에서 처음 따옴표를 보면 그때부터 필드마다 그 안의 따옴표 수를 함께 적는다(블록의 따옴표 비트 누적 개수, 필드마다 popcount 한 번). 0개면 따옴표 없는 필드, 2개면 처음과 끝 바이트가 따옴표인지만 보면 되는 필드, 그 밖(4개 이상이나 홀수)은 `""`를 확인할 필드다. 따옴표 규칙 검사(`check_quotes`)는 따옴표가 있는 필드만 보고, 값을 만들 때 `""`를 바꾸는 복사도 그 필드만 한다. 필드마다 `memchr`로 따옴표를 찾던 때보다 따옴표 열이 있는 파일이 1스레드 약 1.3배 빠르다 (FR9).
- 맨 앞의 UTF-8 BOM은 header 토크나이저가 건너뛴다(본문은 BOM 다음에서 시작한다).
- 블록의 바이트 최댓값도 함께 모은다. 청크에 0x80 이상의 바이트가 없으면 문자열 값마다 하던 UTF-8 검사를 건너뛴다.
- header는 같은 마스크를 필드마다 꺼내는 토크나이저(`Tokenizer`) 하나로 읽는다.

### 10.3 한 번에 읽기
청크 하나는 한 태스크 안에서 끝난다.
1. 레코드를 색인한다(§10.2). 따옴표가 있으면 따옴표 규칙을 검사한다.
2. 출력 컬럼마다 색인을 따라 그 열의 필드를 차례로 읽는 루프 하나로 컬럼 조각(`Part`)을 만든다(`parse_column`). 필드마다 "이 필드는 어느 열인가"를 고르는 분기가 없고, 한 열의 종류(int, float, bool, text)가 루프 하나를 정한다.
3. 색인을 버린다.
- 색인은 청크 하나만큼이다(필드마다 8바이트, 청크는 많아야 512KB). 모든 필드의 위치를 파일 끝까지 모으는 방식(필드당 16바이트, 짧은 필드에서 원본의 8배)과 달리 메모리는 결과와 실행 중인 청크들의 색인이다 (FR9).
- **필드마다 콜백으로 바로 값을 쓰던 한 패스 방식은 없앴다.** 색인 + 열 단위 루프와 숫자 파서(§10.4)를 합쳐 1스레드 약 1.7~2.5배 빠르다.
- **추론하면서 값을 만든다** (§10.5). 조각은 int로 읽다가 int가 아닌 숫자를 만나면 그 조각의 앞 값들을 double로 바꾸고(가장 가까운 값으로 반올림) float로 계속 읽는다. 첫 값이 `True`나 `False`면 bool(flag 바이트)로 읽는다. 숫자도 bool도 아닌 필드를 만나면 그 조각을 색인에서 처음부터 문자열로 다시 읽는다.
- 조각의 크기는 색인으로 레코드 수를 알고 나서 정확히 잡는다. 문자열 바이트는 열의 필드 바이트 합으로 한 번에 잡고, 16바이트 이하의 필드는 라이브러리 `memcpy` 없이 8바이트 읽기 둘로 옮긴다.
- NA가 없으면 조각은 validity를 들지 않는다. NA 행의 값 자리는 0이다(§4.5). 최종 컬럼에서 NA가 없으면 비트맵이 없고, 있으면 바이트를 8개씩 모아 비트로 쓴다.
- `na_values`는 길이 집합으로 먼저 거른다. 그 길이의 텍스트가 없는 필드는 해시를 계산하지 않고, 있으면 `string_view`로 찾는다(새 `std::string`을 만들지 않는다).
- **줄 번호는 에러가 났을 때만 센다** (`line_at`). 값 에러의 줄은 그 청크를 그 행까지 다시 색인해 찾는다.

### 10.4 숫자 파싱 (`ops/parse.hpp`)
```
"123.456e-7"  →  w = 123456 (정수 가수),  q = -10
1. 흔한 꼴 [+-]?자리[.자리] (지수 없음, 19자리 이하): fast_decimal
   소수부를 8자리 묶음(SWAR)과, 텍스트 끝에서 읽은 마지막 1~7자리 묶음으로
2. 그 밖의 십진수: parse_decimal (자리마다 읽기, 지수, 앞의 0)
3. w × 10^q → double: Eisel-Lemire. 5^q의 128비트 근사값(표)과 w의 128비트 곱 → 가수, 2의 지수는 q·log₂10
   반올림 경계에 너무 가까우면 "모름"
4. "모름", nan/inf, 19자리를 넘으면 std::from_chars
```
- 5^q 표(651개, q = −342..308, `pow5_table.hpp`)는 `tools/gen_pow5_table.py`가 Python 정수로 정확히 계산해 만든다. fast_float의 생성 규칙과 같다. libc++ `from_chars`만 쓰던 때보다 float 하나가 약 2.2배 빠르다.
- **SWAR로 8자리씩**: 8바이트를 uint64 하나로 읽고 정수 곱셈 3번으로 8자리 십진수를 바꾼다.
- **끝에서 읽기**: 텍스트의 끝을 알므로 마지막 1~7자리는 끝에서 8바이트를 읽고 앞쪽 바이트를 `'0'`으로 바꿔(`keep_last`) 한 묶음으로 읽는다. 자리 수로 갈라지는 분기와 자리마다의 검사가 없다. int도 16자리 이하면 끝에서 읽는 묶음 둘이다(`parse_int64`). 텍스트 앞으로 읽는 바이트가 읽을 수 있는 메모리 안이어야 해서 호출하는 쪽이 그 시작(`begin`)을 넘긴다. CSV는 입력 버퍼다. `""`를 바꾼 복사본은 넘기지 않는다 (FR10).
- **흔한 십진 꼴(`fast_decimal`: 부호, 숫자, 소수점만 있고 유효 자리가 19개 이하)은 Clinger의 빠른 길(w ≤ 2⁵³이고 |q| ≤ 22이면 IEEE 곱셈이나 나눗셈 한 번)을 쓰지 않는다.** 그 꼴의 값은 모두 Eisel-Lemire로 간다. Eisel-Lemire는 그 정확한 경우도 올바르게 반올림하고, 한 열에 16자리와 17자리 float가 섞이면 "빠른 길인가"의 분기가 예측되지 않았다. float 하나의 파싱이 약 20% 빨라졌다 (FR10).
  - 그 꼴이 아닌 텍스트(지수가 있는 `1e5`, 앞의 0 때문에 20자리를 넘는 수 등)는 일반 파서(`parse_decimal`)로 가고, 거기서는 Clinger의 빠른 길을 먼저 보고, 해당하지 않으면 Eisel-Lemire로 간다.
- **문법 검사와 파싱이 한 패스다** (`parse_double`, int는 `parse_int64`).
- 정답은 Python `float(text)`다: 무작위 double의 `repr`, 자릿수가 많은 수, 반올림 경계의 어려운 수(2⁻¹⁰⁷⁴ 근처, 최대값 근처, 정확히 중간인 수)에서 비트까지 같다. 바꾼 뒤에도 무작위 텍스트 2천만 개를 `from_chars`와 비교해 차이가 없다.

### 10.5 타입 추론
- 컬럼마다 int64 → float64 → bool → string 중 첫 dtype이다. 청크의 조각은 Int, Float, Bool, Text 중 하나이고(§10.3), 열 전체로 합친다(§10.6).
  - int64: 모두 int 문법이고 int64 범위 안
  - float64: 모두 float 문법이다. int 문법도 float 문법이다
  - bool: 모두 `True` 또는 `False`
- **float 컬럼의 int는 list 추론처럼 float64로 반올림한다** (루트 DESIGN §10.1 3번, M8). 한 열에 `1.5`와 `9007199254740993`(2⁵³ + 1)이 있으면 둘째 값은 `9007199254740992.0`이다. pandas, numpy, `Series([1.5, 2**53 + 1])`과 같다. 조각이 float가 되기 전의 int는 `parse_int64` 뒤에 `double`로 바꾸고(가장 가까운 값, 짝수 쪽), float가 된 뒤의 필드는 int 문법이어도 `parse_double`이 바로 읽는다(int를 먼저 시도하는 두 번째 파싱이 없다). 둘 다 올바르게 반올림하므로 Python의 `float(int(text))`와 비트까지 같다. 하나 다른 곳은 `-0`이다: int로는 0이라 `+0.0`이어야 하므로 `parse_double`의 `-0.0`을 고친다.
  - M7까지는 정확히 표현되지 않는 int가 있으면 string으로 두었다(`int_is_exact_double`). list는 반올림하고 CSV는 string으로 남기는 차이를 없앴다. 값을 그대로 지키려면 `dtype="string"`으로 고른다.
  - list는 int만 있고 int64 밖의 int가 있으면 OverflowError지만, CSV의 추론은 에러를 내지 않고 다음 dtype으로 넘어가므로 float64다.
- 모두 NA인 열은 float64다 (list 추론과 같다).

### 10.6 파일을 청크로
CSV는 아무 데서나 자를 수 없다. 따옴표 안의 줄바꿈은 레코드 경계가 아니다. 그런데 **"따옴표 안"은 그 앞 따옴표 개수의 홀짝이다** (§10.2).
```
1. 청크마다 따옴표 개수의 홀짝 (NEON: 16바이트씩 레인별 XOR)   병렬
   → 누적 XOR = 각 청크 시작이 따옴표 안인가          (정확, 추측 없음)
   → 청크의 첫 레코드 = 그 상태에서 따옴표 밖 줄바꿈 다음
2. 청크는 자기 안에서 시작하는 레코드를 색인하고 열마다 파싱한다   병렬
   (마지막 레코드는 경계를 넘어 끝까지, §10.3)
3. 컬럼마다 청크들의 종류를 합치고 조각을 최종 버퍼로 옮긴다       병렬
```
- simdcsv는 경계의 따옴표 상태를 **추측**하고 틀리면 다시 읽는다. 여기서는 Polars처럼 홀짝을 먼저 세므로 추측이 없다.
- **청크 크기**: 풀을 쓸 수 있으면 본문의 1/32을 64KB~512KB로 자른 것, 아니면 512KB다. 결과가 자르는 방식과 무관하므로 파일 크기와 스레드로 정한다 (§11.5). 몇 MB 파일도 스레드마다 여러 청크를 받는다(512KB 고정이면 2MB 파일이 청크 넷이었다).
- **경로는 하나다.** 1스레드와 `nrows`도 같은 청크 경로를 차례로 돈다. 한 토크나이저가 처음부터 순서대로 읽던 직렬 경로는 없앴다 (FR9). 홀짝 세기는 NEON이라 1스레드에서도 싸다.
- **종류 합치기**: 값이 있는 조각의 종류로 정한다. text가 없고 bool도 없으면 숫자(float가 있으면 int 조각의 값을 double로 반올림해 바꾼다: 2⁵³ + 1은 2⁵³이 된다), text와 숫자가 없으면 bool, 그 밖은 string이고 text가 아닌 조각은 그 청크만 다시 색인해 문자열로 읽는다. §10.5의 규칙을 열 전체에 적용한 것과 같다.
- **에러 순서는 직렬과 같다**: 가장 앞 청크의 토큰화 에러 → 열 순서 → 열 안에서 가장 앞 행 (§11.6). 청크 안에서는 직렬 토크나이저가 만나는 순서다: 필드 수 에러나 입력 끝의 열린 따옴표를 만나면, 그 앞(그 레코드의 필드 포함)의 따옴표 규칙을 먼저 검사한다.
- **컬럼은 하나씩 만든다.** 청크들의 조각을 최종 버퍼로 옮기고 그 조각을 바로 버린다. 메모리는 결과와 아직 만들지 않은 컬럼의 조각이다.
- **파일 읽기**: 정규 파일은 Python이 연 파일의 fd로 C++가 4MB 조각을 `pread`로 병렬로 읽는다(`read_fd`, `read_csv_fd`). Python의 `f.read()`(직렬 복사 한 번)가 10스레드 시간의 약 1/4이었다. 읽다가 실패하거나 정규 파일이 아니면 Python이 `f.read()`로 읽는다(그 OSError가 그대로 나온다). 연 뒤 커진 파일은 끝까지, 줄어든 파일은 줄어든 끝까지 읽는다.
- **`nrows`**: Python 쪽이 파일을 64KB부터 두 배씩 자라는 블록으로 읽고, 블록마다 한 번 따옴표 밖 줄바꿈을 센다(`csv_scan_records`, NEON 마스크와 popcount, 블록 사이로 따옴표 상태를 넘긴다). `nrows`개가 끝나는 곳에서 자른 바이트만 C++에 넘기므로 C++는 그것을 모두, 병렬로 읽는다. C++의 `nrows`(레코드를 순서대로 그 수만큼)는 청크를 차례로 읽는 길로 남아 있다.
- 태스크의 `std::bad_alloc`은 파일의 에러로 바꾸지 않고 그대로 던진다(MemoryError).
- 파싱은 Python 객체를 만지지 않으므로 GIL을 놓고 돈다.
- 10스레드에서 pandas의 pyarrow 엔진보다 빠르다(1M 행 mixed 약 2배). 1스레드는 여러 스레드로 도는 pyarrow 엔진에 진다(§12.3). 숫자는 `bench/frame/speed.md`.


---

## 11. 병렬화 (`frame/parallel.hpp`)

### 11.1 공용 풀과 frame의 정책
- 풀은 모든 영역이 함께 쓰는 `sd::parallel` 하나다 (`parallel/pool.hpp`, array.md §10.1). frame에 필요한 의미가 모두 들어 있다.
  - `parallel_for`는 태스크가 둘 이상이면 스레드 수와 상관없이 GIL을 놓는다. 태스크가 하나이면 GIL을 그대로 두고 그 자리에서 돈다 (array.md §10.2). frame은 태스크가 하나여도 놓도록 `run_unlocked`로 감싸서 부른다 (§11.4).
  - `parallel::Unlocked(bool)`은 직렬 구간에서 GIL을 놓는 공용 도구다. frame의 `Unlocked(work)`는 여기에 frame의 기준값을 붙인 것이다.
  - `available()`은 2스레드 이상, 태스크 밖, 풀이 다른 Python 스레드의 일로 바쁘지 않을 때 참이다. 바쁘면 직렬 경로를 고른다(같은 결과, 나눠 놓고 직렬로 도는 것보다 빠름).
  - `FirstRow` (§11.6)
- **언제, 어떻게 나눌지는 frame의 정책이다** (`frame/parallel.hpp`, `sd::frame`).

  | 이름 | 값 | 뜻 |
  |---|---|---|
  | `kParallelThreshold` | 2¹⁷ (131,072) | 이보다 작은 일은 호출 스레드에서 돈다 (§11.2) |
  | `kChunkRows` | 2¹⁴ (16,384)행 | `for_chunks`의 청크 (§11.3) |
  | `num_chunks(n)` | ⌈n / kChunkRows⌉ | 청크별 부분 결과의 개수 |
  | `for_chunks(n, f, cost)` | | `f(lo, hi)`를 청크마다. `n × cost`가 기준값 아래면 차례로, 아니면 `run_unlocked` |
  | `run_unlocked(ntasks, fn, work)` | | GIL을 놓고 `parallel_tasks`를 부른다 (§11.4). `work`(원소 × cost)로 태스크 길이를 어림한다 |
  | `parallel_tasks(ntasks, fn, task_ns)` | | `parallel::parallel_for(ntasks, fn, task_ns)`: 태스크 하나의 길이(ns) 어림을 풀에 넘긴다 |
  | `kWorkPerNs` | 4 | 일 단위(단순 패스의 원소 하나) 4개가 1ns라는 어림 |
  | `Unlocked(work)` | | 기준값 이상인 직렬 구간에서 GIL을 놓는다. `parallel::Unlocked(work >= kParallelThreshold)`다 (§11.4) |

- **기준값의 이름은 영역마다 같고 값은 다르다.** 배열은 `sd::kParallelThreshold` = 2¹⁵(array.md §10.4), frame은 `sd::frame::kParallelThreshold` = 2¹⁷이다. 각 영역이 자기 커널로 쟀다 (루트 ARCHITECTURE §4.1).
- `cost`는 행마다 한 번의 덧셈보다 일이 몇 배인 커널이 기준값을 옮길 때 쓴다. 청크는 바꾸지 않는다.
- **자는 일꾼을 깨울지는 풀이 정한다** (`parallel/pool.hpp`, array.md §10.1): 태스크 수 × 태스크 길이가 `kWakeWorkNs`(20µs) 이상이면 바로 깨운다. frame은 길이를 어림해 넘긴다: `for_chunks`와 `run_unlocked`는 `work / kWorkPerNs / ntasks`(f + g가 2¹⁸행에서 원소 하나에 약 0.19ns), groupby의 그룹 태스크는 행 × 4, CSV는 청크 바이트 × 1ns(파싱)·0.05ns(홀짝)·0.1ns(파일 조각 읽기)다. 어림이 없는 호출(CSV 컬럼 만들기)은 풀이 첫 태스크들을 재서 정한다. 어림은 속도만 바꾼다 (FR11).
- 스레드 수 API는 `sd.set_num_threads`, `sd.get_num_threads` 하나다. `sd.frame`에는 없다.

### 11.2 기준값 2¹⁷
- 배열의 2¹⁵로 시작해 1스레드와 10스레드를 크기별로 쟀다.
  - 2¹⁶(64k)에서도 10스레드가 `f + g`, `sum`, `isna`에서 졌다.
  - 2¹⁷(128k)부터 이겼다: `f + g`, `sum`, 필터가 약 2배.
- 청크가 16K행이라 기준값 아래에서는 태스크가 몇 개뿐이고, 풀을 깨우는 고정비(수 µs)를 이기지 못한다.
- 상수 하나로 둔다. 연산마다 다르게 두지 않고, 행마다 일이 큰 커널만 `cost`로 옮긴다(문자열 factorize 4, §8.7). 2026-11-22에 다시 보고 그대로 두었다: 기준값 근처의 10만~2¹⁸행 사례가 깨우기 규칙(§11.1)이 고쳐진 뒤 모두 나뉘고, 기준값 아래의 단순 패스는 여전히 1스레드가 빠르다 (FR17).
- 기준값 아래의 일이 여럿 모이는 곳은 일을 합쳐 나눈다: 넓은 프레임의 필터(§7.3), 리덕션(§7.4), groupby(§9.1).

### 11.3 청크 16K행
- **64의 배수다.** 두 태스크가 비트맵의 같은 바이트(또는 64비트 워드)에 쓰지 않는다. 그래서 각 청크가 자기 비트맵 바이트를 경쟁 없이 쓴다.
- **청크는 n으로만 정해진다.** 청크별 결과를 청크 순서로 합치면 스레드 수와 무관하다.
- 청크 순서로 합치는 커널: 원소별 연산, 필터와 압축(청크마다 개수 → 시작 위치), 문자열 결과(청크마다 바이트 수 → 시작 위치), 정수 합, min/max, `valid_bytes`, 기수 정렬의 키 훑기
- 위치를 청크로 나누는 커널(필터의 모으기, merge의 빠른 길, 직접 factorize)은 청크가 자기 출력 자리에만 쓴다. 버리는 쓰기는 태스크의 지역 칸에 한다 (§9.7).
- `finish_fixed`처럼 청크마다 비트맵을 바이트로 펼치는 커널은 16K 바이트 스택 버퍼 하나를 쓴다.
- groupby는 그룹을 약 16K행씩 묶는다 (§9.1).

### 11.4 GIL
- **일이 기준값 이상이면 스레드 수와 상관없이 GIL을 놓는다.** `set_num_threads(1)`이어도 다른 Python 스레드가 돈다 (`docs/api/frame.md`).
  - `for_chunks`는 기준값 이상이면 `run_unlocked`를 부른다. `run_unlocked`는 먼저 GIL을 놓고(`parallel::Unlocked`) `parallel_for`를 부른다. `parallel_for`만 부르면 태스크가 하나일 때 GIL을 쥔 채 돌기 때문이다(array.md §10.2). 기준값 이상에서만 태스크를 나누는 float 합의 잎(`pairwise_parallel`), groupby의 그룹 태스크(`for_groups`), 컬럼 단위 태스크(`reduce_columns`, `filter_columns`)도 `run_unlocked`를 쓴다.
  - 이미 GIL을 놓은 곳(태스크 안, `Unlocked` 아래)에서는 `GilRelease`가 아무 일도 하지 않는다. 그런 곳의 `parallel_for`(병렬 factorize와 counting sort의 파티션 단계, CSV의 청크)는 그대로 부른다.
  - 1스레드에서 더 드는 비용은 `run_unlocked`마다 GIL을 한 번 놓고 잡는 것이다. 131,072행부터 1M 행까지 여덟 연산을 전후 빌드를 번갈아 쟀을 때 기하평균 +0.6%로 측정 잡음 안이었다.
  - 나누지 않는 직렬 구간(직렬 factorize, 직렬 기수 정렬과 counting sort, merge의 번호 매기기, CSV 파싱, 대량 복사)은 `Unlocked(work)`로 놓는다. 대량 복사는 numpy와 `sd.ndarray`에서 컬럼 만들기(`from_numpy`의 넓히기, `from_array`), `columns_to_array`, NA 없는 `isna`, 스칼라 채우기(`full`), 다중 키 merge의 압축, 그룹 순서의 순위 매기기다. NDArray를 만들고 없애는 일은 GIL 구간에 남긴다(그 `base`가 Python 참조다).
  - 기준값 아래면 GIL을 쥔 채 돈다. 작은 호출에 GIL을 놓고 잡는 비용이 더해지지 않는다.
- 다른 영역도 같은 규칙을 쓴다. 기준값만 영역마다 다르다 (array.md §10.2, API.md §3).
- **태스크는 Python 객체를 만지지 않는다.** 에러 메시지(값의 `repr`이 들어간다)는 병렬 구간이 끝나고 GIL을 다시 잡은 뒤 만든다 (§11.6).
- **GIL을 쥐는 곳**: `set_rows`(§5.3), Index 엔진의 CSR을 만들 때(§6.2).

### 11.5 결정성 규칙
**같은 입력은 같은 스레드 수에서 실행마다 비트까지 같다.** 이것이 요구다(2026-11-22부터 스레드 수와 무관한 비트는 요구가 아니다). frame은 지금 모든 결과(float 합, 행 순서, factorize 코드, 에러 메시지)가 스레드 수와도 무관하다. 스레드 수에 따라 비트가 달라지는 길은 만들지 않았다 (FR16).
- 기본 규칙: 비트를 바꿀 수 있는 나누기(float 합)는 데이터 길이(와 dtype)로만 정한다. 부분 결과는 고정된 순서로 합친다. 1스레드도 같은 구조로 계산한다. 원자 연산의 순서에 기대는 합은 없다.
- **결과가 나누는 방식과 무관하게 정확히 정해지는 연산은 나누는 방식이 스레드 수와 데이터를 따른다.** 1스레드(또는 태스크 안, 풀이 바쁠 때)는 직렬 알고리즘을 그대로 쓴다.
  - 대상: factorize(처음 나타난 순서의 코드, §8.7, §8.9), 안정 counting sort(§9.6), 기수 정렬(§9.5), CSV 읽기(§10.6), merge의 빠른 길(§9.7), 필터의 압축과 모으기(§7.3)
  - 이유: 나누면 패스가 늘어 1스레드가 1.5~2.5배 느려졌다. 규칙의 목적은 같은 결과이고, 이 연산들은 결과가 같다.
  - 직접 주소 factorize의 원자 최솟값(§8.9)은 쓰는 순서와 상관없이 같은 값이다. 합이 아니다.
- **분기하는 계산의 동률**: 정렬의 같은 값은 안정 정렬이라 원래 순서로 정해진다. 어느 정렬 길(이미 정렬됨, 역순 묶음, 바이트 자리, 넓은 자리, 나눔)이든 같은 순서다. groupby의 min/max는 같으면 앞의 행이다(그룹 안의 행 순서는 스레드 수와 무관하다).

| 연산 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 원소별, astype, 필터, take | 16K행 청크 (모으기는 출력 위치의 청크) | 행마다 따로 계산한다. 출력 위치는 청크 순서의 누적합 |
| 정수 합, min/max | 청크, 청크 순서로 접기 | 결합법칙. IEEE min/max도 결합적이다 |
| float 합, mean, var, std | pairwise 트리의 잎(2¹⁴) | 같은 트리 순서로 합친다 (§7.4) |
| groupby 리덕션 | 그룹 묶음, 또는 값 컬럼 | 그룹마다, 컬럼마다 독립이다 |
| factorize | 해시 파티션 (P = 256 또는 1), 직접 표 (원자 최솟값 + 비트맵 순위, 또는 직렬) | 코드는 처음 나타난 순서다 (§8.7, §8.9) |
| 기수 정렬, counting sort | 청크, 버킷 | 원소의 자리가 직렬과 정확히 같다 (§9.5, §9.6) |
| merge | probe 행 청크 | 청크마다 자기 출력 자리 |
| read_csv | 64KB~512KB 바이트 청크 | 따옴표 홀짝은 정확하다. 종류는 열 전체 규칙으로 합친다 (§10.6) |
| DataFrame 리덕션, `df[mask]` | 컬럼 (또는 컬럼 × 청크) | 컬럼마다 독립이다 |
| 에러 | 청크마다 첫 실패 | 가장 앞의 행이 이긴다 (§11.6). 컬럼 단위 태스크는 태스크 전에 컬럼 순서로 검사한다 |

- 테스트는 1, 3, 기본 스레드 수에서 같은 결과(값의 비트, 행 순서, 에러 메시지)가 나오는지 비교한다.

### 11.6 에러 위치 (`FirstRow`)
- string 컬럼의 `astype("int64")`에서 1M 행 중 세 곳에 숫자가 아닌 값이 있으면, 어느 청크가 먼저 실패할지는 스레드 일정에 달렸다.
- 청크마다 자기의 첫 실패 행만 `FirstRow`에 내고, 가장 작은 행이 이긴다. 앞에서 이미 실패했으면 뒤 청크는 일찍 멈춘다(`failed_before`).
- 메시지는 병렬 구간이 끝나고 GIL을 다시 잡은 뒤 만든다.
- CSV는 여기에 "토큰화 에러가 값 에러보다 먼저, 열 순서로"를 더한다 (§10.6).

### 11.7 확장성
- 1M 행에서 1스레드 대비 10스레드 배율은 `bench/frame/speed.md`의 생성 표에 있다(`t10x`).
- 원소별 연산과 리덕션은 메모리 대역폭에 가깝다(약 4~6배). float + float 1M은 24MB를 읽고 쓴다.
- 키가 적은 해시(1,000종)는 직렬도 테이블이 L1에 들어가서 약 2배다. 키가 많은 작은 범위의 정수 factorize는 직렬이다(§8.9).
- CSV, 정렬, 필터는 약 5~7배다.


---

## 12. 작은 데이터

### 12.1 같은 입력 반복의 함정
- 같은 입력으로 수천 번 부르고 최솟값을 쓰면 **분기 예측기가 그 입력을 외운다.** M4 Pro의 예측기는 1천 행 입력의 "이 칸이 비었나, 키가 같나" 분기 1천 개를 통째로 기억한다. 분기가 많은 코드일수록 실제보다 빨라 보인다.
- 키 1,000종 테이블만 따로 잰 값 (ns/행):

  | 행 수 | linear probing, 같은 입력 | linear probing, 새 입력 | Swiss, 같은 입력 | Swiss, 새 입력 |
  |---:|---:|---:|---:|---:|
  | 1,000 | 3.65 | **11.8** | 6.49 | 8.50 |
  | 10,000 | 1.25 | 5.73 | 2.45 | 3.67 |
  | 1,000,000 | 4.11 | 4.35 | 1.84 | 1.86 |

  - 같은 입력이면 linear probing이 이기고, 새 입력이면 1천 행부터 Swiss table이 이긴다. 1M 행은 외울 수 없어서 차이가 없다.
  - Swiss table은 바로 그 예측 실패를 없애려고 고른 것이다 (§8.3). 같은 입력 반복은 그 실패를 측정에서 지운다.
- 그래서 측정은 **새 입력** 모드를 함께 쓴다 (§13.2).

### 12.2 작은 데이터에서 고친 것
새 입력으로 재면 작은 데이터의 느려짐 중 해시(`nunique`, groupby의 일부)는 측정이 만든 것이었고, 나머지는 진짜였다. 고친 것:
- **Swiss table을 처음부터 잡는다** (§8.5)
- **나누지 않는 기수 정렬은 한 번 읽기 히스토그램** (§9.5)
- **조회 루프의 dtype과 NA를 호출마다 한 번만** (§8.8)
- **작은 범위의 정수는 직접 주소 표** (§8.9): 1천 행의 int `nunique`가 Swiss table보다 약 3배 빠르다.
- 결과는 모두 비트까지 같다.

### 12.3 남은 것
- **read_csv의 1스레드**: 비교 대상인 pandas의 pyarrow 엔진은 여러 스레드로 돌아서 1스레드 sd가 1M 행에서 0.25~0.4배다(10스레드 sd는 약 2배 빠르다). 남은 1스레드 시간의 약 1/3이 float 파싱(자리 읽기 약 60%, Eisel-Lemire 약 30%), 약 1/5이 색인이다. NEON 16자리 묶음으로 자리를 읽는 파서는 하지 않았다: 자리 읽기가 이미 SWAR이고 어림한 이득(float 하나에 약 1ns, 파일 전체 수 %)이 불확실하다.
- **문자열 factorize의 1스레드**: pandas(pyarrow 문자열)의 0.7~0.9배다. 시간 대부분이 키 비교(`memcmp`)와 메모리 접근이다. 10스레드는 약 2배 빠르다.
- **키가 많은 작은 범위 정수의 병렬 factorize**: 직렬 한 번(§8.9)보다 빠른 병렬 판이 없다. 키 범위로 나눈 파티션에 (키, 행)을 흩뿌리고 파티션마다 작은 직접 표를 쓰는 판이 후보다(흩뿌리기와 번호 매기기가 §8.7처럼 남아 10스레드 이득은 어림으로 2배 안팎).
- 10행 마스크 필터는 컬럼을 한꺼번에 거르는 길(§7.3)에서 `nonzero`를 먼저 만들어 약 10% 늘었다.
- Polars와의 비교는 이번 변경 뒤에 다시 재지 않았다 (§13.3).


---

## 13. 테스트와 측정

### 13.1 테스트 (`tests/frame/`)
- pytest + hypothesis. 예제 수는 환경 변수로 늘린다(기본 200~300, 긴 실행 2,000).
- **정답(oracle)**
  - pandas 3.1.1. `docs/api/frame.md`의 "pandas와 다름" 항목은 변환 규칙을 따로 두고 비교한다.
  - 규칙을 Python으로 그대로 옮긴 참조 구현: factorize(Python dict), merge(행 순서 규칙의 이중 반복문), read_csv(옵션과 추론 규칙)
  - Python 자체: 정렬은 `sorted()`(`reverse=True`에서도 안정), `str(float)`, `float(text)`
  - 배열 영역: float 합과 mean/var/std, int 평균은 같은 배열의 `sd.sum`, `sd.mean`, `sd.var`, `sd.std`와 **비트 단위로** 같아야 한다. int–float 비교와 float → int 변환은 `sd.equal`, `sd.astype`의 결과와 같아야 한다 (`tests/interop/`, `tests/frame/test_reduce.py`)
  - numpy, pandas: 합과 mean/var/std는 반올림 오차의 허용 범위 안에서 비교한다. min, max, `-0.0`과 `0.0`, NaN 같은 반올림이 아닌 동작은 비트로 비교한다.
  - groupby: 그 그룹의 행만 고른 Series의 리덕션 (§9.1)
- repr은 pandas 출력과 글자 단위로 같다(행 0~100, 열 0~23, 극단적 float, 탭과 줄바꿈이 든 문자열).
- **스레드**: 1, 3, 기본 스레드 수에서 같은 결과. CSV는 여러 청크에 걸친 파일(따옴표 안 줄바꿈, `""`, CRLF, 한글, 여러 청크에 걸친 1.5MB 따옴표 필드, 앞뒤의 에러 두 개)을 직렬 정답과 비교한다. 청크 읽기를 바꿀 때는 아주 작은 청크(수십 바이트)로 빌드한 읽기와 옛 직렬 토크나이저를 무작위 파일(에러, 따옴표, 지정 dtype, `nrows` 포함)에서 비교했다.
- **스냅샷**: §5.3의 동시 대입 테스트
- Copy-on-Write가 정말 그 자리에서 쓰는지(`to_numpy()` 주소)도 본다 (§5.2).
- **배열과의 경계** (`tests/interop/test_frame_array.py`): §5.4의 공유, 복사, 읽기 전용 뷰, 뷰가 살아 있을 때의 Copy-on-Write.
- 공개 이름과 시그니처는 `test_api_surface.py`가 `docs/api/frame.md`와 맞춘다 (루트 DESIGN §5.3).

### 13.2 측정 방법
- 호출당 최소 시간: 수 ms 걸리도록 묶어 7번 재고 최솟값을 쓴다.
- **새 입력 모드**: 크기마다 시드가 다른 입력을 여러 개(1천 행 64개, 10만 행 40개, 1M 행 4개) 만들고 부를 때마다 다음 것을 쓴다 (§12.1). 1M 행의 원소별 연산은 입력 4개가 캐시에 다 들어가지 않아 같은 입력보다 느리다(float + float 10스레드 36 → 80µs). 이 숫자가 실제에 더 가깝다.
- 두 모드를 같이 보면 어느 코드가 분기에 기대는지 보인다.
- **비교 대상과 번갈아 재기**: 우리, pandas, Polars를 같은 스크립트로 두 바퀴 번갈아 돌리고 줄마다 빠른 쪽을 쓴다. 기계 상태의 변화가 모든 열에 고르게 들어간다.
  - 다른 프로세스의 벤치마크가 함께 돌아 모든 값이 1.5~2배로 나온 일이 있었다. 시작 전에 기기가 조용한지 확인한다.
  - pandas 정렬은 `kind="stable"`, Polars groupby는 결과를 키로 정렬, Polars join은 `maintain_order="left"`로 맞춘다.
  - Polars 스레드 수는 `POLARS_MAX_THREADS`로 정하므로 열마다 프로세스를 따로 띄운다. Polars 기본은 14스레드(E코어 포함), 우리 기본은 P코어 10개다.
- 크기: 10행, 1천, 10만, 1M 행

### 13.3 위치
- 경우마다의 시간과 pandas 대비 비율, 10스레드 배율은 `bench/frame/speed.md`(생성 표)에 있다. 이 문서는 숫자를 옮겨 적지 않는다.
- 2026-11-22의 전체 측정에서 1스레드 sd가 pandas의 0.9배에 못 미치는 경우는 §12.3의 것들(여러 스레드 pyarrow 엔진과 견준 read_csv, 문자열 `nunique`)이다. 10행과 1천 행은 호출 비용이 결정하고(§3), 모든 경우에 pandas보다 빠르다.
- 우리는 기준값 아래에서 스레드를 쓰지 않는다(§11.2). Polars는 스레드를 켜면 작은 데이터에서 오히려 느려졌다(옛 측정, 1천 행: Polars@1이 @14보다 2배 빠르다).
- Polars와의 기하평균(옛 측정: 10만 행에서 Polars@14가 우리@10보다 빨랐다)은 §8.9, §9.7, §10의 변경 뒤에 다시 재지 않았다.

---

## 14. 결정 목록 (2026-11-22 성능 작업)
2026-11-22의 우선순위(성능, 그다음 정확도; numpy·pandas와의 비트 일치는 목표가 아님; 스레드 수와 무관한 비트는 요구가 아니고 같은 스레드 수의 실행마다 같은 비트는 요구)로 다시 정하거나 다시 보고 둔 것이다. 정확도: 아래 모든 항목에서 결과의 비트가 바뀌지 않았다(골든 frame 2,834개 그대로). frame에는 Arb 정확도 계약이 없어 계약의 상태는 해당하지 않는다.

| 항목 | 내용 |
|---|---|
| FR1 기수 정렬의 패스 | 키를 한 번 훑어 정렬됨·역순(같은 값 묶음째 뒤집기)·필요한 비트를 정한다. `키 − 최솟값`의 비트나 서로 다른 바이트만 정렬하고, 나누지 않는 정렬은 11비트 이하의 자리를 쓴다(§9.3). 바꾼 것: "정한 바이트 자리 8개(또는 키 종류의 상한)를 모두 지난다, 상수 자리 건너뛰기는 데이터를 보는 길이라 하지 않는다"(옛 §9.3, §1.3). 이유: 0..10⁶의 int64가 8패스에서 2~3패스, 무작위 int64 1M이 약 2.7배; 상수 자리의 패스는 행을 옮기지 않으므로 결과가 같다. 정확도: 같은 안정 정렬, 비트·순서 그대로 (추천안, 사용자 위임 2026-11-22) |
| FR2 이미 정렬된 키 하나 | 키 하나, NA 없음, 이미 그 방향이면 `argsort`가 None을 돌려주고 컬럼과 Index를 Copy-on-Write로 공유한다; RangeIndex는 take의 Index로 바꾼다(§9.3, §2.4). 바꾼 것: 언제나 정렬하고 take. 이유: 정렬된 1M int64가 약 70배. 정확도: 값·순서·결과 타입 그대로 (추천안, 사용자 위임 2026-11-22) |
| FR3 기수 정렬을 나누는 기준 | 키가 기준값(2¹⁷) 이상일 때(§9.5). 바꾼 것: 흩뿌리기 일 `n × 2`가 기준값 이상(65,536키부터). 이유: 그 사이에서는 직렬 넓은 자리가 약 1.3배 빠르다. 정확도: 같은 정렬 (추천안, 사용자 위임 2026-11-22) |
| FR4 작은 범위 정수의 factorize | 범위 ≤ min(max(2n, 2¹⁶), 2²⁸)인 int64·bool은 직접 주소 표; 키가 많으면 분기 없는 직렬 한 번, 적으면 원자 최솟값과 비트맵 순위의 병렬 네 패스(§8.9). 바꾼 것: 모든 키에 Swiss table과 해시 분할(§8.3, §8.7). 이유: `nunique`·`duplicated`가 약 3~5배, 1천 행도 약 3배. 키가 많을 때의 병렬 판은 10스레드가 직렬보다 약 4배 느려 쓰지 않는다. 정확도: 같은 코드(처음 나타난 순서) (추천안, 사용자 위임 2026-11-22) |
| FR5 `unique_values`의 직접 경로 | 작은 범위의 int64·bool은 비트맵에 나온 키를 표시해 차례로 읽는다(§8.9). 바꾼 것: factorize + take + 그룹 정렬. 이유: 약 5배(키 63만 종; 옛 Swiss 경로 전체 대비 약 24배). 정확도: 같은 값, NA 하나 맨 뒤 (추천안, 사용자 위임 2026-11-22) |
| FR6 merge의 빠른 길 | 정수 키 하나에 오른쪽 키가 겹치지 않으면 직접 주소 표 한 번 훑기(inner, left, outer); 작은 범위의 정수 키 여럿은 키 하나로 합침; 왼쪽 위치가 0..n-1이면 None을 돌려주고 왼쪽 컬럼을 공유; inner·left는 왼쪽 행이 없는 줄을 찾지 않음(§9.7). 바꾼 것: 언제나 build/probe, 코드, CSR, 모든 컬럼의 take. 이유: 1M × 10만 inner 약 3배, left 약 4배, 2키 약 3배. 정확도: 같은 행, 같은 순서 (추천안, 사용자 위임 2026-11-22) |
| FR7 groupby의 행 배치와 그룹 합 | 코드가 0, 0, 1, …로만 오르면 그룹을 구간으로 두고 행을 옮기지 않는다; 리덕션은 dtype과 행 위치를 템플릿으로 고른다(§9.1). 그룹 합은 pairwise로 두었다: 그룹마다 차례로 더하기가 허용되었지만 CSR이 여러 집계에 함께 쓰이고 뚜렷이 빠르지 않아, 그룹 합 = 그 행의 `Series.sum()`의 비트 일치를 지킨다. 바꾼 것: 언제나 counting sort, 행마다 dtype을 고르는 리덕션. 이유: 키로 정렬된 1M 행의 `sum` 약 4배, `std`·`max` 약 1.5배. 정확도: 비트 그대로 (추천안, 사용자 위임 2026-11-22) |
| FR8 그룹 순서: 숫자 키 하나 | 대표 값의 안정 기수 정렬(`argsort`)이 곧 그룹 순서다(§9.2). 바꾼 것: 고유값 정렬 → 순위 → counting sort. 이유: 약 3배. 정확도: 같은 순서 (추천안, 사용자 위임 2026-11-22) |
| FR9 read_csv의 구조 | 청크마다 필드 끝을 색인한 뒤 열마다 루프 하나로 파싱; 따옴표 수 플래그; Bool 조각; 청크는 64KB~512KB; 1스레드와 `nrows`도 같은 청크 경로; `nrows`는 Python이 레코드까지 잘라 넘김; 정규 파일은 C++가 병렬 `pread`로 읽음(§10.2, §10.3, §10.6). 바꾼 것: 필드마다 콜백으로 값을 바로 쓰는 한 패스, 512KB 고정 청크, 1스레드·`nrows`의 직렬 토크나이저, Python `f.read()`. 이유: 1스레드 약 1.7~2.5배, 10스레드 약 2.3배, `nrows` 약 3.5배(1스레드)·13배(10스레드). 메모리는 결과 + 청크 하나만큼의 색인이다. 정확도: 같은 값·에러·에러 순서(옛 읽기와 무작위 파일 비교) (추천안, 사용자 위임 2026-11-22) |
| FR10 숫자 파서 | 흔한 십진 꼴은 끝에서 읽는 SWAR 묶음(`fast_decimal`), 16자리 이하 int는 끝에서 읽는 묶음 둘; 흔한 꼴의 값은 모두 Eisel-Lemire(Clinger의 빠른 길 없음; 그 밖의 텍스트를 받는 `parse_decimal`에는 있다)(§10.4). 바꾼 것: 자리마다 끝을 검사하는 읽기, Clinger 먼저. 이유: float 하나 약 1.6배, int 약 1.2배. 정확도: Python `float(text)`와 비트 일치 그대로(무작위 텍스트 2천만 개, `from_chars`와 차이 0) (추천안, 사용자 위임 2026-11-22) |
| FR11 풀 깨우기 | `parallel_for(ntasks, fn, task_ns)`에 태스크 길이 어림을 넘긴다(`kWorkPerNs` = 4, CSV는 바이트 기준)(§11.1). 바꾼 것: 16태스크 이하의 일을 빈 태스크로 17개까지 채우는 우회. 이유: 풀이 길이 어림을 받게 되어 우회가 필요 없다; 10만~2¹⁸행 사례가 10스레드에서 같거나 빠르다(차이는 측정 잡음 안). 정확도: 해당 없음(속도만) (추천안, 사용자 위임 2026-11-22) |
| FR12 문자열 해시 | 꼬리 바이트를 겹친 읽기로; 문자열 factorize는 비용 4로 기준값에 닿아 10만 행부터 나뉨(§8.4, §8.7). 바꾼 것: 가변 길이 `memcpy` 꼬리, 행 수만으로 정하는 분할. 이유: 짧은 키의 1스레드 5~15%, 10만 행 10스레드 약 2.5배. 정확도: 해시 값만 바뀌고 코드는 같다 (추천안, 사용자 위임 2026-11-22) |
| FR13 NA를 다루는 커널 | `take`는 한 패스(범위 검사, gather, 비트맵 워드, NA 행 0); `fill_na`는 펼친 바이트로 고르기; 원소별 연산은 한쪽만 NA면 비트맵을 공유하고 NA 행의 0을 값의 루프에서(8바이트 결과는 NEON 비트 검사)(§4.4, §7.1, §7.2). 바꾼 것: 범위 검사·gather·`finish_fixed`의 세 패스, 행마다 `memcpy`, 비트맵 복사와 두 번째 0 쓰기 패스. 이유: `take` 약 2배, `fillna` 약 4배, NA가 있는 `f + g`가 NA 없는 것과 같은 시간. 정확도: 같은 값과 NA 행 (추천안, 사용자 위임 2026-11-22) |
| FR14 여러 컬럼을 한 번에 | `df[mask]`(`filter_columns`: 마스크 계수와 위치 한 번, 남는 행 ≤ 2/3이면 위치로 모으기), DataFrame 리덕션(`reduce_columns`), groupby의 값 컬럼들(`Groups.reduce_columns`)을 컬럼 단위로 나눈다(§7.3, §7.4, §9.1). 바꾼 것: Python에서 컬럼마다 따로 부르기, 언제나 압축. 이유: 10만 × 100의 필터 1스레드 약 2배·10스레드 약 6배, `groupby().sum()` 10스레드 약 5배. 정확도: 컬럼마다 부른 것과 같다; 에러는 컬럼 순서로 먼저 검사 (추천안, 사용자 위임 2026-11-22) |
| FR15 concat | 고정폭 복사를 행 청크로 나누고, RangeIndex들의 라벨은 결과에 바로 쓴다(§9.8). 바꾼 것: 직렬 `memcpy`, 부분마다 라벨 배열을 만든 뒤 잇기. 이유: 1스레드 약 1.3배, 10스레드 약 3배. 정확도: 같은 값과 Index (추천안, 사용자 위임 2026-11-22) |
| FR16 결정성 규칙 | 요구는 같은 스레드 수에서 실행마다 같은 비트다; frame은 스레드 수와도 무관한 결과를 지금 그대로 갖는다(§11.5). 다시 보고 둔 것: 스레드 수에 따라 비트를 바꿔 얻을 이득이 있는 곳(float 합의 분할)이 이미 빨라 바꾸지 않았다. 정확도: 해당 없음 (추천안, 사용자 위임 2026-11-22) |
| FR17 기준값 2¹⁷ | 그대로 둔다(§11.2). 깨우기 규칙이 고쳐진 뒤 10만~2¹⁸행이 모두 나뉘고, 기준값 아래의 단순 패스는 여전히 1스레드가 빠르다. 정확도: 해당 없음 (추천안, 사용자 위임 2026-11-22) |
| FR18 데이터를 보고 고르는 길 | 결과(값, 순서, dtype, 결과 타입, 에러)가 모든 길에서 같을 때만 둔다(§1.3). 바꾼 것: "데이터를 보는 휴리스틱은 넣지 않는다". 이유: FR1, FR2, FR4~FR8, FR14의 이득이 모두 이런 길이다; 결과 타입이 데이터에 따라 바뀌는 길(라벨 슬라이스의 이진 탐색)은 여전히 두지 않는다. 정확도: 해당 없음 (추천안, 사용자 위임 2026-11-22) |
