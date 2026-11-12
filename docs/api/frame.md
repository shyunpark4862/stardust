# 데이터프레임 API (sd.frame)

`import stardust as sd`의 하위 모듈 `sd.frame`이 데이터프레임 API다. 이 문서는 데이터프레임의 공개 API와 동작 규칙을 모두 적는다.

- 라이브러리 전체의 공통 규칙(에러 종류, 스레드)은 루트 [API.md](../../API.md)에 있다. 이 문서는 데이터프레임 영역의 규칙이다.
  - 루트 API.md §1의 dtype 객체는 이 영역에서 아직 쓰지 않는다. 이 영역의 dtype은 문자열이다 (§1.2).
- pandas와 다른 점은 **[pandas와 다름]**으로 표시한다.
- 절 번호(`§N`, `§N.M`)는 코드 주석이 인용한다. 번호를 바꾸지 않는다.
- 언어 표시가 없는 코드 블록은 `tests/frame/test_api_surface.py`가 읽는다.
  - 이 블록에는 공개 이름과 시그니처만 쓴다. 한 줄에 이름 하나이고, 줄 끝에 `#` 주석을 붙일 수 있다.
  - `sd.frame`의 이름은 그대로 쓴다. 예: `merge(left, right, ...)`, `NA`
  - 클래스의 멤버는 클래스 이름을 붙인다. 예: `Series.head(n=5)`. 클래스는 `Series`, `DataFrame`, `Index`, `RangeIndex`, `GroupBy`다.
  - 괄호가 있는 줄은 함수나 메서드다. 클래스 이름 자체의 줄이면 생성자의 시그니처다. 괄호가 없는 줄은 속성(모듈에서는 상수)이다.
  - 블록에 나온 이름과 공개하는 이름(밑줄로 시작하지 않는 이름)은 정확히 같아야 한다. 인자의 이름, 종류, 기본값도 같아야 한다.
    - positional-only 인자(`/` 앞)는 이름이 API가 아니다. 개수와 기본값만 맞춘다.
    - 기본값 `<없음>`은 "주지 않았다"를 뜻하는 내부 값이다.
  - 연산자, `[]`, `len()`, 반복, `in`은 이름이 아니라서 블록에 없다. §3, §5에 있다.
  - 예시는 본문의 인라인 코드로 쓴다.

---

## §1. 데이터프레임 공통 규칙

### §1.1 시그니처와 이름
- **pandas의 시그니처가 일관되지 않으면 일관되게 바꾼다.** 아래 규칙이 그 결과다.
- **주 인자는 positional-only(`/` 앞)이고, 나머지는 모두 keyword-only(`*` 뒤)다.** 배열 영역과 같은 규칙이다 (array.md §1.1).
  - 주 인자: `Series(data)`, `DataFrame(data)`, `Index(data)`, `RangeIndex(start, stop, step)`, `head(n)`, `tail(n)`, `astype(dtype)`, `fillna(value)`, `reindex(labels)`, `equals(other)`, `sort_values(by)`, `groupby(by)`, `duplicated(subset)`, `drop_duplicates(subset)`, `value_counts(subset)`, `merge(left, right)`, `merge(right)`, `concat(objs)`, `read_csv(path)`
  - 주 인자를 키워드로 주면 TypeError다. 예: `s.head(n=3)`, `df.sort_values(by="a")`
  - 주 인자의 이름은 API가 아니다. 문서에 적은 이름은 설명을 위한 것이다.
- **이름은 배열 영역(Python Array API)을 따른다.**
  - 정렬 방향은 `descending=False`다 (`ascending` 없음). `value_counts`의 기본값은 `descending=True`다 (§4.3).
  - 정렬은 `stable=True`도 받는다. frame의 정렬은 언제나 안정이라서 `stable=False`여도 결과가 같다 (§8.1).
  - 리스트로 바꾸기는 `tolist()`다. 고유값은 `unique_values()`이고 정렬한 순서다 (§4.2).
  - `to_numpy`의 `copy`는 배열의 `asarray`와 같은 뜻이다 (§2.7).
- **bool 옵션**은 `True`나 `False`만 받는다. 다른 값은 TypeError다 (API.md §2). 예: `descending=1`, `stable=None`
- **별칭을 두지 않는다.**
  - `tolist`만 둔다 (`to_list` 없음).
  - `unique_values`만 둔다 (`unique` 없음).
  - `isna`, `notna`만 둔다 (`isnull`, `notnull` 없음).
  - `merge`는 함수 `sd.frame.merge(left, right)`와 메서드 `df.merge(right)`를 모두 둔다.
  - `concat`은 함수만 둔다.
- **연산은 연산자로만 한다.** `add`, `sub`, `mul`, `div`, `truediv`, `eq`, `abs()` 같은 메서드는 두지 않는다.
  - 이 메서드들은 연산자의 별칭이고, `div`/`divide`/`truediv`처럼 서로도 별칭이다.
  - 절댓값은 `abs(s)`로 구한다.
  - `fill_value`가 필요하면 먼저 `fillna`를 한 뒤 연산한다.
- **리덕션 인자는 Series, DataFrame, GroupBy에서 같다** (§6.1).
- **이 설계에서 의미가 없는 인자는 두지 않는다.**
  - `copy`: Copy-on-Write를 쓰므로 필요 없다 (`astype`에도 없다). `to_numpy`만 예외다.
  - `inplace`
  - `level`: MultiIndex가 없다.
  - `kind`: 정렬은 항상 안정 정렬이다. 배열 영역의 `stable`은 받는다 (§8.1).
  - `errors`: 항상 에러를 낸다.
  - `engine`
  - Series 메서드의 `axis`
- **`keep`은 `"first"`, `"last"`, `"none"`이다.** pandas의 `keep=False`가 `"none"`이다.
- **하위 모듈은 공개하지 않는다.** 구현 모듈은 `_series`, `_frame`, `_index`처럼 밑줄로 시작한다. 그래서 `sd.frame.index` 같은 이름은 없다. 공개 이름은 `sd.frame.__all__`이 전부다.

### §1.2 dtype
- dtype은 넷이다: `sd.bool`, `sd.int64`, `sd.float64`, `sd.string` (API.md §1의 객체)
  - `Series.dtype`, `Index.dtype`은 이 객체를 돌려준다. `DataFrame.dtypes`는 `{컬럼 이름: dtype 객체}` dict다(§2.6). repr은 이름(`dtype: int64`)을 쓴다.
  - **NA가 있어도 dtype은 바뀌지 않는다.** pandas는 nullable dtype을 쓰면 `Int64`, `boolean` 등으로 쓴다.
- dtype을 받는 인자는 이 객체나 그 이름(`"bool"`, `"int64"`, `"float64"`, `"string"`)을 받는다 (API.md §1).
  - Python 타입(`bool`, `int`, `float`, `str`), `sd.complex128`, 다른 이름은 `TypeError`다.
  - 해당: 생성자의 `dtype=` (§2.3), `astype` (§3.5), `to_numpy(dtype=)` (§2.7), `read_csv(dtype=)` (§10.1)

### §1.3 NaN
NaN은 결측값이 아니라 평범한 float 값이다.
- `isna()`는 NaN을 결측으로 보지 않는다. NaN은 `isnan()`으로 찾는다 (§3.6).
- **리덕션**: `sum`, `mean` 등에 NaN이 섞이면 결과가 NaN이 된다. **[pandas와 다름]**
  - `skipna`는 NA만 건너뛴다.
  - `min`, `max`도 NaN을 전파한다.
- **정렬**: NaN은 +inf보다 큰 값으로 친다. 그래서 오름차순에서는 +inf 뒤에, 내림차순에서는 맨 앞에 온다 (§8.2).
  - NA의 위치는 `na_position`으로 따로 정한다.
- **키 비교**: NaN끼리는 같은 키다 (§1.6).
- 비교 연산자는 IEEE 754를 따른다. `NaN == NaN`은 False다 (§3.3).

### §1.4 NA
```
NA
```
- 결측 스칼라는 싱글턴 `sd.frame.NA` 하나다. 모든 dtype에 쓴다.
- **비교와 산술**: `NA`가 들어간 연산의 결과는 `NA`다.
- **논리 연산**: Kleene 3값 논리를 쓴다.
  - `True | NA == True`
  - `False & NA == False`
  - 나머지 조합은 `NA`다.
- **필터**: 마스크 값이 NA인 행은 False로 취급해서 버린다 (§5.5).
- **`bool(NA)`**: TypeError를 낸다.
- **값이 모두 NA인 경우의 리덕션** (빈 컬럼도 같다, §6.4)
  - `sum`은 0이다.
  - `count`는 0이다.
  - `mean`, `min`, `max`, `std`, `var`는 `NA`다.
- **`std`, `var`의 기본 자유도**: `ddof=1`
  - 배열 영역의 `std`, `var`는 `correction=0`이 기본이라서 다르다 (array.md §7.4).

### §1.5 정수 연산
- `int / int`의 결과는 float64다 (true division).
- 정수 오버플로는 wrap한다. 예: `INT64_MIN // -1`과 `abs(INT64_MIN)`은 `INT64_MIN`이다.
- **`int // 0`과 `int % 0`의 결과는 `NA`다.** **[pandas와 다름]**
- **`int ** 음수`의 결과도 `NA`다.** **[pandas와 다름]**
- float를 0으로 나누면 IEEE 754를 따른다 (`inf`, `nan`).
  - 결과가 `NA`가 되지 않는다.

### §1.6 같은 키
groupby, `unique_values`, `nunique`, `value_counts`, `duplicated`, `drop_duplicates`, merge, Series의 `in`은 같은 규칙으로 키를 비교한다.
- **NaN끼리는 같은 키다.** 비트가 서로 다른 NaN들도 같은 키다.
- **`-0.0`과 `0.0`은 같은 키다.**
- 해시 테이블에 넣기 전에 키를 정규화해서 처리한다.
- **NA끼리는 같은 키다.** 예외는 merge 하나다: NA 키는 아무것과도 매칭되지 않는다 (§9.1).
- 배열의 `unique_*`도 같은 규칙이다: NaN은 하나, `-0.0`과 `0.0`도 하나, 처음 나온 값을 보여 준다 (array.md §8.3).
- **대표값**: 같은 키로 묶인 값 중에서 먼저 나온 값을 그대로 보여 준다 (pandas와 같다).
  - 예: `[-0.0, 0.0]`의 `unique_values()`는 `[-0.0]`이다.

### §1.7 라벨
- **Index에 쓸 수 있는 dtype**: `int64`, `string`, 그리고 `RangeIndex`
  - float와 bool은 Index로 쓸 수 없다.
  - Index에는 NA를 넣을 수 없다.
  - RangeIndex는 int64 라벨로 친다.
  - `datetime64`는 §13에 있다.
- **중복 라벨**: 허용한다.
  - 정렬(alignment)이 필요한 연산에서 라벨이 고유하지 않으면 에러를 낸다. **[pandas와 다름]**
  - 두 Index가 같으면 위치대로 계산하므로 중복이 있어도 된다.
- **DataFrame 컬럼 이름**: 고유한 `str`만 허용한다.
- **정렬(alignment) 결과의 라벨 순서**: merge outer와 같은 규칙을 쓴다 (§9.1). **[pandas와 다름]**
  - left 라벨 순서를 먼저 따르고, right에만 있는 라벨을 right 순서대로 뒤에 붙인다.
  - pandas는 두 Index가 다르면 합집합을 정렬한다.
- 정렬이 일어나는 곳과 그 규칙은 §5.6에 있다.

### §1.8 스칼라, 경고, 에러
- **리덕션 결과**: Python 스칼라를 돌려준다. 원소 하나를 꺼낼 때도 마찬가지다.
- **결측 스칼라**: 싱글턴 `NA` (§1.4)
- **경고**: 일절 내지 않는다. 문제가 있으면 에러를 낸다 (API.md §2).
- **대입으로 dtype이 바뀌는 경우**: 에러를 낸다 (§5.7).
  - 예: int 컬럼에 `1.5`를 대입하는 것

---

## §2. 생성과 변환

### §2.1 생성자
```
Series(data=None, /, *, index=None, dtype=None, name=None)
DataFrame(data=None, /, *, index=None, columns=None, dtype=None)
```
- `data`를 주지 않으면 빈 Series, 빈 DataFrame이다 (pandas와 같다).
- `data`에서 dtype을 정하는 규칙은 §2.2–§2.4, DataFrame이 받는 입력은 §2.5에 있다.
- **`index=`**
  - list나 배열을 주면 길이가 `index`와 같아야 한다.
  - Series를 주면 `index`에 맞춰 다시 배열한다(reindex). 없는 라벨은 NA다.
- `name`은 §2.6에 있다.

### §2.2 list에서 dtype 추론
- `None`과 `NA`는 NA다.
- `float('nan')`은 NaN 값이다.
- bool만 있으면 `bool`, int만 있으면 `int64`, float만 있거나 int와 float가 섞이면 `float64`, str만 있으면 `string`이다.
- 빈 리스트나 값이 모두 NA인 리스트는 `float64`다.
- **다음 경우는 에러다.**
  - bool이 다른 타입과 섞인 경우 (예: `[True, 1]`) **[pandas와 다름]**
  - str이 다른 타입과 섞인 경우 **[pandas와 다름]**
  - int64 범위를 넘는 정수
  - int와 float가 섞일 때 2^53을 넘어 float64로 정확히 표현할 수 없는 int **[pandas와 다름]**

### §2.3 생성자의 `dtype=`
- **손실 없는 숫자 변환만 허용한다.**
  - int → `float64`: 2^53을 넘어 정확히 표현할 수 없으면 에러다.
  - 정수 값만 가진 float → `int64`: `[1.0, 2.0]`은 되고 `[1.5]`나 NaN은 에러다.
  - bool과 숫자 사이, 문자열과 다른 타입 사이의 변환은 에러다. 이런 변환은 `astype`으로 한다 (§3.5).
- dtype의 표기는 §1.2에 있다.
- 같은 손실 없는 규칙을 `fillna` (§3.6)와 행 대입 (§5.7)이 쓴다.

### §2.4 numpy 배열에서 만들기
- **항상 복사한다.**
  - numpy 배열은 밖에서 수정될 수 있어서, 메모리를 공유하면 Copy-on-Write가 깨진다.
- dtype 변환
  - int8~int32, uint8~uint32 → `int64`
  - uint64 → `int64`. 범위를 넘으면 에러다.
  - float16, float32 → `float64`
  - U 문자열 → `string`
  - object 배열 → list와 같은 규칙 (§2.2)
- float 배열의 NaN은 NaN 값이다.

### §2.5 DataFrame 생성 입력
- `{이름: list | 1D numpy 배열 | Series}` 형태의 dict
  - 길이가 다르면 에러다.
  - Series는 라벨로 정렬(alignment)한다 (§5.6).
- 행의 list
  - dict의 list: 키가 없는 칸은 NA다. 컬럼 순서는 키가 처음 나타난 순서다.
  - list의 list: `columns=`가 꼭 필요하다.
- 2D numpy 배열: `columns=`가 꼭 필요하다.
- 행 단위 입력은 컬럼으로 바꾼 뒤 list와 같은 규칙으로 dtype을 추론한다 (§2.2).

### §2.6 속성과 `head`, `tail`
```
Series.dtype
Series.index
Series.name                # 대입할 수 있다
Series.shape
Series.head(n=5, /)
Series.tail(n=5, /)
DataFrame.columns
DataFrame.dtypes
DataFrame.index
DataFrame.shape
DataFrame.head(n=5, /)
DataFrame.tail(n=5, /)
```
- `shape`: Series는 `(행,)`, DataFrame은 `(행, 열)`이다. pandas와 같다.
- **`Series.name`**: `str` 또는 `None`만 쓸 수 있다. **[pandas와 다름]**
  - Index의 `name`도 같은 규칙이다 (§5.1).
- `df.columns`는 string `Index`다.
- `df.dtypes`는 `{컬럼 이름: dtype 객체}` dict다. 순서는 컬럼 순서다. frame에는 객체를 담는 dtype이 없어서 Series가 아니다.

### §2.7 `tolist`, `to_numpy`
```
Series.tolist()
Series.to_numpy(*, dtype=None, copy=None, na_value=<없음>)
DataFrame.to_numpy(*, dtype=None, copy=None, na_value=<없음>)
```
- **`tolist()`**: 값을 Python 스칼라의 list로 돌려준다. NA 행은 `NA`다. `Index.tolist()`도 있다 (§5.1).
- **`to_numpy`**
  - **`copy`는 배열의 `asarray(copy=)`와 같은 뜻이다** (array.md §1.8, §3.2).
    - `None`(기본값): 필요할 때만 복사한다. NA가 없고 dtype 변환이 없으면 복사 없이 읽기 전용 뷰를 돌려준다.
    - `True`: 언제나 쓰기 가능한 복사본을 돌려준다.
    - `False`: 복사하지 않는다. 복사가 필요하면 ValueError다. 복사가 필요한 경우는 NA 행을 채울 때, dtype을 바꿀 때, string 컬럼(object 배열을 새로 만든다)이다.
    - 그 밖의 값은 TypeError다.
    - numpy 프로토콜(`np.asarray(s, copy=...)`)도 같은 규칙이다.
  - NA가 있으면 `na_value`가 필요하다. 없으면 에러다.
  - `na_value`가 그 dtype에 들어가지 않으면 에러다.
    - 예: int 컬럼에 `na_value=np.nan`. 이때는 `dtype="float64"`도 함께 준다.
  - `string`은 Python str을 담은 object 배열로 돌려준다.
- **`DataFrame.to_numpy`**
  - 모든 컬럼의 dtype이 같으면 2D 배열을 돌려준다.
  - 컬럼은 버퍼가 따로따로라서 2D 배열은 언제나 새로 만든다. 그래서 `copy=False`는 ValueError다.
  - int64와 float64가 섞이면 float64로 만든다. 2^53 규칙은 §2.3과 같다.
  - 그 밖의 조합은 에러다.

### §2.8 배열과 주고받기 (`to_array`, `Series(배열)`)
```
Series.to_array(*, dtype=None, copy=None, na_value=<없음>)
DataFrame.to_array(*, dtype=None, copy=None, na_value=<없음>)
```
- **`to_array`**: `sd.ndarray`를 돌려준다. 규칙은 `to_numpy`(§2.7)와 같다.
  - `copy=None`이고 NA와 dtype 변환이 없으면 컬럼의 메모리를 함께 쓰는 **읽기 전용** 뷰다(복사 없음).
  - 뷰가 살아 있는 동안 frame 쪽에 대입하면, frame이 먼저 복사한다(Copy-on-Write, §5.7). 뷰의 값은 바뀌지 않는다.
  - string 컬럼은 TypeError다. 배열 영역에는 문자열 dtype이 없다.
- **`DataFrame.to_array`**: 새 2차원 배열을 **F-order**로 만든다. 컬럼 하나가 메모리에서 이어져 있고, 컬럼마다 한 번씩 복사한다. `sd.linalg`는 F-order 행렬을 그대로 받는다. dtype 규칙은 `DataFrame.to_numpy`와 같고, `copy=False`는 ValueError다.
- **`Series(배열)`, `DataFrame(2차원 배열)`**: `sd.ndarray`에서 만들면 **언제나 복사한다.** 배열은 다른 쓰기 가능한 뷰를 통해 바뀔 수 있어서 frame의 Copy-on-Write가 따라갈 수 없다.
  - bool, int64, float64 배열을 받는다. complex128은 TypeError다. `Series`는 1차원, `DataFrame`은 2차원만 받는다.
  - `dtype=`는 명시적 변환이다 (§2.3).

---

## §3. 원소별 연산

### §3.1 피연산자
- **Series와 Series**: 라벨로 정렬(alignment)한다 (§5.6).
- **Series와 스칼라, 스칼라와 Series**
  - 스칼라는 Python 스칼라, numpy 스칼라, `NA`다.
  - int64 범위를 넘는 int 스칼라는 에러다.
- **DataFrame과 스칼라**
- **DataFrame과 DataFrame**
  - 컬럼을 이름으로 짝짓는다. 행과 열 모두 정렬(alignment)한다 (§5.6).
  - 결과의 컬럼은 왼쪽의 컬럼 순서를 따르고, 오른쪽에만 있는 컬럼을 오른쪽 순서대로 뒤에 붙인다 (§1.7).
- **DataFrame과 Series는 지원하지 않는다.** **[pandas와 다름]**
- **결과 이름**: Series끼리 연산할 때 이름이 같으면 유지하고, 다르면 `None`이다. 스칼라와 연산하면 유지한다 (pandas와 같다).

### §3.2 산술
`+`, `-`, `*`, `/`, `//`, `%`, `**`, 단항 `-`와 `+`, `abs(s)`
- **결과 dtype**
  - int64와 int64는 int64다. 단, `/`는 float64다.
  - int64나 float64가 섞이면 float64다. int는 float64로 바꾼 뒤 계산한다.
  - string `+` string은 string이다 (이어 붙이기). string의 다른 산술은 에러다.
  - **bool의 산술은 에러다.** 숫자가 필요하면 `astype("int64")`를 쓴다. **[pandas와 다름]**
- `//`와 `%`는 floor 규칙을 따른다 (Python, numpy와 같다). `%`의 결과는 나누는 수의 부호를 따른다.
- int 규칙은 §1.5에 있다: wrap, `// 0`과 `% 0`은 NA, `** 음수`는 NA
- float는 IEEE 754를 따른다.
- **NA가 들어가면 결과는 NA다.** 예외가 없다. **[pandas와 다름]**
  - pandas는 `NA ** 0`과 `1 ** NA`를 1로 만든다.

### §3.3 비교
`==`, `!=`, `<`, `<=`, `>`, `>=`
- 결과는 bool이다. 어느 한쪽이 NA인 행은 NA다.
- **int와 float의 비교는 반올림 없이 수학적으로 정확하게 한다.** **[pandas와 다름]**
  - 예: `2**53 + 1 == 2.0**53`은 False다. pandas는 int를 float로 바꿔서 비교하므로 True다.
- NaN은 IEEE 754를 따른다. `NaN == NaN`은 False, `NaN != NaN`은 True다.
- string은 코드포인트 순서로 비교한다. UTF-8 바이트 순서와 같고, Python str의 비교와도 같다.
- bool은 bool끼리만 비교한다. `False < True`다.
- **비교할 수 없는 dtype끼리 비교하면 `==`도 에러다** (예: string과 int). **[pandas와 다름]**

### §3.4 논리와 비트 연산
`&`, `|`, `^`, `~`, `<<`, `>>`
- **bool과 bool**: `&`, `|`, `^`, `~`는 논리 연산이다. 결과는 bool이다.
  - `&`와 `|`는 Kleene 규칙을 따른다 (§1.4).
  - `^`와 `~`는 NA가 들어가면 NA다.
- **int64**: `&`, `|`, `^`, `~`, `<<`, `>>`는 비트 연산이다. 결과는 int64다. 규칙은 배열 영역의 `bitwise_*`와 같다 (array.md §6.4).
  - bool이 int64와 만나면 bool을 int64(0, 1)로 바꾼 뒤 계산한다. 예: `Series([True]) & Series([3])`은 int64 `[1]`이다.
  - `<<`, `>>`는 int64끼리만 받는다. bool은 TypeError다 (배열과 같다).
  - 시프트 수는 부호 없는 수로 읽는다. 그래서 음수이거나 64 이상이면 모든 비트가 밀려 나간다. `<<`는 0, `>>`는 부호(0 또는 -1)다. 예: `s << -1`은 0이다.
  - `<<`의 오버플로는 wrap한다 (§1.5). `>>`는 산술 시프트다.
  - **NA가 들어가면 결과는 NA다.** Kleene 규칙은 bool끼리의 논리 연산에만 쓴다.
- float64와 string은 TypeError다. Python float 스칼라도 TypeError다 (`s & 1.0`).

### §3.5 `astype`
```
Series.astype(dtype, /)
DataFrame.astype(dtype, /)
```
- DataFrame은 dtype 하나나 `{컬럼: dtype}` dict를 받는다.
- `copy` 인자는 없다. Copy-on-Write라서 뜻이 없다 (§1.1).
- NA는 그대로 NA다.
- int → float: 반올림한다.
- float → int: 0 쪽으로 자른다. NaN, inf, int64 범위를 넘는 값은 에러다.
- bool → 숫자: 0과 1이다.
- 숫자 → bool: 0이 아니면 True다. NaN도 True다.
- → string: Python `str()`과 같다. 예: `1.0` → `"1.0"`, `1e20` → `"1e+20"`, NaN → `"nan"`
- **string → int64, float64: 엄격하게 파싱한다.**
  - 받는 것: 부호, 숫자, 소수점, 지수. float는 `nan`, `inf`도 받는다.
  - 공백과 `_`는 에러다.
- **string → bool**: `"True"`와 `"False"`만 받고 나머지는 에러다. **[pandas와 다름]**

### §3.6 결측: `isna`, `notna`, `isnan`, `fillna`, `dropna`
```
Series.isna()
Series.notna()
Series.isnan()
Series.fillna(value, /)
Series.dropna(*, ignore_index=False)
DataFrame.isna()
DataFrame.notna()
DataFrame.isnan()
DataFrame.fillna(value, /)
DataFrame.dropna(*, how="any", subset=None, ignore_index=False)
```
- **`isna()`, `notna()`**: NA만 결측으로 본다. NaN은 결측이 아니다 (§1.3).
  - 결과는 bool이고 NA가 없다.
- **`isnan()`**: 값이 NaN인지 본다. `isna()`와 별개다.
  - float64 컬럼: NaN이면 True다. 부호나 payload가 다른 NaN도 True다.
  - 다른 dtype: NaN이 없으므로 모두 False다 (배열의 `isnan`이 int에 False인 것과 같다).
  - **NA 행은 NA다.** 비교 연산(§3.3)처럼 NA가 결과로 전파된다. 마스크로 쓰면 NA 행은 버려진다 (§5.5).
  - DataFrame은 컬럼마다 같은 규칙이다. 결과는 같은 컬럼 이름의 bool DataFrame이다.
- **`fillna(value)`**
  - 값은 컬럼 dtype으로 손실 없이 바뀌어야 한다. 규칙은 생성자 `dtype=`와 같다 (§2.3).
    - 예: int 컬럼에 `2.0`은 되고 `1.5`는 에러다.
  - DataFrame은 `{컬럼: 값}` dict도 받는다.
- **`dropna`**
  - DataFrame의 `how="all"`은 subset 안의 값이 모두 NA인 행만 지운다.
  - `axis`(열 지우기)와 `thresh`는 없다.

---

## §4. 중복과 고유값

### §4.1 함수
```
Series.unique_values()
Series.nunique(*, dropna=True)
Series.duplicated(*, keep="first")
Series.drop_duplicates(*, keep="first", ignore_index=False)
Series.value_counts(*, normalize=False, sort=True, descending=True, dropna=True)
DataFrame.nunique(*, dropna=True)
DataFrame.duplicated(subset=None, /, *, keep="first")
DataFrame.drop_duplicates(subset=None, /, *, keep="first", ignore_index=False)
DataFrame.value_counts(subset=None, /, *, normalize=False, sort=True, descending=True, dropna=True)
```
- 키를 비교하는 규칙은 §1.6에 있다.
- `keep`은 `"first"`, `"last"`, `"none"`이다 (§1.1).
- `dropna=`는 NA만 버린다.
- `bins`는 없다.
- factorize는 공개하지 않는다. 값을 정수 코드로 바꾸는 내부 기능으로만 쓴다.

### §4.2 결과
- **`Series.unique_values()`는 Series를 돌려준다.**
  - **값은 정렬한 순서다.** 배열의 `unique_values`와 같다 (array.md §8.3). 처음 나타난 순서가 아니다. **[pandas와 다름]**
    - 순서는 groupby의 `sort=True`와 같다 (§7.2): 수 순서이고 NaN은 +inf 뒤, string은 코드포인트 순서, bool은 `False < True`다.
    - NaN은 하나다. `-0.0`과 `0.0`도 하나이고, 먼저 나온 값을 보여 준다 (§1.6).
    - NA도 포함하고, 맨 뒤에 온다.
  - 이름은 유지한다. Index는 RangeIndex다.
  - 처음 나타난 순서가 필요하면 `drop_duplicates()`를 쓴다.
- **`Series.nunique()`는 int를 돌려준다.**
- **`DataFrame.nunique()`**는 컬럼 이름을 Index로 하는 int64 Series를 돌려준다.
- **NaN은 값이다.** `nunique`와 `value_counts`에서 NaN도 하나의 키로 센다. **[pandas와 다름]**
  - pandas는 NaN을 결측으로 보고 기본으로 버린다.

### §4.3 `value_counts`
- **`value_counts`는 항상 DataFrame을 돌려준다.** **[pandas와 다름]**
  - 이유: pandas처럼 값을 Index로 삼으면, float와 bool은 Index로 쓸 수 없다는 규칙(§1.7)과 충돌한다.
  - `DataFrame.value_counts()`는 다중 키 groupby와 같은 방식이다 (§7.2).
- **컬럼**
  - 키 컬럼과 `count` 컬럼이 있다. `normalize=True`이면 `count` 대신 `proportion`(float64)이다.
  - `Series.value_counts`의 키 컬럼 이름은 Series 이름이다. 이름이 없으면 `"value"`다.
  - 키 컬럼 이름이 `count`(또는 `proportion`)와 겹치면 에러다.
- **행 순서**
  - `sort=True`이면 개수의 내림차순이다(`descending=True`가 기본값). `descending=False`이면 오름차순이다.
    - `descending`은 bool 옵션이다 (§1.1). 정렬 함수와 이름이 같고, 기본값만 다르다.
  - 개수가 같으면 처음 나타난 순서를 따른다.
  - `sort=False`이면 처음 나타난 순서다.
  - NA 키도 같은 규칙을 따른다. **[pandas와 다름]**
    - pandas는 `dropna=False`일 때 NA를 처음 나타난 자리가 아니라 맨 뒤에 놓는다 (`sort=False`여도 그렇다).
    - 그런데 pandas의 `unique()`는 NA를 나타난 자리에 둔다. 여기서는 `value_counts`를 처음 나타난 순서로 맞춘다.
- Index는 RangeIndex다.
- `dropna=True`이면 키 중에 NA가 있는 행을 버린다.

---

## §5. Index, 선택, 대입

### §5.1 Index와 RangeIndex
```
Index(data, /, *, name=None)
RangeIndex(start=0, stop=None, step=1, /, *, name=None)
Index.dtype
Index.name
Index.is_unique
Index.tolist()
Index.equals(other, /)
RangeIndex.dtype
RangeIndex.name
RangeIndex.is_unique
RangeIndex.start
RangeIndex.stop
RangeIndex.step
RangeIndex.tolist()
RangeIndex.equals(other, /)
```
- Index는 바뀌지 않는다(불변).
- **`Index(data)`**: 라벨은 int64나 string이다. float, bool, NA는 에러다 (§1.7).
- **`RangeIndex`**: `start`, `stop`, `step`은 생성 인자다. 배열의 `arange`처럼 위치로만 준다. int64 라벨로 친다.
- **`name`**은 `str` 또는 `None`이다 (`Series.name`과 같은 규칙, §2.6).
  - 연산 결과의 Index 이름은 Series 이름 규칙을 따른다: 같으면 유지하고, 다르면 `None`이다 (§3.1).
  - repr은 pandas처럼 이름을 보여 준다.
- 공개하는 것: `dtype`, `name`, `len()`, 반복(라벨), `in`(라벨), `tolist()`, `equals()`, `is_unique`
  - `equals()`는 라벨만 비교한다. 이름은 보지 않는다 (pandas와 같다).
- 라벨 조회용 해시 테이블은 처음 조회할 때 만들어서 캐시한다.
- **라벨의 타입은 Index의 dtype과 맞아야 한다.**
  - int Index에는 int 라벨만 쓴다. bool과 float는 쓸 수 없다.
  - string Index에는 str 라벨만 쓴다.
  - 타입이 맞지 않으면 TypeError다.

### §5.2 `reindex`
```
Series.reindex(labels, /)
DataFrame.reindex(labels, /)        # 행만
```
- 주어진 라벨 순서로 행을 다시 배열한다. 없는 라벨은 NA다.
- 원래 라벨은 고유해야 한다.
- `method`, `fill_value` 등은 없다.

### §5.3 `[]`
- **DataFrame**
  - `df["a"]`: 컬럼 하나 (Series)
  - `df[["a", "b"]]`: 컬럼 여러 개 (DataFrame)
  - `df[bool Series]`: 행 필터 (§5.5)
  - `df[1:3]`처럼 `[]`로 행을 자르는 것은 지원하지 않는다. `.iloc`을 쓴다. **[pandas와 다름]**
- **Series**
  - `s[x]`는 라벨 기준이다. `s.loc[x]`와 같다.
  - 위치 기준 접근은 `.iloc`으로만 한다.

### §5.4 `.loc`, `.iloc`
```
Series.loc
Series.iloc
DataFrame.loc
DataFrame.iloc
```
- **`.loc`에 쓸 수 있는 것**
  - 라벨 하나: 항상 스칼라를 돌려준다. **[pandas와 다름]**
    - 라벨이 없으면 KeyError다.
    - 라벨이 여러 행에 있으면 에러다. 그런 행을 모두 원하면 `s.loc[["a"]]`처럼 리스트로 조회한다.
  - 라벨 리스트: 각 라벨에 맞는 행을 모두 Index 순서대로 돌려준다. 없는 라벨이 하나라도 있으면 KeyError다.
  - 라벨 슬라이스 `a:b`
    - 양 끝을 포함한다.
    - 양 끝 라벨은 있어야 하고 고유해야 한다.
    - Index가 정렬되어 있는지 보고 이진 탐색으로 바꾸는 휴리스틱은 쓰지 않는다. **[pandas와 다름]**
  - bool Series 마스크 (§5.5)
- **`.iloc`에 쓸 수 있는 것**: 정수(음수 가능), 슬라이스, 정수 리스트
  - 범위를 벗어나면 IndexError다.
- **DataFrame**
  - `df.loc[행]`, `df.loc[행, 열]`, `df.iloc[행]`, `df.iloc[행, 열]`
  - 열 자리에는 이름 하나, 이름 리스트, 이름 슬라이스(양 끝 포함), `:`를 쓴다.
  - **행을 하나만 고르면 1행짜리 DataFrame을 돌려준다.** **[pandas와 다름]**
    - pandas는 행을 Series로 돌려주는데, dtype이 섞이면 object가 된다.
    - 행 하나와 열 하나를 함께 고르면 스칼라다. 예: `df.loc["r", "a"]`
- **결과의 Index**
  - 슬라이스로 고르면 RangeIndex는 RangeIndex로 남는다. 라벨 리스트나 마스크로 고르면 Index가 된다.
  - 결과 타입은 무엇으로 골랐는지에 따라 정해지고, 데이터에 따라 바뀌지 않는다.

### §5.5 마스크
- 해당: `s[mask]`, `df[mask]`, `.loc`의 마스크
- mask는 bool Series다.
- mask가 NA인 행은 버린다 (§1.4).
- **마스크와 대상의 라벨 집합이 같아야 한다.** **[pandas와 다름]**
  - 라벨 순서는 달라도 된다.
  - 대상의 라벨이 마스크에 없거나, 마스크에만 있는 라벨이 있으면 에러다.
  - pandas는 마스크에만 있는 라벨을 조용히 무시한다.
- 결과의 Index는 항상 남은 행 라벨의 `Index`다. RangeIndex에서 고르면 int64 `Index`다. 데이터에 따라 RangeIndex가 되는 일은 없다.

### §5.6 정렬(alignment)
- **정렬이 일어나는 곳**
  - Series와 Series의 연산 (§3.1)
  - DataFrame과 DataFrame의 연산 (행과 열 모두)
  - Series dict로 DataFrame을 만들 때 (§2.5)
  - `df["c"] = Series` (§5.7)
  - 마스크 (§5.5)
- 두 Index가 같으면 위치대로 계산한다.
- 다르면 양쪽 라벨이 모두 고유해야 한다 (§1.7).
  - 합집합의 순서는 왼쪽 라벨 순서, 그 다음 오른쪽에만 있는 라벨을 오른쪽 순서대로 붙인 것이다 (§1.7).
  - 한쪽에 없는 라벨의 행은 NA다.
- **int 라벨과 string 라벨은 정렬할 수 없다(에러).** RangeIndex는 int64 라벨로 친다. **[pandas와 다름]**
- DataFrame끼리 연산할 때 한쪽에만 있는 컬럼은, 다른 쪽을 같은 dtype의 NA 컬럼으로 보고 계산한다.

### §5.7 대입
Copy-on-Write를 따른다.
- **지원하는 형태**: `df["c"] = 값`, `s.loc[...] = 값`, `s.iloc[...] = 값`, `df.loc[행, "c"] = 값`, `df.iloc[행, j] = 값`
  - 행은 선택과 같은 방법으로 고른다 (§5.4).
- **넣을 수 있는 값**
  - 스칼라 (`NA` 포함)
  - list: 길이가 고른 행 수와 같아야 한다.
  - Series: 고른 행의 라벨에 맞춰 정렬한다 (§5.6).
- `df["c"] = 값`은 컬럼 전체를 새로 만든다. 새 컬럼이면 끝에 붙는다.
- 행에 대입하는 값은 컬럼 dtype으로 손실 없이 바뀌어야 한다. 규칙은 생성자 `dtype=`와 같다 (§2.3).
- **없는 라벨에 대입하면 KeyError다.** 행을 늘리지 않는다. **[pandas와 다름]**
- 쓰기 전에 버퍼를 다른 객체(다른 Series, DataFrame, 슬라이스, `to_numpy` 뷰)와 공유하고 있으면 먼저 복사한다.
  - 예: `s = df["a"]; s.iloc[0] = 1`을 해도 `df`는 바뀌지 않는다.
- 다른 Python 스레드와 동시에 대입할 때의 규칙은 §12에 있다.

### §5.8 반복과 `in`
- **Series**: 반복과 `in` 모두 값 기준이다. **[pandas와 다름]**
  - pandas는 반복은 값, `in`은 라벨이다.
  - `x in s`는 키 규칙(§1.6)으로 판단한다. NaN은 NaN과, NA는 NA와 같다.
  - 라벨이 있는지는 `x in s.index`로 확인한다.
- **DataFrame**: 반복과 `in` 모두 컬럼 이름 기준이다.
- **Index**: 반복과 `in` 모두 라벨 기준이다.

---

## §6. 리덕션

### §6.1 함수
```
Series.count()
Series.sum(*, skipna=True)
Series.mean(*, skipna=True)
Series.min(*, skipna=True)
Series.max(*, skipna=True)
Series.std(*, skipna=True, correction=0)
Series.var(*, skipna=True, correction=0)
DataFrame.count()
DataFrame.sum(*, skipna=True)
DataFrame.mean(*, skipna=True)
DataFrame.min(*, skipna=True)
DataFrame.max(*, skipna=True)
DataFrame.std(*, skipna=True, correction=0)
DataFrame.var(*, skipna=True, correction=0)
```
- **인자는 Series, DataFrame, GroupBy(§7.3)에서 같다.**
  - `sum`, `mean`, `min`, `max`: `(*, skipna=True)`
  - `std`, `var`: `(*, skipna=True, ddof=1)`. `ddof`는 0 이상의 int다.
  - `count()`: 인자가 없다. NA가 아닌 값의 개수다.
  - `axis`(행 방향 리덕션), `min_count`, `numeric_only`는 없다.

### §6.2 반환값
- **Series**: Python 스칼라 또는 `NA`다 (§1.8).
- **DataFrame**: 1행짜리 DataFrame이다. **[pandas와 다름]**
  - 행 라벨은 리덕션 이름(`"sum"`, `"mean"` 등)이다.
  - 컬럼마다 자기 결과 dtype을 유지한다.
  - 리덕션을 할 수 없는 컬럼이 하나라도 있으면 TypeError다 (예: string 컬럼의 `mean`).
  - pandas는 dtype이 섞이면 object Series를 돌려준다.

### §6.3 dtype별 결과

| dtype | sum | mean | min, max | std, var |
|---|---|---|---|---|
| int64 | int64 (wrap, §1.5) | float64: int128로 정확히 더한 뒤 나눔 | int64 | float64 |
| float64 | float64 | float64 | float64 | float64 |
| bool | int64: True의 개수 | float64: True의 비율 | bool | TypeError |
| string | TypeError | TypeError | string (코드포인트 순서) | TypeError |

### §6.4 NA와 NaN
- **NA**
  - `skipna=True`(기본값)이면 NA를 건너뛴다.
  - `skipna=False`이면 NA가 하나라도 있을 때 결과가 NA다.
  - 값이 모두 NA이거나 없으면: `sum`은 0, `count`는 0이고, 나머지는 NA다 (§1.4).
  - `std`, `var`에서 `NA가 아닌 값의 개수 - ddof <= 0`이면 NA다. **[pandas와 다름]** pandas는 NaN이다.
- **NaN**: `skipna`와 상관없이 결과로 전파된다 (§1.3).
- **0.0과 -0.0**: `min`과 `max`는 `-0.0 < 0.0`으로 본다. numpy와 같고, IEEE 754-2019의 minimum, maximum이다.

### §6.5 계산 순서
- **float 합의 순서**: numpy의 pairwise summation이다. 블록 128개, 누산기 8개, 절반씩 재귀하고, 0.0에서 시작한다.
  - NA가 없으면 pandas, numpy와 비트까지 같다.
  - NA 행은 0.0으로 채워져 있으므로 그대로 더한다.
  - 재귀 트리의 가지를 나눠서 병렬로 계산하므로, 스레드 수와 상관없이 비트가 같다 (§12).
- **`var`**는 두 번 훑어서 계산한다.
  1. 평균을 구한다.
  2. 편차 제곱 `(x - 평균)²`을 pairwise로 더한다.
  3. `개수 - ddof`로 나눈다.
  - `std`는 `sqrt(var)`이다.

---

## §7. groupby

### §7.1 함수
```
DataFrame.groupby(by, /, *, sort=True, dropna=True)
GroupBy.agg(func=None, /, **named)
GroupBy.size()
GroupBy.count()
GroupBy.sum(*, skipna=True)
GroupBy.mean(*, skipna=True)
GroupBy.min(*, skipna=True)
GroupBy.max(*, skipna=True)
GroupBy.std(*, skipna=True, correction=0)
GroupBy.var(*, skipna=True, correction=0)
GroupBy.first(*, skipna=True)
GroupBy.last(*, skipna=True)
```
- `GroupBy`는 `df.groupby(...)`가 돌려주는 객체다.
- `by`는 컬럼 이름 하나 또는 리스트다. Series 키와 `Series.groupby`는 없다.
- 키가 아닌 컬럼이 모두 값 컬럼이 된다.
- `g["c"]`, `g[["c", "d"]]`로 값 컬럼을 고른다. 키 컬럼은 값 컬럼으로 고를 수 없다.

### §7.2 결과와 그룹 순서
- **결과는 DataFrame이다.**
  - 컬럼은 키 컬럼(`by` 순서)이 먼저 오고, 그 다음 집계 컬럼이다.
  - **키는 단일 키든 다중 키든 결과의 일반 컬럼이 된다.** 결과의 Index는 RangeIndex다. **[pandas와 다름]**
    - 이유: 키를 Index로 삼으면 "float와 bool은 Index로 쓸 수 없다"(§1.7)와 충돌한다.
  - `value_counts`(§4.3)와 같은 모양이다.
- **그룹 순서**
  - `sort=True`(기본값): 키의 오름차순이다.
    - int와 float는 수 순서다. NaN은 +inf 뒤에 온다 (§1.3). `0.0`과 `-0.0`은 같은 키다.
    - string은 코드포인트 순서다. bool은 `False < True`다.
    - 다중 키는 키를 앞에서부터 차례로 비교한다.
    - NA 키 그룹(`dropna=False`일 때)은 맨 뒤에 온다. 다중 키에서는 키마다 이 규칙을 적용한다.
  - `sort=False`: 그룹이 처음 나타난 순서다.
- **NA 키**
  - `dropna=True`(기본값): 키 중 하나라도 NA인 행을 버린다. NaN은 값이므로 버리지 않는다. **[pandas와 다름]**
  - `dropna=False`: NA 키도 그룹이 된다.

### §7.3 집계 메서드
- 인자는 리덕션과 같다 (§6.1).
- `sum`, `mean`, `min`, `max`, `std`, `var`
  - 결과 dtype과 할 수 없는 조합(예: string의 `mean`)은 §6과 같다.
- `count()`: 값 컬럼마다 NA가 아닌 값의 개수다.
- `size()`: 그룹의 행 수다. 결과 컬럼 이름은 `size`다.
- `first`, `last`: 그룹에서 NA가 아닌 처음(마지막) 값이다.
  - `skipna=False`이면 처음(마지막) 행의 값이고, NA일 수도 있다.
  - 컬럼 dtype을 그대로 유지한다.
- 값이 모두 NA인 그룹: `sum`은 0, `count`는 0이고, 나머지는 NA다.

### §7.4 `agg`
- 함수는 이름(문자열)으로만 준다. `sum mean min max count std var first last` 중 하나다.
  - `g.agg("sum")`: 모든 값 컬럼에 적용한다.
  - `g.agg({"f": "sum", "i": "max"})`: 컬럼마다 함수 하나다.
  - `g.agg(total=("f", "sum"), n=("i", "count"))`: 결과 컬럼 이름을 직접 정한다. 같은 컬럼에 함수 여러 개를 적용할 때 쓴다.
- 함수 리스트는 에러다 (MultiIndex가 없다). **[pandas와 다름]**
- 결과 컬럼 이름이 키 컬럼과 겹치면 에러다.

### §7.5 계산 순서
- **float 합은 그룹마다 numpy pairwise로 더한다** (§6.5). **[pandas와 다름]**
  - 행을 그룹별로 모은 뒤(counting sort), 각 그룹의 행을 원래 순서대로 더한다.
  - 그래서 한 그룹의 합은 그 행들만 골라낸 Series의 `sum()`과 비트까지 같다.
  - pandas는 행 순서대로 Kahan 보정 합을 쓴다.

---

## §8. 정렬 (sort)

### §8.1 함수
```
Series.sort_values(*, descending=False, stable=True, na_position="last", ignore_index=False)
Series.sort_index(*, descending=False, stable=True, ignore_index=False)
DataFrame.sort_values(by, /, *, descending=False, stable=True, na_position="last", ignore_index=False)
DataFrame.sort_index(*, descending=False, stable=True, ignore_index=False)
```
- **`descending`**: 방향이다. 배열 영역의 `sort`, `argsort`와 같은 이름이다. pandas의 `ascending=True`가 `descending=False`다. **[pandas와 다름]**
- **`descending`과 `na_position`**: 둘 다 값 하나(모든 키에 적용) 또는 키 수만큼의 리스트를 받는다. **[pandas와 다름]**
  - pandas는 `na_position`에 리스트를 받지 않는다.
  - `descending`의 값은 `True`나 `False`다. 그 밖의 값은 TypeError다 (§1.1).
  - `na_position`은 `"first"` 또는 `"last"`다.
  - 리스트의 길이가 키 수와 다르면 ValueError다.
- **`stable`**: `True`나 `False`를 받는다. 다른 값은 TypeError다.
  - frame의 정렬은 언제나 안정 정렬이라서 `stable=False`여도 결과가 같다. 배열 영역과 같은 규칙이다: `stable=False`는 "안정이 아니어도 된다"는 허락일 뿐이다.
- **없는 인자**
  - `kind`: 정렬은 항상 안정 정렬이다. `stable`을 쓴다.
  - `key=`: Python 함수를 받는 인자다.
  - `sort_index`의 `na_position`: Index에는 NA가 없다.
  - `sort_index`의 `level`, `sort_remaining`: MultiIndex가 없다.

### §8.2 순서
- int와 float는 수 순서다.
  - **NaN은 +inf보다 큰 값이다.** 그래서 내림차순에서는 맨 앞에 온다. **[pandas와 다름]** pandas는 NaN을 NA처럼 `na_position`으로 놓는다.
  - `0.0`과 `-0.0`은 같은 값이다.
- string은 코드포인트 순서다. bool은 `False < True`다.
- NA는 `na_position`이 정하는 곳에 온다. 방향과 상관없다.
- **안정성**: 키가 같은 행은 원래 순서를 지킨다. 오름차순과 내림차순 모두 그렇다.
  - 다중 키는 앞의 키부터 차례로 비교한 순서다.
- **결과**: 행이 그 순서로 재배열되고, 라벨도 행을 따라간다(Index). `ignore_index=True`이면 RangeIndex다.

### §8.3 알고리즘
- 숫자와 bool은 64비트 키에 대한 안정 LSD 기수 정렬이다.
- string은 고유값의 순위로 바꾼 뒤 안정 counting sort를 한다.
- 다중 키는 마지막 키부터 차례로 안정 정렬한다.
- 모든 키가 같은 자릿수를 건너뛰는 일은 하지 않는다. 데이터를 보고 경로를 고르지 않기 위해서다.

---

## §9. merge와 concat

### §9.1 merge
```
merge(left, right, /, *, how="inner", on=None, left_on=None, right_on=None, sort=False, suffixes=("_x", "_y"))
DataFrame.merge(right, /, *, how="inner", on=None, left_on=None, right_on=None, sort=False, suffixes=("_x", "_y"))
```
- `df.merge(right, ...)`는 `merge(df, right, ...)`와 같다.
- `how`는 `"inner"`, `"left"`, `"right"`, `"outer"`다.
- **키**
  - 키는 꼭 준다. `on`(양쪽에 같은 이름) 또는 `left_on`과 `right_on`(같은 개수)이다. **[pandas와 다름]**
    - pandas는 키를 생략하면 공통 컬럼 전체를 키로 쓴다.
  - 키 컬럼의 dtype은 양쪽이 같아야 한다. 예를 들어 int64와 float64는 에러다. **[pandas와 다름]**
- **매칭 규칙** (§1.6)
  - **NA 키는 아무것과도 매칭되지 않는다** (SQL, Polars와 같다). **[pandas와 다름]**
  - NaN은 값이라서 NaN과 매칭된다. `0.0`은 `-0.0`과 매칭된다.
- **빼는 기능**: `left_index`, `right_index`, `indicator`, `validate`, `how="cross"`, `copy`
- **결과 컬럼**
  - 왼쪽 컬럼(왼쪽 순서)이 먼저 오고, 그 다음 오른쪽 컬럼(오른쪽 순서)이다.
  - `on`으로 조인하면 키 컬럼은 하나로 합친다.
    - 위치는 왼쪽에서의 위치다.
    - 값은 왼쪽 행이 있으면 왼쪽 값, 없으면(오른쪽에만 있는 행) 오른쪽 값이다.
  - `left_on`/`right_on`으로 조인하면 양쪽 키 컬럼이 모두 남는다.
  - 이름이 겹치는 컬럼(합친 키는 제외)에는 `suffixes`를 붙인다. 기본값은 `_x`, `_y`다.
    - 접미사를 붙인 뒤에도 이름이 겹치면 에러다.
  - 매칭되지 않은 쪽의 값은 NA다. dtype은 그대로다.
- **결과 행 순서** **[pandas와 다름]**
  - `inner`, `left`, `outer`: left의 행 순서를 따른다.
    - 한 키에 여러 행이 매칭되면 그 안에서는 right의 순서를 따른다.
    - `outer`에서 매칭되지 않은 right 행은 right 순서대로 끝에 붙인다.
  - `right`: 위 규칙을 좌우 대칭으로 적용한다.
  - `sort=True`이면 키로 안정 정렬한다. 오름차순이고 NA는 맨 뒤다 (§8.2).
    - `on`이면 합친 키로 정렬한다.
    - `left_on`/`right_on`이면 왼쪽 키로 정렬하고, 왼쪽 행이 없으면 오른쪽 키 값을 쓴다.
- **결과 Index**: RangeIndex다.

### §9.2 concat
```
concat(objs, /, *, join="outer", ignore_index=False)
```
- 행 방향만 잇는다. `axis`는 없다. 모양은 배열의 `concat(arrays, /, *, axis=0)`과 같다.
- `objs`는 DataFrame들이거나 Series들이다. 섞으면 에러이고, 비어 있어도 에러다.
- **컬럼**
  - `join="outer"`: 처음 나타난 순서로 합친다. 어떤 DataFrame에 없는 컬럼은 그 행들이 NA다.
  - `join="inner"`: 모두에 있는 컬럼만, 첫 DataFrame의 순서로 둔다.
- **같은 이름의 컬럼은 dtype이 같아야 한다.** Series끼리도 dtype이 같아야 한다. **[pandas와 다름]**
- **라벨**
  - 라벨을 이어 붙인다. 중복이 생겨도 된다.
  - int 라벨과 string 라벨은 섞을 수 없다(에러). RangeIndex는 int64로 친다.
  - Index와 Series의 이름은 모두 같으면 유지하고, 아니면 `None`이다.
- `ignore_index=True`이면 RangeIndex다.
- `keys`, `levels`, `names`, `verify_integrity`, `sort`, `copy`는 없다.

---

## §10. read_csv
```
read_csv(path, /, *, sep=",", header=True, names=None, usecols=None, dtype=None, na_values=None, nrows=None, index_col=None)
```

### §10.1 인자
- `sep`: 구분 문자 하나다. 따옴표나 줄바꿈 문자는 쓸 수 없다.
- `header`
  - `True`: 첫 레코드가 컬럼 이름이다.
  - `False`: `names=`가 꼭 필요하다 (list의 list에 `columns=`를 요구하는 것과 같다, §2.5).
- `names`: 컬럼 이름이다. `header=True`와 함께 주면 파일의 이름을 바꾼다. 개수가 같아야 한다.
- `usecols`: 읽을 컬럼 이름의 리스트다. 결과는 이 리스트의 순서를 따른다.
- `dtype`: dtype 하나(모든 컬럼) 또는 `{컬럼: dtype}`이다.
  - 지정한 컬럼은 추론하지 않고, `astype`의 엄격한 파싱(§3.5)으로 읽는다. 읽을 수 없는 값은 에러이고, 줄 번호를 알려 준다.
- `na_values`: 빈 필드 말고도 NA로 읽을 문자열의 리스트다.
- `nrows`: 데이터 행을 최대 몇 개까지 읽을지다.
- `index_col`: Index가 될 컬럼 이름 하나다.
  - int64나 string이어야 하고, NA가 없어야 한다 (§1.7).
  - Index 이름은 컬럼 이름이다. 이 컬럼은 컬럼 목록에서 빠진다.
- `encoding`, `skiprows`, `comment`, `quotechar`, `thousands`, `decimal`, `parse_dates`, `converters` 등 나머지 옵션은 없다.

### §10.2 파일 형식
- UTF-8만 받는다. 맨 앞의 BOM은 건너뛴다. string 값은 올바른 UTF-8이어야 한다.
- **줄바꿈**은 `\n` 또는 `\r\n`이다. 파일 끝의 마지막 줄바꿈은 빈 줄을 만들지 않는다.
  - `\n` 바로 앞이 아닌 `\r`은 값의 일부다. 예: `x\r,y`의 첫 필드는 `x\r`이다.
  - 닫는 따옴표 뒤의 `\r`은 에러다.
- **따옴표**는 RFC 4180을 따른다.
  - 큰따옴표로 감싼 필드 안에서는 구분 문자와 줄바꿈도 값이다. `""`는 `"`다.
  - 닫는 따옴표 뒤에는 구분 문자나 줄바꿈이나 파일 끝만 올 수 있다.
  - 따옴표로 감싸지 않은 필드 안의 `"`는 에러다.
- **모든 행의 필드 개수가 컬럼 수와 같아야 한다.** 다르면 에러이고, 줄 번호를 알려 준다. **[pandas와 다름]**
- **완전히 빈 줄은 빈 필드 하나짜리 행이다.** **[pandas와 다름]**
  - 컬럼이 하나인 파일에서는 NA 행이 되고, 컬럼이 여럿이면 필드 개수 에러다.
  - pandas는 빈 줄을 건너뛴다.
- 헤더의 컬럼 이름은 고유해야 한다 (§1.7). pandas는 `a.1`처럼 이름을 바꾼다.

### §10.3 NA
**[pandas와 다름]**
- 따옴표 없는 빈 필드와 `na_values`에 있는 문자열만 NA다.
- 따옴표로 감싼 빈 필드(`""`)는 빈 문자열이다. 그래서 `""`와 NA를 구분할 수 있다.
- 텍스트 `NaN`, `nan`, `inf`는 float 컬럼에서 NaN 값과 inf 값이다. NA가 아니다 (§1.3).
- pandas는 `NA`, `null`, `NaN` 같은 목록을 모두 결측으로 읽는다.

### §10.4 타입 추론
- 컬럼마다, NA가 아닌 모든 필드를 본다.
- 따옴표는 필드의 경계일 뿐이다. `"12"`는 `12`와 같다.
- **추론 순서**
  1. 모든 필드가 int 문법(§3.5)이고 int64 범위 안이면 **int64**다.
  2. 그렇지 않고 모든 필드가 float 문법이면 **float64**다.
     - 단, int 문법인 필드 중에 int64 범위를 넘거나 float64로 정확히 표현할 수 없는 값이 있으면 string이다. 값이 조용히 바뀌지 않게 하기 위해서다.
  3. 그렇지 않고 모든 필드가 `True` 또는 `False`이면 **bool**이다.
  4. 그 밖에는 **string**이다.
- 공백이 들어간 필드(예: `" 1"`)는 숫자가 아니다 (§3.5). **[pandas와 다름]**
- 모두 NA인 컬럼은 float64다 (§2.2).

---

## §11. repr
- **배치**: pandas와 같다.
  - 값은 오른쪽 정렬하고, Index를 왼쪽에 둔다.
  - Series는 아래에 `Name: v, dtype: int64` 줄을 붙인다.
- **생략**
  - 60행이 넘으면 앞 5행과 뒤 5행만 보여 준다.
    - DataFrame은 끝에 `[100 rows x 3 columns]`를 붙인다.
    - Series는 끝 줄을 `Length: 100, dtype: int64`로 쓴다.
  - 20열이 넘으면 앞 10열과 뒤 10열만 보여 준다.
  - 터미널 폭에 맞춰 줄을 바꾸는 기능은 없다. **[pandas와 다름]**
  - 열을 생략할 때, 숫자 컬럼 헤더 앞의 공백은 화면에 나오는 열 자신의 dtype으로 정한다. **[pandas와 다름]**
    - pandas는 생략 전 frame에서 같은 위치에 있던 열의 dtype을 쓴다. pandas의 버그라서 따르지 않는다.
- **float 표기**: pandas 규칙을 따른다.
  - 유효숫자는 6자리다.
  - 컬럼 안에서 소수 자릿수를 맞춘다.
  - 값이 너무 크거나 작으면 컬럼 전체를 지수 표기로 바꾼다.
- **결측 표기**: NA는 모든 dtype에서 `<NA>`로, NaN은 `NaN`으로 쓴다.
  - pandas 3의 `str` dtype은 결측을 `NaN`으로 쓴다. **[pandas와 다름]**
- **int 표기**: NA가 있어도 numpy int64 규칙을 따른다.
  - pandas의 nullable `Int64`는 음수를 한 칸 넓게 쓴다. 예: `-1`과 `<NA>`가 섞이면 `   -1`이 된다.
- **dtype 이름**: `bool`, `int64`, `float64`, `string` (§1.2)
- Index의 이름은 pandas처럼 보여 준다 (§5.1).

---

## §12. 스레드
- 스레드 수는 라이브러리 공통의 `sd.set_num_threads(n, /)`, `sd.get_num_threads()`로 정한다 (API.md §3). 스레드풀은 모든 영역이 함께 쓴다.
  - `n`이 1이면 스레드풀을 쓰지 않는다.
  - 환경 변수는 두지 않는다.
- **결과는 스레드 수와 무관하게 비트까지 같다** (API.md §3). 이 영역에서는 float 합, 행 순서, factorize 코드(처음 나타난 순서), 에러 메시지까지 같다.
- **에러**도 스레드 수와 상관없다. 잘못된 값이 여러 개면 항상 첫 번째(행 순서, CSV는 파일 순서)를 알린다.
- **GIL**: 작업량이 임계값(131,072 원소) 이상이면 스레드 수와 상관없이 커널을 실행하는 동안 GIL을 놓는다. 그래서 `sd.set_num_threads(1)`이어도 다른 Python 스레드가 함께 돈다.
- **동시 대입**: 연산은 호출한 시점의 값을 본다(스냅샷).
  - 대입(`s[i] = v`, `s.iloc[...] = ...`)은 GIL을 쥔 채로 실행한다. 그래서 다른 연산은 대입 전이나 후의 값만 본다.
  - 대입은 먼저 자기만의 Column 객체를 만들고 거기에 쓴다. 진행 중인 연산은 옛 Column 객체를 잡고 있어서 버퍼가 공유 상태이므로, 대입이 Copy-on-Write 규칙(§5.7)에 따라 버퍼를 복사한다.

---

## §13. 미확정
- **`datetime64` 라벨**: 처음 결정은 Index에 쓸 수 있는 dtype으로 `int64`, `string`과 함께 `datetime64`를 꼽았다. 이 영역에는 아직 `datetime64` dtype이 없다 (§1.2). 지금 Index는 int64와 string만 받는다 (§1.7, §5.1).
