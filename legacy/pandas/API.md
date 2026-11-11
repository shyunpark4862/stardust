# stardust-pandas API 의미론

DESIGN.md의 결정(NA 하나, NaN은 값, Copy-on-Write)에서 따라 나오는 동작 규칙이다.
- 3라운드(S1–S8)에서 사용자가 모두 권장안으로 결정했다 (2026-11-10).
- §6의 정렬 라벨 순서는 4라운드 A6에서 정했다.
- §9–§11은 M1을 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §12와 §11의 C7은 M2를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §13은 M3를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §14는 M4를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §15는 M5를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- pandas와 다른 점은 **[pandas와 다름]**으로 표시한다.

---

## 1. NaN (S1)
NaN은 결측값이 아니라 평범한 float 값이다 (DESIGN §2.1).

- **리덕션**: `sum`, `mean` 등에 NaN이 섞이면 결과가 NaN이 된다. **[pandas와 다름]**
  - `skipna`는 NA만 건너뛴다.
- **`min`, `max`**: NaN을 전파한다.
- **정렬**: NaN은 +inf 뒤에 놓는다.
  - NA의 위치는 `na_position`으로 따로 정한다.
- **키 비교** (groupby, unique, value_counts, duplicated, merge)
  - NaN끼리는 같은 키다.
  - `-0.0`과 `0.0`은 같은 키다.
  - 해시 테이블에 넣기 전에 키를 정규화해서 처리한다.
  - 이 규칙은 stardust-numpy와 다르다. sdnp의 `unique`는 NaN을 서로 다른 값으로 취급한다 (sdnp DESIGN §2.4 N3).

## 2. NA (S2)
- **비교와 산술**: `NA`가 들어간 연산의 결과는 `NA`다.
- **논리 연산**: Kleene 3값 논리를 쓴다.
  - `True | NA == True`
  - `False & NA == False`
  - 나머지 조합은 `NA`다.
- **필터**: 마스크 값이 NA인 행은 False로 취급해서 버린다.
- **`bool(NA)`**: TypeError를 낸다.
- **값이 모두 NA인 경우의 리덕션** (빈 컬럼도 같다)
  - `sum`은 0이다.
  - `count`는 0이다.
  - `mean`, `min`, `max`, `std`, `var`는 `NA`다.
- **`std`, `var`의 기본 자유도**: `ddof=1`
  - stardust-numpy는 `correction=0`이 기본이라서 sdnp와 다르다.

## 3. 정수 연산 (S3)
- `int / int`의 결과는 float64다 (true division).
- 정수 오버플로는 wrap한다.
- **`int // 0`과 `int % 0`의 결과는 `NA`다.** **[pandas와 다름]**
- `int ** 음수`의 결과도 `NA`다 (§12).
- float를 0으로 나누면 IEEE 754를 따른다 (`inf`, `nan`).
  - 결과가 `NA`가 되지 않는다.

## 4. groupby (S4)
- **결과 순서**: `sort=True`가 기본이다. 결과를 키 오름차순으로 정렬한다.
  - `sort=False`를 주면 키가 처음 나타난 순서를 따른다.
- **NA 키**: `dropna=True`가 기본이다. NA 키 그룹은 결과에서 버린다.
  - `dropna=False`를 주면 NA 키도 그룹이 된다.
- **키의 위치**
  - 단일 키: 결과의 Index가 된다.
  - 다중 키: 결과의 일반 컬럼이 된다 (DESIGN §2.2).

## 5. merge (S5)
- **결과 행 순서** **[pandas와 다름]**
  - `inner`, `left`, `outer`: left의 행 순서를 따른다.
    - 한 키에 여러 행이 매칭되면 그 안에서는 right의 순서를 따른다.
    - `outer`에서 매칭되지 않은 right 행은 right 순서대로 끝에 붙인다.
  - `right`: 위 규칙을 좌우 대칭으로 적용한다.
  - `sort=True`를 주면 키 기준으로 정렬한다.
- **NA 키**: NA는 아무것과도 매칭되지 않는다 (SQL, Polars와 같다). **[pandas와 다름]**
  - NaN은 값이라서 NaN끼리는 매칭된다 (§1).
- **이름이 겹치는 컬럼**: 접미사 `_x`, `_y`를 붙인다.

## 6. 라벨 (S6)
- **Index에 쓸 수 있는 dtype**: `int64`, `string`, `datetime64`, 그리고 `RangeIndex`
  - float와 bool은 Index로 쓸 수 없다.
  - Index에는 NA를 넣을 수 없다.
- **중복 라벨**: 허용한다.
  - 정렬(alignment)이 필요한 연산에서 라벨이 고유하지 않으면 에러를 낸다. **[pandas와 다름]**
  - 두 Index가 같으면 위치대로 계산하므로 중복이 있어도 된다.
- **DataFrame 컬럼 이름**: 고유한 `str`만 허용한다.
- **정렬(alignment) 결과의 라벨 순서**: merge outer와 같은 규칙을 쓴다 (§5). **[pandas와 다름]**
  - left 라벨 순서를 먼저 따르고, right에만 있는 라벨을 right 순서대로 뒤에 붙인다.
  - pandas는 두 Index가 다르면 합집합을 정렬한다.
  - 4라운드 A6에서 사용자가 결정했다 (2026-11-10).

## 7. `[]` (S7)
- **DataFrame**
  - `df["a"]`: 컬럼 하나
  - `df[["a", "b"]]`: 컬럼 여러 개
  - `df[bool Series]`: 행 필터
  - `df[1:3]`처럼 `[]`로 행을 자르는 것은 지원하지 않는다. `.iloc`을 쓴다. **[pandas와 다름]**
- **Series**
  - `s[x]`는 라벨 기준이다 (`.loc`과 같다).
  - 위치 기준 접근은 `.iloc`으로만 한다.

## 8. 스칼라, 경고, 에러 (S8)
- **리덕션 결과**: Python 스칼라를 돌려준다. 원소 하나를 꺼낼 때도 마찬가지다.
- **결측 스칼라**: 싱글턴 `sdpd.NA`
- **경고**: 일절 내지 않는다. 문제가 있으면 에러를 낸다.
- **대입으로 dtype이 바뀌는 경우**: 에러를 낸다.
  - 예: int 컬럼에 `1.5`를 대입하는 것

## 9. repr
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
    - pandas는 생략 전 frame에서 같은 위치에 있던 열의 dtype을 쓰는 버그가 있다. 사용자가 이 버그를 따라 하지 않기로 했다 (2026-11-11).
- **float 표기**: pandas 규칙을 따른다.
  - 유효숫자는 6자리다.
  - 컬럼 안에서 소수 자릿수를 맞춘다.
  - 값이 너무 크거나 작으면 컬럼 전체를 지수 표기로 바꾼다.
- **결측 표기**: NA는 모든 dtype에서 `<NA>`로, NaN은 `NaN`으로 쓴다.
  - pandas 3의 `str` dtype은 결측을 `NaN`으로 쓴다. **[pandas와 다름]**
- **int 표기**: NA가 있어도 numpy int64 규칙을 따른다.
  - pandas의 nullable `Int64`는 음수를 한 칸 넓게 쓴다. 예: `-1`과 `<NA>`가 섞이면 `   -1`이 된다.
- **dtype 이름**: `bool`, `int64`, `float64`, `string`
  - NA가 있어도 dtype 이름은 바뀌지 않는다. pandas는 nullable dtype을 쓰면 `Int64`, `boolean` 등으로 쓴다.

## 10. 생성과 변환
- **list에서 dtype 추론**
  - `None`과 `sdpd.NA`는 NA다.
  - `float('nan')`은 NaN 값이다.
  - bool만 있으면 `bool`, int만 있으면 `int64`, float만 있거나 int와 float가 섞이면 `float64`, str만 있으면 `string`이다.
  - 빈 리스트나 값이 모두 NA인 리스트는 `float64`다.
  - 다음 경우는 에러다.
    - bool이 다른 타입과 섞인 경우 (예: `[True, 1]`) **[pandas와 다름]**
    - str이 다른 타입과 섞인 경우 **[pandas와 다름]**
    - int64 범위를 넘는 정수
    - int와 float가 섞일 때 2^53을 넘어 float64로 정확히 표현할 수 없는 int **[pandas와 다름]**
- **생성자의 `dtype=`**: 손실 없는 숫자 변환만 허용한다.
  - int → `float64`: 2^53을 넘어 정확히 표현할 수 없으면 에러다.
  - 정수 값만 가진 float → `int64`: `[1.0, 2.0]`은 되고 `[1.5]`나 NaN은 에러다.
  - bool과 숫자 사이, 문자열과 다른 타입 사이의 변환은 에러다. 이런 변환은 `astype`으로 한다.
  - `dtype`은 `"bool"`, `"int64"`, `"float64"`, `"string"` 또는 Python 타입 `bool`, `int`, `float`, `str`로 준다.
- **numpy 배열에서 만들기**: 항상 복사한다.
  - numpy 배열은 밖에서 수정될 수 있어서, 메모리를 공유하면 CoW가 깨진다.
  - dtype 변환
    - int8~int32, uint8~uint32 → `int64`
    - uint64 → `int64`. 범위를 넘으면 에러다.
    - float16, float32 → `float64`
    - U 문자열 → `string`
    - object 배열 → list와 같은 규칙
  - float 배열의 NaN은 NaN 값이다.
- **DataFrame 생성 입력**
  - `{이름: list | 1D numpy 배열 | Series}` 형태의 dict
    - 길이가 다르면 에러다.
  - 행의 list
    - dict의 list: 키가 없는 칸은 NA다. 컬럼 순서는 키가 처음 나타난 순서다.
    - list의 list: `columns=`가 꼭 필요하다.
  - 2D numpy 배열: `columns=`가 꼭 필요하다.
  - 행 단위 입력은 컬럼으로 바꾼 뒤 list와 같은 규칙으로 dtype을 추론한다.
- **`Series.name`**: `str` 또는 `None`만 쓸 수 있다. **[pandas와 다름]**
- **`to_numpy(*, dtype=None, copy=False, na_value=<없음>)`**
  - NA가 없으면 복사 없이 읽기 전용 뷰를 돌려준다. `copy=True`이면 쓰기 가능한 복사본을 돌려준다.
  - NA가 있으면 `na_value`가 필요하다. 없으면 에러다.
  - `na_value`가 그 dtype에 들어가지 않으면 에러다.
    - 예: int 컬럼에 `na_value=np.nan`. 이때는 `dtype="float64"`도 함께 준다.
  - `string`은 Python str을 담은 object 배열로 돌려준다.
  - `DataFrame.to_numpy`
    - 모든 컬럼의 dtype이 같으면 2D 배열을 돌려준다.
    - int64와 float64가 섞이면 float64로 만든다. 2^53 규칙은 위와 같다.
    - 그 밖의 조합은 에러다.

## 11. 시그니처 일관성
- **원칙**: pandas 시그니처가 일관되지 않으면 일관되게 바꾼다 (사용자 지시 2026-11-11).
  - 마일스톤을 시작할 때마다 그 범위의 시그니처를 점검한다.
  - 바꿀 안은 Claude가 제안하고 사용자가 정한다.
- **C1. 별칭을 두지 않는다.**
  - `to_list`만 둔다 (`tolist` 없음).
  - `isna`, `notna`만 둔다 (`isnull`, `notnull` 없음).
  - `merge`는 함수 `sdpd.merge(left, right)`와 메서드 `df.merge(right)`를 모두 둔다.
  - `concat`은 함수만 둔다.
- **C2. 주 인자만 위치 인자로 받는다.** 나머지는 모두 키워드 전용이다.
  - 주 인자의 예: `Series(data)`, `DataFrame(data)`, `head(n)`, `tail(n)`, `astype(dtype)`, `fillna(value)`, `sort_values(by)`, `groupby(by)`, `duplicated(subset)`, `merge(right)`, `concat(objs)`
- **C3. 리덕션 인자는 Series, DataFrame, GroupBy에서 같다.**
  - `sum`, `mean`, `min`, `max`: `(*, skipna=True)`
  - `std`, `var`: `(*, skipna=True, ddof=1)`
  - `count`: 인자가 없다.
  - `axis`(행 방향 리덕션), `min_count`, `numeric_only`는 없다.
- **C4. 우리 설계에서 의미가 없는 인자는 두지 않는다.**
  - `copy`: CoW를 쓰므로 필요 없다. `to_numpy`만 예외다.
  - `inplace`
  - `level`: MultiIndex가 없다.
  - `kind`: 정렬은 항상 안정 정렬이다.
  - `errors`: 항상 에러를 낸다.
  - `engine`
  - Series 메서드의 `axis`
- **C5. `keep`은 `"first"`, `"last"`, `"none"`이다.** pandas의 `keep=False`가 `"none"`이다.
- **C7. 연산은 연산자로만 한다.** `add`, `sub`, `mul`, `div`, `truediv`, `eq`, `abs()` 같은 메서드는 두지 않는다.
  - 이 메서드들은 연산자의 별칭이고, `div`/`divide`/`truediv`처럼 서로도 별칭이다.
  - 절댓값은 `abs(s)`로 구한다.
  - `fill_value`가 필요하면 먼저 `fillna`를 한 뒤 연산한다.
- **C6. 반환 타입**
  - `Series.unique()`는 Series를 돌려준다.
    - 이름은 유지한다. Index는 RangeIndex다. 값은 처음 나타난 순서다.
  - `DataFrame.value_counts()`는 키 컬럼과 `count` 컬럼을 가진 DataFrame을 돌려준다.
    - 다중 키 groupby와 같은 방식이다 (§4).

## 12. 원소별 연산 (M2)
- **피연산자**
  - Series와 Series: M2에서는 두 Index가 같아야 한다. 다르면 정렬이 필요한데, 정렬은 M4에서 만든다.
  - Series와 스칼라, 스칼라와 Series
    - 스칼라는 Python 스칼라, numpy 스칼라, `NA`다.
    - int64 범위를 넘는 int 스칼라는 에러다.
  - DataFrame과 스칼라
  - DataFrame과 DataFrame
    - 컬럼을 이름으로 짝짓는다.
    - M2에서는 컬럼 집합과 Index가 같아야 한다. 결과는 왼쪽의 컬럼 순서를 따른다.
  - DataFrame과 Series는 지원하지 않는다. **[pandas와 다름]**
  - 결과 이름: Series끼리 연산할 때 이름이 같으면 유지하고, 다르면 `None`이다. 스칼라와 연산하면 유지한다 (pandas와 같다).
- **산술**: `+`, `-`, `*`, `/`, `//`, `%`, `**`, 단항 `-`와 `+`, `abs(s)`
  - 결과 dtype
    - int64와 int64는 int64다. 단, `/`는 float64다.
    - int64나 float64가 섞이면 float64다. int는 float64로 바꾼 뒤 계산한다.
    - string `+` string은 string이다 (이어 붙이기). string의 다른 산술은 에러다.
    - bool의 산술은 에러다. 숫자가 필요하면 `astype("int64")`를 쓴다. **[pandas와 다름]**
  - `//`와 `%`는 floor 규칙을 따른다 (Python, numpy와 같다). `%`의 결과는 나누는 수의 부호를 따른다.
  - int 규칙 (§3에 더해)
    - `int ** 음수`는 NA다. **[pandas와 다름]**
    - 오버플로는 wrap한다. 예: `INT64_MIN // -1`과 `abs(INT64_MIN)`은 `INT64_MIN`이다.
  - float는 IEEE 754를 따른다.
  - NA가 들어가면 결과는 NA다. 예외가 없다. **[pandas와 다름]**
    - pandas는 `NA ** 0`과 `1 ** NA`를 1로 만든다.
- **비교**: `==`, `!=`, `<`, `<=`, `>`, `>=`
  - 결과는 bool이다. 어느 한쪽이 NA인 행은 NA다.
  - int와 float의 비교는 반올림 없이 수학적으로 정확하게 한다. **[pandas와 다름]**
    - 예: `2**53 + 1 == 2.0**53`은 False다. pandas는 int를 float로 바꿔서 비교하므로 True다.
  - NaN은 IEEE 754를 따른다. `NaN == NaN`은 False, `NaN != NaN`은 True다.
  - string은 코드포인트 순서로 비교한다. UTF-8 바이트 순서와 같고, Python str의 비교와도 같다.
  - bool은 bool끼리만 비교한다. `False < True`다.
  - 비교할 수 없는 dtype끼리 비교하면 `==`도 에러다 (예: string과 int). **[pandas와 다름]**
- **논리**: `&`, `|`, `^`, `~`는 bool에만 쓴다.
  - `&`와 `|`는 Kleene 규칙을 따른다 (§2).
  - `^`와 `~`는 NA가 들어가면 NA다.
  - int의 비트 연산은 지원하지 않는다. **[pandas와 다름]**
- **`astype(dtype)`**
  - DataFrame은 dtype 하나나 `{컬럼: dtype}` dict를 받는다.
  - NA는 그대로 NA다.
  - int → float: 반올림한다.
  - float → int: 0 쪽으로 자른다. NaN, inf, int64 범위를 넘는 값은 에러다.
  - bool → 숫자: 0과 1이다.
  - 숫자 → bool: 0이 아니면 True다. NaN도 True다.
  - → string: Python `str()`과 같다. 예: `1.0` → `"1.0"`, `1e20` → `"1e+20"`, NaN → `"nan"`
  - string → int64, float64: 엄격하게 파싱한다.
    - 받는 것: 부호, 숫자, 소수점, 지수. float는 `nan`, `inf`도 받는다.
    - 공백과 `_`는 에러다.
  - string → bool: `"True"`와 `"False"`만 받고 나머지는 에러다. **[pandas와 다름]**
- **`isna()`, `notna()`**: NA만 결측으로 본다. NaN은 결측이 아니다.
- **`fillna(value)`**
  - 값은 컬럼 dtype으로 손실 없이 바뀌어야 한다. 규칙은 생성자 `dtype=`와 같다 (§10).
    - 예: int 컬럼에 `2.0`은 되고 `1.5`는 에러다.
  - DataFrame은 `{컬럼: 값}` dict도 받는다.
- **`dropna`**
  - `Series.dropna(*, ignore_index=False)`
  - `DataFrame.dropna(*, how="any", subset=None, ignore_index=False)`
    - `how="all"`은 subset 안의 값이 모두 NA인 행만 지운다.
  - `axis`(열 지우기)와 `thresh`는 없다.
- **마스크 필터**: `s[mask]`, `df[mask]`
  - mask는 bool Series다. M2에서는 Index가 같아야 한다.
  - mask가 NA인 행은 버린다 (§2).
  - 결과의 Index는 항상 남은 행 라벨의 `Index`(int64)다. 데이터에 따라 RangeIndex가 되는 일은 없다.
- **`Index` (M2)**: 불변 int64 Column 하나를 감싼다. repr과 `equals`만 있다.
  - 라벨 조회 엔진, `.loc`, 정렬은 M4에서 만든다.

## 13. 중복과 고유값 (M3)
- **키가 같다는 것** (§1에 더해)
  - NaN은 값이다. NaN끼리는 같은 키다. `nunique`와 `value_counts`에서 NaN도 하나의 키로 센다. **[pandas와 다름]**
    - pandas는 NaN을 결측으로 보고 기본으로 버린다.
  - `0.0`과 `-0.0`은 같은 키다. 비트가 서로 다른 NaN들도 같은 키다.
  - NA끼리는 같은 키다. 예외는 merge 하나다 (§5).
  - `dropna=`는 NA만 버린다.
- **대표값**: 같은 키로 묶인 값 중에서 먼저 나온 값을 그대로 보여 준다 (pandas와 같다).
  - 예: `[-0.0, 0.0]`의 unique는 `[-0.0]`이다.
- **시그니처** (C2: 주 인자만 위치 인자로 받는다)
  - `Series.unique()`
    - Series를 돌려준다 (C6). 값은 처음 나타난 순서이고 NA도 포함한다.
  - `Series.nunique(*, dropna=True)`
    - int를 돌려준다.
  - `DataFrame.nunique(*, dropna=True)`
    - 컬럼별 개수를 dict로 돌려준다. `dtypes`처럼 M4까지만 임시로 이렇게 둔다.
  - `Series.duplicated(*, keep="first")`
  - `DataFrame.duplicated(subset=None, *, keep="first")`
  - `Series.drop_duplicates(*, keep="first", ignore_index=False)`
  - `DataFrame.drop_duplicates(subset=None, *, keep="first", ignore_index=False)`
  - `Series.value_counts(*, normalize=False, sort=True, ascending=False, dropna=True)`
  - `DataFrame.value_counts(subset=None, *, normalize=False, sort=True, ascending=False, dropna=True)`
  - `keep`은 `"first"`, `"last"`, `"none"`이다 (C5).
  - `bins`는 없다.
- **`value_counts`는 항상 DataFrame을 돌려준다.** **[pandas와 다름]**
  - 이유: pandas처럼 값을 Index로 삼으면, float와 bool은 Index로 쓸 수 없다는 §6의 규칙과 충돌한다.
  - 컬럼
    - 키 컬럼과 `count` 컬럼이 있다. `normalize=True`이면 `count` 대신 `proportion`(float64)이다.
    - `Series.value_counts`의 키 컬럼 이름은 Series 이름이다. 이름이 없으면 `"value"`다.
    - 키 컬럼 이름이 `count`(또는 `proportion`)와 겹치면 에러다.
  - 행 순서
    - `sort=True`이면 개수의 내림차순이다. `ascending=True`이면 오름차순이다.
    - 개수가 같으면 처음 나타난 순서를 따른다.
    - `sort=False`이면 처음 나타난 순서다.
    - NA 키도 같은 규칙을 따른다. **[pandas와 다름]**
      - pandas는 `dropna=False`일 때 NA를 처음 나타난 자리가 아니라 맨 뒤에 놓는다 (`sort=False`여도 그렇다).
      - 그런데 pandas의 `unique()`는 NA를 나타난 자리에 둔다. 우리는 둘 다 처음 나타난 순서로 맞춘다.
  - Index는 RangeIndex다.
  - `dropna=True`이면 키 중에 NA가 있는 행을 버린다.
- **factorize**: 공개하지 않는다. 값을 정수 코드로 바꾸는 내부 기능으로만 쓴다.

## 14. Index, 선택, 정렬, 대입 (M4)
- **Index**
  - `Index(data, *, name=None)`: 라벨은 int64나 string이다. float, bool, NA는 에러다 (§6).
  - `RangeIndex(start=0, stop=None, step=1, *, name=None)`
  - `name`은 `str` 또는 `None`이다 (Series.name과 같은 규칙).
    - 연산 결과의 Index 이름은 Series 이름 규칙을 따른다: 같으면 유지하고, 다르면 `None`이다.
    - repr은 pandas처럼 이름을 보여 준다.
  - 공개하는 것: `dtype`, `name`, `len()`, 반복(라벨), `in`(라벨), `to_list()`, `equals()`, `is_unique`
    - `equals()`는 라벨만 비교한다. 이름은 보지 않는다 (pandas와 같다).
  - 라벨 조회용 해시 테이블은 처음 조회할 때 만들어서 캐시한다 (DESIGN §2.5).
  - 라벨의 타입은 Index의 dtype과 맞아야 한다.
    - int Index에는 int 라벨만 쓴다. bool과 float는 쓸 수 없다.
    - string Index에는 str 라벨만 쓴다.
    - 타입이 맞지 않으면 TypeError다.
- **생성자의 `index=`**
  - `Series(data, *, index=None, dtype=None, name=None)`
  - `DataFrame(data, *, index=None, columns=None, dtype=None)`
  - list나 배열을 주면 길이가 `index`와 같아야 한다.
  - Series를 주면 `index`에 맞춰 다시 배열한다(reindex). 없는 라벨은 NA다.
- **`reindex(labels)`**: `Series.reindex(labels)`, `DataFrame.reindex(labels)` (행만)
  - 주어진 라벨 순서로 행을 다시 배열한다. 없는 라벨은 NA다.
  - 원래 라벨은 고유해야 한다.
  - `method`, `fill_value` 등은 없다.
- **M1~M3에서 임시로 둔 반환 타입을 바꾼다.**
  - `df.columns`는 string `Index`다.
  - `df.dtypes`는 컬럼 이름을 Index로 하는 string Series다.
  - `df.nunique()`는 컬럼 이름을 Index로 하는 int64 Series다.
- **선택**
  - `s[x]`는 `s.loc[x]`와 같다 (§7).
  - `.loc`에 쓸 수 있는 것
    - 라벨 하나: 항상 스칼라를 돌려준다. **[pandas와 다름]**
      - 라벨이 없으면 KeyError다.
      - 라벨이 여러 행에 있으면 에러다. 그런 행을 모두 원하면 `s.loc[["a"]]`처럼 리스트로 조회한다.
    - 라벨 리스트: 각 라벨에 맞는 행을 모두 Index 순서대로 돌려준다. 없는 라벨이 하나라도 있으면 KeyError다.
    - 라벨 슬라이스 `a:b`
      - 양 끝을 포함한다.
      - 양 끝 라벨은 있어야 하고 고유해야 한다.
      - Index가 정렬되어 있는지 보고 이진 탐색으로 바꾸는 휴리스틱은 쓰지 않는다 (DESIGN §2.5). **[pandas와 다름]**
    - bool Series 마스크
  - `.iloc`에 쓸 수 있는 것: 정수(음수 가능), 슬라이스, 정수 리스트
    - 범위를 벗어나면 IndexError다.
  - DataFrame
    - `df.loc[행]`, `df.loc[행, 열]`, `df.iloc[행]`, `df.iloc[행, 열]`
    - 열 자리에는 이름 하나, 이름 리스트, 이름 슬라이스(양 끝 포함), `:`를 쓴다.
    - **행을 하나만 고르면 1행짜리 DataFrame을 돌려준다.** **[pandas와 다름]**
      - pandas는 행을 Series로 돌려주는데, dtype이 섞이면 object가 된다.
      - 행 하나와 열 하나를 함께 고르면 스칼라다. 예: `df.loc["r", "a"]`
  - `df[...]`는 이름 하나, 이름 리스트, bool Series를 받는다 (§7).
  - 슬라이스로 고르면 RangeIndex는 RangeIndex로 남는다. 라벨 리스트나 마스크로 고르면 Index가 된다.
    - 결과 타입은 무엇으로 골랐는지에 따라 정해지고, 데이터에 따라 바뀌지 않는다.
- **정렬(alignment)**
  - 정렬이 일어나는 곳
    - Series와 Series의 연산
    - DataFrame과 DataFrame의 연산 (행과 열 모두)
    - Series dict로 DataFrame을 만들 때
    - `df["c"] = Series`
    - 마스크
  - 두 Index가 같으면 위치대로 계산한다.
  - 다르면 양쪽 라벨이 모두 고유해야 한다 (§6).
    - 합집합의 순서는 왼쪽 라벨 순서, 그 다음 오른쪽에만 있는 라벨을 오른쪽 순서대로 붙인 것이다 (A6).
    - 한쪽에 없는 라벨의 행은 NA다.
  - int 라벨과 string 라벨은 정렬할 수 없다(에러). RangeIndex는 int64 라벨로 친다. **[pandas와 다름]**
  - DataFrame끼리 연산할 때 한쪽에만 있는 컬럼은, 다른 쪽을 같은 dtype의 NA 컬럼으로 보고 계산한다.
  - **마스크**: 마스크와 대상의 라벨 집합이 같아야 한다. **[pandas와 다름]**
    - 라벨 순서는 달라도 된다.
    - 대상의 라벨이 마스크에 없거나, 마스크에만 있는 라벨이 있으면 에러다.
    - pandas는 마스크에만 있는 라벨을 조용히 무시한다.
- **대입** (Copy-on-Write, §8, DESIGN §2.3)
  - 지원하는 형태: `df["c"] = 값`, `s.loc[...] = 값`, `s.iloc[...] = 값`, `df.loc[행, "c"] = 값`, `df.iloc[행, j] = 값`
    - 행은 선택과 같은 방법으로 고른다.
  - 넣을 수 있는 값
    - 스칼라 (`NA` 포함)
    - list: 길이가 고른 행 수와 같아야 한다.
    - Series: 고른 행의 라벨에 맞춰 정렬한다.
  - `df["c"] = 값`은 컬럼 전체를 새로 만든다. 새 컬럼이면 끝에 붙는다.
  - 행에 대입하는 값은 컬럼 dtype으로 손실 없이 바뀌어야 한다. 규칙은 생성자 `dtype=`와 같다 (§10).
  - 없는 라벨에 대입하면 KeyError다. 행을 늘리지 않는다. **[pandas와 다름]**
  - 쓰기 전에 버퍼를 다른 객체(다른 Series, DataFrame, 슬라이스, `to_numpy` 뷰)와 공유하고 있으면 먼저 복사한다.
    - 예: `s = df["a"]; s.iloc[0] = 1`을 해도 `df`는 바뀌지 않는다.
- **반복과 `in`**
  - Series: 반복과 `in` 모두 값 기준이다. **[pandas와 다름]**
    - pandas는 반복은 값, `in`은 라벨이다.
    - `x in s`는 §13의 키 규칙으로 판단한다. NaN은 NaN과, NA는 NA와 같다.
    - 라벨이 있는지는 `x in s.index`로 확인한다.
  - DataFrame: 반복과 `in` 모두 컬럼 이름 기준이다.
  - Index: 반복과 `in` 모두 라벨 기준이다.

## 15. 리덕션 (M5)
- **시그니처** (C3)
  - `sum`, `mean`, `min`, `max`: `(*, skipna=True)`
  - `std`, `var`: `(*, skipna=True, ddof=1)`. `ddof`는 0 이상의 int다.
  - `count()`: NA가 아닌 값의 개수다.
- **반환값**
  - Series: Python 스칼라 또는 `NA`다 (§8).
  - DataFrame: 1행짜리 DataFrame이다. **[pandas와 다름]**
    - 행 라벨은 리덕션 이름(`"sum"`, `"mean"` 등)이다.
    - 컬럼마다 자기 결과 dtype을 유지한다.
    - 리덕션을 할 수 없는 컬럼이 하나라도 있으면 TypeError다 (예: string 컬럼의 `mean`).
    - pandas는 dtype이 섞이면 object Series를 돌려준다.
- **dtype별 규칙**

| dtype | sum | mean | min, max | std, var |
|---|---|---|---|---|
| int64 | int64 (wrap, §3) | float64: int128로 정확히 더한 뒤 나눔 | int64 | float64 |
| float64 | float64 | float64 | float64 | float64 |
| bool | int64: True의 개수 | float64: True의 비율 | bool | TypeError |
| string | TypeError | TypeError | string (코드포인트 순서) | TypeError |

- **NA**
  - `skipna=True`(기본값)이면 NA를 건너뛴다.
  - `skipna=False`이면 NA가 하나라도 있을 때 결과가 NA다.
  - 값이 모두 NA이거나 없으면: `sum`은 0, `count`는 0이고, 나머지는 NA다 (§2).
  - `std`, `var`에서 `NA가 아닌 값의 개수 - ddof <= 0`이면 NA다. **[pandas와 다름]** pandas는 NaN이다.
- **NaN**: `skipna`와 상관없이 결과로 전파된다 (§1).
- **0.0과 -0.0**: `min`과 `max`는 `-0.0 < 0.0`으로 본다. numpy와 같고, IEEE 754-2019의 minimum, maximum이다.
- **float 합의 순서**: numpy의 pairwise summation이다. 블록 128개, 누산기 8개, 절반씩 재귀하고, 0.0에서 시작한다.
  - NA가 없으면 pandas, numpy와 비트까지 같다.
  - NA 행은 0.0으로 채워져 있으므로 그대로 더한다.
  - M11에서는 재귀 트리의 가지를 나눠서 병렬로 계산하므로, 스레드 수와 상관없이 비트가 같다.
- **`var`**는 두 번 훑어서 계산한다.
  1. 평균을 구한다.
  2. 편차 제곱 `(x - 평균)²`을 pairwise로 더한다.
  3. `개수 - ddof`로 나눈다.
  - `std`는 `sqrt(var)`이다.

## 16. 미확정
- `read_csv`의 옵션과 타입 추론
