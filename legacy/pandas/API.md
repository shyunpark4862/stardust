# stardust-pandas API 의미론

DESIGN.md의 결정(NA 하나, NaN은 값, Copy-on-Write)에서 따라 나오는 동작 규칙이다.
- 3라운드(S1–S8)에서 사용자가 모두 권장안으로 결정했다 (2026-11-10).
- §6의 정렬 라벨 순서는 4라운드 A6에서 정했다.
- §9–§11은 M1을 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §12와 §11의 C7은 M2를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §13은 M3를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §14는 M4를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §15는 M5를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §16은 M6를 시작하기 전에 사용자가 결정했다 (2026-11-11). 이때 §4의 '키의 위치'를 고쳤다.
- §17은 M7을 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §18은 M8을 시작하기 전에 사용자가 결정했다 (2026-11-11).
- §19는 M9를 시작하기 전에 사용자가 결정했다 (2026-11-11).
- pandas와 다른 점은 **[pandas와 다름]**으로 표시한다.

---

## 1. NaN (S1)
NaN은 결측값이 아니라 평범한 float 값이다 (DESIGN §2.1).

- **리덕션**: `sum`, `mean` 등에 NaN이 섞이면 결과가 NaN이 된다. **[pandas와 다름]**
  - `skipna`는 NA만 건너뛴다.
- **`min`, `max`**: NaN을 전파한다.
- **정렬**: NaN은 +inf보다 큰 값으로 친다. 그래서 오름차순에서는 +inf 뒤에, 내림차순에서는 맨 앞에 온다 (§17).
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
- **키의 위치**: 단일 키든 다중 키든 결과의 일반 컬럼이 된다. 결과의 Index는 RangeIndex다 (§16). **[pandas와 다름]**
  - 처음(3라운드)에는 단일 키가 결과의 Index가 되도록 정했다.
  - 이것이 "float와 bool은 Index로 쓸 수 없다"(§6)와 충돌해서, M6을 시작할 때 사용자가 고쳤다 (2026-11-11).

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

## 16. groupby (M6)
- **`df.groupby(by, *, sort=True, dropna=True)`**
  - `by`는 컬럼 이름 하나 또는 리스트다. Series 키와 `Series.groupby`는 없다.
  - 키가 아닌 컬럼이 모두 값 컬럼이 된다.
  - `g["c"]`, `g[["c", "d"]]`로 값 컬럼을 고른다. 키 컬럼은 값 컬럼으로 고를 수 없다.
- **결과**: DataFrame이다.
  - 컬럼은 키 컬럼(`by` 순서)이 먼저 오고, 그 다음 집계 컬럼이다.
  - Index는 RangeIndex다. `value_counts`(§13)와 같은 모양이다.
- **그룹 순서**
  - `sort=True`: 키의 오름차순이다.
    - int와 float는 수 순서다. NaN은 +inf 뒤에 온다 (§1). `0.0`과 `-0.0`은 같은 키다.
    - string은 코드포인트 순서다. bool은 `False < True`다.
    - 다중 키는 키를 앞에서부터 차례로 비교한다.
    - NA 키 그룹(`dropna=False`일 때)은 맨 뒤에 온다. 다중 키에서는 키마다 이 규칙을 적용한다.
  - `sort=False`: 그룹이 처음 나타난 순서다.
- **`dropna=True`**: 키 중 하나라도 NA인 행을 버린다. NaN은 값이므로 버리지 않는다. **[pandas와 다름]**
- **메서드**: 인자는 C3와 같다.
  - `sum`, `mean`, `min`, `max`: `(*, skipna=True)`
  - `std`, `var`: `(*, skipna=True, ddof=1)`
  - `count()`: 값 컬럼마다 NA가 아닌 값의 개수다.
  - `size()`: 그룹의 행 수다. 결과 컬럼 이름은 `size`다.
  - `first`, `last`: `(*, skipna=True)`. 그룹에서 NA가 아닌 처음(마지막) 값이다. `skipna=False`이면 처음(마지막) 행의 값이고, NA일 수도 있다.
  - 결과 dtype과 할 수 없는 조합(예: string의 `mean`)은 §15와 같다. `first`와 `last`는 컬럼 dtype을 그대로 유지한다.
  - 값이 모두 NA인 그룹: `sum`은 0, `count`는 0이고, 나머지는 NA다.
- **`agg`**: 함수는 이름(문자열)으로만 준다. `sum mean min max count std var first last` 중 하나다.
  - `g.agg("sum")`: 모든 값 컬럼에 적용한다.
  - `g.agg({"f": "sum", "i": "max"})`: 컬럼마다 함수 하나다.
  - `g.agg(total=("f", "sum"), n=("i", "count"))`: 결과 컬럼 이름을 직접 정한다. 같은 컬럼에 함수 여러 개를 적용할 때 쓴다.
  - 함수 리스트는 에러다 (MultiIndex가 없다). **[pandas와 다름]**
  - 결과 컬럼 이름이 키 컬럼과 겹치면 에러다.
- **float 합의 순서**: 그룹마다 numpy pairwise로 더한다 (§15). **[pandas와 다름]**
  - 행을 그룹별로 모은 뒤(counting sort), 각 그룹의 행을 원래 순서대로 더한다.
  - 그래서 한 그룹의 합은 그 행들만 골라낸 Series의 `sum()`과 비트까지 같다.
  - pandas는 행 순서대로 Kahan 보정 합을 쓴다.

## 17. 정렬 (M7)
- **시그니처** (C2, C4)
  - `Series.sort_values(*, ascending=True, na_position="last", ignore_index=False)`
  - `DataFrame.sort_values(by, *, ascending=True, na_position="last", ignore_index=False)`
  - `Series.sort_index(*, ascending=True, ignore_index=False)`
  - `DataFrame.sort_index(*, ascending=True, ignore_index=False)`
  - 없는 인자
    - `kind`: 정렬은 항상 안정 정렬이다.
    - `key=`: Python 함수를 받는 인자다.
    - `sort_index`의 `na_position`: Index에는 NA가 없다.
    - `sort_index`의 `level`, `sort_remaining`: MultiIndex가 없다.
- **`ascending`과 `na_position`**: 둘 다 값 하나(모든 키에 적용) 또는 키 수만큼의 리스트를 받는다. **[pandas와 다름]**
  - pandas는 `na_position`에 리스트를 받지 않는다.
  - `na_position`은 `"first"` 또는 `"last"`다.
- **순서**
  - int와 float는 수 순서다.
    - NaN은 +inf보다 큰 값이다. 그래서 내림차순에서는 맨 앞에 온다. **[pandas와 다름]** pandas는 NaN을 NA처럼 `na_position`으로 놓는다.
    - `0.0`과 `-0.0`은 같은 값이다.
  - string은 코드포인트 순서다. bool은 `False < True`다.
  - NA는 `na_position`이 정하는 곳에 온다. 방향과 상관없다.
- **안정성**: 키가 같은 행은 원래 순서를 지킨다. 오름차순과 내림차순 모두 그렇다.
  - 다중 키는 앞의 키부터 차례로 비교한 순서다.
- **결과**: 행이 그 순서로 재배열되고, 라벨도 행을 따라간다(Index). `ignore_index=True`이면 RangeIndex다.
- **알고리즘** (DESIGN §2.5)
  - 숫자와 bool은 64비트 키에 대한 안정 LSD 기수 정렬이다.
  - string은 고유값의 순위로 바꾼 뒤 안정 counting sort를 한다.
  - 다중 키는 마지막 키부터 차례로 안정 정렬한다.
  - 모든 키가 같은 자릿수를 건너뛰는 일은 하지 않는다. 데이터를 보고 경로를 고르지 않기 위해서다.

## 18. merge와 concat (M8)
- **시그니처** (C1, C2)
  - `sdpd.merge(left, right, *, how="inner", on=None, left_on=None, right_on=None, sort=False, suffixes=("_x", "_y"))`
  - `df.merge(right, *, ...)`: 인자는 함수와 같다.
  - `how`는 `"inner"`, `"left"`, `"right"`, `"outer"`다.
- **키**
  - 키는 꼭 준다. `on`(양쪽에 같은 이름) 또는 `left_on`과 `right_on`(같은 개수)이다. **[pandas와 다름]**
    - pandas는 키를 생략하면 공통 컬럼 전체를 키로 쓴다.
  - 키 컬럼의 dtype은 양쪽이 같아야 한다. 예를 들어 int64와 float64는 에러다. **[pandas와 다름]**
  - 매칭 규칙 (§1, §5)
    - NA 키는 아무것과도 매칭되지 않는다.
    - NaN은 NaN과, `0.0`은 `-0.0`과 매칭된다.
- **빼는 기능**: `left_index`, `right_index`, `indicator`, `validate`, `how="cross"`, `copy`
- **결과 컬럼**
  - 왼쪽 컬럼(왼쪽 순서)이 먼저 오고, 그 다음 오른쪽 컬럼(오른쪽 순서)이다.
  - `on`으로 조인하면 키 컬럼은 하나로 합친다.
    - 위치는 왼쪽에서의 위치다.
    - 값은 왼쪽 행이 있으면 왼쪽 값, 없으면(오른쪽에만 있는 행) 오른쪽 값이다.
  - `left_on`/`right_on`으로 조인하면 양쪽 키 컬럼이 모두 남는다.
  - 이름이 겹치는 컬럼(합친 키는 제외)에는 `suffixes`를 붙인다.
    - 접미사를 붙인 뒤에도 이름이 겹치면 에러다.
  - 매칭되지 않은 쪽의 값은 NA다. dtype은 그대로다.
- **결과 행 순서**: §5 그대로다.
  - `sort=True`이면 키로 안정 정렬한다. 오름차순이고 NA는 맨 뒤다 (§17).
    - `on`이면 합친 키로 정렬한다.
    - `left_on`/`right_on`이면 왼쪽 키로 정렬하고, 왼쪽 행이 없으면 오른쪽 키 값을 쓴다.
- **결과 Index**: RangeIndex다.
- **`sdpd.concat(objs, *, join="outer", ignore_index=False)`**: 행 방향만 잇는다. `axis`는 없다.
  - `objs`는 DataFrame들이거나 Series들이다. 섞으면 에러이고, 비어 있어도 에러다.
  - 컬럼
    - `join="outer"`: 처음 나타난 순서로 합친다. 어떤 DataFrame에 없는 컬럼은 그 행들이 NA다.
    - `join="inner"`: 모두에 있는 컬럼만, 첫 DataFrame의 순서로 둔다.
  - 같은 이름의 컬럼은 dtype이 같아야 한다. Series끼리도 dtype이 같아야 한다. **[pandas와 다름]**
  - 라벨
    - 라벨을 이어 붙인다. 중복이 생겨도 된다.
    - int 라벨과 string 라벨은 섞을 수 없다(에러). RangeIndex는 int64로 친다.
    - Index와 Series의 이름은 모두 같으면 유지하고, 아니면 `None`이다.
  - `ignore_index=True`이면 RangeIndex다.
  - `keys`, `levels`, `names`, `verify_integrity`, `sort`, `copy`는 없다.

## 19. read_csv (M9)
- **시그니처**: `sdpd.read_csv(path, *, sep=",", header=True, names=None, usecols=None, dtype=None, na_values=None, nrows=None, index_col=None)`
  - `sep`: 구분 문자 하나다. 따옴표나 줄바꿈 문자는 쓸 수 없다.
  - `header`
    - `True`: 첫 레코드가 컬럼 이름이다.
    - `False`: `names=`가 꼭 필요하다 (§10에서 list의 list에 `columns=`를 요구한 것과 같다).
  - `names`: 컬럼 이름이다. `header=True`와 함께 주면 파일의 이름을 바꾼다. 개수가 같아야 한다.
  - `usecols`: 읽을 컬럼 이름의 리스트다. 결과는 이 리스트의 순서를 따른다.
  - `dtype`: dtype 하나(모든 컬럼) 또는 `{컬럼: dtype}`이다.
    - 지정한 컬럼은 추론하지 않고, §12의 엄격한 파싱으로 읽는다. 읽을 수 없는 값은 에러이고, 줄 번호를 알려 준다.
  - `na_values`: 빈 필드 말고도 NA로 읽을 문자열의 리스트다.
  - `nrows`: 데이터 행을 최대 몇 개까지 읽을지다.
  - `index_col`: Index가 될 컬럼 이름 하나다.
    - int64나 string이어야 하고, NA가 없어야 한다 (§6).
    - Index 이름은 컬럼 이름이다. 이 컬럼은 컬럼 목록에서 빠진다.
- **파일 형식**
  - UTF-8만 받는다. 맨 앞의 BOM은 건너뛴다. string 값은 올바른 UTF-8이어야 한다.
  - 줄바꿈은 `\n` 또는 `\r\n`이다. 파일 끝의 마지막 줄바꿈은 빈 줄을 만들지 않는다.
  - 따옴표는 RFC 4180을 따른다.
    - 큰따옴표로 감싼 필드 안에서는 구분 문자와 줄바꿈도 값이다. `""`는 `"`다.
    - 닫는 따옴표 뒤에는 구분 문자나 줄바꿈이나 파일 끝만 올 수 있다.
    - 따옴표로 감싸지 않은 필드 안의 `"`는 에러다.
  - 모든 행의 필드 개수가 컬럼 수와 같아야 한다. 다르면 에러이고, 줄 번호를 알려 준다. **[pandas와 다름]**
  - 완전히 빈 줄은 빈 필드 하나짜리 행이다. **[pandas와 다름]**
    - 컬럼이 하나인 파일에서는 NA 행이 되고, 컬럼이 여럿이면 필드 개수 에러다.
    - pandas는 빈 줄을 건너뛴다.
  - 헤더의 컬럼 이름은 고유해야 한다 (§6). pandas는 `a.1`처럼 이름을 바꾼다.
- **NA** **[pandas와 다름]**
  - 따옴표 없는 빈 필드와 `na_values`에 있는 문자열만 NA다.
  - 따옴표로 감싼 빈 필드(`""`)는 빈 문자열이다. 그래서 `""`와 NA를 구분할 수 있다.
  - 텍스트 `NaN`, `nan`, `inf`는 float 컬럼에서 NaN 값과 inf 값이다. NA가 아니다 (§1).
  - pandas는 `NA`, `null`, `NaN` 같은 목록을 모두 결측으로 읽는다.
- **타입 추론**: 컬럼마다, NA가 아닌 모든 필드를 본다.
  - 따옴표는 필드의 경계일 뿐이다. `"12"`는 `12`와 같다.
  - 추론 순서
    1. 모든 필드가 §12의 int 문법이고 int64 범위 안이면 **int64**다.
    2. 그렇지 않고 모든 필드가 float 문법이면 **float64**다.
       - 단, int 문법인 필드 중에 int64 범위를 넘거나 float64로 정확히 표현할 수 없는 값이 있으면 string이다. 값이 조용히 바뀌지 않게 하기 위해서다.
    3. 그렇지 않고 모든 필드가 `True` 또는 `False`이면 **bool**이다.
    4. 그 밖에는 **string**이다.
  - 공백이 들어간 필드(예: `" 1"`)는 숫자가 아니다 (§12). **[pandas와 다름]**
  - 모두 NA인 컬럼은 float64다 (§10).
- `encoding`, `skiprows`, `comment`, `quotechar`, `thousands`, `decimal`, `parse_dates`, `converters` 등 나머지 옵션은 없다.

## 20. 미확정
- 없음.
