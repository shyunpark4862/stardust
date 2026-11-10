# stardust-pandas API 의미론

DESIGN.md의 결정(NA 하나, NaN은 값, Copy-on-Write)에서 따라 나오는 동작 규칙이다.
- 3라운드(S1–S8)에서 사용자가 모두 권장안으로 결정했다 (2026-11-10).
- §6의 정렬 라벨 순서는 4라운드 A6에서 정했다.
- §9–§11은 M1을 시작하기 전에 사용자가 결정했다 (2026-11-11).
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
- **C6. 반환 타입**
  - `Series.unique()`는 Series를 돌려준다.
    - 이름은 유지한다. Index는 RangeIndex다. 값은 처음 나타난 순서다.
  - `DataFrame.value_counts()`는 키 컬럼과 `count` 컬럼을 가진 DataFrame을 돌려준다.
    - 다중 키 groupby와 같은 방식이다 (§4).

## 12. 미확정
- `read_csv`의 옵션과 타입 추론
