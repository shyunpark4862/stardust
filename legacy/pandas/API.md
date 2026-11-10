# stardust-pandas API 의미론

DESIGN.md의 결정(NA 하나, NaN은 값, Copy-on-Write)에서 따라 나오는 동작 규칙이다.
- 3라운드(S1–S8)에서 사용자가 모두 권장안으로 결정했다 (2026-11-10).
- §6의 정렬 라벨 순서는 4라운드 A6에서 정했다.
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

## 9. 미확정
- repr 형식
- `read_csv`의 옵션과 타입 추론
