# 11장. 해시, groupby, 정렬, merge, CSV

> "같은 키"와 "순서"를 한 곳에서 정하면 무엇이 쉬워지나? 1M 행의 groupby와 merge는 어디서 시간을 쓰나?

## 1. 원칙: 같은 결과는 같은 코드에서
frame의 큰 연산들(unique, groupby, merge, sort, Index 조회)은 세 가지 바탕 위에 올라가 있다 (docs/design/frame.md §1.2).

| 질문 | 정하는 곳 | 쓰는 연산 |
|---|---|---|
| 무엇이 같은 키인가 | factorize 한 곳 (`ops/hash.cpp`) | unique, duplicated, groupby, merge, Index 조회 |
| 값의 순서는 무엇인가 | 정렬 키 변환 한 곳 (`ops/radix.hpp`) | sort, groupby의 그룹 순서 |
| float의 합은 무엇인가 | pairwise 정의 하나 (`base/pairwise.hpp`) | sum, mean, var, groupby의 그룹 합 |

그래서 규칙이 연산마다 갈라지지 않는다. 예: `-0.0`과 `0.0`이 groupby에서는 같은데 unique에서는 다르다, 같은 일이 생기지 않는다.

## 2. factorize
값을 **처음 나타난 순서의 정수 코드**로 바꾼다.

```
값      [3.0, NaN, 1.0, 3.0, -0.0, 0.0, NA, NaN]
codes   [0,   1,   2,   0,   3,    3,   4,  1  ]
first   [0, 1, 2, 4, 6]                           코드마다 처음 나온 행
```

| 기능 | 계산 |
|---|---|
| `nunique()` | 코드의 개수 (NA는 빼고) |
| `duplicated(keep="first")` | `위치 != first[codes]` |
| `drop_duplicates()` | 처음 나온 행만 |
| `value_counts()` | 개수로 안정 정렬한 코드 순서대로 `take(first)`와 개수 |
| `unique_values()` | `take(first)`를 그룹 순서(§4)로 정렬 |
| groupby | 코드가 그룹 번호 |
| merge | 오른쪽 키의 코드로 왼쪽 키를 조회 |

```python
import stardust as sd

s = sd.frame.Series([3.0, float("nan"), 1.0, 3.0, -0.0, 0.0, None, float("nan")])
print(s.nunique(), s.duplicated().tolist())  # 4 [False, False, False, True, False, True, False, True]
print(s.drop_duplicates().tolist())  # [3.0, nan, 1.0, -0.0, <NA>]
vc = s.value_counts()
print(vc["values"].tolist(), vc["counts"].tolist())  # [3.0, nan, -0.0, 1.0] [2, 2, 2, 1]
```

- **같은 키**: NaN끼리 같다(페이로드와 부호 무관). `-0.0`과 `0.0`이 같다. NA끼리 같다(merge만 예외, §5).
- 결과에는 **먼저 나온 원래 값**이 나온다(`-0.0`이 먼저면 `-0.0`). `take(first)`이기 때문이다.
- 해시 전에 규칙을 적용한다(`float_key`): NaN은 하나의 NaN 비트로, `-0.0`은 `+0.0`의 비트로. 그러면 해시 테이블은 "비트가 같으면 같은 키"만 알면 된다. 정렬 키(§4)도 같은 함수에서 시작한다.

### 2.1 Swiss table
```
제어 바이트 (칸마다 1바이트): 0 = 빈 칸, 0x80 | 해시의 아래 7비트 = 찬 칸
조회: 그룹(16칸)의 제어 바이트 16개를 NEON으로 한 번에 태그와 비교 → 후보 칸만 키를 비교
```
- **왜 Swiss table인가: 분기 예측 실패.** linear probing으로 1M 행, 키 1,000종이 4.2ms였다. 테이블(32KB)이 L1에 다 들어가는데도 행마다 약 19사이클이었다.
  - probe 횟수는 원인이 아니었다(평균 1.48회).
  - 프로파일 샘플의 2/3가 "첫 칸의 키가 다르다 → 다음 칸" 분기 바로 뒤에 몰렸다. 첫 칸에서 찾을지 한 칸 더 갈지를 예측할 수 없고, 틀릴 때마다 지연 사슬이 처음부터 다시 시작된다.
  - Swiss table에 남는 분기("후보가 있나", "빈 칸이 있나")는 거의 항상 같은 쪽으로 간다. 4.29 → 2.03ms.
- 빈 칸이 0이라 `calloc`한 메모리가 그대로 빈 테이블이다. 8바이트 키 테이블은 처음부터 행 수만큼 잡는다(키 수의 상한). 0 페이지라서 쓴 페이지만 메모리를 차지한다.
- NEON에는 x86의 `movemask`가 없다. `vshrn_n_u16(…, 4)`로 비교 결과 16바이트를 64비트 정수 하나로 줄이고 `ctz`로 다음 후보를 찾는다.

### 2.2 병렬 factorize: 해시로 나눈다
행이 기준값(2¹⁷) 이상이면 해시의 위쪽 8비트로 **256개 파티션**에 나눈다 (docs/design/frame.md §8.7).
- 같은 키는 같은 해시라서 한 파티션에 간다. 파티션 안에서는 행이 원래 순서다. 파티션마다 테이블을 만들고, 마지막에 "그 키의 첫 등장 행보다 앞선 첫 등장 행의 수"로 전역 코드를 다시 매긴다.
- **코드는 직렬 factorize와 정확히 같다.** 그래서 이 연산은 나누는 방식이 스레드 수를 따라도 된다(6장 §5의 frame 예외).
- 처음 구현은 1스레드에서 2–4배 느렸다. 단계별로 재서 다섯 가지를 고쳤다(행 번호 대신 (키, 행) 쌍을 흩뿌림, 이진 탐색 대신 순위 배열, 거짓 공유를 없애려 출력을 행 청크 단위로 등). 결과: int `nunique` 63만 종 1스레드 16.7ms → 10스레드 3.37ms.

## 3. groupby
```
키      [b,  a,  b,  NA]        factorize
codes   [0,  1,  0,  2 ]
                                안정 counting sort로 행을 그룹별로 모은다 (CSR)
start   [0, 2, 3, 4]            그룹 g의 행 = rows[start[g] : start[g+1]]
rows    [0, 2, 1, 3]            그룹 안에서는 원래 행 순서
```
- 그룹마다 5장의 리덕션 함수를 그대로 부른다. 그룹의 행을 원래 순서로 pairwise로 더하므로, **그룹 합은 그 그룹만 고른 `Series.sum()`과 비트까지 같다.** 테스트의 정답이 이것이다.
- pandas의 groupby 합은 Kahan 보정 합이라 우리와 비트가 다를 수 있다.
- 키는 결과의 일반 컬럼이고, 결과 Index는 RangeIndex다(MultiIndex가 없다).
- 병렬: 그룹을 약 16K행씩 통째로 묶어 태스크로 나눈다. 그룹마다 독립이라 경계가 어디든 결과가 같다.

```python
import stardust as sd

df = sd.frame.DataFrame({"k": ["b", "a", "b", None], "v": [0.1, 0.2, 0.7, 4.0]})
g = df.groupby("k").sum()
print(g["k"].tolist(), g["v"].tolist())  # ['a', 'b'] [0.2, 0.7999999999999999]
print(g["v"].tolist()[1] == df[df["k"] == "b"]["v"].sum())  # True
print(df.groupby("k", dropna=False).sum()["k"].tolist())  # ['a', 'b', <NA>]
print(df.groupby("k", sort=False).sum()["k"].tolist())  # ['b', 'a']
```

- `sort=True`(기본)면 그룹을 키 값의 순서로, `sort=False`면 처음 나타난 순서로 둔다. NA 그룹은 맨 뒤다.
- **그룹이 아니라 키 값을 정렬한다.** 키 열마다 고유값을 기수 정렬해 순위를 매기고, 순위로 그룹을 안정 counting sort한다(마지막 키부터, LSD). 그룹 번호를 비교 정렬하면 그룹 63만 개에서 비교가 1,260만 번이고 비교마다 문자열 비교다: 1M 행 2키 `sum` 306ms → 78ms (pandas 122ms).

## 4. 정렬: 값을 순서가 보존되는 uint64로
`sort_values`는 값을 대소 관계가 보존되는 64비트 키로 바꾼 뒤 LSD 기수 정렬(1바이트 자리 8개)을 한다. 비교가 없다.

| dtype | 키 | 이유 |
|---|---|---|
| int64 | `x ^ (1 << 63)` | 부호 비트를 뒤집으면 음수가 양수보다 작은 부호 없는 수가 된다 |
| float64 | `float_key` 뒤 양수는 `bits \| (1 << 63)`, 음수는 `~bits` | NaN의 비트가 +inf보다 커서 "NaN이 가장 크다"가 저절로 성립한다 |
| string | 앞 8바이트를 big-endian으로 | 접두사가 같은 구간만 남은 바이트로 비교 정렬 |

- 내림차순은 키의 모든 비트를 뒤집는다. 같은 키는 계속 같아서 **안정성이 유지된다.** "오름차순 뒤 뒤집기"는 같은 값의 순서까지 뒤집는다.
- NA는 키에 넣지 않고 떼어 두었다가 `na_position`대로 붙인다.
- **모든 자리를 지난다.** 모든 키에서 같은 바이트 자리를 히스토그램을 보고 건너뛰면 빠르지만, 데이터를 보는 경로 선택이라 하지 않는다. (배열의 int64 기수 정렬은 최솟값을 빼고 모든 키에서 같은 바이트 자리를 건너뛴다(docs/design/array.md §8.4). 두 영역이 옮겨 온 설계가 다른 곳이다. 둘 다 안정 정렬이라 결과는 같고 속도만 다르다.)
- 결과: int64 1M 행 1스레드 14.6ms, 10스레드 2.72ms (pandas `kind="stable"` 87.0ms, Polars 14스레드 3.18ms).

## 5. merge
```
1. 오른쪽 키만 테이블에 넣는다 (build)               → 오른쪽 행의 코드
2. 왼쪽 키는 조회만 한다 (probe)                     → 왼쪽 행의 코드, 없으면 -1
3. 오른쪽 행을 코드별로 모은다 (counting sort, CSR)
4. 왼쪽 행을 차례로: 맞는 오른쪽 행들 → (l, r) 쌍. 없으면 left/outer는 (l, -1)
5. outer: 맞은 적 없는 오른쪽 행 → (-1, r)
```
- **행 순서는 이 반복문의 순서다**: 왼쪽 순서, 한 왼쪽 행 안에서는 오른쪽 순서. 결과가 (왼쪽 위치, 오른쪽 위치) 두 배열이고, 컬럼은 모두 `take`로 만든다(-1은 NA).
- **NA 키는 아무것과도 맞지 않는다.** groupby와 다른 유일한 점이다.

```python
import stardust as sd

left = sd.frame.DataFrame({"k": [1, 2, 2, None], "a": [10, 20, 21, 40]})
right = sd.frame.DataFrame({"k": [2, 2, 1, None], "b": ["x", "y", "z", "w"]})
print(sd.frame.merge(left, right, on="k")["b"].tolist())  # ['z', 'x', 'y', 'x', 'y']
print(sd.frame.merge(left, right, on="k", how="left")["b"].tolist())  # ['z', 'x', 'y', 'x', 'y', <NA>]
```

- 결과 (1M × 10만 행 int inner): 1스레드 12.6ms, 10스레드 2.37ms (pandas 7.04ms, Polars 2.54ms).

## 6. read_csv
원칙은 **"파일에 적힌 것과 다른 값이 나오지 않는다"**이다 (docs/design/frame.md §10.1).
- NA는 따옴표 없는 빈 필드와 `na_values`의 텍스트뿐이다. 텍스트 `NaN`은 float의 NaN이고, `""`는 빈 문자열이다.
- 숫자의 반올림만은 예외다: float 컬럼의 int는 list에서 만들 때처럼 float64로 반올림한다(2⁵³을 넘거나 int64 밖이어도). pandas, numpy와 같다. 값을 그대로 지키려면 `dtype=`으로 고른다.
- 형식이 틀린 파일은 줄 번호와 함께 ValueError다.

```python
import pathlib
import tempfile
import stardust as sd

p = pathlib.Path(tempfile.mkdtemp()) / "a.csv"
p.write_text('id,v,name\n1,1.5,"a, b"\n2,,""\n3,NaN,c\n')
t = sd.frame.read_csv(str(p))
print(t.dtypes)  # {'id': int64, 'v': float64, 'name': string}
print(t["v"].isna().tolist(), t["name"].tolist())  # [False, True, False] ['a, b', '', 'c']
p.write_text("x\n1.5\n9007199254740993\n")
print(sd.frame.read_csv(str(p)).dtypes)  # {'x': string}
p.write_text('x\n"abc\n')
try:
    sd.frame.read_csv(str(p))
except ValueError as e:
    print(e)  # read_csv(): line 2: a quoted field is never closed
```

- list에서 만들 때는 같은 값이 float64로 반올림된다(`Series([1.5, 9007199254740993])`은 float64). 배열의 `asarray`와 같은 규칙이다. CSV의 추론 규칙과는 다르다. M6에서 문서에만 적고 남긴 차이다 (DESIGN.md §8.4).

**구조** (simdjson/simdcsv의 방식):
- **structural index**: 64바이트 블록마다 따옴표, 구분자, 줄바꿈의 64비트 마스크를 NEON으로 만든다. "따옴표 안쪽" 마스크는 prefix XOR이고, 이것은 캐리 없는 곱셈 한 번(`vmull_p64`, PMULL)이다. 필드 끝을 `ctz`로 하나씩 꺼내므로 바이트마다 분기가 없다.
- **한 번에 읽기**: 추론하면서 값을 만든다. int로 읽다가 int가 아닌 필드를 만나면 앞 값들을 double로 바꾸고 계속 읽는다.
- **숫자 파싱**: Clinger의 빠른 길 → Eisel-Lemire(5^q의 128비트 근사 표와 128비트 곱) → 마지막에 `from_chars`. 8자리씩 SWAR로 읽는다. 정답은 Python `float(text)`이고, 반올림 경계의 어려운 수까지 비트가 같다. libc++ `from_chars`만 쓰던 때보다 float 파일 48.6 → 24.8ms.
- **청크로 나누기**: CSV는 아무 데서나 자를 수 없다(따옴표 안의 줄바꿈). 그런데 "따옴표 안"은 그 앞 따옴표 개수의 홀짝이다. 512KB 청크마다 따옴표 개수를 먼저 세고(병렬) 누적 XOR로 각 청크의 시작 상태를 **정확히** 안다. simdcsv와 Polars는 추측하고 틀리면 다시 읽는다.
- 결과 (10스레드): mixed 1M × 4 12.9ms. pandas 90.1ms, Polars 11.9ms.

## 7. 어디에 서 있나
시간 비율의 기하평균이다. 1보다 크면 sd.frame이 빠르다 (docs/design/frame.md §13.3).

| 행 수 | pandas / 우리@1 | Polars@1 / 우리@1 | pandas / 우리@10 | Polars@14 / 우리@10 |
|---:|---:|---:|---:|---:|
| 10 | 21.1 | 2.10 | 21.2 | 2.21 |
| 1,000 | 4.83 | 1.72 | 4.88 | 3.65 |
| 100,000 | 1.76 | 1.07 | 2.19 | 0.82 |
| 1,000,000 | 1.70 | 1.16 | 6.09 | 1.60 |

- 10만 행은 기준값(2¹⁷행)의 그늘이다. 대부분 1스레드로 돌아 Polars(14스레드)에 진다.
- 키가 많은 해시(`nunique`, groupby, merge)는 크기와 상관없이 Polars가 10–30% 빠르다. 결과를 처음 나타난 순서로 다시 번호 매기는 단계(스레드 수와 무관한 결과를 위해)가 그 일부다.

## 더 읽을 곳
- [docs/design/frame.md §8.1](../design/frame.md) factorize, [docs/design/frame.md §8.2](../design/frame.md) 키 정규화, [docs/design/frame.md §8.3](../design/frame.md) Swiss table, [docs/design/frame.md §8.7](../design/frame.md) 병렬 factorize
- [docs/design/frame.md §9.1](../design/frame.md) groupby, [docs/design/frame.md §9.2](../design/frame.md) 그룹 순서, [docs/design/frame.md §9.3](../design/frame.md) 정렬 키, [docs/design/frame.md §9.7](../design/frame.md) merge
- [docs/design/frame.md §10.2](../design/frame.md) 토크나이저, [docs/design/frame.md §10.4](../design/frame.md) 숫자 파싱, [docs/design/frame.md §10.6](../design/frame.md) 청크
- [docs/api/frame.md §7](../api/frame.md) groupby, [docs/api/frame.md §9.1](../api/frame.md) merge, [docs/api/frame.md §10](../api/frame.md) read_csv
