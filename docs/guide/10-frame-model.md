# 10장. `sd.frame`의 데이터 모델

> Column, NA, Copy-on-Write는 무엇이고, 왜 frame을 배열 위에 다시 올리지 않았나?

## 1. 쓰기
`sd.frame`에는 `Series`, `DataFrame`, `Index`, `RangeIndex`, `NA`와 함수 `concat`, `merge`, `read_csv`가 있다. pandas와 비슷하게 쓰지만 이름과 시그니처는 라이브러리 전체의 규칙(Array API 관례)을 따른다.

```python
import stardust as sd

df = sd.frame.DataFrame({
    "k": ["b", "a", "b", "a"],
    "x": [1, 2, 3, None],
    "y": [0.5, 1.5, float("nan"), 2.0],
})
print(df.dtypes)  # {'k': string, 'x': int64, 'y': float64}
print(df["x"].dtype is sd.int64)  # True
print(df.sort_values("y", descending=True)["y"].tolist())  # [nan, 2.0, 1.5, 0.5]
try:
    df.sort_values(by="y")
except TypeError as e:
    print(type(e).__name__)  # TypeError
```

- dtype은 넷이다: `sd.bool`, `sd.int64`, `sd.float64`, `sd.string`. complex는 없다. `.dtype`은 dtype 객체, `DataFrame.dtypes`는 `{컬럼: dtype 객체}` dict다.
- 주 인자(`sort_values`의 `by`, `head`의 `n`)는 위치로만, 옵션은 키워드로만 준다. 정렬 방향은 `descending`이다 (pandas의 `ascending`이 아니다).
- `repr`은 pandas와 글자까지 같다 (테스트가 행 0–100, 열 0–23에서 비교한다).

## 2. Column: 1차원, 결측을 아는 열
frame의 C++ 쪽 단위는 `Column`(`frame/column.hpp`)이다.

```
Column (값 타입: 정수 넷 + shared_ptr 셋)
├─ dtype, length, offset, null_count
├─ values   ──► Buffer [v0 v1 v2 ...]          bool 1B, int64/float64 8B, string은 UTF-8 바이트
├─ validity ──► Buffer [1 0 1 1 ...] (비트)     null_count == 0이면 없다
└─ offsets  ──► Buffer [0 3 3 7 ...] (int64)    string만. 행 i는 values[o[i], o[i+1])
```

- Buffer는 배열과 같은 `base/buffer.hpp`의 것이다(2장 §2).
- **슬라이스는 복사하지 않는다.** `offset`과 `length`만 바꾼다. `head()`가 1M 행에서 656ns다.
- 문자열은 Arrow 레이아웃이다(offsets + 연속 UTF-8). 해시, 비교, 정렬이 Python 객체를 거치지 않고 바이트를 바로 읽는다. 대가로 `tolist()`는 원소마다 str 객체를 새로 만든다.

### 2.1 왜 배열(NDArray) 위에 올리지 않았나
| | NDArray | Column |
|---|---|---|
| 차원 | N차원 | 1차원 |
| 메모리 | 임의의 strides | 연속, `offset`만 |
| 결측 | 없다 | validity 비트맵 |
| 문자열 | 없다 | offsets + UTF-8 |

- 둘은 다른 물건이다. Column을 NDArray 위에 다시 올리면 옛 pandas 프로젝트의 설계(M1–M12)를 뒤집게 되고 얻는 것이 적다 (DESIGN.md §4.1).
- 대신 **아래층을 공유한다**: Buffer, 스레드풀, pairwise 합의 정의, `saturate_to_int64`, 빌드 플래그, `DType`. 그리고 둘을 잇는 파일을 하나 두었다 (12장).

## 3. NA와 NaN
- **결측은 `NA` 하나다.** 모든 dtype이 validity 비트맵으로 결측을 갖는다.
- **int 컬럼은 결측이 생겨도 int다.** pandas의 고전 방식(결측이 생기면 float로 바뀜)을 따르지 않는다.
- **NaN은 결측이 아니라 평범한 float 값이다.** `isna()`는 NA만 본다. NaN 판별은 `isnan()`이다. `skipna=True`는 NA만 건너뛰고 NaN은 전파된다.

```python
import stardust as sd

s = sd.frame.Series([1, None, 3])
print(s.dtype, s.isna().tolist(), s.sum(), s.mean())  # int64 [False, True, False] 4 2.0
f = sd.frame.Series([1.0, float("nan"), None])
print(f.isna().tolist(), f.isnan().tolist())  # [False, False, True] [False, True, <NA>]
print(f.sum(), f.sum(skipna=False))  # nan <NA>
```

- NA 스칼라와의 연산은 **반대쪽 dtype의 1행 NA Column**과의 연산이다. `int + NA`는 int64, `int / NA`는 float64다. NA를 위한 dtype 규칙이 따로 없다.
- 배열과의 차이는 NA에서만 나온다: `int // 0`, `int % 0`, `int ** 음수`가 NA이고(배열은 0, `a`, 정수 결과), 값이 모자란 리덕션도 NA다(배열은 NaN이나 ValueError).

## 4. 비트맵과 "NA 자리의 값은 0"
- 비트맵은 행마다 1비트, LSB 우선(Arrow와 같다), 1이 값 있음이다. **결측이 없으면 비트맵 자체가 없다.** 커널은 이것을 보고 NA 검사 없는 루프를 고른다. 데이터를 보는 휴리스틱이 아니라 컬럼의 속성이다.
- 비트를 행마다 꺼내 분기하면 느리다. 비트맵 1바이트를 256칸 표로 읽어 0/1 바이트 8개로 펼치고, 커널은 그 바이트로 분기 없이 고른다.

  | NA 10%, 1M 행 | 행마다 비트 | 바이트로 펼침 |
  |---|---:|---:|
  | `isna` | 377 µs | 47 µs |
  | `bool & bool` (Kleene) | 1.43 ms | 542 µs |
  | float `std` | 1.03 ms | 475 µs |

- **NA 행의 값 자리는 언제나 0이다.** 결과를 만드는 커널은 끝에 `finish_fixed`로 NA 행의 값을 분기 없이 지운다.
  - 해시와 비교에 쓰레기 값이 섞이지 않는다(`NA + 5`의 값 자리에 5가 남으면 나중에 키가 된다).
  - float 합이 NA 행(0.0)을 분기 없이 그대로 더한다. NA 10%인 `sum`이 86µs, pandas 1.33ms다.

## 5. 얇은 Python, C++ Column
배열은 ndarray까지 C++로 만들었다. frame은 그렇게 하지 않았다 (docs/design/frame.md §3.1).
- **C++**: Buffer, 비트맵, Column, 해시 테이블, 모든 커널
- **Python**: `Series`, `DataFrame`, `Index`, `GroupBy` 클래스, 인자 검증, 정렬(alignment) 절차, `repr`
- 이유를 쟀다. 같은 Column 위에 세 모양을 만들었다(원소 10개).

  | 연산 | C++ Series | Python Series | pandas |
  |---|---:|---:|---:|
  | `s + s` | 92 ns | 256 ns | 9.16 µs |
  | `df[df["a"] > 0]` | 400 ns | 1.10 µs | 39.5 µs |

  Python 층이 더하는 비용은 연산 하나에 수백 ns다. pandas의 고정비(수 µs)보다 훨씬 작고, 데이터가 크면 커널 시간이 전부다. `merge`처럼 옵션이 많은 API를 C++로 파싱하면 코드가 길고 읽기 어렵다.
- `Column`은 nanobind에 **값 타입**으로 바인딩한다. `shared_ptr<Column>` holder로 돌려주면 nanobind가 객체마다 힙 복사본을 만들고 전역 표에 등록해서 `add`가 190ns, 값 타입이면 89ns다.

## 6. Copy-on-Write
```python
import stardust as sd

df = sd.frame.DataFrame({"a": [1, 2, 3]})
col = df["a"]
col.iloc[0] = 100
print(df["a"].tolist(), col.tolist())  # [1, 2, 3] [100, 2, 3]
```

`col`만 바뀌고 `df`는 그대로다(pandas 3과 같다). 쓰기는 두 단계다 (docs/design/frame.md §5.2).
1. 쓰는 쪽은 먼저 자기만의 Column 객체를 갖는다(`self._col = self._col.share()`). Buffer는 공유한 채다.
2. `set_rows`가 values와 validity Buffer의 **참조 카운트**(`use_count()`)를 본다.
   - 1이면 그 자리에서 쓴다.
   - 다른 주인(다른 Series, DataFrame, 슬라이스, 내보낸 numpy/배열 뷰)이 있으면 자기 행 범위만 새 Buffer로 복사한 뒤 쓴다.

```
df._cols["a"] ──► Column A ──┐
                              ├──► Buffer (use_count 2)
col._col ───────► Column B ──┘          col.iloc[0] = 100 → 복사한 뒤 쓴다
```

- 복사는 필요할 때만 한다. 아무도 Buffer를 들고 있지 않으면 그 자리에서 쓴다. 10행 Series의 `s.iloc[3] = 1`이 0.94µs다(pandas 5.94µs).
- 그래서 frame의 API에는 `copy`, `inplace` 인자가 없다. `to_numpy`, `to_array`만 예외다(밖으로 내보내는 메모리를 공유할지 정한다).
- **스냅샷 규칙**: 연산은 호출 시점의 값을 본다. 커널이 GIL을 놓으므로 다른 스레드의 대입과 겹칠 수 있지만, 진행 중인 연산이 옛 Column을 잡고 있어 참조 카운트가 2 이상이고 대입은 복사한다. 남은 틈 하나(그 자리에서 쓰는 도중에 연산이 시작되는 경우) 때문에 `set_rows`는 크기와 상관없이 GIL을 쥔 채 돈다.

**밖으로 내보낸 메모리**도 같은 규칙으로 지킨다.
- `to_numpy()`, `to_array()`는 NA 없는 숫자 컬럼의 Buffer를 그대로 가리키는 **읽기 전용** 배열이다. 그 배열이 Buffer 참조를 하나 쥐고 있으므로, frame 쪽에 쓰면 frame이 먼저 복사한다. 배열의 값은 바뀌지 않는다.
- 들어올 때(`Series(numpy 배열)`, `Series(sd 배열)`)는 **언제나 복사한다.** 밖의 메모리는 frame이 모르는 뷰로 언제든 바뀔 수 있다.

```python
import numpy as np
import stardust as sd

s = sd.frame.Series([1, 2, 3])
view = s.to_numpy()
s.iloc[1] = 7
print(view.tolist(), view.flags.writeable, s.tolist())  # [1, 2, 3] False [1, 7, 3]
a = np.arange(3.0)
t = sd.frame.Series(a)
a[0] = 9.0
print(t.tolist())  # [0.0, 1.0, 2.0]
```

## 7. 값 규칙은 배열과 같다
M6에서 frame의 값 규칙을 배열 규칙으로 맞췄다 (DESIGN.md §9.1). 3장의 세 갈래 변환이 frame에도 그대로 있다.

| 갈래 | frame에서 | 규칙 |
|---|---|---|
| 명시적 | 생성자의 `dtype=`, `astype`, `to_numpy(dtype=)` | 손실이 있어도 바꾼다. `Series([1.5], dtype="int64")`는 `[1]` |
| 암묵적 | 행 대입, `fillna` | dtype만 보고 위로만. int 컬럼에 `2.0`도 TypeError |
| 정확한 값 | 조회 `x in s` | 컬럼 dtype에 정확히 같은 값이 있을 때만 |

```python
import stardust as sd

print(sd.frame.Series([1.5, -2.5], dtype="int64").tolist())  # [1, -2]
try:
    sd.frame.Series([1, None]).fillna(0.5)
except TypeError as e:
    print(e)  # fillna(): cannot implicitly convert float64 to int64 (that would lose information); convert explicitly with astype
big = sd.frame.Series([2.0**53])
print((big == 2**53 + 1).tolist(), 2**53 + 1 in big)  # [True] False
```

- `==`는 배열처럼 int를 float64로 바꿔 비교한다(`2**53 + 1`은 `2.0**53`으로 반올림된다). `x in s`는 "이 컬럼에 `x`와 같은 값이 있는가"를 묻는다. float64 컬럼에는 `2**53 + 1`과 같은 값이 없으므로 False다. 둘은 일부러 다르다.

## 8. 작은 데이터
10행 연산에서는 호출 비용이 시간을 정한다 (docs/design/frame.md §3.3).

| 원소 10개 | sd.frame | pandas | Polars |
|---|---:|---:|---:|
| `s + s` | 454 ns | 9.93 µs | 737 ns |
| `df["a"]` | 175 ns | 3.22 µs | 300 ns |
| `df[df["a"] > 0]` | 1.73 µs | 43.0 µs | 31.2 µs |

- 10행에서 pandas보다 기하평균 21배, Polars보다 2.1배 빠르다.
- 측정에서 하나 주의할 것: 같은 입력으로 수천 번 부르면 분기 예측기가 그 입력을 외운다. 분기가 많은 코드일수록 실제보다 빨라 보인다. 그래서 frame은 "새 입력" 모드로도 잰다 (docs/design/frame.md §12.1, 13장).

## 더 읽을 곳
- [docs/api/frame.md §1.1](../api/frame.md) 시그니처와 이름, [docs/api/frame.md §1.4](../api/frame.md) NA, [docs/api/frame.md §1.3](../api/frame.md) NaN
- [docs/design/frame.md §4.3](../design/frame.md) Column, [docs/design/frame.md §4.4](../design/frame.md) 비트맵, [docs/design/frame.md §4.5](../design/frame.md) NA 자리의 값
- [docs/design/frame.md §5.2](../design/frame.md) Copy-on-Write, [docs/design/frame.md §5.3](../design/frame.md) 스냅샷 규칙, [docs/design/frame.md §3.4](../design/frame.md) numpy와 주고받기
- [docs/design/frame.md §2.4](../design/frame.md) 변환의 세 갈래, [docs/design/frame.md §2.6](../design/frame.md) 배열 영역과 다른 곳
