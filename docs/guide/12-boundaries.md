# 12장. 영역의 경계

> 세 패키지가 하나가 되면서 경계는 어디에 남았고, 무엇이 사라졌나? frame의 컬럼은 어떻게 복사 없이 linalg까지 가나?

세 영역 사이의 경계는 셋이다.

| 경계 | 지금 | 이 장 |
|---|---|---|
| 확장 모듈 사이 | 없다. `_core` 하나 | §1 |
| 배열 ↔ linalg | C++에서 `NDArray`를 직접 주고받는다 | §2 |
| 배열 ↔ frame | `frame/interop.{hpp,cpp}` 한 파일 | §3 |

그리고 바깥(numpy)과의 경계가 있다 (§4).

## 1. 확장 모듈은 하나
옛 세 패키지는 확장 모듈이 셋이었다. 모듈마다 nanobind 타입 테이블과 정적 링크한 부품(스레드풀 등)이 따로 있었다.
- linalg는 배열의 C++ 타입을 볼 수 없어서 buffer protocol로 읽어야 했다.
- 스레드풀이 세 벌이라 세 패키지를 함께 쓰면 워커도 세 벌이었다.

지금은 `stardust._core` 하나에 모든 영역이 있고, `def_submodule`로 `_core.linalg`, `_core.frame`을 만든다 (DESIGN.md §3.2).
- 검토하고 택하지 않은 안
  - **영역마다 확장 모듈**: 위의 문제가 그대로 남는다.
  - **공유 라이브러리(.dylib) + 영역별 모듈**: 문제는 풀리지만 빌드와 설치(rpath, nanobind 공유 빌드)가 복잡하다. 영역이 아주 많아져 빌드 시간이 문제가 되면 다시 본다.

## 2. 배열 ↔ linalg: 경계를 없앴다
별도 패키지였을 때 linalg는 배열을 buffer protocol로 읽고, 결과를 자기 블록에 만든 뒤 `sd.asarray`로 감쌌다. 1단계 끝(M3)에 이 경계를 없앴다 (docs/learn/M3-linalg-boundary.md).

| 별도 패키지 시절 | 지금 |
|---|---|
| 입력: `PyObject_GetBuffer` → format 문자열로 dtype 판정 → `PyBuffer_Release` | `is_ndarray`로 확인하고 `NDArray`를 빌린다 |
| 결과: 128B 블록 → buffer protocol로 내보냄 → 캐시해 둔 `sd.asarray`로 감쌈 | linalg가 strides를 정하고 배열의 `new_array_strided`가 `NDArray`를 만든다 |
| 결과의 `.base`: 내부 블록 객체 | `None` (배열 규칙과 같다) |
| 나눌 때마다 Python으로 `get_num_threads()`를 부름 | 공용 풀의 `parallel::available()` |
| `DType`, float → int 변환, pairwise 합이 따로 | 하나씩 (`base/`) |

**잰 값** (`bench/merge/m3_boundary.py`, 같은 자리에서 앞뒤로, ns):

| 호출 | 전 | 후 | 차이 |
|---|---:|---:|---:|
| `inv` 2×2 | 402 | 283 | −119 |
| `solve` 3×3 | 414 | 300 | −115 |
| `cholesky` 3×3 | 312 | 188 | −123 |
| `qr` 3×3 (결과 2개) | 1178 | 935 | −243 |
| `svd` 3×3 (결과 3개) | 3246 | 3008 | −238 |
| `det` 2×2 (스칼라) | 136 | 137 | +1 |

- **결과 배열 하나에 약 120ns가 줄었다.** 3×3 `cholesky`는 40% 빨라졌다. 작은 행렬에서는 호출 경로가 계산보다 크다.
- 예상(약 200ns)보다 작았다. `NDArray`와 그 Python 객체를 만드는 데 여전히 약 75ns가 든다. 줄어든 것은 `asarray` 호출, 블록 객체, buffer protocol 왕복이다.
- 스칼라를 돌려주는 함수(`det`, `trace`)는 결과 배열을 만들지 않으므로 그대로다.

**설계 하나를 바꿨다.** 처음 계획은 배열에 `new_array_fortran(dtype, batch, m, n)`을 더하는 것이었다. 대신 strides를 받는 `new_array_strided`를 더했다.
- "행렬마다 열 우선, 배치 축은 바깥에 C 순서"는 LAPACK에서 나온 **linalg의 정책**이다. 배열은 메모리와 플래그만 맡는다.
- 길이 0인 축에서 두 영역의 strides 공식이 다르다(배열은 0인 축을 1로 친다). 골든 출력은 strides도 비교하므로, linalg의 공식을 linalg에 두어야 옛 결과와 같았다.

## 3. 배열 ↔ frame: 파일 하나
frame의 C++은 `array/`를 모른다. 둘을 잇는 코드는 `frame/interop.{hpp,cpp}` 한 곳에만 있다. `array/` 헤더를 include하는 frame 파일은 `interop.cpp` 하나다 (docs/design/frame.md §5.4).

| 함수 | 하는 일 | 쓰는 곳 |
|---|---|---|
| `to_array_view(col, writeable)` | NA 없는 bool/int64/float64 컬럼의 values Buffer 위에 1차원 `NDArray`. 복사 없음 | `Series.to_array` |
| `from_array(arr)` | 1차원 배열을 새 Column으로 복사 | `Series(arr)`, `DataFrame(arr2d)` |
| `columns_to_array(cols, n)` | 같은 dtype의 k개 컬럼으로 (n, k) **F-order** 배열. 컬럼마다 `memcpy` 한 번 | `DataFrame.to_array` |
| `array_sum_float64(col)` | 컬럼 위의 읽기 전용 뷰로 배열의 `sd.sum(x, dtype=float64)` | int64 `mean` |

### 3.1 나갈 때는 공유, 읽기 전용
```python
import stardust as sd

s = sd.frame.Series([1.0, 2.0, 3.0])
v = s.to_array()
print(type(v) is sd.ndarray, v._flags()["writeable"])  # True False
try:
    v[0] = 5.0
except ValueError as e:
    print(e)  # assignment destination is read-only
s.iloc[0] = 10.0
print(v.tolist(), s.tolist())  # [1.0, 2.0, 3.0] [10.0, 2.0, 3.0]
```

- 뷰의 `NDArray::buffer`는 Column과 **같은 `shared_ptr<Buffer>`**다. 뷰가 살아 있는 동안 참조 카운트가 하나 늘어 있으므로, frame 쪽 대입(`s.iloc[0] = 10.0`)은 10장의 Copy-on-Write 규칙대로 먼저 복사한다. **Copy-on-Write 코드는 한 줄도 고치지 않았다.**
- 읽기 전용이어야 하는 이유: 쓸 수 있으면 배열을 통해 Series가 바뀐다.
- 이 호출이 새로 만든 메모리(NA를 채운 것, dtype을 바꾼 것, `copy=True`)는 아무도 들고 있지 않으므로 쓸 수 있는 배열로 내준다.
- NA가 있으면 `na_value=`가 있어야 한다. NA를 몰래 NaN이나 0으로 바꾸지 않는다. string 컬럼은 TypeError다(배열에는 문자열 dtype이 없다).

```python
import stardust as sd

filled = sd.frame.Series([1, 2, None]).to_array(na_value=0)
print(filled.tolist(), filled._flags()["writeable"])  # [1, 2, 0] True
try:
    sd.frame.Series([1, None]).to_array()
except ValueError as e:
    print(e)  # to_array(): cannot convert an int64 column with NA to an array; pass na_value= to fill the NA rows
try:
    sd.frame.Series(["a"]).to_array()
except TypeError as e:
    print(e)  # to_array(): a string column has no array (the array area has no string dtype)
```

### 3.2 들어올 때는 언제나 복사
- `Series(arr)`, `DataFrame(arr2d, columns=...)`는 배열을 복사한다. 배열은 다른 **쓰기 가능한** 뷰를 통해 언제든 바뀔 수 있어서 frame의 Copy-on-Write가 그 쓰기를 볼 수 없다.
- complex128 배열은 TypeError다(frame에 complex가 없다).

```python
import stardust as sd

a = sd.asarray([1.0, 2.0])
t = sd.frame.Series(a)
a[0] = 7.0
print(t.tolist())  # [1.0, 2.0]
d = sd.frame.DataFrame(sd.reshape(sd.arange(6.0), (3, 2)), columns=["p", "q"])
print(d["q"].tolist())  # [1.0, 3.0, 5.0]
```

### 3.3 `DataFrame.to_array`는 F-order
- Column은 컬럼마다 따로 연속이다. (n, k) 배열을 **F-order**로 만들면 컬럼 하나가 연속된 한 덩어리라서 복사가 컬럼마다 `memcpy` 한 번이다.
- 그리고 linalg는 F-order 입력을 가장 빠른 패킹 경로(열마다 복사와 유한성 검사를 한 패스로, 9장 §2)로 읽는다. frame → 배열 → linalg가 레이아웃을 한 번도 뒤집지 않는다.

```python
import stardust as sd

df = sd.frame.DataFrame({
    "x1": [1.0, 2.0, 3.0, 4.0],
    "x2": [0.0, 1.0, 0.0, 1.0],
    "y":  [1.0, 3.0, 3.0, 5.0],
})
X = df[["x1", "x2"]].to_array()
print(X.strides, X._flags()["f_contiguous"])  # (8, 32) True
fit = sd.linalg.lstsq(X, df["y"].to_array())
print([round(b, 12) for b in fit.solution.tolist()], fit.rank)  # [1.0, 1.0] 2
```

### 3.4 int 평균이 배열의 합을 부르는 이유
- 두 영역의 `mean(int64)`가 비트까지 같아야 한다 (DESIGN.md §9.1의 11번).
- 같은 계산을 frame에 다시 쓰면 배열의 덧셈 순서(4096원소 버퍼 단위의 형변환과 블록, 3장 §4, 5장 §7)를 흉내 내야 한다. 배열의 함수를 그대로 부르면 순서가 저절로 같다.
- 대가: 이 호출은 Python 객체(읽기 전용 뷰)를 하나 만들고 배열의 공개 함수를 거치므로 GIL이 필요하다. 그래서 int 평균의 합은 frame이 아니라 배열의 GIL 규칙을 따른다 (6장 §3).
- 그룹별 int 평균은 배열을 부르지 않고 그룹마다 float64 pairwise로 더한다. 그래서 `Series.mean`과 비트가 같은 것은 그룹이 4,096행 이하일 때다. M6의 목록과 다르게 구현한 곳이라 문서에 적었다 (DESIGN.md §9).

```python
import numpy as np
import stardust as sd

x = np.random.default_rng(0).integers(-2**62, 2**62, 100_000)
print(sd.frame.Series(x).mean() == sd.mean(sd.asarray(x)))  # True
```

## 4. 바깥과의 경계: numpy
| 방향 | 배열 `sd` | frame `sd.frame` |
|---|---|---|
| numpy → sd | 버퍼 프로토콜, format이 같으면 **복사 없음** (2장 §7) | **언제나 복사** (int32 등은 넓히면서) |
| sd → numpy | 버퍼 프로토콜, 복사 없음, 쓰기 가능 | `to_numpy()`: NA 없는 숫자 컬럼은 **읽기 전용** 공유 |
| 섞어 쓰기 | `__array_ufunc__ = None`: `np_array + sd_array`는 TypeError | `__array_ufunc__ = None`, `__array__` 있음 |

- 배열과 frame이 다른 이유는 데이터 모델에 있다. 배열에는 Copy-on-Write가 없고, 뷰를 통해 원본이 바뀌는 것이 배열의 의미 그대로다. frame은 Copy-on-Write를 지켜야 해서 바깥 메모리를 받아들이지 않는다 (DESIGN.md §8.4의 "numpy에서 만들기").

## 5. 경계에서 지킨 것
- **층**: `frame/`에서 `array/`를 아는 파일은 하나, `linalg/`는 `array/`를 위에서 쓰고, 아래층(`base/`, `parallel/`, `accelerate/`)은 어느 영역도 모른다. M4에서 이름만 보고 `base/`에 둔 NEON 커널이 사실은 배열의 연산 정의에 의존하는 것을 찾아 배열로 돌려보냈다. 파일 이름이 아니라 **무엇에 의존하는지로** 층을 정한다 (docs/learn/M4-shared-foundation.md).
- **규칙**: 경계를 넘어도 같은 규칙을 만난다. `to_array(dtype=)`는 생성자의 `dtype=`과 같은 명시적 변환이고, `copy`는 배열 `asarray`의 `copy`와 같은 뜻이다(None, True, False).
- **테스트**: `tests/interop/test_frame_array.py`가 공유와 복사, 읽기 전용 뷰, 뷰가 살아 있을 때의 Copy-on-Write, F-order strides, `lstsq(df.to_array(), ...)`, 두 영역이 같아야 하는 규칙(int 평균, int–float 비교, float → int 변환)을 본다.

## 더 읽을 곳
- [DESIGN.md §3.1](../../DESIGN.md) 경계를 없앤다, [DESIGN.md §3.2](../../DESIGN.md) 확장 모듈은 하나, [DESIGN.md §4.3](../../DESIGN.md) Column과 배열 사이
- [ARCHITECTURE.md §3](../../ARCHITECTURE.md) Python과 C++의 경계, [ARCHITECTURE.md §1](../../ARCHITECTURE.md) 층 구조
- [docs/design/linalg.md §3.2](../design/linalg.md) 입력, [docs/design/linalg.md §3.3](../design/linalg.md) 결과, [docs/design/linalg.md §3.6](../design/linalg.md) 경계를 없앤 효과
- [docs/design/frame.md §5.4](../design/frame.md) 배열과의 경계, [docs/api/frame.md §2.8](../api/frame.md) `to_array`와 `Series(배열)`
