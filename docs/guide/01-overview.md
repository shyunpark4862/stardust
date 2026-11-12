# 1장. 전체 그림

> 무엇이 있고, 세 영역은 어떻게 한 라이브러리가 되었나? `x + y` 한 줄은 코드 어디를 지나가나?

## 1. 무엇을 만들었나
stardust는 배열, 선형대수, 데이터프레임을 처음부터 다시 만든 교육용 라이브러리다. `import stardust as sd` 하나로 세 영역을 쓴다.

| 영역 | 이름공간 | 닮은 것 | 핵심 객체 |
|---|---|---|---|
| 배열 | `sd` | numpy | `sd.ndarray` (N차원, 임의의 strides) |
| 선형대수 | `sd.linalg` | numpy.linalg, scipy.linalg | 함수 37개. 배열을 받아 배열을 돌려준다 |
| 데이터프레임 | `sd.frame` | pandas | `Series`, `DataFrame`, `Index` |

- 계산 코어는 C++20, Python 바인딩은 nanobind다. "얇은 Python, 두꺼운 C++"가 원칙이다.
- 대상 기기는 Apple M4 Pro 한 대다. NEON(128비트 SIMD), Accelerate(BLAS, LAPACK, vForce)를 쓰고, 스레드풀은 직접 만들었다.
- numpy, scipy, pandas, Polars가 정답지이자 비교 기준이다. 결과는 대부분 numpy와 비트까지 같고, 다른 곳은 문서에 이유와 함께 적었다.

세 영역을 함께 쓰는 짧은 예다. 표에서 열을 골라 행렬로 바꾸고, 최소제곱으로 계수를 구하고, 예측값을 다시 표에 넣는다.

```python
import stardust as sd

df = sd.frame.DataFrame({
    "x1": [1.0, 2.0, 3.0, 4.0],
    "x2": [0.0, 1.0, 0.0, 1.0],
    "y":  [1.0, 3.0, 3.0, 5.0],
})
X = df[["x1", "x2"]].to_array()                # (4, 2) sd.ndarray
beta = sd.linalg.lstsq(X, df["y"].to_array()).solution
df["pred"] = sd.frame.Series(X @ beta)         # 배열 → Series (복사)
print(X.shape)  # (4, 2)
print([round(b, 12) for b in beta.tolist()])  # [1.0, 1.0]
print([round(v, 12) for v in df["pred"].tolist()])  # [1.0, 3.0, 3.0, 5.0]
```

- `to_array()`는 frame의 컬럼을 배열로 내보낸다. 12장에서 이 경계를 자세히 본다.
- `sd.linalg.lstsq`는 `sd.ndarray`만 받는다. 리스트를 주면 TypeError다 (8장).
- `X @ beta`는 배열 영역의 `matmul`이다. linalg가 행렬곱을 다시 만들지 않는다.

## 2. 셋이 하나가 된 방식
옛 세 프로젝트는 따로 만든 패키지였다(`stardust_numpy`, `stardust_linalg`, `stardust_pandas`). 합칠 때 정한 큰 줄기는 DESIGN.md §1에 있다.

- **하나의 라이브러리를 키운다.** 새 영역은 하위 모듈로 붙는다. 최상위 `sd`에는 배열만 둔다 (DESIGN.md §1.1).
- **두 단계로 합쳤다** (DESIGN.md §1.2).
  - 1단계: 배열 + linalg를 깊게 합쳤다. 둘은 같은 배열을 다루므로 linalg가 배열의 C++ 내부를 직접 쓴다.
  - 2단계: pandas를 `sd.frame`으로 옮겼다. frame의 Column은 배열 위에 다시 올리지 않았다. 아래층(Buffer, 스레드풀, pairwise 합, 빌드)만 공유한다.
- **일관성이 가장 앞이다** (DESIGN.md §1.4). 같은 개념은 어느 영역에서든 같은 이름, 같은 시그니처, 같은 동작이다. 차이는 데이터 모델에 이유가 있을 때만 남긴다(예: frame에만 결측 NA가 있다). 13장의 주제다.

## 3. 층 구조
```
          Python:  stardust (sd)    sd.linalg    sd.frame
                        │               │            │
bind/  ─────────────────┴───────────────┴────────────┴──────  _core 확장 모듈 하나
                        │               │            │
                     array/  ◀───── linalg/        frame/
                        │                            │
                        ├────────────┬───────────────┤
                   accelerate/   parallel/         base/
```
- **아래로만 의존한다.** `array/`는 `linalg/`와 `frame/`을 모른다. `base/`, `parallel/`, `accelerate/`는 어느 영역도 모른다 (ARCHITECTURE.md §1).
- `linalg/`는 `array/`의 `NDArray`를 직접 받고 만든다 (8장, 12장).
- `frame/`은 `array/`를 모른다. 예외는 둘을 잇는 파일 `frame/interop.cpp` 하나다 (12장).
- 아래층에 있는 공통 부품

  | 디렉터리 | 부품 | 쓰는 영역 |
  |---|---|---|
  | `base/` | `Buffer`(참조 카운트 메모리), `DType`, pairwise 합의 정의, float → int64 변환 | 셋 다 |
  | `parallel/` | 스레드풀 하나 | 셋 다 |
  | `accelerate/` | BLAS, LAPACK, vForce 래퍼 | 배열, linalg |

- 확장 모듈은 `stardust._core` 하나다. linalg와 frame은 그 하위 모듈(`_core.linalg`, `_core.frame`)이다 (DESIGN.md §3.2).

```python
import stardust as sd

print(sd.linalg.solve.__module__)  # stardust._core.linalg
print(type(sd.frame.Series([1]).to_array()) is sd.ndarray)  # True
```

## 4. `x + y` 한 줄의 여정
배열 영역에서 덧셈 한 번이 지나가는 길이다. 뒤의 장들이 이 길의 각 칸을 하나씩 연다.

```
x + y
 │  Python이 ndarray.__add__를 부른다            bind/array_bind.cpp (C++ 바인딩, Python 래퍼 없음)
 ▼
binary<Add>(x, y, out=None, "+")
 │  1. 피연산자: ndarray 또는 Python 스칼라         array/ops/ufunc.cpp
 │  2. 루프 고르기: dtype 승격 → 루프가 있는 첫 dtype   (3장)
 │  3. 변환: dtype이 다른 입력은 4096원소씩 캐스팅      array/kernels/buffered.hpp (3장)
 │  4. 실행: 브로드캐스팅 → 축 재정렬 → 합치기         array/iter.cpp의 NdIter (3장)
 ▼
1차원 커널 table[float64]                            array/kernels/elementwise.hpp (4장)
 │  연속이면 인덱스 루프 → 컴파일러가 NEON으로 벡터화
 │  일이 2¹⁵원소를 넘으면 구간을 스레드풀에 나눔       parallel/pool.cpp (6장)
 ▼
새 NDArray (Buffer 하나, 128B 정렬) → Python 객체      array/ndarray.hpp (2장)
```

- 작은 배열에서는 계산보다 이 길의 고정 비용이 크다. 그래서 인자 파싱, 브로드캐스팅, 인덱스 해석까지 C++에서 한다. 작은 `a + b` 하나가 약 180ns다(numpy 158ns, docs/design/array.md §11).
- 같은 덧셈이 frame에서는 다른 길로 간다. Series는 Python 클래스이고, C++ 커널 `binary_op` 하나를 부른다 (10장). linalg의 `solve(a, b)`는 패킹, 배치 드라이버, LAPACK을 지난다 (9장).

## 5. 세 영역이 함께 지키는 약속
| 약속 | 뜻 | 자세히 |
|---|---|---|
| dtype 객체 다섯 | `sd.bool, sd.int64, sd.float64, sd.complex128, sd.string`. 인자는 객체나 그 이름 문자열 | API.md §1 |
| 에러 종류 | 타입이면 TypeError, 값이면 ValueError. 경고는 없다 | API.md §2 |
| 에러 메시지 | 공개 함수의 에러는 `이름(): `으로 시작한다 | API.md §2.1 |
| 결정성 | 결과는 스레드 수와 무관하게 비트까지 같다 | API.md §3 |
| 스레드 수 | `sd.set_num_threads(n)` 하나가 세 영역을 함께 바꾼다 | API.md §3 |

```python
import stardust as sd

print(sd.zeros(2, dtype="int64").dtype is sd.int64)  # True
print(sd.frame.Series([1.5]).dtype is sd.float64)  # True
try:
    sd.linalg.inv(sd.zeros((2, 2)))
except sd.linalg.LinAlgError as e:
    print(e)  # inv(): singular matrix
```

## 더 읽을 곳
- [DESIGN.md §1.1](../../DESIGN.md) 이름과 확장 방식, [DESIGN.md §1.4](../../DESIGN.md) 일관성 원칙
- [ARCHITECTURE.md §1](../../ARCHITECTURE.md) 층 구조, [ARCHITECTURE.md §2](../../ARCHITECTURE.md) 디렉터리
- [docs/design/array.md §12.1](../design/array.md) 얇은 Python, 두꺼운 C++
- [docs/design/array.md §4.4](../design/array.md) ufunc 디스패치의 네 단계
