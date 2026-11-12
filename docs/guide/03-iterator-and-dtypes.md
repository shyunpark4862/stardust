# 3장. 브로드캐스팅, 이터레이터, dtype

> N차원 연산을 빠른 1차원 루프의 반복으로 바꾸는 방법은? 결과 dtype과 변환 규칙은 어떻게 정했고, 왜 세 영역이 같은 규칙을 쓰나?

## 1. 커널은 1차원이다
배열 영역의 모든 원소별 연산과 리덕션은 같은 모양의 C++ 함수 하나로 계산한다 (`array/iter.hpp`).

```cpp
using Loop = void (*)(std::byte* const* args, std::int64_t n, const std::int64_t* strides, void* ctx);
```

- 피연산자 포인터들, 원소 수 `n`, 피연산자마다의 바이트 stride. numpy의 inner loop와 같은 모양이다.
- 커널은 N차원을 모른다. N차원 문제를 "1차원 구간 여러 개"로 바꾸는 것이 이터레이터의 일이다.

## 2. nd 이터레이터 `NdIter`
`array/iter.cpp`의 `NdIter`는 numpy `nditer`의 축소판이다. 목표는 **적고, 길고, 메모리를 따라가는** 1차원 구간이다.

```
입력 x: (3, 1, 4) C 순서,  y: (4,)          z = x + y
1. 브로드캐스팅   공통 shape (3, 1, 4). y는 앞에 축이 붙고 그 축의 stride는 0
2. 크기 1 축 제거  (3, 4)
3. 축 재정렬     stride 절댓값이 큰 축을 바깥에. 출력부터 보고, 정해지지 않으면 앞 피연산자를 차례로
4. 합치기        모든 피연산자에서 "바깥 한 칸 = 안쪽 extent칸"이면 두 축을 하나로
5. 실행          바깥 축은 odometer로 돌고, 가장 안쪽 구간마다 커널을 한 번 부른다
```

- **합치기(coalescing)**: 연속 (3, 4, 5) 배열끼리의 덧셈은 길이 60인 구간 하나가 된다. 커널 호출 한 번이다.
- **재정렬**: 전치한 배열(`x.T + 1`)도 메모리 순서로 읽는다. 원소별 연산의 새 출력은 첫 배열 피연산자의 레이아웃을 따르므로(2장 §4) 재정렬 뒤에도 쓰기가 순차적이다.
- stride 0 축(브로드캐스트)은 순서를 정할 정보가 없어서 건너뛴다.
- 이 단계들의 효과를 따로 잴 수 있게 모드 셋(`Naive`, `Reorder`, `Full`)을 남겨 두었다. 실제 연산은 `Full`이다. 예외: 인덱스 하나짜리 scatter는 반복된 인덱스에서 마지막 쓰기가 이겨야 하므로 `Naive`(C 순서, 1스레드)다 (7장).
- 일이 기준값(float64 원소 2¹⁵개)을 넘으면 `NdIter::plan`이 구간을 태스크로 나눈다. 나누는 방식은 shape, 레이아웃, dtype만으로 정한다 (6장).

```python
import stardust as sd

x = sd.arange(12.0).reshape((3, 1, 4))
y = sd.asarray([10.0, 20.0, 30.0, 40.0])
print((x + y).shape)  # (3, 1, 4)
try:
    sd.asarray([1.0, 2.0]) + sd.asarray([1.0, 2.0, 3.0])
except ValueError as e:
    print(e)  # shapes could not be broadcast together: (2,) (3,)
```

## 3. dtype 승격
dtype은 넷이고 순서가 일직선이다: `bool < int64 < float64 < complex128`.
- 결과 dtype은 두 피연산자 중 높은 쪽이다. `promote(a, b) = max(a, b)` 하나다 (`base/dtype.hpp`).
- **Python 스칼라도 같은 규칙이다.** `int`는 int64, `float`는 float64다. numpy의 "약한 스칼라"(int 배열 + 0.5가 상황에 따라 다른 것) 같은 예외가 없다.
- **루프 고르기**: 승격한 dtype부터 위로 올라가며, 그 연산의 루프가 있는 첫 dtype을 쓴다. 아래 규칙은 모두 이 한 줄에서 나온다.

| 연산 | 있는 루프 | 그래서 |
|---|---|---|
| 산술 `+ - *` | int64, float64, complex128 | bool 산술은 int64다 (`True + True == 2`) |
| `/`, 초월함수, `sqrt` | float64, complex128 | 정수는 float64로 계산한다 |
| `//`, `%` | int64, float64 | 정수 결과, 0으로 나누면 0과 `a` (4장) |
| 비교 | 모든 dtype | bool끼리도 비교한다 |
| `logical_*` | bool | int64는 TypeError (Array API) |
| 순서 비교, `max`, `sort` | complex128 없음 | complex에는 순서가 없다 |

```python
import stardust as sd

i = sd.asarray([1, 2])
print((i + 0.5).dtype, (i / 2).dtype, (i // 2).dtype)  # float64 float64 int64
print((sd.asarray([True]) + True).tolist())  # [2]
try:
    sd.logical_and(i, i)
except TypeError as e:
    print(e)  # logical_and(): not supported for int64 and int64
```

- frame도 같은 규칙이다. bool 산술은 int64이고(옛 pandas판은 TypeError였다), int와 float의 비교는 int를 float64로 바꿔서 한다 (DESIGN.md §9.1).

## 4. 버퍼 단위 캐스팅
피연산자의 dtype이 루프 dtype과 다르면(`int64 + float64`는 float64 루프) 변환이 필요하다.
- 입력 전체를 먼저 float64 배열로 바꾸면, 그 임시 배열을 쓰고 다시 읽느라 메모리를 두 번 더 지난다.
- 그래서 **4096원소씩** 작은 버퍼로 변환하고 바로 커널을 돌린다 (`array/kernels/buffered.hpp`의 `BufferedLoop`). 메모리 왕복은 입력과 출력의 것뿐이다.
- 왜 4096인가: 피연산자 3개 × 32KB가 P코어 L1 캐시(128KB)에 들어간다. 256부터 65536까지 쟀고 1024–4096이 가장 빨랐다.
- `BufferedLoop`는 커널과 같은 시그니처다. 이터레이터는 그 존재를 모른다.
- 리덕션도 같은 래퍼로 입력을 바꾼다. 그래서 `mean(int64)`는 4096원소 단위로 접힌다. 이 순서는 5장과 12장에서 다시 나온다(frame의 int 평균이 배열의 합을 부르는 이유).

## 5. 변환은 세 갈래
변환이 언제 허용되는지는 "누가 요청했는가"로 정한다. 값을 보고 정하지 않는다.

| 갈래 | 어디서 | 규칙 |
|---|---|---|
| **명시적** | `astype`, 생성 함수의 `dtype=` | 손실이 있어도 바꾼다. float → int64는 0 쪽으로 자르고 포화, NaN → 0 |
| **암묵적** | 대입 `x[...] = v`, `out=`, `x += y`, `dtype=` 없는 `full_like` | 승격 순서의 **위로만**. 아래로 가면 값과 상관없이 TypeError |
| (frame만) **정확한 값** | 조회 `x in s` | 컬럼 dtype에 정확히 같은 값이 있을 때만 (10장) |

```python
import stardust as sd

f = sd.asarray([1.7, -1.7, float("nan"), 1e300])
print(sd.astype(f, sd.int64).tolist())  # [1, -1, 0, 9223372036854775807]
x = sd.asarray([1, 2, 3])
try:
    x[0] = 1.5
except TypeError as e:
    print(e)  # cannot implicitly convert float64 to int64 (that would lose information); convert explicitly with astype
y = sd.zeros(3)
y[0] = 1          # int → float64는 위로 가는 변환
print(y.tolist())  # [1.0, 0.0, 0.0]
```

- **float → int64**는 ARM의 `FCVTZS` 명령 하나가 정확히 "0 쪽으로 자르고, 포화, NaN → 0"이다. C++의 평범한 캐스트는 NaN과 범위 밖에서 정의되지 않은 동작이고, 분기 셋으로 쓰면 벡터화되지 않는다(`astype` 464µs → `FCVTZS`로 89µs, docs/design/array.md §2.4).
  - 이 함수(`saturate_to_int64`)는 `base/saturate.hpp`에 있고 세 영역이 함께 쓴다: 배열의 `astype`, linalg의 `trace(dtype=)`, frame의 `astype`.
- **complex → 실수**는 TypeError다. 허수부를 잃는다. `sd.real(x)`를 쓴다.
- **범위 밖 Python int**는 int64가 되어야 할 때만 OverflowError다. float와 만나면 float가 된다.

```python
import stardust as sd

try:
    sd.asarray([1, 2]) + 2**70
except OverflowError as e:
    print(e)  # Python int too large to convert to int64
print(sd.asarray([2**70, 0.5]).tolist())  # [1.1805916207174113e+21, 0.5]
try:
    sd.astype(sd.asarray([1j]), sd.float64)
except TypeError as e:
    print(type(e).__name__)  # TypeError
```

## 6. 왜 이렇게 정했나
- **결과의 dtype은 데이터가 아니라 입력의 dtype으로 정한다.** 같은 코드가 어떤 데이터에서는 int, 어떤 데이터에서는 float를 내지 않는다. frame도 같은 원칙이다 (docs/design/frame.md §2.4).
- **에러는 타입 때문에만, 값 때문에는 아니다.** 원소별 연산은 값 때문에 실패하지 않는다: 정수 오버플로는 wrap, 0 나누기는 정해진 값, float는 IEEE 754 (4장). 변환도 "아래로 가는가"만 보고 값은 보지 않는다.
  - 옛 pandas판은 대입에서 값을 봤다(int 컬럼에 `2.0`은 되고 `1.5`는 ValueError). 일관성 원칙에 따라 배열 규칙으로 맞췄다 (DESIGN.md §9.1의 4번).
- 같은 규칙이 세 영역에 있으므로 frame에서 꺼낸 배열이나 linalg에 넣은 배열이 다른 규칙을 만나지 않는다.

## 더 읽을 곳
- [docs/design/array.md §4.1](../design/array.md) 커널 형태, [docs/design/array.md §4.2](../design/array.md) nd 이터레이터, [docs/design/array.md §4.3](../design/array.md) 버퍼 단위 캐스팅
- [docs/design/array.md §2.1](../design/array.md) dtype과 승격, [docs/design/array.md §2.4](../design/array.md) 명시적 변환, [docs/design/array.md §2.5](../design/array.md) 암묵적 변환
- [docs/api/array.md §1.4](../api/array.md) dtype 승격, [docs/api/array.md §1.5](../api/array.md) 변환 규칙
- [docs/design/frame.md §2.4](../design/frame.md) frame의 변환 세 갈래
