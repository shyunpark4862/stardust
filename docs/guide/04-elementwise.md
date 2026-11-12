# 4장. 원소별 연산

> `sin(x)`나 `x * y`는 어떻게 빠르게 계산되고, numpy와 비트까지 같게 하려면 무엇이 필요한가?

## 1. 연산 하나는 구조체 하나
원소별 연산의 **의미**는 `array/kernels/scalar_ops.hpp`에 연산마다 구조체 하나로 적는다.
- 어느 저장 타입에 루프가 있는지(`ok<T>`)와 원소 하나의 함수(`f`)다. 3장의 루프 고르기 표가 이 `ok<T>`에서 나온다.
- `array/kernels/elementwise.hpp`가 이것을 1차원 루프로 바꾸고, 연산마다 `table[dtype]`(루프와 출력 dtype)을 템플릿 인스턴스화로 채운다.
- 캐스트도 같은 모양의 표 `cast[from][to]`다 (16개, `array/kernels/cast.hpp`).
- Python 스칼라 피연산자는 루프 dtype의 원소 1개짜리 배열이 되고, 이터레이터가 stride 0으로 브로드캐스트한다.

## 2. 값 때문에 실패하지 않는다
원소별 연산은 타입이나 shape이 맞지 않을 때만 에러다. 값은 자기 타입의 표준 산술을 따른다 (docs/design/array.md §2.3).

| 경우 | 결과 | 이유 |
|---|---|---|
| int64 오버플로 | wrap (`INT64_MAX + 1 == INT64_MIN`) | C++의 부호 있는 오버플로는 정의되지 않은 동작이라 부호 없는 정수로 계산한 뒤 되돌린다 |
| `a // 0`, `a % 0` | `0`, `a` | `a == (a // b) * b + a % b`가 b = 0에서도 성립한다 |
| 정수 `** 음수` | 밑이 1이면 1, −1이면 ±1, 그 외 0 | 정수 결과를 낸다 |
| 시프트 64 이상(음수 포함) | 모든 비트가 빠진다 | 시프트 수를 부호 없는 수로 읽는다 |
| float, complex | IEEE 754 (`inf`, `nan`) | |
| 경고 | 없다. `errstate`도 없다 | |

```python
import stardust as sd

print((sd.asarray([7, -7]) // 0).tolist(), (sd.asarray([7, -7]) % 0).tolist())  # [0, 0] [7, -7]
print((sd.asarray([2, 1, -1]) ** -1).tolist())  # [0, 1, -1]
print((sd.asarray([2**63 - 1]) + 1).tolist())  # [-9223372036854775808]
print((sd.asarray([1]) << 64).tolist(), (sd.asarray([-8]) >> 70).tolist())  # [0] [-1]
```

- frame에서는 `int // 0`, `int % 0`, `int ** 음수`가 **NA**다. frame에는 결측이 있어서 "정의되지 않음"을 값으로 나타낼 수 있다. 세 영역 사이에 남은 몇 안 되는 차이고, 이유가 데이터 모델에 있다 (DESIGN.md §9.1의 1번, 10장).

## 3. 연속 fast path
- 원소마다 바이트 stride를 더하는 루프는 어떤 최적화 수준에서도 벡터화되지 않는다.
- 그래서 루프가 먼저 흔한 레이아웃을 본다: **모두 연속**(`o[i] = f(a[i], b[i])`), **한쪽이 스칼라**(`x + 1.0`). 이때는 인덱스로 접근한다. 같은 `Op::f`인데 컴파일러가 NEON 레지스터로 묶는다.
- 효과: 대부분의 원소별 연산이 1.4–3배 빨라져 numpy와 같아졌다. bool `&`는 레지스터 하나에 16개가 들어가 17배다.
- 이 fast path는 `-O3`와 짝이다. nanobind는 기본으로 `-Os`를 붙이는데, `-Os`로는 같은 코드가 약 2배 느리다. 그래서 `NOMINSIZE`로 끈다 (docs/design/array.md §1.3).
- 같은 dtype의 연속 복사는 `memcpy`다. OS가 칩에 맞춰 튜닝한 함수다.

## 4. 손으로 짠 NEON
컴파일러의 벡터화가 모자란 곳에만 둔다. 모든 NEON 커널은 스칼라 정의와 비트가 같아야 하고, 테스트가 세 레이아웃(연속, 스칼라 브로드캐스트, strided)을 모두 비트로 비교한다.
- **float64 `maximum`/`minimum`**: IEEE 754-2019의 maximum/minimum(NaN 전파, `-0 < +0`)이 ARM `FMAX`/`FMIN` 명령 하나다. 스칼라 정의는 분기 셋이라 자동 벡터화하면 오히려 느려진다(0.66배). 명령 하나로 3.9배다.
- **`clip`**: `FMAX` 다음 `FMIN`을 한 패스로. 1스레드 143µs(numpy 230µs).
- **complex 곱셈**: numpy 바이너리의 비트는 `re = fma(ar, br, -(ai*bi))`, `im = fma(ar, bi, ai*br)`다. ARM `FCMLA`를 rot 90 → rot 0 순서로 두 번 쓰면 정확히 이 순서가 된다. 누산기를 `-0.0`에서 시작해야 부호 있는 0까지 같다.

```python
import stardust as sd

print(sd.maximum(sd.asarray([-0.0, 1.0]), sd.asarray([0.0, float("nan")])).tolist())  # [0.0, nan]
print(sd.clip(sd.asarray([-1.0, 0.5, 2.0]), min=0.0, max=1.0).tolist())  # [0.0, 0.5, 1.0]
```

## 5. numpy와 비트를 맞추는 방법
"numpy와 같다"를 비트 수준에서 지키려면 계산 순서와 반올림 횟수까지 같아야 한다.
- **같은 식, 같은 순서**: numpy가 식을 적어 둔 연산은 그 식을 쓴다. complex 곱셈과 나눗셈(Smith 알고리즘), float `//`(`npy_divmod`: `fmod` 후 부호 보정), complex `pow`.
  - float `//`를 정의대로 `floor(a / b)`로 계산하면 `a / b`가 먼저 반올림되어 다르다. `1.0 // 0.1`이 numpy는 9이고, 정의대로면 10이다.
- **FMA를 컴파일러에 맡기지 않는다.** `-ffp-contract=off`가 없으면 clang이 `a + b*c`를 FMA 하나로 합쳐 반올림이 한 번 준다. FMA가 필요한 자리(numpy 바이너리가 FMA로 빌드된 곳)는 `std::fma`로 적는다.
- **`-fno-builtin-sin -fno-builtin-cos`**: 없으면 clang이 같은 인자의 `sin`과 `cos`를 `__sincos_stret` 하나로 합친다. numpy는 따로 부르므로 1 ULP 달라질 수 있다.
- `std::complex`의 연산자는 C99 Annex G의 inf/nan 복구를 더하는데 numpy에는 없다. 그래서 식을 직접 적는다. complex 초월함수는 C 라이브러리의 `cexp` 등을 부른다.
- 이 플래그들은 모듈 전체의 것이다. frame의 float 합과 `//`도 이 덕분에 numpy와 비트가 같다 (docs/design/frame.md §2.5).

```python
import numpy as np
import stardust as sd

print((sd.asarray([1.0]) // 0.1).tolist(), float(np.floor_divide(1.0, 0.1)))  # [9.0] 9.0
a, b = [1 + 2j, -0.5 + 3j], [3 - 1j, 2e-300 + 1j]
print((sd.asarray(a) * sd.asarray(b)).tolist() == (np.asarray(a) * np.asarray(b)).tolist())  # True
```

## 6. vForce: 정확도를 조금 내주고 빠르게
float64 초월함수는 함수마다 구현을 고른다 (docs/design/array.md §5.4).
- **Accelerate의 vForce를 쓰는 16개**: `exp log log2 log10 sin asin acos atan sinh cosh tanh asinh acosh atanh pow atan2`
- **libm을 쓰는 넷**: `expm1 log1p cos tan`. vForce가 IEEE 의미론을 어긴다: `cos(±0) ≠ 1`, `-0`의 부호를 잃음, 서브노멀 입력을 0으로 만듦.
- vForce를 써도 되는지 독립 실험으로 확인했다.

  | 항목 | 결과 |
  |---|---|
  | 위치 독립성 | 약 24만 회 검사에서 불일치 0. `exp(x)[i] == exp(x[i])`이고, 스레드 청크가 결과를 바꾸지 않는다 |
  | 스레드 안전성, 내부 스레드 | 14스레드 동시 호출에서 불일치 0. 내부 스레드를 쓰지 않아 우리 풀과 부딪히지 않는다 |
  | 특수값 (±0, ±inf, NaN) | 16개 함수 모두 부호까지 정확하다 |
  | 정확도 | numpy 대비 측정 최대 4 ULP (`atanh`) |

- **계약은 numpy와 최대 8 ULP다.** 테스트는 float를 순서가 보존되는 정수로 바꿔 ULP 거리를 잰다(`assert_ulp`). 특수값은 비트까지 같아야 한다.
- 속도는 libm 대비 대부분 2–10배다(`sin` 7.6배, `exp` 4.2배). 원소 100만 개, 10스레드에서 `sin` 92µs, numpy 3.83ms다.
- vForce는 연속 배열을 받는다. 비연속이나 스칼라 입력은 커널 안에서 1024원소 스택 버퍼에 모았다가 흩는다. 스칼라 하나도 같은 길(n = 1)로 간다. 그래야 위치 독립성이 성립한다.

```python
import numpy as np
import stardust as sd

x = np.random.default_rng(1).uniform(-3, 3, 10_000)
ulp = np.abs(np.asarray(sd.sin(sd.asarray(x))).view(np.int64) - np.sin(x).view(np.int64)).max()
print(ulp <= 8)  # True
print(sd.cos(sd.asarray(x)).tolist() == np.cos(x).tolist())  # True
print(sd.cos(sd.asarray([-0.0])).tolist(), sd.expm1(sd.asarray([-0.0])).tolist())  # [1.0] [-0.0]
```

- `sin`(vForce)은 몇 ULP 다를 수 있고, `cos`(libm)는 비트까지 같다. 무엇이 어느 쪽인지는 `array/kernels/vforce.hpp`의 표가 정한다.

## 7. 연산자와 Python
- 연산자는 ufunc의 얇은 래퍼다. 받지 않는 피연산자(리스트, numpy 배열, `None`)면 `NotImplemented`를 돌려 Python이 TypeError를 내게 한다.
  - `==`, `!=`는 그대로 두면 Python이 객체 동일성으로 대신해 조용히 False가 된다. 그래서 직접 TypeError를 낸다 (DESIGN.md §9.1의 13번).
- `__array_ufunc__ = None`: numpy가 `np_array + sd_array`를 보고 우리 배열을 버퍼 프로토콜로 바꿔 numpy 결과를 돌려주는 것을 막는다.
- **에러 메시지는 부르는 길을 따른다** (API.md §2.1). 연산자는 Python의 형식, 함수는 `이름(): `으로 시작한다.

```python
import stardust as sd

c = sd.asarray([1 + 1j])
try:
    c < c
except TypeError as e:
    print(e)  # unsupported operand dtypes for <: complex128 and complex128
try:
    sd.less(c, c)
except TypeError as e:
    print(e)  # less(): not supported for complex128 and complex128
try:
    sd.asarray([1, 2]) == [1, 2]
except TypeError as e:
    print(e)  # '==' not supported between instances of 'stardust._core.ndarray' and 'list'
```

## 8. frame의 원소별 연산
frame은 같은 값 규칙을 다른 엔진으로 계산한다 (10장).
- 진입점은 C++ `binary_op` 하나이고, 스칼라는 1행 Column이다. "스칼라냐 컬럼이냐"와 연산 종류를 **템플릿 인자**로 골라 행 루프에 분기가 없다. 행마다 `switch`로 고르던 비교가 1.90ms, 템플릿 인자로 0.71ms였다 (docs/design/frame.md §7.2).
- 결측(NA)은 값과 따로 있는 validity 비트맵이 들고, 결과의 비트맵은 두 입력의 AND다.

## 더 읽을 곳
- [docs/design/array.md §5.1](../design/array.md) 연속 fast path, [docs/design/array.md §5.2](../design/array.md) NEON, [docs/design/array.md §5.3](../design/array.md) numpy와 비트 맞추기, [docs/design/array.md §5.4](../design/array.md) vForce
- [docs/design/array.md §1.3](../design/array.md) 컴파일 옵션과 이유
- [docs/design/array.md §2.7](../design/array.md) numpy와의 비트 동일성 표
- [docs/api/array.md §1.6](../api/array.md) 수치 의미론, [docs/api/array.md §6.4](../api/array.md) 함수별 규칙
