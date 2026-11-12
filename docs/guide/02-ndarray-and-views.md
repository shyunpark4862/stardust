# 2장. 배열의 정체: Buffer, strides, 뷰

> 배열은 메모리 한 덩어리와 그것을 읽는 방법이다. 복사 없이 전치와 슬라이스를 하는 방법은? 그 한계는?

## 1. 메모리와 설명
`sd.ndarray`의 C++ 쪽은 `NDArray`(`array/ndarray.hpp`)다. 메모리를 들고 있지 않고, **메모리를 읽는 방법을 설명한다.**

```
buffer   shared_ptr<Buffer>    메모리를 소유하는 블록 (여러 배열이 함께 쓴다)
data     std::byte*            Buffer 안의 시작 위치
dtype    bool / int64 / float64 / complex128
shape    [32], strides [32]    축마다 길이와, 한 칸 갈 때 건너뛰는 바이트 수
flags    c_contiguous, f_contiguous, writeable
base     Python의 .base (메모리를 실제로 가진 객체)
```

- 원소 `(i0, i1, …)`의 주소는 `data + Σ i_d · strides[d]`다. strides는 **바이트** 단위다.
- (2, 3) float64 C 순서 배열이면 한 행을 건너뛸 때 24바이트(원소 3개), 한 열은 8바이트다.

```python
import stardust as sd

x = sd.arange(6.0).reshape((2, 3))
print(x.strides)  # (24, 8)
print(x.T.strides)  # (8, 24)
print(x[:, ::2].strides)  # (24, 16)
```

- `x.T`는 strides의 순서만 바꾼다. 메모리는 그대로다. 그래서 1GB 배열의 전치도 2×2의 전치와 비용이 같다.
- `x[:, ::2]`는 열 stride를 두 배로 한다. 역시 메모리를 건드리지 않는다.
- 차원 수 상한은 32다. shape과 strides를 객체 안의 고정 배열로 두어 배열을 만들 때 shape용 힙 할당이 없다 (docs/design/array.md §3.3).

## 2. Buffer: 세 영역이 함께 쓰는 메모리 블록
`base/buffer.hpp`의 `Buffer`는 참조 카운트로 수명을 관리하는 메모리 블록이다. 배열의 뷰, linalg의 결과, frame의 Column이 모두 이것을 쓴다.

| 성질 | 이유 |
|---|---|
| 시작을 128B(M4의 캐시 라인)로 맞춘다 | SIMD 루프의 첫 load가 캐시 줄 두 개에 걸치지 않는다 |
| `zeros`는 `calloc`을 쓴다 | 큰 할당은 OS가 0 페이지를 첫 접근 때 채운다. 쓰기 전까지 거의 공짜다 |
| 256B 이하의 데이터는 Buffer 객체 안에 둔다 | 작은 배열 하나의 할당이 세 번(Buffer, 데이터, `shared_ptr` 제어 블록)에서 한 번이 된다 |
| 외부 메모리를 `adopt`한다 | 버퍼 프로토콜로 가져온 numpy 배열을 복사 없이 쓴다 (§6) |

- 옛 numpy판과 pandas판의 Buffer는 두 벌이었다. 합치면서 한 벌이 되었고, frame의 작은 Column도 할당 한 번으로 만들어지게 되었다 (docs/design/frame.md §4.1).
- frame의 Copy-on-Write는 이 Buffer의 참조 카운트를 본다 (10장).

## 3. 뷰와 `.base`
뷰는 같은 Buffer를 다른 설명으로 읽는 배열이다.

| 뷰를 만드는 것 | 복사하는 것 |
|---|---|
| 기본 인덱싱(int, slice, `...`, `newaxis`), strides로 표현되는 `reshape`, `permute_dims`, `.T`, `.mT`, `broadcast_to`, complex의 `real`/`imag` | strides로 표현할 수 없는 `reshape`, `concat`, `stack`, 고급 인덱싱(정수 배열, bool 마스크) |

```python
import stardust as sd

a = sd.arange(6.0)
m = a.reshape((2, 3))
print(m.base is a)  # True
print(m.T.base is a)  # True
m[0, 0] = 100.0
print(a[0])  # 100.0
```

- `.base`는 메모리를 실제로 가진 객체다. 뷰의 뷰를 만들어도 중간 뷰를 거치지 않고 **최종 소유자**를 가리킨다(`m.T.base`는 `m`이 아니라 `a`). numpy 2도 같다.
- 뷰에 쓰면 원본이 바뀐다. 같은 메모리이기 때문이다.

strides로 표현할 수 없으면 `reshape`는 복사한다. 전치한 배열을 1차원으로 펴면 원소 순서가 메모리 순서와 다르다.

```python
import stardust as sd

m = sd.arange(6.0).reshape((2, 3))
flat = m.T.reshape((6,))
print(flat.base is None)  # True
print(flat.tolist())  # [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
try:
    m.T.reshape((6,), copy=False)
except ValueError as e:
    print(e)  # reshape(): copy=False, but this reshape cannot be done without copying
```

- `copy` 인자는 세 값이다: `None`(필요할 때만 복사), `True`(항상), `False`(복사가 필요하면 에러). `asarray`, `astype`, `reshape`가 받는다. frame의 `to_numpy`, `to_array`도 같은 뜻이다 (docs/api/array.md §1.8).

**쓰기 금지 뷰**: `broadcast_to`는 원소 여러 개가 같은 메모리를 가리킨다(stride 0). 쓰면 위험하므로 `writeable=False`다.

```python
import stardust as sd

b = sd.broadcast_to(sd.asarray([1.0, 2.0]), (3, 2))
print(b.strides)  # (0, 8)
try:
    b[0, 0] = 5.0
except ValueError as e:
    print(e)  # assignment destination is read-only
```

## 4. 연속성
- 플래그 둘이 있다: C 연속(마지막 축이 가장 빠름), F 연속(첫 축이 가장 빠름). shape과 strides가 바뀔 때마다 다시 계산한다.
- 크기 1인 축의 stride는 보지 않는다. 빈 배열은 양쪽 다 연속이다 (numpy의 relaxed strides).
- **새 배열의 순서**: 대부분 C 순서다. 예외가 둘이다 (docs/api/array.md §1.9).
  - 원소별 연산의 결과는 첫 배열 피연산자의 메모리 순서를 따른다. F 순서 입력이면 F 순서 결과가 나온다. 그래야 다음 연산도 메모리를 순서대로 읽는다 (3장).
  - linalg의 행렬 결과는 F 순서다. LAPACK이 만든 레이아웃 그대로다 (8장).

```python
import stardust as sd

m = sd.zeros((2, 3))
print(m._flags()["c_contiguous"], m.T._flags()["f_contiguous"])  # True True
print((m.T + 1.0)._flags()["f_contiguous"])  # True
print(sd.linalg.inv(sd.eye(2))._flags()["f_contiguous"])  # True
```

## 5. 0차원 배열은 없다
- 결과가 0차원이 되면 **Python 스칼라**를 돌려준다: `x[1, 2]`, `sd.sum(x)`, 벡터 `@` 벡터, `sd.linalg.det(2×2)`.
- 0차원을 명시적으로 요청하면 ValueError다: `asarray(5)`, `zeros(())`.
- 배열은 크기가 1이어도 스칼라가 아니다. `bool(x)`, `float(x)`는 TypeError다.
- 이유: numpy의 0차원 배열과 스칼라 사이의 미묘한 차이(읽기 전용인지, 연산 결과가 무엇인지)를 통째로 없앤다. int 계산은 int64 규칙(wrap)으로 한 뒤 Python int가 된다.

```python
import stardust as sd

x = sd.arange(6.0).reshape((2, 3))
print(type(x[1, 2]).__name__, type(sd.sum(x)).__name__)  # float float
try:
    sd.asarray(5)
except ValueError as e:
    print(type(e).__name__)  # ValueError
try:
    bool(sd.asarray([1]))
except TypeError as e:
    print(e)  # the truth value of an array is ambiguous; use any() or all()
```

## 6. 메모리 겹침
`a[1:] += a[:-1]`은 입력과 출력이 메모리를 일부 나눈다. 앞에서부터 쓰면 뒤에서 읽을 값이 이미 바뀌어 있다.
- **보장**: 결과는 "모든 입력을 다 읽은 다음 출력을 쓴 것"과 같다 (numpy와 같다).
- 판정은 보수적이다. 두 배열이 닿는 바이트 범위가 겹치면 겹치는 입력을 먼저 임시 배열로 복사한다.
- 예외: 입력과 출력이 완전히 같은 뷰(`a += a`)면 복사하지 않는다. 원소마다 읽은 직후 같은 자리에 쓰므로 안전하다.
- 인덱스 대입에서는 인덱스 배열과 마스크도 검사한다. `a[a] = 2`에서 앞의 쓰기가 뒤에서 읽을 인덱스를 바꾸면, 범위 검사를 통과한 인덱스가 범위 밖이 될 수 있다 (docs/design/array.md §3.5).

```python
import stardust as sd

a = sd.arange(5.0)
a[1:] += a[:-1]
print(a.tolist())  # [0.0, 1.0, 3.0, 5.0, 7.0]
```

## 7. numpy와 복사 없이 주고받기
`sd.ndarray`는 Python 버퍼 프로토콜을 양방향으로 쓴다 (docs/design/array.md §12.3).
- **내보내기**: `np.asarray(x)`는 복사하지 않는다. 테스트도 이 길로 결과를 numpy에 가져와 비교한다.
- **가져오기**: format이 저장 타입과 같고 정렬되어 있으면 복사하지 않는다(`Buffer::adopt`). 읽기 전용 버퍼는 쓰기 금지 배열이 된다.
  - 손실 없는 확장(int32 → int64, float32 → float64 등)은 복사하면서 변환한다. `copy=False`면 에러다.
  - 손실이 있을 수 있는 dtype(uint64 → int64)은 값과 상관없이 에러다. 데이터프레임도 같다.
  - big-endian 배열(`>f8` 등)은 바이트를 뒤집으며 복사한다. 값은 같다.

```python
import numpy as np
import stardust as sd

n = np.zeros(3)
s = sd.asarray(n)
n[0] = 5.0
print(s.tolist(), s.base is n)  # [5.0, 0.0, 0.0] True
print(sd.asarray(np.arange(3, dtype=np.int32)).dtype)  # int64
try:
    sd.asarray(np.arange(3, dtype=np.uint64))
except TypeError as e:
    print(e)  # asarray(): uint64 buffers cannot be converted to int64 without loss
```

- frame은 반대로 numpy에서 들어올 때 **언제나 복사한다.** 밖에서 바뀔 수 있는 메모리를 Copy-on-Write가 추적할 수 없기 때문이다 (10장, 12장).

## 8. 작은 배열의 비용
작은 배열에서는 계산이 아니라 객체를 만들고 옮기는 비용이 시간을 정한다. 프로파일러(`sample`)로 본 비용과 고친 것이다 (docs/design/array.md §11).
- 32차원 고정 배열을 통째로 복사하던 `memmove`: 실제 `ndim`칸만 옮긴다.
- 배열 하나에 `malloc` 세 번: Buffer를 `make_shared`로, 작은 데이터는 Buffer 안에 (§2).
- `x[1, 2]`처럼 앞쪽 축이 Python int인 키는 키 해석을 건너뛴다. 25ns(numpy 33ns)다.

## 더 읽을 곳
- [docs/design/array.md §3.2](../design/array.md) Buffer, [docs/design/array.md §3.3](../design/array.md) NDArray, [docs/design/array.md §3.4](../design/array.md) 뷰
- [docs/design/array.md §2.2](../design/array.md) 0차원 배열 없음
- [docs/design/array.md §12.3](../design/array.md) 버퍼 프로토콜
- [docs/api/array.md §1.8](../api/array.md) `copy` 인자, [docs/api/array.md §1.9](../api/array.md) 새 배열의 메모리 순서
