# 배열 API (sd)

`import stardust as sd`의 최상위 이름공간이 배열 API다. 이 문서는 배열의 공개 API와 동작 규칙을 모두 적는다.

- 라이브러리 전체의 공통 규칙(dtype, 에러 종류, 스레드)은 루트 [API.md](../../API.md)에 있다. 이 문서는 배열 영역의 규칙이다.
- 절 번호(`§N`, `§N.M`)는 코드 주석이 인용한다. 번호를 바꾸지 않는다.
- 언어 표시가 없는 코드 블록은 `tests/array/test_api_surface.py`가 읽는다 (API.md의 블록과 함께).
  - 이 블록에는 공개 이름과 시그니처만 쓴다.
  - 블록에 나온 이름과 `sd`가 공개하는 이름은 정확히 같아야 한다. 예시는 본문의 인라인 코드로 쓴다.

---

## §1. 배열 공통 규칙

### §1.1 시그니처와 이름
- **positional-only (`/` 앞)**
  - 배열 인자
  - 함수의 본 인자. 예: `reshape`의 `shape`, `full`의 `fill_value`, `permute_dims`의 `axes`
- **keyword-only (`*` 뒤)**: 이름이 붙은 옵션은 모두 keyword-only다.
  - 옵션 이름: `dtype`, `axis`, `keepdims`, `out`, `copy`, `stable`, `descending`, `correction`, `min`, `max`, `k`, `endpoint`, `include_initial`
- **bool 옵션**(`keepdims`, `stable`, `descending`, `endpoint`, `include_initial`)은 `True`/`False`만 받는다. `0`, `1`은 `TypeError`다.
- **이름**
  - Python Array API Standard의 이름을 쓴다. 예: `acos`, `concat`, `pow`, `permute_dims`
  - Array API에 없는 함수만 numpy 이름을 쓴다 (`ravel`).
  - 함수 하나에 이름 하나다. 별칭은 없다.
- **모듈 함수가 정식 API다.** 메서드는 정해진 목록만 있다 (§10.2).
- 배열 인자에는 `sd.ndarray`만 받는다. 리스트나 numpy 배열은 `TypeError`다. 필요하면 먼저 `asarray`로 변환한다.
  - 예외: 원소별 함수는 Python 스칼라도 받는다 (§6.2). `asarray`는 변환 대상을 받는다 (§3.2).

### §1.2 0차원 배열은 없다
- **결과가 암묵적으로 0차원이 되면 Python 스칼라를 반환한다.**
  - 예: `x[1, 2]`, `sum(x)`, 1차원 배열의 `sum(x, axis=0)`, 피연산자가 모두 Python 스칼라인 원소별 함수, 1차원끼리의 `matmul`
  - 스칼라의 타입은 dtype을 따른다: bool → `bool`, int64 → `int`, float64 → `float`, complex128 → `complex`
  - `keepdims=True`이면 배열로 남는다.
- **0차원을 명시적으로 요청하면 `ValueError`다.**
  - 예: `asarray(5)`, `asarray(numpy 0차원 배열)`, `zeros(())`, `reshape(x, ())`, `broadcast_to(x, ())`, 모든 축을 없애는 `squeeze`
- Python 스칼라로 꺼낸 다음에는 Python 의미론을 따른다. 오버플로가 없다.

### §1.3 shape과 axis 인자
- **shape**은 인자 하나로 받는다: int 또는 int의 tuple
  - 리스트, float, bool은 `TypeError`다.
  - int64 범위 밖의 int는 `OverflowError`다 (API.md §2).
  - 음수 길이는 `ValueError`다. 길이 0은 허용한다.
  - 차원 수는 1 이상 32 이하다. 32를 넘으면 `ValueError`다.
  - 전체 크기가 너무 크면 `ValueError`다.
- **axis**도 인자 하나로 받는다: int 또는 int의 tuple
  - 음수는 뒤에서 센다.
  - 범위 밖이면 `ValueError`, 중복되면 `ValueError`다. int64 범위 밖의 int는 `OverflowError`다.
  - 리스트는 `TypeError`다. linalg의 `vector_norm`은 리스트도 받는다 (linalg.md §3).
- 그 밖의 int 인자(`eye`의 `n_rows`, `n_cols`, `k`, `linspace`의 `num`)도 int64 범위 밖이면 `OverflowError`다.
  - int만 받는 함수: `argmin`, `argmax`, `cumulative_sum`, `cumulative_prod`, `sort`, `argsort`, `concat`, `stack`
- **기본 axis**
  - 리덕션: `axis=None` (모든 축)
  - `sort`, `argsort`: `axis=-1`
  - `cumulative_sum`, `cumulative_prod`: 1차원이면 생략할 수 있다. 2차원 이상이면 필수다.

### §1.4 dtype 승격
- 승격 순서는 일직선이다: `bool < int64 < float64 < complex128`
- 이항 연산의 결과 dtype은 두 피연산자 중 높은 쪽이다.
- Python 스칼라도 같은 규칙을 따른다: `bool` → bool, `int` → int64, `float` → float64, `complex` → complex128
  - 예: bool 배열 `+ 1`은 int64, int64 배열 `+ 1.5`는 float64다.
  - **numpy 스칼라는 같은 종류의 Python 스칼라로 본다**: `numpy.bool` → `bool`, numpy 정수(`int8`…`uint64`) → `int`, numpy 실수(`float16`…`float64`) → `float`, numpy 복소수 → `complex`. 이 문서에서 "Python 스칼라"는 이것까지 뜻한다.
    - 예: `x + np.int64(1)`, `asarray([np.float32(0.5)])`, `x[0] = np.bool_(True)`
    - numpy 정수도 int64 범위 규칙(§1.5)을 따른다. 예: int64 배열 `+ np.uint64(2**64 - 1)`은 `OverflowError`다.
    - `numpy.datetime64`, `numpy.timedelta64`, `numpy.str_`는 수가 아니다. `TypeError`다.
- **산술에서 bool은 int64로 승격한다.**
  - 해당: `add`, `subtract`, `multiply`, `floor_divide`, `remainder`, `pow`, `negative`, `positive`, `abs`, `square`, `sign`, `matmul`, `sum`, `prod`, `cumulative_sum`, `cumulative_prod`
  - 예: `True + True`는 `2`, bool 배열끼리의 `@`는 int64다.
  - 논리, 비트, 비교, `maximum`/`minimum`, 반올림 함수는 bool을 그대로 둔다.
- float 결과만 있는 함수(`divide`, `sqrt`, `exp`, 삼각함수 등)는 bool과 int64 입력을 float64로 계산한다.
- 함수별 결과 dtype은 §6.3, §7.3에 있다.

### §1.5 변환 규칙
**암묵적 변환**
- 해당: `out=`, in-place 연산자, 인덱스 대입, `clip`의 `min`/`max`, `dtype=` 없는 `full_like`의 `fill_value`
- 결과를 대상 dtype으로 바꿀 때 상향 변환만 허용한다. 하향 변환은 `TypeError`다.
  - 판단은 dtype으로만 한다. 값은 보지 않는다. 예: int64 배열에 `2.0`을 대입해도 `TypeError`다.
  - bool 대상에 int64를 넣는 것도 하향 변환이다.
  - 예: `full_like(int64 배열, 1.5)`는 `x[...] = 1.5`와 같이 `TypeError`다.

**명시적 변환**
- 해당: `astype`, 생성 함수의 `dtype=`, `sum`/`prod`/`cumulative_*`의 `dtype=`. `full`, `full_like`에 `dtype=`을 주면 `fill_value`를 그 dtype으로 바꾸는 일도 여기에 든다.
- 손실이 있어도 변환한다.
  - 예: `asarray([1.5], dtype=sd.int64)`는 `[1]`, `full(3, 1.5, dtype=sd.int64)`와 `full_like(int64 배열, 1.5, dtype=sd.int64)`는 1로 채운다.
- **float → int64**: 0 방향으로 자른다. 범위를 벗어나면 포화시킨다. NaN은 0이 된다.
- **complex → float64, complex → int64**: `TypeError`다. 허수부를 잃기 때문이다. 실수부가 필요하면 `real(x)`를 쓴다.
- **X → bool**: 모든 dtype에서 허용한다. 결과는 `x != 0`이다. complex도 허용하고, NaN은 `True`가 된다.

**int64 범위를 벗어난 Python int**
- **int64로 바뀌어야 할 때만** `OverflowError`다.
  - 원소별 함수에서는 **승격한 피연산자 dtype**(§1.4)이 기준이다. 그것이 bool이나 int64이면 int는 int64가 되어야 한다. 함수가 float64로 계산해도 그렇다. 모든 원소별 함수가 이 규칙 하나를 따른다.
  - 피연산자에 float64나 complex128이 있으면 float로 바꿔서 정상 동작한다.
  - 예: int64 배열 `+ 2**70`은 에러, float64 배열 `+ 2**70`은 정상이다.
  - 예: `divide(int64 배열, 2**70)`, `atan2(int64 배열, 2**70)`, `sqrt(2**70)`도 에러다. `divide(float64 배열, 2**70)`은 정상이다.
  - 그 밖의 곳은 변환 대상 dtype이 기준이다. 예: `asarray([2**70, 0.5])`는 정상, `asarray([2**70])`는 에러다. `full_like(float64 배열, 2**70)`은 정상이다.
- 예외: `arange`의 int 인자는 `dtype=`과 상관없이 int64 범위여야 한다.

### §1.6 수치 의미론
- **원소 연산은 값 때문에 실패하지 않는다.** 각 dtype은 자기 dtype의 표준 산술을 따른다.
  - 에러는 타입이나 shape이 맞지 않을 때만 낸다.
- **정수 오버플로**: 조용히 wrap한다. 예: `INT_MIN // -1 == INT_MIN`, `abs(INT_MIN) == INT_MIN`
- **정수를 0으로 나누기**: `a // 0 == 0`, `a % 0 == a`
  - 그래서 `a == (a // b) * b + a % b`가 `b == 0`일 때도 성립한다.
- **`//`와 `%`**: `//`는 내림 나눗셈이다. `%`의 부호는 나누는 수를 따른다 (Python과 같다).
- **정수 거듭제곱의 음수 지수**: 정수 결과를 낸다.
  - `1 ** -n == 1`, `(-1) ** -n == ±1`, 그 외에는 `0`이다.
  - `0 ** -n`은 0으로 나누기 규칙을 따라 `0`이다.
- **float, complex**: IEEE 754를 따른다 (`inf`, `nan`). 예: `1 / 0`은 `inf`, `sqrt(-1.0)`은 `nan`이다.
- **NaN 비교**: `nan == nan`은 `False`, `nan != nan`은 `True`다.
- **int와 float의 비교**: int를 float64로 바꿔서 비교한다. 예: `equal(2**53 + 1, 2.0**53)`은 `True`다.
- **complex에는 순서가 없다.** 순서가 필요한 연산은 `TypeError`다.
  - 해당: `<` `<=` `>` `>=`, `maximum`, `minimum`, `clip`, `min`, `max`, `argmin`, `argmax`, `sort`, `argsort`, `floor`, `ceil`, `trunc`, `floor_divide`, `remainder`, `atan2`
  - `==`, `!=`, `unique_*`는 complex를 받는다.
- 경고는 일절 없다. `errstate`도 없다.

### §1.7 메모리 겹침
- 해당: 원소별 함수의 `out=`, in-place 연산자, 인덱스 대입
- 결과는 **모든 입력을 다 읽은 다음에 출력을 쓴 것**과 같다.
  - 예: `x[1:] = x[:-1]`, `x[1:] += x[:-1]`, `add(x, x[::-1], out=x)`

### §1.8 `copy` 인자
- `copy=True`: 항상 복사한다.
- `copy=False`: 복사가 필요하면 `ValueError`다.
- `copy=None`: 필요할 때만 복사한다.
- `True`, `False`, `None` 외의 값은 `TypeError`다.

### §1.9 새 배열의 메모리 순서
- **결과의 메모리 순서는 영역마다 정해서 적는다.** 값은 순서와 무관하고 strides만 다르다. 아래는 배열 영역(`sd`)의 규칙이다.
  - linalg의 행렬 결과는 F 순서다 (linalg.md §1.7).
  - frame의 `DataFrame.to_array`는 F 순서다 (frame.md §2.8).
- **입력의 메모리 순서를 따르는 연산**: 원소별 함수와 연산자, `where`, `clip`, 축 리덕션, `copy`, `astype`, `asarray(copy=True)`, `*_like`, `sort`, `argsort`, 고급 인덱싱 (§5.2)
  - 예: F 순서 입력이면 결과도 F 순서다.
- **그 밖에 배열 영역이 새로 만드는 배열은 C 순서다.** 예: 생성 함수, 복사가 필요한 `reshape`/`ravel`, `concat`, `stack`, `cumulative_*`, `matmul`, `unique_*`

---

## §2. 상수
```
e  pi  inf  nan
newaxis
ndarray
```
- `e`, `pi`, `inf`, `nan`은 Python `float`다.
- `newaxis`는 `None`이다.
- `ndarray`는 배열 타입이다. `isinstance` 검사에 쓴다. 생성자는 없다 (§10).
- dtype 객체는 API.md §1에 있다.

---

## §3. 배열 생성

### §3.1 함수
```
asarray(obj, /, *, dtype=None, copy=None)
zeros(shape, /, *, dtype=None)
ones(shape, /, *, dtype=None)
empty(shape, /, *, dtype=None)
full(shape, fill_value, /, *, dtype=None)
zeros_like(x, /, *, dtype=None)
ones_like(x, /, *, dtype=None)
empty_like(x, /, *, dtype=None)
full_like(x, fill_value, /, *, dtype=None)
arange(start, stop=None, step=1, /, *, dtype=None)
linspace(start, stop, num, /, *, dtype=None, endpoint=True)
eye(n_rows, n_cols=None, /, *, k=0, dtype=None)
```
- **`dtype=None`일 때의 dtype**
  - `zeros`, `ones`, `empty`, `eye`: float64
  - `full`: `fill_value`의 타입을 따른다 (§1.4의 Python 스칼라 대응)
  - `*_like`: `x`의 dtype
  - `arange`: 인자가 모두 int(또는 bool)이면 int64, 하나라도 float이면 float64
  - `linspace`: `start`나 `stop`이 complex이면 complex128, 아니면 float64
  - `asarray`: §3.2
- `dtype=`을 주면 명시적 변환이다 (§1.5).
- `array` 함수는 없다. 변환은 `asarray`, 복사는 `asarray(x, copy=True)`로 한다.
- `fill_value`는 Python `bool`, `int`, `float`, `complex`만 받는다 (numpy 스칼라 포함, §1.4). 배열은 `TypeError`다.
- `full_like`에 `dtype=`이 없으면 `fill_value`는 `x`의 dtype으로 **암묵적 변환**된다 (§1.5). 예: `full_like(int64 배열, 1.5)`는 `TypeError`다. `dtype=`을 주면 명시적 변환이다.
- `*_like`의 `x`는 `sd.ndarray`여야 한다. 결과는 `x`의 메모리 순서를 따른다 (§1.9).
- `empty`, `empty_like`의 값은 정해져 있지 않다.

### §3.2 `asarray`
- **받는 입력**
  - `sd.ndarray`
  - 중첩 시퀀스: `list`와 `tuple`만 받는다. 원소는 Python `bool`, `int`, `float`, `complex`(numpy 스칼라 포함, §1.4) 또는 `sd.ndarray`다.
    - `range`, 문자열, `None`은 `TypeError`다.
  - 버퍼 프로토콜 객체: 예: numpy 배열, `bytes`, `memoryview` (§3.3)
  - Python 스칼라는 0차원이 되므로 `ValueError`다 (§1.2).
- **중첩 시퀀스의 dtype 추론**
  - 원소 중 가장 높은 dtype을 따른다 (§1.4). 예: `[True, 1]`은 int64, `[1, 2.5]`는 float64다.
  - 들쭉날쭉한(ragged) 시퀀스는 `ValueError`다.
  - `[]`는 shape `(0,)`, `[[]]`는 shape `(1, 0)`이다. 원소가 없으면 dtype은 float64다.
- **복사 여부** (`copy=None`)
  - `sd.ndarray`이고 dtype 변환이 없으면 **`x` 자신**을 반환한다.
  - 시퀀스는 언제나 복사한다. 그래서 `copy=False`이면 `ValueError`다.
  - 버퍼는 §3.3을 따른다.

### §3.3 버퍼 프로토콜로 가져오기
- **format이 정확히 맞으면 복사하지 않는다.** `?` → bool, `q` → int64, `d` → float64, `Zd` → complex128
  - 결과는 외부 메모리를 공유한다. `base`는 내보낸 객체다.
  - 임의의 strides(음수 포함)를 그대로 쓴다.
  - 읽기 전용 버퍼는 쓰기 금지 배열이 된다.
  - 예외: `?` 버퍼에 0과 1이 아닌 바이트가 있으면 0/1로 고쳐서 복사한다. 이때 `copy=False`이면 `ValueError`다.
- **정보 손실이 없는 확장이면 복사하면서 바꾼다.**
  - `int8`/`int16`/`int32`, `uint8`/`uint16`/`uint32` → int64
  - `float16`/`float32` → float64
  - `complex64` → complex128
  - `copy=False`이면 `ValueError`다.
- **그 외 format은 `TypeError`다.** 예: `uint64`(int64로 바꾸면 손실), 문자열, big-endian 바이트 순서
- 0차원 버퍼는 `ValueError`, 32차원을 넘는 버퍼도 `ValueError`다.

### §3.4 `arange`, `linspace`, `eye`
- **`arange(start, stop=None, step=1, /, *, dtype=None)`**
  - 인자 셋 모두 positional-only다.
  - 인자가 하나면 그 값이 `stop`이고 `start`는 0이다. 예: `arange(5)`
  - 인자는 실수(Python `bool`, `int`, `float`)만 받는다. complex는 `TypeError`다.
  - 길이는 `ceil((stop - start) / step)`이고, 음수이면 0이다.
  - `step == 0`은 `ValueError`다.
  - float 값은 `start + i*step`으로 계산한다. numpy와 몇 ULP 다를 수 있다.
  - `dtype=`을 주면 위의 값을 그 dtype으로 명시적 변환한다. 예: `arange(0, 3, 0.5, dtype=sd.int64)`는 `[0, 0, 1, 1, 2, 2]`다.
  - 인자에 NaN이나 inf가 있으면 `ValueError`다. 길이를 정할 수 없기 때문이다.
- **`linspace(start, stop, num, /, *, dtype=None, endpoint=True)`**
  - `num`은 필수다. int만 받는다(bool 제외). 음수는 `ValueError`다.
  - `endpoint=True`이면 `stop`을 포함한다.
  - `start`, `stop`은 complex도 받는다.
  - `dtype=`을 주면 계산한 값을 그 dtype으로 명시적 변환한다.
- **`eye(n_rows, n_cols=None, /, *, k=0, dtype=None)`**
  - `n_cols`를 생략하면 `n_rows`와 같다.
  - `k`는 대각선 오프셋이다. 양수는 위쪽, 음수는 아래쪽이다. 범위를 벗어나면 0 행렬이다.
  - `n_rows`, `n_cols`는 0 이상의 int다.

---

## §4. shape 조작과 뷰

### §4.1 함수
```
reshape(x, shape, /, *, copy=None)
ravel(x, /)
permute_dims(x, axes, /)
matrix_transpose(x, /)
moveaxis(x, source, destination, /)
squeeze(x, /, *, axis)
expand_dims(x, /, *, axis)
broadcast_to(x, shape, /)
broadcast_arrays(*arrays)
concat(arrays, /, *, axis=0)
stack(arrays, /, *, axis=0)
astype(x, dtype, /, *, copy=True)
```
- **`reshape`**: 가능하면 뷰를 반환한다. 아니면 복사한다 (§1.8).
  - `shape`에 `-1`을 하나 쓸 수 있다. 둘 이상이면 `ValueError`다.
  - 크기가 맞지 않으면 `ValueError`다.
- **`ravel`**: 1차원으로 편다. 가능하면 뷰를 반환한다. `flatten`은 없다.
- **`permute_dims`**: `axes`는 모든 축의 순열이다. int 또는 tuple이고, 리스트는 `TypeError`다.
- **`matrix_transpose`**: 마지막 두 축을 바꾼다. `.mT`와 같다. 2차원 미만은 `ValueError`다.
- **`moveaxis`**: `source`, `destination`은 int 또는 같은 길이의 tuple이다.
- **`squeeze`**: `axis`는 필수다. 길이가 1이 아닌 축은 `ValueError`다. 모든 축을 없애면 `ValueError`다 (§1.2).
- **`expand_dims`**: `axis`는 필수다. int 또는 tuple이다. 축 번호는 결과 배열 기준이다.
- **`broadcast_to`, `broadcast_arrays`**
  - 결과는 쓰기 금지 뷰다. 브로드캐스트된 축의 stride는 0이다.
  - `broadcast_arrays`는 tuple을 반환한다. 인자는 모두 `sd.ndarray`여야 한다. 인자가 없으면 `()`다.
- 위 함수들의 뷰는 원본과 메모리를 공유한다. `base`는 메모리를 실제로 소유한 배열이다 (§10.1).

### §4.2 `concat`, `stack`
- `arrays`는 `sd.ndarray`의 `list` 또는 `tuple`만 받는다.
  - 다른 객체가 섞이면 `TypeError`다. 필요하면 먼저 `asarray`로 변환한다.
  - 빈 목록은 `ValueError`다.
- 결과 dtype은 승격 규칙을 따른다 (§1.4).
- `concat`: `axis` 밖의 shape이 모두 같아야 한다. `axis=None`이면 각 배열을 평탄화해서 잇는다.
- `stack`: 모든 shape이 같아야 한다. `axis`는 결과 배열 기준이다. `axis=None`은 받지 않는다.

### §4.3 `astype`
- `dtype`으로 명시적 변환을 한다 (§1.5).
- `copy=True`(기본값)이면 같은 dtype이어도 항상 새 배열이다.
- `copy=False`이면 같은 dtype일 때 `x` 자신을 반환하고, 변환이 필요하면 `ValueError`다.
- `copy=None`이면 같은 dtype일 때 `x` 자신을 반환한다.

---

## §5. 인덱싱

### §5.1 기본 인덱스
- `int`, `slice`, `...`, `newaxis`를 쓴다. **결과는 뷰다.**
- 모든 축을 int로 고르면 Python 스칼라다 (§1.2).
- 음수 인덱스는 뒤에서 센다. 범위 밖 int는 `IndexError`다.
- `slice`의 step이 0이면 `ValueError`다.
- `...`은 하나만 쓸 수 있다. 축보다 인덱스가 많으면 `IndexError`다.
- `x[(0, 1)]`은 `x[0, 1]`과 같다. `x[[0, 1]]`은 고급 인덱스다.
- bool 스칼라(`x[True]`, `x[False]`)는 `IndexError`다. 0차원 마스크가 되기 때문이다.
- float, 문자열 등 다른 객체는 `IndexError`다.

### §5.2 고급 인덱스
- 정수 배열과 bool 마스크를 쓴다. **결과는 복사본이다.**
  - `sd.ndarray`, 중첩 `list`/`tuple`, 버퍼 객체(numpy 배열 등)를 받는다. 맨 바깥 `tuple`은 축별 인덱스다 (§5.1). 예: `x[(0, 1),]`은 고급 인덱스다. `sd.ndarray`가 아니면 `asarray`와 같은 규칙으로 변환한다 (§3.2, §3.3).
    - 변환할 수 없는 버퍼(예: `uint64`)는 `TypeError`다.
  - 변환한 dtype이 int64나 bool이 아니면 `IndexError`다. 예: `x[[1.0]]`
  - 빈 리스트 `[]`는 위치가 하나도 없는 정수 인덱스다.
- **numpy의 의미론을 따른다.** 인덱스 배열끼리 브로드캐스트하고, 마스크는 참인 위치들의 정수 인덱스와 같다.
  - 정수 배열 안의 음수도 뒤에서 센다. 범위 밖이면 `IndexError`다.
  - 마스크의 shape이 인덱싱하는 축과 맞지 않으면 `IndexError`다.
- **떨어진 고급 인덱스는 `IndexError`다.**
  - 고급 인덱스들 사이에 slice, `newaxis`, `...`이 끼어 있는 경우다. numpy는 이때 결과 축을 맨 앞으로 옮기는데, 그 규칙을 없애기 위해서다.
  - 정수 스칼라도 배열 인덱스와 함께 쓰이면 고급 인덱스로 친다. 그래서 `x[0, :, [0, 1]]`, `x[[0, 1], :, 0]`도 `IndexError`다.
  - 두 단계로 나눠 쓴다. 예: `x[0][:, [0, 1]]`
  - 붙어 있는 경우(`x[:, [0, 1], [0, 1]]`, `x[..., [0, 1]]`)는 결과가 numpy와 같다.
- **결과의 메모리 순서는 입력을 따른다** (§1.9).
  - 남는 축은 입력에서의 메모리 순서를 유지한다. 인덱스 축은 인덱싱된 축의 자리에 들어간다.
  - 예: F 순서 `m`의 `m[:, rows]`는 F 순서, C 순서 `m`이면 C 순서다.

### §5.3 대입 (`x[idx] = value`)
- `value`로 받는 것
  - Python `bool`, `int`, `float`, `complex`
  - `sd.ndarray`
  - `asarray`가 받는 시퀀스와 버퍼. `asarray`로 변환한 뒤 대입한다.
- **원소 하나를 고르는 인덱스에는 Python 스칼라만 대입할 수 있다.** 배열이나 리스트는 `ValueError`다.
- **브로드캐스팅은 엄격하다.** `value`는 선택된 영역의 shape으로 `broadcast_to`될 수 있어야 한다.
  - numpy처럼 `value`의 앞쪽 길이-1 축을 떼어내지 않는다. 예: shape `(3,)` 자리에 `(1, 3)`은 `ValueError`다.
- 하향 변환은 `TypeError`다 (§1.5). int64 범위 밖 Python int를 int64 배열에 넣으면 `OverflowError`다.
- 고급 인덱스에 같은 위치가 여러 번 나오면 마지막 값이 남는다.
- 고급 인덱스 대입은 인덱스를 모두 검사한 뒤에 쓴다. 범위 밖 인덱스가 있으면 `IndexError`가 나고 `x`는 바뀌지 않는다.
- 쓰기 금지 배열에 대입하면 `ValueError`다.
- `value`와 `x`의 메모리가 겹쳐도 된다 (§1.7).
- `del x[idx]`는 지원하지 않는다.

---

## §6. 원소별 함수

### §6.1 함수
```
# 산술
add(x1, x2, /, *, out=None)
subtract(x1, x2, /, *, out=None)
multiply(x1, x2, /, *, out=None)
divide(x1, x2, /, *, out=None)
floor_divide(x1, x2, /, *, out=None)
remainder(x1, x2, /, *, out=None)
pow(x1, x2, /, *, out=None)
negative(x, /, *, out=None)
positive(x, /, *, out=None)
abs(x, /, *, out=None)
square(x, /, *, out=None)
sqrt(x, /, *, out=None)
sign(x, /, *, out=None)
# 지수와 로그
exp(x, /, *, out=None)
expm1(x, /, *, out=None)
log(x, /, *, out=None)
log1p(x, /, *, out=None)
log2(x, /, *, out=None)
log10(x, /, *, out=None)
# 삼각함수와 쌍곡선함수
sin(x, /, *, out=None)
cos(x, /, *, out=None)
tan(x, /, *, out=None)
asin(x, /, *, out=None)
acos(x, /, *, out=None)
atan(x, /, *, out=None)
atan2(x1, x2, /, *, out=None)
sinh(x, /, *, out=None)
cosh(x, /, *, out=None)
tanh(x, /, *, out=None)
asinh(x, /, *, out=None)
acosh(x, /, *, out=None)
atanh(x, /, *, out=None)
# 반올림
floor(x, /, *, out=None)
ceil(x, /, *, out=None)
trunc(x, /, *, out=None)
round(x, /, *, out=None)
# 비교
equal(x1, x2, /, *, out=None)
not_equal(x1, x2, /, *, out=None)
less(x1, x2, /, *, out=None)
less_equal(x1, x2, /, *, out=None)
greater(x1, x2, /, *, out=None)
greater_equal(x1, x2, /, *, out=None)
# 논리
logical_and(x1, x2, /, *, out=None)
logical_or(x1, x2, /, *, out=None)
logical_xor(x1, x2, /, *, out=None)
logical_not(x, /, *, out=None)
# 비트
bitwise_and(x1, x2, /, *, out=None)
bitwise_or(x1, x2, /, *, out=None)
bitwise_xor(x1, x2, /, *, out=None)
bitwise_invert(x, /, *, out=None)
bitwise_left_shift(x1, x2, /, *, out=None)
bitwise_right_shift(x1, x2, /, *, out=None)
# 최대, 최소, 판별
maximum(x1, x2, /, *, out=None)
minimum(x1, x2, /, *, out=None)
isnan(x, /, *, out=None)
isinf(x, /, *, out=None)
isfinite(x, /, *, out=None)
# complex
real(x, /, *, out=None)
imag(x, /, *, out=None)
conj(x, /, *, out=None)
# 기타
clip(x, /, *, min=None, max=None, out=None)
where(condition, x1, x2, /, *, out=None)
```
- ufunc 객체는 없다. `add.reduce` 같은 메서드도 없다. `where=` 마스크 인자도 없다.
- 피연산자는 서로 브로드캐스트한다.

### §6.2 피연산자, 결과, `out=`
- **피연산자는 `sd.ndarray`와 Python 스칼라(`bool`, `int`, `float`, `complex`)만 받는다.**
  - 리스트, numpy 배열, 문자열, `None`은 `TypeError`다. 필요하면 먼저 `asarray`로 변환한다.
  - Python 스칼라 타입의 서브클래스와 numpy 스칼라도 받는다 (§1.4). 예: `numpy.float64`, `numpy.int64`, `numpy.bool`
  - 모든 원소별 함수가 같다. `real`, `imag`, `conj`도 Python 스칼라를 받는다.
  - 연산자의 규칙은 §10.3에 있다.
- **피연산자가 모두 Python 스칼라이면 결과도 Python 스칼라다** (§1.2). 계산은 dtype 의미론을 따른다.
  - 예: `add(2**62, 2**62)`는 wrap되어 `-2**63`이다. `add(True, True)`는 `2`다.
- **`out=`**
  - `sd.ndarray`여야 한다. 쓰기 금지 배열이면 `ValueError`다.
  - 입력은 `out`의 shape으로 브로드캐스트될 수 있어야 한다. 아니면 `ValueError`다.
  - 결과 dtype에서 `out`의 dtype으로는 상향 변환만 허용한다 (§1.5). 예: `divide(int 배열, ..., out=int 배열)`은 `TypeError`다.
  - `out`을 반환한다.
  - 입력과 메모리가 겹쳐도 된다 (§1.7).
- 새 결과 배열의 메모리 순서는 입력을 따른다 (§1.9).

### §6.3 결과 dtype
| 함수 | bool | int64 | float64 | complex128 |
|---|---|---|---|---|
| `add` `subtract` `multiply` `pow` `negative` `positive` `square` `sign` | int64 | int64 | float64 | complex128 |
| `floor_divide` `remainder` | int64 | int64 | float64 | `TypeError` |
| `abs` | int64 | int64 | float64 | float64 |
| `divide`, 지수·로그, 삼각·쌍곡선, `sqrt` | float64 | float64 | float64 | complex128 |
| `atan2` | float64 | float64 | float64 | `TypeError` |
| `floor` `ceil` `trunc` | bool | int64 | float64 | `TypeError` |
| `round` | bool | int64 | float64 | complex128 |
| `equal` `not_equal` | bool | bool | bool | bool |
| `less` `less_equal` `greater` `greater_equal` | bool | bool | bool | `TypeError` |
| `logical_*` | bool | `TypeError` | `TypeError` | `TypeError` |
| `bitwise_and` `bitwise_or` `bitwise_xor` `bitwise_invert` | bool | int64 | `TypeError` | `TypeError` |
| `bitwise_left_shift` `bitwise_right_shift` | `TypeError` | int64 | `TypeError` | `TypeError` |
| `maximum` `minimum` | bool | int64 | float64 | `TypeError` |
| `isnan` `isinf` `isfinite` | bool | bool | bool | bool |
| `real` `imag` | bool | int64 | float64 | float64 |
| `conj` | bool | int64 | float64 | complex128 |
- 표의 열은 승격한 입력 dtype이다 (§1.4). 예: bool과 int64의 `bitwise_and`는 int64다.
- `where`의 결과는 `x1`, `x2`를 승격한 dtype이다. `clip`의 결과는 `x`의 dtype이다.

### §6.4 함수별 규칙
- **`divide`**: 정수 입력도 float64로 계산한다.
- **`abs`**: complex는 크기를 float64로 반환한다.
- **`sign`**: 결과는 `-1`, `0`, `1`이다. float의 NaN은 NaN이다. complex는 `z / |z|`이고, `0`은 `0`이다.
- **`round`**: half-to-even이다. 예: `round(2.5)`는 `2.0`이다. 자릿수 인자는 없다. complex는 실수부와 허수부를 각각 반올림한다.
- **비교**: 결과는 bool이다. int와 float의 비교는 §1.6을 따른다.
- **`logical_and`, `logical_or`, `logical_xor`, `logical_not`**: bool만 받는다. 다른 dtype은 `x != 0`처럼 명시적으로 비교해서 넣는다.
- **비트 연산**: bool과 int64를 받는다. bool끼리는 bool이다.
- **시프트**: int64만 받는다. bool도 받지 않는다.
  - 시프트 양은 부호 없는 수로 해석한다. 64 이상이면(음수도 여기에 포함된다) 모든 비트가 밀려난다.
  - 그래서 왼쪽 시프트는 `0`이 되고, 오른쪽 시프트는 부호 비트로 채워져 `0` 또는 `-1`이 된다.
- **`maximum`, `minimum`**
  - NaN을 전파한다. 어느 쪽이든 NaN이면 결과는 NaN이다.
  - 부호 있는 0: `maximum(0.0, -0.0)`은 순서와 상관없이 `0.0`, `minimum`은 `-0.0`이다.
- **`isnan`, `isinf`, `isfinite`**: 모든 dtype을 받는다. complex는 실수부나 허수부 중 하나가 해당하면 참이다(`isfinite`는 둘 다 유한할 때 참).
- **`real`, `imag`, `conj`**
  - complex 입력의 `real`, `imag`는 stride를 이용한 **뷰**다. 쓰면 원본이 바뀐다.
  - complex가 아닌 입력의 `real`은 `x` 자신을 반환한다.
  - complex가 아닌 입력의 `imag`는 같은 dtype의 0 배열을 새로 만들어 반환한다. 이 배열은 쓰기 금지다.
  - `conj`는 언제나 새 배열이다. complex가 아니면 값이 같은 사본이다.
  - 셋 다 Python 스칼라를 받는다. 결과는 §6.3의 dtype에 맞는 Python 스칼라다. 예: `real(3 - 4j)`는 `3.0`, `imag(3)`은 `0`, `real(True)`는 `True`다.
- **`clip(x, /, *, min=None, max=None, out=None)`**
  - 결과는 `x`의 dtype이다.
  - `min`, `max`는 Python 스칼라 또는 `sd.ndarray`다. 배열이면 `x`에 브로드캐스트한다. 생략하면 그쪽은 자르지 않는다.
  - `min`, `max`는 `x`의 dtype으로 암묵적 변환될 수 있어야 한다 (§1.5). 예: int 배열에 `min=0.5`는 `TypeError`다.
  - `min > max`이면 결과는 `max`다.
  - NaN과 부호 있는 0은 `maximum`/`minimum`과 같은 규칙이다. `min`이나 `max`가 NaN이면 결과는 NaN이다.
  - complex는 `TypeError`다.
- **`where(condition, x1, x2, /, *, out=None)`**
  - 인자 3개 형태만 있다.
  - `condition`은 bool만 받는다. 다른 dtype은 `TypeError`다.
  - 세 인자를 서로 브로드캐스트한다.
- **정확도**
  - float64 입력의 `exp` `log` `log2` `log10` `sin` `asin` `acos` `atan` `atan2` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `pow`는 numpy와 몇 ULP까지 다를 수 있다 (8 ULP 이내).
  - 특수값(±0, ±inf, NaN)은 부호까지 정확하다.
  - 같은 입력이면 위치, stride, 스레드 수와 상관없이 같은 비트가 나온다. 예: `exp(x)[i] == exp(x[i])`
  - 그 밖의 원소별 함수는 numpy와 비트까지 같다.

---

## §7. 리덕션

### §7.1 함수
```
sum(x, /, *, axis=None, dtype=None, keepdims=False)
prod(x, /, *, axis=None, dtype=None, keepdims=False)
min(x, /, *, axis=None, keepdims=False)
max(x, /, *, axis=None, keepdims=False)
mean(x, /, *, axis=None, keepdims=False)
var(x, /, *, axis=None, correction=0, keepdims=False)
std(x, /, *, axis=None, correction=0, keepdims=False)
argmin(x, /, *, axis=None, keepdims=False)
argmax(x, /, *, axis=None, keepdims=False)
any(x, /, *, axis=None, keepdims=False)
all(x, /, *, axis=None, keepdims=False)
cumulative_sum(x, /, *, axis=None, dtype=None, include_initial=False)
cumulative_prod(x, /, *, axis=None, dtype=None, include_initial=False)
```
- `x`는 `sd.ndarray`여야 한다.
- 같은 이름의 메서드가 있다 (§10.2).

### §7.2 `axis`와 `keepdims`
- `axis`는 int 또는 tuple이다 (§1.3). `axis=None`은 모든 축이다.
  - `axis=()`는 아무 축도 줄이지 않는다.
  - `argmin`, `argmax`는 int만 받는다. `axis=None`이면 평탄화한 위치를 반환한다.
- 결과가 0차원이면 Python 스칼라다 (§1.2).
- `keepdims=True`이면 줄인 축을 길이 1로 남긴다. 결과는 언제나 배열이다.

### §7.3 결과 dtype
- `sum`, `prod`, `cumulative_*`: bool은 int64로 승격한다. 나머지는 입력 dtype을 유지한다.
- `min`, `max`: 입력 dtype을 유지한다. complex는 `TypeError`다.
- `mean`: bool과 int64는 float64로 계산한다. complex는 complex128이다.
- `var`, `std`: 결과는 float64다. complex 입력도 float64다 (`|x - mean|²`의 평균).
- `argmin`, `argmax`: int64. complex는 `TypeError`다.
- `any`, `all`: bool
- 정수 합과 곱은 wrap한다 (§1.6).

### §7.4 빈 배열과 자유도
- **빈 배열**
  - `sum`은 `0`, `prod`는 `1`이다.
  - `any`는 `False`, `all`은 `True`다.
  - `mean`, `var`, `std`는 NaN이다.
  - `min`, `max`, `argmin`, `argmax`는 `ValueError`다.
    - 정확히는 줄일 원소가 하나도 없는 출력 원소가 있을 때다. 출력 자체가 비어 있으면(예: shape `(3, 0)`에서 `axis=0`) 빈 배열을 반환한다.
- **`var`, `std`의 자유도**
  - 분모는 `N - correction`이다. 기본값 `correction=0`은 모분산이다. 표본 분산은 `correction=1`로 쓴다.
  - `N - correction ≤ 0`이면 결과는 NaN이다. 빈 배열의 NaN도 이 규칙에 포함된다.
  - `correction`은 int 또는 float다. bool은 `TypeError`다.

### §7.5 `dtype=` (`sum`, `prod`, `cumulative_*`)
- 명시적 변환이다 (§1.5). 입력을 그 dtype으로 변환한 뒤 누적한다.
  - 예: `sum(asarray([1.5, 1.5]), dtype=sd.int64)`는 `2`다.
  - complex 입력에 `dtype=sd.float64`는 `TypeError`다.
- `dtype=sd.bool`은 `TypeError`다. 덧셈과 곱셈에는 bool 계산이 없다 (§1.4). 대신 `any`/`all`을 쓴다.

### §7.6 NaN과 진릿값
- `sum`, `prod`, `mean`, `var`, `std`, `min`, `max`: NaN을 전파한다.
- `argmin`, `argmax`: NaN이 있으면 첫 NaN의 위치를 반환한다. 동점이면 첫 위치다.
- `any`, `all`: 모든 dtype을 받는다. 판정 기준은 `x != 0`이고, NaN은 참이다.

### §7.7 누적 (`cumulative_sum`, `cumulative_prod`)
- `axis`는 int다. 1차원이면 생략할 수 있고, 2차원 이상이면 필수다 (생략하면 `ValueError`).
- 결과의 shape은 `x`와 같다.
- `include_initial=True`이면 맨 앞에 시작값(합은 0, 곱은 1)을 넣는다. 그 축의 길이가 1 늘어난다.

### §7.8 계산 순서
- float/complex 리덕션의 덧셈·곱셈 순서는 numpy와 같다고 보장하지 않는다.
  - 그래서 결과의 마지막 비트가 numpy와 다를 수 있다. 결과는 결정적이다 (API.md §3).
  - 형변환이 없고 출력이 하나인 긴 합(예: float64 배열의 `sum(x)`)은 numpy와 비트가 같다.
- **float/complex `prod`, `cumulative_prod`의 오버플로**
  - 곱하는 순서가 numpy와 달라서, 중간값이 float 범위를 넘는 입력에서는 결과가 numpy와 크게 다를 수 있다.
  - numpy가 유한한 값을 낼 때 `inf`나 `nan`이 나올 수 있고, 그 반대도 있다.
  - 예: `prod(asarray([1e200, 1e-200] * 8))`은 `nan`이다. numpy는 `1.0`이다.
  - 이 경우에도 결과는 스레드 수와 무관하다.

---

## §8. 정렬과 집합

### §8.1 함수
```
sort(x, /, *, axis=-1, descending=False, stable=True)
argsort(x, /, *, axis=-1, descending=False, stable=True)
unique_values(x, /)
unique_counts(x, /)      # -> (values, counts)
unique_inverse(x, /)     # -> (values, inverse_indices)
unique_all(x, /)         # -> (values, indices, inverse_indices, counts)
```
- `x`는 `sd.ndarray`여야 한다.

### §8.2 `sort`, `argsort`
- `axis`는 int다. `axis=None`은 없다.
- bool, int64, float64를 받는다. complex는 `TypeError`다.
- `argsort`의 결과는 int64다.
- **NaN은 가장 큰 값이다.** `+inf`보다도 크다. 그래서 오름차순에서는 맨 끝, 내림차순에서는 맨 앞에 온다 (frame과 같다, frame.md §8.2).
  - NaN끼리는 언제나 원래 순서를 유지한다. 부호나 payload가 다른 NaN도 같은 값이다.
- `stable=True`이면 같은 값의 원래 순서를 유지한다. `descending=True`일 때도 그렇다.
  - 그래서 내림차순은 오름차순 결과를 뒤집은 것과 값의 순서는 같지만, 같은 값끼리의 순서는 뒤집지 않는다. 예: `argsort([1.0, nan, 3.0, 1.0, nan], descending=True)`는 `[1, 4, 2, 0, 3]`이다.
- `stable=False`이면 같은 값의 순서는 정해져 있지 않다. 예: `-0.0`과 `0.0`의 순서, `argsort`에서 같은 값의 인덱스 순서
  - 그래도 결정적이다. 스레드 수와 무관하다.

### §8.3 `unique_*`
- 입력을 평탄화해서 처리한다.
- 결과 `values`는 오름차순으로 정렬한다.
- **NaN은 모두 하나의 값이다.** 가장 큰 값이라 맨 끝에 온다 (§8.2). 부호나 payload가 다른 NaN도 같은 값이다.
  - frame의 키 규칙(frame.md §1.6), numpy 2의 `np.unique`(기본값 `equal_nan=True`)와 같다. Array API와 `np.unique_all`은 NaN을 서로 다른 값으로 본다.
- `-0.0`과 `0.0`은 같은 값이다.
- **같은 값으로 묶인 것 중에서는 처음 나온 값을 그대로 보여 준다.** `indices`와 같은 원소다.
  - 예: `unique_values([-0.0, 0.0])`는 `[-0.0]`, `[0.0, -0.0]`이면 `[0.0]`이다. NaN도 처음 나온 NaN의 비트를 그대로 보여 준다.
  - numpy는 정렬 결과에 따라 고르므로 `-0.0`과 `0.0` 중 무엇을 보여 줄지 정해져 있지 않다.
- bool과 complex도 받는다. complex의 순서는 (실수부, 허수부)의 사전순이다.
  - 실수부나 허수부가 NaN인 complex는 모두 NaN 하나로 묶여 맨 끝에 온다. numpy 2의 `np.unique`도 하나로 묶지만, 보여 주는 값은 numpy 순서에서 가장 앞인 NaN이다.
- 여러 값을 돌려주는 함수는 namedtuple을 반환한다. 타입은 세 영역이 함께 쓰는 `stardust._results`에 있고, 이름은 `<함수>Result`다 (linalg.md §1.6).

  | 함수 | 타입 | 필드 |
  |---|---|---|
  | `unique_counts` | `UniqueCountsResult` | `values`, `counts` |
  | `unique_inverse` | `UniqueInverseResult` | `values`, `inverse_indices` |
  | `unique_all` | `UniqueAllResult` | `values`, `indices`, `inverse_indices`, `counts` |
  - `indices`: 각 값이 평탄화한 입력에서 처음 나온 위치
  - `inverse_indices`: 입력의 각 원소가 `values`의 몇 번째인지. shape은 `x`와 같다.
  - `indices`, `inverse_indices`, `counts`는 int64다.

---

## §9. 행렬곱
```
matmul(x1, x2, /)
```
- `x1`, `x2`는 `sd.ndarray`여야 한다. Python 스칼라도 `TypeError`다. 연산자는 `@`, `@=`다.
- `dot`은 없다.
- **shape 규칙** (numpy, Array API와 같다)
  - 둘 다 1차원 이상이어야 한다.
  - `x1`이 1차원이면 앞에 축을 붙여 계산하고, 결과에서 그 축을 뺀다. `x2`가 1차원이면 뒤에 붙였다가 뺀다.
  - 1차원끼리의 곱(내적)은 Python 스칼라다 (§1.2).
  - 마지막 두 축 앞의 배치 축은 서로 브로드캐스트한다.
  - `x1`의 마지막 축과 `x2`의 끝에서 두 번째 축의 길이가 다르면 `ValueError`다.
- 결과 dtype은 승격 규칙을 따른다. bool끼리는 int64다 (§1.4). 정수 곱과 합은 wrap한다.
- complex도 받는다. `x1`을 켤레로 바꾸지 않는다.

---

## §10. ndarray
배열은 생성 함수로만 만든다. `sd.ndarray(...)`를 직접 부르면 `TypeError`다.

### §10.1 속성
```
dtype  shape  ndim  size  strides  itemsize  nbytes  base  T  mT
```
- 모두 읽기 전용이다. 대입하면 `AttributeError`다. 메시지에 속성 이름이 나온다: `property 'shape' of 'ndarray' object has no setter`
- `dtype`: dtype 객체 (API.md §1)
- `shape`: int의 tuple. `ndim`은 1 이상이다 (§1.2).
- `strides`: 바이트 단위의 tuple이다. 음수일 수 있다.
- `itemsize`: bool 1, int64 8, float64 8, complex128 16
- `nbytes`: `size * itemsize`
- `base`: 메모리를 실제로 소유한 객체다.
  - 뷰의 뷰도 중간 뷰를 거치지 않고 소유한 배열을 가리킨다.
  - 버퍼 프로토콜로 가져온 배열은 내보낸 객체를 가리킨다.
  - 배열이 메모리를 직접 소유하면 `None`이다.
- `T`: 2차원 배열의 전치 뷰다. 2차원이 아니면 `ValueError`다. 일반적인 축 순서 변경은 `permute_dims`로 한다.
- `mT`: 마지막 두 축을 바꾼 뷰다. `matrix_transpose`와 같다.

### §10.2 메서드
```
x.reshape(shape, /, *, copy=None)
x.astype(dtype, /, *, copy=True)
x.copy()
x.tolist()
sum  prod  min  max  mean  var  std  argmin  argmax  any  all
```
- `reshape`, `astype`과 리덕션 메서드는 같은 이름의 모듈 함수와 같다. `x`가 `self`이고 나머지 인자는 그대로다.
  - 예: `x.sum(axis=0)`, `x.var(correction=1)`, `x.reshape((3, 2))`
  - `reshape`의 shape은 인자 하나다. `x.reshape(3, 2)`는 `TypeError`다.
- `copy()`: 언제나 새 배열이다. 메모리 순서는 `x`를 따른다. `base`는 `None`이다.
- `tolist()`: 중첩 리스트를 반환한다. 원소는 Python 스칼라다 (§1.2의 대응).

### §10.3 연산자
| 연산자 | 함수 |
|---|---|
| `+` `-` `*` `/` `//` `%` `**` | `add` `subtract` `multiply` `divide` `floor_divide` `remainder` `pow` |
| `@` | `matmul` |
| `-x` `+x` `abs(x)` `~x` | `negative` `positive` `abs` `bitwise_invert` |
| `&` `\|` `^` `<<` `>>` | `bitwise_and` `bitwise_or` `bitwise_xor` `bitwise_left_shift` `bitwise_right_shift` |
| `==` `!=` `<` `<=` `>` `>=` | `equal` `not_equal` `less` `less_equal` `greater` `greater_equal` |
- 반사 연산자(`2 ** x`, `1 / x` 등)도 있다.
- `~`를 bool에 쓰면 논리 부정이다.
- **in-place 연산자** (`+=`, `@=` 등): 결과를 `x`에 쓴다.
  - 결과 dtype에서 `x`의 dtype으로는 상향 변환만 허용한다 (§1.5). 예: int64 배열의 `/=`, bool 배열의 `+=`는 `TypeError`다.
  - 오른쪽은 `x.shape`으로 브로드캐스트될 수 있어야 한다. `x`의 shape은 바뀌지 않는다.
  - 쓰기 금지 배열이면 `ValueError`다.
- **피연산자**는 원소별 함수와 같다 (§6.2). `@`는 `sd.ndarray`끼리만 된다 (§9).
  - 리스트나 numpy 배열과 섞어 쓰면 어느 쪽이 왼쪽이든 `TypeError`다.
  - **`==`와 `!=`도 같다.** `x == [1, 2]`, `x == np.array([1, 2])`, `x == None`, `x != "a"`는 `TypeError`다. `sd.equal(x, [1, 2])`와 같다.
    - Python의 기본 동작(양쪽이 `NotImplemented`면 객체 동일성 비교)으로 떨어지지 않는다. 받지 않는 피연산자이면 먼저 반대쪽 타입의 비교를 부르고, 그쪽도 처리하지 않으면 `TypeError`다.
    - 원소별로 비교하려면 먼저 `asarray`로 변환한다. `None`인지는 `x is None`으로 본다.
- `divmod`와 `in`은 지원하지 않는다 (`TypeError`).

### §10.4 Python 프로토콜
- `len(x)`는 `shape[0]`이다.
- 반복(iteration)은 축 0을 따라간다. 1차원 배열은 Python 스칼라, 2차원 이상은 뷰가 나온다.
- **`TypeError`가 나는 것**
  - 배열을 스칼라로 바꾸는 변환: `bool(x)`, `int(x)`, `float(x)`, `complex(x)`, `operator.index(x)`. 크기와 상관없이 언제나다. `if x:`도 여기에 해당한다.
  - 해싱
  - `copy.copy`, `copy.deepcopy`, `pickle`. 복사는 `x.copy()`로 한다.
- **numpy와의 관계**
  - `x.__array_ufunc__`는 `None`이다. 그래서 numpy 함수와 연산자에 `sd.ndarray`를 넣으면 `TypeError`다.
  - `np.asarray(x)`는 버퍼 프로토콜로 복사 없이 numpy 배열을 만든다 (§10.5).
  - numpy 배열을 `sd.ndarray`로 바꿀 때는 `sd.asarray`를 쓴다 (§3.3).

### §10.5 버퍼 프로토콜 내보내기
- format 문자는 bool `?`, int64 `q`, float64 `d`, complex128 `Zd`다.
- shape과 strides를 그대로 내보낸다. 복사가 없다.
- 쓰기 금지 배열은 읽기 전용 버퍼로 내보낸다.

### §10.6 `repr`, `str`
- `repr(x)`는 `array([...], dtype=<dtype>)` 형태다.
  - 빈 배열은 `array([], dtype=float64)`, 2차원 이상이면 `array([], shape=(2, 0), dtype=float64)`처럼 shape을 붙인다.
- `str(x)`는 대괄호 안의 값만 보여 준다. 예: `[1.0 2.0 3.0]`
- 모든 원소는 배열 전체에서 가장 넓은 원소에 맞춰 오른쪽 정렬한다. 한 줄은 75자를 넘지 않는다.
- 원소가 1000개를 넘으면 각 축의 양 끝 3개씩만 보여 주고 사이에 `...`을 넣는다.
- numpy의 출력과 같을 필요는 없다.
