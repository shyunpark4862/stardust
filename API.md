# stardust API

`import stardust as sd`. 이 문서는 라이브러리 전체의 공통 규칙과 영역별 API 문서의 색인이다.

- 영역 사이의 일관성이 가장 앞선 원칙이다 (DESIGN.md §1.4). 같은 개념은 어느 영역에서든 같은 이름, 같은 시그니처, 같은 동작이다. 기준은 Python Array API 관례다 (DESIGN.md §8.1).
- 절 번호(`§N`)는 코드 주석이 인용한다. 번호를 바꾸지 않는다.
- 언어 표시가 없는 코드 블록은 영역마다의 API 표면 테스트(`tests/<영역>/test_api_surface.py`)가 읽는다. 이 문서의 블록은 배열 영역의 테스트가 읽는다.

| 영역 | 이름공간 | 문서 |
|---|---|---|
| 배열 | `sd` | [docs/api/array.md](docs/api/array.md) |
| 선형대수 | `sd.linalg` | [docs/api/linalg.md](docs/api/linalg.md) |
| 데이터프레임 | `sd.frame` | [docs/api/frame.md](docs/api/frame.md) |
| 특수함수 | `sd.special` | [docs/api/special.md](docs/api/special.md) (S0–S11 완료: 로지스틱·정보 7, 오차함수 8, 감마·베타·제타 9, 불완전 감마·베타 8, 리덕션·지수적분·타원 7, Bessel·Airy 21, 초기하·Lambert W 3, 복소수 2와 erf·erfc·erfcx·lambertw의 complex) |
| 통계 | `sd.stats`, `sd.stats.tests` | [docs/api/stats.md](docs/api/stats.md) (설계 초안, 구현 전) |
| 난수 | `sd.random` | [docs/api/random.md](docs/api/random.md) (설계 초안, 구현 전) |

- 설계의 근거와 측정은 `docs/design/<영역>.md`, 통합의 결정은 [DESIGN.md](DESIGN.md)에 있다.

---

## §1. dtype
```
bool  int64  float64  complex128  string
```
- dtype은 위의 객체 다섯이다. 배열과 데이터프레임 모두 `.dtype`이 이 객체를 돌려준다 (`x.dtype is sd.float64`).
  - 배열은 앞의 넷을, 데이터프레임은 `complex128`을 뺀 넷을 쓴다. 영역이 쓰지 않는 dtype을 주면 `TypeError`다 (예: `sd.zeros(3, dtype=sd.string)`).
- dtype을 받는 인자(`dtype=`, `astype`)는 **이 객체나 그 이름 문자열**을 받는다: `dtype=sd.float64`와 `dtype="float64"`는 같다.
  - Python 타입(`float` 등), numpy dtype, 다른 이름(`"float32"`)은 `TypeError`다.
- dtype 객체는 `==`로 비교하고 해싱할 수 있다. `str(sd.int64)`는 `'int64'`다. 이름 문자열과 `==`로 같지는 않다 (`sd.int64 == "int64"`는 False).
- `sd.bool`은 Python의 `bool`과 다른 객체다.

## §2. 에러 종류
- **`TypeError`**: 인자의 타입이 맞지 않는다.
  - 배열이 필요한 자리에 다른 객체를 넣었다.
  - 함수가 그 dtype을 받지 않는다 (예: complex의 순서 비교).
  - 암묵적 하향 변환이다 (docs/api/array.md §1.5). 데이터프레임의 대입과 `fillna`도 같다 (docs/api/frame.md §3.6, §5.7).
  - bool 옵션에 `True`/`False`가 아닌 값을 넣었다 (docs/api/array.md §1.1). `0`, `1`, `None`, `numpy.bool_`도 그렇다.
  - int 인자에 int가 아닌 값을 넣었다. **int 인자는 `operator.index`의 규칙으로 읽는다**: Python int와 numpy 정수(`np.int64(3)` 등 `__index__`가 있는 객체)는 int이고, bool과 `numpy.bool_`은 아니다. 세 영역의 모든 int 인자(`shape`, `axis`, `k`, `n`, `num`, `head(n)`, `RangeIndex`, `set_num_threads` …)와 인덱싱이 같다. int나 float를 받는 인자(`correction`, `ord`, `rtol`)도 numpy 정수를 int로 받는다 (DESIGN.md §10.1 13번).
  - 원소 하나에 스칼라가 아닌 값을 대입했다: `x[0] = 'a'`, `x[0] = [7]` (docs/api/array.md §5.3)
- **`ValueError`**: 타입은 맞지만 값이 맞지 않는다.
  - shape이 맞지 않는다. 축 번호가 범위 밖이거나 중복되었다.
  - 인자 값이 허용 범위 밖이다 (음수 길이, `step=0`, 스레드 수).
  - 0차원 배열을 요청했다 (docs/api/array.md §1.2).
  - `copy=False`인데 복사가 필요하다 (docs/api/array.md §1.8).
  - 쓰기 금지 배열에 썼다.
  - 원소가 없는 `min`/`max`/`argmin`/`argmax` (docs/api/array.md §7.4)
- **`IndexError`**: 인덱스가 범위 밖이거나 허용하지 않는 형태다 (docs/api/array.md §5). 데이터프레임의 위치 인덱스(`.iloc`)도 같다: `s.iloc[1.0]`, `s.iloc[True]`, `s.iloc[[1.0]]` (docs/api/frame.md §5.4).
- **`KeyError`**: 없는 라벨을 찾았다 (데이터프레임의 `.loc`, docs/api/frame.md §5.4).
- **`OverflowError`**: int64 범위를 벗어난 Python int를 int64로 바꿔야 한다 (docs/api/array.md §1.5, docs/api/frame.md §2.2). float로 계산할 수 있으면 float로 바꾼다.
  - int 인자(`shape`, `axis`, `k`, `n`, `num` 등)도 int64 범위 밖이면 `OverflowError`다. 배열과 linalg가 같다 (docs/api/array.md §1.3, docs/api/linalg.md §1.1). 예외는 `set_num_threads`의 `n`이다: 범위 밖은 크기와 상관없이 `ValueError`다 (§3).
- **영역 고유의 예외**는 위 내장 예외의 하위 클래스다. 지금은 `sd.linalg.LinAlgError(ValueError)` 하나다: 특이 행렬, 양정치가 아닌 행렬, 수렴 실패 (docs/api/linalg.md).
- 배열의 원소별 연산은 원소의 값 때문에 실패하지 않는다 (docs/api/array.md §1.6).
- 어느 영역도 경고를 내지 않는다. 문제가 있으면 에러를 낸다.

### §2.1 에러 메시지
- **공개 함수와 메서드가 내는 에러는 그 이름과 `(): `로 시작한다.** 예: `vector_norm(): 'ord' must be ...`, `astype(): 'dtype' must be ...`
  - 메서드는 클래스 이름 없이 메서드 이름만 쓴다. `x.astype(...)`과 `sd.astype(x, ...)`은 둘 다 `astype(): `, `s.sort_values()`와 `df.sort_values()`는 둘 다 `sort_values(): `다. 같은 이름은 어느 클래스, 어느 영역에서든 같은 일을 하기 때문이다 (DESIGN.md §1.4).
  - 생성자는 클래스 이름이다: `Series(): `, `DataFrame(): `, `Index(): `, `RangeIndex(): `
  - 프로퍼티는 점을 붙인 이름이다: `.T: `, `.mT: `
  - 사용자가 부른 함수의 이름이다. 여러 함수가 함께 쓰는 검사와 커널은 이름을 넘겨받는다. 예: `drop_duplicates(keep=...)`의 검사는 `duplicated(): `가 아니라 `drop_duplicates(): `, `GroupBy.sum()`은 `sum(): `, `agg("sum")`은 `agg(): `
  - `LinAlgError`도 같다: `inv(): singular matrix`
- **인자는 작은따옴표 안에 이름을 쓴다**: `'axis'`, `'dtype'`. 인자와 값을 함께 보일 때는 호출 모양 그대로 쓴다: `copy=False`, `header=False`
  - 인자가 아닌 낱말에는 따옴표가 없다. 예: `axis 3 is out of bounds for an array of dimension 2`의 axis는 배열의 축이다.
  - 받은 값은 타입 이름으로 보인다(`not list`). 문자열과 타입은 그 자체로 보인다(`not 'float32'`, `not <class 'float'>`).
  - **타입 이름은 모듈 경로 없는 짧은 이름이다**: `not ndarray`, `not int64`, `not Int64DType`, `not <class 'int64'>`. `stardust._core.ndarray`, `numpy.int64`, `numpy.dtypes.Int64DType`처럼 쓰지 않는다.
- **같은 실수에는 어느 영역에서든 같은 메시지다** (DESIGN.md §10.3). 옵션의 종류마다 검사가 하나라서 형식이 하나다.
  - bool 옵션: `sort(): 'stable' must be True or False, not int`, `sum(): 'skipna' must be True or False, not NoneType`
  - int 인자: `head(): 'n' must be an int, not float`, int64 범위 밖은 `head(): 'n' is out of range`(`OverflowError`)
  - 문자열 옵션: `merge(): 'how' must be 'inner', 'left', 'right' or 'outer', not NoneType`(타입, `TypeError`), `..., not 'x'`(값, `ValueError`). linalg의 `mode`도 같다.
  - 수 인자: `std(): 'correction' must be an int or a float, not NoneType`
  - **`None`도 같은 검사를 거친다.** `None`이 뜻이 있는 옵션(`axis=None`, `dtype=None`, `copy=None`, `rtol=None`)이 아니면 다른 틀린 값과 같다: `svd(): 'mode' must be 'reduced' or 'complete', not NoneType`, `astype(): 'dtype' must be ... not NoneType`
- **공개 함수의 메시지에 내부 이름이 나오지 않는다.** C++ 클래스(`Column`), 바인딩의 시그니처(nanobind의 `incompatible function arguments` 목록), Python 내부의 메시지(`object of type 'int' has no len()`, `'>=' not supported between ...`, `AttributeError`)가 그렇다. 인자를 먼저 검사해서 위의 형식으로 낸다.
- **Python 프로토콜은 Python의 형식이고 이름을 붙이지 않는다**: 연산자, 인덱싱과 그 대입(`x[i]`, `.loc`, `.iloc`, `df["c"]`), `len`, `in`, `bool`, `iter`, `repr`, 속성 대입
  - 연산자: `unsupported operand dtypes for <: complex128 and complex128`, 단항 연산자는 `unsupported operand dtype for unary -: string`, `abs()`는 `unsupported operand dtype for abs(): string`. 받지 않는 타입의 피연산자는 Python 자신의 `unsupported operand type(s) for +: ...`, `'==' not supported between instances of ...`다.
  - 같은 연산도 부르는 길에 따라 형식이 다르다. `sd.add(x, y)`의 shape 오류는 `add(): shapes could not be broadcast together: ...`이고 `x + y`는 `shapes could not be broadcast together: ...`다.
  - 인덱싱이 없는 라벨이나 열을 찾으면 dict처럼 `KeyError(label)`다. 함수가 없는 열을 받으면 `sort_values(): no column 'x'`다.
  - 속성 대입: `property 'shape' of 'ndarray' object has no setter`
- Python과 nanobind가 직접 내는 에러는 그대로다: 틀린 호출 모양(`Series.head() got some positional-only arguments passed as keyword arguments: 'n'`, 없는 키워드나 인자 개수가 틀린 호출에 대한 nanobind의 `incompatible function arguments`), 파일을 열 때의 `FileNotFoundError`. 인자의 **값**이 틀린 것은 언제나 위의 형식이다.
- **메시지에 문서의 절 번호를 넣지 않는다.** 문서의 번호는 바뀌어도 메시지는 사용자 코드가 붙잡고 있을 수 있다.

## §3. 스레드와 결정성
```
set_num_threads(n, /)
get_num_threads()
```
- `n`은 1 이상 1024 이하의 int다.
  - 범위 밖이면 `ValueError`, int가 아니면(`bool` 포함) `TypeError`다.
- 기본값은 P코어 수(10)다.
- **결과는 스레드 수와 무관하게 비트까지 같다.** 리덕션과 정렬도 결정적이다.
  - 같은 데이터와 같은 메모리 레이아웃이면 언제나 같은 비트가 나온다.
- `n`이 1이면 스레드풀을 쓰지 않는다. 환경 변수는 두지 않는다.
- 여러 스레드로 계산할 때는 GIL을 놓는다. 데이터프레임 영역은 큰 작업이면 1스레드여도 GIL을 놓는다 (docs/api/frame.md §12).
- 다른 Python 스레드가 이미 스레드풀을 쓰고 있으면, 새 호출은 1스레드로 계산한다. 결과는 같다.
