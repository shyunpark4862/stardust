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

---

## §1. dtype
```
bool  int64  float64  complex128
```
- dtype은 위의 객체 넷이다. 배열의 `.dtype`도 이 객체를 돌려준다 (`x.dtype is sd.float64`).
- dtype을 받는 인자(`dtype=`, `astype`)는 이 객체만 받는다.
  - Python 타입(`float` 등)과 문자열(`"float64"`)은 `TypeError`다.
- dtype 객체는 `==`로 비교하고 해싱할 수 있다. `str(sd.int64)`는 `'int64'`다.
- `sd.bool`은 Python의 `bool`과 다른 객체다.
- **예외**: 데이터프레임 영역은 아직 dtype을 이름 문자열로 받고 돌려준다 (docs/api/frame.md §1.2). 두 영역의 표기는 M6에서 맞춘다 (DESIGN.md §4.2).

## §2. 에러 종류
- **`TypeError`**: 인자의 타입이 맞지 않는다.
  - 배열이 필요한 자리에 다른 객체를 넣었다.
  - 함수가 그 dtype을 받지 않는다 (예: complex의 순서 비교).
  - 암묵적 하향 변환이다 (docs/api/array.md §1.5).
  - bool 옵션에 `True`/`False`가 아닌 값을 넣었다 (docs/api/array.md §1.1).
- **`ValueError`**: 타입은 맞지만 값이 맞지 않는다.
  - shape이 맞지 않는다. 축 번호가 범위 밖이거나 중복되었다.
  - 인자 값이 허용 범위 밖이다 (음수 길이, `step=0`, 스레드 수).
  - 0차원 배열을 요청했다 (docs/api/array.md §1.2).
  - `copy=False`인데 복사가 필요하다 (docs/api/array.md §1.8).
  - 쓰기 금지 배열에 썼다.
  - 원소가 없는 `min`/`max`/`argmin`/`argmax` (docs/api/array.md §7.4)
- **`IndexError`**: 인덱스가 범위 밖이거나 허용하지 않는 형태다 (docs/api/array.md §5).
- **`KeyError`**: 없는 라벨을 찾았다 (데이터프레임의 `.loc`, docs/api/frame.md §5.4).
- **`OverflowError`**: int64 범위를 벗어난 Python int를 int64로 바꿔야 한다 (docs/api/array.md §1.5).
- **영역 고유의 예외**는 위 내장 예외의 하위 클래스다. 지금은 `sd.linalg.LinAlgError(ValueError)` 하나다: 특이 행렬, 양정치가 아닌 행렬, 수렴 실패 (docs/api/linalg.md).
- 배열의 원소별 연산은 원소의 값 때문에 실패하지 않는다 (docs/api/array.md §1.6).
- 어느 영역도 경고를 내지 않는다. 문제가 있으면 에러를 낸다.

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
