# M2. linalg 옮기기

> stardust-linalg를 같은 확장 모듈 안의 하위 모듈 `sd.linalg`로 옮겼다. **동작은 하나도 바꾸지 않았다.** 배열과의 경계(buffer protocol로 읽고 `sd.asarray`로 감싸는 길)도 일부러 그대로 두었다. 경계를 없애는 것은 M3다.

## 1. 결과
| 항목 | 결과 |
|---|---|
| 옮긴 파일 | C++ 37개, Python 3개, 테스트 16개 (`git mv`), linalg의 스레드풀 복사본 2개는 지움 |
| 테스트 | 2,424개 통과 (배열 2,085 + linalg 339) |
| 골든 출력 | linalg 1,844개, 배열 7,746개 모두 같음 (메시지 포함) |
| 이력 | `git log --follow csrc/linalg/ops/expm.cpp`가 옛 linalg M10 커밋까지 이어진다 |
| 새 문서 | `docs/api/linalg.md` (34KB), `docs/design/linalg.md` (87KB), 공통 규칙은 루트 `API.md` §1–§3 |
| 주석 참조 | 46개 파일, 약 270줄을 새 문서로 바꿈. 코드 토큰은 그대로임을 확인 |

## 2. 하위 모듈로 만들기
- 옛 linalg는 자기 확장 모듈(`stardust_linalg._core`)이 따로 있었다. 지금은 `stardust._core` 하나 안에서 `def_submodule("linalg")`로 하위 모듈을 만든다 (DESIGN §3.2).
  - 옛 `NB_MODULE(_core, m)`이 `bind/linalg_bind.cpp`의 `bind_linalg(core)`가 되었다. `module.cpp`가 배열 바인딩을 모두 마친 뒤 **마지막에** 부른다.
- **초기화 순서 문제 하나**: linalg는 시작할 때 배열 쪽 Python 객체(`ndarray` 타입, `asarray`, `get_num_threads`, dtype 객체 넷)를 찾아 캐시한다. 옛 코드는 `import stardust_numpy`로 찾았다.
  - 같은 모듈이 된 지금 `import stardust`를 하면, `stardust/__init__.py`가 아직 `from . import _core`를 실행하는 중이라 반쯤 만들어진 모듈이 나온다. `stardust.ndarray`가 아직 없다.
  - 그래서 `load_sdnp_refs(core)`가 `stardust._core` 모듈 자체에서 찾는다. 이 객체들은 모두 `_core`의 속성이고, 배열 바인딩이 먼저 끝났으니 이미 있다.
- Python 쪽: `sd.linalg`(`src/stardust/linalg/__init__.py`)가 `_core.linalg`의 함수를 `__all__`대로 가져온다. `stardust/__init__.py`의 맨 끝에서 `from . import linalg`를 해서 `import stardust`만으로 `sd.linalg`가 생긴다.

## 3. 이름공간과 파일
| 옛 위치 | 새 위치 |
|---|---|
| `csrc/core/*` (args, batch, cost, errors, input, outblock, pack, pairwise, pyrefs, types) | `csrc/linalg/` |
| `csrc/ops/*` | `csrc/linalg/ops/` |
| `csrc/lapack/lapack.hpp` | `csrc/accelerate/lapack.hpp` (vForce와 한곳에) |
| `csrc/bind/module.cpp` | `csrc/bind/linalg_bind.cpp` |
| `csrc/parallel/pool.*` (복사본) | **지움** |

- C++ 이름공간 `sdla` → `sd::linalg`. linalg 코드 안에서 `DType`, `itemsize` 같은 이름은 여전히 linalg 자기 것(`sd::linalg::DType`)을 가리킨다. 이름 찾기가 안쪽 이름공간에서 먼저 멈추기 때문이다. 배열의 `sd::DType`과 합치는 것은 M3다.
- `parallel::`은 linalg 안에 더 이상 없으니 바깥의 `sd::parallel`(배열의 풀)을 찾는다.

## 4. 스레드풀이 하나가 되었다
- 세 벌 중 두 벌(배열, linalg)이 하나로 줄었다. 두 벌의 차이는 linalg판의 `available_here()` 하나뿐이었다(옛 M0 대조). 이 함수를 공용 풀에 더했다. 기존 동작에는 영향이 없는 추가다.
- linalg는 호출마다 `sd.get_num_threads()`를 Python으로 불러 자기 풀의 크기를 맞추던 코드가 있다. 풀이 하나가 되어 이제 "같은 풀의 크기를 같은 값으로 맞추기"라 아무 일도 하지 않는다. 경계 코드라서 M3에서 없앤다.
- **사용자 입장의 변화**: `sd.set_num_threads(n)` 하나가 배열과 linalg를 모두 바꾼다. 옛날에도 linalg가 sdnp의 값을 따랐으므로 결과는 같다.

## 5. 컴파일 플래그
- linalg는 이제 배열과 같은 플래그로 컴파일된다. 새로 더해진 것은 `-fno-builtin-sin -fno-builtin-cos`다 (clang이 같은 x의 sin과 cos를 `__sincos_stret` 하나로 합치지 못하게 함).
- linalg의 결과는 골든 1,844개에서 하나도 바뀌지 않았다. 당연한 결과다: `csrc/linalg/`에는 `sin`, `cos` 호출이 하나도 없다 (grep으로 확인).

## 6. 테스트
- `tests/linalg/`도 패키지다 (`from .conftest import ...`).
- linalg의 conftest에는 numpy 배열을 sd 배열로 바꾸는 도우미 `sd(x)`가 있었다. 모듈 별칭 `sd`와 겹쳐서 **먼저** `as_sd(x)`로 바꾼 뒤 `sdla.` → `sd.linalg.`, `sdnp` → `sd`를 바꿨다.
- 한 번 실수가 있었다: 하위 프로세스로 돌리는 테스트 코드 문자열 안의 `import stardust_linalg as sdla`가 `import stardust.linalg as sd.linalg`가 되어 문법 오류가 났다. `import stardust as sd`로 고쳤다.

## 7. 공통 규칙을 루트 API.md로
- M1에서 `docs/api/array.md §0`에 두었던 공통 규칙(dtype, 에러 종류, 스레드)을 루트 `API.md` §1–§3으로 옮겼다.
- 옮기면서 하나를 고쳤다: "원소의 값 때문에 연산이 실패하는 일은 없다"는 배열 규칙이다. linalg는 특이 행렬에서 `LinAlgError`를 낸다. 공통 문서에는 "영역 고유의 예외는 내장 예외의 하위 클래스다(`LinAlgError(ValueError)`)"를 적고, 값으로 실패하지 않는 규칙은 배열 규칙으로 남겼다.
- 배열 API 표면 테스트는 `API.md`와 `docs/api/array.md`를 함께 읽는다(`set_num_threads`가 이제 API.md에 있다).

## 8. 문서와 주석

### 8.1 새 문서
- `docs/api/linalg.md`: §1 공통 규칙(시그니처, 입력과 배치, 결과 dtype, 결과 레이아웃, 실패, 유한성 검사, 빈 행렬, 결정성), §2–§7 함수 계열, §8 `LinAlgError`. 라이브러리 공통 부분은 `API.md`를 인용한다.
- `docs/design/linalg.md`: §1 기반 … §13 테스트. 배열과의 경계(§3)는 ROADMAP대로 **M3의 모습으로 미리** 썼고, 지금의 임시 경로(buffer protocol, 결과 하나에 약 200ns)는 "M2 현재" 표시로 한 곳(§3.6)에만 적었다.
- 쓰면서 옛 문서가 코드와 달랐던 곳
  - 옛 API의 결과 namedtuple 이름 규칙("행렬은 대문자 한 글자, 벡터는 소문자 단어")은 코드와 맞지 않았다. `SVDResult.S`는 벡터이고 `eigenvectors`는 행렬이다. 새 문서는 코드가 따르는 규칙을 적었다.
  - 옛 설계의 "호출마다 `sdnp.get_num_threads()`를 부른다"는 실제로는 병렬로 나눌지 따져 볼 때만 부른다. 지금은 같은 풀이라 아무 일도 하지 않는다.
- **알려진 정리 항목** (`docs/design/linalg.md` §10.9): 옛 코드 리뷰에서 남겨 둔 것들이다. 하나는 정의되지 않은 동작이다: `trace`/`diagonal`에서 k = −2⁶³이면 `-k`가 부호 있는 오버플로다. 지금은 우연히 맞는 결과가 나온다. 나머지는 중복 코드다. M3에서 함께 고친다 (ROADMAP M3).

### 8.2 배열과 linalg 사이의 불일치 (2라운드로)
linalg API 문서를 쓰다가 찾은 것을 DESIGN §8.4에 옮겼다. 일관성 원칙(DESIGN §1.4)에서 특히 큰 둘:
- **결과 dtype**: linalg는 int 입력을 float64로 계산한다. `sd.linalg.trace(int 행렬)`은 float인데 `sd.sum(sd.linalg.diagonal(int 행렬))`은 int다.
- **결과 레이아웃과 `.base`**: linalg 결과는 F-order이고 `.base`가 내부 블록이다. 배열의 "새 배열은 C-order, `.base`는 None" 규칙과 다르다. `.base`는 M3에서 결과를 `NDArray`로 직접 만들면 자연히 맞는다.

### 8.3 주석
- 46개 파일, 약 270줄: 옛 linalg 문서와 옛 numpy 문서를 인용하던 곳을 네 개의 대응표(`docs/*/array-refmap.md`, `docs/*/linalg-refmap.md`)로 새 문서에 이었다.
- "stardust-numpy에서 복사한 풀", "스레드 수는 호출마다 읽는다"처럼 **두 패키지 시절을 지금 일처럼 말하던 주석**은 지금의 사실(모듈 하나, 풀 하나, 경계 코드는 M3까지 남음)로 고쳤다.
- 검증은 M1과 같다: 주석과 docstring을 뺀 토큰열이 같고, 테스트 2,424개 통과, 골든 linalg 1,844개·배열 7,746개 모두 같다.

### 8.4 API 표면 테스트의 파서 하나로
- 배열과 linalg의 API 표면 테스트가 문서를 읽는 방식이 달랐다. 배열은 언어 표시 없는 블록만, linalg는 펜스를 만날 때마다 안팎을 뒤집어서 ```` ```python ```` 블록 안의 줄까지 읽었다.
- `tests/_api_doc.py`의 `signature_blocks` 하나로 합쳤다. 두 영역이 같은 규칙으로 문서를 읽는다 (일관성 원칙은 테스트에도 적용).

## 9. 리뷰 포인트
1. **경계 코드를 일부러 남겼다.** 같은 모듈 안에서 buffer protocol로 배열을 읽고 `sd.asarray`로 결과를 감싸는 어색한 상태다. 옮기기와 고치기를 나눈다는 원칙 때문이다. M3에서 없애고 줄어든 시간을 잰다.
2. **2라운드에 더할 것**: 결과 dtype(§8.2)은 "int 입력의 `trace`가 int를 내야 하는가"라는 동작 결정이다. M6 전에 함께 정한다.
3. **정의되지 않은 동작 하나**(`-k` 오버플로)가 옛 코드에서 넘어왔다. M3에서 고친다.
4. 다음은 M3(linalg 경계 없애기)다. 결과가 비트까지 같은 채로 호출 비용을 얼마나 줄이는지 잰다.
