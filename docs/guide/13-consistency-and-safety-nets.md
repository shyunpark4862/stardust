# 13장. 일관성과 안전망

> 세 프로젝트의 규칙을 어떻게 하나로 맞췄고, 무엇이 아직 다르며 왜인가? 합치는 동안 무엇이 결과를 지켰나?

## 1. 원칙: 같은 개념은 같게
DESIGN.md §1.4가 라이브러리 전체의 첫 원칙이다.
- **동작**: 같은 연산은 같은 입력에 같은 결과(값, dtype, 에러 종류)를 낸다.
- **시그니처**: 같은 개념의 인자는 이름, 기본값, 위치/키워드 규칙이 같다.
- **이름**: 같은 연산은 같은 이름을 쓴다.
- 차이를 남기려면 이유가 **데이터 모델**에 있어야 하고(예: frame에만 NA가 있다), 그 차이도 문서 한곳에 모은다.

옛 세 프로젝트는 각자 일관성 규칙이 있었다(linalg의 "모든 함수가 일관된 시그니처와 동작", pandas의 시그니처 일관성). 그런데 서로 출신이 달랐다. 배열과 linalg는 Python Array API, frame은 pandas 관례였다. 그래서 **기준을 하나로 정했다: Array API 관례** (DESIGN.md §8.1). 배열에 없는 개념(NA, 라벨, groupby, merge, CSV)에만 frame 고유의 이름을 쓴다.

## 2. 시그니처와 이름
| 규칙 | 배열 | linalg | frame |
|---|---|---|---|
| 주 인자는 `/` 앞(위치로만), 옵션은 `*` 뒤(키워드로만) | `reshape(x, shape, /, *, copy=None)` | `solve(a, b, /)` | `sort_values(by, /, *, descending=False, ...)` |
| 정렬 방향 | `descending=False` | (`eig`는 정해진 순서) | `descending=False` (`value_counts`는 `True`) |
| 정렬 안정성 | `stable=True` | | `stable=True` (언제나 안정이라 `False`여도 같다) |
| 자유도 | `correction=0` | | `correction=0` |
| 리스트로 | `tolist()` | | `tolist()` |
| 고유값 | `unique_values` (정렬 순서) | | `unique_values()` (정렬 순서) |
| `copy` | None / True / False | | `to_numpy`, `to_array`만, 같은 뜻 |
| `axis` | int, tuple, 리스트 | `vector_norm`도 리스트 | |
| 결과 namedtuple | `<함수>Result`, `stardust._results` 한 모듈 | 같은 모듈 | 같은 모듈 |

- **별칭이 없다.** 함수 하나에 이름 하나다. `flatten`, `dot`, `to_list`, `unique`, `isnull`이 없다.
- frame의 구현 모듈은 모두 밑줄 이름이다(`_series`, `_frame`, `_index` …). 전에는 `sd.frame.index`가 Index가 아니라 모듈이었다.

```python
import inspect
import stardust as sd

print(inspect.signature(sd.frame.DataFrame.sort_values))  # (self, by, /, *, descending=False, stable=True, na_position='last', ignore_index=False)
print(hasattr(sd.frame.Series, "to_list"), hasattr(sd.frame.Series, "unique"), hasattr(sd.frame, "index"))  # False False False
try:
    sd.frame.Series([3, 1]).head(n=1)
except TypeError as e:
    print(type(e).__name__)  # TypeError
```

## 3. dtype과 에러
- **dtype은 객체 다섯이다**: `sd.bool, sd.int64, sd.float64, sd.complex128, sd.string`. 모든 영역이 객체나 그 이름 문자열을 받고, `.dtype`은 객체다. Python 타입(`float`)과 다른 이름(`"float32"`)은 어디서나 TypeError다 (API.md §1).
  - 전에는 배열은 객체만, frame은 문자열과 Python 타입만 받았다.
  - Python 타입을 받지 않으니 `sd.bool`과 Python `bool`이 헷갈리는 문제도 같이 없어졌다.
- **에러 종류**: 타입이면 TypeError, 값이면 ValueError, 없는 라벨은 KeyError, int64 밖의 int는 OverflowError, 수치적 실패는 `LinAlgError(ValueError)`. 경고는 없다 (API.md §2).
- **에러 메시지**: 공개 함수와 메서드의 에러는 `이름(): `으로 시작하고, 인자는 작은따옴표로 쓴다. 연산자와 인덱싱은 Python의 형식이다. 메시지에 문서의 절 번호를 넣지 않는다(문서의 번호는 바뀌어도 메시지는 사용자 코드가 붙잡고 있을 수 있다) (API.md §2.1).
  - 배열과 linalg는 바인딩이 이름을 붙인다: `named("zeros", &zeros)`가 예외를 잡아 같은 종류로 메시지만 바꿔 다시 던진다. 에러가 없을 때의 비용은 없다. 그래서 여러 함수가 함께 쓰는 헬퍼는 이름을 몰라도 된다.
  - frame은 Python이 이름을 안다(`_errors.named`). 사용자가 닿는 C++ 커널은 `func=`를 받아 바인딩의 `call_named`가 붙인다.

```python
import stardust as sd

names = []
for call in (
    lambda: sd.astype(sd.ones(1), "float32"),                 # 배열
    lambda: sd.frame.Series([1]).astype("float32"),           # frame
    lambda: sd.linalg.vector_norm(sd.ones(2), ord="x"),       # linalg
):
    try:
        call()
    except TypeError as e:
        names.append(str(e).split(":")[0])
print(names)  # ['astype()', 'astype()', 'vector_norm()']
```

## 4. 값 규칙: 무엇을 어느 쪽으로 맞췄나
1라운드(§4.4), 2라운드(§8), M6 목록(§9)의 결정이다. 대부분 배열 규칙으로 맞췄다. 배열 규칙이 numpy, pandas와 같았고 배열 쪽 비용이 없었기 때문이다 (DESIGN.md §9.8).

| # | 항목 | 정한 규칙 | 바뀐 쪽 |
|---|---|---|---|
| 2 | bool 산술 | int64 (`True + True == 2`) | frame (전에 TypeError) |
| 3 | int와 float 비교 | int를 float64로 바꿔 비교 | frame (전에 반올림 없는 정확한 비교) |
| 4 | 대입, `fillna` | dtype만 보고 위로만 | frame (전에 값을 봄) |
| 5 | 생성자 `dtype=` | 명시적 변환, 손실 허용 | frame |
| 6 | `astype` float → int | 포화, NaN → 0 | frame (전에 ValueError) |
| 7 | 자유도 | `correction=0` | frame (전에 `ddof=1`) |
| 9 | 내림차순의 NaN | 맨 앞 (NaN이 가장 큰 값) | 배열 (전에 맨 뒤) |
| 10 | unique의 NaN | 하나의 값, `-0.0 == 0.0` | 배열 (전에 NaN마다 따로) |
| 11 | `mean(int64)` | 배열과 같은 float64 합 | frame (전에 int128 정확한 합) |
| 12 | 범위 밖 Python int | int64가 되어야 할 때만 OverflowError | frame (전에 언제나 에러) |
| 13 | 받지 않는 피연산자와 `==` | TypeError | 배열 (전에 조용히 False) |

- 9번과 10번은 frame의 규칙을 택했다. 내림차순이 오름차순의 거울이 되고, groupby에서 NaN마다 그룹이 생기지 않는다.
- 3번과 11번은 처음(1라운드)에 frame 규칙으로 정했다가 비용을 잰 뒤 사용자가 배열 규칙으로 바꿨다. 배열을 frame 규칙으로 바꾸면 int–float 비교가 1.55배(int가 ±2⁵³ 밖이면 10배) 느려지기 때문이다 (DESIGN.md §9.8).

## 5. 남은 차이와 이유
**NA에서 나온 차이** (데이터 모델, 유지)
- `int // 0`, `int % 0`, `int ** 음수`: 배열은 0, `a`, 정수 결과, frame은 NA (DESIGN.md §9.1의 1번)
- 값이 모자란 리덕션: 배열은 NaN이나 ValueError, frame은 NA (8번)
- numpy에서 만들기: 배열은 되도록 공유, frame은 언제나 복사 (Copy-on-Write, 12장 §4)

**영역의 성격에서 나온 차이** (유지)
- linalg는 bool/int64 입력을 float64로 계산한다(`trace(int)`는 float, `sum(diagonal(int))`는 int). BLAS/LAPACK이 float에서만 돈다.
- 대각 오프셋 `k`(Array API는 `offset`), `svd(mode=)`(Array API는 `full_matrices`). 라이브러리 안에서는 이미 일관된다(`sd.eye`의 `k`, `qr`의 `mode`).
- 배열 `.size`(원소 수)와 `GroupBy.size()`(그룹별 행 수)

**코드에 남은 차이** (M6에서 찾아 문서에만 적음, DESIGN.md §8.4)
- M8에서 고친 것: bool과 int의 비교(frame도 bool을 int64로 올린다), CSV의 정확성 규칙(`read_csv`도 list처럼 반올림한다, 11장 §6), `df.groupby(Series)`(이제 `'by'` 인자의 TypeError다). DESIGN.md §10.1에 M8의 목록이 있다.
- `arange`의 int 인자는 `dtype=float64`여도 int64 범위여야 한다.
- 메시지 몇 개: `df.iloc[0, 9]`가 열 위치인데 "rows"라고 한다 등.

```python
import stardust as sd

print((sd.asarray([True]) == 1).tolist())  # [True]
try:
    sd.frame.Series([True]) == 1
except TypeError as e:
    print(type(e).__name__)  # TypeError
```

## 6. 안전망
"옮기기와 고치기를 같은 마일스톤에 섞지 않는다"가 진행 원칙이었다 (ROADMAP.md). 옮기는 마일스톤(M1, M2, M5)은 결과가 비트까지 같아야 하고, 고치는 마일스톤(M3, M4, M6)은 바뀌는 것을 미리 적는다. 그것을 확인한 장치가 셋이다.

### 6.1 테스트
- 세 프로젝트의 테스트를 모두 옮겼고 마일스톤마다 전부 통과했다. 지금 2,968개다.
- 정답지는 numpy, scipy, pandas, 그리고 Python 자체(`sorted()`, `str(float)`, `float(text)`)다. 비교는 비트로 한다(`assert_same`: NaN은 어떤 NaN과도 같고, `-0.0`과 `+0.0`은 다르다). vForce만 ULP로 잰다.
- 스레드 결정성, 여러 Python 스레드의 동시 사용, 레이아웃(연속, 스칼라 브로드캐스트, strided)을 따로 시험한다.
- **API 표면 테스트**(`tests/<영역>/test_api_surface.py`)가 API 문서의 시그니처 블록을 읽어, 문서의 이름과 시그니처가 코드와 정확히 같은지 본다. 문서가 계약이다.
- 버그를 막는 테스트는 고치기 전 빌드에서 실패하는지 확인한다.

### 6.2 골든 출력
테스트는 대부분 "numpy와 같은가"를 본다. "옮기기 전과 비트까지 같은가"는 보지 못한다. numpy와 일부러 다르게 만든 곳(바깥 축 합의 순서, `prod`의 오버플로)은 허용 오차로 비교하고, linalg 결과의 strides 같은 것은 거의 보지 않는다. 그래서 M0에서 골든 출력을 만들었다 (docs/learn/M0-repository-history-baseline.md).
- **케이스 파일 하나를 옛 패키지와 새 패키지 양쪽에서 돌린다.** 케이스는 모듈을 `L.A`, `L.LA`, `L.F`로만 부른다. 기록할 때는 옛 모듈, 검사할 때는 `stardust`다.
- 결과는 dtype, shape, **strides**, 바이트의 SHA-256으로 저장한다. float 스칼라는 `float.hex()`, 에러는 종류와 메시지다.
- 케이스는 12,424개다(배열 7,746, linalg 1,844, frame 2,834). 모두 1, 3, 10스레드에서 돈다.
- **M1–M5 동안 비트까지 같았다.** M3는 linalg 결과의 `.base`(내부 블록 → None) 하나만 일부러 바꿨다. M5는 스레드 함수가 `sd`로 옮겨 가 메시지 6개만 달랐다.
- **M6는 결과를 일부러 바꾼 마일스톤이다.** 바꿀 것을 먼저 `bench/merge/golden/expected.py`에 적었다.
  - `EXPECTED`: `(케이스 이름의 정규식, 이유)`. 이유에는 DESIGN의 절 번호를 쓴다.
  - `EXPECTED_MESSAGES`: 메시지만 다른 차이만 받는다. 값이나 예외 종류가 다르면 패턴이 맞아도 MISMATCH다. 그래서 넓은 패턴도 메시지 말고는 아무것도 숨기지 않는다.
  - 아무것도 맞추지 못한 패턴은 STALE로 보인다. 목록이 실제 차이보다 넓어지지 않는다.
  - 결과: 예상된 차이 배열 1,697, linalg 78, frame 449개. **그 밖의 차이는 세 영역 모두 0개.**
- M6가 인자 이름을 바꾼 곳은 케이스 파일의 어댑터가 새 이름으로 바꿔 부른다(`ddof` → `correction`). 어댑터가 계산을 바꾸지 않았는지는 케이스를 고친 뒤 **옛 패키지로 다시 기록해** 저장된 골든과 같은지로 확인했다.

```bash
.venv/bin/python bench/merge/golden/golden.py check array    # linalg, frame도 같다
```

### 6.3 같은 자리 벤치마크
골든 출력은 비트만 본다. 성능은 따로 재야 한다. 5장 §3.3의 회귀(정의를 한곳으로 옮기기만 했는데 `sum`이 37% 느려짐)가 그 예다.
- 옛 패키지와 통합판을 **같은 자리에서 번갈아** 잰다 (`bench/merge/sameseat.sh`). 다른 날 잰 기준선은 큰 변화를 거르는 데만 쓴다. M0 기준선의 몇 줄은 그날만의 이상값이었다.
- 10% 넘게 느려진 줄은 원인을 밝힌다. 짧은 병렬 호출이나 메모리에 묶인 호출은 프로세스마다 두 갈래로 갈리므로, **프로세스를 여러 번 새로 띄워** 분포로 본다 (6장 §6).
- 그 밖의 함정: 같은 입력을 반복하면 분기 예측기가 외운다(frame은 "새 입력" 모드로 잰다), 새 메모리의 첫 쓰기 페이지 폴트, 짧은 루프가 놓인 주소(64바이트 인출 블록 경계에 따라 1.7배).

### 6.4 문서 참조
- 코드 주석과 문서는 `docs/design/array.md §6.2` 같은 꼴로 서로를 가리킨다. 절 번호를 바꾸지 않는 것이 규칙이다.
- `tools/check_refs.py`가 코드, 테스트, 문서(이 안내서 포함)의 모든 참조가 있는 절을 가리키는지 본다.

```bash
.venv/bin/python tools/check_refs.py
```

## 더 읽을 곳
- [DESIGN.md §1.4](../../DESIGN.md) 일관성 원칙, [DESIGN.md §8.1](../../DESIGN.md) 기준을 하나로, [DESIGN.md §8.4](../../DESIGN.md) 이름과 시그니처 목록(남은 차이), [DESIGN.md §9.1](../../DESIGN.md) 값 규칙
- [API.md §1](../../API.md) dtype, [API.md §2](../../API.md) 에러 종류, [API.md §2.1](../../API.md) 에러 메시지
- [docs/design/array.md §13.1](../design/array.md), [docs/design/frame.md §13.1](../design/frame.md) 테스트, [docs/design/array.md §13.2](../design/array.md), [docs/design/frame.md §13.2](../design/frame.md) 측정 방법
- `docs/learn/M6-consistency.md` M6의 단계와 골든 장치, `docs/learn/M7-final-report.md` 통합의 최종 보고서
