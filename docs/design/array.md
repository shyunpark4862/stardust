# 배열 영역 설계 (sd)

최상위 `sd`(배열 영역)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙과 통합 결정은 [DESIGN.md](../../DESIGN.md), 층 구조와 디렉터리는 [ARCHITECTURE.md](../../ARCHITECTURE.md)에 있다. 공개 함수 목록과 시그니처는 `docs/api/array.md`에 있다.
- 이 문서는 "지금 코드가 어떻게 되어 있고 왜 그런가"만 적는다. 코드 주석은 이 문서의 절 번호(`array.md §N.M`)를 가리킨다.
- 수치는 Apple M4 Pro 한 대에서 잰 값이다. 따로 적지 않으면 원소 100만 개, float64, 1스레드다. 비교 대상은 numpy 2.5.4다.
- 통합 중에 바뀔 곳은 한 줄로만 표시한다 (`통합 중 Mn`). 진행 순서는 [ROADMAP.md](../../ROADMAP.md)에 있다.

---

## 1. 기반

### 1.1 대상 하드웨어
`sysctl`로 확인한 값이다. 버퍼 크기, 정렬 단위, 누산기 개수가 여기서 나온다.

| 항목 | 값 | 설계에 주는 영향 |
|---|---|---|
| 캐시 라인 | 128B | 할당을 128B로 정렬한다 (§3.2) |
| 페이지 | 16KB | 새 메모리의 첫 쓰기에서 페이지 폴트가 난다 (§6.7, §13.2) |
| P코어 L1d | 128KB | 버퍼 캐스팅의 버퍼 크기 기준 (§4.3) |
| P코어 L2 | 16MB, 코어 5개가 공유, 클러스터 2개 | — |
| E코어 L1d / L2 | 64KB / 4MB | — |
| 코어 | P코어 10 + E코어 4 | 기본 스레드 수 10 (§10.3) |
| NEON 폭 | 128비트 | float64·int64는 2개, complex128은 1개 |
| 64비트 정수 벡터 곱셈 | 없음 | int64 곱셈은 스칼라 (§9.3) |
| 64비트 정수 벡터 min/max | 없음 | 비교 + 선택 두 명령 (§6.3, §8.3) |
| FCMA (복소 곱셈-누산) | 있음 | complex 곱셈을 `vcmlaq_f64`로 (§5.2) |
| SME | 있음 | 직접 쓰지 않는다. streaming 모드 전환 비용과 복잡도 때문이다. BLAS 안에서만 간접적으로 쓰인다 |
| SVE | 없음 | — |

- `-mcpu=native`는 `apple-m4`로 해석된다. 런타임 CPU 디스패치는 하지 않는다 (§1.4).

### 1.2 범위
- **있는 것**
  - ndarray: 버퍼, shape, strides, 시작 위치(`data` 포인터), dtype, base, 연속성 플래그, writeable
  - 생성 함수, 뷰 연산, 브로드캐스팅, `concat`/`stack`
  - 인덱싱: 기본 인덱싱(int, slice, `...`, `newaxis`)은 뷰, 고급 인덱싱(정수 배열, bool 마스크)은 복사본. 대입도 지원한다.
  - ufunc: 산술, 비교, 논리, 비트, 수학 함수, `maximum`/`minimum`, `where`, `clip`, `out=`, in-place 연산
  - 리덕션: `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `argmin`, `argmax`, `any`, `all`, `cumulative_sum`, `cumulative_prod`
  - 캐스팅과 type promotion, `repr`
  - `matmul`, `sort`/`argsort`, `unique_*`
- **없는 것**
  - 0차원 배열 (§2.2), `nonzero`, `dot`
  - 문자열/object/datetime/structured dtype, masked array, memmap, polynomial
  - 서브클래싱, `__array_function__`

### 1.3 빌드와 컴파일 옵션
`CMakeLists.txt`의 `_core` 타깃에 건다. 모두 결과의 비트나 속도에 영향을 주므로 함부로 바꾸지 않는다.

| 옵션 | 이유 |
|---|---|
| `NOMINSIZE` (`-O3` 유지) | nanobind는 기본으로 `-Os`를 `-O3` 뒤에 붙인다. `-Os`에서는 연속 메모리 루프가 거의 벡터화되지 않는다. 연속 fast path(§5.1)와 짝을 이뤄야 효과가 난다: 같은 fast path 코드를 `-Os`로 빌드하면 `-O3`보다 약 2배 느리다 |
| `-mcpu=native` | 대상은 이 Mac 하나다 |
| `-ffp-contract=off` | 이것이 없으면 clang이 `a + b*c`를 FMA 하나로 합친다. 반올림이 한 번 줄어 결과가 소스 코드 및 numpy와 달라진다. FMA가 필요한 곳은 코드에 `std::fma`로 적는다 (§5.3) |
| `-fno-builtin-sin -fno-builtin-cos` | 이것이 없으면 clang이 같은 인자의 `sin`/`cos`를 `__sincos_stret` 호출 하나로 합친다. 결과가 numpy(따로 호출)와 1 ULP 달라질 수 있다. complex `expm1`에서 확인했다 |
| `-ffast-math` 쓰지 않음 | NaN/inf가 없다고 가정하고, 재배치를 컴파일러에 맡긴다. 재배치는 코드에서 명시적으로 한다 (§6) |
| `ACCELERATE_NEW_LAPACK`, `ACCELERATE_LAPACK_ILP64` | Accelerate의 새 인터페이스, 64비트 정수 인자 (§9.2) |
| `-Wall -Wextra -Wpedantic` | — |

- 확장 모듈은 import 시험 때 `cblas_ddot`을 한 번 불러, 링크나 ILP64 설정이 틀리면 바로 드러나게 한다 (`bind/module.cpp`의 `_build_info`).

### 1.4 넣지 않은 최적화
- **임시 배열 제거(temporary elision)**: 안전하게 만들 수 없어서 넣지 않는다.
  - `PyUnstable_Object_IsUniqueReferencedTemporary`는 Python 코드에서는 임시 객체를 정확히 판정한다 (인자를 `nb::handle`로 받거나 raw `nb_add` 슬롯을 쓸 때. `nb::object`로 받으면 참조가 하나 늘어 한 번도 잡지 못한다).
  - 그러나 다른 C 확장이 빌린 참조로 받은 `x`로 `x + y`를 계산한 뒤 `x`를 다시 읽으면, API가 `x`를 임시 객체로 판정해 버퍼를 재사용한다. 다시 읽은 `x`는 덮어써져 있다. Cython 코드에서 흔한 패턴이다.
  - 막으려면 numpy처럼 C 콜스택을 검사해야 한다. "C 콜스택을 뒤지는 수준의 해킹이 필요하면 넣지 않는다"는 규칙에 따라 제외한다. 현재 프레임의 바이트코드가 `BINARY_OP`인지 보는 방법도 인터프리터 내부에 의존하고, 반사 연산(`__radd__`) 경로에 구멍이 남아 택하지 않는다.
  - 그래서 연산자는 항상 새 출력 배열을 할당한다.
- 런타임 CPU 디스패치, 손으로 짠 어셈블리
- ULP 단위로 튜닝한 SIMD 초월함수 (vForce를 쓴다, §5.4)
- **데이터를 보고 알고리즘을 고르는 휴리스틱**. dtype과 길이로 고르는 것(int64 기수 정렬, §8.4)은 여기에 해당하지 않는다.
- 작은 메모리 할당 캐시, 특수 케이스용 fast path 남발
- SME 직접 사용 (§1.1)

### 1.5 층 사이의 의존
- 배열 영역의 C++은 `csrc/base/`(Buffer, dtype, pairwise 합), `csrc/parallel/`(스레드풀), `csrc/accelerate/`(BLAS, vForce 래퍼), `csrc/array/`(ndarray, 이터레이터, 커널, 연산, 병렬화 기준값), `csrc/bind/`(nanobind)로 나뉜다.
- **아래 층은 위 층을 모른다.** `base/`, `parallel/`, `accelerate/`는 `array/`를 include하지 않는다.
  - NEON 커널(`array/kernels/neon.hpp`)과 "어느 연산이 vForce로 가는가"의 표(`array/kernels/vforce.hpp`)는 배열의 연산 정의(`ops::Maximum`, `ops::Exp` …)에 묶여 있어서 배열 영역에 둔다. `accelerate/vforce.hpp`에는 연산과 무관한 부분(함수 타입, `vvpow`/`vvatan2`의 인자 순서를 맞추는 래퍼, strided 실행기)만 있다.

---

## 2. 의미론 규칙

### 2.1 dtype과 승격
- dtype은 넷이다: `bool`, `int64`, `float64`, `complex128`.
- 승격 순서는 일직선이다: `bool < int64 < float64 < complex128`. 결과 dtype은 두 피연산자 중 높은 쪽이다 (`promote(a, b) = max(a, b)`, `base/dtype.hpp`).
- Python 스칼라도 같은 규칙을 따른다: `bool`→bool, `int`→int64, `float`→float64, `complex`→complex128. numpy의 "약한 스칼라" 같은 예외는 없다.
- **루프 선택** (`array/ops/ufunc.cpp`의 `select`): 승격한 dtype부터 위로 올라가며 그 연산의 루프가 있는 첫 dtype을 쓴다 (numpy와 같은 규칙). 아래 규칙은 이 표에서 저절로 나온다.
  - **bool 산술은 int64다.** 산술 연산에는 bool 루프가 없다. 그래서 `True + True == 2`, `bool @ bool`은 int64다. `sum`/`prod`/누적 연산도 bool을 int64로 쌓는다.
  - 비교는 모든 dtype이 자기 루프를 갖는다 (bool 포함: 비교는 산술이 아니다).
  - 논리 연산(`logical_*`)은 bool만 받는다. int64에는 루프가 없어 TypeError다 (Array API).
  - 비트 연산은 int64와 bool. 시프트는 int64만이고 bool 입력은 명시적으로 막는다 (`no_bool_inputs`).
  - `divide`는 int64 루프가 없다. 정수는 float64로 나눈다. 초월함수와 `sqrt`도 정수 루프가 없어 float64로 계산한다.
  - 반올림(`floor`/`ceil`/`trunc`/`round`)의 정수 루프는 항등 함수이고 dtype을 유지한다. complex는 `round`만 있다(실수부, 허수부 따로).
- int와 float의 비교는 승격 규칙대로 float64 루프에서 한다: int를 float64로 바꾼 뒤 비교한다 (numpy와 같다). frame도 같은 규칙이다 (DESIGN §9.1 3번).

### 2.2 0차원 배열 없음
- 결과가 암묵적으로 0차원이 되면 Python 스칼라를 반환한다.
  - 예: `a[1, 2]`, `sum(a)`, 1차원 배열의 `sum(axis=0)`, 스칼라만 넣은 ufunc, 벡터 `@` 벡터
  - int 계산은 int64 의미론(wrap)을 따른 다음 Python int가 된다.
- 0차원을 명시적으로 요청하면 ValueError다: `asarray(5)`, `zeros(())`, `reshape(())`, `broadcast_to(x, ())`. `checked_size`가 한곳에서 막는다.
- Python 스칼라로 꺼낸 다음부터는 Python 의미론이다. 오버플로가 없다.
- int64 범위를 벗어난 Python int는 **int64로 변환되어야 할 때만** OverflowError다.
  - `int_arr + 2**70`은 에러, `float_arr + 2**70`은 float로 바뀌어 정상이다.
  - ufunc에서는 **승격한 피연산자 dtype**으로 판정한다 (`check_int_scalars`). 그것이 bool이나 int64이면 루프가 float64여도(`divide`, `atan2`, `sqrt`) 에러다. 루프 dtype으로 판정하던 때는 `divide(int_arr, 2**64)`만 정상이라 다른 이항 함수와 어긋났다 (DESIGN §9.5). frame도 같은 규칙이다.
  - 이름 붙은 int 인자(`shape`, `axis`, `k`, `num`)도 범위 밖이면 OverflowError다 (`parse_int`, API.md §2). `set_num_threads`만 범위 1~1024 밖을 모두 ValueError로 낸다.
- 배열은 크기와 상관없이 스칼라가 아니다. `bool()`, `int()`, `float()`, `complex()`는 원소가 하나여도 TypeError다 (§12.2).

### 2.3 원소 연산은 값 때문에 실패하지 않는다
- 에러는 타입이나 shape이 맞지 않을 때만 낸다. 각 타입은 자기 타입의 표준 산술을 따른다.
- **정수**: 조용히 wrap한다. C++의 부호 있는 오버플로는 정의되지 않은 동작이라 부호 없는 정수로 계산한 뒤 되돌린다.
  - `INT_MIN // -1 == INT_MIN`, `abs(INT_MIN) == INT_MIN`
  - `a // 0 == 0`, `a % 0 == a`. 그래서 `a == (a // b) * b + a % b`가 b가 0일 때도 성립한다.
  - `//`는 바닥 나눗셈, `%`의 부호는 b를 따른다 (Python과 같다).
  - 거듭제곱은 제곱을 반복하는 방식으로 계산하고 wrap한다. 음수 지수는 정수 결과를 낸다: 밑이 1이면 1, -1이면 ±1, 그 외는 0. `0 ** -n`은 0으로 나누기 규칙을 따라 0이다.
  - 시프트 횟수는 부호 없는 값으로 읽는다. 64 이상(음수 포함)이면 모든 비트가 빠진다: 왼쪽은 0, 오른쪽은 산술 시프트라 음수면 -1.
- **float와 complex**: IEEE 754를 따른다 (`inf`, `nan`).
- **경고는 일절 없다.** `errstate`도 없다.

### 2.4 명시적 변환
`astype`, 생성 함수의 `dtype=` 인자, 버퍼 가져오기가 같은 규칙 하나를 쓴다 (`array/convert.hpp`의 `cast_value`, `check_castable`).
- `dtype=`을 주는 것은 명시적 변환이다. 손실이 있어도 `astype`과 같이 변환한다. 예: `asarray([1.5], dtype=int64)`는 `[1]`이다.
- **X → bool**: `x != 0`. 모든 dtype에서 허용한다. complex도 허용하고, nan은 True다.
- **float → int64**: 0 방향으로 자르고, 범위를 벗어나면 포화시키고, nan은 0이다.
  - ARM의 `FCVTZS` 명령 하나가 정확히 이 일을 한다. 그래서 `vcvtd_s64_f64`를 직접 부른다 (`saturate_to_int64`). 컴파일러가 이 루프를 벡터 `FCVTZS`로 바꾸고, 연속 캐스트 루프는 `vcvtq_s64_f64`를 직접 쓴다 (`array/kernels/cast.hpp`).
  - C++의 평범한 캐스트는 nan, inf, 범위 밖 값에서 정의되지 않은 동작이다. 규칙을 분기 세 개로 적으면 벡터화되지 않는다 (`astype` f→i가 464µs, `FCVTZS`로 89µs, numpy 243µs).
- **complex → float/int**: TypeError다. 허수부를 잃기 때문이다. 실수부가 필요하면 `real(x)`를 쓴다. 이 검사는 dtype 단계에서 하고, 원소 단계의 변환은 실패하지 않는다.
- 범위 밖의 Python int는 int64로 변환될 때만 OverflowError다 (§2.2). 변환 대상이 float/complex이면 float로 바꾼다: `asarray([2**70, 0.5])`는 정상이다.

### 2.5 암묵적 변환은 위로만
- 대입(`x[...] = v`), `out=`, in-place 연산자(`x += y`), `dtype=` 없는 `full_like`의 채움 값은 승격 순서의 **위로만** 변환한다. 아래로 가면 TypeError다 (`check_assignable`, `array/ops/copy.hpp`). `full_like(int_arr, 1.5)`는 `int_arr[...] = 1.5`와 같은 일이다. `dtype=`을 주면 `full`처럼 명시적 변환이다 (§2.4).
  - 예: `int_arr += 1.5`는 TypeError, `float_arr[0] = 1`은 정상이다.
- `out=`가 결과보다 넓은 dtype이면 결과 dtype으로 계산한 뒤 `out`으로 변환한다 (§4.3).
- `clip`은 x의 dtype을 유지한다. `min`/`max`는 x의 dtype으로 암묵적으로 변환되어야 한다.
- 쓰기 금지 배열에 쓰면 ValueError다.

### 2.6 순서, NaN, 같음
- **complex에는 순서가 없다.** `<` 등의 비교, `max`/`min`, `sort`/`argsort`, `argmax`/`argmin`, `clip`은 complex에서 TypeError다. 해당 연산에는 complex 루프가 없다.
- `max`, `min`, `maximum`, `minimum`, `clip`은 NaN을 전파한다. `maximum`/`minimum`은 `-0 < +0`으로 본다 (IEEE 754-2019 maximum/minimum).
- `argmax`/`argmin`은 첫 NaN의 위치를 반환한다. 동점이면 첫 인덱스다.
- **NaN은 가장 큰 값이다.** `sort`는 오름차순에서 맨 끝, 내림차순에서 맨 앞에 놓는다 (§8.1). frame의 정렬과 같다 (DESIGN §9.1 9번).
- `unique_*`는 NaN을 모두 하나의 값으로 본다 (frame의 키 규칙, numpy 2 `np.unique`의 기본값. Array API와는 다르다: DESIGN §9.1 10번). `-0.0`과 `0.0`도 하나다. 보여 주는 값은 처음 나온 것이다. complex는 (실수부, 허수부)의 사전순으로 비교하고, NaN 부분이 있으면 NaN이다.
- `==`, `!=`는 받지 않는 피연산자(리스트, numpy 배열, `None`)에 TypeError다 (§12.2). 다른 연산자와 같다 (DESIGN §9.1 13번).

### 2.7 numpy와의 비트 동일성
무엇이 numpy와 비트까지 같고, 무엇이 다른지 정해 두었다. 테스트가 이 구분을 그대로 고정한다 (§13.1).

| 범주 | numpy와의 관계 |
|---|---|
| 원소별 연산 (vForce 함수 제외) | 비트까지 같다. 이 Mac의 numpy float64 초월함수는 libm과 비트가 같다(`tanh`만 최대 1 ULP) |
| vForce를 쓰는 float64 함수 16개 (§5.4) | 최대 8 ULP (계약). 측정한 최대는 4 ULP (`atanh`) |
| 출력이 하나인 긴 합, 형변환 없음 (`sum(f)`) | 비트까지 같다 (§6.2) |
| 순서와 무관한 리덕션 (정수 합·곱, max/min, any/all, argmax) | 같다 |
| 블록으로 나누는 리덕션, float/complex `prod`, 긴 float 누적 스캔, 형변환이 있는 리덕션 | 다를 수 있다 (§6.9) |
| `arange`의 float 값 | `start + i*step`으로 계산한다. numpy는 step을 `(start+step) - start`로 다시 구하므로 몇 ULP 다를 수 있다 |
| `repr` | 같을 필요가 없다 |

- 어느 경우든 결과는 **결정적**이다. 같은 데이터, 같은 레이아웃이면 스레드 수와 무관하게 같은 비트가 나온다 (§10.5).

### 2.8 API 규칙
기준은 Python Array API Standard다. 함수별 시그니처는 `docs/api/array.md`에 있다.
- 이름: Array API의 이름을 쓴다 (`acos`, `concat`, `pow`, `permute_dims`). Array API에 없는 함수만 numpy 이름을 쓴다 (`ravel`).
- 모듈 함수가 정식 API다. 정해진 목록(`reshape`, `astype`, `copy`, 리덕션 등)만 메서드로도 제공한다. 메서드는 같은 C++ 함수를 x를 self로 해서 부른다.
- 배열 인자와 그 함수의 "본 인자"(`reshape`의 `shape`, `full`의 `fill_value`)는 positional-only, 이름 붙은 옵션(`dtype`, `axis`, `keepdims`, `out`, `copy`, …)은 keyword-only다.
- shape은 int 또는 tuple, axes는 int, tuple 또는 리스트 인자 하나로 받는다 (`bind/pyutil.hpp`의 `parse_shape`, `parse_axes`). 리스트를 받는 것은 linalg와 맞춘 것이다 (DESIGN §7). 반복된 축은 ValueError다.
- 함수 하나에 이름 하나. 별칭이 없다. `flatten`은 없고 `ravel`만, `dot`은 없고 `matmul`과 `@`만 있다.
- 기본 axis: 리덕션은 `axis=None`, `sort`/`argsort`는 `axis=-1`, 누적 연산은 ndim>1이면 axis가 필수다.
- 모든 리덕션은 int/tuple axis와 `keepdims`를 받는다. `argmin`/`argmax`는 int axis만 받는다.
- `var`/`std`의 자유도 인자는 `correction`이다 (기본 0).
- `sort`는 `stable: bool`(기본 True), `descending: bool`을 받는다.
- `unique`는 반환 형태별로 나눈다: `unique_values`, `unique_counts`, `unique_inverse`, `unique_all`. 여러 값은 namedtuple로 돌려준다 (`src/stardust/_results.py`).
- `copy: bool | None`: True는 항상 복사, False는 복사가 필요하면 에러, None은 필요할 때만 복사.
- 전치: `.T`는 2차원에서만, `.mT`는 마지막 두 축, 일반 축 순서는 `permute_dims`.
- `where`는 인자 3개 형태만 있다.
- dtype 지정은 dtype 객체(`sd.bool`, `sd.int64`, `sd.float64`, `sd.complex128`)와 그 이름 문자열(`"float64"`)을 받는다 (API.md §1, DESIGN §9.2). Python 타입, 다른 이름, 배열에 없는 `sd.string`은 TypeError다 (`bind/pyutil.cpp`의 `parse_any_dtype`, 이름은 `base/dtype.hpp`의 `dtype_from_name`).
- ufunc의 피연산자는 ndarray와 Python 스칼라(numpy 스칼라 포함, §12.2)뿐이다. 리스트나 다른 라이브러리의 배열은 `asarray`로 먼저 바꾼다 (`is_operand`). `real`, `imag`도 `conj`처럼 스칼라를 받는다.

---

## 3. 데이터 모델

### 3.1 DType (`base/dtype.hpp`)
- `enum class DType : uint8_t { Bool, Int64, Float64, Complex128, String }`. 배열은 앞의 넷(`kNumArrayDTypes`, `is_array_dtype`)을 쓰고, 그 값의 크기 순서가 승격 순서다. `String`은 frame의 컬럼용이다 (DESIGN §4.2).
- 저장 타입: `bool`, `int64_t`, `double`, `std::complex<double>`.
- `visit_dtype(d, f)`가 런타임 dtype을 컴파일 시점 타입으로 바꾼다. 템플릿 함수 하나가 dtype마다 한 번씩 인스턴스화된다. 배열 dtype만 받는다 (`String`을 넘기지 않는 것은 호출하는 쪽의 몫이다).
- Python 쪽 dtype 객체는 C++가 만든 싱글턴 넷이다. 그래서 dtype 비교는 동일성 비교다 (`bind/pyutil.hpp`).

### 3.2 Buffer (`base/buffer.hpp`)
참조 카운트로 수명을 관리하는 메모리 블록이다. 여러 배열(뷰)이 한 Buffer를 공유한다.
- **정렬**: 할당 시작을 128B(캐시 라인)로 맞춘다. SIMD 루프의 첫 load가 두 줄에 걸치지 않는다. `aligned_alloc`에는 calloc판이 없으므로, 128B를 더 할당하고 시작을 올림한다.
- **0 채우기**: `zeros`는 `calloc`을 쓴다. 큰 할당은 OS가 0 페이지를 첫 접근 때 채우므로 `zeros((10000, 10000))`은 쓰기 전까지 거의 비용이 없다.
- **작은 배열은 Buffer 안에**: 256B(`kInline`) 이하의 데이터는 Buffer 객체 안의 배열에 둔다. Buffer는 `make_shared`로 만들어 참조 카운트와 함께 할당된다.
  - 그래서 작은 배열 하나의 할당이 세 번(Buffer, 데이터, `shared_ptr` 제어 블록)에서 한 번이 된다 (§11).
  - 대가로 모든 Buffer 객체가 256B 커진다. 큰 배열에서는 무시할 만하다.
- **외부 메모리** (`adopt`): 버퍼 프로토콜로 가져온 메모리다. `Py_buffer`를 쥐고 있다가 소멸할 때 release한다. 마지막 소유자는 GIL이 없는 곳(워커 스레드)에서 사라질 수 있으므로 소멸자가 GIL을 잡는다.
- `size()`: 데이터의 바이트 수(할당한 크기, 또는 가져온 버퍼의 `len`). frame의 Column이 범위를 확인할 때 쓴다.

### 3.3 NDArray (`array/ndarray.hpp`)
배열은 Buffer를 읽는 방법에 대한 **설명**이다. 원소 `(i0, …, ik)`는 `data + Σ i_d·strides[d]`에 있다 (strides는 바이트 단위). 전치, 슬라이스, 브로드캐스트는 이 설명만 바꾸고 메모리는 건드리지 않는다.

```
buffer        shared_ptr<Buffer>   메모리 소유
data          std::byte*           Buffer 안의 시작 위치 (별도 offset 필드가 없다)
dtype, ndim
shape[32], strides[32]             객체 안의 고정 배열
c_contiguous, f_contiguous, writeable
base          nb::object           Python의 .base
```
- **차원 수 상한은 32다** (`kMaxNdim`). shape과 strides를 객체 안에 고정 배열로 둔다. 배열을 만들 때 shape용 힙 할당이 없다. 32를 넘으면 ValueError다.
- **복사와 이동은 앞의 ndim칸만 옮긴다.** 고정 배열 전체(512B)를 복사하면 작은 연산의 비용이 눈에 띄게 늘어난다 (§11). 같은 이유로 이터레이터의 고정 배열은 0으로 채우지 않는다.
- **`base`**: 메모리를 실제로 소유한 객체를 가리킨다. 뷰의 뷰를 만들어도 중간 뷰를 거치지 않고 최종 소유자로 접는다 (`make_view`, `bind/pyutil.hpp`). numpy 2.x도 같다. 배열이 메모리를 직접 소유하면 `None`, 버퍼 프로토콜로 가져온 배열이면 내보낸 객체다.
- **연속성 플래그**: shape과 strides가 바뀔 때마다 `update_flags()`로 다시 계산한다. 크기 1인 축은 stride를 보지 않고, 빈 배열은 양쪽 모두 연속이다 (numpy의 relaxed strides 규칙).
- **새 배열의 레이아웃**: `new_array`는 C 순서다. `new_array_like`는 원본 strides의 절댓값 내림차순으로 축 순서를 정한다(numpy의 `order='K'`). F 순서 원본이면 F 순서 결과가 나온다. 축 정렬은 insertion sort다: 최대 32축이고 `std::stable_sort`는 호출마다 임시 버퍼를 할당한다.
- 바이트 크기는 `checked_size`가 오버플로를 검사한다.

### 3.4 뷰 (`array/ops/shape.cpp`, `array/ops/indexing.cpp`)
- 기본 인덱싱, `reshape`(strides로 표현될 때), `permute_dims`, `.T`, `.mT`, `broadcast_to`, `broadcast_arrays`, `real`/`imag`(complex의 부분 뷰)은 모두 뷰다. 1GB 배열의 전치도 2×2의 전치와 비용이 같다.
- 복사하는 것: strides로 표현할 수 없는 `reshape`(`copy` 규칙에 따라), `concat`, `stack`, 고급 인덱싱.
- **쓰기 금지 뷰**: `broadcast_to`와 `broadcast_arrays`의 결과는 `writeable=False`다. 여러 원소가 같은 메모리를 공유하므로 쓰기를 허용하면 위험하다 (numpy도 같다). 읽기 전용으로 가져온 외부 버퍼도 쓰기 금지다.
- `*_like` 함수는 원본의 메모리 축 순서를 따른다 (`new_array_like`).

### 3.5 메모리 겹침
`a[1:] += a[:-1]`처럼 입력과 출력이 메모리를 일부 공유하는 경우다.
- **보장**: 결과는 "모든 입력을 다 읽은 다음 출력을 쓴 것"과 같다 (numpy와 같은 의미론).
- **판정**: 두 배열이 닿을 수 있는 바이트 범위 `[최소, 최대]`가 겹치는지만 보는 보수적 검사다 (`may_share_memory`).
- **겹치면** 겹치는 입력을 임시 배열로 복사한 뒤 연산한다.
- **예외**: 입력과 출력의 data, dtype, shape, strides가 모두 같으면(`same_view`) 복사하지 않는다. 원소별 연산은 각 원소를 읽은 직후 같은 자리에 쓰므로 안전하다 (`a += a`).
- **적용 대상**: 원소별 연산의 `out=`, in-place 연산자, 인덱스 대입. 인덱스 대입에서는 값뿐 아니라 **인덱스 배열과 마스크**도 대상과 겹치면 복사한다. 그러지 않으면 `a[a] = 2`에서 앞의 쓰기가 뒤에서 읽을 인덱스를 바꾸고, 범위 검사를 이미 통과한 인덱스가 범위 밖 값으로 바뀌어 메모리를 넘어 쓸 수 있다.

---

## 4. 연산 엔진

### 4.1 커널 형태 (`array/iter.hpp`)
```cpp
using Loop = void (*)(std::byte* const* args, std::int64_t n, const std::int64_t* strides, void* ctx);
```
- 커널 하나는 1차원 strided 루프다. numpy의 inner loop와 같다. 입력이 앞, 출력이 뒤다.
- 원소별 연산의 의미는 `array/kernels/scalar_ops.hpp`에 연산마다 구조체 하나로 정의한다: 어느 저장 타입에 루프가 있는지(`ok<T>`)와 원소 하나의 함수(`f`).
- `array/kernels/elementwise.hpp`가 이것을 루프로 바꾸고, 연산마다 `table[dtype]`(루프와 출력 dtype)을 템플릿 인스턴스화로 채운다.
- 캐스트도 같은 형태의 `cast[from][to]` 표다 (16개, `array/kernels/cast.hpp`).
- **Python 스칼라 피연산자**는 루프 dtype으로 미리 변환한 원소 1개짜리 배열이 되고, 이터레이터가 stride 0으로 브로드캐스트한다.
- 피연산자는 최대 4개다 (`kMaxOperands`, `where(cond, x1, x2) → out`).

### 4.2 nd 이터레이터 (`array/iter.cpp`의 `NdIter`)
numpy `nditer`의 축소판이다. N차원 문제를 되도록 적고, 길고, 메모리를 따라가는 1차원 구간으로 바꾼다.
1. **브로드캐스팅**: 공통 shape에 맞춘다. 브로드캐스트되는 축의 stride는 0이다. shape은 고정 배열로 구해 할당이 없다 (`broadcast_into`).
2. **크기 1인 축 제거**
3. **축 재정렬**: stride 절댓값이 큰 축을 바깥에 둔다. 마지막 피연산자(관례상 출력)부터 비교해 쓰기가 순차적이 되게 하고, 거기서 순서가 정해지지 않으면 앞의 피연산자를 거꾸로 차례로 본다. stride 0은 정보가 없으므로 건너뛴다. 같으면 원래 순서를 유지하므로 C 연속 배열은 그대로다. 전치된 배열은 메모리 순서로 읽힌다.
4. **차원 합치기(coalescing)**: 모든 피연산자에서 "바깥 축 한 칸 = 안쪽 축 extent칸"이면 두 축을 합친다. 연속 `(3, 4, 5)`는 길이 60인 구간 하나가 된다.
5. **실행**: 바깥 축은 odometer로 돌고, 가장 안쪽 구간마다 커널을 한 번 부른다. 일이 충분하면 구간을 스레드풀에 나눈다 (§10.4).
- 모드는 셋이다: `Naive`(브로드캐스트만), `Reorder`, `Full`. 각 단계의 효과를 따로 재기 위해 남겨 두었고, 실제 연산은 `Full`을 쓴다. 예외: 인덱스 하나짜리 scatter는 반복된 인덱스에서 마지막 쓰기가 이기도록 `Naive`(C 순서, 1스레드)로 돈다 (§7.3).
- **출력 할당**: 원소별 연산의 새 출력은 이미 전체 shape인 첫 배열 피연산자의 레이아웃을 따른다 (`new_result`). F 순서 입력이면 F 순서 결과가 나와 coalescing이 유지된다. 그런 피연산자가 없으면 C 순서다.

### 4.3 버퍼 단위 캐스팅 (`array/kernels/buffered.hpp`)
피연산자의 dtype이 루프 dtype과 다를 때다 (`int64 + float64`는 float64 루프).
- 입력을 4096원소(`kBufferElements`)씩 작은 버퍼로 캐스트하고 바로 커널을 돌린다. 메모리 왕복은 입력과 출력의 것뿐이다. 입력 전체를 임시 배열로 바꾸면 쓰고 다시 읽느라 메모리를 두 번 더 지난다.
  - 효과: `i + f` 1.37배, `sum(bool)` 1.38배, `mean(int64)` 1.21배. `f * i + g`는 1스레드 397µs, numpy 635µs다.
- **버퍼 크기 4096**: 피연산자 3개 × 32KB가 P코어 L1(128KB)에 들어간다. 256~65536을 쟀다: 1024~4096이 가장 빨랐고, 16384 이상은 L1을 넘쳐 10~25% 느렸다.
- `BufferedLoop`는 커널 시그니처를 그대로 가진 래퍼다. 이터레이터는 그 존재를 모른다.
- 버퍼는 스레드마다 하나다 (`thread_local`). ctx는 루프를 도는 모든 스레드가 공유하므로 버퍼를 ctx에 둘 수 없다.
- **출력은 버퍼링하지 않는다.** `out=`가 결과보다 넓은 dtype이면 결과 dtype의 새 배열에 계산한 뒤 `assign`으로 `out`에 변환해 넣는다.
- 리덕션도 같은 래퍼로 입력을 변환한다 (`run_fold`). 그래서 형변환이 있는 리덕션은 4096원소 단위로 접힌다 (§6.9).

### 4.4 ufunc 디스패치 (`array/ops/ufunc.cpp`)
모든 호출이 네 단계를 거친다.
1. **피연산자**: ndarray 또는 Python bool/int/float/complex
2. **루프**: dtype을 승격한 뒤 루프가 있는 첫 dtype을 고른다 (§2.1)
3. **변환**: dtype이 다른 입력은 실행 중에 버퍼 단위로 변환한다 (§4.3)
4. **실행**: 입력 + 출력을 `NdIter`로 돈다. 출력은 `out=`(§2.5, §3.5) 또는 새 배열(§4.2)
- 배열 피연산자가 하나도 없으면 결과는 Python 스칼라다 (§2.2).
- 형태 배열(shape, 피연산자 목록)은 고정 크기 배열로 받는다. 호출마다 할당이 없다.

### 4.5 레인 처리 (`array/ops/lanes.hpp`)
- 축 하나의 원소를 **인덱스 순서**로 봐야 하는 연산이 있다: `argmin`/`argmax`(동점은 첫 위치), 누적 스캔, 정렬.
- 이터레이터가 그 축을 뺀 나머지 축을 (메모리 순서로) 돌고, 방문한 원소마다 레인 하나를 0부터 n-1까지 처리한다.
- 커널 호출 하나가 레인 하나를 통째로 처리하므로 레인 길이를 작업량 가중치로 넘긴다. 이것이 없으면 행이 2000개인 정렬도 일이 "2000"으로 세어져 병렬로 돌지 않는다.

---

## 5. 원소별 커널

### 5.1 연속 fast path (`array/kernels/elementwise.hpp`)
- 루프는 먼저 흔한 레이아웃을 검사한다.
  - 모두 연속: `o[i] = f(a[i], b[i])`
  - 한쪽이 스칼라(stride 0): `x + 1.0`, `1.0 - x`
- 이 경우 바이트 stride 대신 인덱스로 접근하는 루프를 쓴다. 같은 `Op::f`지만 컴파일러가 NEON 레지스터로 묶을 수 있다. 원소마다 바이트 stride를 더하는 루프는 어떤 최적화 수준에서도 벡터화되지 않는다.
- 나머지 레이아웃은 일반 strided 루프로 간다.
- 효과: 대부분의 원소별 연산이 1.4~3배 빨라져 numpy와 같아졌다. bool `&`는 레지스터 하나에 16개가 들어가 17배다.
- `sqrt`, `floor`/`ceil`/`trunc`/`round`, `abs`, `square`, `sign`은 손으로 짜지 않는다. 연속 루프를 컴파일러가 NEON으로 벡터화한다 (`fsqrt.2d`, `frintm.2d` 등이 바이너리에 있음을 확인했다).
- 같은 dtype의 연속 복사(캐스트 표의 대각선)는 `memcpy`다. OS가 칩에 맞춰 튜닝한 함수다 (`x.copy()` 118 → 94µs, numpy와 같음).

### 5.2 손으로 짠 NEON (`array/kernels/neon.hpp`)
컴파일러의 벡터화가 부족한 곳에만 둔다. 모든 커널은 스칼라 정의(`scalar_ops.hpp`)와 비트가 같아야 하고, 테스트가 세 레이아웃(연속, 스칼라 브로드캐스트, strided)을 모두 비트로 비교한다.
- **float64 `maximum`/`minimum`**: `FMAX`/`FMIN` 명령 하나가 IEEE 754-2019 maximum/minimum이다(NaN 전파, `-0 < +0`). 스칼라 정의는 분기 셋(NaN, NaN, ±0)이라 자동 벡터화하면 비교와 선택이 늘어 오히려 느려진다(0.66배). 명령 하나로 3.9배, 캐시 안에서는 원소당 0.77 → 0.16ns다. 레지스터 두 개씩 처리해 load를 더 띄운다. 한쪽이 스칼라인 경우(`maximum(x, 0.0)`)도 같은 커널이다.
- **`clip`**: `minimum(maximum(x, lo), hi)`를 한 패스로 한다. `FMAX` 다음 `FMIN`이라 의미가 정확히 같다. 두 경계가 모두 스칼라인 흔한 경우는 splat을 루프 밖에서 한 번만 한다. 1스레드 143µs, numpy 230µs (원래의 범용 루프는 910µs).
- **complex 곱셈**: `FCMLA` 두 번. numpy 바이너리의 비트는 `re = fma(ar, br, -(ai*bi))`, `im = fma(ar, bi, ai*br)`다. FCMLA를 **rot 90 → rot 0** 순서로 쓰면 정확히 이 순서가 된다. 누산기는 `+0.0`이 아니라 `-0.0`에서 시작한다: `-0 + x == x`가 부호 있는 0까지 성립하므로 첫 단계의 결과가 반올림된 곱 그 자체다. 원소 100만 개에서는 메모리에 묶여 1.05배지만, 캐시 안(4096개)에서는 0.47 → 0.33ns다.
- 그 밖의 NEON: float→int64 변환 (`FCVTZS`, §2.4), 리덕션의 max/min/argmax (§6.3), 정렬 네트워크 (§8.3).

### 5.3 numpy와 비트를 맞추는 방법 (`array/kernels/scalar_ops.hpp`)
- numpy가 식을 적어 둔 연산은 같은 식을 같은 순서로 쓴다: complex 곱셈과 나눗셈(Smith 알고리즘), float의 `floor_divide`(`npy_divmod`: `fmod` 후 부호 보정), complex `pow`(작은 정수 지수는 반복 곱셈, 나머지는 `cpow`).
- `std::complex`의 연산자는 C99 Annex G의 inf/nan 복구를 더하는데 numpy에는 없다. 그래서 쓰지 않고 식을 직접 적는다.
- **명시적 FMA**: numpy 바이너리는 컴파일러 기본 FMA 축약으로 빌드되어 있어 `a*b - c*d`가 `fma(a, b, -(c*d))`가 된다. 같은 자리에 `std::fma`를 명시한다: complex 곱셈, 나눗셈, 제곱, `expm1`. 무작위 곱 4000개로 패턴을 확인했다.
- **`|z|`**: numpy의 SIMD cabs 공식 `큰 쪽 × sqrt(1 + (작은 쪽/큰 쪽)²)`을 쓴다. inf 부분이 nan 부분을 이긴다. libm `hypot`과는 최대 1 ULP 다르다.
- complex 초월함수는 C 라이브러리의 C99 함수(`cexp` 등)를 스칼라로 부른다. libc++의 `std::exp(complex)`는 자체 구현이라 numpy와 다르다. C++에는 `<complex.h>`가 없어 직접 선언한다.
- 반올림 `round`는 `rint`(현재 모드 = 가장 가까운 짝수)다.

### 5.4 vForce (`array/kernels/vforce.hpp`, `accelerate/vforce.hpp`)
float64 초월함수는 함수마다 구현을 나눈다.
- **vForce를 쓰는 16개**: `exp` `log` `log2` `log10` `sin` `asin` `acos` `atan` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh`, 이항 `pow` `atan2`
- **libm을 쓰는 넷**: `expm1` `log1p` `cos` `tan`. vForce가 IEEE / C99 Annex F 의미론을 어긴다: `cos(±0) ≠ 1`, `expm1`/`log1p`/`tan`이 `-0`의 부호를 잃음, `expm1`이 서브노멀 입력을 0으로 만듦.
- **vForce를 써도 되는 근거** (독립 실험으로 확인)

  | 항목 | 결과 |
  |---|---|
  | 위치 독립성 (인덱스, 배열 길이, 정렬) | 약 24만 회 검사에서 불일치 0. 그래서 `exp(x)[i] == exp(x[i])`이고, 스레드 청크가 결과를 바꾸지 않는다 |
  | 스레드 안전성 | 14개 스레드 동시 호출에서 불일치 0 |
  | 내부 스레드 | 쓰지 않는다 (CPU 시간 / 경과 시간 = 1.00). 우리 풀과 충돌하지 않는다 |
  | 특수값 (±0, ±inf, NaN) | 16개 함수는 부호까지 정확하다 |
  | 정확도 (libm 대비) | 최대 1~6 ULP. numpy 대비 측정 최대 4 ULP (`atanh`) |

- **속도** (1스레드, libm 대비): 대부분 2~10배. `sin` 7.6배, `atan2` 10배, `exp` 4.2배.
  - `asinh`/`acosh`는 입력에 따라 갈린다. vForce는 입력과 상관없이 거의 일정하고, libm은 크게 달라진다. 작은 입력에서 vForce가 1.8~3.3배 빠르고, 큰 입력에서 libm이 10~20% 빠르다. 데이터를 보고 고르지 않으므로(§1.4) vForce를 쓴다.
- **정확도 계약**: vForce 함수는 numpy와 최대 8 ULP까지 다를 수 있다 (테스트 허용치). 특수값은 비트까지 같아야 한다.
- **호출 방식**: vForce는 연속 배열을 받는다.
  - 입력과 출력이 연속이면 그대로 넘긴다. vForce의 개수 인자는 `int`라서 한 번에 2²⁴개까지 넘긴다.
  - 비연속 입력, 스칼라(stride 0), 비연속 출력은 커널 안에서 1024원소(8KB) 스택 버퍼에 모으고 흩는다. 레이아웃과 상관없이 vForce 경로를 탄다.
  - int/bool 입력은 버퍼 캐스팅(§4.3)을 거쳐 float64로 넘긴다.
  - 스칼라 입력도 n=1로 같은 경로를 탄다. 그래야 `exp(x)[i] == exp(x[i])`가 성립한다.
- 어느 연산이 vForce로 가는지는 `array/kernels/vforce.hpp`의 표가, vForce 호출과 strided 처리는 `accelerate/vforce.hpp`가 맡는다 (§1.5).

---

## 6. 리덕션

### 6.1 구조 (`array/kernels/reduce.hpp`)
- 리덕션도 이터레이터 위에서 돈다. 피연산자는 둘이다: 입력, 그리고 줄이는 축의 extent가 1인 출력을 입력 shape으로 브로드캐스트한 것(줄이는 축의 stride가 0). 커널은 `out = f(out, x)`를 한다.
- 이터레이터가 넘겨 준 구간의 모양이 전략을 정한다.
  - **출력 stride 0 → 안쪽 축 리덕션**: 구간 하나를 누산기 하나로 접는다. 다중 누산기 + NEON + pairwise. 예: `sum(axis=-1)`
  - **출력 stride ≠ 0 → 바깥 축 리덕션**: 결과 행에 입력 행을 원소별로 더한다. 유지되는 축 방향으로 벡터화된다. 예: C 순서의 `sum(axis=0)`
- 시작값: 합은 `+0`(그래서 빈 합은 0, `sum([-0.0])`은 `+0.0`으로 numpy와 같다), 곱은 1, `max`는 dtype의 최솟값(`-inf`), `min`은 최댓값. 첫 원소가 항상 시작값을 대신하므로 `-inf`, `-0.0`이 정확히 남는다. `max`/`min`은 항등원이 없으므로 접을 원소가 없는 출력은 에러다.
- 결과: 모든 축을 줄이면 `keepdims=True`가 아닌 한 Python 스칼라다 (§2.2). keepdims 결과는 입력의 축 순서를 따라 할당해서, 이터레이터가 입력을 메모리 순서로 걷게 한다.
- 출력 dtype: `sum`/`prod`/누적은 bool을 int64로 쌓는다. `dtype=`은 명시적 변환이지만 bool은 안 된다(덧셈, 곱셈에 bool 루프가 없다). `mean`/`var`/`std`는 float64, complex 입력이면 complex128(`var`/`std`의 결과는 float64)이다.

### 6.2 pairwise 합
numpy의 `DOUBLE_pairwise_sum`/`CDOUBLE_pairwise_sum`과 같은 알고리즘, 같은 순서다.
- n < 8이면 순서대로 더한다. n ≤ 128(`kPairwiseBlock`)이면 누산기 8개가 독립적으로 굴러가고 `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`로 합친다. 그보다 크면 반(8의 배수로 내림)으로 나눠 재귀한다.
- complex는 numpy가 double 개수로 세므로 블록 하나가 complex 64개, 누산기는 4개다.
- **이유**
  - 속도: 덧셈 지연에 묶인 사슬 하나가 8개가 된다. 순차 합 대비 5.6배다.
  - 정확도: 오차가 n이 아니라 log n에 비례해 자란다. 원소 1e7개의 상대 오차가 3.7e-14에서 0이 된다.
  - 알고리즘과 순서가 같으니 결과가 numpy와 비트까지 같다.
- **병렬**: 재귀 트리는 n에만 의존한다. 4096개(`kLeaf = kParallelTask`, complex는 2048개) 이하인 부분 트리를 잎으로 삼아 병렬로 계산하고, 같은 트리를 따라 합친다 (`pairwise_parallel`). 덧셈 하나하나가 직렬 계산과 같은 순서이므로 **스레드 수와 상관없이 numpy와 비트가 같다.** 원소 300만 개에서 1스레드와 10스레드 모두 `sum(x) == np.sum(x)`를 테스트한다.
- `var`의 두 번째 패스(편차 제곱합)도 안쪽 축이면 같은 pairwise로 접는다.
- **정의는 `base/pairwise.hpp` 한 벌이다**: 트리 모양, 잎의 합, 병렬 계획(`plan`)과 합치기(`combine`). 배열의 `pairwise_sum`, `pairwise_parallel`(`array/kernels/reduce.hpp`)은 항을 접근자로 넘겨 그것을 부른다. linalg와 frame도 같은 정의를 쓴다 (docs/design/linalg.md §8.2, docs/design/frame.md §7.4).
  - 잎의 합(`pairwise::leaf`, `leaf_complex`)은 `[[gnu::always_inline]]`이다. 배열은 트리 → 람다 → 잎 → 접근자 람다의 여러 겹으로 부르는데, 잎이 인라인되지 않으면 원소마다 호출이 남아 1스레드 `sum(f)`(원소 100만 개)가 110µs에서 152µs로 느려졌다(잰 값).

### 6.3 max/min, argmax/argmin
- **float64 `max`/`min`**: NEON 레지스터 4개(레인 8개)로 `FMAX`/`FMIN`을 접고, 레인끼리 `vmaxvq`로 합친다. NaN 전파가 명령에 들어 있다. 순차 비교 대비 14배, numpy와 같다.
- **int64 `max`/`min`**: 64비트 정수 max/min 명령이 없어 비교(`vcgtq_s64`)와 선택(`vbslq_s64`) 두 단계이고 지연이 약 5사이클이다. 레지스터 4개로는 지연을 덮지 못해 전혀 빨라지지 않았고(113µs), **8개(레인 16개)** 로 65µs다(numpy 69µs). 누산기 수는 "명령 지연 × 사이클당 처리량"으로 정한다. float 덧셈과 같은 개수를 쓰면 안 된다.
- **argmax/argmin** (레인 단위, §4.5): float64 연속이면 두 레인이 각각 (최댓값, 위치)를 추적한다. 레인 안에서는 strict `>`로 첫 위치를 지키고, 두 레인이 동점이면 작은 인덱스를 고른다. NaN은 "본 적 있음"만 표시하고, 있으면 스칼라로 처음부터 다시 훑어 첫 NaN을 찾는다 (NaN은 드물다). NEON에는 64비트 비트 반전이 없어 "NaN이 아님"을 AND로 누적한다.
- 긴 연속 레인 하나의 argmax는 조각마다 답을 구한 뒤 조각 순서대로 비교한다: NaN이 있는 첫 조각이 이기고, 아니면 최댓값, 동점이면 앞 조각. 직렬 답과 정확히 같다.

### 6.4 순서와 무관한 접기, any/all
- 결과가 순서와 무관한 접기(wrap하는 정수 +/×, max/min, and/or)는 출력이 하나인 긴 구간을 고정 조각으로 나눠 병렬로 접고, 조각 결과를 순서대로 합친다. 결과는 numpy와 같다.
- **bool `any`/`all`은 조기 종료한다.** 4096원소 블록마다 답이 정해졌는지 보고, 정해지면 멈춘다. 같은 출력의 다음 구간도 바로 돌아온다. 블록 안은 분기 없는 벡터 OR/AND다. 앞에 True가 있는 `any(b)`는 0.1µs(numpy 0.4µs)다.
  - 블록은 `uint8_t`로 접는다. `bool`로 쓰면 컴파일러가 OR을 "첫 True 찾기" 루프로 바꿔 원소마다 분기가 생긴다 (0.9ns/원소, 모두 False인 100만 개에 906µs).
  - 조기 종료와 조각 병렬은 함께 쓰지 않는다. bool 입력의 any/all은 직렬로 접다가 멈추는 경로가 먼저다.
- `any`/`all`은 모든 dtype에서 `x != 0`을 본다 (nan은 True).

### 6.5 블록 리덕션 (`array/ops/reduction.cpp`의 `plan_blocks`, `reduce_blocked`)
줄이는 축이 메모리상 바깥에 있으면(C 순서 행렬의 `sum(axis=0)`) 남는 축이 안쪽 구간이 된다. 리덕션은 출력이 걸친 축만 나눌 수 있고(§10.5) 안쪽 축은 1024보다 짧게 자르지 않으므로(§10.4), 1000열짜리 행렬은 태스크 하나로 돈다. 그래서 줄이는 축 하나를 고정 개수의 블록으로 나눈다.
- 블록마다 출력의 사본(시작값으로 채움)에 접고(풀에서 병렬), 사본들을 블록 순서대로 출력에 접는다. 모든 접기의 시작값은 다시 접어도 값이 그대로다 (`0 + p`, `1 × p`, `max(최솟값, p)`, `false ∨ p`).
- **쓰는 조건**: 이터레이터의 분할 계획(`NdIter::plan`)이 태스크를 8개보다 적게 만들고, 일이 병렬 기준값 이상이고, (출력이 원소 2개 이상이거나 형변환이 있을 때).
  - 형변환이 있는 전체 리덕션(`sum(bool)`, `mean(int64)`)은 버퍼 캐스팅이 커널에 4096원소씩 넘기므로 커널이 구간을 나눌 수 없다. 블록이 없으면 1스레드로 돈다 (bool 1e7개 `sum`: 1스레드 1.47ms, 10스레드 1.33ms).
- **블록 수**: 가장 긴 줄이는 축을 `min(16, extent/32, 일/(8·kParallelTask))`개로 나눈다. 2개 미만이거나 원래 계획의 태스크 수보다 많지 않으면 나누지 않는다.
  - 상한 16(`kMaxBlocks`).
  - 블록 하나는 인덱스 32개 이상(`kMinBlock`): 블록마다 출력 크기만큼의 일(사본 채우기, 마지막 접기)이 더 든다. (64, 2000)을 8행짜리 블록 8개로 나누면 1스레드가 30% 느리다. 하한 32면 이 비용이 약 10분의 1이다.
- 블록 구조는 shape, 레이아웃, dtype만으로 정한다. 스레드가 1개여도 같은 구조로 계산한다. 그래서 스레드 수와 무관하게 같은 비트이고, float 덧셈·곱셈은 numpy와 비트가 다를 수 있다. max/min, any/all, 정수 합은 그래도 같다.
- 블록 뷰는 호출 스레드에서 미리 만든다. `NDArray`를 복사하면 `base`(Python 참조)의 참조 횟수가 바뀌는데, 태스크는 GIL 없이 돈다.
- 측정 (`sum(axis=0)`, 1스레드 / 10스레드, numpy)

  | shape | 블록 없음 | 블록 | numpy |
  |---|---:|---:|---:|
  | (1000, 1000) | 123 / 115µs | 130 / **32µs** | 123µs |
  | (4000, 250) | 122 / 122µs | 126 / **42µs** | 129µs |
  | (300, 300) | 10.7 / 10.7µs | 12.4 / 7.3µs | 11.4µs |
  | (64, 2000) | 14.6 / 11.3µs | 나누지 않음 | 15.4µs |

- 1000×1000 정규분포 `sum(axis=0)`의 numpy 대비 최대 상대오차는 2.7e-13이다. 한 사슬에 더해지는 항이 블록 길이로 줄어 오차 한계는 직렬 합보다 작다.

### 6.6 float/complex `prod` (`product_piece`, `product_mt`)
- 왼쪽부터 곱하는 product는 곱셈 하나(지연 4사이클)씩 기다리는 사슬 하나다 (100만 개 0.78ms, numpy와 같음).
- 구간을 4096원소(`kLeaf`) 조각으로 나눈다. 조각 안에서 누산기 8개가 8칸 간격으로 원소를 맡고 `((a0·a1)·(a2·a3))·((a4·a5)·(a6·a7))`로 합친다. 원소가 8개 미만인 조각은 왼쪽부터 곱한다. 조각들은 왼쪽부터 곱한다.
- 구조는 n에만 의존하고, 조각은 병렬로 돌 수 있다. 1스레드 99µs(7.8배), 10스레드 22µs다. complex 30만 개는 numpy 1.17ms 대비 75µs / 13µs다.
- **곱하는 순서가 바뀌면 오버플로/언더플로 시점도 바뀐다.** 그래서 차이가 마지막 비트에 그치지 않는 입력이 있다. numpy가 유한한 값을 낼 때 이쪽이 `inf`나 `nan`을 낼 수 있고, 그 반대도 있다.
  - 예: `[1e200, 1e-200]`을 8번 반복한 16개의 `prod`는 numpy가 `1.0`, sd가 `nan`이다. numpy는 왼쪽부터 곱해 중간값이 1과 1e200 사이를 오간다. sd에서는 누산기 0이 짝수 번째 두 개를 맡아 `1e200·1e200 = inf`, 누산기 1이 홀수 번째 두 개를 맡아 `1e-200·1e-200 = 0`이 되고, 합칠 때 `inf·0 = nan`이다.
  - 블록 `prod(axis=0)`(§6.5)과 긴 `cumulative_prod`(§6.7)도 같다. 블록마다 1부터 새로 곱하므로, 한 블록 안의 곱이 앞 블록의 작은 값과 만나기 전에 넘칠 수 있다.
  - 넘쳤을 때 왼쪽부터 다시 계산하는 대안은 택하지 않았다. NaN 전파와 `inf·0 = nan` 규칙, 스레드 수와 무관한 결과는 그대로다.

### 6.7 누적 스캔 (`scan_lane`)
- 첫 출력은 항등원 + x가 아니라 첫 입력 자신이다. 그래서 `-0.0`이 남는다 (numpy와 같다). `include_initial`이면 항등원을 먼저 쓴다 (Array API).
- **float/complex 레인이 8192원소 이상이면 2단계 블록 스캔**을 한다 (`kScanBlock = 4096`).
  1. 블록마다 따로 스캔한다. 각 블록은 자기 첫 원소에서 출발한다. 한 루프에서 블록 4개(`kScanGroup`)를 나란히 돌려 사슬 4개를 동시에 진행한다. 블록 그룹은 풀에서 병렬로 돈다.
  2. 블록 끝값으로 carry를 직렬로 구한다.
  3. 블록 1부터 각 원소에 carry를 더한다(곱한다). 벡터화되는 스트리밍 루프다.
- 블록은 n에만 의존하므로 스레드 수와 무관하게 같은 비트다. float는 numpy와 비트가 다를 수 있다.
- **정수 스캔**은 순서와 무관하게 결과가 같다. 3단계가 순수한 추가 패스이므로, 병렬로 돌 수 있을 때(n ≥ 기준값, 풀 사용 가능)만 이 경로를 탄다.
- 구현 세부 (모두 측정으로 정한 것이다)
  - 블록마다 8개씩 모아 load → scan → store 한다. 4개 블록에 원소 하나씩 번갈아 저장하면, 새로 할당한 출력의 첫 쓰기에서 4개 페이지의 폴트가 겹쳐 5.7배 느리다 (100만 개 1.7ms 대 0.3ms). 결과는 항상 새 배열에 쓰므로 이것이 실제 상황이다.
  - 연속이면 stride를 컴파일 시점 상수로 둔다 (런타임 stride는 2.3배 느리다).
  - 첫 덩어리는 컴파일 시점 플래그(`template <bool First>`)로 분리한다. 시작 인덱스가 런타임 값이면 루프를 완전히 펼치지 못해 8×4 버퍼가 스택에 남는다 (2배 느리다).
- 결과: `cumulative_sum` 1스레드 354µs(직렬 495µs), 10스레드 70µs, numpy 2.2ms. `cumulative_prod` 386µs / 74µs, numpy 2.4ms.

### 6.8 mean, var, std
- `mean`: 결과 dtype(float64/complex128)으로 합한 뒤 개수로 나눈다. 빈 입력은 `0/0 = nan`이다. 정수 입력은 버퍼 캐스팅으로 float64가 되어 합해진다.
- `var`: 두 패스다. 평균을 구한 뒤, 평균과의 편차 제곱합(`|x - m|²`)을 임시 배열 없이 커널 하나로 접는다. `N - correction`으로 나눈다. `N - correction ≤ 0`이면 NaN이다 (빈 입력 포함, Array API). 두 번째 패스를 커널 하나로 하므로 numpy의 0.66배 시간이다.
- `std = sqrt(var)`.
- frame의 int64 `mean`도 이 계산을 그대로 쓴다: `sd.sum(x, dtype=sd.float64)`을 개수로 나눈다 (`frame/interop.cpp`의 `array_sum_float64`, DESIGN §9.1 11번). 그래서 두 영역의 결과가 비트까지 같다.

### 6.9 numpy와 비트가 달라지는 경우
결정성(스레드 수와 무관)은 모든 경우에 지킨다. numpy와의 비트 동일성은 아래에서만 포기했다. 이유는 모두 같다: numpy의 순서를 지키면 병렬로 나눌 수 없거나 사슬 하나에 묶여 느리다.

| 경우 | 바뀐 순서 | 얻은 것 |
|---|---|---|
| 줄이는 축이 바깥에 있어 남는 축만으로 나눌 수 없는 리덕션 (`sum`/`mean`/`var`/`prod(axis=0)` 등) | 줄이는 축을 고정 블록으로 (§6.5) | 1000×1000 `sum(axis=0)` 10스레드 115 → 32µs |
| float/complex `prod` | 4096원소 조각 안에 누산기 8개 (§6.6) | 1스레드 7.8배 |
| 8192원소 이상의 float/complex `cumulative_sum`/`cumulative_prod` | 4096원소 블록 2단계 스캔 (§6.7) | 1스레드 1.4배, 10스레드 7배 |
| 형변환이 필요한 리덕션 (`mean(int64)`, `sum(x, dtype=float64)` 등) | 4096원소 버퍼 단위로 접고, 전체 리덕션은 고정 블록으로 (§4.3, §6.5) | 메모리 왕복 감소, 병렬화 |

- 정수 결과(`sum(bool)` 등)는 순서와 무관하므로 그대로 정확하다.
- 형변환이 없고 출력이 하나인 긴 합(`sum(f)`)은 numpy와 비트가 같다 (§6.2).

---

## 7. 인덱싱

### 7.1 키 해석 (`array/ops/indexing.cpp`)
- 파싱은 C++에서 한다. 키의 항목은 고정 용량 저장소(`Items`, 축을 쓰는 항목 32 + `newaxis` 32 + `...` 1)에 둔다. `std::vector`의 증가 할당이 `x[1, 2]` 비용의 대부분이었다.
- 기본 항목(slice, `newaxis`, `...`, int)을 적용해 `view`를 만들고, 고급 항목이 가리키는 축은 연속한 범위 `[first, first + count)`로 남긴다. 그 축마다 int64 인덱스 배열이 하나씩 생기고, 이것들이 함께 브로드캐스트된 "인덱스 shape" B가 된다.
  - 결과 shape = `view.shape[:first] + B + view.shape[first + count:]`
- **기본 항목으로 떨어진 고급 인덱스는 IndexError다** (파싱 단계). numpy의 "인덱스 축을 맨 앞으로 옮기는" 규칙을 두지 않는다.
- **Python bool은 인덱스가 아니다** (IndexError). numpy는 bool을 0차원 마스크로 읽는데, 0차원 배열이 없는 여기서는 `x[1]`과 나란히 놓여 헷갈리기만 한다. bool 배열 마스크는 된다.
- **앞쪽 축이 모두 정확한 Python int인 키**(`x[1, 2]`, `x[1]`)는 키 해석을 건너뛰고 오프셋을 바로 계산한다 (`leading_ints_fast`). 축마다 int가 있으면 원소를, 아니면 남은 축의 뷰를 돌려준다. bool, 너무 큰 int, 범위 밖 인덱스는 일반 경로로 넘겨 평소의 에러를 낸다. `x[1, 2]` 58 → 22ns (numpy 32ns).
- 반복(`iter(x)`)은 키를 해석하지 않고 `item_along_axis0`으로 축 0을 걷는다.

### 7.2 기본 인덱싱
- data 포인터, shape, strides만 다시 계산한다. 복사가 없다. `x[2:8:3]`은 "2칸 들어가서 2개를, stride ×3으로".
- 모든 축을 int로 지정하면 결과는 Python 스칼라다 (§2.2).

### 7.3 정수 배열: gather와 scatter
- **결과 레이아웃은 view를 따른다** (`advanced_result`). 원소별 연산의 결과와 같은 규칙이다. view의 축은 메모리 순서를 유지하고, 인덱스 축 B는 인덱싱된 축의 자리(그중 가장 바깥)에 C 순서로 들어간다.
  - F 순서 `mT[:, rows]`는 F 순서 결과가 되어 양쪽이 연속으로 채워진다 (C 결과로 만들면 386µs, view를 따르면 114µs, numpy 149µs).
  - numpy는 C 순서 `m[:, rows]`에도 F 결과를 만든다. 이 규칙이 더 단순하다.
  - gather 루프는 B를 한 덩어리로 다룬다(점 b는 `b × 마지막 B 축 stride`). 크기 1이나 stride 0인 축 때문에 이 성질이 깨지는 드문 경우는 C 순서로 돌아간다.
- **1차원 view, 인덱스 배열 하나, 같은 dtype** (`gather_one`, `scatter_one`): 가장 흔한 경우를 위한 합친 루프다. 인덱스를 읽고, 음수를 보정하고, 원소 하나를 옮긴다.
  - 일반 경로는 원소마다 "인덱스 배열 k개" 루프를 돌며 stride와 shape을 다시 읽고, 오프셋을 따로 버퍼에 썼다. 이것이 합친 루프의 3.5배 일이다. 디스어셈블로 확인했다 (원소당 명령어 약 26개 대 몇 개).
  - gather는 검사 패스가 없다. 나쁜 인덱스는 분기 없이 표시만 하고 0번으로 읽은 뒤, 끝나고 IndexError를 낸다. 결과는 새 배열이라 지킬 것이 없다. 이터레이터 위에서 돌아 병렬로 나뉜다. 축 길이가 0이면 0번 원소가 없으므로 루프 전에 검사한다.
  - scatter는 **모든 인덱스를 먼저 검사한다.** 그래서 루프에 에러 경로가 없고, `x[idx] = v`는 전부 아니면 전무다: 범위 밖 인덱스가 있으면 아무것도 쓰지 않는다. 쓰기는 C 순서, 1스레드로 해서 반복된 인덱스에서는 마지막 쓰기가 이긴다.
  - 결과 (인덱스 100만 개): gather 1스레드 733µs, 10스레드 81µs, numpy 757µs. scatter 1스레드 1.04ms, numpy 954µs (10스레드도 직렬이다).
- **인덱스 배열 하나 + 블록** (`gather_take`, `x[..., idx, ...]`): `result[pre, b, post] = view[pre, idx[b], post]`를 결과 shape 위의 원소별 연산 하나로 본다.

  | 피연산자 | pre 축 | B 축 | post 축 |
  |---|---|---|---|
  | view (인덱스 축을 뺀 것) | view의 stride | 0 | view의 stride |
  | 인덱스 배열 | 0 | 인덱스의 stride | 0 |
  | 결과 | 결과의 stride | 결과의 stride | 결과의 stride |

  - 메모리 순서, coalescing, 병렬 분할은 이터레이터가 정한다. 커널은 인덱스를 바이트 오프셋으로 바꾸고 원소를 옮긴다. 인덱스 검사는 B개만 먼저 한다.
  - 안쪽 구간에서 인덱스 stride가 0이면(`m[rows]`: 한 행 전체가 같은 인덱스) 오프셋을 한 번만 계산하고 `memcpy`한다.
  - 블록마다 이터레이터를 만들던 일반 경로는 블록이 짧거나(`p3[idx]`, 3원소 블록 30만 개: numpy의 3.4배) 멀리 떨어져 있을 때(`m[:, rows]`: 1.4배) 느리다.

  | 연산 | 1스레드 | 10스레드 | numpy |
  |---|---:|---:|---:|
  | `m[:, rows]` (1000×1000, 인덱스 1000개) | 332µs | 51µs | 486µs |
  | `t3[:, :, idx]` | 198µs | 29µs | 230µs |
  | `m[rows]` | 111µs | 32µs | 143µs |
  | `p3[idx]` | 2145µs | 303µs | 2709µs |

- **일반 경로** (인덱스 배열 여럿, dtype 변환): 인덱스 shape B를 C 순서로 걸으며 각 점의 블록 오프셋을 계산한다. 오프셋은 1024개씩 작은 버퍼에 만들어 바로 소비한다 (원소 100만 개면 8MB인 오프셋 배열을 쓰고 다시 읽지 않는다). 블록 복사 방법은 호출당 한 번 고른다: 원소 하나이고 dtype이 같으면 타입이 정해진 대입문이 되어 인라인된다.

### 7.4 bool 마스크
- 마스크 하나만 고급 항목이면 좌표 배열로 바꾸지 않고 마스크 자체로 적용한다: 한 번 훑으며 True인 자리만 복사한다(compress).
- **고정 청크 두 단계** (`kMaskChunk = 2¹⁴`): 마스크를 C 순서로 16K원소 청크로 나눈다.
  1. 청크마다 True 수를 센다 (병렬, 연속이면 바이트 합이라 벡터화된다).
  2. 누적합이 각 청크 선택분의 시작 위치다 (직렬).
  3. 청크마다 자기 구간만 걷는다 (병렬).
  - 모든 원소가 C 순서의 자기 자리에 들어가므로 결과는 스레드 수와 무관하게 직렬과 같다. 대입(`x[m] = v`)도 같은 구조다. 청크마다 x의 서로 다른 원소만 건드린다.
- **분기 예측**: 연속 구간은 256원소 조각마다 True 수를 먼저 센다(벡터 합).

  | 조각의 True 수 | 방법 |
  |---|---|
  | 0 | 건너뛴다 |
  | 1~31 | 8바이트씩 읽어 0이면 8개를 건너뛴다 |
  | 32 이상 | 분기 없이: 모든 원소를 작은 버퍼의 `buf[cnt]`에 쓰고 `cnt += 마스크 바이트`, 끝에 `memcpy` |

  - 무작위 절반 True 마스크는 원소마다 분기가 50% 틀린다 (분기 하나짜리 루프: 100만 개에 2.2ms).
  - 결과가 아니라 버퍼에 쓴다. 결과에 바로 쓰면 청크 끝에서 다음 청크 영역에 한 칸을 쓰게 되어 다른 스레드와 경합한다.
  - strided 구간은 sparse 경로로 통째로 간다.
- **마스크 대입**: `*e = q ? val : *e`는 컴파일러가 분기 있는 조건부 저장으로 되돌린다. 그래서 원소의 원시 워드에 마스크 바이트로 만든 비트 마스크를 적용한다: `keep = q - 1`, `e = (e & keep) | (val & ~keep)`. False 원소의 비트(NaN 페이로드, `-0.0`)는 그대로 남는다.
  - 배열 값의 조밀한 경로는 False 원소에서도 `w[b]`를 읽는다. 선택의 마지막 True가 든 조각 하나만 값 배열 끝을 넘어 읽을 수 있으므로, 그 조각(`b + trues == total`)만 sparse 경로로 보낸다. 루프 안에서 매번 막지 않는다.
- **bool 바이트는 0/1이어야 한다.** 분기 없는 압축은 마스크 바이트를 정수로 더한다. 우리가 만든 bool은 항상 0/1이고, 바깥에서 들어오는 bool 버퍼는 가져올 때 검사한다 (§12.3).

  | 연산 | 1스레드 | 10스레드 | numpy |
  |---|---:|---:|---:|
  | `f[b]` (절반 True) | 324µs | 48µs | 3054µs |
  | `f[s]` (1% True) | 96µs | 20µs | 233µs |
  | `f[b] = 0.0` | 412µs | 80µs | 2711µs |

### 7.5 인덱스 대입
- 값(배열, 중첩 리스트, Python 스칼라)은 선택된 shape으로 브로드캐스트되고, 위로만 변환된다 (§2.5). 스칼라는 대상 dtype의 원소 1개짜리 배열이 된다.
- 기본 인덱스 대입은 view에 `assign_checked`(writeable 검사, 위로만 변환, 겹침 처리)를 한다.
- 겹침: 값, 인덱스 배열, 마스크가 대상과 겹치면 먼저 복사한다 (§3.5). 대입은 대입 전 x의 값으로 읽은 인덱스를 쓴다.

---

## 8. 정렬과 unique

### 8.1 순서 규칙 (`array/ops/sort.cpp`)
- 비교 함수는 하나다. NaN은 가장 큰 값이다: 오름차순이면 맨 끝, 내림차순이면 맨 앞이다.
- **NaN 분리** (`load_split_nans`): 레인을 버퍼로 읽어 올 때 NaN과 나머지를 따로 채운다(각각 원래 순서). 오름차순이면 NaN을 뒤에, 내림차순이면 앞에 둔다. 나머지 부분만 NaN 검사 없는 평범한 비교로 정렬한다. 비교마다 있던 분기가 사라져 1.22배다.
  - NaN끼리는 언제나 원래 순서다. 그래서 내림차순의 NaN 묶음도 안정 정렬의 결과와 같다. 내림차순으로 바꾸는 데 드는 비용은 없다 (정렬할 구간의 시작이 NaN 개수만큼 밀릴 뿐이다).
- **`descending=True`에서도 동점 원소의 원래 순서가 유지된다.** 비교 함수를 뒤집지, 오름차순 결과를 뒤집지 않는다. 예외는 동점을 구별할 수 없는 경로(NEON 값 정렬, §8.3)로, 거기서는 결과를 뒤집는다.
- complex는 정렬할 수 없다 (§2.6). `unique_*`만 (실수부, 허수부) 사전순을 쓴다.

### 8.2 알고리즘 선택
dtype과 레인 길이만 본다. 데이터를 보고 고르지 않는다 (§1.4).

| 경우 | 알고리즘 |
|---|---|
| int64 값 (`sort`, `unique_values`), 1024개 이상 | LSD 기수 정렬 (§8.4) |
| int64 (값, 인덱스) 쌍 (`argsort`, `unique_*`), 128개 이상 | LSD 기수 정렬 |
| int64 값, 32~1023개 | NEON 정렬 (§8.3) |
| float64 값, 32개 이상, `stable=False` 또는 -0.0과 +0.0이 섞여 있지 않음 | NEON 정렬 |
| 나머지, `stable=True` | 하향식 merge sort, 16개 이하 구간은 insertion sort |
| 나머지, `stable=False` | introsort (quicksort, 재귀 깊이 `2 log₂ n`을 넘으면 heapsort, 16개 이하는 insertion sort) |

- `argsort`는 (값, 인덱스) 쌍을 정렬한다. 간접 비교보다 캐시 지역성이 좋다. 인덱스로 순서가 드러나므로 NEON 정렬은 쓰지 않는다.
- merge sort의 병합은 오른쪽이 엄격히 작을 때만 오른쪽을 가져온다 (안정성). 두 반이 이미 순서대로면 병합하지 않는다.
- quicksort는 3개의 중앙값을 피벗으로 Hoare 분할하고, 작은 쪽을 재귀, 큰 쪽을 루프로 돈다(스택 O(log n)).

### 8.3 NEON 정렬 (`array/kernels/simd_sort.hpp`)
- **16개 블록**: 레지스터 8개에 싣고 8입력 최적 네트워크(비교기 19개)를 **세로로** 적용한다. 비교-교환 하나가 레지스터 전체의 min 하나와 max 하나라서 두 레인이 동시에 정렬된다. 분기가 없다. 2×2 전치(zip)로 길이 8인 런 두 개로 쓴다. 네트워크의 정확성은 0-1 원리로 확인했다(0/1 입력 256가지).
- **병합**: 4+4를 레지스터 안에서 bitonic 네트워크로 합친다. 아래 4개를 내보내고 위 4개는 남겨 다음 4개와 만난다. 분기가 원소마다가 아니라 4개당 하나다.
- int64는 64비트 정수 min/max가 없어 비교 + 선택(`vcgtq` + `vbslq`)으로 대신한다.
- 32개(`kSimdMin`) 미만은 insertion/merge sort와 차이가 없어 쓰지 않는다.
- **안정 정렬이 아니다.** 그래서 결과가 안정 정렬과 구별되지 않을 때만 쓴다: int64에서 같은 값은 비트가 같고, float64에서 같은 값이 다른 비트일 수 있는 경우는 NaN(이미 분리)과 ±0뿐이다. `stable=False`이면 float의 ±0 순서가 입력과 달라질 수 있다.
- 결과: float `sort` 100만 개 1스레드 14.2ms, numpy 26.6ms. merge sort 대비 4.9배.

### 8.4 int64 기수 정렬
- LSD 기수 정렬, 8비트 자리다. 계수 정렬이 안정적이므로 전체가 **안정 정렬**이다. 그래서 `sort`, `argsort`, `unique_*`에 같은 함수를 쓴다.
- **키**: 부호 비트를 뒤집어(`v ^ 1<<63`) 부호 있는 순서를 부호 없는 순서로 만든다. 내림차순은 키 전체를 비트 반전한다. 결과를 뒤집지 않으므로 동점 순서가 유지된다.
- **최솟값 기준**: 한 번 훑어 최솟값과 최댓값을 구하고, (최댓값 − 최솟값)의 자리만 정렬한다. 모든 키에서 같은 바이트인 자리는 건너뛴다.
  - 음수와 양수가 섞이면 부호를 뒤집은 키의 위쪽 바이트가 0x7F/0x80, 0xFF/0x00을 오가 건너뛸 자리가 없다. 최솟값을 빼면 `[-1000, 1000)`은 두 자리면 된다 (10.5 → 2.4ms).
- **임계값** (2²⁰개를 길이 n인 레인으로, 1스레드). 레인마다 카운터(최대 8×256)를 비우고 누적하는 고정 비용이 있다.

  | n | 값: NEON | 값: 기수 | 쌍: merge | 쌍: 기수 |
  |---:|---:|---:|---:|---:|
  | 64 | 3.2ms | 14.7ms | 13.2ms | 15.2ms |
  | 128 | 3.9ms | 10.1ms | 16.0ms | 10.6ms |
  | 512 | 5.9ms | 6.4ms | 21.6ms | 6.9ms |
  | 1024 | 6.8ms | 6.0ms | 24.3ms | 6.4ms |

  - 값은 1024개(`kRadixMinValues`), 쌍은 128개(`kRadixMinPairs`)부터 쓴다. 넓은 범위 기준으로 고정한다. 좁은 범위에서 더 일찍 이기지만, 그것을 보고 고르면 데이터 기반 휴리스틱이 된다.
- 결과 (100만 개, 1스레드): `sort(i)` 넓은 범위 5.8ms (numpy 19.1ms), `[-1000, 1000)` 2.4ms (numpy 8.8ms), `argsort(i)` 6.7ms (numpy 82.2ms).

### 8.5 긴 레인 하나의 병렬 정렬
- 여러 행은 레인 단위로 병렬화한다 (§4.5). 2000행 정렬: 1스레드 12.7ms, 10스레드 1.45ms (numpy 26.6ms).
- **2¹⁷개 이상인 레인 하나**도 병렬로 정렬한다. 조각은 2¹⁶개(`kSortPiece`)이고 길이로만 정한다. 2¹⁵, 2¹⁶, 2¹⁷ 중 어느 것도 모든 경우에 이기지 않아 가운데를 골랐다.
- **비교 정렬** (int64 제외): 조각마다 직렬 알고리즘으로 정렬한 뒤, 이웃한 런을 둘씩 병합하는 단계를 반복한다. 병합 하나를 출력 구간 2¹⁵개(`kMergeChunk`) 단위로 자른다. 구간 시작이 각 런의 어디인지는 이진 탐색(merge path)으로 찾으므로 모든 단계가 병렬로 돈다. **동점이면 왼쪽 런이 먼저다.** 그래서 안정 정렬된 런을 병합한 결과가 직렬 안정 정렬과 비트까지 같다.
- **int64**: 병합 없이 기수 정렬의 자리마다 병렬로 한다. 조각마다 바이트별 개수를 세고(병렬), "바이트 값 순 → 조각 순"으로 누적해 조각별 시작 위치를 정하고(직렬), 조각마다 흩뿌린다(병렬). 직렬 계수 정렬과 같은 안정 순서다. 조각 + 병합을 int64에 쓰면 기수 정렬이 이미 빨라 병합 패스가 더 비싸다 (2스레드 7.4ms 대 1스레드 6.0ms).
- **풀을 쓸 수 있을 때만 조각을 쓴다.** 병합은 직렬 정렬에 없는 추가 패스다 (1스레드에서 항상 조각으로 하면 `sort(i)` 5.8 → 13.6ms, `sort(f)` 13.1 → 23.6ms). 안정 정렬은 조각과 무관하게 결과가 같으므로 경로를 바꿔도 된다.
- **예외: `stable=False`는 항상 조각으로 정렬한다.** float `sort`의 NEON 정렬은 ±0의 순서가, int64가 아닌 `argsort`의 introsort는 동점 인덱스의 순서가 조각 경계에 따라 달라진다. 항상 조각을 써야 결과가 스레드 수와 무관하다.
- 결과 (100만 개, 1스레드 / 10스레드, numpy): `argsort(f)` 72.4 / 11.3ms (74.4), `sort(f)` 14.2 / 4.0ms (26.6), `sort(i)` 5.9 / 2.3ms (19.6), `argsort(i)` 6.8 / 3.3ms (84.9).

### 8.6 unique_*
- 처리 순서: 평탄화 → 안정 argsort → 이웃 비교. 안정하므로 같은 값 묶음의 첫 원소가 x에서의 첫 등장이다. 그래서 `indices`가 나오고, `-0.0`과 `+0.0` 중 먼저 나온 쪽이 남는다.
- `unique_values`는 위치와 개수가 필요 없으므로 값만 안정 정렬하고 제자리에서 압축한다. 옮길 데이터가 절반이다 (1.24배).
- **NaN은 하나의 묶음이다** (§2.6). NaN 분리(§8.1)로 이미 맨 뒤에 원래 순서대로 모여 있으므로, 이웃 비교는 NaN이 아닌 부분에서만 하고 NaN 구간 전체를 마지막 묶음 하나로 친다. 그 묶음의 값은 x에서 처음 나온 NaN이다(비트 그대로). NaN끼리 비교하지 않으므로 비트가 다른 NaN도 같은 묶음이다. 추가 비용은 없다.
  - numpy는 정렬된 결과의 첫 원소를 보여 주므로 `-0.0`/`+0.0`과 complex NaN에서 보여 주는 값이 입력 순서와 무관하게 정해지지 않는다. 우리는 frame과 같이 처음 나온 값이다.
- **해시 기반 경로는 두지 않는다.** int64는 기수 정렬로 이미 numpy보다 빠르다 (`unique_counts` 값 1000종 3.9ms 대 8.4ms, 모두 다른 값 9.3ms 대 20ms). 해시는 값 종류가 적으면 더 빠를 수 있지만 모두 다르면 느려질 수 있고, 데이터를 보고 경로를 고르면 §1.4의 휴리스틱이 된다.

### 8.7 작업 버퍼
- 64KB 이하의 레인은 스레드별 scratch를 쓴다 (`kScratchBytes`). 슬롯 0은 레인, 슬롯 1은 merge sort의 절반 복사본이다. 짧은 레인을 많이 정렬할 때(`(100000, 8)`의 `sort(axis=1)`) 레인마다 할당하지 않는다. 더 긴 레인은 할당한다.

---

## 9. matmul (`array/ops/matmul.cpp`)

### 9.1 형태와 dtype
- `(..., m, k) @ (..., k, n) → (..., m, n)`. 앞의 배치 축은 브로드캐스트한다. 1차원 피연산자는 왼쪽이면 행, 오른쪽이면 열 벡터이고 그 축은 결과에서 빠진다. 벡터 `@` 벡터는 Python 스칼라다.
- 계산 dtype은 `promote(promote(a, b), int64)`다. 그래서 `bool @ bool`은 int64다.
- `@`는 배열만 받는다(Python 스칼라 없음). `x @= y`는 결과가 x의 shape이어야 하고 위로만 변환한다.

### 9.2 BLAS 경로 (float64, complex128)
- Accelerate의 BLAS를 쓴다. ILP64 새 인터페이스다 (§1.3).
  - 행렬 × 행렬: `cblas_dgemm`/`cblas_zgemm`
  - 한 행 또는 한 열: `gemv`
  - 원소 하나(벡터 × 벡터): `dot`. complex는 `cblas_zdotu_sub`(켤레 없음, matmul은 켤레를 취하지 않는다)
- **레이아웃**: BLAS는 포인터, leading dimension, 전치 플래그로 행렬을 읽는다. 행이나 열 중 한 방향이 연속이면 그대로 넘긴다. 열이 연속이면 "cols×rows 행 우선 행렬의 전치"로 넘긴다. 그 외에는 연속 행 우선 복사본을 만든다. 벡터의 stride가 0이거나 음수면 복사한다.
- 한 행렬은 항상 BLAS 호출 하나다. 그래서 배치를 어떻게 나눠도 비트가 같다.
- BLAS 호출은 `accelerate/blas.hpp`의 `blas::dot`, `blas::gemv`, `blas::gemm`(float64와 complex128을 템플릿 하나로)을 거친다.

### 9.3 int64 커널 (int64, bool)
- 자체 커널이다. 루프 순서는 i-k-j라서 가장 안쪽 루프가 b의 한 행과 c의 한 행을 걷는다. 블로킹과 패킹은 하지 않는다. 곱셈과 덧셈은 wrap한다.
- **곱셈은 스칼라 + 4배 언롤**이다. AArch64 NEON에는 64비트 정수 벡터 곱셈이 없다. 독립 곱셈-덧셈 4개가 곱셈기를 바쁘게 한다.
  - 언롤에는 이유가 하나 더 있다. 명령어 7개짜리 루프는 반복마다 명령 인출 속도에 묶이고, 64바이트 인출 블록 경계를 가로지르면 1.7배 느려진다. 같은 소스가 링커가 놓는 주소에 따라 7.5ms와 12.9ms로 갈렸다(`-falign-loops=64`로 확인). 몸체가 커지면 위치에 둔감해진다. 정렬 옵션은 넣지 않는다.
- **병렬**: 출력 행 묶음을 태스크로 한다. 태스크 하나는 곱셈-덧셈 약 4096개(`kParallelTask`) 분량의 행이다. 행끼리 독립이므로 나누는 방식이 비트를 바꾸지 않는다. 태스크가 하나뿐이면 직렬이다.
- 결과: 300³ 1스레드 5.16ms, 10스레드 784µs, numpy 8.47ms. 500³은 10스레드에서 7.5배 (§10.6).

### 9.4 배치
- 큰 행렬의 배치는 차례로 돈다. Accelerate가 내부에서 스레드를 쓰므로 우리 풀과 겹치면 경쟁한다.
- **작은 행렬(m·n·k ≤ 256³, `kSmallBlas`)의 배치는 풀로 나눈다.** Accelerate는 정사각 dgemm을 256까지 한 스레드로, 288부터 여러 스레드로 돈다 (CPU 시간 / 경과 시간: 256에서 1.00, 288에서 1.62, 512에서 1.86).
  - 태스크 하나에 곱셈-덧셈 약 2¹⁵개(`kBlasTaskWork`)가 되도록 행렬을 묶는다 (8×8이면 64개).
- 측정 (1스레드 / 10스레드)

  | 배치 | 1스레드 | 10스레드 | 배율 |
  |---|---:|---:|---:|
  | 4000 × 4×4 | 113µs | 28µs | 4.1× |
  | 1000 × 8×8 | 65µs | 14µs | 4.6× (numpy 66µs) |
  | 500 × 16×16 | 96µs | 17µs | 5.5× |
  | 200 × 32×32 | 75µs | 28µs | 2.7× (numpy 77µs) |
  | 100 × 64×64 | 150µs | 64µs | 2.3× |
  | 10 × 256×256 | 735µs | 474µs | 1.6× |

  - 32×32부터 약 2배에서 멈춘다. Accelerate가 행렬 연산을 클러스터마다 하나인 행렬 유닛(AMX)으로 보내고, P코어 클러스터가 둘이기 때문으로 추정한다. 측정 모양에서 추론한 것이고 직접 확인하지는 않았다.

---

## 10. 스레드 (`parallel/pool.hpp`, `parallel/pool.cpp`)

### 10.1 풀의 구조
- **상주 워커**를 둔다. 작업 하나는 태스크 여러 개다. 워커와 호출 스레드가 **atomic 카운터 하나**에서 다음 태스크 번호를 가져간다. 빠른 스레드가 더 많이 가져갈 뿐, 스레드에 일을 고정 배정하지 않는다.
- **호출한 스레드도 태스크를 처리한다.**
- **spin 후 대기**: 워커는 일이 끝나면 최대 100µs 동안 spin하며 다음 일을 기다리고, 그동안 일이 없으면 condition variable로 잠든다.
  - spin은 반복 횟수가 아니라 시계로 잰다. ARM `yield`는 1~2사이클이라 큰 반복 횟수도 수 µs면 끝난다.
  - 잠든 워커를 깨우는 비용은 macOS에서 작업당 수십 µs다 (원소 100만 개 덧셈 전체보다 크다).
- **mutex 없이 시작**: 일의 내용은 세대 번호를 올리기 전에 기록되고, 세대 번호는 release로 쓰고 acquire로 읽는다. spin 중에 새 세대를 본 워커는 mutex를 잡지 않고 바로 시작한다. mutex는 잠들 때만 잡는다.
  - 병렬 호출 한 번의 고정 비용: 깨우기만 하면 약 25µs, spin을 넣어도 18µs(워커 9개가 한꺼번에 mutex를 잡는 경합, `__psynch_mutexwait`), mutex 없이 시작하면 **약 1.5µs**.
  - 대가: 일이 끝난 뒤 워커가 최대 100µs 동안 CPU를 쓴다. OpenMP 같은 라이브러리도 같은 절충을 한다(기본 대기가 훨씬 길다).
- 새로 만든 워커는 **현재 세대 번호**에서 시작한다. 0에서 시작하면 이미 끝난 일을 새 일로 착각해 완료 보고가 어긋나고, 다음 작업이 영원히 기다린다.
- **중첩 금지**: 태스크 안에서 부른 `parallel_for`는 그 스레드에서 직렬로 돈다 (`t_in_task`).
- **다른 Python 스레드가 풀을 쓰는 중이면** 새 호출은 호출 스레드 혼자 직렬로 실행한다 (`busy_`). 기다리거나 과다 구독하지 않는다.
- 태스크의 예외는 하나만 잡아 호출 스레드에서 다시 던진다.
- 워커는 macOS QoS `USER_INTERACTIVE`로 올려 P코어가 비어 있으면 P코어에 머문다.
- 풀은 소멸시키지 않는다. 인터프리터 종료 중에 스레드를 join하는 것은 불안정하고, 종료하면 OS가 회수한다.
- 풀은 모든 영역이 같이 쓴다. 언제 나눌지(기준값)는 영역의 몫이다: 배열은 `array/parallel_limits.hpp`의 `kParallelThreshold`, `kParallelTask` (§10.4).
- `parallel::FirstRow`: 여러 태스크가 검사하다 실패한 가장 앞의 위치(행, 원소)를 고른다. 스레드 수와 무관하게 같은 위치를 에러로 보고하기 위해서다 (frame이 쓴다).

### 10.2 GIL
- `parallel_for`는 부르면 언제나 GIL을 놓는다 (`GilRelease`). 풀이 바빠 직렬로 돌게 되어도 놓는다. 다른 Python 스레드는 그동안 계속 돈다.
- `parallel_for`를 부를지(일이 기준값 이상인지)는 호출하는 쪽이 정한다. 기준값 미만, 1스레드, 중첩인 작업은 `parallel_for`를 부르지 않고 GIL을 쥔 채 돈다.
- `available()`은 풀이 다른 Python 스레드의 일로 바쁘면 false다. 그런 호출은 어차피 직렬로 도므로, 경로를 고르는 알고리즘이 직렬 경로를 고른다(같은 결과, 나눠 놓고 직렬로 도는 것보다 빠름).
- 태스크 안에서는 Python 객체를 만지지 않는다. `NDArray` 복사도 `base` 참조 횟수를 바꾸므로 태스크 밖에서 미리 한다 (§6.5).

### 10.3 스레드 수
- `sd.set_num_threads(n, /)`, `sd.get_num_threads()`. 1이면 풀을 쓰지 않는다.
- **기본값은 P코어 수(10)다** (`sysctlbyname("hw.perflevel0.physicalcpu")`).
  - 동적 분배에서는 E코어가 전체를 기다리게 하지 않는다. E코어는 태스크를 적게 가져갈 뿐이다.
  - 14스레드에서 계산에 묶인 연산은 5~27% 빨라지고(`sin` 9.4 → 10.9배), 메모리에 묶인 연산은 3~12% 느려진다. 그래서 기본은 10으로 두고, 필요하면 `set_num_threads(14)`를 쓴다.
- `set_num_threads`는 실행 중인 작업이 끝나기를 기다린 뒤 워커를 다시 만든다. 스레드 수는 atomic으로 읽으므로 다른 스레드가 언제 물어도 안전하다.

### 10.4 기준값과 태스크 크기
이터레이터의 분할(`NdIter::plan`)이 쓰는 상수다.

| 상수 | 값 | 근거 |
|---|---|---|
| `kParallelThreshold` | 2¹⁵ (float64 원소 기준) | 이보다 작은 일은 1스레드로 돈다. 10스레드가 이기기 시작하는 크기가 `f + g` 약 1.6만, `exp` 약 8천, `sum` 약 3.2만이다. 상수 하나로 두고 가장 늦게 이득이 나는 `sum`에 맞췄다 |
| `kParallelTask` | 2¹² (float64 원소 기준) | 태스크 하나의 일. 고정 비용이 약 1.5µs라 작은 태스크로도 부하가 고르게 나뉜다 |
| `kMinInnerPiece` | 1024원소 | 안쪽 축을 자를 때의 최소 조각. 3000×3000 `sum(axis=0)` 10스레드: 조각 10이면 1스레드의 2.7배 느림, 256 → 1.50ms, **1024 → 0.91ms**, 4096 → 1.11ms (1스레드 1.05ms) |

- **일은 바이트로 센다.** 원소 수 × 피연산자 안쪽 stride 중 가장 큰 것(8바이트 상한)이고, 기준값과 태스크 크기는 위 값 × 8바이트다.
  - bool처럼 1바이트 피연산자만 있는 루프는 8배 많은 원소를 받아야 병렬로 돌고, 태스크 하나도 8배 많은 원소를 받는다. 4096원소(4KB) bool 태스크는 atomic 카운터와 캐시 줄 주고받기 비용이 실제 일보다 커서 `b & b`가 10스레드에서 1스레드보다 느렸다 (1000만 개 187µs 대 122µs). 바이트로 세면 38µs다.
  - 8바이트 피연산자가 하나라도 있으면 원소 수로 센 것과 같다.
- **나눌 축**: 줄 수 있는 축 중 extent ≥ 16인 가장 바깥 축, 없으면 가장 긴 축. 안쪽 축을 자를 때는 조각을 `kMinInnerPiece` 이상으로 한다.
- 분할 계획은 shape, 레이아웃, dtype만으로 정하고 스레드 수를 보지 않는다. 그래서 결과를 바꾸는 판단(블록 리덕션, §6.5)이 이 계획을 근거로 삼아도 된다.
- **이터레이터 밖의 병렬 경로**는 자기 기준을 쓴다. 모두 길이나 크기로만 정한다.

  | 경로 | 나누는 기준 |
  |---|---|
  | pairwise 합, `prod`, 정수 스캔 | n ≥ `kParallelThreshold`, 잎·조각 4096원소 |
  | float/complex 스캔 | n ≥ 8192, 블록 4096 (구조는 1스레드에서도 같다) |
  | 블록 리덕션 | §6.5 |
  | bool 마스크 | 청크 2¹⁴원소, 청크가 둘 이상이면 |
  | 긴 레인 정렬 | n ≥ 2¹⁷, 조각 2¹⁶, 병합 구간 2¹⁵ |
  | int64 matmul | 곱셈-덧셈 약 2¹² 분량의 출력 행 |
  | 작은 BLAS 배치 | 행렬 ≤ 256³, 태스크 약 2¹⁵ 곱셈-덧셈 |

### 10.5 결정성 규칙
**결과는 스레드 수와 무관하게 비트까지 같다.** 나누는 방식은 데이터 크기와 레이아웃으로만 정하고, 부분 결과는 고정된 순서로 합친다. "스레드가 있을 때만 다른 경로"는 쓰지 않는다. 결과가 같다고 보장되는 경우(안정 정렬의 조각 등)만 예외다.

| 연산 종류 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 원소별 (산술, 초월함수, 캐스트, 복사, gather) | 아무 축이나 | 원소마다 따로 계산한다. vForce도 위치에 독립이다 |
| 축 리덕션 (`Split::Reduction`) | 출력이 실제로 걸친 축(출력 stride ≠ 0)만 | 출력 원소 하나에 입력이 접히는 순서가 직렬과 같다 |
| 블록 리덕션 | 줄이는 축의 고정 블록 | 블록 구조가 스레드 수와 무관하다 (§6.5) |
| 출력이 하나인 긴 float 합 | pairwise 트리의 잎 | 같은 트리 순서로 합친다 (§6.2) |
| 정수 +/×, max/min, and/or, argmax | 고정 조각, 조각 순서대로 합침 | 결합법칙. argmax는 "앞 조각 우선" |
| float/complex `prod`, 긴 스캔 | 고정 조각·블록 | 구조가 n에만 의존한다 (§6.6, §6.7) |
| 정렬, 누적, argmax(axis) | 레인 단위 | 레인끼리 독립이다 |
| 긴 레인 정렬 | 고정 조각 + 왼쪽 우선 병합 | 안정 정렬과 같다. `stable=False`는 항상 조각 (§8.5) |
| bool 마스크 | 고정 청크 + 누적합 | 원소마다 C 순서 자리가 정해진다 |
| 인덱스 하나짜리 scatter | 나누지 않는다 | 반복 인덱스에서 마지막 쓰기가 이겨야 한다 |
| matmul | 출력 행, 배치 | 행렬 하나·행 하나는 한 스레드가 계산한다 |

- 테스트는 1, 2, 3, 7, 10스레드에서 같은 바이트가 나오는지 확인한다 (§13.1).

### 10.6 확장성
원소 400만 개, 1스레드 대비 배율.

| 연산 | 1스레드 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | numpy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `f + g` (메모리) | 0.72ms | 1.0 | 1.4 | 1.6 | 1.7 | 1.7 | 1.6 | 1.5 | 0.73ms |
| `i + f` (버퍼 캐스트) | 1.00ms | 1.1 | 1.7 | 2.0 | 2.3 | 2.4 | 2.3 | 2.2 | 1.87ms |
| `sqrt` | 0.95ms | 1.5 | 2.7 | 3.3 | 3.7 | 4.2 | 4.1 | 4.3 | 0.92ms |
| `sum(f)` | 0.43ms | 1.4 | 2.2 | 2.6 | 3.0 | 3.5 | 3.4 | 3.4 | 0.48ms |
| `argmax(f)` | 1.81ms | 1.9 | 3.4 | 5.1 | 6.5 | 8.0 | 8.8 | 9.7 | 2.05ms |
| `var(f)` | 1.22ms | 1.5 | 2.5 | 3.3 | 4.0 | 4.7 | 4.6 | 4.7 | 1.94ms |
| `sort` 2000행 | 12.7ms | 1.9 | 3.7 | 5.3 | 6.9 | 8.5 | 9.3 | 9.7 | 25.8ms |
| int64 matmul 500³ | 34.0ms | 1.8 | 3.3 | 4.7 | 6.0 | 7.5 | 7.7 | 8.1 | 37.0ms |

- 메모리에 묶인 연산은 대역폭에서 멈춘다. `f + g`(96MB 이동)는 1스레드가 이미 초당 약 130GB를 옮기고, 1.7배가 한계다. 새 결과 배열의 첫 쓰기 페이지 폴트도 이 크기에서는 무시할 수 없다.
- 계산에 묶인 연산은 거의 코어 수만큼 빨라진다. 원소 100만 개, 10스레드에서 `exp` 122µs, `sin` 92µs다 (numpy 1.51ms, 3.83ms, `bench/merge/baseline.md`).

---

## 11. 작은 배열의 호출 비용
작은 배열에서는 계산이 아니라 호출 경로의 고정 비용이 시간을 정한다. `sample` 프로파일러로 본 비용은 이렇다.
- 고정 크기 배열(§3.3)을 통째로 복사하고 0으로 채우는 `memmove`/`memset`. `NDArray` 하나가 약 560바이트, 이터레이터가 약 1.3KB다.
- 작은 배열 하나에 `malloc` 세 번(Buffer, 데이터, `shared_ptr` 제어 블록), 그리고 `std::vector`, `std::stable_sort`의 임시 할당
- nanobind의 함수 호출 경로 자체는 약 10ns로 작다.

그래서 이렇게 한다. 32차원 고정 배열은 그대로 두고 "필요한 만큼만 만지기"로 비용을 없앤다.
1. `NDArray`의 복사·이동과 이터레이터는 실제 ndim칸만 만진다 (§3.3, §4.2).
2. Buffer는 `make_shared`로, 256B 이하의 데이터는 Buffer 안에 둔다 (§3.2).
3. 축 정렬은 insertion sort(할당 없음), 브로드캐스트는 고정 배열, 키 항목은 고정 용량 저장소를 쓴다 (§3.3, §4.2, §7.1).
4. `x[1, 2]`처럼 앞쪽 축이 Python int인 키는 키 해석을 건너뛴다 (§7.1).

| 연산 | sd | numpy |
|---|---:|---:|
| 작은 `a + b` | 180ns | 158ns |
| `x[1, 2]` | 25ns | 33ns |
| `x[1]` | 65ns | 39ns |
| `zeros(3)` | 99ns | 61ns |

(`a + b`, `x[1, 2]`는 `bench/merge/baseline.md`의 값이다.)
- `x[1]`의 남은 비용은 nanobind 인스턴스 등록과 할당, `NDArray` 복사, 메서드 호출 경로에 고르게 퍼져 있다.
- 시도하고 넣지 않은 것: ndarray 타입을 캐시하고 nanobind 저수준 API(`inst_alloc`/`inst_mark_ready`)로 인스턴스를 직접 만들기 (`x[1]` 83 → 80ns, 잡음 수준).
- 하지 않은 것: `__getitem__`을 C 슬롯(`mp_subscript`)으로 직접 등록하기. 예상 이득은 약 7ns이고, nanobind의 예외 변환을 직접 처리해야 한다.

---

## 12. Python과 C++의 경계

### 12.1 얇은 Python, 두꺼운 C++
- **C++가 하는 것**: ndarray 타입 자체, 모든 연산자와 공개 함수, 인자 파싱, 인덱스 해석, 브로드캐스팅, dtype 결정
- **Python이 하는 것** (호출 빈도가 낮은 경로만)
  - `src/stardust/__init__.py`: `_core`의 재노출과 상수(`e`, `pi`, `inf`, `nan`, `newaxis`)
  - `src/stardust/_format.py`: `repr`/`str`. 문자열 배치는 Python이 훨씬 읽기 쉽다. C++는 요약된 값(`ndarray._summary`: 원소가 1000개를 넘으면 축마다 양 끝 3개)만 넘긴다. 큰 배열을 Python 객체로 만들지 않는다.
  - `src/stardust/_results.py`: `unique_*`의 namedtuple (세 영역의 결과 namedtuple이 다 여기 있다)
- **이유**: 작은 배열에서 연산 한 번의 비용이 Python 로직이 끼면 수 µs로 는다. numpy는 약 1µs 이하다. shape, 브로드캐스팅, 인덱싱을 Python으로 짜고 C++은 커널만 맡는 방식은 읽기 쉽지만, 작은 배열에서 numpy보다 몇 배 느려진다.
- 확장 모듈은 `stardust._core` 하나다. linalg와 frame은 그 하위 모듈(`_core.linalg`, `_core.frame`)이다 (DESIGN §3.2).

### 12.2 ndarray 타입 (`bind/array_bind.cpp`)
- 직접 생성(`sd.ndarray()`)은 막는다. `isinstance` 검사를 위해 이름만 공개한다.
- **연산자**는 ufunc의 얇은 래퍼다. 받지 않는 피연산자(리스트, numpy 배열, None)에는 `NotImplemented`를 돌려 Python이 반대쪽을 시도한 뒤 TypeError를 내게 한다. 비교의 반사는 Python이 한다.
  - **`==`, `!=`는 예외다** (`op_equality`). 양쪽이 모두 `NotImplemented`면 Python은 TypeError 대신 객체 동일성으로 대체한다. 그래서 받지 않는 피연산자이면 반대쪽 타입의 비교(`tp_richcompare`)를 직접 불러 보고, 그쪽도 `NotImplemented`이면 TypeError를 낸다. `[1] == x`처럼 반대쪽이 왼쪽이어도 Python이 우리 `__eq__`를 부르므로 같다.
- **피연산자의 numpy 스칼라** (`scalar_dtype`): `numpy.float64`, `numpy.complex128`은 Python 타입의 서브클래스라 그대로 받는다. 그 밖의 numpy 스칼라는 타입 이름이 `numpy.`으로 시작할 때만 numpy의 추상 타입(`bool`, `integer`, `floating`, `complexfloating`)과 `isinstance`로 대조한다. numpy를 직접 import하지 않고, 이미 import된 모듈(`PyImport_GetModule`)에서 한 번 가져온다. `timedelta64`는 `integer`의 하위 타입이지만 `__index__`가 없어 받지 않는다. 값은 Python 스칼라와 같은 변환 함수(`__index__`, `__float__`, `__complex__`)로 읽는다. frame도 numpy 스칼라를 받는다.
- **속성의 이름**: nanobind의 `def_prop_ro`가 만든 `property`에는 이름이 없어 대입하면 "property '' of 'ndarray' object has no setter"가 나왔다. 바인딩 끝에 각 property의 `__set_name__`을 불러 이름을 넣는다.
- **in-place 연산자** `x op= y`는 `op(x, y, out=x)`다. `out=`와 같은 규칙(위로만 변환, 겹침 안전)이고 같은 객체를 돌려준다.
- **스칼라 변환 금지**: `__bool__`, `__int__`, `__float__`, `__complex__`가 모두 TypeError다. 원소가 하나여도 그렇다. 넷 다 있어야 한다: `__int__`/`__float__`가 없으면 `int()`/`float()`가 버퍼 프로토콜로 넘어가 원시 바이트를 문자열로 해석하려 한다.
- `__hash__ = None` (가변 컨테이너). `v in x`는 TypeError다: `__contains__`가 없으면 조용히 반복으로 떨어진다. `==`와 `any()`를 쓴다.
- `__array_ufunc__ = None`: numpy가 `np_array + sd_array`를 보고 우리 배열을 버퍼 프로토콜로 바꿔 numpy 결과를 돌려주는 것을 막는다. 섞으면 TypeError다.
- 리덕션 등 정해진 메서드는 모듈 함수를 x를 self로 해서 부른다 (§2.8).
- **에러 메시지의 함수 이름** (API.md §2.1): 헬퍼와 커널은 이름 없이 메시지를 쓰고, 공개 함수와 메서드의 바인딩이 `bind/errors.hpp`의 `named("zeros", &zeros)`로 `zeros(): `를 붙인다. 예외를 잡아 같은 종류로 메시지만 바꿔 다시 던지므로, 에러가 없을 때의 비용은 없다. 이미 그 이름으로 시작하는 메시지(linalg가 직접 붙인 것)는 그대로 둔다.
  - 연산자는 `named` 없이 바인딩하고, ufunc에 연산자 기호를 넘긴다(`binary<Op>(a, b, out, "+")`). dtype이 맞지 않으면 Python의 형식인 `unsupported operand dtypes for +: ...`가 된다. 같은 커널을 `sd.add`로 부르면 `add(): not supported for ...`다.
  - 프로퍼티(`.T`, `.mT`)는 `named(".T", ...)`로 `.T: `가 붙는다.
- `__getitem__`은 `None`을 받는다(`x[None]`은 `newaxis`). nanobind는 기본으로 None을 거부한다.
- 연속성 플래그는 공개 속성이 아니다. 테스트와 해설용 비공개 메서드 `_flags()`가 `c_contiguous`, `f_contiguous`, `writeable`을 dict로 돌려준다.
- nanobind의 종료 시 누수 경고는 끈다. 사용자 전역에 남은 dtype 객체 같은, 실제 누수가 아닌 경우에도 뜬다. 참조 횟수는 테스트가 직접 잰다.

### 12.3 버퍼 프로토콜
- **내보내기**: `bf_getbuffer` 슬롯을 직접 구현한다. format은 `?`, `q`, `d`, `Zd`다. 내보낸 동안 배열(과 그 Buffer)을 살려 둔다. numpy와 `memoryview`가 우리 메모리를 복사 없이 읽는다. 테스트도 이 경로로 결과를 numpy로 가져온다.
- **가져오기** (`array/ops/asarray.cpp`): `PyObject_GetBuffer`를 쓴다.
  - **format이 우리 저장 타입과 정확히 같고, 원소가 자연 정렬되어 있으면** 복사하지 않는다 (`Buffer::adopt`). 커널은 `alignof(T)` 정렬을 가정하므로 정렬되지 않은 버퍼는 복사한다. 읽기 전용 버퍼는 쓰기 금지 배열이 된다.
  - **`?` 버퍼에 0/1이 아닌 바이트가 있으면** 0/1로 정규화해서 복사한다(`copy=False`이면 에러). bool 커널은 바이트를 더해 True를 센다(§7.4, `sum`). 검사는 모든 바이트의 OR이 1 이하인지 보는 벡터 리덕션 하나다 (1e6개 8.3µs, 원소별 비교는 235µs). 공유한 뒤 바깥에서 바이트를 바꾸는 경우는 막지 않는다. 입구 한 곳에서 막는 것이 모든 bool 커널을 한 번에 지킨다.
  - **정보 손실이 없는 확장**이면 복사하면서 변환한다: `int8/16/32`, `uint8/16/32` → int64, `float16/32` → float64, `complex64` → complex128. `copy=False`이면 에러다.
  - `dtype=`을 주면 그 dtype으로 명시적 변환한다 (§2.4).
  - **그 외는 에러다**: `uint64`(int64로 손실), big-endian, 지원하지 않는 format, 0차원 버퍼.
  - 복사 결과는 원본의 메모리 축 순서를 따른다.

### 12.4 중첩 시퀀스에서 만들기
- `asarray`는 세 원천을 이 순서로 본다: ndarray, 중첩 list/tuple, 버퍼 프로토콜.
- 중첩 시퀀스는 두 패스다. 1a) 각 깊이의 첫 원소로 shape을 읽고, 1b) 모든 원소가 그 shape인지 확인하며 원소 종류를 승격하고, 2) 행 우선 순서로 값을 쓴다.

---

## 13. 테스트와 측정

### 13.1 테스트 (`tests/array/`)
- pytest + hypothesis. C++ 단위 테스트는 없다. 모든 기능을 Python에서 시험한다.
- **numpy가 정답지다.** 무작위 shape, stride, dtype 조합을 만들어 numpy 결과와 비교한다. numpy는 정답지 용도로만 설치한다.
- **비교 방법** (`tests/array/conftest.py`)
  - `assert_same`: shape, dtype, 비트까지 같다. NaN은 어떤 NaN과도 같지만 `-0.0`과 `+0.0`은 다르다. 결과는 버퍼 프로토콜로 복사 없이 가져온다.
  - `assert_ulp`: vForce 함수용. float를 순서가 보존되는 정수로 바꿔 ULP 거리를 잰다. 허용치는 8 ULP이고, 특수값(±0, ±inf, NaN)은 비트까지 같아야 한다.
  - numpy와 의미론을 일부러 다르게 한 곳(`% 0`, bool 산술, 정수 음수 지수, 0차원 없음 등)은 기대값을 따로 정의한다.
  - numpy와 순서가 다른 리덕션(§6.9)은 조각·블록 경계 근처 길이(0, 1, 7~9, 15~17, 4095~4097, 8191~8193, 16384~16385, 49169, 4만, 10만)에서 상대오차 1e-12 이내를 본다. 정수는 정확히 같아야 한다. pairwise 합의 상대오차 허용치는 1e-15다.
- **레이아웃**: 원소별 연산의 dtype 쌍 테스트는 세 레이아웃(연속, 스칼라 브로드캐스트, strided)을 모두 돈다. 연산마다 실행 경로가 레이아웃에 따라 갈리기 때문이다.
- **스레드 결정성** (`test_m8_threads.py`): 1, 2, 3, 7, 10스레드에서 같은 바이트가 나와야 한다. 블록 리덕션, `prod`, 긴 스캔, 정렬 조각, 마스크 청크 경로를 모두 포함한다. 여러 Python 스레드가 동시에 풀을 쓰는 경우와 실행 중 `set_num_threads`도 시험한다.
- **API 표면** (`test_api_surface.py`): 문서에 적힌 이름이 모두 있는지, 문서에 없는 이름을 내보내지 않는지 확인한다. `docs/api/array.md`의 시그니처 블록을 읽는다.
- **고치기 전 코드에서 실패하는지 확인한다.** 버그를 막는 테스트는 고치기 전 빌드에서 실패해야 한다. 패턴이 우연히 대칭이면(예: 길이 40000의 `[::3]`을 뒤집기) 고치기 전에도 통과할 수 있다.
- 통합 중에는 골든 출력(`bench/merge/golden/`)을 스레드 수 1, 3, 10에서 바이트로 비교한다 (ROADMAP 진행 원칙).

### 13.2 측정 방법
- **벤치마크는 numpy와 같은 자리에서 잰다.** 축은 크기, 레이아웃(연속 / 비연속 / 브로드캐스트), 스레드 수다. 여러 라운드 중 가장 빠른 값을 쓴다. 기준선은 `bench/merge/baseline.md`에 있다.
- 1스레드 수치와 기본 10스레드 수치를 따로 적는다. numpy는 BLAS 말고는 싱글스레드다.
- **원인은 프로파일러로 찾는다.** macOS `sample`로 뜨거운 함수를 보고, 기호를 남긴 빌드(`NOSTRIP`)를 `otool -tV`로 디스어셈블한다. 이 문서의 결정 여럿이 이렇게 나왔다: mutex 경합(§10.1), 메모리가 아니라 명령어 수가 문제였던 gather(§7.3), 스택에 남은 스캔 버퍼(§6.7).
- **가설은 독립 실험으로 확인한다.** 확장 모듈과 떨어진 작은 프로그램으로 한 변수만 바꿔 잰다 (vForce 검증, 임시 배열 제거, gather, 스캔의 저장 순서).
- **함정**
  - 코드가 놓인 주소: 짧고 뜨거운 루프는 64바이트 인출 블록 경계에 따라 1.7배가 갈린다 (§9.3). 버전 사이의 비교는 각 시점을 다시 빌드해서 잰다.
  - 새 메모리의 첫 쓰기 페이지 폴트: 결과는 항상 새 배열에 쓰므로, 출력 버퍼를 재사용하는 측정은 실제보다 빠르게 나온다 (§6.7).
  - 캐시 크기: 원소 100만 개(8MB)는 상당 부분 캐시에 남는다. 메모리에 묶인 연산의 확장성은 400만 개에서 따로 잰다 (§10.6). 계산 커널의 효과는 캐시 안 크기(4096개)에서 잰다 (§5.2).
  - 메모리에 묶인 연산의 1스레드 값은 측정마다 20~30% 흔들린다. 한 번의 차이로 회귀를 판단하지 않는다.
