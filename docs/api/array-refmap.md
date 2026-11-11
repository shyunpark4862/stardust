# 배열 문서 참조 대응표 (임시)

> 코드 주석과 테스트의 옛 참조("API.md §9", "Z1" 등)를 새 문서로 바꾸는 데 쓴다. 참조를 모두 바꾸면 이 파일을 지운다.

- 옛 문서: `stardust-numpy/API.md`, `DESIGN.md`(규칙 ID), `ARCHITECTURE.md`(R ID)
- 새 문서: `docs/api/array.md` (아래 표의 `§`는 모두 이 문서의 절)
- "인용 수"는 옛 저장소의 `src/ csrc/ tests/ bench/`에서 센 값이다.

## 1. 옛 API.md의 절

| 옛 참조 | 내용 | 새 절 | 인용 수 |
|---|---|---|---|
| API.md §0 | 시그니처 규칙 | §1.1 (dtype 객체는 §0.1, 0차원은 §1.2) | 1 |
| API.md §1 | 모듈 상수와 dtype | §0.1 (dtype), §2 (상수) | 0 |
| API.md §2 | 배열 생성 | §3 (함수 §3.1, `asarray` §3.2, 타입 추론 §3.2) | 1 |
| API.md §3 | shape 조작과 뷰 | §4 (`concat`/`stack` §4.2, `astype` §4.3) | 1 |
| API.md §4 | 인덱싱 | §5 (기본 §5.1, 고급 §5.2, 대입 §5.3) | 2 |
| API.md "§고급 인덱스" (ARCHITECTURE §6.1에서 인용) | 고급 인덱스의 결과 레이아웃 | §5.2, §1.9 | 0 |
| API.md §5 | 원소별 함수 | §6 (피연산자 §6.2, dtype §6.3, 함수별 §6.4) | 1 |
| API.md §6 | 리덕션 | §7 | 1 |
| API.md §7 | 정렬, 집합, 선형대수 | §8 (정렬 §8.2, `unique_*` §8.3), §9 (`matmul`) | 3 |
| API.md §8 | ndarray | §10 | 0 |
| API.md §9 | 실행 설정 (스레드) | §0.3 | 1 |

## 2. 옛 DESIGN.md의 규칙 ID

| ID | 내용 | 새 절 | 인용 수 |
|---|---|---|---|
| Z1 | 암묵적 0차원 → Python 스칼라 | §1.2 | 10 |
| Z2 | 명시적 0차원 요청은 에러 | §1.2 | 3 |
| Z3 | 스칼라로 꺼낸 뒤에는 Python 의미론 | §1.2 | 0 |
| Z4 | 범위 밖 Python int는 int64로 바뀔 때만 OverflowError | §1.5 ("int64 범위를 벗어난 Python int") | 7 |
| A1 | Array API 이름 | §1.1 | 0 |
| A2 | 모듈 함수가 정식, 메서드는 정해진 목록 | §1.1, §10.2 | 2 |
| C1 | 배열 인자 positional-only, 옵션 keyword-only | §1.1 | 2 |
| C2 | shape, axes는 인자 하나(int 또는 tuple) | §1.3 | 4 |
| C3 | 함수 하나에 이름 하나 | §1.1 | 0 |
| C4 | 기본 axis 규칙 (누적은 ndim>1이면 필수) | §1.3 ("기본 axis"), §7.7 | 2 |
| C5 | 리덕션 axis는 int/tuple, `argmin`/`argmax`는 int만 | §7.2 | 2 |
| C6 | `var`/`std`의 `correction` | §7.4 | 0 |
| C7 | `sort`의 `stable`, `descending` | §8.2 | 0 |
| C8 | `unique_*` 네 함수, namedtuple | §8.3 | 2 |
| C9 | `copy: bool \| None` | §1.8 | 1 |
| C10 | `.T`는 2차원만, `.mT`, `permute_dims` | §10.1 | 2 |
| C11 | `dot` 없음 | §9 | 0 |
| C12 | `where`는 인자 3개 | §6.4 | 0 |
| C13 | 산술에서 bool → int64 | §1.4 | 11 |
| C14 | `flatten` 없음, `ravel`만 | §4.1 | 0 |
| D1 | 원소 연산은 값 때문에 실패하지 않는다 (정수 0 나누기, wrap, 음수 지수) | §1.6 | 6 |
| S3 | float → int 포화, NaN → 0 | §1.5 | 2 |
| S4 | complex → float 에러 | §1.5 | 1 |
| S5 | complex 순서 비교는 TypeError | §1.6 ("complex에는 순서가 없다") | 6 |
| S6 | `out=`, in-place의 하향 캐스팅은 에러 | §1.5 ("암묵적 변환") | 21 |
| N3 | `unique`에서 NaN은 각각 별개 | §8.3 | 2 |
| "M1의 `dtype=` 규칙" / "명시적 변환 규칙" | `astype`, `dtype=`의 손실 변환 | §1.5 ("명시적 변환") | (문장으로 인용) |

## 3. 옛 ARCHITECTURE.md의 R ID 중 API에 해당하는 것

| ID | 내용 | 새 절 | 인용 수 |
|---|---|---|---|
| R2 | 최대 32차원 | §1.3 | 0 |
| R3 | 메모리 겹침: 입력을 다 읽은 뒤 쓴 것과 같다 | §1.7 (구현 방식은 design 문서) | 4 |
| R4 | 스레드 수 API | §0.3 | 0 |
| R5 | 버퍼 가져오기 규칙 | §3.3 | 1 |

## 4. 대응하지 못한 것

| 참조 | 이유 | 제안 |
|---|---|---|
| Q1 (`csrc/ops/creation.hpp`: "argument rules (C1, C2, Q1, Z1-Z4)") | 지금 남은 옛 문서 어디에도 정의가 없다 (문서가 git에 없어서 이력도 없다). 생성 함수의 인자 규칙 목록에 있으므로 dtype 객체만 받는 규칙으로 추정한다 | §0.1 또는 §1.1. 확인이 필요하다 |
| Q4 (`ndarray_bind.cpp`, `test_m1_ndarray.py`: "an array is never a scalar") | 정의가 없다. 주석의 내용으로는 배열 → 스칼라 변환 금지다 | §10.4 |
| R1 (`bench/m3_views.py`: "thin Python / thick C++") | API가 아니라 구조 결정이다 | `docs/design/array.md` |
| R6 | 테스트와 벤치마크 방침. API가 아니다 | `docs/design/array.md` |
| DESIGN §1.3 (1회) | dtype과 승격 | §0.1, §1.4 |
| DESIGN §2.4 (11회) | 수치 의미론. API에 해당하는 부분만 있다 | 값 규칙은 §1.5, §1.6, 리덕션 순서는 §7.8. 컴파일 옵션(FMA, `-ffp-contract`)과 구현 근거는 `docs/design/array.md` |
| DESIGN §2.5, §2.6, §2.7 (3, 4, 1회) | 행렬곱, 원소별 함수, 최적화의 구현 | `docs/design/array.md`. 단, 원소별 함수의 정확도 계약은 §6.4 ("정확도") |
| ARCHITECTURE §1, §3.2, §3.3, §4.1, §4.2, §4.4, §5, §6.2–§6.5 | 내부 구조 | `docs/design/array.md`. 단, §3.3의 `base` 규칙은 §10.1, §6.5의 버퍼 규칙은 §3.3과 §10.5 |
| M0–M12 등 마일스톤 번호 | 문서 참조가 아니라 이력이다 | DESIGN §5.4에 따라 지운다 |
| `C99`, `L1`, `L2`, `T3`, `I64`, `F64` 등 | 규칙 ID가 아니다 (C99 표준, 캐시, 변수 이름, 타입 이름) | 그대로 둔다 |
