# 옛 문서 참조 → array.md (임시)

> 코드와 테스트의 옛 문서 참조를 `docs/design/array.md`로 바꾸는 데 쓰는 표다. 바꾸기가 끝나면 지운다 (ROADMAP M1).
>
> - 옛 문서: `stardust-numpy/DESIGN.md`, `stardust-numpy/ARCHITECTURE.md` (동결된 옛 저장소)
> - "→ §N"은 `array.md §N`이다. 여러 개면 첫 번째가 주된 자리다.
> - 옛 `API.md §N` 참조는 이 표의 대상이 아니다. `docs/api/array.md`의 해당 절로 바꾼다.

## 1. 옛 DESIGN.md

| 옛 참조 | 내용 | 새 자리 |
|---|---|---|
| DESIGN §1.1 | 형태와 도구, 패키지 이름 | §1.3, §12.1 (이름은 루트 DESIGN.md §1.1) |
| DESIGN §1.2 | 타깃 환경 (M4 Pro, NEON, clang) | §1.1 |
| DESIGN §1.3 | dtype 넷, 승격 순서, Python 스칼라 | §2.1 (구현은 §3.1) |
| DESIGN §2.1 | 범위 (Tier 1, Tier 2, 제외) | §1.2 |
| DESIGN §2.2 | 0차원 배열 없음 | §2.2 |
| Z1 | 암묵적 0차원 → Python 스칼라 | §2.2 |
| Z2 | 명시적 0차원 요청 → 에러 | §2.2 |
| Z3 | 스칼라로 꺼낸 뒤는 Python 의미론 | §2.2 |
| Z4 | 범위 밖 Python int는 int64로 바뀔 때만 OverflowError | §2.2 (변환 규칙은 §2.4) |
| DESIGN §2.3 | API 일관성 규칙 | §2.8 |
| A1 | Array API 이름 | §2.8 |
| A2 | 모듈 함수가 정식, 정해진 메서드만 | §2.8 (바인딩은 §12.2) |
| C1 | 배열 인자 positional-only, 옵션 keyword-only | §2.8 |
| C2 | shape/axes는 인자 하나 (int 또는 tuple) | §2.8 |
| C3 | 별칭 없음 | §2.8 |
| C4 | 기본 axis 규칙 | §2.8 |
| C5 | 리덕션 axis/keepdims, argmin/argmax는 int axis만 | §2.8 |
| C6 | `correction` | §2.8 (구현은 §6.8) |
| C7 | `sort`의 `stable`, `descending` | §2.8 (구현은 §8.1) |
| C8 | `unique_*` 네 함수, namedtuple | §2.8 (구현은 §8.6) |
| C9 | `copy: bool \| None` | §2.8 (reshape는 §3.4) |
| C10 | `.T` 2차원만, `.mT`, `permute_dims` | §2.8 (§3.4) |
| C11 | `dot` 없음 | §2.8 |
| C12 | `where`는 인자 3개만 | §2.8 |
| C13 | 산술에서 bool → int64, 논리/비트는 bool 유지, `bool @ bool`은 int64 | §2.1 (matmul은 §9.1, 리덕션 dtype은 §6.1) |
| C14 | `flatten` 없음, `ravel`만 | §2.8 |
| DESIGN §2.4 | 수치 의미론 (전체) | §2.3–§2.7 |
| D1 | 원소 연산은 값 때문에 실패하지 않는다 | §2.3 |
| (§2.4 정수 오버플로, `//0`, `%0`, 음수 지수) | | §2.3 |
| (§2.4 float/complex IEEE, 경고 없음) | | §2.3 |
| S3 | float → int 포화, nan → 0 | §2.4 |
| S4 | complex → float는 TypeError | §2.4 |
| (§2.4 명시적 변환 규칙) | `astype`, `dtype=` 공통 규칙 | §2.4 |
| S5 | complex 순서 비교는 TypeError | §2.6 |
| S6 | `out=`, in-place, 대입의 하향 캐스팅은 에러 | §2.5 |
| (§2.4 리덕션 순서) | numpy와 같은 순서는 보장 안 함, 결정성 | §2.7, §6.9, §10.5 |
| (§2.4 비트 동일성을 포기한 경우) | 블록 리덕션, prod, 스캔, 형변환 리덕션 | §6.9 (각각 §6.5, §6.6, §6.7) |
| (§2.4 곱셈 순서와 오버플로) | `[1e200, 1e-200]*8` 예 | §6.6 |
| (§2.4 NaN 규칙) | max/min 전파, argmax 첫 NaN, sort NaN 끝 | §2.6 |
| N3 | unique에서 NaN은 각각 별개 | §2.6 (구현은 §8.6) |
| (§2.4 컴파일 옵션) | `-ffast-math` 없음, `-ffp-contract=off`, 명시적 FMA, `-fno-builtin-sin/cos` | §1.3 (명시적 FMA는 §5.3) |
| (§2.4 arange) | `start + i*step` | §2.7 |
| (§2.4 repr) | numpy와 같을 필요 없음 | §2.7 |
| DESIGN §2.5 | 행렬곱 (BLAS, ILP64, int64 i-k-j, 스칼라+언롤, bool→int64) | §9 (BLAS §9.2, int64 §9.3) |
| DESIGN §2.6 (Q8) | 원소별 함수의 구현 | §5 |
| (§2.6 vForce / libm 함수 목록, 호출 방식) | | §5.4 |
| (§2.6 NEON 함수, 손으로 짠 NEON) | | §5.1, §5.2 |
| (§2.6 complex 초월함수 libm) | | §5.3 |
| (§2.6 정수 `pow`) | | §2.3 |
| (§2.6 측정 결과, 정확도 계약 8 ULP) | | §5.4 (테스트는 §13.1) |
| DESIGN §2.7 | 최적화 포함/제외 | §1.3, §1.4, §4, §10 |
| (§2.7 `-O3`, `NOMINSIZE`) | | §1.3 |
| (§2.7 스레드풀, 단일 임계값, GIL, BLAS 예외) | | §10.1, §10.4, §10.2, §9.4 |
| (§2.7 버퍼 단위 캐스팅) | | §4.3 |
| (§2.7 temporary elision 제외) | | §1.4 |
| (§2.7 제외 목록, 정렬 휴리스틱의 범위) | | §1.4 |
| DESIGN §2.8 | M0 검증 (vForce, elision, 환경) | §5.4 (vForce 표), §1.4 (elision), §1.1·§2.7 (환경) |
| DESIGN §3 | 미확정 | 대응 없음 (아래 3절) |

## 2. 옛 ARCHITECTURE.md

| 옛 참조 | 내용 | 새 자리 |
|---|---|---|
| ARCHITECTURE §0 | 하드웨어 사실 | §1.1 |
| ARCHITECTURE §1 (R1) | Python과 C++의 경계 | §12.1 |
| ARCHITECTURE §2 | 디렉터리 구조 | 대응 없음: 루트 ARCHITECTURE.md §2, §2.1. 층 사이 의존은 §1.5 |
| ARCHITECTURE §3.1 | dtype enum, Python dtype 싱글턴 | §3.1 |
| ARCHITECTURE §3.2 | Buffer | §3.2 |
| ARCHITECTURE §3.3 (R2) | NDArray, 32차원, base, 플래그, 쓰기 금지 뷰 | §3.3 (쓰기 금지 뷰는 §3.4) |
| ARCHITECTURE §4.1 | 커널 형태, 표, 스칼라 피연산자 | §4.1 (연속 fast path는 §5.1) |
| ARCHITECTURE §4.2 | nd 이터레이터 | §4.2 |
| (§4.2 6단계 버퍼링, 4096) | | §4.3 |
| (§4.2 7단계 출력 할당) | | §4.2 |
| ARCHITECTURE §4.3 (R3) | 메모리 겹침 | §3.5 |
| ARCHITECTURE §4.4 | 리덕션 엔진 | §6 |
| (§4.4 안쪽/바깥 축) | | §6.1 |
| (§4.4 결정성) | | §10.5 |
| (§4.4 블록 리덕션) | | §6.5 |
| (§4.4 float/complex 곱) | | §6.6 |
| (§4.4 누적 스캔) | | §6.7 |
| (§4.4 NaN 규칙 FMAX/argmax) | | §6.3 |
| ARCHITECTURE §4.5 | elision 제외 | §1.4 |
| ARCHITECTURE §5 (R4) | 스레드풀 | §10 |
| (§5 spin, 세대 번호, 고정 비용) | | §10.1 |
| (§5 청크, kTaskWork, 바이트 기준) | | §10.4 |
| (§5 풀 사용 중이면 직렬) | | §10.1 |
| (§5 임계값 2^15) | | §10.4 |
| (§5 결정성 분할 규칙) | | §10.5 |
| (§5 스레드 수 API, 기본 10, E코어) | | §10.3 |
| (§5 GIL 해제) | | §10.2 |
| ARCHITECTURE §6.1 | 인덱싱 | §7 (키 해석 §7.1, gather/scatter §7.3, 마스크 §7.4, 결과 레이아웃 §7.3) |
| ARCHITECTURE §6.2 | sort / argsort | §8 (순서 §8.1, 선택 §8.2, NEON §8.3, 기수 §8.4, 병렬 §8.5) |
| ARCHITECTURE §6.3 | unique_* | §8.6 |
| ARCHITECTURE §6.4 | matmul | §9 (배치 §9.4) |
| ARCHITECTURE §6.5 (R5) | 버퍼 프로토콜 | §12.3 |
| ARCHITECTURE §7 (R6) | 테스트와 벤치마크 | §13 |

## 3. 대응이 없는 옛 항목

| 옛 항목 | 이유 |
|---|---|
| DESIGN §3 "미확정" | 미확정 항목이 없다는 이력 기록이다 |
| DESIGN §2.8의 실험 경로 (`experiments/m0_*`), 날짜 | 옛 저장소의 이력이다. 결과와 근거만 §1.4, §5.4로 옮겼다 |
| DESIGN §1.1의 `stardust-numpy`, `sdnp` | 이름이 바뀌었다 (루트 DESIGN.md §1.1, §2) |
| ARCHITECTURE §2 디렉터리 | 경로가 모두 바뀌었다. 루트 ARCHITECTURE.md §2가 대신한다 |
| ARCHITECTURE 머리말의 "R1~R6과 ✱ 확정" | 확정 절차의 이력이다 |
| ARCHITECTURE §0의 "페이지 16KB: elision 이득 계산 기준" | elision을 하지 않는다. 페이지 크기는 페이지 폴트 근거로만 남겼다 (§1.1) |
| 각 절의 "정정", "M10에서 고쳤다", "이전에는 …였다" 문장 | 이력이다. 지금 맞는 내용만 새 자리에 있다 |
| 옛 ARCHITECTURE §5의 "실수 곱은 싱글스레드", "줄이는 축은 나눌 수 없다(한계)" | 지금 코드와 다르다 (§6.5, §6.6이 지금의 모습) |

## 4. 코드에만 있는 표시 (옛 문서에 정의가 없음)

| 코드의 표시 | 뜻 | 새 자리 |
|---|---|---|
| Q1 (`ops/creation.hpp`) | 옛 문서에 정의가 없다. 문맥(C1, C2, Q1, Z1-Z4와 나란히, 인자 규칙)과 M1 커밋으로 보아 "`dtype=`은 명시적 변환" 결정으로 추정한다 | §2.4 |
| Q4 (`bind/ndarray_bind.cpp`, `test_m1_ndarray.py`) | 옛 문서에 정의가 없다. "배열은 크기와 상관없이 스칼라가 아니다" | §12.2 (규칙은 §2.2) |
| Q8 | DESIGN §2.6의 제목에 붙은 질문 번호 | §5 |
| Tier 3 item 12 | int64 기수 정렬 | §8.4 |
| Tier 3 item 13 | 긴 레인 병렬 정렬 | §8.5 |
| Tier 3 item 14 | 해시 unique 하지 않음 | §8.6 |
| Tier 3 item 15 | 작은 행렬 배치 BLAS | §9.4 |
| "M4 decision" logical bool only (`scalar_ops.hpp`) | | §2.1 |
| "M4 decision" ufunc 피연산자에 리스트 없음 (`ufunc.hpp`) | | §2.8 |
| "M4 decision" clip은 x의 dtype 유지 (`ufunc.cpp`) | | §2.5 |
| "M4 decision" `__array_ufunc__ = None` (`ndarray_bind.cpp`) | | §12.2 |
| "M5 decision" 리덕션 `dtype=`에 bool 없음 (`reduction.cpp`) | | §6.1 |
| "M5 decision" `N - correction ≤ 0`이면 NaN | | §6.8 |
| "M6 decision" Python bool은 인덱스가 아님 (`indexing.cpp`) | | §7.1 |
| "M1 rules" (`reduction.cpp`) | 명시적 변환 | §2.4 |
| "user decision 2026-11-10" (prod, 스캔, 블록 리덕션, 결과 레이아웃, 배치 BLAS, 기수 정렬) | | 각각 §6.6, §6.7, §6.5, §7.3, §9.4, §8.4 |
| "DESIGN decision in M7.9c" (`simd_sort.hpp`, `sort.cpp`) | NEON 정렬은 안정 정렬과 구별되지 않을 때만 | §8.3 |

### 마일스톤 표시 → 측정 근거의 자리
DESIGN.md §5.4에 따라 주석의 마일스톤 번호는 지우고, 측정 근거가 필요하면 아래 절을 가리킨다.

| 표시 | 새 자리 |
|---|---|
| M0 | §1.4 (elision), §5.4 (vForce 검증) |
| M1 | §2.4, §3 |
| M2 (이터레이터, 커널 시그니처) | §4.1, §4.2 |
| M4 (FMA 패턴, sincos, cabs) | §5.3, §1.3 |
| M7.1 | §1.3 |
| M7.2 | §5.1 |
| M7.3 | §5.2 |
| M7.4 | §5.1 (memcpy) |
| M7.5 | §6.2, §6.3 |
| M7.6 | §4.3 |
| M7.7 | §11 (구체적 자리는 §3.2, §3.3, §4.2, §7.1) |
| M7.8 | §7.3 (일반 경로), §7.4 (compress) |
| M7.9, M7.9a | §8.1 |
| M7.9b | §8.6 |
| M7.9c | §8.3 |
| M8 | §10 (고정 비용 §10.1, kMinInnerPiece §10.4, 결정성 §10.5, 레인 가중치 §4.5) |
| M9 | §5.4 |
| M9+ (gather_probe) | §7.3 |
| M10 (int64 matmul 언롤, 루프 위치) | §9.3 |
| "after M10" FCVTZS (`convert.hpp`), int64 min/max 8 레지스터 (`reduce.hpp`) | §2.4, §6.3 |
| M10+ | 블록 리덕션 §6.5, prod §6.6, 스캔 §6.7, any/all §6.4, clip §5.2, 바이트 기준 분할 §10.4, 인덱싱 §7.1·§7.3·§7.4 |
| "review fix" (형변환 리덕션의 블록) | §6.5 |
| "stardust-pandas's M12+ review" (`pool.cpp`, 스레드 수 atomic) | §10.3 |
