# 난수 영역 설계 (sd.random)

> 상태: **초안** (2026-11-15). 큰 결정은 사용자가 정했다(§1). 시작 전에 물을 것은 §6에 있다.
>
> **구현은 사용자가 시작을 지시한 뒤에만 진행한다.**

`sd.random`은 난수의 바탕이다. 분포별 표본은 여기가 아니라 `sd.stats`의 각 분포(`X.sample(shape, rng=rng)`, [stats.md](stats.md) §4)에 있다.

- 공개 함수와 시그니처는 [docs/api/random.md](../api/random.md)에 있다.
- 쓰는 곳: `sd.stats`의 표본, `sd.stats.tests`의 순열·부트스트랩, frame의 표본 뽑기·섞기, 테스트 데이터.

---

## 1. 결정 기록 (사용자, 2026-11-15)
| 번호 | 결정 |
|---|---|
| R1 위치 | 별도 모듈 `sd.random`. stats 안이 아니다 |
| R2 생성기 | **계수기형 자체 설계**: 원소 번호에서 바로 난수를 만든다. 모든 연산이 병렬이고 스레드 수와 무관하게 같은 값이다. numpy와 값은 다르다(numpy의 비트 호환 대신 고른 것) |
| R3 API | Generator 객체: `rng = sd.random.default_rng(seed)`. 전역 함수와 전역 상태가 없다 |
| R4 범위 | 생성기 + 기본 도구만: `random`([0, 1) 균등), `integers`, `choice`, `permutation`, `shuffle`. 분포별 표본은 stats에 (R1과 함께 정함: 중복이 없고, 매개변수 규칙이 하나다) |

## 2. 생성기
- 후보: **Philox4×64-10** (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC 2011; Random123). 키(128비트)와 계수기(256비트)에서 64비트 넷을 만드는 함수다. 상태가 없어서 어느 계수기 값이든 바로 계산된다.
- 정확성: Random123의 알려진 답(KAT) 벡터와 비트까지 맞춘다.
- 다른 후보(마일스톤에서 정한다): Philox4×32-10(32비트 곱셈, NEON 친화), Threefry, 다른 해시형.

## 3. 결정성: 원소의 부분 수열
- Generator는 키(시드에서)와 위치 `pos`(지금까지 쓴 원소 수)를 든다.
- 크기 n의 호출에서 원소 i(0 ≤ i < n)는 계수기 (`pos + i`, j)를 쓴다. j는 그 원소 안에서 몇 번째 난수인지다(기각 표본이 다시 뽑을 때 늘어난다). 호출이 끝나면 `pos += n`.
- 그래서 원소 하나의 값은 (키, 원소 번호)만으로 정해진다. 이터레이터가 작업을 어떻게 나누든, 스레드가 몇 개든 같은 값이다(DESIGN.md의 결정성 규칙).
- 같은 시드와 같은 호출 순서면 같은 값이다. 호출을 나누어 부르면(n을 두 번에) 앞의 원소들은 같다.

## 4. 기본 도구
| 함수 | 방법(안) |
|---|---|
| `random` | 64비트의 위 53비트 × 2⁻⁵³ (numpy와 같은 변환) |
| `integers` | Lemire의 거의 나눗셈 없는 방법(기각은 원소 안에서) |
| `permutation`, `shuffle` | 원소마다 무작위 키를 만들어 정렬(병렬, 결정적, O(n log n)) 또는 Fisher–Yates(순차). 마일스톤에서 재서 정한다 |
| `choice` | 복원은 `integers`, 비복원은 `permutation`의 앞부분, 가중치는 누적합과 이분 탐색 |
- dtype: 실수는 float64, 정수는 int64. 범위·인자 규칙은 docs/api/random.md.

## 5. 측정과 테스트
- KAT 벡터, 스레드 1·3·10에서 같은 비트, 호출을 나눈 것과 한 번에 부른 것의 일치.
- 통계 검사: 균등성(KS·카이제곱), 정수의 치우침, 섞기의 균등성. 필요하면 외부 검사기(PractRand 등)로 한 번 잰다.
- 속도: numpy Generator(PCG64, Philox)와 1·10스레드로 비교한다.

## 6. 시작하기 전에 물을 것
- 생성기의 최종 선택(Philox4×64-10 / 4×32-10 / 기타)과 라운드 수
- 시드: 받는 타입(int, 정수 배열, None = OS 엔트로피), 시드에서 키를 만드는 방법(numpy의 SeedSequence 같은 해시)
- 독립 생성기 나누기(`spawn`), 상태 읽기·쓰기, 피클
- 같은 Generator를 여러 Python 스레드가 함께 쓸 때의 규칙(위치 `pos`의 갱신)
- `permutation`·`shuffle`의 방법(정렬 대 Fisher–Yates), `shuffle`이 제자리인지
