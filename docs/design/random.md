# 난수 영역 설계 (sd.random)

> 상태: **T3 완료·승인** (2026-11-15). 결정은 §1(R1–R11).

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
| R4 범위 | 생성기 + 기본 도구만: `random`([0, 1) 균등), `integers`, `choice`, `permutation`. 분포별 표본은 stats에 (R1과 함께 정함: 중복이 없고, 매개변수 규칙이 하나다). `shuffle`은 R9로 뺐다 |
| R5 생성기 (T3, 2026-11-15) | Philox4×64-10 (§2) |
| R6 시드 | int(0 이상, 크기 제한 없음)·그런 int의 시퀀스·None(OS 엔트로피). 키는 numpy SeedSequence와 같은 해시로 만든다(그래서 키는 numpy Philox의 키와 같다; 값은 계수기 쓰는 법이 달라 다르다) |
| R7 spawn·상태 | `spawn(n)`(독립 생성기), 상태 읽기·쓰기(`state`), 피클·복사 모두 |
| R8 공유 | 한 Generator를 여러 Python 스레드가 함께 쓰면 호출마다 자기 구간을 원자적으로 예약한다. 깨지지 않고 같은 난수를 두 번 쓰지 않지만, 어느 스레드가 어느 구간을 받는지는 실행 순서에 따른다. 재현하려면 스레드마다 `spawn` |
| R9 섞기 | `shuffle` 없이 `permutation`만. stardust에는 인자를 제자리에서 바꾸는 함수가 없다(값을 바꾸는 것은 `out=`, `+=`, 인덱스 대입). 제자리가 필요하면 `x[...] = rng.permutation(x)`. 방법은 재서 정하고, 결과는 스레드 수와 무관하다 |
| R10 인자 | `integers`의 `low`·`high`는 int 또는 int64 배열이고 `shape`와 함께 브로드캐스트한다(numpy와 같은 규칙). `choice`의 `p`는 음수·NaN이 없고 합이 1에서 sqrt(eps) 안이어야 하며(numpy와 같다), 합으로 나눠 쓴다 |
| R11 기본값 (T3 승인, 2026-11-15) | `integers`의 `shape=None`은 `low`·`high`를 브로드캐스트한 모양이다(둘 다 스칼라면 Python `int`). `default_rng`에 Generator를 주면 같은 객체를 그대로 돌려준다: 시드든 Generator든 `rng = sd.random.default_rng(rng)` 한 줄로 받고, 넘긴 생성기의 상태가 그대로 이어진다(복사본을 주면 같은 난수가 반복된다) |

## 2. 생성기 (`csrc/random/philox.hpp`)
- **Philox4×64-10** (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC 2011; Random123). 키(128비트)와 계수기(256비트)에서 64비트 넷을 만드는 함수다. 상태가 없어서 어느 계수기 값이든 바로 계산된다.
- 정확성: Random123의 알려진 답(KAT) 세 벡터와 비트까지 같고, numpy의 Philox와도 같다: 같은 키의 numpy Philox 수열은 우리 수열을 원소 4개(한 블록) 민 것이다(numpy는 계수기 1에서 시작한다. tests/random/test_random.py).
- **블록 넷을 함께**: 한 블록은 곱셈 20번이 이어지는 의존 사슬이라 지연 시간에 묶인다(블록당 약 12 ns). 독립인 블록 넷의 라운드를 섞어 곱셈기를 채운다(`philox4`). 균등 난수 10⁶개 2.98 → 1.14 ms. 블록마다 비트는 같다.

## 3. 결정성: 원소의 부분 수열
- Generator는 키와 위치 `pos`(지금까지 쓴 원소 수)를 든다. 크기 n인 호출의 원소 i는 스트림의 원소 e = pos + i이고, 호출이 끝나면 `pos += n`이다.
- **원소 e의 단어**: 첫 단어는 계수기 (e / 4, 0, 0, 0) 블록의 e mod 4번째 단어다(이웃한 원소 넷이 블록 하나를 나눈다). 기각 표본에 더 필요한 단어는 계수기 (e, j, 0, 0), j ≥ 1 블록에서 차례로 받는다. 두 계수기는 둘째 단어(0과 1 이상)로 갈려서 만나지 않는다.
- 그래서 원소 하나의 값은 (키, e)만으로 정해진다. 작업을 어떻게 나누든, 스레드가 몇 개든 같은 값이다. 호출을 나누어 불러도 앞의 원소들은 같다.
- 위치의 예약은 잠금 안에서 한다(R8). 계산은 잠금 밖이다.

## 4. 기본 도구 (`csrc/random/random.cpp`, `src/stardust/random/__init__.py`)
| 함수 | 방법 |
|---|---|
| `random` | 첫 단어의 위 53비트 × 2⁻⁵³ (numpy와 같은 변환) |
| `integers` | Lemire의 거의 나눗셈 없는 방법(r = high − low, w·r의 위 64비트; 아래 64비트가 2⁶⁴ mod r보다 작으면 기각하고 그 원소의 추가 단어로 다시). 범위가 하나인 흔한 호출은 2⁶⁴ mod r을 한 번만 구한다(범위 2⁶³에서 원소 절반이 나눗셈을 했다: 5.7 → 1.5 ms). 64비트 부호 없는 산술이라 int64의 어떤 쌍이든 정확하다 |
| `permutation` | Fisher–Yates. 걸음 i는 원소 pos + i의 단어로 [i, n)을 고른다. **재서 골랐다**: 무작위 64비트 키를 정렬하는 방법(병렬)은 10⁶개 1스레드 8.6 ms, 10스레드 3.8 ms였고 Fisher–Yates(순차)는 3.0 ms다 |
| `choice` 복원 | 균등은 `integers`, 가중치는 누적합(마지막을 1로 나눔)과 `searchsorted(side='right')`(확률 0인 원소는 뽑히지 않는다) |
| `choice` 비복원 | 균등은 Fisher–Yates의 앞 k걸음. k ≤ n/16이면 건드린 항목만 해시 맵에 둔다(O(k): 10⁶개에서 1000개 0.03 ms; 처음에는 전체를 섞어 8.9 ms였다). 가중치는 Efraimidis–Spirakis: 원소마다 E/p(E는 표준 지수)를 만들어 작은 k개(확률 0은 inf) |
- dtype: 실수는 float64, 정수는 int64.

## 5. 시드 (R6)
- numpy `SeedSequence`(풀 크기 4)를 Python으로 옮겼다(numpy에 기대지 않으려고). 키는 `generate_state(2, uint64)`라 numpy Philox의 키와 같다. `spawn`도 numpy와 같은 키다(spawn_key에 번호를 붙인다). 테스트가 numpy와 비교한다.
- `None`은 `secrets.randbits(128)`이다.

## 6. 측정
- bench/random/t3_speed.md: numpy Generator(PCG64, Philox)와 1·10스레드 비교. 요약은 docs/api/random.md §1.8.
- 테스트(tests/random/test_random.py): KAT, 계수기 배치, SeedSequence·spawn의 numpy 일치, 스레드 1·3·10과 나눈 호출의 같은 비트, 경계, 통계 검사(KS, 카이제곱: 정수·기각 구간·섞기 24가지·가중치), 비복원, 상태·피클·복사, 공유 Generator의 구간 예약, 에러 메시지.
- **2026-11-18 정리**: 메서드는 키와 위치를 한 잠금 안에서 함께 읽는다(전에는 키를 잠금 밖에서 읽어, 다른 스레드의 `state` 대입과 겹치면 옛 키와 새 위치가 섞일 수 있었다). 가중치 비복원 추출의 키 E/p는 double 최댓값에서 자른다: p가 약 2e-307 아래이면 inf로 넘쳐 확률 0인 원소(inf)와 구별되지 않았고, 인덱스 순서로 0인 원소가 뽑혔다. 잘린 원소끼리는 인덱스 순서다(그 확률을 다 합쳐도 1e-300 아래다). `integers`의 치우침 테스트는 3·2⁶¹ 범위의 값을 3으로 나눈 나머지로 본다(기각을 빼면 3/8, 3/8, 2/8이 된다. 전의 구간 셋 세기는 기각이 없어도 통과했다). 에러 메시지의 접두사를 메서드 이름으로 맞췄다(`choice(2**63)`이 `integers():`로, `shape=()`가 메서드마다 다르게, `.state`가 `.state:`와 `.state():`로 섞여 나왔다).
