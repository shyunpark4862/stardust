# 6장. 스레드풀과 결정성

> 스레드풀 하나를 세 영역이 어떻게 나눠 쓰고, 스레드 수와 무관하게 같은 비트를 내려면 무엇을 지켜야 하나?

## 1. 약속
API.md §3의 약속은 짧다.
- `sd.set_num_threads(n, /)`, `sd.get_num_threads()` 하나가 세 영역을 함께 바꾼다. 기본값은 P코어 수(10)다.
- **결과는 스레드 수와 무관하게 비트까지 같다.** 리덕션과 정렬, 에러 메시지까지 같다.

```python
import stardust as sd

print(sd.get_num_threads())  # 10
errors = []
for bad in (0, True):
    try:
        sd.set_num_threads(bad)
    except (ValueError, TypeError) as e:
        errors.append(f"{type(e).__name__}: {e}")
print(errors[0])  # ValueError: set_num_threads(): 'n' must be between 1 and 1024
print(errors[1])  # TypeError: set_num_threads(): 'n' must be an int, not bool
```

이 장은 풀의 구조(§2–§3), 언제 나누는가(§4), 결정성 규칙(§5), 그리고 측정의 함정(§6)이다.

## 2. 풀 하나 (`parallel/pool.cpp`)
옛 세 프로젝트에는 같은 핵심을 가진 풀이 세 벌(939줄) 있었다. 지금은 한 벌(314줄)이다. 의미는 가장 나중까지 다듬은 pandas판을 따랐다 (docs/learn/M4-shared-foundation.md).

| 구조 | 이유 |
|---|---|
| **상주 워커**, 작업 하나 = 태스크 여러 개 | 작업마다 스레드를 만들지 않는다 |
| 워커와 호출 스레드가 **atomic 카운터 하나**에서 다음 태스크 번호를 가져간다 | 빠른 스레드(P코어)가 더 많이 가져갈 뿐, 일을 고정 배정하지 않는다. E코어가 전체를 기다리게 하지 않는다 |
| 호출 스레드도 태스크를 처리한다 | 스레드 하나를 놀리지 않는다 |
| 일이 끝나면 최대 100µs **spin**, 그 뒤 condition variable로 잠든다 | macOS에서 잠든 워커를 깨우는 비용은 작업당 수십 µs다. 원소 100만 개 덧셈 전체보다 크다 |
| spin 중에는 **mutex 없이** 새 일을 시작한다 | 일의 내용은 세대 번호를 올리기 전에 기록하고, 세대 번호는 release/acquire로 읽고 쓴다 |
| 태스크 안의 `parallel_for`는 직렬 | 중첩하면 스레드가 넘친다 |
| 다른 Python 스레드가 풀을 쓰는 중이면 직렬 | 기다리거나 과다 구독하지 않는다 |

- 병렬 호출 한 번의 고정 비용: 깨우기만 하면 약 25µs, spin을 넣어도 18µs(워커 9개가 한꺼번에 mutex를 잡는 경합), mutex 없이 시작하면 **약 1.5µs** (docs/design/array.md §10.1).
  - 대가는 일이 끝난 뒤 워커가 최대 100µs 동안 CPU를 쓰는 것이다. OpenMP 같은 라이브러리도 같은 절충을 한다.
- 새로 만든 워커는 **현재 세대 번호**에서 시작해야 한다. 0에서 시작하면 이미 끝난 일을 새 일로 착각해 완료 보고가 어긋나고, 다음 작업이 영원히 기다린다. 옛 numpy판과 pandas판이 각자 고친 경쟁 조건이다.
- 워커는 macOS QoS `USER_INTERACTIVE`로 올려 P코어에 머물게 한다.

## 3. GIL
- `parallel_for`는 부르면 **언제나** GIL을 놓는다. 풀이 바빠 직렬로 돌게 되어도 다른 Python 스레드는 돌 수 있어야 한다.
- `parallel_for`를 부를지(일이 기준값 이상인지)는 부르는 쪽이 정한다. 작은 호출은 부르지 않으므로 GIL을 놓고 잡는 비용이 더해지지 않는다.
- **태스크 안에서는 Python 객체를 만지지 않는다.** `NDArray`를 복사하는 것조차 `.base`(Python 객체)의 참조 횟수를 바꾸므로, 블록 뷰는 태스크 밖에서 미리 만든다. 에러 메시지(값의 `repr`이 들어간다)는 병렬 구간이 끝나고 GIL을 다시 잡은 뒤 만든다.
- 영역 사이에 하나 다른 점이 있다 (DESIGN.md §5.1).

  | | 1스레드, 큰 일 |
  |---|---|
  | 배열, linalg | `parallel_for`를 부르지 않는다. GIL을 쥔 채 돈다 |
  | frame | 기준값 이상이면 직렬 구간도 GIL을 놓는다 (`Unlocked`). 큰 CSV를 읽는 동안 다른 Python 스레드가 돈다 |

## 4. 언제 나누나: 영역의 정책
풀은 태스크를 돌리기만 한다. **언제, 얼마나 잘게 나눌지는 영역이 자기 커널로 재서 정한다** (ARCHITECTURE.md §4.1).

| 영역 | 기준 | 값 | 근거 |
|---|---|---|---|
| 배열 | `sd::kParallelThreshold` | float64 원소 2¹⁵개 (바이트로 센다) | 10스레드가 이기기 시작하는 크기: `f + g` 약 1.6만, `exp` 약 8천, `sum` 약 3.2만. 가장 늦은 `sum`에 맞췄다 |
| 배열 | `kParallelTask` | 2¹²원소 | 고정비가 1.5µs라 작은 태스크로도 고르게 나뉜다 |
| frame | `sd::frame::kParallelThreshold` | 2¹⁷행 | 2¹⁶에서도 10스레드가 `f + g`, `sum`, `isna`에서 졌다 |
| frame | `kChunkRows` | 2¹⁴행 | 64의 배수라 두 태스크가 비트맵의 같은 바이트에 쓰지 않는다 |
| linalg | 비용 모델 `a + c·w` ns | 배치 전체 추정 10µs 이상이면 나누고, 태스크 하나는 약 5µs | 행렬 함수는 원소 수가 아니라 시간으로 재야 맞는다 (9장) |

- 기준값의 **이름은 같고 값은 다르다.** 배열 코드는 `sd`에, frame 코드는 `sd::frame`에 있어서 각자 자기 값을 찾는다. 옛 풀 헤더에는 이 값이 풀 안에 있었는데, 한 풀에 한 값을 두면 둘 중 하나는 틀린다.
- **일을 바이트로 센다**(배열): bool 같은 1바이트 피연산자만 있는 루프는 8배 많은 원소를 받아야 병렬로 돈다. 원소로 세던 때 `b & b`가 10스레드에서 1스레드보다 느렸다(1000만 개 187µs 대 122µs). 바이트로 세면 38µs다.
- 스레드 수: 14스레드(E코어 포함)에서 계산에 묶인 연산은 5–27% 빨라지고 메모리에 묶인 연산은 3–12% 느려진다. 그래서 기본은 10이다 (docs/design/array.md §10.3).

## 5. 결정성 규칙
**나누는 방식은 데이터의 크기와 레이아웃으로만 정하고, 부분 결과는 고정된 순서로 합친다.** "스레드가 있을 때만 다른 경로"는 쓰지 않는다. 1스레드도 같은 구조로 계산한다.

| 연산 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 원소별 (세 영역) | 아무 축, 16K행 청크 | 원소마다 따로 계산한다. vForce도 위치에 독립이다 (4장) |
| 긴 float 합 (세 영역) | pairwise 트리의 잎 | 같은 트리 순서로 합친다 (5장) |
| 바깥 축 리덕션 (배열) | 줄이는 축의 고정 블록 | 블록 구조가 스레드 수와 무관하다 |
| 정수 합, max/min, any/all | 고정 조각, 조각 순서로 접기 | 결합법칙 |
| 정렬 (배열) | 레인, 긴 레인은 고정 조각 + 왼쪽 우선 병합 | 안정 정렬의 결과와 같다 (7장) |
| bool 마스크, 필터 | 고정 청크 + 누적합 | 원소마다 출력 자리가 정해진다 |
| linalg 배치 | 원소 묶음 | 원소 하나는 어디서 돌든 같은 계산이다(자기 작업 메모리, 같은 workspace 크기, 16바이트 정렬) (9장) |
| groupby 리덕션 | 그룹 묶음 | 그룹마다 독립이다 |

**frame의 예외**: 결과가 나누는 방식과 무관하게 **정확히** 정해지는 연산은 나누는 방식이 스레드 수를 따른다 (docs/design/frame.md §11.5).
- 대상: factorize(처음 나타난 순서의 코드), 안정 counting sort, 기수 정렬, CSV 읽기
- 이유: 나누면 패스가 늘어 1스레드가 1.5–2.5배 느려졌다. 규칙의 목적은 같은 결과이고, 이 연산들은 어떻게 나눠도 결과가 같다.
- float 합처럼 나누는 방식이 비트를 바꾸는 연산은 이 예외에 들지 않는다.

**에러도 스레드 수와 무관하다.** 여러 태스크가 동시에 실패하면 가장 앞의 위치를 보고한다.
- linalg: 배치에서 **가장 작은 실패 인덱스**. 태스크마다 첫 실패에서 멈추고, 이미 알려진 실패보다 뒤의 원소는 건너뛴다 (docs/design/linalg.md §6.5).
- frame: `parallel::FirstRow`가 청크들의 첫 실패 중 가장 앞의 행을 고른다.

```python
import numpy as np
import stardust as sd

a = sd.asarray(np.tile(np.eye(3), (1000, 1, 1)))
a[500] = 0.0
a[700] = 0.0
messages = []
for n in (1, 10):
    sd.set_num_threads(n)
    try:
        sd.linalg.inv(a)
    except sd.linalg.LinAlgError as e:
        messages.append(str(e))
print(messages[0])  # inv(): singular matrix (at batch index (500,))
print(messages[0] == messages[1])  # True
```

```python
import numpy as np
import stardust as sd

rng = np.random.default_rng(0)
df = sd.frame.DataFrame({"k": rng.integers(0, 1000, 300_000), "v": rng.standard_normal(300_000)})
results = []
for n in (1, 3, 10):
    sd.set_num_threads(n)
    g = df.groupby("k").sum()
    results.append((g["k"].tolist(), g["v"].tolist()))
print(results[0] == results[1] == results[2])  # True
```

- 테스트는 배열을 1, 2, 3, 7, 10스레드에서, frame을 1, 3, 기본 스레드에서 돌려 같은 바이트인지 본다. 골든 출력(13장)도 모든 케이스를 1, 3, 10스레드에서 돌린다. 옛 패키지로 기록할 때 스레드 수에 따라 달라진 케이스는 12,424개 중 하나도 없었다.

## 6. 측정의 함정: 프로세스마다 갈리는 시간
병렬 호출의 시간은 같은 빌드에서도 **프로세스마다** 두 갈래로 갈릴 수 있다. 통합 중에 세 번 만났다.
- `solve_tridiagonal` (1000, 3, 3), 10스레드: 프로세스마다 약 7µs 아니면 약 10.5µs (M3)
- 원소 100만 개 1스레드 `f + g`: 프로세스마다 약 148µs 아니면 약 195µs (M4, 옛 패키지도 같았다)
- `lu_solve` (1000, 3, 3), 10스레드: 프로세스마다 약 13.5µs 아니면 약 17µs (M6)

처음 둘은 원인을 확인하지 못했다. 셋째는 찾았다 (커밋 eac7ac3, docs/learn/M6-consistency.md).
- **원인: 거짓 공유(false sharing).** 배치를 나누면 태스크마다 작업 메모리(`Scratch`)를 만든다. 그 안의 아주 작은 버퍼(`lu_solve`의 3바이트 `seen`, 24바이트 `perm`)를 malloc이 여러 스레드에 **같은 128바이트 캐시 줄**로 내줄 때가 있다. 원소마다 쓰는 버퍼라 그 줄이 코어 사이를 오갔다.
  - 주소를 기록한 12개 프로세스 중 9개에서 줄이 겹쳤다. malloc의 상태가 프로세스마다 달라서, 빌드가 바뀌면 느린 쪽에 빠지는 비율이 바뀌었다(옛 패키지 56/68이 빠름, 고치기 전 HEAD 2/44). 그래서 커밋 사이의 회귀처럼 보였다.
- **고친 것**: `Scratch`의 버퍼마다 128바이트 경계에서 시작하고 크기도 128의 배수로 올린다 (`linalg/batch.hpp`, docs/design/linalg.md §6.3). 코드 배치가 다른 세 빌드에서 모두 빠른 쪽이다(12.8–13.9µs). 1스레드는 그대로다.
- **교훈**
  - 짧은 병렬 호출 하나는 "같은 자리에서 번갈아" 재도 프로세스 몇 개로는 판단할 수 없다. 프로세스를 여러 번 새로 띄워 분포를 본다 (`bench/merge/m6/procs.py`).
  - 스레드마다 쓰는 메모리는 캐시 줄 단위로 떼어 둔다. frame의 청크가 64행의 배수인 것(비트맵 바이트를 나누지 않는다)도 같은 원리다.

## 더 읽을 곳
- [docs/design/array.md §10.1](../design/array.md) 풀의 구조, [docs/design/array.md §10.2](../design/array.md) GIL, [docs/design/array.md §10.4](../design/array.md) 기준값, [docs/design/array.md §10.5](../design/array.md) 결정성 규칙, [docs/design/array.md §10.6](../design/array.md) 확장성
- [docs/design/linalg.md §8.5](../design/linalg.md) linalg의 결정성, [docs/design/linalg.md §6.3](../design/linalg.md) 작업 메모리
- [docs/design/frame.md §11.1](../design/frame.md) frame의 정책, [docs/design/frame.md §11.5](../design/frame.md) frame의 결정성, [docs/design/frame.md §11.6](../design/frame.md) 에러 위치
- [API.md §3](../../API.md) 스레드와 결정성, [DESIGN.md §5.1](../../DESIGN.md) 스레드 결정
