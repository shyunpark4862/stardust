# 5장. 리덕션과 pairwise 합

> 합계 하나에도 왜 여러 전략이 필요하고, 부동소수점 덧셈의 순서는 왜 그렇게 중요한가? 세 영역은 어떻게 같은 합을 내나?

## 1. 덧셈의 순서가 결과다
부동소수점 덧셈은 결합법칙이 성립하지 않는다. `(a + b) + c`와 `a + (b + c)`는 마지막 비트가 다를 수 있다.
- 그래서 "합"은 값 하나가 아니라 **덧셈의 순서**까지 정해야 하나로 정해진다.
- 순서를 고르는 기준은 셋이다: 빠른가(사슬 하나에 묶이지 않는가), 정확한가(오차가 얼마나 자라는가), 기준(numpy)과 같은가.
- 그리고 이 라이브러리에는 넷째 기준이 있다: **스레드 수와 무관한가** (6장).

```python
import numpy as np
import stardust as sd

f = np.random.default_rng(0).standard_normal(1_000_000)
seq = 0.0
for v in f:          # 왼쪽부터 차례로
    seq += v
print(seq == float(np.sum(f)))  # False
print(sd.sum(sd.asarray(f)) == float(np.sum(f)))  # True
```

## 2. 리덕션의 구조
리덕션도 3장의 이터레이터 위에서 돈다 (`array/kernels/reduce.hpp`).
- 피연산자는 둘이다: 입력, 그리고 줄이는 축의 길이를 1로 만든 출력을 입력 shape으로 브로드캐스트한 것(줄이는 축의 stride가 0). 커널은 `out = f(out, x)`를 한다.
- 이터레이터가 넘긴 구간의 모양이 전략을 정한다.
  - **출력 stride가 0**이면 구간 하나를 누산기 하나로 접는다(안쪽 축 리덕션, `sum(axis=-1)`). 여기에 pairwise 합을 쓴다.
  - **출력 stride가 0이 아니면** 결과 행에 입력 행을 원소별로 더한다(바깥 축 리덕션, C 순서의 `sum(axis=0)`). 남는 축 방향으로 벡터화된다.
- 시작값: 합은 `+0`, 곱은 1, `max`는 dtype의 최솟값. 첫 원소가 시작값을 대신하므로 `-inf`, `-0.0`이 정확히 남는다. `max`/`min`은 항등원이 없어 빈 입력이면 ValueError다.

```python
import stardust as sd

print(sd.sum(sd.zeros(0)), sd.mean(sd.zeros(0)))  # 0.0 nan
try:
    sd.max(sd.zeros(0))
except ValueError as e:
    print(e)  # max(): undefined for an empty array (or along an empty axis)
```

## 3. pairwise 합
numpy의 `DOUBLE_pairwise_sum`과 같은 알고리즘, 같은 순서다 (docs/design/array.md §6.2).

```
n < 8     : 차례로 더한다
n ≤ 128   : 누산기 8개가 8칸 간격으로 원소를 맡는다
            → ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)), 나머지를 차례로
그 밖     : 반(8의 배수로 내림)으로 나눠 재귀한다
합 = 0.0 + 트리 전체                      ← numpy reduce는 항등원 0.0에서 시작한다
```

| 얻는 것 | 이유 |
|---|---|
| 속도 5.6배 (순차 합 대비) | 덧셈 지연에 묶인 사슬 하나가 8개가 된다 |
| 정확도 | 오차가 n이 아니라 log n에 비례해 자란다. 1e7개의 상대 오차 3.7e-14 → 0 |
| numpy와 비트까지 같음 | 알고리즘과 순서가 같다 |

- complex는 numpy가 double 개수로 세므로 잎 하나가 complex 64개, 누산기 4개다.
- `0.0 +`로 시작하므로 `-0.0` 아홉 개의 합은 `+0.0`이다(numpy와 같다).

### 3.1 병렬로 나눠도 같은 비트
- 재귀 트리의 모양은 n에만 의존한다.
- 그래서 4096개 이하의 부분 트리(잎)를 태스크로 병렬 계산하고, **같은 트리를 따라** 합친다. 덧셈 하나하나가 직렬과 같은 자리, 같은 순서다.
- 결과: 스레드 수와 상관없이 numpy와 비트가 같다.

```python
import numpy as np
import stardust as sd

f = np.random.default_rng(0).standard_normal(1_000_000)
x = sd.asarray(f)
sums = []
for n in (1, 3, 10):
    sd.set_num_threads(n)
    sums.append(sd.sum(x))
print(len(set(sums)) == 1, sums[0] == float(np.sum(f)))  # True True
```

### 3.2 정의는 한 벌, 항을 가져오는 방식은 영역마다
옛 세 프로젝트에는 pairwise 합이 세 벌 있었다(배열의 리덕션 커널 안, linalg 220줄, pandas 110줄). 지금은 `base/pairwise.hpp`(144줄) 한 벌이다.
- **비트를 정하는 부분만** 공통이다: 트리 모양(`split`), 잎의 합(`leaf`, `leaf_complex`), 병렬 계획과 합치기(`plan`, `combine`).
- **항을 가져오는 방식**은 영역에 남겼다.

  | 영역 | 항을 가져오는 법 | 잎 태스크 크기 | 쓰는 곳 |
  |---|---|---|---|
  | 배열 | 접근자 `get(i)` | 2¹² | `sum`, `mean`, `var`의 둘째 패스 |
  | linalg | 잎(최대 128개)을 버퍼에 먼저 채운 뒤 더한다. 레이아웃과 무관하게 더하는 루프가 하나라 벡터화된다 | 2¹² | `vecdot`, `trace`, 노름 |
  | frame | 접근자. NA 행은 값이 0이라 분기 없이 함께 더한다 | 2¹⁴ | float `sum`, `mean`, `var`, groupby의 그룹 합 |

- 잎 크기가 달라도 같은 트리 위의 다른 자르기라서 비트는 같다.
- 그래서 같은 항이면 세 영역의 합이 비트까지 같다: `sd.linalg.trace(x) == sd.sum(sd.linalg.diagonal(x))`, `s.sum() == sd.sum(s.to_array())`.

```python
import stardust as sd

x = sd.reshape(sd.arange(9.0) / 7.0, (3, 3))
print(sd.linalg.trace(x) == sd.sum(sd.linalg.diagonal(x)))  # True
s = sd.frame.Series([0.1, 0.2, 0.3, 0.4])
print(s.sum() == sd.sum(s.to_array()))  # True
```

### 3.3 한곳으로 옮기기만 해도 느려질 수 있다
정의를 한 벌로 합친 직후(M3) 배열의 1스레드 `sum(f)`가 110µs에서 152µs로 느려졌다 (docs/learn/M3-linalg-boundary.md).
- 원인: 배열이 트리 → 람다 → 잎 → 접근자 람다의 여러 겹으로 부르게 되었는데, 잎의 합(원소마다 도는 루프)이 끝까지 인라인되지 않았다.
- 고친 것: `pairwise::leaf`에 `[[gnu::always_inline]]`. 110µs로 돌아왔고 비트는 그대로다.
- linalg의 짧은 합(3×3 행렬의 열 합)도 느려졌다. "잎 하나면 트리를 부르지 않는다"는 빠른 길을 되살려 고쳤다.
- 교훈: 골든 출력(13장)은 비트만 본다. 성능은 따로 재야 한다.

## 4. 바깥 축 리덕션은 블록으로
C 순서 (1000, 1000)의 `sum(axis=0)`은 줄이는 축이 바깥에 있다. 출력이 걸친 축만 나눌 수 있으므로(6장) 태스크 하나로 돈다.
- 그래서 **줄이는 축을 고정 개수의 블록으로 나눈다** (docs/design/array.md §6.5). 블록마다 출력 사본에 접고, 사본들을 블록 순서대로 합친다.
- 블록 구조는 shape, 레이아웃, dtype만으로 정한다. 1스레드도 같은 구조로 계산한다. 그래서 스레드 수와 무관하다.
- 대가: 덧셈 순서가 numpy와 달라 마지막 비트가 다를 수 있다. 정확도는 더 좋다(한 사슬에 더해지는 항이 줄어든다).
- 효과: 10스레드 115µs → 32µs (numpy 123µs).

```python
import numpy as np
import stardust as sd

f = np.random.default_rng(0).standard_normal((1000, 1000))
m = sd.asarray(f)
print(sd.sum(m, axis=1).tolist() == f.sum(axis=1).tolist())  # True
a0, n0 = np.asarray(sd.sum(m, axis=0)), f.sum(axis=0)
print(bool((a0 == n0).all()), float(np.max(np.abs(a0 - n0) / np.abs(n0))) < 1e-12)  # False True
```

## 5. 곱과 누적
- **float `prod`**: 왼쪽부터 곱하면 곱셈 지연(4사이클)에 묶인 사슬 하나다. 4096원소 조각 안에서 누산기 8개를 쓴다. 1스레드 7.8배.
  - 곱하는 순서가 바뀌면 오버플로가 일어나는 시점도 바뀐다. 마지막 비트에 그치지 않는 입력이 있다.

```python
import numpy as np
import stardust as sd

v = [1e200, 1e-200] * 8
print(sd.prod(sd.asarray(v)), float(np.prod(v)))  # nan 1.0
```

  - numpy는 왼쪽부터 곱해 중간값이 1과 1e200 사이를 오간다. sd에서는 누산기 하나가 짝수 번째 둘을 맡아 `1e200 · 1e200 = inf`, 다른 하나가 홀수 번째 둘을 맡아 `0`이 되고, 합칠 때 `inf · 0 = nan`이다. 문서에 적은 대가다 (docs/design/array.md §6.6).
- **누적 스캔** (`cumulative_sum`): float 레인이 8192원소 이상이면 4096원소 블록으로 2단계 스캔을 한다. 블록마다 따로 스캔(병렬) → 블록 끝값으로 carry(직렬) → carry를 더함(병렬). 1스레드 354µs, 10스레드 70µs, numpy 2.2ms.
- 정수 합과 곱, `max`/`min`, `any`/`all`은 순서와 무관하다. 그래서 numpy와 언제나 같다.
- `any`/`all`은 4096원소 블록마다 답이 정해졌는지 보고 멈춘다. 앞에 True가 있는 `any`가 0.1µs다.

## 6. mean, var, std
- `mean`: 합을 개수로 나눈다. 정수 입력은 버퍼 캐스팅(3장 §4)으로 float64가 되어 합해진다.
- `var`: 두 패스다. 평균을 구한 뒤, 편차 제곱합을 임시 배열 없이 커널 하나로 접는다. `N - correction`으로 나눈다.
- **자유도 인자는 `correction`, 기본값은 0이다** (Array API, numpy). 세 영역이 같다. 옛 pandas판은 `ddof=1`이었고, 라이브러리 전체의 기준을 하나로 정하며 바꿨다 (DESIGN.md §8.2).

```python
import stardust as sd

x = sd.asarray([1.0, 2.0, 3.0, 4.0])
print(sd.var(x), sd.var(x, correction=1))  # 1.25 1.6666666666666667
s = sd.frame.Series([1.0, 2.0, 3.0, 4.0])
print(s.var(), s.var(correction=1))  # 1.25 1.6666666666666667
```

- **frame의 int64 평균은 배열의 합을 그대로 부른다.** `sd.sum(x, dtype=sd.float64)`를 개수로 나눈다. 같은 계산을 frame에 다시 쓰면 배열의 덧셈 순서(4096원소 버퍼 단위)를 흉내 내야 하는데, 부르면 순서가 저절로 같다 (12장).

## 7. numpy와 비트가 다른 곳
결정성(스레드 수와 무관)은 모든 경우에 지킨다. numpy와의 비트 동일성은 아래에서만 내줬다. 이유는 모두 같다: numpy의 순서로는 나눌 수 없거나 사슬 하나에 묶인다.

| 경우 | 바뀐 순서 | 얻은 것 |
|---|---|---|
| 바깥 축 리덕션 (`sum(axis=0)` 등) | 줄이는 축의 고정 블록 | 10스레드 3.6배 |
| float/complex `prod` | 조각 안에 누산기 8개 | 1스레드 7.8배 |
| 8192원소 이상의 float 누적 | 블록 2단계 스캔 | 1스레드 1.4배, 10스레드 7배 |
| 형변환이 있는 리덕션 (`mean(int64)`) | 4096원소 버퍼, 고정 블록 | 메모리 왕복 감소, 병렬화 |

## 더 읽을 곳
- [docs/design/array.md §6.1](../design/array.md) 구조, [docs/design/array.md §6.2](../design/array.md) pairwise 합, [docs/design/array.md §6.5](../design/array.md) 블록 리덕션
- [docs/design/array.md §6.6](../design/array.md) prod, [docs/design/array.md §6.7](../design/array.md) 누적 스캔, [docs/design/array.md §6.8](../design/array.md) mean, var, std, [docs/design/array.md §6.9](../design/array.md) numpy와 다른 곳
- [docs/design/linalg.md §8.2](../design/linalg.md) linalg의 pairwise 합, [docs/design/frame.md §7.4](../design/frame.md) frame의 리덕션
- [docs/api/array.md §7.8](../api/array.md) 계산 순서, [docs/api/frame.md §6.5](../api/frame.md) frame의 계산 순서
