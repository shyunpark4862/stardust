# 7장. 인덱싱, 정렬, unique

> `x[idx]`, `x[mask] = v`는 어떻게 동작하고 어디서 버그가 생기기 쉬운가? 정렬 알고리즘은 어떻게 고르고, NaN은 어디에 놓이나?

## 1. 기본 인덱싱은 뷰, 고급 인덱싱은 복사
| 키 | 결과 | 비용 |
|---|---|---|
| int, slice, `...`, `newaxis`(`None`) | 뷰 (2장) | data 포인터, shape, strides만 다시 계산한다 |
| 모든 축이 int | Python 스칼라 | 0차원 배열이 없다 |
| 정수 배열 | 새 배열 (gather) | 원소를 옮긴다 |
| bool 배열 | 새 배열 (compress) | True인 자리만 옮긴다 |

```python
import stardust as sd

x = sd.arange(10.0)
v = x[2:8:3]
print(v.tolist(), v.base is x)  # [2.0, 5.0] True
g = x[sd.asarray([1, 1, 3])]
print(g.tolist(), g.base is None)  # [1.0, 1.0, 3.0] True
print(x[x > 6].tolist())  # [7.0, 8.0, 9.0]
```

- 키 해석은 C++에서 한다. 키 항목을 고정 용량 저장소에 둔다. `std::vector`의 증가 할당이 `x[1, 2]` 비용의 대부분이었다.
- `x[1, 2]`, `x[1]`처럼 앞쪽 축이 정확한 Python int인 키는 해석을 건너뛰고 오프셋을 바로 계산한다. 58 → 22ns (docs/design/array.md §7.1).

## 2. numpy와 다르게 막은 것
헷갈리는 규칙 둘은 두지 않고 에러로 막았다 (docs/design/array.md §7.1).
- **기본 항목으로 떨어진 고급 인덱스**: numpy는 `t[idx, :, idx]`의 인덱스 축을 맨 앞으로 옮긴다. 결과 shape을 예측하기 어렵다. 여기서는 IndexError다.
- **Python bool 인덱스**: numpy는 `x[True]`를 0차원 마스크로 읽는다. 0차원 배열이 없는 이 라이브러리에서는 `x[1]`과 나란히 놓여 헷갈리기만 한다. bool **배열** 마스크는 된다.

```python
import stardust as sd

t = sd.zeros((2, 3, 4))
print(t[:, sd.asarray([0, 1]), sd.asarray([1, 2])].shape)  # (2, 2)
try:
    t[sd.asarray([0]), :, sd.asarray([0])]
except IndexError as e:
    print(type(e).__name__)  # IndexError
try:
    t[True]
except IndexError as e:
    print(e)  # a bool is not a valid index; use an int, or a boolean array
```

## 3. gather와 scatter
- **가장 흔한 경우를 위한 합친 루프** (`gather_one`, `scatter_one`): 1차원, 인덱스 배열 하나, 같은 dtype. 일반 경로는 원소마다 stride와 shape을 다시 읽고 오프셋을 따로 버퍼에 썼다. 디스어셈블해 보니 원소당 명령어가 약 26개 대 몇 개였다.
- **gather는 검사 패스가 없다.** 나쁜 인덱스는 분기 없이 표시만 하고, 끝나고 IndexError를 낸다. 결과는 새 배열이라 지킬 것이 없다. 병렬로 나뉜다: 인덱스 100만 개 1스레드 733µs, 10스레드 81µs (numpy 757µs).
- **scatter는 모든 인덱스를 먼저 검사한다.** 그래서 `x[idx] = v`는 전부 아니면 전무다. 쓰기는 C 순서, 1스레드라서 **반복된 인덱스에서는 마지막 쓰기가 이긴다.**

```python
import stardust as sd

x = sd.zeros(3)
try:
    x[sd.asarray([0, 99])] = 1.0
except IndexError as e:
    print(e)  # index 99 is out of bounds for axis 0 with size 3
print(x.tolist())  # [0.0, 0.0, 0.0]
y = sd.zeros(3)
y[sd.asarray([0, 0, 0])] = sd.asarray([1.0, 2.0, 3.0])
print(y.tolist())  # [3.0, 0.0, 0.0]
```

- 0번 원소도 바뀌지 않았다: 범위 밖 인덱스가 하나라도 있으면 아무것도 쓰지 않는다.
- `y`의 0번에는 마지막 값 3.0이 남았다.

## 4. bool 마스크
- 마스크 하나만 고급 항목이면 좌표 배열로 바꾸지 않고 한 번 훑으며 True인 자리만 복사한다.
- **고정 청크 두 단계** (16K원소): 청크마다 True 수를 세고(병렬) → 누적합이 각 청크 출력의 시작 위치(직렬) → 청크마다 자기 구간을 쓴다(병렬). 원소마다 출력 자리가 정해지므로 스레드 수와 무관하다.
- **분기 예측**: 무작위 절반 True 마스크는 원소마다 분기가 50% 틀린다. 그래서 256원소 조각마다 True 수를 먼저 세고, 많으면 분기 없이 쓴다(항상 쓰고 출력 위치만 마스크 바이트만큼 민다).
- 결과: `f[b]`(절반 True) 1스레드 324µs, 10스레드 48µs, numpy 3054µs.
- 마스크 대입 `x[m] = v`는 원소의 비트에 마스크로 만든 비트 마스크를 적용한다. `*e = q ? val : *e`는 컴파일러가 분기 있는 조건부 저장으로 되돌리기 때문이다.
- **bool 바이트는 0/1이어야 한다.** 분기 없는 압축은 마스크 바이트를 정수로 더한다. 바깥에서 들어오는 bool 버퍼는 가져올 때 한 번 검사하고 정규화한다 (2장 §7). 입구 한 곳이 모든 bool 커널을 지킨다.
- frame의 마스크 필터도 같은 생각(항상 쓰고, 남길 때만 출력 위치를 민다)이다. 단, 마지막으로 남길 행 다음 칸에도 쓰므로 청크가 자기 마지막 행까지만 돈다 (docs/design/frame.md §7.3).

## 5. 정렬
### 5.1 순서 규칙: NaN은 가장 큰 값
- 비교 함수는 하나다. **NaN은 가장 큰 값이다.** 오름차순이면 맨 끝, 내림차순이면 맨 앞이다.
- 이 규칙은 세 영역이 같다. 옛 numpy판은 내림차순에서도 NaN을 맨 뒤에 두었다. 일관성 원칙에 따라 frame의 규칙("내림차순은 오름차순의 정확한 거울")으로 맞췄다 (DESIGN.md §9.1의 9번).
- **`descending=True`에서도 같은 값은 원래 순서를 지킨다.** 결과를 뒤집지 않고 비교 함수를 뒤집는다.
- 구현: 레인을 버퍼로 읽을 때 NaN과 나머지를 따로 모은다(각각 원래 순서). 나머지만 NaN 검사 없는 비교로 정렬한다. 비교마다 있던 분기가 사라져 1.22배다. 내림차순이면 NaN 묶음을 앞에 둘 뿐이라 비용이 없다.

```python
import stardust as sd

z = sd.asarray([3.0, float("nan"), 1.0, 3.0])
print(sd.sort(z).tolist())  # [1.0, 3.0, 3.0, nan]
print(sd.sort(z, descending=True).tolist())  # [nan, 3.0, 3.0, 1.0]
print(sd.argsort(sd.asarray([2, 1, 2, 1]), descending=True).tolist())  # [0, 2, 1, 3]
s = sd.frame.Series([3.0, float("nan"), 1.0, None])
print(s.sort_values(descending=True).tolist())  # [nan, 3.0, 1.0, <NA>]
```

- frame에는 NA가 하나 더 있다. NA는 값의 순서에 들어가지 않고 `na_position`(기본 맨 뒤)으로 붙는다.

### 5.2 알고리즘은 dtype과 길이로만 고른다
데이터를 보고 고르지 않는다. 정렬되어 있는지, 값의 범위가 좁은지를 보고 경로를 바꾸면 성능이 데이터에 따라 몰래 바뀐다 (docs/design/array.md §1.4).

| 경우 | 알고리즘 |
|---|---|
| int64 값 1024개 이상, (값, 인덱스) 쌍 128개 이상 | LSD 기수 정렬 (8비트 자리, 안정) |
| int64 값 32–1023개, float64 값 32개 이상 | NEON 정렬 네트워크 + bitonic 병합 |
| 나머지, `stable=True` | merge sort (16개 이하는 insertion sort) |
| 나머지, `stable=False` | introsort |

- **NEON 정렬**: 레지스터 8개에 16개를 싣고 8입력 최적 네트워크(비교기 19개)를 세로로 적용한다. 분기가 없다. 네트워크의 정확성은 0-1 원리(0/1 입력 256가지)로 확인했다. 안정 정렬이 아니므로, 결과가 안정 정렬과 구별되지 않을 때(int64, ±0이 섞이지 않은 float)만 쓴다. float 100만 개 14.2ms, numpy 26.6ms.
- **기수 정렬**: 부호 비트를 뒤집어(`v ^ 1<<63`) 부호 있는 순서를 부호 없는 순서로 만든다. 내림차순은 키 전체를 비트 반전한다. 최솟값을 빼고 필요한 자리만 정렬한다. `argsort(i)` 100만 개 6.7ms, numpy 82.2ms.
  - frame의 정렬도 LSD 기수 정렬이다. float는 `float_key`로 정규화한 뒤 양수는 `bits | 1<<63`, 음수는 `~bits`로 바꾼다. NaN의 비트가 +inf보다 커서 규칙이 저절로 지켜진다 (11장).
- **긴 레인 하나**(2¹⁷개 이상): 2¹⁶개 조각마다 정렬한 뒤 둘씩 병합한다. 병합 하나도 출력 구간으로 잘라 병렬로 돈다(merge path). **동점이면 왼쪽 런이 먼저**라서 직렬 안정 정렬과 비트까지 같다.
  - 풀을 쓸 수 있을 때만 조각을 쓴다. 안정 정렬은 조각과 무관하게 결과가 같기 때문이다.
  - 예외: `stable=False`는 항상 조각으로 정렬한다. 불안정 정렬은 조각 경계에 따라 동점의 순서가 달라지므로, 항상 같은 경계를 써야 스레드 수와 무관하다.

## 6. unique
- 함수는 반환 형태별로 넷이다: `unique_values`, `unique_counts`, `unique_inverse`, `unique_all`. 여러 값은 namedtuple(`UniqueAllResult` 등)이다. 세 영역의 결과 namedtuple은 모두 `stardust._results` 한 모듈에 있다.
- 처리: 평탄화 → 안정 argsort → 이웃 비교. 안정하므로 같은 값 묶음의 첫 원소가 첫 등장이다.
- **NaN은 하나의 값이다.** `-0.0`과 `0.0`도 하나다. 보여 주는 값은 처음 나온 것이다. Array API(NaN마다 다른 값)와 다르고, numpy 2의 `np.unique` 기본값, frame의 키 규칙과 같다 (DESIGN.md §9.1의 10번). groupby에서 NaN마다 그룹이 따로 생기면 쓸모가 없기 때문이다.
- NaN은 정렬에서 이미 맨 뒤에 원래 순서로 모여 있으므로, NaN 구간 전체를 마지막 묶음 하나로 친다. 추가 비용이 없다.
- 해시 기반 경로는 두지 않았다. int64는 기수 정렬로 이미 numpy보다 빠르고, 해시가 이기는지는 데이터(값 종류의 수)에 달려 있다.

```python
import stardust as sd

u = sd.unique_all(sd.asarray([2.0, float("nan"), -0.0, 2.0, 0.0, float("nan")]))
print(type(u).__name__)  # UniqueAllResult
print(u.values.tolist(), u.counts.tolist())  # [-0.0, 2.0, nan] [2, 2, 2]
print(u.indices.tolist())  # [2, 0, 1]
v = sd.frame.Series([2.0, float("nan"), -0.0, 0.0, None]).unique_values()
print(v.tolist())  # [-0.0, 2.0, nan, <NA>]
```

## 더 읽을 곳
- [docs/design/array.md §7.1](../design/array.md) 키 해석, [docs/design/array.md §7.3](../design/array.md) gather와 scatter, [docs/design/array.md §7.4](../design/array.md) bool 마스크
- [docs/design/array.md §8.1](../design/array.md) 순서 규칙, [docs/design/array.md §8.2](../design/array.md) 알고리즘 선택, [docs/design/array.md §8.5](../design/array.md) 긴 레인의 병렬 정렬, [docs/design/array.md §8.6](../design/array.md) unique
- [docs/api/array.md §5](../api/array.md) 인덱싱, [docs/api/array.md §8](../api/array.md) 정렬과 집합
- [docs/design/frame.md §9.3](../design/frame.md) frame의 정렬 키
