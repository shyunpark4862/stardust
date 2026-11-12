# 9장. LAPACK 위의 층

> 배열 하나가 LAPACK에 들어가서 결과 배열로 나오기까지 무엇이 일어나나? 배치는 언제 여러 코어로 나누고, 그래도 비트가 같은 이유는?

## 1. 호출 하나의 여정
모든 행렬 함수는 같은 순서를 거친다 (`linalg/batch.hpp`).

```
sd.linalg.solve(a, b)                                    bind: named("solve", ...) (C++ 바로 바인딩)
 │ 1. 인자 검사: sd.ndarray인가, dtype, shape            linalg/args.*, ops/common.hpp
 │    배치 축 브로드캐스팅 → 결과 shape                    Batch
 │ 2. 결과 배열 할당: F-order strides를 linalg가 정하고    linalg/outblock.cpp → array의 new_array_strided
 │    메모리와 플래그는 배열이 맡는다
 │ 3. run_batch: 배치 원소마다                            (일이 크면 풀에서 나눔, §4)
 │      패킹 (F-order로, float64로, 유한성 검사를 한 패스에)  linalg/pack.hpp
 │      LAPACK (getrf → getrs)                            accelerate/lapack.hpp
 │      info 기록, 후처리 (삼각 0 채우기, eig 정렬 …)
 │    몸체는 0 또는 실패 코드를 돌려준다
 │ 4. 실패가 있으면 가장 작은 인덱스로 raise              ops/common.hpp
 ▼
sd.ndarray (F-order, base None)
```

- LAPACK 래퍼는 루틴 하나에 템플릿 함수 하나다. `double`이면 `d*`, `complex<double>`이면 `z*`를 부른다(예: `getrf<T>`). 음수 `info`는 우리 인자의 버그라서 `std::logic_error`다.
- 인터페이스는 Accelerate의 새 LAPACK, 64비트 정수 인자(ILP64)다.

## 2. 패킹: 한 패스에 세 가지
`pack<T>`는 strided 행렬 하나를 F-order 작업 버퍼로 옮긴다 (docs/design/linalg.md §5).
1. 임의의 행·열 stride(음수, 0 포함)로 읽어 열 우선으로 쓴다.
2. bool/int64를 float64로 바꾼다.
3. 유한성을 검사한다(8장 §5).

어차피 LAPACK은 대부분 입력 행렬을 덮어쓰므로, 사용자 입력을 지키려면 작업용 복사가 필요하다. 그 복사에 일을 몰았다.
- **출력 자리에 바로 패킹한다.** `cholesky`는 입력의 아래 삼각을 출력 자리에 패킹하고 `potrf`를 그 자리에서 부른다. `solve`는 `b`를 출력 자리에 패킹하고 `getrs`가 해로 덮어쓴다. 많은 함수가 "복사 한 번 + LAPACK"으로 끝난다.
- **레이아웃별 빠른 경로**
  - 열이 연속(F-order): 열마다 복사와 유한성 검사를 한 패스로. 검사는 비트로 한다(지수 필드가 모두 1이면 NaN 또는 inf). 정수 연산이라 벡터화된다.
  - 행이 연속(C-order): 32×32 타일로 전치한다. 그냥 열 순서로 읽으면 1000×1000에서 원소마다 8KB씩 건너뛴다. 큰 `solve_triangular`가 이 때문에 scipy보다 2–3배 느렸다.
  - 원소 2¹⁷개 이상인 C-order 행렬 하나는 전치를 열 32개 단위로 풀에서 나눈다. 1000²에서 `solve_triangular` 1.44배.
- **삼각만 읽는 함수는 그 삼각만** 복사하고 검사한다. 읽기만 하는 삼각(`solve_triangular`의 `a`)은 조건이 맞으면 복사하지 않고 그 자리에서 쓴다.

### 2.1 레이아웃 불변성
**결과는 입력 레이아웃(C, F, strided, 음수 stride, 브로드캐스트)과 무관하게 비트까지 같다.** 패킹이 모든 레이아웃을 같은 F-order 버퍼로 만들기 때문이다.
- 그래서 "C-order 입력을 `trans='T'`로 그 자리에서 읽기" 같은 최적화는 하지 않는다. 같은 행렬이 레이아웃에 따라 다른 비트를 받게 된다. 큰 C-order `solve_triangular`가 scipy보다 조금 느린 이유다 (docs/design/linalg.md §5.7).

```python
import numpy as np
import stardust as sd

rng = np.random.default_rng(0)
a = rng.standard_normal((5, 5))
a = a @ a.T + 5 * np.eye(5)
big = np.zeros((10, 10))
big[::2, ::2] = a
layouts = [sd.asarray(a), sd.asarray(np.asfortranarray(a)), sd.asarray(big)[::2, ::2]]
bits = {np.asarray(sd.linalg.eigh(x).eigenvalues).tobytes() for x in layouts}
print(len(bits))  # 1
```

## 3. 작업 메모리와 16바이트 정렬
### 3.1 `Scratch`
배치의 원소마다 쓰는 작업 메모리다. 슬롯 9개의 바이트 버퍼와 정수 버퍼 2개다 (docs/design/linalg.md §6.3).
- **0으로 채우지 않는다.** `std::vector`의 `resize`는 새 메모리를 0으로 채운다. 1000²에서 이 비용이 패킹만큼 들었다. 모든 사용처가 읽기 전에 쓴다.
- **workspace 크기는 `Scratch`마다 한 번만 묻는다.** 배치 안의 행렬은 모두 같은 크기라서 첫 행렬만 `lwork = -1`로 묻는다. 그리고 행렬마다 물을 때와 **같은 `lwork`**를 쓴다. `gesdd`는 `lwork`를 보고 내부 경로를 고르므로, 크기가 바뀌면 결과 비트가 바뀐다.
- **버퍼마다 캐시 줄을 따로 갖는다.** 128바이트 경계에서 시작하고 크기도 128의 배수다. 6장 §6의 거짓 공유를 고친 것이다.

### 3.2 Accelerate는 정렬에 따라 다르게 반올림한다
측정으로 찾은 Accelerate의 성질이다 (docs/design/linalg.md §6.4).
- float64 배치에서 행렬 크기가 홀수이면(3×3이면 72바이트 간격) 홀수 번째 행렬의 시작이 8바이트 어긋난다. **같은 행렬이 배치 안의 위치에 따라 다른 비트를 받았다.**
  - n = 3, 5, 7, 9의 무작위 120개 중 실수 `schur` 18–30개, `svd` 3–17개, `eigh` 27–45개가 달랐다. complex(원소 16바이트라 늘 정렬)는 0개였다.
- 그래서 **LAPACK과 BLAS에 넘기는 배열은 모두 16바이트 정렬한다.** 출력 자리가 정렬이 아니면 정렬된 scratch에서 계산하고 복사한다(`LapackSlot`).
- 결과: 모든 함수에서 배치 결과가 행렬 하나씩 부른 결과와 비트까지 같다.

```python
import numpy as np
import stardust as sd

m = np.random.default_rng(0).standard_normal((120, 3, 3))
one_by_one = [np.asarray(sd.linalg.svd(sd.asarray(m[i])).S).tobytes() for i in range(120)]
batched = np.asarray(sd.linalg.svd(sd.asarray(m)).S)
print(all(batched[i].tobytes() == one_by_one[i] for i in range(120)))  # True
```

## 4. 배치를 언제 나누나: 비용 모델
배열은 원소 수로 일을 잰다(6장). 행렬 함수는 그렇게 잴 수 없다. 3×3 `svd` 하나와 3×3 `trace` 하나는 원소 수가 같고 시간이 수십 배 다르다.
- **원소 하나의 직렬 비용을 `a + c·w` ns로 추정한다** (`linalg/cost.hpp`). w는 행렬의 m·n이다. 함수 계열마다 n = 2–256에서 잰 시간에 맞췄다.

  | 계열 | a | c | 함수 |
  |---|---:|---:|---|
  | `kLU` | 70 | 3 | `solve`, `inv`, `det`, `lu`, `cholesky` … |
  | `kQR` | 180 | 10 | `qr`, `qr_pivoted`, `hessenberg` |
  | `kEigh` | 250 | 45 | `eigh`, `eigvalsh` |
  | `kSVD` | 1000 | 80 | `svd`, `pinv`, `lstsq`, `matrix_rank` … |
  | `kStream` | 12 | 0.3 | `vecdot`, `trace`, 노름 |

- 배치 전체의 추정이 **10µs 이상이면 나누고**, 태스크 하나는 약 **5µs** 분량이다. 직렬로 8–10µs를 넘으면 나누는 쪽이 이기기 시작했다.
- 첫 원소의 시간을 재서 정하는 안은 택하지 않았다. 실행마다 분할 여부가 달라질 수 있다.
- **크기 기준은 없다.** 큰 행렬의 배치도 나눈다. Accelerate는 거의 모든 루틴을 1스레드로 돌기 때문이다.

### 4.1 Accelerate에 대해 잰 사실
설계의 전제가 된 측정이다 (docs/design/linalg.md §1.2).
- 내부 스레드를 쓰는 것은 `getrf`의 n ≥ 768뿐이다. `gesv`, `potrf`, `geqrf`, `syevd`, `gesdd`, `geev`는 n = 1024까지 사실상 1스레드다. 그래서 큰 행렬 하나는 numpy, scipy와 마찬가지로 대부분 코어 하나로 계산된다.
- 여러 스레드에서 동시에 불러도, Accelerate의 스레드 수를 바꿔도 결과 비트가 같다. 배치를 나눠도 되는 근거다.
- **일부 작은 호출은 코어들이 함께 쓰는 행렬 유닛(AMX)에서 직렬로 돈다.** `getrf`의 n = 32–40, 우변 8개 이상의 삼각 풀기의 n = 24–40. 여기서는 스레드를 늘리면 오히려 느려진다. 그래서 태스크 수에 상한(3 또는 5)을 둔다(`lu_task_cap`). 이 경로를 피하는 방법(우변을 나눠 풀기)은 결과 비트를 바꾸므로 하지 않았다.
- 배열의 `matmul`은 반대다. 큰 행렬은 Accelerate가 스레드를 쓰므로 차례로 돌고, 작은 행렬(256³ 이하)의 배치만 풀로 나눈다 (docs/design/array.md §9.4).

### 4.2 나눠도 비트가 같은 이유
- 배치 원소 하나는 어디서 돌든 같은 계산이다: 자기 `Scratch`, 같은 `lwork`(§3.1), 16바이트 정렬(§3.2), LAPACK 결과는 스레드와 무관(§4.1).
- 실패 보고는 가장 작은 인덱스다(6장 §5).
- 출력 하나의 긴 합(`vecdot`, 노름)은 배열과 같은 pairwise 트리로 나눈다(5장).
- `solve_sylvester`의 Schur 분해 둘은 독립이라 풀의 태스크 둘로 동시에 돈다. 같은 배열과 workspace 크기라 비트가 같다. 1000²에서 1.6배.

## 5. 호출 비용
작은 행렬에서는 LAPACK 계산보다 호출 경로의 고정비가 크다. 결과를 바꾸지 않고 줄인 것들이다 (docs/design/linalg.md §9).
- **Python 래퍼 없음**: 함수는 C++에서 바로 바인딩한다. 함수마다 Python 래퍼를 두면 호출마다 1–2µs가 더 든다. 3×3 `solve`(약 0.5µs)보다 크다.
- **경계**: 배열을 직접 받고 직접 만든다. 별도 패키지였을 때보다 결과 배열 하나에 약 120ns 빠르다 (12장).
- workspace 크기 기억, 0 채우기 없음, 짧은 합에는 할당 없음, 작은 행렬 경로에는 크기 검사 하나만
- 결과: 3×3 `solve` 0.48µs, numpy 1.99µs. 작은 행렬 하나의 기하평균 2.8배. 같은 자리에서 잰 numpy/scipy 대비 248줄 중 191줄이 10% 넘게 빠르고 9줄이 느리다 (`bench/merge/m6/final/table.md`).

## 더 읽을 곳
- [docs/design/linalg.md §1.2](../design/linalg.md) Accelerate LAPACK, [docs/design/linalg.md §5.1](../design/linalg.md) 패킹, [docs/design/linalg.md §5.3](../design/linalg.md) 빠른 경로, [docs/design/linalg.md §5.7](../design/linalg.md) 레이아웃 불변성
- [docs/design/linalg.md §6.2](../design/linalg.md) `run_batch`, [docs/design/linalg.md §6.3](../design/linalg.md) 작업 메모리, [docs/design/linalg.md §6.4](../design/linalg.md) 16바이트 정렬
- [docs/design/linalg.md §7.1](../design/linalg.md) 비용 모델, [docs/design/linalg.md §7.2](../design/linalg.md) 두 상수, [docs/design/linalg.md §7.5](../design/linalg.md) AMX 상한
- [docs/design/linalg.md §10.1](../design/linalg.md) 함수별 드라이버, [docs/design/linalg.md §11.1](../design/linalg.md) expm 알고리즘
