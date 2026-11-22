# 공간 영역 설계 (sd.spatial)

> 상태: **설계 초안** (2026-11-22). 큰 결정(§1.1 SPT1–SPT5)은 사용자가 정했다. §9의 질문 스물은 모두 추천안으로 정했다(SPT6–SPT25, 사용자 위임 2026-11-22). SPT0(준비)을 마쳤다: API 초안 `docs/api/spatial.md`, 시험 점·참값·탐침의 도구와 기록, §3.2의 상수와 증명, 짝 커널 두 순서의 측정(SPT22의 결과는 타일 순서, §3.1), §5의 다시 잰 시간. SPT0에서 정한 것은 SPT26–SPT28이다. SPT1(거리)을 마쳤다: `pdist cdist squareform`과 거리 함수 열넷, 타일 순서의 짝 커널(`csrc/spatial/metric.hpp`), 잰 정확도 계약(§4), SPT1에서 정한 것은 SPT29–SPT36이다. 거리 부분(§2.1, §2.2, §3.1, §4의 거리, §6의 파일)은 구현과 맞다. 나머지는 안이다.

`sd.spatial`(점 사이의 거리와 가장 가까운 이웃 찾기)의 확정 사항과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처의 안은 SPT0에서 `docs/api/spatial.md`로 쓴다.
- 비교 대상은 scipy 1.18.1의 `scipy.spatial`(`distance`, `KDTree`)과 scikit-learn 1.9.1의 `sklearn.neighbors`(`KDTree`, `BallTree`, `NearestNeighbors`)·`sklearn.metrics.pairwise`(`pairwise_distances_argmin_min`, `euclidean_distances`, `haversine_distances`), R 4.6.2의 `stats::dist`다. R에는 이웃 찾기 패키지(`RANN`, `FNN`, `dbscan`, `nabor`)가 설치되어 있지 않다. 설치하지 않는다(거리 함수만 R과 견준다).
- 바닥은 배열(`csrc/array/`), Accelerate의 BLAS(`csrc/accelerate/blas.hpp`의 `gemm`), 스레드풀(`csrc/parallel/`)이다. 짝 거리의 합은 이 영역의 타일 순서다(SPT22, SPT26; §3.1). 반경 질의의 `.tocsr()`는 `sd.sparse.csr_array`를 만든다.
- 이 문서의 잰 값은 Apple M4 Pro, macOS 26.6.2에서 본 것이다. 다른 작업과 기기를 나눠 썼다. 짝 커널(§3.1), GEMM의 동시 호출(§3.5), §5의 시간은 SPT0에서 다시 쟀다(부하 평균 1.4–1.5; `tools/spatial_probe.py`).
- ε은 float64의 기계 엡실론 2⁻⁵² ≈ 2.22×10⁻¹⁶, u는 단위 반올림 2⁻⁵³(= ε/2), γ_k = ku/(1 − ku)다(sparse.md §4와 같은 기호).

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-22)
| 번호 | 결정 |
|---|---|
| SPT1 범위 | 세 덩어리다. **거리**: `pdist`, `cdist`, `squareform`과 거리 함수 `euclidean sqeuclidean cityblock chebyshev minkowski(p=) seuclidean mahalanobis cosine correlation canberra braycurtis hamming jaccard`, 가중치 `w=`는 scipy의 뜻. `haversine`은 넣을지 조사한다(§9). **전체 행렬을 만들지 않는 축약**: 행마다 가장 가까운 점과 그 거리(sklearn `pairwise_distances_argmin_min`에 해당하는 것)를 조각으로 나눠 구한다. **정확한 이웃 찾기**: `KDTree`, KD 트리가 가지를 칠 수 없는 거리를 위한 `BallTree`는 넣을지 평가한다(§9), 한 번에 끝나는 kNN·반경 질의의 편의 함수. 빼는 것(나중 후보): Qhull 기하(`ConvexHull`, `Delaunay`, `Voronoi`), 회전·보간·정렬(`Rotation`, `slerp`, `procrustes`, `transform`), 근사 이웃(NN-descent), 최소 신장 트리 같은 그래프 도우미(나중의 `sd.graph` 영역) (사용자 결정 2026-11-22) |
| SPT2 유클리드 계산 | 섞는다. GEMM(‖x‖² + ‖y‖² − 2x·y, Accelerate)은 **후보를 빨리 고르는 데만** 쓴다. 돌려주는 모든 거리와 모든 경계 비교(argmin, k번째 이웃, 반경 판정)는 차이에서 직접 다시 계산한다. 정확도는 직접 계산과 같고 빠르기는 GEMM에 가깝다. 같은 짝은 `pdist`, `cdist`, 트리, 축약에서 같은 비트다. GEMM의 앞 고르기가 참 후보를 절대 빠뜨리지 않게 하는 여유(전개식의 오차 한계)를 구해 적는다(§3.2) (사용자 결정 2026-11-22) |
| SPT3 반경 질의의 결과 | CSR 꼴의 `indptr`, `indices`, `distances` 배열을 가진 결과 객체이고, `.tocsr()`가 `sd.sparse.csr_array`를 준다. 거리가 0인 짝(저장된 0)을 어떻게 다루는지 적는다(§2.5) (사용자 결정 2026-11-22) |
| SPT4 이름 | scipy.spatial의 이름이다(`KDTree.query(k)`, `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, `sparse_distance_matrix`, `distance`의 함수들). 동점은 (거리, 색인) 순서이고 이것을 약속한다. scipy는 약속하지 않으므로 이런 차이는 `[scipy와 다름]`으로 적는다 (사용자 결정 2026-11-22) |
| SPT5 위치 | 최상위 `sd.spatial`, `sd.sparse`처럼 평평한 이름공간이다. 거리 함수를 `sd.spatial.distance`에도 둘지는 하나를 골라 추천한다(별칭 없음; §1.2, §9) (사용자 결정 2026-11-22) |
| SPT6 `distance` 하위 모듈 | 두지 않는다. 거리 함수도 `sd.spatial`에 바로 둔다(§9의 1) (추천안, 사용자 위임 2026-11-22) |
| SPT7 편의 함수의 이름 | `kneighbors(XA, XB, /, *, k=1, …)`와 `radius_neighbors(XA, XB, r, /, …)` 둘이다. `pairwise_distances_argmin_min`은 `kneighbors(k=1)`이다(§9의 2) (추천안, 사용자 위임 2026-11-22) |
| SPT8 `BallTree` | 첫 판에 넣지 않는다(나중 후보). KD 트리가 칠 수 없는 거리는 전수가 맡는다. SPT6 마일스톤 뒤에 다시 본다(§9의 3) (추천안, 사용자 위임 2026-11-22) |
| SPT9 `haversine` | 넣는다. sklearn의 뜻(위도·경도 라디안 2열, 단위 구의 라디안)이고 모든 범위에서 계약을 지키는 식을 Arb로 골라 쓴다. 트리는 쓰지 않는다(§9의 4) (추천안, 사용자 위임 2026-11-22) |
| SPT10 `workers=` | 두지 않는다. 스레드 수는 `sd.set_num_threads` 하나다(`[scipy와 다름]`; §9의 5) (추천안, 사용자 위임 2026-11-22) |
| SPT11 `eps` | 두지 않는다. 결과는 언제나 정확하다(§9의 6) (추천안, 사용자 위임 2026-11-22) |
| SPT12 유한하지 않은 값 | 거리 함수는 NaN·inf를 IEEE로 퍼뜨리고(`chebyshev`, `canberra`도), 이웃 찾기는 데이터와 질의점의 NaN·inf를 `ValueError`로 막는다(§9의 7) (추천안, 사용자 위임 2026-11-22) |
| SPT13 받는 입력 | API.md §5의 상위 함수 입력(`sd.ndarray`, 시퀀스, numpy, `DataFrame`)이다(§9의 8) (추천안, 사용자 위임 2026-11-22) |
| SPT14 호출 가능한 `metric`, `out=` | 둘 다 첫 판에 두지 않는다(§9의 9) (추천안, 사용자 위임 2026-11-22) |
| SPT15 `metric`·`k`의 자리 | 키워드만이다(`[scipy와 다름]`; §9의 10) (추천안, 사용자 위임 2026-11-22) |
| SPT16 `squareform`의 검사 | scipy와 같다. NaN이 대칭 검사에 걸리면 메시지는 NaN이 있다고 말한다(§9의 11) (추천안, 사용자 위임 2026-11-22) |
| SPT17 결과 객체 | `QueryResult(distances, indices)` namedtuple을 `query`·`kneighbors`가, `RadiusResult`(`indptr`, `indices`, `distances`, `shape`, `tocsr()`)를 반경 질의 셋이 쓴다. 행 안은 색인 오름차순이다. `query_pairs`는 정렬된 (k, 2) 배열, `sparse_distance_matrix`는 `csr_array`다(§9의 12) (추천안, 사용자 위임 2026-11-22) |
| SPT18 `KDTree`의 옵션 | `leafsize`만 둔다(기본 10에서 시작해 SPT6에서 재서 1.2배 넘게 빠른 값이 있으면 바꾼다; §9의 13) (추천안, 사용자 위임 2026-11-22) |
| SPT19 `distance_upper_bound` | 둔다. 엄격한 부등호, 0 이상이거나 inf, NaN·음수는 `ValueError`다. `kneighbors`에도 둔다(§9의 14) (추천안, 사용자 위임 2026-11-22) |
| SPT20 `boxsize` | 첫 판에 두지 않는다(나중 후보; §9의 15) (추천안, 사용자 위임 2026-11-22) |
| SPT21 `query`의 전수 길 | 모양으로 트리와 전수를 고른다. 결과는 길과 무관하다(§9의 16) (추천안, 사용자 위임 2026-11-22) |
| SPT22 짝 커널의 순서 | `vector_norm`(`ord`)·`vecdot`과 비트까지 같게 한다. SPT0에서 재서 이 순서의 묶음 커널이 타일 순서보다 1.3배 넘게 느리면 타일 순서로 바꾸고 기록한다(§9의 17) (추천안, 사용자 위임 2026-11-22). **SPT0의 결과: 타일 순서로 바꾼다.** `vector_norm`의 순서는 n ≥ 16의 모든 길이에서 1.34–2.68배, BLAS 구간(n = 4096)에서 3.85배 느렸다(§3.1의 표) |
| SPT26 타일 순서의 꼴 | 짝마다 항을 32개씩의 덩어리로 나눠 덩어리 안은 0에서 시작하는 차례의 `fma`, 덩어리의 합은 차례로 더한다. 모든 길이가 같은 꼴이다(BLAS를 부르지 않는다). `cityblock`, `minkowski`, 가중 거리, `seuclidean`, `canberra`, `braycurtis`의 합도 같은 꼴이고 `chebyshev`는 최댓값이다. n < 16이면 `vector_norm`과 같은 비트가 된다(그 길이에서는 두 순서가 같은 차례 합이다). 까닭: 덩어리 16, 32, 64, 나누지 않음이 빠르기가 같고(±2%) 합의 깊이 32 + ⌈n/32⌉ − 1은 n ≈ 1000 둘레에서 가장 작다. BLAS 구간을 없애 거리가 Accelerate의 스레드 모드와도 무관해진다 (추천안, 사용자 위임 2026-11-22) |
| SPT27 신중한 경로의 비율 | 제곱합이 넘치거나 아주 작을 때의 두 번째 패스는 2의 거듭제곱으로 비율을 맞춘다(최댓값의 지수에서 고른 2ᵏ; 곱셈이 정확하다). 까닭: §3.2의 보조정리 2가 그대로 서고, 비율을 맞추는 반올림이 거리 계약에 더해지지 않는다(`vnorm.hpp`는 최댓값 자체로 나눈다) (추천안, 사용자 위임 2026-11-22) |
| SPT28 GEMM 고르기의 범위와 상수 | GEMM 고르기는 1 ≤ n ≤ 2²¹ − 2에서만 쓰고, 여유는 §3.2의 상수(c₁ = (n + 3)u, c₂ = 1 − (n + 16)u 등, 모두 정확한 double)로 계산한다. 더 긴 행은 직접 전수다. 까닭: 상수가 정확히 표현되어 증명이 짧고, 행 하나가 16 MB를 넘는 길이에서는 고르기가 이득이 없다 (추천안, 사용자 위임 2026-11-22) |
| SPT23 `hamming`·`jaccard`의 dtype | bool, int64, float64를 입력의 dtype 그대로 비교한다(bool은 비트 묶음). 다른 거리는 int64·bool을 float64로 바꾼다(§9의 18) (추천안, 사용자 위임 2026-11-22) |
| SPT24 `w`의 검사 | 유한하고 0 이상이다. NaN·inf는 `ValueError`다(§9의 19) (추천안, 사용자 위임 2026-11-22) |
| SPT25 가중 `count_neighbors` | 고정 깊이의 노드 쌍 태스크와 태스크 번호 순서의 합이다. 스레드 수와 무관하다(§9의 20) (추천안, 사용자 위임 2026-11-22) |
| SPT29 `cosine`·`correlation`의 식 | 행을 한 번 정규화하고(x̂ = x/‖x‖_w, 노름은 `euclidean`의 커널과 신중한 경로) 짝마다 d = ‖x̂ − ŷ‖²_w/2를 `sqeuclidean`의 타일 커널로 구해 2로 자른다(§3.1의 안 (b)). 까닭: Arb와 견준 64차원 거의 평행한 짝에서 상대오차가 d ≈ 4×10⁻³에서 (a) 1.2×10⁻¹³ 대 (b) 5.5×10⁻¹⁶, d ≈ 5×10⁻¹⁷에서 (a) 6.2(모든 자릿수를 잃음) 대 (b) 4×10⁻⁹이고, 빠르기는 `euclidean`과 같다(2000² n = 64 `cdist` 10스레드 2.6 ms 대 3.6 ms). 정확히 반대인 두 행은 2가 아니라 2‖x̂‖²(1 ULP 안)일 수 있다 (추천안, 사용자 위임 2026-11-22) |
| SPT30 `minkowski`의 근과 신중한 경로 | 근은 `pow(s, 1/p)`이고 1/p가 2의 거듭제곱이 아니면 `pow(r, p)`에 대한 뉴턴 한 걸음 r − r((t − s)/s)/p를 더한다. 신중한 경로는 2ᵏ 대신 최댓값 M = max\|zᵢ\|로 나눈 \|zᵢ\|/M의 합과 M·근이다. 까닭: 1/p의 반올림은 s^(1/p)에 상대 (u/p)·\|ln s\|의 오차를 더해 s ≈ 10³⁰⁰에서 수백 ULP가 된다(scipy의 `edge/huge`, `scaled/1e150`에서 보인다); 뉴턴 걸음 뒤에는 2u/p + u다. 큰 p에서 \|z′\|ᵖ의 2ᵏ 비율은 넘치거나(z′ ≥ 1) 언더플로한다(z′ < 1, p = 2000). \|zᵢ\|/M의 반올림 u는 그 p제곱에서 pu, 근에서 u라 계약에 넣었다(§4). 가중이면 wᵢ = 0인 항은 wᵢ·zᵢ(NaN·inf만 퍼진다)라서 \|zᵢ\|ᵖ의 넘침이 0·inf의 NaN을 만들지 않는다 (추천안, 사용자 위임 2026-11-22) |
| SPT31 `haversine`의 식 | d = 2·atan2(√h, √g), h = sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2), g = 1 − h를 같은 값의 다른 꼴 cos²(Δφ/2) cos²(Δλ/2) + sin²((φ₁ + φ₂)/2) sin²(Δλ/2)로 구한다. h와 g가 모두 음이 아닌 항의 합이라 상쇄가 없다. 계약은 \|d̂ − d\| ≤ 20u·d다. 까닭: sklearn의 2·asin(√h)은 대척점 근처에서 asin의 인자가 1에 가까워 자릿수를 잃고(대척점에서 10⁻⁸ 떨어진 짝의 기록이 참값과 수천 ULP 다르다), atan2(√h, √(1 − h))는 1 − h의 상쇄로 같은 곳에서 잃는다. 잰 최대는 2 ULP(상대 1.84u, 대척점 넷 포함 32짝)다 (추천안, 사용자 위임 2026-11-22) |
| SPT32 거리의 정확도 계약 | §4의 표다(잰 값과 함께). 모든 한계에 2⁻¹⁰⁷⁴(가장 작은 비정상 수)를 더한다: 비정상 수인 결과는 올바르게 반올림해도 그만큼 다르다. `seuclidean`은 1/Vᵢ의 반올림까지 γₘ₊₄, 가중 `sqeuclidean`은 w·z의 반올림까지 γₘ₊₃, `braycurtis`는 두 합과 나눗셈의 γ₂ₘ₊₃, `cosine`은 u(3√d + (2m + 8)d) + (m + 4)²u², `correlation`은 그것에 평균의 반올림이 만드는 2(ρₓ² + ρᵧ²)(평균의 오차에 대해 2차라서 첫째 차수가 없다), `mahalanobis`는 min(E/d, √E) + ud(E = γ₂ₘ₊₃\|z\|ᵀ\|VI\|\|z\|)다. 까닭: 각 계산 순서의 반올림을 센 한계이고(§4), Arb와 견주면 `chebyshev`와 가중치 없는 셈은 올바른 반올림이고 나머지는 모든 점에서 계약의 0.68 이하다 (추천안, 사용자 위임 2026-11-22) |
| SPT33 신중한 경로의 문턱과 행의 비율 | 제곱합(과 `minkowski`의 합, `mahalanobis`의 이차식)이 inf이거나 2⁻⁹⁶⁸보다 작으면 신중한 경로다(초안의 n·2⁻¹⁰⁰⁰ 대신 상수; n < 2⁴⁰이면 언더플로한 항의 몫이 상대 2⁻¹⁰⁷ 아래다). `cosine`·`correlation`의 행은 정규화(와 평균 빼기) 전에 가장 큰 원소의 지수로 고른 2ᵏ로 비율을 맞춘다(정확; 정상 범위의 행은 비트가 같다). 까닭: 비정상 수 행의 노름과 평균은 자릿수를 잃는다(`edge/subnormal`의 가중 `cosine`이 계약의 2×10¹⁴배, `correlation`의 평균 5×10⁻³²⁴/2가 0으로 반올림되었다) (추천안, 사용자 위임 2026-11-22) |
| SPT34 기본 `VI`와 유한하지 않은 자료 | 자료에 NaN·inf가 있어 공분산이 유한하지 않으면 기본 `VI`는 NaN 행렬이고 모든 거리가 NaN이다(scipy와 같은 결과; `inv()`의 `ValueError`를 내지 않는다). 공분산이 특이하면 `LinAlgError`(`pdist(): the covariance of the data is singular, so the default 'VI' does not exist; give 'VI'`)다. 까닭: 거리 함수는 NaN을 퍼뜨린다(SPT12) (추천안, 사용자 위임 2026-11-22) |
| SPT35 `hamming`·`jaccard`의 두 dtype | `u`와 `v`(`XA`와 `XB`)의 dtype이 다르면 하나라도 float64면 float64, 아니면 int64(bool과 int64)로 맞춘 뒤 비교한다. 가중 bool 행은 바이트 비교의 한 짝 커널이다(비트 묶음은 가중치 없는 bool만). 까닭: 배열의 형 승격과 같은 규칙이고, 개수가 정확해 어느 길이든 같은 값이다 (추천안, 사용자 위임 2026-11-22) |
| SPT36 SPT0 시제품의 자리 | `csrc/spatial/`의 `pair_kernel.hpp`(두 순서의 시제품)는 지우지 않고 `tools/spatial_probe_pair_kernel.hpp`로 옮긴다. 라이브러리의 커널은 `metric.hpp`다. 까닭: `tools/spatial_probe.py kernels`가 SPT22의 측정을 다시 할 수 있어야 하고, 라이브러리 밖의 코드는 `csrc/`에 두지 않는다 (추천안, 사용자 위임 2026-11-22) |

### 1.2 범위 (안)
마일스톤은 §8이다. "SPTn"은 그 마일스톤에서 만든다는 뜻이고 "나중"은 첫 판에 넣지 않는 후보다.

**이름공간**: 거리 함수도 `sd.spatial`에 바로 둔다(`sd.spatial.pdist`, `sd.spatial.cosine`). `sd.spatial.distance`는 없다(§9의 1번). scipy의 `from scipy.spatial.distance import cdist`는 `from stardust.spatial import cdist`가 된다. sparse가 `sd.sparse.linalg` 없이 평평한 것과 같다(sparse.md §1.1). 평평하게 해도 이름이 겹치지 않는다(`KDTree`, 거리 함수 열셋, `pdist`, `cdist`, `squareform`, `kneighbors`, `radius_neighbors`).

**scipy.spatial.distance → sd.spatial**
| scipy | sd.spatial | 마일스톤 |
|---|---|---|
| `pdist`, `cdist`, `squareform` | 같은 이름 (`metric`과 그 인자는 키워드만, §2.2) | SPT1 |
| `euclidean sqeuclidean cityblock chebyshev minkowski seuclidean mahalanobis cosine correlation canberra braycurtis hamming jaccard` | 같은 이름. 모두 (…, n) 배치를 받는다(`[scipy와 다름]`: scipy 1.18.1은 `minkowski`·`euclidean`·`sqeuclidean`만 배치를 받고 나머지는 1차원만) | SPT1 |
| `haversine`(sklearn `haversine_distances`) | 넣는다(SPT9) | SPT1 |
| 불 거리 `dice rogerstanimoto russellrao sokalsneath yule`, `jensenshannon`, `directed_hausdorff` | 없음, 나중 | 나중 |
| `is_valid_dm`, `is_valid_y`, `num_obs_dm`, `num_obs_y` | 없음, 나중 (`squareform`의 검사는 안에 있다) | 나중 |
| 거리 이름의 별칭(`'e'`, `'euclid'`, `'cb'`, `'matching'`, `'cos'` …)과 대문자 | 없음: 소문자 정식 이름만(optimize의 `method`와 같은 규칙, §2.2) | — |
| 호출 가능한 `metric`, `'test_…'` 이름 | 없음, 나중 (§9의 9번) | — |

**scipy.spatial → sd.spatial**
| scipy | sd.spatial | 마일스톤 |
|---|---|---|
| `KDTree` | 같은 이름 (§2.4) | SPT3 |
| `KDTree.query`, `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, `sparse_distance_matrix` | 같은 이름 | SPT3, SPT4 |
| `cKDTree` | 없음: `KDTree`와 같은 것이라 별칭이 된다 | — |
| `KDTree.tree`(노드 객체), `Rectangle`, `minkowski_distance`, `minkowski_distance_p` | 없음: 트리의 속은 공개하지 않는다. 앞의 둘은 `minkowski`와 그 p제곱이다 | — |
| `distance_matrix` | 없음: scipy 1.18.0에서 폐기 예고되었고 `cdist`와 같다 | — |
| `ConvexHull`, `Delaunay`, `Voronoi`, `SphericalVoronoi`, `HalfspaceIntersection`, `tsearch`, `procrustes`, `geometric_slerp`, `transform` | 없음, 나중 (SPT1) | 나중 |

**sklearn → sd.spatial**
| sklearn | sd.spatial | 마일스톤 |
|---|---|---|
| `pairwise_distances_argmin_min(X, Y)` | `kneighbors(X, Y, k=1)` (§2.3) | SPT2 |
| `pairwise_distances_argmin(X, Y)` | `kneighbors(X, Y, k=1).indices` | SPT2 |
| `NearestNeighbors(...).fit(Y).kneighbors(X)`, `.radius_neighbors(X)` | `kneighbors(X, Y, k=)`, `radius_neighbors(X, Y, r)` (클래스는 두지 않는다) | SPT2 |
| `radius_neighbors_graph(mode='distance')` | `radius_neighbors(X, Y, r).tocsr()` | SPT2 |
| `euclidean_distances(X, Y, squared=)` | `cdist(X, Y, metric='euclidean' 또는 'sqeuclidean')` | SPT1 |
| `paired_distances` | 거리 함수의 배치(`sd.spatial.euclidean(U, V)`) | SPT1 |
| `haversine_distances` | `cdist(metric='haversine')`, `haversine(u, v)` (SPT9) | SPT1 |
| `sklearn.neighbors.KDTree` | sd `KDTree` (scipy의 API) | SPT3 |
| `BallTree` | §9의 3번 (추천: 나중) | 나중 |
| `kneighbors_graph`, `nan_euclidean_distances`, `KernelDensity`, `two_point_correlation` | 없음 (`count_neighbors`가 짝 세기를 한다) | — |

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다. 별칭이 없다.
- **같은 기계, 같은 스레드 수, 같은 입력이면 실행마다 같은 비트다**(API.md §3). 스레드 수와 무관한 비트는 2026-11-22부터 약속이 아니지만 이 영역은 지킨다: 모든 분할을 크기로만 정하고, 짝 하나의 거리는 어디서 계산하든 같은 커널이다(§3.5). **이웃의 집합, 동점의 순서, 결과의 정렬은 스레드 수뿐 아니라 경로(트리·전수·GEMM 고르기)와 Accelerate의 스레드 모드와도 무관하다.** 갈래가 결과의 모양을 바꾸는 곳이라 반올림의 차이와 달리 다른 프로그램이 되기 때문이다.
- 반올림이 아닌 동작(결과의 모양과 형, 기본값, 경계의 포함 여부, k > n의 채움, 정렬 순서)은 scipy 1.18.1과 같다(API.md §4의 규칙 2). 다르게 정한 곳은 `[scipy와 다름]`으로 이유와 함께 적는다. 반올림 크기의 결과는 정확도 계약(§4) 안이면 된다.
- 성능이 먼저이고 정확도가 그다음이다(API.md §4). 다만 GEMM 전개식이 내는 오차는 몇 ULP가 아니라 상쇄로 모든 자릿수를 잃는 크기라(§3.2의 sklearn 예) 계약을 그만큼 늦추는 것은 받아들일 수 있는 완화가 아니다. SPT2가 그 둘을 함께 얻는 길이다.
- 경고를 내지 않는다. scipy가 조용히 받는 틀린 인자(음수 반경, 음수 상한, `k=True`, `pdist`의 음수 `p`)는 에러다(§2.6).
- 에러 종류와 메시지는 API.md §2·§2.1이다. 메시지는 부른 함수의 이름으로 시작한다(`pdist(): ...`, 메서드는 `query(): ...`, 생성자는 `KDTree(): ...`).
- 참값은 Arb(python-flint 0.9.0)와 정확한 유리수만 쓴다. mpmath는 쓰지 않는다.
- 모든 호출은 정해진 시간 안에 끝난다. 반복이 없는 영역이라 작업량은 입력 크기와 결과 크기로 정해진다.

### 1.4 다른 영역과의 관계
- **linalg**: 거리 커널은 linalg의 합을 쓰지 않는다. SPT0에서 linalg와 같은 순서가 1.34–3.85배 느려 타일 순서로 바꿨기 때문이다(SPT22, §3.1). 그래서 `euclidean(x, y)`와 `sd.linalg.vector_norm(x − y)`는 n ≥ 16에서 끝자리가 다를 수 있다(n < 16과 `chebyshev`는 같다). `mahalanobis`의 기본 `VI`는 `sd.linalg.inv`(Python 층), `seuclidean`의 기본 `V`는 `sd.var(correction=1)`이다.
- **sparse**: `RadiusResult.tocsr()`와 `KDTree.sparse_distance_matrix`가 `sd.sparse.csr_array`를 만든다(Python 층에서만 의존한다; C++의 `spatial/`은 `sparse/`를 모른다).
- **API.md §7의 규칙**(희소에만 있을 이유가 없는 기능은 밀집에도 있다)은 이 영역에 짐이 없다: 희소 쪽에 새 기능을 더하지 않는다.
- **나중의 영역**: 군집(k-평균의 할당 단계가 `kneighbors(k=1)`), 그래프(`sd.graph`: 최소 신장 트리, 연결 요소 — 입력은 `.tocsr()`), 커널 밀도 추정이 이 영역을 쓸 수 있다. 약속이 아니라 메모다.

---

## 2. API (안)

### 2.1 입력
- **데이터**(`X`, `XA`, `XB`, `u`, `v`, `data`, 질의점 `x`): API.md §5의 상위 함수 입력이다(§9의 8번). `sd.ndarray`, 수의 시퀀스, numpy 배열, frame의 `DataFrame`(2차원; 결측은 NaN)을 받는다. 계산은 float64다.
  - bool·int64는 float64로 바꾼다(linalg의 int 입력 규칙과 같다, DESIGN §9.6; scipy도 바꾼다). 예외는 `hamming`·`jaccard`다: 입력의 dtype 그대로 비교하고, 두 입력의 dtype이 다르면 하나라도 float64면 float64, 아니면 int64로 맞춘다(§9의 18번, SPT23, SPT35).
  - complex128은 `TypeError`다(`[scipy와 다름]`: scipy의 `pdist`·`cdist`는 `ValueError: Unsupported dtype complex128`, `KDTree`는 `TypeError`. API.md §2는 받지 않는 dtype을 `TypeError`로 정한다).
  - 모양: `pdist`는 (m, n), `cdist`는 (mA, n)와 (mB, n)이고 n이 같아야 한다(`ValueError`). n = 0(특징이 없음)도 받는다: 거리는 그 거리의 정의를 빈 합에 쓴 값이다(`euclidean`·`cityblock`·`chebyshev`는 0, `cosine`·`correlation`·`braycurtis`·`hamming`은 0/0의 NaN; scipy와 같다). 행이 0이거나 1이면 `pdist`는 빈 배열이다(scipy와 같다).
- **유한하지 않은 값**: 거리 함수는 IEEE 규칙으로 퍼뜨린다(NaN → NaN, inf − inf → NaN, inf와 유한 값의 거리는 inf). 이웃 찾기(`KDTree`, `kneighbors`, `radius_neighbors`)는 데이터와 질의점이 모두 유한해야 한다(`ValueError`; scipy의 `KDTree`와 같다). NaN이 있으면 "가장 가까운"이 정해지지 않고, 트리의 경계 상자가 뜻을 잃는다(§9의 7번).
- **가중치 `w`**: 1차원, 길이 n, 모든 원소가 유한하고 0 이상(`ValueError`). `[scipy와 다름]`: scipy는 음수만 막고 NaN·inf 가중치를 받아 NaN·inf 거리를 낸다. `w`를 받는 거리는 scipy와 같다: `seuclidean`·`mahalanobis`를 뺀 모두다(그 둘에 주면 `TypeError`).
- **결과 dtype**: 거리는 float64, 색인은 int64, 셈은 int64(가중이면 float64)다. `squareform`만 입력의 dtype을 지킨다(scipy와 같다).

### 2.2 거리 함수
```
pdist(X, /, *, metric='euclidean', p=None, w=None, V=None, VI=None)
cdist(XA, XB, /, *, metric='euclidean', p=None, w=None, V=None, VI=None)
squareform(X, /, *, force='no', checks=True)
braycurtis(u, v, /, *, w=None)
canberra(u, v, /, *, w=None)
chebyshev(u, v, /, *, w=None)
cityblock(u, v, /, *, w=None)
correlation(u, v, /, *, w=None, centered=True)
cosine(u, v, /, *, w=None)
euclidean(u, v, /, *, w=None)
hamming(u, v, /, *, w=None)
haversine(u, v, /)
jaccard(u, v, /, *, w=None)
mahalanobis(u, v, VI, /)
minkowski(u, v, /, *, p=2.0, w=None)
seuclidean(u, v, V, /)
sqeuclidean(u, v, /, *, w=None)
```
- `metric`과 그 인자는 키워드로만 받는다(§9의 10번). `[scipy와 다름]`: scipy는 `pdist(X, 'cityblock')`처럼 위치로도 받고 거리의 인자를 `**kwargs`로 받는다. 그 거리가 쓰지 않는 인자(`metric='euclidean'`에 `p=3`)는 `TypeError`다(`pdist(): 'p' applies only to metric='minkowski'`). scipy는 같은 경우 바인딩의 시그니처 덤프를 낸다.
- `p`의 기본(None)은 `minkowski`에서 2다. `p`는 0보다 큰 실수나 inf다(`p < 1`은 삼각 부등식이 없는 준거리이고 scipy처럼 받는다). 0 이하와 NaN은 `ValueError`다(`[scipy와 다름]`: scipy의 `pdist`는 `p` 검사를 건너뛰어 `p=-1`에서 값을 내고 `p=0`에서 inf를 낸다; 거리 함수 `minkowski`는 막는다).
- `V`의 기본은 `pdist`가 `X`, `cdist`가 `XA`와 `XB`를 세로로 이은 것의 열 분산(`correction=1`)이고, `VI`의 기본은 같은 자료의 공분산의 역행렬이다(scipy와 같다; 관측이 차원 이하이면 `ValueError`, scipy의 메시지 뜻과 같다; 자료에 NaN·inf가 있으면 NaN 행렬이라 모든 거리가 NaN, 공분산이 특이하면 `LinAlgError`, SPT34). 받은 `V`는 1차원 길이 n이고 양수여야 한다(`[scipy와 다름]`: scipy는 음수 `V`를 받아 뜻 없는 값을, 0이면 inf를 낸다). `VI`는 (n, n)이다. 양정치가 아니면 이차식이 음수인 짝의 거리가 NaN이다(scipy와 같다).
- 모든 거리 함수는 (…, n) 배치를 받아 앞 축끼리 브로드캐스트하고 결과는 앞 축의 모양이다. 짝 하나는 `cdist`의 그 짝과 비트까지 같다.

**거리의 정의** (z = x − y를 원소마다 한 번 반올림한 것; 합의 순서는 §3.1)
| 이름 | 정의 | `w`가 있을 때 (scipy와 같은 뜻) | 경계의 값 |
|---|---|---|---|
| `euclidean` | ‖z‖₂ | √(Σ wᵢzᵢ²) | 넘침은 다시 비율을 맞춰 유한한 값(`[scipy와 다름]`: scipy는 1e200 좌표에서 inf. R도 inf) |
| `sqeuclidean` | Σ zᵢ² | Σ wᵢzᵢ² | 넘치면 inf(참값이 넘친다) |
| `cityblock` | Σ \|zᵢ\| | Σ wᵢ\|zᵢ\| | |
| `chebyshev` | maxᵢ \|zᵢ\| | max over wᵢ > 0 | NaN을 퍼뜨린다(`[scipy와 다름]`: scipy의 C 커널은 NaN을 건너뛴다: `[[nan, 0], [1, 5]]`의 거리가 5다; §9의 7번) |
| `minkowski` | (Σ \|zᵢ\|ᵖ)^(1/p); p = 1, 2, inf는 위의 셋과 비트까지 같다 | (Σ wᵢ\|zᵢ\|ᵖ)^(1/p), p = inf면 wᵢ > 0의 최댓값 | 넘침·언더플로는 다시 비율을 맞춘다(`[scipy와 다름]`: scipy는 (0,0)과 (3,4)의 p = 2000 거리가 inf) |
| `seuclidean` | √(Σ zᵢ²/Vᵢ) | 없음 | |
| `mahalanobis` | √(zᵀ VI z) | 없음 | 이차식이 음수면 NaN |
| `cosine` | 1 − x·y/(‖x‖‖y‖), [0, 2]로 자른다 | 가중 내적과 가중 노름 | 영벡터가 있으면 NaN(scipy와 같다) |
| `correlation` | 행마다 (가중) 평균을 뺀 뒤의 `cosine` | 가중 평균 Σwx/Σw | 상수 행은 NaN |
| `canberra` | Σ \|zᵢ\|/(\|xᵢ\| + \|yᵢ\|), 0/0 항은 0 | Σ wᵢ(…) | NaN을 퍼뜨린다(`[scipy와 다름]`: scipy의 함수 `canberra`와 가중 `pdist`는 `nansum`이라 NaN 항을 건너뛰고, 가중치 없는 `pdist`는 퍼뜨린다. scipy 안에서 둘이 다르다) |
| `braycurtis` | Σ\|zᵢ\| / Σ\|xᵢ + yᵢ\| | 두 합 모두 wᵢ를 곱한다 | 0/0은 NaN, 분모만 0이면 inf(scipy와 같다) |
| `hamming` | (Σ wᵢ[xᵢ ≠ yᵢ]) / Σ wᵢ, 기본 wᵢ = 1 | 같은 식 | NaN ≠ NaN이라 NaN끼리는 다른 값으로 센다(scipy와 같다) |
| `jaccard` | Σ wᵢ[(xᵢ≠0) xor (yᵢ≠0)] / Σ wᵢ[(xᵢ≠0) or (yᵢ≠0)] | 같은 식 | 분모가 0이면 0(scipy와 같다). NaN은 0이 아니다 |
| `haversine` (§9의 4번) | 2·asin(√(sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2))), 입력은 (위도, 경도) 라디안의 2열, 결과는 단위 구의 라디안(sklearn과 같은 뜻) | 없음 | 열이 2가 아니면 `ValueError` |

- **대칭**: fl(x − y) = −fl(y − x)이므로 모든 거리에서 d(x, y)와 d(y, x)는 비트까지 같다. 그래서 `pdist(X)`의 원소는 `cdist(X, X)`의 그 원소와 같다(scipy도 같다).
- **`squareform`**: scipy와 같다. 벡터 → 행렬은 길이가 n(n−1)/2가 아니면 `ValueError`, 빈 벡터는 (1, 1)의 0이다. 행렬 → 벡터는 `checks=True`면 대칭(값이 정확히 같음; NaN끼리는 같지 않다)과 0 대각을 검사한다. (1, 1)과 (0, 0)은 빈 벡터다. `force`는 `'no'`, `'tomatrix'`, `'tovector'`다. dtype을 지키고(bool·int64·float64), 3차원 이상은 `ValueError`다(§9의 11번).

### 2.3 한 번에 끝나는 이웃 찾기 (편의 함수와 축약)
```
kneighbors(XA, XB, /, *, k=1, metric='euclidean', p=None, w=None, V=None, VI=None, distance_upper_bound=inf)
radius_neighbors(XA, XB, r, /, *, metric='euclidean', p=None, w=None, V=None, VI=None, return_length=False)
```
- `XA`의 행마다 `XB`에서 가장 가까운 k개(또는 거리가 r 이하인 모두)를 찾는다. 전체 거리 행렬을 만들지 않는다(§3.6의 메모리).
- **결과는 정의로 정해진다**: `kneighbors(XA, XB, k=k)`는 `cdist(XA, XB)`의 각 행을 (거리, 열 번호)로 정렬해 앞 k개를 고른 것과 같고, 거리는 그 원소와 비트까지 같다. Minkowski 계열(`euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`, `minkowski` p ≥ 1)에서는 `KDTree(XB).query(XA, k=k, p=…)`와도 비트까지 같다. 어떤 길(전수, GEMM 고르기, 트리)로 계산할지는 거리 이름과 모양(mA, mB, n, k)으로만 고르고(§3.4), 길은 결과를 바꾸지 않는다.
- k = 1이 축약이다: sklearn의 `pairwise_distances_argmin_min(X, Y)`는 `kneighbors(X, Y)`의 `(distances, indices)`다(풀기 순서만 반대). `[sklearn과 다름]`: sklearn은 전개식이라 거리가 틀릴 수 있고(§3.2), 이 함수는 언제나 직접 계산한 거리다.
- 모든 거리 이름을 받는다(트리를 쓸 수 없는 거리는 전수와 GEMM 고르기로 한다). 결과의 모양과 채움은 `KDTree.query`의 규칙이다(§2.4).
- 이름은 sklearn `NearestNeighbors`의 두 메서드 이름이다(scipy에는 이 일을 하는 함수가 없다; §9의 2번).

### 2.4 `KDTree`
```
KDTree(data, /, *, leafsize=10)
KDTree.query(x, /, *, k=1, p=2.0, distance_upper_bound=inf)
KDTree.query_ball_point(x, r, /, *, p=2.0, return_length=False)
KDTree.query_ball_tree(other, r, /, *, p=2.0)
KDTree.query_pairs(r, /, *, p=2.0)
KDTree.count_neighbors(other, r, /, *, p=2.0, weights=None, cumulative=True)
KDTree.sparse_distance_matrix(other, max_distance, /, *, p=2.0)
```
- **클래스**: `KDTree`는 Python 클래스이고 C++ 트리(네이티브 객체)를 필드로 쥔다. frame의 `Series`, 희소 배열과 같은 데이터 구조의 예외다(ARCHITECTURE §3; sparse의 DOK가 네이티브 해시 맵을 쥐는 것과 같은 꼴). 계산은 네이티브 호출 한 번이다.
- **만들기**: `data`는 (n, m)이다(1차원과 3차원 이상은 `ValueError`, scipy와 같다). 데이터는 언제나 복사한다: 트리가 자기 사본을 잎 순서로 놓고 쥔다(`[scipy와 다름]`: scipy는 복사하지 않을 수 있어 만든 뒤에 원본을 고치면 답이 틀린다고 적는다. 희소 생성자가 색인을 언제나 복사하는 것과 같은 이유다). n = 0도 받는다(질의는 모두 채움 값). m = 0도 받는다(모든 거리가 0; scipy는 질의에서 `IndexError`다).
- **옵션**: `leafsize`만 둔다(1 이상의 int; bool·실수는 `TypeError`, `[scipy와 다름]`: scipy는 2.5를 2로 자른다). `balanced_tree`, `compact_nodes`, `copy_data`, `boxsize`는 없다(§9의 13번, 15번). 트리는 언제나 중앙값으로 나누고 노드 상자를 실제 점의 범위로 줄인다(scipy의 기본과 같은 꼴, §3.3). 결과는 `leafsize`와 무관하다(빠르기만 바뀐다).
- **속성**: `data`(쓰기 금지 (n, m) float64, 입력 순서), `n`, `m`, `leafsize`, `maxes`, `mins`(쓰기 금지 (m,)), `size`(노드 수). scipy의 `tree`, `indices`, `boxsize`는 없다. 피클할 수 있다(같은 데이터와 `leafsize`로 다시 만든다; 만들기가 결정적이라 같은 트리다).
- **`p`**: 1 이상의 실수나 inf다(scipy와 같다; 1 미만은 `ValueError`). p = 1, 2, inf는 `cityblock`, `euclidean`, `chebyshev`와, 그 밖의 p는 `minkowski(p=)`와 같은 커널이다.
- **`query`**: 데이터 점 가운데 x에 가장 가까운 k개다.
  - x는 (…, m)이다. 마지막 축이 m이 아니면 `ValueError`.
  - `k`는 1 이상의 int거나 1 이상의 int 시퀀스(1부터 센 순위의 목록, scipy의 뜻)다. 0, 음수, 빈 시퀀스는 `ValueError`, bool·실수는 `TypeError`다(`[scipy와 다름]`: scipy 1.18.1은 `k=True`와 `k=2.0`을 받고, `k=0`은 numpy 내부 메시지의 `ValueError`이며, `k=[0]`은 프로세스를 죽인다(세그먼트 오류, 이 초안에서 확인함)).
  - 결과는 `QueryResult(distances, indices)`이고 scipy처럼 `d, i = tree.query(x)`로 풀린다(§2.5). 모양은 scipy와 같다: k가 int면 (…, k)이고 k = 1이면 마지막 축을 없앤다. k가 시퀀스면 (…, len(k))다. x가 (m,)이고 k = 1이면 Python `float`와 `int`다(0차원 결과는 Python 스칼라, array.md §2.2; scipy는 `float`와 `numpy.int64`).
  - **순서**: 각 행은 (거리, 색인)의 사전 순서다. 같은 거리는 색인이 작은 것이 먼저다. 이것을 약속한다(`[scipy와 다름]`: scipy는 약속하지 않고 실제로도 다르다: 같은 점 6개에서 scipy의 순서는 `[1, 2, 3, 4, 5, 0]`, sklearn은 `[5, 2, 4, 0, 1, 3]`이다). 비교하는 거리는 돌려주는 값 그대로다(제곱 거리가 아니다; §3.4).
  - **채움**: 찾은 이웃이 k보다 적으면(k > n, 또는 `distance_upper_bound`) 남은 자리는 거리 inf, 색인 n이다(scipy와 같다. sklearn은 k > n이면 에러다).
  - **`distance_upper_bound`**: 거리가 이 값보다 **작은** 이웃만 돌려준다(scipy와 같이 엄격한 부등호; 거리가 꼭 1인 점은 상한 1에서 빠진다). 0 이상의 실수나 inf여야 한다(`[scipy와 다름]`: scipy는 p = 2에서 상한을 제곱해 비교하므로 −1이 1처럼 동작하고, NaN이면 모든 자리가 채움 값이다; §9의 14번).
- **`query_ball_point`**: 거리가 r **이하**인 데이터 점이다(scipy와 같이 경계를 넣는다). r은 0 이상의 실수나 inf(또는 x의 배치 모양으로 브로드캐스트되는 배열)이고 음수·NaN은 `ValueError`다(`[scipy와 다름]`: scipy는 r을 제곱해 비교하므로 r = −1이 r = 1처럼 동작하고, NaN이면 빈 목록이다). 판정은 돌려주는 거리 d̂ ≤ r이다(§3.4).
  - 결과는 `RadiusResult`다(§2.5). 질의점 x의 배치 축은 C 순서로 펼쳐 행이 된다. 점 하나(m,)도 행 하나의 결과다(`[scipy와 다름]`: scipy는 점 하나면 list, 여럿이면 list의 object 배열이다).
  - 행 안은 색인 오름차순이다. `return_sorted`는 없다(§9의 12번; `[scipy와 다름]`: scipy의 기본은 점 하나면 정렬하지 않고 여럿이면 정렬한다).
  - `return_length=True`면 행마다의 개수를 int64 배열(x의 배치 모양; 점 하나면 Python `int`)로 돌려준다(scipy와 같다). 색인을 모으지 않아 메모리가 적다.
- **`query_ball_tree(other, r)`**: `self`의 점마다 `other`의 점 가운데 거리가 r 이하인 것이다. 결과는 `RadiusResult`이고 모양은 (self.n, other.n)이다. `other.m`이 다르면 `ValueError`(scipy와 같다). `other.query_ball_point(self.data, r)`, `radius_neighbors(self.data, other.data, r)`와 같은 결과다.
- **`query_pairs(r)`**: i < j이고 거리가 r 이하인 짝이다. (k, 2) int64 배열이고 (i, j)의 사전 순서다(§9의 12번; `[scipy와 다름]`: scipy의 기본은 순서 없는 `set`이고 `output_type='ndarray'`면 정렬된 배열이다. `output_type`은 없다).
- **`count_neighbors(other, r)`**: `self`의 점 x₁과 `other`의 점 x₂의 순서쌍 가운데 거리가 r 이하인 것의 수다(scipy와 같다: `other`가 `self`면 (i, i)와 (i, j)·(j, i)를 모두 센다). r은 실수 하나나 1차원 배열이다. `cumulative=False`면 r은 줄지 않아야 하고 결과의 i번째는 r[i−1] < d ≤ r[i]의 수다(첫째는 d ≤ r[0]). 가중치는 scipy의 꼴(배열 하나는 같은 트리일 때만, 튜플 `(w_self, w_other)`의 한쪽은 None일 수 있다)이고 가중이면 짝의 가중치 곱의 합이다. 결과는 r이 실수 하나면 Python `int`(가중이면 `float`), 배열이면 int64(가중이면 float64) 배열이다(scipy와 같다).
- **`sparse_distance_matrix(other, max_distance)`**: 거리가 `max_distance` 이하인 짝의 거리를 담은 `sd.sparse.csr_array` (self.n, other.n)이다. `self.query_ball_tree(other, max_distance).tocsr()`와 같다. 거리 0인 짝은 저장된 0이다(§2.5). `[scipy와 다름]`: scipy의 기본은 `dok_matrix`이고 `output_type`으로 `dok_array`, `coo_array`, `dict`, 구조 배열을 고른다. sd에는 옛 `*_matrix`가 없고(sparse.md §1.1), 형식을 바꾸려면 `asformat`을 쓴다.
- `workers`, `eps`는 없다(§9의 5번, 6번).

### 2.5 결과 객체
- **`QueryResult(distances, indices)`**: `query`와 `kneighbors`가 함께 쓰는 namedtuple이다(`stardust._results`, DESIGN §9.7). 두 함수는 같은 것을 돌려주므로 이름이 하나다. 배열은 쓰기 금지다(API.md §2.2).
- **`RadiusResult`**: `query_ball_point`, `query_ball_tree`, `radius_neighbors`가 함께 쓰는 결과 클래스다(`stardust.spatial`의 `_result.py`; htest처럼 클래스를 내보내지 않는다, §9의 12번).
  - 필드: `indptr`(int64, 행 수 + 1), `indices`(int64), `distances`(float64), `shape`(행 수, 데이터 점 수). 행 i의 이웃은 `indices[indptr[i]:indptr[i+1]]`이고 색인 오름차순, 그 거리는 같은 자리의 `distances`다. 배열은 쓰기 금지이고 읽을 때 복사하지 않는다(API.md §2.2).
  - `tocsr()`: `sd.sparse.csr_array((distances, indices, indptr), shape=shape)`다. 색인이 이미 표준 꼴(정렬, 중복 없음)이라 sparse의 표준 꼴 규칙(sparse.md §2.3)을 그대로 만족한다. 부를 때마다 새 배열이다(희소 배열의 `data`는 쓸 수 있어야 하므로 거리를 복사한다; 색인도 희소 생성자의 규칙대로 복사하고 검사한다. 비용은 찾기에 비해 작다. 재서 크면 SPT6에서 다시 본다).
  - **거리 0인 짝**: 같은 점(자기 자신, 중복된 점)의 거리 0은 **저장된 0**으로 들어간다. `nnz`가 그 짝을 세고, `count_nonzero()`는 세지 않는다. 지우려면 `eliminate_zeros()`를 부른다. 이유: 희소 행렬의 저장 자리가 이웃 그래프의 변이다. 0을 지우면 중복된 점과 자기 자신이 이웃 목록에서 사라져 `indices`와 `.tocsr()`의 자리가 서로 다른 것을 말하게 된다. scipy의 `sparse_distance_matrix`도 0인 짝을 저장한다(이 초안에서 확인: 같은 두 점의 `coo_array`는 nnz 4, `dok_array`는 nnz 4에 `count_nonzero()` 0).
  - `repr`는 한 줄이다(`RadiusResult(shape=(1000, 5000), nnz=10234)`, API.md §2.3의 꼴). 회귀·검정·최적화의 결과가 아니므로 `summary()`는 없다.
- 결과 객체가 아닌 결과(`pdist`, `cdist`, `squareform`, `query_pairs`, `count_neighbors`, `return_length`)는 보통의 새 배열이라 쓸 수 있다.

### 2.6 scipy·sklearn과 다른 곳 (모음)
위의 `[scipy와 다름]`을 모은다. 이유는 그 자리에 있다.
- 거리: 별칭과 대문자 이름 없음, `metric`과 그 인자는 키워드만, 쓰지 않는 인자는 `TypeError`, complex는 `TypeError`, `p ≤ 0`·`w`의 NaN·inf·음수 `V`는 `ValueError`, `chebyshev`와 `canberra`는 NaN을 퍼뜨린다, `euclidean`·`minkowski`는 넘치지 않는다, 모든 거리 함수가 배치를 받는다, 호출 가능한 `metric`과 `out=` 없음(§9의 9번).
- 트리: 동점은 (거리, 색인), 데이터는 언제나 복사, `leafsize`만, `k`·`leafsize`는 int만, `k=[0]`·음수 반경·음수 상한은 `ValueError`, 반경 결과는 `RadiusResult`이고 언제나 색인 순서, `query_pairs`는 정렬된 배열, `sparse_distance_matrix`는 CSR, `workers`·`eps`·`boxsize`·`return_sorted`·`output_type` 없음.

---

## 3. 수치 방법 (안)

### 3.1 거리 커널
**짝 하나의 거리는 한 커널이 정한다.** `pdist`, `cdist`, 거리 함수, `kneighbors`·`radius_neighbors`, 트리의 잎, GEMM 고르기 뒤의 다시 계산이 모두 같은 함수 `pair_distance<Metric>(x, y, n, params)`의 순서로 계산한다. 그래서 같은 짝은 어디서든 같은 비트다(SPT2). 묶음 커널(여러 짝을 레지스터에 함께 두는 것)은 짝 하나의 덧셈 순서를 바꾸지 않는다.

- **차이**: zᵢ = fl(xᵢ − yᵢ). 레지스터에서 빼고 바로 쓴다(배열에 쓰지 않는다). `sd.subtract`와 같은 반올림 한 번이다.
- **합의 순서는 타일 순서다**(SPT22의 측정, SPT26): 짝마다 항을 kTileBlock = 32개씩 덩어리로 나누고, 덩어리 안은 0에서 시작해 `fma(zᵢ, zᵢ, t)`를 차례로, 덩어리의 합은 차례로 더한다(s = t₁, s = s + t₂, …). 묶음 커널은 x 4행 × y 8행의 32짝을 누산 레지스터 16개(레지스터 하나에 두 짝)에 두고, y 블록을 열 우선의 판(panel)으로 한 번 옮겨 둔다. 짝 하나를 계산하는 커널(트리의 잎, 다시 계산)도 같은 순서라 같은 비트다. 구현은 `csrc/spatial/metric.hpp`의 `tile<Op, R, W>`(R × 8짝: 누산기가 하나인 거리는 R = 4, 둘인 `braycurtis`·`jaccard`는 R = 2)와 `pair_scalar<Op, W>`이고, 연산(Op)마다 NEON과 스칼라의 같은 IEEE 연산(뺄셈, 절댓값, fma, 덧셈, 나눗셈, FMAX, 비교)으로 적어 레인과 스칼라가 같은 비트다. 남는 행과 열은 채워서 계산하고 버린다. SPT0의 시제품은 `tools/spatial_probe_pair_kernel.hpp`로 옮겼다(SPT36).
  - 합의 깊이(한 항이 거치는 반올림의 수)는 m = min(n, 32) + ⌈n/32⌉ − 1이다(§4의 계약). n < 16이면 덩어리 하나의 차례 합이라 `sd.linalg.vector_norm(x − y)`와 같은 비트다(탐침: n = 3, 8, 15에서 256짝 모두 같다). n ≥ 16에서는 linalg의 16레인 잎과 달라 끝자리가 다를 수 있다(탐침: 256짝 가운데 106–216짝이 같다). 둘 다 정확도 계약 안이다.
  - `sqeuclidean`은 같은 제곱합 s 그대로다. `euclidean`은 √s이고, s가 inf이거나 2⁻⁹⁶⁸ 아래면(NaN은 그대로) 신중한 경로다(최댓값의 지수로 고른 2ᵏ로 비율을 맞춘 같은 순서의 두 번째 패스, SPT27, SPT33). `sqeuclidean`도 2⁻⁹⁶⁸ 아래에서 같은 두 번째 패스를 하고 2²ᵏ를 곱한다(결과가 비정상 수일 수 있다). 참 제곱 거리가 넘치면 `sqeuclidean`의 inf가 맞는 값이다. `pdist`·`cdist`의 마무리(√, 나눗셈, 자르기)는 두 짝씩 NEON으로 하고 신중한 경로가 필요한 레인만 짝 하나씩 다시 한다(같은 비트).
  - `cityblock`은 같은 꼴의 |zᵢ| 합(가중이면 `fma(w, |z|, t)`), `minkowski(p)`는 |zᵢ|ᵖ(`std::pow`)의 같은 꼴의 합과 근이다. 근은 `pow(s, 1/p)`에 1/p가 2의 거듭제곱이 아니면 뉴턴 한 걸음을 더하고, 합이 inf이거나 2⁻⁹⁶⁸ 아래면 |zᵢ|/max|z|의 합과 max|z|·근이다(SPT30). p = 1, 2, inf는 `cityblock`, `euclidean`, `chebyshev`로 간다(비트까지 같다; 가중도).
  - `chebyshev`는 NEON 최댓값(`FMAX`가 NaN을 퍼뜨린다)이다. 정확한 연산이라 순서와 무관하고 결과는 최대 |zᵢ|를 올바르게 반올림한 값이다(반올림이 단조이므로). `vector_norm(ord=inf)`와 언제나 같은 비트다.
- **`vector_norm`과 같은 순서는 얼마나 비쌌나**(SPT0의 측정, `tools/spatial_probe.py`의 `kernels`): 두 순서를 `cdist` 꼴의 고리(`XB`를 L2에 드는 블록으로, 블록마다 `XA`의 모든 행)로 1스레드에서 쟀다. `vector_norm`의 순서(linalg의 `sq_leaf`와 같은 16레인 잎, 512 넘으면 pairwise 트리, 2048 넘으면 `cblas_ddot`)는 짝 하나가 누산 레지스터 8개를 쓰므로 묶음 커널이 x 한 행과 y 두 행(누산 16개)까지만 든다. 그 커널의 거리는 `sd.linalg.vector_norm(x − y)`와 모든 n에서 256짝 모두 비트까지 같았다(같게 만들 수는 있다). 시간(ms, 세 번 이상 가운데 가장 좋은 값; 부하 평균 1.5):

  | n | mA = mB | `vector_norm` 순서 | 타일 순서 | 배 |
  |---:|---:|---:|---:|---:|
  | 3 / 8 / 15 | 1024 | — | 0.24 / 0.57 / 1.03 | 같은 순서 |
  | 16 | 1024 | 2.20 | 1.10 | 2.01 |
  | 24 | 1024 | 3.85 | 1.63 | 2.37 |
  | 31 | 1024 | 5.60 | 2.09 | 2.68 |
  | 32 | 1024 | 3.60 | 2.16 | 1.67 |
  | 64 | 1024 | 6.67 | 4.37 | 1.53 |
  | 100 | 1024 | 10.3 | 6.88 | 1.50 |
  | 128 | 1024 | 13.4 | 8.64 | 1.56 |
  | 256 | 1024 | 25.0 | 17.7 | 1.41 |
  | 512 | 1024 | 48.5 | 36.0 | 1.35 |
  | 1024 | 512 | 24.0 | 17.9 | 1.34 |
  | 2048 | 512 | 48.9 | 36.3 | 1.35 |
  | 4096 | 256 | 68.8 | 17.9 | 3.85 |

  모든 길이에서 1.3배를 넘어 SPT22의 규칙대로 타일 순서로 바꿨다. 느린 까닭: 짝마다 덩어리를 접는 덧셈 8개, 16개에 모자라는 꼬리의 스칼라 `fma` 사슬(n = 17–31에서 가장 크다), y 한 행을 x 한 행에만 쓰는 읽기(타일은 판의 y 하나를 x 넷에 쓴다), BLAS 구간에서는 짝마다의 버퍼 쓰기와 호출이다. 타일 커널은 n = 512에서 짝·항마다 약 0.25주기로 NEON 4파이프의 약 80%다. 덩어리를 16, 64, 나누지 않음으로 바꿔도 시간은 ±2% 안이었다(SPT26).
- **가중 거리, `seuclidean`, `canberra`, `braycurtis`, `hamming`, `jaccard`**: 같은 타일 순서의 합이다. 항: `euclidean`의 가중은 `fma(wᵢ·zᵢ, zᵢ, s)`, `seuclidean`은 미리 구한 rᵢ = 1/Vᵢ로 `fma(rᵢ·zᵢ, zᵢ, s)`(scipy는 √(1/V)를 곱한 차이의 노름이다; 정확도는 §4), `braycurtis`는 두 합을 같은 덩어리에서 함께, `canberra`의 항은 |zᵢ| / (|xᵢ| + |yᵢ|)이고 분모가 0이면 0(분자도 0이다), `hamming`·`jaccard`는 개수를 int64로 정확히 센다(가중이면 float64 합).
- **`hamming`·`jaccard`의 bool 입력**: 행을 64비트 낱말로 묶어 두고(데이터마다 한 번) 짝마다 `xor`·`or`와 NEON `vcnt`로 센다. n비트의 짝 하나가 n/64 낱말이다. 정수 셈이라 순서와 무관하고 바이트 입력과 비트가 같다.
- **`cosine`·`correlation`**: 행마다 2ᵏ로 비율을 맞추고(`correlation`은 그 뒤 (가중) 평균을 빼고 다시 맞춘다) 노름으로 나눈 x̂를 데이터마다 한 번 구한다(행에만 기대므로 `pdist`와 `cdist`에서 같다). 짝마다는 x̂, ŷ의 (가중) `sqeuclidean` 타일 커널 하나이고 d = s/2를 2로 자른다(SPT29: (b)를 골랐다). SPT1에서 잰 두 안: (a) scipy의 1 − x·y/√(xx·yy)와 (b) ‖x̂ − ŷ‖²/2를 Arb와 견주면 거의 평행한 짝(d ≈ 5×10⁻¹⁷)의 상대오차가 (a) 6.2, (b) 4×10⁻⁹였고 빠르기는 같았다. 상수 행은 평균을 그 값 자체로 두어(가중이면 wᵢ > 0인 원소가 모두 같을 때) 언제나 NaN이다.
- **`mahalanobis`**: 짝마다 z를 만들고 tᵢ = VI의 i행·z(타일 순서의 `fma`), q = Σzᵢtᵢ(타일 순서), √q(q < 0이면 NaN)다. |q|가 2⁻⁹⁶⁸ 아래이거나 inf면 z를 2ᵏ로 비율을 맞춰 다시 한다. 짝 하나가 O(n²)이다(scipy도 같다; 1000² n = 16에서 1스레드 76 ms, scipy 75 ms). BLAS를 쓰면 비트가 Accelerate의 묶음에 따라 달라지므로 이 라이브러리의 커널이다. z 여러 개를 모은 NEON 타일은 SPT6의 후보다.
- **`haversine`**(§9의 4번, SPT31): d = 2·atan2(√h, √g), h = sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2), g = cos²(Δφ/2) cos²(Δλ/2) + sin²((φ₁ + φ₂)/2) sin²(Δλ/2)(= 1 − h, 상쇄 없이)다. Δφ/2, Δλ/2의 sin·cos는 `base/sincos.hpp`로 함께, cos φ는 행마다 한 번 구한다. 대척점 넷을 포함한 32짝에서 최대 2 ULP였다. 짝마다 sincos 둘, sin, atan2라 1스레드에서 sklearn(numpy의 묶음 libm)의 약 3배 느리다(2000² 153 ms 대 45 ms; 10스레드 18 ms). 묶음 sin·cos는 SPT6의 후보다.
- **`pdist`의 순서**: 압축 꼴(i < j, 행 우선)이다. 행을 R개씩 묶은 행 묶음이 열 판 (i₀ + 1)/8부터 끝까지를 타일로 계산하고 j > i만 쓴다(행 i의 압축 자리는 j를 따라 이어져 있어 두 짝씩 쓴다). 행 묶음을 짝·항이 2²⁰개쯤인 태스크로 나눈다(크기로만 정함). 짝 하나는 어디서 돌든 같은 계산이라 나누는 방식은 비트를 바꾸지 않는다.
- **`cdist`**: `XB`를 8행 판으로 한 번 옮겨 두고, (mA, mB) 출력을 행 블록 × 판 묶음의 태스크로 나눈다. 판 묶음은 256 KB(2¹⁵개 double), 행 블록은 태스크 하나가 짝·항 2²⁰개쯤이 되는 R의 배수다(크기로만 정함). 전체 행렬을 돌려주므로 GEMM은 쓰지 않는다(SPT2는 고르기에만 쓴다). 그래서 큰 n의 `cdist`는 직접 커널의 빠르기가 한계다(§5). 커널을 쓰지 않는 거리(`minkowski`, `mahalanobis`, `haversine`, int64와 가중 bool의 셈, 비트 묶음의 popcount)는 행 구간마다 짝 하나씩이다.

### 3.2 유클리드 혼합 계산 (SPT2)
GEMM은 후보를 빨리 고르는 데만 쓰고, 돌려주는 값과 경계의 판정은 모두 §3.1의 직접 커널이 한다. 여기서는 고르기가 참 후보를 빠뜨리지 않는다는 것을 보인다.

**왜 직접 계산인가**: 전개식은 ‖x‖²과 ‖y‖²이 거리²보다 훨씬 크면 상쇄로 자릿수를 다 잃는다. sklearn 1.9.1의 기록(`tests/spatial/data/sklearn.json`):
- `euclidean_distances([[1e8, 1e8], [1e8 + 1, 1e8]])`는 거리 1이 아니라 0이다.
- `pairwise_distances_argmin_min`은 1e8 + [0, 1)³의 200점과 질의 20개(`tests/spatial/points.py`의 `far/1e8`)에서 거리를 모두 0으로 내고, 가장 가까운 점을 20개 가운데 하나도 맞히지 못했다(참 거리 0.03–0.14).
- scipy의 `cdist`는 직접 계산이다(차례 합, 곱과 덧셈을 따로 반올림; n = 2–3000의 무작위 짝 780개가 모두 그 순서와 비트까지 같았다). 그러나 scipy의 `KDTree`는 다른 순서로 더해 같은 짝의 거리가 `cdist`와 n = 2에서 9%, n = 3에서 12%, n = 64에서 55%, n = 3000에서 96%의 짝이 다르다(`tests/spatial/data/scipy.json`의 `order/…`). 그래서 scipy에서는 트리의 k번째 거리와 `cdist`의 값이 다를 수 있다. sd는 한 커널이라 같다.

**기호와 모형**: x, y ∈ ℝⁿ는 float64 벡터, X = ‖x‖, Y = ‖y‖, D = ‖x − y‖(참값)다. GEMM이 주는 것은 ĝ = x·y의 계산값이고, n̂ₓ = fl(‖x‖²)은 §3.1의 제곱합 커널이 한 번 구한 값이다. 추정 Ê = fl(fl(n̂ₓ + n̂ᵧ) − 2ĝ). 직접 커널의 거리는 d̂ = fl(√ŝ), ŝ는 직접 제곱합이다. 부동소수 연산 하나는 fl(a ∘ b) = (a ∘ b)(1 + δ) + η, |δ| ≤ u, |η| ≤ 2⁻¹⁰⁷⁵이고(FMA도 반올림 한 번이다), 덧셈·뺄셈과 넘치지 않는 2의 거듭제곱 곱은 η = 0이다(점진적 언더플로에서 덧셈의 결과는 정확하다). 넘침은 따로 다룬다(아래의 대체 길). 아래에서 1 ≤ n ≤ 2²¹ − 2다(SPT28). 그 범위에서 γₖ ≤ ku(1 + 2⁻³⁰) ≤ (k + 1)u이고 γⱼ + ku ≤ γⱼ₊ₖ다.

**가정 A** (GEMM): 출력 원소 하나는 n개 곱의 어떤 순서·묶음의 합이고, 곱과 덧셈은 각각 반올림하거나 FMA로 합친다. Strassen 같은 빠른 행렬 곱이 아니다. 그러면 어떤 순서든 |ĝ − x·y| ≤ γₙ·Σ|xᵢyᵢ| + n·2⁻¹⁰⁷⁴ ≤ γₙXY + n·2⁻¹⁰⁷⁴다(Higham 2002, 3장의 내적 오차; 곱마다 η 하나가 뒤의 덧셈으로 많아야 (1 + γₙ)배 커진다). Accelerate의 `dgemm`은 이진 파일이라 이 가정을 소스로 보일 수 없다. 그래서 (i) 여유를 넉넉히 잡고, (ii) 적대적인 시험(§7)이 가정이 깨지는 입력을 찾게 하고, (iii) 깨진 것이 보이면 고르기의 GEMM을 이 라이브러리의 NEON 타일 곱으로 바꾼다(결과는 그대로이고 빠르기만 준다).

**보조정리 1** (전개식의 오차): |Ê − D²| ≤ γₙ₊₂·(X + Y)² + (n + 1)·2⁻¹⁰⁷³ =: Δ(x, y).
- 증명: 제곱합은 항이 모두 양수라 |n̂ₓ − X²| ≤ γₙX² + n·2⁻¹⁰⁷⁵(1 + γₙ), y도 같다. s = fl(n̂ₓ + n̂ᵧ) = (n̂ₓ + n̂ᵧ)(1 + θ₁), Ê = (s − 2ĝ)(1 + θ₂), |θᵢ| ≤ u(덧셈과 뺄셈은 η가 없고 2ĝ는 정확하다). s − 2ĝ − D² = (n̂ₓ − X²) + (n̂ᵧ − Y²) + θ₁(n̂ₓ + n̂ᵧ) − 2(ĝ − x·y)이므로 반올림의 몫은 γₙ(X² + Y²) + u(1 + γₙ)(X² + Y²) + 2γₙXY ≤ γₙ₊₁(X + Y)²이고(γₖ + u(1 + γₖ) ≤ γₖ₊₁), 언더플로의 몫은 4n·2⁻¹⁰⁷⁵(1 + γₙ)(1 + u) 이하다. Ê − D² = (s − 2ĝ − D²)(1 + θ₂) + θ₂D²이고 D ≤ X + Y이므로 γₙ₊₁(1 + u) + u ≤ γₙ₊₂, 4n·2⁻¹⁰⁷⁵(1 + γₙ)(1 + u)² ≤ (n + 1)·2⁻¹⁰⁷³에서 끝난다. ∎

**보조정리 2** (직접 계산의 아래 한계): D ≥ 2⁻⁵⁰¹이면 ŝ ≥ (1 − γₙ₊₆)D²이고 d̂ ≥ (1 − γₙ₊₇)D다.
- 증명: zᵢ = (xᵢ − yᵢ)(1 + δᵢ)(뺄셈은 η가 없다)이고 어떤 순서의 합이든 ŝ ≥ (1 − γₙ)Σzᵢ² − n·2⁻¹⁰⁷⁴ ≥ (1 − γₙ₊₂)D² − n·2⁻¹⁰⁷⁴다. D² ≥ 2⁻¹⁰⁰²이고 n < 2²¹이라 n·2⁻¹⁰⁷⁴ < 2⁻⁵¹D² = 4uD²이므로 ŝ ≥ (1 − γₙ₊₆)D²다. √는 상대오차를 줄이고(√(1 − t) ≥ 1 − t) 마지막 반올림이 u를 더한다. ∎ 신중한 경로(SPT27)는 2ᵏ로 비율을 맞춘 같은 순서의 합이고 비율의 곱과 되돌림이 정확하므로 같은 한계를 지킨다(비율을 맞춘 값에는 언더플로가 없어 언더플로의 몫도 없다). 이 보조정리는 합의 순서와 무관하다(m ≤ n).

**고르기의 규칙**: 질의 x와 후보 yⱼ에 대해 Lⱼ = (1 − γₙ₊₇)·√max(Êⱼ − Δⱼ, 0)는 Dⱼ ≥ 2⁻⁵⁰¹일 때 d̂ⱼ의 아래 한계다(두 보조정리). 그래서 **Lⱼ > T이면 d̂ⱼ > T**이고, j는 거리가 T 이하인 점들 가운데 있을 수 없다. 버리는 것은 이것뿐이다. 실제로 계산하는 값 L̃ⱼ와 그것이 Lⱼ 쪽의 아래 한계라는 증명은 아래(정리 3)다.
- **k개 이웃**(`kneighbors`, 트리 대신 전수로 갈 때의 `query`): `XB`를 블록으로 나눠 차례로 본다. 블록마다 GEMM으로 Ê를 구하고, 질의마다 지금까지의 (d̂, 색인) 최대 힙(크기 k)의 맨 위를 T로 한다(힙이 덜 찼으면 T = ∞이고, 첫 블록은 Ê가 가장 작은 k개를 먼저 직접 계산해 T를 정한다). L̃ⱼ > T가 아닌 j만 직접 다시 계산해 힙에 넣는다. T는 줄기만 하므로 버린 점은 끝까지 버려도 된다. 마지막 힙이 답이다: 버린 점은 d̂ > T ≥ (참 k번째 d̂)라서 답에도, k번째와의 동점에도 들 수 없다(엄격한 부등호라 같은 거리에 색인이 작은 점은 남는다). 그래서 결과는 같은 커널로 모든 점을 직접 계산해 (d̂, 색인)으로 정렬한 것과 같다.
- **반경**: T = r(판정은 d̂ ≤ r, 버리기는 L̃ > r). `distance_upper_bound`는 판정이 d̂ < 상한이라 L̃ ≥ 상한이면 버린다.
- **비용**: 여유의 상대 크기는 Δⱼ/D² ≈ (n + 3)u·(‖x‖ + ‖y‖)²/D²다. 점들이 원점 근처에 퍼져 있으면 아주 작아 다시 계산하는 후보가 k개를 거의 넘지 않는다. 점들이 원점에서 멀고 서로 가까우면(1e8 + [0, 1]³) 여유가 퍼짐보다 커져 거의 모든 점이 후보가 된다. 그래서 GEMM에 넣기 전에 **중심을 뺀다**: c = `XB`의 열 평균(호출마다 한 번)으로 x′ = fl(x − c), y′ = fl(y − c)를 GEMM에 쓴다. 옮김의 반올림은 원소마다 u|xᵢ − cᵢ| 이하라서 D ≥ ‖x′ − y′‖ − u(X′ + Y′)/(1 − u)다(X′ = ‖x′‖). 다시 계산은 언제나 옮기지 않은 원래 점으로 한다.

**계산하는 꼴과 상수** (SPT28; C++ 주석이 이 목록을 인용한다). 모든 상수는 정확한 double이다: (n + 3)u와 (n + 4)·2⁻¹⁰⁷⁰은 정수와 2의 거듭제곱의 곱, 1 − ku는 1 아래의 간격이 u라서, 1 + 2ku는 1 위의 간격이 2u라서 정확하다.
- 행마다 한 번(`XA`와 `XB`의, 중심을 뺀 뒤의 행 x′; 빼지 않으면 x′ = x): n̂ = Σx′ᵢ²(직접 커널의 제곱합), R = fl(fl(fl(√n̂)·c₃) + ρ), c₃ = 1 + 2(n + 6)u, ρ = 2⁻⁵²⁵.
- 짝마다:
  - Ê = fl(fl(n̂ₓ + n̂ᵧ) − 2ĝ), â = fl(Rₓ + Rᵧ)
  - Δ̃ = fl(fl(c₁·fl(â·â)) + η₀), c₁ = (n + 3)u, η₀ = (n + 4)·2⁻¹⁰⁷⁰
  - t = fl(Ê − Δ̃). t > 0이 아니면(NaN 포함) 버리지 않는다.
  - e = fl(fl(2⁻⁵⁰·â) + 2⁻¹⁰⁷⁴)(중심을 뺐을 때; 빼지 않으면 e = 0), v = fl(fl(√t) − e). v ≥ 2⁻⁵⁰⁰이 아니면 버리지 않는다(거리가 약 3×10⁻¹⁵¹ 아래인 자료는 고르기 없이 모두 다시 계산한다: 답은 같고 빠르기만 잃는다).
  - `euclidean`: L̃ = fl(c₂·v), c₂ = 1 − (n + 16)u. `sqeuclidean`: L̃ = fl(fl(v·v)·c₅), c₅ = 1 − (n + 24)u.
  - 비교는 NaN이면 거짓이 되게 쓴다(넘침으로 inf − inf가 나도 버리지 않는다).

**정리 3**: 위의 L̃는 L̃ ≤ d̂(`sqeuclidean`은 L̃ ≤ ŝ)를 만족한다. 그래서 L̃ > T이면 d̂ > T다.
- (1) R ≥ X′: n̂ ≥ (1 − γₙ)X′² − n·2⁻¹⁰⁷⁴에서 X′ ≤ √n̂·(1 + γₙ) + 2⁻⁵²⁶이다(√(a + b) ≤ √a + √b, n < 2²¹). r = fl(√n̂) ≥ √n̂(1 − u), fl(r·c₃) ≥ r·c₃(1 − u) − 2⁻¹⁰⁷⁵, R ≥ (fl(r·c₃) + ρ)(1 − u) ≥ √n̂·c₃(1 − u)³ + (ρ − 2⁻¹⁰⁷⁵)(1 − u)이고 c₃(1 − u)³ ≥ 1 + (2n + 8)u ≥ 1 + γₙ, (ρ − 2⁻¹⁰⁷⁵)(1 − u) ≥ 2⁻⁵²⁶이다.
- (2) Δ̃ ≥ Δ(x′, y′): A = X′ + Y′라 하면 â ≥ (Rₓ + Rᵧ)(1 − u) ≥ A(1 − u), fl(â·â) ≥ â²(1 − u) − 2⁻¹⁰⁷⁵, fl(c₁·…) ≥ c₁â²(1 − u)² − 2⁻¹⁰⁷⁴(c₁ < 1), Δ̃ ≥ (c₁â²(1 − u)² − 2⁻¹⁰⁷⁴ + η₀)(1 − u) ≥ c₁(1 − u)⁵A² + (η₀ − 2⁻¹⁰⁷⁴)(1 − u)다. c₁(1 − u)⁵ = (n + 3)u(1 − u)⁵ ≥ (n + 2)u(1 + 2⁻²¹)(1 − u)⁵ ≥ γₙ₊₂이고 (η₀ − 2⁻¹⁰⁷⁴)(1 − u) ≥ (n + 1)·2⁻¹⁰⁷³이므로 Δ̃는 보조정리 1의 한계 이상이다.
- (3) t > 0이면 D′² ≥ Ê′ − Δ̃ ≥ t/(1 + u)(뺄셈은 η가 없다; D′ = ‖x′ − y′‖)이고 D′ ≥ √t(1 − u) ≥ fl(√t)(1 − 2u)다.
- (4) 중심 빼기: D ≥ D′ − uA/(1 − u)이고 A ≤ â/(1 − u)다. e ≥ (2⁻⁵⁰â + 2⁻¹⁰⁷⁵)(1 − u) ≥ 8uâ(1 − u) ≥ uâ(1 + 5u) ≥ uA/((1 − u)(1 − 2u))이므로 D ≥ (1 − 2u)(fl(√t) − e)다(빼지 않으면 e = 0이고 D = D′).
- (5) v > 0이면 v ≤ (fl(√t) − e)(1 + u)이므로 D ≥ (1 − 2u)v/(1 + u) ≥ (1 − 3u)v다.
- (6) v ≥ 2⁻⁵⁰⁰이면 D ≥ 2⁻⁵⁰¹이라 보조정리 2가 서고 d̂ ≥ (1 − γₙ₊₇)(1 − 3u)v ≥ (1 − (n + 11)u)v다. L̃ = fl(c₂v) ≤ c₂v(1 + u)(정상 범위라 η가 없다)이고 c₂(1 + u) ≤ 1 − (n + 15)u이므로 L̃ ≤ d̂다. 제곱: ŝ ≥ (1 − γₙ₊₆)(1 − 3u)²v² ≥ (1 − (n + 13)u)v²이고 L̃ ≤ c₅v²(1 + u)² ≤ (1 − (n + 21)u)v²다. ∎
- 수치 확인(SPT0, NumPy의 `@`와 타일 순서의 Python 판): `far/1e8`, `far/1e15`, `far/1e8x16`, `nn/normal3`, `lattice/dyadic3`, `normal/64`의 질의 40개 × 점 80개, 중심 빼기의 켬과 끔에서 L̃ ≤ d̂가 아닌 짝은 없었고 L̃/d̂의 최댓값은 1 − 7×10⁻¹⁵였다(여유가 꼭 필요한 만큼에 가깝다). 중심을 빼지 않으면 먼 군집(1e8 둘레)에서는 t > 0인 짝이 없어 고르기가 아무것도 버리지 못한다.
- 여유를 줄이는 것(γ 대신 합의 깊이 m, (X + Y)² 대신 더 좁은 한계)은 빠르기에만 영향이 있다. 위의 상수는 SPT2에서 다시 계산 비율을 재 보고 필요하면 좁힌다(증명과 함께).
- **대체 길**: 노름 n̂이나 (‖x′‖ + ‖y′‖)²이 유한하지 않으면(좌표가 약 1e154를 넘음) 그 질의·블록은 GEMM 없이 직접 전수로 간다. 이웃 찾기의 입력은 유한하므로(§2.1) NaN은 오지 않는다.
- **결과는 GEMM과 무관하다**: 고르기가 참 후보를 빠뜨리지 않으므로 마지막 답은 직접 커널만으로 정해진다. 그래서 결과는 Accelerate의 스레드 모드(`VECLIB_MAXIMUM_THREADS`)와도 무관하다(linalg의 결과와 다른 점이다). GEMM 블록의 크기와 순서는 빠르기만 바꾼다.
- **어디에 쓰나**: GEMM 고르기는 n이 어느 크기(n₀, SPT2에서 잼; scipy의 `cdist`가 n = 3에서 sklearn의 GEMM보다 빨랐다, §5) 이상일 때의 `kneighbors`·`radius_neighbors`와 그 길로 가는 `query`다. `pdist`·`cdist`는 모든 거리를 돌려주므로 쓰지 않는다.
- **넓히기(나중)**: 같은 틀은 고르기를 위한 어떤 변환에도 쓴다. 변환의 반올림을 여유에 넣으면 된다: `seuclidean`·가중 `euclidean`(좌표에 √(1/V)·√w를 곱함), `mahalanobis`(VI의 Cholesky 인자로 옮김, VI가 양정치일 때), `cosine`(정규화한 행의 유클리드). 첫 판은 `euclidean`과 `sqeuclidean`만이다.

### 3.3 KD 트리 만들기
- **나누기**: 노드마다 점들의 범위가 가장 넓은 축(같으면 번호가 작은 축)을 고르고, 그 축의 좌표로 (값, 색인)의 전순서를 써서 가운데 자리 ⌊n_node/2⌋에서 `nth_element`로 나눈다(sklearn의 `IndexComparator`와 같은 비교). 전순서라 두 쪽의 점 **집합**이 하나로 정해진다(같은 좌표가 많아도). 그래서 트리의 모양은 데이터와 `leafsize`로만 정해지고, `nth_element`의 구현, 스레드 수, 만드는 순서와 무관하다.
  - scipy의 기본(`balanced_tree=True`)도 중앙값이다. scipy의 다른 길(미끄러지는 중점)은 두지 않는다(§9의 13번).
- **노드 상자**: 노드의 실제 점들의 최솟값·최댓값이다(scipy의 `compact_nodes=True`와 같은 꼴). 가지치기의 거리 한계는 이 상자로 구한다. 나누는 값과 같은 좌표가 양쪽에 있어도 상자가 실제 범위라 한계가 맞다.
- **배치**: 노드 크기가 n과 `leafsize`로만 정해지므로 노드 수와 앞순회 번호를 미리 셀 수 있다. 노드 배열을 한 번에 잡고 하위 트리를 태스크로 나눠 만든다(위 몇 단계는 호출 스레드, 그 아래는 풀; 각 하위 트리는 자기 자리에만 쓴다). 결과는 직렬로 만든 것과 바이트까지 같다.
- **데이터 사본**: 점들을 잎 순서로 옮긴 (n, m) 사본과 원래 색인(int64)을 쥔다. 잎의 점이 메모리에 붙어 있어 잎 안의 거리 계산이 이어 읽기다. 공개 속성 `data`는 입력 순서의 사본이다(§2.4; 같은 메모리를 두 번 쥐는 비용은 SPT3에서 재서, 크면 `data`를 처음 읽을 때 만든다).
- **비용**: O(n log n)이다. scipy의 n = 10⁶, m = 3 만들기가 246 ms(1스레드)였다(§5).

### 3.4 질의
- **거리와 비교**: 잎의 점은 §3.1의 직접 커널로 거리를 구한다. 순서와 판정은 돌려주는 값 d̂로 한다. p = 2에서 비교를 제곱합으로 먼저 하고(ŝ < ŝ_top이면 거의 언제나 d̂ < d̂_top), √는 경계(제곱합의 비교가 d̂의 동점일 수 있는 상대 4u 안)에서만 구한다. 제곱합만으로 순서를 정하면 √의 반올림이 두 값을 같게 만들 때 색인의 순서가 틀린다.
  - scipy는 제곱의 자리에서 r²과 비교한다. 그래서 돌려준 거리가 r과 같은데도 빠지거나 그 반대인 경계가 있을 수 있다(무작위 시험에서 r = 계산한 거리일 때 √(d²) ≤ r와 d² ≤ r²이 2,000번 가운데 460번(23%) 달랐다; `tests/spatial/data/scipy.json`). sd의 판정은 돌려주는 거리 그 자체라 결과와 거리가 언제나 맞는다.
- **가지치기는 보수적으로**: 상자까지의 거리 한계 lb도 부동소수로 구하므로 참 한계보다 조금 클 수 있다. 가지를 치는 조건은 lb > T·(1 + 4(m + 2)u)이고(엄격한 부등호), 그래서 T와 같은 거리(동점)이고 색인이 더 작은 점이 있는 노드는 치지 않는다. 여유는 빠르기에만 영향이 있다(아주 작다).
- **kNN**: 깊이 우선, 가까운 자식 먼저, 질의마다 크기 k의 (d̂, 색인) 최대 힙이다. scipy처럼 상자 거리를 차원마다 덧셈으로 고쳐 가며(증분) 구한다. `distance_upper_bound`는 처음 T다. k가 시퀀스면 max(k)개를 찾고 그 순위만 고른다.
- **반경**: 노드의 lb > r·(1 + 4(m + 2)u)면 치고, 상자의 먼 쪽 한계 ub·(1 + 4(m + 2)u) < r면 노드 전체가 안이다. 전체가 안인 노드도 거리를 돌려줘야 하므로 거리는 계산한다(개수만 묻는 `return_length`와 `count_neighbors`는 계산하지 않고 노드의 점 수를 더한다). 행마다 찾은 (색인, 거리)를 색인으로 정렬해 쓴다.
- **짝**(`query_pairs`, `query_ball_tree`, `sparse_distance_matrix`): 두 트리를 함께 내려가는 이중 트리 순회다(노드 쌍의 상자 사이 거리로 치고 넣는다). `query_pairs`는 한 트리 안에서 i < j만 센다. 결과는 행(i)마다 모아 정렬한다.
- **`count_neighbors`**: Gray–Moore의 이중 트리 셈이다(scipy와 같은 알고리즘의 틀). r 여럿은 한 번의 순회로 센다. 노드 쌍의 거리 범위가 r의 경계를 걸치지 않으면 점 수의 곱(가중이면 노드 가중치 합의 곱)을 그 칸에 한 번에 더한다. 걸치는 잎 쌍은 직접 d̂ ≤ rₖ로 센다. 정수 셈은 정확하다. 가중 셈은 float64 합이라 순서가 비트를 정한다: 순회를 고정된 깊이의 노드 쌍 태스크로 나누고 태스크의 부분합을 태스크 번호 순서로 더한다(크기로만 정함).
- **트리 대신 전수로 가도 된다**: 결과가 길과 무관하므로(§1.3) `query`는 모양(n, m, 질의 수, k)을 보고 GEMM 고르기의 전수로 가도 된다. m이 크면 KD 트리는 전수보다 느리다(scipy의 n = 10⁵, m = 16 무작위 점에서 질의 10⁴개 k = 10이 8.9초다. 같은 일의 GEMM은 1.6×10¹⁰ 플롭이라 수십 ms다). 문턱은 SPT6에서 잰다(§9의 16번).
- **병렬**: 질의점을 크기로만 정한 조각(예: 256점)으로 나눈다. 질의 하나는 한 태스크가 처음부터 끝까지 계산하므로 순회 순서가 스레드와 무관하다. 반경 결과는 조각마다 버퍼에 모은 뒤 조각 순서로 잇는다(`indptr`은 조각마다의 개수의 누적합).
- **`kneighbors`·`radius_neighbors`의 길**: 거리 이름과 모양으로만 고른다. Minkowski 계열이고 m이 작으면 `XB`로 트리를 만들어 질의한다. `euclidean`·`sqeuclidean`이고 m ≥ n₀면 GEMM 고르기의 전수다. 그 밖의 거리는 직접 전수다(`XB`를 L2 블록으로 나눠 질의 블록마다 훑고, 질의마다 (d̂, 색인) 힙). 어느 길이든 결과는 같다.

### 3.5 병렬화와 결정성
**같은 입력이면 스레드 수, 경로, Accelerate의 스레드 모드와 상관없이 같은 결과다.** 거리 값은 짝마다 한 커널이고(BLAS를 부르지 않으므로 Accelerate 모드와도 무관하다, SPT26), 이웃의 집합과 순서는 (d̂, 색인)의 전순서로 정해진다. 테스트는 1, 2, 3, 7, 10스레드에서 같은 바이트를 확인한다(§7).

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| `pdist`, `cdist` | 짝의 수로 고르게 나눈 행 구간, 타일 | 짝 하나는 어디서든 같은 커널이다 |
| 데이터마다의 준비(노름, 평균, 비트 묶음) | 행 구간 | 행 하나의 계산이다 |
| GEMM 고르기 | 질의 블록 × `XB` 블록 | 고르기는 답을 바꾸지 않는다(§3.2). 답은 직접 커널과 전순서로 정해진다 |
| 트리 만들기 | 하위 트리 | 노드의 점 집합은 전순서로 정해지고 자리는 미리 센 번호다(§3.3) |
| 질의(kNN, 반경, 개수) | 질의점 조각(크기로만) | 질의 하나는 한 태스크가 다 한다 |
| 짝·이중 트리 | 고정된 깊이의 노드 쌍 | 결과를 행으로 모아 정렬한다. 가중 셈은 태스크 번호 순서로 더한다 |

- 병렬 기준값은 배열의 규칙이다(array.md §10.4): 작업량(짝의 수 × n, 질의 수 × 추정 노드 방문)이 float64 연산 2¹⁵개 분량 이상이면 나누고, 그 이상이면 스레드 수와 상관없이 GIL을 놓는다(API.md §3).
- GEMM은 태스크 안에서 부른다. AMX는 클러스터마다 하나라 태스크를 늘려도 GEMM은 그만큼 빨라지지 않는다. SPT0의 측정(`tools/spatial_probe.py`의 `gemm`; C (256 × 2048) = A (256 × n) Bᵀ를 T개 스레드가 동시에 부름, 합한 GFLOP/s):

  | n | 모드 | T = 1 | 2 | 4 | 6 | 10 |
  |---:|---|---:|---:|---:|---:|---:|
  | 16 | 단일 스레드 | 258 | 360 | 470 | 541 | 601 |
  | 64 | 단일 스레드 | 348 | 610 | 726 | 766 | 824 |
  | 512 | 단일 스레드 | 401 | 735 | 824 | 873 | 900 |
  | 16 | 기본 | 258 | 346 | 473 | 530 | 602 |
  | 64 | 기본 | 551 | 634 | 789 | 830 | 895 |
  | 512 | 기본 | 769 | 851 | 908 | 915 | 931 |

  합한 속도는 4–6스레드에서 1스레드의 약 2.1–2.3배(단일 스레드 모드)로 거의 차고, 기본 모드에서는 호출 하나가 이미 AMX 둘을 쓴다(n = 512에서 1스레드 769). 그래서 GEMM 고르기는 태스크를 늘리는 것보다 블록을 크게 하는 쪽이 낫다. 태스크 수와 블록 크기는 SPT2에서 정한다(결과와 무관하다).

### 3.6 메모리
- 사용자가 모양으로 요청한 출력(`pdist`의 n(n−1)/2, `cdist`의 mA·mB, `kneighbors`의 mA·k)은 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2).
- 축약과 이웃 찾기의 작업 메모리는 입력과 무관한 상한이 있다: 태스크마다 GEMM 타일(안: 256 × 2048 float64 = 4 MB)과 힙. 10스레드에서 약 40 MB다. 전체 (mA, mB) 행렬을 만들지 않는다(SPT1).
- 반경 결과는 찾은 짝의 수에 비례한다(`indices` 8바이트 + `distances` 8바이트). 개수만 필요하면 `return_length`, `count_neighbors`가 색인을 모으지 않는다.
- `KDTree`: 잎 순서의 사본 n·m·8 + 색인 n·8 + 노드(상자 2m·8 + 32바이트) × 약 2n/`leafsize` + 입력 순서의 `data` n·m·8(§3.3).

---

## 4. 정확도 계약 (거리는 SPT1에서 잼, 이웃은 안)
참값은 Arb다. 입력은 float64라 차이와 제곱합을 Arb에서 정확히 구할 수 있고, 참 거리를 올바르게 반올림한 값 RN(d)과 견준다. ULP는 `bench`의 `ulp_distance`다. 계약 단위(오차 ÷ 한계)로 적고 경계는 1이다(sparse.md §4와 같은 꼴). 판정은 `tests/spatial/refdata.py`의 `bound()`·`units()` 하나이고 테스트(`tests/spatial/test_reference.py`)와 정확도 체계(`bench/accuracy/spatial.py`, 표 `bench/spatial/accuracy.md`)가 같이 쓴다.

- **거리** (SPT1에서 잼, SPT32. m은 합의 깊이: 타일 순서에서 m = min(n, 32) + ⌈n/32⌉ − 1, §3.1·SPT26. u = 2⁻⁵³, c_pow = 2u. 모든 한계에 2⁻¹⁰⁷⁴을 더한다. 잰 값은 `distances.npz` 645짝, `counting.npz` 80짝, `haversine.npz` 32짝의 최대다)
  | 거리 | 계약 | 잰 최대 ULP | 계약 대비 | 메모 |
  |---|---|---:|---:|---|
  | `chebyshev`, 가중 | RN(d)와 같다(0 ULP) | 0 | 1 | 최대 \|fl(zᵢ)\|가 올바른 반올림이다(계약 대비 1은 참값이 두 double의 한가운데인 짝) |
  | `euclidean`, 가중 | \|d̂ − d\| ≤ (γₘ₊₃/2 + u)·d | 2, 2 | 0.46, 0.43 | 항이 모두 음이 아니라 상쇄가 없다 |
  | `seuclidean` | (γₘ₊₄/2 + u)·d | 2 | 0.45 | 1/Vᵢ의 반올림까지 |
  | `sqeuclidean` | γₘ₊₂·s(가중 γₘ₊₃) | 3, 4 | 0.68, 0.46 | |
  | `cityblock`, 가중 | γₘ₊₂·s | 3, 3 | 0.36, 0.35 | |
  | `minkowski(p)` | (4u + (γₘ + 2c_pow)/p + c_pow)·d | 6 (p = 0.5), 3, 2, 2; 가중 p = 3에서 2 | 0.2, 0.17, 0.21, 0.4; 0.22 | SPT30의 근과 신중한 경로까지. libm의 `pow`는 SPT0에서 정상 범위 0.504 ULP 이하였다 |
  | `canberra`, 가중 | γₘ₊₃·d | 3, 2 | 0.36, 0.46 | 항마다 세 번의 반올림 |
  | `braycurtis`, 가중 | γ₂ₘ₊₃·d | 5, 7 | 0.35, 0.46 | 두 합과 나눗셈. 분모의 \|xᵢ + yᵢ\|는 한 번 반올림이라 상대로 적힌다 |
  | `cosine`, 가중 | u(3√d + (2m + 8)d) + (m + 4)²u² | d ≥ 10⁻⁸에서 15, 22 | 0.42, 0.25 | SPT29의 식. 참값이 0(평행)이거나 아주 작으면 절대오차 2.6×10⁻²⁵ 이하라 ULP는 크다 |
  | `correlation`, 가중 | `cosine`의 한계 + 2(ρₓ² + ρᵧ²), ρ = γ₂ₘ₊₂·max\|xᵢ\|·√(Σw)/‖x_c‖_w | — | 0.39, 0.21 | 평균의 반올림이 행을 1 방향으로 옮긴 몫이고 2차다. ρ가 큰 행(1e15 + 정수처럼 중심이 퍼짐보다 훨씬 먼 행)에서는 ULP가 크다(scipy는 같은 곳에서 상대 5×10⁻⁵) |
  | `mahalanobis` | min(E/d, √E) + u·d, E = γ₂ₘ₊₃·\|z\|ᵀ\|VI\|\|z\| | 2 | 0.29 | VI가 나쁜 조건이면 상대오차가 크다: 계약은 절댓값의 이차식 기준이다(n ≤ 16의 양정치 VI에서 잼) |
  | `hamming`, `jaccard` | 정확한 개수의 나눗셈 한 번(올바른 반올림); 가중이면 γ₂ₘ₊₁·d | 0, 0; 가중 2, 2 | 0.88, 0.93; 0.12, 0.12 | bool·int64·float64가 같은 개수 |
  | `haversine` | 20u·d(SPT31) | 2 | 0.09 | 대척점 넷 포함 |
- 비교 패키지(같은 짝에서 그 자리에서; `bench/spatial/accuracy.md`): scipy 1.18.1은 넘침·언더플로(1e160, 1e-160, 비정상 수의 짝)에서 inf·0·NaN을 내 `euclidean`·`seuclidean`·`minkowski`·`mahalanobis`·`cosine`·`correlation`의 계약을 넘고(scipy의 `minkowski` p = 1.5는 `edge/subnormal`에서 상대 1), 정상 범위의 짝에서는 계약 안이다. sklearn 1.9.1의 전개식(`euclidean_distances`)과 `cosine_distances`는 계약을 넘는다(1 − cos의 상쇄, 넘침). R 4.6.2의 `stats::dist`는 `euclidean`·`minkowski`가 넘침에서 inf다.
- **이웃 찾기** (세 겹)
  1. **같은 커널의 전수와 정확히 같다**: `kneighbors`, `query`, `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, `sparse_distance_matrix`의 결과(색인, 순서, 거리의 비트, 개수)는 같은 직접 커널로 모든 짝을 계산해 (d̂, 색인)으로 고른 것과 같다. 길(트리, GEMM 고르기, 전수)마다, 스레드 수마다 같다. 이것이 SPT2의 여유(§3.2)와 가지치기의 여유(§3.4)를 시험한다.
  2. **Arb로 정렬한 전수와 정확히 같다**(거리가 올바르게 반올림되는 데이터에서): 정수 격자(차이의 제곱합이 2⁵³ 아래: 3차원 좌표 \|x\| < 2²⁴), 2진 유리수 격자, 같은 점이 많은 데이터에서는 차이와 제곱합이 정확하고 √가 올바른 반올림이라 d̂ = RN(d)다. 그런 데이터에서 결과는 Arb로 구한 (RN(d), 색인)의 정렬과 정확히 같아야 한다. 격자는 참 거리가 같은 짝(동점)이 아주 많아 동점 순서의 약속을 직접 시험한다.
  3. **일반 데이터에서 참 kNN이다**(거리 계약 η 안에서): 돌려준 k번째 거리 d̂₍ₖ₎에 대해, 빠진 모든 점의 참 거리는 d̂₍ₖ₎(1 − η) 이상이고 돌려준 모든 점의 참 거리는 d̂₍ₖ₎(1 + η) 이하다(Arb의 참 거리로 확인). 일반 데이터에서는 참 거리가 1 ULP 안에서 다른 두 점의 d̂ 순서가 참 순서와 다를 수 있다. 1번과 3번이 그 경우의 약속이다: 순서는 언제나 돌려준 값의 순서이고, 그 값은 거리 계약 안이다.
- scipy·sklearn·R과의 차이는 계약이 아니라 정확도 표의 비교 패키지 열로 보인다. 이웃은 Arb의 kNN과 같은 집합인 비율을 비교 패키지마다 적는다(sd는 2번의 데이터에서 100%가 계약이다).

---

## 5. 성능 (안)
SPT0에서 다시 잰 scipy 1.18.1과 sklearn 1.9.1의 시간이다(`tools/spatial_probe.py`의 `time`; ms, 세 번 중 가장 좋은 값(m = 16의 `query`는 한 번), 기본 설정; scipy의 거리와 트리 커널은 1스레드, sklearn의 `euclidean_distances`와 전개식은 BLAS의 다중 스레드, 부하 평균 1.4–1.5). 무작위 표준정규(거리)와 [0, 1)ᵐ 균등(트리). 처음 초안의 값(부하 평균 약 4)과 3–10% 안에서 같았다. 사례가 생기면(SPT1–SPT4) `bench/speed`가 같은 크기를 잰다.

| 일 | scipy | sklearn | 메모 |
|---|---:|---:|---|
| `cdist` euclidean 1000 × 1000, n = 3 | 0.71 | 1.21 (GEMM) | 작은 n은 직접 계산이 GEMM보다 빠르다 |
| `cdist` euclidean 1000 × 1000, n = 64 | 10.0 | 1.67 | |
| `cdist` euclidean 4000 × 4000, n = 3 | 11.3 | 20.8 | |
| `cdist` euclidean 4000 × 4000, n = 64 | 161 | 21.9 | |
| `cdist` euclidean 4000 × 4000, n = 512 | 1484 | 40.9 | `cityblock`은 scipy 1914 |
| `pairwise_distances_argmin_min` 4000 × 4000, n = 3 / 64 / 512 | — | 7.0 / 11.8 / 31.3 | 전개식(§3.2의 오차) |
| `pdist` euclidean m = 2000, n = 3 / 64 | 1.57 / 21.9 | — | `cosine` 3.14 / 46.5 |
| `KDTree` 만들기 n = 10⁵ / 10⁶, m = 3 | 18.1 / 228 | 14.8 / 206 | |
| `query` 10⁴점, n = 10⁶, m = 3, k = 1 / 10 | 10.1 / 23.5 | 14.8 / 31.7 | scipy `workers=-1`은 2.0 / 3.8 |
| `query` 10⁴점, n = 10⁵, m = 16, k = 1 / 10 | 3547 / 8675 | 3244 / 6020 | scipy `workers=-1`도 442 / 1130. 전수 GEMM의 영역이다(§3.4) |
| `query_ball_point` 10⁴점(평균 9.9개), n = 10⁶, m = 3 | 27.5 | — | |

**SPT1의 거리**(2026-11-22; ms, 다섯 번(n = 512는 두 번) 가운데 가장 좋은 값, 무작위 표준정규, 부하 평균 1.6–1.8. 기준선이 아닌 개발 중의 확인이고, 속도 사례 `bench/speed/cases/spatial.py`는 `--smoke`로 한 번씩만 불러 결과를 견줬다):

| 일 | sd 1스레드 | sd 10스레드 | scipy |
|---|---:|---:|---:|
| `cdist` euclidean 1000², n = 3 | 0.589 | 0.101 | 0.663 |
| `cdist` euclidean 1000², n = 64 | 4.46 | 0.649 | 9.45 |
| `cdist` euclidean 4000², n = 3 | 10.5 | 1.36 | 11.1 |
| `cdist` euclidean 4000², n = 64 | 74.5 | 8.51 | 155 |
| `cdist` euclidean 4000², n = 512 | 564 | 62.7 | 1450 |
| `cdist` cityblock 4000², n = 512 | 545 | 61.3 | 1910 |
| `cdist` cosine 1000², n = 64 | 4.92 | 0.753 | 21.9 |
| `cdist` canberra 1000², n = 64 | 25.6 | 3.33 | 27.5 |
| `cdist` minkowski p = 3 1000², n = 64 | 332 | 37.4 | 305 |
| `pdist` euclidean m = 2000, n = 3 / 64 | 1.24 / 9.14 | 0.485 / 1.17 | 1.45 / 19.5 |
| `pdist` cosine m = 2000, n = 64 | 9.25 | 1.27 | 43.3 |
| `cdist` hamming bool 1000², n = 256 | 1.15 | 0.261 | 63.2 |

- 1스레드에서 scipy 이상이 아닌 것은 `minkowski`(짝·항마다의 `pow`가 시간을 정하고, 짝마다의 뉴턴 걸음이 `pow` 하나를 더한다: 9% 느림), `haversine`(sklearn의 묶음 libm보다 약 3배 느림, §3.1), `mahalanobis`(1000² n = 16에서 76 대 75 ms)다. SPT6에서 다시 본다. n = 512의 `cdist`는 §5의 어림(1스레드 0.55초, 10스레드 60 ms)과 같다.
- n = 3에서 처음에는 짝마다의 마무리(합을 꺼내 스칼라 √와 신중한 경로 검사, 결과 복사)가 시간의 대부분이라 scipy보다 느렸다(4000² 18.5 대 11.5 ms). 마무리를 두 짝씩 NEON으로 하고 출력에 바로 쓰게 바꿔 10.5 ms가 되었다(비트는 같다).

- **목표**(SPT6에서 확인한다)
  - `pdist`·`cdist`: 1스레드에서 scipy 이상(같은 직접 계산에 NEON), 10스레드에서 거의 스레드 수만큼. n이 큰 `cdist`는 sklearn의 GEMM만큼 빠를 수 없다: 모든 거리를 직접 계산하므로 n·mA·mB의 뺄셈과 FMA가 하한이다(n = 512, 4000²에서 짝·항 8.2×10⁹개; SPT0의 타일 커널은 1스레드에서 짝·항마다 약 0.07 ns(§3.1의 n = 512)라 1스레드 약 0.55초, 10스레드 약 60 ms를 어림한다). 이것은 SPT2가 정한 정확도의 값이다.
  - `kneighbors`(k = 1, n ≥ n₀): sklearn `pairwise_distances_argmin_min`의 1.5배 안(같은 GEMM에 다시 계산과 중심 빼기를 더한다).
  - 트리: 만들기와 질의 모두 scipy `workers=-1`보다 빠르게(sd는 기본으로 10스레드다).
- **나중에 볼 것**: GEMM 고르기를 다른 거리로 넓히기(§3.2), 작은 m(2, 3)의 트리 잎에서 짝 방향 SIMD, 질의를 공간 순서로 정렬해 캐시를 살리기(결과는 원래 순서로 돌려놓는다).

---

## 6. 층과 파일 (안)
- **층**: `spatial/`은 `array/`, `accelerate/`(`blas.hpp`의 `gemm`: §3.2의 고르기), `parallel/`, `base/`(`sincos.hpp`) 위에 있다. 합의 순서가 이 영역의 타일 순서라(SPT22, SPT26) `linalg/`의 합과 노름에 기대지 않는다. 다른 영역은 `spatial/`을 모른다. Python 층의 `stardust.spatial`은 `stardust.sparse`(`.tocsr()`), `stardust.linalg`(`inv`), 배열의 `var`를 쓴다.
- `bench/areas.py`의 끝에 `spatial`이 있다(영역 열셋째). ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표에 한 줄씩 있다(SPT0).
- `csrc/spatial/`
  - `metric.hpp`(SPT1): 거리마다의 연산(Op)과 두 커널 `tile<Op, R, W>`·`pair_scalar<Op, W>`, 신중한 경로, `minkowski`·`mahalanobis`·`haversine`의 짝 함수, 데이터마다의 준비(정규화, 평균, 비트 묶음). SPT0의 시제품 `pair_kernel.hpp`는 `tools/spatial_probe_pair_kernel.hpp`로 옮겼다(SPT36)
  - `distance.cpp`(SPT1): `_pdist`, `_cdist`, `_paired`(배치 거리 함수), `_squareform_tomatrix`, `_squareform_tovector`와 구동기(타일, 짝 하나씩, 태스크 나누기), `spatial.hpp`가 `bind_*`를 알린다
  - `hybrid.hpp`: §3.2의 GEMM 고르기(중심 빼기, 여유, 블록 순회)
  - `heap.hpp`: (d̂, 색인) 최대 힙과 행마다의 반경 버퍼
  - `kdtree.hpp`, `kdtree.cpp`: 만들기(§3.3), 노드 배열, 피클의 상태
  - `query.cpp`: kNN, 반경, 개수(§3.4)
  - `dual.cpp`: 이중 트리의 짝과 `count_neighbors`
  - `brute.cpp`: `kneighbors`·`radius_neighbors`의 전수 길
- 바인딩 `bind/spatial_bind.cpp`(SPT1): `_core.spatial` 하위 모듈.
- `src/stardust/spatial/` (하위 패키지 없음): `__init__.py`(공개 이름), `_common.py`(입력 정리, `metric`과 인자의 검사, 메시지; SPT1), `_distance.py`(`pdist cdist squareform`과 거리 함수, 기본 `V`·`VI`; SPT1), `_neighbors.py`(`kneighbors radius_neighbors`), `_kdtree.py`(`KDTree`), `_result.py`(`RadiusResult`). `QueryResult`는 `stardust/_results.py`에 더한다.

---

## 7. 참값과 검증 (안)
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다. 긴 참값 생성은 프로세스 10개까지, 메모리를 지켜보며(RSS가 평평한지) 돈다.

### 7.1 시험 데이터 (`tests/spatial/points.py`)
- 무작위: 표준정규, 균등, 섞인 군집(가우스 혼합), 저차원 다양체(높은 m에 낮은 내재 차원), m = 0, 1, 2, 3, 8, 16, 64, 512, 2048, 2049, 4096(합의 길이 경계: 16, 512, 1024, 2048).
- **적대적**(SPT2의 여유): 원점에서 먼 작은 군집(1e8 + [0, 1]ᵐ, 1e15 + 정수), 거리가 1 ULP씩 다른 점들(질의 둘레의 껍질), 같은 점의 무더기, k번째 경계에 동점이 많은 데이터, 노름이 넘치기 직전(1e150)의 좌표, 아주 작은 좌표(1e-160, 신중한 경로).
- **정수와 2진 격자**: §4 이웃 2번의 데이터. 참 거리의 동점이 아주 많다.
- 거리 함수의 경계: NaN·inf의 모든 자리, 0 벡터, 상수 행, 0/0 항(`canberra`, `braycurtis`, `jaccard`), 음수 좌표, `w`에 0이 섞임, 양정치가 아닌 VI, 대척점(`haversine`).

### 7.2 참값 (`tools/spatial_ref.py` → `tests/spatial/ref/`, `tests/reffile.py`의 npz)
- Arb(python-flint 0.9.0)로 짝마다 참 거리와 RN(d)를 구한다. 차이와 제곱합은 입력이 float64라 충분한 정밀도에서 정확하다. `pow`, `asin`, `sin`이 드는 거리는 Arb의 구간으로 구하고 반올림이 정해질 때까지 정밀도를 늘린다.
- 이웃의 참값: 2번 데이터의 (RN(d), 색인) 정렬, 일반 데이터의 참 거리 표(3번의 확인용). 파일은 작게 둔다(점 수천 개).

### 7.3 테스트 (`tests/spatial/`)
- `test_api_surface.py`(API 문서의 시그니처), `test_distance.py`(거리마다의 정의, `w`, `V`, `VI`, 경계의 값, 배치, 대칭, `pdist`와 `cdist`의 비트, scipy와 계약 안), `test_squareform.py`, `test_pair_kernel.py`(§3.1의 타일 순서: 묶음 커널과 짝 하나의 비트, 덩어리 경계 31·32·33과 길이 경계마다, n < 16에서 `vector_norm`과 같은 비트, `chebyshev`는 언제나 같은 비트), `test_neighbors.py`(`kneighbors`·`radius_neighbors`가 같은 커널의 전수와 정확히 같음: 길마다, 거리마다), `test_hybrid.py`(§7.1의 적대적 데이터와 hypothesis로 GEMM 고르기가 답을 바꾸지 않음; 고르기를 끈 비공개 길과 비교), `test_kdtree.py`(만들기의 불변식, 노드 상자, 피클, 스레드마다 같은 트리), `test_query.py`(k, k 시퀀스, 모양, 채움, 상한, 동점 순서, scipy와 집합이 같음), `test_radius.py`, `test_pairs.py`, `test_count.py`(가중 포함), `test_reference.py`(§4 이웃의 2번과 3번, Arb 참값), `test_errors.py`(§2의 모든 에러와 메시지), `test_threads.py`(1, 2, 3, 7, 10스레드에서 같은 바이트; `VECLIB_MAXIMUM_THREADS` 1과 기본에서 이웃 결과가 같음).
- SPT1에서 만든 것: `test_distance.py`(scipy 1.18.1과 그 기록, sklearn의 `haversine`, R의 `dist`, 경계의 값, 배치, dtype, 입력), `test_pair_kernel.py`(묶음 커널과 짝 하나의 비트를 모든 거리·길이 경계·남는 행과 열·NaN과 inf에서, hypothesis 포함; n < 16의 `vector_norm`, `chebyshev`와 `ord=inf`, p = 1, 2, inf; 신중한 경로), `test_squareform.py`, `test_reference.py`(Arb 참값의 모든 점이 계약 안), `test_errors.py`, `test_threads.py`(1, 3, 10스레드에서 같은 바이트).
- hypothesis는 tests/README.md의 프로필을 쓴다(quick은 짧게, deep은 길게). 적대적 생성기는 군집의 중심, 퍼짐, 동점의 수, m, k를 고른다.
- scipy·sklearn과는 그 자리에서 견준다: 거리는 계약 안, 이웃은 동점이 없는 데이터에서 같은 집합과 순서, 동점이 있으면 같은 집합(scipy의 순서는 약속이 아니다).

### 7.4 scipy·sklearn·R의 기록 (`tools/spatial_probe.py` → `tests/spatial/data/`)
- 설계 초안의 탐침(scipy의 거리 경계, `KDTree`의 경계, 커널의 덧셈 순서, sklearn의 경계, 0의 저장, 시간)을 도구 하나로 옮겼다(SPT0). `k=[0]`처럼 프로세스를 죽이는 호출은 자식 프로세스에서 부른다(기록: 신호 11로 죽는다).
- 기록(SPT0, 2026-11-22): `scipy.json`(scipy 1.18.1; 거리 열셋 × 가장자리 짝 27개의 함수·`pdist`·가중 `pdist`, 문서의 `[scipy와 다름]` 하나하나, 덧셈 순서, `KDTree`의 경계 스물여섯, 이웃 집합 열둘의 k = 5 이웃; 1,074개), `sklearn.json`(scikit-learn 1.9.1; 전개식의 오차, 동점의 순서, 에러, `haversine_distances` 32짝; 41개), `r.json`(R 4.6.2 `stats::dist`, 점 집합 넷 × 방법 여섯과 가장자리 셋; 27개). 모든 기록은 그 점의 `digest`(tests/spatial/points.py)를 가진다.
- 참값(SPT0): `distances.npz`(645짝, 거리 스물셋), `counting.npz`(80짝, `hamming`·`jaccard`와 가중), `haversine.npz`(32짝), `neighbors.npz`(질의 308개 × 데이터 전부의 p = 1, 2, inf 거리)를 `tools/spatial_ref.py`가 4프로세스로 4초에 만든다(4.8 MB). 떨어진 점은 없다. scipy와 견주면 넘치거나(1e160) 아주 작은 좌표에서 scipy가 inf·0을 내는 것과 `correlation`의 상쇄(1e15 둘레에서 상대 5×10⁻⁵)를 빼고 모두 ε 크기 안이다.
- R은 `stats::dist`(`euclidean`, `manhattan`, `maximum`, `canberra`, `binary`, `minkowski`)만 비교 패키지다. R의 `canberra`는 0/0 항과 NA를 빼고 남은 항의 수로 다시 비율을 맞추므로(이 초안에서 확인) 그 거리의 R 열은 그런 항이 없는 점만 잰다.

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/spatial.py`): 스윕 `distances`(거리마다, 점은 짝 하나, 판정은 §4의 계약, 비교 패키지 scipy·sklearn(`euclidean_distances`는 전개식의 오차를 그대로 보인다)·R)와 `neighbors`(kNN과 반경의 Arb 일치율; sd는 계약이 100%이고 scipy·sklearn은 기록).
- **속도** (`bench/speed/cases/spatial.py`): 사례마다 마일스톤과 필요한 이름을 적어 두고 그 마일스톤이 끝나면 사례가 된다(sparse의 사례 파일과 같은 틀). 거리마다의 `pdist`·`cdist`(§5의 크기), `kneighbors` k = 1, 10(sklearn `pairwise_distances_argmin_min`, `NearestNeighbors(algorithm='brute')`와), `KDTree` 만들기, `query` k = 1, 10, `query_ball_point`, `query_pairs`, `count_neighbors`(r 하나와 20개), `sparse_distance_matrix`. scipy는 `workers=1`과 `workers=-1` 둘 다 비교 대상이다(sd는 기본이 10스레드라서). 결과는 이웃의 색인과 거리를 견준다.
- **골든** (`tests/golden/cases_spatial.py`, 기록 `tests/golden/data/spatial.json`): 모든 거리 × 경계 데이터, `squareform`, 이웃 함수의 모든 꼴, 틀린 호출. 1·3·10스레드의 결과가 모두 같아야 한다. 큰 케이스는 병렬 길과 GEMM 고르기를 탈 크기로 두되 워커 하나의 메모리가 1 GB를 넘지 않게 한다.

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| SPT0 준비 | API 문서 초안(`docs/api/spatial.md`: 모든 공개 이름의 시그니처, scipy·sklearn → sd 대응표, §9의 답), 영역 등록(`bench/areas.py`, ARCHITECTURE, API.md의 색인), 시험 데이터(§7.1), 참값 도구(§7.2), 탐침 도구와 기록(§7.4), §3.2의 상수와 증명의 마무리, 짝 커널의 두 안(§3.1의 `vector_norm`과 같은 순서 대 타일 순서)과 GEMM의 동시 호출(AMX) 측정, 속도 사례의 틀 | 문서와 도구가 커밋되고 탐침 기록이 있다. §9의 답이 §1.1의 결정으로 옮겨졌다 |
| SPT1 거리 — **완료·승인(위임, 2026-11-22)** | `pdist cdist squareform`, 거리 함수 열넷(`haversine` 포함, SPT9), `w`·`V`·`VI`, 배치, bool의 비트 묶음 | 정확도 스윕이 계약 안, 타일 순서의 비트 테스트(SPT22·SPT26: 묶음 커널과 짝 하나, n < 16에서 `vector_norm`), 골든, scipy·R과의 속도 사례 |
| SPT2 축약과 전수 이웃 | `kneighbors radius_neighbors`, `QueryResult`, `RadiusResult`와 `tocsr()`, GEMM 고르기(§3.2)와 직접 전수 | 같은 커널의 전수와 정확히 같음(모든 거리, 적대적 데이터, hypothesis), Accelerate 모드와 무관함, sklearn과의 속도 사례 |
| SPT3 KD 트리와 kNN | `KDTree`(만들기, 속성, 피클), `query`(k, k 시퀀스, 상한, p) | §4 이웃의 1–3번, 스레드마다 같은 트리와 결과, scipy와의 속도 사례 |
| SPT4 반경·짝·개수 | `query_ball_point query_ball_tree query_pairs count_neighbors sparse_distance_matrix` | 같은 커널의 전수와 같음, 가중 셈의 결정성, scipy와 집합이 같음, 속도 사례 |
| SPT5 BallTree (건너뜀) | `BallTree`와 그 거리들 | 열지 않는다: SPT8이 첫 판에서 뺐다(나중 후보). SPT6에서 KD 트리가 칠 수 없는 거리의 전수 시간을 재 보고 다시 본다 |
| SPT6 성능과 일관성 점검 | 모든 공개 함수를 크기마다 scipy·sklearn·R과 견주어 고치기, 길의 문턱(n₀, 트리 대 전수, `leafsize`, 블록 크기) 정하기, 메시지·시그니처의 영역 간 일관성, 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다 |
| SPT7 문서 | `docs/api/spatial.md`의 확정, 이 문서를 구현대로 고치기(잰 값, 정한 상수) | 문서가 코드와 맞고 `test_api_surface`가 통과한다 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. SPT1이 짝 커널을 만들므로 맨 앞이다. SPT2와 SPT3은 SPT1에만 기대어 순서를 바꿀 수 있다. SPT4는 SPT3 뒤다. 기존 영역의 결과는 비트까지 같아야 한다(이 영역은 다른 영역을 고치지 않는다; linalg의 잎을 쓰기만 한다).

**나중 후보** (첫 판 밖): `BallTree`(§9의 3번), 주기 경계 `boxsize`(§9의 15번), 근사 질의 `eps`(§9의 6번), 불 거리 다섯과 `jensenshannon`·`directed_hausdorff`, 호출 가능한 `metric`, GEMM 고르기를 다른 거리로 넓히기(§3.2), `kneighbors_graph`, Qhull 기하, 회전과 변환, 근사 이웃, 그래프 도우미(`sd.graph`).

---

## 9. 마일스톤을 시작하기 전에 물을 것
추천을 첫째에 둔다. 모두 (a)로 정했다(§1.1 SPT6–SPT25, 추천안, 사용자 위임 2026-11-22).

**SPT0 전**
1. **`sd.spatial.distance`를 둘까** (SPT5). (a) 두지 않는다: 거리 함수도 `sd.spatial`에 바로 둔다. 추천. sparse가 `sd.sparse.linalg`를 두지 않은 것과 같고, 별칭 없이 한 이름이다. 겹치는 이름이 없다(§1.2). (b) `sd.spatial.distance`에만 둔다: scipy의 가져오기 경로와 같지만 `sd.spatial`의 다른 이름과 층이 달라진다. (c) 둘 다: 별칭이라 안 된다.
2. **편의 함수와 축약의 이름**. (a) `kneighbors(XA, XB, /, *, k=1, …)`와 `radius_neighbors(XA, XB, r, /, …)` 둘만 둔다. sklearn `NearestNeighbors`의 메서드 이름이고, `pairwise_distances_argmin_min`은 `kneighbors(k=1)`이다. 추천: 같은 일에 이름이 하나다. (b) sklearn처럼 `pairwise_distances_argmin_min`·`pairwise_distances_argmin`도 둔다: 같은 결과에 두 이름이다. (c) scipy 쪽 말로 `cdist_argmin`·`cdist_min`을 둔다: `cdist`와 같은 말을 쓰지만 k > 1과 반경을 담지 못해 결국 다른 이름이 또 든다.
3. **`BallTree`를 넣을까**. (a) 첫 판에 넣지 않는다(나중 후보). 추천. KD 트리가 Minkowski 계열을 맡고, 그 밖의 거리는 전수(+ GEMM 고르기)로 `kneighbors`가 맡는다. Ball 트리가 이기는 곳은 KD 트리가 칠 수 없는 거리(`haversine`, `canberra` 등)의 중간 차원이다. sklearn의 Ball 트리는 거리마다 가상 호출이고, m이 크면 어느 트리든 전수보다 느리다(§5의 m = 16). SPT6에서 그 거리들의 전수 시간을 재 보고 Ball 트리가 2배 넘게 이길 크기가 흔하면 그때 SPT5를 연다. (b) 넣는다: 거리마다 공 경계(삼각 부등식)가 필요하므로 거리 함수의 일부(`cosine`, `correlation`, p < 1)는 Ball 트리에서도 칠 수 없다.
4. **`haversine`을 넣을까**. (a) 거리 함수 `haversine(u, v)`와 `metric='haversine'`(`pdist`, `cdist`, `kneighbors`, `radius_neighbors`)로 넣는다. sklearn과 같은 뜻(위도·경도 라디안 2열, 단위 구의 라디안)이고 식은 Arb로 재서 모든 범위에서 계약을 지키는 것을 고른다(§3.1). 트리는 쓰지 않는다(전수). 추천: 지리 자료의 kNN이 흔하고, sklearn의 식은 대척점 근처에서 정확도를 잃는다. (b) 넣지 않는다: scipy에 없다.
5. **`workers=`**. (a) 두지 않는다: 스레드 수는 `sd.set_num_threads` 하나다(ARCHITECTURE §4.1, 하위 모듈에 스레드 API를 두지 않는다). 기본으로 병렬이고 결과는 스레드 수와 무관하다. 추천. `workers=`를 주면 모르는 키워드의 `TypeError`다(`[scipy와 다름]`: scipy의 기본은 1스레드). (b) 받되 무시한다: 틀린 기대를 남긴다.
6. **`eps`(근사 질의)**. (a) 두지 않는다: 결과는 언제나 정확하다. 근사 이웃은 SPT1이 뺐다. 추천. 근사 질의의 결과는 순회 순서에 기대므로 길과 무관한 결과(§1.3)를 지키려면 길마다 순서를 고정해야 하고, 그만큼 빠르게 할 자유를 잃는다. (b) scipy처럼 둔다(음수는 `ValueError`): 첫 판의 일이 늘고 결과의 약속이 둘로 갈린다.
7. **유한하지 않은 값**. (a) 거리 함수는 IEEE로 퍼뜨리고(`chebyshev`도 NaN을 퍼뜨린다, `canberra`도), 이웃 찾기(`KDTree`, `kneighbors`, `radius_neighbors`)는 데이터와 질의점의 NaN·inf를 `ValueError`로 막는다. 추천: scipy의 `KDTree`와 sklearn이 막는 것과 같고, `vector_norm(ord=inf)`가 NaN을 퍼뜨리는 것과 같다. (b) `kneighbors`는 `cdist` 행의 `argmin`처럼 NaN을 다룬다(첫 NaN이 가장 가까움): 정의는 단순하지만 `KDTree.query`와 결과가 갈린다.
8. **받는 입력**. (a) API.md §5의 상위 함수 입력(`sd.ndarray`, 시퀀스, numpy, frame의 `DataFrame`)이다. 추천: 사용자가 밖의 자료를 처음 넣는 자리이고 `sd.stats`·`sd.htest`와 같다. (b) 배열 함수처럼 `sd.ndarray`만: 부품으로 자주 불리는 함수가 아니다.
9. **호출 가능한 `metric`과 `out=`**. (a) 둘 다 첫 판에 두지 않는다. 추천: 호출 가능한 거리는 짝마다 Python 호출이라 GIL을 쥔 채 돌고(scipy도 Python 고리다), 이 영역의 길(트리, GEMM)이 그 거리를 다룰 수 없다. `out=`은 다른 영역에 없는 꼴이다. (b) scipy처럼 둔다.
10. **`metric`과 `k`의 자리**. (a) 키워드만(`pdist(X, metric='cityblock')`, `tree.query(x, k=3)`). 추천: 옵션은 키워드만이라는 공통 원칙이고, sparse의 `eigsh(A, /, *, k=6)`와 같다. scipy 코드의 `pdist(X, 'cityblock')`, `tree.query(x, 3)`는 `TypeError`다(`[scipy와 다름]`). (b) scipy처럼 위치로도 받는다: 다른 영역에 없는 꼴이 된다.

**SPT1 전**
11. **`squareform`의 검사**. (a) scipy와 같다: `checks=True`면 정확한 대칭과 0 대각, `force`, dtype을 지킨다, 빈 벡터는 (1, 1). NaN이 있으면 대칭 검사에서 걸린다(scipy의 메시지가 "symmetric"이라 헷갈리므로 sd는 NaN이 있다고 말한다). 추천. (b) NaN을 대칭 검사에서 같은 값으로 본다: 거리 행렬에 NaN이 있는 경우(결측 자료)를 받지만 scipy와 다르다.
12. **결과 객체의 이름과 꼴**. (a) `QueryResult(distances, indices)` namedtuple(`stardust._results`)을 `query`·`kneighbors`가, `RadiusResult`(`indptr`, `indices`, `distances`, `shape`, `tocsr()`; 내보내지 않음)를 반경 질의 셋이 함께 쓴다. 반경 결과의 행 안은 색인 오름차순이고 `return_sorted`는 없다. `query_pairs`는 (k, 2)의 정렬된 배열, `sparse_distance_matrix`는 `csr_array`다. 추천: 같은 것은 같은 이름이고, 색인 순서는 CSR의 표준 꼴과 scipy의 정렬 순서와 같다. (b) 반경 결과의 행 안을 (거리, 색인) 순서로: 거리 순서가 필요한 사용자에게 편하지만 `tocsr()`가 다시 정렬해야 한다. (c) 함수마다 다른 이름(`QueryBallPointResult` …): 이름의 규칙(`<함수>Result`)에는 맞지만 같은 결과가 세 이름이다.
13. **`KDTree`의 옵션**. (a) `leafsize`만 둔다. `balanced_tree`, `compact_nodes`는 결과를 바꾸지 않는 빠르기 손잡이라 언제나 켠 꼴(중앙값, 줄인 상자)이고, `copy_data`는 언제나 복사라 뜻이 없다. 기본 `leafsize`는 scipy의 10에서 시작하고 SPT6에서 재서 1.2배 넘게 빠른 값이 있으면 그것으로 바꾼다(`[scipy와 다름]`으로 적는다; 결과는 같다). 추천. (b) scipy의 옵션을 모두 받는다: 뜻 없는 인자가 남는다.
14. **`distance_upper_bound`**. (a) 둔다. 엄격한 부등호(d̂ < 상한)는 scipy와 같고, 상한은 0 이상이거나 inf, NaN·음수는 `ValueError`다. 추천: 가지치기에 쓰여 빠르고 `kneighbors`에도 둔다. (b) 두지 않는다: `query` 뒤에 거르면 되지만 느리다.
15. **`boxsize`(주기 경계)**. (a) 첫 판에 두지 않는다(나중 후보). 추천: 모든 거리 커널과 상자 한계에 감기(min(\|z\|, L − \|z\|))가 들어가고, 그러면 `pdist`에 없는 거리를 트리만 갖게 되어 같은 짝 같은 비트(SPT2)의 범위가 갈린다. 쓰는 곳(우주론·분자 모의)이 좁다. (b) `KDTree`와 거리 함수 모두에 둔다: 일이 크다. (c) `KDTree`에만 둔다: scipy와 같지만 위의 갈림이 생긴다.

**SPT2·SPT3 전**
16. **`query`가 전수로 가도 될까**. (a) 된다: 결과가 길과 무관하므로 모양으로 길을 고른다(§3.4). m이 큰 데이터에서 수십 배 빠르다. 추천. (b) `query`는 언제나 트리를 쓰고 전수는 `kneighbors`만: 사용자가 길을 고르지만, 같은 결과에 느린 길을 남긴다.
17. **짝 커널의 순서: `vector_norm`과 같게 할까**. (a) 같게 한다: `euclidean(x, y)`가 `sd.linalg.vector_norm(x − y)`와, `sqeuclidean`이 `vecdot(z, z)`와, `cityblock`·`chebyshev`·`minkowski`가 `ord=1, inf, p`와 비트까지 같다(§3.1). 다만 SPT0에서 재서 이 순서의 묶음 커널이 타일 순서보다 1.3배 넘게 느리면 (b)로 간다. 추천: 영역 사이의 일관성이고, 비용은 재서 정한다. **SPT0의 결과: 1.34–3.85배 느려 (b)로 갔다(§3.1, SPT26).** (b) 짝마다 타일에 맞는 이 영역만의 순서: 같은 짝은 영역 안에서 여전히 같은 비트지만 linalg와는 끝자리가 다르다. n > 2048에서 짝마다 BLAS를 부르는 대신 이 라이브러리의 pairwise 트리를 써서 Accelerate 모드와도 무관해진다.
18. **`hamming`·`jaccard`의 입력과 int64**. (a) bool, int64, float64를 받아 입력의 dtype 그대로 비교한다(bool은 비트 묶음의 빠른 길). 다른 거리는 int64·bool을 float64로 바꾼다. 추천: 같음 비교는 정확해야 하고(scipy는 float64로 바꾼 뒤 비교해 2⁶⁰과 2⁶⁰ + 1을 같은 값으로 본다), 바꾸는 비용도 없다. 2⁵³을 넘는 정수에서만 scipy와 다르다(`[scipy와 다름]`). (b) scipy처럼 모두 float64로 바꾼다.
19. **`w`의 검사**. (a) 유한하고 0 이상(NaN·inf는 `ValueError`). 추천: NaN 가중치가 모든 거리를 NaN으로 만드는 것은 입력의 실수다. (b) scipy처럼 음수만 막는다.

**SPT4 전**
20. **`count_neighbors`의 가중 셈 순서**. (a) 고정된 깊이의 노드 쌍 태스크와 태스크 번호 순서의 합(§3.4): 결과는 데이터와 트리로만 정해지고 스레드 수와 무관하다. 추천. (b) 정확한 합(정수로 바꿀 수 없는 실수 가중치라 이중 배정도 합): 순서와 무관하지만 느리다.
