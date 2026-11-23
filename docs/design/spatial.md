# 공간 영역 설계 (sd.spatial)

> 상태: **확정** (2026-11-23). 큰 결정(§1.1 SPT1–SPT5)과 `haversine` 이웃의 트리 길(SPT83)은 사용자가 정했고, 나머지 결정(SPT6–SPT82, SPT84–SPT89)은 추천안이다(사용자 위임). SPT0–SPT4, SPT6, SPT7은 완료·승인(위임)이고 SPT5(BallTree)는 건너뛰었다(SPT8, SPT83). 모든 공개 이름(`docs/api/spatial.md`)이 구현되어 있고 이 문서는 구현과 맞다.

`sd.spatial`(점 사이의 거리와 가장 가까운 이웃 찾기)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처는 [docs/api/spatial.md](../api/spatial.md)에 있다.
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
| SPT33 신중한 경로의 문턱과 행의 비율 | 제곱합(과 `minkowski`의 합, `mahalanobis`의 이차식)이 inf이거나 2⁻⁹⁶⁸보다 작으면 신중한 경로다(초안의 n·2⁻¹⁰⁰⁰ 대신 상수; 언더플로한 항의 몫 n·2⁻¹⁰⁷⁵이 상대 n·2⁻¹⁰⁷ 이하라 n < 2⁴⁰이면 2⁻⁶⁷ 아래다). `cosine`·`correlation`의 행은 정규화(와 평균 빼기) 전에 가장 큰 원소의 지수로 고른 2ᵏ로 비율을 맞춘다(정확; 정상 범위의 행은 비트가 같다). 까닭: 비정상 수 행의 노름과 평균은 자릿수를 잃는다(`edge/subnormal`의 가중 `cosine`이 계약의 2×10¹⁴배, `correlation`의 평균 5×10⁻³²⁴/2가 0으로 반올림되었다) (추천안, 사용자 위임 2026-11-22) |
| SPT34 기본 `VI`와 유한하지 않은 자료 | 자료에 NaN·inf가 있어 공분산이 유한하지 않으면 기본 `VI`는 NaN 행렬이고 모든 거리가 NaN이다(scipy와 같은 결과; `inv()`의 `ValueError`를 내지 않는다). 공분산이 특이하면 `LinAlgError`(`pdist(): the covariance of the data is singular, so the default 'VI' does not exist; give 'VI'`)다. 까닭: 거리 함수는 NaN을 퍼뜨린다(SPT12) (추천안, 사용자 위임 2026-11-22) |
| SPT35 `hamming`·`jaccard`의 두 dtype | `u`와 `v`(`XA`와 `XB`)의 dtype이 다르면 하나라도 float64면 float64, 아니면 int64(bool과 int64)로 맞춘 뒤 비교한다. 가중 bool 행은 바이트 비교의 한 짝 커널이다(비트 묶음은 가중치 없는 bool만). 까닭: 배열의 형 승격과 같은 규칙이고, 개수가 정확해 어느 길이든 같은 값이다 (추천안, 사용자 위임 2026-11-22) |
| SPT36 SPT0 시제품의 자리 | `csrc/spatial/`의 `pair_kernel.hpp`(두 순서의 시제품)는 지우지 않고 `tools/spatial_probe_pair_kernel.hpp`로 옮긴다. 라이브러리의 커널은 `metric.hpp`다. 까닭: `tools/spatial_probe.py kernels`가 SPT22의 측정을 다시 할 수 있어야 하고, 라이브러리 밖의 코드는 `csrc/`에 두지 않는다 (추천안, 사용자 위임 2026-11-22) |
| SPT37 상한과 반경의 판정 | `kneighbors`는 거리 d̂ < `distance_upper_bound`인 짝만 이웃으로 친다(기본 inf에서도). 그래서 거리가 inf이거나 NaN인 짝(넘치는 `sqeuclidean`, 영벡터의 `cosine`, 0/0의 `braycurtis`)은 나오지 않고 그 자리는 (inf, mB)다. `radius_neighbors`는 d̂ ≤ r이라 r = inf면 inf인 짝을 넣고 NaN인 짝은 넣지 않는다. 까닭: `KDTree.query`가 상한을 처음 T로 두고 엄격한 비교로 넣는 것과 한 규칙이 되고(같은 결과의 약속), NaN에는 순서가 없다. scipy의 트리도 제곱합이 넘친 점을 상한 inf에서 빼므로 반올림이 아닌 동작이 같다 (추천안, 사용자 위임 2026-11-22) |
| SPT38 전수 길의 계산 | 전수 길은 `cdist`의 구동기(타일, 짝 하나씩의 구동기, 데이터마다의 준비)를 블록마다 그대로 부른다(`csrc/spatial/distance.cpp`의 `CdistBlocks`, 선언은 `neighbors.hpp`). 따로 전수 커널을 두지 않는다. 까닭: 거리가 만들어질 때부터 `cdist`와 같은 비트이고 마무리 함수의 두 번째 사본이 없다. 블록마다 `XB`의 판을 다시 싸는 비용은 `XA` 64행의 블록에서 1/64이고, `cosine`의 정규화는 블록마다 다시 해 약 10%다 (추천안, 사용자 위임 2026-11-22) |
| SPT39 L̃의 계산 | §3.2의 꼴을 그대로 쓰고(상수도 그대로) NEON 두 레인으로 행 하나의 블록 전체에 한 번에 구한다. 레인과 스칼라가 같은 IEEE 연산(덧셈, 곱셈, 뺄셈, √, 비교)이라 같은 값이다. 버리지 못하는 짝(t > 0이나 v ≥ 2⁻⁵⁰⁰가 아님, NaN 포함)은 L̃ = −inf로 적는다. 까닭: 짝마다의 거르기가 GEMM 길의 시간을 정하고(n = 3에서 1스레드 짝당 약 1.4 ns), 정리 3이 그 꼴에 대해 증명되었다 (추천안, 사용자 위임 2026-11-22) |
| SPT40 GEMM 고르기를 쓰는 곳(n₀) | 가중치 없는 `euclidean`·`sqeuclidean`(과 `minkowski` p = 2)에서 12 ≤ n ≤ 2¹⁶이고 mB ≥ 64일 때다. 그 밖은 직접 전수다. 까닭: 4000² k = 1에서 10스레드는 n = 12에서 같고(3.29 대 3.37 ms) n = 16에서 GEMM이 11%, n = 64에서 2.3배, n = 512에서 3.4배 빠르다. 1스레드에서는 n ≈ 8에서 갈린다. SPT28은 2²¹ − 2까지 허락하지만 n > 2¹⁶의 행은 태스크마다의 메모리만 늘고 이득이 없다 (추천안, 사용자 위임 2026-11-22) |
| SPT41 첫 후보의 규칙 | 블록을 시작할 때 힙이 차 있지 않으면 모자라는 수만큼 L̃가 가장 작은 짝(−inf가 먼저, 같으면 열 번호)을 먼저 다시 계산해 T를 정한다. §3.2의 "Ê가 가장 작은 k개" 대신 이미 구한 L̃로 고르고(â가 같으면 L̃는 Ê에 단조), 블록을 한 번 훑는 크기 k의 작은 힙으로 고른다. 고른 짝은 L̃ 자리에 NaN을 적어 훑기에서 건너뛴다. 까닭: 결과와 무관하고, 처음 판의 `nth_element`가 k = 1에서 4배 느렸다(4000² n = 3 1스레드 73 대 18 ms) (추천안, 사용자 위임 2026-11-22) |
| SPT42 나누기 | `XA`를 행 조각(전수 64행; GEMM은 256행까지이고 `XA`가 허락하면 조각이 32개 이상이 되게 줄이고 2¹⁹/n행 이하)으로 나누고, 조각이 32개보다 적으면 `XB`를 4096행 이상의 구간으로도 나눈다. (조각, 구간)마다 한 태스크의 힙이고, 구간들의 정렬된 목록을 (d̂, j)로 합친다. 태스크의 힙은 2¹⁶자리 이하다(k가 크면 조각의 행을 줄인다). 모두 크기로만 정한다. 까닭: (d̂, j)가 전순서라 어떻게 나눠도 답이 같고, 질의가 적은 호출(질의 20개 × 10⁶점)도 스레드를 다 쓴다 (추천안, 사용자 위임 2026-11-22) |
| SPT43 중심 빼기 | GEMM 길은 언제나 중심을 뺀다. c는 `XB`의 열 평균이고 4096행씩의 부분합을 그 순서로 더한다. 까닭: c가 무엇이든 증명이 서므로 결정적이기만 하면 되고, 원점 근처의 자료에서도 손해가 없다(1e8 + [0, 1)¹⁶ 4000²에서 원점 근처와 같은 시간, 2.97 ms) (추천안, 사용자 위임 2026-11-22) |
| SPT44 `RadiusResult`의 자리와 생성자 | `src/stardust/spatial/_radius.py`(§6 초안의 `_result.py` 대신)이고 생성자는 `RadiusResult(indptr, indices, distances, shape)`다. 받은 배열을 검사하지도 복사하지도 않고 쓰기 금지로 만든다(부르는 쪽, 네이티브 찾기가 꼴을 보장한다). 피클하지 않는다(`sd.ndarray`가 피클되지 않는다). 까닭: SPT4의 `query_ball_point`·`query_ball_tree`가 같은 생성자로 만든다 (추천안, 사용자 위임 2026-11-22) |
| SPT45 큰 k와 순위 | 네이티브 찾기는 min(max(k), mB)개만 찾고 Python 층이 넘는 순위를 (inf, mB)로 채운다. k가 시퀀스면 그 순위의 열을 고른다. 요청한 결과(mA × 순위 수 × 16바이트)를 할당 전에 물리 메모리와 견준다. 까닭: k = 10⁹ 같은 호출이 태스크마다 k자리의 힙을 잡지 않는다 (추천안, 사용자 위임 2026-11-22) |
| SPT46 `r`의 꼴 | 실수 하나, 또는 (mA,)·(1,)·0차원 배열이다. 2차원 이상과 다른 길이는 `ValueError`다. 까닭: `query_ball_point`의 "x의 배치 모양으로 브로드캐스트"를 `XA`의 행 축 하나에 쓴 것이다 (추천안, 사용자 위임 2026-11-22) |
| SPT47 공통 검사 | 데이터의 유한성 `finite()`와 반경 `radius()`를 `_common.py`에 두고, `k`와 상한은 SPT3이 둔 `k_ranks()`·`upper_bound()`를 쓴다. 메시지: `kneighbors(): 'XA' must be finite`, `radius_neighbors(): 'r' must be a number >= 0 or inf, not -1`. 까닭: `KDTree`의 메서드와 같은 말과 같은 함수다 (추천안, 사용자 위임 2026-11-22) |
| SPT48 반경 결과 모으기 | 태스크마다 행별 목록을 모은 뒤, 개수의 누적합으로 `indptr`를 만들고 태스크마다 제자리에 복사한다. 결과 크기의 두 배가 잠깐 든다. 까닭: 한 번 훑기로 끝나고(개수를 먼저 세는 두 번 훑기는 거리를 두 번 계산한다), 복사는 찾기에 비해 작다 (추천안, 사용자 위임 2026-11-22) |
| SPT49 `XA`의 차원 | `kneighbors`·`radius_neighbors`의 `XA`는 `cdist`처럼 2차원만 받는다(점 하나도 (1, n)). 1차원은 `ValueError`다. 까닭: 편의 함수는 `cdist`의 행 고르기이고, 점 하나의 스칼라 결과는 `KDTree.query`의 꼴이다 (추천안, 사용자 위임 2026-11-22) |
| SPT50 트리 만들기의 꼴 | 점이 `leafsize`보다 많은 노드는 상자의 폭이 0이어도(같은 점뿐이어도) 나눈다. 그래서 노드 수와 앞순회 번호가 n과 `leafsize`로만 정해진다(노드 수 c ↦ 1 + N(⌊c/2⌋) + N(⌈c/2⌉)). 잎 안의 점은 원래 색인 순서로 둔다. `size`는 같은 점이 많은 데이터에서 scipy와 다르다(scipy는 폭이 0인 노드를 잎으로 둔다: 같은 점 50개에서 scipy 1, sd 15). 까닭: 병렬 만들기가 노드 자리를 미리 셀 수 있고, 트리가 `nth_element`의 구현과도 무관한 데이터의 함수가 된다 (추천안, 사용자 위임 2026-11-22) |
| SPT51 가지치기의 한계 | 노드의 한계는 노드 상자에서 x에 가장 가까운 점 c(cᵢ = min(max(xᵢ, loᵢ), hiᵢ), 정확)까지의 짝 커널 거리 b다. 노드마다 새로 구하고(증분 갱신이 아님), b > fl(fl(T·c) + η)일 때만 친다(c = 1 + 4(m + 12)u, η = 2⁻¹⁰⁷⁰; p = inf는 b > T). 증명은 거리 계약(§4)만 쓴다(§3.4). 까닭: scipy식 증분 갱신은 반올림이 쌓이는 경로를 따로 증명해야 하고, 한계 하나가 점 하나의 거리 값(O(m))이라 잎의 점 수(≈ `leafsize`)에 비해 작다 (추천안, 사용자 위임 2026-11-22) |
| SPT52 p = 2의 제곱합 비교 | 잎의 점과 노드 한계는 먼저 제곱합 r로 견준다. r이 보통 범위 [2⁻⁹⁶⁸, DBL_MAX]이고 r > fl(fl(T·T)·(1 + 8u))면 √ 없이 버린다(그때 d̂ > T가 보장된다, §3.4). 그 밖은 d̂를 구해 그대로 견준다. 까닭: 결과는 그대로이고 1스레드 `query`(10⁶점, m = 3, 질의 10⁴개)가 k = 1에서 17.7 → 8.6 ms, k = 10에서 39.5 → 22.4 ms가 되었다(scipy 9.4, 28.9 ms) (추천안, 사용자 위임 2026-11-22) |
| SPT53 병렬 만들기 | n·max(m, 1) ≥ 2¹⁵이면 점이 max(2¹², n/64)개 이하인 하위 트리를 태스크로 만들고(크기로만 정함), 그 위의 노드는 호출 스레드가 나눈다(큰 노드의 상자, 정렬 키 채우기, 행 옮기기는 2¹⁴점 조각으로 풀에서). 나누기는 행 자체를 옮겨 노드의 점이 이어진 행이 되게 하고, 키는 (좌표, 행 자리) 16바이트에 동점일 때만 원래 색인을 읽는다. 까닭: 모든 훑기가 이어 읽기가 되어 10⁶점 m = 3의 1스레드 만들기가 335 → 225 ms(scipy 230–260 ms), 10스레드 79 ms다. 상자의 최솟값·최댓값은 정확한 연산이라 조각으로 나눠도 같은 트리다 (추천안, 사용자 위임 2026-11-22) |
| SPT54 질의의 병렬 | 질의점 64개가 태스크 하나다(크기로만 정함). 질의점 하나는 한 태스크가 처음부터 끝까지 찾는다. 작업량 질의 수 × max(m, 1) × 64가 2¹⁵ 이상이면 나누고 스레드 수와 상관없이 GIL을 놓는다. 까닭: 결과가 스레드 수와 무관하고, 질의 10⁴개가 태스크 157개라 10스레드에 고르게 나뉜다 (추천안, 사용자 위임 2026-11-22) |
| SPT55 `k`의 꼴 | 1 이상의 int, 또는 1 이상의 int 시퀀스(list, tuple, 1차원 정수 배열; 순서와 중복은 자유, scipy의 뜻: 돌려줄 순위의 목록)다. 힙은 min(max k, n)개만 둔다. int64를 넘으면 `OverflowError`다. 까닭: scipy의 시퀀스 뜻을 그대로 받고(`k=[3, 1]`, `k=[2, 2]`도 scipy가 받는다), 큰 k가 n개 넘게 메모리를 쓰지 않는다 (추천안, 사용자 위임 2026-11-22) |
| SPT56 피클 | `data`의 바이트와 n, m, `leafsize`로 다시 만든다(`__reduce__`). 까닭: `sd.ndarray`가 피클되지 않는다(`cannot pickle 'ndarray' object`; 희소 배열도 같다). 만들기가 결정적이라 다시 만든 트리는 노드·상자·색인이 바이트까지 같다(테스트가 확인한다) (추천안, 사용자 위임 2026-11-22) |
| SPT57 `data`의 사본 | 입력 순서의 읽기 전용 사본은 Python 객체가 쥐고, 트리는 잎 순서의 사본을 따로 쥔다(메모리 2·n·m·8바이트; 10⁶ × 3에서 24 MB 더). 까닭: `data`를 읽을 때 복사가 없고 처음 읽을 때 만드는 것보다 단순하다. 메모리가 문제가 되면 SPT6에서 다시 본다(§3.3) (추천안, 사용자 위임 2026-11-22) |
| SPT58 인자 검사의 공용 자리 | `k`, 트리의 `p`, `distance_upper_bound`의 검사와 메시지는 `_common.py`의 `k_ranks`, `tree_p`, `upper_bound`, `real` 하나다(`query(): 'k' must be at least 1, not 0`, `every entry of 'k' must be at least 1, not 0`, `'k' must not be an empty sequence`, `'k' must be an int or a sequence of ints, not bool`, `'p' must be a number of at least 1 or inf, not 0.5`, `'distance_upper_bound' must be a number >= 0 or inf, not -1.0`). 까닭: `kneighbors`가 같은 규칙(§2.3)을 같은 함수로 쓸 수 있다 (추천안, 사용자 위임 2026-11-22) |
| SPT59 전수 길의 자리 | `query`가 길을 고르는 곳은 `_kdtree._search` 하나이고 SPT3에서는 언제나 트리다. SPT21의 전수 길(문턱)은 SPT6에서 재서 그 자리에 더한다. 까닭: 결과가 길과 무관하므로(§1.3) 고르기는 빠르기만 바꾸고, 문턱은 `kneighbors`의 GEMM 고르기(SPT2)와 함께 재야 한다 (추천안, 사용자 위임 2026-11-22) |
| SPT60 `repr` | `KDTree(n=1000, m=3, leafsize=10)` 한 줄이다(API.md §2.3의 꼴). 까닭: 트리의 속은 공개하지 않으므로 만든 인자와 크기만 보인다 (추천안, 사용자 위임 2026-11-22) |
| SPT61 빈 트리 | n = 0이면 빈 잎 하나(`size` 1)이고 `mins`·`maxes`는 0, 질의는 모두 (inf, 0)이다(scipy와 같다). m = 0이면 모든 거리가 0이라 결과는 색인 순서다(scipy는 만들기에서 `IndexError`). 까닭: scipy와 같은 결과이고, 없는 축은 빈 합의 정의(§2.1)를 따른다 (추천안, 사용자 위임 2026-11-22) |
| SPT62 질의점의 검사 | x는 1차원 이상의 (…, m)이고(스칼라는 상위 함수 입력의 `TypeError`), 마지막 축이 m이 아니면 `query(): 'x' must have 3 entries on its last axis (the tree's m), not 2`, NaN·inf는 `query(): 'x' must be finite`다(SPT12). 유한성은 네이티브 층이 질의 전에 한 번 훑는다. 까닭: 메시지가 데이터의 `KDTree(): 'data' must be finite`와 같은 꼴이고, 훑기가 질의보다 훨씬 싸다 (추천안, 사용자 위임 2026-11-22) |
| SPT63 두 상자 사이의 한계 | 노드 쌍의 가까운 한계는 각 축의 틈을 사이에 둔 두 모서리 a, b(겹치면 0과 0)의 짝 커널 거리이고 b̂ > KdCut(r)이면 친다(SPT51의 증명이 D(x, c) 대신 D(a, b)로 그대로 선다). 먼 한계는 축마다 fl(hiᴮ − loᴬ)와 fl(hiᴬ − loᴮ) 중 큰 쪽의 두 모서리이고, KdCut(b̂) ≤ r이면 노드 쌍의 모든 짝이 r 안이다(§3.4의 증명: 반올림한 차이가 같아 고른 쪽이 작을 때도 참 차이가 (1 − 2u)배 안이고, c(1 − u)² ≥ 1 + (4m + 46)u가 (1 + κ)/((1 − κ)(1 − 2u)) ≤ 1 + (2m + 24)u를 넘는다; p = inf는 b̂ ≤ r 그대로). 질의점은 상자 (x, x)다(가까운 한계가 SPT51의 고정점과 같다). 까닭: SPT3의 여유 c, η를 그대로 쓰고 한계가 짝 하나의 거리라 따로 증명할 반올림 경로가 없다. 개수(`return_length`, `count_neighbors`)만 먼 한계를 쓴다: 목록은 어차피 모든 거리를 계산하고, 판정은 언제나 d̂ ≤ r이다 (추천안, 사용자 위임 2026-11-22) |
| SPT64 이중 트리의 태스크 | 두 트리를 깊이 4의 노드(더 얕은 잎 포함, 트리마다 16개 이하)로 자르고 조각의 짝 (a, b)마다 태스크 하나, 번호 a·(B의 조각 수) + b다. `query_pairs`는 a ≤ b만이고 (a, a)는 노드와 자기 자신(두 반쪽 각각과 그 둘의 짝)이다. 노드 쌍에서는 점이 많은 쪽(같으면 A)을 나누고 잎끼리는 모든 짝을 견준다. 자르기는 트리로만 정해지고 병렬 기준값 아래에서도 같은 자르기를 직렬로 돈다. `query_ball_point`는 질의점 64개가 태스크 하나다(SPT54와 같다). 까닭: 결과가 집합(목록은 행마다 정렬)이라 자르기와 무관하고, 가중 셈은 태스크 번호 순서로 더하므로(SPT25) 스레드 수와 무관하다. 깊이 4는 256태스크라 10스레드에 고르게 나뉜다 (추천안, 사용자 위임 2026-11-22) |
| SPT65 `count_neighbors`의 셈 | 반경들을 줄지 않게 정렬하고(같은 값은 원래 순서, 안정 정렬) 태스크마다 delta[k] = "d̂ ≤ r_k인 첫 k가 k인 짝의 수(가중이면 무게 곱의 합)"를 센다(k = R은 버림). 노드 쌍은 가까운 한계로 아래 반경을, 먼 한계로 위 반경을 정하고 둘이 만나면 그 칸에 노드 무게의 곱을 한 번에 더한다(Gray–Moore). 잎의 짝은 d̂로 이분 탐색한다(p = 2는 제곱합으로 먼저: s > above(r_hi)면 hi, s ≤ below(r_lo)면 lo). `cumulative=False`는 delta 그대로이고 `True`는 정렬 순서의 누적합을 원래 자리에 쓴다. 노드 무게는 잎이 잎 순서의 차례 합, 노드가 왼쪽 + 오른쪽이다. 반경이 많으면 깊이를 줄여 태스크 × (R + 1)을 2²²칸 이하로 둔다. 까닭: 구간 셈이 뺄셈 없이 정확하고(가중이면 반올림이 한 번 적다) 반경 수와 무관한 순회 한 번이다 (추천안, 사용자 위임 2026-11-22) |
| SPT66 `count_neighbors`의 인자 | `r`은 실수 하나(Python·numpy 스칼라, numpy 0차원 배열; 결과는 Python `int`, 가중이면 `float`)나 1차원 배열(빈 배열도; 결과는 int64·float64 배열)이다. 2차원 이상은 `ValueError`(`'r' must be a number or a 1-D array, not a 2-D array`), 음수·NaN은 `ValueError`(scipy는 NaN에서 모든 짝을 세고 음수에서도 짝을 센다), `cumulative=False`면 줄지 않아야 한다(`'r' must be nondecreasing when cumulative=False`). `weights`는 scipy의 꼴이고 `(None, None)`은 가중치 없음(정수 결과, scipy와 같다), 배열 하나는 `other`가 `self`일 때만이다. 가중치는 유한해야 하고(`'weights' must be finite`; scipy는 NaN 셈을 낸다) 음수는 받는다. 까닭: 반경의 규칙이 `query_ball_point`와 같고, 무게는 곱의 계수일 뿐 거리가 아니라 부호를 막을 까닭이 없다(SPT24의 `w`와 다르다) (추천안, 사용자 위임 2026-11-22) |
| SPT67 `query_ball_point`의 `r` | 실수 하나나, x의 배치 모양으로 그 모양을 바꾸지 않고 브로드캐스트되는 배열이다(numpy의 규칙; 점 하나 (m,)에 (1,)은 받지 않는다, scipy도 같다). 그 밖은 `ValueError`(`'r' of shape (2,) does not broadcast to the batch shape (3,) of 'x'`). `query_ball_tree`·`query_pairs`의 `r`과 `sparse_distance_matrix`의 `max_distance`는 실수 하나다(scipy와 같다). 까닭: scipy가 받는 꼴 그대로이고, 결과의 행이 x의 배치를 C 순서로 펼친 것이라 r이 배치를 늘리면 행의 뜻이 없다 (추천안, 사용자 위임 2026-11-22) |
| SPT68 큰 결과 | 목록을 모으는 질의(`query_ball_point`, `query_ball_tree`, `sparse_distance_matrix`, `query_pairs`)는 찾은 짝의 수를 원자적으로 세어 물리 메모리 ÷ 짝 하나의 바이트(모으는 동안과 결과: 40, 64, 32)를 넘으면 찾기를 멈추고 `MemoryError`(`query_pairs(): the result is larger than this machine's physical memory`)다. 까닭: 결과의 크기를 미리 알 수 없고(먼저 세면 순회가 두 번이다), r = inf의 `query_pairs`(10⁵점에서 5×10⁹짝)가 기계를 멈추게 하지 않는다 (추천안, 사용자 위임 2026-11-22) |
| SPT69 잎 쌍의 점마다 상자 검사 | 넣지 않는다. 잎 쌍에서 A의 점마다 B 잎의 상자와 먼저 견주는 길을 재 보니 10⁵ × 10⁵ 균등(점마다 이웃 약 10) 1스레드에서 `count_neighbors` 70 → 94 ms, `query_ball_tree` 68 → 86 ms로 느려졌다(잎이 r과 같은 크기라 거르는 짝보다 한계의 비용이 크다). 까닭: 재서 손해다. `leafsize`와 함께 SPT6에서 다시 본다 (추천안, 사용자 위임 2026-11-22) |
| SPT70 SPT4의 파일 | `csrc/spatial/query.cpp`(`_ball_point`)와 `dual.cpp`(`_ball_tree`, `_pairs`, `_count`)이고, 두 상자 사이의 한계(`KdPair`)와 공용 판정(`kd_distance`, `kd_beyond`, 거리 정책의 `below`)은 `kdtree.hpp`에 둔다. Python의 검사는 `_common.py`(`scalar_radius`, `ball_radius`, `tree_other`)와 `_kdtree.py`(`count_neighbors`의 반경과 무게)다. 까닭: §6의 안 그대로이고 한계가 SPT51의 증명 곁에 있다 (추천안, 사용자 위임 2026-11-22) |
| SPT71 GEMM 고르기의 블록 공통 한계 | 블록마다 a_max = fl(Rₓ + 블록의 최대 Rᵧ)로 Δ̃와 e를 한 번 구하고, 짝마다는 t′ = fl(Ê − Δ̃(a_max))만 계산해 t′ > τ면 버린다(τ는 L′(next(τ)) > T를 확인한 값, L′는 a_max로 구한 L̃의 꼴). L̃의 모든 단계가 a에 단조라(반올림이 단조) L′ ≤ L̃이고 t′ > 0, v′ ≥ 2⁻⁵⁰⁰이면 그 짝의 t > 0, v ≥ 2⁻⁵⁰⁰이므로 버린 짝은 정리 3대로 d̂ > T다. 남은 짝만 짝마다의 L̃(정리 3의 꼴 그대로)로 견준다. 옮긴 `XB`(fl(y − c))는 호출마다 한 번 만들어 태스크가 함께 읽는다(2²⁷개 double 이하일 때; 넘으면 전처럼 블록마다). 까닭: 짝마다 √와 열댓 연산이던 거르기가 덧셈·뺄셈·비교 하나가 되고 `XB`를 조각 수만큼 다시 옮기지 않는다: 4000² 1스레드 k = 1에서 n = 64 24.6 → 14.5 ms, n = 512 53.0 → 36.2 ms, `radius_neighbors` n = 64 25.0 → 13.8 ms(답은 같다) (추천안, 사용자 위임 2026-11-23) |
| SPT72 GEMM 길의 후보 순서 | 힙이 차지 않은 블록은 L̃ 대신 Ê가 작은 짝을 모자라는 수만큼 먼저 다시 계산하고(Ê는 L̃에 가깝고 싸다), 그래도 차지 않으면 블록 전체의 L̃로 견준다. 힙이 찬 블록은 SPT71로 거르고 L̃ ≤ T인 후보를 모아 2k개를 넘으면 L̃가 가장 작은 k개(`nth_element`, (L̃, 열))를 먼저 다시 계산해 T를 낮춘 뒤 나머지를 견준다. 까닭: 어느 순서든 힙은 버리지 않은 짝 가운데 (d̂, j)가 가장 작은 k개로 끝나 답이 같고, 먼 점부터 가까운 점 순으로 정렬한 `XB`(블록마다 T가 내려가는 적대적 순서, 20000 × 32 k = 10)에서 GEMM 길이 직접 전수보다 1.3배 느리던 것이 0.26–0.49배가 되었다 (추천안, 사용자 위임 2026-11-23) |
| SPT73 `kneighbors`·`radius_neighbors`의 트리 길 | 가중치 없는 `euclidean`·`cityblock`·`chebyshev`·`minkowski`(p ≥ 1)는 `XB`로 KD 트리(`leafsize` 10)를 만들어 `query`·`query_ball_point`의 네이티브 질의로 답한다: 그 밖의 거리는 n ≤ 4, k ≤ 32, mB ≥ 2048이고 mA ≥ 2048이거나(mA ≥ 512, mB ≥ 32768)일 때, `minkowski`(pow 거리)는 n ≤ 8, mB ≥ 1024, mA·mB ≥ 2¹⁷일 때다. `sqeuclidean`은 √의 반올림이 두 제곱합을 같은 d로 만들 수 있어 동점 순서가 갈리므로 트리를 쓰지 않는다. 시험용 `path=3`이 트리를 강제한다. 까닭: 트리의 거리와 순서가 `cdist`의 커널과 (d̂, j)라 답이 같고(§1.3), 4000² n = 3 k = 1이 1스레드 17.6 → 2.2 ms, 10스레드 2.2 → 0.7 ms, `minkowski` p = 3 4000² n = 4 k = 10이 10스레드 97 → 6.9 ms다. 문턱은 10스레드에서 잰 갈림점(트리 만들기가 mB에 비례해 mA가 작으면 손해)이다 (추천안, 사용자 위임 2026-11-23) |
| SPT74 GEMM 고르기의 문턱(n₀) | 1 ≤ n ≤ 2¹⁶, mA ≥ 64, mB ≥ 2048(n ≥ 8이면 1024)이다(SPT40의 12 ≤ n, mB ≥ 64를 바꾼다). 직접 길의 `XA` 조각은 64행에서 `XA`가 32조각이 되도록 8행까지 줄인다. 까닭: SPT71·SPT72 뒤 GEMM 길은 4000²에서 n = 1–3도 1스레드 0.46–0.57배, 10스레드 0.62–0.72배(직접 대비)이고, mB ≤ 1000이나 mA < 64(옮긴 사본과 노름의 준비가 질의보다 크다)에서는 직접 길이 빨랐다. 조각을 줄이면 50 × 2000 같은 작은 `XA`도 10스레드가 다 일한다(0.11 → 0.04 ms) (추천안, 사용자 위임 2026-11-23) |
| SPT75 `query`의 전수 길(SPT21의 문턱) | p = 2이고 5m/6 + log₂(k)/4 + 7 − log₂ n ≥ 0.5이면 `_search`가 입력 순서의 `data`에 `kneighbors`의 전수(GEMM 고르기)를 쓴다. 까닭: 트리의 질의당 일은 대략 2^(5m/6)로, 전수는 n으로 늘어 [0, 1)ᵐ 균등 n = 10⁴, 10⁵에서 잰 갈림점(m ≈ 7.5, 11.5; k = 10이면 1 낮다)을 이 식이 0.5의 여유로 따른다. n = 10⁵, m = 16, 질의 10⁴개 k = 10이 1스레드 4568 → 445 ms, 10스레드 496 → 83 ms다. p ≠ 2는 전수에 GEMM이 없어 언제나 트리다. 결과는 같다(같은 커널, (d̂, j)) (추천안, 사용자 위임 2026-11-23) |
| SPT76 `KDTree.data`의 사본(SPT57을 바꾼다) | 트리는 잎 순서의 사본만 쥐고, `data`는 처음 읽을 때 그 사본을 원래 자리로 흩어 만든 읽기 전용 배열을 쥔다(피클과 SPT75의 전수 길도 이것을 쓴다). 까닭: `data`를 읽지 않으면 메모리가 n·m·8바이트 줄고(10⁶ × 3에서 24 MB) 만들기에서 복사 하나가 빠진다. 흩기는 값을 그대로 옮기므로 입력과 비트가 같다 (추천안, 사용자 위임 2026-11-23) |
| SPT77 `query_pairs`의 모으기 | 찾은 짝을 4096행씩의 행 블록으로 출력에 바로 흩고(태스크 × 블록의 개수와 (블록, 태스크) 순서의 누적합), 블록마다 i로 세는 정렬과 행마다 j 정렬을 따로 한다. 모든 단계가 풀에서 돈다(누적합만 직렬). 까닭: 결과는 정렬된 목록이라 순서와 무관하고, 직렬이던 i로 세는 정렬이 없어진다 (추천안, 사용자 위임 2026-11-23) |
| SPT78 `leafsize`(SPT18의 재기) | 기본 10을 그대로 둔다. 6, 8, 10, 12, 16, 24, 32를 만들기·`query`(m = 2, 3, 6)·`query_ball_point`·`query_ball_tree`·`query_pairs`·`count_neighbors`(r 하나와 20개)에서 1, 10스레드로 재니 모든 사례에서 1.2배 넘게 빠른 값이 없었다(16은 `count_neighbors` r 20개가 1.13배 빠르지만 `query_pairs`가 1.4배 느리다). 까닭: SPT18의 규칙이다 (추천안, 사용자 위임 2026-11-23) |
| SPT79 트리 만들기의 위 단계 | 위 단계를 수준마다 나눈다: 노드가 4개 이상인 수준은 노드들을 나란히(노드마다 한 스레드), 그보다 적으면 노드마다 풀로(상자·키·행 옮기기를 조각으로) 나누고, 65536점 이상의 노드는 키 4096개의 정렬한 표본에서 순위 둘레의 두 키로 띠를 잡아 조각마다 아래·띠·위를 세고 흩은 뒤 띠에만 `nth_element`를 한다(띠가 순위를 놓치면 전체 `nth_element`). m ≤ 8의 행 복사는 인라인 복사다(24바이트의 `std::copy`가 `memmove` 호출이었다). 까닭: 노드의 점 집합은 (좌표, 색인)의 전순서로 정해져 트리가 바이트까지 같고, 10⁶점 m = 3 만들기가 1스레드 226 → 205 ms, 10스레드 76 → 40 ms가 되었다 (추천안, 사용자 위임 2026-11-23) |
| SPT80 `minkowski` p = 3의 항 | \|zᵢ\|³은 `pow` 대신 (a·a)·a다. 계약의 c_pow를 2u에서 γ₂ = 2u/(1 − 2u)로 늦춘다(곱 둘의 상대오차 (1 + u)² − 1 ≤ γ₂; 차이는 4u²라 한계가 사실상 같다). 까닭: `pow`가 시간을 정해 1000² n = 64 `cdist`가 1스레드 339 → 41 ms(scipy 313 ms)가 되었고, 잰 최대는 1 ULP·계약의 0.21(가중 2 ULP·0.22)로 그대로다. p = 3의 `minkowski` 거리는 끝자리가 바뀐다(골든 다섯) (추천안, 사용자 위임 2026-11-23) |
| SPT81 `haversine`의 끝(SPT31을 바꾼다) | h ≤ 1/2이면 d = 2·asin(√h), 아니면 d = π − 2·asin(√g)(g는 SPT31의 상쇄 없는 1 − h, 그 sin((φ₁ + φ₂)/2)는 이 갈래에서만)이다. 계약 20u·d는 그대로다(asin의 조건수는 인자 √(1/2) 아래에서 1.3 이하, π의 반올림과 뺄셈은 d ≥ π/2에서 상대 2u 안). 까닭: atan2 하나와 반쪽 짝의 sin 하나가 빠져 2000² `cdist`가 1스레드 151 → 103 ms, 10스레드 17.3 → 12.5 ms다. 잰 최대는 2 ULP·계약의 0.09(대척점 넷 포함)로 그대로이고 끝자리가 바뀐다(골든 셋) (추천안, 사용자 위임 2026-11-23) |
| SPT82 비트가 같은 빠른 커널 | (1) `pair_scalar`는 32항보다 긴 짝의 덩어리를 8·4·2개씩 나란히(서로 독립인 사슬) 구해 덩어리 순서로 더한다. 한 덩어리 이하는 그대로 인라인된다. (2) `mahalanobis`는 같은 행의 짝 8개를 NEON 레인에 나란히 두고 tᵢ와 q를 짝마다의 순서 그대로 구한다(신중한 경로가 필요한 레인만 짝 하나씩). 까닭: 같은 값에 같은 연산이라 비트가 같고(골든 그대로), `mahalanobis` 1000² `cdist`가 n = 16 80 → 21 ms(scipy 77), n = 64 1163 → 295 ms(scipy 1373)다 (추천안, 사용자 위임 2026-11-23) |
| SPT83 `haversine` 이웃의 트리 길 | `kneighbors`·`radius_neighbors`의 `metric='haversine'`은 (위도, 경도)를 단위 구 위의 3차원 점 (cos φ cos λ, cos φ sin λ, sin φ)로 옮기고 기존 KD 트리를 현(유클리드) 거리로 써서 후보를 고른다. 돌려주는 모든 거리와 모든 경계 판정은 `haversine`의 짝 커널(`cdist(metric='haversine')`과 같은 비트)이다. 정확한 산술에서는 `haversine`이 현 거리에 단조이지만 반올림 뒤에는 아니라서, 참 이웃과 (거리, 색인)의 동점이 빠지지 않는 여유를 증명해 쓴다(§3.4). BallTree는 넣지 않는다(SPT8 그대로). 까닭: 10⁴ × 10⁵ k = 5의 전수가 10스레드 3.06초로 sklearn `BallTree`(97 ms)보다 30배 넘게 느렸다(§5) (사용자 결정 2026-11-23) |
| SPT84 여유의 꼴 | cut(T) = fl(fl(2·sin(fl(fl(T·c) + 2⁻⁴²)/2))·c) + 2⁻⁴⁶), c = 1 + 2⁻⁴¹이고 fl(fl(T·c) + 2⁻⁴²) ≥ 3.14이거나 T = inf면 inf다. 질의의 3차원 점에서 상자(또는 점)까지의 커널 거리가 cut(T)보다 크면 그 안의 모든 점은 d̂ > T다(정리 4). 상수는 증명한 오차(커널 2⁻⁴⁶(1 + d), 옮기기 7.5u, 3차원 커널 κ = γ₆/2 + u)의 4배 넘게 잡았다. 까닭: libm의 sin·cos·asin이 1 ULP(가정 L; Arb로 잰 최대 상대 1.14u)보다 4배 나빠도 증명이 서고, 여유가 단위 구 위 약 2.6×10⁻¹³(지구 위 2 µm 아래)이라 다시 계산하는 점이 거의 늘지 않는다 (추천안, 사용자 위임 2026-11-23) |
| SPT85 여유의 정의역 | 모든 점(`XA`와 `XB`)이 \|φ\| ≤ fl(π/2), \|λ\| ≤ 8일 때만 트리 길이고, 아니면 네이티브 함수가 None을 돌려 전수가 답한다. 까닭: 위도 차의 반이 π/2를 넘거나 경도가 크면 보조정리 H(§3.4)의 증명이 서지 않는다. 경도 [−π, π], [0, 2π), [−2π, 2π]를 모두 담고, 도 단위를 라디안으로 잘못 준 자료는 전수로 같은 답을 낸다 (추천안, 사용자 위임 2026-11-23) |
| SPT86 잎의 점 거르기와 cut의 캐시 | 잎의 점마다 3차원 제곱합을 KdEuclid::above(cut(T))와 먼저 견주어(SPT52의 거르기) 넘으면 `haversine`을 계산하지 않는다. cut(T)는 sin 하나라 T(힙의 맨 위)가 바뀔 때만 다시 구한다. `return_length`는 잎에서 센다(먼 한계로 노드를 한꺼번에 세지 않는다). 까닭: `haversine` 하나(sincos 둘과 asin)가 제곱합(곱셈과 fma 셋)보다 훨씬 비싸고, 증명은 점을 폭이 0인 상자로 본 정리 4 그대로다. 먼 한계는 현 → `haversine`의 반대 방향 여유를 따로 증명해야 해서 넣지 않았다 (추천안, 사용자 위임 2026-11-23) |
| SPT87 트리 길의 문턱과 질의 조각 | mB ≥ 128, mA ≥ 16, 8k ≤ mB이면 트리 길이다(시험용 `path=3`이 강제한다). 질의 조각은 64점에서 시작해 조각이 32개보다 적으면 4점까지 반으로 줄인다(크기로만 정함). 까닭: 1·10스레드로 mB 64–4096, mA 1–2048, k 1·8·32의 격자를 재니 mA ≥ 16, mB ≥ 128, 8k ≤ mB에서 모두 전수보다 빨랐고(1.15–205배), mA ≤ 4는 같거나 느렸고(0.2–1.2배), k가 mB/2에 가까우면 트리가 느렸다(k = 32, mB = 64, mA = 64의 10스레드 0.21배). 조각을 줄이면 mA = 64, mB = 128, k = 8의 10스레드가 0.63 → 1.32배가 된다 (추천안, 사용자 위임 2026-11-23) |
| SPT88 `haversine`의 경도 차 | Δλ = fl(λ₂ − λ₁)의 반올림 오차 e를 TwoSum으로 정확히 구해 sin((Δλ + e)/2)와 cos((Δλ + e)/2)를 e/2의 1차까지(fma 하나씩) 구한다. h < 2⁻⁹⁶⁸이면 두 사인을 2⁶⁰⁰배 해(정확) h를 다시 계산한다. 계약 20u·d는 그대로다. 까닭: 날짜 변경선을 사이에 둔 가까운 짝(Δλ ≈ ±2π)은 Δλ의 절대 반올림 u\|Δλ\|/2가 sin(Δλ/2)에 그대로 들어가 상대오차가 약 uπ/d였다(d = 10⁻¹¹에서 계약의 2×10⁹배; Arb로 확인. sklearn도 같은 식이라 d = 10⁻¹³에서 상대 4×10⁻³). 위도·경도가 10⁻³⁰⁰ 둘레인 짝은 sin²이 언더플로해 d̂ = 0이었다. 고친 뒤 Arb와 견준 최대는 적대적 집합 열(날짜 변경선, 2π 다른 경도, 극, 대척점, 10⁻³⁰⁰ 둘레)에서 계약의 0.20, 참값 파일 32짝에서 1 ULP·0.071이고, 2000² `cdist`는 1스레드 100 ms·10스레드 11.5 ms로 같다. 끝자리가 바뀐다(골든 셋) (추천안, 사용자 위임 2026-11-23) |
| SPT89 트리 길의 파일과 참값 | 트리 길은 `csrc/spatial/sphere.cpp`(`_sphere_kneighbors`, `_sphere_radius`)이고 트리는 `kd_build` 그대로, 잎 순서의 (위도, 경도, cos 위도)를 따로 쥔다. 참값은 `tests/spatial/points.py`의 `sphere` 점 집합 여섯(질의 20개 × 점 400개)에서 질의와 모든 점의 Arb `haversine`(`tools/spatial_ref.py`의 `sphere` → `tests/spatial/ref/sphere.npz`)이고, 정확도 체계의 `sphere` 스윕과 테스트가 함께 쓴다. 까닭: 다른 길(`query.cpp`, `dual.cpp`)처럼 한 파일이 한 길이고, 판정이 §4 이웃 3번의 꼴 그대로다 (추천안, 사용자 위임 2026-11-23) |

### 1.2 범위
마일스톤은 §8이다. "SPTn"은 그 마일스톤에서 만들었다는 뜻이고 "나중"은 첫 판에 넣지 않은 후보다.

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
| `BallTree` | 없음, 나중(SPT8). `BallTree(metric='haversine')`의 이웃 찾기는 `kneighbors`·`radius_neighbors`의 단위 구 KD 트리가 한다(SPT83) | 나중 |
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

## 2. API

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
- **결과는 정의로 정해진다**: `kneighbors(XA, XB, k=k)`는 `cdist(XA, XB)`의 각 행에서 거리가 `distance_upper_bound`보다 작은 원소를 (거리, 열 번호)로 정렬해 앞 k개를 고른 것과 같고(모자라면 (inf, mB)), 거리는 그 원소와 비트까지 같다. 그래서 거리가 inf이거나 NaN인 짝은 기본 상한 inf에서도 나오지 않는다(SPT37). `radius_neighbors`는 그 행에서 d̂ ≤ r인 열 전부(오름차순)다. Minkowski 계열(`euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`, `minkowski` p ≥ 1)에서는 `KDTree(XB).query(XA, k=k, p=…)`와도 비트까지 같다. 어떤 길(전수, GEMM 고르기, 트리)로 계산할지는 거리 이름과 모양(mA, mB, n, k)으로만 고르고(§3.4), 길은 결과를 바꾸지 않는다. 길은 직접 전수, GEMM 고르기(SPT74), `XB`의 KD 트리(SPT73), `haversine`의 단위 구 KD 트리(SPT83)다.
- **입력**: `XA`·`XB`는 `cdist`처럼 2차원이다(SPT49). 거리 이름, 그 인자, 기본 `V`·`VI`도 `cdist`와 같다. 데이터는 유한해야 한다(SPT12). `k`와 `distance_upper_bound`는 `KDTree.query`의 규칙(SPT55, SPT19), `r`은 실수 하나나 (mA,)로 브로드캐스트되는 배열이다(SPT46). 큰 k는 SPT45다.
- k = 1이 축약이다: sklearn의 `pairwise_distances_argmin_min(X, Y)`는 `kneighbors(X, Y)`의 `(distances, indices)`다(풀기 순서만 반대). `[sklearn과 다름]`: sklearn은 전개식이라 거리가 틀릴 수 있고(§3.2), 이 함수는 언제나 직접 계산한 거리다.
- 모든 거리 이름을 받는다(KD 트리가 칠 수 없는 거리는 전수로 하고, `haversine`은 단위 구 위의 KD 트리로 한다: SPT83). 결과의 모양과 채움은 `KDTree.query`의 규칙이다(§2.4).
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
- **속성**: `data`(쓰기 금지 (n, m) float64, 입력 순서), `n`, `m`, `leafsize`, `maxes`, `mins`(쓰기 금지 (m,); n = 0이면 0, scipy와 같다), `size`(노드 수; 같은 점이 많으면 scipy와 다르다, SPT50). scipy의 `tree`, `indices`, `boxsize`는 없다. `repr`는 `KDTree(n=1000, m=3, leafsize=10)`이다(SPT60). 피클과 `copy`는 같은 데이터와 `leafsize`로 다시 만든다(SPT56; `sd.ndarray`가 피클되지 않아 데이터의 바이트를 담는다; 만들기가 결정적이라 같은 트리다).
- **`p`**: 1 이상의 실수나 inf다(scipy와 같다; 1 미만은 `ValueError`). p = 1, 2, inf는 `cityblock`, `euclidean`, `chebyshev`와, 그 밖의 p는 `minkowski(p=)`와 같은 커널이다.
- **`query`**: 데이터 점 가운데 x에 가장 가까운 k개다.
  - x는 (…, m)이다. 마지막 축이 m이 아니면 `ValueError`.
  - `k`는 1 이상의 int거나 1 이상의 int 시퀀스(1부터 센 순위의 목록, scipy의 뜻)다. 0, 음수, 빈 시퀀스는 `ValueError`, bool·실수는 `TypeError`다(`[scipy와 다름]`: scipy 1.18.1은 `k=True`와 `k=2.0`을 받고, `k=0`은 numpy 내부 메시지의 `ValueError`이며, `k=[0]`은 프로세스를 죽인다(세그먼트 오류, 이 초안에서 확인함)).
  - 결과는 `QueryResult(distances, indices)`이고 scipy처럼 `d, i = tree.query(x)`로 풀린다(§2.5). 시퀀스 `k`는 순서와 중복이 자유다(scipy와 같다, SPT55). 모양은 scipy와 같다: k가 int면 (…, k)이고 k = 1이면 마지막 축을 없앤다. k가 시퀀스면 (…, len(k))다. x가 (m,)이고 k = 1이면 Python `float`와 `int`다(0차원 결과는 Python 스칼라, array.md §2.2; scipy는 `float`와 `numpy.int64`).
  - **순서**: 각 행은 (거리, 색인)의 사전 순서다. 같은 거리는 색인이 작은 것이 먼저다. 이것을 약속한다(`[scipy와 다름]`: scipy는 약속하지 않고 실제로도 다르다: 같은 점 6개에서 scipy의 순서는 `[1, 2, 3, 4, 5, 0]`, sklearn은 `[5, 2, 4, 0, 1, 3]`이다). 비교하는 거리는 돌려주는 값 그대로다(제곱 거리가 아니다; §3.4).
  - **채움**: 찾은 이웃이 k보다 적으면(k > n, 또는 `distance_upper_bound`) 남은 자리는 거리 inf, 색인 n이다(scipy와 같다. sklearn은 k > n이면 에러다).
  - **`distance_upper_bound`**: 거리가 이 값보다 **작은** 이웃만 돌려준다(scipy와 같이 엄격한 부등호; 거리가 꼭 1인 점은 상한 1에서 빠진다). 0 이상의 실수나 inf여야 한다(`[scipy와 다름]`: scipy는 p = 2에서 상한을 제곱해 비교하므로 −1이 1처럼 동작하고, NaN이면 모든 자리가 채움 값이다; §9의 14번).
- **`query_ball_point`**: 거리가 r **이하**인 데이터 점이다(scipy와 같이 경계를 넣는다). r은 0 이상의 실수나 inf(또는 x의 배치 모양을 바꾸지 않고 브로드캐스트되는 배열, SPT67)이고 음수·NaN은 `ValueError`다(`[scipy와 다름]`: scipy는 r을 제곱해 비교하므로 r = −1이 r = 1처럼 동작하고, NaN이면 빈 목록이다). 판정은 돌려주는 거리 d̂ ≤ r이다(§3.4).
  - 결과는 `RadiusResult`다(§2.5). 질의점 x의 배치 축은 C 순서로 펼쳐 행이 된다. 점 하나(m,)도 행 하나의 결과다(`[scipy와 다름]`: scipy는 점 하나면 list, 여럿이면 list의 object 배열이다).
  - 행 안은 색인 오름차순이다. `return_sorted`는 없다(§9의 12번; `[scipy와 다름]`: scipy의 기본은 점 하나면 정렬하지 않고 여럿이면 정렬한다).
  - `return_length=True`면 행마다의 개수를 int64 배열(x의 배치 모양; 점 하나면 Python `int`)로 돌려준다(scipy와 같다). 색인을 모으지 않아 메모리가 적고, 먼 한계가 노드 전체를 안으로 증명하면 거리를 계산하지 않고 점 수를 더한다(SPT63).
  - 결과는 정의로 정해진다: `radius_neighbors(x, data, r)`(p = 1, 2, inf는 `cityblock`·`euclidean`·`chebyshev`, 그 밖의 p는 `metric='minkowski', p=p`)와 `indptr`·`indices`·`distances`의 바이트까지 같다. 결과가 물리 메모리를 넘으면 `MemoryError`다(SPT68).
- **`query_ball_tree(other, r)`**: `self`의 점마다 `other`의 점 가운데 거리가 r 이하인 것이다. 결과는 `RadiusResult`이고 모양은 (self.n, other.n)이다. r은 실수 하나다. `other`가 `KDTree`가 아니면 `TypeError`, `other.m`이 다르면 `ValueError`(scipy와 같다). 두 트리를 함께 내려간다(§3.4). `other.query_ball_point(self.data, r)`, `radius_neighbors(self.data, other.data, r)`와 같은 결과다.
- **`query_pairs(r)`**: i < j이고 거리가 r 이하인 짝이다. (k, 2) int64 배열이고 (i, j)의 사전 순서다(§9의 12번; `[scipy와 다름]`: scipy의 기본은 순서 없는 `set`이고 `output_type='ndarray'`도 scipy 순회의 순서라 정렬되어 있지 않다(SPT4에서 확인). `output_type`은 없다). `pdist`의 짝 (i, j) 가운데 그 원소가 r 이하인 것과 같다.
- **`count_neighbors(other, r)`**: `self`의 점 x₁과 `other`의 점 x₂의 순서쌍 가운데 거리가 r 이하인 것의 수다(scipy와 같다: `other`가 `self`면 (i, i)와 (i, j)·(j, i)를 모두 센다). r은 실수 하나나 1차원 배열이다. `cumulative=False`면 r은 줄지 않아야 하고 결과의 i번째는 r[i−1] < d ≤ r[i]의 수다(첫째는 d ≤ r[0]). 가중치는 scipy의 꼴(배열 하나는 같은 트리일 때만, 튜플 `(w_self, w_other)`의 한쪽은 None일 수 있다)이고 가중이면 짝의 가중치 곱의 합이다. 결과는 r이 실수 하나면 Python `int`(가중이면 `float`), 배열이면 int64(가중이면 float64) 배열이다(scipy와 같다). 정수 셈은 `cdist(self.data, other.data)`의 원소 가운데 r 이하인 것의 수와 정확히 같다. r의 꼴과 무게의 검사는 SPT66, 셈의 방법은 SPT65다(`[scipy와 다름]`: scipy는 r = NaN에서 모든 짝을, 음수 r에서도 짝을 세고, NaN 무게를 받는다).
- **`sparse_distance_matrix(other, max_distance)`**: 거리가 `max_distance` 이하인 짝의 거리를 담은 `sd.sparse.csr_array` (self.n, other.n)이다. `self.query_ball_tree(other, max_distance).tocsr()`와 같다. 거리 0인 짝은 저장된 0이다(§2.5). `[scipy와 다름]`: scipy의 기본은 `dok_matrix`이고 `output_type`으로 `dok_array`, `coo_array`, `dict`, 구조 배열을 고른다. sd에는 옛 `*_matrix`가 없고(sparse.md §1.1), 형식을 바꾸려면 `asformat`을 쓴다.
- `workers`, `eps`는 없다(§9의 5번, 6번).

### 2.5 결과 객체
- **`QueryResult(distances, indices)`**: `query`와 `kneighbors`가 함께 쓰는 namedtuple이다(`stardust._results`, DESIGN §9.7). 두 함수는 같은 것을 돌려주므로 이름이 하나다. 배열은 쓰기 금지다(API.md §2.2).
- **`RadiusResult`**: `query_ball_point`, `query_ball_tree`, `radius_neighbors`가 함께 쓰는 결과 클래스다(`stardust.spatial`의 `_radius.py`, SPT44; htest처럼 클래스를 내보내지 않는다, §9의 12번). 생성자 `RadiusResult(indptr, indices, distances, shape)`는 받은 배열을 검사하거나 복사하지 않고 쓰기 금지로 만든다(SPT4의 반경 질의도 이것으로 만든다).
  - 필드: `indptr`(int64, 행 수 + 1), `indices`(int64), `distances`(float64), `shape`(행 수, 데이터 점 수). 행 i의 이웃은 `indices[indptr[i]:indptr[i+1]]`이고 색인 오름차순, 그 거리는 같은 자리의 `distances`다. 배열은 쓰기 금지이고 읽을 때 복사하지 않는다(API.md §2.2).
  - `tocsr()`: `sd.sparse.csr_array((distances, indices, indptr), shape=shape)`다. 색인이 이미 표준 꼴(정렬, 중복 없음)이라 sparse의 표준 꼴 규칙(sparse.md §2.3)을 그대로 만족한다. 부를 때마다 새 배열이다(희소 배열의 `data`는 쓸 수 있어야 하므로 거리를 복사한다; 색인도 희소 생성자의 규칙대로 복사하고 검사한다. 비용은 찾기에 비해 작다).
  - **거리 0인 짝**: 같은 점(자기 자신, 중복된 점)의 거리 0은 **저장된 0**으로 들어간다. `nnz`가 그 짝을 세고, `count_nonzero()`는 세지 않는다. 지우려면 `eliminate_zeros()`를 부른다. 이유: 희소 행렬의 저장 자리가 이웃 그래프의 변이다. 0을 지우면 중복된 점과 자기 자신이 이웃 목록에서 사라져 `indices`와 `.tocsr()`의 자리가 서로 다른 것을 말하게 된다. scipy의 `sparse_distance_matrix`도 0인 짝을 저장한다(이 초안에서 확인: 같은 두 점의 `coo_array`는 nnz 4, `dok_array`는 nnz 4에 `count_nonzero()` 0).
  - `repr`는 한 줄이다(`RadiusResult(shape=(1000, 5000), nnz=10234)`, API.md §2.3의 꼴). 회귀·검정·최적화의 결과가 아니므로 `summary()`는 없다.
- 결과 객체가 아닌 결과(`pdist`, `cdist`, `squareform`, `query_pairs`, `count_neighbors`, `return_length`)는 보통의 새 배열이라 쓸 수 있다.

### 2.6 scipy·sklearn과 다른 곳 (모음)
위의 `[scipy와 다름]`을 모은다. 이유는 그 자리에 있다.
- 거리: 별칭과 대문자 이름 없음, `metric`과 그 인자는 키워드만, 쓰지 않는 인자는 `TypeError`, complex는 `TypeError`, `p ≤ 0`·`w`의 NaN·inf·음수 `V`는 `ValueError`, `chebyshev`와 `canberra`는 NaN을 퍼뜨린다, `euclidean`·`minkowski`는 넘치지 않는다, 모든 거리 함수가 배치를 받는다, 호출 가능한 `metric`과 `out=` 없음(§9의 9번).
- 트리: 동점은 (거리, 색인), 데이터는 언제나 복사, `leafsize`만, `k`·`leafsize`는 int만, `k=[0]`·음수와 NaN 반경·음수 상한·NaN 무게는 `ValueError`, 반경 결과는 `RadiusResult`이고 언제나 색인 순서, `query_pairs`는 정렬된 배열, `sparse_distance_matrix`는 CSR, `workers`·`eps`·`boxsize`·`return_sorted`·`output_type` 없음.

---

## 3. 수치 방법

### 3.1 거리 커널
**짝 하나의 거리는 한 커널이 정한다.** `pdist`, `cdist`, 거리 함수, `kneighbors`·`radius_neighbors`, 트리의 잎, GEMM 고르기 뒤의 다시 계산이 모두 같은 함수 `pair_distance<Metric>(x, y, n, params)`의 순서로 계산한다. 그래서 같은 짝은 어디서든 같은 비트다(SPT2). 묶음 커널(여러 짝을 레지스터에 함께 두는 것)은 짝 하나의 덧셈 순서를 바꾸지 않는다.

- **차이**: zᵢ = fl(xᵢ − yᵢ). 레지스터에서 빼고 바로 쓴다(배열에 쓰지 않는다). `sd.subtract`와 같은 반올림 한 번이다.
- **합의 순서는 타일 순서다**(SPT22의 측정, SPT26): 짝마다 항을 kTileBlock = 32개씩 덩어리로 나누고, 덩어리 안은 0에서 시작해 `fma(zᵢ, zᵢ, t)`를 차례로, 덩어리의 합은 차례로 더한다(s = t₁, s = s + t₂, …). 묶음 커널은 x 4행 × y 8행의 32짝을 누산 레지스터 16개(레지스터 하나에 두 짝)에 두고, y 블록을 열 우선의 판(panel)으로 한 번 옮겨 둔다. 짝 하나를 계산하는 커널(트리의 잎, 다시 계산)도 같은 순서라 같은 비트다. 구현은 `csrc/spatial/metric.hpp`의 `tile<Op, R, W>`(R × 8짝: 누산기가 하나인 거리는 R = 4, 둘인 `braycurtis`·`jaccard`는 R = 2)와 `pair_scalar<Op, W>`(32항보다 긴 짝은 서로 독립인 덩어리 사슬을 8·4·2개씩 나란히 구해 덩어리 순서로 더한다: SPT82)이고, 연산(Op)마다 NEON과 스칼라의 같은 IEEE 연산(뺄셈, 절댓값, fma, 덧셈, 나눗셈, FMAX, 비교)으로 적어 레인과 스칼라가 같은 비트다. 남는 행과 열은 채워서 계산하고 버린다. SPT0의 시제품은 `tools/spatial_probe_pair_kernel.hpp`로 옮겼다(SPT36).
  - 합의 깊이(한 항이 거치는 반올림의 수)는 m = min(n, 32) + ⌈n/32⌉ − 1이다(§4의 계약). n < 16이면 덩어리 하나의 차례 합이라 `sd.linalg.vector_norm(x − y)`와 같은 비트다(탐침: n = 3, 8, 15에서 256짝 모두 같다). n ≥ 16에서는 linalg의 16레인 잎과 달라 끝자리가 다를 수 있다(탐침: 256짝 가운데 106–216짝이 같다). 둘 다 정확도 계약 안이다.
  - `sqeuclidean`은 같은 제곱합 s 그대로다. `euclidean`은 √s이고, s가 inf이거나 2⁻⁹⁶⁸ 아래면(NaN은 그대로) 신중한 경로다(최댓값의 지수로 고른 2ᵏ로 비율을 맞춘 같은 순서의 두 번째 패스, SPT27, SPT33). `sqeuclidean`도 2⁻⁹⁶⁸ 아래에서 같은 두 번째 패스를 하고 2²ᵏ를 곱한다(결과가 비정상 수일 수 있다). 참 제곱 거리가 넘치면 `sqeuclidean`의 inf가 맞는 값이다. `pdist`·`cdist`의 마무리(√, 나눗셈, 자르기)는 두 짝씩 NEON으로 하고 신중한 경로가 필요한 레인만 짝 하나씩 다시 한다(같은 비트).
  - `cityblock`은 같은 꼴의 |zᵢ| 합(가중이면 `fma(w, |z|, t)`), `minkowski(p)`는 |zᵢ|ᵖ(`std::pow`, p = 3은 (a·a)·a: SPT80)의 같은 꼴의 합과 근이다. 근은 `pow(s, 1/p)`에 1/p가 2의 거듭제곱이 아니면 뉴턴 한 걸음을 더하고, 합이 inf이거나 2⁻⁹⁶⁸ 아래면 |zᵢ|/max|z|의 합과 max|z|·근이다(SPT30). p = 1, 2, inf는 `cityblock`, `euclidean`, `chebyshev`로 간다(비트까지 같다; 가중도).
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
- **`mahalanobis`**: 짝마다 z를 만들고 tᵢ = VI의 i행·z(타일 순서의 `fma`), q = Σzᵢtᵢ(타일 순서), √q(q < 0이면 NaN)다. |q|가 2⁻⁹⁶⁸ 아래이거나 inf면 z를 2ᵏ로 비율을 맞춰 다시 한다. 짝 하나가 O(n²)이다(scipy도 같다). BLAS를 쓰면 비트가 Accelerate의 묶음에 따라 달라지므로 이 라이브러리의 커널이다. `pdist`·`cdist`·배치 함수는 짝 8개를 NEON 레인에 나란히 두고 짝마다의 순서 그대로 구한다(SPT82; 비트가 같다): 1000² n = 16에서 1스레드 21 ms(scipy 77 ms).
- **`haversine`**(§9의 4번, SPT31, SPT81, SPT88): h = sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2)이고 h ≤ 1/2이면 d = 2·asin(√h), 아니면 d = π − 2·asin(√g), g = cos²(Δφ/2) cos²(Δλ/2) + sin²((φ₁ + φ₂)/2) sin²(Δλ/2)(= 1 − h, 상쇄 없이)다(SPT31의 2·atan2(√h, √g)를 SPT81이 바꿨다). Δλ는 TwoSum으로 반올림 오차 e까지 정확히 두고 sin·cos((Δλ + e)/2)를 e/2의 1차까지 구한다(fma 하나씩; SPT88: 날짜 변경선을 사이에 둔 가까운 짝). h < 2⁻⁹⁶⁸이면 두 사인을 2⁶⁰⁰배 해 다시 계산한다(언더플로; SPT88). Δφ/2, Δλ/2의 sin·cos는 `base/sincos.hpp`로 함께, cos φ는 행마다 한 번 구한다. 대척점 넷을 포함한 32짝에서 최대 1 ULP다. 짝마다 sincos 둘과 asin(먼 반쪽은 sin 하나 더)이라 1스레드에서 sklearn보다 약 2.3배 느리다(2000² 100 ms 대 44 ms; 10스레드 11.5 ms). 묶음 sin·cos(Accelerate vForce)는 원소의 비트가 배열 안의 자리에 따라 같다는 보장이 없어 쓰지 않는다.
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

**보조정리 1** (전개식의 오차): |Ê − D²| ≤ γₙ₊₂·(X + Y)² + (6n + 1)·2⁻¹⁰⁷⁵ =: Δ(x, y).
- 증명: 제곱합은 항이 모두 양수라 |n̂ₓ − X²| ≤ γₙX² + n·2⁻¹⁰⁷⁵(1 + γₙ), y도 같다. s = fl(n̂ₓ + n̂ᵧ) = (n̂ₓ + n̂ᵧ)(1 + θ₁), Ê = (s − 2ĝ)(1 + θ₂), |θᵢ| ≤ u(덧셈과 뺄셈은 η가 없고 2ĝ는 정확하다). s − 2ĝ − D² = (n̂ₓ − X²) + (n̂ᵧ − Y²) + θ₁(n̂ₓ + n̂ᵧ) − 2(ĝ − x·y)이므로 반올림의 몫은 γₙ(X² + Y²) + u(1 + γₙ)(X² + Y²) + 2γₙXY ≤ γₙ₊₁(X + Y)²이고(γₖ + u(1 + γₖ) ≤ γₖ₊₁), 언더플로의 몫은 6n·2⁻¹⁰⁷⁵(1 + γₙ)(1 + u) 이하다(n̂ₓ와 n̂ᵧ에서 2n·2⁻¹⁰⁷⁵(1 + γₙ)(1 + u), 2ĝ에서 가정 A의 두 배인 4n·2⁻¹⁰⁷⁵). Ê − D² = (s − 2ĝ − D²)(1 + θ₂) + θ₂D²이고 D ≤ X + Y이므로 γₙ₊₁(1 + u) + u ≤ γₙ₊₂, 6n·2⁻¹⁰⁷⁵(1 + γₙ)(1 + u)² ≤ (6n + 1)·2⁻¹⁰⁷⁵에서 끝난다(n < 2²¹이라 6n((1 + γₙ)(1 + u)² − 1) < 1). ∎

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
- (2) Δ̃ ≥ Δ(x′, y′): A = X′ + Y′라 하면 â ≥ (Rₓ + Rᵧ)(1 − u) ≥ A(1 − u), fl(â·â) ≥ â²(1 − u) − 2⁻¹⁰⁷⁵, fl(c₁·…) ≥ c₁â²(1 − u)² − 2⁻¹⁰⁷⁴(c₁ < 1), Δ̃ ≥ (c₁â²(1 − u)² − 2⁻¹⁰⁷⁴ + η₀)(1 − u) ≥ c₁(1 − u)⁵A² + (η₀ − 2⁻¹⁰⁷⁴)(1 − u)다. c₁(1 − u)⁵ = (n + 3)u(1 − u)⁵ ≥ (n + 2)u(1 + 2⁻²¹)(1 − u)⁵ ≥ γₙ₊₂이고 (η₀ − 2⁻¹⁰⁷⁴)(1 − u) = (32n + 126)(1 − u)·2⁻¹⁰⁷⁵ ≥ (6n + 1)·2⁻¹⁰⁷⁵이므로 Δ̃는 보조정리 1의 한계 이상이다.
- (3) t > 0이면 D′² ≥ Ê′ − Δ̃ ≥ t/(1 + u)(뺄셈은 η가 없다; D′ = ‖x′ − y′‖)이고 D′ ≥ √t(1 − u) ≥ fl(√t)(1 − 2u)다.
- (4) 중심 빼기: D ≥ D′ − uA/(1 − u)이고 A ≤ â/(1 − u)다. e ≥ (2⁻⁵⁰â + 2⁻¹⁰⁷⁵)(1 − u) ≥ 8uâ(1 − u) ≥ uâ(1 + 5u) ≥ uA/((1 − u)(1 − 2u))이므로 D ≥ (1 − 2u)(fl(√t) − e)다(빼지 않으면 e = 0이고 D = D′).
- (5) v > 0이면 v ≤ (fl(√t) − e)(1 + u)이므로 D ≥ (1 − 2u)v/(1 + u) ≥ (1 − 3u)v다.
- (6) v ≥ 2⁻⁵⁰⁰이면 D ≥ 2⁻⁵⁰¹이라 보조정리 2가 서고 d̂ ≥ (1 − γₙ₊₇)(1 − 3u)v ≥ (1 − (n + 11)u)v다. L̃ = fl(c₂v) ≤ c₂v(1 + u)(정상 범위라 η가 없다)이고 c₂(1 + u) ≤ 1 − (n + 15)u이므로 L̃ ≤ d̂다. 제곱: ŝ ≥ (1 − γₙ₊₆)(1 − 3u)²v² ≥ (1 − (n + 13)u)v²이고 L̃ ≤ c₅v²(1 + u)² ≤ (1 − (n + 21)u)v²다. ∎
- 수치 확인(SPT0, NumPy의 `@`와 타일 순서의 Python 판): `far/1e8`, `far/1e15`, `far/1e8x16`, `nn/normal3`, `lattice/dyadic3`, `normal/64`의 질의 40개 × 점 80개, 중심 빼기의 켬과 끔에서 L̃ ≤ d̂가 아닌 짝은 없었고 L̃/d̂의 최댓값은 1 − 7×10⁻¹⁵였다(여유가 꼭 필요한 만큼에 가깝다). 중심을 빼지 않으면 먼 군집(1e8 둘레)에서는 t > 0인 짝이 없어 고르기가 아무것도 버리지 못한다.
- 여유를 줄이는 것(γ 대신 합의 깊이 m, (X + Y)² 대신 더 좁은 한계)은 빠르기에만 영향이 있다. 상수는 좁히지 않았다(SPT71의 블록 공통 한계와 SPT72의 후보 순서가 거르기의 비용을 줄였다).
- **대체 길**: 노름 n̂이나 (‖x′‖ + ‖y′‖)²이 유한하지 않으면(좌표가 약 1e154를 넘음) 그 질의·블록은 GEMM 없이 직접 전수로 간다. 이웃 찾기의 입력은 유한하므로(§2.1) NaN은 오지 않는다.
- **결과는 GEMM과 무관하다**: 고르기가 참 후보를 빠뜨리지 않으므로 마지막 답은 직접 커널만으로 정해진다. 그래서 결과는 Accelerate의 스레드 모드(`VECLIB_MAXIMUM_THREADS`)와도 무관하다(linalg의 결과와 다른 점이다). GEMM 블록의 크기와 순서는 빠르기만 바꾼다.
- **어디에 쓰나**: GEMM 고르기는 가중치 없는 `euclidean`·`sqeuclidean`(`minkowski` p = 2 포함)에서 1 ≤ n ≤ 2¹⁶, mA ≥ 64, mB ≥ 2048(n ≥ 8이면 1024)일 때의 `kneighbors`·`radius_neighbors`(트리 길이 아닐 때, SPT73)와 그 길로 가는 `query`(SPT75)다(SPT74가 SPT40의 n₀ = 12를 바꿨다). 짝마다의 거르기 앞에 블록 공통 한계(SPT71)가 있고, 후보의 순서는 SPT72다. `pdist`·`cdist`는 모든 거리를 돌려주므로 쓰지 않는다.
- **구현** (SPT2, `csrc/spatial/neighbors.cpp`): 태스크(SPT42)마다 `XA` 조각의 옮긴 행 x′와 그 n̂·R을 한 번 구하고, `XB`를 블록(2048행까지, 2¹⁹/n행 이하)으로 나눠 블록마다 옮긴 행 y′를 만들어 `dgemm`으로 G = X′Y′ᵀ를 구한다(n̂ᵧ·Rᵧ는 호출마다 한 번 구해 둔다: `XB`의 행마다 두 double). 블록과 조각의 n̂이 유한하지 않거나 (max Rₓ + max Rᵧ)²이 넘치면 그 블록은 대체 길이다(모든 짝을 다시 계산). 행마다 L̃를 NEON으로 한 번에 구하고(SPT39), kNN은 첫 후보의 규칙(SPT41) 뒤에 T = 힙의 맨 위로 훑고, 반경은 T = r로 훑는다. 다시 계산은 `pair_scalar<SqOp>`와 `euclid_finish`·`sq_finish`로 `cdist`의 타일 레인과 같은 비트다.
- **잰 고르기** (4000², 표준정규, 10스레드; ms): n = 3 2.2(직접)/3.1(GEMM), n = 16 3.9/3.4, n = 64 10.7/4.7, n = 512 75.5/22.3. 1스레드 n = 3에서 GEMM 길의 거르기가 짝당 약 1.4 ns다. 1e8 + [0, 1)¹⁶의 먼 군집도 원점 근처와 같은 시간이다(2.97 ms: 중심 빼기). 적대적 순서(질의에서 먼 점부터 가까운 점 순으로 정렬한 `XB`, 20000 × 32, k = 10)에서는 블록마다 많은 짝이 T를 줄여 GEMM 길이 직접 전수보다 1.3배 느리다(55.8 대 42.6 ms; 답은 같다).
- **넓히기(나중)**: 같은 틀은 고르기를 위한 어떤 변환에도 쓴다. 변환의 반올림을 여유에 넣으면 된다: `seuclidean`·가중 `euclidean`(좌표에 √(1/V)·√w를 곱함), `mahalanobis`(VI의 Cholesky 인자로 옮김, VI가 양정치일 때), `cosine`(정규화한 행의 유클리드). 첫 판은 `euclidean`과 `sqeuclidean`만이다.

### 3.3 KD 트리 만들기 (SPT3에서 구현, `csrc/spatial/kdtree.cpp`)
- **나누기**: 점이 `leafsize`보다 많은 노드마다 점들의 범위가 가장 넓은 축(같으면 번호가 작은 축)을 고르고, 그 축의 좌표로 (값, 색인)의 전순서를 써서 가운데 자리 ⌊n_node/2⌋에서 `nth_element`로 나눈다(sklearn의 `IndexComparator`와 같은 비교). 전순서라 두 쪽의 점 **집합**이 하나로 정해진다(같은 좌표가 많아도). 상자의 폭이 0이어도 나누고(SPT50), 잎 안의 점은 원래 색인 순서로 둔다. 그래서 트리는 데이터와 `leafsize`의 함수이고, `nth_element`의 구현, 스레드 수, 만드는 순서와 무관하다.
  - scipy의 기본(`balanced_tree=True`)도 중앙값이다. scipy의 다른 길(미끄러지는 중점)은 두지 않는다(§9의 13번).
- **노드 상자**: 노드의 실제 점들의 최솟값·최댓값이다(scipy의 `compact_nodes=True`와 같은 꼴). 가지치기의 거리 한계는 이 상자로 구한다. 나누는 값과 같은 좌표가 양쪽에 있어도 상자가 실제 범위라 한계가 맞다.
- **배치**: 노드 크기가 n과 `leafsize`로만 정해지므로 노드 수(c ↦ 1 + N(⌊c/2⌋) + N(⌈c/2⌉), c ≤ `leafsize`면 1)와 앞순회 번호(왼쪽 자식은 다음 번호, 오른쪽 자식은 그 뒤 왼쪽 하위 트리의 노드 수만큼 건넌 번호)를 미리 셀 수 있다. 노드 배열을 한 번에 잡고, n·max(m, 1) ≥ 2¹⁵이면 점이 max(2¹², n/64)개 이하인 하위 트리를 태스크로 나눠 만든다(그 위는 호출 스레드; 큰 노드의 상자·키·행 옮기기는 2¹⁴점 조각으로 풀; 각 하위 트리는 자기 자리에만 쓴다; SPT53). 결과는 직렬로 만든 것과 바이트까지 같다(테스트가 1, 3, 10스레드의 노드·상자·색인 바이트를 견준다).
- **행 옮기기**: 나누기는 행 자체를 옮긴다. 노드의 점이 트리 사본의 이어진 행 [s, e)라 상자의 훑기와 키 채우기가 이어 읽기다. 키는 (좌표, 행 자리)의 16바이트이고 동점일 때만 원래 색인을 읽는다. `nth_element` 뒤 행과 색인을 키 순서로 임시 버퍼에 모았다가 되돌려 쓴다.
- **데이터 사본**: 만들기가 끝나면 그 행이 곧 잎 순서의 (n, m) 사본이고 원래 색인(int64)과 함께 쥔다. 잎의 점이 메모리에 붙어 있어 잎 안의 거리 계산이 이어 읽기다. 공개 속성 `data`는 처음 읽을 때 잎 순서의 사본을 원래 자리로 흩어 만드는 입력 순서의 읽기 전용 사본이다(SPT76이 SPT57을 바꿨다: 읽지 않으면 사본은 하나다).
- **비용**: O(n log n)이다. n = 10⁶, m = 3에서 1스레드 약 205 ms(scipy 230–260, sklearn 207–240), 10스레드 약 40 ms다(§5). 위 단계는 수준마다 나누고(노드가 4개 이상이면 나란히), 65536점 이상의 노드는 표본으로 잡은 띠에만 `nth_element`를 한다(SPT79).

### 3.4 질의 (kNN은 SPT3, 반경·짝·개수는 SPT4에서 구현; `csrc/spatial/kdtree.hpp`, `query.cpp`, `dual.cpp`)
- **거리와 비교**: 잎의 점은 §3.1의 직접 커널로 거리를 구한다(p = 1, 2, inf는 `cityblock`·`euclidean`·`chebyshev`의 `pair_scalar`, 그 밖의 p는 `minkowski_pair`; `cdist`·`minkowski`와 같은 비트). 순서와 판정은 돌려주는 값 d̂로 한다. p = 2에서는 제곱합 r로 먼저 걸러 낸다(SPT52): r이 보통 범위 [2⁻⁹⁶⁸, DBL_MAX](신중한 경로가 아니라 d̂ = RN(√r))이고 r > A(T) = fl(fl(T·T)·(1 + 8u))면 d̂ > T가 보장되어 √ 없이 버린다. 증명: T²이 보통 수면 fl(T·T) ≥ T²(1 − u)라 A(T) ≥ T²(1 + 5u)이고, r > A(T)면 √r·(1 − u) > T이므로 d̂ = RN(√r) ≥ √r(1 − u) > T다. T²이 보통 범위 아래면 √r ≥ 2⁻⁴⁸⁴ > T라 그대로 맞고, T = inf면 A = inf라 아무것도 버리지 않는다. 그 밖의 점은 d̂를 구해 (d̂, 색인)으로 견준다. 제곱합만으로 순서를 정하면 √의 반올림이 두 값을 같게 만들 때 색인의 순서가 틀리므로, 넣을지 말지는 언제나 d̂로 정한다.
  - scipy는 제곱의 자리에서 r²과 비교한다. 그래서 돌려준 거리가 r과 같은데도 빠지거나 그 반대인 경계가 있을 수 있다(무작위 시험에서 r = 계산한 거리일 때 √(d²) ≤ r와 d² ≤ r²이 2,000번 가운데 460번(23%) 달랐다; `tests/spatial/data/scipy.json`). sd의 판정은 돌려주는 거리 그 자체라 결과와 거리가 언제나 맞는다.
- **가지치기는 보수적으로**(SPT51): 노드의 한계 b는 노드 상자에서 x에 가장 가까운 점 c(cᵢ = min(max(xᵢ, loᵢ), hiᵢ), 정확한 연산)까지의 같은 커널의 거리다. 상자 안의 모든 점 y는 좌표마다 |yᵢ − xᵢ| ≥ |cᵢ − xᵢ|라 참 거리가 D(x, y) ≥ D(x, c)다. T가 점이 이겨야 할 거리(힙이 덜 찼으면 `distance_upper_bound`, 찼으면 맨 위의 거리)일 때 b > T₁ = fl(fl(T·c) + η)이면 친다(c = 1 + 4(m + 12)u, η = 2⁻¹⁰⁷⁰; c는 1 위의 간격이 2u라 정확하다). 증명: 거리 계약(§4)이 |d̂ − D| ≤ κD + 2⁻¹⁰⁷⁴이고 p ≥ 1에서 κ ≤ (m + 11)u다(`euclidean` γₘ₊₃/2 + u, `cityblock` γₘ₊₂, `minkowski` 4u + (γₘ + 4u)/p + 2u; 합의 깊이 ≤ m). 곱의 반올림은 절대 2⁻¹⁰⁷⁵ 안이고 합은 절대 오차가 없으므로 T₁ ≥ T·c(1 − u)² + (η − 2⁻¹⁰⁷⁵)(1 − u)다. 그러면 d̂(y) ≥ (1 − κ)D(x, c) − 2⁻¹⁰⁷⁴ ≥ (1 − κ)(b − 2⁻¹⁰⁷⁴)/(1 + κ) − 2⁻¹⁰⁷⁴ > T다: c(1 − u)²(1 − κ)/(1 + κ) ≥ 1 + (2m + 24)u − O(m²u²) > 1이고 η가 2⁻¹⁰⁷⁴의 네 배보다 크다. 그래서 친 노드의 점은 모두 T보다 **엄격히** 멀어 이웃도, 동점(같은 거리에 색인이 작은 점)도 빠지지 않는다. p = inf의 커널은 올바르게 반올림한 차이의 최댓값이라 |yᵢ − xᵢ|마다 단조이고 d̂(y) ≥ b가 정확히 서므로 b > T 그대로다. p = 2의 노드도 합 r이 보통 범위면 r > A(T₁)로 견준다(그때 b > T₁). 한계는 노드마다 새로 구한다(증분 갱신이 아님; 한계 하나가 점 하나의 거리 값이다). 여유는 빠르기에만 영향이 있다(아주 작다).
- **kNN**: 깊이 우선이다. 내부 노드에서 두 자식의 한계를 구해 작은 쪽(같으면 왼쪽)을 먼저 내려가고, 돌아오면 다른 쪽을 그때의 T로 다시 견준다. 질의마다 크기 min(k, n)의 (d̂, 색인) 최대 힙이다. `distance_upper_bound`는 힙이 찰 때까지의 T이고 판정은 d̂ < 상한이다. k가 시퀀스면 max(k)개를 찾고 그 순위만 고른다(없는 순위는 (inf, n)).
- **두 상자 사이의 한계**(SPT63, `KdPair`): 상자 A, B(질의점 x는 상자 (x, x))의 한계는 상자 모서리로 만든 두 점 a, b의 짝 커널 거리다. 그래서 한계도 짝 하나의 거리이고 위의 증명을 그대로 쓴다.
  - **가까운 한계**: 축마다 틈을 사이에 둔 모서리(loᴮᵢ > hiᴬᵢ면 hiᴬᵢ와 loᴮᵢ, loᴬᵢ > hiᴮᵢ면 loᴬᵢ와 hiᴮᵢ, 겹치면 0과 0)다. |aᵢ − bᵢ|가 틈 그 자체라 A의 모든 x, B의 모든 y에서 D(a, b) ≤ D(x, y)이고, b̂ > T₁ = KdCut(r)이면 위의 증명으로 노드 쌍의 모든 짝이 d̂ > r이다. 질의점과 상자에서는 SPT51의 고정점 c와 같은 차이다.
  - **먼 한계**: 축마다 fl(hiᴮᵢ − loᴬᵢ)와 fl(hiᴬᵢ − loᴮᵢ) 가운데 큰 쪽(같으면 앞쪽)의 두 모서리다. 상자 위의 가장 큰 |xᵢ − yᵢ|인 fᵢ는 두 참 차이 중 큰 것이고 반올림은 단조라, 고른 쪽의 참 차이 tᵢ는 fᵢ이거나(반올림이 다르면) 두 값이 같은 R로 반올림된 경우 tᵢ ≥ fᵢ(1 − u)/(1 + u) ≥ fᵢ(1 − 2u)다(비정상 수 차이는 정확하다). p-노름은 좌표마다 단조이고 동차라 D(x, y) ≤ D(f) ≤ D(a, b)/(1 − 2u)다. 거리 계약 d̂(x, y) ≤ (1 + κ)D(x, y) + 2⁻¹⁰⁷⁴와 D(a, b) ≤ (b̂ + 2⁻¹⁰⁷⁴)/(1 − κ)를 합치면 d̂(x, y) ≤ (1 + κ)/((1 − κ)(1 − 2u))·(b̂ + 2⁻¹⁰⁷⁴) + 2⁻¹⁰⁷⁴ ≤ fl(fl(b̂·c) + η)다: c(1 − u)² ≥ 1 + (4m + 46)u − O(u²)가 (1 + κ)/((1 − κ)(1 − 2u)) ≤ 1 + (2m + 24)u + O(u²)를 넘고(κ ≤ (m + 11)u), (η − 2⁻¹⁰⁷⁵)(1 − u) > 15·2⁻¹⁰⁷⁴이 절대 항을 덮는다. 그래서 KdCut(b̂) ≤ r이면 노드 쌍의 모든 짝이 d̂ ≤ r이다. 차이가 넘치면 b̂ = inf라 r = inf일 때만 안이고, 유한한 데이터의 d̂는 언제나 inf 이하라 그때도 맞다. p = inf는 커널이 올바르게 반올림한 |xᵢ − yᵢ|의 최댓값이라 d̂(x, y) ≤ b̂가 정확히 서므로 b̂ ≤ r 그대로다.
- **반경**(`query_ball_point`, `query.cpp`): 질의점마다 깊이 우선으로 내려가 가까운 한계가 T₁보다 큰 노드를 치고, 잎의 점은 d̂ ≤ r로 넣는다(p = 2는 제곱합으로 먼저: 보통 범위의 s > above(r)면 √ 없이 버린다). 개수만 묻는 `return_length`는 먼 한계가 안을 증명한 노드의 점 수를 더하고 잎에서는 s ≤ below(r) = fl(fl(r·r)·(1 − 8u))도 √ 없이 센다(r²이 보통 수일 때 s < r²이라 RN(√s) ≤ r). 행은 색인으로 정렬해 쓴다.
- **짝**(`query_ball_tree`, `sparse_distance_matrix`, `query_pairs`; `dual.cpp`): 두 트리를 함께 내려가는 이중 트리 순회다. 노드 쌍의 가까운 한계가 T₁보다 크면 치고, 아니면 점이 많은 쪽(같으면 A, 잎은 나누지 않는다)을 나누고, 잎끼리는 모든 짝을 d̂ ≤ r로 견준다. `query_pairs`는 한 트리 안에서 노드와 자기 자신(두 반쪽 각각과 두 반쪽의 짝, 한계 없음)과 서로 다른 노드의 짝을 돌고 (min(i, j), max(i, j))를 낸다. 목록은 태스크마다 모은 뒤 행(`query_ball_tree`는 A의 조각마다 잎 순서의 위치로 세는 정렬, `query_pairs`는 i로 세는 정렬)으로 모으고 행 안을 색인으로 정렬한다.
- **`count_neighbors`**(SPT65): Gray–Moore의 이중 트리 셈이다. 반경을 정렬해 두고 노드 쌍마다 아직 정해지지 않은 반경의 구간 [lo, hi)를 들고 내려간다: 가까운 한계 d̂ₙ이 T₁(r_k) = KdCut(r_k)보다 큰 앞쪽 반경은 빼고(lo를 올림), 먼 한계의 KdCut(b̂_far) ≤ r_k인 뒤쪽 반경은 모든 짝이 안이다(hi를 내림). lo = hi가 되면 짝의 "d̂ ≤ r_k인 첫 k"가 모두 hi라 delta[hi]에 노드 무게의 곱(가중치 없으면 점 수의 곱)을 한 번에 더한다. 잎끼리는 짝마다 d̂를 구해 [lo, hi)에서 이분 탐색한다. 정수 셈은 정확하다. 가중 셈은 float64 합이라 순서가 비트를 정한다: 태스크(SPT64)마다 delta를 따로 두고 태스크 번호 순서로 더한다(SPT25). `cumulative=False`는 delta가 그대로 구간의 셈이고, `True`는 정렬 순서의 누적합이다.
- **트리 대신 전수로 간다**: 결과가 길과 무관하므로(§1.3) `query`는 p = 2이고 5m/6 + log₂(k)/4 + 7 − log₂ n ≥ 0.5이면 입력 순서의 `data`에 GEMM 고르기의 전수를 쓴다(SPT75). n = 10⁵, m = 16, 질의 10⁴개 k = 10이 10스레드 496 → 83 ms다(scipy `workers=-1` 1208 ms).
- **병렬**: 질의점을 크기로만 정한 조각(kNN과 반경 모두 64점, SPT54)으로 나눈다. 질의 하나는 한 태스크가 처음부터 끝까지 계산하므로 순회 순서가 스레드와 무관하다. 반경 결과는 조각마다 버퍼에 모은 뒤 조각 순서로 잇는다(`indptr`은 조각마다의 개수의 누적합). 이중 트리는 깊이 4의 조각 짝이 태스크다(SPT64).
- **전수 길의 자리**: `query`가 길을 고르는 곳은 `src/stardust/spatial/_kdtree.py`의 `_search`(`_brute_pays`) 하나다(SPT59, SPT75).
- **`kneighbors`·`radius_neighbors`의 길**: 거리 이름과 모양으로만 고른다. 가중치 없는 Minkowski 계열(`euclidean`·`cityblock`·`chebyshev`·`minkowski` p ≥ 1)의 작은 n은 `XB`로 만든 KD 트리(SPT73의 문턱), `euclidean`·`sqeuclidean`은 1 ≤ n ≤ 2¹⁶, mA ≥ 64, mB ≥ 2048(n ≥ 8이면 1024)에서 GEMM 고르기의 전수(SPT74), `haversine`은 mB ≥ 128, mA ≥ 16, 8k ≤ mB에서 단위 구의 KD 트리(SPT87; 아래), 그 밖은 직접 전수다(SPT38: `XA` 조각 × `XB` 블록의 `cdist`를 `cdist`의 구동기로 구해 질의마다 (d̂, 색인) 힙이나 반경 목록에 넣는다). 어느 길이든 결과는 같다. 시험용 `path=1, 2, 3`이 직접 전수, GEMM 고르기, 트리를 강제한다.
- **`haversine`의 트리 길**(SPT83–SPT87, `csrc/spatial/sphere.cpp`): `XA`·`XB`의 행마다 cos φ(`cdist`와 같은 `std::cos`)와 3차원 점 P̂ = (ĉφ·ĉλ, ĉφ·ŝλ, ŝφ)(sincos 둘)를 구하고, `XB`의 P̂로 `leafsize` 10의 KD 트리를 만든다. 찾기는 `KdSearch`·반경 찾기의 걸음 그대로이고 노드 한계는 3차원 `euclidean`의 커널이다. 잎의 점은 3차원 제곱합으로 먼저 거르고(SPT86), 남은 점만 `haversine_pair(x, y)`(x는 `XA`의 행)로 d̂를 구해 d̂ < 상한, d̂ ≤ r, (d̂, j)로 판정한다. 그래서 거리는 `cdist`의 비트이고 답은 전수와 같다. T는 kNN에서 힙이 차지 않으면 상한, 차면 맨 위의 거리이고 반경에서는 r이며, 치는 문턱은 cut(T)(SPT84)다.
  - **현과 `haversine`**: P(φ, λ) = (cos φ cos λ, cos φ sin λ, sin φ)이면 참 현 C = ‖P(x) − P(y)‖에 대해 C² = 2 − 2(cos φ₁ cos φ₂ cos Δλ + sin φ₁ sin φ₂) = 4 sin²(Δφ/2) + 4 cos φ₁ cos φ₂ sin²(Δλ/2) = 4h이다(모든 실수 φ, λ의 항등식). 그래서 d = 2·asin(C/2), C = 2 sin(d/2)이고 d는 C에 단조다. 반올림 뒤에는 단조가 아니므로 아래 세 오차를 여유에 넣는다.
  - **보조정리 H** (커널): 정의역 |φ| ≤ fl(π/2), |λ| ≤ 8(SPT85)과 가정 L(libm의 sin·cos·asin은 상대 2u 안, √는 올바른 반올림)에서 |d̂ − d| ≤ 16u·d + 2⁻⁹⁹(가까운 갈래), 43u(먼 갈래)이고 둘 다 2⁻⁴⁶(1 + d) = 128u(1 + d) 안이다.
    - 가까운 갈래(ĥ ≤ 1/2): Δφ̂ = Δφ(1 + δ)이고 |Δφ/2| ≤ π/2라 a·cot a ≤ 1에서 sin(Δφ̂/2)의 상대오차는 u 안이다. Δλ는 TwoSum으로 정확하고(SPT88) 1차 보정의 나머지는 e²/8 + u|e| ≤ 2⁻¹⁰¹이다(|e| ≤ 8u). √h = ‖(sin(Δφ/2), √(cos φ₁ cos φ₂)·sin(Δλ/2))‖이고 h의 항이 모두 음이 아니라 성분마다의 상대오차(3u, 5.5u)와 합·√의 반올림이 그대로 노름의 상대오차가 되어 |fl(√ĥ) − √h| ≤ 9.5u√h + 2⁻¹⁰¹이다. asin의 기울기는 √(1/2) 둘레까지 1.415 이하이고 asin x ≥ x이므로 asin과 그 반올림(2u)까지 |d̂ − d| ≤ 16u·d + 2⁻⁹⁹다. h < 2⁻⁹⁶⁸은 2⁶⁰⁰배의 정확한 비율로 다시 구해 언더플로의 몫이 없다(h ≥ 2⁻⁹⁶⁸에서 언더플로한 항은 상대 2⁻¹⁰⁶ 아래다).
    - 먼 갈래(ĥ > 1/2, d ≥ π/2 − O(u)): √g = ‖(cos(Δφ/2)cos(Δλ/2), sin(S/2) sin(Δλ/2))‖의 성분마다 절대오차가 7.6u 안이다(Δφ̂·Ŝ의 반올림 πu가 기울기 1/2로, libm 2u, 곱 u). 그래서 |fl(√ĝ) − √g| ≤ 12.6u이고 asin 기울기 1.415, fl(π)의 0.55u, 뺄셈의 πu까지 |d̂ − d| ≤ 43u다.
  - **옮기기**: 성분마다 libm 값 하나나 둘과 곱 하나라 상대오차가 (1 + 2u)²(1 + u) − 1 ≤ 5.1u 안이다: |P̂ᵢ − Pᵢ| ≤ 5.1u·|Pᵢ|. φ와 λ는 받은 double 그대로라 입력의 오차가 없고 곱은 언더플로하지 않는다(정의역에서 cos φ ≥ 6×10⁻¹⁷, |cos λ|·|sin λ|는 0이 아니면 10⁻¹⁷보다 크다). 그래서 ‖P̂ − P‖ ≤ 5.1u·‖P‖ = 5.1u다(Arb로 잰 최대 1.27u).
  - **3차원 커널**: 깊이 3의 `euclidean`이라 |b̂ − D| ≤ κD + 2⁻¹⁰⁷⁴, κ = γ₆/2 + u다(§4).
  - **정리 4** (cut, SPT84): d̂(x, y) ≤ T이면 P̂(y)를 담는 모든 상자(점 하나도)의 한계 b̂(P̂(x)에서 상자의 가장 가까운 점까지의 커널 거리)는 cut(T) 이하다. 증명: 보조정리 H의 상수를 4배로 잡아 d ≤ T′ = (T + 2⁻⁴⁴)/(1 − 2⁻⁴⁴) ≤ T(1 + 2⁻⁴³) + 2⁻⁴³이고, fl(T·c) ≥ Tc(1 − u) − 2⁻¹⁰⁷⁵, 2⁻⁴¹ − 2u ≥ 2⁻⁴³, 2⁻⁴²(1 − u) ≥ 2⁻⁴³에서 T′ ≤ t = fl(fl(T·c) + 2⁻⁴²)다. t < 3.14 < π이면 sin이 [0, π/2]에서 늘어나 C = 2 sin(d/2) ≤ 2 sin(t/2)이다. 옮기기의 5.1u를 4배 넘게 2⁻⁴⁸(= 32u)로 잡으면 ‖P̂(x) − P̂(y)‖ ≤ C + 2⁻⁴⁷이고, 상자의 가장 가까운 점은 P̂(y)보다 멀지 않으므로(SPT51) b̂ ≤ (1 + κ)(2 sin(t/2) + 2⁻⁴⁷) + 2⁻¹⁰⁷⁴이다. 계산한 sin이 상대 2⁻⁴⁴ 안이면 fl(2ŝ·c)(1 − u) ≥ 2 sin(t/2)(1 + 2⁻⁴²)이 (1 + κ)배를 덮고 2⁻⁴⁶(1 − u)가 (1 + κ)2⁻⁴⁷ + 2⁻¹⁰⁷⁴을 덮는다. ∎ 그래서 b̂ > cut(T)인 노드와 점의 모든 y는 d̂ > T로 엄격히 멀어 이웃도, 동점(같은 d̂에 색인이 작은 점)도 빠지지 않는다. 제곱합으로는 KdEuclid::above(cut(T))와 견준다(SPT52의 증명).
  - **시험**(`tests/spatial/test_neighbors_haversine.py`): 거의 같은 점(퍼짐 10⁻¹⁵–10⁻⁶)과 중복, 좌표의 마지막 비트만 다른 점, 대척점(10⁻¹⁴–10⁻³ 옮김)과 π 바로 아래의 반경, 극(위도 ±fl(π/2), 경도 −8–8), 날짜 변경선, 거울상의 정확한 동점, 1 ULP 껍질과 그 경계의 반경(계산한 거리와 그 앞뒤 double), 경도 [0, 2π)·[−8, 8]·2π 다른 경도, 정의역 밖(위도 1.6, 경도 8.5·−10⁶, 도 단위), 채움과 빈 쪽, hypothesis 군집(극·날짜 변경선 둘레의 중심, 퍼짐 10⁻¹⁵–1, 반올림한 동점, 중복)에서 전수와 색인·거리·`indptr`의 바이트까지 같고 1·3·10스레드가 같다. Arb 참값(`sphere.npz`)에서는 계약 안의 참 이웃이다. cut의 여유를 지우고 상대 2⁻⁵⁰만큼 줄이면(옮기기 오차도 덮지 않게) 이 열 시험 가운데 여섯이 실패한다(여유가 꼭 필요하다는 확인).

### 3.5 병렬화와 결정성
**같은 입력이면 스레드 수, 경로, Accelerate의 스레드 모드와 상관없이 같은 결과다.** 거리 값은 짝마다 한 커널이고(BLAS를 부르지 않으므로 Accelerate 모드와도 무관하다, SPT26), 이웃의 집합과 순서는 (d̂, 색인)의 전순서로 정해진다. 테스트는 1, 2, 3, 7, 10스레드에서 같은 바이트를 확인한다(§7).

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| `pdist`, `cdist` | 짝의 수로 고르게 나눈 행 구간, 타일 | 짝 하나는 어디서든 같은 커널이다 |
| 데이터마다의 준비(노름, 평균, 비트 묶음) | 행 구간 | 행 하나의 계산이다 |
| GEMM 고르기 | 질의 블록 × `XB` 블록 | 고르기는 답을 바꾸지 않는다(§3.2). 답은 직접 커널과 전순서로 정해진다 |
| 트리 만들기 | 하위 트리 | 노드의 점 집합은 전순서로 정해지고 자리는 미리 센 번호다(§3.3) |
| 질의(kNN, 반경, 개수) | 질의점 조각(크기로만) | 질의 하나는 한 태스크가 다 한다 |
| 짝·이중 트리 | 고정된 깊이(4)의 노드 쌍, 반경이 많은 셈은 더 얕게(SPT64, SPT65) | 결과를 행으로 모아 정렬한다. 가중 셈은 태스크 번호 순서로 더한다 |
| `haversine`의 트리 길 | 행의 준비는 2¹⁴행 조각, 질의는 4–64점 조각(SPT87) | 질의 하나는 한 태스크가 다 하고, 답은 `haversine` 커널과 (d̂, 색인)으로 정해진다(정리 4) |

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

  합한 속도는 4–6스레드에서 1스레드의 약 2.1–2.3배(단일 스레드 모드)로 거의 차고, 기본 모드에서는 호출 하나가 이미 AMX 둘을 쓴다(n = 512에서 1스레드 769). 그래서 GEMM 고르기는 태스크를 늘리는 것보다 블록을 크게 하는 쪽이 낫다. 태스크 수와 블록 크기는 SPT42가 정했다(결과와 무관하다).

### 3.6 메모리
- 사용자가 모양으로 요청한 출력(`pdist`의 n(n−1)/2, `cdist`의 mA·mB, `kneighbors`의 mA·k)은 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2).
- 축약과 이웃 찾기의 작업 메모리는 입력과 무관한 상한이 있다(SPT2에서 만든 대로): 직접 길은 태스크마다 거리 블록 2¹⁶개 double(512 KB)과 `cdist` 구동기가 싸는 `XB` 블록(2¹⁸개 double 이하), GEMM 길은 태스크마다 옮긴 행(조각과 블록 각각 2¹⁹개 double 이하), G(256 × 2048 = 4 MB 이하), L̃의 줄이다. 힙은 태스크마다 2¹⁶자리(1 MB) 이하이고, k가 그보다 크면 조각이 한 행이라 k자리다. `XB`가 구간으로 나뉘면 구간마다의 목록이 2²²자리 이하다. 10스레드에서 약 100 MB 아래다. GEMM 길은 `XB`의 행마다 두 double(n̂ᵧ, Rᵧ)을 더 쥔다. 전체 (mA, mB) 행렬을 만들지 않는다(SPT1).
- 반경 결과는 찾은 짝의 수에 비례한다(`indices` 8바이트 + `distances` 8바이트). 태스크마다 모은 뒤 제자리에 복사하므로 결과 크기의 두 배가 잠깐 든다(SPT48). `query_ball_tree`·`sparse_distance_matrix`는 짝마다 모으는 동안 24바이트, 행으로 모을 때 16바이트, 결과 16바이트라 결과의 약 3.5배, `query_pairs`는 모으는 16바이트와 결과 16바이트다. 찾은 짝이 물리 메모리를 넘을 수(SPT68)가 되면 찾기를 멈추고 `MemoryError`다(`radius_neighbors`는 전수·GEMM·트리·`haversine` 트리의 어느 길이든 `radius_neighbors(): the result is larger than this machine's physical memory`; 짝 하나를 40바이트로 센다). 개수만 필요하면 `return_length`, `count_neighbors`가 색인을 모으지 않는다(`count_neighbors`는 태스크 × (R + 1)칸, 2²² 이하).
- `KDTree`: 잎 순서의 사본 n·m·8 + 색인 n·8 + 노드(상자 2m·8 + 24바이트) × 노드 수(잎의 점이 `leafsize`/2개 이상 `leafsize`개 이하라 2n/`leafsize` − 1 이상 4n/`leafsize` 미만; n = 1000, `leafsize` 10에서 255), `data`를 읽거나 전수 길(SPT75)을 탄 뒤에는 입력 순서의 사본 n·m·8(§3.3, SPT76). GEMM 길은 호출마다 옮긴 `XB`의 사본(2²⁷개 double 이하일 때, SPT71)을 더 쥔다.
- `haversine`의 트리 길: 호출마다 행의 cos φ와 3차원 점((mA + mB)·4 double), `XB`의 트리(잎 순서 사본 3·mB, 색인 mB, 노드)와 잎 순서의 (위도, 경도, cos φ) 3·mB double이다. 10⁵점에서 약 12 MB다. 반경 결과는 반경 질의와 같다(SPT48, SPT68).

---

## 4. 정확도 계약
참값은 Arb다. 입력은 float64라 차이와 제곱합을 Arb에서 정확히 구할 수 있고, 참 거리를 올바르게 반올림한 값 RN(d)과 견준다. ULP는 `bench`의 `ulp_distance`다. 계약 단위(오차 ÷ 한계)로 적고 경계는 1이다(sparse.md §4와 같은 꼴). 판정은 `tests/spatial/refdata.py`의 `bound()`·`units()` 하나이고 테스트(`tests/spatial/test_reference.py`)와 정확도 체계(`bench/accuracy/spatial.py`, 표 `bench/spatial/accuracy.md`)가 같이 쓴다.

- **거리** (SPT1에서 잼, SPT32. m은 합의 깊이: 타일 순서에서 m = min(n, 32) + ⌈n/32⌉ − 1, §3.1·SPT26. u = 2⁻⁵³, c_pow = γ₂ = 2u/(1 − 2u)(SPT80; SPT1은 2u). 모든 한계에 2⁻¹⁰⁷⁴을 더한다. 잰 값은 `distances.npz` 645짝, `counting.npz` 80짝, `haversine.npz` 32짝의 최대다)
  | 거리 | 계약 | 잰 최대 ULP | 계약 대비 | 메모 |
  |---|---|---:|---:|---|
  | `chebyshev`, 가중 | RN(d)와 같다(0 ULP) | 0 | 1 | 최대 \|fl(zᵢ)\|가 올바른 반올림이다(계약 대비 1은 참값이 두 double의 한가운데인 짝) |
  | `euclidean`, 가중 | \|d̂ − d\| ≤ (γₘ₊₃/2 + u)·d | 2, 2 | 0.46, 0.43 | 항이 모두 음이 아니라 상쇄가 없다 |
  | `seuclidean` | (γₘ₊₄/2 + u)·d | 2 | 0.45 | 1/Vᵢ의 반올림까지 |
  | `sqeuclidean` | γₘ₊₂·s(가중 γₘ₊₃) | 3, 4 | 0.68, 0.46 | |
  | `cityblock`, 가중 | γₘ₊₂·s | 3, 3 | 0.36, 0.35 | |
  | `minkowski(p)` | (4u + (γₘ + 2c_pow)/p + c_pow)·d | 6 (p = 0.5), 3, 1, 2; 가중 p = 3에서 2 | 0.2, 0.17, 0.21, 0.4; 0.22 | SPT30의 근과 신중한 경로까지. libm의 `pow`는 SPT0에서 정상 범위 0.504 ULP 이하였다. p = 3의 항은 (a·a)·a다(SPT80) |
  | `canberra`, 가중 | γₘ₊₃·d | 3, 2 | 0.36, 0.46 | 항마다 세 번의 반올림 |
  | `braycurtis`, 가중 | γ₂ₘ₊₃·d | 5, 7 | 0.35, 0.46 | 두 합과 나눗셈. 분모의 \|xᵢ + yᵢ\|는 한 번 반올림이라 상대로 적힌다 |
  | `cosine`, 가중 | u(3√d + (2m + 8)d) + (m + 4)²u² | d ≥ 10⁻⁸에서 15, 22 | 0.42, 0.25 | SPT29의 식. 참값이 0(평행)이거나 아주 작으면 절대오차 2.6×10⁻²⁵ 이하라 ULP는 크다 |
  | `correlation`, 가중 | `cosine`의 한계 + 2(ρₓ² + ρᵧ²), ρ = γ₂ₘ₊₂·max\|xᵢ\|·√(Σw)/‖x_c‖_w | — | 0.39, 0.21 | 평균의 반올림이 행을 1 방향으로 옮긴 몫이고 2차다. ρ가 큰 행(1e15 + 정수처럼 중심이 퍼짐보다 훨씬 먼 행)에서는 ULP가 크다(scipy는 같은 곳에서 상대 5×10⁻⁵) |
  | `mahalanobis` | min(E/d, √E) + u·d, E = γ₂ₘ₊₃·\|z\|ᵀ\|VI\|\|z\| | 2 | 0.29 | VI가 나쁜 조건이면 상대오차가 크다: 계약은 절댓값의 이차식 기준이다(n ≤ 16의 양정치 VI에서 잼) |
  | `hamming`, `jaccard` | 정확한 개수의 나눗셈 한 번(올바른 반올림); 가중이면 γ₂ₘ₊₁·d | 0, 0; 가중 2, 2 | 0.88, 0.93; 0.12, 0.12 | bool·int64·float64가 같은 개수 |
  | `haversine` | 20u·d(SPT31) | 1 | 0.071 | 대척점 넷 포함, SPT88 뒤에 다시 쟀다. 증명한 한계는 보조정리 H(§3.4: 16u·d + 2⁻⁹⁹, 먼 갈래 43u)이고, 적대적 집합 열(날짜 변경선, 2π 다른 경도, 극, 대척점, 10⁻³⁰⁰ 둘레)에서 Arb와 견준 최대는 계약의 0.20이다 |
- 비교 패키지(같은 짝에서 그 자리에서; `bench/spatial/accuracy.md`): scipy 1.18.1은 넘침·언더플로(1e160, 1e-160, 비정상 수의 짝)에서 inf·0·NaN을 내 `euclidean`·`seuclidean`·`minkowski`·`mahalanobis`·`cosine`·`correlation`의 계약을 넘고(scipy의 `minkowski` p = 1.5는 `edge/subnormal`에서 상대 1), 정상 범위의 짝에서는 계약 안이다. sklearn 1.9.1의 전개식(`euclidean_distances`)과 `cosine_distances`는 계약을 넘는다(1 − cos의 상쇄, 넘침). R 4.6.2의 `stats::dist`는 `euclidean`·`minkowski`가 넘침에서 inf다.
- **이웃 찾기** (세 겹)
  1. **같은 커널의 전수와 정확히 같다**: `kneighbors`(`haversine`의 트리 길 포함), `query`, `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, `sparse_distance_matrix`의 결과(색인, 순서, 거리의 비트, 개수)는 같은 직접 커널로 모든 짝을 계산해 (d̂, 색인)으로 고른 것과 같다. 길(트리, GEMM 고르기, 전수)마다, 스레드 수마다 같다. 이것이 SPT2의 여유(§3.2)와 가지치기의 여유(§3.4), `haversine` 트리 길의 여유(정리 4)를 시험한다.
  2. **Arb로 정렬한 전수와 정확히 같다**(거리가 올바르게 반올림되는 데이터에서): 정수 격자(차이의 제곱합이 2⁵³ 아래: 3차원 좌표 \|x\| < 2²⁴), 2진 유리수 격자, 같은 점이 많은 데이터에서는 차이와 제곱합이 정확하고 √가 올바른 반올림이라 d̂ = RN(d)다. 그런 데이터에서 결과는 Arb로 구한 (RN(d), 색인)의 정렬과 정확히 같아야 한다. 격자는 참 거리가 같은 짝(동점)이 아주 많아 동점 순서의 약속을 직접 시험한다.
  3. **일반 데이터에서 참 kNN이다**(거리 계약 η 안에서): 돌려준 k번째 거리 d̂₍ₖ₎에 대해, 빠진 모든 점의 참 거리는 d̂₍ₖ₎(1 − η) 이상이고 돌려준 모든 점의 참 거리는 d̂₍ₖ₎(1 + η) 이하다(Arb의 참 거리로 확인). 일반 데이터에서는 참 거리가 1 ULP 안에서 다른 두 점의 d̂ 순서가 참 순서와 다를 수 있다. 1번과 3번이 그 경우의 약속이다: 순서는 언제나 돌려준 값의 순서이고, 그 값은 거리 계약 안이다. `haversine`은 단위 구 위의 점 집합 여섯(`sphere.npz`)에서 3번을 η = 20u로 확인한다(k = 5 정확도 스윕에서 계약의 0.096 이하, 반경도 테스트가 확인한다).
- scipy·sklearn·R과의 차이는 계약이 아니라 정확도 표의 비교 패키지 열로 보인다. 이웃은 Arb의 kNN과 같은 집합인 비율을 비교 패키지마다 적는다(sd는 2번의 데이터에서 100%가 계약이다).

---

## 5. 성능
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

**SPT2의 이웃 찾기**(2026-11-22; ms, `bench/speed/cases/spatial.py`를 `--smoke`로 한 번씩 부른 값이라 잡음이 크다. 기준선이 아니다. sklearn은 BLAS의 스레드를 따른다(1스레드 열은 BLAS 1스레드). 표준정규, 부하 평균 약 1.2):

| 일 | sd 1 | sd 10 | sklearn 1 | sklearn 10 |
|---|---:|---:|---:|---:|
| `kneighbors` k = 1, 4000², n = 3 (직접) | 18.5 | 3.25 | 29.5 (`argmin_min`) | 5.04 |
| `kneighbors` k = 1, 4000², n = 64 (GEMM) | 27.4 | 4.91 | 29.2 (`argmin_min`) | 8.60 |
| `kneighbors` k = 1, 4000², n = 512 (GEMM) | 58.3 | 23.0 | 53.4 (`argmin_min`) | 40.0 |
| `kneighbors` k = 10, 4000², n = 3 / 64 / 512 | 25.2 / 34.3 / 98.9 | 2.97 / 5.11 / 23.1 | 29.5 / 33.9 / 56.1 (brute) | 8.08 / 18.9 / 34.2 |
| `radius_neighbors` 4000², n = 3 / 64 (행마다 약 10개) | 17.2 / 28.4 | 2.38 / 5.11 | 19.9 / 23.7 (brute) | 7.31 / 10.2 |
| `kneighbors` k = 5, 질의 20개 × 10⁶점, n = 3 (구간으로 나눔) | 26.9 | 3.61 | 47.5 (brute) | 6.36 |
| `kneighbors` k = 5 `cosine`, 2000², n = 64 (직접) | 29.5 | 4.00 | 40.0 (brute) | 32.9 |

- 10스레드에서는 모든 사례가 sklearn보다 빠르고 §5 목표(`argmin_min`의 1.5배 안)를 넘는다. 1스레드에서 n = 512의 k = 10은 sklearn의 1.8배 느리다: k개를 다시 계산하는 짝 커널(n = 512에서 짝 하나가 512항)과 첫 후보의 규칙이 시간을 더한다. SPT6에서 다시 본다.
- 결과는 sklearn과 견줬다: `radius_neighbors`의 모든 행(sklearn의 행을 색인으로 정렬)과 k = 10의 색인·거리가 상대 10⁻⁶ 안에서 같았다.

**SPT3의 KD 트리**(2026-11-22; ms, `bench/speed/cases/spatial.py`의 `tree` 사례 여섯을 `--smoke`로 한 번씩 부른 값, 부하 평균 1.4–2. [0, 1)ᵐ 균등, 질의 10⁴개. 기준선이 아닌 개발 중의 확인이고, 결과는 scipy와 견줘 모든 색인과 거리가 맞았다):

| 일 | sd 1스레드 | sd 10스레드 | scipy | scipy `workers=-1` | sklearn |
|---|---:|---:|---:|---:|---:|
| `KDTree` 만들기 10⁵점, m = 3 | 20.3 | 8.7 | 18.7 | — | 16.2 |
| `KDTree` 만들기 10⁶점, m = 3 | 235 | 79.3 | 260 | — | 240 |
| `query` 10⁶점, m = 3, k = 1 | 12.1 | 1.54 | 13.7 | 3.15 | 26.3 |
| `query` 10⁶점, m = 3, k = 10 | 26.4 | 3.25 | 31.8 | 5.36 | 53.1 |
| `query` 10⁵점, m = 16, k = 1 | 2289 | 243 | 5510 | 517 | 4337 |
| `query` 10⁵점, m = 16, k = 10 | 5022 | 546 | 11350 | 1166 | 5969 |

- 10스레드에서 모든 사례가 scipy `workers=-1`보다 빠르다(§5의 트리 목표). 1스레드 만들기는 scipy와 비슷하다(10⁵점에서 1.09배 느리고 10⁶점에서 1.1배 빠르다). 10스레드 만들기가 3배에 머무는 까닭은 §3.3이다.
- 처음 구현(점마다 √, 순열을 통한 훑기)은 1스레드 `query` k = 10이 39.5 ms, 만들기 10⁶점이 335 ms였다. 제곱합 거르기(SPT52)와 행 옮기기(SPT53)로 위의 값이 되었다(결과는 같다).
- m = 16은 트리가 전수보다 느린 영역이다(§3.4: 같은 일의 GEMM 고르기는 수십 ms). SPT21의 전수 길(SPT59의 자리)은 SPT6에서 더했다(SPT75).

**SPT4의 반경·짝·개수**(2026-11-22; ms, `bench/speed/cases/spatial.py`의 SPT4 사례 여덟을 `--smoke`로 한 번씩 부른 값, 부하 평균 2.4–2.5. 기준선이 아닌 개발 중의 값이다. [0, 1)³ 균등. `query_ball_point`는 10⁶점에 질의 10⁴개(질의마다 약 10점), 나머지는 10⁵점 두 벌(점마다 이웃 약 10, `query_pairs`는 약 4.8×10⁵짝), 20개 반경은 0.001–0.05)

| 일 | sd 1스레드 | sd 10스레드 | scipy | scipy `workers=-1` |
|---|---:|---:|---:|---:|
| `query_ball_point` | 19.9 | 3.1 | 35.5 | 7.1 |
| `query_ball_point` `return_length` | 19.1 | 2.8 | 30.8 | 5.9 |
| `query_ball_tree` | 72.0 | 25.3 | 138.5 | — |
| `query_pairs` | 39.6 | 14.6 | 33.8 (`ndarray`), 85.4 (`set`, 기본) | — |
| `count_neighbors` r 하나 | 72.6 | 11.2 | 91.8 | — |
| `count_neighbors` r 20개 | 255 | 35.0 | 349 | — |
| `count_neighbors` r 20개, 가중 | 279 | 37.0 | 378 | — |
| `sparse_distance_matrix` | 74.2 | 26.3 | 84.3 (`coo_array`), 373 (`dok_matrix`, 기본) | — |

- 10스레드에서 모든 사례가 scipy(`workers=-1`이 있으면 그것)보다 빠르다. 1스레드도 `query_pairs`(scipy의 `ndarray` 꼴보다 1.17배 느리다; 기본 `set`보다는 2.2배 빠르다)를 빼고 scipy보다 빠르다. 같은 일을 다섯 번 재 가장 좋은 값(10⁵점 두 벌)은 1스레드 `query_pairs` 35.8, `count_neighbors` 70.4, `query_ball_tree` 67.8 ms, 10스레드 9.5, 10.1, 20.1 ms였다. 목록을 모으는 `query_ball_tree`·`query_pairs`의 10스레드가 3–4배에 머무는 것은 모으기(행으로 세는 정렬, 메모리 할당)가 순회보다 덜 나뉘기 때문이다(SPT6의 후보). 잎 쌍의 점마다 상자 검사는 재서 느려 넣지 않았다(SPT69).
- 결과는 scipy와 견줬다: 행의 색인(scipy의 행을 정렬), `query_pairs`의 짝(scipy의 `ndarray`를 정렬), 셈은 정확히 같고, 가중 셈은 상대 10⁻¹² 안, `sparse_distance_matrix`의 거리는 상대 10⁻¹⁴ 안이었다.

**SPT6의 성능 점검**(2026-11-23; ms, 다섯 번 가운데 가장 좋은 값(비교 패키지는 세 번), 앞(SPT4 끝)과 뒤를 같은 스크립트로 잼. 기계를 다른 세션과 나눠 써 ±5–10%의 잡음이 있다. sklearn은 기본 설정(BLAS·OpenMP 다중 스레드), scipy는 1스레드. 무작위 표준정규(거리, 한 번에 끝나는 이웃)와 [0, 1)ᵐ 균등(트리))

| 일 | 앞 1스레드 | 앞 10스레드 | 뒤 1스레드 | 뒤 10스레드 | 비교 | 바꾼 것 |
|---|---:|---:|---:|---:|---:|---|
| `cdist` minkowski p = 3, 1000², n = 64 | 339 | 37.7 | 39.4 | 4.62 | scipy 310 | SPT80 |
| `cdist` minkowski p = 1.5, 1000², n = 64 | 344 | 38.0 | 343 | 37.3 | scipy 313 | — |
| `cdist` haversine 2000² | 151 | 17.3 | 102 | 11.4 | sklearn 44.9 | SPT81 |
| `cdist` mahalanobis 1000², n = 16 / 64 | 80.5 / 1163 | 9.30 / 132 | 20.9 / 292 | 2.69 / 31.9 | scipy 78.6 / 1351 | SPT82 |
| `kneighbors` k = 1 / 10, 4000², n = 3 | 18.6 / 24.1 | 2.40 / 2.95 | 2.02 / 6.22 | 0.79 / 1.18 | sklearn 5.96 / 9.17 | SPT73 |
| `kneighbors` k = 1 / 10, 4000², n = 64 | 24.6 / 31.7 | 4.54 / 5.41 | 16.9 / 23.1 | 3.60 / 4.27 | sklearn 11.5 / 15.4 | SPT71, SPT72 |
| `kneighbors` k = 1 / 10, 4000², n = 512 | 53.0 / 71.3 | 22.6 / 23.6 | 36.1 / 54.8 | 19.8 / 20.9 | sklearn 47.3 / 46.4 | SPT71, SPT72 |
| `kneighbors` k = 10, 200 × 20000, n = 32, 먼 점부터 정렬 | 11.9 | 1.53 | 7.40 | 1.14 | 직접 길 2.05(10스레드) | SPT72 |
| `radius_neighbors` 4000², n = 3 / 64 | 16.3 / 25.0 | 2.22 / 4.87 | 4.35 / 13.7 | 1.18 / 3.58 | — | SPT73, SPT71 |
| `kneighbors` k = 5, 20 × 10⁶, n = 3 / `cosine` 2000², n = 64 | 24.7 / 27.5 | 3.49 / 3.79 | 26.3 / 27.3 | 3.16 / 3.73 | — | — |
| `KDTree` 만들기 10⁵ / 10⁶점, m = 3 | 19.1 / 226 | 7.76 / 76.0 | 17.3 / 206 | 4.34 / 40.6 | scipy 18.3 / 231 | SPT79 |
| `query` 10⁶점, m = 3, k = 1 / 10 | 10.6 / 23.9 | 1.29 / 3.07 | 10.5 / 22.9 | 1.29 / 2.83 | scipy `workers=-1` 2.16 / 4.12 | — |
| `query` 10⁵점, m = 16, k = 1 / 10 | 2029 / 4568 | 230 / 496 | 410 / 449 | 75.3 / 77.5 | scipy `workers=-1` 434 / 1169 | SPT75 |
| `query_ball_point` / `return_length` | 18.9 / 19.3 | 2.47 / 2.35 | 19.6 / 19.2 | 2.65 / 2.21 | scipy 34.4 / 29.4 | — |
| `query_ball_tree` / `sparse_distance_matrix` | 73.4 / 71.4 | 23.6 / 22.1 | 70.5 / 71.3 | 18.8 / 22.9 | scipy 121 / 80.8 | — |
| `query_pairs` | 37.9 | 12.8 | 41.9 | 13.1 | scipy `ndarray` 33.0 | SPT77 |
| `count_neighbors` r 하나 / 20개 / 20개 가중 | 73.2 / 254 / 275 | 9.37 / 31.6 / 34.8 | 69.6 / 244 / 260 | 9.44 / 30.5 / 32.7 | scipy 93.6 / 347 / — | — |
| `cdist` euclidean 1000² n = 3 / 64, 4000² n = 3 / 64 | 0.70 / 4.93 / 11.0 / 80.2 | 0.16 / 0.66 / 1.40 / 8.63 | 0.68 / 4.90 / 11.1 / 79.5 | 0.15 / 0.66 / 1.94 / 8.54 | scipy 0.80 / 10.4 / 12.1 / 167 | — |
| `pdist` euclidean n = 3 / 64, cosine n = 64 | 1.42 / 10.0 / 10.0 | 0.58 / 1.21 / 1.30 | 1.42 / 9.96 / 9.99 | 0.58 / 1.20 / 1.30 | scipy 1.58 / 20.7 / 45.9 | — |
| 작은 호출(µs, 고리로 잼): `euclidean(u, v)` n = 3 / `cdist` 10² / `query` 점 하나 k = 3 / `KDTree` 100점 | — | — | 0.82 / 1.16 / 3.00 / 5.01 | — | scipy 2.07 / 0.96 / 5.70 / 8.04 | `metric` 검사의 빠른 길 |

- 거리 함수는 `minkowski`의 p = 3이 아닌 p(`pow`가 시간을 정해 scipy보다 약 1.1배 느리다: 짝마다의 뉴턴 걸음과 같은 `pow` 경로), `haversine`(sklearn보다 약 2.3배 느리다; 짝마다 libm의 sincos 둘과 asin, 묶음 libm은 원소의 비트가 자리에 따라 같다는 보장이 없어 쓰지 않는다)을 빼고 1스레드에서 scipy 이상이다. 작은 `cdist`(10²)는 scipy보다 0.2 µs 느리다(검사의 Python 비용).
- 1스레드 `query_pairs`는 scipy의 `ndarray` 꼴보다 1.27배 느리다(순회가 시간의 약 80%: 노드 쌍마다 새로 구하는 한계(SPT63)와 정렬된 결과의 비용). 목록을 모으는 `query_ball_tree`·`query_pairs`의 10스레드가 3.2–3.7배에 머무는 것은 순회 태스크 안의 목록 쌓기(메모리)로 보인다: 모으기를 병렬로 한 뒤(SPT77) 작업 스레드는 85%가 태스크 안이다. 둘 다 결과를 바꾸지 않는 다음 후보다.
- `kneighbors` k = 10 n = 512의 1스레드는 이제 sklearn(OpenMP·BLAS 다중 스레드)의 1.18배다. `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`로 둘 다 1스레드면 앞은 87.9 대 70.6 ms였다. 남은 시간의 반쯤은 GEMM이다.
- **BallTree(SPT5, SPT8)의 다시 보기**: KD 트리가 칠 수 없는 거리의 전수는 `haversine` 10⁴ × 10⁵ k = 5가 sd 1스레드 27.5초, 10스레드 3.06초인데 sklearn의 `BallTree`(만들기 + 질의)는 97 ms다(2차원이라 공 경계가 아주 잘 친다). `canberra`(m = 8)는 sd 10스레드 444 ms 대 5400 ms, `braycurtis`는 218 대 1399 ms로 전수가 이긴다. 지리 자료의 kNN에서 30배 넘게 지므로 SPT8의 "2배 넘게 이길 크기가 흔하면 SPT5를 연다"에 해당한다. 범위의 결정이라 이 마일스톤에서 넣지 않고 사용자에게 물었고, 사용자는 단위 구 위의 3차원 점에 KD 트리와 현(chord) 거리의 한계를 골랐다(SPT83; 아래 표).
**`haversine` 이웃의 트리 길**(SPT83–SPT88, 2026-11-23; ms, 다섯 번 가운데 가장 좋은 값(sklearn과 전수는 두세 번), 단위 구 위의 균등한 점, 부하 평균 2.3–2.7. 앞은 SPT6 끝의 전수(10⁴ × 10⁵은 질의 10³개의 10배), 뒤는 자동 길(트리). sklearn `BallTree(metric='haversine')`는 만들기와 질의를 함께 잰다(sd도 호출마다 트리를 만든다; `BallTree`는 1스레드))

| 일 | 앞 1스레드 | 앞 10스레드 | 뒤 1스레드 | 뒤 10스레드 | sklearn `BallTree` | 바꾼 것 |
|---|---:|---:|---:|---:|---:|---|
| `kneighbors` k = 5, 10⁴ × 10⁵ | 26100 | 3610 | 27.9 | 6.91 | 98.2 | SPT83–SPT87 |
| `radius_neighbors` r = 0.01(행마다 약 2.5개), 10⁴ × 10⁵ | 25400 | 3580 | 23.3 | 6.70 | 47.4 (`query_radius`) | SPT83–SPT87 |
| `kneighbors` k = 5, 2000² | 103 | 14.3 | 1.34 | 0.41 | 6.49 | SPT83–SPT87 |
| `cdist` haversine 2000² | 102 | 11.4 | 99.9 | 11.5 | `haversine_distances` 43.9 | SPT88 |

- 10⁴ × 10⁵ k = 5는 10스레드 523배, 1스레드 940배 빨라졌고 sklearn `BallTree`보다 10스레드 14배, 1스레드 3.5배 빠르다. 10스레드가 1스레드의 4배에 머무는 것은 트리 만들기와 행의 준비(10⁵행의 sincos 둘과 cos)가 질의보다 크기 때문이다.
- 길의 문턱(SPT87)은 mB 64–4096, mA 1–2048, k 1·8·32의 격자를 1·10스레드로 잰 갈림점이다: mA ≥ 16, mB ≥ 128, 8k ≤ mB에서 트리가 1.15–205배 빠르고(4096 × 2048 k = 1의 1스레드가 205배), mA ≤ 4는 0.2–1.2배다. 큰 반경(r ≥ 2)에서는 거의 모든 점이 결과라 두 길이 같다(4096 × 2048 r = 2의 10스레드 37.7 대 34.7 ms).
- 결과는 sklearn `BallTree`와 견줬다(`--smoke`의 속도 사례 두 개: 색인과 거리가 상대 10⁻⁶ 안).

- 정확도 계약은 SPT80의 c_pow(2u → γ₂, 4u²의 차이) 하나만 바뀌었다. 골든은 p = 3의 `minkowski` 다섯(`pdist`·`cdist`·가중 `cdist`의 minkowski big, `query`·`query_ball_point`의 big p = 3)과 `haversine` 셋(`pdist haversine`, `haversine batch`, `kneighbors haversine`)이 1 ULP씩 바뀌었다(이웃의 집합과 순서는 같다). 나머지 419케이스(새 길과 GEMM 고르기의 바뀜 포함)는 같다.
- 코드 리뷰(SPT6)에서 고친 것: `_KDTree`의 생성이 만들기 중의 `bad_alloc`에서 트리를 새게 했다(`new`/`delete` → 값). `kneighbors`·`radius_neighbors`의 일의 어림 mA·mB·비용이 int64를 넘을 수 있었다(출력이 mA·mB를 묶지 않는다; 넘으면 음수가 되어 GIL을 쥔 채 직렬로 돌았다 → 포화). `pair_scalar`를 키운 뒤 인라인이 풀려 트리의 짝마다 함수 호출이 생긴 것을 재서 찾아 짧은 길만 인라인했다. GIL을 놓은 동안 Python 객체를 만드는 곳은 없었고(`too_large`와 결과 배열은 범위 밖), 중첩된 `Unlocked`는 `GilRelease`가 이미 놓은 GIL을 건드리지 않는다.

- **목표**(SPT6에서 확인했다: `pdist`·`cdist`는 p ≠ 3의 `minkowski`(1.1배 느림)와 sklearn보다 느린 `haversine`을 빼고 1스레드에서 scipy 이상, 10스레드는 4000² n = 64에서 9.3배; `kneighbors` k = 1은 n = 64·512에서 sklearn `argmin_min`(SPT2의 1스레드 29.2·53.4 ms)보다 빠르다(16.9·36.1 ms); 트리의 만들기와 질의는 모든 사례에서 scipy `workers=-1`보다 빠르다; `haversine`의 kNN은 sklearn `BallTree`보다 10스레드 14배 빠르다(SPT83))
  - `pdist`·`cdist`: 1스레드에서 scipy 이상(같은 직접 계산에 NEON), 10스레드에서 거의 스레드 수만큼. n이 큰 `cdist`는 sklearn의 GEMM만큼 빠를 수 없다: 모든 거리를 직접 계산하므로 n·mA·mB의 뺄셈과 FMA가 하한이다(n = 512, 4000²에서 짝·항 8.2×10⁹개; SPT0의 타일 커널은 1스레드에서 짝·항마다 약 0.07 ns(§3.1의 n = 512)라 1스레드 약 0.55초, 10스레드 약 60 ms를 어림한다). 이것은 SPT2가 정한 정확도의 값이다.
  - `kneighbors`(k = 1, n ≥ n₀): sklearn `pairwise_distances_argmin_min`의 1.5배 안(같은 GEMM에 다시 계산과 중심 빼기를 더한다).
  - 트리: 만들기와 질의 모두 scipy `workers=-1`보다 빠르게(sd는 기본으로 10스레드다).
- **나중에 볼 것**: GEMM 고르기를 다른 거리로 넓히기(§3.2), 작은 m(2, 3)의 트리 잎에서 짝 방향 SIMD, 질의를 공간 순서로 정렬해 캐시를 살리기(결과는 원래 순서로 돌려놓는다).

---

## 6. 층과 파일
- **층**: `spatial/`은 `array/`, `accelerate/`(`blas.hpp`의 `gemm`: §3.2의 고르기), `parallel/`, `base/`(`sincos.hpp`) 위에 있다. 합의 순서가 이 영역의 타일 순서라(SPT22, SPT26) `linalg/`의 합과 노름에 기대지 않는다. 다른 영역은 `spatial/`을 모른다. Python 층의 `stardust.spatial`은 `stardust.sparse`(`.tocsr()`), `stardust.linalg`(`inv`), 배열의 `var`를 쓴다.
- `bench/areas.py`에 `spatial`이 있다(영역 열셋째, `plot` 앞). ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표에 한 줄씩 있다(SPT0).
- `csrc/spatial/`
  - `metric.hpp`(SPT1): 거리마다의 연산(Op)과 두 커널 `tile<Op, R, W>`·`pair_scalar<Op, W>`, 신중한 경로, `minkowski`·`mahalanobis`·`haversine`의 짝 함수, 데이터마다의 준비(정규화, 평균, 비트 묶음). SPT0의 시제품 `pair_kernel.hpp`는 `tools/spatial_probe_pair_kernel.hpp`로 옮겼다(SPT36)
  - `distance.cpp`(SPT1): `_pdist`, `_cdist`, `_paired`(배치 거리 함수), `_squareform_tomatrix`, `_squareform_tovector`와 구동기(타일, 짝 하나씩, 태스크 나누기), `spatial.hpp`가 `bind_*`를 알린다
  - `neighbors.hpp`, `neighbors.cpp`(SPT2): `CdistBlocks`(`distance.cpp`가 구현하는 블록마다의 `cdist`, SPT38)의 선언, `_kneighbors`·`_radius_neighbors`, (d̂, 색인) 최대 힙, 나누기(SPT42), §3.2의 GEMM 고르기(중심 빼기, 여유, 블록 순회). 초안의 `hybrid.hpp`·`heap.hpp`·`brute.cpp` 셋을 한 파일로 했다
  - `kdtree.hpp`(SPT3): 트리의 꼴(`KdTree`, `KdNode`), 질의점과 점의 거리(`KdEuclid`, `KdCity`, `KdCheb`, `KdMink`: §3.1의 커널), 가지치기의 문턱(`KdCut`), kNN 찾기(`KdSearch`)와 그 증명의 주석
  - `kdtree.cpp`(SPT3): 만들기(§3.3), `_KDTree` 바인딩(만들기, 속성, `query`, 테스트용 `_state`)
  - `query.cpp`(SPT4): `_ball_point`(`query_ball_point`의 목록과 개수, 질의 64점 태스크)
  - `dual.cpp`(SPT4): 이중 트리 순회와 조각 짝의 태스크(SPT64), `_ball_tree`(`query_ball_tree`, `sparse_distance_matrix`), `_pairs`, `_count`(SPT65). 두 상자 사이의 한계 `KdPair`와 공용 판정은 `kdtree.hpp`에 있다(SPT70)
  - `sphere.cpp`(SPT83–SPT89): `haversine` 이웃의 트리 길 `_sphere_kneighbors`·`_sphere_radius`(행의 준비와 정의역 검사, 단위 구 위 점의 트리, cut(T)와 정리 4의 주석, kNN·반경 찾기)
  - `spatial.hpp`: 파일마다의 `bind_*`(`bind_distance`, `bind_kdtree`, `bind_neighbors`, `bind_query`, `bind_dual`, `bind_sphere`)
- 바인딩 `bind/spatial_bind.cpp`(SPT1): `_core.spatial` 하위 모듈.
- `src/stardust/spatial/` (하위 패키지 없음): `__init__.py`(공개 이름), `_common.py`(입력 정리, `metric`과 인자의 검사, 메시지; SPT1), `_distance.py`(`pdist cdist squareform`과 거리 함수, 기본 `V`·`VI`; SPT1), `_neighbors.py`(`kneighbors radius_neighbors`와 길 고르기: 트리 SPT73, `haversine`의 트리 SPT87; SPT2), `_kdtree.py`(`KDTree`; SPT3, 반경·짝·개수는 SPT4), `_radius.py`(`RadiusResult`; SPT2, SPT44). `QueryResult`는 `stardust/_results.py`에 있다.

---

## 7. 참값과 검증
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다. 긴 참값 생성은 프로세스 10개까지, 메모리를 지켜보며(RSS가 평평한지) 돈다.

### 7.1 시험 데이터 (`tests/spatial/points.py`)
- 무작위: 표준정규, 균등, 섞인 군집(가우스 혼합), 저차원 다양체(높은 m에 낮은 내재 차원), m = 0, 1, 2, 3, 8, 16, 64, 512, 2048, 2049, 4096(합의 길이 경계: 16, 512, 1024, 2048).
- **적대적**(SPT2의 여유): 원점에서 먼 작은 군집(1e8 + [0, 1]ᵐ, 1e15 + 정수), 거리가 1 ULP씩 다른 점들(질의 둘레의 껍질), 같은 점의 무더기, k번째 경계에 동점이 많은 데이터, 노름이 넘치기 직전(1e150)의 좌표, 아주 작은 좌표(1e-160, 신중한 경로).
- **정수와 2진 격자**: §4 이웃 2번의 데이터. 참 거리의 동점이 아주 많다.
- **단위 구**(`sphere` 태그, SPT89): (위도, 경도) 라디안의 점 400개와 질의 20개의 집합 여섯 — 균등(경도 [0, 2π)), 거의 같은 점의 군집(퍼짐 10⁻¹²–10⁻⁶, 다섯째마다 중복), 질의의 대척점 둘레, 두 극(위도 ±fl(π/2), 경도 −8–8), 날짜 변경선의 양쪽, 적도 위 질의의 거울상(정확한 동점). 거리 묶음(`distances`)의 짝에는 들지 않는다.
- 거리 함수의 경계: NaN·inf의 모든 자리, 0 벡터, 상수 행, 0/0 항(`canberra`, `braycurtis`, `jaccard`), 음수 좌표, `w`에 0이 섞임, 양정치가 아닌 VI, 대척점(`haversine`).

### 7.2 참값 (`tools/spatial_ref.py` → `tests/spatial/ref/`, `tests/reffile.py`의 npz)
- Arb(python-flint 0.9.0)로 짝마다 참 거리와 RN(d)를 구한다. 차이와 제곱합은 입력이 float64라 충분한 정밀도에서 정확하다. `pow`, `asin`, `sin`이 드는 거리는 Arb의 구간으로 구하고 반올림이 정해질 때까지 정밀도를 늘린다.
- 이웃의 참값: 2번 데이터의 (RN(d), 색인) 정렬, 일반 데이터의 참 거리 표(3번의 확인용). 파일은 작게 둔다(점 수천 개). `sphere.npz`는 단위 구의 집합 여섯에서 질의마다 모든 점까지의 Arb `haversine`이다(120점 × 400, 0.6 MB; 같은 점은 정확히 0).

### 7.3 테스트 (`tests/spatial/`)
- 파일: `test_api_surface.py`(API 문서의 시그니처), `test_distance.py`, `test_squareform.py`, `test_pair_kernel.py`, `test_errors.py`, `test_threads.py`, `test_reference.py`, `test_reference_data.py`(SPT0·SPT1), `test_neighbors.py`, `test_neighbors_reference.py`, `test_neighbors_errors.py`, `test_neighbors_threads.py`(SPT2), `test_kdtree.py`, `test_kdtree_query.py`(SPT3), `test_kdtree_radius.py`, `test_kdtree_pairs.py`(SPT4), `test_neighbors_haversine.py`(SPT83). 마일스톤마다 무엇을 시험하는지는 아래에 있다.
- SPT2에서 만든 것: `test_neighbors.py`(모든 거리·가중·dtype·`haversine`에서 `cdist`로 만든 정의와 색인·거리의 비트가 같음을 세 길(자동, 직접, GEMM)마다; 블록과 구간이 여럿인 크기, k > mB, 빈 쪽, n = 0, 순위의 k, 엄격한 상한과 inf·NaN 거리의 제외; §7.1의 적대적 집합 열셋(먼 군집, 1 ULP 껍질, 중복, 동점, 넘침 직전·넘침·언더플로의 좌표)과 경계 반경; hypothesis의 군집(중심 0–1e15, 퍼짐 1e-6–1e3, 반올림한 동점, 중복)에서 GEMM 길; `RadiusResult`의 필드·쓰기 금지·`tocsr()`의 저장된 0), `test_neighbors_reference.py`(§4 이웃 2번: 격자 집합에서 Arb의 (RN(d), 색인) 순서와 거리; 3번: 일반 집합에서 계약 η 안의 참 kNN; scipy `KDTree` 기록과 집합, sklearn `argmin_min` 기록), `test_neighbors_errors.py`, `test_neighbors_threads.py`(1, 3, 10스레드에서 같은 바이트; `VECLIB_MAXIMUM_THREADS=1`과 기본 모드의 자식 프로세스가 같은 결과의 SHA-256).
- SPT3에서 만든 것: `test_kdtree.py`(만들기의 불변식: 가장 넓은 축의 (좌표, 색인) 중앙값, 상자가 점의 범위, 노드 수와 앞순회 번호, 잎의 색인 순서, 같은 점이 많은 데이터, n = 0과 m = 0; 속성과 쓰기 금지, 데이터의 복사, 입력의 꼴, 피클과 `copy`가 바이트까지 같은 트리, 1·3·10스레드의 같은 트리 바이트, 생성자의 모든 에러), `test_kdtree_query.py`(§4 이웃의 1–3번: `cdist`로 만든 전수와 색인·거리의 비트가 같음을 무작위·중복·동점·격자·군집·먼 군집·1e±200·비정상 수·DBL_MAX 둘레·m = 64에서 `leafsize` 1, 3, 10, 64와 p = 1, 2, inf, 1.5, 3마다, hypothesis; 격자 참값과 (RN(d), 색인) 정렬이 정확히 같음; 일반 참값에서 계약 안의 참 kNN; 모양과 형, 채움, 엄격한 상한, k의 모든 꼴, scipy와 그 기록, 1·3·10스레드의 같은 바이트, `query`의 모든 에러).
- SPT4에서 만든 것: `test_kdtree_radius.py`(`query_ball_point`·`query_ball_tree`·`sparse_distance_matrix`가 `cdist`로 만든 전수와 `indptr`·색인·거리의 바이트까지 같음을 무작위·중복·동점·격자·군집·먼 군집·1e±200·비정상 수·DBL_MAX 둘레·m = 64에서 `leafsize` 1, 3, 10, 64와 p = 1, 2, inf, 1.5, 3마다, 반경은 계산한 거리와 그 앞뒤 double; `radius_neighbors`와 같은 바이트; 점마다의 반경과 브로드캐스트; `return_length`; 격자 참값에서 Arb의 RN(d) ≤ r인 점과 그 거리; 저장된 0; hypothesis; scipy와 그 기록; 1·3·10스레드의 같은 바이트; 모든 에러), `test_kdtree_pairs.py`(`query_pairs`가 `pdist`의 짝과 같음, `count_neighbors`가 `cdist`의 원소 셈과 정확히 같음: r 하나와 여럿, 정렬되지 않은 r과 같은 r, 0과 inf, 구간 셈, 자기 자신과의 셈, 가중(정확한 합에서 정확히, 그 밖은 반올림 안), scipy의 무게 꼴, 반경이 많아 얕게 자르는 길; hypothesis; scipy와 그 기록; 1·3·10스레드의 같은 바이트(가중 셈 포함); 모든 에러).
- SPT1에서 만든 것: `test_distance.py`(scipy 1.18.1과 그 기록, sklearn의 `haversine`, R의 `dist`, 경계의 값, 배치, dtype, 입력), `test_pair_kernel.py`(묶음 커널과 짝 하나의 비트를 모든 거리·길이 경계·남는 행과 열·NaN과 inf에서, hypothesis 포함; n < 16의 `vector_norm`, `chebyshev`와 `ord=inf`, p = 1, 2, inf; 신중한 경로), `test_squareform.py`, `test_reference.py`(Arb 참값의 모든 점이 계약 안), `test_errors.py`, `test_threads.py`(1, 3, 10스레드에서 같은 바이트).
- SPT83에서 만든 것: `test_neighbors_haversine.py`(§3.4의 시험 목록: 적대적 집합과 hypothesis 군집에서 세 길(자동, 전수, 트리)의 `kneighbors`·`radius_neighbors`가 `cdist`로 만든 정의와 바이트까지 같음, 정의역 밖의 전수, 채움과 빈 쪽, Arb 참값 `sphere.npz`에서 계약 안의 참 이웃과 반경, 1·3·10스레드의 같은 바이트), `test_reference_data.py`에 `sphere` 묶음의 점과 덮음.
- hypothesis는 tests/README.md의 프로필을 쓴다(quick은 짧게, deep은 길게). 적대적 생성기는 군집의 중심, 퍼짐, 동점의 수, m, k를 고른다.
- scipy·sklearn과는 그 자리에서 견준다: 거리는 계약 안, 이웃은 동점이 없는 데이터에서 같은 집합과 순서, 동점이 있으면 같은 집합(scipy의 순서는 약속이 아니다).

### 7.4 scipy·sklearn·R의 기록 (`tools/spatial_probe.py` → `tests/spatial/data/`)
- 설계 초안의 탐침(scipy의 거리 경계, `KDTree`의 경계, 커널의 덧셈 순서, sklearn의 경계, 0의 저장, 시간)을 도구 하나로 옮겼다(SPT0). `k=[0]`처럼 프로세스를 죽이는 호출은 자식 프로세스에서 부른다(기록: 신호 11로 죽는다).
- 기록(SPT0, 2026-11-22): `scipy.json`(scipy 1.18.1; 거리 열셋 × 가장자리 짝 27개의 함수·`pdist`·가중 `pdist`, 문서의 `[scipy와 다름]` 하나하나, 덧셈 순서, `KDTree`의 경계 스물여섯, 이웃 집합 열둘의 k = 5 이웃; 1,074개), `sklearn.json`(scikit-learn 1.9.1; 전개식의 오차, 동점의 순서, 에러, `haversine_distances` 32짝; 41개), `r.json`(R 4.6.2 `stats::dist`, 점 집합 넷 × 방법 여섯과 가장자리 셋; 27개). 모든 기록은 그 점의 `digest`(tests/spatial/points.py)를 가진다.
- 참값(SPT0): `distances.npz`(645짝, 거리 스물셋), `counting.npz`(80짝, `hamming`·`jaccard`와 가중), `haversine.npz`(32짝), `neighbors.npz`(질의 308개 × 데이터 전부의 p = 1, 2, inf 거리)를 `tools/spatial_ref.py`가 4프로세스로 4초에 만든다(4.8 MB). 떨어진 점은 없다. scipy와 견주면 넘치거나(1e160) 아주 작은 좌표에서 scipy가 inf·0을 내는 것과 `correlation`의 상쇄(1e15 둘레에서 상대 5×10⁻⁵)를 빼고 모두 ε 크기 안이다.
- R은 `stats::dist`(`euclidean`, `manhattan`, `maximum`, `canberra`, `binary`, `minkowski`)만 비교 패키지다. R의 `canberra`는 0/0 항과 NA를 빼고 남은 항의 수로 다시 비율을 맞추므로(이 초안에서 확인) 그 거리의 R 열은 그런 항이 없는 점만 잰다.

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/spatial.py`): 스윕 `distances`(거리마다, 점은 짝 하나, 판정은 §4의 계약, 비교 패키지 scipy·sklearn(`euclidean_distances`는 전개식의 오차를 그대로 보인다)·R)와 `neighbors`(kNN과 반경의 Arb 일치율; sd는 계약이 100%이고 scipy·sklearn은 기록). SPT3이 `kdtree`, SPT4가 `ball`(`query_ball_point`의 경계 반경, r은 질의의 다섯째 RN(d); 36영역 모두 계약 안. scipy `query_ball_point`와 sklearn `query_radius`는 `euclidean` 격자 넷(`dup/3`, `lattice/int3d`, `bigint3`, `dyadic3`)에서 질의 110개 중 28개가 거리가 꼭 r인 점을 빠뜨린다: r²과 견주기 때문이다)를 더했다. SPT83이 `sphere`(`sphere.npz`의 k = 5 `haversine` 이웃, §4 이웃 3번의 판정, η = 20u; 6영역 모두 계약 안, 최대 0.096. 비교 패키지 sklearn `BallTree`는 극의 집합에서 계약의 3.8×10¹⁴배)를 더했다.
- **속도** (`bench/speed/cases/spatial.py`): 사례마다 마일스톤과 필요한 이름을 적어 두고 그 마일스톤이 끝나면 사례가 된다(sparse의 사례 파일과 같은 틀). 거리마다의 `pdist`·`cdist`(§5의 크기), `kneighbors` k = 1, 10(sklearn `pairwise_distances_argmin_min`, `NearestNeighbors(algorithm='brute')`와), `KDTree` 만들기, `query` k = 1, 10, `query_ball_point`, `query_pairs`, `count_neighbors`(r 하나와 20개), `sparse_distance_matrix`, `haversine`의 `kneighbors` k = 5와 `radius_neighbors`(10⁴ × 10⁵, sklearn `BallTree`와; SPT83). scipy는 `workers=1`과 `workers=-1` 둘 다 비교 대상이다(sd는 기본이 10스레드라서). 결과는 이웃의 색인과 거리를 견준다.
- **골든** (`tests/golden/cases_spatial.py`, 기록 `tests/golden/data/spatial.json`): 모든 거리 × 경계 데이터, `squareform`, 이웃 함수의 모든 꼴(`haversine`의 트리 길 넷과 정의역 밖 하나 포함), 틀린 호출. 1·3·10스레드의 결과가 모두 같아야 한다. 큰 케이스는 병렬 길과 GEMM 고르기를 탈 크기로 두되 워커 하나의 메모리가 1 GB를 넘지 않게 한다.

---

## 8. 마일스톤
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| SPT0 준비 | API 문서 초안(`docs/api/spatial.md`: 모든 공개 이름의 시그니처, scipy·sklearn → sd 대응표, §9의 답), 영역 등록(`bench/areas.py`, ARCHITECTURE, API.md의 색인), 시험 데이터(§7.1), 참값 도구(§7.2), 탐침 도구와 기록(§7.4), §3.2의 상수와 증명의 마무리, 짝 커널의 두 안(§3.1의 `vector_norm`과 같은 순서 대 타일 순서)과 GEMM의 동시 호출(AMX) 측정, 속도 사례의 틀 | 문서와 도구가 커밋되고 탐침 기록이 있다. §9의 답이 §1.1의 결정으로 옮겨졌다 |
| SPT1 거리 — **완료·승인(위임, 2026-11-22)** | `pdist cdist squareform`, 거리 함수 열넷(`haversine` 포함, SPT9), `w`·`V`·`VI`, 배치, bool의 비트 묶음 | 정확도 스윕이 계약 안, 타일 순서의 비트 테스트(SPT22·SPT26: 묶음 커널과 짝 하나, n < 16에서 `vector_norm`), 골든, scipy·R과의 속도 사례 |
| SPT2 축약과 전수 이웃 — **완료·승인(위임, 2026-11-22)** | `kneighbors radius_neighbors`, `QueryResult`, `RadiusResult`와 `tocsr()`, GEMM 고르기(§3.2)와 직접 전수 | 같은 커널의 전수와 정확히 같음(모든 거리, 적대적 데이터, hypothesis), Accelerate 모드와 무관함, sklearn과의 속도 사례 |
| SPT3 KD 트리와 kNN — **완료·승인(위임, 2026-11-22)** | `KDTree`(만들기, 속성, 피클), `query`(k, k 시퀀스, 상한, p) | §4 이웃의 1–3번, 스레드마다 같은 트리와 결과, scipy와의 속도 사례 |
| SPT4 반경·짝·개수 — **완료·승인(위임, 2026-11-22)** | `query_ball_point query_ball_tree query_pairs count_neighbors sparse_distance_matrix` | 같은 커널의 전수와 같음, 가중 셈의 결정성, scipy와 집합이 같음, 속도 사례 |
| SPT5 BallTree (건너뜀) | `BallTree`와 그 거리들 | 열지 않는다: SPT8이 첫 판에서 뺐다(나중 후보). SPT6에서 KD 트리가 칠 수 없는 거리의 전수 시간을 재 보니 `haversine`만 크게 졌고, 사용자가 단위 구의 KD 트리로 풀게 했다(SPT83) |
| SPT6 성능과 일관성 점검 — **완료·승인(위임, 2026-11-23)** | 모든 공개 함수를 크기마다 scipy·sklearn·R과 견주어 고치기, 길의 문턱(n₀, 트리 대 전수, `leafsize`, 블록 크기) 정하기, 메시지·시그니처의 영역 간 일관성, 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다 |
| SPT7 문서 — **완료·승인(위임, 2026-11-23)** | `docs/api/spatial.md`의 확정, 이 문서를 구현대로 고치기(잰 값, 정한 상수, 파일), `haversine` 이웃의 트리 길(SPT83–SPT89) | 문서가 코드와 맞고 `test_api_surface`가 통과한다 |

진행 원칙은 다른 영역과 같았다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. SPT1이 짝 커널을 만들므로 맨 앞이다. SPT2와 SPT3은 SPT1에만 기대어 순서를 바꿀 수 있다. SPT4는 SPT3 뒤다. 기존 영역의 결과는 비트까지 같아야 한다(이 영역은 다른 영역을 고치지 않는다; linalg의 잎을 쓰기만 한다).

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
