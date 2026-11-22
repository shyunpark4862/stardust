# sd.spatial API

> 상태: **초안** (SPT0, 2026-11-22). 공개된 이름은 아직 없다. 아래 시그니처는 scipy 1.18.1의 `scipy.spatial`(`distance`, `KDTree`)의 이름과 인자 이름에 stardust의 규칙과 SPT1–SPT28의 결정을 입힌 안이고, 마일스톤 SPT1–SPT4에서 하나씩 공개된다(docs/design/spatial.md §8: 거리는 SPT1, `kneighbors`·`radius_neighbors`와 결과 객체는 SPT2, `KDTree`와 `query`는 SPT3, 반경·짝·개수는 SPT4).

- 설계와 결정의 근거는 [docs/design/spatial.md](../design/spatial.md)(§1.1 SPT1–SPT28), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/spatial/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 클래스는 `name(...)`의 생성자), 클래스의 메서드(`Class.method(...)`)와 속성(`Class.attr`)이다. 모듈 수준의 이름이 계획한 공개 이름 전부이고, 공개된 것은 시그니처가 블록과 같아야 한다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- **이름공간은 평평하다**(SPT5, SPT6): 거리 함수도 `sd.spatial`에 바로 있다(`sd.spatial.pdist`, `sd.spatial.cosine`, `sd.spatial.KDTree`). `sd.spatial.distance`는 없다(`AttributeError`; 별칭을 두지 않는다). scipy의 `from scipy.spatial.distance import cdist`는 `from stardust.spatial import cdist`다. 평평하게 해도 이름이 겹치지 않는다.
- **이름은 scipy.spatial의 것이다**(SPT4): `pdist`, `cdist`, `squareform`, 거리 함수, `KDTree`와 그 메서드(`query`, `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, `sparse_distance_matrix`). scipy에 없는 한 번에 끝나는 이웃 찾기는 sklearn `NearestNeighbors`의 두 메서드 이름 `kneighbors`, `radius_neighbors`다(SPT7).
- **데이터는 위치로만, 옵션은 키워드로만** 받는다(API.md, 다른 영역과 같다). `metric`과 그 인자(`p`, `w`, `V`, `VI`), `k`, `leafsize`도 키워드만이다(SPT15, `[scipy와 다름]`: scipy의 `pdist(X, 'cityblock')`, `tree.query(x, 3)`은 `TypeError`다). 거리가 필요로 하는 행렬은 거리 함수에서만 위치다(`seuclidean(u, v, V)`, `mahalanobis(u, v, VI)`; scipy와 같다).
- **별칭이 없다**: 거리 이름은 소문자 정식 이름 열넷만 받는다(`'e'`, `'euclid'`, `'cb'`, `'matching'`, `'cos'`, 대문자 이름은 `ValueError`). `cKDTree`, `distance_matrix`(scipy 1.18.0에서 폐기 예고, `cdist`와 같다), `minkowski_distance`는 없다.
- `workers`(SPT10; 스레드 수는 `sd.set_num_threads` 하나다), `eps`(SPT11; 결과는 언제나 정확하다), `boxsize`(SPT20), `balanced_tree`, `compact_nodes`, `copy_data`(SPT18), `return_sorted`, `output_type`(SPT17), 호출 가능한 `metric`, `out=`(SPT14)는 없다. 주면 모르는 키워드의 `TypeError`다.
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1): `pdist(): unexpected keyword argument 'workers'`, `query(): takes 1 positional argument but 2 were given`.

### §1.2 입력 (SPT12, SPT13, SPT23, SPT24)
- **데이터**(`X`, `XA`, `XB`, `u`, `v`, `data`, `x`, `r`의 배열)는 API.md §5의 상위 함수 입력이다: `sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `DataFrame`(2차원; 결측은 NaN). 계산은 float64다.
  - bool·int64는 float64로 바꾼다(linalg의 규칙과 같다; scipy도 바꾼다). 예외는 `hamming`·`jaccard`다: bool, int64, float64를 입력의 dtype 그대로 비교한다(bool은 비트 묶음의 빠른 길). 그래서 2⁶⁰과 2⁶⁰ + 1은 다른 값이다(`[scipy와 다름]`: scipy는 float64로 바꾼 뒤 비교해 같은 값으로 본다).
  - complex128은 `TypeError`다(`[scipy와 다름]`: scipy의 `pdist`·`cdist`는 `ValueError: Unsupported dtype complex128`). numpy의 정보 손실 없는 dtype은 `sd.asarray`의 규칙으로 넓히고, 문자열·object는 `TypeError`다.
  - 모양: `pdist`는 (m, n), `cdist`는 (mA, n)과 (mB, n)이다. n이 다르면 `ValueError`다. n = 0(특징이 없음)도 받는다: 거리는 그 정의를 빈 합에 쓴 값이다(`euclidean`·`cityblock`·`chebyshev`는 0, `cosine`·`correlation`·`braycurtis`·`hamming`은 0/0의 NaN; scipy의 `pdist`와 같다. `[scipy와 다름]`: scipy의 함수 `chebyshev`는 n = 0에서 `ValueError`다). 행이 0이거나 1이면 `pdist`는 빈 배열이다.
- **유한하지 않은 값**: 거리 함수는 IEEE 규칙으로 퍼뜨린다(NaN → NaN, inf − inf → NaN, inf와 유한 값의 거리는 inf; `chebyshev`와 `canberra`도). 이웃 찾기(`KDTree`, `kneighbors`, `radius_neighbors`)는 데이터와 질의점이 모두 유한해야 한다(`ValueError`: `KDTree(): 'data' must be finite`). NaN이 있으면 "가장 가까운"이 정해지지 않는다.
- **가중치 `w`**: 1차원, 길이 n, 모든 원소가 유한하고 0 이상이다(`ValueError`; `[scipy와 다름]`: scipy는 음수만 막고 NaN·inf 가중치로 NaN·inf 거리를 낸다). `w`를 받는 거리는 scipy와 같다: `seuclidean`·`mahalanobis`·`haversine`을 뺀 모두다.
- **결과 dtype**: 거리는 float64, 색인은 int64, 셈은 int64(가중이면 float64)다. `squareform`만 입력의 dtype을 지킨다(bool·int64·float64; scipy와 같다).

### §1.3 결정성과 같은 짝의 같은 비트 (SPT2, SPT21, SPT22)
- **짝 하나의 거리는 한 커널이 정한다.** `pdist`, `cdist`, 거리 함수, `kneighbors`, `radius_neighbors`, `KDTree`의 모든 메서드가 같은 짝 (x, y)에 같은 비트를 낸다. 그래서 트리의 k번째 거리는 `cdist`의 그 원소와 같고, 반경의 판정은 돌려주는 거리 그대로다(d ≤ r). `[scipy와 다름]`: scipy는 `cdist`와 `KDTree`가 덧셈 순서가 달라 같은 짝의 거리가 n = 3에서 12%, n = 3000에서 96%의 짝이 다르다(docs/design/spatial.md §3.1).
- 합의 순서는 이 영역의 것이다(SPT22, 잰 결과에 따른 타일 순서): 같은 거리를 `sd.linalg.vector_norm(x - y)`로 구하면 끝자리가 다를 수 있다(둘 다 정확도 계약 안이다).
- **같은 입력이면 스레드 수, 계산 길(트리, 전수, GEMM 고르기), Accelerate의 스레드 모드와 상관없이 같은 결과다.** 이웃의 집합, 동점의 순서, 결과의 정렬, 거리의 비트가 모두 그렇다. 길은 거리 이름과 입력의 모양으로만 고르고(SPT21), 결과를 바꾸지 않는다.
- **동점은 (거리, 색인)의 사전 순서다**(SPT4). 같은 거리면 색인이 작은 것이 먼저이고 이것을 약속한다. `[scipy와 다름]`: scipy와 sklearn은 약속하지 않고 실제로도 다르다(같은 점 6개에서 scipy `[1, 2, 3, 4, 5, 0]`, sklearn `[3, 4, 1, 5, 2, 0]`).

### §1.4 에러
- API.md §2의 종류와 꼴이다. 공개 함수의 에러는 `pdist(): `로 시작하고, 메서드는 메서드 이름(`query(): `), 생성자는 클래스 이름(`KDTree(): `)이다. 경고는 내지 않는다.
- scipy가 조용히 받는 틀린 인자는 에러다(§1.5의 목록). 거리가 쓰지 않는 인자는 `TypeError`다: `pdist(): 'p' applies only to metric='minkowski'`(scipy는 바인딩의 시그니처 덤프).
- 요청한 출력(`pdist`의 m(m−1)/2, `cdist`의 mA·mB, `kneighbors`의 mA·k)이 물리 메모리보다 크면 할당 전에 `MemoryError`다(API.md §2).

### §1.5 scipy·sklearn과 다른 점 (모음)
- 거리: 별칭과 대문자 이름 없음, `metric`과 그 인자는 키워드만, 쓰지 않는 인자는 `TypeError`, complex는 `TypeError`, `p ≤ 0`·NaN `p`·`w`의 NaN·inf·0 이하의 `V`는 `ValueError`, `chebyshev`와 `canberra`는 NaN을 퍼뜨린다, `euclidean`·`minkowski`는 중간값이 넘쳐도 유한한 거리다, 모든 거리 함수가 (…, n) 배치를 받는다, `hamming`·`jaccard`는 int64를 그대로 비교한다, 호출 가능한 `metric`과 `out=` 없음.
- 이웃: 동점은 (거리, 색인), 데이터는 언제나 복사, `leafsize`만, `k`·`leafsize`는 int만, `k=0`·`k=[0]`·음수·NaN 반경과 상한은 `ValueError`, 반경 결과는 `RadiusResult`이고 언제나 색인 순서, `query_pairs`는 정렬된 배열, `sparse_distance_matrix`는 `csr_array`, `workers`·`eps`·`boxsize`·`return_sorted`·`output_type` 없음, `kneighbors`의 거리는 언제나 직접 계산(`[sklearn과 다름]`: sklearn의 전개식은 1e8 + [0, 1)³의 점들에서 거리를 모두 0으로, 가장 가까운 점을 20개 가운데 하나도 맞히지 못했다; docs/design/spatial.md §3.2).

## §2. 거리 (SPT1)
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
- **`pdist(X)`**: (m, n)의 행 i < j 짝의 거리, 행 우선의 압축 꼴 (m(m−1)/2,)이다(scipy와 같다). `pdist(X)`의 원소는 `cdist(X, X)`의 그 원소와 비트까지 같다(대칭: fl(x − y) = −fl(y − x)).
- **`cdist(XA, XB)`**: (mA, mB)의 거리 행렬이다. 모든 거리를 직접 계산한다(GEMM 전개식을 쓰지 않는다; SPT2).
- **`metric`**: `'euclidean'`, `'sqeuclidean'`, `'cityblock'`, `'chebyshev'`, `'minkowski'`, `'seuclidean'`, `'mahalanobis'`, `'cosine'`, `'correlation'`, `'canberra'`, `'braycurtis'`, `'hamming'`, `'jaccard'`, `'haversine'`다. 다른 문자열은 `ValueError`, 문자열이 아니면 `TypeError`다(API.md §2.1의 문자열 옵션 꼴).
- **`p`**: `metric='minkowski'`에서만 받는다. 기본(None)은 2이고, 0보다 큰 실수나 inf다(`p < 1`은 삼각 부등식이 없는 준거리이고 scipy처럼 받는다). 0 이하와 NaN은 `ValueError`다(`[scipy와 다름]`: scipy의 `pdist`는 `p=-1`에서 값을, `p=0`에서 inf를 낸다). p = 1, 2, inf는 `cityblock`, `euclidean`, `chebyshev`와 비트까지 같다.
- **`V`**(`seuclidean`): 기본은 `pdist`가 `X`, `cdist`가 `XA`와 `XB`를 세로로 이은 것의 열 분산(`correction=1`)이다(scipy와 같다). 받은 `V`는 1차원 길이 n이고 유한한 양수여야 한다(`[scipy와 다름]`: scipy는 음수 `V`에서 뜻 없는 값을, 0에서 inf를 낸다).
- **`VI`**(`mahalanobis`): 기본은 같은 자료의 공분산의 역행렬이다(scipy와 같다; 관측이 차원 이하이면 `ValueError`). 받은 `VI`는 (n, n)이다. 양정치가 아니면 이차식이 음수인 짝의 거리가 NaN이다(scipy와 같다).
- **배치**: 모든 거리 함수가 (…, n) 배열을 받아 앞 축끼리 브로드캐스트하고, 결과는 앞 축의 모양이다(1차원 둘이면 Python `float`). 짝 하나는 `cdist`의 그 짝과 비트까지 같다. `[scipy와 다름]`: scipy 1.18.1은 `minkowski`·`euclidean`·`sqeuclidean`만 배치를 받는다.

**거리의 정의** (z = x − y를 원소마다 한 번 반올림한 것)
| 이름 | 정의 | `w`가 있을 때 (scipy와 같은 뜻) | 경계의 값 |
|---|---|---|---|
| `euclidean` | ‖z‖₂ | √(Σ wᵢzᵢ²) | 제곱합이 넘쳐도 비율을 맞춰 유한한 값(`[scipy와 다름]`: scipy와 R은 좌표 1e200에서 inf) |
| `sqeuclidean` | Σ zᵢ² | Σ wᵢzᵢ² | 참값이 넘치면 inf |
| `cityblock` | Σ \|zᵢ\| | Σ wᵢ\|zᵢ\| | |
| `chebyshev` | maxᵢ \|zᵢ\| | wᵢ > 0인 i의 최댓값 | NaN을 퍼뜨린다(`[scipy와 다름]`: scipy의 `cdist`는 NaN을 건너뛰어 `[nan, 0]`과 `[1, 5]`의 거리가 5다) |
| `minkowski` | (Σ \|zᵢ\|ᵖ)^(1/p) | (Σ wᵢ\|zᵢ\|ᵖ)^(1/p) | 넘침·언더플로는 비율을 맞춘다(`[scipy와 다름]`: scipy는 (0, 0)과 (3, 4)의 p = 2000 거리가 inf) |
| `seuclidean` | √(Σ zᵢ²/Vᵢ) | 없음 | |
| `mahalanobis` | √(zᵀ VI z) | 없음 | 이차식이 음수면 NaN |
| `cosine` | 1 − x·y/(‖x‖‖y‖), [0, 2]로 자른다 | 가중 내적과 가중 노름 | 영벡터가 있으면 NaN |
| `correlation` | 행마다 (가중) 평균을 뺀 뒤의 `cosine`(`centered=False`면 빼지 않는다) | 가중 평균 Σwx/Σw | 상수 행은 NaN |
| `canberra` | Σ \|zᵢ\|/(\|xᵢ\| + \|yᵢ\|), 0/0 항은 0 | Σ wᵢ(…) | NaN을 퍼뜨린다(`[scipy와 다름]`: scipy의 함수 `canberra`는 NaN 항을 건너뛰고 `pdist`는 퍼뜨린다) |
| `braycurtis` | Σ\|zᵢ\| / Σ\|xᵢ + yᵢ\| | 두 합 모두 wᵢ를 곱한다 | 0/0은 NaN, 분모만 0이면 inf |
| `hamming` | (Σ wᵢ[xᵢ ≠ yᵢ]) / Σ wᵢ, 기본 wᵢ = 1 | 같은 식 | NaN끼리는 다른 값으로 센다 |
| `jaccard` | Σ wᵢ[(xᵢ≠0) xor (yᵢ≠0)] / Σ wᵢ[(xᵢ≠0) or (yᵢ≠0)] | 같은 식 | 분모가 0이면 0. NaN은 0이 아니다 |
| `haversine` | 2·asin(√(sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2))), 입력은 (위도, 경도) 라디안의 2열, 결과는 단위 구의 중심각(라디안) | 없음 | 열이 2가 아니면 `ValueError`. 식의 꼴은 SPT1에서 대척점까지 계약을 지키는 것으로 고른다 |

- `haversine`은 sklearn `haversine_distances`의 뜻이다(SPT9; scipy에는 없다). 이웃 찾기에서는 트리를 쓰지 않고 전수로 간다.
- **`squareform`**: scipy와 같다(SPT16). 벡터 → 행렬은 길이가 n(n−1)/2가 아니면 `ValueError`, 빈 벡터는 (1, 1)의 0이다. 행렬 → 벡터는 `checks=True`면 정확한 대칭과 0 대각을 검사한다. NaN이 대칭 검사에 걸리면 메시지가 NaN을 말한다(`squareform(): 'X' has NaN off the diagonal, so it is not symmetric`; scipy의 메시지는 "must be symmetric"뿐이다). (1, 1)과 (0, 0)은 빈 벡터다. `force`는 `'no'`, `'tomatrix'`, `'tovector'`다. dtype을 지키고, 3차원 이상은 `ValueError`다.

## §3. 한 번에 끝나는 이웃 찾기 (SPT2)
```
kneighbors(XA, XB, /, *, k=1, metric='euclidean', p=None, w=None, V=None, VI=None, distance_upper_bound=inf)
radius_neighbors(XA, XB, r, /, *, metric='euclidean', p=None, w=None, V=None, VI=None, return_length=False)
```
- `XA`의 행마다 `XB`에서 가장 가까운 k개, 또는 거리가 r 이하인 모두를 찾는다. 전체 거리 행렬을 만들지 않는다(작업 메모리는 입력과 무관한 상한 안이다).
- **결과는 정의로 정해진다**: `kneighbors(XA, XB, k=k)`는 `cdist(XA, XB)`의 각 행을 (거리, 열 번호)로 정렬해 앞 k개를 고른 것과 같고, 거리는 그 원소와 비트까지 같다. Minkowski 계열(`euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`, `minkowski` p ≥ 1)은 `KDTree(XB).query(XA, k=k, p=…)`와도 같다. `radius_neighbors(XA, XB, r)`는 `KDTree(XB).query_ball_point(XA, r)`, `KDTree(XA).query_ball_tree(KDTree(XB), r)`와 같다.
- 모든 거리 이름과 그 인자를 받는다. `k`, `distance_upper_bound`, 채움의 규칙은 `KDTree.query`(§4)이고, `r`과 `return_length`는 `query_ball_point`(§4)다.
- 결과는 `kneighbors`가 `QueryResult(distances, indices)`, `radius_neighbors`가 `RadiusResult`(행 수 mA, 열 수 mB)다(§5). `return_length=True`면 행마다의 개수(int64 (mA,))다.
- sklearn의 `pairwise_distances_argmin_min(X, Y)`는 `kneighbors(X, Y)`의 `(distances, indices)`다(풀기 순서만 반대). `[sklearn과 다름]`: 거리가 언제나 직접 계산한 값이다(§1.5).

## §4. `KDTree` (SPT3, SPT4)
```
KDTree(data, /, *, leafsize=10)
KDTree.query(x, /, *, k=1, p=2.0, distance_upper_bound=inf)
KDTree.query_ball_point(x, r, /, *, p=2.0, return_length=False)
KDTree.query_ball_tree(other, r, /, *, p=2.0)
KDTree.query_pairs(r, /, *, p=2.0)
KDTree.count_neighbors(other, r, /, *, p=2.0, weights=None, cumulative=True)
KDTree.sparse_distance_matrix(other, max_distance, /, *, p=2.0)
KDTree.data
KDTree.n
KDTree.m
KDTree.leafsize
KDTree.maxes
KDTree.mins
KDTree.size
```
- **만들기**: `data`는 (n, m)이다(1차원과 3차원 이상은 `ValueError`; scipy와 같다). 데이터는 언제나 복사한다(`[scipy와 다름]`: scipy는 복사하지 않을 수 있어 만든 뒤 원본을 고치면 답이 틀린다). n = 0(질의는 모두 채움 값)과 m = 0(모든 거리가 0; scipy는 질의에서 `IndexError`)도 받는다. 트리는 언제나 가장 넓은 축의 중앙값으로 나누고 노드 상자를 실제 점의 범위로 줄인다. 만들기는 결정적이다: 트리의 모양은 데이터와 `leafsize`로만 정해진다.
- **`leafsize`**: 1 이상의 int다(bool·실수는 `TypeError`; `[scipy와 다름]`: scipy는 2.5를 2로 자른다). 결과는 `leafsize`와 무관하고 빠르기만 바뀐다. 기본 10은 SPT6에서 재서 1.2배 넘게 빠른 값이 있으면 바꾼다(SPT18).
- **속성**: `data`(쓰기 금지 (n, m) float64, 입력 순서), `n`, `m`, `leafsize`(int), `maxes`, `mins`(쓰기 금지 (m,)), `size`(노드 수). scipy의 `tree`, `indices`, `boxsize`는 없다. 피클할 수 있다(같은 데이터와 `leafsize`로 다시 만든 같은 트리다).
- **`p`**: 1 이상의 실수나 inf다(1 미만과 NaN은 `ValueError`; scipy와 같다). p = 1, 2, inf는 `cityblock`, `euclidean`, `chebyshev`와, 그 밖의 p는 `minkowski(p=)`와 같은 커널이다.
- **`query(x)`**: 데이터 점 가운데 x에 가장 가까운 k개다.
  - x는 (…, m)이다. 마지막 축이 m이 아니면 `ValueError`.
  - `k`는 1 이상의 int거나 1 이상의 int 시퀀스(1부터 센 순위의 목록, scipy의 뜻)다. 0, 음수, 빈 시퀀스는 `ValueError`, bool·실수는 `TypeError`다(`[scipy와 다름]`: scipy는 `k=True`와 `k=2.0`을 받고, `k=0`은 numpy 내부 메시지의 `ValueError`이며, `k=[0]`은 프로세스를 죽인다).
  - 결과는 `QueryResult(distances, indices)`이고 `d, i = tree.query(x)`로 풀린다. 모양은 scipy와 같다: k가 int면 (…, k)이고 k = 1이면 마지막 축을 없앤다. k가 시퀀스면 (…, len(k))다. x가 (m,)이고 k = 1이면 Python `float`와 `int`다(scipy는 `float`와 `numpy.int64`).
  - 각 행은 (거리, 색인) 순서다(§1.3).
  - **채움**: 찾은 이웃이 k보다 적으면(k > n, 또는 `distance_upper_bound`) 남은 자리는 거리 inf, 색인 n이다(scipy와 같다; sklearn은 k > n이면 에러다).
  - **`distance_upper_bound`**: 거리가 이 값보다 **작은** 이웃만이다(엄격한 부등호; scipy와 같다). 0 이상의 실수나 inf여야 한다(`[scipy와 다름]`: scipy는 상한을 제곱해 비교해 −1이 1처럼 동작하고, NaN이면 모든 자리가 채움 값이다).
- **`query_ball_point(x, r)`**: 거리가 r **이하**인 데이터 점이다(경계를 넣는다; scipy와 같다). r은 0 이상의 실수나 inf, 또는 x의 배치 모양으로 브로드캐스트되는 배열이다. 음수·NaN은 `ValueError`다(`[scipy와 다름]`: scipy는 r = −1을 r = 1처럼, NaN을 빈 목록으로 다룬다). 판정은 돌려주는 거리 그 자체다(scipy는 제곱의 자리에서 비교해, 돌려줄 거리가 r과 같은 짝의 판정이 무작위 시험의 23%에서 √의 자리와 갈렸다).
  - 결과는 `RadiusResult`다(§5). x의 배치 축은 C 순서로 펼쳐 행이 된다. 점 하나 (m,)도 행 하나의 결과다(`[scipy와 다름]`: scipy는 점 하나면 list, 여럿이면 list의 object 배열이다).
  - 행 안은 색인 오름차순이다(`[scipy와 다름]`: scipy의 기본은 점 하나면 정렬하지 않고 여럿이면 정렬한다).
  - `return_length=True`면 행마다의 개수를 int64 배열(x의 배치 모양; 점 하나면 Python `int`)로 돌려준다(scipy와 같다).
- **`query_ball_tree(other, r)`**: `self`의 점마다 `other`의 점 가운데 거리가 r 이하인 것이다. 결과는 `RadiusResult`이고 모양은 (self.n, other.n)이다. `other`는 `KDTree`이고(아니면 `TypeError`) `other.m`이 다르면 `ValueError`다.
- **`query_pairs(r)`**: i < j이고 거리가 r 이하인 짝이다. (k, 2) int64 배열이고 (i, j)의 사전 순서다(`[scipy와 다름]`: scipy의 기본은 순서 없는 `set`이다).
- **`count_neighbors(other, r)`**: `self`의 점 x₁과 `other`의 점 x₂의 순서쌍 가운데 거리가 r 이하인 것의 수다(`other`가 `self`면 (i, i)와 (i, j)·(j, i)를 모두 센다; scipy와 같다). r은 실수 하나나 1차원 배열이다. `cumulative=False`면 r은 줄지 않아야 하고 결과의 i번째는 r[i−1] < d ≤ r[i]의 수다(첫째는 d ≤ r[0]). `weights`는 scipy의 꼴이다(배열 하나는 `other`가 `self`일 때만, 튜플 `(w_self, w_other)`의 한쪽은 None일 수 있다). 가중이면 짝의 가중치 곱의 합이고 순서는 데이터와 트리로만 정해진다(SPT25; 스레드 수와 무관하다). 결과는 r이 실수 하나면 Python `int`(가중이면 `float`), 배열이면 int64(가중이면 float64) 배열이다(scipy와 같다).
- **`sparse_distance_matrix(other, max_distance)`**: 거리가 `max_distance` 이하인 짝의 거리를 담은 `sd.sparse.csr_array` (self.n, other.n)이다. `self.query_ball_tree(other, max_distance).tocsr()`와 같다. 거리가 0인 짝은 저장된 0이다(§5). `[scipy와 다름]`: scipy의 기본은 `dok_matrix`이고 `output_type`으로 형식을 고른다. 형식을 바꾸려면 `asformat`을 쓴다.

## §5. 결과 객체 (SPT17)
```
QueryResult.distances
QueryResult.indices
RadiusResult.indptr
RadiusResult.indices
RadiusResult.distances
RadiusResult.shape
RadiusResult.tocsr()
```
- **`QueryResult(distances, indices)`**: `query`와 `kneighbors`가 함께 쓰는 namedtuple이다(`stardust._results`; 이름은 `<함수>Result`의 규칙). 배열은 쓰기 금지다(API.md §2.2).
- **`RadiusResult`**: `query_ball_point`, `query_ball_tree`, `radius_neighbors`가 함께 쓰는 결과 클래스다(내보내지 않는다; sd.htest의 결과처럼).
  - `indptr`(int64, 행 수 + 1), `indices`(int64), `distances`(float64), `shape`(행 수, 데이터 점 수). 행 i의 이웃은 `indices[indptr[i]:indptr[i+1]]`이고 색인 오름차순, 그 거리는 같은 자리의 `distances`다. 배열은 쓰기 금지이고 읽을 때 복사하지 않는다.
  - `tocsr()`: `sd.sparse.csr_array((distances, indices, indptr), shape=shape)`다. 색인이 이미 표준 꼴(정렬, 중복 없음)이다. 부를 때마다 새 희소 배열이다(희소 배열의 `data`는 쓸 수 있어야 하므로 거리를 복사한다).
  - **거리가 0인 짝**(자기 자신, 중복된 점)은 **저장된 0**이다(SPT3): `nnz`가 그 짝을 세고 `count_nonzero()`는 세지 않는다. 지우려면 `eliminate_zeros()`를 부른다. 희소 행렬의 저장 자리가 이웃 그래프의 변이라 0을 지우면 중복된 점과 자기 자신이 이웃에서 사라지기 때문이다. scipy의 `sparse_distance_matrix`도 0인 짝을 저장한다(같은 두 점에서 nnz 4, `count_nonzero()` 0).
  - `repr`는 한 줄이다: `RadiusResult(shape=(1000, 5000), nnz=10234)`. `summary()`는 없다.
- 결과 객체가 아닌 결과(`pdist`, `cdist`, `squareform`, `query_pairs`, `count_neighbors`, `return_length`의 개수)는 보통의 새 배열이라 쓸 수 있다.

## §6. scipy·sklearn → sd.spatial

**scipy.spatial.distance**
| scipy | sd.spatial | 마일스톤 |
|---|---|---|
| `pdist`, `cdist`, `squareform` | 같은 이름(`metric`과 그 인자는 키워드만) | SPT1 |
| `euclidean sqeuclidean cityblock chebyshev minkowski seuclidean mahalanobis cosine correlation canberra braycurtis hamming jaccard` | 같은 이름, 모두 배치를 받는다 | SPT1 |
| 불 거리 `dice rogerstanimoto russellrao sokalsneath yule`, `jensenshannon`, `directed_hausdorff` | 없음(나중 후보) | — |
| `is_valid_dm`, `is_valid_y`, `num_obs_dm`, `num_obs_y` | 없음(`squareform`의 검사는 안에 있다) | — |
| 거리 이름의 별칭과 대문자, 호출 가능한 `metric` | 없음 | — |

**scipy.spatial**
| scipy | sd.spatial | 마일스톤 |
|---|---|---|
| `KDTree`, `KDTree.query` | 같은 이름 | SPT3 |
| `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, `sparse_distance_matrix` | 같은 이름 | SPT4 |
| `cKDTree` | 없음(`KDTree`와 같은 것) | — |
| `KDTree.tree`, `Rectangle`, `minkowski_distance`, `minkowski_distance_p` | 없음(트리의 속은 공개하지 않는다; 앞의 둘은 `minkowski`와 그 p제곱) | — |
| `distance_matrix` | 없음(`cdist`) | — |
| `ConvexHull`, `Delaunay`, `Voronoi`, `SphericalVoronoi`, `HalfspaceIntersection`, `tsearch`, `procrustes`, `geometric_slerp`, `transform` | 없음(나중 후보, SPT1) | — |

**scikit-learn**
| sklearn | sd.spatial | 마일스톤 |
|---|---|---|
| `pairwise_distances_argmin_min(X, Y)` | `kneighbors(X, Y)`(풀기 순서가 `(distances, indices)`) | SPT2 |
| `pairwise_distances_argmin(X, Y)` | `kneighbors(X, Y).indices` | SPT2 |
| `NearestNeighbors(...).fit(Y).kneighbors(X)`, `.radius_neighbors(X)` | `kneighbors(X, Y, k=)`, `radius_neighbors(X, Y, r)`(클래스는 없다) | SPT2 |
| `radius_neighbors_graph(mode='distance')` | `radius_neighbors(X, Y, r).tocsr()` | SPT2 |
| `euclidean_distances(X, Y, squared=)` | `cdist(X, Y, metric='euclidean')`, `metric='sqeuclidean'` | SPT1 |
| `paired_distances` | 거리 함수의 배치(`sd.spatial.euclidean(U, V)`) | SPT1 |
| `haversine_distances` | `cdist(metric='haversine')`, `haversine(u, v)` | SPT1 |
| `sklearn.neighbors.KDTree` | `KDTree`(scipy의 API) | SPT3 |
| `BallTree` | 없음(나중 후보, SPT8) | — |
| `kneighbors_graph`, `nan_euclidean_distances`, `KernelDensity`, `two_point_correlation` | 없음(짝 세기는 `count_neighbors`) | — |

## §7. 정확도
- 거리는 Arb로 구한 참 거리와 견준 계약 안이다(docs/design/spatial.md §4). `chebyshev`는 올바른 반올림이고, `euclidean`은 n = 3에서 약 3.5 ULP 이하다. 반올림 크기의 결과는 scipy와 끝자리가 다를 수 있다(API.md §4의 규칙 3).
- 이웃은 같은 커널의 전수 결과와 정확히 같고(길·스레드와 무관), 거리가 올바르게 반올림되는 데이터(정수·2진 격자)에서는 Arb로 정렬한 참 이웃과 정확히 같다. 일반 데이터에서는 참 거리가 1 ULP 안에서 다른 두 점의 순서가 돌려준 거리의 순서다.
