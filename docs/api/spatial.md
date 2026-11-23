# sd.spatial API

> 상태: **확정** (2026-11-23; 승인(위임)). §2–§9의 모든 이름이 구현되어 있다: 거리(§2), 한 번에 끝나는 이웃 찾기 `kneighbors`·`radius_neighbors`(§3), `KDTree`와 그 여섯 질의(§4), 결과 객체(§5), 계산 기하 `Delaunay`·`ConvexHull`·`Voronoi`(§6–§9). 시그니처는 scipy 1.18.1의 `scipy.spatial`(`distance`, `KDTree`, `Delaunay`, `ConvexHull`, `Voronoi`)의 이름과 인자 이름에 stardust의 규칙과 결정(SPT1–SPT89, SPT90–SPT195)을 입힌 것이다.

- 설계와 결정의 근거는 거리와 이웃 찾기가 [docs/design/spatial.md](../design/spatial.md)(§1.1 SPT1–SPT89), 계산 기하가 [docs/design/spatial_geometry.md](../design/spatial_geometry.md)(§1.1 SPT90–SPT195)이고, 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/spatial/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 클래스는 `name(...)`의 생성자), 클래스의 메서드(`Class.method(...)`)와 속성(`Class.attr`)이다. 모듈 수준의 이름이 공개 이름 전부이고, 시그니처가 블록과 같아야 한다.

## §1. 공통 규칙 (거리와 이웃 찾기)

### §1.1 이름과 호출 모양
- **이름공간은 평평하다**(SPT5, SPT6): 거리 함수도 `sd.spatial`에 바로 있다(`sd.spatial.pdist`, `sd.spatial.cosine`, `sd.spatial.KDTree`). `sd.spatial.distance`는 없다(`AttributeError`; 별칭을 두지 않는다). scipy의 `from scipy.spatial.distance import cdist`는 `from stardust.spatial import cdist`다. 평평하게 해도 이름이 겹치지 않는다.
- **이름은 scipy.spatial의 것이다**(SPT4): `pdist`, `cdist`, `squareform`, 거리 함수, `KDTree`와 그 메서드(`query`, `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, `sparse_distance_matrix`). scipy에 없는 한 번에 끝나는 이웃 찾기는 sklearn `NearestNeighbors`의 두 메서드 이름 `kneighbors`, `radius_neighbors`다(SPT7).
- **데이터는 위치로만, 옵션은 키워드로만** 받는다(API.md, 다른 영역과 같다). `metric`과 그 인자(`p`, `w`, `V`, `VI`), `k`, `leafsize`도 키워드만이다(SPT15, `[scipy와 다름]`: scipy의 `pdist(X, 'cityblock')`, `tree.query(x, 3)`은 `TypeError`다). 거리가 필요로 하는 행렬은 거리 함수에서만 위치다(`seuclidean(u, v, V)`, `mahalanobis(u, v, VI)`; scipy와 같다).
- **별칭이 없다**: 거리 이름은 소문자 정식 이름 열넷만 받는다(`'e'`, `'euclid'`, `'cb'`, `'matching'`, `'cos'`, 대문자 이름은 `ValueError`). `cKDTree`, `distance_matrix`(scipy 1.18.0에서 폐기 예고, `cdist`와 같다), `minkowski_distance`는 없다.
- `workers`(SPT10; 스레드 수는 `sd.set_num_threads` 하나다), `eps`(SPT11; 결과는 언제나 정확하다), `boxsize`(SPT20), `balanced_tree`, `compact_nodes`, `copy_data`(SPT18), `return_sorted`, `output_type`(SPT17), 호출 가능한 `metric`, `out=`(SPT14)는 없다. 주면 모르는 키워드의 `TypeError`다.
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1): `pdist(): unexpected keyword argument 'workers'`, `query(): takes 1 positional argument but 2 were given`.

### §1.2 입력 (SPT12, SPT13, SPT23, SPT24)
- **데이터**(`X`, `XA`, `XB`, `u`, `v`, `data`, `x`, `r`의 배열)는 API.md §5의 상위 함수 입력이다: `sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `DataFrame`(2차원; 결측은 NaN). 계산은 float64다.
  - bool·int64는 float64로 바꾼다(linalg의 규칙과 같다; scipy도 바꾼다). 예외는 `hamming`·`jaccard`다: bool, int64, float64를 입력의 dtype 그대로 비교한다(가중치 없는 bool은 비트 묶음의 빠른 길; 개수가 정확해 어느 dtype이든 같은 값이면 같은 결과다). 그래서 2⁶⁰과 2⁶⁰ + 1은 다른 값이다(`[scipy와 다름]`: scipy는 float64로 바꾼 뒤 비교해 같은 값으로 본다). 두 입력의 dtype이 다르면 하나라도 float64면 float64, 아니면 int64로 맞춘다.
  - complex128은 `TypeError`다(`[scipy와 다름]`: scipy의 `pdist`·`cdist`는 `ValueError: Unsupported dtype complex128`). numpy의 정보 손실 없는 dtype은 `sd.asarray`의 규칙으로 넓히고, 문자열·object는 `TypeError`다. stardust에는 0차원 배열이 없으므로 0차원 numpy 배열은 `ValueError`(`KDTree(): 'data' must have at least 1 dimension`)이고, 파이썬 수 하나는 배열이 아니므로 `TypeError`다.
  - 모양: `pdist`는 (m, n), `cdist`는 (mA, n)과 (mB, n)이다. n이 다르면 `ValueError`다. n = 0(특징이 없음)도 받는다: 거리는 그 정의를 빈 합에 쓴 값이다(`euclidean`·`cityblock`·`chebyshev`는 0, `cosine`·`correlation`·`braycurtis`·`hamming`은 0/0의 NaN; scipy의 `pdist`와 같다. `[scipy와 다름]`: scipy의 함수 `chebyshev`는 n = 0에서 `ValueError`다). 행이 0이거나 1이면 `pdist`는 빈 배열이다.
- **유한하지 않은 값**: 거리 함수는 IEEE 규칙으로 퍼뜨린다(NaN → NaN, inf − inf → NaN). inf와 유한 값의 거리는 차의 합·최댓값 꼴(`euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`, `minkowski`, `seuclidean`)에서 inf이고, 비율 꼴(`cosine`, `correlation`, `canberra`, `braycurtis`: inf/inf)과 `mahalanobis`(이차식의 0·inf), `haversine`(sin(inf))에서는 NaN이다. `hamming`은 같고 다름만 세고 `jaccard`는 0인지만 보므로 inf도 보통 값이다. 이웃 찾기(`KDTree`, `kneighbors`, `radius_neighbors`)는 데이터와 질의점이 모두 유한해야 한다(`ValueError`: `KDTree(): 'data' must be finite`). NaN이 있으면 "가장 가까운"이 정해지지 않는다.
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
- 요청한 출력(`pdist`의 m(m−1)/2, `cdist`의 mA·mB, `kneighbors`의 mA·k)이 물리 메모리보다 크면 할당 전에 `MemoryError`다(API.md §2). 크기를 미리 알 수 없는 `KDTree`의 목록(`query_ball_point`, `query_ball_tree`, `query_pairs`, `sparse_distance_matrix`)은 찾은 짝이 물리 메모리를 넘을 수가 되면 찾기를 멈추고 `MemoryError`(`query_pairs(): the result is larger than this machine's physical memory`)다(docs/design/spatial.md SPT68).

### §1.5 scipy·sklearn과 다른 점 (모음)
- 거리: 별칭과 대문자 이름 없음, `metric`과 그 인자는 키워드만, 쓰지 않는 인자는 `TypeError`, complex는 `TypeError`, `p ≤ 0`·NaN `p`·`w`의 NaN·inf·0 이하의 `V`는 `ValueError`, `chebyshev`와 `canberra`는 NaN을 퍼뜨린다, `euclidean`·`minkowski`는 중간값이 넘쳐도 유한한 거리다, 모든 거리 함수가 (…, n) 배치를 받는다, `hamming`·`jaccard`는 int64를 그대로 비교한다, 호출 가능한 `metric`과 `out=` 없음.
- 이웃: 동점은 (거리, 색인), 데이터는 언제나 복사, `leafsize`만, `k`·`leafsize`는 int만, `k=0`·`k=[0]`·음수·NaN 반경과 상한·NaN 무게는 `ValueError`, 반경 결과는 `RadiusResult`이고 언제나 색인 순서, `query_pairs`는 정렬된 배열(scipy의 `output_type='ndarray'`도 정렬되어 있지 않다), `sparse_distance_matrix`는 `csr_array`, `workers`·`eps`·`boxsize`·`return_sorted`·`output_type` 없음, `kneighbors`의 거리는 언제나 직접 계산(`[sklearn과 다름]`: sklearn의 전개식은 1e8 + [0, 1)³의 점들에서 거리를 모두 0으로, 가장 가까운 점을 20개 가운데 하나도 맞히지 못했다; docs/design/spatial.md §3.2).

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
- **`V`**(`seuclidean`): 기본은 `pdist`가 `X`, `cdist`가 `XA`와 `XB`를 세로로 이은 것의 열 분산(`correction=1`)이다(scipy와 같다). 값이 모두 같은 열이 있으면(행이 둘 이상일 때; 계산된 분산이 평균의 반올림 때문에 0이 아닌 아주 작은 수여도) 또는 계산된 분산이 0이면 `ValueError`다(`pdist(): the variance of a column of the data is 0, so the default 'V' is not positive; give 'V'`; `[scipy와 다름]`: scipy는 0/0으로 모든 거리가 NaN이다). 자료에 NaN·inf가 있으면 그 열의 분산이 NaN이라 모든 거리가 NaN이다. 받은 `V`는 1차원 길이 n이고 유한한 양수여야 한다(`[scipy와 다름]`: scipy는 음수 `V`에서 뜻 없는 값을, 0에서 inf를 낸다).
- **`VI`**(`mahalanobis`): 기본은 같은 자료의 공분산의 역행렬이다(scipy와 같다; 관측이 차원 이하이면 `ValueError`: `pdist(): the default 'VI' needs more observations than columns, not 3 observations of 3 columns; give 'VI'`; 공분산이 특이하면 `sd.linalg.LinAlgError`; 자료에 NaN·inf가 있으면 모든 거리가 NaN이다). 받은 `VI`는 (n, n)이다. 양정치가 아니면 이차식이 음수인 짝의 거리가 NaN이다(scipy와 같다).
- **배치**: 모든 거리 함수가 (…, n) 배열을 받아 앞 축끼리 브로드캐스트하고, 결과는 앞 축의 모양이다(1차원 둘이면 Python `float`). 짝 하나는 `cdist`의 그 짝과 비트까지 같다. `[scipy와 다름]`: scipy 1.18.1은 `minkowski`·`euclidean`·`sqeuclidean`만 배치를 받는다.

**거리의 정의** (z = x − y를 원소마다 한 번 반올림한 것)
| 이름 | 정의 | `w`가 있을 때 (scipy와 같은 뜻) | 경계의 값 |
|---|---|---|---|
| `euclidean` | ‖z‖₂ | √(Σ wᵢzᵢ²) | 제곱합이 넘쳐도 비율을 맞춰 유한한 값(`[scipy와 다름]`: scipy와 R은 좌표 1e200에서 inf) |
| `sqeuclidean` | Σ zᵢ² | Σ wᵢzᵢ² | 참값이 넘치면 inf |
| `cityblock` | Σ \|zᵢ\| | Σ wᵢ\|zᵢ\| | |
| `chebyshev` | maxᵢ \|zᵢ\| | wᵢ > 0인 i의 최댓값 | NaN을 퍼뜨린다(`[scipy와 다름]`: scipy의 `cdist`는 NaN을 건너뛰어 `[nan, 0]`과 `[1, 5]`의 거리가 5다) |
| `minkowski` | (Σ \|zᵢ\|ᵖ)^(1/p); p = 1, 2, inf는 `cityblock`, `euclidean`, `chebyshev`와 비트까지 같다 | (Σ wᵢ\|zᵢ\|ᵖ)^(1/p) | 넘침·언더플로는 최댓값으로 나눠 비율을 맞춘다(`[scipy와 다름]`: scipy는 (0, 0)과 (3, 4)의 p = 2000 거리가 inf) |
| `seuclidean` | √(Σ zᵢ²/Vᵢ) | 없음 | |
| `mahalanobis` | √(zᵀ VI z) | 없음 | 이차식이 음수면 NaN |
| `cosine` | 1 − x·y/(‖x‖‖y‖), [0, 2]로 자른다. 정규화한 행의 ‖x̂ − ŷ‖²/2로 구한다(거의 평행한 행에서도 작은 거리의 자릿수가 남는다; 정확히 반대인 행은 2와 1 ULP 안) | 가중 내적과 가중 노름 | 영벡터가 있으면 NaN |
| `correlation` | 행마다 (가중) 평균을 뺀 뒤의 `cosine`(`centered=False`면 빼지 않는다) | 가중 평균 Σwx/Σw | 상수 행은 NaN |
| `canberra` | Σ \|zᵢ\|/(\|xᵢ\| + \|yᵢ\|), 0/0 항은 0 | Σ wᵢ(…) | NaN을 퍼뜨린다(`[scipy와 다름]`: scipy의 함수 `canberra`는 NaN 항을 건너뛰고 `pdist`는 퍼뜨린다) |
| `braycurtis` | Σ\|zᵢ\| / Σ\|xᵢ + yᵢ\| | 두 합 모두 wᵢ를 곱한다 | 0/0은 NaN, 분모만 0이면 inf |
| `hamming` | (Σ wᵢ[xᵢ ≠ yᵢ]) / Σ wᵢ, 기본 wᵢ = 1 | 같은 식 | NaN끼리는 다른 값으로 센다 |
| `jaccard` | Σ wᵢ[(xᵢ≠0) xor (yᵢ≠0)] / Σ wᵢ[(xᵢ≠0) or (yᵢ≠0)] | 같은 식 | 분모가 0이면 0. NaN은 0이 아니다 |
| `haversine` | 2·asin(√(sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2))), 입력은 (위도, 경도) 라디안의 2열, 결과는 단위 구의 중심각(라디안) | 없음 | 열이 2가 아니면 `ValueError`. h ≤ 1/2이면 2·asin(√h), 아니면 π − 2·asin(√(1 − h))이고 1 − h를 상쇄 없는 꼴로 구해 대척점에서도 자릿수를 지킨다(sklearn의 asin 꼴은 대척점 근처에서 잃는다). 경도의 차이는 반올림 오차까지 정확히 다뤄 날짜 변경선을 사이에 둔 가까운 두 점도 자릿수를 지킨다(sklearn은 경도 −π와 π 근처의 두 점이 거리 10⁻¹³일 때 상대 4×10⁻³을 잃는다) |

- `haversine`은 sklearn `haversine_distances`의 뜻이다(SPT9; scipy에는 없다). 이웃 찾기(`kneighbors`, `radius_neighbors`)에서는 점을 단위 구 위의 3차원 점으로 옮긴 KD 트리가 후보를 고르고, 돌려주는 거리와 모든 판정은 이 거리 자체다(§3). `KDTree`는 Minkowski 계열만 받는다.
- **`squareform`**: scipy와 같다(SPT16). 벡터 → 행렬은 길이가 n(n−1)/2가 아니면 `ValueError`, 빈 벡터는 (1, 1)의 0이다. 행렬 → 벡터는 `checks=True`면 정확한 대칭과 0 대각을 검사한다. NaN이 대칭 검사에 걸리면 메시지가 NaN을 말한다(`squareform(): 'X' has NaN off the diagonal, so it is not symmetric`; scipy의 메시지는 "must be symmetric"뿐이다). (1, 1)과 (0, 0)은 빈 벡터다. `force`는 `'no'`, `'tomatrix'`, `'tovector'`다. dtype을 지키고, 3차원 이상은 `ValueError`다.

## §3. 한 번에 끝나는 이웃 찾기 (SPT2)
```
kneighbors(XA, XB, /, *, k=1, metric='euclidean', p=None, w=None, V=None, VI=None, distance_upper_bound=inf)
radius_neighbors(XA, XB, r, /, *, metric='euclidean', p=None, w=None, V=None, VI=None, return_length=False)
```
- `XA`의 행마다 `XB`에서 가장 가까운 k개, 또는 거리가 r 이하인 모두를 찾는다. 전체 거리 행렬을 만들지 않는다(작업 메모리는 입력과 무관한 상한 안이다; 반경의 결과만 찾은 짝의 수에 비례한다).
- **입력**: `XA`와 `XB`는 `cdist`와 같다(2차원 (mA, n)과 (mB, n), n이 같아야 하고, 점 하나도 (1, n)이다). 거리 이름과 그 인자(`p`, `w`, `V`, `VI`)의 규칙과 기본 `V`·`VI`(`XA`와 `XB`를 세로로 이은 자료)도 `cdist`와 같다. 데이터는 모두 유한해야 한다(`ValueError`: `kneighbors(): 'XA' must be finite`; SPT12. sklearn도 NaN을 막는다).
- **결과는 정의로 정해진다**: `kneighbors(XA, XB, k=k)`의 행 i는 `cdist(XA, XB)`의 행 i에서 거리가 `distance_upper_bound`보다 작은 원소를 (거리, 열 번호)로 정렬해 앞 k개를 고른 것이고, 거리는 그 원소와 비트까지 같다. 모자라는 자리는 (inf, mB)로 채운다. `radius_neighbors(XA, XB, r)`의 행 i는 그 행에서 거리가 r 이하인 열 번호 전부(오름차순)와 그 거리다. Minkowski 계열(`euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`, `minkowski` p ≥ 1)은 `KDTree(XB).query(XA, k=k, p=…)`와도 같다. `radius_neighbors(XA, XB, r)`는 `KDTree(XB).query_ball_point(XA, r)`, `KDTree(XA).query_ball_tree(KDTree(XB), r)`와 같다.
  - 그래서 거리가 inf이거나 NaN인 짝은 `kneighbors`에 나오지 않는다(기본 상한 inf에서도 d < inf가 아니다): 넘치는 `sqeuclidean`, 영벡터의 `cosine`, 0/0의 `braycurtis`, 양정치가 아닌 `VI`의 `mahalanobis` 같은 것이다. `radius_neighbors`는 r = inf면 거리가 inf인 짝을 넣고(inf ≤ inf), NaN인 짝은 어떤 r에도 넣지 않는다.
  - 계산하는 길(전수, GEMM 고르기, `XB`로 만든 KD 트리, `haversine`은 `XB`의 점을 단위 구 위의 3차원 점으로 옮긴 KD 트리)은 거리 이름과 모양으로만 고르고 결과를 바꾸지 않는다. 스레드 수와 Accelerate의 스레드 모드와도 무관하다(§1.3).
  - `haversine`의 트리는 3차원 점 사이의 현(chord) 거리로 후보를 고르고, 돌려주는 거리와 경계·순서의 판정은 `cdist(metric='haversine')`의 그 원소다. 반올림 뒤에는 현 거리와 `haversine`의 순서가 어긋날 수 있어, 그 어긋남보다 넉넉한 여유로 후보를 고르므로 참 이웃과 동점이 빠지지 않는다(docs/design/spatial.md §3.4). 위도가 [−π/2, π/2]를 벗어나거나 경도의 크기가 8을 넘는 점이 있으면(도 단위를 라디안으로 잘못 준 자료 같은 것) 전수로 간다(결과는 같고 느리다).
- **`k`**: `KDTree.query`(§4)와 같다. 1 이상의 int거나 1 이상의 int 시퀀스(1부터 센 순위, 순서와 반복을 그대로 쓴다; scipy의 뜻)다. 0, 음수, 빈 시퀀스는 `ValueError`, bool·실수는 `TypeError`다. k가 int면 결과는 (mA, k)이고 k = 1이면 (mA,)다. 시퀀스면 (mA, len(k))다. k > mB의 자리와 mB보다 큰 순위는 (inf, mB)다(scipy와 같다; sklearn은 에러다). 요청한 결과가 물리 메모리보다 크면 `MemoryError`다.
- **`distance_upper_bound`**: 거리가 이 값보다 **작은** 이웃만이다(엄격한 부등호). 0 이상의 실수나 inf다(NaN·음수는 `ValueError`, 수가 아니면 `TypeError`; §4).
- **`r`**: 0 이상의 실수나 inf, 또는 그런 수의 배열로 (mA,)로 브로드캐스트되는 것((mA,)이나 (1,)이나 0차원)이다. 음수·NaN은 `ValueError`, bool과 수가 아닌 것은 `TypeError`다(`[scipy와 다름]`은 §4의 `query_ball_point`와 같다). 경계를 넣는다(d ≤ r). 판정하는 거리는 돌려주는 거리 그 자체다.
- **결과**: `kneighbors`는 `QueryResult(distances, indices)`(쓰기 금지의 float64와 int64 배열; `d, i = kneighbors(XA, XB)`로 풀린다), `radius_neighbors`는 `RadiusResult`(모양 (mA, mB); §5)다. `return_length=True`면 행마다의 개수(int64 (mA,), 보통의 새 배열)다.
- sklearn의 `pairwise_distances_argmin_min(X, Y)`는 `kneighbors(X, Y)`의 `(distances, indices)`다(풀기 순서만 반대). `[sklearn과 다름]`: 거리가 언제나 직접 계산한 값이다(§1.5). sklearn의 `NearestNeighbors(...).fit(Y).radius_neighbors(X)`는 행마다 object 배열이고 정렬하지 않는다(`sort_results=False`); sd는 CSR 꼴이고 언제나 색인 순서다.

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
- **만들기**: `data`는 (n, m)이다(1차원과 3차원 이상은 `ValueError`; scipy와 같다). 데이터는 언제나 복사한다(`[scipy와 다름]`: scipy는 복사하지 않을 수 있어 만든 뒤 원본을 고치면 답이 틀린다). n = 0(질의는 모두 채움 값)과 m = 0(모든 거리가 0; scipy는 질의에서 `IndexError`)도 받는다. 트리는 언제나 가장 넓은 축의 중앙값으로 나누고 노드 상자를 실제 점의 범위로 줄인다. 만들기는 결정적이다: 트리는 데이터와 `leafsize`로만 정해지고 스레드 수와 무관하다. 데이터에 NaN·inf가 있으면 `ValueError`(`KDTree(): 'data' must be finite`), complex는 `TypeError`다.
- **`leafsize`**: 1 이상의 int다(bool·실수는 `TypeError`; `[scipy와 다름]`: scipy는 2.5를 2로 자른다). 결과는 `leafsize`와 무관하고 빠르기만 바뀐다. 기본 10은 SPT6에서 6–32를 재서 1.2배 넘게 빠른 값이 없어 그대로 두었다(SPT78).
- **속성**: `data`(쓰기 금지 (n, m) float64, 입력 순서; 처음 읽을 때 트리의 사본에서 한 번 만들고 그 뒤로는 복사하지 않는다), `n`, `m`, `leafsize`(int), `maxes`, `mins`(쓰기 금지 (m,); n = 0이면 0, scipy와 같다), `size`(노드 수, int). 점이 `leafsize`보다 많은 노드는 같은 점뿐이어도 나누므로 `size`는 n과 `leafsize`로만 정해진다(`[scipy와 다름]`: scipy는 폭이 0인 노드를 나누지 않아 같은 점 50개에서 scipy 1, sd 15). scipy의 `tree`, `indices`, `boxsize`는 없다. `repr`는 `KDTree(n=1000, m=3, leafsize=10)`이다. 피클과 `copy.copy`·`copy.deepcopy`는 같은 데이터와 `leafsize`로 다시 만든 같은 트리다.
- **`p`**: 1 이상의 실수나 inf다(1 미만과 NaN은 `ValueError`; scipy와 같다). p = 1, 2, inf는 `cityblock`, `euclidean`, `chebyshev`와, 그 밖의 p는 `minkowski(p=)`와 같은 커널이다.
- **`query(x)`**: 데이터 점 가운데 x에 가장 가까운 k개다.
  - x는 (…, m)이다. 마지막 축이 m이 아니면 `ValueError`.
  - `k`는 1 이상의 int거나 1 이상의 int 시퀀스(list, tuple, 1차원 정수 배열; 1부터 센 순위의 목록이고 순서와 중복은 자유, scipy의 뜻)다. 0, 음수, 빈 시퀀스는 `ValueError`, bool·실수는 `TypeError`, int64를 넘으면 `OverflowError`다(`[scipy와 다름]`: scipy는 `k=True`와 `k=2.0`을 받고, `k=0`은 numpy 내부 메시지의 `ValueError`이며, `k=[0]`은 프로세스를 죽인다).
  - 결과는 `QueryResult(distances, indices)`이고 `d, i = tree.query(x)`로 풀린다. 모양은 scipy와 같다: k가 int면 (…, k)이고 k = 1이면 마지막 축을 없앤다. k가 시퀀스면 (…, len(k))다. x가 (m,)이고 k = 1이면 Python `float`와 `int`다(scipy는 `float`와 `numpy.int64`). 질의점에 NaN·inf가 있으면 `ValueError`(`query(): 'x' must be finite`)다.
  - 각 행은 (거리, 색인) 순서다(§1.3). 결과는 정의로 정해진다: `cdist(x, data)`(p = 1, 2, inf는 `cityblock`·`euclidean`·`chebyshev`, 그 밖의 p는 `metric='minkowski', p=p`)의 각 행을 (거리, 색인)으로 정렬해 상한보다 작은 앞 k개를 고른 것과 같고, 거리는 그 원소와 비트까지 같다. `leafsize`와 스레드 수는 빠르기만 바꾼다.
  - **채움**: 찾은 이웃이 k보다 적으면(k > n, 또는 `distance_upper_bound`) 남은 자리는 거리 inf, 색인 n이다(scipy와 같다; sklearn은 k > n이면 에러다).
  - **`distance_upper_bound`**: 거리가 이 값보다 **작은** 이웃만이다(엄격한 부등호; scipy와 같다). 0 이상의 실수나 inf여야 한다(`[scipy와 다름]`: scipy는 상한을 제곱해 비교해 −1이 1처럼 동작하고, NaN이면 모든 자리가 채움 값이다).
- **`query_ball_point(x, r)`**: 거리가 r **이하**인 데이터 점이다(경계를 넣는다; scipy와 같다). r은 0 이상의 실수나 inf, 또는 x의 배치 모양을 바꾸지 않고 브로드캐스트되는 배열이다(numpy의 규칙; 점 하나 (m,)에 (1,)은 받지 않는다, scipy와 같다). 음수·NaN은 `ValueError`다(`[scipy와 다름]`: scipy는 r = −1을 r = 1처럼, NaN을 빈 목록으로 다룬다). 판정은 돌려주는 거리 그 자체다(scipy는 제곱의 자리에서 비교해, 돌려줄 거리가 r과 같은 짝의 판정이 무작위 시험의 23%에서 √의 자리와 갈렸다).
  - 결과는 `RadiusResult`다(§5). x의 배치 축은 C 순서로 펼쳐 행이 된다. 점 하나 (m,)도 행 하나의 결과다(`[scipy와 다름]`: scipy는 점 하나면 list, 여럿이면 list의 object 배열이다).
  - 행 안은 색인 오름차순이다(`[scipy와 다름]`: scipy의 기본은 점 하나면 정렬하지 않고 여럿이면 정렬한다).
  - `return_length=True`면 행마다의 개수를 int64 배열(x의 배치 모양; 점 하나면 Python `int`)로 돌려준다(scipy와 같다).
  - 결과는 정의로 정해진다: `radius_neighbors(x, data, r)`(p = 1, 2, inf는 `metric='cityblock'`·`'euclidean'`·`'chebyshev'`, 그 밖의 p는 `metric='minkowski', p=p`)와 바이트까지 같다. 질의점에 NaN·inf가 있으면 `ValueError`(`query_ball_point(): 'x' must be finite`)다.
- **`query_ball_tree(other, r)`**: `self`의 점마다 `other`의 점 가운데 거리가 r 이하인 것이다. 결과는 `RadiusResult`이고 모양은 (self.n, other.n)이며 `radius_neighbors(self.data, other.data, r)`와 바이트까지 같다. r은 0 이상의 실수 하나나 inf다. `other`는 `KDTree`이고(아니면 `TypeError`: `query_ball_tree(): 'other' must be a KDTree, not list`) `other.m`이 다르면 `ValueError`(`query_ball_tree(): 'other' must have the same m as this tree (3), not 2`)다.
- **`query_pairs(r)`**: i < j이고 거리가 r 이하인 짝이다. (k, 2) int64 배열이고 (i, j)의 사전 순서다(`[scipy와 다름]`: scipy의 기본은 순서 없는 `set`이고 `output_type='ndarray'`도 scipy 순회의 순서다). `pdist(data)`의 원소가 r 이하인 짝 (i, j)와 같다. r은 실수 하나다. 보통의 새 배열이라 쓸 수 있다.
- **`count_neighbors(other, r)`**: `self`의 점 x₁과 `other`의 점 x₂의 순서쌍 가운데 거리가 r 이하인 것의 수다(`other`가 `self`면 (i, i)와 (i, j)·(j, i)를 모두 센다; scipy와 같다). r은 실수 하나나 1차원 배열이다. `cumulative=False`면 r은 줄지 않아야 하고 결과의 i번째는 r[i−1] < d ≤ r[i]의 수다(첫째는 d ≤ r[0]). `cumulative=True`의 r은 순서와 중복이 자유다. r은 0 이상이거나 inf이고 음수·NaN은 `ValueError`, 2차원 이상도 `ValueError`다(`[scipy와 다름]`: scipy는 r = NaN에서 모든 짝을, 음수 r에서도 짝을 센다). `weights`는 scipy의 꼴이다(배열 하나는 `other`가 `self`일 때만, 튜플 `(w_self, w_other)`의 한쪽은 None일 수 있고 `(None, None)`은 가중치 없음). 가중치는 1차원, 점마다 하나, 유한해야 한다(음수는 받는다; `[scipy와 다름]`: scipy는 NaN 무게로 NaN을 낸다). 가중이면 짝의 가중치 곱의 합이고 순서는 데이터와 트리로만 정해진다(SPT25; 스레드 수와 무관하다). 결과는 r이 실수 하나(numpy 스칼라와 0차원 배열도)면 Python `int`(가중이면 `float`), 배열이면 int64(가중이면 float64) 배열이다(scipy와 같다). 정수 셈은 `cdist(self.data, other.data)`(p의 거리)의 원소 가운데 r 이하인 것의 수와 정확히 같다.
- **`sparse_distance_matrix(other, max_distance)`**: 거리가 `max_distance` 이하인 짝의 거리를 담은 `sd.sparse.csr_array` (self.n, other.n)이다. `max_distance`는 0 이상의 실수 하나나 inf다. `self.query_ball_tree(other, max_distance).tocsr()`와 같다. 거리가 0인 짝은 저장된 0이다(§5). `[scipy와 다름]`: scipy의 기본은 `dok_matrix`이고 `output_type`으로 형식을 고른다. 형식을 바꾸려면 `asformat`을 쓴다.

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
  - `repr`는 한 줄이다: `RadiusResult(shape=(1000, 5000), nnz=10234)`. `summary()`는 없다. 피클할 수 없다(`sd.ndarray`와 같다).
- 결과 객체가 아닌 결과(`pdist`, `cdist`, `squareform`, `query_pairs`, `count_neighbors`, `return_length`의 개수)는 보통의 새 배열이라 쓸 수 있다.

## §6. 계산 기하: 공통 규칙

### §6.1 이름과 호출 모양
- `Delaunay`(삼각분할), `ConvexHull`(볼록 껍질), `Voronoi`(보로노이 그림) 세 클래스가 `sd.spatial`에 바로 있다(`sd.spatial.Delaunay`). 클래스와 속성의 이름은 scipy의 것이다. 안은 직접 쓴 C++이고 Qhull을 쓰지 않는다(SPT90).
- 생성자는 점만 위치로 받는다. Qhull의 옵션(`qhull_options`, `furthest_site`, `incremental`)과 메서드 `add_points`, `close`는 없다(SPT99). 주면 모르는 키워드의 `TypeError`다(`Delaunay(): unexpected keyword argument 'qhull_options'`). 점을 키워드로 주어도 `TypeError`다(`Delaunay(): got some positional-only arguments passed as keyword arguments: 'points'`).
- 차원은 2와 3뿐이다(SPT101). 한 클래스가 둘을 받는다.
- 에러의 꼴은 §1.4와 같다: 생성자의 에러는 클래스 이름(`Voronoi(): `), 메서드는 메서드 이름(`find_simplex(): `)으로 시작하고 경고는 내지 않는다.
- 스레드 수는 `sd.set_num_threads` 하나다. 같은 점이면 스레드 수와 상관없이 모든 배열이 같은 바이트다.
- 속성의 배열은 쓰기 금지다(API.md §2.2). 객체에 새 속성을 붙일 수 없다(`AttributeError`). `repr`는 한 줄이다: `Delaunay(n=1000, d=2, nsimplex=1980)`, `ConvexHull(n=1000, d=2, nvertices=18)`, `Voronoi(n=1000, d=2, nvertices=1980)`. 피클과 `copy.copy`·`copy.deepcopy`는 점으로 다시 만든 같은 객체다(결과가 점 집합으로만 정해지므로; SPT105).

### §6.2 입력 (SPT96–SPT98, SPT100)
- `points`는 (n, d) 배열이고 d = 2 또는 3이다. API.md §5의 상위 함수 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `DataFrame`)을 받고 float64로 계산한다. bool·int64는 float64로 바꾼다.
- complex128은 `TypeError`다(`Delaunay(): 'points' must be real, not a complex128 array`; `[scipy와 다름]`: scipy는 허수부를 버리고 `ComplexWarning`을 낸다). 문자열·object도 `TypeError`다.
- 2차원 배열이 아니면 `ValueError`(`Delaunay(): 'points' must have 2 dimensions, not 1`), 열이 2·3개가 아니면 `ValueError`다(`Delaunay(): 'points' must have 2 or 3 columns, not 4`; `[scipy와 다름]`: scipy는 4차원 이상도 받는다).
- NaN·inf는 `ValueError`다(`Delaunay(): 'points' must be finite`). scipy는 NaN을 `ValueError`로, inf를 `QhullError`(QH6019)로 막는다.
- **퇴화 입력**(SPT97): 서로 다른 점이 d + 1개보다 적으면(점이 없을 때도) `ValueError`(`Delaunay(): needs at least 3 distinct points, got 2`; d = 3은 `at least 4`), d = 2에서 모든 점이 한 직선 위면 `Delaunay(): 'points' are collinear`, d = 3에서 모든 점이 한 평면(한 직선 포함) 위면 `Delaunay(): 'points' are coplanar`다. 판정은 정확하다(좌표를 유리수로 읽은 답): 거의 한 직선인 점도 퇴화가 아니면 삼각분할한다. scipy는 같은 자리에서 `QhullError`(QH6214, QH6154)다.
- 점은 언제나 복사한다. `points` 속성은 그 쓰기 금지 사본(C 순서)이다. `[scipy와 다름]`: scipy의 `Delaunay(P).points`는 `P`와 메모리를 함께 써서, 만든 뒤 `P`를 고치면 객체가 틀려진다.
- 좌표의 크기에 제한이 없다(SPT100): 1e300과 1e-300이 섞인 점, 비정상 수, `DBL_MAX` 둘레의 점에서도 조합 구조가 정확하다.
- **같은 점**(SPT96): 좌표가 같은 점(−0.0과 0.0은 같다)은 하나로 합친다. 가장 작은 색인이 대표이고, 삼각분할·껍질·보로노이의 꼭짓점은 대표뿐이다. 서로 다른 점의 수가 퇴화 판정의 점 수다.

### §6.3 결과의 꼴 (SPT93–SPT95)
- 색인은 int64이고(`[scipy와 다름]`: scipy는 int32) 없음은 −1이다.
- **표준 꼴**: 단체(삼각형, 사면체)는 양의 방향이다(d = 2는 반시계, d = 3은 det[b − a, c − a, d − a] > 0). 단체 안은 양의 방향을 지키는 차례 가운데 가장 작은 색인이 먼저이고(d = 3은 그다음 자리도 남은 돌리기 가운데 가장 작은 것), 단체의 차례는 행의 사전 순서다. 경계 면은 바깥을 향하고(d = 2는 내부가 왼쪽인 변, d = 3은 밖에서 볼 때 반시계) 가장 작은 색인이 먼저이며 행의 사전 순서다. 색인 목록은 오름차순이다.
- **동점**: 한 원(d = 3은 구) 위에 d + 2개 이상의 점이 있으면 삼각분할이 여럿인데, 좌표의 사전 순서로 정한 규칙 하나로 고른다. 그래서 삼각분할은 점 집합의 함수다: 입력의 행을 섞으면 색인만 바뀌고 기하는 같다. 볼록 껍질과 보로노이 그림은 이 규칙과 무관하다(점 집합의 참 껍질과 참 그림이다). `[scipy와 다름]`: scipy의 차례와 방향은 Qhull이 만든 순서라 약속이 없다(무작위 1,000점의 3차원 삼각분할에서 단체 6,370개 중 3,197개가 음의 방향이다).
- **정확성**: 어느 단체가 있는지, 어느 점이 껍질의 꼭짓점인지, 어느 점이 이웃인지, 평가점이 어느 단체 안인지는 입력 float64를 유리수로 읽은 정확한 답이다. 허용오차 인자가 없다. `[scipy와 다름]`: scipy는 가까운 점을 버리고(10⁸ + 0.1 × (30 × 30 격자)의 900점에서 896점) 거의 퇴화인 입력에서 Delaunay가 아닌 삼각형과 뒤집힌 삼각형을 낸다(10⁻¹³ 크기로 흔든 30 × 30 격자에서 삼각형 1,751개 중 17개가 뒤집혔다).

### §6.4 크기와 메모리
- 만들기 전에 점의 수로 어림한 작업 메모리가 이 기기의 물리 메모리보다 크면 `MemoryError`다(`Delaunay(): cannot allocate 360000000000 bytes: more than this machine's physical memory`; API.md §2).
- d = 3의 사면체 수는 입력에 따라 O(n)에서 O(n²)이다(균등 입력에서 약 6.7n, 모멘트 곡선 (t, t², t³) 위의 점은 n²에 비례). 그래서 `Delaunay`와 `Voronoi`는 만드는 동안 사면체를 세어 물리 메모리에 들지 않을 수가 되면 멈추고 `MemoryError`다(`Delaunay(): the triangulation is larger than this machine's physical memory`, `Voronoi(): …`). 사면체가 2²⁹개에 이를 때도 같다.
- 색인을 int32로 셀 수 없을 만큼 점이 많으면 `ValueError`다(`Delaunay(): too many points`).

## §7. `Delaunay`
```
Delaunay(points, /)
Delaunay.find_simplex(xi, /)
Delaunay.points
Delaunay.simplices
Delaunay.neighbors
Delaunay.convex_hull
Delaunay.vertex_to_simplex
Delaunay.vertex_neighbor_vertices
Delaunay.coplanar
Delaunay.transform
Delaunay.ndim
Delaunay.npoints
Delaunay.nsimplex
Delaunay.min_bound
Delaunay.max_bound
```
- 점 집합의 Delaunay 삼각분할이다: 모든 단체의 외접원(구)의 열린 안쪽에 점이 없고, 단체들이 서로 다른 점의 볼록 껍질을 겹치지 않게 덮으며, 서로 다른 모든 점이 꼭짓점이다. 한 원 위의 점처럼 답이 여럿이면 §6.3의 동점 규칙이 고른다.
- `points` (n, d) float64는 입력의 쓰기 금지 사본이다. `ndim`은 d, `npoints`는 n, `nsimplex`는 단체의 수 t이고 Python `int`다. `min_bound`·`max_bound`는 축마다의 가장 작은·큰 좌표 (d,) float64다.
- `simplices` (t, d + 1) int64: 표준 꼴의 단체다. d = 2에서 서로 다른 점이 m개이고 껍질 경계 위의 점이 b개면 t = 2m − 2 − b다. 변 위에 놓인 경계점도 꼭짓점이다(경계 변이 그 점에서 나뉜다).
- `neighbors` (t, d + 1) int64: `neighbors[i, k]`는 단체 i에서 꼭짓점 k의 맞은편 면을 함께 가진 단체이고, 없으면 −1이다(scipy와 같다).
- `convex_hull` (경계 면의 수, d) int64: 삼각분할의 경계 면(표준 꼴)이다. 변·면 위의 경계점에서 나뉘므로 `ConvexHull(points).simplices`와 다를 수 있다.
- `vertex_to_simplex` (n,) int64: 그 점을 꼭짓점으로 가진 가장 작은 번호의 단체다. 대표가 아닌 같은 점은 대표의 값이다.
- `vertex_neighbor_vertices`: `(indptr, indices)` int64 짝이다. 점 i의 이웃 꼭짓점(삼각분할의 변으로 이어진 점)은 `indices[indptr[i]:indptr[i + 1]]`이고 오름차순이다(scipy는 차례를 정해 두지 않는다). 대표가 아닌 같은 점의 행은 비어 있다. 두 배열은 CSR의 것이라 `sd.sparse.csr_array((data, indices, indptr), shape=(n, n))`에 그대로 들어간다.
- `coplanar` (k, 3) int64: 삼각분할에 들지 않은 점의 행 (점, 단체, 꼭짓점)이고 scipy의 꼴이다. sd에서 삼각분할에 들지 않는 점은 대표가 아닌 같은 점뿐이다: 꼭짓점 열은 대표의 색인, 단체 열은 `vertex_to_simplex[대표]`이고 행은 점의 오름차순이다. `[scipy와 다름]`: scipy는 가까운 점도 버린다(§6.3; 10⁶ + 10⁻³ × (균등 500점)에서 486점을 `coplanar`로 돌리고 단체 12개만 남긴다).
- `transform` (t, d + 1, d) float64: scipy의 뜻이다. `transform[i, :d, :]`는 T⁻¹(T의 열은 꼭짓점 k − 꼭짓점 d, k < d), `transform[i, d, :]`는 꼭짓점 d다. 점 x의 무게중심 좌표는 c = T⁻¹(x − r)과 1 − Σc다. 평평한 단체가 없으므로 NaN이 없다(scipy는 평평한 단체에 NaN을 적는다).
- `simplices`, `neighbors`, `vertex_to_simplex`는 만들 때 만든다. `convex_hull`, `vertex_neighbor_vertices`, `coplanar`, `transform`, `min_bound`, `max_bound`는 처음 읽을 때 만들고 쥔다(d = 2, 10⁶점의 `transform`은 96 MB다).
- **`find_simplex(xi)`**: 평가점을 품은 **닫힌** 단체의 번호다. 변이나 꼭짓점 위에 있어 여러 단체가 품으면 가장 작은 번호다. 볼록 껍질 밖(경계 위는 안이다)과 NaN·inf 좌표의 점은 −1이다(scipy와 같다).
  - `xi`는 (…, d)인 상위 함수 입력이다. 마지막 축이 d가 아니면 `ValueError`(`find_simplex(): 'xi' must have 2 entries on its last axis, not 3`), complex는 `TypeError`(`find_simplex(): 'xi' must be real, not a complex128 array`)다.
  - 결과는 `xi`의 앞 모양을 가진 쓰기 가능한 새 int64 배열이고, 점 하나 (d,)는 Python `int`다(scipy는 0차원 int32 배열).
  - `[scipy와 다름]`: `tol`과 `bruteforce`가 없다(주면 `TypeError`). scipy는 무게중심 좌표의 허용오차 때문에 단위 정사각형의 변에서 10⁻¹⁵ 밖의 점을 안으로 치고(10⁻¹⁴ 밖은 밖), 두 단체가 함께 가진 변 위의 점에서 고르는 단체를 정해 두지 않는다.

```python
import stardust as sd

pts = [[0, 0], [2, 0], [0, 2], [2, 2], [1, 1.2]]
tri = sd.spatial.Delaunay(pts)          # Delaunay(n=5, d=2, nsimplex=4)
tri.simplices                           # [[0, 1, 4], [0, 4, 2], [1, 3, 4], [2, 4, 3]]
tri.find_simplex([[1.0, 0.5], [3.0, 0.0]])   # [0, -1]
tri.find_simplex([1.0, 0.5])                 # 0
```

## §8. `ConvexHull`
```
ConvexHull(points, /)
ConvexHull.points
ConvexHull.vertices
ConvexHull.simplices
ConvexHull.neighbors
ConvexHull.equations
ConvexHull.area
ConvexHull.volume
ConvexHull.ndim
ConvexHull.npoints
ConvexHull.nsimplex
ConvexHull.min_bound
ConvexHull.max_bound
```
- 점의 볼록 껍질이다. 삼각분할을 거치지 않고 직접 구하므로 `Delaunay`보다 훨씬 싸다(SPT108). 모든 속성을 만들 때 구한다.
- `vertices` (h,) int64: 극점(볼록 껍질의 꼭짓점)만이다. 변이나 면의 안쪽에 놓인 경계점과 대표가 아닌 같은 점은 들지 않는다(scipy의 기본과 같다: 정사각형의 변 위 점은 `vertices`에 없다). d = 2는 반시계이고 가장 작은 색인에서 시작한다. d = 3은 오름차순이다.
- `simplices` (면 수, d) int64: d = 2는 이웃한 극점을 잇는 변, d = 3은 삼각형이고 표준 꼴(§6.3)이다. 3차원의 평평한 면(꼭짓점이 넷 이상인 볼록 다각형)은 그 면의 가장 작은 색인에서 부채꼴로 나눈다. `nsimplex`는 면의 수다. `[scipy와 다름]`: scipy의 면은 방향이 섞여 있다(무작위 1,000점에서 2차원 변 16개 중 7개, 3차원 삼각형 142개 중 71개만 바깥을 향한다).
- `neighbors` (면 수, d) int64: `neighbors[i, k]`는 면 i에서 꼭짓점 k의 맞은편 이웃 면이다(scipy와 같다). 닫힌 곡면이라 −1이 없다.
- `equations` (면 수, d + 1) float64: 행은 (법선, 절편)이고 법선은 바깥을 향하는 단위 벡터다. 내부의 점 x에서 법선·x + 절편 < 0이다(scipy와 같다). 한 평평한 면에서 나온 삼각형들은 같은 행(같은 비트)이다.
- `area`, `volume`(Python `float`): scipy의 뜻이다. d = 3은 겉넓이와 부피, d = 2는 **둘레**와 **넓이**다. 결과가 float64를 넘을 때만 inf다.
- `min_bound`·`max_bound`는 축마다의 가장 작은·큰 좌표다. `ConvexHull`의 `simplices`와 `Delaunay.convex_hull`은 다를 수 있다(§7).
- `coplanar`, `good`(scipy의 기본 옵션에서 비어 있거나 None인 속성)은 없다.

## §9. `Voronoi`
```
Voronoi(points, /)
Voronoi.points
Voronoi.vertices
Voronoi.ridge_points
Voronoi.ridge_vertices
Voronoi.regions
Voronoi.point_region
Voronoi.ndim
Voronoi.npoints
Voronoi.min_bound
Voronoi.max_bound
```
- 점 집합의 참 보로노이 그림이다(SPT111). 한 원(구) 위의 d + 2개 이상의 점은 꼭짓점 하나를 함께 갖고, 길이(넓이)가 0인 능선은 없다. 3 × 3 격자의 꼭짓점은 4개, 능선은 12개다(scipy와 같다).
- `vertices` (꼭짓점 수, d) float64: Delaunay 단체의 외심이다. 외접원(구)이 같은 단체들은 꼭짓점 하나다. 번호는 그 외접원(구)을 가진 단체 가운데 가장 작은 번호(`Delaunay(points).simplices`의 행 번호)의 차례다. 외심이 float64를 넘으면 ±inf다(아주 큰 좌표의 거의 한 직선인 점에서만 난다).
- `regions`: `(indptr, indices)` int64 짝이고 영역 i는 점 i의 것이다(SPT110). `indices[indptr[i]:indptr[i + 1]]`가 그 영역의 꼭짓점 번호이고, 끝없는 영역(껍질 위의 점의 것)은 −1을 하나 갖는다. d = 2는 점 둘레의 반시계 차례(−1이 있으면 −1이, 없으면 가장 작은 번호가 먼저), d = 3은 오름차순이다. 대표가 아닌 같은 점의 행은 비어 있다.
- `point_region` (n,) int64: 점의 영역 번호다. 대표인 점은 자기 색인, 같은 점은 대표의 색인이다(scipy도 같은 점에 같은 영역을 준다). scipy 코드의 `vor.regions[vor.point_region[i]]`는 `indices[indptr[r]:indptr[r + 1]]`(`r = point_region[i]`)이다.
- `ridge_points` (능선 수, 2) int64: 능선을 함께 가진 점의 짝 (i, j), i < j이고 행의 사전 순서다. 대표가 아닌 같은 점은 나오지 않는다.
- `ridge_vertices`: `(indptr, indices)` int64 짝이고 능선 k의 꼭짓점 번호다. 끝없는 능선은 −1을 하나 가지며 그것이 먼저다. d = 2는 꼭짓점 둘(오름차순)이거나 −1과 하나다. d = 3은 `ridge_points[k, 0]`에서 `ridge_points[k, 1]` 쪽을 볼 때 반시계의 차례이고, −1이 없으면 가장 작은 번호가 먼저다.
- 능선은 두 영역이 d − 1차원의 면을 함께 가질 때만 있다. 길이(넓이)가 0인 것은 없다: 한 원 위의 네 점의 대각선, 평평한 껍질 면의 대각선(3 × 3 × 3 격자의 면 대각선; 능선은 축 이웃의 54개다)이 그렇다.
- `[scipy와 다름]`: scipy의 `regions`·`ridge_vertices`는 list의 list이고, `regions`에 빈 영역 하나가 끼어 있으며, 2차원 영역의 도는 방향이 섞여 있고(무작위 1,000점의 유한한 영역에서 반시계 489개, 시계 495개) 3차원 영역의 차례는 정해 두지 않는다. scipy는 3차원에서 유한한 꼭짓점이 하나뿐인 끝없는 능선을 빼놓는다(무작위 1,000점에서 sd의 능선 7,440개 중 23개). `ridge_dict`, `furthest_site`, `add_points`, `close`는 없다(`ridge_dict`는 `ridge_points`와 `ridge_vertices`로 만든다).

```python
import stardust as sd

pts = [[0, 0], [2, 0], [0, 2], [2, 2], [1, 1.2]]
vor = sd.spatial.Voronoi(pts)           # Voronoi(n=5, d=2, nvertices=4)
indptr, indices = vor.regions
r = vor.point_region[4]                 # 4: 영역 i는 점 i의 것이다
indices[indptr[r]:indptr[r + 1]]        # [0, 2, 3, 1]: 가운데 점의 유한한 영역, 반시계
indices[indptr[0]:indptr[1]]            # [-1, 0, 1]: 껍질 위의 점은 끝없는 영역
```

## §10. scipy·sklearn → sd.spatial

**scipy.spatial.distance**
| scipy | sd.spatial |
|---|---|
| `pdist`, `cdist`, `squareform` | 같은 이름(`metric`과 그 인자는 키워드만) |
| `euclidean sqeuclidean cityblock chebyshev minkowski seuclidean mahalanobis cosine correlation canberra braycurtis hamming jaccard` | 같은 이름, 모두 배치를 받는다 |
| 불 거리 `dice rogerstanimoto russellrao sokalsneath yule`, `jensenshannon`, `directed_hausdorff` | 없음(나중 후보) |
| `is_valid_dm`, `is_valid_y`, `num_obs_dm`, `num_obs_y` | 없음(`squareform`의 검사는 안에 있다) |
| 거리 이름의 별칭과 대문자, 호출 가능한 `metric` | 없음 |

**scipy.spatial**
| scipy | sd.spatial |
|---|---|
| `KDTree`, `KDTree.query` | 같은 이름 |
| `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, `sparse_distance_matrix` | 같은 이름 |
| `cKDTree` | 없음(`KDTree`와 같은 것) |
| `KDTree.tree`, `Rectangle`, `minkowski_distance`, `minkowski_distance_p` | 없음(트리의 속은 공개하지 않는다; 앞의 둘은 `minkowski`와 그 p제곱) |
| `distance_matrix` | 없음(`cdist`) |
| `Delaunay(points, furthest_site=False, incremental=False, qhull_options=None)` | `Delaunay(points, /)`(d = 2, 3) |
| `Delaunay.points`, `ndim`, `npoints`, `nsimplex`, `min_bound`, `max_bound` | 같은 이름 |
| `Delaunay.simplices`, `neighbors`, `convex_hull`, `vertex_to_simplex`, `coplanar` (int32) | 같은 이름, int64, 표준 꼴 |
| `Delaunay.vertex_neighbor_vertices` | 같은 이름, `(indptr, indices)`, 행 안은 오름차순 |
| `Delaunay.transform` | 같은 이름 |
| `Delaunay.find_simplex(xi, bruteforce=False, tol=None)` | `Delaunay.find_simplex(xi, /)` |
| `Delaunay.equations`, `paraboloid_scale`, `paraboloid_shift`, `good`, `furthest_site`, `plane_distance`, `lift_points`, `add_points`, `close` | 없음(Qhull의 올림 좌표와 옵션에 묶인 것) |
| `tsearch(tri, xi)` | `tri.find_simplex(xi)` |
| `ConvexHull(points, incremental=False, qhull_options=None)` | `ConvexHull(points, /)`(d = 2, 3) |
| `ConvexHull.points`, `ndim`, `npoints`, `nsimplex`, `min_bound`, `max_bound`, `vertices`, `simplices`, `neighbors`, `equations`, `area`, `volume` | 같은 이름 |
| `ConvexHull.coplanar`, `good`, `add_points`, `close` | 없음 |
| `Voronoi(points, furthest_site=False, incremental=False, qhull_options=None)` | `Voronoi(points, /)`(d = 2, 3) |
| `Voronoi.points`, `ndim`, `npoints`, `min_bound`, `max_bound`, `vertices`, `ridge_points`, `point_region` | 같은 이름 |
| `Voronoi.regions`, `ridge_vertices` (list의 list) | 같은 이름, `(indptr, indices)` 짝 |
| `Voronoi.ridge_dict`, `furthest_site`, `add_points`, `close` | 없음 |
| `SphericalVoronoi`, `HalfspaceIntersection`, d ≥ 4의 기하 | 없음(나중 후보) |
| `delaunay_plot_2d`, `convex_hull_plot_2d`, `voronoi_plot_2d` | 없음(그림은 `sd.plot`의 일) |
| `procrustes`, `geometric_slerp`, `transform`(`Rotation` 등) | 없음(나중 후보) |

**scikit-learn**
| sklearn | sd.spatial |
|---|---|
| `pairwise_distances_argmin_min(X, Y)` | `kneighbors(X, Y)`(풀기 순서가 `(distances, indices)`) |
| `pairwise_distances_argmin(X, Y)` | `kneighbors(X, Y).indices` |
| `NearestNeighbors(...).fit(Y).kneighbors(X)`, `.radius_neighbors(X)` | `kneighbors(X, Y, k=)`, `radius_neighbors(X, Y, r)`(클래스는 없다) |
| `radius_neighbors_graph(mode='distance')` | `radius_neighbors(X, Y, r).tocsr()` |
| `euclidean_distances(X, Y, squared=)` | `cdist(X, Y, metric='euclidean')`, `metric='sqeuclidean'` |
| `paired_distances` | 거리 함수의 배치(`sd.spatial.euclidean(U, V)`) |
| `haversine_distances` | `cdist(metric='haversine')`, `haversine(u, v)` |
| `sklearn.neighbors.KDTree` | `KDTree`(scipy의 API) |
| `BallTree` | 없음(나중 후보, SPT8). `BallTree(metric='haversine')`의 이웃 찾기는 `kneighbors(X, Y, metric='haversine')`, `radius_neighbors(X, Y, r, metric='haversine')`가 단위 구 위의 KD 트리로 한다(SPT83) |
| `kneighbors_graph`, `nan_euclidean_distances`, `KernelDensity`, `two_point_correlation` | 없음(짝 세기는 `count_neighbors`) |

## §11. 정확도
- 거리는 Arb로 구한 참 거리와 견준 계약 안이다(docs/design/spatial.md §4의 표; 판정은 `tests/spatial/refdata.py`). `chebyshev`와 가중치 없는 `hamming`·`jaccard`는 올바른 반올림이다. 참값 파일의 모든 짝에서 잰 최대는 `haversine` 1 ULP, `euclidean`·`seuclidean`·`mahalanobis` 2 ULP, `sqeuclidean`·`cityblock`·`canberra` 3 ULP, `braycurtis` 5 ULP(가중 7), `minkowski` 6 ULP(p = 0.5)이고, `cosine`·`correlation`은 절대와 상대를 섞은 한계다(참값이 0에 가까우면 ULP가 크다). 반올림 크기의 결과는 scipy와 끝자리가 다를 수 있다(API.md §4의 규칙 3). 같은 짝은 `pdist`, `cdist`, 거리 함수에서 같은 비트다.
- 이웃은 같은 커널의 전수 결과와 정확히 같고(길·스레드와 무관), 거리가 올바르게 반올림되는 데이터(정수·2진 격자)에서는 Arb로 정렬한 참 이웃과 정확히 같다. 일반 데이터에서는 참 거리가 1 ULP 안에서 다른 두 점의 순서가 돌려준 거리의 순서다. `haversine`의 이웃은 단위 구 위의 점 집합 여섯(균등, 거의 같은 점의 군집, 대척점, 극, 날짜 변경선, 거울상의 동점)에서 Arb의 참 거리로 계약 안의 참 이웃이다(계약의 0.096 이하; sklearn의 `BallTree`는 극의 집합에서 계약의 3.8×10¹⁴배).
- 계산 기하의 조합 구조(단체, 극점, 이웃, 능선, `find_simplex`의 답)는 정확하다(§6.3): 시험 점 집합 48개(무작위, 격자·원·구 위, 같은 점, 거의 퇴화, 먼 군집, 극단 범위)에서 모두 유리수 검사기의 답과 같다(scipy는 그 가운데 7개를 거부하고 14개에서 Delaunay 조건이나 "모든 점이 꼭짓점"을 어긴다). 실수 출력의 계약은 [docs/design/spatial_geometry.md](../design/spatial_geometry.md) §4의 표다(u = 2⁻⁵³). 넓이·부피·둘레·겉넓이는 γ_{s+2}–γ_{s+6}의 상대오차(s는 합이 거치는 덧셈 수), `equations`의 법선은 성분마다 8u, `transform`의 T⁻¹는 원소마다 4u, 보로노이 꼭짓점은 좌표마다 u\|v_k\| + 4u\|v_k − a_k\|(a는 그 단체의 첫 꼭짓점)이고, 한계가 단체의 모양(가늘거나 납작함)에 기대지 않는다. 참값(유리수와 Arb)과 견준 가장 큰 오차는 계약의 0.32(넓이·부피·둘레·겉넓이), 0.61(`transform`), 0.98(보로노이 꼭짓점)이다.
