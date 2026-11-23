# sd.spatial API

> 상태: **확정** (2026-11-23; 승인(위임)). §2–§5의 모든 이름이 구현되어 있다: 거리(SPT1), `kneighbors`·`radius_neighbors`와 결과 객체(SPT2), `KDTree`와 `query`(SPT3), 반경·짝·개수의 다섯 메서드(SPT4). 시그니처는 scipy 1.18.1의 `scipy.spatial`(`distance`, `KDTree`)의 이름과 인자 이름에 stardust의 규칙과 SPT1–SPT89의 결정을 입힌 것이다.

- 설계와 결정의 근거는 [docs/design/spatial.md](../design/spatial.md)(§1.1 SPT1–SPT89), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/spatial/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 클래스는 `name(...)`의 생성자), 클래스의 메서드(`Class.method(...)`)와 속성(`Class.attr`)이다. 모듈 수준의 이름이 공개 이름 전부이고, 시그니처가 블록과 같아야 한다.

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
  - bool·int64는 float64로 바꾼다(linalg의 규칙과 같다; scipy도 바꾼다). 예외는 `hamming`·`jaccard`다: bool, int64, float64를 입력의 dtype 그대로 비교한다(가중치 없는 bool은 비트 묶음의 빠른 길; 개수가 정확해 어느 dtype이든 같은 값이면 같은 결과다). 그래서 2⁶⁰과 2⁶⁰ + 1은 다른 값이다(`[scipy와 다름]`: scipy는 float64로 바꾼 뒤 비교해 같은 값으로 본다). 두 입력의 dtype이 다르면 하나라도 float64면 float64, 아니면 int64로 맞춘다.
  - complex128은 `TypeError`다(`[scipy와 다름]`: scipy의 `pdist`·`cdist`는 `ValueError: Unsupported dtype complex128`). numpy의 정보 손실 없는 dtype은 `sd.asarray`의 규칙으로 넓히고, 문자열·object는 `TypeError`다.
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
| `BallTree` | 없음(나중 후보, SPT8). `BallTree(metric='haversine')`의 이웃 찾기는 `kneighbors(X, Y, metric='haversine')`, `radius_neighbors(X, Y, r, metric='haversine')`가 단위 구 위의 KD 트리로 한다(SPT83) | — |
| `kneighbors_graph`, `nan_euclidean_distances`, `KernelDensity`, `two_point_correlation` | 없음(짝 세기는 `count_neighbors`) | — |

## §7. 정확도
- 거리는 Arb로 구한 참 거리와 견준 계약 안이다(docs/design/spatial.md §4의 표; 판정은 `tests/spatial/refdata.py`). `chebyshev`와 가중치 없는 `hamming`·`jaccard`는 올바른 반올림이다. 참값 파일의 모든 짝에서 잰 최대는 `haversine` 1 ULP, `euclidean`·`seuclidean`·`mahalanobis` 2 ULP, `sqeuclidean`·`cityblock`·`canberra` 3 ULP, `braycurtis` 5 ULP(가중 7), `minkowski` 6 ULP(p = 0.5)이고, `cosine`·`correlation`은 절대와 상대를 섞은 한계다(참값이 0에 가까우면 ULP가 크다). 반올림 크기의 결과는 scipy와 끝자리가 다를 수 있다(API.md §4의 규칙 3). 같은 짝은 `pdist`, `cdist`, 거리 함수에서 같은 비트다.
- 이웃은 같은 커널의 전수 결과와 정확히 같고(길·스레드와 무관), 거리가 올바르게 반올림되는 데이터(정수·2진 격자)에서는 Arb로 정렬한 참 이웃과 정확히 같다. 일반 데이터에서는 참 거리가 1 ULP 안에서 다른 두 점의 순서가 돌려준 거리의 순서다. `haversine`의 이웃은 단위 구 위의 점 집합 여섯(균등, 거의 같은 점의 군집, 대척점, 극, 날짜 변경선, 거울상의 동점)에서 Arb의 참 거리로 계약 안의 참 이웃이다(계약의 0.096 이하; sklearn의 `BallTree`는 극의 집합에서 계약의 3.8×10¹⁴배).
