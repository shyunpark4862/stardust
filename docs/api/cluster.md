# sd.cluster API

> 상태: **확정** (2026-11-24; 설계 결정 CL1–CL81). CL4 마일스톤(혼합, 2026-11-27)까지 마쳤다: §2·§3·§4·§5의 이름이 공개됐고 나머지 이름은 CL5·CL6 마일스톤에서 생긴다(tests/cluster/test_api_surface.py가 이름마다의 마일스톤을 쥔다). 이름과 시그니처는 [docs/design/cluster.md](../design/cluster.md) §9의 답(13번만 (b), 나머지 (a))을 따른다. 시그니처는 scikit-learn 1.9.1(`sklearn.cluster`, `sklearn.mixture`, `sklearn.metrics`), scipy 1.18.1(`scipy.cluster.hierarchy`), R 4.6.2의 `stats`와 `cluster` 2.1.8.3의 이름과 뜻에 stardust의 규칙과 CL1–CL40의 결정을 입힌 것이다. `fuzzy_cmeans`의 `tol`과 `kmedoids`의 `max_iter`는 CL2 마일스톤이 정했다(설계 문서 CL65·CL66). 남은 임시 값(`affinity_propagation`의 `metric='precomputed'`의 이름, `gap_statistic`의 `select` 이름)은 해당 마일스톤에서 정한다(표시가 그 자리에 있다).

- 설계와 결정의 근거는 [docs/design/cluster.md](../design/cluster.md)(§1.1 CL1–CL81), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/cluster/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 함수(`name(...)`), 결과 클래스(`Name`), 결과의 메서드(`Name.method(...)`)다. 블록의 모듈 수준 이름이 마일스톤을 마칠 때마다 `sd.cluster.__all__`에 들어가 CL6 마일스톤에서 그 전부가 된다.
- 절 끝의 "(§9의 n번)"은 그 줄을 정한 설계 문서의 질문이다(답은 §1.1의 CL4–CL40).

## §1. 공통 규칙

### §1.1 이름과 호출 모양 (CL1; §9의 1·2·11번)
- **이름공간은 평평하다**: 모든 공개 이름이 `sd.cluster`에 바로 있다. 하위 모듈은 없다. sklearn의 `cluster`·`mixture`·`metrics.cluster`와 scipy의 `cluster.hierarchy`가 한곳에 있다.
- **맞추는 함수는 소문자 이름이고 `<이름>Result`를 돌려준다**: `kmeans` → `KMeansResult`, `kmedoids` → `KMedoidsResult`, `agglomerative`·`divisive` → `HierarchyResult`, `dbscan` → `DBSCANResult`, `gaussian_mixture` → `GaussianMixtureResult` 등. 추정기 클래스(`KMeans(...).fit(X)`)는 없다(CL1). 맞춘 뒤의 일(새 점의 라벨, 자르기)은 결과의 메서드다.
- **scipy 이름의 계층 도구는 배열을 받고 배열을 돌려준다**: `linkage` → Z((n − 1, 4) float64), `fcluster`, `cut_tree`, `cophenet`, `leaves_list` 등(§2). `HierarchyResult.linkage`가 같은 Z다.
- **지표는 sklearn의 이름 그대로다**: `adjusted_rand_score`, `silhouette_samples` 등(§6). 수 하나는 Python `float`다.
- **인자의 순서는 (X, k)다**: `X`는 첫째 위치 인자, 군집 수 `k`는 둘째 위치 인자다(분할과 혼합 모형; 혼합 모형도 `n_components`가 아니라 `k`). 계층과 밀도 방법에는 `k`가 없다.
- **데이터는 위치로만, 옵션은 키워드로만** 받는다. 별칭이 없다: `method`, `metric`, `init`, `covariance_type`, `criterion`의 문자열은 소문자 정식 이름만 받는다. 틀린 호출 모양은 부른 이름으로 시작한다(API.md §2.1): `kmeans(): unexpected keyword argument 'n_clusters'`, 결과의 메서드는 `predict(): ...`.

### §1.2 입력 (§9의 4·8·9번)
- **받는 것**: API.md §5의 상위 함수 입력이다: `sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `DataFrame`(열이 변수)과 `Series`(한 변수). 점은 행이다: `X`는 (n, p), 1차원은 (n, 1)로 읽는다. 계산은 float64다. bool·정수는 float64로 바꾼다. 복소수는 `TypeError`다.
- **라벨 입력**(지표): 1차원의 정수, bool, 문자열 배열(sklearn과 같다). 두 라벨의 길이가 다르면 `ValueError`다.
- **미리 계산한 거리**: 거리로 정의되는 방법(`linkage`, `agglomerative`, `divisive`, `kmedoids`, `dbscan`, `hdbscan`, `optics`, `silhouette_samples`, `silhouette_score`, `affinity_propagation`)은 `metric='precomputed'`로 (n, n) 정사각(대칭, 대각 0, 음수·NaN 없음; `ValueError`) 또는 1차원 압축 꼴(길이 n(n − 1)/2)을 받는다. scipy 이름의 `linkage`만 scipy처럼 1차원 입력을 압축 꼴로 읽고, 2차원은 언제나 관측값이다. 거리의 이름(`'euclidean'`, `'sqeuclidean'`, `'cityblock'`, `'chebyshev'`, `'minkowski'` 등)과 값은 `sd.spatial.cdist`와 같다.
- **`nan_policy`**: `'raise'`(기본; `ValueError`)와 `'omit'`(NaN이 있는 행을 빼고 맞춘다; 그 행의 라벨은 −1, `nobs`는 뺀 뒤의 수, `omitted`가 뺀 행의 bool 배열). ±inf는 언제나 `ValueError`다. 평가점(`predict`)의 NaN은 `ValueError`다(라벨이 NaN이 될 수 없다).
- **무게**: `kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `dbscan`은 `weights=`(점의 무게; sklearn의 `sample_weight`)를 받는다. 유한한 0 이상이고 양수가 하나 이상이어야 한다.
- **입력 순서**: 결정적인 방법(계층, PAM, DBSCAN, HDBSCAN, OPTICS, AP)은 같은 점들을 어떤 순서로 주든 같은 분할이고 라벨은 입력 순서로 처음 나오는 차례다(§1.3). 무작위 방법도 같은 `rng` 상태면 입력 순서와 무관한 분할이다(표본을 좌표의 사전순 번호로 뽑는다).

### §1.3 결과의 공통 필드 (CL1; §9의 3번)
| 필드·메서드 | 뜻 |
|---|---|
| `labels` | 점의 군집 (n,) int64, 입력의 순서. 0 … K − 1, 잡음(밀도 방법)과 `'omit'`이 뺀 행은 −1. 번호는 입력 순서로 처음 나오는 차례다(첫 점의 군집이 0) |
| `n_clusters` | 군집 수 K(잡음은 세지 않는다) |
| `nobs` | 맞춤에 든 점의 수 |
| `omitted` | `'omit'`이 뺀 행 (n,) bool(없으면 모두 False) |
| `n_iter`, `converged` | 반복하는 방법만. 상한에서 멈추면 `converged=False`이고 에러가 아니다(API.md §2.2) |
| `objective` | 반복하는 방법의 목적값(k-평균의 관성, PAM의 평균 거리, EM의 로그우도) |
| `predict(X_new, /)` | 새 점의 라벨(중심·medoid·성분이 있는 결과만) |
| `summary()` | API.md §2.3의 상자: 머리에 방법, `nobs`, `n_clusters`, 목적값, 반복 수, 수렴, 조절 모수. 표는 군집마다 한 줄(크기, 군집 안 제곱합이나 목적의 몫, 혼합 모형은 무게) |

- 배열 필드는 쓰기 금지이고 처음 읽을 때 계산하는 필드도 있다(API.md §2.2). 결과의 생성자는 공개 API가 아니다.
- `HierarchyResult`는 분할이 하나가 아니라 `labels`·`n_clusters`가 없고 `cut(k=, height=)`이 분할을 준다(설계 문서 CL57).
- 중심·medoid·성분의 순서는 라벨의 순서다(`centers[j]`가 라벨 j의 중심).
- `repr`는 한 줄이다(API.md §2.3): `KMeansResult(nobs=300, n_clusters=3, inertia=573.168, n_iter=6)`.
- 결과는 자료의 사본을 쥐지 않는다. 자료가 필요한 게으른 필드(`HDBSCANResult.centroids` 등)는 자료의 읽기 전용 참조를 쥔다.

### §1.4 결정성, 난수, 정확도 (§9의 5·6번)
- **같은 입력이면 스레드 수와 상관없이 같은 비트다**. 빠른 길과 느린 길(Lloyd와 경계 가속, NN-chain과 일반 알고리즘, 행렬 길과 행렬 없는 길, KD 트리와 훑기)은 같은 비트를 낸다. 길을 고르는 기준은 모양뿐이다.
- **`rng`**: 무작위를 쓰는 함수(`kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `kmedoids(method='clara')`, `fuzzy_cmeans`, `gaussian_mixture`, `gap_statistic`, `GaussianMixtureResult.sample`)는 `rng`(`sd.random.default_rng(seed)`의 `Generator`)를 키워드로 꼭 받는다(없으면 `TypeError`). `kmedoids(method='pam')`에 주면 `ValueError`다. 호출 하나가 `rng`에서 64비트 두 단어만 받고(Philox의 키), 여러 시작(`n_init`)과 참조 자료(`gap_statistic`)는 그 키의 원소 번호로 뽑는다(시작 r의 t번째 수는 원소 r·2⁴⁰ + t; 설계 문서 CL41). 그래서 같은 `rng` 상태면 같은 비트이고 스레드 수와 무관하다. 규칙은 회귀의 `rlm(method='MM')`과 같다.
- **정확도 계약**(설계 문서 §4): 이산 결과(라벨, 트리, medoid)는 계산한 값과 정한 동점 규칙으로 정의대로 고른 것이고, 정확한 산술의 선택과는 가까운 동점에서만 다르다. 실수 결과(높이, 중심, 목적값, 모수, 지표)는 정확한 값과 견준 한계 안이다. ARI와 RI는 올바르게 반올림한 값이다.
- **모든 호출은 정해진 시간 안에 끝난다**: 반복이 있는 곳은 입력과 무관한 상한(`max_iter`)이 있다.

### §1.5 에러와 경고
- API.md §2의 종류와 꼴이다. 경고는 내지 않는다. 다른 라이브러리가 경고하거나 조용히 넘어가는 곳은 이렇다.

  | 다른 라이브러리 | sd |
  |---|---|
  | sklearn `KMeans`: 서로 다른 점이 k보다 적다(`ConvergenceWarning`, 적은 라벨) | `ValueError`: `kmeans(): 'k' (3) is larger than the number of distinct points (2)` |
  | sklearn `KMeans`·`GaussianMixture`: 반복 상한(`ConvergenceWarning`) | `converged=False` |
  | scipy `kmeans2`: 빈 군집(`UserWarning`을 반복마다) | 가장 먼 점으로 옮긴다(sklearn과 같다) |
  | R `kmeans`: 빈 군집(에러 또는 경고) | 위와 같다 |
  | scipy `linkage`: 거리 행렬처럼 보이는 2차원 입력(`ClusterWarning`) | 없음: 2차원은 관측값이다 |
  | scipy `fcluster`·`cut_tree`: 역전 트리의 높이 자르기(서로 다른 답) | `ValueError`: `cut_tree(): cannot cut a non-monotonic tree by height (its heights decrease somewhere)`, 결과의 `cut(): cannot cut a non-monotonic tree by height (method='centroid')` |
  | scipy `fcluster(criterion='maxclust')`: t < 1(정해지지 않은 답) | `ValueError`: `fcluster(): 't' must be >= 1 for criterion='maxclust', not 0` |
  | scipy `linkage`: 1차원(압축 꼴) 입력의 `metric`을 무시 | `ValueError`: `linkage(): a 1-D 'y' holds condensed distances; 'metric' does not apply (got 'cityblock')` |
  | sklearn `HDBSCAN`: `copy`의 기본값 변경(`FutureWarning`) | 없음: `copy` 인자가 없다(입력을 고치지 않는다) |
  | sklearn `GaussianMixture`: 공분산 실패(`ValueError`) | `LinAlgError`: `gaussian_mixture(): component 2 has an ill-defined covariance; increase 'reg_covar' or check for duplicate points` |
  | R `daisy`: 논리형 열(경고, 비대칭 이진으로) | 대칭 이진(`types`로 바꾼다) |

- 할당 전에 `MemoryError`인 곳(API.md §2): 계층 군집의 압축 꼴 거리 행렬(8·n(n − 1)/2 바이트)이 물리 메모리보다 클 때(행렬 없는 길이 있는 연결법은 그 길로 간다), `metric='precomputed'`의 정사각을 압축 꼴로 바꿀 때.

### §1.6 다른 라이브러리와 다른 점 (모음)
- 공통: 함수와 결과 객체, (X, k), 입력 순서와 무관한 분할과 처음 나오는 차례의 라벨, 스레드와 무관한 비트, `rng` 필수, 경고 없음.
- 계층: 동점은 정의와 열쇠, 높이는 최종 트리 위의 Lance–Williams 재귀(고른 알고리즘과 무관한 비트; scipy와 끝자리가 다를 수 있다), 역전 트리의 높이 자르기는 에러, 행렬 없는 길(큰 n).
- k-평균: 차의 제곱합 거리(sklearn은 전개식), `algorithm`이 없다(가속은 같은 비트), 서로 다른 점이 k보다 적으면 에러.
- DBSCAN: 경계 점은 가장 가까운 핵심 점의 군집(sklearn은 먼저 펼친 군집).
- HDBSCAN: 같은 높이의 병합을 한꺼번에 줄인다(sklearn은 그때 입력 순서에 따라 다르다), GLOSH `outlier_scores`가 더 있다, `copy`·`store_centers`가 없다.
- OPTICS: 거리를 소수 15자리로 반올림하지 않는다, 가장 작은 열쇠에서 시작한다.
- 혼합 모형: `k`, `bic`·`aic`는 필드, `lower_bound`와 `loglik`을 나눈다, 이차형식은 차의 꼴과 평균을 뺀 2차 모멘트(sklearn은 펼친다), 표본은 (*shape, p)이고 표본마다 성분이 독립이다.
- 지표: ARI·RI는 정확한 정수 셈, `silhouette_score`에 표본 뽑기가 없다.

## §2. 계층 (§9의 12–19번)
```
linkage(y, /, *, method='single', metric='euclidean', optimal_ordering=False)
agglomerative(X, /, *, method='single', metric='euclidean', optimal_ordering=False, nan_policy='raise')
divisive(X, /, *, metric='euclidean', nan_policy='raise')
HierarchyResult
HierarchyResult.cut(*, k=None, height=None)
HierarchyResult.cophenet()
HierarchyResult.dendrogram(*, p=30, truncate_mode=None, color_threshold=None, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True)
HierarchyResult.summary()
fcluster(Z, t, /, *, criterion='inconsistent', depth=2, R=None, monocrit=None)
cut_tree(Z, /, *, n_clusters=None, height=None)
cophenet(Z, /)
cophenet_correlation(Z, y, /)
CophenetResult
leaves_list(Z, /)
optimal_leaf_ordering(Z, y, /, *, metric='euclidean')
dendrogram(Z, /, *, p=30, truncate_mode=None, color_threshold=None, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True)
DendrogramResult
inconsistent(Z, /, *, d=2)
maxdists(Z, /)
maxinconsts(Z, R, /)
maxRstat(Z, R, i, /)
leaders(Z, T, /)
is_monotonic(Z, /)
is_valid_linkage(Z, /)
num_obs_linkage(Z, /)
is_isomorphic(T1, T2, /)
```
- **`method`**: `'single'`, `'complete'`, `'average'`, `'weighted'`, `'centroid'`, `'median'`, `'ward'`. 기본값은 scipy와 같은 `'single'`이다(CL16; R `hclust`는 `'complete'`, sklearn `AgglomerativeClustering`은 `'ward'`라 그쪽에서 옮겨 오는 코드는 `method`를 적어 준다). 높이의 뜻은 scipy와 같다: `ward`는 R `ward.D2`, `centroid`·`median`은 유클리드 거리의 뜻(R은 제곱 거리를 주어야 같은 트리이고 높이는 제곱), `weighted`는 R `mcquitty`다.
- 관측값과 비유클리드 `metric`의 `ward`·`centroid`·`median`은 `ValueError`(`linkage(): method='ward' requires metric='euclidean'`)다. 미리 계산한 거리는 받고 유클리드 거리라고 가정한다.
- **Z**: scipy의 꼴이다. 행 i는 i번째 병합(두 군집 번호, 높이, 크기)이고 군집 번호는 점이 0 … n − 1, 병합이 n + i다. 두 군집 번호는 (작은 것, 큰 것)이다. 행의 순서는 병합 순서다(단조 연결법에서는 높이 순서).
- **동점과 입력 순서**: 같은 높이의 후보는 두 군집의 가장 작은 점 번호(좌표의 사전순 정렬의 번호; 미리 계산한 거리는 입력 색인)의 짝이 작은 쪽을 먼저 합친다. 그래서 점들을 섞어 주어도 같은 트리다(Z의 점 번호는 입력의 번호). 행은 두 자식이 이미 있는 병합 가운데 (높이, 짝)이 가장 작은 것부터다(설계 문서 CL51).
- **높이**: 트리가 정해진 뒤 최종 트리 위에서 Lance–Williams 재귀로 다시 계산한 값이다(`single`·`complete`는 입력 거리 그대로; 설계 문서 CL45). 그래서 행렬 길, 행렬 없는 길, 동점의 일반 알고리즘이 같은 비트를 낸다. 단조인 연결법(`centroid`·`median` 밖)의 높이는 행을 따라 줄지 않는다(CL51). scipy(병합 순서의 누적)와 끝자리가 다를 수 있고, 정확도는 같은 크기다(참값과 2 ULP 이하, scipy 3 ULP 이하). 계약은 설계 문서 §4.1에서 증명한 한계다(CL55).
- **큰 n**: 관측값의 `single`, `ward`, `centroid`, `median`은 n × n 행렬을 만들지 않는다(n = 10⁵의 `single`이 0.17초, `ward`가 7초; 10스레드). `complete`, `average`, `weighted`, 그 밖의 거리와 미리 계산한 거리는 행렬이다(작업 사본까지 16·n(n − 1)/2 바이트).
- **`agglomerative`** → `HierarchyResult`: 같은 트리의 결과 객체다. **`divisive`**: R `diana`의 분할식 계층(높이는 지름)이다. `[R과 다름]` 지름이 같은 나누기가 이어지면 R의 `merge`는 배너에서 만든 다른 중첩을 보인다. sd는 실제로 나눈 트리다(같은 높이의 k 자르기가 R과 다를 수 있다; 서로 다른 높이 사이의 자르기는 같다).
- **자르기**:
  - `HierarchyResult.cut(k=)`, `cut_tree(Z, n_clusters=)`, `fcluster(Z, t, criterion='maxclust')`: 병합 순서로 마지막 k − 1개의 병합을 되돌린 분할(R `cutree(k=)`). `fcluster(maxclust)`는 scipy의 뜻("t개 이하"; 같은 높이의 병합은 함께 되돌리거나 함께 남긴다)이라 동점의 높이에서 k보다 적을 수 있다. `cut`과 `cut_tree`는 언제나 k개다.
  - `cut(height=)`, `cut_tree(height=)`, `fcluster(criterion='distance')`: 높이 h 이하의 병합만 남긴다(`[scipy와 다름]`: scipy `cut_tree(height=)`는 h 미만이다). 역전 트리(`is_monotonic(Z)`가 False)에서는 `ValueError`다(`[scipy와 다름]`; R과 같다).
  - 같은 높이의 병합이 있을 때 `cut(k=)`·`cut_tree(n_clusters=)`는 행의 차례로 되돌린다(R `cutree(k=)`와 같고, scipy `cut_tree`의 너비 우선 차례와는 다를 수 있다 `[scipy와 다름]`).
  - `cut`은 (n,) 라벨(0부터, 처음 나오는 차례), `cut_tree`는 scipy처럼 (n, len) 배열(인자가 하나면 (n, 1)), `fcluster`는 scipy처럼 1부터의 라벨이다.
  - `fcluster`의 기준: `'inconsistent'`(기본), `'distance'`, `'maxclust'`, `'monocrit'`, `'maxclust_monocrit'`(scipy와 같고 번호도 scipy의 순회 차례다). maxclust의 두 기준은 t가 1 이상의 정수다(t ≥ n은 scipy처럼 1 … n). `monocrit`은 아래에서 위로 줄지 않아야 한다(`ValueError`).
- **`cophenet(Z)`**: 압축 꼴의 cophenetic 거리. **`cophenet_correlation(Z, y)`**: `CophenetResult(correlation, distances)`(`[scipy와 다름]`: scipy는 `cophenet(Z, Y)`가 인자에 따라 다른 꼴을 돌려준다).
- **`dendrogram`** → `DendrogramResult`(namedtuple `icoord`, `dcoord`, `ivl`, `leaves`, `color_list`, `leaves_color_list`): scipy `dendrogram(no_plot=True)`과 같은 자리 자료다. 그리지 않는다. `icoord`·`dcoord`는 (m, 4) float64 배열, `leaves`는 int64 배열, `ivl`은 잎 이름의 리스트다. `color_list`는 matplotlib 색 이름이 아니라 묶음 번호(0부터 차례로, scipy처럼 돌려 쓰지 않는다; `color_threshold` 이상의 가지는 −1)이고, `leaves_color_list`는 잎마다 그 잎을 잇는 가지의 번호다. `orientation`은 검사만 한다(그림의 것이다).
- `is_valid_linkage`는 bool만 돌려준다(scipy의 `warning`, `throw`, `name`은 없다; 검사가 필요한 곳은 함수가 스스로 `ValueError`를 낸다). scipy가 보는 것에 더해 군집 번호가 정수인지, 개수가 두 자식의 합인지, 높이가 유한한지도 본다(`[scipy와 다름]`).
- **`optimal_leaf_ordering`**: 이웃한 잎의 거리 합이 가장 작은 순서(Bar-Joseph 외 2001)의 Z다. 뿌리의 첫 군집은 왼쪽에 둔다. 같은 합의 순서는 점 번호가 작은 쪽이다. `[scipy와 다름]` scipy 1.18.1은 최소가 아닌 순서를 낼 때가 있다(설계 문서 CL53).
- **`inconsistent`**: scipy의 네 열(깊이 d까지의 높이의 평균, 표준편차(n − 1로 나눈다), 개수, 불일치 계수). 표준편차는 평균을 뺀 두 번째 훑기라 scipy(한 번 훑기)와 끝자리가 다르고, 높이 하나일 때 scipy의 10⁻⁹ 꼴 대신 0이다.
- **`leaders`** → (L, M) int64 배열의 튜플(scipy와 같은 차례). **`cophenet_correlation`** → `CophenetResult(correlation, distances)`(namedtuple; 설계 문서 CL58).

| `HierarchyResult`의 필드·메서드 | 뜻 |
|---|---|
| `linkage` | Z (n − 1, 4) |
| `method`, `metric` | 연결법(`divisive`는 `'diana'`)과 거리 |
| `heights` | Z의 높이 열 (n − 1,) |
| `order` | 잎의 순서(`leaves_list(Z)`; `optimal_ordering=True`면 최적 순서) |
| `coefficient` | 응집 계수(R `agnes`의 `ac`) 또는 분할 계수(`diana`의 `dc`): 점마다 1 − (처음 합쳐지는 높이)/(마지막 병합의 높이)의 평균(마지막 높이가 0이면 NaN) |
| `nobs`, `omitted` | 트리의 점 수와 `nan_policy='omit'`이 뺀 행(Z의 점 번호는 남은 행의 번호; `cut`은 뺀 행에 −1) |
| `cut(k=, height=)`, `cophenet()`, `dendrogram(...)` | 위의 함수들 |

## §3. 분할 (§9의 20–24번)
```
kmeans(X, k, /, *, rng, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, weights=None, nan_policy='raise')
minibatch_kmeans(X, k, /, *, rng, init='k-means++', n_init='auto', batch_size=1024, max_iter=100, max_no_improvement=10, tol=0.0, init_size=None, reassignment_ratio=0.01, weights=None, nan_policy='raise')
bisecting_kmeans(X, k, /, *, rng, init='random', n_init=1, max_iter=300, tol=0.0001, strategy='biggest_inertia', weights=None, nan_policy='raise')
KMeansResult
KMeansResult.predict(X_new, /)
KMeansResult.transform(X_new, /)
KMeansResult.summary()
BisectingKMeansResult
BisectingKMeansResult.predict(X_new, /)
BisectingKMeansResult.transform(X_new, /)
BisectingKMeansResult.summary()
kmedoids(X, k, /, *, method='pam', metric='euclidean', rng=None, max_iter=1000, samples=5, sample_size=None, nan_policy='raise')
KMedoidsResult
KMedoidsResult.predict(X_new, /)
KMedoidsResult.summary()
fuzzy_cmeans(X, k, /, *, rng, m=2.0, init='random', max_iter=100, tol=1e-05, nan_policy='raise')
FuzzyCMeansResult
FuzzyCMeansResult.predict(X_new, /)
FuzzyCMeansResult.predict_proba(X_new, /)
FuzzyCMeansResult.summary()
```
- **`kmeans`**: 목적은 관성 Σwᵢ‖xᵢ − c_{lᵢ}‖²이고 Lloyd 반복이다. 거리는 차의 제곱합(`sd.spatial.cdist(metric='sqeuclidean')`의 한 짝과 같은 비트)이다(`[sklearn과 다름]`: sklearn은 전개식). `algorithm` 인자는 없다: 경계 가속(Hamerly)과 전개식의 증명된 거르기를 모양에 따라 쓰고 결과는 같은 비트다(설계 문서 CL60, §4.2).
  - `init`: `'k-means++'`(sklearn의 탐욕 판, 단계마다 후보 2 + ⌊ln k⌋개), `'random'`(무게에 비례해 점 k개를 비복원으로), (k, p) 배열(그 중심에서 시작; `n_init`은 1이어야 한다). 배열을 주어도 `rng`는 두 단어를 움직인다.
  - `n_init`: `'auto'`(sklearn과 같다: `'k-means++'`와 배열은 1, `'random'`은 10) 또는 1 이상의 int. 가장 좋은 시작은 (관성, 시작 번호)가 가장 작은 것이다.
  - 멈춤: 라벨이 바뀌지 않거나 중심 이동의 제곱합이 `tol`·(축마다 분산의 평균) 이하. 멈춘 뒤 라벨을 마지막 중심으로 다시 매긴다(sklearn과 같다). `max_iter`에서 멈추면 `converged=False`다.
  - 같은 거리의 중심은 맞춤 안의 번호(시작이 만든 차례)가 작은 쪽이다(sklearn·scipy와 같은 규칙; `predict`도 같아서 맞춘 점에 맞춤의 라벨을 준다). 빈 군집(무게 0)은 양의 무게 점 가운데 제 중심에서 가장 먼 점으로 옮긴다(sklearn의 규칙; 거리는 정확한 계산의 것이라 sklearn과 다른 점을 고를 수 있다).
  - k는 1 이상, 서로 다른 점(무게가 있으면 양의 무게 점)의 수 이하(`ValueError`)다.
- **`minibatch_kmeans`**: sklearn `MiniBatchKMeans`의 갱신과 멈춤(`max_no_improvement`, `tol`)이고 결과는 `KMeansResult`다(`n_steps`가 더 있고 `n_iter`는 ⌈n_steps·batch_size/n⌉). `n_init='auto'`는 `'random'`에서 3이다(sklearn과 같다). `init_size`(기본 3·batch_size, k보다 작으면 3k; n 이하로)가 k보다 작으면 `ValueError`다(sklearn은 경고하고 바꾼다). `max_no_improvement=None`은 그 멈춤을 끈다. `labels`와 `inertia`는 마지막 중심에서 모든 점의 것이다.
- **`bisecting_kmeans`**: sklearn `BisectingKMeans`다. `strategy`는 `'biggest_inertia'` 또는 `'largest_cluster'`(sklearn의 `bisecting_strategy`), `init`은 `'random'`(기본) 또는 `'k-means++'`다. 나누기마다 `kmeans`(k = 2, `n_init`개 시작, 가장 좋은 것은 (관성, 번호))이고 `tol`은 전체 자료의 분산에 곱한다. 서로 다른 점이 둘보다 적은 군집은 나누지 않는다. `predict`는 sklearn처럼 트리를 내려간다(맞춘 점이 제 라벨로 가지 않을 수 있다).
- **`kmedoids`**: 목적은 가장 가까운 medoid까지 거리의 평균이다(R `pam`의 `objective`). `method='pam'`은 BUILD와 SWAP(R `pam(variant='original')`; 변화가 음수인 가장 좋은 교환을 하고, 같으면 열쇠가 작은 medoid와 점), `method='clara'`는 R `clara`의 표본 방법이다(`samples` 기본 5, `sample_size` 기본 min(n, 40 + 2k), k < `sample_size` ≤ n; 첫 표본 다음부터 지금까지의 최선 medoid를 표본에 넣는다; 표본의 PAM은 R `clara(pamLike=TRUE)`의 것; `rng`가 꼭 있어야 한다). `max_iter`(기본 1000)는 교환 수의 상한이다(R에는 상한이 없다; 0이면 BUILD만). `metric`은 sd.spatial의 이름 또는 `'precomputed'`이고, 관측값의 `'euclidean'`·`'cityblock'`은 거리 행렬 없이 잰다(그 밖은 `sd.spatial.pdist`의 압축 꼴 행렬). 같은 목적의 선택은 열쇠(점의 사전순 차례)가 작은 쪽이다.
- **`fuzzy_cmeans`**: Bezdek의 퍼지 c-평균(목적 Σuᵢⱼᵐ‖xᵢ − cⱼ‖², m > 1). `init`은 `'random'`(점 k개) 또는 (k, p) 배열이다. 멈춤은 소속도 변화의 최대 노름 < `tol`(기본 10⁻⁵; e1071 `cmeans`의 기본 멈춤과 같거나 늦게 멈춘다)이다. 중심 위의 점은 그 중심들에 똑같이 나눠 속한다. `labels`는 소속도가 가장 큰 군집이다.

| 결과 | 필드와 메서드 |
|---|---|
| `KMeansResult` | 공통 필드, `centers`(k, p), `inertia`(= `objective`), `cluster_inertia`(k,), `n_init`, `best_init`, `method`, `init`, `predict`, `transform`(중심까지의 유클리드 거리 (m, k), `sd.spatial.cdist`); `minibatch_kmeans`의 결과는 `n_steps`도 |
| `BisectingKMeansResult` | 위와 같고 `tree`(잎이 군집 0 … k − 1인 연결 행렬 (k − 1, 4): 행마다 나누기 하나를 되돌린 병합, 높이는 나눈 군집의 관성을 위로 누적 최댓값, 넷째 열은 그 아래 잎의 수; `dendrogram(tree)`가 그린다), `strategy`, `best_init`은 나누기마다의 (k − 1,), `n_iter`는 모든 나누기의 합 |
| `KMedoidsResult` | 공통 필드, `medoids`(입력 행의 색인 (k,)), `medoid_points`(k, p; 관측값만, 아니면 None), `objective`(SWAP 뒤의 평균 거리), `build_objective`(BUILD 뒤; CLARA는 고른 표본의 BUILD medoid로 모든 점을 잰 값), `n_swaps`, `converged`, `method`, `metric`, `predict`(가장 가까운 medoid, 같으면 열쇠가 작은 medoid; 관측값만) |
| `FuzzyCMeansResult` | 공통 필드, `memberships`(nobs, k; 남은 행, 열 j가 라벨 j), `centers`, `m`, `objective`, `partition_coefficient`(Σuᵢⱼ²의 평균), `cluster_membership`(열의 합 (k,)), `n_iter`, `converged`, `predict`(가장 큰 소속도, 같으면 작은 라벨), `predict_proba`(소속도) |

## §4. 밀도 (§9의 25–28번)
```
dbscan(X, /, *, eps=0.5, min_samples=5, metric='euclidean', weights=None, nan_policy='raise')
DBSCANResult
DBSCANResult.summary()
hdbscan(X, /, *, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, max_cluster_size=None, metric='euclidean', alpha=1.0, cluster_selection_method='eom', allow_single_cluster=False, nan_policy='raise')
HDBSCANResult
HDBSCANResult.dbscan_clustering(cut_distance, /, *, min_cluster_size=5)
HDBSCANResult.summary()
optics(X, /, *, min_samples=5, max_eps=inf, metric='euclidean', cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, nan_policy='raise')
OPTICSResult
OPTICSResult.extract_xi(*, xi=0.05, min_samples=None, min_cluster_size=None, predecessor_correction=True)
OPTICSResult.extract_dbscan(eps, /)
OPTICSResult.summary()
```
- **`dbscan`**: 핵심 점은 거리 ≤ `eps`인 점(자기 포함)의 수(무게가 있으면 무게 합; 열쇠 차례로 더한다)가 `min_samples` 이상인 점이다(sklearn과 같다). 군집은 `eps` 안의 핵심 점끼리 이은 연결 요소다. 경계 점은 `eps` 안의 핵심 점 가운데 가장 가까운 것(같으면 열쇠)의 군집이다(`[sklearn과 다름]`: sklearn은 먼저 펼친 군집이라 입력 순서에 달려 있다). 잡음은 −1이다. `min_samples` ≥ 1(int), `eps` > 0(유한)이다. `weights`는 핵심 판정에만 쓴다. `metric='precomputed'`의 무게에 NaN이 있으면 `ValueError`다.
- **`hdbscan`**: sklearn 1.9의 뜻과 기본값이다. `min_samples=None`은 `min_cluster_size`이고 자기를 센다(원본 `hdbscan` 꾸러미의 값보다 1 크다). 상호 도달 거리는 max(core_a, core_b, d/`alpha`)이고 핵심 거리는 `alpha`로 나누지 않는다(`[sklearn과 다름]`: sklearn의 미리 계산한 거리 길은 핵심 거리도 나눈다; 설계 문서 CL67). 줄인 트리는 같은 높이의 병합을 한꺼번에 본다(Campello 외 2013): 같은 높이가 없으면 sklearn과 같은 트리이고, 있으면(`min_samples` > 1에서 흔하다) sklearn은 입력 순서에 따라 라벨이 달라지지만 sd는 같은 점이면 같은 라벨이다(`[sklearn과 다름]`; CL68). 고르기(`'eom'`, `'leaf'`, `cluster_selection_epsilon`, `max_cluster_size`, `allow_single_cluster`)와 확률은 sklearn의 규칙이다. `min_cluster_size` ≥ 2, `min_samples` 1 이상 n 이하, `cluster_selection_epsilon` ≥ 0(유한), `max_cluster_size`는 None 또는 1 이상, `alpha` > 0(유한), n ≥ 2다. 잡음은 −1이다. sklearn의 `copy`, `store_centers`, `leaf_size`, `algorithm`, `n_jobs`, `metric_params`는 없다(중심과 medoid는 게으른 필드, 길은 모양으로 정한다).
  - `dbscan_clustering(cut_distance, /, *, min_cluster_size=5)`: 단일 연결 트리를 `cut_distance`에서 자른 분할(높이 < `cut_distance`의 병합을 잇고, `min_cluster_size`보다 작은 성분은 잡음; sklearn과 같다)의 (n,) 라벨.
- **`optics`**: sklearn `OPTICS`의 뜻과 기본값이다(`metric`의 기본은 `'euclidean'`: sklearn의 `'minkowski'`, p = 2와 같은 거리). 순서의 동점(같은 도달 거리, 그리고 모두 inf일 때의 시작점)은 열쇠다(`[sklearn과 다름]`: sklearn은 입력 색인). 핵심 거리와 도달 거리는 계산한 거리 그대로다(`[sklearn과 다름]`: sklearn은 소수 15자리로 반올림한다). `cluster_method`는 `'xi'` 또는 `'dbscan'`(`eps`의 기본은 `max_eps`; `eps` ≤ `max_eps`). `min_samples`와 `min_cluster_size`는 sklearn처럼 2 이상의 int 또는 (0, 1]의 float(점 수의 비율, 적어도 2)이다. `max_eps` ≥ 0(inf 가능), `xi`는 [0, 1].
  - `extract_xi(...)`와 `extract_dbscan(eps)`는 같은 차례에서 다시 뽑은 새 `OPTICSResult`다(sklearn의 `cluster_optics_xi`, `cluster_optics_dbscan`과 같은 연산). `extract_dbscan`도 `eps` ≤ `max_eps`다.
- **`nan_policy='omit'`**: 뺀 행의 라벨은 −1, 점마다의 실수 필드(`probabilities`, `outlier_scores`, `reachability`, `core_distances`)는 NaN, `core_mask`는 False, `predecessor`는 −1이다. `ordering`과 `predecessor`는 입력 행의 번호, `single_linkage`와 `condensed_tree`의 점 번호는 남은 행의 번호다(`HierarchyResult`와 같다).

| 결과 | 필드와 메서드 |
|---|---|
| `DBSCANResult` | 공통 필드, `core_mask`(n,) bool, `eps`, `min_samples`, `metric` |
| `HDBSCANResult` | 공통 필드, `probabilities`(n,; 잡음은 0), `outlier_scores`(n,; GLOSH: 점이 떠나는 군집의 부분 트리에서 가장 큰 λ로 1 − λ/λ_max), `single_linkage`(상호 도달 거리의 Z), `condensed_tree`(namedtuple `parent`, `child`, `lambda_val`, `child_size`; 뿌리 군집은 n, 점은 0 … n − 1), `persistence`(군집마다: 안정성/(크기 × 트리의 가장 큰 λ), 원본 `hdbscan`의 꼴), `centroids`, `medoids`(게으른 필드, 확률을 무게로; 미리 계산한 거리면 None), `min_cluster_size`, `min_samples`, `cluster_selection_method`, `metric`, `dbscan_clustering(cut_distance)` |
| `OPTICSResult` | 공통 필드(라벨은 `cluster_method`의 것), `ordering`(nobs,), `reachability`, `core_distances`(n,; 정해지지 않으면 inf), `predecessor`(n,; 시작점은 −1), `cluster_hierarchy`(ξ의 (m, 2): 차례의 [시작, 끝], 안의 작은 군집이 먼저; `'dbscan'`이면 None), `min_samples`, `max_eps`, `cluster_method`, `eps`, `xi`, `metric`, `extract_xi`, `extract_dbscan` |

## §5. 혼합 모형 (§9의 29–32번)
```
gaussian_mixture(X, k, /, *, rng, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init='kmeans', weights_init=None, means_init=None, precisions_init=None, nan_policy='raise')
GaussianMixtureResult
GaussianMixtureResult.predict(X_new, /)
GaussianMixtureResult.predict_proba(X_new, /)
GaussianMixtureResult.score_samples(X_new, /)
GaussianMixtureResult.sample(shape=None, /, *, rng)
GaussianMixtureResult.summary()
```
- sklearn `GaussianMixture`의 EM과 기본값이다. `covariance_type`은 `'full'`(성분마다 행렬), `'tied'`(하나를 나눠 쓴다), `'diag'`(성분마다 대각), `'spherical'`(성분마다 분산 하나)이고 `reg_covar`(≥ 0)를 공분산의 대각에 더한다. 시작마다 `init`의 책임도에서 모수를 만든다: `'kmeans'`(이 영역의 `kmeans(X, k, init='k-means++', n_init=1)` 한 번의 라벨; 그 두 단어는 `rng`의 흐름에서 받는다), `'k-means++'`(k-means++가 고른 k점), `'random'`(무작위 책임도), `'random_from_data'`(무작위 k점). `weights_init`(합이 1인 (k,)), `means_init`((k, p)), `precisions_init`(sklearn의 모양: 정밀도 행렬 (k, p, p), (p, p), (k, p), (k,); full·tied는 대칭 양정치)이 주어지면 그 값으로 바꾸고, 셋 다 주어지면 아무것도 뽑지 않는다. 멈춤은 E 단계의 로그우도 평균(하한)의 변화 < `tol`(상한 `max_iter`; 0이면 시작 그대로)이고, `n_init`개 시작 가운데 하한이 가장 큰 것(같으면 앞)을 남긴다. n ≥ 2, 1 ≤ k ≤ n이고 `'kmeans'`는 서로 다른 점이 k개 이상이어야 한다(`ValueError`).
- 이차형식은 차의 꼴 ‖Uᵀ(x − μ)‖²이고 공분산은 평균을 뺀 두 번째 훑기다(`[sklearn과 다름]`: sklearn은 펼친 꼴이라 원점에서 먼 자료의 diag·spherical 밀도와 tied·diag·spherical의 공분산이 자릿수를 잃는다; 설계 문서 CL75). 같은 시작이면 sklearn과 같은 반복 수로 같은 고정점에 닿는다.
- `reg_covar`를 더해도 공분산이 양정치가 아니면 `LinAlgError`다(`gaussian_mixture(): component 2 has an ill-defined covariance; ...`, tied는 `the tied covariance is ill-defined`). 양정치가 아닌 `precisions_init`은 `ValueError`다.
- `bic`·`aic`는 필드다(sklearn은 `bic(X)` 메서드): −2·`loglik` + `n_parameters`·log n, −2·`loglik` + 2·`n_parameters`. 모수의 수는 sklearn의 `_n_parameters`와 같다.
- `labels`는 결과의 모수에서 가중 밀도가 가장 큰 성분이다(`predict(X)`와 같다; 같으면 맞춤이 먼저 만든 성분). 성분의 차례는 라벨의 처음 나오는 차례다.
- `predict`, `predict_proba`(책임도 (m, k), 열 j가 라벨 j), `score_samples`(점마다 혼합의 로그 밀도; sklearn과 같다).
- `sample(shape)`은 (*shape, p)의 점과 그 성분 라벨을 namedtuple `(X, labels)`로 돌려준다(sklearn은 `n_samples`와 튜플; 모양 규칙은 `sd.stats`의 `sample`과 같다: `shape=None`이면 (p,)와 int). 표본마다 무게로 성분을 뽑고 그 정규 분포에서 뽑는다(`[sklearn과 다름]`: sklearn은 성분별 개수를 다항 분포로 뽑아 성분 차례로 묶는다). `rng`에서 두 단어만 받는다.

| `GaussianMixtureResult`의 필드 | 뜻 |
|---|---|
| `weights`(k,), `means`(k, p), `covariances`, `precisions_cholesky` | 모수(꼴은 sklearn과 같다: `'full'` (k, p, p), `'tied'` (p, p), `'diag'` (k, p), `'spherical'` (k,)) |
| `loglik` | 결과의 모수에서 로그우도의 합(= `objective`) |
| `lower_bound` | 마지막 E 단계에서 점마다 평균한 하한(sklearn의 `lower_bound_`) |
| `bic`, `aic`, `n_parameters` | 정보 기준과 모수의 수 |
| `covariance_type`, `n_init`, `best_init` | 설정과 고른 시작 |
| `n_iter`, `converged` | 고른 시작의 반복 수와 수렴 |

## §6. 지표와 k 고르기 (§9의 10·11·33–35번)
```
silhouette_samples(X, labels, /, *, metric='euclidean')
silhouette_score(X, labels, /, *, metric='euclidean')
calinski_harabasz_score(X, labels, /)
davies_bouldin_score(X, labels, /)
adjusted_rand_score(labels_true, labels_pred, /)
rand_score(labels_true, labels_pred, /)
adjusted_mutual_info_score(labels_true, labels_pred, /, *, average_method='arithmetic')
normalized_mutual_info_score(labels_true, labels_pred, /, *, average_method='arithmetic')
mutual_info_score(labels_true, labels_pred, /)
homogeneity_score(labels_true, labels_pred, /)
completeness_score(labels_true, labels_pred, /)
v_measure_score(labels_true, labels_pred, /, *, beta=1.0)
homogeneity_completeness_v_measure(labels_true, labels_pred, /, *, beta=1.0)
HomogeneityResult
fowlkes_mallows_score(labels_true, labels_pred, /)
pair_confusion_matrix(labels_true, labels_pred, /)
contingency_matrix(labels_true, labels_pred, /, *, sparse=False)
gap_statistic(X, k_max, /, *, rng, method='kmeans', n_reference=100, d_power=1, reference='scaledPCA', select='firstSEmax', se_factor=1.0, nan_policy='raise')
GapResult
GapResult.summary()
gower_distance(data, /, *, types=None, weights=None)
```
- **내부 지표**: 뜻과 특수 경우는 sklearn과 같다: 홑원소 군집의 실루엣은 0, 군집 수가 1이거나 n이면 `ValueError`(`silhouette_score(): the number of clusters must be between 2 and n - 1 (got 1)`). 라벨 −1(잡음)도 한 군집으로 센다(sklearn과 같다). `silhouette_score`는 `silhouette_samples`의 평균이고 표본 뽑기(`sample_size`)가 없다.
- **외부 지표**: 분할표의 정수 셈이다. ARI·RI·짝 혼동 행렬은 정확한 정수 셈을 한 번 반올림한 값이다. 특수 경우(두 분할이 같은 한 군집 또는 모두 홑원소)의 값은 sklearn과 같다(ARI = 1, AMI = 1, 한 군집 대 모두 홑원소의 NMI = 0). `average_method`는 `'min'`, `'geometric'`, `'arithmetic'`, `'max'`. `contingency_matrix`는 (군집 수, 군집 수) int64이고 `sparse=True`면 `sd.sparse.csr_array`다. 행과 열은 라벨 값의 정렬 순서다.
- `homogeneity_completeness_v_measure` → `HomogeneityResult(homogeneity, completeness, v_measure)`(namedtuple).
- **`gap_statistic`**: R `clusGap`의 뜻이다(`n_reference`는 R의 `B`, `reference`는 `spaceH0`). `method`는 `'kmeans'`(`kmeans(X, k, rng=…)`, `n_init='auto'`) 또는 `'pam'`(`kmedoids`). `select`는 R `maxSE`의 다섯 규칙(`'firstSEmax'`, `'Tibs2001SEmax'`는 `'tibs2001semax'`로, `'globalSEmax'`, `'firstmax'`, `'globalmax'`; 소문자 이름) — 이름의 꼴은 임시이고 CL5 마일스톤에서 정한다.
- **`gower_distance`**: R `daisy(metric='gower')`의 뜻이다. `data`는 `DataFrame`(수 열은 범위로 나눈 차, 범주 열은 같은지, 순서형 범주는 순위, bool은 대칭 이진)이다. `types`는 열 이름 → `'asymm'`, `'symm'`, `'ordratio'`, `'logratio'`의 사전, `weights`는 열의 무게. 결측은 그 짝에서 그 변수를 빼고 무게를 다시 맞춘다. 압축 꼴 거리(길이 n(n − 1)/2)를 돌려준다.

| `GapResult`의 필드 | 뜻 |
|---|---|
| `k` | 고른 군집 수 |
| `gap`, `se`, `log_w`, `e_log_w` | k = 1 … `k_max`의 (k_max,) 배열(R `Tab`의 열 `gap`, `SE.sim`, `logW`, `E.logW`) |
| `method`, `select`, `n_reference`, `d_power`, `reference` | 설정 |

## §7. 그 밖 (§9의 36·37번)
```
affinity_propagation(X, /, *, damping=0.5, max_iter=200, convergence_iter=15, preference=None, metric='euclidean', nan_policy='raise')
AffinityPropagationResult
AffinityPropagationResult.predict(X_new, /)
AffinityPropagationResult.summary()
```
- sklearn `AffinityPropagation`의 반복과 멈춤이다. 유사도는 음의 제곱 유클리드 거리(sklearn의 `affinity='euclidean'`) 또는 `metric='precomputed'`의 유사도 행렬(이 함수만 거리가 아니라 유사도를 받는다; 이 이름은 임시이고 CL6 마일스톤에서 정한다). `preference=None`은 유사도의 중앙값이다.
- `[sklearn과 다름]` sklearn은 퇴화를 깨려고 유사도에 작은 잡음(`random_state`)을 더한다. sd는 더하지 않고 동점을 열쇠로 깬다. 그래서 `rng`가 없다.
- 결과: 공통 필드, `exemplars`(점의 색인), `exemplar_points`, `predict`(가장 가까운 exemplar; sklearn과 같다).
- 스펙트럼 군집은 나중의 `sd.embed` 뒤에 둔다. Birch와 바이클러스터링은 없다.

## §8. 다른 라이브러리의 이름 (대응표)
| 다른 라이브러리 | sd.cluster |
|---|---|
| sklearn `KMeans(n_clusters=k, random_state=s).fit(X)` → `.labels_`, `.cluster_centers_`, `.inertia_` | `kmeans(X, k, rng=sd.random.default_rng(s))` → `.labels`, `.centers`, `.inertia`(같은 시드여도 같은 시작은 아니다) |
| sklearn `KMeans(algorithm='elkan')` | 인자 없음(가속은 같은 비트) |
| R `kmeans(x, centers=k, nstart=10, iter.max=100)` | `kmeans(X, k, rng=…, n_init=10, max_iter=100)`(Lloyd; R의 기본은 Hartigan–Wong) |
| scipy `kmeans2(data, k, minit='++')`, `vq(obs, code_book)` | `kmeans(X, k, rng=…)`, `KMeansResult.predict` |
| sklearn `MiniBatchKMeans`, `BisectingKMeans(bisecting_strategy=…)` | `minibatch_kmeans`, `bisecting_kmeans(strategy=…)` |
| R `pam(x, k)`, `clara(x, k)` | `kmedoids(X, k)`, `kmedoids(X, k, method='clara', rng=…)`; `$id.med`(1부터) → `.medoids`(0부터) |
| e1071 `cmeans(x, centers, m=2)` | `fuzzy_cmeans(X, k, rng=…, m=2.0)` |
| scipy `linkage(y, 'single')` | `linkage(y)` 또는 `linkage(y, method='single')`(`method`는 키워드로만) |
| R `hclust(dist(x), 'complete')` | `agglomerative(X, method='complete')` |
| R `hclust(dist(x), 'ward.D2')` / `'ward.D'` | `method='ward'` / 없음(R `ward.D`에 제곱 거리를 주면 같은 트리, 높이는 제곱) |
| R `hclust(dist(x)^2, 'centroid')`, `'median'` | `method='centroid'`, `'median'`(높이는 R의 제곱근) |
| R `hclust(…, 'mcquitty')` | `method='weighted'` |
| R `cutree(h, k=3)`, `cutree(h, h=2.5)` | `.cut(k=3)`, `.cut(height=2.5)`(라벨은 0부터) |
| R `agnes(x)$ac`, `diana(x)` | `agglomerative(X, method='average').coefficient`, `divisive(X)` |
| sklearn `AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X).labels_` | `agglomerative(X, method='ward').cut(k=3)` |
| scipy `cophenet(Z, Y)` | `cophenet_correlation(Z, y)` |
| scipy `dendrogram(Z, no_plot=True)` | `dendrogram(Z)` |
| sklearn `DBSCAN(eps, min_samples).fit(X, sample_weight=w)` | `dbscan(X, eps=…, min_samples=…, weights=w)` |
| sklearn `HDBSCAN(min_cluster_size=m)`, 원본 `hdbscan.HDBSCAN(min_samples=s)` | `hdbscan(X, min_cluster_size=m)`, `hdbscan(X, min_samples=s + 1)` |
| sklearn `OPTICS(...)`, `cluster_optics_dbscan(...)` | `optics(...)`, `OPTICSResult.extract_dbscan(eps)` |
| sklearn `GaussianMixture(n_components=k, random_state=s).fit(X)`, `.bic(X)` | `gaussian_mixture(X, k, rng=…)`, `.bic` |
| sklearn `GaussianMixture.score(X)` | `.loglik / .nobs`(같은 X에서) |
| sklearn `metrics.adjusted_rand_score` 등 | 같은 이름 |
| sklearn `silhouette_score(X, labels, sample_size=…, random_state=…)` | `silhouette_score(X, labels)`(언제나 전체) |
| R `silhouette(cl, dist(x))` | `silhouette_samples(X, labels)` |
| R `clusGap(x, kmeans, K.max, B=100)`, `maxSE` | `gap_statistic(X, k_max, rng=…, n_reference=100)` |
| R `daisy(df, metric='gower')` | `gower_distance(df)` |
| sklearn `AffinityPropagation(random_state=…)` | `affinity_propagation(X)`(잡음 없음) |
| sklearn `SpectralClustering`, `MeanShift`, `Birch`, `BayesianGaussianMixture`, `FeatureAgglomeration`, `connectivity=` | 없음(나중; 설계 문서 §8) |
| sklearn `SpectralBiclustering`, `SpectralCoclustering`, scipy `to_tree`, `ClusterNode`, `fclusterdata`, `whiten` | 없음 |
| 분류기와 그 회귀판(트리, 포레스트, k-NN, SVM, 부스팅), 차원 축소 | 나중의 `sd.classify`, `sd.embed`(CL2) |
