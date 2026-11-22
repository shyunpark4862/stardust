# sd.graph API

> 상태: **초안** (GR0 마일스톤, 2026-11-23). GR1 마일스톤(2026-11-23)에서 §2의 여덟 이름이 공개되었다. 아래 시그니처는 scipy 1.18.1의 `scipy.sparse.csgraph`의 이름과 인자 이름(과 `scipy.optimize.linear_sum_assignment`, scipy에 없는 일곱은 networkx 3.7의 뜻)에 stardust의 규칙과 GR1–GR28의 결정을 입힌 안이고, GR1–GR5 마일스톤에서 하나씩 공개된다(docs/design/graph.md §8: 순회·연결 요소·위상 정렬·변환 도구는 GR1, 최단 경로는 GR2, 신장 트리·흐름·매칭·배정은 GR3, 순서·구조적 랭크·라플라시안은 GR4, 더한 알고리즘은 GR5 마일스톤).

- 설계와 결정의 근거는 [docs/design/graph.md](../design/graph.md)(§1.1 GR1–GR33), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/graph/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 예외 클래스는 이름만)과 결과 객체의 필드(`Class.field`)다. 모듈 수준의 이름이 계획한 공개 이름 전부이고, 공개된 것은 시그니처가 블록과 같아야 한다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- **이름공간은 평평하다**(GR3): 모든 함수가 `sd.graph`에 바로 있다. scipy의 `from scipy.sparse.csgraph import dijkstra`는 `from stardust.graph import dijkstra`이고, `scipy.optimize.linear_sum_assignment`는 `sd.graph.linear_sum_assignment`다(GR4: 같은 배정 문제의 셋이 한곳에 있고 같은 C++ 고리를 쓴다).
- **이름은 scipy의 것이다**: `csgraph_from_dense`처럼 긴 이름도 그대로 둔다(GR14). scipy에 없는 일곱은 networkx의 이름이다(`topological_sort`, `articulation_points`, `bridges`, `biconnected_components`, `core_number`, `pagerank`, `betweenness_centrality`; GR18). 별칭이 없다.
- **Graph 클래스가 없다**(GR2): 그래프는 인접 행렬이고 함수는 배열이나 결과 객체를 돌려준다.
- **데이터와 주된 인자는 위치로, 옵션은 키워드로만** 받는다(GR21). 위치 인자는 그래프와 `i_start`, `source`, `sink`, `K`, `predecessors`뿐이다. `[scipy와 다름]`: scipy의 `connected_components(G, False)`, `dijkstra(G, False)`는 `TypeError`다.
- scipy의 `overwrite`(최단 경로·최소 신장 트리), `laplacian`의 `copy`와 `form='function'`, 가린 배열 도구 셋(`csgraph_masked_from_dense csgraph_from_masked csgraph_to_masked`)은 없다(GR14). 주면 모르는 키워드의 `TypeError`다. 입력을 고치지 않고, `form='lo'`가 `form='function'`의 일을 한다.
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1): `dijkstra(): unexpected keyword argument 'overwrite'`.

### §1.2 입력 (GR7, GR8, GR22)
- **그래프**는 정방 (n, n) 인접 행렬이다. 행 i, 열 j의 원소가 i → j 변의 무게다.
  - `sd.sparse`의 `csr_array`, `csc_array`, `coo_array`, `dok_array`: **저장된 원소**가 변이고 저장된 0도 무게 0의 변이다(scipy와 같다). CSR이 아니면 한 번 CSR로 바꾼다(COO의 중복은 더한다).
  - 밀집(`sd.ndarray`, numpy 배열, 수의 중첩 시퀀스, 2차원): **0이 아니고 NaN도 ±inf도 아닌 원소**가 변이다(scipy와 같다). 무게 0의 변은 희소의 저장된 0이나 `csgraph_from_dense(…, null_value=inf)`로 만든다.
  - scipy의 희소 객체는 `TypeError`다(메시지가 `sd.sparse.csr_array(A)`를 짚는다; GR7). frame의 `DataFrame`도 받지 않는다.
  - dtype은 bool, int64, float64다. 무게를 쓰는 계산은 float64로 한다. complex128은 `TypeError`다(`[scipy와 다름]`: scipy는 허수부를 버리고 `ComplexWarning`을 낸다).
  - 2차원이 아니거나 정방이 아니면 `ValueError`다. 매칭 셋과 `structural_rank`는 직사각을 받는다. n = 0도 받는다(빈 결과).
- **무게의 값**(GR8): 무게를 쓰는 함수(최단 경로, `minimum_spanning_tree`, `min_weight_full_bipartite_matching`, `linear_sum_assignment`, `maximum_flow`, `pagerank`, 무게 있는 `betweenness_centrality`)는 희소 입력의 저장된 NaN·±inf를 `ValueError`로 막는다(`dijkstra(): stored edge weights must be finite`). 구조만 보는 함수는 패턴만 읽는다. `laplacian`은 행렬 함수라 값을 그대로 쓴다.
- **무향**(`directed=False`): i와 j 사이의 변은 i → j와 j → i 가운데 저장된 것이고, 둘 다 있으면 작은 무게다(scipy와 같다; NaN이 끼면 NaN, 0.0과 −0.0이면 −0.0; GR29). 패턴만 보는 함수는 두 방향 패턴의 합집합이다.
- **노드 인자**(`i_start`, `source`, `sink`, `indices`의 원소; GR22)는 0 이상 n 미만의 int다. 범위 밖과 음수는 `ValueError`, bool·실수는 `TypeError`다. `indices`는 0차원이나 1차원이고 중복을 받는다. `[scipy와 다름]`: scipy는 `indices=1.5`를 1로 자르고 `indices=-1`을 n − 1로 감으며, `breadth_first_order(G, 5)`는 범위 밖에서 빈 결과를 낸다.
- **자기 고리**: 최단 경로와 순회에서는 쓰이지 않고(음수 무게의 고리는 음수 순환이다), `minimum_spanning_tree`·무향 구조 함수·`laplacian`은 무시하며, `topological_sort`에서는 순환이다. `pagerank`는 변으로 센다(networkx의 뜻; GR28).

### §1.3 결정성
- **같은 입력이면 스레드 수와 상관없이 같은 결과다**(값, 선행 노드, 순서, 레이블, 흐름, 매칭). 결과는 알고리즘이 아니라 정의로 정해진다: 연결 요소의 번호는 가장 작은 노드의 순서(GR12), 순회의 이웃은 번호 오름차순(GR13), 최단 거리는 원점에서부터의 차례 합의 최솟값과 정한 선행 노드 규칙(GR6), 최소 신장 트리는 (무게, min(i, j), max(i, j))로 유일한 트리(GR17)다. 흐름·매칭·배정의 여러 최적해 가운데 하나는 고정된 훑기 순서로 정해진다(GR16).
- **최단 경로는 `method`와 무관하다, `'FW'`만 빼고**(GR5, GR25, GR26): `'D'`, `'BF'`, `'J'`, `'auto'`와 `dijkstra`, `bellman_ford`, `johnson`은 거리와 선행 노드가 비트까지 같다. `'auto'`는 FW를 고르지 않는다. `method='FW'`와 `floyd_warshall`은 블록 Floyd–Warshall의 반올림(가운데에서 잇는 합)이라 끝자리가 다를 수 있다(정확도 계약 안이다; §3).
- 경고는 내지 않는다. scipy가 경고하거나 조용히 틀린 값을 내는 입력은 에러다(§1.5).

### §1.4 에러
- API.md §2의 종류와 꼴이다. 메시지는 함수 이름으로 시작한다(`dijkstra(): ...`).
- `NegativeCycleError`는 `ValueError`의 하위 클래스다(`[scipy와 다름]`: scipy의 것은 `Exception`의 하위 클래스).
- 요청한 출력(모든 쌍의 n² 거리와 선행 노드, `yen`의 K × n)이 물리 메모리보다 크면 할당 전에 `MemoryError`다.

### §1.5 scipy와 다른 점 (모음)
- 입력: complex·scipy 희소 객체는 `TypeError`, 저장된 NaN·±inf 무게는 `ValueError`, 노드 인자는 0 … n−1의 int만, 2차원 `indices`는 `ValueError`, 옵션은 키워드만, `overwrite`·`copy`·`form='function'`·가린 배열 도구 없음, `maximum_flow`·`reverse_cuthill_mckee`·`structural_rank`·매칭도 밀집을 받는다.
- 결과: 색인 배열은 모두 int64(GR10), 스칼라는 Python `int`, 결과 튜플은 namedtuple(GR11), 강한 연결 요소의 번호는 가장 작은 노드 순서, 무향 순회는 합친 이웃의 오름차순, 무향 트리의 저장된 0은 0, `csgraph_to_dense`의 저장된 NaN은 NaN, `'FW'`를 뺀 최단 경로는 방법과 무관, `auto`는 FW를 고르지 않음, `construct_dist_matrix`는 원점에서부터 더함, `laplacian`의 희소 결과는 CSR, 최소 신장 트리는 밀집의 작은 원소를 버리지 않는다.
- 에러: 음수 무게의 `dijkstra`(GR9), NaN `limit`(GR23), 음수 용량, int64를 넘는 흐름(`OverflowError`), 실수 용량(`TypeError`), `NegativeCycleError`는 `ValueError`이고 "요청한 원점에서 닿는 순환" 한 규칙이다.

## §2. 순회, 연결 요소, 변환 (GR1 마일스톤)
```
connected_components(csgraph, /, *, directed=True, connection='weak', return_labels=True)
breadth_first_order(csgraph, i_start, /, *, directed=True, return_predecessors=True)
depth_first_order(csgraph, i_start, /, *, directed=True, return_predecessors=True)
breadth_first_tree(csgraph, i_start, /, *, directed=True)
depth_first_tree(csgraph, i_start, /, *, directed=True)
topological_sort(csgraph, /)
csgraph_from_dense(graph, /, *, null_value=0, nan_null=True, infinity_null=True)
csgraph_to_dense(csgraph, /, *, null_value=0)
```
- **`connected_components`**: `connection`은 `'weak'` 또는 `'strong'`이고 `directed=False`면 읽지 않는다. 요소 번호는 요소의 가장 작은 노드의 순서다(노드 0의 요소가 0, 그다음 처음 나오는 노드의 요소가 1, …; GR12). 약한 연결과 무향은 scipy와 같고, 강한 연결은 `[scipy와 다름]`(scipy는 알고리즘의 순서; 분할은 같다). 결과는 `ConnectedComponentsResult(n_components, labels)`(Python `int`와 int64 배열), `return_labels=False`면 Python `int` 하나다.
- **순회**: 이웃은 노드 번호 오름차순으로 본다(무향이면 두 방향 이웃을 합친 오름차순; GR13). 너비 우선은 큐, 깊이 우선은 재귀 깊이 우선과 같은 순서다. 유향은 scipy와 같은 순서이고, 무향은 `[scipy와 다름]`(scipy는 나가는 이웃 먼저): sd는 G와 Gᵀ에서 같은 답을 낸다. 결과는 `TraversalResult(node_array, predecessors)`(int64; 시작 노드와 닿지 않은 노드의 선행 노드는 −9999), `return_predecessors=False`면 `node_array`만이다. 트리 함수는 (n, n) `csr_array` float64이고 변은 부모 → 자식 방향에 원래 무게(무향이면 있는 쪽, 둘 다면 작은 쪽)다. 무게 0의 변은 저장된 0이다(`[scipy와 다름]`: scipy의 무향 트리는 저장된 0을 inf나 다른 방향의 무게로 바꾼다; GR29).
- **`topological_sort`**: 모든 변 i → j에서 i가 j보다 앞서는 노드의 int64 (n,) 배열 가운데 **사전 순서로 가장 작은 것**이다(networkx의 `lexicographical_topological_sort`와 같은 답; GR18). 순환(자기 고리 포함)이 있으면 `ValueError`(`topological_sort(): the graph has a cycle`)다.
- **`csgraph_from_dense`**: 정방 밀집 행렬의 원소 가운데 `null_value`와 같은 것(`nan_null`이면 NaN, `infinity_null`이면 ±inf도)을 뺀 나머지를 저장한 `csr_array` float64다. `null_value=None`이면 같음으로 빼는 값이 없고, NaN인 `null_value`는 `nan_null=True`, 무한대인 것은 `infinity_null=True`로 읽는다(scipy와 같다). `null_value`는 int나 float다.
- **`csgraph_to_dense`**: 저장된 원소를 그 자리에, 나머지를 `null_value`(int나 float)로 채운 (n, n) `sd.ndarray` float64다. 밀집 입력은 그래프 규칙으로 읽는다. sd의 CSR에는 중복 원소가 없고, 저장된 NaN은 NaN이다(`[scipy와 다름]`: scipy는 inf; GR30).

## §3. 최단 경로 (GR2 마일스톤)
```
shortest_path(csgraph, /, *, method='auto', directed=True, return_predecessors=False, unweighted=False, indices=None)
dijkstra(csgraph, /, *, directed=True, indices=None, return_predecessors=False, unweighted=False, limit=inf, min_only=False)
bellman_ford(csgraph, /, *, directed=True, indices=None, return_predecessors=False, unweighted=False)
johnson(csgraph, /, *, directed=True, indices=None, return_predecessors=False, unweighted=False)
floyd_warshall(csgraph, /, *, directed=True, return_predecessors=False, unweighted=False)
yen(csgraph, source, sink, K, /, *, directed=True, return_predecessors=False, unweighted=False)
construct_dist_matrix(csgraph, predecessors, /, *, directed=True, null_value=inf)
reconstruct_path(csgraph, predecessors, /, *, directed=True)
NegativeCycleError
```
- **거리의 정의**(GR5, GR6): 원점 s에서 v까지의 거리는 s–v 경로 가운데 경로의 무게를 원점에서부터 차례로 더한 float64 값이 가장 작은 것이다. 선행 노드는 꼭 맞는 변(fl(d(u) + w(u, v)) = d(v)) 가운데 원점에서 변 수가 가장 적은 것, 그다음 번호가 가장 작은 것이다. 그래서 선행 노드는 무게 0의 변이 있어도 언제나 트리다. `'D'`, `'BF'`, `'J'`, `'auto'`와 `dijkstra`·`bellman_ford`·`johnson`은 이 정의의 같은 비트를 낸다. `[scipy와 다름]`: scipy는 방법마다 선행 노드의 동점이 다르다(마름모 그래프에서 D·J는 2, BF·FW는 1).
- **`method`**: `'auto'`, `'D'`, `'BF'`, `'J'`, `'FW'`다(그 밖은 `ValueError`, 문자열이 아니면 `TypeError`). `'auto'`는 음수 무게가 있으면 J, 없으면 D다(밀집한 그래프에서는 같은 비트의 힙 없는 밀집 Dijkstra로 빠르게 한다). `[scipy와 다름]`: scipy의 auto는 `indices`가 없고 변이 n²/4개 이상이면 FW를 고른다(GR26). `method='FW'`와 `indices`를 함께 주면 `ValueError`다(scipy와 같다).
- **`floyd_warshall`과 `method='FW'`**(GR25, GR27): 블록 Floyd–Warshall이다. 거리는 경로를 가운데에서 잇는 합이라 다른 방법과 끝자리가 다를 수 있고(scipy에서도 같은 일이 있다: 상대 8.7×10⁻¹⁶), 정확도 계약 안이며 스레드 수와 무관하다. 선행 노드는 Floyd–Warshall의 갱신(더 작아질 때 pred(i, j) ← pred(k, j))이 정하고(scipy와 같은 규칙), 그 경로를 원점에서부터 다시 더한 값은 FW의 거리와 끝자리가 다를 수 있다.
- **결과의 꼴**(scipy와 같다): `indices=None`이면 (n, n) float64, int 하나면 (n,), 1차원이면 (len, n)이다. 닿지 않으면 inf와 −9999다. `return_predecessors=True`면 `ShortestPathResult(dist_matrix, predecessors)`(선행 노드 int64)다. 2차원 `indices`는 `ValueError`다(`[scipy와 다름]`).
- **`unweighted=True`**: 모든 변의 무게가 1이다(저장된 0의 변도). 거리는 변의 수이고 정확하다.
- **`limit`**(`dijkstra`만): 거리 ≤ `limit`인 노드만 찾고 나머지는 inf와 −9999다(경계를 넣는다). 0 이상이거나 inf여야 하고 음수와 NaN은 `ValueError`다(`[scipy와 다름]`: scipy는 NaN에서 모든 노드를 inf로 낸다; GR23).
- **`min_only=True`**(`dijkstra`만): 모든 `indices`를 함께 원점으로 둔 거리 하나 (n,)다. `return_predecessors=True`면 `MinOnlyResult(dist_matrix, predecessors, sources)`이고 `sources[v]`는 v의 선행 노드 사슬이 끝나는 원점(닿지 않으면 −9999)이다.
- **음수 무게**: `dijkstra`와 `method='D'`는 `ValueError`다(`dijkstra(): negative edge weights are not allowed; use bellman_ford or johnson`; GR9; `[scipy와 다름]`: scipy는 경고와 함께 값을 낸다). `bellman_ford`, `johnson`, `floyd_warshall`, `'auto'`는 받는다. 무향 그래프의 음수 변은 그 자체로 음수 순환이다.
- **`NegativeCycleError`**: 요청한 원점에서 닿을 수 있는 음수 순환이 있으면 낸다(모든 방법이 같다; `indices=None`이면 그래프 어디든). 메시지는 순환 위의 가장 작은 노드를 적는다(`bellman_ford(): negative cycle through node 3`). `[scipy와 다름]`: scipy의 `bellman_ford(indices=0)`은 닿지 않는 순환을 무시하고 `johnson(indices=0)`은 에러를 낸다.
- **`yen(csgraph, source, sink, K)`**: 고리 없는 s–t 경로 가운데 (원점에서부터의 차례 합, 변의 수, 노드 열의 사전 순서)로 앞선 K개다. 결과는 길이 M ≤ K의 거리 배열 하나 또는 `YenResult(dist_array, predecessors)`(M × n, int64)다. 음수 무게는 받고 음수 순환은 `NegativeCycleError`다. `source == sink`면 길이 0의 경로 하나, `K`는 0 이상의 int다(0이면 빈 결과).
- **`construct_dist_matrix`**: 선행 노드 행렬이 가리키는 경로를 원점에서부터 차례로 더한 거리다. 그래서 FW가 아닌 방법의 `shortest_path`가 돌려준 선행 노드에서 그 거리와 비트까지 같다. `[scipy와 다름]`: scipy는 도착점에서 거꾸로 더한다. 닿지 않는 자리는 `null_value`다.
- **`reconstruct_path`**: 선행 노드 1차원 배열이 그리는 트리를 (n, n) `csr_array`로 만든다(변은 부모 → 자식).

## §4. 신장 트리, 흐름, 매칭, 배정 (GR3 마일스톤)
```
minimum_spanning_tree(csgraph, /)
maximum_flow(csgraph, source, sink, /, *, method='dinic')
maximum_bipartite_matching(graph, /, *, perm_type='row')
min_weight_full_bipartite_matching(biadjacency, /, *, maximize=False)
linear_sum_assignment(cost_matrix, /, *, maximize=False)
```
- **`minimum_spanning_tree`**: 무향으로 본 그래프의 최소 신장 숲이다. 변을 (무게, min(i, j), max(i, j))로 견주면 답이 하나다(GR17). 결과는 (n, n) `csr_array` float64이고 고른 변은 그 무게를 준 저장 자리(같은 무게면 i < j의 위 삼각)에 있다. 음수 무게는 받는다. `[scipy와 다름]`: scipy는 밀집 입력에서 1e-8보다 작은 원소를 버리고 저장된 NaN·inf 변을 트리에 넣는다.
- **`maximum_flow`**: `method`는 `'dinic'` 또는 `'edmonds_karp'`다. 결과는 `MaximumFlowResult`(`flow_value`는 Python `int`, `flow`는 반대칭 (n, n) `csr_array` int64)다. 용량은 bool·int64만 받는다(float64는 `TypeError`), 음수 용량은 `ValueError`, 원점에서 나가는 용량의 합이 int64를 넘으면 `OverflowError`다(GR15; `[scipy와 다름]`: scipy는 용량을 int32로 줄여 2⁴⁰의 흐름이 0이 된다). `source == sink`는 `ValueError`다. 밀집도 받는다.
- **`maximum_bipartite_matching`**: 행 쪽과 열 쪽의 이분 그래프에서 가장 큰 매칭이다. `perm_type='row'`면 열마다 짝지은 행(없으면 −1), `'column'`이면 행마다의 열(int64)이다. 저장된 0도 변이다.
- **`min_weight_full_bipartite_matching`**: 크기 min(행 수, 열 수)인 매칭 가운데 무게 합이 가장 작은(`maximize=True`면 큰) 것이다. 결과는 `AssignmentResult(row_ind, col_ind)`(int64, `row_ind` 오름차순)다. 그런 매칭이 없으면 `ValueError`다. 저장된 0은 무게 0의 변이다(`[scipy와 다름]`: scipy는 변 없음으로 읽는다).
- **`linear_sum_assignment`**: scipy의 뜻 그대로다. 모든 원소가 짝이고(0도 비용 0), +inf는 고를 수 없는 짝이다. 모든 짝을 고를 수 없으면 `ValueError`(`linear_sum_assignment(): cost matrix is infeasible`), NaN과 −inf는 `ValueError`다. 직사각을 받고 결과는 `AssignmentResult`다. 입력은 밀집만이다.
- 흐름 값과 매칭의 크기는 유일하다. 흐름 자체와 여러 최적 배정 가운데 하나는 행과 이웃을 오름차순으로 보는 고정된 순서의 결과다(GR16).

## §5. 순서, 구조적 랭크, 라플라시안 (GR4 마일스톤)
```
reverse_cuthill_mckee(graph, /, *, symmetric_mode=False)
structural_rank(graph, /)
laplacian(csgraph, /, *, normed=False, return_diag=False, use_out_degree=False, form='array', dtype=None, symmetrized=False)
```
- **`reverse_cuthill_mckee`**: 띠 폭을 줄이는 순열(int64 (n,))이다. `symmetric_mode=False`면 A + Aᵀ의 패턴이다. 요소마다 차수가 가장 작은 노드 가운데 가장 작은 번호에서 시작하고, 이웃은 (차수, 번호) 순서로 넣고, 요소는 시작 노드의 순서로 이은 뒤 전체를 뒤집는다. 밀집도 받는다.
- **`structural_rank`**: 패턴의 가장 큰 이분 매칭의 크기(Python `int`)다. 직사각을 받고 저장된 0도 원소다.
- **`laplacian`**: L = D − A(D는 들어오는 무게의 합, `use_out_degree=True`면 나가는 무게의 합; 대각은 무시). `normed=True`면 D^(−1/2) L D^(−1/2)이고 차수 0인 노드는 1로 나눈다. `symmetrized=True`면 A + Aᵀ로 먼저 만든다. 결과는 밀집 입력이면 `sd.ndarray`, 희소면 `csr_array`(`[scipy와 다름]`: scipy는 `coo_array`), `form='lo'`면 `sd.sparse.LinearOperator`다. `return_diag=True`면 `LaplacianResult(laplacian, diag)`다. `dtype=None`이면 입력의 dtype(bool은 int64), `normed`면 float64다. 값을 그대로 쓰므로 NaN·inf는 IEEE로 퍼진다.

## §6. 더한 알고리즘 (GR5 마일스톤; scipy에 없음, 뜻은 networkx)
```
articulation_points(csgraph, /)
bridges(csgraph, /)
biconnected_components(csgraph, /)
core_number(csgraph, /)
pagerank(csgraph, /, *, alpha=0.85, personalization=None, dangling=None, directed=True, unweighted=False, tol=1e-12)
betweenness_centrality(csgraph, /, *, directed=True, unweighted=False, normalized=True, endpoints=False)
```
- **무향의 넷**은 그래프를 언제나 무향으로 본다(두 방향 패턴의 합집합, 대각 무시). `articulation_points`는 지우면 연결 요소가 늘어나는 노드의 오름차순 int64 배열, `bridges`는 그런 변의 (k, 2) int64 배열(행 (i, j), i < j, 사전 순서)이다. `biconnected_components`는 `BiconnectedComponentsResult(n_components, edges, labels)`: `edges`는 모든 변(`bridges`의 꼴, 사전 순서), `labels`는 변마다의 요소 번호이고 번호는 요소의 가장 작은 변의 순서다. `core_number`는 노드마다의 코어 수(int64)이고 k-코어는 `core_number(G) >= k`인 노드다(GR18).
- **`pagerank`**: x = α·Pᵀx + (1 − α)·v의 해(합 1의 float64 (n,))다. P는 나가는 무게로 정규화한 전이 행렬(`unweighted=True`면 변마다 1, 자기 고리도 변)이고, 나가는 무게가 0인 행은 `dangling` 분포다. 기본 `personalization`은 균등, 기본 `dangling`은 `personalization`이다. `alpha`는 0 이상 1 미만, `personalization`·`dangling`은 길이 n의 음이 아닌 유한 배열이고 합이 0보다 커야 한다(합 1로 정규화한다). 음수 무게는 `ValueError`다. **결과는 해와 ℓ₁로 `tol` 안이다**(증명된 사후 한계로 멈추고 반복 수의 상한이 α와 `tol`로 정해져 `max_iter`가 없다; GR19). `[networkx와 다름]`: networkx는 `tol=1e-6`, `max_iter=100`이고 연속한 반복의 차로 멈춘다.
- **`betweenness_centrality`**: Brandes의 정의다. 무향이면 (s, t)와 (t, s)를 한 번 세고, `normalized=True`면 유향 1/((n − 1)(n − 2)), 무향 2/((n − 1)(n − 2))를 곱하고, `endpoints=True`면 끝점도 센다(networkx의 뜻과 정규화). float64 (n,)이다. 무게는 0보다 커야 하고(`ValueError`), `unweighted=True`면 패턴만 본다. 표본 어림(`k`, `seed`)은 없다(GR20).

## §7. 결과 객체 (GR11)
```
ShortestPathResult.dist_matrix
ShortestPathResult.predecessors
MinOnlyResult.dist_matrix
MinOnlyResult.predecessors
MinOnlyResult.sources
YenResult.dist_array
YenResult.predecessors
TraversalResult.node_array
TraversalResult.predecessors
ConnectedComponentsResult.n_components
ConnectedComponentsResult.labels
AssignmentResult.row_ind
AssignmentResult.col_ind
LaplacianResult.laplacian
LaplacianResult.diag
BiconnectedComponentsResult.n_components
BiconnectedComponentsResult.edges
BiconnectedComponentsResult.labels
MaximumFlowResult.flow_value
MaximumFlowResult.flow
```
- `MaximumFlowResult`를 뺀 모두가 `stardust._results`의 namedtuple이다. scipy가 튜플을 돌려주는 곳은 같은 순서라 `d, p = dijkstra(G, return_predecessors=True)`처럼 풀린다. 배열은 쓰기 금지다(API.md §2.2). 내보내지 않는다.
- `MaximumFlowResult`는 scipy처럼 클래스다. `repr`는 한 줄 `MaximumFlowResult(flow_value=5, flow=<csr_array (3, 3), nnz 6>)`이고, `flow`는 부를 때마다 새 `csr_array`다.
- 결과 객체가 아닌 결과(거리 배열, 순열, 매칭 배열, 트리, 라플라시안)는 보통의 새 배열이라 쓸 수 있다.

## §8. scipy·networkx → sd.graph

**scipy.sparse.csgraph** (scipy 1.18.1의 `__all__` 스물여섯)
| scipy | sd.graph | 마일스톤 |
|---|---|---|
| `connected_components` | 같은 이름, 레이블은 가장 작은 노드 순서 | GR1 |
| `breadth_first_order depth_first_order breadth_first_tree depth_first_tree` | 같은 이름, 이웃은 번호 오름차순 | GR1 |
| `csgraph_from_dense csgraph_to_dense` | 같은 이름 | GR1 |
| `csgraph_masked_from_dense csgraph_from_masked csgraph_to_masked` | 없음(가린 배열이 없다; 무게 0의 변은 저장된 0) | — |
| `shortest_path dijkstra bellman_ford johnson floyd_warshall yen` | 같은 이름; FW를 뺀 방법은 같은 결과, `auto`는 FW를 고르지 않는다 | GR2 |
| `construct_dist_matrix reconstruct_path NegativeCycleError` | 같은 이름 | GR2 |
| `minimum_spanning_tree maximum_flow maximum_bipartite_matching min_weight_full_bipartite_matching` | 같은 이름 | GR3 |
| `reverse_cuthill_mckee structural_rank laplacian` | 같은 이름(`laplacian`의 `form='function'`과 `copy`는 없다) | GR4 |

**scipy.optimize**
| scipy | sd | 마일스톤 |
|---|---|---|
| `linear_sum_assignment(cost_matrix, maximize=False)` | `sd.graph.linear_sum_assignment` | GR3 |

**networkx** (scipy에 없는 것)
| networkx | sd.graph | 마일스톤 |
|---|---|---|
| `topological_sort`, `lexicographical_topological_sort` | `topological_sort`(사전 최소 위상 순서 하나) | GR1 |
| `articulation_points`, `bridges`, `biconnected_components` | 같은 이름(이중 연결 요소는 변의 분할) | GR5 |
| `core_number`, `k_core` | `core_number`(k-코어는 `core_number(G) >= k`) | GR5 |
| `pagerank`, `betweenness_centrality` | 같은 이름 | GR5 |
| 군집 찾기, 무작위 그래프, 다른 중심성, 표본 매개 중심성 | 없음(나중 후보) | — |

## §9. 정확도
- 정확한 것(참값과 같다): 연결 요소, 순회, 위상 정렬, 최소 신장 트리, 역 Cuthill–McKee, 구조적 랭크, 매칭의 크기, 흐름 값, 단절점, 다리, 이중 연결 요소, 코어 수.
- 최단 경로: FW를 뺀 방법끼리 비트까지 같다. 정수 무게(모든 부분합 ≤ 2⁵³)와 `unweighted`는 정확한 최단 거리와 정확히 계산한 선행 노드 규칙과 같다. 일반 무게의 거리는 |d̂ − d\*| ≤ γ_{H−1}·max(Σ_{p̂}|w|, Σ_{p\*}|w|)이고(H는 두 경로 가운데 긴 쪽의 변 수), FW의 거리는 같은 꼴의 한계를 GR2 마일스톤에서 세어 적는다.
- `pagerank`는 ‖x̂ − x\*‖₁ ≤ tol + (반올림 바닥), 매개 중심성은 노드마다의 상대 한계, 배정은 정수 비용에서 정확한 최적, `laplacian`은 줄 합의 한계다(docs/design/graph.md §4). 참값은 정확한 유리수와 Arb다(`tests/graph/refdata.py`).
