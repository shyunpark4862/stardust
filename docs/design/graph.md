# 그래프 영역 설계 (sd.graph)

> 상태: **확정** (2026-11-23). 큰 결정(§1.1 GR1–GR3)과 비교 패키지(GR24: networkx 3.7만 설치)는 사용자가 정했고, 나머지 결정(GR4–GR23, GR25–GR68)은 추천안이다(사용자 위임). GR0–GR7 마일스톤은 완료·승인(위임)이다. 모든 공개 이름(`docs/api/graph.md`의 31개)이 구현되어 있고 이 문서는 구현과 맞다.

`sd.graph`(인접 행렬로 주는 그래프의 알고리즘: 순회, 연결 요소, 최단 경로, 신장 트리, 흐름, 매칭, 순서 매기기, 중심성)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처는 [docs/api/graph.md](../api/graph.md)에 있다.
- 비교 대상은 scipy 1.18.1의 `scipy.sparse.csgraph`(공개 이름 스물여섯)와 `scipy.optimize.linear_sum_assignment`, scipy에 없는 일곱 알고리즘(위상 정렬, 단절점, 다리, 이중 연결 요소, 코어 수, PageRank, 매개 중심성)은 networkx 3.7이다(GR24로 `.venv`에 설치). python-igraph는 설치하지 않았고, R 4.6.2에는 `igraph`를 비롯한 그래프 꾸러미가 없다(설치된 것은 `Matrix`뿐이다).
- 바닥은 배열(`csrc/array/`), 희소의 표준 꼴 CSR(`csrc/sparse/format.hpp`의 `Compressed`; sparse.md §2.2–§2.3), 공용 pairwise 합(`csrc/base/pairwise.hpp`), 스레드풀(`csrc/parallel/`)이다. 트리·흐름·라플라시안 같은 행렬 결과는 `sd.sparse.csr_array`다.
- 이 문서의 잰 값은 Apple M4 Pro, macOS 26.6.2에서 본 것이다. 다른 작업과 기기를 나눠 썼고, 부하 평균은 측정마다 적는다.
- u는 단위 반올림 2⁻⁵³, γ_k = ku/(1 − ku)다(sparse.md §4, spatial.md와 같은 기호). n은 노드 수, E는 변(저장된 원소)의 수다.

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-23)
| 번호 | 결정 |
|---|---|
| GR1 범위 | scipy 1.18.1의 `scipy.sparse.csgraph` 전부다: `shortest_path`(`method` D, BF, J, FW, auto), `dijkstra`, `bellman_ford`, `johnson`, `floyd_warshall`, `yen`, `connected_components`(weak, strong), `minimum_spanning_tree`, `breadth_first_order`, `breadth_first_tree`, `depth_first_order`, `depth_first_tree`, `maximum_flow`(dinic, edmonds_karp), `maximum_bipartite_matching`, `min_weight_full_bipartite_matching`, `laplacian`, `reverse_cuthill_mckee`, `structural_rank`, `construct_dist_matrix`, `reconstruct_path`, 변환 도구, `NegativeCycleError`. 여기에 위상 정렬, 단절점, 다리, 이중 연결 요소, k-코어, PageRank, 매개 중심성을 더한다. 나중(지금은 아님): 군집 찾기(Louvain, Leiden, 레이블 전파), 무작위 그래프, 다른 중심성. scipy가 `optimize`에 둔 `linear_sum_assignment`의 자리는 §9에서 정한다 (사용자 결정 2026-11-23) |
| GR2 꼴 | 인접 행렬(`sd.sparse` 배열이나 밀집 `sd.ndarray`; 노드는 0 … n−1)을 받는 함수들이고, 다른 영역처럼 배열이나 결과 객체를 돌려준다. Graph 클래스는 두지 않는다 (사용자 결정 2026-11-23) |
| GR3 위치 | 최상위 `sd.graph`, 평평한 이름공간이다 (사용자 결정 2026-11-23) |
| GR4 `linear_sum_assignment`의 자리 | §9의 1번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR5 최단 경로의 결과를 방법과 무관하게 할까 | §9의 2번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR6 선행 노드의 동점 규칙 | §9의 3번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR7 받는 입력 | §9의 4번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR8 유한하지 않은 무게 | §9의 5번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR9 음수 무게의 `dijkstra` | §9의 6번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR10 색인 결과의 dtype과 표식 | §9의 7번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR11 결과의 꼴 | §9의 8번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR12 연결 요소의 번호 | §9의 9번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR13 무향 순회의 이웃 순서 | §9의 10번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR14 scipy의 옵션 가운데 뺄 것 | §9의 11번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR15 흐름의 용량 | §9의 12번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR16 흐름·매칭·배정의 동점 | §9의 13번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR17 최소 신장 트리의 동점과 자리 | §9의 14번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR18 더한 알고리즘의 이름과 꼴 | §9의 15번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR19 PageRank의 멈춤과 기본값 | §9의 16번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR20 매개 중심성의 범위 | §9의 17번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR21 옵션의 자리 | §9의 19번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR22 노드 인자의 꼴 | §9의 20번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR23 `limit`의 NaN | §9의 21번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| GR24 비교 패키지 | networkx(3.7)만 `.venv`에 설치해 scipy에 없는 일곱 알고리즘의 뜻과 결과를 견준다. igraph는 설치하지 않는다(§9의 18번 (a)) (사용자 결정 2026-11-23) |
| GR25 `'FW'`의 반올림 | GR0 마일스톤의 측정(§3.2)에서 원점마다의 밀집 Dijkstra가 블록 min-plus Floyd–Warshall보다 1스레드에서 1.8–2.5배, 10스레드에서 1.1–2.4배 느려 §9의 2번의 규칙대로 (b)로 바꾼다: `method='FW'`와 `floyd_warshall`은 블록 Floyd–Warshall(64 × 64 블록, 블록 하나가 태스크 하나라 스레드 수와 무관)의 반올림이고, 거리는 계약 안이다(§4). 다른 방법(D, BF, J, auto)의 결과는 GR5 그대로 한 정의의 같은 비트다 (추천안, 사용자 위임 2026-11-23) |
| GR26 `auto`는 FW를 고르지 않는다 | `method='auto'`는 음수 무게가 있으면 J, 없으면 D이고, `indices`가 없고 변이 n²/4개 이상이면 D와 같은 비트의 힙 없는 밀집 Dijkstra(§3.2)로 빠르게 한다. 그래서 GR25 뒤에도 auto의 결과가 밀도의 문턱에 따라 바뀌지 않는다. `[scipy와 다름]`: scipy의 auto는 그 밀도에서 FW다. 밀집 Dijkstra는 scipy의 `floyd_warshall`보다 1스레드에서 2.5배 빠르다(n = 1024) (추천안, 사용자 위임 2026-11-23) |
| GR27 FW의 선행 노드 | FW의 거리는 원점에서부터의 차례 합이 아니어서 §3.2의 꼭 맞는 변이 없는 노드가 생길 수 있다. 그래서 FW의 선행 노드는 Floyd–Warshall의 갱신(더 작아질 때 pred(i, j) ← pred(k, j), 같으면 그대로; scipy와 같은 규칙)이 정한다. 블록의 순서가 고정이라 결정적이다. `construct_dist_matrix`가 `shortest_path`의 거리와 비트까지 같다는 약속은 FW가 아닌 방법에서만이다. FW의 음수 순환은 끝난 뒤 대각이 음수인 노드로 알린다 (추천안, 사용자 위임 2026-11-23) |
| GR28 PageRank의 자기 고리 | `pagerank`는 자기 고리를 변으로 센다(나가는 무게에 들고 자기에게 돌아간다; networkx의 뜻). 다른 함수의 자기 고리 규칙(§2.1)은 그대로다 (추천안, 사용자 위임 2026-11-23) |
| GR29 무향 무게의 가장자리 | 두 방향이 다 저장되면 작은 무게이고, 어느 하나가 NaN이면 NaN(numpy의 `minimum`), 0.0과 −0.0이면 −0.0이다. 그래서 G와 Gᵀ의 무향 읽기가 비트까지 같다. 무향 트리(`breadth_first_tree`, `depth_first_tree`)의 저장된 0은 무게 0의 변으로 남는다. `[scipy와 다름]`: scipy의 무향 트리는 저장된 0을 변 없음으로 읽어 한 방향만이면 inf, 두 방향이면 다른 쪽의 무게를 넣는다(GR1 마일스톤에서 확인) (추천안, 사용자 위임 2026-11-23) |
| GR30 `csgraph_to_dense`의 저장된 NaN | 저장된 값을 그대로 쓴다(NaN은 NaN). `[scipy와 다름]`: scipy는 중복의 최솟값을 +inf에서 시작해 채우는 꼴이라 저장된 NaN이 inf가 된다. sd의 CSR에는 중복이 없어 그 꼴이 필요 없다 (추천안, 사용자 위임 2026-11-23) |
| GR31 약한 연결 요소의 병렬 | 저장된 원소가 2¹⁸개 이상이고 풀을 쓸 수 있으면 변 구간 태스크(원소 2¹⁶개씩, 크기로만 나눈다)가 비교 교환으로 큰 뿌리를 작은 뿌리 밑에 잇는 합집합–찾기, 그 밖은 직렬이다. 분할이 유일하고 뿌리가 요소의 가장 작은 노드라 번호가 같다. GR1 마일스톤의 측정(ms, 1스레드 / 10스레드): 원소 10⁵개 0.096 / 0.262, 2 × 10⁵ 0.253 / 0.198, 5 × 10⁵ 0.80 / 0.41, 10⁶ 1.74 / 0.77 (추천안, 사용자 위임 2026-11-23) |
| GR32 무향 순회의 읽기 | 합친 CSR을 만들지 않는다: sparse의 전치(§3.1)를 한 번 만들고, G가 자기 전치와 같으면(패턴과, 트리면 무게까지) G만, 아니면 G와 전치의 두 줄을 함께 읽는다. 두 줄은 원소마다 합치지 않고 줄마다 찾은 노드를 따로 건너뛰며, 너비 우선은 새로 찾은 두 오름차순 묶음만 합치고 깊이 우선은 두 줄의 머리 가운데 작은 것을 고른다. 원소마다 합치는 꼴은 비교마다 앞의 비교를 기다려 한 줄 꼴보다 네 배 느렸다(n = 10⁵, d = 10의 무향 너비 우선 10.9 ms → 4.8 ms, scipy 6.7 ms) (추천안, 사용자 위임 2026-11-23) |
| GR33 속도 사례의 크기 | 사례의 크기는 인접 행렬의 모양 `(n, n)`이고(bench/README.md의 id 표기) 나가는 차수 d와 대칭·DAG 같은 그래프의 종류는 사례 이름에 둔다(`connected_components weak d=10 [(100000, 100000)]`) (추천안, 사용자 위임 2026-11-23) |
| GR34 `unweighted=True`는 값을 읽지 않는다 | 최단 경로 함수(`yen` 포함)의 `unweighted=True`는 패턴만 읽는다: 저장된 NaN·±inf와 음수 무게를 막지 않고(`dijkstra(G, unweighted=True)`도 음수 무게의 그래프를 받는다), 모든 변이 무게 1이다. 무게 없는 `betweenness_centrality`와 같은 구조 함수의 규칙(§2.1)이다 (추천안, 사용자 위임 2026-11-23) |
| GR35 음수 순환의 공통 검사 | 저장된 음수 무게가 있으면 BF, J, auto(J), `yen`은 계산 전에 같은 검사 하나를 한다: 요청한 원점에서 닿는 노드(패턴의 너비 우선)를 무게 0의 변으로 잇는 가상 뿌리에서 큐 Bellman–Ford(닿는 노드를 오름차순으로 넣는다)를 돌리고, 노드가 닿는 노드 수보다 많이 다시 들어가면 부모 포인터의 순환을 찾아 그 순환의 가장 작은 노드를 알린다(음수 자기 고리는 그 자체로 순환이다). 그래서 메시지의 노드는 방법과 스레드 수와 무관하다(그 검사가 찾은 순환의 가장 작은 노드이고, 모든 음수 순환의 가장 작은 노드는 아니다). 검사가 끝나면 그 값이 J의 퍼텐셜이다. FW는 끝난 뒤 대각의 음수로 알고 같은 검사(모든 노드)로 노드를 정한다(검사가 순환을 못 찾으면 음수 대각의 가장 작은 노드) (추천안, 사용자 위임 2026-11-23) |
| GR36 auto의 밀집 길과 음수 무게 | auto의 밀집 Dijkstra는 음수 무게가 없을 때만이다. 음수 무게면 밀도와 상관없이 CSR 위의 J다(밀집 J와 같은 비트라 결과는 같고 코드가 하나 준다). 밀도의 문턱은 입력의 저장된 원소 수로 센다(4·nnz ≥ n², 무향도 입력의 nnz; scipy와 같은 셈) (추천안, 사용자 위임 2026-11-23) |
| GR37 FW의 계약 | FW의 거리는 §4의 3번과 같은 한계 \|d̂ − d\*\| ≤ γ_{H−1}·max(Σ_{p̂}\|w\|, Σ_{p\*}\|w\|), 상수 1이다(p̂는 FW의 선행 노드가 그리는 경로). 블록 FW의 값은 어떤 걸음의 덧셈 나무의 반올림이고 덧셈 나무가 어떤 꼴이든 H개 항의 합은 γ_{H−1}·Σ\|w\| 안이며(Higham), fl(+)와 min이 단조라 그 값은 참 최단 경로를 FW가 잇는 나무의 반올림 이하다. 그래서 블록의 합 깊이를 셀 필요가 없다. GR2 마일스톤의 측정(일반 무게의 참값 7,639점)에서 가장 큰 값이 0.995 단위로 D와 같았다 (추천안, 사용자 위임 2026-11-23) |
| GR38 `construct_dist_matrix`·`reconstruct_path`의 선행 노드 | 정수 dtype이면 받는다(scipy의 int32도 int64로 바꾼다; 실수·bool은 `TypeError`). 모양은 (n, n)·(n,)이고 아니면 `ValueError`. 노드도 −9999도 아닌 값, 그래프에 없는 변(pred → v), 순환, 원점이 아닌 노드에서 끝나는 사슬, 자기를 가리키는 선행 노드(`reconstruct_path`)는 `ValueError`이고 메시지가 그 자리를 짚는다. 대각의 값은 범위만 보고 읽지 않는다(scipy와 같다). 무게를 쓰는 함수라 저장된 NaN·±inf는 `ValueError`(GR8). `[scipy와 다름]`: scipy의 `construct_dist_matrix`는 int32만 받고 없는 변에 inf를 더하며 순환에서 끝나지 않고, `reconstruct_path`는 없는 변을 말없이 0으로 넣거나 버린다(GR2 마일스톤에서 확인) (추천안, 사용자 위임 2026-11-23) |
| GR39 Dijkstra의 큐 | 이진 힙의 게으른 지우기 대신 노드마다 한 칸인 4진 색인 힙(키가 작아지면 그 자리에서 올린다)이다. 결과는 고정점이라 같고, n = 10⁵, d = 10의 원점 하나가 17.4 ms → 10.4 ms, n = 2000의 모든 쌍이 388 → 209 ms(1스레드)가 됐다. J(다시 넣기는 꺼낸 노드를 다시 넣는다)와 `yen`의 갈래 탐색도 같은 힙이다 (추천안, 사용자 위임 2026-11-23) |
| GR40 원점 태스크 | 여러 원점의 계산은 태스크 하나가 원점 1–64개(크기로만: 4096 / (n + E))를 맡고, 원점마다 자기 행을 쓴다. 원점 하나의 답이 정의로 정해지므로 나누기와 스레드 수가 결과를 바꾸지 않는다. 작업량(행 수 × (n + E))이 2¹⁵ 이상이면 GIL을 놓는다 (추천안, 사용자 위임 2026-11-23) |
| GR41 `yen`의 갈래 탐색 | 갈래 노드에서 뿌리의 차례 합으로 시작하는 Dijkstra가 싱크를 꺼낸 뒤 같은 키의 노드까지 꺼내고 멈춘다(무게 0의 변이 그 노드를 싱크에 이을 수 있다). 그 거리의 꼭 맞는 변을 갈래 노드에서의 너비 우선 변 수로 층을 나누고, 싱크에 닿는 층의 변만 남겨 갈래 노드에서 가장 작은 다음 노드를 차례로 고른다: (합, 변 수, 노드 열)로 가장 앞선 갈래 경로다. 음수 무게면 공통 검사(GR35)의 퍼텐셜로 다시 넣는 탐색을 끝까지 한다. `K`는 int 범위 밖도 받고(있는 경로만큼) `return_predecessors=True`면 K × n × 8바이트를 물리 메모리와 견준다 (추천안, 사용자 위임 2026-11-23) |
| GR42 FW의 선행 노드와 거리 | 선행 노드를 함께 구해도 거리는 같은 비트다: 두 커널이 같은 FMIN(−0과 +0이면 −0)으로 거리를 정하고, 선행 노드는 합이 더 작을 때만(FMIN이 −0을 고르는 같은 값은 그대로) 바꾼다. 선행 노드를 쓰는 커널은 4 × 4 타일이라 거리만의 4 × 8 타일보다 2배 느리다(n = 1024, 192 ms 대 92 ms, 1스레드) (추천안, 사용자 위임 2026-11-23) |
| GR43 최소 신장 트리의 계산 | 무향 변을 (min(i, j), max(i, j)) 순서로 만들고 (w + 0.0의 순서 키, 변 번호)를 안정 LSD 기수 정렬(11비트 자리, 모든 키가 같은 자리는 건너뜀)로 정렬한 뒤 Kruskal(합집합–찾기)로 고른다. 안정 정렬이라 같은 무게는 (min, max) 순서로 남아 GR17의 키 순서 그대로이고, −0.0과 0.0은 같은 값으로 정렬된다. 직렬이다. Borůvka는 넣지 않는다: GR3 마일스톤의 측정(n = 10⁵, d = 5 대칭, 1스레드)에서 이 Kruskal이 11.1 ms로 scipy(76.2 ms)의 6.9배였다(GR6 마일스톤에서 큰 그래프를 다시 본다). 고른 변의 자리는 그 무게와 비트가 같은 저장 원소이고, 두 방향의 비트가 같으면 위 삼각이다(0.0과 −0.0이면 −0.0의 자리). 결과는 입력의 저장 원소를 거른 것이라 표준 꼴이고 값은 저장된 그대로다 (추천안, 사용자 위임 2026-11-23) |
| GR44 흐름의 잔여 그래프와 결과의 패턴 | 잔여 그래프는 G와 Gᵀ의 패턴(대각의 저장 원소도 용량 0으로 남긴다)이고 반대 원소의 자리를 쥔다. 잔여 용량 c − f는 c(u, v) + c(v, u)까지 커져 int64를 넘을 수 있어 uint64로 쥐고(0 ≤ c − f < 2⁶⁴라 정확하다) 끝에 f = c − (c − f)로 읽는다. 결과 `flow`의 패턴은 그 잔여 패턴이다: 저장된 용량 0의 변과 대각의 저장 원소도 흐름 0으로 들어간다(scipy와 같은 패턴). 원점 용량 합의 넘침 검사(GR15)는 원점의 자기 고리를 빼고 센다(흐름에 쓰이지 않는다). 두 방법은 잔여 행을 오름차순으로 훑으므로 GR3 마일스톤의 측정에서 용량이 int32 안이면 scipy의 흐름과 배열까지 같았다(무작위 망 8개와 n = 10⁴, d = 10의 망, 두 방법) (추천안, 사용자 위임 2026-11-23) |
| GR45 배정 풀이의 시작 | 두 배정 함수의 공용 풀이는 Jonker–Volgenant(LAPJV)다: 정방이면 열 줄이기와 줄인 값 옮기기, 그다음 모양과 상관없이 증가 행 줄이기 두 번(한 번에 행마다 4걸음까지; 넘으면 풀린 행은 다음 번이나 탐색으로 간다), 남은 행마다 오름차순으로 최단 증가 경로(축소 비용의 Dijkstra와 Crouse의 쌍대 갱신)다. 탐색은 열을 (거리, 빈 열 먼저, 번호)의 전순서로 고르고 시작의 걸음도 행을 열 번호 오름차순으로 훑어, 같은 짝이면 밀집과 희소가 같은 배정을 낸다. 직사각의 증가 행 줄이기는 v = 0에서 시작해 v만 낮추므로 빈 열의 v = 0(쌍대 조건)이 지켜진다. 시작이 없는 꼴(scipy의 `linear_sum_assignment`와 같은 Crouse)은 동점에서 scipy와 같은 배정을 냈지만 느렸다(GR3 마일스톤, 1스레드, ms: 밀집 1000² 21.5, 희소 꼴 1000² 56, 희소 5000(d = 20 + 대각) 69; 걸음의 한도 없는 증가 행 줄이기는 54, 72, 43). 이 꼴은 6.95, 10.8–13.0, 16.9이고 scipy는 20.1, 11.1–26.9, 23.1이다. 그래서 동점에서 고르는 최적 배정은 scipy와 다를 수 있다 (추천안, 사용자 위임 2026-11-23) |
| GR46 `linear_sum_assignment`의 `maximize`와 ±inf | scipy의 뜻 그대로 비용을 먼저 부호를 바꾼다: `maximize=False`면 +inf가 고를 수 없는 짝이고 NaN·−inf가 `ValueError`, `maximize=True`면 −inf가 고를 수 없는 짝이고 NaN·+inf가 `ValueError`다(`linear_sum_assignment(): 'cost_matrix' must not hold NaN or +inf`). `min_weight_full_bipartite_matching`의 `maximize`도 저장된 무게의 부호를 바꾼 최소화다 (추천안, 사용자 위임 2026-11-23) |
| GR47 가장 큰 이분 매칭의 차례 | Hopcroft–Karp 앞에 탐욕 시작(행마다 오름차순으로 가장 작은 빈 열)을 두고, 단계마다 빈 행(오름차순)에서의 너비 우선 층과 빈 행마다의 깊이 우선(현재 호, 이웃 오름차순)으로 겹치지 않는 가장 짧은 증가 경로를 찾는다. 매칭의 크기는 유일하고 어떤 매칭인지는 이 차례가 정한다(GR16). scipy와 크기는 같고 매칭은 다를 수 있다. n = 10⁵, d = 10에서 21.4 ms(scipy 31.1) (추천안, 사용자 위임 2026-11-23) |
| GR48 밀집 입력의 메모리 꼴 | 밀집 입력(그래프, 비용, 용량, 선행 노드, `indices`)은 C 순서가 아니면(열 조각, F 순서, 거꾸로 걷는 보기) 한 번 C 순서로 복사한다(`_common.contiguous`). GR1 마일스톤의 꼴은 `reshape`가 보기를 돌려줄 때 복사하지 않아 `connected_components(b[:, ::2])`가 내부의 `TypeError: _dense_csr(): wrong matrix`를 냈다(다른 세션이 찾음; GR3 마일스톤에서 고치고 tests/graph/test_layouts.py로 지킨다) (추천안, 사용자 위임 2026-11-23) |
| GR49 배정의 계약 상수 | 일반 비용의 한계 C(σ̂) ≤ C\* + γ_{2r}·Σᵢ(\|c_{iσ̂(i)}\| + \|c_{iσ\*(i)}\|)의 상수는 1이다. 쌍대 변수의 반올림을 하나하나 세어 증명하지는 않았고, 정확한 유리수 풀이(tests/graph/definitions.py `assignment_exact`)의 참 최적과 견준 GR3 마일스톤의 측정(14영역: 참값의 실수 두 점, 10⁻⁴–10⁴의 넓은 범위와 양·음의 무작위 정방·직사각, 1000의 배수에 10⁻¹⁰ 아래의 차이만 있는 거의 동점; 최소·최대)에서 sd와 scipy가 모두 참 최적을 냈다(0 단위). 정수 비용은 정확하다: 쌍대 변수가 정수라 u(i) + v(j) ≤ c(i, j)와 고른 짝의 등식, 빈 열의 v ≤ 0을 정수로 확인한다(테스트, 비용 2⁴⁰까지) (추천안, 사용자 위임 2026-11-23) |
| GR50 정규화 라플라시안의 대각 | scipy처럼 차수가 0이 아니면 정확히 1, 0이면 0이다(이 초안 §3.6의 d_i / w_i / w_i를 바꾼다). 참 D^(−1/2) L D^(−1/2)의 대각은 d > 0에서 정확히 1이라 반올림할 것이 없고, 참값(tools/graph_ref.py)도 그렇게 만들었다. 차수가 NaN이거나 음수면 대각 밖은 NaN(√)이고 대각은 1이다(scipy와 같다). `return_diag`의 `diag`는 w = √d(d = 0이면 1)다(scipy와 같다) (추천안, 사용자 위임 2026-11-23) |
| GR51 `form='lo'` | `form='array'`와 같은 행렬(희소 입력이면 CSR, 밀집이면 `sd.ndarray`)을 만들어 `sd.sparse.aslinearoperator`로 감싼다. 그래서 곱이 `laplacian(G) @ x`와 비트까지 같고 `rmatvec`(Lᵀ의 곱)이 있으며 sparse의 병렬 곱을 그대로 쓴다. 행렬 하나(희소: 대각 밖의 저장 원소 + n)만큼 메모리가 더 든다. `[scipy와 다름]`: scipy는 행렬을 만들지 않고 v·(대각을 넣은 줄 합) − A v로 곱해 대각이 더해졌다 빠지므로 끝자리가 다르다. 행렬 없는 곱을 `laplacian(G) @ x`와 같은 비트로 하려면 sparse의 곱 규칙(sparse/arith.hpp의 fma와 4096항 조각)을 옮겨야 해서, 그 이득(입력만 한 메모리)은 GR6 마일스톤에서 다시 본다. §2.5의 "행렬을 만들지 않는다"를 바꾼다 (추천안, 사용자 위임 2026-11-23) |
| GR52 역 Cuthill–McKee의 정의 | scipy 1.18.1의 `_reverse_cuthill_mckee`와 같은 알고리즘이다: 노드의 차수는 읽는 패턴의 줄 길이에 저장된 대각이면 1을 더한 것(자기 고리의 두 끝), 요소마다 방문하지 않은 노드 가운데 (차수, 번호)로 첫 노드에서 너비 우선, 새 이웃은 (차수, 번호) 순서로 큐에 넣고, 요소는 시작 노드의 순서로 이은 뒤 전체를 뒤집는다. 이웃 묶음은 정렬로 줄 세운다(scipy의 삽입 정렬은 별 그래프에서 O(n²)이고 결과는 같다). `[scipy와 다름]` 둘: (1) scipy는 시작 노드의 순서를 int32 차수의 numpy 불안정 `argsort`로 정해 같은 차수의 시작 노드가 번호 순서가 아닐 수 있고, 그 순서는 numpy의 정렬 구현(기계의 SIMD 갈래)에 달려 있다. GR4 마일스톤에서 scipy의 순열은 무작위 그래프 688개(n = 5–10⁴)에서 그 `argsort`를 넣은 이 정의와 모두 같았다. numpy는 16개 이하를 삽입 정렬(안정)로 정렬해 sd는 16노드 이하에서 scipy와 같았고, 17노드부터는 대부분 달랐다(탐침의 무작위 그래프 400개 가운데 224개). (2) `symmetric_mode=False`에서 scipy는 A + Aᵀ를 값으로 더해 0이 되는 자리(한 방향의 저장된 0, A_ij = −A_ji)를 변으로 보지 않고, sd는 두 패턴의 합집합을 읽는다(구조 함수의 규칙, §2.1) (추천안, 사용자 위임 2026-11-23) |
| GR53 라플라시안의 패턴과 −0.0 | `symmetrized=True`의 A + Aᵀ는 sd.sparse의 덧셈 규칙(정확히 0인 합은 저장하지 않는다; scipy의 덧셈도 같다)이라 결과의 패턴이 scipy와 같다. `symmetrized=False`면 A의 저장된 0이 그 자리의 −0.0으로 남는다(scipy와 같다). 결과의 패턴은 그 대각 밖 원소와 대각 전부(차수 0도 저장된 0)다. 밀집 입력은 대각 밖의 모든 0이 −0.0이다(scipy의 `m *= -1`과 같다). GR0 마일스톤의 탐침이 −0.0을 보지 못한 것은 `toarray()`가 +0.0 위에 더해 부호를 지운 탓이다(scipy의 `coo_array.data`에는 −0.0이 있다; tests/graph/test_laplacian.py). 부호 바꾸기는 NaN을 그대로 둔다(scipy의 −1 곱과 같은 비트) (추천안, 사용자 위임 2026-11-23) |
| GR54 차수의 합과 dtype | 차수는 대각을 뺀 원소를 색인 오름차순으로 0부터 하나씩 더한다(들어오는 차수는 행 오름차순의 열 합; sparse의 4096항 조각 규칙은 쓰지 않는다). `[scipy와 다름]`: scipy는 대각을 넣은 합에서 대각을 빼서 끝자리가 다르고 대각이 inf나 NaN이면 차수가 NaN이다. 정규화 원소는 −((A_ij / w_i) / w_j)로 밀집과 희소가 같은 순서다(scipy의 희소와 같고, scipy의 밀집은 w_j로 먼저 나눠 끝자리가 다를 수 있다). 계산 dtype은 `normed`이거나 float64 입력이면 float64, 아니면 int64(bool은 int64; 합은 감긴다, numpy와 같다)이고 `dtype`(None, int64, float64)은 끝에 바꾼다(scipy와 같다). bool·complex128의 `dtype`은 `TypeError`다. `[scipy와 다름]`: scipy는 bool 그래프의 비정규화 라플라시안에서 numpy의 `UFuncTypeError`를 낸다 (추천안, 사용자 위임 2026-11-23) |
| GR55 라플라시안의 계약 상수 | 항 m개의 차수는 \|d̂ − d\| ≤ γ_k·s(k = m, 대칭화면 항이 반올림된 합이라 m + 1; s는 그 항을 이루는 \|A\|의 합), 대칭화의 대각 밖은 γ₁·(\|A_ij\| + \|A_ji\|), 정규화의 대각 밖은 \|L̂ − L\| ≤ \|L\|·((1 + γ_c)/√((1 − ρ_i)(1 − ρ_j)) − 1)(ρ = γ_k·s/\|d\|, c = 4, 대칭화면 5), 그 밖(비정규화의 대각 밖, 정규화의 대각)은 정확하다. ρ ≥ 1이면(반올림이 차수를 0 너머로 옮길 수 있으면) 약속이 없다. 근거: ŵ = √d·√(1 + η)·(1 + δ₁)(\|η\| ≤ ρ), 두 나눗셈이 (1 + δ₂)(1 + δ₃)이고 (1 + u)²/(1 − u)² ≤ 1 + γ₄다. GR4 마일스톤의 측정(참값 51점의 float 그래프 11영역 4,943점)에서 가장 큰 값이 0.87 단위(dangling/50 대칭화의 대각 밖), 정규화는 0.32 단위였다. scipy도 같은 한계 안이었다(0.87, 0.32) (추천안, 사용자 위임 2026-11-23) |
| GR56 PageRank의 멈춤과 계약 | 한 걸음의 반올림 ε(GR57의 K로 ε = γ_K·‖T(x̂_k)‖₁)를 넣은 사후 한계 (α‖x̂_{k+1} − x̂_k‖₁ + ε)/(1 − α)가 `tol` 이하면 멈추고(계산한 ℓ₁ 거리와 Σx̂를 위로 어림하고 1 + 2⁻⁴⁰을 곱한다), 아니면 사전 횟수 k_max = ⌈log(tol/2)/log α⌉ + 1(‖x₀ − x\*‖₁ ≤ 2라 2α^k ≤ tol이면 넉넉하다; α = 0이면 1)에서 멈춘다. 그래서 ‖x̂ − x\*‖₁ ≤ tol + F, F = γ_K·(1 + 10⁻⁹)/(1 − α)다(§4). 이 초안의 반복 수 식(log 안의 (1 − α))과 §4의 γ_{m+3}을 바꾼다: 바닥 근처에서 반복은 2αε/(1 − α)²까지 흔들려 사후 한계만으로는 멈추지 않을 수 있고, 사전 횟수가 멈춤을 보장한다. 처음 값은 v이고 결과는 마지막 반복 그대로다(합을 다시 1로 나누지 않는다; 합은 1과 반올림만큼 다르다). `tol`은 양의 유한한 수, `alpha`는 0 이상 1 미만이고 아니면 `ValueError`다. GR5 마일스톤의 측정(pagerank.npz 25점): tol = 10⁻¹²에서 가장 큰 값이 0.79 단위(components/40), tol = 10⁻¹⁵(대부분 바닥 F 아래라 사전 횟수에서 멈춘다)에서 0.022 단위, ulp로 14 이하였다. 무작위 그래프(n = 10⁵, d = 10)는 25번에 멈췄다(사전 횟수 176) (추천안, 사용자 위임 2026-11-23) |
| GR57 PageRank의 계산 | 변의 전이 확률 w/out(j)(out(j)는 그 행의 pairwise 합)를 전치 CSR에 한 번 만들고, 무게 없음은 걸음마다 x(j)/차수(j)를 만든다. 나가는 무게가 0인 노드(저장된 무게가 모두 0인 행도)는 매달린 노드이고 그 변의 확률은 0이다(networkx와 같다). 행의 합은 sparse의 4096개 조각이 아니라 공용 pairwise 나무(csrc/base/pairwise.hpp)다: 깊이가 로그라 큰 허브에서도 바닥이 작다(차수 10⁵의 줄을 차례로 더하면 γ_{10⁵} ≈ 1.1·10⁻¹¹로 기본 `tol`보다 크다). 행은 원소와 행 2¹⁵개쯤의 구간(패턴으로만 정함)이 태스크이고, 매달린 질량, ‖x̂_{k+1} − x̂_k‖₁, Σx̂는 같은 나무를 2¹⁶항 잎으로 자른 합(pairwise의 `plan`·`combine`)이라 스레드 수와 무관하다. K는 항 하나가 거치는 반올림의 수다: max(t + h(m_in) + 3, h(n_d) + h(n) + 5, h(n) + 4), t = h(m_out) + 2(무게 있음; 무게 없음 1), h는 pairwise 나무의 깊이(tests/graph/refdata.py `pairwise_depth`), m_in·m_out은 가장 긴 들어오는·나가는 줄, n_d는 매달린 노드 수다(모든 항이 음이 아니라 상쇄가 없다). 걸음 하나가 같은 크기 sd.sparse 곱(A @ x)의 1.47배다(n = 10⁵, d = 10; 1스레드 0.92 대 0.62 ms, 10스레드 0.135 대 0.092 ms; 짧은 줄을 나무 대신 한 사슬로 쓰고 두 벡터를 바꿔 쓰기 전에는 1.65배) (추천안, 사용자 위임 2026-11-23) |
| GR58 매개 중심성의 DAG와 계산 | DAG의 변은 꼭 맞는 변(fl(d(u) + w) = d(v)) 가운데 (d(u), h(u)) < (d(v), h(v))인 것이고, h는 원점에서 꼭 맞는 변을 따른 너비 우선 변 수(§3.2의 선행 노드 규칙과 같은 h)다. 이 초안 §3.7의 (d, 번호)를 바꾼다: 흡수(fl(d + w) = d)가 만든 같은 거리의 걸음이 번호가 큰 노드에서 작은 노드로 가면 그 노드가 선행 노드 없이 남아 σ = 0이고 비가 0/0이 됐다(tests/graph/test_centrality.py의 흡수 그래프). 흡수가 없으면 모든 꼭 맞는 변이 d(u) < d(v)라 표준 Brandes와 같다. σ는 들어오는 변을, δ는 나가는 변을 번호 오름차순으로 끌어와 더하므로 합의 순서가 처리 순서와 무관하다. σ는 double이고 한 원점에서 2⁹⁶⁰에 닿으면 그 원점을 (double 가수, int64 지수) 꼴로 다시 센다(덧셈 하나에 반올림 하나라 둘 다 되는 곳에서 같은 비트다; 다이아몬드 1,100개의 사슬, 경로 2¹¹⁰⁰개도 정확하다). 원점은 ⌈n/256⌉개(1–64)씩 태스크이고 4 × 스레드 수씩(부분합 128 MB까지) 돌린 뒤 부분합을 태스크 번호 순서로 더한다. 무게는 저장된 모든 원소(대각 포함)가 양의 유한한 수여야 한다. 비율은 networkx 3.7의 `_rescale`이다: N = n(`endpoints`) 또는 n − 1, N < 2면 그대로(값이 모두 0이다), 정규화는 1/(N(N − 1)), 정규화 없는 무향은 1/2. 계약의 K = L(2A + Δ_out + 2) + n + 3(L은 가장 긴 DAG 경로의 변 수, Δ_out·Δ_in은 가장 큰 DAG 진출·진입 차수, A는 모든 σ < 2⁵³이면 0(정확하다) 아니면 L(Δ_in − 1); 비 σ(u)/σ(w)가 2⁻¹⁰²² 아래로 떨어지지 않는다고 본다). GR5 마일스톤의 측정(betweenness.npz 76점, 노드값 2,360개)에서 가장 큰 값이 0.076 단위(diamond/4), ulp로 5 이하였고 networkx도 같은 한계 안이었다 (추천안, 사용자 위임 2026-11-23) |
| GR59 무향 구조 넷의 계산 | 패턴이 자기 전치와 같고 저장된 대각이 없으면 그 CSR을 그대로 읽고, 아니면 두 방향의 합집합에서 대각을 뺀 CSR을 만든다. 단절점·다리·이중 연결 요소는 깊이 우선 하나(뿌리와 이웃 오름차순, 부모로 돌아가는 나무 변은 노드로 건너뛴다; 중복 변이 없다)이고, 변 번호(e는 (i, j), i < j의 행 순서라 사전 순서)와 변 스택은 `biconnected_components`에서만 만든다. 다리는 찾은 뒤 정렬한다. 코어 수는 Batagelj–Zaversnik의 통 정렬이다. 모두 직렬이다. `[networkx와 다름]`: `core_number`는 자기 고리를 무시한다(networkx는 `NetworkXNotImplemented`), `biconnected_components`는 변의 분할이다(networkx의 `biconnected_component_edges`와 같은 분할; 노드 집합은 `edges[labels == c]`의 노드). GR5 마일스톤의 측정(ms, 1스레드, 단절점 / 다리 / 이중 연결 / 코어 수): n = 10⁴, d = 5 대칭 0.75 / 0.84 / 1.33 / 0.58(networkx 22.5 / 130 / 29.4 / 19.1), n = 10⁵, d = 10 대칭 14.3 / 14.3 / 23.1 / 10.0(같은 그래프의 sd `connected_components` 3.0). 변 번호를 단절점·다리에서 빼고 대칭 입력을 그대로 읽기 전에는 26.0 / 26.3 / 26.7 / 13.7이었다 (추천안, 사용자 위임 2026-11-23) |
| GR60 FW의 블록 배치 | 블록 Floyd–Warshall은 블록마다 연속한 작업 사본(블록 (bi, bj)의 64 × 64 원소가 32 KB 한 덩어리, 행 간격 64)에서 돌고 끝에 결과로 옮긴다. 행 우선 행렬에서는 블록의 64행이 N × 8바이트 간격이라(N = 1024면 8 KB) 1차 캐시의 몇 집합에 몰려 밀려났다. 연산의 순서는 그대로라 비트가 같다. GR6 마일스톤의 측정(n = 1000, 번갈아 잰 가장 좋은 값): 1스레드 92.2 → 77.6 ms(scipy 377–383의 4.9배), 10스레드 15.3 → 10.7 ms(1스레드의 7.3배). 독립 블록 커널은 이제 원소 갱신 14.7개/ns로 fmin의 처리량(fadd의 약 0.6배)에 묶인다. 작업 사본은 N ≠ n일 때만이던 것이 늘 N²(선행 노드도)라, `room` 검사가 결과와 사본을 함께 센다 (추천안, 사용자 위임 2026-11-23) |
| GR61 대칭 확인 | 무향 읽기가 쓰는 "G가 자기 전치와 같은가"는 전치를 만들지 않고 행마다 커서 하나로 한 번 훑는다(행 u의 원소 (u, v)는 대칭이면 행 v에서 아직 쓰지 않은 다음 원소와 짝이다; 무게를 읽으면 GR29의 규칙으로 견준다). 같으면 순회·최단 경로(`paths::reading`)·매개 중심성·PageRank·구조 넷·`symmetric_mode=False`의 역 Cuthill–McKee가 G를 그대로 읽는다(전에는 순회가 전치를 만든 뒤 견주었고, 최단 경로는 합친 CSR을 늘 만들었다). 측정: 대칭 n = 10⁵, d = 10의 무향 `breadth_first_order` 7.67 → 5.77 ms(1스레드), 대칭 n = 10⁴, d = 5의 `articulation_points`·`bridges` 1.11·1.13배 (추천안, 사용자 위임 2026-11-23) |
| GR62 모든 짝을 저장한 희소 배정 | `min_weight_full_bipartite_matching`의 입력이 rows × cols개를 모두 저장하면 CSR의 값이 곧 행 우선 밀집 행렬이라 밀집 훑기의 풀이(`_assign_dense`)로 푼다. GR45의 전순서 때문에 같은 짝이면 두 훑기가 같은 배정을 내고, 무작위 정방·직사각 6,000문제(정수 동점, 실수, 최소·최대)에서 배열까지 같았다. 밀집 1000²의 `csr_array` 11.7 → 7.86 ms(scipy 11.7) (추천안, 사용자 위임 2026-11-23) |
| GR63 PageRank의 준비 | 나가는 무게의 합을 행 구간 태스크로 나누고, 유향의 전치는 풀이 있으면 sparse의 병렬 전치를 만들어 그 값 자리에서 전이 확률로 바꾼다(1스레드는 확률을 쓰며 세는 정렬 하나 그대로; 둘 다 표준 꼴의 전치라 줄과 항이 같다). 무게 없음의 걸음마다 x(j)/차수(j)도 행 구간 태스크다. n = 10⁵, d = 10에서 10스레드 7.50 → 5.21 ms, 1스레드는 그대로(29.0 → 28.5) (추천안, 사용자 위임 2026-11-23) |
| GR64 라플라시안의 대칭화와 밀집 | 희소 대칭화 A + Aᵀ는 줄 i를 A와 Aᵀ의 앞 줄들이 가질 수 있는 가장 많은 원소 뒤에 한 번의 합치기로 쓰고 줄 끝을 따로 쥔다(세는 합치기를 뺐다; 자리는 nnz(A) + nnz(Aᵀ)칸, 초기화하지 않는다): n = 10⁵, d = 10에서 1스레드 19.5 → 13.5 ms(scipy 21.1–22.3), 10스레드 3.41 → 2.52 ms. 밀집의 비정규화는 풀이 없을 때(1스레드) 한 번의 훑기다(들어오는 차수는 열마다 누산기를 두고 행을 차례로, 나가는 차수와 대칭화는 8행씩 더한 뒤 캐시에 있는 동안 부호를 바꿔 쓴다; 같은 합, 같은 순서): 2000²에서 1.70 → 1.41 ms(scipy 1.40–1.45). 풀이 있으면 나눈 두 번의 훑기가 빨라 그대로 둔다(열 묶음의 한 번 훑기는 10스레드 0.52 → 0.89 ms로 느렸다). 들어오는 차수를 병렬 전치의 행 합으로 나누는 꼴도 쟀지만 10스레드에서 2.26 → 2.63 ms로 느려 넣지 않았다(흩뿌림 한 번이 전치보다 싸다) (추천안, 사용자 위임 2026-11-23) |
| GR65 갈래의 문턱 | GR6 마일스톤에서 다시 쟀고 모두 그대로 둔다. auto의 밀집 길(4·nnz ≥ n²; GR26, GR36): 원점마다의 힙 Dijkstra와 밀집 Dijkstra의 비가 밀도 0.22에서 0.91–0.95, 0.39에서 1.21–1.37(n = 500, 1000, 1·10스레드)이라 1/4이 갈림점이다. 병렬 합집합–찾기(GR31, 저장 원소 2¹⁸개): 1스레드와 10스레드(문턱을 낮춘 빌드)의 갈림점이 2¹⁷(0.13 대 0.20 ms)과 2¹⁸(0.39–0.42 대 0.21–0.35 ms) 사이다. 원점 태스크(GR40)의 1–64개는 그대로다. Borůvka는 넣지 않는다(GR43): Kruskal이 저장 원소 10⁶·8 × 10⁶·10⁷개에서 scipy의 6.2·7.4·5.1배(13.6·120·252 ms)이고, 정렬을 병렬 기수 정렬로 바꿔도 10스레드에서 1.06배라(정렬은 시간의 작은 몫이다) 병렬 Borůvka의 몫이 작다 (추천안, 사용자 위임 2026-11-23) |
| GR66 `form='lo'`의 행렬 | GR51을 그대로 둔다: 행렬을 만들어 감싸는 꼴은 행렬 하나(희소: 대각 밖의 저장 원소 + n)만큼 메모리가 더 들지만, 행렬 없는 곱이 `laplacian(G) @ x`와 비트가 같으려면 sparse의 곱 규칙(fma와 4096항 조각)을 이 영역에 옮겨야 하고, 그 이득(입력만 한 메모리)이 영역 사이의 코드 중복보다 작다 (추천안, 사용자 위임 2026-11-23) |
| GR67 메모리 검사와 GIL | `csgraph_to_dense`는 n² × 8바이트를 할당 전에 물리 메모리와 견준다(테스트 하나가 51 GB를 잡은 일; API.md §2의 `MemoryError`). FW의 `room`은 결과와 블록 작업 사본을 함께 센다(GR60). `reconstruct_path`의 검사와 트리 만들기, `yen`의 선행 노드 채우기, `biconnected_components`의 변 배열 채우기, `maximum_flow`의 결과 채우기는 크기가 기준값 이상이면 GIL을 놓고 돈다(Python 객체는 GIL을 쥔 채로 만든다) (추천안, 사용자 위임 2026-11-23) |
| GR68 측정으로 넣지 않은 것 | 여러 원점의 작업 공간을 태스크 사이에 다시 쓰기(100 원점 968 → 991 ms, 차이 없음), 강한 연결 요소의 Pearce 꼴(배열 하나로 Tarjan의 셋을 대신; 5.29 → 5.22 ms, 차이 없음), 무향 순회 전치의 1바이트 값(1.91 → 1.88 ms), 최소 신장 트리의 병렬 기수 정렬(GR65)은 차이가 잡음 안이라 코드를 늘리지 않았다. 순회의 찾은 노드 표시를 비트로, 깊이 우선의 머리 고르기를 가지 없이 바꾼 것은 넣었다(깊이 우선 1.04–1.07배; 무향 깊이 우선은 1스레드 6.62–6.95 ms로 scipy 6.53–6.86와 같은 수준이고, 그 시간의 대부분은 노드마다 두 줄을 읽는 메모리 지연이다) (추천안, 사용자 위임 2026-11-23) |

### 1.2 범위
마일스톤은 §8이다. "GRn"이 표의 결정 번호와 마일스톤 이름에 함께 쓰이므로, 이 문서는 마일스톤을 언제나 "GR2 마일스톤"처럼 적는다. "나중"은 첫 판에 넣지 않은 후보다.

**이름공간**: 모든 함수가 `sd.graph`에 바로 있다. scipy의 `from scipy.sparse.csgraph import dijkstra`는 `from stardust.graph import dijkstra`가 된다. sparse가 `sd.sparse.linalg`를, spatial이 `sd.spatial.distance`를 두지 않은 것과 같다. 이름이 겹치지 않는다.

**scipy.sparse.csgraph → sd.graph** (scipy 1.18.1의 `__all__` 스물여섯 전부)
| scipy | sd.graph | 마일스톤 |
|---|---|---|
| `connected_components` | 같은 이름. 레이블은 가장 작은 노드의 순서(§2.2) | GR1 |
| `breadth_first_order`, `depth_first_order`, `breadth_first_tree`, `depth_first_tree` | 같은 이름. 이웃은 색인 오름차순(§2.2) | GR1 |
| `csgraph_from_dense`, `csgraph_to_dense` | 같은 이름(§2.7) | GR1 |
| `csgraph_masked_from_dense`, `csgraph_from_masked`, `csgraph_to_masked` | 없음: sd에는 가린 배열(numpy의 `MaskedArray`)이 없다. 0 무게의 변은 희소 배열의 저장된 0으로 준다 | — |
| `shortest_path`, `dijkstra`, `bellman_ford`, `johnson`, `floyd_warshall` | 같은 이름. 어느 방법이든 같은 결과(§3.2) | GR2 |
| `yen` | 같은 이름 | GR2 |
| `construct_dist_matrix`, `reconstruct_path` | 같은 이름. 거리는 `shortest_path`와 같은 순서의 합(§2.3) | GR2 |
| `NegativeCycleError` | 같은 이름, `ValueError`의 하위 클래스(§2.3) | GR2 |
| `minimum_spanning_tree` | 같은 이름 | GR3 |
| `maximum_flow` | 같은 이름, `method='dinic'`과 `'edmonds_karp'` | GR3 |
| `maximum_bipartite_matching`, `min_weight_full_bipartite_matching` | 같은 이름 | GR3 |
| `reverse_cuthill_mckee`, `structural_rank` | 같은 이름 | GR4 |
| `laplacian` | 같은 이름(`form='function'`과 `copy`는 없다, §2.5) | GR4 |

**scipy.optimize → sd.graph** (§9의 1번)
| scipy | sd | 마일스톤 |
|---|---|---|
| `linear_sum_assignment(cost_matrix, maximize=False)` | `sd.graph.linear_sum_assignment` | GR3 |

**scipy에 없는 것 → sd.graph** (뜻은 networkx의 같은 이름 함수; §2.6)
| networkx | sd.graph | 마일스톤 |
|---|---|---|
| `topological_sort`, `lexicographical_topological_sort` | `topological_sort`: 사전 순서로 가장 작은 위상 순서 하나 | GR1 |
| `articulation_points`, `bridges`, `biconnected_components` | 같은 이름 | GR5 |
| `core_number`, `k_core` | `core_number`(k-코어는 `core_number(G) >= k`인 노드다; §9의 15번) | GR5 |
| `pagerank` | 같은 이름 | GR5 |
| `betweenness_centrality` | 같은 이름 | GR5 |
| 군집 찾기(`louvain_communities`, `label_propagation_communities`), 무작위 그래프, 다른 중심성(근접, 고유벡터, Katz, HITS) | 없음, 나중(GR1) | 나중 |

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다. 별칭이 없다.
- **같은 기계, 같은 스레드 수, 같은 입력이면 실행마다 같은 비트다**(API.md §3). 스레드 수와 무관한 비트는 2026-11-22부터 약속이 아니지만 이 영역은 지킨다: 결과를 알고리즘이 아니라 **정의**로 정하고(최단 거리는 고정 순서의 합의 최솟값, 동점은 정한 규칙, 연결 요소의 번호는 가장 작은 노드 순서; §3), 나누기는 크기로만 정한다(§3.8). **결과(값, 선행 노드, 순서, 레이블)는 스레드 수뿐 아니라 `method`(D, BF, J, auto)와도 무관하다(FW만 블록 Floyd–Warshall의 반올림이다: GR25).** 갈래가 결과를 바꾸면 반올림의 차이가 아니라 다른 프로그램이 되기 때문이다(spatial.md §1.3과 같은 원칙).
- 결과를 정의로 정하면 알고리즘의 자유가 생긴다: 집합이나 분할이 답인 문제(연결 요소, 단절점, 다리, 최소 신장 트리)는 어떤 순서로 찾든 답이 하나라서 병렬 알고리즘을 쓸 수 있다.
- 반올림이 아닌 동작(결과의 모양과 형, 기본값, 경계의 포함 여부, 표식 −9999·−1, 에러를 내는 경우)은 scipy 1.18.1과 같다(API.md §4의 규칙 2). 다르게 정한 곳은 `[scipy와 다름]`으로 이유와 함께 적는다. scipy는 동점의 선택을 약속하지 않는다(문서마다 "If multiple valid solutions are possible, output may vary with SciPy and Python version"이라고 적는다). sd가 정한 동점 규칙은 그 빈자리를 채우는 것이다.
- 경고를 내지 않는다. scipy가 경고하거나 조용히 틀린 값을 내는 입력(음수 무게의 `dijkstra`, 복소 입력, 범위 밖 시작 노드, int64 용량, 음수 용량, 저장된 NaN 무게)은 에러다(§2.9).
- 에러 종류와 메시지는 API.md §2·§2.1이다. 메시지는 부른 함수의 이름으로 시작한다(`dijkstra(): ...`).
- 참값은 정확한 유리수(Python `fractions.Fraction`; float64는 유리수로 정확히 바뀐다)와 Arb(python-flint 0.9.0)만 쓴다. mpmath는 쓰지 않는다.
- 모든 호출은 정해진 시간 안에 끝난다. 반복이 있는 것(PageRank)도 반복 수의 상한이 입력으로 정해진다(§3.7).

### 1.4 다른 영역과의 관계
- **sparse**: 입력의 주된 꼴이고 결과 행렬의 꼴이다. sparse의 CSR은 언제나 표준 꼴(줄 안 색인 오름차순, 중복 없음)이고 색인은 int64다(sparse.md §2.2–§2.3). 그래서 그래프의 이웃 목록은 언제나 색인 오름차순이고 중복 변이 없다. scipy의 csgraph가 비표준 CSR의 중복 원소를 "가장 작은 무게"로 읽는 규칙(`csgraph_to_dense`의 문서)은 sd에서 생기지 않는다: COO의 중복은 CSR로 바꿀 때 더해진다(scipy의 `tocsr()`도 더한다). C++의 `graph/`는 `sparse/format.hpp`(표준 꼴의 세 배열을 읽는 것)만 쓰고, sparse는 graph를 모른다.
- **API.md §7**(희소에만 있을 이유가 없는 기능은 밀집에도 있다): 이 영역의 모든 함수는 밀집 인접 행렬도 받는다(GR2). 그래서 희소 전용 기능이 생기지 않는다.
- **spatial**: `KDTree.sparse_distance_matrix`, `radius_neighbors(...).tocsr()`가 만든 CSR이 바로 입력이 된다(거리 0인 짝은 저장된 0, 곧 무게 0의 변이다; spatial.md §2.5). 그래서 "점 → 반경 그래프 → 연결 요소·최소 신장 트리"가 변환 없이 이어진다.
- **linalg·sparse의 고유값**: `laplacian`의 결과(CSR이나 `LinearOperator`)는 `sd.sparse.eigsh`의 입력이다. `laplacian(form='lo')`는 `sd.sparse.LinearOperator`를 만든다.
- **optimize**: `linear_sum_assignment`의 자리에 따라(§9의 1번) 다른 영역을 고치지 않는다(이 영역의 이름으로 둔다).
- **random**: 첫 판에는 무작위를 쓰는 함수가 없다(무작위 그래프와 표본 매개 중심성은 나중; §9의 17번).
- **나중의 영역**: 군집(cluster)의 스펙트럼 군집과 연결성 기반 군집, 임베딩(embedding)의 Isomap(최단 경로)과 스펙트럼 임베딩(`laplacian`)이 이 영역을 쓸 수 있다. 약속이 아니라 메모다.

---

## 2. API

### 2.1 입력
- **그래프**: 정방 (n, n) 인접 행렬이다. 행 i, 열 j의 원소가 i → j 변의 무게다. 받는 꼴(§9의 4번):
  - `sd.sparse`의 `csr_array`, `csc_array`, `coo_array`, `dok_array`. CSR이 아니면 한 번 CSR로 바꾼다(COO의 중복은 더해진다). 변은 **저장된 원소**이고 저장된 0도 무게 0의 변이다(scipy와 같다).
  - 밀집: `sd.ndarray`, numpy 배열, 수의 중첩 시퀀스(2차원). 변은 **0이 아니고 NaN도 ±inf도 아닌 원소**다(scipy와 같은 규칙: 밀집에서는 0, NaN, ±inf가 변 없음이다). 무게 0의 변은 밀집으로 줄 수 없다: 희소의 저장된 0이나 `csgraph_from_dense(…, null_value=inf)`로 만든다.
  - scipy의 희소 배열은 받지 않는다(`TypeError`, 메시지가 `sd.sparse.csr_array(A)`를 짚는다; §9의 4번). frame의 `DataFrame`도 받지 않는다(인접 행렬은 표가 아니다).
  - dtype: bool, int64, float64. 무게를 쓰는 계산은 float64로 한다(bool·int64는 float64로 바꾼다. scipy도 바꾼다). complex128은 `TypeError`다(`[scipy와 다름]`: scipy는 허수부를 버리고 `ComplexWarning`을 낸다).
  - 모양: 2차원이 아니면 `ValueError`, 정방이 아니면 `ValueError`(scipy와 같다). 매칭 셋(`maximum_bipartite_matching`, `min_weight_full_bipartite_matching`, `linear_sum_assignment`)과 `structural_rank`는 직사각을 받는다(이분 그래프의 행 쪽·열 쪽; scipy와 같다). n = 0도 받는다(빈 결과; scipy와 같다).
- **무게의 값** (§9의 5번)
  - 무게를 쓰는 함수(최단 경로, `minimum_spanning_tree`, `min_weight_full_bipartite_matching`, `linear_sum_assignment`, `maximum_flow`, `pagerank`, 가중 `betweenness_centrality`)는 희소 입력의 저장된 NaN과 ±inf를 `ValueError`로 막는다(`dijkstra(): stored edge weights must be finite`). `[scipy와 다름]`: scipy는 최단 경로에서 저장된 NaN 변을 비교 실패로 조용히 건너뛰고, 같은 NaN을 `minimum_spanning_tree`는 트리에 넣으며, −inf 변의 `bellman_ford`는 음수 순환을 알리지 않고 −inf 거리를 낸다(설계 때 확인; `tests/graph/data/scipy.json`). 밀집 입력은 위의 규칙대로 NaN·±inf가 변 없음이라 막을 것이 없다.
  - 구조만 보는 함수(연결 요소, 순회, 위상 정렬, `reverse_cuthill_mckee`, `structural_rank`, `maximum_bipartite_matching`, 단절점, 다리, 이중 연결 요소, `core_number`, 무게 없는 `betweenness_centrality`)는 **패턴**(희소의 저장된 자리, 밀집의 변 규칙)만 읽는다. 값이 NaN이어도 변이다.
  - 예외는 `laplacian`이다: 그래프 함수가 아니라 행렬 함수라서 밀집 입력도 값 그대로(0, NaN, inf가 그 값으로) 계산한다(scipy와 같다; §2.5).
- **무향 그래프** (`directed=False`): i와 j 사이의 변은 i → j와 j → i 가운데 있는 것이고, 둘 다 있으면 작은 무게다(scipy와 같다; 설계 때 D, BF, J, FW가 모두 그렇게 계산함을 확인했다). NaN이 끼면 NaN, 0.0과 −0.0이면 −0.0이다(GR29). 패턴만 보는 함수는 두 방향 패턴의 합집합이다.
- **노드 인자**(`i_start`, `source`, `sink`, `indices`의 원소): 0 이상 n 미만의 int다(`operator.index`의 규칙, API.md §2). 범위 밖이면 `ValueError`, bool·실수는 `TypeError`다(§9의 20번; `[scipy와 다름]`: scipy의 `dijkstra(indices=1.5)`는 1로 자르고 `indices=-1`은 n − 1로 감고, `breadth_first_order(G, 5)`는 범위 밖에서 내부 `IndexError`를 삼키고 빈 결과를 내며(설계 때 확인), `i_start=-1`은 `OverflowError`다).
- **자기 고리**(대각 원소): 최단 경로와 순회에서는 쓰이지 않는다(양수 무게의 고리는 경로를 짧게 하지 않는다). 음수 무게의 고리는 음수 순환이다(scipy와 같다). `minimum_spanning_tree`, 무향 구조 함수(단절점, 다리, 이중 연결 요소, `core_number`)는 대각을 무시한다. `topological_sort`에서는 순환이다. `laplacian`은 대각을 무시한다(scipy와 같다). `pagerank`는 변으로 센다(networkx와 같다; GR28).

### 2.2 순회와 연결 요소
```
connected_components(csgraph, /, *, directed=True, connection='weak', return_labels=True)
breadth_first_order(csgraph, i_start, /, *, directed=True, return_predecessors=True)
depth_first_order(csgraph, i_start, /, *, directed=True, return_predecessors=True)
breadth_first_tree(csgraph, i_start, /, *, directed=True)
depth_first_tree(csgraph, i_start, /, *, directed=True)
topological_sort(csgraph, /)
```
- **옵션은 키워드만이다**(§9의 19번; `[scipy와 다름]`: scipy는 `connected_components(G, False)`처럼 위치로도 받는다). `connection`은 `'weak'` 또는 `'strong'`이고 `directed=False`면 읽지 않는다(scipy와 같다).
- **레이블**: 요소 번호는 **요소의 가장 작은 노드의 순서**다(노드 0의 요소가 0, 그다음 처음 나오는 노드의 요소가 1, …; §9의 9번). 약한 연결과 무향은 scipy와 같고(설계 때 확인), 강한 연결은 `[scipy와 다름]`: scipy의 번호는 알고리즘(Pearce)의 순서라 `[0, 1, 3, 4, 2, 5]`처럼 나온다. 분할은 같다. 결과는 `ConnectedComponentsResult(n_components, labels)`(`n_components`는 Python `int`, `labels`는 int64; scipy는 int32)이고 `return_labels=False`면 Python `int` 하나다(scipy와 같은 꼴).
- **순회의 순서**: 이웃은 **노드 번호 오름차순**으로 본다. 유향이면 행의 저장 순서(CSR 표준 꼴)이고, 무향이면 두 방향 이웃을 합친 오름차순이다(§9의 10번). 너비 우선은 큐, 깊이 우선은 재귀 깊이 우선과 같은 순서(가장 작은 미방문 이웃부터)다. 유향 그래프에서는 scipy와 같은 순서다(5 × 5 격자에서 확인). `[scipy와 다름]`: scipy의 무향 순회는 나가는 이웃을 먼저, 들어오는 이웃을 나중에 본다(설계 때 본 예: `[0, 3, 4, 1, 2]` 대 sd `[0, 1, 2, 3, 4]`). sd는 G와 Gᵀ(같은 무향 그래프)에서 같은 답을 낸다.
- **결과**: `TraversalResult(node_array, predecessors)`(둘 다 int64; scipy는 int32), `return_predecessors=False`면 `node_array`만이다. 선행 노드는 순회 트리의 부모이고 시작 노드와 닿지 않은 노드는 −9999다(scipy와 같다). 트리 함수는 (n, n) `csr_array` float64이고 변은 부모 → 자식 방향에 원래 무게(무향이면 있는 쪽의 무게; 둘 다 있으면 작은 쪽)다(scipy와 같다). 무게 0의 변은 저장된 0으로 남는다(유향은 scipy와 같고, 무향은 `[scipy와 다름]`: GR29).
- **`topological_sort`**(networkx의 뜻): 유향 그래프의 노드를 모든 변 i → j에서 i가 j보다 앞서게 늘어놓은 int64 (n,) 배열이다. **사전 순서로 가장 작은 위상 순서**를 돌려준다(진입 차수 0인 노드 가운데 가장 작은 번호를 먼저; networkx의 `lexicographical_topological_sort`와 같은 답). 순환(자기 고리 포함)이 있으면 `ValueError`(`topological_sort(): the graph has a cycle`)다. 이유: 위상 순서는 보통 여럿이고, 사전 최소는 데이터로만 정해지는 하나의 답이며 힙 하나의 비용(O(E + n log n))이다.

### 2.3 최단 경로
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
- **결과는 정의로 정해진다**(§3.2; §9의 2번, 3번): 원점 s에서 v까지의 거리 d̂(v)는 s에서 v로 가는 모든 경로 가운데, 경로의 무게를 **원점에서부터 차례로**(왼쪽에서 오른쪽으로, 0에서 시작해 한 변씩) 더한 float64 값이 가장 작은 것이다. 선행 노드는 §3.2의 규칙(꼭 맞는 변, 가장 적은 변 수, 가장 작은 번호)으로 정한다. **D, BF, J, auto와 `dijkstra`·`bellman_ford`·`johnson`은 같은 입력에서 거리와 선행 노드가 비트까지 같다.** 방법은 빠르기만 바꾼다. FW와 `floyd_warshall`만 블록 Floyd–Warshall의 반올림이다(GR25; 거리는 계약 안, 선행 노드는 GR27). `[scipy와 다름]`: scipy의 D·BF·J는 거리가 같지만(차례 합; 설계 때 무게 10⁻⁸–10⁸의 무작위 그래프로 확인) 선행 노드의 동점이 다르고(마름모 그래프에서 D·J는 2, BF·FW는 1), FW는 거리의 끝자리도 다르다(같은 그래프에서 상대 8.7×10⁻¹⁶).
- **`method`**: `'auto'`, `'D'`, `'BF'`, `'J'`, `'FW'`(scipy와 같은 이름; 그 밖은 `ValueError`, 타입이 틀리면 `TypeError`). auto는 음수 무게가 있으면 J, 없으면 D다(GR26). 음수 무게가 없고 `indices`가 없고 변(입력의 저장된 원소)이 n²/4개 이상이면 D와 같은 비트의 밀집 Dijkstra로 한다(GR36). `[scipy와 다름]`: scipy의 auto는 그 밀도에서 FW다(GR0 마일스톤의 기록: 밀도 0.26부터 FW의 비트). 문턱은 빠르기만 정한다(GR6 마일스톤에서 다시 재어 그대로 두었다: GR65). `method='FW'`와 `indices`를 함께 주면 `ValueError`다(scipy와 같다).
- **결과의 꼴**(scipy와 같다): `indices=None`이면 거리 (n, n) float64, int 하나면 (n,), 1차원이면 (len, n)이다. 닿지 않는 노드는 inf, 선행 노드는 −9999다. `return_predecessors=True`면 `ShortestPathResult(dist_matrix, predecessors)`(선행 노드 int64; scipy는 int32)이고 아니면 거리 배열 하나다. `indices`는 0차원 또는 1차원이고 중복을 받는다(scipy와 같다). 2차원은 `ValueError`다(`[scipy와 다름]`: scipy는 받아 3차원 결과를 낸다).
- **`unweighted=True`**: 모든 변의 무게를 1로 본다(저장된 0의 변도 변 하나다; scipy와 같다). 거리는 변의 수이고 언제나 정확하다. 값은 읽지 않는다: 저장된 NaN·±inf와 음수 무게도 막지 않는다(GR34).
- **`limit`**(`dijkstra`만): 거리 d̂ ≤ `limit`인 노드만 찾고 나머지는 inf, −9999다(경계를 넣는다: 설계 때 scipy의 `limit=1`이 거리 1을 넣음을 확인). 0 이상이거나 inf여야 한다. 음수는 `ValueError`(scipy와 같다), NaN도 `ValueError`(`[scipy와 다름]`: scipy는 NaN에서 모든 노드를 inf로 낸다).
- **`min_only=True`**(`dijkstra`만): 모든 `indices`를 함께 원점으로 둔 거리 하나(`(n,)`)다. `return_predecessors=True`면 `MinOnlyResult(dist_matrix, predecessors, sources)`이고 `sources[v]`는 v의 선행 노드 사슬이 끝나는 원점, 닿지 않으면 −9999다(scipy와 같은 꼴; 동점 규칙은 §3.2). `indices=None`이면 모든 노드가 원점이다(scipy와 같다: 거리가 모두 0).
- **음수 무게**: `dijkstra`(와 `method='D'`)는 저장된 음수 무게를 `ValueError`로 막는다(§9의 6번; `[scipy와 다름]`: scipy는 `UserWarning`을 내고 값을 낸다). `bellman_ford`, `johnson`, `floyd_warshall`, auto(J를 고른다)는 받는다. 무향 그래프의 음수 변은 그 자체로 음수 순환이다(scipy와 같다).
- **`NegativeCycleError`**: `sd.graph.NegativeCycleError(ValueError)`다(API.md §2의 영역 고유 예외 규칙; `[scipy와 다름]`: scipy의 것은 `Exception`의 하위 클래스다). **요청한 원점에서 닿을 수 있는 음수 순환이 있으면** 낸다(모든 방법이 같다; `indices=None`이면 그래프 어디든). `[scipy와 다름]`: scipy의 `bellman_ford(indices=0)`은 닿지 않는 순환을 무시하고 `johnson(indices=0)`은 그 순환에서 에러를 낸다(설계 때 확인). 메시지는 순환 위의 노드 하나를 적는다(`bellman_ford(): negative cycle through node 3`; 그 노드는 모든 방법이 같이 쓰는 검사가 찾은 순환 위의 가장 작은 번호다: GR35).
- **`yen(csgraph, source, sink, K)`**: 고리 없는 s–t 경로 가운데 (차례 합의 길이, 변의 수, 노드 열의 사전 순서)로 앞선 K개다(§3.2). 결과는 `dist_array`(길이 M ≤ K) 하나 또는 `YenResult(dist_array, predecessors)`(M × n, int64)다(scipy와 같은 꼴). 음수 무게는 받고(scipy처럼 J의 퍼텐셜을 쓴다) 음수 순환은 `NegativeCycleError`다. `source == sink`면 길이 0의 경로 하나다(scipy와 같다). `K`는 0 이상의 int다(0이면 빈 결과; scipy와 같다).
- **`construct_dist_matrix`**: 선행 노드 행렬이 가리키는 경로를 따라 무게를 원점에서부터 차례로 더한 거리다. 그래서 **FW가 아닌 방법에서 `construct_dist_matrix(G, shortest_path(G, return_predecessors=True).predecessors)`는 `shortest_path`의 거리와 비트까지 같다**(FW는 GR27). `[scipy와 다름]`: scipy는 도착점에서 거꾸로 더해 끝자리가 다르다(설계 때 무작위 그래프로 확인: scipy의 `construct_dist_matrix`는 거꾸로 합과 같고 자기 `shortest_path`와 다르다). 닿지 않는 자리는 `null_value`다(scipy와 같다). 선행 노드는 정수 dtype의 (n, n)이고 그래프의 경로를 그려야 한다: 노드도 −9999도 아닌 값, 없는 변, 순환, 원점이 아닌 곳에서 끝나는 사슬은 `ValueError`다(GR38; `[scipy와 다름]`: scipy는 int32만 받고 없는 변에 inf를 더하며 순환에서 끝나지 않는다).
- **`reconstruct_path(csgraph, predecessors)`**: 선행 노드 1차원 배열이 그리는 트리를 (n, n) `csr_array`로 만든다(변은 부모 → 자식에 그 변의 무게, 무향이면 두 방향의 작은 무게, 무게 0의 변은 저장된 0; scipy와 같다). 없는 변과 자기를 가리키는 선행 노드는 `ValueError`다(GR38; `[scipy와 다름]`: scipy는 없는 변을 말없이 0으로 넣거나 버린다).
- `overwrite`는 없다(§9의 11번): 입력을 고치지 않는다(`[scipy와 다름]`).

### 2.4 신장 트리, 흐름, 매칭
```
minimum_spanning_tree(csgraph, /)
maximum_flow(csgraph, source, sink, /, *, method='dinic')
maximum_bipartite_matching(graph, /, *, perm_type='row')
min_weight_full_bipartite_matching(biadjacency, /, *, maximize=False)
linear_sum_assignment(cost_matrix, /, *, maximize=False)
```
- **`minimum_spanning_tree`**: 무향으로 본 그래프(§2.1의 무향 규칙)의 최소 신장 숲이다(연결되지 않으면 요소마다의 트리; scipy와 같다). **답은 하나다**: 변을 (무게, min(i, j), max(i, j))의 전순서로 견주면 최소 신장 숲이 유일하다(§3.4; §9의 14번). 결과는 (n, n) `csr_array` float64이고 고른 변은 원래 저장된 방향의 자리에 있다(두 방향이 다르면 작은 무게의 자리, 같으면 i < j인 위 삼각; scipy도 원래 자리에 둔다; GR43). 계산은 기수 정렬과 Kruskal이다(GR43). 음수 무게는 받는다(scipy와 같다). `[scipy와 다름]`: scipy는 밀집 입력에서 1e-8보다 작은 원소를 0(변 없음)으로 버린다(문서에 적혀 있고 설계 때 확인). sd는 밀집의 변 규칙(§2.1) 그대로다. 저장된 NaN·±inf는 `ValueError`(scipy는 NaN·inf 변을 트리에 넣는다).
- **`maximum_flow`**: 결과는 `MaximumFlowResult(flow_value, flow)`다(scipy의 이름과 필드). `flow_value`는 Python `int`, `flow`는 (n, n) `csr_array` int64이고 반대 방향에 음수 흐름이 있는 반대칭 꼴이다(scipy와 같다; scipy는 int32). 패턴은 G와 Gᵀ의 패턴이다: 저장된 용량 0의 변과 대각의 저장 원소도 흐름 0으로 있다(scipy와 같다; GR44). 흐름 값은 유일하지만 흐름 자체는 여럿일 수 있고, 돌려주는 흐름은 고정된 훑기 순서의 결정적인 결과다(§9의 13번).
  - 용량(§9의 12번): bool·int64 dtype만 받는다(float64는 정수 값이어도 `TypeError`; `[scipy와 다름]`: scipy는 `ValueError`다). 음수 용량은 `ValueError`(`[scipy와 다름]`: scipy는 조용히 0으로 본다). 원점에서 나가는 용량의 합(원점의 자기 고리는 뺀다)이 int64를 넘으면 `OverflowError`(흐름 값이 그 합 이하라서 넘침이 미리 막힌다; 잔여 용량은 uint64로 쥔다, GR44). 메시지는 `maximum_flow(): the capacities must be bool or int64, not float64`, `the capacities must be nonnegative`, `the capacities out of the source sum past the int64 range`다. `[scipy와 다름]`: scipy는 용량을 int32로 줄여 2⁴⁰ 용량의 흐름이 0, 2³² + 5가 5가 된다(설계 때 확인).
  - 입력은 희소와 밀집 모두다(`[scipy와 다름]`: scipy는 CSR만 받고 밀집은 `TypeError`). `source == sink`는 `ValueError`(scipy와 같다). 자기 고리는 흐름에 쓰이지 않는다. 용량이 int32 안이면 돌려주는 흐름이 scipy의 흐름과 배열까지 같다(두 방법 모두; GR44).
- **`maximum_bipartite_matching`**: 행 쪽과 열 쪽의 이분 그래프(직사각 가능)에서 가장 큰 매칭이다. `perm_type='row'`면 길이(열 수)의 int64 배열(열 j에 짝지은 행, 없으면 −1), `'column'`이면 행마다의 열이다(scipy와 같다). 저장된 0도 변이다(scipy와 같다). 매칭의 크기는 유일하고 매칭 자체는 결정적인 알고리즘의 결과다(§9의 13번; GR47). scipy와 크기는 같고 매칭은 다를 수 있다.
- **`min_weight_full_bipartite_matching`**: 크기가 min(행 수, 열 수)인 매칭 가운데 무게 합이 가장 작은(`maximize=True`면 가장 큰) 것이다. 결과는 `AssignmentResult(row_ind, col_ind)`(둘 다 int64; scipy는 `col_ind`가 int32)이고 `row_ind`는 오름차순이다(scipy와 같다). 그런 매칭이 없으면 `ValueError`(scipy와 같다). `[scipy와 다름]`: scipy는 무게 0을 막는다고 문서에 적고 실제로는 경고(`explicit zero weights are removed before matching`)와 함께 저장된 0을 변 없음으로 읽는다(설계 때와 GR3 마일스톤에서 확인). sd의 저장된 0은 무게 0의 변이다(§2.1). 저장된 NaN·±inf는 `ValueError`(scipy는 조용히 답을 낸다). 메시지는 `min_weight_full_bipartite_matching(): no full matching exists`(scipy와 같은 말)다.
- **`linear_sum_assignment`**(자리는 §9의 1번): scipy의 뜻 그대로다. `cost_matrix`의 모든 원소가 변이고(0도 비용 0의 변; 인접 행렬의 변 규칙이 아니다), +inf는 고를 수 없는 짝이다. 모든 짝을 고를 수 없으면 `ValueError`(`cost matrix is infeasible`), NaN과 −inf는 `ValueError`(scipy와 같다). `maximize=True`면 부호를 먼저 바꾸므로 −inf가 고를 수 없는 짝이고 +inf가 `ValueError`다(scipy와 같다; GR46). 직사각을 받고 결과는 `AssignmentResult(row_ind, col_ind)`(int64, `row_ind` 오름차순)다. 입력은 밀집만이다(희소 비용은 `min_weight_full_bipartite_matching`; 희소 배열은 그것을 짚는 `TypeError`). 비용 dtype은 bool, int64, float64다(float64로 바꾼다). 동점에서 고르는 최적 배정은 scipy와 다를 수 있다(GR45; scipy는 시작 단계가 없는 Crouse의 꼴).

### 2.5 순서, 구조적 랭크, 라플라시안
```
reverse_cuthill_mckee(graph, /, *, symmetric_mode=False)
structural_rank(graph, /)
laplacian(csgraph, /, *, normed=False, return_diag=False, use_out_degree=False, form='array', dtype=None, symmetrized=False)
```
- **`reverse_cuthill_mckee`**: 정방 그래프의 띠 폭을 줄이는 순열(int64 (n,); scipy 1.18.1도 int64다: GR0 마일스톤의 기록)이다. `symmetric_mode=False`면 A와 Aᵀ의 패턴의 합집합으로 한다(`[scipy와 다름]`: scipy는 값의 합 A + Aᵀ라 0이 된 자리가 빠진다; GR52). 동점 규칙(§3.6, GR52): 차수는 줄 길이 + 저장된 대각이면 1, 요소마다 시작 노드는 방문하지 않은 노드 가운데 (차수, 번호)로 첫 노드, 이웃은 (차수, 번호) 순서로 큐에 넣고, 요소는 시작 노드의 순서로 이어 붙인 뒤 전체를 뒤집는다. 이 순서는 scipy의 알고리즘 그대로이고, scipy의 시작 노드의 동점만 numpy의 불안정 정렬이라 17노드부터 다를 수 있다(`[scipy와 다름]`; 16노드 이하는 같다). 밀집 입력도 받는다(`[scipy와 다름]`: scipy는 CSR·CSC만).
- **`structural_rank`**: 패턴의 가장 큰 이분 매칭의 크기(Python `int`; scipy는 `numpy.int64`)다. 직사각을 받는다(scipy와 같다). 저장된 0도 원소다(scipy와 같다).
- **`laplacian`**: L = D − A(D는 들어오는 무게의 합, `use_out_degree=True`면 나가는 무게의 합; scipy와 같다). 대각의 원소는 무시한다. `normed=True`면 D^(−1/2) L D^(−1/2)이고 차수 0인 노드는 1로 나누며, 대각은 차수가 0이 아니면 1, 0이면 0이다(scipy와 같다; GR50). `symmetrized=True`면 A + Aᵀ로 먼저 대칭으로 만든다(sd.sparse의 덧셈 규칙: 정확히 0인 합은 저장하지 않는다; scipy와 같다; GR53). 값을 그대로 쓰므로 NaN·inf는 IEEE로 퍼진다(행렬 함수다, §2.1). 음수 무게의 `normed`는 √ 음수에서 NaN이다(scipy와 같다).
  - 결과: 밀집 입력이면 `sd.ndarray`, 희소 입력이면 `csr_array`(`[scipy와 다름]`: scipy는 `coo_array`. sd의 계산 형식이고 `eigsh`가 바로 쓴다; 패턴은 대각 밖 원소와 대각 전부이고 저장된 0은 −0.0이다, GR53), `form='lo'`면 그 행렬을 감싼 `sd.sparse.LinearOperator`(곱이 `laplacian(G) @ x`와 같은 비트; `[scipy와 다름]`: scipy는 행렬을 만들지 않는다; GR51). `form='function'`과 `copy`는 없다(§9의 11번). `return_diag=True`면 `LaplacianResult(laplacian, diag)`다(밀집 행렬과 `diag`는 쓰기 금지).
  - dtype: `dtype=None`이면 입력의 dtype(bool은 int64로 올린다), `normed=True`면 float64다(scipy와 같다). `dtype`은 None, int64, float64이고 계산한 뒤에 바꾼다(GR54).

### 2.6 더한 알고리즘 (scipy에 없음; 뜻은 networkx의 같은 이름)
```
articulation_points(csgraph, /)
bridges(csgraph, /)
biconnected_components(csgraph, /)
core_number(csgraph, /)
pagerank(csgraph, /, *, alpha=0.85, personalization=None, dangling=None, directed=True, unweighted=False, tol=1e-12)
betweenness_centrality(csgraph, /, *, directed=True, unweighted=False, normalized=True, endpoints=False)
```
- **무향의 넷**(`articulation_points`, `bridges`, `biconnected_components`, `core_number`): 그래프를 언제나 무향으로 본다(두 방향 패턴의 합집합, 대각은 무시). 답이 집합이므로 정렬된 꼴로 돌려준다.
  - `articulation_points`: 지우면 연결 요소가 늘어나는 노드의 오름차순 int64 배열이다.
  - `bridges`: 지우면 연결 요소가 늘어나는 변의 (k, 2) int64 배열이고 행은 (i, j), i < j, 사전 순서다.
  - `biconnected_components`: 변을 이중 연결 요소로 나눈 것이다. 결과는 `BiconnectedComponentsResult(n_components, edges, labels)`: `edges`는 `bridges`와 같은 꼴의 모든 변(사전 순서), `labels`는 변마다의 요소 번호이고 번호는 요소의 사전 순서로 가장 작은 변의 순서다(§9의 15번). networkx는 노드 집합을 돌려준다: 노드 집합은 `edges[labels == c]`의 노드다.
  - `core_number`: 노드마다의 코어 수(int64 (n,))다. k-코어는 `core_number(G) >= k`인 노드다(§9의 15번). 답이 유일하다.
- **`pagerank`**: 확률 벡터 x(float64 (n,); 합은 1과 반올림만큼 다르다)이고 x = α·Pᵀx + (1 − α)·v의 해다. P는 나가는 무게로 정규화한 전이 행렬이고(`unweighted=True`면 변마다 1), 나가는 무게가 0인 노드(매달린 노드)의 행은 `dangling` 분포로 바꾼다. 기본 `personalization`은 균등, 기본 `dangling`은 `personalization`이다(networkx의 뜻). `directed=False`면 §2.1의 무향 규칙으로 만든 대칭 그래프다. `alpha`는 0 이상 1 미만(1은 해가 하나가 아닐 수 있어 `ValueError`), `personalization`·`dangling`은 길이 n의 음이 아닌 유한 배열이고 합이 0보다 커야 한다(합 1로 정규화한다). 음수 무게는 `ValueError`다. **결과는 연립방정식의 해 x\*와 ℓ₁로 `tol` + F 안이다**(F는 한 걸음의 반올림을 1 − α로 나눈 것; 증명된 한계로 멈춘다; §3.7, §9의 16번, GR56). `personalization`·`dangling`은 bool·int64·float64 배열이나 수의 시퀀스다(networkx의 사전은 받지 않는다). 반복 수의 상한은 α와 `tol`로 정해지므로 `max_iter`가 없다(`[networkx와 다름]`: networkx는 `tol=1e-6`, `max_iter=100`, 연속한 반복의 차로 멈추고 수렴하지 못하면 예외다).
- **`betweenness_centrality`**: Brandes의 정의다. 노드 v의 값은 v가 아닌 모든 (s, t)에서 s–t 최단 경로 가운데 v를 지나는 몫의 합이다. 무향이면 (s, t)와 (t, s)를 한 번 센다(합을 2로 나눈다), `normalized=True`면 순서쌍의 수 (n − 1)(n − 2)(`endpoints=True`면 n(n − 1))로 나누고(무향은 쌍을 한 번 센 값에 2/((n − 1)(n − 2))를 곱한 셈이다), `endpoints=True`면 끝점도 센다(networkx 3.7의 `_rescale`: 쌍의 수 N이 2보다 작으면 그대로; GR58). 최단 경로와 그 개수는 §3.2의 거리와 §3.7의 DAG(꼭 맞는 변, 같은 거리는 너비 우선 변 수로 순서; GR58)로 정한다. float64 (n,)이다. 무게는 0보다 커야 하고(무게 0의 변이 있으면 최단 경로가 무한히 많을 수 있다; `ValueError`), `unweighted=True`면 패턴만 본다. 표본으로 어림하는 `k`, `seed`는 없다(나중; §9의 17번).

### 2.7 변환 도구
```
csgraph_from_dense(graph, /, *, null_value=0, nan_null=True, infinity_null=True)
csgraph_to_dense(csgraph, /, *, null_value=0)
```
- `csgraph_from_dense`: 밀집 행렬(float64로 읽는다)의 원소 가운데 `null_value`와 같은 것(`nan_null`이면 NaN도, `infinity_null`이면 ±inf도)을 뺀 나머지를 저장한 `csr_array`(float64)다. `null_value=inf`로 0 무게의 변을 만든다(scipy와 같다). `null_value=None`이면 같음으로 빼는 값이 없고(NaN·±inf는 두 플래그대로), NaN인 `null_value`는 `nan_null=True`, 무한대인 `null_value`는 `infinity_null=True`로 읽는다(scipy와 같다; GR1 마일스톤에서 확인). 정방이 아니면 `ValueError`, 희소 입력은 `TypeError`다.
- `csgraph_to_dense`: 저장된 원소를 그 자리에, 나머지를 `null_value`로 채운 (n, n) `sd.ndarray` float64다. 밀집 입력은 그래프 규칙(0, NaN, ±inf는 변 없음)으로 읽는다. sd의 CSR에는 중복 원소가 없으므로 scipy가 적은 "중복의 최솟값" 규칙은 생기지 않고, 저장된 NaN은 NaN이다(GR30; `[scipy와 다름]`: scipy는 inf).
- 이름은 scipy의 것이다(§9의 11번). 그래서 scipy 코드가 가져오기 경로만 바꿔 돈다.

### 2.8 결과 객체
- **namedtuple**(`stardust._results`, DESIGN §9.7): scipy가 튜플을 돌려주는 곳은 같은 순서의 namedtuple이다. 그래서 `d, p = dijkstra(G, return_predecessors=True)`처럼 풀린다(§9의 8번). 배열은 쓰기 금지다(API.md §2.2).
  | 이름 | 필드 | 쓰는 함수 |
  |---|---|---|
  | `ShortestPathResult` | `dist_matrix`, `predecessors` | `shortest_path dijkstra bellman_ford johnson floyd_warshall` (`return_predecessors=True`) |
  | `MinOnlyResult` | `dist_matrix`, `predecessors`, `sources` | `dijkstra(min_only=True, return_predecessors=True)` |
  | `YenResult` | `dist_array`, `predecessors` | `yen(return_predecessors=True)` |
  | `TraversalResult` | `node_array`, `predecessors` | `breadth_first_order depth_first_order` |
  | `ConnectedComponentsResult` | `n_components`, `labels` | `connected_components` |
  | `AssignmentResult` | `row_ind`, `col_ind` | `min_weight_full_bipartite_matching linear_sum_assignment` |
  | `LaplacianResult` | `laplacian`, `diag` | `laplacian(return_diag=True)` |
  | `BiconnectedComponentsResult` | `n_components`, `edges`, `labels` | `biconnected_components` |
- **`MaximumFlowResult`**: scipy처럼 클래스다(`flow_value`, `flow`). `repr`는 한 줄 `MaximumFlowResult(flow_value=5, flow=<csr_array (3, 3), nnz 6>)`이다(API.md §2.3). `flow`는 부를 때마다 새 `csr_array`다(희소 배열의 `data`는 쓸 수 있어야 하므로; spatial의 `tocsr()`와 같다). 클래스는 내보내지 않고 `type(r).__module__`은 `stardust.graph`다(src/stardust/graph/_flow.py).
- 결과 객체가 아닌 결과(거리 배열, 순열, 매칭 배열, 트리, 라플라시안)는 보통의 새 배열이라 쓸 수 있다.
- 최단 경로의 결과(`indices=None`이면 n² 원소; 선행 노드를 함께 돌려주면 16n² 바이트, auto의 밀집 길은 밀집 행렬 8n² 바이트를 더한다), `yen`의 K × n 선행 노드, `csgraph_to_dense`의 n²는 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2; §3.9).

### 2.9 scipy와 다른 곳 (모음)
위의 `[scipy와 다름]`을 모은다. 이유는 그 자리에 있다.
- 입력: complex는 `TypeError`, scipy 희소 객체는 `TypeError`(변환을 짚는다), 밀집 입력의 |w| ≤ 1e-8인 원소도 변이다(scipy는 최소 신장 트리뿐 아니라 모든 csgraph 함수의 밀집 입력에서 그 원소를 버린다: `dijkstra([[0, 1e-9], [0, 0]], indices=0)`이 [0, inf], GR2 마일스톤에서 확인), 희소 입력의 저장된 NaN·±inf 무게는 `ValueError`, `construct_dist_matrix`·`reconstruct_path`의 선행 노드는 정수 dtype이면 받고 경로가 아니면 `ValueError`(GR38), 노드 인자는 0 … n−1의 int만(실수 자르기와 음수 감기 없음), 2차원 `indices`는 `ValueError`, 옵션은 키워드만, `overwrite`·`copy`·`form='function'`·가린 배열 도구 없음, `maximum_flow`·`reverse_cuthill_mckee`·`structural_rank`·매칭도 밀집 입력을 받는다, `maximum_flow`의 int64 용량을 int32로 줄이지 않는다(scipy는 2⁴⁰을 0으로, 2³² + 5를 5로 읽는다), `min_weight_full_bipartite_matching`의 저장된 0은 무게 0의 짝(scipy는 경고와 함께 버린다), `reverse_cuthill_mckee`의 `symmetric_mode=False`는 A와 Aᵀ의 패턴의 합집합(scipy는 값의 합이라 0이 된 자리가 빠진다; GR52), bool 그래프의 `laplacian`(scipy는 `UFuncTypeError`; GR54).
- 결과: 색인 배열은 모두 int64(선행 노드, 레이블, 순서, 순열, 매칭, 흐름), 스칼라는 Python `int`, 강한 연결 요소의 번호는 가장 작은 노드 순서, 무향 순회는 합친 이웃의 오름차순, 무향 트리의 저장된 0은 0(GR29), `csgraph_to_dense`의 저장된 NaN은 NaN(GR30), FW를 뺀 최단 경로의 거리와 선행 노드는 방법과 무관, `auto`는 FW를 고르지 않음, `yen`의 같은 길이의 경로는 (변 수, 노드 열)의 순서(scipy의 마름모에서 첫 경로는 0–2–3, sd는 0–1–3), `construct_dist_matrix`는 `shortest_path`와 같은 합 순서, `laplacian`의 희소 결과는 CSR, 최소 신장 트리는 밀집의 작은 원소를 버리지 않는다, `maximum_flow`의 흐름은 int64(scipy는 int32; 용량이 int32 안이면 값은 scipy와 같다), 매칭과 배정은 동점에서 scipy와 다른 최적해를 고를 수 있다(GR45, GR47), `reverse_cuthill_mckee`의 시작 노드의 동점은 번호 순서(scipy는 numpy의 불안정 정렬이라 17노드부터 다를 수 있다; GR52), `laplacian`의 `form='lo'`는 행렬을 감싸 곱이 array 꼴과 같은 비트(GR51), 차수는 대각을 뺀 원소의 합(scipy는 전체 합 − 대각; GR54).
- 에러: 음수 무게의 `dijkstra`, NaN `limit`, 음수 용량(scipy는 0으로 본다), int64를 넘는 흐름(`OverflowError`), 실수 용량(`TypeError`; scipy는 `ValueError`), 최소 신장 트리와 `min_weight_full_bipartite_matching`의 저장된 NaN·±inf는 에러이고, `NegativeCycleError`는 `ValueError`의 하위 클래스이며 모든 방법이 "요청한 원점에서 닿는 순환"이라는 한 규칙이다.

---

## 3. 수치 방법

### 3.1 그래프의 내부 표현
- 그래프 함수는 Python 층(`src/stardust/graph/_common.py`의 `graph()`; 직사각을 받는 매칭 셋과 `structural_rank`는 `biadjacency()`, 흐름은 int64 용량의 `capacities()`)에서 입력을 **CSR 하나**(int64 `indptr`, `indices`, float64(흐름은 int64) 또는 패턴만의 값)로 바꾼 뒤 C++를 한 번 부른다. 행렬을 값 그대로 읽는 `laplacian`의 밀집 입력과 `linear_sum_assignment`의 비용 행렬은 C 순서 밀집 배열 그대로 넘긴다. 희소 CSR 입력은 복사하지 않는다(표준 꼴이 보장되어 그대로 읽는다; 다른 꼴은 `tocsr()` 한 번, 무게를 쓰면 float64가 아닌 값만 바꾼다). 밀집 입력은 변 규칙(§2.1)으로 CSR을 만든다(`_core.graph._dense_csr`: 크기로만 정한 행 묶음(원소 2¹⁵개씩)마다 한 번은 가지 없이 세고, 한 번은 행을 가지 없이 임시 행에 모아 제자리로 옮긴다; 원소 2¹⁵개부터 병렬. GR1 마일스톤에서 가지 있는 꼴의 25.6 ms를 4.1 ms로 줄였다, n = 2000, 1스레드).
- 무향(`directed=False`) 순회는 먼저 G가 자기 전치와 같은지를 전치를 만들지 않고 본다(`graph.hpp`의 `symmetric`: 행 u의 원소 (u, v)는 대칭이면 행 v에서 아직 쓰지 않은 다음 원소와 짝이므로 행마다 커서 하나로 한 번 훑는다; GR61). 같으면 G만 읽고, 아니면 sparse의 전치(`sparse::transpose`, sparse.md §2.6의 계수 정렬; 패턴만 읽으면 `indices`를 값 자리에 넣는다)를 한 번 만들어 G와 전치를 함께 읽는다(GR32). 최단 경로·`yen`·매개 중심성·PageRank의 무향 읽기(`paths::reading`), 구조 넷, `symmetric_mode=False`의 역 Cuthill–McKee도 같은 검사로 대칭 입력을 그대로 읽는다. 대칭이 아니면 `paths::reading`은 합친 CSR(두 방향 이웃의 합집합, 무게는 GR29; `paths::undirected`의 직렬 계수 정렬)을, 구조 넷은 대각을 뺀 합친 패턴을 만들고, 역 Cuthill–McKee는 두 줄을 함께 읽는다. 연결 요소는 합집합–찾기가 방향을 보지 않으므로 전치도 필요 없다. 최소 신장 트리는 아래 삼각을 열마다 모아 무향 변 목록을 만들고(§3.4), 흐름은 G와 Gᵀ의 패턴으로 잔여 CSR을 만든다(§3.5). scipy의 문서는 대칭 행렬이면 `directed=True`가 더 빠르다고 적는다. sd는 대칭 확인으로 그 차이를 줄인다.
- 들어오는 변이 필요한 계산은 전치 CSR을 한 번 만든다: 대칭화 라플라시안의 Aᵀ와 비대칭 입력의 무향 순회는 sparse의 전치(안정 계수 정렬; sparse.md §2.6), 유향 PageRank의 Pᵀx는 풀이 있으면 sparse의 병렬 전치, 1스레드면 확률을 쓰며 세는 정렬 하나(GR63), 유향 매개 중심성의 σ는 직렬 계수 정렬이다. 모두 표준 꼴의 전치라 줄과 항이 같다. 강한 연결 요소(Tarjan)는 나가는 변만 읽는다.
- 무게의 사전 검사(NaN·±inf, 음수)는 저장된 값을 한 번 훑는다. 검사는 계산 전에 하므로 실패한 호출은 아무것도 만들지 않는다.

### 3.2 최단 경로의 정의와 계산
**거리.** 원점 s(여럿이면 모두)에서 v까지의 거리는
d̂(v) = min over s–v 경로 p = (s = v₀, v₁, …, v_k = v) of L(p), L(p) = fl(…fl(fl(0 + w₁) + w₂) … + w_k)
다(원점에서부터의 차례 합; 0 + w₁ = w₁은 정확하다). 이완 x ↦ fl(x + w)는 x에 대해 단조이므로, 이 값은 이완 방정식 d(v) = min_u fl(d(u) + w(u, v))(원점은 0)의 **가장 작은 고정점**이고 어떤 이완 순서로 구해도 같다. 음이 아닌 무게에서는 fl(x + w) ≥ x라 순환을 넣어도 값이 작아지지 않아 최솟값이 단순 경로에서 나온다(Dijkstra의 정당성이 반올림 아래에서도 선다; Knuth의 일반화). 그래서:
- **D**(4진 색인 힙의 Dijkstra; GR39), **BF**(큐 기반의 이완: 바뀐 노드만 다시 본다), **J**(아래)는 모두 이 고정점에 도달하고 비트까지 같다. scipy의 D·BF·J도 거리가 차례 합과 같았다(설계 때 확인).
- **J**: Bellman–Ford로 구한 퍼텐셜 h는 **우선순위에만** 쓰고, 거리는 원래 무게의 차례 합으로 쥔다. 반올림 때문에 우선순위가 조금 어긋나 이미 꺼낸 노드가 나중에 더 작아질 수 있으므로, 더 작아지면 다시 넣는다(이완 기반이라 결과는 가장 작은 고정점 그대로다). 퍼텐셜은 요청한 원점에서 닿는 부분 그래프에서만 구한다(§2.3의 음수 순환 규칙). 퍼텐셜이 좋으면 다시 넣기는 드물다: GR2 마일스톤에서 음수 무게가 28–48%인 무작위 그래프(n = 2000, d = 6과 n = 5000, d = 10; 무게 + p(u) − p(v))의 원점 20개에서 다시 꺼낸 노드는 0번이었다(꺼낸 39,900번, 100,000번 가운데). 퍼텐셜은 음수 순환의 공통 검사(GR35)가 남긴 값이다. 큐는 4진 색인 힙이다(GR39).
- **FW**(밀집의 모든 쌍; GR25–GR27): 고전 Floyd–Warshall의 d(i, k) + d(k, j)는 경로를 가운데에서 잇는 합이라 차례 합의 정의와 끝자리가 다르다(scipy에서 상대 3.4–5.4×10⁻¹⁶, GR0 마일스톤의 기록). 처음 안은 `'FW'`도 **밀집 행렬 위에서 원점마다 도는 힙 없는 Dijkstra**로 같은 고정점을 구하는 것이었다. GR0 마일스톤에서 두 시제품을 쟀다(`tools/graph_probe_fw.cpp`, `tools/graph_probe.py fw`; 무게 [0.01, 1.01)의 완전 그래프와 밀도 1/4, 가장 좋은 값(세 번), ms):

  | n | 스레드 | 밀집 Dijkstra(가림 꼴) | 밀집 Dijkstra(NaN 꼴) | 블록 FW | 비(NaN 꼴 / FW) | scipy `floyd_warshall` |
  |---:|---:|---:|---:|---:|---:|---:|
  | 256 | 1 | 4.76 | 3.51 | 1.38 | 2.54 | 9.50 |
  | 256 | 10 | 0.68 | 0.67 | 0.62 | 1.08 | |
  | 512 | 1 | 35.6 | 23.0 | 11.0 | 2.08 | 58.9 |
  | 512 | 10 | 4.50 | 4.04 | 2.54 | 1.59 | |
  | 1024 | 1 | 277 | 160 | 87.3 | 1.84 | 397 |
  | 1024 (밀도 1/4) | 1 | 273 | 158 | 87.9 | 1.80 | |
  | 1024 | 10 | 33.3 | 20.1 | 15.6 | 1.29 | |
  | 2048 | 1 | 2577 | 1821 | 843 | 2.16 | |
  | 2048 | 10 | 316 | 312 | 130 | 2.41 | |

  - 밀집 Dijkstra의 한 걸음은 꺼낸 노드의 행 이완과 다음 노드 찾기를 한 번의 NEON 훑기로 한다. 가림 꼴은 방문한 노드에 +inf를 더해 막고, NaN 꼴은 방문한 키를 NaN으로 두어 FMIN이 그대로 두고 FMINNM이 건너뛰게 한다(64개 블록의 최솟값과 그 블록 안의 첫 색인; 결과는 두 꼴과 이진 힙 Dijkstra가 비트까지 같았다). 원소 하나에 적재 둘·저장 하나·연산 셋이라 레지스터 블록의 min-plus(원소 하나에 연산 둘)보다 원리상 1.5배 무겁고, n = 2048에서는 W(32 MB)가 L2를 넘어 원점마다 W 전체를 메모리에서 읽는다(10스레드의 비 2.4).
  - 1.3배를 넘으므로 §9의 2번의 규칙대로 `'FW'`는 블록 Floyd–Warshall의 반올림으로 바꿨다(GR25). 블록 크기 64는 고정이고(바꾸면 끝자리가 바뀐다), 블록 하나는 한 태스크가 맡아 스레드 수와 무관하다. 행렬은 블록마다 연속한 작업 사본(블록 (bi, bj)의 64 × 64 원소가 32 KB 한 덩어리)에서 돈다: 행 우선 행렬에서는 블록의 64행이 N × 8바이트(1 KB의 배수) 간격이라 캐시의 몇 집합에 몰려 계속 밀려났다. 배치만 바뀌고 연산은 같아 비트가 같다(GR60). 블록 FW는 같은 k 순서의 고전 FW와 같은 갱신을 하지만 단계 3에서 d(i, k)가 같은 블록의 뒤 k까지 갱신된 값이라 고전 FW와도 끝자리가 다를 수 있다. 그래서 FW의 거리는 "블록 FW의 값"으로만 정해지고, 계약(§4)은 다른 방법과 같은 한계, 상수 1이다(GR37: 덧셈 나무의 꼴과 무관한 한계라 블록의 합 깊이를 세지 않는다). 선행 노드를 함께 구해도 거리는 같은 비트다(GR42). n이 64의 배수가 아니면 +inf로 채운 행렬에서 돌고 앞 n × n을 옮긴다. 음수 무게는 그대로 받고 음수 순환은 대각이 음수가 되는 것으로 알린다(GR27).
  - auto는 FW를 고르지 않고, 음수 무게가 없고 `indices`가 없는 밀집한 그래프(4·nnz ≥ n²)에서 NaN 꼴의 밀집 Dijkstra를 쓴다(GR26, GR36; `paths.cpp`의 `dense_dijkstra`, 밀집 행렬 n² 하나를 만든다): 블록 FW보다 느리지만(위 표의 비 1.1–2.5) scipy의 `floyd_warshall`보다 1스레드에서 2.1배 빠르고(GR6 마일스톤, n = 1000) 결과가 D와 같다. 음수 무게면 밀도와 상관없이 CSR 위의 J다(GR36).
- **`unweighted=True`**: 무게 1의 차례 합은 변의 수와 같고 정확하다. 너비 우선(원점마다)으로 구한다.
- **`limit`**: d̂ ≤ `limit`인 노드에서 멈추는 것은 고정점의 그 부분과 같다(Dijkstra는 `limit`을 넘는 키를 꺼내면 멈춘다).

**선행 노드.** 거리가 정해지면 꼭 맞는 변의 집합 T = {(u, v) : fl(d̂(u) + w(u, v)) = d̂(v), u ≠ v}을 만든다(정확한 비교; `unweighted`면 d̂(u) + 1 = d̂(v)). h(v)를 T 위에서 원점으로부터의 너비 우선 변 수라 하면
pred(v) = min{u : (u, v) ∈ T, h(u) = h(v) − 1}
이다(원점과 닿지 않는 노드는 −9999). h가 사슬을 따라 하나씩 줄어 선행 노드 그래프는 언제나 트리다. 무게 0의 변(희소의 저장된 0)이나 흡수(fl(d + w) = d)가 있으면 "같은 거리의 가장 작은 번호"만으로는 순환이 생길 수 있어 변 수를 먼저 본다(§9의 3번). 모든 방법이 같은 d̂에서 같은 T, h, pred를 만들므로 선행 노드도 방법과 무관하다. 비용은 원점마다 O(n + E)의 한 번 훑기다. 정수 무게(와 `unweighted`)에서는 이것이 "가장 적은 변 수의 최단 경로 가운데 마지막 변의 출발이 가장 작은 것"이고 정확한 유리수 참값과 같다(§4).
- `min_only`: 모든 원점의 h가 0이고 `sources[v]`는 pred 사슬의 끝이다.
- 무향: T와 pred는 무향 읽기(대칭이면 G 그대로, 아니면 합친 CSR; §3.1)에서 만든다.

**합의 꼴과 `construct_dist_matrix`.** 거리 d̂(v)는 pred 사슬이 그리는 경로의 차례 합과 같다(T의 정의에서 바로 나온다). `construct_dist_matrix`는 그 경로를 원점에서부터 다시 더하므로 비트가 같다.

**Yen.** 경로의 전순서를 (L(p), 변 수, 노드 열의 사전 순서)로 둔다. 갈래 경로(spur)는 뿌리 경로의 차례 합 x에서 시작하는 Dijkstra(갈래 노드의 처음 값이 x)로 구해 전체 경로의 L이 원점에서부터의 차례 합이 되게 하고, 후보 힙도 이 전순서다. 그래서 동점이 많은 그래프에서도 앞선 K개가 데이터로만 정해진다. 작은 그래프에서 모든 고리 없는 경로를 나열한 참값과 정확히 같은지 테스트한다(§7.3).

**음수 순환.** 모든 방법이 같은 검사 하나(GR35: 요청한 원점에서 닿는 노드를 잇는 가상 뿌리의 큐 Bellman–Ford, 다시 넣은 횟수가 닿는 노드 수를 넘으면 부모 포인터의 순환)를 쓰고, 메시지의 노드는 그 순환의 가장 작은 노드다. 원점마다의 BF는 한 노드를 n번 넘게 넣으면, J는 n + 1번 넘게 꺼내면 멈추고, 멈춘 원점 가운데 가장 작은 것에서 같은 검사로 알린다(검사가 순환을 못 찾으면 그 원점을 적는다; 아래의 반올림 순환에서만 생긴다). 정확한 합이 음수인 순환은 언제나 걸린다. 정확한 합이 0인데 차례 합의 반올림이 계속 줄어드는 순환은 이론상 걸릴 수 있다(정수 무게에서는 생기지 않는다). 이 경우는 §4에 적는다.

### 3.3 순회, 연결 요소, 위상 정렬
- 너비 우선은 배열 큐, 깊이 우선은 명시적 스택(재귀 없음; 노드마다 다음에 볼 이웃의 위치와 줄의 끝)이다. 찾은 노드의 표시는 노드마다 한 비트라(n/8바이트) 1차 캐시에 남는다. 무향 깊이 우선은 두 줄의 머리 가운데 작은 것을 가지 없이 고른다(GR6 마일스톤). 이웃 순서는 CSR의 오름차순이라 순서가 데이터로만 정해진다. 무향은 GR32의 두 줄 읽기다. 트리는 노드마다 찾은 변의 무게를 적어 두고 끝에 노드 오름차순으로 부모의 줄에 넣는다(그래서 줄 안이 오름차순인 표준 꼴이다).
- 약한 연결(과 무향): 합집합–찾기(경로 반감, 번호가 작은 뿌리로 합침)로 분할을 구하고 마지막에 가장 작은 노드 순서로 번호를 다시 매긴다. 분할이 유일하므로 저장된 원소가 2¹⁸개 이상이고 풀을 쓸 수 있으면 행 구간(원소 2¹⁶개씩, 크기로만 정함)마다 병렬로 합친다: 큰 뿌리를 작은 뿌리 밑에 비교 교환으로 잇고, 어떤 순서로 합쳐도 같은 분할이다(GR31).
- 강한 연결: 반복형 Tarjan(발견 번호, low link, 노드마다 다음 변의 위치, 미배정 노드의 스택)으로 분할을 구하고 번호를 다시 매긴다. O(n + E), 직렬이다(설계 때 scipy의 강한 연결이 10⁶ 변에 6.2 ms였고, 병렬 알고리즘이 이길 크기가 드물다. Pearce의 꼴도 재어 보고 넣지 않았다: GR68).
- 위상 정렬: 진입 차수의 Kahn 알고리즘이고 준비된 노드 가운데 가장 작은 것을 꺼낸다. 준비된 노드의 집합은 힙 대신 층을 이룬 비트맵(위 층의 비트가 아래 64비트 낱말이 비지 않았음을 적는다)이라 넣기와 가장 작은 것 꺼내기가 낱말 몇 번의 연산이다. 남은 노드가 있으면(자기 고리 포함) 순환이다.

### 3.4 최소 신장 트리
- 변을 키 (w, min(i, j), max(i, j))로 견주면 전순서이고, 모든 무게가 다른 그래프의 최소 신장 숲이 유일하다는 정리가 그대로 서서 답이 하나다. 무향 규칙(작은 무게)으로 합친 변 목록을 이 키로 정렬하고 Kruskal(합집합–찾기)로 고른다(scipy도 Kruskal이다).
- 변 목록은 행 i마다 위 삼각의 저장 원소 (i, j)와 아래 삼각의 저장 원소 (j, i)(열마다 모은 계수 정렬)를 j 순서로 합쳐 (min, max) 순서로 만든다. 정렬은 (w + 0.0의 순서 키, 변 번호)의 안정 LSD 기수 정렬(11비트 자리 여섯, 모든 키가 같은 자리는 건너뜀; 변 256개 이하는 안정 비교 정렬)이라 같은 무게가 (min, max) 순서로 남는다(GR43). 직렬이고 저장 원소 2¹⁵개부터 GIL을 놓는다.
- Borůvka는 넣지 않았다: GR3 마일스톤에서 이 Kruskal이 n = 10⁵, d = 5 대칭에서 11.1 ms(scipy 76.2 ms, 1스레드)라 병렬 Borůvka의 이득을 잴 만한 몫이 작았다(GR43). GR6 마일스톤에서 저장 원소 10⁷개까지 scipy의 5–7배였고 정렬의 병렬화도 1.06배라 그대로 두었다(GR65).
- 결과의 자리: 고른 무향 변 {i, j}는 그 무게를 준 저장 원소(비트가 같은 쪽)의 자리에 둔다(두 방향의 비트가 같으면 i < j의 위 삼각). 고른 원소의 표시로 입력을 걸러 결과를 만들므로 표준 꼴이고 값은 저장된 그대로다.

### 3.5 흐름과 매칭
- **최대 흐름**: Dinic(층 그래프와 막는 흐름)과 Edmonds–Karp(너비 우선 증가 경로)다. 잔여 그래프는 변마다 반대 변의 자리를 쥔 CSR(입력에 없는 반대 방향을 더한 패턴; 패턴이 대칭이라 반대 자리는 행을 차례로 걸으며 센다)이다. 너비 우선과 깊이 우선의 이웃 순서는 오름차순이라 돌려주는 흐름이 데이터로만 정해진다. Edmonds–Karp는 싱크에 닿으면 너비 우선을 멈추고 그 경로를 병목만큼 민다. Dinic은 싱크의 층까지 층을 매기고, 현재 호(가장 작은 이웃부터)의 깊이 우선으로 경로를 하나씩 밀며 막다른 노드는 층에서 뺀다. 잔여 용량은 uint64, 흐름은 int64이고 덧셈이 넘치지 않음은 원점 용량 합의 검사(§2.4)가 보장한다(GR44).
- **가장 큰 이분 매칭**(`maximum_bipartite_matching`, `structural_rank`): 탐욕 시작과 Hopcroft–Karp(scipy와 같은 알고리즘)이고 행과 이웃을 오름차순으로 본다. 깊이 우선은 명시적 스택이라 긴 경로에서도 재귀가 없다(GR47).
- **가장 작은 무게의 완전 매칭과 배정**: 두 함수가 같은 C++ 풀이(csrc/graph/matching.cpp의 `Solver`; LAPJV의 시작과 최단 증가 경로, GR45)를 쓰고 밀집과 희소의 행 읽기만 다르다(밀집은 남은 열을 모두 훑고, 희소는 이웃 열만 이완하고 닿은 열을 이진 색인 힙에 둔다). 고르는 순서가 (거리, 빈 열 먼저, 번호)의 전순서라 같은 짝이면 두 함수가 같은 배정을 낸다. 동점(여러 최적 배정)에서는 이 고정된 순서의 결과를 돌려준다(§9의 13번). 정수 비용은 쌍대 변수와 비용이 모두 정확하여 최적이 정확하다. 직사각은 scipy처럼 작은 쪽을 기준으로 한다(행이 많으면 전치해 풀고 행 순서로 돌려준다). 희소 탐색의 작업 공간은 닿은 열만 되돌려 행마다 O(n)을 쓰지 않는다. `min_weight_full_bipartite_matching`의 입력이 모든 짝을 저장하면(저장 원소가 rows × cols개) CSR의 값이 곧 행 우선 밀집 행렬이라 밀집 훑기로 푼다: 같은 짝이면 두 훑기가 같은 배정을 내므로 결과가 같고 더 빠르다(GR62).

### 3.6 순서와 라플라시안
- **역 Cuthill–McKee**(csrc/graph/ordering.cpp `_rcm`; GR52): `symmetric_mode=False`면 패턴이 자기 전치와 같은지 먼저 보고(GR61) 같으면 G만, 아니면 sparse의 전치를 한 번 만들어 두 줄을 합쳐 읽는다(합친 CSR은 만들지 않는다; GR32의 꼴). 차수를 한 번 세고, 시작 노드의 순서는 차수의 계수 정렬(안정이라 동점은 번호 순서), 너비 우선의 새 이웃 묶음은 (차수, 번호)로 정렬한다. O(n + E + 묶음의 정렬)이고 직렬이다(작업량이 2¹⁵ 이상이면 GIL을 놓는다).
- **라플라시안**(csrc/graph/ordering.cpp `_laplacian_csr`, `_laplacian_dense`; GR50, GR53, GR54): 행렬 M은 A, 대칭화면 A + Aᵀ(전치를 만들어 줄마다 합치고 정확히 0인 합은 버린다; 줄 i는 A와 Aᵀ의 앞 줄들이 가질 수 있는 가장 많은 원소 뒤에서 시작하는 자리에 한 번의 합치기로 쓰고 끝을 따로 쥔다: 세는 합치기와 쓰는 합치기의 두 번을 한 번으로 줄였고, 그 대신 nnz(A) + nnz(Aᵀ)칸을 잡는다; GR64. 밀집은 64 × 64 타일로 더한다). 차수는 행(또는 열)마다 대각을 뺀 원소를 **색인 오름차순으로 0부터 하나씩** 더한 것이다(나가는 차수와 대칭화는 행 합, 들어오는 차수는 희소에서 행 순서의 흩뿌림, 밀집에서 열 32개 묶음의 열 합). `normed`는 w = √d(d = 0이면 1)를 한 번 구하고 L_ij = −((A_ij / w_i) / w_j)(scipy 희소의 순서), 대각은 d ≠ 0이면 1, d = 0이면 0이다(scipy와 같다; GR50). 비정규화는 L_ij = −A_ij, 대각은 d다. 부호 바꾸기는 NaN을 그대로 두고(scipy의 −1 곱) 0은 −0.0이 된다(scipy와 같다; GR53). 희소 결과는 행마다 (대각 밖 원소 수 + 1)을 세어 자리를 정하고 오름차순으로 대각을 끼워 쓴다. 밀집의 비정규화는 1스레드에서 한 번의 훑기다: 들어오는 차수는 열마다 누산기 하나를 두고 행을 차례로 읽으며 더하고 부호를 바꿔 쓰며, 나가는 차수와 대칭화는 8행을 더한 뒤 캐시에 있는 동안 쓴다(같은 합, 같은 순서라 비트가 같다). 풀이 있으면 따로 나눈 두 번의 훑기가 더 빨라 그대로 둔다(GR64).

### 3.7 단절점·다리·이중 연결·코어 수, PageRank, 매개 중심성
- **무향 구조 넷**(csrc/graph/structure.cpp; GR59): 패턴이 대칭이고 대각이 없으면 그대로, 아니면 두 방향의 합집합에서 대각을 뺀 CSR을 읽는다. 단절점·다리·이중 연결 요소는 반복형 Hopcroft–Tarjan 하나다: 뿌리를 오름차순으로, 노드마다 다음에 볼 이웃의 위치를 쥔 명시적 스택의 깊이 우선(이웃 오름차순)이고 low(v)는 v의 부분 트리에서 뒤 변 하나로 닿는 가장 이른 발견 번호다. 나무 변 p–v는 low(v) ≥ disc(p)면 이중 연결 요소 하나를 닫고(그 변을 넣은 뒤 쌓인 변 전부), low(v) > disc(p)면 다리이며, p를 단절점으로 만든다(뿌리는 자식이 둘 이상일 때만). 요소의 번호는 끝에 변의 사전 순서로 다시 매긴다. 코어 수는 Batagelj–Zaversnik(차수의 통 정렬, 가장 작은 차수부터 떼며 큰 이웃의 차수를 하나씩 줄인다)이다. 답이 집합과 분할이라 훑는 순서와 무관하다. O(n + E), 직렬이다.
- **PageRank**(csrc/graph/centrality.cpp `_pagerank`; GR56, GR57): T(x) = α·Gᵀx + (1 − α)·v(G는 매달린 행을 `dangling`으로 바꾼 행 확률 행렬)는 ℓ₁에서 비율 α의 축소 사상이다. 계산한 반복 x̂_{k+1} = T(x̂_k) + e_k(‖e_k‖₁ ≤ ε)에서 ‖x̂_{k+1} − x\*‖₁ ≤ (α‖x̂_{k+1} − x̂_k‖₁ + ε)/(1 − α)가 증명된 사후 한계이고, 이것이 `tol` 이하가 되거나 사전 횟수 ⌈log(tol/2)/log α⌉ + 1(α = 0.85, tol = 10⁻¹²에서 176)에 닿으면 멈춘다. 처음 값은 v다.
  - 준비: 나가는 무게의 합은 행 구간(패턴으로만 정함)이 태스크다. 유향의 전치는 풀이 있으면 sparse의 병렬 전치를 만들어 그 값 자리에서 전이 확률로 바꾸고, 1스레드면 확률을 쓰며 세는 정렬 하나다(둘 다 표준 꼴의 전치라 줄과 항이 같다; GR63). 무게 없음의 x(j)/차수(j)도 걸음마다 행 구간이 태스크다.
  - Gᵀx는 전치 CSR의 행 모으기다: 행(노드)마다 들어오는 항(w/out(j)·x(j), 무게 없음은 x(j)/차수(j))을 공용 pairwise 나무로 더한다(16항 미만의 줄은 같은 한 사슬). 행 구간은 패턴으로만 정한다. 매달린 질량, 걸음의 ℓ₁ 차, Σx̂는 같은 나무를 고정 잎으로 자른 합이다. 그래서 결과는 스레드 수와 무관하다.
  - 모든 항이 음이 아니라 상쇄가 없고, 걸음 하나의 반올림은 항마다 γ_K의 상대오차다(K는 GR57).
- **매개 중심성**(csrc/graph/centrality.cpp `_betweenness`; GR58, Brandes): 원점마다 최단 경로 DAG를 만들고 경로 수 σ를 앞으로, 의존 δ를 뒤로 쌓는다.
  - 무게 없음: 너비 우선(층의 순서가 DAG의 위상 순서다). 무게 있음: §3.2의 d̂와 원점마다의 Dijkstra(4진 색인 힙; GR39)이고, 같은 거리가 이어지면 꼭 맞는 변의 너비 우선 변 수 h를 구해 (h, 번호)로 줄 세운다. DAG의 변은 fl(d̂(u) + w) = d̂(v)이고 (d̂(u), h(u)) < (d̂(v), h(v))인 (u, v)다(무게가 양수라 흡수가 없으면 표준 정의와 같다; 흡수가 만드는 같은 거리의 걸음도 순환 없이 선행 노드를 가진다).
  - σ(v)는 들어오는 변을 번호 오름차순으로, δ(u) = Σ σ(u)/σ(w)·(1 + δ(w))는 나가는 변을 번호 오름차순으로 끌어와 더한다. σ는 경로 수라 n이 수백이어도 double을 넘칠 수 있다(다이아몬드 사슬에서 2^(n/3)). 한 원점에서 σ가 2⁹⁶⁰에 닿으면 그 원점을 (double 가수, int64 지수) 꼴로 다시 센다: 덧셈마다 반올림 하나라 double로 되는 곳에서 같은 비트이고 넘침이 없다. δ에 쓰는 비만 double이다.
  - 원점은 ⌈n/256⌉개(1–64; 크기로만 정함)씩 태스크이고, 태스크의 부분합을 태스크 번호 순서로 더한다. 결과가 스레드 수와 무관하다. 메모리는 태스크마다 O(n)과 동시에 도는 태스크의 부분합(4 × 스레드 수, 128 MB까지)이다.

### 3.8 병렬화와 결정성
**같은 입력이면 스레드 수와 `method`(FW를 뺀)에 상관없이 같은 결과다.** 테스트(`tests/graph/test_threads.py`)와 골든은 1, 3, 10스레드에서 같은 바이트를 확인한다(§7).

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 입력 변환(밀집 → CSR, `csgraph_to_dense`)과 sparse의 전치 | 행 구간(원소 2¹⁵개씩), 고정 조각의 계수 정렬 | 행 하나는 한 태스크이고 조각 순서가 정해져 있다(sparse.md §2.6) |
| 여러 원점의 최단 경로(`indices`, 모든 쌍, auto의 밀집 Dijkstra) | 원점 1–64개(크기로만 정한 수: 4096 / (n + E))가 태스크 하나(GR40) | 원점 하나의 결과는 정의로 정해진다(§3.2) |
| `construct_dist_matrix`(n² ≥ 2¹⁵) | 행(원점) 하나가 태스크 하나 | 행 하나의 합이 정해져 있고, 잘못된 선행 노드는 행 순서로 가장 앞선 것을 알린다 |
| FW(블록 Floyd–Warshall; GR25) | 단계마다 64 × 64 블록 하나가 태스크 하나, 단계 사이에 장벽 | 블록 하나의 연산 순서가 블록 크기로만 정해진다 |
| 약한 연결 요소(저장된 원소 2¹⁸개부터; GR31) | 행 구간(원소 2¹⁶개씩) | 분할이 유일하고 번호는 가장 작은 노드 순서다 |
| 라플라시안(희소: 대칭화의 합, 나가는 차수, 결과의 행; 밀집: 행 구간과 열 32개 묶음; GR4 마일스톤) | 크기로만 정한 행 구간·열 묶음 | 행 하나, 열 하나의 합이 한 태스크이고 순서가 정해져 있다(들어오는 차수의 희소 흩뿌림은 직렬) |
| PageRank의 준비(나가는 무게, 전치, 전이 확률; GR63)와 Pᵀx와 벡터 합(GR57) | 원소와 행 2¹⁵개쯤의 행 구간, 벡터 합은 2¹⁶항 잎 | 행 하나의 나무가 정해져 있고, 벡터 합은 pairwise 나무의 잎을 같은 나무로 합친다 |
| 매개 중심성(GR58) | 원점 ⌈n/256⌉개(1–64)의 태스크, 4 × 스레드 수씩 | 부분합을 태스크 번호 순서로 더한다 |
| 직렬(단일 원점 Dijkstra·BF·J, `min_only`, `yen`, `reconstruct_path`, 합친 CSR과 대칭 확인, 무게 검사, 순회, 강한 연결, 위상 정렬, 최소 신장 트리(GR43), 흐름, 매칭, 배정, RCM, 단절점·다리·이중 연결, 코어 수) | — | 고정된 훑기 순서(최소 신장 트리는 키가 전순서라 유일하다) |

- 병렬 기준값은 배열의 규칙이다(array.md §10.4): 작업량(변의 수 × 원점 수 등)이 float64 연산 2¹⁵개 분량 이상이면 나누고, 그 이상이면 스레드 수와 상관없이 GIL을 놓는다(API.md §3). 직렬 알고리즘도 기준값 이상이면 GIL을 놓는다(`Unlocked`, ARCHITECTURE §4.1).
- 단일 원점의 병렬 최단 경로(Δ-stepping)는 결과가 같은 고정점이라 넣을 수 있지만 첫 판에는 넣지 않는다(나중; GR6 마일스톤에서도 넣지 않았다: 원점 하나의 Dijkstra가 이미 scipy의 2.4배이고, 여러 원점은 원점 태스크로 10스레드에서 7.5배 이상 빨라진다).

### 3.9 메모리
- 사용자가 모양으로 요청한 출력(모든 쌍의 n² 거리와 선행 노드, `yen`의 K × n, `csgraph_to_dense`의 n²)은 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2). FW는 결과와 블록 작업 사본(64의 배수로 채운 N²)을 함께 센다(GR60, GR67).
- 내부: CSR 사본(무향의 합친 CSR, 전치, 잔여 그래프)은 입력 크기의 1–2배다. 원점마다의 작업(색인 힙, 거리, h, 순서)은 태스크마다 길이 n의 배열 몇 줄이다. auto의 밀집 길은 밀집 행렬 n²(8n² 바이트)를 한 번 만든다. FW 길은 결과 n²와 블록마다 연속한 작업 사본 N²(선행 노드도 같다; GR60)다. 희소 라플라시안의 대칭화는 A + Aᵀ의 자리로 nnz(A) + nnz(Aᵀ)칸을 잡는다(GR64). 매개 중심성은 태스크마다 σ·δ·순서 O(n)과 결과의 부분합 n이다.

---

## 4. 정확도 계약
판정은 `tests/graph/refdata.py`의 `bound()`·`units()` 하나이고 테스트와 정확도 체계(`bench/accuracy/graph.py`의 스윕 다섯과 그 표 `bench/graph/accuracy.md`)가 같이 쓴다(spatial.md §4와 같은 꼴: 계약 단위로 적고 경계는 1이다).

- **정확한 것**(참값과 같아야 한다): 연결 요소, 순회의 순서와 트리, 위상 정렬, 최소 신장 트리(고른 변의 집합과 자리), 역 Cuthill–McKee, 구조적 랭크, 매칭의 크기, 흐름 값(과 흐름의 보존·용량 조건), 단절점, 다리, 이중 연결 요소, 코어 수. 참값은 작은 그래프에서 정의를 그대로 옮긴 Python 구현(정확한 유리수)이다.
- **최단 경로**
  1. **방법과 무관하다**: D, BF, J, auto의 거리와 선행 노드가 비트까지 같다(모든 시험 그래프, 스레드 수마다). FW의 거리는 3번의 한계 안이고(상수 1, p̂는 FW의 선행 노드가 그리는 경로; GR37), 스레드 수마다 같은 비트다.
  2. **정수 무게(와 `unweighted`)는 정확하다**: 모든 부분합이 2⁵³ 이하이면 거리는 정확한 최단 거리와 같고, 선행 노드는 정확한 유리수로 계산한 §3.2의 규칙과 같다. 동점이 아주 많은 격자에서 동점 규칙을 직접 시험한다.
  3. **일반 무게**: 거리는 그 선행 경로의 차례 합과 비트까지 같다(`construct_dist_matrix`와 같다). 참 최단 거리 d\*와의 차이는 |d̂ − d\*| ≤ γ_{H−1}·max(Σ_{p̂}|w|, Σ_{p\*}|w|)다. H는 돌려준 경로 p̂와 참 최단 경로 p\* 가운데 긴 쪽의 변 수, Σ|w|는 그 경로의 무게 절댓값의 합이다(차례 합의 표준 한계: d̂ = L(p̂) ≥ S(p̂) − γΣ|w| ≥ d\* − γΣ|w|, d̂ ≤ L(p\*) ≤ d\* + γΣ|w|). 음이 아닌 무게면 상대 γ_{H−1}·d\*/(1 − γ_{H−1})다. 참 최단 거리는 정확한 유리수 Dijkstra(작은 그래프), 경로의 정확한 합은 Fraction이다.
  4. **음수 순환**: 정확한 합이 음수인 순환이 요청한 원점에서 닿으면 언제나 `NegativeCycleError`다. 정확한 합이 0인 순환에서 반올림이 값을 계속 줄이면 에러가 날 수 있다(정수 무게에서는 없다). 그 밖에는 에러가 나지 않는다.
- **`yen`**: 정수 무게에서 모든 고리 없는 경로를 나열한 정확한 정렬의 앞 K개와 같다. 일반 무게에서 경로마다 거리 계약(3번)이다.
- **`linear_sum_assignment`, `min_weight_full_bipartite_matching`**: 정수 비용에서 돌려준 배정의 정확한 비용이 최적과 같다(작은 문제는 나열, 큰 문제는 쌍대 실현 가능성의 정확한 확인). 일반 비용에서 돌려준 배정의 정확한 비용 C(σ̂)는 C\* + γ_{2r}·Σᵢ(|c_{iσ̂(i)}| + |c_{iσ\*(i)}|) 이하다(r은 작은 쪽의 크기; 상수 1, GR49). 판정은 `tests/graph/refdata.py`의 `assignment_units()`이고 참 최적은 참값 파일(작은 문제의 나열)과 정확한 유리수 풀이(`tests/graph/definitions.py`의 `assignment_exact`)다. GR3 마일스톤의 측정(`bench/accuracy/graph.py`의 `assignment`, 14영역)에서 sd와 scipy 모두 0 단위(참 최적)였다.
- **`laplacian`**(GR55): 항 m개의 차수는 |d̂ − d| ≤ γ_k·s(k = m, 대칭화면 m + 1; s는 그 항을 이루는 |A|의 합), 대칭화의 대각 밖은 γ₁·(|A_ij| + |A_ji|), `normed`의 대각 밖은 |L̂ − L| ≤ |L|·((1 + γ_c)/√((1 − ρ_i)(1 − ρ_j)) − 1)(ρ = γ_k·s/|d|, c = 4, 대칭화면 5), 비정규화의 대각 밖과 정규화의 대각은 정확하다. 정수·bool 입력은 int64로 정확하다(감기지 않는 한). 판정은 `tests/graph/refdata.py`의 `laplacian_bound()`·`laplacian_units()`, 참값은 Arb(tests/graph/ref/laplacian.npz 51점)다. GR4 마일스톤의 측정(`bench/accuracy/graph.py`의 `laplacian`, float 그래프 11영역 4,943점)에서 가장 큰 값이 0.87 단위(dangling/50 대칭화의 대각 밖), 정규화는 0.32 단위, ulp로 3 이하였고 scipy도 같은 한계 안이었다.
- **`pagerank`**(GR56, GR57): ‖x̂ − x\*‖₁ ≤ tol + F, F = γ_K·(1 + 10⁻⁹)/(1 − α). x\*는 (I − αMᵀ)x = (1 − α)v(매달린 행을 바꾼 M)를 Arb의 행렬 풀이(`arb_mat.solve`)로 푼 참값, K는 걸음 하나에서 항이 거치는 반올림의 수다(GR57; tests/graph/refdata.py `pagerank_k`·`pagerank_bound`). 바닥 F가 `tol`보다 크면 사전 횟수에서 멈추고 결과는 같은 한계 안이다. GR5 마일스톤의 측정(`bench/accuracy/graph.py`의 `pagerank`, pagerank.npz 25점 × tol 10⁻¹²·10⁻¹⁵)에서 가장 큰 값이 0.79 단위(components/40, tol = 10⁻¹²)였다. networkx(tol = 10⁻¹²/n의 연속 차)는 세 영역에서 이 한계를 넘었다(가장 큰 값 4.43배).
- **`betweenness_centrality`**(GR58): 노드마다 상대오차 ≤ γ_K·(참값)이고 참값 0은 정확히 0이다. 모든 항이 음이 아니라 상쇄가 없고, K = L(2A + Δ_out + 2) + n + 3(L은 가장 긴 DAG 경로의 변 수, Δ_out·Δ_in은 가장 큰 DAG 진출·진입 차수, A는 모든 σ < 2⁵³이면 0 아니면 L(Δ_in − 1); tests/graph/refdata.py `betweenness_k`)이다. 참값은 계산한 DAG 위의 정확한 유리수 Brandes(σ는 Python 정수; tests/graph/definitions.py `betweenness_dag`)와 정확한 거리의 Brandes(betweenness.npz; 이 그래프들에서 두 DAG가 같다)다. 정수·2진 유리수 무게와 `unweighted`는 계산한 DAG가 참 DAG다. 일반 무게에서 두 경로의 정확한 합이 같은데 반올림한 합이 다르면(또는 그 반대) DAG가 달라 참 Brandes와의 계약은 없다(정의는 §3.2의 거리다). GR5 마일스톤의 측정(`bench/accuracy/graph.py`의 `betweenness`, 76점, 노드값 2,360개)에서 가장 큰 값이 0.076 단위였다.
- GR2 마일스톤에서 잰 최단 경로(`bench/accuracy/graph.py`의 `paths`: 일반 무게의 참값 7,639점(정의와 FW로 15,278점), 그래프 11개 × 유향·무향): 정의의 거리와 FW의 거리 모두 가장 큰 값이 0.995 단위(heavytail/120 유향), ulp로 2 이하였고, 정수·2진 유리수·bool 무게와 `unweighted`는 참값과 비트까지 같았다(tests/graph/test_paths_reference.py). scipy의 D와 FW도 같은 그래프에서 같은 값이었다(밀집 dense/special6만 scipy가 1e-9의 변을 버려 다르다; §2.9).
- scipy와의 차이(동점, FW의 끝자리, 거꾸로 더한 `construct_dist_matrix`)는 계약이 아니라 정확도 표의 비교 패키지 열로 보인다.

---

## 5. 성능
설계 때 잰 scipy 1.18.1의 시간이다(ms, 1스레드(scipy의 csgraph는 1스레드다), 가장 좋은 값(세 번, 1초를 넘는 것은 한 번), 부하 평균 1.5). 그래프는 무작위 유향(노드마다 나가는 변 d개, 무게 [0.01, 1.01)), "대칭"은 그것과 전치의 큰 값이다. GR0 마일스톤에서 `tools/graph_probe.py time`이 같은 크기를 다시 쟀고(아래), 속도 사례(`bench/speed/cases/graph.py`의 57개)가 같은 크기를 sd와 scipy·networkx로 잰다. 아래의 마일스톤별 sd 시간은 개발 중의 확인이고, 기준선은 메인 세션의 전체 측정 때 승격한다.

| 일 | scipy | 메모 |
|---|---:|---|
| `connected_components` weak / strong, n = 10⁵, d = 10 | 7.75 / 6.21 | |
| `breadth_first_order` / `depth_first_order`, n = 10⁵, d = 10 | 2.32 / 3.98 | |
| `dijkstra` 원점 1개, n = 10⁵, d = 10 | 21.8 | 검증·변환이 큰 몫이다(같은 그래프의 `unweighted` 9.64) |
| `dijkstra` 원점 100개 / `min_only` | 2120 / 21.4 | 원점마다 독립이라 sd는 병렬 |
| `bellman_ford` 원점 1개, n = 10⁴, d = 10 | 1110 | scipy는 매 회 모든 변을 이완한다. 큐 기반은 보통 수십 배 적게 본다 |
| `johnson` / `dijkstra` 모든 쌍, n = 2000, d = 10 | 606 / 563 | |
| `floyd_warshall` 밀집 n = 1000 | 390 | |
| `minimum_spanning_tree` 대칭 n = 10⁵, d = 5 | 76.6 | |
| `yen` K = 10, n = 10⁴, d = 10 | 105 | 그래프 만들기 포함 |
| `reverse_cuthill_mckee` 대칭 n = 10⁵ | 9.53 | |
| `structural_rank` / `maximum_bipartite_matching` n = 10⁵, d = 10 | 51.5 / 51.6 | |
| `maximum_flow` dinic / edmonds_karp, n = 10⁴, d = 10, 정수 용량 1–99 | 9.35 / 5.6 | |
| `min_weight_full_bipartite_matching` 밀집 1000² / 희소 5000(d = 20 + 대각) | 12.4 / 22.6 | |
| `linear_sum_assignment` 1000² | 18.7 | |
| `laplacian` / `normed` 대칭 n = 10⁵ | 5.21 / 6.64 | |

- GR0 마일스톤에서 `tools/graph_probe.py time`이 같은 크기를 다시 쟀다(부하 평균 2.2): 대부분 위의 10% 안이고, 다른 것은 `bellman_ford` 884(위 1110), `structural_rank`·`maximum_bipartite_matching` 32.2·32.3(51.5·51.6), `yen` 127(105), `maximum_flow` edmonds_karp 8.32(5.6), 밀집 `min_weight_full_bipartite_matching` 37.4(12.4; `csr_array`로 바꾸는 시간 포함), 희소 29.9(22.6), `laplacian` 9.23 / 12.2(5.21 / 6.64)다. 같은 때의 networkx 3.7(ms): `articulation_points` 16.5, `bridges` 93.9, `biconnected_components` 21.8, `core_number` 15.4(n = 10⁴, d = 5 대칭), `pagerank` 13.5(n = 10⁴, d = 5, tol 10⁻⁶), 무게 없는 `betweenness_centrality` 779(n = 1000, d = 5).

- GR1 마일스톤에서 잰 sd(ms, 가장 좋은 값 일곱 번, 1스레드 / 10스레드, 같은 때의 scipy 1스레드; 부하 평균 1.5–2.2): n = 10⁵, d = 10에서 `connected_components` weak 1.76 / 0.77(scipy 6.92), strong 5.12 / 5.14(5.82), `directed=False` 1.75 / 0.88(7.06), `breadth_first_order` 1.29(1.64), `depth_first_order` 3.01(3.32), 무향 `breadth_first_order` 4.78 / 5.13(6.73), 대칭 그래프의 무향 7.44 / 6.13(18.4), `breadth_first_tree` 1.80(7.58), `depth_first_tree` 3.70(9.08); DAG(노드마다 나가는 변 약 5)의 `topological_sort` 3.09(networkx의 `lexicographical_topological_sort` 239); n = 2000의 절반이 0인 밀집 행렬에서 `csgraph_from_dense` 4.7 / 0.84(42.2), `csgraph_to_dense` 0.85 / 0.24(12.8), 밀집 입력의 `connected_components` 5.1 / 0.94(46.3). 순회와 강한 연결은 직렬이라 10스레드도 같다.
- GR2 마일스톤에서 잰 sd(ms, 가장 좋은 값 일곱 번, 1스레드 / 10스레드, 같은 때의 scipy 1스레드; 부하 평균 1.4–1.8): n = 10⁵, d = 10에서 `dijkstra` 원점 1개 9.50 / 9.30(21.5), `unweighted` 1.48(9.54), 원점 100개 917 / 128(2181), `min_only` 9.31(23.3); n = 10⁴, d = 10의 `bellman_ford` 원점 1개 0.74(887; 큐 기반이라 양수 무게에서 너비 우선과 비슷하다), `yen` K = 10 74.0(129); n = 2000, d = 10의 모든 쌍 `johnson` 213 / 24.8(621), `dijkstra` 212 / 24.9(586); 밀집 n = 1000의 `floyd_warshall` 92.6 / 14.6(369; 1스레드 3.98배, 64의 배수로 채운 n = 1024의 커널이 거의 전부다), auto의 밀집 Dijkstra 179 / 21.7(376). 선행 노드를 함께 구하는 FW는 192 ms(n = 1024, 1스레드)다.
- GR3 마일스톤에서 잰 sd(ms, 가장 좋은 값 다섯–일곱 번, 1스레드, 같은 때의 scipy; 부하 평균 1.7): `minimum_spanning_tree` 대칭 n = 10⁵, d = 5 11.1(76.2), `maximum_flow` dinic 3.46(8.97) / edmonds_karp 10.2(11.05)(n = 10⁴, d = 10, 용량 1–99), `maximum_bipartite_matching` n = 10⁵, d = 10 21.4(31.1), `linear_sum_assignment` 1000² 6.95(20.1), `min_weight_full_bipartite_matching` 밀집 1000²의 `csr_array` 9.8–11.9(11.1–26.7; 시드 셋) / 희소 5000(d = 20 + 대각) 16.9(23.1). 모두 직렬이라 10스레드도 같다.
- GR4 마일스톤에서 잰 sd(ms, 가장 좋은 값 일곱 번, 1스레드 / 10스레드, 같은 때의 scipy 1스레드; 부하 평균 1.5–1.7): n = 10⁵, d = 10에서 `reverse_cuthill_mckee` 대칭 `symmetric_mode=True` 3.84(11.8), A와 Aᵀ의 합집합 15.5(23.4), `structural_rank` 21.1(31.7)(셋은 직렬이라 10스레드도 같다), `laplacian` 대칭 5.21 / 2.29(9.22; 들어오는 차수의 흩뿌림이 직렬), `normed` 5.96 / 2.45(12.0), 유향의 `use_out_degree=True` 1.87 / 0.34(3.85), `symmetrized=True` 19.0 / 3.42(20.4; 전치와 합치기); 밀집 2000²의 `laplacian` 1.50 / 0.61(1.37), `normed` 1.90 / 0.58(3.10). 밀집의 비정규화는 1스레드에서 scipy와 비슷하다(읽기 두 번과 쓰기 한 번; 열 합과 부호 바꾸기를 한 번에 하면 줄마다 흩어진 쓰기라 3.2 ms로 느려졌다). 들어오는 차수의 병렬과 밀집의 1스레드는 GR6 마일스톤에서 다시 봤다(GR64: 밀집 1스레드는 한 번의 훑기로 바꿨고, 들어오는 차수의 병렬 전치는 느려 넣지 않았다).
- GR5 마일스톤에서 잰 sd(ms, 1스레드 / 10스레드, 같은 때의 networkx 3.7; 부하 평균 1.3–1.5; 구조 넷과 PageRank의 n = 10⁵은 가장 좋은 값 다섯–일곱 번, 나머지는 `--smoke`의 한 번): 대칭 n = 10⁴, d = 5에서 `articulation_points` 0.75(networkx 22.5), `bridges` 0.84(130), `biconnected_components` 1.33(29.4), `core_number` 0.58(19.1), 대칭 n = 10⁴, d = 1(단절점과 다리가 많다)에서 0.37(8.06), 0.79(29.2), 0.54(10.3), 0.22(7.4); 대칭 n = 10⁵, d = 10에서 14.3, 14.3, 23.1, 10.0(같은 그래프의 sd `connected_components` 3.0; 구조 넷은 직렬); `pagerank` n = 10⁴, d = 5 1.85(networkx의 tol 10⁻⁶ 16.0), `tol=1e-6` 1.01, n = 10⁵, d = 10 26.1 / 6.9(25걸음; 준비 5.1 / 3.8, 걸음 하나 0.92 / 0.135, 같은 크기 sd.sparse `A @ x` 0.62 / 0.092); 무게 없는 `betweenness_centrality` n = 1000, d = 5 46.4 / 6.6(networkx 821), 무게 있음 74.1 / 10.5(2000), 무향 77.2 / 10.9(1080), n = 10⁴, d = 5 4960 / 593.
- GR6 마일스톤에서 잰 sd(ms, 1스레드 / 10스레드; 부하 평균 1.4–5.1이라 바꾼 것마다 고치기 전의 빌드(사본)와 번갈아 셋–다섯 번 재어 가장 좋은 값을 견줬다; 전체 사례는 `bench.speed run graph --rounds 1`로 고치기 전과 뒤를 한 번씩 쟀다). 결과의 비트는 바뀌지 않았다(골든 1,717케이스 가운데 바뀐 것 0, 1·3·10스레드가 같다).

  | 일 | 고치기 전 | 고친 뒤 | 비 | scipy(1스레드) | 무엇을 |
  |---|---:|---:|---:|---:|---|
  | `floyd_warshall` 밀집 n = 1000 | 92.2 / 15.3 | 77.6 / 10.7 | 1.19 / 1.43 | 377–383 | 블록마다 연속한 작업 사본(GR60) |
  | 무향 `breadth_first_order` 대칭 n = 10⁵, d = 10 | 7.67 / 6.05 | 5.77 / 5.73 | 1.33 / 1.06 | 18.6 | 전치 없는 대칭 확인(GR61) |
  | `articulation_points` / `bridges` 대칭 n = 10⁴, d = 5 | 0.457 / 0.463 | 0.412 / 0.411 | 1.11 / 1.13 | networkx 16.5 / 106 | 같은 것(GR61) |
  | `biconnected_components` 대칭 n = 10⁵, d = 10 | 28.2 | 24.7 | 1.14 | (`connected_components` 18.2–24.8) | 같은 것(GR61) |
  | `depth_first_order` / `depth_first_tree` n = 10⁵, d = 10 | 2.89 / 3.59 | 2.74 / 3.37 | 1.05 / 1.07 | 3.26–3.5 / 9.4 | 비트 표시, 줄 끝을 쥔 스택(GR68) |
  | `min_weight_full_bipartite_matching` 밀집 1000²의 `csr_array` | 11.7 | 7.86 | 1.48 | 11.5–11.7 | 밀집 훑기(GR62) |
  | `pagerank` n = 10⁵, d = 10 | 29.0 / 7.50 | 28.5 / 5.21 | 1.02 / 1.44 | (`A @ x` 0.93–0.99) | 준비의 병렬(GR63) |
  | `laplacian` `symmetrized=True` n = 10⁵, d = 10 | 19.5 / 3.41 | 13.5 / 2.52 | 1.44 / 1.35 | 21.1–22.3 | 합치기 한 번(GR64) |
  | `laplacian` 밀집 2000² | 1.70 / 0.52 | 1.41 / 0.52 | 1.21 / 1.0 | 1.40–1.45 | 1스레드의 한 번 훑기(GR64) |

  그 밖의 사례는 고치기 전과 같다(잡음 안). 같은 때의 1스레드 비(sd가 빠른 배수; 고친 뒤의 전체 측정): `connected_components` weak 3.8, strong 1.2, 무향 3.7, `breadth_first_order` 1.3, `depth_first_order` 1.2, 무향 `breadth_first_order` 1.4, 무향 `depth_first_order` 0.95–1.0, `dijkstra` 원점 1개 2.4, 여러 원점 2.3, 모든 쌍 2.8–2.9, `yen` 1.7, `minimum_spanning_tree` 7.0, `maximum_flow` dinic 2.8·edmonds_karp 1.1, `maximum_bipartite_matching` 1.5, `linear_sum_assignment` 2.9, 희소 `min_weight_full_bipartite_matching` 1.4, `reverse_cuthill_mckee` 3.0·1.5, `structural_rank` 1.5, `laplacian` 1.7–2.0, 밀집 `laplacian` 1.0, `csgraph_from_dense`·`csgraph_to_dense`·밀집 입력의 `connected_components` 9–15.
- **목표**(GR6 마일스톤에서 확인했다)
  - 모든 함수가 1스레드에서 scipy 이상이다. 직렬 알고리즘은 같은 알고리즘에 sd의 이점(표준 꼴 CSR이라 변환이 없음, int64 색인 그대로, 연속한 배열)만 있으므로 같거나 조금 빠른 것이 목표다. 큰 차이는 검증과 변환에서 난다(scipy는 모든 호출에서 CSR로 바꾸고 float64로 바꾼다). — 확인: 무향 `depth_first_order`(비대칭 입력; 두 줄을 읽는 메모리 지연이 시간의 대부분이다, GR68)와 밀집 `laplacian`(읽기 한 번, 쓰기 한 번; GR64)이 scipy와 같은 수준(0.95–1.0배)이고 나머지는 모두 scipy보다 빠르다. 작은 그래프(n = 4)의 호출 하나는 1.0–3.9 µs로 scipy(1.4–48 µs)보다 짧다.
  - 여러 원점(`indices`, 모든 쌍, auto의 밀집 Dijkstra, `johnson`)과 매개 중심성: 10스레드에서 스레드 수에 가깝게(7배 이상). — 확인: 원점 100개 7.5배, 모든 쌍 8.6–8.9배, auto의 밀집 Dijkstra 8.1–8.5배, 매개 중심성 8.1–8.5배.
  - `bellman_ford`: 큐 기반 이완으로 scipy보다 크게 빠르다(같은 고정점; 최악은 같은 O(nE)).
  - FW(블록 Floyd–Warshall; GR25): 1스레드에서 scipy의 `floyd_warshall`의 4배 이상(GR0 마일스톤의 시제품 4.5배, n = 1024), 10스레드에서 그 5배 이상. auto의 밀집 Dijkstra(GR26)는 1스레드에서 scipy의 `floyd_warshall` 이상. — 확인: 1스레드 4.9배(GR4 마일스톤까지 3.98–4.0배), 10스레드는 1스레드의 7.3배, auto의 밀집 Dijkstra는 scipy의 2.1배.
  - PageRank: 반복 하나가 sd.sparse의 같은 크기 SpMV(1스레드·10스레드)의 1.5배 안. — 확인: 1.47배(GR57; 걸음은 바꾸지 않았다).
  - scipy에 없는 다섯(단절점, 다리, 이중 연결 요소, 코어 수, 위상 정렬)은 O(n + E)의 한 번 훑기가 목표이고 같은 크기의 `connected_components`(약한 연결)와 networkx 3.7(GR24; Python 구현이라 속도의 기준은 아니다)과 견준다. — 확인: 대칭 n = 10⁵, d = 10에서 구조 넷의 시간이 같은 그래프의 `connected_components`의 3.3–7.7배이고(직렬 깊이 우선 하나와 통 정렬, GR59; GR5 마일스톤), 1스레드에서 networkx보다 구조 넷 19–155배, `topological_sort` 77배(GR1 마일스톤), `pagerank` 8.6배(networkx는 tol 10⁻⁶), 매개 중심성 14–27배 빠르다(GR5 마일스톤).
- **나중에 볼 것**: 단일 원점의 Δ-stepping, 너비 우선의 방향 바꾸기(위로 훑기; 결과의 트리가 달라지므로 순회가 아닌 거리 계산에만), FW의 의존 블록(대각·행·열 블록; n = 1024에서 시간의 약 16%)의 레지스터 타일. Borůvka와 Kruskal의 갈림점은 GR6 마일스톤에서 저장 원소 10⁷개까지 Kruskal로 정했다(GR65).

---

## 6. 층과 파일
- **층**: `graph/`는 `sparse/`(`format.hpp`의 표준 꼴 CSR 읽기와 전치의 계수 정렬), `array/`, `base/`(`pairwise.hpp`), `parallel/` 위에 있다. 다른 영역은 `graph/`를 모른다. Python 층의 `stardust.graph`는 `stardust.sparse`(입력, `csr_array` 결과, `aslinearoperator`)를 쓴다.
- `bench/areas.py`에 `graph`가 있다(`spatial` 뒤, `interpolate`·`fft`·`plot` 앞; plot은 맨 위 층이다). ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표에 한 줄씩 있다.
- `csrc/graph/`
  - `graph.hpp`: CSR 보기(`View`), 배열 인자의 확인(`view_arg`), sparse의 전치를 감싼 `Transposed`·`transpose`, 대칭 확인(`symmetric`; GR61), 무향 무게(`undirected_weight`; GR29), 표식 상수 `kNoNode`(−9999), 병렬 기준값 `kParallelWork`(2¹⁵)
  - `input.cpp`: `_dense_csr`(밀집 → CSR, `csgraph_from_dense`의 규칙과 흐름의 int64 값도), `_to_dense`(`csgraph_to_dense`), `view_arg`·`transpose`·`symmetric`의 구현
  - `traversal.cpp`: `_traverse`(너비·깊이 우선의 순서와 트리, 무향의 두 줄 읽기; GR32), `_components`(약한 연결의 직렬·병렬 합집합–찾기(GR31), 강한 연결의 반복형 Tarjan), `_toposort`(층을 이룬 비트맵의 Kahn)
  - `paths.hpp`: 길 함수가 같이 쓰는 것: 무향 읽기(`reading`, 합친 CSR `undirected`), 변 찾기(`edge_weight`), 닿는 노드(`reach`), 음수 순환의 공통 검사(`negative_cycle`; GR35), 탐색의 작업 공간(`Work`의 4진 색인 힙; GR39), 단일 탐색(`dijkstra`, `bfs`, 큐 `bellman_ford`, 퍼텐셜 `johnson`), 선행 노드 규칙(`predecessors`)
  - `paths.cpp`: `paths.hpp`의 구현과 `_scan`(무게 검사), `_paths`(원점 태스크, 힙 없는 밀집 Dijkstra; GR40, GR26), `_floyd_warshall`(블록마다 연속한 배치의 블록 FW와 선행 노드 커널; GR25, GR42, GR60), `_construct`, `_reconstruct`
  - `yen.cpp`: `_yen`(갈래 탐색, 후보의 전순서; GR41)
  - `tree.cpp`: `_mst`(안정 기수 정렬과 Kruskal; GR43)
  - `flow.cpp`: `_max_flow`(잔여 CSR, Dinic, Edmonds–Karp; GR44)
  - `matching.cpp`: `_bipartite_matching`(탐욕 시작과 Hopcroft–Karp; `maximum_bipartite_matching`과 `structural_rank`가 쓴다; GR47), `_assign_dense`·`_assign_sparse`(두 배정의 공용 풀이 `Solver`: LAPJV의 시작과 최단 증가 경로, 밀집 훑기와 희소 힙; GR45)
  - `ordering.cpp`: `_rcm`(역 Cuthill–McKee; GR52), `_laplacian_csr`·`_laplacian_dense`(라플라시안의 차수와 원소, 대칭 합; GR50, GR53, GR54, GR64)
  - `structure.cpp`: `_biconnected`(단절점, 다리, 이중 연결 요소; 한 번의 깊이 우선), `_core_number`(Batagelj–Zaversnik; GR59)
  - `centrality.cpp`: `_pagerank`(GR56, GR57, GR63), `_betweenness`(GR58)
  - `graph_mod.hpp`: 파일마다의 `bind_*`
- 바인딩 `bind/graph_bind.cpp`: `_core.graph` 하위 모듈.
- `src/stardust/graph/` (하위 패키지 없음): `__init__.py`(공개 이름), `_common.py`(입력을 CSR로(`graph`, `biadjacency`, `capacities`, `dense`, `contiguous`), dtype·모양·무게·노드 인자의 검사와 메시지, `NegativeCycleError`, `count`, `room`), `_traversal.py`, `_convert.py`(`csgraph_from_dense`, `csgraph_to_dense`), `_paths.py`, `_tree.py`, `_flow.py`(`MaximumFlowResult`), `_matching.py`, `_ordering.py`(RCM, `structural_rank`, `laplacian`), `_structure.py`, `_centrality.py`. namedtuple 결과는 `stardust/_results.py`에 있다.

---

## 7. 참값과 검증
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다. 긴 참값 생성은 프로세스 10개까지, 메모리를 지켜보며(RSS가 평평한지) 돈다.

### 7.1 시험 그래프 (`tests/graph/graphs.py`)
공식과 고정 시드로 만든 61개다(내려받은 것 없음). 테스트, 참값 도구, 탐침, 측정이 모두 id로 여기서 그래프를 가져온다. 시간 재기의 무작위 그래프는 `random_graph`다.
- **모양**: 빈 그래프(n = 0, 1), 고립 노드, 경로, 순환, 별, 완전 그래프, 격자(2·3차원; 동점이 아주 많다), 사다리(경로 수 2^(n/2)), 이분 그래프(직사각), DAG, 무작위 유향·무향(Erdős–Rényi, 차수 분포가 긴 꼬리), 여러 요소, 자기 고리, 한 방향만의 변(무향 규칙), 두 방향 무게가 다른 변.
- **무게**: 정수(작은 범위라 동점이 많음; 2⁵³ 둘레의 큰 정수), 2진 유리수, 무작위 float(10⁻⁸–10⁸의 넓은 범위, 흡수가 생기는 무게), 저장된 0(무게 0의 변과 0 무게 순환), 음수(순환 없음, 닿는 음수 순환, 닿지 않는 음수 순환, 정확한 합 0의 순환), bool.
- **경계**: 밀집의 0·NaN·±inf, 희소의 저장된 NaN·±inf(에러), 범위 밖 노드, `limit`의 경계 값과 그 앞뒤 double, 정확히 같은 거리의 여러 경로.
- **흐름과 매칭**: 정수 용량(int64 경계 근처), 완전 매칭이 없는 경우, 동점이 많은 비용, 직사각.

### 7.2 참값 (`tools/graph_ref.py` → `tests/graph/ref/`, `tests/reffile.py`의 npz)
여섯 묶음, 536점이다(그래프마다 `digest`를 가진다). 2프로세스로 돌고 워커는 8태스크마다 바꾼다.
- `paths`(315점, 그래프 39개, n ≤ 120): 정확한 유리수(Fraction) 큐 Bellman–Ford의 참 거리와 §3.2의 규칙을 정확히 계산한 선행 노드(경로의 합이 double에서 정확한 그래프에서 sd의 선행 노드와 같아야 한다), 그 경로의 변 수와 Σ|w|(계약의 H와 크기), 닿는 음수 순환의 표시. 읽기는 유향·무게 없음·무향·무향 무게 없음 넷이다.
- `yen`(55점, 그래프 14개, n ≤ 30): 고리 없는 경로 전부의 나열을 (정확한 길이, 변 수, 노드 열)로 정렬한 앞 20개와 경로 수.
- `assignment`(14점, 8 × 8까지): 작은 쪽의 모든 배정을 나열한 정확한 최적 비용(최소·최대), 최적 배정의 수, 사전 순서로 가장 작은 최적 배정, 실현 가능성.
- `pagerank`(25점, 그래프 11개, n ≤ 120): Arb `arb_mat.solve`의 참값(원소마다 정해질 때까지 정밀도를 늘린다).
- `betweenness`(76점, 그래프 13개, n ≤ 120): 정확한 유리수 Brandes(경로 수는 Python 정수, networkx의 비율과 끝점).
- `laplacian`(51점, 그래프 9개, n ≤ 100): Arb의 차수와 정규화 원소(√는 Arb).
- 구조 함수는 참값 파일이 아니라 테스트 안의 Python 정의 구현(`tests/graph/definitions.py`)과 견준다(빠르고 작다). 큰 배정 문제의 참 최적은 같은 파일의 정확한 유리수 풀이 `assignment_exact`다.

### 7.3 테스트 (`tests/graph/`)
- 파일: `test_api_surface.py`(API 문서의 시그니처와 영역의 호출 규칙), `test_input.py`(변환 규칙, dtype, 밀집의 변 규칙, 에러와 메시지, 노드 인자, 옵션, 호출 모양), `test_traversal.py`, `test_components.py`, `test_toposort.py`, `test_convert.py`, `test_paths.py`, `test_paths_reference.py`, `test_paths_methods.py`(D, BF, J, auto의 바이트가 같음, 정수 무게의 정확한 거리와 선행 노드, 일반 무게의 계약, 음수 순환; hypothesis), `test_yen.py`, `test_mst.py`, `test_flow.py`, `test_matching.py`, `test_ordering.py`, `test_laplacian.py`, `test_structure.py`, `test_centrality.py`, `test_layouts.py`(밀집 입력의 메모리 꼴; GR48), `test_threads.py`(1, 3, 10스레드의 같은 바이트: 병렬 길과 GIL을 놓는 크기), `test_reference_data.py`(참값과 기록이 시험 그래프와 맞고 스스로의 검사를 통과하는지). 정의 구현은 `definitions.py`, 시험 그래프의 입력 꼴(희소 넷과 밀집)은 `forms.py`, 참값과 계약의 판정은 `refdata.py`다.
- hypothesis는 tests/README.md의 프로필을 쓴다. `test_paths_methods.py`의 생성기는 n(1–13), 밀도, 무게의 종류(정수·2진 유리수·float·넓은 범위·0·음수), 대칭 여부, 유향·무향, 무게 없음을 고른다.
- scipy·networkx와는 그 자리에서 견준다: 동점이 없는 그래프에서 같은 결과(거리는 계약 안, 선행 노드·순서·레이블은 같음), 동점이 있으면 같은 분할·같은 거리·같은 흐름 값·같은 매칭 크기.

### 7.4 scipy·networkx의 기록 (`tools/graph_probe.py` → `tests/graph/data/`)
- 설계 때의 탐침(입력 규칙, 저장된 0·NaN·inf, 음수 무게와 순환, auto의 규칙, 동점, `limit`·`indices`·`min_only`, 순회 순서, 레이블, MST의 경계, 흐름의 int32 자르기, 매칭의 0, 라플라시안의 꼴, `yen`, 변환 도구, 시간)을 도구 하나로 옮겼다(GR0 마일스톤; FW 길의 두 시제품은 `tools/graph_probe_fw.cpp`). 내부 에러를 삼키는 호출(`breadth_first_order(G, 5)`)도 기록한다.
- 기록 `scipy.json`(scipy 1.18.1): 문서의 `[scipy와 다름]` 하나하나(`rule/…`)와 시험 그래프의 결과(`graph/…`). `networkx.json`(networkx 3.7; GR24): scipy에 없는 일곱의 결과(사전 최소 위상 순서, 단절점, 다리, 이중 연결 요소의 변, 코어 수, PageRank, 매개 중심성)와 가장자리의 동작(순환과 자기 고리의 위상 정렬, 자기 고리의 `core_number`는 에러, 수렴하지 못한 PageRank, 작은 n의 매개 중심성). 모든 기록은 그 그래프의 `digest`(tests/graph/graphs.py)를 가진다.

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/graph.py`, 표 `bench/graph/accuracy.md`): 스윕 다섯이다. `paths`(일반 무게의 거리 계약 단위, 정의와 FW; 비교 패키지 scipy의 D(음수 무게면 J)·FW), `assignment`(일반 비용의 계약 단위: 참값의 실수 두 점과 정확한 유리수 풀이와 견준 무작위·거의 동점 문제, 14영역; 비교 패키지 scipy `linear_sum_assignment`·`min_weight_full_bipartite_matching`), `laplacian`(참값의 float 그래프, 정규화·비정규화마다 한 영역; 비교 패키지 scipy), `pagerank`(pagerank.npz의 모든 점, 그래프마다 한 영역, tol 10⁻¹²와 바닥 아래 10⁻¹⁵; 비교 패키지 networkx), `betweenness`(betweenness.npz의 모든 점, 그래프와 무게마다 한 영역, 노드값마다 한 점; 비교 패키지 networkx). 기준선의 `paths`·`assignment`는 비교 패키지 없이 기록한 실행이라 표의 scipy 열이 비어 있다(§4의 scipy 값은 마일스톤 때의 측정이다).
- **속도** (`bench/speed/cases/graph.py`, 사례 57개): 사례마다 마일스톤과 필요한 이름을 적어 두고 그 이름이 공개되면 사례가 된다(sparse·spatial의 사례 파일과 같은 틀). §5의 크기와 1·10스레드. 비교 대상은 scipy 1.18.1(`linear_sum_assignment` 포함)과, scipy에 없는 일곱은 networkx 3.7이고, `--smoke`에서 결과(거리, 레이블, 흐름 값, 희소 결과는 표준 꼴 CSR의 세 배열)를 견준다.
- **골든** (`tests/golden/cases_graph.py`, 기록 `tests/golden/data/graph.json`): 모든 공개 함수 × 시험 그래프의 대표, 방법 다섯, 틀린 호출, 병렬 길을 탈 큰 케이스(워커 하나의 메모리가 1 GB를 넘지 않게). 1·3·10스레드의 결과가 모두 같아야 한다. 케이스는 1,717개이고, 기록에는 GR4 마일스톤까지의 1,490개가 있다(GR5 마일스톤의 227개는 1·3·10스레드의 같은 결과를 임시 기록으로 확인했고, 기록은 메인 세션이 한다).

---

## 8. 마일스톤
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| GR0 준비 — **완료·승인(위임, 2026-11-23)** | API 문서 초안(`docs/api/graph.md`: 모든 공개 이름의 시그니처, scipy·networkx → sd 대응표, §9의 답), 영역 등록(`bench/areas.py`, ARCHITECTURE, API.md의 색인), 시험 그래프(§7.1), 참값 도구(§7.2), 탐침 도구와 scipy 기록(§7.4), FW 길의 두 안(원점마다 밀집 Dijkstra 대 블록 min-plus FW)의 측정(§9의 2번), 속도 사례의 틀 | 문서와 도구가 커밋되고 탐침 기록이 있다. §9의 답이 §1.1의 결정으로 옮겨졌다 |
| GR1 순회와 연결 요소 — **완료·승인(위임, 2026-11-23)** | 입력 변환(§2.1, §3.1), `csgraph_from_dense csgraph_to_dense`, `breadth_first_order depth_first_order breadth_first_tree depth_first_tree`, `connected_components`, `topological_sort` | 정의 구현과 정확히 같음, scipy와의 차이가 문서대로, 스레드마다 같은 바이트, 속도 사례 |
| GR2 최단 경로 — **완료·승인(위임, 2026-11-23)** | `shortest_path dijkstra bellman_ford johnson floyd_warshall yen construct_dist_matrix reconstruct_path NegativeCycleError`, §3.2의 고정점과 선행 노드 규칙 | D, BF, J, auto가 바이트까지 같음, FW는 계약 안, 정수 무게의 정확한 참값과 같음, 일반 무게의 계약, 음수 순환 규칙, scipy와의 속도 사례 |
| GR3 신장 트리·흐름·매칭 — **완료·승인(위임, 2026-11-23)** | `minimum_spanning_tree maximum_flow maximum_bipartite_matching min_weight_full_bipartite_matching`, `linear_sum_assignment`(§9의 1번) | 유일한 MST, 흐름 값과 최소 절단이 같음, 정수 비용의 정확한 최적, 일반 비용의 계약, 속도 사례 |
| GR4 순서·구조적 랭크·라플라시안 — **완료·승인(위임, 2026-11-23)** | `reverse_cuthill_mckee structural_rank laplacian`(`form='lo'` 포함) | scipy와 같은 순열(동점 규칙 확인), Arb 계약, 속도 사례 |
| GR5 더한 알고리즘 — **완료·승인(위임, 2026-11-23)** | `articulation_points bridges biconnected_components core_number pagerank betweenness_centrality` | 정의 구현과 정확히 같음(구조 넷), PageRank·매개 중심성의 Arb·정확한 유리수 계약, 스레드마다 같은 바이트, 속도 사례 |
| GR6 성능과 일관성 점검 — **완료·승인(위임, 2026-11-23)** | 모든 공개 함수를 크기마다 scipy·networkx와 견주어 고치기, 갈래의 문턱(auto, 병렬 합집합–찾기, Borůvka, 원점 태스크), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다(GR60–GR68, 기준선 승격은 메인 세션의 전체 측정 때) |
| GR7 문서 — **완료·승인(위임, 2026-11-23)** | `docs/api/graph.md`의 확정, 이 문서를 구현대로 고치기(잰 값, 정한 상수, 파일) | 문서가 코드와 맞고 `test_api_surface`가 통과한다 |

진행 원칙은 다른 영역과 같았다: 마일스톤마다 시작 전에 §9의 질문을 정하고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달했다. GR1 마일스톤이 입력 변환을 만들므로 맨 앞이었다. GR2–GR5 마일스톤은 GR1 마일스톤에만 기대어 서로 독립이었다(GR5 마일스톤의 매개 중심성은 GR2 마일스톤의 Dijkstra와 무향 읽기를 쓰므로 그 뒤). 기존 영역의 결과는 비트까지 같아야 한다(이 영역은 다른 영역을 고치지 않는다).

**나중 후보** (첫 판 밖): 군집 찾기(Louvain, Leiden, 레이블 전파), 무작위 그래프, 다른 중심성(근접, 고유벡터, Katz, HITS), 표본 매개 중심성(`k`, `rng`), 최소 절단의 노드 집합, 단일 원점의 Δ-stepping, 너비 우선의 방향 바꾸기.

---

## 9. 마일스톤을 시작하기 전에 물을 것
모두 (a)로 정했다(§1.1 GR4–GR23은 추천안, 사용자 위임 2026-11-23; 18번은 GR24, 사용자 결정 2026-11-23: networkx 3.7만 설치).

추천을 첫째에 둔다.

**GR0 마일스톤 전**
1. **`linear_sum_assignment`의 자리**. (a) `sd.graph`에 둔다. 추천: `maximum_bipartite_matching`·`min_weight_full_bipartite_matching`과 같은 배정 문제이고 같은 C++ 고리(§3.5)를 쓴다. 이 영역이 기존 영역(optimize)을 고치지 않는다. sd.optimize는 선형 계획과 조합 최적화를 범위에서 뺐다. scipy 코드는 가져오기 경로가 `scipy.optimize` → `stardust.graph`로 바뀐다(대응표에 적는다). (b) `sd.optimize`에 둔다: scipy와 같은 경로지만 optimize의 범위와 API 문서를 넓히고, C++는 optimize가 graph를 부르는 의존이 생긴다. (c) 둘 다: 별칭이라 안 된다.
2. **최단 경로의 결과를 방법과 무관하게 할까** (§3.2). (a) 모든 방법이 한 정의(차례 합의 가장 작은 고정점과 선행 노드 규칙)의 같은 비트를 낸다. `'FW'`는 이름을 두되 밀집 행렬 위의 원점마다 힙 없는 Dijkstra로 구현한다. GR0 마일스톤에서 재서 블록 min-plus FW보다 1.3배 넘게 느리면 (b)로 간다(GR0 마일스톤에서 1.8–2.5배라 (b)로 바꿨다: GR25–GR27). 추천: 갈래가 결과를 바꾸지 않는다는 원칙(spatial과 같다)이고 auto의 문턱이 결과를 흔들지 않는다. scipy의 D·BF·J도 이미 같은 거리다. (b) `'FW'`만 고전 Floyd–Warshall의 반올림(가운데에서 잇는 합)을 쓴다: 거리는 계약 안이지만 auto가 밀도 n²/4를 넘는 그래프에서 끝자리가 다른 답을 낸다. 선행 노드는 그 거리에서 같은 규칙으로 만든다.
3. **선행 노드의 동점 규칙** (§3.2). (a) 꼭 맞는 변 T 위에서 가장 적은 변 수, 그다음 가장 작은 번호. 추천: 무게 0의 변과 흡수가 있어도 트리이고, 방법과 무관하며, 정수 무게에서 정확한 참값으로 시험할 수 있다. 비용은 원점마다 O(n + E)의 한 번 훑기다. (b) 꼭 맞는 변 가운데 가장 작은 번호만: 무게 0의 변이 두 방향에 있으면 선행 노드가 순환할 수 있어 따로 막아야 한다. (c) 알고리즘의 처리 순서(scipy처럼): 방법마다 다르다.
4. **받는 입력** (§2.1). (a) `sd.sparse` 배열과 밀집(`sd.ndarray`, numpy, 중첩 시퀀스)을 받고, scipy의 희소 객체는 `TypeError`로 `sd.sparse.csr_array(A)`를 짚는다. 추천: scipy 객체를 바꾸는 곳은 희소 생성자 하나(sparse.md §2.5; int32 색인을 넓히고 언제나 복사한다)이고, 그래프 함수가 같은 변환을 몰래 하면 큰 입력에서 복사 비용이 숨는다. (b) scipy의 희소 객체도 바로 받는다(안에서 `csr_array`로 바꾼다): 옮겨 오는 코드에 편하다.
5. **유한하지 않은 무게** (§2.1). (a) 희소 입력의 저장된 NaN·±inf는 무게를 쓰는 함수에서 `ValueError`, 밀집은 scipy의 변 규칙(0, NaN, ±inf는 변 없음), 구조 함수는 패턴만 본다. 추천: scipy는 같은 NaN을 함수마다 다르게 다룬다(최단 경로는 건너뛰고 MST는 넣는다). (b) 저장된 +inf는 "있지만 지날 수 없는 변"으로 받는다(최단 경로에서는 없는 변과 같고 MST에서는 scipy처럼 트리에 들어간다): 함수마다 뜻이 달라진다.
6. **음수 무게의 `dijkstra`**. (a) `ValueError`(`dijkstra(): negative edge weights are not allowed; use bellman_ford or johnson`). 추천: scipy는 경고와 함께 틀릴 수 있는 값을 내고, sd는 경고를 내지 않는다. (b) 받아서 큐 기반으로 바르게 계산한다: `dijkstra`라는 이름과 맞지 않고 `bellman_ford`와 같은 함수가 된다.
7. **색인 결과의 dtype과 표식**. (a) 모든 색인 결과는 int64, 표식은 scipy대로 −9999(선행 노드, `sources`)와 −1(매칭)이다. 추천: sd의 정수 dtype은 int64 하나이고 희소의 색인도 int64다(sparse.md §2.2). 표식은 사용자 코드가 비교하는 값이라 그대로 둔다. (b) scipy처럼 int32: sd에 없는 dtype이다.
8. **결과의 꼴**. (a) scipy가 튜플을 돌려주는 곳은 같은 순서의 namedtuple(§2.8), `MaximumFlowResult`는 클래스. 추천: 풀기가 scipy와 같고 이름으로도 읽힌다(DESIGN §9.7). (b) scipy처럼 보통 튜플.
9. **연결 요소의 번호**. (a) 가장 작은 노드의 순서. 추천: 분할의 함수라 알고리즘·병렬과 무관하다. 약한 연결과 무향은 scipy와 같고 강한 연결만 다르다. (b) scipy의 강한 연결 번호(Pearce 알고리즘의 순서)를 따라 한다: 알고리즘에 묶여 병렬화할 수 없다.
10. **무향 순회의 이웃 순서**. (a) 두 방향 이웃을 합친 오름차순. 추천: 순서가 무향 그래프의 함수라 G와 Gᵀ에서 같다. (b) scipy처럼 나가는 이웃 먼저, 들어오는 이웃 나중: 같은 무향 그래프가 저장 방향에 따라 다른 순서가 된다.
11. **scipy의 옵션 가운데 뺄 것**. (a) `overwrite`(최단 경로·MST), `laplacian`의 `copy`·`form='function'`, 가린 배열 도구 셋을 뺀다. 이름은 scipy 그대로(`csgraph_from_dense` 등) 둔다. 추천: sd는 입력을 고치지 않고(희소 생성자와 같은 원칙), 가린 배열이 없으며, `form='lo'`가 `form='function'`의 일을 한다. 이름을 바꾸면 옮겨 오는 코드만 번거롭다. (b) `overwrite`를 받되 무시: 틀린 기대를 남긴다. (c) 변환 도구를 `from_dense`·`to_dense`로 줄인다: 이름공간 안에서는 짧지만 scipy와 달라진다.
12. **흐름의 용량**. (a) bool·int64만(float64는 `TypeError`), 음수는 `ValueError`, 원점 용량 합이 int64를 넘으면 `OverflowError`. 추천: 정수 알고리즘의 정확성(흐름이 정확한 정수)을 dtype으로 보인다. (b) 정수 값의 float64도 받는다(2.0): 2⁵³ 위에서 정확성이 깨진다.
13. **흐름·매칭·배정의 동점**. (a) 고정된 훑기 순서(행과 이웃 오름차순)의 결정적인 결과를 돌려주고, 약속은 최적성(정수에서 정확)과 재현성이다. 어떤 최적해인지는 골든으로 지킨다. 추천: 정준 최적해를 고르는 데는 추가 계산이 든다. (b) 정수 비용·용량에서 정준해(배정은 `col_ind`가 사전 순서로 가장 작은 최적해; 흐름은 정의하기 어렵다)를 돌려준다: 배정은 최적 쌍대의 같은 비용 부분 그래프에서 사전 최소 완전 매칭을 다시 찾는 비용(O(n³) 가량)이 든다.
14. **최소 신장 트리의 동점과 자리**. (a) 키 (w, min(i, j), max(i, j))로 유일한 트리, 고른 변은 그 무게를 준 저장 자리(같으면 위 삼각), 밀집에서 작은 원소를 버리지 않는다. 추천. (b) scipy의 자리 규칙을 탐침으로 정확히 따라 한다: 동점에서 scipy가 고르는 트리가 정렬의 안정성에 기대어 정의가 없다.
15. **더한 알고리즘의 이름과 꼴**. (a) `topological_sort`는 사전 최소 위상 순서, 순환은 `ValueError`; `articulation_points`는 오름차순 배열, `bridges`는 사전 순서의 (k, 2); `biconnected_components`는 `BiconnectedComponentsResult(n_components, edges, labels)`; k-코어는 `core_number` 하나(k-코어는 `core_number >= k`). 추천: 모두 데이터로만 정해지는 하나의 답이고, 배열로 돌려준다. (b) networkx처럼 `k_core(G, k)`(노드 배열)와 노드 집합 꼴의 `biconnected_components`(CSR 꼴의 `indptr`·`nodes`): 노드 집합이 단절점에서 겹쳐 변 꼴보다 다루기 어렵다.
16. **PageRank의 멈춤과 기본값** (§3.7). (a) 증명된 ℓ₁ 한계 α/(1 − α)·‖x_{k+1} − x_k‖₁ ≤ `tol`로 멈추고 기본 `tol=1e-12`, `max_iter` 없음(반복 수는 α와 `tol`로 정해진다). 추천: 결과가 해와 얼마나 가까운지를 약속할 수 있고, 10⁻⁶ 대신 10⁻¹²는 반복이 약 1.8배(α = 0.85에서 101 대 186)다. (b) networkx의 뜻(`tol=1e-6`, 연속 반복의 차 < n·tol, `max_iter=100`, 실패하면 예외): networkx와 같은 값이 나오지만 그 값이 해에서 얼마나 먼지 약속이 없다. (c) 희소 직접 풀이(`spsolve`): 반복이 없지만 큰 그래프에서 채움이 크다.
17. **매개 중심성의 범위**. (a) 정확한 Brandes만(`directed`, `unweighted`, `normalized`, `endpoints`), 표본 어림(`k`, `rng`)은 나중. 추천: 정확한 값은 데이터로만 정해지고, 원점이 병렬이라 10스레드에서 10⁵ 노드 희소 그래프도 다룰 수 있다. (b) networkx처럼 `k`와 `rng`(sd.random의 `Generator` 필수)도 넣는다: 결과의 약속이 둘로 갈린다.
18. **비교 패키지**. networkx와 igraph가 없어 scipy에 없는 일곱 알고리즘(위상 정렬, 단절점, 다리, 이중 연결 요소, 코어 수, PageRank, 매개 중심성)의 뜻을 공개 문서로만 맞췄다. (a) networkx(순수 Python 꾸러미, 작다)를 `.venv`에 설치해 뜻과 결과를 비교 패키지로 견주고, igraph는 설치하지 않는다. 추천: 뜻(정규화, 매달린 노드, 동점)을 실제 출력과 견줄 수 있다. 속도 비교로는 쓸모가 적다(Python 구현). (b) networkx와 python-igraph(C 코어, 속도의 기준) 둘 다: igraph는 컴파일된 꾸러미라 더 크다. (c) 설치하지 않는다: 정확한 유리수·Arb 참값으로만 확인하고 뜻의 일치는 문서로만 본다.

**GR1 마일스톤 전**
19. **옵션의 자리**. (a) 데이터와 주된 인자(`i_start`, `source`, `sink`, `K`, `predecessors`)는 위치, 옵션은 키워드만. 추천: 공통 원칙이고 spatial의 `metric`·`k`와 같다. scipy 코드의 `dijkstra(G, False)`는 `TypeError`다. (b) scipy처럼 옵션도 위치로.
20. **노드 인자의 꼴**. (a) 0 이상 n 미만의 int(`indices`는 0·1차원, 중복 가능), 음수·범위 밖은 `ValueError`, bool·실수는 `TypeError`. 추천: 한 규칙이 모든 노드 인자에 쓰이고, scipy는 함수마다 다르다(`indices`의 음수는 감고 `i_start`의 음수는 `OverflowError`, 범위 밖 `i_start`는 빈 결과). (b) 음수를 배열 인덱싱처럼 감는다: `-1`이 노드 n − 1이 되는 것은 노드 번호의 뜻과 맞지 않는다.

**GR2 마일스톤 전**
21. **`limit`의 NaN**. (a) `ValueError`. 추천: 0 이상이거나 inf가 아닌 값은 모두 같은 에러다(spatial의 `distance_upper_bound`와 같다). (b) scipy처럼 모두 inf.
