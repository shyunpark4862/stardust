# 그래프 영역 설계 (sd.graph)

> 상태: **설계 초안** (2026-11-23). 큰 결정(§1.1 GR1–GR3)은 사용자가 정했다. §9의 질문은 모두 정했다: 18번(비교 패키지)은 사용자 결정(GR24: networkx만 `.venv`에 설치, 2026-11-23), 나머지는 추천안(GR4–GR23, 사용자 위임 2026-11-23). GR0 마일스톤은 완료·승인(위임, 2026-11-23)이고 그때 FW 길의 측정으로 GR5를 바꿨다(GR25–GR27). GR1 마일스톤(순회, 연결 요소, 위상 정렬, 변환 도구; 공개 이름 8개)은 완료·승인(위임, 2026-11-23)이고 그때 GR29–GR33을 정했다. GR2 마일스톤(최단 경로; 공개 이름 9개)은 완료·승인(위임, 2026-11-23)이고 그때 GR34–GR42를 정했다. GR3 마일스톤(신장 트리, 흐름, 매칭, 배정; 공개 이름 5개)은 완료·승인(위임, 2026-11-23)이고 그때 GR43–GR49를 정했다.

`sd.graph`(인접 행렬로 주는 그래프의 알고리즘: 순회, 연결 요소, 최단 경로, 신장 트리, 흐름, 매칭, 순서 매기기, 중심성)의 결정과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처의 안은 GR0 마일스톤에서 `docs/api/graph.md`로 쓴다.
- 비교 대상은 scipy 1.18.1의 `scipy.sparse.csgraph`(공개 이름 스물여섯)와 `scipy.optimize.linear_sum_assignment`다. networkx와 python-igraph는 `.venv`에 없고, R 4.6.2에는 `igraph`를 비롯한 그래프 꾸러미가 없다(설치된 것은 `Matrix`뿐이다). 설치하지 않았다. 그래서 scipy에 없는 알고리즘(위상 정렬, 단절점, 다리, 이중 연결 요소, 코어 수, PageRank, 매개 중심성)은 이 초안에서 비교 패키지로 확인하지 못했고, 그 뜻은 networkx의 공개 문서를 기준으로 적었다(§9의 18번).
- 바닥은 배열(`csrc/array/`), 희소의 표준 꼴 CSR(`csrc/sparse/format.hpp`의 `Compressed`; sparse.md §2.2–§2.3), 스레드풀(`csrc/parallel/`)이다. 트리·흐름·라플라시안 같은 행렬 결과는 `sd.sparse.csr_array`다.
- 이 문서의 잰 값은 Apple M4 Pro, macOS 26.6.2에서 본 것이다(부하 평균 1.5). 다른 작업과 기기를 나눠 썼다.
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
| `linear_sum_assignment(cost_matrix, maximize=False)` | `sd.graph.linear_sum_assignment`(안) | GR3 |

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
  - 무게를 쓰는 함수(최단 경로, `minimum_spanning_tree`, `min_weight_full_bipartite_matching`, `linear_sum_assignment`, `maximum_flow`, `pagerank`, 가중 `betweenness_centrality`)는 희소 입력의 저장된 NaN과 ±inf를 `ValueError`로 막는다(`dijkstra(): stored edge weights must be finite`). `[scipy와 다름]`: scipy는 최단 경로에서 저장된 NaN 변을 비교 실패로 조용히 건너뛰고, 같은 NaN을 `minimum_spanning_tree`는 트리에 넣으며, −inf 변의 `bellman_ford`는 음수 순환을 알리지 않고 −inf 거리를 낸다(이 초안에서 확인). 밀집 입력은 위의 규칙대로 NaN·±inf가 변 없음이라 막을 것이 없다.
  - 구조만 보는 함수(연결 요소, 순회, 위상 정렬, `reverse_cuthill_mckee`, `structural_rank`, `maximum_bipartite_matching`, 단절점, 다리, 이중 연결 요소, `core_number`, 무게 없는 `betweenness_centrality`)는 **패턴**(희소의 저장된 자리, 밀집의 변 규칙)만 읽는다. 값이 NaN이어도 변이다.
  - 예외는 `laplacian`이다: 그래프 함수가 아니라 행렬 함수라서 밀집 입력도 값 그대로(0, NaN, inf가 그 값으로) 계산한다(scipy와 같다; §2.5).
- **무향 그래프** (`directed=False`): i와 j 사이의 변은 i → j와 j → i 가운데 있는 것이고, 둘 다 있으면 작은 무게다(scipy와 같다; 이 초안에서 D, BF, J, FW가 모두 그렇게 계산함을 확인했다). NaN이 끼면 NaN, 0.0과 −0.0이면 −0.0이다(GR29). 패턴만 보는 함수는 두 방향 패턴의 합집합이다.
- **노드 인자**(`i_start`, `source`, `sink`, `indices`의 원소): 0 이상 n 미만의 int다(`operator.index`의 규칙, API.md §2). 범위 밖이면 `ValueError`, bool·실수는 `TypeError`다(§9의 20번; `[scipy와 다름]`: scipy의 `dijkstra(indices=1.5)`는 1로 자르고 `indices=-1`은 n − 1로 감고, `breadth_first_order(G, 5)`는 범위 밖에서 내부 `IndexError`를 삼키고 빈 결과를 내며(이 초안에서 확인), `i_start=-1`은 `OverflowError`다).
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
- **레이블**: 요소 번호는 **요소의 가장 작은 노드의 순서**다(노드 0의 요소가 0, 그다음 처음 나오는 노드의 요소가 1, …; §9의 9번). 약한 연결과 무향은 scipy와 같고(이 초안에서 확인), 강한 연결은 `[scipy와 다름]`: scipy의 번호는 알고리즘(Pearce)의 순서라 `[0, 1, 3, 4, 2, 5]`처럼 나온다. 분할은 같다. 결과는 `ConnectedComponentsResult(n_components, labels)`(`n_components`는 Python `int`, `labels`는 int64; scipy는 int32)이고 `return_labels=False`면 Python `int` 하나다(scipy와 같은 꼴).
- **순회의 순서**: 이웃은 **노드 번호 오름차순**으로 본다. 유향이면 행의 저장 순서(CSR 표준 꼴)이고, 무향이면 두 방향 이웃을 합친 오름차순이다(§9의 10번). 너비 우선은 큐, 깊이 우선은 재귀 깊이 우선과 같은 순서(가장 작은 미방문 이웃부터)다. 유향 그래프에서는 scipy와 같은 순서다(5 × 5 격자에서 확인). `[scipy와 다름]`: scipy의 무향 순회는 나가는 이웃을 먼저, 들어오는 이웃을 나중에 본다(이 초안에서 `[0, 3, 4, 1, 2]` 대 sd `[0, 1, 2, 3, 4]`). sd는 G와 Gᵀ(같은 무향 그래프)에서 같은 답을 낸다.
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
- **결과는 정의로 정해진다**(§3.2; §9의 2번, 3번): 원점 s에서 v까지의 거리 d̂(v)는 s에서 v로 가는 모든 경로 가운데, 경로의 무게를 **원점에서부터 차례로**(왼쪽에서 오른쪽으로, 0에서 시작해 한 변씩) 더한 float64 값이 가장 작은 것이다. 선행 노드는 §3.2의 규칙(꼭 맞는 변, 가장 적은 변 수, 가장 작은 번호)으로 정한다. **D, BF, J, auto와 `dijkstra`·`bellman_ford`·`johnson`은 같은 입력에서 거리와 선행 노드가 비트까지 같다.** 방법은 빠르기만 바꾼다. FW와 `floyd_warshall`만 블록 Floyd–Warshall의 반올림이다(GR25; 거리는 계약 안, 선행 노드는 GR27). `[scipy와 다름]`: scipy의 D·BF·J는 거리가 같지만(차례 합; 이 초안에서 무게 10⁻⁸–10⁸의 무작위 그래프로 확인) 선행 노드의 동점이 다르고(마름모 그래프에서 D·J는 2, BF·FW는 1), FW는 거리의 끝자리도 다르다(같은 그래프에서 상대 8.7×10⁻¹⁶).
- **`method`**: `'auto'`, `'D'`, `'BF'`, `'J'`, `'FW'`(scipy와 같은 이름; 그 밖은 `ValueError`, 타입이 틀리면 `TypeError`). auto는 음수 무게가 있으면 J, 없으면 D다(GR26). 음수 무게가 없고 `indices`가 없고 변(입력의 저장된 원소)이 n²/4개 이상이면 D와 같은 비트의 밀집 Dijkstra로 한다(GR36). `[scipy와 다름]`: scipy의 auto는 그 밀도에서 FW다(GR0 마일스톤의 기록: 밀도 0.26부터 FW의 비트). 문턱은 빠르기만 정하고 GR6 마일스톤에서 재서 바꿀 수 있다. `method='FW'`와 `indices`를 함께 주면 `ValueError`다(scipy와 같다).
- **결과의 꼴**(scipy와 같다): `indices=None`이면 거리 (n, n) float64, int 하나면 (n,), 1차원이면 (len, n)이다. 닿지 않는 노드는 inf, 선행 노드는 −9999다. `return_predecessors=True`면 `ShortestPathResult(dist_matrix, predecessors)`(선행 노드 int64; scipy는 int32)이고 아니면 거리 배열 하나다. `indices`는 0차원 또는 1차원이고 중복을 받는다(scipy와 같다). 2차원은 `ValueError`다(`[scipy와 다름]`: scipy는 받아 3차원 결과를 낸다).
- **`unweighted=True`**: 모든 변의 무게를 1로 본다(저장된 0의 변도 변 하나다; scipy와 같다). 거리는 변의 수이고 언제나 정확하다. 값은 읽지 않는다: 저장된 NaN·±inf와 음수 무게도 막지 않는다(GR34).
- **`limit`**(`dijkstra`만): 거리 d̂ ≤ `limit`인 노드만 찾고 나머지는 inf, −9999다(경계를 넣는다: 이 초안에서 scipy의 `limit=1`이 거리 1을 넣음을 확인). 0 이상이거나 inf여야 한다. 음수는 `ValueError`(scipy와 같다), NaN도 `ValueError`(`[scipy와 다름]`: scipy는 NaN에서 모든 노드를 inf로 낸다).
- **`min_only=True`**(`dijkstra`만): 모든 `indices`를 함께 원점으로 둔 거리 하나(`(n,)`)다. `return_predecessors=True`면 `MinOnlyResult(dist_matrix, predecessors, sources)`이고 `sources[v]`는 v의 선행 노드 사슬이 끝나는 원점, 닿지 않으면 −9999다(scipy와 같은 꼴; 동점 규칙은 §3.2). `indices=None`이면 모든 노드가 원점이다(scipy와 같다: 거리가 모두 0).
- **음수 무게**: `dijkstra`(와 `method='D'`)는 저장된 음수 무게를 `ValueError`로 막는다(§9의 6번; `[scipy와 다름]`: scipy는 `UserWarning`을 내고 값을 낸다). `bellman_ford`, `johnson`, `floyd_warshall`, auto(J를 고른다)는 받는다. 무향 그래프의 음수 변은 그 자체로 음수 순환이다(scipy와 같다).
- **`NegativeCycleError`**: `sd.graph.NegativeCycleError(ValueError)`다(API.md §2의 영역 고유 예외 규칙; `[scipy와 다름]`: scipy의 것은 `Exception`의 하위 클래스다). **요청한 원점에서 닿을 수 있는 음수 순환이 있으면** 낸다(모든 방법이 같다; `indices=None`이면 그래프 어디든). `[scipy와 다름]`: scipy의 `bellman_ford(indices=0)`은 닿지 않는 순환을 무시하고 `johnson(indices=0)`은 그 순환에서 에러를 낸다(이 초안에서 확인). 메시지는 순환 위의 노드 하나를 적는다(`bellman_ford(): negative cycle through node 3`; 그 노드는 모든 방법이 같이 쓰는 검사가 찾은 순환 위의 가장 작은 번호다: GR35).
- **`yen(csgraph, source, sink, K)`**: 고리 없는 s–t 경로 가운데 (차례 합의 길이, 변의 수, 노드 열의 사전 순서)로 앞선 K개다(§3.2). 결과는 `dist_array`(길이 M ≤ K) 하나 또는 `YenResult(dist_array, predecessors)`(M × n, int64)다(scipy와 같은 꼴). 음수 무게는 받고(scipy처럼 J의 퍼텐셜을 쓴다) 음수 순환은 `NegativeCycleError`다. `source == sink`면 길이 0의 경로 하나다(scipy와 같다). `K`는 0 이상의 int다(0이면 빈 결과; scipy와 같다).
- **`construct_dist_matrix`**: 선행 노드 행렬이 가리키는 경로를 따라 무게를 원점에서부터 차례로 더한 거리다. 그래서 **FW가 아닌 방법에서 `construct_dist_matrix(G, shortest_path(G, return_predecessors=True).predecessors)`는 `shortest_path`의 거리와 비트까지 같다**(FW는 GR27). `[scipy와 다름]`: scipy는 도착점에서 거꾸로 더해 끝자리가 다르다(이 초안에서 무작위 그래프로 확인: scipy의 `construct_dist_matrix`는 거꾸로 합과 같고 자기 `shortest_path`와 다르다). 닿지 않는 자리는 `null_value`다(scipy와 같다). 선행 노드는 정수 dtype의 (n, n)이고 그래프의 경로를 그려야 한다: 노드도 −9999도 아닌 값, 없는 변, 순환, 원점이 아닌 곳에서 끝나는 사슬은 `ValueError`다(GR38; `[scipy와 다름]`: scipy는 int32만 받고 없는 변에 inf를 더하며 순환에서 끝나지 않는다).
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
- **`minimum_spanning_tree`**: 무향으로 본 그래프(§2.1의 무향 규칙)의 최소 신장 숲이다(연결되지 않으면 요소마다의 트리; scipy와 같다). **답은 하나다**: 변을 (무게, min(i, j), max(i, j))의 전순서로 견주면 최소 신장 숲이 유일하다(§3.4; §9의 14번). 결과는 (n, n) `csr_array` float64이고 고른 변은 원래 저장된 방향의 자리에 있다(두 방향이 다르면 작은 무게의 자리, 같으면 i < j인 위 삼각; scipy도 원래 자리에 둔다; GR43). 계산은 기수 정렬과 Kruskal이다(GR43). 음수 무게는 받는다(scipy와 같다). `[scipy와 다름]`: scipy는 밀집 입력에서 1e-8보다 작은 원소를 0(변 없음)으로 버린다(문서에 적혀 있고 이 초안에서 확인). sd는 밀집의 변 규칙(§2.1) 그대로다. 저장된 NaN·±inf는 `ValueError`(scipy는 NaN·inf 변을 트리에 넣는다).
- **`maximum_flow`**: 결과는 `MaximumFlowResult(flow_value, flow)`다(scipy의 이름과 필드). `flow_value`는 Python `int`, `flow`는 (n, n) `csr_array` int64이고 반대 방향에 음수 흐름이 있는 반대칭 꼴이다(scipy와 같다; scipy는 int32). 패턴은 G와 Gᵀ의 패턴이다: 저장된 용량 0의 변과 대각의 저장 원소도 흐름 0으로 있다(scipy와 같다; GR44). 흐름 값은 유일하지만 흐름 자체는 여럿일 수 있고, 돌려주는 흐름은 고정된 훑기 순서의 결정적인 결과다(§9의 13번).
  - 용량(§9의 12번): bool·int64 dtype만 받는다(float64는 정수 값이어도 `TypeError`; `[scipy와 다름]`: scipy는 `ValueError`다). 음수 용량은 `ValueError`(`[scipy와 다름]`: scipy는 조용히 0으로 본다). 원점에서 나가는 용량의 합(원점의 자기 고리는 뺀다)이 int64를 넘으면 `OverflowError`(흐름 값이 그 합 이하라서 넘침이 미리 막힌다; 잔여 용량은 uint64로 쥔다, GR44). 메시지는 `maximum_flow(): the capacities must be bool or int64, not float64`, `the capacities must be nonnegative`, `the capacities out of the source sum past the int64 range`다. `[scipy와 다름]`: scipy는 용량을 int32로 줄여 2⁴⁰ 용량의 흐름이 0, 2³² + 5가 5가 된다(이 초안에서 확인).
  - 입력은 희소와 밀집 모두다(`[scipy와 다름]`: scipy는 CSR만 받고 밀집은 `TypeError`). `source == sink`는 `ValueError`(scipy와 같다). 자기 고리는 흐름에 쓰이지 않는다. 용량이 int32 안이면 돌려주는 흐름이 scipy의 흐름과 배열까지 같다(두 방법 모두; GR44).
- **`maximum_bipartite_matching`**: 행 쪽과 열 쪽의 이분 그래프(직사각 가능)에서 가장 큰 매칭이다. `perm_type='row'`면 길이(열 수)의 int64 배열(열 j에 짝지은 행, 없으면 −1), `'column'`이면 행마다의 열이다(scipy와 같다). 저장된 0도 변이다(scipy와 같다). 매칭의 크기는 유일하고 매칭 자체는 결정적인 알고리즘의 결과다(§9의 13번; GR47). scipy와 크기는 같고 매칭은 다를 수 있다.
- **`min_weight_full_bipartite_matching`**: 크기가 min(행 수, 열 수)인 매칭 가운데 무게 합이 가장 작은(`maximize=True`면 가장 큰) 것이다. 결과는 `AssignmentResult(row_ind, col_ind)`(둘 다 int64; scipy는 `col_ind`가 int32)이고 `row_ind`는 오름차순이다(scipy와 같다). 그런 매칭이 없으면 `ValueError`(scipy와 같다). `[scipy와 다름]`: scipy는 무게 0을 막는다고 문서에 적고 실제로는 경고(`explicit zero weights are removed before matching`)와 함께 저장된 0을 변 없음으로 읽는다(이 초안과 GR3 마일스톤에서 확인). sd의 저장된 0은 무게 0의 변이다(§2.1). 저장된 NaN·±inf는 `ValueError`(scipy는 조용히 답을 낸다). 메시지는 `min_weight_full_bipartite_matching(): no full matching exists`(scipy와 같은 말)다.
- **`linear_sum_assignment`**(자리는 §9의 1번): scipy의 뜻 그대로다. `cost_matrix`의 모든 원소가 변이고(0도 비용 0의 변; 인접 행렬의 변 규칙이 아니다), +inf는 고를 수 없는 짝이다. 모든 짝을 고를 수 없으면 `ValueError`(`cost matrix is infeasible`), NaN과 −inf는 `ValueError`(scipy와 같다). `maximize=True`면 부호를 먼저 바꾸므로 −inf가 고를 수 없는 짝이고 +inf가 `ValueError`다(scipy와 같다; GR46). 직사각을 받고 결과는 `AssignmentResult(row_ind, col_ind)`(int64, `row_ind` 오름차순)다. 입력은 밀집만이다(희소 비용은 `min_weight_full_bipartite_matching`; 희소 배열은 그것을 짚는 `TypeError`). 비용 dtype은 bool, int64, float64다(float64로 바꾼다). 동점에서 고르는 최적 배정은 scipy와 다를 수 있다(GR45; scipy는 시작 단계가 없는 Crouse의 꼴).

### 2.5 순서, 구조적 랭크, 라플라시안
```
reverse_cuthill_mckee(graph, /, *, symmetric_mode=False)
structural_rank(graph, /)
laplacian(csgraph, /, *, normed=False, return_diag=False, use_out_degree=False, form='array', dtype=None, symmetrized=False)
```
- **`reverse_cuthill_mckee`**: 정방 그래프의 띠 폭을 줄이는 순열(int64 (n,); scipy 1.18.1도 int64다: GR0 마일스톤의 기록)이다. `symmetric_mode=False`면 A + Aᵀ의 패턴으로 한다(scipy와 같다). 동점 규칙(§3.6): 요소마다 시작 노드는 차수가 가장 작은 노드 가운데 가장 작은 번호, 이웃은 (차수, 번호) 순서로 큐에 넣고, 요소는 시작 노드의 순서로 이어 붙인 뒤 전체를 뒤집는다. 이 초안에서 별 그래프(`[3, 2, 0, 1]`)와 경로 그래프(`[3, 2, 1, 0]`)에서 scipy와 같았다. 다른 동점의 일치는 GR4 마일스톤에서 확인한다. 밀집 입력도 받는다(`[scipy와 다름]`: scipy는 CSR·CSC만).
- **`structural_rank`**: 패턴의 가장 큰 이분 매칭의 크기(Python `int`; scipy는 `numpy.int64`)다. 직사각을 받는다(scipy와 같다). 저장된 0도 원소다(scipy와 같다).
- **`laplacian`**: L = D − A(D는 들어오는 무게의 합, `use_out_degree=True`면 나가는 무게의 합; scipy와 같다). 대각의 원소는 무시한다. `normed=True`면 D^(−1/2) L D^(−1/2)이고 차수 0인 노드는 1로 나눈다(scipy와 같다). `symmetrized=True`면 A + Aᵀ로 먼저 대칭으로 만든다(scipy와 같다). 값을 그대로 쓰므로 NaN·inf는 IEEE로 퍼진다(행렬 함수다, §2.1). 음수 무게의 `normed`는 √ 음수에서 NaN이다(scipy와 같다).
  - 결과: 밀집 입력이면 `sd.ndarray`, 희소 입력이면 `csr_array`(`[scipy와 다름]`: scipy는 `coo_array`. sd의 계산 형식이고 `eigsh`가 바로 쓴다), `form='lo'`면 `sd.sparse.LinearOperator`(행렬을 만들지 않는다; scipy와 같다). `form='function'`과 `copy`는 없다(§9의 11번). `return_diag=True`면 `LaplacianResult(laplacian, diag)`다.
  - dtype: `dtype=None`이면 입력의 dtype(bool은 int64로 올린다), `normed=True`면 float64다(scipy와 같다).

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
- **`pagerank`**: 확률 벡터 x(합 1, float64 (n,))이고 x = α·Pᵀx + (1 − α)·v의 해다. P는 나가는 무게로 정규화한 전이 행렬이고(`unweighted=True`면 변마다 1), 나가는 무게가 0인 노드(매달린 노드)의 행은 `dangling` 분포로 바꾼다. 기본 `personalization`은 균등, 기본 `dangling`은 `personalization`이다(networkx의 뜻). `directed=False`면 §2.1의 무향 규칙으로 만든 대칭 그래프다. `alpha`는 0 이상 1 미만(1은 해가 하나가 아닐 수 있어 `ValueError`), `personalization`·`dangling`은 길이 n의 음이 아닌 유한 배열이고 합이 0보다 커야 한다(합 1로 정규화한다). 음수 무게는 `ValueError`다. **결과는 연립방정식의 해 x\*와 ℓ₁로 `tol` 안이다**(증명된 한계로 멈춘다; §3.7, §9의 16번). 반복 수의 상한은 α와 `tol`로 정해지므로 `max_iter`가 없다(`[networkx와 다름]`: networkx는 `tol=1e-6`, `max_iter=100`, 연속한 반복의 차로 멈추고 수렴하지 못하면 예외다).
- **`betweenness_centrality`**: Brandes의 정의다. 노드 v의 값은 v가 아닌 모든 (s, t)에서 s–t 최단 경로 가운데 v를 지나는 몫의 합이다. 무향이면 (s, t)와 (t, s)를 한 번 센다(합을 2로 나눈다), `normalized=True`면 유향 1/((n − 1)(n − 2)), 무향 2/((n − 1)(n − 2))를 곱하고, `endpoints=True`면 끝점도 센다(networkx의 뜻; n ≤ 2의 정규화는 GR5 마일스톤에서 networkx의 문서대로 맞춘다). 최단 경로와 그 개수는 §3.2의 거리와 §3.7의 순서로 정한다. float64 (n,)이다. 무게는 0보다 커야 하고(무게 0의 변이 있으면 최단 경로가 무한히 많을 수 있다; `ValueError`), `unweighted=True`면 패턴만 본다. 표본으로 어림하는 `k`, `seed`는 없다(나중; §9의 17번).

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
- 최단 경로의 결과가 n² 원소(`indices=None`)면 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2; 선행 노드를 함께 돌려주면 16n² 바이트).

### 2.9 scipy와 다른 곳 (모음)
위의 `[scipy와 다름]`을 모은다. 이유는 그 자리에 있다.
- 입력: complex는 `TypeError`, scipy 희소 객체는 `TypeError`(변환을 짚는다), 밀집 입력의 |w| ≤ 1e-8인 원소도 변이다(scipy는 최소 신장 트리뿐 아니라 모든 csgraph 함수의 밀집 입력에서 그 원소를 버린다: `dijkstra([[0, 1e-9], [0, 0]], indices=0)`이 [0, inf], GR2 마일스톤에서 확인), 희소 입력의 저장된 NaN·±inf 무게는 `ValueError`, `construct_dist_matrix`·`reconstruct_path`의 선행 노드는 정수 dtype이면 받고 경로가 아니면 `ValueError`(GR38), 노드 인자는 0 … n−1의 int만(실수 자르기와 음수 감기 없음), 2차원 `indices`는 `ValueError`, 옵션은 키워드만, `overwrite`·`copy`·`form='function'`·가린 배열 도구 없음, `maximum_flow`·`reverse_cuthill_mckee`·`structural_rank`·매칭도 밀집 입력을 받는다, `maximum_flow`의 int64 용량을 int32로 줄이지 않는다(scipy는 2⁴⁰을 0으로, 2³² + 5를 5로 읽는다), `min_weight_full_bipartite_matching`의 저장된 0은 무게 0의 짝(scipy는 경고와 함께 버린다).
- 결과: 색인 배열은 모두 int64(선행 노드, 레이블, 순서, 순열, 매칭, 흐름), 스칼라는 Python `int`, 강한 연결 요소의 번호는 가장 작은 노드 순서, 무향 순회는 합친 이웃의 오름차순, 무향 트리의 저장된 0은 0(GR29), `csgraph_to_dense`의 저장된 NaN은 NaN(GR30), FW를 뺀 최단 경로의 거리와 선행 노드는 방법과 무관, `auto`는 FW를 고르지 않음, `yen`의 같은 길이의 경로는 (변 수, 노드 열)의 순서(scipy의 마름모에서 첫 경로는 0–2–3, sd는 0–1–3), `construct_dist_matrix`는 `shortest_path`와 같은 합 순서, `laplacian`의 희소 결과는 CSR, 최소 신장 트리는 밀집의 작은 원소를 버리지 않는다, `maximum_flow`의 흐름은 int64(scipy는 int32; 용량이 int32 안이면 값은 scipy와 같다), 매칭과 배정은 동점에서 scipy와 다른 최적해를 고를 수 있다(GR45, GR47).
- 에러: 음수 무게의 `dijkstra`, NaN `limit`, 음수 용량(scipy는 0으로 본다), int64를 넘는 흐름(`OverflowError`), 실수 용량(`TypeError`; scipy는 `ValueError`), 최소 신장 트리와 `min_weight_full_bipartite_matching`의 저장된 NaN·±inf는 에러이고, `NegativeCycleError`는 `ValueError`의 하위 클래스이며 모든 방법이 "요청한 원점에서 닿는 순환"이라는 한 규칙이다.

---

## 3. 수치 방법

### 3.1 그래프의 내부 표현
- 모든 함수는 Python 층(`src/stardust/graph/_common.py`의 `graph()`)에서 입력을 **CSR 하나**(int64 `indptr`, `indices`, float64 또는 패턴만의 값)로 바꾼 뒤 C++를 한 번 부른다. 희소 CSR 입력은 복사하지 않는다(표준 꼴이 보장되어 그대로 읽는다; 다른 꼴은 `tocsr()` 한 번, 무게를 쓰면 float64가 아닌 값만 바꾼다). 밀집 입력은 변 규칙(§2.1)으로 CSR을 만든다(`_core.graph._dense_csr`: 크기로만 정한 행 묶음(원소 2¹⁵개씩)마다 한 번은 가지 없이 세고, 한 번은 행을 가지 없이 임시 행에 모아 제자리로 옮긴다; 원소 2¹⁵개부터 병렬. GR1 마일스톤에서 가지 있는 꼴의 25.6 ms를 4.1 ms로 줄였다, n = 2000, 1스레드).
- 무향(`directed=False`) 순회는 sparse의 전치(`sparse::transpose`, sparse.md §2.6의 계수 정렬; 패턴만 읽으면 `indices`를 값 자리에 넣는다)를 한 번 만들고, G가 자기 전치와 같으면 G만, 아니면 G와 전치를 함께 읽는다(GR32). 연결 요소는 합집합–찾기가 방향을 보지 않으므로 전치도 필요 없다. 합친 CSR(두 방향 이웃의 합집합, 무게는 GR29)은 그것이 필요한 마일스톤(최단 경로, 최소 신장 트리 등)에서 만든다. scipy의 문서는 대칭 행렬이면 `directed=True`가 더 빠르다고 적는다. sd는 대칭 확인으로 그 차이를 줄인다.
- 들어오는 변이 필요한 계산(강한 연결 요소의 일부, PageRank의 Pᵀx, 역방향 너비 우선)은 전치 CSR을 한 번 만든다(sparse의 안정 계수 정렬; sparse.md §2.6).
- 무게의 사전 검사(NaN·±inf, 음수)는 저장된 값을 한 번 훑는다. 검사는 계산 전에 하므로 실패한 호출은 아무것도 만들지 않는다.

### 3.2 최단 경로의 정의와 계산
**거리.** 원점 s(여럿이면 모두)에서 v까지의 거리는
d̂(v) = min over s–v 경로 p = (s = v₀, v₁, …, v_k = v) of L(p), L(p) = fl(…fl(fl(0 + w₁) + w₂) … + w_k)
다(원점에서부터의 차례 합; 0 + w₁ = w₁은 정확하다). 이완 x ↦ fl(x + w)는 x에 대해 단조이므로, 이 값은 이완 방정식 d(v) = min_u fl(d(u) + w(u, v))(원점은 0)의 **가장 작은 고정점**이고 어떤 이완 순서로 구해도 같다. 음이 아닌 무게에서는 fl(x + w) ≥ x라 순환을 넣어도 값이 작아지지 않아 최솟값이 단순 경로에서 나온다(Dijkstra의 정당성이 반올림 아래에서도 선다; Knuth의 일반화). 그래서:
- **D**(이진 힙의 Dijkstra), **BF**(큐 기반의 이완: 바뀐 노드만 다시 본다), **J**(아래)는 모두 이 고정점에 도달하고 비트까지 같다. scipy의 D·BF·J도 거리가 차례 합과 같았다(이 초안에서 확인).
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
  - 1.3배를 넘으므로 §9의 2번의 규칙대로 `'FW'`는 블록 Floyd–Warshall의 반올림으로 바꿨다(GR25). 블록 크기 64는 고정이고(바꾸면 끝자리가 바뀐다), 블록 하나는 한 태스크가 맡아 스레드 수와 무관하다. 블록 FW는 같은 k 순서의 고전 FW와 같은 갱신을 하지만 단계 3에서 d(i, k)가 같은 블록의 뒤 k까지 갱신된 값이라 고전 FW와도 끝자리가 다를 수 있다. 그래서 FW의 거리는 "블록 FW의 값"으로만 정해지고, 계약(§4)은 다른 방법과 같은 한계, 상수 1이다(GR37: 덧셈 나무의 꼴과 무관한 한계라 블록의 합 깊이를 세지 않는다). 선행 노드를 함께 구해도 거리는 같은 비트다(GR42). n이 64의 배수가 아니면 +inf로 채운 행렬에서 돌고 앞 n × n을 옮긴다. 음수 무게는 그대로 받고 음수 순환은 대각이 음수가 되는 것으로 알린다(GR27).
  - auto는 FW를 고르지 않고 밀집한 그래프에서 NaN 꼴의 밀집 Dijkstra를 쓴다(GR26): FW보다 1.3–2.4배 느리지만 scipy의 `floyd_warshall`보다 1스레드에서 2.5배 빠르고 결과가 D와 같다. 음수 무게면 BF의 퍼텐셜로 같은 다시 넣기 규칙을 쓴다(J와 같은 비트).
- **`unweighted=True`**: 무게 1의 차례 합은 변의 수와 같고 정확하다. 너비 우선(원점마다)으로 구한다.
- **`limit`**: d̂ ≤ `limit`인 노드에서 멈추는 것은 고정점의 그 부분과 같다(Dijkstra는 `limit`을 넘는 키를 꺼내면 멈춘다).

**선행 노드.** 거리가 정해지면 꼭 맞는 변의 집합 T = {(u, v) : fl(d̂(u) + w(u, v)) = d̂(v), u ≠ v}을 만든다(정확한 비교; `unweighted`면 d̂(u) + 1 = d̂(v)). h(v)를 T 위에서 원점으로부터의 너비 우선 변 수라 하면
pred(v) = min{u : (u, v) ∈ T, h(u) = h(v) − 1}
이다(원점과 닿지 않는 노드는 −9999). h가 사슬을 따라 하나씩 줄어 선행 노드 그래프는 언제나 트리다. 무게 0의 변(희소의 저장된 0)이나 흡수(fl(d + w) = d)가 있으면 "같은 거리의 가장 작은 번호"만으로는 순환이 생길 수 있어 변 수를 먼저 본다(§9의 3번). 모든 방법이 같은 d̂에서 같은 T, h, pred를 만들므로 선행 노드도 방법과 무관하다. 비용은 원점마다 O(n + E)의 한 번 훑기다. 정수 무게(와 `unweighted`)에서는 이것이 "가장 적은 변 수의 최단 경로 가운데 마지막 변의 출발이 가장 작은 것"이고 정확한 유리수 참값과 같다(§4).
- `min_only`: 모든 원점의 h가 0이고 `sources[v]`는 pred 사슬의 끝이다.
- 무향: T와 pred는 합친 CSR(§3.1)에서 만든다.

**합의 꼴과 `construct_dist_matrix`.** 거리 d̂(v)는 pred 사슬이 그리는 경로의 차례 합과 같다(T의 정의에서 바로 나온다). `construct_dist_matrix`는 그 경로를 원점에서부터 다시 더하므로 비트가 같다.

**Yen.** 경로의 전순서를 (L(p), 변 수, 노드 열의 사전 순서)로 둔다. 갈래 경로(spur)는 뿌리 경로의 차례 합 x에서 시작하는 Dijkstra(갈래 노드의 처음 값이 x)로 구해 전체 경로의 L이 원점에서부터의 차례 합이 되게 하고, 후보 힙도 이 전순서다. 그래서 동점이 많은 그래프에서도 앞선 K개가 데이터로만 정해진다. 작은 그래프에서 모든 고리 없는 경로를 나열한 참값과 정확히 같은지 테스트한다(§7.3).

**음수 순환.** 모든 방법이 같은 검사 하나(GR35: 요청한 원점에서 닿는 노드를 잇는 가상 뿌리의 큐 Bellman–Ford, 다시 넣은 횟수가 닿는 노드 수를 넘으면 부모 포인터의 순환)를 쓰고, 메시지의 노드는 그 순환의 가장 작은 노드다. 원점마다의 BF·J는 그 뒤에도 노드마다 넣은 횟수 n을 넘으면 멈추고 같은 검사로 알린다(아래의 반올림 순환에서만 생긴다). 정확한 합이 음수인 순환은 언제나 걸린다. 정확한 합이 0인데 차례 합의 반올림이 계속 줄어드는 순환은 이론상 걸릴 수 있다(정수 무게에서는 생기지 않는다). 이 경우는 §4에 적는다.

### 3.3 순회, 연결 요소, 위상 정렬
- 너비 우선은 배열 큐, 깊이 우선은 명시적 스택(재귀 없음; 노드마다 다음에 볼 이웃의 위치)이다. 이웃 순서는 CSR의 오름차순이라 순서가 데이터로만 정해진다. 무향은 GR32의 두 줄 읽기다. 트리는 노드마다 찾은 변의 무게를 적어 두고 끝에 노드 오름차순으로 부모의 줄에 넣는다(그래서 줄 안이 오름차순인 표준 꼴이다).
- 약한 연결(과 무향): 합집합–찾기(경로 반감, 번호가 작은 뿌리로 합침)로 분할을 구하고 마지막에 가장 작은 노드 순서로 번호를 다시 매긴다. 분할이 유일하므로 E가 크면 변 구간마다 병렬로 합칠 수 있다(원자적 합침; 순서와 무관한 같은 분할). GR1 마일스톤에서 재서 저장된 원소 2¹⁸개부터 병렬로 정했다(GR31).
- 강한 연결: 반복형 Tarjan(발견 번호, low link, 노드마다 다음 변의 위치, 미배정 노드의 스택)으로 분할을 구하고 번호를 다시 매긴다. O(n + E), 직렬이다(이 초안에서 scipy의 강한 연결이 10⁶ 변에 6.2 ms이고, 병렬 알고리즘이 이길 크기가 드물다).
- 위상 정렬: 진입 차수의 Kahn 알고리즘이고 준비된 노드 가운데 가장 작은 것을 꺼낸다. 준비된 노드의 집합은 힙 대신 층을 이룬 비트맵(위 층의 비트가 아래 64비트 낱말이 비지 않았음을 적는다)이라 넣기와 가장 작은 것 꺼내기가 낱말 몇 번의 연산이다. 남은 노드가 있으면(자기 고리 포함) 순환이다.

### 3.4 최소 신장 트리
- 변을 키 (w, min(i, j), max(i, j))로 견주면 전순서이고, 모든 무게가 다른 그래프의 최소 신장 숲이 유일하다는 정리가 그대로 서서 답이 하나다. 무향 규칙(작은 무게)으로 합친 변 목록을 이 키로 정렬하고 Kruskal(합집합–찾기)로 고른다(scipy도 Kruskal이다).
- 변 목록은 행 i마다 위 삼각의 저장 원소 (i, j)와 아래 삼각의 저장 원소 (j, i)(열마다 모은 계수 정렬)를 j 순서로 합쳐 (min, max) 순서로 만든다. 정렬은 (w + 0.0의 순서 키, 변 번호)의 안정 LSD 기수 정렬(11비트 자리 여섯, 모든 키가 같은 자리는 건너뜀; 변 256개 이하는 안정 비교 정렬)이라 같은 무게가 (min, max) 순서로 남는다(GR43). 직렬이고 저장 원소 2¹⁵개부터 GIL을 놓는다.
- Borůvka는 넣지 않았다: GR3 마일스톤에서 이 Kruskal이 n = 10⁵, d = 5 대칭에서 11.1 ms(scipy 76.2 ms, 1스레드)라 병렬 Borůvka의 이득을 잴 만한 몫이 작았다(GR43; GR6 마일스톤에서 더 큰 그래프로 다시 본다).
- 결과의 자리: 고른 무향 변 {i, j}는 그 무게를 준 저장 원소(비트가 같은 쪽)의 자리에 둔다(두 방향의 비트가 같으면 i < j의 위 삼각). 고른 원소의 표시로 입력을 걸러 결과를 만들므로 표준 꼴이고 값은 저장된 그대로다.

### 3.5 흐름과 매칭
- **최대 흐름**: Dinic(층 그래프와 막는 흐름)과 Edmonds–Karp(너비 우선 증가 경로)다. 잔여 그래프는 변마다 반대 변의 자리를 쥔 CSR(입력에 없는 반대 방향을 더한 패턴; 패턴이 대칭이라 반대 자리는 행을 차례로 걸으며 센다)이다. 너비 우선과 깊이 우선의 이웃 순서는 오름차순이라 돌려주는 흐름이 데이터로만 정해진다. Edmonds–Karp는 싱크에 닿으면 너비 우선을 멈추고 그 경로를 병목만큼 민다. Dinic은 싱크의 층까지 층을 매기고, 현재 호(가장 작은 이웃부터)의 깊이 우선으로 경로를 하나씩 밀며 막다른 노드는 층에서 뺀다. 잔여 용량은 uint64, 흐름은 int64이고 덧셈이 넘치지 않음은 원점 용량 합의 검사(§2.4)가 보장한다(GR44).
- **가장 큰 이분 매칭**(`maximum_bipartite_matching`, `structural_rank`): 탐욕 시작과 Hopcroft–Karp(scipy와 같은 알고리즘)이고 행과 이웃을 오름차순으로 본다. 깊이 우선은 명시적 스택이라 긴 경로에서도 재귀가 없다(GR47).
- **가장 작은 무게의 완전 매칭과 배정**: 두 함수가 같은 C++ 풀이(csrc/graph/matching.cpp의 `Solver`; LAPJV의 시작과 최단 증가 경로, GR45)를 쓰고 밀집과 희소의 행 읽기만 다르다(밀집은 남은 열을 모두 훑고, 희소는 이웃 열만 이완하고 닿은 열을 이진 색인 힙에 둔다). 고르는 순서가 (거리, 빈 열 먼저, 번호)의 전순서라 같은 짝이면 두 함수가 같은 배정을 낸다. 동점(여러 최적 배정)에서는 이 고정된 순서의 결과를 돌려준다(§9의 13번). 정수 비용은 쌍대 변수와 비용이 모두 정확하여 최적이 정확하다. 직사각은 scipy처럼 작은 쪽을 기준으로 한다(행이 많으면 전치해 풀고 행 순서로 돌려준다). 희소 탐색의 작업 공간은 닿은 열만 되돌려 행마다 O(n)을 쓰지 않는다.

### 3.6 순서와 라플라시안
- **역 Cuthill–McKee**: §2.5의 동점 규칙. 차수 계산과 정렬은 O(n + E)이고 직렬이다.
- **라플라시안**: 차수는 행(또는 열)마다 대각을 뺀 저장된 무게를 **색인 오름차순으로 차례로** 더한 것이다(sparse의 줄 합 규칙과 같은 순서; sparse.md §2.6). `normed`는 w = √d(d = 0이면 1)를 한 번 구하고 L_ij = −A_ij / w_i / w_j(왼쪽부터 두 번 나눔; scipy의 순서), 대각은 d_i / w_i / w_i(d > 0이면 1이 아닐 수 있다: 반올림까지 그대로 적는다)다. scipy가 만드는 −0.0(A_ij = 0의 저장된 원소)은 같은 식이라 같게 나온다.

### 3.7 PageRank와 매개 중심성
- **PageRank**: 반복 x_{k+1} = α·Pᵀx_k + α·(매달린 질량)·dangling + (1 − α)·v는 ℓ₁에서 비율 α의 축소 사상이다. 그래서 ‖x_{k+1} − x\*‖₁ ≤ α/(1 − α)·‖x_{k+1} − x_k‖₁이 증명된 사후 한계다. 이 한계(와 §4의 반올림 바닥)가 `tol` 이하가 되면 멈춘다. 반복 수는 ⌈log(tol·(1 − α)/2)/log α⌉ + 1 이하로 미리 정해진다(α = 0.85, tol = 10⁻¹²에서 약 186).
  - Pᵀx는 전치 CSR의 행 모으기다: 행(노드)마다 들어오는 항을 색인 오름차순으로 차례로 더한다(sparse의 SpMV 규칙; 긴 줄은 4096개 조각과 고정 트리). 행 구간은 크기로만 나눈다(sparse와 같다). 그래서 결과는 스레드 수와 무관하다. 매달린 질량과 ‖·‖₁은 배열의 pairwise 합이다.
  - 처음 값은 `personalization`이다. 모든 항이 음이 아니라 상쇄가 없다.
- **매개 중심성**(Brandes): 원점마다 최단 경로 DAG를 만들고 경로 수 σ를 앞으로, 의존 δ를 뒤로 쌓는다.
  - 무게 없음: 너비 우선. 무게 있음: §3.2의 d̂와 원점마다의 Dijkstra. DAG의 변은 fl(d̂(u) + w) = d̂(v)인 (u, v)이고 (d̂, 번호)로 u가 v보다 앞서야 한다(무게가 양수라 흡수가 없으면 표준 정의와 같다; 흡수가 만드는 같은 거리의 변이 순환을 만들지 않게 한다).
  - σ는 경로 수라 n이 수백이어도 double을 넘칠 수 있다(사다리 그래프에서 2^(n/2)). σ를 (double, 2의 지수) 꼴로 쥐고 δ에 쓰는 비 σ(u)/σ(v)만 double로 만든다. 그래서 넘침도 언더플로도 없다.
  - 원점마다의 δ는 태스크(원점 64개씩, 크기로만 정함)의 부분합에 더하고, 태스크의 부분합을 태스크 번호 순서로 더한다. 결과가 스레드 수와 무관하다. 메모리는 태스크마다 O(n + E)다.

### 3.8 병렬화와 결정성
**같은 입력이면 스레드 수와 `method`에 상관없이 같은 결과다.** 테스트는 1, 2, 3, 7, 10스레드에서 같은 바이트를 확인한다(§7).

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 입력 변환(밀집 → CSR, `csgraph_to_dense`, 합친 CSR, 전치, 검사) | 행 구간(원소 2¹⁵개씩), 고정 조각의 계수 정렬 | 행 하나는 한 태스크이고 조각 순서가 정해져 있다(sparse.md §2.6) |
| 여러 원점의 최단 경로(`indices`, 모든 쌍, auto의 밀집 Dijkstra) | 원점 1–64개(크기로만 정한 수)가 태스크 하나(GR40) | 원점 하나의 결과는 정의로 정해진다(§3.2) |
| FW(블록 Floyd–Warshall; GR25) | 단계마다 64 × 64 블록 하나가 태스크 하나, 단계 사이에 장벽 | 블록 하나의 연산 순서가 블록 크기로만 정해진다 |
| 약한 연결 요소(저장된 원소 2¹⁸개부터; GR31) | 행 구간(원소 2¹⁶개씩) | 분할이 유일하고 번호는 가장 작은 노드 순서다 |
| PageRank의 Pᵀx | nnz로 정한 행 구간 | 행 하나의 합 순서가 정해져 있다 |
| 매개 중심성 | 원점 64개의 태스크 | 태스크 번호 순서로 더한다 |
| 직렬(단일 원점 Dijkstra·BF, 순회, 강한 연결, 위상 정렬, 최소 신장 트리(GR43), 흐름, 매칭, 배정, RCM, 단절점·다리·이중 연결, 코어 수) | — | 고정된 훑기 순서(최소 신장 트리는 키가 전순서라 유일하다) |

- 병렬 기준값은 배열의 규칙이다(array.md §10.4): 작업량(변의 수 × 원점 수 등)이 float64 연산 2¹⁵개 분량 이상이면 나누고, 그 이상이면 스레드 수와 상관없이 GIL을 놓는다(API.md §3). 직렬 알고리즘도 기준값 이상이면 GIL을 놓는다(`Unlocked`, ARCHITECTURE §4.1).
- 단일 원점의 병렬 최단 경로(Δ-stepping)는 결과가 같은 고정점이라 넣을 수 있지만 첫 판에는 넣지 않는다(나중; GR6 마일스톤에서 재 본다).

### 3.9 메모리
- 사용자가 모양으로 요청한 출력(모든 쌍의 n² 거리와 선행 노드, `yen`의 K × n)은 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2).
- 내부: CSR 사본(무향의 합친 CSR, 전치, 잔여 그래프)은 입력 크기의 1–2배다. 원점마다의 작업(힙, 거리, h, T의 표시)은 태스크마다 O(n + E) 비트·double 몇 줄이다. FW 길은 밀집 행렬 n² double 하나(밀집 입력이면 바꾼 float64 사본)다. 매개 중심성은 태스크마다 σ·δ·순서 O(n)과 결과의 부분합 n이다.

---

## 4. 정확도 계약
판정은 `tests/graph/refdata.py`의 `bound()`·`units()` 하나이고 테스트와 정확도 체계(`bench/accuracy/graph.py`, 스윕이 생기면 만드는 표 `bench/<영역>/accuracy.md`)가 같이 쓴다(spatial.md §4와 같은 꼴: 계약 단위로 적고 경계는 1이다).

- **정확한 것**(참값과 같아야 한다): 연결 요소, 순회의 순서와 트리, 위상 정렬, 최소 신장 트리(고른 변의 집합과 자리), 역 Cuthill–McKee, 구조적 랭크, 매칭의 크기, 흐름 값(과 흐름의 보존·용량 조건), 단절점, 다리, 이중 연결 요소, 코어 수. 참값은 작은 그래프에서 정의를 그대로 옮긴 Python 구현(정확한 유리수)이다.
- **최단 경로**
  1. **방법과 무관하다**: D, BF, J, auto의 거리와 선행 노드가 비트까지 같다(모든 시험 그래프, 스레드 수마다). FW의 거리는 3번의 한계 안이고(상수 1, p̂는 FW의 선행 노드가 그리는 경로; GR37), 스레드 수마다 같은 비트다.
  2. **정수 무게(와 `unweighted`)는 정확하다**: 모든 부분합이 2⁵³ 이하이면 거리는 정확한 최단 거리와 같고, 선행 노드는 정확한 유리수로 계산한 §3.2의 규칙과 같다. 동점이 아주 많은 격자에서 동점 규칙을 직접 시험한다.
  3. **일반 무게**: 거리는 그 선행 경로의 차례 합과 비트까지 같다(`construct_dist_matrix`와 같다). 참 최단 거리 d\*와의 차이는 |d̂ − d\*| ≤ γ_{H−1}·max(Σ_{p̂}|w|, Σ_{p\*}|w|)다. H는 돌려준 경로 p̂와 참 최단 경로 p\* 가운데 긴 쪽의 변 수, Σ|w|는 그 경로의 무게 절댓값의 합이다(차례 합의 표준 한계: d̂ = L(p̂) ≥ S(p̂) − γΣ|w| ≥ d\* − γΣ|w|, d̂ ≤ L(p\*) ≤ d\* + γΣ|w|). 음이 아닌 무게면 상대 γ_{H−1}·d\*/(1 − γ_{H−1})다. 참 최단 거리는 정확한 유리수 Dijkstra(작은 그래프), 경로의 정확한 합은 Fraction이다.
  4. **음수 순환**: 정확한 합이 음수인 순환이 요청한 원점에서 닿으면 언제나 `NegativeCycleError`다. 정확한 합이 0인 순환에서 반올림이 값을 계속 줄이면 에러가 날 수 있다(정수 무게에서는 없다). 그 밖에는 에러가 나지 않는다.
- **`yen`**: 정수 무게에서 모든 고리 없는 경로를 나열한 정확한 정렬의 앞 K개와 같다. 일반 무게에서 경로마다 거리 계약(3번)이다.
- **`linear_sum_assignment`, `min_weight_full_bipartite_matching`**: 정수 비용에서 돌려준 배정의 정확한 비용이 최적과 같다(작은 문제는 나열, 큰 문제는 쌍대 실현 가능성의 정확한 확인). 일반 비용에서 돌려준 배정의 정확한 비용 C(σ̂)는 C\* + γ_{2r}·Σᵢ(|c_{iσ̂(i)}| + |c_{iσ\*(i)}|) 이하다(r은 작은 쪽의 크기; 상수 1, GR49). 판정은 `tests/graph/refdata.py`의 `assignment_units()`이고 참 최적은 참값 파일(작은 문제의 나열)과 정확한 유리수 풀이(`tests/graph/definitions.py`의 `assignment_exact`)다. GR3 마일스톤의 측정(`bench/accuracy/graph.py`의 `assignment`, 14영역)에서 sd와 scipy 모두 0 단위(참 최적)였다.
- **`laplacian`**: 차수는 줄 합의 계약 |d̂ − d| ≤ γ_m·Σ|A_ij|(m은 그 줄의 항 수; sparse.md §4와 같은 꼴), `normed`의 원소는 상대 γ_{m+2} + 3u 꼴(√와 두 나눗셈까지; 상수는 GR4 마일스톤에서 세어 적는다). Arb로 견준다.
- **`pagerank`**: ‖x̂ − x\*‖₁ ≤ tol + γ_{m+3}/(1 − α). x\*는 (I − αMᵀ)x = (1 − α)v(매달린 행을 바꾼 M)를 Arb의 행렬 풀이(`arb_mat.solve`)로 푼 참값, m은 가장 긴 들어오는 줄의 합 깊이다(§3.7). 반올림 바닥이 `tol`보다 크면(아주 큰 차수) 그 바닥에서 멈추고 결과는 위의 한계 안이다.
- **`betweenness_centrality`**: 노드마다 상대오차 ≤ γ_K·(참값)이다. 모든 항이 음이 아니라 상쇄가 없고, K는 의존 합의 깊이(가장 긴 DAG 경로의 변 수 × 최대 진출 차수)와 σ의 곱셈·나눗셈 수로 정한다(GR5 마일스톤에서 세어 적는다). 참값은 정확한 유리수 Brandes(σ는 Python 정수)다.
- GR2 마일스톤에서 잰 최단 경로(`bench/accuracy/graph.py`의 `paths`: 일반 무게의 참값 7,639점, 그래프 11개 × 유향·무향): 정의의 거리와 FW의 거리 모두 가장 큰 값이 0.995 단위(heavytail/120 유향), ulp로 2 이하였고, 정수·2진 유리수·bool 무게와 `unweighted`는 참값과 비트까지 같았다(tests/graph/test_paths_reference.py). scipy의 D와 FW도 같은 그래프에서 같은 값이었다(밀집 dense/special6만 scipy가 1e-9의 변을 버려 다르다; §2.9).
- scipy와의 차이(동점, FW의 끝자리, 거꾸로 더한 `construct_dist_matrix`)는 계약이 아니라 정확도 표의 비교 패키지 열로 보인다.

---

## 5. 성능
이 초안에서 잰 scipy 1.18.1의 시간이다(ms, 1스레드(scipy의 csgraph는 1스레드다), 가장 좋은 값(세 번, 1초를 넘는 것은 한 번), 부하 평균 1.5). 그래프는 무작위 유향(노드마다 나가는 변 d개, 무게 [0.01, 1.01)), "대칭"은 그것과 전치의 큰 값이다. GR0 마일스톤에서 `tools/graph_probe.py`가 같은 크기를 다시 재고, 사례가 생기면 `bench/speed`가 잰다.

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
- **목표**(GR6 마일스톤에서 확인한다)
  - 모든 함수가 1스레드에서 scipy 이상이다. 직렬 알고리즘은 같은 알고리즘에 sd의 이점(표준 꼴 CSR이라 변환이 없음, int64 색인 그대로, 연속한 배열)만 있으므로 같거나 조금 빠른 것이 목표다. 큰 차이는 검증과 변환에서 난다(scipy는 모든 호출에서 CSR로 바꾸고 float64로 바꾼다).
  - 여러 원점(`indices`, 모든 쌍, auto의 밀집 Dijkstra, `johnson`)과 매개 중심성: 10스레드에서 스레드 수에 가깝게(7배 이상).
  - `bellman_ford`: 큐 기반 이완으로 scipy보다 크게 빠르다(같은 고정점; 최악은 같은 O(nE)).
  - FW(블록 Floyd–Warshall; GR25): 1스레드에서 scipy의 `floyd_warshall`의 4배 이상(GR0 마일스톤의 시제품 4.5배, n = 1024), 10스레드에서 그 5배 이상. auto의 밀집 Dijkstra(GR26)는 1스레드에서 scipy의 `floyd_warshall` 이상.
  - PageRank: 반복 하나가 sd.sparse의 같은 크기 SpMV(1스레드·10스레드)의 1.5배 안.
  - 비교 패키지가 없는 다섯(단절점, 다리, 이중 연결 요소, 코어 수, 위상 정렬)은 O(n + E)의 한 번 훑기가 목표이고 같은 크기의 `connected_components`(약한 연결)와 견준다. networkx를 설치하면(§9의 18번) networkx와 견준다(그 구현은 Python이라 수십 배 차이가 예상된다; 속도의 기준이 되는 것은 C로 된 igraph다).
- **나중에 볼 것**: 단일 원점의 Δ-stepping, Borůvka와 Kruskal의 갈림점, 너비 우선의 방향 바꾸기(위로 훑기; 결과의 트리가 달라지므로 순회가 아닌 거리 계산에만).

---

## 6. 층과 파일
- **층**: `graph/`는 `sparse/`(`format.hpp`의 표준 꼴 CSR 읽기와 전치의 계수 정렬), `array/`, `base/`(`pairwise.hpp`), `parallel/` 위에 있다. 다른 영역은 `graph/`를 모른다. Python 층의 `stardust.graph`는 `stardust.sparse`(입력, `csr_array` 결과, `LinearOperator`)를 쓴다.
- `bench/areas.py`에 `graph`를 `spatial` 뒤, `plot` 앞에 넣는다(plot은 맨 위 층이다). 같은 때에 설계하는 영역(interpolate, fft)과의 순서는 GR0 마일스톤에서 먼저 커밋하는 쪽부터 정한다. ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표에 한 줄씩 더한다(GR0 마일스톤).
- C++의 `graph/` 폴더(csrc 아래, GR1 마일스톤부터; 안)
  - `graph.hpp`(GR1): CSR 보기(`View`), 배열 인자의 확인(`view_arg`), sparse의 전치를 감싼 `Transposed`, 대칭 확인, 무향 무게(GR29), 표식 상수(−9999), 병렬 기준값. 합친 무향 CSR과 무게 검사는 그것을 쓰는 마일스톤에서 더한다
  - `input.cpp`(GR1): `_dense_csr`(밀집 → CSR, `csgraph_from_dense`의 규칙도), `_to_dense`, `view_arg`, 전치, 대칭 확인
  - `traversal.cpp`(GR1): 너비·깊이 우선 순서와 트리, 약한·강한 연결 요소, 위상 정렬
  - `paths.hpp`(GR2): 길 함수가 같이 쓰는 것: 무향 읽기(합친 CSR), 변 찾기, 닿는 노드, 음수 순환의 공통 검사(GR35), 탐색의 작업 공간(4진 색인 힙; GR39), 단일 탐색(Dijkstra, 너비 우선, 큐 BF, 퍼텐셜 J), 선행 노드 규칙
  - `paths.cpp`(GR2): `_scan`(무게 검사), `_paths`(원점 태스크, 밀집 Dijkstra), `_floyd_warshall`(블록 FW와 선행 노드 커널), `_construct`, `_reconstruct`
  - `yen.cpp`(GR2): `_yen`(갈래 탐색, 후보의 전순서)
  - `tree.cpp`(GR3): 최소 신장 트리(기수 정렬과 Kruskal; GR43)
  - `flow.cpp`(GR3): Dinic, Edmonds–Karp, 잔여 CSR
  - `matching.cpp`(GR3): Hopcroft–Karp, 두 배정의 공용 풀이(LAPJV의 시작과 최단 증가 경로; 밀집 훑기와 희소 힙)
  - `ordering.cpp`(GR4): 역 Cuthill–McKee, `structural_rank`, 라플라시안의 차수와 원소
  - `structure.cpp`(GR5): 단절점, 다리, 이중 연결 요소(한 번의 깊이 우선), 코어 수(Batagelj–Zaversnik)
  - `centrality.cpp`(GR5): PageRank, 매개 중심성
  - `graph_mod.hpp`: 파일마다의 `bind_*`
- 바인딩 `bind/graph_bind.cpp`: `_core.graph` 하위 모듈(GR1 마일스톤부터).
- `src/stardust/graph/` (하위 패키지 없음): `__init__.py`(공개 이름), `_common.py`(입력을 CSR로, dtype·모양·무게·노드 인자의 검사와 메시지; GR1, `NegativeCycleError`·`count`·`room`은 GR2), `_traversal.py`, `_paths.py`(GR2), `_tree.py`, `_flow.py`(`MaximumFlowResult`), `_matching.py`, `_ordering.py`(RCM, `structural_rank`, `laplacian`), `_structure.py`, `_centrality.py`, `_convert.py`(`csgraph_from_dense`, `csgraph_to_dense`). namedtuple 결과는 `stardust/_results.py`에 있다.

---

## 7. 참값과 검증
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다. 긴 참값 생성은 프로세스 10개까지, 메모리를 지켜보며(RSS가 평평한지) 돈다.

### 7.1 시험 그래프 (`tests/graph/graphs.py`)
- **모양**: 빈 그래프(n = 0, 1), 고립 노드, 경로, 순환, 별, 완전 그래프, 격자(2·3차원; 동점이 아주 많다), 사다리(경로 수 2^(n/2)), 이분 그래프(직사각), DAG, 무작위 유향·무향(Erdős–Rényi, 차수 분포가 긴 꼬리), 여러 요소, 자기 고리, 한 방향만의 변(무향 규칙), 두 방향 무게가 다른 변.
- **무게**: 정수(작은 범위라 동점이 많음; 2⁵³ 둘레의 큰 정수), 2진 유리수, 무작위 float(10⁻⁸–10⁸의 넓은 범위, 흡수가 생기는 무게), 저장된 0(무게 0의 변과 0 무게 순환), 음수(순환 없음, 닿는 음수 순환, 닿지 않는 음수 순환, 정확한 합 0의 순환), bool.
- **경계**: 밀집의 0·NaN·±inf, 희소의 저장된 NaN·±inf(에러), 범위 밖 노드, `limit`의 경계 값과 그 앞뒤 double, 정확히 같은 거리의 여러 경로.
- **흐름과 매칭**: 정수 용량(int64 경계 근처), 완전 매칭이 없는 경우, 동점이 많은 비용, 직사각.

### 7.2 참값 (`tools/graph_ref.py` → `tests/graph/ref/`, `tests/reffile.py`의 npz)
- 최단 경로: 정확한 유리수(Fraction) Dijkstra·Bellman–Ford(작은 그래프, n ≤ 200)의 참 거리와 §3.2의 규칙을 정확히 계산한 선행 노드(정수 무게), 일반 무게는 참 거리와 참 최단 경로의 변 수·Σ|w|(계약의 H와 크기).
- `yen`: 고리 없는 경로 전부의 나열과 정확한 정렬(n ≤ 12).
- 매칭과 배정: 나열(n ≤ 8)과 정확한 쌍대 확인.
- PageRank: Arb `arb_mat.solve`의 참값(n ≤ 300, 충분한 정밀도에서 반지름이 ulp보다 작아질 때까지).
- 매개 중심성: 정확한 유리수 Brandes(n ≤ 100).
- `laplacian`: Arb의 차수와 정규화 원소.
- 구조 함수는 참값 파일이 아니라 테스트 안의 Python 정의 구현과 견준다(빠르고 작다).

### 7.3 테스트 (`tests/graph/`)
- 파일: `test_api_surface.py`, `test_input.py`(변환 규칙, dtype, 밀집의 변 규칙, 에러와 메시지, 노드 인자, 옵션, 호출 모양), `test_traversal.py`, `test_components.py`, `test_toposort.py`, `test_convert.py`, `test_threads.py`(GR1; 정의 구현은 `definitions.py`, 시험 그래프의 입력 꼴 넷·셋은 `forms.py`), `test_paths.py`, `test_paths_reference.py`, `test_paths_methods.py`(D, BF, J, auto의 바이트가 같음, FW는 계약 안; hypothesis), `test_yen.py`(GR2), `test_mst.py`, `test_flow.py`, `test_matching.py`, `test_layouts.py`(GR3; `test_layouts.py`는 밀집 입력의 메모리 꼴, GR48), `test_ordering.py`, `test_laplacian.py`(GR4), `test_structure.py`, `test_centrality.py`(GR5), `test_errors.py`, `test_threads.py`(1, 2, 3, 7, 10스레드의 같은 바이트).
- hypothesis는 tests/README.md의 프로필을 쓴다. 생성기는 n, 밀도, 무게의 종류(정수·float·0·음수), 대칭 여부, 요소 수를 고른다.
- scipy와는 그 자리에서 견준다: 동점이 없는 그래프에서 같은 결과(거리는 계약 안, 선행 노드·순서·레이블은 같음), 동점이 있으면 같은 분할·같은 거리·같은 흐름 값·같은 매칭 크기.

### 7.4 scipy의 기록 (`tools/graph_probe.py` → `tests/graph/data/`)
- 이 초안의 탐침(입력 규칙, 저장된 0·NaN·inf, 음수 무게와 순환, auto의 규칙, 동점, `limit`·`indices`·`min_only`, 순회 순서, 레이블, MST의 경계, 흐름의 int32 자르기, 매칭의 0, 라플라시안의 꼴, `yen`, 변환 도구, 시간)을 도구 하나로 옮긴다(GR0 마일스톤). 내부 에러를 삼키는 호출(`breadth_first_order(G, 5)`)도 기록한다.
- 기록 `scipy.json`(scipy 1.18.1): 문서의 `[scipy와 다름]` 하나하나와 시험 그래프의 결과. `networkx.json`(networkx 3.7; GR24): scipy에 없는 일곱의 결과(사전 최소 위상 순서, 단절점, 다리, 이중 연결 요소의 변, 코어 수, PageRank, 매개 중심성)와 가장자리의 동작(자기 고리의 `core_number`는 에러, 수렴하지 못한 PageRank). 모든 기록은 그 그래프의 `digest`(tests/graph/graphs.py)를 가진다.

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/graph.py`): 스윕 `paths`(일반 무게의 거리 계약 단위, 비교 패키지 scipy의 D·FW), `pagerank`, `betweenness`, `laplacian`, `assignment`(일반 비용의 계약 단위; 비교 패키지 scipy `linear_sum_assignment`·`min_weight_full_bipartite_matching`; GR3 마일스톤에서 만들었다: 참값의 실수 두 점과 정확한 유리수 풀이와 견준 무작위·거의 동점 문제).
- **속도** (`bench/speed/cases/graph.py`): 사례마다 마일스톤과 필요한 이름을 적어 두고 그 마일스톤이 끝나면 사례가 된다(sparse·spatial의 사례 파일과 같은 틀). §5의 크기 전부와 1·10스레드. 비교 대상은 scipy 1.18.1(설치하면 networkx)이고 결과는 거리, 레이블, 흐름 값을 견준다.
- **골든** (`tests/golden/cases_graph.py`, 기록 `tests/golden/data/graph.json`): 모든 공개 함수 × 시험 그래프의 대표, 방법 다섯, 틀린 호출. 1·3·10스레드의 결과가 모두 같아야 한다. 큰 케이스는 병렬 길을 탈 크기로 두되 워커 하나의 메모리가 1 GB를 넘지 않게 한다.

---

## 8. 마일스톤
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| GR0 준비 | API 문서 초안(`docs/api/graph.md`: 모든 공개 이름의 시그니처, scipy·networkx → sd 대응표, §9의 답), 영역 등록(`bench/areas.py`, ARCHITECTURE, API.md의 색인), 시험 그래프(§7.1), 참값 도구(§7.2), 탐침 도구와 scipy 기록(§7.4), FW 길의 두 안(원점마다 밀집 Dijkstra 대 블록 min-plus FW)의 측정(§9의 2번), 속도 사례의 틀 | 문서와 도구가 커밋되고 탐침 기록이 있다. §9의 답이 §1.1의 결정으로 옮겨졌다 — 완료·승인(위임, 2026-11-23) |
| GR1 순회와 연결 요소 | 입력 변환(§2.1, §3.1), `csgraph_from_dense csgraph_to_dense`, `breadth_first_order depth_first_order breadth_first_tree depth_first_tree`, `connected_components`, `topological_sort` | 정의 구현과 정확히 같음, scipy와의 차이가 문서대로, 스레드마다 같은 바이트, 속도 사례 — 완료·승인(위임, 2026-11-23) |
| GR2 최단 경로 | `shortest_path dijkstra bellman_ford johnson floyd_warshall yen construct_dist_matrix reconstruct_path NegativeCycleError`, §3.2의 고정점과 선행 노드 규칙 | D, BF, J, auto가 바이트까지 같음, FW는 계약 안, 정수 무게의 정확한 참값과 같음, 일반 무게의 계약, 음수 순환 규칙, scipy와의 속도 사례 — 완료·승인(위임, 2026-11-23) |
| GR3 신장 트리·흐름·매칭 | `minimum_spanning_tree maximum_flow maximum_bipartite_matching min_weight_full_bipartite_matching`, `linear_sum_assignment`(§9의 1번) | 유일한 MST, 흐름 값과 최소 절단이 같음, 정수 비용의 정확한 최적, 일반 비용의 계약, 속도 사례 — 완료·승인(위임, 2026-11-23) |
| GR4 순서·구조적 랭크·라플라시안 | `reverse_cuthill_mckee structural_rank laplacian`(`form='lo'` 포함) | scipy와 같은 순열(동점 규칙 확인), Arb 계약, 속도 사례 |
| GR5 더한 알고리즘 | `articulation_points bridges biconnected_components core_number pagerank betweenness_centrality` | 정의 구현과 정확히 같음(구조 넷), PageRank·매개 중심성의 Arb·정확한 유리수 계약, 스레드마다 같은 바이트, 속도 사례 |
| GR6 성능과 일관성 점검 | 모든 공개 함수를 크기마다 scipy(와 설치했다면 networkx)와 견주어 고치기, 갈래의 문턱(auto, 병렬 합집합–찾기, Borůvka, 원점 태스크), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다 |
| GR7 문서 | `docs/api/graph.md`의 확정, 이 문서를 구현대로 고치기(잰 값, 정한 상수, 파일) | 문서가 코드와 맞고 `test_api_surface`가 통과한다 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 정하고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달한다. GR1 마일스톤이 입력 변환을 만들므로 맨 앞이다. GR2–GR5 마일스톤은 GR1 마일스톤에만 기대어 서로 독립이라 나란히 할 수 있다(GR5의 매개 중심성은 GR2의 Dijkstra를 쓰므로 그 뒤). 기존 영역의 결과는 비트까지 같아야 한다(이 영역은 다른 영역을 고치지 않는다).

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
