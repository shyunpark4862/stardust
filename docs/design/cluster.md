# 군집 영역 설계 (sd.cluster)

> 상태: **확정** (2026-11-24). 큰 결정(§1.1 CL1–CL3)은 사용자가 정했다. §9의 질문 가운데 범위(미루는 것과 넣는 것: 19·22·27·28·29·32·36·37번), 결정성(6번), 라벨 번호(3번), 지표의 자리(10번)는 사용자가 (a)로 정했고(CL6, CL9, CL13, CL22, CL25, CL30–CL32, CL35, CL39, CL40), 연결법의 기본값(13번)은 메인 세션이 (b)로 정했으며(CL16, 사용자 위임), 나머지는 추천안 (a)다(사용자 위임). CL0 마일스톤(준비)을 마쳤다(2026-11-24): 결과의 공통 기반, 시험 자료, 참값, 비교 대상의 기록, 시제품의 측정과 그로 정한 CL41–CL48(추천안, 사용자 위임)이 있고 공개 이름은 아직 없다. k-평균의 거리 꼴(§9의 38번)은 사용자가 (a)로 정했다(CL49, 2026-11-26). CL1 마일스톤(계층)을 마쳤다(2026-11-27): `linkage`, `agglomerative`, `divisive`, `HierarchyResult`, scipy의 계층 도구와 `dendrogram`의 자료가 공개됐고, 그 세부를 CL50–CL57(추천안, 사용자 위임)과 CL58(메인 결정, 사용자 위임)로 정했다(§1.1; 높이 계약의 증명과 NN-chain 동점 검출의 충분함은 §4.1). CL2 마일스톤(분할)을 마쳤다(2026-11-27): `kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `kmedoids`(PAM·CLARA), `fuzzy_cmeans`와 결과들이 공개됐고, 그 세부를 CL59–CL66(추천안, 사용자 위임)으로 정했다(퍼지 c-평균의 `tol` = 10⁻⁵와 PAM의 `max_iter` = 1000을 확정; 경계 가속과 CL49 거르기의 증명은 §4.2). CL3 마일스톤(밀도)을 마쳤다(2026-11-27): `dbscan`, `hdbscan`, `optics`와 결과들이 공개됐고, 그 세부를 CL67–CL74(추천안, 사용자 위임)로 정했다(HDBSCAN은 같은 높이의 병합을 한꺼번에 줄여 입력 순서와 무관하고, 같은 높이가 없는 트리에서 scikit-learn과 같다; 계약과 증명은 §4.4). CL4 마일스톤(혼합)을 마쳤다(2026-11-27): `gaussian_mixture`와 `GaussianMixtureResult`가 공개됐고, 그 세부를 CL75–CL81(추천안, 사용자 위임)로 정했다(EM은 sklearn `GaussianMixture`의 단계와 기본값을 차의 꼴 이차형식과 평균을 뺀 2차 모멘트로 계산해, 같은 시작에서 sklearn과 같은 반복 수로 같은 고정점에 닿는다; 주어진 모수·책임도의 계약과 증명은 §4.5). 1스레드의 큰 p(약 35 이상)에서 full·tied의 E 단계가 같은 정의의 Accelerate 꼴보다 1.5배 넘게 느린 것은 §9의 40번으로 묻는다. CL5 마일스톤(지표와 k 고르기)과 CL6 마일스톤(그 밖)을 마쳤다(2026-11-27): sklearn 이름의 지표 17개와 `HomogeneityResult`, `gap_statistic`과 `GapResult`, `gower_distance`, `affinity_propagation`과 `AffinityPropagationResult`가 공개됐고, 그 세부를 CL82–CL94(추천안, 사용자 위임)로 정했다(ARI·RI·FM은 정확한 정수 셈의 올바른 반올림, AMI의 기대 MI는 초기하 비의 점화식; gap은 R `clusGap`·`maxSE`와 같은 뜻이고 PAM이면 R과 같은 log W; Affinity Propagation은 잡음 없이 열쇠 순서로 sklearn과 같은 연산이라, 잡음이 0인 sklearn과 같은 exemplar·라벨·반복 수; 계약과 증명은 §4.6). 임시로 두었던 이름은 정했다: `gap_statistic`의 `reference`·`select`는 R 이름의 소문자 꼴(CL87), `affinity_propagation`의 유사도 입력은 sklearn의 `affinity='precomputed'`(CL91). 이 문서의 잰 값은 `tools/cluster_probe.py`의 것이다(기록은 `tests/cluster/data/`의 `scipy.json`·`sklearn.json`·`r.json`, 시제품의 측정은 `prototype.json`; 설계 때의 탐침 `site/notes/cluster-design/probe_behavior.py`를 옮겨 넓혔다).

`sd.cluster`(점들을 무리로 나누기: 분할, 계층, 밀도, 혼합 모형, 그리고 나눈 결과를 재는 지표)의 안과 내부 구조를 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처는 [docs/api/cluster.md](../api/cluster.md)에 있다.
- 비교 대상은 `.venv`의 scikit-learn 1.9.1(`sklearn.cluster`의 `KMeans`·`MiniBatchKMeans`·`BisectingKMeans`·`DBSCAN`·`HDBSCAN`·`OPTICS`·`AgglomerativeClustering`·`AffinityPropagation`·`SpectralClustering`·`MeanShift`·`Birch`, `sklearn.mixture`, `sklearn.metrics`의 군집 지표), scipy 1.18.1(`scipy.cluster.hierarchy`, `scipy.cluster.vq`), R 4.6.2의 `stats`(`kmeans`, `hclust`, `cutree`, `cophenetic`)와 `cluster` 2.1.8.3(`pam`, `clara`, `fanny`, `agnes`, `diana`, `daisy`, `silhouette`, `clusGap`)이다. R의 `mclust`, `dbscan`, `fastcluster`, `apcluster`, `kernlab`, `e1071`, `fpc`, `NbClust`와 Python의 원본 `hdbscan`, `scikit-learn-extra`(KMedoids)는 설치되어 있지 않다. 설치하지 않는다(CL3). 그래서 mclust의 공분산 꼴, R `dbscan`의 HDBSCAN, `e1071::cmeans`는 공개 문서로만 뜻을 맞추고 참값으로만 지킨다(§7.4).
- 바닥은 배열(`csrc/array/`), 공용 커널(`csrc/base/`의 `buffer.hpp`·`pairwise.hpp`), 스레드풀(`csrc/parallel/`), spatial의 거리 커널과 KD 트리(`csrc/spatial/metric.hpp`·`kdtree.hpp`·`neighbors.hpp`), graph의 합집합–찾기와 Kruskal의 키 순서(graph.md GR43)다. Python 층은 `sd.random`(`Generator`), `sd.linalg`(`LinAlgError`), `sd.special`(`logsumexp`), `sd.stats`(`MultivariateNormal`의 이름과 규칙), `sd.frame`, `stardust._summary`, `stardust._calls`를 쓴다.
- 이 문서의 잰 값은 Apple M4 Pro, macOS 26.6.2에서 본 것이다. 다른 작업과 기기를 나눠 썼고 부하 평균은 1.2–2.6이었다. 시간은 한 번씩 잰 어림이다(§5).
- u는 단위 반올림 2⁻⁵³, γ_k = ku/(1 − ku)다(spatial.md, smooth.md와 같은 기호). n은 점 수, p는 차원(변수 수), k는 군집 수, m은 평가점 수다. 계층 군집의 Z는 scipy의 연결 행렬((n − 1, 4): 합친 두 군집, 높이, 크기)이다.

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-24)
"CLn"이 결정 번호와 마일스톤 이름에 함께 쓰이므로 마일스톤은 언제나 "CL2 마일스톤"처럼 적는다.

| 번호 | 결정 |
|---|---|
| CL1 API의 꼴 | 함수가 결과 객체를 돌려준다: `kmeans(X, k)` → `KMeansResult`(`.labels`, `.centers`, `.predict(X_new)`). scikit-learn식 추정기 클래스(`KMeans(...).fit(X)`)가 아니다. 회귀·평활과 같은 꼴이다: `summary()`, 한 줄 `repr`, 읽기 전용 배열 필드, `_calls`의 호출 모양 에러 (사용자 결정 2026-11-24) |
| CL2 영역 나누기 | 군집은 `sd.cluster`, 차원 축소·임베딩(PCA, t-SNE, UMAP 등)은 나중의 `sd.embed`, 분류기와 그 회귀판(트리, 포레스트, k-NN, SVM, 부스팅의 분류와 회귀)은 나중의 `sd.classify`다. 영역의 순서는 cluster → embed → classify다 (사용자 결정 2026-11-24) |
| CL3 비교 대상 | Python(scikit-learn 1.9.1, scipy 1.18.1)과 R(설치된 `stats`, `cluster`)을 모두 조사하고 견준다. 설치되지 않은 R 꾸러미(mclust, dbscan, fastcluster, apcluster, kernlab, e1071, fpc, NbClust)는 설치하지 않는다 (사용자 결정 2026-11-24) |
| CL4 결과의 꼴과 계층 군집의 두 얼굴 | §9의 1번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL5 군집 수와 자료의 인자 | §9의 2번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL6 라벨의 번호 | §9의 3번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL7 입력 순서와 무관한 분할 | §9의 4번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL8 난수 (`rng`) | §9의 5번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL9 결정성의 범위와 BLAS | §9의 6번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL10 k-평균의 거리 계산 | §9의 7번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL11 NaN | §9의 8번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL12 거리 입력 | §9의 9번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL13 지표의 자리 | §9의 10번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL14 지표의 이름 | §9의 11번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL15 연결법과 입력 규칙 | §9의 12번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL16 연결법의 기본값 | §9의 13번 (b)를 따른다: 기본값은 scipy의 `'single'`이다. 추천안 (a)(기본값 없음)를 바꿨다: API.md §4의 규칙 2(기본값은 기준 라이브러리를 따른다)와 맞춘다 (메인 결정, 사용자 위임 2026-11-24) |
| CL17 계층 군집의 동점과 높이 | §9의 14번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL18 행렬 없는 계층 군집 | §9의 15번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL19 역전이 있는 트리의 자르기 | §9의 16번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL20 scipy 계층 도구의 범위 | §9의 17번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL21 덴드로그램 | §9의 18번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL22 분할식 계층과 연결 제약 | §9의 19번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL23 k-평균의 알고리즘 | §9의 20번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL24 k-평균의 초기화·반복·빈 군집 | §9의 21번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL25 미니배치·이분 k-평균 | §9의 22번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL26 k-medoids | §9의 23번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL27 퍼지 군집 | §9의 24번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL28 DBSCAN의 뜻 | §9의 25번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL29 HDBSCAN의 기준 | §9의 26번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL30 OPTICS | §9의 27번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL31 Mean Shift | §9의 28번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL32 혼합 모형의 공분산 꼴 | §9의 29번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL33 EM의 세부 | §9의 30번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL34 혼합 모형의 k 고르기 | §9의 31번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL35 베이즈 혼합 | §9의 32번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL36 지표의 계산과 기본값 | §9의 33번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL37 gap 통계량 | §9의 34번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL38 Gower 거리 | §9의 35번 (a)를 따른다 (추천안, 사용자 위임 2026-11-24) |
| CL39 스펙트럼 군집 | §9의 36번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL40 Affinity Propagation과 그 밖 | §9의 37번 (a)를 따른다 (사용자 결정, 2026-11-24) |
| CL41 난수의 원소 번호 | 호출 하나가 `rng`에서 64비트 두 단어를 받아 Philox4×64-10의 키로 쓰고(CL8), 시작(CLARA의 표본, gap의 참조 자료) r의 t번째 수는 원소 r·2⁴⁰ + t다(r < 2²⁴, t < 2⁴⁰: 10⁷점의 미니배치 100번도 들어간다). 이 영역의 다른 함수를 안에서 부르는 곳(gap의 군집, 혼합 모형 시작의 k-평균)은 하위 호출의 두 단어를 부모 흐름의 정해진 시작 번호에서 받는다. 회귀 `rlm(method='MM')`의 j·2³² + t와 같은 꼴이고 나눈 자리만 다르다 (추천안, 사용자 위임 2026-11-24) |
| CL42 경계 가속의 여유 | Hamerly·Elkan의 상한 U(올림)와 하한 L(내림)을 참 거리의 한계로 지키고, γ = 2(p + 4)u로 L > U(1 + 2γ)(1 + 4u)일 때만 건너뛴다. 그러면 건너뛴 중심의 계산한 차의 제곱합이 가장 가까운 중심의 것보다 크다는 것이 증명되어 Lloyd와 같은 라벨과 같은 비트의 중심이다. 시제품에서 여유의 계수 0, 1, 4, 16이 건너뛴 비율을 바꾸지 않았다(§5) (추천안, 사용자 위임 2026-11-24) |
| CL43 2의 거듭제곱 척도 | 제곱 거리를 쓰는 커널(k-평균의 거리, 높이의 재귀, 혼합 모형의 이차형식)은 자료의 max\|x\|가 [2⁻²⁵⁰, 2²⁵⁰] 밖이면 자료에 2의 거듭제곱을 곱해 [1, 2)로 옮겨 계산하고 결과(중심, 높이, 관성)를 되돌린다(정확한 곱셈이라 범위 안의 자료와 같은 비트다). 시제품에서 비정상 수 좌표(subnormal/k3p2)의 제곱이 0으로 아래넘쳐 60점 중 40점의 라벨이 틀렸다(§5) (추천안, 사용자 위임 2026-11-24) |
| CL44 결과의 공통 기반 | `_results.py`의 `ClusterResult`: 필드 `labels`·`n_clusters`·`nobs`·`omitted`, 처음 읽을 때 한 번 계산하는 필드, 쓰기와 지우기는 `AttributeError`, 배열은 읽기 전용. `summary()`는 회귀 상자의 머리(왼쪽: Method, No. Observations, No. Clusters, 있을 때만 Noise points와 Omitted rows (NaN), 목적값, Iterations, Converged; 오른쪽: 조절 모수)와 군집마다 한 줄의 표(cluster, Size, 하위 클래스의 열; 20개를 넘으면 양 끝 5개씩), `repr`는 한 줄, 메서드의 호출 모양 에러는 자기 이름으로 시작한다 (추천안, 사용자 위임 2026-11-24) |
| CL45 계층 군집의 높이 | CL17의 "최종 트리에서 정의대로 다시 계산"을 최종 트리 위의 Lance–Williams 재귀로 정한다: 병합 (P₁ ∪ P₂, Q)의 값은 (P₁, Q)와 (P₂, Q)의 값과 P의 높이로 그 연결법의 식이고, 점 x와 군집 Q는 Q를 같은 식으로 점까지 내려간다(점끼리는 spatial의 유클리드 거리, ward·centroid·median은 그 제곱; 자식은 Z의 순서). 트리와 자료만이 연산을 정해 고른 알고리즘(NN-chain, 일반, 행렬 없는 길)과 무관한 비트이고, 모든 점 짝이 공통 조상에서 한 번씩만 쓰여 O(n²p) 시간·O(n) 메모리다. 시제품에서 모든 참값 집합·연결법이 2 ULP 안이었고(scipy의 누적은 3 ULP), 정의대로의 합은 ward에서 5.2×10⁸ ULP까지 틀렸다(§5) (추천안, 사용자 위임 2026-11-24) |
| CL46 NN-chain의 동점 검출 | 이웃 찾기 한 번에서 가장 작은 거리와 정확히 같은 다른 후보(사슬의 앞 점 포함)를 만나면 표시하고, 표시된 입력만 일반 알고리즘으로 다시 푼다(CL17의 "계산한 높이에 같은 값이 둘 이상" 대신). 시제품에서 사슬이 정의의 트리와 다른 경우는 모두 표시됐고(128가지 중 4가지), 검사의 비용은 잴 수 없었다(§5). 이 검출이 충분하다는 증명은 §4.1에 있다(CL1 마일스톤) (추천안, 사용자 위임 2026-11-24) |
| CL47 Prim과 Borůvka의 문턱 (초안) | 관측값의 단일 연결과 HDBSCAN의 최소 신장 트리는 p ≤ 5이고 n ≥ 2048이거나 p ≤ 8이고 n ≥ 32768이면 KD 트리 위의 Borůvka, 그 밖은 행렬 없는 Prim이다(모양만으로 고른다). 시제품의 단일 트리 Borůvka로 잰 문턱이고 CL1 마일스톤이 이중 트리 Borůvka로 다시 잰다 (추천안, 사용자 위임 2026-11-24). → CL54가 바꿨다 |
| CL48 k-평균 중심의 합 | 점을 열쇠의 순서(CL7)로 4096점 덩이로 나눠 덩이마다 군집별 부분합을 내고 덩이 순서의 짝 합으로 합친다. 그래서 중심의 비트가 입력 순서와도 무관하다. 정렬한 사본을 만들지 색인으로 모을지는 CL2 마일스톤이 잰다 (추천안, 사용자 위임 2026-11-24) |
| CL49 k-평균의 거리 꼴 | 차의 꼴 Σ(x − c)²이 정의다. 1스레드의 큰 p·k를 위해 증명된 오차 한계의 거르기(전개식으로 후보를 거르고 한계 안의 후보만 차의 꼴로 다시 잰다)를 CL2·CL7 마일스톤에서 넣는다. 결과는 차의 꼴과 같은 비트이고 CL9(스레드와 무관한 같은 바이트)를 지킨다 (사용자 결정 2026-11-26, §9의 38번 (a)) |
| CL50 단일 연결의 동점 | 최소 신장 트리의 변을 (거리, 작은 열쇠, 큰 열쇠)로 정렬해 합친다. 같은 거리의 변이 둘 이상이면 그 변들이 잇는 군집들의 묶음마다 정의(CL17)대로 푼다: 묶음 안에서 열쇠가 가장 작은 군집이, 그 거리에 정확히 있는 이웃 군집을 열쇠 순서로 하나씩 흡수한다(이웃은 그 거리의 점 짝이고 KD 트리의 반경 찾기나 훑기로 찾는다). 같은 거리의 두 군집 사이에 최소 신장 트리의 변이 없을 수 있어서 트리의 변만으로는 정의의 트리가 나오지 않는다(§4.1의 증명). 동점이 없으면 Kruskal의 차례가 곧 정의의 트리다 (추천안, 사용자 위임 2026-11-26) |
| CL51 행의 순서와 단조 연결법의 높이 | Z의 행은 최종 높이 위의 primitive 차례다: 두 자식이 이미 있는 병합 가운데 (높이, 작은 열쇠, 큰 열쇠)가 가장 작은 것부터. 정확한 산술에서 primitive 알고리즘의 차례와 같고, 고른 알고리즘(NN-chain, 일반, 행렬 없는 길)과 무관하게 트리와 자료만으로 정해진다. 단조인 연결법(single, complete, average, weighted, ward)의 높이는 트리 위로 누적 최댓값을 취한다(CL17의 "줄어든 자리는 앞 높이로 맞춘다"를 트리 위에서 정한 것; 행의 높이가 줄지 않는다) (추천안, 사용자 위임 2026-11-26) |
| CL52 DIANA의 세부 | R `diana`의 나누기(갈라진 무리를 다른 점들과의 평균 거리가 가장 큰 점으로 시작하고, 나머지에서 (나머지와의 평균 거리 − 갈라진 무리와의 평균 거리)가 0보다 큰 점 가운데 그 값이 가장 큰 점을 하나씩 옮긴다; 동점은 열쇠)의 트리를 그대로 Z로 낸다(높이는 지름, 행은 CL51). 지름이 같은 나누기가 이어지면 R의 `merge`(배너에서 만든 것)는 다른 중첩을 보인다: sd는 실제로 나눈 트리다 `[R과 다름]`. `coefficient`는 R의 ac·dc 공식(점마다 1 − (처음 합쳐지는 높이)/(마지막 병합의 높이)의 평균)이고 마지막 높이가 0이면 NaN이다 (추천안, 사용자 위임 2026-11-26) |
| CL53 최적 잎 순서 | Bar-Joseph 외(2001)의 동적 계획으로 정확한 최소를 낸다. 뿌리의 첫 자식(Z[-1, 0])은 왼쪽에 둔다. 같은 비용의 선택은 열쇠가 작은 끝점((u, w), 안에서는 (m, k))이다. scipy 1.18.1의 `optimal_leaf_ordering`은 최소가 아닌 순서를 낼 때가 있다(n = 4에서도; 테스트가 모든 순서와 견준다) `[scipy와 다름]` (추천안, 사용자 위임 2026-11-26) |
| CL54 KD 트리 Borůvka와 길의 문턱 | Borůvka는 점마다 KD 트리를 뿌리부터 찾는다(가까운 자식 먼저; 자기 성분만 든 마디와, spatial의 증명된 한계로 반경 밖인 마디를 자른다; 반경은 자기 최선과 성분이 지금까지 찾은 최선 중 작은 것). 이중 트리(질의 마디 × 참조 마디)도 만들어 쟀지만 p ≥ 3에서 1.2–6배 느려 쓰지 않는다(`_core.cluster._mst`의 측정용으로만 남긴다). 문턱(CL47 초안을 바꾼다): p ≤ 3이고 n ≥ 1024, p ≤ 5이고 n ≥ 2048, p = 6이고 n ≥ 4096, p ≤ 8이고 n ≥ 16384면 Borůvka, 그 밖은 Prim이다. 1스레드에서 Prim과 비슷하거나 빠르고 10스레드에서 여러 배 빠른 곳부터다(§5, tests/cluster/data/prototype.json의 "mst/time") (추천안, 사용자 위임 2026-11-26) |
| CL55 높이의 계약 | CL45의 재귀는 병합 (A, B)에서 열쇠가 작은 자식을 바깥(먼저 펼치는 쪽)으로 한다(Z의 번호는 입력 순서에 달려 입력 순서와 무관한 비트를 깬다). 계약은 §4.1에서 증명한 꼴 \|Ĥ − H*\| ≤ γ_{cμ+p+2}·V⁺다: H는 재귀의 눈금(ward·centroid·median은 높이의 제곱), V⁺는 뺄셈을 덧셈으로 바꾼 재귀(average·weighted는 H* 그 자체), μ는 그 병합 뒤의 반올림 사슬의 길이, c는 한 단계의 반올림 수(average 3, weighted 1, centroid·ward 5, median 2). single·complete의 높이는 spatial의 거리 그 자체다. CL0 마일스톤의 초안 C_HEIGHT·u·(p + d)·h*를 바꾼다(그 꼴은 뺄셈의 상쇄를 담지 못한다) (추천안, 사용자 위임 2026-11-26) |
| CL56 길의 선택 | 관측값과 유클리드의 single·ward·centroid·median은 언제나 행렬 없는 길이다(single: Prim이나 Borůvka; 나머지: 군집의 무게 중심과 크기를 열 단위로 쥐고 NEON 두 칸씩 훑는다). complete·average·weighted와 그 밖의 거리는 열쇠 순서로 놓은 점들의 `sd.spatial.pdist`(같은 비트) 위의 행렬 길이고, 미리 계산한 거리는 그 행렬이다(열쇠는 입력 색인). CL43의 2의 거듭제곱은 ward·centroid·median의 관측값에만 쓴다(나머지의 거리는 spatial의 조심 길이 맡는다). 길을 강제하는 `_core.cluster.linkage`의 `path`는 테스트의 두 길 비교용이다 (추천안, 사용자 위임 2026-11-26) |
| CL57 scipy 도구의 세부 | (1) 높이로 자르기(`cut(height=)`, `cut_tree(height=)`, `fcluster(criterion='distance')`)는 h **이하**의 병합을 남긴다(R `cutree(h=)`, scipy `fcluster`와 같고 scipy `cut_tree(height=)`는 h 미만이다 `[scipy와 다름]`). (2) `cut_tree(n_clusters=)`는 행의 차례로 되돌린다(같은 높이에서 scipy의 너비 우선 차례와 다를 수 있다 `[scipy와 다름]`; R `cutree(k=)`와 같다). (3) `fcluster`의 maxclust 기준은 t가 1 이상의 정수다(scipy는 t < 1에서 정해지지 않은 답을 낸다; t ≥ n은 scipy처럼 1 … n). (4) `inconsistent`의 표준편차는 평균을 뺀 두 번째 훑기다(scipy는 한 번 훑기라 높이가 하나일 때 0 대신 10⁻⁹ 꼴을 낸다). (5) `is_valid_linkage`는 정수 번호, 개수의 합, 유한한 높이도 본다(scipy는 보지 않는다). (6) `leaders`는 scipy의 (L, M) 튜플이다(int64). (7) `dendrogram`은 icoord·dcoord를 (m, 4) 배열로, 색을 묶음 번호(0부터 돌려 쓰지 않는다; 문턱 이상은 −1)로, 잎의 색을 그 잎을 잇는 가지의 색으로 준다(scipy는 좌표로 잎을 다시 찾아 높이 0의 가지에서 다른 잎을 칠할 수 있다). (8) `HierarchyResult`에는 분할이 하나가 아니라 `labels`·`n_clusters`가 없고 `cut`이 분할을 준다. `nan_policy='omit'`의 Z는 남은 행의 번호이고 `cut`이 뺀 행에 −1을 넣는다. (9) `linkage`의 1차원 입력(압축 꼴)에 유클리드 아닌 `metric`은 `ValueError`다(scipy는 무시한다). (추천안, 사용자 위임 2026-11-26) |
| CL58 `CophenetResult`의 공개 | `cophenet_correlation`의 결과 namedtuple `CophenetResult(correlation, distances)`를 `__all__`과 API 문서의 서명 묶음에 더해 공개한다(결과 namedtuple은 공개하는 관례: smooth의 `DensityResult`, 이 영역의 `DendrogramResult`·`HomogeneityResult`) (메인 결정, 사용자 위임 2026-11-27) |
| CL59 k-평균의 거리와 동점 | CL10·CL49의 "차의 꼴"을 spatial의 `sqeuclidean` 한 짝(타일 순서의 차례 합과 2⁻⁹⁶⁸ 아래의 조심 끝내기; `sd.spatial.cdist(metric='sqeuclidean')`와 같은 비트)으로 정한다. CL43의 2의 거듭제곱으로 옮긴 자료에서 잰다. 같은 거리의 중심은 시작이 만든 중심의 차례(맞춤 안의 번호)로 작은 쪽이고, `predict`도 맞춤의 격자와 그 차례를 써서 맞춘 점에 맞춤의 라벨을 준다. 결과의 `centers`는 처음 나오는 라벨의 차례(CL6)다 (추천안, 사용자 위임 2026-11-27) |
| CL60 k-평균의 길 | 경계(Hamerly 2010; CL42의 여유)는 k ≥ 2면 언제나, CL49의 거르기는 p ≥ 20이고 p·k ≥ 500이면 쓴다(모양만으로; §5의 측정에서 경계가 Lloyd보다 느린 곳이 없고, 거르기는 p·k가 클수록 빠르다). 거르기의 x·c는 Accelerate의 `dgemm`이다: §4.2의 한계는 내적을 어떤 차례로 더해도 성립해서, 거르기는 후보만 바꾸고 결과(차의 꼴로 다시 잰 최솟값)의 비트를 바꾸지 않는다. Elkan(2003)은 넣지 않는다(경계와 거르기로 §5의 목표에 이르렀고, 큰 k의 이득은 CL7 마일스톤이 잰다) (추천안, 사용자 위임 2026-11-27) |
| CL61 열쇠 순서의 사본 | CL48의 남은 물음: 커널이 점을 열쇠 순서로 모은 사본(C++이 모으면서 CL43의 척도도 곱한다)을 만들어 쓴다(색인으로 모으면 반복마다 흩어진 읽기다). 사본은 O(np)이고 10⁵ × 10에서 반복 한 번(5.5 ms)보다 작다. `lex_order`는 첫 좌표의 16비트 기수 정렬과 같은 첫 좌표 안의 비교 정렬로 바꿨다(같은 순서; 10⁵ × 10에서 6.4 → 1.2 ms; 계층 군집의 골든이 그대로다) (추천안, 사용자 위임 2026-11-27) |
| CL62 빈 군집 | sklearn의 옮기기(CL24)를 계산한 거리로 정한다: 무게가 0인 군집마다, 양의 무게 점 가운데 이번 배정의 중심까지 sqd가 가장 큰 점(동점은 열쇠)을 그 군집의 라벨로 옮기고 중심을 라벨의 무게 평균으로 다시 잰다. 옮긴 반복은 라벨이 바뀐 반복이다(참값 도구와 같다). sklearn은 평균을 뺀 자료의 반올림으로 다른 점을 고를 수 있다(규칙 자료 [0, 0.1, 10, 10.1]에서 정확한 거리로는 0.1이 더 멀어 sd는 [0, 1, 2, 2], sklearn은 [0, 0, 1, 2]) `[sklearn과 다름]` (추천안, 사용자 위임 2026-11-27) |
| CL63 난수의 원소 (CL41의 세부) | `kmeans` 시작 r: k-means++의 첫 중심 t = 0, 중심 c의 후보 q는 t = 1 + (c − 1)L + q(L = 2 + ⌊ln k⌋); `'random'`은 Efraimidis–Spirakis의 열쇠 log(uᵢ)/wᵢ(uᵢ는 원소 r·2⁴⁰ + 점의 열쇠)가 큰 k개(무게에 비례한 비복원 차례 뽑기와 같은 분포). `minibatch_kmeans`: 시작 s의 부분 표본 r = 2²² + s, 시작의 중심 r = s, 검증 표본 r = 2²³, 배치 r = 2²³ + 1(t = 단계·배치 크기 + 자리), 다시 놓기 r = 2²³ + 2. `bisecting_kmeans`: 나누기 s의 시작 i는 r = s·n_init + i. CLARA의 표본 s는 r = s. `fuzzy_cmeans`의 `'random'`은 r = 0의 균등 비복원. 무게에 비례한 복원 뽑기는 CL48 순서의 누적 합의 이진 찾기다 (추천안, 사용자 위임 2026-11-27) |
| CL64 미니배치·이분의 세부 | `minibatch_kmeans`는 sklearn `MiniBatchKMeans.fit`의 단계(검증 표본의 관성으로 시작 고르기, 누적 개수의 학습률, 10k점마다 또는 개수 0인 중심이 있을 때의 다시 놓기, 매끈한 배치 관성의 `max_no_improvement`)이고 `n_init='auto'`는 `'random'`에서 3이다(sklearn과 같다). `init_size` < k는 `ValueError`다(sklearn은 경고하고 3k로 바꾼다). `converged`는 단계 상한 전에 멈췄는지, `n_iter`는 ⌈단계·배치/n⌉이다. `bisecting_kmeans`의 나누기마다 최선 시작은 (관성, 번호)가 가장 작은 것(`kmeans`와 같다; sklearn은 상대 10⁻⁶ 넘게 나아질 때만 바꾼다 `[sklearn과 다름]`), 서로 다른 양의 무게 점이 둘보다 적은 잎은 나누지 않는다. `tree`는 잎이 군집(라벨)인 연결 행렬이다: 행마다 나누기 하나를 되돌린 병합, 높이는 나눈 군집의 관성을 트리 위로 누적 최댓값(그래서 단조), 넷째 열은 그 아래 잎의 수, 행은 (높이, 나중 나누기 먼저)의 차례다. `predict`는 sklearn처럼 트리를 내려간다(나누기마다 sqd가 작은 자식, 같으면 왼쪽). `best_init`은 나누기마다의 (k − 1,) 배열, `n_iter`는 반복의 합이다 (추천안, 사용자 위임 2026-11-27) |
| CL65 k-medoids의 세부 | `objective`는 SWAP 뒤의 평균 거리 하나이고(다른 결과의 `objective`처럼 수 하나; API 표의 "둘"을 바꾼다) BUILD 뒤의 값은 `build_objective`다(CLARA는 고른 표본의 BUILD medoid로 모든 점을 잰 값). SWAP은 계산한 변화가 음수인 가장 작은 (변화, medoid의 열쇠, 점의 열쇠)를 하고, 변화는 FastPAM1의 분해(공유 합 S_o와 medoid마다의 고침, 각각 열쇠 순서의 차례 합)로 정의한다. 기본 `max_iter` = 1000(교환 수의 상한; 측정한 교환 수는 n ≤ 2000, k ≤ 200에서 많아야 111). 관측값의 euclidean·cityblock은 필요할 때 재고(압축 꼴이 2²⁸ 바이트 이하면 한 번 재어 쥔다; 같은 비트), 그 밖의 거리는 `sd.spatial.pdist`다. CLARA는 R `clara`처럼 첫 표본 다음부터 지금까지의 최선 medoid를 표본에 넣고 나머지를 균등 비복원으로 채우며, 표본의 군집은 PAM 그대로다(R의 기본 `pamLike=FALSE`의 SWAP과 다르고 `pamLike=TRUE`와 같다 `[R과 다름]`); `sample_size = n`이고 표본 하나면 PAM과 같은 답이다. `predict`의 동점은 medoid의 열쇠다 (추천안, 사용자 위임 2026-11-27) |
| CL66 퍼지 c-평균의 세부 | 기본 `tol` = 10⁻⁵(소속도 변화의 최대 노름)을 확정한다: e1071 `cmeans`(문서의 기본 `reltol` = √ε의 목적 상대 변화)보다 늦게 멈추거나 같은 반복에서 멈춘다(시험 자료 28개 중 27개; 하나는 한 반복 먼저; §5). 소속도는 가장 가까운 중심과의 비 r_j = min D/D_j로 r_j^q / Σ_l r_l^q(q = 1/(m − 1))이고, 중심 위의 점은 그 중심들이 똑같이 나눈다. 시작은 `'random'`(점 k개) 또는 (k, p) 배열이다. `labels`는 소속도가 가장 큰 군집(같으면 맞춤 안의 작은 번호), `predict`는 결과의 라벨 차례에서 작은 쪽이다. 서로 다른 점이 k보다 적으면 `ValueError`(`kmeans`와 같다) (추천안, 사용자 위임 2026-11-27) |
| CL67 HDBSCAN의 트리 | 상호 도달 거리는 max(core_a, core_b, d/α)다(d/α는 한 번 반올림하고 α = 1이면 d 그대로; 핵심 거리는 α로 나누지 않은 거리의 `min_samples`번째, 자기 포함). sklearn의 KD 트리 길과 같은 뜻이고, sklearn의 미리 계산한 거리 길(`_hdbscan_brute`)은 거리 행렬을 먼저 α로 나눠 핵심 거리도 나뉜다 `[sklearn과 다름]`(α = 1이면 같다). 최소 신장 트리는 변의 차례 (w, 작은 열쇠, 큰 열쇠)(CL50의 키)로 하나로 정해지고, 단일 연결 트리(결과의 `single_linkage`)는 그 차례의 Kruskal이다(행 i가 변 i, 두 군집 번호는 (작은 것, 큰 것)). 트리는 관측값과 유클리드면 핵심 거리의 KD 트리 위의 Borůvka(mst.cpp를 넓혔다: 점의 모든 변은 core_a 이상, 마디의 점들은 그 마디의 가장 작은 핵심 거리 이상이고, d > T_d(r) = up(up(r)·α)이면 반올림한 d/α > r이라 셋 중 하나가 반경 r을 엄격히 넘는 마디를 자른다; §4.4), 그 밖은 Prim(행렬 없이, 또는 행렬 위)이다. 두 길은 같은 트리다(테스트) (추천안, 사용자 위임 2026-11-27) |
| CL68 HDBSCAN의 줄인 트리와 고르기 | 줄인 트리는 Campello 외(2013)의 정의대로 같은 높이의 병합을 한꺼번에 본다(같은 무게의 변을 함께 지운다): 뿌리부터 아래로, 높이 h의 병합에 선 군집은 그 아래 같은 높이의 병합을 모두 거둬 h 아래의 성분들(가장 작은 열쇠의 차례)로 나뉘고, `min_cluster_size` 이상인 성분이 둘 이상이면 각각 새 군집(n + 1부터 만든 차례), 하나면 그 성분이 군집을 잇고, 나머지 성분의 점들은 λ = 1/h(h = 0이면 inf)에 떨어진다(성분마다 너비 우선). 같은 높이가 없으면 sklearn 1.9.1의 `_condense_tree`와 같은 트리다. 같은 높이(상호 도달 거리에서는 한 점의 핵심 거리가 그 이웃 모두와의 값이라 흔하다)에서 sklearn은 Prim이 만난 차례로 이진 트리를 만들어 결과가 입력 순서에 달린다: 무작위 군집 자료 20벌 × (eom, leaf) × `min_cluster_size` 5, 15의 80가지에서 sklearn을 섞은 순서로 다시 돌리면 50가지의 라벨이 달랐다(평균 5.9점, 많게는 71점; `min_samples=1`이면 같은 높이가 없어 sd와 라벨·확률이 모두 같다) `[sklearn과 다름]`. 안정성(Σ(λ − λ_birth)·크기, 행의 차례), 고르기(`'eom'`의 동점은 부모; `'leaf'`; `cluster_selection_epsilon`의 찾기는 군집 번호의 오름차순, sklearn의 집합 반복 차례; `max_cluster_size`; `allow_single_cluster`), 라벨(가장 가까운 고른 조상), `probabilities`(min(λ, 군집의 가장 큰 λ)/그것)는 sklearn `_tree.pyx`를 그 트리에서 옮겼다. sklearn이 실패하는 두 곳: `allow_single_cluster`에서 하나 남은 군집이 뿌리가 아니면 뿌리 아래의 점에서 `KeyError`(sd는 잡음), scikit-learn 1.9.1과 numpy 2.5.4에서 `cluster_selection_epsilon` > 0의 찾기가 잎 하나라도 그 아래에서 태어나면 `TypeError`(1원소 배열을 정수로; sd는 의도대로) (추천안, 사용자 위임 2026-11-27) |
| CL69 GLOSH, 지속성, 중심 | `outlier_scores`는 GLOSH(Campello 외 2015)다: 점이 떠나는 군집 C의 부분 트리에서 가장 큰 λ_max로 1 − λ/λ_max(λ = inf면 0, λ_max = inf이고 λ가 유한하면 1(극한); 원본 `hdbscan`은 그때 NaN을 낼 수 있다). `persistence`는 원본 `hdbscan`의 `cluster_persistence_` 꼴: 고른 군집의 안정성/(크기 × 트리 전체의 가장 큰 λ), 그 λ가 0이나 inf면 1. `centroids`(확률을 무게로 한 평균; 합은 열쇠 차례)와 `medoids`(확률을 무게로 한 거리 합 Σ_j d(i, j)s_j가 가장 작은 점, 같으면 작은 열쇠)는 sklearn `store_centers`의 뜻이고 처음 읽을 때 계산한다(자료의 참조를 쥔다; 미리 계산한 거리면 None) (추천안, 사용자 위임 2026-11-27) |
| CL70 OPTICS의 차례 | sklearn `compute_optics_graph`의 차례: 단계마다 처리하지 않은 점 가운데 (도달 거리, 열쇠)가 가장 작은 점을 고르고(모두 inf면 가장 작은 열쇠: sklearn은 입력 색인이라 첫 점부터 다를 수 있다 `[sklearn과 다름]`), 그 점이 핵심(핵심 거리 ≤ `max_eps`)이면 `max_eps` 안(d ≤ max_eps)의 처리하지 않은 점의 도달 거리를 max(핵심 거리, d)가 엄격히 작을 때 낮추고 선행자를 적는다. 갱신과 다음 점 찾기를 처리하지 않은 점의 한 번 훑기로 한다(나란히; 도달 거리가 이미 핵심 거리 이하인 점은 거리를 재지 않는다). sklearn은 핵심 거리와 도달 거리를 소수 15자리로 반올림한다(`np.around`); sd는 계산한 거리 그대로다 `[sklearn과 다름]`(점을 열쇠 차례로 주면 차례·선행자가 sklearn과 같고 값은 6×10⁻¹⁶ 안이다; 테스트) (추천안, 사용자 위임 2026-11-27) |
| CL71 OPTICS의 추출 | ξ 추출(`_xi_cluster`, `_extract_xi_labels`)과 DBSCAN 추출(`cluster_optics_dbscan`)은 sklearn의 함수를 같은 부동소수 연산으로 C++에 옮겼다(선행자 보정의 "선행자가 [s, e)의 차례에 있는가"는 자리 배열로 O(1)). 그래서 sd의 도달 거리 배열에 sklearn의 함수를 돌린 것과 같은 군집이다(테스트). `OPTICSResult.extract_xi`와 `extract_dbscan`은 같은 차례의 새 `OPTICSResult`(`cluster_method`가 그 추출)를 돌려준다. `extract_dbscan(eps)`도 `optics(cluster_method='dbscan')`처럼 eps ≤ `max_eps`여야 한다(sklearn의 `cluster_optics_dbscan` 함수는 보지 않는다). `min_samples`, `min_cluster_size`는 sklearn처럼 2 이상의 int 또는 (0, 1]의 float(점 수의 비율, 적어도 2)이다. `cluster_hierarchy`는 ξ 추출의 (m, 2) int64(없으면 (0, 2)), DBSCAN 추출이면 None (추천안, 사용자 위임 2026-11-27) |
| CL72 밀도 방법의 길 | 관측값의 `'euclidean'`, `'cityblock'`, `'chebyshev'`는 행렬 없이 spatial의 짝 커널로 재고(같은 비트), n ≥ 64이고 DBSCAN의 반경 찾기는 p ≤ 12, 핵심 거리(k 최근접)와 HDBSCAN의 Borůvka는 p ≤ 16이면 KD 트리, 그 밖은 모든 짝의 훑기다(모양만으로; §5의 측정에서 DBSCAN은 p = 16부터, HDBSCAN은 p = 24부터 훑기가 빨랐다). 그 밖의 거리는 열쇠 차례의 `sd.spatial.pdist`(압축 꼴, 물리 메모리보다 크면 `MemoryError`)이고 미리 계산한 거리는 그 행렬이다. 길을 강제하는 `_core.cluster`의 `path`(1 훑기, 2 트리)는 테스트의 두 길 비교용이다 (추천안, 사용자 위임 2026-11-27) |
| CL73 DBSCAN의 세부 | 무게가 있으면 eps 안의 무게를 열쇠 차례로 더해 `min_samples`와 견준다(KD 트리와 훑기가 같은 비트; 무게가 없으면 정수 셈이고 KD 트리는 먼 모서리까지 eps 안이 증명된 마디를 통째로 센다). 핵심 점의 연결은 graph에서 꺼낸 `graph/unionfind.hpp`의 병렬 합집합–찾기(graph의 약한 연결 요소와 같은 함수; graph의 테스트·골든이 그대로다), 군집 번호는 가장 작은 열쇠의 차례에서 CL6의 처음 나오는 차례로 바뀐다. 미리 계산한 거리와 무게의 NaN은 `ValueError`(`nan_policy`는 관측값의 행에만) (추천안, 사용자 위임 2026-11-27) |
| CL74 밀도 결과의 필드 | `DBSCANResult`: `core_mask`, `eps`, `min_samples`, `metric`. `HDBSCANResult`: `probabilities`, `outlier_scores`, `single_linkage`, `condensed_tree`(namedtuple `CondensedTree(parent, child, lambda_val, child_size)`; 공개 이름이 아니다), `persistence`, `centroids`, `medoids`, `min_cluster_size`, `min_samples`, `cluster_selection_method`, `metric`, 메서드 `dbscan_clustering(cut_distance, /, *, min_cluster_size=5)`(sklearn `labelling_at_cut`: 높이 < cut의 병합을 잇고 작은 성분은 잡음). `OPTICSResult`: `ordering`, `reachability`, `core_distances`, `predecessor`, `cluster_hierarchy`, `min_samples`(바꾼 int), `max_eps`, `cluster_method`, `eps`, `xi`, `metric`. `nan_policy='omit'`이 뺀 행은 점마다의 실수 필드가 NaN, 선행자가 −1, `core_mask`가 False이고, `ordering`·`predecessor`는 입력 행 번호, `single_linkage`·`condensed_tree`는 남은 행의 번호다(`HierarchyResult`와 같다) (추천안, 사용자 위임 2026-11-27) |
| CL75 EM의 정의 | sklearn `GaussianMixture`의 단계를 이렇게 계산한다. E 단계: 성분 j의 로그 밀도는 차의 꼴 z = x − μⱼ, y = Uⱼᵀz(y_e = Σ_{d≤e} U_de z_d, d의 차례로 fma), q = Σ_e y_e²(e의 차례)로 ℓⱼ = −½(p log 2π + q) + Σ_d log U_dd(sklearn의 나머지 꼴)이고, 가중 로그 밀도 wⱼ = ℓⱼ + log πⱼ, m = maxⱼ wⱼ, eⱼ = exp(wⱼ − m)(배열의 NEON exp), s = Σⱼ eⱼ, log p(x) = m + log s, 책임도 eⱼ/s. U는 sklearn의 `precisions_cholesky_`(UUᵀ = Σ⁻¹인 위 삼각)이고 Σ의 Cholesky L(고정 순서, 아래 삼각을 읽는다)의 역의 전치다(diag·spherical은 1/√σ²). M 단계: nkⱼ = Σᵢ rᵢⱼ + 10ε(sklearn의 지킴: 빈 성분의 평균이 정해진다), 평균 Σᵢ rᵢⱼxᵢ/nkⱼ, 공분산은 그 평균을 뺀 두 번째 훑기 Σᵢ rᵢⱼzzᵀ/nkⱼ + reg_covar·I(tied: 성분을 모두 더해 Σⱼnkⱼ로 나눈다, diag: 대각, spherical: reg를 더한 대각의 평균), 무게 nkⱼ/Σnk(시작의 책임도에서는 sklearn `_initialize`처럼 nkⱼ/n). 점에 걸친 합은 모두 CL48의 덩이 합이다(덩이와 성분으로 나눈 작업이 같은 나무로 만나 스레드와 무관한 비트). 반복은 sklearn `fit`과 같다: 시작마다 \|하한 − 앞 하한\| < `tol`이면 멈추고(하한은 E 단계의 log p(x) 평균, sklearn의 `lower_bound_`), 가장 좋은 시작은 하한이 가장 큰 것(같으면 앞), `labels`는 그 모수에서 가중 로그 밀도가 가장 큰 성분(같으면 맞춤 안의 앞 번호; `predict`와 같아 맞춘 점에 맞춤의 라벨), `loglik`는 그 모수에서 log p(x)의 덩이 합이다. `max_iter=0`은 시작 그대로(`n_iter` 0, `converged` False, `lower_bound` −inf; sklearn과 같다; 시작이 여럿이면 sd는 첫 시작, sklearn은 마지막 `[sklearn과 다름]`). sklearn과 다른 계산: full·tied의 이차형식(sklearn은 XU − μU의 곱), diag·spherical의 이차형식(sklearn은 Σμ²λ − 2xμλ + x²λ로 펼친다; 10⁸ 평행 이동에서 q가 10⁻³ 넘게 틀린다), tied·diag·spherical의 2차 모멘트(sklearn은 E[xxᵀ] − μμᵀ의 한 번 훑기; 주어진 책임도의 M 단계가 sd 계약의 9.3배까지 틀린다) `[sklearn과 다름]` (추천안, 사용자 위임 2026-11-27) |
| CL76 혼합 모형의 시작과 난수 | 시작 s(0 … `n_init` − 1; CL41)의 책임도: `'kmeans'`는 부모 흐름의 원소 s·2⁴⁰과 s·2⁴⁰ + 1의 첫 단어를 두 단어로 삼은 `kmeans(X, k, init='k-means++', n_init=1, max_iter=300, tol=1e-4)`(sklearn `KMeans(n_clusters=k, n_init=1)`의 기본값)의 라벨(맞춤 안의 번호)의 one-hot, `'k-means++'`는 `kmeans`의 k-means++(CL63의 원소, 시작 r = s)가 고른 k점의 one-hot(sklearn `kmeans_plusplus`의 색인과 같은 뜻), `'random'`은 원소 s·2⁴⁰ + i·k + j의 (0, 1) 균등 수를 행마다 그 합으로 나눈 것, `'random_from_data'`는 원소 s·2⁴⁰ + i의 균등 수가 큰 k점(j번째로 큰 점이 성분 j; 비복원 균등 뽑기)이다. i는 열쇠의 차례(CL7)라 입력 순서와 무관하다. `weights_init`·`means_init`·`precisions_init`는 그 시작의 값을 바꾸고, 셋 다 있으면 책임도를 만들지 않는다(sklearn과 같다; `rng`는 그래도 두 단어 움직인다). `'kmeans'`는 서로 다른 점이 k개 이상이어야 한다(`kmeans`와 같은 `ValueError`; sklearn은 경고와 적은 군집) (추천안, 사용자 위임 2026-11-27) |
| CL77 혼합 모형의 척도 | CL43의 격자 e를 max(max\|x\|, max\|means_init\|, √reg_covar)로 정한다: reg_covar가 자료의 척도보다 크면 공분산을 reg가 지배하므로 옮기지 않고(옮기면 reg·2²ᵉ가 넘친다), reg_covar = 0의 아주 작거나 큰 자료는 [1, 2)로 옮겨 2차 모멘트의 아래넘침·넘침을 피한다. 격자에서는 reg_covar·2²ᵉ, precisions_init·2⁻²ᵉ를 쓰고, 결과는 자료의 단위로 되돌린다(평균 2⁻ᵉ, 공분산 2⁻²ᵉ, 인자 2ᵉ; 로그 밀도·`loglik`·`lower_bound`에 p·e·log 2를 더한다). `predict`·`predict_proba`·`score_samples`는 맞춤의 격자에서 재고, 격자에서 좌표가 넘치는 새 점은 `ValueError`다(맞춘 자료에서 2¹⁰²⁴⁻ᵉ배 먼 점) (추천안, 사용자 위임 2026-11-27) |
| CL78 혼합 모형의 입력 | n ≥ 2(sklearn `ensure_min_samples=2`), 1 ≤ k ≤ n, `tol`·`reg_covar`는 0 이상 유한, `max_iter` ≥ 0(0 허용: sklearn과 같다), `n_init` ≥ 1. `weights_init`은 (k,)이고 [0, 1] 안, 합이 1에서 10⁻⁸ 안(sklearn `_check_weights`), `means_init`은 (k, p)(p = 1이면 (k,)도), `precisions_init`은 sklearn의 모양이고 full·tied는 numpy `isclose`(10⁻⁸ + 10⁻⁵\|Pᵀ\|)의 대칭(sklearn과 같다; 위 삼각을 읽는다)과 양정치(UUᵀ = P의 인자를 맨 아래 행부터 만드는 피벗이 양수; 실패는 `ValueError`: `'precisions_init' must be positive definite (component j is not)`), diag·spherical은 양수다. reg_covar를 더한 공분산의 인자가 실패하면 `LinAlgError`(`component j has an ill-defined covariance; increase 'reg_covar' or check for duplicate points`, j는 맞춤 안의 번호; tied는 `the tied covariance is ill-defined`) (추천안, 사용자 위임 2026-11-27) |
| CL79 거의 특이한 공분산 | §4.5의 마지막 물음(smooth SM58처럼 막을지)의 답: 막지 않는다(sklearn과 같다; `reg_covar`가 그 장치다). 공분산에서 인자로 가는 계약이 κ₂(Σ)에 비례한다(§4.5의 계약 8): full·tied는 max\|Û − U*\| ≤ C_PREC·p·γ_{p+1}·κ₂(Σ)·max\|U*\|(1차, C_PREC = 2), diag·spherical은 γ₂\|U*\|(증명). 측정에서 κ = 10⁴(full/k2p5)도 계약의 10⁻³배 안이다 (추천안, 사용자 위임 2026-11-27) |
| CL80 혼합 모형의 표본 | `sample(shape=None, /, *, rng)`는 호출마다 `rng`의 두 단어를 받고(CL41) 표본 t(C 차례)는 원소 t다: 첫 단어로 라벨 차례의 누적 무게에서 성분을(무게에 비례), 이어지는 단어로 sd.stats의 ziggurat 정규 수 p개 z를(`stats/sample_dist.hpp`; `sd.stats.Normal`의 `sample`과 같은 꼴), 점은 μ + Lz, L = (U⁻¹)ᵀ(맞춤의 격자에서, 그다음 2⁻ᵉ)다. 표본마다 성분이 독립이다(sklearn은 다항 분포로 개수를 뽑아 성분 차례로 묶는다 `[sklearn과 다름]`). 결과는 namedtuple `MixtureSample(X, labels)`(공개 이름이 아니다; CL74의 `CondensedTree`와 같다), `shape=None`이면 (p,)와 int다 (추천안, 사용자 위임 2026-11-27) |
| CL81 혼합 결과의 필드 | `GaussianMixtureResult`: `weights`, `means`, `covariances`, `precisions_cholesky`(sklearn의 모양), `loglik`(= `objective`), `lower_bound`, `bic` = −2·loglik + n_parameters·log n, `aic` = −2·loglik + 2·n_parameters(sklearn의 `score(X)·n`과 같은 값), `n_parameters`(sklearn `_n_parameters`), `n_iter`, `converged`(가장 좋은 시작의 것), `covariance_type`, `n_init`, `best_init`. 성분은 라벨의 처음 나오는 차례(CL6; 가장 큰 점이 없는 성분은 뒤에 맞춤의 차례로)이고 tied의 `covariances`·`precisions_cholesky`는 하나다. sklearn의 `precisions_`는 두지 않는다(`precisions_cholesky`에서 UUᵀ). `summary()`의 표는 크기와 `Weight`, 오른쪽은 공분산 꼴, 시작 수, 고른 시작, 하한, BIC, AIC다 (추천안, 사용자 위임 2026-11-27) |
| CL82 지표의 라벨 | 라벨은 1차원의 정수, bool, 문자열(목록, numpy, `Series`)과 값이 정수인 실수다. 군집은 서로 다른 값이고 그 정렬 순서(numpy `np.unique`; 문자열은 Python의 정렬)가 분할표의 행·열이다. 소수가 있는 실수 라벨은 `ValueError`(sklearn은 `UserWarning`과 받기 `[sklearn과 다름]`), 라벨이 없으면 `ValueError`(sklearn은 많은 지표에서 1.0 `[sklearn과 다름]`), 점은 n(n − 1) < 2⁶³까지(짝 혼동 행렬의 int64; §3.8의 "C(n, 2) < 2⁶³, n < 2³²"를 이 한계로 바꾼다) (추천안, 사용자 위임 2026-11-27) |
| CL83 정확한 외부 지표 | 분할표(C++: 칸 수가 max(4n, 2²⁰) 이하이면 밀집 셈, 넘으면 (행·열수 + 열) 열쇠의 정렬)의 정수 셈 Σ C(nᵢⱼ, 2), Σ C(aᵢ, 2), Σ C(bⱼ, 2)를 Python 정수로 받아, ARI·RI는 정확한 분자·분모의 Python 정수 나눗셈(CPython의 정수 나눗셈은 올바르게 반올림한다), FM = TP/√((TP + FP)(TP + FN))은 정수 제곱근과 끈적한 비트로 올바르게 반올림한 √(TP²/((TP + FP)(TP + FN)))다(§4.6의 "1 ULP 꼴"을 0.5 ULP로 좁힌다). 짝 혼동 행렬은 sklearn의 순서 있는 짝(2배)이다 (추천안, 사용자 위임 2026-11-27) |
| CL84 정보 지표 | 로그는 L(x, y) = log(x/y)를 정확한 정수 x, y에서 x/y가 [1/2, 3/2] 안이면 log1p((x − y)/y)로 잰다(어디서나 상대오차 9u 안). MI = Σ(nᵢⱼ/n)L(n·nᵢⱼ, aᵢbⱼ)와 엔트로피는 칸 차례의 짝 합이고, MI는 0 아래를 0으로 자른다(sklearn처럼; sklearn은 ε 아래의 항을 0으로 지우기도 한다: sd는 지우지 않는다 `[sklearn과 다름]`). AMI의 기대 MI는 (행의 크기 α, 열의 크기 β)마다 초기하 무게를 비의 점화식 w(c + 1)/w(c) = (α − c)(β − c)/((c + 1)(n − α − β + c + 1))로 최빈값에서 밖으로 만들고(꼬리가 합의 2⁻⁸⁰ 아래이면 멈춘다; log-오목이라 남은 꼬리의 한계가 증명된다) 그 합으로 나눈다: sklearn의 lgamma 차는 n = 10⁶에서 10⁻⁸까지 상쇄한다. 같은 크기의 짝은 한 번만 재어 개수를 곱한다(그래서 n = 10⁶에서 10 ms; sklearn 397 ms). 특수 경우: 양쪽이 한 군집이거나 양쪽이 모두 홑원소면 AMI = 1, 한쪽만 한 군집이면 0(sklearn과 같다), 한쪽만 모두 홑원소면 모든 재배치의 MI가 다른 쪽 엔트로피와 같아(MI = E[MI]) 0이고 `'min'`은 1이다(그 정규화 값도 그 엔트로피다; 반올림한 sklearn도 1.0과 0 근처의 잡음을 낸다). 그 밖은 sklearn의 꼴(분모를 ±2⁻⁵² 밖으로)이다. 정확한 값이 [0, 1]인 NMI·동질성·완전성·V와 AMI는 1을 넘지 않게 자른다(sklearn의 동질성은 1.0000000000000002가 된다). V는 (1 + β)MI/(H_true + βH_pred)(h·c의 꼴보다 반올림이 적다; β = 0이면 h)라 β = 1이면 NMI(arithmetic)와 같은 비트다 (추천안, 사용자 위임 2026-11-27) |
| CL85 실루엣의 길 | 관측값의 euclidean·sqeuclidean·cityblock·chebyshev는 spatial의 타일 커널(4점 × 8행의 판; `cdist`와 같은 비트)로 n × n 행렬 없이 재고, 그 밖의 거리는 열쇠 순서의 `sd.spatial.pdist`, 미리 계산한 거리는 그 행렬이다(CL43의 2의 거듭제곱으로 옮긴 자료: 비만 쓰여 값이 같다). 점마다 군집마다의 합은 군집 안 열쇠 순서에서 군집 안 자리의 4 나머지로 나눈 네 사슬을 (s₀ + s₁) + (s₂ + s₃)로 합친다(그래서 입력 순서와도, 군집의 라벨 값과도 무관한 비트다). a = b = 0인 점의 값은 0(sklearn의 `nan_to_num`과 같다), 평균은 CL48의 덩이 합이다 (추천안, 사용자 위임 2026-11-27) |
| CL86 Calinski–Harabasz와 Davies–Bouldin | 두 지표가 평행 이동과 척도에 불변이라, 첫 열쇠의 점을 뺀 자료(그 점의 두 배 안의 좌표는 정확하다)에 CL43의 척도를 곱해 잰다. 군집 평균은 군집의 첫 점을 뺀 덩이 합(같은 점들의 군집이면 정확하다), W = 0이면 CH = 1(sklearn과 같다). DB는 모든 군집의 산포가 0이거나 모든 중심 사이 거리가 정확히 0일 때만 0이고(sklearn의 `np.allclose(…, 0)`은 10⁻⁸ 아래를 0으로 보아 척도에 따라 답이 다르다 `[sklearn과 다름]`), 중심이 같은 두 군집은 서로의 최댓값에서 뺀다(sklearn의 inf 거리와 같다). shifted/k3p2(10⁸ 평행 이동)에서 평균을 원점에서 잰 첫 판은 상대 2×10⁻⁹ 틀렸고 이 꼴은 4×10⁻¹⁵ 안이다 (추천안, 사용자 위임 2026-11-27) |
| CL87 gap의 이름과 입력 | `reference='scaledpca'\|'original'`, `select='firstsemax'\|'tibs2001semax'\|'globalsemax'\|'firstmax'\|'globalmax'`: R 이름의 소문자 꼴이고, 다른 대소문자는 optimize의 방법 이름(O15)처럼 소문자 꼴을 짚는 `ValueError`다(`'select' must be lowercase: 'firstsemax', not 'firstSEmax'`). `k_max`는 2 이상(R과 같다)이고 서로 다른 점의 수 이하, `n_reference`는 2 이상(R의 B = 1은 SE가 NA라 `maxSE`가 실패한다 `[R과 다름]`), `d_power`는 0보다 큰 유한한 수, `se_factor`는 0 이상이다. 결과 `GapResult`는 분할이 아니라 `labels`·`n_clusters`가 없고 `log_w_reference`((B, k_max))와 `k_max`를 더한다 (추천안, 사용자 위임 2026-11-27) |
| CL88 gap의 계산 | R `clusGap`의 단계를 이렇게 계산한다. 자료는 열쇠 순서, 가운데는 첫 점을 뺀 덩이 합의 평균, `'scaledpca'`의 축은 가운데로 옮긴 자료의 XᵀX(덩이 합)를 순환 Jacobi로 푼 고유벡터(고유값이 큰 차례, 같으면 번호; 열마다 절댓값이 가장 큰 성분을 양수로; R은 SVD의 V: 같은 축이고 부호는 상자의 분포를 바꾸지 않는다)이고 상자는 그 축의 좌표 범위다. 참조 자료 b(0 … B − 1)의 i번째 점의 d좌표는 원소 b·2⁴⁰ + i·p + d의 균등 수 u로 lo + (hi − lo)u(R의 `runif`), 그다음 축을 되돌려 가운데를 더한다. 벌마다 점을 열쇠 순서로 놓고, 군집은 k = 1이면 하나, 그 밖은 `'kmeans'`면 sd `kmeans`의 기본값(k-means++, 시작 하나; 집합 s(0이 자료, b + 1이 참조 b)의 k는 부모 흐름의 시작 2²³ + s·k_max + k의 두 단어; R은 Hartigan–Wong의 무작위 시작 하나 `[R과 다름]`), `'pam'`이면 sd `kmedoids`(PAM; R `pam`과 같은 medoid라 R과 같은 log W)다. W = ½Σ_r D_r/n_r, D_r은 군집 안 짝의 dᵖᵒʷᵉʳ(spatial의 유클리드; 행마다 네 사슬, 행들의 짝 합)이고 CL43의 격자에서 재어 log W에서 e·power·log 2를 뺀다(비정상 수 자료의 d²가 아래넘쳐 log 0이 되던 것을 막는다). E[log W]는 벌 차례의 평균, SE = √((1 + 1/B)·var)(평균을 뺀 두 번째 훑기, B − 1로 나눔). 벌(자료 포함 B + 1개)은 스레드마다 하나씩 나란히 돌고 벌 안의 군집은 차례다(풀 안에서는 중첩된 나란히가 차례로 돈다: 같은 비트). 하위 호출의 두 단어는 `_core.cluster.subkey`다(CL90) (추천안, 사용자 위임 2026-11-27) |
| CL89 Gower의 세부 | CL38에 더한다: `data`는 `DataFrame` 또는 수의 2차원 배열(모든 열이 수)이고 `types`의 열쇠는 열 이름(배열이면 위치)이다. 문자열 열은 명목, bool 열은 대칭 이진, `'symm'`·`'asymm'`은 0·1(또는 bool)과 결측만, `'ordratio'`는 값의 조밀한 순위(문자열은 정렬 순서; R의 `as.ordered`)를 수로, `'logratio'`는 양수의 log10을 수로 읽는다(0 이하는 `ValueError`; R은 NaN·−Inf). R의 순서형 factor는 그 정수 코드를 수 열로 주면 같다(R도 코드를 범위로 나눈다). 수의 δ는 \|x − y\|/범위를 곧바로 잰다(R은 열을 (x − min)/범위로 먼저 옮긴다: 끝자리만 다르다), 상수 열의 범위는 1이다(R과 같다). 같이 가진 변수가 없는 짝은 NaN이다(R은 남은 무게의 합이 0.5 이하이면 NA라 무게가 작으면 값이 있는 짝도 NA가 된다 `[R과 다름]`). 차례는 변수 차례의 합 Σwδ/Σw이고 짝마다 나란하다 (추천안, 사용자 위임 2026-11-27) |
| CL90 하위 호출의 두 단어 | CL4 마일스톤의 `_core.cluster.gmm_subkey`를 공용 이름 `_core.cluster.subkey(key0, key1, r)`(원소 r·2⁴⁰과 r·2⁴⁰ + 1의 첫 단어)로 옮겨 혼합 모형의 시작과 gap의 군집이 함께 쓴다(C++의 `metrics.cpp`; 결과의 비트는 그대로다) (추천안, 사용자 위임 2026-11-27) |
| CL91 Affinity Propagation의 유사도 입력 | 임시로 둔 `metric='precomputed'`(유사도)를 sklearn의 `affinity='euclidean'\|'precomputed'`로 바꾼다: 이 영역의 `metric='precomputed'`는 언제나 거리(CL12)라 한 이름에 두 뜻을 두지 않는다. `'precomputed'`는 (n, n) 유사도 행렬(클수록 비슷하고 대칭이 아니어도 된다)이고, `'euclidean'`의 유사도는 음의 제곱 유클리드 거리(spatial의 차의 꼴)다. CL12의 "거리로 정의되는 방법" 목록에서 `affinity_propagation`을 뺀다 (추천안, 사용자 위임 2026-11-27) |
| CL92 Affinity Propagation의 계산 | sklearn 1.9.1 `_affinity_propagation`의 연산을 하나씩 같은 차례로 하되 잡음을 더하지 않고 점을 열쇠 순서로 놓는다(미리 계산한 유사도는 입력 색인): 행마다 Y = max(A + S), 처음 나온 자리 I, 나머지의 최댓값 Y2; R = fl(fl(R·d) + fl(R_new·(1 − d))); 열의 합은 행 차례의 순차 합(numpy의 axis=0 줄이기와 같다); A = fl(fl(A·d) − fl(T·(1 − d))); E = A_kk + R_kk > 0. −0.0은 +0.0으로 바꾼다(잡음이 0인 sklearn의 S + 0이 그렇다). NEON의 max는 numpy의 `maximum`과 0의 부호만 다를 수 있고, 0의 부호는 비교와 0 아닌 값을 바꾸지 않는다. 그래서 열쇠 순서의 유사도에서 잡음이 0인 sklearn과 exemplar·라벨·반복 수가 같다(시험 자료 35개 × 세 설정의 105가지와 주어진 선호도). sklearn의 잡음은 퇴화(같은 유사도)를 깨려는 것이고 열쇠의 동점 규칙이 같은 일을 한다(CL40) (추천안, 사용자 위임 2026-11-27) |
| CL93 Affinity Propagation의 결과와 멈춤 | `preference=None`은 유사도 n²개의 중앙값(numpy `median`: 짝수 개면 가운데 둘의 평균)이고 결과의 `preference`가 쓴 값이다(배열이면 입력 행마다; `'omit'`은 뺀 행의 값도 뺀다). 모든 비대각 유사도와 선호도가 같거나 n = 1이면 sklearn처럼 반복하지 않는다(선호도가 유사도보다 크면 모든 점이 exemplar, 아니면 첫 열쇠의 점 하나; `n_iter` 0, `converged` True). `convergence_iter`번 동안 모든 점의 exemplar 여부가 그대로이고 exemplar가 하나 이상이면 멈추며(`n_iter` = it + 1), 아니면 `max_iter`에서 `converged=False`다. 그때 exemplar가 없으면 모든 라벨이 −1, `n_clusters` 0이다(sklearn과 같다; 경고 없음). 라벨은 CL6의 처음 나오는 차례, `exemplars[j]`가 라벨 j의 exemplar다. `predict`는 제곱 거리가 가장 작은 exemplar(같으면 맞춤의 I 차례: 맞춘 점은 맞춤의 라벨을 받는다; exemplar가 둘 다 같은 점일 때만 예외), `'precomputed'`의 맞춤은 `ValueError`(sklearn과 같다). `damping`은 [0.5, 1), `max_iter`·`convergence_iter`는 1 이상(sklearn과 같다) (추천안, 사용자 위임 2026-11-27) |
| CL94 Affinity Propagation의 메모리와 나란히 | S, R, A의 n × n 세 행렬(sklearn은 tmp까지 넷)과 중앙값을 고르는 사본 하나: 할당 전에 4·8n² 바이트가 물리 메모리보다 크면 `MemoryError`다. 반복마다 행의 훑기 한 번(앞 반복의 A 갱신과 이번 R 갱신을 한 행 안에서, 행 버퍼의 NEON)과 열 합의 훑기 한 번(256열 덩이마다 행 차례)이고, exemplar 판정은 열 합에서 대각만 O(n)으로 먼저 잰다. 행과 열 덩이가 나란하고 값마다 한 점 또는 한 열의 고정된 연산 차례라 스레드와 무관한 비트다 (추천안, 사용자 위임 2026-11-27) |

### 1.2 범위
마일스톤 열은 그 이름을 만드는 §8의 마일스톤이고 "나중"은 첫 판에 넣지 않는 후보, "밖"은 이 영역에 두지 않는 것이다. 이름은 §9의 1·2·11번의 답(a)을 따른다.

**이름공간**: 모든 공개 이름이 `sd.cluster`에 바로 있다(평평하다). sklearn이 `cluster`·`mixture`·`metrics`에, scipy가 `cluster.hierarchy`·`cluster.vq`에 나눠 둔 것이 한곳에 모인다.

**분할 → sd.cluster**
| 다른 라이브러리 | sd.cluster | 마일스톤 |
|---|---|---|
| sklearn `KMeans(n_clusters=8, init='k-means++', n_init='auto', max_iter=300, tol=1e-4, algorithm='lloyd')`, `k_means`, `kmeans_plusplus` | `kmeans(X, k, *, rng, …)` → `KMeansResult`. `algorithm`이 없다(§9의 20번) | CL2 |
| R `kmeans(x, centers, iter.max=10, nstart=1, algorithm='Hartigan-Wong')` | 같은 함수. 기본 알고리즘·반복 상한이 다르다(§2.3) | CL2 |
| scipy `vq.kmeans`, `vq.kmeans2`, `vq.vq`, `vq.whiten` | `kmeans`와 `KMeansResult.predict`. `whiten`은 없다(배열의 `std`로 나눈다) | CL2 |
| sklearn `MiniBatchKMeans`, `BisectingKMeans` | `minibatch_kmeans`, `bisecting_kmeans`(§9의 22번) | CL2 |
| R `cluster::pam`, `cluster::clara` (sklearn에는 없다) | `kmedoids(X, k, *, method='pam')`, `method='clara'` → `KMedoidsResult`(§9의 23번) | CL2 |
| R `cluster::fanny`, `e1071::cmeans`(설치 안 됨) | `fuzzy_cmeans`(§9의 24번) | CL2 |

**계층 → sd.cluster**
| 다른 라이브러리 | sd.cluster | 마일스톤 |
|---|---|---|
| scipy `linkage(y, method='single', metric='euclidean', optimal_ordering=False)` | 같은 이름, 배열 Z를 돌려준다. 그리고 결과 객체를 주는 `agglomerative(X, *, method, …)` → `HierarchyResult`(§9의 1번) | CL1 |
| R `hclust(d, method='complete')`, `cluster::agnes` | `agglomerative`. R의 이름 대응(`ward.D2` = `ward`, `mcquitty` = `weighted`, `centroid`·`median`은 제곱 거리의 규칙; §2.2) | CL1 |
| sklearn `AgglomerativeClustering(n_clusters=2, linkage='ward', connectivity=None)`, `ward_tree`, `linkage_tree` | `agglomerative`와 `HierarchyResult.cut(k=)`. 연결 제약(`connectivity`)은 나중(§9의 19번) | CL1, 나중 |
| scipy `fcluster`, `cut_tree`, `cophenet`, `leaves_list`, `optimal_leaf_ordering`, `inconsistent`, `maxdists`, `maxinconsts`, `maxRstat`, `leaders`, `is_monotonic`, `is_valid_linkage`, `num_obs_linkage`, `is_isomorphic` | 같은 이름(§9의 17번) | CL1 |
| scipy `dendrogram(Z, no_plot=True)` | `dendrogram`: 그림 없이 자리 자료만(§9의 18번) | CL1 |
| R `cluster::diana` | `divisive(X, *, …)` → `HierarchyResult`(§9의 19번) | CL1 |
| R `cutree`, `cophenetic`, `order` | `HierarchyResult.cut`, `cophenet`, `leaves_list` | CL1 |
| scipy `to_tree`·`ClusterNode`, `from_mlab_linkage`·`to_mlab_linkage`, `correspond`, `set_link_color_palette`, `DisjointSet`, `single`·`complete`·… 줄임 함수, `fclusterdata` | 없음(§9의 17번) | 밖 |

**밀도 → sd.cluster**
| 다른 라이브러리 | sd.cluster | 마일스톤 |
|---|---|---|
| sklearn `DBSCAN(eps=0.5, min_samples=5, metric='euclidean')`, `dbscan` | `dbscan(X, *, eps=0.5, min_samples=5, …)` → `DBSCANResult`. 경계 점의 규칙이 다르다(§9의 25번) | CL3 |
| sklearn `HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_method='eom', …)` | `hdbscan(X, *, …)` → `HDBSCANResult`(§9의 26번) | CL3 |
| sklearn `OPTICS(min_samples=5, max_eps=inf, cluster_method='xi', xi=0.05)`, `cluster_optics_xi`, `cluster_optics_dbscan`, `compute_optics_graph` | `optics(X, *, …)` → `OPTICSResult`와 그 메서드 `extract_xi`, `extract_dbscan`(§9의 27번) | CL3 |
| sklearn `MeanShift`, `estimate_bandwidth` | 나중(§9의 28번) | 나중 |

**혼합 모형 → sd.cluster**
| 다른 라이브러리 | sd.cluster | 마일스톤 |
|---|---|---|
| sklearn `GaussianMixture(n_components=1, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans')` | `gaussian_mixture(X, k, *, rng, covariance_type='full', …)` → `GaussianMixtureResult`(§9의 29–31번) | CL4 |
| mclust `Mclust`의 14 공분산 꼴(설치 안 됨) | 나중(§9의 29번) | 나중 |
| sklearn `BayesianGaussianMixture` | 나중(§9의 32번) | 나중 |

**지표와 k 고르기 → sd.cluster** (§9의 10·11번)
| 다른 라이브러리 | sd.cluster | 마일스톤 |
|---|---|---|
| sklearn `silhouette_samples`, `silhouette_score`, R `cluster::silhouette` | 같은 이름. `silhouette_score`의 `sample_size`·`random_state`는 없다(언제나 전체) | CL5 — **마침** |
| sklearn `calinski_harabasz_score`, `davies_bouldin_score` | 같은 이름 | CL5 — **마침** |
| sklearn `adjusted_rand_score`, `rand_score`, `adjusted_mutual_info_score`, `normalized_mutual_info_score`, `mutual_info_score`, `homogeneity_score`, `completeness_score`, `v_measure_score`, `homogeneity_completeness_v_measure`, `fowlkes_mallows_score`, `pair_confusion_matrix`, `contingency_matrix`(`sklearn.metrics.cluster`) | 같은 이름 | CL5 — **마침** |
| R `cluster::clusGap(x, FUNcluster, K.max, B=100, d.power=1, spaceH0='scaledPCA')`, `maxSE` | `gap_statistic` → `GapResult`(§9의 34번; 이름은 소문자, CL87) | CL5 — **마침** |
| R `cluster::daisy(x, metric='gower')` | `gower_distance`(§9의 35번; CL89) | CL5 — **마침** |

**그 밖**
| 다른 라이브러리 | sd.cluster | 마일스톤 |
|---|---|---|
| sklearn `AffinityPropagation`, `affinity_propagation` | `affinity_propagation`(§9의 37번; 유사도 입력은 `affinity`, CL91) | CL6 — **마침** |
| sklearn `SpectralClustering`, `spectral_clustering` | 나중, `sd.embed`의 스펙트럼 임베딩 뒤(§9의 36번) | 나중 |
| sklearn `Birch`, `FeatureAgglomeration` | 나중(§9의 19·37번) | 나중 |
| sklearn `SpectralBiclustering`, `SpectralCoclustering` | 밖(§9의 37번) | 밖 |
| 차원 축소(PCA, t-SNE, UMAP), 분류기와 그 회귀판 | 밖: 나중의 `sd.embed`, `sd.classify`(CL2) | 밖 |

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다. 별칭이 없다(`method`, `metric`, `init`, `covariance_type`의 문자열은 소문자 정식 이름만).
- **같은 기계, 같은 스레드 수, 같은 입력이면 실행마다 같은 비트다**(API.md §3). 새 영역의 관례대로 이 영역도 스레드 수와 무관하게 하고(§9의 6번; 시제품에서 Accelerate 꼴보다 1.5배 넘게 느린 경로가 나오면 그 경로만 다시 묻는다), 테스트와 골든은 1·3·10스레드에서 같은 바이트를 확인한다. 나누기는 크기로만 정한다(§3.10). 빠른 길과 느린 길이 있는 곳(k-평균의 Lloyd와 경계 가속, 계층 군집의 NN-chain과 일반 알고리즘, 행렬 길과 행렬 없는 길, KD 트리와 훑기)은 **두 길이 같은 정의의 같은 비트를 낸다**. 길을 고르는 기준은 모양(n, p, k)뿐이고 값이나 스레드 수가 아니다.
- **결과를 알고리즘이 아니라 정의로 정한다**(graph.md §1.3과 같은 원칙). 군집의 답은 이산적이다(라벨, 트리, 중심 고르기). 그래서 동점(같은 거리, 같은 높이, 같은 목적값)에서 무엇을 고르는지가 반올림의 차이가 아니라 다른 답을 만든다. 모든 동점 규칙을 문서에 적고, 결정적인 방법의 분할은 입력의 순서와 무관하게 한다(§9의 4번).
- 반올림이 아닌 동작(기본값, 멈춤 규칙, 반복 수, 결과의 모양, 라벨의 뜻)은 함수마다 기준 라이브러리 하나와 같다(API.md §4의 규칙 2): 계층은 scipy 1.18.1, 분할·밀도·혼합·지표는 scikit-learn 1.9.1, sklearn에 없는 것(k-medoids, 퍼지, DIANA, gap, Gower)은 R `cluster` 2.1.8.3이다. 다르게 정하는 곳은 `[scipy와 다름]`, `[sklearn과 다름]`, `[R과 다름]`으로 이유와 함께 적는다(§2.8). 비교 대상끼리 서로 다른 곳이 많아서(탐침: 동점의 트리, centroid의 거리 꼴, ward의 두 꼴, k-평균의 기본 알고리즘) 기준이 아닌 쪽과는 처음부터 다르다.
- 조용히 틀린 값을 돌려주지 않는다. 경고를 내지 않는다. 다른 라이브러리가 경고하거나 조용히 넘어가는 곳(sklearn의 `ConvergenceWarning`, scipy `kmeans2`의 빈 군집 경고, scipy `linkage`가 비유클리드 거리의 ward를 받는 것, scipy `fcluster`·`cut_tree`가 역전 트리에서 서로 다른 답을 내는 것; §2.8)은 에러이거나 상태 필드다.
- 에러 종류와 메시지는 API.md §2·§2.1이다. 메시지는 부른 함수의 이름으로 시작한다(`kmeans(): ...`, 결과의 메서드는 `predict(): ...`). 모듈 함수는 `_calls.function`으로 감싸 호출 모양 에러도 같은 꼴이다.
- 참값은 정확한 유리수(`fractions.Fraction`, python-flint의 `fmpq`·`fmpq_mat`; float64 좌표의 제곱 거리는 유리수로 정확하다)와 Arb(python-flint 0.9.0)만 쓴다. mpmath는 쓰지 않는다.
- 모든 호출은 정해진 시간 안에 끝난다. 반복이 있는 곳(k-평균, EM, PAM의 교환, 퍼지, Affinity Propagation)은 입력과 무관한 상한이 있고 `converged`·`n_iter`가 결과에 있다(API.md §2.2: 반복 상한은 에러가 아니다).
- 받는 입력은 API.md §5의 "데이터를 받는 상위 함수"다: `sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `DataFrame`(열이 변수)과 `Series`(1변수). 점은 행이다((n, p); 1차원 배열은 (n, 1)). 계산은 float64다. 라벨을 받는 지표는 정수·bool·문자열의 1차원 배열을 받는다(sklearn과 같다).

### 1.4 다른 영역과의 관계
- **spatial**: 거리의 정의와 계산은 spatial의 것이다. `metric`의 이름과 뜻(`'euclidean'`, `'sqeuclidean'`, `'cityblock'`, `'chebyshev'`, `'minkowski'` 등)은 `sd.spatial.cdist`와 같고, 거리 값은 `cdist`·`pdist`와 같은 비트다(같은 커널 `spatial/metric.hpp`를 C++에서 부른다). 이웃 찾기(DBSCAN의 반경 이웃, HDBSCAN·OPTICS의 핵심 거리, k-medoids의 가까운 medoid)는 `kd_build`와 그 질의다(동점이 (거리, 색인)으로 정해져 있다, spatial.md). 계산 기하(`Delaunay`)는 쓰지 않는다(2차원 단일 연결을 Delaunay 변 위의 MST로 빠르게 할 수 있지만 첫 판에 넣지 않는다; §5). `squareform`은 spatial의 것을 쓰고 이 영역에 다시 두지 않는다.
- **graph**: 단일 연결은 최소 신장 트리이고 HDBSCAN은 상호 도달 거리 그래프의 최소 신장 트리다. graph의 `minimum_spanning_tree`는 희소 인접 행렬을 받는 Kruskal이라(graph.md GR43) 완전 그래프(n²/2개의 변)에는 맞지 않는다. 이 영역은 밀집 Prim(행렬 없이 거리를 다시 계산)과 KD 트리 위의 Borůvka를 두고, 동점의 키는 graph의 규칙((무게, min(i, j), max(i, j)))과 같게 해서 같은 입력이면 graph의 MST와 같은 트리를 낸다(테스트). DBSCAN의 핵심 점 연결은 graph의 병렬 합집합–찾기(GR31)를 그대로 쓴다(CL3 마일스톤이 `graph/unionfind.hpp`로 꺼냈다). HDBSCAN의 최소 신장 트리도 같은 키 규칙이다. graph는 이 영역을 모른다.
- **linalg**: 혼합 모형의 공분산 Cholesky와 그 역(정밀도 행렬의 Cholesky)은 p×p의 작은 행렬이라 이 영역의 고정 순서 커널로 계산한다(§9의 6번). 실패(양정치가 아님)는 `reg_covar`를 더한 뒤에도 생기면 `LinAlgError`다(sklearn과 같은 뜻, 메시지는 sd의 꼴).
- **random**: 무작위를 쓰는 함수(`kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `kmedoids(method='clara')`, `fuzzy_cmeans`, `gaussian_mixture`, `gap_statistic`, 결과의 `sample`)는 `rng`(`sd.random.Generator`)를 꼭 받는다. 규칙은 회귀의 `rlm(method='MM')`과 같다(docs/api/regression.md): 키워드 인자에 기본값이 없고, 받지 않는 함수나 방법에 주면 `ValueError`, 호출마다 64비트 두 단어를 받아 Philox4×64-10의 키로 쓰고 원소 번호로 표본을 뽑는다. 그래서 같은 `rng` 상태면 같은 비트이고 스레드 수와 무관하며, `n_init`의 여러 시작을 나란히 돌려도 같다(§3.10, §9의 5번).
- **stats**: `GaussianMixtureResult`는 분포 객체의 이름(`logpdf` 대신 sklearn의 `score_samples`를 쓸지는 §9의 1번)과 `sample(shape, /, *, rng)`의 꼴을 따른다. 성분 하나의 로그 밀도는 `sd.stats.MultivariateNormal`과 같은 정의이고 참값도 같은 Arb 식이다. 이 영역은 stats의 객체를 부르지 않는다(n×k번의 객체 호출이 무겁다). 결과의 `sample`은 stats의 정규 난수(`stats/sample_dist.hpp`의 ziggurat)를 C++ 머리 파일로 쓴다(CL80).
- **special**: 혼합 모형과 AMI의 로그 합은 `logsumexp`와 같은 꼴(최댓값을 빼는 합)이고, AMI의 기대 상호정보(EMI)는 `lgamma`를 쓴다.
- **regression**: 결과 객체의 규칙을 따른다: 읽기 전용 배열, 처음 읽을 때 계산하는 필드, `summary()`(`stardust._summary`의 상자), 한 줄 `repr`, `nobs`, `weights`(Σwᵢ·…의 뜻), `nan_policy`의 이름. 겹 나누기(`rng.permutation(n)`)는 회귀의 교차검증 장치이고, 나중의 공통 CV·튜닝 도구가 군집 지표를 점수로 쓸 수 있다(§9의 10번).
- **plot**: 지금 sd.plot에는 덴드로그램이 없다. 이 영역의 `dendrogram`은 그리지 않고 자리 자료(가지의 x·y 좌표, 잎의 순서, 색 묶음의 번호)만 돌려준다. 그림은 나중에 sd.plot이 이 자료를 받아 그린다(§9의 18번). 산점도의 군집 색은 sd.plot의 `scatter(color=labels)`가 이미 한다.
- **나중의 영역**: `sd.embed`의 스펙트럼 임베딩 위에 스펙트럼 군집을 둔다(§9의 36번). `sd.classify`의 k-NN은 spatial의 이웃 찾기를 쓰고 이 영역과 겹치지 않는다. 공통 지표 영역(`sd.metrics` 후보)이 생기면 분류·회귀의 점수는 그쪽에, 군집 지표는 여기에 둔다(§9의 10번).

---

## 2. API
시그니처와 모든 규칙(인자의 검사, 에러와 메시지, 결과의 필드)은 [docs/api/cluster.md](../api/cluster.md)에 있다. 이 절은 꼴과 그 까닭, 그리고 탐침으로 본 비교 대상의 동작을 적는다.

### 2.1 하나의 꼴 (CL1)
- **맞추는 함수는 소문자 이름이고 `<이름>Result`를 돌려준다**: `kmeans` → `KMeansResult`, `kmedoids` → `KMedoidsResult`, `agglomerative`·`divisive` → `HierarchyResult`, `dbscan` → `DBSCANResult`, `hdbscan` → `HDBSCANResult`, `optics` → `OPTICSResult`, `gaussian_mixture` → `GaussianMixtureResult`. 회귀의 `ols` → `OLSResult`, 평활의 `lowess` → `LowessResult`와 같다.
- **공통 필드** (공통 기반 `_results.py`의 `ClusterResult`; smooth의 `SmootherResult`와 같은 장치: 배열은 쓰기 금지, 비싼 필드는 처음 읽을 때 한 번 계산, 쓰기와 지우기는 `AttributeError`): `labels`((n,) int64, 입력 순서; §9의 3번), `n_clusters`, `nobs`, `summary()`, 한 줄 `repr`. 새 점에 라벨을 줄 수 있는 결과(`KMeansResult`, `KMedoidsResult`, `FuzzyCMeansResult`, `GaussianMixtureResult`)는 `predict(X_new, /)`가 있다. 반복하는 결과는 `n_iter`, `converged`, `objective`(k-평균의 관성, PAM의 평균 거리, EM의 로그우도)가 있다.
- **scipy 이름의 계층 도구는 배열을 받고 배열을 돌려준다**(§9의 1번): `linkage` → Z, `fcluster(Z, t)` → 라벨, `cut_tree`, `cophenet`, `leaves_list` 등. 평활의 `make_smoothing_spline`(→ `BSpline`)과 `smoothing_spline`(→ 결과 객체)의 두 얼굴과 같다. `HierarchyResult.linkage`가 같은 Z이고, 결과의 메서드(`cut`, `cophenet`, `dendrogram`)는 그 함수들을 부른다.
- **지표는 수(Python `float`) 또는 배열을 돌려주는 함수다**: sklearn의 이름 그대로(§9의 11번). `homogeneity_completeness_v_measure`는 namedtuple이다(sklearn은 튜플).
- 인자의 순서는 (X, k)다(§9의 2번). `k`는 위치 인자이고 군집 수가 정해지지 않는 방법(계층, 밀도)에는 없다. 계층은 `cut(k=…)`이나 `cut(height=…)`로 자른다.

### 2.2 계층 (CL1 마일스톤)
- **연결법** (§9의 12번): scipy의 일곱 `'single'`, `'complete'`, `'average'`, `'weighted'`, `'centroid'`, `'median'`, `'ward'`. 높이의 뜻은 scipy와 같다. R과 이름·눈금이 다르다(탐침, 12점 난수):
  - scipy `ward`의 높이는 R `ward.D2`와 같다(차 8.9×10⁻¹⁶). R `ward.D`(제곱하지 않은 거리에 Ward의 갱신)와는 1.03 다르다. `ward.D`는 두지 않는다.
  - scipy `centroid`·`median`은 유클리드 거리를 받아 안에서 제곱한다. 높이는 R의 `hclust(dist(x)^2, 'centroid')`의 제곱근과 같다(차 4.4×10⁻¹⁶). R에 제곱하지 않은 거리를 주면 0.46(`centroid`), 0.56(`median`) 다르다. R `mcquitty`는 `weighted`다.
  - 관측값과 비유클리드 `metric`의 `ward`·`centroid`·`median`은 scipy처럼 `ValueError`다. scipy는 미리 계산한 거리(1차원 압축 꼴)에는 어떤 거리든 경고 없이 받는다(cityblock 거리의 `ward`가 돌아간다). sd도 받되(어떤 거리인지 알 수 없다) 문서에 적는다(§9의 12번).
- **역전**: `centroid`·`median`은 높이가 줄어들 수 있다. (0, 0), (2, 0), (1, 1.8)의 `centroid` 높이는 2.0 다음 1.8이다(`is_monotonic` = False). 역전 트리에서 scipy의 세 자르기는 서로 다르고 R과도 다르다(네 점 (0, 0), (2, 0), (1, 1.8), (8, 0)):
  - `fcluster(Z, 3, 'maxclust')`는 두 군집 [1, 1, 1, 2], `cut_tree(Z, n_clusters=3)`도 두 군집 [0, 0, 0, 2](라벨 1이 비었다), R `cutree(h, k=3)`은 세 군집 [1, 1, 2, 3].
  - 높이 1.9로 자르면 `fcluster(t=1.9, 'distance')`는 네 홑원소, `cut_tree(height=1.9)`는 [0, 0, 0, 2], R `cutree(h=)`는 에러(`the 'height' component of 'tree' is not sorted`).
  - 난수 30점 200벌 × k ∈ {2, 3, 5, 8}에서 `cut_tree`와 `fcluster(maxclust)`의 분할이 `centroid` 118번, `median` 129번 달랐다(800번 중). 단조인 연결법에서는 한 번도 다르지 않았다.
  - sd는 §9의 16번(CL19)대로 k로 자르기는 병합 순서, 높이로 자르기는 역전 트리에서 `ValueError`다.
- **동점** (§9의 14번): 같은 높이의 후보가 여럿이면 무엇을 먼저 합치는지가 트리를 바꾼다. 탐침:
  - 한 줄의 다섯 점(간격 1)은 네 연결법 모두 scipy와 R의 병합 순서가 같다. 3×3 격자(가장 가까운 거리가 모두 1)에서 `complete`·`average`는 scipy와 R이 **다른 트리**를, `ward`는 같은 트리를 다른 병합 순서로 낸다. `single`은 같다.
  - scipy의 트리는 입력의 순서에 달려 있다: 같은 다섯 점·아홉 점을 섞어 주면 네 연결법 모두 다른 트리가 나온다(한 줄의 `single`은 왼쪽 끝부터 합치던 것이 섞은 순서의 3, 4부터 합친다).
  - 동점의 높이에서 `fcluster(maxclust)`는 청한 수보다 적은 군집을 낸다: 한 줄 네 점의 `single`에 k = 2를 청하면 한 군집이다(높이 1로 자르면 모두 합쳐지므로 scipy의 문서대로 "t개 이하"다). `cut_tree`와 R `cutree`는 두 군집이다.
- **잎의 순서**: scipy `leaves_list`와 R `hclust`의 `order`는 탐침의 두 트리(난수 10점의 `complete`, `average`)에서 같았다. `optimal_leaf_ordering`(Bar-Joseph 외 2001)은 이웃한 잎의 거리 합이 가장 작은 순서다.
- **자르기 도구**: `fcluster`(scipy의 다섯 기준과 기본 `criterion='inconsistent'`), `cut_tree`(여러 k를 한 번에, (n, len(k)) 배열), `HierarchyResult.cut(*, k=None, height=None)`(R `cutree`의 뜻). CL1 마일스톤에서 scipy 1.18.1과 견준 것(CL57; `tests/cluster/test_hierarchy.py`): 동점 없는 트리에서 `fcluster`의 다섯 기준, `leaders`, `leaves_list`, `maxdists`, `maxinconsts`, `maxRstat`, `cophenet`, `dendrogram`의 좌표·잎·색 묶음은 scipy와 같다. 다른 곳: 높이로 자르기는 h 이하(scipy `cut_tree`는 h 미만), 같은 높이의 `cut_tree(n_clusters=)`는 행의 차례(scipy는 너비 우선 차례로 되돌려 다른 분할이 된다), maxclust의 t < 1은 에러(scipy는 정해지지 않은 답), `inconsistent`의 표준편차는 두 번 훑기(scipy는 높이 하나에서 10⁻⁹ 꼴).
- **최적 잎 순서**: scipy 1.18.1의 `optimal_leaf_ordering`은 최소가 아닌 순서를 낼 때가 있다(무작위 3–8점의 트리 12개 가운데 여러 개에서, n = 4에서도 비용이 4.95 대신 5.73; 가능한 모든 순서를 견주는 테스트). sd는 정확한 최소다(CL53).
- **덴드로그램**: `dendrogram(Z, *, …)`은 scipy의 `no_plot=True`가 돌려주는 사전(`icoord`, `dcoord`, `ivl`, `leaves`, `color_list`, `leaves_color_list`)과 같은 내용을 namedtuple로 돌려준다(§9의 18번). 색은 matplotlib 색 이름이 아니라 묶음 번호다.
- **DIANA** (§9의 19번): `divisive(X)`는 R `diana`의 분할식 계층이다(가장 지름이 큰 군집을 고르고, 평균 거리가 가장 큰 점을 떼어 "갈라진 무리"를 키운다). 결과는 같은 `HierarchyResult`이고 높이는 지름이다. R `agnes`는 `agglomerative(method='average')`와 같은 높이다(탐침: R `hclust(…, 'average')`와 `agnes`의 높이가 같다). 응집 계수(`agnes`의 `ac`, `diana`의 `dc`)는 `HierarchyResult.coefficient`다. CL1 마일스톤에서 R과 견줬다: 동점 없는 참값 집합 모두에서 `dc`와 높이가 같고(차 10⁻¹⁵ 꼴), 서로 다른 높이 사이의 모든 자르기가 R `cutree(as.hclust(diana(x)), h=)`와 같다. R의 `merge`는 지름이 같은 나누기가 이어질 때(갈라진 점이 지름의 짝이 아니면 흔하다) 배너에서 다른 중첩을 만들어 같은 높이의 k 자르기가 다를 수 있다(CL52). 격자·정육각형의 정확한 동점은 열쇠라 R과 다를 수 있다.

### 2.3 분할 (CL2 마일스톤)
- **`kmeans`**: 목적은 관성 Σᵢwᵢ‖xᵢ − c_{lᵢ}‖²이다. Lloyd 반복이고, 결과는 Lloyd의 고정점(모든 점의 라벨이 계산한 거리의 가장 가까운 중심이고 모든 중심이 그 점들의 무게 평균) 또는 반복 상한에서 멈춘 점이다(§4.2). 탐침으로 본 비교 대상:
  - sklearn의 `n_init='auto'`는 `init='k-means++'`에서 1, `'random'`에서 10이다(1.4에서 10이 'auto'로 바뀌었다). R `kmeans`는 `nstart=1`, `iter.max=10`, 기본 알고리즘이 Hartigan–Wong이다. 시험 자료 probe/300에서 같은 시작의 Lloyd는 관성 574.888(sd의 정확한 Lloyd 참값, sklearn과 같다)에, Hartigan–Wong은 더 낮은 573.897에 닿았다(점을 하나씩 옮기는 Hartigan–Wong이 다른 극소로 간다).
  - 같은 거리의 중심이 둘이면 sklearn과 scipy `vq`가 모두 번호가 작은 중심을 고른다(중심의 순서를 뒤집어도 0). sd도 같다(§9의 21번).
  - 서로 다른 점이 k보다 적으면 sklearn은 `ConvergenceWarning`과 함께 k보다 적은 라벨을 내고, R은 에러(`more cluster centers than distinct data points.`)다. sd는 `ValueError`다(§9의 21번).
  - 빈 군집: sklearn은 가장 먼 점으로 옮긴다(탐침의 네 점 [0, 0.1, 10, 10.1]과 시작 중심 [0, 10, 100]에서 라벨 [0, 0, 1, 2]). R의 Lloyd는 경고(`empty cluster: try a better set of initial centers`)와 함께 두 군집, Hartigan–Wong은 같은 메시지의 에러, scipy `kmeans2`는 반복마다 경고하고 두 군집이다. sd는 sklearn의 옮기기를 따른다(§9의 21번).
  - NaN은 sklearn이 `ValueError`다(§9의 8번).
  - sklearn의 `algorithm='lloyd'`와 `'elkan'`은 같은 라벨에 중심이 2.2×10⁻¹⁶ 달랐다(2만 점, 4차원, k = 8). 같은 입력을 OpenMP 1스레드와 8스레드로 돌리면 라벨은 같고 중심이 6.4×10⁻¹⁵, 관성의 끝자리가 다르다(20만 점, 8차원, k = 20, 50번 반복). sd는 가속과 스레드 수가 비트를 바꾸지 않는다(§3.3, §3.10).
- **k-means++** (§9의 21번): sklearn의 탐욕 k-means++(Arthur–Vassilvitskii 2007에 후보를 여럿 뽑는 꼴)이다. 단계마다 2 + ⌊ln k⌋개의 후보를 D² 확률로 뽑고 관성을 가장 줄이는 것을 고른다(sklearn `_kmeans_plusplus`의 `n_local_trials = 2 + int(np.log(n_clusters))`).
- **`kmedoids`** (§9의 23번): R `pam`의 BUILD와 SWAP이 정의이고(Kaufman–Rousseeuw 1990), 목적은 가장 가까운 medoid까지의 거리 합이다. `method='clara'`는 R `clara`의 표본 방법(표본 5개, 표본 크기 min(n, 40 + 2k); R의 기본 `rngR=FALSE`는 R의 난수가 아닌 자체 난수라 sd와 같은 표본을 뽑을 수 없다)이다.
- **`fuzzy_cmeans`** (§9의 24번): Bezdek의 퍼지 c-평균(목적 Σᵢⱼuᵢⱼᵐ‖xᵢ − cⱼ‖², 기본 m = 2)이다. R `fanny`는 다른 목적(거리 행렬 위의 Σ_v Σᵢⱼ u_{iv}ʳu_{jv}ʳd(i, j)/(2Σⱼu_{jv}ʳ))이라 같은 함수가 아니다.
- **CL2 마일스톤에서 견준 것** (`tests/cluster/test_kmeans.py`, `test_kmedoids.py`, `test_fuzzy.py`): 같은 시작(참값 도구의 시작, `tol=0`)에서 sklearn `KMeans`와 R `kmeans(algorithm='Lloyd')`가 근처 동점이 없는 참값 집합 모두에서 sd와 같은 라벨과 관성(상대 10⁻¹⁰, 10⁻¹²)을 낸다. 비정상 수 집합에서 sklearn은 한 군집이다(전개식 거리가 모두 0). R `pam`은 동점 없는 집합에서 medoid와 목적(상대 10⁻¹³)이 같고, 같은 목적이 여럿인 곳(1차원의 짝수 군집의 가운데 두 점, 격자)에서는 계산한 합의 끝자리로 고른다(정확한 동점; 참값 63가지 중 8가지가 다른 medoid, 목적은 같다). R `clara(samples=1, sampsize=n, pamLike=TRUE)`는 sd의 `method='clara', sample_size=n, samples=1`과 같은 medoid와 목적이다. `minibatch_kmeans`·`bisecting_kmeans`는 난수 흐름이 달라 같은 답을 견줄 수 없어, 같은 자료에서 sklearn과 관성이 2% 안인지만 본다.

### 2.4 밀도 (CL3 마일스톤)
- **`dbscan`** (§9의 25번): 핵심 점은 eps 안(경계 포함, d ≤ eps)의 점 수(자기 포함, 무게가 있으면 무게 합)가 `min_samples` 이상인 점이다(sklearn과 같다; 탐침: 거리가 꼭 eps인 두 점과 `min_samples=2`는 한 군집). 군집은 eps 안에서 이어진 핵심 점의 연결 요소다. **경계 점**(핵심이 아니지만 어떤 핵심 점의 eps 안인 점)은 sklearn에서 **먼저 펼친 군집**에 든다: 탐침에서 두 군집의 가운데 점이 입력 순서를 바꾸면 왼쪽 군집에서 오른쪽 군집으로 옮겨 갔다. sd는 가장 가까운 핵심 점의 군집(동점은 §9의 4번의 열쇠)에 넣는다(`[sklearn과 다름]`). 잡음은 라벨 −1이다.
- **`hdbscan`** (§9의 26번): 기준은 sklearn 1.9의 `HDBSCAN`이다. `min_samples`는 자기를 센다. 원본 `hdbscan` 꾸러미는 자기를 세지 않아 같은 결과를 얻으려면 sklearn의 값이 1 크다(sklearn의 문서). 핵심 거리 → 상호 도달 거리(`alpha`로 나눈 거리와 두 핵심 거리의 최댓값) → 최소 신장 트리 → 단일 연결 트리 → `min_cluster_size`로 줄인 트리 → 안정성으로 군집 고르기(`'eom'` 또는 `'leaf'`)다. 결과에 `probabilities`, 원본의 GLOSH 이상치 점수 `outlier_scores`(sklearn에는 없다), 줄인 트리, 단일 연결 트리(Z)를 둔다.
  - sklearn 1.9.1은 `copy`의 기본값이 1.10에서 바뀐다는 `FutureWarning`을 낸다. sd에는 `copy`가 없다(입력을 고치지 않는다).
  - 같은 점 30개를 더하면 그 점들이 따로 한 군집이 된다(탐침: 두 덩어리 + 같은 점 30개 → 군집 셋). 같은 점의 핵심 거리가 0이라 정의대로의 결과다. sd도 같다.
- **`optics`** (§9의 27번): sklearn의 `OPTICS`(도달 거리의 순서, ξ 추출과 DBSCAN 추출)다. 순서에서 같은 도달 거리의 동점은 §9의 4번의 열쇠다.
- **CL3 마일스톤에서 견준 것** (`tests/cluster/test_dbscan.py`, `test_hdbscan.py`, `test_optics.py`; CL67–CL74):
  - DBSCAN은 참값 24가지에서 정확한 분할과 같고(eps에 꼭 놓인 짝의 eps/6과 격자 포함), 무작위 자료 12벌(무게 포함)에서 sklearn과 핵심 점·잡음·핵심 점의 분할이 같으며 경계 점만 다르다(sd는 가장 가까운 핵심 점). 탐침의 가운데 점(양쪽 핵심 점까지 정확히 1)은 sd에서 입력 순서와 무관하게 작은 열쇠 쪽(왼쪽)이다.
  - HDBSCAN은 참값 14벌(`min_samples` = 5, `min_cluster_size` = 5)에서 핵심 거리·트리의 무게·`'eom'`과 `'leaf'`의 라벨·확률·GLOSH·안정성이 정의(CL67–CL69)와 같거나 계약 안이다. sklearn과는 `min_samples=1`(같은 높이가 없다)의 무작위 자료 16벌 × 여섯 옵션(`'leaf'`, `min_cluster_size`, `allow_single_cluster`, `max_cluster_size`, `alpha`)에서 라벨과 확률(상대 10⁻¹³), 중심과 medoid가 같다. `min_samples` > 1에서는 sklearn 자신이 입력 순서에 따라 다르고(CL68), 높이로 자르기(`dbscan_clustering`)는 트리의 성분만의 함수라 sklearn과 같다. `cluster_selection_epsilon`은 sklearn이 이 판에서 실패해 손으로 푼 사례로 본다.
  - OPTICS는 참값 14벌에서 차례·선행자가 정확한 차례와 같고 도달 거리가 계약 안이다. 점을 열쇠 차례로 주면(그때 sklearn의 동점 규칙이 같다) 무작위 자료 6벌(`max_eps` 유한 포함)에서 sklearn과 차례·선행자·ξ 군집과 그 계층이 같고, 도달 거리·핵심 거리는 sklearn의 15자리 반올림만큼(상대 4×10⁻¹⁶) 다르다. sd의 배열에 sklearn의 `cluster_optics_xi`·`cluster_optics_dbscan`을 돌리면 sd의 추출과 같다(ξ 넷 × 옵션 셋, eps 넷).

### 2.5 혼합 모형 (CL4 마일스톤)
- **`gaussian_mixture`**: sklearn `GaussianMixture`의 EM이다. 공분산 꼴은 `'full'`, `'tied'`, `'diag'`, `'spherical'`(§9의 29번)이고, `reg_covar`를 공분산의 대각에 더한다. 멈춤은 sklearn과 같은 "점마다 평균한 하한(로그우도)의 변화 < `tol`"이다. 탐침에서 잘 떨어진 두 덩어리는 2번에 멈췄고, sklearn의 `lower_bound_`(마지막 E 단계의 값)와 `score(X)`(마지막 M 단계 뒤의 값)는 3×10⁻⁶ 달랐다. sd는 두 값을 다른 이름으로 둔다(`lower_bound`, `loglik`; §9의 30번).
- `bic`, `aic`는 결과의 필드다(sklearn은 X를 다시 받는 메서드). 모수의 수는 sklearn의 `_n_parameters`와 같다.
- `predict`(가장 큰 책임도), `predict_proba`, `score_samples`(점마다의 로그 밀도), `sample(shape=None, /, *, rng)`.
- **CL4 마일스톤에서 견준 것** (`tests/cluster/test_mixture.py`; CL75–CL81): 같은 시작(sd의 시작 모수를 `weights_init`·`means_init`·`precisions_init`로 sklearn에 준다; `'kmeans'`와 `'random'`, 네 공분산 꼴)에서 sklearn 1.9.1과 반복 수가 같고 `lower_bound`가 상대 10⁻¹³, 무게·평균이 10⁻¹⁰, 공분산이 10⁻⁹ 안이며 라벨, `score(X)·n`(= `loglik`), `bic(X)`, `aic(X)`, `_n_parameters`, 새 점의 `score_samples`·`predict_proba`·`predict`가 같다. 기본값으로 각자의 k-평균 시작을 쓰면 잘 떨어진 덩어리에서 같은 고정점이다(속도 사례의 자료에서 `lower_bound`가 비트까지 같았다). 10⁸만큼 옮긴 자료의 diag·spherical에서 sklearn의 펼친 이차형식은 q가 10⁻³ 넘게 틀리고 sd는 정확한 유리수 값과 상대 10⁻¹² 안이다.

### 2.6 지표와 k 고르기 (CL5 마일스톤 — 마침)
- **라벨** (CL82): 정수, bool, 문자열과 값이 정수인 실수. 군집은 서로 다른 값의 정렬 순서다.
- **내부 지표**: `silhouette_samples`, `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`. 홑원소 군집의 실루엣은 0이다(sklearn과 R `silhouette`이 같다; 탐침 [0.98, 0.979592, 0, 0.95, 0.952381]). 군집이 하나이거나 n개면 sklearn은 세 지표 모두 `ValueError`(`Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)`)이고 sd도 같다(`silhouette_score(): the number of clusters must be between 2 and n - 1 (got 1)`). 실루엣의 길은 CL85, CH·DB는 CL86.
- **외부 지표**: 분할표의 정수 셈으로 정의된다. sklearn의 뜻 그대로다: `normalized_mutual_info_score`(기본 `average_method='arithmetic'`)는 `v_measure_score`(β = 1)와 같은 값이다(탐침에서 비트까지; sd는 V를 (1 + β)MI/(H_true + βH_pred)로 재어 언제나 같은 비트다, CL84). ARI(한 군집, 한 군집) = 1, ARI(모두 홑원소, 모두 홑원소) = 1, AMI(한 군집, 한 군집) = 1, NMI(한 군집, 모두 홑원소) = 0이다(sklearn의 특수 경우; sd도 같다). ARI·RI·FM은 올바르게 반올림한 값(CL83), AMI의 한쪽이 모두 홑원소인 경우는 CL84.
- **`gap_statistic`** (§9의 34번): R `clusGap`의 뜻이다. 기본 `n_reference=100`(R의 `B`), `d_power=1`, `reference='scaledpca'`, k 고르기는 `maxSE`의 `'firstsemax'`와 `se_factor=1`이다(이름은 R의 소문자 꼴, CL87; 계산은 CL88). R의 `d.power=1`은 Tibshirani 외(2001)의 제곱 거리(d.power = 2)와 다르다(R의 문서가 적는다). 탐침의 세 덩어리 90점(blobs/k3p2)에서 R은 k = 3을 골랐고 sd도 3이다. `method='pam'`의 log W는 R `clusGap(x, pam)`의 `logW`와 10⁻¹⁵ 안이고(같은 medoid), `maxSE`의 다섯 규칙은 무작위 60가지 × SE 계수 둘에서 R과 같다.
- **`gower_distance`** (§9의 35번): R `daisy(metric='gower')`의 뜻이다(수는 범위로 나눈 절댓값, 범주는 같은지, 순서형은 순위, 비대칭 이진은 둘 다 0인 짝을 뺀다). R은 논리형 열을 비대칭 이진으로 바꾸며 경고한다(탐침). 압축 꼴 거리를 돌려주어 `linkage`, `kmedoids`, `dbscan`의 `metric='precomputed'`에 바로 들어간다. 섞인 자료(수, 명목, 논리, 비대칭 이진, 순위, 로그 비, 결측, 무게)에서 R과 상대 4×10⁻¹⁶ 안이다(CL89).

### 2.7 결과 객체
| 이름 | 만드는 함수 | 이 결과만의 필드와 메서드 |
|---|---|---|
| `KMeansResult` | `kmeans`, `minibatch_kmeans` | `centers`, `inertia`, `cluster_inertia`, `n_iter`, `converged`, `n_init`, `best_init`, `method`, `init`, `predict`, `transform`(중심까지의 거리); 미니배치는 `n_steps`도 |
| `BisectingKMeansResult` | `bisecting_kmeans` | 위와 같고 `tree`(잎이 군집인 Z 꼴; CL64), `strategy` |
| `KMedoidsResult` | `kmedoids` | `medoids`(점의 색인), `medoid_points`, `objective`(SWAP 뒤), `build_objective`, `n_swaps`, `converged`, `method`, `metric`, `predict`(CL65) |
| `FuzzyCMeansResult` | `fuzzy_cmeans` | `memberships`(n, k), `centers`, `m`, `objective`, `partition_coefficient`, `cluster_membership`, `n_iter`, `converged`, `predict`, `predict_proba` |
| `HierarchyResult` | `agglomerative`, `divisive` | `linkage`(Z), `method`, `heights`, `order`, `coefficient`, `cut`, `cophenet`, `dendrogram` |
| `DBSCANResult` | `dbscan` | `core_mask`, `eps`, `min_samples`, `metric` |
| `HDBSCANResult` | `hdbscan` | `probabilities`, `outlier_scores`, `condensed_tree`, `single_linkage`(Z), `persistence`, `centroids`, `medoids`, `min_cluster_size`, `min_samples`, `cluster_selection_method`, `metric`, `dbscan_clustering`(CL74) |
| `OPTICSResult` | `optics` | `ordering`, `reachability`, `core_distances`, `predecessor`, `cluster_hierarchy`, `min_samples`, `max_eps`, `cluster_method`, `eps`, `xi`, `metric`, `extract_xi`, `extract_dbscan`(새 결과; CL71) |
| `GaussianMixtureResult` | `gaussian_mixture` | `weights`, `means`, `covariances`, `precisions_cholesky`, `loglik`, `lower_bound`, `bic`, `aic`, `n_parameters`, `n_iter`, `converged`, `covariance_type`, `n_init`, `best_init`, `predict`, `predict_proba`, `score_samples`, `sample`(namedtuple `MixtureSample(X, labels)`; CL80, CL81) |
| `GapResult` | `gap_statistic` | `k`, `k_max`, `gap`, `se`, `log_w`, `e_log_w`, `log_w_reference`, `nobs`, `omitted`, `method`, `reference`, `select`, `se_factor`, `n_reference`, `d_power`(`labels`·`n_clusters`는 없다; CL87) |
| `AffinityPropagationResult` | `affinity_propagation` | `exemplars`, `exemplar_points`, `n_iter`, `converged`, `damping`, `preference`, `affinity`, `predict`(CL93) |
| `DendrogramResult` | `dendrogram` | namedtuple (`icoord`, `dcoord`, `leaves`, `ivl`, `color_list`, `leaves_color_list`) |

### 2.8 다른 라이브러리와 다른 곳 (모음)
| 곳 | 다른 라이브러리 | sd |
|---|---|---|
| 꼴 | sklearn 추정기 클래스, scipy 배열, R 목록 | 함수 → 결과 객체, scipy 이름의 계층 도구는 배열(CL1) |
| 연결법의 기본값 | scipy `'single'`, R `'complete'`, sklearn `'ward'` | scipy와 같은 `'single'`(CL16) |
| 계층의 동점 | scipy·R이 서로 다른 트리, 입력 순서에 따라 다른 트리 | 정의(가장 작은 (높이, 열쇠)의 쌍을 먼저)와 입력 순서와 무관한 열쇠(§9의 4·14번) |
| 역전 트리 자르기 | scipy의 세 자르기가 서로 다르고 R과도 다르다 | k는 병합 순서, 높이는 `ValueError`(§9의 16번) |
| 높이로 자르기의 경계, 같은 높이의 k 자르기 | scipy `cut_tree(height=)`는 h 미만, 같은 높이는 너비 우선 차례 | h 이하(R·`fcluster`와 같다), 행의 차례(R `cutree(k=)`; CL57) |
| 최적 잎 순서 | scipy 1.18.1은 최소가 아닌 순서를 낼 때가 있다 | 정확한 최소(CL53) |
| DIANA의 같은 지름 | R의 `merge`는 배너에서 만들어 같은 지름의 나누기를 다르게 중첩한다 | 실제로 나눈 트리(CL52) |
| 미리 계산한 거리의 ward·centroid·median | scipy는 어떤 거리든 받는다 | 받되 유클리드 거리를 가정한다고 적는다(§9의 12번) |
| k-평균의 스레드 의존 | sklearn은 스레드 수에 따라 중심의 끝자리가 다르다 | 스레드와 무관한 비트(§9의 6번) |
| k-평균의 거리 | sklearn은 ‖x‖² − 2x·c + ‖c‖²(상쇄) | 차의 제곱합(§9의 7번) |
| 군집보다 적은 서로 다른 점 | sklearn 경고와 적은 라벨, R 에러 | `ValueError` |
| DBSCAN의 경계 점 | sklearn은 먼저 펼친 군집(입력 순서) | 가장 가까운 핵심 점의 군집(§9의 25번) |
| HDBSCAN의 같은 높이 | sklearn은 Prim이 만난 차례의 이진 트리라 입력 순서에 따라 라벨이 다르다(80가지 중 50) | 같은 높이의 병합을 한꺼번에(Campello 외 2013; CL68): 입력 순서와 무관 |
| HDBSCAN의 미리 계산한 거리와 `alpha` | sklearn은 행렬을 α로 나눠 핵심 거리도 나뉜다 | 핵심 거리는 나누지 않는다(KD 길의 뜻; CL67) |
| HDBSCAN의 실패 | sklearn: `allow_single_cluster`의 뿌리 아닌 단일 군집에서 `KeyError`, numpy 2.5에서 `cluster_selection_epsilon`의 `TypeError` | 정의대로(CL68) |
| OPTICS의 값과 시작 | sklearn은 소수 15자리로 반올림하고 입력 색인 0에서 시작한다 | 계산한 거리 그대로, 가장 작은 열쇠에서 시작(CL70) |
| 경고 | sklearn `ConvergenceWarning`·`FutureWarning`, scipy `ClusterWarning`·빈 군집 경고 | 없음: 상태 필드 또는 에러 |
| 실루엣의 표본 | sklearn `sample_size`, `random_state` | 언제나 전체(빠른 O(n²)) |
| 먼 평행 이동(10⁸ + 덩어리) | sklearn의 전개식 거리: `KMeans.predict`가 2100점 중 308점(10¹²에서 1425점)을 틀리고, `silhouette_samples`가 0.21까지 틀린다 | 차의 제곱합(CL10): 0점 |
| 비정상 수 좌표 | sklearn `KMeans`는 모든 점을 한 군집에, scipy `linkage`와 R `hclust`는 높이 0, R `pam`은 목적 0을 낸다(제곱의 아래넘침) | 2의 거듭제곱 척도(CL43)와 spatial의 조심 길 |
| 큰 좌표(10¹⁵⁰) | R `hclust`의 ward.D2·centroid·median이 제곱 거리의 갱신에서 넘쳐 다른 트리를 낸다 | 같은 척도(CL43) |
| PAM의 동점 | R `pam`은 같은 목적의 medoid 가운데 입력 색인으로 먼저 찾은 것; 섞은 격자에서 R이 다른(더 낮은) 극소에 닿았다 | 열쇠(CL7) |
| 대각 공분산의 M 단계 | sklearn은 E[x²] − 2E[x]μ + μ²의 한 번 훑기(diag/k4p3에서 상대 6.6×10⁻¹²) | 평균을 뺀 두 번째 훑기(§3.7) |
| 혼합 모형의 이차형식과 tied의 M 단계 | sklearn은 full·tied에서 XU − μU의 곱, diag·spherical에서 Σμ²λ − 2xμλ + x²λ로 펼치고(10⁸ 평행 이동에서 q가 10⁻³ 넘게 틀린다), tied의 공분산은 XᵀX − Σnkμμᵀ의 한 번 훑기(주어진 책임도에서 sd 계약의 9.3배) | 차의 꼴 Uᵀ(x − μ)와 평균을 뺀 두 번째 훑기(CL75) |
| 혼합 모형의 `max_iter=0`과 여러 시작 | sklearn은 마지막 시작의 모수 | 첫 시작(하한이 모두 −inf: (하한, 번호)의 규칙; CL75) |
| 혼합 모형의 표본 | sklearn은 다항 분포의 개수를 뽑아 성분 차례로 묶는다(`n_samples`) | 표본마다 독립인 성분, (*shape, p)의 모양(CL80) |
| 빈 군집의 옮기기 | sklearn은 평균을 뺀 자료의 반올림으로 가장 먼 점을 고른다 | 정확한 계산의 sqd로 고른다(CL62) |
| 이분 k-평균의 시작 고르기 | sklearn은 관성이 상대 10⁻⁶ 넘게 나아질 때만 바꾼다 | (관성, 시작 번호)(CL64, `kmeans`와 같다) |
| CLARA의 SWAP | R의 기본 `pamLike=FALSE`는 원래 Fortran의 한 글자 다른 SWAP | PAM 그대로(`pamLike=TRUE`; CL65) |
| 미니배치의 `init_size` < k | sklearn은 경고하고 3k로 바꾼다 | `ValueError`(CL64) |
| 혼합 모형의 `'kmeans'` 시작과 서로 다른 점 | sklearn은 k-평균의 경고와 적은 군집 | `ValueError`(`kmeans`와 같다; CL76) |
| R의 이름 | `ward.D2`, `mcquitty`, `nstart`, `iter.max`, `memb.exp` | `ward`, `weighted`, `n_init`, `max_iter`, `m` |
| 외부 지표의 셈 | sklearn은 짝 수를 int64로 세지만 마지막 나눗셈을 float 둘의 나눗셈으로, FM을 √ 둘의 곱으로 한다 | 정확한 유리수의 올바른 반올림(CL83) |
| AMI의 기대 MI | sklearn은 lgamma 차(n = 10⁶에서 10⁻⁸ 상쇄)이고 칸 수 전부를 훑는다(n = 10⁶에서 397 ms) | 초기하 비의 점화식, 같은 크기의 짝을 한 번(10 ms; CL84) |
| 한쪽이 모두 홑원소인 AMI | sklearn은 반올림의 잡음(0 근처, `'min'`은 1.0) | 정확히 0, `'min'`은 1(CL84) |
| 동질성·완전성·NMI의 1 넘음 | sklearn은 1.0000000000000002 | 1에서 자른다(CL84) |
| 실수 라벨 | sklearn은 소수가 있어도 경고하고 받는다 | 값이 정수일 때만(CL82) |
| Davies–Bouldin의 0 | sklearn은 `np.allclose(…, 0)`(10⁻⁸ 아래)이면 0 | 정확히 0일 때만(CL86) |
| 먼 평행 이동의 CH·DB | sklearn은 원점에서 잰 평균(shifted/k3p2에서 상대 10⁻⁹ 꼴) | 첫 점을 뺀 자료(CL86) |
| gap의 군집과 이름 | R `clusGap(x, kmeans)`는 Hartigan–Wong의 무작위 시작 하나, `spaceH0 = "scaledPCA"`, `maxSE(method = "firstSEmax")` | sd `kmeans`(k-means++), `reference='scaledpca'`, `select='firstsemax'`(CL87, CL88) |
| Gower의 작은 무게 | R은 남은 무게의 합이 0.5 이하이면 NA | 같이 가진 변수가 없을 때만 NaN(CL89) |
| Affinity Propagation의 잡음 | sklearn은 유사도에 `random_state`의 작은 잡음을 더한다 | 잡음 없이 열쇠의 동점 규칙(CL40, CL92) |
| Affinity Propagation의 유사도 입력 | sklearn `affinity='precomputed'` | 같은 이름(`metric`이 아니다; CL91) |

### 2.9 그 밖 (CL6 마일스톤 — 마침)
- **`affinity_propagation`** (§9의 37번, CL40): sklearn `AffinityPropagation`의 반복과 멈춤과 기본값(`damping=0.5`, `max_iter=200`, `convergence_iter=15`, `preference=None`은 유사도의 중앙값)이다. 유사도는 `affinity='euclidean'`(음의 제곱 유클리드 거리) 또는 `'precomputed'`(주어진 행렬; CL91). 잡음을 더하지 않고 열쇠의 동점 규칙을 써서 `rng`가 없고, 입력 순서와 무관한 분할이며, 잡음이 0인 sklearn과 같은 답이다(CL92). 결과와 멈춤은 CL93, 메모리와 나란히는 CL94·§3.12.

---

## 3. 수치 방법

### 3.1 거리 커널
- 거리는 spatial의 커널(`spatial/metric.hpp`)을 C++에서 바로 부른다: 관측값 사이의 거리가 `sd.spatial.cdist`와 같은 비트다. 유클리드는 차의 제곱합(축 순서의 차례 합; spatial의 정의)의 제곱근이다.
- **k-평균과 혼합 모형의 제곱 거리** (§9의 7번): 차의 제곱합 Σⱼ(xᵢⱼ − cⱼ)²을 NEON으로 계산한다. sklearn은 ‖x‖² − 2x·c + ‖c‖²(행렬 곱)를 써서 점이 원점에서 멀고 중심과 가까우면 자릿수를 잃는다(spatial의 탐침에서 sklearn의 전개식은 far/1e8·far/1e15에서 k번째 거리가 참값과 10¹⁸ ULP 넘게 달랐다; ROADMAP.md의 SPT 기록). 차의 꼴은 곱셈이 p번 많지만 정확하고 상쇄가 없다. CL0 마일스톤의 시제품(§5)에서 차의 꼴은 1스레드로 Accelerate의 전개식(`cblas_dgemm`)보다 1.8–8.3배 느렸고(p·k가 클수록), 10스레드로는 0.3–1.0배였다. CL9의 규칙(Accelerate 꼴보다 1.5배 넘게 느린 경로)에 걸려 §9의 38번으로 물었고, 차의 꼴을 정의로 두고 증명된 거르기로 1스레드를 되찾기로 했다(CL49). 증명된 오차 한계로 "전개식으로 후보를 거르고 차의 꼴로 다시 재는" 길(같은 비트)은 시제품에서 점마다 후보 1.00개(10⁸·10¹² 평행 이동에서는 3개, 그래도 라벨은 정확)였다.
- **미리 계산한 거리**(`metric='precomputed'`; §9의 9번): (n, n) 정사각 또는 1차원 압축 꼴. 정사각은 대칭이고 대각이 0이며 음수와 NaN이 없어야 한다(`ValueError`; 대칭은 비트까지 본다). 압축 꼴의 길이는 n(n − 1)/2여야 한다.

### 3.2 계층 군집
구현은 `csrc/cluster/hierarchy.cpp`(트리, 높이, 행, DIANA, `cophenet`, 최적 잎 순서)와 `mst.cpp`(최소 신장 트리), Python 층은 `src/stardust/cluster/_hierarchy.py`다(CL1 마일스톤).
- **정의** (§9의 14번): 계층 군집의 트리는 "모든 군집 쌍 가운데 (연결 거리, 열쇠)가 가장 작은 쌍을 합친다"를 n − 1번 한 것이다(Müllner 2011의 "primitive" 알고리즘). 열쇠는 두 군집의 가장 작은 점 번호의 짝(작은 쪽, 큰 쪽)이고, 점 번호는 §9의 4번의 순서(좌표의 사전순 정렬 뒤의 번호; 미리 계산한 거리는 입력 색인)다. 연결 거리는 그 연결법의 정의(single: 두 군집 사이의 가장 작은 거리, complete: 가장 큰 거리, average: 평균, weighted·median: Lance–Williams의 재귀, centroid: 무게 중심 사이의 거리, ward: √(2|A||B|/(|A| + |B|))·‖c_A − c_B‖)다.
- **열쇠의 공간**: 커널은 점을 열쇠 순서로 받고(Python 층이 `_core.cluster.lex_order`로 정렬한 사본을 준다), 군집은 가장 작은 열쇠의 칸에 산다(합친 군집은 작은 칸에 남는다). 그래서 모든 동점 규칙의 "작은 칸"이 곧 "작은 열쇠"이고, 같은 점들을 섞어 주면 같은 Z가 점 번호만 바뀌어 나온다(테스트가 비트까지 본다; 같은 점이 여럿이면 그 점들끼리는 바뀔 수 있다).
- **single**: 최소 신장 트리(변의 순서 (거리, 작은 열쇠, 큰 열쇠); 거리는 spatial의 유클리드 거리와 그 조심 길). 관측값이면 **행렬 없이** Prim(O(n²p) 시간, O(n) 메모리; 단계마다 바깥 점을 점 덩이로 나눠 갱신하고 덩이의 최솟값을 덩이 순서로 고른다)이나 KD 트리 위의 Borůvka(CL54: 점마다 뿌리부터 찾고, 자기 성분만 든 마디와 반경 밖의 마디를 spatial의 증명된 한계로 자른다; 반경은 자기 최선과 그 성분이 지금까지 찾은 최선 중 작은 것)다. 압축 꼴 입력과 비유클리드 거리는 행렬 위의 Prim이다. 길의 문턱은 CL54(p ≤ 3이고 n ≥ 1024, p ≤ 5이고 n ≥ 2048, p = 6이고 n ≥ 4096, p ≤ 8이고 n ≥ 16384면 Borůvka). 트리의 변을 정렬해 합치고, 같은 거리의 변이 둘 이상인 묶음은 CL50대로 정의를 푼다. 높이는 고른 변의 거리 그 자체라 세 길이 같은 트리와 같은 비트를 낸다.
- **complete·average·weighted·ward** (축소 가능한 연결법): NN-chain(Murtagh 1983; scipy의 `nn_chain`)이다. 행렬 길은 Lance–Williams로 행렬을 갱신하고(ward·centroid·median은 제곱 거리의 눈금), 행렬 없는 길(관측값의 ward)은 군집의 무게 중심과 크기를 쥐고 값을 그때그때 계산한다(무게 중심을 열 단위로 두어 NEON 한 칸에 두 군집씩; 한 칸이 spatial의 제곱합과 같은 연산이라 스칼라와 같은 비트다). 이웃 고르기는 (값, 열쇠)이고, 이웃 찾기 한 번에서 가장 작은 값과 정확히 같은 다른 후보(사슬의 앞 군집 포함)를 만나면 그 입력을 일반 알고리즘으로 다시 푼다(CL46; 충분함의 증명은 §4.1).
- **centroid·median과 표시된 입력** (일반 알고리즘): Müllner의 generic(우선순위 큐의 (값, 칸)과 칸마다의 가장 가까운 뒤 칸; 거리가 줄 수 있어 합친 뒤 앞 칸들의 후보를 다시 본다)이다. 행렬 없는 길은 무게 중심을 칸 순서로 두어 앞·뒤 범위를 이진 찾기로 정한다. 역전이 생긴다(§2.2).
- **길** (CL56): 관측값과 유클리드의 single·ward·centroid·median은 언제나 행렬 없는 길, 나머지는 열쇠 순서의 점들의 `sd.spatial.pdist`(같은 비트) 위의 행렬 길이다.
- **높이** (CL45, CL55): Lance–Williams 갱신은 병합 순서에 따라 반올림이 다르다(NN-chain은 높이 순서가 아닌 순서로 합친다). 그래서 트리가 정해진 뒤 높이를 최종 트리 위의 Lance–Williams 재귀로 다시 계산한다: 병합 (A, B)에서 열쇠가 작은 자식 A를 바깥으로 두고, A의 점 x마다 B의 부분 트리를 후위 순서로 훑어 (x, B)의 값을 구하고(점끼리는 spatial의 유클리드 거리, ward·centroid·median은 그 제곱), 그 값들을 A의 부분 트리 위로 올린다. 한 단계의 식은 두 자식에 대칭인 꼴이라(덧셈과 곱셈은 바꿔도 같은 비트) 자식의 차례는 상관없다. 모든 점 짝이 공통 조상에서 한 번씩 쓰여 O(n²p) 시간(행렬 만들기와 같은 크기), O(n) 메모리라 행렬 없는 길에도 맞는다. 바깥 점 넷을 한 번에 훑고(같은 부분 트리 구조, 칸마다 독립인 연산), 큰 병합은 바깥 점 덩이로 나란히 한다. single·complete는 고른 거리 그 자체다(재귀의 min·max가 정확하다). 그래서 높이가 고른 알고리즘(NN-chain, 일반, 행렬 없는 길)과 무관한 비트다. ward·centroid·median의 관측값은 CL43의 2의 거듭제곱으로 옮겨 계산하고 높이를 되돌린다. 단조인 연결법은 높이를 트리 위로 누적 최댓값으로 올린다(CL51).
- **행** (CL51): 두 자식이 이미 있는 병합 가운데 (높이, 작은 열쇠, 큰 열쇠)가 가장 작은 것부터 행을 쓴다. 두 군집 번호는 (작은 것, 큰 것)이다.
- **행렬 없는 길** (§9의 15번): 압축 꼴 거리 행렬은 8·n(n − 1)/2 바이트다(n = 10⁵에서 40 GB). 관측값과 유클리드의 single(Prim·Borůvka)과 ward·centroid·median(무게 중심 길)은 O(np) 메모리로 푼다(n = 10⁵의 ward가 10스레드 7초, single 0.17초; §5). complete·average·weighted는 행렬이 필요하고(작업 사본까지 16·n(n − 1)/2 바이트), 물리 메모리보다 크면 할당 전에 `MemoryError`다(API.md §2). scipy는 언제나 행렬을 만든다.
- **`optimal_leaf_ordering`** (CL53): Bar-Joseph 외(2001)의 O(n³) 동적 계획(M[u][w] = 병합의 잎을 u에서 w로 늘어놓는 최소 비용; 한 병합에서 먼저 바깥 쪽 끝 m에 대해 최소를 구하고 다시 안쪽 끝 k에 대해 구해 O(|L|²|R| + |L||R|²))이고 거리는 입력 거리다. 뿌리의 첫 자식은 왼쪽에 둔다. 같은 비용의 선택은 열쇠가 작은 끝점이다. 메모리 8n² 바이트(scipy와 같다).
- **`inconsistent`, `cophenet`**: `inconsistent`는 깊이 d까지의 높이들의 평균과 평균을 뺀 두 번째 훑기의 표준편차(n − 1로 나눈다)다. `cophenet`은 병합마다 두 자식의 점 짝에 높이를 쓴다(큰 병합은 점 덩이로 나란히). 상관은 sd 배열의 평균을 뺀 합이다.
- **DIANA** (CL52): R `diana`의 단계 그대로(지름이 2 이상인 군집마다: 다른 점들과의 평균 거리가 가장 큰 점으로 갈라진 무리를 시작하고, 나머지에서 (나머지와의 평균 거리 − 갈라진 무리와의 평균 거리)가 0보다 큰 점 가운데 가장 큰 점을 하나씩 옮긴다; 거리 합은 옮길 때마다 고친다). 군집을 나누는 순서는 트리를 바꾸지 않아 위에서부터 나누고 아래에서부터 합친다. 동점은 열쇠다. 거리는 열쇠 순서의 점들의 `sd.spatial.pdist`(또는 주어진 압축 꼴)다. O(n²) 메모리, O(n³) 최악 시간이다.
- **자르기와 scipy의 도구**: `fcluster`의 다섯 기준은 scipy의 순회(왼쪽 부분 트리, 오른쪽 부분 트리, 그다음 마디의 점 자식)로 1부터 번호를 매기고, maxclust는 단조 트리에서 높이의 값들 가운데 군집이 t 이하인 가장 작은 문턱이다(scipy와 같다; 역전 트리는 행의 차례로 정확히 t개). `cut`·`cut_tree`는 행의 차례로 되돌리고 라벨은 처음 나오는 차례다(scipy `cut_tree`의 번호와 같다). `leaders`는 scipy의 순회 차례다. `dendrogram`은 scipy `_dendrogram_calculate_info`의 재귀를 명시적 스택으로 옮겼다(CL57의 색). 이 도구들은 O(n)(`cut_tree`의 전체는 O(n²))이라 Python이다.

### 3.3 k-평균
구현은 `csrc/cluster/kmeans.cpp`(공용 머리 `partition.hpp`), Python 층은 `src/stardust/cluster/_kmeans.py`다(CL2 마일스톤).
- **Lloyd 반복의 한 단계**: (1) 점마다 가장 가까운 중심(계산한 제곱 거리의 최솟값, 동점은 작은 중심 번호; 거리는 CL59), (2) 중심마다 무게 평균. 평균은 점을 열쇠의 순서(CL7)로 4096점 덩이로 나눠 덩이마다 군집별 부분합을 내고 덩이 번호 순서의 짝 합으로 합친다(CL48). 그래서 스레드 수와도 입력 순서와도 무관한 비트다(§3.10).
- **경계 가속** (§9의 20번): Hamerly(2010)·Elkan(2003)의 삼각 부등식 경계는 "그 중심이 지금 가장 가까운 중심보다 가까울 수 없다"를 증명해 거리 계산을 건너뛴다. 경계에 반올림 여유(CL42: γ = 2(p + 4)u, 건너뛰기는 L > U(1 + 2γ)(1 + 4u))를 두어, 건너뛴 중심이 계산한 거리의 최솟값이 될 수 없음이 증명될 때만 건너뛴다. 그래서 가속한 길과 Lloyd가 같은 라벨과 같은 비트의 중심을 낸다(sklearn의 `'elkan'`은 `'lloyd'`와 끝자리가 다르다). 길은 모양(n, p, k)으로만 고르고 `algorithm` 인자가 없다. 경계는 Hamerly의 둘(제 중심의 상한 U, 다른 중심들의 하한 L)과 중심 사이 거리의 반 s(a)이고, 건너뛰지 못한 점은 제 중심까지 다시 잰 U로 한 번 더 보고 그래도 안 되면 모든 중심을 훑는다. Elkan은 넣지 않았다(CL60).
- **훑기** (CL49, CL60): 차의 꼴은 spatial의 타일 커널(점 넷 × 중심 여덟의 판)이고, 가장 작은 값과 둘째 값을 가지 없이 쥔다(둘 다 2⁻⁹⁶⁸ 이상이면 타일의 합이 곧 sqd; 아니면 그 점을 sqd로 다시 훑는다). 거르기는 작업(4096점)마다 Accelerate `dgemm`으로 x·c를 내고, NEON 두 칸씩 전개식 값 v와 한계 E로 A = v − E와 v + E의 최솟값을 구한 뒤 A가 한계를 넘는 중심을 빼고(가지 없이 남은 중심의 번호를 적는다) 남은 짝을 넷씩 나란히(`sqd4`: 짝마다 sqd와 같은 차례) 잰다. 뺀 중심의 A의 최솟값이 경계의 L을 준다.
- **멈춤**: sklearn과 같다: 라벨이 바뀌지 않으면(엄격한 수렴) 또는 중심 이동의 제곱합이 `tol`·(축마다 분산의 평균) 이하이면 멈춘다. 멈춘 뒤 마지막 중심으로 라벨을 다시 매긴다(sklearn과 같다; 라벨과 중심이 늘 맞는다).
- **빈 군집**: sklearn의 `_relocate_empty_clusters_dense`(`_k_means_common.pyx`)와 같은 뜻이다: 빈 군집의 수만큼 현재 중심까지 가장 먼 점(동점은 열쇠)을 골라 그 점을 새 중심으로 하고 원래 군집의 합에서 뺀다.
- **`n_init`**: 시작마다 Philox의 다른 하위 흐름(§3.10)을 쓰므로 시작들을 나란히 돌릴 수 있다. 가장 좋은 시작은 (관성, 시작 번호)가 가장 작은 것이다.
- **미니배치** (§9의 22번): sklearn `MiniBatchKMeans`의 갱신(중심마다 누적 개수의 학습률, `reassignment_ratio`의 다시 놓기, `max_no_improvement`의 멈춤)이다. 배치의 표본은 Philox의 (단계, 자리) 원소로 뽑는다(CL63). 배치 안의 갱신은 차례다: 중심마다 (옛 중심 × 누적 개수 + 배치의 점들을 배치 차례로) / 새 누적 개수. 멈춘 뒤 모든 점을 마지막 중심으로 배정하고 관성을 잰다(CL64).
- **이분 k-평균**: sklearn `BisectingKMeans`(`strategy='biggest_inertia'` 또는 `'largest_cluster'`)다. 나눌 잎의 점들을 모은 사본에서 k = 2의 `kmeans`(같은 커널)를 돌리고, 허용 오차는 전체 자료의 것이다. 나눈 차례를 Z 꼴의 트리로 결과에 둔다(CL64).
- **시작**: k-means++는 점마다의 w·D²의 누적 합(CL48 순서)에서 균등 수로 후보를 찾고, 후보들의 잠재값을 덩이 합으로 잰다. `'random'`은 Efraimidis–Spirakis 열쇠(CL63). `tol`의 분산은 열마다의 평균을 뺀 두 번째 훑기(덩이 합)이고 무게를 쓰지 않는다(sklearn `_tolerance`와 같다).

### 3.4 k-medoids
- **BUILD**: R `pam`과 같다: 첫 medoid는 거리 합이 가장 작은 점, 다음은 목적을 가장 줄이는 점(동점은 열쇠).
- **SWAP**: (medoid, 비medoid)의 모든 교환 가운데 목적을 가장 줄이는 것을 하고 줄지 않으면 멈춘다(R의 `variant='original'`). FastPAM1(Schubert–Rousseeuw 2019)의 계산 재사용은 같은 교환을 O(k) 배 빨리 찾는다: 같은 교환을 고르도록 동점 규칙을 맞추면 같은 결과다(CL2 마일스톤에서 확인). FasterPAM(첫 개선 교환)은 다른 결과라 나중이다(§9의 23번).
- 거리는 관측값이면 그때그때 계산(O(n) 메모리; 교환 한 번의 일이 O(n·k) 거리), 미리 계산한 거리면 그 행렬이다.
- **CLARA**: 표본마다 PAM, 전체 점의 목적으로 가장 좋은 표본을 고른다(같으면 앞 표본). 첫 표본 다음부터 지금까지의 최선 medoid를 표본에 넣는다(R과 같다). 나머지는 Philox의 (표본 번호, 점의 열쇠) 원소로 뽑는다(R의 표본과 같지 않다). R의 `correct.d=TRUE`(NA 처리의 고침)는 NaN이 없으니 뜻이 없다.
- **구현** (CL2 마일스톤, `csrc/cluster/kmedoids.cpp`, `_kmedoids.py`; CL65): BUILD의 합과 SWAP의 변화는 후보 점마다 한 작업이 열쇠 순서로 차례 합을 내고(점 덩이로 나란히), 가장 좋은 교환은 작업들의 (변화, 열쇠)를 합친다. 가장 가까운·둘째 medoid는 교환마다 다시 잰다(O(nk)). 목적값의 합은 CL48의 덩이 합이다.

### 3.5 퍼지 c-평균
- Bezdek의 교대 갱신(소속도 uᵢⱼ = 1/Σ_l(dᵢⱼ/dᵢₗ)^{2/(m−1)}, 중심 = uᵐ 무게 평균)이다. 점이 중심과 정확히 겹치면(dᵢⱼ = 0) 그 중심에 소속도 1(여럿이면 똑같이 나눈다)이다. 멈춤은 소속도 변화의 최대 노름 < `tol`, 상한 `max_iter`다(기본 `tol` = 10⁻⁵, CL66).
- **구현** (`csrc/cluster/fuzzy.cpp`, `_fuzzy.py`): 시작의 중심에서 소속도, 그다음 반복마다 중심(CL48의 덩이 합으로 Σuᵐx와 Σuᵐ), 소속도와 그 변화의 최댓값. 거리는 k-평균과 같은 sqd(타일 커널; CL43의 척도)이고, 소속도는 가장 가까운 중심과의 비 r = min D/D로 계산해 넘침이 없다(m = 2면 거듭제곱 없이 r).

### 3.6 밀도
구현은 `csrc/cluster/density.cpp`(DBSCAN, 핵심 거리, HDBSCAN의 트리·줄인 트리·고르기, OPTICS와 그 추출)와 `mst.cpp`(상호 도달 거리의 Prim·Borůvka), Python 층은 `src/stardust/cluster/_density.py`다(CL3 마일스톤). 점은 열쇠 차례로 넘기고(미리 계산한 거리는 입력 색인) 모든 동점은 작은 열쇠다.
- **거리와 길** (CL72): 관측값의 euclidean·cityblock·chebyshev는 spatial의 짝 커널(`KdEuclid`·`KdCity`·`KdCheb`의 `raw`와 `finish`; `cdist`와 같은 비트)을 KD 트리 찾기나 모든 짝의 훑기로 부른다. 트리 찾기는 spatial의 증명된 가지치기(`KdCut`, `kd_beyond`)로 반경 밖이 증명된 마디만 자르므로 훑기와 같은 짝을 같은 커널로 잰다. 그 밖의 거리는 `sd.spatial.pdist`의 압축 꼴 행렬이다.
- **DBSCAN** (CL28, CL73): (1) 점마다 eps 안(d ≤ eps, 자기 포함)의 점 수(무게가 있으면 이웃의 열쇠를 정렬해 무게를 그 차례로 더한 값)를 `min_samples`와 견줘 핵심 점을 정한다. KD 트리는 먼 모서리 쌍까지의 거리(spatial의 `KdPair::far`)가 eps 안으로 증명된 마디를 통째로 센다(무게 없을 때). (2) 핵심 점마다 eps 안의 더 큰 열쇠의 핵심 점과 `graph/unionfind.hpp`의 `unite_shared`로 잇는다(점 덩이로 나란히). (3) 핵심이 아닌 점은 eps 안의 핵심 점 가운데 (d, 열쇠)가 가장 작은 점의 군집이고, 없으면 잡음이다. 이웃 목록을 쥐지 않는다.
- **핵심 거리**: `min_samples`번째로 가까운 점까지의 거리(자기 포함)다. KD 트리는 spatial의 `KdSearch`(k 최근접, (거리, 색인)의 차례), 훑기는 모든 거리의 `nth_element`, 행렬은 그 행이다. k번째 값은 고르는 방법과 무관하다.
- **HDBSCAN** (CL67–CL69): 핵심 거리 → 상호 도달 거리의 최소 신장 트리(핵심 거리의 KD 트리 위의 Borůvka, 그 밖은 Prim; 변의 차례 (w, 작은 열쇠, 큰 열쇠)) → 그 차례의 Kruskal로 단일 연결 트리 → 같은 높이의 병합을 한꺼번에 거두는 줄인 트리 → 안정성(행의 차례로 Σ(λ − λ_birth)·크기) → sklearn `_tree.pyx`의 고르기·라벨·확률 → GLOSH와 지속성이다. 안정성의 비교(`'eom'`)에서 자식의 합과 부모가 정확히 같으면 sklearn처럼 부모를 고른다(sklearn 소스의 `subtree_stability > stability[node]`). 줄인 트리부터는 O(n) 차례다.
- **OPTICS** (CL70, CL71): 처리하지 않은 점의 목록(자리 배열로 지운다)을 단계마다 한 번 훑어 도달 거리를 낮추고(현재 점이 핵심일 때; 도달 거리가 이미 핵심 거리 이하인 점은 거리를 재지 않고, d > min(도달 거리, max_eps)가 증명된 짝은 제곱근을 구하지 않는다) 다음 점((도달 거리, 열쇠)의 최소)을 찾는다. 훑기는 점 덩이로 나란하고 덩이의 최소를 덩이 차례로 합친다. O(n²p) 시간, O(n) 메모리다(sklearn도 O(n²)). ξ 추출은 sklearn의 `cluster_optics_xi`를 같은 연산으로 옮겼다.

### 3.7 혼합 모형의 EM
- 성분 j의 로그 밀도는 정밀도 행렬의 Cholesky 인자 Lⱼ(sklearn의 `precisions_cholesky_`)로 −½‖Lⱼᵀ(x − μⱼ)‖² + log det Lⱼ − (p/2)log 2π이다. 책임도는 log 공간에서 최댓값을 빼는 합(`logsumexp`의 꼴)이다.
- M 단계: 무게, 평균, 공분산(+ `reg_covar`·I)은 크기로만 정한 덩이의 부분합을 짝 합으로 합친다(k-평균과 같은 장치). 공분산은 평균을 뺀 두 번째 훑기로 구한다(sklearn과 같은 꼴; 한 번 훑기의 E[xxᵀ] − μμᵀ는 상쇄가 있다).
- 공분산의 Cholesky는 p×p의 고정 순서 커널이다. 실패하면 `LinAlgError`(sklearn의 메시지는 "Fitting the mixture model failed because some components have ill-defined empirical covariance"; sd는 `gaussian_mixture(): component 2 has an ill-defined covariance; increase 'reg_covar' or check for duplicate points`의 꼴).
- 시작(§9의 30번): sklearn의 `init_params='kmeans'`(k-평균 한 번의 라벨로 책임도를 만든다), `'k-means++'`, `'random'`, `'random_from_data'`. k-평균은 같은 `rng`의 하위 흐름이다(CL76).
- **구현** (CL4 마일스톤, `csrc/cluster/mixture.cpp`, `_mixture.py`; CL75–CL81): 점은 열쇠 차례로 CL77의 격자에 옮겨 넘긴다. E 단계는 512점 작업마다 여덟 점을 한 덩이로 행을 열로 놓고(차례대로) 성분마다 z = x − μ를 NEON 두 칸씩, full·tied의 y = Uᵀz는 네 열 × 여덟 점의 판(누적기 16개, 열 끝을 0으로 채운 U)으로, q = Σy²를 판의 차례로 잰다(판의 0은 y에 정확한 0을 더하고 q에서 빠진다; 한 칸의 연산은 그 점만의 것이라 덩이와 무관한 비트). exp는 배열의 NEON exp(`array/kernels/vmath.hpp`, 1 ULP)다. M 단계의 첫 훑기(Σlog p, nk, Σrx)는 덩이마다, 두 번째 훑기(2차 모멘트)는 (덩이, 성분)마다의 작업이고 full·tied는 항(점, 또는 tied의 (점, 성분))을 128개씩 모아 네 행 × 여덟 열의 판을 누적기에 쥔 채 항의 차례로 더한다(원소마다의 합의 차례가 항의 차례 그대로다; r = 0인 항은 정확한 +0이라 건너뛴다). 덩이의 합은 덩이 차례의 짝 나무로 만난다(`pairwise_chunks`와 같은 나무). 작은 Cholesky와 삼각 역은 고정 순서의 fma다. `n_init`의 시작은 차례이고 시작 안이 나란하다. `'kmeans'`의 라벨과 `'k-means++'`의 점은 Python이 `_core.cluster.kmeans`·`kmeans_plusplus`로 만들어 넘긴다.

### 3.8 지표
- **분할표**: 두 라벨 배열의 정수 셈(라벨을 정렬한 고유값의 번호로 바꾸고, 칸이 적으면 밀집 셈, 많으면 열쇠의 정렬; CL83). 셈은 int64이고 n(n − 1) < 2⁶³까지 받는다(짝 혼동 행렬의 항; 넘으면 `ValueError`, CL82).
- **ARI, RI, FM, 짝 혼동 행렬**: 정수 셈의 유리식이다. 짝 수 Σ C(nᵢⱼ, 2), Σ C(aᵢ, 2), Σ C(bⱼ, 2)를 C++에서 128비트로 세어 Python 정수로 넘기고, ARI = 2(TP·TN − FN·FP)/((TP + FN)(FN + TN) + (TP + FP)(FP + TN))의 분자·분모를 Python 정수로 정확히 구해 마지막 나눗셈만 반올림한다: **올바르게 반올림한 값**이다(§4.6). FM = TP/√((TP + FP)(TP + FN))도 정수 제곱근과 끈적한 비트로 올바르게 반올림한다(CL83).
- **MI, NMI, AMI, 동질성·완전성·V**: Σ(nᵢⱼ/n)log(n·nᵢⱼ/(aᵢbⱼ))와 엔트로피는 칸 차례의 짝 합이고, 로그는 1 근처에서 log1p다(CL84). AMI의 기대 상호정보(Vinh 외 2010)는 초기하 분포의 합이고, 무게를 비의 점화식으로 최빈값에서 밖으로 구해 그 합으로 나눈다(lgamma 차의 상쇄가 없다; 꼬리가 합의 2⁻⁸⁰ 아래이면 멈춘다). 같은 (행 크기, 열 크기)의 짝은 한 번만 재고(서로 다른 크기는 한쪽에 √(2n)개 아래), 짝들은 나란히 재어 크기 차례의 짝 합으로 합친다.
- **실루엣**: 점마다 군집별 평균 거리. O(n²p) 시간, 행렬 없이 spatial의 타일 커널(4점 × 8행)로 32점 덩이마다 나란히(점 하나의 군집 합은 군집 안 열쇠 순서의 네 사슬; CL85). CH와 DB는 첫 점을 뺀 자료의 무게 중심과 덩이 합이다(CL86).

### 3.9 gap 통계량
- R `clusGap`의 단계: k = 1 … K마다 log W_k(W = ½Σ_r D_r/n_r, D_r은 군집 r 안 짝의 `d_power`제곱 거리 합; R의 `W.k`), 참조 자료 B벌(`'scaledpca'`: 주성분 축에서 상자 안의 고른 난수; `'original'`: 원래 축의 상자)마다 같은 군집을 다시 하고, gap = E*[log W] − log W, SE = sd·√(1 + 1/B)다. 군집은 `method`(`'kmeans'`, `'pam'`; §9의 34번)로 이 영역의 C++ 함수(`partition.hpp`의 `kmeans_labels`, `pam_labels`: 공개 함수와 같은 비트)를 부른다. 참조 자료와 군집의 난수는 `rng`의 하위 흐름이고 B벌은 C++ 안에서 나란하다(CL88). k는 R `maxSE`의 규칙으로 Python에서 고른다.

### 3.10 병렬화와 결정성
**같은 입력이면 스레드 수와 상관없이 같은 비트다**(§9의 6번 (a)). 테스트는 함수마다 1·3·10스레드(또는 1, 2, 3, 7, 10)에서 같은 바이트를 확인하고, 골든은 1·3·10스레드에서 같아야 한다.

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 거리 행렬(압축 꼴) | 행 덩이(크기로만) | 원소 하나는 spatial의 한 커널이다 |
| Prim(단일 연결, HDBSCAN) | 단계마다 후보 갱신을 점 덩이로, 최솟값은 (거리, 열쇠)의 전순서 | 최솟값이 하나로 정해진다 |
| Borůvka(KD 트리) | 질의 잎의 덩이(크기로만)마다 점의 찾기; 성분의 지금까지의 최선은 원자적으로 낮춘다 | 점마다 가장 가까운 다른 성분의 점은 (거리, 열쇠)의 전순서라 하나다. 성분의 최선은 자르기에만 쓰고 자르기는 엄격해서 같은 값의 변을 놓치지 않는다 |
| NN-chain·일반 알고리즘 | 차례(사슬은 본디 차례다); 무게 중심 길의 찾기는 군집 덩이로 나란히 | 덩이의 (값, 열쇠) 최소와 동점 수를 덩이 순서로 합친다 |
| 높이 다시 계산 | 병합마다 바깥 점 덩이로 나란히(점 넷씩 한 번에) | 점 하나의 값은 그 점의 정해진 연산 차례다 |
| k-평균의 라벨 | 점 덩이 | 점 하나는 한 커널이다 |
| k-평균의 중심 합 | 점 덩이(크기로만)의 부분합, 덩이 순서의 짝 합 | 합의 순서가 n, k, p로만 정해진다 |
| `n_init`의 시작들 | 시작마다 | 하위 흐름이 시작 번호로 정해지고, 고르기는 (목적, 번호) |
| PAM의 교환 비용 | medoid × 점 덩이 | 가장 좋은 교환은 (비용, 열쇠)의 전순서 |
| DBSCAN의 이웃과 합집합–찾기 | 점 덩이(64점), 병렬 합집합–찾기(`graph/unionfind.hpp`) | 핵심 판정은 정수 셈이나 열쇠 차례의 합, 연결 요소는 분할의 함수, 경계 점은 (거리, 열쇠)의 전순서 |
| 핵심 거리 | 점 덩이 | 점 하나의 k번째 거리는 고르는 방법과 무관하다 |
| Borůvka(HDBSCAN) | 단일 연결의 Borůvka와 같다; 상호 도달 거리의 가지치기도 엄격하다 | 같은 무게의 변을 놓치지 않는다(§4.4) |
| OPTICS의 단계 | 처리하지 않은 점의 덩이마다 갱신과 최소, 덩이 차례로 합치기 | 점마다의 갱신은 독립이고 최소는 (도달 거리, 열쇠)의 전순서 |
| EM의 E·M 단계 | E 단계는 점 작업(512점), M 단계는 (4096점 덩이, 성분)의 부분합, 덩이 차례의 짝 합 | 점마다의 값은 그 점의 연산 차례이고 합의 나무는 n으로만 정해진다(CL75) |
| 혼합 모형의 표본 | 1024표본 작업 | 표본 t는 원소 t의 함수다(CL80) |
| 실루엣·외부 지표의 셈 | 행 덩이, 정수 셈 | 정수 셈은 순서와 무관하고 실수 합은 덩이 순서다 |
| gap의 참조 자료 B벌 | 벌마다(자료 포함 B + 1; 벌 안의 군집은 차례) | 하위 흐름이 벌 번호로 정해진다(CL88) |
| 기대 MI | (행 크기, 열 크기)의 짝마다 | 짝의 값은 그 짝의 연산 차례, 합은 크기 차례의 짝 합(CL84) |
| Affinity Propagation | 16행 덩이의 A·R 갱신, 256열 덩이의 열 합 | 값마다 한 점 또는 한 열의 고정된 연산 차례(CL94) |

- **난수의 하위 흐름** (CL41): 호출마다 `rng`에서 64비트 두 단어를 받아 키로 쓴다(회귀 `rlm(method='MM')`의 규칙). 시작(표본, 참조 자료) r의 t번째 수는 원소 r·2⁴⁰ + t의 Philox 출력이다(k-means++의 단계들은 t를 차례로 쓴다). 안에서 부르는 이 영역의 함수는 부모 흐름의 정해진 시작 번호에서 두 단어를 받아 자기 키로 쓴다. 그래서 `rng`는 호출 하나에 언제나 두 단어만 움직이고, 결과는 스레드 수와 무관하다.
- 병렬 기준값은 배열의 규칙이다(float64 연산 2¹⁵개 분량 이상이면 나누고 GIL을 놓는다; API.md §3).

### 3.11 메모리
- 계층: 행렬 길은 압축 꼴 8·n(n − 1)/2 바이트(n = 2×10⁴에서 1.6 GB, 10⁵에서 40 GB)와 그 작업 사본(scipy도 사본을 만든다)이고 둘의 합이 물리 메모리보다 크면 할당 전에 `MemoryError`다. 행렬 없는 길(§3.2)은 O(np)(무게 중심의 열 사본, KD 트리). 일반 알고리즘의 우선순위 큐와 높이 재귀는 O(n). `optimal_leaf_ordering`은 8n² 바이트다.
- k-평균: 자료(사본 없음, 정렬 순서는 색인 배열; §9의 4번), 중심 O(kp), 경계 가속 O(n) 또는 O(nk)(Elkan; nk가 2²⁶을 넘으면 Hamerly만), 덩이 부분합 O(덩이 수 × kp).
- k-medoids: 관측값 길은 O(n + k), 미리 계산한 거리는 그 행렬.
- DBSCAN: 이웃 목록 O(Σ 이웃 수)를 만들지 않고 핵심 판정(셈), 연결(합집합–찾기), 경계 점을 세 번 훑는다(sklearn은 이웃 목록 전체를 쥔다). 무게가 있으면 점 하나의 이웃 열쇠만 잠시 쥔다.
- HDBSCAN: KD 트리, 핵심 거리 O(n), MST·단일 연결 트리·줄인 트리 O(n). Prim은 행렬 없이 O(n). `medoids`는 군집 안 거리 합을 행렬 없이 잰다(관측값의 세 거리; 그 밖은 군집마다 `pdist`).
- OPTICS: 도달 거리·선행자·처리하지 않은 점의 목록 O(n). 행렬 없다(sklearn도 행렬은 없지만 점마다 반경 이웃 목록을 만든다).
- 혼합 모형(CL4 마일스톤): 자료의 열쇠 차례 사본(격자로 옮긴 것) O(np), 책임도 O(nk), log p(x) O(n), 2차 모멘트의 덩이 잎 O(⌈n/4096⌉·k·p²)(sklearn은 성분마다 (n, p)의 차 행렬을 만든다). 결과는 모수와 맞춤의 격자 값만 쥔다.
- 실루엣: O(n·k)(군집별 합)이고 n × n 행렬을 만들지 않는다(sklearn은 행 덩이의 거리 행렬을 만든다).
- 결과 객체는 `predict`를 위해 중심·medoid·모수만 쥐고 자료의 사본을 쥐지 않는다(`HDBSCANResult.medoids` 같은 게으른 필드는 결과가 자료의 참조를 쥔다; §9의 1번에서 정한다).
- 지표(CL5 마일스톤): 분할표는 칸이 max(4n, 2²⁰) 이하면 밀집 int64, 넘으면 n개의 열쇠를 정렬한다. 기대 MI는 짝마다 잘린 무게 하나(최빈값 둘레). 실루엣은 군집으로 묶은 자료의 사본과 판 O(np). gap은 벌마다 자료 크기의 사본 O(np)(스레드마다 하나)와 군집 안 행 사본. Gower는 결과(압축 꼴)와 열마다의 수 행렬 O(np).
- Affinity Propagation(CL6 마일스톤): S, R, A의 3·8n² 바이트와 중앙값의 사본 8n²(CL94; sklearn은 tmp까지 4·8n²).

### 3.12 Affinity Propagation (CL6 마일스톤)
- 반복 하나: 행마다(16행 덩이로 나란히) 앞 반복의 열 합으로 A 행을 고치고(대각은 자르지 않는다), 행 버퍼에 A + S를 써서 최댓값·처음 자리·나머지의 최댓값을 NEON으로 찾고 R 행을 고친다. 그다음 열 합을 256열 덩이마다 행 차례로 더한다(numpy의 axis=0 줄이기와 같은 차례). exemplar 판정은 대각의 새 A를 열 합에서 O(n)으로 먼저 잰다(멈추면 A의 나머지는 고치지 않는다). 연산은 sklearn의 것과 하나씩 같다(CL92).
- n = 2000, p = 2(200번)는 1스레드 1.47 s, 10스레드 0.22 s이고 sklearn은 2.4 s다(§5). 1스레드는 메모리 대역에 묶인다(반복마다 약 190 MB).

---

## 4. 정확도 계약
군집의 결과는 이산적이다. 그래서 계약은 두 겹이다. (1) **이산 결과의 계약**: 라벨·트리·medoid가 "주어진 실수 값(계산한 거리, 계산한 목적)과 정한 동점 규칙으로 정의대로 고른 것"이다. 정확한 값으로 고른 것과는 가까운 동점(두 후보의 차가 계산의 오차 한계 안)에서만 다를 수 있다. 참값 도구는 정확한 산술로 고른 답과 견주고, 다를 때는 그 자리가 가까운 동점인지(정확한 두 값의 차가 한계 안인지)를 확인한다. (2) **실수 결과의 계약**: 높이, 중심, 목적값, 모수, 지표가 정확한 값과 견준 한계 안이다(참값: 정확한 유리수와 Arb; 상수는 CL0 마일스톤의 시제품 측정으로 정한다, smooth §4.7과 같은 절차).

### 4.1 계층 군집
- **트리**: 계산한 값(거리, Lance–Williams의 값, 무게 중심의 값)과 열쇠로 정의(§3.2)대로 고른 트리다. 정확한 산술로 고른 트리와는 비교가 가까운 동점(두 후보의 차가 그 값의 반올림 안)일 때만 다르다. single·complete는 비교하는 값이 입력 거리 그 자체라 트리가 계산한 거리의 정확한 함수다. 참값 집합 37개 × 일곱 연결법에서 행(두 군집, 크기, 차례)이 모두 참값과 같았다(격자·같은 점의 정확한 동점 포함; `tests/cluster/test_hierarchy.py`).
- **정리 1 (단일 연결의 동점, CL50)**. 계산한 거리 d̂로, 변을 (d̂, 작은 열쇠, 큰 열쇠)로 정렬한 Kruskal의 최소 신장 트리를 T라 하자. 거리 w 바로 앞의 군집들(d̂ < w인 변의 연결 요소)을 C, C 위의 그래프 G_w를 "두 군집 사이에 d̂ = w인 점 짝이 있다"로 두면 (i) G_w의 연결 요소는 T의 무게 w인 변들이 잇는 군집 묶음과 같고, (ii) 정의의 트리는 각 요소에서 열쇠가 가장 작은 군집 m이 G_w의 이웃을 열쇠 순서로 흡수하는 트리다. 증명: C의 서로 다른 두 군집 사이 거리는 w 이상이다(w보다 작으면 이미 합쳐졌다). (i) G_w의 변 (A, B)는 T 위에서 A와 B를 잇는 경로를 닫는 고리를 만들고, 고리 성질로 그 경로의 가장 큰 변은 w 이하이며 A ≠ B라 w인 변이 경로에 있다. 그래서 G_w의 요소는 T의 w인 변의 요소 안에 있고, 그 역은 T의 변 자체가 G_w의 변이라 당연하다. (ii) 높이 w에서 정의는 열쇠가 가장 작은 인접 쌍을 고른다. 쌍의 열쇠 (min, max)의 첫째가 가장 작은 것은 요소의 최소 열쇠 m을 품은 쌍이고, 합친 군집은 열쇠 m과 두 군집 이웃의 합집합을 가지며 다른 쌍의 값은 그대로다. 그래서 m이 이웃을 남기는 동안 m의 쌍이 계속 먼저이고, 그 쌍의 둘째는 지금 이웃 가운데 가장 작은 열쇠다. ∎ 같은 거리의 군집 짝 사이에 T의 변이 없을 수 있어(정삼각형의 세 점: T는 두 변만 쥔다) (ii)는 T의 변만으로 정해지지 않는다. 동점이 없으면 Kruskal의 차례가 곧 정의의 트리다.
- **정리 2 (NN-chain의 동점 검출이 충분하다, CL46)**. 연결법이 축소 가능하다고 하자(complete, average, weighted, ward: d(i, j) ≤ min(d(i, k), d(j, k))이면 d(i ∪ j, k) ≥ min(d(i, k), d(j, k)); Lance–Williams 계수로 확인된다). 비교가 그 값들의 정확한 비교라 할 때, NN-chain의 어떤 이웃 찾기도 최솟값과 정확히 같은 다른 후보(사슬의 앞 군집 포함)를 만나지 않으면 NN-chain의 병합 집합은 정의(어떤 열쇠로든)의 병합 집합과 같다. 증명: (1) NN-chain이 합치는 (x, y)는 그때 **엄격한** 서로 가장 가까운 짝이다: x의 찾기가 y를 다른 모든 후보보다 엄격히 작게 찾았고(동점 없음), y의 찾기도 x를 그렇게 찾았다. x의 찾기 뒤의 병합 (c, d)는 엄격한 짝이라 d(c, d) ≤ min(d(c, x), d(d, x))이고 축소 가능성으로 d(x, c ∪ d) ≥ min(d(x, c), d(x, d)) > d(x, y)라 엄격함이 이어진다. (2) 분할 P에서 (a, b)가 값 δ의 엄격한 짝이면, P에서 출발한 정의는 a나 b를 품은 다른 쌍을 (a, b)보다 먼저 고르지 않고(그 값은 δ보다 크고, 축소 가능한 연결법의 정의는 고르는 값이 줄지 않아 (a, b)를 고를 때까지 δ 이하만 고른다), (a, b)를 먼저 합친 P′에서 출발해도 a ∪ b의 쌍의 값이 δ보다 커서(축소 가능성) 같은 쌍을 같은 차례로 고르다가 두 상태가 같아진다. 그래서 병합 집합이 같다. (3) NN-chain의 병합마다 (2)를 쓰면 귀납으로 집합이 같다. ∎ 동점이 찾기 밖에 있는 것(서로 먼 두 쌍이 같은 높이)은 트리를 바꾸지 않고, 행의 차례는 CL51이 정한다. 표시된 입력은 일반 알고리즘이 (값, 열쇠)로 정의대로 푼다. 시제품의 128가지 가운데 다른 4가지가 모두 표시됐던 것(§5)과 맞는다.
- **정리 3 (높이, CL45·CL55)**. 모형: IEEE double, 가장 가까운 값으로 반올림, FMA 축약 없음(spatial의 제곱합의 fma만), 연산 하나의 상대 오차 |δ| ≤ u, γ_k = ku/(1 − ku). 재귀의 한 단계는 v = α₁v₁ + α₂v₂ − βh 꼴이고(average: β = 0; weighted: α = ½, β = 0; centroid: αᵢ = nᵢ/(n₁ + n₂), β = n₁n₂/(n₁ + n₂)²; median: ½, ½, ¼; ward: αᵢ = (nᵢ + n_Q)/T, β = n_Q/T), 항 하나가 겪는 반올림은 c번 이하다(average 3, weighted 1, centroid·ward 5, median 2: 계수의 계산, 곱, 더하기와 빼기). 밑값은 점 짝의 제곱합(차 하나에 1, fma 사슬 p단계: 상대 γ_{p+2}; CL43의 척도와 조심 길은 정확하다)이거나 그 제곱근(γ_{p+2} 안)이다. 재귀의 값 w마다 정확한 값 w*(같은 재귀를 정확한 밑값과 정확한 높이로), 양의 값 w⁺(각 −βh를 +βh⁺로 바꾼 같은 재귀), 사슬의 길이 λ(w)(밑값 0, 단계마다 1 + 세 입력의 λ 최댓값; 병합의 높이에서는 μ)를 두면 |ŵ − w*| ≤ γ_{cλ(w)+p+2}·w⁺다. 증명: 단계의 계산값은 α₁v̂₁(1 + θ₁) + α₂v̂₂(1 + θ₂) − βĥ(1 + θ₃), |θᵢ| ≤ γ_c(Higham 2002, 보조정리 3.1·3.3)이고, v̂ᵢ = vᵢ* + εᵢ, |εᵢ| ≤ γ_{cλᵢ+p+2}vᵢ⁺(귀납 가정), |vᵢ*| ≤ vᵢ⁺이므로 |ŵ − w*| ≤ (α₁v₁⁺ + α₂v₂⁺ + βh⁺)(γ_{c·maxλᵢ+p+2} + γ_c + γ_cγ_{…}) ≤ γ_{c(maxλᵢ+1)+p+2}·w⁺(γ_a + γ_b + γ_aγ_b ≤ γ_{a+b}). ∎ 출력 높이: ward·centroid·median은 ĥ = RN(√max(Ĥ, 0))·2^{−e}라 \|ĥ² − H*\| ≤ γ_{cμ+p+5}V⁺, 그래서 \|ĥ − h*\| ≤ γ_{cμ+p+5}V⁺/h* + (아래넘침의 절대항)다. 단조 연결법의 누적 최댓값(CL51)으로 올라간 높이는 그 자식의 한계를 지킨다(h* ≥ 자식의 h*). 이 한계가 `tests/cluster/refdata.py`의 `height_bound`이고 μ와 V⁺를 그 트리에서 계산한다.
  - **상쇄**: V⁺/H*는 average·weighted에서 1이다(뺄셈이 없다). ward·centroid·median의 맨 위 단계는 단조 트리에서 βh ≤ v라 2배 안이지만 안쪽 단계(병합이 아닌 군집 짝의 값)는 원리상 묶이지 않는다. 참값 집합에서 V⁺/H*는 ward 6.2(blobs/k5p5), centroid 5.3, median 5.4(lattice/3x3x3)까지, μ는 깊이의 3.7배(44)까지였다. 측정한 오차는 일곱 연결법 모두 참값과 2 ULP 이하(scipy의 누적 3 ULP)이고 한계의 0.21–0.28배(single·complete는 spatial 거리의 한계 γ_{p/2+2}h*의 0.67배)다(`bench/cluster/accuracy.md`).
- **DIANA**: 높이는 지름(입력 거리의 최댓값)이라 정확하다. 트리는 계산한 평균 거리의 비교로 고르며 정확한 비교와는 가까운 동점에서만 다르다.
- `cophenet`은 높이를 옮겨 쓴다(정확). `inconsistent`는 정의대로의 두 번 훑기(γ 꼴). `optimal_leaf_ordering`은 계산한 비용 합의 정확한 최소이고(같은 비용은 열쇠), 정확한 최소와는 비용 합이 가까운 동점일 때만 다르다.

### 4.2 k-평균
- **고정점**: `converged=True`이면 (1) 모든 점 i의 라벨은 결과의 중심까지 계산한 제곱 거리의 최솟값(동점은 작은 번호)이고 정확한 제곱 거리의 최솟값과는 두 중심의 정확한 거리 차가 c·u·(‖xᵢ − c‖² 꼴의 한계) 안일 때만 다르다, (2) 모든 중심은 그 라벨의 점들의 정확한 무게 평균과 ‖c − c*‖_∞ ≤ c·γ_{⌈log₂ n⌉+1}·max|x| 안이다(짝 합의 한계). `tol` > 0으로 멈춘 결과는 (2)만 약속하고 (1)은 마지막 라벨 다시 매기기의 것이다(§3.3).
- **관성**: 결과의 라벨과 중심으로 정확히 계산한 값과 상대 c·γ_{p + log₂ n} 안.
- 반복의 경로(몇 번째에 어디로 가는지)는 약속하지 않는다: 반복 r은 반복 r − 1의 계산값으로 정의한다(smooth의 강건 반복과 같다). 같은 입력이면 같은 비트라는 것만 약속한다.
- **거리의 오차** (CL59): sqd(x, c)는 |sqd − d²| ≤ γ_{p+3}·d² + 2⁻¹⁰⁷⁴이다(d = ‖x − c‖; 차 하나에 u, 음이 아닌 항의 fma 사슬 γ_p, 2⁻⁹⁶⁸ 아래의 조심 끝내기는 2의 거듭제곱의 정확한 척도와 마지막 `ldexp`의 아래넘침 하나).
- **정리 4 (경계의 여유, CL42)**. γ = 2(p + 4)u, κ = (1 + 2γ)(1 + 4u)를 올려 반올림한 수라 하자. U = up₂(fl(√fl(s + 2⁻¹⁰⁷³))·(1 + γ))(s = sqd, up₂는 다음 수로 두 번)는 참 거리 d 이상이고 L = down₂(fl(√max(s − 2⁻¹⁰⁷³, 0))·(1 − γ))는 d 이하다(√의 반올림 u, 곱 u, 위의 오차가 γ_{p+3} < γ 안이라 두 칸이 남는다). 중심이 움직이면 U ← U + δ_a(다음 수로 올림), L ← L − max_{j≠a} δ_j(다음 수로 내림)이고 δ_j는 중심 이동의 U다(삼각 부등식). 중심 a의 점을 건너뛰는 조건은 z = max(L, s(a)) > fl(U·κ)이고 z > 2⁻⁴⁸⁰이다(s(a)는 다른 중심까지 거리의 반의 하한). 그러면 j ≠ a마다 d_j ≥ z(또는 d_j ≥ 2s(a) − d_a ≥ z)이고 d_j > d_a(1 + 2γ)(1 + 3u), d_j² > 2⁻⁹⁶⁰라서 sqd_j ≥ d_j²(1 − γ_{p+3}) − 2⁻¹⁰⁷⁴ > d_a²(1 + γ_{p+3}) + 2⁻¹⁰⁷⁴ ≥ sqd_a다((1 + 4γ)(1 − γ_{p+3}) − (1 + γ_{p+3}) ≥ 4γ − 2γ_{p+3} − 4γγ_{p+3} > 0이고 그 몫에 2⁻⁹⁶⁰을 곱한 것이 2·2⁻¹⁰⁷⁴보다 크다). 곧 훑기의 최솟값이 엄격하게 a이고 동점 규칙이 끼지 않는다. 그래서 건너뛴 점의 라벨은 Lloyd의 것이고, 라벨이 같으니 중심(CL48의 같은 합)과 다음 반복도 같은 비트다. ∎
- **정리 5 (전개식의 거르기, CL49·CL60)**. 전개식 v = fl(fl(n_x − 2g) + n_c)에서 n_x = ‖x‖², n_c = ‖c‖², g = x·c의 계산값은 어떤 차례로 더하든(fma든 아니든) |n_x − ‖x‖²| ≤ γ_p‖x‖², |g − x·c| ≤ γ_p Σ|x_d c_d| ≤ γ_p‖x‖‖c‖이다(Higham 2002, §3.1의 내적 한계는 합의 차례와 무관하다). 덧셈 둘을 더해 |v − d²| ≤ γ_{p+2}(‖x‖ + ‖c‖)² + η ≤ 2γ_{p+2}(‖x‖² + ‖c‖²) + η, η ≤ 4·2⁻¹⁰⁷⁴(아래넘침)다. 커널은 r_x = up₂(fl(n_x)(1 + γ)) ≥ ‖x‖², r_c도 같게 두고 E = fl(c_E·fl(r_x + r_c)), c_E = 8(p + 4)u ≥ 2·(2γ_{p+2})로 그 한계를 덮는다(반올림 둘의 몫까지). b를 fl(v + E)가 가장 작은 중심, lim = fl(fl(fl(v_b + E_b)·κ′) + τ), κ′ = 1 + 8(p + 4)u, τ = 2⁻⁹⁶⁰라 하자. A_j = fl(v_j − E_j) > lim인 중심 j는 d_j² ≥ A_j(1 − u) − η > ((v_b + E_b)κ′(1 − u)² + τ)(1 − u)² − η ≥ d_b²·κ′(1 − u)⁴ + τ/2라서 sqd_j ≥ d_j²(1 − γ_{p+3}) − 2⁻¹⁰⁷⁴ > d_b²(1 + γ_{p+3}) + 2⁻¹⁰⁷⁴ ≥ sqd_b이다(κ′(1 − u)⁴(1 − γ_{p+3}) ≥ 1 + γ_{p+3}). 곧 j는 (sqd, 번호)의 최솟값일 수 없고, b는 A_b ≤ v_b + E_b ≤ lim이라 빠지지 않는다. 그래서 남은 중심들의 최솟값이 모든 중심의 최솟값이고, `dgemm`이 어떤 차례로 더하든(스레드 수, Accelerate의 판) 결과가 같은 비트다. 뺀 중심들의 거리 하한은 √(A_min(1 − 4u))(1 − 2u)를 내린 값이다(경계의 L). ∎
- **측정** (CL2 마일스톤, 참값 33집합; `bench/cluster/accuracy.md`): 중심은 계약의 0.17배, 관성 0.28배, 근처 동점 없이 라벨이 다른 집합 0. sklearn과 R의 같은 시작도 같은 정도이고 sklearn은 비정상 수 집합에서 한 군집이다. 네 길(Lloyd, 거르기, 경계, 경계와 거르기)이 참값 집합과 10⁴–10¹² 평행 이동, 30 × 30 격자, 비정상 수 척도, 40차원 37군집에서 같은 비트다(`test_every_path_gives_lloyds_bits`).


### 4.3 k-medoids·퍼지
- PAM: 결과의 medoid 집합은 "계산한 변화(CL65의 FastPAM1 분해)로 음수인 교환이 없는" 집합이다(`converged=True`). 목적값은 정확한 합과 |O − O*| ≤ C_PAM·γ_{⌈log₂ n⌉+p+2}·O* 안이다(C_PAM = 2; 거리 하나가 γ_{p+2}, 덩이 합이 γ_{⌈log₂ n⌉}). 교환과 BUILD의 비교는 그 오차 안의 가까운 동점에서만 정확한 산술의 선택과 다르다. 측정: 참값 63가지에서 계약의 0.39배, 다른 medoid 8가지는 모두 정확한 동점(1차원 짝수 군집의 가운데 두 점, 격자)이다.
- 퍼지: 주어진 중심에서 소속도는 |u − u*| ≤ 2u·(max(q, 1)(2p + 7) + k + 2), q = 1/(m − 1)(거리의 γ_{p+3}이 비 r에서 두 배, 거듭제곱에서 q배, 정규화의 합에서 k가 된다; 소속도가 [0, 1]이라 절댓값), 한 걸음의 중심은 |c − c*| ≤ 4γ_{⌈log₂ n⌉+2p+k+8}·max|x|다(소속도의 오차가 무게 uᵐ으로 옮고, 덩이 합). 측정: m = 1.5, 2, 3의 소속도는 계약의 0.046, 0.062, 0.031배, 중심은 0.065배(`tests/cluster/ref/fuzzy.npz`; 참값은 m = 1.5, 2에서 정확한 유리수, m = 3에서 Arb).

### 4.4 밀도
- **거리**: 반경의 비교, 핵심 거리, 상호 도달 거리, 도달 거리는 spatial의 계산한 거리(유클리드: |d̂ − d| ≤ γ_{p/2+2}·d와 아래넘침의 절대항; 단일 연결의 높이와 같은 한계, `refdata.distance_bound`)와 그 선택(최댓값, k번째 값, 비교)이다. d/α는 반올림 하나가 더해진다.
- **이산 결과**: DBSCAN의 분할, HDBSCAN의 트리·줄인 트리·고르기, OPTICS의 차례는 계산한 값과 정한 동점 규칙으로 정의대로 고른 것이다(근사 없음). 정확한 거리로 고른 것과는 가까운 동점(eps에 계약 안으로 붙은 짝, 계약 안의 두 무게, 한계 안의 두 안정성)에서만 다를 수 있고, 참값 도구는 그런 자리(`near`, `selection_gap`)를 표시한다. 참값 14벌·24가지에서 그런 차이는 없었다.
- **정리 6 (트리와 훑기, Borůvka와 Prim이 같은 비트, CL67·CL72)**. (1) DBSCAN·핵심 거리의 KD 트리 찾기는 spatial의 증명(docs/design/spatial.md §3.4; kdtree.hpp의 머리)대로 모든 점이 반경 밖으로 증명된 마디만 자르고, 통째로 세는 마디는 모든 점이 반경 안으로 증명된 것이다. 남은 짝은 훑기와 같은 커널로 재므로 이웃의 집합과 거리가 같다. 무게의 합은 열쇠 차례라 찾는 차례와 무관하고, 핵심 거리는 k번째 값이라 고르는 방법과 무관하다. (2) 상호 도달 거리의 Borůvka는 점 q의 반경 r(q의 지금 최선과 그 성분의 지금 최선 중 작은 것) 밖의 변만 자른다: core_q > r이면 q의 모든 변이 core_q 이상이고, 마디의 가장 작은 핵심 거리가 r보다 크면 그 마디의 모든 변이 그 이상이며, spatial의 증명으로 d̂ > T_d(r) = up(up(r)·α)이면 d̂/α > up(r)라 fl(d̂/α) ≥ up(r) > r이다(fl은 단조, up(r)은 수). 세 경우 모두 변의 무게가 r보다 엄격히 크다. 성분 C의 가장 작은 변 e* = (w*, a, b)(변의 전순서에서)는 언제나 w* ≤ r이라(r은 C의 실제 변의 무게 이상) 잘리지 않고, 같은 무게의 변도 잘리지 않아 e*의 주인 점은 e*를 고른다. 그래서 라운드마다 성분의 가장 작은 변이 정해지고, 변의 전순서에서 최소 신장 트리는 하나이므로 Prim(같은 무게의 변을 작은 다른 끝으로 바꾸는 규칙, 엄격한 건너뛰기)과 같은 변의 집합이다. 정렬과 Kruskal은 그 집합만의 함수다. ∎ 테스트가 세 거리 × 두 길 × α ∈ {1, 0.8}에서 비트까지 본다.
- **HDBSCAN의 실수 결과**. ŵ가 위의 한계 안이면 λ̂ = fl(1/ŵ)는 |λ̂ − λ| ≤ γ_{p/2+4}·λ다. 안정성의 항 (λ − λ_birth)·m은 |t̂ − t| ≤ γ_{p/2+6}·(λ + λ_birth)·m이고(뺄셈의 상쇄는 절대 한계로 덮는다) 행의 차례로 rows개를 더하면 |Ŝ − S| ≤ γ_{p/2+6+rows}·S⁺, S⁺ = Σ(λ + λ_birth)·m이다(`refdata.stability_bound`). 확률 min(λ, M)/M과 GLOSH (M − λ)/M은 둘 다 [0, 1]이고 절대 2γ_{p/2+5} 안이다(`membership_strength_bound`; 분자의 오차 γ(M + λ)를 M으로 나눈다). 측정(`bench/cluster/accuracy.md`): 핵심 거리 계약의 0.66배, 트리의 무게 0.56배, 안정성 0.19배, 확률 0.21배, GLOSH 0.17배, OPTICS의 도달 거리 0.56배.
- **OPTICS**: 도달 거리는 계산한 거리와 핵심 거리의 최댓값(정확한 선택), 차례는 (도달 거리, 열쇠)의 정의, ξ 추출은 sklearn과 같은 부동소수 연산이라 그 배열의 정해진 함수다.

### 4.5 혼합 모형
CL4 마일스톤이 정하고 `tests/cluster/test_mixture.py`와 `bench/cluster/accuracy.md`(스윕 `mixture`)가 지킨다. 모형은 §4.1의 정리 3과 같다(IEEE double, 가장 가까운 값으로 반올림, 적은 fma 말고는 축약 없음). 더해 libm의 `log`와 배열의 NEON `exp`가 1 ULP 안이라고 둔다(상대 2u; 배열 영역의 측정, docs/design/array.md §5.4). 아래 한계는 1차다(γ끼리의 곱을 버린다). 시험과 스윕은 2배(`refdata.C_FIRST`)로 잰다.
- **정리 7 (주어진 인자에서의 로그 밀도, CL75)**. 성분의 μ와 위 삼각 U가 주어졌을 때 z = x − μ, a_e = Σ_{d≤e}\|U_de\|\|z_d\|, q⁺ = Σ_e a_e²(\|U\|와 \|z\|의 이차형식), Λ⁺ = Σ_d\|log U_dd\|(spherical은 p\|log U\|), c = p log 2π라 하면 \|ℓ̂ − ℓ\| ≤ E := γ_{3p+7}(q⁺ + c)/2 + γ_{p+2}Λ⁺ + u\|ℓ\|다. 증명: ẑ_d = z_d(1 + δ). y_e는 e + 1개 항의 fma 사슬이라 \|ŷ_e − y_e\| ≤ γ_{p+1}a_e(판의 0은 정확한 0을 더한다; diag·spherical은 곱 하나와 차 하나로 γ₂). ŷ_e² − y_e²는 2\|y_e\|\|η_e\| + η_e² ≤ (2γ_{p+1} + γ_{p+1}²)a_e²(\|y_e\| ≤ a_e)이고 q의 p항 fma 사슬이 γ_p Σŷ² ≤ γ_p(1 + γ_{p+1})²q⁺를 더해 \|q̂ − q\| ≤ γ_{3p+5}q⁺다(γ_a + γ_b + γ_aγ_b ≤ γ_{a+b}). ĉ = fl(p·fl(log fl(2π)))는 γ₄c, 더하기 c + q는 u(c + q⁺), −½은 정확하고, log U_dd의 합은 log마다 2u\|log U_dd\|에 p항의 합 γ_p라 γ_{p+2}Λ⁺, 마지막 더하기가 u\|ℓ\|다. 모으면 E다. ∎ 가중 로그 밀도 w = ℓ + log π는 2u\|log π\| + u\|w\|가 더해진다(Ê).
  - **혼합의 로그 밀도와 책임도**. m̂ = ŵ_{j*}, â_l = fl(ŵ_l − m̂)라 하면 \|â_l − a_l\| ≤ Ê_l + Ê_{j*} + u\|a_l\|(a_l = w_l − w_{j*}), exp가 2u를 더해 ê_l의 상대 오차는 η_l ≤ Ê_l + Ê_{j*} + u\|a_l\| + 2u, ŝ의 상대 오차는 max η + γ_{k−1}이다. 그래서 \|log p̂ − log p\| ≤ 2 max Ê + u max\|a\| + γ_{k+2} + u(log k + \|log p\|), \|r̂_j − r_j\| ≤ r_j(4 max Ê + 2u max\|a\| + γ_{k+4})(+ 아래넘침의 2⁻¹⁰⁷⁴)다. 설계 초안의 "c·u·(p² 꼴 + \|마할라노비스 항\| + \|log det\|)"와 "상대 c·u·(1 + 지수의 인자 크기)"를 이렇게 정했다(`refdata.mixture_bounds`).
- **계약 8 (공분산의 인자, CL79)**. diag·spherical은 Û = fl(1/fl(√σ²))라 \|Û − U*\| ≤ γ₂\|U*\|다(√와 나눗셈이 올바르게 반올림된다; 증명). full·tied의 Û(Cholesky와 삼각 역)는 노름 꼴의 1차 한계 max\|Û − U*\| ≤ C_PREC·p·γ_{p+1}·κ₂(Σ)·max\|U*\|, C_PREC = 2로 지킨다(Higham 2002의 Cholesky 후방 오차 \|ΔΣ\| ≤ γ_{p+1}\|L\|\|Lᵀ\|와 삼각 역의 전방 오차를 κ로 묶은 꼴이고 상수는 측정이다). 거의 특이한 공분산은 막지 않는다(CL79). `refdata.precision_bound`.
- **정리 9 (주어진 책임도에서의 M 단계, CL75)**. m개 항(tied는 m·k개)의 덩이 합이 음이 아닌 항의 합이라 \|n̂k − nk\| ≤ γ_m nk, 평균은 A_d = Σrᵢ\|x_{id}\|/nk로 \|μ̂_d − μ_d\| ≤ γ_{2m+3}A_d, 공분산은 B_ab = Σr\|z_a\|\|z_b\|/nk, C_b = Σr\|z_b\|로 \|Σ̂_ab − Σ_ab\| ≤ γ_{2m+5}B_ab + (\|δμ_a\|C_b + \|δμ_b\|C_a)/nk + u\|Σ_ab\|(평균의 오차가 두 번째 훑기에 1차로 들어가는 몫; tied는 γ_{2mk+k+5}와 Σnk, spherical은 대각의 평균이 γ_{p+1}을 더한다), 무게는 γ_{2m+k+2}w다. 증명은 각 단계의 반올림을 차례로 더한 것이다(fma 사슬 γ_m, 나눗셈 u, 평균을 뺀 차 u\|z\| + \|δμ\|). `refdata.mstep_bound`.
- **측정** (참값 `mixture.npz`: 6가지 혼합의 점 50개, 주어진 인자 U_f = sklearn `precisions_cholesky_`의 double, 정확한 인자, 반올림한 책임도의 M 단계; `bench/cluster/accuracy.md`): 성분의 로그 밀도는 계약의 0.095배, 혼합의 로그 밀도 0.031배, 책임도 0.016배, 인자는 full·tied 0.017배(κ = 10⁴의 full/k2p5는 0.0028배)와 diag·spherical 0.71배(증명한 γ₂의 몫), M 단계 0.053배다. sklearn의 같은 입력은 성분의 로그 밀도가 sd 계약의 1.36배(diag의 펼친 꼴), M 단계가 9.3배(diag·tied·spherical의 한 번 훑기)이고 인자의 가장 큰 오차는 sd의 21배다(9.5×10⁻¹² 대 4.5×10⁻¹³).
- **반복**: EM의 경로와 멈춘 자리는 약속하지 않는다(정해진 멈춤 규칙의 결과). `converged=True`이면 마지막 두 하한의 차가 `tol` 미만이고(시험이 `max_iter`를 하나 줄여 확인한다), `labels`는 결과의 모수에서의 `predict`와 같으며, `loglik`·`bic`·`aic`는 결과의 모수에서 위의 한계(정리 7의 합)다. 같은 시작에서 sklearn과 같은 반복 수로 같은 고정점 근처에 닿는다(§2.5).

### 4.6 지표
CL5 마일스톤이 정하고 증명했다(참값은 `tools/cluster_ref.py`의 `metrics`; 한계 함수는 `tests/cluster/refdata.py`; 잰 값은 `bench/cluster/accuracy.md`의 `metrics`). γ_k = ku/(1 − ku), 1차 꼴은 γ끼리의 곱을 버린다.

1. **ARI, RI, 짝 혼동 행렬, 분할표**: 정수 셈은 정확하다(n(n − 1) < 2⁶³이라 짝 수와 그 두 배가 int64에 든다; ARI의 분자·분모는 Python 정수라 넘치지 않는다). ARI와 RI는 정확한 정수 둘의 몫을 CPython의 정수 나눗셈으로 한 번 반올림한다: **올바르게 반올림한 값**(0.5 ULP). 특수 경우(sklearn의 1.0)는 정확한 값이다.
2. **FM**: FM = √(TP²/((TP + FP)(TP + FN))). s = 60 − ⌊(bits(분자) − bits(분모))/2⌋로 q = ⌊분자·2²ˢ/분모⌋, r = ⌊√q⌋(`math.isqrt`)이면 ⌊√⌊x⌋⌋ = ⌊√x⌋라 r = ⌊FM·2ˢ⌋이고 r ≥ 2⁵⁹다. m = 2r + [나머지가 있거나 r² ≠ q]는 FM·2ˢ⁺¹을 아래로 자른 값에 끈적한 비트를 붙인 것이라, 53비트로의 가장 가까운 짝수 반올림(`float(m)`)은 FM·2ˢ⁺¹의 올바른 반올림이다(반올림 자리가 r의 끝자리보다 6비트 이상 위이고, 끈적한 비트가 정확히 가운데인 경우와 가운데를 넘는 경우를 가른다). 2⁻ˢ⁻¹을 곱하는 것은 정확하다: **올바르게 반올림한 값**.
3. **로그 L(x, y)** (정확한 정수 x, y > 0): x/y가 [1/2, 3/2] 밖이면 x, y의 변환과 나눗셈으로 몫은 상대 γ₃ 안이고 |log| ≥ log(3/2)라 L의 상대오차는 γ₃/log(3/2) + u ≤ 8.5u다. 안이면 d = (x − y)/y는 정확한 차의 변환·y의 변환·나눗셈으로 상대 γ₃, |d| ≤ 1/2에서 |d/((1 + d)log1p(d))| ≤ 1.443이라 L의 상대오차는 1.443γ₃ + u ≤ 5.4u다(log·log1p 자체는 1 ULP 안으로 본다). 그래서 어디서나 |L̂ − L| ≤ 9u|L|.
4. **MI와 엔트로피**: 항 t = (c/n)·L은 나눗셈과 곱의 두 반올림이 더해져 |t̂ − t| ≤ γ₁₁|t|이고, m개 항의 짝 합(여덟 항까지 차례, 그 위는 반으로)은 깊이 ⌈log₂ m⌉ + 5 안이라 |MÎ − MI| ≤ γ_{⌈log₂ m⌉+16}·T⁺, T⁺ = Σ|tᵢⱼ|다(엔트로피는 항이 모두 양수라 T⁺ = H). 0에서 자르기는 참값(≥ 0) 쪽으로만 움직인다.
5. **기대 MI**: 짝 (α, β)마다 무게의 비 r̂(c)는 정확한 정수의 변환(2⁵³ 아래), 두 곱, 나눗셈으로 상대 γ₃이고 누적 곱 하나가 더해져, 최빈값 c*에서 |c − c*|걸음 떨어진 ŵ(c)는 상대 γ_{4|c−c*|} 안이다. 잘라 낸 꼬리는 log-오목(비가 c에 줄어든다)이라 남은 질량이 ŵ·r/(1 − r) 아래이고 2⁻⁸⁰·합 아래에서 멈춘다(무시할 만하다). 합 S와 분자 Σŵt̂(차례 합)의 오차를 1차로 모으면 짝의 값 v = Σ P t의 오차는 u·[Σ_c P(c)|t(c)|(4|c − c*| + 2L + 12) + |v|(4D + 2L + 2)] 안이다(D = Σ P|c − c*|, L = ⌈log₂(2n)⌉ + 5는 합의 깊이의 한계: 창의 길이와 짝의 수가 2n 아래). 개수를 곱하는 반올림과 짝들의 짝 합을 더해 **|ÊMI − EMI| ≤ u·E⁺**(1차), E⁺는 위의 괄호를 모든 행–열 짝(개수만큼)에 더한 것이고 참값 도구가 함께 낸다(필드 `plus`).
6. **NMI, AMI, 동질성, 완전성, V**: 몫 q = N/D의 오차는 1차로 (ΔN + q·ΔD)/|D| + u·q다(`refdata.ratio_bound`). NMI는 N = MI, D = 정규화 값(평균·기하 평균의 반올림 γ₂·D를 더한다), AMI는 N = MI − E, D = 정규화 값 − E(ΔN = ΔMI + ΔE, ΔD = ΔH + ΔE + γ₂·D), 동질성·완전성은 MI/H, V는 2MI/(H_true + H_pred)다. 자르기(≤ 1)는 참값 쪽으로만 움직인다. AMI의 분모가 ±2⁻⁵² 안으로 가는 sklearn의 지킴은 특수 경우(CL84)가 아니면 참값의 분모가 0이 아니어서(한쪽이 한 군집이나 모두 홑원소가 아니면 재배치로 MI가 줄어 E < min(H)) 한계가 그 분모로 정해진다.
7. **실루엣**: 거리는 spatial의 계약(유클리드 |d̂ − d| ≤ γ_{p/2+2}·d; 나머지 셋은 그 안)이고, 군집 합은 네 사슬(길이 ⌈m/4⌉)과 두 덧셈이라 상대 γ_{⌈m/4⌉+p/2+4}, 평균은 한 번 더, b의 최솟값은 정확하다. (b̂ − â)/max(â, b̂)의 절대오차는 1차로 2γ_K·(a + b)/max(a, b) + 2u 안이고 (a + b)/max(a, b) ≤ 2라 **|ŝ − s| ≤ 4γ_{⌈n_max/4⌉+p/2+8} + 2u**(n_max는 가장 큰 군집)다. 평균은 덩이 합의 한계를 더한다(값이 [−1, 1]이라 γ_{⌈log₂ n⌉+...}의 작은 항).
8. **Calinski–Harabasz**: 자료에서 첫 점을 뺀 값 y(상대 u)와 그 최대 크기 M으로, 군집 평균(첫 점을 뺀 덩이 합)의 좌표 오차는 δ = γ_{⌈log₂ n⌉+13}·M 안이다. W(μ̂) − W(μ) = Σn_c‖μ̂_c − μ_c‖² ≤ npδ²(평균이 W를 가장 작게 하므로 2차), B의 오차는 Σn_c·2‖μ_c − μ‖·‖Δ(μ_c − μ)‖ ≤ 4√p·δ·Σn_c‖μ_c − μ‖ ≤ 4√p·δ·√(nB)(코시–슈바르츠)에 합의 반올림 γ_A·B를 더한다(A = p + ⌈log₂ n⌉ + K + 16). 그래서 **|ĈH − CH| ≤ CH·[γ_A(3 + 4M√(np/B)) + npδ²/W]**(1차). B가 작은(군집의 평균이 모두 가까운) 자료에서 한계가 커지는 것은 그 조건수다.
9. **Davies–Bouldin**: 산포 s_c와 중심 거리 d_ce는 평균의 오차로 각각 √p·δ, 2√p·δ 안에서 움직이고 거리의 계약과 합의 반올림이 더해져, 몫 (s_c + s_e)/d_ce의 상대오차는 γ_A + 4√p·δ/m 안이다(m: 0이 아닌 s_c + s_e와 d_ce의 가장 작은 값). 최댓값과 평균은 상대오차를 키우지 않아 **|D̂B − DB| ≤ DB·(2γ_A + 4√p·δ/m)**(1차)다. 0의 판정은 정확한 0이다(CL86).
10. **gap의 log W** (주어진 분할): dᵖᵒʷᵉʳ은 상대 max(1, power)·γ_{p/2+2} + u, 행의 네 사슬과 행들의 짝 합, 군집 몫과 군집의 합으로 Ŵ = W(1 + θ), |θ| ≤ γ_G, G = max(1, power)(p/2 + 2) + ⌈n/4⌉ + ⌈log₂ n⌉ + K + 12이고, log와 격자의 되돌림(e·power·log 2의 곱과 뺄셈)으로 **|log Ŵ − log W| ≤ γ_G/(1 − γ_G) + 2u|log W|**다. gap 통계량은 난수가 정해지면 벌마다 이 한계와 그 벌의 군집(k-평균·PAM의 계약)을 따른다.
11. **Affinity Propagation**: 수의 계약이 아니라 연산의 계약이다: 잡음이 0인 sklearn과 같은 연산 차례(CL92)라 열쇠 순서의 같은 유사도에서 exemplar·라벨·반복 수가 같다(테스트가 105가지에서 본다).

잰 값(정확도 스윕 `metrics`, 2026-11-27; 한계의 몫은 `bench/cluster/accuracy.md`): ARI·RI·FM은 라벨 짝 13가지 모두 참값의 가장 가까운 double(0 ULP), MI·엔트로피는 상대 1.8u 안(몫 0.1 아래), 기대 MI는 몫 0.024, NMI·AMI·동질성·완전성·V는 몫 0.043 아래, 실루엣은 절대 1.7u 안(몫 0.043), CH·DB는 상대 4×10⁻¹⁵ 안(몫 0.023 아래), log W는 상대 2u 안(몫 0.46: huge/k3p2의 격자 되돌림)이다. sklearn은 같은 한계로 FM이 1 ULP(large/1e6; 몫 2), AMI가 몫 4.4, 기대 MI가 몫 6.2이고(lgamma 차), shifted·subnormal 자료의 실루엣(0.75)과 CH(절대 142)·DB(0.51)가 한계를 크게 넘는다(전개식 거리, 원점에서 잰 평균).

---

## 5. 성능
**성능이 먼저이고 정확도가 그다음이다**(API.md §4의 규칙 3). 목표는 1스레드에서 비교 대상과 같거나 빠르고, 10스레드에서 그보다 여러 배 빠른 것이다. 잰 값은 탐침의 한 번씩 잰 어림이다(Apple M4 Pro, macOS 26.6.2, 부하 평균 1.2–2.6; sklearn은 OpenMP 기본 스레드 수, R은 1스레드, scipy `linkage`는 1스레드).

| 사례 | 비교 대상의 시간 | sd 목표 (1 / 10스레드) |
|---|---|---|
| `linkage`, n = 5000, p = 5, 관측값 | scipy single 65, complete 151, average 159, ward 147, centroid 155 ms; R `hclust(average)` 169 ms(`dist` 제외) | scipy 이하 / scipy의 1/3 이하(행렬 만들기와 높이 다시 계산이 나란하다; NN-chain은 차례) |
| `linkage` single·ward, n = 10⁵, p = 5 | scipy는 행렬 40 GB라 못 한다 | 행렬 없이 끝난다(single은 Borůvka로 1초 아래, ward는 O(n²p)라 수십 초 어림; CL1 마일스톤에서 잰다) |
| `kmeans`, n = 10⁵, p = 10, k = 10, `n_init=1` | sklearn 127 ms(여러 스레드), R `kmeans` 355 ms(Hartigan–Wong, `Quick-TRANSfer stage steps exceeded maximum` 경고) | sklearn 이하 / sklearn의 1/3 이하 |
| `dbscan`, n = 2×10⁴, p = 2 | sklearn 31 ms | sklearn 이하 / 1/4 이하 |
| `hdbscan`, n = 2×10⁴, p = 2 | sklearn 623 ms | sklearn의 1/3 이하 / 1/10 이하(Borůvka) |
| `gaussian_mixture` full, k = 5, n = 2×10⁴, p = 10 | sklearn 65 ms | sklearn 이하 / 1/3 이하 |
| `silhouette_score`, n = 2×10⁴, p = 10, k = 5 | sklearn 1090 ms(1스레드 1089), R `silhouette(dist)` 1738 ms | sklearn 이하 / 1/8 이하: 525 / 61 ms(CL5 마일스톤) |

- 큰 n의 계층 군집: O(n²) 메모리가 첫 벽이다. 행렬 없는 길(§3.2)이 scipy가 못 푸는 크기를 푼다. single은 MST라 O(n log n) 어림(낮은 차원의 Borůvka), ward·centroid·median은 O(n²p) 시간이다. NN-chain은 본디 차례 알고리즘이라 병렬의 몫은 거리 갱신과 높이 다시 계산이다.
- 2차원 단일 연결은 Delaunay 변 위의 MST로 O(n log n)이 되지만(유클리드 MST는 Delaunay의 부분 그래프) 첫 판에 넣지 않는다: Borůvka로 충분한지 CL1 마일스톤에서 본다.
- **CL0 마일스톤의 시제품** (CL2 마일스톤이 지운 `prototype.cpp`의 `_core.cluster._proto_*`, `tools/cluster_probe.py proto`, 기록 `tests/cluster/data/prototype.json`; Apple M4 Pro, 부하 평균 1.3–1.8, 세 번 중 가장 빠른 값; "1 / 10"은 sd의 스레드 수, Accelerate는 자기 스레드):

  | 측정 | 결과 |
  |---|---|
  | k-평균의 거리, n = 10⁵ (2만 점의 p = 300) | 차의 꼴(spatial의 타일 커널) 1 / 10스레드와 Accelerate 전개식(`cblas_dgemm`, 4096행 덩이): p = 2·k = 10 1.51 / 0.28 대 0.86 ms, p = 10·k = 100 11.3 / 1.47 대 4.67, p = 50·k = 100 38.0 / 4.46 대 7.50, p = 100·k = 100 73.7 / 8.60 대 10.6, p = 300·k = 100 43.1 / 4.98 대 5.21. 1스레드의 비는 1.76–8.28(p·k가 클수록 크다), 10스레드는 0.29–0.96. 증명된 한계로 거르는 꼴은 점마다 후보 1.00개, 시제품의 1스레드 5.4–43.9 ms(점마다의 거르기 고리가 느리다; 다듬지 않았다) |
  | 거리의 정확성 | 2100점의 덩어리를 10⁴, 10⁸, 10¹²만큼 옮기면 차의 꼴·거르는 꼴은 정확한 라벨과 0점 다르고 전개식과 sklearn `predict`는 0, 308, 1425점 다르다. 참값의 반올림한 중심에서 차의 꼴은 subnormal/k3p2의 40점만 틀렸다(제곱의 아래넘침; CL43) |
  | 경계 가속 (CL42) | 참값 집합 31개(무게 없는 것)에서 여유 계수 0, 1, 4, 16 모두 Lloyd와 라벨·중심·반복 수가 같은 비트이고 건너뛴 거리의 비율이 계수와 무관(평균 0.401). 동점이 많은 60×60 격자, 이등분선에서 1 ULP 떨어진 점 4000개, 1e-15의 잡음을 곱한 덩어리에서도 같은 비트. 10⁵점에서 Lloyd 대 Hamerly(여유 1): p = 2·k = 10 126 / 18 대 92 / 14 ms(1 / 10스레드), p = 10·k = 10 332 / 44 대 141 / 24, p = 2·k = 100 1988 / 245 대 772 / 95, p = 10·k = 100 478 / 55 대 276 / 34; 건너뛴 비율 0.85–0.95 |
  | 최소 신장 트리 (CL47) | Prim, KD 트리 Borůvka, 훑기 Borůvka가 같은 변과 무게(p = 2, 5, 16의 3000점, 40×40 격자). 1스레드 Prim 대 Borůvka: n = 10⁴에서 p = 2 163 대 26 ms, p = 3 163 대 50, p = 5 181 대 124, p = 8 209 대 533, p = 16 366 대 5635; n = 3×10⁴에서 p = 5 1698 대 520, p = 8 2140 대 1935. 10스레드 n = 10⁴ p = 8 100 대 64. Borůvka만 n = 10⁵: p = 2 468 / 78, p = 5 2216 / 275, p = 8 11267 / 1277 ms |
  | NN-chain의 동점 (CL46) | 참값 32개 집합 × complete·average·weighted·ward 128가지에서 사슬((거리, 색인)의 이웃; 점은 열쇠 순서로 넣었다)이 정의의 트리와 다른 4가지(격자 셋과 비정상 수 척도의 ward)는 모두 이웃 찾기에서 정확한 동점을 만났고, 동점 없이 다른 것은 0이다. 동점 표시는 46가지(동점 자료), 같은 높이 표시는 42가지. n = 5000, p = 5에서 사슬 206–212 ms, 검사판 157–163 ms(검사판이 더 빨랐다: 코드 배치의 차이로 본다), scipy 154–166 ms |
  | 높이 (CL45) | 트리 위의 Lance–Williams 재귀(정밀도 float64, 자료는 CL43의 척도로): 일곱 연결법 모두 참값과 2 ULP 안(single·complete 1, average 2, weighted 1, centroid 2, median 1, ward 2), scipy의 누적은 1–3 ULP. 설계 초안의 정의대로의 합은 average 3 ULP, ward 무게 중심 5.2×10⁸ ULP(shifted/k3p2), 91 ULP(collinear/p2). n = 5000에서 다시 계산 average 4.5 ms·ward 1.0 ms 대 사슬 162·155 ms |
  | 중심의 합 (CL48) | 참값 라벨의 무게 평균을 덩이 합으로: γ_{⌈log₂ n⌉+1}·max\|x\|의 0.38배(shifted/k3p2)라 초안 상수 C_CENTER = 2 |

- **CL1 마일스톤의 측정** (계층; Apple M4 Pro, 부하 평균 2.3–3.5, 세 번 중 가장 빠른 값, "1 / 10"은 sd의 스레드 수, scipy와 R은 1스레드, R은 `dist` 포함):

  | 사례 | sd 1 / 10스레드 | scipy | R |
  |---|---|---|---|
  | `linkage` n = 5000, p = 5: single | 28.9 / 5.3 ms | 61.0 | `hclust` 175 |
  | complete | 110 / 103 | 155 | 203 |
  | average | 154 / 108 | 165 | 205 |
  | weighted | 153 / 107 | 160 | 206 (`mcquitty`) |
  | centroid | 110 / 78 | 162 | 212 (제곱 거리) |
  | median | 105 / 77 | 151 | 208 (제곱 거리) |
  | ward | 115 / 70 | 154 | 198 (`ward.D2`) |
  | `linkage` n = 10⁵, p = 5, 행렬 없이: single | 1150 / 173 ms | (행렬 40 GB) | — |
  | ward | — / 7.0 s | (행렬 40 GB) | — |
  | `divisive` n = 1000, p = 5 | 4.8 / 4.6 ms | — | `diana` 354 |
  | `cophenet` n = 5000 | 25 / 8.7 ms | 20.8 | — |
  | `optimal_leaf_ordering` n = 1000 | 40.8 / 5.7 ms | 399 (최소가 아닐 때가 있다) | — |

  1스레드는 모든 연결법이 scipy보다 빠르다. 10스레드의 몫은 single(Borůvka)·높이 재귀·`pdist`·무게 중심 찾기이고, NN-chain과 일반 알고리즘은 본디 차례라(§3.10) complete·average·weighted는 행렬 NN-chain(약 100 ms)이 남는다: §5 첫 표의 10스레드 목표(scipy의 1/3)는 single만 이르렀다. complete·average·weighted·centroid·median·ward의 10스레드는 CL7 마일스톤의 점검거리다(메인 결정, 사용자 위임 2026-11-27; §8). 높이 재귀는 n = 5000에서 1스레드 24–47 ms, 10스레드 4–10 ms다.
- **최소 신장 트리** (CL54; `tools/cluster_probe.py proto.mst`, 기록 `tests/cluster/data/prototype.json`의 "mst/time", 1 / 10스레드 ms): 세 길(Prim, 점마다의 Borůvka, 이중 트리 Borůvka)이 같은 변과 무게를 낸다(p = 2, 5, 16의 3000점, 40×40 격자). 점마다의 Borůvka 대 Prim: p = 2는 n = 1024에서 1.1 대 1.8(1스레드), p = 5는 n = 2048에서 9.4 / 1.6 대 7.8 / 7.8, n = 10⁴에서 80 / 12 대 187 / 58, p = 8은 n = 10⁴에서 291 / 36 대 237 / 59, n = 3×10⁴에서 1176 / 136 대 2398 / 396, p = 16은 어디서나 Prim이 빠르다(n = 10⁴에서 408 대 3536). 이중 트리는 p = 2에서 점마다보다 조금 빠르고(n = 3×10⁴에서 42 대 56) p ≥ 3에서 1.2–6배 느리다(p = 8, n = 10⁴에서 1420 대 291). Borůvka만 n = 10⁵: p = 2 217 / 40, p = 3 373 / 59, p = 5 1280 / 169 ms.
- **CL2 마일스톤의 측정** (분할; Apple M4 Pro, 부하 평균 2–3, 세 번 중 가장 빠른 값; "1 / 10"은 sd의 스레드 수, sklearn은 OpenMP 스레드("1"은 threadpoolctl로 1), R은 1스레드에 자료 옮기기 포함):

  | 사례 | sd 1 / 10스레드 | sklearn 1 / 여러 스레드 | R |
  |---|---|---|---|
  | `kmeans` n = 10⁵, p = 10, k = 10, 같은 시작, `tol=1e-4`(11번) | 26.4 / 5.9 ms | 32.0 / 14.6 | `kmeans(algorithm='Lloyd')` 117번 반복(멈춤 규칙이 없다) |
  | `kmeans` 같은 자료, k-means++ | 20.3 / 5.5 | 19.1 / 16.4 | — |
  | `minibatch_kmeans` 같은 자료 | 7.6 / 4.7 | 90.4 / 128 | — |
  | `bisecting_kmeans` 같은 자료 | 40.8 / 12.8 | 31.2 / 30.9 | — |
  | `kmedoids` PAM n = 2000, p = 5, k = 10 | 98 / 18 | — | `pam` 672 |
  | `kmedoids` CLARA n = 10⁵, p = 5, k = 10 | 25.9 / 5.6 | — | `clara` 27.8 |
  | `fuzzy_cmeans` n = 10⁵, p = 10, k = 10(15번) | 169 / 23 | — | — |

  거리 커널(`tools/cluster_probe.py proto.kmeans`, 기록 `tests/cluster/data/prototype.json`의 "kmeans/assign"과 "kmeans/paths"; 10⁵점의 배정 한 번, ms, 1 / 10스레드): 차의 꼴(타일), 거르기(CL49), Accelerate의 전개식(sklearn의 꼴; Accelerate의 스레드, 1스레드로 묶어도 같았다)은 p = 2·k = 10 1.12 / 0.54, 2.31 / 0.52 대 0.86, p = 10·k = 10 2.30 / 0.80, 3.10 / 0.93 대 1.01, p = 10·k = 100 11.9 / 2.16, 9.95 / 2.24 대 4.79, p = 50·k = 100 42.4 / 7.60, 14.5 / 5.28 대 7.88, p = 100·k = 100 80.8 / 14.6, 21.7 / 8.38 대 10.8, p = 300·k = 100 235 / 42.9, 41.7 / 23.5 대 26.7. 1스레드에서 sd의 빠른 꼴은 Accelerate 전개식의 1.30–2.48배(k = 100에서 거르기가 1.57–2.07배, k = 10에서 차의 꼴이 1.30–3.81배)이고 10스레드에서는 0.38–1.29배다. 1스레드의 1.5배를 넘는 경로는 §9의 39번으로 묻는다. 모든 크기에서 두 꼴의 라벨이 같았다. 맞춤 전체(같은 시작, `tol=1e-4`)의 길별 시간(Lloyd, Lloyd와 거르기, 경계, 경계와 거르기; 1스레드): p = 2·k = 10 53.7, 93.0, 50.1, 64.3, p = 10·k = 10 21.5, 26.3, 18.2, 19.8, p = 10·k = 100 570, 445, 149, 134, p = 50·k = 100 1418, 446, 291, 152, p = 100·k = 100 2790, 557, 513, 227, p = 50·k = 10 231, 141, 90.0, 81.2 ms이고 네 길이 모두 같은 비트다(CL60의 문턱). 열쇠 순서는 10⁵ × 10에서 1.2 ms, 반복 한 번은 5.5 ms다(CL61).
- **CL3 마일스톤의 측정** (밀도; Apple M4 Pro, 부하 평균 2–3, 세 번(sklearn 한 번) 중 가장 빠른 값; "1 / 10"은 sd의 스레드 수, sklearn은 기본 설정; n = 2×10⁴의 표준 정규 점):

  | 사례 | sd 1 / 10스레드 | sklearn |
  |---|---|---|
  | `dbscan` p = 2, eps = 0.05 | 20.7 / 3.8 ms | 34.8 |
  | `dbscan` p = 5, eps = 0.3 | 48.1 / 7.2 | 79.9 |
  | `dbscan` p = 10, eps = 0.9 | 303 / 34 | 476 |
  | `hdbscan` p = 2 | 42.9 / 13.4 | 684 |
  | `hdbscan` p = 5 | 147 / 23 | 897 |
  | `hdbscan` p = 10 | 1365 / 158 | 1488 |
  | `optics` n = 5000, p = 2 | 45.6 / 44.5 | 1370 |
  | `optics` p = 2 | 677 / 218 | (단계마다 Python의 O(n) 고리; 재지 않았다) |

  §5 첫 표의 목표(`dbscan` sklearn 이하 / 1/4 이하, `hdbscan` 1/3 이하 / 1/10 이하)에 이른다. `optics`는 단계마다 처리하지 않은 점 전체를 훑는 O(n²)라 10스레드의 몫이 단계당 갈래의 비용에 묶인다(n = 5000은 단계의 일이 나누는 문턱 아래). 길의 측정(CL72; 1 / 10스레드 ms): DBSCAN의 트리 대 훑기는 p = 2에서 20.7 / 3.8 대 622 / 78, p = 12(n = 2×10⁴, 핵심 점 11194개)에서 2235 / 304 대 3069 / 341, p = 16에서 5018 / 561 대 3876 / 434(훑기가 빠르다). HDBSCAN의 KD 트리(핵심 거리)와 Borůvka 대 훑기와 Prim은 n = 512, p = 2에서 0.5 / 0.4 대 2.1 / 0.9, n = 2×10⁴, p = 12에서 2700 / 321 대 5604 / 708, p = 16에서 6155 / 710 대 6319 / 806, p = 24에서 11399 / 1322 대 7703 / 976(훑기가 빠르다).
- **CL4 마일스톤의 측정** (혼합; Apple M4 Pro, 부하 평균 1.7–2.3, 다섯 번(EM 20번은 세 번) 중 가장 빠른 값; "1 / 10"은 sd의 스레드 수, sklearn은 threadpoolctl의 1스레드 / 기본; 잘 떨어진 가우스 덩어리: `bench/speed/cases/cluster.py`의 `_blobs`와 같은 자료):

  | 사례 (n, p, k) | 꼴 | 맞춤(각자의 k-평균 시작) sd / sklearn | 같은 시작의 EM 20번 sd / sklearn |
  |---|---|---|---|
  | 2×10⁴, 10, 5 | full | 7.1 / 2.2 대 15.0 / 17.0 ms | 37.6 / 6.3 대 82.5 / 84.1 |
  | | tied | 7.2 / 2.4 대 11.9 / 12.9 | 37.8 / 8.1 대 61.9 / 66.9 |
  | | diag | 5.1 / 2.0 대 8.6 / 9.6 | 20.5 / 4.7 대 39.4 / 43.5 |
  | | spherical | 5.8 / 2.1 대 9.1 / 10.1 | 21.2 / 4.7 대 42.6 / 47.2 |
  | 10⁵, 10, 10 | full | 65.7 / 12.9 대 125.5 / 133.5 | 360 / 49.6 대 751 / 759 |
  | | tied | 64.6 / 12.9 대 95.3 / 99.6 | 354 / 51.5 대 549 / 552 |
  | | diag | 44.5 / 11.2 대 56.9 / 57.7 | 184 / 37.7 대 277 / 276 |
  | 2×10⁴, 50, 5 | full | 45.2 / 9.5 대 41.6 / 43.7 | 299 / 46.5 대 244 / 244 |
  | | tied | 43.5 / 10.4 대 25.6 / 27.2 | 285 / 55.5 대 146 / 141 |
  | | diag | 16.1 / 5.6 대 13.3 / 14.0 | 62.6 / 15.6 대 54.8 / 59.9 |
  | 2×10⁵, 3, 8 | full | 125 / 21.4 대 428 / 359 | 257 / 38.0 대 1018 / 1022 |
  | | diag | 93.2 / 19.3 대 249 / 180 | 170 / 37.9 대 476 / 478 |

  §5 첫 표의 목표(full, n = 2×10⁴, p = 10, k = 5: sklearn 이하 / 1/3 이하)에 이른다(속도 사례의 세 번 중 가장 빠른 값 7.2 / 2.2 ms 대 sklearn 16.5 / 16.2). sklearn의 GMM은 numpy(Accelerate) 위에서 스레드 수와 무관하게 거의 같은 시간이다. 1스레드의 p = 50에서 sd가 느리다: 같은 정의(차의 꼴)의 Accelerate 꼴(성분마다 (X − μ)U의 `dgemm`과 행의 제곱합)과 견준 E 단계는 p = 20에서 1.74 대 1.49 ms(1.17배), p = 30에서 2.78 대 1.94(1.43배), p = 50에서 6.3 대 3.7(1.72배), p = 100에서 21.3 대 7.4(2.9배)이고(n = 2×10⁴, k = 5, 1스레드), M 단계의 2차 모멘트는 같은 정의의 Accelerate 꼴((rZ)ᵀZ)과 1.23배 안이다(p = 100에서는 sd가 빠르다: r = 0의 항을 건너뛴다). 10스레드에서는 E 단계도 Accelerate 꼴의 1/3–1/4이다(p = 50에서 0.81 대 3.0 ms). tied의 M 단계는 성분마다 평균을 빼는 두 번째 훑기라 sklearn의 한 번 훑기(XᵀX − Σnkμμᵀ)보다 k배 많은 곱이다(CL75의 정의). 1스레드에서 Accelerate 꼴보다 1.5배 넘게 느린 큰 p의 E 단계는 §9의 40번으로 묻는다. 커널의 판(E 단계 네 열 × 여덟 점, 2차 모멘트 네 행 × 여덟 열)은 스칼라 고리보다 full·tied의 E 단계를 4.6배, 2차 모멘트를 2배 빠르게 했다(비트는 같다).
- **퍼지 c-평균의 멈춤** (CL66): 시험 자료 28개와 무작위 자료 3개에서 `tol=1e-5`가 멈춘 반복은 e1071의 규칙(목적의 상대 변화 < √ε)이 멈출 반복과 같거나 늦었다(27개; 하나는 한 반복 일찍, 셋은 둘 다 `max_iter`=100에 닿았다).
- **CL5·CL6 마일스톤의 측정** (Apple M4 Pro, 부하 평균 1.3–1.8, 세 번 중 가장 빠른 값; `notes/cluster/cl5_time.py`의 기록; sd는 1 / 10스레드, sklearn은 numpy(1스레드)나 OpenMP 기본, R은 1스레드에 자료 옮기기 포함):

  | 사례 | sd 1 / 10스레드 | 비교 대상 |
  |---|---|---|
  | `silhouette_score` n = 2×10⁴, p = 10, k = 5 | 525 / 61 ms | sklearn 1089(1스레드) / 1090, R 1738 |
  | `calinski_harabasz_score` n = 10⁵, p = 10 | 4.2 / 2.9 ms | sklearn 4.6 |
  | `davies_bouldin_score` n = 10⁵, p = 10 | 4.2 / 2.9 ms | sklearn 4.1 |
  | `adjusted_rand_score` n = 10⁶, k = 20 | 8.1 / 4.8 ms | sklearn 63 |
  | `normalized_mutual_info_score` n = 10⁶ | 8.2 / 5.0 ms | sklearn 78 |
  | `adjusted_mutual_info_score` n = 10⁴ / 10⁵ / 10⁶ | 0.3 / 1.6 / 10.3 ms(1스레드) | sklearn 4.7 / 41 / 398 |
  | `gap_statistic` k-평균 n = 1000, p = 4, k_max = 8, B = 50 | 217 / 27 ms | R `clusGap(kmeans)` 985 |
  | `gap_statistic` PAM, 같은 크기 | 7137 / 875 ms | R `clusGap(pam)` 30581 |
  | `gower_distance` n = 3000, 섞인 열 6개 | 51 / 7.0 ms | R `daisy` 159 |
  | `affinity_propagation` n = 2000, p = 2(200번) | 1481 / 210 ms | sklearn 2390 |
  | `affinity_propagation` n = 3000, p = 10, 선호도 −2000 | 961 / 150 ms | sklearn 1291 |

  실루엣은 §5 첫 표의 목표(sklearn 이하 / 1/8 이하)에 이른다. CH·DB의 1스레드는 sklearn과 비슷하고(1.03배 안) 10스레드의 몫이 작다: 열쇠 순서(`lex_order` 1.1 ms)와 Python 층의 검사가 시간의 반이다(CL7 마일스톤의 점검거리). Affinity Propagation의 1스레드는 메모리 대역에 묶인다(§3.12).
- 다른 라이브러리가 틀리는 곳(기록): §2.8의 표. sklearn `silhouette_samples`는 shifted/k3p2에서 0.21, subnormal/k3p2에서 0.75까지 참값과 다르다(전개식 거리). R `silhouette(dist(x))`는 subnormal/k3p2에서 모든 값이 0이다(`dist`의 아래넘침).

---

## 6. 층과 파일
- **층**: `cluster/`는 `array/`, `base/`(`buffer.hpp`, `pairwise.hpp`), `parallel/`, `bind/`, `spatial/`(`metric.hpp`, `kdtree.hpp`, `neighbors.hpp`), `graph/`(합집합–찾기: `csrc/graph/traversal.cpp`의 이름 없는 이름공간에 있던 차례판(뿌리는 집합의 가장 작은 마디)과 원자적 병렬판을 CL3 마일스톤이 `graph/unionfind.hpp`로 꺼냈다; graph의 테스트와 골든이 그대로다. CL1의 Borůvka는 `common.hpp`의 차례판을 쓴다), `stats/sample_dist.hpp`(혼합 모형 표본의 정규 난수; CL4 마일스톤) 위에 있다. 다른 영역은 `cluster/`를 모른다. Python 층의 `stardust.cluster`는 `stardust.random`, `stardust.linalg`(`LinAlgError`), `stardust.spatial`(`squareform`, 거리 이름의 검사), `stardust.frame`, `stardust._summary`, `stardust._calls`를 쓴다.
- 등록: `bench/areas.py`에서 `cluster`는 `smooth` 뒤, `plot` 앞이다(어림). ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표, tests/README.md·bench/README.md의 영역 목록에 한 줄씩 더한다.
- C++ 폴더 `cluster/` (CL0 마일스톤의 시제품 `prototype.cpp`는 CL2 마일스톤이 지웠다; 그 측정 기록은 `tests/cluster/data/prototype.json`에 남는다)
  - `common.hpp`: 점의 사전순 정렬(첫 좌표의 기수 정렬; CL61)과 열쇠, 압축 꼴 색인, CL43의 지수, 가장 작은 원소가 뿌리인 합집합–찾기 (CL1 마일스톤), CL48의 덩이 짝 합(`pairwise_chunks`)·누적 합과 CL41의 원소 번호 (CL2 마일스톤)
  - `partition.hpp` (CL2 마일스톤): 분할 커널의 공용(sqd와 넷을 나란히 재는 `sqd4`, CL42의 여유 `Margin`, 인자 읽기)
  - `hierarchy.cpp` (CL1 마일스톤): 단일 연결의 트리(정렬한 최소 신장 트리와 CL50의 동점), NN-chain(행렬과 무게 중심), 일반 알고리즘(행렬과 무게 중심), 높이 다시 계산(CL45·CL55), 행(CL51), DIANA, `cophenet`, `optimal_leaf_ordering`. 자르기·`inconsistent`·`dendrogram`의 자리 계산은 O(n)이라 Python(`_hierarchy.py`)이다
  - `mst.hpp`·`mst.cpp` (CL1 마일스톤): Prim(점, 행렬)과 KD 트리 위의 Borůvka(점마다; 측정용 이중 트리), 길의 문턱(CL54). CL3 마일스톤이 상호 도달 거리(핵심 거리와 α; Prim은 cityblock·chebyshev도)로 넓혔다(CL67; 핵심 거리 없이는 전과 같은 비트)
  - `kmeans.cpp` (CL2 마일스톤): 훑기(타일, CL49 거르기), Lloyd와 Hamerly 경계, 빈 군집, k-means++·`'random'`, 미니배치, 이분, `predict`, 측정용 `_kmeans_assign_expanded`
  - `kmedoids.cpp` (CL2 마일스톤): BUILD, SWAP(FastPAM1의 재사용), CLARA
  - `fuzzy.cpp` (CL2 마일스톤): 퍼지 c-평균
  - `density.cpp` (CL3 마일스톤): 반경·k 최근접 찾기(KD 트리, 훑기, 행렬), DBSCAN, 핵심 거리, HDBSCAN(Kruskal의 단일 연결 트리, 같은 높이를 한꺼번에 거두는 줄인 트리, 안정성, 고르기, 확률, GLOSH, 지속성, 중심과 medoid), `dbscan_clustering`의 자르기, OPTICS의 차례, ξ·DBSCAN 추출
  - `mixture.cpp` (CL4 마일스톤): EM(네 공분산 꼴; E 단계의 NEON 판과 2차 모멘트의 판), 작은 Cholesky와 그 역, 시작의 책임도, 표본(stats의 `sample_dist.hpp`의 정규 난수)
  - `metrics.cpp` (CL5 마일스톤): 분할표와 짝 수(128비트), MI·엔트로피·기대 MI(초기하 비의 점화식), 실루엣(spatial의 타일 커널)·CH·DB, gap(참조 자료, Jacobi의 축, 벌마다의 군집, log W), Gower의 거리, 공용 `subkey`(CL90)
  - `affinity.cpp` (CL6 마일스톤): Affinity Propagation(행의 A·R 갱신, 열 합, 멈춤, exemplar 다듬기)
  - `cluster.hpp`: 파일마다의 `bind_*`
- 바인딩 `bind/cluster_bind.cpp`: `_core.cluster` 하위 모듈(CL0 마일스톤부터; CL1 마일스톤의 `lex_order`, `linkage`, `diana`, `cophenet`, `optimal_leaf_ordering`, 측정용 `_mst`; CL2 마일스톤의 `kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `bisecting_predict`, `kmeans_predict`, `kmeans_distinct`, `first_seen`, `kmedoids`, `fuzzy_cmeans`, `fuzzy_memberships`, 측정용 `_kmeans_assign_expanded`; CL3 마일스톤의 `dbscan`, `core_distances`, `hdbscan`, `hdbscan_cut`, `hdbscan_centres`, `weighted_medoid`, `optics`, `optics_xi`, `optics_dbscan`; CL4 마일스톤의 `gmm_fit`, `gmm_estimate`, `gmm_sample`, `kmeans_plusplus`(혼합의 시작), 시험용 `gmm_precisions`, `gmm_mstep`; CL5 마일스톤의 `contingency`, `external`, `silhouette`, `ch_db`, `gap`, 시험용 `gap_logw`, `gower`, 그리고 CL4의 `gmm_subkey`를 옮긴 공용 `subkey`(CL90); CL6 마일스톤의 `affinity_propagation`). `partition.hpp`는 gap이 부르는 `kmeans_labels`(kmeans.cpp)와 `pam_labels`(kmedoids.cpp)를 선언한다(공개 함수의 기본값과 같은 비트).
- `src/stardust/cluster/`(하위 패키지 없음): `__init__.py`(`__all__`), `_common.py`(입력 정리, `metric`과 `'precomputed'`, `nan_policy`, `rng`의 검사, 메시지), `_results.py`(공통 기반 `ClusterResult`와 `summary`; CL44), `_hierarchy.py`, `_kmeans.py`, `_kmedoids.py`, `_fuzzy.py`, `_density.py`, `_mixture.py`, `_metrics.py`, `_gap.py`, `_gower.py`, `_affinity.py`.

---

## 7. 참값과 검증
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다. 참값 생성은 프로세스 10개까지, 워커를 정한 태스크 수마다 바꾸고 RSS가 평평한지 지켜보며 돈다.

### 7.1 시험 자료 (`tests/cluster/samples.py`)
공식과 고정 시드로 만든다(내려받은 것 없음).
- **덩어리**: 잘 떨어진 가우스 덩어리(k = 2–10, p = 1–10), 겹치는 덩어리, 크기가 크게 다른 덩어리, 길쭉한 덩어리(공분산의 조건수 10⁴), 밀도가 다른 덩어리(HDBSCAN), 고리·초승달(밀도 방법이 맞고 k-평균이 틀리는 것).
- **동점**: 정수 격자(1–3차원), 한 줄의 고른 점, 같은 점이 여러 번(모두 같은 점 포함), 정다각형의 꼭짓점과 중심(같은 거리의 중심), 거리가 꼭 eps인 짝.
- **퇴화**: n = 1, 2, 3, k = n, k > 서로 다른 점의 수, 한 차원에 놓인 점(공분산 특이), 상수 열.
- **척도**: 아주 좁은 범위(10⁻⁸), 큰 평행 이동(10⁸ + 덩어리; 전개식 거리의 상쇄), 큰 척도(10¹⁵⁰), 비정상 수 간격.
- **역전**: centroid·median이 역전하는 삼각형과 그 변형.
- **라벨 짝**(지표): 같은 분할, 한 군집, 모두 홑원소, 무작위 분할(기대값 근처), 큰 n(10⁶)의 셈.

### 7.2 참값 (`tools/cluster_ref.py` → `tests/cluster/ref/`, `tests/reffile.py`의 npz)
CL0 마일스톤의 여섯 묶음이다(점은 시험 자료의 id와 종류·번호로 찾는다; 모두 합쳐 0.5 MB, 6개 프로세스로 14초, 워커 RSS 0.09 GB):
- `hierarchy`: 160점 이하의 집합 37개 × 일곱 연결법. 정의(§3.2, CL17)의 트리를 정확한 산술로: single·complete·centroid·median·ward는 제곱 거리의 유리수(ward·centroid·median은 Lance–Williams의 유리식), average·weighted는 √q의 유리 계수 합(같은 꼴이면 같고, 아니면 Arb로 4096비트까지 견준다). 병합마다 높이(Arb의 제곱근), Z의 행, 다음 후보와의 상대 간격 `gap`(0은 정확한 동점).
- `kmeans`: 33개 집합. 정해진 시작에서 정확한 산술의 Lloyd(같은 거리는 작은 중심, 빈 군집은 가장 먼 점; 300번까지)의 라벨·중심·관성과 반복 수, 걸음마다의 가장 작은 간격 `path_gap`; 그 중심을 반올림한 점에서의 정확한 라벨과 관성; 참 라벨의 정확한 무게 평균.
- `kmedoids`: 160점 이하 집합의 k = 2, 3(과 제안 k) 54가지. Arb의 거리로 PAM의 BUILD·SWAP 경로(R `pam`의 단계, 동점은 열쇠)와 두 목적, 20점 이하는 모든 k-집합의 최적과 그 동점 수.
- `fuzzy` (CL2 마일스톤): 300점 이하 집합 31개의 주어진 중심(k-평균 시작의 중점, 반올림)에서 m = 2, 1.5의 소속도(정확한 유리수)와 m = 3(Arb), m = 2의 한 걸음 중심과 목적(유리수).
- `density`: DBSCAN 24가지(정확한 d² ≤ eps², 무게, 경계 점은 가장 가까운 핵심 점, 핵심·eps 근처 표시)와 HDBSCAN의 14벌(min_samples = 5의 핵심 거리, 상호 도달 거리의 정확한 MST). CL3 마일스톤이 같은 14벌에 더했다: HDBSCAN의 정의(CL67–CL69; `min_cluster_size` = 5)를 정확한 산술로(높이는 정확한 제곱 상호 도달 거리, λ = 1/√w²는 Arb) 푼 `'eom'`·`'leaf'`의 라벨, 확률, GLOSH, 줄인 군집마다의 안정성(계약의 S⁺와 행 수), 고르기 비교의 가장 작은 상대 간격(`selection_gap`); OPTICS(min_samples = 5, max_eps = inf)의 정확한 차례(단계마다 가까운 동점 표시)와 도달 거리·선행자. 38 태스크, 6 프로세스로 1초 아래, 워커 RSS 0.02 GB.
- `mixture`: 네 공분산 꼴의 6가지 혼합에서 점마다 로그 밀도·성분 밀도·책임도(정확한 역행렬과 행렬식, Arb의 log·exp)와 반올림한 책임도의 M 단계(유리수, `reg_covar` = 10⁻⁶). CL4 마일스톤이 더하고 고쳤다: 주어진 인자 U_f(sklearn `precisions_cholesky_`의 double; 그룹 입력 `prec_chol`)에서의 같은 값(`score_u`, `logdens_u_<j>`, `resp_u_<j>`: 이차형식은 정확한 유리수, log det U_f는 Arb), 정확한 공분산의 정확한 인자 U*(`prec_chol`, Arb의 맨 아래 행부터의 Cholesky), M 단계는 CL75의 정의(nk = Σr + 10ε, 무게 nk/Σnk). 6 태스크, 1초, 워커 RSS 0.2 GB.
- `metrics`: 라벨 짝 13가지의 ARI·RI(정확한 유리수; sklearn의 특수 경우), FM·MI·NMI 넷·AMI 넷(EMI는 Arb의 초기하 합; 10⁵점 넘는 짝은 뺀다)·동질성·완전성·V, 참 라벨이 있는 점 집합 18개의 실루엣(점마다와 평균)·Calinski–Harabasz(유리수)·Davies–Bouldin. CL5 마일스톤이 더했다: 엔트로피 `h_true`·`h_pred`, 기대 MI `emi`(필드 `plus`: §4.6의 E⁺; 라벨링마다 한 번 재어 AMI 넷이 함께 쓴다), AMI의 한쪽이 모두 홑원소인 경우(0, `'min'`은 1; CL84), 참 분할의 gap log W `logw_1`·`logw_2`(R `W.k`의 d.power 1·2; Arb의 제곱근 합과 정확한 유리수). 31 태스크, 6 프로세스로 1초 아래, 워커 RSS 0.08 GB.
- mpmath는 쓰지 않는다.

### 7.3 테스트 (`tests/cluster/`)
- 파일(어림; CL3 마일스톤까지 있는 것: `test_dbscan.py`, `test_hdbscan.py`, `test_optics.py`가 참값, sklearn과의 견줌, 길·스레드·입력 순서, 결과, 에러를 함수마다 보고, CL4 마일스톤의 `test_mixture.py`가 주어진 인자·공분산·책임도의 계약, sklearn의 고정점, 멈춤, 시작과 `rng`, 스레드·입력 순서, 표본, NaN, 척도, 에러를 본다): `test_api_surface.py`, `test_reference_data.py`, `test_results.py`, `test_hierarchy.py`, `test_kmeans.py`, `test_kmedoids.py`, `test_fuzzy.py`, `test_dbscan.py`, `test_hdbscan.py`, `test_optics.py`, `test_mixture.py`, `test_metrics.py`, `test_gap.py`, `test_gower.py`, `test_affinity.py`(CL5·CL6 마일스톤: 참값의 비트와 계약, sklearn·R의 값, 라벨의 꼴, 길·스레드·입력 순서, 잡음이 0인 sklearn의 AP, 에러), 그리고 `conftest.py`, `samples.py`, `refdata.py`(참값 읽기와 계약의 한계 함수).
- 성질: 1·3·10스레드의 같은 바이트, 입력 순서를 섞어도 같은 분할(결정적인 방법; §9의 4번), 두 길의 같은 비트(NN-chain과 일반, 행렬 길과 행렬 없는 길, Prim과 Borůvka, Lloyd와 경계 가속, KD 트리와 훑기), 같은 `rng` 상태의 같은 비트와 `rng`가 두 단어만 움직이는 것, 2의 거듭제곱 척도의 같은 분할(같은 비트로 옮겨 가는 높이·중심).
- 비교 대상과는 그 자리에서 견준다: scipy `linkage`(동점 없는 자료에서 같은 트리, 높이 계약), R `hclust`(이름 대응과 눈금), sklearn `KMeans`(같은 시작에서 같은 고정점), R `pam`(같은 medoid; 동점 없는 자료), sklearn `DBSCAN`(경계 점 밖에서 같은 분할), sklearn `HDBSCAN`·`OPTICS`(같은 라벨; 동점 없는 자료), sklearn `GaussianMixture`(같은 시작에서 같은 고정점의 근처), sklearn·R의 지표 값.

### 7.4 비교 대상의 기록 (`tools/cluster_probe.py` → `tests/cluster/data/`)
- 설계 때의 탐침(`site/notes/cluster-design/probe_behavior.py`, `probe_signatures.py`, `probe_r_formals.R`)을 옮겨 넓혔다(CL0 마일스톤): 동점의 트리, 연결법의 눈금, 역전 자르기, 잎의 순서, k-평균의 기본값·동점·빈 군집·퇴화·스레드 의존, DBSCAN의 경계 점, HDBSCAN의 같은 점, 혼합 모형의 멈춤, 지표의 특수 경우, R `cluster`의 함수들, 시간.
- 기록: `scipy.json`, `sklearn.json`, `r.json`(R은 `bench/accuracy/rsession.py`의 세션으로 부른다). 문서의 `[… 와 다름]`은 `rule/…` 항목이나 같은 자료의 기록이 받친다. CL5 마일스톤이 `r.json`에 더했다(앞의 기록은 그대로): 참 라벨 집합의 `silhouette/…`, `clusgap_pam/<집합>/<d.power>/<spaceH0>`(log W, E.logW, SE.sim; B = 100), `cluster/maxse`(무작위 60가지 × 규칙 다섯 × SE 계수 둘), `cluster/daisy_mixed`·`daisy_default`·`daisy_numeric`.
- 비교 대상이 없는 것: mclust의 공분산 꼴(나중), R `dbscan`·원본 `hdbscan`(sklearn만), `e1071::cmeans`(퍼지 c-평균은 참값으로만), `apcluster`(Affinity Propagation은 sklearn만: 테스트가 잡음이 0인 `random_state`로 sklearn의 `_affinity_propagation`을 그 자리에서 부른다), scikit-learn-extra의 `KMedoids`(R `pam`만).

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/cluster.py`, 표는 `bench/cluster/accuracy.md`): 스윕 `hierarchy`(CL1 마일스톤, 기준선: 높이의 ULP와 계약의 몫, 참값과 다른 트리의 수; scipy·R 열), `kmeans`(중심, 관성, 고정점), `kmedoids`, `density`(CL3 마일스톤: DBSCAN의 분할, HDBSCAN의 핵심 거리·트리·안정성·확률·GLOSH·라벨, OPTICS의 도달 거리와 차례; sklearn 열), `mixture`(CL4 마일스톤: 주어진 인자의 성분·혼합 로그 밀도와 책임도, 공분산의 인자, 주어진 책임도의 M 단계; sklearn 열), `metrics`(CL5 마일스톤: ARI·RI·FM의 ULP, MI·엔트로피·기대 MI·NMI·AMI·동질성·완전성·V, 실루엣·CH·DB, gap의 log W를 §4.6의 한계 단위로; sklearn 열). `--compare`가 비교 패키지의 같은 양을 같은 한계로 잰다.
- **속도** (`bench/speed/cases/cluster.py`): §5의 사례와 크기, 1·10스레드. `--smoke`에서 결과를 비교 대상과 견준다.
- **골든** (`tests/golden/cases_cluster.py`, 기록은 `tests/golden/data/cluster.json`): 모든 공개 함수 × 시험 자료의 대표, 옵션의 값마다, 틀린 호출, 병렬 길을 탈 큰 케이스(워커 하나 1 GB 아래). 1·3·10스레드의 결과가 모두 같아야 한다.

---

## 8. 마일스톤
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| CL0 준비 | API 문서(`docs/api/cluster.md`)를 시제품의 측정에 맞춰 고치기, 영역 등록(`bench/areas.py`, ARCHITECTURE, API.md의 색인), 시험 자료(§7.1), 참값 도구(§7.2), 탐침 도구와 기록(§7.4), 결과의 공통 기반과 `summary`, 시제품의 측정(k-평균의 거리 꼴, 경계 가속의 여유, Prim 대 Borůvka, NN-chain의 동점 검출, 높이 다시 계산의 비용과 정확도), 속도 사례와 골든의 틀 | 문서와 도구가 커밋되고 탐침 기록이 있다 |
| CL1 계층 | `linkage`, `agglomerative`, `divisive`, `HierarchyResult`, scipy의 도구(§9의 17번), `dendrogram`의 자료, 행렬 없는 길 | 트리와 높이의 계약, scipy·R과의 차이가 문서대로, 두 길의 같은 비트, 스레드마다 같은 바이트, 속도 사례 — **마침**(2026-11-27; CL50–CL57) |
| CL2 분할 | `kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `kmedoids`(PAM·CLARA), `fuzzy_cmeans`와 결과들 | 고정점의 계약, sklearn·R과 같은 시작에서 같은 답, 경계 가속이 Lloyd와 같은 비트, `rng` 규칙, 속도 사례 — **마침**(2026-11-27; CL59–CL66) |
| CL3 밀도 | `dbscan`, `hdbscan`, `optics`와 결과들 | 정의대로의 분할(참값), sklearn과의 차이가 문서대로(경계 점), KD 트리와 훑기의 같은 비트, 속도 사례 — **마침**(2026-11-27; CL67–CL74) |
| CL4 혼합 | `gaussian_mixture`와 결과(네 공분산 꼴, `bic`·`aic`, `sample`) | 주어진 모수·책임도의 계약, sklearn과 같은 시작에서 같은 고정점 근처, 속도 사례 — **마침**(2026-11-27; CL75–CL81) |
| CL5 지표와 k 고르기 | 내부·외부 지표, `gap_statistic`, `gower_distance` | 지표의 계약(ARI 등은 올바른 반올림), sklearn·R과 같은 값, 속도 사례 — **마침**(2026-11-27; CL82–CL90) |
| CL6 그 밖 | `affinity_propagation`(CL40) | 정의와 멈춤 규칙이 문서대로, sklearn과 같은 답(잡음 없이) — **마침**(2026-11-27; CL91–CL94) |
| CL7 성능과 일관성 점검 | 모든 공개 함수를 크기마다 비교 대상과 견주어 고치기(CL2 마일스톤에서 넘긴 것: k-평균 배정의 1스레드 커널(§9의 39번), Elkan 경계의 측정(CL60), `bisecting_kmeans`의 1스레드(sklearn 31 대 40.8 ms); CL1 마일스톤에서 넘긴 것: 10스레드 목표에 못 미친 `linkage`의 complete·average·weighted·centroid·median·ward, n = 5000에서 70–108 ms 대 목표 scipy의 1/3 ≈ 50 ms; NN-chain과 일반 알고리즘의 차례 부분), 갈래의 문턱(Prim·Borůvka, 경계 가속, 트리와 훑기), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다 |
| CL8 문서 | `docs/api/cluster.md`의 확정, 이 문서를 구현대로 고치기, 등록 문서의 확인 | 문서가 코드와 맞고 `test_api_surface`가 통과한다 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 정하고, 구현하고, 정확도·속도를 재고, 커밋한다. 마일스톤 에이전트는 한 번에 하나다(C++ 재빌드가 모든 import를 멈춘다). CL1–CL5 마일스톤은 CL0 마일스톤에만 기대어 대체로 독립이지만, CL3 마일스톤의 HDBSCAN은 CL1 마일스톤의 MST(Prim·Borůvka)와 Z 꼴을, CL4 마일스톤의 시작은 CL2 마일스톤의 `kmeans`를, CL5 마일스톤의 gap은 CL2 마일스톤의 `kmeans`·`kmedoids`를 쓴다. 그래서 순서는 CL1 → CL2 → CL3 → CL4 → CL5 → CL6이다. 기존 영역의 결과는 비트까지 같아야 한다(이 영역은 다른 영역의 커널을 부르기만 한다; graph의 합집합–찾기를 머리 파일로 꺼내면 graph의 골든으로 확인한다).

**나중 후보** (첫 판 밖): 연결 제약이 있는 계층 군집(sklearn `connectivity`)과 `FeatureAgglomeration`, mclust의 14 공분산 꼴과 BIC 고르기, 베이즈 가우스 혼합, Mean Shift, 스펙트럼 군집(`sd.embed` 뒤), Birch, FasterPAM, `fanny`, R `dbscan`의 `extractFOSC`·`glosh` 계열의 더한 것, 군집 안정성(부트스트랩 Jaccard; fpc `clusterboot`), 2차원 Delaunay 단일 연결.

---

## 9. 마일스톤을 시작하기 전에 물을 것
모두 (a)로 정했다(38번은 CL0 마일스톤에서 생긴 질문이고 사용자가 (a)로 정했다, CL49, 2026-11-26; 39번(CL2 마일스톤)과 40번(CL4 마일스톤)은 답을 기다린다)(§1.1 CL4–CL40; 3·6·10번과 범위의 19·22·27·28·29·32·36·37번은 사용자 결정 2026-11-24, 나머지는 추천안, 사용자 위임 2026-11-24; 13번은 (b): 메인 결정, 사용자 위임 2026-11-24). 아래는 물었을 때의 기록이다.

추천을 첫째에 둔다.

**CL0 마일스톤 전**
1. **결과의 꼴과 계층 군집의 두 얼굴** (CL4; §2.1). (a) 맞추는 함수는 `<이름>Result`를 돌려주고(CL1), 계층은 두 얼굴이다: scipy 이름의 배열 함수(`linkage` → Z, `fcluster`, `cut_tree`, `cophenet`, `leaves_list` 등)와 결과 객체 `agglomerative(X, *, method)` → `HierarchyResult`(`.linkage`가 같은 Z, `.cut(k=, height=)`). 결과는 자료의 사본을 쥐지 않고, 자료가 필요한 게으른 필드(`HDBSCANResult.medoids`)는 자료의 읽기 전용 참조를 쥔다. 혼합 모형의 점마다 로그 밀도는 sklearn의 이름 `score_samples`다. 추천: 평활의 `make_smoothing_spline` / `smoothing_spline`과 같은 두 얼굴이라 scipy 코드가 그대로 옮겨 오고 결과 객체의 꼴(CL1)도 선다. (b) 결과 객체만: scipy의 Z를 다루는 코드가 옮겨 오지 못한다. (c) scipy 이름의 배열 함수만: CL1과 어긋난다.
2. **군집 수와 자료의 인자** (CL5; §2.1). (a) `X`가 첫째 위치 인자((n, p), 점이 행), 군집 수는 모든 분할·혼합 방법에서 둘째 위치 인자 `k`(혼합 모형도 `n_components`가 아니라 `k`)다. 추천: 영역 안에서 한 이름이고 CL1의 `kmeans(X, k)` 그대로이며 scipy `kmeans2(data, k)`와 같다. sklearn 코드는 `n_clusters=`를 `k`로 바꾼다(대응표). (b) sklearn의 이름(`n_clusters`, `n_components`)을 키워드로: sklearn과 같지만 영역 안에서 두 이름이고 CL1의 꼴과 다르다.
3. **라벨의 번호** (CL6; §2.1). (a) int64, 군집은 0 … K − 1, 잡음(밀도 방법)은 −1이고, 번호는 입력 순서로 처음 나오는 차례다(첫 점의 군집이 0). 중심·medoid·성분의 순서가 라벨을 따른다. 추천: 라벨이 분할만의 함수가 되어(알고리즘·시작·스레드와 무관) 두 결과를 견주기 쉽고, 결정적인 방법은 입력을 섞어도 섞은 순서의 같은 라벨이 나온다. (b) 알고리즘의 순서(sklearn: k-평균은 시작 중심의 순서, DBSCAN은 펼친 순서): sklearn과 같은 라벨이지만 같은 분할이 다른 라벨이 된다. (c) 크기가 큰 군집부터: 보기 좋지만 크기의 동점 규칙이 또 필요하다.
4. **입력 순서와 무관한 분할** (CL7; §1.3). (a) 안에서 점을 좌표의 사전순(같으면 입력 색인)으로 정렬한 순서를 열쇠로 쓰고(smooth SM36과 같다), 모든 동점 규칙(계층의 병합, DBSCAN의 경계 점, OPTICS의 순서, PAM의 교환, 빈 군집의 옮기기)이 이 열쇠를 쓴다. 난수의 표본도 이 순서의 번호로 뽑는다. 그래서 같은 점들을 어떤 순서로 주든 같은 분할(과 (a)의 라벨 규칙으로 섞은 순서의 같은 라벨)이다. 미리 계산한 거리는 좌표가 없어 입력 색인이 열쇠다. 비용은 O(n log n·p)의 정렬 한 번(색인 배열; 자료를 옮기지 않는다). 추천: scipy와 sklearn은 같은 자료를 섞으면 다른 트리·라벨을 낸다(탐침). (b) 입력 색인을 열쇠로(scipy·sklearn처럼): 정렬이 없지만 순서에 따라 답이 바뀐다.
5. **난수 (`rng`)** (CL8; §1.4, §3.10). (a) 무작위를 쓰는 함수(`kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `kmedoids(method='clara')`, `fuzzy_cmeans`, `gaussian_mixture`, `gap_statistic`)는 `rng`(`sd.random.Generator`)가 키워드 전용이고 기본값이 없다. `init`에 배열을 주어 무작위가 필요 없어도 받는다(한 규칙; 두 단어를 움직인다). 받지 않는 방법(`kmedoids(method='pam')`)에 주면 `ValueError`다. 호출마다 64비트 두 단어를 받아 Philox의 키로 쓰고 (시작, 단계, 자리)의 원소 번호로 뽑는다. 추천: 회귀 `rlm(method='MM')`·`lasso_cv`와 같은 규칙이고 `n_init`의 시작들을 나란히 돌려도 스레드와 무관하다. (b) sklearn처럼 정수 `random_state`: 다른 영역과 어긋난다. (c) `rng`를 차례로 소비(시작마다 이어서 뽑기): sklearn의 흐름과 닮지만 시작들을 나란히 돌리면 결과가 스레드에 달린다.
6. **결정성의 범위와 BLAS** (CL9; §3.10). (a) 모든 경로를 고정 순서의 자체 커널(NEON)로 계산해 1·3·10스레드에서 같은 바이트다. 혼합 모형과 k-평균의 큰 곱도 Accelerate를 쓰지 않는다. CL0 마일스톤의 시제품에서 자체 커널이 Accelerate 꼴보다 1.5배 넘게 느리면 그 경로만 다시 묻는다. 추천: 새 영역이 모두 지켜 온 관례이고, 군집은 반올림 차이가 다른 라벨로 번진다(sklearn의 스레드 의존: 탐침에서 중심 6.4×10⁻¹⁵). (b) 큰 곱은 Accelerate이고 같은 스레드 수의 같은 비트만 약속(API.md §3이 허용하는 최소): 큰 p와 k에서 빠를 수 있지만 스레드 수와 Accelerate의 모드가 라벨을 바꿀 수 있다.
7. **k-평균의 거리 계산** (CL10; §3.1). (a) 차의 제곱합 Σ(x − c)²(NEON). 추천: 상쇄가 없어 먼 평행 이동(10⁸ + 덩어리)에서도 라벨이 정확하다. sklearn의 전개식은 spatial의 탐침에서 10¹⁸ ULP 틀렸다. 곱셈이 p번 많은 값은 CL0 마일스톤에서 잰다. (b) sklearn처럼 ‖x‖² − 2x·c + ‖c‖²(행렬 곱): 큰 p에서 빠르지만 상쇄가 있다. (c) 전개식으로 거르고 차의 꼴로 다시 재기: 같은 비트이고 빠를 수 있지만 거르기의 여유를 증명해야 한다(CL7 마일스톤의 후보로 둔다).
8. **NaN** (CL11). (a) `nan_policy='raise'|'omit'`, 기본 `'raise'`(`ValueError`). `'omit'`은 NaN이 있는 행을 빼고 맞추며 그 행의 라벨은 −1(잡음과 같은 "군집 없음"), `nobs`는 뺀 뒤의 수, `omitted`(bool 배열)가 뺀 행을 보인다. ±inf는 언제나 `ValueError`다. 추천: sklearn은 NaN이 에러이고 회귀·평활의 이름을 쓰되, 라벨은 NaN이 될 수 없어 `'propagate'`는 두지 않는다. (b) `nan_policy` 없이 언제나 `ValueError`(sklearn과 같다): 가장 작지만 다른 영역과 다르다. (c) `'omit'`의 라벨을 −2로 잡음과 가르기: 새 표식이 생긴다.
9. **거리 입력** (CL12; §3.1). (a) 관측값이 기본이고, 거리로 정의되는 방법(`linkage`, `agglomerative`, `divisive`, `kmedoids`, `dbscan`, `hdbscan`, `optics`, `silhouette_*`, `affinity_propagation`)은 `metric='precomputed'`로 (n, n) 정사각 또는 1차원 압축 꼴을 받는다. scipy 이름의 `linkage`만 scipy처럼 1차원 입력을 압축 꼴로 읽는다(2차원은 언제나 관측값이고 scipy의 "거리 행렬처럼 보인다" 경고는 없다). `metric`의 이름은 spatial의 것이다. 추천: 모양으로 뜻을 짐작하지 않는다. (b) sklearn처럼 정사각만: R `dist`·scipy `pdist`의 압축 꼴이 바로 들어가지 못한다.
10. **지표의 자리** (CL13; §1.4). (a) 군집 지표(내부·외부, gap)는 `sd.cluster`에 둔다. 나중의 공통 지표 영역(`sd.metrics` 후보)에는 분류·회귀의 점수를 두고 군집 지표의 별칭은 두지 않는다. 나중의 CV·튜닝 도구는 `sd.cluster.adjusted_rand_score`처럼 함수로 받는다. 추천: 군집 지표는 군집에서만 쓰이고(외부 지표는 두 분할의 비교), sklearn도 `metrics.cluster`로 따로 묶는다. 영역이 생기기 전에 그 영역의 꼴을 정하지 않는다. (b) 지금 `sd.metrics`를 만들어 군집 지표만 넣는다: 나중의 분류 지표와 한곳이지만 지금은 내용이 군집뿐인 영역이다. (c) 지금은 `sd.cluster`에 두고 `sd.metrics`가 생기면 옮긴다: 사용자 코드가 깨진다.
11. **지표의 이름** (CL14; §2.6). (a) sklearn의 이름 그대로(`adjusted_rand_score`, `silhouette_samples`, `calinski_harabasz_score` 등)와 인자 이름(`labels_true`, `labels_pred`, `average_method`). 위치 인자다. 추천: 옮겨 오는 코드가 이름만 바꾸지 않고 그대로 돈다(smooth의 scipy 이름과 같은 원칙). (b) 짧은 이름(`ari`, `ami`, `silhouette`): 짧지만 대응표를 늘 봐야 한다.

**CL1 마일스톤 전**
12. **연결법과 입력 규칙** (CL15; §2.2). (a) scipy의 일곱 이름과 높이의 뜻(ward = R `ward.D2`, centroid·median은 유클리드 거리를 받아 안에서 제곱), R `ward.D`는 두지 않고 대응표에 환산을 적는다(R `ward.D`에 제곱 거리를 주면 `ward`와 같은 트리, 높이는 제곱). 관측값과 비유클리드 `metric`의 ward·centroid·median은 `ValueError`(scipy와 같다), 미리 계산한 거리는 받고 "유클리드 거리를 가정한다"고 문서에 적는다. 추천: scipy가 기준이고 탐침으로 눈금이 확인됐다. (b) R의 여덟 이름(`ward.D` 포함)도 받는다: 같은 뜻의 이름이 둘이 된다. (c) 미리 계산한 거리의 ward·centroid·median을 막는다: 유클리드 거리 행렬을 가진 사용자도 막힌다.
13. **연결법의 기본값** (CL16). (a) 기본값이 없다(`method`는 꼭 준다). 추천: 세 기준이 다 다르다(scipy `'single'`, R `'complete'`, sklearn `'ward'`). 어느 쪽을 고르든 다른 쪽에서 옮겨 온 코드가 조용히 다른 트리를 낸다. scipy 코드의 `linkage(y)`는 `TypeError`로 알려 준다(`[scipy와 다름]`). (b) scipy의 `'single'`: 기준 라이브러리와 같다(API.md §4의 규칙 2). (c) `'ward'`: 가장 많이 쓰이지만 관측값만 받는다. → **(b)로 정했다**(CL16): API.md §4의 규칙 2(기본값은 기준 라이브러리를 따른다)와 맞춘다. 다른 라이브러리에서 옮겨 오는 코드의 차이는 대응표에 적는다.
14. **계층 군집의 동점과 높이** (CL17; §3.2). (a) 트리는 primitive 정의((높이, 열쇠)가 가장 작은 쌍부터)이고, NN-chain은 (거리, 열쇠)로 이웃을 고르며, 계산한 높이에 정확히 같은 값이 나오면 그 입력만 일반 알고리즘으로 다시 푼다. 출력 높이는 최종 트리에서 정의대로 다시 계산하고(고정 순서의 합), 단조인 연결법에서 끝자리만큼 줄어든 자리는 앞 높이로 맞춘다. 추천: 트리와 높이가 알고리즘과 무관하게 정의로 적히고, average는 Lance–Williams의 누적 반올림보다 정확하다. 다시 계산은 O(n²)로 행렬 만들기와 같은 크기다. (b) NN-chain과 (거리, 열쇠)의 이웃 고르기를 정의로: 싸지만 동점에서 primitive와 다른 트리가 될 수 있고 그 트리를 말로 적기 어렵다. (c) 높이는 Lance–Williams 그대로(scipy와 같은 값): 다시 계산이 없지만 높이가 알고리즘의 경로에 달린다.
15. **행렬 없는 계층 군집** (CL18; §3.2, §3.11). (a) 관측값과 유클리드의 single(Prim, 낮은 차원은 KD 트리 Borůvka), ward·centroid·median(무게 중심 길)은 n × n 행렬 없이 O(np) 메모리로 푼다. complete·average·weighted는 행렬이고 물리 메모리보다 크면 `MemoryError`다. 같은 입력의 행렬 길과 같은 비트다. 추천: scipy가 못 푸는 n = 10⁵ 이상을 푼다(행렬 40 GB). (b) 언제나 행렬(scipy와 같다): 단순하지만 n이 2–3×10⁴에서 막힌다.
16. **역전이 있는 트리의 자르기** (CL19; §2.2). (a) k로 자르기(`cut(k=)`, `cut_tree(n_clusters=)`, `fcluster(criterion='maxclust')`)는 병합 순서로 마지막 k − 1개의 병합을 되돌린 분할이다(R `cutree(k=)`와 같다; 단조 트리에서는 scipy와 같다). 높이로 자르기(`cut(height=)`, `cut_tree(height=)`, `fcluster(criterion='distance')`)는 역전 트리에서 `ValueError`다(R과 같다). `fcluster(maxclust)`의 동점 규칙("t개 이하", 한 줄 네 점의 k = 2가 한 군집)은 단조 트리에서 scipy 그대로다. 추천: 탐침에서 scipy의 세 자르기가 역전 트리에서 서로 다른 답을 내고 `cut_tree`는 빈 라벨 번호를 낸다(`[scipy와 다름]`). (b) scipy의 동작을 따라 한다: 정의할 수 없는 동작을 옮긴다. (c) 높이로 자를 때 역전을 단조로 고친 높이(누적 최댓값)를 쓴다: 에러가 없지만 정의가 하나 더 생긴다.
17. **scipy 계층 도구의 범위** (CL20). (a) `linkage`, `fcluster`, `cut_tree`, `cophenet`, `leaves_list`, `optimal_leaf_ordering`, `dendrogram`(자료만), `inconsistent`, `maxdists`, `maxinconsts`, `maxRstat`, `leaders`, `is_monotonic`, `is_valid_linkage`, `num_obs_linkage`, `is_isomorphic`. 빼는 것: `to_tree`·`ClusterNode`(Python 객체 트리), `from_mlab_linkage`·`to_mlab_linkage`, `correspond`, `set_link_color_palette`, `DisjointSet`, 줄임 함수(`single(y)` 등), `fclusterdata`, `scipy.cluster.vq` 전부(`kmeans`가 맡는다). `cophenet(Z, y=None)`의 반환 꼴이 인자에 따라 바뀌는 것은 `cophenet(Z)`(거리)와 `cophenet_correlation(Z, y)`(상관과 거리)로 나눈다. 추천: 쓰이는 것은 두고 Python 객체·옛 형식·그림 상태는 뺀다. (b) scipy 전부: 이름은 같지만 sd에 없는 개념(전역 색 팔레트, Python 노드 객체)이 들어온다. (c) `linkage`와 자르기만: scipy 코드의 `cophenet`·`dendrogram`이 옮겨 오지 못한다.
18. **덴드로그램** (CL21; §1.4). (a) `dendrogram(Z, *, p=30, truncate_mode=None, color_threshold=None, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True)`은 scipy의 `no_plot=True` 결과와 같은 내용의 `DendrogramResult`를 돌려주고 그림은 그리지 않는다. 색은 묶음 번호다. 그림(`sd.plot.dendrogram`)은 나중에 plot 영역이 이 자료를 받아 만든다. 추천: 이 영역이 plot을 고치지 않고(그림 라이브러리는 따로 만든다), 자리 계산은 한곳이다. (b) 이 영역의 CL1 마일스톤에서 `sd.plot.dendrogram`도 만든다: 한 번에 끝나지만 plot의 설계 결정(PL)과 겹친다. (c) `dendrogram`을 두지 않는다: scipy의 `dendrogram(no_plot=True)` 코드가 옮겨 오지 못한다.
19. **분할식 계층과 연결 제약** (CL22; §2.2). (a) R `diana`를 `divisive(X, *, metric)`로 넣고, `agnes`의 응집 계수와 `diana`의 분할 계수는 `HierarchyResult.coefficient`다. sklearn의 연결 제약(`connectivity`: 희소 그래프의 이웃끼리만 합치기)과 `FeatureAgglomeration`은 나중이다. 추천: DIANA는 R에만 있는 고유한 방법이고 작다. 연결 제약은 희소 그래프 위의 다른 알고리즘(sklearn `ward_tree`의 힙)이라 일이 크다. (b) 연결 제약도 CL1 마일스톤에: 영상 분할 같은 쓰임에 필요하지만 범위가 커진다. (c) DIANA를 뺀다.

**CL2 마일스톤 전**
20. **k-평균의 알고리즘** (CL23; §3.3). (a) Lloyd 하나가 정의이고, Hamerly·Elkan의 경계 가속은 반올림 여유를 둔 증명된 건너뛰기라 Lloyd와 같은 비트다. `algorithm` 인자는 없다(길은 모양으로만). R의 Hartigan–Wong은 두지 않고 대응표에 적는다. 추천: sklearn의 기본(`'lloyd'`)과 같은 고정점을 내고, 가속이 답을 바꾸지 않는다. (b) Hartigan–Wong도 `method='hartigan'`으로: R의 기본이고 국소 최소가 더 좋을 때가 있지만(점 하나씩 옮긴다) 차례 알고리즘이라 병렬이 어렵다. 탐침의 자료에서는 같은 관성이었다.
21. **k-평균의 초기화·반복·빈 군집** (CL24; §2.3, §3.3). (a) sklearn과 같다: `init='k-means++'`(탐욕, 후보 2 + ⌊ln k⌋개), `'random'`, 배열; `n_init='auto'`(k-means++와 배열은 1, `'random'`은 10); `max_iter=300`, `tol=1e-4`(축마다 분산의 평균에 곱한다), 엄격한 수렴, 멈춘 뒤 라벨 다시 매기기; 빈 군집은 가장 먼 점으로 옮기기; 같은 거리는 작은 중심 번호. 서로 다른 점이 k보다 적으면 `ValueError`(`kmeans(): 'k' (3) is larger than the number of distinct points (2)`). 추천: 기준 라이브러리의 뜻이고 탐침으로 확인했다. sklearn은 같은 경우에 경고와 적은 라벨이다(`[sklearn과 다름]`). (b) R의 기본(`nstart=1`, `iter.max=10`): 10번은 큰 자료에서 수렴 전에 멈춘다. (c) `n_init=10`을 언제나(sklearn 1.4 전): 10배 느리다.
22. **미니배치·이분 k-평균** (CL25). (a) 둘 다 넣는다: `minibatch_kmeans`(sklearn의 갱신·멈춤과 기본값; 결과는 `KMeansResult`), `bisecting_kmeans`(`strategy='biggest_inertia'|'largest_cluster'`; 결과에 나눈 차례의 트리). 추천: 큰 n(10⁷)의 실용 도구이고 k-평균의 커널을 그대로 쓴다. (b) 미니배치만: 이분 k-평균은 쓰임이 적다. (c) 둘 다 나중.
23. **k-medoids** (CL26; §3.4). (a) `kmedoids(X, k, *, method='pam'|'clara', metric='euclidean', rng=None, …)`: PAM은 R `pam(variant='original')`의 BUILD + SWAP(같은 결과의 FastPAM1 재사용으로 빠르게; 교환 수의 상한 `max_iter`의 기본값은 CL2 마일스톤에서 정한다), CLARA는 R의 기본(`samples=5`, `sampsize=min(n, 40 + 2k)`)과 `rng`(R과 같은 표본은 아니다). FasterPAM은 나중. 추천: sklearn에 없는 R의 방법이라 R이 기준이고, PAM의 답이 R과 같은지 동점 없는 자료에서 견줄 수 있다. (b) R의 이름으로 `pam`, `clara` 두 함수: R과 같지만 같은 결과 객체에 이름이 둘이다. (c) FasterPAM도 넣는다(`method='fasterpam'`): 큰 k에서 수십 배 빠르지만 PAM과 다른 답이다.
24. **퍼지 군집** (CL27; §2.3). (a) Bezdek의 `fuzzy_cmeans(X, k, *, rng, m=2.0, max_iter=100, tol=…, init='random')`만 넣고, R `fanny`는 나중이다. 기본 `tol`은 CL2 마일스톤에서 e1071의 공개 문서를 보고 정한다(e1071이 설치되어 있지 않다). 추천: 퍼지 군집으로 가장 널리 쓰이는 꼴이고 k-평균의 커널을 쓴다. 비교 대상이 없어 참값으로만 지킨다. (b) `fanny`만(R과 견줄 수 있다): 거리 행렬 위의 다른 목적이라 큰 n에 맞지 않는다. (c) 둘 다.
38. **k-평균의 거리 꼴: 1스레드에서 Accelerate보다 느린 경로** (CL0 마일스톤에서 생김; CL9의 규칙으로 사용자에게 물었다; (a)로 정했다, CL49). 시제품(§5)에서 CL10의 차의 꼴은 1스레드로 Accelerate의 전개식(sklearn의 꼴)보다 1.76–8.28배 느리고(p·k가 클수록), 10스레드로는 0.29–0.96배다. (a) 차의 꼴을 정의로 두고(라벨·중심의 비트가 스레드와 Accelerate와 무관, 10⁸ 평행 이동에서도 정확), CL2·CL7 마일스톤에서 증명된 오차 한계의 거르기(전개식으로 후보를 거르고 차의 꼴로 다시 잰다; 같은 비트)를 넣어 1스레드를 줄인다. 거르기 시제품은 점마다 후보 1.00개였지만 1스레드 시간은 아직 다듬지 않았다. 추천: 결과를 바꾸지 않고 속도만 되찾는 길이고 10스레드에서는 이미 빠르다. (b) 1스레드에서 큰 p·k는 Accelerate 전개식을 쓰고 같은 스레드 수의 같은 비트만 약속: 1스레드가 빠르지만 스레드 수와 Accelerate가 라벨을 바꿀 수 있고(CL9의 결정을 낮춘다) 평행 이동한 자료에서 틀린다. (c) 차의 꼴만(거르기 없이): 가장 단순하지만 1스레드 큰 p·k에서 8배까지 느리다.

39. **k-평균 배정의 1스레드: 거르기를 넣은 뒤에도 Accelerate 전개식보다 1.5배 넘게 느린 곳** (CL2 마일스톤에서 생김; CL9의 규칙으로 묻는다; 답을 기다린다). 거르기(CL49)를 넣은 뒤 배정 한 번의 1스레드는 Accelerate 전개식의 1.30–2.48배다(k = 100: 1.57–2.07배, k = 10: 1.30–3.81배; §5). 경계(Hamerly)가 반복마다 대부분의 점을 건너뛰어 맞춤 전체는 1스레드로 sklearn(1스레드)보다 빠르거나 같다(10⁵ × 10, k = 10: 26.4 대 32.0 ms; k-means++ 20.3 대 19.1). 10스레드는 0.38–1.29배다. (a) 지금대로(차의 꼴이 정의, 같은 비트) 두고 CL7 마일스톤에서 커널을 다듬는다(타일 결과의 NEON 축약, 거르기의 점마다 고정 비용 약 40 ns를 줄이기). 추천: 결과와 결정성이 그대로이고 맞춤 전체는 이미 목표 안이다. (b) 1스레드의 작은 k·큰 p만 Accelerate 전개식으로 라벨을 정하고 같은 스레드 수의 같은 비트만 약속: 빠르지만 CL9·CL49를 낮추고 평행 이동한 자료에서 틀린다. (c) 이분 k-평균(1스레드 40.8 대 sklearn 31.2 ms)도 같이 CL7로 넘긴다(a와 함께).

**CL3 마일스톤 전**
25. **DBSCAN의 뜻** (CL28; §2.4). (a) sklearn의 뜻(`min_samples`는 자기를 센다, eps는 경계 포함, `weights`는 표본 무게)이고, 경계 점은 eps 안의 핵심 점 가운데 (거리, 열쇠)가 가장 작은 것의 군집이다. 추천: 분할이 입력 순서와 무관하다. sklearn은 먼저 펼친 군집이라 입력 순서에 따라 다르다(탐침). 경계 점이 아닌 모든 점은 sklearn과 같다. (b) sklearn처럼 펼친 순서: 입력 순서에 달린다. (c) 경계 점을 잡음으로(DBSCAN*, Campello 외 2013): 정의가 깔끔하지만 sklearn·원 논문과 다른 군집이다.
26. **HDBSCAN의 기준** (CL29; §2.4). (a) sklearn 1.9의 뜻과 기본값(`min_cluster_size=5`, `min_samples=None`은 `min_cluster_size`, `min_samples`는 자기를 센다, `cluster_selection_method='eom'`, `allow_single_cluster=False`, `cluster_selection_epsilon=0.0`, `max_cluster_size=None`, `alpha=1.0`)에 원본의 GLOSH `outlier_scores`와 줄인 트리를 더한다. `copy`·`store_centers`는 없다(중심은 게으른 필드). 추천: 설치된 비교 대상이 sklearn뿐이고 같은 라벨을 견줄 수 있다. 원본과의 `min_samples` 차이는 대응표에 적는다. (b) 원본 `hdbscan`의 뜻(`min_samples`가 자기를 세지 않는다): 원본 사용자에게 익숙하지만 비교 대상이 없다.
27. **OPTICS** (CL30; §2.4). (a) 넣는다: sklearn의 `OPTICS`(기본 `min_samples=5`, `max_eps=inf`, `xi=0.05`, `predecessor_correction=True`), 결과에 도달 거리 그림의 자료와 `extract_xi`, `extract_dbscan`. 기본 `metric`은 `'euclidean'`(sklearn은 `'minkowski'`, p = 2; 같은 거리). 추천: DBSCAN의 eps를 정하기 어려울 때의 표준 도구이고 핵심 거리·KD 트리를 나눠 쓴다. (b) 나중: HDBSCAN이 같은 쓰임의 대부분을 맡는다.
28. **Mean Shift** (CL31). (a) 나중. 추천: 평탄 핵의 반복이 O(n²·반복)이고 sklearn의 `estimate_bandwidth`는 기본으로 `random_state=0`의 표본을 써서 정의가 흐리다. 밀도의 봉우리 찾기는 sd.smooth의 `kde`로도 할 수 있다. (b) CL3 마일스톤에 넣는다(sklearn의 평탄 핵과 `bin_seeding`).

**CL4 마일스톤 전**
29. **혼합 모형의 공분산 꼴** (CL32; §2.5). (a) sklearn의 넷(`'full'`, `'tied'`, `'diag'`, `'spherical'`). mclust의 14 꼴(부피·모양·방향의 같음/다름; Celeux–Govaert 1995)은 나중이다. 추천: 비교 대상이 sklearn뿐이고(mclust 없음) 넷이 쓰임의 대부분이다. (b) mclust의 14 꼴까지: 표현력이 크지만 비교 대상 없이 참값으로만 지키고, 일부 꼴은 M 단계가 반복(고유분해)이라 일이 크다.
30. **EM의 세부** (CL33; §3.7). (a) sklearn과 같다: `tol=1e-3`(점마다 평균한 하한의 변화), `reg_covar=1e-6`, `max_iter=100`, `n_init=1`, `init='kmeans'`(그 밖 `'k-means++'`, `'random'`, `'random_from_data'`), `weights_init`·`means_init`·`precisions_init`. 시작의 k-평균은 같은 `rng`의 하위 흐름. 결과는 `lower_bound`(sklearn의 `lower_bound_`: 마지막 E 단계)와 `loglik`(마지막 모수의 로그우도 합)을 둘 다 둔다. 추천: sklearn의 고정점과 견줄 수 있다. 두 값이 탐침에서 3×10⁻⁶ 달라 이름을 나눈다. (b) mclust처럼 모형 기반 계층 군집으로 시작: 무작위가 없지만 비교 대상이 없고 O(n²)이다.
31. **혼합 모형의 k 고르기** (CL34). (a) 결과에 `bic`, `aic`만 두고 k를 고르는 함수는 두지 않는다(사용자가 k마다 부른다). 추천: 고르기 규칙(BIC 최소, 1-SE)은 나중의 공통 튜닝 도구의 몫이다. (b) `gaussian_mixture(X, k=[1, …, 9])`처럼 k의 목록을 받아 BIC로 고른다(mclust의 `Mclust`): 편하지만 결과의 꼴이 둘이 된다.
32. **베이즈 혼합** (CL35). (a) 나중. 추천: 변분 추론의 사전 분포 모수가 많고 범위가 크다. (b) CL4 마일스톤에 `bayesian_gaussian_mixture`를 넣는다.

40. **혼합 모형 E 단계의 1스레드: 큰 p에서 같은 정의의 Accelerate 꼴보다 1.5배 넘게 느린 곳** (CL4 마일스톤에서 생김; CL9의 규칙으로 묻는다; 답을 기다린다). full·tied의 E 단계(차의 꼴 ‖Uᵀ(x − μ)‖², 자체 NEON 판)는 1스레드에서 같은 정의를 Accelerate `dgemm`으로 잰 꼴의 p = 30 1.43배, p = 50 1.72배, p = 100 2.9배다(n = 2×10⁴, k = 5; §5). 맞춤 전체의 1스레드는 p = 50에서 sklearn의 full 1.09배(EM 20번 1.23배), tied 1.7배(EM 1.95배; sklearn의 tied M 단계가 한 번 훑기라서이기도 하다), diag 1.2배이고 p ≤ 10에서는 sklearn의 1.3–3.4배 빠르다. 10스레드는 모든 크기에서 sklearn보다 빠르다(2.5–17배; p = 50의 diag·tied가 2.5배). (a) 지금대로(자체 고정 순서 커널, 스레드와 무관한 비트) 두고 CL7 마일스톤에서 E 단계의 판을 다듬는다(판을 키우기, 성분 둘을 한 번에). 추천: 결정성(CL9)과 정확도(차의 꼴)가 그대로이고 목표 사례(p = 10)는 이미 목표 안이다. (b) 1스레드의 큰 p만 Accelerate `dgemm`으로 y = Uᵀz를 계산하고 같은 스레드 수의 같은 비트만 약속: 큰 p에서 빠르지만 CL9를 낮춘다(Accelerate의 합 차례가 스레드와 판에 달린다). (c) tied의 M 단계를 sklearn의 한 번 훑기로: tied가 k배 빨라지지만 상쇄가 생긴다(주어진 책임도에서 sklearn이 sd 계약의 9.3배 틀린 꼴).

**CL5 마일스톤 전**
33. **지표의 계산과 기본값** (CL36; §3.8, §4.6). (a) ARI·RI·FM과 짝 혼동 행렬은 128비트 정수로 정확히 셈해 올바르게 반올림하고, AMI·NMI의 `average_method` 기본은 sklearn처럼 `'arithmetic'`, 특수 경우(한 군집, 모두 홑원소)는 sklearn과 같은 값, `contingency_matrix`는 밀집 int64(sklearn의 `sparse=True`는 `sd.sparse` 배열을 돌려주는 키워드로)다. 추천: 정수의 유리식은 정확히 계산할 수 있고 비용이 거의 없다. (b) sklearn처럼 float64로 셈한다: 큰 n에서 C(n, 2)가 2⁵³을 넘어 끝자리를 잃는다.
34. **gap 통계량** (CL37; §3.9). (a) `gap_statistic(X, k_max, /, *, rng, method='kmeans'|'pam', B=100, d_power=1, reference='scaledPCA'|'original', select='firstSEmax', se_factor=1.0)` → `GapResult`. 기본값은 R `clusGap`·`maxSE`와 같고 `method`의 군집은 이 영역의 함수(k-평균은 `n_init='auto'`)다. 임의의 Python 함수는 받지 않는다. 추천: R의 뜻이고 B벌과 k마다의 군집이 C++ 안에서 나란하다(Python 콜백은 GIL에 묶인다). (b) R처럼 군집 함수를 콜백으로 받는다: 아무 방법이나 쓸 수 있지만 느리고 결정성이 그 함수에 달린다. (c) `d_power=2`(Tibshirani 외 2001의 원래 뜻)를 기본으로: 논문과 같지만 R과 다르다.
35. **Gower 거리** (CL38; §2.6). (a) `sd.cluster.gower_distance(data, /, *, types=None, weights=None)`: `DataFrame`(수 열, 범주 열, 순서형 범주, bool은 대칭 이진; `types`로 비대칭 이진·로그 비율을 지정)을 받아 압축 꼴 거리를 돌려준다. 결측은 R처럼 그 변수를 빼고 무게를 다시 맞춘다. 추천: R `daisy`가 `cluster` 꾸러미에 있고 쓰임이 혼합 자료의 PAM·계층 군집이다. spatial의 거리는 수 배열만 받는 원칙을 지킨다. R은 논리형 열을 비대칭 이진으로 바꾸며 경고하지만 sd는 대칭 이진(`[R과 다름]`; `types`로 바꾼다). (b) spatial의 `pdist(metric='gower')`: 거리의 자리는 하나지만 spatial이 `DataFrame`과 열의 형을 알아야 한다. (c) 나중.

**CL6 마일스톤 전**
36. **스펙트럼 군집** (CL39). (a) 나중: `sd.embed`의 스펙트럼 임베딩(라플라시안의 고유벡터; `sd.graph.laplacian`, `sd.sparse.eigsh`)이 생긴 뒤 그 위에 `spectral_clustering`을 둔다. 추천: 임베딩과 k-평균의 합성이라 고유벡터 계산의 자리를 한 번만 만든다. `eigsh`의 시작 벡터와 부호 고정이 결정성의 문제라 embed 영역에서 한 번 정한다. (b) 이 영역에 지금 넣는다(임베딩을 이 영역에 둔다): 나중의 embed와 겹친다.
37. **Affinity Propagation과 그 밖** (CL40). (a) `affinity_propagation(X, /, *, damping=0.5, max_iter=200, convergence_iter=15, preference=None, metric='euclidean')`을 넣고 sklearn이 유사도에 더하는 작은 잡음(`random_state`로 `standard_normal`의 n×n 배열)은 넣지 않는다(정해진 동점 규칙으로 대신한다; 그래서 `rng`가 없다). Birch는 나중, 바이클러스터링(`SpectralBiclustering`, `SpectralCoclustering`)은 이 영역 밖이다. 추천: AP는 k를 정하지 않는 방법으로 쓰임이 있고 결정적인 반복이다. 잡음은 퇴화(같은 유사도)를 깨려는 것이라 열쇠 규칙이 같은 일을 한다(`[sklearn과 다름]`: 잡음 때문에 sklearn과 끝자리가 다를 수 있다). (b) AP도 나중: CL6 마일스톤이 없어진다. (c) Birch도 넣는다: 흐름 자료의 도구지만 결과가 `threshold`와 입력 순서에 크게 달린다.
