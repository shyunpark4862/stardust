# 임베딩 영역 설계 (sd.embed)

> 상태: **초안** (2026-11-27). 큰 결정(§1.1 EM1–EM3)은 사용자가 정했고(2026-11-24, 2026-11-27), 결정성의 규칙(EM4)은 사용자 결정(2026-11-27)이다. 나머지 결정(EM5–EM48)은 추천안이다(사용자 위임 2026-11-27). §9에는 사용자에게만 물을 네 가지(범위의 의심스러운 이름들, 이웃 찾기의 기본값, 재현성을 지킬 수 없는 UMAP 최적화의 선택지, 비교 대상의 설치)를 모았다. 잰 값은 아직 없다: 시간과 비용의 수는 어림이고 EM0 마일스톤의 탐침이 잰다(§5).

`sd.embed`(점들을 낮은 차원으로 옮기기: 선형 분해, 거리 보존, 이웃 그래프 위의 다양체 학습, 확률적 이웃 임베딩, 그리고 옮긴 결과를 재는 지표)의 안과 내부 구조를 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처는 [docs/api/embed.md](../api/embed.md)에 있다. 꼴과 규칙은 [docs/design/cluster.md](cluster.md)를 본으로 삼는다.
- 비교 대상(설치된 것, 2026-11-27 확인): `.venv`의 scikit-learn 1.9.1(`sklearn.decomposition`의 `PCA`·`TruncatedSVD`·`KernelPCA`·`FactorAnalysis`·`FastICA`·`NMF`, `sklearn.manifold`의 `TSNE`·`MDS`·`ClassicalMDS`·`Isomap`·`LocallyLinearEmbedding`·`SpectralEmbedding`·`trustworthiness`, `sklearn.random_projection`), scipy 1.18.1(`linalg.svd`, `sparse.linalg.svds`·`eigsh`), statsmodels 0.15.0(`multivariate.pca.PCA`, `multivariate.factor.Factor`, `factor_rotation`), R 4.6.2의 `stats`(`prcomp`, `princomp`, `cmdscale`, `factanal`, `varimax`, `promax`), `MASS` 7.3.66(`isoMDS`, `sammon`, `corresp`, `mca`), `irlba`(`irlba`, `prcomp_irlba`), `RSpectra`(`eigs_sym`, `svds`)다. 설치되지 않은 것(umap-learn, pynndescent, openTSNE, pacmap, trimap; R의 Rtsne, uwot, fastICA, NMF, vegan, smacof, FactoMineR, psych, kernlab, dimRed)은 사용자의 허락 없이 설치하지 않는다(§9의 4번). 그때까지 UMAP과 FIt-SNE는 공개 문서와 원 논문으로 뜻을 맞추고 참값과 품질 지표로만 지킨다(§7.4).
- 바닥은 배열(`csrc/array/`), 공용 커널(`csrc/base/`), 스레드풀(`csrc/parallel/`), Accelerate의 BLAS·LAPACK(`csrc/accelerate/blas.hpp`, linalg의 `lapack.hpp`), spatial의 거리 커널·KD 트리·이웃 찾기(`csrc/spatial/metric.hpp`·`kdtree.hpp`·`neighbors.hpp`·`sfc.hpp`), sparse의 CSR·SpMV·`eigsh`·`svds`·LU(`csrc/sparse/`), graph의 최단 경로·라플라시안·연결 요소(`csrc/graph/`), fft의 Stockham(`csrc/fft/`), random의 Philox(`csrc/random/philox.hpp`)와 stats의 정규 난수(`stats/sample_dist.hpp`)다. Python 층은 `sd.random`(`Generator`), `sd.linalg`(`LinAlgError`), `sd.sparse`, `sd.spatial`, `sd.graph`, `sd.optimize`(인자 분석의 L-BFGS-B, UMAP의 a·b 맞춤), `sd.frame`, `stardust._summary`, `stardust._calls`를 쓴다.
- u는 단위 반올림 2⁻⁵³, γ_k = ku/(1 − ku)다. n은 점 수, p는 차원(변수 수), d는 임베딩의 차원(`n_components`), k는 이웃 수, m은 평가점 수다. "열쇠"는 cluster CL7의 점 열쇠(좌표의 사전순 차례, 같으면 입력 색인)다.

---

## 1. 기반

### 1.1 결정 기록
"EMn"이 결정 번호와 마일스톤 이름에 함께 쓰이므로 마일스톤은 언제나 "EM2 마일스톤"처럼 적는다.

| 번호 | 결정 |
|---|---|
| EM1 API의 꼴 | 함수가 결과 객체를 돌려준다: `pca(X, n_components)` → `PCAResult`(`.embedding`, `.components`, `.transform(X_new)`). 추정기 클래스(`PCA(...).fit(X)`)가 아니다. cluster CL1, 회귀·평활과 같은 꼴이다: `summary()`, 한 줄 `repr`, 읽기 전용 배열 필드, `_calls`의 호출 모양 에러 (사용자 결정 2026-11-24, 2026-11-27) |
| EM2 영역의 범위 | 차원 축소와 임베딩("PCA, UMAP, t-SNE 같은 것")이고 선형 분해(PCA, 절단 SVD, 인자 분석, ICA, NMF, 무작위 투영, 커널 PCA)도 같은 영역이다. 판별 분석의 투영(LDA)은 `sd.classify`, 스펙트럼 군집은 `sd.cluster`의 나중 후보(CL39; 이 영역의 `spectral_embedding`을 부품으로 쓴다)다 (사용자 결정 2026-11-24, 2026-11-27) |
| EM3 비교 대상 | Python과 R의 라이브러리를 모두 조사하고(§1.2) 설치된 것과 견준다. 설치되지 않은 것은 허락을 받아야 설치한다(§9의 4번) (사용자 결정 2026-11-27) |
| EM4 결정성의 규칙 | 지키는 약속은 D-33이다: 같은 기계, 같은 스레드 수, 같은 입력(같은 `rng` 상태)이면 실행마다 같은 바이트다(API.md §3). 스레드 수와 무관한 바이트(1·3·10스레드가 같다)는 그 비용이 작을 때만 지키고, 잴 수 있는 속도를 잃는 경로는 D-33으로 물러선다. 길마다의 선택은 §3.11이고 이 영역이 스스로 정한다. D-33조차 지킬 수 없는 알고리즘(고정 스레드 수에서도 실행마다 다른 결과: Hogwild SGD 등)은 쓰지 않거나 사용자에게 묻는다(§9의 3번) (사용자 결정 2026-11-27) |
| EM5 이름공간과 이름 | 모든 공개 이름이 `sd.embed`에 바로 있다(평평하다). 맞추는 함수는 sklearn 클래스 이름의 소문자 밑줄 꼴(`pca`, `truncated_svd`, `kernel_pca`, `factor_analysis`, `nmf`, `isomap`, `spectral_embedding`, `tsne`, `umap`, `mds`)이고, sklearn에 함수가 있으면 그 이름(`fastica`, `locally_linear_embedding`, `johnson_lindenstrauss_min_dim`, `trustworthiness`), sklearn에 없으면 R의 이름(`sammon`, `classical_mds`는 sklearn 1.9의 `ClassicalMDS`)이다. 결과는 `<이름>Result`(`PCAResult`, `TSNEResult`, `UMAPResult`)이고 `mds`·`sammon`은 `MDSResult`를 나눠 쓴다(cluster의 `kmeans`·`minibatch_kmeans`와 같은 규칙) (추천안, 사용자 위임 2026-11-27) |
| EM6 인자의 순서 | (X, n_components)다: `X`는 첫째 위치 인자((n, p), 점이 행), 임베딩의 차원은 둘째 위치 인자 `n_components`(위치로만)이고 기본값은 기준 라이브러리의 것이다(API.md §4의 규칙 2: `pca`·`kernel_pca`·`fastica`는 None, `nmf`는 `'auto'`, `random_projection`은 `'auto'`, `factor_analysis`는 기본값 없음(R `factanal`), 나머지는 2). cluster의 `k`와 다른 이름인 것은 다른 개념(군집 수와 차원)이라서다. 옵션은 키워드로만, 별칭 없이 받는다 (추천안, 사용자 위임 2026-11-27) |
| EM7 결과의 공통 기반 | `_results.py`의 `EmbeddingResult`(cluster `ClusterResult`와 같은 장치: 처음 읽을 때 한 번 계산하는 필드, 쓰기·지우기는 `AttributeError`, 배열은 읽기 전용): `embedding`((n, d) float64, 입력 순서; `'omit'`이 뺀 행은 NaN), `n_components`, `nobs`, `omitted`, `summary()`, 한 줄 `repr`. 새 점을 옮길 수 있는 결과는 `transform(X_new, /)`, 되돌릴 수 있는 결과는 `inverse_transform(Z, /)`이다(cluster `KMeansResult.transform`과 sklearn의 이름). 반복하는 결과는 `n_iter`, `converged`, `objective`(t-SNE의 KL 발산, MDS의 stress, 인자 분석의 로그우도 등)가 있다 (추천안, 사용자 위임 2026-11-27) |
| EM8 입력 | cluster §1.2와 같다: `sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `DataFrame`(열이 변수)과 `Series`(한 변수), 계산은 float64, bool·정수는 float64로, 복소수는 `TypeError`. 희소 배열(`sd.sparse`)은 `pca`(`svd_solver='lanczos'`·`'randomized'`·`'covariance_eigh'`; 가운데로 옮기기는 암묵적), `truncated_svd`, `nmf`, `random_projection`, `spectral_embedding(affinity='precomputed')`만 받고, 다른 함수에는 `TypeError`(`isomap(): 'X' must be dense; for a sparse array use X.toarray()`)다 (추천안, 사용자 위임 2026-11-27) |
| EM9 NaN | `nan_policy='raise'|'omit'`, 기본 `'raise'`(cluster CL11과 같다). `'omit'`은 NaN이 있는 행을 빼고 맞추고, `embedding`의 그 행은 NaN, `nobs`는 뺀 뒤의 수, `omitted`가 뺀 행이다. ±inf는 언제나 `ValueError`, `X_new`·`Z`의 NaN·±inf는 `ValueError`(cluster CL101의 메시지 꼴). `'propagate'`는 두지 않는다(반복법이 NaN을 퍼뜨리면 모든 결과가 무의미하다) (추천안, 사용자 위임 2026-11-27) |
| EM10 거리 입력 | `metric`의 이름과 값은 `sd.spatial.cdist`와 같다. 거리로 정의되는 함수(`classical_mds`, `mds`, `sammon`, `isomap`, `tsne`, `umap`, `trustworthiness`, `continuity`, `stress`)는 `metric='precomputed'`로 (n, n) 정사각(대칭, 대각 0, 음수·NaN 없음) 또는 1차원 압축 꼴을 받는다(cluster CL12). 유사도는 `affinity`라는 다른 이름이다: `spectral_embedding(affinity='precomputed')`는 유사도(인접) 행렬이다(cluster CL91과 같은 갈래). sklearn의 기본 `'minkowski'`(p = 2)는 `'euclidean'`으로 적는다(같은 거리; cluster OPTICS와 같다) (추천안, 사용자 위임 2026-11-27) |
| EM11 난수 (`rng`) | 고른 옵션이 난수를 뽑을 때만 `rng`(`sd.random.Generator`)를 받는다. 무엇이 뽑는지는 옵션의 값(방법, 시작)으로만 정하고 자료의 모양으로 정하지 않는다. 언제나 뽑는 함수(`fastica`, `random_projection`, `umap`)는 기본값 없는 키워드(`rng`가 없으면 `TypeError`), 옵션에 달린 함수(`pca`·`truncated_svd`·`kernel_pca`의 `'randomized'`, `nmf`의 `init='random'`·`'nndsvdar'`·`shuffle=True`, `mds`의 `init='random'`, `tsne`의 `init='random'`)는 `rng=None`이고 필요한데 없으면 `TypeError`(`tsne(): init='random' needs 'rng' (sd.random.default_rng(seed))`), 쓰지 않는데 주면 `ValueError`(`pca(): 'rng' is only used with svd_solver='randomized'`; 회귀 `rlm`, cluster `kmedoids`의 꼴)다. 호출 하나가 64비트 두 단어를 받아 Philox4×64-10의 키로 쓰고, 흐름 r의 t번째 수는 원소 r·2⁴⁰ + t다(cluster CL41). 그래서 같은 `rng` 상태면 같은 바이트이고 나란히 뽑아도 스레드 수와 무관하다. 결과의 메서드가 뽑으면(`UMAPResult.transform`) 그 메서드도 `rng`를 받는다(cluster `GaussianMixtureResult.sample`과 같다) (추천안, 사용자 위임 2026-11-27) |
| EM12 입력 순서와 무관한 결과 | 결정적인 함수는 같은 점들을 어떤 순서로 주어도 행을 그 순서로 옮긴 같은 바이트의 `embedding`을 낸다. 안에서 점을 열쇠 순서로 놓은 사본으로 계산하고(cluster CL7, CL61의 사본; O(np) 메모리, 사본은 n = 10⁵ × p = 100에서 수 ms 어림), 이웃의 동점은 (거리, 열쇠)다. 난수를 쓰는 함수도 원소 번호를 열쇠의 차례로 매겨 같은 `rng` 상태면 입력 순서와 무관하다. 미리 계산한 거리는 좌표가 없어 입력 색인이 열쇠다 (추천안, 사용자 위임 2026-11-27) |
| EM13 dtype | 모든 계산은 float64다(API.md §1). `[sklearn과 다름]`: sklearn `TSNE`는 거리·P·임베딩·기울기를 float32로 계산하고 umap-learn도 임베딩을 float32로 둔다. float64는 메모리 대역이 두 배라 Barnes–Hut 한 번이 1.2–1.5배 느릴 수 있다(어림; EM6 마일스톤이 잰다). float32 길은 두지 않는다(영역의 dtype 규칙을 깨지 않는다) (추천안, 사용자 위임 2026-11-27) |
| EM14 성분의 부호 | 고유벡터·특이벡터에서 나온 축은 부호를 정한다: 성분 벡터(PCA의 `components` 행, 커널·고전 MDS·Isomap·스펙트럼 임베딩은 `embedding`의 열)마다 절댓값이 가장 큰 원소를 양수로, 같은 절댓값이면 작은 열쇠의 원소다. sklearn 1.5 이후의 `svd_flip(u_based_decision=False)`와 같은 규칙이고(동점은 argmax의 첫 자리), cluster gap(CL88)의 축 규칙과 같다. 그래서 결과가 LAPACK의 부호 선택과 풀이 방법에 달리지 않는다. R `prcomp`는 부호를 정하지 않는다 `[R과 다름]` (추천안, 사용자 위임 2026-11-27) |
| EM15 `'auto'`는 어림을 고르지 않는다 | 풀이를 모양으로 고르는 `'auto'`(`pca`의 `svd_solver`, `kernel_pca`의 `eigen_solver`, 이웃 그래프 방법의 고유값 풀이)는 무작위 어림(`'randomized'`)을 고르지 않는다. 고르는 것은 정확도 계약(§4)이 같은 풀이들(밀집 분해, Gram의 고유값, Lanczos의 허용오차 수렴)이다. `[sklearn과 다름]`: sklearn `PCA`의 `'auto'`는 큰 입력에서 `'randomized'`를 골라 `random_state=None`이면 실행마다 다른 어림을 낸다. 무작위 어림은 `rng`가 필요하고(EM11) 모양으로 `rng`의 필요가 바뀌면 안 되므로 사용자가 이름으로 고른다 (추천안, 사용자 위임 2026-11-27) |
| EM16 밀집 선형대수 | 중심화 뒤의 Gram(XᵀX), 밀집 SVD·대칭 고유값·QR, 큰 곱(무작위 SVD의 XΩ, FastICA·NMF·인자 분석의 행렬 곱)은 Accelerate의 BLAS·LAPACK이다(linalg·regression·interpolate와 같은 바닥). 자체 고정 순서 커널(NEON의 Gram, Jacobi·삼중대각 QR의 고유값)은 1스레드에서 2–8배 느리다(어림: cluster §5의 k-평균 거리 측정 1.76–8.28배, linalg §10.5의 드라이버 측정). Accelerate를 sd가 나누지 않고 한 번 부르면 결과가 sd 스레드 수와 무관하고 Accelerate의 스레드 모드가 같을 때 같다(API.md §3). sd가 행 덩이로 나눠 부를 때는 덩이를 크기로만 정하고, 스레드 수로 나누는 쪽이 잴 수 있게 빠르면 D-33으로 물러선다(EM4; §3.11) (추천안, 사용자 위임 2026-11-27) |
| EM17 이웃 그래프 | k 최근접은 `sd.spatial.kneighbors`의 정의다(길: KD 트리, GEMM 고르기, 전수; 결과는 같은 바이트이고 Accelerate 모드와도 무관하다). 자기 자신은 빼고, 동점은 (거리, 열쇠)다(EM12). 같은 점이 여러 번 있으면 거리 0의 이웃이다(sklearn과 같다). 근사 이웃(NN-descent)은 EM8 마일스톤의 옵션이고(§3.9), 기본값은 §9의 2번에 달렸다 (추천안, 사용자 위임 2026-11-27) |
| EM18 고유값 풀이 | n × n 밀집(고전 MDS, Isomap의 중심화한 핵, 커널 PCA)은 `eigen_solver='auto'`가 d ≥ n/10이거나 n ≤ 2000이면 LAPACK `syevd`, 그 밖은 `sd.sparse.eigsh`(ARPACK 이식, 정해진 시작 벡터; 밀집 행렬도 받는다)의 `'LA'`다. 희소(LLE의 (I − W)ᵀ(I − W), 스펙트럼 임베딩의 라플라시안)는 `eigsh`의 shift-invert(σ = 0 근처; Accelerate LU/AMD, 결정적)이고 n ≤ 2000이면 밀집 `syevd`다. 문턱은 초안이고 EM4·EM5 마일스톤이 잰다. sklearn의 `'arpack'`은 이름이 `'lanczos'`다(EM23) (추천안, 사용자 위임 2026-11-27) |
| EM19 PCA의 뜻 | sklearn `PCA`의 뜻과 필드다: 분산은 n − 1로 나눈다(`explained_variance`), `explained_variance_ratio`, `singular_values`, `noise_variance`(버린 성분의 분산 평균), `whiten`, 확률적 PCA의 `score_samples`와 `covariance`·`precision`(sklearn `get_covariance`·`get_precision`; 게으른 필드). `n_components`는 None(min(n, p)), 1 이상의 int, (0, 1)의 float(누적 분산 비율), `'mle'`(Minka; `'full'`·`'covariance_eigh'`만)다. R `prcomp`의 `scale.`(열을 표준편차로 나누기)을 `scale=False`로 더한다(sklearn에는 없다; statsmodels `standardize`, R `princomp(cor=TRUE)`) (추천안, 사용자 위임 2026-11-27) |
| EM20 PCA의 풀이 | `svd_solver`는 `'auto'`, `'full'`(가운데로 옮긴 X의 `gesdd`), `'covariance_eigh'`(Gram의 `syevd`; n ≫ p), `'lanczos'`(`sd.sparse.svds`의 Baglama–Reichel, 암묵적 중심화의 `LinearOperator`; scipy `'arpack'` 대신), `'randomized'`(Halko–Martinsson–Tropp, `rng` 필요)다. `'auto'`(EM15)는 p ≤ 1000이고 n ≥ 10p면 `'covariance_eigh'`(sklearn의 문턱), 그 밖에 d ≤ min(n, p)/10이고 min(n, p) > 500이면 `'lanczos'`, 나머지는 `'full'`이다(초안; EM1 마일스톤이 잰다). 희소 입력의 `'auto'`는 `'lanczos'`다 (추천안, 사용자 위임 2026-11-27) |
| EM21 무작위 SVD의 세부 | Ω는 원소 (행, 열)의 Philox 정규 수(stats의 ziggurat)이고 거듭제곱 반복의 정규화는 sklearn의 `'auto'`(`n_power_iter` ≤ 2면 LU, 넘으면 QR; LAPACK)를 따른다. 거듭제곱 반복 수의 이름은 `pca`와 `truncated_svd`에서 `n_power_iter` 하나다 `[sklearn과 다름]`(sklearn은 `iterated_power`와 `n_iter`) (추천안, 사용자 위임 2026-11-27) |
| EM22 절단 SVD | `truncated_svd`는 가운데로 옮기지 않는 SVD다(sklearn `TruncatedSVD`, 잠재 의미 분석). 기본 `algorithm='lanczos'`(`sd.sparse.svds`)다 `[sklearn과 다름]`: sklearn의 기본 `'randomized'`(n_iter = 5)는 어림이고 `random_state`에 달린다(EM15). `'randomized'`(`rng` 필요)와 `'full'`(밀집 `gesdd`)도 있다 (추천안, 사용자 위임 2026-11-27) |
| EM23 풀이 이름의 통일 | 이 영역의 Lanczos 계열 풀이는 모두 `'lanczos'`다(`pca`의 `svd_solver`, `truncated_svd`의 `algorithm`, `kernel_pca`·`isomap`·`locally_linear_embedding`·`spectral_embedding`의 `eigen_solver`). sklearn의 `'arpack'`은 대응표에 적는다(sd의 `svds`는 ARPACK이 아니라 irlba의 방법이라 `'arpack'`이라 부르면 틀린다). 인자의 이름은 sklearn의 것(`svd_solver`, `algorithm`, `eigen_solver`)을 그대로 둔다 (추천안, 사용자 위임 2026-11-27) |
| EM24 무작위 투영 | sklearn의 두 클래스를 `random_projection(X, n_components='auto', /, *, rng, method='gaussian'|'sparse', density='auto', eps=0.1)` 하나로 둔다. `'gaussian'`은 N(0, 1/d), `'sparse'`는 Li 외(2006)의 ±√(1/(density·d))(`density='auto'`는 1/√p)이고 `components`는 `sd.sparse.csr_array`다. `n_components='auto'`는 `johnson_lindenstrauss_min_dim(n, eps=eps)`다(sklearn과 같다; p보다 크면 `ValueError`). 원소 (i, j)는 Philox의 원소 번호로 뽑아 나란히 만들어도 같다. `inverse_transform`은 `components`의 유사역행렬(게으른 필드; sklearn `compute_inverse_components`)이다 (추천안, 사용자 위임 2026-11-27) |
| EM25 커널 PCA | sklearn `KernelPCA`의 뜻이다. 커널 이름은 sklearn `pairwise_kernels`의 `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, `'cosine'`, `'precomputed'`이고 `gamma=None`은 1/p다(나중의 `sd.classify` SVM이 같은 이름과 정의를 쓴다). 핵 행렬은 열쇠 순서의 타일 커널(spatial의 차의 꼴 제곱 거리, 같은 바이트), 중심화는 덩이 합이다. `fit_inverse_transform=True`면 `alpha`의 핵 릿지로 되돌림을 맞춘다. `remove_zero_eig`는 sklearn과 같다(`n_components=None`이면 언제나 참) (추천안, 사용자 위임 2026-11-27) |
| EM26 인자 분석의 기준과 계산 | 기준 라이브러리는 R `factanal`이다(최대우도, 검정 통계량, varimax·promax). 정의: 상관행렬 R에서 고유도 ψ ∈ [`lower`, 1]의 우도를 R과 같은 목적(`factanal.fit.mle`의 `FAfn`·`FAgr`)으로 L-BFGS-B(`sd.optimize.minimize`)로 최소화하고, 적재는 R^{-1/2}가 아닌 Ψ^{-1/2}RΨ^{-1/2}의 고유분해(LAPACK)다. 결과는 R의 상관 눈금 `loadings`·`uniquenesses`와 sklearn의 자료 눈금 `components`·`noise_variance`(열의 표준편차를 곱한 것: 최대우도 인자 분석은 척도에 공변이라 같은 해다)를 함께 준다. 시작은 R의 `(1 − 0.5·factors/p)/diag(R⁻¹)`이다(난수 없음). `[sklearn과 다름]`: sklearn `FactorAnalysis`는 기본이 무작위 SVD(`random_state=0`)의 반복이라 같은 고정점에 덜 수렴한 채 멈출 수 있다(EM2 마일스톤이 견준다) (추천안, 사용자 위임 2026-11-27) |
| EM27 인자 회전 | `rotation`은 None, `'varimax'`(R의 기본; `normalize=True`, `eps=1e-5`), `'quartimax'`(sklearn), `'promax'`(R, `power=4`)이고 기본은 R과 같은 `'varimax'`다. 같은 계산을 공개 함수 `rotate_factors(loadings, /, *, method='varimax', normalize=True, eps=1e-05, power=4)` → namedtuple `RotationResult(loadings, rotation, phi)`(phi는 promax의 인자 상관, 직교 회전은 None)로도 둔다(statsmodels `rotate_factors`, R `varimax`·`promax`). 회전 뒤 열의 부호는 열의 합이 양수로(R `factanal`의 규칙), 열의 차례는 제곱합이 큰 차례다(R과 같다) (추천안, 사용자 위임 2026-11-27) |
| EM28 인자 점수 | `FactorAnalysisResult.transform(X_new, /, *, method='regression')`: `'regression'`(Thomson; sklearn `transform`과 같은 사후 평균)과 `'bartlett'`(R `scores='Bartlett'`)이다. `embedding`은 `'regression'`의 점수다 (추천안, 사용자 위임 2026-11-27) |
| EM29 FastICA | sklearn `FastICA`의 뜻과 기본값이다: `algorithm='parallel'|'deflation'`, `whiten='unit-variance'|'arbitrary-variance'|False`, `fun='logcosh'|'exp'|'cube'`(`alpha`는 logcosh의 모수; sklearn `fun_args`), `max_iter=200`, `tol=1e-4`, `whiten_solver='svd'|'eigh'`. `w_init=None`이면 원소 (i, j)의 Philox 정규 수다(`rng` 필수). 성분은 FastICA의 부호·순서 비결정성을 정한다: 부호는 EM14, 차례는 원천 신호의 첨도(4차 누율) 절댓값이 큰 차례다(같으면 맞춤 안의 번호). `[sklearn과 다름]`: sklearn은 반복이 끝난 순서 그대로다. 사용자 함수 `fun`(호출 가능)은 받지 않는다(R `fastICA`도 둘뿐이다) (추천안, 사용자 위임 2026-11-27) |
| EM30 NMF | sklearn `NMF`의 뜻과 기본값이다: `solver='cd'|'mu'`, `beta_loss`(`'mu'`만 `'kullback-leibler'`·`'itakura-saito'`·수), `init=None`(d ≤ min(n, p)면 `'nndsvda'`, 아니면 `'random'`), `'nndsvd'`·`'nndsvdar'`·`'random'`·`'custom'`(`W_init`·`H_init`), `alpha_W`, `alpha_H='same'`, `l1_ratio`, `tol=1e-4`, `max_iter=200`, `shuffle=False`. NNDSVD의 SVD는 결정적인 `'lanczos'`다 `[sklearn과 다름]`(sklearn은 `randomized_svd`). CD는 성분의 차례로, 행렬 곱은 Accelerate(EM16)다. `'random'`은 sklearn의 √(mean(X)/d)·|N(0, 1)|이다. 음수 값은 `ValueError`, 희소 입력을 받는다. `embedding`은 W, `components`는 H, `transform`은 H를 고정한 W의 풀이(sklearn과 같다) (추천안, 사용자 위임 2026-11-27) |
| EM31 고전 MDS | `classical_mds`는 R `cmdscale`과 sklearn 1.9 `ClassicalMDS`의 뜻이다: B = −½JD²J의 큰 고유값 d개(EM18)와 √λ로 늘린 고유벡터. 음의 고유값은 버리고 그 축은 0이다(R은 경고와 함께; sd는 결과의 `eigenvalues`가 모두 보인다). `add=True`는 R의 Cailliez 상수(2n × 2n 비대칭 고유값 문제의 가장 큰 실수 고유값, LAPACK `geev`)다. `gof`는 R `GOF`의 두 값이다 (추천안, 사용자 위임 2026-11-27) |
| EM32 SMACOF MDS | `mds`는 sklearn 1.9 `MDS`의 SMACOF(Guttman 변환, 멈춤은 (old − new)/(Σδ²/2) < `eps`, `max_iter=300`, `eps=1e-6`), `metric_mds=True|False`(비계량은 등위 회귀의 불일치도; sklearn과 같은 Kruskal의 첫째 동점 처리)다. `init`의 기본은 `'classical_mds'`이다 `[sklearn과 다름]`: sklearn 1.9.1의 기본 `'warn'`은 `'random'`처럼 돌며 1.10에 `'classical_mds'`로 바뀐다고 경고한다(sd는 바뀐 뒤의 기본값을 쓴다; 그래서 기본 호출에 `rng`가 필요 없다). `init='random'`은 `rng`가 필요하고 `n_init` > 1은 `'random'`에서만이다(시작들은 원소 번호로 나란히, 고르기는 (stress, 번호)). `normalized_stress='auto'`는 sklearn과 같다(비계량이면 Stress-1). 등위 회귀(PAVA)는 이 영역의 내부 커널이다(공개 `isotonic_regression`은 regression의 나중 후보 G126) (추천안, 사용자 위임 2026-11-27) |
| EM33 Sammon | `sammon`은 R `MASS::sammon`의 뜻과 기본값이다(`init='classical_mds'`이 R의 `y = cmdscale(d, k)`, `max_iter=100`(R `niter`), `magic=0.2`, `tol=1e-4`; 걸음의 절반 줄이기도 R과 같다). 같은 거리 0의 짝은 R처럼 `ValueError`다(`sammon(): points 3 and 7 are at distance 0; Sammon's stress divides by every distance`) (추천안, 사용자 위임 2026-11-27) |
| EM34 Isomap | sklearn `Isomap`의 뜻이다: k 최근접 또는 반경 그래프(둘 중 하나), `sd.graph.shortest_path`(Dijkstra, 열쇠 순서의 행; 같은 바이트), 중심화한 −½G²의 고유값(EM18). 그래프가 여러 성분이면 sklearn처럼 성분 사이에 가장 가까운 짝의 변을 더해 잇는다(sklearn `_fix_connected_components`; sd는 경고 대신 결과의 `n_connected_components`가 원래 성분 수를 보인다). `transform`은 sklearn과 같다(새 점의 이웃을 거친 측지 거리와 핵의 사영). `reconstruction_error()`는 sklearn의 메서드다 (추천안, 사용자 위임 2026-11-27) |
| EM35 LLE | sklearn `LocallyLinearEmbedding`의 네 방법(`'standard'`, `'modified'`, `'hessian'`, `'ltsa'`)과 기본값이다(`reg=1e-3`, `hessian_tol=1e-4`, `modified_tol=1e-12`, `tol=1e-6`, `max_iter=100`). 점마다의 국소 풀이는 독립이다(k × k의 작은 Cholesky·고유값은 이 영역의 고정 순서 커널: 점 수만큼 부르는 LAPACK의 호출 비용이 계산보다 크다). sklearn의 `random_state`(ARPACK 시작 벡터)는 없다: `eigsh`의 시작 벡터가 정해져 있다(sparse SP84). `transform`은 sklearn과 같다(무게 중심 무게) (추천안, 사용자 위임 2026-11-27) |
| EM36 스펙트럼 임베딩 | sklearn `SpectralEmbedding`의 뜻이다: `affinity='nearest_neighbors'`(대칭화한 0/1 k-NN, `n_neighbors=None`은 max(n/10, 1)), `'rbf'`(`gamma=None`은 1/p), `'precomputed'`(유사도), `'precomputed_nearest_neighbors'`. 정규화 라플라시안은 `sd.graph.laplacian(normed=True)`, 첫 고유벡터를 버리고 D^{-1/2}로 되돌린다(sklearn `spectral_embedding`의 `norm_laplacian=True`, `drop_first=True`). 연결되지 않은 그래프도 계산하고 결과의 `n_connected_components`가 보인다(sklearn은 경고한다). sklearn의 `eigen_solver='amg'`(pyamg)와 `'lobpcg'`는 없다 (추천안, 사용자 위임 2026-11-27) |
| EM37 t-SNE의 뜻 | sklearn `TSNE`의 뜻과 기본값이다: `perplexity=30`, `early_exaggeration=12`(처음 250걸음, 운동량 0.5, 그 뒤 0.8), `learning_rate='auto'`(max(n/early_exaggeration/4, 50)), `max_iter=1000`, `n_iter_without_progress=300`, `min_grad_norm=1e-7`(50걸음마다 검사), 이득(gains)의 0.2·0.8 갱신과 최소 0.01, 이웃 수 min(n − 1, ⌊3·perplexity + 1⌋), P의 대칭화 (P + Pᵀ)/(2n), `init='pca'`(PCA의 첫 성분 표준편차를 10⁻⁴로; 결정적인 `pca`), `'random'`(N(0, 10⁻⁸), `rng` 필요) 또는 (n, d) 배열, `method='barnes_hut'|'exact'`와 EM8 마일스톤의 `'fft'`, `angle=0.5`. 자유도는 sklearn처럼 max(d − 1, 1)이다. 퍼플렉서티의 이분 찾기는 sklearn `_binary_search_perplexity`(100걸음, 엔트로피 허용오차 10⁻⁵)를 float64로 한다. `'barnes_hut'`은 d ≤ 3, `'fft'`는 d ≤ 2다(sklearn·FIt-SNE와 같다; `ValueError`) (추천안, 사용자 위임 2026-11-27) |
| EM38 Barnes–Hut의 나무 | 걸음마다 Y의 경계 상자(정확한 최솟값·최댓값)에서 Morton 열쇠(축마다 2²¹ 칸, spatial `sfc.hpp`의 비트 끼우기)를 매기고 (열쇠, 점 열쇠)로 기수 정렬한 배열에서 나무를 만든다(삽입 차례에 달리지 않는다). 마디의 질량 중심은 자식 차례의 합이다. 점마다의 반발력은 정해진 자식 차례로 나무를 내려가며 그 점의 연산 차례로 더하고, 정규화 Z = Σ_i Z_i는 크기로만 정한 덩이의 짝 합이다. 그래서 스레드 수와 무관하고 그 비용은 덩이 합 하나다(§3.11). 같은 좌표의 점들은 깊이 상한에서 한 잎이다. sklearn의 `sum_Q`는 OpenMP `prange` 안의 누적이라 스레드 수에 달린다(EM0 탐침이 확인한다) (추천안, 사용자 위임 2026-11-27) |
| EM39 UMAP의 뜻 | 기준 라이브러리는 umap-learn(원본)이다: `n_neighbors=15`, `min_dist=0.1`, `spread=1.0`, `n_epochs=None`(n ≤ 10⁴이면 500, 넘으면 200), `learning_rate=1.0`, `init='spectral'`(정규화 라플라시안의 `eigsh`, 크기를 10으로 늘리고 표준편차 10⁻⁴의 잡음; 실패하면 umap-learn처럼 `'random'`으로 가지 않고 `LinAlgError`), `'random'`(균등 [−10, 10]) 또는 배열, `set_op_mix_ratio=1.0`, `local_connectivity=1.0`, `repulsion_strength=1.0`, `negative_sample_rate=5`, `a`·`b`(None이면 umap-learn `find_ab_params`의 곡선 맞춤: 300점, `sd.optimize.least_squares`). 퍼지 단체 집합(`smooth_knn_dist`의 64걸음 이분 찾기, 허용오차 10⁻⁵, ρ와 σ의 최솟값 규칙)과 퍼지 합집합, 걸음 일정(`epochs_per_sample`), 기울기 자르기 ±4, 학습률의 선형 감소도 umap-learn과 같다. 최적화의 꼴은 EM40이다. `[uwot과 다름]`: uwot은 `min_dist=0.01`이 기본이다 (추천안, 사용자 위임 2026-11-27) |
| EM40 UMAP의 최적화 | umap-learn의 SGD는 변을 하나씩 처리하며 그 자리에서 좌표를 고친다(Gauss–Seidel). 여러 스레드는 Hogwild(잠금 없는 경쟁 쓰기)라 고정 스레드 수에서도 실행마다 다르고(D-33을 깬다), `random_state`를 주면 umap-learn은 1스레드로 돈다. sd는 걸음마다 동기인 꼴(uwot의 `batch=TRUE`와 같은 꼴)이다: 한 걸음의 모든 변의 기울기를 걸음 처음의 좌표에서 재고, 꼭짓점마다 그 꼭짓점에 닿는 변(머리로는 CSR, 꼬리로는 CSC의 정해진 차례)의 기여를 모아 한 번에 고친다(변을 두 끝에서 한 번씩 재어 경쟁 쓰기가 없다; 음의 표본은 머리만 움직인다: umap-learn과 같다). 음의 표본의 꼭짓점은 원소 (걸음, 변, 표본)의 Philox 수다. 그래서 스레드 수와 무관하고 나란하다. 좌표의 갱신 규칙(SGD의 선형 감소 또는 uwot의 Adam: β₁ = 0.5, β₂ = 0.9, ε = 10⁻⁷)은 EM0 마일스톤의 시제품이 품질(trustworthiness, 걸음 수)로 고른다(추천: Adam, uwot이 동기 꼴에 권한다). Hogwild 꼴을 따로 둘지는 §9의 3번이다 (추천안, 사용자 위임 2026-11-27) |
| EM41 UMAP의 새 점 | `UMAPResult.transform(X_new, /, *, rng)`: umap-learn `transform`의 단계(새 점의 k 최근접을 맞춘 점에서 찾고, 같은 퍼지 집합의 무게, 이웃 임베딩의 무게 평균에서 시작, 맞춘 점을 고정한 채 n_epochs/3(n_epochs ≤ 200이면 n_epochs) 걸음의 같은 최적화)다. umap-learn은 `transform_seed=42`로 고정하지만 sd는 `rng`를 받는다(EM11) `[umap-learn과 다름]`. 결과는 자료의 읽기 전용 참조와 이웃 찾기의 나무를 쥔다 (추천안, 사용자 위임 2026-11-27) |
| EM42 품질 지표 | `trustworthiness(X, X_embedded, /, *, n_neighbors=5, metric='euclidean')`는 sklearn과 같은 식(Venna–Kaski)이고, `continuity`는 그 짝(두 공간의 역할을 바꾼 것)이다. 순위의 동점은 (거리, 열쇠)이고 값은 정확한 정수 합을 한 번 반올림한 것이다(cluster CL83의 꼴). `n_neighbors` < n/2(sklearn과 같다). `stress(X, X_embedded, /, *, metric='euclidean', kind='kruskal')`의 `kind`는 `'raw'`(Σ_{i<j}(d̂ − d)²; sklearn `MDS.stress_`의 뜻), `'kruskal'`(Stress-1), `'sammon'`(Sammon의 오차)이다. 셋 다 n × n 행렬 없이 타일로 잰다 (추천안, 사용자 위임 2026-11-27) |
| EM43 경고 없음 | 경고는 내지 않는다(cluster와 같다). 다른 라이브러리가 경고하는 곳(sklearn `MDS`의 `init` 예고, `Isomap`·`SpectralEmbedding`의 연결되지 않은 그래프, `FastICA`·`NMF`·`FactorAnalysis`의 `ConvergenceWarning`, R `cmdscale`의 음의 고유값, umap-learn의 `random_state`가 `n_jobs`를 덮는다는 경고)은 상태 필드(`converged`, `n_connected_components`, `eigenvalues`)이거나 문서의 기본값이다 (추천안, 사용자 위임 2026-11-27) |
| EM44 공용 장치 | cluster의 영역 중립 장치를 EM0 마일스톤이 공용 자리로 꺼낸다: Python의 입력·`nan_policy`·`metric`·`rng`·`X_new`·할당 검사(`cluster/_common.py`의 `points`, `rows`, `metric`, `check_rng`, `no_rng`, `key`, `new_points`, `check_alloc`)는 `stardust/_learn.py`로, C++의 열쇠 순서(`lex_order`), 덩이 짝 합(`pairwise_chunks`), 원소 번호는 `csrc/base/keyorder.hpp`로(CL3 마일스톤이 합집합–찾기를 `graph/unionfind.hpp`로 꺼낸 것과 같다). cluster의 메시지와 골든이 그대로여야 한다(cluster 테스트·골든으로 확인). 나중의 `sd.classify`도 같은 장치를 쓴다 (추천안, 사용자 위임 2026-11-27) |
| EM45 `summary()` | 회귀 상자의 머리(왼쪽: Method, No. Observations, No. Components, 있을 때만 Omitted rows (NaN), 목적값, Iterations, Converged; 오른쪽: 조절 모수)와, 성분이 있는 결과(PCA, 절단 SVD, 인자 분석, 고전 MDS, 커널 PCA)는 성분마다 한 줄의 표(component, 분산 또는 고유값, 비율, 누적 비율; 인자 분석은 고유도의 표를 더한다; 20줄을 넘으면 양 끝 5줄씩)다. 비선형 결과는 표가 없다 (추천안, 사용자 위임 2026-11-27) |
| EM46 참값 | 참값은 정확한 유리수(`fractions.Fraction`, python-flint의 `fmpq`·`fmpq_mat`)와 Arb(python-flint 0.9.0의 `arb_mat`·`acb_mat.eig`)만 쓴다. mpmath는 프로젝트 전체에서 금지다(설치되어 있어도 부르지 않는다). 참값 생성은 프로세스 10개까지, RSS를 보며 돈다 (사용자 규칙, 2026-11-22) |
| EM47 비교의 눈금 | 무작위 시작과 비볼록 목적(t-SNE, UMAP, SMACOF의 `'random'`, FastICA, NMF)은 다른 라이브러리와 좌표가 같을 수 없다. 견주는 것은 (1) 같은 시작을 주었을 때 같은 고정점(SMACOF, NMF의 `'custom'`, FastICA의 `w_init`, t-SNE의 배열 `init`과 `'exact'`), (2) 같은 단계의 같은 양(P 행렬, 퍼지 집합, 핵, 기울기 한 번), (3) 같은 자료에서 품질 지표(trustworthiness, KL 발산, stress)의 분포다 (추천안, 사용자 위임 2026-11-27) |
| EM48 마일스톤의 검증 범위 | 마일스톤마다의 검사는 이 영역만이다(`tools/test.py embed`, `tools/golden.py check embed`, `bench.accuracy run embed`). 전체 테스트·골든은 마지막 마일스톤(EM10)에서 한 번이다(공용 장치를 꺼내는 EM0 마일스톤은 cluster의 테스트·골든도 돈다) (추천안, 사용자 위임 2026-11-27) |

### 1.2 범위
마일스톤 열은 그 이름을 만드는 §8의 마일스톤이고 "나중"은 첫 판에 넣지 않는 후보, "밖"은 이 영역에 두지 않는 것이다. 이름공간은 평평하다(EM5): sklearn이 `decomposition`·`manifold`·`random_projection`에, R이 `stats`·`MASS`·여러 꾸러미에 나눠 둔 것이 한곳에 모인다.

**선형 분해 → sd.embed**
| 다른 라이브러리 | sd.embed | 마일스톤 |
|---|---|---|
| sklearn `PCA(n_components=None, whiten=False, svd_solver='auto')`; R `prcomp(x, center, scale., rank.)`, `princomp(x, cor)`; statsmodels `PCA(data, ncomp, standardize)`; irlba `prcomp_irlba` | `pca(X, n_components=None, /, *, scale=False, whiten=False, svd_solver='auto', …)` → `PCAResult`(EM19–EM21) | EM1 |
| sklearn `TruncatedSVD(n_components=2, algorithm='randomized')`, `randomized_svd`; R `irlba`, `RSpectra::svds` | `truncated_svd(X, n_components=2, /, *, algorithm='lanczos', …)` → `TruncatedSVDResult`(EM22). `randomized_svd`는 함수로 두지 않는다(`algorithm='randomized'`; 행렬 분해 자체는 `sd.linalg.svd`, `sd.sparse.svds`) | EM1 |
| sklearn `GaussianRandomProjection`, `SparseRandomProjection`, `johnson_lindenstrauss_min_dim` | `random_projection(X, n_components='auto', /, *, rng, method=…)` → `RandomProjectionResult`, `johnson_lindenstrauss_min_dim`(EM24) | EM1 |
| sklearn `FactorAnalysis(rotation=None|'varimax'|'quartimax')`; R `factanal(x, factors, rotation='varimax', scores)`; statsmodels `Factor(method='ml'|'pa')` | `factor_analysis(X, n_components, /, *, rotation='varimax', …)` → `FactorAnalysisResult`(EM26–EM28). 주축 인자법(`'pa'`)은 나중 | EM2 |
| R `varimax`, `promax`; statsmodels `rotate_factors` | `rotate_factors(loadings, /, *, method='varimax', …)` → `RotationResult`(EM27). oblimin 등 GPArotation의 회전은 나중 | EM2 |
| sklearn `FastICA`, `fastica`; R `fastICA`(설치 안 됨) | `fastica(X, n_components=None, /, *, rng, …)` → `FastICAResult`(EM29) | EM2 |
| sklearn `NMF`, `non_negative_factorization`; R `NMF`(설치 안 됨) | `nmf(X, n_components='auto', /, *, …)` → `NMFResult`(EM30). `MiniBatchNMF`는 나중 | EM3 |
| sklearn `KernelPCA`; R `kernlab::kpca`(설치 안 됨) | `kernel_pca(X, n_components=None, /, *, kernel='linear', …)` → `KernelPCAResult`(EM25) | EM3 |
| sklearn `IncrementalPCA` | 없음: 함수 → 결과의 꼴에 `partial_fit`의 상태가 없다. 큰 자료는 `svd_solver='lanczos'`·`'randomized'`와 희소 입력. 나중(덩이를 받는 `pca(..., batch_size=)` 꼴을 그때 정한다) | 나중 |
| sklearn `SparsePCA`, `MiniBatchSparsePCA`, `DictionaryLearning`, `SparseCoder`, `sparse_encode`, `dict_learning*` | 나중(희소 부호화는 회귀의 lasso와 묶어 설계해야 한다) | 나중 |
| sklearn `LatentDirichletAllocation` | 밖: 주제 모형은 임베딩이 아니라 확률 모형이다(나중에 따로 정한다) | 밖 |
| R `cancor`, statsmodels `CanCorr`, sklearn `CCA` | 나중: 두 자료 묶음의 방법이라 (X, Y)의 꼴을 따로 정해야 한다 | 나중 |
| sklearn `PLSRegression`, `PLSSVD`, `PLSCanonical` | 밖: 회귀의 나중 후보 PLS(regression G126) | 밖 |
| R `MASS::corresp`, `MASS::mca`, FactoMineR `CA`·`MCA`·`FAMD`(설치 안 됨) | 나중(§9의 1번): 분할표의 SVD라 계산은 작지만 입력(분할표, 범주형 `DataFrame`)과 결과(행·열 좌표, 관성)가 따로 설계할 것이 많다 | 나중 |
| sklearn `LinearDiscriminantAnalysis.transform`, R `MASS::lda` | 밖: `sd.classify`(EM2) | 밖 |

**거리 보존 → sd.embed**
| 다른 라이브러리 | sd.embed | 마일스톤 |
|---|---|---|
| R `cmdscale(d, k=2, eig, add)`; sklearn `ClassicalMDS(n_components=2, metric='euclidean')` | `classical_mds(X, n_components=2, /, *, metric='euclidean', add=False)` → `ClassicalMDSResult`(EM31) | EM4 |
| sklearn `MDS(metric_mds=True, n_init=1, init, max_iter=300, eps=1e-6, normalized_stress='auto')`, `smacof`; R `smacof`(설치 안 됨) | `mds(X, n_components=2, /, *, metric_mds=True, init='classical_mds', …)` → `MDSResult`(EM32) | EM4 |
| R `MASS::isoMDS(d, y=cmdscale(d, k), k=2, maxit=50, tol=1e-3)` | 같은 목적(Kruskal Stress-1)은 `mds(metric_mds=False)`다(SMACOF와 등위 회귀; isoMDS는 최급강하라 다른 국소 최소에 닿을 수 있다). 따로 두지 않는다(대응표) | EM4 |
| R `MASS::sammon(d, y=cmdscale(d, k), k=2, niter=100, magic=0.2, tol=1e-4)` | `sammon(X, n_components=2, /, *, …)` → `MDSResult`(EM33) | EM4 |
| R `vegan::metaMDS`(설치 안 됨) | 밖: 생태학 꾸러미의 여러 시작·회전 묶음이다(`mds`와 `rng`로 같은 일을 한다) | 밖 |

**이웃 그래프 위의 다양체 → sd.embed**
| 다른 라이브러리 | sd.embed | 마일스톤 |
|---|---|---|
| sklearn `Isomap(n_neighbors=5, radius=None, n_components=2, eigen_solver='auto', path_method='auto')`; R `vegan::isomap`, `dimRed`(설치 안 됨) | `isomap(X, n_components=2, /, *, n_neighbors=5, radius=None, …)` → `IsomapResult`(EM34) | EM5 |
| sklearn `LocallyLinearEmbedding(method='standard'|'modified'|'hessian'|'ltsa')`, `locally_linear_embedding` | `locally_linear_embedding(X, n_components=2, /, *, method='standard', …)` → `LLEResult`(EM35) | EM5 |
| sklearn `SpectralEmbedding(affinity='nearest_neighbors')`, `spectral_embedding` | `spectral_embedding(X, n_components=2, /, *, affinity='nearest_neighbors', …)` → `SpectralEmbeddingResult`(EM36) | EM5 |
| R `diffusionMap`, `destiny`(설치 안 됨) | 나중(§9의 1번): 스펙트럼 임베딩의 확산 시간 t와 α 정규화 변형이다 | 나중 |
| sklearn `trustworthiness`; R `coRanking`(설치 안 됨) | `trustworthiness`, `continuity`, `stress`(EM42). 공동 순위 행렬과 R_NX(K) 곡선은 나중 | EM5, EM4 |

**확률적 이웃 임베딩 → sd.embed**
| 다른 라이브러리 | sd.embed | 마일스톤 |
|---|---|---|
| sklearn `TSNE(method='barnes_hut'|'exact')`; R `Rtsne`(설치 안 됨); openTSNE(설치 안 됨) | `tsne(X, n_components=2, /, *, perplexity=30.0, method='barnes_hut', …)` → `TSNEResult`(EM37, EM38) | EM6 |
| FIt-SNE(openTSNE의 `negative_gradient_method='fft'`) | `tsne(method='fft')`(§3.8; §9의 1번) | EM8 |
| openTSNE `TSNEEmbedding.transform`(새 점) | 나중: sklearn에 없고 비교 대상이 설치되지 않았다 | 나중 |
| umap-learn `UMAP`; R `uwot::umap`(설치 안 됨) | `umap(X, n_components=2, /, *, rng, n_neighbors=15, min_dist=0.1, …)` → `UMAPResult`(`transform`)(EM39–EM41) | EM7 |
| umap-learn `inverse_transform`, 지도·반지도 UMAP(`y`), densMAP, `ParametricUMAP`, 정렬 UMAP | 나중(파라메트릭은 신경망이 필요해 밖) | 나중 |
| pynndescent `NNDescent` | `neighbors='nndescent'`(이웃 그래프 함수의 옵션; §3.9, §9의 1·2번). 공개 근사 이웃 함수는 두지 않는다(나중에 둔다면 `sd.spatial`의 것) | EM8 |
| pacmap `PaCMAP`, trimap `TRIMAP`(설치 안 됨) | 나중(§9의 1번) | 나중 |
| R `Rdimtools`, `dimRed`의 그 밖의 방법(LTSA 말고 MVU, KECA 등) | 밖 | 밖 |

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다. 별칭이 없다(`method`, `metric`, `init`, `kernel`, `svd_solver`, `eigen_solver`의 문자열은 소문자 정식 이름만).
- **같은 기계, 같은 스레드 수, 같은 입력이면 실행마다 같은 바이트다**(D-33, API.md §3; EM4). 스레드 수와 무관한 바이트는 값싼 곳에서 지키고(대부분의 길이 그렇다: §3.11), 테스트와 골든은 길마다 그 약속(1·3·10스레드의 같은 바이트, 또는 D-33의 길은 스레드 수마다 두 번 돌린 같은 바이트)을 확인한다. 길을 고르는 기준은 모양(n, p, d, k)뿐이고 값이나 스레드 수가 아니다.
- **결과를 알고리즘의 우연이 아니라 정의로 정한다**: 부호(EM14), 성분의 차례(EM29), 이웃의 동점(EM17), 입력 순서(EM12)를 문서에 적는다. 임베딩은 좌표의 회전·반사에 자유로운 문제가 많아서, 정하지 않으면 같은 자료가 실행·판·풀이마다 다른 그림이 된다.
- 반올림이 아닌 동작(기본값, 멈춤 규칙, 반복 수, 결과의 모양)은 함수마다 기준 라이브러리 하나와 같다(API.md §4의 규칙 2): 선형 분해·다양체·t-SNE·SMACOF·지표는 scikit-learn 1.9.1, 인자 분석·회전·Sammon·고전 MDS의 `add`는 R(`stats`, `MASS`), UMAP은 umap-learn이다. 다르게 정하는 곳은 `[sklearn과 다름]`, `[R과 다름]`, `[umap-learn과 다름]`으로 이유와 함께 적는다(§2.8).
- 조용히 어림하지 않는다(EM15). 무작위 어림은 사용자가 이름으로 고르고 그때 `rng`를 준다.
- 경고를 내지 않는다(EM43). 에러 종류와 메시지는 API.md §2·§2.1이고 부른 함수의 이름으로 시작한다(`pca(): ...`, 결과의 메서드는 `transform(): ...`).
- 참값은 정확한 유리수와 Arb만이다(EM46). 모든 호출은 정해진 시간 안에 끝난다: 반복이 있는 곳은 입력과 무관한 상한(`max_iter`, `n_epochs`)이 있고 상한에서 멈추면 `converged=False`이며 에러가 아니다(API.md §2.2).

### 1.4 다른 영역과의 관계
- **linalg**: 밀집 SVD·대칭 고유값·QR·일반 고유값(Cailliez)은 linalg의 LAPACK 래퍼(`csrc/linalg/lapack.hpp`)를 C++에서 부른다(같은 드라이버: `gesdd`, `syevd`, `geqrf`, `geev`). 실패는 `LinAlgError`(메시지는 부른 함수의 이름). 공개 `sd.linalg.svd`와 같은 바이트일 필요는 없다(중심화·부호 규칙이 더해진다).
- **sparse**: `svds`(Baglama–Reichel, 정해진 시작 벡터), `eigsh`(ARPACK 이식, shift-invert의 Accelerate LU/AMD), CSR·SpMV, `LinearOperator`(암묵적 중심화)를 쓴다. 희소 결과(그래프, 무작위 투영의 성분)는 `sd.sparse.csr_array`다.
- **spatial**: 거리의 정의와 계산, k 최근접과 반경 이웃(`kneighbors`, `radius_neighbors`, KD 트리)은 spatial의 것이고 같은 바이트다. Morton·Hilbert 열쇠(`sfc.hpp`)를 Barnes–Hut 나무에 쓴다.
- **graph**: Isomap의 최단 경로(`shortest_path`, Dijkstra), 스펙트럼 임베딩의 라플라시안(`laplacian`), 연결 요소(`connected_components`)를 C++에서 부른다. graph는 이 영역을 모른다.
- **fft**: FIt-SNE의 격자 합성곱은 sd.fft의 자체 Stockham(결정적; vDSP를 쓰지 않는다)이다.
- **optimize**: 인자 분석의 L-BFGS-B(`minimize(method='L-BFGS-B')`, bounds)와 UMAP의 a·b(`least_squares`)를 쓴다. 같은 멈춤 규칙이다.
- **random·stats**: `rng`는 cluster CL41의 규칙이다(EM11). 정규 수는 stats의 ziggurat(`sample_dist.hpp`)다.
- **cluster**: 공용 장치를 나눠 쓴다(EM44). 스펙트럼 군집(CL39의 나중 후보)은 `spectral_embedding` 위에 `kmeans`를 얹은 것이 된다. 군집 지표와 임베딩 지표는 겹치지 않는다(임베딩 지표는 이 영역).
- **plot**: 임베딩의 그림은 `sd.plot.scatter(Z[:, 0], Z[:, 1], color=labels)`가 이미 한다. 이 영역은 그리지 않는다.
- **classify(나중)**: LDA의 투영, 커널 이름(EM25), k-NN 그래프(spatial)를 공유한다.

---

## 2. API
시그니처와 모든 규칙(인자의 검사, 에러와 메시지, 결과의 필드)은 [docs/api/embed.md](../api/embed.md)에 있다. 이 절은 꼴과 그 까닭을 적는다.

### 2.1 하나의 꼴 (EM1, EM5–EM7)
- **맞추는 함수는 소문자 이름이고 `<이름>Result`를 돌려준다**: `pca` → `PCAResult`, `tsne` → `TSNEResult`, `umap` → `UMAPResult`, `mds`·`sammon` → `MDSResult`. cluster의 `kmeans` → `KMeansResult`, 회귀의 `ols` → `OLSResult`와 같다.
- **공통 필드**(EM7): `embedding`, `n_components`, `nobs`, `omitted`, `summary()`, 한 줄 `repr`(`PCAResult(nobs=300, n_components=2, explained_variance_ratio=[0.73, 0.23])`). sklearn의 `fit_transform(X)`이 `embedding`이고 `fit(X).transform(X)`와 같은 값이다(선형 방법에서 sklearn과 같다; Isomap·LLE의 `transform(X)`는 sklearn처럼 맞춘 점에서 `embedding`과 다를 수 있다).
- **값만 돌려주는 함수**: `johnson_lindenstrauss_min_dim`(int 또는 배열), `trustworthiness`·`continuity`·`stress`(Python `float`), `rotate_factors`(namedtuple).
- **새 점**: `transform(X_new, /)`(선형 방법, 커널 PCA, 인자 분석, Isomap, LLE, UMAP), `inverse_transform(Z, /)`(PCA, 절단 SVD, 무작위 투영, FastICA, NMF, `fit_inverse_transform=True`의 커널 PCA). 고전·SMACOF MDS, Sammon, 스펙트럼 임베딩, t-SNE는 sklearn처럼 새 점을 옮기지 않는다(Nyström 꼴은 나중).

### 2.2 선형 분해 (EM1–EM3 마일스톤)
- `pca`는 sklearn의 필드 이름(`components`, `explained_variance`, `explained_variance_ratio`, `singular_values`, `mean`, `noise_variance`)이고 R의 이름은 대응표다(`sdev` = √`explained_variance`, `rotation` = `components`ᵀ, `x` = `embedding`, `center` = `mean`, `scale` = `scale`). statsmodels의 `standardize=True`(기본)는 `scale=True`다.
- `factor_analysis`는 R `factanal`의 필드(`loadings`, `uniquenesses`, `statistic`, `dof`, `pvalue`, `rotation`)와 sklearn의 필드(`components`, `noise_variance`, `mean`, `loglik`)를 함께 준다(EM26). 검정은 R과 같은 Bartlett 보정의 χ²다(자유도가 0 이하면 NaN).

### 2.3 거리 보존 (EM4 마일스톤)
- `classical_mds`는 고유값 전부(`eigenvalues`, n개)를 준다(R `eig=TRUE`; 음수도). `mds`와 `sammon`은 `stress`(= `objective`)와 `n_iter`, `converged`다. `mds`의 비계량은 `disparities`(등위 회귀의 값, 압축 꼴)를 더 준다.

### 2.4 이웃 그래프 (EM5 마일스톤)
- `graph`(희소 대칭 무게 행렬; 스펙트럼은 `affinity_matrix`)와 `n_connected_components`를 준다. Isomap의 측지 거리 행렬(`dist_matrix`, n × n)은 게으른 필드다(sklearn `dist_matrix_`).

### 2.5 t-SNE와 UMAP (EM6–EM8 마일스톤)
- `TSNEResult`: `kl_divergence`(= `objective`), `n_iter`, `converged`(`n_iter_without_progress`나 `min_grad_norm`에서 멈췄으면 True, `max_iter`면 False; sklearn에는 없는 필드), `learning_rate`(쓴 값), `affinities`(대칭 P, `csr_array`).
- `UMAPResult`: `graph`(퍼지 단체 집합, `csr_array`), `a`, `b`, `n_epochs`, `sigmas`, `rhos`, `transform(X_new, /, *, rng)`.

### 2.6 지표 (EM4·EM5 마일스톤)
- sklearn의 이름과 인자(`X`, `X_embedded`, `n_neighbors`, `metric`)다(cluster CL14의 원칙).

### 2.7 결과 객체
| 이름 | 만드는 함수 | 이 결과만의 필드와 메서드 |
|---|---|---|
| `PCAResult` | `pca` | `components`, `explained_variance`, `explained_variance_ratio`, `singular_values`, `mean`, `scale`, `noise_variance`, `n_features`, `svd_solver`(쓴 풀이), `whiten`, `covariance`, `precision`, `transform`, `inverse_transform`, `score_samples` |
| `TruncatedSVDResult` | `truncated_svd` | `components`, `singular_values`, `explained_variance`, `explained_variance_ratio`, `algorithm`, `transform`, `inverse_transform` |
| `RandomProjectionResult` | `random_projection` | `components`, `method`, `density`, `inverse_components`, `transform`, `inverse_transform` |
| `FactorAnalysisResult` | `factor_analysis` | `loadings`, `uniquenesses`, `components`, `noise_variance`, `mean`, `scale`, `rotation`, `rotation_matrix`, `phi`, `loglik`, `statistic`, `dof`, `pvalue`, `n_iter`, `converged`, `transform` |
| `FastICAResult` | `fastica` | `components`, `mixing`, `mean`, `whitening`, `n_iter`, `converged`, `transform`, `inverse_transform` |
| `NMFResult` | `nmf` | `components`(H), `reconstruction_err`, `n_iter`, `converged`, `solver`, `beta_loss`, `transform`, `inverse_transform` |
| `KernelPCAResult` | `kernel_pca` | `eigenvalues`, `eigenvectors`, `kernel`, `gamma`, `transform`, `inverse_transform` |
| `ClassicalMDSResult` | `classical_mds` | `eigenvalues`, `gof`, `additive_constant`, `metric` |
| `MDSResult` | `mds`, `sammon` | `stress`, `disparities`, `n_iter`, `converged`, `n_init`, `best_init`, `method`, `metric` |
| `IsomapResult` | `isomap` | `eigenvalues`, `graph`, `dist_matrix`, `n_connected_components`, `reconstruction_error()`, `transform` |
| `LLEResult` | `locally_linear_embedding` | `reconstruction_error`, `method`, `graph`(W), `transform` |
| `SpectralEmbeddingResult` | `spectral_embedding` | `eigenvalues`, `affinity_matrix`, `n_connected_components`, `affinity` |
| `TSNEResult` | `tsne` | `kl_divergence`, `n_iter`, `converged`, `learning_rate`, `affinities`, `perplexity`, `method` |
| `UMAPResult` | `umap` | `graph`, `a`, `b`, `n_epochs`, `sigmas`, `rhos`, `transform` |
| `RotationResult` | `rotate_factors` | namedtuple (`loadings`, `rotation`, `phi`) |

### 2.8 다른 라이브러리와 다른 곳 (모음)
| 곳 | 다른 라이브러리 | sd |
|---|---|---|
| 꼴 | sklearn 추정기, R 목록 | 함수 → 결과 객체(EM1) |
| `'auto'` 풀이 | sklearn `PCA`가 큰 입력에서 무작위 어림 | 어림을 고르지 않는다(EM15) |
| 절단 SVD의 기본 | sklearn `'randomized'`, n_iter = 5 | `'lanczos'`(EM22) |
| 거듭제곱 반복의 이름 | sklearn `iterated_power`·`n_iter` | `n_power_iter`(EM21) |
| 성분의 부호 | R은 정하지 않음 | sklearn의 규칙과 같고 모든 고유 방법에 쓴다(EM14) |
| FastICA의 차례 | 반복이 끝난 차례 | 첨도의 크기 차례(EM29) |
| NNDSVD의 SVD | sklearn `randomized_svd` | 결정적인 Lanczos(EM30) |
| MDS의 `init` | sklearn 1.9.1 `'random'`(경고) | `'classical_mds'`(EM32) |
| t-SNE의 dtype | sklearn float32 | float64(EM13) |
| t-SNE의 합 | sklearn의 `sum_Q`가 스레드에 달림 | 스레드와 무관(EM38) |
| UMAP의 SGD | umap-learn의 Hogwild(여러 스레드), `random_state`면 1스레드 | 걸음마다 동기, 나란히 하고 재현된다(EM40) |
| UMAP `transform`의 난수 | `transform_seed=42` | `rng`(EM41) |
| UMAP의 시작 실패 | umap-learn은 무작위 시작으로 넘어가며 경고 | `LinAlgError`(EM39) |
| 연결되지 않은 그래프 | sklearn 경고 | 상태 필드 `n_connected_components`(EM34, EM36) |
| 거리 0의 Sammon | R 에러 | 같은 `ValueError`(EM33) |
| 입력 순서 | sklearn·R은 섞으면 다른 결과(동점, 난수의 차례) | 같은 바이트를 그 순서로(EM12) |

---

## 3. 수치 방법

### 3.1 중심화와 Gram
열쇠 순서의 사본(EM12)에서 열 평균은 4096행 덩이의 부분합을 덩이 순서의 짝 합으로 합친다(cluster CL48; 같은 바이트). 가운데로 옮긴 사본 X_c에서 `'full'`은 `gesdd`(Accelerate), `'covariance_eigh'`는 Gram C = X_cᵀX_c(`dsyrk`, regression과 같은 문턱: 열 32개까지는 열 짝마다 `ddot`)와 `syevd`다. 큰 n의 Gram은 크기로만 정한 행 덩이(2¹⁶행)의 `dsyrk`를 덩이 순서로 더한다: sd 스레드 수와 무관하다. 덩이를 스레드마다 나란히 부르는 쪽(덩이 수 = 스레드 수)이 잴 수 있게 빠르면 D-33으로 물러선다(EM16; regression의 측정은 `dsyrk`가 나눠도 빨라지지 않았다).

### 3.2 Lanczos와 무작위 SVD
- `'lanczos'`는 `sd.sparse.svds`(Baglama–Reichel)를 연산자 A = X − 1μᵀ에 부른다(Av = Xv − (μᵀv)1, Aᵀu = Xᵀu − μ(1ᵀu)). 희소 X는 그대로 둔다. `tol`·`max_iter`는 `svds`의 뜻이다.
- `'randomized'`: Y = A Ω(Ω는 p × (d + `n_oversamples`)), 거듭제곱 반복 `n_power_iter`번(sklearn의 `'auto'`: d < 0.1·min(n, p)이면 7, 아니면 4), 정규화(LU 또는 QR), Q의 QR, B = QᵀA의 `gesdd`. 모두 Accelerate의 한 번씩의 호출이라 sd 스레드 수와 무관하다.

### 3.3 인자 분석, FastICA, NMF
- 인자 분석(EM26): 목적 f(ψ) = Σ_{j>k} (log λ_j − λ_j) + k − p(λ는 Ψ^{-1/2}RΨ^{-1/2}의 고유값, 큰 k개를 뺀 나머지)과 그 기울기(R `FAgr`)를 C++에서 재고(p × p `syevd`), `sd.optimize.minimize`의 L-BFGS-B가 [`lower`, 1]에서 최소화한다(R의 `control$opt` 기본: `maxit` 제한 없음 대신 `max_iter=1000`). 고유값 문제의 크기가 p라 호출당 비용이 작다.
- FastICA(EM29): 희게 하기(SVD 또는 eigh), 대칭 비상관화 W ← (WWᵀ)^{-1/2}W(작은 `syevd`), 대비 함수의 기댓값은 열쇠 순서 덩이 합이다. 멈춤은 sklearn의 max|diag(W_newWᵀ)| − 1| < tol이다.
- NMF(EM30): CD는 sklearn `_update_cdnmf_fast`의 성분 차례와 같은 갱신(WᵀW, XHᵀ는 Accelerate), 행마다 독립이라 행 덩이로 나란하다. MU는 행렬 곱과 원소별 갱신이다. 멈춤은 sklearn의 상대 오차 변화(10걸음마다)다.

### 3.4 커널 PCA
핵 행렬 K(n × n, 열쇠 순서 타일; 메모리 8n²가 물리 메모리를 넘으면 `MemoryError`), 이중 중심화(덩이 합), 고유값(EM18). `transform`은 새 점과 맞춘 점의 핵 행(m × n 타일)을 같은 중심화로 사영한다.

### 3.5 MDS
- 고전(EM31): D²(관측값의 유클리드는 spatial의 제곱 거리 커널, 같은 바이트)의 이중 중심화, 큰 d개의 고유쌍.
- SMACOF(EM32): Guttman 변환 Z ← n⁻¹B(Z)Z의 행 i는 그 점의 정해진 j 차례 합이라 점마다 독립이고 나란하다. 거리 행렬은 압축 꼴(8·n(n − 1)/2 바이트)을 한 번 재어 쥔다. stress는 덩이 합이다. 비계량의 등위 회귀는 불일치도의 정렬(정확한 동점은 sklearn처럼 Kruskal의 첫째 방법) 뒤 PAVA(차례; O(n²))이고 그 정렬은 한 번만 한다(δ는 바뀌지 않는다).
- Sammon(EM33): R의 대각 뉴턴 걸음(`magic`), stress가 늘면 걸음을 절반으로(R과 같다; 걸음 절반 줄이기 10번).

### 3.6 이웃 그래프 위의 방법
- k-NN(EM17): spatial의 길(KD 트리는 p ≤ 16, 그 밖은 GEMM 고르기; spatial의 문턱). 대칭화(최댓값 또는 합집합)는 CSR 전치와 열쇠 차례의 병합이다.
- Isomap(EM34): 원천마다 Dijkstra(graph; 원천 덩이로 나란히), G²의 이중 중심화(n × n), 고유값.
- LLE(EM35): 점마다 국소 Gram(k × k)에 `reg`·tr을 더한 풀이(고정 순서 Cholesky), 희소 W, M = (I − W)ᵀ(I − W)(희소 곱), 가장 작은 고유값(σ = 0 근처의 shift-invert `eigsh` 또는 밀집 `syevd`). modified·hessian·ltsa는 sklearn의 국소 단계를 같은 식으로 옮긴다.
- 스펙트럼(EM36): 유사도 행렬, `laplacian(normed=True)`, 가장 작은 d + 1개의 고유쌍.

### 3.7 t-SNE
- P: 이웃 수 k = min(n − 1, ⌊3·perplexity + 1⌋)의 k-NN 거리 제곱에서 점마다 β의 이분 찾기(독립, 나란히), 조건부 P의 CSR, (P + Pᵀ)/(2n)(전치·병합), 합은 덩이 짝 합.
- 끌림: P의 행마다 그 점의 이웃 차례 합(점마다 독립).
- 밀어냄: `'exact'`는 점마다 j의 차례 합(O(n²d), 타일), `'barnes_hut'`는 EM38의 나무(2차원은 사분, 3차원은 팔분 나무; 이 영역의 C++), `'fft'`는 §3.8이다.
- 걸음: 운동량과 이득은 원소별이라 결정적이다. KL 발산은 sklearn처럼 50걸음마다(나머지는 재지 않는다).

### 3.8 FIt-SNE (`method='fft'`, EM8 마일스톤)
Linderman 외(2019)의 다항 보간: 임베딩을 상자 격자(1차원·2차원, 상자마다 보간점 3개, 상자 수는 FIt-SNE의 `min_num_intervals=50`과 `intervals_per_integer=1`의 규칙)로 나눠, 점의 전하를 상자의 보간점에 퍼뜨리고, 격자의 핵 합성곱을 sd.fft로 하고, 다시 점에 보간한다. 퍼뜨리기는 원본(FIt-SNE·openTSNE)에서 점마다 격자에 더하는 경쟁 쓰기다. sd는 점을 상자 번호로 셈 정렬(같은 상자 안은 열쇠 차례)하고 상자마다 제 점들을 차례로 모은다(모으기; 상자끼리 독립이라 나란하다). 그래서 스레드 수와 무관하다. 셈 정렬은 걸음마다 O(n)이라 FFT보다 작다(어림 5–15%; EM8 마일스톤이 잰다). 크면 스레드마다의 격자를 스레드 차례로 더하는 D-33의 길로 물러선다(EM4). 보간의 매개(보간점 수, 상자 규칙)는 공개하지 않는다(openTSNE의 기본값 그대로).

### 3.9 NN-descent (`neighbors='nndescent'`, EM8 마일스톤)
Dong 외(2011)와 pynndescent의 단계: 시작은 무작위 투영 나무의 잎(초평면은 원소 번호의 Philox 정규 수; 나무마다 나란히) 또는 무작위 이웃, 라운드마다 각 점의 새·옛 후보(역이웃은 CSR 전치의 열쇠 차례, 표본은 원소 (라운드, 점, 자리)의 Philox 수), 국소 결합으로 후보 짝 (u, v, d)를 낸다. 원본은 여러 스레드가 이웃 힙을 잠그며 고친다(힙의 차례에 따라 결과가 달라진다). sd는 라운드 동기다: 라운드의 모든 후보를 받는 점별로 모으고, 새 이웃 목록 = (옛 목록 ∪ 받은 후보)에서 (거리, 열쇠)로 앞 k개다. 그 집합은 후보를 모은 차례와 무관하므로 스레드 수와 무관하다. 멈춤은 고친 횟수(정확한 정수) < δ·n·k(pynndescent의 δ = 0.001)와 라운드 상한이다. 비용은 후보 버퍼(라운드마다 O(n·k·ρk))와 점별 모으기의 정렬이고, 원본의 잠금 힙보다 라운드마다 1.2–2배 어림이다(EM8 마일스톤이 잰다). 거리는 spatial의 짝 커널이라 `metric` 이름이 같다.

### 3.10 UMAP
- 그래프(EM39): k-NN, 점마다 ρ(가장 가까운 0 아닌 거리, `local_connectivity`의 보간)와 σ의 이분 찾기, 무게 exp(−(d − ρ)/σ), 퍼지 합집합 A + Aᵀ − A∘Aᵀ(`set_op_mix_ratio`로 교집합과 섞기). 같은 점의 무게 규칙도 umap-learn과 같다.
- 시작: `'spectral'`은 정규화 라플라시안의 `eigsh`(성분이 여럿이면 umap-learn처럼 성분마다 메타 임베딩; EM7 마일스톤이 그 세부를 옮긴다).
- 최적화(EM40): 걸음 e에서 `epochs_per_sample`로 정해진 변들을 고르고(결정적), 꼭짓점 i의 기울기 g_i = Σ(머리가 i인 변의 끌림 + 그 변의 음의 표본 밀어냄) + Σ(꼬리가 i인 변의 끌림의 반대)를 정해진 차례로 더한 뒤 좌표를 한 번에 고친다. 변의 끌림은 두 끝에서 한 번씩 재므로 계산이 끌림 몫만큼 늘지만(어림 1.3배) 쓰기 경쟁과 변 버퍼가 없다. 메모리는 기울기 n·d와 Adam의 두 모멘트 2·n·d다.

### 3.11 병렬화와 결정성 (EM4)
**약속은 D-33이고, 스레드 수와 무관한 바이트는 값싼 곳에서 지킨다.** 표의 "무관"은 1·3·10스레드에서 같은 바이트(테스트와 골든이 확인), "D-33"은 같은 스레드 수에서 실행마다 같은 바이트다.

| 경로 | 다른 라이브러리의 비결정 | sd의 길 | 스레드와 무관하게 하는 비용(어림) | 정한 것 |
|---|---|---|---|---|
| 열 평균, 합, stress, KL, Z | numpy의 짝 합(결정적), sklearn OpenMP 누적(스레드에 달림) | 크기로만 정한 덩이의 짝 합 | 0 | 무관 |
| Gram, 밀집 SVD·eigh·QR, 무작위 SVD의 곱 | Accelerate·OpenBLAS의 스레드 모드 | Accelerate를 크기로만 나눠 부름(EM16) | 덩이를 스레드마다 나란히 부르는 쪽이 빠르면 그만큼 | 무관(Accelerate 모드가 같을 때); 나란한 덩이가 빠르면 D-33 |
| 무작위 행렬, 시작, 음의 표본 | numpy `RandomState`의 차례 소비 | Philox 원소 번호(EM11) | 0 | 무관 |
| Lanczos(`svds`, `eigsh`) | scipy의 v0 = 무작위 | 정해진 v0(sparse SP84) | 0 | sparse의 약속을 따른다 |
| k-NN(정확) | sklearn은 결정적 | spatial(같은 바이트) | 0 | 무관 |
| 이분 찾기(퍼플렉서티, σ·ρ), LLE의 국소 풀이 | 점마다 독립 | 점 덩이 | 0 | 무관 |
| SMACOF의 Guttman 변환, 고전 MDS의 중심화 | sklearn numpy(결정적) | 점마다의 정해진 j 차례 | 0 | 무관 |
| NMF CD, FastICA의 갱신 | sklearn(결정적) | 행 덩이와 Accelerate의 곱 | 0 | 무관(곱은 Accelerate 모드) |
| Dijkstra(Isomap) | scipy 차례 | 원천 덩이 | 0 | 무관 |
| t-SNE 끌림 | sklearn 점마다 | 점마다의 행 차례 | 0 | 무관 |
| t-SNE Barnes–Hut | sklearn의 `sum_Q` 누적(스레드), 삽입 차례의 나무 | 정렬한 Morton 열쇠의 나무, 점마다의 차례, 덩이 합(EM38) | 걸음마다 O(n log n)의 정렬(sklearn도 걸음마다 나무를 만든다): 0–5% | 무관 |
| t-SNE FFT 퍼뜨리기 | 경쟁 쓰기(원본) | 상자 셈 정렬과 모으기(§3.8) | 5–15% | 무관; 잰 비용이 크면 스레드 격자의 D-33 |
| NN-descent | 잠금 힙(차례에 달림) | 라운드 동기, 집합의 앞 k(§3.9) | 라운드마다 1.2–2배 | 무관이 기본; 원본 꼴은 D-33조차 깨므로 두지 않는다 |
| UMAP SGD | Hogwild: 고정 스레드에서도 실행마다 다름 | 걸음마다 동기(EM40) | 끌림을 두 번: 1.3배 어림, 대신 나란함 | 무관. Hogwild는 D-33을 깨므로 두지 않는다(§9의 3번) |

- **난수의 하위 흐름**(cluster CL41과 같다): 호출마다 두 단어, 흐름 r의 t번째 수는 원소 r·2⁴⁰ + t. 흐름 번호는 함수마다 API 문서와 §3에 적는다(예: UMAP 음의 표본 r = 1 + 걸음, t = 변의 열쇠 차례 번호 × `negative_sample_rate` + s; 시작의 잡음 r = 0).
- 병렬 기준값은 배열의 규칙이다(float64 연산 2¹⁵개 분량 이상이면 나누고 GIL을 놓는다; API.md §3).

### 3.12 메모리
- n × n이 드는 곳: 커널 PCA·고전 MDS·Isomap의 핵(8n²), SMACOF·Sammon의 압축 꼴 거리(4n²), `'exact'` t-SNE(타일이라 행렬은 없다). 할당 전에 물리 메모리를 넘으면 `MemoryError`(cluster CL94의 꼴). Isomap의 `dist_matrix`는 게으른 필드다.
- 이웃 그래프 방법은 O(nk), Barnes–Hut은 O(n), FFT는 격자(상자 수² × 9)다.

---

## 4. 정확도 계약 (초안; 마일스톤마다 증명과 측정으로 고친다)
결정적인 정의가 있는 양은 참값과 견준 한계이고, 비볼록 반복의 끝 좌표는 계약이 없다(정해진 상태에서 한 걸음의 양과 품질 지표로 지킨다; EM47).

### 4.1 선형 분해
- PCA·절단 SVD: 특이값 |σ̂ − σ| ≤ C_SVD·u·√(min(n, p))·σ₁(`'full'`, `'lanczos'`는 `svds`의 계약 C_EIG·ε·‖A‖₂), `'covariance_eigh'`의 분산 |λ̂ − λ| ≤ C·u·n·λ₁(제곱의 손실; 작은 분산의 상대 정확도는 약속하지 않는다), 성분 sin∠(v̂, v) ≤ 계약·σ₁²/간격(Davis–Kahan 꼴). 참값은 가운데로 옮긴 자료의 정확한 유리수 Gram(`fmpq_mat`)과 Arb의 고유값(`acb_mat.eig`)이다. `'randomized'`는 계약이 없다(어림; 잰 오차를 기록한다).
- 인자 분석: 주어진 ψ에서 목적과 기울기(Arb), 끝의 ψ는 KKT 잔차와 R `factanal`과의 차이(같은 고정점)로 지킨다.
- FastICA·NMF: 주어진 W·H에서 한 걸음의 값(유리수)과 끝의 고정점 조건.
- 무작위 투영: 성분의 원소가 Philox 원소의 함수(같은 비트)이고 곱은 곱의 계약이다.

### 4.2 MDS
- 고전: B의 정확한 유리수(유클리드 제곱 거리)와 Arb의 고유쌍.
- SMACOF·Sammon: 주어진 Z에서 Guttman 변환 한 번과 stress(유리수·Arb). 끝 좌표는 sklearn·R과 같은 시작에서 같은 고정점 근처(EM47).

### 4.3 이웃 그래프와 t-SNE·UMAP
- k-NN의 집합은 정의(spatial)다. 이분 찾기의 β·σ·ρ는 정확한 목적의 근과 이분의 허용오차 안이다(Arb의 엔트로피). P·퍼지 집합의 원소는 Arb와 견준 상대 γ_k 꼴이다.
- t-SNE `'exact'`의 기울기(주어진 Y)는 Arb와 견준 한계 안이고, Barnes–Hut·FFT의 기울기는 같은 근사식의 Arb 값과 견준다(근사 자체의 오차는 계약이 아니라 기록이다: θ = 0.5에서 상대 오차).
- 품질: trustworthiness와 KL 발산이 sklearn·(설치되면) openTSNE·umap-learn의 같은 자료 분포 안이다(EM47).

### 4.4 지표
- trustworthiness·continuity는 정확한 정수 순위 합의 한 번 반올림(올바른 반올림), stress는 γ_n 꼴의 합 계약이다.

---

## 5. 성능
**성능이 먼저이고 정확도가 그다음이다**(API.md §4의 규칙 3). 목표는 1스레드에서 비교 대상과 같거나 빠르고 10스레드에서 그보다 여러 배 빠른 것이다. 아래 비교 대상의 시간은 EM0 마일스톤의 탐침이 잰다(Apple M4 Pro; sklearn은 OpenMP 기본 스레드, R은 1스레드; 속도 사례는 비교 대상을 먼저 불러온다: cluster CL103).

| 사례 | 비교 대상 | sd 목표 (1 / 10스레드) |
|---|---|---|
| `pca` n = 10⁵, p = 100, d = 10, 각 풀이 | sklearn `PCA`(full, covariance_eigh, arpack, randomized), R `prcomp`, `prcomp_irlba` | 같은 풀이의 sklearn 이하 / 1/3 이하 |
| `truncated_svd` 희소 10⁵ × 10⁴, 밀도 10⁻³, d = 100 | sklearn (randomized, arpack), R `irlba` | sklearn arpack 이하 / 1/3 이하 |
| `factor_analysis` n = 10⁴, p = 50, d = 5 | R `factanal`, sklearn | R 이하 / R의 1/3 이하 |
| `fastica` n = 10⁵, p = 20 | sklearn | sklearn 이하 / 1/3 이하 |
| `nmf` 10⁴ × 10³, d = 20, cd | sklearn | sklearn 이하 / 1/3 이하 |
| `kernel_pca` rbf n = 5000 | sklearn | sklearn 이하 / 1/3 이하 |
| `mds` n = 2000 (SMACOF), `sammon` n = 1000 | sklearn `MDS`, R `sammon` | 같은 반복 수에서 1/2 / 1/8 |
| `isomap` n = 10⁴, `locally_linear_embedding` n = 10⁴ | sklearn | sklearn 이하 / 1/3 이하 |
| `spectral_embedding` n = 2×10⁴ | sklearn(arpack) | sklearn 이하 / 1/3 이하 |
| `tsne` n = 2×10⁴, p = 50, BH | sklearn `TSNE` | sklearn의 1/2 / 1/8 |
| `tsne` n = 10⁵, `'fft'` | (openTSNE: 설치되면) | sklearn BH의 1/5 / 1/20 |
| `umap` n = 7×10⁴, p = 50 | (umap-learn·uwot: 설치되면) | umap-learn 1스레드(결정적 판) 이하 / 1/5 |
| `trustworthiness` n = 2×10⁴ | sklearn | sklearn의 1/2 / 1/8 |

- **EM0 마일스톤의 시제품**(측정만, 공개하지 않는다): (1) Gram·SVD의 Accelerate 대 자체 NEON의 1·10스레드 비(EM16의 어림을 잰다), (2) 덩이 `dsyrk`의 크기 나누기 대 스레드 나누기(D-33으로 물러설지), (3) Barnes–Hut의 정렬 나무 대 삽입 나무와 float64 대 float32의 한 걸음, (4) UMAP의 동기 SGD·Adam 대 차례 SGD의 품질(trustworthiness, 걸음 수)과 시간, (5) 정확한 k-NN(spatial)의 시간 n = 10⁴–10⁶, p = 10–784(§9의 2번의 자료).
- 큰 n의 벽: n × n 방법(커널 PCA, 고전 MDS, Isomap, SMACOF)은 메모리 8n²가 첫 벽이다(n = 5×10⁴에서 20 GB). 랜드마크·Nyström 꼴은 나중이다.

---

## 6. 층과 파일
- **층**: `embed/`는 `array/`, `base/`(`buffer.hpp`, `pairwise.hpp`, EM44의 `keyorder.hpp`), `parallel/`, `bind/`, `accelerate/`, `linalg/`(LAPACK 래퍼), `spatial/`(`metric.hpp`, `kdtree.hpp`, `neighbors.hpp`, `sfc.hpp`), `sparse/`(CSR, SpMV, `svds`, `eigsh`, LU), `graph/`(최단 경로, 라플라시안, 연결 요소), `fft/`, `random/philox.hpp`, `stats/sample_dist.hpp` 위에 있다. 다른 영역은 `embed/`를 모른다. Python 층의 `stardust.embed`는 `stardust.random`, `stardust.linalg`(`LinAlgError`), `stardust.sparse`, `stardust.spatial`, `stardust.graph`, `stardust.optimize`, `stardust.frame`, `stardust._learn`(EM44), `stardust._summary`, `stardust._calls`를 쓴다.
- 등록: `bench/areas.py`에서 `embed`는 `cluster` 뒤다. ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표, tests/README.md·bench/README.md의 영역 목록, ROADMAP의 새 절에 EM0 마일스톤이 한 줄씩 더한다.
- C++ 폴더 `embed/`(어림): `common.hpp`(입력의 열쇠 사본, 중심화, 부호 규칙, 원소 번호), `linear.cpp`(PCA의 풀이, 절단 SVD, 무작위 SVD, 무작위 투영), `factor.cpp`(인자 분석의 목적과 기울기, 회전, FastICA), `nmf.cpp`, `kernel.cpp`(커널 PCA의 핵과 중심화), `mds.cpp`(고전, SMACOF, 등위 회귀, Sammon, stress), `graph.cpp`(k-NN 그래프, 대칭화, Isomap, LLE, 스펙트럼), `tsne.cpp`(P, 끌림, 정확한 밀어냄, Barnes–Hut 나무, FFT 보간), `umap.cpp`(퍼지 집합, a·b, 동기 최적화, `transform`), `nndescent.cpp`, `quality.cpp`(trustworthiness, continuity), `embed.hpp`(파일마다의 `bind_*`).
- 바인딩 `bind/embed_bind.cpp`: `_core.embed` 하위 모듈(측정용 `_proto_*`는 EM0 마일스톤이 만들고 쓰는 마일스톤이 지운다).
- `src/stardust/embed/`(하위 패키지 없음): `__init__.py`(`__all__`), `_results.py`(`EmbeddingResult`, `summary`; EM7·EM45), `_linear.py`, `_factor.py`, `_nmf.py`, `_kernel.py`, `_mds.py`, `_graph.py`, `_tsne.py`, `_umap.py`, `_quality.py`.

---

## 7. 참값과 검증
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다.

### 7.1 시험 자료 (`tests/embed/samples.py`)
공식과 고정 시드로 만든다(내려받은 것 없음).
- **선형**: 낮은 계수 + 잡음(계수 1–20, p = 2–200), 고유값이 겹치는 자료(같은 분산의 두 축), 고유값 간격이 아주 작은 자료, 큰 평행 이동(10⁸ + 자료: 중심화의 상쇄), 아주 작거나 큰 척도(10⁻¹⁵⁰, 10¹⁵⁰), 상수 열, n < p, 희소 자료, 음수 없는 자료(NMF), 독립 원천의 섞임(균등, 라플라스, 정현; ICA), 인자 모형에서 뽑은 자료(인자 분석).
- **다양체**: 스위스 롤, S 곡선, 원·고리, 구면, 두 개의 떨어진 덩어리(연결되지 않은 그래프), 정수 격자(거리의 동점), 같은 점의 반복, 한 직선 위의 점.
- **이웃 임베딩**: 가우스 덩어리 k = 3–10(p = 10–50), 크기가 크게 다른 덩어리, 손글씨 숫자 꼴의 합성 자료(공식으로 만든 8 × 8 그림).
- **퇴화**: n = 1, 2, 3, d ≥ 서로 다른 점의 수, perplexity ≥ n − 1, 모든 점이 같다.

### 7.2 참값 (`tools/embed_ref.py` → `tests/embed/ref/`, `tests/reffile.py`의 npz)
- `linear`: 정확한 유리수의 가운데 Gram과 Arb 고유쌍(분산, 성분, 특이값; 300점 × 50열 이하), 커널 PCA의 핵(유리수: linear·poly; rbf는 Arb), 인자 분석의 주어진 ψ에서 목적·기울기(Arb), FastICA·NMF의 한 걸음(유리수).
- `mds`: B의 유리수와 Arb 고유쌍, Guttman 한 번(유리수), stress 세 꼴, 등위 회귀(유리수, 동점 규칙), Sammon 한 걸음.
- `graph`: k-NN 그래프(정확한 제곱 거리의 동점 표시), Isomap의 측지 거리(Arb의 제곱근 합), LLE의 국소 무게(유리수의 정규 방정식), 정규화 라플라시안의 고유쌍(Arb).
- `sne`: 이분 찾기의 β·σ·ρ(Arb의 엔트로피와 무게), 대칭 P, 퍼지 집합, 주어진 Y의 t-SNE 기울기(정확·BH·FFT의 근사식; Arb)와 KL, UMAP 한 걸음의 끌림·밀어냄(주어진 표본).
- `quality`: trustworthiness·continuity(정확한 유리수).
- mpmath는 쓰지 않는다(EM46).

### 7.3 테스트 (`tests/embed/`)
- 파일(어림): `test_api_surface.py`, `test_reference_data.py`, `test_results.py`, `test_pca.py`, `test_svd.py`, `test_projection.py`, `test_factor.py`, `test_ica.py`, `test_nmf.py`, `test_kernel_pca.py`, `test_mds.py`, `test_isomap.py`, `test_lle.py`, `test_spectral.py`, `test_tsne.py`, `test_umap.py`, `test_nndescent.py`, `test_quality.py`, `conftest.py`, `samples.py`, `refdata.py`.
- 성질: 길마다의 결정성 약속(§3.11), 입력 순서를 섞으면 같은 바이트를 그 순서로(EM12), 같은 `rng` 상태의 같은 바이트와 `rng`가 두 단어만 움직이는 것, 부호 규칙(EM14), 풀이 사이의 계약 안 일치(`'full'`·`'covariance_eigh'`·`'lanczos'`), 2의 거듭제곱 척도의 같은 임베딩(정확히 옮겨지는 양).
- 비교 대상: sklearn(같은 시작의 같은 고정점, 같은 단계의 같은 양), R `prcomp`·`princomp`·`factanal`·`varimax`·`promax`·`cmdscale`·`sammon`·`isoMDS`(같은 목적), `irlba`·`RSpectra`(특이값), statsmodels `PCA`·`Factor`.

### 7.4 비교 대상의 기록 (`tools/embed_probe.py` → `tests/embed/data/`)
- 기본값, 부호, 동점, 퇴화, 연결되지 않은 그래프, 경고, 스레드 의존(sklearn `TSNE`의 `sum_Q`), 시간. 기록은 `sklearn.json`, `scipy.json`, `statsmodels.json`, `r.json`(R은 `bench/accuracy/rsession.py`의 세션).
- 설치되지 않은 것(umap-learn, openTSNE, pynndescent, Rtsne, uwot, fastICA, NMF, smacof, vegan, FactoMineR): 허락을 받기 전까지는 공개 문서와 논문으로 뜻을 맞추고(umap-learn의 `smooth_knn_dist`·`fuzzy_simplicial_set`·`find_ab_params`·`optimize_layout_euclidean`, FIt-SNE의 보간) 참값과 품질 지표로만 지킨다. 허락되면 EM6–EM8 마일스톤의 기록에 더한다.

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/embed.py`, 표 `bench/embed/accuracy.md`): 스윕 `linear`, `mds`, `graph`, `sne`, `quality`(§4의 한계 단위; sklearn·R 열). `--compare`가 비교 패키지의 같은 양을 같은 한계로 잰다.
- **속도** (`bench/speed/cases/embed.py`): §5의 사례와 크기, 1·10스레드. `--smoke`에서 결과를 비교 대상과 견준다.
- **골든** (`tests/golden/cases_embed.py`, 기록 `tests/golden/data/embed.json`): 모든 공개 함수 × 시험 자료의 대표, 옵션의 값마다, 틀린 호출, 병렬 길을 탈 큰 케이스(워커 하나 1 GB 아래). 스레드 무관 길은 1·3·10스레드가 같아야 하고, D-33 길(있으면)은 스레드 수마다의 기록이다.

---

## 8. 마일스톤
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| EM0 준비 | API 문서를 시제품의 측정에 맞춰 고치기, 영역 등록, 공용 장치 꺼내기(EM44; cluster의 테스트·골든이 그대로), 시험 자료(§7.1), 참값 도구(§7.2), 탐침 도구와 기록(§7.4), 결과의 공통 기반과 `summary`, 시제품의 측정(§5의 다섯), 속도 사례와 골든의 틀 | 문서와 도구가 커밋되고 탐침 기록과 시제품의 잰 값이 있다 |
| EM1 PCA·절단 SVD·무작위 투영 | `pca`(네 풀이, 희소 입력, `'mle'`, `score_samples`), `truncated_svd`, `random_projection`, `johnson_lindenstrauss_min_dim` | 특이값·성분의 계약, sklearn·R과 같은 값과 부호, 풀이 사이의 일치, 결정성, 속도 사례 |
| EM2 인자 분석·회전·ICA | `factor_analysis`(최대우도, 검정, 점수), `rotate_factors`, `fastica` | R `factanal`·`varimax`·`promax`와 같은 고정점·값, sklearn과 같은 시작의 같은 고정점, 속도 사례 |
| EM3 NMF·커널 PCA | `nmf`(cd·mu, 시작 다섯, 정칙화, 희소 입력), `kernel_pca`(커널 여섯, 되돌림) | 고정점 조건, sklearn과 같은 시작의 같은 답, 핵의 계약, 속도 사례 |
| EM4 MDS | `classical_mds`(`add`), `mds`(계량·비계량), `sammon`, `stress` | 고유쌍의 계약, Guttman 한 걸음의 참값, sklearn·R과 같은 시작의 같은 고정점, 속도 사례 |
| EM5 이웃 그래프와 지표 | k-NN 그래프의 공용 커널, `isomap`, `locally_linear_embedding`(네 방법), `spectral_embedding`, `trustworthiness`, `continuity` | 그래프·측지 거리·국소 무게의 참값, sklearn과 같은 임베딩(부호 규칙 뒤), 지표의 올바른 반올림, 속도 사례 |
| EM6 t-SNE | 퍼플렉서티와 P, `'exact'`, `'barnes_hut'`(2·3차원 나무), 걸음의 일정 | 기울기·KL의 참값, 같은 시작의 sklearn `'exact'`와 같은 궤적 근처, BH가 스레드와 무관, 품질 분포, 속도 사례 |
| EM7 UMAP | 퍼지 집합, a·b, 스펙트럼 시작, 동기 최적화(EM40), `transform` | 퍼지 집합의 참값, 스레드와 무관한 바이트, 품질 분포(설치되면 umap-learn·uwot), 속도 사례 |
| EM8 큰 n | `neighbors='nndescent'`(§3.9, 다섯 함수), `tsne(method='fft')`(§3.8) | NN-descent의 재현율(정확한 k-NN 대비), 스레드와 무관한 바이트, FFT 기울기의 근사식 참값, 속도 사례 |
| EM9 성능과 일관성 점검 | 모든 공개 함수를 크기마다 비교 대상과 견주어 고치기, 길의 문턱(EM18, EM20), D-33으로 물러설 곳(§3.11)의 측정, 메시지·시그니처의 영역 간 일관성(cluster·regression과), 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다 |
| EM10 문서 | `docs/api/embed.md`의 확정, 이 문서를 구현대로, 등록 문서, **전체 테스트·골든 한 번**(EM48) | 문서가 코드와 맞고 `test_api_surface`와 전체 검사가 통과한다 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 정할 것을 추천안으로 정하고(범위·재현성·설치만 묻는다), 구현하고, 정확도·속도를 재고, 커밋한다. 마일스톤 에이전트는 한 번에 하나다(C++ 재빌드가 모든 import를 멈춘다). 한 마일스톤은 한 에이전트가 한 번에 끝낼 크기다. 의존: EM2–EM4는 EM1의 중심화·부호·풀이를, EM5는 EM4의 고전 MDS(Isomap의 핵)를, EM6은 EM1의 `pca`(시작)와 EM5의 k-NN 그래프를, EM7은 EM5의 그래프·라플라시안을, EM8은 EM6·EM7을 쓴다. 그래서 순서는 EM0 → EM1 → … → EM10이다. 기존 영역의 결과는 비트까지 같아야 한다(이 영역은 다른 영역의 커널을 부르기만 한다; 공용 장치를 꺼내는 EM0만 cluster의 골든으로 확인한다).

**나중 후보**(첫 판 밖; §9의 1번): 대응 분석·다중 대응 분석(MASS `corresp`·`mca`, FactoMineR), 확산 지도, PaCMAP·TriMap, t-SNE의 새 점(openTSNE), UMAP의 `inverse_transform`·지도 UMAP·densMAP, 점진 PCA, 희소 PCA·사전 학습, 정준 상관(CCA), 주축 인자법과 GPArotation의 사각 회전, 랜드마크 MDS·Nyström 꼴, 공동 순위 행렬과 R_NX 곡선, 공개 근사 이웃 함수(spatial).

---

## 9. 사용자에게 물을 것
EM5–EM48은 추천안으로 정했다(사용자 위임 2026-11-27). 결정성은 EM4(사용자 결정 2026-11-27)대로 길마다 이 문서가 정했다(§3.11): D-33으로 물러서는 것은 묻지 않는다. 아래 넷만 사용자의 답이 필요하다. 추천을 첫째에 둔다.

1. **범위의 의심스러운 이름들** (§1.2). (a) §1.2의 "EM 마일스톤" 열이 있는 이름 전부(선형 분해 아홉, MDS 셋과 `stress`, 다양체 셋과 지표 둘, `tsne`(정확·BH·FFT), `umap`, NN-descent 옵션)를 넣고, 대응 분석·MCA, 확산 지도, PaCMAP·TriMap, t-SNE의 새 점, 점진·희소 PCA, CCA는 나중으로 둔다. 추천: 사용자가 든 PCA·UMAP·t-SNE와 sklearn·R의 표준 묶음을 덮고, 성능을 위한 큰 n의 길(FFT, NN-descent)을 첫 판에 넣는다. 나중의 것은 입력·결과의 꼴을 따로 설계해야 하거나 비교 대상이 설치되지 않았다. (b) (a)에 대응 분석·MCA(R `MASS`가 설치되어 있어 견줄 수 있다)와 PaCMAP을 더한다: 마일스톤이 하나 는다. (c) (a)에서 FIt-SNE와 NN-descent(EM8 마일스톤)를 뺀다: 첫 판이 작아지지만 n ≳ 10⁵의 t-SNE·UMAP이 느리다.
2. **이웃 찾기의 기본값** (EM17; `isomap`, `locally_linear_embedding`, `spectral_embedding`, `tsne`, `umap`의 `neighbors`). (a) 기본은 `'exact'`(spatial의 정확한 k-NN), `'nndescent'`는 사용자가 이름으로 고른다. 추천: EM15(조용히 어림하지 않는다)와 같은 원칙이고 n과 p의 문턱에서 결과가 갑자기 바뀌지 않는다. 낮은 p는 KD 트리라 빠르고, 높은 p·큰 n(10⁶ × 50 어림 수 분)에서만 느리다(EM0 마일스톤이 잰다). (b) `'auto'`가 기본: 모양으로 정확과 근사를 고른다(umap-learn·uwot은 n < 4096이면 정확, 넘으면 근사; openTSNE도 비슷하다). 큰 자료에서 빠르지만 같은 함수가 크기에 따라 다른 이웃을 쓴다. (c) 근사 이웃을 두지 않는다((1)의 (c)와 함께).
3. **UMAP의 Hogwild 꼴** (EM40). Hogwild(umap-learn의 여러 스레드 SGD)는 고정 스레드 수에서도 실행마다 결과가 달라 D-33을 지킬 수 없다. (a) 두지 않는다: 걸음마다 동기인 꼴(uwot `batch=TRUE`와 같은 꼴)만 두고 그것이 나란하고 스레드와 무관하다. 추천: 재현성을 지키면서 10스레드의 나란함도 얻는다. 걸음의 꼴이 umap-learn과 달라 같은 걸음 수에서 좌표의 수렴이 조금 다를 수 있다(EM0 시제품이 품질로 잰다). (b) `optimizer='hogwild'`를 옵션으로 더 둔다: 원본과 같은 걸음이지만 그 옵션만 재현되지 않는다(이 프로젝트의 첫 재현되지 않는 길). (c) umap-learn의 차례 SGD(1스레드, 재현됨)를 옵션으로 둔다: 원본의 걸음과 같고 느리다.
4. **비교 대상의 설치** (EM3, §7.4). 설치되지 않은 것: Python umap-learn(pynndescent, numba, llvmlite가 따라온다), openTSNE, pacmap, trimap; R Rtsne, uwot, fastICA, NMF, smacof, vegan, FactoMineR, psych, kernlab, dimRed, coRanking. (a) 핵심 다섯을 설치한다: umap-learn, openTSNE, R uwot, Rtsne, fastICA. 추천: UMAP·FIt-SNE·Barnes–Hut의 품질과 시간을 원본과 견줄 수 있고(지금은 sklearn `TSNE`뿐), uwot은 sd와 같은 동기 꼴(`batch=TRUE`)이 있다. (b) 설치하지 않는다: sklearn·scipy·statsmodels와 R `stats`·`MASS`·`irlba`·`RSpectra`로만 견주고 UMAP·FIt-SNE는 참값과 품질 지표로만 지킨다. (c) 목록 전부를 설치한다(smacof, FactoMineR, psych 등은 나중 후보나 이미 R `stats`·`MASS`와 겹치는 것을 견주는 데만 쓴다).
