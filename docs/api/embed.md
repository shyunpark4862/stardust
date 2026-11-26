# sd.embed API

> 상태: **초안** (2026-11-27; 설계 결정 EM1–EM48). 아직 공개된 이름이 없다. §2–§7의 공개 이름은 선형 분해(§2: `pca PCAResult truncated_svd TruncatedSVDResult random_projection RandomProjectionResult johnson_lindenstrauss_min_dim factor_analysis FactorAnalysisResult rotate_factors RotationResult fastica FastICAResult nmf NMFResult kernel_pca KernelPCAResult`), 거리 보존(§3: `classical_mds ClassicalMDSResult mds sammon MDSResult`), 이웃 그래프(§4: `isomap IsomapResult locally_linear_embedding LLEResult spectral_embedding SpectralEmbeddingResult`), 확률적 이웃 임베딩(§5: `tsne TSNEResult umap UMAPResult`), 지표(§6: `trustworthiness continuity stress`)이고 마일스톤 EM1–EM8에서 하나씩 공개된다(설계 문서 §8). 시그니처는 scikit-learn 1.9.1(`sklearn.decomposition`, `sklearn.manifold`, `sklearn.random_projection`), R 4.6.2의 `stats`·`MASS`, umap-learn(설치되지 않음; 공개 문서)의 이름과 뜻에 stardust의 규칙과 설계 결정을 입힌 안이다.

- 설계와 결정의 근거는 [docs/design/embed.md](../design/embed.md)(§1.1 EM1–EM48), 공통 규칙은 [API.md](../../API.md)다. 꼴은 [docs/api/cluster.md](cluster.md)와 같다.
- 언어 표시가 없는 코드 블록은 `tests/embed/test_api_surface.py`(EM0 마일스톤)가 읽는다. 블록의 줄은 모듈의 함수(`name(...)`), 결과 클래스(`Name`), 결과의 메서드(`Name.method(...)`)다. 블록의 모듈 수준 이름이 `sd.embed.__all__` 전부이고, 시그니처가 블록과 같아야 한다.
- 설계 문서의 §9에 걸린 곳은 "(§9의 n번)"으로 적었다. 답에 따라 그 줄이 바뀐다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양 (EM1, EM5, EM6)
- **이름공간은 평평하다**: 모든 공개 이름이 `sd.embed`에 바로 있다. 하위 모듈은 없다.
- **맞추는 함수는 소문자 이름이고 `<이름>Result`를 돌려준다**: `pca` → `PCAResult`, `tsne` → `TSNEResult`, `umap` → `UMAPResult`, `mds`·`sammon` → `MDSResult` 등. 추정기 클래스(`PCA(...).fit(X)`)는 없다. 맞춘 뒤의 일(새 점 옮기기, 되돌리기)은 결과의 메서드다.
- **인자의 순서는 (X, n_components)다**: `X`는 첫째 위치 인자, `n_components`는 둘째 위치 인자(위치로만)이고 기본값은 기준 라이브러리의 것이다. `factor_analysis`만 기본값이 없다(R `factanal`의 `factors`).
- **데이터는 위치로만, 옵션은 키워드로만** 받는다. 별칭이 없다. 틀린 호출 모양은 부른 이름으로 시작한다(API.md §2.1): `pca(): unexpected keyword argument 'random_state'`, 결과의 메서드는 `transform(): ...`.
- **풀이의 이름**(EM23): Lanczos 계열은 모두 `'lanczos'`다(sklearn의 `'arpack'`). 거듭제곱 반복의 수는 `n_power_iter`다(sklearn `iterated_power`, `n_iter`).

### §1.2 입력 (EM8–EM10, EM12)
- **받는 것**: API.md §5의 상위 함수 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `DataFrame`(열이 변수)과 `Series`(한 변수)). 점은 행이다: `X`는 (n, p), 1차원은 (n, 1)로 읽는다. 계산은 float64다. bool·정수는 float64로 바꾼다. 복소수는 `TypeError`다.
- **희소 배열**(`sd.sparse`의 모든 형식): `pca`(`svd_solver='lanczos'`, `'randomized'`, `'covariance_eigh'`, `'auto'`), `truncated_svd`, `nmf`, `random_projection`, `spectral_embedding(affinity='precomputed')`만 받는다. 다른 곳은 `TypeError`: `isomap(): 'X' must be dense; for a sparse array use X.toarray()`. `pca(svd_solver='full')`의 희소 입력도 같은 꼴이다.
- **미리 계산한 거리**: `classical_mds`, `mds`, `sammon`, `isomap`, `tsne`, `umap`, `trustworthiness`, `continuity`, `stress`는 `metric='precomputed'`로 (n, n) 정사각(대칭, 대각 0, 음수·NaN 없음; `ValueError`) 또는 1차원 압축 꼴(길이 n(n − 1)/2)을 받는다. 거리의 이름과 값은 `sd.spatial.cdist`와 같다. 유사도는 `affinity`다(`spectral_embedding`).
- **`nan_policy`**: `'raise'`(기본; `ValueError`)와 `'omit'`(NaN이 있는 행을 빼고 맞춘다; `embedding`의 그 행은 NaN, `nobs`는 뺀 뒤의 수, `omitted`가 뺀 행의 bool 배열). ±inf는 언제나 `ValueError`다. `X_new`·`Z`의 NaN·±inf는 `ValueError`다(`transform(): 'X_new' contains NaN`). 열 수가 맞춤과 다르면 `'X_new' must have 5 columns like the fit's 'X', not 3`, `inverse_transform`은 `'Z' must have 2 columns like the embedding, not 3`이다.
- **입력 순서**: 같은 점들을 어떤 순서로 주어도 `embedding`의 행이 그 순서로 옮겨진 같은 바이트다(난수를 쓰는 함수도 같은 `rng` 상태면). 이웃·순위의 동점은 (거리, 좌표의 사전순 열쇠)다.

### §1.3 결과의 공통 필드 (EM7, EM45)
| 필드·메서드 | 뜻 |
|---|---|
| `embedding` | (n, d) float64, 입력의 순서. `'omit'`이 뺀 행은 NaN. sklearn의 `fit_transform(X)`, R `prcomp`의 `x`, umap-learn·openTSNE의 `embedding_` |
| `n_components` | d(쓴 값; `'mle'`·비율·`'auto'`에서 고른 수) |
| `nobs`, `omitted` | 맞춤에 든 점의 수, `'omit'`이 뺀 행 (n,) bool |
| `n_iter`, `converged`, `objective` | 반복하는 방법만. 상한에서 멈추면 `converged=False`이고 에러가 아니다 |
| `transform(X_new, /)` | 새 점의 임베딩 (m, d) (옮길 수 있는 결과만; UMAP은 `rng`도 받는다) |
| `inverse_transform(Z, /)` | 임베딩에서 자료 공간으로 (m, p) (되돌릴 수 있는 결과만) |
| `summary()` | API.md §2.3의 상자: 머리에 방법, `nobs`, `n_components`, 목적값, 반복 수, 수렴, 조절 모수. 성분이 있는 결과는 성분마다 한 줄(분산·비율·누적) |

- 배열 필드는 쓰기 금지이고 처음 읽을 때 계산하는 필드가 있다(API.md §2.2). 결과의 생성자는 공개 API가 아니다. `repr`는 한 줄이다: `PCAResult(nobs=300, n_components=2, explained_variance_ratio=[0.73, 0.23])`.
- 성분·축의 부호(EM14): 성분 벡터마다 절댓값이 가장 큰 원소가 양수다(같으면 작은 열쇠). sklearn의 `svd_flip(u_based_decision=False)`와 같은 규칙이고 고전 MDS·커널 PCA·Isomap·스펙트럼 임베딩의 `embedding` 열에도 쓴다.
- 자료가 필요한 메서드(`transform` 등)를 위해 결과는 자료의 읽기 전용 참조를 쥔다(사본이 아니다).

### §1.4 결정성, 난수, 정확도 (EM4, EM11, EM15)
- **같은 기계, 같은 스레드 수, 같은 입력(같은 `rng` 상태)이면 실행마다 같은 바이트다**(API.md §3). 거의 모든 길은 스레드 수와도 무관하다(설계 문서 §3.11). BLAS·LAPACK을 부르는 계산(밀집 SVD, Gram, 행렬 곱)은 Accelerate의 스레드 모드가 같을 때 같다(API.md §3).
- **`rng`**: 고른 옵션이 난수를 뽑을 때만 받는다. 언제나 뽑는 함수(`fastica`, `random_projection`, `umap`)는 기본값 없는 키워드다(없으면 `TypeError`). 옵션에 달린 함수는 `rng=None`이고, 필요한데 없으면 `TypeError`(`pca(): svd_solver='randomized' needs 'rng' (sd.random.default_rng(seed))`), 쓰지 않는데 주면 `ValueError`(`pca(): 'rng' is only used with svd_solver='randomized'`)다. 필요를 정하는 것은 옵션의 값뿐이고 자료의 모양이 아니다. 호출 하나가 `rng`에서 64비트 두 단어만 받는다(Philox의 키; 흐름 r의 t번째 수는 원소 r·2⁴⁰ + t; cluster와 같은 규칙).
- **`'auto'`는 어림을 고르지 않는다**: 무작위 어림(`'randomized'`)은 이름으로만 고른다(`[sklearn과 다름]`).
- **정확도 계약**(설계 문서 §4): 정의가 있는 양(특이값, 고유쌍, 핵, P, 퍼지 집합, 한 걸음의 기울기, 지표)은 참값과 견준 한계 안이다. 비볼록 반복의 끝 좌표는 계약이 없다. `trustworthiness`·`continuity`는 올바르게 반올림한 값이다.

### §1.5 에러와 경고
- API.md §2의 종류와 꼴이다. 경고는 내지 않는다.

  | 다른 라이브러리 | sd |
  |---|---|
  | sklearn `FastICA`·`NMF`·`FactorAnalysis`·`TSNE`: 반복 상한(`ConvergenceWarning`) | `converged=False` |
  | sklearn `MDS`: `init`의 기본값 예고(`FutureWarning`) | 없음: 기본은 `'classical_mds'` |
  | sklearn `Isomap`: 연결되지 않은 그래프(경고, 성분을 잇는다) | 같이 잇고 `n_connected_components`가 원래 성분 수 |
  | sklearn `SpectralEmbedding`: 연결되지 않은 그래프(경고) | 계산하고 `n_connected_components` |
  | R `cmdscale`: 음의 고유값(경고) | `eigenvalues`에 모두 보인다 |
  | umap-learn: `random_state`가 `n_jobs`를 덮는다(경고) | 없음: 언제나 나란하고 재현된다 |
  | umap-learn: 스펙트럼 시작 실패(경고, 무작위 시작으로) | `LinAlgError`: `umap(): the spectral initialisation did not converge; use init='random'` |
  | R `sammon`: 거리 0의 짝(에러) | `ValueError`: `sammon(): points 3 and 7 are at distance 0; Sammon's stress divides by every distance` |
  | sklearn `TSNE`: perplexity ≥ n(`ValueError`) | 같은 `ValueError`: `tsne(): 'perplexity' (30.0) must be less than the number of points (20)` |

- 할당 전에 `MemoryError`인 곳: n × n 핵이나 거리(`kernel_pca`, `classical_mds`, `isomap`, `mds`, `sammon`)가 물리 메모리보다 클 때.

## §2. 선형 분해 (EM1–EM3 마일스톤)
```
pca(X, n_components=None, /, *, scale=False, whiten=False, svd_solver='auto', tol=0.0, max_iter=None, n_oversamples=10, n_power_iter='auto', rng=None, nan_policy='raise')
PCAResult
PCAResult.transform(X_new, /)
PCAResult.inverse_transform(Z, /)
PCAResult.score_samples(X_new, /)
PCAResult.summary()
truncated_svd(X, n_components=2, /, *, algorithm='lanczos', tol=0.0, max_iter=None, n_oversamples=10, n_power_iter=5, rng=None, nan_policy='raise')
TruncatedSVDResult
TruncatedSVDResult.transform(X_new, /)
TruncatedSVDResult.inverse_transform(Z, /)
TruncatedSVDResult.summary()
random_projection(X, n_components='auto', /, *, rng, method='gaussian', density='auto', eps=0.1, nan_policy='raise')
RandomProjectionResult
RandomProjectionResult.transform(X_new, /)
RandomProjectionResult.inverse_transform(Z, /)
RandomProjectionResult.summary()
johnson_lindenstrauss_min_dim(n_samples, /, *, eps=0.1)
factor_analysis(X, n_components, /, *, rotation='varimax', lower=0.005, max_iter=1000, nan_policy='raise')
FactorAnalysisResult
FactorAnalysisResult.transform(X_new, /, *, method='regression')
FactorAnalysisResult.summary()
rotate_factors(loadings, /, *, method='varimax', normalize=True, eps=1e-05, power=4)
RotationResult
fastica(X, n_components=None, /, *, rng, algorithm='parallel', whiten='unit-variance', fun='logcosh', alpha=1.0, max_iter=200, tol=0.0001, w_init=None, whiten_solver='svd', nan_policy='raise')
FastICAResult
FastICAResult.transform(X_new, /)
FastICAResult.inverse_transform(Z, /)
FastICAResult.summary()
nmf(X, n_components='auto', /, *, init=None, solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=200, alpha_W=0.0, alpha_H='same', l1_ratio=0.0, shuffle=False, W_init=None, H_init=None, rng=None, nan_policy='raise')
NMFResult
NMFResult.transform(X_new, /)
NMFResult.inverse_transform(Z, /)
NMFResult.summary()
kernel_pca(X, n_components=None, /, *, kernel='linear', gamma=None, degree=3, coef0=1.0, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0.0, max_iter=None, n_oversamples=10, n_power_iter='auto', remove_zero_eig=False, rng=None, nan_policy='raise')
KernelPCAResult
KernelPCAResult.transform(X_new, /)
KernelPCAResult.inverse_transform(Z, /)
KernelPCAResult.summary()
```
- **`pca`**: sklearn `PCA`의 뜻이다(분산은 n − 1로 나눈다). `n_components`는 None(min(n, p)), 1 이상 min(n, p) 이하의 int, (0, 1)의 float(누적 분산 비율이 그 값을 넘는 가장 작은 수), `'mle'`(Minka; `'full'`·`'covariance_eigh'`만; 다른 풀이는 `ValueError`)다. 언제나 가운데로 옮긴다. `scale=True`는 열을 표본 표준편차로 나눈다(R `prcomp(scale.=TRUE)`, statsmodels `standardize=True`; 표준편차가 0인 열은 `ValueError`, R과 같다). `whiten=True`는 `embedding`·`transform`의 열을 √`explained_variance`로 나눈다.
  - `svd_solver`: `'auto'`, `'full'`, `'covariance_eigh'`, `'lanczos'`, `'randomized'`. `'auto'`는 모양으로만 고른다: p ≤ 1000이고 n ≥ 10p면 `'covariance_eigh'`, 그 밖에 d ≤ min(n, p)/10이고 min(n, p) > 500이면 `'lanczos'`, 나머지는 `'full'`이다(희소 입력은 `'lanczos'`; 문턱은 EM1 마일스톤이 잰다). `[sklearn과 다름]` sklearn의 `'auto'`는 큰 입력에서 `'randomized'`를 고른다. 결과의 `svd_solver`가 쓴 풀이다.
  - `tol`·`max_iter`는 `'lanczos'`의 것(`sd.sparse.svds`의 뜻; 0은 기계 엡실론, None은 10·min(n, p))이고 다른 풀이에 0·None이 아닌 값을 주면 `ValueError`다. `n_oversamples`, `n_power_iter`(`'auto'`: d < 0.1·min(n, p)이면 7, 아니면 4; sklearn `iterated_power`)는 `'randomized'`의 것이다.
  - `score_samples(X_new)`: 확률적 PCA(Tipping–Bishop)의 점마다 로그 우도(sklearn과 같다). sklearn `score(X)`는 그 평균이다.
- **`truncated_svd`**: 가운데로 옮기지 않는 SVD(sklearn `TruncatedSVD`). `algorithm`은 `'lanczos'`(기본; `[sklearn과 다름]`: sklearn은 `'randomized'`, n_iter = 5), `'randomized'`(`rng` 필요), `'full'`(밀집만). 1 ≤ `n_components` < min(n, p)(sklearn과 같다; `'full'`은 ≤). `explained_variance`는 `embedding` 열의 분산(sklearn과 같은 뜻)이다.
- **`random_projection`**: sklearn `GaussianRandomProjection`(`method='gaussian'`: N(0, 1/d))과 `SparseRandomProjection`(`method='sparse'`: 원소가 −√(s/d), 0, √(s/d)이고 확률 1/(2s), 1 − 1/s, 1/(2s), s = 1/`density`)이다. `density`는 `'auto'`(1/√p) 또는 (0, 1]이고 `'gaussian'`에 주면 `ValueError`다. `n_components='auto'`는 `johnson_lindenstrauss_min_dim(n, eps=eps)`이고 p보다 크면 `ValueError`(sklearn과 같다). `components`는 `'gaussian'`이면 (d, p) 밀집, `'sparse'`면 `sd.sparse.csr_array`다. `inverse_transform`은 `inverse_components`(`components`의 유사역행렬, 게으른 필드)를 쓴다.
- **`johnson_lindenstrauss_min_dim(n_samples, /, *, eps=0.1)`**: sklearn과 같은 식 4·log(n)/(ε²/2 − ε³/3)의 올림. `n_samples`·`eps`는 수 또는 배열(브로드캐스트)이고 결과는 int 또는 int64 배열이다. `eps`는 (0, 1)이다.
- **`factor_analysis`**: 최대우도 인자 분석이다(기준은 R `factanal`; 설계 문서 EM26). 상관행렬에서 고유도를 [`lower`, 1]에서 최적화한다(L-BFGS-B; `max_iter`는 최적화 반복의 상한). 1 ≤ `n_components`이고 자유도 ((p − d)² − (p + d))/2가 음수가 아니어야 한다(R과 같다; `ValueError`). `rotation`은 None, `'varimax'`(기본; R과 같다), `'quartimax'`, `'promax'`이다(sklearn의 기본은 None `[sklearn과 다름]`).
  - `transform(X_new, /, *, method='regression')`: `'regression'`(Thomson; sklearn `transform`과 같은 사후 평균) 또는 `'bartlett'`(R `scores='Bartlett'`). `embedding`은 `'regression'`의 점수다.
- **`rotate_factors`** → `RotationResult(loadings, rotation, phi)`(namedtuple): `method`는 `'varimax'`(R `varimax`의 Kaiser 정규화 `normalize`, `eps`), `'quartimax'`, `'promax'`(R `promax`의 `m` = `power`; varimax 뒤의 사각 회전)다. `rotation`은 회전 행렬(R `rotmat`; 회전한 적재 = `loadings @ rotation`), `phi`는 promax의 인자 상관이고 직교 회전은 None이다. `loadings`는 (p, d)다.
- **`fastica`**: sklearn `FastICA`의 뜻과 기본값이다. `algorithm`은 `'parallel'`·`'deflation'`, `whiten`은 `'unit-variance'`·`'arbitrary-variance'`·False(False면 `n_components`는 None이거나 p), `fun`은 `'logcosh'`(`alpha`는 [1, 2])·`'exp'`·`'cube'`(호출 가능한 함수는 받지 않는다 `[sklearn과 다름]`), `whiten_solver`는 `'svd'`·`'eigh'`. `w_init`은 (d, d) 배열이다(주어도 `rng`는 두 단어를 움직인다; `kmeans`의 배열 `init`과 같다). 성분의 부호는 §1.3, 차례는 원천의 첨도 크기 차례다(`[sklearn과 다름]`).
- **`nmf`**: sklearn `NMF`의 뜻과 기본값이다. `init`은 None(d ≤ min(n, p)면 `'nndsvda'`, 아니면 `'random'`), `'nndsvd'`, `'nndsvda'`, `'nndsvdar'`, `'random'`, `'custom'`(`W_init` (n, d)과 `H_init` (d, p)). `n_components='auto'`는 `'custom'`의 모양에서, 그 밖은 p다(sklearn과 같다). `solver`는 `'cd'`·`'mu'`이고 `beta_loss`는 `'frobenius'`, `'kullback-leibler'`, `'itakura-saito'` 또는 수(`'mu'`만 2가 아닌 값). `alpha_H='same'`은 `alpha_W`다. `rng`는 `init='random'`·`'nndsvdar'`이거나 `shuffle=True`일 때만이다. 음수 원소는 `ValueError`다. NNDSVD의 SVD는 결정적인 Lanczos다(`[sklearn과 다름]`). `embedding`은 W, `components`는 H, `reconstruction_err`는 sklearn의 β-발산 꼴이다. `transform`은 H를 고정한 W의 풀이, `inverse_transform`은 Z·H다.
- **`kernel_pca`**: sklearn `KernelPCA`의 뜻이다. `kernel`은 `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, `'cosine'`, `'precomputed'`((n, n) 핵), `gamma=None`은 1/p다. `eigen_solver`는 `'auto'`(n ≤ 2000이거나 d ≥ n/10이면 `'dense'`, 아니면 `'lanczos'`), `'dense'`, `'lanczos'`(sklearn `'arpack'`), `'randomized'`(`rng` 필요). `fit_inverse_transform=True`면 `alpha`의 핵 릿지로 되돌림을 맞춘다(아니면 `inverse_transform`은 `ValueError`: sklearn과 같다; `'precomputed'`에서는 쓸 수 없다). `remove_zero_eig`는 sklearn과 같다.

| 결과 | 필드와 메서드 |
|---|---|
| `PCAResult` | 공통 필드, `components`(d, p), `explained_variance`(d,), `explained_variance_ratio`, `singular_values`, `mean`(p,), `scale`(p,; `scale=False`면 None), `noise_variance`, `n_features`, `svd_solver`, `whiten`, `covariance`·`precision`(p, p; 게으른 필드; sklearn `get_covariance`·`get_precision`), `transform`, `inverse_transform`, `score_samples` |
| `TruncatedSVDResult` | 공통 필드, `components`, `singular_values`, `explained_variance`, `explained_variance_ratio`, `algorithm`, `n_iter`·`converged`(`'lanczos'`만; 그 밖은 None), `transform`, `inverse_transform` |
| `RandomProjectionResult` | 공통 필드, `components`, `inverse_components`(게으른 필드), `method`, `density`, `transform`, `inverse_transform` |
| `FactorAnalysisResult` | 공통 필드, `loadings`(p, d; 상관 눈금, 회전 뒤), `uniquenesses`(p,), `components`(d, p; 자료 눈금, sklearn), `noise_variance`(p,), `mean`, `scale`, `rotation`, `rotation_matrix`(d, d 또는 None), `phi`(promax의 인자 상관 또는 None), `loglik`(= `objective`), `statistic`, `dof`, `pvalue`(R의 χ² 검정; 자유도 0이면 NaN), `n_iter`, `converged`, `transform` |
| `FastICAResult` | 공통 필드, `components`(d, p; 분리 행렬), `mixing`(p, d), `mean`, `whitening`(d, p 또는 None), `n_iter`, `converged`, `transform`, `inverse_transform` |
| `NMFResult` | 공통 필드, `components`(d, p), `reconstruction_err`(= `objective`), `n_iter`, `converged`, `solver`, `beta_loss`, `transform`, `inverse_transform` |
| `KernelPCAResult` | 공통 필드, `eigenvalues`(d,), `eigenvectors`(n, d), `kernel`, `gamma`, `transform`, `inverse_transform` |

## §3. 거리 보존 (EM4 마일스톤)
```
classical_mds(X, n_components=2, /, *, metric='euclidean', add=False, nan_policy='raise')
ClassicalMDSResult
ClassicalMDSResult.summary()
mds(X, n_components=2, /, *, metric_mds=True, init='classical_mds', n_init=1, max_iter=300, eps=1e-06, normalized_stress='auto', metric='euclidean', rng=None, nan_policy='raise')
sammon(X, n_components=2, /, *, init='classical_mds', max_iter=100, magic=0.2, tol=0.0001, metric='euclidean', nan_policy='raise')
MDSResult
MDSResult.summary()
```
- **`classical_mds`**: R `cmdscale`, sklearn `ClassicalMDS`의 뜻이다. B = −½JD²J의 큰 고유값 d개와 √λ로 늘린 고유벡터이고, 0 이하의 고유값에 해당하는 축은 0이다. `add=True`는 Cailliez의 더하는 상수(R `add=TRUE`)다. `eigenvalues`는 n개 모두(R `eig=TRUE`), `gof`는 R `GOF`의 두 값이다.
- **`mds`**: sklearn `MDS`의 SMACOF다. `metric_mds=True`는 계량, False는 비계량(등위 회귀의 불일치도; 같은 목적의 R `isoMDS`와는 다른 국소 최소에 닿을 수 있다). `init`은 `'classical_mds'`(기본; `[sklearn과 다름]`: sklearn 1.9.1은 `'random'`처럼 돌며 1.10에 바뀐다고 경고한다), `'random'`(`rng` 필요), (n, d) 배열. `n_init` > 1은 `init='random'`에서만이고(`ValueError`) 가장 좋은 시작은 (stress, 번호)가 가장 작은 것이다. 멈춤은 sklearn과 같다(`eps`는 Σδ²/2에 대한 stress 감소의 비). `normalized_stress`는 `'auto'`(비계량이면 True), True(Stress-1), False(날 stress).
- **`sammon`**: R `MASS::sammon`의 뜻과 기본값이다(`init='classical_mds'`이 R의 `cmdscale` 시작, `max_iter`는 R `niter`). `init`은 배열도 받는다(난수 없음). 거리가 0인 짝이 있으면 `ValueError`(R과 같다).

| 결과 | 필드 |
|---|---|
| `ClassicalMDSResult` | 공통 필드, `eigenvalues`(n,), `gof`(2,), `additive_constant`(`add=False`면 0.0), `metric` |
| `MDSResult` | 공통 필드, `stress`(= `objective`; `mds`는 `normalized_stress`의 꼴, `sammon`은 Sammon의 오차), `disparities`(비계량 `mds`만, 압축 꼴; 아니면 None), `n_iter`, `converged`, `n_init`, `best_init`, `method`(`'smacof'`, `'sammon'`), `metric_mds`, `metric` |

## §4. 이웃 그래프 (EM5 마일스톤; `neighbors='nndescent'`는 EM8 마일스톤)
```
isomap(X, n_components=2, /, *, n_neighbors=5, radius=None, metric='euclidean', neighbors='exact', path_method='auto', eigen_solver='auto', tol=0.0, max_iter=None, nan_policy='raise')
IsomapResult
IsomapResult.transform(X_new, /)
IsomapResult.reconstruction_error()
IsomapResult.summary()
locally_linear_embedding(X, n_components=2, /, *, n_neighbors=5, reg=0.001, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors='exact', eigen_solver='auto', tol=1e-06, max_iter=100, nan_policy='raise')
LLEResult
LLEResult.transform(X_new, /)
LLEResult.summary()
spectral_embedding(X, n_components=2, /, *, affinity='nearest_neighbors', gamma=None, n_neighbors=None, neighbors='exact', eigen_solver='auto', tol=0.0, nan_policy='raise')
SpectralEmbeddingResult
SpectralEmbeddingResult.summary()
```
- **이웃**: k 최근접은 `sd.spatial.kneighbors`의 정의다(자기 자신은 뺀다; 동점은 (거리, 열쇠)). `neighbors`는 `'exact'`(기본) 또는 `'nndescent'`(라운드 동기의 NN-descent; 재현되고 스레드와 무관하다)다(기본값은 설계 문서 §9의 2번). `'nndescent'`는 `metric='precomputed'`와 함께 쓸 수 없다(`ValueError`).
- **`isomap`**: sklearn `Isomap`의 뜻이다. `n_neighbors`와 `radius` 가운데 하나만 준다(둘 다면 `ValueError`; sklearn과 같다: `radius`를 쓰려면 `n_neighbors=None`). `path_method`는 `'auto'`, `'D'`(Dijkstra), `'FW'`(Floyd–Warshall)이고 `sd.graph.shortest_path`와 같은 값이다. `eigen_solver`는 `'auto'`, `'dense'`, `'lanczos'`. 그래프가 여러 성분이면 가장 가까운 짝의 변으로 잇는다(sklearn과 같다; 경고 없이 `n_connected_components`). `transform`은 sklearn과 같은 사영, `reconstruction_error()`는 sklearn과 같은 값이다.
- **`locally_linear_embedding`**: sklearn의 네 `method`(`'standard'`, `'modified'`, `'hessian'`, `'ltsa'`)와 기본값이다. `'hessian'`은 `n_neighbors` > d(d + 3)/2, `'modified'`는 `n_neighbors` ≥ d가 필요하다(sklearn과 같다; `ValueError`). sklearn의 `random_state`는 없다(고유값 풀이의 시작 벡터가 정해져 있다). `transform`은 sklearn과 같다.
- **`spectral_embedding`**: sklearn `SpectralEmbedding`의 뜻이다. `affinity`는 `'nearest_neighbors'`(`n_neighbors=None`은 max(n/10, 1)), `'rbf'`(`gamma=None`은 1/p), `'precomputed'`(유사도 행렬; 희소 배열도), `'precomputed_nearest_neighbors'`(희소 거리 그래프를 0/1 연결로). 정규화 라플라시안의 가장 작은 고유벡터들이고 첫째를 버린다. sklearn의 `'amg'`·`'lobpcg'`는 없다.

| 결과 | 필드와 메서드 |
|---|---|
| `IsomapResult` | 공통 필드, `eigenvalues`(d,), `graph`(k-NN 무게 그래프, `csr_array`), `dist_matrix`(n, n; 게으른 필드), `n_connected_components`, `n_neighbors`, `radius`, `metric`, `transform`, `reconstruction_error()` |
| `LLEResult` | 공통 필드, `reconstruction_error`(= `objective`), `graph`(W, `csr_array`), `method`, `n_neighbors`, `transform` |
| `SpectralEmbeddingResult` | 공통 필드, `eigenvalues`(d,), `affinity_matrix`(`csr_array`), `n_connected_components`, `affinity`, `n_neighbors`, `gamma` |

## §5. 확률적 이웃 임베딩 (EM6–EM8 마일스톤)
```
tsne(X, n_components=2, /, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', max_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='pca', method='barnes_hut', angle=0.5, neighbors='exact', rng=None, nan_policy='raise')
TSNEResult
TSNEResult.summary()
umap(X, n_components=2, /, *, rng, n_neighbors=15, metric='euclidean', n_epochs=None, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, a=None, b=None, neighbors='exact', nan_policy='raise')
UMAPResult
UMAPResult.transform(X_new, /, *, rng)
UMAPResult.summary()
```
- **`tsne`**: sklearn `TSNE`의 뜻과 기본값이다(설계 문서 EM37). `perplexity` > 0이고 n보다 작다. `learning_rate`는 `'auto'`(max(n/`early_exaggeration`/4, 50)) 또는 양수. `init`은 `'pca'`(결정적인 `pca`의 앞 d 성분을 첫 열 표준편차 10⁻⁴로 줄인 것), `'random'`(N(0, 10⁻⁸); `rng` 필요), (n, d) 배열. `method`는 `'barnes_hut'`(d ≤ 3), `'exact'`, `'fft'`(FIt-SNE의 다항 보간, d ≤ 2; EM8 마일스톤). `angle`은 `'barnes_hut'`의 θ다(다른 방법에서는 무시하지 않고 기본값이 아니면 `ValueError`). `metric='precomputed'`는 거리(제곱하지 않은 값; sklearn과 같다)이고 `init='pca'`와 함께면 `ValueError`(sklearn과 같다). 계산은 float64다(`[sklearn과 다름]`: sklearn은 float32). Barnes–Hut의 합은 스레드 수와 무관하다(`[sklearn과 다름]`).
- **`umap`**: umap-learn `UMAP`의 뜻과 기본값이다(설계 문서 EM39). `n_neighbors`는 2 이상 n − 1 이하, `n_epochs=None`은 n ≤ 10⁴이면 500, 넘으면 200. `init`은 `'spectral'`, `'random'`(균등 [−10, 10]), (n, d) 배열이고 언제나 `rng`가 필요하다(음의 표본). `a`·`b`가 None이면 `min_dist`·`spread`에서 umap-learn과 같은 곡선 맞춤으로 정한다(둘 다 주거나 둘 다 None; `ValueError`). 최적화는 걸음마다 동기인 꼴이다(설계 문서 EM40; §9의 3번): 같은 `rng` 상태면 스레드 수와 무관하게 같은 바이트이고 나란히 돈다. `[umap-learn과 다름]`: umap-learn은 여러 스레드에서 Hogwild라 재현되지 않고 `random_state`를 주면 1스레드로 돈다. 지도 UMAP(`y`), `densmap`, `inverse_transform`은 없다(나중).
  - `transform(X_new, /, *, rng)`: umap-learn `transform`의 단계(맞춘 점을 고정한 최적화)다. umap-learn의 고정 `transform_seed=42` 대신 `rng`를 받는다(`[umap-learn과 다름]`). `metric='precomputed'`의 맞춤에서는 `ValueError`(새 점과 맞춘 점의 거리가 없다).

| 결과 | 필드 |
|---|---|
| `TSNEResult` | 공통 필드, `kl_divergence`(= `objective`; 마지막으로 잰 값), `n_iter`, `converged`(`n_iter_without_progress`나 `min_grad_norm`에서 멈추면 True), `learning_rate`(쓴 값), `affinities`(대칭 P, `csr_array`; `'exact'`는 밀집에서 0을 뺀 것), `perplexity`, `method`, `angle` |
| `UMAPResult` | 공통 필드, `graph`(퍼지 단체 집합, `csr_array`), `a`, `b`, `n_epochs`, `sigmas`(n,), `rhos`(n,), `n_neighbors`, `min_dist`, `metric`, `transform` |

## §6. 지표 (EM4·EM5 마일스톤)
```
trustworthiness(X, X_embedded, /, *, n_neighbors=5, metric='euclidean')
continuity(X, X_embedded, /, *, n_neighbors=5, metric='euclidean')
stress(X, X_embedded, /, *, metric='euclidean', kind='kruskal')
```
- **`trustworthiness`**: sklearn과 같은 식이다(임베딩의 k 이웃 가운데 원래 공간에서 이웃이 아닌 점의 순위 벌점). `metric`은 원래 공간의 거리(`'precomputed'`면 `X`가 거리)이고 임베딩은 언제나 유클리드다(sklearn과 같다). 1 ≤ `n_neighbors` < n/2(`ValueError`). 순위의 동점은 (거리, 열쇠)이고 값은 정확한 정수 합을 한 번 반올림한 것이다.
- **`continuity`**: 같은 식에서 두 공간의 역할을 바꾼 것(Venna–Kaski)이다. 인자와 규칙은 `trustworthiness`와 같다.
- **`stress`**: 원래 공간의 거리 δ와 임베딩의 유클리드 거리 d의 오차. `kind`는 `'raw'`(Σ_{i<j}(d − δ)²; sklearn `MDS(normalized_stress=False).stress_`의 뜻), `'kruskal'`(Stress-1: √(Σ(d − δ)²/Σd²)), `'sammon'`(Σ(d − δ)²/δ / Σδ; 거리 0의 짝은 `ValueError`). Python `float`다.
- `X`와 `X_embedded`는 행 수가 같아야 한다(`ValueError`). NaN·±inf는 `ValueError`다.

## §7. 다른 라이브러리의 이름 (대응표)
| 다른 라이브러리 | sd.embed |
|---|---|
| sklearn `PCA(n_components=k, random_state=s).fit_transform(X)` | `pca(X, k).embedding` |
| sklearn `PCA(svd_solver='arpack')`, `'randomized'` | `svd_solver='lanczos'`, `svd_solver='randomized', rng=…` |
| sklearn `PCA.components_`, `.explained_variance_ratio_`, `.get_covariance()` | `.components`, `.explained_variance_ratio`, `.covariance` |
| R `prcomp(x, scale.=TRUE, rank.=k)` → `$sdev`, `$rotation`, `$x` | `pca(X, k, scale=True)` → `sqrt(.explained_variance)`, `.components.T`, `.embedding`(부호는 sd의 규칙) |
| R `princomp(x, cor=TRUE)` | `pca(X, scale=True)`(분산은 n − 1로; princomp는 n) |
| statsmodels `PCA(data, ncomp=k)` | `pca(X, k, scale=True)`(statsmodels의 `normalize`는 없다) |
| sklearn `TruncatedSVD(n_components=k, n_iter=5)` | `truncated_svd(X, k)`(Lanczos) 또는 `algorithm='randomized', n_power_iter=5, rng=…` |
| R `irlba(A, nv=k)`, `prcomp_irlba(x, n=k)` | `truncated_svd(X, k)`, `pca(X, k, svd_solver='lanczos')` |
| sklearn `GaussianRandomProjection`, `SparseRandomProjection` | `random_projection(X, k, rng=…)`, `method='sparse'` |
| sklearn `FactorAnalysis(n_components=k, rotation='varimax')` | `factor_analysis(X, k)`(R의 기본 회전과 같다; 회전 없이는 `rotation=None`) |
| R `factanal(x, factors=k, scores='Bartlett')` | `factor_analysis(X, k)`, `.transform(X, method='bartlett')`; `$STATISTIC`, `$dof`, `$PVAL` → `.statistic`, `.dof`, `.pvalue` |
| R `varimax(L)`, `promax(L, m=4)`; statsmodels `rotate_factors(A, 'varimax')` | `rotate_factors(L)`, `rotate_factors(L, method='promax', power=4)` |
| sklearn `FastICA(n_components=k, random_state=s, fun_args={'alpha': a})` | `fastica(X, k, rng=sd.random.default_rng(s), alpha=a)`(같은 시드여도 같은 시작은 아니다) |
| sklearn `NMF(n_components=k, init='nndsvda')` | `nmf(X, k, init='nndsvda')` |
| sklearn `KernelPCA(kernel='rbf', eigen_solver='arpack')` | `kernel_pca(X, kernel='rbf', eigen_solver='lanczos')` |
| R `cmdscale(d, k, eig=TRUE, add=TRUE)` | `classical_mds(D, k, metric='precomputed', add=True)` → `.embedding`, `.eigenvalues`, `.gof`, `.additive_constant` |
| sklearn `MDS(n_components=2, metric_mds=False, init='classical_mds')` | `mds(X, metric_mds=False)` |
| R `MASS::isoMDS(d, k=2)` | `mds(D, metric='precomputed', metric_mds=False)`(같은 목적 Stress-1, 다른 최적화) |
| R `MASS::sammon(d, k=2)` | `sammon(D, metric='precomputed')` |
| sklearn `Isomap(n_neighbors=5)`, `.reconstruction_error()` | `isomap(X)`, `.reconstruction_error()` |
| sklearn `LocallyLinearEmbedding(method='ltsa', random_state=s)` | `locally_linear_embedding(X, method='ltsa')`(`rng` 없음) |
| sklearn `SpectralEmbedding(affinity='rbf')`, `spectral_embedding(adjacency)` | `spectral_embedding(X, affinity='rbf')`, `spectral_embedding(A, affinity='precomputed')` |
| sklearn `TSNE(n_components=2, random_state=s).fit_transform(X)`, `.kl_divergence_` | `tsne(X).embedding`(`init='pca'`이면 `rng` 없음), `.kl_divergence` |
| openTSNE `TSNE(negative_gradient_method='fft')` | `tsne(X, method='fft')` |
| R `Rtsne(X, theta=0.5)` | `tsne(X, angle=0.5)`(R은 기본이 PCA로 50차원을 먼저 줄인다: `tsne(pca(X, 50).embedding)`) |
| umap-learn `UMAP(random_state=s).fit_transform(X)`, `.transform(Y)` | `umap(X, rng=…).embedding`, `.transform(Y, rng=…)` |
| R `uwot::umap(X, batch=TRUE)` | `umap(X, rng=…)`(같은 동기 꼴; uwot의 기본 `min_dist`는 0.01) |
| pynndescent `NNDescent` | `neighbors='nndescent'` |
| sklearn `trustworthiness(X, X_embedded, n_neighbors=5)` | 같은 이름 |
| sklearn `IncrementalPCA`, `SparsePCA`, `DictionaryLearning`, `LatentDirichletAllocation`, `MiniBatchNMF`, `CCA`·PLS; R `corresp`, `mca`, FactoMineR; pacmap, trimap; 확산 지도 | 없음(나중 또는 밖; 설계 문서 §1.2) |
| sklearn `LinearDiscriminantAnalysis.transform`, R `MASS::lda` | 나중의 `sd.classify` |
