# cluster 속도

이 파일은 `bench/speed/results/cluster.json`에서 만든 것이다. 손으로 고치지 않는다. 다시 만들기: `.venv/bin/python -m bench.speed table cluster` (재는 법은 bench/README.md).

- 시간: 한 번 부를 때의 시간. 한 배치가 20 ms 이상이 되도록 여러 번 부르고, 모든 라운드의 배치 중 가장 빠른 값이다.
- 배율(÷ sd): 비교 대상 시간 ÷ sd 시간. 1보다 크면 sd가 빠르다. **굵은 배율**은 sd가 느린 줄이다.
- 잡음: 그 줄의 측정마다 (배치 시간의 중앙값 ÷ 최솟값 − 1)을 구한 것 중 가장 큰 값.
- 스레드: sd의 스레드 수. 한 줄에 sd 스레드별 열이 있는 표에서 비교 대상은 그 라이브러리의 기본 스레드로 한 번 잰다(numpy·scipy의 원소별 계산은 1스레드). BLAS·LAPACK이 일의 대부분인 비교 대상은 BLAS 스레드를 sd와 같게 묶어 '스레드' 열이 있는 표로 보인다. R은 늘 1스레드다.
- 비고: 비교 대상이 sd보다 NaN을 1%p 넘게 더 내면 그 비율을 적는다(그만큼 일찍 끝나는 호출이 섞인다).

## 측정 환경

| 실행 | 커밋 | 기계 | 사례 수 |
|---|---|---|---:|
| 20261127-051007-25f6ac27+ | 25f6ac27 + 작업 중 변경 | Apple M4 Pro, 14코어, macOS 26.6.2 | 39 |
| 20261127-052619-846df10b+ | 846df10b + 작업 중 변경 | Apple M4 Pro, 14코어, macOS 26.6.2 | 3 |

라이브러리: stardust 0.0.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, pyarrow 26.0.1, statsmodels 0.15.0, sklearn 1.9.1, threadpoolctl 3.8.0, R 4.6.2, R cluster 2.1.8.3.

BLAS: R libopenblasp-r0.3.34.dylib, numpy accelerate, scipy Accelerate.

## 비교 대상을 잰 방법

- numpy, scipy, pandas, statsmodels, scikit-learn: 같은 프로세스에서 Python으로 부르고 sd와 같은 규칙으로 잰다. 입력은 그 라이브러리의 배열(numpy)로 미리 만들어 둔다.
- R: 실행 하나에 R 세션(`R --vanilla`) 하나를 띄우고, 사례의 입력을 float64 파일로 넘긴 뒤 R 안에서 같은 규칙(예열, 배치 크기를 두 배씩 늘려 20 ms 이상, 배치 중 가장 빠른 값)으로 잰다. R의 벡터 함수를 한 번 부르는 식이다. R은 1스레드로 돈다(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; R이 쓰는 BLAS는 위 'BLAS' 줄). R의 쓰레기 수집은 배치 안에서 끌 수 없어 배치 전에 한 번 돌린다. 사례의 비고에 R의 호출이 다른 점을 적는다.

## 계층 군집

- scipy ÷ sd: 1스레드 중앙값 1.44배(기하평균 1.75배, sd가 느린 줄 0/9), 10스레드 중앙값 2.34배(기하평균 3.85배, sd가 느린 줄 0/9)
- R ÷ sd: 1스레드 중앙값 2.12배(기하평균 2.06배, sd가 느린 줄 0/2), 10스레드 중앙값 3.17배(기하평균 3.16배, sd가 느린 줄 0/2)
- R cluster ÷ sd: 1스레드 중앙값 68.5배(기하평균 68.5배, sd가 느린 줄 0/1), 10스레드 중앙값 72.0배(기하평균 72.0배, sd가 느린 줄 0/1)

| 사례 | 크기 | sd 1스레드 | sd 10스레드 | scipy | ÷ sd 1 | ÷ sd 10 | R | ÷ sd 1 | ÷ sd 10 | R cluster | ÷ sd 1 | ÷ sd 10 | 잡음 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `linkage single p=5` † | 5000 | 28.5 ms | 4.91 ms | 60.2 ms | 2.11 | 12.3 | - | - | - | - | - | - | 5% |  |
| `linkage complete p=5` † | 5000 | 77.0 ms | 59.5 ms | 153 ms | 1.99 | 2.58 | 200 ms | 2.60 | 3.36 | - | - | - | 5% |  |
| `linkage average p=5` † | 5000 | 122 ms | 67.3 ms | 162 ms | 1.32 | 2.41 | 201 ms | 1.64 | 2.98 | - | - | - | 5% |  |
| `linkage weighted p=5` † | 5000 | 123 ms | 67.8 ms | 158 ms | 1.29 | 2.34 | - | - | - | - | - | - | 5% |  |
| `linkage ward p=5` † | 5000 | 114 ms | 68.0 ms | 156 ms | 1.38 | 2.30 | - | - | - | - | - | - | 4% |  |
| `linkage centroid p=5` † | 5000 | 109 ms | 78.7 ms | 156 ms | 1.44 | 1.99 | - | - | - | - | - | - | 8% |  |
| `linkage median p=5` † | 5000 | 106 ms | 78.2 ms | 157 ms | 1.47 | 2.00 | - | - | - | - | - | - | 4% |  |
| `linkage single p=5 matrix-free` † | 10^5 | 1.15 s | 158 ms | - | - | - | - | - | - | - | - | - | 3% | no comparator: scipy and R need the 40 GB condensed matrix |
| `linkage ward p=5 matrix-free` † | 10^5 | 46.8 s | 6.96 s | - | - | - | - | - | - | - | - | - | 8% | no comparator: scipy and R need the 40 GB condensed matrix |
| `divisive p=5` † | 10^3 | 5.31 ms | 5.06 ms | - | - | - | - | - | - | 364 ms | 68.5 | 72.0 | 7% | R cluster: R's diana gives its banner (the heights in leaf order), not the merges |
| `cophenet` | 5000 | 19.1 ms | 9.09 ms | 20.6 ms | 1.07 | 2.26 | - | - | - | - | - | - | 1% |  |
| `optimal_leaf_ordering` † | 10^3 | 61.9 ms | 8.60 ms | 431 ms | 6.97 | 50.1 | - | - | - | - | - | - | 1% | scipy: scipy 1.18.1's order is not always the optimal one (decision CL53) |

† 잴 때 다른 프로세스가 CPU를 1코어 넘게 썼다(사례의 `other_cpu`). 다시 재는 것이 좋다.

## 분할 (`kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `kmedoids`, `fuzzy_cmeans`)

- sklearn ÷ sd: 1스레드 중앙값 3.34배(기하평균 3.20배, sd가 느린 줄 1/4), 10스레드 중앙값 5.58배(기하평균 6.68배, sd가 느린 줄 0/4)
- R Lloyd ÷ sd: 1스레드 중앙값 15.6배(기하평균 15.6배, sd가 느린 줄 0/1), 10스레드 중앙값 82.3배(기하평균 82.3배, sd가 느린 줄 0/1)
- R ÷ sd: 1스레드 중앙값 2.29배(기하평균 2.87배, sd가 느린 줄 0/3), 10스레드 중앙값 12.1배(기하평균 14.8배, sd가 느린 줄 0/3)

| 사례 | 크기 | 스레드 | sd | sklearn | ÷ sd | R Lloyd | ÷ sd | R | ÷ sd | 잡음 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `kmeans p=10 k=10` † | 10^5 | 1 | 23.6 ms | 31.5 ms | 1.33 | 368 ms | 15.6 | 54.1 ms | 2.29 | 5% | R Lloyd: R's Lloyd has no tol: it runs until no label changes (117 iterations here, sd's tol 1e-4 stops at 11); R: R's default Hartigan-Wong moves single points: another local minimum |
| `kmeans p=10 k=10` † | 10^5 | 10 | 4.47 ms | 13.2 ms | 2.95 | 368 ms | 82.3 | 54.1 ms | 12.1 | 5% | R Lloyd: R's Lloyd has no tol: it runs until no label changes (117 iterations here, sd's tol 1e-4 stops at 11); R: R's default Hartigan-Wong moves single points: another local minimum |
| `minibatch_kmeans p=10 k=10` † | 10^5 | 1 | 5.90 ms | 89.1 ms | 15.1 | - | - | - | - | 4% |  |
| `minibatch_kmeans p=10 k=10` † | 10^5 | 10 | 3.20 ms | 104 ms | 32.5 | - | - | - | - | 4% |  |
| `bisecting_kmeans p=10 k=10` † | 10^5 | 1 | 31.4 ms | 30.8 ms | **0.98** | - | - | - | - | 2% |  |
| `bisecting_kmeans p=10 k=10` † | 10^5 | 10 | 8.87 ms | 22.4 ms | 2.53 | - | - | - | - | 11% |  |
| `kmedoids pam p=5 k=10` | 2000 | 1 | 100 ms | - | - | - | - | 689 ms | 6.87 | 0% |  |
| `kmedoids pam p=5 k=10` | 2000 | 10 | 15.7 ms | - | - | - | - | 689 ms | 43.9 | 1% |  |
| `kmedoids clara p=5 k=10` | 10^5 | 1 | 14.9 ms | - | - | - | - | 22.4 ms | 1.50 | 7% | R: R's own sampler: other samples |
| `kmedoids clara p=5 k=10` | 10^5 | 10 | 3.65 ms | - | - | - | - | 22.4 ms | 6.13 | 7% | R: R's own sampler: other samples |
| `fuzzy_cmeans p=10 k=10` † | 10^5 | 1 | 772 ms | - | - | - | - | - | - | 1% | no comparator: e1071 is not installed and R's fanny minimizes another objective on n x n distances |
| `fuzzy_cmeans p=10 k=10` † | 10^5 | 10 | 91.4 ms | - | - | - | - | - | - | 5% | no comparator: e1071 is not installed and R's fanny minimizes another objective on n x n distances |
| `kmeans p=50 k=100` † | 10^5 | 1 | 150 ms | 803 ms | 5.36 | - | - | - | - | 8% | the same start (the data's first 100 rows) |
| `kmeans p=50 k=100` † | 10^5 | 10 | 38.4 ms | 316 ms | 8.21 | - | - | - | - | 4% | the same start (the data's first 100 rows) |

R Lloyd, R은(는) 스레드 수와 상관없이 한 번 잰 값이다(그 라이브러리의 기본 스레드).

† 잴 때 다른 프로세스가 CPU를 1코어 넘게 썼다(사례의 `other_cpu`). 다시 재는 것이 좋다.

## 밀도 군집

- sklearn ÷ sd: 1스레드 중앙값 1.57배(기하평균 4.11배, sd가 느린 줄 0/5), 10스레드 중앙값 13.6배(기하평균 20.7배, sd가 느린 줄 0/5)

| 사례 | 크기 | sd 1스레드 | sd 10스레드 | sklearn | ÷ sd 1 | ÷ sd 10 | 잡음 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `dbscan p=2 eps=0.05` † | 20000 | 21.1 ms | 3.93 ms | 32.2 ms | 1.53 | 8.19 | 4% | sklearn: border points join the first cluster that reaches them (sd: the nearest core point's) |
| `hdbscan p=2` † | 20000 | 42.9 ms | 13.5 ms | 670 ms | 15.6 | 49.7 | 3% |  |
| `optics p=2` † | 5000 | 46.2 ms | 18.2 ms | 1.35 s | 29.2 | 74.1 | 12% | scikit-learn starts from the first input point, sd from the first in key order (CL7): other orders |
| `dbscan p=10 eps=0.9` † | 20000 | 297 ms | 34.3 ms | 466 ms | 1.57 | 13.6 | 5% | sklearn: border points join the first cluster that reaches them (sd: the nearest core point's) |
| `hdbscan p=10` | 20000 | 1.36 s | 156 ms | 1.47 s | 1.08 | 9.38 | 2% |  |
| `optics p=2` † | 20000 | 677 ms | 143 ms | - | - | - | 7% | no comparator: scikit-learn's step loop in Python takes minutes at this size |

† 잴 때 다른 프로세스가 CPU를 1코어 넘게 썼다(사례의 `other_cpu`). 다시 재는 것이 좋다.

## 가우스 혼합

- sklearn ÷ sd: 1스레드 중앙값 1.99배(기하평균 1.75배, sd가 느린 줄 1/6), 10스레드 중앙값 6.96배(기하평균 7.23배, sd가 느린 줄 0/6)

| 사례 | 크기 | 스레드 | sd | sklearn | ÷ sd | 잡음 | 비고 |
|---|---|---:|---:|---:|---:|---:|---|
| `gaussian_mixture full p=10 k=5` † | 20000 | 1 | 7.70 ms | 15.5 ms | 2.01 | 4% | each library's own k-means start; the blobs are well separated, so both reach one fixed point |
| `gaussian_mixture full p=10 k=5` † | 20000 | 10 | 1.94 ms | 16.2 ms | 8.39 | 39% | each library's own k-means start; the blobs are well separated, so both reach one fixed point |
| `gaussian_mixture full p=10 k=5 same start` † | 20000 | 1 | 7.12 ms | 15.7 ms | 2.20 | 3% | the same start (weights, means, precisions) on both sides: the same EM iterations |
| `gaussian_mixture full p=10 k=5 same start` † | 20000 | 10 | 1.50 ms | 15.8 ms | 10.5 | 30% | the same start (weights, means, precisions) on both sides: the same EM iterations |
| `gaussian_mixture tied p=10 k=5 same start` † | 20000 | 1 | 6.97 ms | 12.5 ms | 1.80 | 4% | the same start (weights, means, precisions) on both sides: the same EM iterations |
| `gaussian_mixture tied p=10 k=5 same start` † | 20000 | 10 | 1.78 ms | 12.4 ms | 6.98 | 22% | the same start (weights, means, precisions) on both sides: the same EM iterations |
| `gaussian_mixture diag p=10 k=5 same start` † | 20000 | 1 | 4.01 ms | 7.91 ms | 1.97 | 2% | the same start (weights, means, precisions) on both sides: the same EM iterations |
| `gaussian_mixture diag p=10 k=5 same start` † | 20000 | 10 | 1.24 ms | 7.92 ms | 6.38 | 29% | the same start (weights, means, precisions) on both sides: the same EM iterations |
| `gaussian_mixture spherical p=10 k=5 same start` † | 20000 | 1 | 4.02 ms | 8.53 ms | 2.12 | 3% | the same start (weights, means, precisions) on both sides: the same EM iterations |
| `gaussian_mixture spherical p=10 k=5 same start` † | 20000 | 10 | 1.24 ms | 8.61 ms | 6.95 | 29% | the same start (weights, means, precisions) on both sides: the same EM iterations |
| `gaussian_mixture full p=50 k=5 same start` † | 20000 | 1 | 267 ms | 233 ms | **0.87** | 2% | the same start on both sides: the same EM iterations |
| `gaussian_mixture full p=50 k=5 same start` † | 20000 | 10 | 39.7 ms | 208 ms | 5.23 | 6% | the same start on both sides: the same EM iterations |

† 잴 때 다른 프로세스가 CPU를 1코어 넘게 썼다(사례의 `other_cpu`). 다시 재는 것이 좋다.

## 지표 (`gap_statistic`, `gower_distance` 포함)

- sklearn ÷ sd: 1스레드 중앙값 3.76배(기하평균 3.76배, sd가 느린 줄 0/1), 10스레드 중앙값 19.2배(기하평균 19.2배, sd가 느린 줄 0/1)
- R cluster ÷ sd: 1스레드 중앙값 4.11배(기하평균 4.00배, sd가 느린 줄 0/4), 10스레드 중앙값 24.9배(기하평균 27.5배, sd가 느린 줄 0/4)
- sklearn ÷ sd: 1스레드 중앙값 3.76배(기하평균 3.76배, sd가 느린 줄 0/1), 10스레드 중앙값 19.2배(기하평균 19.2배, sd가 느린 줄 0/1)

| 사례 | 크기 | 스레드 | sd | sklearn | ÷ sd | R cluster | ÷ sd | sklearn | ÷ sd | 잡음 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `silhouette_score p=10 k=5` † | 20000 | 1 | 312 ms | 1.17 s | 3.76 | 1.51 s | 4.84 | 1.17 s | 3.76 | 4% | R cluster: R needs the 1.6 GB dist object (made in the timing) |
| `silhouette_score p=10 k=5` † | 20000 | 10 | 60.5 ms | 1.16 s | 19.2 | 1.51 s | 24.9 | 1.16 s | 19.2 | 6% | R cluster: R needs the 1.6 GB dist object (made in the timing) |
| `calinski_harabasz_score p=10 k=5` | 10^5 | 1 | 4.25 ms | - | - | - | - | - | - | 5% |  |
| `calinski_harabasz_score p=10 k=5` | 10^5 | 10 | 1.87 ms | - | - | - | - | - | - | 21% |  |
| `davies_bouldin_score p=10 k=5` † | 10^5 | 1 | 4.36 ms | - | - | - | - | - | - | 3% |  |
| `davies_bouldin_score p=10 k=5` † | 10^5 | 10 | 1.85 ms | - | - | - | - | - | - | 21% |  |
| `adjusted_rand_score k=20` † | 10^6 | 1 | 8.13 ms | - | - | - | - | - | - | 1% |  |
| `adjusted_rand_score k=20` † | 10^6 | 10 | 4.89 ms | - | - | - | - | - | - | 6% |  |
| `normalized_mutual_info_score k=20` † | 10^6 | 1 | 7.99 ms | - | - | - | - | - | - | 3% |  |
| `normalized_mutual_info_score k=20` † | 10^6 | 10 | 4.94 ms | - | - | - | - | - | - | 6% |  |
| `adjusted_mutual_info_score k=20` † | 10^5 | 1 | 1.66 ms | - | - | - | - | - | - | 1% | the expected MI by the hypergeometric ratio recurrence (scikit-learn: lgamma at every cell count) |
| `adjusted_mutual_info_score k=20` † | 10^5 | 10 | 992 µs | - | - | - | - | - | - | 3% | the expected MI by the hypergeometric ratio recurrence (scikit-learn: lgamma at every cell count) |
| `gap_statistic kmeans p=4 k_max=8 B=50` † | 10^3 | 1 | 192 ms | - | - | 1.02 s | 5.29 | - | - | 2% | R cluster: R's kmeans is Hartigan-Wong with one random start, its reference sets another stream |
| `gap_statistic kmeans p=4 k_max=8 B=50` † | 10^3 | 10 | 23.5 ms | - | - | 1.02 s | 43.4 | - | - | 5% | R cluster: R's kmeans is Hartigan-Wong with one random start, its reference sets another stream |
| `gap_statistic pam p=2 k_max=6 B=20` † | 300 | 1 | 182 ms | - | - | 613 ms | 3.37 | - | - | 4% |  |
| `gap_statistic pam p=2 k_max=6 B=20` † | 300 | 10 | 24.6 ms | - | - | 613 ms | 24.9 | - | - | 4% |  |
| `gower_distance p=6` † | 3000 | 1 | 50.7 ms | - | - | 150 ms | 2.95 | - | - | 5% |  |
| `gower_distance p=6` † | 3000 | 10 | 7.01 ms | - | - | 150 ms | 21.4 | - | - | 5% |  |

R cluster, sklearn은(는) 스레드 수와 상관없이 한 번 잰 값이다(그 라이브러리의 기본 스레드).

† 잴 때 다른 프로세스가 CPU를 1코어 넘게 썼다(사례의 `other_cpu`). 다시 재는 것이 좋다.

## Affinity Propagation

- sklearn ÷ sd: 1스레드 중앙값 2.35배(기하평균 2.35배, sd가 느린 줄 0/2), 10스레드 중앙값 10.5배(기하평균 10.4배, sd가 느린 줄 0/2)

| 사례 | 크기 | sd 1스레드 | sd 10스레드 | sklearn | ÷ sd 1 | ÷ sd 10 | 잡음 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `affinity_propagation p=2` † | 2000 | 1.00 s | 205 ms | 2.45 s | 2.45 | 12.0 | 10% | both run the 200 iterations (no convergence); scikit-learn's noise gives other exemplars |
| `affinity_propagation p=10 preference` † | 3000 | 583 ms | 146 ms | 1.32 s | 2.26 | 9.04 | 3% |  |

† 잴 때 다른 프로세스가 CPU를 1코어 넘게 썼다(사례의 `other_cpu`). 다시 재는 것이 좋다.
