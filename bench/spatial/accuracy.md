# sd.spatial 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/spatial.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/spatial/ref의 Arb와 정확한 유리수 참값이다(tools/spatial_ref.py; 시험 점은 tests/spatial/points.py). 묶음(참값 파일)마다 한 줄 묶음이고 거리 방법(가중 판 w_…)이 영역이며 짝 하나가 점 하나다. sd의 값은 거리 함수(sd.spatial.euclidean(u, v) 등)이고 pdist·cdist는 같은 짝에 같은 비트다(한 짝 커널). 계약은 tests/spatial/refdata.py의 한계다(docs/design/spatial.md §4; 테스트가 같은 함수로 판정한다): 합의 깊이 m = min(n, 32) + ⌈n/32⌉ − 1의 γ로 적은 거리마다의 상대 한계, chebyshev와 가중치 없는 hamming·jaccard는 올바른 반올림, cosine·correlation은 절대와 상대를 섞은 한계, mahalanobis는 |z|ᵀ|VI||z|에 대한 한계, haversine은 20u다. 비교 패키지는 scipy 1.18.1(같은 이름의 거리 함수), scikit-learn 1.9.1(euclidean_distances·manhattan_distances·cosine_distances·haversine_distances: 전개식과 1 − cos의 꼴), R 4.6.2의 stats::dist(euclidean·manhattan·maximum·minkowski, 좌표가 음이 아닌 정상 수이고 0/0 항이 없는 짝의 canberra(R의 항은 |x − y|/|x + y|이고 분모가 DBL_MIN 아래인 항을 빼고 비율을 다시 맞춘다), counting의 jaccard에 binary)이고 같은 짝에서 그 자리에서 잰다. neighbors 묶음(SPT2)은 neighbors.npz의 질의마다 k = 5 이웃(sd.spatial.kneighbors; euclidean·cityblock·chebyshev)이 점 하나이고 거리와 점 집합이 영역이다: 격자 집합은 Arb의 (RN(d), 색인) 순서와 거리가 같으면 0, 아니면 inf이고, 그 밖은 계약 η 안의 참 kNN인지(돌려준 점의 참 거리 ≤ d₍ₖ₎(1 + η), 빠진 점 ≥ d₍ₖ₎(1 − η))를 ηd₍ₖ₎ 단위로 잰다. 비교 패키지는 scipy의 KDTree.query와 scikit-learn의 NearestNeighbors(algorithm='brute')다. kdtree 묶음(SPT3)은 같은 점과 판정을 sd.spatial.KDTree(X).query(Q, k=5, p=1·2·inf)로 재고, 비교 패키지는 scipy와 scikit-learn의 KDTree.query다.

- ULP: sd의 값과 참값을 double로 반올림한 값 사이에 있는 double의 수(ulp_distance; 0이면 올바르게 반올림한 값, 1이면 그 이웃). 99.9%는 그 분위수다.
- 상대오차: |sd − 참값|/|참값|.
- 최대 절대오차: |sd − 참값|의 최대.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.
- 비교 패키지(scipy 1.18.1, R 4.6.2, sklearn): 같은 참값, 같은 점에서 같은 척도로 잰 그 라이브러리의 오차다. 같은 양을 계산하는 공개 함수가 없으면 —, R이 없는 기계에서는 not run, 함수가 입력 전체를 거부하면 에러, 시간(120·600초)·메모리(2GB) 한도를 넘으면 멈춤, R의 glm이 기본 설정으로 시작값을 찾지 못하면 시작값 필요다. 제외: 참값은 유한한데 그 라이브러리가 NaN·inf를 낸 점의 수(정의가 다르거나 실패한 점; 다른 열에서 빠진다). compare는 이 열들의 변화를 알리기만 하고 종료 코드에 넣지 않는다.

측정: 20261122-225345-3e3a1b4+ (2026-11-22T22:53:45, 3e3a1b4 dirty), 20261123-013752-8832c53+ (2026-11-23T01:37:52, 8832c53 dirty), 20261123-214327-eed868f+ (2026-11-23T21:43:27, eed868f dirty).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | 최대 상대오차 | 최대 절대오차 | 계약 대비 | scipy 최대 ULP | scipy 99.9% ULP | scipy 최대 상대오차 | scipy 계약 대비 | scipy 제외 | R 최대 ULP | R 99.9% ULP | R 최대 상대오차 | R 계약 대비 | sklearn 최대 ULP | sklearn 99.9% ULP | sklearn 최대 상대오차 | sklearn 계약 대비 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ball` | euclidean/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/nn/normal3 | 50 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/nn/normal16 | 30 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | euclidean/nn/mixture8 | 30 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/nn/normal3 | 50 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/nn/normal16 | 30 | 2 | 2 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | cityblock/nn/mixture8 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/nn/normal3 | 50 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/nn/normal16 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `ball` | chebyshev/nn/mixture8 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | uniform2/12 | 55 | 1 | 1 |  | 2.12e-16 | 0.34 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | uniform2/1000 | 64 | 1 | 1 |  | 2.15e-16 | 0.35 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | normal2/1000 | 47 | 1 | 1 |  | 3.14e-16 | 0.43 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | cluster2/1000 | 27 | 1 | 1 |  | 1.01e-15 | 0.41 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | ball2/1000 | 45 | 1 | 1 |  | 1.84e-16 | 0.49 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | dup2 | 61 | 0 | 0 |  | 1.48e-16 | 0.12 |  |  |  |  | 45 | — | — | — | — | — | — | — | — |
| `barycentric` | line2+1 | 8 | 0 | 0 |  | 4.44e-16 | 0.12 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | mixed2/1e150 | 34 | 1 | 1 |  | 2.75e-16 | 0.37 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | mixed2/1e300 | 33 | 0 | 0 |  | 1 | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | dblmax2 | 41 | 0 | 0 |  | 0.998 | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | ulp2/200 | 51 | 1 | 1 |  | 4.47e-16 | 0.41 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | minimal2 | 31 | 0 | 0 |  | 0 | 0 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | minimal2+1 | 64 | 0 | 0 |  | 0 | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | uniform3/12 | 25 | 1 | 1 |  | 1.76e-15 | 0.5 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | uniform3/1000 | 56 | 2 | 2 |  | 1.10e-15 | 0.51 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | normal3/1000 | 22 | 2 | 2 |  | 1.81e-15 | 0.59 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | cluster3/1000 | 14 | 2 | 2 |  | 2.94e-15 | 0.43 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | ball3/1000 | 34 | 1 | 1 |  | 4.55e-16 | 0.61 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | dup3 | 39 | 0 | 0 |  | 2.22e-16 | 0.2 |  |  |  |  | 22 | — | — | — | — | — | — | — | — |
| `barycentric` | mixed3/1e150 | 16 | 0 | 0 |  | 0.998 | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | ulp3/200 | 40 | 1 | 1 |  | 1.16e-15 | 0.52 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | minimal3 | 13 | 0 | 0 |  | 0 | 0 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | minimal3+1 | 25 | 0 | 0 |  | 1.11e-16 | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | grid2/30 | 64 | 0 | 0 |  | 0 | 0 |  |  |  |  | 36 | — | — | — | — | — | — | — | — |
| `barycentric` | grid2/5 | 64 | 0 | 0 |  | 0 | 0 |  |  |  |  | 40 | — | — | — | — | — | — | — | — |
| `barycentric` | grid3/6 | 64 | 0 | 0 |  | 0 | 0 |  |  |  |  | 54 | — | — | — | — | — | — | — | — |
| `barycentric` | grid3/2 | 64 | 0 | 0 |  | 0 | 0 |  |  |  |  | 39 | — | — | — | — | — | — | — | — |
| `barycentric` | cube/5 | 64 | 0 | 0 |  | 0 | 0 |  |  |  |  | 58 | — | — | — | — | — | — | — | — |
| `barycentric` | circle2/int | 55 | 0 | 0 |  | 1.27e-16 | 0.18 |  |  |  |  | 45 | — | — | — | — | — | — | — | — |
| `barycentric` | knight2 | 55 | 0 | 0 |  | 1.48e-16 | 0.12 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | sphere3/int | 35 | 0 | 0 |  | 1.37e-15 | 0.18 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | sphere3/int9 | 25 | 0 | 0 |  | 2.22e-16 | 0.14 |  |  |  |  | 21 | — | — | — | — | — | — | — | — |
| `barycentric` | boundary2 | 64 | 0 | 0 |  | 2.22e-16 | 0.17 |  |  |  |  | 13 | — | — | — | — | — | — | — | — |
| `barycentric` | boundary3 | 64 | 0 | 0 |  | 1.78e-16 | 0.21 |  |  |  |  | 21 | — | — | — | — | — | — | — | — |
| `barycentric` | parabola2/int | 8 | 0 | 0 |  | 1.82e-14 | 0.25 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | plane3+1 | 1 | 0 | 0 |  | 1.66e-15 | 0.18 |  |  |  |  | 0 | — | — | — | — | — | — | — | — |
| `barycentric` | polygon2/200 | 52 | 1 | 1 |  | 3.15e-16 | 0.44 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | grid2/30+1e-13 | 64 | 1 | 1 |  | 1.39e-16 | 0.35 |  |  |  |  | 40 | — | — | — | — | — | — | — | — |
| `barycentric` | parabola2/1e-9 | 18 | 1 | 1 |  | 5.78e-15 | 0.31 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | parabola2/1e-14 | 9 | 1 | 1 |  | 1.13e-15 | 0.4 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | far2/1e8grid | 64 | 0 | 0 |  | 1.16e-16 | 1.6e-6 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | far2/1e6uniform | 64 | 0 | 0 |  | 1.42e-16 | 0.23 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `barycentric` | far3/1e8grid | 64 | 1 | 1 |  | 1.76e-16 | 0.4 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `counting` | hamming | 80 | 0 | 0 | 6.34e-17 |  | 0.88 | 0 | 0 | 6.34e-17 | 0.88 | 0 | — | — | — | — | — | — | — | — |
| `counting` | jaccard | 80 | 0 | 0 | 7.07e-17 |  | 0.93 | 0 | 0 | 7.07e-17 | 0.93 | 0 | 0 | 0 | 7.07e-17 | 0.93 | — | — | — | — |
| `counting` | w_hamming | 80 | 2 | 2 | 3.01e-16 |  | 0.12 | 2 | 2 | 2.95e-16 | 0.12 | 0 | — | — | — | — | — | — | — | — |
| `counting` | w_jaccard | 80 | 2 | 2 | 2.91e-16 |  | 0.12 | 1 | 1 | 2.20e-16 | 0.12 | 0 | — | — | — | — | — | — | — | — |
| `delaunay_exact` | d2 | 26 |  |  |  |  | 0 |  |  |  | **inf 초과** |  | — | — | — | — | — | — | — | — |
| `delaunay_exact` | d3 | 22 |  |  |  |  | 0 |  |  |  | **inf 초과** |  | — | — | — | — | — | — | — | — |
| `distances` | braycurtis | 624 | 5 | 5 | 9.31e-16 |  | 0.35 | 3 | 3 | 4.20e-16 | 0.35 | 0 | — | — | — | — | — | — | — | — |
| `distances` | canberra | 645 | 3 | 3 | 4.50e-16 |  | 0.36 | 2 | 2 | 2.12e-16 | 0.36 | 0 | 2 | 2 | 2.00e-16 | 0.29 | — | — | — | — |
| `distances` | chebyshev | 645 | 0 | 0 | 1.09e-16 |  | 1 | 0 | 0 | 1.09e-16 | 1 | 17 | 0 | 0 | 1.09e-16 | 1 | — | — | — | — |
| `distances` | cityblock | 645 | 3 | 3 | 5.31e-16 |  | 0.36 | 2 | 2 | 2.50e-16 | 0.36 | 0 | 17 | 17 | 3.34e-15 | 0.36 | 17 | 17 | 3.34e-15 | 0.36 |
| `distances` | correlation | 606 | 4.16e+18 | 4.16e+18 | 1.60e-4 |  | 0.39 | inf | inf | inf | **inf 초과** | 0 | — | — | — | — | — | — | — | — |
| `distances` | cosine | 626 | 9.73e+14 | 9.73e+14 | 0.128 |  | 0.41 | inf | inf | inf | **inf 초과** | 0 | — | — | — | — | 4.36e+18 | 4.36e+18 | 1.14e+12 | **3.24e+15 초과** |
| `distances` | euclidean | 645 | 2 | 2 | 2.57e-16 |  | 0.46 | inf | inf | inf | **inf 초과** | 0 | inf | inf | inf | **inf 초과** | inf | inf | inf | **inf 초과** |
| `distances` | mahalanobis | 492 | 2 | 2 | 2.01e-16 |  | 0.29 | inf | inf | inf | **inf 초과** | 0 | — | — | — | — | — | — | — | — |
| `distances` | minkowski_p0.5 | 645 | 6 | 6 | 6.86e-16 |  | 0.2 | 4 | 4 | 6.37e-16 | 0.2 | 0 | 33 | 33 | 3.81e-15 | 0.2 | — | — | — | — |
| `distances` | minkowski_p1.5 | 645 | 3 | 3 | 3.78e-16 |  | 0.17 | 2.02e+03 | 2.02e+03 | 1 | **2.02e+03 초과** | 0 | 2.02e+03 | 2.02e+03 | 1 | **2.02e+03 초과** | — | — | — | — |
| `distances` | minkowski_p3 | 645 | 1 | 1 | 1.94e-16 |  | 0.21 | inf | inf | inf | **inf 초과** | 0 | inf | inf | inf | **inf 초과** | — | — | — | — |
| `distances` | minkowski_p7.5 | 645 | 2 | 2 | 3.08e-16 |  | 0.4 | inf | inf | inf | **inf 초과** | 0 | inf | inf | inf | **inf 초과** | — | — | — | — |
| `distances` | seuclidean | 645 | 2 | 2 | 2.85e-16 |  | 0.45 | inf | inf | inf | **inf 초과** | 0 | — | — | — | — | — | — | — | — |
| `distances` | sqeuclidean | 645 | 3 | 3 | inf |  | 0.68 | 2 | 2 | inf | 1 | 0 | — | — | — | — | inf | inf | inf | **inf 초과** |
| `distances` | w_braycurtis | 584 | 7 | 7 | 7.83e-16 |  | 0.46 | 3 | 3 | 4.08e-16 | 0.46 | 0 | — | — | — | — | — | — | — | — |
| `distances` | w_canberra | 645 | 2 | 2 | 4.28e-16 |  | 0.46 | 2 | 2 | 2.67e-16 | 0.46 | 0 | — | — | — | — | — | — | — | — |
| `distances` | w_chebyshev | 645 | 0 | 0 | 1.09e-16 |  | 1 | 0 | 0 | 1.09e-16 | 1 | 17 | — | — | — | — | — | — | — | — |
| `distances` | w_cityblock | 645 | 3 | 3 | 4.37e-16 |  | 0.35 | 2 | 2 | 2.41e-16 | 0.43 | 0 | — | — | — | — | — | — | — | — |
| `distances` | w_correlation | 567 | 4.19e+18 | 4.19e+18 | 0.00952 |  | 0.21 | inf | inf | inf | **inf 초과** | 0 | — | — | — | — | — | — | — | — |
| `distances` | w_cosine | 586 | 4.13e+18 | 4.13e+18 | 0.00622 |  | 0.25 | inf | inf | inf | **inf 초과** | 0 | — | — | — | — | — | — | — | — |
| `distances` | w_euclidean | 645 | 2 | 2 | 2.71e-16 |  | 0.42 | inf | inf | inf | **inf 초과** | 0 | — | — | — | — | — | — | — | — |
| `distances` | w_minkowski_p3 | 645 | 2 | 2 | 2.07e-16 |  | 0.22 | inf | inf | inf | **inf 초과** | 0 | — | — | — | — | — | — | — | — |
| `distances` | w_sqeuclidean | 645 | 4 | 4 | inf |  | 0.46 | 2 | 2 | inf | 1 | 0 | — | — | — | — | — | — | — | — |
| `geom_measure` | perimeter | 26 | 16 | 16 | 2.31e-15 |  | 0.1 | inf | inf | inf |  | 2 | — | — | — | — | — | — | — | — |
| `geom_measure` | area | 26 | 4 | 4 | 6.09e-16 |  | 0.32 | inf | inf | inf |  | 1 | — | — | — | — | — | — | — | — |
| `geom_measure` | surface | 22 | 9 | 9 | 1.25e-15 |  | 0.16 | inf | inf | inf |  | 2 | — | — | — | — | — | — | — | — |
| `geom_measure` | volume | 22 | 6 | 6 | 7.48e-16 |  | 0.083 | inf | inf | inf |  | 1 | — | — | — | — | — | — | — | — |
| `haversine` | haversine | 32 | 1 | 1 | 1.58e-16 |  | 0.071 | — | — | — | — | — | — | — | — | — | 6.71e+7 | 6.71e+7 | 9.49e-9 | **4.27e+6 초과** |
| `kdtree` | euclidean/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/nn/normal3 | 50 | 1 | 1 |  |  | 0.26 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/nn/normal16 | 30 | 1 | 1 |  |  | 0.15 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | euclidean/nn/mixture8 | 30 | 1 | 1 |  |  | 0.26 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/nn/normal3 | 50 | 1 | 1 |  |  | 0.27 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/nn/normal16 | 30 | 2 | 2 |  |  | 0.16 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | cityblock/nn/mixture8 | 30 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/nn/normal3 | 50 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/nn/normal16 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `kdtree` | chebyshev/nn/mixture8 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/nn/normal3 | 50 | 1 | 1 |  |  | 0.26 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/nn/normal16 | 30 | 1 | 1 |  |  | 0.15 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | euclidean/nn/mixture8 | 30 | 1 | 1 |  |  | 0.26 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/nn/normal3 | 50 | 1 | 1 |  |  | 0.27 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/nn/normal16 | 30 | 2 | 2 |  |  | 0.16 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | cityblock/nn/mixture8 | 30 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/far/1e8 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/far/1e15 | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/shell/3 | 3 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/dup/3 | 10 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/ties/2 | 5 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/lattice/int2d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/lattice/int3d | 40 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/lattice/bigint3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/lattice/dyadic3 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/nn/normal3 | 50 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/nn/normal16 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `neighbors` | chebyshev/nn/mixture8 | 30 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `sphere` | sphere/uniform | 20 | 1 | 1 |  |  | 0.06 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `sphere` | sphere/close | 20 | 1 | 1 |  |  | 0.096 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `sphere` | sphere/antipode | 20 | 2 | 2 |  |  | 0.093 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `sphere` | sphere/poles | 20 | 1 | 1 |  |  | 0.06 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `sphere` | sphere/dateline | 20 | 1 | 1 |  |  | 0.082 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `sphere` | sphere/ties | 20 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | uniform2/12 | 26 | 1 | 1 |  |  | 0.48 | 2 | 2 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | uniform2/1000 | 3964 | 38 | 10 |  |  | 0.96 | 44 | 12 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | normal2/1000 | 3976 | 847 | 81 |  |  | 0.92 | 800 | 547 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | cluster2/1000 | 3970 | 10 | 1 |  |  | 0.97 | 25 | 16 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | ball2/1000 | 3930 | 396 | 19 |  |  | 0.92 | 1.84e+03 | 174 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | dup2 | 82 | 3 | 3 |  |  | 0.47 | 2 | 2 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | line2+1 | 16 | 1 | 1 |  |  | 0.47 | 6 | 6 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | mixed2/1e150 | 146 | 3 | 3 |  |  | 0.81 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | mixed2/1e300 | 146 | 6 | 6 |  |  | 0.57 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | subnormal2 | 140 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | dblmax2 | 140 | 29 | 29 |  |  | 0.61 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | ulp2/200 | 396 | 4.36e+18 | 4.36e+18 |  |  | 0.46 | 8.75e+18 | 8.75e+18 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | minimal2 | 2 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | minimal2+1 | 2 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | uniform3/12 | 66 | 2 | 2 |  |  | 0.79 | 19 | 19 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | uniform3/1000 | 19110 | 241 | 8 |  |  | 0.93 | 1.04e+03 | 309 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | normal3/1000 | 19419 | 814 | 46 |  |  | 0.94 | 2.06e+03 | 476 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | cluster3/1000 | 18423 | 2.21e+03 | 10 |  |  | 0.98 | 5.43e+5 | 2.43e+04 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | ball3/1000 | 18390 | 385 | 62 |  |  | 0.94 | 2.31e+03 | 523 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | dup3 | 396 | 3 | 3 |  |  | 0.73 | 4.37e+18 | 4.37e+18 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | mixed3/1e150 | 492 | 46 | 46 |  |  | 0.55 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | mixed3/1e300 | 525 | 294 | 294 |  |  | 0.79 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | subnormal3 | 501 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | dblmax3 | 486 | 88 | 88 |  |  | 0.75 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | ulp3/200 | 1722 | 8.73e+18 | 8.73e+18 |  |  | 0.53 | 8.78e+18 | 8.78e+18 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | minimal3 | 3 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | minimal3+1 | 3 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | grid2/30 | 1682 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | grid2/5 | 32 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | grid3/6 | 375 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | grid3/2 | 3 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | cube/5 | 192 | 0 | 0 |  |  | 0 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | circle2/int | 2 | 0 | 0 |  |  | 0 | 4.38e+18 | 4.38e+18 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | knight2 | 16 | 1 | 1 |  |  | 0.22 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | sphere3/int | 3 | 0 | 0 |  |  | 0 | 4.37e+18 | 4.37e+18 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | sphere3/int9 | 3 | 0 | 0 |  |  | 0 | 4.37e+18 | 4.37e+18 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | boundary2 | 126 | 1 | 1 |  |  | 0.55 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | boundary3 | 636 | 14 | 14 |  |  | 0.86 | 12 | 12 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | parabola2/int | 396 | 0 | 0 |  |  | 0 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | moment3 | 135 | 0 | 0 |  |  | 0 | 1 | 1 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | plane3+1 | 135 | 3 | 3 |  |  | 0.57 | 3 | 3 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | polygon2/200 | 396 | 8.72e+18 | 8.72e+18 |  |  | 0.47 | 2.17e+17 | 2.17e+17 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | grid2/30+1e-13 | 3564 | 2 | 1 |  |  | 0.86 | 1.71e+16 | 1.71e+16 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | parabola2/1e-9 | 596 | 3.75e+15 | 3.75e+15 |  |  | 0.39 | 7.94e+14 | 7.94e+14 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | parabola2/1e-14 | 596 | 4.39e+18 | 4.39e+18 |  |  | 0.46 | 8.79e+18 | 8.79e+18 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | far2/1e8grid | 1682 | 1 | 1 |  |  | 0.67 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | far2/1e6uniform | 1968 | 0 | 0 |  |  | 0.52 | 8.19e+04 | 8.19e+04 |  |  | 0 | — | — | — | — | — | — | — | — |
| `voronoi_vertices` | far3/1e8grid | 375 | 1 | 1 |  |  | 0.67 | 0 | 0 |  |  | 0 | — | — | — | — | — | — | — | — |
