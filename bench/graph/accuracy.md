# sd.graph 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/graph.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/graph/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다.

- ULP: sd의 값과 참값을 double로 반올림한 값 사이에 있는 double의 수(ulp_distance; 0이면 올바르게 반올림한 값, 1이면 그 이웃). 99.9%는 그 분위수다.
- 상대오차: |sd − 참값|/|참값|.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.
- 비교 패키지(networkx, scipy 1.18.1): 같은 참값, 같은 점에서 같은 척도로 잰 그 라이브러리의 오차다. 같은 양을 계산하는 공개 함수가 없으면 —, R이 없는 기계에서는 not run, 함수가 입력 전체를 거부하면 에러, 시간(300초)·메모리(2GB) 한도를 넘으면 멈춤, R의 glm이 기본 설정으로 시작값을 찾지 못하면 시작값 필요다. 제외: 참값은 유한한데 그 라이브러리가 NaN·inf를 낸 점의 수(정의가 다르거나 실패한 점; 다른 열에서 빠진다). compare는 이 열들의 변화를 알리기만 하고 종료 코드에 넣지 않는다.

측정: 20261123-061938-3e69193+ (2026-11-23T06:19:38, 3e69193 dirty), 20261123-064954-3e69193+ (2026-11-23T06:49:54, 3e69193 dirty), 20261123-080952-3e69193+ (2026-11-23T08:09:52, 3e69193 dirty).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | 최대 상대오차 | 계약 대비 | networkx 최대 ULP | networkx 99.9% ULP | networkx 최대 상대오차 | networkx 계약 대비 | scipy 최대 ULP | scipy 99.9% ULP | scipy 최대 상대오차 | scipy 계약 대비 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `assignment` | cost/float7x7 min | 1 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | cost/float7x7 max | 1 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | bipartite/float8x8 min | 1 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | bipartite/float8x8 max | 1 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | random 10x10 min | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | random 10x10 max | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | random 25x25 min | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | random 25x25 max | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | random 40x40 min | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | random 40x40 max | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | random 12x30 min | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | random 12x30 max | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | near ties 30x30 min | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `assignment` | near ties 30x30 max | 4 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `betweenness` | complete/8 weighted | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — |
| `betweenness` | complete/8 unweighted | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — |
| `betweenness` | components/40 weighted | 80 | 2 | 2 | 2.04e-16 | 0.02 | 1 | 1 | 2.04e-16 | 0.02 | — | — | — | — |
| `betweenness` | components/40 unweighted | 80 | 2 | 2 | 2.84e-16 | 0.031 | 2 | 2 | 2.84e-16 | 0.031 | — | — | — | — |
| `betweenness` | dag/30 weighted | 60 | 1 | 1 | 2.40e-16 | 0.022 | 1 | 1 | 2.40e-16 | 0.022 | — | — | — | — |
| `betweenness` | dag/30 unweighted | 60 | 3 | 3 | 3.17e-16 | 0.034 | 3 | 3 | 3.17e-16 | 0.034 | — | — | — | — |
| `betweenness` | diamond/4 weighted | 24 | 1 | 1 | 1.27e-16 | 0.076 | 1 | 1 | 1.27e-16 | 0.076 | — | — | — | — |
| `betweenness` | diamond/4 unweighted | 24 | 1 | 1 | 1.27e-16 | 0.076 | 1 | 1 | 1.27e-16 | 0.076 | — | — | — | — |
| `betweenness` | empty/1 weighted | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — |
| `betweenness` | empty/1 unweighted | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — |
| `betweenness` | er/int60 weighted | 360 | 3 | 3 | 4.08e-16 | 0.023 | 3 | 3 | 4.08e-16 | 0.023 | — | — | — | — |
| `betweenness` | er/int60 unweighted | 360 | 4 | 4 | 5.87e-16 | 0.039 | 4 | 4 | 5.87e-16 | 0.039 | — | — | — | — |
| `betweenness` | er/sym60 weighted | 120 | 2 | 2 | 2.98e-16 | 0.018 | 2 | 2 | 2.98e-16 | 0.018 | — | — | — | — |
| `betweenness` | er/sym60 unweighted | 120 | 2 | 2 | 3.39e-16 | 0.024 | 2 | 2 | 3.83e-16 | 0.027 | — | — | — | — |
| `betweenness` | grid2d/5x5 weighted | 150 | 4 | 4 | 4.61e-16 | 0.055 | 3 | 3 | 4.06e-16 | 0.048 | — | — | — | — |
| `betweenness` | grid2d/5x5 unweighted | 150 | 4 | 4 | 4.61e-16 | 0.055 | 3 | 3 | 4.06e-16 | 0.048 | — | — | — | — |
| `betweenness` | heavytail/120 weighted | 240 | 1 | 1 | 1.37e-16 | 0.0036 | 1 | 1 | 1.37e-16 | 0.0036 | — | — | — | — |
| `betweenness` | heavytail/120 unweighted | 240 | 5 | 5 | 7.75e-16 | 0.025 | 6 | 6 | 7.75e-16 | 0.025 | — | — | — | — |
| `betweenness` | isolated/5 weighted | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — |
| `betweenness` | isolated/5 unweighted | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — |
| `betweenness` | ladder/20 weighted | 80 | 4 | 4 | 4.93e-16 | 0.031 | 3 | 3 | 4.40e-16 | 0.028 | — | — | — | — |
| `betweenness` | ladder/20 unweighted | 80 | 4 | 4 | 4.93e-16 | 0.031 | 3 | 3 | 4.40e-16 | 0.028 | — | — | — | — |
| `betweenness` | path/10 weighted | 20 | 1 | 1 | 1.27e-16 | 0.029 | 1 | 1 | 1.27e-16 | 0.029 | — | — | — | — |
| `betweenness` | path/10 unweighted | 20 | 1 | 1 | 1.27e-16 | 0.029 | 1 | 1 | 1.27e-16 | 0.029 | — | — | — | — |
| `betweenness` | star/9 weighted | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — |
| `betweenness` | star/9 unweighted | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — |
| `laplacian` | dangling/50 unnormalized | 166 | 1 | 1 | 2.00e-16 | 0.87 | — | — | — | — | 1 | 1 | 2.08e-16 | 0.87 |
| `laplacian` | dangling/50 normed | 772 | 2 | 2 | 3.60e-16 | 0.3 | — | — | — | — | 2 | 2 | 3.60e-16 | 0.31 |
| `laplacian` | er/float100 unnormalized | 332 | 2 | 2 | 2.85e-16 | 0.78 | — | — | — | — | 2 | 2 | 2.93e-16 | 0.78 |
| `laplacian` | er/float100 normed | 2384 | 3 | 3 | 3.93e-16 | 0.31 | — | — | — | — | 3 | 3 | 3.51e-16 | 0.31 |
| `laplacian` | er/symfloat50 unnormalized | 372 | 1 | 1 | 1.85e-16 | 0.44 | — | — | — | — | 2 | 2 | 1.85e-16 | 0.44 |
| `laplacian` | er/symfloat50 normed | 666 | 2 | 2 | 3.44e-16 | 0.32 | — | — | — | — | 2 | 2 | 3.44e-16 | 0.32 |
| `laplacian` | negative/dag30 unnormalized | 90 | 0 | 0 | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 |
| `laplacian` | selfloop/10 unnormalized | 30 | 0 | 0 | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 |
| `laplacian` | selfloop/10 normed | 40 | 1 | 1 | 1.34e-16 | 0.17 | — | — | — | — | 1 | 1 | 1.34e-16 | 0.17 |
| `laplacian` | star/9 unnormalized | 43 | 0 | 0 | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 |
| `laplacian` | star/9 normed | 48 | 1 | 1 | 1.38e-16 | 0.12 | — | — | — | — | 1 | 1 | 1.38e-16 | 0.12 |
| `pagerank` | bool/30 | 1 | 990 | 990 | 4.93e-14 | 0.048 | 5.02e+03 | 5.02e+03 | 1.92e-13 | 0.19 | — | — | — | — |
| `pagerank` | bool/30 floor | 1 | 2 | 2 | 9.00e-17 | 0.005 | 5 | 5 | 2.09e-16 | 0.012 | — | — | — | — |
| `pagerank` | components/40 | 1 | 9.36e+03 | 9.36e+03 | 8.04e-13 | 0.79 | 5.23e+04 | 5.23e+04 | 4.49e-12 | **4.43 초과** | — | — | — | — |
| `pagerank` | components/40 floor | 1 | 2 | 2 | 1.15e-16 | 0.0076 | 53 | 53 | 4.50e-15 | 0.3 | — | — | — | — |
| `pagerank` | dangling/50 | 8 | 2.35e+03 | 2.35e+03 | 1.32e-13 | 0.13 | 1.2e+04 | 1.2e+04 | 6.19e-13 | 0.61 | — | — | — | — |
| `pagerank` | dangling/50 floor | 8 | 10 | 10 | 9.93e-16 | 0.017 | 15 | 15 | 1.21e-15 | 0.036 | — | — | — | — |
| `pagerank` | er/float100 | 8 | 4.19e+03 | 4.19e+03 | 1.74e-13 | 0.17 | 1.4e+04 | 1.4e+04 | 3.84e-13 | 0.37 | — | — | — | — |
| `pagerank` | er/float100 floor | 8 | 14 | 14 | 1.45e-15 | 0.013 | 16 | 16 | 1.25e-15 | 0.031 | — | — | — | — |
| `pagerank` | er/sym60 | 1 | 1.69e+03 | 1.69e+03 | 6.82e-14 | 0.067 | 1.3e+04 | 1.3e+04 | 4.31e-13 | 0.42 | — | — | — | — |
| `pagerank` | er/sym60 floor | 1 | 4 | 4 | 2.33e-16 | 0.012 | 10 | 10 | 4.43e-16 | 0.024 | — | — | — | — |
| `pagerank` | grid2d/6x6 | 1 | 679 | 679 | 5.33e-14 | 0.053 | 4.42e+03 | 4.42e+03 | 3.47e-13 | 0.34 | — | — | — | — |
| `pagerank` | grid2d/6x6 floor | 1 | 3 | 3 | 1.59e-16 | 0.014 | 7 | 7 | 3.62e-16 | 0.032 | — | — | — | — |
| `pagerank` | heavytail/120 | 1 | 4.49e+03 | 4.49e+03 | 5.30e-14 | 0.052 | 1.53e+04 | 1.53e+04 | 3.13e-13 | 0.3 | — | — | — | — |
| `pagerank` | heavytail/120 floor | 1 | 4 | 4 | 2.18e-16 | 0.0077 | 14 | 14 | 3.29e-16 | 0.012 | — | — | — | — |
| `pagerank` | isolated/5 | 1 | 0 | 0 | 5.55e-17 | 5.5e-5 | 0 | 0 | 5.55e-17 | 5.5e-5 | — | — | — | — |
| `pagerank` | isolated/5 floor | 1 | 0 | 0 | 5.55e-17 | 0.0052 | 0 | 0 | 5.55e-17 | 0.0052 | — | — | — | — |
| `pagerank` | path/10 | 1 | 1.58e+03 | 1.58e+03 | 1.24e-13 | 0.12 | 1.42e+04 | 1.42e+04 | 1.22e-12 | **1.2 초과** | — | — | — | — |
| `pagerank` | path/10 floor | 1 | 2 | 2 | 1.79e-16 | 0.016 | 8 | 8 | 7.91e-16 | 0.07 | — | — | — | — |
| `pagerank` | selfloop/10 | 1 | 2.83e+03 | 2.83e+03 | 2.54e-13 | 0.25 | 1.76e+04 | 1.76e+04 | 1.52e-12 | **1.5 초과** | — | — | — | — |
| `pagerank` | selfloop/10 floor | 1 | 3 | 3 | 2.50e-16 | 0.022 | 19 | 19 | 1.64e-15 | 0.14 | — | — | — | — |
| `pagerank` | star/9 | 1 | 2.44e+03 | 2.44e+03 | 2.70e-13 | 0.27 | 3.96e+03 | 3.96e+03 | 4.40e-13 | 0.43 | — | — | — | — |
| `pagerank` | star/9 floor | 1 | 2 | 2 | 2.20e-16 | 0.015 | 5 | 5 | 4.36e-16 | 0.029 | — | — | — | — |
| `paths` | absorb/20 directed | 81 | 0 | 0 | 1.00e-17 | 0.09 | — | — | — | — | — | — | — | — |
| `paths` | absorb/20 directed FW | 81 | 0 | 0 | 1.00e-17 | 0.09 | — | — | — | — | — | — | — | — |
| `paths` | bigint/over40 directed | 312 | 2 | 2 | 2.30e-16 | 0.47 | — | — | — | — | — | — | — | — |
| `paths` | bigint/over40 directed FW | 312 | 2 | 2 | 2.30e-16 | 0.47 | — | — | — | — | — | — | — | — |
| `paths` | dangling/50 directed | 384 | 2 | 2 | 2.09e-16 | 0.97 | — | — | — | — | — | — | — | — |
| `paths` | dangling/50 directed FW | 384 | 1 | 1 | 1.90e-16 | 0.97 | — | — | — | — | — | — | — | — |
| `paths` | dense/special6 directed | 30 | 0 | 0 | 8.27e-17 | 0.75 | — | — | — | — | — | — | — | — |
| `paths` | dense/special6 directed FW | 30 | 0 | 0 | 8.27e-17 | 0.75 | — | — | — | — | — | — | — | — |
| `paths` | er/float100 directed | 776 | 1 | 1 | 2.16e-16 | 0.95 | — | — | — | — | — | — | — | — |
| `paths` | er/float100 directed FW | 776 | 1 | 1 | 1.69e-16 | 0.95 | — | — | — | — | — | — | — | — |
| `paths` | er/symfloat50 directed | 384 | 1 | 1 | 2.62e-16 | 0.99 | — | — | — | — | — | — | — | — |
| `paths` | er/symfloat50 directed FW | 384 | 1 | 1 | 1.56e-16 | 0.99 | — | — | — | — | — | — | — | — |
| `paths` | er/wide80 directed | 632 | 2 | 2 | 3.70e-16 | 0.85 | — | — | — | — | — | — | — | — |
| `paths` | er/wide80 directed FW | 632 | 2 | 2 | 2.98e-16 | 0.85 | — | — | — | — | — | — | — | — |
| `paths` | heavytail/120 directed | 864 | 1 | 1 | 2.01e-16 | 0.99 | — | — | — | — | — | — | — | — |
| `paths` | heavytail/120 directed FW | 864 | 1 | 1 | 2.13e-16 | 0.99 | — | — | — | — | — | — | — | — |
| `paths` | oneway/12 directed | 34 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | oneway/12 directed FW | 34 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | twoway/12 directed | 68 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | twoway/12 directed FW | 68 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | zero/15 directed | 109 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | zero/15 directed FW | 109 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | absorb/20 undirected | 133 | 0 | 0 | 1.00e-17 | 0.09 | — | — | — | — | — | — | — | — |
| `paths` | absorb/20 undirected FW | 133 | 0 | 0 | 1.00e-17 | 0.09 | — | — | — | — | — | — | — | — |
| `paths` | bigint/over40 undirected | 312 | 0 | 0 | 1.09e-16 | 0.49 | — | — | — | — | — | — | — | — |
| `paths` | bigint/over40 undirected FW | 312 | 0 | 0 | 1.09e-16 | 0.49 | — | — | — | — | — | — | — | — |
| `paths` | dangling/50 undirected | 384 | 1 | 1 | 1.06e-16 | 0.86 | — | — | — | — | — | — | — | — |
| `paths` | dangling/50 undirected FW | 384 | 1 | 1 | 1.25e-16 | 0.86 | — | — | — | — | — | — | — | — |
| `paths` | dense/special6 undirected | 30 | 0 | 0 | 8.27e-17 | 0.75 | — | — | — | — | — | — | — | — |
| `paths` | dense/special6 undirected FW | 30 | 0 | 0 | 8.27e-17 | 0.75 | — | — | — | — | — | — | — | — |
| `paths` | er/float100 undirected | 792 | 1 | 1 | 1.50e-16 | 0.58 | — | — | — | — | — | — | — | — |
| `paths` | er/float100 undirected FW | 792 | 1 | 1 | 1.81e-16 | 0.58 | — | — | — | — | — | — | — | — |
| `paths` | er/symfloat50 undirected | 384 | 1 | 1 | 2.62e-16 | 0.99 | — | — | — | — | — | — | — | — |
| `paths` | er/symfloat50 undirected FW | 384 | 1 | 1 | 1.56e-16 | 0.99 | — | — | — | — | — | — | — | — |
| `paths` | er/wide80 undirected | 632 | 2 | 2 | 3.15e-16 | 0.99 | — | — | — | — | — | — | — | — |
| `paths` | er/wide80 undirected FW | 632 | 2 | 2 | 2.40e-16 | 0.99 | — | — | — | — | — | — | — | — |
| `paths` | heavytail/120 undirected | 952 | 1 | 1 | 1.70e-16 | 0.57 | — | — | — | — | — | — | — | — |
| `paths` | heavytail/120 undirected FW | 952 | 1 | 1 | 1.70e-16 | 0.57 | — | — | — | — | — | — | — | — |
| `paths` | oneway/12 undirected | 68 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | oneway/12 undirected FW | 68 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | twoway/12 undirected | 68 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | twoway/12 undirected FW | 68 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | zero/15 undirected | 210 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| `paths` | zero/15 undirected FW | 210 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
