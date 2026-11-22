# sd.graph 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/graph.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/graph/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다.

- ULP: sd의 값과 참값을 double로 반올림한 값 사이에 있는 double의 수(ulp_distance; 0이면 올바르게 반올림한 값, 1이면 그 이웃). 99.9%는 그 분위수다.
- 상대오차: |sd − 참값|/|참값|.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.

측정: 20261123-045535-5d6bfc9+ (2026-11-23T04:55:35, 5d6bfc9 dirty).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | 최대 상대오차 | 계약 대비 |
|---|---|---:|---:|---:|---:|---:|
| `paths` | absorb/20 directed | 81 | 0 | 0 | 1.00e-17 | 0.09 |
| `paths` | absorb/20 directed FW | 81 | 0 | 0 | 1.00e-17 | 0.09 |
| `paths` | bigint/over40 directed | 312 | 2 | 2 | 2.30e-16 | 0.47 |
| `paths` | bigint/over40 directed FW | 312 | 2 | 2 | 2.30e-16 | 0.47 |
| `paths` | dangling/50 directed | 384 | 2 | 2 | 2.09e-16 | 0.97 |
| `paths` | dangling/50 directed FW | 384 | 1 | 1 | 1.90e-16 | 0.97 |
| `paths` | dense/special6 directed | 30 | 0 | 0 | 8.27e-17 | 0.75 |
| `paths` | dense/special6 directed FW | 30 | 0 | 0 | 8.27e-17 | 0.75 |
| `paths` | er/float100 directed | 776 | 1 | 1 | 2.16e-16 | 0.95 |
| `paths` | er/float100 directed FW | 776 | 1 | 1 | 1.69e-16 | 0.95 |
| `paths` | er/symfloat50 directed | 384 | 1 | 1 | 2.62e-16 | 0.99 |
| `paths` | er/symfloat50 directed FW | 384 | 1 | 1 | 1.56e-16 | 0.99 |
| `paths` | er/wide80 directed | 632 | 2 | 2 | 3.70e-16 | 0.85 |
| `paths` | er/wide80 directed FW | 632 | 2 | 2 | 2.98e-16 | 0.85 |
| `paths` | heavytail/120 directed | 864 | 1 | 1 | 2.01e-16 | 0.99 |
| `paths` | heavytail/120 directed FW | 864 | 1 | 1 | 2.13e-16 | 0.99 |
| `paths` | oneway/12 directed | 34 | 0 | 0 | 0 | 0 |
| `paths` | oneway/12 directed FW | 34 | 0 | 0 | 0 | 0 |
| `paths` | twoway/12 directed | 68 | 0 | 0 | 0 | 0 |
| `paths` | twoway/12 directed FW | 68 | 0 | 0 | 0 | 0 |
| `paths` | zero/15 directed | 109 | 0 | 0 | 0 | 0 |
| `paths` | zero/15 directed FW | 109 | 0 | 0 | 0 | 0 |
| `paths` | absorb/20 undirected | 133 | 0 | 0 | 1.00e-17 | 0.09 |
| `paths` | absorb/20 undirected FW | 133 | 0 | 0 | 1.00e-17 | 0.09 |
| `paths` | bigint/over40 undirected | 312 | 0 | 0 | 1.09e-16 | 0.49 |
| `paths` | bigint/over40 undirected FW | 312 | 0 | 0 | 1.09e-16 | 0.49 |
| `paths` | dangling/50 undirected | 384 | 1 | 1 | 1.06e-16 | 0.86 |
| `paths` | dangling/50 undirected FW | 384 | 1 | 1 | 1.25e-16 | 0.86 |
| `paths` | dense/special6 undirected | 30 | 0 | 0 | 8.27e-17 | 0.75 |
| `paths` | dense/special6 undirected FW | 30 | 0 | 0 | 8.27e-17 | 0.75 |
| `paths` | er/float100 undirected | 792 | 1 | 1 | 1.50e-16 | 0.58 |
| `paths` | er/float100 undirected FW | 792 | 1 | 1 | 1.81e-16 | 0.58 |
| `paths` | er/symfloat50 undirected | 384 | 1 | 1 | 2.62e-16 | 0.99 |
| `paths` | er/symfloat50 undirected FW | 384 | 1 | 1 | 1.56e-16 | 0.99 |
| `paths` | er/wide80 undirected | 632 | 2 | 2 | 3.15e-16 | 0.99 |
| `paths` | er/wide80 undirected FW | 632 | 2 | 2 | 2.40e-16 | 0.99 |
| `paths` | heavytail/120 undirected | 952 | 1 | 1 | 1.70e-16 | 0.57 |
| `paths` | heavytail/120 undirected FW | 952 | 1 | 1 | 1.70e-16 | 0.57 |
| `paths` | oneway/12 undirected | 68 | 0 | 0 | 0 | 0 |
| `paths` | oneway/12 undirected FW | 68 | 0 | 0 | 0 | 0 |
| `paths` | twoway/12 undirected | 68 | 0 | 0 | 0 | 0 |
| `paths` | twoway/12 undirected FW | 68 | 0 | 0 | 0 | 0 |
| `paths` | zero/15 undirected | 210 | 0 | 0 | 0 | 0 |
| `paths` | zero/15 undirected FW | 210 | 0 | 0 | 0 | 0 |
