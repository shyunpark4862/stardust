# sd.integrate 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/integrate.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/integrate/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다.

- 상대오차: |sd − 참값|/|참값|.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.
- 비교 패키지(scipy 1.18.1): 같은 참값, 같은 점에서 같은 척도로 잰 그 라이브러리의 오차다. 같은 양을 계산하는 공개 함수가 없으면 —, R이 없는 기계에서는 not run, 함수가 입력 전체를 거부하면 에러, 시간(120초)·메모리(2GB) 한도를 넘으면 멈춤, R의 glm이 기본 설정으로 시작값을 찾지 못하면 시작값 필요다. 제외: 참값은 유한한데 그 라이브러리가 NaN·inf를 낸 점의 수(정의가 다르거나 실패한 점; 다른 열에서 빠진다). compare는 이 열들의 변화를 알리기만 하고 종료 코드에 넣지 않는다.

측정: 20261122-021425-cda05f3+ (2026-11-22T02:14:25, cda05f3 dirty), 20261122-033710-9aacd38+ (2026-11-22T03:37:10, 9aacd38 dirty), 20261122-034527-9aacd38+ (2026-11-22T03:45:27, 9aacd38 dirty), 20261122-043713-f8ebce7+ (2026-11-22T04:37:13, f8ebce7 dirty).

| 함수 | 영역 | 점 | 최대 상대오차 | 계약 대비 | scipy 최대 상대오차 | scipy 계약 대비 |
|---|---|---:|---:|---:|---:|---:|
| `cubature` | gk21 | 106 | inf | 0.96 | inf | 0.96 |
| `cubature` | batch gk21 | 62 | 5.04e-8 | 0.64 | 5.04e-8 | 0.64 |
| `cubature` | gk15 | 106 | 0.428 | 0.96 | 0.428 | 0.96 |
| `cubature` | batch gk15 | 62 | 5.03e-8 | 0.62 | 5.03e-8 | 0.62 |
| `cubature` | multiple gk21 | 4 | 1.74e-16 | 0.093 | 1.74e-16 | 0.093 |
| `cubature` | multiple gk15 | 4 | 3.90e-17 | 1.2e-5 | 3.90e-17 | 1.2e-5 |
| `cubature` | multiple genz-malik | 4 | 2.02e-11 | 0.082 | 2.02e-11 | 0.082 |
| `fixed_quad` | n=5 | 30 | 299 |  | 299 |  |
| `fixed_quad` | n=20 | 30 | 461 |  | 461 |  |
| `fixed_quad` | n=100 | 30 | 7.93 |  | 7.93 |  |
| `multiple` | dblquad, tplquad | 7 | 5.13e-15 | 0.0067 | 5.13e-15 | 0.0067 |
| `nsum` | series | 9 | 3.00e-12 | 0.89 | 3.00e-12 | 0.89 |
| `quad` | book | 62 | 1.98 | 0.7 | 1.98 | 0.7 |
| `quad` | quadpack | 3 | 8.99e-14 | 0.074 | 8.99e-14 | 0.074 |
| `quad` | scipy | 28 | 0.431 | 0.39 | 0.431 | 0.39 |
| `quad` | hard | 13 | 1.30e-13 | 0.24 | 1.30e-13 | 0.24 |
| `quad_vec` | collection | 106 | inf | 0.95 | inf | 0.95 |
| `quad_vec` | families, norm 2 | 3 | 1.91e-9 | 0.84 | 1.91e-9 | 0.84 |
| `quad_vec` | families, norm max | 3 | 1.88e-9 | 0.84 | 1.88e-9 | 0.84 |
| `quad_vec` | vector | 2 | 1.74e-16 | 0.0047 | 1.74e-16 | 0.0047 |
| `quad_weights` | qawo | 20 | 4.42e-15 | 0.46 | 4.42e-15 | 0.46 |
| `quad_weights` | qawf | 20 | 1.18e-9 | 0.0042 | 1.72 | **1.08e+10 초과** |
| `quad_weights` | qaws | 43 | 1.34e-8 | 0.22 | 1.34e-8 | 0.22 |
| `quad_weights` | qawc | 26 | 6.47e-14 | 0.24 | 6.47e-14 | 0.24 |
| `samples` | trapezoid | 12 | 1.43e-16 | 0.023 | 9.82e-17 | 0.018 |
| `samples` | simpson | 12 | 3.34e-16 | 0.05 | 1.35e-16 | 0.02 |
| `samples` | romb | 6 | 1.73e-16 | 0.024 | 1.26e-16 | 0.022 |
| `samples` | cumulative_trapezoid | 12 | 1.75e-15 | 0.047 | 1.75e-15 | 0.047 |
| `samples` | cumulative_simpson | 12 | 3.99e-15 | 0.044 | 3.99e-15 | 0.044 |
| `tanhsinh` | book | 62 | 5.82 | 0.78 | 5.82 | 0.78 |
| `tanhsinh` | quadpack | 3 | 9.07e-6 | 0.5 | 9.07e-6 | 0.5 |
| `tanhsinh` | scipy | 27 | 1.9 | 0.72 | 1.9 | 0.72 |
| `tanhsinh` | hard | 9 | 0.00991 | 0.36 | 0.00991 | 0.36 |
