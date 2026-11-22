# sd.interpolate 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/interpolate.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/interpolate/ref의 정확한 유리수 참값이다(tools/interpolate_ref.py; 시험 데이터는 tests/interpolate/samples.py). linear 묶음(IP1)은 linear.npz의 정확한 a + t(b − a)를 한 번 반올림한 값과 sd.interp를 견준다: curves 영역은 실수 곡선마다 [x[0], x[-1]] 안의 평가점(곡선 전체로 부른다), intervals 영역은 선형 커널의 구간(끝점 둘레의 이웃 double, 반대 부호, 거의 같은 값, ±0, 비정상 수와 넘치는 폭)의 안쪽 점(두 노드로 부른다)이다. 바깥은 sd.interp가 numpy처럼 끝값으로 자르므로 재지 않는다. 계약은 tests/interpolate/refdata.py의 linear_bound다(docs/design/interpolate.md §4: γ₆(|a| + |b|) + 3η). 비교 패키지는 numpy 2.5.4의 interp, scipy 1.18.1의 make_interp_spline(k=1), R 4.6.2의 approx(rule = 2)이고 같은 점에서 그 자리에서 잰다. 비교 패키지가 유한하지 않은 값을 낸 점(numpy의 기울기가 비정상 수 간격에서 넘치는 곳 등)은 excluded로 센다.

- ULP: sd의 값과 참값을 double로 반올림한 값 사이에 있는 double의 수(ulp_distance; 0이면 올바르게 반올림한 값, 1이면 그 이웃). 99.9%는 그 분위수다.
- 상대오차: |sd − 참값|/|참값|.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.

측정: 20261123-040451-a741a97+ (2026-11-23T04:04:51, a741a97 dirty).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | 최대 상대오차 | 계약 대비 |
|---|---|---:|---:|---:|---:|---:|
| `linear` | curves | 1736 | 4.52e+13 | 59 | 0.00516 | 0.23 |
| `linear` | intervals | 1072 | 5.62e+6 | 4.92e+6 | 7.20e-4 | 0.5 |
