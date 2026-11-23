# sd.smooth 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/smooth.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/smooth/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다.

- 상대오차: |sd − 참값|/|참값|.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.

측정: 20261123-193036-734df2d+ (2026-11-23T19:30:36, 734df2d dirty).

| 함수 | 영역 | 점 | 최대 상대오차 | 계약 대비 |
|---|---|---:|---:|---:|
| `bandwidth` | closed | 29 | 2.23e-16 | 0.25 |
| `bandwidth` | cv | 11 | 1.25e-13 | 1.2e-5 |
| `bandwidth` | isj | 6 | 1.67e-15 | 1.7e-7 |
| `bandwidth` | sj | 12 | 4.38e-15 | 4.4e-7 |
| `fitpack` | condition | 34 | 1 | 0.82 |
| `kde` | binned/1d | 14 | 0.00311 | 0.21 |
| `kde` | binned/nd | 2 | 11.5 | 0.064 |
| `kde` | cdf/1d | 80 | 9.94e-15 | 0.13 |
| `kde` | cdf/1d/tail | 18 | 3.87e-15 | 0.15 |
| `kde` | kernels | 490 | 6.71e-16 | 0.22 |
| `kde` | logpdf/1d | 80 | 6.34e-16 | 0.31 |
| `kde` | logpdf/1d/tail | 18 | 1.61e-16 | 0.086 |
| `kde` | logpdf/nd | 11 | 5.25e-14 | 0.19 |
| `kde` | logpdf/nd/tail | 9 | 2.07e-13 | 0.75 |
| `kde` | pdf/1d | 80 | 5.10e-16 | 0.31 |
| `kde` | pdf/1d/tail | 18 | 1.89e-16 | 0.0085 |
| `kde` | pdf/nd | 11 | 6.77e-14 | 0.34 |
| `kde` | pdf/nd/tail | 9 | 0 | 0 |
| `kde` | reflect | 14 | 4.13e-16 | 0.2 |
| `kreg` | ll/1d | 168 | 1.17e-15 | 0.18 |
| `kreg` | ll/1d/far | 12 | 1.85e-16 | 9.6e-16 |
| `kreg` | ll/nd | 48 | 1.39e-15 | 0.52 |
| `kreg` | ll/nd/far | 6 | 4.30e-16 | 2.1e-15 |
| `kreg` | nw/1d | 168 | 6.03e-16 | 0.33 |
| `kreg` | nw/1d/far | 12 | 1.42e-16 | 0.051 |
| `kreg` | nw/nd | 48 | 3.95e-16 | 0.18 |
| `kreg` | nw/nd/far | 6 | 2.02e-16 | 0.11 |
| `local` | fit/plain | 1470 | 3.22e-15 | 0.75 |
| `local` | fit/spread | 536 | 1.36e-15 | 0.41 |
| `local` | fit/ties | 186 | 5.70e-16 | 0.18 |
| `local` | fit/weights | 90 | 5.80e-16 | 0.18 |
| `local` | leverage | 2282 | 2.62e-15 | 0.15 |
| `local` | predict/plain | 420 | 3.63e-15 | 0.75 |
| `local` | predict/spread | 293 | 8.74e-8 | 0.41 |
| `local` | predict/ties | 136 | 5.59e-15 | 0.24 |
| `local` | predict/weights | 42 | 2.16e-15 | 0.16 |
| `loess` | beyond | 162 | 0.0762 | 4.0e-11 |
| `loess` | fit/local/plain | 2450 | 2.45e-15 | 0.4 |
| `loess` | fit/local/spread | 850 | 6.07e-8 | 0.3 |
| `loess` | fit/local/ties | 328 | 7.96e-16 | 0.24 |
| `loess` | fit/local/weights | 150 | 4.25e-16 | 0.16 |
| `loess` | fit/local_nd/nd | 887 | 8.37e-16 | 0.26 |
| `loess` | leverage/local | 3778 | 1.00e-7 | 0.11 |
| `loess` | leverage/local_nd | 887 | 1.57e-14 | 0.1 |
| `loess` | predict/local/plain | 700 | 3.11e-15 | 0.29 |
| `loess` | predict/local/spread | 475 | 3.21e-7 | 0.2 |
| `loess` | predict/local/ties | 275 | 2.42e-15 | 0.2 |
| `loess` | predict/local/weights | 70 | 8.95e-16 | 0.15 |
| `loess` | predict/local_nd/nd | 120 | 8.71e-16 | 0.18 |
| `savgol` | coeffs | 3021 | 9.84e-17 | 0.11 |
| `savgol` | coeffs_high | 1371 | 1.13e-31 | 1.3e-16 |
| `savgol` | filter_interp | 971 |  | 0.12 |
| `savgol` | filter_mirror | 971 |  | 0.12 |
| `savgol` | filter_nearest | 971 |  | 0.12 |
| `savgol` | filter_constant | 971 |  | 0.17 |
| `savgol` | filter_wrap | 971 |  | 0.14 |
| `spline` | coef/lsq/plain | 98 | 1.11e-15 | 0.27 |
| `spline` | coef/lsq/spread | 59 | 3.90e-10 | 0.25 |
| `spline` | coef/lsq/ties | 13 | 4.80e-16 | 0.11 |
| `spline` | coef/lsq/weights | 10 | 7.88e-16 | 0.055 |
| `spline` | coef/ps/plain | 303 | 2.86e-15 | 0.13 |
| `spline` | coef/ps/spread | 186 | 2.21e-15 | 0.13 |
| `spline` | coef/ps/ties | 75 | 1.38e-15 | 0.14 |
| `spline` | coef/ps/weights | 33 | 2.10e-15 | 0.056 |
| `spline` | coef/ss/plain | 1232 | 1.20e-10 | 0.22 |
| `spline` | coef/ss/spread | 602 | 4.06e-6 | 0.045 |
| `spline` | coef/ss/ties | 168 | 1.75e-14 | 0.2 |
| `spline` | coef/ss/weights | 128 | 6.99e-14 | 0.0017 |
| `spline` | leverage/ps | 1638 | 1.10e-15 | 0.067 |
| `spline` | leverage/ss | 2064 | 2.00e-6 | 0.28 |
| `spline` | trace/ps | 57 | 1.01e-14 | 0.015 |
| `spline` | trace/ss | 73 | 1.39e-5 | 0.044 |
| `spline` | refused/ss | 1 |  |  |
