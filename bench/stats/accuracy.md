# sd.stats 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/stats.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/stats/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다.

- ULP: |sd − 참값|을 참값에서의 float64 간격으로 나눈 값. 99.9%는 그 분위수다.
- ULP/배율: ULP를 그 점의 계약 배율로 나눈 값. 배율은 상대 한계 1e-14에 곱하는 수로, 역함수는 max(1, κ)(κ는 입력에 대한 상대 조건수), Bessel·Airy와 복소 오차함수·wofz는 영점 근처에서 max(1, M/|f|)(M은 크기의 포락선)다. 배율이 1보다 큰 점이 있는 함수에만 있다. 계약 대비처럼 배율을 나눈 값이라 둘이 함께 움직인다(상대 계약이면 ULP/배율 약 45–90이 계약 대비 1이다). 최대 ULP 열은 배율을 나누지 않은 값이라, 배율이 큰 점(역함수가 0을 지나는 곳, 진동 함수의 영점 근처)에서는 계약 안이어도 크다.
- mixed: |sd − 참값|/max(1, |참값|)를 2⁻⁵² 단위로 잰 값(45가 1e-14다). 0을 지나는 값(log 밀도, 엔트로피, gammaln 등)의 계약 척도이고, 그 함수에만 있다. 최대와 99.9%다.
- 상대오차: |sd − 참값|/|참값|.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.
- 영역: guaranteed는 보장 영역, loose와 mixed는 그 밖(mixed는 그 척도로 잰다).

측정: 20261120-151558-deeafe2 (2026-11-20T15:15:58, deeafe2 dirty).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | ULP/배율 | mixed (최대 / 99.9%) | 최대 상대오차 | 계약 대비 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bernoulli_ccdf` | guaranteed | 6000 | 0 | 0 |  |  | 0 | 0 |
| `bernoulli_cdf` | guaranteed | 6000 | 0.5 | 0.421 |  |  | 9.55e-17 | 0.0095 |
| `bernoulli_entropy` | guaranteed | 2500 | 0.5 | 0.499 |  | 0.25 / 0.249 | 1.09e-16 | 0.0055 |
| `bernoulli_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `bernoulli_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `bernoulli_ilogccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `bernoulli_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `bernoulli_kurtosis` | guaranteed | 2500 | 1.63 | 1.49 |  |  | 2.30e-16 | 0.023 |
| `bernoulli_logccdf` | guaranteed | 6000 | 0.5 | 0.5 |  |  | 1.11e-16 | 0.011 |
| `bernoulli_logcdf` | guaranteed | 6000 | 0.504 | 0.499 |  |  | 1.10e-16 | 0.011 |
| `bernoulli_logpmf` | guaranteed | 6000 | 0.504 | 0.5 |  |  | 1.11e-16 | 0.011 |
| `bernoulli_mean` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `bernoulli_median` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `bernoulli_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `bernoulli_pmf` | guaranteed | 6000 | 0.461 | 0.349 |  |  | 5.12e-17 | 0.0051 |
| `bernoulli_skewness` | guaranteed | 2500 | 1.75 | 1.42 |  |  | 2.03e-16 | 0.02 |
| `bernoulli_standard_deviation` | guaranteed | 2500 | 0.747 | 0.74 |  |  | 1.54e-16 | 0.015 |
| `bernoulli_variance` | guaranteed | 2500 | 0.712 | 0.604 |  |  | 1.11e-16 | 0.011 |
| `beta_ccdf` | guaranteed | 6000 | 13.2 | 8.44 |  |  | 1.55e-15 | 0.16 |
| `beta_cdf` | guaranteed | 6000 | 10.1 | 7.05 |  |  | 1.59e-15 | 0.16 |
| `beta_entropy` | guaranteed | 2500 | 1.96e+306 | 1.96e+306 |  | 33.3 / 27.1 | inf | 0.74 |
| `beta_iccdf` | guaranteed | 4000 | 555 | 243 | 8.29 |  | 7.59e-14 | 0.1 |
| `beta_icdf` | guaranteed | 4000 | 233 | 123 | 18.5 |  | 4.37e-13 | 0.36 |
| `beta_ilogccdf` | guaranteed | 4000 | 119 | 33.8 | 41.4 |  | 2.63e-14 | 0.51 |
| `beta_ilogcdf` | guaranteed | 4000 | 77.6 | 19 | 5.63 |  | 8.67e-15 | 0.063 |
| `beta_kurtosis` | guaranteed | 2500 | 0.5 | 0.499 |  |  | 1.08e-16 | 0.011 |
| `beta_logccdf` | guaranteed | 6000 | 11 | 8.48 |  |  | 1.62e-15 | 0.16 |
| `beta_logcdf` | guaranteed | 6000 | 12.8 | 8.66 |  |  | 1.69e-15 | 0.17 |
| `beta_logpdf` | guaranteed | 6000 | 1.60e+305 | 1.60e+305 |  | 1.96 / 1.13 | inf | 0.044 |
| `beta_mean` | guaranteed | 2500 | 1.4 | 1.26 |  |  | 1.96e-16 | 0.02 |
| `beta_median` | guaranteed | 2500 | 603 | 224 | 9.53 |  | 8.21e-14 | 0.14 |
| `beta_mode` | guaranteed | 2500 | 8.42 | 1.58 |  |  | 1.04e-15 | 0.1 |
| `beta_pdf` | guaranteed | 6000 | 6.08 | 5.36 |  |  | 1.13e-15 | 0.11 |
| `beta_skewness` | guaranteed | 2500 | 3.55 | 3 |  |  | 4.38e-16 | 0.044 |
| `beta_standard_deviation` | guaranteed | 2500 | 2.14 | 1.92 |  |  | 2.89e-16 | 0.029 |
| `beta_variance` | guaranteed | 2500 | 3.54 | 3.4 |  |  | 4.85e-16 | 0.048 |
| `betabinomial_ccdf` | guaranteed | 6000 | 4.51 | 3.23 |  |  | 6.29e-16 | 0.063 |
| `betabinomial_cdf` | guaranteed | 6000 | 4.46 | 3.2 |  |  | 6.08e-16 | 0.061 |
| `betabinomial_entropy` | guaranteed | 403 | 1.82 | 1.82 |  | 0.982 / 0.982 | 2.93e-16 | 0.022 |
| `betabinomial_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `betabinomial_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `betabinomial_ilogccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `betabinomial_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `betabinomial_kurtosis` | guaranteed | 403 | 5 | 5 |  |  | 5.64e-16 | 0.056 |
| `betabinomial_logccdf` | guaranteed | 6000 | 4.68 | 3.9 |  |  | 6.54e-16 | 0.065 |
| `betabinomial_logcdf` | guaranteed | 6000 | 4.98 | 3.69 |  |  | 7.25e-16 | 0.073 |
| `betabinomial_logpmf` | guaranteed | 6000 | 4.68 | 2.1 |  |  | 7.66e-16 | 0.077 |
| `betabinomial_mean` | guaranteed | 403 | 1.8 | 1.8 |  |  | 2.15e-16 | 0.022 |
| `betabinomial_median` | guaranteed | 403 | 0 | 0 |  |  | 0 | 0 |
| `betabinomial_mode` | guaranteed | 403 | 0 | 0 |  |  | 0 | 0 |
| `betabinomial_pmf` | guaranteed | 6000 | 1.59 | 1.53 |  |  | 2.62e-16 | 0.026 |
| `betabinomial_skewness` | guaranteed | 403 | 3.26 | 3.26 |  |  | 4.66e-16 | 0.047 |
| `betabinomial_standard_deviation` | guaranteed | 403 | 2.05 | 2.05 |  |  | 3.53e-16 | 0.035 |
| `betabinomial_variance` | guaranteed | 403 | 3.23 | 3.23 |  |  | 5.50e-16 | 0.055 |
| `binomial_ccdf` | guaranteed | 6000 | 15.6 | 10.9 |  |  | 2.12e-15 | 0.21 |
| `binomial_cdf` | guaranteed | 6000 | 17.4 | 10.4 |  |  | 2.96e-15 | 0.3 |
| `binomial_entropy` | guaranteed | 806 | 2.44 | 2.44 |  | 0.809 / 0.809 | 3.82e-16 | 0.018 |
| `binomial_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `binomial_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `binomial_ilogccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `binomial_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `binomial_kurtosis` | guaranteed | 804 | 1.88 | 1.88 |  |  | 2.79e-16 | 0.028 |
| `binomial_logccdf` | guaranteed | 6000 | 17.4 | 10.4 |  |  | 2.96e-15 | 0.3 |
| `binomial_logcdf` | guaranteed | 6000 | 14.5 | 11.1 |  |  | 2.22e-15 | 0.22 |
| `binomial_logpmf` | guaranteed | 6000 | 4.12 | 2.84 |  |  | 4.67e-16 | 0.047 |
| `binomial_mean` | guaranteed | 806 | 0.5 | 0.5 |  |  | 1.09e-16 | 0.011 |
| `binomial_median` | guaranteed | 806 | 0 | 0 |  |  | 0 | 0 |
| `binomial_mode` | guaranteed | 806 | 0 | 0 |  |  | 0 | 0 |
| `binomial_pmf` | guaranteed | 6000 | 4.82 | 4.14 |  |  | 3.79e-14 | 0.065 |
| `binomial_skewness` | guaranteed | 804 | 1.76 | 1.76 |  |  | 2.09e-16 | 0.021 |
| `binomial_standard_deviation` | guaranteed | 806 | 1.04 | 1.04 |  |  | 1.74e-16 | 0.017 |
| `binomial_variance` | guaranteed | 806 | 1.16 | 1.16 |  |  | 2.15e-16 | 0.022 |
| `cauchy_ccdf` | guaranteed | 6000 | 2.12 | 1.88 |  |  | 3.35e-16 | 0.034 |
| `cauchy_cdf` | guaranteed | 6000 | 2.5 | 2.09 |  |  | 3.52e-16 | 0.035 |
| `cauchy_entropy` | guaranteed | 2500 | 0.5 | 0.499 |  | 0.49 / 0.481 | 1.09e-16 | 0.011 |
| `cauchy_iccdf` | guaranteed | 4000 | 30.4 | 16.8 | 1.98 |  | 4.63e-15 | 0.025 |
| `cauchy_icdf` | guaranteed | 4000 | 119 | 10.4 | 1.98 |  | 1.65e-14 | 0.025 |
| `cauchy_ilogccdf` | guaranteed | 4000 | 24.7 | 2.23 | 2.2 |  | 4.61e-15 | 0.028 |
| `cauchy_ilogcdf` | guaranteed | 4000 | 325 | 4.39 | 2.52 |  | 5.67e-14 | 0.03 |
| `cauchy_kurtosis` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `cauchy_logccdf` | guaranteed | 6000 | 2.73 | 2.38 |  |  | 4.37e-16 | 0.044 |
| `cauchy_logcdf` | guaranteed | 6000 | 2.35 | 2.21 |  |  | 4.07e-16 | 0.041 |
| `cauchy_logpdf` | guaranteed | 6000 | 127 | 36.2 |  | 2.04 / 1.52 | 1.73e-14 | 0.045 |
| `cauchy_mean` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `cauchy_median` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `cauchy_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `cauchy_pdf` | guaranteed | 6000 | 3.03 | 2.57 |  |  | 4.28e-16 | 0.043 |
| `cauchy_skewness` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `cauchy_standard_deviation` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `cauchy_variance` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `chisquared_ccdf` | guaranteed | 6000 | 6.55 | 5.83 |  |  | 1.22e-15 | 0.12 |
| `chisquared_cdf` | guaranteed | 6000 | 8.22 | 5.97 |  |  | 1.05e-15 | 0.1 |
| `chisquared_entropy` | guaranteed | 2500 | 241 | 109 |  | 7.08 / 6.59 | 4.03e-14 | 0.16 |
| `chisquared_iccdf` | guaranteed | 4000 | 382 | 228 | 10.6 |  | 5.02e-14 | 0.13 |
| `chisquared_icdf` | guaranteed | 4000 | 544 | 318 | 4.53 |  | 6.42e-14 | 0.068 |
| `chisquared_ilogccdf` | guaranteed | 4000 | 66.3 | 19.9 | 4.81 |  | 1.04e-14 | 0.082 |
| `chisquared_ilogcdf` | guaranteed | 4000 | 16.3 | 10 | 1.8 |  | 3.20e-15 | 0.022 |
| `chisquared_kurtosis` | guaranteed | 2500 | 0.5 | 0.499 |  |  | 1.09e-16 | 0.011 |
| `chisquared_logccdf` | guaranteed | 6000 | 8.29 | 6.55 |  |  | 1.14e-15 | 0.11 |
| `chisquared_logcdf` | guaranteed | 6000 | 9.16 | 7.13 |  |  | 1.49e-15 | 0.15 |
| `chisquared_logpdf` | guaranteed | 6000 | 15.1 | 3.56 |  | 1.89 / 0.889 | 1.80e-15 | 0.042 |
| `chisquared_mean` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `chisquared_median` | guaranteed | 2500 | 119 | 103 | 5.3 |  | 1.90e-14 | 0.08 |
| `chisquared_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `chisquared_pdf` | guaranteed | 6000 | 5.7 | 4.72 |  |  | 6.95e-16 | 0.069 |
| `chisquared_skewness` | guaranteed | 2500 | 0.78 | 0.756 |  |  | 1.54e-16 | 0.015 |
| `chisquared_standard_deviation` | guaranteed | 2500 | 0.5 | 0.499 |  |  | 1.10e-16 | 0.011 |
| `chisquared_variance` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_ccdf` | guaranteed | 6000 | 0.5 | 0.499 |  |  | 1.09e-16 | 0.011 |
| `discreteuniform_cdf` | guaranteed | 6000 | 0.5 | 0.499 |  |  | 1.10e-16 | 0.011 |
| `discreteuniform_entropy` | guaranteed | 2500 | 0.5 | 0.499 |  | 0.483 / 0.476 | 1.07e-16 | 0.011 |
| `discreteuniform_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_ilogccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_kurtosis` | guaranteed | 2500 | 1.61 | 1.54 |  |  | 2.98e-16 | 0.03 |
| `discreteuniform_logccdf` | guaranteed | 6000 | 1.27 | 1.15 |  |  | 2.09e-16 | 0.021 |
| `discreteuniform_logcdf` | guaranteed | 6000 | 1.21 | 1.16 |  |  | 2.05e-16 | 0.021 |
| `discreteuniform_logpmf` | guaranteed | 6000 | 0.5 | 0.499 |  |  | 1.09e-16 | 0.011 |
| `discreteuniform_mean` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_median` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_pmf` | guaranteed | 6000 | 0.5 | 0.499 |  |  | 1.09e-16 | 0.011 |
| `discreteuniform_skewness` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `discreteuniform_standard_deviation` | guaranteed | 2500 | 0.911 | 0.818 |  |  | 1.40e-16 | 0.014 |
| `discreteuniform_variance` | guaranteed | 2500 | 1 | 1 |  |  | 1.65e-16 | 0.017 |
| `exponential_ccdf` | guaranteed | 5999 | 2.12 | 1.79 |  |  | 3.43e-16 | 0.034 |
| `exponential_cdf` | guaranteed | 5999 | 1.92 | 1.53 |  |  | 2.89e-16 | 0.029 |
| `exponential_entropy` | guaranteed | 2499 | 3.19e+04 | 1.02e+04 |  | 0.488 / 0.476 | 5.82e-12 | 0.011 |
| `exponential_iccdf` | guaranteed | 3999 | 3.13 | 2.14 | 2 |  | 3.74e-16 | 0.027 |
| `exponential_icdf` | guaranteed | 3999 | 1.33 | 1.15 | 0.944 |  | 1.80e-16 | 0.016 |
| `exponential_ilogccdf` | guaranteed | 3999 | 0.5 | 0.5 |  |  | 1.08e-16 | 0.011 |
| `exponential_ilogcdf` | guaranteed | 3999 | 1.59 | 1.27 | 1.05 |  | 2.25e-16 | 0.015 |
| `exponential_kurtosis` | guaranteed | 2499 | 0 | 0 |  |  | 0 | 0 |
| `exponential_logccdf` | guaranteed | 5999 | 0.5 | 0.499 |  |  | 1.03e-16 | 0.01 |
| `exponential_logcdf` | guaranteed | 5999 | 3.18 | 2.66 |  |  | 6.06e-16 | 0.061 |
| `exponential_logpdf` | guaranteed | 5999 | 41.7 | 5.04 |  | 2.25 / 1.3 | 6.94e-15 | 0.05 |
| `exponential_mean` | guaranteed | 2499 | 0 | 0 |  |  | 0 | 0 |
| `exponential_median` | guaranteed | 2499 | 0.787 | 0.78 |  |  | 1.41e-16 | 0.014 |
| `exponential_mode` | guaranteed | 2499 | 0 | 0 |  |  | 0 | 0 |
| `exponential_pdf` | guaranteed | 5999 | 2.95 | 2.49 |  |  | 4.10e-16 | 0.041 |
| `exponential_skewness` | guaranteed | 2499 | 0 | 0 |  |  | 0 | 0 |
| `exponential_standard_deviation` | guaranteed | 2499 | 0 | 0 |  |  | 0 | 0 |
| `exponential_variance` | guaranteed | 2499 | 0.5 | 0.499 |  |  | 1.06e-16 | 0.011 |
| `f_ccdf` | guaranteed | 6000 | 25.1 | 7.4 |  |  | 3.52e-15 | 0.35 |
| `f_cdf` | guaranteed | 6000 | 13 | 8.75 |  |  | 2.28e-15 | 0.23 |
| `f_entropy` | guaranteed | 2528 | 4.88e+03 | 1.58e+03 |  | 41.6 / 34.6 | 5.96e-13 | 0.92 |
| `f_entropy` | loose | 30 | 18.3 | 18.3 |  | 5.4 / 5.4 | 3.10e-15 | 0.12 |
| `f_iccdf` | guaranteed | 4000 | 58.9 | 50.2 | 25.6 |  | 1.07e-14 | 0.36 |
| `f_icdf` | guaranteed | 4000 | 122 | 60.4 | 23.4 |  | 1.37e-14 | 0.46 |
| `f_ilogccdf` | guaranteed | 4000 | 39.8 | 36.5 | 39.8 |  | 6.23e-15 | 0.62 |
| `f_ilogcdf` | guaranteed | 4000 | 43.1 | 24.2 | 43.1 |  | 6.44e-15 | 0.48 |
| `f_kurtosis` | guaranteed | 2528 | 3.54 | 3.2 |  |  | 5.21e-16 | 0.052 |
| `f_kurtosis` | loose | 30 | 3.55 | 3.55 |  |  | 3.99e-16 | 0.04 |
| `f_logccdf` | guaranteed | 6000 | 13.9 | 10.2 |  |  | 2.33e-15 | 0.23 |
| `f_logcdf` | guaranteed | 6000 | 26.2 | 9.68 |  |  | 3.58e-15 | 0.36 |
| `f_logpdf` | guaranteed | 6000 | 159 | 32.7 |  | 3.48 / 2.2 | 2.26e-14 | 0.077 |
| `f_mean` | guaranteed | 2528 | 0.5 | 0.5 |  |  | 1.11e-16 | 0.011 |
| `f_mean` | loose | 30 | 0.473 | 0.473 |  |  | 1.05e-16 | 0.01 |
| `f_median` | guaranteed | 2528 | 110 | 107 | 9.36 |  | 1.52e-14 | 0.14 |
| `f_median` | loose | 30 | 17.3 | 17.3 | 2.5 |  | 3.55e-15 | 0.033 |
| `f_mode` | guaranteed | 2528 | 1.79 | 1.54 |  |  | 2.62e-16 | 0.026 |
| `f_mode` | loose | 30 | 0.959 | 0.959 |  |  | 1.25e-16 | 0.012 |
| `f_pdf` | guaranteed | 6000 | 6.5 | 4.93 |  |  | 1.84e-15 | 0.091 |
| `f_skewness` | guaranteed | 2528 | 3.18 | 2.69 |  |  | 4.39e-16 | 0.044 |
| `f_skewness` | loose | 30 | 2.79 | 2.79 |  |  | 3.20e-16 | 0.032 |
| `f_standard_deviation` | guaranteed | 2528 | 2.06 | 1.82 |  |  | 2.77e-16 | 0.028 |
| `f_standard_deviation` | loose | 30 | 1.32 | 1.32 |  |  | 1.80e-16 | 0.018 |
| `f_variance` | guaranteed | 2528 | 3.78 | 3.34 |  |  | 5.54e-16 | 0.055 |
| `f_variance` | loose | 30 | 2.18 | 2.18 |  |  | 3.68e-16 | 0.037 |
| `gamma_ccdf` | guaranteed | 5905 | 15.8 | 5.84 |  |  | 1.76e-15 | 0.18 |
| `gamma_ccdf` | loose | 95 | 1.67 | 1.67 |  |  | 2.61e-16 | 0.026 |
| `gamma_cdf` | guaranteed | 5905 | 8.88 | 6.32 |  |  | 1.37e-15 | 0.14 |
| `gamma_cdf` | loose | 95 | 3.22 | 3.22 |  |  | 4.76e-16 | 0.048 |
| `gamma_entropy` | guaranteed | 2460 | 430 | 176 |  | 23.7 / 13.1 | 6.56e-14 | 0.53 |
| `gamma_entropy` | loose | 40 | 0.486 | 0.486 |  | 0.446 / 0.446 | 9.90e-17 | 0.0099 |
| `gamma_iccdf` | guaranteed | 3938 | 546 | 320 | 8.98 |  | 1.16e-13 | 0.13 |
| `gamma_iccdf` | loose | 62 | 0.834 | 0.834 | 0.834 |  | 1.62e-16 | 0.016 |
| `gamma_icdf` | guaranteed | 3938 | 654 | 529 | 5.6 |  | 2.26e-12 | 0.079 |
| `gamma_icdf` | loose | 62 | 1.03 | 1.03 | 1.03 |  | 1.32e-16 | 0.013 |
| `gamma_ilogccdf` | guaranteed | 3938 | 456 | 64.3 | 4.18 |  | 6.01e-14 | 0.073 |
| `gamma_ilogccdf` | loose | 62 | 1.09 | 1.09 |  |  | 1.74e-16 | 0.017 |
| `gamma_ilogcdf` | guaranteed | 3938 | 58.5 | 11.2 | 2.32 |  | 9.81e-15 | 0.044 |
| `gamma_ilogcdf` | loose | 62 | 0.851 | 0.851 | 0.844 |  | 1.43e-16 | 0.014 |
| `gamma_kurtosis` | guaranteed | 2460 | 0.5 | 0.498 |  |  | 1.09e-16 | 0.011 |
| `gamma_kurtosis` | loose | 40 | 0.487 | 0.487 |  |  | 9.38e-17 | 0.0094 |
| `gamma_logccdf` | guaranteed | 5905 | 14.1 | 6.97 |  |  | 1.76e-15 | 0.18 |
| `gamma_logccdf` | loose | 95 | 3.56 | 3.56 |  |  | 4.76e-16 | 0.048 |
| `gamma_logcdf` | guaranteed | 5905 | 10.8 | 7.18 |  |  | 1.99e-15 | 0.2 |
| `gamma_logcdf` | loose | 95 | 1.67 | 1.67 |  |  | 2.61e-16 | 0.026 |
| `gamma_logpdf` | guaranteed | 5905 | 4.78e+297 | 4.94e+272 |  | 1.58 / 1.03 | 7.57e+281 | 0.035 |
| `gamma_logpdf` | loose | 95 | 0.914 | 0.914 |  | 0.873 / 0.873 | 1.94e-16 | 0.019 |
| `gamma_mean` | guaranteed | 2460 | 0.5 | 0.499 |  |  | 1.09e-16 | 0.011 |
| `gamma_mean` | loose | 40 | 0.456 | 0.456 |  |  | 9.45e-17 | 0.0094 |
| `gamma_median` | guaranteed | 2460 | 155 | 116 | 5.76 |  | 1.87e-14 | 0.082 |
| `gamma_median` | loose | 40 | 0.859 | 0.859 |  |  | 1.31e-16 | 0.013 |
| `gamma_mode` | guaranteed | 2460 | 0.5 | 0.498 |  |  | 1.06e-16 | 0.011 |
| `gamma_mode` | loose | 40 | 0.467 | 0.467 |  |  | 7.97e-17 | 0.008 |
| `gamma_pdf` | guaranteed | 5905 | 6.38 | 4.66 |  |  | 7.39e-16 | 0.074 |
| `gamma_pdf` | loose | 95 | 7.77 | 7.77 |  |  | 1.09e-15 | 0.11 |
| `gamma_skewness` | guaranteed | 2460 | 1.42 | 1.33 |  |  | 1.62e-16 | 0.016 |
| `gamma_skewness` | loose | 40 | 1.08 | 1.08 |  |  | 1.24e-16 | 0.012 |
| `gamma_standard_deviation` | guaranteed | 2460 | 1.13 | 1.1 |  |  | 1.85e-16 | 0.019 |
| `gamma_standard_deviation` | loose | 40 | 0.71 | 0.71 |  |  | 1.28e-16 | 0.013 |
| `gamma_variance` | guaranteed | 2460 | 1.31 | 1.24 |  |  | 1.90e-16 | 0.019 |
| `gamma_variance` | loose | 40 | 1 | 1 |  |  | 1.40e-16 | 0.014 |
| `geometric_ccdf` | guaranteed | 6000 | 1.53 | 1.36 |  |  | 2.53e-16 | 0.025 |
| `geometric_cdf` | guaranteed | 6000 | 1 | 0.934 |  |  | 1.84e-16 | 0.018 |
| `geometric_entropy` | guaranteed | 2500 | 0.5 | 0.5 |  | 0.495 / 0.481 | 1.10e-16 | 0.011 |
| `geometric_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `geometric_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `geometric_ilogccdf` | guaranteed | 4000 | 512 | 4 |  |  | 8.79e-14 | 0 |
| `geometric_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `geometric_kurtosis` | guaranteed | 2500 | 0.93 | 0.886 |  |  | 1.67e-16 | 0.017 |
| `geometric_logccdf` | guaranteed | 6000 | 0.5 | 0.499 |  |  | 1.10e-16 | 0.011 |
| `geometric_logcdf` | guaranteed | 6000 | 2.18 | 2.01 |  |  | 3.68e-16 | 0.037 |
| `geometric_logpmf` | guaranteed | 6000 | 0.5 | 0.5 |  |  | 1.10e-16 | 0.011 |
| `geometric_mean` | guaranteed | 2500 | 0.5 | 0.5 |  |  | 1.11e-16 | 0.011 |
| `geometric_median` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `geometric_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `geometric_pmf` | guaranteed | 6000 | 2 | 1.81 |  |  | 3.41e-16 | 0.034 |
| `geometric_skewness` | guaranteed | 2500 | 2.12 | 1.89 |  |  | 3.02e-16 | 0.03 |
| `geometric_standard_deviation` | guaranteed | 2500 | 1.31 | 1.13 |  |  | 1.77e-16 | 0.018 |
| `geometric_variance` | guaranteed | 2500 | 1.6 | 1.53 |  |  | 2.29e-16 | 0.023 |
| `gev_ccdf` | guaranteed | 5652 | 2.86 | 2.44 |  |  | 4.47e-16 | 0.045 |
| `gev_ccdf` | loose | 348 | 0 | 0 |  |  | 0 | 0 |
| `gev_cdf` | guaranteed | 5652 | 2.69 | 1.18 |  |  | 3.08e-16 | 0.031 |
| `gev_cdf` | loose | 348 | 0 | 0 |  |  | 0 | 0 |
| `gev_entropy` | guaranteed | 2539 | 0.5 | 0.499 |  | 0.489 / 0.488 | 1.09e-16 | 0.011 |
| `gev_iccdf` | guaranteed | 4000 | 369 | 41.2 | 2.54 |  | 7.87e-14 | 0.039 |
| `gev_icdf` | guaranteed | 4000 | 1.57e+03 | 65 | 2.6 |  | 1.81e-13 | 0.041 |
| `gev_ilogccdf` | guaranteed | 4000 | 23.7 | 8.99 | 1.09 |  | 3.33e-15 | 0.02 |
| `gev_ilogcdf` | guaranteed | 4000 | 0.979 | 0.949 | 0.979 |  | 1.79e-16 | 0.017 |
| `gev_kurtosis` | guaranteed | 2539 | 8.06e+5 | 8.05e+5 |  | 8.22 / 4.49 | 1.65e-10 | 0.18 |
| `gev_logccdf` | guaranteed | 5652 | 2.4 | 1.57 |  |  | 3.50e-16 | 0.035 |
| `gev_logccdf` | loose | 348 | 0 | 0 |  |  | 0 | 0 |
| `gev_logcdf` | guaranteed | 5652 | 5.14 | 3.13 |  |  | 7.43e-16 | 0.074 |
| `gev_logcdf` | loose | 348 | 0 | 0 |  |  | 0 | 0 |
| `gev_logpdf` | guaranteed | 5652 | 73.1 | 2.26 |  | 1.78 / 1.33 | 1.55e-14 | 0.04 |
| `gev_logpdf` | loose | 348 | 0 | 0 |  | 0 / 0 | 0 | 0 |
| `gev_mean` | guaranteed | 2539 | 2.85 | 0.935 |  |  | 4.61e-16 | 0.046 |
| `gev_median` | guaranteed | 2539 | 0.961 | 0.933 |  |  | 1.66e-16 | 0.017 |
| `gev_mode` | guaranteed | 2539 | 0.929 | 0.901 |  |  | 1.58e-16 | 0.016 |
| `gev_pdf` | guaranteed | 5652 | 2.91 | 1.95 |  |  | 3.73e-16 | 0.037 |
| `gev_pdf` | loose | 348 | 0 | 0 |  |  | 0 | 0 |
| `gev_skewness` | guaranteed | 2539 | 33.3 | 9.29 |  | 0.887 / 0.868 | 4.67e-15 | 0.02 |
| `gev_standard_deviation` | guaranteed | 2539 | 1.2 | 1.18 |  |  | 2.20e-16 | 0.022 |
| `gev_variance` | guaranteed | 2539 | 1.79 | 1.58 |  |  | 2.51e-16 | 0.025 |
| `gumbel_ccdf` | guaranteed | 6000 | 3.21 | 2.93 |  |  | 5.60e-16 | 0.056 |
| `gumbel_cdf` | guaranteed | 6000 | 3.21 | 2.16 |  |  | 4.21e-16 | 0.042 |
| `gumbel_entropy` | guaranteed | 2500 | 0.5 | 0.499 |  | 0.483 / 0.47 | 1.07e-16 | 0.011 |
| `gumbel_iccdf` | guaranteed | 4000 | 90.2 | 31.3 | 2.65 |  | 1.98e-14 | 0.047 |
| `gumbel_icdf` | guaranteed | 4000 | 785 | 43.2 | 2.45 |  | 1.70e-13 | 0.039 |
| `gumbel_ilogccdf` | guaranteed | 4000 | 21.7 | 1.15 | 0.993 |  | 2.91e-15 | 0.019 |
| `gumbel_ilogcdf` | guaranteed | 4000 | 0.923 | 0.782 | 0.83 |  | 1.53e-16 | 0.014 |
| `gumbel_kurtosis` | guaranteed | 2500 | 0.2 | 0.2 |  |  | 3.70e-17 | 0.0037 |
| `gumbel_logccdf` | guaranteed | 6000 | 4.04 | 2.6 |  |  | 4.92e-16 | 0.049 |
| `gumbel_logcdf` | guaranteed | 6000 | 1.35 | 1.29 |  |  | 2.75e-16 | 0.027 |
| `gumbel_logpdf` | guaranteed | 6000 | 501 | 11.6 |  | 1.81 / 1.23 | 6.08e-14 | 0.04 |
| `gumbel_mean` | guaranteed | 2500 | 0.866 | 0.658 |  |  | 1.42e-16 | 0.014 |
| `gumbel_median` | guaranteed | 2500 | 0.975 | 0.961 |  |  | 1.73e-16 | 0.017 |
| `gumbel_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `gumbel_pdf` | guaranteed | 6000 | 3.94 | 2.69 |  |  | 6.30e-16 | 0.063 |
| `gumbel_skewness` | guaranteed | 2500 | 0.258 | 0.258 |  |  | 5.03e-17 | 0.005 |
| `gumbel_standard_deviation` | guaranteed | 2500 | 0.608 | 0.602 |  |  | 1.20e-16 | 0.012 |
| `gumbel_variance` | guaranteed | 2500 | 1.42 | 1.36 |  |  | 1.92e-16 | 0.019 |
| `hypergeometric_ccdf` | guaranteed | 6000 | 7.73 | 6.95 |  |  | 1.08e-15 | 0.11 |
| `hypergeometric_cdf` | guaranteed | 6000 | 9.06 | 6.66 |  |  | 1.08e-15 | 0.11 |
| `hypergeometric_entropy` | guaranteed | 500 | 1.24 | 1.24 |  | 0.748 / 0.748 | 2.12e-16 | 0.017 |
| `hypergeometric_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `hypergeometric_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `hypergeometric_ilogccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `hypergeometric_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `hypergeometric_kurtosis` | guaranteed | 421 | 3 | 3 |  |  | 4.44e-16 | 0.044 |
| `hypergeometric_logccdf` | guaranteed | 6000 | 9.05 | 6.84 |  |  | 1.13e-15 | 0.11 |
| `hypergeometric_logcdf` | guaranteed | 6000 | 9.39 | 6.9 |  |  | 1.11e-15 | 0.11 |
| `hypergeometric_logpmf` | guaranteed | 6000 | 25.9 | 6.84 |  |  | 4.86e-15 | 0.49 |
| `hypergeometric_mean` | guaranteed | 500 | 0.991 | 0.991 |  |  | 1.67e-16 | 0.017 |
| `hypergeometric_median` | guaranteed | 500 | 1.41e+14 | 1.41e+14 |  |  | 0.0213 | 0 |
| `hypergeometric_mode` | guaranteed | 500 | 0 | 0 |  |  | 0 | 0 |
| `hypergeometric_pmf` | guaranteed | 6000 | 9.97 | 7.59 |  |  | 1.27e-15 | 0.13 |
| `hypergeometric_skewness` | guaranteed | 421 | 2.94 | 2.94 |  |  | 3.94e-16 | 0.039 |
| `hypergeometric_standard_deviation` | guaranteed | 500 | 1.41 | 1.41 |  |  | 2.16e-16 | 0.022 |
| `hypergeometric_variance` | guaranteed | 500 | 2 | 2 |  |  | 3.17e-16 | 0.032 |
| `inversegamma_ccdf` | guaranteed | 5569 | 8.79 | 5.75 |  |  | 1.36e-15 | 0.14 |
| `inversegamma_ccdf` | loose | 431 | 3.3 | 3.3 |  |  | 5.00e-16 | 0.05 |
| `inversegamma_cdf` | guaranteed | 5569 | 15.5 | 5.99 |  |  | 1.74e-15 | 0.17 |
| `inversegamma_cdf` | loose | 431 | 1.88 | 1.88 |  |  | 3.32e-16 | 0.033 |
| `inversegamma_entropy` | guaranteed | 2460 | 480 | 187 |  | 19.3 / 13.9 | 6.29e-14 | 0.43 |
| `inversegamma_entropy` | loose | 40 | 1.41 | 1.41 |  | 1.06 / 1.06 | 2.36e-16 | 0.024 |
| `inversegamma_iccdf` | guaranteed | 3938 | 1.03e+03 | 550 | 6.63 |  | 1.25e-13 | 0.075 |
| `inversegamma_iccdf` | loose | 62 | 1.17 | 1.17 | 1.09 |  | 1.92e-16 | 0.019 |
| `inversegamma_icdf` | guaranteed | 3938 | 1e+03 | 329 | 9.7 |  | 1.16e-13 | 0.13 |
| `inversegamma_icdf` | loose | 62 | 0.999 | 0.999 | 0.999 |  | 1.66e-16 | 0.017 |
| `inversegamma_ilogccdf` | guaranteed | 3938 | 66.9 | 11.8 | 4.24 |  | 9.84e-15 | 0.048 |
| `inversegamma_ilogccdf` | loose | 62 | 0.962 | 0.962 | 0.899 |  | 1.89e-16 | 0.019 |
| `inversegamma_ilogcdf` | guaranteed | 3938 | 271 | 73.5 | 6.06 |  | 6.00e-14 | 0.074 |
| `inversegamma_ilogcdf` | loose | 62 | 1.16 | 1.16 |  |  | 1.47e-16 | 0.015 |
| `inversegamma_kurtosis` | guaranteed | 2460 | 2.09 | 1.99 |  |  | 2.92e-16 | 0.029 |
| `inversegamma_kurtosis` | loose | 40 | 1.78 | 1.78 |  |  | 2.33e-16 | 0.023 |
| `inversegamma_logccdf` | guaranteed | 5569 | 10.9 | 7.41 |  |  | 1.96e-15 | 0.2 |
| `inversegamma_logccdf` | loose | 431 | 1.88 | 1.88 |  |  | 3.32e-16 | 0.033 |
| `inversegamma_logcdf` | guaranteed | 5569 | 8.79 | 6.55 |  |  | 1.36e-15 | 0.14 |
| `inversegamma_logcdf` | loose | 431 | 3.3 | 3.3 |  |  | 5.00e-16 | 0.05 |
| `inversegamma_logpdf` | guaranteed | 5569 | 18 | 6.05 |  | 1.62 / 1.04 | 2.52e-15 | 0.036 |
| `inversegamma_logpdf` | loose | 431 | 7.71 | 7.71 |  | 1.93 / 1.93 | 1.21e-15 | 0.043 |
| `inversegamma_mean` | guaranteed | 2460 | 0.5 | 0.499 |  |  | 1.08e-16 | 0.011 |
| `inversegamma_mean` | loose | 40 | 0.499 | 0.499 |  |  | 7.71e-17 | 0.0077 |
| `inversegamma_median` | guaranteed | 2460 | 1.63e+03 | 1.16e+03 | 6.61 |  | 2.77e-13 | 0.081 |
| `inversegamma_median` | loose | 40 | 1.09 | 1.09 |  |  | 1.36e-16 | 0.014 |
| `inversegamma_mode` | guaranteed | 2460 | 1.44 | 1.33 |  |  | 1.94e-16 | 0.019 |
| `inversegamma_mode` | loose | 40 | 0.476 | 0.476 |  |  | 7.29e-17 | 0.0073 |
| `inversegamma_pdf` | guaranteed | 5569 | 5.37 | 4.17 |  |  | 1.45e-13 | 0.072 |
| `inversegamma_pdf` | loose | 431 | 10.5 | 10.5 |  |  | 1.24e-15 | 0.12 |
| `inversegamma_skewness` | guaranteed | 2460 | 1.28 | 1.22 |  |  | 1.89e-16 | 0.019 |
| `inversegamma_skewness` | loose | 40 | 1.18 | 1.18 |  |  | 1.48e-16 | 0.015 |
| `inversegamma_standard_deviation` | guaranteed | 2460 | 1.78 | 1.64 |  |  | 2.55e-16 | 0.025 |
| `inversegamma_standard_deviation` | loose | 40 | 1.01 | 1.01 |  |  | 1.64e-16 | 0.016 |
| `inversegamma_variance` | guaranteed | 2460 | 2.29 | 2.13 |  |  | 3.69e-16 | 0.037 |
| `inversegamma_variance` | loose | 40 | 1.67 | 1.67 |  |  | 2.76e-16 | 0.028 |
| `laplace_ccdf` | guaranteed | 6000 | 2.13 | 1.8 |  |  | 0.00161 | 0.032 |
| `laplace_cdf` | guaranteed | 6000 | 1.76 | 1.69 |  |  | 4.26e-4 | 0.037 |
| `laplace_entropy` | guaranteed | 2500 | 0.499 | 0.497 |  | 0.485 / 0.478 | 1.10e-16 | 0.011 |
| `laplace_iccdf` | guaranteed | 4000 | 28.5 | 6.24 | 6.75 |  | 4.23e-15 | 0.089 |
| `laplace_icdf` | guaranteed | 4000 | 97 | 16.2 | 27.1 |  | 1.45e-14 | 0.33 |
| `laplace_ilogccdf` | guaranteed | 4000 | 34.8 | 11.7 | 15.6 |  | 5.34e-15 | 0.22 |
| `laplace_ilogcdf` | guaranteed | 4000 | 114 | 6.13 | 6.41 |  | 1.61e-14 | 0.093 |
| `laplace_kurtosis` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `laplace_logccdf` | guaranteed | 6000 | 3.71 | 2.88 |  |  | 4.26e-4 | 0.063 |
| `laplace_logcdf` | guaranteed | 6000 | 3.22 | 2.98 |  |  | 0.00161 | 0.05 |
| `laplace_logpdf` | guaranteed | 6000 | 0.5 | 0.5 |  | 0.492 / 0.473 | 1.09e-16 | 0.011 |
| `laplace_mean` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `laplace_median` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `laplace_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `laplace_pdf` | guaranteed | 6000 | 2.11 | 1.91 |  |  | 9.43e-15 | 0.036 |
| `laplace_skewness` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `laplace_standard_deviation` | guaranteed | 2500 | 1.1 | 1.09 |  |  | 1.77e-16 | 0.018 |
| `laplace_variance` | guaranteed | 2500 | 0.5 | 0.5 |  |  | 1.09e-16 | 0.011 |
| `logistic_ccdf` | guaranteed | 6000 | 2.11 | 1.86 |  |  | 1.23e-13 | 0.038 |
| `logistic_cdf` | guaranteed | 6000 | 2.35 | 1.99 |  |  | 4.01e-16 | 0.04 |
| `logistic_entropy` | guaranteed | 2500 | 0.5 | 0.499 |  | 0.493 / 0.486 | 1.09e-16 | 0.011 |
| `logistic_iccdf` | guaranteed | 4000 | 7.57e+6 | 4.18e+6 | 2.65 |  | 1.42e-9 | 0.047 |
| `logistic_icdf` | guaranteed | 4000 | 7.57e+6 | 4.18e+6 | 3.38 |  | 1.42e-9 | 0.044 |
| `logistic_ilogccdf` | guaranteed | 4000 | 1.88 | 0.967 | 0.977 |  | 2.54e-16 | 0.019 |
| `logistic_ilogcdf` | guaranteed | 4000 | 2.74 | 0.99 | 1.08 |  | 3.85e-16 | 0.017 |
| `logistic_kurtosis` | guaranteed | 2500 | 0.2 | 0.2 |  |  | 3.70e-17 | 0.0037 |
| `logistic_logccdf` | guaranteed | 6000 | 3.15 | 2.7 |  |  | 5.46e-16 | 0.055 |
| `logistic_logcdf` | guaranteed | 6000 | 3.4 | 2.9 |  |  | 1.23e-13 | 0.057 |
| `logistic_logpdf` | guaranteed | 6000 | 589 | 28.9 |  | 2.04 / 1.68 | 7.05e-14 | 0.045 |
| `logistic_mean` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `logistic_median` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `logistic_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `logistic_pdf` | guaranteed | 6000 | 4.4 | 3.32 |  |  | 3.89e-13 | 0.064 |
| `logistic_skewness` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `logistic_standard_deviation` | guaranteed | 2500 | 0.566 | 0.561 |  |  | 1.15e-16 | 0.012 |
| `logistic_variance` | guaranteed | 2500 | 1.42 | 1.36 |  |  | 1.92e-16 | 0.019 |
| `lognormal_ccdf` | guaranteed | 6000 | 5.51 | 4.35 |  |  | 8.66e-16 | 0.087 |
| `lognormal_cdf` | guaranteed | 6000 | 5.49 | 3.84 |  |  | 0.2 | 0.076 |
| `lognormal_entropy` | guaranteed | 2500 | 0.5 | 0.499 |  | 0.466 / 0.462 | 1.03e-16 | 0.01 |
| `lognormal_iccdf` | guaranteed | 4000 | 13.1 | 9.74 | 10.7 |  | 1.96e-15 | 0.18 |
| `lognormal_icdf` | guaranteed | 4000 | 13.8 | 9.63 | 13.8 |  | 1.76e-15 | 0.18 |
| `lognormal_ilogccdf` | guaranteed | 4000 | 15 | 7.5 | 3.44 |  | 2.29e-15 | 0.042 |
| `lognormal_ilogcdf` | guaranteed | 4000 | 15.9 | 8.63 | 2.92 |  | 2.29e-15 | 0.058 |
| `lognormal_kurtosis` | guaranteed | 2500 | 5.42 | 5.32 |  |  | 8.23e-16 | 0.082 |
| `lognormal_logccdf` | guaranteed | 6000 | 6.46 | 4.18 |  |  | 0.2 | 0.076 |
| `lognormal_logcdf` | guaranteed | 6000 | 5.38 | 4.12 |  |  | 8.87e-16 | 0.089 |
| `lognormal_logpdf` | guaranteed | 6000 | 0.5 | 0.499 |  | 0.487 / 0.474 | 1.08e-16 | 0.011 |
| `lognormal_mean` | guaranteed | 2500 | 1.57 | 1.48 |  |  | 2.44e-16 | 0.024 |
| `lognormal_median` | guaranteed | 2500 | 0.505 | 0.501 |  |  | 1.08e-16 | 0.011 |
| `lognormal_mode` | guaranteed | 2500 | 1.64 | 1.54 |  |  | 2.48e-16 | 0.025 |
| `lognormal_pdf` | guaranteed | 6000 | 2.04 | 1.78 |  |  | 3.19e-16 | 0.032 |
| `lognormal_skewness` | guaranteed | 2500 | 2.32 | 2.11 |  |  | 3.46e-16 | 0.035 |
| `lognormal_standard_deviation` | guaranteed | 2500 | 2.74 | 2.4 |  |  | 3.74e-16 | 0.037 |
| `lognormal_variance` | guaranteed | 2500 | 2.6 | 2.35 |  |  | 3.97e-16 | 0.04 |
| `logseries_ccdf` | guaranteed | 6000 | 17.3 | 12.7 |  |  | 2.30e-15 | 0.23 |
| `logseries_cdf` | guaranteed | 6000 | 9.77 | 6.67 |  |  | 1.66e-15 | 0.17 |
| `logseries_entropy` | guaranteed | 600 | 2.42 | 2.42 |  | 0.921 / 0.921 | 3.21e-16 | 0.02 |
| `logseries_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `logseries_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `logseries_ilogccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `logseries_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `logseries_kurtosis` | guaranteed | 600 | 18.2 | 18.2 |  |  | 2.39e-15 | 0.24 |
| `logseries_logccdf` | guaranteed | 6000 | 43.3 | 25.8 |  |  | 6.31e-15 | 0.63 |
| `logseries_logcdf` | guaranteed | 6000 | 16 | 12.5 |  |  | 3.13e-15 | 0.31 |
| `logseries_logpmf` | guaranteed | 6000 | 1.85 | 1.41 |  |  | 3.15e-16 | 0.031 |
| `logseries_mean` | guaranteed | 600 | 1.91 | 1.91 |  |  | 3.08e-16 | 0.031 |
| `logseries_median` | guaranteed | 600 | 0 | 0 |  |  | 0 | 0 |
| `logseries_mode` | guaranteed | 600 | 0 | 0 |  |  | 0 | 0 |
| `logseries_pmf` | guaranteed | 6000 | 3.68 | 2.94 |  |  | 4.54e-16 | 0.045 |
| `logseries_skewness` | guaranteed | 600 | 19.6 | 19.6 |  |  | 2.89e-15 | 0.29 |
| `logseries_standard_deviation` | guaranteed | 600 | 2.22 | 2.22 |  |  | 2.70e-16 | 0.027 |
| `logseries_variance` | guaranteed | 600 | 3.29 | 3.29 |  |  | 4.92e-16 | 0.049 |
| `negativebinomial_ccdf` | guaranteed | 6000 | 18.3 | 11.1 |  |  | 2.59e-15 | 0.26 |
| `negativebinomial_cdf` | guaranteed | 6000 | 17.4 | 9.28 |  |  | 2.05e-15 | 0.2 |
| `negativebinomial_entropy` | guaranteed | 599 | 1.85 | 1.85 |  | 0.94 / 0.94 | 2.87e-16 | 0.021 |
| `negativebinomial_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `negativebinomial_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `negativebinomial_ilogccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `negativebinomial_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `negativebinomial_kurtosis` | guaranteed | 602 | 1.73 | 1.73 |  |  | 2.28e-16 | 0.023 |
| `negativebinomial_logccdf` | guaranteed | 6000 | 16.7 | 11.8 |  |  | 2.01e-15 | 0.2 |
| `negativebinomial_logcdf` | guaranteed | 6000 | 27.1 | 13.3 |  |  | 3.05e-15 | 0.31 |
| `negativebinomial_logpmf` | guaranteed | 6000 | 2.99 | 2.26 |  |  | 4.47e-16 | 0.045 |
| `negativebinomial_mean` | guaranteed | 602 | 1.5 | 1.5 |  |  | 2.36e-16 | 0.024 |
| `negativebinomial_median` | guaranteed | 602 | 0 | 0 |  |  | 0 | 0 |
| `negativebinomial_mode` | guaranteed | 602 | 0 | 0 |  |  | 0 | 0 |
| `negativebinomial_pmf` | guaranteed | 6000 | 6.89 | 5.71 |  |  | 9.17e-16 | 0.092 |
| `negativebinomial_skewness` | guaranteed | 602 | 1.9 | 1.9 |  |  | 2.72e-16 | 0.027 |
| `negativebinomial_standard_deviation` | guaranteed | 602 | 1.19 | 1.19 |  |  | 1.91e-16 | 0.019 |
| `negativebinomial_variance` | guaranteed | 602 | 1.63 | 1.63 |  |  | 2.79e-16 | 0.028 |
| `noncentralchisquared_ccdf` | guaranteed | 5339 | 9 | 7.11 |  |  | 1.30e-15 | 0.13 |
| `noncentralchisquared_ccdf` | loose | 661 | 0 | 0 |  |  | 0 | 0 |
| `noncentralchisquared_cdf` | guaranteed | 5339 | 7.74 | 4.37 |  |  | 9.22e-16 | 0.092 |
| `noncentralchisquared_cdf` | loose | 661 | 0 | 0 |  |  | 0 | 0 |
| `noncentralchisquared_iccdf` | guaranteed | 3500 | 702 | 466 | 11.2 |  | 1 | 0.15 |
| `noncentralchisquared_iccdf` | loose | 500 | 0.946 | 0.946 |  |  | 1.36e-16 | 0.014 |
| `noncentralchisquared_icdf` | guaranteed | 3500 | 525 | 404 | 6.44 |  | 7.93e-14 | 0.073 |
| `noncentralchisquared_icdf` | loose | 500 | 5.83 | 5.83 | 3.61 |  | 1.03e-15 | 0.056 |
| `noncentralchisquared_ilogccdf` | guaranteed | 2514 | 150 | 78.2 | 6.98 |  | 5.18e-11 | 0.098 |
| `noncentralchisquared_ilogccdf` | loose | 1397 | 0.975 | 0.941 | 0.975 |  | 1.32e-16 | 0.013 |
| `noncentralchisquared_ilogcdf` | guaranteed | 2514 | 639 | 354 | 3.7 |  | 8.47e-14 | 0.053 |
| `noncentralchisquared_ilogcdf` | loose | 1486 | 3.82 | 1.45 | 0.378 |  | 2.48e-11 | 0.0059 |
| `noncentralchisquared_kurtosis` | guaranteed | 600 | 2.85 | 2.85 |  |  | 3.66e-16 | 0.037 |
| `noncentralchisquared_logccdf` | guaranteed | 5339 | 10.7 | 6.78 |  |  | 1.38e-15 | 0.14 |
| `noncentralchisquared_logccdf` | loose | 661 | 0 | 0 |  |  | 0 | 0 |
| `noncentralchisquared_logcdf` | guaranteed | 5339 | 11.9 | 7.14 |  |  | 1.61e-15 | 0.16 |
| `noncentralchisquared_logcdf` | loose | 661 | 0 | 0 |  |  | 0 | 0 |
| `noncentralchisquared_logpdf` | guaranteed | 5339 | 1.61 | 1.12 |  | 0.982 / 0.762 | 3.17e-16 | 0.022 |
| `noncentralchisquared_logpdf` | loose | 661 | 0 | 0 |  | 0 / 0 | 0 | 0 |
| `noncentralchisquared_mean` | guaranteed | 600 | 0.5 | 0.5 |  |  | 1.07e-16 | 0.011 |
| `noncentralchisquared_median` | guaranteed | 600 | 538 | 538 | 4.78 |  | 7.26e-14 | 0.064 |
| `noncentralchisquared_mode` | guaranteed | 600 | 3.12 | 3.12 |  |  | 6.46e-16 | 0.065 |
| `noncentralchisquared_pdf` | guaranteed | 5339 | 6.12 | 5.35 |  |  | 8.27e-16 | 0.083 |
| `noncentralchisquared_pdf` | loose | 661 | 0 | 0 |  |  | 0 | 0 |
| `noncentralchisquared_skewness` | guaranteed | 600 | 3.48 | 3.48 |  |  | 4.03e-16 | 0.04 |
| `noncentralchisquared_standard_deviation` | guaranteed | 600 | 0.779 | 0.779 |  |  | 1.34e-16 | 0.013 |
| `noncentralchisquared_variance` | guaranteed | 600 | 0.5 | 0.5 |  |  | 1.10e-16 | 0.011 |
| `noncentralf_ccdf` | guaranteed | 5393 | 11.6 | 7.98 |  |  | 1.45e-15 | 0.15 |
| `noncentralf_ccdf` | loose | 607 | 0 | 0 |  |  | 0 | 0 |
| `noncentralf_cdf` | guaranteed | 5393 | 12.1 | 8.44 |  |  | 1.70e-15 | 0.17 |
| `noncentralf_cdf` | loose | 607 | 0 | 0 |  |  | 0 | 0 |
| `noncentralf_iccdf` | guaranteed | 3500 | 709 | 576 | 15.2 |  | 1.04e-13 | 0.23 |
| `noncentralf_iccdf` | loose | 500 | 2.63 | 2.63 |  |  | 4.93e-16 | 0.049 |
| `noncentralf_icdf` | guaranteed | 3500 | 864 | 557 | 9.24 |  | 1.06e-13 | 0.15 |
| `noncentralf_icdf` | loose | 500 | 55.8 | 55.8 | 4.77 |  | 1.11e-11 | 0.058 |
| `noncentralf_ilogccdf` | guaranteed | 2590 | 404 | 161 | 11.5 |  | 1 | 0.16 |
| `noncentralf_ilogccdf` | loose | 1410 | 2.82 | 2.18 | 0.43 |  | 4.22e-16 | 0.0084 |
| `noncentralf_ilogcdf` | guaranteed | 2590 | 667 | 236 | 7.72 |  | 1 | 0.12 |
| `noncentralf_ilogcdf` | loose | 1410 | 2.83 | 2.62 | 0.681 |  | 5.04e-16 | 0.014 |
| `noncentralf_kurtosis` | guaranteed | 600 | 6.68 | 6.68 |  |  | 9.63e-16 | 0.096 |
| `noncentralf_logccdf` | guaranteed | 5393 | 16.2 | 11.1 |  |  | 2.05e-15 | 0.2 |
| `noncentralf_logccdf` | loose | 607 | 0 | 0 |  |  | 0 | 0 |
| `noncentralf_logcdf` | guaranteed | 5393 | 12.9 | 10.4 |  |  | 1.95e-15 | 0.19 |
| `noncentralf_logcdf` | loose | 607 | 0 | 0 |  |  | 0 | 0 |
| `noncentralf_logpdf` | guaranteed | 5393 | 539 | 158 |  | 3.83 / 3.04 | 7.93e-14 | 0.085 |
| `noncentralf_logpdf` | loose | 607 | 0 | 0 |  | 0 / 0 | 0 | 0 |
| `noncentralf_mean` | guaranteed | 600 | 1.93 | 1.93 |  |  | 3.27e-16 | 0.033 |
| `noncentralf_median` | guaranteed | 600 | 997 | 997 | 7.32 |  | 1.35e-13 | 0.11 |
| `noncentralf_mode` | guaranteed | 600 | 5.48 | 5.48 |  |  | 8.03e-16 | 0.08 |
| `noncentralf_pdf` | guaranteed | 5393 | 8.88 | 5.9 |  |  | 1.02e-15 | 0.1 |
| `noncentralf_pdf` | loose | 607 | 0 | 0 |  |  | 0 | 0 |
| `noncentralf_skewness` | guaranteed | 600 | 4.51 | 4.51 |  |  | 5.77e-16 | 0.058 |
| `noncentralf_standard_deviation` | guaranteed | 600 | 2.06 | 2.06 |  |  | 3.33e-16 | 0.033 |
| `noncentralf_variance` | guaranteed | 600 | 4.48 | 4.48 |  |  | 5.44e-16 | 0.054 |
| `noncentralt_ccdf` | guaranteed | 5712 | 10.7 | 6 |  |  | 1.25e-15 | 0.13 |
| `noncentralt_ccdf` | loose | 288 | 0 | 0 |  |  | 0 | 0 |
| `noncentralt_cdf` | guaranteed | 5712 | 7.88 | 5.9 |  |  | 1.19e-15 | 0.12 |
| `noncentralt_cdf` | loose | 288 | 0 | 0 |  |  | 0 | 0 |
| `noncentralt_iccdf` | guaranteed | 3499 | 1.32e+6 | 1.51e+5 | 5.31 |  | 2.00e-10 | 0.085 |
| `noncentralt_iccdf` | loose | 500 | 2.59 | 2.59 |  |  | 4.30e-16 | 0.043 |
| `noncentralt_icdf` | guaranteed | 3498 | 1.32e+6 | 1.51e+5 | 6.48 |  | 2.00e-10 | 0.09 |
| `noncentralt_icdf` | loose | 500 | 2.22 | 2.22 |  |  | 3.86e-16 | 0.039 |
| `noncentralt_ilogccdf` | guaranteed | 2557 | 33.5 | 22.9 | 4.67 |  | 7.11e-15 | 0.069 |
| `noncentralt_ilogccdf` | loose | 1422 | 1.09 | 0.969 | 0.969 |  | 1.82e-16 | 0.011 |
| `noncentralt_ilogcdf` | guaranteed | 2557 | 34.3 | 21.5 | 7.04 |  | 5.12e-15 | 0.086 |
| `noncentralt_ilogcdf` | loose | 1420 | 1.19 | 1.09 | 0.491 |  | 1.82e-16 | 0.0083 |
| `noncentralt_kurtosis` | guaranteed | 300 | 21.3 | 21.3 |  |  | 4.66e-15 | 0.47 |
| `noncentralt_logccdf` | guaranteed | 5712 | 7.96 | 6.57 |  |  | 1.18e-15 | 0.12 |
| `noncentralt_logccdf` | loose | 288 | 0 | 0 |  |  | 0 | 0 |
| `noncentralt_logcdf` | guaranteed | 5712 | 8.64 | 6.25 |  |  | 1.41e-15 | 0.14 |
| `noncentralt_logcdf` | loose | 288 | 0 | 0 |  |  | 0 | 0 |
| `noncentralt_logpdf` | guaranteed | 5712 | 4.99 | 2.52 |  | 2.49 / 1.69 | 5.83e-16 | 0.055 |
| `noncentralt_logpdf` | loose | 288 | 0 | 0 |  | 0 / 0 | 0 | 0 |
| `noncentralt_mean` | guaranteed | 300 | 1.33 | 1.33 |  |  | 1.82e-16 | 0.018 |
| `noncentralt_median` | guaranteed | 300 | 888 | 888 | 4.22 |  | 1.46e-13 | 0.053 |
| `noncentralt_mode` | guaranteed | 300 | 4.48 | 4.48 |  |  | 6.82e-16 | 0.068 |
| `noncentralt_pdf` | guaranteed | 5712 | 4.95 | 4.41 |  |  | 1.59e-14 | 0.084 |
| `noncentralt_pdf` | loose | 288 | 0 | 0 |  |  | 0 | 0 |
| `noncentralt_skewness` | guaranteed | 300 | 5.15 | 5.15 |  |  | 7.94e-16 | 0.079 |
| `noncentralt_standard_deviation` | guaranteed | 300 | 1.12 | 1.12 |  |  | 2.01e-16 | 0.02 |
| `noncentralt_variance` | guaranteed | 300 | 1.87 | 1.87 |  |  | 2.97e-16 | 0.03 |
| `normal_ccdf` | guaranteed | 6000 | 6.8 | 4.59 |  |  | 9.46e-16 | 0.095 |
| `normal_cdf` | guaranteed | 6000 | 7 | 4.87 |  |  | 1.22e-15 | 0.12 |
| `normal_entropy` | guaranteed | 2517 | 4.52e+03 | 1.85e+03 |  | 0.493 / 0.48 | 8.68e-13 | 0.011 |
| `normal_iccdf` | guaranteed | 4000 | 244 | 8.18 | 3.89 |  | 4.77e-14 | 0.076 |
| `normal_icdf` | guaranteed | 4000 | 56.2 | 21.6 | 5.47 |  | 6.44e-15 | 0.074 |
| `normal_ilogccdf` | guaranteed | 4000 | 6.3 | 4.03 | 2.98 |  | 9.65e-16 | 0.046 |
| `normal_ilogcdf` | guaranteed | 4000 | 22.3 | 6.16 | 2.98 |  | 2.67e-15 | 0.046 |
| `normal_kurtosis` | guaranteed | 2517 | 0 | 0 |  |  | 0 | 0 |
| `normal_logccdf` | guaranteed | 6000 | 8.09 | 4.76 |  |  | 1.35e-15 | 0.13 |
| `normal_logcdf` | guaranteed | 6000 | 7.71 | 4.84 |  |  | 1.02e-15 | 0.1 |
| `normal_logpdf` | guaranteed | 6000 | 0.5 | 0.499 |  | 0.492 / 0.483 | 1.09e-16 | 0.011 |
| `normal_mean` | guaranteed | 2517 | 0 | 0 |  |  | 0 | 0 |
| `normal_median` | guaranteed | 2517 | 0 | 0 |  |  | 0 | 0 |
| `normal_mode` | guaranteed | 2517 | 0 | 0 |  |  | 0 | 0 |
| `normal_pdf` | guaranteed | 6000 | 3.56 | 2.43 |  |  | 4.65e-16 | 0.047 |
| `normal_skewness` | guaranteed | 2517 | 0 | 0 |  |  | 0 | 0 |
| `normal_standard_deviation` | guaranteed | 2517 | 0 | 0 |  |  | 0 | 0 |
| `normal_variance` | guaranteed | 2517 | 0.5 | 0.5 |  |  | 1.09e-16 | 0.011 |
| `pareto_ccdf` | guaranteed | 6000 | 1.66 | 1.14 |  |  | 3.18e-16 | 0.032 |
| `pareto_cdf` | guaranteed | 6000 | 3.25 | 1.46 |  |  | 4.17e-16 | 0.042 |
| `pareto_entropy` | guaranteed | 2500 | 0.5 | 0.499 |  | 0.495 / 0.483 | 1.10e-16 | 0.011 |
| `pareto_iccdf` | guaranteed | 4000 | 2.83 | 2.24 | 2.24 |  | 3.94e-16 | 0.028 |
| `pareto_icdf` | guaranteed | 4000 | 6.92 | 3.81 | 3.81 |  | 9.90e-16 | 0.05 |
| `pareto_ilogccdf` | guaranteed | 4000 | 1.88 | 1.66 | 1.39 |  | 2.79e-16 | 0.018 |
| `pareto_ilogcdf` | guaranteed | 4000 | 11.5 | 4.4 | 1.98 |  | 1.29e-15 | 0.026 |
| `pareto_kurtosis` | guaranteed | 2500 | 2.79 | 2.59 |  |  | 3.70e-16 | 0.037 |
| `pareto_logccdf` | guaranteed | 6000 | 0.5 | 0.492 |  |  | 1.04e-16 | 0.01 |
| `pareto_logcdf` | guaranteed | 6000 | 1.46 | 0.907 |  |  | 1.97e-16 | 0.02 |
| `pareto_logpdf` | guaranteed | 6000 | 5.45e+5 | 1.06e+03 |  | 1.04 / 0.932 | 1.03e-10 | 0.023 |
| `pareto_mean` | guaranteed | 2500 | 1.31 | 1.2 |  |  | 1.94e-16 | 0.019 |
| `pareto_median` | guaranteed | 2500 | 1.69 | 1.6 |  |  | 2.93e-16 | 0.029 |
| `pareto_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `pareto_pdf` | guaranteed | 6000 | 1.92 | 1.57 |  |  | 3.34e-16 | 0.033 |
| `pareto_skewness` | guaranteed | 2500 | 2.43 | 1.59 |  |  | 3.20e-16 | 0.032 |
| `pareto_standard_deviation` | guaranteed | 2500 | 1.88 | 1.77 |  |  | 2.85e-16 | 0.029 |
| `pareto_variance` | guaranteed | 2500 | 2.03 | 1.92 |  |  | 3.15e-16 | 0.032 |
| `poisson_ccdf` | guaranteed | 6000 | 9.88 | 5.2 |  |  | 1.32e-15 | 0.13 |
| `poisson_cdf` | guaranteed | 6000 | 6.18 | 4.97 |  |  | 9.59e-16 | 0.096 |
| `poisson_entropy` | guaranteed | 605 | 1.63 | 1.63 |  | 0.707 / 0.707 | 2.59e-16 | 0.016 |
| `poisson_iccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `poisson_icdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `poisson_ilogccdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `poisson_ilogcdf` | guaranteed | 4000 | 0 | 0 |  |  | 0 | 0 |
| `poisson_kurtosis` | guaranteed | 605 | 0.498 | 0.498 |  |  | 1.09e-16 | 0.011 |
| `poisson_logccdf` | guaranteed | 6000 | 8.08 | 6.79 |  |  | 1.14e-15 | 0.11 |
| `poisson_logcdf` | guaranteed | 6000 | 9.88 | 6.18 |  |  | 1.32e-15 | 0.13 |
| `poisson_logpmf` | guaranteed | 6000 | 1.62 | 1.35 |  |  | 2.91e-16 | 0.029 |
| `poisson_mean` | guaranteed | 605 | 0 | 0 |  |  | 0 | 0 |
| `poisson_median` | guaranteed | 605 | 0 | 0 |  |  | 0 | 0 |
| `poisson_mode` | guaranteed | 605 | 0 | 0 |  |  | 0 | 0 |
| `poisson_pmf` | guaranteed | 6000 | 4.33 | 3.86 |  |  | 5.92e-16 | 0.059 |
| `poisson_skewness` | guaranteed | 605 | 1.31 | 1.31 |  |  | 1.54e-16 | 0.015 |
| `poisson_standard_deviation` | guaranteed | 605 | 0.498 | 0.498 |  |  | 1.03e-16 | 0.01 |
| `poisson_variance` | guaranteed | 605 | 0 | 0 |  |  | 0 | 0 |
| `rayleigh_ccdf` | guaranteed | 6000 | 1.21 | 0.981 |  |  | 2.06e-16 | 0.021 |
| `rayleigh_cdf` | guaranteed | 6000 | 2.88 | 2.24 |  |  | 4.34e-16 | 0.043 |
| `rayleigh_entropy` | guaranteed | 2500 | 0.5 | 0.499 |  | 0.454 / 0.447 | 1.01e-16 | 0.01 |
| `rayleigh_iccdf` | guaranteed | 4000 | 1.89 | 1.61 | 1.89 |  | 2.49e-16 | 0.023 |
| `rayleigh_icdf` | guaranteed | 4000 | 2.13 | 1.68 | 2.13 |  | 2.51e-16 | 0.025 |
| `rayleigh_ilogccdf` | guaranteed | 4000 | 1.29 | 1.08 |  |  | 1.88e-16 | 0.019 |
| `rayleigh_ilogcdf` | guaranteed | 4000 | 2.17 | 1.5 | 1.21 |  | 2.82e-16 | 0.018 |
| `rayleigh_kurtosis` | guaranteed | 2500 | 0.169 | 0.169 |  |  | 1.91e-17 | 0.0019 |
| `rayleigh_logccdf` | guaranteed | 6000 | 0.5 | 0.5 |  |  | 1.11e-16 | 0.011 |
| `rayleigh_logcdf` | guaranteed | 6000 | 1.46 | 1.21 |  |  | 2.72e-16 | 0.027 |
| `rayleigh_logpdf` | guaranteed | 6000 | 12.7 | 1.81 |  | 0.985 / 0.881 | 2.24e-15 | 0.022 |
| `rayleigh_mean` | guaranteed | 2500 | 1.15 | 1.13 |  |  | 1.80e-16 | 0.018 |
| `rayleigh_median` | guaranteed | 2500 | 0.71 | 0.7 |  |  | 1.27e-16 | 0.013 |
| `rayleigh_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `rayleigh_pdf` | guaranteed | 6000 | 1.76 | 1.4 |  |  | 2.49e-16 | 0.025 |
| `rayleigh_skewness` | guaranteed | 2500 | 0.193 | 0.193 |  |  | 3.40e-17 | 0.0034 |
| `rayleigh_standard_deviation` | guaranteed | 2500 | 0.817 | 0.807 |  |  | 1.47e-16 | 0.015 |
| `rayleigh_variance` | guaranteed | 2500 | 1.38 | 1.33 |  |  | 1.95e-16 | 0.019 |
| `studentt_ccdf` | guaranteed | 6000 | 9.19 | 6.9 |  |  | 1.47e-15 | 0.15 |
| `studentt_cdf` | guaranteed | 6000 | 10.1 | 7.1 |  |  | 1.51e-15 | 0.15 |
| `studentt_entropy` | guaranteed | 2528 | 53.1 | 36.7 |  | 36.3 / 25.1 | 8.05e-15 | 0.81 |
| `studentt_entropy` | loose | 30 | 0.854 | 0.854 |  | 0.602 / 0.602 | 1.34e-16 | 0.013 |
| `studentt_iccdf` | guaranteed | 3935 | 2.33e+7 | 2.83e+6 | 8.33 |  | 2.85e-9 | 0.1 |
| `studentt_iccdf` | loose | 65 | 1.73 | 1.73 |  |  | 3.22e-16 | 0.032 |
| `studentt_icdf` | guaranteed | 3935 | 2.33e+7 | 2.83e+6 | 8.33 |  | 2.85e-9 | 0.1 |
| `studentt_icdf` | loose | 65 | 1.73 | 1.73 |  |  | 3.22e-16 | 0.032 |
| `studentt_ilogccdf` | guaranteed | 3789 | 26.1 | 13.7 | 4.11 |  | 4.38e-15 | 0.049 |
| `studentt_ilogccdf` | loose | 211 | 8.12 | 8.12 | 2.08 |  | 1.12e-15 | 0.028 |
| `studentt_ilogcdf` | guaranteed | 3789 | 26.1 | 13.7 | 4.11 |  | 4.38e-15 | 0.049 |
| `studentt_ilogcdf` | loose | 211 | 8.12 | 8.12 | 2.08 |  | 1.12e-15 | 0.028 |
| `studentt_kurtosis` | guaranteed | 2528 | 0.5 | 0.5 |  |  | 1.09e-16 | 0.011 |
| `studentt_kurtosis` | loose | 30 | 0.442 | 0.442 |  |  | 7.92e-17 | 0.0079 |
| `studentt_logccdf` | guaranteed | 6000 | 10.8 | 7.1 |  |  | 1.57e-15 | 0.16 |
| `studentt_logcdf` | guaranteed | 6000 | 11.1 | 7.22 |  |  | 1.62e-15 | 0.16 |
| `studentt_logpdf` | guaranteed | 6000 | 1.56 | 1.18 |  | 1.07 / 0.799 | 2.37e-16 | 0.024 |
| `studentt_mean` | guaranteed | 2528 | 0 | 0 |  |  | 0 | 0 |
| `studentt_mean` | loose | 30 | 0 | 0 |  |  | 0 | 0 |
| `studentt_median` | guaranteed | 2528 | 0 | 0 |  |  | 0 | 0 |
| `studentt_median` | loose | 30 | 0 | 0 |  |  | 0 | 0 |
| `studentt_mode` | guaranteed | 2528 | 0 | 0 |  |  | 0 | 0 |
| `studentt_mode` | loose | 30 | 0 | 0 |  |  | 0 | 0 |
| `studentt_pdf` | guaranteed | 6000 | 3.74 | 3.13 |  |  | 7.47e-13 | 0.057 |
| `studentt_skewness` | guaranteed | 2528 | 0 | 0 |  |  | 0 | 0 |
| `studentt_skewness` | loose | 30 | 0 | 0 |  |  | 0 | 0 |
| `studentt_standard_deviation` | guaranteed | 2528 | 0.762 | 0.732 |  |  | 1.64e-16 | 0.016 |
| `studentt_standard_deviation` | loose | 30 | 0.7 | 0.7 |  |  | 1.55e-16 | 0.016 |
| `studentt_variance` | guaranteed | 2528 | 0.5 | 0.5 |  |  | 1.11e-16 | 0.011 |
| `studentt_variance` | loose | 30 | 0.475 | 0.475 |  |  | 1.06e-16 | 0.011 |
| `uniform_ccdf` | guaranteed | 6000 | 1.9 | 1.76 |  |  | 2.35e-16 | 0.024 |
| `uniform_cdf` | guaranteed | 6000 | 1.53 | 1.35 |  |  | 2.25e-16 | 0.023 |
| `uniform_entropy` | guaranteed | 2500 | 8.87e+15 | 8.58e+15 |  | 0.5 / 0.5 | 1 | 0.011 |
| `uniform_iccdf` | guaranteed | 4000 | 2.09 | 1.12 | 0.952 |  | 2.88e-16 | 0.018 |
| `uniform_icdf` | guaranteed | 4000 | 2.02 | 1.07 | 1.33 |  | 3.92e-16 | 0.019 |
| `uniform_ilogccdf` | guaranteed | 4000 | 4.48 | 0.998 | 1.15 |  | 7.26e-16 | 0.017 |
| `uniform_ilogcdf` | guaranteed | 4000 | 187 | 2.38 | 0.736 |  | 2.08e-14 | 0.011 |
| `uniform_kurtosis` | guaranteed | 2500 | 0.2 | 0.2 |  |  | 3.70e-17 | 0.0037 |
| `uniform_logccdf` | guaranteed | 6000 | 2.27 | 1.73 |  |  | 3.01e-16 | 0.03 |
| `uniform_logcdf` | guaranteed | 6000 | 1.71 | 1.33 |  |  | 2.22e-16 | 0.022 |
| `uniform_logpdf` | guaranteed | 6000 | 8.90e+15 | 8.73e+15 |  | 0.632 / 0.5 | 1 | 0.014 |
| `uniform_mean` | guaranteed | 2500 | 0.5 | 0.5 |  |  | 1.11e-16 | 0.011 |
| `uniform_median` | guaranteed | 2500 | 0.5 | 0.5 |  |  | 1.11e-16 | 0.011 |
| `uniform_mode` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `uniform_pdf` | guaranteed | 6000 | 1.37 | 1.22 |  |  | 1.60e-16 | 0.016 |
| `uniform_skewness` | guaranteed | 2500 | 0 | 0 |  |  | 0 | 0 |
| `uniform_standard_deviation` | guaranteed | 2500 | 1.53 | 1.4 |  |  | 2.33e-16 | 0.023 |
| `uniform_variance` | guaranteed | 2500 | 2.4 | 2.26 |  |  | 3.63e-16 | 0.036 |
| `weibull_ccdf` | guaranteed | 6000 | 1.41 | 1.2 |  |  | 2.44e-16 | 0.024 |
| `weibull_cdf` | guaranteed | 6000 | 2.51 | 1.89 |  |  | 4.36e-16 | 0.044 |
| `weibull_entropy` | guaranteed | 2589 | 0.5 | 0.499 |  | 0.494 / 0.485 | 1.10e-16 | 0.011 |
| `weibull_iccdf` | guaranteed | 4000 | 54.9 | 24.5 | 4.28 |  | 9.30e-15 | 0.054 |
| `weibull_icdf` | guaranteed | 4000 | 57.3 | 25.7 | 3.18 |  | 7.15e-15 | 0.038 |
| `weibull_ilogccdf` | guaranteed | 4000 | 3.63 | 3.06 | 3.63 |  | 5.52e-16 | 0.055 |
| `weibull_ilogcdf` | guaranteed | 4000 | 26.2 | 6.4 | 1.96 |  | 5.71e-15 | 0.032 |
| `weibull_kurtosis` | guaranteed | 2589 | 1.03e+6 | 8.06e+5 |  | 6.59 / 6.01 | 2.11e-10 | 0.15 |
| `weibull_logccdf` | guaranteed | 6000 | 1.97 | 1.8 |  |  | 3.51e-16 | 0.035 |
| `weibull_logcdf` | guaranteed | 6000 | 2.12 | 1.2 |  |  | 3.05e-16 | 0.031 |
| `weibull_logpdf` | guaranteed | 6000 | 8.99e+15 | 8.72e+15 |  | 1.3 / 1.02 | 1 | 0.029 |
| `weibull_mean` | guaranteed | 2589 | 8.68 | 8.15 |  |  | 1.07e-15 | 0.11 |
| `weibull_median` | guaranteed | 2589 | 1.86 | 1.65 |  |  | 2.60e-16 | 0.026 |
| `weibull_mode` | guaranteed | 2589 | 1.79 | 1.33 |  |  | 2.43e-16 | 0.024 |
| `weibull_pdf` | guaranteed | 6000 | 2.91 | 2.52 |  |  | 3.82e-16 | 0.038 |
| `weibull_skewness` | guaranteed | 2589 | 32.7 | 21 |  | 1.04 / 0.89 | 5.08e-15 | 0.023 |
| `weibull_standard_deviation` | guaranteed | 2589 | 9.5 | 8.73 |  |  | 1.08e-15 | 0.11 |
| `weibull_variance` | guaranteed | 2589 | 17.3 | 17.1 |  |  | 2.14e-15 | 0.21 |
| `zipf_ccdf` | guaranteed | 6000 | 10.9 | 4.36 |  |  | 1.45e-10 | 0.16 |
| `zipf_cdf` | guaranteed | 6000 | 5.6 | 4.06 |  |  | 6.96e-16 | 0.07 |
| `zipf_entropy` | guaranteed | 600 | 1.81e+14 | 1.81e+14 |  | 2.02 / 2.02 | 0.0261 | 0.045 |
| `zipf_iccdf` | guaranteed | 4000 | 973 | 663 |  |  | 1.62e-13 | 0 |
| `zipf_icdf` | guaranteed | 4000 | 709 | 630 |  |  | 1.26e-13 | 0 |
| `zipf_ilogccdf` | guaranteed | 4000 | 256 | 16 |  |  | 3.13e-14 | 0 |
| `zipf_ilogcdf` | guaranteed | 4000 | 71.7 | 34 |  |  | 1.14e-14 | 0 |
| `zipf_kurtosis` | guaranteed | 600 | 12.7 | 12.7 |  |  | 1.67e-15 | 0.17 |
| `zipf_logccdf` | guaranteed | 6000 | 7.49 | 4.16 |  |  | 8.82e-16 | 0.088 |
| `zipf_logcdf` | guaranteed | 6000 | 10.9 | 4.74 |  |  | 1.45e-10 | 0.16 |
| `zipf_logpmf` | guaranteed | 6000 | 3.2 | 1.85 |  |  | 4.39e-16 | 0.044 |
| `zipf_mean` | guaranteed | 600 | 1.97 | 1.97 |  |  | 3.84e-16 | 0.038 |
| `zipf_median` | guaranteed | 600 | 1.29e+03 | 1.29e+03 |  |  | 2.31e-13 | 0 |
| `zipf_mode` | guaranteed | 600 | 0 | 0 |  |  | 0 | 0 |
| `zipf_pmf` | guaranteed | 6000 | 2.54 | 2.24 |  |  | 1.61e-14 | 0.04 |
| `zipf_skewness` | guaranteed | 600 | 9.29 | 9.29 |  |  | 1.12e-15 | 0.11 |
| `zipf_standard_deviation` | guaranteed | 600 | 1.67 | 1.67 |  |  | 3.41e-16 | 0.034 |
| `zipf_variance` | guaranteed | 600 | 2.96 | 2.96 |  |  | 5.53e-16 | 0.055 |
