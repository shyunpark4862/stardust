# sd.regression 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/regression.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/regression/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다. 문제마다 결과의 필드 하나가 한 줄이고(영역 열), 문서에 적힌 수치 계약이 없어 계약 대비는 없다. 기준선보다 나빠지면 compare가 짚는다.

- ULP: |sd − 참값|을 참값에서의 float64 간격으로 나눈 값. 99.9%는 그 분위수다.
- 상대오차: |sd − 참값|/|참값|.
- 노름 상대오차: 벡터 필드의 ‖sd − 참값‖₂/‖참값‖₂.
- NaN·inf 점: 참값은 유한한데 sd가 NaN이나 inf를 낸 점의 수. 그런 점의 오차는 inf로 센다.

측정: 20261120-151558-deeafe2 (2026-11-20T15:15:58, deeafe2 dirty).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | 최대 상대오차 | 노름 상대오차 | NaN·inf 점 |
|---|---|---:|---:|---:|---:|---:|---:|
| `glm/binomial_cloglog` | params | 3 | 0.182 | 0.182 | 2.83e-17 | 2.18e-17 | 0 |
| `glm/binomial_cloglog` | bse | 3 | 1.11 | 1.11 | 1.59e-16 | 1.08e-16 | 0 |
| `glm/binomial_cloglog` | tvalues | 3 | 1.11 | 1.11 | 2.29e-16 | 9.86e-17 | 0 |
| `glm/binomial_cloglog` | pvalues | 3 | 3.77 | 3.77 | 7.43e-16 | 6.78e-16 | 0 |
| `glm/binomial_cloglog` | deviance | 1 | 0.406 | 0.406 | 7.88e-17 |  | 0 |
| `glm/binomial_cloglog` | pearson_chi2 | 1 | 0.394 | 0.394 | 4.72e-17 |  | 0 |
| `glm/binomial_cloglog` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/binomial_cloglog` | llf | 1 | 0.406 | 0.406 | 7.88e-17 |  | 0 |
| `glm/binomial_cloglog` | null_deviance | 1 | 0.0407 | 0.0407 | 6.98e-18 |  | 0 |
| `glm/binomial_cloglog` | llnull | 1 | 0.0407 | 0.0407 | 6.98e-18 |  | 0 |
| `glm/binomial_cloglog` | aic | 1 | 0.406 | 0.406 | 7.28e-17 |  | 0 |
| `glm/binomial_cloglog` | bic | 1 | 0.526 | 0.526 | 8.73e-17 |  | 0 |
| `glm/binomial_cloglog` | fittedvalues | 60 | 1.03 | 1.03 | 1.18e-16 | 5.10e-17 | 0 |
| `glm/binomial_cloglog` | bse_HC0 | 3 | 1.18 | 1.18 | 2.30e-16 | 1.41e-16 | 0 |
| `glm/binomial_cloglog` | bse_HC1 | 3 | 2.15 | 2.15 | 2.65e-16 | 1.92e-16 | 0 |
| `glm/binomial_cloglog` | bse_HC2 | 3 | 1.27 | 1.27 | 2.40e-16 | 1.26e-16 | 0 |
| `glm/binomial_cloglog` | bse_HC3 | 3 | 1.38 | 1.38 | 1.75e-16 | 1.66e-16 | 0 |
| `glm/binomial_cloglog` | bse_HAC | 3 | 1.85 | 1.85 | 2.33e-16 | 1.87e-16 | 0 |
| `glm/binomial_cloglog` | bse_cluster | 3 | 0.623 | 0.623 | 1.27e-16 | 1.02e-16 | 0 |
| `glm/binomial_cloglog` | pvalues_cluster | 3 | 2.9 | 2.9 | 5.76e-16 | 3.33e-16 | 0 |
| `glm/binomial_cloglog` | resid_pearson | 60 | 1.21 | 1.21 | 2.42e-16 | 1.05e-16 | 0 |
| `glm/binomial_log` | params | 3 | 0.889 | 0.889 | 1.79e-16 | 1.12e-16 | 0 |
| `glm/binomial_log` | bse | 3 | 2.98 | 2.98 | 4.52e-16 | 3.11e-16 | 0 |
| `glm/binomial_log` | tvalues | 3 | 2.65 | 2.65 | 4.17e-16 | 3.95e-16 | 0 |
| `glm/binomial_log` | pvalues | 3 | 65.9 | 65.9 | 1.34e-14 | 2.29e-16 | 0 |
| `glm/binomial_log` | deviance | 1 | 0.0514 | 0.0514 | 9.56e-18 |  | 0 |
| `glm/binomial_log` | pearson_chi2 | 1 | 0.394 | 0.394 | 4.71e-17 |  | 0 |
| `glm/binomial_log` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/binomial_log` | llf | 1 | 0.0514 | 0.0514 | 9.56e-18 |  | 0 |
| `glm/binomial_log` | null_deviance | 1 | 0.169 | 0.169 | 3.00e-17 |  | 0 |
| `glm/binomial_log` | llnull | 1 | 0.169 | 0.169 | 3.00e-17 |  | 0 |
| `glm/binomial_log` | aic | 1 | 0.0514 | 0.0514 | 8.87e-18 |  | 0 |
| `glm/binomial_log` | bic | 1 | 0.0682 | 0.0682 | 1.09e-17 |  | 0 |
| `glm/binomial_log` | fittedvalues | 60 | 1.74 | 1.74 | 2.77e-16 | 1.38e-16 | 0 |
| `glm/binomial_log` | bse_HC0 | 3 | 6.78 | 6.78 | 7.91e-16 | 6.12e-16 | 0 |
| `glm/binomial_log` | bse_HC1 | 3 | 5.82 | 5.82 | 8.56e-16 | 6.27e-16 | 0 |
| `glm/binomial_log` | bse_HC2 | 3 | 4.67 | 4.67 | 9.14e-16 | 6.14e-16 | 0 |
| `glm/binomial_log` | bse_HC3 | 3 | 4.56 | 4.56 | 9.49e-16 | 6.17e-16 | 0 |
| `glm/binomial_log` | bse_HAC | 3 | 5.22 | 5.22 | 7.28e-16 | 5.22e-16 | 0 |
| `glm/binomial_log` | bse_cluster | 3 | 4.59 | 4.59 | 8.10e-16 | 5.45e-16 | 0 |
| `glm/binomial_log` | pvalues_cluster | 3 | 16.1 | 16.1 | 2.84e-15 | 6.81e-16 | 0 |
| `glm/binomial_log` | resid_pearson | 60 | 1.65 | 1.65 | 2.85e-16 | 1.33e-16 | 0 |
| `glm/binomial_logit` | params | 3 | 2.47 | 2.47 | 5.46e-16 | 7.87e-17 | 0 |
| `glm/binomial_logit` | bse | 3 | 0.434 | 0.434 | 7.89e-17 | 5.50e-17 | 0 |
| `glm/binomial_logit` | tvalues | 3 | 3.6 | 3.6 | 4.75e-16 | 1.61e-16 | 0 |
| `glm/binomial_logit` | pvalues | 3 | 13.8 | 13.8 | 1.84e-15 | 2.06e-16 | 0 |
| `glm/binomial_logit` | deviance | 1 | 0.0857 | 0.0857 | 1.84e-17 |  | 0 |
| `glm/binomial_logit` | pearson_chi2 | 1 | 0.072 | 0.072 | 8.78e-18 |  | 0 |
| `glm/binomial_logit` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/binomial_logit` | llf | 1 | 0.0857 | 0.0857 | 1.84e-17 |  | 0 |
| `glm/binomial_logit` | null_deviance | 1 | 0.116 | 0.116 | 1.99e-17 |  | 0 |
| `glm/binomial_logit` | llnull | 1 | 0.116 | 0.116 | 1.99e-17 |  | 0 |
| `glm/binomial_logit` | aic | 1 | 0.0857 | 0.0857 | 1.68e-17 |  | 0 |
| `glm/binomial_logit` | bic | 1 | 0.0339 | 0.0339 | 6.13e-18 |  | 0 |
| `glm/binomial_logit` | fittedvalues | 60 | 2.64 | 2.64 | 4.95e-16 | 7.93e-17 | 0 |
| `glm/binomial_logit` | bse_HC0 | 3 | 2.4 | 2.4 | 4.33e-16 | 3.24e-16 | 0 |
| `glm/binomial_logit` | bse_HC1 | 3 | 3.02 | 3.02 | 5.25e-16 | 3.85e-16 | 0 |
| `glm/binomial_logit` | bse_HC2 | 3 | 3.53 | 3.53 | 4.19e-16 | 3.55e-16 | 0 |
| `glm/binomial_logit` | bse_HC3 | 3 | 3.47 | 3.47 | 4.01e-16 | 3.77e-16 | 0 |
| `glm/binomial_logit` | bse_HAC | 3 | 2.59 | 2.59 | 5.37e-16 | 3.24e-16 | 0 |
| `glm/binomial_logit` | bse_cluster | 3 | 4.53 | 4.53 | 5.06e-16 | 3.68e-16 | 0 |
| `glm/binomial_logit` | pvalues_cluster | 3 | 10.5 | 10.5 | 1.61e-15 | 4.87e-16 | 0 |
| `glm/binomial_logit` | resid_pearson | 60 | 1.76 | 1.76 | 2.42e-16 | 9.72e-17 | 0 |
| `glm/binomial_probit` | params | 3 | 1.32 | 1.32 | 1.88e-16 | 5.28e-17 | 0 |
| `glm/binomial_probit` | bse | 3 | 1.92 | 1.92 | 2.69e-16 | 1.73e-16 | 0 |
| `glm/binomial_probit` | tvalues | 3 | 1.65 | 1.65 | 2.17e-16 | 2.03e-16 | 0 |
| `glm/binomial_probit` | pvalues | 3 | 18.2 | 18.2 | 2.65e-15 | 9.24e-17 | 0 |
| `glm/binomial_probit` | deviance | 1 | 0.0647 | 0.0647 | 1.41e-17 |  | 0 |
| `glm/binomial_probit` | pearson_chi2 | 1 | 0.257 | 0.257 | 5.64e-17 |  | 0 |
| `glm/binomial_probit` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/binomial_probit` | llf | 1 | 0.0647 | 0.0647 | 1.41e-17 |  | 0 |
| `glm/binomial_probit` | null_deviance | 1 | 0.0671 | 0.0671 | 1.16e-17 |  | 0 |
| `glm/binomial_probit` | llnull | 1 | 0.0671 | 0.0671 | 1.16e-17 |  | 0 |
| `glm/binomial_probit` | aic | 1 | 0.0647 | 0.0647 | 1.29e-17 |  | 0 |
| `glm/binomial_probit` | bic | 1 | 0.0549 | 0.0549 | 1.01e-17 |  | 0 |
| `glm/binomial_probit` | fittedvalues | 60 | 4.01 | 4.01 | 5.44e-16 | 6.96e-17 | 0 |
| `glm/binomial_probit` | bse_HC0 | 3 | 3.55 | 3.55 | 5.46e-16 | 3.65e-16 | 0 |
| `glm/binomial_probit` | bse_HC1 | 3 | 3.67 | 3.67 | 5.51e-16 | 4.22e-16 | 0 |
| `glm/binomial_probit` | bse_HC2 | 3 | 3.83 | 3.83 | 5.75e-16 | 4.17e-16 | 0 |
| `glm/binomial_probit` | bse_HC3 | 3 | 3.06 | 3.06 | 3.77e-16 | 3.11e-16 | 0 |
| `glm/binomial_probit` | bse_HAC | 3 | 3.13 | 3.13 | 4.61e-16 | 3.99e-16 | 0 |
| `glm/binomial_probit` | bse_cluster | 3 | 4.25 | 4.25 | 6.57e-16 | 5.18e-16 | 0 |
| `glm/binomial_probit` | pvalues_cluster | 3 | 19.1 | 19.1 | 3.13e-15 | 8.96e-16 | 0 |
| `glm/binomial_probit` | resid_pearson | 60 | 3.11 | 3.11 | 4.17e-16 | 1.46e-16 | 0 |
| `glm/binomial_trials` | params | 3 | 1.26 | 1.26 | 1.65e-16 | 4.46e-17 | 0 |
| `glm/binomial_trials` | bse | 3 | 1.1 | 1.1 | 1.30e-16 | 8.08e-17 | 0 |
| `glm/binomial_trials` | tvalues | 3 | 1.97 | 1.97 | 2.42e-16 | 1.30e-16 | 0 |
| `glm/binomial_trials` | pvalues | 3 | 34.6 | 34.6 | 6.16e-15 | 2.91e-16 | 0 |
| `glm/binomial_trials` | deviance | 1 | 1.61 | 1.61 | 2.17e-16 |  | 0 |
| `glm/binomial_trials` | pearson_chi2 | 1 | 0.36 | 0.36 | 5.51e-17 |  | 0 |
| `glm/binomial_trials` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/binomial_trials` | llf | 1 | 0.888 | 0.888 | 1.59e-16 |  | 0 |
| `glm/binomial_trials` | null_deviance | 1 | 0.0986 | 0.0986 | 1.21e-17 |  | 0 |
| `glm/binomial_trials` | llnull | 1 | 0.0904 | 0.0904 | 1.16e-17 |  | 0 |
| `glm/binomial_trials` | aic | 1 | 0.888 | 0.888 | 1.54e-16 |  | 0 |
| `glm/binomial_trials` | bic | 1 | 0.947 | 0.947 | 1.58e-16 |  | 0 |
| `glm/binomial_trials` | fittedvalues | 60 | 1.37 | 1.37 | 2.02e-16 | 7.44e-17 | 0 |
| `glm/binomial_trials` | bse_HC0 | 3 | 2.85 | 2.85 | 3.48e-16 | 2.64e-16 | 0 |
| `glm/binomial_trials` | bse_HC1 | 3 | 2.78 | 2.78 | 3.30e-16 | 2.35e-16 | 0 |
| `glm/binomial_trials` | bse_HC2 | 3 | 3.63 | 3.63 | 4.86e-16 | 2.97e-16 | 0 |
| `glm/binomial_trials` | bse_HC3 | 3 | 2.69 | 2.69 | 3.43e-16 | 2.74e-16 | 0 |
| `glm/binomial_trials` | bse_HAC | 3 | 3.67 | 3.67 | 4.91e-16 | 3.30e-16 | 0 |
| `glm/binomial_trials` | bse_cluster | 3 | 5.09 | 5.09 | 8.26e-16 | 7.36e-16 | 0 |
| `glm/binomial_trials` | pvalues_cluster | 3 | 27.8 | 27.8 | 3.97e-15 | 7.63e-16 | 0 |
| `glm/binomial_trials` | resid_pearson | 60 | 25.1 | 25.1 | 3.08e-15 | 3.18e-16 | 0 |
| `glm/gamma_dev` | params | 3 | 0.562 | 0.562 | 1.21e-16 | 1.03e-16 | 0 |
| `glm/gamma_dev` | bse | 3 | 1.13 | 1.13 | 1.91e-16 | 1.16e-16 | 0 |
| `glm/gamma_dev` | tvalues | 3 | 1.67 | 1.67 | 2.04e-16 | 1.69e-16 | 0 |
| `glm/gamma_dev` | pvalues | 3 | 22 | 22 | 4.26e-15 | 2.41e-15 | 0 |
| `glm/gamma_dev` | deviance | 1 | 0.212 | 0.212 | 3.44e-17 |  | 0 |
| `glm/gamma_dev` | pearson_chi2 | 1 | 1.3 | 1.3 | 2.18e-16 |  | 0 |
| `glm/gamma_dev` | scale | 1 | 0.203 | 0.203 | 2.94e-17 |  | 0 |
| `glm/gamma_dev` | llf | 1 | 0.549 | 0.549 | 9.88e-17 |  | 0 |
| `glm/gamma_dev` | null_deviance | 1 | 0.281 | 0.281 | 3.23e-17 |  | 0 |
| `glm/gamma_dev` | llnull | 1 | 0.618 | 0.618 | 9.69e-17 |  | 0 |
| `glm/gamma_dev` | aic | 1 | 0.549 | 0.549 | 9.52e-17 |  | 0 |
| `glm/gamma_dev` | bic | 1 | 0.609 | 0.609 | 1.02e-16 |  | 0 |
| `glm/gamma_dev` | fittedvalues | 60 | 1.22 | 1.22 | 2.08e-16 | 1.11e-16 | 0 |
| `glm/gamma_dev` | bse_HC0 | 3 | 2.07 | 2.07 | 4.32e-16 | 2.69e-16 | 0 |
| `glm/gamma_dev` | bse_HC1 | 3 | 2.89 | 2.89 | 5.88e-16 | 3.24e-16 | 0 |
| `glm/gamma_dev` | bse_HC2 | 3 | 1.36 | 1.36 | 2.73e-16 | 1.84e-16 | 0 |
| `glm/gamma_dev` | bse_HC3 | 3 | 3.25 | 3.25 | 6.24e-16 | 3.53e-16 | 0 |
| `glm/gamma_dev` | bse_HAC | 3 | 1.89 | 1.89 | 3.56e-16 | 2.38e-16 | 0 |
| `glm/gamma_dev` | bse_cluster | 3 | 2.72 | 2.72 | 4.03e-16 | 2.98e-16 | 0 |
| `glm/gamma_dev` | pvalues_cluster | 3 | 13.5 | 13.5 | 2.08e-15 | 1.15e-15 | 0 |
| `glm/gamma_dev` | resid_pearson | 60 | 25.3 | 25.3 | 3.24e-15 | 1.60e-16 | 0 |
| `glm/gamma_identity` | params | 3 | 0.696 | 0.696 | 1.25e-16 | 4.29e-17 | 0 |
| `glm/gamma_identity` | bse | 3 | 0.938 | 0.938 | 1.36e-16 | 1.05e-16 | 0 |
| `glm/gamma_identity` | tvalues | 3 | 1.75 | 1.75 | 2.20e-16 | 6.92e-17 | 0 |
| `glm/gamma_identity` | pvalues | 3 | 13 | 13 | 2.84e-15 | 1.43e-16 | 0 |
| `glm/gamma_identity` | deviance | 1 | 0.0118 | 0.0118 | 1.58e-18 |  | 0 |
| `glm/gamma_identity` | pearson_chi2 | 1 | 1.05 | 1.05 | 1.55e-16 |  | 0 |
| `glm/gamma_identity` | scale | 1 | 1.11 | 1.11 | 1.45e-16 |  | 0 |
| `glm/gamma_identity` | llf | 1 | 0.409 | 0.409 | 8.53e-17 |  | 0 |
| `glm/gamma_identity` | null_deviance | 1 | 0.206 | 0.206 | 2.68e-17 |  | 0 |
| `glm/gamma_identity` | llnull | 1 | 0.255 | 0.255 | 5.28e-17 |  | 0 |
| `glm/gamma_identity` | aic | 1 | 0.409 | 0.409 | 8.35e-17 |  | 0 |
| `glm/gamma_identity` | bic | 1 | 0.939 | 0.939 | 1.87e-16 |  | 0 |
| `glm/gamma_identity` | fittedvalues | 60 | 0.902 | 0.902 | 1.41e-16 | 6.02e-17 | 0 |
| `glm/gamma_identity` | bse_HC0 | 3 | 2.38 | 2.38 | 2.99e-16 | 1.64e-16 | 0 |
| `glm/gamma_identity` | bse_HC1 | 3 | 2.38 | 2.38 | 2.91e-16 | 1.61e-16 | 0 |
| `glm/gamma_identity` | bse_HC2 | 3 | 1.2 | 1.2 | 1.52e-16 | 1.27e-16 | 0 |
| `glm/gamma_identity` | bse_HC3 | 3 | 1.73 | 1.73 | 3.07e-16 | 2.28e-16 | 0 |
| `glm/gamma_identity` | bse_HAC | 3 | 2.23 | 2.23 | 3.05e-16 | 1.55e-16 | 0 |
| `glm/gamma_identity` | bse_cluster | 3 | 1.32 | 1.32 | 2.68e-16 | 2.15e-16 | 0 |
| `glm/gamma_identity` | pvalues_cluster | 3 | 6.07 | 6.07 | 1.27e-15 | 2.80e-16 | 0 |
| `glm/gamma_identity` | resid_pearson | 60 | 80.7 | 80.7 | 9.83e-15 | 1.21e-16 | 0 |
| `glm/gamma_inverse` | params | 3 | 2.42 | 2.42 | 3.78e-16 | 8.06e-17 | 0 |
| `glm/gamma_inverse` | bse | 3 | 0.907 | 0.907 | 1.48e-16 | 1.10e-16 | 0 |
| `glm/gamma_inverse` | tvalues | 3 | 2.3 | 2.3 | 4.37e-16 | 2.33e-16 | 0 |
| `glm/gamma_inverse` | pvalues | 3 | 61.1 | 61.1 | 9.64e-15 | 4.29e-16 | 0 |
| `glm/gamma_inverse` | deviance | 1 | 0.319 | 0.319 | 4.72e-17 |  | 0 |
| `glm/gamma_inverse` | pearson_chi2 | 1 | 0.396 | 0.396 | 7.17e-17 |  | 0 |
| `glm/gamma_inverse` | scale | 1 | 0.76 | 0.76 | 1.23e-16 |  | 0 |
| `glm/gamma_inverse` | llf | 1 | 0.0839 | 0.0839 | 1.41e-17 |  | 0 |
| `glm/gamma_inverse` | null_deviance | 1 | 0.243 | 0.243 | 3.52e-17 |  | 0 |
| `glm/gamma_inverse` | llnull | 1 | 0.548 | 0.548 | 9.07e-17 |  | 0 |
| `glm/gamma_inverse` | aic | 1 | 0.0839 | 0.0839 | 1.32e-17 |  | 0 |
| `glm/gamma_inverse` | bic | 1 | 0.0357 | 0.0357 | 5.24e-18 |  | 0 |
| `glm/gamma_inverse` | fittedvalues | 60 | 1.94 | 1.94 | 2.49e-16 | 1.20e-16 | 0 |
| `glm/gamma_inverse` | bse_HC0 | 3 | 1.93 | 1.93 | 3.30e-16 | 2.72e-16 | 0 |
| `glm/gamma_inverse` | bse_HC1 | 3 | 2.36 | 2.36 | 3.93e-16 | 3.08e-16 | 0 |
| `glm/gamma_inverse` | bse_HC2 | 3 | 2.7 | 2.7 | 4.51e-16 | 3.36e-16 | 0 |
| `glm/gamma_inverse` | bse_HC3 | 3 | 2.17 | 2.17 | 3.53e-16 | 2.73e-16 | 0 |
| `glm/gamma_inverse` | bse_HAC | 3 | 2.55 | 2.55 | 3.68e-16 | 2.81e-16 | 0 |
| `glm/gamma_inverse` | bse_cluster | 3 | 1.76 | 1.76 | 2.80e-16 | 2.38e-16 | 0 |
| `glm/gamma_inverse` | pvalues_cluster | 3 | 21.7 | 21.7 | 2.70e-15 | 4.43e-16 | 0 |
| `glm/gamma_inverse` | resid_pearson | 60 | 380 | 380 | 4.39e-14 | 2.36e-16 | 0 |
| `glm/gamma_log` | params | 3 | 0.254 | 0.254 | 4.61e-17 | 3.86e-17 | 0 |
| `glm/gamma_log` | bse | 3 | 0.738 | 0.738 | 1.38e-16 | 1.15e-16 | 0 |
| `glm/gamma_log` | tvalues | 3 | 1.43 | 1.43 | 1.63e-16 | 1.42e-16 | 0 |
| `glm/gamma_log` | pvalues | 3 | 25.6 | 25.6 | 4.79e-15 | 4.34e-16 | 0 |
| `glm/gamma_log` | deviance | 1 | 0.0287 | 0.0287 | 4.46e-18 |  | 0 |
| `glm/gamma_log` | pearson_chi2 | 1 | 0.107 | 0.107 | 1.82e-17 |  | 0 |
| `glm/gamma_log` | scale | 1 | 0.506 | 0.506 | 7.67e-17 |  | 0 |
| `glm/gamma_log` | llf | 1 | 0.443 | 0.443 | 7.55e-17 |  | 0 |
| `glm/gamma_log` | null_deviance | 1 | 0.21 | 0.21 | 4.57e-17 |  | 0 |
| `glm/gamma_log` | llnull | 1 | 0.466 | 0.466 | 6.85e-17 |  | 0 |
| `glm/gamma_log` | aic | 1 | 0.443 | 0.443 | 7.29e-17 |  | 0 |
| `glm/gamma_log` | bic | 1 | 0.383 | 0.383 | 6.08e-17 |  | 0 |
| `glm/gamma_log` | fittedvalues | 60 | 1.13 | 1.13 | 1.72e-16 | 7.10e-17 | 0 |
| `glm/gamma_log` | bse_HC0 | 3 | 4.3 | 4.3 | 4.91e-16 | 3.23e-16 | 0 |
| `glm/gamma_log` | bse_HC1 | 3 | 4.32 | 4.32 | 4.81e-16 | 3.01e-16 | 0 |
| `glm/gamma_log` | bse_HC2 | 3 | 1.91 | 1.91 | 3.23e-16 | 3.02e-16 | 0 |
| `glm/gamma_log` | bse_HC3 | 3 | 2.43 | 2.43 | 5.24e-16 | 3.25e-16 | 0 |
| `glm/gamma_log` | bse_HAC | 3 | 4.79 | 4.79 | 6.52e-16 | 3.39e-16 | 0 |
| `glm/gamma_log` | bse_cluster | 3 | 2.27 | 2.27 | 4.90e-16 | 2.44e-16 | 0 |
| `glm/gamma_log` | pvalues_cluster | 3 | 10.1 | 10.1 | 2.10e-15 | 8.89e-17 | 0 |
| `glm/gamma_log` | resid_pearson | 60 | 204 | 204 | 3.48e-14 | 1.38e-16 | 0 |
| `glm/gamma_weights` | params | 3 | 0.283 | 0.283 | 4.66e-17 | 3.82e-17 | 0 |
| `glm/gamma_weights` | bse | 3 | 1.23 | 1.23 | 1.95e-16 | 1.11e-16 | 0 |
| `glm/gamma_weights` | tvalues | 3 | 0.961 | 0.961 | 1.86e-16 | 1.29e-16 | 0 |
| `glm/gamma_weights` | pvalues | 3 | 50.7 | 50.7 | 1.00e-14 | 1.30e-15 | 0 |
| `glm/gamma_weights` | deviance | 1 | 0.835 | 0.835 | 1.38e-16 |  | 0 |
| `glm/gamma_weights` | pearson_chi2 | 1 | 0.091 | 0.091 | 1.95e-17 |  | 0 |
| `glm/gamma_weights` | scale | 1 | 0.228 | 0.228 | 4.45e-17 |  | 0 |
| `glm/gamma_weights` | llf | 1 | 0.518 | 0.518 | 1.08e-16 |  | 0 |
| `glm/gamma_weights` | null_deviance | 1 | 0.397 | 0.397 | 8.55e-17 |  | 0 |
| `glm/gamma_weights` | llnull | 1 | 0.925 | 0.925 | 1.49e-16 |  | 0 |
| `glm/gamma_weights` | aic | 1 | 0.518 | 0.518 | 1.06e-16 |  | 0 |
| `glm/gamma_weights` | bic | 1 | 0.0862 | 0.0862 | 1.71e-17 |  | 0 |
| `glm/gamma_weights` | fittedvalues | 60 | 0.631 | 0.631 | 1.04e-16 | 5.45e-17 | 0 |
| `glm/gamma_weights` | bse_HC0 | 3 | 4.56 | 4.56 | 6.77e-16 | 4.60e-16 | 0 |
| `glm/gamma_weights` | bse_HC1 | 3 | 4.26 | 4.26 | 6.24e-16 | 4.54e-16 | 0 |
| `glm/gamma_weights` | bse_HC2 | 3 | 4.39 | 4.39 | 6.39e-16 | 4.34e-16 | 0 |
| `glm/gamma_weights` | bse_HC3 | 3 | 4.77 | 4.77 | 6.80e-16 | 4.68e-16 | 0 |
| `glm/gamma_weights` | bse_HAC | 3 | 4.26 | 4.26 | 6.78e-16 | 4.49e-16 | 0 |
| `glm/gamma_weights` | bse_cluster | 3 | 3.7 | 3.7 | 6.30e-16 | 4.81e-16 | 0 |
| `glm/gamma_weights` | pvalues_cluster | 3 | 14 | 14 | 2.64e-15 | 2.08e-15 | 0 |
| `glm/gamma_weights` | resid_pearson | 60 | 61.7 | 61.7 | 1.22e-14 | 1.36e-16 | 0 |
| `glm/gaussian_identity` | params | 3 | 0.275 | 0.275 | 5.06e-17 | 4.95e-17 | 0 |
| `glm/gaussian_identity` | bse | 3 | 0.973 | 0.973 | 1.46e-16 | 9.43e-17 | 0 |
| `glm/gaussian_identity` | tvalues | 3 | 1.33 | 1.33 | 2.25e-16 | 5.05e-17 | 0 |
| `glm/gaussian_identity` | pvalues | 3 | 53.8 | 53.8 | 8.17e-15 | 4.28e-16 | 0 |
| `glm/gaussian_identity` | deviance | 1 | 0.494 | 0.494 | 6.61e-17 |  | 0 |
| `glm/gaussian_identity` | pearson_chi2 | 1 | 0.494 | 0.494 | 6.61e-17 |  | 0 |
| `glm/gaussian_identity` | scale | 1 | 0.537 | 0.537 | 6.40e-17 |  | 0 |
| `glm/gaussian_identity` | llf | 1 | 1.3 | 1.3 | 2.42e-16 |  | 0 |
| `glm/gaussian_identity` | null_deviance | 1 | 0.273 | 0.273 | 4.30e-17 |  | 0 |
| `glm/gaussian_identity` | llnull | 1 | 0.277 | 0.277 | 3.54e-17 |  | 0 |
| `glm/gaussian_identity` | aic | 1 | 1.3 | 1.3 | 2.09e-16 |  | 0 |
| `glm/gaussian_identity` | bic | 1 | 1.54 | 1.54 | 2.17e-16 |  | 0 |
| `glm/gaussian_identity` | fittedvalues | 60 | 0.904 | 0.904 | 1.49e-16 | 7.94e-17 | 0 |
| `glm/gaussian_identity` | bse_HC0 | 3 | 3.59 | 3.59 | 5.10e-16 | 3.34e-16 | 0 |
| `glm/gaussian_identity` | bse_HC1 | 3 | 2.99 | 2.99 | 4.14e-16 | 3.00e-16 | 0 |
| `glm/gaussian_identity` | bse_HC2 | 3 | 3.27 | 3.27 | 4.46e-16 | 3.19e-16 | 0 |
| `glm/gaussian_identity` | bse_HC3 | 3 | 3.03 | 3.03 | 3.96e-16 | 2.80e-16 | 0 |
| `glm/gaussian_identity` | bse_HAC | 3 | 2.48 | 2.48 | 4.18e-16 | 3.11e-16 | 0 |
| `glm/gaussian_identity` | bse_cluster | 3 | 2.99 | 2.99 | 4.49e-16 | 3.74e-16 | 0 |
| `glm/gaussian_identity` | pvalues_cluster | 3 | 13.7 | 13.7 | 2.32e-15 | 2.23e-15 | 0 |
| `glm/gaussian_identity` | resid_pearson | 60 | 415 | 415 | 4.82e-14 | 5.19e-16 | 0 |
| `glm/gaussian_log` | params | 3 | 0.288 | 0.288 | 5.89e-17 | 5.36e-17 | 0 |
| `glm/gaussian_log` | bse | 3 | 0.884 | 0.884 | 1.40e-16 | 1.04e-16 | 0 |
| `glm/gaussian_log` | tvalues | 3 | 1.03 | 1.03 | 1.48e-16 | 6.29e-17 | 0 |
| `glm/gaussian_log` | pvalues | 3 | 14.3 | 14.3 | 2.47e-15 | 1.20e-15 | 0 |
| `glm/gaussian_log` | deviance | 1 | 0.369 | 0.369 | 4.68e-17 |  | 0 |
| `glm/gaussian_log` | pearson_chi2 | 1 | 0.369 | 0.369 | 4.68e-17 |  | 0 |
| `glm/gaussian_log` | scale | 1 | 0.906 | 0.906 | 1.02e-16 |  | 0 |
| `glm/gaussian_log` | llf | 1 | 0.196 | 0.196 | 3.36e-17 |  | 0 |
| `glm/gaussian_log` | null_deviance | 1 | 0.106 | 0.106 | 2.30e-17 |  | 0 |
| `glm/gaussian_log` | llnull | 1 | 0.496 | 0.496 | 8.88e-17 |  | 0 |
| `glm/gaussian_log` | aic | 1 | 0.196 | 0.196 | 3.13e-17 |  | 0 |
| `glm/gaussian_log` | bic | 1 | 0.316 | 0.316 | 4.71e-17 |  | 0 |
| `glm/gaussian_log` | fittedvalues | 60 | 1.05 | 1.05 | 1.75e-16 | 7.67e-17 | 0 |
| `glm/gaussian_log` | bse_HC0 | 3 | 2.39 | 2.39 | 2.84e-16 | 1.42e-16 | 0 |
| `glm/gaussian_log` | bse_HC1 | 3 | 2.09 | 2.09 | 2.43e-16 | 1.42e-16 | 0 |
| `glm/gaussian_log` | bse_HC2 | 3 | 1.27 | 1.27 | 1.46e-16 | 7.43e-17 | 0 |
| `glm/gaussian_log` | bse_HC3 | 3 | 0.897 | 0.897 | 1.29e-16 | 9.33e-17 | 0 |
| `glm/gaussian_log` | bse_HAC | 3 | 0.741 | 0.741 | 1.15e-16 | 1.03e-16 | 0 |
| `glm/gaussian_log` | bse_cluster | 3 | 2.02 | 2.02 | 2.99e-16 | 2.07e-16 | 0 |
| `glm/gaussian_log` | pvalues_cluster | 3 | 8.51 | 8.51 | 1.31e-15 | 1.30e-15 | 0 |
| `glm/gaussian_log` | resid_pearson | 60 | 317 | 317 | 3.59e-14 | 3.00e-16 | 0 |
| `glm/logit_noint` | params | 2 | 0.545 | 0.545 | 9.06e-17 | 8.81e-17 | 0 |
| `glm/logit_noint` | bse | 2 | 1.38 | 1.38 | 1.76e-16 | 1.26e-16 | 0 |
| `glm/logit_noint` | tvalues | 2 | 0.871 | 0.871 | 1.25e-16 | 1.23e-16 | 0 |
| `glm/logit_noint` | pvalues | 2 | 7.53 | 7.53 | 1.07e-15 | 9.52e-16 | 0 |
| `glm/logit_noint` | deviance | 1 | 0.0988 | 0.0988 | 1.20e-17 |  | 0 |
| `glm/logit_noint` | pearson_chi2 | 1 | 0.724 | 0.724 | 8.59e-17 |  | 0 |
| `glm/logit_noint` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/logit_noint` | llf | 1 | 0.0988 | 0.0988 | 1.20e-17 |  | 0 |
| `glm/logit_noint` | null_deviance | 1 | 0.258 | 0.258 | 4.41e-17 |  | 0 |
| `glm/logit_noint` | llnull | 1 | 0.258 | 0.258 | 4.41e-17 |  | 0 |
| `glm/logit_noint` | aic | 1 | 0.0988 | 0.0988 | 1.12e-17 |  | 0 |
| `glm/logit_noint` | bic | 1 | 0.0376 | 0.0376 | 8.00e-18 |  | 0 |
| `glm/logit_noint` | fittedvalues | 60 | 2.72 | 2.72 | 4.18e-16 | 7.60e-17 | 0 |
| `glm/logit_noint` | bse_HC0 | 2 | 2.77 | 2.77 | 5.12e-16 | 4.09e-16 | 0 |
| `glm/logit_noint` | bse_HC1 | 2 | 2.18 | 2.18 | 3.97e-16 | 3.16e-16 | 0 |
| `glm/logit_noint` | bse_HC2 | 2 | 3.4 | 3.4 | 6.02e-16 | 4.87e-16 | 0 |
| `glm/logit_noint` | bse_HC3 | 2 | 2.29 | 2.29 | 3.88e-16 | 3.12e-16 | 0 |
| `glm/logit_noint` | bse_HAC | 2 | 2.78 | 2.78 | 4.53e-16 | 3.89e-16 | 0 |
| `glm/logit_noint` | bse_cluster | 2 | 2.86 | 2.86 | 5.52e-16 | 4.13e-16 | 0 |
| `glm/logit_noint` | pvalues_cluster | 2 | 17.6 | 17.6 | 2.04e-15 | 1.92e-15 | 0 |
| `glm/logit_noint` | resid_pearson | 60 | 2.21 | 2.21 | 4.07e-16 | 1.07e-16 | 0 |
| `glm/poisson_exposure` | params | 3 | 0.61 | 0.61 | 7.32e-17 | 6.06e-17 | 0 |
| `glm/poisson_exposure` | bse | 3 | 1.06 | 1.06 | 1.83e-16 | 1.08e-16 | 0 |
| `glm/poisson_exposure` | tvalues | 3 | 0.814 | 0.814 | 1.11e-16 | 7.98e-17 | 0 |
| `glm/poisson_exposure` | pvalues | 3 | 12.1 | 12.1 | 1.88e-15 | 9.65e-16 | 0 |
| `glm/poisson_exposure` | deviance | 1 | 1.52 | 1.52 | 1.70e-16 |  | 0 |
| `glm/poisson_exposure` | pearson_chi2 | 1 | 0.0738 | 0.0738 | 8.72e-18 |  | 0 |
| `glm/poisson_exposure` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/poisson_exposure` | llf | 1 | 0.0101 | 0.0101 | 1.30e-18 |  | 0 |
| `glm/poisson_exposure` | null_deviance | 1 | 0.235 | 0.235 | 3.45e-17 |  | 0 |
| `glm/poisson_exposure` | llnull | 1 | 0.238 | 0.238 | 2.66e-17 |  | 0 |
| `glm/poisson_exposure` | aic | 1 | 0.0101 | 0.0101 | 1.26e-18 |  | 0 |
| `glm/poisson_exposure` | bic | 1 | 0.0497 | 0.0497 | 6.06e-18 |  | 0 |
| `glm/poisson_exposure` | fittedvalues | 60 | 1.8 | 1.8 | 2.31e-16 | 7.86e-17 | 0 |
| `glm/poisson_exposure` | bse_HC0 | 3 | 3.1 | 3.1 | 4.48e-16 | 3.08e-16 | 0 |
| `glm/poisson_exposure` | bse_HC1 | 3 | 3.09 | 3.09 | 4.36e-16 | 3.19e-16 | 0 |
| `glm/poisson_exposure` | bse_HC2 | 3 | 1.95 | 1.95 | 2.69e-16 | 2.20e-16 | 0 |
| `glm/poisson_exposure` | bse_HC3 | 3 | 2.16 | 2.16 | 3.00e-16 | 2.25e-16 | 0 |
| `glm/poisson_exposure` | bse_HAC | 3 | 2.24 | 2.24 | 3.96e-16 | 2.41e-16 | 0 |
| `glm/poisson_exposure` | bse_cluster | 3 | 3.37 | 3.37 | 4.13e-16 | 2.97e-16 | 0 |
| `glm/poisson_exposure` | pvalues_cluster | 3 | 7.04 | 7.04 | 1.29e-15 | 4.04e-16 | 0 |
| `glm/poisson_exposure` | resid_pearson | 60 | 8.86 | 8.86 | 1.11e-15 | 1.93e-16 | 0 |
| `glm/poisson_identity` | params | 3 | 0.468 | 0.468 | 1.02e-16 | 9.91e-17 | 0 |
| `glm/poisson_identity` | bse | 3 | 1.28 | 1.28 | 2.66e-16 | 2.27e-16 | 0 |
| `glm/poisson_identity` | tvalues | 3 | 1.52 | 1.52 | 2.01e-16 | 1.27e-16 | 0 |
| `glm/poisson_identity` | pvalues | 3 | 197 | 197 | 3.01e-14 | 6.96e-16 | 0 |
| `glm/poisson_identity` | deviance | 1 | 1.28 | 1.28 | 1.55e-16 |  | 0 |
| `glm/poisson_identity` | pearson_chi2 | 1 | 0.342 | 0.342 | 4.09e-17 |  | 0 |
| `glm/poisson_identity` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/poisson_identity` | llf | 1 | 0.794 | 0.794 | 9.11e-17 |  | 0 |
| `glm/poisson_identity` | null_deviance | 1 | 0.776 | 0.776 | 1.52e-16 |  | 0 |
| `glm/poisson_identity` | llnull | 1 | 0.208 | 0.208 | 4.52e-17 |  | 0 |
| `glm/poisson_identity` | aic | 1 | 0.794 | 0.794 | 8.90e-17 |  | 0 |
| `glm/poisson_identity` | bic | 1 | 0.367 | 0.367 | 8.03e-17 |  | 0 |
| `glm/poisson_identity` | fittedvalues | 60 | 1.38 | 1.38 | 2.12e-16 | 1.14e-16 | 0 |
| `glm/poisson_identity` | bse_HC0 | 3 | 2.42 | 2.42 | 4.79e-16 | 4.56e-16 | 0 |
| `glm/poisson_identity` | bse_HC1 | 3 | 3.18 | 3.18 | 6.50e-16 | 4.82e-16 | 0 |
| `glm/poisson_identity` | bse_HC2 | 3 | 2.75 | 2.75 | 5.43e-16 | 4.94e-16 | 0 |
| `glm/poisson_identity` | bse_HC3 | 3 | 4.18 | 4.18 | 6.79e-16 | 6.07e-16 | 0 |
| `glm/poisson_identity` | bse_HAC | 3 | 4.65 | 4.65 | 7.02e-16 | 5.93e-16 | 0 |
| `glm/poisson_identity` | bse_cluster | 3 | 4.68 | 4.68 | 7.43e-16 | 6.02e-16 | 0 |
| `glm/poisson_identity` | pvalues_cluster | 3 | 19.8 | 19.8 | 3.15e-15 | 8.77e-16 | 0 |
| `glm/poisson_identity` | resid_pearson | 60 | 65.6 | 65.6 | 1.16e-14 | 2.53e-16 | 0 |
| `glm/poisson_log` | params | 3 | 0.856 | 0.856 | 1.62e-16 | 8.44e-17 | 0 |
| `glm/poisson_log` | bse | 3 | 1.11 | 1.11 | 1.39e-16 | 1.17e-16 | 0 |
| `glm/poisson_log` | tvalues | 3 | 2.16 | 2.16 | 2.59e-16 | 1.83e-16 | 0 |
| `glm/poisson_log` | pvalues | 3 | 30.3 | 30.3 | 3.68e-15 | 8.41e-16 | 0 |
| `glm/poisson_log` | deviance | 1 | 0.0586 | 0.0586 | 1.20e-17 |  | 0 |
| `glm/poisson_log` | pearson_chi2 | 1 | 0.187 | 0.187 | 4.07e-17 |  | 0 |
| `glm/poisson_log` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/poisson_log` | llf | 1 | 0.483 | 0.483 | 7.37e-17 |  | 0 |
| `glm/poisson_log` | null_deviance | 1 | 0.246 | 0.246 | 3.68e-17 |  | 0 |
| `glm/poisson_log` | llnull | 1 | 0.11 | 0.11 | 1.48e-17 |  | 0 |
| `glm/poisson_log` | aic | 1 | 0.483 | 0.483 | 7.14e-17 |  | 0 |
| `glm/poisson_log` | bic | 1 | 0.543 | 0.543 | 7.77e-17 |  | 0 |
| `glm/poisson_log` | fittedvalues | 60 | 1.65 | 1.65 | 2.10e-16 | 1.00e-16 | 0 |
| `glm/poisson_log` | bse_HC0 | 3 | 1.43 | 1.43 | 2.75e-16 | 1.89e-16 | 0 |
| `glm/poisson_log` | bse_HC1 | 3 | 1.16 | 1.16 | 1.80e-16 | 1.48e-16 | 0 |
| `glm/poisson_log` | bse_HC2 | 3 | 2.11 | 2.11 | 3.86e-16 | 2.95e-16 | 0 |
| `glm/poisson_log` | bse_HC3 | 3 | 1.99 | 1.99 | 3.46e-16 | 2.60e-16 | 0 |
| `glm/poisson_log` | bse_HAC | 3 | 1.46 | 1.46 | 3.08e-16 | 2.55e-16 | 0 |
| `glm/poisson_log` | bse_cluster | 3 | 2.42 | 2.42 | 2.79e-16 | 1.66e-16 | 0 |
| `glm/poisson_log` | pvalues_cluster | 3 | 5.27 | 5.27 | 1.11e-15 | 1.48e-16 | 0 |
| `glm/poisson_log` | resid_pearson | 60 | 28.8 | 28.8 | 4.76e-15 | 1.40e-16 | 0 |
| `glm/poisson_sqrt` | params | 3 | 1.28 | 1.28 | 1.81e-16 | 7.39e-17 | 0 |
| `glm/poisson_sqrt` | bse | 3 | 0.899 | 0.899 | 1.91e-16 | 1.07e-16 | 0 |
| `glm/poisson_sqrt` | tvalues | 3 | 1.47 | 1.47 | 1.68e-16 | 1.65e-16 | 0 |
| `glm/poisson_sqrt` | pvalues | 3 | 796 | 796 | 1.63e-13 | 1.48e-15 | 0 |
| `glm/poisson_sqrt` | deviance | 1 | 1.02 | 1.02 | 1.61e-16 |  | 0 |
| `glm/poisson_sqrt` | pearson_chi2 | 1 | 0.659 | 0.659 | 1.14e-16 |  | 0 |
| `glm/poisson_sqrt` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `glm/poisson_sqrt` | llf | 1 | 0.623 | 0.623 | 7.49e-17 |  | 0 |
| `glm/poisson_sqrt` | null_deviance | 1 | 0.481 | 0.481 | 7.87e-17 |  | 0 |
| `glm/poisson_sqrt` | llnull | 1 | 0.196 | 0.196 | 4.00e-17 |  | 0 |
| `glm/poisson_sqrt` | aic | 1 | 0.623 | 0.623 | 7.30e-17 |  | 0 |
| `glm/poisson_sqrt` | bic | 1 | 0.563 | 0.563 | 6.44e-17 |  | 0 |
| `glm/poisson_sqrt` | fittedvalues | 60 | 3.22 | 3.22 | 4.88e-16 | 1.77e-16 | 0 |
| `glm/poisson_sqrt` | bse_HC0 | 3 | 3.18 | 3.18 | 4.00e-16 | 2.76e-16 | 0 |
| `glm/poisson_sqrt` | bse_HC1 | 3 | 2.88 | 2.88 | 3.52e-16 | 2.74e-16 | 0 |
| `glm/poisson_sqrt` | bse_HC2 | 3 | 2.93 | 2.93 | 4.83e-16 | 3.59e-16 | 0 |
| `glm/poisson_sqrt` | bse_HC3 | 3 | 3.64 | 3.64 | 4.19e-16 | 3.17e-16 | 0 |
| `glm/poisson_sqrt` | bse_HAC | 3 | 4.39 | 4.39 | 4.93e-16 | 2.94e-16 | 0 |
| `glm/poisson_sqrt` | bse_cluster | 3 | 4.35 | 4.35 | 6.89e-16 | 3.75e-16 | 0 |
| `glm/poisson_sqrt` | pvalues_cluster | 3 | 19.9 | 19.9 | 2.61e-15 | 2.60e-15 | 0 |
| `glm/poisson_sqrt` | resid_pearson | 60 | 84.3 | 84.3 | 1.03e-14 | 5.09e-16 | 0 |
| `linear/nist_Filip` | params | 11 | 0.444 | 0.444 | 9.19e-17 | 1.32e-17 | 0 |
| `linear/nist_Filip` | bse | 11 | 5.63e+8 | 5.63e+8 | 6.60e-8 | 6.25e-8 | 0 |
| `linear/nist_Filip` | tvalues | 11 | 3.62e+8 | 3.62e+8 | 6.60e-8 | 6.45e-8 | 0 |
| `linear/nist_Filip` | pvalues | 11 | 9.65e+9 | 9.65e+9 | 1.22e-6 | 1.12e-6 | 0 |
| `linear/nist_Filip` | conf_int_lower | 11 | 1.74e+8 | 1.74e+8 | 2.02e-8 | 1.79e-8 | 0 |
| `linear/nist_Filip` | conf_int_upper | 11 | 3.47e+8 | 3.47e+8 | 5.24e-8 | 4.21e-8 | 0 |
| `linear/nist_Filip` | fittedvalues | 82 | 1.85e+6 | 1.85e+6 | 2.67e-10 | 6.81e-11 | 0 |
| `linear/nist_Filip` | resid | 82 | 4.48e+9 | 4.48e+9 | 8.70e-7 | 1.86e-8 | 0 |
| `linear/nist_Filip` | hat | 82 | 2.61e+9 | 2.61e+9 | 5.36e-7 | 3.38e-8 | 0 |
| `linear/nist_Filip` | cooks_distance | 82 | 9.60e+9 | 9.60e+9 | 1.75e-6 | 4.12e-8 | 0 |
| `linear/nist_Filip` | ssr | 1 | 2.9 | 2.9 | 3.95e-16 |  | 0 |
| `linear/nist_Filip` | ess | 1 | 0.0692 | 0.0692 | 7.93e-18 |  | 0 |
| `linear/nist_Filip` | centered_tss | 1 | 0.37 | 0.37 | 4.23e-17 |  | 0 |
| `linear/nist_Filip` | rsquared | 1 | 0.297 | 0.297 | 3.31e-17 |  | 0 |
| `linear/nist_Filip` | rsquared_adj | 1 | 0.098 | 0.098 | 1.09e-17 |  | 0 |
| `linear/nist_Filip` | scale | 1 | 2.91 | 2.91 | 4.40e-16 |  | 0 |
| `linear/nist_Filip` | llf | 1 | 1.12 | 1.12 | 1.79e-16 |  | 0 |
| `linear/nist_Filip` | aic | 1 | 1.12 | 1.12 | 1.85e-16 |  | 0 |
| `linear/nist_Filip` | bic | 1 | 1.47 | 1.47 | 2.51e-16 |  | 0 |
| `linear/nist_Filip` | durbin_watson | 1 | 2.41e+7 | 2.41e+7 | 3.50e-9 |  | 0 |
| `linear/nist_Filip` | jarque_bera | 1 | 4.68e+8 | 4.68e+8 | 7.36e-8 |  | 0 |
| `linear/nist_Filip` | jarque_bera_pvalue | 1 | 1.64e+8 | 1.64e+8 | 2.60e-8 |  | 0 |
| `linear/nist_Filip` | resid_studentized_internal | 82 | 5.43e+9 | 5.43e+9 | 8.70e-7 | 2.11e-8 | 0 |
| `linear/nist_Filip` | resid_studentized_external | 82 | 5.39e+9 | 5.39e+9 | 8.70e-7 | 2.13e-8 | 0 |
| `linear/nist_Filip` | bse_HC0 | 11 | 1.00e+9 | 1.00e+9 | 1.23e-7 | 1.21e-7 | 0 |
| `linear/nist_Filip` | bse_HC1 | 11 | 1.05e+9 | 1.05e+9 | 1.23e-7 | 1.21e-7 | 0 |
| `linear/nist_Filip` | bse_HC2 | 11 | 9.20e+8 | 9.20e+8 | 1.06e-7 | 1.02e-7 | 0 |
| `linear/nist_Filip` | bse_HC3 | 11 | 8.02e+8 | 8.02e+8 | 9.57e-8 | 8.61e-8 | 0 |
| `linear/nist_Filip` | bse_HAC | 11 | 1.35e+9 | 1.35e+9 | 1.63e-7 | 1.59e-7 | 0 |
| `linear/nist_Filip` | bp_lm | 1 | 1.56e+7 | 1.56e+7 | 3.31e-9 |  | 0 |
| `linear/nist_Filip` | bp_lm_pvalue | 1 | 9.55e+7 | 9.55e+7 | 1.63e-8 |  | 0 |
| `linear/nist_Filip` | bp_fvalue | 1 | 3.59e+7 | 3.59e+7 | 4.39e-9 |  | 0 |
| `linear/nist_Filip` | bp_f_pvalue | 1 | 1.06e+8 | 1.06e+8 | 1.99e-8 |  | 0 |
| `linear/nist_Filip` | bp_lm_nonstudentized | 1 | 1.91e+5 | 1.91e+5 | 3.53e-11 |  | 0 |
| `linear/nist_Filip` | bp_lm_nonstudentized_pvalue | 1 | 1.17e+6 | 1.17e+6 | 2.15e-10 |  | 0 |
| `linear/nist_Filip` | fvalue | 1 | 2.69 | 2.69 | 5.67e-16 |  | 0 |
| `linear/nist_Filip` | f_pvalue | 1 | 112 | 112 | 2.00e-14 |  | 0 |
| `linear/nist_Longley` | params | 7 | 0.401 | 0.401 | 5.44e-17 | 1.90e-17 | 0 |
| `linear/nist_Longley` | bse | 7 | 1.5e+03 | 1.5e+03 | 1.71e-13 | 2.66e-14 | 0 |
| `linear/nist_Longley` | tvalues | 7 | 1.22e+03 | 1.22e+03 | 1.71e-13 | 1.14e-13 | 0 |
| `linear/nist_Longley` | pvalues | 7 | 7.96e+03 | 7.96e+03 | 1.05e-12 | 3.23e-14 | 0 |
| `linear/nist_Longley` | conf_int_lower | 7 | 693 | 693 | 1.37e-13 | 9.93e-15 | 0 |
| `linear/nist_Longley` | conf_int_upper | 7 | 1.7e+03 | 1.7e+03 | 2.06e-13 | 3.68e-14 | 0 |
| `linear/nist_Longley` | fittedvalues | 16 | 32.9 | 32.9 | 3.92e-15 | 3.62e-15 | 0 |
| `linear/nist_Longley` | resid | 16 | 1.34e+5 | 1.34e+5 | 1.81e-11 | 1.04e-12 | 0 |
| `linear/nist_Longley` | hat | 16 | 2.76e+03 | 2.76e+03 | 3.12e-13 | 1.56e-13 | 0 |
| `linear/nist_Longley` | cooks_distance | 16 | 3.00e+5 | 3.00e+5 | 3.62e-11 | 2.34e-12 | 0 |
| `linear/nist_Longley` | ssr | 1 | 0.00925 | 0.00925 | 1.29e-18 |  | 0 |
| `linear/nist_Longley` | ess | 1 | 0.437 | 0.437 | 7.08e-17 |  | 0 |
| `linear/nist_Longley` | centered_tss | 1 | 0 | 0 | 0 |  | 0 |
| `linear/nist_Longley` | rsquared | 1 | 0.499 | 0.499 | 5.57e-17 |  | 0 |
| `linear/nist_Longley` | rsquared_adj | 1 | 0.501 | 0.501 | 5.61e-17 |  | 0 |
| `linear/nist_Longley` | scale | 1 | 0.453 | 0.453 | 7.09e-17 |  | 0 |
| `linear/nist_Longley` | llf | 1 | 0.451 | 0.451 | 5.85e-17 |  | 0 |
| `linear/nist_Longley` | aic | 1 | 0.451 | 0.451 | 5.49e-17 |  | 0 |
| `linear/nist_Longley` | bic | 1 | 0.287 | 0.287 | 3.42e-17 |  | 0 |
| `linear/nist_Longley` | durbin_watson | 1 | 0.82 | 0.82 | 1.42e-16 |  | 0 |
| `linear/nist_Longley` | jarque_bera | 1 | 41.5 | 41.5 | 6.73e-15 |  | 0 |
| `linear/nist_Longley` | jarque_bera_pvalue | 1 | 14 | 14 | 2.19e-15 |  | 0 |
| `linear/nist_Longley` | resid_studentized_internal | 16 | 1.60e+5 | 1.60e+5 | 1.81e-11 | 1.09e-12 | 0 |
| `linear/nist_Longley` | resid_studentized_external | 16 | 1.51e+5 | 1.51e+5 | 1.81e-11 | 1.21e-12 | 0 |
| `linear/nist_Longley` | bse_HC0 | 7 | 4.89e+03 | 4.89e+03 | 6.79e-13 | 5.09e-13 | 0 |
| `linear/nist_Longley` | bse_HC1 | 7 | 3.37e+03 | 3.37e+03 | 6.79e-13 | 5.09e-13 | 0 |
| `linear/nist_Longley` | bse_HC2 | 7 | 3.83e+03 | 3.83e+03 | 8.07e-13 | 4.87e-13 | 0 |
| `linear/nist_Longley` | bse_HC3 | 7 | 6.55e+03 | 6.55e+03 | 1.02e-12 | 4.32e-13 | 0 |
| `linear/nist_Longley` | bse_HAC | 7 | 9.3e+03 | 9.3e+03 | 1.42e-12 | 1.42e-12 | 0 |
| `linear/nist_Longley` | bp_lm | 1 | 6.94e+03 | 6.94e+03 | 1.23e-12 |  | 0 |
| `linear/nist_Longley` | bp_lm_pvalue | 1 | 3.11e+03 | 3.11e+03 | 3.99e-13 |  | 0 |
| `linear/nist_Longley` | bp_fvalue | 1 | 7.32e+03 | 7.32e+03 | 1.46e-12 |  | 0 |
| `linear/nist_Longley` | bp_f_pvalue | 1 | 1.89e+03 | 1.89e+03 | 2.25e-13 |  | 0 |
| `linear/nist_Longley` | bp_lm_nonstudentized | 1 | 33.8 | 33.8 | 4.17e-15 |  | 0 |
| `linear/nist_Longley` | bp_lm_nonstudentized_pvalue | 1 | 5.42 | 5.42 | 6.42e-16 |  | 0 |
| `linear/nist_Longley` | fvalue | 1 | 0.228 | 0.228 | 3.93e-17 |  | 0 |
| `linear/nist_Longley` | f_pvalue | 1 | 1.31 | 1.31 | 2.73e-16 |  | 0 |
| `linear/nist_NoInt1` | params | 1 | 0.0579 | 0.0579 | 1.24e-17 |  | 0 |
| `linear/nist_NoInt1` | bse | 1 | 0.132 | 0.132 | 2.78e-17 |  | 0 |
| `linear/nist_NoInt1` | tvalues | 1 | 6.22e-61 | 6.22e-61 | 7.05e-77 |  | 0 |
| `linear/nist_NoInt1` | pvalues | 1 | 5.14 | 5.14 | 6.25e-16 |  | 0 |
| `linear/nist_NoInt1` | conf_int_lower | 1 | 0.276 | 0.276 | 6.03e-17 |  | 0 |
| `linear/nist_NoInt1` | conf_int_upper | 1 | 0.161 | 0.161 | 3.38e-17 |  | 0 |
| `linear/nist_NoInt1` | fittedvalues | 11 | 0.157 | 0.157 | 3.07e-17 | 1.91e-17 | 0 |
| `linear/nist_NoInt1` | resid | 11 | 60.2 | 60.2 | 1.01e-14 | 5.18e-16 | 0 |
| `linear/nist_NoInt1` | hat | 11 | 1.81 | 1.81 | 3.24e-16 | 1.72e-16 | 0 |
| `linear/nist_NoInt1` | cooks_distance | 11 | 180 | 180 | 2.07e-14 | 7.16e-16 | 0 |
| `linear/nist_NoInt1` | ssr | 1 | 0.455 | 0.455 | 5.08e-17 |  | 0 |
| `linear/nist_NoInt1` | ess | 1 | 0.273 | 0.273 | 3.96e-17 |  | 0 |
| `linear/nist_NoInt1` | centered_tss | 1 | 0 | 0 | 0 |  | 0 |
| `linear/nist_NoInt1` | rsquared | 1 | 0.0728 | 0.0728 | 8.09e-18 |  | 0 |
| `linear/nist_NoInt1` | rsquared_adj | 1 | 0.28 | 0.28 | 3.11e-17 |  | 0 |
| `linear/nist_NoInt1` | scale | 1 | 0.364 | 0.364 | 5.08e-17 |  | 0 |
| `linear/nist_NoInt1` | llf | 1 | 0.698 | 0.698 | 8.53e-17 |  | 0 |
| `linear/nist_NoInt1` | aic | 1 | 0.698 | 0.698 | 8.25e-17 |  | 0 |
| `linear/nist_NoInt1` | bic | 1 | 0.431 | 0.431 | 5.05e-17 |  | 0 |
| `linear/nist_NoInt1` | durbin_watson | 1 | 0.623 | 0.623 | 9.53e-17 |  | 0 |
| `linear/nist_NoInt1` | jarque_bera | 1 | 4.27 | 4.27 | 6.96e-16 |  | 0 |
| `linear/nist_NoInt1` | jarque_bera_pvalue | 1 | 1.01 | 1.01 | 1.58e-16 |  | 0 |
| `linear/nist_NoInt1` | resid_studentized_internal | 11 | 71.2 | 71.2 | 1.02e-14 | 5.25e-16 | 0 |
| `linear/nist_NoInt1` | resid_studentized_external | 11 | 68.3 | 68.3 | 1.03e-14 | 5.89e-16 | 0 |
| `linear/nist_NoInt1` | bse_HC0 | 1 | 0.208 | 0.208 | 4.61e-17 |  | 0 |
| `linear/nist_NoInt1` | bse_HC1 | 1 | 0.541 | 0.541 | 1.14e-16 |  | 0 |
| `linear/nist_NoInt1` | bse_HC2 | 1 | 0.291 | 0.291 | 6.13e-17 |  | 0 |
| `linear/nist_NoInt1` | bse_HC3 | 1 | 0.24 | 0.24 | 4.82e-17 |  | 0 |
| `linear/nist_NoInt1` | bse_HAC | 1 | 0.753 | 0.753 | 1.10e-16 |  | 0 |
| `linear/nist_NoInt1` | bp_lm | 1 | 98.1 | 98.1 | 2.06e-14 |  | 0 |
| `linear/nist_NoInt1` | bp_lm_pvalue | 1 | 24.2 | 24.2 | 3.75e-15 |  | 0 |
| `linear/nist_NoInt1` | bp_fvalue | 1 | 163 | 163 | 2.08e-14 |  | 0 |
| `linear/nist_NoInt1` | bp_f_pvalue | 1 | 22.4 | 22.4 | 3.32e-15 |  | 0 |
| `linear/nist_NoInt1` | bp_lm_nonstudentized | 1 | 156 | 156 | 2.09e-14 |  | 0 |
| `linear/nist_NoInt1` | bp_lm_nonstudentized_pvalue | 1 | 16.9 | 16.9 | 2.29e-15 |  | 0 |
| `linear/nist_NoInt1` | fvalue | 1 | 1.24e-60 | 1.24e-60 | 1.44e-76 |  | 0 |
| `linear/nist_NoInt1` | f_pvalue | 1 | 5.14 | 5.14 | 6.25e-16 |  | 0 |
| `linear/nist_NoInt2` | params | 1 | 0.182 | 0.182 | 2.78e-17 |  | 0 |
| `linear/nist_NoInt2` | bse | 1 | 0.0861 | 0.0861 | 1.42e-17 |  | 0 |
| `linear/nist_NoInt2` | tvalues | 1 | 0.444 | 0.444 | 9.13e-17 |  | 0 |
| `linear/nist_NoInt2` | pvalues | 1 | 0.786 | 0.786 | 1.02e-16 |  | 0 |
| `linear/nist_NoInt2` | conf_int_lower | 1 | 0.535 | 0.535 | 1.09e-16 |  | 0 |
| `linear/nist_NoInt2` | conf_int_upper | 1 | 0.171 | 0.171 | 2.09e-17 |  | 0 |
| `linear/nist_NoInt2` | fittedvalues | 3 | 0.727 | 0.727 | 8.88e-17 | 7.27e-17 | 0 |
| `linear/nist_NoInt2` | resid | 3 | 5.82 | 5.82 | 8.88e-16 | 3.39e-16 | 0 |
| `linear/nist_NoInt2` | hat | 3 | 3.81 | 3.81 | 5.04e-16 | 4.69e-16 | 0 |
| `linear/nist_NoInt2` | cooks_distance | 3 | 10.9 | 10.9 | 1.51e-15 | 1.41e-15 | 0 |
| `linear/nist_NoInt2` | ssr | 1 | 0.364 | 0.364 | 7.40e-17 |  | 0 |
| `linear/nist_NoInt2` | ess | 1 | 0.0909 | 0.0909 | 1.59e-17 |  | 0 |
| `linear/nist_NoInt2` | centered_tss | 1 | 0 | 0 | 0 |  | 0 |
| `linear/nist_NoInt2` | rsquared | 1 | 0.492 | 0.492 | 5.50e-17 |  | 0 |
| `linear/nist_NoInt2` | rsquared_adj | 1 | 0.262 | 0.262 | 2.93e-17 |  | 0 |
| `linear/nist_NoInt2` | scale | 1 | 0.364 | 0.364 | 7.40e-17 |  | 0 |
| `linear/nist_NoInt2` | llf | 1 | 3.65 | 3.65 | 6.13e-16 |  | 0 |
| `linear/nist_NoInt2` | aic | 1 | 1.82 | 1.82 | 2.44e-16 |  | 0 |
| `linear/nist_NoInt2` | bic | 1 | 1.12 | 1.12 | 2.05e-16 |  | 0 |
| `linear/nist_NoInt2` | durbin_watson | 1 | 0.576 | 0.576 | 1.16e-16 |  | 0 |
| `linear/nist_NoInt2` | jarque_bera | 1 | 2.17 | 2.17 | 3.71e-16 |  | 0 |
| `linear/nist_NoInt2` | jarque_bera_pvalue | 1 | 0.889 | 0.889 | 1.16e-16 |  | 0 |
| `linear/nist_NoInt2` | resid_studentized_internal | 3 | 4.01 | 4.01 | 8.05e-16 | 3.30e-16 | 0 |
| `linear/nist_NoInt2` | resid_studentized_external | 3 | 32.6 | 32.6 | 4.54e-15 | 4.06e-15 | 0 |
| `linear/nist_NoInt2` | bse_HC0 | 1 | 1.8 | 1.8 | 3.35e-16 |  | 0 |
| `linear/nist_NoInt2` | bse_HC1 | 1 | 2.61 | 2.61 | 3.98e-16 |  | 0 |
| `linear/nist_NoInt2` | bse_HC2 | 1 | 4.31 | 4.31 | 6.16e-16 |  | 0 |
| `linear/nist_NoInt2` | bse_HC3 | 1 | 3.2 | 3.2 | 6.95e-16 |  | 0 |
| `linear/nist_NoInt2` | bse_HAC | 1 | 3.56 | 3.56 | 4.30e-16 |  | 0 |
| `linear/nist_NoInt2` | bp_lm | 1 | 7 | 7 | 1.38e-15 |  | 0 |
| `linear/nist_NoInt2` | bp_lm_pvalue | 1 | 8.62 | 8.62 | 1.79e-15 |  | 0 |
| `linear/nist_NoInt2` | bp_fvalue | 1 | 36 | 36 | 5.33e-15 |  | 0 |
| `linear/nist_NoInt2` | bp_f_pvalue | 1 | 13.3 | 13.3 | 2.22e-15 |  | 0 |
| `linear/nist_NoInt2` | bp_lm_nonstudentized | 1 | 15.1 | 15.1 | 1.80e-15 |  | 0 |
| `linear/nist_NoInt2` | bp_lm_nonstudentized_pvalue | 1 | 4.79 | 4.79 | 5.37e-16 |  | 0 |
| `linear/nist_NoInt2` | fvalue | 1 | 0.333 | 0.333 | 6.34e-17 |  | 0 |
| `linear/nist_NoInt2` | f_pvalue | 1 | 0.214 | 0.214 | 2.79e-17 |  | 0 |
| `linear/nist_Norris` | params | 2 | 0.431 | 0.431 | 9.55e-17 | 9.25e-17 | 0 |
| `linear/nist_Norris` | bse | 2 | 2.63 | 2.63 | 3.32e-16 | 2.48e-16 | 0 |
| `linear/nist_Norris` | tvalues | 2 | 2.53 | 2.53 | 4.93e-16 | 4.93e-16 | 0 |
| `linear/nist_Norris` | pvalues | 2 | 88.3 | 88.3 | 1.65e-14 | 2.39e-16 | 0 |
| `linear/nist_Norris` | conf_int_lower | 2 | 1.33 | 1.33 | 2.00e-16 | 1.67e-16 | 0 |
| `linear/nist_Norris` | conf_int_upper | 2 | 3.64 | 3.64 | 4.79e-16 | 1.08e-16 | 0 |
| `linear/nist_Norris` | fittedvalues | 36 | 3.48 | 3.48 | 6.30e-16 | 1.03e-16 | 0 |
| `linear/nist_Norris` | resid | 36 | 6.29e+03 | 6.29e+03 | 1.06e-12 | 6.03e-14 | 0 |
| `linear/nist_Norris` | hat | 36 | 3.34 | 3.34 | 5.83e-16 | 3.56e-16 | 0 |
| `linear/nist_Norris` | cooks_distance | 36 | 1.32e+04 | 1.32e+04 | 2.13e-12 | 8.63e-14 | 0 |
| `linear/nist_Norris` | ssr | 1 | 0.0267 | 0.0267 | 3.56e-18 |  | 0 |
| `linear/nist_Norris` | ess | 1 | 0.199 | 0.199 | 4.36e-17 |  | 0 |
| `linear/nist_Norris` | centered_tss | 1 | 0.317 | 0.317 | 6.93e-17 |  | 0 |
| `linear/nist_Norris` | rsquared | 1 | 0.185 | 0.185 | 2.06e-17 |  | 0 |
| `linear/nist_Norris` | rsquared_adj | 1 | 0.25 | 0.25 | 2.78e-17 |  | 0 |
| `linear/nist_Norris` | scale | 1 | 0.496 | 0.496 | 7.03e-17 |  | 0 |
| `linear/nist_Norris` | llf | 1 | 1.54 | 1.54 | 2.40e-16 |  | 0 |
| `linear/nist_Norris` | aic | 1 | 1.54 | 1.54 | 2.30e-16 |  | 0 |
| `linear/nist_Norris` | bic | 1 | 1.68 | 1.68 | 2.42e-16 |  | 0 |
| `linear/nist_Norris` | durbin_watson | 1 | 166 | 166 | 2.90e-14 |  | 0 |
| `linear/nist_Norris` | jarque_bera | 1 | 2.88e+03 | 2.88e+03 | 4.08e-13 |  | 0 |
| `linear/nist_Norris` | jarque_bera_pvalue | 1 | 2.63e+03 | 2.63e+03 | 3.20e-13 |  | 0 |
| `linear/nist_Norris` | resid_studentized_internal | 36 | 7.25e+03 | 7.25e+03 | 1.06e-12 | 6.05e-14 | 0 |
| `linear/nist_Norris` | resid_studentized_external | 36 | 7.14e+03 | 7.14e+03 | 1.06e-12 | 6.33e-14 | 0 |
| `linear/nist_Norris` | bse_HC0 | 2 | 83.9 | 83.9 | 9.51e-15 | 1.98e-15 | 0 |
| `linear/nist_Norris` | bse_HC1 | 2 | 43 | 43 | 9.47e-15 | 2.04e-15 | 0 |
| `linear/nist_Norris` | bse_HC2 | 2 | 45.3 | 45.3 | 9.81e-15 | 2.73e-15 | 0 |
| `linear/nist_Norris` | bse_HC3 | 2 | 49 | 49 | 1.02e-14 | 2.77e-15 | 0 |
| `linear/nist_Norris` | bse_HAC | 2 | 87.7 | 87.7 | 1.72e-14 | 1.25e-14 | 0 |
| `linear/nist_Norris` | bp_lm | 1 | 423 | 423 | 5.06e-14 |  | 0 |
| `linear/nist_Norris` | bp_lm_pvalue | 1 | 1.55e+03 | 1.55e+03 | 2.09e-13 |  | 0 |
| `linear/nist_Norris` | bp_fvalue | 1 | 316 | 316 | 6.35e-14 |  | 0 |
| `linear/nist_Norris` | bp_f_pvalue | 1 | 1.52e+03 | 1.52e+03 | 2.44e-13 |  | 0 |
| `linear/nist_Norris` | bp_lm_nonstudentized | 1 | 196 | 196 | 3.77e-14 |  | 0 |
| `linear/nist_Norris` | bp_lm_nonstudentized_pvalue | 1 | 1.04e+03 | 1.04e+03 | 1.90e-13 |  | 0 |
| `linear/nist_Norris` | fvalue | 1 | 0.969 | 0.969 | 1.66e-16 |  | 0 |
| `linear/nist_Norris` | f_pvalue | 1 | 16.7 | 16.7 | 3.13e-15 |  | 0 |
| `linear/nist_Pontius` | params | 3 | 0.301 | 0.301 | 4.35e-17 | 3.84e-17 | 0 |
| `linear/nist_Pontius` | bse | 3 | 4.23 | 4.23 | 5.31e-16 | 5.31e-16 | 0 |
| `linear/nist_Pontius` | tvalues | 3 | 3.83 | 3.83 | 5.48e-16 | 5.48e-16 | 0 |
| `linear/nist_Pontius` | pvalues | 3 | 157 | 157 | 2.01e-14 | 1.09e-14 | 0 |
| `linear/nist_Pontius` | conf_int_lower | 3 | 2.27 | 2.27 | 2.71e-16 | 2.71e-16 | 0 |
| `linear/nist_Pontius` | conf_int_upper | 3 | 0.66 | 0.66 | 8.02e-17 | 8.02e-17 | 0 |
| `linear/nist_Pontius` | fittedvalues | 40 | 0.844 | 0.844 | 1.07e-16 | 5.24e-17 | 0 |
| `linear/nist_Pontius` | resid | 40 | 1.99e+04 | 1.99e+04 | 3.15e-12 | 2.92e-13 | 0 |
| `linear/nist_Pontius` | hat | 40 | 10.6 | 10.6 | 1.20e-15 | 4.36e-16 | 0 |
| `linear/nist_Pontius` | cooks_distance | 40 | 4.34e+04 | 4.34e+04 | 6.29e-12 | 3.43e-13 | 0 |
| `linear/nist_Pontius` | ssr | 1 | 0.249 | 0.249 | 3.39e-17 |  | 0 |
| `linear/nist_Pontius` | ess | 1 | 0.184 | 0.184 | 2.09e-17 |  | 0 |
| `linear/nist_Pontius` | centered_tss | 1 | 0.293 | 0.293 | 3.33e-17 |  | 0 |
| `linear/nist_Pontius` | rsquared | 1 | 0.291 | 0.291 | 3.23e-17 |  | 0 |
| `linear/nist_Pontius` | rsquared_adj | 1 | 0.144 | 0.144 | 1.60e-17 |  | 0 |
| `linear/nist_Pontius` | scale | 1 | 0.486 | 0.486 | 7.64e-17 |  | 0 |
| `linear/nist_Pontius` | llf | 1 | 1.18 | 1.18 | 2.37e-16 |  | 0 |
| `linear/nist_Pontius` | aic | 1 | 1.18 | 1.18 | 2.39e-16 |  | 0 |
| `linear/nist_Pontius` | bic | 1 | 1.34 | 1.34 | 2.73e-16 |  | 0 |
| `linear/nist_Pontius` | durbin_watson | 1 | 36.5 | 36.5 | 4.95e-15 |  | 0 |
| `linear/nist_Pontius` | jarque_bera | 1 | 4.7e+03 | 4.7e+03 | 6.86e-13 |  | 0 |
| `linear/nist_Pontius` | jarque_bera_pvalue | 1 | 534 | 534 | 6.52e-14 |  | 0 |
| `linear/nist_Pontius` | resid_studentized_internal | 40 | 2.42e+04 | 2.42e+04 | 3.15e-12 | 2.93e-13 | 0 |
| `linear/nist_Pontius` | resid_studentized_external | 40 | 2.39e+04 | 2.39e+04 | 3.15e-12 | 2.95e-13 | 0 |
| `linear/nist_Pontius` | bse_HC0 | 3 | 105 | 105 | 1.83e-14 | 6.64e-15 | 0 |
| `linear/nist_Pontius` | bse_HC1 | 3 | 110 | 110 | 1.83e-14 | 6.63e-15 | 0 |
| `linear/nist_Pontius` | bse_HC2 | 3 | 111 | 111 | 1.81e-14 | 7.13e-15 | 0 |
| `linear/nist_Pontius` | bse_HC3 | 3 | 118 | 118 | 1.80e-14 | 7.79e-15 | 0 |
| `linear/nist_Pontius` | bse_HAC | 3 | 112 | 112 | 1.55e-14 | 4.38e-15 | 0 |
| `linear/nist_Pontius` | bp_lm | 1 | 2e+03 | 2e+03 | 3.49e-13 |  | 0 |
| `linear/nist_Pontius` | bp_lm_pvalue | 1 | 1.06e+03 | 1.06e+03 | 2.22e-13 |  | 0 |
| `linear/nist_Pontius` | bp_fvalue | 1 | 1.97e+03 | 1.97e+03 | 3.60e-13 |  | 0 |
| `linear/nist_Pontius` | bp_f_pvalue | 1 | 1.05e+03 | 1.05e+03 | 2.13e-13 |  | 0 |
| `linear/nist_Pontius` | bp_lm_nonstudentized | 1 | 1.87e+03 | 1.87e+03 | 3.69e-13 |  | 0 |
| `linear/nist_Pontius` | bp_lm_nonstudentized_pvalue | 1 | 1.07e+03 | 1.07e+03 | 2.08e-13 |  | 0 |
| `linear/nist_Pontius` | fvalue | 1 | 0.0638 | 0.0638 | 1.03e-17 |  | 0 |
| `linear/nist_Pontius` | f_pvalue | 1 | 1.42 | 1.42 | 1.86e-16 |  | 0 |
| `linear/nist_Wampler1` | params | 6 | 3.57e-281 | 3.57e-281 | 7.93e-297 | 0 | 0 |
| `linear/nist_Wampler1` | fittedvalues | 21 | 3.57e-281 | 3.57e-281 | 7.93e-297 | 0 | 0 |
| `linear/nist_Wampler1` | resid | 21 | 0 | 0 | 0 | 0 | 0 |
| `linear/nist_Wampler1` | ssr | 1 | 0 | 0 | 0 |  | 0 |
| `linear/nist_Wampler1` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `linear/nist_Wampler1` | bse | 6 | 0 | 0 | 0 | 0 | 0 |
| `linear/nist_Wampler1` | rsquared | 1 | 0 | 0 | 0 |  | 0 |
| `linear/nist_Wampler2` | params | 6 | 0.466 | 0.466 | 8.28e-17 | 3.87e-17 | 0 |
| `linear/nist_Wampler2` | bse | 6 | 8.70e+15 | 8.70e+15 | 1 | 1 | 0 |
| `linear/nist_Wampler2` | tvalues | 6 | inf | inf | inf | nan | 6 |
| `linear/nist_Wampler2` | pvalues | 6 | inf | inf | inf | nan | 6 |
| `linear/nist_Wampler2` | conf_int_lower | 6 | 296 | 296 | 6.41e-14 | 2.10e-15 | 0 |
| `linear/nist_Wampler2` | conf_int_upper | 6 | 296 | 296 | 6.42e-14 | 2.05e-15 | 0 |
| `linear/nist_Wampler2` | fittedvalues | 21 | 0.591 | 0.591 | 8.22e-17 | 2.32e-17 | 0 |
| `linear/nist_Wampler2` | resid | 21 | 2.11e+17 | 2.11e+17 | 46 | 1.47 | 0 |
| `linear/nist_Wampler2` | hat | 21 | 308 | 308 | 5.53e-14 | 1.29e-14 | 0 |
| `linear/nist_Wampler2` | cooks_distance | 21 | inf | inf | inf | nan | 21 |
| `linear/nist_Wampler2` | ssr | 1 | 5.25e+15 | 5.25e+15 | 1 |  | 0 |
| `linear/nist_Wampler2` | ess | 1 | 0.0362 | 0.0362 | 4.99e-18 |  | 0 |
| `linear/nist_Wampler2` | centered_tss | 1 | 0.0362 | 0.0362 | 4.99e-18 |  | 0 |
| `linear/nist_Wampler2` | rsquared | 1 | 5.02e-18 | 5.02e-18 | 1.11e-33 |  | 0 |
| `linear/nist_Wampler2` | rsquared_adj | 1 | 6.69e-18 | 6.69e-18 | 1.48e-33 |  | 0 |
| `linear/nist_Wampler2` | scale | 1 | 5.60e+15 | 5.60e+15 | 1 |  | 0 |
| `linear/nist_Wampler2` | llf | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | aic | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | bic | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | durbin_watson | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | jarque_bera | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | jarque_bera_pvalue | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | resid_studentized_internal | 21 | inf | inf | inf | nan | 21 |
| `linear/nist_Wampler2` | resid_studentized_external | 21 | inf | inf | inf | nan | 21 |
| `linear/nist_Wampler2` | bse_HC0 | 6 | 7.27e+15 | 7.27e+15 | 1 | 1 | 0 |
| `linear/nist_Wampler2` | bse_HC1 | 6 | 8.60e+15 | 8.60e+15 | 1 | 1 | 0 |
| `linear/nist_Wampler2` | bse_HC2 | 6 | 8.91e+15 | 8.91e+15 | 1 | 1 | 0 |
| `linear/nist_Wampler2` | bse_HC3 | 6 | 7.54e+15 | 7.54e+15 | 1 | 1 | 0 |
| `linear/nist_Wampler2` | bse_HAC | 6 | 8.66e+15 | 8.66e+15 | 1 | 1 | 0 |
| `linear/nist_Wampler2` | bp_lm | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | bp_lm_pvalue | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | bp_fvalue | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | bp_f_pvalue | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | bp_lm_nonstudentized | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | bp_lm_nonstudentized_pvalue | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | fvalue | 1 | inf | inf | inf |  | 1 |
| `linear/nist_Wampler2` | f_pvalue | 1 | 5.43e+15 | 5.43e+15 | 1 |  | 0 |
| `linear/nist_Wampler3` | params | 6 | 1.56e-48 | 1.56e-48 | 3.46e-64 | 1.42e-64 | 0 |
| `linear/nist_Wampler3` | bse | 6 | 165 | 165 | 3.49e-14 | 2.42e-14 | 0 |
| `linear/nist_Wampler3` | tvalues | 6 | 299 | 299 | 3.49e-14 | 6.05e-15 | 0 |
| `linear/nist_Wampler3` | pvalues | 6 | 655 | 655 | 7.69e-14 | 4.06e-16 | 0 |
| `linear/nist_Wampler3` | conf_int_lower | 6 | 177 | 177 | 3.50e-14 | 2.43e-14 | 0 |
| `linear/nist_Wampler3` | conf_int_upper | 6 | 177 | 177 | 3.50e-14 | 2.42e-14 | 0 |
| `linear/nist_Wampler3` | fittedvalues | 21 | 1.56e-48 | 1.56e-48 | 3.46e-64 | 1.68e-70 | 0 |
| `linear/nist_Wampler3` | resid | 21 | 3.04e-51 | 3.04e-51 | 4.56e-67 | 9.56e-68 | 0 |
| `linear/nist_Wampler3` | hat | 21 | 308 | 308 | 5.53e-14 | 1.29e-14 | 0 |
| `linear/nist_Wampler3` | cooks_distance | 21 | 611 | 611 | 1.05e-13 | 3.50e-14 | 0 |
| `linear/nist_Wampler3` | ssr | 1 | 4.98e-60 | 4.98e-60 | 8.88e-76 |  | 0 |
| `linear/nist_Wampler3` | ess | 1 | 0.333 | 0.333 | 6.92e-17 |  | 0 |
| `linear/nist_Wampler3` | centered_tss | 1 | 0.333 | 0.333 | 6.92e-17 |  | 0 |
| `linear/nist_Wampler3` | rsquared | 1 | 0.415 | 0.415 | 4.60e-17 |  | 0 |
| `linear/nist_Wampler3` | rsquared_adj | 1 | 0.447 | 0.447 | 4.97e-17 |  | 0 |
| `linear/nist_Wampler3` | scale | 1 | 0.133 | 0.133 | 2.23e-17 |  | 0 |
| `linear/nist_Wampler3` | llf | 1 | 0.328 | 0.328 | 4.93e-17 |  | 0 |
| `linear/nist_Wampler3` | aic | 1 | 0.328 | 0.328 | 4.78e-17 |  | 0 |
| `linear/nist_Wampler3` | bic | 1 | 0.0333 | 0.0333 | 4.76e-18 |  | 0 |
| `linear/nist_Wampler3` | durbin_watson | 1 | 0.148 | 0.148 | 1.68e-17 |  | 0 |
| `linear/nist_Wampler3` | jarque_bera | 1 | 1.17 | 1.17 | 1.66e-16 |  | 0 |
| `linear/nist_Wampler3` | jarque_bera_pvalue | 1 | 1.71 | 1.71 | 2.27e-16 |  | 0 |
| `linear/nist_Wampler3` | resid_studentized_internal | 21 | 77.9 | 77.9 | 1.24e-14 | 3.74e-15 | 0 |
| `linear/nist_Wampler3` | resid_studentized_external | 21 | 80.1 | 80.1 | 1.33e-14 | 4.00e-15 | 0 |
| `linear/nist_Wampler3` | bse_HC0 | 6 | 218 | 218 | 2.48e-14 | 1.66e-14 | 0 |
| `linear/nist_Wampler3` | bse_HC1 | 6 | 130 | 130 | 2.49e-14 | 1.67e-14 | 0 |
| `linear/nist_Wampler3` | bse_HC2 | 6 | 230 | 230 | 2.92e-14 | 1.97e-14 | 0 |
| `linear/nist_Wampler3` | bse_HC3 | 6 | 82.1 | 82.1 | 1.15e-14 | 7.28e-15 | 0 |
| `linear/nist_Wampler3` | bse_HAC | 6 | 408 | 408 | 5.07e-14 | 2.87e-14 | 0 |
| `linear/nist_Wampler3` | bp_lm | 1 | 0.443 | 0.443 | 5.18e-17 |  | 0 |
| `linear/nist_Wampler3` | bp_lm_pvalue | 1 | 2.74 | 2.74 | 4.94e-16 |  | 0 |
| `linear/nist_Wampler3` | bp_fvalue | 1 | 0.237 | 0.237 | 2.69e-17 |  | 0 |
| `linear/nist_Wampler3` | bp_f_pvalue | 1 | 1.02 | 1.02 | 1.31e-16 |  | 0 |
| `linear/nist_Wampler3` | bp_lm_nonstudentized | 1 | 0.0885 | 0.0885 | 1.20e-17 |  | 0 |
| `linear/nist_Wampler3` | bp_lm_nonstudentized_pvalue | 1 | 0.271 | 0.271 | 3.08e-17 |  | 0 |
| `linear/nist_Wampler3` | fvalue | 1 | 0.677 | 0.677 | 1.17e-16 |  | 0 |
| `linear/nist_Wampler3` | f_pvalue | 1 | 9.32 | 9.32 | 1.09e-15 |  | 0 |
| `linear/nist_Wampler4` | params | 6 | 1.56e-48 | 1.56e-48 | 3.46e-64 | 1.42e-64 | 0 |
| `linear/nist_Wampler4` | bse | 6 | 258 | 258 | 3.49e-14 | 2.41e-14 | 0 |
| `linear/nist_Wampler4` | tvalues | 6 | 191 | 191 | 3.49e-14 | 5.93e-15 | 0 |
| `linear/nist_Wampler4` | pvalues | 6 | 3.53 | 3.53 | 4.21e-16 | 1.62e-16 | 0 |
| `linear/nist_Wampler4` | conf_int_lower | 6 | 276 | 276 | 3.51e-14 | 2.42e-14 | 0 |
| `linear/nist_Wampler4` | conf_int_upper | 6 | 276 | 276 | 3.51e-14 | 2.42e-14 | 0 |
| `linear/nist_Wampler4` | fittedvalues | 21 | 1.56e-48 | 1.56e-48 | 3.46e-64 | 1.68e-70 | 0 |
| `linear/nist_Wampler4` | resid | 21 | 2.38e-53 | 2.38e-53 | 4.56e-69 | 9.56e-70 | 0 |
| `linear/nist_Wampler4` | hat | 21 | 308 | 308 | 5.53e-14 | 1.29e-14 | 0 |
| `linear/nist_Wampler4` | cooks_distance | 21 | 611 | 611 | 1.05e-13 | 3.51e-14 | 0 |
| `linear/nist_Wampler4` | ssr | 1 | 4.67e-61 | 4.67e-61 | 6.82e-77 |  | 0 |
| `linear/nist_Wampler4` | ess | 1 | 0.333 | 0.333 | 6.92e-17 |  | 0 |
| `linear/nist_Wampler4` | centered_tss | 1 | 0.333 | 0.333 | 6.63e-17 |  | 0 |
| `linear/nist_Wampler4` | rsquared | 1 | 0.386 | 0.386 | 4.47e-17 |  | 0 |
| `linear/nist_Wampler4` | rsquared_adj | 1 | 0.181 | 0.181 | 2.13e-17 |  | 0 |
| `linear/nist_Wampler4` | scale | 1 | 0.333 | 0.333 | 4.57e-17 |  | 0 |
| `linear/nist_Wampler4` | llf | 1 | 0.394 | 0.394 | 7.83e-17 |  | 0 |
| `linear/nist_Wampler4` | aic | 1 | 0.394 | 0.394 | 7.67e-17 |  | 0 |
| `linear/nist_Wampler4` | bic | 1 | 0.542 | 0.542 | 1.04e-16 |  | 0 |
| `linear/nist_Wampler4` | durbin_watson | 1 | 0.148 | 0.148 | 1.68e-17 |  | 0 |
| `linear/nist_Wampler4` | jarque_bera | 1 | 1.17 | 1.17 | 1.66e-16 |  | 0 |
| `linear/nist_Wampler4` | jarque_bera_pvalue | 1 | 1.71 | 1.71 | 2.27e-16 |  | 0 |
| `linear/nist_Wampler4` | resid_studentized_internal | 21 | 77.9 | 77.9 | 1.24e-14 | 3.73e-15 | 0 |
| `linear/nist_Wampler4` | resid_studentized_external | 21 | 80.1 | 80.1 | 1.33e-14 | 3.99e-15 | 0 |
| `linear/nist_Wampler4` | bse_HC0 | 6 | 278 | 278 | 4.05e-14 | 2.28e-14 | 0 |
| `linear/nist_Wampler4` | bse_HC1 | 6 | 329 | 329 | 4.05e-14 | 2.28e-14 | 0 |
| `linear/nist_Wampler4` | bse_HC2 | 6 | 188 | 188 | 3.05e-14 | 2.06e-14 | 0 |
| `linear/nist_Wampler4` | bse_HC3 | 6 | 86 | 86 | 1.18e-14 | 8.82e-15 | 0 |
| `linear/nist_Wampler4` | bse_HAC | 6 | 353 | 353 | 5.61e-14 | 3.04e-14 | 0 |
| `linear/nist_Wampler4` | bp_lm | 1 | 0.557 | 0.557 | 6.52e-17 |  | 0 |
| `linear/nist_Wampler4` | bp_lm_pvalue | 1 | 3.26 | 3.26 | 5.89e-16 |  | 0 |
| `linear/nist_Wampler4` | bp_fvalue | 1 | 0.237 | 0.237 | 2.69e-17 |  | 0 |
| `linear/nist_Wampler4` | bp_f_pvalue | 1 | 1.02 | 1.02 | 1.31e-16 |  | 0 |
| `linear/nist_Wampler4` | bp_lm_nonstudentized | 1 | 0.911 | 0.911 | 1.23e-16 |  | 0 |
| `linear/nist_Wampler4` | bp_lm_nonstudentized_pvalue | 1 | 0.271 | 0.271 | 3.08e-17 |  | 0 |
| `linear/nist_Wampler4` | fvalue | 1 | 0.495 | 0.495 | 1.04e-16 |  | 0 |
| `linear/nist_Wampler4` | f_pvalue | 1 | 2.26 | 2.26 | 4.92e-16 |  | 0 |
| `linear/nist_Wampler5` | params | 6 | 1.56e-48 | 1.56e-48 | 3.46e-64 | 1.42e-64 | 0 |
| `linear/nist_Wampler5` | bse | 6 | 202 | 202 | 3.50e-14 | 2.42e-14 | 0 |
| `linear/nist_Wampler5` | tvalues | 6 | 245 | 245 | 3.50e-14 | 5.91e-15 | 0 |
| `linear/nist_Wampler5` | pvalues | 6 | 0.461 | 0.461 | 5.12e-17 | 3.18e-17 | 0 |
| `linear/nist_Wampler5` | conf_int_lower | 6 | 215 | 215 | 3.50e-14 | 2.42e-14 | 0 |
| `linear/nist_Wampler5` | conf_int_upper | 6 | 215 | 215 | 3.50e-14 | 2.42e-14 | 0 |
| `linear/nist_Wampler5` | fittedvalues | 21 | 1.56e-48 | 1.56e-48 | 3.46e-64 | 1.68e-70 | 0 |
| `linear/nist_Wampler5` | resid | 21 | 3.72e-55 | 3.72e-55 | 4.56e-71 | 9.56e-72 | 0 |
| `linear/nist_Wampler5` | hat | 21 | 308 | 308 | 5.53e-14 | 1.29e-14 | 0 |
| `linear/nist_Wampler5` | cooks_distance | 21 | 611 | 611 | 1.05e-13 | 3.50e-14 | 0 |
| `linear/nist_Wampler5` | ssr | 1 | 6.22e-61 | 6.22e-61 | 7.45e-77 |  | 0 |
| `linear/nist_Wampler5` | ess | 1 | 0.333 | 0.333 | 6.92e-17 |  | 0 |
| `linear/nist_Wampler5` | centered_tss | 1 | 0.333 | 0.333 | 3.98e-17 |  | 0 |
| `linear/nist_Wampler5` | rsquared | 1 | 0.108 | 0.108 | 2.09e-17 |  | 0 |
| `linear/nist_Wampler5` | rsquared_adj | 1 | 3.73 | 3.73 | 6.26e-16 |  | 0 |
| `linear/nist_Wampler5` | scale | 1 | 0.333 | 0.333 | 3.74e-17 |  | 0 |
| `linear/nist_Wampler5` | llf | 1 | 0.547 | 0.547 | 8.13e-17 |  | 0 |
| `linear/nist_Wampler5` | aic | 1 | 0.547 | 0.547 | 8.00e-17 |  | 0 |
| `linear/nist_Wampler5` | bic | 1 | 0.4 | 0.4 | 5.80e-17 |  | 0 |
| `linear/nist_Wampler5` | durbin_watson | 1 | 0.148 | 0.148 | 1.68e-17 |  | 0 |
| `linear/nist_Wampler5` | jarque_bera | 1 | 0.829 | 0.829 | 1.18e-16 |  | 0 |
| `linear/nist_Wampler5` | jarque_bera_pvalue | 1 | 1.29 | 1.29 | 1.72e-16 |  | 0 |
| `linear/nist_Wampler5` | resid_studentized_internal | 21 | 77.9 | 77.9 | 1.24e-14 | 3.75e-15 | 0 |
| `linear/nist_Wampler5` | resid_studentized_external | 21 | 80.1 | 80.1 | 1.33e-14 | 4.01e-15 | 0 |
| `linear/nist_Wampler5` | bse_HC0 | 6 | 148 | 148 | 2.76e-14 | 1.80e-14 | 0 |
| `linear/nist_Wampler5` | bse_HC1 | 6 | 176 | 176 | 2.77e-14 | 1.80e-14 | 0 |
| `linear/nist_Wampler5` | bse_HC2 | 6 | 110 | 110 | 2.28e-14 | 1.51e-14 | 0 |
| `linear/nist_Wampler5` | bse_HC3 | 6 | 70.4 | 70.4 | 1.16e-14 | 5.88e-15 | 0 |
| `linear/nist_Wampler5` | bse_HAC | 6 | 283 | 283 | 5.77e-14 | 3.17e-14 | 0 |
| `linear/nist_Wampler5` | bp_lm | 1 | 0.557 | 0.557 | 6.52e-17 |  | 0 |
| `linear/nist_Wampler5` | bp_lm_pvalue | 1 | 3.26 | 3.26 | 5.89e-16 |  | 0 |
| `linear/nist_Wampler5` | bp_fvalue | 1 | 1.24 | 1.24 | 1.40e-16 |  | 0 |
| `linear/nist_Wampler5` | bp_f_pvalue | 1 | 6.02 | 6.02 | 7.74e-16 |  | 0 |
| `linear/nist_Wampler5` | bp_lm_nonstudentized | 1 | 0.0885 | 0.0885 | 1.20e-17 |  | 0 |
| `linear/nist_Wampler5` | bp_lm_nonstudentized_pvalue | 1 | 0.271 | 0.271 | 3.08e-17 |  | 0 |
| `linear/nist_Wampler5` | fvalue | 1 | 0.508 | 0.508 | 6.52e-17 |  | 0 |
| `linear/nist_Wampler5` | f_pvalue | 1 | 0.268 | 0.268 | 2.97e-17 |  | 0 |
| `linear/ols_cond` | params | 4 | 0.44 | 0.44 | 7.61e-17 | 6.38e-17 | 0 |
| `linear/ols_cond` | bse | 4 | 2.33e+03 | 2.33e+03 | 3.06e-13 | 3.04e-13 | 0 |
| `linear/ols_cond` | tvalues | 4 | 2.66e+03 | 2.66e+03 | 3.06e-13 | 2.62e-13 | 0 |
| `linear/ols_cond` | pvalues | 4 | 1.16 | 1.16 | 1.29e-16 | 9.24e-17 | 0 |
| `linear/ols_cond` | conf_int_lower | 4 | 2.34e+03 | 2.34e+03 | 3.06e-13 | 3.04e-13 | 0 |
| `linear/ols_cond` | conf_int_upper | 4 | 2.34e+03 | 2.34e+03 | 3.06e-13 | 3.04e-13 | 0 |
| `linear/ols_cond` | fittedvalues | 50 | 0.723 | 0.723 | 1.23e-16 | 4.79e-17 | 0 |
| `linear/ols_cond` | resid | 50 | 132 | 132 | 1.66e-14 | 4.94e-17 | 0 |
| `linear/ols_cond` | hat | 50 | 8.66e+04 | 8.66e+04 | 1.73e-11 | 1.02e-12 | 0 |
| `linear/ols_cond` | cooks_distance | 50 | 8.92e+04 | 8.92e+04 | 1.86e-11 | 3.87e-13 | 0 |
| `linear/ols_cond` | ssr | 1 | 0.197 | 0.197 | 2.97e-17 |  | 0 |
| `linear/ols_cond` | ess | 1 | 0.668 | 0.668 | 1.01e-16 |  | 0 |
| `linear/ols_cond` | centered_tss | 1 | 0.432 | 0.432 | 6.54e-17 |  | 0 |
| `linear/ols_cond` | rsquared | 1 | 0.683 | 0.683 | 7.58e-17 |  | 0 |
| `linear/ols_cond` | rsquared_adj | 1 | 2.35 | 2.35 | 2.86e-16 |  | 0 |
| `linear/ols_cond` | scale | 1 | 0.137 | 0.137 | 2.97e-17 |  | 0 |
| `linear/ols_cond` | llf | 1 | 0.138 | 0.138 | 1.89e-17 |  | 0 |
| `linear/ols_cond` | aic | 1 | 0.138 | 0.138 | 1.96e-17 |  | 0 |
| `linear/ols_cond` | bic | 1 | 0.187 | 0.187 | 2.76e-17 |  | 0 |
| `linear/ols_cond` | durbin_watson | 1 | 0.376 | 0.376 | 4.58e-17 |  | 0 |
| `linear/ols_cond` | jarque_bera | 1 | 4.83 | 4.83 | 1.03e-15 |  | 0 |
| `linear/ols_cond` | jarque_bera_pvalue | 1 | 1.9 | 1.9 | 3.56e-16 |  | 0 |
| `linear/ols_cond` | resid_studentized_internal | 50 | 2.7e+03 | 2.7e+03 | 3.11e-13 | 2.81e-14 | 0 |
| `linear/ols_cond` | resid_studentized_external | 50 | 2.69e+03 | 2.69e+03 | 3.12e-13 | 2.80e-14 | 0 |
| `linear/ols_cond` | bse_HC0 | 4 | 2.52e+03 | 2.52e+03 | 3.31e-13 | 3.29e-13 | 0 |
| `linear/ols_cond` | bse_HC1 | 4 | 2.62e+03 | 2.62e+03 | 3.31e-13 | 3.29e-13 | 0 |
| `linear/ols_cond` | bse_HC2 | 4 | 2.88e+03 | 2.88e+03 | 3.54e-13 | 3.52e-13 | 0 |
| `linear/ols_cond` | bse_HC3 | 4 | 3.31e+03 | 3.31e+03 | 3.79e-13 | 3.76e-13 | 0 |
| `linear/ols_cond` | bse_HAC | 4 | 3e+03 | 3e+03 | 4.20e-13 | 4.16e-13 | 0 |
| `linear/ols_cond` | bse_cluster | 4 | 895 | 895 | 1.51e-13 | 7.11e-14 | 0 |
| `linear/ols_cond` | pvalues_cluster | 4 | 0.596 | 0.596 | 6.61e-17 | 4.83e-17 | 0 |
| `linear/ols_cond` | fvalue_cluster | 1 | 6.84e+04 | 6.84e+04 | 9.25e-12 |  | 0 |
| `linear/ols_cond` | f_pvalue_cluster | 1 | 1.78e+5 | 1.78e+5 | 2.73e-11 |  | 0 |
| `linear/ols_cond` | bp_lm | 1 | 2.28e+04 | 2.28e+04 | 3.20e-12 |  | 0 |
| `linear/ols_cond` | bp_lm_pvalue | 1 | 4.88e+04 | 4.88e+04 | 7.68e-12 |  | 0 |
| `linear/ols_cond` | bp_fvalue | 1 | 2.35e+04 | 2.35e+04 | 3.20e-12 |  | 0 |
| `linear/ols_cond` | bp_f_pvalue | 1 | 4.65e+04 | 4.65e+04 | 7.04e-12 |  | 0 |
| `linear/ols_cond` | bp_lm_nonstudentized | 1 | 1.98e+04 | 1.98e+04 | 3.20e-12 |  | 0 |
| `linear/ols_cond` | bp_lm_nonstudentized_pvalue | 1 | 5.58e+04 | 5.58e+04 | 6.44e-12 |  | 0 |
| `linear/ols_cond` | fvalue | 1 | 0.419 | 0.419 | 6.47e-17 |  | 0 |
| `linear/ols_cond` | f_pvalue | 1 | 4.02 | 4.02 | 5.72e-16 |  | 0 |
| `linear/ols_int` | params | 4 | 0.385 | 0.385 | 6.16e-17 | 5.70e-17 | 0 |
| `linear/ols_int` | bse | 4 | 1.24 | 1.24 | 2.08e-16 | 1.03e-16 | 0 |
| `linear/ols_int` | tvalues | 4 | 1.47 | 1.47 | 1.79e-16 | 1.44e-16 | 0 |
| `linear/ols_int` | pvalues | 4 | 15 | 15 | 2.81e-15 | 8.59e-16 | 0 |
| `linear/ols_int` | conf_int_lower | 4 | 21.4 | 21.4 | 3.55e-15 | 1.45e-16 | 0 |
| `linear/ols_int` | conf_int_upper | 4 | 1.32 | 1.32 | 2.22e-16 | 7.86e-17 | 0 |
| `linear/ols_int` | fittedvalues | 30 | 4.76 | 4.76 | 7.58e-16 | 6.95e-17 | 0 |
| `linear/ols_int` | resid | 30 | 15.8 | 15.8 | 1.83e-15 | 1.62e-16 | 0 |
| `linear/ols_int` | hat | 30 | 4.81 | 4.81 | 9.11e-16 | 2.57e-16 | 0 |
| `linear/ols_int` | cooks_distance | 30 | 28 | 28 | 4.01e-15 | 5.85e-16 | 0 |
| `linear/ols_int` | ssr | 1 | 0.472 | 0.472 | 8.14e-17 |  | 0 |
| `linear/ols_int` | ess | 1 | 0.494 | 0.494 | 5.63e-17 |  | 0 |
| `linear/ols_int` | centered_tss | 1 | 0.056 | 0.056 | 1.10e-17 |  | 0 |
| `linear/ols_int` | rsquared | 1 | 0.151 | 0.151 | 1.95e-17 |  | 0 |
| `linear/ols_int` | rsquared_adj | 1 | 0.245 | 0.245 | 3.23e-17 |  | 0 |
| `linear/ols_int` | scale | 1 | 0.35 | 0.35 | 4.91e-17 |  | 0 |
| `linear/ols_int` | llf | 1 | 0.149 | 0.149 | 2.87e-17 |  | 0 |
| `linear/ols_int` | aic | 1 | 0.149 | 0.149 | 2.59e-17 |  | 0 |
| `linear/ols_int` | bic | 1 | 0.5 | 0.5 | 8.13e-17 |  | 0 |
| `linear/ols_int` | durbin_watson | 1 | 0.796 | 0.796 | 1.43e-16 |  | 0 |
| `linear/ols_int` | jarque_bera | 1 | 5.88 | 5.88 | 6.56e-16 |  | 0 |
| `linear/ols_int` | jarque_bera_pvalue | 1 | 2.34 | 2.34 | 3.52e-16 |  | 0 |
| `linear/ols_int` | resid_studentized_internal | 30 | 10.8 | 10.8 | 1.82e-15 | 1.78e-16 | 0 |
| `linear/ols_int` | resid_studentized_external | 30 | 11.2 | 11.2 | 1.95e-15 | 1.95e-16 | 0 |
| `linear/ols_int` | bse_HC0 | 4 | 2.58 | 2.58 | 3.77e-16 | 2.22e-16 | 0 |
| `linear/ols_int` | bse_HC1 | 4 | 2.59 | 2.59 | 3.52e-16 | 2.02e-16 | 0 |
| `linear/ols_int` | bse_HC2 | 4 | 1.16 | 1.16 | 1.54e-16 | 1.03e-16 | 0 |
| `linear/ols_int` | bse_HC3 | 4 | 2.48 | 2.48 | 2.98e-16 | 1.96e-16 | 0 |
| `linear/ols_int` | bse_HAC | 4 | 2.77 | 2.77 | 3.63e-16 | 2.68e-16 | 0 |
| `linear/ols_int` | bse_cluster | 4 | 2.89 | 2.89 | 4.72e-16 | 2.52e-16 | 0 |
| `linear/ols_int` | pvalues_cluster | 4 | 13.8 | 13.8 | 2.07e-15 | 1.16e-15 | 0 |
| `linear/ols_int` | fvalue_cluster | 1 | 3.55 | 3.55 | 6.84e-16 |  | 0 |
| `linear/ols_int` | f_pvalue_cluster | 1 | 11.5 | 11.5 | 1.61e-15 |  | 0 |
| `linear/ols_int` | bp_lm | 1 | 1 | 1 | 1.69e-16 |  | 0 |
| `linear/ols_int` | bp_lm_pvalue | 1 | 0.439 | 0.439 | 5.51e-17 |  | 0 |
| `linear/ols_int` | bp_fvalue | 1 | 0.801 | 0.801 | 1.15e-16 |  | 0 |
| `linear/ols_int` | bp_f_pvalue | 1 | 0.224 | 0.224 | 2.76e-17 |  | 0 |
| `linear/ols_int` | bp_lm_nonstudentized | 1 | 1.87 | 1.87 | 3.07e-16 |  | 0 |
| `linear/ols_int` | bp_lm_nonstudentized_pvalue | 1 | 0.603 | 0.603 | 7.62e-17 |  | 0 |
| `linear/ols_int` | fvalue | 1 | 0.28 | 0.28 | 3.79e-17 |  | 0 |
| `linear/ols_int` | f_pvalue | 1 | 3.1 | 3.1 | 5.54e-16 |  | 0 |
| `linear/ols_kappa11` | params | 6 | 1.55 | 1.55 | 1.86e-16 | 1.32e-16 | 0 |
| `linear/ols_kappa11` | bse | 6 | 4.44e+9 | 4.44e+9 | 5.48e-7 | 5.39e-7 | 0 |
| `linear/ols_kappa11` | tvalues | 6 | 3.47e+9 | 3.47e+9 | 5.48e-7 | 5.39e-7 | 0 |
| `linear/ols_kappa11` | pvalues | 6 | 1.05e+04 | 1.05e+04 | 1.17e-12 | 1.15e-12 | 0 |
| `linear/ols_kappa11` | conf_int_lower | 6 | 4.48e+9 | 4.48e+9 | 5.48e-7 | 5.39e-7 | 0 |
| `linear/ols_kappa11` | conf_int_upper | 6 | 4.48e+9 | 4.48e+9 | 5.48e-7 | 5.39e-7 | 0 |
| `linear/ols_kappa11` | fittedvalues | 50 | 9.21e+5 | 9.21e+5 | 1.26e-10 | 1.02e-10 | 0 |
| `linear/ols_kappa11` | resid | 50 | 1.75e+6 | 1.75e+6 | 1.98e-10 | 1.02e-12 | 0 |
| `linear/ols_kappa11` | hat | 50 | 7.93e+10 | 7.93e+10 | 1.50e-5 | 1.21e-6 | 0 |
| `linear/ols_kappa11` | cooks_distance | 50 | 9.19e+10 | 9.19e+10 | 1.74e-5 | 8.68e-7 | 0 |
| `linear/ols_kappa11` | ssr | 1 | 0.134 | 0.134 | 1.55e-17 |  | 0 |
| `linear/ols_kappa11` | ess | 1 | 1.44e+6 | 1.44e+6 | 2.04e-10 |  | 0 |
| `linear/ols_kappa11` | centered_tss | 1 | 0.473 | 0.473 | 5.49e-17 |  | 0 |
| `linear/ols_kappa11` | rsquared | 1 | 1.50e+6 | 1.50e+6 | 2.04e-10 |  | 0 |
| `linear/ols_kappa11` | rsquared_adj | 1 | 833 | 833 | 1.70e-13 |  | 0 |
| `linear/ols_kappa11` | scale | 1 | 0.37 | 0.37 | 5.90e-17 |  | 0 |
| `linear/ols_kappa11` | llf | 1 | 0.389 | 0.389 | 6.79e-17 |  | 0 |
| `linear/ols_kappa11` | aic | 1 | 0.389 | 0.389 | 6.55e-17 |  | 0 |
| `linear/ols_kappa11` | bic | 1 | 0.175 | 0.175 | 2.86e-17 |  | 0 |
| `linear/ols_kappa11` | durbin_watson | 1 | 456 | 456 | 5.61e-14 |  | 0 |
| `linear/ols_kappa11` | jarque_bera | 1 | 1.1e+04 | 1.1e+04 | 1.55e-12 |  | 0 |
| `linear/ols_kappa11` | jarque_bera_pvalue | 1 | 1e+04 | 1e+04 | 1.22e-12 |  | 0 |
| `linear/ols_kappa11` | resid_studentized_internal | 50 | 3.15e+9 | 3.15e+9 | 5.93e-7 | 6.95e-8 | 0 |
| `linear/ols_kappa11` | resid_studentized_external | 50 | 3.15e+9 | 3.15e+9 | 5.98e-7 | 7.08e-8 | 0 |
| `linear/ols_kappa11` | bse_HC0 | 6 | 7.23e+9 | 7.23e+9 | 9.04e-7 | 8.90e-7 | 0 |
| `linear/ols_kappa11` | bse_HC1 | 6 | 7.71e+9 | 7.71e+9 | 9.04e-7 | 8.90e-7 | 0 |
| `linear/ols_kappa11` | bse_HC2 | 6 | 7.17e+9 | 7.17e+9 | 9.88e-7 | 9.74e-7 | 0 |
| `linear/ols_kappa11` | bse_HC3 | 6 | 8.78e+9 | 8.78e+9 | 1.08e-6 | 1.07e-6 | 0 |
| `linear/ols_kappa11` | bse_HAC | 6 | 6.71e+9 | 6.71e+9 | 8.26e-7 | 8.11e-7 | 0 |
| `linear/ols_kappa11` | bp_lm | 1 | 2.67e+9 | 2.67e+9 | 5.77e-7 |  | 0 |
| `linear/ols_kappa11` | bp_lm_pvalue | 1 | 2.89e+9 | 2.89e+9 | 4.84e-7 |  | 0 |
| `linear/ols_kappa11` | bp_fvalue | 1 | 3.33e+9 | 3.33e+9 | 5.77e-7 |  | 0 |
| `linear/ols_kappa11` | bp_f_pvalue | 1 | 2.59e+9 | 2.59e+9 | 4.13e-7 |  | 0 |
| `linear/ols_kappa11` | bp_lm_nonstudentized | 1 | 3.16e+9 | 3.16e+9 | 5.77e-7 |  | 0 |
| `linear/ols_kappa11` | bp_lm_nonstudentized_pvalue | 1 | 1.38e+9 | 1.38e+9 | 1.75e-7 |  | 0 |
| `linear/ols_kappa11` | fvalue | 1 | 1.38e+6 | 1.38e+6 | 2.04e-10 |  | 0 |
| `linear/ols_kappa11` | f_pvalue | 1 | 0.417 | 0.417 | 4.63e-17 |  | 0 |
| `linear/ols_noint` | params | 3 | 0.444 | 0.444 | 8.59e-17 | 7.80e-17 | 0 |
| `linear/ols_noint` | bse | 3 | 0.482 | 0.482 | 1.04e-16 | 7.75e-17 | 0 |
| `linear/ols_noint` | tvalues | 3 | 1.22 | 1.22 | 2.00e-16 | 9.33e-17 | 0 |
| `linear/ols_noint` | pvalues | 3 | 17.5 | 17.5 | 2.88e-15 | 1.19e-15 | 0 |
| `linear/ols_noint` | conf_int_lower | 3 | 1.2 | 1.2 | 1.94e-16 | 3.76e-17 | 0 |
| `linear/ols_noint` | conf_int_upper | 3 | 1.12 | 1.12 | 1.47e-16 | 1.12e-16 | 0 |
| `linear/ols_noint` | fittedvalues | 30 | 2.11 | 2.11 | 3.30e-16 | 1.18e-16 | 0 |
| `linear/ols_noint` | resid | 30 | 15.2 | 15.2 | 3.13e-15 | 1.81e-16 | 0 |
| `linear/ols_noint` | hat | 30 | 3.46 | 3.46 | 7.01e-16 | 1.49e-16 | 0 |
| `linear/ols_noint` | cooks_distance | 30 | 35.7 | 35.7 | 6.65e-15 | 4.16e-16 | 0 |
| `linear/ols_noint` | ssr | 1 | 0.0854 | 0.0854 | 1.29e-17 |  | 0 |
| `linear/ols_noint` | ess | 1 | 0.678 | 0.678 | 7.74e-17 |  | 0 |
| `linear/ols_noint` | centered_tss | 1 | 0.349 | 0.349 | 6.71e-17 |  | 0 |
| `linear/ols_noint` | rsquared | 1 | 0.0684 | 0.0684 | 9.04e-18 |  | 0 |
| `linear/ols_noint` | rsquared_adj | 1 | 0.146 | 0.146 | 1.97e-17 |  | 0 |
| `linear/ols_noint` | scale | 1 | 0.286 | 0.286 | 3.65e-17 |  | 0 |
| `linear/ols_noint` | llf | 1 | 1.5 | 1.5 | 2.73e-16 |  | 0 |
| `linear/ols_noint` | aic | 1 | 1.5 | 1.5 | 2.54e-16 |  | 0 |
| `linear/ols_noint` | bic | 1 | 2.01 | 2.01 | 3.25e-16 |  | 0 |
| `linear/ols_noint` | durbin_watson | 1 | 0.371 | 0.371 | 7.52e-17 |  | 0 |
| `linear/ols_noint` | jarque_bera | 1 | 2.87 | 2.87 | 3.23e-16 |  | 0 |
| `linear/ols_noint` | jarque_bera_pvalue | 1 | 0.323 | 0.323 | 4.81e-17 |  | 0 |
| `linear/ols_noint` | resid_studentized_internal | 30 | 17.7 | 17.7 | 3.19e-15 | 1.91e-16 | 0 |
| `linear/ols_noint` | resid_studentized_external | 30 | 17.1 | 17.1 | 3.13e-15 | 2.34e-16 | 0 |
| `linear/ols_noint` | bse_HC0 | 3 | 0.816 | 0.816 | 1.32e-16 | 8.48e-17 | 0 |
| `linear/ols_noint` | bse_HC1 | 3 | 0.535 | 0.535 | 8.22e-17 | 5.58e-17 | 0 |
| `linear/ols_noint` | bse_HC2 | 3 | 1 | 1 | 1.42e-16 | 1.07e-16 | 0 |
| `linear/ols_noint` | bse_HC3 | 3 | 0.174 | 0.174 | 2.17e-17 | 1.63e-17 | 0 |
| `linear/ols_noint` | bse_HAC | 3 | 1.15 | 1.15 | 2.10e-16 | 8.76e-17 | 0 |
| `linear/ols_noint` | bse_cluster | 3 | 0.212 | 0.212 | 2.50e-17 | 1.81e-17 | 0 |
| `linear/ols_noint` | pvalues_cluster | 3 | 2.9 | 2.9 | 4.08e-16 | 2.96e-16 | 0 |
| `linear/ols_noint` | fvalue_cluster | 1 | 2.16 | 2.16 | 3.23e-16 |  | 0 |
| `linear/ols_noint` | f_pvalue_cluster | 1 | 3.39 | 3.39 | 6.72e-16 |  | 0 |
| `linear/ols_noint` | bp_lm | 1 | 6.31 | 6.31 | 7.93e-16 |  | 0 |
| `linear/ols_noint` | bp_lm_pvalue | 1 | 1.6 | 1.6 | 2.14e-16 |  | 0 |
| `linear/ols_noint` | bp_fvalue | 1 | 3.93 | 3.93 | 8.30e-16 |  | 0 |
| `linear/ols_noint` | bp_f_pvalue | 1 | 0.776 | 0.776 | 1.01e-16 |  | 0 |
| `linear/ols_noint` | bp_lm_nonstudentized | 1 | 5.84 | 5.84 | 1.10e-15 |  | 0 |
| `linear/ols_noint` | bp_lm_nonstudentized_pvalue | 1 | 1.27 | 1.27 | 1.57e-16 |  | 0 |
| `linear/ols_noint` | fvalue | 1 | 0.93 | 0.93 | 1.39e-16 |  | 0 |
| `linear/ols_noint` | f_pvalue | 1 | 6.99 | 6.99 | 1.39e-15 |  | 0 |
| `linear/wls_noint` | params | 3 | 0.0769 | 0.0769 | 1.60e-17 | 1.43e-17 | 0 |
| `linear/wls_noint` | bse | 3 | 1.5 | 1.5 | 2.35e-16 | 1.41e-16 | 0 |
| `linear/wls_noint` | tvalues | 3 | 2.05 | 2.05 | 2.69e-16 | 2.60e-16 | 0 |
| `linear/wls_noint` | pvalues | 3 | 36.6 | 36.6 | 6.35e-15 | 9.34e-17 | 0 |
| `linear/wls_noint` | conf_int_lower | 3 | 0.667 | 0.667 | 8.67e-17 | 6.82e-17 | 0 |
| `linear/wls_noint` | conf_int_upper | 3 | 0.714 | 0.714 | 1.04e-16 | 8.78e-17 | 0 |
| `linear/wls_noint` | fittedvalues | 30 | 0.694 | 0.694 | 1.13e-16 | 6.60e-17 | 0 |
| `linear/wls_noint` | resid | 30 | 1.82e+03 | 1.82e+03 | 2.32e-13 | 6.61e-17 | 0 |
| `linear/wls_noint` | hat | 30 | 4.64 | 4.64 | 5.44e-16 | 1.33e-16 | 0 |
| `linear/wls_noint` | cooks_distance | 30 | 2.46e+03 | 2.46e+03 | 4.63e-13 | 2.74e-16 | 0 |
| `linear/wls_noint` | ssr | 1 | 0.371 | 0.371 | 6.75e-17 |  | 0 |
| `linear/wls_noint` | ess | 1 | 0.342 | 0.342 | 7.42e-17 |  | 0 |
| `linear/wls_noint` | centered_tss | 1 | 0.0458 | 0.0458 | 8.65e-18 |  | 0 |
| `linear/wls_noint` | rsquared | 1 | 0.422 | 0.422 | 5.39e-17 |  | 0 |
| `linear/wls_noint` | rsquared_adj | 1 | 0.136 | 0.136 | 1.76e-17 |  | 0 |
| `linear/wls_noint` | scale | 1 | 0.44 | 0.44 | 6.75e-17 |  | 0 |
| `linear/wls_noint` | llf | 1 | 0.0689 | 0.0689 | 1.20e-17 |  | 0 |
| `linear/wls_noint` | aic | 1 | 0.0689 | 0.0689 | 1.12e-17 |  | 0 |
| `linear/wls_noint` | bic | 1 | 0.582 | 0.582 | 8.99e-17 |  | 0 |
| `linear/wls_noint` | durbin_watson | 1 | 0.586 | 0.586 | 1.17e-16 |  | 0 |
| `linear/wls_noint` | jarque_bera | 1 | 14.3 | 14.3 | 1.87e-15 |  | 0 |
| `linear/wls_noint` | jarque_bera_pvalue | 1 | 2.72 | 2.72 | 3.73e-16 |  | 0 |
| `linear/wls_noint` | resid_studentized_internal | 30 | 1.2e+03 | 1.2e+03 | 2.32e-13 | 1.56e-16 | 0 |
| `linear/wls_noint` | resid_studentized_external | 30 | 1.17e+03 | 1.17e+03 | 2.32e-13 | 1.56e-16 | 0 |
| `linear/wls_noint` | bse_HC0 | 3 | 1.23 | 1.23 | 2.08e-16 | 1.65e-16 | 0 |
| `linear/wls_noint` | bse_HC1 | 3 | 1.34 | 1.34 | 2.15e-16 | 1.67e-16 | 0 |
| `linear/wls_noint` | bse_HC2 | 3 | 1.46 | 1.46 | 2.11e-16 | 1.69e-16 | 0 |
| `linear/wls_noint` | bse_HC3 | 3 | 1.27 | 1.27 | 1.56e-16 | 1.20e-16 | 0 |
| `linear/wls_noint` | bse_HAC | 3 | 1.35 | 1.35 | 2.27e-16 | 1.85e-16 | 0 |
| `linear/wls_noint` | bse_cluster | 3 | 2 | 2 | 2.24e-16 | 1.72e-16 | 0 |
| `linear/wls_noint` | pvalues_cluster | 3 | 5.6 | 5.6 | 8.62e-16 | 8.56e-16 | 0 |
| `linear/wls_noint` | fvalue_cluster | 1 | 5.6 | 5.6 | 1.00e-15 |  | 0 |
| `linear/wls_noint` | f_pvalue_cluster | 1 | 14.4 | 14.4 | 2.39e-15 |  | 0 |
| `linear/wls_noint` | bp_lm | 1 | 3.44 | 3.44 | 4.24e-16 |  | 0 |
| `linear/wls_noint` | bp_lm_pvalue | 1 | 2.95 | 2.95 | 5.32e-16 |  | 0 |
| `linear/wls_noint` | bp_fvalue | 1 | 2.63 | 2.63 | 4.94e-16 |  | 0 |
| `linear/wls_noint` | bp_f_pvalue | 1 | 4.95 | 4.95 | 8.18e-16 |  | 0 |
| `linear/wls_noint` | bp_lm_nonstudentized | 1 | 1.65 | 1.65 | 2.89e-16 |  | 0 |
| `linear/wls_noint` | bp_lm_nonstudentized_pvalue | 1 | 1.81 | 1.81 | 2.15e-16 |  | 0 |
| `linear/wls_noint` | fvalue | 1 | 1.32 | 1.32 | 1.56e-16 |  | 0 |
| `linear/wls_noint` | f_pvalue | 1 | 8.16 | 8.16 | 1.55e-15 |  | 0 |
| `linear/wls_zero` | params | 4 | 0.466 | 0.466 | 7.40e-17 | 5.53e-17 | 0 |
| `linear/wls_zero` | bse | 4 | 1.27 | 1.27 | 2.33e-16 | 1.23e-16 | 0 |
| `linear/wls_zero` | tvalues | 4 | 1.38 | 1.38 | 2.22e-16 | 8.59e-17 | 0 |
| `linear/wls_zero` | pvalues | 4 | 15.7 | 15.7 | 2.02e-15 | 1.27e-15 | 0 |
| `linear/wls_zero` | conf_int_lower | 4 | 10.9 | 10.9 | 1.27e-15 | 7.14e-17 | 0 |
| `linear/wls_zero` | conf_int_upper | 4 | 1.77 | 1.77 | 2.64e-16 | 1.75e-16 | 0 |
| `linear/wls_zero` | fittedvalues | 30 | 8.11 | 8.11 | 1.03e-15 | 7.53e-17 | 0 |
| `linear/wls_zero` | resid | 30 | 78.1 | 78.1 | 1.63e-14 | 1.64e-16 | 0 |
| `linear/wls_zero` | hat | 30 | 6.65 | 6.65 | 8.27e-16 | 2.06e-16 | 0 |
| `linear/wls_zero` | cooks_distance | 30 | 144 | 144 | 3.19e-14 | 3.65e-16 | 0 |
| `linear/wls_zero` | ssr | 1 | 0.0627 | 0.0627 | 9.04e-18 |  | 0 |
| `linear/wls_zero` | ess | 1 | 0.0728 | 0.0728 | 1.09e-17 |  | 0 |
| `linear/wls_zero` | centered_tss | 1 | 0.0649 | 0.0649 | 8.63e-18 |  | 0 |
| `linear/wls_zero` | rsquared | 1 | 0.248 | 0.248 | 3.11e-17 |  | 0 |
| `linear/wls_zero` | rsquared_adj | 1 | 0.0287 | 0.0287 | 3.66e-18 |  | 0 |
| `linear/wls_zero` | scale | 1 | 0.292 | 0.292 | 6.30e-17 |  | 0 |
| `linear/wls_zero` | llf | 1 | 0.82 | 0.82 | 1.77e-16 |  | 0 |
| `linear/wls_zero` | aic | 1 | 0.82 | 0.82 | 1.58e-16 |  | 0 |
| `linear/wls_zero` | bic | 1 | 1.23 | 1.23 | 2.20e-16 |  | 0 |
| `linear/wls_zero` | durbin_watson | 1 | 0.158 | 0.158 | 3.11e-17 |  | 0 |
| `linear/wls_zero` | jarque_bera | 1 | 7.91 | 7.91 | 1.26e-15 |  | 0 |
| `linear/wls_zero` | jarque_bera_pvalue | 1 | 3.37 | 3.37 | 5.31e-16 |  | 0 |
| `linear/wls_zero` | resid_studentized_internal | 30 | 125 | 125 | 1.62e-14 | 1.80e-16 | 0 |
| `linear/wls_zero` | resid_studentized_external | 30 | 122 | 122 | 1.62e-14 | 2.46e-16 | 0 |
| `linear/wls_zero` | bse_HC0 | 4 | 1.85 | 1.85 | 2.27e-16 | 1.79e-16 | 0 |
| `linear/wls_zero` | bse_HC1 | 4 | 2.15 | 2.15 | 2.44e-16 | 1.79e-16 | 0 |
| `linear/wls_zero` | bse_HC2 | 4 | 0.95 | 0.95 | 2.09e-16 | 1.37e-16 | 0 |
| `linear/wls_zero` | bse_HC3 | 4 | 1.01 | 1.01 | 2.00e-16 | 1.27e-16 | 0 |
| `linear/wls_zero` | bse_HAC | 4 | 1.79 | 1.79 | 2.11e-16 | 1.38e-16 | 0 |
| `linear/wls_zero` | bse_cluster | 4 | 3.09 | 3.09 | 4.37e-16 | 2.20e-16 | 0 |
| `linear/wls_zero` | pvalues_cluster | 4 | 8.58 | 8.58 | 1.03e-15 | 7.85e-16 | 0 |
| `linear/wls_zero` | fvalue_cluster | 1 | 4.39 | 4.39 | 6.58e-16 |  | 0 |
| `linear/wls_zero` | f_pvalue_cluster | 1 | 8.4 | 8.4 | 1.66e-15 |  | 0 |
| `linear/wls_zero` | bp_lm | 1 | 2.09 | 2.09 | 3.58e-16 |  | 0 |
| `linear/wls_zero` | bp_lm_pvalue | 1 | 3.64 | 3.64 | 7.50e-16 |  | 0 |
| `linear/wls_zero` | bp_fvalue | 1 | 1.88 | 1.88 | 3.20e-16 |  | 0 |
| `linear/wls_zero` | bp_f_pvalue | 1 | 0.434 | 0.434 | 8.08e-17 |  | 0 |
| `linear/wls_zero` | bp_lm_nonstudentized | 1 | 0.432 | 0.432 | 5.24e-17 |  | 0 |
| `linear/wls_zero` | bp_lm_nonstudentized_pvalue | 1 | 0.867 | 0.867 | 1.06e-16 |  | 0 |
| `linear/wls_zero` | fvalue | 1 | 0.704 | 0.704 | 8.16e-17 |  | 0 |
| `linear/wls_zero` | f_pvalue | 1 | 6.4 | 6.4 | 9.76e-16 |  | 0 |
| `penalized/binomial_0` | params | 7 | 2.79 | 2.79 | 4.31e-16 | 1.02e-16 | 0 |
| `penalized/binomial_0` | deviance | 1 | 0.437 | 0.437 | 6.24e-17 |  | 0 |
| `penalized/binomial_0.5` | params | 7 | 1.51 | 1.51 | 2.42e-16 | 9.36e-17 | 0 |
| `penalized/binomial_0.5` | deviance | 1 | 0.141 | 0.141 | 2.07e-17 |  | 0 |
| `penalized/binomial_1` | params | 7 | 20.6 | 20.6 | 2.38e-15 | 1.87e-16 | 0 |
| `penalized/binomial_1` | deviance | 1 | 0.384 | 0.384 | 5.75e-17 |  | 0 |
| `penalized/binomial_weights_std` | params | 7 | 2.3 | 2.3 | 4.41e-16 | 2.38e-16 | 0 |
| `penalized/binomial_weights_std` | deviance | 1 | 0.131 | 0.131 | 1.95e-17 |  | 0 |
| `penalized/enet` | params | 7 | 48.3 | 48.3 | 6.79e-15 | 1.12e-15 | 0 |
| `penalized/enet` | ssr | 1 | 1.36 | 1.36 | 2.88e-16 |  | 0 |
| `penalized/enet_kappa1e6_refined` | params | 7 | 11.8 | 11.8 | 1.67e-15 | 5.33e-17 | 0 |
| `penalized/enet_kappa1e6_refined` | ssr | 1 | 2.51 | 2.51 | 4.31e-16 |  | 0 |
| `penalized/enet_noint` | params | 6 | 43 | 43 | 7.71e-15 | 6.87e-16 | 0 |
| `penalized/enet_noint` | ssr | 1 | 2.97 | 2.97 | 5.36e-16 |  | 0 |
| `penalized/enet_std` | params | 7 | 27.2 | 27.2 | 3.12e-15 | 9.92e-16 | 0 |
| `penalized/enet_std` | ssr | 1 | 8.87 | 8.87 | 1.12e-15 |  | 0 |
| `penalized/enet_weights` | params | 7 | 48.8 | 48.8 | 9.88e-15 | 1.46e-15 | 0 |
| `penalized/enet_weights` | ssr | 1 | 2.18 | 2.18 | 3.02e-16 |  | 0 |
| `penalized/lasso` | params | 7 | 189 | 189 | 4.08e-14 | 1.30e-15 | 0 |
| `penalized/lasso` | ssr | 1 | 3.26 | 3.26 | 3.66e-16 |  | 0 |
| `penalized/lasso_constant_column` | params | 7 | 23.8 | 23.8 | 5.19e-15 | 9.07e-16 | 0 |
| `penalized/lasso_constant_column` | ssr | 1 | 0.287 | 0.287 | 6.32e-17 |  | 0 |
| `penalized/lasso_kappa1e6` | params | 7 | 0.463 | 0.463 | 7.61e-17 | 7.59e-17 | 0 |
| `penalized/lasso_kappa1e6` | ssr | 1 | 0.732 | 0.732 | 1.26e-16 |  | 0 |
| `penalized/lasso_small_alpha` | params | 7 | 514 | 514 | 7.23e-14 | 1.30e-15 | 0 |
| `penalized/lasso_small_alpha` | ssr | 1 | 1.84 | 1.84 | 2.20e-16 |  | 0 |
| `penalized/lasso_weights_std` | params | 7 | 47.5 | 47.5 | 7.69e-15 | 1.33e-15 | 0 |
| `penalized/lasso_weights_std` | ssr | 1 | 2.21 | 2.21 | 2.88e-16 |  | 0 |
| `penalized/poisson_0` | params | 7 | 13.1 | 13.1 | 2.40e-15 | 1.05e-16 | 0 |
| `penalized/poisson_0` | deviance | 1 | 0.281 | 0.281 | 4.42e-17 |  | 0 |
| `penalized/poisson_0.5` | params | 7 | 6.49 | 6.49 | 1.25e-15 | 2.79e-16 | 0 |
| `penalized/poisson_0.5` | deviance | 1 | 0.21 | 0.21 | 3.29e-17 |  | 0 |
| `penalized/poisson_1` | params | 7 | 2.07 | 2.07 | 3.02e-16 | 1.44e-16 | 0 |
| `penalized/poisson_1` | deviance | 1 | 0.017 | 0.017 | 2.66e-18 |  | 0 |
| `penalized/ridge` | params | 7 | 848 | 848 | 1.35e-13 | 8.30e-16 | 0 |
| `penalized/ridge` | ssr | 1 | 4.46 | 4.46 | 5.27e-16 |  | 0 |
| `penalized/ridge_kappa1e6` | params | 7 | 1.56 | 1.56 | 2.13e-16 | 1.36e-17 | 0 |
| `penalized/ridge_kappa1e6` | ssr | 1 | 0.767 | 0.767 | 1.34e-16 |  | 0 |
| `penalized/ridge_noint` | params | 6 | 40.4 | 40.4 | 6.06e-15 | 5.08e-16 | 0 |
| `penalized/ridge_noint` | ssr | 1 | 0.205 | 0.205 | 4.05e-17 |  | 0 |
| `penalized/ridge_weights_std` | params | 7 | 185 | 185 | 2.59e-14 | 1.23e-15 | 0 |
| `penalized/ridge_weights_std` | ssr | 1 | 2.29 | 2.29 | 3.08e-16 |  | 0 |
| `robust/bisquare_huber_even` | params | 3 | 0.812 | 0.812 | 1.40e-16 | 8.40e-17 | 0 |
| `robust/bisquare_huber_even` | scale | 1 | 1.41 | 1.41 | 1.99e-16 |  | 0 |
| `robust/bisquare_huber_even` | weights | 60 | 2.78 | 2.78 | 3.13e-16 | 1.58e-16 | 0 |
| `robust/bisquare_huber_even` | bse | 3 | 1.06 | 1.06 | 2.10e-16 | 1.27e-16 | 0 |
| `robust/bisquare_huber_odd` | params | 4 | 1.21 | 1.21 | 2.01e-16 | 8.16e-17 | 0 |
| `robust/bisquare_huber_odd` | scale | 1 | 0.733 | 0.733 | 9.10e-17 |  | 0 |
| `robust/bisquare_huber_odd` | weights | 61 | 7.54 | 7.54 | 8.60e-16 | 1.72e-16 | 0 |
| `robust/bisquare_huber_odd` | bse | 4 | 2.28 | 2.28 | 2.97e-16 | 1.81e-16 | 0 |
| `robust/bisquare_mad_400` | params | 5 | 0.449 | 0.449 | 5.02e-17 | 3.46e-17 | 0 |
| `robust/bisquare_mad_400` | scale | 1 | 0.619 | 0.619 | 1.25e-16 |  | 0 |
| `robust/bisquare_mad_400` | weights | 400 | 3.31 | 3.31 | 4.78e-16 | 1.54e-16 | 0 |
| `robust/bisquare_mad_400` | bse | 5 | 3.73 | 3.73 | 4.81e-16 | 3.75e-16 | 0 |
| `robust/bisquare_mad_even` | params | 3 | 0.418 | 0.418 | 9.04e-17 | 6.96e-17 | 0 |
| `robust/bisquare_mad_even` | scale | 1 | 1.01 | 1.01 | 1.90e-16 |  | 0 |
| `robust/bisquare_mad_even` | weights | 60 | 3.11 | 3.11 | 4.69e-16 | 1.73e-16 | 0 |
| `robust/bisquare_mad_even` | bse | 3 | 0.887 | 0.887 | 1.54e-16 | 1.41e-16 | 0 |
| `robust/bisquare_mad_odd` | params | 4 | 1.96 | 1.96 | 3.50e-16 | 7.59e-17 | 0 |
| `robust/bisquare_mad_odd` | scale | 1 | 0.0276 | 0.0276 | 4.48e-18 |  | 0 |
| `robust/bisquare_mad_odd` | weights | 61 | 3.35 | 3.35 | 3.72e-16 | 1.60e-16 | 0 |
| `robust/bisquare_mad_odd` | bse | 4 | 1.38 | 1.38 | 2.11e-16 | 1.39e-16 | 0 |
| `robust/hampel_huber_even` | params | 3 | 1.98 | 1.98 | 2.42e-16 | 9.76e-17 | 0 |
| `robust/hampel_huber_even` | scale | 1 | 1.15 | 1.15 | 1.59e-16 |  | 0 |
| `robust/hampel_huber_even` | weights | 60 | 15.8 | 15.8 | 3.43e-15 | 4.07e-17 | 0 |
| `robust/hampel_huber_even` | bse | 3 | 1.21 | 1.21 | 1.84e-16 | 1.34e-16 | 0 |
| `robust/hampel_huber_odd` | params | 4 | 0.78 | 0.78 | 1.27e-16 | 8.69e-17 | 0 |
| `robust/hampel_huber_odd` | scale | 1 | 0.82 | 0.82 | 9.91e-17 |  | 0 |
| `robust/hampel_huber_odd` | weights | 61 | 1.37 | 1.37 | 2.86e-16 | 3.21e-17 | 0 |
| `robust/hampel_huber_odd` | bse | 4 | 2.22 | 2.22 | 4.19e-16 | 2.65e-16 | 0 |
| `robust/hampel_mad_even` | params | 3 | 1.07 | 1.07 | 1.72e-16 | 4.86e-17 | 0 |
| `robust/hampel_mad_even` | scale | 1 | 0.316 | 0.316 | 5.87e-17 |  | 0 |
| `robust/hampel_mad_even` | weights | 60 | 5.4 | 5.4 | 7.86e-16 | 5.48e-18 | 0 |
| `robust/hampel_mad_even` | bse | 3 | 2.08 | 2.08 | 4.05e-16 | 3.00e-16 | 0 |
| `robust/hampel_mad_odd` | params | 4 | 0.602 | 0.602 | 1.15e-16 | 5.72e-17 | 0 |
| `robust/hampel_mad_odd` | scale | 1 | 0.369 | 0.369 | 6.34e-17 |  | 0 |
| `robust/hampel_mad_odd` | weights | 61 | 5.17 | 5.17 | 8.45e-16 | 1.21e-17 | 0 |
| `robust/hampel_mad_odd` | bse | 4 | 1.29 | 1.29 | 1.94e-16 | 1.54e-16 | 0 |
| `robust/huber_huber_even` | params | 3 | 0.26 | 0.26 | 3.11e-17 | 2.42e-17 | 0 |
| `robust/huber_huber_even` | scale | 1 | 1.07 | 1.07 | 1.41e-16 |  | 0 |
| `robust/huber_huber_even` | weights | 60 | 1.77 | 1.77 | 3.14e-16 | 2.98e-17 | 0 |
| `robust/huber_huber_even` | bse | 3 | 0.57 | 0.57 | 7.97e-17 | 7.17e-17 | 0 |
| `robust/huber_huber_odd` | params | 4 | 0.674 | 0.674 | 1.45e-16 | 7.98e-17 | 0 |
| `robust/huber_huber_odd` | scale | 1 | 1.27 | 1.27 | 1.56e-16 |  | 0 |
| `robust/huber_huber_odd` | weights | 61 | 1.92 | 1.92 | 3.76e-16 | 5.19e-17 | 0 |
| `robust/huber_huber_odd` | bse | 4 | 0.821 | 0.821 | 1.24e-16 | 5.61e-17 | 0 |
| `robust/huber_mad_even` | params | 3 | 0.499 | 0.499 | 9.25e-17 | 4.94e-17 | 0 |
| `robust/huber_mad_even` | scale | 1 | 0.163 | 0.163 | 3.01e-17 |  | 0 |
| `robust/huber_mad_even` | weights | 60 | 0.983 | 0.983 | 1.23e-16 | 1.85e-17 | 0 |
| `robust/huber_mad_even` | bse | 3 | 0.168 | 0.168 | 2.46e-17 | 1.67e-17 | 0 |
| `robust/huber_mad_noint` | params | 3 | 0.327 | 0.327 | 6.68e-17 | 5.65e-17 | 0 |
| `robust/huber_mad_noint` | scale | 1 | 0.101 | 0.101 | 1.62e-17 |  | 0 |
| `robust/huber_mad_noint` | weights | 61 | 0.98 | 0.98 | 1.22e-16 | 1.05e-17 | 0 |
| `robust/huber_mad_noint` | bse | 3 | 2.06 | 2.06 | 3.15e-16 | 2.72e-16 | 0 |
| `robust/huber_mad_odd` | params | 4 | 0.654 | 0.654 | 1.38e-16 | 5.05e-17 | 0 |
| `robust/huber_mad_odd` | scale | 1 | 1.16 | 1.16 | 1.91e-16 |  | 0 |
| `robust/huber_mad_odd` | weights | 61 | 2.46 | 2.46 | 2.85e-16 | 3.92e-17 | 0 |
| `robust/huber_mad_odd` | bse | 4 | 1.21 | 1.21 | 1.75e-16 | 7.87e-17 | 0 |
| `robust/mm_leverage` | params | 3 | 0.482 | 0.482 | 1.01e-16 | 5.00e-17 | 0 |
| `robust/mm_leverage` | scale | 1 | 0 | 0 | 0 |  | 0 |
| `robust/mm_leverage` | weights | 120 | 3.29 | 3.29 | 3.75e-16 | 1.71e-16 | 0 |
| `robust/mm_leverage` | bse | 3 | 8.14 | 8.14 | 1.22e-15 | 7.66e-16 | 0 |
| `robust/mm_stackloss` | params | 4 | 0.827 | 0.827 | 1.38e-16 | 8.74e-17 | 0 |
| `robust/mm_stackloss` | scale | 1 | 2 | 2 | 2.32e-16 |  | 0 |
| `robust/mm_stackloss` | weights | 21 | 12.5 | 12.5 | 1.43e-15 | 3.14e-16 | 0 |
| `robust/mm_stackloss` | bse | 4 | 3.8 | 3.8 | 6.60e-16 | 6.37e-16 | 0 |
| `robust/stackloss_bisquare` | params | 4 | 1.14 | 1.14 | 1.94e-16 | 7.16e-17 | 0 |
| `robust/stackloss_bisquare` | scale | 1 | 7.31 | 7.31 | 1.42e-15 |  | 0 |
| `robust/stackloss_bisquare` | weights | 21 | 710 | 710 | 1.39e-13 | 4.01e-16 | 0 |
| `robust/stackloss_bisquare` | bse | 4 | 10 | 10 | 1.12e-15 | 7.81e-16 | 0 |
| `robust/stackloss_hampel` | params | 4 | 0.845 | 0.845 | 1.27e-16 | 1.12e-16 | 0 |
| `robust/stackloss_hampel` | scale | 1 | 2.82 | 2.82 | 4.05e-16 |  | 0 |
| `robust/stackloss_hampel` | weights | 21 | 2.86 | 2.86 | 3.93e-16 | 6.98e-17 | 0 |
| `robust/stackloss_hampel` | bse | 4 | 5.04 | 5.04 | 1.04e-15 | 1.35e-16 | 0 |
| `robust/stackloss_huber` | params | 4 | 2.33 | 2.33 | 2.79e-16 | 9.58e-17 | 0 |
| `robust/stackloss_huber` | scale | 1 | 3.54 | 3.54 | 6.44e-16 |  | 0 |
| `robust/stackloss_huber` | weights | 21 | 4.96 | 4.96 | 7.01e-16 | 1.48e-16 | 0 |
| `robust/stackloss_huber` | bse | 4 | 7.71 | 7.71 | 9.64e-16 | 1.09e-16 | 0 |
