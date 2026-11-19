# sd.special 정확도

`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/special.json에서 만든다. 손으로 고치지 않는다.

참값은 tests/special/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다.

- ULP: |sd − 참값|을 참값에서의 float64 간격으로 나눈 값. 99.9%는 그 분위수다.
- ULP/max(1, κ): 역함수의 계약 단위(κ는 입력에 대한 상대 조건수). 역함수만 있다.
- mixed: |sd − 참값|/max(1, |참값|)를 2⁻⁵² 단위로 잰 값(45가 1e-14다). 0을 지나는 값(log 밀도, 엔트로피, gammaln 등)의 계약 척도이고, 그 함수에만 있다. 최대와 99.9%다.
- 상대오차: |sd − 참값|/|참값|.
- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 |오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.
- 영역: guaranteed는 보장 영역, loose와 mixed는 그 밖(mixed는 그 척도로 잰다).

측정: 20261120-053742-b640cf3 (2026-11-20T05:37:42, b640cf3 dirty).

| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | ULP/max(1, κ) | mixed | 최대 상대오차 | 계약 대비 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `_cf_tan` | guaranteed | 6038 | 6.71 | 5.44 |  |  | 8.52e-16 | 0.085 |
| `_cf_tan` | loose | 2026 | 5.19e+15 | 2.62e+15 |  |  | 0.835 | 0.83 |
| `airy_ai` | guaranteed | 2975 | 5.06e+03 | 430 | 57.3 |  | 9.12e-13 | 0.9 |
| `airy_aip` | guaranteed | 2975 | 2.5e+03 | 1.22e+03 | 34.8 |  | 4.06e-13 | 0.59 |
| `airy_bi` | guaranteed | 2975 | 1.95e+03 | 476 | 43.5 |  | 2.26e-13 | 0.59 |
| `airy_bip` | guaranteed | 2975 | 1.79e+03 | 416 | 50.3 |  | 2.58e-13 | 0.9 |
| `bdtr` | guaranteed | 1200 | 4.94 | 4.84 |  |  | 6.98e-16 | 0.07 |
| `bdtrc` | guaranteed | 1200 | 7.44 | 6.58 |  |  | 9.63e-16 | 0.096 |
| `beta` | guaranteed | 4752 | 52.8 | 34.5 |  |  | 8.22e-15 | 0.82 |
| `beta` | mixed | 748 | 10.5 | 10.5 |  | 6.78 / 6.78 | 1.50e-15 | 0.15 |
| `betainc` | guaranteed | 2004 | 21.8 | 14.2 |  |  | 2.51e-15 | 0.25 |
| `betaincc` | guaranteed | 2004 | 18.6 | 17.4 |  |  | 2.66e-15 | 0.27 |
| `betainccinv` | guaranteed | 1502 | 3.98e+03 | 3.43e+03 | 9.55 |  | 7.05e-13 | 0.11 |
| `betaincinv` | guaranteed | 1502 | 3.43e+03 | 2.96e+03 | 9.2 |  | 4.08e-13 | 0.13 |
| `betaln` | mixed | 4500 | 1.10e+12 | 3.44e+11 |  | 27.3 / 12.9 | 2.10e-4 | 0.61 |
| `binom` | guaranteed | 5480 | 9.26 | 8.39 |  |  | 1.61e-15 | 0.16 |
| `binom` | mixed | 1120 | 14.5 | 10.7 |  | 7.98 / 6.99 | 1.77e-15 | 0.18 |
| `chdtr` | guaranteed | 1200 | 5.3 | 5.05 |  |  | 6.72e-16 | 0.067 |
| `chdtrc` | guaranteed | 1200 | 9.69 | 7.93 |  |  | 1.46e-15 | 0.15 |
| `chdtri` | guaranteed | 1200 | 507 | 414 | 5.39 |  | 6.95e-14 | 0.071 |
| `chndtr` | guaranteed | 1200 | 4.67 | 3.98 |  |  | 5.20e-16 | 0.052 |
| `chndtrc` | guaranteed | 1187 | 9.52 | 7.95 |  |  | 1.37e-15 | 0.14 |
| `chndtrix` | guaranteed | 953 | 152 | 152 | 3.48 |  | 0.0769 | 0.053 |
| `digamma` | guaranteed | 4136 | 2.11 | 1.66 |  |  | 4.06e-16 | 0.041 |
| `digamma` | mixed | 1695 | 77.8 | 49.2 |  | 14.9 / 9.24 | 1.21e-14 | 0.33 |
| `ellipe` | guaranteed | 3036 | 2.04 | 1.36 |  |  | 2.88e-16 | 0.029 |
| `ellipk` | guaranteed | 3036 | 2.3 | 1.92 |  |  | 3.18e-16 | 0.032 |
| `entr` | guaranteed | 5024 | 1.25 | 0.785 |  |  | 4.27e-16 | 0.022 |
| `erf` | guaranteed | 7564 | 2.29 | 1.9 |  |  | 2.82e-16 | 0.028 |
| `erf_complex` | guaranteed | 5498 | 3.43e+13 | 3.08e+13 | 3.44 |  | 0.00458 | 0.047 |
| `erfc` | guaranteed | 8073 | 7.77 | 4.71 |  |  | 1 | 0.11 |
| `erfc_complex` | guaranteed | 5402 | 1.94e+13 | 1.19e+13 | 3.44 |  | 0.00313 | 0.047 |
| `erfcinv` | guaranteed | 7550 | 4.51 | 3.83 |  |  | 7.40e-16 | 0.074 |
| `erfcx` | guaranteed | 7555 | 6.05 | 3.81 |  |  | 7.55e-16 | 0.075 |
| `erfcx_complex` | guaranteed | 5402 | 2.52 | 1.85 | 1.94 |  | 4.54e-16 | 0.028 |
| `erfinv` | guaranteed | 7563 | 4.64 | 3.9 |  |  | 8.34e-15 | 0.054 |
| `expi` | guaranteed | 3313 | 4.69 | 3.92 |  |  | 7.61e-16 | 0.076 |
| `expit` | guaranteed | 8082 | 2.06 | 1.73 |  |  | 1.28e-15 | 0.033 |
| `expn` | guaranteed | 2705 | 6.28 | 5.03 |  |  | 8.25e-16 | 0.082 |
| `fdtr` | guaranteed | 1500 | 12.4 | 7.92 |  |  | 1.47e-15 | 0.15 |
| `fdtrc` | guaranteed | 1500 | 10.7 | 9.11 |  |  | 1.31e-15 | 0.13 |
| `fdtri` | guaranteed | 1007 | 5.7e+03 | 1.97e+03 | 23 |  | 8.19e-13 | 0.29 |
| `gamma` | guaranteed | 7019 | 10.2 | 7.35 |  |  | 1.16e-15 | 0.12 |
| `gammainc` | guaranteed | 2413 | 8.87 | 5.72 |  |  | 1.43e-15 | 0.14 |
| `gammaincc` | guaranteed | 2413 | 13.4 | 8.02 |  |  | 1.70e-15 | 0.17 |
| `gammainccinv` | guaranteed | 2004 | 1.68e+03 | 581 | 5.33 |  | 2.31e-13 | 0.077 |
| `gammaincinv` | guaranteed | 2004 | 1.3e+03 | 625 | 5.32 |  | 1.99e-13 | 0.073 |
| `gammaln` | guaranteed | 4937 | 5.53 | 2.53 |  |  | 6.36e-16 | 0.064 |
| `gammaln` | mixed | 1732 | 4.45e+15 | 1.98e+15 |  | 2.09 / 1.89 | 0.976 | 0.046 |
| `hyp1f1` | guaranteed | 1600 | 15.9 | 9.26 |  |  | 1.81e-15 | 0.18 |
| `hyp1f1` | loose | 400 | 1.17e+03 | 1.17e+03 |  |  | 2.57e-13 | 0.26 |
| `hyp2f1` | guaranteed | 509 | 11 | 11 |  |  | 1.43e-15 | 0.14 |
| `hyp2f1` | loose | 991 | 2.73e+03 | 2.73e+03 |  |  | 3.55e-13 | 0.36 |
| `i0` | guaranteed | 2400 | 3.5 | 2.97 |  |  | 4.07e-16 | 0.041 |
| `i0e` | guaranteed | 2400 | 1.79 | 1.65 |  |  | 3.01e-16 | 0.03 |
| `i1` | guaranteed | 2400 | 3.47 | 3.11 |  |  | 4.38e-16 | 0.044 |
| `i1e` | guaranteed | 2400 | 1.65 | 1.59 |  |  | 2.40e-16 | 0.024 |
| `iv` | guaranteed | 2000 | 38.8 | 32 | 38.8 |  | 5.31e-15 | 0.53 |
| `ive` | guaranteed | 2000 | 44.6 | 42.9 |  |  | 5.05e-15 | 0.5 |
| `j0` | guaranteed | 2100 | 455 | 82.9 | 10.9 |  | 5.89e-14 | 0.13 |
| `j1` | guaranteed | 2100 | 269 | 60.4 | 9.32 |  | 3.09e-14 | 0.16 |
| `jv` | guaranteed | 2000 | 460 | 340 | 23.2 |  | 5.20e-14 | 0.26 |
| `k0` | guaranteed | 2400 | 2.3 | 1.96 |  |  | 4.03e-15 | 0.035 |
| `k0e` | guaranteed | 2400 | 1.83 | 1.7 |  |  | 2.66e-16 | 0.027 |
| `k1` | guaranteed | 2400 | 2.13 | 1.95 |  |  | 4.63e-16 | 0.038 |
| `k1e` | guaranteed | 2400 | 2.05 | 1.96 |  |  | 3.03e-16 | 0.03 |
| `kv` | guaranteed | 2000 | 22.2 | 16.3 |  |  | 2.62e-15 | 0.26 |
| `kve` | guaranteed | 2000 | 16 | 13.5 |  |  | 2.34e-15 | 0.23 |
| `lambertw` | guaranteed | 3000 | 1.89 | 1.1 |  |  | 2.48e-16 | 0.025 |
| `lambertw_c0` | guaranteed | 5138 | 1.86 | 1.37 |  |  | 2.84e-16 | 0.028 |
| `lambertw_c1` | guaranteed | 5138 | 1.76 | 0.955 |  |  | 2.80e-16 | 0.028 |
| `lambertw_c4` | guaranteed | 5138 | 0.947 | 0.704 |  |  | 1.58e-16 | 0.016 |
| `lambertw_cm1` | guaranteed | 5138 | 1.75 | 1 |  |  | 2.28e-16 | 0.023 |
| `lambertw_m1` | guaranteed | 2250 | 2.63 | 1.33 |  |  | 3.09e-16 | 0.031 |
| `log_expit` | guaranteed | 8082 | 1.24 | 0.938 |  |  | 1.95e-16 | 0.019 |
| `log_ndtr` | guaranteed | 8064 | 6.97 | 4.45 |  |  | 1 | 0.08 |
| `loggamma_complex` | guaranteed | 7363 | 8.21 | 5.93 |  |  | 1.01e-15 | 0.1 |
| `logit` | guaranteed | 7535 | 2.8 | 2.55 |  |  | 3.74e-16 | 0.037 |
| `nbdtr` | guaranteed | 1200 | 7.21 | 6.74 |  |  | 9.21e-16 | 0.092 |
| `nbdtrc` | guaranteed | 1200 | 8.79 | 8.48 |  |  | 1.71e-15 | 0.17 |
| `ncfdtr` | guaranteed | 1200 | 6.57 | 6.2 |  |  | 8.23e-16 | 0.082 |
| `ncfdtrc` | guaranteed | 1200 | 15 | 10.3 |  |  | 1.68e-15 | 0.17 |
| `ncfdtri` | guaranteed | 955 | 496 | 496 | 9.88 |  | 6.36e-14 | 0.14 |
| `nctdtr` | guaranteed | 1200 | 4.77 | 4.71 |  |  | 6.67e-16 | 0.067 |
| `nctdtrc` | guaranteed | 1198 | 6.73 | 5.58 |  |  | 1.25e-15 | 0.13 |
| `nctdtrit` | guaranteed | 1076 | 250 | 159 | 9.8 |  | 3.75e-14 | 0.11 |
| `ndtr` | guaranteed | 8064 | 6.73 | 5.3 |  |  | 1 | 0.088 |
| `ndtri` | guaranteed | 7559 | 5.43 | 4.56 |  |  | 6.64e-16 | 0.066 |
| `ndtri_exp` | guaranteed | 2427 | 4.72e+15 | 1.18e+15 | 6.53 |  | 1 | 0.079 |
| `pdtr` | guaranteed | 1400 | 4.26 | 3.74 |  |  | 6.56e-16 | 0.066 |
| `pdtrc` | guaranteed | 1400 | 13.1 | 7.7 |  |  | 9.48e-4 | 0.15 |
| `polygamma` | guaranteed | 3918 | 17.3 | 8.09 |  |  | 3.56e-15 | 0.36 |
| `polygamma` | mixed | 974 | 5.01 | 5.01 |  | 3.13 / 3.13 | 6.95e-16 | 0.069 |
| `rel_entr` | guaranteed | 5000 | 2.07 | 1.64 |  |  | 2.74e-16 | 0.027 |
| `rgamma` | guaranteed | 7719 | 5.11 | 4.04 |  |  | 1.17e-14 | 0.076 |
| `stdtr` | guaranteed | 1506 | 8.95 | 8.21 |  |  | 1.58e-15 | 0.16 |
| `stdtrit` | guaranteed | 1244 | 7.16e+5 | 5.34e+03 | 7.35 |  | 1.21e-10 | 0.082 |
| `wofz` | guaranteed | 7401 | 1.95e+13 | 1.32e+13 | 2.01 |  | 0.00272 | 0.031 |
| `xlog1py` | guaranteed | 4000 | 1.24 | 1.13 |  |  | 1.72e-16 | 0.017 |
| `xlogy` | guaranteed | 4000 | 1.19 | 1.12 |  |  | 1.86e-16 | 0.019 |
| `y0` | guaranteed | 2100 | 237 | 136 | 21.1 |  | 2.84e-14 | 0.24 |
| `y1` | guaranteed | 2100 | 454 | 64.8 | 46.4 |  | 5.44e-14 | 0.69 |
| `yv` | guaranteed | 2000 | 393 | 258 | 13.8 |  | 5.44e-14 | 0.24 |
| `zeta` | guaranteed | 6112 | 9.42 | 7.98 |  |  | 1.05e-15 | 0.1 |
| `zeta_hurwitz` | guaranteed | 3500 | 5.51 | 3 |  |  | 7.76e-16 | 0.078 |
| `zeta_hurwitz` | mixed | 800 | 71 | 71 |  | 6.23 / 6.23 | 1.49e-14 | 0.14 |
