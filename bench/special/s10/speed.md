Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gamma` | `scipy.special.gamma` | 19.997 | 2.883 | 17.641 | 0.88 | 6.12 | 0.164 | 0.078 |
| `digamma` | `scipy.special.digamma` | 9.605 | 1.571 | 11.832 | 1.23 | 7.53 | 0.145 | 0.068 |
| `beta` | `scipy.special.beta` | 63.830 | 8.424 | 35.072 | 0.55 | 4.16 | 0.191 | 0.341 |
| `zeta` | `scipy.special.zeta` | 17.689 | 2.846 | 17.067 | 0.96 | 6.00 | 0.167 | 0.258 |
| `gammaincinv` | `scipy.special.gammaincinv` | 369.210 | 45.000 | 493.380 | 1.34 | 10.96 | 0.238 | 0.383 |
| `gammainccinv` | `scipy.special.gammainccinv` | 302.439 | 36.495 | 294.121 | 0.97 | 8.06 | 0.235 | 0.382 |
| `betainc` | `scipy.special.betainc` | 402.807 | 50.076 | 162.533 | 0.40 | 3.25 | 0.561 | 0.474 |
| `betaincc` | `scipy.special.betaincc` | 398.841 | 47.411 | 163.400 | 0.41 | 3.45 | 0.290 | 0.442 |
| `betaincinv` | `scipy.special.betaincinv` | 1021.888 | 124.862 | 1242.487 | 1.22 | 9.95 | 0.381 | 0.648 |
| `betainccinv` | `scipy.special.betainccinv` | 1161.325 | 144.458 | 972.051 | 0.84 | 6.73 | 1.649 | 0.942 |
| `i0` | `scipy.special.i0` | 4.322 | 0.819 | 12.862 | 2.98 | 15.70 | 0.140 | 0.091 |
| `i1` | `scipy.special.i1` | 4.650 | 0.857 | 12.970 | 2.79 | 15.13 | 0.144 | 0.089 |
| `k0` | `scipy.special.k0` | 11.286 | 1.885 | 17.930 | 1.59 | 9.51 | 0.146 | 0.093 |
| `k1` | `scipy.special.k1` | 11.443 | 1.937 | 18.514 | 1.62 | 9.56 | 0.145 | 0.093 |
| `iv` | `scipy.special.iv` | 1898.989 | 222.499 | 211.288 | 0.11 | 0.95 | 3.254 | 0.362 |
| `kv` | `scipy.special.kv` | 872.994 | 103.655 | 383.679 | 0.44 | 3.70 | 0.359 | 0.544 |
| `expi` | `scipy.special.expi` | 12.183 | 1.959 | 63.803 | 5.24 | 32.56 | 0.144 | 0.064 |
| `lambertw` | `scipy.special.lambertw.real` | 10.694 | 1.702 | 82.604 | 7.72 | 48.53 | 0.179 | 0.897 |
| `lambertw_m1` | `scipy.special.lambertw(k=-1).real` | 18.163 | 2.754 | 344.685 | 18.98 | 125.15 | 0.263 | 0.814 |
| `hyp1f1` | `scipy.special.hyp1f1` | 1435.234 | 170.736 | 234.986 | 0.16 | 1.38 | 0.316 | 0.508 |
