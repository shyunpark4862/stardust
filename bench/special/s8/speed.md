Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `wofz` | `scipy.special.wofz` | 56.025 | 7.600 | 50.053 | 0.89 | 6.59 | 0.222 | 0.106 |
| `erf_complex` | `scipy.special.erf` | 46.528 | 6.454 | 38.946 | 0.84 | 6.03 | 0.269 | 0.116 |
| `erfc_complex` | `scipy.special.erfc` | 57.180 | 7.335 | 49.600 | 0.87 | 6.76 | 0.266 | 0.119 |
| `erfcx_complex` | `scipy.special.erfcx` | 47.345 | 6.673 | 44.359 | 0.94 | 6.65 | 0.249 | 0.107 |
| `loggamma` | `scipy.special.loggamma` | 110.643 | 13.399 | 76.700 | 0.69 | 5.72 | 0.156 | 0.140 |
| `lambertw_c0` | `scipy.special.lambertw(k=0)` | 136.305 | 15.805 | 193.196 | 1.42 | 12.22 | 0.361 | 0.747 |
| `lambertw_cm1` | `scipy.special.lambertw(k=-1)` | 128.352 | 14.653 | 295.189 | 2.30 | 20.14 | 0.357 | 1.131 |
| `lambertw_c1` | `scipy.special.lambertw(k=1)` | 135.407 | 15.917 | 280.644 | 2.07 | 17.63 | 0.389 | 0.846 |
| `lambertw_c4` | `scipy.special.lambertw(k=4)` | 108.003 | 12.615 | 205.105 | 1.90 | 16.26 | 0.397 | 0.838 |
