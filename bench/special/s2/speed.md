Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `erf` | `scipy.special.erf` | 5.441 | 0.770 | 11.077 | 2.04 | 14.38 | 0.131 | 0.055 |
| `erfc` | `scipy.special.erfc` | 5.663 | 0.803 | 10.038 | 1.77 | 12.49 | 0.130 | 0.085 |
| `erfcx` | `scipy.special.erfcx` | 7.347 | 1.035 | 12.354 | 1.68 | 11.94 | 0.134 | 0.086 |
| `erfinv` | `scipy.special.erfinv` | 7.136 | 0.994 | 14.903 | 2.09 | 14.99 | 0.150 | 0.073 |
| `erfcinv` | `scipy.special.erfcinv` | 8.775 | 1.235 | 16.649 | 1.90 | 13.48 | 0.155 | 0.068 |
| `ndtr` | `scipy.special.ndtr` | 6.121 | 0.878 | 11.737 | 1.92 | 13.37 | 0.134 | 0.083 |
| `log_ndtr` | `scipy.special.log_ndtr` | 14.828 | 2.080 | 15.522 | 1.05 | 7.46 | 0.157 | 0.089 |
| `ndtri` | `scipy.special.ndtri` | 7.953 | 1.133 | 15.001 | 1.89 | 13.24 | 0.151 | 0.068 |
