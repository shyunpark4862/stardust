| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |
|---|---:|---|---:|---:|---|---:|---:|
| `erf` | 7564 | guaranteed | 2.29 | 1.82 | `scipy.special.erf` | 2.29 | 1.73 |
| `erfc` | 8073 | guaranteed | 7.77 | 4.7 | `scipy.special.erfc` | 2.34e+13 | 1.9e+13 |
| `erfcx` | 7555 | guaranteed | 6.05 | 3.78 | `scipy.special.erfcx` | 500 | 426 |
| `ndtr` | 8064 | guaranteed | 6.73 | 5.28 | `scipy.special.ndtr` | 1.09e+13 | 6.59e+12 |
| `log_ndtr` | 8064 | guaranteed | 6.97 | 4.45 | `scipy.special.log_ndtr` | 1.14e+13 | 7.25e+12 |
| `ndtri` | 7559 | guaranteed | 5.43 | 4.51 | `scipy.special.ndtri` | 2.81 | 2.42 |
| `erfinv` | 7563 | guaranteed | 4.64 | 3.9 | `scipy.special.erfinv` | 1.95 | 1.72 |
| `erfcinv` | 7550 | guaranteed | 4.51 | 3.82 | `scipy.special.erfcinv` | inf | 1.69e+11 |
