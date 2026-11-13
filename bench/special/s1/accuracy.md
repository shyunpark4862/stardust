| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |
|---|---:|---|---:|---:|---|---:|---:|
| `_cf_tan` | 6038 | guaranteed | 6.71 | 5.43 | `numpy.tan` | 2.04 | 1.58 |
| `_cf_tan` | 2026 | loose | 5.19e+15 | 2.62e+15 | `numpy.tan` | 2.22 | 2.11 |
| `entr` | 5024 | guaranteed | 1.25 | 0.785 | `scipy.special.entr` | 1.25 | 0.785 |
| `expit` | 8082 | guaranteed | 2.06 | 1.73 | `scipy.special.expit` | 1.11e+15 | 8.63e+14 |
| `log_expit` | 8082 | guaranteed | 1.24 | 0.938 | `scipy.special.log_expit` | 1.24 | 0.938 |
| `logit` | 7535 | guaranteed | 2.8 | 2.52 | `scipy.special.logit` | 1.22 | 1.13 |
| `rel_entr` | 5000 | guaranteed | 2.07 | 1.62 | `scipy.special.rel_entr` | 2.07 | 1.62 |
| `xlog1py` | 4000 | guaranteed | 1.24 | 1.12 | `scipy.special.xlog1py` | 1.24 | 1.12 |
| `xlogy` | 4000 | guaranteed | 1.19 | 1.09 | `scipy.special.xlogy` | 1.19 | 1.09 |
