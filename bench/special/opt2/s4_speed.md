Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 2 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `betainc` | `scipy.special.betainc` | 194.965 | 23.353 | 152.830 | 0.78 | 6.54 | 0.186 | 0.483 |
| `betaincc` | `scipy.special.betaincc` | 192.839 | 23.313 | 153.900 | 0.80 | 6.60 | 0.117 | 0.444 |
| `betaincinv` | `scipy.special.betaincinv` | 577.466 | 66.754 | 1179.044 | 2.04 | 17.66 | 0.134 | 0.627 |
| `betainccinv` | `scipy.special.betainccinv` | 663.127 | 76.641 | 924.417 | 1.39 | 12.06 | 0.693 | 0.919 |
| `hyp1f1` | `scipy.special.hyp1f1` | 504.571 | 60.035 | 221.070 | 0.44 | 3.68 | 0.139 | 0.485 |
| `hyp2f1` | `scipy.special.hyp2f1` | 761.300 | 89.847 | 114.147 | 0.15 | 1.27 | 0.535 | 0.528 |
