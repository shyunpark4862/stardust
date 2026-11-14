Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `expi` | `scipy.special.expi` | 25.613 | 3.764 | 62.896 | 2.46 | 16.71 | 0.141 | 0.064 |
| `expn` | `scipy.special.expn` | 97.796 | 11.831 | 77.420 | 0.79 | 6.54 | 0.463 | 0.480 |
| `ellipk` | `scipy.special.ellipk` | 31.061 | 4.174 | 8.337 | 0.27 | 2.00 | 0.159 | 0.066 |
| `ellipe` | `scipy.special.ellipe` | 64.497 | 8.300 | 7.869 | 0.12 | 0.95 | 0.146 | 0.062 |
