Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 2 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `hyp1f1` | `scipy.special.hyp1f1` | 279.446 | 31.707 | 227.048 | 0.81 | 7.16 | 0.142 | 0.493 |
| `hyp2f1` | `scipy.special.hyp2f1` | 381.233 | 43.439 | 117.363 | 0.31 | 2.70 | 0.334 | 0.530 |
