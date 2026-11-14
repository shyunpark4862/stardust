Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `hyp1f1` | `scipy.special.hyp1f1` | 1626.727 | 180.478 | 228.500 | 0.14 | 1.27 | 0.302 | 0.491 |
| `hyp2f1` | `scipy.special.hyp2f1` | 2231.234 | 246.751 | 117.577 | 0.05 | 0.48 | 1.650 | 0.535 |
| `lambertw` | `scipy.special.lambertw.real` | 20.956 | 2.750 | 79.189 | 3.78 | 28.80 | 0.169 | 0.860 |
| `lambertw_m1` | `scipy.special.lambertw(k=-1).real` | 41.659 | 6.113 | 329.692 | 7.91 | 53.94 | 0.245 | 0.769 |
