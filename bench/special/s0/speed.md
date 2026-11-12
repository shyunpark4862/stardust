Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `_cf_tan` | `numpy.tan` | 15.795 | 2.242 | 2.175 | 0.14 | 0.97 | 0.126 | 0.086 |
