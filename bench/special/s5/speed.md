Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `expi` | `scipy.special.expi` | 25.709 | 3.611 | 62.416 | 2.43 | 17.28 | 0.139 | 0.062 |
| `expn` | `scipy.special.expn` | 95.336 | 12.094 | 75.519 | 0.79 | 6.24 | 0.478 | 0.475 |
| `ellipk` | `scipy.special.ellipk` | 7.974 | 1.102 | 8.504 | 1.07 | 7.72 | 0.137 | 0.063 |
| `ellipe` | `scipy.special.ellipe` | 9.393 | 1.233 | 7.786 | 0.83 | 6.32 | 0.139 | 0.059 |
