Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gammainc` | `scipy.special.gammainc` | 146.170 | 17.487 | 95.374 | 0.65 | 5.45 | 0.215 | 0.348 |
| `gammaincc` | `scipy.special.gammaincc` | 146.198 | 17.353 | 212.247 | 1.45 | 12.23 | 0.216 | 0.426 |
| `gammaincinv` | `scipy.special.gammaincinv` | 427.558 | 48.422 | 491.973 | 1.15 | 10.16 | 0.219 | 0.381 |
| `gammainccinv` | `scipy.special.gammainccinv` | 602.501 | 68.621 | 293.358 | 0.49 | 4.28 | 0.218 | 0.387 |
| `betainc` | `scipy.special.betainc` | 556.997 | 64.315 | 161.400 | 0.29 | 2.51 | 0.580 | 0.467 |
| `betaincc` | `scipy.special.betaincc` | 552.412 | 63.068 | 162.225 | 0.29 | 2.57 | 0.663 | 0.446 |
| `betaincinv` | `scipy.special.betaincinv` | 1093.371 | 123.554 | 1243.386 | 1.14 | 10.06 | 0.382 | 0.635 |
| `betainccinv` | `scipy.special.betainccinv` | 1624.771 | 184.079 | 972.252 | 0.60 | 5.28 | 1.694 | 0.937 |
