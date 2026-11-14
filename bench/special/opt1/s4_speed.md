Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 2 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gammainc` | `scipy.special.gammainc` | 80.414 | 9.798 | 89.637 | 1.11 | 9.15 | 0.092 | 0.342 |
| `gammaincc` | `scipy.special.gammaincc` | 80.609 | 10.075 | 202.778 | 2.52 | 20.13 | 0.093 | 0.411 |
| `gammaincinv` | `scipy.special.gammaincinv` | 340.113 | 39.640 | 459.088 | 1.35 | 11.58 | 0.115 | 0.372 |
| `gammainccinv` | `scipy.special.gammainccinv` | 281.717 | 33.248 | 283.445 | 1.01 | 8.53 | 0.113 | 0.367 |
| `betainc` | `scipy.special.betainc` | 225.768 | 25.983 | 154.712 | 0.69 | 5.95 | 0.192 | 0.484 |
| `betaincc` | `scipy.special.betaincc` | 223.855 | 26.339 | 154.153 | 0.69 | 5.85 | 0.139 | 0.449 |
| `betaincinv` | `scipy.special.betaincinv` | 609.158 | 69.458 | 1191.035 | 1.96 | 17.15 | 0.161 | 0.637 |
| `betainccinv` | `scipy.special.betainccinv` | 657.305 | 75.158 | 932.520 | 1.42 | 12.41 | 0.695 | 0.897 |
| `expn` | `scipy.special.expn` | 90.364 | 11.459 | 72.857 | 0.81 | 6.36 | 0.335 | 0.466 |
