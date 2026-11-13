Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `expit` | `scipy.special.expit` | 1.313 | 0.199 | 6.581 | 5.01 | 33.02 | 0.130 | 0.082 |
| `logit` | `scipy.special.logit` | 3.394 | 0.494 | 4.631 | 1.36 | 9.37 | 0.141 | 0.056 |
| `log_expit` | `scipy.special.log_expit` | 14.455 | 1.988 | 15.576 | 1.08 | 7.84 | 0.125 | 0.081 |
| `xlogy` | `scipy.special.xlogy` | 3.285 | 0.478 | 4.006 | 1.22 | 8.39 | 0.171 | 0.297 |
| `xlog1py` | `scipy.special.xlog1py` | 4.737 | 0.680 | 6.462 | 1.36 | 9.50 | 0.172 | 0.296 |
| `entr` | `scipy.special.entr` | 2.861 | 0.400 | 3.382 | 1.18 | 8.45 | 0.126 | 0.055 |
| `rel_entr` | `scipy.special.rel_entr` | 10.399 | 1.502 | 11.841 | 1.14 | 7.88 | 0.171 | 0.329 |
