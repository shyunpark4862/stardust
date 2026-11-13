Machine: arm64 26.6.2, python 3.15.0, 14 cpus. 10^6 elements in ms, one scalar call in us; best of 3 rounds.

| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gamma` | `scipy.special.gamma` | 17.975 | 2.448 | 16.398 | 0.91 | 6.70 | 0.149 | 0.076 |
| `rgamma` | `scipy.special.rgamma` | 20.357 | 2.742 | 17.061 | 0.84 | 6.22 | 0.149 | 0.076 |
| `gammaln` | `scipy.special.gammaln` | 10.893 | 1.469 | 12.108 | 1.11 | 8.24 | 0.151 | 0.061 |
| `digamma` | `scipy.special.digamma` | 8.917 | 1.207 | 11.025 | 1.24 | 9.13 | 0.134 | 0.059 |
| `polygamma` | `scipy.special.polygamma` | 116.556 | 16.502 | 118.693 | 1.02 | 7.19 | 0.236 | 3.102 |
| `beta` | `scipy.special.beta` | 59.734 | 8.418 | 32.443 | 0.54 | 3.85 | 0.202 | 0.328 |
| `betaln` | `scipy.special.betaln` | 78.695 | 10.981 | 25.767 | 0.33 | 2.35 | 0.271 | 0.337 |
| `binom` | `scipy.special.binom` | 99.156 | 13.841 | 44.478 | 0.45 | 3.21 | 0.285 | 0.355 |
| `zeta` | `scipy.special.zeta` | 89.323 | 12.381 | 16.187 | 0.18 | 1.31 | 0.247 | 0.249 |
| `zeta_hurwitz` | `scipy.special.zeta` | 86.202 | 11.952 | 85.908 | 1.00 | 7.19 | 0.190 | 0.353 |
