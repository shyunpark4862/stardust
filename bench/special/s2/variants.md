Machine: arm64 26.6.2, python 3.15.0, 14 cpus. ns per element, 1 thread, 10^6 elements, best of 5 rounds.

| inputs | erfc (branch, shipped) | NEON, every region, select | NEON Estrin, every region, select |
|---|---:|---:|---:|
| |x| < 1 | 2.29 | 4.15 | 4.03 |
| 1 <= x < 8 | 3.83 | 4.17 | 4.14 |
| x >= 8 | 3.26 | 4.17 | 4.07 |
| mixed, sorted | 3.51 | 4.16 | 4.02 |
| mixed, random | 4.72 | 4.20 | 4.06 |

| accuracy on tests/special/ref/erfc.npz | max ULP | 99.9% ULP |
|---|---:|---:|
| erfc (branch, shipped) | 7.77 | 4.70 |
| NEON, every region, select | 7.77 | 4.70 |
| NEON Estrin, every region, select | 9.13 | 4.19 |
