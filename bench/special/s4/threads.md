machine: arm64 26.6.2, python 3.15.0, 14 cpus

| function | inputs | 1 thread (ms) | 10 threads (ms) | speed-up |
|---|---|---:|---:|---:|
| `betainc` | shuffled | 440.9 | 51.2 | 8.61 |
| `betainc` | sorted | 436.8 | 50.4 | 8.66 |
| `gammainc` | shuffled | 159.4 | 18.8 | 8.49 |
| `gammainc` | sorted | 159.1 | 18.6 | 8.54 |

Before the weight cap (csrc/special/cost.hpp `kMaxWeight` = 256), the same table gave
4.84-4.99x for `betainc` and 6.81-6.94x for `gammainc`: with weights of 800-8500 every
task was one element. Sorted and shuffled inputs differ by under 2% either way: the
pool's single counter hands the next task to whichever thread is free.
