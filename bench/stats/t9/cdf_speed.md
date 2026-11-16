# Multivariate normal and t cdf: speed (bench/stats/t9/cdf_speed.py)

Per point; ours 1 / 10 threads vs scipy (single-threaded, its defaults). Apple M4 Pro.

| case | atol | points | ours 1 thr | ours 10 thr | scipy | scipy / ours 1 thr | max error estimate | max abs(ours - scipy) |
|---|---|---|---|---|---|---|---|---|
| normal d=2 | 1e-06 | 20000 | 0.4 us | 0.0 us | 6.6 us | 17.9 | 0.0e+00 | 2.2e-16 |
| t df=4 d=2 | 1e-06 | 200 | 161.8 us | 18.3 us | 567.5 us | 3.5 | 0.0e+00 | 1.8e-04 |
| normal d=3 | 1e-06 | 20000 | 1.0 us | 0.1 us | 1.80 ms | 1733.1 | 0.0e+00 | 3.7e-05 |
| t df=4 d=3 | 1e-06 | 200 | 515.6 us | 57.1 us | 809.6 us | 1.6 | 0.0e+00 | 2.2e-04 |
| normal d=5 | 1e-06 | 40 | 5.53 ms | 1.55 ms | 1.66 ms | 0.3 | 9.9e-07 | 3.2e-06 |
| normal d=5 | 1e-05 | 40 | 731.7 us | 134.2 us | 1.66 ms | 2.3 | 1.0e-05 | 7.3e-06 |
| normal d=5 | 1e-04 | 40 | 576.6 us | 67.8 us | 1.66 ms | 2.9 | 3.1e-05 | 6.6e-06 |
| t df=4 d=5 | 1e-06 | 20 | 21.69 ms | 5.45 ms | 1.52 ms | 0.1 | 9.7e-07 | 5.3e-05 |
| t df=4 d=5 | 1e-05 | 20 | 2.12 ms | 445.6 us | 1.52 ms | 0.7 | 8.1e-06 | 5.2e-05 |
| t df=4 d=5 | 1e-04 | 20 | 816.2 us | 106.2 us | 1.52 ms | 1.9 | 5.4e-05 | 5.0e-05 |
| normal d=10 | 1e-06 | 40 | 191.79 ms | 32.75 ms | 33.96 ms | 0.2 | 1.9e-06 | 6.8e-06 |
| normal d=10 | 1e-05 | 40 | 25.60 ms | 9.69 ms | 33.96 ms | 1.3 | 8.7e-06 | 1.0e-05 |
| normal d=10 | 1e-04 | 40 | 1.58 ms | 207.8 us | 33.96 ms | 21.5 | 9.1e-05 | 3.0e-05 |
| t df=4 d=10 | 1e-06 | 20 | 296.47 ms | 71.93 ms | 4.20 ms | 0.0 | 4.4e-06 | 4.4e-05 |
| t df=4 d=10 | 1e-05 | 20 | 44.89 ms | 33.08 ms | 4.20 ms | 0.1 | 9.1e-06 | 4.7e-05 |
| t df=4 d=10 | 1e-04 | 20 | 2.01 ms | 432.8 us | 4.20 ms | 2.1 | 9.9e-05 | 4.1e-05 |
