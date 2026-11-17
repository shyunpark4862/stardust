# Multivariate normal and t cdf: speed (bench/stats/t9/cdf_speed.py)

Per point; ours 1 / 10 threads vs scipy (single-threaded, its defaults). Apple M4 Pro.

| case | atol | points | ours 1 thr | ours 10 thr | scipy | scipy / ours 1 thr | max error estimate | max abs(ours - scipy) |
|---|---|---|---|---|---|---|---|---|
| normal d=2 | 1e-06 | 20000 | 0.3 us | 0.0 us | 6.6 us | 20.6 | 0.0e+00 | 2.2e-16 |
| t df=4 d=2 | 1e-06 | 200 | 34.0 us | 3.9 us | 589.6 us | 17.3 | 0.0e+00 | 1.8e-04 |
| normal d=3 | 1e-06 | 20000 | 1.1 us | 0.1 us | 1.87 ms | 1728.9 | 0.0e+00 | 3.7e-05 |
| t df=4 d=3 | 1e-06 | 200 | 473.9 us | 53.8 us | 872.3 us | 1.8 | 0.0e+00 | 2.2e-04 |
| normal d=5 | 1e-06 | 40 | 5.76 ms | 1.62 ms | 1.74 ms | 0.3 | 9.9e-07 | 3.2e-06 |
| normal d=5 | 1e-05 | 40 | 773.0 us | 139.7 us | 1.74 ms | 2.3 | 1.0e-05 | 7.3e-06 |
| normal d=5 | 1e-04 | 40 | 601.8 us | 67.7 us | 1.74 ms | 2.9 | 3.1e-05 | 6.6e-06 |
| t df=4 d=5 | 1e-06 | 20 | 22.98 ms | 5.45 ms | 1.62 ms | 0.1 | 9.7e-07 | 5.3e-05 |
| t df=4 d=5 | 1e-05 | 20 | 2.19 ms | 449.4 us | 1.62 ms | 0.7 | 8.1e-06 | 5.2e-05 |
| t df=4 d=5 | 1e-04 | 20 | 837.1 us | 113.5 us | 1.62 ms | 1.9 | 5.4e-05 | 5.0e-05 |
| normal d=10 | 1e-06 | 40 | 199.33 ms | 32.68 ms | 35.03 ms | 0.2 | 1.9e-06 | 6.8e-06 |
| normal d=10 | 1e-05 | 40 | 26.36 ms | 9.63 ms | 35.03 ms | 1.3 | 8.7e-06 | 1.0e-05 |
| normal d=10 | 1e-04 | 40 | 1.66 ms | 230.0 us | 35.03 ms | 21.1 | 9.1e-05 | 3.0e-05 |
| t df=4 d=10 | 1e-06 | 20 | 306.67 ms | 71.64 ms | 4.45 ms | 0.0 | 4.4e-06 | 4.4e-05 |
| t df=4 d=10 | 1e-05 | 20 | 45.96 ms | 33.65 ms | 4.45 ms | 0.1 | 9.1e-06 | 4.7e-05 |
| t df=4 d=10 | 1e-04 | 20 | 2.07 ms | 439.7 us | 4.45 ms | 2.2 | 9.9e-05 | 4.1e-05 |
