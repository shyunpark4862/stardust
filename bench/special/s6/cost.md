add: 0.204 ns per element (cost.hpp kAddNs)

| function | ns per element | weight from it | weight in cost.hpp |
|---|---:|---:|---:|
| `i0` | 15.71 | 77 | 79 |
| `i0e` | 12.38 | 61 | 63 |
| `i1` | 16.36 | 80 | 79 |
| `i1e` | 12.78 | 63 | 63 |
| `k0` | 24.08 | 118 | 105 |
| `k0e` | 16.19 | 79 | 105 |
| `k1` | 24.18 | 119 | 105 |
| `k1e` | 15.85 | 78 | 105 |
| `j0` | 11.43 | 56 | 79 |
| `j1` | 11.53 | 57 | 79 |
| `y0` | 13.69 | 67 | 105 |
| `y1` | 13.08 | 64 | 105 |
| `jv` | 751.31 | 3686 | 256 |
| `yv` | 754.21 | 3700 | 256 |
| `iv` | 2015.73 | 9889 | 256 |
| `kv` | 1606.41 | 7881 | 256 |
| `ive` | 1879.54 | 9221 | 256 |
| `kve` | 1804.51 | 8853 | 256 |
Traceback (most recent call last):
  File "/Users/sanghyunpark/Desktop/stardust/stardust/bench/special/cost.py", line 46, in <module>
    main(sys.argv[1:])
    ~~~~^^^^^^^^^^^^^^
  File "/Users/sanghyunpark/Desktop/stardust/stardust/bench/special/cost.py", line 42, in main
    print(f"| `{name}` | {ns:.2f} | {max(1, round(ns / add_ns))} | {sd.special._core._weight(name)} |")
                                                                    ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
KeyError: 'airy_ai'
