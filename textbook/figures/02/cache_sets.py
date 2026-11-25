"""L1 cache sets touched by a strided walk (Section 2.2; csrc/fft/driver.cpp cache_friendly).

The L1 data cache of the M4 Pro holds 128 KB in 128 sets of 8 lines of 128 bytes, so the
set of a byte address is (address mod 16384) div 128. Positions s bytes apart fall into
16384 / max(gcd(s, 16384), 128) sets (Proposition prop:cyclic, counted in lines). For
complex128 data (16 bytes) along an axis whose rows hold m elements, s = 16 m.
Writes cache_sets.dat (m, sets) for m = 1..2100.
"""
import math

with open("cache_sets.dat", "w") as f:
    f.write("m sets\n")
    for m in range(1, 2101):
        s = 16 * m
        g = max(math.gcd(s, 16384), 128)
        f.write(f"{m} {16384 // g}\n")
for m in (1, 8, 64, 100, 256, 512, 1000, 1024, 2048):
    s = 16 * m
    print(m, 16384 // max(math.gcd(s, 16384), 128))
