"""Review fixes 5-8 (2026-11-10): full reductions with a cast on threads,
sort scratch reused across short lanes, the shared mask piece walk.

Run:  .venv/bin/python bench/review_fixes_rest.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

rng = np.random.default_rng(0)
S = sdnp.asarray
b = S(rng.random(10**7) < 0.5)
i = S(rng.integers(-(10**6), 10**6, 10**7))
f = S(rng.random(10**7))
rows = S(rng.random((100_000, 8)))
f1 = S(rng.random(10**6))
m1 = S(rng.random(10**6) < 0.5)
v1 = S(rng.random(int(sdnp.sum(m1))))

CASES = [  # label, statement, number
    ("sum(bool 1e7)", lambda: sdnp.sum(b), 10),
    ("mean(int64 1e7)", lambda: sdnp.mean(i), 10),
    ("any(float 1e7 > 0, as bool)", lambda: sdnp.any(f, axis=None), 10),
    ("sum(float 1e7) (no cast, unchanged)", lambda: sdnp.sum(f), 10),
    ("sort (100000, 8) axis=1", lambda: sdnp.sort(rows, axis=1), 10),
    ("argsort (100000, 8) axis=1", lambda: sdnp.argsort(rows, axis=1), 10),
    ("f[m] (1e6, half true)", lambda: f1[m1], 50),
    ("f[m] = v", lambda: f1.__setitem__(m1, v1), 50),
]

print("| case | 1 thread | 10 threads |")
print("|---|---:|---:|")
for label, fn, number in CASES:
    row = []
    for n in (1, 10):
        sdnp.set_num_threads(n)
        row.append(min(timeit.repeat(fn, number=number, repeat=7)) / number * 1e6)
    print(f"| {label} | {row[0]:.1f}µs | {row[1]:.1f}µs |")
