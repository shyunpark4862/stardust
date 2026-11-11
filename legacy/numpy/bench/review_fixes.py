"""Review fixes (2026-11-10): the paths the four memory-safety fixes touch,
timed to check that the fixes cost nothing measurable (10 threads).

Run:  .venv/bin/python bench/review_fixes.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

rng = np.random.default_rng(0)
nb = rng.random(10**6) < 0.5
big = rng.random(10**7) < 0.5
b, f = sdnp.asarray(nb), sdnp.asarray(rng.random(10**6))
idx = sdnp.asarray(rng.integers(0, 10**6, 10**5))
v = sdnp.asarray(rng.random(int(nb.sum())))

CASES = [  # label, statement
    ("asarray(numpy bool, 1e6)", lambda: sdnp.asarray(nb)),  # now scans the bytes (fix 2)
    ("asarray(numpy bool, 1e7)", lambda: sdnp.asarray(big)),
    ("f[b]", lambda: f[b]),
    ("f[b] = 0.0", lambda: f.__setitem__(b, 0.0)),
    ("f[b] = v", lambda: f.__setitem__(b, v)),  # dense expand path (fix 3)
    ("f[idx]", lambda: f[idx]),  # gather_one (fix 4)
    ("f[idx] = 1.0", lambda: f.__setitem__(idx, 1.0)),  # scatter overlap checks (fix 1)
]

for label, fn in CASES:
    t = min(timeit.repeat(fn, number=50, repeat=7)) / 50
    print(f"| {label} | {t * 1e6:.1f}µs |")
