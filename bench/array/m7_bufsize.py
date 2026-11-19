"""M7.6: the cast-heavy cases at one buffer size (rebuilt per size by hand).

Run:  .venv/bin/python bench/m7_bufsize.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

rng = np.random.default_rng(0)
N = 1_000_000
ns = {
    "M": sdnp,
    "f": sdnp.asarray(rng.random(N)),
    "g": sdnp.asarray(rng.random(N)),
    "i": sdnp.asarray(rng.integers(-1000, 1000, N)),
    "b": sdnp.asarray(rng.random(N) > 0.5),
}
for label, stmt in [("i + f", "i + f"), ("mean(i)", "M.mean(i)"), ("sum(b)", "M.sum(b)"), ("f * i + g", "f * i + g")]:
    t = min(timeit.repeat(stmt, globals=ns, number=20, repeat=7)) / 20
    print(f"{label:10} {t * 1e6:7.1f}us")
