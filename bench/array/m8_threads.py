"""M8: how far does each kind of operation scale with threads?

Memory-bound operations (f + g) stop scaling once the memory system is
saturated; compute-bound ones (exp, sin, sort, integer matmul) keep scaling
up to the performance cores. 12 and 14 threads bring in the 4 efficiency
cores. numpy (single-threaded except for BLAS) is shown for reference.

Run:  .venv/bin/python bench/m8_threads.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

THREADS = [1, 2, 4, 6, 8, 10, 12, 14]


def main():
    rng = np.random.default_rng(0)
    n = 4_000_000
    arrays = {
        "f": rng.random(n) + 0.5,
        "g": rng.random(n),
        "i": rng.integers(-1000, 1000, n),
        "m": rng.random((3000, 3000)),
        "w": rng.random((300, 30000)),
        "s": rng.random((2000, 1000)),
        "ia": rng.integers(-100, 100, (500, 500)),
        "t3": rng.random((200, 200, 100)).transpose(2, 0, 1),
    }
    cases = [
        ("f + g          memory-bound", "f + g", 10),
        ("i + f          buffered cast", "i + f", 10),
        ("t3.copy()      transposed", "t3.copy()", 10),
        ("sqrt(f)", "M.sqrt(f)", 10),
        ("exp(f)         compute-bound", "M.exp(f)", 3),
        ("sin(f)", "M.sin(f)", 3),
        ("sum(f)         pairwise tree", "M.sum(f)", 10),
        ("max(f)", "M.max(f)", 10),
        ("argmax(f)", "M.argmax(f)", 10),
        ("var(f)", "M.var(f)", 5),
        ("sum(m, axis=1) rows", "M.sum(m, axis=1)", 10),
        ("sum(m, axis=0) columns", "M.sum(m, axis=0)", 10),
        ("sum(w, axis=0) wide", "M.sum(w, axis=0)", 10),
        ("sort(s)        2000 rows", "M.sort(s)", 3),
        ("argsort(s)", "M.argsort(s)", 3),
        ("ia @ ia        int64 500^3", "ia @ ia", 3),
    ]
    sd = {k: sdnp.asarray(v) for k, v in arrays.items()} | {"M": sdnp}
    npns = dict(arrays) | {"M": np}
    head = "".join(f"{t:>7}" for t in THREADS)
    print(f"{'(speedup over 1 thread)':38}{'1 thread':>10}{head}{'numpy':>10}")
    for label, stmt, number in cases:
        times = []
        for t in THREADS:
            sdnp.set_num_threads(t)
            times.append(min(timeit.repeat(stmt, globals=sd, number=number, repeat=5)) / number)
        t_np = min(timeit.repeat(stmt, globals=npns, number=number, repeat=5)) / number
        cells = "".join(f"{times[0] / x:6.1f}x" for x in times)
        print(f"{label:38}{times[0] * 1e3:8.2f}ms{cells}{t_np * 1e3:8.2f}ms")
    sdnp.set_num_threads(10)


if __name__ == "__main__":
    main()
