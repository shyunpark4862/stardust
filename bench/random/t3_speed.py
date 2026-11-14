"""T3: sd.random against numpy's Generator (PCG64, the default, and Philox), 10^6 elements.

    .venv/bin/python bench/random/t3_speed.py > bench/random/t3_speed.md
"""

import timeit

import numpy as np

import stardust as sd


def best(f, n=7):
    return min(timeit.repeat(f, number=1, repeat=n)) * 1e3


def main():
    N = 1_000_000
    r = sd.random.default_rng(1)
    pcg, phx = np.random.default_rng(1), np.random.Generator(np.random.Philox(1))
    x = sd.arange(N, dtype=sd.float64)
    xn = np.arange(N, dtype=np.float64)
    p = sd.asarray(np.random.default_rng(0).random(1000))
    p = p / sd.sum(p)
    pn = np.asarray(p)
    cases = [
        ("random(10^6)", lambda: r.random(N), lambda g: g.random(N)),
        ("integers(0, 1000, shape=10^6)", lambda: r.integers(0, 1000, shape=N), lambda g: g.integers(0, 1000, N)),
        ("integers(-2^62, 2^62, shape=10^6)", lambda: r.integers(-2**62, 2**62, shape=N), lambda g: g.integers(-2**62, 2**62, N)),
        ("choice(1000, shape=10^6)", lambda: r.choice(1000, shape=N), lambda g: g.choice(1000, N)),
        ("choice(1000, shape=10^6, p=...)", lambda: r.choice(1000, shape=N, p=p), lambda g: g.choice(1000, N, p=pn)),
        ("choice(10^6, shape=1000, replace=False)", lambda: r.choice(N, shape=1000, replace=False),
         lambda g: g.choice(N, 1000, replace=False)),
        ("permutation(10^6)", lambda: r.permutation(N), lambda g: g.permutation(N)),
        ("permutation(float64 array 10^6)", lambda: r.permutation(x), lambda g: g.permutation(xn)),
    ]
    print("| call | sd 1T (ms) | sd 10T (ms) | numpy PCG64 (ms) | numpy Philox (ms) | PCG64 / sd 1T | PCG64 / sd 10T |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    before = sd.get_num_threads()
    try:
        for name, f, g in cases:
            sd.set_num_threads(1)
            t1 = best(f)
            sd.set_num_threads(10)
            t10 = best(f)
            tp, tx = best(lambda: g(pcg)), best(lambda: g(phx))
            print(f"| `{name}` | {t1:.2f} | {t10:.2f} | {tp:.2f} | {tx:.2f} | {tp / t1:.2f} | {tp / t10:.1f} |", flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
