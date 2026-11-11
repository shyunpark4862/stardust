"""M5: the reduction baseline -- speed, and the accuracy of a plain running sum.

Part 1: time per call against numpy (one thread, sequential loops).
Part 2: accuracy. A running sum's rounding error grows with n; numpy's
pairwise summation grows with log n. M7 adds pairwise summation and is
measured against these numbers.

Run:  .venv/bin/python bench/m5_reduction.py
"""

import math
import timeit

import numpy as np

import stardust_numpy as sdnp


def best_ms(stmt, ns, number=10):
    return min(timeit.repeat(stmt, globals=ns, number=number, repeat=5)) / number * 1e3


def main():
    rng = np.random.default_rng(0)
    m = rng.random((2000, 2000))
    arrays = {"m": m, "mf": np.asfortranarray(m), "v": rng.random(4_000_000), "iv": rng.integers(0, 100, 4_000_000)}
    cases = [
        ("sum(m)            everything", "M.sum(m)"),
        ("sum(m, axis=1)    inner axis (rows)", "M.sum(m, axis=1)"),
        ("sum(m, axis=0)    outer axis (columns)", "M.sum(m, axis=0)"),
        ("sum(mf, axis=0)   F-order, inner in memory", "M.sum(mf, axis=0)"),
        ("sum(iv)           int64", "M.sum(iv)"),
        ("max(v)", "M.max(v)"),
        ("argmax(v)", "M.argmax(v)"),
        ("mean(m, axis=0)", "M.mean(m, axis=0)"),
        ("var(v)            two passes", "M.var(v)"),
        ("cumulative_sum(v)", "M.cumulative_sum(v)"),
    ]
    print("1) time per call (4M elements)")
    print(f"{'':42} {'stardust':>9} {'numpy':>9} {'ratio':>7}")
    for label, stmt in cases:
        t_sd = best_ms(stmt, {k: sdnp.asarray(v) for k, v in arrays.items()} | {"M": sdnp})
        t_np = best_ms(stmt, dict(arrays) | {"M": np})
        print(f"{label:42} {t_sd:7.2f}ms {t_np:7.2f}ms {t_sd / t_np:6.1f}x")

    print("\n2) accuracy of sum(x): relative error against the exact sum (math.fsum)")
    print(f"{'n':>12} {'stardust (running)':>20} {'numpy (pairwise)':>18}")
    for n in (10**3, 10**5, 10**7):
        x = np.full(n, 0.1) + rng.random(n) * 1e-3
        exact = math.fsum(x.tolist())
        e_sd = abs(sdnp.sum(sdnp.asarray(x)) - exact) / exact
        e_np = abs(float(np.sum(x)) - exact) / exact
        print(f"{n:>12,} {e_sd:20.1e} {e_np:18.1e}")


if __name__ == "__main__":
    main()
