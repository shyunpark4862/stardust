"""M8: where do threads start to pay? (1 vs all threads, by size)

Meant for an experimental build with parallel::kThreshold = 1 and a small
kTaskWork, so that even small arrays are split (see the M8 explainer).

Run:  .venv/bin/python bench/m8_threshold.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

CASES = [("f + g", "f + g"), ("exp(f)", "M.exp(f)"), ("sum(f)", "M.sum(f)")]


def main():
    rng = np.random.default_rng(0)
    full = sdnp.get_num_threads()
    print(f"{'elements':>10} " + " ".join(f"{name:>22}" for name, _ in CASES))
    for n in [2**k for k in range(12, 23, 1)]:
        ns = {"M": sdnp, "f": sdnp.asarray(rng.random(n)), "g": sdnp.asarray(rng.random(n))}
        cells = []
        for _, stmt in CASES:
            t = []
            for threads in (1, full):
                sdnp.set_num_threads(threads)
                number = max(1, 2_000_000 // n)
                t.append(min(timeit.repeat(stmt, globals=ns, number=number, repeat=5)) / number * 1e6)
            cells.append(f"{t[0]:8.1f}/{t[1]:7.1f}us {t[0] / t[1]:4.1f}x")
        print(f"{n:>10} " + " ".join(cells))
    sdnp.set_num_threads(full)


if __name__ == "__main__":
    main()
