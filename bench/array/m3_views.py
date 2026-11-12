"""M3: views cost O(1); per-call overhead is what is left to compare.

Part 1: a view's cost does not depend on the array size (no element is
touched), while a copy's grows with it.
Part 2: the fixed cost of one small operation (argument parsing, object
creation) against numpy's -- the "thin Python / thick C++" bet of R1.
Part 3: concat, which does copy.

Run:  .venv/bin/python bench/m3_views.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp


def per_call_ns(stmt, ns, number=200_000):
    best = min(timeit.repeat(stmt, globals=ns, number=number, repeat=5))
    return best / number * 1e9


def main():
    print("1) view vs copy, by array size (per call)")
    print(f"{'elements':>12} {'a.T':>10} {'reshape view':>14} {'reshape copy':>14}")
    for n in (10**2, 10**4, 10**6, 10**7):
        a = sdnp.asarray(np.random.default_rng(0).random(n)).reshape((n // 10, 10))
        ns = {"a": a, "sdnp": sdnp}
        number = 20 if n >= 10**6 else 2000
        t = per_call_ns("a.T", ns, number)
        v = per_call_ns("sdnp.reshape(a, (10, -1))", ns, number)
        c = per_call_ns("sdnp.reshape(a.T, (-1,))", ns, number)
        print(f"{n:>12,} {t:>8.0f}ns {v:>12.0f}ns {c / 1e3:>11.1f}µs")

    print("\n2) fixed overhead of one small operation (ns per call)")
    a_np = np.arange(12.0).reshape(3, 4)
    a_sd = sdnp.asarray(a_np.copy())
    cases = [
        ("x[1, 2]  (scalar read)", "x[1, 2]"),
        ("x[1]     (row view)", "x[1]"),
        ("x[:, ::2]", "x[:, ::2]"),
        ("x[1, 2] = 5.0", "x[1, 2] = 5.0"),
        ("x.T", "x.T"),
        ("reshape (view)", "R(x, (4, 3))"),
    ]
    print(f"{'':28} {'stardust':>9} {'numpy':>9}")
    for label, stmt in cases:
        t_sd = per_call_ns(stmt, {"x": a_sd, "R": sdnp.reshape})
        t_np = per_call_ns(stmt, {"x": a_np, "R": np.reshape})
        print(f"{label:28} {t_sd:7.0f}ns {t_np:7.0f}ns")

    print("\n3) concat of two (2000, 2000) float64 arrays")
    p = np.random.default_rng(1).random((2000, 2000))
    q = np.random.default_rng(2).random((2000, 2000))
    ps, qs = sdnp.asarray(p), sdnp.asarray(q)
    for axis in (0, 1):
        t_sd = per_call_ns("sdnp.concat([ps, qs], axis=ax)", {"sdnp": sdnp, "ps": ps, "qs": qs, "ax": axis}, 20)
        t_np = per_call_ns("np.concatenate([p, q], axis=ax)", {"np": np, "p": p, "q": q, "ax": axis}, 20)
        print(f"axis={axis}: stardust {t_sd / 1e6:5.2f}ms   numpy {t_np / 1e6:5.2f}ms")


if __name__ == "__main__":
    main()
