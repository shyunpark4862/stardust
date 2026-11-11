"""M4: the baseline. Plain scalar loops, one thread, whole-array casts.

Every number here is what M7 (NEON, contiguous fast paths, buffered casts),
M8 (threads) and M9 (vForce) will be measured against.

Run:  .venv/bin/python bench/m4_elementwise.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

N = 1_000_000


def best(stmt, ns, number, setup="pass"):
    return min(timeit.repeat(stmt, setup, globals=ns, number=number, repeat=5)) / number


def main():
    rng = np.random.default_rng(0)
    f = rng.random(N) + 0.5
    g = rng.random(N) + 0.5
    i = rng.integers(-1000, 1000, N)
    c = f + 1j * g
    arrays = {"f": f, "g": g, "i": i, "c": c, "ft": rng.random((1000, 1000)).T}

    cases = [
        ("f + g          float64", "f + g"),
        ("i + i          int64", "i + i"),
        ("c * c          complex128", "c * c"),
        ("f + 1.0        scalar operand", "f + 1.0"),
        ("i + f          int64 -> float64 cast", "i + f"),
        ("ft + 1.0       transposed input", "ft + 1.0"),
        ("i // 7         floor_divide int64", "i // 7"),
        ("f < g          compare", "f < g"),
        ("sqrt(f)", "M.sqrt(f)"),
        ("exp(f)", "M.exp(f)"),
        ("sin(f)", "M.sin(f)"),
        ("tanh(f)", "M.tanh(f)"),
        ("abs(c)         complex -> float64", "M.abs(c)"),
    ]
    print(f"1) {N:,} elements, one call (ms)")
    print(f"{'':40} {'stardust':>9} {'numpy':>9} {'ratio':>7}")
    for label, stmt in cases:
        sd_ns = {k: sdnp.asarray(v) for k, v in arrays.items()} | {"M": sdnp}
        np_ns = dict(arrays) | {"M": np}
        t_sd = best(stmt, sd_ns, 10) * 1e3
        t_np = best(stmt, np_ns, 10) * 1e3
        print(f"{label:40} {t_sd:7.2f}ms {t_np:7.2f}ms {t_sd / t_np:6.1f}x")

    print("\n2) tiny arrays (3 elements): fixed cost per call (ns)")
    small = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}
    for label, stmt in [("a + b", "a + b"), ("a + 1.0", "a + 1.0"), ("a += b", "x += b"), ("M.sqrt(a)", "M.sqrt(a)")]:
        t_sd = best(stmt, {k: sdnp.asarray(v.copy()) for k, v in small.items()} | {"M": sdnp}, 100_000, "x = a") * 1e9
        t_np = best(stmt, {k: v.copy() for k, v in small.items()} | {"M": np}, 100_000, "x = a") * 1e9
        print(f"{label:40} {t_sd:7.0f}ns {t_np:7.0f}ns {t_sd / t_np:6.1f}x")


if __name__ == "__main__":
    main()
