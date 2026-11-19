"""M7.7: fixed cost per call, on tiny arrays (ns per call; numpy alongside).

Run:  .venv/bin/python bench/m7_tiny.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

CASES = [
    ("a + b", "a + b"),
    ("a + 1.0", "a + 1.0"),
    ("sqrt(a)", "M.sqrt(a)"),
    ("a += b", "x += b"),
    ("x[1, 2]", "m[1, 2]"),
    ("x[1]", "m[1]"),
    ("x[:, ::2]", "m[:, ::2]"),
    ("x.T", "m.T"),
    ("sum(a)", "M.sum(a)"),
    ("zeros(3)", "M.zeros(3)"),
]


def run(mod):
    ns = {
        "M": mod,
        "a": mod.asarray(np.array([1.0, 2.0, 3.0])),
        "b": mod.asarray(np.array([4.0, 5.0, 6.0])),
        "m": mod.asarray(np.arange(12.0).reshape(3, 4)),
    }
    return [min(timeit.repeat(stmt, "x = a", globals=ns, number=100_000, repeat=7)) / 100_000 for _, stmt in CASES]


def main():
    sd, np_ = run(sdnp), run(np)
    print(f"{'':12} {'stardust':>9} {'numpy':>7}")
    for (label, _), a, b in zip(CASES, sd, np_):
        print(f"{label:12} {a * 1e9:7.0f}ns {b * 1e9:5.0f}ns")


if __name__ == "__main__":
    main()
