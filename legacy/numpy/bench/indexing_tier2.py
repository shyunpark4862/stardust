"""Tier 2 of docs/learn/next-optimizations.md (M10+ indexing): the cases
before and after, 1 and 10 threads, against numpy.

Run:  .venv/bin/python bench/indexing_tier2.py [--save NAME]
"""

import argparse
import pathlib
import timeit

import numpy as np

import stardust_numpy as sdnp

CASES = [  # label, stardust statement, numpy statement, number
    ("m[:, rows]", "m[:, rows]", "m[:, rows]", 20),
    ("mT[:, rows] (F order)", "mT[:, rows]", "mT[:, rows]", 20),
    ("t3[:, :, idx3]", "t3[:, :, idx3]", "t3[:, :, idx3]", 20),
    ("m[rows] (rows: contiguous blocks)", "m[rows]", "m[rows]", 20),
    ("p3[idx] (3-element blocks)", "p3[idx]", "p3[idx]", 20),
    ("m[:, rows] = v", "m2[:, rows] = v", "m2[:, rows] = v", 20),
    ("f[b] (mask, half true)", "f[b]", "f[b]", 20),
    ("f[s] (mask, 1% true)", "f[s]", "f[s]", 20),
    ("m[mb] (2-D mask)", "m[mb]", "m[mb]", 20),
    ("f[b] = 0.0", "f2[b] = 0.0", "f2[b] = 0.0", 20),
    ("tiny x[1, 2]", "sm[1, 2]", "sm[1, 2]", 20000),
    ("tiny x[1]", "sm[1]", "sm[1]", 20000),
    ("tiny zeros(3)", "M.zeros(3)", "np.zeros(3)", 20000),
    ("tiny x + y", "sv + sv", "sv + sv", 20000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    args = ap.parse_args()
    r = np.random.default_rng(0)
    n = 1_000_000
    m = r.random((1000, 1000))
    arrays = {"m": m, "m2": m.copy(), "mT": np.asfortranarray(m), "t3": r.random((100, 100, 100)),
              "rows": r.integers(0, 1000, 1000), "idx3": r.integers(-100, 100, 50), "p3": r.random((n, 3)),
              "idx": r.integers(0, n, 300_000), "v": r.random((1000, 1000)), "f": r.random(n), "f2": r.random(n),
              "b": r.random(n) > 0.5, "s": r.random(n) > 0.99, "mb": m > 0.5, "sm": r.random((4, 5)),
              "sv": r.random(3)}
    lines = ["| case | 1 thread | 10 threads | numpy |", "|---|---:|---:|---:|"]

    def fmt(t):
        return f"{t * 1e9:.0f}ns" if t < 1e-6 else f"{t * 1e6:.1f}µs"

    for label, sd_stmt, np_stmt, number in CASES:
        res = []
        for threads in (1, 10):
            sdnp.set_num_threads(threads)
            ns = {k: sdnp.asarray(v) for k, v in arrays.items()} | {"M": sdnp}
            res.append(min(timeit.repeat(sd_stmt, globals=ns, number=number, repeat=7)) / number)
        t_np = min(timeit.repeat(np_stmt, globals=dict(arrays) | {"np": np}, number=number, repeat=7)) / number
        lines.append(f"| {label} | {fmt(res[0])} | {fmt(res[1])} | {fmt(t_np)} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.save:
        (pathlib.Path(__file__).parent / "results" / f"tier2-{args.save}.md").write_text(text)


if __name__ == "__main__":
    main()
