"""Tier 3 items 12-15 of docs/learn/next-optimizations.md: radix sort,
parallel 1-D sort, hash unique, batched small matmul. 1 and 10 threads
against numpy.

Run:  .venv/bin/python bench/tier3.py [--save NAME]
"""

import argparse
import pathlib
import timeit

import numpy as np

import stardust_numpy as sdnp

N = 1_000_000
r = np.random.default_rng(0)
data = {
    "iw": r.integers(-10**15, 10**15, N), "i": r.integers(-1000, 1000, N), "f": r.random(N) + 0.5,
    "iv": r.integers(0, 1000, N), "iu": r.integers(-10**12, 10**12, N), "im": r.integers(-1000, 1000, (1000, 1000)),
    "bm": r.random((1000, 8, 8)), "bm32": r.random((200, 32, 32)),
}
S = {k: sdnp.asarray(v) for k, v in data.items()}

CASES = [  # label, stardust, numpy, number
    ("sort(i) wide range", lambda: sdnp.sort(S["iw"]), lambda: np.sort(data["iw"]), 5),
    ("sort(i) [-1000, 1000)", lambda: sdnp.sort(S["i"]), lambda: np.sort(data["i"]), 5),
    ("argsort(i) wide range", lambda: sdnp.argsort(S["iw"]), lambda: np.argsort(data["iw"], kind="stable"), 5),
    ("argsort(i) [-1000, 1000)", lambda: sdnp.argsort(S["i"]), lambda: np.argsort(data["i"], kind="stable"), 5),
    ("sort(im, axis=1) 1000x1000", lambda: sdnp.sort(S["im"], axis=1), lambda: np.sort(data["im"], axis=1), 5),
    ("argsort(f)", lambda: sdnp.argsort(S["f"]), lambda: np.argsort(data["f"], kind="stable"), 3),
    ("sort(f)", lambda: sdnp.sort(S["f"]), lambda: np.sort(data["f"], kind="stable"), 3),
    ("unique_counts(iv) (1000 values)", lambda: sdnp.unique_counts(S["iv"]), lambda: np.unique_counts(data["iv"]), 5),
    ("unique_values(iv)", lambda: sdnp.unique_values(S["iv"]), lambda: np.unique_values(data["iv"]), 5),
    ("unique_counts(iu) (all distinct)", lambda: sdnp.unique_counts(S["iu"]), lambda: np.unique_counts(data["iu"]), 3),
    ("unique_all(iv)", lambda: sdnp.unique_all(S["iv"]), lambda: np.unique_all(data["iv"]), 3),
    ("batch 1000 x 8x8 @", lambda: S["bm"] @ S["bm"], lambda: data["bm"] @ data["bm"], 20),
    ("batch 200 x 32x32 @", lambda: S["bm32"] @ S["bm32"], lambda: data["bm32"] @ data["bm32"], 20),
]


def best(fn, number):
    return min(timeit.repeat(fn, number=number, repeat=5)) / number


def fmt(t):
    return f"{t * 1e3:.2f}ms" if t >= 1e-3 else f"{t * 1e6:.1f}µs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    args = ap.parse_args()
    lines = ["| case | 1 thread | 10 threads | numpy |", "|---|---:|---:|---:|"]
    for label, sd_fn, np_fn, number in CASES:
        row = []
        for n in (1, 10):
            sdnp.set_num_threads(n)
            row.append(best(sd_fn, number))
        lines.append(f"| {label} | {fmt(row[0])} | {fmt(row[1])} | {fmt(best(np_fn, number))} |")
        print(lines[-1], flush=True)
    if args.save:
        out = pathlib.Path(__file__).parent / "results" / f"{args.save}.md"
        out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
