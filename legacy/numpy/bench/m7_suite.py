"""M7: one fixed benchmark suite, run before and after every optimization.

Each step of M7 saves its numbers, so the effect of every single change can
be read off as a diff (bench/results/m7/<step>.md):

    .venv/bin/python bench/m7_suite.py --save 00-baseline
    .venv/bin/python bench/m7_suite.py --compare 00-baseline 01-contiguous

One thread throughout (M8 adds threads). Times are the best of 7 runs.
"""

import argparse
import math
import pathlib
import timeit

import numpy as np

import stardust_numpy as sdnp

RESULTS = pathlib.Path(__file__).parent / "results" / "m7"
N = 1_000_000


def data():
    rng = np.random.default_rng(0)
    f = rng.random(N) + 0.5
    return {
        "f": f, "g": rng.random(N) + 0.5, "i": rng.integers(-1000, 1000, N), "j": rng.integers(1, 1000, N),
        "c": f + 1j * rng.random(N), "d": rng.random(N) + 1j * rng.random(N),
        "ft": rng.random((1000, 1000)).T, "m": rng.random((1000, 1000)), "b": rng.random(N) > 0.5,
        "idx": rng.integers(0, N, N), "iv": rng.integers(0, 1000, N),
        "s1": np.array([1.0, 2.0, 3.0]), "s2": np.array([4.0, 5.0, 6.0]), "s3": np.arange(12.0).reshape(3, 4),
        "fnan": np.where(rng.random(N) < 0.01, np.nan, f),
    }


# (name, statement, repetitions). Statements run with M = the module and the
# arrays above as globals.
CASES = [
    # copies and casts
    ("copy  x.copy()", "f.copy()", 20),
    ("cast  int64->float64", "M.astype(i, M.float64)", 20),
    # elementwise, contiguous
    ("f + g", "f + g", 20), ("f * g", "f * g", 20), ("f / g", "f / g", 20), ("f + 1.0", "f + 1.0", 20),
    ("i + i", "i + i", 20), ("i * j", "i * j", 20), ("i // j", "i // j", 10), ("i & j", "i & j", 20),
    ("f < g", "f < g", 20), ("maximum(f, g)", "M.maximum(f, g)", 20), ("-f", "-f", 20), ("abs(f)", "abs(f)", 20),
    ("sqrt(f)", "M.sqrt(f)", 20), ("floor(f)", "M.floor(f)", 20), ("round(f)", "M.round(f)", 20),
    ("c + d", "c + d", 20), ("c * d", "c * d", 20), ("abs(c)", "abs(c)", 10),
    # elementwise, other layouts and dtypes
    ("ft + 1.0  (transposed)", "ft + 1.0", 20), ("i + f  (int64 -> float64)", "i + f", 20),
    ("b & b  (bool)", "b & b", 20),
    # reductions
    ("sum(f)", "M.sum(f)", 20), ("sum(i)", "M.sum(i)", 20), ("sum(c)", "M.sum(c)", 20),
    ("sum(m, axis=0)", "M.sum(m, axis=0)", 20), ("sum(m, axis=1)", "M.sum(m, axis=1)", 20),
    ("max(f)", "M.max(f)", 20), ("max(fnan)", "M.max(fnan)", 20), ("argmax(f)", "M.argmax(f)", 20),
    ("mean(f)", "M.mean(f)", 20), ("var(f)", "M.var(f)", 10),
    ("mean(i)  (int64 -> float64)", "M.mean(i)", 20), ("sum(b)  (bool -> int64)", "M.sum(b)", 20),
    ("f * i + g  (two casts)", "f * i + g", 20),
    # indexing, sorting
    ("x[idx]  gather", "f[idx]", 10), ("x[idx] = 1.0  scatter", "g[idx] = 1.0", 10),
    ("x[b]  mask", "f[b]", 10), ("sort(f)", "M.sort(f)", 3), ("sort(fnan)", "M.sort(fnan)", 3),
    ("unique_values(iv)", "M.unique_values(iv)", 3),
    # fixed cost per call (tiny arrays)
    ("tiny  a + b", "s1 + s2", 20000), ("tiny  a + 1.0", "s1 + 1.0", 20000),
    ("tiny  sqrt(a)", "M.sqrt(s1)", 20000), ("tiny  x[1, 2]", "s3[1, 2]", 20000), ("tiny  x[1]", "s3[1]", 20000),
    ("tiny  sum(a)", "M.sum(s1)", 20000),
]


def measure():
    arrays = data()
    sd = {k: sdnp.asarray(v.copy()) for k, v in arrays.items()} | {"M": sdnp}
    np_ = {k: v.copy() for k, v in arrays.items()} | {"M": np}
    rows = []
    for name, stmt, number in CASES:
        t_sd = min(timeit.repeat(stmt, globals=sd, number=number, repeat=7)) / number
        t_np = min(timeit.repeat(stmt, globals=np_, number=number, repeat=7)) / number
        rows.append((name, t_sd, t_np))
    # accuracy of a float sum (relative error against math.fsum)
    x = np.full(10**7, 0.1) + np.random.default_rng(1).random(10**7) * 1e-3
    exact = math.fsum(x.tolist())
    acc = (abs(sdnp.sum(sdnp.asarray(x)) - exact) / exact, abs(float(np.sum(x)) - exact) / exact)
    return rows, acc


def fmt_time(t):
    return f"{t * 1e9:.0f}ns" if t < 1e-6 else (f"{t * 1e6:.1f}µs" if t < 1e-3 else f"{t * 1e3:.2f}ms")


def render(rows, acc):
    lines = ["| case | stardust | numpy | ratio |", "|---|---:|---:|---:|"]
    for name, t_sd, t_np in rows:
        lines.append(f"| {name} | {fmt_time(t_sd)} | {fmt_time(t_np)} | {t_sd / t_np:.2f}x |")
    lines.append("")
    lines.append(f"sum of 1e7 values, relative error: stardust {acc[0]:.1e}, numpy {acc[1]:.1e}")
    return "\n".join(lines) + "\n"


def parse(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("| ") and not line.startswith("| case"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            out[cells[0]] = cells[1]
    return out


def to_seconds(s):
    for unit, k in (("ns", 1e-9), ("µs", 1e-6), ("ms", 1e-3)):
        if s.endswith(unit):
            return float(s[: -len(unit)]) * k
    raise ValueError(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="STEP")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()
    if args.compare:
        a, b = (parse(RESULTS / f"{s}.md") for s in args.compare)
        print(f"| case | {args.compare[0]} | {args.compare[1]} | speedup |\n|---|---:|---:|---:|")
        for name in a:
            if name in b:
                print(f"| {name} | {a[name]} | {b[name]} | {to_seconds(a[name]) / to_seconds(b[name]):.2f}x |")
        return
    text = render(*measure())
    print(text)
    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{args.save}.md").write_text(text)


if __name__ == "__main__":
    main()
