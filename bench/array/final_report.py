"""M10: one fixed set of operations, measured for numpy and for stardust as
built at the end of each milestone (see bench/stage_compare.sh).

    .venv/bin/python bench/final_report.py --label HEAD --out results/final/HEAD.json
    .venv/bin/python bench/final_report.py --numpy --out results/final/numpy.json
    .venv/bin/python bench/final_report.py --table results/final/*.json

A build from another commit is imported from --src (its src/ directory, with
_core copied in); the editable install of this checkout is bypassed then.
"""

import argparse
import json
import pathlib
import sys
import timeit

import numpy as np

CASES = [
    # (name, statement, repetitions)
    ("f + g", "f + g", 20),
    ("f * i + g  (casts)", "f * i + g", 10),
    ("ft + 1.0  (transposed)", "ft + 1.0", 20),
    ("maximum(f, g)", "M.maximum(f, g)", 20),
    ("exp(f)", "M.exp(f)", 5),
    ("sin(f)", "M.sin(f)", 5),
    ("sum(f)", "M.sum(f)", 20),
    ("sum(m, axis=1)", "M.sum(m, axis=1)", 20),
    ("max(f)", "M.max(f)", 20),
    ("var(f)", "M.var(f)", 10),
    ("x[idx]  gather", "f[idx]", 10),
    ("x[idx] = 1.0  scatter", "g[idx] = 1.0", 10),
    ("sort(f)", "M.sort(f)", 3),
    ("sort(s)  2000 rows", "M.sort(s)", 3),
    ("int64 matmul 300^3", "ia @ ia", 3),
    ("tiny a + b", "s1 + s2", 20000),
    ("tiny x[1, 2]", "s3[1, 2]", 20000),
]


def import_stardust(src):
    if src:
        # Drop the editable-install finder so `src` (another commit's build) is used.
        sys.meta_path[:] = [f for f in sys.meta_path if "ScikitBuild" not in type(f).__name__]
        sys.path.insert(0, src)
    import stardust_numpy

    return stardust_numpy


def arrays():
    rng = np.random.default_rng(0)
    n = 1_000_000
    return {
        "f": rng.random(n) + 0.5,
        "g": rng.random(n) + 0.5,
        "i": rng.integers(-1000, 1000, n),
        "ft": rng.random((1000, 1000)).T,
        "m": rng.random((1000, 1000)),
        "idx": rng.integers(0, n, n),
        "s": rng.random((2000, 1000)),
        "ia": rng.integers(-100, 100, (300, 300)),
        "s1": np.array([1.0, 2.0, 3.0]),
        "s2": np.array([4.0, 5.0, 6.0]),
        "s3": np.arange(12.0).reshape(3, 4),
    }


def measure(mod, convert):
    ns = {k: convert(v.copy()) for k, v in arrays().items()} | {"M": mod}
    return {
        name: min(timeit.repeat(stmt, globals=ns, number=number, repeat=7)) / number for name, stmt, number in CASES
    }


def fmt(t):
    return f"{t * 1e9:.0f}ns" if t < 1e-6 else (f"{t * 1e6:.0f}µs" if t < 1e-3 else f"{t * 1e3:.2f}ms")


def table(paths):
    runs = [json.loads(pathlib.Path(p).read_text()) for p in paths]
    runs.sort(key=lambda r: (r["label"] == "numpy", r.get("order", 99)))
    print("| case | " + " | ".join(r["label"] for r in runs) + " |")
    print("|---|" + "---:|" * len(runs))
    for name, _, _ in CASES:
        print(f"| {name} | " + " | ".join(fmt(r["times"][name]) for r in runs) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--order", type=int, default=99)
    ap.add_argument("--src")
    ap.add_argument("--numpy", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--table", nargs="+")
    ap.add_argument("--threads", type=int, help="stardust thread count (default: its own default)")
    args = ap.parse_args()
    if args.table:
        table(args.table)
        return
    if args.numpy:
        times, label = measure(np, np.asarray), "numpy"
    else:
        sd = import_stardust(args.src)
        if args.threads:
            sd.set_num_threads(args.threads)
        times, label = measure(sd, sd.asarray), args.label
    out = {"label": label, "order": args.order, "times": times}
    path = pathlib.Path(__file__).parent / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(label, "done")


if __name__ == "__main__":
    main()
