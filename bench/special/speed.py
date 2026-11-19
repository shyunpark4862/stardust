"""sd.special against scipy.special in one sitting (docs/design/special.md §6): the same
inputs, measured alternately round by round so the machine's drift hits both, best
time per row over the rounds.

    .venv/bin/python bench/special/speed.py [--rounds R] [--out FILE.md] [NAME ...]

Rows per function: 10^6 elements (sd on 1 and 10 threads; scipy.special runs on one)
and a single Python-float call. Inputs: the guaranteed-region reference inputs
(bench/special/common.py, sample_inputs). Keep the machine quiet.
"""

import argparse
import sys

import common
import numpy as np

N = 10**6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out")
    a = p.parse_args()
    import stardust as sd

    names = a.names or common.ref_names()
    best = {}
    for rnd in range(a.rounds):
        for name in names:
            print(f"round {rnd + 1}/{a.rounds}: {name}", file=sys.stderr, flush=True)  # progress, at once
            f, ref = common.sd_function(name), common.scipy_function(name)
            xs = common.sample_inputs(name, N)
            sxs = tuple(sd.asarray(x) for x in xs)
            one = tuple(complex(x[0]) if np.iscomplexobj(x) else float(x[0]) for x in xs)
            rows = {}
            for t in (1, 10):
                sd.set_num_threads(t)
                rows[f"sd{t}"] = common.best_ns(lambda: f(*sxs), repeat=5)
                rows["sd_scalar" if t == 1 else "_"] = common.best_ns(lambda: f(*one), repeat=5, number=20000)
            sd.set_num_threads(10)
            if ref is not None:
                rows["scipy"] = common.best_ns(lambda: ref(*xs), repeat=5)
                rows["scipy_scalar"] = common.best_ns(lambda: ref(*one), repeat=5, number=20000)
            row = best.setdefault(name, {})
            for k, v in rows.items():
                row[k] = min(row.get(k, float("inf")), v)
    lines = [
        f"Machine: {common.machine()}. 10^6 elements in ms, one scalar call in us; best of {a.rounds} rounds.",
        "",
        "| function | compared with | sd 1 thread | sd 10 threads | it | it / sd 1 | it / sd 10 | sd scalar | its scalar |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in best.items():
        s = r.get("scipy")
        ratio = (f"{s / r['sd1']:.2f} | {s / r['sd10']:.2f}") if s else "- | -"
        lines.append(
            f"| `{name}` | `{common.counterpart(name)}` | {r['sd1'] / 1e6:.3f} | {r['sd10'] / 1e6:.3f} | {s / 1e6 if s else float('nan'):.3f} | {ratio} | "
            f"{r['sd_scalar'] / 1e3:.3f} | {r.get('scipy_scalar', float('nan')) / 1e3:.3f} |"
        )
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
