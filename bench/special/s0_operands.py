"""S0: what raising kMaxOperands from 4 to 5 (and the weight argument) costs the
array's own short elementwise calls (docs/design/special.md §3.1, §10).

The iterator keeps a stride array of kMaxNdim (32) int64 per operand: one more operand
is 256 bytes more NdIter, copied once per parallel task, and one more slot in the
buffered loop's fixed arrays. The calls below are short, so that fixed cost would show.

    .venv/bin/python bench/special/s0_operands.py BASE_SITE [--rounds R] [--only TEXT] [--out FILE.md]

BASE_SITE is an unpacked wheel of the commit before S0 (ROADMAP.md, S0; made with
`pip wheel` from a worktree). Baseline and current build run alternately, each in its
own process, round after round; the table keeps each row's best.
"""

import argparse
import json
import subprocess
import sys

import common


def cases(sd, np):
    rng = np.random.default_rng(0)

    def arr(*shape):
        return sd.asarray(rng.normal(size=shape))

    x1, y1 = arr(1), arr(1)
    x16, y16 = arr(16), arr(16)
    x1k, y1k = arr(1000), arr(1000)
    i16 = sd.asarray(rng.integers(0, 9, size=16))
    c16 = x16 > 0
    t3a = sd.asarray(rng.normal(size=(8, 8, 8)).transpose(2, 0, 1))
    t3b = arr(8, 8, 8)
    big_a, big_b = arr(10**6), arr(10**6)
    return {
        "add(1.0, 2.0) scalars": (1, lambda: sd.add(1.0, 2.0)),
        "add size 1": (1, lambda: sd.add(x1, y1)),
        "add size 16": (1, lambda: sd.add(x16, y16)),
        "x16 + y16 (operator)": (1, lambda: x16 + y16),
        "add size 1000": (1, lambda: sd.add(x1k, y1k)),
        "add int64 + float64 size 16 (buffered)": (1, lambda: sd.add(i16, y16)),
        "exp size 16": (1, lambda: sd.exp(x16)),
        "where size 16": (1, lambda: sd.where(c16, x16, y16)),
        "clip size 16": (1, lambda: sd.clip(x16, min=-0.5, max=0.5)),
        "add (8,8,8) transposed + C": (1, lambda: sd.add(t3a, t3b)),
        "sum size 1000": (1, lambda: sd.sum(x1k)),
        "add 10^6, 10 threads": (10, lambda: sd.add(big_a, big_b)),
    }


def measure(site, only):
    if site != "current":
        common.use_build(site)
    import numpy as np
    import stardust as sd

    out = {}
    for name, (threads, f) in cases(sd, np).items():
        if only and only not in name:
            continue
        sd.set_num_threads(threads)
        number = 50 if "10^6" in name else 20000
        out[name] = common.best_ns(f, repeat=7, number=number)
    print(json.dumps({"core": sd._core.__file__, "ns": out}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--out")
    p.add_argument("--only", default="", help="rows whose name contains this")
    p.add_argument("--measure")
    a = p.parse_args()
    if a.measure:
        return measure(a.measure, a.only)
    best = {"base": {}, "new": {}}
    for r in range(a.rounds):
        for label, site in (("base", a.base), ("new", "current")):
            res = json.loads(
                subprocess.run(
                    [sys.executable, __file__, a.base, "--measure", site, "--only", a.only],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            if r == 0:
                print(label, res["core"], file=sys.stderr)
            for k, v in res["ns"].items():
                best[label][k] = min(best[label].get(k, float("inf")), v)
    lines = [
        f"Machine: {common.machine()}. ns per call, best of {a.rounds} alternating rounds.",
        "",
        "| call | before S0 (4 operands) | S0 (5 operands) | S0 / before |",
        "|---|---:|---:|---:|",
    ]
    for k in best["base"]:
        b, n = best["base"][k], best["new"][k]
        lines.append(f"| {k} | {b:.0f} | {n:.0f} | {n / b:.3f} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
