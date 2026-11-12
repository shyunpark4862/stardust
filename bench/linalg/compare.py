"""Compares two baseline runs row by row (sdla's own times).

Usage: .venv/bin/python bench/compare.py bench/results/baseline_m7.txt bench/results/baseline_m8.txt [--all]
Prints rows whose time changed by more than 10% (the noise band between two runs
of the same build, DESIGN.md §2.16), or every row with --all.
"""

import sys


def load(path):
    rows = {}
    for line in open(path).read().splitlines()[2:]:
        f = line.split()
        i = next(k for k, t in enumerate(f) if t.startswith("(") or t.startswith("n="))
        ref = None if f[i + 3] == "-" else float(f[i + 3])
        rows[(" ".join(f[:i]), f[i])] = (float(f[i + 1]), ref)
    return rows


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    show_all = "--all" in sys.argv
    print(f"{'func':18} {'shape':>14} {'before us':>11} {'after us':>11} {'speedup':>8} {'ref/sdla':>9}")
    for key in a:
        if key not in b:
            continue
        t0, t1 = a[key][0], b[key][0]
        r = b[key][1]
        if show_all or abs(t0 / t1 - 1) > 0.10:
            rs = "-" if r is None else f"{r:9.2f}"
            print(f"{key[0]:18} {key[1]:>14} {t0:11.2f} {t1:11.2f} {t0 / t1:8.2f} {rs:>9}")


if __name__ == "__main__":
    main()
