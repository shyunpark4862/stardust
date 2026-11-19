"""S2: how erfc's kernel evaluates its rationals (docs/design/special.md §3.4).

    erfc           Horner, each element branching to its own region -- shipped
    _erfc_select   Horner in NEON lanes, every region's rational computed, then selected (same bits)
    _erfc_estrin   the same with Estrin's scheme (different bits)

Timed alternately round by round on one thread, 10^6 elements, for inputs that stay in
one region and inputs that mix them; plus the accuracy of the Estrin variant against the
Arb reference (tests/special/ref/erfc.npz).

    .venv/bin/python bench/special/s2_variants.py [--rounds R] [--out FILE.md]
"""

import argparse

import numpy as np

import common

N = 10**6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--out")
    a = p.parse_args()
    import stardust as sd

    sd.set_num_threads(1)
    C = sd.special._core
    variants = {
        "erfc (branch, shipped)": sd.special.erfc,
        "NEON, every region, select": C._erfc_select,
        "NEON Estrin, every region, select": C._erfc_estrin,
    }
    rng = np.random.default_rng(0)
    inputs = {
        "|x| < 1": rng.uniform(-1, 1, N),
        "1 <= x < 8": rng.uniform(1, 8, N),
        "x >= 8": rng.uniform(8, 27, N),
        "mixed, sorted": np.sort(rng.normal(scale=6, size=N)),
        "mixed, random": rng.normal(scale=6, size=N),
    }
    sx = {k: sd.asarray(v) for k, v in inputs.items()}
    best = {}
    for _ in range(a.rounds):
        for iname, x in sx.items():
            for vname, f in variants.items():
                t = common.best_ns(lambda: f(x), repeat=3) / N
                best[(iname, vname)] = min(best.get((iname, vname), float("inf")), t)
    lines = [
        f"Machine: {common.machine()}. ns per element, 1 thread, 10^6 elements, best of {a.rounds} rounds.",
        "",
        "| inputs | " + " | ".join(variants) + " |",
        "|---|" + "---:|" * len(variants),
    ]
    for iname in inputs:
        lines.append(f"| {iname} | " + " | ".join(f"{best[(iname, v)]:.2f}" for v in variants) + " |")
    xs, hi, lo, _ = common.ref_load("erfc")
    import sys

    sys.path.insert(0, str(common.ROOT))
    from tests.special import refdata

    lines += ["", "| accuracy on tests/special/ref/erfc.npz | max ULP | 99.9% ULP |", "|---|---:|---:|"]
    for vname, f in variants.items():
        e = refdata.ulp_error(np.asarray(f(sd.asarray(xs[0]))), hi, lo)
        lines.append(f"| {vname} | {e.max():.2f} | {np.quantile(e, 0.999):.2f} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
