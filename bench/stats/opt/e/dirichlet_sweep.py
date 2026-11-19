"""Group E: the Dirichlet's log density (the plain-double sum where its bound fits kFastBudget,
the double-double sum elsewhere) against Arb at 128 bits (each reference's radius checked
below 2^-80 of max(1, |value|)): bulk, corners of the simplex, tiny, large and mixed shapes.
Prints the largest error of logpdf in the contract's mixed units and of pdf relative (the
contract: 1e-14).

    .venv/bin/python bench/stats/opt/e/dirichlet_sweep.py [points per case, default 2500]
"""

import sys

import numpy as np
from flint import arb, ctx

import stardust as sd

S = sd.stats
BITS = 80


def _checked(v):
    """v, whose radius must be below 2^-BITS of max(1, |v|) (else the working precision is too low)."""
    if not (v.is_finite() and v.rad() <= arb(2) ** -BITS * max(arb(1), abs(v.mid()))):
        raise ArithmeticError(f"reference not settled: {v}")
    return v


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    ctx.prec = 128
    rng = np.random.default_rng(20261116)
    worst_l = worst_p = 0.0
    total = 0
    print("| K | shapes | logpdf max error (mixed) | pdf max error (rel) |")
    print("|---:|---|---:|---:|")
    for K in (2, 3, 4, 8):
        kinds = {
            "0.5..4": rng.uniform(0.5, 4, K),
            "0.01..1": rng.uniform(0.01, 1, K),
            "1..200": rng.uniform(1, 200, K),
            "1e-3..1e3": 10.0 ** rng.uniform(-3, 3, K),
            "with 1s": np.r_[1.0, rng.uniform(0.2, 3, K - 1)],
        }
        for name, al in kinds.items():
            x = rng.dirichlet(np.maximum(al, 0.05), size=n)
            x[: n // 10] = rng.dirichlet(np.ones(K), size=n // 10)  # off the bulk
            x[n // 10 : n // 5] = rng.dirichlet(np.full(K, 0.05), size=n // 10)  # the corners
            x = np.maximum(x, 1e-300)
            x = x / x.sum(axis=1, keepdims=True)
            D = S.Dirichlet(alpha=sd.asarray(al))
            got, gotp = np.asarray(D.logpdf(sd.asarray(x))), np.asarray(D.pdf(sd.asarray(x)))
            am = [arb(float(a)) for a in al]
            c = sum(am, arb(0)).lgamma() - sum((a.lgamma() for a in am), arb(0))
            el = ep = 0.0
            for i in range(n):
                if not np.isfinite(got[i]):
                    continue
                want = _checked(c + sum(((a - 1) * arb(float(v)).log() for a, v in zip(am, x[i])), arb(0)))
                el = max(el, float((abs(arb(got[i]) - want) / max(arb(1), abs(want))).mid()))
                if 1e-300 < gotp[i] < 1e300:
                    ew = want.exp()
                    ep = max(ep, float((abs(arb(gotp[i]) - ew) / ew).mid()))
            worst_l, worst_p, total = max(worst_l, el), max(worst_p, ep), total + n
            print(f"| {K} | {name} | {el:.2e} | {ep:.2e} |", flush=True)
    print(f"\n{total:,} points. Largest error: logpdf {worst_l:.3e} (mixed), pdf {worst_p:.3e} (relative).")


if __name__ == "__main__":
    main()
