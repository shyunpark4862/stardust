"""Reference derivatives for the optimize accuracy measurement (bench/accuracy, area optimize).

    .venv/bin/python tools/optimize_accuracy_ref.py

writes tests/optimize/ref/derivatives.npz: for each MGH case with factor 1 (the 47 cases of
tests/optimize/test_diff.py), at its starting point x0, the Jacobian J of the residuals and the
gradient g = 2 J^T F of f = sum F_i^2, from the problems' analytic formulas (tests/optimize/mgh.py)
evaluated in Arb on the double x0 (tests/optimize/ad.py's ARB), settled to 2^-110 of the largest
entry (approx_fprime's error is measured normwise), as hi + lo. Keys per case i: g<i>_id, x0,
g_hi, g_lo, J_hi, J_lo (J row-major, m x n).

The other optimize references are files already: the Arb roots (tests/optimize/data/
scalar_roots.json, tools/optimize_ref.py), NIST StRD's certified values
(tests/regression/data/nls), MGH's optima (tests/optimize/mgh.py KNOWN).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from flint import arb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from arbref import MAX_PREC, START_PREC, NotSettled, precision, settled, to_dd  # noqa: E402

OUT = ROOT / "tests" / "optimize" / "ref"


def _settled_vec(fn):
    p = START_PREC
    while p <= MAX_PREC:
        with precision(p):
            vs = [v if isinstance(v, arb) else arb(v) for v in fn()]
            top = max((abs(float(v.mid())) for v in vs), default=0.0)
            if all(settled(v, floor=top) for v in vs):
                dd = [to_dd(v) for v in vs]
                return np.array([h for h, _ in dd]), np.array([lo for _, lo in dd])
        p *= 2
    raise NotSettled(fn)


def main():
    from tests.optimize import ad, mgh

    arrays = {}
    cases = [c for c in mgh.CASES if c.factor == 1.0]
    for i, c in enumerate(cases):
        x0 = [float(v) for v in c.x0()]

        def jac(c=c, x0=x0):
            x = [arb(v) for v in x0]
            return [e for row in c.jac(x, M=ad.ARB) for e in row]

        def grad(c=c, x0=x0):
            x = [arb(v) for v in x0]
            J, F = c.jac(x, M=ad.ARB), c.fun(x, M=ad.ARB)
            return [2 * sum((arb(J[k][j]) * F[k] for k in range(c.m)), arb(0)) for j in range(c.n)]

        jh, jl = _settled_vec(jac)
        gh, gl = _settled_vec(grad)
        arrays.update({f"g{i}_id": c.id, f"g{i}_x0": np.array(x0), f"g{i}_g_hi": gh, f"g{i}_g_lo": gl,
                       f"g{i}_J_hi": jh.reshape(c.m, c.n), f"g{i}_J_lo": jl.reshape(c.m, c.n)})  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "derivatives.npz", **arrays)
    print(f"{len(cases)} cases -> {(OUT / 'derivatives.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
