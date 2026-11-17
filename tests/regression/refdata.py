"""Arb references for the linear models' tests (tools/regref.py; docs/design/regression.md §6).

tests/regression/ref/linear.json holds, per problem, the inputs as doubles (y, X rows without
the constant column, weights, intercept) and regref.ols's values as [hi, lo] pairs: random
problems (OLS with and without the intercept, WLS with zero weights, an ill-conditioned one
with a large residual) and the 11 NIST StRD problems as a user gives them (decimal data and
powers rounded to double once). Regenerate with

    uv run --no-sync python -m tests.regression.refdata
"""

import json
import os
import pathlib
import sys

REF = pathlib.Path(__file__).parent / "ref" / "linear.json"


def problems():
    import numpy as np

    from . import nist

    out = {}
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3))
    y = X @ [1.0, -2.0, 0.5] + 0.3 + rng.standard_normal(30)
    out["ols_int"] = (y.tolist(), X.tolist(), None, True)
    out["ols_noint"] = (y.tolist(), X.tolist(), None, False)
    w = rng.uniform(0.2, 3.0, 30)
    w[[4, 17]] = 0.0
    out["wls_zero"] = (y.tolist(), X.tolist(), w.tolist(), True)
    w2 = rng.uniform(0.2, 3.0, 30)
    out["wls_noint"] = (y.tolist(), X.tolist(), w2.tolist(), False)
    # kappa 1e5 with a residual as large as X b (R^2 = 1/2), an intercept
    n, k = 50, 4
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    Xc = (U[:, :k] * 1e5 ** (-np.arange(k) / (k - 1))) @ V.T
    b = rng.standard_normal(k)
    r = U[:, k:] @ rng.standard_normal(n - k)
    yc = Xc @ b + r * (np.linalg.norm(Xc @ b) / np.linalg.norm(r))
    out["ols_cond"] = (yc.tolist(), Xc.tolist(), None, False)
    # the contract's edge (G15): kappa 1e11 with a residual 100 times X b (R^2 = 1e-4)
    k = 6
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    Xe = (U[:, :k] * 1e11 ** (-np.arange(k) / (k - 1))) @ V.T
    b = rng.standard_normal(k)
    r = U[:, k:] @ rng.standard_normal(n - k)
    ye = Xe @ b + r * (100.0 * np.linalg.norm(Xe @ b) / np.linalg.norm(r))
    out["ols_kappa11"] = (ye.tolist(), Xe.tolist(), None, False)
    for name in nist.NAMES:
        d = nist.load(name)
        ys, Xs = nist.design_strings(d)
        out["nist_" + name] = (ys, Xs, None, d["intercept"])
    return out


def _one(item):
    name, (y, X, w, intercept) = item
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    import regref
    import scipy.stats as st

    k = len(X[0]) + (1 if intercept else 0)
    m = sum(1 for v in w if v > 0) if w is not None else len(y)
    ref, meta = regref.ols(y, X, weights=w, intercept=intercept, t_guess=float(st.t.ppf(0.975, max(m - k, 1))))
    vals = {key: ([list(p) for p in v] if isinstance(v, list) else list(v)) for key, v in ref.items()}
    return name, {"y": y, "X": X, "weights": w, "intercept": intercept, "meta": meta, "ref": vals}


def generate():
    import multiprocessing as mp

    items = list(problems().items())
    with mp.get_context("spawn").Pool(6, maxtasksperchild=4) as pool:
        data = dict(pool.map(_one, items, chunksize=1))
    REF.parent.mkdir(exist_ok=True)
    with open(REF, "w") as fh:
        json.dump(data, fh)


def load():
    with open(REF) as fh:
        return json.load(fh)


def value(hl):
    """A reference [hi, lo] as one float."""
    return hl[0] + hl[1]


def rel_err(got, hl):
    """|got - ref| / |ref| with the reference in double-double."""
    hi, lo = hl
    ref = hi + lo
    d = (got - hi) - lo
    return abs(d) / abs(ref) if ref != 0 else abs(d)


if __name__ == "__main__":
    generate()
    print(REF, os.path.getsize(REF), "bytes")
