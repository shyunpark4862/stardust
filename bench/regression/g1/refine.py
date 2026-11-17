"""G1: the accuracy of the OLS coefficients and standard errors against the number of
refinement steps on the augmented system (docs/design/regression.md §3.1, G15).

Problems: the 11 NIST StRD problems as a user gives them (decimal data and powers rounded to
double once, tests/regression/nist.py design_strings), and random problems X = U diag(s) V^T
with s geometric from 1 to 1/kappa (n = 100, k = 8, no intercept), b standard normal and a
residual r orthogonal to X with |r| = rho |X b| for rho = 1e-10 (small), 1 and 100 (large;
R^2 = 1/2 and 1e-4). The references are Arb's
(tools/regref.py) on the same double inputs. For each step count 0..3: the coefficient
vector's normwise relative error, the worst componentwise one, and the standard errors' worst
relative error. rtol=0 here (nothing aliased) so that the near-singular designs are fitted.

Run: uv run --no-sync python bench/regression/g1/refine.py [out.json]
"""

import json
import math
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

STEPS = (0, 1, 2, 3, 4)
KAPPA_EXPS = (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16)
RHOS = (1e-10, 1.0, 100.0)
SEEDS = 3


def problems():
    import numpy as np
    from tests.regression import nist

    out = []
    for name in nist.NAMES:
        d = nist.load(name)
        y, X = nist.design_strings(d)
        out.append((name, y, X, d["intercept"]))
    rng = np.random.default_rng(20261118)
    n, k = 100, 8
    for kappa_exp in KAPPA_EXPS:
        kappa = 10.0 ** kappa_exp
        for rho in RHOS:
            for seed in range(SEEDS):
                U, _ = np.linalg.qr(rng.standard_normal((n, n)))
                V, _ = np.linalg.qr(rng.standard_normal((k, k)))
                s = kappa ** (-np.arange(k) / (k - 1))
                X = (U[:, :k] * s) @ V.T
                b = rng.standard_normal(k)
                Xb = X @ b
                r = U[:, k:] @ rng.standard_normal(n - k)
                y = Xb + r * (rho * np.linalg.norm(Xb) / np.linalg.norm(r))
                out.append((f"k1e{kappa_exp}-rho{rho:g}-{seed}", y.tolist(), X.tolist(), False))
    return out


def reference(item):
    import regref
    import scipy.stats as st

    name, y, X, intercept = item
    k = len(X[0]) + (1 if intercept else 0)
    ref, meta = regref.ols(y, X, intercept=intercept, t_guess=float(st.t.ppf(0.975, max(len(y) - k, 1))))
    return name, {"params": [a + b for a, b in ref["params"]], "bse": [a + b for a, b in ref["bse"]],
                  "params_dd": ref["params"], "exact_fit": meta["exact_fit"]}


def measure(items, refs):
    import numpy as np
    import stardust as sd

    rows = []
    for name, y, X, intercept in items:
        ref = refs[name]
        bt = np.array(ref["params"])
        st_ = np.array(ref["bse"])
        row = {"name": name}
        for steps in STEPS:
            raw = sd._core.regression.fit("ols", sd.asarray(y), sd.asarray(X), None, None, intercept, 0.0, 0, steps)
            b = np.asarray(raw[0])
            stats = np.asarray(raw[6])
            dfr = stats[0] - stats[1]
            bse = np.sqrt(np.diag(np.asarray(raw[1])) * stats[2] / dfr)
            # componentwise against the double-double reference
            bdd = np.array([hi for hi, lo in ref["params_dd"]])
            bdl = np.array([lo for hi, lo in ref["params_dd"]])
            comp = np.max(np.abs((b - bdd) - bdl) / np.abs(bt))
            norm = np.linalg.norm((b - bdd) - bdl) / np.linalg.norm(bt)
            if ref["exact_fit"] or np.all(st_ == 0):
                bse_err = float(np.max(np.abs(bse - st_)))  # absolute: the reference is 0
            else:
                bse_err = float(np.max(np.abs(bse - st_) / st_))
            row[f"s{steps}"] = {"norm": float(norm), "comp": float(comp), "bse": bse_err}
        r = sd.regression.ols(y, X, intercept=intercept, rtol=0)
        row["kappa"] = float(r.condition_number)
        r2 = sd.regression.ols(y, X, intercept=intercept)
        row["rank_default_rtol"] = r2.rank
        row["k"] = len(bt)
        rows.append(row)
    return rows


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "refine.json")
    items = problems()
    with mp.get_context("spawn").Pool(6, maxtasksperchild=4) as pool:
        refs = dict(pool.map(reference, items, chunksize=1))
    rows = measure(items, refs)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    for row in rows:
        print(f"{row['name']:22s} kappa {row['kappa']:8.1e} rank(default) {row['rank_default_rtol']}/{row['k']}  " +
              "  ".join(f"s{s}: {row[f's{s}']['norm']:.1e}/{row[f's{s}']['comp']:.1e}/{row[f's{s}']['bse']:.1e}"
                        for s in STEPS))
