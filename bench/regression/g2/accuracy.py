"""G2: glm against the Arb references of tests/regression/ref/glm.json (docs/design/regression.md
§3.4). Per problem: the coefficients' normwise relative error, and the largest relative error
of the standard errors, p-values, deviance, Pearson chi2, log-likelihood (with the null
deviance and null log-likelihood), and of the robust standard errors (HC0-HC3, HAC, cluster).

Run: uv run --no-sync python bench/regression/g2/accuracy.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tests.regression import refdata, refdata_glm  # noqa: E402


def nerr(got, ref):
    hi = np.array([v[0] for v in ref])
    lo = np.array([v[1] for v in ref])
    return float(np.linalg.norm((np.asarray(got, float) - hi) - lo) / np.linalg.norm(hi + lo))


def rerr(got, ref):
    return float(max(refdata.rel_err(g, r) for g, r in zip(np.asarray(got, float).ravel(), ref)))


def main():
    refs = refdata_glm.load()
    print("| problem | n_iter | params | bse | pvalues | deviance, pearson_chi2, scale | llf, llnull, null_deviance | "
          "robust bse |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    worst = {}
    for name, p in refs.items():
        ref = p["ref"]
        r = refdata_glm.fit(p)
        e = {"params": nerr(r.params, ref["params"]), "bse": rerr(r.bse, ref["bse"]),
             "pvalues": rerr(r.pvalues, ref["pvalues"]),
             "fit": max(refdata.rel_err(getattr(r, s), ref[s]) for s in ("deviance", "pearson_chi2", "scale")),
             "llf": max(refdata.rel_err(getattr(r, s), ref[s]) for s in ("llf", "llnull", "null_deviance"))}
        rob = 0.0
        for ct in ("HC0", "HC1", "HC2", "HC3", "HAC"):
            rob = max(rob, rerr(refdata_glm.fit(p, cov_type=ct).bse, ref["bse_" + ct]))
        rob = max(rob, rerr(refdata_glm.fit(p, cov_type="cluster", groups=p["groups"]).bse, ref["bse_cluster"]))
        e["robust"] = rob
        for k, v in e.items():
            worst[k] = max(worst.get(k, 0.0), v)
        print(f"| {name} | {r.n_iter} | {e['params']:.1e} | {e['bse']:.1e} | {e['pvalues']:.1e} | {e['fit']:.1e} | "
              f"{e['llf']:.1e} | {e['robust']:.1e} |")
    print(f"| largest | | {worst['params']:.1e} | {worst['bse']:.1e} | {worst['pvalues']:.1e} | {worst['fit']:.1e} | "
          f"{worst['llf']:.1e} | {worst['robust']:.1e} |")


if __name__ == "__main__":
    main()
