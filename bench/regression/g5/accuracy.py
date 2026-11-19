"""G5: quantreg's accuracy (docs/design/regression.md §3.6). Results: bench/regression/g5/accuracy.md
(the raw output in accuracy.log).

  - Exactness: on every problem the returned basis is solved again in rationals
    (tools/regref.py's quantreg): the exact vertex, residuals, objective and the optimality
    certificate (and uniqueness). The coefficients' error is ||params - exact|| / ||exact||, the
    objective's relative, the basis rows' residuals over |y_i| + |x_i| |params|.
  - R (tests/regression/ref/quantreg_r.json, quantreg 6.1): rq "br" coefficients, summary.rq
    standard errors ('nid' with both bandwidths, 'iid' and 'ker' at sd's vertex).
  - statsmodels' QuantReg (IRLS, defaults): its coefficients' distance to the exact vertex and
    its objective's excess.
Problems: the R problems, random designs (n 50 - 5000, k 2 - 10; normal, t(2), exponential
noise; q 0.05 - 0.95), a degree-5 polynomial (kappa ~ 1e5), integer outcomes on dummies.

Run: uv run --no-sync python bench/regression/g5/accuracy.py
"""

import json
import math
import pathlib
import sys
import warnings

import numpy as np

import stardust as sd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import regref  # noqa: E402

U = 2.0**-53


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def problems():
    data = json.loads((ROOT / "tests" / "regression" / "ref" / "quantreg_r.json").read_text())
    for name, p in data.items():
        if name.startswith("_"):
            continue
        y = np.asarray(p["y"], float)
        yield "R: " + name, y, np.asarray(p["X"], float).reshape(len(y), -1), p["q"], p["intercept"]
    rng = np.random.default_rng(2028)
    for n, k in ((50, 2), (400, 5), (5000, 10)):
        for noise in ("normal", "t2", "exp"):
            X = rng.standard_normal((n, k))
            e = {"normal": rng.standard_normal(n), "t2": rng.standard_t(2, n), "exp": rng.exponential(1.0, n)}[noise]
            yield f"random ({n}, {k}) {noise}", X @ rng.standard_normal(k) + e, X, [0.05, 0.25, 0.5, 0.75, 0.95], True
    t = rng.uniform(0, 1, 400)
    yield (
        "polynomial degree 5",
        np.sin(3 * t) + 0.1 * rng.standard_normal(400),
        np.column_stack([t**j for j in range(1, 6)]),
        [0.1, 0.5, 0.9],
        True,
    )
    g = rng.integers(0, 4, 500).astype(float)
    yield (
        "poisson on dummies",
        rng.poisson(2 + g).astype(float),
        np.column_stack([(g == v).astype(float) for v in (1, 2, 3)]),
        [0.2, 0.5, 0.8],
        True,
    )


def main():
    import statsmodels.api as sm

    print("## exactness and statsmodels")
    print(
        "| problem | fits | certified optimal | nonunique (sd / exact) | params error | objective error | "
        "basis residual | statsmodels params | statsmodels objective excess |"
    )
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    tot = dict(pe=0.0, oe=0.0, br=0.0)
    for name, y, X, qs, icpt in problems():
        pe = oe = br = smd = sme = 0.0
        nopt = 0
        nus, nue = 0, 0
        for q in qs:
            r = sd.regression.quantreg(y, X, q=q, intercept=icpt)
            b = [int(i) for i in a(r._raw[7]).astype(int) if i >= 0]
            ref = regref.quantreg(y.tolist(), X.tolist(), b, q, intercept=icpt)
            hi = np.array([v[0] for v in ref["params"]])
            lo = np.array([v[1] for v in ref["params"]])
            pe = max(pe, np.linalg.norm((a(r.params) - hi) - lo) / np.linalg.norm(hi))
            obj = sum(ref["objective"])
            oe = max(oe, abs(r.objective - obj) / obj if obj else abs(r.objective))
            Z = np.column_stack([np.ones(len(y)), X]) if icpt else X
            sc = np.abs(y[b]) + np.abs(Z[b]) @ np.abs(a(r.params))
            br = max(br, float(np.max(np.abs(a(r.resid)[b]) / sc)))
            nopt += ref["optimal"] and r.converged
            nus += bool(r.nonunique)
            nue += (not ref["unique"]) or ref["margin"] <= 1e-14
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = sm.QuantReg(y, Z).fit(q=q)
            if ref["unique"] and ref["margin"] > 1e-14:
                smd = max(smd, np.max(np.abs(f.params - hi)) / np.max(np.abs(hi)))
            res = y - Z @ f.params
            sme = max(sme, float(np.sum(res * (q - (res < 0)))) / obj - 1 if obj else 0.0)
        tot["pe"], tot["oe"], tot["br"] = max(tot["pe"], pe), max(tot["oe"], oe), max(tot["br"], br)
        print(
            f"| {name} | {len(qs)} | {nopt} | {nus} / {nue} | {pe:.1e} | {oe:.1e} | {br:.1e} | {smd:.1e} | {sme:.1e} |",
            flush=True,
        )
    print(f"\nmax: params {tot['pe']:.1e}, objective {tot['oe']:.1e}, basis residual {tot['br']:.1e}")

    print("\n## R (quantreg 6.1): largest relative differences")
    data = json.loads((ROOT / "tests" / "regression" / "ref" / "quantreg_r.json").read_text())
    print("| problem | br params (unique) | nid | nid (Bofinger) | iid at sd's vertex | ker at sd's vertex |")
    print("|---|---:|---:|---:|---:|---:|")

    def rel(g, ref):
        ref = np.array([np.nan if v is None else v for v in ref], float)
        m = np.isfinite(ref)
        return float(np.max(np.abs(a(g)[m] - ref[m]) / np.abs(ref[m]))) if m.any() else math.nan

    for name, p in data.items():
        if name.startswith("_"):
            continue
        y = np.asarray(p["y"], float)
        X = np.asarray(p["X"], float).reshape(len(y), -1)
        vals = dict(br=0.0, nid=0.0, nidb=0.0, iid=0.0, ker=0.0)
        for j, q in enumerate(p["q"]):
            t, Rv = f"t{j}_", p["R"]
            kw = dict(q=q, intercept=p["intercept"])
            r = sd.regression.quantreg(y, X, **kw)
            if not r.nonunique:
                ref = np.array(Rv[t + "br"])
                vals["br"] = max(vals["br"], np.linalg.norm(a(r.params) - ref) / np.linalg.norm(ref))
            vals["nid"] = max(vals["nid"], rel(r.bse, Rv[t + "se_nid"]))
            vals["nidb"] = max(
                vals["nidb"], rel(sd.regression.quantreg(y, X, bandwidth="bofinger", **kw).bse, Rv[t + "se_nid_bof"])
            )
            vals["iid"] = max(
                vals["iid"], rel(sd.regression.quantreg(y, X, cov_type="iid", **kw).bse, Rv[t + "sdv_se_iid"])
            )
            vals["ker"] = max(
                vals["ker"], rel(sd.regression.quantreg(y, X, cov_type="ker", **kw).bse, Rv[t + "sdv_se_ker"])
            )
        print(
            f"| {name} | {vals['br']:.1e} | {vals['nid']:.1e} | {vals['nidb']:.1e} | {vals['iid']:.1e} | "
            f"{vals['ker']:.1e} |"
        )


if __name__ == "__main__":
    main()
