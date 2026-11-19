"""G4: rlm's accuracy against Arb (tests/regression/ref/robust.json), statsmodels' RLM at a tight
tolerance, R's MASS::rlm and robustbase::lmrob (tests/regression/ref/robust_r.json), and the
slopes against scipy (docs/design/regression.md §3.6). Results: bench/regression/g4/accuracy.md
(the raw output in accuracy.log).

Arb: params as the relative error of the vector (2-norm), scale, bse and weights as the largest
relative (weights: absolute) error. statsmodels: RLM(...).fit(tol=1e-15, maxiter=1000),
HuberScale(d=2.5, tol=1e-15, maxiter=1000) for Huber's proposal 2, on data with 10% vertical
outliers (n = 150, 157, 400; k = 3).

Run: uv run --no-sync python bench/regression/g4/accuracy.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np

import stardust as sd

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.regression import refdata_robust  # noqa: E402


def a(x):
    return np.asarray(x.tolist(), dtype=float)


def dd(v):
    return np.array([p[0] for p in v]) + np.array([p[1] for p in v])


def arb():
    print("## against Arb")
    print("| problem | params | scale | bse | weights | n_iter |")
    print("|---|---:|---:|---:|---:|---:|")
    worst = np.zeros(4)
    for name, p in refdata_robust.load().items():
        r = refdata_robust.fit(p)
        f = p["ref"]
        hi = np.array([v[0] for v in f["params"]])
        lo = np.array([v[1] for v in f["params"]])
        e = [
            np.linalg.norm((a(r.params) - hi) - lo) / np.linalg.norm(hi + lo),
            abs(r.scale - sum(f["scale"])) / sum(f["scale"]),
            np.max(np.abs(a(r.bse) - dd(f["bse"])) / dd(f["bse"])),
            np.max(np.abs(a(r.weights) - dd(f["weights"]))),
        ]
        worst = np.maximum(worst, e)
        print(f"| {name} | {e[0]:.1e} | {e[1]:.1e} | {e[2]:.1e} | {e[3]:.1e} | {r.n_iter} |")
    print(f"| largest | {worst[0]:.1e} | {worst[1]:.1e} | {worst[2]:.1e} | {worst[3]:.1e} | |")


def statsmodels():
    import statsmodels.api as sm
    from statsmodels.robust import norms, scale

    print("\n## against statsmodels (tight tolerance), largest over three problems")
    print("| norm | scale_est | params | scale | bse | weights (abs) | sd n_iter | sm iterations |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    probs = []
    for seed, n in ((0, 150), (1, 157), (2, 400)):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3)) * rng.uniform(0.5, 2.0, 3) + 1.0
        y = X @ np.linspace(1.0, -1.0, 3) + 0.5 + rng.standard_normal(n)
        y[: n // 10] += 10.0
        probs.append((y, X))
    for norm, M in (("huber", norms.HuberT()), ("bisquare", norms.TukeyBiweight()), ("hampel", norms.Hampel())):
        for se in ("mad", "huber"):
            w = np.zeros(4)
            its, smit = [], []
            for y, X in probs:
                r = sd.regression.rlm(y, X, norm=norm, scale_est=se)
                est = "mad" if se == "mad" else scale.HuberScale(d=2.5, tol=1e-15, maxiter=1000)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = sm.RLM(y, sm.add_constant(X), M=M).fit(scale_est=est, tol=1e-15, maxiter=1000)
                p = a(r.params)
                w = np.maximum(
                    w,
                    [
                        np.max(np.abs(p - s.params)) / np.max(np.abs(s.params)),
                        abs(r.scale - s.scale) / s.scale,
                        np.max(np.abs(a(r.bse) - s.bse) / s.bse),
                        np.max(np.abs(a(r.weights) - s.weights)),
                    ],
                )
                its.append(r.n_iter)
                smit.append(s.fit_history["iteration"])
            print(
                f"| {norm} | {se} | {w[0]:.1e} | {w[1]:.1e} | {w[2]:.1e} | {w[3]:.1e} | "
                f"{'/'.join(map(str, its))} | {'/'.join(map(str, smit))} |"
            )


def r_refs():
    from statsmodels.robust import norms

    d = json.loads((ROOT / "tests" / "regression" / "ref" / "robust_r.json").read_text())
    print(f"\n## against R ({d['_versions']})")
    print("| problem | fit | coef | scale | se | note |")
    print("|---|---|---:|---:|---:|---|")
    nm = {"huber": norms.HuberT(), "bisquare": norms.TukeyBiweight(), "hampel": norms.Hampel()}
    for name, pr in d.items():
        if name.startswith("_"):
            continue
        Rv = pr["R"]
        y, X = np.array(pr["y"]), np.array(pr["X"])
        for norm in ("huber", "bisquare", "hampel"):
            for se in ("huber", "mad"):
                t = f"mass_{norm}_{se}"
                r = sd.regression.rlm(y, X, norm=norm, scale_est=se)
                z = a(r.resid) / r.scale
                dp = nm[norm].psi_deriv(z)
                nn, p, m = len(z), r.rank, dp.mean()
                kf = (1 + p / nn * dp.var(ddof=1) / m**2) / (1 + p / nn * dp.var() / m**2)
                c = np.max(np.abs(a(r.params) - Rv[t + "_coef"])) / np.max(np.abs(Rv[t + "_coef"]))
                s = abs(r.scale - Rv[t + "_scale"][0]) / Rv[t + "_scale"][0]
                e = np.max(np.abs(a(r.bse) * kf - Rv[t + "_se"]) / np.asarray(Rv[t + "_se"]))
                note = "proposal 2, k2 = 2.5" if se == "huber" else "MAD / 0.6745"
                print(f"| {name} | MASS {norm} | {c:.1e} | {s:.1e} | {e:.1e} | {note}; se with var(psi') over n - 1 |")
        r = sd.regression.rlm(y, X, method="MM", rng=sd.random.default_rng(pr["seed"]))
        c = np.max(np.abs(a(r.params) - Rv["mstep_coef"])) / np.max(np.abs(Rv["mstep_coef"]))
        e = np.max(np.abs(a(r.bse) - Rv["mstep_se"]) / np.asarray(Rv["mstep_se"]))
        print(f"| {name} | lmrob M step from sd's S | {c:.1e} | 0 | {e:.1e} | rel.tol 1e-15 |")
        c = np.max(np.abs(a(r.params) - Rv["lmrob_coef"])) / np.max(np.abs(Rv["lmrob_coef"]))
        s = abs(r.scale - Rv["lmrob_scale"][0]) / Rv["lmrob_scale"][0]
        e = np.max(np.abs(a(r.bse) - Rv["lmrob_se"]) / np.asarray(Rv["lmrob_se"]))
        print(f"| {name} | lmrob (its own subsamples) | {c:.1e} | {s:.1e} | {e:.1e} | S refine.tol 1e-7 |")


if __name__ == "__main__":
    arb()
    statsmodels()
    r_refs()
