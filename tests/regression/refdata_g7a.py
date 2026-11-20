"""Arb references for the G7a tests (tools/regref_g7a.py; docs/design/regression.md §3.8, §6).

tests/regression/ref/g7a.json holds, per problem, the inputs as doubles and the references as
[hi, lo] pairs: linear models (OLS, WLS with zero weights, no intercept, a factor's dummies as
one term, an ill-conditioned design with kappa near 1e8) and GLMs (binomial logit and probit,
poisson log, gamma log). The GLMs' Newton starts are sd's fits (regref certifies the root).
Regenerate with (at most 10 processes)

    uv run --no-sync python -m tests.regression.refdata_g7a
"""

import json
import os
import pathlib
import sys

REF = pathlib.Path(__file__).parent / "ref" / "g7a.json"


def problems():
    import numpy as np

    out = {}
    rng = np.random.default_rng(71)
    n = 40
    X = np.round(rng.standard_normal((n, 3)), 6)
    y = np.round(1.0 + X @ [0.7, -0.4, 0.1] + rng.standard_normal(n) * (0.4 + np.abs(X[:, 0])), 6)
    R = [[0, 1, -1, 0], [0, 0, 1, 1], [0, 1, 1, 0]]
    out["ols"] = dict(
        kind="lin",
        y=y,
        X=X,
        weights=None,
        intercept=True,
        R=R,
        q=[0.3, 0.0, 0.1],
        small=1,
        terms=[[0], [1], [2]],
        gq=(None, 0, None),
    )
    w = np.round(rng.uniform(0.3, 3.0, n), 6)
    w[[5, 22]] = 0.0
    out["wls_zero"] = dict(
        kind="lin",
        y=y,
        X=X,
        weights=w,
        intercept=True,
        R=[[0, 1, 1, 1]],
        q=[0.0],
        small=2,
        terms=[[0, 1], [2]],
        gq=(15, 4, 1),
    )
    Xn = np.round(rng.standard_normal((n, 3)) + 1.0, 6)
    yn = np.round(Xn @ [0.5, 0.5, -0.3] + rng.standard_normal(n), 6)
    out["noint"] = dict(
        kind="lin",
        y=yn,
        X=Xn,
        weights=None,
        intercept=False,
        R=[[1, -1, 0], [0, 0, 1]],
        q=[0.0, 0.2],
        small=2,
        terms=[[0], [1], [2]],
        gq=(None, 6, 2),
    )
    # nearly collinear columns: kappa of the design near 1e8
    x1 = np.round(rng.standard_normal(50), 6)
    x2 = x1 + np.round(1e-7 * rng.standard_normal(50), 13)
    x3 = np.round(rng.standard_normal(50), 6)
    Xc = np.column_stack([x1, x2, x3])
    yc = np.round(1.0 + x1 + 0.5 * x3 + 0.3 * rng.standard_normal(50), 6)
    out["cond"] = dict(
        kind="lin",
        y=yc,
        X=Xc,
        weights=None,
        intercept=True,
        R=[[0, 1, -1, 0], [0, 1, 1, 0]],
        q=[0.0, 1.0],
        small=1,
        terms=[[0], [1], [2]],
        gq=(None, 0, 2),
    )
    nf = 45
    g = rng.integers(0, 3, nf)
    Xf = np.column_stack([np.round(rng.standard_normal((nf, 2)), 6), (g == 1).astype(float), (g == 2).astype(float)])
    yf = np.round(0.5 + Xf @ [1.0, 0.3, 0.6, -0.2] + rng.standard_normal(nf), 6)
    out["factor"] = dict(
        kind="lin",
        y=yf,
        X=Xf,
        weights=None,
        intercept=True,
        R=[[0, 0, 0, 1, -1], [0, 1, 0, 0, 0]],
        q=[0.0, 0.0],
        small=2,
        terms=[[0], [1], [2, 3]],
        gq=(None, 5, 0),
    )
    ng = 60
    Xg = np.round(rng.standard_normal((ng, 2)), 6)
    eta = 0.3 + Xg @ [0.9, -0.6]
    Rg = [[0, 1, 1], [0, 0, 1]]
    out["glm_logit"] = dict(
        kind="glm",
        y=(rng.uniform(size=ng) < 1 / (1 + np.exp(-eta))).astype(float),
        X=Xg,
        family="binomial",
        link="logit",
        R=Rg,
        small=1,
        terms=[[0], [1]],
    )
    out["glm_probit"] = dict(
        kind="glm",
        y=(rng.uniform(size=ng) < 1 / (1 + np.exp(-eta))).astype(float),
        X=Xg,
        family="binomial",
        link="probit",
        R=Rg,
        small=1,
        terms=[[0], [1]],
    )
    out["glm_poisson"] = dict(
        kind="glm",
        y=rng.poisson(np.exp(eta)).astype(float),
        X=Xg,
        family="poisson",
        link="log",
        R=Rg,
        small=1,
        terms=[[0], [1]],
    )
    out["glm_gamma"] = dict(
        kind="glm",
        y=np.round(rng.gamma(3.0, np.exp(eta) / 3.0), 6),
        X=Xg,
        family="gamma",
        link="log",
        R=Rg,
        small=1,
        terms=[[0], [1]],
    )
    return out


def _plain(v):
    import numpy as np

    return np.asarray(v).tolist() if hasattr(v, "shape") else v


def _one(item):
    name, p = item
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    import regref_g7a

    p = {k: _plain(v) for k, v in p.items()}
    if p["kind"] == "lin":
        ref, meta = regref_g7a.lin(
            p["y"],
            p["X"],
            weights=p["weights"],
            intercept=p["intercept"],
            R=p["R"],
            q=p["q"],
            small=p["small"],
            terms=p["terms"],
            gq=tuple(p["gq"]),
        )
    else:
        import stardust as sd

        def start(cols):
            return sd.regression.glm(
                p["y"], [[r[c] for c in cols] for r in p["X"]], family=p["family"], link=p["link"]
            ).params.tolist()

        k = len(p["X"][0])
        acc, seq = [], []
        for t in p["terms"][:-1]:
            acc = acc + t
            seq.append(start(acc))
        starts = {"big": start(list(range(k))), "small": start(list(range(p["small"]))), "seq": seq}
        ref, meta = regref_g7a.glm(
            p["y"],
            p["X"],
            family=p["family"],
            link=p["link"],
            R=p["R"],
            small=p["small"],
            terms=p["terms"],
            starts=starts,
        )
    vals = {key: ([list(x) for x in v] if isinstance(v, list) else list(v)) for key, v in ref.items()}
    return name, p | {"meta": meta, "ref": vals}


def generate():
    import multiprocessing as mp

    items = list(problems().items())
    with mp.get_context("spawn").Pool(min(10, len(items)), maxtasksperchild=1) as pool:
        data = dict(pool.map(_one, items, chunksize=1))
    REF.parent.mkdir(exist_ok=True)
    with open(REF, "w") as fh:
        json.dump(data, fh)


def load():
    with open(REF) as fh:
        return json.load(fh)


if __name__ == "__main__":
    generate()
    print(REF, os.path.getsize(REF), "bytes")
