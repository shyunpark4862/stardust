"""The Arb references of sd.differentiate (docs/design/integrate.md §7.2), in tests/differentiate/ref/:

    .venv/bin/python tools/differentiate_ref.py                # both
    .venv/bin/python tools/differentiate_ref.py elementary     # only this one

  elementary  derivatives.json: per function of tests/differentiate/problems.py ELEMENTARY and
              per point (its nine points and the EDGES' points), f'(x) and f''(x) from their closed
              forms in Arb at the double x, as [hi, lo] pairs: {"<name>/<x!r>": {"df", "ddf"}}
  mgh         mgh.npz (layout: tests/reffile.py): per More-Garbow-Hillstrom case with factor 1
              (problems.mgh_cases()), at x0, the Jacobian of the residuals F (from the problems'
              analytic Jacobians evaluated in Arb, as tools/optimize_ref.py's derivatives.npz) and,
              for n <= 10, the Hessian of f = sum F_i^2 (second-order forward-mode jets over Arb,
              tests/differentiate/jets.py), and the gradient 2 J^T F of f (jacobian's scalar f) in Arb

Every value is settled to 2^-110 relative (tools/arbref.py; a vector or matrix to 2^-110 of its
largest entry, as the measurements are normwise). The JSON file carries "_provenance"
(tests/differentiate/refdata.py leaves it out). An unknown name is a usage error (exit status 2);
the console output also goes to build/logs/differentiate_ref/. No mpmath.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import arbref  # noqa: E402
from arbref import MAX_PREC, START_PREC, NotSettled, precision, settled, to_dd  # noqa: E402

OUT = ROOT / "tests" / "differentiate" / "ref"
TOOL = "tools/differentiate_ref.py"


def point_key(name, x):
    return f"{name}/{float(x)!r}"


def elementary():
    from flint import arb

    from tests.differentiate import problems as P

    pts = {f.name: list(f.points) for f in P.ELEMENTARY}
    for e in P.EDGES:
        if e.x not in pts[e.function]:
            pts[e.function].append(e.x)
    rows = []
    for f in P.ELEMENTARY:
        for x in pts[f.name]:
            d1 = arbref.rigorous(lambda f=f, x=x: f.d1(arb(x)), floor=arbref.DOUBLE_TINY)
            d2 = arbref.rigorous(lambda f=f, x=x: f.d2(arb(x)), floor=arbref.DOUBLE_TINY)
            rows.append((point_key(f.name, x), {"df": list(to_dd(d1)), "ddf": list(to_dd(d2))}))
    prov = {**arbref.provenance(), "tool": TOOL, "n_requested": len(rows), "dropped": []}
    lines = [f'"_provenance": {json.dumps(prov)}'] + [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in rows]
    (OUT / "derivatives.json").write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"{len(rows)} points -> {(OUT / 'derivatives.json').relative_to(ROOT)}")


def _settled_vec(fn):
    """fn() -> list of arb, settled to 2^-TARGET_BITS of the largest entry: (hi, lo) arrays."""
    p = START_PREC
    while p <= MAX_PREC:
        with precision(p):
            vals = fn()
            scale = max((abs(v.mid()) for v in vals), default=0)
            floor = float(scale) if scale != 0 else arbref.DOUBLE_TINY
            if all(settled(v, floor=floor) for v in vals):
                pairs = [to_dd(v) for v in vals]
                return np.array([h for h, _ in pairs]), np.array([lo for _, lo in pairs])
        p *= 2
    raise NotSettled(fn)


def mgh():
    from flint import arb

    from tests import reffile
    from tests.differentiate import jets
    from tests.differentiate import problems as P
    from tests.optimize import ad

    cases = P.mgh_cases()
    x0s, js, gs, hs, hidx = [], [], [], [], []
    for i, c in enumerate(cases):
        x0 = [float(v) for v in c.x0()]

        def jac(c=c, x0=x0):
            x = [arb(v) for v in x0]
            return [arb(e) for row in c.jac(x, M=ad.ARB) for e in row]

        jh, jl = _settled_vec(jac)
        js.append((jh.reshape(c.m, c.n), jl.reshape(c.m, c.n)))

        def grad(c=c, x0=x0):  # of f = sum F_i^2: 2 J^T F (jacobian's scalar f, its gradient)
            x = [arb(v) for v in x0]
            J, F = c.jac(x, M=ad.ARB), c.fun(x, M=ad.ARB)
            return [2 * sum((arb(J[k][j]) * arb(F[k]) for k in range(c.m)), arb(0)) for j in range(c.n)]

        gs.append(_settled_vec(grad))
        x0s.append(np.array(x0))
        if c.n <= P.HESSIAN_MAX_N:

            def hess(c=c, x0=x0):
                H = jets.hessian_sum_squares(lambda x, M: c.fun(x, M=M), x0)
                return [H[r, s] for r in range(c.n) for s in range(c.n)]

            hh, hl = _settled_vec(hess)
            hs.append((hh.reshape(c.n, c.n), hl.reshape(c.n, c.n)))
            hidx.append(i)
        print(f"  {c.id}: n = {c.n}, m = {c.m}" + (", Hessian" if c.n <= P.HESSIAN_MAX_N else ""))
    reffile.write(
        OUT / "mgh.npz",
        **arbref.provenance(),
        tool=TOOL,
        n_requested=len(cases),
        inputs={"id": np.array([c.id for c in cases]), "x0": x0s},
        methods={
            "jacobian": {"over": "point", "index": np.arange(len(cases)), "hi": [j[0] for j in js],
                         "lo": [j[1] for j in js]},
            "hessian": {"over": "point", "index": np.array(hidx, dtype=np.int64), "hi": [h[0] for h in hs],
                        "lo": [h[1] for h in hs]},
            "gradient": {"over": "point", "index": np.arange(len(cases)), "hi": [g[0] for g in gs],
                         "lo": [g[1] for g in gs]},
        },
    )  # fmt: skip
    print(f"{len(cases)} cases ({len(hidx)} with the Hessian) -> {(OUT / 'mgh.npz').relative_to(ROOT)}")


PARTS = {"elementary": elementary, "mgh": mgh}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="tools/differentiate_ref.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="NAME", help=", ".join(PARTS))
    a = ap.parse_args(argv)
    bad = [n for n in a.names if n not in PARTS]
    if bad:
        ap.error(f"unknown reference {bad[0]!r}")
    OUT.mkdir(parents=True, exist_ok=True)
    with arbref.logged("differentiate_ref"):
        for n in a.names or PARTS:
            PARTS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
