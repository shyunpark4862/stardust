"""The fixed reference data of sd.optimize (docs/design/optimize.md §7): scipy 1.18.1 on the
collection (tools/gen_optimize_scipy.py), the Fortran originals of MINPACK and L-BFGS-B 3.0
(tools/gen_optimize_fortran.py) and the Arb roots (tools/optimize_ref.py) are complete and
consistent with each other and with the problems' definitions."""

import json
import math
import pathlib
import random
import re
import sys

import numpy as np
import pytest

from . import functions, mgh, refdata, systems

DATA = pathlib.Path(__file__).parent / "data"
TOOLS = pathlib.Path(__file__).parents[2] / "tools"


def load(name):
    return json.loads((DATA / name).read_text())


# ---- scipy

SCIPY = {
    name: load(name)
    for name in (
        "scipy_least_squares.json",
        "scipy_minimize.json",
        "scipy_root.json",
        "scipy_root_scalar.json",
        "scipy_minimize_scalar.json",
        "scipy_elementwise.json",
        "scipy_lsq_linear.json",
        "scipy_approx_fprime.json",
    )
}


def test_scipy_data_cover_the_collection():
    for d in SCIPY.values():
        assert d["scipy"] == "1.18.1"
    ls, mn = SCIPY["scipy_least_squares.json"], SCIPY["scipy_minimize.json"]
    for c in mgh.CASES:
        assert set(ls[c.id]) == {"trf/analytic", "trf/2-point", "lm/analytic", "lm/2-point"}
        assert set(mn[c.id]) == {
            "Nelder-Mead",
            "BFGS",
            "BFGS/fd",
            "L-BFGS-B",
            "L-BFGS-B/fd",
            "Newton-CG",
            "trust-ncg",
            "trust-exact",
        }
    rt = SCIPY["scipy_root.json"]
    for c in systems.CASES:
        assert set(rt[c.id]) == {"hybr/analytic", "hybr/none", "lm/analytic", "lm/none"}
    rs = SCIPY["scipy_root_scalar.json"]
    for c in functions.ROOT_CASES:
        assert {"brentq", "toms748", "bisect"} <= set(rs[c.id])
    for c in functions.MIN_CASES:
        assert set(SCIPY["scipy_minimize_scalar.json"][c.id]) == {"brent", "golden", "bounded"}
    assert set(SCIPY["scipy_lsq_linear.json"]) - {"scipy"} == set(functions.LSQ_LINEAR)


def test_scipy_runs_reach_the_paper_optima():
    # the definitions of the problems MINPACK's drivers leave out are checked by the optima
    ls, mn = SCIPY["scipy_least_squares.json"], SCIPY["scipy_minimize.json"]
    found = {}
    for c in mgh.CASES:
        vals = [2.0 * r["cost"] for r in ls[c.id].values() if "cost" in r]
        vals += [r["fun"] for r in mn[c.id].values() if r.get("fun") is not None]
        found.setdefault((c.number, c.n, c.m), []).extend(vals)
    for key, known in mgh.KNOWN.items():
        for k in known:
            ok = [v for v in found[key] if (abs(v - k) <= 2e-5 * k if k else v <= 1e-8)]
            assert ok, (key, k)


def test_chandrupatla_counts():
    # scipy's elementwise functions with the papers' tolerances reproduce the papers' counts,
    # which checks the transcription of the 45 + 55 cases
    ew = SCIPY["scipy_elementwise.json"]
    for c in functions.CHANDRUPATLA:
        assert ew[c.id]["find_root/paper"]["nfev"] == c.nfev_paper, c.id
    for c in functions.MIN_CASES:
        if c.nit_paper is not None:
            assert ew[c.id]["find_minimum/paper"]["nit"] == c.nit_paper, c.id


def test_bracketing_roots_within_tolerance_of_arb():
    ref = refdata.roots()
    rs = SCIPY["scipy_root_scalar.json"]
    assert set(ref) == set(functions.ROOTS_BY_ID)
    for c in functions.ROOT_CASES:
        r = ref[c.id]
        lo, hi = (r["x"], r["x"]) if r["kind"] == "exact" else (r["a"], r["b"])
        assert lo <= hi and (r["kind"] == "exact" or math.nextafter(lo, math.inf) == hi), c.id
        for m in ("brentq", "toms748", "bisect"):
            s = rs[c.id][m]
            if not s["converged"]:
                continue
            x = s["root"]
            d = 0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))
            with np.errstate(all="ignore"):
                zero_in_double = float(c.f(np.float64(x))) == 0.0  # e.g. an underflow to 0
            assert d <= 2e-12 + 4 * 2.220446049250313e-16 * abs(x) or zero_in_double, (c.id, m, x, lo)


def test_arb_root_references_spot_check():
    sys.path.insert(0, str(TOOLS))
    try:
        import optimize_ref
    finally:
        sys.path.remove(str(TOOLS))
    ref = refdata.roots()
    for cid in ("aps.01.00", "aps.06.09", "aps.12.18", "chandrupatla.1.5", "chandrupatla.8.3", "original.03"):
        c = functions.ROOTS_BY_ID[cid]
        r = ref[cid]
        if r["kind"] == "exact":
            assert optimize_ref.sign(c.f, r["x"]) == 0
        else:
            assert optimize_ref.sign(c.f, r["a"]) * optimize_ref.sign(c.f, r["b"]) == -1


def test_scipy_reproduces_its_reference():
    # a sample of runs again: the installed scipy gives the recorded values exactly
    import scipy

    if scipy.__version__ != "1.18.1":
        pytest.skip("the reference is scipy 1.18.1's")
    sys.path.insert(0, str(TOOLS))
    try:
        import gen_optimize_scipy as gen
    finally:
        sys.path.remove(str(TOOLS))
    rnd = random.Random(20261120)
    jobs = [
        ("scipy_least_squares.json", gen._lsq_task, gen.least_squares_tasks()),
        ("scipy_minimize.json", gen._min_task, gen.minimize_tasks()),
        ("scipy_root.json", gen._root_task, gen.root_tasks()),
        ("scipy_root_scalar.json", gen._scalar_root_task, gen.root_scalar_tasks()),
        ("scipy_elementwise.json", gen._elementwise_task, gen.elementwise_tasks()),
    ]
    for name, fn, tasks in jobs:
        for t in rnd.sample(tasks, 6):
            (cid, method), res = fn(t)
            # as text: NaN fields compare equal
            assert json.dumps(res, sort_keys=True) == json.dumps(SCIPY[name][cid][method], sort_keys=True), (
                name,
                cid,
                method,
            )


# ---- the Fortran originals


def _summary_rows(text, driver):
    block = text.split(f"== {driver}")[1].split("== ")[0]
    rows = []
    for ln in block.splitlines():
        v = ln.split()
        if len(v) >= 5 and v[0].isdigit():
            rows.append(v)
    return rows


def test_original_drivers_match_the_full_precision_runs():
    text = (DATA / "minpack_drivers.txt").read_text()
    fm = load("fortran_minpack.json")
    for drv, kind in (("file17", "lmder"), ("file19", "lmdif"), ("file16", "hybrj"), ("file15", "hybrd")):
        rows = _summary_rows(text, drv)
        runs = fm[kind]
        assert len(rows) == len(runs) == (53 if kind.startswith("lm") else 55)
        for v, r in zip(rows, runs):
            if kind.startswith("lm"):  # NPROB N M NFEV NJEV INFO FNORM
                nprob, n, m, nfev, njev, info, fnorm = (
                    int(v[0]),
                    int(v[1]),
                    int(v[2]),
                    int(v[3]),
                    int(v[4]),
                    int(v[5]),
                    float(v[6].replace("D", "E")),
                )
                assert (nprob, n, m) == (r["nprob"], r["n"], r["m"])
                assert nfev == r["calls_f"]
                assert njev == (r["calls_j"] if kind == "lmder" else r["calls_d"] // n)
            else:  # NPROB N NFEV INFO FNORM (HYBRJ: NPROB N NFEV NJEV INFO FNORM)
                if kind == "hybrj":
                    nprob, n, nfev, njev, info, fnorm = (
                        int(v[0]),
                        int(v[1]),
                        int(v[2]),
                        int(v[3]),
                        int(v[4]),
                        float(v[5].replace("D", "E")),
                    )
                    assert njev == r["calls_j"]
                    assert nfev == r["calls_f"]
                else:
                    nprob, n, nfev, info, fnorm = (
                        int(v[0]),
                        int(v[1]),
                        int(v[2]),
                        int(v[3]),
                        float(v[4].replace("D", "E")),
                    )
                    assert nfev == r["calls_f"] + r["calls_d"]
                assert (nprob, n) == (r["nprob"], r["n"])
            assert info == r["info"]
            assert math.isclose(fnorm, r["fnorm"], rel_tol=1e-6, abs_tol=1e-300) or fnorm == 0.0


def test_fortran_results_satisfy_the_python_problems():
    fm = load("fortran_minpack.json")
    with np.errstate(all="ignore"):
        for kind in ("lmder", "lmdif"):
            for r in fm[kind]:
                c = mgh.BY_ID[r["id"]]
                f = np.array(c.fun([np.float64(v) for v in r["x"]]))
                assert math.isclose(float(np.sqrt(np.sum(f * f))), r["fnorm"], rel_tol=1e-14, abs_tol=1e-300), r["id"]
        for kind in ("hybrj", "hybrd"):
            for r in fm[kind]:
                c = systems.BY_ID[r["id"]]
                f = np.array(c.fun([np.float64(v) for v in r["x"]]))
                assert math.isclose(float(np.sqrt(np.sum(f * f))), r["fnorm"], rel_tol=1e-14, abs_tol=1e-300), r["id"]


def test_lbfgsb_driver1():
    # L-BFGS-B 3.0's driver1 (unchanged) and this tool's driver loop on the same problem
    text = (DATA / "lbfgsb_drivers.txt").read_text()
    d1 = text.split("== driver1.f")[1].split("== ")[0]
    m = re.search(r"\n\s+25\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\S+\s+\S+\s*\n\s*F =\s+(\S+)", d1)
    tit, tnf, f = int(m.group(1)), int(m.group(2)), float(m.group(3).replace("D", "E"))
    lb = load("fortran_lbfgsb.json")["driver1-m5"]
    assert (lb["n_iter"], lb["nfev"]) == (tit, tnf) == (23, 28)
    assert lb["fun"] == f == 1.0834900834300615e-09
    cmp_ = {r["id"]: r for r in load("fortran_vs_scipy.json")["lbfgsb"]}["driver1-m5"]
    assert (cmp_["scipy"]["n_iter"], cmp_["scipy"]["nfev"]) == (23, 28)
    assert math.isclose(cmp_["scipy"]["fun"], f, rel_tol=1e-9)


def test_fortran_vs_scipy_records():
    d = load("fortran_vs_scipy.json")
    assert {k: len(v) for k, v in d.items()} == {"lmder": 53, "lmdif": 53, "hybrj": 55, "hybrd": 55, "lbfgsb": 83}
    for kind, rows in d.items():
        for r in rows:
            if r["same_trace"]:
                # the same points in the same order end at the same point
                assert r["x_rel_difference"] == 0.0, (kind, r["id"])
                assert r["first_difference"] is None
            else:
                assert r["first_difference"] >= 1  # both start at the same x0
            if kind != "lbfgsb":
                # scipy's port evaluates fun at x0 once more than the Fortran before MINPACK starts
                assert r["scipy"]["extra_calls_at_x0"] == 1
    # where the problem has a unique, well-conditioned solution, the two end within 1e-10 of
    # each other however their rounding differs on the way
    good = [r for r in d["lmder"] if r["id"].startswith(("mgh17", "mgh19", "mgh08-n3-m15-x1", "mgh10"))]
    good = [r for r in good if r["fortran"]["info"] == r["scipy"]["info"] and 1 <= r["fortran"]["info"] <= 4]
    assert len(good) >= 5 and all(r["x_rel_difference"] < 1e-10 for r in good)
