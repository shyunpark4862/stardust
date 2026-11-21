"""scipy 1.18.1's records on the differentiate collection (tools/gen_differentiate_scipy.py,
docs/design/integrate.md §7.3): complete, consistent with the Arb references, with the call
shapes of docs/design/integrate.md §7.4, and a sample reproduces to the bit."""

import json
import pathlib
import random
import sys

import numpy as np
import pytest

from . import problems as P
from . import refdata

TOOLS = pathlib.Path(__file__).parents[2] / "tools"
SCIPY_VERSION = "1.18.1"


def _gen():
    sys.path.insert(0, str(TOOLS))
    try:
        import gen_differentiate_scipy
    finally:
        sys.path.remove(str(TOOLS))
    return gen_differentiate_scipy


def test_records_cover_the_collection():
    d = refdata.data("scipy_derivative.json")
    assert d.pop("scipy") == SCIPY_VERSION
    want = {f"{f.name}/{x!r}" for f in P.ELEMENTARY for x in f.points} | {f"batch/{f.name}" for f in P.ELEMENTARY}
    want |= {f"edge/{e.id}" for e in P.EDGES}
    want |= {f"nd/{f.name}" for f in P.ELEMENTARY} | {a[0] for a in P.ARGS}
    assert set(d) == want
    j = refdata.data("scipy_jacobian.json")
    h = refdata.data("scipy_hessian.json")
    assert j.pop("scipy") == h.pop("scipy") == SCIPY_VERSION
    assert set(j) == {c.id for c in P.mgh_cases()}
    assert set(h) == {c.id for c in P.mgh_cases() if c.n <= P.HESSIAN_MAX_N}


def test_call_shapes():
    # docs/design/integrate.md §7.4: (), (1, 8), (1, 2) for exp at 1; a batch the same per element
    d = refdata.data("scipy_derivative.json")
    assert d["exp/1.0"]["calls"]["shapes"] == [[[], 1], [[1, 8], 1], [[1, 2], 1]]
    assert d["exp/1.0"]["nfev"] == 11 and d["exp/1.0"]["n_iter"] == 2
    assert d["batch/exp"]["calls"]["shapes"] == [[[9], 1], [[9, 8], 1], [[9, 2], 1]]
    assert d["edge/log/1e-3/default"]["status"] == -3 and d["edge/sqrt/1e-6/default"]["status"] == -3
    assert d["edge/cube/0/default"]["status"] == -2 and d["edge/cube/0/default"]["nfev"] == 27


def test_against_the_references():
    ref = refdata.derivatives()
    d = refdata.data("scipy_derivative.json")
    worst, differ = 0.0, 0
    for f in P.ELEMENTARY:
        b = d[f"batch/{f.name}"]
        for i, x in enumerate(f.points):
            one = d[f"{f.name}/{x!r}"]
            # the same steps and calls alone or in a batch; df is fc @ wc (BLAS dgemv), whose sum
            # order depends on the batch's size, so 29 of the 99 differ in the last bits (by less
            # than the error estimate: the stencil's cancellation magnifies them)
            assert [one[k] for k in ("status", "nfev", "n_iter")] == [b[k][i] for k in ("status", "nfev", "n_iter")]
            assert abs(one["df"] - b["df"][i]) <= max(one["error"], b["error"][i]), (f.name, x)
            differ += one["df"] != b["df"][i]
            want = ref[f"{f.name}/{x!r}"]["df"][0]
            if one["status"] == 0:
                worst = max(worst, abs(one["df"] - want) / max(abs(want), 1e-300) if want else abs(one["df"]))
    assert worst < 1e-9, worst
    assert differ == 29
    J = refdata.mgh()
    for cid, r in refdata.data("scipy_jacobian.json").items():
        if cid == "scipy":
            continue
        df, want = np.array(r["df"]), J[cid]["jacobian"][0]
        ok = np.array(r["status"]) == 0
        assert np.all(np.abs(df - want)[ok] <= 1e-6 * np.maximum(1.0, np.abs(want))[ok]), cid


def test_scipy_reproduces_its_records():
    import scipy

    if scipy.__version__ != SCIPY_VERSION:
        pytest.fail(f"the records are scipy {SCIPY_VERSION}'s, not {scipy.__version__}")
    gen = _gen()
    d = refdata.data("scipy_derivative.json")
    rnd = random.Random(20261121)
    for f in rnd.sample(list(P.ELEMENTARY), 4):
        x = rnd.choice(f.points)
        got = gen.derivative_run(f, x, {})
        assert json.dumps(_tojson(got), sort_keys=True) == json.dumps(d[f"{f.name}/{x!r}"], sort_keys=True)
    j = refdata.data("scipy_jacobian.json")
    cid, rec = gen._jacobian_task(0)
    assert json.dumps(_tojson(rec), sort_keys=True) == json.dumps(j[cid], sort_keys=True)


def _tojson(v):
    sys.path.insert(0, str(TOOLS))
    try:
        from gen_integrate_scipy import _jsonable
    finally:
        sys.path.remove(str(TOOLS))
    return json.loads(json.dumps(_jsonable(v)))
