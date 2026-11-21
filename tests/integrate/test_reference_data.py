"""The fixed reference data of sd.integrate (docs/design/integrate.md §7.3, §7.6): scipy 1.18.1 on
the collection (tools/gen_integrate_scipy.py) and the QUADPACK Fortran originals
(tools/gen_integrate_fortran.py) are complete and consistent with each other, with the problems
and with the Arb references; a sample of scipy's runs reproduces to the bit."""

import hashlib
import json
import math
import pathlib
import random
import shutil
import sys

import numpy as np
import pytest

from . import problems as P
from . import refdata

TOOLS = pathlib.Path(__file__).parents[2] / "tools"
QUADPACK_SRC = pathlib.Path(__file__).parents[2] / "build" / "comparators" / "quadpack"
SCIPY_VERSION = "1.18.1"

FILES = ("scipy_quad.json", "scipy_tanhsinh.json", "scipy_nsum.json", "scipy_multiple.json", "scipy_vector.json",
         "scipy_samples.json")  # fmt: skip


def _gen(name):
    sys.path.insert(0, str(TOOLS))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(TOOLS))


def live_scipy():
    import scipy

    if scipy.__version__ != SCIPY_VERSION:  # another scipy is a failure to look at, never a skip
        pytest.fail(f"the records are scipy {SCIPY_VERSION}'s, not {scipy.__version__}")


# ---- scipy


def test_scipy_records_cover_the_collection():
    d = {n: refdata.data(n) for n in FILES}
    for v in d.values():
        assert v.pop("scipy") == SCIPY_VERSION
    assert set(d["scipy_quad.json"]) == set(P.BY_ID)
    assert set(d["scipy_tanhsinh.json"]) == {p.id for p in P.INTEGRALS if p.tanhsinh} | {
        f"batch/{b.family}" for b in P.BATCHES
    }
    assert set(d["scipy_nsum.json"]) == set(P.SERIES_BY_ID)
    assert set(d["scipy_multiple.json"]) == {m.id for m in P.MULTI}
    assert set(d["scipy_vector.json"]) == {v.id for v in P.VECTOR}
    assert set(d["scipy_samples.json"]) == {"samples", "fixed_quad", "newton_cotes"}


def test_quad_call_orders_are_complete():
    q = refdata.data("scipy_quad.json")
    calls = np.load(refdata.DATA / "scipy_calls.npz")
    assert set(calls.files) == {f"quad/{p.id}" for p in P.INTEGRALS}
    for p in P.INTEGRALS:
        pts, rec = calls[f"quad/{p.id}"], q[p.id]
        assert pts.dtype == np.float64 and pts.size == rec["calls"] == rec["nfev"], p.id
        lo, hi = min(p.a, p.b), max(p.a, p.b)
        if p.weight in ("cos", "sin") and math.isinf(lo):
            continue  # QAWF on (-inf, b]: the points are -x of [-b, inf)'s
        if math.isfinite(lo) and math.isfinite(hi) and p.weight is None:
            assert np.all((pts > lo) & (pts < hi)), p.id  # Gauss-Kronrod: inside the range
        else:
            assert np.all((pts >= lo) & (pts <= hi)), p.id  # Clenshaw-Curtis rules touch the ends


def test_quad_status_and_messages():
    q = refdata.data("scipy_quad.json")
    q.pop("scipy")
    for pid, r in q.items():
        assert r["status"] in (0, 1, 2, 3, 4, 5), pid
        assert (r["message"] is None) == (r["status"] == 0), pid
    assert q["book/osc/9"]["status"] == 1 and q["book/osc/9"]["message"].startswith(
        "The maximum number of subdivisions (50) has been achieved."
    )
    assert {pid for pid, r in q.items() if r["status"] != 0} == (
        {f"book/peak/{k}" for k in range(10, 19)} | {"book/osc/9", "book/osc/10", "scipy/tanhsinh/f15"}
    )


# The converged results whose error estimate is below the true error (docs/design/integrate.md §4):
# quad misses the narrow peak with its first 21 points (a = 19, 20: about 0 for pi), and on f10
# its extrapolation reaches the integral to the exact pi/2, 1.6e-8 past the double limit's;
# tanhsinh's estimate is short on the moderate peaks and fast oscillations (and on f9 by the
# double limit's sliver, 2e-15).
QUAD_SHORT = {"book/peak/19", "book/peak/20", "scipy/tanhsinh/f10"}
TANHSINH_SHORT = {"book/peak/0", "book/peak/2", "book/peak/4", "book/osc/6", "book/osc/7", "book/osc/9", "book/osc/10",
                  "scipy/tanhsinh/f9"}  # fmt: skip


def test_scipy_against_the_references():
    ref = refdata.integrals()
    q = refdata.data("scipy_quad.json")
    q.pop("scipy")
    short = {
        pid for pid, r in q.items() if r["status"] == 0 and abs(r["integral"] - refdata.value(ref[pid])) > r["error"]
    }
    assert short == QUAD_SHORT
    t = refdata.data("scipy_tanhsinh.json")
    t.pop("scipy")
    short = set()
    for pid, r in t.items():
        if pid.startswith("batch/"):
            for i, iid in enumerate(r["ids"]):
                # a batch's element is the scalar problem's run, field by field
                one = t.get(iid)
                assert one is not None and all(one[k] == r[k][i] for k in ("integral", "error", "status", "nfev")), iid
            continue
        if r["status"] == 0 and abs(r["integral"] - refdata.value(ref[pid])) > r["error"]:
            short.add(pid)
    assert short == TANHSINH_SHORT
    s, n = refdata.series(), refdata.data("scipy_nsum.json")
    for sid, r in n.items():
        if sid != "scipy":
            assert r["status"] == 0 and abs(r["sum"] - refdata.value(s[sid])) <= r["error"], sid
    m, md = refdata.multiple(), refdata.data("scipy_multiple.json")
    for mid, r in md.items():
        if mid == "scipy":
            continue
        for kind, rr in r.items():
            assert (
                abs(rr["integral"] - refdata.value(m[mid])) <= max(rr["error"], 1e-14 * abs(refdata.value(m[mid]))) * 4
            ), (mid, kind)


def test_call_shapes_of_the_elementwise_family():
    # docs/design/integrate.md §7.4: tanhsinh calls () then (1, 66) for a scalar, a batch's
    # remaining elements level by level; nsum starts with (), (1,), (1, 66)
    t = refdata.data("scipy_tanhsinh.json")
    assert t["scipy/tanhsinh/f1"]["calls"]["shapes"][:2] == [[[], 1], [[1, 66], 1]]
    xb = t["batch/book/xlog"]["calls"]
    assert xb["shapes"][0] == [[30], 1] and xb["shapes"][1] == [[30, 66], 1]
    n = refdata.data("scipy_nsum.json")["series/zeta/2"]["calls"]
    assert n["calls"] == 10 and n["shapes"][:3] == [[[], 1], [[1], 1], [[1, 66], 1]]


def test_scipy_reproduces_its_records():
    live_scipy()
    gen = _gen("gen_integrate_scipy")
    q = refdata.data("scipy_quad.json")
    calls = np.load(refdata.DATA / "scipy_calls.npz")
    for pid in random.Random(20261121).sample(sorted(P.BY_ID), 8):
        res, pts = gen.quad_run(P.get(pid))
        assert json.dumps(res, sort_keys=True) == json.dumps(q[pid], sort_keys=True), pid
        assert np.array_equal(pts.view(np.int64), calls[f"quad/{pid}"].view(np.int64)), pid
    t = refdata.data("scipy_tanhsinh.json")
    for pid in ("scipy/tanhsinh/f1", "book/xlog/-0.9", "hard/gauss"):
        assert json.dumps(gen._jsonable(gen.tanhsinh_run(P.get(pid))), sort_keys=True) == json.dumps(
            t[pid], sort_keys=True
        )
    n = refdata.data("scipy_nsum.json")
    got = gen._jsonable(gen.nsum_run(P.SERIES_BY_ID["series/zeta/2"]))
    assert json.dumps(got, sort_keys=True) == json.dumps(n["series/zeta/2"], sort_keys=True)


# ---- the Fortran originals


def test_fortran_records_cover_the_collection():
    f = refdata.data("fortran_quadpack.json")
    assert f["flags"] == "-O2 -ffp-contract=off -std=legacy -w -shared -fPIC"
    assert sorted(f["sha256"]) == sorted(f"{s}.f" for s in _gen("gen_integrate_fortran").SOURCES)
    assert set(f["replacements"]) == {"d1mach", "xerror"}  # dgtsl is LINPACK's own file (C18)
    st = f["dgtsl_standin"]  # the earlier stand-in against the original: the same to the bit
    assert st["runs"] == len(P.BY_ID) and st["differ"] == [] and st["runs_calling_dgtsl"] == 5
    runs = {k: v for k, v in f.items() if isinstance(v, dict) and "routine" in v}
    assert {k for k in runs if not k.startswith("dqng/")} == set(P.BY_ID)
    routine = {"dqagse": 0, "dqagie": 0, "dqagpe": 0, "dqawoe": 0, "dqawfe": 0, "dqawse": 0, "dqawce": 0}
    for k, v in runs.items():
        if not k.startswith("dqng/"):
            routine[v["routine"]] += 1
            assert v["calls"] == v["nfev"], k
    assert all(routine.values()), routine  # every routine scipy's quad calls is exercised
    # dgtsl is called by QAWO and QAWF only
    assert {v["routine"] for v in runs.values() if v.get("dgtsl_calls")} == {"dqawoe", "dqawfe"}


# The Fortran originals against scipy's C port (docs/design/integrate.md §7.6): the same points in
# the same order but for QAWF's first cycle (its QK15W points differ in the last bit) and one QAGS
# run whose stopping test falls the other way; the integral and error differ in the last bits in
# most runs (scipy's accumulations are contracted to FMA; tools/gen_integrate_fortran.py
# --probe-contract) and QAWS's log weight example by 4e-9 with the same points.
CALLS_DIFFER = {
    "book/xlog/+1.0",
    "quadpack/qawf",
    "scipy/quad/sin-weighted-infinite",
    "scipy/quad/cos-weighted-infinite",
}


def test_fortran_against_scipy():
    c = refdata.data("fortran_vs_scipy.json")
    assert c.pop("scipy") == SCIPY_VERSION
    same, of = c.pop("same"), c.pop("of")
    assert of == len(P.BY_ID) and set(c) == set(P.BY_ID)
    assert same == sum(1 for v in c.values() if v["same_result"] and v["same_calls"])
    assert {k for k, v in c.items() if not v["same_calls"]} == CALLS_DIFFER
    assert {k for k, v in c.items() if not (v["same"]["nfev"] and v["same"]["status"])} == {"book/xlog/+1.0"}
    f = refdata.data("fortran_quadpack.json")
    ref = refdata.integrals()
    for pid in P.BY_ID:
        if f[pid]["status"] == 0 and pid not in QUAD_SHORT:
            assert abs(f[pid]["integral"] - refdata.value(ref[pid])) <= f[pid]["error"], pid


@pytest.mark.external
def test_the_fortran_build_reproduces_its_records(tmp_path):
    if not QUADPACK_SRC.is_dir() or shutil.which("/opt/homebrew/bin/gfortran") is None:
        pytest.skip("the QUADPACK sources (build/comparators/quadpack) or gfortran are not here")
    live_scipy()
    gen = _gen("gen_integrate_fortran")
    lib, _ = gen.build(QUADPACK_SRC.resolve(), tmp_path)
    qp = gen.Quadpack(lib)
    f = refdata.data("fortran_quadpack.json")
    for pid in random.Random(1121).sample(sorted(P.BY_ID), 12):
        rec, _ = gen.run_routine(qp, P.get(pid))
        want = {k: v for k, v in f[pid].items() if k != "dgtsl_calls"}  # counted by the stand-in build only
        got = {k: v for k, v in rec.items() if k != "dgtsl_calls"}
        assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), pid
    assert hashlib.sha256(gen.DGTSL_STANDIN.encode()).hexdigest() == f["dgtsl_standin"]["sha256"]
