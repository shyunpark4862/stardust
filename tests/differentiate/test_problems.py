"""The differentiate test collection (tests/differentiate/problems.py) and its Arb references
(tools/differentiate_ref.py): the closed-form derivatives are the derivatives of the functions,
every point has a reference, and the MGH references agree with optimize's Jacobians and with
forward-mode Hessians in double precision."""

import math

import numpy as np
import pytest

from . import problems as P
from . import refdata


def test_collection_shape():
    assert len(P.ELEMENTARY) == 11 and all(len(f.points) == 9 for f in P.ELEMENTARY)
    assert len({f.name for f in P.ELEMENTARY}) == 11
    for e in P.EDGES:
        assert e.function in P.BY_NAME, e.id
        assert set(e.options) <= {"atol", "rtol", "max_iter", "order", "initial_step", "step_factor", "step_direction"}
    cases = P.mgh_cases()
    assert len(cases) == 47 and sum(c.n <= P.HESSIAN_MAX_N for c in cases) == 42


@pytest.mark.parametrize("name", [f.name for f in P.ELEMENTARY])
def test_closed_forms_are_the_derivatives(name):
    # a central difference in Arb at h = 2^-70 (error h^2 f''' / 6) agrees with d1, and d1's with d2
    from flint import arb, ctx

    f = P.BY_NAME[name]
    old = ctx.prec
    ctx.prec = 300
    try:
        h = arb(2) ** -70
        for x in f.points:
            x = arb(x)
            num = (f.expr(x + h, P.ARB) - f.expr(x - h, P.ARB)) / (2 * h)
            assert abs(num - f.d1(x)) < arb(2) ** -100 * (1 + abs(f.d1(x))), (name, x)
            num2 = (f.d1(x + h) - f.d1(x - h)) / (2 * h)
            assert abs(num2 - f.d2(x)) < arb(2) ** -100 * (1 + abs(f.d2(x))), (name, x)
    finally:
        ctx.prec = old


def test_numpy_form_is_the_function():
    for f in P.ELEMENTARY:
        x = np.array(f.points)
        got = f.f_np(x)
        want = np.array([float(f.expr(__import__("flint").arb(v), P.ARB).mid()) for v in f.points])
        assert np.allclose(got, want, rtol=4e-16, atol=0), f.name


def test_every_point_has_a_reference():
    ref = refdata.derivatives(keep_provenance=True)
    prov = ref.pop("_provenance")
    assert prov["tool"] == "tools/differentiate_ref.py" and prov["target_bits"] == 110
    want = {f"{f.name}/{x!r}" for f in P.ELEMENTARY for x in f.points}
    want |= {f"{e.function}/{e.x!r}" for e in P.EDGES}
    assert set(ref) == want
    assert ref["exp/1.0"]["df"][0] == math.e and ref["log/0.001"]["df"][0] == 1000.0
    assert ref["cube/0.0"]["df"] == [0.0, 0.0]


def test_mgh_references():
    from tests.optimize import ad
    from tests.optimize import refdata as orefdata

    ref = refdata.mgh()
    assert set(ref) == {c.id for c in P.mgh_cases()}
    jac = {cid: J for cid, _, _, J in orefdata.derivatives()}
    for c in P.mgh_cases():
        r = ref[c.id]
        assert np.array_equal(r["jacobian"][0], jac[c.id][0]), c.id  # the same Arb Jacobians as optimize's
        if c.n > P.HESSIAN_MAX_N:
            assert r["hessian"] is None
            continue
        _, _, H = ad.value_grad_hess(lambda x, M: sum((t * t for t in c.fun(x, M=M)), 0.0 * x[0]), r["x0"].tolist())
        hi = r["hessian"][0]
        assert np.max(np.abs(hi - H)) <= 1e-10 * max(1.0, np.max(np.abs(hi))), c.id
        assert np.array_equal(hi, hi.T), c.id
