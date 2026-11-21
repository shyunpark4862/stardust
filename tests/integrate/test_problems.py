"""The integrate test collection (tests/integrate/problems.py) and its Arb references
(tools/integrate_ref.py, tests/integrate/ref): the three forms of every integrand agree, every
problem has a current reference, the references agree with each other's methods and with the
closed forms the families are known by."""

import math
import random
import sys

import numpy as np
import pytest

from . import problems as P
from . import refdata

TOOLS = __import__("pathlib").Path(__file__).parents[2] / "tools"


def _tool():
    sys.path.insert(0, str(TOOLS))
    try:
        import integrate_ref
    finally:
        sys.path.remove(str(TOOLS))
    return integrate_ref


def sample_points(p, k=5):
    """k points strictly inside the range of p, away from the ends and the break points."""
    a, b = min(p.a, p.b), max(p.a, p.b)
    if math.isinf(a) and math.isinf(b):
        pts = [-3.0, -0.7, 0.3, 1.1, 2.5][:k]
    elif math.isinf(b):
        pts = [a + t for t in (0.05, 0.4, 1.3, 2.9, 7.0)][:k]
    elif math.isinf(a):
        pts = [b - t for t in (0.05, 0.4, 1.3, 2.9, 7.0)][:k]
    else:
        pts = [a + (b - a) * t for t in (0.07, 0.29, 0.45, 0.71, 0.93)][:k]
    return [x for x in pts if all(abs(x - c) > 1e-3 for c in p.points)]


def test_ids_are_unique_and_named_by_collection():
    assert len(P.BY_ID) == len(P.INTEGRALS) == 116
    assert {p.collection for p in P.INTEGRALS} == {"book", "quadpack", "scipy", "hard"}
    assert len(P.BOOK) == 30 + 21 + 11  # the book's three families
    assert {p.id.split("/")[1] for p in P.QUADPACK} == {"qags", "qagp", "qagi", "qawo", "qawf", "qaws", "qawc"}
    assert len(P.SCIPY_TANHSINH) == 20  # f1-f15 and the five transforms
    for p in P.INTEGRALS:
        assert p.source, p.id
        assert (p.weight is None) or not p.tanhsinh, p.id  # tanhsinh has no weight
        assert not p.id.endswith("/points") or not p.tanhsinh, p.id  # the twin without break points has it


@pytest.mark.parametrize("pid", [p.id for p in P.INTEGRALS])
def test_the_three_forms_agree(pid):
    from flint import arb, ctx

    p = P.get(pid)
    old = ctx.prec
    ctx.prec = 128
    try:
        for x in sample_points(p):
            m = p.f(x)
            with np.errstate(all="ignore"):
                n = float(p.f_np(np.array([x]))[0])
            assert math.isclose(m, n, rel_tol=4e-15, abs_tol=1e-300), (x, m, n)
            if pid == "scipy/quad/singular":  # piecewise: no Arb form (its reference is a closed form)
                continue
            v = p.expr(arb(x), P.Acb(False))
            got = float(v.real.mid())
            assert math.isclose(m, got, rel_tol=1e-13, abs_tol=1e-12), (x, m, got)  # cos(2^10 sin x): a large condition
    finally:
        ctx.prec = old


def test_every_problem_has_a_current_reference():
    tool = _tool()
    ref = refdata.integrals(keep_provenance=True)
    prov = ref.pop("_provenance")
    assert prov["tool"] == "tools/integrate_ref.py" and prov["target_bits"] == 110 and prov["dropped"] == []
    assert set(ref) == set(P.BY_ID)
    for pid, e in ref.items():
        p = P.get(pid)
        assert e["digest"] == tool.integral_digest(p), f"{pid}: stale reference (run tools/integrate_ref.py)"
        names = [n for n, _ in tool.methods_of(p)]
        assert [e["method"], *e["checks"]] == names, pid
        hi, lo = e["value"]
        assert math.isfinite(hi) and abs(lo) <= abs(hi) * 2.0**-52, pid


def test_the_methods_cover_the_kinds_of_integrand():
    ref = refdata.integrals()
    methods = [e["method"] for e in ref.values()]
    checks = [c for e in ref.values() for c in e["checks"]]
    assert methods.count("acb.integral") >= 10  # no closed form for the double limits (osc, f3, Bessel)
    assert "substitution" in checks and checks.count("acb.integral") >= 10


def test_the_oscillation_family_against_its_closed_form():
    # pi J0(2^a) holds for the exact pi; the double pi is 1.2e-16 short, which moves the integral by
    # about cos(2^a sin pi) (pi - math.pi): the numerical references are that close to the closed forms
    from flint import ctx

    ref = refdata.integrals()
    old = ctx.prec
    ctx.prec = 200
    try:
        for al in P.BOOK_OSC_ALPHAS:
            closed = float(P.book_osc_closed(al).mid())
            got = refdata.value(ref[f"book/osc/{int(al)}"])
            assert abs(got - closed) < 2.5e-16, (al, got, closed)
    finally:
        ctx.prec = old


def test_the_double_upper_limit_below_pi_over_2():
    # f9 and f10 on [0, math.pi / 2]: the references are for the double limit, below pi/2
    ref = refdata.integrals()
    f9 = refdata.value(ref["scipy/tanhsinh/f9"])
    f10 = refdata.value(ref["scipy/tanhsinh/f10"])
    d = math.pi / 2 - P.HALF_PI + 6.123233995736766e-17  # pi/2 - the double, to double precision
    assert abs(f9 - (-math.pi * math.log(2) / 2 - (d * math.log(d) - d))) < 1e-15
    assert abs(f10 - (math.pi * math.sqrt(2) / 2 - 2 * math.sqrt(d))) < 1e-15


def test_series_multiple_and_vector_references():
    s = refdata.series()
    assert set(s) == set(P.SERIES_BY_ID)
    assert refdata.value(s["series/zeta/2"]) == math.pi**2 / 6
    m = refdata.multiple()
    assert set(m) == {x.id for x in P.MULTI}
    assert math.isclose(refdata.value(m["multi/dbl-gauss-plane"]), math.pi, rel_tol=1e-16)
    v = refdata.vector()
    assert set(v) == {x.id for x in P.VECTOR}
    assert [len(e["value"]) for e in v.values()] == [3, 2]


def test_samples_and_batches():
    sm = P.samples()
    assert all(y.shape[-1] >= 2 for y, _, _ in sm.values())
    for bt in P.BATCHES:
        assert all(i in P.BY_ID for i in bt.ids())
        for i, prm in zip(bt.ids(), bt.params):
            x = np.array([0.123, 0.5, 0.877])
            with np.errstate(all="ignore"):
                assert np.array_equal(bt.f_np(x, np.full(3, prm)), P.get(i).f_np(x)), i


@pytest.mark.deep
def test_a_sample_of_references_recomputes_to_the_bit():
    tool = _tool()
    ref = refdata.integrals()
    for pid in random.Random(20261121).sample(sorted(ref), 12):
        got = tool._integral_task(pid)[1]
        assert got["value"] == ref[pid]["value"], pid
