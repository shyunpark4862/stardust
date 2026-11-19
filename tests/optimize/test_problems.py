"""The optimize test collection (docs/design/optimize.md §7): the Moré-Garbow-Hillstrom problems,
MINPACK's systems and scipy's test functions are what they claim to be.

  - every analytic Jacobian (and gradient, Hessian, derivative) agrees with an Arb central
    difference at 320 bits and with the exact derivatives of the dual numbers (ad.Dual2);
  - the functions written after MINPACK's drivers agree with the Fortran to the bit at the
    drivers' starting points and at a second point (data/minpack_functions.json);
  - the paper's exact minimizers give its optimal values; the transcriptions of scipy's test
    functions agree with scipy's own.
"""

import json
import math
import pathlib

import numpy as np
import pytest

from . import ad, functions, mgh, systems

DATA = pathlib.Path(__file__).parent / "data"


def _close(got, want, rtol=1e-12, atol_scale=1e-14):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    scale = max(float(np.max(np.abs(want))), 1e-300) if want.size else 1.0
    return np.all(np.abs(got - want) <= rtol * np.abs(want) + atol_scale * scale)


def _points(x0, n, seed):
    rng = np.random.default_rng(seed)
    x0 = np.array(x0, dtype=float)
    return [x0, x0 + 0.1 * (np.abs(x0) + 1.0) * rng.standard_normal(n)]


# one case per (problem, n, m)
DISTINCT = list({(c.number, c.n, c.m): c for c in mgh.CASES}.values())


@pytest.mark.parametrize("case", DISTINCT, ids=lambda c: c.id)
def test_mgh_jacobian(case):
    p = case.problem
    for k, x in enumerate(_points(case.x0(), case.n, case.number)):
        if case.number == 11:  # Gulf: keep x2 away from the data's y_i, where |y - x2|^x3 has no derivative
            x = np.array([5.0, 2.5, 0.15]) * (1.0 + 0.01 * k)
        xl = [float(v) for v in x]
        jh = np.array(case.jac(xl), dtype=float)
        assert jh.shape == (case.m, case.n)
        ja = ad.arb_jacobian(lambda v, M: p.fun(v, case.m, M), xl)
        jd = ad.jacobian_ad(lambda v, M: p.fun(v, case.m, M), xl)
        assert _close(jh, ja), case.id
        assert _close(jd, ja), case.id


@pytest.mark.parametrize("case", [c for c in systems.CASES if c.factor == 1.0], ids=lambda c: c.id)
def test_system_jacobian(case):
    fun = systems.SYSTEMS[case.number][1]
    for x in _points(case.x0(), case.n, case.number):
        xl = [float(v) for v in x]
        jh = np.array(case.jac(xl), dtype=float)
        assert jh.shape == (case.n, case.n)
        assert _close(jh, ad.arb_jacobian(fun, xl)), case.id
        assert _close(ad.jacobian_ad(fun, xl), ad.arb_jacobian(fun, xl)), case.id


@pytest.mark.parametrize("case", DISTINCT[::7], ids=lambda c: c.id)
def test_mgh_hessian_is_the_jacobian_of_the_gradient(case):
    # the Dual2 Hessian of f = sum F^2 against an Arb difference of the analytic gradient 2 J^T F
    x = [float(v) for v in case.x0()]
    if case.number == 11:
        x = [5.0, 2.5, 0.15]

    def grad(v, M):
        r = case.problem.fun(v, case.m, M)
        j = case.problem.jac(v, case.m, M)
        return [2.0 * sum(j[i][c] * r[i] for i in range(case.m)) for c in range(case.n)]

    h = mgh.hessian(case, x)
    ha = ad.arb_jacobian(grad, x)
    assert _close(h, ha, rtol=1e-11), case.id
    assert np.array_equal(h, h.T)


def test_minimizers_give_the_paper_values():
    for (num, n, m), x in mgh.MINIMIZERS.items():
        c = mgh.Case(num, n, m, 1.0)
        f = float(c.f([np.float64(v) for v in x]))
        known = mgh.KNOWN[(num, n, m)][0]
        assert abs(f - known) <= 1e-12 * max(1.0, known), (num, f)


def test_starting_points_follow_initpt():
    # MINPACK's rule for multiples: a zero start (Watson) becomes the vector of the factor
    assert mgh.x0(20, 6, 10.0) == [10.0] * 6
    assert mgh.x0(1, 2, 100.0) == [-120.0, 100.0]
    assert systems.x0(6, 9, 100.0) == [100.0] * 9
    assert len(mgh.CASES) == 110 and len(mgh.MINPACK_LSQ_CASES) == 53 and len(systems.CASES) == 55
    assert sorted({c.number for c in mgh.CASES}) == list(range(1, 36))


def test_functions_agree_with_fortran():
    # data/minpack_functions.json: SSQFCN/SSQJAC and VECFCN/VECJAC compiled with gfortran
    # -ffp-contract=off (tools/gen_optimize_fortran.py)
    rows = json.loads((DATA / "minpack_functions.json").read_text())
    assert len(rows) == 2 * (53 + 55)
    with np.errstate(all="ignore"):
        for e in rows:
            x = [np.float64(v) for v in e["x"]]
            if e["set"] == "lsq":
                c = mgh.Case(mgh.MINPACK_LSQ[e["nprob"]], e["n"], e["m"], e["factor"])
            else:
                c = systems.SystemCase(e["nprob"], e["n"], e["factor"])
            if e["point"] == 1:
                assert x == [np.float64(v) for v in c.x0()], c.id
            # bit for bit
            assert np.array_equal(np.array(c.fun(x), dtype=float), np.array(e["f"])), c.id
            assert np.array_equal(np.array(c.jac(x), dtype=float), np.array(e["jac"])), c.id


# ---- scipy's functions


def test_rosen_family_against_scipy():
    from scipy.optimize import rosen, rosen_der, rosen_hess, rosen_hess_prod

    for x in ([1.3, 0.7, 0.8, 1.9, 1.2], [-1.2, 1.0], [0.5, -0.25, 2.0]):
        xa = np.array(x)
        assert math.isclose(functions.rosen(x), rosen(xa), rel_tol=1e-15)
        assert np.allclose(functions.rosen_der(x), rosen_der(xa), rtol=1e-15, atol=0)
        assert np.allclose(np.array(functions.rosen_hess(x)), rosen_hess(xa), rtol=1e-15, atol=0)
        p = [0.3, -1.0, 2.0, 0.5, 1.5][: len(x)]
        assert np.allclose(functions.rosen_hess_prod(x, p), rosen_hess_prod(xa, np.array(p)), rtol=1e-14)
        assert _close(functions.rosen_der(x), ad.arb_jacobian(lambda v, M: [functions.rosen(v, M)], x)[0])
        assert _close(np.array(functions.rosen_hess(x)), ad.value_grad_hess(functions.rosen, x)[2])


def test_maxent_and_driver1():
    x = [0.1, -0.4, 0.3]
    assert _close(functions.maxent_der(x), ad.arb_jacobian(lambda v, M: [functions.maxent(v, M)], x)[0])
    y = [3.0 - 0.1 * i for i in range(25)]
    assert _close(functions.lbfgsb_driver1_der(y), ad.arb_jacobian(lambda v, M: [functions.lbfgsb_driver1(v, M)], y)[0])
    # scipy's CheckOptimize problem
    f0 = math.log(5.0)
    assert functions.maxent(functions.MAXENT_X0) == f0


def test_scalar_root_cases_against_scipy():
    from scipy.optimize import _tstutils

    sc = {d["ID"]: d for d in _tstutils.get_tests("aps")}
    assert len(functions.APS) == len(sc) == 154
    for c in functions.APS:
        d = sc[c.id]
        assert c.bracket == tuple(map(float, d["bracket"])) and c.x0 == d["x0"] and c.root == d["root"]
        a, b = c.bracket
        # scipy's second derivatives of the families 2, 9 (n > 1) and 10 are wrong (functions.py)
        wrong_fpp = c.id[:6] in ("aps.02", "aps.10") or (c.id[:6] == "aps.09" and c.id != "aps.09.00")
        for x in np.linspace(a, b, 13)[1:-1]:
            pairs = [(c.f, d["f"]), (c.fprime, d["fprime"])] + ([] if wrong_fpp else [(c.fprime2, d["fprime2"])])
            for mine, theirs in pairs:
                u, v = mine(float(x)), theirs(float(x), *d["args"])
                assert abs(u - v) <= 1e-14 * abs(v) + 1e-300, (c.id, x)
            if wrong_fpp and c.id[:6] == "aps.02":
                assert math.isclose(c.fprime2(float(x)), -d["fprime2"](float(x)), rel_tol=1e-14)
    ch = _tstutils.get_tests("chandrupatla")
    for c, d in zip(functions.CHANDRUPATLA, ch, strict=True):
        assert c.bracket == tuple(map(float, d["bracket"])) and c.root == d["root"] and c.nfev_paper == d["nfeval"]
        for x in np.linspace(*c.bracket, 11)[1:-1]:
            assert abs(c.f(float(x)) - d["f"](np.float64(x))) <= 1e-14 * abs(d["f"](np.float64(x))) + 1e-300


@pytest.mark.parametrize(
    "case", [c for c in functions.ROOT_CASES if c.fprime is not None and c.smooth], ids=lambda c: c.id
)
def test_scalar_derivatives(case):
    a, b = case.bracket
    for x in (a + 0.3 * (b - a), a + 0.71 * (b - a)):
        if case.id.startswith("aps.02"):
            x = case.root  # away from the poles at i^2
        d = ad.arb_derivative(case.f, x)
        assert abs(case.fprime(x) - d) <= 1e-12 * abs(d) + 1e-280, case.id
        if case.fprime2 is not None:
            d2 = ad.arb_derivative(case.fprime, x)
            assert abs(case.fprime2(x) - d2) <= 1e-11 * abs(d2) + 1e-280, case.id


def test_minimization_brackets():
    for c in functions.MIN_CASES:
        if len(c.bracket) == 2:  # scipy's two-point bracket, which minimize_scalar widens
            continue
        x1, x2, x3 = c.bracket
        f1, f2, f3 = (c.f(np.float64(v)) for v in c.bracket)
        assert x1 < x2 < x3 and f1 >= f2 <= f3 and (f1 > f2 or f3 > f2), c.id
    assert len([c for c in functions.MIN_CASES if c.nit_paper]) == 55


def test_lsq_linear_problems():
    for pid, (A, b, bounds) in functions.LSQ_LINEAR.items():
        A = np.array(A)
        assert A.ndim == 2 and A.shape[0] == len(b), pid
        if bounds is not None:
            assert len(bounds[0]) == len(bounds[1]) == A.shape[1], pid
