"""sd.optimize.root (docs/api/optimize.md §8, docs/design/optimize.md §3.6, §7): scipy 1.18.1 on
MINPACK's 14 HYBRD/HYBRJ systems (the recorded runs, tests/optimize/data/scipy_root.json, and
every call point live), MINPACK's Fortran HYBRJ1 and HYBRD1, the options, the counts, the
result and the errors."""

import json
import math
import pathlib
import warnings

import numpy as np
import pytest
from scipy.optimize import root as scipy_root

import stardust as sd

from . import systems

DATA = pathlib.Path(__file__).parent / "data"
REF = json.loads((DATA / "scipy_root.json").read_text())
F77 = json.loads((DATA / "fortran_minpack.json").read_text())
F77_SCIPY = json.loads((DATA / "fortran_vs_scipy.json").read_text())
TOL = math.sqrt(2.0**-52)  # the drivers' tolerance


def fns(c, log=None):
    """fun and jac of a system for sd (numpy values), logging (kind, x) when asked."""
    F, J = systems.as_numpy(c)

    def f(x):
        if log is not None:
            log.append(("f", x.tolist()))
        return F(np.asarray(x.tolist()))

    def j(x):
        if log is not None:
            log.append(("j", x.tolist()))
        return J(np.asarray(x.tolist()))

    return f, j


def run(c, method, jac, log=None, **options):
    f, j = fns(c, log)
    with np.errstate(all="ignore"):
        return sd.optimize.root(f, c.x0(), method=method, jac=j if jac else None, options=options or None)


def scipy_calls(c, method, jac, **options):
    """scipy's call points without its two extra evaluations at x0 (its shape checks and its
    wrapper's), so in the order MINPACK makes them."""
    F, J = systems.as_numpy(c)
    log = []

    def f(x):
        log.append(("f", x.tolist()))
        return F(x)

    def j(x):
        log.append(("j", x.tolist()))
        return J(x)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        r = scipy_root(f, np.array(c.x0()), method=method, jac=j if jac else None, options=options or None)
    x0, out, nf, nj = c.x0(), [], 0, 0
    for kind, x in log:
        if kind == "f" and nf < 2 and x == x0:
            nf += 1
            continue
        if kind == "j" and nj < 1 and x == x0:
            nj += 1
            continue
        out.append((kind, x))
    return r, out


def dist(a, b, x0):
    return max(abs(u - v) for u, v in zip(a, b)) / max(max(abs(v) for v in b), max(abs(v) for v in x0), 1e-300)


# ---- scipy 1.18.1, recorded (O9)


@pytest.mark.parametrize("method", ["hybr", "lm"])
@pytest.mark.parametrize("jac", ["analytic", "none"])
def test_root_is_scipy_to_the_bit(method, jac):
    # every field of all 55 runs: x's bits, F, status, message, converged; nfev is scipy's less
    # its two extra calls at x0, njev scipy's ('lm': lmder in scipy's build, O95)
    for c in systems.CASES:
        rec = REF[c.id][f"{method}/{jac}"]
        r = run(c, method, jac == "analytic")
        assert r.x.tolist() == rec["x"], c.id
        assert r.fun.tolist() == rec["fun"], c.id
        assert (r.status, r.message, r.converged) == (rec["status"], rec["message"], rec["success"]), c.id
        assert r.nfev == rec["nfev"] - 2, c.id
        if jac == "analytic":
            assert r.njev == rec["njev"], c.id


@pytest.mark.parametrize("method", ["hybr", "lm"])
@pytest.mark.parametrize("jac", [True, False])
def test_root_calls_scipys_points(method, jac):
    # live: every point fun and jac are called at, in order, to the bit (hybrj / hybrd, lmder /
    # lmdif)
    for c in systems.CASES:
        log = []
        run(c, method, jac, log)
        _, want = scipy_calls(c, method, jac)
        assert log == want, c.id


OPTION_CASES = [
    "hybrd01-n2-x10",
    "hybrd04-n4-x1",
    "hybrd06-n6-x1",
    "hybrd08-n10-x1",
    "hybrd12-n10-x10",
    "hybrd14-n10-x1",
]


@pytest.mark.parametrize(
    "jac, sd_options, scipy_options",
    [
        (True, {"factor": 1.0}, {"factor": 1.0}),
        (True, {"xtol": 1e-3}, {"xtol": 1e-3}),
        (True, {"max_nfev": 7}, {"maxfev": 7}),
        (True, {"diag": [2.0]}, {"diag": [2.0]}),
        (False, {"eps": 1e-10}, {"eps": 1e-10}),
        (False, {"band": (1, 1)}, {"band": (1, 1)}),
        (False, {"band": (0, 2), "factor": 0.5, "max_nfev": 30}, {"band": (0, 2), "factor": 0.5, "maxfev": 30}),
    ],
)
def test_hybr_options_follow_scipy(jac, sd_options, scipy_options):
    for cid in OPTION_CASES:
        c = systems.BY_ID[cid]
        so, do = dict(scipy_options), dict(sd_options)
        if "diag" in so:
            so["diag"] = do["diag"] = so["diag"] * c.n
        log = []
        r = run(c, "hybr", jac, log, **do)
        rs, want = scipy_calls(c, "hybr", jac, **so)
        assert log == want, cid
        assert (r.status, r.message, r.nfev) == (rs.status, rs.message, rs.nfev - 2), cid


@pytest.mark.parametrize(
    "jac, options",
    [
        (True, {"factor": 1.0}),
        (True, {"xtol": 1e-3, "ftol": 1e-3}),
        (True, {"gtol": 1e-6}),
        (True, {"diag": [2.0]}),
        (False, {"eps": 1e-10}),
        (False, {"factor": 0.5}),
    ],
)
def test_lm_options_follow_scipy(jac, options):
    # the options of lmder / lmdif: the same call points to the bit (scipy's names; max_nfev is
    # its maxfev, tested above)
    for cid in OPTION_CASES:
        c = systems.BY_ID[cid]
        o = dict(options)
        if "diag" in o:
            o["diag"] = o["diag"] * c.n
        log = []
        r = run(c, "lm", jac, log, **o)
        rs, want = scipy_calls(c, "lm", jac, **o)
        assert log == want, cid
        assert (r.status, r.message, r.nfev) == (rs.status, rs.message, rs.nfev - 2), cid
        assert r.x.tolist() == rs.x.tolist(), cid


def test_jac_true_is_memoized_as_scipy():
    # fun returns (F, J) and is called only at a new point (scipy's MemoizeJac): hybr calls it at
    # scipy's points; both methods take the steps of the separate fun and jac
    for cid in OPTION_CASES:
        c = systems.BY_ID[cid]
        F, J = systems.as_numpy(c)

        def pair(x, log):
            log.append(x.tolist())
            x = np.asarray(x.tolist())
            return F(x), J(x)

        for method in ("hybr", "lm"):
            mine, theirs = [], []
            with np.errstate(all="ignore"):
                r = sd.optimize.root(lambda x: pair(x, mine), c.x0(), method=method, jac=True)
                scipy_root(lambda x: pair(x, theirs), np.array(c.x0()), method=method, jac=True)
            if method == "hybr":
                assert mine == theirs, cid
            assert all(mine[k] != mine[k - 1] for k in range(1, len(mine))), cid
            assert len(mine) <= r.nfev + r.njev, cid
            r2 = run(c, method, True)
            assert (r.x.tolist(), r.status, r.nfev, r.njev) == (r2.x.tolist(), r2.status, r2.nfev, r2.njev), cid


# ---- MINPACK's Fortran HYBRJ1 / HYBRD1 (diag = 1, mode 2; factor 100; tol sqrt(eps))

F77_AGREE = {  # same number of calls, same info, x within 1e-12, within 1e-6 (of 55): scipy's
    "hybrj": (42, 50, 37, 44),
    "hybrd": (41, 48, 33, 44),
}


@pytest.mark.parametrize("kind", ["hybrj", "hybrd"])
def test_against_fortran_hybrj1_hybrd1(kind):
    # the same runs as scipy's fsolve with the drivers' options: x's bits, info and the number of
    # calls are scipy's in all 55, so the distance to the Fortran original is scipy's
    got = [0, 0, 0, 0]
    jac = kind == "hybrj"
    for run77, cmp in zip(F77[kind], F77_SCIPY[kind]):
        c = systems.BY_ID[run77["id"]]
        log = []
        n = c.n
        r = run(c, "hybr", jac, log, xtol=TOL, diag=[1.0] * n, factor=100, max_nfev=(100 if jac else 200) * (n + 1))
        assert r.x.tolist() == cmp["scipy"]["x"], c.id
        assert r.status == cmp["scipy"]["info"], c.id
        assert len(log) == cmp["scipy"]["calls"], c.id
        calls77 = run77["calls_f"] + run77["calls_d"] + run77["calls_j"]
        x77 = np.array(run77["x"])
        d = float(np.max(np.abs(x77 - np.array(r.x.tolist()))) / max(np.max(np.abs(x77)), 1e-300))
        for i, s in enumerate([len(log) == calls77, r.status == run77["info"], d <= 1e-12, d <= 1e-6]):
            got[i] += s
    assert tuple(got) == F77_AGREE[kind]


# ---- counts, stopping, values


def test_counts():
    # nfev: fun's calls, x0 and the differences included (n per difference Jacobian); njev: the
    # Jacobians; n_iter: the steps tried (hybr: one call each), the Jacobians of the iteration (lm)
    c = systems.BY_ID["hybrd04-n4-x1"]
    for method in ("hybr", "lm"):
        for jac in (True, False):
            log = []
            r = run(c, method, jac, log)
            assert r.nfev == sum(k == "f" for k, _ in log)
            if jac:
                assert r.njev == sum(k == "j" for k, _ in log)
            if method == "hybr":
                assert r.nfev == 1 + r.n_iter + (0 if jac else c.n * r.njev)
            else:
                assert r.n_iter == r.njev and r.nfev > 1 + (0 if jac else c.n * r.njev)


def test_default_max_nfev():
    # scipy's maxfev = 0: 100 (n + 1) with a Jacobian, 200 (n + 1) without (both methods, O14)
    c = systems.BY_ID["hybrd06-n9-x10"]  # Watson n = 9 from 10 x0: lm reaches the cap
    r = run(c, "lm", True)
    assert r.status == 5 and r.nfev == 1000
    assert r.message == "Number of calls to function has reached maxfev = 1000."
    r = run(c, "hybr", True, max_nfev=20)
    assert r.status == 2 and not r.converged
    assert r.message == "The number of calls to function has reached maxfev = 20."


def test_messages_are_scipys():
    c = systems.BY_ID["hybrd01-n2-x1"]
    r = run(c, "hybr", True, xtol=0.5)
    assert r.status == 1 and r.message == "The solution converged."
    r = run(c, "lm", True, xtol=0.5, ftol=0.25)
    assert r.message.startswith("The relative error between two consecutive iterates is at most 0.500000") or (
        "0.250000" in r.message
    )


def test_tol_is_xtol():
    c = systems.BY_ID["hybrd04-n4-x1"]
    for method in ("hybr", "lm"):
        a = sd.optimize.root(fns(c)[0], c.x0(), method=method, tol=1e-4)
        b = run(c, method, False, xtol=1e-4)
        assert a.x.tolist() == b.x.tolist() and a.nfev == b.nfev


def test_root_is_a_root():
    for c in systems.CASES:
        if c.number in (6, 7):  # Watson and Chebyquad have no root at some sizes
            continue
        r = run(c, "hybr", True)
        if r.converged:
            F, _ = systems.as_numpy(c)
            assert np.max(np.abs(F(np.array(r.x.tolist())))) <= 1e-6, c.id


def test_non_finite_trial_points_are_failed_steps():
    # F is not finite away from the root: the step fails and the bound shrinks (MINPACK's
    # arithmetic), the rank-one update is skipped; the iteration still reaches the root
    def f(x):
        if x[0] > 2.0:
            return [math.nan, math.nan]
        return [x[0] ** 2 - 2.0, x[1] - 1.0]

    for method in ("hybr", "lm"):
        r = sd.optimize.root(f, [1.0, 0.0], method=method)
        assert r.converged and abs(r.x[0] - math.sqrt(2)) < 1e-12, method


def test_non_finite_jacobian_is_an_error():
    def j(x):
        return [[math.inf, 0.0], [0.0, 1.0]]

    for method in ("hybr", "lm"):
        with pytest.raises(ValueError, match=r"root\(\): 'jac' is not finite at x = \[1.0, 0.0\]"):
            sd.optimize.root(lambda x: [x[0] - 2, x[1]], [1.0, 0.0], method=method, jac=j)
    with pytest.raises(ValueError, match="the differences of 'fun' are not finite"):
        sd.optimize.root(lambda x: [0.0 if x[0] == 1.0 else math.nan, x[1]], [1.0, 0.0])


def test_result_fields():
    c = systems.BY_ID["hybrd01-n2-x1"]
    for method in ("hybr", "lm"):
        r = run(c, method, True)
        for name in ("x", "fun", "converged", "status", "message", "n_iter", "nfev", "njev", "method"):
            assert hasattr(r, name), name
        # O79: no fjac, r, qtf (nor lm's ipvt, cov_x)
        for name in ("fjac", "r", "qtf", "ipvt", "cov_x", "jac", "cost"):
            assert not hasattr(r, name), name
        assert isinstance(r.x, sd.ndarray) and r.x.dtype == sd.float64 and r.fun.shape == (2,)
        assert type(r.status) is int and type(r.converged) is bool and r.method == method
        assert repr(r).startswith(f"OptimizeResult(method='{method}', converged=True")


def test_one_variable_and_sequences():
    for method in ("hybr", "lm"):
        r = sd.optimize.root(lambda x: [x[0] ** 3 - 8.0], [1.0], method=method, jac=lambda x: [3 * x[0] ** 2])
        assert abs(r.x[0] - 2.0) < 1e-10
        r = sd.optimize.root(lambda x: (x[0] ** 3 - 8.0,), np.array([1.0]), method=method)
        assert abs(r.x[0] - 2.0) < 1e-8


def test_fun_gets_a_new_array():
    # a new float64 sd.ndarray at each call; changing it does not change the iteration
    seen = []

    def f(x):
        seen.append(x)
        out = [x[0] - 1.0, x[1] - 2.0]
        x[0] = 100.0
        return out

    for method in ("hybr", "lm"):
        seen.clear()
        r = sd.optimize.root(f, [0.0, 0.0], method=method)
        assert r.converged and r.x.tolist() == [1.0, 2.0] or max(abs(r.x[0] - 1), abs(r.x[1] - 2)) < 1e-12
        assert all(isinstance(x, sd.ndarray) and x.dtype == sd.float64 for x in seen)
        assert len({id(x) for x in seen}) == len(seen)


def test_same_bits_with_1_and_10_threads():
    c = systems.BY_ID["hybrd12-n10-x10"]
    old = sd.get_num_threads()
    out = []
    try:
        for method in ("hybr", "lm"):
            for t in (1, 10):
                sd.set_num_threads(t)
                out.append(run(c, method, False))
    finally:
        sd.set_num_threads(old)
    for a, b in (out[0:2], out[2:4]):
        assert a.x.tolist() == b.x.tolist() and a.nfev == b.nfev


def test_errors():
    f = lambda x: [x[0] - 1.0, x[1]]  # noqa: E731
    with pytest.raises(TypeError, match=r"root\(\): 'fun' must be callable, not int"):
        sd.optimize.root(3, [0.0, 0.0])
    with pytest.raises(ValueError, match=r"root\(\): 'method' must be lowercase: 'hybr', not 'HYBR'"):
        sd.optimize.root(f, [0.0, 0.0], method="HYBR")
    # O78: only 'hybr' and 'lm'
    with pytest.raises(ValueError, match=r"root\(\): 'method' must be 'hybr' or 'lm', not 'broyden1'"):
        sd.optimize.root(f, [0.0, 0.0], method="broyden1")
    with pytest.raises(ValueError, match=r"root\(\): 'x0' is empty"):
        sd.optimize.root(f, [])
    with pytest.raises(ValueError, match=r"root\(\): 'x0' must be finite"):
        sd.optimize.root(f, [0.0, math.nan])
    with pytest.raises(TypeError, match=r"root\(\): 'x0' must be real"):
        sd.optimize.root(f, [1j, 0.0])
    with pytest.raises(ValueError, match=r"root\(\): 'jac' must be callable, True or None, not False"):
        sd.optimize.root(f, [0.0, 0.0], jac=False)
    with pytest.raises(TypeError, match=r"root\(\): 'jac' must be callable, True or None, not str"):
        sd.optimize.root(f, [0.0, 0.0], jac="2-point")
    # the system's shape
    with pytest.raises(ValueError, match=r"root\(\): 'fun' must return 2 values \(one per variable\), not 3"):
        sd.optimize.root(lambda x: [1.0, 2.0, 3.0], [0.0, 0.0])
    with pytest.raises(ValueError, match=r"root\(\): 'fun' must return a 1-dimensional array, not a scalar"):
        sd.optimize.root(lambda x: 1.0, [0.0])
    with pytest.raises(ValueError, match=r"root\(\): 'jac' must return shape \(2, 2\), not \(2, 3\)"):
        sd.optimize.root(f, [0.0, 0.0], jac=lambda x: np.zeros((2, 3)))
    with pytest.raises(TypeError, match=r"root\(\): 'fun' must return a pair \(F, J\) when jac=True, not 3 values"):
        sd.optimize.root(lambda x: [[1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], 3], [0.0, 0.0], jac=True)
    with pytest.raises(TypeError, match=r"root\(\): 'fun' must return a pair \(F, J\) when jac=True, not ndarray"):
        sd.optimize.root(lambda x: x, [0.0, 0.0], jac=True)
    with pytest.raises(ValueError, match=r"root\(\): 'fun' is not finite at 'x0'"):
        sd.optimize.root(lambda x: [math.inf, 0.0], [0.0, 0.0])
    # options (O16, O14, O62)
    with pytest.raises(TypeError, match=r"root\(\): 'options' must be a dict or None, not list"):
        sd.optimize.root(f, [0.0, 0.0], options=[])
    with pytest.raises(ValueError, match=r"root\(\): method='hybr' has no option 'ftol'"):
        sd.optimize.root(f, [0.0, 0.0], options={"ftol": 1e-3})
    with pytest.raises(ValueError, match=r"method='hybr' has no option 'maxfev' \(the cap on function"):
        sd.optimize.root(f, [0.0, 0.0], options={"maxfev": 10})
    with pytest.raises(ValueError, match=r"method='lm' has no option 'maxiter' \(the iteration cap is 'max_iter'\)"):
        sd.optimize.root(f, [0.0, 0.0], method="lm", options={"maxiter": 10})
    with pytest.raises(ValueError, match=r"method='lm' has no option 'band'"):
        sd.optimize.root(f, [0.0, 0.0], method="lm", options={"band": (1, 1)})
    with pytest.raises(ValueError, match=r"method='hybr' has no option 'col_deriv'"):
        sd.optimize.root(f, [0.0, 0.0], options={"col_deriv": True})
    with pytest.raises(ValueError, match=r"root\(\): 'eps' is only used when 'jac' is None"):
        sd.optimize.root(f, [0.0, 0.0], jac=lambda x: np.eye(2), options={"eps": 1e-8})
    with pytest.raises(ValueError, match=r"root\(\): 'band' is only used when 'jac' is None"):
        sd.optimize.root(f, [0.0, 0.0], jac=True, options={"band": (0, 0)})
    with pytest.raises(ValueError, match=r"root\(\): give 'tol' or options\['xtol'\], not both"):
        sd.optimize.root(f, [0.0, 0.0], tol=1e-6, options={"xtol": 1e-6})
    with pytest.raises(ValueError, match=r"root\(\): 'xtol' must be a finite number >= 0"):
        sd.optimize.root(f, [0.0, 0.0], options={"xtol": -1.0})
    with pytest.raises(ValueError, match=r"root\(\): 'factor' must be finite and > 0"):
        sd.optimize.root(f, [0.0, 0.0], options={"factor": 0.0})
    with pytest.raises(ValueError, match=r"root\(\): 'diag' must be finite and > 0"):
        sd.optimize.root(f, [0.0, 0.0], options={"diag": [1.0, 0.0]})
    with pytest.raises(ValueError, match=r"root\(\): 'diag' must have 2 values like 'x0', not 3"):
        sd.optimize.root(f, [0.0, 0.0], options={"diag": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match=r"root\(\): 'max_nfev' must be >= 1"):
        sd.optimize.root(f, [0.0, 0.0], options={"max_nfev": 0})
    with pytest.raises(ValueError, match=r"root\(\): 'band' must be a pair \(ml, mu\), not 3 values"):
        sd.optimize.root(f, [0.0, 0.0], options={"band": (1, 1, 1)})
    with pytest.raises(ValueError, match=r"root\(\): 'band' must be >= 0"):
        sd.optimize.root(f, [0.0, 0.0], options={"band": (-1, 1)})


def test_user_errors_pass_through():
    def f(x):
        raise KeyError("mine")

    for method in ("hybr", "lm"):
        with pytest.raises(KeyError, match="mine"):
            sd.optimize.root(f, [0.0], method=method)
        calls = []

        def g(x):
            calls.append(1)
            if len(calls) > 1:
                raise ZeroDivisionError("later")
            return [x[0] - 1.0]

        with pytest.raises(ZeroDivisionError, match="later"):
            sd.optimize.root(g, [0.0], method=method)
