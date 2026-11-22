"""sd.optimize.least_squares (docs/api/optimize.md §3, docs/design/optimize.md §3.4, §7): scipy 1.18.1
on the Moré-Garbow-Hillstrom collection (the recorded runs, tests/optimize/data/scipy_least_squares.json,
and every call point live), NIST StRD, MINPACK's Fortran LMDER1, the counts (O18), the bounds, the
stopping (O20), the result and the errors."""

import json
import math
import pathlib
import warnings

import numpy as np
import pytest
from scipy.optimize import least_squares as scipy_ls

import stardust as sd

from ..regression import nist_nls as N
from . import mgh

DATA = pathlib.Path(__file__).parent / "data"
REF = json.loads((DATA / "scipy_least_squares.json").read_text())
F77 = json.loads((DATA / "fortran_minpack.json").read_text())
DIFF_CALLS = {"analytic": 0, "2-point": 1, "3-point": 2}  # calls of fun per variable in a difference Jacobian


def sd_fns(c):
    return (lambda x: sd.asarray(c.fun(x.tolist()))), (lambda x: sd.asarray(c.jac(x.tolist())))


def box(c):
    x0 = c.x0()
    return [v - 0.5 * (abs(v) + 1.0) for v in x0], [v + 0.5 * (abs(v) + 1.0) for v in x0]


def dist(a, b, x0):
    """max |a - b| over max(|b|_inf, |x0|_inf): iterates near 0 are compared at x0's scale."""
    return max(abs(u - v) for u, v in zip(a, b)) / max(max(abs(v) for v in b), max(abs(v) for v in x0), 1e-300)


def run(c, method, jac, bounded=False, **kw):
    F, J = sd_fns(c)
    if bounded:
        kw["bounds"] = box(c)
    with np.errstate(all="ignore"):  # the problems' overflows far from the solution
        return sd.optimize.least_squares(F, c.x0(), jac=J if jac == "analytic" else jac, method=method, **kw)


# ---- scipy 1.18.1, recorded (O9): the same steps, so the same status, counts and x, except
# where rounding decides: near-singular or exactly rank-deficient Jacobians (the linear rank-1
# functions, Watson n = 9, 12, Biggs EXP6 whose x0 makes two columns equal, Gaussian at 100 x0,
# Brown almost-linear at 100 x0 with cond ~ 1e16), and the differences, which turn the last bits
# of an iterate into ~1e-8 of the next Jacobian. Counts of agreement (of 110 cases, 47 boxed):
AGREE = {  # status, nfev (scipy's count: sd's less the difference calls), njev, x within 1e-6
    "trf/analytic": (110, 107, 107, 108),
    "trf/analytic/box": (47, 45, 46, 45),
    "trf/2-point": (106, 97, 94, 93),
    "trf/2-point/box": (47, 42, 43, 41),
    "lm/analytic": (110, 108, 108, 108),
    "lm/2-point": (110, 104, 110, 104),
}
# trf with the analytic Jacobian: every status agrees; the paths that part
TRF_APART = {
    "mgh08-n3-m15-x100": "rounding growth, one trial more at the end",
    "mgh27-n10-m10-x100": "cond(J(x0)) ~ 1e16: the first trust-region step differs by 3e-3",
    "mgh18-n6-m13-x1": "two equal columns at x0: rounding growth from the start",
}


@pytest.mark.parametrize("group", sorted(AGREE))
def test_against_recorded_scipy(group):
    method, jac, *rest = group.split("/")
    bounded = bool(rest)
    cases = [c for c in mgh.CASES if not bounded or c.factor == 1.0]
    got = [0, 0, 0, 0]
    for c in cases:
        rec = REF[c.id + ("-box" if bounded else "")][f"{method}/{jac}"]
        r = run(c, method, jac, bounded)
        n = c.n
        # converged is scipy's success (no case here ends with a non-finite cost)
        assert r.converged == rec["success"], c.id
        assert r.message == rec["message"] or r.status != rec["status"], c.id
        same = [
            r.status == rec["status"],
            r.nfev - DIFF_CALLS[jac] * n * r.njev == rec["nfev"],
            rec["njev"] is None or r.n_iter == rec["njev"],
            dist(r.x.tolist(), rec["x"], c.x0()) <= 1e-6,
        ]
        for i, s in enumerate(same):
            got[i] += s
        if group == "trf/analytic":
            assert same[0], c.id
            assert all(same) or c.id in TRF_APART, c.id
        # the active bounds: scipy's
        assert r.active_mask.tolist() == rec["active_mask"] or not same[3], c.id
    assert all(g >= a for g, a in zip(got, AGREE[group])), (got, AGREE[group])


def calls_of(fn):
    seen = []

    def wrapped(x):
        seen.append([float(v) for v in x.tolist()] if isinstance(x, sd.ndarray) else [float(v) for v in x])
        return fn(x)

    return wrapped, seen


@pytest.mark.parametrize("method, bounded, least", [("trf", False, 103), ("trf", True, 43), ("lm", False, 104)])
def test_every_call_point_follows_scipy(method, bounded, least):
    # live: the points where fun is called, in order, with the analytic Jacobian (no
    # difference noise): the same number of calls and every point within 1e-10 in this many
    # cases (the rest are the ill-conditioned ones above)
    good = 0
    cases = [c for c in mgh.CASES if not bounded or c.factor == 1.0]
    for c in cases:
        F, J = sd_fns(c)
        Fn, Jn, _, _ = mgh.as_numpy(c)
        fs, a = calls_of(F)
        fn, b = calls_of(Fn)
        kw = {"bounds": box(c)} if bounded else {}
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            sd.optimize.least_squares(fs, c.x0(), jac=J, method=method, **kw)
            scipy_ls(fn, np.array(c.x0()), jac=Jn, method=method, **kw)
        x0 = c.x0()
        if len(a) == len(b) and all(dist(u, v, x0) <= 1e-10 for u, v in zip(a, b)):
            good += 1
        # the start and the first Jacobian's point are the same bits
        assert a[0] == b[0]
    assert good >= least, good


def as_scipy_calls(log, jac):
    """sd's calls as scipy makes them. scipy's VectorFunction keeps the values at the last x, so
    the residuals at a trial point equal to x (a step lost to rounding), which MINPACK evaluates
    again, are not computed again; with differences a call at the point just called is such a
    call."""
    if jac != "analytic":
        return [t for i, t in enumerate(log) if i == 0 or t != log[i - 1]]
    out, last, have = [], None, set()
    for kind, x in log:
        if x != last:
            last, have = x, set()
        if kind in have:
            continue
        have.add(kind)
        out.append((kind, x))
    return out


# where the call points of 'lm' part from scipy's, by the decisions (of 110): a trial point
# where the residuals are not finite is rejected (G74; scipy goes on with NaN), and max_nfev
# counts the difference calls (O25; the trials then stop elsewhere when steps are rejected)
LM_APART = {
    "analytic": {"mgh09-n3-m15-x100", "mgh18-n6-m13-x10"},
    "2-point": {
        "mgh09-n3-m15-x100",
        "mgh03-n2-m2-x10",
        "mgh03-n2-m2-x100",
        "mgh05-n2-m3-x10",
        "mgh05-n2-m3-x100",
        "mgh20-n12-m31-x100",
    },
}


@pytest.mark.parametrize("jac", ["analytic", "2-point"])
def test_lm_calls_scipys_points_to_the_bit(jac):
    # O95: lmder in scipy's build. Live, the same residual function for both: every point fun
    # and jac are called at, in order, to the bit, and x, fun and jac of the result; scipy then
    # computes jac(x) once more where the last Jacobian is already x's (O24), which sd does not
    for c in mgh.CASES:
        a, b = [], []

        def logged(log, kind, fn):
            def w(x):
                log.append((kind, [float(v) for v in x.tolist()]))
                return fn([float(v) for v in x.tolist()])

            return w

        def jn(x):
            return np.array(c.jac(x), dtype=float).reshape(c.m, c.n)

        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            sj = logged(a, "j", lambda x: sd.asarray(c.jac(x))) if jac == "analytic" else jac
            r = sd.optimize.least_squares(logged(a, "f", lambda x: sd.asarray(c.fun(x))), c.x0(), jac=sj, method="lm")
            qj = logged(b, "j", jn) if jac == "analytic" else jac
            q = scipy_ls(logged(b, "f", lambda x: np.array(c.fun(x))), np.array(c.x0()), jac=qj, method="lm")
        mine, theirs = as_scipy_calls(a, jac), as_scipy_calls(b, jac) if jac != "analytic" else b
        xf = r.x.tolist()
        rest = theirs[len(mine) :]
        same = theirs[: len(mine)] == mine and all(sum(u != v for u, v in zip(x, xf)) <= 1 for _, x in rest)
        assert same == (c.id not in LM_APART[jac]), c.id
        if same:
            assert len(rest) <= c.n + 1, c.id
            assert xf == q.x.tolist() and r.fun.tolist() == q.fun.tolist(), c.id
            assert r.jac.tolist() == q.jac.tolist() and r.status == q.status, c.id


def test_3_point_and_diff_step_follow_scipy():
    # live, on well-conditioned cases
    ids = [
        "mgh01-n2-m2-x1",
        "mgh07-n3-m3-x1",
        "mgh08-n3-m15-x1",
        "mgh12-n3-m10-x1",
        "mgh17-n5-m33-x1",
        "mgh19-n11-m65-x1",
        "mgh27-n10-m10-x1",
    ]
    for cid in ids:
        c = mgh.BY_ID[cid]
        F, _ = sd_fns(c)
        Fn, _, _, _ = mgh.as_numpy(c)
        for method in ("trf", "lm"):
            for jac, ds in (("3-point", None), ("2-point", 1e-6), ("3-point", [1e-4] * c.n)):
                r = sd.optimize.least_squares(F, c.x0(), jac=jac, method=method, diff_step=ds)
                q = scipy_ls(Fn, np.array(c.x0()), jac=jac, method=method, diff_step=ds)
                assert r.status == q.status, (cid, method, jac)
                assert r.nfev - DIFF_CALLS[jac] * c.n * r.njev == q.nfev, (cid, method, jac)
                assert dist(r.x.tolist(), q.x, c.x0()) <= 1e-6, (cid, method, jac)


# ---- NIST StRD and MINPACK


# 'lm' parts from scipy only where a trial point's residuals are not finite (G74: MGH17 and
# BoxBOD from start 1) or at max_nfev (O25: Bennett5 from start 1)
NIST_LM_APART = {"analytic": {"MGH17/start1", "BoxBOD/start1"}, "2-point": {"MGH17/start1", "Bennett5/start1"}}


def test_nist_against_scipy():
    # 'trf' with each library's model (sd's and numpy's functions round differently); 'lm' with
    # numpy's model for both, since its steps are scipy's to the bit (O95)
    agree = {}
    for name in N.NAMES:
        d = N.load(name)
        xs, ys = N.data(d, sd)
        xn, yn = N.data(d, np)
        fs, js = N.model(name, sd)
        fn, jn = N.model(name, np)
        for start in ("start1", "start2"):
            p0 = [float(v) for v in d[start]]
            for method in ("trf", "lm"):
                for jac in ("analytic", "2-point"):
                    if method == "trf":
                        f, j = (lambda b: fs(xs, *b.tolist()) - ys), (lambda b: js(xs, *b.tolist()))
                    else:
                        f = lambda b: sd.asarray(fn(xn, *np.array(b.tolist())) - yn)  # noqa: E731
                        j = lambda b: sd.asarray(jn(xn, *np.array(b.tolist())))  # noqa: E731
                    with warnings.catch_warnings(), np.errstate(all="ignore"):
                        warnings.simplefilter("ignore")
                        r = sd.optimize.least_squares(f, p0, method=method, jac=j if jac == "analytic" else jac)
                        q = scipy_ls(
                            lambda b: fn(xn, *b) - yn,
                            np.array(p0),
                            method=method,
                            jac=(lambda b: jn(xn, *b)) if jac == "analytic" else jac,
                        )
                    a = agree.setdefault((method, jac), [0, 0, 0])
                    a[0] += r.status == q.status
                    # the same point (1e-6), or, both converged, the same minimum as far as the
                    # stopping rule tells (costs within ftol = 1e-8 relative): Lanczos3's valley is
                    # so flat that the cost changes by 1e-10 over 1e-6 in x, so where the forward
                    # differences stop there follows the model's last bits (start 1 moved from
                    # within 1e-6 to 1.18e-6 when sd.exp's kernel changed, the cost to 1.5e-10)
                    same = dist(r.x.tolist(), q.x, p0) <= 1e-6 or (
                        r.status > 0 and q.status > 0 and abs(r.cost - q.cost) <= 1e-8 * q.cost
                    )
                    a[1] += same
                    # the certified values as often as scipy (6 digits), where the two stopped at
                    # different minima: at the same one, which lands nearer the certified point is
                    # the stopping rule's noise (Bennett5 from start 2 with differences: costs
                    # within 2e-10, 4.7 digits against 6.4, with numpy's model as with sd's)
                    if not same:
                        a[2] += (min(N.lre(v, cv) for v, cv in zip(r.x.tolist(), d["params"])) >= 6) - (
                            min(N.lre(v, cv) for v, cv in zip(q.x, d["params"])) >= 6
                        )
                    if method == "lm" and f"{name}/{start}" not in NIST_LM_APART[jac]:
                        m, p = len(yn), len(p0)
                        if m >= 10 * p and m * p >= 2000:
                            # nls's build of lmder (m >> n, BP-3): the same steps up to rounding
                            assert r.status == q.status and dist(r.x.tolist(), q.x, p0) <= 1e-8, (name, start, jac)
                            calls = r.nfev - (len(p0) * r.njev if jac == "2-point" else 0)
                            assert abs(calls - q.nfev) <= 3, (name, start, jac, calls, q.nfev)
                        else:  # scipy's build: its steps to the bit (O95)
                            assert r.x.tolist() == q.x.tolist() and r.status == q.status, (name, start, jac)
    # of 54 fits each: 'trf' parts on MGH09, MGH10, MGH17, BoxBOD and Bennett5 (start 1); in x
    # (or the same minimum) 'trf' with differences only on MGH17 and Bennett5 from start 1 (both
    # stopped by max_nfev or ftol far from the other's point)
    least = {
        ("trf", "analytic"): (52, 54),
        ("trf", "2-point"): (49, 52),
        ("lm", "analytic"): (52, 52),
        ("lm", "2-point"): (53, 52),
    }
    for k, (st, x) in least.items():
        assert agree[k][0] >= st and agree[k][1] >= x, (k, agree[k])
    assert agree["trf", "analytic"][2] >= 0 and agree["trf", "2-point"][2] >= 0, agree
    # 'lm': one more from start 1 of MGH17 with the analytic Jacobian, one fewer with differences
    assert agree["lm", "analytic"][2] + agree["lm", "2-point"][2] >= 0, agree


def test_lm_against_fortran_lmder1():
    # MINPACK's LMDER1 driver runs (tol = sqrt(u), gtol 0, maxfev 100 (n + 1)), compiled from
    # netlib: as close as scipy's leastsq (docs/design/optimize.md §7: calls 47, x 37/45 of 53),
    # whose steps 'lm' takes to the bit (O95). 'lm' needs gtol >= u (scipy's rule): u, whose
    # test |g| <= u does not stop these runs before the others
    tol = math.sqrt(2.0**-52)
    info_of = {2: 1, 3: 2, 4: 3, 1: 4, 0: 5}
    same = [0, 0, 0, 0]
    for rec in F77["lmder"]:
        c = mgh.BY_ID[rec["id"]]
        F, J = sd_fns(c)
        r = sd.optimize.least_squares(
            F, c.x0(), jac=J, method="lm", ftol=tol, xtol=tol, gtol=2.0**-52, max_nfev=100 * (c.n + 1)
        )
        d = max(abs(u - v) for u, v in zip(r.x.tolist(), rec["x"])) / max(max(abs(v) for v in rec["x"]), 1e-300)
        same[0] += info_of[r.status] == rec["info"]
        same[1] += r.nfev == rec["calls_f"] and r.n_iter == rec["calls_j"]
        same[2] += d <= 1e-12
        same[3] += d <= 1e-6
    # (gtol = u: Powell's singular function from x0 stops one call earlier, near its solution 0,
    # so one fewer x within 1e-12 and 1e-6 of the Fortran run's than with gtol 0)
    assert same[0] >= 47 and same[1] >= 47 and same[2] >= 36 and same[3] >= 44, same


# ---- counts (O18, O23), bounds (O19), stopping (O20)


@pytest.mark.parametrize("method", ["trf", "lm"])
@pytest.mark.parametrize("jac", ["analytic", "2-point", "3-point"])
def test_nfev_and_njev_count_every_call(method, jac):
    c = mgh.BY_ID["mgh19-n11-m65-x1"]
    F, J = sd_fns(c)
    nf, nj = [0], [0]

    def f(x):
        nf[0] += 1
        return F(x)

    def j(x):
        nj[0] += 1
        return J(x)

    r = sd.optimize.least_squares(f, c.x0(), jac=j if jac == "analytic" else jac, method=method)
    assert r.nfev == nf[0]
    if jac == "analytic":
        assert r.njev == nj[0]
    else:
        assert nj[0] == 0 and r.nfev > r.njev * DIFF_CALLS[jac] * c.n
    # n_iter: the iteration's Jacobians; 'lm' evaluates one more at its last iterate when the
    # last Jacobian is elsewhere (the result's jac, as scipy's jac(x))
    assert r.njev - r.n_iter in ((0, 1) if method == "lm" else (0,))


def test_max_nfev_counts_the_differences():
    # Powell's badly scaled function from 10 x0, 'lm': scipy stops on maxfev (100 n calls,
    # without the differences'); here the default 100 n (n + 1) counts the difference calls
    c = mgh.BY_ID["mgh03-n2-m2-x10"]
    F, J = sd_fns(c)
    r = sd.optimize.least_squares(F, c.x0(), method="lm")
    assert r.status == 0 and not r.converged and r.message == "The maximum number of function evaluations is exceeded."
    assert 600 <= r.nfev <= 600 + c.n  # the last iterate's Jacobian may go past it
    r = sd.optimize.least_squares(F, c.x0(), jac=J, method="lm")
    assert r.status == 0 and r.nfev == 200
    for method in ("trf", "lm"):
        for jac, k in (("2-point", 1), ("3-point", 2)):
            r = sd.optimize.least_squares(F, c.x0(), method=method, jac=jac, max_nfev=10)
            assert r.status == 0 and 10 <= r.nfev <= 10 + 2 * k * c.n


def test_3_point_default_cap_allows_scipys_iterations():
    # O25: max_nfev = 100 n (k n + 1) with k = 2 for '3-point': these cases stopped at a cap of
    # 100 n (n + 1); now they finish where scipy does (100 n calls besides the differences')
    for cid, method, bounded in [
        ("mgh10-n3-m16-x10", "trf", False),
        ("mgh21-n10-m10-x100", "trf", False),
        ("mgh16-n4-m20-x100", "lm", False),
        ("mgh10-n3-m16-x1", "trf", True),
    ]:
        c = mgh.BY_ID[cid]
        F, _ = sd_fns(c)
        Fn, _, _, _ = mgh.as_numpy(c)
        kw = {"bounds": box(c)} if bounded else {}
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = sd.optimize.least_squares(F, c.x0(), jac="3-point", method=method, **kw)
            q = scipy_ls(Fn, np.array(c.x0()), jac="3-point", method=method, **kw)
        assert r.converged and q.success, cid
        assert r.nfev - 2 * c.n * r.njev == q.nfev, cid
        assert r.nfev <= 100 * c.n * (2 * c.n + 1) + 2 * c.n, cid
        assert abs(r.cost - q.cost) <= 1e-10 * q.cost + 1e-20, cid


def test_bounds():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 5, 40)
    y = 3.0 * np.exp(-0.7 * t) + 0.2 - 0.05 * t + 0.03 * rng.standard_normal(40)
    ts, ys = sd.asarray(t), sd.asarray(y)
    F = lambda b: b[0] * sd.exp(-b[1] * ts) + b[2] - ys  # noqa: E731
    Fn = lambda b: b[0] * np.exp(-b[1] * t) + b[2] - y  # noqa: E731
    J = lambda b: sd.stack([sd.exp(-b[1] * ts), -b[0] * ts * sd.exp(-b[1] * ts), ts * 0.0 + 1.0], axis=1)  # noqa: E731
    Jn = lambda b: np.stack([np.exp(-b[1] * t), -b[0] * t * np.exp(-b[1] * t), t * 0.0 + 1.0], axis=1)  # noqa: E731
    for bounds, x0 in [
        (([0, 0, 0.5], [10, 10, 10]), [1.0, 1.0, 1.0]),
        (([0, 0.9, -1], [2.8, 10, 10]), [1.0, 1.0, 1.0]),
        ((0, 10), [1.0, 0.0, 1.0]),
        ((-math.inf, [math.inf, 0.5, math.inf]), [1.0, 0.4, 1.0]),
    ]:
        r = sd.optimize.least_squares(F, x0, jac=J, bounds=bounds)
        q = scipy_ls(Fn, np.array(x0), jac=Jn, bounds=bounds)
        assert r.status == q.status and r.nfev == q.nfev and r.njev == q.njev
        assert dist(r.x.tolist(), q.x, x0) <= 1e-12
        assert r.active_mask.tolist() == q.active_mask.tolist() and any(q.active_mask)
        assert abs(r.optimality - q.optimality) <= 1e-9 * q.optimality + 1e-12  # the sum J^T f
        # with differences: the same solution
        r = sd.optimize.least_squares(F, x0, bounds=bounds)
        assert dist(r.x.tolist(), q.x, x0) <= 1e-6 and r.active_mask.tolist() == q.active_mask.tolist()
    # None is no bounds: the same as infinite ones, and as scipy's default
    a = sd.optimize.least_squares(F, [1.0, 1.0, 1.0])
    b = sd.optimize.least_squares(F, [1.0, 1.0, 1.0], bounds=(-math.inf, math.inf))
    assert a.x.tolist() == b.x.tolist() and a.nfev == b.nfev
    # x0 on a bound: moved inside as scipy does (trf's strictly feasible iterates)
    seen = []
    sd.optimize.least_squares(lambda b: seen.append(b.tolist()) or F(b), [1.0, 0.0, 1.0], bounds=(0, 10))
    assert seen[0] == [1.0, 1e-10, 1.0]


def test_fewer_residuals_than_variables_trf():
    # m < n: the trust region reflective method still steps (scipy too); 'lm' refuses
    F = lambda x: sd.asarray([x[0] + 2 * x[1] - x[2] - 1.0, x[0] * x[1] - 0.5])  # noqa: E731
    Fn = lambda x: np.array([x[0] + 2 * x[1] - x[2] - 1.0, x[0] * x[1] - 0.5])  # noqa: E731
    r = sd.optimize.least_squares(F, [1.0, 1.0, 1.0])
    q = scipy_ls(Fn, np.array([1.0, 1.0, 1.0]))
    assert r.converged and r.status == q.status and r.cost <= 1e-15
    assert dist(r.x.tolist(), q.x, [1.0]) <= 1e-6
    with pytest.raises(
        ValueError,
        match=r"least_squares\(\): method='lm' needs at least as many residuals as "
        r"variables, not 2 < 3",
    ):
        sd.optimize.least_squares(F, [1.0, 1.0, 1.0], method="lm")


def test_non_finite_cost_is_not_converged():
    # O20: the stopping test passes (J = 0, so the gradient is 0) but the cost overflows
    F = lambda x: sd.asarray([1e200, 1e200]) + 0.0 * x[0]  # noqa: E731
    for method in ("trf", "lm"):
        r = sd.optimize.least_squares(F, [1.0], jac=lambda x: sd.zeros((2, 1)), method=method)
        assert r.status == 1 and r.message == "`gtol` termination condition is satisfied."
        assert math.isinf(r.cost) and not r.converged


def test_non_finite_trial_points_are_rejected():
    # as nls (regression G74): a trial point where fun is not finite shrinks the trust region
    t = np.linspace(0, 5, 40)
    y = 3.0 * np.exp(-0.7 * t) + 0.5
    ts, ys = sd.asarray(t), sd.asarray(y)
    F = lambda b: sd.where(b[1] > 3.0, math.nan, b[0] * sd.exp(-b[1] * ts) + b[2] - ys)  # noqa: E731
    for method in ("trf", "lm"):
        r = sd.optimize.least_squares(F, [1.0, 2.9, 1.0], method=method)
        assert r.converged and dist(r.x.tolist(), [3.0, 0.7, 0.5], [1.0]) <= 1e-6


def test_same_bits_with_1_and_10_threads():
    rng = np.random.default_rng(0)
    X = sd.asarray(rng.standard_normal((20000, 30)) / math.sqrt(30))
    y = sd.exp(sd.matmul(X, sd.full((30,), 0.3))) + sd.asarray(0.05 * rng.standard_normal(20000))
    F = lambda b: y - sd.exp(sd.matmul(X, b))  # noqa: E731
    J = lambda b: -sd.reshape(sd.exp(sd.matmul(X, b)), (20000, 1)) * X  # noqa: E731
    old = sd.get_num_threads()
    try:
        out = []
        for t in (1, 10):
            sd.set_num_threads(t)
            out.append([sd.optimize.least_squares(F, [0.0] * 30, jac=J, method=m).x.tolist() for m in ("trf", "lm")])
        assert out[0] == out[1]
    finally:
        sd.set_num_threads(old)


# ---- the result


def test_result_fields():
    c = mgh.BY_ID["mgh17-n5-m33-x1"]
    F, J = sd_fns(c)
    for method in ("trf", "lm"):
        r = sd.optimize.least_squares(F, c.x0(), jac=J, method=method)
        assert r.method == method
        for k in (
            "x",
            "cost",
            "fun",
            "jac",
            "grad",
            "optimality",
            "active_mask",
            "converged",
            "status",
            "message",
            "n_iter",
            "nfev",
            "njev",
            "method",
        ):
            assert hasattr(r, k), k
        for k in ("hess", "hess_inv", "nhev", "bracket"):
            assert not hasattr(r, k), k
        x = r.x
        assert x.shape == (5,) and x.dtype == sd.float64
        assert r.fun.shape == (33,) and r.fun.tolist() == F(x).tolist()
        assert r.jac.shape == (33, 5) and r.jac.tolist() == J(x).tolist()
        assert r.cost == 0.5 * float(sd.linalg.vecdot(r.fun, r.fun))
        assert r.grad.tolist() == sd.matmul(r.fun, r.jac).tolist()
        assert r.optimality == max(abs(v) for v in r.grad.tolist())
        assert r.active_mask.dtype == sd.int64 and r.active_mask.tolist() == [0] * 5
        assert isinstance(r.converged, bool) and isinstance(r.status, int) and isinstance(r.n_iter, int)
        assert repr(r).startswith(f"OptimizeResult(method='{method}', converged=True, cost=")
        with pytest.raises(AttributeError):
            r.x = x
        with pytest.raises(ValueError, match="read-only"):  # D-4: read-only, no copy
            r.x[0] = 99.0
        assert r.x is x


def test_fun_gets_a_new_array_and_results_may_be_any_sequence():
    seen = []

    def F(x):
        seen.append(x)
        out = [x[0] - 1.0, 10.0 * (x[1] - x[0] ** 2)]
        x[0] = 1e9  # changing it does not change the iteration
        return out

    r = sd.optimize.least_squares(
        F, np.array([-1.2, 1.0]), jac=lambda x: np.array([[1.0, 0.0], [-20.0 * x[0], 10.0]]), method="lm"
    )
    assert r.converged and dist(r.x.tolist(), [1.0, 1.0], [1.0]) <= 1e-12
    assert all(isinstance(x, sd.ndarray) and x.dtype == sd.float64 and x.shape == (2,) for x in seen)
    assert len({id(x) for x in seen}) == len(seen)
    # one variable: jac may return (m,)
    r = sd.optimize.least_squares(lambda x: x * x - 2.0, [1.0], jac=lambda x: 2.0 * x)
    assert abs(float(r.x[0]) - math.sqrt(2.0)) <= 1e-10


def test_errors():
    F = lambda x: x - 1.0  # noqa: E731
    cases = [
        (lambda: sd.optimize.least_squares(3, [1.0]), TypeError, "least_squares(): 'fun' must be callable, not int"),
        (
            lambda: sd.optimize.least_squares(F, [[1.0, 2.0], [3.0, 4.0]]),
            ValueError,
            "least_squares(): 'x0' must be 1-dimensional, not shape (2, 2)",
        ),
        (lambda: sd.optimize.least_squares(F, []), ValueError, "least_squares(): 'x0' is empty (no variables)"),
        (lambda: sd.optimize.least_squares(F, [math.nan]), ValueError, "least_squares(): 'x0' must be finite"),
        (
            lambda: sd.optimize.least_squares(F, [1j]),
            TypeError,
            "least_squares(): 'x0' must be real, not a complex128 array",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], method="TRF"),
            ValueError,
            "least_squares(): 'method' must be lowercase: 'trf', not 'TRF'",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], method="dogbox"),
            ValueError,
            "least_squares(): 'method' must be 'trf' or 'lm', not 'dogbox'",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], jac="cs"),
            ValueError,
            "least_squares(): 'jac' must be '2-point', '3-point' or callable, not 'cs'",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], jac=None),
            TypeError,
            "least_squares(): 'jac' must be '2-point', '3-point' or callable, not NoneType",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], method="lm", bounds=(0, 2)),
            ValueError,
            "least_squares(): method='lm' takes no 'bounds' (method='trf' does)",
        ),
        (
            lambda: sd.optimize.least_squares(F, [3.0], bounds=(0, 2)),
            ValueError,
            "least_squares(): 'x0' must be within 'bounds'",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], bounds=(0, 2, 3)),
            ValueError,
            "least_squares(): 'bounds' must be a pair (lower, upper), not 3 values",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], bounds=([0, 0], 2)),
            ValueError,
            "least_squares(): 'bounds' must have 1 value like 'x0', not 2",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], bounds=[(0, 2)]),
            ValueError,
            "least_squares(): 'bounds' must be a pair (lower, upper), not a sequence of (low, high) pairs",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], bounds=(2, 2)),
            ValueError,
            "least_squares(): each lower bound must be less than its upper bound",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], bounds=3),
            TypeError,
            "least_squares(): 'bounds' must be None or a pair (lower, upper), not int",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], ftol=0, xtol=0, gtol=1e-17),
            ValueError,
            "least_squares(): at least one of 'ftol', 'xtol' and 'gtol' must be >= 2.2e-16 (the machine epsilon)",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], method="lm", gtol=0.0),
            ValueError,
            "least_squares(): 'ftol', 'xtol' and 'gtol' must each be >= 2.2e-16 (the machine epsilon) when method='lm'",
        ),
        (
            # a large point is shown by its first values and its size (LSQ-11)
            lambda: sd.optimize.least_squares(lambda x: x, [1.0] * 12, jac=lambda x: sd.full((12, 12), math.nan)),
            ValueError,
            "least_squares(): 'jac' is not finite at x = [1.0, 1.0, 1.0, 1.0, 1.0, ...] (12 values)",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], foo=1),
            TypeError,
            "least_squares(): unexpected keyword argument 'foo'",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], ftol=-1.0),
            ValueError,
            "least_squares(): 'ftol' must be a finite number >= 0",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], max_nfev=0),
            ValueError,
            "least_squares(): 'max_nfev' must be >= 1",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], max_nfev=1.5),
            TypeError,
            "least_squares(): 'max_nfev' must be an int, not float",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], diff_step=0.0),
            ValueError,
            "least_squares(): 'diff_step' must be finite and > 0",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], jac=lambda x: sd.ones((1, 1)), diff_step=1e-6),
            ValueError,
            "least_squares(): 'diff_step' is only used when 'jac' is '2-point' or '3-point'",
        ),
        (
            lambda: sd.optimize.least_squares(lambda x: sd.ones((3, 2)), [1.0]),
            ValueError,
            "least_squares(): 'fun' must return a 1-dimensional array, not shape (3, 2)",
        ),
        (
            lambda: sd.optimize.least_squares(lambda x: 1.0, [1.0]),
            ValueError,
            "least_squares(): 'fun' must return a 1-dimensional array, not a scalar",
        ),
        (
            lambda: sd.optimize.least_squares(lambda x: "x", [1.0]),
            TypeError,
            "least_squares(): 'fun' must return an sd.ndarray or a sequence of numbers, not str",
        ),
        (
            lambda: sd.optimize.least_squares(lambda x: x * 1j, [1.0]),
            TypeError,
            "least_squares(): 'fun' must return real values, not a complex128 array",
        ),
        (
            lambda: sd.optimize.least_squares(lambda x: sd.zeros((0,)), [1.0]),
            ValueError,
            "least_squares(): 'fun' must return at least 1 value",
        ),
        (
            lambda: sd.optimize.least_squares(lambda x: x * math.inf, [1.0]),
            ValueError,
            "least_squares(): 'fun' is not finite at 'x0'",
        ),
        (
            lambda: sd.optimize.least_squares(
                lambda x: sd.ones((40,)) * x[0], [1.0, 2.0, 3.0], jac=lambda x: sd.ones((40, 2))
            ),
            ValueError,
            "least_squares(): 'jac' must return shape (40, 3), not (40, 2)",
        ),
        (
            lambda: sd.optimize.least_squares(lambda x: sd.ones((2 if x[0] == 1.0 else 3,)) * x[0], [1.0]),
            ValueError,
            "least_squares(): 'fun' must return 2 values as at 'x0', not 3",
        ),
        (
            lambda: sd.optimize.least_squares(F, [1.0], jac=lambda x: sd.asarray([[math.nan]])),
            ValueError,
            "least_squares(): 'jac' is not finite at x = [1.0]",
        ),
    ]
    for fn, typ, msg in cases:
        with pytest.raises(typ) as e:
            fn()
        assert str(e.value) == msg


def test_user_errors_pass_through():
    def F(x):
        raise KeyError("mine")

    with pytest.raises(KeyError, match="mine"):
        sd.optimize.least_squares(F, [1.0])
    calls = [0]

    def G(x):
        calls[0] += 1
        if calls[0] > 2:
            raise ValueError("later")
        return x - 1.0

    with pytest.raises(ValueError, match="^later$"):
        sd.optimize.least_squares(G, [3.0])


# ---- trf: an overflowing sum of squares (LSQ-2), fewer residuals than variables (LSQ-3);
# 'lm' on tall problems (BP-3); the int64 caps (LSQ-9)


def test_trf_overflowing_cost_stops():
    # |f| near 1e160: the sum of squares and the gradient overflow; scipy goes on without progress
    # to max_nfev and stops with status 0; here at once, with no LAPACK error
    for F, x0 in (
        (lambda x: sd.asarray([1e160 * x[0], x[1] - 1.0]), [1.0, 0.0]),
        (lambda x: sd.asarray([1e200 * (x[0] - 1.0), x[1]]), [0.0, 0.0]),
    ):
        r = sd.optimize.least_squares(F, x0)
        assert r.status == 0 and not r.converged and r.message.startswith("The maximum number")
    r = sd.optimize.least_squares(lambda x: sd.asarray([1e160 * x[0], x[1] - 1.0]), [1.0, 0.0], method="lm")
    assert r.converged and r.x.tolist() == [0.0, 1.0]


def test_trf_wide_problems():
    # m < n: scipy's SVD of J_h (m x n), or of [J_h; diag(dh)^1/2] with bounds, not an n x n
    # problem; the same steps as scipy's to rounding
    rng = np.random.default_rng(1)
    for m, n, bounded in ((5, 300, False), (3, 40, True), (1, 8, False)):
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        As, bs = sd.asarray(A.tolist()), sd.asarray(b.tolist())
        kw = {"bounds": (-1.0, 1.0)} if bounded else {}
        r = sd.optimize.least_squares(lambda x: As @ (x + 0.1 * x**3) - bs, [0.1] * n, **kw)
        q = scipy_ls(lambda x: A @ (x + 0.1 * x**3) - b, np.full(n, 0.1), **kw)
        assert r.converged and q.success and r.cost <= 1e-12 + q.cost, (m, n)
        if not bounded:
            assert r.status == q.status and abs(r.njev - q.njev) <= 2, (m, n, r.njev, q.njev)


def test_lm_tall_problems_take_the_qr_first():
    # BP-3: from m >= 10 n and m n >= 2000, lmder after one LAPACK QR of J (nls's build): scipy's
    # steps to rounding
    rng = np.random.default_rng(2)
    for m, n in ((400, 5), (3000, 20)):
        A = rng.standard_normal((m, n))
        b = A @ rng.standard_normal(n) + 0.01 * rng.standard_normal(m)
        As, bs = sd.asarray(A.tolist()), sd.asarray(b.tolist())
        r = sd.optimize.least_squares(lambda x: As @ (x + 0.1 * x**3) - bs, [0.0] * n, method="lm")
        q = scipy_ls(lambda x: A @ (x + 0.1 * x**3) - b, np.zeros(n), method="lm")
        # (nfev counts the difference calls here, n per Jacobian; scipy's 'lm' does not)
        assert r.status == q.status and abs(r.nfev - n * r.njev - q.nfev) <= 2, (m, n)
        assert max(abs(u - v) for u, v in zip(r.x.tolist(), q.x)) <= 1e-10 * max(1.0, max(abs(q.x)))


def test_caps_are_int64():
    # LSQ-9: the default max_nfev can pass 2^31 for large n; a value given may too
    r = sd.optimize.least_squares(lambda x: x - 1.0, [0.0, 0.0], max_nfev=2**40)
    assert r.converged
    with pytest.raises(OverflowError, match=r"least_squares\(\): 'max_nfev' is out of range"):
        sd.optimize.least_squares(lambda x: x - 1.0, [0.0], max_nfev=2**63)
