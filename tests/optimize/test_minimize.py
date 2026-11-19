"""sd.optimize.minimize (docs/api/optimize.md §7, docs/design/optimize.md §3.5, §7): scipy 1.18.1 on
the Moré-Garbow-Hillstrom collection as minimization of the sum of squares and on scipy's test
functions (the recorded runs, tests/optimize/data/scipy_minimize.json, and every call point live, bit
for bit), the options and the edge cases live, hess_inv, L-BFGS-B 3.0's Fortran original
(fortran_lbfgsb.json), the decisions O50-O57, the result and the errors."""

import json
import math
import pathlib
import warnings

import numpy as np
import pytest
import scipy
import scipy.optimize as so

import stardust as sd
from stardust.optimize._result import _Lazy

from . import functions as fn
from . import mgh

DATA = pathlib.Path(__file__).parent / "data"
REF = json.loads((DATA / "scipy_minimize.json").read_text())
FORTRAN = json.loads((DATA / "fortran_lbfgsb.json").read_text())
F_VS_S = {r["id"]: r for r in json.loads((DATA / "fortran_vs_scipy.json").read_text())["lbfgsb"]}
EPS = 2.0**-52
LIVE = pytest.mark.skipif(scipy.__version__ != "1.18.1", reason="the steps are scipy 1.18.1's")
DRIVER1_BOUNDS = ([b[0] for b in fn.LBFGSB_DRIVER1_BOUNDS], [b[1] for b in fn.LBFGSB_DRIVER1_BOUNDS])


def problems():
    """(id, f, g, x0, bounds) of the recorded problems: f and g take and return numpy values (the
    same code for both libraries)."""
    out = []
    for c in mgh.CASES:
        _, _, f, g = mgh.as_numpy(c)
        out.append((c.id, f, g, c.x0(), None))

    def np_f(h):
        return lambda x: h([np.float64(v) for v in x])

    def np_g(h):
        return lambda x: np.array(h([np.float64(v) for v in x]))

    for n in (2, 5):
        out.append((f"rosen-n{n}", np_f(fn.rosen), np_g(fn.rosen_der), fn.ROSEN_X0[n], None))
    out.append(("maxent", np_f(fn.maxent), np_g(fn.maxent_der), fn.MAXENT_X0, None))
    out.append(
        ("lbfgsb_driver1", np_f(fn.lbfgsb_driver1), np_g(fn.lbfgsb_driver1_der), fn.LBFGSB_DRIVER1_X0, DRIVER1_BOUNDS)
    )
    return out


PROBLEMS = problems()


def box(x0):
    return [v - 0.5 * (abs(v) + 1.0) for v in x0], [v + 0.5 * (abs(v) + 1.0) for v in x0]


def sd_side(f, g, log=None):
    """sd's callbacks: x (an sd.ndarray) as numpy, the call logged as (kind, x)."""

    def F(x):
        xl = x.tolist()
        if log is not None:
            log.append(("f", tuple(xl)))
        return f(np.array(xl))

    def G(x):
        xl = x.tolist()
        if log is not None:
            log.append(("g", tuple(xl)))
        return g(np.array(xl))

    return F, G


def scipy_side(f, g, log):
    def F(x):
        log.append(("f", tuple(float(v) for v in x)))
        return f(x)

    def G(x):
        log.append(("g", tuple(float(v) for v in x)))
        return g(x)

    return F, G


def run_sd(method, f, g, x0, bounds, fd, log=None, **kw):
    F, G = sd_side(f, g, log)
    if method != "nelder-mead" and not fd:
        kw["jac"] = G
    if bounds is not None:
        kw["bounds"] = bounds
    with np.errstate(all="ignore"):
        return sd.optimize.minimize(F, x0, method=method, **kw)


def run_scipy(method, f, g, x0, bounds, fd, log, **kw):
    F, G = scipy_side(f, g, log)
    if method != "Nelder-Mead" and not fd:
        kw["jac"] = G
    if bounds is not None:
        kw["bounds"] = list(zip(*bounds))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return so.minimize(F, np.array(x0, dtype=float), method=method, **kw)


# ---- scipy 1.18.1, recorded (O9): the same steps bit for bit, so every field is scipy's


RECORDED = ("Nelder-Mead", "BFGS", "BFGS/fd", "L-BFGS-B", "L-BFGS-B/fd")


@pytest.mark.parametrize("group", RECORDED)
def test_against_recorded_scipy(group):
    method, _, fd = group.partition("/")
    for pid, f, g, x0, bounds in PROBLEMS:
        rec = REF[pid].get(group)
        if rec is None:  # BFGS takes no bounds (driver1)
            continue
        r = run_sd(method.lower(), f, g, x0, bounds if method != "BFGS" else None, bool(fd))
        assert r.x.tolist() == rec["x"], pid
        # O59: after ABNORMAL, fun is the value at x (scipy's is the last trial's)
        assert r.fun == (f(np.array(rec["x"])) if rec["message"].startswith("ABNORMAL") else rec["fun"]), pid
        assert (r.nfev, r.n_iter, r.status, r.message) == (rec["nfev"], rec["nit"], rec["status"], rec["message"]), pid
        assert r.converged == rec["success"], pid
        if "njev" in rec:
            assert r.njev == rec["njev"], pid


# ---- live: the points where fun and jac are called, in order, bit for bit (and all fields)

LIVE_GROUPS = {
    "nelder-mead": ("Nelder-Mead", False, False),
    "nelder-mead/box": ("Nelder-Mead", False, True),
    "bfgs": ("BFGS", False, False),
    "bfgs/fd": ("BFGS", True, False),
    "l-bfgs-b": ("L-BFGS-B", False, False),
    "l-bfgs-b/fd": ("L-BFGS-B", True, False),
    "l-bfgs-b/box": ("L-BFGS-B", False, True),
    "l-bfgs-b/fd/box": ("L-BFGS-B", True, True),
}


def same_run(a, b, f):
    """scipy's result a and sd's b agree in every field; after L-BFGS-B's ABNORMAL, fun is the value
    at x (O59; scipy's is the last trial's)."""
    assert b.x.tolist() == [float(v) for v in a.x]
    want = f(np.array(a.x)) if a.message.startswith("ABNORMAL") else a.fun
    assert b.fun == want or (math.isnan(want) and math.isnan(b.fun))
    assert (b.nfev, b.n_iter, b.status, b.message, b.converged) == (
        a.nfev,
        a.get("nit", 0),
        a.get("status", 0),
        a.message,
        bool(a.success),
    )
    if "njev" in a:
        assert b.njev == a.njev


@LIVE
@pytest.mark.parametrize("group", sorted(LIVE_GROUPS))
def test_every_call_point_follows_scipy(group):
    method, fd, bounded = LIVE_GROUPS[group]
    n = 0
    for pid, f, g, x0, bounds in PROBLEMS:
        if method == "BFGS" and bounds is not None:
            continue
        if bounded and bounds is None:
            if not pid.endswith("-x1") and pid.startswith("mgh"):
                continue
            bounds = box(x0)
        if not bounded and pid == "lbfgsb_driver1" and method == "Nelder-Mead":
            pass
        la, lb = [], []
        a = run_scipy(method, f, g, x0, bounds, fd, la)
        b = run_sd(method.lower(), f, g, x0, bounds, fd, lb)
        assert la == lb, pid
        same_run(a, b, f)
        n += 1
    assert n >= 51


# ---- live: options and edge cases (non-finite values on the way, fixed variables, the caps)

SWEEP_IDS = ("mgh01-n2-m2-x1", "mgh07-n3-m3-x10", "mgh20-n9-m31-x1", "mgh26-n10-m10-x1")


def _nonfinite(f, g, cut, val):
    return (lambda x: val if x[0] > cut else f(x)), (lambda x: np.full(len(x), val) if x[0] > cut else g(x))


def sweep_cases():
    out = []
    for pid in SWEEP_IDS:
        c = mgh.BY_ID[pid]
        _, _, f, g = mgh.as_numpy(c)
        x0 = c.x0()
        n = len(x0)
        lo, hi = box(x0)
        lo1, hi1 = list(lo), list(hi)
        lo1[0] = hi1[0] = x0[0]
        lo2, hi2 = list(lo), list(hi)
        lo2[-1], hi2[0] = -math.inf, math.inf
        sim = [list(x0)] + [[v + (0.1 * (k + 1) if j == k else 0.0) for j, v in enumerate(x0)] for k in range(n)]
        H0 = [[0.5 * (i + 1) if i == j else 0.0 for j in range(n)] for i in range(n)]
        cut = x0[0] + 0.3 * (abs(x0[0]) + 1)
        rows = [
            ("bfgs", "true", {}, {}),
            ("bfgs", "2-point", {}, {}),
            ("bfgs", "3-point", {}, {}),
            ("bfgs", None, {"eps": 1e-6}, {}),
            ("bfgs", "2-point", {"finite_diff_rel_step": 1e-5}, {}),
            ("bfgs", "g", {"norm": 2.0, "xrtol": 1e-6}, {}),
            ("bfgs", "g", {"norm": -math.inf}, {}),
            ("bfgs", "g", {"max_iter": 3}, {}),
            ("bfgs", "g", {"c1": 0.3, "c2": 0.4}, {}),
            ("bfgs", "g", {}, {"tol": 1e-9}),
            ("bfgs", "g", {"hess_inv0": H0}, {}),
            ("l-bfgs-b", "g", {"maxcor": 3}, {}),
            ("l-bfgs-b", "g", {"max_ls": 1}, {}),
            ("l-bfgs-b", "g", {"max_ls": 2}, {"bounds": (lo, hi)}),
            ("l-bfgs-b", "g", {"max_iter": 4}, {}),
            ("l-bfgs-b", "g", {"max_nfev": 7}, {}),
            ("l-bfgs-b", "g", {}, {"tol": 1e-10}),
            ("l-bfgs-b", "true", {}, {"bounds": (lo, hi)}),
            ("l-bfgs-b", "2-point", {}, {"bounds": (lo, hi)}),
            ("l-bfgs-b", "3-point", {}, {"bounds": (lo, hi)}),
            ("l-bfgs-b", None, {"eps": 1e-5}, {"bounds": (lo, hi)}),
            ("l-bfgs-b", None, {}, {"bounds": (lo1, hi1)}),
            ("l-bfgs-b", "g", {}, {"bounds": (lo1, hi1)}),
            ("l-bfgs-b", "g", {}, {"bounds": (list(x0), list(x0))}),
            ("l-bfgs-b", "g", {}, {"bounds": (lo2, hi2)}),
            ("nelder-mead", None, {"adaptive": True}, {}),
            ("nelder-mead", None, {"initial_simplex": sim}, {}),
            ("nelder-mead", None, {"max_iter": 30}, {}),
            ("nelder-mead", None, {"max_nfev": 37}, {}),
            ("nelder-mead", None, {"max_nfev": 300, "max_iter": 50}, {}),
            ("nelder-mead", None, {}, {"tol": 1e-8, "bounds": (lo, hi)}),
        ]
        for val in (math.inf, math.nan):
            ff, gg = _nonfinite(f, g, cut, val)
            for method, jac, kw in (
                ("bfgs", "g", {}),
                ("bfgs", None, {}),
                ("l-bfgs-b", "g", {}),
                ("l-bfgs-b", "g", {"bounds": (lo, hi)}),
                ("nelder-mead", None, {}),
            ):
                out.append((f"{pid} {method} {jac} {val}", method, ff, gg, x0, jac, {}, kw))
        for method, jac, opts, kw in rows:
            out.append((f"{pid} {method} {jac} {sorted(opts)} {sorted(kw)}", method, f, g, x0, jac, opts, kw))
    return out


SCIPY_NAMES = {"max_iter": "maxiter", "max_nfev": None, "max_ls": "maxls"}


def scipy_options(method, opts):
    out = {}
    for k, v in opts.items():
        if k == "max_nfev":
            k = "maxfev" if method == "nelder-mead" else "maxfun"
        elif k in SCIPY_NAMES:
            k = SCIPY_NAMES[k]
        out[k] = np.array(v) if k in ("initial_simplex", "hess_inv0") else v
    return out


@LIVE
@pytest.mark.parametrize("case", sweep_cases(), ids=lambda c: c[0])
def test_options_and_edge_cases_follow_scipy(case):
    _, method, f, g, x0, jac, opts, kw = case
    la, lb = [], []
    Fa, Ga = scipy_side(f, g, la)
    Fb, Gb = sd_side(f, g, lb)
    skw = {k: v for k, v in kw.items() if k != "bounds"}
    dkw = dict(skw)
    if "bounds" in kw:
        skw["bounds"] = list(zip(*kw["bounds"]))
        dkw["bounds"] = kw["bounds"]
    if jac == "g":
        skw["jac"], dkw["jac"] = Ga, Gb
    elif jac == "true":
        skw["jac"] = dkw["jac"] = True
        Fa0, Fb0 = Fa, Fb
        Fa = lambda x: (Fa0(x), g(x))  # noqa: E731 (the pair counts as one call)
        Fb = lambda x: (Fb0(x), g(np.array(x.tolist())))  # noqa: E731
    elif jac is not None:
        skw["jac"] = dkw["jac"] = jac
    dopts = {k: (sd.asarray(v) if k in ("initial_simplex", "hess_inv0") else v) for k, v in opts.items()}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = so.minimize(Fa, np.array(x0, dtype=float), method=method, options=scipy_options(method, opts), **skw)
        b = sd.optimize.minimize(Fb, x0, method=method, options=dopts, **dkw)
    assert la == lb
    same_run(a, b, f)


# ---- hess_inv (O55)


@LIVE
def test_hess_inv_is_scipys():
    # BFGS: the same matrix bit for bit (scipy's products, by the same BLAS). L-BFGS-B: the dense
    # matrix of scipy's LbfgsInvHessProduct(...).todense(), the pairs in scipy's storage order
    # (also when the memory has wrapped around), to rounding
    worst = 0.0
    wrapped = 0
    for c in mgh.CASES[::3]:
        _, _, f, g = mgh.as_numpy(c)
        F, G = sd_side(f, g)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = so.minimize(f, np.array(c.x0()), jac=g, method="BFGS")
            b = sd.optimize.minimize(F, c.x0(), jac=G, method="bfgs")
            assert np.array(b.hess_inv.tolist()).tolist() == a.hess_inv.tolist(), c.id
            for maxcor in (10, 3):
                a = so.minimize(f, np.array(c.x0()), jac=g, method="L-BFGS-B", options={"maxcor": maxcor})
                b = sd.optimize.minimize(F, c.x0(), jac=G, method="l-bfgs-b", options={"maxcor": maxcor})
                A, B = a.hess_inv.todense(), np.array(b.hess_inv.tolist())
                wrapped += a.hess_inv.n_corrs == maxcor and a.nit > maxcor
                worst = max(worst, float(np.max(np.abs(A - B)) / max(np.max(np.abs(A)), 1e-300)))
    assert worst <= 1e-12 and wrapped >= 20


def test_lbfgsb_hess_inv_is_dense_and_lazy():
    calls = []
    r = sd.optimize.minimize(lambda x: float(sd.sum((x - 1.0) ** 2)), [0.0, 3.0, -1.0], method="l-bfgs-b")
    calls.append(r._fields["hess_inv"])
    assert isinstance(calls[0], _Lazy)  # not computed yet
    H = r.hess_inv
    assert isinstance(H, sd.ndarray) and H.shape == (3, 3) and H.dtype == sd.float64
    assert not isinstance(r._fields["hess_inv"], _Lazy)  # computed once, kept
    H[0, 0] = 99.0
    assert r.hess_inv[0, 0] != 99.0  # a new copy each read (G113)
    b = sd.optimize.minimize(lambda x: float(sd.sum((x - 1.0) ** 2)), [0.0, 3.0, -1.0], method="bfgs")
    assert b.hess_inv.shape == (3, 3)


# ---- L-BFGS-B 3.0's Fortran original (O13): the same differences from it as scipy's


def _fortran_runs():
    from tests.optimize.mgh import MINPACK_LSQ_CASES

    runs = []
    for c in MINPACK_LSQ_CASES:
        runs.append((c.id, c, None))
        if c.factor == 1.0:
            runs.append((c.id + "-box", c, box(c.x0())))
    for m in (5, 10):
        runs.append((f"driver1-m{m}", None, DRIVER1_BOUNDS))
    return runs


def _fg_minpack(c):
    def fg(x):
        xl = [np.float64(v) for v in x.tolist()]
        r, jac = c.fun(xl), c.jac(xl)
        f = 0.0
        for v in r:
            f = f + v * v
        g = []
        for j in range(c.n):
            s = 0.0
            for i in range(c.m):
                s = s + jac[i][j] * r[i]
            g.append(2.0 * s)
        return f, np.array(g)

    return fg


def test_lbfgsb_against_the_fortran_original():
    # gen_optimize_fortran.py's runs (setulb with scipy's driver loop, f = sum F^2 of MINPACK's 18
    # problems with and without a box, driver1 with m = 5, 10): sd's run is scipy's recorded one
    # (every x bit, the counts, the message); against the Fortran original that is the same
    # agreement as scipy's (docs/design/optimize.md §7): the counts in 74 of 83 runs, x within
    # 1e-12 in 62, 1e-6 in 79, the same stopping test in 82
    same_counts = x12 = x6 = stop = 0
    for rid, c, bounds in _fortran_runs():
        if c is None:
            fg = lambda x: (  # noqa: E731
                fn.lbfgsb_driver1([np.float64(v) for v in x.tolist()]),
                np.array(fn.lbfgsb_driver1_der([np.float64(v) for v in x.tolist()])),
            )
            x0, m, ftol = fn.LBFGSB_DRIVER1_X0, int(rid.split("-m")[1]), 1e7 * EPS
        else:
            fg, x0, m, ftol = _fg_minpack(c), c.x0(), 10, 2.220446049250313e-09
        opts = {"maxcor": m, "ftol": ftol, "gtol": 1e-5, "max_iter": 15000, "max_nfev": 15000, "max_ls": 20}
        with np.errstate(all="ignore"):
            r = sd.optimize.minimize(fg, x0, jac=True, method="l-bfgs-b", bounds=bounds, options=opts)
        s, fo = F_VS_S[rid]["scipy"], FORTRAN[rid]
        assert (r.x.tolist(), r.nfev, r.n_iter, r.message) == (s["x"], s["nfev"], s["n_iter"], s["message"]), rid
        same_counts += (r.nfev, r.n_iter) == (fo["nfev"], fo["n_iter"])
        d = max(abs(a - b) for a, b in zip(fo["x"], r.x.tolist())) / max(max(abs(v) for v in fo["x"]), 1e-300)
        x12 += d <= 1e-12
        x6 += d <= 1e-6
        # the original's messages are written with underscores (REL_REDUCTION_OF_F_<=_FACTR*EPSMCH)
        stop += ("RELATIVE" in r.message) == ("REL_REDUCTION" in fo["task"])
    assert (same_counts, x12, x6, stop) == (74, 62, 79, 82)


def test_lbfgsb_driver1_m5_is_the_drivers_run():
    # L-BFGS-B 3.0's driver1 (m = 5, factr 1e7, pgtol 1e-5): 23 iterations, 28 evaluations, f = 1.083490083e-9
    fg = lambda x: (  # noqa: E731
        fn.lbfgsb_driver1(x.tolist()),
        sd.asarray(fn.lbfgsb_driver1_der(x.tolist())),
    )
    r = sd.optimize.minimize(
        fg, fn.LBFGSB_DRIVER1_X0, jac=True, bounds=DRIVER1_BOUNDS, options={"maxcor": 5, "ftol": 1e7 * EPS}
    )
    assert r.method == "l-bfgs-b" and r.converged
    assert (r.n_iter, r.nfev) == (23, 28)
    assert abs(r.fun - 1.0834900834300615e-09) <= 1e-17


# ---- the decisions


def quad(x):
    return float(sd.sum((x - 1.0) ** 2))


def quad_g(x):
    return 2.0 * (x - 1.0)


def test_default_method_follows_scipy():
    # O50: 'bfgs' without bounds, 'l-bfgs-b' with them
    assert sd.optimize.minimize(quad, [3.0, -2.0]).method == "bfgs"
    assert sd.optimize.minimize(quad, [3.0, -2.0], bounds=(-5.0, 5.0)).method == "l-bfgs-b"


def test_bounds_take_one_form():
    # O51: (lower, upper), each a number or n numbers, +-inf for none; for l-bfgs-b and nelder-mead
    r = sd.optimize.minimize(quad, [0.0, 3.0], bounds=([-1.0, 2.0], math.inf), method="l-bfgs-b")
    assert r.x[1] == 2.0 and abs(r.x[0] - 1.0) < 1e-7
    r = sd.optimize.minimize(quad, [0.0, 0.0], bounds=(-math.inf, [0.5, 0.25]), method="nelder-mead")
    assert max(abs(a - b) for a, b in zip(r.x.tolist(), [0.5, 0.25])) < 1e-4
    with pytest.raises(ValueError, match=r"'bounds' must be a pair \(lower, upper\), not 3 values"):
        sd.optimize.minimize(quad, [0.0, 0.0, 0.0], bounds=[(0, 1), (0, 1), (0, 1)])
    with pytest.raises(ValueError, match=r"'bounds' must have 2 values like 'x0', not 3"):
        sd.optimize.minimize(quad, [0.0, 0.0], bounds=([0.0, 0.0, 0.0], 1.0))
    with pytest.raises(TypeError, match="'bounds' must be None or a pair"):
        sd.optimize.minimize(quad, [0.0, 0.0], bounds=so.Bounds(0, 1))
    with pytest.raises(ValueError, match="each lower bound must not be greater than its upper bound"):
        sd.optimize.minimize(quad, [0.0, 0.0], bounds=(1.0, 0.0))
    with pytest.raises(ValueError, match="'bounds' contains NaN"):
        sd.optimize.minimize(quad, [0.0, 0.0], bounds=(math.nan, 1.0))
    with pytest.raises(ValueError, match=r"method='bfgs' takes no bounds"):
        sd.optimize.minimize(quad, [0.0, 0.0], method="bfgs", bounds=(0.0, 1.0))


def test_x0_outside_the_bounds_is_an_error():
    # O52: not scipy's silent clipping
    for method in ("l-bfgs-b", "nelder-mead"):
        with pytest.raises(ValueError, match=r"minimize\(\): 'x0' must be within 'bounds'"):
            sd.optimize.minimize(quad, [2.0, 0.0], method=method, bounds=(-1.0, 1.0))
    r = sd.optimize.minimize(quad, [1.0, -1.0], bounds=(-1.0, 1.0))  # on the bounds is within
    assert r.converged


def test_non_finite_at_x0_is_an_error():
    # O53: at x0 a ValueError; on the way scipy's handling (the live sweep above)
    inf = lambda x: math.inf  # noqa: E731
    for method in ("nelder-mead", "bfgs", "l-bfgs-b"):
        with pytest.raises(ValueError, match=r"minimize\(\): 'fun' is not finite at 'x0'"):
            sd.optimize.minimize(inf, [1.0, 2.0], method=method)
    with pytest.raises(ValueError, match="'jac' is not finite at 'x0'"):
        sd.optimize.minimize(quad, [1.0, 2.0], jac=lambda x: sd.asarray([math.nan, 0.0]))
    with pytest.raises(ValueError, match="'jac' is not finite at 'x0'"):
        sd.optimize.minimize(lambda x: (quad(x), sd.asarray([math.inf, 0.0])), [1.0, 2.0], jac=True, method="l-bfgs-b")
    steep = lambda x: quad(x) if x[0] <= 1.0 else math.inf  # noqa: E731
    with pytest.raises(ValueError, match="the differences of 'fun' are not finite at 'x0'"):
        sd.optimize.minimize(steep, [1.0, 2.0])
    with pytest.raises(ValueError, match="first vertex of 'initial_simplex'"):
        sd.optimize.minimize(inf, [1.0], method="nelder-mead", options={"initial_simplex": [[0.0], [1.0]]})


def test_non_finite_final_value_is_not_converged():
    # O53 with O20's rule: scipy's status and message, converged only with a finite fun
    def f(x):
        return -math.inf if x[0] > 3.0 else (x[0] - 10.0) ** 2

    r = sd.optimize.minimize(f, [0.0], method="l-bfgs-b", jac=lambda x: sd.asarray([2.0 * (x[0] - 10.0)]))
    # scipy: success with fun = -inf (its gradient test passes at x = 10)
    assert r.status == 0 and r.message == "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
    assert r.fun == -math.inf and not r.converged


def test_no_callback():
    # O54
    import inspect

    assert "callback" not in inspect.signature(sd.optimize.minimize).parameters
    with pytest.raises(TypeError):
        sd.optimize.minimize(quad, [0.0], callback=lambda r: None)


def test_final_simplex():
    # O56: the vertices (N + 1, N), sorted by value, and their values
    r = sd.optimize.minimize(quad, [0.0, 3.0], method="nelder-mead")
    sim, fsim = r.final_simplex
    assert sim.shape == (3, 2) and fsim.shape == (3,)
    assert sim[0].tolist() == r.x.tolist() and fsim[0] == r.fun
    assert fsim.tolist() == sorted(fsim.tolist())
    assert [quad(sim[k]) for k in range(3)] == fsim.tolist()
    assert not hasattr(r, "jac") and not hasattr(r, "njev") and not hasattr(r, "hess_inv")


def test_line_search_cap_is_max_ls():
    # O57 (O14's form); maxcor keeps its name
    r = sd.optimize.minimize(quad, [3.0, -2.0], method="l-bfgs-b", options={"max_ls": 5, "maxcor": 4})
    assert r.converged
    with pytest.raises(ValueError, match=r"has no option 'maxls' \(the cap on line-search steps is 'max_ls'\)"):
        sd.optimize.minimize(quad, [3.0], method="l-bfgs-b", options={"maxls": 5})


def test_iteration_caps_are_max_iter_and_max_nfev():
    # O14
    for method, key, hint in (
        ("bfgs", "maxiter", "max_iter"),
        ("nelder-mead", "maxfev", "max_nfev"),
        ("l-bfgs-b", "maxfun", "max_nfev"),
    ):
        with pytest.raises(ValueError, match=rf"method={method!r} has no option {key!r} \(.*{hint!r}\)"):
            sd.optimize.minimize(quad, [3.0], method=method, options={key: 5})
    r = sd.optimize.minimize(quad, [3.0, 1.0, 2.0], method="nelder-mead", options={"max_nfev": 20})
    assert r.nfev == 20 and r.status == 1 and not r.converged
    r = sd.optimize.minimize(quad, [3.0, 1.0, 2.0], method="nelder-mead", options={"max_iter": 5})
    assert r.n_iter == 5 and r.status == 2
    r = sd.optimize.minimize(lambda x: float(fn.rosen(x.tolist())), [-1.2, 1.0], options={"max_iter": 4})
    assert r.n_iter == 4 and r.status == 1 and r.message == "Maximum number of iterations has been exceeded."


def test_counts_include_the_difference_calls():
    # O18: nfev is every call of fun, the differences' included (scipy's ScalarFunction counts
    # them too); njev the gradients
    calls = []

    def f(x):
        calls.append(1)
        return quad(x)

    for method in ("bfgs", "l-bfgs-b"):
        calls.clear()
        r = sd.optimize.minimize(f, [3.0, -2.0, 0.5], method=method)
        assert r.nfev == len(calls) and r.njev >= 1
        calls.clear()
        r = sd.optimize.minimize(f, [3.0, -2.0, 0.5], method=method, jac="3-point")
        assert r.nfev == len(calls)
        calls.clear()
        r = sd.optimize.minimize(lambda x: (f(x), quad_g(x)), [3.0, -2.0, 0.5], method=method, jac=True)
        assert r.nfev == len(calls) == r.njev


# ---- the result and the user's functions


def test_result_fields():
    r = sd.optimize.minimize(quad, [3.0, -2.0], jac=quad_g)
    assert r.method == "bfgs" and r.converged is True and r.status == 0
    assert isinstance(r.x, sd.ndarray) and r.x.dtype == sd.float64 and r.x.shape == (2,)
    assert isinstance(r.fun, float) and isinstance(r.n_iter, int) and isinstance(r.nfev, int)
    assert isinstance(r.jac, sd.ndarray) and r.jac.shape == (2,) and r.hess_inv.shape == (2, 2)
    with pytest.raises(AttributeError, match="has no setter"):
        r.x = None
    x = r.x
    x[0] = 7.0
    assert r.x[0] != 7.0
    assert repr(r).startswith("OptimizeResult(method='bfgs', converged=True, fun=")


def test_fun_gets_a_new_float64_array():
    seen = []

    def f(x):
        seen.append(x)
        assert isinstance(x, sd.ndarray) and x.dtype == sd.float64 and x.shape == (2,)
        v = quad(x)
        x[0] = 1e300  # changing it changes nothing
        return v

    for method in ("nelder-mead", "bfgs", "l-bfgs-b"):
        seen.clear()
        r = sd.optimize.minimize(f, [3.0, -2.0], method=method)
        assert r.converged and len({id(v) for v in seen}) == len(seen)
        assert max(abs(v - 1.0) for v in r.x.tolist()) < 1e-3


def test_return_values_are_checked():
    import numpy

    ok = sd.optimize.minimize(lambda x: numpy.float64(quad(x)), [3.0], jac=lambda x: [2.0 * (x[0] - 1.0)])
    assert ok.converged
    ok = sd.optimize.minimize(lambda x: quad(x), [3.0], jac=lambda x: 2.0 * (x[0] - 1.0))  # n = 1: a number
    assert ok.converged
    with pytest.raises(ValueError, match=r"'fun' must return a real number, not shape \(1,\)"):
        sd.optimize.minimize(lambda x: x, [3.0])
    with pytest.raises(TypeError, match="'fun' must return a real number, not str"):
        sd.optimize.minimize(lambda x: "a", [3.0], method="nelder-mead")
    with pytest.raises(ValueError, match=r"'jac' must return 2 values \(one per variable\), not shape \(3,\)"):
        sd.optimize.minimize(quad, [3.0, 1.0], jac=lambda x: sd.zeros(3))
    with pytest.raises(TypeError, match=r"'fun' must return a pair \(f, g\) when jac=True, not float"):
        sd.optimize.minimize(quad, [3.0, 1.0], jac=True)
    with pytest.raises(ZeroDivisionError):
        sd.optimize.minimize(lambda x: 1 / 0, [3.0])  # the user's errors pass through


def test_equal_bounds():
    # O58, scipy's rule: every variable fixed -> fun at the bounds, nothing else; some fixed with a
    # difference gradient -> they are taken out (their jac is NaN, no hess_inv field); least_squares
    # keeps lower < upper
    r = sd.optimize.minimize(quad, [2.0, 3.0], bounds=([2.0, 3.0], [2.0, 3.0]))
    assert r.x.tolist() == [2.0, 3.0] and r.fun == 2.0 + 4.0 - 1.0 and r.nfev == 1 and r.njev == 0
    assert r.converged and r.message == "All independent variables were fixed by bounds."
    r = sd.optimize.minimize(quad, [2.0, 3.0, 0.0], bounds=([2.0, -5.0, -5.0], [2.0, 5.0, 5.0]))
    assert r.x[0] == 2.0 and math.isnan(r.jac[0])
    assert not hasattr(r, "hess_inv")  # O58: no field
    with pytest.raises(AttributeError, match="has no attribute 'hess_inv'"):
        _ = r.hess_inv
    assert max(abs(v - 1.0) for v in r.x.tolist()[1:]) < 1e-6
    r = sd.optimize.minimize(quad, [2.0, 3.0, 0.0], jac=quad_g, bounds=([2.0, -5.0, -5.0], [2.0, 5.0, 5.0]))
    assert r.x[0] == 2.0 and r.hess_inv.shape == (3, 3)
    with pytest.raises(ValueError, match="each lower bound must be less than its upper bound"):
        sd.optimize.least_squares(lambda x: x - 1.0, [2.0, 0.0], bounds=([2.0, -5.0], [2.0, 5.0]))


def test_abnormal_fun_is_the_value_at_x():
    # O59: the line search fails (max_ls 1) with an empty memory: x and jac go back to the last
    # iterate and fun is f there (scipy gives the last trial's value); the steps are scipy's
    c = mgh.BY_ID["mgh01-n2-m2-x1"]
    _, _, f, g = mgh.as_numpy(c)
    la, lb = [], []
    a = run_scipy("L-BFGS-B", f, g, c.x0(), None, False, la, options={"maxls": 1})
    b = run_sd("l-bfgs-b", f, g, c.x0(), None, False, lb, options={"max_ls": 1})
    assert a.message == b.message == "ABNORMAL: " and la == lb
    assert b.x.tolist() == a.x.tolist() and b.jac.tolist() == a.jac.tolist()
    assert b.fun == f(np.array(b.x.tolist())) != a.fun


def test_errors():
    m = sd.optimize.minimize
    with pytest.raises(TypeError, match=r"minimize\(\): 'fun' must be callable, not int"):
        m(3, [0.0])
    with pytest.raises(ValueError, match=r"'method' must be lowercase: 'bfgs', not 'BFGS'"):
        m(quad, [0.0], method="BFGS")
    with pytest.raises(ValueError, match=r"'method' must be 'nelder-mead', 'bfgs', 'l-bfgs-b', 'newton-cg', "):
        m(quad, [0.0], method="cg")
    for later in ("newton-cg", "trust-ncg", "trust-exact"):
        with pytest.raises(NotImplementedError, match=rf"method={later!r} is not implemented yet"):
            m(quad, [0.0], method=later)
    with pytest.raises(ValueError, match=r"method='bfgs' takes no 'hess'"):
        m(quad, [0.0], hess=lambda x: None)
    with pytest.raises(ValueError, match=r"method='l-bfgs-b' takes no 'hessp'"):
        m(quad, [0.0], method="l-bfgs-b", hessp=lambda x, p: None)
    with pytest.raises(ValueError, match=r"method='nelder-mead' takes no 'jac'"):
        m(quad, [0.0], method="nelder-mead", jac=quad_g)
    with pytest.raises(ValueError, match=r"'jac' must be callable, True, '2-point', '3-point' or None, not 'cs'"):
        m(quad, [0.0], jac="cs")
    with pytest.raises(ValueError, match=r"not False"):
        m(quad, [0.0], jac=False)
    with pytest.raises(ValueError, match=r"'x0' must be 1-dimensional, not shape \(2, 2\)"):
        m(quad, [[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match=r"'x0' must be finite"):
        m(quad, [math.nan])
    with pytest.raises(TypeError, match=r"'x0' must be real, not a complex128 array"):
        m(quad, sd.asarray([1j]))
    with pytest.raises(ValueError, match=r"'x0' is empty"):
        m(quad, [])
    with pytest.raises(TypeError, match=r"'options' must be a dict or None, not list"):
        m(quad, [0.0], options=[1])
    with pytest.raises(ValueError, match=r"method='bfgs' has no option 'xatol'"):
        m(quad, [0.0], options={"xatol": 1.0})
    with pytest.raises(ValueError, match=r"give 'tol' or options\['gtol'\], not both"):
        m(quad, [0.0], tol=1e-3, options={"gtol": 1e-3})
    with pytest.raises(ValueError, match=r"'eps' is only used when 'jac' is None"):
        m(quad, [0.0], jac=quad_g, options={"eps": 1e-6})
    with pytest.raises(ValueError, match=r"'finite_diff_rel_step' is only used when 'jac' is '2-point' or '3-point'"):
        m(quad, [0.0], options={"finite_diff_rel_step": 1e-6})
    with pytest.raises(ValueError, match=r"'c1' and 'c2' must satisfy 0 < c1 < c2 < 1"):
        m(quad, [0.0], options={"c1": 0.5, "c2": 0.4})
    with pytest.raises(ValueError, match=r"'hess_inv0' must be symmetric positive definite"):
        m(quad, [0.0, 1.0], options={"hess_inv0": [[1.0, 2.0], [2.0, 1.0]]})
    with pytest.raises(ValueError, match=r"'hess_inv0' must have shape \(2, 2\)"):
        m(quad, [0.0, 1.0], options={"hess_inv0": [[1.0]]})
    with pytest.raises(TypeError, match=r"'adaptive' must be True or False, not int"):
        m(quad, [0.0], method="nelder-mead", options={"adaptive": 1})
    with pytest.raises(ValueError, match=r"'initial_simplex' must have shape \(N \+ 1, N\)"):
        m(quad, [0.0, 1.0], method="nelder-mead", options={"initial_simplex": [[0.0, 1.0]]})
    with pytest.raises(TypeError, match=r"'max_iter' must be an int, not float"):
        m(quad, [0.0], options={"max_iter": 5.0})
    with pytest.raises(ValueError, match=r"'max_ls' must be >= 1"):
        m(quad, [0.0], method="l-bfgs-b", options={"max_ls": 0})
    with pytest.raises(ValueError, match=r"'gtol' must be a finite number >= 0"):
        m(quad, [0.0], options={"gtol": -1.0})
