"""sd.optimize.minimize's Newton methods, O4 (docs/api/optimize.md §7, docs/design/optimize.md §3.5,
§7): newton-cg, trust-ncg and trust-exact against scipy 1.18.1 on the Moré-Garbow-Hillstrom
collection as minimization of the sum of squares with its exact Hessian (ad.Dual2) and on scipy's
test functions (the recorded runs, tests/optimize/data/scipy_minimize.json, and every call point
live, bit for bit), with hess, hessp and the difference Hessians; the options and the non-finite
cases live; the decisions O63-O66, the result and the errors."""

import json
import math
import pathlib
import warnings

import numpy as np
import pytest
import scipy
import scipy.optimize as so
import scipy.sparse as sps
from scipy.sparse.linalg import aslinearoperator

import stardust as sd

from . import functions as fn
from . import mgh
from .ad import value_grad_hess

DATA = pathlib.Path(__file__).parent / "data"
REF = json.loads((DATA / "scipy_minimize.json").read_text())
LIVE = pytest.mark.skipif(scipy.__version__ != "1.18.1", reason="the steps are scipy 1.18.1's")
SCIPY_NAME = {"newton-cg": "Newton-CG", "trust-ncg": "trust-ncg", "trust-exact": "trust-exact"}


def problems():
    """(id, f, g, h, x0) of the recorded problems: f, g, h take and return numpy values (the same
    code for both libraries); h is the exact Hessian (the Dual2 of the generator)."""
    out = []
    for c in mgh.CASES:
        _, _, f, g = mgh.as_numpy(c)
        out.append((c.id, f, g, (lambda c: lambda x: mgh.hessian(c, x))(c), c.x0()))

    def np_(h, wrap=np.array):
        return lambda x: wrap(h([np.float64(v) for v in x]))

    for n in (2, 5):
        out.append(
            (
                f"rosen-n{n}",
                np_(fn.rosen, float),
                np_(fn.rosen_der),
                np_(fn.rosen_hess),
                fn.ROSEN_X0[n],
            )
        )
    out.append(
        (
            "maxent",
            np_(fn.maxent, float),
            np_(fn.maxent_der),
            lambda x: value_grad_hess(fn.maxent, x)[2],
            fn.MAXENT_X0,
        )
    )
    return out


PROBLEMS = problems()
BY_ID = {p[0]: p for p in PROBLEMS}


def calls_equal(a, b):
    """Two call logs [(kind, x, ...)] are the same calls at the same points, NaN equal to NaN."""
    if len(a) != len(b):
        return False
    for u, v in zip(a, b):
        if u[0] != v[0]:
            return False
        for p, q in zip(sum(u[1:], ()), sum(v[1:], ())):
            if not (p == q or (p != p and q != q)):
                return False
    return True


def sides(f, g, h, log):
    """(fun, jac, hess, hessp, fun with jac=True) for either library: x as numpy, the call logged
    as (kind, x[, p]); hess values cached by x (the Dual2 Hessian is slow), hessp = H(x) p."""
    cache = {}

    def H(k):
        if k not in cache:
            cache[k] = np.asarray(h(np.array(k)))
        return cache[k]

    def F(x):
        log.append(("f", tuple(x.tolist())))
        return f(np.array(x.tolist()))

    def G(x):
        log.append(("g", tuple(x.tolist())))
        return g(np.array(x.tolist()))

    def Hf(x):
        log.append(("h", tuple(x.tolist())))
        return H(tuple(x.tolist()))

    def HP(x, p):
        log.append(("hp", tuple(x.tolist()), tuple(p.tolist())))
        return np.dot(H(tuple(x.tolist())), np.array(p.tolist()))

    def FG(x):
        log.append(("fg", tuple(x.tolist())))
        return f(np.array(x.tolist())), g(np.array(x.tolist()))

    return F, G, Hf, HP, FG


def run_both(method, f, g, h, x0, hv, opts=None, sopts=None, pair=False, tol=None):
    """scipy's and sd's (result or exception, call log); hv is 'h' (hess), 'hp' (hessp), 'h+hp',
    a difference form or None."""
    out = []
    for lib in ("scipy", "sd"):
        log = []
        F, G, Hf, HP, FG = sides(f, g, h, log)
        kw = {"jac": True} if pair else {"jac": G}
        if hv in ("h", "h+hp"):
            kw["hess"] = Hf
        if hv in ("hp", "h+hp"):
            kw["hessp"] = HP
        if hv in ("2-point", "3-point"):
            kw["hess"] = hv
        if tol is not None:
            kw["tol"] = tol
        fun = FG if pair else F
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if lib == "scipy":
                    r = so.minimize(fun, np.array(x0, dtype=float), method=SCIPY_NAME[method], options=sopts, **kw)
                else:
                    r = sd.optimize.minimize(fun, x0, method=method, options=opts, **kw)
            except Exception as e:  # noqa: BLE001 (both libraries' errors are compared)
                r = e
        out.append((r, log))
    return out


def same_run(a, b, la, lb, method, hv, g):
    """scipy's run (result a, call log la) and sd's (b, lb) are the same steps and agree in every
    field (x, fun, jac and hess bit for bit), but for the decisions that leave scipy: newton-cg's
    jac is the gradient at x (O70: one more call at the end when the memo does not have it, counted
    in njev), and nhev counts the calls of hess and hessp made (O68, O69; scipy adds 1 for its
    placeholder hess with hessp alone and leaves out trust-ncg's call for the result's hess)."""
    extra = lb[len(la) :]
    assert calls_equal(la, lb[: len(la)])
    if method == "newton-cg":
        assert extra in ([], [("g", tuple(b.x.tolist()))], [("fg", tuple(b.x.tolist()))])
        assert calls_equal([("j", tuple(b.jac.tolist()))], [("j", tuple(float(v) for v in g(np.array(b.x.tolist()))))])
    else:
        assert extra == []
        if a.jac is not None:
            assert calls_equal([("j", tuple(b.jac.tolist()))], [("j", tuple(float(v) for v in a.jac))])
    if hv in ("h", "hp", "h+hp"):
        assert b.nhev == sum(1 for c in lb if c[0] in ("h", "hp"))
    else:
        assert b.nhev == a.nhev
    assert b.x.tolist() == [float(v) for v in a.x]
    assert b.fun == a.fun or (math.isnan(a.fun) and math.isnan(b.fun))
    assert (b.nfev, b.njev - len(extra), b.n_iter, b.status, b.message) == (
        a.nfev,
        a.njev,
        a.nit,
        a.status,
        a.message,
    )
    assert b.converged == (bool(a.success) and math.isfinite(a.fun))
    if "hess" in a:
        assert np.array(b.hess.tolist()).tolist() == np.asarray(a.hess).tolist()
    else:
        assert not hasattr(b, "hess")


# ---- scipy 1.18.1, recorded (O9): the same steps bit for bit, so every field is scipy's


@pytest.mark.parametrize("method", ["newton-cg", "trust-ncg", "trust-exact"])
def test_against_recorded_scipy(method):
    # every field is scipy's but njev and nhev where the decisions leave scipy: newton-cg's jac at
    # x may cost one more jac call (O70); nhev is the calls of hess made (O69: trust-ncg's call for
    # the result's hess counts)
    group = SCIPY_NAME[method]
    for pid, f, g, h, x0 in PROBLEMS:
        rec = REF[pid][group]
        calls = {"g": 0, "h": 0}

        def G(x, g=g):
            calls["g"] += 1
            return g(np.array(x.tolist()))

        def H(x, h=h):
            calls["h"] += 1
            return h(np.array(x.tolist()))

        with np.errstate(all="ignore"):
            r = sd.optimize.minimize(lambda x: f(np.array(x.tolist())), x0, method=method, jac=G, hess=H)
        assert r.x.tolist() == rec["x"], pid
        assert r.fun == rec["fun"], pid
        got = (r.nfev, r.n_iter, r.status, r.message)
        assert got == (rec["nfev"], rec["nit"], rec["status"], rec["message"]), pid
        assert r.converged == rec["success"], pid
        assert r.njev == calls["g"] and r.nhev == calls["h"], pid
        if method == "newton-cg":
            assert r.njev - rec["njev"] in (0, 1), pid
            assert r.jac.tolist() == g(np.array(r.x.tolist())).tolist(), pid
        else:
            assert r.njev == rec["njev"], pid
            assert r.nhev - rec["nhev"] in ((0, 1) if method == "trust-ncg" else (0,)), pid


# ---- live: the points where fun, jac, hess and hessp are called, in order, bit for bit

VARIANTS = (
    "newton-cg/h",
    "newton-cg/hp",
    "newton-cg/2-point",
    "newton-cg/3-point",
    "newton-cg/none",
    "trust-ncg/h",
    "trust-ncg/hp",
    "trust-ncg/2-point",
    "trust-ncg/3-point",
    "trust-exact/h",
)


@LIVE
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_call_point_follows_scipy(variant):
    method, _, hv = variant.partition("/")
    hv = None if hv == "none" else hv
    for _pid, f, g, h, x0 in PROBLEMS:
        (a, la), (b, lb) = run_both(method, f, g, h, x0, hv)
        same_run(a, b, la, lb, method, hv, g)


# ---- live: options and edge cases (jac=True, both hess and hessp, the caps, non-finite values)

SWEEP_IDS = (
    "mgh01-n2-m2-x1",
    "mgh04-n2-m3-x1",
    "mgh07-n3-m3-x10",
    "mgh13-n4-m4-x1",
    "mgh20-n9-m31-x1",
    "mgh26-n10-m10-x1",
    "mgh35-n1-m8-x1",
)


def _cut(f, g, h, x0, val, what):
    """f, g, h with the value val (inf or NaN) where x[0] is past a point of the path."""
    cut = x0[0] + 0.3 * (abs(x0[0]) + 1)
    n = len(x0)
    ff = (lambda x: val if x[0] > cut else f(x)) if "f" in what else f
    gg = (lambda x: np.full(n, val) if x[0] > cut else g(x)) if "g" in what else g
    hh = (lambda x: np.full((n, n), val) if x[0] > cut else h(x)) if "h" in what else h
    return ff, gg, hh


SCIPY_KEYS = {"max_iter": "maxiter", "subproblem_max_iter": "subproblem_maxiter"}


def sweep_cases():
    rows = []
    hvs = {
        "newton-cg": ("h", "hp", "2-point", None),
        "trust-ncg": ("h", "hp", "3-point", "h+hp"),
        "trust-exact": ("h",),
    }
    for method in ("newton-cg", "trust-ncg", "trust-exact"):
        for hv in hvs[method]:
            rows.append((method, hv, {}, {"pair": True}))
        rows += [
            (method, "h", {"max_iter": 3}, {}),
            (method, "h", {"max_iter": 1}, {}),
            (method, "h", {}, {"tol": 1e-10}),
        ]
        if method == "newton-cg":
            rows.append((method, "h", {"max_iter": 0}, {}))
        if method == "newton-cg":
            rows += [
                (method, "h", {"xtol": 1e-9}, {}),
                (method, None, {"eps": 1e-6}, {}),
                (method, "h", {"c1": 0.3, "c2": 0.4}, {}),
                (method, "hp", {}, {"tol": 1e-10}),
            ]
        else:
            rows += [
                (method, "h", {"gtol": 1e-9}, {}),
                (method, "h", {"initial_trust_radius": 0.01, "max_trust_radius": 0.5}, {}),
                (method, "h", {"eta": 0.0}, {}),
                (method, "h", {"eta": 0.2}, {}),
                (method, "h", {"max_trust_radius": math.inf}, {}),
            ]
        if method == "trust-exact":
            rows += [(method, "h", {"subproblem_max_iter": 3}, {}), (method, "h", {"subproblem_max_iter": 1}, {})]
        for val in (math.inf, math.nan):
            for what in ("f", "g", "h", "fgh"):
                for hv in {"newton-cg": ("h", None), "trust-ncg": ("h", "hp"), "trust-exact": ("h",)}[method]:
                    rows.append((method, hv, {}, {"cut": (val, what)}))
    out = []
    for pid in SWEEP_IDS:
        for method, hv, opts, kw in rows:
            name = f"{pid} {method} {hv} {sorted(opts.items())} {sorted(kw.items())}"
            out.append((name, pid, method, hv, opts, kw))
    return out


@LIVE
@pytest.mark.parametrize("case", sweep_cases(), ids=lambda c: c[0])
def test_options_and_edge_cases_follow_scipy(case):
    _, pid, method, hv, opts, kw = case
    _, f, g, h, x0 = BY_ID[pid]
    if "cut" in kw:
        f, g, h = _cut(f, g, h, x0, *kw["cut"])
    sopts = {SCIPY_KEYS.get(k, k): v for k, v in opts.items()}
    (a, la), (b, lb) = run_both(method, f, g, h, x0, hv, opts, sopts, pair=kw.get("pair", False), tol=kw.get("tol"))
    if isinstance(a, Exception):
        assert calls_equal(la, lb)
    if isinstance(a, UnboundLocalError):
        # scipy's trust-exact subproblem ran out of iterations before any step (its p is unbound):
        # here status 3, scipy's linalg message, at the last iterate
        assert b.status == 3 and not b.converged and b.message.startswith("A linalg error")
        return
    if isinstance(a, ValueError):
        # scipy.linalg's check_finite (on |g|, a new subproblem's Hessian, the subproblem's
        # vectors): a ValueError here too, naming what and where
        assert isinstance(b, ValueError)
        assert "array must not contain infs or NaNs" in str(a)
        assert str(b).startswith("minimize(): ") and "not finite at x = [" in str(b)
        return
    assert not isinstance(b, Exception), b
    same_run(a, b, la, lb, method, hv, g)


# ---- the decisions


def quad(x):
    return float(sd.sum((x - 1.0) ** 2 * sd.asarray([1.0, 3.0, 5.0][: x.shape[0]])))


def quad_g(x):
    return 2.0 * (x - 1.0) * sd.asarray([1.0, 3.0, 5.0][: x.shape[0]])


def quad_h(x):
    n = x.shape[0]
    return sd.asarray([[2.0 * [1.0, 3.0, 5.0][i] if i == j else 0.0 for j in range(n)] for i in range(n)])


def quad_hp(x, p):
    return quad_h(x) @ p


X0 = [3.0, -2.0, 0.5]


def test_hess_and_hessp_forms():
    # O63: hess(x) gives (n, n) (sd.ndarray, numpy array or nested sequence; a number when n = 1),
    # hessp(x, p) gives (n,); scipy's sparse matrices and LinearOperator are not taken
    for method in ("newton-cg", "trust-ncg", "trust-exact"):
        for hess in (quad_h, lambda x: np.array(quad_h(x).tolist()), lambda x: quad_h(x).tolist()):
            r = sd.optimize.minimize(quad, X0, method=method, jac=quad_g, hess=hess)
            assert r.converged and max(abs(v - 1.0) for v in r.x.tolist()) < 1e-6
    for method in ("newton-cg", "trust-ncg"):
        for hessp in (quad_hp, lambda x, p: np.array(quad_hp(x, p).tolist()), lambda x, p: quad_hp(x, p).tolist()):
            r = sd.optimize.minimize(quad, X0, method=method, jac=quad_g, hessp=hessp)
            assert r.converged and max(abs(v - 1.0) for v in r.x.tolist()) < 1e-6
    r = sd.optimize.minimize(
        lambda x: (x[0] - 2.0) ** 2, [5.0], method="trust-exact", jac=lambda x: 2 * (x[0] - 2.0), hess=lambda x: 2.0
    )
    assert r.converged and abs(r.x[0] - 2.0) < 1e-12
    m = sd.optimize.minimize
    with pytest.raises(TypeError, match=r"minimize\(\): 'hess' must return an sd.ndarray .*, not csr_array"):
        m(quad, X0, method="newton-cg", jac=quad_g, hess=lambda x: sps.csr_array(np.eye(3)))
    with pytest.raises(TypeError, match=r"'hess' must return an sd.ndarray .*, not MatrixLinearOperator"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hess=lambda x: aslinearoperator(np.eye(3)))
    with pytest.raises(TypeError, match=r"'hessp' must return an sd.ndarray .*, not MatrixLinearOperator"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hessp=lambda x, p: aslinearoperator(np.eye(3)))
    with pytest.raises(ValueError, match=r"'hess' must return shape \(3, 3\), not \(2, 2\)"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=lambda x: sd.eye(2))
    with pytest.raises(ValueError, match=r"'hess' must return shape \(3, 3\), not a scalar"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=lambda x: 2.0)
    with pytest.raises(ValueError, match=r"'hessp' must return 3 values \(one per variable\), not shape \(2,\)"):
        m(quad, X0, method="newton-cg", jac=quad_g, hessp=lambda x, p: [1.0, 2.0])
    with pytest.raises(TypeError, match=r"'hess' must be callable, '2-point', '3-point' or None, not BFGS"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hess=so.BFGS())
    with pytest.raises(TypeError, match=r"'hessp' must be callable or None, not int"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hessp=3)


def test_difference_hessians():
    # O64: hess='2-point' / '3-point' (products by differences of jac, scipy's linear operator);
    # no 'cs'; newton-cg with neither hess nor hessp differences the gradient (scipy's
    # approx_fhess_p with the step eps, no Hessian counted); trust-exact needs the matrix
    for method in ("newton-cg", "trust-ncg"):
        for hess in ("2-point", "3-point"):
            r = sd.optimize.minimize(quad, X0, method=method, jac=quad_g, hess=hess)
            assert r.converged and max(abs(v - 1.0) for v in r.x.tolist()) < 1e-6
            assert r.nhev > 0 and r.njev > r.n_iter and not hasattr(r, "hess")
    r = sd.optimize.minimize(quad, X0, method="newton-cg", jac=quad_g)
    assert r.converged and r.nhev == 0
    m = sd.optimize.minimize
    with pytest.raises(ValueError, match=r"'hess' must be callable, '2-point', '3-point' or None, not 'cs'"):
        m(quad, X0, method="newton-cg", jac=quad_g, hess="cs")
    with pytest.raises(
        ValueError, match=r"method='trust-exact' needs 'hess' as a callable \(the matrix\), not '2-point'"
    ):
        m(quad, X0, method="trust-exact", jac=quad_g, hess="2-point")
    with pytest.raises(ValueError, match=r"give 'hessp' or hess='3-point', not both"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hess="3-point", hessp=quad_hp)


def test_jac_is_required():
    # O65: a callable jac or jac=True; the difference forms are an error too (scipy makes them None)
    m = sd.optimize.minimize
    for method in ("newton-cg", "trust-ncg", "trust-exact"):
        with pytest.raises(ValueError, match=rf"'jac' is required when method={method!r} \(a callable, or True"):
            m(quad, X0, method=method, hess=quad_h)
        for jac in ("2-point", "3-point"):
            with pytest.raises(ValueError, match=rf"method={method!r} needs 'jac' as a callable or True, not {jac!r}"):
                m(quad, X0, method=method, jac=jac, hess=quad_h)
    r = m(lambda x: (quad(x), quad_g(x)), X0, method="trust-exact", jac=True, hess=quad_h)
    assert r.converged and r.nfev == r.njev  # one call of fun per point gives both


def test_bounds_are_an_error():
    # O66: scipy warns and ignores them; here bfgs's rule
    for method in ("newton-cg", "trust-ncg", "trust-exact"):
        with pytest.raises(ValueError, match=rf"method={method!r} takes no bounds \(method='l-bfgs-b' and"):
            sd.optimize.minimize(quad, X0, method=method, jac=quad_g, hess=quad_h, bounds=(-5.0, 5.0))


def test_trust_methods_need_their_hessian():
    m = sd.optimize.minimize
    with pytest.raises(ValueError, match=r"method='trust-ncg' needs 'hess' or 'hessp'"):
        m(quad, X0, method="trust-ncg", jac=quad_g)
    with pytest.raises(ValueError, match=r"'hess' is required when method='trust-exact'"):
        m(quad, X0, method="trust-exact", jac=quad_g)
    with pytest.raises(ValueError, match=r"method='trust-exact' takes no 'hessp'"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h, hessp=quad_hp)


def test_counts():
    # nfev, njev as ScalarFunction counts them; nhev the calls of hess and hessp made (O68, O69):
    # no 1 for scipy's placeholder hess with hessp alone, and trust-ncg's call for the result's hess
    # counted
    calls = {"f": 0, "g": 0, "h": 0, "hp": 0}

    def count(k, fn_):
        def w(*a):
            calls[k] += 1
            return fn_(*a)

        return w

    def go(method, options=None, **kw):
        for k in calls:
            calls[k] = 0
        kw = {k: count("hp" if k == "hessp" else "h", v) for k, v in kw.items()}
        return sd.optimize.minimize(count("f", quad), X0, method=method, jac=count("g", quad_g), options=options, **kw)

    for method, kw in (
        ("newton-cg", {"hess": quad_h}),
        ("newton-cg", {"hessp": quad_hp}),
        ("trust-ncg", {"hess": quad_h}),
        ("trust-ncg", {"hessp": quad_hp}),
        ("trust-ncg", {"hess": quad_h, "hessp": quad_hp}),
        ("trust-exact", {"hess": quad_h}),
    ):
        r = go(method, **kw)
        assert (r.nfev, r.njev, r.nhev) == (calls["f"], calls["g"], calls["h"] + calls["hp"]), (method, kw)
    r = go("newton-cg", hess=quad_h, options={"max_iter": 0})
    assert r.nhev == calls["h"] == 1  # the call at x0 (scipy's count is 0)


@LIVE
def test_counts_leave_scipy():
    # O68: hessp alone, 1 less than scipy; O69: trust-ncg's final hess call, 1 more when made
    _, f, g, h, x0 = BY_ID["mgh01-n2-m2-x1"]
    (a, la), (b, lb) = run_both("trust-ncg", f, g, h, x0, "hp")
    assert b.nhev == a.nhev - 1 == sum(1 for c in lb if c[0] == "hp")
    (a, la), (b, lb) = run_both("trust-ncg", f, g, h, x0, "h")
    assert la == lb and la[-1][0] == "h"  # the result's hess, at x
    assert b.nhev == a.nhev + 1 == sum(1 for c in lb if c[0] == "h")


def test_result_fields():
    r = sd.optimize.minimize(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h)
    assert r.method == "trust-exact" and r.converged is True and r.status == 0
    assert r.message == "Optimization terminated successfully."
    assert isinstance(r.x, sd.ndarray) and r.x.shape == (3,) and r.jac.shape == (3,)
    assert isinstance(r.hess, sd.ndarray) and r.hess.shape == (3, 3) and r.hess.dtype == sd.float64
    assert r.hess.tolist() == quad_h(r.x).tolist()  # at x
    assert isinstance(r.nhev, int) and isinstance(r.njev, int) and isinstance(r.n_iter, int)
    assert not hasattr(r, "hess_inv")
    H = r.hess
    H[0, 0] = 99.0
    assert r.hess[0, 0] != 99.0  # a new copy each read (G113)
    r = sd.optimize.minimize(quad, X0, method="trust-ncg", jac=quad_g, hessp=quad_hp)
    assert not hasattr(r, "hess")  # with hess only
    r = sd.optimize.minimize(quad, X0, method="newton-cg", jac=quad_g, hess=quad_h)
    assert not hasattr(r, "hess") and r.nhev >= 1
    assert repr(r).startswith("OptimizeResult(method='newton-cg', converged=True, fun=")


@LIVE
def test_newton_cg_jac_is_at_x():
    # O70: the gradient at the returned x (scipy's is the one at the start of the last iteration):
    # from the memo when it is there, else one more jac call, counted in njev
    extra = 0
    for pid, hv in (("mgh01-n2-m2-x1", "h"), ("rosen-n5", "h"), ("mgh04-n2-m3-x1", None), ("mgh31-n10-m10-x100", None)):
        _, f, g, h, x0 = BY_ID[pid]
        (a, la), (b, lb) = run_both("newton-cg", f, g, h, x0, hv)
        assert b.jac.tolist() == g(np.array(b.x.tolist())).tolist() != a.jac.tolist()
        assert b.njev == a.njev + len(lb) - len(la)
        extra += len(lb) - len(la)
    assert extra == 2  # the last two: the gradient differences moved the memo away from x
    _, f, g, h, x0 = BY_ID["mgh01-n2-m2-x1"]
    (a, la), (b, lb) = run_both("newton-cg", f, g, h, x0, "h", {"max_iter": 0}, {"maxiter": 0})
    assert a.jac is None and b.jac.tolist() == g(np.array(x0)).tolist() and b.status == 1 and la == lb


def test_newton_cg_takes_hess_or_hessp():
    # O74 (O62's rule): with both, hessp would not be used; scipy ignores it
    with pytest.raises(ValueError, match=r"minimize\(\): method='newton-cg' takes 'hess' or 'hessp', not both"):
        sd.optimize.minimize(quad, X0, method="newton-cg", jac=quad_g, hess=quad_h, hessp=quad_hp)
    r = sd.optimize.minimize(quad, X0, method="trust-ncg", jac=quad_g, hess=quad_h, hessp=quad_hp)
    assert r.converged and hasattr(r, "hess")  # trust-ncg uses both (hessp's products, hess's result)


def test_trust_max_iter_zero_stops_at_x0():
    # O75: no iteration (as newton-cg and bfgs); scipy's trust methods do one
    calls = []

    def f(x):
        calls.append("f")
        return quad(x)

    for method, kw in (("trust-ncg", {"hessp": quad_hp}), ("trust-exact", {"hess": quad_h})):
        calls.clear()
        r = sd.optimize.minimize(f, X0, method=method, jac=quad_g, options={"max_iter": 0}, **kw)
        assert (r.status, r.n_iter, r.nfev, r.converged) == (1, 0, 1, False) and r.x.tolist() == X0
        assert r.message == "Maximum number of iterations has been exceeded." and calls == ["f"]
    r = sd.optimize.minimize(
        quad, [1.0, 1.0, 1.0], method="trust-exact", jac=quad_g, hess=quad_h, options={"max_iter": 0}
    )
    assert (r.status, r.n_iter, r.converged) == (0, 0, True)  # |g| < gtol at x0: nothing to do
    for method in ("trust-ncg", "trust-exact"):
        r = sd.optimize.minimize(quad, X0, method=method, jac=quad_g, hess=quad_h, options={"max_iter": 1})
        assert r.n_iter == 1 and r.status == 1


def test_non_finite_at_x0_is_an_error():
    # O53: fun, jac and hess at x0 (scipy's ScalarFunction computes all three there)
    m = sd.optimize.minimize
    for method in ("newton-cg", "trust-ncg", "trust-exact"):
        with pytest.raises(ValueError, match=r"minimize\(\): 'fun' is not finite at 'x0'"):
            m(lambda x: math.inf, X0, method=method, jac=quad_g, hess=quad_h)
        with pytest.raises(ValueError, match=r"minimize\(\): 'jac' is not finite at 'x0'"):
            m(quad, X0, method=method, jac=lambda x: sd.full(3, math.nan), hess=quad_h)
        with pytest.raises(ValueError, match=r"minimize\(\): 'hess' is not finite at 'x0'"):
            m(quad, X0, method=method, jac=quad_g, hess=lambda x: sd.full((3, 3), math.inf))


def test_non_finite_on_the_way_where_scipy_raises():
    # O53: scipy's handling (the live sweep above: Newton-CG's line search shrinks the step, the
    # trust-region methods reject it); where scipy.linalg raises (a non-finite gradient at an
    # accepted point, a non-finite Hessian of a new trust-exact subproblem) a ValueError naming
    # what and where
    def h(x):
        return sd.full((3, 3), math.nan) if x[0] < 2.9 else quad_h(x)

    with pytest.raises(ValueError, match=r"minimize\(\): 'hess' is not finite at x = \["):
        sd.optimize.minimize(quad, X0, method="trust-exact", jac=quad_g, hess=h)

    def g(x):
        return sd.full(3, math.nan) if x[0] < 2.9 else quad_g(x)

    with pytest.raises(ValueError, match=r"minimize\(\): 'jac' is not finite at x = \["):
        sd.optimize.minimize(quad, X0, method="trust-ncg", jac=g, hess=quad_h)


def test_functions_get_new_arrays():
    seen = []

    def h(x):
        seen.append(x)
        assert isinstance(x, sd.ndarray) and x.dtype == sd.float64 and x.shape == (3,)
        v = quad_h(x)
        x[0] = 1e300  # changing it changes nothing
        return v

    def hp(x, p):
        seen.append(x)
        seen.append(p)
        assert isinstance(p, sd.ndarray) and p.dtype == sd.float64 and p.shape == (3,)
        v = quad_hp(x, p)
        x[0] = p[0] = 1e300
        return v

    for method, kw in (("newton-cg", {"hess": h}), ("trust-exact", {"hess": h}), ("trust-ncg", {"hessp": hp})):
        seen.clear()
        r = sd.optimize.minimize(quad, X0, method=method, jac=quad_g, **kw)
        assert r.converged and len({id(v) for v in seen}) == len(seen)


def test_options():
    m = sd.optimize.minimize
    r = m(quad, X0, method="newton-cg", jac=quad_g, options={"eps": [1e-6, 1e-7, 1e-8]})
    assert r.converged
    r = m(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h, options={"max_iter": 1})
    assert r.n_iter == 1 and r.status == 1 and r.message == "Maximum number of iterations has been exceeded."
    r = m(quad, X0, method="newton-cg", jac=quad_g, hess=quad_h, options={"max_iter": 1})
    assert r.status == 1 and r.message == "Warning: Maximum number of iterations has been exceeded."
    with pytest.raises(
        ValueError, match=r"method='newton-cg' has no option 'maxiter' \(the iteration cap is 'max_iter'\)"
    ):
        m(quad, X0, method="newton-cg", jac=quad_g, options={"maxiter": 3})
    with pytest.raises(ValueError, match=r"has no option 'subproblem_maxiter' \(.*'subproblem_max_iter'\)"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h, options={"subproblem_maxiter": 3})
    with pytest.raises(ValueError, match=r"method='trust-ncg' has no option 'subproblem_max_iter'"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hess=quad_h, options={"subproblem_max_iter": 3})
    with pytest.raises(ValueError, match=r"method='trust-ncg' has no option 'xtol'"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hess=quad_h, options={"xtol": 3})
    with pytest.raises(ValueError, match=r"'eps' is only used when method='newton-cg' has no 'hess' and no 'hessp'"):
        m(quad, X0, method="newton-cg", jac=quad_g, hess=quad_h, options={"eps": 1e-6})
    with pytest.raises(ValueError, match=r"'eps' must be finite and > 0"):
        m(quad, X0, method="newton-cg", jac=quad_g, options={"eps": 0.0})
    with pytest.raises(ValueError, match=r"give 'tol' or options\['xtol'\], not both"):
        m(quad, X0, method="newton-cg", jac=quad_g, tol=1e-3, options={"xtol": 1e-3})
    with pytest.raises(ValueError, match=r"give 'tol' or options\['gtol'\], not both"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hess=quad_h, tol=1e-3, options={"gtol": 1e-3})
    with pytest.raises(ValueError, match=r"'c1' and 'c2' must satisfy 0 < c1 < c2 < 1"):
        m(quad, X0, method="newton-cg", jac=quad_g, options={"c1": 0.5, "c2": 0.4})
    with pytest.raises(ValueError, match=r"'eta' must be >= 0 and < 0.25"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h, options={"eta": 0.25})
    with pytest.raises(ValueError, match=r"'max_trust_radius' must be > 0"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h, options={"max_trust_radius": 0.0})
    with pytest.raises(ValueError, match=r"'initial_trust_radius' must be finite and > 0"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hess=quad_h, options={"initial_trust_radius": -1.0})
    with pytest.raises(ValueError, match=r"'initial_trust_radius' must be less than 'max_trust_radius'"):
        m(quad, X0, method="trust-ncg", jac=quad_g, hess=quad_h, options={"initial_trust_radius": 1000.0})
    with pytest.raises(ValueError, match=r"'subproblem_max_iter' must be >= 1"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h, options={"subproblem_max_iter": 0})
    with pytest.raises(TypeError, match=r"'max_iter' must be an int, not float"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h, options={"max_iter": 3.0})
    with pytest.raises(ValueError, match=r"'gtol' must be a finite number >= 0"):
        m(quad, X0, method="trust-exact", jac=quad_g, hess=quad_h, options={"gtol": -1.0})
    with pytest.raises(ValueError, match=r"'xtol' must be a finite number >= 0"):
        m(quad, X0, method="newton-cg", jac=quad_g, options={"xtol": math.inf})


def test_trust_ncg_cg_is_capped():
    # scipy's Steihaug CG has no cap: with this nonsymmetric hess it cycles forever in the 26th
    # subproblem. Here 20n CG steps (with a symmetric Hessian scipy's runs take about n, at most
    # 1.2n) stop with status 3 at the last iterate
    rng = np.random.default_rng(3)
    for t in range(97):
        n = int(rng.integers(2, 13))
        a = rng.uniform(0.01, 1.0, n)
        B = rng.standard_normal((n, n))
        B = (B + B.T) / 2 * rng.uniform(0.5, 5)
        c = rng.standard_normal(n)
        if t % 4 == 0:
            B = B + 0.3 * rng.standard_normal((n, n))
        x0 = rng.standard_normal(n) * rng.choice([0.01, 1.0, 10.0])
    assert not (B == B.T).all()

    def f(x):
        x = np.array(x.tolist())
        return float(np.sum(a * x**4) + 0.5 * x @ B @ x + c @ x)

    def g(x):
        x = np.array(x.tolist())
        return 4 * a * x**3 + 0.5 * (B + B.T) @ x + c

    def h(x):
        return np.diag(12 * a * np.array(x.tolist()) ** 2) + B

    r = sd.optimize.minimize(f, x0.tolist(), method="trust-ncg", jac=g, hess=h)
    assert (r.status, r.n_iter, r.converged) == (3, 25, False)
    assert r.message == "A linalg error occurred, such as a non-psd Hessian."
