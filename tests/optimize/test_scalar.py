"""sd.optimize.root_scalar and minimize_scalar (docs/api/optimize.md §4, §5, docs/design/optimize.md
§3.2, §3.3, §7): scipy 1.18.1 on the 204 root and 57 minimum problems of functions.py (the recorded
runs, tests/optimize/data/scipy_root_scalar.json and scipy_minimize_scalar.json, and every call point
live), the Arb roots (scalar_roots.json), the counts, the edge cases, the result and the errors."""

import json
import math
import pathlib
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize_scalar as scipy_min
from scipy.optimize import root_scalar as scipy_root

import stardust as sd
from bench.accuracy.common import ulp_distance

from . import functions as fn
from . import refdata
from .conftest import LIVE

DATA = pathlib.Path(__file__).parent / "data"
REF_ROOT = json.loads((DATA / "scipy_root_scalar.json").read_text())
REF_MIN = json.loads((DATA / "scipy_minimize_scalar.json").read_text())
ARB = refdata.roots()
EPS = 2.220446049250313e-16

BRACKETING = ("brentq", "toms748", "bisect")


def methods(c):
    out = list(BRACKETING)
    if c.x0 is not None:
        out += ["secant", "newton/fd"]
        if c.fprime is not None:
            out.append("newton")
        if c.fprime2 is not None:
            out.append("halley")
    return out


ROOT_RUNS = [(c.id, m) for c in fn.ROOT_CASES for m in methods(c)]
MIN_RUNS = [(c.id, m) for c in fn.MIN_CASES for m in ("brent", "golden", "bounded")]


def scalar(g, log=None):
    def f(x):
        if log is not None:
            log.append(float(x))
        with np.errstate(all="ignore"):
            return float(g(np.float64(x)))

    return f


def root_kw(c, m, log=None):
    base = m.split("/")[0]
    kw = {"method": base}
    if base in BRACKETING:
        kw["bracket"] = c.bracket
    else:
        kw["x0"] = c.x0
        if m in ("newton", "halley"):
            kw["fprime"] = scalar(c.fprime)
        if m == "halley":
            kw["fprime2"] = scalar(c.fprime2)
    return kw


def run_root(c, m, log=None):
    """The result and scipy's function_calls: f, fprime and fprime2 together (sd counts them apart;
    newton's difference calls f twice per iteration, which scipy counts as f and fprime)."""
    r = sd.optimize.root_scalar(scalar(c.f, log), **root_kw(c, m))
    n = r.nfev if m == "newton/fd" else r.nfev + getattr(r, "njev", 0) + getattr(r, "nhev", 0)
    return r, n


# scipy's flag 'convergence error' is this library's message, or the text scipy warns with (it
# returns no reason): a zero derivative, a secant step of zero
CONV_ERR = ("convergence error", "Derivative was zero.")


def same_message(flag, message):
    if flag == "convergence error":
        return message in CONV_ERR or (message.startswith("Tolerance of ") and message.endswith(" reached."))
    return flag == message


# ---- scipy 1.18.1, recorded: the same root, iterations, calls and stopping on every run


@pytest.mark.parametrize("cid,m", ROOT_RUNS, ids=[f"{c}-{m}" for c, m in ROOT_RUNS])
def test_root_against_recorded_scipy(cid, m):
    c = fn.ROOTS_BY_ID[cid]
    s = REF_ROOT[cid][m]
    r, n = run_root(c, m)
    assert r.x == s["root"]
    assert r.n_iter == s["iterations"]
    assert n == s["function_calls"]
    assert r.converged == s["converged"]
    assert same_message(s["flag"], r.message)
    assert r.status == (0 if s["converged"] else -2)


@pytest.mark.parametrize("cid,m", MIN_RUNS, ids=[f"{c}-{m}" for c, m in MIN_RUNS])
def test_minimum_against_recorded_scipy(cid, m):
    c = fn.MINS_BY_ID[cid]
    s = REF_MIN[cid][m]
    f = scalar(c.f)
    if m == "bounded":
        r = sd.optimize.minimize_scalar(f, bounds=c.bounds, method=m)
    else:
        r = sd.optimize.minimize_scalar(f, bracket=c.bracket, method=m)
    assert (r.x, r.fun, r.n_iter, r.nfev, r.converged, r.message) == (
        s["x"][0],  # recorded as a list
        s["fun"],
        s["nit"],
        s["nfev"],
        s["success"],
        s["message"],
    )
    if "status" in s:
        assert r.status == s["status"]


# ---- live: every point where f is called, bit for bit


@LIVE
def test_root_every_call_point_follows_scipy():
    # 1244 runs. brentq and bisect are scipy's C code (built with fused multiply-adds; scalar.cpp
    # writes them out); newton without fprime calls f(x) once per iteration (scipy twice: newton
    # and approx_derivative)
    warnings.simplefilter("ignore")
    for cid, m in ROOT_RUNS:
        c = fn.ROOTS_BY_ID[cid]
        ls, ld = [], []
        s = scipy_root(scalar(c.f, ls), **root_kw(c, m))
        r, n = run_root(c, m, ld)
        if m == "newton/fd":
            ls = [v for i, v in enumerate(ls) if i % 3 != 1]  # scipy's second f(x)
        assert ls == ld, (cid, m)
        assert (r.x, r.n_iter, n, r.converged) == (s.root, s.iterations, s.function_calls, s.converged)


@LIVE
def test_minimum_every_call_point_follows_scipy():
    # the three methods, and brent and golden from a two-point bracket and from none (scipy's
    # bracket() search)
    warnings.simplefilter("ignore")
    for c in fn.MIN_CASES:
        for m, kw in (
            ("brent", {"bracket": c.bracket}),
            ("golden", {"bracket": c.bracket}),
            ("bounded", {"bounds": c.bounds}),
            ("brent", {"bracket": c.bracket[:2]}),
            ("golden", {"bracket": c.bracket[:2]}),
            ("brent", {}),
            ("golden", {}),
        ):
            ls, ld = [], []
            s = scipy_min(scalar(c.f, ls), method=m, **kw)
            r = sd.optimize.minimize_scalar(scalar(c.f, ld), method=m, **kw)
            assert ls == ld, (c.id, m, kw)
            assert (r.x, r.fun, r.n_iter, r.nfev, r.converged, r.message) == (
                s.x,
                s.fun,
                s.nit,
                s.nfev,
                s.success,
                s.message,
            ), (c.id, m, kw)


@LIVE
@pytest.mark.parametrize(
    "kw",
    [
        dict(method="brentq", xtol=1e-6, rtol=1e-10, max_iter=7),
        dict(method="bisect", xtol=1e-3, max_iter=12),
        dict(method="toms748", xtol=1e-9, rtol=1e-12, max_iter=3),
        dict(method="toms748", options={"k": 2}),
        dict(method="newton", xtol=1e-12, rtol=1e-9, max_iter=4),
        dict(method="secant", x1=0.7, max_iter=5),
        dict(method="halley", rtol=1e-14, max_iter=3),
    ],
)
def test_root_options_follow_scipy(kw):
    warnings.simplefilter("ignore")
    kw = dict(kw)
    scipy_kw = dict(kw)
    if "max_iter" in scipy_kw:
        scipy_kw["maxiter"] = scipy_kw.pop("max_iter")
    for c in fn.ROOT_CASES:
        if kw["method"] in BRACKETING:
            extra = {"bracket": c.bracket}
        elif c.x0 is None or (kw["method"] == "halley" and c.fprime2 is None):
            continue
        else:
            extra = {"x0": c.x0}
            if kw["method"] == "secant" and c.x0 == 0.7:
                continue
            if kw["method"] in ("newton", "halley") and c.fprime is not None:
                extra["fprime"] = scalar(c.fprime)
            if kw["method"] == "halley":
                extra["fprime2"] = scalar(c.fprime2)
        ls, ld = [], []
        s = scipy_root(scalar(c.f, ls), **scipy_kw, **extra)
        r = sd.optimize.root_scalar(scalar(c.f, ld), **kw, **extra)
        if kw["method"] == "newton" and "fprime" not in extra:
            ls = [v for i, v in enumerate(ls) if i % 3 != 1]
        assert ls == ld and r.x == s.root and r.n_iter == s.iterations, (c.id, kw)


# ---- the Arb roots


def ulps_from(x, lo, hi):
    """The ULP distance (bench.accuracy.common.ulp_distance) from x to the interval [lo, hi]."""
    return 0 if lo <= x <= hi else int(min(ulp_distance(x, lo), ulp_distance(x, hi)))


def arb_bracket(cid):
    r = ARB[cid]
    return (r["x"], r["x"]) if r["kind"] == "exact" else (r["a"], r["b"])


@pytest.mark.parametrize("m", BRACKETING)
def test_bracketing_roots_meet_the_tolerance_against_arb(m):
    # the contract: |x - x*| <= xtol + rtol |x| for the root x* (between two neighbouring doubles,
    # or exact), or f(x) == 0 in doubles (an underflow, e.g. Chandrupatla 7's x exp(-1/x^2))
    for c in fn.ROOT_CASES:
        r = sd.optimize.root_scalar(scalar(c.f), bracket=c.bracket, method=m)
        if not r.converged:
            continue
        lo, hi = arb_bracket(c.id)
        d = 0.0 if lo <= r.x <= hi else min(abs(r.x - lo), abs(r.x - hi))
        assert d <= 2e-12 + 4 * EPS * abs(r.x) or scalar(c.f)(r.x) == 0.0, (c.id, r.x, lo)


@pytest.mark.parametrize(
    "m,kw,limit,converged",
    [
        ("brentq", dict(xtol=0, max_iter=200), 3, 197),
        ("toms748", dict(xtol=0, rtol=EPS), 1, 199),
        ("bisect", dict(xtol=0, max_iter=1100), 4, 204),
    ],
)
def test_xtol_0_goes_to_the_last_bits(m, kw, limit, converged):
    # O33: the default tolerances are scipy's; xtol=0 (and rtol at its least) goes on to within a
    # few units in the last place of the Arb root (or an x where f is 0 in doubles)
    n = 0
    for c in fn.ROOT_CASES:
        r = sd.optimize.root_scalar(scalar(c.f), bracket=c.bracket, method=m, **kw)
        if not r.converged:
            continue
        n += 1
        lo, hi = arb_bracket(c.id)
        assert ulps_from(r.x, lo, hi) <= limit or scalar(c.f)(r.x) == 0.0, (c.id, r.x, lo)
    assert n == converged


# ---- counts, results


def test_counts_are_separate():
    f = lambda x: x**3 - 2.0
    r = sd.optimize.root_scalar(f, x0=1.0, fprime=lambda x: 3 * x * x)
    assert r.method == "newton" and r.nfev == r.njev == r.n_iter
    assert not hasattr(r, "nhev")
    r = sd.optimize.root_scalar(f, x0=1.0, fprime=lambda x: 3 * x * x, fprime2=lambda x: 6 * x)
    assert r.method == "halley" and r.njev == r.nhev == r.nfev - 1  # f(x) == 0 at the last x: no f'
    r = sd.optimize.root_scalar(f, x0=1.0)  # the difference: f twice per iteration, counted in nfev
    assert r.method == "newton" and r.nfev == 2 * r.n_iter and r.njev == r.n_iter
    r = sd.optimize.root_scalar(f, bracket=(0, 2))
    assert r.method == "brentq" and not hasattr(r, "njev") and not hasattr(r, "fun")
    r = sd.optimize.root_scalar(f, x0=1.0, x1=2.0)
    assert r.method == "secant" and not hasattr(r, "njev")


def test_method_choice_is_scipys():
    f = lambda x: x * x - 2
    d = lambda x: 2 * x
    assert sd.optimize.root_scalar(f, bracket=(0, 2)).method == "brentq"
    assert sd.optimize.root_scalar(f, x0=1.0).method == "newton"
    assert sd.optimize.root_scalar(f, x0=1.0, fprime=d).method == "newton"
    assert sd.optimize.root_scalar(f, x0=1.0, fprime=d, fprime2=lambda x: 2.0).method == "halley"
    assert sd.optimize.root_scalar(f, x0=1.0, x1=1.5).method == "secant"
    assert sd.optimize.minimize_scalar(lambda x: (x - 1) ** 2).method == "brent"
    assert sd.optimize.minimize_scalar(lambda x: (x - 1) ** 2, bounds=(0, 3)).method == "bounded"


def test_result_types():
    r = sd.optimize.root_scalar(lambda x: x - 0.5, bracket=(0, 1))
    assert type(r.x) is float and type(r.n_iter) is int and type(r.nfev) is int
    assert type(r.converged) is bool and type(r.status) is int and r.message == "converged"
    r = sd.optimize.minimize_scalar(lambda x: (x - 1) ** 2)
    assert type(r.x) is float and type(r.fun) is float and type(r.status) is int
    with pytest.raises(AttributeError, match="has no setter"):
        r.x = 1.0
    assert repr(r).startswith("OptimizeResult(method='brent', converged=True, fun=")


def test_f_gets_a_float_and_may_return_any_real():
    seen = []

    def f(x):
        seen.append(type(x))
        return np.float64(x - 0.25)

    assert sd.optimize.root_scalar(f, bracket=(0, 1)).x == 0.25
    assert set(seen) == {float}
    for v in (1, np.float32(1.5), np.array(2.0), np.int64(3)):
        r = sd.optimize.root_scalar(lambda x, v=v: v if x > 0.5 else -1, bracket=(0, 1), method="bisect")
        assert r.converged
    r = sd.optimize.minimize_scalar(lambda x: int(abs(x - 3)), bracket=(0, 1, 9))
    assert r.converged


# ---- edge cases (open points of docs/api/optimize.md §9: scipy's behaviour, provisionally)


@LIVE
def test_nan_in_the_bracket_stops_like_scipy():
    # brentq, toms748, bisect: a NaN from f stops at that point (scipy's _wrap_nan_raise; status -3
    # and its message); secant and newton go on with it to max_iter like scipy
    def f(x):
        return math.nan if 0.3 < x < 0.6 else x - 0.45

    for m in BRACKETING:
        s = scipy_root(f, bracket=(0.0, 1.0), method=m)
        r = sd.optimize.root_scalar(f, bracket=(0.0, 1.0), method=m)
        assert (r.x, r.nfev, r.converged, r.status) == (s.root, s.function_calls, False, -3)
        assert r.message == s.flag == f"The function value at x={r.x} is NaN; solver cannot continue."
    s = scipy_root(lambda x: math.nan, x0=1.0, x1=2.0, method="secant")
    r = sd.optimize.root_scalar(lambda x: math.nan, x0=1.0, x1=2.0, method="secant")
    assert (r.n_iter, r.nfev, r.converged, r.status) == (s.iterations, s.function_calls, False, -2)


def test_toms748_infinite_value_stops_like_nan():
    # O45: toms748 stops at an infinite value with status -3 (scipy raises ValueError), at a bracket
    # end or inside; brentq and bisect go on with it (they use its sign)
    f = lambda x: math.inf if x > 0.6 else x - 0.5
    r = sd.optimize.root_scalar(f, bracket=(0.0, 1.0), method="toms748")
    assert (r.converged, r.status, r.x, r.nfev) == (False, -3, 1.0, 2)
    assert r.message == "The function value at x=1.0 is inf; solver cannot continue."
    r = sd.optimize.root_scalar(lambda x: -math.inf, bracket=(0.0, 1.0), method="toms748")
    assert (r.status, r.x, r.nfev, r.message) == (
        -3,
        0.0,
        1,
        "The function value at x=0.0 is -inf; solver cannot continue.",
    )
    g = lambda x: math.inf if 0.4 < x < 0.6 else x - 0.45  # inside the bracket
    r = sd.optimize.root_scalar(g, bracket=(0.0, 1.0), method="toms748")
    assert r.status == -3 and 0.4 < r.x < 0.6 and r.message.endswith("is inf; solver cannot continue.")
    for m in ("brentq", "bisect"):
        assert sd.optimize.root_scalar(f, bracket=(0.0, 1.0), method=m).converged


def test_zero_at_an_end_and_zero_derivative():
    r = sd.optimize.root_scalar(lambda x: x - 1.0, bracket=(1.0, 2.0))
    assert (r.x, r.n_iter, r.nfev, r.converged) == (1.0, 0, 2, True)
    r = sd.optimize.root_scalar(lambda x: x - 1.0, bracket=(1.0, 2.0), method="toms748")
    assert (r.x, r.nfev) == (1.0, 1)  # toms748 stops before f(b)
    r = sd.optimize.root_scalar(lambda x: x * x - 1, x0=0.0, fprime=lambda x: 2 * x)
    assert (r.converged, r.status, r.message, r.x, r.n_iter) == (False, -2, "Derivative was zero.", 0.0, 1)


def test_reversed_bracket():
    # brentq and bisect take (b, a) like scipy's C code; toms748 needs a < b like scipy's
    f = lambda x: x * x - 2
    assert abs(sd.optimize.root_scalar(f, bracket=(2, 0)).x - math.sqrt(2)) < 1e-11
    assert abs(sd.optimize.root_scalar(f, bracket=(2, 0), method="bisect").x - math.sqrt(2)) < 1e-11
    with pytest.raises(ValueError, match=r"needs a < b in 'bracket', not \(2.0, 0.0\)"):
        sd.optimize.root_scalar(f, bracket=(2, 0), method="toms748")


@LIVE
def test_minimum_edge_cases_follow_scipy():
    warnings.simplefilter("ignore")
    # no bracket found (BracketError): scipy's result; the linear f overflows the search to inf
    for f, kw in ((lambda x: -x, {}), (lambda x: 1.0, {"bracket": (0.0, 1.0)}), (lambda x: x, {"bracket": (1.0, 1.0)})):
        for m in ("brent", "golden"):
            s = scipy_min(f, method=m, **kw)
            r = sd.optimize.minimize_scalar(f, method=m, **kw)
            assert (r.converged, r.status, r.message, r.nfev, r.n_iter) == (False, 3, s.message, s.nfev, 0)
            assert r.x == s.x or (math.isnan(r.x) and math.isnan(s.x))
    # NaN, max_iter
    s = scipy_min(lambda x: math.nan if x > 0.3 else (x - 0.2) ** 2, bounds=(0, 1), method="bounded")
    r = sd.optimize.minimize_scalar(lambda x: math.nan if x > 0.3 else (x - 0.2) ** 2, bounds=(0, 1))
    assert (r.status, r.message, r.nfev) == (s.status, s.message, s.nfev) == (2, "NaN result encountered.", s.nfev)
    s = scipy_min(lambda x: (x - 0.2) ** 2, method="brent", options={"maxiter": 3})
    r = sd.optimize.minimize_scalar(lambda x: (x - 0.2) ** 2, method="brent", options={"max_iter": 3})
    assert (r.status, r.converged, r.message, r.x) == (1, False, s.message, s.x)
    s = scipy_min(lambda x: (x - 0.2) ** 2, bounds=(0, 1), options={"maxiter": 4, "xatol": 1e-9})
    r = sd.optimize.minimize_scalar(lambda x: (x - 0.2) ** 2, bounds=(0, 1), options={"max_iter": 4, "xatol": 1e-9})
    assert (r.status, r.message, r.x, r.nfev) == (s.status, s.message, s.x, s.nfev)
    # equal bounds: scipy allows them (one evaluation)
    r = sd.optimize.minimize_scalar(lambda x: x * x, bounds=(2.0, 2.0))
    assert (r.x, r.nfev, r.converged) == (2.0, 1, True)


def test_tol_sets_the_methods_tolerance():
    f = lambda x: (x - 1.0 / 3.0) ** 2
    for m, key in (("brent", "xtol"), ("golden", "xtol")):
        a = sd.optimize.minimize_scalar(f, method=m, tol=1e-4)
        b = sd.optimize.minimize_scalar(f, method=m, options={key: 1e-4})
        assert (a.x, a.nfev) == (b.x, b.nfev) and a.message.endswith("(using xtol = 0.0001 )")
    a = sd.optimize.minimize_scalar(f, bounds=(0, 1), tol=1e-3)
    b = sd.optimize.minimize_scalar(f, bounds=(0, 1), options={"xatol": 1e-3})
    assert (a.x, a.nfev) == (b.x, b.nfev)


# ---- errors (docs/api/optimize.md §1.5)


def err(exc, msg, fn_, *a, **kw):
    with pytest.raises(exc) as e:
        fn_(*a, **kw)
    assert str(e.value) == msg, str(e.value)


def test_root_errors():
    rs = sd.optimize.root_scalar
    f = lambda x: x - 0.5
    err(TypeError, "root_scalar(): 'f' must be callable, not int", rs, 3, bracket=(0, 1))
    err(TypeError, "root_scalar(): 'fprime' must be callable, not bool", rs, f, x0=1.0, fprime=True)  # O34
    err(ValueError, "root_scalar(): give 'bracket' or 'x0'", rs, f)
    err(
        ValueError,
        "root_scalar(): 'method' must be lowercase: 'brentq', not 'Brentq'",
        rs,
        f,
        bracket=(0, 1),
        method="Brentq",
    )
    err(
        ValueError,
        "root_scalar(): 'method' must be 'brentq', 'toms748', 'bisect', 'newton', 'halley' or 'secant', not 'ridder'",
        rs,
        f,
        bracket=(0, 1),
        method="ridder",
    )
    err(ValueError, "root_scalar(): 'bracket' is required when method='bisect'", rs, f, x0=1.0, method="bisect")
    err(ValueError, "root_scalar(): 'x0' is required when method='newton'", rs, f, bracket=(0, 1), method="newton")
    err(ValueError, "root_scalar(): method='brentq' takes no 'x0'", rs, f, bracket=(0, 1), x0=1.0)
    err(ValueError, "root_scalar(): method='brentq' takes no 'fprime'", rs, f, bracket=(0, 1), fprime=f)
    err(ValueError, "root_scalar(): method='secant' takes no 'fprime'", rs, f, x0=1.0, fprime=f, method="secant")
    err(ValueError, "root_scalar(): method='newton' takes no 'x1'", rs, f, x0=1.0, x1=2.0, method="newton")
    err(ValueError, "root_scalar(): method='newton' takes no 'bracket'", rs, f, x0=1.0, bracket=(0, 1), method="newton")
    err(
        ValueError,
        "root_scalar(): method='newton' takes no 'fprime2' (method='halley' takes it)",
        rs,
        f,
        x0=1.0,
        fprime2=f,
    )
    err(
        ValueError,
        "root_scalar(): 'fprime2' is required when method='halley'",
        rs,
        f,
        x0=1.0,
        fprime=f,
        method="halley",
    )
    err(ValueError, "root_scalar(): 'bracket' must be a pair (a, b), not 3 values", rs, f, bracket=(0, 1, 2))
    err(TypeError, "root_scalar(): 'bracket' must be a pair (a, b), not float", rs, f, bracket=1.0)
    err(ValueError, "root_scalar(): the values of 'bracket' must be finite", rs, f, bracket=(0, math.inf))
    err(TypeError, "root_scalar(): the values of 'bracket' must be real numbers, not complex", rs, f, bracket=(0, 1j))
    err(ValueError, "root_scalar(): 'x0' must be finite", rs, f, x0=math.nan)
    err(TypeError, "root_scalar(): 'x0' must be an int or a float, not str", rs, f, x0="1")
    err(ValueError, "root_scalar(): 'x1' must differ from 'x0'", rs, f, x0=1.0, x1=1.0)
    err(ValueError, "root_scalar(): 'xtol' must be a finite number >= 0", rs, f, bracket=(0, 1), xtol=-1)
    err(
        ValueError,
        "root_scalar(): 'rtol' must be >= 8.9e-16 (4 times the machine epsilon) when method='brentq'",
        rs,
        f,
        bracket=(0, 1),
        rtol=1e-16,
    )
    err(
        ValueError,
        "root_scalar(): 'rtol' must be >= 2.2e-16 (the machine epsilon) when method='toms748'",
        rs,
        f,
        bracket=(0, 1),
        rtol=1e-17,
        method="toms748",
    )
    err(TypeError, "root_scalar(): 'max_iter' must be an int, not float", rs, f, bracket=(0, 1), max_iter=10.0)
    err(ValueError, "root_scalar(): 'max_iter' must be >= 1", rs, f, x0=1.0, max_iter=0)
    err(ValueError, "root_scalar(): method='brentq' has no option 'k'", rs, f, bracket=(0, 1), options={"k": 2})
    err(
        ValueError,
        "root_scalar(): method='toms748' has no option 'maxiter' (the iteration cap is 'max_iter')",
        rs,
        f,
        bracket=(0, 1),
        method="toms748",
        options={"maxiter": 2},
    )
    err(TypeError, "root_scalar(): 'options' must be a dict or None, not list", rs, f, bracket=(0, 1), options=[])
    err(ValueError, "root_scalar(): 'k' must be <= 100", rs, f, bracket=(0, 1), method="toms748", options={"k": 101})
    err(  # O32
        ValueError,
        "root_scalar(): 'bracket' does not enclose a root: f(a) and f(b) have the same sign "
        "(f(1.0) = 0.5, f(2.0) = 1.5)",
        rs,
        f,
        bracket=(1, 2),
    )
    for m in BRACKETING:
        with pytest.raises(ValueError, match="does not enclose a root"):
            rs(f, bracket=(1, 2), method=m)
    err(TypeError, "root_scalar(): 'f' must return a real number, not str", rs, lambda x: "a", bracket=(0, 1))
    err(TypeError, "root_scalar(): 'f' must return a real number, not complex", rs, lambda x: 1j, bracket=(0, 1))
    err(ValueError, "root_scalar(): 'f' must return a real number, not shape (1,)", rs, lambda x: [x], bracket=(0, 1))
    err(
        TypeError,
        "root_scalar(): 'fprime' must return a real number, not NoneType",
        rs,
        f,
        x0=1.0,
        fprime=lambda x: None,
    )


def test_minimize_errors():
    ms = sd.optimize.minimize_scalar
    f = lambda x: (x - 1) ** 2
    err(TypeError, "minimize_scalar(): 'fun' must be callable, not NoneType", ms, None)
    err(ValueError, "minimize_scalar(): 'method' must be lowercase: 'brent', not 'Brent'", ms, f, method="Brent")
    err(
        ValueError,
        "minimize_scalar(): method='brent' takes no 'bounds' (method='bounded' does)",
        ms,
        f,
        bounds=(0, 1),
        method="brent",
    )
    err(
        ValueError,
        "minimize_scalar(): method='bounded' takes no 'bracket' (it searches 'bounds')",
        ms,
        f,
        bracket=(0, 1),
        method="bounded",
    )
    err(ValueError, "minimize_scalar(): 'bounds' is required when method='bounded'", ms, f, method="bounded")
    err(ValueError, "minimize_scalar(): 'bounds' must be a pair (lower, upper), not 3 values", ms, f, bounds=(0, 1, 2))
    err(
        ValueError,
        "minimize_scalar(): 'bracket' must be a pair (xa, xb) or a triple (xa, xb, xc), not 1 value",
        ms,
        math.cos,
        bracket=(1,),
    )
    err(ValueError, "minimize_scalar(): the lower bound must not be greater than the upper bound", ms, f, bounds=(1, 0))
    err(ValueError, "minimize_scalar(): the values of 'bounds' must be finite", ms, f, bounds=(0, math.inf))
    err(
        ValueError,
        "minimize_scalar(): 'bracket' must be a pair (xa, xb) or a triple (xa, xb, xc), not 4 values",
        ms,
        f,
        bracket=(0, 1, 2, 3),
    )
    err(
        ValueError,
        "minimize_scalar(): 'bracket' must have xa < xb < xc (or xc < xb < xa), not (0.0, 3.0, 2.0)",
        ms,
        f,
        bracket=(0, 3, 2),
    )
    err(
        ValueError,
        "minimize_scalar(): 'bracket' must have f(xb) < f(xa) and f(xb) < f(xc), "
        "not f(xa) = 1.0, f(xb) = 0.0, f(xc) = 0.0",
        ms,
        lambda x: 0.0 if x > 0.5 else 1.0,
        bracket=(0, 1, 2),
    )
    err(
        ValueError,
        "minimize_scalar(): method='bounded' has no option 'xtol'",
        ms,
        f,
        bounds=(0, 1),
        options={"xtol": 1},
    )
    err(
        ValueError,
        "minimize_scalar(): method='golden' has no option 'disp'",
        ms,
        f,
        method="golden",
        options={"disp": 1},
    )
    err(
        ValueError,
        "minimize_scalar(): give 'tol' or options['xtol'], not both",
        ms,
        f,
        tol=1e-3,
        options={"xtol": 1e-3},
    )
    err(ValueError, "minimize_scalar(): 'tol' must be a finite number >= 0", ms, f, tol=-1.0)
    err(ValueError, "minimize_scalar(): 'xtol' must be a finite number >= 0", ms, f, options={"xtol": -1.0})
    # None means the default for every option (D-28); a conflict is between values given
    assert ms(f, tol=1e-6, options={"xtol": None}).converged
    assert ms(f, options={"max_iter": None, "xtol": None}).converged
    err(ValueError, "minimize_scalar(): 'max_iter' must be >= 0", ms, f, options={"max_iter": -1})
    err(ValueError, "minimize_scalar(): 'fun' must return a real number, not shape (2,)", ms, lambda x: [x, x])


def test_user_errors_pass_through():
    class Boom(Exception):
        pass

    def f(x):
        raise Boom("inside")

    for kw in (dict(bracket=(0, 1)), dict(bracket=(0, 1), method="toms748"), dict(x0=1.0), dict(x0=1.0, x1=2.0)):
        with pytest.raises(Boom, match="^inside$"):
            sd.optimize.root_scalar(f, **kw)
    for kw in (dict(), dict(bounds=(0, 1)), dict(method="golden")):
        with pytest.raises(Boom):
            sd.optimize.minimize_scalar(f, **kw)


# ---- bisect's signs by sign bits (SCALAR-1), the bracket's range (SCALAR-5)


def test_bisect_tiny_values():
    # the product fa * fb underflows to 0 below about 1e-162: the signs are read from the sign
    # bits, as scipy's bisect.c does
    rs = sd.optimize.root_scalar
    r = rs(lambda x: math.sin(x) * 1e-300, bracket=(2, 4), method="bisect")
    assert r.converged and abs(r.x - math.pi) < 1e-11
    assert r.x == scipy_root(lambda x: math.sin(x) * 1e-300, bracket=(2, 4), method="bisect").root
    r = rs(lambda x: (x - 3.3) * 1e-170, bracket=(2, 4), method="bisect")
    assert r.converged and abs(r.x - 3.3) < 1e-11
    with pytest.raises(ValueError, match=r"root_scalar\(\): 'bracket' does not enclose a root"):
        rs(lambda x: 1e-200, bracket=(2, 4), method="bisect")
    rng = np.random.default_rng(5)
    for _ in range(50):
        s, a = 10.0 ** float(rng.integers(-320, 0)), float(rng.uniform(-3, 3))
        lo, hi = a - float(rng.uniform(0.1, 4)), a + float(rng.uniform(0.1, 4))
        fx = lambda x, s=s, a=a: s * math.tanh(x - a)  # noqa: E731
        r = rs(fx, bracket=(lo, hi), method="bisect")
        assert r.converged and r.x == scipy_root(fx, bracket=(lo, hi), method="bisect").root


def test_bracket_out_of_range():
    with pytest.raises(OverflowError, match=r"^root_scalar\(\): 'bracket' is out of range$"):
        sd.optimize.root_scalar(lambda x: x, bracket=(0, 10**400))
    with pytest.raises(OverflowError, match=r"^minimize_scalar\(\): 'bounds' is out of range$"):
        sd.optimize.minimize_scalar(lambda x: x, bounds=(0, 10**400))
