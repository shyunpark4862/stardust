"""quad, dblquad, tplquad and nquad (docs/api/integrate.md §2, §3; docs/design/integrate.md §3.1,
§3.2, §7.10; decisions C5, C55-C66).

The calls and results against scipy 1.18.1's records (tests/integrate/data): every integral of
the collection without a weight calls the integrand at the same points in the same order and
gives the same integral, error, nfev, status, message and intervals to the bit; the multiple
integrals call at the same points and give the same values. Against the Arb references: a
converged result is within its error estimate but for the cases scipy shares (QUAD_SHORT). And
the options, the input forms, the results' fields and the errors."""

import hashlib
import math

import numpy as np
import pytest

import stardust as sd

from . import problems as P
from . import refdata
from .test_reference_data import QUAD_SHORT, live_scipy

I = sd.integrate  # noqa: E741
UNWEIGHTED = [p for p in P.INTEGRALS if p.weight is None]
WEIGHTED = [p for p in P.INTEGRALS if p.weight is not None]


def run(p, **kw):
    """quad on problem p with its break points, recording the points it calls."""
    pts = []

    def f(x):
        pts.append(x)
        return p.f(x)

    if p.points:
        kw["points"] = list(p.points)
    return I.quad(f, p.a, p.b, **kw), np.array(pts, dtype=np.float64)


def same(x, y):
    return x == y or (x != x and y != y)


# ---- scipy's records, one quad at a time


def test_collection():
    assert len(UNWEIGHTED) == 106 and len(WEIGHTED) == 10


@pytest.mark.parametrize("p", UNWEIGHTED, ids=lambda p: p.id)
def test_quad_matches_scipy_to_the_bit(p):
    rec = refdata.data("scipy_quad.json")[p.id]
    calls = np.load(refdata.DATA / "scipy_calls.npz")[f"quad/{p.id}"]
    r, pts = run(p)
    assert pts.size == calls.size and np.array_equal(pts.view(np.int64), calls.view(np.int64))
    assert same(r.integral, rec["integral"]) and same(r.error, rec["error"])
    assert r.nfev == rec["nfev"] == pts.size and r.status == rec["status"]
    assert r.converged is (rec["status"] == 0)
    if rec["status"]:
        assert r.message == rec["message"]
    info = rec["info"]
    last = info.get("last", 0)
    assert r.n_intervals == last and r.intervals.shape == (last, 2)
    stored = sorted(zip(info["alist"], info["blist"], info["rlist"], info["elist"])) if last else []
    rows = [tuple(r.intervals[i].tolist()) + (r.interval_integrals[i], r.interval_errors[i]) for i in range(last)]
    assert sorted(rows) == stored  # every interval with its integral and error
    for k, j in enumerate(info["iord"]):  # QUADPACK's ranking first (scipy's iord, 0-based)
        assert rows[k] == (info["alist"][j], info["blist"][j], info["rlist"][j], info["elist"][j])
    assert all(rows[k][3] >= rows[k + 1][3] for k in range(len(info["iord"]), last - 1))


def test_quad_partition_sums_to_the_integral():
    # when QUADPACK ends without extrapolation the integral is the intervals' sum, in creation order
    p = P.get("book/peak/5")
    r, _ = run(p)
    assert r.converged and abs(sum(r.interval_integrals.tolist()) - r.integral) <= 4 * r.error


def test_quad_routes():
    # QAGS: 21 per rule; QAGI: 15 per rule (30 on the real line); QAGP: 21 per first interval
    r = I.quad(math.exp, 0.0, 1.0)
    assert r.nfev == 21 and r.n_intervals == 1
    r = I.quad(lambda x: math.exp(-x), 0.0, math.inf)
    assert r.nfev == 135 and float(r.intervals[0, 1]) <= 1.0  # the intervals of t in (0, 1]
    r = I.quad(lambda x: math.exp(-x * x), -math.inf, math.inf)
    assert r.nfev == 270
    r = I.quad(lambda x: abs(x - 0.3), 0.0, 1.0, points=[0.3])
    assert r.nfev == 42 and r.n_intervals == 2 and r.error < 1e-14
    r = I.quad(lambda x: abs(x - 0.3), 0.0, 1.0, points=[])  # QAGP with no point (scipy's dispatch)
    assert r.nfev == I.quad(lambda x: abs(x - 0.3), 0.0, 1.0, points=[2.0, -1.0, 0.0, 1.0]).nfev


# ---- live scipy: options and cases past the records


def _scipy_quad(f, a, b, **kw):
    import warnings

    from scipy import integrate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return integrate.quad(f, a, b, full_output=1, **kw)


LIVE = [
    ("limit 3", lambda x: x**-0.5 * math.log(x) if x > 0 else 0.0, 0.0, 1.0, {"limit": 3}),
    ("limit 1", lambda x: math.sqrt(x), 0.0, 1.0, {"limit": 1}),
    ("limit 400", lambda x: math.cos(2**10 * math.sin(x)), 0.0, math.pi, {"limit": 400}),
    ("tight", lambda x: math.log(x) / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0, {"epsabs": 0.0, "epsrel": 1e-13}),
    ("loose", math.exp, 0.0, 10.0, {"epsabs": 1e-2, "epsrel": 0.0}),
    ("nan", lambda x: math.nan if x > 0.5 else 1.0, 0.0, 1.0, {}),
    ("inf", lambda x: math.inf if x > 0.9 else x, 0.0, 1.0, {}),
    ("left half line", lambda x: math.exp(x), -math.inf, 0.5, {}),
    ("reversed points", lambda x: abs(x - 0.25) ** 0.5, 1.0, 0.0, {"points": [0.25, 0.25, 7.0]}),
    ("zero", lambda x: 0.0, -1.0, 1.0, {}),
    ("step", lambda x: 1.0 if x <= 0 else 0.0, -1.0, 100.0, {}),
    ("many points", lambda x: math.floor(8 * x), 0.0, 1.0, {"points": [k / 8 for k in range(1, 8)], "limit": 8}),
]


@pytest.mark.parametrize("case", LIVE, ids=lambda c: c[0])
def test_quad_matches_live_scipy(case):
    live_scipy()
    _, f, a, b, kw = case
    calls_sd, calls_sp = [], []
    sp = _scipy_quad(lambda x: (calls_sp.append(x), f(x))[1], a, b, **kw)
    ours = {"atol": kw.get("epsabs", 1.49e-8), "rtol": kw.get("epsrel", 1.49e-8)}
    ours.update({k: v for k, v in kw.items() if k in ("limit", "points")})
    r = I.quad(lambda x: (calls_sd.append(x), f(x))[1], a, b, **ours)
    assert np.array_equal(np.array(calls_sd).view(np.int64), np.array(calls_sp).view(np.int64))
    assert same(r.integral, sp[0]) and same(r.error, sp[1]) and r.nfev == sp[2]["neval"]
    if len(sp) > 3:
        assert not r.converged and r.message == sp[3]
    else:
        assert r.converged


def test_complex_func_matches_scipy():
    live_scipy()
    f = lambda x: complex(math.cos(3 * x), math.exp(-x) * math.sqrt(x))  # noqa: E731
    sp = _scipy_quad(f, 0.0, 2.0, complex_func=True)
    r = I.quad(f, 0.0, 2.0, complex_func=True)
    assert r.integral == sp[0] and r.error == sp[1]
    assert r.nfev == sp[2]["real"][0]["neval"] + sp[2]["imag"][0]["neval"]
    assert isinstance(r.real, I.IntegrateResult) and r.real.integral == sp[0].real and r.imag.error == sp[1].imag
    assert r.real.n_intervals == sp[2]["real"][0]["last"] and r.converged and r.status == 0
    a, e = r
    assert isinstance(a, complex) and isinstance(e, complex)
    assert not hasattr(r, "intervals")  # the parts have them
    # a real value is a complex one with no imaginary part; numpy's complex and 0-d arrays too
    r = I.quad(lambda x: x, 0.0, 1.0, complex_func=True)
    assert r.integral == 0.5 + 0j and r.imag.integral == 0.0 and r.imag.nfev == 21
    r = I.quad(lambda x: np.complex128(1j * x), 0.0, 1.0, complex_func=True)
    assert r.integral == 0.5j
    r = I.quad(lambda x: np.array(1j * x), 0.0, 1.0, complex_func=True)
    assert r.integral == 0.5j


# ---- the Arb references


def test_quad_against_the_references():
    ref = refdata.integrals()
    short, n = set(), 0
    for p in UNWEIGHTED:
        r, _ = run(p)
        if r.converged:
            n += 1
            if abs(r.integral - refdata.value(ref[p.id])) > r.error:
                short.add(p.id)
    assert n == 94 and short == QUAD_SHORT


# ---- the multiple integrals


def _digest(points):
    h = hashlib.sha256()
    for x in points:
        h.update(np.asarray(x, dtype=np.float64).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("m", P.MULTI, ids=lambda m: m.id)
def test_multiple_match_scipy(m):
    rec = refdata.data("scipy_multiple.json")[m.id]
    pts = []

    def func(*xs):
        pts.append(xs)
        return float(m.func(*xs))

    r = (I.dblquad if m.kind == "dblquad" else I.tplquad)(func, m.a, m.b, *m.limits)
    want = rec[m.kind]
    assert r.integral == want["integral"] and r.error == want["error"] and r.nfev == want["nfev"] == len(pts)
    assert _digest(pts) == want["calls"]["sha256"] and r.converged and r.status == 0
    ref = refdata.value(refdata.multiple()[m.id])
    assert abs(r.integral - ref) <= max(r.error, 4e-16 * abs(ref))
    if m.box is not None:
        n = I.nquad(lambda *xs: float(m.func(*xs)), [list(b) for b in m.box])
        assert (n.integral, n.error, n.nfev) == (rec["nquad"]["integral"], rec["nquad"]["error"], rec["nquad"]["nfev"])


def _scipy_nquad(func, ranges, opts=None):
    import warnings

    from scipy import integrate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return integrate.nquad(func, ranges, opts=opts, full_output=True)


def test_nquad_functions_and_options_match_scipy():
    live_scipy()

    # scipy's docstring example: ranges and options as functions of the outer variables
    def func(x0, x1, x2, x3):
        return x0**2 + x1 * x2 - x3**3 + math.sin(x0) + (1 if (x0 - 0.2 * x3 - 0.5 - 0.25 * x1 > 0) else 0)

    def opts0(*args):
        return {"points": [0.2 * args[2] + 0.5 + 0.25 * args[0]]}

    ranges = [[0, 1], [-1, 1], [0.13, 0.8], [-0.15, 1]]
    seen_sd, seen_sp = [], []
    sp = _scipy_nquad(lambda *x: (seen_sp.append(x), func(*x))[1], ranges, [opts0, {}, {}, {}])
    r = I.nquad(lambda *x: (seen_sd.append(x), func(*x))[1], ranges, opts=[opts0, {}, {}, {}])
    assert seen_sd == seen_sp and (r.integral, r.error, r.nfev) == (sp[0], sp[1], sp[2]["neval"])

    def lim1(x2):
        return [x2, 2 * x2 + 0.5]

    def o1(x1, x2):  # the innermost level's options see the two outer variables
        return {"rtol": 1e-10, "limit": 20}

    seen_sd, seen_sp = [], []
    sp = _scipy_nquad(lambda *x: (seen_sp.append(x), math.cos(x[0] * x[1]) * x[2])[1], [[0, 1], lim1, [0, 2]],
                      [{"epsrel": 1e-10, "limit": 20}] + [{}] * 2)  # fmt: skip
    r = I.nquad(lambda *x: (seen_sd.append(x), math.cos(x[0] * x[1]) * x[2])[1], [[0, 1], lim1, [0, 2]],
                opts=[o1, {}, {}])  # fmt: skip
    assert seen_sd == seen_sp and (r.integral, r.error, r.nfev) == (sp[0], sp[1], sp[2]["neval"])


def test_nquad_error_is_the_largest_of_every_quad():
    # scipy's _NQuad keeps max(self.abserr, abserr) over the inner quads too
    live_scipy()
    f = lambda y, x: math.sqrt(abs(x - y))  # noqa: E731
    sp = _scipy_nquad(f, [[0, 1], [0, 1]])
    r = I.nquad(f, [[0, 1], [0, 1]])
    assert (r.integral, r.error, r.nfev) == (sp[0], sp[1], sp[2]["neval"])
    outer = I.quad(lambda x: I.quad(lambda y: f(y, x), 0.0, 1.0).integral, 0.0, 1.0)
    assert r.error > outer.error


def test_nested_status_is_the_first_failure():
    # an inner quad that hits the limit: converged False, its status and message (the inner
    # quads finish first)
    r = I.nquad(lambda y, x: math.cos(2**10 * math.sin(y)) + x, [[0, math.pi], [0, 1]])
    assert not r.converged and r.status == 1 and r.message.startswith("The maximum number of subdivisions (50)")
    r = I.dblquad(lambda y, x: 1.0, 0.0, 1.0, 0.0, 1.0)
    assert r.converged and r.status == 0 and r.nfev == 441 and r.message.startswith("Normal")


def test_dblquad_tplquad_bounds():
    # gfun ... rfun: numbers or functions; tplquad's qfun(x, y) (scipy's argument order)
    seen = []

    def q(x, y):
        seen.append((x, y))
        return 0.0

    r = I.tplquad(lambda z, y, x: 1.0, 0.0, 1.0, 0.0, lambda x: x, q, 1.0)
    assert abs(r.integral - 0.5) < 1e-14 and all(y <= x for x, y in seen)
    assert I.dblquad(lambda y, x: 1.0, 0.0, 2.0, np.float64(0.0), lambda x: np.float64(x)).integral == 2.0
    assert I.dblquad(lambda y, x: 1.0, 0, 1, 0, 3).integral == 3.0


# ---- the result


def test_result_fields_and_unpacking():
    r = I.quad(math.exp, 0.0, 1.0)
    v, e = r
    assert (v, e) == (r.integral, r.error) and r[0] == v and len(r) == 2
    assert type(r.integral) is float and type(r.nfev) is int and type(r.status) is int
    assert type(r.n_intervals) is int and r.converged is True
    for name in ("intervals", "interval_integrals", "interval_errors"):
        a = getattr(r, name)
        assert isinstance(a, sd.ndarray) and a.dtype == sd.float64 and getattr(r, name) is a
        with pytest.raises(ValueError):
            a[0] = 0.0
    assert r.intervals.tolist() == [[0.0, 1.0]]
    for name in ("success", "neval", "abserr", "last", "alist", "real", "max_level"):
        assert not hasattr(r, name)
    n = I.nquad(lambda x: x, [[0, 1]])
    assert tuple(n) == (n.integral, n.error) and not hasattr(n, "intervals")


def test_empty_and_reversed():
    calls = []
    r = I.quad(lambda x: calls.append(x) or 1.0, 2.0, 2.0)
    assert tuple(r) == (0.0, 0.0) and r.nfev == 0 and r.n_intervals == 0 and calls == [] and r.converged
    assert r.intervals.shape == (0, 2)
    assert I.quad(math.exp, math.inf, math.inf).integral == 0.0
    a, b = I.quad(math.exp, 0.0, 1.0), I.quad(math.exp, 1.0, 0.0)
    assert b.integral == -a.integral and b.error == a.error
    g = lambda x: math.exp(-x * x)  # noqa: E731
    assert I.quad(g, math.inf, -math.inf).integral == -I.quad(g, -math.inf, math.inf).integral


def test_summary_and_repr():
    r = I.quad(math.exp, 0.0, 1.0)
    assert repr(r) == "IntegrateResult(integral=1.71828, error=1.90768e-14, converged=True, nfev=21)"
    s = r.summary()
    assert s.splitlines()[0].strip() == "Integration Results"
    for text in ("Converged:", "True", "Integral:", "1.71828", "Error estimate:", "1.908e-14", "Function Evals:",
                 "Intervals:", "Message:", "Normal and reliable"):  # fmt: skip
        assert text in s, text
    assert ["a", "b", "integral", "error"] in [line.split() for line in s.splitlines()]
    r = I.quad(lambda x: 4.0**-12 / ((x - math.pi / 4) ** 2 + 16.0**-12), 0.0, 1.0)
    s = r.summary()
    assert "The integral is probably divergent, or slowly convergent." in s and "[13]" in s
    lines = s.splitlines()
    assert lines[-15].startswith("[0]") and lines[-2].startswith("[13]") and lines[-1].startswith("=")
    c = I.quad(lambda x: complex(x, 1.0), 0.0, 1.0, complex_func=True)
    assert repr(c).startswith("IntegrateResult(integral=(0.5+1j), error=(")
    s = c.summary()
    assert "(0.5+1j)" in s and "real" in s and "imag" in s
    assert "Integration Results" in I.dblquad(lambda y, x: 1.0, 0, 1, 0, 1).summary()


def test_long_message_breaks_kept():
    # scipy's "\n  " breaks are kept in the message (O86); summary() joins them into one line
    r = I.quad(lambda x: math.cos(2**9 * math.sin(x)), 0.0, math.pi)
    assert r.status == 1 and "\n  " in r.message
    assert "\n  If increasing" not in r.summary()


def test_large_limit_is_cheap():
    r = I.quad(math.exp, 0.0, 1.0, limit=2**31 - 1)
    assert r.converged and r.nfev == 21


def test_user_errors_propagate():
    def f(x):
        raise ZeroDivisionError("mine")

    with pytest.raises(ZeroDivisionError, match="mine"):
        I.quad(f, 0.0, 1.0)
    with pytest.raises(ZeroDivisionError, match="mine"):
        I.dblquad(lambda y, x: f(x), 0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ZeroDivisionError, match="mine"):
        I.dblquad(lambda y, x: 1.0, 0.0, 1.0, 0.0, f)


# ---- errors


ERRORS = [
    (lambda: I.quad(1.0, 0, 1), TypeError, "quad(): 'func' must be callable, not float"),
    (lambda: I.quad(math.exp, math.nan, 1), ValueError, "quad(): 'a' must be a number or +-inf, not NaN"),
    (lambda: I.quad(math.exp, 0, "1"), TypeError, "quad(): 'b' must be an int or a float, not str"),
    (lambda: I.quad(math.exp, 0, 1, epsabs=1e-3), TypeError, "quad(): unexpected keyword argument 'epsabs'"),
    (lambda: I.quad(math.exp, 0, 1, args=(1,)), TypeError, "quad(): unexpected keyword argument 'args'"),
    (lambda: I.quad(math.exp, 0, 1, full_output=1), TypeError, "quad(): unexpected keyword argument 'full_output'"),
    (lambda: I.quad(math.exp, 0, 1, atol=-1.0), ValueError, "quad(): 'atol' must be a finite number >= 0"),
    (lambda: I.quad(math.exp, 0, 1, rtol=math.inf), ValueError, "quad(): 'rtol' must be a finite number >= 0"),
    (lambda: I.quad(math.exp, 0, 1, atol=0.0, rtol=1e-15), ValueError,
     "quad(): if 'atol' is 0, 'rtol' must be at least 50 * machine epsilon (1.11e-14)"),
    (lambda: I.quad(math.exp, 0, 1, limit=0), ValueError, "quad(): 'limit' must be >= 1"),
    (lambda: I.quad(math.exp, 0, 1, limit=2.0), TypeError, "quad(): 'limit' must be an int, not float"),
    (lambda: I.quad(math.exp, 0, 1, limit=2**31), OverflowError, "quad(): 'limit' is out of range"),
    (lambda: I.quad(math.exp, 0, 1, points=0.5), TypeError,
     "quad(): 'points' must be a sequence of numbers or None, not float"),
    (lambda: I.quad(math.exp, 0, 1, points=[math.nan]), ValueError, "quad(): the values of 'points' must not be NaN"),
    (lambda: I.quad(math.exp, 0, math.inf, points=[1.0]), ValueError,
     "quad(): 'points' cannot be used with an infinite range"),
    (lambda: I.quad(math.exp, 0, 1, points=[0.1, 0.2, 0.3], limit=3), ValueError,
     "quad(): the number of break points (3) must be less than 'limit' (3)"),
    (lambda: I.quad(math.exp, 0, 1, weight="cos", wvar=1.0), NotImplementedError,
     "quad(): 'weight' (QAWO, QAWF, QAWS, QAWC) comes with milestone C5"),
    (lambda: I.quad(math.exp, 0, 1, weight="tan"), ValueError, "quad(): 'weight' must be one of 'cos', 'sin'"),
    (lambda: I.quad(math.exp, 0, 1, wvar=1.0), ValueError, "quad(): 'wvar' is only used with 'weight'"),
    (lambda: I.quad(math.exp, 0, 1, max_p1=10), ValueError, "quad(): 'max_p1' is only used with weight='cos' or 'sin'"),
    (lambda: I.quad(math.exp, 0, 1, limlst=2), ValueError, "quad(): 'limlst' must be >= 3"),
    (lambda: I.quad(math.exp, 0, 1, complex_func=1), TypeError,
     "quad(): 'complex_func' must be True or False, not int"),
    (lambda: I.quad(lambda x: [1.0, 2.0], 0, 1), ValueError,
     "quad(): 'func' must return a real number, not shape (2,)"),
    (lambda: I.quad(lambda x: 1j, 0, 1), TypeError, "quad(): 'func' must return a real number, not complex"),
    (lambda: I.quad(lambda x: True, 0, 1), TypeError, "quad(): 'func' must return a real number, not bool"),
    (lambda: I.quad(lambda x: "a", 0, 1, complex_func=True), TypeError,
     "quad(): 'func' must return a real number, not str"),
    (lambda: I.dblquad(lambda y, x: 1.0, 0, 1, 0, lambda x: math.nan), ValueError, "dblquad(): 'hfun' returned NaN"),
    (lambda: I.dblquad(lambda y, x: 1.0, 0, 1, "0", 1), TypeError, "dblquad(): 'gfun' must return a real number"),
    (lambda: I.dblquad(lambda y, x: 1j, 0, 1, 0, 1), TypeError,
     "dblquad(): 'func' must return a real number, not complex"),
    (lambda: I.dblquad(lambda y, x: 1.0, 0, 1, 0, 1, epsabs=1e-3), TypeError,
     "dblquad(): unexpected keyword argument 'epsabs'"),
    (lambda: I.tplquad(lambda z, y, x: 1.0, 0, 1, 0, 1, 0, lambda x, y: [1, 2]), ValueError,
     "tplquad(): 'rfun' must return a real number, not shape (2,)"),
    (lambda: I.nquad(lambda x: x, []), ValueError, "nquad(): 'ranges' must have at least one range"),
    (lambda: I.nquad(lambda x: x, [[0, 1, 2]]), ValueError, "nquad(): 'ranges[0]' must be a pair (lo, hi) of numbers"),
    (lambda: I.nquad(lambda x: x, [[0, math.nan]]), ValueError, "nquad(): 'ranges[0]' must not be NaN"),
    (lambda: I.nquad(lambda x: x, [[0, 1]], opts={"epsabs": 1e-3}), ValueError,
     "nquad(): 'opts' has no option 'epsabs' (scipy's 'epsabs' is 'atol')"),
    (lambda: I.nquad(lambda x, y: x, [[0, 1], [0, 1]], opts=[{}, {"limit": 0}]), ValueError,
     "nquad(): 'limit' must be >= 1"),
    (lambda: I.nquad(lambda x, y: x, [[0, 1], [0, 1]], opts=[{}]), ValueError,
     "nquad(): 'opts' must have one entry per range (2), not 1"),
    (lambda: I.nquad(lambda x, y: x, [[0, 1], lambda: 3.0]), TypeError,
     "nquad(): the range of 'ranges[1]' must be a pair (lo, hi) of numbers, not float"),
    (lambda: I.nquad(lambda x, y: x, [[0, 1], [0, 1]], opts=[lambda y: {"rtol": -1.0}, {}]), ValueError,
     "nquad(): 'rtol' must be a finite number >= 0"),
    (lambda: I.nquad(lambda x, y: x, [[0, 1], [0, 1]], opts=[lambda y: 3, {}]), TypeError,
     "nquad(): the options of 'opts[0]' must be a dict, not int"),
    (lambda: I.nquad(lambda x: x, [[0, math.inf]], opts={"points": []}), ValueError,
     "nquad(): 'points' cannot be used with an infinite range"),
    (lambda: I.nquad(lambda x: x, [[0, 1]], opts={"weight": "sin", "wvar": 2.0}), NotImplementedError,
     "nquad(): 'weight' (QAWO, QAWF, QAWS, QAWC) comes with milestone C5"),
    (lambda: I.nquad(lambda x: x, [[0, 1]], args=(1,)), TypeError, "nquad(): unexpected keyword argument 'args'"),
]  # fmt: skip


@pytest.mark.parametrize("call, kind, text", ERRORS, ids=[e[2][:60] for e in ERRORS])
def test_errors(call, kind, text):
    with pytest.raises(kind) as e:
        call()
    assert str(e.value).startswith(text), str(e.value)


def test_weighted_problems_wait_for_c5():
    for p in WEIGHTED:
        with pytest.raises(NotImplementedError):
            I.quad(p.f, p.a, p.b, weight=p.weight, wvar=p.wvar)
