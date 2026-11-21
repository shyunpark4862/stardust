"""tanhsinh and nsum (docs/api/integrate.md §6; docs/design/integrate.md §3.5, §3.6, §7.12;
decisions C4, C5, C84-C93).

The calls and results against scipy 1.18.1's records (tests/integrate/data): every integral of
the collection's tanhsinh records, the three batches and every series call the function at the
same points in the same order (the sha256 of the points' bytes) and give the same integral (sum),
error, nfev, status and max_level to the bit. The same against the live scipy on the cases the
records do not cover (log=True, the levels, the limits' transforms, non-finite values, nsum's
branches). The table is scipy's to the bit and its distance from the exact values is measured
(C84). Against the Arb references: a converged result is within its error estimate but for the
cases scipy shares (TANHSINH_SHORT). And the call form, the threads, the inputs and the
results."""

import hashlib
import math
import warnings

import numpy as np
import pytest

import stardust as sd

from . import problems as P
from . import refdata
from .test_reference_data import TANHSINH_SHORT, live_scipy

I = sd.integrate  # noqa: E741
INF = math.inf
TANHSINH = [p for p in P.INTEGRALS if p.tanhsinh]


class Recorder:
    """The points f is called with, as tools/gen_integrate_scipy.py records scipy's: the shapes,
    the number of points and the sha256 of their float64 bytes in call order; fn on numpy arrays."""

    def __init__(self, fn):
        self.fn, self.shapes, self.count, self.h = fn, [], 0, hashlib.sha256()

    def __call__(self, x, *args):
        a = np.array(x, dtype=np.float64)
        self.shapes.append(tuple(a.shape))
        self.count += a.size
        self.h.update(a.tobytes())
        with np.errstate(all="ignore"):
            v = self.fn(a, *[np.asarray(t) for t in args])
        return sd.asarray(np.asarray(v, dtype=np.float64))


def _scipy_shapes(rec):
    """scipy's call shapes, one entry per call ((): sd's first call is 1-D, O48)."""
    out = []
    for shape, times in rec["shapes"]:
        out += [tuple(shape) or (1,)] * times
    return out


def same(x, y):
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    return x.shape == y.shape and bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))))


# ---- scipy's records


def test_collection():
    assert len(TANHSINH) == 101 and len(P.BATCHES) == 3 and len(P.SERIES) == 9


@pytest.mark.parametrize("p", TANHSINH, ids=lambda p: p.id)
def test_tanhsinh_matches_scipy_to_the_bit(p):
    rec = refdata.data("scipy_tanhsinh.json")[p.id]
    f = Recorder(p.f_np)
    r = I.tanhsinh(f, p.a, p.b)
    calls = rec["calls"]
    assert f.h.hexdigest() == calls["sha256"] and f.count == calls["count"] == r.nfev
    assert f.shapes == _scipy_shapes(calls)
    assert same(r.integral, rec["integral"]) and same(r.error, rec["error"])
    assert (r.nfev, r.status, r.max_level, r.converged) == (
        rec["nfev"],
        rec["status"],
        rec["max_level"],
        rec["converged"],
    )


@pytest.mark.parametrize("bt", P.BATCHES, ids=lambda b: b.family)
def test_tanhsinh_batches_match_scipy(bt):
    rec = refdata.data("scipy_tanhsinh.json")[f"batch/{bt.family}"]
    f = Recorder(bt.f_np)
    r = I.tanhsinh(f, bt.a, bt.b, args=(list(bt.params),))
    assert f.h.hexdigest() == rec["calls"]["sha256"] and f.shapes == _scipy_shapes(rec["calls"])
    for k in ("integral", "error", "nfev", "status", "max_level", "converged"):
        assert same(np.asarray(getattr(r, k)), rec[k]), k
    # the batch's elements are the scalar runs, field by field
    one = refdata.data("scipy_tanhsinh.json")
    for i, pid in enumerate(bt.ids()):
        assert same(r.integral[i], one[pid]["integral"]) and r.nfev[i] == one[pid]["nfev"], pid


@pytest.mark.parametrize("s", P.SERIES, ids=lambda s: s.id)
def test_nsum_matches_scipy_to_the_bit(s):
    rec = refdata.data("scipy_nsum.json")[s.id]
    f = Recorder(s.f_np)
    r = I.nsum(f, s.a, s.b, step=s.step)
    calls = rec["calls"]
    assert f.h.hexdigest() == calls["sha256"] and f.count == calls["count"]
    assert f.shapes == _scipy_shapes(calls)
    assert same(r.sum, rec["sum"]) and same(r.error, rec["error"])
    assert (r.nfev, r.status, r.converged) == (rec["nfev"], rec["status"], rec["converged"])


# ---- the live scipy on the cases outside the records


def _scipy_and_sd(fn):
    """A pair of recorders of the same numpy integrand, scipy's (numpy in, numpy out) and sd's."""

    class NpRecorder(Recorder):
        def __call__(self, x, *args):
            a = np.array(x, dtype=np.float64)
            self.shapes.append((a.size,) if not self.shapes else tuple(a.shape))  # sd's first call is (k,)
            self.count += a.size
            self.h.update(a.tobytes())
            with np.errstate(all="ignore"):
                return self.fn(a, *args)

    return NpRecorder(fn), Recorder(fn)


def _quiet(call):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return call()


_EXP2 = lambda x: np.exp(-x * x)  # noqa: E731
_XLOG = lambda x: np.sqrt(x) * np.log(x)  # noqa: E731
_LOG_XLOG = lambda x: 0.5 * np.log(x) + np.log(-np.log(x))  # noqa: E731
TANHSINH_LIVE = [
    ("gauss on the real line", _EXP2, -INF, INF, {}),
    ("gauss to 1000", _EXP2, -INF, 1000.0, {}),
    ("gauss far tail rtol", _EXP2, 20.0, 30.0, {"rtol": 1e-10}),
    ("log gauss far tail", lambda x: -x * x, 20.0, 30.0, {"log": True, "rtol": math.log(1e-10)}),
    ("log gauss real line", lambda x: -x * x, -INF, INF, {"log": True}),
    ("log gauss right", lambda x: -x * x, 0.0, INF, {"log": True}),
    ("log gauss left", lambda x: -x * x, -INF, 0.0, {"log": True}),
    ("log x^-1/2", lambda x: -0.5 * np.log(x), 0.0, 1.0, {"log": True}),
    ("log equal limits", lambda x: -x * x, 1.0, 1.0, {"log": True}),
    ("log zero integrand", lambda x: np.full_like(x, -np.inf), 0.0, 1.0, {"log": True}),
    *[(f"min_level {m}", _XLOG, 0.0, 1.0, {"min_level": m}) for m in (0, 1, 3, 5)],
    *[(f"log min_level {m}", _LOG_XLOG, 0.0, 1.0, {"min_level": m, "log": True}) for m in (0, 1, 3)],
    *[(f"max_level {m}", lambda x: 1 / (x * x + 1e-6), -1.0, 1.0, {"max_level": m}) for m in (0, 1, 2, 3, 12)],
    ("min_level above max_level", np.cos, 0.0, 3.0, {"min_level": 5, "max_level": 3}),
    ("reversed", np.exp, 2.0, -1.0, {}),
    ("reversed infinite", _EXP2, INF, -INF, {}),
    ("equal", np.exp, 2.0, 2.0, {}),
    ("equal infinite", np.exp, INF, INF, {}),
    ("NaN limit", np.exp, math.nan, 1.0, {}),
    ("NaN limit and inf", np.exp, math.nan, INF, {}),
    ("NaN at the midpoint", lambda x: np.where(np.abs(x - 0.5) < 1e-12, np.nan, x), 0.0, 1.0, {}),
    ("singular end", lambda x: 1 / np.sqrt(x), 0.0, 1.0, {}),
    ("inf everywhere", lambda x: np.full_like(x, np.inf), 0.0, 1.0, {}),
    ("NaN everywhere", lambda x: np.full_like(x, np.nan), 0.0, 1.0, {}),
    ("atol", np.cos, 0.0, 10.0, {"atol": 1e-3}),
    ("atol and rtol 0", np.cos, 0.0, 10.0, {"atol": 1e-12, "rtol": 0.0}),
    ("zero integrand", lambda x: 0 * x, 0.0, 1.0, {}),
    ("oscillation", lambda x: np.cos(64 * np.sin(x)), 0.0, math.pi, {}),
]
TANHSINH_LIVE_BATCHES = [
    ("args", lambda x, c: np.sin(c * x), 0.0, 1.0, ([1.0, 10.0, 30.0, 100.0],), {"min_level": 1}),
    ("limits", lambda x: np.exp(-x), [0.0, 1.0, -INF, 2.0, 5.0], [1.0, INF, 0.0, 2.0, -3.0], (), {}),
    ("2-d", lambda x, c: x**c, 0.0, [[1.0], [2.0]], ([0.5, 1.5, -0.5],), {}),
    ("log", lambda x, c: c * np.log(x), 0.0, 1.0, ([0.5, 1.5, -0.5, 3.0],), {"log": True}),
    ("large", lambda x, c: np.exp(-c * x * x), -1.0, 2.0, (np.linspace(0.1, 50.0, 2000),), {}),
]


def _scipy_tanhsinh_kwargs(kw):
    names = {"max_level": "maxlevel", "min_level": "minlevel"}
    return {names.get(k, k): v for k, v in kw.items()}


def _check_tanhsinh(fn, a, b, args, kw):
    from scipy import integrate

    fs, fd = _scipy_and_sd(fn)
    rs = _quiet(lambda: integrate.tanhsinh(fs, a, b, args=tuple(np.asarray(t) for t in args),
                                           **_scipy_tanhsinh_kwargs(kw)))  # fmt: skip
    rd = I.tanhsinh(fd, a, b, args=args, **kw)
    assert fs.h.hexdigest() == fd.h.hexdigest() and fs.shapes == fd.shapes
    assert same(rs.integral, rd.integral) and same(rs.error, rd.error)
    assert same(rs.nfev, rd.nfev) and same(rs.status, rd.status) and same(rs.maxlevel, rd.max_level)


@pytest.mark.parametrize("case", TANHSINH_LIVE, ids=lambda c: c[0])
def test_tanhsinh_matches_live_scipy(case):
    live_scipy()
    _, fn, a, b, kw = case
    _check_tanhsinh(fn, a, b, (), kw)


@pytest.mark.parametrize("case", TANHSINH_LIVE_BATCHES, ids=lambda c: c[0])
def test_tanhsinh_batches_match_live_scipy(case):
    live_scipy()
    _, fn, a, b, args, kw = case
    _check_tanhsinh(fn, a, b, args, kw)


_ZETA2 = lambda k: 1 / k**2  # noqa: E731
NSUM_LIVE = [
    ("zeta(p) batch", lambda k, p: 1 / k**p, 1.0, INF, (np.arange(3.0, 10.0),), {"max_terms": 1000}),
    ("alternating harmonic", lambda x: 1 / x - 1 / (x + 1), 1.0, INF, (), {"step": 2.0}),
    ("finite, many terms", _ZETA2, 1.0, 1e7, (), {}),
    ("log finite, many terms", lambda k: -2 * np.log(k), 1.0, 1e7, (), {"log": True}),
    ("log zeta(2)", lambda k: -2 * np.log(k), 1.0, INF, (), {"log": True}),
    ("log direct", lambda k: -2 * np.log(k), 1.0, 100.0, (), {"log": True}),
    ("log both sides", lambda k: -k * k, -INF, INF, (), {"log": True}),
    ("log left", lambda k: -2 * np.log(-k), -INF, -1.0, (), {"log": True}),
    ("log tolerances", lambda k: -2 * np.log(k), 1.0, INF, (), {"log": True, "atol": math.log(1e-4), "rtol": -INF}),
    ("both sides", lambda k: np.exp(-k * k / 10), -INF, INF, (), {}),
    ("left", _ZETA2, -INF, -1.0, (), {}),
    ("negative step", _ZETA2, 1.0, 10.0, (), {"step": -1.0}),
    ("b below a", _ZETA2, 10.0, 1.0, (), {}),
    ("b below a, negative step", _ZETA2, 10.0, 1.0, (), {"step": -1.0}),
    ("NaN limit", _ZETA2, math.nan, 10.0, (), {}),
    *[(f"max_terms {m}", _ZETA2, 1.0, INF, (), {"max_terms": m}) for m in (0, 1, 1000)],
    ("short of the tolerance (-4)", lambda k: 1 / k**1.01, 1.0, INF, (), {"max_terms": 100}),
    ("divergent", lambda k: 1 / k, 1.0, INF, (), {}),
    ("infinite terms", lambda k: np.where(k < 5, np.inf, 1 / k**2), 1.0, INF, (), {}),
    ("mixed branches", lambda k, c: np.exp(-c * k * k), [1.0, -INF, 0.0, -INF, 3.0, 2.0],
     [INF, 0.0, 50.0, INF, 2.0, 2e7], ([0.1, 0.2, 0.3, 0.01, 1.0, 1e-12],), {}),
    ("atol", _ZETA2, 1.0, INF, (), {"atol": 1e-3}),
    ("rtol", lambda k: 1 / k**3, 1.0, INF, (), {"rtol": 1e-12}),
    ("unimodal", lambda k: k**3 * np.exp(-k / 5), 0.0, INF, (), {}),
    ("unimodal far from a", lambda k: np.exp(-((k - 3000) ** 2) / 100), 0.0, INF, (), {}),
    ("a = b = -inf", lambda k: np.exp(-k * k), -INF, -INF, (), {}),
    ("a = b = inf", lambda k: np.exp(-k * k), INF, INF, (), {}),
    ("a = b", lambda k: np.exp(-k * k), 2.0, 2.0, (), {}),
]  # fmt: skip


@pytest.mark.parametrize("case", NSUM_LIVE, ids=lambda c: c[0])
def test_nsum_matches_live_scipy(case):
    from scipy import integrate

    live_scipy()
    _, fn, a, b, args, kw = case
    fs, fd = _scipy_and_sd(fn)
    skw = {"maxterms" if k == "max_terms" else k: v for k, v in kw.items() if k not in ("atol", "rtol")}
    tol = {k: kw[k] for k in ("atol", "rtol") if k in kw}
    rs = _quiet(lambda: integrate.nsum(fs, a, b, args=tuple(np.asarray(t) for t in args), tolerances=tol or None,
                                       **skw))  # fmt: skip
    rd = I.nsum(fd, a, b, args=args, **kw)
    assert fs.h.hexdigest() == fd.h.hexdigest() and fs.shapes == fd.shapes
    assert same(rs.sum, rd.sum) and same(rs.error, rd.error)
    assert same(rs.nfev, rd.nfev) and same(rs.status, rd.status)


def test_nsum_log_with_both_limits_minus_inf():
    # scipy sums on both sides from 0 to -inf, a reversed integral whose log is complex; with real
    # values only that integral is NaN, and the sum runs on to the level limit (C85)
    r = I.nsum(lambda k: -k * k, -INF, -INF, log=True)
    assert r.status == -2 and math.isnan(r.sum)


# ---- the table (C84)


def test_table_is_scipys():
    live_scipy()
    from scipy.integrate import _tanhsinh as T

    xjc, wj, h0 = sd._core.integrate.tanhsinh_table(10)
    assert h0 == float(T._get_base_step(np.float64, np))
    sx = np.concatenate([T._compute_pair(k, np.float64(h0), np)[0] for k in range(11)])
    sw = np.concatenate([T._compute_pair(k, np.float64(h0), np)[1] for k in range(11)])
    assert xjc.shape == (8193,) and np.array_equal(np.asarray(xjc).view(np.int64), sx.view(np.int64))
    assert np.array_equal(np.asarray(wj).view(np.int64), sw.view(np.int64))
    x16, w16, _ = sd._core.integrate.tanhsinh_table(16)
    assert x16.shape == (9 + 4 * (2**17 - 2),) and np.array_equal(np.asarray(x16)[:8193], np.asarray(xjc))


def test_table_against_the_exact_formula():
    # docs/design/integrate.md §7.12: the same formula in Arb on the same double jh (j * h
    # rounded) is 0-980 ULP away (47 on average), on the exact j h of the double h 0-3449 (161):
    # the rounding of u2 grows u2-fold in exp(u2)
    from flint import arb, ctx

    xjc, wj, h0 = (np.asarray(v) if not isinstance(v, float) else v for v in sd._core.integrate.tanhsinh_table(10))
    old = ctx.prec
    ctx.prec = 200
    try:
        pi2 = arb.pi() / 2
        on_double, on_exact = [], []
        q = 0
        for k in range(11):
            h = h0 / 2**k
            for j in range(0, 9) if k == 0 else range(1, 8 * 2**k + 1, 2):
                for jh, out in ((arb(float(j) * h), on_double), (arb(j) * arb(h), on_exact)):
                    u1, u2 = pi2 * jh.cosh(), pi2 * jh.sinh()
                    c = u2.cosh()
                    w = u1 / (c * c) / (2 if q == 0 else 1)
                    x = 1 / (u2.exp() * c)
                    for got, want in ((xjc[q], x), (wj[q], w)):
                        out.append(abs(float((arb(float(got)) - want).mid())) / math.ulp(float(want.mid())))
                q += 1
    finally:
        ctx.prec = old
    on_double, on_exact = np.array(on_double), np.array(on_exact)
    assert 979 < on_double.max() < 980 and 46 < on_double.mean() < 47.5
    assert 3449 < on_exact.max() < 3450 and 160 < on_exact.mean() < 162


# ---- the Arb references (docs/design/integrate.md §4)


def test_converged_results_are_within_their_error_estimate():
    ref = refdata.integrals()
    short, rho, conv = set(), [], 0
    for p in TANHSINH:
        r = I.tanhsinh(Recorder(p.f_np), p.a, p.b)
        hi, lo = ref[p.id]["value"]
        err = abs((r.integral - hi) - lo)
        if r.converged:
            conv += 1
            if err > r.error:
                short.add(p.id)
            elif err:
                rho.append(r.error / err)
    assert conv == 76 and short == TANHSINH_SHORT
    assert 1.28 < min(rho) < 1.29 and len(rho) == 66  # two converged runs are exact
    sref = refdata.series()
    for s in P.SERIES:
        r = I.nsum(Recorder(s.f_np), s.a, s.b, step=s.step)
        hi, lo = sref[s.id]["value"]
        assert r.converged and abs((r.sum - hi) - lo) <= r.error, s.id


# ---- the call form, the threads, the endpoints


def test_call_form():
    # C26, O38, O48: the first call (k,) at the midpoints, then (k, p) of the elements in progress
    # with the args as (k, 1); p = 66 (levels 0-2), then 64, 128, ...
    seen = []

    def f(x, c):
        seen.append((tuple(x.shape), tuple(c.shape), np.asarray(c).ravel().tolist()))
        return sd.cos(c * x)

    r = I.tanhsinh(f, 0.0, [1.0, 2.0, 3.0], args=([1.0, 40.0, 1.0],))
    assert seen[0] == ((3,), (3,), [1.0, 40.0, 1.0])
    assert seen[1] == ((3, 66), (3, 1), [1.0, 40.0, 1.0])
    assert all(s[2] == [40.0] and s[0][0] == 1 for s in seen[3:])  # only the element in progress
    assert r.nfev.tolist()[0] == 67 and r.max_level.tolist()[0] == 2
    assert [s[0][1] for s in seen[1:]] == [66, 64, 128, 256, 512][: len(seen) - 1]
    # nsum: f(a) on every element, then each branch's calls in scipy's order
    shapes = []
    I.nsum(lambda k: shapes.append(tuple(k.shape)) or 1 / (k * k), [1.0, 1.0], [10.0, INF])
    assert shapes[:3] == [(2,), (1, 10), (1,)]


def test_endpoints_are_called():
    # docs/api/integrate.md §1.2: rounding brings abscissae onto the limits (here x = 0 of the
    # transform of [0, inf)); the value there is discarded, so f must return inf or NaN, not raise
    calls = []

    def f(x):
        calls.append(np.asarray(x).ravel().tolist())
        with np.errstate(all="ignore"):
            return sd.asarray(np.exp(-np.asarray(x)) / np.sqrt(np.asarray(x)))

    r = I.tanhsinh(f, 0.0, INF)
    assert any(0.0 in c for c in calls) and r.status == -2

    def g(x):
        if bool(sd.any(x == 0)):
            raise ZeroDivisionError("x = 0")
        return 1 / sd.sqrt(x) * sd.exp(-x)

    with pytest.raises(ZeroDivisionError, match="x = 0"):
        I.tanhsinh(g, 0.0, INF)


def test_threads_and_large_batches():
    c = sd.linspace(0.5, 3.0, 40001)  # above the 2^15 elements of the parallel passes
    got = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        try:
            r = I.tanhsinh(lambda x, c: sd.exp(-c * x * x), 0.0, 2.0, args=(c,))
            s = I.nsum(lambda k, c: 1 / (k**c + 1), 1.0, [INF] * 3 + [5e7], args=(sd.asarray([2.0, 3.0, 4.0, 2.5]),),
                       max_terms=1000)  # fmt: skip
        finally:
            sd.set_num_threads(10)
        got.append([np.asarray(getattr(r, k)).tobytes() for k in ("integral", "error", "status", "nfev", "max_level")]
                   + [np.asarray(getattr(s, k)).tobytes() for k in ("sum", "error", "status", "nfev")])  # fmt: skip
    assert got[0] == got[1] == got[2]
    cv = np.asarray(c)
    want = np.sqrt(np.pi / cv) / 2 * np.array([math.erf(2 * math.sqrt(v)) for v in cv])
    assert bool(np.all(np.asarray(r.converged))) and np.max(np.abs(np.asarray(r.integral) - want) / want) < 2e-8


def test_optimistic_error_estimates():
    # docs/design/integrate.md §7.12: e^(-c x^2) on [0, 2] for 40,001 c in [0.5, 3]: every run
    # converges, as scipy's (the same points and values), and 1,093 of them below the true error
    # (the error estimate is not a bound: up to 1.1e-8 relative where it says 1e-12)
    from scipy import integrate

    live_scipy()
    c = np.linspace(0.5, 3.0, 40001)
    g = lambda x, c: np.exp(-c * x * x)  # noqa: E731
    rs = _quiet(lambda: integrate.tanhsinh(g, 0.0, 2.0, args=(c,)))
    rd = I.tanhsinh(Recorder(g), 0.0, 2.0, args=(c,))
    assert same(rs.integral, rd.integral) and same(rs.error, rd.error) and same(rs.nfev, rd.nfev)
    want = np.sqrt(np.pi / c) / 2 * np.array([math.erf(2 * math.sqrt(v)) for v in c])
    err = np.abs(np.asarray(rd.integral) - want)
    assert bool(np.all(np.asarray(rd.converged))) and int(np.sum(err > np.asarray(rd.error))) == 1093
    assert 1.1e-8 < np.max(err / want) < 1.2e-8


# ---- the results and the inputs


def test_results():
    r = I.tanhsinh(sd.exp, 0.0, 1.0)
    assert type(r.integral) is float and type(r.converged) is bool and type(r.max_level) is int
    assert abs(r.integral - (math.e - 1)) < 1e-15 and r.nfev == 67 and r.status == 0
    for name in ("sum", "message", "n_iter", "success", "maxlevel", "intervals"):
        assert not hasattr(r, name)
    with pytest.raises(TypeError, match="does not unpack"):
        tuple(r)
    assert repr(r) == "IntegrateResult(integral=1.71828, error=3.23821e-13, converged=True, nfev=67)"
    assert "Max Level:" in r.summary() and "Integration Results" in r.summary()
    b = I.tanhsinh(sd.exp, sd.zeros((2, 3)), 1.0)
    assert b.integral.shape == (2, 3) and b.status.dtype == sd.int64 and b.converged.dtype == sd.bool
    assert repr(b) == "IntegrateResult(batch shape (2, 3), 6 of 6 converged)"
    with pytest.raises(ValueError, match="read-only"):
        b.integral[0, 0] = 1.0
    e = I.tanhsinh(sd.exp, [], 1.0)
    assert e.integral.shape == (0,) and repr(e) == "IntegrateResult(batch shape (0,), 0 of 0 converged)"
    s = I.nsum(lambda k: 1 / (k * k), 1, INF)
    assert type(s.sum) is float and not hasattr(s, "integral") and not hasattr(s, "max_level")
    assert repr(s) == "IntegrateResult(sum=1.64493, error=7.44876e-09, converged=True, nfev=8561)"
    assert "Sum:" in s.summary()


def test_argument_errors():
    with pytest.raises(TypeError, match=r"tanhsinh\(\): 'f' must be callable, not float"):
        I.tanhsinh(1.0, 0.0, 1.0)
    for kw in ("maxlevel", "minlevel", "kwargs", "callback", "preserve_shape"):
        with pytest.raises(TypeError, match=rf"tanhsinh\(\): unexpected keyword argument '{kw}'"):
            I.tanhsinh(sd.exp, 0.0, 1.0, **{kw: 1})
    for kw in ("maxterms", "tolerances", "kwargs"):
        with pytest.raises(TypeError, match=rf"nsum\(\): unexpected keyword argument '{kw}'"):
            I.nsum(sd.exp, 0.0, 1.0, **{kw: 1})
    with pytest.raises(ValueError, match=r"tanhsinh\(\): 'max_level' must be at most 16"):
        I.tanhsinh(sd.exp, 0.0, 1.0, max_level=17)
    with pytest.raises(ValueError, match=r"tanhsinh\(\): 'min_level' must be >= 0"):
        I.tanhsinh(sd.exp, 0.0, 1.0, min_level=-1)
    with pytest.raises(TypeError, match=r"tanhsinh\(\): 'max_level' must be an int, not float"):
        I.tanhsinh(sd.exp, 0.0, 1.0, max_level=10.0)
    with pytest.raises(ValueError, match=r"tanhsinh\(\): 'rtol' must be a finite number >= 0"):
        I.tanhsinh(sd.exp, 0.0, 1.0, rtol=-1.0)
    with pytest.raises(ValueError, match=r"tanhsinh\(\): with log=True, 'atol' must be a real number below inf"):
        I.tanhsinh(sd.exp, 0.0, 1.0, log=True, atol=INF)
    with pytest.raises(TypeError, match=r"tanhsinh\(\): 'log' must be True or False, not int"):
        I.tanhsinh(sd.exp, 0.0, 1.0, log=1)
    with pytest.raises(ValueError, match=r"tanhsinh\(\): with log=True, 'b' must not be less than 'a'"):
        I.tanhsinh(lambda x: -x, [0.0, 1.0], [1.0, 0.0], log=True)
    with pytest.raises(ValueError, match=r"tanhsinh\(\): the inputs must broadcast together"):
        I.tanhsinh(sd.exp, [0.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match=r"tanhsinh\(\): 'f' must return real values, not a complex128 array"):
        I.tanhsinh(lambda x: x * 1j, 0.0, 1.0)
    with pytest.raises(
        ValueError, match=r"tanhsinh\(\): 'f' must return shape \(1, 66\) like its argument, not \(66,\)"
    ):
        I.tanhsinh(lambda x: sd.ravel(x) if x.ndim == 2 else x, 0.0, 1.0)
    with pytest.raises(ValueError, match=r"nsum\(\): 'max_terms' must be at most 16777216"):
        I.nsum(sd.exp, 0.0, 1.0, max_terms=2**24 + 1)
    with pytest.raises(TypeError, match=r"nsum\(\): 'max_terms' must be an int, not float"):
        I.nsum(sd.exp, 0.0, 1.0, max_terms=1e3)
    with pytest.raises(TypeError, match=r"nsum\(\): 'step' must be a real number or an array, not bool"):
        I.nsum(sd.exp, 0.0, 1.0, step=True)
