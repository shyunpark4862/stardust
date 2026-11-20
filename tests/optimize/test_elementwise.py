"""sd.optimize.find_root, bracket_root, find_minimum, bracket_minimum (docs/api/optimize.md §6,
docs/design/optimize.md §3.2, §3.3, §7): scipy 1.18.1's optimize.elementwise on the problems of
functions.py one element at a time (the recorded runs, tests/optimize/data/scipy_elementwise.json,
and every call point live), batches of many elements against scipy's batch and against each element
alone, the shapes, the calling convention (O38), the result and the errors."""

import json
import math
import pathlib
import warnings

import numpy as np
import pytest
import scipy
from scipy.optimize import elementwise as ew

import stardust as sd

from . import functions as fn

DATA = pathlib.Path(__file__).parent / "data"
REF = json.loads((DATA / "scipy_elementwise.json").read_text())
LIVE = pytest.mark.skipif(scipy.__version__ != "1.18.1", reason="the steps are scipy 1.18.1's")
PAPER_ROOT = dict(xrtol=4e-10, xatol=1e-5)
PAPER_MIN = dict(xatol=1e-4, fatol=1e-6, xrtol=1e-16, frtol=1e-16)


def vec(c, log=None):
    """c's function on an sd.ndarray (k,), element by element in doubles."""

    def f(x):
        xs = x.tolist()
        if log is not None:
            log.append(xs)
        with np.errstate(all="ignore"):
            return sd.asarray([float(c.f(np.float64(t))) for t in xs])

    return f


def vec_np(c, log):
    def f(x):
        x = np.asarray(x, dtype=float)
        log.append([float(t) for t in np.ravel(x)])
        with np.errstate(all="ignore"):
            return np.array([float(c.f(np.float64(t))) for t in np.ravel(x)]).reshape(x.shape)

    return f


def same(a, b):
    a, b = float(a), float(b)
    return a == b or (math.isnan(a) and math.isnan(b))


def check(s, r, keys):
    """The recorded scipy fields s (scipy's names) against the result r."""
    names = {"f_x": "fun", "nit": "n_iter", "success": "converged"}
    for k in keys:
        if k not in s:
            continue
        v = getattr(r, names.get(k, k))
        if k in ("bracket", "f_bracket"):
            assert len(v) == len(s[k]) and all(same(p, q) for p, q in zip(v, s[k])), k
        else:
            assert same(v, s[k]), (k, v, s[k])


def check_bracket_root(s, r):
    """bracket_root against scipy's record: the same fields, except where both sides found a
    sign change in the same iteration with brackets as wide. scipy then returns the right side's
    left end and the left side's right end, a reversed interval; here it is the left side's
    bracket, which ends at the same right end and brackets the root."""
    if s["status"] == 0 and s["bracket"][0] > s["bracket"][1]:
        check(s, r, ("nfev", "nit", "status", "success"))
        (xl, xr), (fl, fr) = r.bracket, r.f_bracket
        assert xl < xr and same(xr, s["bracket"][1]) and same(fr, s["f_bracket"][1])
        assert fl == 0 or fr == 0 or (fl < 0) != (fr < 0)
        return 1
    check(s, r, BRACKET_KEYS)
    return 0


ROOT_KEYS = ("x", "f_x", "nfev", "nit", "status", "success", "bracket", "f_bracket")
BRACKET_KEYS = ("nfev", "nit", "status", "success", "bracket", "f_bracket")
MIN3 = [c for c in fn.MIN_CASES if len(c.bracket) == 3]


@pytest.mark.parametrize("c", fn.ROOT_CASES, ids=repr)
def test_find_root_and_bracket_root_against_recorded_scipy(c):
    check(REF[c.id]["find_root"], sd.optimize.find_root(vec(c), c.bracket), ROOT_KEYS)
    if c.nfev_paper is not None:  # Chandrupatla's counts with his tolerances
        r = sd.optimize.find_root(vec(c), c.bracket, **PAPER_ROOT)
        check(REF[c.id]["find_root/paper"], r, ROOT_KEYS)
        assert r.nfev == c.nfev_paper
    x0 = c.x0 if c.x0 is not None else c.bracket[0]
    check_bracket_root(REF[c.id]["bracket_root"], sd.optimize.bracket_root(vec(c), x0))


@pytest.mark.parametrize("c", fn.MIN_CASES, ids=repr)
def test_find_minimum_and_bracket_minimum_against_recorded_scipy(c):
    if len(c.bracket) == 3:
        check(REF[c.id]["find_minimum"], sd.optimize.find_minimum(vec(c), c.bracket), ROOT_KEYS)
    if c.nit_paper is not None:
        r = sd.optimize.find_minimum(vec(c), c.bracket, **PAPER_MIN)
        check(REF[c.id]["find_minimum/paper"], r, ROOT_KEYS)
        assert r.n_iter == c.nit_paper
    check(REF[c.id]["bracket_minimum"], sd.optimize.bracket_minimum(vec(c), c.bracket[0]), BRACKET_KEYS)


@LIVE
def test_every_call_point_follows_scipy():
    # one element: scipy's calls (its first ones have the input's shape, here (1,)) and these are
    # the same points bit for bit
    warnings.simplefilter("ignore")
    runs = []
    for c in fn.ROOT_CASES:
        x0 = c.x0 if c.x0 is not None else c.bracket[0]
        runs.append((c, "find_root", (c.bracket,), {}))
        runs.append((c, "bracket_root", (x0,), {}))
        if c.nfev_paper is not None:
            runs.append((c, "find_root", (c.bracket,), PAPER_ROOT))
    for c in fn.MIN_CASES:
        runs.append((c, "bracket_minimum", (c.bracket[0],), {}))
        if len(c.bracket) == 3:
            runs.append((c, "find_minimum", (c.bracket,), {}))
    for c, name, a, kw in runs:
        ls, ld = [], []
        getattr(ew, name)(vec_np(c, ls), *a, **({"tolerances": kw} if kw else {}))
        getattr(sd.optimize, name)(vec(c, ld), *a, **kw)
        assert ls == ld or all(len(p) == len(q) and all(same(a, b) for a, b in zip(p, q)) for p, q in zip(ls, ld)), (
            c.id,
            name,
        )
        assert len(ls) == len(ld)


# ---- batches: many elements at once, against scipy's batch and against each element alone


def g_np(x):
    return x * x * x - 2 * x - 5 + 1 / (1 + x * x)


def g_sd(x):
    return x * x * x - 2 * x - 5 + 1 / (1 + x * x)


def h(x):
    return (x - 1.3) * (x - 1.3) + 0.3 / (1 + x * x * x * x)


def logged(g, log):
    def f(x):
        log.append(np.ravel(np.array(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)))
        return g(x)

    return f


def as_np(v):
    if isinstance(v, tuple):
        return tuple(as_np(t) for t in v)
    return np.array(v.tolist()) if isinstance(v, sd.ndarray) else np.asarray(v)


def eq(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


SCIPY_NAMES = {"fun": "f_x", "n_iter": "nit", "converged": "success"}


def assert_batch_equal(rs, rd, fields):
    for k in fields:
        sv = rs[SCIPY_NAMES.get(k, k)]
        dv = getattr(rd, k)
        if isinstance(sv, tuple):
            assert len(sv) == len(dv) and all(eq(p, as_np(q)) for p, q in zip(sv, dv)), k
        else:
            assert eq(sv, as_np(dv)), k


def batch_inputs():
    rng = np.random.default_rng(20261120)
    n = 2000
    xl = rng.uniform(-4, 2.5, n)
    xr = xl + rng.uniform(0.01, 5, n)
    xl[:5] = np.nan  # -3
    xr[5:10] = np.inf  # -3
    xl[10:20] = xr[10:20]  # same sign: -1
    return xl.reshape(40, 50), xr.reshape(40, 50)


RF = ("x", "fun", "bracket", "f_bracket", "converged", "status", "n_iter", "nfev")
BF = ("bracket", "f_bracket", "converged", "status", "n_iter", "nfev")


@LIVE
@pytest.mark.parametrize(
    "name,kw",
    [
        ("find_root", {}),
        ("find_root", dict(xatol=1e-3, frtol=1e-9, max_iter=6)),
        ("bracket_root", dict(factor=1.5, xmin="every third -10")),
        ("bracket_root", dict(xr0=True, xmax=3.0, max_iter=20)),
    ],
)
def test_root_batches_follow_scipy(name, kw):
    warnings.simplefilter("ignore")
    xl, xr = batch_inputs()
    kw = dict(kw)
    if kw.get("xmin") == "every third -10":
        kw["xmin"] = np.where(np.arange(2000).reshape(40, 50) % 3 == 0, -10.0, -np.inf)
    ls, ld = [], []
    if name == "find_root":
        tol = {k: kw.pop(k) for k in ("xatol", "frtol") if k in kw}
        mi = kw.pop("max_iter", None)
        rs = ew.find_root(logged(g_np, ls), (xl, xr), tolerances=tol or None, **({"maxiter": mi} if mi else {}))
        rd = sd.optimize.find_root(logged(g_sd, ld), (xl, xr), **tol, **({"max_iter": mi} if mi else {}))
        fields = RF
    else:
        a = (xl, xr) if kw.pop("xr0", False) else (xl,)
        mi = kw.pop("max_iter", None)
        rs = ew.bracket_root(logged(g_np, ls), *a, **kw, **({"maxiter": mi} if mi else {}))
        rd = sd.optimize.bracket_root(logged(g_sd, ld), *a, **kw, **({"max_iter": mi} if mi else {}))
        fields = BF
    assert_batch_equal(rs, rd, fields)
    assert len(ls) == len(ld) and all(eq(p, q) for p, q in zip(ls, ld))
    assert len(set(as_np(rd.status).ravel().tolist())) >= 2


@LIVE
def test_minimum_batches_follow_scipy():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(7)
    n = 2000
    xm = rng.uniform(-3, 3, n)
    w = rng.uniform(0.05, 2, n)
    init = (xm - w, xm, xm + w)
    xm[:3] = np.nan
    xmax = np.where(np.arange(n) % 4 == 0, 2.0, np.inf)
    for name, a, kw in (
        ("find_minimum", (init,), {}),
        ("bracket_minimum", (xm,), {"xmax": xmax}),
        ("bracket_minimum", (xm,), {"xl0": xm - 0.1, "factor": 3.0}),
    ):
        ls, ld = [], []
        rs = getattr(ew, name)(logged(h, ls), *a, **kw)
        rd = getattr(sd.optimize, name)(logged(h, ld), *a, **kw)
        assert_batch_equal(rs, rd, RF if name == "find_minimum" else BF)
        assert len(ls) == len(ld) and all(eq(p, q) for p, q in zip(ls, ld)), name


def test_batch_equals_each_element_alone():
    xl, xr = batch_inputs()
    r = sd.optimize.find_root(g_sd, (xl, xr))
    b = sd.optimize.bracket_root(g_sd, xl, xmax=2.0)
    for i in range(0, 2000, 13):
        a, j = divmod(i, 50)
        e = sd.optimize.find_root(g_sd, (float(xl[a, j]), float(xr[a, j])))
        for k in ("x", "fun", "converged", "status", "n_iter", "nfev"):
            assert same(as_np(getattr(r, k))[a, j], getattr(e, k)), (i, k)
        e = sd.optimize.bracket_root(g_sd, float(xl[a, j]), xmax=2.0)
        for k in ("converged", "status", "n_iter", "nfev"):
            assert same(as_np(getattr(b, k))[a, j], getattr(e, k)), (i, k)
        assert all(same(as_np(p)[a, j], q) for p, q in zip(b.bracket, e.bracket))


def test_large_batches_split_over_threads_keep_every_bit():
    # from 2^15 elements in progress the passes over them split over the threads: every field is
    # the same with 1 and 10 threads, and the same as for each block of the batch on its own
    n = 150_000
    c = np.linspace(1.0, 10.0, n)
    sc = sd.asarray(c)

    def f(x, c):
        return x * x * x - 2.0 * x - c

    def g(x, c):
        return (x - c) * (x - c) + 0.01 * c * x * x * x * x

    runs = {
        "find_root": lambda c: sd.optimize.find_root(f, (0.0, 3.0), args=(c,)),
        "bracket_root": lambda c: sd.optimize.bracket_root(f, 0.0, args=(c,)),
        "find_minimum": lambda c: sd.optimize.find_minimum(g, (-1.0, 0.5, 11.0), args=(c,)),
        "bracket_minimum": lambda c: sd.optimize.bracket_minimum(g, 0.0, args=(c,)),
    }
    fields = ("x", "fun", "bracket", "f_bracket", "converged", "status", "n_iter", "nfev")

    def flat(r):
        out = []
        for k in fields:
            if hasattr(r, k):
                v = getattr(r, k)
                out += [as_np(t).tolist() for t in v] if isinstance(v, tuple) else [as_np(v).tolist()]
        return out

    old = sd.get_num_threads()
    try:
        for name, run in runs.items():
            got = []
            for t in (1, 10):
                sd.set_num_threads(t)
                got.append(flat(run(sc)))
            assert got[0] == got[1], name
            parts = [flat(run(sd.asarray(c[i : i + 30_000]))) for i in range(0, n, 30_000)]
            assert got[0] == [sum((p[k] for p in parts), []) for k in range(len(got[0]))], name
    finally:
        sd.set_num_threads(old)


def test_roots_agree_with_root_scalar():
    # the same roots as root_scalar's brentq to the tolerances, element by element
    c0 = np.linspace(-20, 20, 101)
    r = sd.optimize.find_root(lambda x: x * x * x - 2 * x - 5, (sd.full((101,), -30.0), 30 + c0 * 0))
    s = sd.optimize.root_scalar(lambda x: x**3 - 2 * x - 5, bracket=(-30.0, 30.0))
    assert np.allclose(as_np(r.x), s.x, rtol=1e-14, atol=0)
    for c in fn.ROOT_CASES[:60]:
        e = sd.optimize.find_root(vec(c), c.bracket)
        q = sd.optimize.root_scalar(lambda x, c=c: float(c.f(np.float64(x))), bracket=c.bracket, method="toms748")
        if e.converged and e.status == 0:
            with np.errstate(all="ignore"):
                assert abs(e.x - q.x) <= 1e-9 * max(1.0, abs(q.x)) or float(c.f(np.float64(e.x))) == 0


# ---- args (O39): one value per element, reduced with x


@LIVE
def test_args_follow_scipy():
    # x^3 - c = 0 and (x - c)^2 + k x^4 with a different c per element: the same calls and results
    # as scipy's args, which are reduced to the elements in progress alongside x
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(3)
    c = rng.uniform(-50, 50, 500)
    k = np.array([0.0, 0.01, 0.1, 1.0, 10.0])[:, None]  # broadcasts to (5, 500)
    xl, xr = np.full(500, -10.0), np.full(500, 10.0)
    f = lambda x, c: x * x * x - c
    g = lambda x, c, k: (x - c / 10) * (x - c / 10) + k * (x * x) * (x * x)
    for name, a, kw, fn_ in (
        ("find_root", ((xl, xr),), {"args": (c,)}, f),
        ("bracket_root", (xl / 10,), {"args": (c,)}, f),
        ("find_minimum", ((xl, 0.5 * xl / 10, xr),), {"args": (c, k)}, g),
        ("bracket_minimum", (xl / 10,), {"args": (c, k)}, g),
    ):
        ls, ld = [], []

        def log_np(x, *p, fn_=fn_, ls=ls):
            ls.append((np.ravel(x).copy(), [np.ravel(np.broadcast_to(q, np.shape(x))).copy() for q in p]))
            return fn_(x, *p)

        def log_sd(x, *p, fn_=fn_, ld=ld):
            ld.append((as_np(x), [as_np(q) for q in p]))
            return fn_(x, *p)

        rs = getattr(ew, name)(log_np, *a, **kw)
        rd = getattr(sd.optimize, name)(log_sd, *a, **kw)
        assert_batch_equal(rs, rd, RF if name.startswith("find") else BF)
        assert len(ls) == len(ld)
        for (xs, ps), (xd, pd) in zip(ls, ld):
            assert eq(xs, xd) and all(eq(p, q) for p, q in zip(ps, pd)), name


def test_args_shapes_and_kinds():
    seen = []

    def f(x, c, k, m):
        seen.append((x.shape, c.shape, k.shape, m.shape, c.dtype, k.dtype, m.dtype))
        return x * x * x - c * k

    c = sd.asarray([[1.0], [8.0], [27.0]])  # (3, 1) with init (4,): batch (3, 4)
    r = sd.optimize.find_root(
        f, (0.0, sd.asarray([5.0, 6.0, 7.0, 8.0])), args=(c, 2, sd.asarray([True, False, True, True]))
    )
    assert r.x.shape == (3, 4)
    assert np.allclose(as_np(r.x), np.cbrt(2 * np.array([1.0, 8.0, 27.0]))[:, None] * np.ones(4), rtol=1e-13)
    assert all(s[0] == s[1] == s[2] == s[3] for s in seen)  # every arg reduced with x
    assert {s[4:] for s in seen} == {(sd.float64, sd.int64, sd.bool)}  # their own dtypes
    assert seen[0][0] == (12,) and seen[-1][0][0] < 12
    r = sd.optimize.find_root(lambda x, c: x - c, (0.0, 10.0), args=(3.5,))  # numbers only: numbers
    assert type(r.x) is float and r.x == 3.5
    r = sd.optimize.bracket_root(lambda x, c: x - c, 0.0, args=([5.0, -5.0, 0.5],))
    assert r.converged.tolist() == [True, True, True]
    assert as_np(r.bracket[0]).tolist()[1] <= -5 <= as_np(r.bracket[1]).tolist()[1]
    err(TypeError, "find_root(): 'args' must be a tuple, not float", sd.optimize.find_root, f, (0, 1), args=1.0)
    err(
        TypeError,
        "find_root(): 'args[0]' must be a number or an array, not str",
        sd.optimize.find_root,
        f,
        (0, 1),
        args=("a",),
    )
    err(
        ValueError,
        "find_root(): the inputs must broadcast together, not shapes (), (2,), (3,)",
        sd.optimize.find_root,
        f,
        (0, [1.0, 2.0]),
        args=([1.0, 2.0, 3.0],),
    )


# ---- the calling convention (O38), shapes, results


def test_f_gets_the_remaining_elements_as_a_1d_array():
    seen = []

    def f(x):
        seen.append((type(x), x.dtype, x.shape))
        return x * x - 2

    r = sd.optimize.find_root(f, (sd.asarray([[0.0, 1.0, 1.4], [0.5, 1.41, 1.0]]), 2.0))
    assert all(t is sd.ndarray and d == sd.float64 and len(s) == 1 for t, d, s in seen)
    assert seen[0][2] == seen[1][2] == (6,)  # f(xl), f(xr): all elements
    sizes = [s[0] for _, _, s in seen[2:]]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] < 6  # then only those in progress
    assert r.x.shape == (2, 3) and r.status.dtype == sd.int64 and r.converged.dtype == sd.bool


def test_f_may_return_sequences_and_other_dtypes():
    r = sd.optimize.find_root(lambda x: [t - 0.25 for t in x.tolist()], (0.0, 1.0))
    assert r.converged and abs(r.x - 0.25) < 1e-15
    r = sd.optimize.find_root(lambda x: np.array(x.tolist()) - 0.5, (0.0, [1.0, 2.0]))
    assert r.x.tolist() == [0.5, 0.5]
    r = sd.optimize.bracket_root(lambda x: sd.asarray(x > 3.0, dtype=sd.int64) * 2 - 1, 0.0)
    assert r.converged


def test_scalar_inputs_give_numbers_and_broadcasting():
    r = sd.optimize.find_root(lambda x: x - 0.3, (0, 1))
    assert type(r.x) is float and type(r.converged) is bool and type(r.status) is int and type(r.nfev) is int
    assert isinstance(r.bracket, tuple) and type(r.bracket[0]) is float
    assert repr(r).startswith("OptimizeResult(converged=True, fun=")
    r = sd.optimize.find_minimum(
        lambda x: (x - 0.3) ** 2, (sd.asarray([[-1.0], [-2.0]]), 0.5, sd.asarray([2.0, 3.0, 4.0]))
    )
    assert r.x.shape == (2, 3) and len(r.bracket) == 3 and r.f_bracket[1].shape == (2, 3)
    assert repr(r) == "OptimizeResult(batch shape (2, 3), 6 of 6 converged)"
    r = sd.optimize.bracket_minimum(lambda x: (x - 5) ** 2, sd.zeros((0,)))
    assert r.bracket[0].shape == (0,) and r.nfev.shape == (0,)
    with pytest.raises(AttributeError):
        r.message  # noqa: B018 (batch results have no message)


def test_status_codes():
    f = lambda x: x * x - 1
    xl = sd.asarray([0.0, 2.0, math.nan, 0.0])
    xr = sd.asarray([3.0, 3.0, 3.0, math.inf])
    assert sd.optimize.find_root(f, (xl, xr)).status.tolist() == [0, -1, -3, -3]  # O32: -1, not an error
    assert sd.optimize.find_root(f, (0.0, 3.0), max_iter=2).status == -2
    r = sd.optimize.find_minimum(lambda x: x, (0.0, 1.0, 2.0))
    assert r.status == -1 and math.isnan(r.x)
    g = lambda x: 1.0 - 2.0 * (x < -3.5)  # a sign change only at -3.5
    r = sd.optimize.bracket_root(
        g, sd.asarray([2.0, 5.0, 0.0]), sd.asarray([3.0, 4.0, 0.5]), xmin=[-3.0, -4, -4], xmax=4
    )
    assert r.status.tolist() == [-1, -5, 0]  # limits reached, invalid start, found
    r = sd.optimize.bracket_minimum(lambda x: -x, 0.0, xmax=10.0)
    assert r.status == -1 and r.bracket[2] == 10.0


# ---- errors


def err(exc, msg, fn_, *a, **kw):
    with pytest.raises(exc) as e:
        fn_(*a, **kw)
    assert str(e.value) == msg, str(e.value)


def test_errors():
    o = sd.optimize
    f = lambda x: x
    err(TypeError, "find_root(): 'f' must be callable, not int", o.find_root, 1, (0, 1))
    err(ValueError, "find_root(): 'init' must be a pair (xl, xr), not 3 values", o.find_root, f, (0, 1, 2))
    err(TypeError, "find_root(): 'init' must be a pair (xl, xr), not float", o.find_root, f, 1.0)
    err(ValueError, "find_minimum(): 'init' must be a triple (x1, x2, x3), not 2 values", o.find_minimum, f, (0, 1))
    err(
        ValueError,
        "find_root(): the inputs must broadcast together, not shapes (2,), (3,)",
        o.find_root,
        f,
        ([0.0, 1.0], [1.0, 2.0, 3.0]),
    )
    err(TypeError, "find_root(): 'init' must be real, not a complex128 array", o.find_root, f, ([1j], 1.0))
    err(TypeError, "bracket_root(): 'xl0' must be a real number or an array, not bool", o.bracket_root, f, True)
    err(ValueError, "find_root(): 'xatol' must be a finite number >= 0", o.find_root, f, (0, 1), xatol=-1.0)
    err(ValueError, "find_minimum(): 'max_iter' must be >= 0", o.find_minimum, f, (0, 1, 2), max_iter=-1)
    err(TypeError, "bracket_root(): 'max_iter' must be an int, not float", o.bracket_root, f, 0.0, max_iter=1.5)
    err(ValueError, "bracket_root(): 'factor' must be > 1", o.bracket_root, f, 0.0, factor=1.0)
    err(ValueError, "bracket_minimum(): 'factor' must be > 1", o.bracket_minimum, f, 0.0, factor=[2.0, math.nan])
    err(
        ValueError,
        "find_root(): 'f' must return shape (1,) like its argument, not (1, 1)",
        o.find_root,
        lambda x: sd.reshape(x, (1, 1)),
        (0, 1),
    )
    err(
        ValueError,
        "find_root(): 'f' must return shape (2,) like its argument, not a scalar",
        o.find_root,
        lambda x: 1.0,
        (0, [1, 2]),
    )
    err(
        TypeError,
        "find_root(): 'f' must return real values, not a complex128 array",
        o.find_root,
        lambda x: x * 1j,
        (0, 1),
    )
    err(
        TypeError,
        "find_root(): 'f' must return an sd.ndarray or a sequence of numbers, not NoneType",
        o.find_root,
        lambda x: None,
        (0, 1),
    )
    for name in ("find_root", "find_minimum", "bracket_root", "bracket_minimum"):
        with pytest.raises(TypeError, match="unexpected keyword argument 'callback'"):  # O37
            getattr(o, name)(
                f, (0, 1, 2) if name == "find_minimum" else (0, 1) if name == "find_root" else 0.0, callback=None
            )
        with pytest.raises(TypeError, match="unexpected keyword argument 'tolerances'"):  # O36
            getattr(o, name)(
                f, (0, 1, 2) if name == "find_minimum" else (0, 1) if name == "find_root" else 0.0, tolerances={}
            )


def test_user_errors_pass_through():
    class Boom(Exception):
        pass

    def f(x):
        raise Boom("inside")

    for call in (
        lambda: sd.optimize.find_root(f, (0, 1)),
        lambda: sd.optimize.find_minimum(f, (0, 1, 2)),
        lambda: sd.optimize.bracket_root(f, 0.0),
        lambda: sd.optimize.bracket_minimum(f, 0.0),
    ):
        with pytest.raises(Boom, match="^inside$"):
            call()


# ---- bool values (SCALAR-6), xr0 by keyword (SCALAR-4, D-25), bracket_root's passes (SCALAR-3)


def test_a_bool_array_is_an_error():
    with pytest.raises(TypeError, match=r"find_root\(\): 'f' must return real values, not a bool array"):
        sd.optimize.find_root(lambda x: x > 1.26, (sd.zeros(2), sd.full((2,), 2.0)))


def test_bracket_root_xr0_by_keyword():
    a = sd.optimize.bracket_root(lambda x: x - 5, 0.0, 1.0)
    b = sd.optimize.bracket_root(lambda x: x - 5, 0.0, xr0=1.0)
    assert a.bracket == b.bracket and a.status == b.status


def test_bracket_root_threads_give_the_same_bits():
    # from 2^15 sides in progress the passes split over the threads without the GIL
    rng = np.random.default_rng(3)
    xl0 = sd.asarray(rng.uniform(-10, 10, 50_000).tolist())
    c = sd.asarray(rng.uniform(-100, 100, 50_000).tolist())
    old = sd.get_num_threads()
    try:
        out = []
        for t in (1, 4):
            sd.set_num_threads(t)
            r = sd.optimize.bracket_root(lambda x, c: x - c, xl0, args=(c,))
            out.append([v.tolist() for v in r.bracket + r.f_bracket] + [r.status.tolist(), r.nfev.tolist()])
        assert out[0] == out[1]
    finally:
        sd.set_num_threads(old)


# ---- the bracket is sorted and brackets a root whenever status is 0


def test_bracket_root_when_both_sides_find_a_sign_change_as_narrow():
    # f(0) = -1, f(1) = -2; the next points are -1 (f = 0) and 2 (f = 3): both sides find a sign
    # change in the same iteration with brackets of width 1. scipy returns (1, 0), f (-2, -1).
    r = sd.optimize.bracket_root(lambda x: x**3 - 2 * x - 1, 0.0)
    assert (r.bracket, r.f_bracket, r.status, r.converged) == ((-1.0, 0.0), (0.0, -1.0), 0, True)
    # unequal widths: the narrower side's, as scipy
    r = sd.optimize.bracket_root(lambda x: x - 0.5, 0.0)
    assert (r.bracket, r.f_bracket, r.status) == ((0.0, 1.0), (-0.5, 0.5), 0)


def test_bracket_root_status_0_brackets_a_root_and_is_sorted():
    rng = np.random.default_rng(11)
    n = 3000
    # cubics with three integer roots and integer starts: many sides land on a root exactly, and
    # in 215 of them both sides find a sign change as narrow (scipy returns a reversed interval)
    r1, r2, r3 = (np.round(rng.uniform(-4, 4, n), 0) for _ in range(3))
    xl0 = np.round(rng.uniform(-3, 3, n), 0)
    args = tuple(sd.asarray(v.tolist()) for v in (r1, r2, r3))
    r = sd.optimize.bracket_root(lambda x, a, b, c: (x - a) * (x - b) * (x - c), sd.asarray(xl0.tolist()), args=args)
    xl, xr = (v.tolist() for v in r.bracket)
    fl, fr = (v.tolist() for v in r.f_bracket)
    status = r.status.tolist()
    assert status.count(0) > n // 2
    for i in range(n):
        assert xl[i] <= xr[i]
        if status[i] == 0:
            assert fl[i] == 0 or fr[i] == 0 or (fl[i] < 0) != (fr[i] < 0), i


def test_bracket_root_invalid_start_is_sorted():
    r = sd.optimize.bracket_root(lambda x: x, 2.0, 1.0)
    assert (r.bracket, r.f_bracket, r.status) == ((1.0, 2.0), (1.0, 2.0), -5)
