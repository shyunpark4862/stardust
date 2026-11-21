"""cubature (docs/api/integrate.md §5; docs/design/integrate.md §3.4, §7.13; decisions C5, C21,
C94-C107).

Against scipy 1.18.1's records (tests/integrate/data/scipy_multiple.json: the boxes of
problems.MULTI with the rules gk21, gk15 and genz-malik) and the live scipy: the same points in the
same order (the sha256 of their bytes; genz-malik's argument in Fortran order, as scipy's), the same
estimate, error, subdivisions and number of regions to the bit, on the records, the live cases of
tests/integrate/adaptive.py and the collection. Where sd differs on purpose: (-inf, b], points with
infinite limits, a non-finite value. Against the Arb references: a converged value's true error is
within its error + 8 eps |ref| but for adaptive.CUBATURE_SHORT (scipy's too). And the inputs, the
errors and the result."""

import hashlib
import math
import warnings

import numpy as np
import pytest

import stardust as sd

from . import adaptive as A
from . import problems as P
from . import refdata
from .test_reference_data import live_scipy

I = sd.integrate  # noqa: E741
INF = math.inf


class Recorder:
    """f's points in call order (the sha256 of their float64 bytes in C order, the number of
    points, the calls' shapes); fn on numpy arrays in the argument's layout, its value given to sd
    as an sd.ndarray in its own layout."""

    def __init__(self, fn, to_sd):
        self.fn, self.to_sd, self.h, self.count, self.shapes = fn, to_sd, hashlib.sha256(), 0, []

    def __call__(self, x):
        if isinstance(x, sd.ndarray):
            a = np.array(x.tolist(), dtype=np.float64)
            if x.ndim == 2 and x.shape[1] > 1 and x.strides[0] == 8:
                a = np.asfortranarray(a)
        else:
            a = x
        self.h.update(np.ascontiguousarray(a).tobytes())
        self.count += a.shape[0]
        self.shapes.append(list(a.shape))
        with np.errstate(all="ignore"):
            v = self.fn(a)
        return sd.asarray(v) if self.to_sd else v


def same(x, y):
    x = np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return x.shape == y.shape and bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))))


def scipy_run(fn, a, b, **kw):
    from scipy import integrate

    rec = Recorder(fn, False)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        r = integrate.cubature(rec, np.asarray(a, dtype=float), np.asarray(b, dtype=float), **kw)
    return rec, r


def assert_same_as_scipy(fn, a, b, **kw):
    live_scipy()
    rs, s = scipy_run(fn, a, b, **kw)
    rd = Recorder(fn, True)
    r = I.cubature(rd, a, b, **kw)
    assert rd.h.digest() == rs.h.digest() and rd.count == rs.count == r.nfev
    assert same(r.integral, s.estimate) and same(r.error, s.error)
    assert r.subdivisions == s.subdivisions and r.n_regions == len(s.regions)
    finite = bool(np.all(np.isfinite(np.asarray(s.estimate))) and np.all(np.isfinite(np.asarray(s.error))))
    assert r.converged == (s.status == "converged" and finite)
    assert r.status == (0 if r.converged else 1 if s.status == "not_converged" else 3)
    # the regions: scipy's list is its heap; sd's from the largest error down (C95)
    sign = (-1.0) ** sum(x > y for x, y in zip(a, b))
    want = sorted((tuple(g.a.tolist()), tuple(g.b.tolist())) for g in s.regions)
    got = sorted((tuple(lo), tuple(hi)) for lo, hi in r.regions.tolist())
    assert got == want
    by = {(tuple(g.a.tolist()), tuple(g.b.tolist())): g for g in s.regions}
    for (lo, hi), v, e in zip(r.regions.tolist(), r.region_integrals.tolist(), r.region_errors.tolist()):
        g = by[(tuple(lo), tuple(hi))]
        assert same(v, np.asarray(g.estimate) * sign) and same(e, g.error)
    return r


# ---- scipy's records


BOXES = [(m, rule) for m in P.MULTI if m.box is not None for rule in ("gk21", "gk15", "genz-malik")]


@pytest.mark.parametrize(("m", "rule"), BOXES, ids=lambda v: getattr(v, "id", v))
def test_records(m, rule):
    rec = refdata.data("scipy_multiple.json")[m.id][f"cubature/{rule}"]
    f = Recorder(lambda X: m.func(*[X[:, i] for i in range(X.shape[1])]), True)
    r = I.cubature(f, [lo for lo, _ in m.box], [hi for _, hi in m.box], rule=rule)
    calls = rec["calls"]
    assert f.h.hexdigest() == calls["sha256"] and f.count * len(m.box) == calls["count"] and r.nfev == rec["nfev"]
    assert len(f.shapes) == calls["calls"]
    if "shapes" in calls:
        assert [s for s, k in calls["shapes"] for _ in range(k)] == f.shapes
    assert same(r.integral, rec["integral"]) and same(r.error, rec["error"])
    assert r.subdivisions == rec["subdivisions"] and r.converged == (rec["status"] == "converged") and r.status == 0


# ---- the live scipy


@pytest.mark.parametrize("case", A.CUBATURE_LIVE, ids=lambda c: c[0])
def test_live_cases(case):
    _, fn, a, b, kw = case
    assert_same_as_scipy(fn, [float(t) for t in a], [float(t) for t in b], **kw)


def test_live_statuses():
    st = {c[0]: I.cubature(lambda X, c=c: sd.asarray(c[1](np.asarray(X.tolist()))), c[2], c[3], **c[4]).status
          for c in A.CUBATURE_LIVE}  # fmt: skip
    assert st["max_subdivisions"] == 1 and st["gauss gk21 2-D"] == 0 and st["zero width"] == 0


def _left(p):  # (-inf, b]: scipy's transform calls f elsewhere (C101)
    return min(p.a, p.b) == -INF and max(p.a, p.b) < INF


@pytest.mark.parametrize("rule", ["gk21", "gk15"])
@pytest.mark.parametrize("p", [p for p in P.INTEGRALS if p.weight is None and not _left(p)], ids=lambda p: p.id)
def test_collection_matches_scipy(p, rule):
    kw = {"points": [[t] for t in p.points]} if p.points else {}
    assert_same_as_scipy(lambda X: p.f_np(X[:, 0]), [p.a], [p.b], rule=rule, **kw)


@pytest.mark.parametrize("rule", ["gk21", "gk15"])
@pytest.mark.parametrize("bt", P.BATCHES, ids=lambda b: b.family)
def test_families_match_scipy(bt, rule):
    params = np.array(bt.params)
    assert_same_as_scipy(lambda X: bt.f_np(X[:, :1], params[None, :]), [bt.a], [bt.b], rule=rule)


def test_gauss_lower_rules_are_scipys():
    # the error rules' lower nodes: scipy's roots_legendre(10) and (7) to the bit (C99), at
    # (x + 1) * 1.0 + 0.0 on [0, 2]; the second call holds the rule's nodes, then the lower rule's
    from scipy.special import roots_legendre

    live_scipy()
    for rule, k, n in (("gk21", 21, 10), ("gk15", 15, 7)):
        seen = []
        I.cubature(
            lambda X: (seen.append(np.asarray(X.tolist())[:, 0]), sd.ones(X.shape[0]))[1], [0.0], [2.0], rule=rule
        )
        assert [len(v) for v in seen] == [k, k + n]
        x, _ = roots_legendre(n)
        assert seen[1][k:].tobytes() == ((x + 1) * 1.0 + 0.0).tobytes()


# ---- where sd differs from scipy on purpose


def test_left_half_line_is_the_documented_transform():
    # scipy calls f at -b + (1 - t)/t on (-inf, b] (its integral over [-b, inf)); sd at b - (1 - t)/t,
    # the transform scipy's documentation gives (C101)
    r = I.cubature(lambda X: sd.exp(X[:, 0]), [-INF], [0.0])
    assert abs(r.integral - 1.0) < 1e-14 and r.converged
    r = I.cubature(lambda X: sd.exp(-X[:, 0] * X[:, 0]) * (X[:, 0] > 0), [-INF], [1.0], points=[[0.0]])
    assert abs(r.integral - math.sqrt(math.pi) / 2 * math.erf(1.0)) < 1e-12
    # the mirror image of [-1, inf): f at the mirrored points
    f = Recorder(lambda X: np.exp(-((X[:, 0] - 0.3) ** 2)), True)
    g = Recorder(lambda X: np.exp(-((-X[:, 0] - 0.3) ** 2)), True)
    left, right = I.cubature(f, [-INF], [1.0]), I.cubature(g, [-1.0], [INF])
    assert same(left.integral, right.integral) and same(left.error, right.error) and left.nfev == right.nfev
    # the collection's integrals on (-inf, b]: within their error estimate of the Arb value
    ref = refdata.integrals()
    lefts = [p for p in P.INTEGRALS if p.weight is None and _left(p)]
    assert lefts
    for p in lefts:
        r = I.cubature(lambda X, p=p: sd.asarray(p.f_np(np.asarray(X.tolist())[:, 0])), [p.a], [p.b])
        assert r.converged and abs(r.integral - refdata.value(ref[p.id])) <= r.error + 8 * 2**-52, p.id


def test_points_with_infinite_limits():
    # scipy's inverse transform raises IndexError or ValueError for points with an infinite limit
    # other than one doubly infinite variable; sd maps each variable (C102)
    g = lambda X: sd.exp(-sd.sum(X * X, axis=-1))  # noqa: E731
    r = I.cubature(g, [-INF, -INF], [INF, INF], points=[[1.0, 1.0]], rule="genz-malik")
    assert abs(r.integral - math.pi) < 1e-7 and r.converged
    r = I.cubature(g, [0.0, -1.0], [INF, 1.0], points=[[1.0, 0.0]], rule="genz-malik")
    want = math.sqrt(math.pi) / 2 * math.sqrt(math.pi) * math.erf(1.0)
    assert abs(r.integral - want) < 1e-7 and r.converged


def test_non_finite_values_do_not_converge():
    # scipy's test err > atol + rtol |est| is false for NaN and it reports 'converged' (C103)
    r = I.cubature(lambda X: sd.log(X[:, 0] - 0.5), [0.0, 0.0], [1.0, 1.0], rule="genz-malik")
    assert math.isnan(r.integral) and not r.converged and r.status == 3
    assert r.message == "Non-finite values encountered."


# ---- the Arb references


def _sd_cub(f, a, b, **kw):
    r = I.cubature(lambda X: sd.asarray(f(np.asarray(X.tolist()))), a, b, **kw)
    v = r.integral.tolist() if isinstance(r.integral, sd.ndarray) else [r.integral]
    e = r.error.tolist() if isinstance(r.error, sd.ndarray) else [r.error]
    return v, e, r.status


def test_contract_against_the_references():
    runs = A.cubature_runs(_sd_cub)
    short, worst = {}, 0.0
    for run in runs:
        if run[4] != 0:
            continue
        ratio, known = A.cubature_contract(run)
        if ratio > 1:
            short.setdefault(run[0], set()).add(run[1])
        elif not known:
            worst = max(worst, ratio)
    assert short == A.CUBATURE_SHORT
    assert worst < 0.97


# ---- inputs, errors, result


def test_result_fields():
    r = I.cubature(lambda X: sd.stack([sd.exp(-X[:, 0]), X[:, 1]], axis=1), [0.0, 0.0], [1.0, 2.0], rule="genz-malik")
    assert r.integral.shape == (2,) and r.error.shape == (2,) and r.converged and r.status == 0
    assert r.regions.shape == (r.n_regions, 2, 2) and r.region_integrals.shape == (r.n_regions, 2)
    assert r.region_errors.shape == (r.n_regions, 2) and r.subdivisions >= 1 and r.nfev > 0
    for name in ("integral", "error", "regions", "region_integrals", "region_errors"):
        with pytest.raises(ValueError):
            getattr(r, name)[0] = 0.0
    with pytest.raises(TypeError, match="does not unpack"):
        tuple(r)
    with pytest.raises(AttributeError):
        _ = r.intervals
    s = I.cubature(lambda X: X[:, 0] * X[:, 0], [0.0], [1.0])
    assert isinstance(s.integral, float) and isinstance(s.error, float) and s.region_integrals.shape == (1,)
    assert "Regions:" in s.summary() and "max error" in r.summary()
    assert repr(s).startswith("IntegrateResult(integral=0.333333")
    z = I.cubature(lambda X: X[:, 0], [1.0, 0.0], [1.0, 1.0], points=[[1.0, 0.5]])
    assert z.integral == 0.0 and z.nfev == 0 and z.n_regions == 0 and z.converged


@pytest.mark.parametrize(
    ("call", "err", "msg"),
    [
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0, 2.0]), ValueError, "the same length, not 1 and 2"),
        (lambda: I.cubature(lambda X: X[:, 0], [], []), ValueError, "must not be empty"),
        (lambda: I.cubature(lambda X: X[:, 0], 0.0, 1.0), TypeError, "'a' must be a 1-D sequence"),
        (lambda: I.cubature(lambda X: X[:, 0], sd.zeros((1, 1)), [1.0]), ValueError, "'a' must be 1-dimensional"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], rule="gauss-kronrod"), ValueError,
         "(scipy's 'gauss-kronrod' is 'gk21')"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], rule=3), TypeError, "'rule' must be"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], rule="genz-malik"), ValueError, "ndim >= 2"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], max_subdivisions=0), ValueError, ">= 1"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], max_subdivisions=None), TypeError, "must be an int"),
        (lambda: I.cubature(lambda X: X[:, 0], [INF], [INF]), ValueError, "the same infinity"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [math.nan]), ValueError, "must not be NaN"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], points=[[0.5, 1.0]]), ValueError, "ndim = 1 values"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], points=0.5), TypeError, "'points' must be a sequence"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0, 0.0], [1.0, 1.0], points=[[0.5, math.nan]]), ValueError,
         "the values of 'points' must not be NaN"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], rtol=-1.0), ValueError, "'rtol' must be a finite"),
        (lambda: I.cubature(lambda X: 1.0, [0.0], [1.0]), ValueError,
         "'f' must return shape (21,) or (21, ...) for its argument of shape (21, 1), not a scalar"),
        (lambda: I.cubature(lambda X: X[:5, 0], [0.0], [1.0]), ValueError, "not (5,)"),
        (lambda: I.cubature(lambda X: X[:, :0], [0.0], [1.0]), ValueError, "at least one value per point"),
        (lambda: I.cubature(lambda X: X[:, 0] > 0.5, [0.0], [1.0]), TypeError, "bool"),
        (lambda: I.cubature(lambda X: X[:, 0] if X.shape[0] == 21 else X, [0.0], [1.0]), ValueError,
         "the same shape per point at every call: (31,) here, not (31, 1)"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], args=()), TypeError, "unexpected keyword argument 'args'"),
        (lambda: I.cubature(lambda X: X[:, 0], [0.0], [1.0], workers=2), TypeError, "unexpected keyword argument"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err) as e:
        call()
    assert str(e.value).startswith("cubature(): ") and msg in str(e.value), str(e.value)


def test_user_errors_pass_through():
    with pytest.raises(ZeroDivisionError):
        I.cubature(lambda X: 1 / 0, [0.0], [1.0])


def test_bits_do_not_depend_on_the_thread_count():
    """docs/api/integrate.md §1.4 for cubature (its sums over points are sd's) and the quad family."""

    def f(X):
        return sd.exp(-(X[:, 0] ** 2) - 0.5 * X[:, 1] * X[:, 2]) * sd.cos(X[:, 0] + X[:, 1])

    def runs():
        out = []
        for rule in ("gk21", "gk15", "genz-malik"):
            r = I.cubature(f, [0.0, -1.0, 0.0], [1.0, 1.0, 2.0], rule=rule, rtol=1e-10)
            out.append((r.integral, r.error, r.nfev, r.n_regions))
        r = I.cubature(lambda X: sd.stack([X[:, 0], X[:, 1] ** 2], axis=1), [0.0, 0.0], [INF, 1.0])
        out.append((tuple(np.asarray(r.integral.tolist()).view(np.int64).tolist()), r.nfev))
        q = I.nquad(lambda z, y, x: math.exp(-x * y * z), [(0.0, 1.0), lambda x: (0.0, x), (0.0, 2.0)])
        out.append((q.integral, q.error, q.nfev))
        return out

    old = sd.get_num_threads()
    try:
        got = {}
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            got[t] = runs()
    finally:
        sd.set_num_threads(old)
    assert got[1] == got[3] == got[10]


@pytest.mark.parametrize("bid", sorted(P.BOXES_BY_ID))
def test_boxes_match_live_scipy(bid):
    """problems.CUBATURE_BOXES (4-D and 5-D genz-malik, break points in 2-D and 3-D) bit for bit
    against scipy 1.18.1 (their Arb contract is in test_arb_contract through adaptive.cubature_runs)."""
    import warnings

    from scipy import integrate

    live_scipy()
    bx = P.BOXES_BY_ID[bid]
    kw = {"points": [list(t) for t in bx.points]} if bx.points else {}
    for rule in ("gk21", "gk15", "genz-malik"):
        if rule != "genz-malik" and len(bx.a) > 3:
            continue
        r = I.cubature(lambda X: sd.asarray(bx.func(np.asarray(X.tolist()))), list(bx.a), list(bx.b), rule=rule, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = integrate.cubature(bx.func, np.asarray(bx.a), np.asarray(bx.b), rule=rule, **kw)
        assert (r.integral, r.error, r.subdivisions) == (float(s.estimate), float(s.error), s.subdivisions), rule
