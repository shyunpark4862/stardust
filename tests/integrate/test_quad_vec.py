"""quad_vec (docs/api/integrate.md §4; docs/design/integrate.md §3.3, §7.13; decisions C5, C10,
C94-C107).

Against scipy 1.18.1's records (tests/integrate/data/scipy_vector.json) and the live scipy: the same
points in the same order (the sha256 of their bytes), the same integral, error, nfev, status and
intervals to the bit, on the records, the live cases of tests/integrate/adaptive.py, the
collection's integrals and families, and a run past scipy's cache of the intervals' integrals.
Against the Arb references: a converged run's true error is within error + 8 eps |ref| but for
adaptive.QUAD_VEC_SHORT (scipy's too). And the inputs, the errors and the result."""

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
_SCIPY_NAMES = {"atol": "epsabs", "rtol": "epsrel"}


class Recorder:
    """f's points in call order (the sha256 of their float64 bytes, their count)."""

    def __init__(self, fn):
        self.fn, self.h, self.count = fn, hashlib.sha256(), 0

    def __call__(self, x):
        self.h.update(np.float64(x).tobytes())
        self.count += 1
        return self.fn(x)


def same(x, y):
    x = np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return x.shape == y.shape and bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))))


def scipy_run(fn, a, b, **kw):
    from scipy import integrate

    rec = Recorder(fn)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        y, err, info = integrate.quad_vec(
            rec, a, b, full_output=True, **{_SCIPY_NAMES.get(k, k): v for k, v in kw.items()}
        )
    return rec, y, err, info


def assert_same_as_scipy(fn, a, b, **kw):
    live_scipy()
    rs, y, err, info = scipy_run(fn, a, b, **kw)
    rd = Recorder(fn)
    with np.errstate(all="ignore"):
        r = I.quad_vec(rd, a, b, **kw)
    assert rd.h.digest() == rs.h.digest() and rd.count == rs.count
    assert same(r.integral, y) and same(r.error, err)
    assert (r.nfev, r.status, r.converged, r.message) == (info.neval, info.status, info.success, info.message)
    # the intervals: scipy's are in its heap's order, sd's from the largest error down (C95)
    assert r.n_intervals == len(info.intervals)
    sign = -1.0 if (math.isinf(a) and not math.isinf(b)) or (math.isinf(a) and math.isinf(b) and b < a) else 1.0
    want = {
        (iv[0], iv[1]): (np.asarray(v) * sign, e)
        for iv, v, e in zip(info.intervals.tolist(), info.integrals, info.errors.tolist())
    }
    got = list(zip(r.intervals.tolist(), r.interval_integrals.tolist(), r.interval_errors.tolist()))
    assert len(want) == len({(iv[0], iv[1]) for iv, _, _ in got}) or a == b
    for iv, v, e in got:
        wv, we = want[(iv[0], iv[1])]
        assert same(v, wv) and same(e, we), iv
    errs = r.interval_errors.tolist()
    if not any(math.isnan(x) for x in errs):  # heappop's order: the largest error first
        assert all(x >= y for x, y in zip(errs, errs[1:]))
    return r


# ---- scipy's records


@pytest.mark.parametrize("v", P.VECTOR, ids=lambda v: v.id)
def test_records(v):
    rec = refdata.data("scipy_vector.json")[v.id]
    f = Recorder(v.func)
    r = I.quad_vec(f, v.a, v.b)
    assert f.h.hexdigest() == rec["calls"]["sha256"] and f.count == rec["calls"]["count"] == r.nfev
    assert same(r.integral, rec["integral"]) and same(r.error, rec["error"])
    assert (r.nfev, r.status, r.converged, r.n_intervals) == (rec["nfev"], rec["status"], rec["converged"],
                                                              rec["intervals"])  # fmt: skip


# ---- the live scipy


@pytest.mark.parametrize("case", A.QUAD_VEC_LIVE, ids=lambda c: c[0])
def test_live_cases(case):
    _, fn, a, b, kw = case
    assert_same_as_scipy(fn, a, b, **kw)


def test_live_statuses():
    st = {c[0]: I.quad_vec(c[1], c[2], c[3], **c[4]).status for c in A.QUAD_VEC_LIVE}
    assert st["limit 40"] == 1 and st["limit 1"] == 1 and st["NaN"] == 3 and st["rounding"] == 2
    assert st["scalar"] == 0 and st["tolerances 0"] in (1, 2)


@pytest.mark.parametrize("p", [p for p in P.INTEGRALS if p.weight is None], ids=lambda p: p.id)
def test_collection_matches_scipy(p):
    kw = {"points": list(p.points)} if p.points else {}
    assert_same_as_scipy(A._nan_safe(p), p.a, p.b, **kw)


@pytest.mark.parametrize("norm", ["2", "max"])
@pytest.mark.parametrize("bt", P.BATCHES, ids=lambda b: b.family)
def test_families_match_scipy(bt, norm):
    params = np.array(bt.params)

    def f(x):
        return bt.f_np(np.full(len(params), x), params)

    assert_same_as_scipy(f, bt.a, bt.b, norm=norm)


def test_past_scipys_cache():
    # scipy keeps the intervals' integrals in an LRU cache of 100e6 bytes; with 8 * 50000-byte values
    # it holds 249 of them, and an interval halved after its integral was evicted is integrated again
    # (21 more calls); its full_output has NaN for the evicted ones (C98)
    c = np.linspace(0.0, 1.0, 50000)

    def f(x):
        return np.sqrt(np.abs(x - c))

    r = assert_same_as_scipy(f, 0.0, 1.0, rtol=1e-6, limit=400)
    assert r.n_intervals > 249 and r.nfev > 42 * (r.n_intervals - 1) // 2
    assert np.isnan(np.asarray(r.interval_integrals[:, 0].tolist())).sum() == r.n_intervals - 249


def test_large_values_norm_is_deterministic():
    # from 897 values Accelerate's dot product depends on the address; sd's buffer is page-aligned
    k = np.linspace(0.0, 30.0, 3000)
    runs = [I.quad_vec(lambda x: np.cos(k * x), 0.0, 2.0) for _ in range(3)]
    assert all(r.error == runs[0].error and r.nfev == runs[0].nfev for r in runs)
    assert all(same(r.integral, np.asarray(runs[0].integral.tolist())) for r in runs)


def test_threads_do_not_matter():
    f = lambda x: np.array([math.exp(-x * x), math.cos(40 * x)])  # noqa: E731
    try:
        sd.set_num_threads(1)
        one = I.quad_vec(f, 0.0, 5.0)
        sd.set_num_threads(10)
        ten = I.quad_vec(f, 0.0, 5.0)
    finally:
        sd.set_num_threads(10)
    assert same(one.integral, np.asarray(ten.integral.tolist())) and one.error == ten.error


# ---- the Arb references


def _sd_wrap(f, a, b, **kw):
    r = I.quad_vec(f, a, b, **kw)
    v = r.integral.tolist() if isinstance(r.integral, sd.ndarray) else r.integral
    return v, r.error, r.status


def test_contract_against_the_references():
    runs = A.quad_vec_runs(_sd_wrap)
    short, worst = set(), 0.0
    for run in runs:
        if run[4] != 0:
            continue
        ratio, known = A.quad_vec_contract(run)
        if ratio > 1:
            short.add(run[1])
        elif not known:
            worst = max(worst, ratio)
    assert short == A.QUAD_VEC_SHORT
    assert worst < 0.95
    assert sum(run[4] == 0 for run in runs) == 99 + 4 + 2  # collection 99 of 106; the families but osc


# ---- inputs, errors, result


def test_result_fields_and_unpacking():
    r = I.quad_vec(lambda x: np.array([math.exp(x), x]), 0.0, 1.0)
    v, e = r
    assert isinstance(v, sd.ndarray) and v.shape == (2,) and e == r.error and len(r) == 2
    for name in ("integral", "intervals", "interval_integrals", "interval_errors"):
        with pytest.raises(ValueError):
            getattr(r, name)[0] = 0.0
    assert r.intervals.shape == (2, 2) and r.interval_integrals.shape == (2, 2) and r.n_intervals == 2
    assert r.message == "Target precision reached." and r.converged and r.status == 0
    s = I.quad_vec(lambda x: x * x, 0.0, 1.0)
    assert isinstance(s.integral, float) and s.interval_integrals.shape == (2,)
    assert "Intervals:" in s.summary() and "shape (2,)" in r.summary()
    assert repr(s).startswith("IntegrateResult(integral=0.333333")
    with pytest.raises(AttributeError):
        _ = s.subdivisions


@pytest.mark.parametrize(
    ("call", "err", "msg"),
    [
        (lambda: I.quad_vec(lambda x: np.zeros(3) if x < 0.5 else np.zeros(2), 0.0, 1.0), ValueError,
         "same shape at every point, not (2,) and then (3,)"),
        (lambda: I.quad_vec(lambda x: 1.0 if x < 0.5 else np.zeros(2), 0.0, 1.0), ValueError, "and then a scalar"),
        (lambda: I.quad_vec(lambda x: 1j, 0.0, 1.0), TypeError, "real values"),
        (lambda: I.quad_vec(lambda x: "a", 0.0, 1.0), TypeError, "not str"),
        (lambda: I.quad_vec(lambda x: np.zeros(0), 0.0, 1.0), ValueError, "non-empty"),
        (lambda: I.quad_vec(lambda x: np.array([True]), 0.0, 1.0), TypeError, "bool"),
        (lambda: I.quad_vec(lambda x: x, 0.0, math.nan), ValueError, "'b' must be a number or +-inf"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, norm=2), TypeError, "'norm' must be '2' or 'max'"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, norm="fro"), ValueError, "'norm' must be '2' or 'max'"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, limit=0), ValueError, "'limit' must be >= 1"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, limit=10.0), TypeError, "'limit' must be an int"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, quadrature="gk31"), ValueError, "'quadrature' must be"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, epsabs=1.0), TypeError, "unexpected keyword argument 'epsabs'"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, rtol=-1.0), ValueError, "'rtol' must be a finite number >= 0"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, points=[math.nan]), ValueError, "must not be NaN"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, points=0.5), TypeError, "'points' must be a sequence"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, args=(1,)), TypeError, "unexpected keyword argument 'args'"),
        (lambda: I.quad_vec(lambda x: x, 0.0, 1.0, workers=2), TypeError, "unexpected keyword argument 'workers'"),
        (lambda: I.quad_vec(3, 0.0, 1.0), TypeError, "'f' must be callable"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err) as e:
        call()
    assert str(e.value).startswith("quad_vec(): ") and msg in str(e.value)


def test_user_errors_pass_through():
    with pytest.raises(ZeroDivisionError):
        I.quad_vec(lambda x: 1 / 0, 0.0, 1.0)


def test_value_forms():
    ref = I.quad_vec(lambda x: np.array([x, 2 * x]), 0.0, 1.0)
    for f in (lambda x: [x, 2 * x], lambda x: sd.asarray([x, 2 * x]), lambda x: (x, 2 * x)):
        r = I.quad_vec(f, 0.0, 1.0)
        assert same(r.integral, np.asarray(ref.integral.tolist())) and r.error == ref.error
    # float32 values are computed in float64 (scipy computes them in float32, C104)
    f32 = I.quad_vec(lambda x: np.array([x, 2 * x], dtype=np.float32), 0.0, 1.0)
    assert f32.integral.dtype == sd.float64 and f32.converged
    s = I.quad_vec(lambda x: np.float64(x), 0.0, 1.0)
    assert abs(s.integral - 0.5) < 1e-15 and isinstance(s.integral, float)
