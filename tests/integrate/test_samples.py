"""The sample rules trapezoid, simpson, romb, cumulative_trapezoid and cumulative_simpson
(docs/api/integrate.md §8; docs/design/integrate.md §3.7, decisions C39-C41, C44-C51): the
contract against each rule's formula computed exactly (tests/integrate/ref/samples.json), scipy
1.18.1's values (tests/integrate/data/scipy_samples.json), the summation rule (a lane's value
depends on its numbers only: not on the axis, the layout or the thread count), the inputs' rules
and the errors."""

import hashlib
import math
import pathlib
import sys

import numpy as np
import pytest
import scipy.integrate as si

import stardust as sd

from . import problems as P
from . import refdata

I = sd.integrate  # noqa: E741
EPS = 2.0**-52
RULES = ("trapezoid", "simpson", "romb", "cumulative_trapezoid", "cumulative_simpson")
CUMULATIVE = ("cumulative_trapezoid", "cumulative_simpson")


def _np(v):
    return np.asarray(v.tolist() if isinstance(v, sd.ndarray) else v)


def _call(rule, y, x, dx, lib=I, wrap=sd.asarray):
    kw = {"x": wrap(x)} if x is not None else {"dx": dx}
    return getattr(lib, rule)(wrap(y), **kw)


def _applies(rule, y, x):
    n = y.shape[-1] - 1
    return rule != "romb" or (x is None and n >= 1 and n & (n - 1) == 0)


def _scale(y, x, dx):
    """sum |d_i| (|y_i| + |y_i+1|) along the last axis, running: the size of the terms."""
    d = np.abs(np.diff(x)) if x is not None else np.full(y.shape[-1] - 1, abs(dx))
    a = np.abs(y)
    return np.cumsum(d * (a[..., 1:] + a[..., :-1]), axis=-1)


# ---- the contract: the rule's formula computed exactly from the same doubles


def test_references_are_current():
    tools = pathlib.Path(__file__).parents[2] / "tools"
    sys.path.insert(0, str(tools))
    try:
        import integrate_ref
    finally:
        sys.path.remove(str(tools))
    ref = refdata.samples(keep_provenance=True)
    prov = ref.pop("_provenance")
    assert prov["tool"] == "tools/integrate_ref.py" and "exact rationals" in prov["arb"]
    for sid, (y, x, dx) in P.samples().items():
        assert ref[sid]["digest"] == integrate_ref.samples_digest(sid, y, x, dx), f"{sid}: stale reference"


def test_exact_contract():
    """|sd - exact| <= c eps S: S the terms' size, c = ceil(log2 n) + 10 for a sum (the pairwise
    tree) and k + 10 for the k-th running value (a scan); measured: the largest error over c eps S
    is CONTRACT_MAX of the bound, and every rule is within a few eps of the exact value relative."""
    ref = refdata.samples()
    assert set(ref) == set(P.samples())
    worst = 0.0
    for sid, (y, x, dx) in P.samples().items():
        e = ref[sid]
        scale = _scale(y, x, dx)
        for rule in RULES:
            if not _applies(rule, y, x):
                assert rule not in e
                continue
            got = _np(_call(rule, y, x, dx)).astype(float)
            r = np.asarray(e[rule], dtype=float)
            err = np.abs((got - r[..., 0]) - r[..., 1])
            if rule in CUMULATIVE:
                k = np.arange(1, err.shape[-1] + 1)
                bound = (k + 10) * EPS * np.maximum(scale, 1e-300)
            else:
                n = y.shape[-1]
                bound = (math.ceil(math.log2(max(n, 2))) + 10) * EPS * np.maximum(scale[..., -1], 1e-300)
            ratio = float(np.max(err / bound))
            assert ratio <= 1, (sid, rule, ratio)
            worst = max(worst, ratio)
    assert worst < 0.06  # 0.0501, the largest ratio measured (docs/design/integrate.md §7.9)


# ---- scipy 1.18.1


def _scipy_equal(rec, got):
    if isinstance(rec, dict):  # a long array: its digest
        a = np.ascontiguousarray(np.asarray(got, dtype=np.float64))
        return hashlib.sha256(a.tobytes()).hexdigest() == rec["sha256"]
    return np.array_equal(np.asarray(rec, dtype=float), np.asarray(got, dtype=float))


def test_scipy_values():
    rec = refdata.data("scipy_samples.json")["samples"]
    same = total = 0
    for sid, (y, x, dx) in P.samples().items():
        r = rec[sid]
        for key, val in r.items():
            rule, _, opt = key.partition("/")
            kw = {"x": sd.asarray(x)} if x is not None else {"dx": dx}
            if opt == "initial=0":
                kw["initial"] = 0
            got = _np(getattr(I, rule)(sd.asarray(y), **kw))
            total += 1
            same += _scipy_equal(val, got)
            if not isinstance(val, dict):  # within a few eps of scipy's
                v = np.asarray(val, dtype=float)
                assert np.all(np.abs(got - v) <= 32 * EPS * np.maximum(np.abs(v), 1e-300)), (sid, key)
    assert total == 65
    assert same == 59  # the others differ in the last digits: the sums' order (D-33, §7.9)


# ---- the summation rule


def test_lane_value_is_independent_of_axis_and_layout():
    rng = np.random.default_rng(3)
    y = rng.standard_normal((5, 1001))
    y2 = rng.standard_normal((5, 1025))
    x = np.cumsum(rng.uniform(0.1, 1.0, 1001))
    for rule in RULES:
        yy, xx = (y2, None) if rule == "romb" else (y, x)
        kw = {"x": sd.asarray(xx)} if xx is not None else {"dx": 0.25}
        rows = [_np(getattr(I, rule)(sd.asarray(yy[i]), **kw)) for i in range(5)]
        base = _np(getattr(I, rule)(sd.asarray(yy), **kw))
        t = sd.asarray(np.ascontiguousarray(yy.T))
        along0 = _np(getattr(I, rule)(t, axis=0, **kw))
        if rule in CUMULATIVE:
            along0 = along0.T
        strided = sd.asarray(np.repeat(yy, 2, axis=1))[:, ::2]
        for i in range(5):
            assert np.array_equal(base[i], rows[i]), rule
            assert np.array_equal(along0[i], rows[i]), rule
        assert np.array_equal(_np(getattr(I, rule)(strided, **kw)), base), rule


@pytest.mark.usefixtures("restore_threads")
def test_threads():
    rng = np.random.default_rng(4)
    one = rng.standard_normal(2**17 + 1)
    many = rng.standard_normal((40, 4097))
    x1 = np.cumsum(rng.uniform(0.5, 1.5, one.size))
    out = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got = []
        for rule in RULES:
            for y in (one, many):
                kw = {"x": sd.asarray(x1 if y is one else x1[: y.shape[-1]])} if rule != "romb" else {"dx": 0.5}
                got.append(_np(getattr(I, rule)(sd.asarray(y), **kw)))
        out[t] = got
    for t in (3, 10):
        assert all(np.array_equal(a, b) for a, b in zip(out[1], out[t]))


# ---- the rules' edges


def test_short_inputs():
    for n in (0, 1):
        assert I.trapezoid([0.0] * n) == 0.0 and I.simpson([1.0] * n) == 0.0  # scipy: simpson of none IndexError
    assert I.simpson([1.0, 2.0]) == 1.5 == I.trapezoid([1.0, 2.0])
    assert I.romb([1.0, 3.0], dx=2.0) == 4.0
    assert I.cumulative_trapezoid([2.0]).shape == (0,)
    assert I.cumulative_simpson([2.0], initial=1.5).tolist() == [1.5]
    assert I.cumulative_simpson([1.0, 2.0]).tolist() == [1.5]  # under three points: the trapezoidal rule
    assert I.cumulative_simpson([1.0, 2.0, 3.0, 4.0], initial=1.0).tolist() == [1.0, 2.5, 5.0, 8.5]
    assert I.cumulative_trapezoid([1.0, 2.0, 3.0], initial=0).tolist() == [0.0, 1.5, 4.0]
    e = I.trapezoid(sd.zeros((0, 4)))
    assert isinstance(e, sd.ndarray) and e.shape == (0,)
    assert I.cumulative_simpson(sd.zeros((3, 0)), axis=0).shape == (2, 0)


def test_simpson_even_points_matches_scipy():
    # scipy 1.11's rule: Simpson on the first intervals, Cartwright's correction on the last
    for n in (4, 6, 10):
        x = np.linspace(0, 2, n) ** 1.5
        y = np.sin(x)
        assert I.simpson(y.tolist(), x.tolist()) == pytest.approx(si.simpson(y, x=x), rel=4 * EPS)
        assert I.simpson(y.tolist(), dx=0.3) == pytest.approx(si.simpson(y, dx=0.3), rel=4 * EPS)
    assert I.simpson([0.0, 1.0, 4.0, 9.0], dx=1.0) == 9.0  # exact for x^2


def test_complex_and_conversions():
    y = [1, 2j, 3, 4]
    assert I.simpson(y) == complex(si.simpson(np.array(y)))
    assert I.trapezoid([True, False, True]) == 1.0
    assert I.trapezoid(np.arange(4)) == 4.5
    s = sd.frame.Series([1.0, None, 3.0])
    assert math.isnan(I.trapezoid(s))
    assert I.trapezoid(sd.frame.Series([1.0, 2.0, 3.0]), sd.frame.Series([0.0, 1.0, 3.0])) == 6.5
    c = I.cumulative_simpson(np.array(y, dtype=complex), initial=1 + 1j)
    assert c.dtype == sd.complex128
    assert np.allclose(_np(c), si.cumulative_simpson(np.array(y), initial=1 + 1j), rtol=4 * EPS)
    assert math.isnan(I.trapezoid([1.0, math.nan, 2.0])) and math.isinf(I.simpson([1.0, math.inf, 2.0]))


def test_per_lane_dx_and_initial():
    y = np.arange(12.0).reshape(3, 4)
    dx = np.array([[0.5], [1.0], [2.0]])
    got = _np(I.cumulative_simpson(y, dx=dx, initial=dx))
    assert np.allclose(got, si.cumulative_simpson(y, dx=dx, initial=dx), rtol=4 * EPS, atol=0)
    for rule in ("trapezoid", "simpson", "cumulative_trapezoid"):
        rows = [_np(getattr(I, rule)(y[i], dx=float(dx[i, 0]))) for i in range(3)]
        assert np.array_equal(_np(getattr(I, rule)(y, dx=dx)), np.array(rows)), rule
    assert I.romb(sd.ones((2, 5)), dx=[[1.0], [2.0]]).tolist() == [4.0, 8.0]


def test_x_forms():
    y = np.arange(12.0).reshape(3, 4)
    x1 = np.array([0.0, 1.0, 3.0, 4.0])
    full = np.vstack([x1, 2 * x1, 3 * x1])
    assert np.array_equal(_np(I.trapezoid(y, x1)), si.trapezoid(y, x=x1))
    assert np.array_equal(_np(I.trapezoid(y, x=full)), si.trapezoid(y, x=full))
    assert np.array_equal(_np(I.trapezoid(y, x1[None, :])), _np(I.trapezoid(y, x1)))  # broadcasts
    x0 = np.array([0.0, 2.0, 5.0])
    assert np.array_equal(_np(I.cumulative_trapezoid(y, x0, axis=0)), si.cumulative_trapezoid(y, x=x0, axis=0))


def test_errors():
    def err(kind, text, f):
        with pytest.raises(kind) as e:
            f()
        assert str(e.value) == text

    err(ValueError, "trapezoid(): 'y' must be at least 1-dimensional, not a scalar", lambda: I.trapezoid(1.0))
    err(TypeError, "trapezoid(): 'y' must be an sd.ndarray, a sequence of numbers or a frame Series, not str",
        lambda: I.trapezoid("abc"))  # fmt: skip
    err(TypeError, "simpson(): 'x' must be real, not a complex128 array", lambda: I.simpson([1.0, 2.0], [0, 1j]))
    err(
        ValueError,
        "simpson(): 'x' must have y's length 3 along 'axis', not 2",
        lambda: I.simpson([1.0, 2.0, 3.0], [0.0, 1.0]),
    )
    err(ValueError, "trapezoid(): 'x' must be 1-dimensional or have the 2 dimensions of 'y', not 3",
        lambda: I.trapezoid(sd.ones((2, 3)), sd.ones((1, 2, 3))))  # fmt: skip
    err(ValueError, "trapezoid(): 'x' of shape (3, 3) does not broadcast to the shape (2, 3) of 'y'",
        lambda: I.trapezoid(sd.ones((2, 3)), sd.ones((3, 3))))  # fmt: skip
    err(ValueError, "trapezoid(): 'dx' is not used with 'x' (the spacing comes from 'x')",
        lambda: I.trapezoid([1.0, 2.0], [0.0, 1.0], dx=2.0))  # fmt: skip
    err(TypeError, "trapezoid(): 'axis' must be an int, not float", lambda: I.trapezoid([1.0, 2.0], axis=0.0))
    err(
        ValueError,
        "trapezoid(): axis 1 is out of bounds for an array of dimension 1",
        lambda: I.trapezoid([1.0], axis=1),
    )
    err(ValueError, "romb(): 'y' must have 2^k + 1 samples along 'axis' (k >= 0), not 4", lambda: I.romb([1.0] * 4))
    err(ValueError, "romb(): 'y' must have 2^k + 1 samples along 'axis' (k >= 0), not 1", lambda: I.romb([1.0]))
    err(ValueError, "cumulative_trapezoid(): 'y' must have at least one point along 'axis'",
        lambda: I.cumulative_trapezoid([]))  # fmt: skip
    err(ValueError, "cumulative_trapezoid(): 'initial' must be None or 0, not 1.0",
        lambda: I.cumulative_trapezoid([1.0, 2.0], initial=1.0))  # fmt: skip
    err(TypeError, "cumulative_trapezoid(): 'initial' must be None or 0, not bool",
        lambda: I.cumulative_trapezoid([1.0, 2.0], initial=False))  # fmt: skip
    err(ValueError, "cumulative_simpson(): 'x' must be strictly increasing along 'axis'",
        lambda: I.cumulative_simpson([1.0, 2.0, 3.0], [0.0, 2.0, 1.0]))  # fmt: skip
    err(ValueError, "cumulative_simpson(): 'dx' must be a number or an array of shape (2, 1) (the shape of 'y' with "
        "1 along 'axis'), not shape (2,)", lambda: I.cumulative_simpson(sd.ones((2, 3)), dx=[1.0, 2.0]))  # fmt: skip
    err(TypeError, "cumulative_simpson(): 'initial' must be real for a real y, not complex",
        lambda: I.cumulative_simpson([1.0, 2.0, 3.0], initial=1j))  # fmt: skip
    err(TypeError, "simpson(): 'dx' must be a number or an array, not bool", lambda: I.simpson([1.0, 2.0], dx=True))
    err(TypeError, "romb(): unexpected keyword argument 'show'", lambda: I.romb([1.0, 2.0], show=True))
    err(TypeError, "trapezoid(): takes from 1 to 2 positional arguments but 3 were given",
        lambda: I.trapezoid([1.0, 2.0], None, 1.0))  # fmt: skip


def test_cumulative_simpson_does_not_catch_nan_positions():
    """docs/api/integrate.md §8 (C50): x must increase, but a NaN in x is not caught (scipy's):
    it spreads into the running values."""
    v = I.cumulative_simpson([1.0, 2.0, 3.0, 4.0], x=[0.0, math.nan, 2.0, 3.0])
    assert all(math.isnan(t) for t in v.tolist())


def variant_errors(rule, y, x, dx, axis, initial, entry, lib=I, wrap=sd.asarray):
    """(the largest error over c eps S, the largest relative error) of one sample variant against
    its exact lanes, the real and imaginary parts each with their own S (test_exact_contract's bound)."""
    kw = {"x": wrap(x)} if x is not None else {"dx": dx}
    if initial is not None:
        kw["initial"] = initial
    got = _np(getattr(lib, rule)(wrap(y), axis=axis, **kw))
    lanes = np.moveaxis(got, axis, -1) if rule in CUMULATIVE else got
    lanes = lanes.reshape(-1, lanes.shape[-1]) if rule in CUMULATIVE else lanes.reshape(-1, 1)
    ys = np.moveaxis(y, axis, -1).reshape(-1, y.shape[axis])
    worst = rel = 0.0
    for key, part, ypart in (("lanes", np.real, np.real), ("imag", np.imag, np.imag)):
        if key not in entry:
            continue
        r = np.asarray(entry[key], dtype=float).reshape(lanes.shape + (2,))
        g = part(lanes).astype(float)
        err = np.abs((g - r[..., 0]) - r[..., 1])
        scale = np.maximum(_scale(ypart(ys), x, dx), 1e-300)
        if rule in CUMULATIVE:
            k = np.arange(1, scale.shape[-1] + 1)
            bound = (k + 10) * EPS * scale
            if initial is not None:  # the initial first, then each running value (one more rounding)
                bound = np.concatenate([np.full((bound.shape[0], 1), EPS), bound + EPS * abs(initial)], axis=1)
        else:
            bound = (math.ceil(math.log2(max(ys.shape[-1], 2))) + 10) * EPS * scale[..., -1:]
        worst = max(worst, float(np.max(err / np.maximum(bound, 1e-300))))
        rel = max(rel, float(np.max(err / np.maximum(np.abs(r[..., 0]), 1e-300))))
    return worst, rel


def test_variants_exact_contract():
    """docs/api/integrate.md §8: an axis other than the last, complex y (C48: the real coefficients
    on each part) and `initial` are within the rules' bound of their exact values
    (ref/sample_variants.json, tools/integrate_ref.py sample_variants)."""
    ref = refdata.sample_variants()
    variants = P.sample_variants()
    assert set(ref) == {v[0] for v in variants}
    for vid, rule, y, x, dx, axis, initial in variants:
        worst, _ = variant_errors(rule, y, x, dx, axis, initial, ref[vid])
        assert worst <= 1, (vid, worst)


@pytest.mark.parametrize("shape", [(2, 3), (1000, 300), (17, 5), (513, 7), (1025, 3), (600, 2, 300)])
@pytest.mark.parametrize("dx", [1.0, 0.37])
def test_trapezoid_outer_axis_has_the_lane_bits(shape, dx):
    # an outer axis (lanes side by side in memory) is summed in blocks of lanes through the same
    # pairwise tree as one lane at a time: the bits of the same lanes made contiguous
    y = np.random.default_rng(1).standard_normal(shape)
    got = np.asarray(sd.integrate.trapezoid(sd.asarray(y), axis=0, dx=dx))
    lanes = np.ascontiguousarray(np.moveaxis(y, 0, -1))
    want = np.asarray(sd.integrate.trapezoid(sd.asarray(lanes), axis=-1, dx=dx))
    assert np.array_equal(got, want)
