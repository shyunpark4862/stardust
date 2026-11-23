"""make_splrep, make_splprep and generate_knots (docs/api/smooth.md §3; docs/design/smooth.md §3.3, §4.3; SM3
milestone): scipy 1.18.1's knots (the same knot vectors, live), the smoothing condition, s = 0 bit for bit equal
to make_interp_spline, the port's pieces against scipy's own (fpknot, the least-squares solve)."""

import math
import warnings

import numpy as np
import pytest
import scipy.interpolate as si
from scipy.interpolate import _bsplines as B
from scipy.interpolate import _dierckx

import stardust as sd

from . import samples as S
from .conftest import LIVE

C = sd._core.smooth
A = sd.asarray


def same_knots(ts, mine):
    return len(ts) == len(mine) and all(np.asarray(b).tobytes() == np.asarray(a).tobytes() for a, b in zip(ts, mine))


def fp_of(spl, x, y, w=None):
    r = np.asarray(spl(A(x))) - y
    if w is not None:
        r = r * w
    return float(np.sum(r * r))


@LIVE
def test_fpknot_is_scipys():
    g = np.random.default_rng(3)
    for trial in range(2000):
        m = int(g.integers(6, 40))
        k = int(g.integers(1, 6))
        x = np.sort(g.random(m))
        if trial % 7 == 0:
            x = np.sort(np.round(x * 5) / 5)  # repeated x
        ni = int(g.integers(0, max(1, m - k - 2)))
        inner = np.unique(g.choice(x[1:-1], size=min(ni, m - 2), replace=False)) if ni else np.zeros(0)
        t = np.r_[[x[0]] * (k + 1), inner, [x[-1]] * (k + 1)]
        res = g.random(m) ** 3
        try:
            want = _dierckx.fpknot(x, t, k, res)
        except Exception:
            want = math.nan
        got = C.fitpack_fpknot(A(x), A(t), k, A(res))
        assert got == want or (math.isnan(got) and math.isnan(want)), trial


@LIVE
@pytest.mark.parametrize("periodic", [False, True])
def test_least_squares_is_scipys(periodic):
    g = np.random.default_rng(4)
    for _ in range(60):
        m = int(g.integers(12, 60))
        k = int(g.integers(1, 6))
        x = np.sort(g.random(m))
        x[0], x[-1] = 0.0, 1.0
        y = np.sin(2 * np.pi * x) + 0.1 * g.standard_normal(m)
        w = 0.5 + g.random(m)
        if periodic:
            y[-1] = y[0]
            xt = x[::3] if x[::3][-1] == 1.0 else np.r_[x[::3], 1.0]
            t = B._periodic_knots(xt, k)
        else:
            t = np.r_[[0.0] * (k + 1), x[2:-2:3], [1.0] * (k + 1)]
        _, _, c, fp, res = B._lsq_solve_qr(x, y[:, None], t, k, w, periodic=periodic)
        c2, res2, fp2 = C.fitpack_lsq(A(x), A(y[:, None].copy()), A(w), A(t), k, periodic)
        assert np.abs(np.asarray(c2)[:, 0] - c[:, 0]).max() <= 1e-12 * np.abs(c).max()
        assert np.abs(np.asarray(res2) - res).max() <= 1e-12 * res.max()


def _random_sets(n_sets, seed):
    g = np.random.default_rng(seed)
    for trial in range(n_sets):
        m = int(g.integers(20, 300))
        k = int(g.choice([1, 2, 3, 4, 5]))
        kind = trial % 3
        if kind == 0:
            x = np.sort(10 * g.random(m))
            y = np.sin(x) + 0.3 * g.standard_normal(m)
            sig = 0.3
        elif kind == 1:
            x = np.linspace(0, 10, m)
            y = np.exp(-((x - 5) ** 2)) + 0.05 * g.standard_normal(m)
            sig = 0.05
        else:
            x = np.sort(g.random(m))
            y = np.where(x < 0.5, 0.0, 1.0) + 0.05 * g.standard_normal(m)
            sig = 0.05
        s = float(m * sig**2 * g.choice([0.3, 1.0, 3.0]))
        yield trial, x, y, k, s


@LIVE
def test_knots_are_scipys():
    """generate_knots and make_splrep's knots are scipy's, knot vector for knot vector, on 300 random sets and the
    samples (none diverged when this was written: decision SM63); the splines then agree to rounding."""
    diverging = []
    for trial, x, y, k, s in _random_sets(300, 3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = list(si.generate_knots(x, y, k=k, s=s))
            sp = si.make_splrep(x, y, k=k, s=s)
        mine = sd.smooth.generate_knots(A(x), A(y), k=k, s=s)
        if not same_knots(ts, mine):
            diverging.append(trial)
            continue
        my = sd.smooth.make_splrep(A(x), A(y), k=k, s=s)
        xs = np.linspace(x[0], x[-1], 400)
        want = sp(xs)
        assert np.abs(want - np.asarray(my(A(xs)))).max() <= 1e-9 * max(1.0, np.abs(want).max()), trial
        assert abs(fp_of(my, x, y) - s) <= 0.001 * s or fp_of(my, x, y) <= s
    assert diverging == []


@LIVE
def test_knots_are_scipys_on_the_samples():
    for sid in S.scatter_ids(tag="sorted", exclude=("subnormal", "degenerate", "tiny", "magnitude", "ties")):
        s0 = S.scatter(sid)
        if s0.w is not None:
            continue
        for f in (0.1, 1.0):
            s = float(s0.n * 0.09 * f) + 1e-3
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ts = list(si.generate_knots(s0.x, s0.y, s=s))
            assert same_knots(ts, sd.smooth.generate_knots(A(s0.x), A(s0.y), s=s)), (sid, f)


@LIVE
def test_periodic_knots_are_scipys():
    g = np.random.default_rng(11)
    for trial in range(80):
        m = int(g.integers(15, 200))
        k = int(g.choice([1, 2, 3, 4, 5]))
        x = np.sort(g.random(m))
        x[0], x[-1] = 0.0, 1.0
        y = np.sin(2 * np.pi * x) + 0.1 * g.standard_normal(m)
        y[-1] = y[0]
        s = float(m * 0.01 * g.choice([0.3, 1.0, 3.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = list(si.generate_knots(x, y, k=k, s=s, bc_type="periodic"))
        mine = sd.smooth.generate_knots(A(x), A(y), k=k, s=s, bc_type="periodic")
        assert same_knots(ts, mine), trial
        sp = sd.smooth.make_splrep(A(x), A(y), k=k, s=s, bc_type="periodic")
        assert sp.extrapolate == "periodic"
        fp = fp_of(sp, x[:-1], y[:-1])
        assert abs(fp - s) <= 0.001 * s or fp <= s, trial


@LIVE
def test_splprep_is_scipys():
    g = np.random.default_rng(12)
    for _ in range(40):
        m = int(g.integers(15, 200))
        k = int(g.choice([1, 2, 3, 5]))
        th = np.sort(g.random(m)) * 2 * np.pi
        X = np.stack([np.cos(th), np.sin(2 * th), th / 7]) + 0.02 * g.standard_normal((3, m))
        s = float(m * 3 * 0.02**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp, u = si.make_splprep(X, k=k, s=s)
        my, mu = sd.smooth.make_splprep(A(X), k=k, s=s)
        assert np.asarray(mu).tobytes() == u.tobytes()  # scipy's sums in scipy's order
        assert np.asarray(my.t).tobytes() == sp.t.tobytes()
        us = np.linspace(0, 1, 100)
        assert np.abs(sp(us) - np.asarray(my(A(us)))).max() <= 1e-9
        assert my.axis == 1 and np.asarray(my(A(us))).shape == (3, 100)


@LIVE
def test_splprep_large_curve():
    """10^4 points of a noisy 3-D curve: the chord-length parameters and the knots are scipy's bit for bit."""
    g = np.random.Generator(np.random.PCG64([20261126, 10**4]))
    th = np.sort(g.random(10**4)) * 2 * np.pi
    X = np.stack([np.cos(th), np.sin(2 * th), th / 7]) + 0.1 * g.standard_normal((3, 10**4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp, u = si.make_splprep(X, s=0.05 * 10**4)
    my, mu = sd.smooth.make_splprep(A(X), s=0.05 * 10**4)
    assert np.asarray(mu).tobytes() == u.tobytes()
    assert np.asarray(my.t).tobytes() == sp.t.tobytes()


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_s_zero_is_interpolation_bit_for_bit(k):
    s = S.scatter("random/50")
    a = sd.smooth.make_splrep(A(s.x), A(s.y), k=k)
    b = sd.interpolate.make_interp_spline(A(s.x), A(s.y), k=k)
    assert np.asarray(a.t).tobytes() == np.asarray(b.t).tobytes()
    assert np.asarray(a.c).tobytes() == np.asarray(b.c).tobytes()
    y = s.y.copy()
    y[-1] = y[0]
    a = sd.smooth.make_splrep(A(s.x), A(y), k=k, bc_type="periodic")
    b = sd.interpolate.make_interp_spline(A(s.x), A(y), k=k, bc_type="periodic")
    assert np.asarray(a.c).tobytes() == np.asarray(b.c).tobytes()
    X = np.stack([np.cos(s.x), np.sin(s.x)])
    sp, u = sd.smooth.make_splprep(A(X), k=k)
    ref = sd.interpolate.make_interp_spline(u, A(X), k=k, axis=1)
    assert np.asarray(sp.c).tobytes() == np.asarray(ref.c).tobytes()
    assert len(sd.smooth.generate_knots(A(s.x), A(s.y), k=k)) == 1


def test_smoothing_condition_and_limits():
    s = S.scatter("random/200")
    for sv in (2.0, 10.0, 18.0, 40.0):
        sp = sd.smooth.make_splrep(A(s.x), A(s.y), s=sv)
        fp = fp_of(sp, s.x, s.y)
        assert abs(fp - sv) <= 0.001 * sv, sv
    big = sd.smooth.make_splrep(A(s.x), A(s.y), s=1e6)  # the least-squares cubic is within s
    assert np.asarray(big.t).size == 8
    coef = np.polyfit(s.x, s.y, 3)
    assert np.abs(np.asarray(big(A(s.x))) - np.polyval(coef, s.x)).max() <= 1e-10
    w = 1.0 / (0.2 + np.random.default_rng(0).random(s.n))
    sw = sd.smooth.make_splrep(A(s.x), A(s.y), w=A(w), s=float(s.n))
    assert abs(fp_of(sw, s.x, s.y, w) - s.n) <= 0.001 * s.n


def test_given_knots_and_nest():
    s = S.scatter("random/200")
    t = np.r_[[s.x[0]] * 4, s.x[20:-20:20], [s.x[-1]] * 4]
    sp = sd.smooth.make_splrep(A(s.x), A(s.y), s=25.0, t=A(t))
    assert np.asarray(sp.t).tobytes() == t.tobytes()
    assert abs(fp_of(sp, s.x, s.y) - 25.0) <= 0.025
    with pytest.raises(ValueError, match="cannot be reached"):
        sd.smooth.make_splrep(A(s.x), A(s.y), s=1.0, nest=12)
    with pytest.raises(ValueError, match="Schoenberg-Whitney"):
        sd.smooth.make_splrep(A(s.x), A(s.y), s=25.0, t=A(np.r_[[s.x[0]] * 4, np.linspace(s.x[5], s.x[6], 8)[1:-1],
                                                                 [s.x[-1]] * 4]))  # fmt: skip
    with pytest.raises(ValueError, match="'t' or 'nest'"):
        sd.smooth.make_splrep(A(s.x), A(s.y), s=25.0, t=A(t), nest=20)


def test_errors():
    x, y = A(np.arange(10.0)), A(np.sin(np.arange(10.0)))
    with pytest.raises(ValueError, match="interpolation"):
        sd.smooth.make_splrep(x, y, w=A(np.ones(10)))
    with pytest.raises(ValueError, match="nondecreasing"):
        sd.smooth.make_splrep(A(np.arange(10.0)[::-1].copy()), y, s=1.0)
    with pytest.raises(ValueError, match="'k' must be in"):
        sd.smooth.make_splrep(x, y, s=1.0, k=6)
    with pytest.raises(ValueError, match="'s' must be"):
        sd.smooth.make_splrep(x, y, s=-1.0)
    with pytest.raises(ValueError, match="first and last points"):
        sd.smooth.make_splrep(x, y, s=1.0, bc_type="periodic")
    with pytest.raises(ValueError, match="'bc_type'"):
        sd.smooth.make_splrep(x, y, s=1.0, bc_type="natural")
    with pytest.raises(ValueError, match="NaN"):
        sd.smooth.make_splrep(A([0.0, 1.0, 2.0, 3.0, math.nan]), A(np.ones(5)), s=1.0)
