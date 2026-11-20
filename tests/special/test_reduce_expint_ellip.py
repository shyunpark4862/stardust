"""logsumexp softmax log_softmax expi expn ellipk ellipe (docs/api/special.md §6, §7, §10).

Accuracy of the elementwise four is tests/special/test_reference.py's; the reductions are
held here against tests/special/ref/reduce_cases.npz (the user's contract: logsumexp and
log_softmax |error| <= 1e-14 max(1, |value|), softmax relative 1e-14 per element). Also the
special values against scipy except where docs/design/special.md §2.4 says otherwise, and
the reductions' axis rules."""

import itertools

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd
from tests._bits import bits_equal_any_nan

from . import refdata

S = sd.special
SUBNORMAL_ABS = 45 * 2.0**-1074


# ---- special values -------------------------------------------------------------


def test_unary_edges_match_scipy():
    for name in ["expi", "ellipk", "ellipe"]:
        for x in [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1.0000001, -1e300, 5e-324, -5e-324]:
            got, want = getattr(S, name)(x), float(getattr(sp, name)(x))
            if want == 0 or not np.isfinite(want) or not np.isfinite(got):
                assert bits_equal_any_nan(got, want), (name, x, got, want)


def test_expn_edges_match_scipy():
    for n, x in itertools.product([0.0, 1.0, 2.0, 5.0, -1.0, np.nan], [0.0, 0.5, 2.0, np.inf, -1.0, np.nan]):
        got, want = S.expn(n, x), float(sp.expn(n, x))
        if want == 0 or not np.isfinite(want) or not np.isfinite(got):
            assert bits_equal_any_nan(got, want), (n, x, got, want)


def test_expn_order_is_an_integer():
    # Q17: int64 or float64; a non-integer, negative or infinite n is NaN (scipy truncates 1.5
    # to 1 and gives 1.7e-10 for n = inf)
    assert np.isnan(S.expn(1.5, 1.0)) and np.isnan(S.expn(-1, 1.0)) and np.isnan(S.expn(np.inf, 1.0))
    assert S.expn(3, 1.0) == S.expn(3.0, 1.0)
    got = np.asarray(S.expn(sd.asarray(np.array([1, 2, 3], dtype=np.int64)), 0.5))
    assert np.array_equal(got, [S.expn(1.0, 0.5), S.expn(2.0, 0.5), S.expn(3.0, 0.5)])


def test_scipy_errors_get_true_values():
    # true values where scipy overflows (docs/design/special.md §2.4)
    assert abs(S.expi(710.0) / 3.1509156882062014e305 - 1) < 1e-14  # scipy inf from 709.78
    assert np.isfinite(S.expi(716.0)) and S.expi(716.5) == np.inf
    assert abs(S.expn(1, 712.0) - 8.49650284055e-313) <= SUBNORMAL_ABS  # scipy 0
    assert abs(S.expn(5, 725.0) - 1.87578e-318) <= SUBNORMAL_ABS
    assert abs(S.expn(2**31, 1.0) / 1.7130721414994557e-10 - 1) < 1e-14  # scipy NaN (n past int32)


def test_closed_forms():
    for x in [0.01, 0.5, 3.0, 50.0, 700.0]:
        assert abs(S.expn(0, x) / (np.exp(-x) / x) - 1) <= 4.5e-16
    assert S.ellipk(0.0) == np.pi / 2 and S.ellipe(0.0) == np.pi / 2 and S.ellipe(1.0) == 1.0
    assert S.ellipk(1.0) == np.inf and np.isnan(S.ellipk(1.5)) and np.isnan(S.ellipe(1.5))
    assert S.ellipk(-np.inf) == 0.0 and S.ellipe(-np.inf) == np.inf


# ---- reductions ---------------------------------------------------------------


def cases():
    return refdata.reduce_cases()


def test_reductions_accuracy():
    worst = {"logsumexp": 0.0, "softmax": 0.0, "log_softmax": 0.0}
    for x, (lh, ll), (sh, sl), (gh, gl) in cases():
        a = sd.asarray(x)
        e = refdata.mixed_error(np.array([S.logsumexp(a)]), np.array([lh]), np.array([ll]))
        worst["logsumexp"] = max(worst["logsumexp"], float(e.max()))
        got = np.asarray(S.softmax(a))
        sub = np.abs(sh) < 2.0**-1022
        rel = refdata.rel_error(got, sh, sl)
        absolute = np.abs((got - sh) - sl) / SUBNORMAL_ABS * 1e-14
        worst["softmax"] = max(worst["softmax"], float(np.where(sub, absolute, rel).max()))
        worst["log_softmax"] = max(
            worst["log_softmax"], float(refdata.mixed_error(np.asarray(S.log_softmax(a)), gh, gl).max())
        )
    for name, w in worst.items():
        assert w <= 1e-14, (name, w)


def test_reduction_axes_and_shapes():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 3, (4, 5, 6))
    a = sd.asarray(x)
    for axis in [None, 0, 1, 2, -1, (0, 2), [1, 2]]:
        np_axis = tuple(axis) if isinstance(axis, list) else axis
        for keep in (False, True):
            got = S.logsumexp(a, axis=axis, keepdims=keep)
            want = sp.logsumexp(x, axis=np_axis, keepdims=keep)
            assert np.allclose(np.asarray(got), want, rtol=1e-14, atol=0)
            assert np.shape(np.asarray(got)) == np.shape(want)
        assert np.allclose(np.asarray(S.softmax(a, axis=axis)), sp.softmax(x, axis=np_axis), rtol=1e-13, atol=0)
        assert np.allclose(
            np.asarray(S.log_softmax(a, axis=axis)), sp.log_softmax(x, axis=np_axis), rtol=1e-13, atol=1e-15
        )
    assert isinstance(S.logsumexp(a), float)  # a 0-d result is a Python float


@pytest.mark.filterwarnings("ignore:invalid value")  # scipy's own inf - inf
def test_reduction_infinities_match_scipy():
    for v in [
        [-np.inf, -np.inf],
        [np.inf, 1.0],
        [np.inf, -np.inf],
        [np.nan, 1.0],
        [np.inf, np.inf],
        [1e308, 1e308],
        [-1e308, 0.0],
        [5.0],
    ]:
        x = np.array(v)
        a = sd.asarray(x)
        assert bits_equal_any_nan(S.logsumexp(a), sp.logsumexp(x)), v
        for name in ["softmax", "log_softmax"]:
            got, want = np.asarray(getattr(S, name)(a)), getattr(sp, name)(x)
            assert all(bits_equal_any_nan(g, w) or abs(g - w) <= 1e-15 * abs(w) for g, w in zip(got, want)), (
                name,
                v,
                got,
                want,
            )


def test_reduction_empty():
    e = sd.asarray(np.zeros((0, 3)))
    assert S.logsumexp(sd.asarray(np.array([]))) == -np.inf
    assert np.array_equal(np.asarray(S.logsumexp(e, axis=0)), [-np.inf] * 3)
    assert np.asarray(S.logsumexp(e, axis=1)).shape == (0,)
    assert np.asarray(S.softmax(e)).shape == (0, 3)


def test_reduction_integer_input():
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert np.allclose(np.asarray(S.logsumexp(sd.asarray(x), axis=1)), sp.logsumexp(x, axis=1), rtol=1e-15)


def test_reduction_errors_name_the_function():
    with pytest.raises(TypeError, match=r"^logsumexp\(\)"):
        S.logsumexp([1.0, 2.0])
    with pytest.raises((ValueError, IndexError), match=r"^softmax\(\)"):
        S.softmax(sd.asarray(np.ones(3)), axis=2)


def test_reduction_call_shape():
    # x is positional-only, axis (and logsumexp's keepdims) keyword-only; softmax and
    # log_softmax keep x's shape and have no keepdims (docs/api/special.md §6). A wrong call
    # shape is a TypeError that starts with the function's name (API.md §2.1).
    x = sd.asarray(np.arange(6.0).reshape(2, 3))
    for name in ("logsumexp", "softmax", "log_softmax"):
        f = getattr(S, name)
        bad = [lambda: f(x, 0), lambda: f(x, foo=1), lambda: f(x=x), lambda: f(), lambda: f(x, axis=0, out=None)]
        if name != "logsumexp":
            bad.append(lambda: f(x, keepdims=True))
        for call in bad:
            with pytest.raises(TypeError, match=rf"^{name}\(\)"):
                call()
        assert np.asarray(f(x, axis=0)).shape == ((3,) if name == "logsumexp" else (2, 3))
    assert np.asarray(S.logsumexp(x, axis=0, keepdims=True)).shape == (1, 3)
