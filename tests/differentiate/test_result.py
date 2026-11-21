"""The result object of sd.differentiate (docs/api/differentiate.md §1.3; C9, C13): the rules of
sd.optimize's OptimizeResult on the base the three areas share; it does not unpack."""

import pytest

import stardust as sd
from stardust._results import _FieldResult
from stardust.differentiate._result import FIELDS

R = sd.differentiate.DifferentiateResult


def one():
    return R(df=2.7182818284590127, error=4.5e-14, converged=True, status=0, n_iter=2, nfev=11, x=1.0)


def test_exported_and_shares_the_base():
    assert sd.differentiate.DifferentiateResult is R
    assert sd.differentiate.__all__ == ["DifferentiateResult"]
    assert issubclass(R, _FieldResult)


def test_field_names():
    # scipy's df and ddf; the other areas' converged, n_iter, nfev (C4), not success and nit
    assert FIELDS == ("df", "ddf", "error", "converged", "status", "n_iter", "nfev", "x")
    r = one()
    for name in ("success", "nit", "derivative", "jac", "hess"):
        assert not hasattr(r, name)


def test_fields_read_only_and_absent():
    r = one()
    assert r.df == 2.7182818284590127 and r.n_iter == 2 and r.nfev == 11 and r.x == 1.0
    with pytest.raises(AttributeError, match="property 'df' of 'DifferentiateResult' object has no setter"):
        r.df = 1.0
    with pytest.raises(AttributeError, match="has no attribute 'ddf' \\(the function does not give it\\)"):
        r.ddf  # noqa: B018
    with pytest.raises(TypeError, match="DifferentiateResult\\(\\): unexpected keyword argument 'nit'"):
        R(nit=2)
    with pytest.raises(TypeError):
        tuple(r)  # not iterable
    assert not hasattr(r, "__dict__")


def test_arrays_read_only():
    r = R(batch=(2,), df=sd.asarray([1.0, 2.0]), converged=sd.asarray([True, True]))
    with pytest.raises(ValueError, match="read-only"):
        r.df[0] = 0.0
    assert r.df is r.df


def test_repr():
    assert repr(one()) == "DifferentiateResult(df=2.71828, error=4.5e-14, converged=True, n_iter=2, nfev=11)"
    b = R(batch=(3,), df=sd.zeros(3), converged=sd.asarray([True, False, True]))
    assert repr(b) == "DifferentiateResult(batch shape (3,), 2 of 3 converged)"
    h = R(ddf=sd.eye(3), error=sd.zeros((3, 3)))
    assert repr(h) == "DifferentiateResult(ddf=<array of shape (3, 3)>, error=<array of shape (3, 3)>)"
