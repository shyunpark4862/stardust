"""The result object of sd.integrate (docs/api/integrate.md §1.3; C9, C10, C13): the rules of
sd.optimize's OptimizeResult on the base the three areas share, and the quad family's unpacking
into (integral, error)."""

import pytest

import stardust as sd
from stardust._results import _FieldResult
from stardust.integrate._result import FIELDS

R = sd.integrate.IntegrateResult


def quad_like():
    return R._make(integral=0.746824132812427, error=8.29e-15, converged=True, status=0, message="ok", nfev=21,
                   unpack=True)  # fmt: skip


def test_exported_and_shares_the_base():
    assert sd.integrate.IntegrateResult is R
    assert "IntegrateResult" in sd.integrate.__all__  # with the functions of the milestones done (test_api_surface)
    assert issubclass(R, _FieldResult) and issubclass(sd.optimize.OptimizeResult, _FieldResult)
    assert issubclass(sd.differentiate.DifferentiateResult, _FieldResult)


def test_field_names_are_the_other_areas():
    # C13: integral (sum for a series), error, nfev everywhere; C4: converged, not success
    assert FIELDS[:3] == ("integral", "sum", "error")
    r = quad_like()
    for name in ("success", "estimate", "neval", "abserr", "y", "nit", "maxlevel"):
        assert not hasattr(r, name)
    assert "estimate" not in FIELDS and "neval" not in FIELDS and "success" not in FIELDS


def test_fields_and_python_types():
    r = quad_like()
    assert r.integral == 0.746824132812427 and r.error == 8.29e-15 and r.converged is True
    assert r.status == 0 and r.nfev == 21 and r.message == "ok"


def test_quad_family_results_unpack():
    integral, error = quad_like()
    assert (integral, error) == (0.746824132812427, 8.29e-15)
    r = quad_like()
    assert r[0] == integral and r[1] == error and r[-1] == error and len(r) == 2
    assert list(r) == [integral, error]


def test_other_results_do_not_unpack():
    for r in (R(integral=1.0, error=2.0), R._make(integral=1.0, error=2.0), R._make(sum=1.0, error=2.0, unpack=True)):
        with pytest.raises(TypeError, match="does not unpack"):
            tuple(r)
        with pytest.raises(TypeError, match="does not unpack"):
            r[0]  # noqa: B018
        with pytest.raises(TypeError, match="does not unpack"):
            len(r)
        assert bool(r) is True  # a value, true even though len() raises


def test_fields_are_read_only():
    r = quad_like()
    for name in ("integral", "error", "converged", "nfev"):
        with pytest.raises(AttributeError, match=f"property '{name}' of 'IntegrateResult' object has no setter"):
            setattr(r, name, 1)
    with pytest.raises(AttributeError):
        r.foo = 1
    assert not hasattr(r, "__dict__")


def test_array_fields_are_read_only_views():
    src = sd.asarray([1.0, 2.0])
    r = R(batch=(2,), integral=src, error=sd.zeros(2), converged=sd.asarray([True, False]))
    src[0] = 9.0  # the public constructor copies once
    assert r.integral.tolist() == [1.0, 2.0]
    x = r.integral
    assert r.integral is x
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        x[0] = 3.0
    m = R._make(batch=(2,), integral=sd.asarray([1.0, 2.0]))  # the functions' arrays: taken without a copy
    assert m.integral.tolist() == [1.0, 2.0]


def test_absent_fields():
    r = quad_like()
    with pytest.raises(AttributeError, match="has no attribute 'sum' \\(the function does not give it\\)"):
        r.sum  # noqa: B018
    assert not hasattr(r, "max_level")


def test_unknown_field_and_positional():
    with pytest.raises(TypeError, match="IntegrateResult\\(\\): unexpected keyword argument 'estimate'"):
        R(estimate=1.0)
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        R(1.0)


def test_repr():
    assert repr(quad_like()) == "IntegrateResult(integral=0.746824, error=8.29e-15, converged=True, nfev=21)"
    assert repr(R(sum=1.6449340668482264, error=7.4e-9, converged=True, nfev=8561)) == (
        "IntegrateResult(sum=1.64493, error=7.4e-09, converged=True, nfev=8561)"
    )
    assert repr(R(integral=float("nan"), converged=False)) == "IntegrateResult(integral=nan, converged=False)"
    b = R(batch=(1000,), integral=sd.zeros(1000), converged=sd.arange(1000) < 998)
    assert repr(b) == "IntegrateResult(batch shape (1000,), 998 of 1000 converged)"
    assert repr(R(batch=(2, 2), integral=sd.zeros((2, 2)))) == "IntegrateResult(batch shape (2, 2))"
    v = R(integral=sd.asarray([1.0, 2.0, 3.0]), error=1e-12)  # quad_vec: an array value
    assert repr(v) == "IntegrateResult(integral=[1, 2, 3], error=1e-12)"
