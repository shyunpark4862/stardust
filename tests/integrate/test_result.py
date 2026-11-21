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
    # the class's name (R248; the rest is Python's sentence, _calls.keyword_init)
    with pytest.raises(TypeError, match=r"^IntegrateResult\(\): takes "):
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


def test_public_constructor_copies_numpy_arrays_and_lists():
    # R249 (C121): a numpy array or a list of numbers becomes an sd.ndarray of the result's own,
    # read-only, shown by repr; the three areas' classes alike
    import numpy as np

    for cls, name in ((R, "integral"), (sd.optimize.OptimizeResult, "x"), (sd.differentiate.DifferentiateResult, "df")):
        a = np.array([1.0, 2.0])
        w = cls(**{name: a})
        v = getattr(w, name)
        assert isinstance(v, sd.ndarray) and v is not a and v.tolist() == [1.0, 2.0]
        a[0] = 9.0
        assert getattr(w, name).tolist() == [1.0, 2.0]
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            v[0] = 3.0
        assert repr(w) == f"{cls.__name__}({name}=[1, 2])"
        assert repr(cls(**{name: [1.0, 2.0]})) == f"{cls.__name__}({name}=[1, 2])"
    # a 0-d numpy array is its number; tuples hold their arrays copied too
    assert R(integral=np.array(2.5)).integral == 2.5
    b = sd.asarray([1.0, 2.0])
    r = sd.optimize.OptimizeResult(bracket=(b, [3.0, 4.0]))
    assert r.bracket[0] is not b and r.bracket[1].tolist() == [3.0, 4.0]
    c = R(batch=(2,), converged=np.array([True, False]), status=[0, 1])
    assert c.converged.dtype == sd.bool and c.status.tolist() == [0, 1]


def test_public_constructor_checks_the_kinds():
    # R249 (C121): the shared fields' kinds, in the class's name
    for kw, text in (({"integral": "abc"}, "'integral' must be a number or an array of numbers, not str"),
                     ({"integral": [1, "a"]}, "'integral' must be a number or an array of numbers, not list"),
                     ({"error": True}, "'error' must be a number or an array of numbers, not bool"),
                     ({"converged": 3}, "'converged' must be a bool or an array of bools, not int"),
                     ({"converged": sd.zeros(2)}, "'converged' must be a bool or an array of bools, not ndarray"),
                     ({"status": 1.5}, "'status' must be an int or an array of ints, not float"),
                     ({"nfev": True}, "'nfev' must be an int or an array of ints, not bool"),
                     ({"n_intervals": 2.0}, "'n_intervals' must be an int or an array of ints, not float"),
                     ({"message": 3}, "'message' must be a str, not int")):  # fmt: skip
        with pytest.raises(TypeError, match=rf"^IntegrateResult\(\): {text}$"):
            R(**kw)
    with pytest.raises(TypeError, match=r"^OptimizeResult\(\): 'method' must be a str, not int$"):
        sd.optimize.OptimizeResult(method=1)
    # the class's name for a positional argument (R248), the three alike
    for cls in (R, sd.optimize.OptimizeResult, sd.differentiate.DifferentiateResult):
        with pytest.raises(TypeError, match=rf"^{cls.__name__}\(\): takes "):
            cls(1.0)
    # what the functions give is taken as it is (no copy, no checks)
    m = R._make(integral=sd.asarray([1.0]), message=3)
    assert m.message == 3
