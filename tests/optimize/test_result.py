"""The result object of sd.optimize (docs/api/optimize.md §1.3): read-only fields (regression
G98), array fields as new copies (G113), absent fields as AttributeError, a short repr."""

import pytest

import stardust as sd
from stardust.optimize._result import FIELDS

R = sd.optimize.OptimizeResult


def one():
    return R(
        x=sd.asarray([1.0, 2.0]),
        fun=0.5,
        jac=sd.asarray([1e-9, -2e-9]),
        converged=True,
        status=0,
        message="Optimization terminated successfully.",
        n_iter=3,
        nfev=5,
        njev=4,
        method="bfgs",
    )


def test_exported_from_sd_optimize():
    assert sd.optimize.OptimizeResult is R
    assert "OptimizeResult" in sd.optimize.__all__


def test_fields_and_python_types():
    r = one()
    assert r.fun == 0.5 and r.converged is True and r.status == 0 and r.n_iter == 3 and r.nfev == 5
    assert r.njev == 4 and r.method == "bfgs" and r.message.startswith("Optimization")
    assert isinstance(r.x, sd.ndarray) and r.x.tolist() == [1.0, 2.0]


def test_names_are_the_other_areas():
    # O4: converged and n_iter, not scipy's success and nit (no aliases)
    r = one()
    for name in ("success", "nit", "root", "iterations", "function_calls", "flag", "f_x"):
        assert not hasattr(r, name)
        with pytest.raises(AttributeError):
            getattr(r, name)
    assert "success" not in FIELDS and "nit" not in FIELDS


def test_fields_are_read_only():
    r = one()
    for name in ("x", "fun", "converged", "n_iter", "nfev", "hess_inv"):
        with pytest.raises(AttributeError, match=f"property '{name}' of 'OptimizeResult' object has no setter"):
            setattr(r, name, 1)
    with pytest.raises(AttributeError, match="no attribute 'foo'"):
        r.foo = 1


def test_array_fields_are_read_only_views():
    # D-4 (API.md §2): the same read-only array at each access, no copy; .copy() to change one
    r = one()
    x = r.x
    assert r.x is x
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        x[0] = 99.0
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        x[0:1][0] = 99.0  # a view of it is read-only too
    y = x.copy()
    y[0] = 99.0
    assert r.x.tolist() == [1.0, 2.0]
    # the arrays given to the public constructor are copied once: the caller's stays writable
    src = sd.asarray([3.0, 4.0])
    r2 = R(x=src, converged=True)
    src[0] = -1.0
    assert r2.x.tolist() == [3.0, 4.0]


def test_solver_results_are_read_only():
    r = sd.optimize.minimize(lambda x: float(x[0] ** 2 + x[1] ** 2), [1.0, 2.0])
    for name in ("x", "jac", "hess_inv"):
        a = getattr(r, name)
        assert getattr(r, name) is a
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 1.0


def test_tuple_of_arrays_fields_are_read_only():
    lo, hi = sd.asarray([0.0, 1.0]), sd.asarray([2.0, 3.0])
    r = R(batch=(2,), x=sd.asarray([1.0, 2.0]), bracket=(lo, hi), converged=sd.asarray([True, True]))
    b = r.bracket
    assert isinstance(b, tuple) and len(b) == 2
    with pytest.raises(ValueError, match="read-only"):
        b[0][0] = 7.0
    lo[0] = 7.0  # the caller's array: copied at construction
    assert r.bracket[0].tolist() == [0.0, 1.0]


def test_absent_fields():
    r = one()
    assert not hasattr(r, "hess_inv")
    with pytest.raises(AttributeError, match="has no attribute 'hess_inv' \\(its method does not give it\\)"):
        r.hess_inv  # noqa: B018


def test_unknown_field_at_construction():
    with pytest.raises(TypeError, match="OptimizeResult\\(\\): unexpected keyword argument 'success'"):
        R(x=sd.zeros(1), success=True)


def test_repr():
    assert repr(one()) == "OptimizeResult(method='bfgs', converged=True, fun=0.5, n_iter=3, nfev=5, x=[1.0 2.0])"
    lsq = R(x=sd.asarray([1.0]), cost=1.25e-17, fun=sd.asarray([1e-9, 2e-9]), converged=False, nfev=10)
    assert repr(lsq) == "OptimizeResult(converged=False, cost=1.25e-17, nfev=10, x=[1.0])"
    nan = R(fun=float("nan"), converged=False)
    assert repr(nan) == "OptimizeResult(converged=False, fun=nan)"
    batch = R(batch=(3,), x=sd.zeros(3), converged=sd.asarray([True, False, True]))
    assert repr(batch) == "OptimizeResult(batch shape (3,), 2 of 3 converged)"
    assert repr(R(batch=(2, 2), x=sd.zeros((2, 2)))) == "OptimizeResult(batch shape (2, 2))"


def test_slots_no_dict():
    r = one()
    assert not hasattr(r, "__dict__")
