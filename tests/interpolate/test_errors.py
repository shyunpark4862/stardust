"""The errors of sd.interp (docs/api/array.md §8.5; API.md §2, §2.1): kind and message, named by the function."""

import math

import numpy as np
import pytest

import stardust as sd

A = sd.asarray
NOT_X = "interp(): 'x' must be an sd.ndarray or a bool, int or float, not "
XP, FP = A([0.0, 1.0]), A([0.0, 1.0])

CASES = [
    # (call, exception, message)
    (lambda: sd.interp(A([0.5]), A([0.0, 2.0, 1.0]), A([0.0, 1.0, 2.0])), ValueError,
     "interp(): 'xp' must be nondecreasing"),
    (lambda: sd.interp(A([0.5]), A([0.0, math.nan]), FP), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A([0.0, math.inf]), FP), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A([math.nan, 0.0]), FP, period=1.0), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A(np.zeros(0)), A(np.zeros(0))), ValueError, "interp(): 'xp' must not be empty"),
    (lambda: sd.interp(A([0.5]), XP, A([1.0])), ValueError, "interp(): 'fp' must be as long as 'xp'"),
    (lambda: sd.interp(A([0.5]), A([[0.0, 1.0]]), A([[0.0, 1.0]])), ValueError, "interp(): 'xp' must be 1-D"),
    (lambda: sd.interp(A([0.5]), XP, A([[0.0], [1.0]])), ValueError, "interp(): 'fp' must be 1-D"),
    (lambda: sd.interp(A([0.5j]), XP, FP), TypeError, "interp(): 'x' must be real, not complex128"),
    (lambda: sd.interp(0.5j, XP, FP), TypeError, "interp(): 'x' must be real, not complex"),
    (lambda: sd.interp(0.5, A([0j, 1.0]), FP), TypeError, "interp(): 'xp' must be real, not complex128"),
    (lambda: sd.interp([0.5], XP, FP), TypeError, NOT_X + "list"),
    (lambda: sd.interp(np.array([0.5]), XP, FP), TypeError, NOT_X + "numpy.ndarray"),
    (lambda: sd.interp(None, XP, FP), TypeError, NOT_X + "NoneType"),
    (lambda: sd.interp(0.5, [0.0, 1.0], FP), TypeError, "interp(): 'xp' must be an sd.ndarray, not list"),
    (lambda: sd.interp(0.5, XP, (0.0, 1.0)), TypeError, "interp(): 'fp' must be an sd.ndarray, not tuple"),
    (lambda: sd.interp(0.5, XP, FP, period=0), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=-0.0), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=math.nan), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=math.inf), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period="1"), TypeError, "interp(): 'period' must be an int or a float, not str"),
    (lambda: sd.interp(0.5, XP, FP, period=True), TypeError, "interp(): 'period' must be an int or a float, not bool"),
    (lambda: sd.interp(0.5, XP, FP, left=1j), TypeError, "interp(): 'left' must be an int or a float, not complex"),
    (lambda: sd.interp(0.5, XP, FP, right="a"), TypeError, "interp(): 'right' must be an int or a float, not str"),
    (lambda: sd.interp(0.5, XP, A([0j, 1.0]), left="a"), TypeError, "interp(): 'left' must be a number, not str"),
    (lambda: sd.interp(0.5, XP, FP, period=10**400), OverflowError, "interp(): 'period' is out of range"),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES)))
def test_error(k):
    call, exc, msg = CASES[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_call_shape():
    # numpy takes left and right by position; sd only by keyword (docs/api/array.md §8.5)
    with pytest.raises(TypeError, match=r"^interp\(\): incompatible function arguments"):
        sd.interp(A([0.5]), XP, FP, 0.0, 1.0)
    with pytest.raises(TypeError, match=r"^interp\(\): incompatible function arguments"):
        sd.interp(x=A([0.5]), xp=XP, fp=FP)


def test_unsorted_xp_with_a_period_is_sorted_not_refused():
    assert sd.interp(0.5, A([1.0, 0.0]), A([1.0, 0.0]), period=2.0) == 0.5


# ---- the cubic family and PPoly (docs/api/interpolate.md §1, §3; milestone IP2)

IP = sd.interpolate
X, Y = [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 2.0]
P = IP.PPoly([[1.0, 2.0, 3.0]], X)
SP = IP.CubicSpline(X, Y)

CASES_IP2 = [
    (lambda: IP.CubicSpline([0.0, 2.0, 1.0], Y[:3]), ValueError, "CubicSpline(): 'x' must be strictly increasing"),
    (lambda: IP.CubicSpline([0.0, 1.0, 1.0], Y[:3]), ValueError, "CubicSpline(): 'x' must be strictly increasing"),
    (lambda: IP.CubicSpline([0.0, math.nan, 1.0], Y[:3]), ValueError, "CubicSpline(): 'x' must be finite"),
    (lambda: IP.CubicSpline([0.0], [1.0]), ValueError, "CubicSpline(): 'x' must have at least 2 points, not 1"),
    (lambda: IP.CubicSpline([[0.0, 1.0]], [1.0, 2.0]), ValueError, "CubicSpline(): 'x' must be 1-D, not 2-D"),
    (lambda: IP.CubicSpline([0.0, 1j], [1.0, 2.0]), TypeError, "CubicSpline(): 'x' must be real, not complex128"),
    (lambda: IP.CubicSpline(X, [0.0, 1.0, math.inf, 2.0]), ValueError, "CubicSpline(): 'y' must be finite"),
    (lambda: IP.CubicSpline(X, Y[:3]), ValueError,
     "CubicSpline(): 'y' must have as many points as 'x' (4) along axis 0, not 3"),
    (lambda: IP.CubicSpline(X, Y, axis=1), ValueError,
     "CubicSpline(): axis 1 is out of bounds for an array of dimension 1"),
    (lambda: IP.CubicSpline(X, Y, axis=0.0), TypeError, "CubicSpline(): 'axis' must be an int, not float"),
    (lambda: IP.CubicSpline(X, Y, bc_type="foo"), ValueError,
     "CubicSpline(): 'bc_type' must be 'not-a-knot', 'clamped', 'natural', 'periodic' or (order, value) pairs, "
     "not 'foo'"),
    (lambda: IP.CubicSpline(X, Y, bc_type="periodic"), ValueError,
     "CubicSpline(): 'y' must have equal first and last values along axis 0 for bc_type='periodic' "
     "(within rtol = atol = 1e-15)"),
    (lambda: IP.CubicSpline(X, Y, bc_type=("periodic", "natural")), ValueError,
     "CubicSpline(): bc_type='periodic' applies to both ends at once"),
    (lambda: IP.CubicSpline(X, Y, bc_type=((3, 0.0), "natural")), ValueError,
     "CubicSpline(): the derivative order of a boundary condition must be 1 or 2, not 3"),
    (lambda: IP.CubicSpline(X, Y, bc_type=((1, [0.0, 1.0]), "natural")), ValueError,
     "CubicSpline(): a boundary value must have the shape (), not (2,)"),
    (lambda: IP.CubicSpline(X, Y, bc_type=("natural",) * 3), ValueError,
     "CubicSpline(): 'bc_type' must be one condition or a pair of them, not 3"),
    (lambda: IP.CubicSpline(X, Y, bc_type=None), TypeError,
     "CubicSpline(): 'bc_type' must be a str or a pair of conditions, not NoneType"),
    (lambda: IP.CubicSpline(X, Y, extrapolate="foo"), ValueError,
     "CubicSpline(): 'extrapolate' must be True, False, 'periodic' or None, not 'foo'"),
    (lambda: IP.CubicSpline(X, Y, extrapolate=1), TypeError,
     "CubicSpline(): 'extrapolate' must be True, False, 'periodic' or None, not int"),
    (lambda: IP.PchipInterpolator(X, [0, 1j, 0, 1]), ValueError,
     "PchipInterpolator(): 'y' must be real, not complex128"),
    (lambda: IP.Akima1DInterpolator(X, [0, 1j, 0, 1]), ValueError,
     "Akima1DInterpolator(): 'y' must be real, not complex128"),
    (lambda: IP.Akima1DInterpolator(X, Y, method="foo"), ValueError,
     "Akima1DInterpolator(): 'method' must be 'akima' or 'makima', not 'foo'"),
    (lambda: IP.CubicHermiteSpline(X, Y, Y[:3]), ValueError,
     "CubicHermiteSpline(): 'dydx' must have the shape of 'y' (4,), not (3,)"),
    (lambda: IP.CubicHermiteSpline(X, Y, [0.0, math.nan, 0.0, 0.0]), ValueError,
     "CubicHermiteSpline(): 'dydx' must be finite"),
    (lambda: IP.pchip_interpolate(X, Y, 0.5, der=-1), ValueError,
     "pchip_interpolate(): 'der' must be 0 or more, not -1"),
    (lambda: IP.pchip_interpolate([0.0, 0.0], [1.0, 2.0], 0.5), ValueError,
     "pchip_interpolate(): 'x' must be strictly increasing"),
    (lambda: IP.PPoly([1.0, 2.0], X[:2]), ValueError, "PPoly(): 'c' must have at least 2 dimensions, not 1"),
    (lambda: IP.PPoly([[1.0, 2.0]], X), ValueError,
     "PPoly(): 'c' must have one piece per interval of 'x' (3) along axis 1, not 2"),
    (lambda: IP.PPoly([[1.0, 2.0]], [0.0, 1.0, 1.0]), ValueError,
     "PPoly(): 'x' must be strictly increasing or strictly decreasing"),
    (lambda: IP.PPoly([[1.0, 2.0]], [0.0, 1.0, math.inf]), ValueError, "PPoly(): 'x' must be finite"),
    (lambda: IP.PPoly(np.zeros((0, 3)), X), ValueError, "PPoly(): 'c' must have at least one coefficient per piece"),
    (lambda: IP.PPoly([[1.0, 2.0, 3.0]], X, axis=1), ValueError,
     "PPoly(): axis 1 is out of bounds for an array of dimension 1"),
    (lambda: P(0.5j), TypeError, "PPoly(): 'x' must be real, not complex"),
    (lambda: P([0.5j]), TypeError, "PPoly(): 'x' must be real, not complex128"),
    (lambda: P("a"), TypeError,
     "PPoly(): 'x' must be an sd.ndarray, a sequence of numbers, a numpy array or a frame Series, not str"),
    (lambda: P(["a"]), TypeError,
     "PPoly(): 'x' must be an sd.ndarray, a sequence of numbers, a numpy array or a frame Series, not list"),
    (lambda: P(0.5, nu=-1), ValueError, "PPoly(): 'nu' must be 0 or more, not -1"),
    (lambda: P(0.5, nu=1.0), TypeError, "PPoly(): 'nu' must be an int, not float"),
    (lambda: P(0.5, nu=True), TypeError, "PPoly(): 'nu' must be an int, not bool"),
    (lambda: SP(0.5, extrapolate="no"), ValueError,
     "CubicSpline(): 'extrapolate' must be True, False, 'periodic' or None, not 'no'"),
    (lambda: P.derivative(nu=0.5), TypeError, "derivative(): 'nu' must be an int, not float"),
    (lambda: P.antiderivative(nu="1"), TypeError, "antiderivative(): 'nu' must be an int, not str"),
    (lambda: P.integrate("0", 1.0), TypeError, "integrate(): 'a' must be an int or a float, not str"),
    (lambda: P.integrate(0.0, 1j), TypeError, "integrate(): 'b' must be an int or a float, not complex"),
    (lambda: P.integrate(0.0, 1.0, extrapolate=3), TypeError,
     "integrate(): 'extrapolate' must be True, False, 'periodic' or None, not int"),
    (lambda: P.solve(math.inf), ValueError, "solve(): 'y' must be finite"),
    (lambda: P.roots(discontinuity=1), TypeError, "roots(): 'discontinuity' must be True or False, not int"),
    (lambda: P.extend([[1.0]], [2.5]), ValueError,
     "extend(): 'x' must lie beyond either end of self.x, without touching it"),
    (lambda: P.extend([[1.0, 2.0]], [4.0]), ValueError,
     "extend(): 'x' must have one point per piece of 'c' (2), not 1"),
    (lambda: P.extend([[1.0, 2.0]], [5.0, 4.0]), ValueError, "extend(): 'x' must be strictly increasing, as self.x"),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES_IP2)))
def test_error_ip2(k):
    call, exc, msg = CASES_IP2[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_call_shape_ip2():
    with pytest.raises(TypeError, match=r"^CubicSpline\(\): takes 2 positional arguments but 3 were given"):
        IP.CubicSpline(X, Y, 0)
    with pytest.raises(TypeError, match=r"^CubicSpline\(\): unexpected keyword argument 'check_finite'"):
        IP.CubicSpline(X, Y, check_finite=False)
    with pytest.raises(TypeError, match=r"^pchip_interpolate\(\): unexpected keyword argument 'foo'"):
        IP.pchip_interpolate(X, Y, 0.5, foo=1)
    with pytest.raises(TypeError):
        IP.PPoly([[1.0]], [0.0, 1.0], True)


# ---- the B-splines (docs/api/interpolate.md §4; milestone IP3)

T8 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
B = IP.BSpline(T8, [1.0, 2.0, 0.0, 1.0], 3)
X6 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
Y6 = [0.0, 1.0, 0.0, 2.0, 1.0, 0.0]

CASES_IP3 = [
    (lambda: IP.BSpline(T8, [1.0, 2.0, 3.0], 3), ValueError,
     "BSpline(): 'c' must have at least n = len(t) - k - 1 = 4 coefficients along axis 0, not 3"),
    (lambda: IP.BSpline(T8, [1.0] * 4, -1), ValueError, "BSpline(): 'k' must be 0 or more, not -1"),
    (lambda: IP.BSpline(T8, [1.0] * 4, 3.0), TypeError, "BSpline(): 'k' must be an int, not float"),
    (lambda: IP.BSpline(T8[:5], [1.0], 3), ValueError,
     "BSpline(): 't' must have at least 2k + 2 = 8 knots for degree 3, not 5"),
    (lambda: IP.BSpline([0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0] * 4, 3), ValueError,
     "BSpline(): 't' must be nondecreasing"),
    (lambda: IP.BSpline([0.0] * 5 + [1.0] * 3, [1.0] * 4, 3), ValueError,
     "BSpline(): the base interval [t[k], t[n]] = [t[3], t[4]] must not be empty"),
    (lambda: IP.BSpline(T8[:-1] + [math.inf], [1.0] * 4, 3), ValueError, "BSpline(): 't' must be finite"),
    (lambda: IP.BSpline(T8, 1.0, 3), ValueError, "BSpline(): 'c' must have at least 1 dimension"),
    (lambda: B(0.5, nu=-1), ValueError, "BSpline(): 'nu' must be 0 or more, not -1"),
    (lambda: B(0.5j), TypeError, "BSpline(): 'x' must be real, not complex"),
    (lambda: B(3.5, extrapolate=2), TypeError,
     "BSpline(): 'extrapolate' must be True, False, 'periodic' or None, not int"),
    (lambda: B.derivative(nu=4), ValueError, "derivative(): 'nu' must be at most the degree k = 3, not 4"),
    (lambda: B.insert_knot(9.0), ValueError,
     "insert_knot(): 'x' must lie in the base interval [t[k], t[n]] = [3.0, 4.0], not 9.0"),
    (lambda: B.insert_knot(3.5, m=0), ValueError, "insert_knot(): 'm' must be positive, not 0"),
    (lambda: IP.BSpline.basis_element([0.0]), ValueError, "basis_element(): 't' must have at least 2 knots, not 1"),
    (lambda: IP.BSpline.design_matrix([0.0, 9.0], T8, 3), ValueError,
     "design_matrix(): 'x' must lie in the base interval [t[k], t[n]] = [3.0, 4.0] when extrapolate=False"),
    (lambda: IP.BSpline.design_matrix([math.nan], T8, 3), ValueError, "design_matrix(): 'x' must be finite"),
    (lambda: IP.BSpline.design_matrix([[3.5]], T8, 3), ValueError, "design_matrix(): 'x' must be 1-D, not 2-D"),
    (lambda: IP.BSpline.design_matrix([3.5], T8, 3, extrapolate=None), TypeError,
     "design_matrix(): 'extrapolate' must be True, False or 'periodic', not NoneType"),
    (lambda: IP.BSpline.from_power_basis(IP.PchipInterpolator(X, Y)), TypeError,
     "from_power_basis(): 'pp' must be a CubicSpline, not PchipInterpolator"),
    (lambda: IP.BSpline.from_power_basis(SP, bc_type="foo"), ValueError,
     "from_power_basis(): 'bc_type' must be 'not-a-knot', 'natural', 'clamped' or 'periodic', not 'foo'"),
    (lambda: IP.PPoly.from_spline((T8, [1.0] * 3, 3)), ValueError,
     "from_spline(): 'c' must have at least n = len(t) - k - 1 = 4 coefficients along axis 0, not 3"),
    (lambda: IP.PPoly.from_spline(5), TypeError,
     "from_spline(): 'tck' must be a BSpline or a (t, c, k) tuple, not int"),
    (lambda: IP.make_interp_spline(X6, Y6, k=-1), ValueError, "make_interp_spline(): 'k' must be 0 or more, not -1"),
    (lambda: IP.make_interp_spline(X6, Y6, k=5, bc_type="natural"), ValueError,
     "make_interp_spline(): the knots need 4 derivative conditions at the ends, not 1 + 1"),
    (lambda: IP.make_interp_spline(X6[:3], Y6[:3], k=3), ValueError,
     "make_interp_spline(): the knots need 1 derivative condition at the ends, not 0 + 0"),
    (lambda: IP.make_interp_spline(X6, Y6, k=1, bc_type="natural"), ValueError,
     "make_interp_spline(): k=1 takes no end conditions in 'bc_type' (unless 't' is given)"),
    (lambda: IP.make_interp_spline(X6, Y6, k=0, t=X6 + [5.0]), ValueError,
     "make_interp_spline(): k=0 takes neither 't' nor end conditions in 'bc_type'"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type="foo"), ValueError,
     "make_interp_spline(): 'bc_type' must be None, 'not-a-knot', 'natural', 'clamped', 'periodic' or a pair of "
     "end conditions, not 'foo'"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type=((1, 0.0), (1, 0.0))), ValueError,
     "make_interp_spline(): an end condition must be a list of (order, value) pairs"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type=([(4, 0.0)], [(1, 0.0)])), ValueError,
     "make_interp_spline(): a derivative order at x[0] must be in [0, k] = [0, 3], not 4"),
    (lambda: IP.make_interp_spline(X6, Y6[:5] + [3.0], bc_type="periodic"), ValueError,
     "make_interp_spline(): 'y' must have equal first and last values along axis 0 for bc_type='periodic' "
     "(within rtol = atol = 1e-15)"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type="periodic", t=[0.0] * 4 + [2.0, 3.0] + [5.0] * 4), ValueError,
     "make_interp_spline(): bc_type='periodic' makes its own knots: 't' must be None"),
    (lambda: IP.make_interp_spline(X6, Y6, t=[0.0] * 4 + [5.0] * 4), ValueError,
     "make_interp_spline(): 't' must have at least len(x) + k + 1 = 10 knots, not 8"),
    (lambda: IP.make_interp_spline(X6, Y6, t=[1.0] * 4 + [2.0, 3.0] + [5.0] * 4), ValueError,
     "make_interp_spline(): 'x' must lie within the knots: t[k] <= x[0] and x[-1] <= t[-k]"),
    (lambda: IP.make_interp_spline(X6, Y6[:5]), ValueError,
     "make_interp_spline(): 'y' must have as many points as 'x' (6) along axis 0, not 5"),
    (lambda: IP.make_interp_spline(X6, Y6[:5] + [math.nan]), ValueError, "make_interp_spline(): 'y' must be finite"),
    (lambda: IP.make_interp_spline([0.0, 2.0, 1.0], Y6[:3], k=1), ValueError,
     "make_interp_spline(): 'x' must be strictly increasing"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type=3), TypeError,
     "make_interp_spline(): 'bc_type' must be None, a str or a pair of end conditions, not int"),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES_IP3)))
def test_error_ip3(k):
    call, exc, msg = CASES_IP3[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_singular_collocation_is_linalg_error():
    with pytest.raises(sd.linalg.LinAlgError, match=r"^make_interp_spline\(\): the collocation matrix is singular"):
        IP.make_interp_spline(X6, Y6, t=[0.0] * 4 + [0.1, 0.2] + [5.0] * 4)  # Schoenberg-Whitney fails
    with pytest.raises(sd.linalg.LinAlgError, match="singular"):
        IP.make_interp_spline(X6, Y6, bc_type=([(0, 0.0)], [(1, 0.0)]))  # the value at x[0] twice
    with pytest.raises(TypeError, match=r"^make_interp_spline\(\): takes 2 positional arguments but 3 were given"):
        IP.make_interp_spline(X6, Y6, 3)
    with pytest.raises(TypeError, match=r"^BSpline\(\): unexpected keyword argument 'check_finite'"):
        IP.BSpline(T8, [1.0] * 4, 3, check_finite=True)
